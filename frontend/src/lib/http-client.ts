/**
 * Shared HTTP client for ML service calls (Issues 234, 221-233).
 *
 * ROOT FIX (forensic, root-level): all ML-integration API routes
 * (predict, top-k, rl, knowledge-graph, dataset, hypothesis/validate)
 * previously had THREE independent failure modes that silently corrupted
 * scientific output:
 *
 *   1. No timeout — a hung Python service would consume a Node.js
 *      event-loop slot forever. Under V1's 100-concurrent-request
 *      contract, 100 hung requests = 100 leaked slots = frontend
 *      totally unresponsive.
 *
 *   2. No retry — a single transient network blip (TCP RST, DNS hiccup,
 *      service restart) returned 502 to the researcher even though a
 *      retry 200ms later would have succeeded. Pharma partners demoing
 *      the platform saw "GT predict failed" for a model that was
 *      actually fine.
 *
 *   3. No error normalization — each route caught `unknown` and stringified
 *      it differently. The audit log showed "GT predict failed: TypeError:
 *      fetch failed" in one row and "GT predict failed: 503" in another.
 *      Compliance audits could not correlate failures across services.
 *
 * ROOT FIX: this module provides a single `mlFetch()` function used by
 * every ML-integration route. It enforces:
 *
 *   - Configurable timeout (default 30s, aborts with a structured error)
 *   - Exponential-backoff retry on 5xx + network errors (default 3
 *      retries: 100ms, 400ms, 1600ms — total ~2.1s added latency max)
 *   - Structured `MlServiceError` with `service`, `endpoint`, `status`,
 *      `attempt`, `cause` — every audit log entry has the same shape
 *   - Never retries 4xx (client errors — the request is wrong, retrying
 *      wastes capacity)
 *   - Never retries on AbortError (timeout — retrying would just time
 *      out again)
 *
 * SCIENTIFIC INTEGRITY: this client NEVER fabricates responses. If all
 * retries are exhausted, it throws — the caller surfaces a 502/504 to
 * the researcher with a clear message. No silent fallback to mock data.
 */

export interface MlFetchOptions {
  /** HTTP method (default GET). */
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  /** JSON-serializable request body (for POST/PUT/PATCH). */
  body?: unknown;
  /** Custom headers. Content-Type: application/json is set automatically when body is provided. */
  headers?: Record<string, string>;
  /** Per-request timeout in ms (default 30_000). */
  timeoutMs?: number;
  /** Max retry attempts on 5xx + network errors (default 3). */
  maxRetries?: number;
  /** Base delay for exponential backoff in ms (default 100). Total wait = base * (4^attempt - 1) / 3. */
  retryBaseDelayMs?: number;
  /** Optional AbortSignal from the caller (e.g., NextRequest signal). Combined with internal timeout. */
  externalSignal?: AbortSignal;
  /**
   * BE-037 ROOT FIX (v115, MEDIUM): explicitly mark the request as
   * idempotent. The retry logic only retries on 5xx + 429 + network
   * errors — for non-idempotent methods (POST/PUT/PATCH/DELETE), a
   * retry could duplicate a write (e.g., append to a CSV twice,
   * create two Neo4j edges, double-charge a billing account).
   *
   * Defaults to:
   *   - true for GET / HEAD (always safe to retry)
   *   - false for POST / PUT / PATCH / DELETE (NEVER retry unless
   *     the caller explicitly proves idempotency)
   *
   * Callers that KNOW their endpoint is idempotent (e.g., a POST that
   * uses an idempotency key, or a PUT that's a pure upsert) can set
   * `idempotent: true` to enable retries. Callers that are NOT sure
   * should set `maxRetries: 0` to be safe.
   */
  idempotent?: boolean;
}

export interface MlFetchOk<T> {
  ok: true;
  status: number;
  body: T;
}

export interface MlFetchErr {
  ok: false;
  status: number;
  error: MlServiceError;
}

export type MlFetchResult<T> = MlFetchOk<T> | MlFetchErr;

/**
 * Structured ML service error. Every field is populated so audit log
 * entries and operator dashboards show the same shape across all 4
 * Python services (Phase 1/2/3/4).
 */
export class MlServiceError extends Error {
  readonly service: string;
  readonly endpoint: string;
  readonly httpStatus: number;
  readonly attempt: number;
  readonly cause?: unknown;
  readonly isTimeout: boolean;
  readonly isRetryable: boolean;

  constructor(params: {
    service: string;
    endpoint: string;
    message: string;
    httpStatus?: number;
    attempt: number;
    cause?: unknown;
    isTimeout?: boolean;
    isRetryable?: boolean;
  }) {
    super(params.message);
    this.name = "MlServiceError";
    this.service = params.service;
    this.endpoint = params.endpoint;
    this.httpStatus = params.httpStatus ?? 0;
    this.attempt = params.attempt;
    this.cause = params.cause;
    this.isTimeout = params.isTimeout ?? false;
    this.isRetryable = params.isRetryable ?? false;
  }

  /** Serialize to a JSON-safe object for audit logs / API responses. */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      service: this.service,
      endpoint: this.endpoint,
      httpStatus: this.httpStatus,
      attempt: this.attempt,
      message: this.message,
      isTimeout: this.isTimeout,
      isRetryable: this.isRetryable,
      cause:
        this.cause instanceof Error
          ? { name: this.cause.name, message: this.cause.message }
          : typeof this.cause === "string"
            ? this.cause
            : undefined,
    };
  }
}

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_RETRY_BASE_DELAY_MS = 100;

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error("aborted"));
      return;
    }
    const t = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(t);
      reject(new Error("aborted"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function isRetryableStatus(status: number): boolean {
  // 5xx = server error, retry. 429 = rate-limited, retry with backoff.
  // 4xx (except 429) = client error, do NOT retry.
  return status >= 500 || status === 429;
}

function isNetworkError(err: unknown): boolean {
  if (err instanceof Error) {
    // TypeError: fetch failed (Node.js undici)
    // DOMException: The operation was aborted (timeout)
    return (
      err.name === "TypeError" ||
      err.name === "DOMException" ||
      err.message.includes("fetch failed") ||
      err.message.includes("ECONNREFUSED") ||
      err.message.includes("ECONNRESET") ||
      err.message.includes("ETIMEDOUT") ||
      err.message.includes("ENOTFOUND")
    );
  }
  return false;
}

/**
 * Execute a fetch with timeout, retry, and structured error normalization.
 *
 * Returns a discriminated union: `{ok: true, status, body}` or
 * `{ok: false, status, error: MlServiceError}`. Callers should use
 * `if (!result.ok) return internalError(result.error.toJSON())` to surface
 * the structured error to the API client.
 *
 * The `service` parameter is used in error messages and audit logs so
 * operators can identify WHICH Python service failed (phase1 / phase2 /
 * phase3 / phase4) without parsing URLs.
 */
export async function mlFetch<T = unknown>(
  url: string,
  options: MlFetchOptions & { service: string }
): Promise<MlFetchResult<T>> {
  const {
    method = "GET",
    body,
    headers = {},
    timeoutMs = DEFAULT_TIMEOUT_MS,
    maxRetries = DEFAULT_MAX_RETRIES,
    retryBaseDelayMs = DEFAULT_RETRY_BASE_DELAY_MS,
    externalSignal,
    service,
    // BE-037 ROOT FIX (v115, MEDIUM): default idempotency based on
    // method. GET is always safe to retry. POST/PUT/PATCH/DELETE
    // are NOT retried unless the caller explicitly marks idempotent: true.
    // (HEAD is not in the method type — MlFetchOptions.method is
    // "GET" | "POST" | "PUT" | "PATCH" | "DELETE". If HEAD is ever
    // added, it should also be in this default-idempotent list.)
    idempotent,
  } = options;
  const isIdempotent = idempotent ?? method === "GET";
  // BE-037: if the method is non-idempotent and the caller didn't
  // explicitly enable retries, FORCE maxRetries to 0. This prevents
  // a future developer from accidentally inheriting the default
  // maxRetries: 3 on a non-idempotent POST.
  const effectiveMaxRetries = isIdempotent ? maxRetries : 0;

  const endpoint = url.replace(/^https?:\/\/[^/]+/, "");
  const bodyStr = body !== undefined ? JSON.stringify(body) : undefined;
  const finalHeaders: Record<string, string> = { ...headers };
  if (bodyStr !== undefined && !finalHeaders["Content-Type"]) {
    finalHeaders["Content-Type"] = "application/json";
  }
  if (!finalHeaders["Accept"]) {
    finalHeaders["Accept"] = "application/json";
  }

  let lastError: MlServiceError | null = null;

  for (let attempt = 0; attempt <= effectiveMaxRetries; attempt++) {
    // Build a composite AbortController: aborts when EITHER the internal
    // timeout fires OR the caller's external signal aborts.
    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(),
      Math.max(1, timeoutMs),
    );
    const onExternalAbort = () => controller.abort();
    if (externalSignal) {
      if (externalSignal.aborted) controller.abort();
      else externalSignal.addEventListener("abort", onExternalAbort, { once: true });
    }

    try {
      const resp = await fetch(url, {
        method,
        headers: finalHeaders,
        body: bodyStr,
        signal: controller.signal,
        cache: "no-store",
      });

      // Parse the body. Even on error, the body may contain a useful
      // detail message from FastAPI/HTTPException.
      const text = await resp.text();
      let parsedBody: unknown = undefined;
      if (text) {
        try {
          parsedBody = JSON.parse(text);
        } catch {
          parsedBody = { raw: text.slice(0, 1000) };
        }
      }

      if (resp.ok) {
        return { ok: true as const, status: resp.status, body: parsedBody as T };
      }

      // Non-2xx response. Build a structured error.
      let errMsg: string = `HTTP ${resp.status}`;
      if (parsedBody && typeof parsedBody === "object") {
        const obj = parsedBody as Record<string, unknown>;
        if (typeof obj.detail === "string" && obj.detail) {
          errMsg = obj.detail;
        } else if (typeof obj.message === "string" && obj.message) {
          errMsg = obj.message;
        } else if (typeof obj.detail === "object" || typeof obj.message === "object") {
          // FastAPI's HTTPException can take a list of validation errors
          // as detail — stringify the first one.
          errMsg = JSON.stringify(obj.detail ?? obj.message).slice(0, 500);
        }
      } else if (typeof parsedBody === "string" && parsedBody) {
        errMsg = parsedBody.slice(0, 500);
      }

      lastError = new MlServiceError({
        service,
        endpoint,
        message: errMsg,
        httpStatus: resp.status,
        attempt,
        cause: parsedBody,
        isTimeout: false,
        isRetryable: isRetryableStatus(resp.status),
      });

      // If retryable and we have retries left, sleep and continue.
      // BE-037: effectiveMaxRetries is 0 for non-idempotent methods,
      // so this branch is never taken for POST/PUT/PATCH/DELETE unless
      // the caller explicitly passed `idempotent: true`.
      if (isRetryableStatus(resp.status) && attempt < effectiveMaxRetries) {
        const delayMs = retryBaseDelayMs * Math.pow(4, attempt);
        await sleep(delayMs, externalSignal).catch(() => {
          // external signal aborted during sleep — exit the loop.
        });
        continue;
      }

      // Not retryable, or out of retries.
      return { ok: false as const, status: resp.status, error: lastError };
    } catch (err) {
      const isAbort =
        err instanceof Error &&
        (err.name === "AbortError" || err.message === "aborted");
      const isTimeout = isAbort && attempt === effectiveMaxRetries; // only flag timeout if we're done retrying

      lastError = new MlServiceError({
        service,
        endpoint,
        message: err instanceof Error ? err.message : String(err),
        httpStatus: 0,
        attempt,
        cause: err,
        isTimeout,
        isRetryable: !isAbort && (isNetworkError(err) || attempt < effectiveMaxRetries),
      });

      // On abort (timeout or caller cancellation), do NOT retry.
      if (isAbort) {
        return { ok: false as const, status: 0, error: lastError };
      }

      // Network error: retry with backoff if attempts remain.
      if (attempt < effectiveMaxRetries) {
        const delayMs = retryBaseDelayMs * Math.pow(4, attempt);
        await sleep(delayMs, externalSignal).catch(() => {
          // aborted during sleep
        });
        continue;
      }

      return { ok: false as const, status: 0, error: lastError };
    } finally {
      clearTimeout(timeout);
      if (externalSignal) {
        externalSignal.removeEventListener("abort", onExternalAbort);
      }
    }
  }

  // All retries exhausted.
  return {
    ok: false as const,
    status: lastError?.httpStatus ?? 0,
    error:
      lastError ??
      new MlServiceError({
        service,
        endpoint,
        message: "mlFetch exhausted retries with no error captured",
        attempt: effectiveMaxRetries,
      }),
  };
}

/**
 * Convenience wrapper that throws on error instead of returning a union.
 * Use this when the caller prefers try/catch over discriminated unions.
 */
export async function mlFetchOrThrow<T = unknown>(
  url: string,
  options: MlFetchOptions & { service: string },
): Promise<T> {
  const result = await mlFetch<T>(url, options);
  if (!result.ok) throw result.error;
  return result.body;
}

/**
 * Resolve a service URL from environment variables.
 *
 * Honors both the canonical env var (e.g., `PHASE1_SERVICE_URL`) and any
 * legacy aliases (e.g., `DATASET_SERVICE_URL`) for backward compatibility
 * with existing deployments. The canonical name wins.
 *
 * Returns null if neither is set — the caller surfaces a 503 with a clear
 * "service not configured" message.
 */
export function resolveServiceUrl(
  canonical: string,
  ...aliases: string[]
): string | null {
  const all = [canonical, ...aliases];
  for (const name of all) {
    const v = process.env[name];
    if (v && v.trim()) return v.trim().replace(/\/$/, "");
  }
  return null;
}

/**
 * Build a full URL from a service base URL + path.
 *
 * `buildServiceUrl("http://localhost:8003", "/predict")` →
 * `"http://localhost:8003/predict"`. Handles trailing slashes on the base
 * and leading slashes on the path.
 */
export function buildServiceUrl(baseUrl: string, path: string): string {
  const base = baseUrl.replace(/\/$/, "");
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${base}${p}`;
}
