/**
 * External API call monitoring (Task 260 root fix).
 *
 * ROOT CAUSE: Every external API service (openfda.ts, clinical-trials.ts,
 * patentsview.ts, rxnorm.ts, mesh.ts, drug-mechanism.ts) called `fetch()`
 * directly with no observability. When an upstream provider is slow or
 * returns errors, operators had NO signal — the dashboard would just spin
 * or fail silently. For an "institutional grade" pharma platform this is
 * unacceptable: every external call must be logged with its duration,
 * status code, and the upstream provider so an operator can:
 *   - alert when an upstream is degraded (e.g. openFDA p95 > 5s)
 *   - attribute a 5xx to the right upstream (was it CT.gov or openFDA?)
 *   - audit which external providers a given user query touched (for
 *     data-provenance / regulatory review)
 *
 * ROOT FIX: this module exports `monitoredFetch()` — a drop-in wrapper
 * around `fetch()` that:
 *   1. Records the start time before the call.
 *   2. Awaits the response (or error).
 *   3. Logs a structured JSON line to stdout with:
 *        provider, url, method, status, durationMs, ok, error?
 *   4. Re-throws / re-returns the original result — the caller's
 *      behavior is unchanged. Monitoring is non-blocking and
 *      non-throwing; a monitoring failure MUST NOT break the request.
 *
 * The log line is consumed by the platform's log aggregator (CloudWatch /
 * Datadog / Loki). The schema is intentionally flat (no nested objects)
 * so it indexes cleanly.
 *
 * USAGE:
 *   import { monitoredFetch } from "@/lib/external-api-monitor";
 *   const res = await monitoredFetch("openfda", url, { headers: ... });
 *
 * The first argument is a short provider label — one of:
 *   "openfda" | "ctgov" | "patentsview" | "rxnorm" | "mesh" | "chembl" | "kg_service"
 *
 * TESTABILITY: `__getRecentCalls()` is exported for unit tests so we can
 * assert that a given provider was called with the expected URL.
 */

export type ExternalApiProvider =
  | "openfda"
  | "ctgov"
  | "patentsview"
  | "rxnorm"
  | "mesh"
  | "chembl"
  | "kg_service"
  // BE-058 ROOT FIX (v115, LOW): added "pubmed" so pubmed.ts can use
  // monitoredFetch. Previously pubmed.ts used raw fetch() — bypassing
  // the external-api-monitor — so operators had NO visibility into
  // PubMed latency, 429s, or outages. Adding "pubmed" here allows
  // pubmed.ts to route through monitoredFetch like every other
  // external service.
  | "pubmed";

export interface ExternalApiCall {
  provider: ExternalApiProvider;
  url: string;
  method: string;
  status: number;
  durationMs: number;
  ok: boolean;
  error?: string;
  timestamp: string;
}

// Bounded ring buffer of recent calls — used by tests and by a future
// /api/admin/external-apis/status endpoint. 1000 entries is ~200KB max.
const MAX_RECENT = 1000;
const recentCalls: ExternalApiCall[] = [];

/**
 * Test-only helper: return a snapshot of the last N external API calls
 * recorded by this process. The array is a copy so the caller can iterate
 * without mutating internal state.
 */
export function __getRecentCalls(limit = 100): ExternalApiCall[] {
  return recentCalls.slice(-limit).map((c) => ({ ...c }));
}

/** Test-only helper: clear the recent-calls buffer. */
export function __clearRecentCallsForTests(): void {
  recentCalls.length = 0;
}

/**
 * Drop-in replacement for `fetch()` that records the call to the
 * monitoring log. Behavior is identical to `fetch()` — the wrapper only
 * adds observability. If the underlying fetch throws (e.g. network error),
 * we record the error and re-throw.
 *
 * `provider` is a short label so the log line is greppable by provider.
 * Never include credentials or PII in the URL — URLs are logged verbatim.
 */
export async function monitoredFetch(
  provider: ExternalApiProvider,
  url: string,
  init?: RequestInit
): Promise<Response> {
  const method = init?.method || "GET";
  const start = Date.now();
  const timestamp = new Date().toISOString();

  let status = 0;
  let ok = false;
  let errorMessage: string | undefined;

  try {
    const res = await fetch(url, init);
    status = res.status;
    ok = res.ok;
    return res;
  } catch (e: unknown) {
    errorMessage = e instanceof Error ? e.message : String(e);
    status = 0;
    ok = false;
    throw e;
  } finally {
    const durationMs = Date.now() - start;
    const call: ExternalApiCall = {
      provider,
      url,
      method,
      status,
      durationMs,
      ok,
      timestamp,
      ...(errorMessage ? { error: errorMessage } : {}),
    };

    // Append to the in-memory ring buffer (for tests + future admin UI).
    recentCalls.push(call);
    if (recentCalls.length > MAX_RECENT) {
      recentCalls.splice(0, recentCalls.length - MAX_RECENT);
    }

    // Structured log line — JSON so the log aggregator can parse it.
    // We use console.info (not console.log) so operators can filter to a
    // dedicated stream if they wish. We never include request bodies,
    // auth headers, or response bodies — only the URL, status, and timing.
    console.info(
      JSON.stringify({
        event: "external_api_call",
        ...call,
      })
    );

    // Warn-level log for slow calls (>3s) or failures — so operators get
    // a fast signal without grepping the info stream.
    if (!ok || durationMs > 3000) {
      console.warn(
        JSON.stringify({
          event: "external_api_call_slow_or_failed",
          ...call,
        })
      );
    }
  }
}
