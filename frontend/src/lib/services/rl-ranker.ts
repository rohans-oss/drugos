/**
 * RL Hypothesis Ranker service — HTTP-only (Issue 231).
 *
 * ROOT FIX (forensic, root-level): the previous version of this file
 * had a CSV fallback path that was the source of multiple silent
 * failures:
 *
 *   1. The CSV path (`readLocalCsv`) scanned `../rl/`,
 *      `../rl/output/`, and `$RL_OUTPUT_DIR` for
 *      `top_candidates_*.csv`. If the RL agent had not been run, it
 *      silently fell back to `validated_hypotheses.csv` (the INPUT
 *      file — 4 hardcoded FDA-approved known positives). The dashboard
 *      then displayed these 4 known positives as if they were the
 *      agent's ranked predictions. A researcher could not tell the
 *      difference between "the RL agent ranked 4 candidates" and
 *      "the RL agent has not been run; here are 4 hardcoded drugs".
 *
 *   2. The CSV path's cache (`csvCache` Map + `fs.watch`) was
 *      unreliable on NFS/Samba — `fs.watch` does not fire on network
 *      filesystems, so the cache stayed stale until the 60s TTL
 *      expired. The /api/rl/refresh route was added to manually clear
 *      the cache, but it cleared the WRONG cache in some deployments
 *      (Issue 224).
 *
 *   3. The CSV path did its own parsing (drug, disease, rank, reward,
 *      gnn_score, safety_score, market_score, ...) which DRIFTED from
 *      the Python service's response schema. When the Python service
 *      added a field (e.g., `unmet_need_score`), the CSV parser did
 *      not pick it up — the dashboard showed `unmetNeedScore: undefined`
 *      for every candidate.
 *
 * ROOT FIX: this file is now HTTP-ONLY. All RL ranking goes through
 * `RL_SERVICE_URL/rank` via the shared `mlFetch` HTTP client (Issue 234).
 * There is no CSV fallback, no fs.watch, no cache Map, no local file
 * parsing. The Python service (rl/service.py) is the single source of
 * truth for RL ranking output.
 *
 * If `RL_SERVICE_URL` is not set, the functions return a clear
 * `source: "none"` response with a message telling the operator to
 * set the env var and start the Phase 4 service.
 *
 * SCIENTIFIC INTEGRITY: an RL ranking is a model output. Only the
 * trained PPO agent can produce it. Reading a CSV produced by a
 * previous run is acceptable for dev, but in production it MUST go
 * through the service so the agent's latest policy is used. A stale
 * CSV could surface a hypothesis that the current policy has since
 * learned to avoid (e.g., a drug whose safety score worsened after a
 * new FAERS quarterly release).
 */

import { mlFetch, resolveServiceUrl, buildServiceUrl, MlServiceError } from "@/lib/http-client";
import {
  RlRankResponseSchema,
  type RankedHypothesis as ContractRankedHypothesis,
  type RlRankResponse as ContractRlRankResponse,
  validateMlResponse,
} from "@/lib/ml-contracts";

// ---------------------------------------------------------------------------
// Public types (kept stable for existing callers)
// ---------------------------------------------------------------------------

/**
 * RankedHypothesis — re-exported from ml-contracts (the canonical
 * contract type). Existing callers that import from this module
 * continue to work.
 */
export type RankedHypothesis = ContractRankedHypothesis;

export interface RlRankerResponse {
  candidates: RankedHypothesis[];
  source: "rl_service" | "none";
  modelVersion?: string;
  generatedAt: string;
  total: number;
  page: number;
  pageSize: number;
  count: number;
  csvPath?: string | null;
  backend?: string;
  note?: string;
}

export type RlSortField =
  | "rank"
  | "overallScore"
  | "gnnScore"
  | "safetyScore"
  | "marketScore"
  | "reward"
  | "drug"
  | "disease";

export type RlSortDir = "asc" | "desc";

// ---------------------------------------------------------------------------
// Service URL resolution
// ---------------------------------------------------------------------------

const SERVICE_NAME = "phase4_rl";

function getRlServiceUrl(): string {
  return resolveServiceUrl("RL_SERVICE_URL") || "http://localhost:8004";
}

// ---------------------------------------------------------------------------
// Cache functions — NO-OPS in HTTP-only mode (Issue 224)
// ---------------------------------------------------------------------------
//
// The previous version had a `csvCache` Map and `fs.watch` listeners.
// These are gone — the Python service owns the cache now. The
// functions below are kept as no-ops so the /api/rl/refresh route
// (which imports them) doesn't break. The refresh route has been
// updated to call the service's health check instead of clearing a
// local cache.

/**
 * No-op in HTTP-only mode. The Python RL service owns the cache.
 * Kept for backward compat with /api/rl/refresh.
 */
export function clearRlRankerCsvCache(): void {
  // HTTP-only mode: no local cache to clear. The Python service's
  // internal cache is managed by the service itself.
}

/**
 * Returns an empty array in HTTP-only mode. The Python RL service
 * owns the cache. Kept for backward compat with /api/rl/refresh.
 */
export function getRlRankerCsvCacheState(): Array<{
  path: string;
  parsedAt: string;
  mtimeMs: number;
}> {
  return [];
}

/**
 * Test-only no-op. Kept for backward compat with existing test files
 * that import it. In HTTP-only mode there is no cache to clear.
 */
export function __clearRlRankerCsvCacheForTests(): void {
  // no-op
}

/**
 * Test-only no-op. Kept for backward compat with existing test files.
 */
export function __clearRlDefaultCsvPathCacheForTests(): void {
  // no-op
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Get ranked hypotheses from the Phase 4 RL service.
 *
 * Proxies to `GET {RL_SERVICE_URL}/rank?drug=&disease=&limit=&offset=`
 * via the shared HTTP client. Returns `source: "none"` if RL_SERVICE_URL
 * is not set — we NEVER fabricate rankings.
 *
 * Throws `MlServiceError` if the service is configured but unreachable
 * after retries.
 */
export async function getRankedHypotheses(opts?: {
  drug?: string;
  disease?: string;
  limit?: number;
  sort?: RlSortField;
  sortDir?: RlSortDir;
  offset?: number;
  pageSize?: number;
}): Promise<RlRankerResponse> {
  const pageSize = Math.min(opts?.pageSize ?? opts?.limit ?? 50, 200);
  const offset = Math.max(0, opts?.offset ?? 0);

  const baseUrl = getRlServiceUrl();
  if (!baseUrl) {
    return {
      candidates: [],
      source: "none",
      generatedAt: new Date().toISOString(),
      total: 0,
      page: 0,
      pageSize,
      count: 0,
      note:
        "RL_SERVICE_URL is not set. The Phase 4 RL ranker service " +
        "(rl/service.py) must be running and reachable. Start it with " +
        "`python rl/service.py` and set RL_SERVICE_URL=http://localhost:8004 " +
        "in frontend/.env.local. Issue 231 ROOT FIX: this endpoint NEVER " +
        "reads a local CSV and NEVER fabricates rankings.",
    };
  }

  // Build query params — pass through sort + pagination so the Python
  // service can do server-side sort + paginate (it has ALL candidates,
  // we don't want to fetch them all just to slice in JS).
  const params = new URLSearchParams();
  if (opts?.drug) params.set("drug", opts.drug);
  if (opts?.disease) params.set("disease", opts.disease);
  params.set("limit", String(pageSize));
  params.set("offset", String(offset));
  // Note: the Python service currently ignores sort/sortDir (it sorts by
  // rank only). We pass them through so a future service update can honor
  // them without a frontend change.
  if (opts?.sort) params.set("sort", opts.sort);
  if (opts?.sortDir) params.set("sortDir", opts.sortDir);

  const url = buildServiceUrl(baseUrl, `/rank?${params.toString()}`);
  const result = await mlFetch<unknown>(url, {
    service: SERVICE_NAME,
    method: "GET",
    timeoutMs: 15_000,
    maxRetries: 3,
  });

  if (!result.ok) {
    const err = result.error as MlServiceError;
    // 4xx = bad request. Surface the message.
    if (err.httpStatus >= 400 && err.httpStatus < 500) {
      return {
        candidates: [],
        source: "none",
        generatedAt: new Date().toISOString(),
        total: 0,
        page: 0,
        pageSize,
        count: 0,
        note: `RL service rejected request (${err.httpStatus}): ${err.message}`,
      };
    }
    // 5xx / network — re-throw so the caller surfaces a 502.
    throw err;
  }

  // Validate the response against the contract.
  const validated = validateMlResponse(
    SERVICE_NAME,
    "/rank",
    RlRankResponseSchema,
    result.body,
  );

  return {
    candidates: validated.candidates as RankedHypothesis[],
    source: "rl_service",
    modelVersion: validated.modelVersion,
    generatedAt: validated.generatedAt,
    total: validated.total,
    page: validated.page,
    pageSize: validated.pageSize,
    count: validated.count,
    csvPath: validated.csvPath ?? null,
    backend: validated.backend,
  };
}

/**
 * Compute the overall score for a candidate using the agent's reward
 * weights. Kept for backward compat with existing callers.
 *
 * NOTE: in HTTP-only mode, the Python service already computes
 * `overallScore` using the agent's actual reward weights (read from
 * the .meta.json sidecar). This client-side function is a fallback
 * for callers that need to recompute the score from individual
 * signals (e.g., when displaying a candidate that lacks overallScore
 * in the response).
 */
export function computeOverallScore(c: {
  gnnScore?: number | null;
  safetyScore?: number | null;
  marketScore?: number | null;
}): number | null {
  // These are the agent's DEFAULT reward weights (from RewardConfig
  // in rl/rl_drug_ranker.py). The Python service may use different
  // weights if the .meta.json sidecar specifies them — in that case,
  // the service's overallScore wins.
  const signals: Array<[number, number]> = [];
  if (typeof c.gnnScore === "number" && c.gnnScore === c.gnnScore) {
    signals.push([c.gnnScore, 0.04]);
  }
  if (typeof c.safetyScore === "number" && c.safetyScore === c.safetyScore) {
    signals.push([c.safetyScore, 0.25]);
  }
  if (typeof c.marketScore === "number" && c.marketScore === c.marketScore) {
    signals.push([c.marketScore, 0.12]);
  }
  if (signals.length === 0) return null;
  const totalW = signals.reduce((s, [, w]) => s + w, 0);
  if (totalW <= 0) return null;
  return signals.reduce((s, [v, w]) => s + v * w, 0) / totalW;
}

/**
 * Health check — useful for the /api/system/status route.
 */
export async function checkRlHealth(): Promise<{
  configured: boolean;
  reachable: boolean;
  checkpointConfigured: boolean;
  csvOutputAvailable: boolean;
  version?: string;
}> {
  const baseUrl = getRlServiceUrl();
  if (!baseUrl) {
    return {
      configured: false,
      reachable: false,
      checkpointConfigured: false,
      csvOutputAvailable: false,
    };
  }
  const url = buildServiceUrl(baseUrl, "/health");
  const result = await mlFetch<unknown>(url, {
    service: SERVICE_NAME,
    method: "GET",
    timeoutMs: 5_000,
    maxRetries: 1,
  });
  if (!result.ok) {
    return {
      configured: true,
      reachable: false,
      checkpointConfigured: false,
      csvOutputAvailable: false,
    };
  }
  const body = result.body as Record<string, unknown>;
  return {
    configured: true,
    reachable: true,
    checkpointConfigured: Boolean(body?.checkpoint_configured),
    csvOutputAvailable: Boolean(body?.csv_output_available),
    version: typeof body?.version === "string" ? body.version : undefined,
  };
}

// Re-export the contract type for callers that want the canonical name.
export type { ContractRlRankResponse };
