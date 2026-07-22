/**
 * Graph Transformer (Phase 3) inference service — HTTP-only (Issue 230).
 *
 * ROOT FIX (forensic, root-level): the previous version of this file
 * had THREE inference paths, all of which were broken in different ways:
 *
 *   1. Subprocess path (`runPythonInference`): spawned `python3
 *      scripts/gt_inference.py` per request. Issue 221/222 documented
 *      that this script DOES NOT EXIST at `frontend/scripts/gt_inference.py`
 *      (the path the route resolved). The actual script lives at
 *      `<repo>/scripts/gt_inference.py` — but the path resolution used
 *      `process.cwd()` which is `frontend/` in dev, so the script was
 *      never found. Every /api/predict and /api/top-k request returned
 *      `source: "none"` with a "GT inference helper not found" note.
 *
 *   2. HTTP path (`runHttpInference`): worked, but only when
 *      `GT_SERVICE_URL` was set. Most deployments didn't set it because
 *      the .env.example didn't document it (fixed in Issue 240).
 *
 *   3. Checkpoint-search path (`findLatestGtCheckpoint`): scanned
 *      `output_v100/`, `output/`, `graph_transformer/checkpoints/` for
 *      a `.pt` file. Even if found, the subprocess would still fail to
 *      spawn (see #1). This was dead code that gave the false impression
 *      of a fallback.
 *
 * ROOT FIX: this file is now HTTP-ONLY. There is no subprocess path,
 * no checkpoint search, no fs.watch, no tmp files. All GT inference
 * goes through `GT_SERVICE_URL/predict` and `GT_SERVICE_URL/top-k`
 * via the shared `mlFetch` HTTP client (Issue 234).
 *
 * If `GT_SERVICE_URL` is not set, the functions return a clear
 * `source: "none"` response with a message telling the operator to
 * set the env var and start the Phase 3 service. We NEVER fabricate
 * predictions, NEVER spawn a subprocess, NEVER read a checkpoint
 * directly from disk.
 *
 * SCIENTIFIC INTEGRITY: a GT score is a model output. Only the trained
 * model can produce it. Reading a checkpoint from disk in JS would
 * require reimplementing PyTorch inference in JS — which would drift
 * from the Python path within one PR. The HTTP path guarantees the
 * JS and Python paths produce IDENTICAL predictions (the Python
 * service is the single source of truth).
 */

import { mlFetch, resolveServiceUrl, buildServiceUrl, MlServiceError } from "@/lib/http-client";
import {
  GtPredictResponseSchema,
  GtTopKResponseSchema,
  type GtPrediction,
  validateMlResponse,
} from "@/lib/ml-contracts";

// ---------------------------------------------------------------------------
// Public types (kept stable for existing callers)
// ---------------------------------------------------------------------------

export interface DrugDiseasePair {
  drug: string;
  disease: string;
}

// Re-export for backward compat with existing imports.
export type { GtPrediction };

export interface GtInferenceResponse {
  predictions: GtPrediction[];
  source: "gt_checkpoint" | "none";
  modelVersion?: string;
  generatedAt: string;
  count: number;
  checkpointPath?: string | null;
  error_count?: number;
  error_rate?: number;
  note?: string;
}

// ---------------------------------------------------------------------------
// Service URL resolution
// ---------------------------------------------------------------------------

const SERVICE_NAME = "phase3_gt";

function getGtServiceUrl(): string | null {
  // Canonical env var is GT_SERVICE_URL. No aliases — the subprocess
  // path is gone, so GT_REPO_ROOT / GT_CHECKPOINT_DIR are no longer
  // relevant (they were only used by the subprocess path).
  return resolveServiceUrl("GT_SERVICE_URL");
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Score arbitrary (drug, disease) pairs with the trained GT model.
 *
 * Proxies to `POST {GT_SERVICE_URL}/predict` via the shared HTTP client.
 * Returns `{source: "none", predictions: [], ...}` if GT_SERVICE_URL is
 * not set — we NEVER fabricate scores.
 *
 * Throws `MlServiceError` if the service is configured but unreachable
 * after retries. Callers should catch and surface as 502.
 */
export async function predictPairs(
  pairs: DrugDiseasePair[],
): Promise<GtInferenceResponse> {
  if (pairs.length === 0) {
    return {
      predictions: [],
      source: "none",
      generatedAt: new Date().toISOString(),
      count: 0,
      checkpointPath: null,
      note: "No pairs supplied.",
    };
  }

  const baseUrl = getGtServiceUrl();
  if (!baseUrl) {
    return {
      predictions: [],
      source: "none",
      generatedAt: new Date().toISOString(),
      count: 0,
      checkpointPath: null,
      note:
        "GT_SERVICE_URL is not set. The Phase 3 Graph Transformer service " +
        "(graph_transformer/service.py) must be running and reachable. " +
        "Start it with `python graph_transformer/service.py` and set " +
        "GT_SERVICE_URL=http://localhost:8003 in frontend/.env.local. " +
        "Issue 230 ROOT FIX: this endpoint NEVER spawns a subprocess and " +
        "NEVER fabricates GT scores.",
    };
  }

  const url = buildServiceUrl(baseUrl, "/predict");
  const result = await mlFetch<unknown>(url, {
    service: SERVICE_NAME,
    method: "POST",
    body: { pairs },
    timeoutMs: 60_000, // GT inference can take time on large pair lists
    maxRetries: 2, // predictions are deterministic; 2 retries is enough
  });

  if (!result.ok) {
    const err = result.error as MlServiceError;
    // 503 from the service = checkpoint not loaded. Surface as source:none
    // so the dashboard shows "model not trained yet" instead of a 502.
    if (err.httpStatus === 503) {
      return {
        predictions: [],
        source: "none",
        generatedAt: new Date().toISOString(),
        count: 0,
        checkpointPath: null,
        note: `GT service is running but no checkpoint is loaded: ${err.message}`,
      };
    }
    // 4xx = bad request (e.g., drug not in graph). Surface the message.
    if (err.httpStatus >= 400 && err.httpStatus < 500) {
      return {
        predictions: [],
        source: "none",
        generatedAt: new Date().toISOString(),
        count: 0,
        checkpointPath: null,
        note: `GT service rejected request (${err.httpStatus}): ${err.message}`,
      };
    }
    // Network error (ECONNREFUSED, timeout, etc.) — the service is
    // configured but not reachable. Return source:"none" with a clear
    // note instead of throwing a 500. The acceptance criteria requires
    // /api/predict to return predictions (not 500) — "no predictions
    // yet, service down" is an acceptable answer; a 500 crash is not.
    if (err.httpStatus === 0 || err.isTimeout) {
      return {
        predictions: [],
        source: "none",
        generatedAt: new Date().toISOString(),
        count: 0,
        checkpointPath: null,
        note:
          `GT service at ${getGtServiceUrl()} is not reachable: ${err.message}. ` +
          `Check that 'python graph_transformer/service.py' is running and ` +
          `GT_SERVICE_URL is correct. Issue 230 ROOT FIX: returning ` +
          `source:"none" instead of 500 so the dashboard shows a clear state.`,
      };
    }
    // Other 5xx — re-throw so the caller surfaces a 502.
    throw err;
  }

  // Validate the response against the contract.
  const validated = validateMlResponse(
    SERVICE_NAME,
    "/predict",
    GtPredictResponseSchema,
    result.body,
  );

  return {
    predictions: validated.predictions,
    source: "gt_checkpoint",
    modelVersion: validated.modelVersion,
    generatedAt: validated.generatedAt,
    count: validated.count,
    checkpointPath: validated.checkpointPath ?? null,
    error_count: validated.error_count,
    error_rate: validated.error_rate,
  };
}

/**
 * Return the top-K highest-scoring NOVEL (drug, disease) pairs from the
 * trained GT model. "Novel" = not in the known_pairs training set.
 *
 * Proxies to `GET {GT_SERVICE_URL}/top-k?k=<n>` via the shared HTTP client.
 */
export async function topKNovel(
  topK: number = 50,
): Promise<GtInferenceResponse> {
  const baseUrl = getGtServiceUrl();
  if (!baseUrl) {
    return {
      predictions: [],
      source: "none",
      generatedAt: new Date().toISOString(),
      count: 0,
      checkpointPath: null,
      note:
        "GT_SERVICE_URL is not set. The Phase 3 Graph Transformer service " +
        "(graph_transformer/service.py) must be running and reachable. " +
        "Issue 230 ROOT FIX: this endpoint NEVER spawns a subprocess.",
    };
  }

  const k = Math.max(1, Math.min(500, Math.floor(topK)));
  const url = buildServiceUrl(baseUrl, `/top-k?k=${k}`);
  const result = await mlFetch<unknown>(url, {
    service: SERVICE_NAME,
    method: "GET",
    timeoutMs: 60_000,
    maxRetries: 2,
  });

  if (!result.ok) {
    const err = result.error as MlServiceError;
    if (err.httpStatus === 503) {
      return {
        predictions: [],
        source: "none",
        generatedAt: new Date().toISOString(),
        count: 0,
        checkpointPath: null,
        note: `GT service is running but no checkpoint is loaded: ${err.message}`,
      };
    }
    if (err.httpStatus >= 400 && err.httpStatus < 500) {
      return {
        predictions: [],
        source: "none",
        generatedAt: new Date().toISOString(),
        count: 0,
        checkpointPath: null,
        note: `GT service rejected request (${err.httpStatus}): ${err.message}`,
      };
    }
    // Network error — same graceful handling as predictPairs.
    if (err.httpStatus === 0 || err.isTimeout) {
      return {
        predictions: [],
        source: "none",
        generatedAt: new Date().toISOString(),
        count: 0,
        checkpointPath: null,
        note:
          `GT service at ${getGtServiceUrl()} is not reachable: ${err.message}. ` +
          `Check that 'python graph_transformer/service.py' is running.`,
      };
    }
    throw err;
  }

  const validated = validateMlResponse(
    SERVICE_NAME,
    "/top-k",
    GtTopKResponseSchema,
    result.body,
  );

  return {
    predictions: validated.predictions,
    source: "gt_checkpoint",
    modelVersion: validated.modelVersion,
    generatedAt: validated.generatedAt,
    count: validated.count,
    checkpointPath: validated.checkpointPath ?? null,
  };
}

/**
 * Health check — useful for the /api/system/status route.
 *
 * Returns the raw health response from the GT service, or null if the
 * service is not configured / unreachable.
 */
export async function checkGtHealth(): Promise<{
  configured: boolean;
  reachable: boolean;
  checkpointLoaded: boolean;
  version?: string;
}> {
  const baseUrl = getGtServiceUrl();
  if (!baseUrl) {
    return { configured: false, reachable: false, checkpointLoaded: false };
  }
  const url = buildServiceUrl(baseUrl, "/health");
  const result = await mlFetch<unknown>(url, {
    service: SERVICE_NAME,
    method: "GET",
    timeoutMs: 5_000,
    maxRetries: 1,
  });
  if (!result.ok) {
    return { configured: true, reachable: false, checkpointLoaded: false };
  }
  const body = result.body as Record<string, unknown>;
  return {
    configured: true,
    reachable: true,
    checkpointLoaded: Boolean(body?.checkpoint_loaded),
    version: typeof body?.version === "string" ? body.version : undefined,
  };
}
