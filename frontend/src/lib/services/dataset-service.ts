/**
 * Dataset service — unified, HTTP-only (Issue 233).
 *
 * ROOT FIX (forensic, root-level): the previous dataset integration
 * had TWO divergent code paths:
 *
 *   1. `dataset-stats.ts` called `DATASET_SERVICE_URL/stats` on the
 *      Python service. This was CORRECT for the URL, but the env var
 *      name (`DATASET_SERVICE_URL`) was inconsistent with the other
 *      3 services which all use `<PHASE>_SERVICE_URL` naming.
 *
 *   2. The local-checkpoint fallback read
 *      `../phase1/data/checkpoints/step_01.json` directly. Issue 226
 *      documented that this is a Phase 2 BRIDGE SUMMARY, NOT a Phase 1
 *      artifact. The path was wrong (it should have been
 *      `../phase1/data/checkpoints/step_01.json` — which it was, but
 *      the issue description noted it as Phase 2 because the previous
 *      version pointed to `../phase2/data/checkpoints/step_01.json`).
 *
 *      Even with the correct Phase 1 path, reading the checkpoint
 *      directly bypassed the Python service — meaning the dashboard
 *      showed stale checkpoint data even after Phase 1 re-ran. The
 *      Python service reads the SAME checkpoint but can also enrich
 *      it with live CSV row counts (from `phase1/data/processed/`).
 *
 * ROOT FIX: this file is the SINGLE source of truth for Phase 1
 * dataset calls. All operations go through `PHASE1_SERVICE_URL`
 * (with `DATASET_SERVICE_URL` honored as a legacy alias for backward
 * compat) via the shared `mlFetch` HTTP client (Issue 234).
 *
 * URLs are aligned with the Python service's actual endpoints
 * (phase1/service.py):
 *
 *   - GET  /health                  → Liveness probe
 *   - GET  /stats                   → Dataset statistics (sources, counts)
 *   - GET  /datasets                → List of datasets
 *   - GET  /datasets/{drug}/mechanism → Drug mechanism-of-action
 *
 * If neither `PHASE1_SERVICE_URL` nor `DATASET_SERVICE_URL` is set,
 * the functions return a clear `status: "no_data"` response. We NEVER
 * read a local checkpoint as a fallback — the Python service is the
 * single source of truth, and it has its own logic for reading the
 * checkpoint AND enriching with live CSV counts.
 *
 * SCIENTIFIC INTEGRITY: dataset statistics must reflect the actual
 * Phase 1 pipeline state. Reading a stale checkpoint JSON would
 * mislead a researcher into believing all 7 sources are loaded when
 * only 3 have actually completed.
 */

import { mlFetch, resolveServiceUrl, buildServiceUrl, MlServiceError } from "@/lib/http-client";
import {
  DatasetStatsResponseSchema,
  type DatasetStatsResponse,
  type DatasetSourceStat,
  type DatasetHealthResponse,
  validateMlResponse,
} from "@/lib/ml-contracts";

// ---------------------------------------------------------------------------
// Public types (kept stable for existing callers)
// ---------------------------------------------------------------------------

export type { DatasetStatsResponse, DatasetSourceStat, DatasetHealthResponse };

/**
 * Backward-compat alias for the response shape. The previous
 * `dataset-stats.ts` exported `DatasetStatsResponse`; callers that
 * import that name should continue to work — it's re-exported above.
 */

// ---------------------------------------------------------------------------
// Service URL resolution
// ---------------------------------------------------------------------------

const SERVICE_NAME = "phase1_dataset";

function getPhase1ServiceUrl(): string | null {
  // Canonical env var is PHASE1_SERVICE_URL (matches the naming
  // convention of GT_SERVICE_URL / RL_SERVICE_URL / KG_SERVICE_URL).
  // DATASET_SERVICE_URL is honored as a legacy alias for deployments
  // that haven't renamed their env vars yet.
  return resolveServiceUrl("PHASE1_SERVICE_URL", "DATASET_SERVICE_URL");
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Get dataset statistics from the Phase 1 service.
 *
 * Proxies to `GET {PHASE1_SERVICE_URL}/stats` via the shared HTTP client.
 * Returns `status: "no_data"` if the service URL is not set.
 */
export async function getDatasetStats(): Promise<DatasetStatsResponse> {
  const baseUrl = getPhase1ServiceUrl();
  if (!baseUrl) {
    return {
      sources: [],
      nodesLoaded: 0,
      edgesLoaded: 0,
      edgeTypesPresent: [],
      warnings: [],
      errors: [],
      generatedAt: new Date().toISOString(),
      status: "no_data",
      source: "none",
      note:
        "PHASE1_SERVICE_URL is not set. The Phase 1 dataset service " +
        "(phase1/service.py) must be running and reachable. Start it " +
        "with `python phase1/service.py` and set " +
        "PHASE1_SERVICE_URL=http://localhost:8001 in frontend/.env.local. " +
        "Issue 233 ROOT FIX: this endpoint NEVER reads a local checkpoint " +
        "JSON and NEVER fabricates dataset statistics. " +
        "(DATASET_SERVICE_URL is honored as a legacy alias.)",
    };
  }

  const url = buildServiceUrl(baseUrl, "/stats");
  const result = await mlFetch<unknown>(url, {
    service: SERVICE_NAME,
    method: "GET",
    timeoutMs: 15_000,
    maxRetries: 3,
  });

  if (!result.ok) {
    const err = result.error as MlServiceError;
    if (err.httpStatus === 404) {
      // /stats endpoint not found — the service is running an older
      // version that doesn't expose /stats. Surface as service_down.
      return {
        sources: [],
        nodesLoaded: 0,
        edgesLoaded: 0,
        edgeTypesPresent: [],
        warnings: [],
        errors: [
          `Phase 1 service at ${baseUrl} returned 404 for /stats. The ` +
            `service is running an older version that does not expose /stats. ` +
            `Update phase1/service.py to the latest version.`,
        ],
        generatedAt: new Date().toISOString(),
        status: "service_down",
        source: "none",
        note: `Phase 1 service /stats endpoint not found (404).`,
      };
    }
    if (err.httpStatus >= 400 && err.httpStatus < 500) {
      return {
        sources: [],
        nodesLoaded: 0,
        edgesLoaded: 0,
        edgeTypesPresent: [],
        warnings: [],
        errors: [`${err.httpStatus}: ${err.message}`],
        generatedAt: new Date().toISOString(),
        status: "service_down",
        source: "none",
        note: `Phase 1 service rejected request (${err.httpStatus}): ${err.message}`,
      };
    }
    // 5xx / network — re-throw so the caller surfaces a 502.
    throw err;
  }

  const validated = validateMlResponse(
    SERVICE_NAME,
    "/stats",
    DatasetStatsResponseSchema,
    result.body,
  );

  // Ensure the response has the status field (the Python service may
  // not set it — the previous frontend lib added it).
  if (!validated.status) {
    (validated as DatasetStatsResponse & { status: string }).status = "ok";
  }
  if (!validated.source) {
    (validated as DatasetStatsResponse & { source: string }).source =
      "dataset_service";
  }

  return validated;
}

/**
 * Get the mechanism-of-action for a specific drug.
 *
 * Proxies to `GET {PHASE1_SERVICE_URL}/datasets/{drug}/mechanism`.
 */
export async function getDrugMechanism(
  drug: string,
): Promise<Record<string, unknown>> {
  const baseUrl = getPhase1ServiceUrl();
  if (!baseUrl) {
    return { status: "no_data", drug, note: "PHASE1_SERVICE_URL is not set." };
  }

  const url = buildServiceUrl(
    baseUrl,
    `/datasets/${encodeURIComponent(drug)}/mechanism`,
  );
  const result = await mlFetch<unknown>(url, {
    service: SERVICE_NAME,
    method: "GET",
    timeoutMs: 10_000,
    maxRetries: 2,
  });

  if (!result.ok) {
    const err = result.error as MlServiceError;
    if (err.httpStatus === 404) {
      return { status: "not_found", drug, note: err.message };
    }
    throw err;
  }

  return result.body as Record<string, unknown>;
}

/**
 * Health check — useful for the /api/system/status route.
 */
export async function checkDatasetHealth(): Promise<{
  configured: boolean;
  reachable: boolean;
  version?: string;
}> {
  const baseUrl = getPhase1ServiceUrl();
  if (!baseUrl) {
    return { configured: false, reachable: false };
  }
  const url = buildServiceUrl(baseUrl, "/health");
  const result = await mlFetch<unknown>(url, {
    service: SERVICE_NAME,
    method: "GET",
    timeoutMs: 5_000,
    maxRetries: 1,
  });
  if (!result.ok) {
    return { configured: true, reachable: false };
  }
  const body = result.body as Record<string, unknown>;
  return {
    configured: true,
    reachable: true,
    version: typeof body?.version === "string" ? body.version : undefined,
  };
}
