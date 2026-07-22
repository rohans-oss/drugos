/**
 * DEPRECATED — use `@/lib/services/dataset-service` instead.
 *
 * This file is kept ONLY as a backward-compat re-export shim so existing
 * imports (`from "@/lib/services/dataset-stats"`) continue to work after
 * Issue 233 consolidated the dataset service into `dataset-service.ts`.
 *
 * Issue 233 ROOT FIX: the previous implementation had a local-checkpoint
 * fallback that read `../phase1/data/checkpoints/step_01.json` directly,
 * bypassing the Python Phase 1 service. This caused the dashboard to
 * display stale checkpoint data. The new `dataset-service.ts` is
 * HTTP-only — no local file reads.
 *
 * All exports below are re-exported from `dataset-service.ts`. Do not
 * add new code to this file — add it to `dataset-service.ts` instead.
 */

export {
  getDatasetStats,
  getDrugMechanism,
  checkDatasetHealth,
} from "./dataset-service";
export type {
  DatasetStatsResponse,
  DatasetSourceStat,
  DatasetHealthResponse,
} from "./dataset-service";
