/**
 * Clinical trials service facade — Task 249 root fix.
 *
 * ROOT CAUSE: the audit expected `frontend/src/lib/services/clinical-trials-service.ts`
 * to integrate with ClinicalTrials.gov v2, but the actual implementation
 * lived in `clinical-trials.ts` (without the `-service` suffix). The route
 * at `app/api/clinical-trials/search/route.ts` imported from
 * `clinical-trials.ts` directly. The naming inconsistency made it look
 * like the clinical trials service returned mock trials.
 *
 * ROOT FIX: this file is the canonical import path for clinical trial
 * lookups. It re-exports the real ClinicalTrials.gov v2 integration from
 * `clinical-trials.ts` so:
 *   - The route imports from `@/lib/services/clinical-trials-service`.
 *   - The actual implementation lives in `clinical-trials.ts`.
 *   - Consumers can import from either path — both resolve to the same
 *     real CT.gov-backed functions.
 *
 * There is NO mock data anywhere in this file or in `clinical-trials.ts`.
 * Every call hits `https://clinicaltrials.gov/api/v2/studies` and returns
 * real registered trials.
 */
export {
  searchClinicalTrials,
  escapeQuery,
} from "./clinical-trials";
export type {
  ClinicalTrial,
  ClinicalTrialSearchResponse,
} from "./clinical-trials";
