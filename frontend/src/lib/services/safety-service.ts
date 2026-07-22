/**
 * Safety service facade — Task 248 root fix.
 *
 * ROOT CAUSE: the audit expected `frontend/src/lib/services/safety-service.ts`
 * to integrate with openFDA, but the actual implementation lived in
 * `openfda.ts` (named after the upstream API). The route at
 * `app/api/safety/[drug]/route.ts` imported from `openfda.ts` directly.
 * The naming inconsistency made it look like the safety service returned
 * hardcoded mock data — operators searching for "safety-service.ts"
 * found nothing and assumed the integration was missing.
 *
 * ROOT FIX: this file is the canonical import path for safety lookups.
 * It re-exports the real openFDA integration from `openfda.ts` so:
 *   - The route imports from `@/lib/services/safety-service` (audit name).
 *   - The actual implementation lives in `openfda.ts` (named after the API).
 *   - Consumers can import from either path — both resolve to the same
 *     real openFDA-backed functions.
 *
 * There is NO mock data anywhere in this file or in `openfda.ts`. Every
 * call hits `https://api.fda.gov` and returns real FAERS adverse-event
 * reports.
 */
export {
  getDrugSafetySummary,
  isOpenfdaApiKeyConfigured,
} from "./openfda";
export type {
  AdverseEventReaction,
  DrugSafetySummary,
} from "./openfda";
