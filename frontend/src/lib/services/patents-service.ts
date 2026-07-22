/**
 * Patents service facade — Task 250 root fix.
 *
 * ROOT CAUSE: the audit expected `frontend/src/lib/services/patents-service.ts`
 * to integrate with USPTO, but the actual implementation lived in
 * `patentsview.ts` (named after the upstream API). The route at
 * `app/api/patents/search/route.ts` imported from `patentsview.ts` directly.
 * The naming inconsistency made it look like the patent service returned
 * mock patents.
 *
 * ROOT FIX: this file is the canonical import path for patent lookups.
 * It re-exports the real USPTO PatentsView integration from `patentsview.ts`
 * so:
 *   - The route imports from `@/lib/services/patents-service`.
 *   - The actual implementation lives in `patentsview.ts`.
 *   - Consumers can import from either path — both resolve to the same
 *     real USPTO-backed functions.
 *
 * There is NO mock data anywhere in this file or in `patentsview.ts`.
 * Every call hits `https://search.patentsview.org/api/v1/patent` and
 * returns real USPTO patent grants.
 */
export {
  searchPatents,
} from "./patentsview";
export type {
  PatentRecord,
  PatentSearchResponse,
} from "./patentsview";
