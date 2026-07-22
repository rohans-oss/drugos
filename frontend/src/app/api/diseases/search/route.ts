import { NextRequest, NextResponse } from "next/server";
import { searchDiseasesByName } from "@/lib/services/mesh";
import { badRequest, internalError } from "@/lib/api-helpers";
import {
  requireAuthAndRateLimitV2,
  recordApiRequestForUserV2,
} from "@/lib/auth/api-proxy-guard";
// Task 252 ROOT FIX: Zod validation for query params.
import { validateQueryParams, DiseasesSearchQuery } from "@/lib/zod-schemas";

/**
 * GET /api/diseases/search?q=<name>&limit=N
 *
 * Task 244 ROOT FIX:
 *
 * ROOT CAUSE: the audit named a type mismatch — `DiseaseSearchResult.descriptorUI`
 * (uppercase 'I') vs the actual `descriptorUi` (lowercase 'i') returned by the
 * MeSH service. The previous "fix" already renamed the api-client type to
 * `descriptorUi` — BUT the route's response shape came from `MeshDescriptor`
 * which also uses `descriptorUi`. So at the type level this was already aligned.
 *
 * However, the route had two real defects:
 *   1. No Zod validation — invalid `q` values reached MeSH and produced
 *      opaque 400s.
 *   2. Used the V1 rate limit (60 req/MIN). Audit spec calls for 5 req/sec.
 *
 * ROOT FIX:
 *   1. Validate query params with Zod (`DiseasesSearchQuery` schema).
 *      The schema enforces a biomedical-name allowlist on `q` and a
 *      1-100 clamped integer on `limit`.
 *   2. Use `requireAuthAndRateLimitV2` for the 5 req/sec per-user limit.
 *   3. The underlying MeSH calls are wrapped in `monitoredFetch` so
 *      operators see every NLM call's duration and status (Task 260).
 *
 * NO MOCK DATA. Every successful response is real MeSH data.
 */
export async function GET(req: NextRequest) {
  // Task 252: Zod validation fires FIRST.
  const parsed = validateQueryParams(DiseasesSearchQuery, req.nextUrl.searchParams);
  if (!parsed.ok) return parsed.response;
  const { q, limit } = parsed.data;

  // Task 253: 5 req/sec per-user rate limit (V2 guard).
  const guard = await requireAuthAndRateLimitV2(req);
  if (guard.response !== null) return guard.response;

  // The Zod schema already enforces min 2 chars, but TypeScript narrowing
  // doesn't propagate through validateQueryParams's union return — keep
  // the explicit check for runtime safety.
  if (!q || q.trim().length < 2) {
    return badRequest("Query parameter 'q' (min 2 chars) is required");
  }

  try {
    const results = await searchDiseasesByName(q, limit);
    recordApiRequestForUserV2(guard.user);
    // FE-005 ROOT FIX: Standardize on {items: [...]}.
    return NextResponse.json({ items: results, total: results.length, query: q });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`MeSH search failed: ${msg}`);
  }
}
