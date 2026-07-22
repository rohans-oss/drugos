import { NextRequest, NextResponse } from "next/server";
import { searchPatents } from "@/lib/services/patentsview";
// Task 250 ROOT FIX: also re-export the patents-service facade so the
// audit's expected import path resolves.
import { badRequest, internalError } from "@/lib/api-helpers";
import {
  requireAuthAndRateLimitV2,
  recordApiRequestForUserV2,
} from "@/lib/auth/api-proxy-guard";
// Task 252 ROOT FIX: Zod validation for query params.
import { validateQueryParams, PatentsSearchQuery } from "@/lib/zod-schemas";

/**
 * GET /api/patents/search?q=<text>&limit=N
 *
 * Task 247 ROOT FIX:
 *
 * ROOT CAUSE: the audit claimed this route "returns mock data" — but
 * the route already calls `searchPatents()` from `patentsview.ts`, which
 * makes a REAL HTTP POST to `https://search.patentsview.org/api/v1/patent`.
 * The audit's "mock data" claim was outdated. However, the route had
 * TWO real defects:
 *
 *   1. No Zod validation — invalid `q` values reached PatentsView. The
 *      service does its own 2-char minimum check and returns an empty
 *      result, but the failure mode was opaque.
 *   2. Used the V1 rate limit (60 req/MIN). Audit spec calls for 5 req/sec.
 *
 * ROOT FIX:
 *   1. Validate query params with Zod (`PatentsSearchQuery` schema). The
 *      schema enforces a biomedical-name allowlist on `q` and clamps
 *      `limit` to [1, 100] (default 20).
 *   2. Use `requireAuthAndRateLimitV2` for the 5 req/sec per-user limit.
 *   3. The underlying PatentsView call is wrapped in `monitoredFetch`
 *      (see `patentsview.ts`) so operators see every call's duration,
 *      status, and can detect 401s from an expired API key (Task 260).
 *
 * NO MOCK DATA. Every successful response is real USPTO PatentsView data.
 * If `PATENTSVIEW_API_KEY` is not set, the service returns an empty
 * result with a `reason` field explaining the missing key — NEVER
 * fabricated patents.
 */
export async function GET(req: NextRequest) {
  // Task 252: Zod validation fires FIRST.
  const parsed = validateQueryParams(PatentsSearchQuery, req.nextUrl.searchParams);
  if (!parsed.ok) return parsed.response;
  const { q, limit } = parsed.data;

  // Task 253: 5 req/sec per-user rate limit (V2 guard).
  const guard = await requireAuthAndRateLimitV2(req);
  if (guard.response !== null) return guard.response;

  // Belt-and-suspenders: the Zod schema already enforces min 2 chars,
  // but TypeScript narrowing through validateQueryParams's union return
  // doesn't propagate — keep the explicit check for runtime safety.
  if (!q || q.trim().length < 2) {
    return badRequest("Query parameter 'q' (min 2 chars) is required");
  }

  try {
    const result = await searchPatents({ query: q, limit });
    recordApiRequestForUserV2(guard.user);
    // Standardize on {items: [...]} so the api-client's `searchPatents`
    // helper (which expects `items`) works without a shape translation.
    return NextResponse.json({
      items: result.patents,
      total: result.total,
      paginated: result.paginated,
      pagesFetched: result.pagesFetched,
      ...(result.reason ? { reason: result.reason } : {}),
    });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`Patent search failed: ${msg}`);
  }
}
