import { NextRequest, NextResponse } from "next/server";
import { searchDrugsByName, getDrugProperties } from "@/lib/services/rxnorm";
import { badRequest, internalError } from "@/lib/api-helpers";
import {
  requireAuthAndRateLimitV2,
  recordApiRequestForUserV2,
} from "@/lib/auth/api-proxy-guard";
// Task 252 ROOT FIX: Zod validation for query params.
import { validateQueryParams, DrugsSearchQuery } from "@/lib/zod-schemas";

/**
 * GET /api/drugs/search?q=<name>&limit=N
 *      /api/drugs/search?rxcui=<id>
 *
 * Task 241 ROOT FIX:
 *
 * ROOT CAUSE: the audit claimed this route "returns mock data" — but the
 * route already calls `searchDrugsByName()` from `rxnorm.ts`, which makes
 * a REAL HTTP call to `https://rxnav.nlm.nih.gov/REST/approximateTerm.json`.
 * The audit's "mock data" claim was outdated by the time the audit was
 * written. However, the route had TWO real defects that the audit did NOT
 * name but which still needed fixing:
 *
 *   1. No Zod validation — a `q=...` containing path-traversal characters
 *      or 10KB of garbage would be forwarded to RxNorm. RxNorm would 400,
 *      but the failure mode was opaque to operators.
 *   2. The route used the V1 per-user rate limit (60 req/MIN = 1 req/sec).
 *      The audit spec calls for 5 req/sec per user. We now use the V2
 *      guard (`requireAuthAndRateLimitV2`) which enforces the correct
 *      5 req/sec sliding window.
 *
 * ROOT FIX:
 *   1. Validate query params with Zod (`DrugsSearchQuery` schema). The
 *      schema enforces a biomedical-name allowlist regex on `q` and a
 *      1-11 digit numeric regex on `rxcui`. Invalid input returns 400
 *      with a structured issue list — never reaches RxNorm.
 *   2. Use `requireAuthAndRateLimitV2` for the 5 req/sec per-user limit.
 *   3. The underlying RxNorm call is wrapped in `monitoredFetch` (see
 *      `rxnorm.ts`) so every call is logged with duration and status.
 *
 * NO MOCK DATA. Every successful response is real RxNorm data.
 */
export async function GET(req: NextRequest) {
  // Task 252: Zod validation fires FIRST — invalid input gets a 400
  // without wasting an auth check. This is the production-safe order:
  // Zod → auth + rate-limit → upstream call.
  const parsed = validateQueryParams(DrugsSearchQuery, req.nextUrl.searchParams);
  if (!parsed.ok) return parsed.response;
  const { q, rxcui, limit } = parsed.data;

  // Task 253: 5 req/sec per-user rate limit (V2 guard).
  const guard = await requireAuthAndRateLimitV2(req);
  if (guard.response !== null) return guard.response;

  try {
    if (rxcui) {
      const props = await getDrugProperties(rxcui);
      recordApiRequestForUserV2(guard.user);
      return NextResponse.json(props);
    }
    if (!q) {
      // Zod allows q to be optional (because rxcui alone is valid). If
      // we get here with no q and no rxcui, the request is malformed.
      return badRequest("Query parameter 'q' (min 2 chars) or 'rxcui' is required");
    }
    const results = await searchDrugsByName(q, limit);
    recordApiRequestForUserV2(guard.user);
    // FE-004 ROOT FIX: Standardize on {items: [...]}.
    return NextResponse.json({ items: results, total: results.length, query: q });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`RxNorm search failed: ${msg}`);
  }
}
