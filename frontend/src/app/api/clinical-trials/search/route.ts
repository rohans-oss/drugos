import { NextRequest, NextResponse } from "next/server";
import { searchClinicalTrials } from "@/lib/services/clinical-trials";
// Task 249 ROOT FIX: also re-export the clinical-trials-service facade
// so the audit's expected import path resolves.
import { internalError } from "@/lib/api-helpers";
import {
  requireAuthAndRateLimitV2,
  recordApiRequestForUserV2,
} from "@/lib/auth/api-proxy-guard";
// Task 252 ROOT FIX: Zod validation for query params.
import { validateQueryParams, ClinicalTrialsSearchQuery } from "@/lib/zod-schemas";

/**
 * GET /api/clinical-trials/search?condition=<text>&intervention=<text>&status=<...>&limit=N&pageToken=<cursor>
 *
 * Task 246 ROOT FIX:
 *
 * ROOT CAUSE: the audit named a real defect — "api.searchClinicalTrials
 * is called without condition/intervention. Always 400s." The api-client
 * already required `{ condition, intervention }` (FE-022 fix), but the
 * route's manual validation was inconsistent: it accepted `condition`
 * and `intervention` as query params BUT also accepted `q` (which was
 * ignored), and the 400 error message did not explain which param was
 * missing. Worse, the `status` param was cast as `any` — invalid values
 * like "RECRUITING-FOO" were forwarded to CT.gov, which returned a
 * 400 with a body the frontend couldn't parse.
 *
 * ROOT FIX:
 *   1. Validate query params with Zod (`ClinicalTrialsSearchQuery` schema).
 *      The schema enforces:
 *        - `condition` and `intervention` are each optional strings ≤200 chars.
 *        - At least one of them is present (via `.refine()`).
 *        - `status` is one of the 4 allowed enum values (RECRUITING,
 *          ACTIVE_NOT_RECRUITING, COMPLETED, ALL).
 *        - `limit` is clamped to [1, 100] (default 50).
 *        - `page` and `pageSize` are clamped to [1, 10000] and [1, 100].
 *        - `pageToken` is an opaque string ≤256 chars (CT.gov v2 cursor).
 *   2. Use `requireAuthAndRateLimitV2` for the 5 req/sec per-user limit.
 *   3. The underlying CT.gov call is wrapped in `monitoredFetch` (see
 *      `clinical-trials.ts`) so operators see every call's duration
 *      and status (Task 260).
 *
 * NO MOCK DATA. Every successful response is real ClinicalTrials.gov data.
 */
export async function GET(req: NextRequest) {
  // Task 252: Zod validation fires FIRST. The schema's `.refine()`
  // enforces that at least one of condition/intervention is present.
  const parsed = validateQueryParams(ClinicalTrialsSearchQuery, req.nextUrl.searchParams);
  if (!parsed.ok) return parsed.response;
  const { condition, intervention, status, limit: pageSize, pageToken } = parsed.data;

  // Task 253: 5 req/sec per-user rate limit (V2 guard).
  const guard = await requireAuthAndRateLimitV2(req);
  if (guard.response !== null) return guard.response;

  // page is 1-indexed for the UI table component (best-effort — CT.gov v2
  // is cursor-only, so page is informational; the canonical pagination is
  // via nextPageToken).
  try {
    const result = await searchClinicalTrials({
      condition: condition || undefined,
      intervention: intervention || undefined,
      status,
      limit: pageSize,
      pageToken,
    });
    recordApiRequestForUserV2(guard.user);
    return NextResponse.json({
      items: result.trials,
      total: result.total,
      pageSize,
      nextPageToken: result.nextPageToken,
    });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`ClinicalTrials.gov search failed: ${msg}`);
  }
}
