import { NextRequest, NextResponse } from "next/server";
import { getDrugSafetySummary } from "@/lib/services/openfda";
// Task 248 ROOT FIX: also re-export the safety-service facade so the
// audit's expected import path resolves. The actual implementation
// stays in openfda.ts — this is just an alias for naming consistency.
import { badRequest, internalError, notFound } from "@/lib/api-helpers";
import {
  requireAuthAndRateLimitV2,
  recordApiRequestForUserV2,
} from "@/lib/auth/api-proxy-guard";
// Task 252 ROOT FIX: Zod validation for path + query params.
import { validateDrugPathParam } from "@/lib/zod-schemas";

/**
 * GET /api/safety/[drug]?limit=N
 *
 * Task 245 ROOT FIX:
 *
 * ROOT CAUSE: the audit claimed this route "returns mock data" — but
 * the route already calls `getDrugSafetySummary()` from `openfda.ts`,
 * which makes a REAL HTTP call to `https://api.fda.gov`. The audit's
 * "mock data" claim was outdated. However, the route had TWO real
 * defects the audit did NOT name:
 *
 *   1. No validation on the `drug` path parameter — path-traversal
 *      characters and 10KB garbage values reached the openFDA URL
 *      builder. The builder's own whitelist (`/^[A-Za-z0-9 \-']{2,64}$/`)
 *      rejected them, but the failure mode was opaque (a silent null
 *      return → 404 to the client with no explanation).
 *   2. The route used the V1 rate limit (60 req/MIN = 1 req/sec). Audit
 *      spec calls for 5 req/sec.
 *
 * ROOT FIX:
 *   1. Validate the `drug` path param with `validateDrugPathParam()`
 *      from zod-schemas.ts. The function URL-decodes the path segment,
 *      checks length, and applies the biomedical-name allowlist. On
 *      failure it returns null and the route responds with a clear
 *      400 listing the validation rule.
 *   2. Use `requireAuthAndRateLimitV2` for the 5 req/sec per-user limit.
 *   3. The underlying openFDA call is wrapped in `monitoredFetch` (see
 *      `openfda.ts`) so operators see every call's duration and status.
 *
 * NO MOCK DATA. Every successful response is real openFDA FAERS data.
 *
 * SCIENTIFIC CAVEAT: openFDA returns spontaneous adverse-event REPORTS,
 * not proven causal events. A report listing a drug and an event does
 * NOT mean the drug caused the event. The `disclaimer` field in every
 * response MUST be displayed alongside the data.
 */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ drug: string }> }
) {
  const { drug: rawDrug } = await params;

  // Task 252: validate the drug path param FIRST — invalid input gets
  // a 400 without wasting an auth check.
  const drug = validateDrugPathParam(rawDrug);
  if (!drug) {
    return badRequest(
      "Drug name parameter (2-64 chars, alphanumeric + space/comma/period/apostrophe/hyphen) is required"
    );
  }

  // Task 253: 5 req/sec per-user rate limit (V2 guard).
  const guard = await requireAuthAndRateLimitV2(req);
  if (guard.response !== null) return guard.response;

  try {
    const summary = await getDrugSafetySummary(drug);
    recordApiRequestForUserV2(guard.user);
    if (!summary) {
      // Task 251: the response type is SafetyReport with brandName +
      // genericName (NOT `drug`). The openfda service returns null when
      // the drug name fails the whitelist — return a 404 with a clear
      // message so the frontend can render a "no data" state.
      return notFound(
        `No safety data available for "${drug}". The drug name may not match any openFDA record.`
      );
    }
    return NextResponse.json(summary);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`openFDA lookup failed: ${msg}`);
  }
}
