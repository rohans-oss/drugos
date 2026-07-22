import { NextRequest, NextResponse } from "next/server";
import { searchPubMed } from "@/lib/services/pubmed";
import { badRequest, internalError } from "@/lib/api-helpers";
import {
  requireAuthAndRateLimit,
  recordApiRequestForUser,
} from "@/lib/auth/api-proxy-guard";

// FE-006 ROOT FIX: This route previously had NO authentication. Anyone on
// the internet could use it as an open proxy to deplete our NCBI_API_KEY
// quota (10 req/sec, 1M req/day). Once exhausted, ALL researchers' PubMed
// queries would fail. Now it requires auth + a per-user rate limit.
//
// FE-006 ROOT FIX (response shape): Previously this route returned the raw
// service response `{ total, articles }`. The api-client.ts's
// searchLiterature() expected `{ items: PubMedArticle[] }` and accessed
// response.items.map(...). Since `items` was undefined, the .map() call
// threw "Cannot read properties of undefined (reading 'map')" and the
// literature search UI crashed on every search.
//
// ROOT FIX: Standardize on `{ items: [...], total: number, ... }`. We map
// the service's `articles` field to `items` and pass through `total`,
// `limit`, and `offset` for paginated follow-up requests.
//
// Issue 229 ROOT FIX: the audit reported that `api.searchClinicalTrials`
// was called without `condition`/`intervention` and always 400'd. After
// forensic code reading, the literature/search route does NOT call
// searchClinicalTrials — it calls searchPubMed directly (line 43 below).
// The `api.searchClinicalTrials` client in api-client.ts (line 577) WAS
// fixed in a prior FE-022 commit to require `{ condition?, intervention? }`.
// This route is verified correct: it calls searchPubMed with a `query`
// string, which is the correct contract for PubMed literature search
// (PubMed accepts free-text queries, not condition/intervention pairs).
export async function GET(req: NextRequest) {
  const guard = await requireAuthAndRateLimit(req);
  if (guard.response !== null) return guard.response;

  const query = req.nextUrl.searchParams.get("q") || "";
  const limit = parseInt(req.nextUrl.searchParams.get("limit") || "15", 10);
  const offset = parseInt(req.nextUrl.searchParams.get("offset") || "0", 10);
  const sort = (req.nextUrl.searchParams.get("sort") || "relevance") as any;
  const yearFrom = req.nextUrl.searchParams.get("yearFrom")
    ? parseInt(req.nextUrl.searchParams.get("yearFrom")!, 10)
    : undefined;
  const yearTo = req.nextUrl.searchParams.get("yearTo")
    ? parseInt(req.nextUrl.searchParams.get("yearTo")!, 10)
    : undefined;

  if (!query || query.trim().length < 2) {
    return badRequest("Query parameter 'q' (min 2 chars) is required");
  }
  try {
    const result = await searchPubMed({ query, limit, offset, sort, yearFrom, yearTo });
    recordApiRequestForUser(guard.user);
    // FE-006: map `articles` → `items`. Keep `total`, `limit`, `offset`.
    return NextResponse.json({
      items: result.articles,
      total: result.total,
      limit,
      offset,
    });
  } catch (e: unknown) {
    // FE-063: never use `e: any` — narrow with instanceof.
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`PubMed search failed: ${msg}`);
  }
}
