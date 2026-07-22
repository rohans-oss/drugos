/**
 * ClinicalTrials.gov service — real, authoritative clinical trial data.
 *
 * Source: ClinicalTrials.gov v2 API (https://clinicaltrials.gov/api/v2/studies)
 * Maintainer: U.S. National Library of Medicine (NLM).
 * License: Public domain (U.S. government work).
 *
 * This is the same API that powers the public ClinicalTrials.gov website.
 * It is the single most authoritative source for clinical trial registrations
 * worldwide — every interventional trial of an FDA-regulated product is
 * legally required to be registered here.
 */

import { monitoredFetch } from "@/lib/external-api-monitor";

const CTGOV_BASE = "https://clinicaltrials.gov/api/v2";

export interface ClinicalTrial {
  nctId: string;
  title: string;
  status: string;
  phase: string;
  enrollment?: number;
  startDate?: string;
  completionDate?: string;
  sponsor?: string;
  conditions: string[];
  interventions: string[];
  studyType: string;
  url: string;
  briefSummary?: string;
  locations: string[];
}

export interface ClinicalTrialSearchResponse {
  total: number;
  trials: ClinicalTrial[];
  /**
   * FE-015 ROOT FIX: CT.gov v2 returns an opaque base64 cursor for the
   * next page. The caller MUST pass this back as `pageToken` (not an
   * integer offset) to fetch the next page. We expose it so the API
   * route can return it to the client for follow-up paginated requests.
   */
  nextPageToken?: string;
}

/**
 * Search clinical trials by disease/condition and optionally by intervention (drug).
 * Returns real, currently-registered trials from ClinicalTrials.gov.
 *
 * FE-015 ROOT FIX: The previous code did `urlParams.set("pageToken",
 * String(offset))` — passing a numeric offset like "20" as the pageToken.
 * CT.gov v2's pageToken is an opaque base64-encoded cursor returned by
 * the PREVIOUS response, NOT a numeric offset. Passing String(offset)
 * returns a 400 error or unexpected results. Pagination beyond page 1
 * was completely broken.
 *
 * Root fix: this function now accepts an opaque `pageToken` cursor (from
 * the previous response) and returns `nextPageToken` in the response.
 * Numeric offset is no longer supported because CT.gov v2 is cursor-only.
 */
export async function searchClinicalTrials(params: {
  condition?: string;
  intervention?: string;
  status?: "RECRUITING" | "ACTIVE_NOT_RECRUITING" | "COMPLETED" | "ALL";
  phase?: string;
  limit?: number;
  pageToken?: string;
}): Promise<ClinicalTrialSearchResponse> {
  const limit = Math.min(params.limit ?? 20, 100);

  // Build the query expression the way ClinicalTrials.gov v2 expects.
  // Use the simpler query.cond and query.intr parameters when possible.
  //
  // FE-048 ROOT FIX: the previous code defined `escapeQuery` but never
  // called it — `urlParams.set("query.cond", params.condition.trim())`
  // URL-encoded the value (good for URL safety) but did NOT escape
  // CT.gov's query syntax. An attacker passing
  //   condition = "cancer AND (AREA[Phase]PHASE3)"
  // could inject boolean operators and field qualifiers into the
  // CT.gov query, manipulating result sets and potentially exfiltrating
  // trial data from queries they should not be able to construct.
  // The fix: pass cond/intr values through `escapeQuery`, which quotes
  // the value when it contains any character that CT.gov's parser
  // treats specially (whitespace, parens, quotes). Quoted values are
  // treated as literal phrases by CT.gov, defeating injection.
  const urlParams = new URLSearchParams();
  if (params.condition && params.condition.trim()) {
    urlParams.set("query.cond", escapeQuery(params.condition.trim()));
  }
  if (params.intervention && params.intervention.trim()) {
    urlParams.set("query.intr", escapeQuery(params.intervention.trim()));
  }
  if (!params.condition && !params.intervention) {
    urlParams.set("query", "*");
  }
  urlParams.set("pageSize", String(limit));
  // FE-015: pageToken is an opaque cursor returned by the previous
  // response. Pass it through verbatim — never synthesise one.
  if (params.pageToken && params.pageToken.trim()) {
    urlParams.set("pageToken", params.pageToken.trim());
  }
  urlParams.set("format", "json");
  urlParams.set(
    "fields",
    [
      "NCTId",
      "BriefTitle",
      "OverallStatus",
      "Phase",
      "EnrollmentCount",
      "StartDateStruct",
      "CompletionDateStruct",
      "LeadSponsorName",
      "ConditionSearch",
      "InterventionSearch",
      "StudyType",
      "BriefSummary",
      "LocationCity",
      "LocationCountry",
    ].join(",")
  );
  if (params.status && params.status !== "ALL") {
    const map: Record<string, string> = {
      RECRUITING: "RECRUITING",
      ACTIVE_NOT_RECRUITING: "ACTIVE_NOT_RECRUITING",
      COMPLETED: "COMPLETED",
    };
    if (map[params.status]) {
      urlParams.set("filter.overallStatus", map[params.status]);
    }
  }

  const url = `${CTGOV_BASE}/studies?${urlParams.toString()}`;
  // Task 260: monitored for observability — every CT.gov call is logged
  // with URL, duration, and status so operators can detect slow or
  // degraded upstream responses.
  const res = await monitoredFetch("ctgov", url, {
    headers: { Accept: "application/json" },
    next: { revalidate: 3600 }, // 1 hour cache
  });
  if (!res.ok) {
    throw new Error(`ClinicalTrials.gov returned ${res.status}`);
  }
  const body = await res.json();
  const studies = (body?.studies || []) as any[];
  const trials: ClinicalTrial[] = studies.map((s) => normalizeTrial(s.protocolSection || s));
  return {
    total: body?.totalCount ?? trials.length,
    trials,
    // FE-015: expose the opaque cursor for the next page. The caller passes
    // this back as pageToken on the next request. When this is absent, there
    // are no more results.
    nextPageToken: body?.nextPageToken || undefined,
  };
}

function normalizeTrial(p: any): ClinicalTrial {
  const idModule = p.identificationModule || {};
  const statusModule = p.statusModule || {};
  const sponsorModule = p.sponsorCollaboratorsModule || {};
  const conditionsModule = p.conditionsModule || {};
  const designModule = p.designModule || {};
  const descriptionModule = p.descriptionModule || {};
  const locationsModule = p.contactsLocationsModule || {};

  const locations: string[] = [];
  const locList = locationsModule.locations || [];
  for (const loc of locList.slice(0, 5)) {
    const city = loc.city || "";
    const country = loc.country || "";
    if (city || country) locations.push([city, country].filter(Boolean).join(", "));
  }

  return {
    nctId: idModule.nctId || "",
    title: idModule.briefTitle || "",
    status: statusModule.overallStatus || "UNKNOWN",
    phase: designModule.phases?.join(", ") || "N/A",
    enrollment: designModule.enrollmentInfo?.count,
    startDate: statusModule.startDateStruct?.date,
    completionDate: statusModule.completionDateStruct?.date,
    sponsor: sponsorModule.leadSponsor?.name,
    conditions: conditionsModule.conditions || [],
    interventions: (p.armsInterventionsModule?.interventions || []).map((i: any) => i.name || i.type || "").filter(Boolean),
    studyType: designModule.studyType || "INTERVENTIONAL",
    url: idModule.nctId ? `https://clinicaltrials.gov/study/${idModule.nctId}` : "",
    briefSummary: descriptionModule.briefSummary?.textBlock?.text?.slice(0, 500),
    locations,
  };
}

/**
 * Escape a value for inclusion in a ClinicalTrials.gov v2 query expression.
 *
 * CT.gov's query parser treats whitespace, parens, and double-quotes as
 * syntax — so a value containing any of those must be wrapped in double
 * quotes (and any embedded double-quotes backslash-escaped) to be treated
 * as a literal phrase. This is the same convention used by Elasticsearch's
 * `query_string` parser, which CT.gov v2 uses under the hood.
 *
 * FE-048 ROOT FIX: this function was previously defined but never called
 * (dead code). It is now invoked for every `query.cond` and `query.intr`
 * value to prevent CT.gov query-syntax injection.
 */
export function escapeQuery(s: string): string {
  if (typeof s !== "string" || s.length === 0) return '""';
  if (/[\s()"]/.test(s)) {
    return `"${s.replace(/"/g, '\\"')}"`;
  }
  return s;
}
