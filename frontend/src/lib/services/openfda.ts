/**
 * openFDA service — real FDA adverse event data.
 *
 * Source: openFDA (https://open.fda.gov/) — U.S. Food & Drug Administration
 * Maintainer: FDA
 * License: Public domain (U.S. government work). OpenFDA data is updated nightly.
 *
 * The adverse event endpoint contains serious adverse event reports submitted
 * to the FDA Adverse Event Reporting System (FAERS). We use it to surface
 * real-world safety signals for a given drug.
 *
 * IMPORTANT: openFDA returns REPORTS, not causal evidence. A report listing
 * a drug and an adverse event does NOT mean the drug caused the event. We
 * always label this as "reported adverse events" in the UI and never imply
 * causation.
 */

import { monitoredFetch } from "@/lib/external-api-monitor";

const OPENFDA_BASE = "https://api.fda.gov";

/**
 * FE-024 ROOT FIX (Team Member 15):
 *
 * ROOT CAUSE: openFDA's free public tier is rate-limited at 240 req/min
 * SHARED across all unauthenticated callers. With 10 concurrent users
 * the effective per-user rate drops to 24 req/min — too slow for a
 * pharma partner demo. Registering an API key (free, via
 * https://open.fda.gov/api/reference/) raises the limit to 120,000
 * req/min per key. The previous code NEVER sent the `api_key` query
 * param even when `OPENFDA_API_KEY` was set, AND never warned when it
 * was missing — so operators had no signal that their demo would be
 * slow until it was slow.
 *
 * ROOT FIX:
 *   1. Append `&api_key=...` to every openFDA request when
 *      `OPENFDA_API_KEY` is set.
 *   2. Log a ONE-TIME warning when the key is missing (not per-request
 *      — that would spam the logs). The warning names the env var so
 *      operators know exactly what to set.
 *   3. Expose `isOpenfdaApiKeyConfigured()` for the `/api/system/status`
 *      endpoint to report the missing key as a degraded-service signal.
 */

let openfdaApiKeyWarned = false;

export function isOpenfdaApiKeyConfigured(): boolean {
  return Boolean(process.env.OPENFDA_API_KEY);
}

function warnIfApiKeyMissing(): void {
  if (openfdaApiKeyWarned) return;
  if (process.env.OPENFDA_API_KEY) return;
  openfdaApiKeyWarned = true;
  console.warn(
    "[openfda] OPENFDA_API_KEY is not set. Requests will use the shared " +
      "public rate limit (240 req/min across ALL unauthenticated callers). " +
      "Register a free key at https://open.fda.gov/api/reference/ and set " +
      "OPENFDA_API_KEY to raise the limit to 120,000 req/min."
  );
}

export interface AdverseEventReaction {
  term: string;
  count: number;
}

export interface DrugSafetySummary {
  brandName: string;
  genericName: string;
  totalReports: number;
  seriousReports: number;
  seriousReportsWithDeath: number;
  topReactions: AdverseEventReaction[];
  // What we display comes from real reports — but we must be explicit that
  // these are spontaneous reports and not proven causal events.
  disclaimer: string;
}

const SAFETY_DISCLAIMER =
  "Adverse event data is sourced from the FDA Adverse Event Reporting System " +
  "(FAERS) via openFDA. Reports are spontaneous and do not prove causation. " +
  "A report listing a drug and an event does not mean the drug caused the event.";

export async function getDrugSafetySummary(drugName: string): Promise<DrugSafetySummary | null> {
  const q = (drugName || "").trim();
  if (q.length < 2) return null;

  // FE-045 ROOT FIX: the previous "best-effort" sanitization (strip a few
  // special chars + boolean operators) was fragile because the openFDA
  // query language reserves far more than quotes/parens/AND/OR/NOT — it
  // also supports field qualifiers (`field:value`), wildcards (`*`),
  // fuzzy (`~`), range queries (`[a TO b]`), and more. An attacker
  // passing `q=patient.drug.openfda.generic_name:ibuprofen` would have
  // their value pass the old sanitizer (no quotes, parens, or boolean
  // words) and end up injected as a field qualifier inside the quoted
  // search string — which openFDA's parser may interpret as a literal
  // OR as a query directive depending on its escape handling.
  //
  // Root fix: instead of trying to escape a syntax we don't fully
  // control, we apply a STRICT WHITELIST. Drug names are alphanumerics,
  // spaces, hyphens, and apostrophes (e.g. "St John's Wort"). Anything
  // else is rejected up-front and we return null — the caller treats
  // null as "no data" and the UI shows "No safety data available".
  //
  // Max 64 chars: FDA generic/brand names are well under this limit
  // (longest generic name ~30 chars; longest brand name ~25 chars), so
  // this is a sane upper bound that also caps any pathological input.
  const WHITELIST = /^[A-Za-z0-9 \-']{2,64}$/;
  if (!WHITELIST.test(q)) return null;
  const sanitized = q.replace(/\s+/g, " ").trim();

  // openFDA uses the generic (non-proprietary) name in the
  // `patient.drug.openfda.generic_name` field. We do an exact
  // (case-insensitive) match on generic_name OR brand_name.
  //
  // FE-045: build the search expression with URLSearchParams so the
  // openFDA query syntax (`"`, `:`, etc.) is properly URL-encoded
  // by the standard library. The `+OR+` separator between the two
  // field clauses is a LITERAL openFDA query operator (not a URL
  // space) — URLSearchParams would encode `+` as `%2B`, which
  // openFDA's parser does NOT accept as the OR separator. So we
  // build the search value with `+OR+` as a literal, URL-encode the
  // value with `encodeURIComponent`, then convert the encoded `%2B`
  // back to literal `+` so the openFDA API sees the operator it
  // expects. This is the same final URL format the openFDA docs
  // publish: `search=field:"value"+OR+field:"value"`.
  //
  // BE-053 ROOT FIX (v115, LOW): the previous whitelist allowed
  // apostrophes (e.g. "St John's Wort"), but openFDA's Lucene query
  // parser treats apostrophes as STRING TERMINATORS. An attacker
  // passing drug="aspirin'-OR-'ibuprofen" would craft:
  //   patient.drug.openfda.generic_name:"aspirin'-OR-'ibuprofen"
  // Lucene would parse this as a multi-clause query and return
  // adverse events for aspirin OR ibuprofen — mixing two drugs' data.
  //
  // ROOT FIX: escape apostrophes inside the quoted value by doubling
  // them ("St John's Wort" → "St John''s Wort"). Lucene interprets
  // a doubled quote inside a quoted string as a literal quote. This
  // preserves legitimate drug names with apostrophes while preventing
  // query injection. (Hyphens don't need escaping in Lucene.)
  const escapedForLucene = sanitized.replace(/'/g, "''");
  const searchValue =
    `patient.drug.openfda.generic_name:"${escapedForLucene}"` +
    `+OR+` +
    `patient.drug.openfda.brand_name:"${escapedForLucene}"`;
  const params = new URLSearchParams({ limit: "100" });
  // FE-024: include api_key when configured — raises rate limit from
  // 240 req/min (shared public) to 120,000 req/min (per-key).
  if (process.env.OPENFDA_API_KEY) {
    params.set("api_key", process.env.OPENFDA_API_KEY);
  } else {
    warnIfApiKeyMissing();
  }
  const encodedSearch = encodeURIComponent(searchValue).replace(/%2B/g, "+");
  const url = `${OPENFDA_BASE}/drug/event.json?search=${encodedSearch}&${params.toString()}`;

  // Note: openFDA responses can exceed Next.js's 2MB fetch cache limit, so
  // we do not pass `next: { revalidate }` here — we always fetch fresh.
  // Task 260: monitored for observability — every openFDA call is logged
  // with URL, duration, and status so operators can detect slow or
  // degraded upstream responses.
  const res = await monitoredFetch("openfda", url, {
    headers: { Accept: "application/json" },
  });
  if (res.status === 404) {
    // openFDA returns 404 when there are zero matches — not an error.
    return {
      brandName: q,
      genericName: q,
      totalReports: 0,
      seriousReports: 0,
      seriousReportsWithDeath: 0,
      topReactions: [],
      disclaimer: SAFETY_DISCLAIMER,
    };
  }
  if (!res.ok) {
    throw new Error(`openFDA returned ${res.status}`);
  }
  const body = await res.json();
  const results: any[] = body?.results || [];
  if (results.length === 0) {
    return {
      brandName: q,
      genericName: q,
      totalReports: 0,
      seriousReports: 0,
      seriousReportsWithDeath: 0,
      topReactions: [],
      disclaimer: SAFETY_DISCLAIMER,
    };
  }

  let serious = 0;
  let seriousWithDeath = 0;
  const reactionCounts: Record<string, number> = {};
  for (const ev of results) {
    if (ev.serious === "1") {
      serious++;
      if (ev.seriousnessdeath === "1") seriousWithDeath++;
    }
    for (const r of ev.patient?.reaction || []) {
      // Prefer reactionmeddrapt Preferred Term text; fall back to meddra code.
      const term = r.reactionmeddrapt || r.termmeddra;
      if (term) {
        reactionCounts[term] = (reactionCounts[term] || 0) + 1;
      }
    }
  }
  const topReactions: AdverseEventReaction[] = Object.entries(reactionCounts)
    .map(([term, count]) => ({ term, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);

  // Pull display names from the first matching record
  const first = results[0];
  const openfda = first?.patient?.drug?.[0]?.openfda || {};
  return {
    brandName: (openfda.brand_name || [q])[0],
    genericName: (openfda.generic_name || [q])[0],
    totalReports: results.length,
    seriousReports: serious,
    seriousReportsWithDeath: seriousWithDeath,
    topReactions,
    disclaimer: SAFETY_DISCLAIMER,
  };
}
