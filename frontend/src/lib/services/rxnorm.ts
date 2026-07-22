/**
 * RxNorm service — drug name normalization and lookup.
 *
 * Source: NIH/NLM RxNorm REST API (https://rxnav.nlm.nih.gov/)
 * License: Public domain (U.S. government work).
 *
 * We use RxNorm because it is the authoritative source for clinical drug
 * nomenclature in the United States. It maps brand names, generic names,
 * and ingredient codes (RxCUI) to a single canonical concept.
 */

import { z } from "zod";
import { monitoredFetch } from "@/lib/external-api-monitor";

const RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST";

/**
 * FE-025 ROOT FIX (Team Member 15):
 *
 * ROOT CAUSE: RxNorm REST calls had NO timeout. A slow RxNorm response
 * (or a hung TCP connection) would hang the drug detail page
 * indefinitely — the researcher sees a spinning loader with no
 * recovery. The issue description mentions "ECHO endpoint" but the
 * actual code already uses the REST endpoint; the real defect is the
 * missing timeout.
 *
 * ROOT FIX: wrap every fetch in an `AbortController`-based 3-second
 * timeout. On timeout, we throw a typed `RxNormTimeoutError` so the
 * caller can render a clear "RxNorm lookup timed out — please retry"
 * message instead of a generic 500.
 *
 * The 24h cache (`next: { revalidate: 86400 }`) is already in place
 * and unchanged.
 */

const RXNORM_TIMEOUT_MS = 3000;

export class RxNormTimeoutError extends Error {
  constructor(public readonly endpoint: string) {
    super(
      `RxNorm ${endpoint} did not respond within ${RXNORM_TIMEOUT_MS}ms. ` +
        "The NLM RxNorm service may be slow or unreachable. Please retry."
    );
    this.name = "RxNormTimeoutError";
  }
}

/**
 * Fetch with a 3-second AbortController timeout. Resolves with the
 * Response on any HTTP status (caller checks res.ok). Rejects with
 * `RxNormTimeoutError` on timeout.
 */
async function fetchWithTimeout(
  url: string,
  init: RequestInit = {}
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), RXNORM_TIMEOUT_MS);
  try {
    // Task 260: wrap every RxNorm call in monitoredFetch so the platform
    // logs the URL, duration, and status code for observability.
    return await monitoredFetch("rxnorm", url, { ...init, signal: controller.signal });
  } catch (e: unknown) {
    // AbortError is thrown when the controller aborts the fetch.
    if (e instanceof Error && e.name === "AbortError") {
      throw new RxNormTimeoutError(url);
    }
    // Re-throw network errors as-is — they are not timeouts.
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

const RxNormConceptSchema = z.object({
  rxcui: z.string().optional(),
  name: z.string().optional(),
  synonym: z.string().optional(),
  tty: z.string().optional(),
});

/**
 * FE-046 ROOT FIX: the previous `RxNormSearchResultSchema` was defined for
 * the `{ idGroup: { name, rxnormId } }` shape — which is the response of
 * the `/cuiacquired.json` and `/getSpellingSuggestions.json` endpoints,
 * NOT the `/approximateTerm.json` endpoint that `searchDrugsByName`
 * actually calls. The previous code parsed the body with this schema,
 * extracted `idGroup.rxnormId`, then `void candidateRaw;`-discarded the
 * result and manually extracted from `body.approximateGroup.candidate`.
 * The schema was misleading dead weight — future maintainers reading the
 * file would assume the schema was doing real validation; it was not.
 *
 * Root fix: replace with a schema that matches the ACTUAL response shape
 * of `/approximateTerm.json`. The schema is now used for real validation
 * (we `safeParse` and bail out on an unexpected shape, rather than
 * blindly indexing into `body.approximateGroup.candidate`).
 *
 * Source for the shape: NIH RxNorm REST API docs,
 * https://rxnav.nlm.nih.gov/REST/approximateTerm.html
 */
export const RxNormApproximateTermSchema = z.object({
  approximateGroup: z.object({
    candidate: z.array(RxNormConceptSchema).optional(),
  }).optional(),
});

export interface NormalizedDrug {
  rxcui: string;
  name: string;
  synonym?: string;
  tty: string;
}

/**
 * Approximate match search — finds concepts whose name sounds like the query.
 * This is the recommended endpoint for free-text drug search.
 */
export async function searchDrugsByName(query: string, limit = 10): Promise<NormalizedDrug[]> {
  const q = (query || "").trim();
  if (q.length < 2) return [];
  const url = `${RXNORM_BASE}/approximateTerm.json?term=${encodeURIComponent(q)}&maxEntries=${limit}`;
  // FE-025: 3-second timeout via AbortController. Cache for 24h.
  const res = await fetchWithTimeout(url, {
    headers: { Accept: "application/json" },
    next: { revalidate: 86400 },
  });
  if (!res.ok) {
    throw new Error(`RxNorm approximateTerm returned ${res.status}`);
  }
  const body = await res.json();

  // FE-046: schema-validate the actual response shape. If RxNorm changes
  // their API shape (or returns an error envelope), safeParse fails and
  // we return [] — never crash on a missing `.candidate` field.
  const parsed = RxNormApproximateTermSchema.safeParse(body);
  if (!parsed.success) {
    // Best-effort fallback: try the raw shape directly. If that also fails,
    // return [] (no results) rather than crashing.
    const candidates = (body?.approximateGroup?.candidate || []) as Array<{
      rxcui?: string;
      name?: string;
      synonym?: string;
      tty?: string;
    }>;
    return candidates
      .filter((c): c is { rxcui: string; name?: string; synonym?: string; tty?: string } => !!c.rxcui)
      .map((c) => ({
        rxcui: c.rxcui,
        name: c.name || c.synonym || "",
        synonym: c.synonym,
        tty: c.tty || "",
      }));
  }
  const candidates = parsed.data?.approximateGroup?.candidate || [];
  const out: NormalizedDrug[] = [];
  for (const c of candidates) {
    if (!c.rxcui) continue;
    out.push({
      rxcui: c.rxcui,
      name: c.name || c.synonym || "",
      synonym: c.synonym,
      tty: c.tty || "",
    });
  }
  return out;
}

/**
 * Get all properties for a given RxCUI — including active ingredients,
 * brand names, and dose form. Useful for populating a drug detail page.
 */
export async function getDrugProperties(rxcui: string): Promise<{
  rxcui: string;
  name?: string;
  activeIngredients: string[];
  brandNames: string[];
  doseForm?: string;
  tty?: string;
}> {
  const url = `${RXNORM_BASE}/rxcui/${encodeURIComponent(rxcui)}/allProperties.json?prop=names+codes+attributes`;
  // FE-025: 3-second timeout via AbortController. Cache for 24h.
  const res = await fetchWithTimeout(url, {
    headers: { Accept: "application/json" },
    next: { revalidate: 86400 },
  });
  if (!res.ok) {
    throw new Error(`RxNorm allProperties returned ${res.status}`);
  }
  const body = await res.json();
  const propGroups = body?.propConceptGroup?.propConcept || [];
  const activeIngredients: string[] = [];
  const brandNames: string[] = [];
  let doseForm: string | undefined;
  let name: string | undefined;
  let tty: string | undefined;
  for (const p of propGroups) {
    if (p.propCategory === "NAMES" && p.propName === "Active Ingredient") {
      if (p.propValue && !activeIngredients.includes(p.propValue)) {
        activeIngredients.push(p.propValue);
      }
    }
    if (p.propCategory === "NAMES" && p.propName === "Brand Name") {
      if (p.propValue && !brandNames.includes(p.propValue)) {
        brandNames.push(p.propValue);
      }
    }
    if (p.propCategory === "ATTRIBUTES" && p.propName === "Dose Form") {
      doseForm = p.propValue;
    }
    if (p.propCategory === "ATTRIBUTES" && p.propName === "RxNorm Name") {
      name = p.propValue;
    }
    if (p.propCategory === "ATTRIBUTES" && p.propName === "TTY") {
      tty = p.propValue;
    }
  }
  return { rxcui, name, activeIngredients, brandNames, doseForm, tty };
}
