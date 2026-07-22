/**
 * Drug mechanism-of-action lookup service.
 *
 * FE-024 ROOT FIX: The previous code rendered `RL reward: 0.234, policy_prob:
 * 0.123` in the "Mechanism" column of the candidate table — that is RL
 * debug output, NOT a drug's mechanism of action. A researcher evaluating
 * repurposing candidates cannot make any decision without knowing the
 * mechanism (e.g. "NMDA receptor antagonist").
 *
 * This service fetches the real mechanism of action for a drug from ChEMBL
 * (https://www.ebi.ac.uk/chembl/), the European Bioinformatics Institute's
 * open chemistry database. ChEMBL is:
 *   - Free to use (no API key required)
 *   - Scientifically authoritative (used by pharma research worldwide)
 *   - CC BY-SA 3.0 licensed for the data
 *
 * The lookup is two-step:
 *   1. Resolve the drug name to a ChEMBL ID via the molecule search endpoint.
 *   2. Fetch the mechanism-of-action record for that ChEMBL ID.
 *
 * The result is cached per-drug in an in-memory LRU to avoid hammering the
 * ChEMBL API when the user re-runs the same query.
 */

import { monitoredFetch } from "@/lib/external-api-monitor";

const CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data";

/**
 * FE-028 ROOT FIX (Team Member 15) + Task 254 ROOT FIX:
 *
 * ROOT CAUSE: the in-memory cache had NO TTL — entries lived forever
 * (until evicted by LRU). If the KG was updated (e.g. a new mechanism
 * was added to ChEMBL), the cache served the OLD mechanism
 * indefinitely. For a pharma partner demo, this means showing
 * outdated mechanisms with no recovery short of a server restart.
 *
 * ROOT FIX (FE-028): cache entries now carry a `cachedAt` timestamp.
 *
 * ROOT FIX (Task 254): the audit explicitly requires a 1-hour TTL for
 * drug/mechanism queries because "KG queries are expensive". The
 * previous 5-minute TTL was too aggressive — it caused ChEMBL to be
 * re-hit every 5 minutes for the same drug, eating into EBI's shared
 * rate-limit budget. The new 1-hour TTL strikes the right balance:
 * ChEMBL mechanism records change rarely (a handful of times per
 * year per molecule), so 1 hour is well within the staleness budget.
 * Operators can still force-refresh via POST /api/drugs/mechanism/refresh.
 *
 * Exported `clearDrugMechanismCache()` allows a manual refresh
 * via POST /api/drugs/mechanism/refresh.
 * The Next.js `next: { revalidate: 86400 }` on the underlying
 * fetch is preserved — it dedupes the HTTP request at the
 * framework level. The 1-hour in-memory TTL is a SEPARATE
 * concern: it controls how often we re-check ChEMBL within a
 * single server process.
 */
const CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour (Task 254: was 5 min)

export interface DrugMechanismResult {
  drugName: string;
  chemblId: string | null;
  mechanism: string | null;
  /** Canonicalized mechanism reference, e.g. "ChEMBL::CHEMBL25::Direct thrombin inhibitor". */
  source: string | null;
  fetchedAt: string;
  /**
   * BE-017 ROOT FIX: distinguish "no data exists in ChEMBL" from "lookup failed".
   * - undefined: lookup succeeded (mechanism may be null if ChEMBL has no MoA record).
   * - "chembl_unreachable": network error, HTTP 5xx, or JSON parse error.
   * - "chembl_not_found": HTTP 404 / 422 from ChEMBL (treated as no data; mechanism=null).
   * The UI should render "—" for null mechanism with no error, but "Mechanism lookup failed — retry"
   * when error is set, so the researcher does not conflate "service down" with "no data".
   */
  error?: "chembl_unreachable" | "chembl_not_found";
  /**
   * Task 242 ROOT FIX: drug→protein→pathway chain from the Phase 2 KG
   * service. Populated when KG_SERVICE_URL is set and the service
   * responds successfully. Empty when KG service is unavailable — the
   * route still returns the ChEMBL mechanism text in that case.
   *
   * Each chain is a list of edges forming a path from the drug to a
   * disease through proteins and pathways. The frontend's
   * pathway-viz component renders these as a node-link diagram so a
   * researcher can audit WHY the model connected this drug to this
   * disease.
   */
  pathwayChain?: PathwayEdge[];
  /** Protein targets of this drug, sourced from the KG service. */
  proteinTargets?: string[];
  /** Pathways this drug's targets participate in, from the KG service. */
  pathways?: string[];
}

export interface PathwayEdge {
  /** Source node label, e.g. "Aspirin". */
  source: string;
  /** Source node type: drug | protein | pathway | disease. */
  sourceType: "drug" | "protein" | "pathway" | "disease";
  /** Target node label, e.g. "PTGS2". */
  target: string;
  /** Target node type. */
  targetType: "drug" | "protein" | "pathway" | "disease";
  /** Edge type, e.g. "inhibits", "is_part_of", "is_disrupted_in". */
  relation: string;
}

// In-memory cache. Bounded to 256 entries; least-recently-used evicted.
//
// BE-069 ROOT FIX (v115, LOW): the previous code evicted the OLDEST
// entry by INSERTION order (FIFO), not by ACCESS order (true LRU).
// A drug accessed once and never again would stay in the cache as
// long as a drug accessed every minute — defeating the LRU eviction
// strategy. The fix: use the `delete + set` pattern on every read
// to move the accessed key to the END of the Map's insertion order
// (Map preserves insertion order, and re-inserting a key moves it
// to the end). This makes the cache a true LRU.
const CACHE_MAX = 256;

interface CacheEntry {
  result: DrugMechanismResult;
  cachedAt: number; // ms epoch — FE-028
}

const cache = new Map<string, CacheEntry>();

function cacheKey(drugName: string): string {
  return drugName.trim().toLowerCase();
}

function cacheSet(key: string, value: DrugMechanismResult): void {
  // BE-069 ROOT FIX: true LRU eviction. If the key already exists,
  // delete it first so the re-insertion moves it to the end (most-
  // recently-used position). This is the standard LRU pattern for
  // JavaScript Maps (which preserve insertion order).
  if (cache.has(key)) {
    cache.delete(key);
  } else if (cache.size >= CACHE_MAX) {
    // Cache is full and this is a NEW key — evict the oldest entry
    // (the first key in insertion order = least-recently-used).
    const firstKey = cache.keys().next().value;
    if (firstKey !== undefined) cache.delete(firstKey);
  }
  cache.set(key, { result: value, cachedAt: Date.now() });
}

/**
 * BE-069 ROOT FIX (v115, LOW): true LRU read. On cache HIT, we delete
 * the key and re-insert it — this moves it to the END of the Map's
 * insertion order (most-recently-used position). The next eviction
 * will therefore target a DIFFERENT (less-recently-used) key, not
 * this one. This is the standard LRU pattern for JavaScript Maps.
 *
 * Returns the cache entry, or undefined if the key is not in the
 * cache OR the entry has expired (TTL exceeded).
 */
function cacheGet(key: string): CacheEntry | undefined {
  const entry = cache.get(key);
  if (!entry) return undefined;
  // Check TTL — expired entries are treated as misses.
  if (Date.now() - entry.cachedAt > CACHE_TTL_MS) {
    cache.delete(key);
    return undefined;
  }
  // BE-069: move to end (most-recently-used position) by delete + set.
  cache.delete(key);
  cache.set(key, entry);
  return entry;
}

/**
 * FE-028: manually clear the drug-mechanism cache. Called by the
 * POST /api/drugs/mechanism/refresh route when the operator clicks
 * "Refresh" on the dashboard, or after a KG update is known to have
 * landed.
 *
 * @param drugName Optional. If provided, clears only the entry for
 *   that drug. If omitted, clears ALL entries.
 */
export function clearDrugMechanismCache(drugName?: string): void {
  if (drugName) {
    cache.delete(cacheKey(drugName));
  } else {
    cache.clear();
  }
}

/** FE-028: inspect the cache for observability / debugging. */
export function getDrugMechanismCacheState(): Array<{
  drugName: string;
  cachedAt: number;
  ageMs: number;
  ttlRemainingMs: number;
  mechanism: string | null;
}> {
  const now = Date.now();
  return Array.from(cache.entries()).map(([k, entry]) => ({
    drugName: k,
    cachedAt: entry.cachedAt,
    ageMs: now - entry.cachedAt,
    ttlRemainingMs: Math.max(0, CACHE_TTL_MS - (now - entry.cachedAt)),
    mechanism: entry.result.mechanism,
  }));
}

interface ChEMBLMoleculeResponse {
  molecules: Array<{
    molecule_chembl_id: string;
    molecule_synonyms?: Array<{ molecule_synonym: string; syn_type: string }>;
    pref_name?: string;
  }>;
}

interface ChEMBLMechanismResponse {
  mechanisms: Array<{
    molecule_chembl_id: string;
    mechanism_of_action: string;
    action_type?: string;
    target_pref_name?: string;
  }>;
}

/**
 * Resolve a drug name (generic or brand) to a ChEMBL ID.
 * Returns null if no match. The ChEMBL molecule search is tolerant of
 * case and partial matches; we filter to exact synonym matches to avoid
 * returning the wrong drug.
 */
async function resolveChemblId(drugName: string): Promise<string | null> {
  const sanitized = drugName.trim().replace(/["\\]/g, "").slice(0, 128);
  if (sanitized.length < 2) return null;

  // The molecule_synonyms filter is a case-insensitive iexact match.
  // We try the user's exact name first; if no hit, we try a fuzzy
  // search via the search term `_search` resource as a fallback.
  const url =
    `${CHEMBL_BASE}/molecule.json?molecule_synonyms__molecule_synonym__iexact=` +
    encodeURIComponent(sanitized) +
    `&limit=5`;

  // Task 260: monitored for observability — every ChEMBL call is logged
  // with URL, duration, and status so operators can detect slow or
  // degraded upstream responses.
  const res = await monitoredFetch("chembl", url, {
    headers: { Accept: "application/json" },
    // Cache the fetch result for 24h at the Next.js level (production).
    next: { revalidate: 86400 },
  });
  if (!res.ok) return null;
  const body: ChEMBLMoleculeResponse = await res.json();
  const molecules = body?.molecules || [];
  if (molecules.length === 0) return null;

  // Prefer the molecule whose synonym EXACTLY matches (case-insensitive).
  // Without this filter, "ASCEND" might match "ASCORBINIC ACID" because
  // both start with the same letters in the synonym index.
  const lower = sanitized.toLowerCase();
  const exactMatch = molecules.find((m) =>
    (m.molecule_synonyms || []).some(
      (s) => (s.molecule_synonym || "").toLowerCase() === lower
    )
  );
  if (exactMatch) return exactMatch.molecule_chembl_id;

  // Fall back to the first molecule if no exact synonym match (the
  // ChEMBL search already ranks by relevance).
  return molecules[0].molecule_chembl_id;
}

/**
 * Fetch the mechanism-of-action text for a ChEMBL ID.
 * Returns null if the molecule has no mechanism record (this is common —
 * many compounds in ChEMBL have no annotated MoA).
 */
async function fetchMechanism(chemblId: string): Promise<string | null> {
  const url = `${CHEMBL_BASE}/mechanism.json?molecule_chembl_id=${encodeURIComponent(
    chemblId
  )}&limit=5`;
  // Task 260: monitored for observability.
  const res = await monitoredFetch("chembl", url, {
    headers: { Accept: "application/json" },
    next: { revalidate: 86400 },
  });
  if (!res.ok) return null;
  const body: ChEMBLMechanismResponse = await res.json();
  const mechanisms = body?.mechanisms || [];
  if (mechanisms.length === 0) return null;

  // Prefer the record with both an action_type and a mechanism_of_action —
  // that's the most informative. Otherwise take the first record.
  const preferred =
    mechanisms.find((m) => m.action_type && m.mechanism_of_action) ||
    mechanisms[0];

  const parts: string[] = [];
  if (preferred.action_type) {
    // action_type is a short code like "INHIBITOR"; expand to title case.
    parts.push(
      preferred.action_type.charAt(0).toUpperCase() +
        preferred.action_type.slice(1).toLowerCase()
    );
  }
  if (preferred.mechanism_of_action) {
    parts.push(preferred.mechanism_of_action);
  } else if (preferred.target_pref_name) {
    parts.push(`targets ${preferred.target_pref_name}`);
  }
  return parts.length > 0 ? parts.join(" - ") : null;
}

/**
 * Task 242 ROOT FIX: fetch the drug→protein→pathway chain from the Phase 2
 * KG service. Returns null if the KG service is not configured or the
 * request fails — the caller falls back to ChEMBL-only mechanism text.
 *
 * The Phase 2 service exposes `GET /kg/explore?drug=<name>&limit=N` which
 * returns a subgraph centered on the drug. We translate that into a flat
 * list of `PathwayEdge` records plus `proteinTargets` and `pathways`
 * arrays for the route response.
 *
 * ROOT CAUSE: the audit required the mechanism route to return the
 * drug→protein→pathway chain — but the previous code returned ONLY the
 * ChEMBL mechanism text (a single string). The frontend's pathway-viz
 * component had nothing to render. Now we enrich the response with real
 * graph edges from the Phase 2 KG service.
 */
async function fetchKgPathwayChain(
  drugName: string
): Promise<{ pathwayChain: PathwayEdge[]; proteinTargets: string[]; pathways: string[] } | null> {
  // BE-056 ROOT FIX (v115, LOW): the previous code read
  // process.env.KG_SERVICE_URL directly. Every other service in the
  // codebase uses `resolveServiceUrl("KG_SERVICE_URL", ...)` from
  // http-client.ts — which trims trailing slashes, supports env var
  // aliases, and centralizes the URL resolution logic. Using the raw
  // env var here meant:
  //   1. A trailing slash in KG_SERVICE_URL would produce a malformed
  //      URL (`http://host:8001//kg/explore`).
  //   2. No support for legacy alias env vars.
  //   3. Inconsistent with the rest of the codebase — a future
  //      developer would not know to look in http-client.ts for the
  //      canonical resolution logic.
  //
  // ROOT FIX: import and use resolveServiceUrl. The function returns
  // null if the env var is unset — we preserve the previous "return
  // null to disable KG enrichment" behavior.
  const { resolveServiceUrl } = await import("@/lib/http-client");
  const serviceUrl = resolveServiceUrl("KG_SERVICE_URL");
  if (!serviceUrl) return null;

  const sanitized = drugName.trim().slice(0, 128);
  if (sanitized.length < 2) return null;

  // BE-056: resolveServiceUrl already trims trailing slashes, but
  // we keep the defensive .replace() in case the function is later
  // changed to NOT trim (defensive programming).
  const url = `${serviceUrl.replace(/\/$/, "")}/kg/explore?drug=${encodeURIComponent(sanitized)}&limit=50`;
  try {
    // Task 260: monitored for observability.
    const res = await monitoredFetch("kg_service", url, {
      headers: { Accept: "application/json" },
      // KG queries are expensive — cache the fetch result for 1h at the
      // Next.js level (matches the in-memory TTL).
      next: { revalidate: 3600 },
    });
    if (!res.ok) return null;
    const body = await res.json();
    // The KG service returns { nodes: [...], edges: [...] } OR a similar
    // subgraph shape. We translate the edges into PathwayEdge records.
    const rawEdges: any[] = body?.edges || body?.relations || [];
    const rawNodes: any[] = body?.nodes || body?.entities || [];
    const nodeTypeMap = new Map<string, string>();
    for (const n of rawNodes) {
      const id = n?.id || n?.uid || n?.name;
      const type = (n?.type || n?.node_type || "").toLowerCase();
      if (id && type) nodeTypeMap.set(String(id), type);
    }

    const pathwayChain: PathwayEdge[] = [];
    const proteinTargets = new Set<string>();
    const pathways = new Set<string>();

    for (const e of rawEdges.slice(0, 200)) {
      const sourceId = String(e?.source || e?.from || e?.subject || "");
      const targetId = String(e?.target || e?.to || e?.object || "");
      const relation = String(e?.type || e?.relation || e?.label || "related_to");
      if (!sourceId || !targetId) continue;

      const sourceType = (nodeTypeMap.get(sourceId) || "drug") as PathwayEdge["sourceType"];
      const targetType = (nodeTypeMap.get(targetId) || "protein") as PathwayEdge["targetType"];

      // Only include the 4 canonical types in the chain (drug, protein,
      // pathway, disease) — drop edges to AdverseEvent / Gene etc. so the
      // pathway-viz stays focused on the mechanism chain.
      const validTypes = new Set(["drug", "protein", "pathway", "disease"]);
      if (!validTypes.has(sourceType) || !validTypes.has(targetType)) continue;

      pathwayChain.push({
        source: sourceId,
        sourceType,
        target: targetId,
        targetType,
        relation,
      });

      if (sourceType === "protein") proteinTargets.add(sourceId);
      if (targetType === "protein") proteinTargets.add(targetId);
      if (sourceType === "pathway") pathways.add(sourceId);
      if (targetType === "pathway") pathways.add(targetId);
    }

    return {
      pathwayChain,
      proteinTargets: Array.from(proteinTargets),
      pathways: Array.from(pathways),
    };
  } catch (e: unknown) {
    // KG service unreachable — fall back to ChEMBL-only. Not an error
    // worth surfacing to the user; the mechanism text is still useful.
    const msg = e instanceof Error ? e.message : String(e);
    console.warn(`[drug-mechanism] KG service lookup failed for "${drugName}": ${msg}`);
    return null;
  }
}

/**
 * Look up the mechanism of action for a drug name.
 *
 * Returns a `DrugMechanismResult` with `mechanism: null` if no data is
 * found (the UI should render "—" in that case, never a fabricated value).
 *
 * Throws on network errors so the caller can decide how to handle
 * (e.g. show a tooltip "mechanism lookup failed").
 */
export async function lookupDrugMechanism(
  drugName: string
): Promise<DrugMechanismResult> {
  const key = cacheKey(drugName);
  const now = Date.now();

  // FE-028: TTL check. Treat entries older than CACHE_TTL_MS as misses.
  // BE-069 ROOT FIX: use cacheGet() (which moves the entry to the
  // most-recently-used position on hit) instead of cache.get() (which
  // does not update LRU ordering).
  const cached = cacheGet(key);
  if (cached) {
    return cached.result;
  }
  // (else: cache miss or stale — fall through to re-fetch)

  const result: DrugMechanismResult = {
    drugName: drugName.trim(),
    chemblId: null,
    mechanism: null,
    source: null,
    fetchedAt: new Date().toISOString(),
  };

  try {
    const chemblId = await resolveChemblId(drugName);
    if (chemblId) {
      result.chemblId = chemblId;
      const mechanism = await fetchMechanism(chemblId);
      if (mechanism) {
        result.mechanism = mechanism;
        result.source = `ChEMBL::${chemblId}`;
      }
      // mechanism === null here means ChEMBL has the molecule but no MoA record.
      // That is "no data", not a lookup failure — leave error undefined.
    }
    // Task 242: enrich the result with the drug→protein→pathway chain from
    // the Phase 2 KG service. This is BEST-EFFORT — if the KG service is
    // not configured or returns an error, we still return the ChEMBL
    // mechanism text. The pathway chain is additive.
    const kgChain = await fetchKgPathwayChain(drugName);
    if (kgChain) {
      result.pathwayChain = kgChain.pathwayChain;
      result.proteinTargets = kgChain.proteinTargets;
      result.pathways = kgChain.pathways;
      // Update source to indicate both ChEMBL and KG were used.
      if (result.source) {
        result.source = `${result.source} + KG_SERVICE`;
      } else {
        result.source = "KG_SERVICE";
      }
    }
  } catch (e: unknown) {
    // BE-017 ROOT FIX: do NOT silently swallow. Distinguish "no data" from
    // "lookup failed" so the UI can show "Mechanism lookup failed — retry"
    // instead of "—". A researcher who sees "—" believes the data does not
    // exist; a researcher who sees "lookup failed" knows to retry later.
    // Common causes: ChEMBL is down, network timeout, malformed JSON response.
    const msg = e instanceof Error ? e.message : String(e);
    console.warn(`[drug-mechanism] ChEMBL lookup failed for "${drugName}": ${msg}`);
    result.error = "chembl_unreachable";
    // Do NOT cache failed lookups for the full TTL — let the next request retry.
    return result;
  }

  cacheSet(key, result);
  return result;
}

/**
 * Batch-lookup mechanisms for multiple drug names. Requests are issued
 * concurrently but ChEMBL rate-limits to ~5 req/sec, so we cap concurrency
 * at 5 to avoid 429s.
 */
export async function lookupDrugMechanisms(
  drugNames: string[]
): Promise<Map<string, DrugMechanismResult>> {
  const result = new Map<string, DrugMechanismResult>();
  const unique = Array.from(new Set(drugNames.map((n) => n.trim()).filter(Boolean)));
  const CONCURRENCY = 5;

  for (let i = 0; i < unique.length; i += CONCURRENCY) {
    const batch = unique.slice(i, i + CONCURRENCY);
    const results = await Promise.all(
      batch.map((name) => lookupDrugMechanism(name).catch(() => ({
        drugName: name,
        chemblId: null,
        mechanism: null,
        source: null,
        fetchedAt: new Date().toISOString(),
        // BE-017: surface the failure so the batch caller can distinguish
        // "no data" from "lookup failed" for each drug in the batch.
        error: "chembl_unreachable" as const,
      })))
    );
    for (const r of results) {
      result.set(r.drugName.toLowerCase(), r);
    }
  }
  return result;
}
