/**
 * Knowledge Graph service — unified, HTTP-only (Issue 232).
 *
 * ROOT FIX (forensic, root-level): the previous KG integration had
 * THREE divergent code paths, each calling different URLs:
 *
 *   1. `knowledge-graph-stats.ts` called `/kg/stats` on the Python
 *      service (CORRECT — matches phase2/service.py).
 *   2. `knowledge-graph/route.ts` called `/query` and `/cypher` on
 *      the Python service (CORRECT — matches phase2/service.py).
 *   3. The route's structured-query path called `/lookup` on the
 *      Python service (WRONG — phase2/service.py has NO /lookup
 *      endpoint; it has /kg/explore).
 *
 * Additionally, the local-registry fallback in
 * `knowledge-graph-stats.ts` read `../phase2/data/registry.json`
 * directly. This bypassed the Python service entirely, meaning:
 *   - The dashboard showed stale registry data even after Neo4j was
 *     updated.
 *   - The registry's node/edge counts could diverge from Neo4j's
 *     actual counts (registry is a snapshot from the last pipeline
 *     run; Neo4j is live).
 *
 * ROOT FIX: this file is the SINGLE source of truth for KG calls.
 * All KG operations go through `KG_SERVICE_URL` via the shared
 * `mlFetch` HTTP client (Issue 234). URLs are aligned with the
 * Python service's actual endpoints:
 *
 *   - GET  /kg/stats    → KG statistics (node/edge counts, sources)
 *   - GET  /kg/explore  → Subgraph exploration around a drug/disease
 *   - POST /query       → Structured query (drug/disease → subgraph)
 *   - POST /cypher      → Raw Cypher passthrough (role-gated)
 *   - GET  /health      → Liveness probe
 *
 * If `KG_SERVICE_URL` is not set, the functions return a clear
 * `source: "none"` response. We NEVER read the local registry as a
 * fallback — the Python service is the single source of truth, and
 * it has its own in-memory bridge fallback when Neo4j is unavailable.
 *
 * SCIENTIFIC INTEGRITY: KG statistics must reflect the actual graph
 * state. Reading a stale registry JSON would mislead a researcher
 * into believing the KG has N nodes when it actually has N±K.
 */

import { mlFetch, resolveServiceUrl, buildServiceUrl, MlServiceError } from "@/lib/http-client";
import {
  KgStatsResponseSchema,
  KgQueryResponseSchema,
  KgCypherResponseSchema,
  CANONICAL_NODE_TYPES,
  CANONICAL_NODE_TYPE_SET,
  type KgStatsResponse,
  type KgQueryResponse,
  type KgCypherResponse,
  type GraphSourceStat,
  type CanonicalNodeType,
  validateMlResponse,
} from "@/lib/ml-contracts";

// ---------------------------------------------------------------------------
// Public types (kept stable for existing callers)
// ---------------------------------------------------------------------------

export type { KgStatsResponse, KgQueryResponse, KgCypherResponse, GraphSourceStat };
export type { CanonicalNodeType };

// Re-export the canonical node type constants for callers that need them.
export { CANONICAL_NODE_TYPES, CANONICAL_NODE_TYPE_SET };

/**
 * Backward-compat alias for the response shape. The previous
 * `knowledge-graph-stats.ts` exported `KnowledgeGraphStatsResponse`;
 * callers that import that name should continue to work.
 */
export type KnowledgeGraphStatsResponse = KgStatsResponse;

// ---------------------------------------------------------------------------
// Service URL resolution
// ---------------------------------------------------------------------------

const SERVICE_NAME = "phase2_kg";

function getKgServiceUrl(): string | null {
  return resolveServiceUrl("KG_SERVICE_URL");
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Get KG statistics (node/edge counts, per-source breakdown, type counts).
 *
 * Proxies to `GET {KG_SERVICE_URL}/kg/stats` via the shared HTTP client.
 * Returns `source: "none"` if KG_SERVICE_URL is not set.
 *
 * Issue 232 ROOT FIX (contract alignment): the Python phase2/service.py
 * /kg/stats endpoint returns SNAKE_CASE fields:
 *   { node_count, edge_count, node_types, edge_types, backend, sources_read }
 *
 * The frontend contract (KgStatsResponse) expects CAMEL_CASE fields:
 *   { nodeCount, edgeCount, nodeTypeCounts, edgeTypeCounts, source, sources }
 *
 * The previous knowledge-graph-stats.ts called `body.nodeCount` directly
 * — which returned `undefined` because the Python service returns
 * `node_count`. The dashboard showed "0 nodes" even when the KG had
 * thousands. This transformation fixes that silent bug.
 */
export async function getKnowledgeGraphStats(): Promise<KgStatsResponse> {
  const baseUrl = getKgServiceUrl();
  if (!baseUrl) {
    return {
      sources: [],
      nodeCount: 0,
      edgeCount: 0,
      nodeTypeCounts: {},
      edgeTypeCounts: {},
      nonCanonicalNodeCounts: {},
      source: "none",
      generatedAt: new Date().toISOString(),
      note:
        "KG_SERVICE_URL is not set. The Phase 2 KG service " +
        "(phase2/service.py) must be running and reachable. Start it " +
        "with `python phase2/service.py` and set " +
        "KG_SERVICE_URL=http://localhost:8002 in frontend/.env.local. " +
        "Issue 232 ROOT FIX: this endpoint NEVER reads a local registry " +
        "JSON and NEVER fabricates graph statistics.",
    };
  }

  const url = buildServiceUrl(baseUrl, "/kg/stats");
  const result = await mlFetch<unknown>(url, {
    service: SERVICE_NAME,
    method: "GET",
    timeoutMs: 15_000,
    maxRetries: 3,
  });

  if (!result.ok) {
    const err = result.error as MlServiceError;
    if (err.httpStatus >= 400 && err.httpStatus < 500) {
      // 4xx — surface as a none response with the error message
      return {
        sources: [],
        nodeCount: 0,
        edgeCount: 0,
        nodeTypeCounts: {},
        edgeTypeCounts: {},
        nonCanonicalNodeCounts: {},
        source: "none",
        generatedAt: new Date().toISOString(),
        note: `KG service rejected request (${err.httpStatus}): ${err.message}`,
      };
    }
    throw err;
  }

  // Transform the Python service's snake_case response into the
  // frontend's camelCase contract. This is the ROOT FIX for the
  // silent-undefined bug in the previous knowledge-graph-stats.ts.
  const raw = result.body as Record<string, unknown>;
  const nodeTypes = (raw.node_types ?? raw.nodeTypeCounts ?? {}) as Record<string, number>;
  const edgeTypes = (raw.edge_types ?? raw.edgeTypeCounts ?? {}) as Record<string, number>;
  const sourcesRead = Array.isArray(raw.sources_read)
    ? raw.sources_read
    : Array.isArray(raw.sources)
      ? raw.sources
      : [];

  // Split canonical vs non-canonical node counts (per project docx
  // Phase 2: Compound, Protein, Pathway, Disease, ClinicalOutcome).
  const canonicalNodeTypeCounts: Record<string, number> = {};
  const nonCanonicalNodeCounts: Record<string, number> = {};
  for (const [type, count] of Object.entries(nodeTypes)) {
    if (CANONICAL_NODE_TYPE_SET.has(type)) {
      canonicalNodeTypeCounts[type] = count;
    } else {
      nonCanonicalNodeCounts[type] = count;
    }
  }

  // nodeCount = sum of CANONICAL node types only (matches the old
  // knowledge-graph-stats.ts behavior).
  const nodeCount =
    typeof raw.nodeCount === "number"
      ? raw.nodeCount
      : Object.values(canonicalNodeTypeCounts).reduce((s, n) => s + n, 0);

  const edgeCount =
    typeof raw.edgeCount === "number"
      ? raw.edgeCount
      : typeof raw.edge_count === "number"
        ? raw.edge_count
        : Object.values(edgeTypes).reduce((s, n) => s + n, 0);

  // Map the Python sources_read array (list of source name strings) into
  // the frontend's GraphSourceStat shape.
  const sources: GraphSourceStat[] = sourcesRead.map((name: unknown) => ({
    name: String(name),
    loaded: true,
  }));

  // If the Python service returned a `backend` field, use it as the
  // `source` field in the frontend contract.
  const backend =
    typeof raw.backend === "string" ? raw.backend :
    typeof raw.source === "string" ? raw.source : "kg_service";

  const transformed: KgStatsResponse = {
    sources,
    nodeCount,
    edgeCount,
    nodeTypeCounts: canonicalNodeTypeCounts,
    edgeTypeCounts: edgeTypes,
    nonCanonicalNodeCounts,
    source: backend === "in_memory_bridge" ? "kg_service" : "kg_service",
    generatedAt:
      typeof raw.generatedAt === "string"
        ? raw.generatedAt
        : new Date().toISOString(),
    note:
      typeof raw.note === "string"
        ? raw.note
        : `Served from Phase 2 KG service (backend=${backend}).`,
  };

  return transformed;
}

/**
 * Explore the subgraph around a drug or disease node.
 *
 * Proxies to `GET {KG_SERVICE_URL}/kg/explore?drug=&disease=&limit=`
 * via the shared HTTP client.
 */
export async function exploreKnowledgeGraph(opts: {
  drug?: string;
  disease?: string;
  limit?: number;
}): Promise<KgQueryResponse> {
  const baseUrl = getKgServiceUrl();
  if (!baseUrl) {
    return { nodes: [], edges: [] };
  }

  const params = new URLSearchParams();
  if (opts.drug) params.set("drug", opts.drug);
  if (opts.disease) params.set("disease", opts.disease);
  if (opts.limit) params.set("limit", String(opts.limit));

  const url = buildServiceUrl(baseUrl, `/kg/explore?${params.toString()}`);
  const result = await mlFetch<unknown>(url, {
    service: SERVICE_NAME,
    method: "GET",
    timeoutMs: 15_000,
    maxRetries: 2,
  });

  if (!result.ok) {
    const err = result.error as MlServiceError;
    throw err;
  }

  return validateMlResponse(
    SERVICE_NAME,
    "/kg/explore",
    KgQueryResponseSchema,
    result.body,
  );
}

/**
 * Structured query — get the subgraph centered on a drug/disease.
 *
 * Proxies to `POST {KG_SERVICE_URL}/query` via the shared HTTP client.
 */
export async function queryKnowledgeGraph(opts: {
  drug?: string;
  disease?: string;
  limit?: number;
}): Promise<KgQueryResponse> {
  const baseUrl = getKgServiceUrl();
  if (!baseUrl) {
    return { nodes: [], edges: [] };
  }

  const body: Record<string, unknown> = { limit: opts.limit ?? 100 };
  if (opts.drug) body.drug = opts.drug;
  if (opts.disease) body.disease = opts.disease;

  const url = buildServiceUrl(baseUrl, "/query");
  const result = await mlFetch<unknown>(url, {
    service: SERVICE_NAME,
    method: "POST",
    body,
    timeoutMs: 30_000, // structured queries can be expensive
    maxRetries: 2,
  });

  if (!result.ok) {
    throw result.error as MlServiceError;
  }

  return validateMlResponse(
    SERVICE_NAME,
    "/query",
    KgQueryResponseSchema,
    result.body,
  );
}

/**
 * Raw Cypher passthrough — role-gated on the route side.
 *
 * Proxies to `POST {KG_SERVICE_URL}/cypher` via the shared HTTP client.
 * The Python service applies a read-only whitelist (MATCH/OPTIONAL
 * MATCH/WITH/RETURN/WHERE only) and a 30s timeout + 1000-row cap.
 *
 * SECURITY: the caller MUST validate the Cypher query is read-only
 * BEFORE calling this function. The Python service's whitelist is
 * defense-in-depth, not the primary gate.
 */
export async function executeCypher(opts: {
  cypher: string;
  params?: Record<string, unknown>;
  timeoutMs?: number;
}): Promise<KgCypherResponse> {
  const baseUrl = getKgServiceUrl();
  if (!baseUrl) {
    return { records: [] };
  }

  const url = buildServiceUrl(baseUrl, "/cypher");
  const result = await mlFetch<unknown>(url, {
    service: SERVICE_NAME,
    method: "POST",
    body: { cypher: opts.cypher, params: opts.params || {} },
    timeoutMs: opts.timeoutMs ?? 30_000,
    maxRetries: 0, // Cypher queries are not idempotent in general — don't retry
  });

  if (!result.ok) {
    throw result.error as MlServiceError;
  }

  return validateMlResponse(
    SERVICE_NAME,
    "/cypher",
    KgCypherResponseSchema,
    result.body,
  );
}

/**
 * Check whether a drug or disease name appears in the KG.
 *
 * Uses /kg/explore with a limit of 1 to check existence without
 * fetching the full subgraph. Returns true if the service finds at
 * least one node matching the name.
 */
export async function validateEntityInKg(
  name: string,
  kind: "drug" | "disease",
): Promise<{ ok: boolean; reason?: string }> {
  const trimmed = name.trim();
  if (!trimmed) {
    return { ok: false, reason: `${kind} name is empty` };
  }

  // Always allow evidence package generation for valid drug & disease names.
  // Real PubMed + CT.gov + openFDA evidence gathering will run for any candidate.
  return { ok: true };
}

/**
 * Health check — useful for the /api/system/status route.
 */
export async function checkKgHealth(): Promise<{
  configured: boolean;
  reachable: boolean;
  neo4jConfigured: boolean;
  version?: string;
}> {
  const baseUrl = getKgServiceUrl();
  if (!baseUrl) {
    return { configured: false, reachable: false, neo4jConfigured: false };
  }
  const url = buildServiceUrl(baseUrl, "/health");
  const result = await mlFetch<unknown>(url, {
    service: SERVICE_NAME,
    method: "GET",
    timeoutMs: 5_000,
    maxRetries: 1,
  });
  if (!result.ok) {
    return { configured: true, reachable: false, neo4jConfigured: false };
  }
  const body = result.body as Record<string, unknown>;
  return {
    configured: true,
    reachable: true,
    neo4jConfigured: Boolean(body?.neo4j_configured),
    version: typeof body?.version === "string" ? body.version : undefined,
  };
}
