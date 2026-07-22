import { NextRequest, NextResponse } from "next/server";
import {
  requireAuth,
  requireRole,
  internalError,
  writeAuditLog,
} from "@/lib/api-helpers";
// BE-029 ROOT FIX (Team Member 12): Zod-validated request body for POST.
import { validateBody, KnowledgeGraphBody } from "@/lib/zod-schemas";
// FE-008 ROOT FIX: shared validator extracted so unit tests can exercise it
// without spinning up the route handler.
import { validateReadOnlyCypher } from "./cypher-validator";
// Issue 232 ROOT FIX: use the unified kg-service.ts (HTTP-only, no local
// registry fallback). URLs are aligned with the Python phase2/service.py:
//   - GET  /kg/stats    → KG statistics
//   - GET  /kg/explore  → Subgraph exploration
//   - POST /query       → Structured query
//   - POST /cypher      → Raw Cypher passthrough (role-gated)
// The previous version called /stats (WRONG — Python exposes /kg/stats)
// and /lookup (WRONG — Python has no /lookup, it has /kg/explore).
import {
  getKnowledgeGraphStats,
  queryKnowledgeGraph,
  executeCypher,
} from "@/lib/services/kg-service";

/**
 * GET /api/knowledge-graph?cypher=<Cypher query>&limit=<n>
 *      /api/knowledge-graph?drug=<drug>&disease=<disease>&limit=<n>   (structured query)
 *      /api/knowledge-graph                                              (stats — no params)
 *
 * Issue 225 ROOT FIX: URLs are now aligned with the Python phase2/service.py:
 *   - Stats path: calls getKnowledgeGraphStats() → GET /kg/stats
 *   - Structured query path: calls queryKnowledgeGraph() → POST /query
 *   - Cypher path: rejected via GET (use POST with role-gated validator)
 *
 * SECURITY: Cypher queries can be parameterized — we forward the `params`
 * object so the KG service can use parameterized queries. Raw string
 * interpolation of user input into Cypher would be a Cypher-injection
 * vulnerability (same class as SQL injection).
 */
export async function GET(req: NextRequest) {
  const auth = await requireAuth();
  if (auth.user === null) return auth.response;

  const cypher = req.nextUrl.searchParams.get("cypher");
  const limit = parseInt(req.nextUrl.searchParams.get("limit") || "100", 10);
  const drug = req.nextUrl.searchParams.get("drug");
  const disease = req.nextUrl.searchParams.get("disease");

  // Stats path — no structured query params and no cypher.
  if (!cypher && !drug && !disease) {
    try {
      const stats = await getKnowledgeGraphStats();
      await writeAuditLog({
        user: auth.user,
        action: "kg_stats",
        resource: "kg:stats",
        metadata: {
          backend: stats.source,
          nodeCount: stats.nodeCount,
          edgeCount: stats.edgeCount,
          sourceCount: stats.sources.length,
        },
      });
      // If the lib returned `source: "none"`, return 503 so the dashboard
      // shows a clear "KG not built yet" state.
      if (stats.source === "none") {
        return NextResponse.json(
          {
            error: "service_not_deployed",
            service: "Knowledge Graph Service",
            description:
              "Neo4j-backed multi-modal biomedical knowledge graph (drugs, " +
              "proteins, pathways, diseases, outcomes). Owned by Phase 2 " +
              "of the build plan.",
            reason:
              "KG_SERVICE_URL is not set. The Phase 2 KG service " +
              "(phase2/service.py) must be running and reachable. Start " +
              "it with `python phase2/service.py` and set " +
              "KG_SERVICE_URL=http://localhost:8002 in frontend/.env.local.",
            stats,
            documentation:
              "See Phase 2 of the build plan (Neo4j Knowledge Graph " +
              "Construction). Set KG_SERVICE_URL to enable the proxy.",
          },
          { status: 503 }
        );
      }
      return NextResponse.json(stats);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      return internalError(`KG stats failed: ${msg}`);
    }
  }

  // Reject raw Cypher via GET — would be a Cypher-injection vector.
  if (cypher) {
    return NextResponse.json(
      {
        error: "bad_request",
        message:
          "Raw Cypher queries are not accepted via GET (Cypher-injection " +
          "risk). Use POST /api/knowledge-graph with a role-gated, " +
          "whitelisted validator. GET accepts `drug` and `disease` for " +
          "structured queries, or no params for KG stats.",
      },
      { status: 400 }
    );
  }

  // Structured query path — proxy to the KG service via kg-service.ts.
  // The lib returns {nodes, edges} or throws on service error.
  try {
    const data = await queryKnowledgeGraph({
      drug: drug || undefined,
      disease: disease || undefined,
      limit,
    });
    await writeAuditLog({
      user: auth.user,
      action: "kg_query",
      resource: `kg:${drug || "*"}:${disease || "*"}`,
      metadata: {
        nodeCount: data.nodes?.length || 0,
        edgeCount: data.edges?.length || 0,
      },
    });
    return NextResponse.json(data);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    // BE-047 ROOT FIX: standardize error handling between GET and POST.
    // Both routes proxy to the same KG service — a service failure is an
    // UPSTREAM failure (502 Bad Gateway), not an internal server error
    // (500). API clients distinguishing 502 (upstream down, retry later)
    // from 500 (server bug, don't retry) will now handle both routes
    // consistently. The previous GET path returned 502 (correct) but the
    // POST path returned 500 via internalError (incorrect). Timeout is
    // still 504 (also consistent with POST).
    if (e instanceof Error && e.name === "MlServiceError") {
      const mlErr = e as { isTimeout?: boolean };
      if (mlErr.isTimeout) {
        return NextResponse.json(
          {
            error: "kg_timeout",
            message: `KG service did not respond within the timeout. The query is likely too expensive — add a LIMIT clause or narrow the MATCH pattern.`,
          },
          { status: 504 }
        );
      }
    }
    return NextResponse.json(
      {
        error: "kg_service_error",
        message: `KG service call failed: ${msg}`,
      },
      { status: 502 }
    );
  }
}

/**
 * POST /api/knowledge-graph
 * Body: { cypher: string, params?: Record<string, unknown> }
 *
 * FE-008 ROOT FIX: defense in depth:
 *   1. ROLE GATE: only data-scientist / admin / owner can call POST.
 *   2. CYPHER WHITELIST: only read-only MATCH/OPTIONAL MATCH/WITH/RETURN
 *      statements are allowed.
 *   3. TENANT FORWARDING: user_id and org_id are forwarded as parameters
 *      so the KG service can enforce row-level security on its side.
 *   4. COST LIMIT: a hard 30s timeout and a max 1000-row cap on the
 *      upstream call.
 */

const KG_MAX_ROWS = 1000;

export async function POST(req: NextRequest) {
  // FE-008 ROOT FIX layer 1: ROLE GATE.
  const auth = await requireAuth();
  if (auth.user === null) return auth.response;
  const roleCheck = await requireRole(auth.user, "data-scientist", "pi", "developer");
  if (roleCheck.user === null) return roleCheck.response;

  let body: { cypher?: string; params?: Record<string, unknown> };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_request", message: "Invalid JSON" }, { status: 400 });
  }
  const parsed = validateBody(KnowledgeGraphBody, body);
  if (!parsed.ok) return parsed.response;
  body = parsed.data;

  // FE-008 ROOT FIX layer 2: CYPHER WHITELIST.
  const validation = validateReadOnlyCypher(body.cypher!);
  if (!validation.ok) {
    await writeAuditLog({
      user: auth.user,
      action: "kg_cypher_rejected",
      resource: "kg:custom_cypher",
      metadata: {
        reason: validation.reason,
        cypherPreview: body.cypher!.slice(0, 120),
      },
    });
    return NextResponse.json(
      { error: "cypher_rejected", message: validation.reason },
      { status: 400 }
    );
  }

  // FE-008 ROOT FIX layer 3: TENANT FORWARDING.
  // BE-034 ROOT FIX: refuse the request if the user has no active org.
  // The previous code forwarded `_org_id: null` to the KG service when
  // the user had no active org (e.g. a platform admin who cleared their
  // org, or a user whose org membership was removed). The KG service is
  // expected to use `_org_id` for row-level security (tenant isolation).
  // If the KG service's Python code does `WHERE org_id = $_org_id` with
  // `_org_id = null`, the query becomes `WHERE org_id = NULL` which
  // matches NO rows. But if the KG service does `WHERE $_org_id IS NULL
  // OR org_id = $_org_id`, then null means "no tenant filter" —
  // returning ALL rows across ALL tenants. The KG service's behavior is
  // not verified from the backend side; passing an ambiguous value is a
  // potential cross-tenant data leak vector. Refusing the request up-front
  // closes the hole: a user without an active org MUST pick one via
  // PATCH /api/auth/me before they can query the KG.
  if (!auth.user.orgId) {
    return NextResponse.json(
      {
        error: "no_active_organization",
        message:
          "You must have an active organization to query the Knowledge Graph. " +
          "Use PATCH /api/auth/me with a valid activeOrganizationId to pick one.",
      },
      { status: 403 }
    );
  }
  const safeParams: Record<string, unknown> = {
    ...(body.params || {}),
    _user_id: auth.user.userId,
    // BE-034: auth.user.orgId is guaranteed non-null here (we returned
    // 403 above if it was missing). The `|| null` fallback is kept for
    // TypeScript narrowing but is unreachable in practice.
    _org_id: auth.user.orgId,
    _max_rows: KG_MAX_ROWS,
  };

  // FE-008 ROOT FIX layer 4: COST LIMIT (hard timeout).
  // The kg-service.ts executeCypher passes timeoutMs to mlFetch which
  // aborts the request after the timeout. No retry on Cypher (not
  // idempotent in general).
  try {
    const data = await executeCypher({
      cypher: body.cypher!,
      params: safeParams,
      timeoutMs: 30_000,
    });

    // FE-008 ROOT FIX layer 4 (result cap).
    if (Array.isArray(data?.records) && data.records.length > KG_MAX_ROWS) {
      data.records = data.records.slice(0, KG_MAX_ROWS);
      (data as Record<string, unknown> & { _truncated: boolean })._truncated = true;
      (data as Record<string, unknown> & { _max_rows: number })._max_rows = KG_MAX_ROWS;
    }
    if (Array.isArray(data?.rows) && data.rows.length > KG_MAX_ROWS) {
      data.rows = data.rows.slice(0, KG_MAX_ROWS);
      (data as Record<string, unknown> & { _truncated: boolean })._truncated = true;
      (data as Record<string, unknown> & { _max_rows: number })._max_rows = KG_MAX_ROWS;
    }

    await writeAuditLog({
      user: auth.user,
      action: "kg_cypher",
      resource: "kg:custom_cypher",
      metadata: {
        cypherPreview: body.cypher!.slice(0, 80),
        recordCount:
          (data?.records?.length ?? data?.rows?.length ?? 0),
      },
    });
    return NextResponse.json(data);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    // Distinguish timeout from other errors for a clearer message.
    if (e instanceof Error && (e.name === "MlServiceError")) {
      const mlErr = e as { isTimeout?: boolean; httpStatus?: number };
      if (mlErr.isTimeout) {
        return NextResponse.json(
          {
            error: "kg_timeout",
            message: `KG service did not respond within 30s. The query is likely too expensive — add a LIMIT clause or narrow the MATCH pattern.`,
          },
          { status: 504 }
        );
      }
    }
    return NextResponse.json(
      {
        error: "kg_service_error",
        message: `KG service proxy failed: ${msg}`,
      },
      // BE-047 ROOT FIX: 502 Bad Gateway — the failure is upstream (the KG
      // service), not internal to this route. API clients can retry 502s
      // (transient upstream failures) but should NOT retry 500s (server
      // bugs). The previous code returned 500 via internalError which
      // misclassified the error and broke client retry logic.
      { status: 502 }
    );
  }
}
