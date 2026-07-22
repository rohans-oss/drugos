import { NextRequest, NextResponse } from "next/server";
import { requireAuth, internalError, writeAuditLog } from "@/lib/api-helpers";
// Issue 226 ROOT FIX: import from the unified dataset-service.ts (HTTP-only).
// The previous version imported from dataset-stats.ts which had a local
// checkpoint fallback that read ../phase2/data/checkpoints/step_01.json
// (a Phase 2 artifact, NOT Phase 1). The new dataset-service.ts is
// HTTP-only and proxies to PHASE1_SERVICE_URL/stats (with
// DATASET_SERVICE_URL honored as a legacy alias).
import { getDatasetStats } from "@/lib/services/dataset-service";

/**
 * GET /api/dataset?source=<chembl|drugbank|uniprot|string|disgenet|omim|pubchem>
 *
 * Issue 226 ROOT FIX: this route now reads from Phase 1 directly via the
 * PHASE1_SERVICE_URL/stats endpoint. The previous version had a local
 * checkpoint fallback that read ../phase2/data/checkpoints/step_01.json
 * — a Phase 2 BRIDGE SUMMARY, not a Phase 1 artifact. This caused the
 * dashboard to display Phase 2's view of the data (post-entity-resolution)
 * instead of Phase 1's actual ingestion state (per-source loaded status,
 * row counts, SHA-256 checksums).
 *
 * The new path is:
 *   1. dataset-service.ts calls GET {PHASE1_SERVICE_URL}/stats
 *   2. phase1/service.py reads the REAL Phase 1 pipeline state (CSV row
 *      counts from phase1/data/processed/, bridge summary from
 *      phase1/data/checkpoints/step_01.json)
 *   3. The response is validated against the DatasetStatsResponseSchema
 *      contract (ml-contracts.ts)
 *
 * SCIENTIFIC INTEGRITY: we NEVER fabricate dataset statistics. If the
 * service is not configured or returns no data, we return
 * status: "no_data" with empty arrays — the dashboard shows "Run Phase 1
 * to populate" instead of fake numbers.
 */
export async function GET(req: NextRequest) {
  const auth = await requireAuth();
  if (auth.user === null) return auth.response;

  try {
    const stats = await getDatasetStats();
    const source = req.nextUrl.searchParams.get("source") || "all";

    await writeAuditLog({
      user: auth.user,
      action: "dataset_query",
      resource: `dataset:${source}`,
      metadata: {
        source: stats.source,
        status: stats.status,
        count: stats.sources.length,
      },
    });

    // 502 only when the proxy was configured but failed AND no data
    // was returned. 200 for ok + no_data (request succeeded; data may
    // be empty).
    if (stats.status === "service_down") {
      return NextResponse.json(stats, { status: 502 });
    }

    // If a specific source was requested, filter the sources list.
    if (source !== "all" && stats.sources.length > 0) {
      const filtered = stats.sources.filter(
        (s) => s.name.toLowerCase() === source.toLowerCase()
      );
      return NextResponse.json({ ...stats, sources: filtered });
    }

    return NextResponse.json(stats);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`Dataset stats failed: ${msg}`);
  }
}
