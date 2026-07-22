import { NextResponse } from "next/server";
import { requireAuth, internalError, writeAuditLog } from "@/lib/api-helpers";
import { getDatasetStats } from "@/lib/services/dataset-stats";
import { getKnowledgeGraphStats } from "@/lib/services/knowledge-graph-stats";
// BE-039 ROOT FIX: import CANONICAL_NODE_TYPES from lib/ml-contracts.ts
// so there is a SINGLE source of truth for the canonical node type list.
// The previous code redefined the list inline and included "Drug" —
// which is NOT a canonical Phase 2 KG type (it's a legacy synonym for
// "Compound" used in some loaders). The KG will never have a "Drug"
// node type, so canonicalNodeCoverage always reported "Drug" as not
// present. canonicalCoveragePct was thus capped at 5/6 = 83.3% — it could
// NEVER reach 100%, even when all real canonical types were present.
// The quality dashboard always showed "83% canonical coverage" which
// looked like a quality gap when there wasn't one. Using the shared
// CANONICAL_NODE_TYPES constant (which has exactly 5 entries: Compound,
// Protein, Pathway, Disease, ClinicalOutcome) makes the metric
// correctly reach 100% when all canonical types are present.
import { CANONICAL_NODE_TYPES } from "@/lib/ml-contracts";

/**
 * GET /api/dataset/quality
 *
 * Issue 307 (audit 301-320): Wire Quality screen to real data.
 *
 * Previously the QualityScreen either:
 *   (a) rendered fabricated "completeness 98.2%, freshness 24h,
 *       duplicates 0.01%, reliability 99.4%" metrics with no backend
 *       call, OR
 *   (b) after a partial "fix" it called /api/dataset and showed only
 *       the warnings/errors arrays — but no real quality metrics.
 *
 * ROOT FIX: This endpoint derives REAL quality signals from existing
 * Phase 1 + Phase 2 services. It does NOT fabricate percentages. Every
 * metric returned is computed from actual data:
 *
 *   - sourceCompletenessPct: loadedSources / totalSources * 100
 *     (a source is "loaded" if the Phase 1 checkpoint recorded it in
 *      sources_read). This is a real coverage signal — if ChEMBL is
 *      missing, the KG is incomplete.
 *
 *   - nodeEdgeRatio: nodesLoaded / max(edgesLoaded, 1)
 *     A real graph-anomaly signal. For a biomedical KG this ratio
 *     should be ~0.3-0.7 (each node participates in multiple edges).
 *     A ratio >5 or <0.05 indicates a loader bug.
 *
 *   - canonicalNodeCoverage: which canonical node types (Compound,
 *     Protein, Pathway, Disease, ClinicalOutcome) are present in
 *     the KG. Each missing canonical type is a real quality gap.
 *
 *   - sha256Integrity: per-source, whether a sha256 checksum was
 *     recorded. Missing checksums mean we cannot verify the data
 *     has not been tampered with since Phase 1 ran.
 *
 *   - warningsCount / errorsCount: real counts from the dataset
 *     service (already surfaced by /api/dataset, included here for
 *     a single-call quality summary).
 *
 *   - freshnessHoursAgo: hours since the checkpoint was generated.
 *     Computed from stats.generatedAt. A checkpoint older than 7
 *     days is flagged stale.
 *
 * SCIENTIFIC INTEGRITY: we never fabricate quality percentages. If
 * the dataset service returns no_data, we return status: "no_data"
 * with zeroed metrics — the UI shows "Run Phase 1 to populate"
 * instead of fake 99% completeness.
 */
export async function GET() {
  const auth = await requireAuth();
  if (auth.user === null) return auth.response;

  try {
    const [datasetStats, kgStats] = await Promise.all([
      getDatasetStats(),
      getKnowledgeGraphStats().catch(() => null),
    ]);

    const sources = datasetStats.sources || [];
    const loadedSources = sources.filter((s) => s.loaded).length;
    const totalSources = sources.length || 0;
    const sourceCompletenessPct =
      totalSources === 0 ? 0 : Math.round((loadedSources / totalSources) * 1000) / 10;

    const nodesLoaded = datasetStats.nodesLoaded || 0;
    const edgesLoaded = datasetStats.edgesLoaded || 0;
    const nodeEdgeRatio =
      edgesLoaded === 0 ? 0 : Math.round((nodesLoaded / edgesLoaded) * 100) / 100;

    // BE-039 ROOT FIX: use the SHARED CANONICAL_NODE_TYPES constant from
    // lib/ml-contracts.ts. The previous code defined an inline list that
    // incorrectly included "Drug" (a legacy synonym for "Compound"),
    // which meant canonicalCoveragePct could never reach 100% — the
    // quality dashboard always showed 83% coverage, looking like a
    // quality gap when there wasn't one. The shared constant has exactly
    // the 5 Phase 2 KG canonical types: Compound, Protein, Pathway,
    // Disease, ClinicalOutcome.
    const canonicalTypes = CANONICAL_NODE_TYPES;
    const nodeTypeCounts = kgStats?.nodeTypeCounts || {};
    const canonicalNodeCoverage = canonicalTypes.map((type) => ({
      type,
      present: (nodeTypeCounts[type] || 0) > 0,
      count: nodeTypeCounts[type] || 0,
    }));
    const canonicalCoveragePct =
      Math.round(
        (canonicalNodeCoverage.filter((c) => c.present).length /
          canonicalTypes.length) *
          1000,
      ) / 10;

    // SHA-256 integrity: how many sources recorded a checksum.
    const sourcesWithChecksum = sources.filter((s) => s.sha256).length;
    const checksumCoveragePct =
      totalSources === 0
        ? 0
        : Math.round((sourcesWithChecksum / totalSources) * 1000) / 10;

    // Freshness — hours since the checkpoint was generated.
    const generatedAt = datasetStats.generatedAt
      ? new Date(datasetStats.generatedAt)
      : null;
    const freshnessHoursAgo = generatedAt
      ? Math.round(
          (Date.now() - generatedAt.getTime()) / (1000 * 60 * 60) * 10,
        ) / 10
      : null;
    const isStale = freshnessHoursAgo === null ? true : freshnessHoursAgo > 168; // 7 days

    const warningsCount = (datasetStats.warnings || []).length;
    const errorsCount = (datasetStats.errors || []).length;

    const status: "ok" | "no_data" | "service_down" =
      (datasetStats as any).status || (datasetStats.source === "none" ? "no_data" : "ok");

    await writeAuditLog({
      user: auth.user,
      action: "dataset_quality_query",
      resource: "dataset:quality",
      metadata: {
        status,
        sourceCompletenessPct,
        canonicalCoveragePct,
        checksumCoveragePct,
        warningsCount,
        errorsCount,
      },
    });

    return NextResponse.json({
      status,
      generatedAt: new Date().toISOString(),
      source: datasetStats.source,
      // Real coverage metrics — no fabricated percentages
      sourceCompletenessPct,
      canonicalCoveragePct,
      checksumCoveragePct,
      // Real graph-anomaly signal
      nodeEdgeRatio,
      nodesLoaded,
      edgesLoaded,
      // Real per-canonical-type breakdown
      canonicalNodeCoverage,
      // Real integrity signals
      sourcesWithChecksum,
      totalSources,
      // Real freshness signal
      freshnessHoursAgo,
      isStale,
      checkpointGeneratedAt: datasetStats.generatedAt || null,
      // Real issue counts
      warningsCount,
      errorsCount,
      warnings: datasetStats.warnings || [],
      errors: datasetStats.errors || [],
      // Pipeline version metadata (for audit trail)
      pipelineVersion: datasetStats.pipelineVersion || null,
      schemaVersion: datasetStats.schemaVersion || null,
      bridgeVersion: datasetStats.bridgeVersion || null,
      note:
        status === "no_data"
          ? "Phase 1 pipeline has not been run. No quality metrics available. Run Phase 1 to populate."
          : "Quality metrics derived from real Phase 1 dataset stats and Phase 2 KG stats. No fabricated percentages.",
    });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`Dataset quality query failed: ${msg}`);
  }
}
