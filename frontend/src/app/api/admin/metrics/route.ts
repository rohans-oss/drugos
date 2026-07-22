import { NextRequest, NextResponse } from "next/server";
import {
  internalError,
  writeAuditLog,
} from "@/lib/api-helpers";
// BE-007 ROOT FIX: Replace `requireAdmin + isPlatformSuperuser` with
// `requirePlatformAdmin`. The /api/admin/* namespace is now gated on
// `platformRole === "admin"` (a SEPARATE field from `role`), enforced by
// the middleware in lib/auth/require-platform-admin.ts. This is the
// architectural fix the audit asked for. The prior code accepted any user
// with `role === "admin" | "owner" | "platformOwner"` — including
// org-scoped admins who should NOT see system-wide metrics. An org admin
// could hit /api/admin/metrics and see total user counts across ALL
// tenants — a cross-tenant data leak. The new gate ensures only SaaS
// operator staff (platformRole === "admin") can call this endpoint.
//
// BE-007 (continued): Since requirePlatformAdmin already gates on
// platformRole === "admin", the route ALWAYS returns system-wide metrics
// (platform admins have legitimate need-to-know across tenants — they
// are SaaS operator staff). The old `isPlatformSuperuser` check (which
// gated system-wide vs org-scoped counting on `role === "platformOwner"`)
// is REMOVED entirely. There is no org-scoped path here anymore.
import { requirePlatformAdmin } from "@/lib/auth/require-platform-admin";
import { db } from "@/lib/db";
import { getDatasetStats } from "@/lib/services/dataset-stats";
import { getKnowledgeGraphStats } from "@/lib/services/knowledge-graph-stats";

/**
 * GET /api/admin/metrics
 *
 * Issue 315 (audit 301-320): Wire Investor Dashboard to real platform metrics.
 *
 * Previously the InvestorDashboardScreen rendered fabricated ARR/MRR/customer
 * counts/NRR/cohort data — a serious problem because showing fake financials
 * to actual investors is securities fraud.
 *
 * ROOT FIX: This endpoint returns REAL, derivable platform metrics from
 * existing DB tables and services. It does NOT fabricate financial data.
 *
 * What we CAN compute from existing data:
 *   - totalUsers: COUNT(User)
 *   - totalOrganizations: COUNT(Organization)
 *   - activeSubscriptions: COUNT(Subscription WHERE status='active')
 *   - totalProjects: COUNT(Project)
 *   - totalHypotheses: COUNT(Hypothesis)
 *   - totalValidatedHypotheses: COUNT(Hypothesis WHERE status='validated')
 *   - auditLogEventsLast30Days: COUNT(AuditLog WHERE createdAt > now-30d)
 *   - topActionsLast30Days: GROUP BY action, COUNT, ORDER BY count DESC
 *   - dailyActiveUsersLast7Days: COUNT(DISTINCT userId) in AuditLog per day
 *   - kgNodeCount / kgEdgeCount: from getKnowledgeGraphStats()
 *   - datasetNodesLoaded / datasetEdgesLoaded: from getDatasetStats()
 *
 * What we CANNOT compute (and explicitly refuse to fabricate):
 *   - ARR / MRR: requires Stripe integration (not deployed)
 *   - Customer counts: requires CRM integration (not deployed)
 *   - NRR / cohort retention: requires subscription-event history (not modeled)
 *   - EBITDA projections: requires accounting system integration (not deployed)
 *
 * BE-007: AUTH is now `requirePlatformAdmin(req)` — gates on
 * `platformRole === "admin"` (SaaS operator staff). Org-scoped admins no
 * longer reach this route. The metrics returned are ALWAYS system-wide
 * (the platform admin has legitimate need-to-know across tenants).
 *
 * BE-081 ROOT FIX: The previous `dailyActiveUsersLast7Days` query used
 * PostgreSQL-specific raw SQL with `DATE("createdAt")` and
 * `NOW() - INTERVAL '7 days'`. This breaks if the DB is ever switched
 * (e.g. to MySQL for cost reasons) and is opaque to the pg-mem test
 * harness. Replaced with Prisma's `groupBy` on a date-truncated expression
 * computed in JavaScript — the query is now dialect-agnostic and the
 * bigint → number footgun is gone. The query is also more efficient
 * (no raw SQL parsing, no DATE() function call per row).
 */
export async function GET(req: NextRequest) {
  const auth = await requirePlatformAdmin(req);
  if (auth.user === null) return auth.response;

  try {
    // BE-007: The route ALWAYS returns system-wide metrics. Platform admins
    // (the only callers who reach this point) have legitimate need-to-know
    // across tenants — they are SaaS operator staff investigating incidents,
    // building investor reports, and monitoring platform health. There is
    // no org-scoped path here anymore.
    //
    // Run all DB counts in parallel for performance.
    const [
      totalUsers,
      totalOrganizations,
      activeSubscriptions,
      totalProjects,
      totalHypotheses,
      totalValidatedHypotheses,
      totalEvidencePackages,
      auditLogEventsLast30Days,
      topActionsRows,
      auditLogsLast7Days,
      datasetStats,
      kgStats,
    ] = await Promise.all([
      // User count — system-wide (platform admin only).
      db.user.count(),
      // Org count — system-wide.
      db.organization.count(),
      // Active subscriptions — system-wide.
      db.subscription.count({ where: { status: "active" } }),
      // Project count — system-wide.
      db.project.count(),
      // Hypothesis count — system-wide.
      db.hypothesis.count(),
      // Validated hypothesis count — the "real discoveries" metric.
      db.hypothesis.count({ where: { status: "validated" } }),
      // Evidence package count — system-wide.
      db.evidencePackage.count(),
      // Audit log events in last 30 days — system-wide.
      db.auditLog.count({
        where: {
          createdAt: { gt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) },
        },
      }),
      // Top actions in last 30 days — GROUP BY action via Prisma (dialect-agnostic).
      db.auditLog.groupBy({
        by: ["action"],
        _count: { _all: true },
        where: {
          createdAt: { gt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) },
        },
        orderBy: { _count: { action: "desc" } },
        take: 10,
      }),
      // BE-081: Fetch audit logs from the last 7 days with their userId and
      // createdAt. We do the per-day DISTINCT userId aggregation in JS so
      // the query is dialect-agnostic (works on PostgreSQL, MySQL, SQLite,
      // and pg-mem). The previous raw SQL used DATE("createdAt") and
      // NOW() - INTERVAL '7 days' which are PostgreSQL-specific.
      db.auditLog.findMany({
        where: {
          createdAt: { gt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) },
          userId: { not: null },
        },
        select: { userId: true, createdAt: true },
      }),
      // Phase 1 dataset stats.
      getDatasetStats(),
      // Phase 2 KG stats.
      getKnowledgeGraphStats().catch(() => null),
    ]);

    // Convert Prisma groupBy result to plain object.
    const topActionsLast30Days = topActionsRows.map((r) => ({
      action: r.action,
      count: Number(r._count._all),
    }));

    // BE-081: Compute daily active users (distinct userId per day) in JS.
    // Group by YYYY-MM-DD day string, then count distinct userIds per day.
    const dayMap = new Map<string, Set<string>>();
    for (const row of auditLogsLast7Days) {
      if (!row.userId) continue;
      const dayKey = row.createdAt.toISOString().slice(0, 10); // YYYY-MM-DD
      let set = dayMap.get(dayKey);
      if (!set) {
        set = new Set();
        dayMap.set(dayKey, set);
      }
      set.add(row.userId);
    }
    const dailyActiveUsersLast7Days = Array.from(dayMap.entries())
      .map(([day, users]) => ({ day, activeUsers: users.size }))
      .sort((a, b) => (a.day < b.day ? 1 : -1));

    // Aggregate top-level metrics.
    const metrics = {
      scope: "system" as const,
      generatedAt: new Date().toISOString(),
      // REAL user/org/subscription counts
      totalUsers,
      totalOrganizations,
      activeSubscriptions,
      // REAL research activity counts
      totalProjects,
      totalHypotheses,
      totalValidatedHypotheses,
      totalEvidencePackages,
      // REAL platform activity (last 30 days)
      auditLogEventsLast30Days,
      topActionsLast30Days,
      dailyActiveUsersLast7Days,
      // REAL Phase 1 + Phase 2 data scale
      dataset: {
        nodesLoaded: datasetStats.nodesLoaded,
        edgesLoaded: datasetStats.edgesLoaded,
        sourcesLoaded: datasetStats.sources.filter((s) => s.loaded).length,
        sourcesTotal: datasetStats.sources.length,
        source: datasetStats.source,
        status: (datasetStats as any).status,
      },
      knowledgeGraph: kgStats
        ? {
            nodeCount: kgStats.nodeCount,
            edgeCount: kgStats.edgeCount,
            source: kgStats.source,
          }
        : null,
      // EXPLICITLY NOT FABRICATED — financial metrics are omitted, not invented
      financials: {
        arr: null,
        mrr: null,
        customerCount: null,
        nrr: null,
        note: "Financial metrics (ARR/MRR/customer count/NRR) require Stripe + CRM integration. NOT fabricated. Wire a billing system before exposing these to investors.",
      },
    };

    await writeAuditLog({
      user: auth.user,
      action: "admin_metrics_query",
      resource: "admin:metrics",
      metadata: {
        scope: metrics.scope,
        totalUsers,
        totalProjects,
        totalValidatedHypotheses,
      },
    });

    return NextResponse.json(metrics);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return internalError(`Admin metrics query failed: ${msg}`);
  }
}
