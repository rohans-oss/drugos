import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/api-helpers";
// TASK-261: import isPlatformAdmin so /api/audit-logs can decide whether
// to scope queries to the user's org (org-scoped admin) or allow
// system-wide access (platform admin). The route stays gated on
// requireAdmin (org-scoped admin / owner / platformOwner) — it's NOT
// gated on requirePlatformAdmin, because org admins have a legitimate
// need to read their own org's audit logs (compliance reporting, DSARs).
import { isPlatformAdmin } from "@/lib/auth/require-platform-admin";
import { db } from "@/lib/db";
// TASK-272: Zod validation on query params.
import { AuditLogsQuery } from "@/lib/zod-schemas";

/**
 * GET /api/audit-logs
 *
 * Returns audit log entries. The route is wired to the REAL AuditLog
 * Prisma model (db.auditLog.findMany) — Task 262's "currently returns
 * mock logs" finding was a stale description; the previous fix already
 * wired the route to the real table. This commit adds:
 *
 *   1. TASK-261: cross-tenant access is now granted to `platformRole ===
 *      "admin"` (the new SaaS-operator field), NOT to `role ===
 *      "platformOwner"`. The two are equivalent in effect but the new
 *      field is the architectural source of truth (see PlatformRole enum
 *      in prisma/schema.prisma for the rationale).
 *
 *   2. TASK-272: Zod validation on query params (limit, action,
 *      dead_letter). The previous code used `parseInt(... || "100", 10)`
 *      which returned NaN for `?limit=abc`, then `Math.min(NaN, 1000) =
 *      NaN`, then `take: NaN` in Prisma — silently returning zero rows
 *      or throwing P2009. The Zod schema coerces and bounds the value.
 *
 *   3. TASK-280: the route returns 503 (not 200) when the underlying
 *      AuditLog table is unreachable. The previous code threw an
 *      unhandled exception → 500. The new behavior lets the monitoring
 *      layer (system/status) detect the outage and alert.
 *
 * Query params (Zod-validated):
 *   - limit: max rows to return (default 100, capped at 1000).
 *   - action: filter by audit action (e.g. "login", "billing_change").
 *   - dead_letter: if "true", return dead-letter entries instead of
 *     primary audit logs (BE-003). Only platform admins can access
 *     this — dead-letter entries may contain cross-tenant error details.
 */
export async function GET(req: NextRequest) {
  const auth = await requireAdmin();
  if (auth.user === null) return auth.response;

  // TASK-272: parse + validate query params with Zod. The schema coerces
  // strings to numbers and applies bounds. On validation failure, return
  // 400 with the structured error envelope.
  const parseResult = AuditLogsQuery.safeParse({
    limit: req.nextUrl.searchParams.get("limit") ?? undefined,
    action: req.nextUrl.searchParams.get("action") ?? undefined,
    dead_letter: req.nextUrl.searchParams.get("dead_letter") ?? undefined,
  });
  if (!parseResult.success) {
    return NextResponse.json(
      {
        error: "bad_request",
        message: "Invalid query parameters.",
        issues: parseResult.error.issues.map((iss) => ({
          path: iss.path.join("."),
          message: iss.message,
        })),
      },
      { status: 400 }
    );
  }
  const { limit, action, dead_letter } = parseResult.data;
  const deadLetter = dead_letter === "true";

  // TASK-261: only `platformRole === "admin"` bypasses org scoping.
  // The prior `isPlatformSuperuser` check (which consulted `role ===
  // "platformOwner"`) is REPLACED with `isPlatformAdmin` (which
  // consults the new `platformRole` field). For backwards compat, we
  // ALSO accept `role === "platformOwner"` so existing SaaS-operator
  // accounts don't lose access until the operator migrates them to
  // the new `platformRole === "admin"` field.
  const crossTenantAllowed =
    isPlatformAdmin(auth.user) || auth.user.role === "platformOwner";

  // BE-003: dead-letter entries are ONLY visible to platform admins.
  // They may contain cross-tenant error details (e.g. the userId of a
  // user in another org whose audit log write failed). Org-scoped
  // admins should NOT see another org's failed audit writes.
  if (deadLetter && !crossTenantAllowed) {
    return NextResponse.json(
      {
        error: "forbidden",
        message:
          "Dead-letter audit entries are only visible to platform administrators. " +
          "Contact your SaaS operator if you need to inspect dead-letter entries.",
      },
      { status: 403 }
    );
  }

  // Build the where-clause. Platform admins see system-wide logs (they
  // are the SaaS operator's staff). Everyone else (admin, owner,
  // billing, etc.) is restricted to their own org.
  const orgFilter = crossTenantAllowed
    ? {}
    : { organizationId: auth.user.orgId ?? "__NO_ORG__" };

  const where = {
    ...orgFilter,
    ...(action ? { action } : {}),
  };

  try {
    if (deadLetter) {
      const logs = await db.auditLogDeadLetter.findMany({
        where,
        orderBy: { createdAt: "desc" },
        take: limit,
      });
      const total = await db.auditLogDeadLetter.count({ where });
      return NextResponse.json({
        items: logs,
        total,
        scope: "system",
        deadLetter: true,
        organizationId: auth.user.orgId ?? null,
      });
    }

    const logs = await db.auditLog.findMany({
      where,
      orderBy: { createdAt: "desc" },
      take: limit,
    });
    const total = await db.auditLog.count({ where });
    return NextResponse.json({
      items: logs,
      total,
      scope: crossTenantAllowed ? "system" : "organization",
      organizationId: auth.user.orgId ?? null,
    });
  } catch (e) {
    // TASK-280: return 503 when the audit log table is unreachable so
    // the monitoring layer (system/status) can detect the outage and
    // alert. The previous code threw an unhandled exception → 500.
    console.error("[AUDIT-LOGS] DB error:", e);
    return NextResponse.json(
      {
        error: "service_unavailable",
        message: "Audit log database is currently unavailable.",
      },
      { status: 503 }
    );
  }
}
