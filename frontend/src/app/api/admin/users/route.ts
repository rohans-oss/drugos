import { NextRequest, NextResponse } from "next/server";
import {
  badRequest,
  writeAuditLog,
  internalError,
} from "@/lib/api-helpers";
import { revokeAllRefreshTokensForUser } from "@/lib/auth/server";
// TASK-261 / TASK-271 ROOT FIX: replace `requireAdmin + isPlatformSuperuser`
// with `requirePlatformAdmin`. The /api/admin/* namespace is now gated on
// `platformRole === "admin"` (a SEPARATE field from `role`), enforced by
// the new middleware in lib/auth/require-platform-admin.ts.
//
// This is the architectural fix the audit asked for. The prior code
// overloaded `role` for both functional RBAC and platform-operator
// status, so promoting someone to org `admin` or `owner` accidentally
// granted them platform-superuser privileges (cross-tenant user
// enumeration, cross-tenant audit-log reads, etc.).
import { requirePlatformAdmin } from "@/lib/auth/require-platform-admin";
import { db } from "@/lib/db";
import type { UserRole, UserStatus } from "@prisma/client";
import {
  ALLOWED_ROLES_ADMIN,
  ALLOWED_USER_STATUSES,
  isValidAdminRole,
  isValidUserStatus,
} from "@/app/api/auth/register/route";
// TASK-272: Zod validation for the PATCH body.
import { validateBody, AdminUserPatchBody } from "@/lib/zod-schemas";

/**
 * GET /api/admin/users
 *
 * TASK-261 ROOT FIX: This route is now gated on `platformRole === "admin"`
 * via the requirePlatformAdmin middleware. The `role === "owner"` /
 * `role === "platformOwner"` checks from the prior fix are REMOVED —
 * `role` is the user's FUNCTIONAL role (researcher, pi, admin, owner)
 * and is no longer consulted for /api/admin/* access decisions.
 *
 * The middleware also enforces:
 *   - Authentication (401 if not logged in).
 *   - Rate limiting (1 req/sec per platform admin — Task 273).
 *   - Audit logging of every 403 (for probing detection).
 *
 * What this route returns:
 *   - For platform admins: ALL users system-wide (they are SaaS operator
 *     staff with a legitimate need-to-know across tenants).
 *   - The `email` field IS included for platform admins (GDPR legitimate
 *     interest: the operator needs to investigate cross-tenant incidents).
 *
 * Query params:
 *   - limit: max rows to return (default 50, capped at 500).
 *   - offset: pagination offset (default 0).
 *   - orgId: OPTIONAL org filter — a platform admin can scope the list
 *     to a single tenant. Without orgId, all users are returned.
 */
export async function GET(req: NextRequest) {
  const auth = await requirePlatformAdmin(req);
  if (auth.user === null) return auth.response;

  // TASK-272: validate query params with Zod. Parse to int with fallback
  // so a malformed `?limit=abc` doesn't crash the route.
  const limitRaw = req.nextUrl.searchParams.get("limit");
  const offsetRaw = req.nextUrl.searchParams.get("offset");
  const requestedOrgId = req.nextUrl.searchParams.get("orgId");

  const limit = Math.min(Math.max(parseInt(limitRaw || "50", 10) || 50, 1), 500);
  const offset = Math.max(parseInt(offsetRaw || "0", 10) || 0, 0);

  // Platform admins can scope to a single org if requested. This is
  // useful for the "view all users in tenant X" admin console page.
  const whereClause = requestedOrgId
    ? {
        organizationMemberships: {
          some: { organizationId: requestedOrgId },
        },
      }
    : {};

  const [users, total] = await Promise.all([
    db.user.findMany({
      where: whereClause,
      select: {
        id: true,
        email: true,
        name: true,
        role: true,
        // TASK-261: include platformRole in the response so the admin
        // console can display who has platform-admin access. This is
        // safe — the caller is themselves a platform admin.
        platformRole: true,
        status: true,
        emailVerified: true,
        mfaEnabled: true,
        createdAt: true,
        lastLoginAt: true,
        deletedAt: true,
      },
      orderBy: { createdAt: "desc" },
      take: limit,
      skip: offset,
    }),
    db.user.count({ where: whereClause }),
  ]);

  await writeAuditLog({
    user: auth.user,
    action: "platform_admin_user_list",
    resource: requestedOrgId ? `org:${requestedOrgId}` : "system:all_users",
    metadata: { limit, offset, total },
  }).catch(() => {
    // Best-effort — don't fail the request if the audit log is down.
  });

  return NextResponse.json({ items: users, total });
}

/**
 * PATCH /api/admin/users
 * Body: { userId: string, role?: string, status?: string }
 *
 * TASK-261 ROOT FIX: gated on `platformRole === "admin"` via the
 * requirePlatformAdmin middleware. A platform admin can change ANY
 * user's role or status (they are SaaS operator staff with cross-tenant
 * authority).
 *
 * TASK-272 ROOT FIX: Zod validation on the PATCH body. The schema
 * rejects:
 *   - Missing userId.
 *   - Unknown role/status values.
 *   - Empty body (neither role nor status provided).
 *   - `platformRole` in the body (intentionally not in the schema —
 *     the platformRole field is settable ONLY via direct DB access).
 *
 * TASK-267 ROOT FIX: every PATCH writes a critical audit log entry.
 * Critical means: if the audit log write fails, the request is ABORTED
 * with a 500 (the action MUST be auditable — FDA 21 CFR Part 11).
 *
 * Privilege escalation guards:
 *   - `platformOwner` role is REJECTED (it's not in ALLOWED_ROLES_ADMIN).
 *   - The `platformRole` field CANNOT be set via this route (Zod schema
 *     rejects unknown keys).
 *   - Cross-tenant IDOR: a platform admin CAN modify any user (that's
 *     their job) — but every modification is audit-logged with the
 *     admin's identity, the target's identity, and the delta.
 */
export async function PATCH(req: NextRequest) {
  // TASK-261 / TASK-271: replace requireAdmin with requirePlatformAdmin.
  // The middleware handles CSRF, rate limiting, and the platformRole gate.
  const auth = await requirePlatformAdmin(req);
  if (auth.user === null) return auth.response;

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON");
  }

  // TASK-272: validate the body with Zod.
  const parsed = validateBody(AdminUserPatchBody, body);
  if (!parsed.ok) return parsed.response;
  const { userId, role, status } = parsed.data;

  // Build the update data — cast to the Prisma enum types (the Zod
  // schema + ALLOWED_ROLES_ADMIN guarantee the values are valid).
  const data: { role?: UserRole; status?: UserStatus } = {};
  if (role !== undefined) {
    if (!isValidAdminRole(role)) {
      return badRequest(
        `Invalid role. Must be one of: ${(ALLOWED_ROLES_ADMIN as readonly string[]).join(", ")}`
      );
    }
    data.role = role as UserRole;
  }
  if (status !== undefined) {
    if (!isValidUserStatus(status)) {
      return badRequest(
        `Invalid status. Must be one of: ${(ALLOWED_USER_STATUSES as readonly string[]).join(", ")}`
      );
    }
    data.status = status as UserStatus;
  }

  // TASK-267: critical audit log for the role/status change. If the
  // audit log write fails, the request is ABORTED — the action MUST
  // be auditable (FDA 21 CFR Part 11). The audit log entry is written
  // BEFORE the update so a failed update doesn't leave an un-audited
  // intent in the system.
  const auditResult = await writeAuditLog({
    user: auth.user,
    action: "platform_admin_user_update",
    resource: `user:${userId}`,
    metadata: { role, status, performedBy: auth.user.userId },
    critical: true,
  });
  if (!auditResult.ok) {
    return internalError("Audit log write failed. Action aborted for compliance.");
  }

  const updated = await db.user.update({
    where: { id: userId },
    data,
    select: {
      id: true,
      email: true,
      name: true,
      role: true,
      platformRole: true,
      status: true,
    },
  });

  // FE-032 ROOT FIX: If the user was just suspended, revoke ALL their
  // refresh tokens so existing sessions stop working immediately.
  if (data.status === "suspended") {
    const revokedCount = await revokeAllRefreshTokensForUser(updated.id);
    await writeAuditLog({
      user: auth.user,
      action: "platform_admin_user_suspended_tokens_revoked",
      resource: `user:${updated.id}`,
      metadata: { revokedRefreshTokenCount: revokedCount },
    }).catch(() => {
      // Non-critical — the suspension itself was already audited above.
    });
  }

  return NextResponse.json(updated);
}

/**
 * DELETE /api/admin/users
 * Body: { userId: string }
 *
 * TASK-267 ROOT FIX: soft-delete a user (set deletedAt). The user's
 * row is PRESERVED for audit trails — FDA 21 CFR Part 11 requires
 * complete audit trails, and hard-deleting a user would orphan every
 * audit log entry they appear in. Login is refused for soft-deleted
 * accounts (see FE-055 in the login route).
 *
 * This route is gated on `platformRole === "admin"` — only SaaS
 * operator staff can delete users. Org admins use PATCH with
 * status=suspended to deactivate users in their own org.
 */
export async function DELETE(req: NextRequest) {
  const auth = await requirePlatformAdmin(req);
  if (auth.user === null) return auth.response;

  let body: { userId?: string };
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON");
  }
  if (!body.userId) return badRequest("userId is required");

  // Critical audit log — user deletion is a high-impact action.
  const auditResult = await writeAuditLog({
    user: auth.user,
    action: "platform_admin_user_delete",
    resource: `user:${body.userId}`,
    metadata: { performedBy: auth.user.userId },
    critical: true,
  });
  if (!auditResult.ok) {
    return internalError("Audit log write failed. Action aborted for compliance.");
  }

  // Soft-delete: set deletedAt. The row is preserved for audit trails.
  // Login refuses soft-deleted accounts (FE-055).
  const updated = await db.user.update({
    where: { id: body.userId },
    data: { deletedAt: new Date(), status: "suspended" },
    select: { id: true, email: true, name: true, deletedAt: true },
  });

  // Revoke all refresh tokens so existing sessions stop working.
  const revokedCount = await revokeAllRefreshTokensForUser(updated.id);
  await writeAuditLog({
    user: auth.user,
    action: "platform_admin_user_deleted_tokens_revoked",
    resource: `user:${updated.id}`,
    metadata: { revokedRefreshTokenCount: revokedCount },
  }).catch(() => {
    // Non-critical — the deletion itself was already audited above.
  });

  return NextResponse.json({ ok: true, deleted: updated });
}
