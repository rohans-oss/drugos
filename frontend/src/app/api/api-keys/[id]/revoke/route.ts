import { NextRequest, NextResponse } from "next/server";
import { requireRoleOrSend, notFound, requireCsrfOrSend, writeAuditLog, internalError } from "@/lib/api-helpers";
import { revokeApiKey } from "@/lib/services/api-keys";

/**
 * POST /api/api-keys/[id]/revoke
 *
 * ROOT FIX for FE-022 (API key revocation not scoped to owning user).
 *
 * A non-admin caller can now revoke ONLY their own keys. Admin/owner can
 * revoke any key in the org.
 *
 * TASK-267 ROOT FIX: Add audit logging for API key revocation. The
 * previous code revoked a key with NO audit trail — a malicious insider
 * could revoke every developer's API key (DoS) and leave no trace.
 * The audit log entry is CRITICAL — if it fails, the request is ABORTED.
 */
export async function POST(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  // FE-011: CSRF protection on every state-changing route.
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  const auth = await requireRoleOrSend("developer", "admin", "owner");
  if (auth.user === null) return auth.response;
  if (!auth.user.orgId) {
    return NextResponse.json({ error: "bad_request", message: "No active organization" }, { status: 400 });
  }
  const { id } = await params;
  // Non-admin/owner callers can only revoke their own keys.
  const ownerFilter = (auth.user.role === "admin" || auth.user.role === "owner")
    ? undefined
    : auth.user.userId;

  // TASK-267: write the audit log BEFORE the revoke. If the audit fails,
  // we ABORT the request — the revocation MUST be auditable (otherwise
  // a malicious insider could revoke every key silently).
  const auditResult = await writeAuditLog({
    user: auth.user,
    action: "api_key_revoke",
    resource: `api-key:${id}`,
    metadata: {
      organizationId: auth.user.orgId,
      revokedBy: auth.user.userId,
      ownerFilter,
    },
    critical: true,
  });
  if (!auditResult.ok) {
    return internalError("Audit log write failed. Revocation aborted for compliance.");
  }

  const ok = await revokeApiKey(auth.user.orgId, id, ownerFilter);
  if (!ok) return notFound("API key not found, already revoked, or not owned by you");
  return NextResponse.json({ ok: true });
}
