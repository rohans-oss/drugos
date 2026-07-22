import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import {
  getAuthenticatedUser,
  hashPassword,
  verifyPassword,
  validatePasswordPolicy,
  revokeAllRefreshTokensForUser,
  clearAuthCookies,
} from "@/lib/auth/server";
import { badRequest, internalError, writeAuditLog, requireCsrfOrSend } from "@/lib/api-helpers";
// BE-029 ROOT FIX (Team Member 12): Zod-validated request body.
import { validateBody, PasswordChangeBody } from "@/lib/zod-schemas";

/**
 * POST /api/auth/password
 * Body: { currentPassword: string, newPassword: string }
 *
 * Verifies the current password, validates the new password against the
 * policy, and updates the user's passwordHash. Returns 200 on success.
 */
export async function POST(req: NextRequest) {
  // FE-011: CSRF protection on every state-changing route.
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  const user = await getAuthenticatedUser();
  if (!user) {
    return NextResponse.json({ error: "unauthorized", message: "Authentication required" }, { status: 401 });
  }

  let body: { currentPassword?: string; newPassword?: string };
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON body");
  }

  // BE-029 ROOT FIX: schema-validate the body before any business logic.
  // Rejects: missing fields, non-string fields, oversize values, and
  // newPassword shorter than 8 chars (defense-in-depth — the policy
  // check below does a richer validation, but the schema catches the
  // obviously-bad cases before we touch the DB).
  const parsed = validateBody(PasswordChangeBody, body);
  if (!parsed.ok) return parsed.response;
  const currentPassword = parsed.data.currentPassword;
  const newPassword = parsed.data.newPassword;

  const policy = validatePasswordPolicy(newPassword);
  if (!policy.ok) {
    return badRequest(policy.reason || "New password does not meet policy.");
  }

  const dbUser = await db.user.findUnique({ where: { id: user.userId } });
  if (!dbUser) {
    return NextResponse.json({ error: "not_found", message: "User not found" }, { status: 404 });
  }

  const ok = await verifyPassword(currentPassword, dbUser.passwordHash);
  if (!ok) {
    // BE-032 ROOT FIX (Team Member 12): return 401 (authentication
    // failure) — NOT 403. A wrong current password is an authentication
    // failure: the user has not proven they own the account. 403 means
    // "authenticated but forbidden", which is wrong here. API clients
    // that distinguish 401 (re-authenticate) from 403 (forbidden) rely
    // on this to prompt the user to re-enter their password.
    return NextResponse.json(
      { error: "invalid_credentials", message: "Current password is incorrect." },
      { status: 401 }
    );
  }

  try {
    const newHash = await hashPassword(newPassword);
    await db.user.update({
      where: { id: user.userId },
      data: { passwordHash: newHash },
    });
    // FE-034: password_change is security-critical — must be auditable.
    const audit = await writeAuditLog({
      user,
      action: "password_change",
      resource: user.userId,
      critical: true,
    });
    // FE-004 ROOT FIX (v2): ALWAYS revoke all refresh tokens after a
    // password change — not just when the audit log fails. This is the
    // OWASP-recommended pattern: a password change MUST invalidate every
    // outstanding session. Two threat models require this:
    //   (a) Attacker stole the password and changed it → victim's old
    //       refresh cookies (still valid for 30 days) would otherwise
    //       keep working, giving the attacker a persistent back door
    //       even after the victim recovers the account.
    //   (b) User changes password because they suspect compromise →
    //       the attacker's stolen refresh tokens would otherwise keep
    //       working for 30 days, defeating the purpose of the change.
    // The previous code only revoked on audit-log failure, which is the
    // INVERSE of safe behavior — it revoked exactly when the password
    // update had already happened but audit failed, leaving the
    // common-path (audit succeeds) WIDE OPEN.
    //
    // BE-016 ROOT FIX: previously, a revocation failure here was SILENTLY
    // SWALLOWED — the user saw "Password updated. All other sessions have
    // been signed out" when in fact the other sessions were NOT signed
    // out (DB error, network timeout, etc.). A user who changed their
    // password due to suspected compromise was given a FALSE sense of
    // security; the attacker's stolen refresh token kept working for up
    // to 30 days. Root fix: surface the failure as a `warning` field in
    // the response AND write a critical audit log entry. We still clear
    // the current session's cookies (so the user must re-authenticate
    // with the new password locally), but the warning tells them other
    // sessions may still be active.
    let revocationWarning: string | null = null;
    try {
      await revokeAllRefreshTokensForUser(user.userId);
    } catch (revErr) {
      revocationWarning =
        "Your password was changed and this session has been signed out, " +
        "but other sessions (e.g. on another browser, or a stolen refresh " +
        "token) could NOT be revoked due to a server error. They will " +
        "remain active for up to 30 days. If you suspect compromise, " +
        "please contact support immediately to force-revoke all sessions.";
      console.error("[password] refresh-token revocation failed", revErr);
      // Critical audit log — this is a security-relevant event.
      try {
        await writeAuditLog({
          user,
          action: "password_change_revocation_failed",
          resource: user.userId,
          metadata: {
            error: revErr instanceof Error ? revErr.message : String(revErr),
          },
          critical: true,
        });
      } catch {
        console.error("[password] CRITICAL: revocation failed AND audit log failed");
      }
    }
    // Clear the current session's cookies so the user is forced to
    // re-authenticate with the new password. This is the user-facing
    // signal that the password change took effect across all sessions.
    await clearAuthCookies();
    if (!audit.ok) {
      // The password WAS changed AND sessions were revoked, but the
      // audit log failed. Inform the user — they need to log in again.
      return internalError("Password changed and all sessions revoked, but the audit log failed. Please log in again with your new password.");
    }
    return NextResponse.json(
      revocationWarning
        ? {
            ok: true,
            warning: revocationWarning,
            message:
              "Password updated. This session has been signed out — please " +
              "log in again. See the warning field for revocation status.",
          }
        : {
            ok: true,
            message:
              "Password updated. All other sessions have been signed out — please log in again.",
          }
    );
  } catch (e) {
    console.error("Password update failed:", e);
    return internalError("Failed to update password.");
  }
}
