import { NextRequest, NextResponse } from "next/server";
import {
  clearAuthCookies,
  getAuthenticatedUser,
  revokeAllRefreshTokensForUser,
  REFRESH_COOKIE,
} from "@/lib/auth/server";
import { writeAuditLog, requireCsrfOrSend, clearCsrfCookie } from "@/lib/api-helpers";
import { cookies } from "next/headers";

/**
 * FE-002 ROOT FIX: Logout MUST revoke refresh tokens server-side.
 *
 * Previously this handler only cleared the browser cookies, leaving the
 * refresh-token row in the DB valid for up to 30 days. Any refresh token
 * captured by XSS, network sniffing, or shared-computer browser cache
 * remained usable via /api/auth/refresh, giving the attacker a persistent
 * back door even after the victim logged out.
 *
 * Fix: after writing the audit log we call revokeAllRefreshTokensForUser()
 * to invalidate every outstanding refresh token for the user. This is the
 * OWASP-recommended pattern — logout is a server-side state change, not
 * just a cookie wipe. We also defensively read the raw refresh-cookie
 * value before clearing so future extensions can revoke that specific
 * token (currently redundant with the user-wide revocation but provides
 * defense-in-depth if the user object ever fails to resolve).
 */
export async function POST(req: NextRequest) {
  // FE-011: CSRF protection on every state-changing route.
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  const user = await getAuthenticatedUser();

  // Read the raw refresh-cookie value BEFORE clearing cookies. The cookie
  // store is mutated by clearAuthCookies() below, so we must snapshot now.
  const store = await cookies();
  const refreshCookieValue = store.get(REFRESH_COOKIE)?.value;

  // BE-015 ROOT FIX: hoist the warning to function scope so the final
  // response can include it regardless of which branch set it.
  let revocationWarning: string | null = null;

  if (user) {
    await writeAuditLog({
      user,
      action: "logout",
      resource: `user:${user.userId}`,
    });
    // Revoke every outstanding refresh token for this user — not just the
    // current one. A compromised sibling session (different browser,
    // stolen token) must also be terminated.
    //
    // BE-015 ROOT FIX: previously, a revocation failure (DB down, network
    // timeout, connection pool exhaustion) was SILENTLY SWALLOWED — the
    // user saw `{ ok: true }` and believed all sessions were ended, when
    // in fact a stolen sibling refresh token kept working for up to 30
    // days. A user who suspects compromise and clicks "Log Out" was given
    // a FALSE sense of security.
    //
    // Root fix: surface the failure as a `warning` field in the response
    // AND write a critical audit log entry. The UI MUST display the
    // warning prominently so the user knows to contact support and change
    // their password. We STILL clear cookies (so the user is logged out
    // locally — failing the logout entirely would trap the user in their
    // session, which is worse UX than a partial logout with a warning).
    // The audit log records the failure for compliance (FDA 21 CFR Part 11
    // requires complete audit trails).
    try {
      await revokeAllRefreshTokensForUser(user.userId);
    } catch (err) {
      revocationWarning =
        "Your current session has been ended, but other sessions (e.g. on " +
        "another browser, or a stolen refresh token) could NOT be revoked " +
        "due to a server error. They will remain active for up to 30 days " +
        "unless you change your password. Please contact support if you " +
        "suspect compromise.";
      console.error("[logout] refresh-token revocation failed", err);
      // Critical audit log — this is a security-relevant event.
      try {
        await writeAuditLog({
          user,
          action: "logout_revocation_failed",
          resource: `user:${user.userId}`,
          metadata: {
            error: err instanceof Error ? err.message : String(err),
          },
          critical: true,
        });
      } catch {
        // If even the audit-log write fails, log to stderr as last resort.
        console.error("[logout] CRITICAL: revocation failed AND audit log failed");
      }
    }
  } else if (refreshCookieValue) {
    // Even if we cannot resolve a user from the access token, the refresh
    // cookie may still be valid. The consumeRefreshToken path will both
    // revoke it and rotate it — but we want revocation only. Calling
    // revokeAllRefreshTokensForUser requires a userId, so we look one up
    // indirectly via the refresh-token row. This branch is defensive: it
    // only triggers when the access token is missing/expired but the
    // refresh cookie is still present.
    // We deliberately do NOT rotate the token here.
    try {
      const { db } = await import("@/lib/db");
      const record = await db.refreshToken.findUnique({
        where: { token: refreshCookieValue },
      });
      if (record && !record.revokedAt) {
        await db.refreshToken.update({
          where: { id: record.id },
          data: { revokedAt: new Date() },
        });
      }
    } catch (err) {
      // BE-015: same root fix — surface the failure as a warning. The
      // orphan-token branch is defensive; if it fails, the token will
      // expire naturally (30-day TTL), but the user should be told.
      console.error("[logout] orphan refresh-token revocation failed", err);
      revocationWarning =
        "Your current session has been ended, but an orphaned refresh " +
        "token could not be revoked due to a server error. It will " +
        "expire naturally within 30 days. Contact support if concerned.";
    }
  }

  await clearAuthCookies();
  // FE-011: clear the CSRF cookie too so a re-login gets a fresh token.
  await clearCsrfCookie();
  // BE-015: return a warning field if revocation failed, so the UI can
  // display it prominently. The `ok: true` is preserved for backwards
  // compatibility with clients that only check that field — the warning
  // is the new signal that the logout was partial.
  return NextResponse.json(
    revocationWarning
      ? { ok: true, warning: revocationWarning }
      : { ok: true }
  );
}
