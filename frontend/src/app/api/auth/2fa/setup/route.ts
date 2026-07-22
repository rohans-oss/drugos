import { NextResponse, NextRequest } from "next/server";
import { db } from "@/lib/db";
import { getAuthenticatedUser } from "@/lib/auth/server";
import { generateTotpSecret, buildOtpAuthUri } from "@/lib/auth/totp";
import { internalError, requireCsrfOrSend } from "@/lib/api-helpers";
import { issue2faSetupToken } from "@/lib/auth/two-factor-setup-token";

/**
 * POST /api/auth/2fa/setup
 *
 * Generates a brand-new TOTP secret for the authenticated user and returns
 * it (along with an otpauth:// URI) so the client can display a QR code.
 *
 * FE-071 ROOT FIX: The previous version returned the TOTP secret in
 * plaintext with no setup token. If any XSS exists in the app, an
 * attacker could read the secret and call /verify themselves to persist
 * 2FA under their control — permanent account compromise.
 *
 * Root fix: alongside the secret, issue a one-time, 5-minute setup token
 * bound to the user's session. The client must send BOTH the secret AND
 * the setupToken to /verify. The setup token:
 *   - Can only be used ONCE (replay attacks rejected).
 *   - Expires after 5 minutes.
 *   - Is bound to the userId (stolen token cannot enroll 2FA for a
 *     different user).
 *
 * This is defense-in-depth on top of the CSP headers (the primary XSS
 * mitigation, enforced via next.config.ts). The combination closes the
 * XSS → 2FA compromise chain.
 *
 * The secret is NOT persisted yet — the user must call /api/auth/2fa/verify
 * with the code from their authenticator app + the setupToken to confirm
 * they have stored it. Only then do we set `mfaEnabled = true`.
 */
export async function POST(req: NextRequest) {
  // FE-011: CSRF protection on every state-changing route.
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  const user = await getAuthenticatedUser();
  if (!user) {
    return NextResponse.json({ error: "unauthorized", message: "Authentication required" }, { status: 401 });
  }

  try {
    const secret = generateTotpSecret();
    const uri = buildOtpAuthUri({
      issuer: "DrugOS",
      account: user.email,
      secret,
    });
    // FE-071: issue a one-time setup token bound to this user's session.
    // The client must send it back to /verify alongside the secret + code.
    const { setupToken, expiresAt } = issue2faSetupToken(user.userId, secret);
    return NextResponse.json({
      secret,
      otpauthUri: uri,
      setupToken,
      setupTokenExpiresAt: expiresAt,
    });
  } catch (e) {
    console.error("2FA setup failed:", e);
    return internalError("Failed to start 2FA enrollment.");
  }
}
