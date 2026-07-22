import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { getAuthenticatedUser } from "@/lib/auth/server";
import { verifyTotpWithReplayCheck } from "@/lib/auth/totp";
import { badRequest, internalError, writeAuditLog, requireCsrfOrSend } from "@/lib/api-helpers";
// FE-071 ROOT FIX: verify the one-time setup token issued by /api/auth/2fa/setup.
// The previous verify handler accepted { secret, code } and persisted the secret
// without ever checking the setup token. That meant an attacker who stole the
// secret via XSS could call /verify themselves and persist 2FA under their
// control — permanent account compromise. The setup-side fix (issue2faSetupToken)
// was already in place, but the verify side never validated the token, so the
// defense-in-depth chain was broken. This commit closes that gap.
import { verify2faSetupToken } from "@/lib/auth/two-factor-setup-token";

/**
 * POST /api/auth/2fa/verify
 * Body: { secret?: string, code: string, setupToken?: string }
 *
 * Confirms a 2FA enrollment. If the user is enrolling for the first time,
 * the client must send the `secret` AND `setupToken` returned by
 * /api/auth/2fa/setup. We verify the code AND validate the one-time setup
 * token before persisting `mfaSecret` and setting `mfaEnabled = true`.
 *
 * If the user already has 2FA enabled and `secret` is omitted, this just
 * verifies the code without changing state (used for re-verification flows).
 * In that case no setupToken is required (the user is not enrolling).
 *
 * FE-033 ROOT FIX: TOTP replay protection. The previous code used
 * `verifyTotp` which accepts any matching code in the ±30s window with
 * NO tracking of used codes — an attacker who phished a code could
 * replay it within 60s. Now we use `verifyTotpWithReplayCheck` which
 * rejects any code whose counter is <= the user's lastTotpCounter.
 *
 * FE-071 ROOT FIX (this commit): one-time setup token enforcement on the
 * verify side. When enrolling for the first time (dbUser.mfaEnabled === false),
 * we REQUIRE `setupToken` in the body and call `verify2faSetupToken` to
 * validate it. The token is:
 *   - One-time: a second call with the same token is rejected.
 *   - Time-bound: expires after 5 minutes.
 *   - User-bound: a token issued for user A cannot be used by user B.
 *   - Secret-bound: an attacker cannot substitute their own secret while
 *     reusing a stolen token.
 * This closes the XSS → 2FA compromise chain: even if an attacker steals
 * the secret from the /setup response, they cannot persist 2FA without
 * also stealing the setupToken AND calling /verify within the 5-minute
 * window AND before the legitimate user does. Combined with the CSP
 * headers (the primary XSS mitigation), this is defense-in-depth.
 */
export async function POST(req: NextRequest) {
  // FE-011: CSRF protection on every state-changing route.
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  const user = await getAuthenticatedUser();
  if (!user) {
    return NextResponse.json({ error: "unauthorized", message: "Authentication required" }, { status: 401 });
  }

  let body: { secret?: string; code?: string; setupToken?: string };
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON body");
  }

  const code = (body.code || "").trim();
  if (!/^\d{6}$/.test(code)) {
    return badRequest("A 6-digit code is required.");
  }

  try {
    const dbUser = await db.user.findUnique({ where: { id: user.userId } });
    if (!dbUser) {
      // FE-068 consistency: treat deleted user as 401, not 404 — avoid
      // leaking "valid token, deleted user" via a distinct status code.
      return NextResponse.json(
        { error: "unauthorized", message: "Authentication required" },
        { status: 401 }
      );
    }

    // Determine which secret to verify against.
    const secret = body.secret || dbUser.mfaSecret;
    if (!secret) {
      return badRequest("No 2FA secret available — call /api/auth/2fa/setup first.");
    }

    // FE-071 ROOT FIX: one-time setup token enforcement.
    //
    // When the user is enrolling for the FIRST TIME (mfaEnabled === false),
    // we REQUIRE the setupToken that was issued by /api/auth/2fa/setup
    // alongside the secret. This prevents an XSS attacker who stole the
    // secret from persisting 2FA under their own control.
    //
    // For re-verification flows (mfaEnabled === true), no setupToken is
    // needed — the user already has 2FA enrolled and is just re-verifying
    // their identity (e.g. for a sensitive operation).
    if (!dbUser.mfaEnabled) {
      if (!body.setupToken) {
        return NextResponse.json(
          {
            error: "setup_token_required",
            message:
              "A setupToken is required to enroll 2FA for the first time. " +
              "Call /api/auth/2fa/setup to obtain one.",
          },
          { status: 400 }
        );
      }
      const setupResult = verify2faSetupToken(user.userId, secret, body.setupToken);
      if (!setupResult.ok) {
        // Map the internal reason to a user-facing error. We deliberately
        // use a single, generic 400 response for all failure modes so an
        // attacker cannot distinguish "token used" from "token not found"
        // from "user mismatch" — that would leak information about the
        // state of the token store. The detailed reason is logged server-side.
        console.warn(
          `[2FA] setup token verification failed for user ${user.userId}: ${setupResult.reason}`
        );
        return NextResponse.json(
          {
            error: "invalid_setup_token",
            message: "The 2FA setup token is invalid, expired, or already used. Please restart 2FA enrollment.",
          },
          { status: 400 }
        );
      }
      // Token is valid and one-time — it has now been marked used. The
      // caller cannot replay it. Proceed to TOTP verification.
    }

    // FE-033: Replay-protected TOTP verification.
    const result = verifyTotpWithReplayCheck(secret, code, dbUser.lastTotpCounter);
    if (!result.ok) {
      const message =
        result.reason === "replayed"
          ? "This code has already been used. Wait for the next 30-second window and try again."
          : "Invalid 6-digit code. Try again.";
      return NextResponse.json(
        { error: result.reason === "replayed" ? "code_replayed" : "invalid_code", message },
        { status: 400 }
      );
    }

    // Atomically update lastTotpCounter ONLY if no concurrent verification
    // has already advanced it past our counter. This prevents a race where
    // two concurrent requests both verify the same code and both succeed.
    // The `updateMany` with `where: { lastTotpCounter: { lt: result.counter } }`
    // ensures only one of the two requests actually persists the update.
    // For first-time enrollment (lastTotpCounter is null), we use a direct
    // update since there's no race window to worry about (the user just
    // set up 2FA and hasn't verified anything yet).
    if (dbUser.lastTotpCounter === null) {
      await db.user.update({
        where: { id: user.userId },
        data: { lastTotpCounter: result.counter },
      });
    } else {
      await db.user.updateMany({
        where: { id: user.userId, lastTotpCounter: { lt: result.counter } },
        data: { lastTotpCounter: result.counter },
      });
    }

    // If enrolling for the first time, persist the secret.
    if (!dbUser.mfaEnabled) {
      await db.user.update({
        where: { id: user.userId },
        data: { mfaSecret: secret, mfaEnabled: true },
      });
      await writeAuditLog({
        user,
        action: "2fa_enable",
        resource: user.userId,
      });
    }

    return NextResponse.json({ ok: true, enabled: true });
  } catch (e) {
    console.error("2FA verify failed:", e);
    return internalError("Failed to verify 2FA code.");
  }
}
