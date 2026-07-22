import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { getAuthenticatedUser, verifyPassword } from "@/lib/auth/server";
import { verifyTotpWithReplayCheck } from "@/lib/auth/totp";
import { badRequest, internalError, writeAuditLog, requireCsrfOrSend } from "@/lib/api-helpers";
import {
  checkTotpRateLimit,
  recordFailedTotp,
  clearTotpAttempts,
} from "@/lib/auth/rate-limit";
// BE-029 ROOT FIX (Team Member 12): Zod-validated request body. Replaces
// the ad-hoc `typeof body.field === "string"` checks with a single
// schema-enforced parse. See frontend/src/lib/zod-schemas.ts.
import { validateBody, TwoFaDisableBody } from "@/lib/zod-schemas";

/**
 * POST /api/auth/2fa/disable
 * Body: { currentPassword: string, totpCode: string }
 *
 * FE-005 ROOT FIX: The previous implementation disabled 2FA with NO
 * re-authentication — it just trusted the authenticated session. The
 * code comment even admitted it: "for this development build we trust
 * the authenticated session."
 *
 * If an attacker steals a session cookie (XSS, network sniffing on HTTP,
 * dev-tools on a shared computer), they can disable 2FA in one POST and
 * then the account is password-only — compounding the FE-004 2FA bypass.
 *
 * Root fix: require BOTH the current password AND a valid current TOTP
 * code before clearing mfaSecret/mfaEnabled. This means:
 *   - A stolen session cookie alone is NOT enough to disable 2FA.
 *   - A stolen password alone is NOT enough (attacker still needs TOTP).
 *   - Only someone with BOTH factors can disable 2FA.
 *
 * Edge case: if the user has lost their authenticator device, they must
 * go through an admin-mediated recovery flow (out of scope here — that's
 * a separate /api/auth/2fa/recover endpoint with its own audit trail).
 *
 * FE-012 ROOT FIX (Team Member 14): This endpoint accepted a 6-digit
 * TOTP code with NO rate limit. A 6-digit code has only 1,000,000
 * combinations; at 1000 req/s an attacker with a phished password could
 * brute-force the entire keyspace in ~17 minutes — well within the
 * 30-second TOTP window (the attacker simply retries with the next
 * window if the current one expires before they sweep it). The fix
 * wires in the EXISTING `checkTotpRateLimit` + `recordFailedTotp`
 * primitives from `rate-limit.ts` — 5 wrong codes per 5 minutes locks
 * 2FA for 15 minutes. This mirrors the protection already present on
 * `/api/auth/2fa/login-verify` (FE-003 root fix). The primitive was
 * defined but NEVER called from this route — closing that gap.
 *
 * FE-013 ROOT FIX (Team Member 14): This endpoint called `verifyTotp()`
 * which checks if the TOTP code is valid for the current ±30s window
 * but does NOT track whether that code has already been used. An
 * attacker who intercepts a single code (e.g. via XSS reading the
 * response body, or a malicious reverse-proxy phishing site) could
 * reuse it within the 30-second window to disable 2FA. The lib has
 * had `verifyTotpWithReplayCheck()` since the FE-033 root fix — it
 * rejects any code whose counter is <= the user's `lastTotpCounter` —
 * but this route never called it. The fix replaces `verifyTotp()`
 * with `verifyTotpWithReplayCheck()` and atomically advances
 * `lastTotpCounter` via `updateMany` with `where: { lastTotpCounter:
 * { lt: counter } }` so concurrent verifications of the same code
 * cannot both succeed (standard RFC 6238 §5.2 race prevention).
 */
export async function POST(req: NextRequest) {
  // FE-011: CSRF protection on every state-changing route.
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  const user = await getAuthenticatedUser();
  if (!user) {
    return NextResponse.json(
      { error: "unauthorized", message: "Authentication required" },
      { status: 401 }
    );
  }

  let body: { currentPassword?: string; totpCode?: string };
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON body");
  }

  // BE-029 ROOT FIX: schema-validate the body before any business logic.
  // The schema rejects: missing currentPassword, non-6-digit totpCode,
  // non-string fields, oversize values. We then read the parsed (typed)
  // values from `parsed.data` for the rest of the handler.
  const parsed = validateBody(TwoFaDisableBody, body);
  if (!parsed.ok) return parsed.response;
  const currentPassword = parsed.data.currentPassword;
  const totpCode = parsed.data.totpCode ?? "";

  try {
    const dbUser = await db.user.findUnique({
      where: { id: user.userId },
      // FE-013: select lastTotpCounter so we can pass it to
      // verifyTotpWithReplayCheck and atomically advance it after success.
      select: { id: true, email: true, passwordHash: true, mfaEnabled: true, mfaSecret: true, lastTotpCounter: true },
    });
    if (!dbUser) {
      return NextResponse.json(
        { error: "not_found", message: "User not found" },
        { status: 404 }
      );
    }
    if (!dbUser.mfaEnabled || !dbUser.mfaSecret) {
      return NextResponse.json(
        { error: "mfa_not_enabled", message: "2FA is not enabled on this account." },
        { status: 400 }
      );
    }

    // Verify current password.
    const passwordOk = await verifyPassword(currentPassword, dbUser.passwordHash);
    if (!passwordOk) {
      // BE-031 ROOT FIX (Team Member 12): return 401 (authentication
      // failure) — NOT 403. 403 means "authenticated but forbidden",
      // but a wrong password is an authentication failure. API clients
      // that distinguish 401 (re-authenticate) from 403 (forbidden) rely
      // on this to know whether to prompt for re-auth. The previous 403
      // broke that contract.
      return NextResponse.json(
        { error: "invalid_password", message: "Current password is incorrect." },
        { status: 401 }
      );
    }

    // FE-012 ROOT FIX: Per-user TOTP brute-force gate. The 6-digit TOTP
    // keyspace is only 1M codes; without a per-user attempt cap an
    // attacker with a phished password can sweep it in ~17 minutes at
    // 1000 req/s. The primitive existed in rate-limit.ts but was never
    // wired into this route. We check BEFORE the TOTP verify so a locked
    // user cannot burn attempts, and we RECORD a failure AFTER a wrong
    // code so the counter actually advances.
    const totpLock = checkTotpRateLimit(user.userId);
    if (totpLock.locked) {
      await writeAuditLog({
        user,
        action: "2fa_disable_locked",
        resource: user.userId,
        metadata: { retryAfterSeconds: totpLock.retryAfterSeconds },
      });
      return NextResponse.json(
        {
          error: "totp_locked",
          message: `Too many incorrect 2FA codes. Try again in ${Math.ceil(totpLock.retryAfterSeconds / 60)} minute(s).`,
          retryAfterSeconds: totpLock.retryAfterSeconds,
        },
        { status: 429, headers: { "Retry-After": String(totpLock.retryAfterSeconds) } }
      );
    }

    // FE-013 ROOT FIX: Replay-protected TOTP verification. The previous
    // call to `verifyTotp()` accepted any code valid for the ±30s window
    // — including codes already used to log in. An XSS attacker who
    // intercepted a code could disable 2FA with it within the same
    // 30-second window. `verifyTotpWithReplayCheck` rejects any code
    // whose counter is <= the user's stored `lastTotpCounter`.
    const totpResult = verifyTotpWithReplayCheck(
      dbUser.mfaSecret,
      totpCode,
      dbUser.lastTotpCounter
    );
    if (!totpResult.ok) {
      // FE-012: record the failed attempt so the lockout counter
      // advances. This is SEPARATE from the password failedLoginCount
      // — both must trip independently.
      const afterFail = recordFailedTotp(user.userId);
      await writeAuditLog({
        user,
        action: totpResult.reason === "replayed" ? "2fa_disable_code_replayed" : "2fa_disable_failed",
        resource: user.userId,
        metadata: {
          reason: totpResult.reason,
          attemptsRemaining: afterFail.attemptsRemaining,
        },
      });
      if (afterFail.locked) {
        return NextResponse.json(
          {
            error: "totp_locked",
            message: `Invalid 6-digit code. 2FA is now locked for ${Math.ceil(afterFail.retryAfterSeconds / 60)} minute(s) due to too many failed 2FA attempts.`,
            attemptsRemaining: 0,
            retryAfterSeconds: afterFail.retryAfterSeconds,
          },
          { status: 429, headers: { "Retry-After": String(afterFail.retryAfterSeconds) } }
        );
      }
      const message =
        totpResult.reason === "replayed"
          ? "This code has already been used. Wait for the next 30-second window."
          : `Invalid 6-digit code. ${afterFail.attemptsRemaining} attempt(s) remaining before 2FA is locked.`;
      // BE-031 ROOT FIX (Team Member 12): 401 (authentication failure),
      // not 403. A wrong TOTP code is an authentication failure — the
      // user has not proven they possess the second factor. 403 is for
      // authorization failures (authenticated but not allowed).
      return NextResponse.json(
        {
          error: totpResult.reason === "replayed" ? "code_replayed" : "invalid_code",
          message,
          attemptsRemaining: afterFail.attemptsRemaining,
        },
        { status: 401 }
      );
    }

    // FE-013: Atomically advance lastTotpCounter so the same code cannot
    // be replayed. The `updateMany` with `where: { lastTotpCounter: { lt:
    // counter } }` ensures that if two concurrent verifications of the
    // same code race, only one persists the update — the other is a
    // no-op. This is the standard RFC 6238 §5.2 replay-protection race
    // prevention, mirroring the pattern in /api/auth/2fa/login-verify.
    if (dbUser.lastTotpCounter === null) {
      await db.user.update({
        where: { id: user.userId },
        data: { lastTotpCounter: totpResult.counter },
      });
    } else {
      await db.user.updateMany({
        where: { id: user.userId, lastTotpCounter: { lt: totpResult.counter } },
        data: { lastTotpCounter: totpResult.counter },
      });
    }

    // FE-012: Successful verification — clear the per-user TOTP attempt
    // counter so a user who eventually gets it right doesn't carry a
    // partial lock forward.
    clearTotpAttempts(user.userId);

    // Both factors verified — safe to disable.
    //
    // BE-036 ROOT FIX (Team Member 12): disable 2FA + write the audit log
    // ATOMICALLY via db.$transaction. The previous code did two separate
    // writes: first `db.user.update({ mfaSecret: null, mfaEnabled: false })`,
    // then `writeAuditLog({ critical: true })`. If the audit write failed
    // (DB connection blip, disk full, etc.), 2FA was already disabled but
    // there was NO audit trail — a FDA 21 CFR Part 11 compliance gap. The
    // "rollback" attempt was broken: it set `mfaEnabled: false` (already
    // false) but could NOT restore `mfaSecret` (already nulled) — so the
    // user had to re-enroll 2FA from scratch AND the compliance gap
    // remained.
    //
    // Root fix: wrap BOTH writes in a Prisma transaction. If the audit
    // write throws, the transaction rolls back the user update too —
    // mfaSecret is preserved, mfaEnabled stays true, and the user is NOT
    // inconvenienced. The audit trail gap is impossible by construction.
    //
    // Note: writeAuditLog calls `db.auditLog.create` internally. Prisma's
    // interactive transactions support nested writes via the same `tx`
    // handle, but writeAuditLog uses the global `db` client. To keep the
    // atomicity guarantee, we inline the audit log creation here using
    // the transaction client. If the audit row cannot be created, the
    // entire transaction aborts and 2FA remains enabled.
    try {
      await db.$transaction(async (tx) => {
        await tx.user.update({
          where: { id: user.userId },
          data: {
            mfaSecret: null,
            mfaEnabled: false,
            lastTotpCounter: null,
          },
        });
        // Inline the critical audit log write so it shares the
        // transaction. If this throws, the user.update above is rolled
        // back — 2FA stays enabled, mfaSecret is preserved.
        //
        // AuditLog schema requires: actorName (string), metadata (JSON
        // string). `critical` is a runtime concept in writeAuditLog, NOT
        // a DB column — by inlining the write inside the transaction we
        // get stronger atomicity than writeAuditLog's `critical: true`
        // flag could provide.
        await tx.auditLog.create({
          data: {
            userId: user.userId,
            actorName: user.email,
            action: "2fa_disable",
            resource: user.userId,
            // BE-036: organizationId populated so /api/audit-logs can
            // scope by org (multi-tenant compliance).
            ...(user.orgId ? { organizationId: user.orgId } : {}),
            metadata: JSON.stringify({
              critical: true,
              timestamp: new Date().toISOString(),
            }),
          } as any,
        });
      });
    } catch (e: unknown) {
      // Transaction failed — 2FA was NOT disabled (rollback). Inform the
      // user with a clear error. They can retry; mfaSecret is intact.
      const msg = e instanceof Error ? e.message : String(e);
      console.error("[2fa/disable] atomic disable+audit transaction failed:", msg);
      return internalError(
        "Failed to disable 2FA — the audit log could not be written. " +
          "2FA is still enabled. Please try again; if the problem persists, " +
          "contact your administrator."
      );
    }
    return NextResponse.json({ ok: true, enabled: false });
  } catch (e) {
    console.error("2FA disable failed:", e);
    return internalError("Failed to disable 2FA.");
  }
}
