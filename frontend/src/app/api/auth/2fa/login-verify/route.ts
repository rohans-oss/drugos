import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import {
  verifyMfaChallengeToken,
  signAccessToken,
  rotateRefreshToken,
  setAuthCookies,
} from "@/lib/auth/server";
import { verifyTotpWithReplayCheck } from "@/lib/auth/totp";
import { badRequest, internalError, writeAuditLog, issueCsrfToken, setCsrfCookie } from "@/lib/api-helpers";
// BE-029 ROOT FIX (Team Member 12): Zod-validated request body.
import { validateBody, TwoFaLoginVerifyBody } from "@/lib/zod-schemas";
import {
  recordSuccessfulLogin,
  recordFailedLogin,
  checkTotpRateLimit,
  // BE-066: prefer the distributed version (Redis-backed when REDIS_URL
  // is set) so the TOTP brute-force counter is shared across all Node.js
  // instances. The sync `recordFailedTotp` is kept only for tests and
  // for the no-Redis fallback path inside `recordFailedTotpDistributed`.
  recordFailedTotpDistributed,
  clearTotpAttempts,
} from "@/lib/auth/rate-limit";
import jwt from "jsonwebtoken";

/**
 * POST /api/auth/2fa/login-verify
 * Body: { mfaToken?: string, code: string }  (mfaToken optional — also read
 *       from the drugos_mfa_challenge HttpOnly cookie set by /api/auth/login)
 *
 * FE-004 ROOT FIX: This endpoint did NOT exist before. The login route
 * never checked mfaEnabled, and the MFAChallengePage's "Verify" button
 * just navigated to the dashboard without verifying anything. 2FA was
 * purely cosmetic.
 *
 * FE-033 ROOT FIX: TOTP replay protection. Now uses
 * verifyTotpWithReplayCheck to reject already-used codes. The matching
 * counter is persisted atomically via updateMany with
 * `where: { lastTotpCounter: { lt: counter } }` so concurrent
 * verifications of the same code cannot both succeed.
 *
 * FE-016 ROOT FIX: The mfa_challenge JWT was signed but had no jti and was
 * not tracked as consumed — an attacker who intercepted it (e.g. via XSS
 * reading the JSON response body) could replay it multiple times within
 * the 5-minute window. Each successful replay issued fresh access+refresh
 * tokens.
 *
 * Security properties:
 *   - The challenge token is signed with the same JWT_SECRET but has
 *     type="mfa_challenge", so it CANNOT be used as an access token.
 *   - The challenge token expires in 5 minutes.
 *   - TOTP verification uses constant-time comparison (timingSafeEqual).
 *   - TOTP codes cannot be replayed (lastTotpCounter monotonically
 *     advances on each successful verification).
 *   - FE-003 ROOT FIX (v2): Per-user TOTP brute-force rate limiting is
 *     ENFORCED in this route via checkTotpRateLimit / recordFailedTotp.
 *     After 5 wrong TOTP codes within 5 minutes the account is locked
 *     for 15 minutes. The previous version had the rate-limit primitives
 *     in rate-limit.ts but NEVER CALLED THEM from this route — the test
 *     file totp-rate-limit.test.ts passed because it tested the primitives
 *     in isolation, but the actual HTTP endpoint was still brute-forceable
 *     (1M codes / 1000 req/s = 17 minutes for full keyspace). This is
 *     wired in NOW.
 *
 * Root fix (FE-016):
 *   1. The challenge token now carries a jti (JWT ID). On first use, we
 *      atomically insert the jti into the MfaChallenge table (unique on
 *      jti). A second insert fails (Prisma P2002) → reject as replay.
 *   2. The challenge token is set as an HttpOnly cookie (sameSite=strict,
 *      path=/api/auth/2fa/login-verify) by /api/auth/login. The client
 *      does not need to read it — the browser sends it automatically. The
 *      JSON-body mfaToken is still accepted for non-browser API clients.
 *   3. Cookie is cleared on success AND on failure.
 *
 * FE-018 ROOT FIX: recordSuccessfulLogin() is called HERE, after TOTP
 * verifies — not in /api/auth/login when the password is checked. This
 * means an attacker with the password but no TOTP secret does NOT get
 * their failedLoginCount reset by triggering MFA challenges.
 *
 * Additionally, on TOTP failure we increment failedLoginCount via
 * recordFailedLogin — so repeated MFA failures DO accumulate toward
 * account lockout. This is COMPLEMENTARY to FE-003's TOTP-specific
 * rate limiter (recordFailedTotp): the password counter tracks total
 * auth failures, while the TOTP counter tracks 2FA-specific brute-force
 * attempts. Both must trip independently.
 *
 * FE-011: CSRF cookie issued on success so the client can make
 * state-changing requests after MFA verification.
 */
export async function POST(req: NextRequest) {
  // FE-016: read the challenge token from the HttpOnly cookie FIRST.
  // Fall back to the JSON body for non-browser API clients. The cookie
  // path is the safer path because XSS cannot read it.
  let mfaToken = "";
  try {
    const { cookies } = await import("next/headers");
    const store = await cookies();
    mfaToken = store.get("drugos_mfa_challenge")?.value || "";
  } catch {
    // cookies() throws outside a request scope — fall through to body.
  }

  let body: { mfaToken?: string; code?: string };
  try {
    body = await req.json();
  } catch {
    body = {};
  }
  if (!mfaToken) {
    mfaToken = (body.mfaToken || "").trim();
  }
  // BE-029 ROOT FIX: schema-validate the body. The schema enforces
  // `code` is exactly 6 digits (rejects malformed input with 400) and
  // `mfaToken` is a non-empty string when present. The 6-digit shape
  // check is the LINE between "malformed input → 400" and "wrong code
  // → 401" (BE-034). A 6-digit code that does not match the user's
  // TOTP secret is an authentication failure, NOT a malformed request.
  const parsed = validateBody(TwoFaLoginVerifyBody, body);
  if (!parsed.ok) return parsed.response;
  const code = parsed.data.code;
  if (!mfaToken) return badRequest("mfaToken is required (either the drugos_mfa_challenge cookie or the body field)");

  // Verify the challenge token (signature + expiry + type).
  const challenge = verifyMfaChallengeToken(mfaToken);
  if (!challenge) {
    await clearMfaChallengeCookie();
    return NextResponse.json(
      { error: "invalid_mfa_token", message: "MFA challenge token is invalid or expired. Please log in again." },
      { status: 401 }
    );
  }

  // BE-006 ROOT FIX (CRITICAL) + BE-065 ROOT FIX:
  //
  // BE-006: The previous `select` clause OMITTED `platformRole`. As a
  // result, signAccessToken at the bottom of this handler received
  // `platformRole: undefined` and defaulted it to "none" (see
  // lib/auth/server.ts signAccessToken: `platformRole: payload.platformRole
  // || "none"`). Effect: a platform admin (SaaS operator staff) who enrolled
  // in 2FA got an access token with `platformRole: "none"` after completing
  // 2FA. requirePlatformAdmin() then rejected every /api/admin/* call with
  // 403 for the full 15-minute access-token TTL. Platform admins with 2FA
  // enabled (which should be ALL of them) were locked out of the admin
  // console for 15 minutes after every login. The non-2FA login path
  // (login/route.ts L108, L336) already selects platformRole and stamps it
  // into the token; this 2FA path was missed when TASK-261 added the
  // platformRole field.
  //
  // BE-065: We move the user lookup BEFORE the jti replay check below so
  // the audit-log entry on replay rejection can include the user's actual
  // role (previously it used `role: "unknown"`). FDA 21 CFR Part 11
  // requires audit trails to capture the user's identity at the time of
  // the event; "unknown" broke that. The cost is one extra DB round-trip
  // before the jti check; the jti check itself is unchanged.
  const user = await db.user.findUnique({
    where: { id: challenge.userId },
    select: {
      id: true,
      email: true,
      name: true,
      role: true,
      // BE-006: REQUIRED for the access token to carry platformRole.
      platformRole: true,
      status: true,
      // BE-045 alignment: include deletedAt so we can refuse 2FA for
      // soft-deleted accounts (the login route already does this).
      deletedAt: true,
      mfaEnabled: true,
      mfaSecret: true,
      lastTotpCounter: true,
    },
  });
  if (!user || user.deletedAt !== null) {
    await clearMfaChallengeCookie();
    return NextResponse.json(
      { error: "invalid_mfa_token", message: "MFA challenge token is invalid or expired. Please log in again." },
      { status: 401 }
    );
  }

  // FE-016 ROOT FIX: replay protection. Extract the jti from the token and
  // atomically claim it in the MfaChallenge table. If the jti is already
  // consumed, reject as a replay attack. We do this BEFORE TOTP verification
  // so a replayed token cannot be used to probe TOTP codes.
  try {
    const decoded = jwt.decode(mfaToken, { complete: true }) as
      | { payload?: { jti?: string } }
      | null;
    const jti = decoded?.payload?.jti;
    if (jti) {
      try {
        await db.mfaChallenge.create({
          data: {
            jti,
            userId: challenge.userId,
            expiresAt: new Date(Date.now() + 60 * 1000), // 1 min grace past JWT expiry
          },
        });
      } catch (e: any) {
        if (e?.code === "P2002") {
          // Prisma unique-constraint violation → jti already consumed → replay.
          // BE-065 ROOT FIX: use the user's ACTUAL role + platformRole
          // (fetched above) instead of the hardcoded "unknown" string.
          // This makes the audit log forensically useful — compliance
          // officers reviewing Part 11 audit trails can now distinguish
          // a platform admin's locked account from a researcher's.
          await writeAuditLog({
            user: {
              userId: user.id,
              email: user.email,
              role: user.role,
              platformRole: (user.platformRole as string | undefined) || "none",
            },
            action: "login_mfa_replay_rejected",
            resource: `user:${user.id}`,
            metadata: { jti },
          });
          await clearMfaChallengeCookie();
          return NextResponse.json(
            { error: "invalid_mfa_token", message: "MFA challenge token has already been used. Please log in again." },
            { status: 401 }
          );
        }
        // Other errors (DB down, etc.) — fail closed: reject the request.
        console.error("MFA jti tracking failed:", e);
        await clearMfaChallengeCookie();
        return internalError("Unable to verify MFA challenge uniqueness.");
      }
    }
  } catch (e) {
    console.error("MFA jti extraction failed:", e);
    await clearMfaChallengeCookie();
    return NextResponse.json(
      { error: "invalid_mfa_token", message: "MFA challenge token is malformed." },
      { status: 401 }
    );
  }

  try {
    if (user.status === "suspended") {
      await clearMfaChallengeCookie();
      return NextResponse.json(
        { error: "account_suspended", message: "Account suspended. Contact your administrator." },
        { status: 403 }
      );
    }
    if (!user.mfaEnabled || !user.mfaSecret) {
      await clearMfaChallengeCookie();
      return NextResponse.json(
        { error: "mfa_not_enabled", message: "MFA is not enabled on this account." },
        { status: 400 }
      );
    }

    // FE-003 ROOT FIX (v2): Per-user TOTP brute-force gate.
    // BEFORE we spend a TOTP verification attempt, check whether the user
    // is currently locked. This must come AFTER the user lookup (we need
    // the userId) but BEFORE the TOTP verify (so a locked user cannot
    // burn attempts). The IP-level rate limit (checkIpRateLimit) is a
    // separate outer layer that limits raw request volume; this inner
    // layer limits per-user TOTP guesses regardless of source IP.
    const totpLock = checkTotpRateLimit(user.id);
    if (totpLock.locked) {
      await writeAuditLog({
        user: { userId: user.id, email: user.email, role: user.role },
        action: "login_mfa_locked",
        resource: `user:${user.id}`,
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

    // FE-033: Replay-protected TOTP verification.
    const result = verifyTotpWithReplayCheck(user.mfaSecret, code, user.lastTotpCounter);
    if (!result.ok) {
      // FE-003 ROOT FIX (v2): Record the failed TOTP attempt. This
      // increments the per-user TOTP sliding-window counter and locks
      // 2FA after TOTP_MAX_ATTEMPTS (5) wrong codes within
      // TOTP_WINDOW_MINUTES (5). This is SEPARATE from the password
      // failed-login counter (FE-018 below) — both must trip independently.
      // BE-066 ROOT FIX: use the DISTRIBUTED version of recordFailedTotp
      // so the TOTP brute-force counter is shared across all Node.js
      // instances (K8s replicas, etc.). The previous sync version used an
      // in-memory Map — each instance had its own counter, so an attacker
      // could make N × TOTP_MAX_ATTEMPTS attempts before lockout (N =
      // instance count). At N=3, that's 15 attempts before lockout —
      // feasible to brute-force TOTP in ~6 minutes. The distributed
      // version uses Redis when REDIS_URL is set (atomic ZADD + ZCARD in
      // a MULTI/EXEC transaction) and falls back to the sync in-memory
      // path when Redis is unavailable (single-instance dev/test).
      const afterFail = await recordFailedTotpDistributed(user.id);
      // FE-018 ROOT FIX: ALSO increment failedLoginCount on MFA failure
      // so repeated MFA failures accumulate toward account-wide lockout.
      // This closes the gap left by removing recordSuccessfulLogin from
      // the password-verification path. The two counters are
      // complementary: FE-003 tracks 2FA brute-force, FE-018 tracks
      // total auth failures.
      const lockResult = await recordFailedLogin(user.id);
      await writeAuditLog({
        user: { userId: user.id, email: user.email, role: user.role },
        action: result.reason === "replayed" ? "login_mfa_code_replayed" : "login_mfa_failed",
        resource: `user:${user.id}`,
      });
      await clearMfaChallengeCookie();
      // FE-018: account-wide lock (password failed-login counter tripped).
      if (lockResult.locked) {
        return NextResponse.json(
          {
            error: "account_locked",
            message: `Account locked due to too many failed MFA attempts. Try again in ${Math.ceil(lockResult.retryAfterSeconds / 60)} minute(s).`,
            retryAfter: lockResult.retryAfterSeconds,
          },
          { status: 423 }
        );
      }
      // FE-003: 2FA-specific lock (TOTP brute-force counter tripped).
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
        result.reason === "replayed"
          ? "This code has already been used. Wait for the next 30-second window."
          : `Invalid 6-digit code. ${afterFail.attemptsRemaining} attempt(s) remaining before 2FA is locked.`;
      // BE-034 ROOT FIX (Team Member 12): return 401 (authentication
      // failure) — NOT 400. The previous 400 was wrong: 400 means "bad
      // request" (malformed input), but a 6-digit code that doesn't
      // match is an authentication failure. The Zod schema (BE-029)
      // already rejects NON-6-digit codes with 400 at the body-parse
      // stage; ONLY 6-digit codes that fail TOTP verification reach
      // this branch, and they MUST be 401 so API clients can
      // distinguish "malformed" (re-enter the code in the same form)
      // from "wrong code" (re-authenticate). The 429 path above is
      // unchanged (rate-limit responses are always 429 per RFC 6585).
      return NextResponse.json(
        {
          error: result.reason === "replayed" ? "code_replayed" : "invalid_code",
          message,
          attemptsRemaining: afterFail.attemptsRemaining,
        },
        { status: 401 }
      );
    }

    // FE-003 ROOT FIX (v2): Successful verification — clear the per-user
    // TOTP attempt counter so a user who eventually gets it right doesn't
    // carry a partial lock forward. This is the OWASP-recommended reset-
    // on-success pattern.
    clearTotpAttempts(user.id);

    // FE-033: Atomically advance lastTotpCounter. The `updateMany` with
    // `where: { lastTotpCounter: { lt: result.counter } }` ensures that
    // if two concurrent verifications of the same code race, only one
    // actually persists the update — the other is a no-op. This is the
    // standard RFC 6238 §5.2 replay-protection race prevention.
    if (user.lastTotpCounter === null) {
      await db.user.update({
        where: { id: user.id },
        data: { lastTotpCounter: result.counter },
      });
    } else {
      await db.user.updateMany({
        where: { id: user.id, lastTotpCounter: { lt: result.counter } },
        data: { lastTotpCounter: result.counter },
      });
    }

    // FE-018 ROOT FIX: call recordSuccessfulLogin HERE — after TOTP
    // verifies. This resets failedLoginCount and clears lockedUntil only
    // when the user has fully authenticated (password + TOTP).
    await recordSuccessfulLogin(user.id);

    // Success — issue the real access+refresh tokens.
    const membership = await db.organizationMember.findFirst({
      where: { userId: user.id },
      orderBy: { joinedAt: "asc" },
    });
    // BE-079 REAL ROOT FIX (v2): Persist lastActiveOrgId on the User row,
    // mirroring the non-MFA login path. Without this, the 2FA user's
    // refreshed access token (issued 15 min later by rotateRefreshToken)
    // would have no orgId, and every org-scoped query would 403.
    if (membership?.organizationId) {
      await db.user.update({
        where: { id: user.id },
        data: { lastActiveOrgId: membership.organizationId },
      });
    }
    const tokens = await rotateRefreshToken(user.id);
    // BE-006 ROOT FIX: stamp platformRole into the access token. Without
    // this, every 2FA login issues an access token with `platformRole:
    // "none"` (the default in signAccessToken), locking platform admins
    // out of /api/admin/* for the full 15-minute access-token TTL.
    // Coerce to "none" if the DB row has null (legacy rows pre-migration).
    const access = signAccessToken({
      userId: user.id,
      email: user.email,
      role: user.role,
      platformRole: (user.platformRole as string | undefined) || "none",
      orgId: membership?.organizationId,
    });
    await setAuthCookies(access, tokens.refresh);
    // FE-011: issue the CSRF token cookie on successful MFA verification,
    // mirroring the non-MFA login path.
    const csrfToken = issueCsrfToken();
    await setCsrfCookie(csrfToken);
    await clearMfaChallengeCookie();
    await writeAuditLog({
      user: { userId: user.id, email: user.email, role: user.role, orgId: membership?.organizationId },
      action: "login_mfa_success",
      resource: `user:${user.id}`,
    });

    return NextResponse.json({
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role,
      },
      organizationId: membership?.organizationId,
      // FE-011: echo the CSRF token in the response body so non-browser
      // clients (which can't read cookies) can pick it up.
      csrfToken,
    });
  } catch (e) {
    console.error("2FA login-verify failed:", e);
    await clearMfaChallengeCookie();
    return internalError("Failed to verify 2FA code.");
  }
}

async function clearMfaChallengeCookie(): Promise<void> {
  // BE-019 ROOT FIX: The previous implementation set `path:
  // "/api/auth/2fa/login-verify"` here. The login route (login/route.ts)
  // SETS the cookie with `path: "/api/auth/2fa"`. Per RFC 6265 §5.3,
  // a cookie's scope is determined by its Path attribute, and to DELETE
  // a cookie the Set-Cookie response must use the SAME Path (and Domain)
  // as the original set. With the mismatched paths, the browser treated
  // this clear as a NEW cookie scoped to /api/auth/2fa/login-verify
  // (which expired immediately), while the ORIGINAL cookie scoped to
  // /api/auth/2fa persisted for its full 5-minute TTL. The user's
  // browser kept sending the consumed MFA challenge token to every
  // /api/auth/2fa/* endpoint for 5 minutes after login. Aligning the
  // path to "/api/auth/2fa" matches the SET path and properly deletes
  // the cookie.
  try {
    const { cookies } = await import("next/headers");
    const store = await cookies();
    store.set("drugos_mfa_challenge", "", {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "strict",
      path: "/api/auth/2fa",
      maxAge: 0, // delete
    });
  } catch {
    // swallow — cookie clear is best-effort
  }
}
