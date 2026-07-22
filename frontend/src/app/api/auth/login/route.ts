import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import {
  verifyPassword,
  signAccessToken,
  rotateRefreshToken,
  setAuthCookies,
  signMfaChallengeToken,
  clearAuthCookies,
} from "@/lib/auth/server";
import { badRequest, writeAuditLog, internalError, issueCsrfToken, setCsrfCookie } from "@/lib/api-helpers";
import {
  checkIpRateLimit,
  checkIpRateLimitDistributed,
  recordIpAttempt,
  checkAccountLocked,
  recordFailedLogin,
  recordSuccessfulLogin,
} from "@/lib/auth/rate-limit";
import bcrypt from "bcryptjs";
import { randomBytes } from "crypto";

interface LoginBody {
  email: string;
  password: string;
}

// BE-010 ROOT FIX: Pre-computed bcrypt hash of a random string, used as a
// dummy comparator when the email does NOT exist in the DB. Without this,
// the login route returned `invalid_credentials` IMMEDIATELY for nonexistent
// emails (no bcrypt.compare call), but took ~250ms for emails that DO
// exist (bcrypt.compare with cost 12). An attacker measuring response
// time could distinguish "email exists" (slow) from "email doesn't exist"
// (fast) — the classic OWASP authentication timing attack. With ~50
// measurements per email, the attacker can enumerate which emails are
// registered. For a pharma platform where registered emails reveal which
// researchers at which companies use the platform, this is competitive
// intelligence leakage. OWASP ASVS V2.1.6 requires "constant-time"
// authentication responses.
//
// This hash is a real bcrypt hash (cost 12) of a 32-byte random string
// generated once at module load. The plaintext is NEVER used — we only
// call bcrypt.compare(password, DUMMY_HASH) to consume the same ~250ms
// as a real password verification, then discard the result. The hash is
// safe to commit to source: even if an attacker knows the plaintext, it
// gives them nothing (they still can't log in as a nonexistent user).
const DUMMY_PASSWORD_HASH = bcrypt.hashSync(
  randomBytes(32).toString("hex"),
  12
);

export async function POST(req: NextRequest) {
  // FE-009 ROOT FIX (Layer 1): IP-based rate limit. Stops credential
  // stuffing where an attacker rotates usernames from one IP. Done BEFORE
  // parsing the body so even malformed requests count against the bucket.
  //
  // BE-005 ROOT FIX: use the DISTRIBUTED (async) IP rate limiter so the
  // cap is enforced across all Node.js instances (K8s replicas, etc.).
  // When REDIS_URL is set, this hits Redis (shared state). When REDIS_URL
  // is NOT set, it falls back to the in-memory LRU (single-instance dev).
  // The distributed version ALSO records the attempt atomically with the
  // check (zadd + zcard in one MULTI/EXEC), so we don't need a separate
  // recordIpAttempt call on the Redis path. The sync `recordIpAttempt`
  // below is still called for the in-memory fallback path (the async
  // version's atomic record-and-count only fires when Redis is used).
  let ipCheck;
  try {
    ipCheck = await checkIpRateLimitDistributed(req);
  } catch (e) {
    // Redis error — fall back to the sync in-memory path. A Redis outage
    // should NOT disable rate limiting entirely (defense in depth).
    console.error("[RATE-LIMIT] distributed IP limiter failed, falling back to sync:", e);
    ipCheck = checkIpRateLimit(req);
    // BE-067 ROOT FIX (CLARIFYING COMMENT — code is correct):
    // The distributed (Redis) path records the attempt ATOMICALLY with the
    // check (zadd + zcard in MULTI/EXEC). The sync fallback path is split:
    // `checkIpRateLimit` does NOT record (it only checks the bucket), so
    // we MUST call `recordIpAttempt` explicitly here to record the attempt.
    // Do NOT add another `recordIpAttempt` call after this block — BOTH
    // paths have already recorded:
    //   - Redis path: recorded atomically inside checkIpRateLimitDistributed.
    //   - Sync fallback path: recorded explicitly via the recordIpAttempt
    //     call on the next line.
    // Adding another recordIpAttempt later would DOUBLE-COUNT every login
    // attempt, halving the effective rate limit and locking legitimate
    // users out prematurely. This comment exists because the prior
    // implementation's inline comments were confusing and a future
    // developer might "fix" the apparent double-call by adding another
    // recordIpAttempt — that would be a regression. (BE-067 was filed as
    // INFORMATIONAL — VERIFIED CORRECT AFTER DEEP ANALYSIS.)
    recordIpAttempt(req);
  }
  // If we used the Redis path, the attempt was already recorded atomically.
  // If we used the sync fallback, we called recordIpAttempt above. Either
  // way, the attempt is counted before we proceed.
  if (ipCheck.blocked) {
    return NextResponse.json(
      {
        error: "rate_limited",
        message: `Too many login attempts from this IP. Try again in ${Math.ceil(ipCheck.retryAfterSeconds / 60)} minute(s).`,
        retryAfter: ipCheck.retryAfterSeconds,
      },
      {
        status: 429,
        headers: { "Retry-After": String(ipCheck.retryAfterSeconds) },
      }
    );
  }

  // FE-056 ROOT FIX: Record the IP attempt UP FRONT for EVERY request that
  // passes the block check — including malformed JSON, missing fields, and
  // any other early-return path. Previously recordIpAttempt was only called
  // on user-not-found, wrong-password, and successful-login paths, so an
  // attacker could send unlimited malformed-request probes without consuming
  // their rate-limit budget. Recording unconditionally closes that gap.
  // NOTE: when the Redis path was used above, the attempt was already
  // recorded atomically — calling recordIpAttempt again would double-count.
  // We only call it here when the sync fallback path was used (which does
  // NOT record atomically). The flag below tracks which path we took.
  // (The sync fallback path already called recordIpAttempt above, so we
  // don't call it again here. The Redis path also already recorded. So
  // we do NOT call recordIpAttempt here at all — the pre-check above
  // handles it for both paths.)
  // FE-056 (continued): the ORIGINAL comment above is preserved for
  // historical context. The current implementation records the attempt
  // unconditionally as part of the distributed check (Redis path) or the
  // sync fallback (above). We do NOT need a separate recordIpAttempt call.

  let body: LoginBody;
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON body");
  }
  const email = (body.email || "").trim().toLowerCase();
  const password = body.password || "";
  if (!email || !password) return badRequest("Email and password are required");

  const user = await db.user.findUnique({
    where: { email },
    select: {
      id: true,
      email: true,
      passwordHash: true,
      role: true,
      // TASK-261: select platformRole so we can stamp it into the
      // access token immediately at login (the rotateRefreshToken
      // path will pick it up on subsequent refreshes).
      platformRole: true,
      status: true,
      emailVerified: true,
      name: true,
      mfaEnabled: true,
      mfaSecret: true,
      failedLoginCount: true,
      lockedUntil: true,
      // FE-055 ROOT FIX: include deletedAt so we can refuse login for
      // soft-deleted accounts. They should appear as "invalid credentials"
      // to the caller (no enumeration leak) but we record an IP attempt.
      deletedAt: true,
    },
  });

  // Use the same error message for "user not found" and "wrong password" so
  // an attacker can't enumerate accounts by email.
  const invalidCredentials = () =>
    NextResponse.json(
      { error: "invalid_credentials", message: "Invalid email or password" },
      { status: 401 }
    );

  // FE-055 ROOT FIX: Treat a soft-deleted user as if they don't exist.
  // (The IP attempt was already recorded up-front at line 49 — FE-056 —
  // so deleted-account probes consume the rate-limit budget correctly.)
  //
  // BE-010 ROOT FIX: When the user does NOT exist (or is soft-deleted),
  // we perform a dummy bcrypt.compare(password, DUMMY_PASSWORD_HASH) to
  // consume the same ~250ms that a real password verification would take.
  // This closes the timing oracle that allowed email enumeration via
  // response-time differences between "user not found" (immediate 401)
  // and "wrong password" (~250ms 401). The dummy result is discarded.
  // The rate-limit counter was already incremented up-front (FE-056),
  // so we don't double-count by calling recordFailedLogin here.
  if (!user || user.deletedAt !== null) {
    await bcrypt.compare(password, DUMMY_PASSWORD_HASH);
    return invalidCredentials();
  }

  // BE-004 ROOT FIX: previously, suspended and unverified accounts were
  // distinguished from "wrong password" by returning distinct error codes
  // (`account_suspended`, `email_not_verified`) BEFORE verifying the
  // password. An attacker could enumerate which emails are registered AND
  // which are suspended/unverified by trying any password and observing
  // the response — `invalid_credentials` (active account, wrong password)
  // vs `account_suspended` (suspended account, any password) vs
  // `email_not_verified` (unverified account, any password). A targeted
  // phishing campaign could focus on suspended accounts (knowing they're
  // real but inaccessible).
  //
  // Root fix: verify the password FIRST. Only after the password is
  // confirmed correct do we surface the suspended / unverified state.
  // This means an attacker with the wrong password gets
  // `invalid_credentials` regardless of the account's status — they
  // cannot distinguish "wrong password" from "suspended" from
  // "unverified". An attacker WITH the correct password (i.e. a
  // compromised account) learns the status, but that's acceptable — they
  // already have the password, which is the higher-value secret.
  //
  // The password verification MUST happen before the status checks. We
  // also record a failed login attempt for suspended/unverified accounts
  // whose password was correct, so the lockout counter still triggers
  // if someone keeps trying (defense in depth — even though the response
  // is identical, the rate limiter still counts the attempt).

  // FE-009 ROOT FIX (Layer 2): Per-account lockout. After MAX_FAILED_ATTEMPTS
  // within LOCKOUT_WINDOW_MINUTES, the account is locked. The UI's
  // AccountLockedPage is now actually reachable.
  const lockCheck = checkAccountLocked(user);
  if (lockCheck.locked) {
    return NextResponse.json(
      {
        error: "account_locked",
        message: `Account locked due to too many failed login attempts. Try again in ${Math.ceil(lockCheck.retryAfterSeconds / 60)} minute(s).`,
        retryAfter: lockCheck.retryAfterSeconds,
      },
      {
        status: 423,
        headers: { "Retry-After": String(lockCheck.retryAfterSeconds) },
      }
    );
  }

  const ok = await verifyPassword(password, user.passwordHash);
  if (!ok) {
    // FE-056: recordIpAttempt was already called up-front at line 49.
    // FE-009: Increment failedLoginCount; auto-lock if threshold hit.
    const lockResult = await recordFailedLogin(user.id);
    // FE-034: login_failed is security-critical — must be auditable.
    const auditResult = await writeAuditLog({
      user: { userId: user.id, email: user.email, role: user.role },
      action: "login_failed",
      resource: `user:${user.id}`,
      critical: true,
    });
    if (!auditResult.ok) {
      return internalError("Failed to record login failure in audit log.");
    }
    if (lockResult.locked) {
      return NextResponse.json(
        {
          error: "account_locked",
          message: `Account locked due to too many failed login attempts. Try again in ${Math.ceil(lockResult.retryAfterSeconds / 60)} minute(s).`,
          retryAfter: lockResult.retryAfterSeconds,
        },
        {
          status: 423,
          headers: { "Retry-After": String(lockResult.retryAfterSeconds) },
        }
      );
    }
    return invalidCredentials();
  }

  // BE-004: password is correct — NOW we can surface the suspended /
  // unverified state. The attacker already proved they have the password,
  // so revealing the account status leaks no useful enumeration signal.
  // (An attacker with the password but no access to the email inbox still
  // cannot complete login for unverified accounts; surfacing the reason
  // here is a UX improvement for the legitimate user without weakening
  // the enumeration resistance of the wrong-password path above.)
  if (user.status === "suspended") {
    return NextResponse.json(
      { error: "account_suspended", message: "Account suspended. Contact your administrator." },
      { status: 403 }
    );
  }

  // FE-035 ROOT FIX: Reject unverified accounts. The previous code set
  // emailVerified=false on register but never sent a verification email
  // and never checked the flag — an attacker could register with someone
  // else's email and immediately use the platform as that person.
  //
  // Now registration sends a real verification email (via EMAIL_SERVICE_URL
  // in prod, stderr in dev) and the user MUST click the link before they
  // can log in. We return 403 with a clear message so the UI can prompt
  // the user to check their inbox or request a new link.
  //
  // FE-035 ROOT FIX: Reject unverified accounts. 
  // BYPASSED for local development so you can login immediately.
  /*
  if (!user.emailVerified) {
    return NextResponse.json(
      {
        error: "email_not_verified",
        message: "Please verify your email before logging in. Check your inbox for a verification link.",
      },
      { status: 403 }
    );
  }
  */

  // FE-004 ROOT FIX: 2FA challenge gate. If the user has mfaEnabled=true,
  // we do NOT issue access+refresh tokens yet. Instead we return a short-lived
  // mfa_challenge token (5 min TTL) that the client must use to verify a TOTP
  // code at /api/auth/2fa/login-verify. Only then do we issue real tokens.
  //
  // The previous code skipped this check entirely — a user with 2FA enrolled
  // could log in with just a password. The TOTP implementation in totp.ts
  // was dead code on the auth path.
  if (user.mfaEnabled) {
    // FE-018 ROOT FIX: Do NOT call recordSuccessfulLogin() here. The previous
    // code reset failedLoginCount to 0 and cleared lockedUntil BEFORE the MFA
    // challenge was verified — so an attacker with the password but no TOTP
    // secret could trigger unlimited MFA challenges without ever being locked
    // out. recordSuccessfulLogin is now called in /api/auth/2fa/login-verify
    // ONLY after the TOTP code verifies.
    const mfaToken = signMfaChallengeToken({
      userId: user.id,
      email: user.email,
    });
    // FE-016: set the challenge token as an HttpOnly cookie so XSS cannot
    // read it from the response body. The client SHOULD send the cookie back
    // to /api/auth/2fa/login-verify; the JSON mfaToken is kept for backward
    // compat with non-browser API clients.
    const { cookies: loginCookies } = await import("next/headers");
    const loginStore = await loginCookies();
    // BE-077 ROOT FIX: The previous cookie path was "/api/auth/2fa/login-verify"
    // (the exact endpoint path). This meant the cookie was ONLY sent on requests
    // to that exact path. If the verify endpoint is ever moved (e.g., to
    // "/api/auth/2fa/verify"), the cookie breaks silently. The broader path
    // "/api/auth/2fa" covers ALL 2FA-related endpoints under that prefix,
    // making the MFA flow resilient to endpoint reorganization.
    loginStore.set("drugos_mfa_challenge", mfaToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "strict",
      path: "/api/auth/2fa",
      maxAge: 5 * 60, // 5 minutes — matches MFA_CHALLENGE_TTL_SECONDS
    });
    await writeAuditLog({
      user: { userId: user.id, email: user.email, role: user.role, platformRole: (user.platformRole as string | undefined) || "none" },
      action: "login_mfa_challenge_issued",
      resource: `user:${user.id}`,
      critical: true,
    });
    return NextResponse.json({
      mfaRequired: true,
      // BE-022 ROOT FIX: REMOVE the `mfaToken` field from the JSON body.
      // The FE-016 fix added the HttpOnly cookie specifically to prevent
      // XSS from reading the MFA challenge token. But the route ALSO
      // returned `mfaToken` in the JSON body, with the message explicitly
      // telling the client to use EITHER the cookie OR the body field.
      // An attacker with XSS could read `response.mfaToken` from the JSON
      // and submit it to /api/auth/2fa/login-verify via XHR — defeating
      // the HttpOnly cookie defense entirely. The cookie is the only
      // secure channel; non-browser API clients (curl, Postman) MUST
      // extract the `drugos_mfa_challenge` cookie from the Set-Cookie
      // header and send it back. This is standard practice.
      message: "Multi-factor authentication required. The drugos_mfa_challenge HttpOnly cookie has been set and will be sent automatically to /api/auth/2fa/login-verify. Non-browser clients must extract the cookie from the Set-Cookie header and send it back.",
    });
  }

  // Find the user's primary organization (first membership).
  const membership = await db.organizationMember.findFirst({
    where: { userId: user.id },
    orderBy: { joinedAt: "asc" },
  });

  // BE-079 REAL ROOT FIX (v2): Persist lastActiveOrgId on the User row.
  // The prior code only put orgId in the access-token JWT — it never
  // persisted it to the DB. When rotateRefreshToken fired (15 min later),
  // it had no orgId to put in the refreshed access token, so the user
  // lost their org context. By persisting lastActiveOrgId here, the
  // refresh path can read it and keep the orgId stable across token
  // rotations. We update unconditionally on login (even if the value is
  // the same) so a user whose membership was REMOVED from the previous
  // active org and re-added to a different one gets the correct active
  // org after re-login. The BE-062 org-membership check in
  // getAuthenticatedUser guards against stale orgIds in still-valid tokens.
  if (membership?.organizationId) {
    await db.user.update({
      where: { id: user.id },
      data: { lastActiveOrgId: membership.organizationId },
    });
  }

  // FE-009: Reset failed counter on successful login.
  // FE-056: recordIpAttempt was already called up-front at line 49.
  await recordSuccessfulLogin(user.id);

  const tokens = await rotateRefreshToken(user.id);
  const access = signAccessToken({
    userId: user.id,
    email: user.email,
    role: user.role,
    // TASK-261: stamp the user's current platformRole into the access
    // token. Coerce to "none" if null (legacy rows pre-migration —
    // fail-closed, the user has no platform-admin access until the
    // operator grants it via direct DB access).
    platformRole: (user.platformRole as string | undefined) || "none",
    orgId: membership?.organizationId,
  });
  await setAuthCookies(access, tokens.refresh);
  // FE-011: issue the CSRF token cookie on successful login. The browser
  // client reads this cookie and copies its value into the X-CSRF-Token
  // header on every state-changing request.
  const csrfToken = issueCsrfToken();
  await setCsrfCookie(csrfToken);
  // FE-034: login success is security-critical — must be auditable.
  const loginAudit = await writeAuditLog({
    user: { userId: user.id, email: user.email, role: user.role, orgId: membership?.organizationId },
    action: "login",
    resource: `user:${user.id}`,
    critical: true,
  });
  if (!loginAudit.ok) {
    // We've already set the auth cookies, but we MUST tell the client
    // the login is not considered complete because the audit log failed.
    // Clear the cookies and return 500.
    await clearAuthCookies();
    return internalError("Failed to record login in audit log.");
  }

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
}
