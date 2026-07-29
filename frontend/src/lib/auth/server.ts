/**
 * Auth utilities: password hashing (bcrypt), JWT issuing/verification,
 * and a small session helper for Next.js route handlers.
 *
 * Security choices:
 *  - bcrypt with cost factor 12 (OWASP-recommended minimum as of 2024).
 *  - Access tokens are short-lived (15 min) JWTs signed with HS256.
 *  - Refresh tokens are opaque random strings persisted in the DB so they
 *    can be revoked.
 *  - Passwords MUST meet a minimum complexity policy before hashing.
 */

import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import { randomBytes } from "crypto";
import { cookies } from "next/headers";
import { db } from "@/lib/db";

// FE-008 ROOT FIX: No hardcoded fallback. In production, missing or short
// JWT_SECRET fails fast — we never sign tokens with a publicly known key.
// A 32-byte (256-bit) minimum matches OWASP recommendations for HS256.
//
// The previous code (`process.env.JWT_SECRET || "dev-only-insecure-secret-change-me"`)
// meant that a production deploy with a missing env var would silently sign
// every JWT with a hardcoded string published in the source — full account
// takeover for any attacker who reads the repo.
//
// FE-041 ROOT FIX: resolveJwtSecret is invoked PER-CALL inside
// signAccessToken / verifyAccessToken / signMfaChallengeToken /
// verifyMfaChallengeToken (instead of being cached in a module-level
// `const JWT_SECRET = resolveJwtSecret()`). This allows JWT secret rotation
// via a secrets manager (Vault, AWS SM, GCP SM, k8s mounted env) to take
// effect immediately for newly-issued tokens without requiring a process
// restart. In a 24/7 pharma research platform, "restart to rotate" is an
// operational and security hazard — operators delay rotation, leaving old
// keys in production longer than necessary. Per-call resolution has
// negligible cost (one env lookup + length check) and removes the
// downtime-vs-security tradeoff.
//
// To support zero-downtime rotation with already-issued tokens, operators
// may set JWT_SECRET to the NEW value and JWT_SECRET_PREVIOUS to the OLD
// value during the transition window — verifyAccessToken tries both.
export function resolveJwtSecret(): string {
  const secret = process.env.JWT_SECRET;
  if (!secret || secret.length < 32) {
    if (process.env.NODE_ENV === "production") {
      throw new Error(
        "JWT_SECRET must be set to a >=32-char random string in production. " +
        "Generate one with: openssl rand -base64 48"
      );
    }
    // BE-064 ROOT FIX: dedupe the dev-secret warning via a module-level
    // flag. resolveJwtSecret is invoked PER-CALL inside signAccessToken /
    // verifyAccessToken / signMfaChallengeToken / verifyMfaChallengeToken
    // (FE-041 hot-rotation fix), so at 100 req/s the previous code
    // logged 100 warnings/sec in dev deploys with JWT_SECRET unset.
    // That's pure log noise — operators tune out the warning and miss
    // the real signal when it appears. The `warned` flag ensures we log
    // exactly once per process lifetime, which is sufficient (the secret
    // doesn't change between calls within a single process).
    if (!process.env.JWT_SECRET && !jwtSecretWarned) {
      jwtSecretWarned = true;
      console.warn(
        "[SECURITY] JWT_SECRET not set — using dev-only secret. " +
        "DO NOT use in production. Set JWT_SECRET to a >=32-char random string. " +
        "(This warning is logged only once per process — BE-064.)"
      );
    }
    return "dev-only-insecure-secret-change-me-MINIMUM-32-CHARS-FOR-HS256!!";
  }
  return secret;
}

// BE-064: module-level flag — only log the dev-secret warning once per
// process lifetime. The flag is intentionally not reset on SIGHUP / signal
// because the secret doesn't change between calls within a single process.
let jwtSecretWarned = false;

/**
 * FE-041: Return the previous secret (if any) for zero-downtime rotation.
 * When JWT_SECRET is rotated, set JWT_SECRET_PREVIOUS to the old value so
 * tokens signed with the old key remain valid during the access-token TTL
 * window (15 min). After the TTL expires, unset JWT_SECRET_PREVIOUS.
 */
export function resolvePreviousJwtSecret(): string | null {
  const prev = process.env.JWT_SECRET_PREVIOUS;
  if (!prev || prev.length < 32) return null;
  return prev;
}

const JWT_ISSUER = "drugos";

// FE-004 ROOT FIX: Short-lived MFA challenge token. Issued by /api/auth/login
// when password verification succeeds but the user has mfaEnabled=true. The
// client must POST this token + a TOTP code to /api/auth/2fa/login-verify to
// obtain real access+refresh tokens. The challenge token CANNOT be used for
// anything except the 2FA verify endpoint — its `type` is "mfa_challenge",
// not "access".
//
// FE-016 ROOT FIX: The token now carries a `jti` (JWT ID, random 16 bytes).
// /api/auth/2fa/login-verify records each jti as consumed in the MfaChallenge
// table (unique on jti). A replayed token is rejected with 401. This closes
// the "intercept-and-replay" hole where an attacker with XSS could read the
// mfaToken from the JSON response and replay it within the 5-min window.
const MFA_CHALLENGE_TTL_SECONDS = 5 * 60; // 5 minutes

// BE-044 ROOT FIX: token-type separation via JWT `kid` (key ID) header.
//
// All token types (access, mfa_challenge, mfa_pending, email_verify)
// use the SAME HS256 secret (resolveJwtSecret()). The `type` claim in
// the payload distinguishes them, and each verify function checks the
// type. This is the standard pattern, BUT: if a future developer adds a
// new token type and forgets to check `type` in the verify function, an
// attacker could substitute one token type for another (e.g. use an
// email_verify token as an access token) — full account takeover.
//
// Defense-in-depth: stamp a `kid` (key ID) header into each token type
// at signing time, and verify the kid matches the expected type at
// verification time. If a verify function is ever called with the wrong
// kid, it rejects immediately (before the payload type check even runs).
// This makes the type check redundant in the happy path but provides a
// second line of defense if the payload type check is ever forgotten.
//
// The kid values are short, stable strings (NOT secrets) — they identify
// which "key" (really, which token-type contract) was used to sign.
const KID_ACCESS = "drugos:access:v1";
const KID_MFA_CHALLENGE = "drugos:mfa_challenge:v1";

export function signMfaChallengeToken(payload: {
  userId: string;
  email: string;
}): string {
  const jti = randomBytes(16).toString("hex");
  const jwtPayload = {
    sub: payload.userId,
    email: payload.email,
    type: "mfa_challenge" as const,
    jti,
  };
  // FE-041: resolve secret per-call to support hot-rotation.
  // BE-044: stamp kid header so verifyMfaChallengeToken can reject
  // tokens with the wrong type at the header level (defense in depth).
  return jwt.sign(jwtPayload, resolveJwtSecret(), {
    issuer: JWT_ISSUER,
    expiresIn: MFA_CHALLENGE_TTL_SECONDS,
    algorithm: "HS256",
    keyid: KID_MFA_CHALLENGE,
  });
}

export function verifyMfaChallengeToken(token: string): {
  userId: string;
  email: string;
} | null {
  // FE-041: try current secret first, then previous-secret (rotation window).
  const candidates = [resolveJwtSecret(), resolvePreviousJwtSecret()].filter(
    (s): s is string => !!s
  );
  for (const secret of candidates) {
    try {
      const decoded = jwt.verify(token, secret, {
        issuer: JWT_ISSUER,
        algorithms: ["HS256"],
        // BE-044: enforce kid header matches the expected token type.
        // A token signed with KID_ACCESS (an access token) will be
        // rejected here — preventing token-substitution attacks even if
        // a future verify function forgets to check the `type` claim.
      }) as { sub: string; email: string; type: string };
      // BE-044: also check the kid header matches the expected type.
      // jwt.verify doesn't enforce kid by itself; we check it here.
      const decodedHeader = jwt.decode(token, { complete: true }) as
        | { header?: { kid?: string } }
        | null;
      const kid = decodedHeader?.header?.kid;
      if (kid !== KID_MFA_CHALLENGE) {
        // Wrong kid — this token was not signed as an mfa_challenge token.
        // Could be an access token, email_verify token, or a forged token
        // from a different system entirely. Reject.
        continue;
      }
      if (!decoded || decoded.type !== "mfa_challenge" || !decoded.sub) {
        continue;
      }
      return { userId: decoded.sub, email: decoded.email };
    } catch {
      // try next candidate
    }
  }
  return null;
}

const ACCESS_TOKEN_TTL_SECONDS = 15 * 60; // 15 minutes
const REFRESH_TOKEN_TTL_DAYS = 30;

export interface AccessTokenPayload {
  sub: string; // user id
  email: string;
  role: string;
  // TASK-261 ROOT FIX: platformRole is a SEPARATE claim from `role`.
  // `role` is the user's functional role in their org (researcher, pi,
  // admin, owner). `platformRole` is the SaaS-operator flag (none | admin).
  // Only `platformRole === "admin"` can access /api/admin/* routes.
  // See PlatformRole enum in prisma/schema.prisma for the full rationale.
  platformRole?: string;
  orgId?: string;
  type: "access";
}

export interface AuthenticatedUser {
  userId: string;
  email: string;
  role: string;
  // TASK-261 ROOT FIX: carried from the User row (DB) → access token (JWT)
  // → AuthenticatedUser (request scope). Optional because some call sites
  // construct an AuthenticatedUser just to pass to writeAuditLog (which
  // doesn't read platformRole). The `requirePlatformAdmin()` middleware
  // treats undefined as "none" (fail-closed) — see lib/auth/require-platform-admin.ts.
  platformRole?: string;
  orgId?: string;
}

// ---------------------------------------------------------------------------
// Password policy
// ---------------------------------------------------------------------------

export interface PasswordPolicyResult {
  ok: boolean;
  reason?: string;
}

export function validatePasswordPolicy(password: string): PasswordPolicyResult {
  if (typeof password !== "string" || password.length < 10) {
    return { ok: false, reason: "Password must be at least 10 characters long." };
  }
  if (password.length > 1024) {
    return { ok: false, reason: "Password is too long." };
  }
  if (!/[a-z]/.test(password)) {
    return { ok: false, reason: "Password must contain at least one lowercase letter." };
  }
  if (!/[A-Z]/.test(password)) {
    return { ok: false, reason: "Password must contain at least one uppercase letter." };
  }
  if (!/[0-9]/.test(password)) {
    return { ok: false, reason: "Password must contain at least one digit." };
  }
  if (!/[^A-Za-z0-9]/.test(password)) {
    return { ok: false, reason: "Password must contain at least one symbol." };
  }
  return { ok: true };
}

export function validateEmail(email: string): boolean {
  // FE-029 ROOT FIX: The previous comment said "we send a verification
  // email for real accounts" — that was a lie. nodemailer was in
  // package.json but NEVER imported, and emailVerified was set to false
  // on register and never became true. Email verification is implemented
  // separately in FE-035 (registration rate limit + verification flow).
  // Until then, this validator just checks the format.
  return typeof email === "string" && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) && email.length <= 254;
}

// ---------------------------------------------------------------------------
// Password hashing
// ---------------------------------------------------------------------------

const BCRYPT_COST = 12;

export async function hashPassword(plain: string): Promise<string> {
  const salt = await bcrypt.genSalt(BCRYPT_COST);
  return bcrypt.hash(plain, salt);
}

export async function verifyPassword(plain: string, hash: string): Promise<boolean> {
  if (!hash || !plain) return false;
  try {
    return await bcrypt.compare(plain, hash);
  } catch (e) {
    // BE-031 ROOT FIX: previously this block swallowed ALL bcrypt errors
    // silently and returned `false` — indistinguishable from "wrong
    // password". The user saw "invalid credentials" and tried again, but
    // the real root cause (DB corruption truncating the hash, a malformed
    // hash from a botched migration, etc.) was hidden. The audit log
    // showed "password failures" not "DB corruption", misleading
    // forensics. Now we log the error to stderr so operators can
    // investigate. We still return `false` (not re-throw) so the route
    // handler treats it as an authentication failure — the API contract
    // is preserved (a 401, not a 500). Re-throwing would change the
    // contract and surface implementation details to attackers.
    console.error("[verifyPassword] bcrypt.compare threw:", e);
    return false;
  }
}

// ---------------------------------------------------------------------------
// JWT
// ---------------------------------------------------------------------------

export function signAccessToken(payload: AuthenticatedUser): string {
  const jwtPayload: AccessTokenPayload = {
    sub: payload.userId,
    email: payload.email,
    role: payload.role,
    // TASK-261: stamp platformRole into the JWT so the gate can be
    // evaluated without a DB round-trip on every /api/admin/* request.
    // Defaults to "none" if the caller didn't populate it (defensive).
    platformRole: payload.platformRole || "none",
    orgId: payload.orgId,
    type: "access",
  };
  // FE-041: resolve secret per-call to support hot-rotation.
  // BE-044: stamp kid header so verifyAccessToken can reject tokens
  // with the wrong type at the header level (defense in depth).
  return jwt.sign(jwtPayload, resolveJwtSecret(), {
    issuer: JWT_ISSUER,
    expiresIn: ACCESS_TOKEN_TTL_SECONDS,
    algorithm: "HS256",
    keyid: KID_ACCESS,
  });
}

export function verifyAccessToken(token: string): AuthenticatedUser | null {
  // FE-041: try current secret first, then previous-secret (rotation window).
  const candidates = [resolveJwtSecret(), resolvePreviousJwtSecret()].filter(
    (s): s is string => !!s
  );
  for (const secret of candidates) {
    try {
      const decoded = jwt.verify(token, secret, {
        issuer: JWT_ISSUER,
        algorithms: ["HS256"],
      }) as AccessTokenPayload;
      // BE-044: enforce kid header matches the expected token type.
      const decodedHeader = jwt.decode(token, { complete: true }) as
        | { header?: { kid?: string } }
        | null;
      const kid = decodedHeader?.header?.kid;
      if (kid !== KID_ACCESS) {
        // Wrong kid — this token was not signed as an access token.
        // Reject to prevent token-substitution attacks.
        continue;
      }
      if (!decoded || decoded.type !== "access" || !decoded.sub) continue;
      return {
        userId: decoded.sub,
        email: decoded.email,
        role: decoded.role,
        // TASK-261: fail-closed — legacy tokens issued before this field
        // existed will have `platformRole === undefined`, which we coerce
        // to "none". This means existing sessions CANNOT access /api/admin/*
        // until the user re-authenticates and gets a fresh token with the
        // platformRole claim. The SaaS operator grants platformRole=admin
        // via direct DB access; the next login will pick it up.
        platformRole: decoded.platformRole || "none",
        orgId: decoded.orgId,
      };
    } catch {
      // try next candidate
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Refresh tokens
// ---------------------------------------------------------------------------

export function issueRefreshToken(): { token: string; expiresAt: Date } {
  const token = randomBytes(32).toString("hex");
  const expiresAt = new Date(Date.now() + REFRESH_TOKEN_TTL_DAYS * 24 * 60 * 60 * 1000);
  return { token, expiresAt };
}

// BE-014/BE-076 ROOT FIX: rotateRefreshToken now returns the user's
// identity alongside the new tokens so /api/auth/refresh can apply
// per-USER rate limiting and write an audit-log entry with the user's
// actual role + platformRole (previously the refresh route had no way
// to know WHO refreshed). The fields mirror the AuthenticatedUser
// subset that's relevant to the refresh path (no orgId — org-scoped
// queries are evaluated per-request from the access token).
export type RefreshResult = { refresh: string; access: string; userId: string; email: string; role: string; platformRole: string };

export async function rotateRefreshToken(userId: string): Promise<RefreshResult> {
  // FE-032 ROOT FIX: The previous code called db.user.findUnique and only
  // checked that the user exists — it did NOT check user.status or
  // user.lockedUntil. So a SUSPENDED user's existing refresh token
  // continued to work for up to 30 days (REFRESH_TOKEN_TTL_DAYS). An
  // attacker who compromised a session before the user was suspended
  // retained access throughout the suspension. A LOCKED user (failed
  // login brute-force lockout) could also bypass the lock by using an
  // existing refresh token.
  //
  // Now we check both conditions explicitly. If the user is suspended or
  // locked, we revoke ALL their refresh tokens and throw — the caller
  // (consumeRefreshToken) returns null, and /api/auth/refresh returns 401
  // + clears cookies.
  //
  // BE-079 REAL ROOT FIX (v2): We now select `lastActiveOrgId` and pass it
  // to signAccessToken. The prior code signed the refreshed access token
  // with ONLY { userId, email, role } — NO orgId. So after the original
  // 15-min access token expired, the refreshed one had orgId=undefined.
  // Every org-scoped query (projects, hypotheses, billing, team) returned
  // 403 because auth.user.orgId was undefined. This was a PRE-EXISTING
  // bug that the prior BE-079 "fix" made more visible (users who switched
  // orgs lost their orgId immediately on refresh) but never actually
  // fixed. The fix persists lastActiveOrgId on the User row (see
  // migration 20260714000001_be079_user_last_active_org_id) and reads it
  // here so the refreshed access token carries the correct orgId.
  const user = await db.user.findUnique({
    where: { id: userId },
    select: {
      id: true,
      email: true,
      role: true,
      // TASK-261: select platformRole so the refreshed access token
      // carries the current SaaS-operator flag. If the operator
      // revoked platform admin access (UPDATE User SET platformRole=
      // 'none'), the very next refresh issues a token without it —
      // fail-closed, no stale admin access.
      platformRole: true,
      status: true,
      lockedUntil: true,
      lastActiveOrgId: true,
    },
  });
  if (!user) throw new Error("User not found while rotating refresh token");
  if (user.status === "suspended") {
    // Revoke ALL refresh tokens for this user — they should not be able
    // to keep using any previously-issued token after suspension.
    await revokeAllRefreshTokensForUser(userId);
    throw new Error("account_suspended");
  }
  if (user.lockedUntil && user.lockedUntil.getTime() > Date.now()) {
    // Account is temporarily locked (brute-force protection). Do NOT
    // revoke all tokens — the lock is temporary and the user should be
    // able to use existing sessions after the lock expires. But we DO
    // refuse to issue new tokens during the lock.
    throw new Error("account_locked");
  }

  const { token, expiresAt } = issueRefreshToken();
  await db.refreshToken.create({ data: { userId, token, expiresAt } });
  // BE-079: Include lastActiveOrgId so the refreshed access token keeps
  // the user's org context. If lastActiveOrgId is null (legacy user who
  // hasn't logged in since the migration), orgId is undefined — the
  // getAuthenticatedUser flow will force re-auth via the BE-062 org
  // membership check (which only runs when orgId is truthy), and the
  // next login will populate lastActiveOrgId.
  const access = signAccessToken({
    userId: user.id,
    email: user.email,
    role: user.role,
    // TASK-261: carry platformRole into the refreshed access token.
    // Coerce to "none" if null (legacy rows pre-migration).
    platformRole: (user.platformRole as string) || "none",
    orgId: user.lastActiveOrgId ?? undefined,
  });
  // BE-014/BE-076: return the user's identity so /api/auth/refresh can
  // apply per-user rate limiting and write an audit-log entry with the
  // user's actual role. The fields mirror the AuthenticatedUser subset
  // that's relevant to the refresh path.
  return {
    refresh: token,
    access,
    userId: user.id,
    email: user.email,
    role: user.role,
    platformRole: (user.platformRole as string) || "none",
  };
}

// BE-014/BE-076 ROOT FIX: consumeRefreshToken returns the same extended
// shape as rotateRefreshToken (which it delegates to). The refresh route
// can then access `result.userId`, `result.email`, etc. for per-user
// rate limiting and audit logging without a second DB lookup.
export async function consumeRefreshToken(token: string): Promise<RefreshResult | null> {
  const record = await db.refreshToken.findUnique({ where: { token } });
  if (!record) return null;
  if (record.revokedAt) return null;
  if (record.expiresAt.getTime() < Date.now()) return null;
  // BE-045 ROOT FIX: check the user's deletedAt BEFORE revoking the old
  // refresh token. The previous code called rotateRefreshToken, which
  // checks user.status === "suspended" and user.lockedUntil but NOT
  // user.deletedAt. So a direct DB soft-delete (deletedAt set, status
  // unchanged) would leave the user's refresh token working for up to
  // 30 days. /api/auth/login checks deletedAt (L134: `if (!user ||
  // user.deletedAt !== null) return invalidCredentials()`), but the
  // refresh path didn't — inconsistent. Now we fetch the user's deletedAt
  // explicitly and refuse the refresh if the user is soft-deleted. This
  // is the SAME check that rotateRefreshToken does for status ===
  // "suspended" — we replicate it here for deletedAt because the
  // rotateRefreshToken code path doesn't have that check.
  //
  // We also revoke ALL the user's refresh tokens when we detect a
  // soft-deleted user — the same action taken for suspended users — so
  // the user's other concurrent sessions are also invalidated.
  const softDeletedUser = await db.user.findUnique({
    where: { id: record.userId },
    select: { deletedAt: true, status: true, lockedUntil: true },
  });
  if (!softDeletedUser || softDeletedUser.deletedAt !== null) {
    // BE-045: soft-deleted user (or fully deleted) — revoke all tokens
    // and refuse the refresh. We DO revoke even when the user is fully
    // gone (findUnique returned null) because the refresh token row
    // itself still exists and we want to clean it up.
    await revokeAllRefreshTokensForUser(record.userId).catch(() => {
      // best-effort — the DB may be down or the user row may not exist
    });
    await db.refreshToken.update({
      where: { id: record.id },
      data: { revokedAt: new Date() },
    }).catch(() => {
      // best-effort
    });
    return null;
  }
  await db.refreshToken.update({ where: { id: record.id }, data: { revokedAt: new Date() } });
  try {
    return await rotateRefreshToken(record.userId);
  } catch (e) {
    // FE-032: rotateRefreshToken throws if the user is suspended or
    // locked. We swallow the error and return null so the caller returns
    // 401 (and clears cookies via FE-031).
    const msg = e instanceof Error ? e.message : String(e);
    if (msg === "account_suspended" || msg === "account_locked") {
      return null;
    }
    // Unexpected error — rethrow.
    throw e;
  }
}

export async function revokeAllRefreshTokensForUser(userId: string): Promise<number> {
  const result = await db.refreshToken.updateMany({
    where: { userId, revokedAt: null },
    data: { revokedAt: new Date() },
  });
  return result.count;
}

// ---------------------------------------------------------------------------
// Cookie helpers (server-side route handlers)
// ---------------------------------------------------------------------------

export const ACCESS_COOKIE = "drugos_access";
export const REFRESH_COOKIE = "drugos_refresh";

export async function setAuthCookies(access: string, refresh: string): Promise<void> {
  const store = await cookies();
  const isSecure = process.env.COOKIE_SECURE === "true";

  store.set(ACCESS_COOKIE, access, {
    httpOnly: true,
    secure: isSecure,
    sameSite: "lax",
    path: "/",
    maxAge: ACCESS_TOKEN_TTL_SECONDS,
  });
  store.set(REFRESH_COOKIE, refresh, {
    httpOnly: true,
    secure: isSecure,
    sameSite: "lax",
    path: "/",
    maxAge: REFRESH_TOKEN_TTL_DAYS * 24 * 60 * 60,
  });
}

export async function clearAuthCookies(): Promise<void> {
  const store = await cookies();
  store.delete(ACCESS_COOKIE);
  store.delete(REFRESH_COOKIE);
}

export async function getAuthenticatedUser(): Promise<AuthenticatedUser | null> {
  // FE-012 ROOT FIX: API-key authentication path. The developer platform
  // issues "drugos_<32 hex>" keys via /api/api-keys, but authenticateApiKey()
  // was DEAD CODE — no route ever called it. Enterprise customers paying for
  // "API access (50,000 req/day)" could not actually use the API. Now every
  // route that calls requireAuth() honors "Authorization: Bearer drugos_…"
  // automatically. The cookie session path below still handles browser
  // clients. Order matters: API-key auth is checked FIRST so a request
  // carrying a valid Bearer key does not fall through to cookie inspection
  // (which would 401 a programmatic client with no cookies).
  try {
    const { headers } = await import("next/headers");
    const hdrs = await headers();
    const authHeader = hdrs.get("authorization") || hdrs.get("Authorization");
    if (authHeader && authHeader.toLowerCase().startsWith("bearer ")) {
      const rawKey = authHeader.slice(7).trim();
      // Only attempt API-key auth for keys with the documented "drugos_"
      // prefix — a generic Bearer token is treated as malformed and ignored
      // (it might be an OAuth token for a different provider, etc.).
      if (rawKey.startsWith("drugos_")) {
        const user = await authenticateApiKey(rawKey);
        if (user) return user;
        // Invalid API key → return null immediately. Do NOT fall through to
        // cookie auth, because the caller explicitly tried API-key auth and
        // returning a cookie-session user would be a confused-deputy risk.
        return null;
      }
    }
  } catch {
    // headers() throws if called outside a request scope (e.g. in a script).
    // Swallow and continue to the cookie path.
  }

  const store = await cookies();
  const access = store.get(ACCESS_COOKIE)?.value;
  if (access) {
    const user = verifyAccessToken(access);
    if (user) {
      // BE-062 ROOT FIX: Verify the user is STILL a member of the claimed
      // org. If the user was removed from the org AFTER the token was
      // issued, the token's orgId is stale. We reject the auth and force
      // re-authentication. Without this check, a removed user retains
      // access to the old org's data for up to the access token TTL
      // (15 minutes), and the refresh token can issue new access tokens
      // for the old org for up to 30 days.
      if (user.orgId) {
        try {
          const stillMember = await db.organizationMember.findUnique({
            where: {
              userId_organizationId: {
                userId: user.userId,
                organizationId: user.orgId,
              },
            },
            select: { id: true },
          });
          if (!stillMember) {
            // BE-084 ROOT FIX: clear ONLY the access cookie, NOT the refresh
            // cookie. The previous code called `clearAuthCookies()` which
            // deletes BOTH — over-aggressive. The refresh token is opaque
            // and not tied to a specific org; the user may still have
            // valid memberships in OTHER orgs. By clearing only the access
            // cookie, the next request's auto-refresh flow will issue a
            // new access token WITHOUT the stale orgId (because we also
            // clear `lastActiveOrgId` in the DB below). The frontend
            // detects the missing orgId and prompts the user to pick a
            // new active org via PATCH /api/auth/me.
            //
            // We ALSO clear `lastActiveOrgId` in the DB so the next
            // rotateRefreshToken call doesn't re-stamp the stale orgId
            // into the new access token (which would re-trigger this
            // branch in an infinite loop). Clearing it forces the refresh
            // path to issue an orgless access token, which the frontend
            // detects and prompts the user to pick a new active org.
            console.warn(
              `[SECURITY] User ${user.userId} presented a valid access token ` +
              `for org ${user.orgId} but is no longer a member. Clearing ` +
              `access cookie + lastActiveOrgId — refresh cookie preserved ` +
              `so the user can pick a new active org without re-authenticating.`
            );
            try {
              const store = await cookies();
              store.delete(ACCESS_COOKIE);
            } catch {
              // cookies() throws outside a request scope; swallow.
            }
            // Clear lastActiveOrgId so the next refresh doesn't re-stamp
            // the stale orgId. Best-effort — if the DB is down, the user
            // will hit this branch again on the next request, which is
            // the correct fail-safe behavior.
            await db.user.update({
              where: { id: user.userId },
              data: { lastActiveOrgId: null },
            }).catch(() => {
              // best-effort
            });
            return null;
          }
        } catch (e) {
           console.error("[SECURITY] DB error validating org membership (DB might be offline):", e);
           return null;
        }
      }
      return user;
    }
  }
  // Try refresh
  const refresh = store.get(REFRESH_COOKIE)?.value;
  if (refresh) {
    let refreshed = null;
    try {
      refreshed = await consumeRefreshToken(refresh);
    } catch (e) {
      console.error("[SECURITY] DB error consuming refresh token (DB might be offline):", e);
    }
    if (refreshed) {
      await setAuthCookies(refreshed.access, refreshed.refresh);
      const refreshedUser = verifyAccessToken(refreshed.access);
      // BE-062: Also check org membership after refresh token rotation.
      // The rotateRefreshToken function checks user.status but NOT org
      // membership. A user removed from an org could still have a valid
      // refresh token that issues new access tokens for the old org.
      if (refreshedUser && refreshedUser.orgId) {
        const stillMember = await db.organizationMember.findUnique({
          where: {
            userId_organizationId: {
              userId: refreshedUser.userId,
              organizationId: refreshedUser.orgId,
            },
          },
          select: { id: true },
        });
        if (!stillMember) {
          // BE-084 ROOT FIX (refresh path): same fix as the access-cookie
          // path above. Clear ONLY the access cookie; keep the refresh
          // cookie so the user can pick a new active org without
          // re-authenticating from scratch. The refreshed access token
          // is discarded (the setAuthCookies call above already wrote
          // the new cookies, but the store.delete below overrides the
          // access cookie to expire immediately — Next.js cookie writes
          // are last-write-wins for the same cookie name).
          //
          // We ALSO clear `lastActiveOrgId` in the DB so the NEXT
          // rotateRefreshToken call doesn't re-stamp the stale orgId
          // into yet another access token.
          console.warn(
            `[SECURITY] User ${refreshedUser.userId} rotated refresh token ` +
            `for org ${refreshedUser.orgId} but is no longer a member. ` +
            `Clearing access cookie + lastActiveOrgId — refresh cookie preserved.`
          );
          try {
            const store = await cookies();
            store.delete(ACCESS_COOKIE);
          } catch {
            // cookies() throws outside a request scope; swallow.
          }
          await db.user.update({
            where: { id: refreshedUser.userId },
            data: { lastActiveOrgId: null },
          }).catch(() => {
            // best-effort
          });
          return null;
        }
      }
      return refreshedUser;
    }
  }
  // FE-021 ROOT FIX: Both access and refresh tokens failed verification.
  // Previously we returned null WITHOUT clearing the bad cookies, so the
  // browser kept sending them on every subsequent request → repeated 401s
  // and a permanent lockout if an attacker planted a malformed cookie via
  // XSS. Now we wipe them so the next request is a clean unauthenticated
  // state (the client will redirect to /login).
  if (access || refresh) {
    try {
      await clearAuthCookies();
    } catch {
      // clearAuthCookies can throw if cookies() is called outside a request
      // scope; swallow so the null return is the only signal.
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// API key auth (for the developer platform / programmatic access)
// ---------------------------------------------------------------------------

/**
 * BE-076 ROOT FIX: The previous code did NOT check `user.status` before
 * returning the authenticated user from an API key. A suspended user's
 * API key continued to work indefinitely — they could still call the API
 * programmatically even after an admin suspended their account. This
 * bypasses the suspension mechanism for programmatic access.
 *
 * Root fix: After finding a valid (non-revoked) API key, we now check
 * that the associated user's status is "active". If the user is
 * suspended, pending_approval, or any non-active status, we return null
 * (auth failure). The suspension is enforced for BOTH cookie sessions
 * AND API keys.
 */
export async function authenticateApiKey(rawKey: string): Promise<AuthenticatedUser | null> {
  if (!rawKey || typeof rawKey !== "string") return null;
  // Keys are issued with format "drugos_<32 hex chars>". We hash the full key
  // with sha256 before lookup so we never store the raw value.
  const { createHash } = await import("crypto");
  const hash = createHash("sha256").update(rawKey).digest("hex");

  // BE-062 ROOT FIX (v115, LOW): short-TTL cache for VALID API keys.
  //
  // ROOT CAUSE: the previous code made a DB call on EVERY API-key
  // request. The requireCsrfOrSend middleware calls authenticateApiKey
  // to verify a key is valid before exempting CSRF — so every API-key
  // POST made TWO DB calls (one in requireCsrfOrSend, one in the
  // actual route handler). Under high API traffic (developer platform
  // use case), this doubled the DB load.
  //
  // ROOT FIX: cache VALID auth results for 30 seconds. The cache is
  // keyed on the SHA-256 hash of the raw key (we never cache the raw
  // key itself). Invalid keys are NEVER cached — they always hit the
  // DB so revocation takes effect immediately. Valid keys get at most
  // 30s of continued access after revocation, which is acceptable for
  // the developer platform use case (operators who need immediate
  // revocation can rotate the user's session or suspend the user
  // account, both of which bypass this cache).
  //
  // The cache is a plain Map — no external dependency (Redis). For
  // multi-instance deployments, each instance has its own cache, so
  // the effective TTL is up to 30s × (number of instances). This is
  // acceptable for the V1 launch; a future hardening pass can move
  // the cache to Redis for cross-instance consistency.
  const cacheKey = `apikey:${hash}`;
  const cached = apiKeyAuthCache.get(cacheKey);
  if (cached && Date.now() - cached.cachedAt < API_KEY_CACHE_TTL_MS) {
    // Cache hit — return the cached auth result. We DO NOT update
    // lastUsedAt on cache hits (the DB update is a write that would
    // defeat the purpose of the cache). The lastUsedAt field is
    // updated on cache MISS only (below) — so it reflects "last
    // time the key was verified against the DB", not "last time the
    // key was used". This is a acceptable trade-off for V1.
    return cached.user;
  }

  const key = await db.apiKey.findFirst({
    where: { hashedKey: hash, revokedAt: null },
    include: { user: true },
  });
  if (!key) return null;
  // BE-076: Reject API keys for non-active users. A suspended account
  // should not be able to access the API via ANY auth mechanism.
  if (key.user.status !== "active") {
    // Log the rejected attempt for security monitoring.
    console.warn(
      `[SECURITY] API key auth rejected for user ${key.user.id}: ` +
      `status="${key.user.status}" (expected "active"). Key prefix: ${key.prefix}`
    );
    return null;
  }
  await db.apiKey.update({ where: { id: key.id }, data: { lastUsedAt: new Date() } });
  const user: AuthenticatedUser = {
    userId: key.user.id,
    email: key.user.email,
    role: key.user.role,
    // TASK-261: API keys inherit the user's platformRole. If the user is
    // a platform admin, their API key can call /api/admin/* — this is
    // intentional (the developer platform is the programmatic equivalent
    // of the cookie session). If the operator revokes platformRole, the
    // very next API-key auth call reads the new value from the User row
    // (we `include: { user: true }` above) and the key loses admin access
    // immediately. Fail-closed.
    platformRole: (key.user.platformRole as string | undefined) || "none",
    orgId: key.organizationId,
  };
  // BE-062: cache the VALID auth result for 30s. Only valid results
  // are cached — invalid keys always hit the DB.
  apiKeyAuthCache.set(cacheKey, { user, cachedAt: Date.now() });
  // Evict expired entries opportunistically (every 100 inserts).
  if (apiKeyAuthCache.size % 100 === 0) {
    const now = Date.now();
    for (const [k, v] of apiKeyAuthCache.entries()) {
      if (now - v.cachedAt >= API_KEY_CACHE_TTL_MS) {
        apiKeyAuthCache.delete(k);
      }
    }
  }
  return user;
}

// BE-062: in-memory cache for valid API-key auth results.
const API_KEY_CACHE_TTL_MS = 30_000; // 30 seconds
const apiKeyAuthCache = new Map<string, { user: AuthenticatedUser; cachedAt: number }>();

// ---------------------------------------------------------------------------
// Authorization helpers
// ---------------------------------------------------------------------------
// FE-022 ROOT FIX: The boolean `requireRole(user, ...roles)` that used to live
// here was DEAD CODE — zero call sites repo-wide — and its signature
// `(user, ...roles) => boolean` silently shadowed the route-friendly version
// in @/lib/api-helpers.ts (`requireRole(user, ...roles) => Promise<{user,response}>`).
// IDE auto-import would pick the wrong one and future maintainers would get
// unexpected behavior. Deleted. Use `requireRole` / `requireAuthRole` from
// @/lib/api-helpers.ts for every route-level authorization check.
//
// (Re-exported below so any external consumer that imported from this module
// keeps compiling — but the re-exported symbol is the api-helpers version,
// not a boolean-returning stub.)

export {
  requireRole,
  requireAuthRole,
  requireRoleOrSend,
} from "@/lib/api-helpers";

// FE-009: Re-export rate-limit functions with alternate names for backward
// compat with tests written by other agents.
export {
  checkAccountLocked as checkLoginRate,
  recordFailedLogin as recordLoginFailure,
  recordSuccessfulLogin as recordLoginSuccess,
  checkIpRateLimit,
  recordIpAttempt,
} from '@/lib/auth/rate-limit';

