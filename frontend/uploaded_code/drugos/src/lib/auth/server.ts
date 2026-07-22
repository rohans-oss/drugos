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
import { randomBytes, createHmac, timingSafeEqual } from "crypto";
import { cookies } from "next/headers";
import { db } from "@/lib/db";

// V100 ROOT FIX (BUG FE-008, P0 CRITICAL): the previous code had a weak
// fallback `"dev-only-insecure-secret-change-me"` when JWT_SECRET was not
// set. In production, an attacker who knows this public fallback (it's in
// the source code / git history) can forge any JWT and take over any
// account. Root fix: NO fallback. If JWT_SECRET is missing or too short
// (<32 chars), the server REFUSES to start (throws immediately). This is
// a hard failure — better to crash than to issue forgeable tokens.
const JWT_SECRET_RAW = process.env.JWT_SECRET;
if (!JWT_SECRET_RAW || JWT_SECRET_RAW.length < 32) {
  throw new Error(
    "JWT_SECRET environment variable must be set to a cryptographically " +
    "random string of at least 32 characters. Refusing to start with a " +
    "weak/missing secret (V100 BUG FE-008 fix)."
  );
}
const JWT_SECRET: string = JWT_SECRET_RAW;
const JWT_ISSUER = "drugos";
const ACCESS_TOKEN_TTL_SECONDS = 15 * 60; // 15 minutes
const REFRESH_TOKEN_TTL_DAYS = 30;
// V100 BUG #11: MFA challenge tokens are short-lived (5 min) JWTs that
// carry just enough to identify the user who passed password verification
// but has NOT yet completed the TOTP challenge.
const MFA_CHALLENGE_TTL_SECONDS = 5 * 60; // 5 minutes

export interface AccessTokenPayload {
  sub: string; // user id
  email: string;
  role: string;
  orgId?: string;
  type: "access";
}

export interface AuthenticatedUser {
  userId: string;
  email: string;
  role: string;
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
  // Pragmatic RFC-5322-ish check; we do not need to be perfect — we send a
  // verification email for real accounts.
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
  } catch {
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
    orgId: payload.orgId,
    type: "access",
  };
  return jwt.sign(jwtPayload, JWT_SECRET, {
    issuer: JWT_ISSUER,
    expiresIn: ACCESS_TOKEN_TTL_SECONDS,
    algorithm: "HS256",
  });
}

export function verifyAccessToken(token: string): AuthenticatedUser | null {
  try {
    const decoded = jwt.verify(token, JWT_SECRET, {
      issuer: JWT_ISSUER,
      algorithms: ["HS256"],
    }) as AccessTokenPayload;
    if (!decoded || decoded.type !== "access" || !decoded.sub) return null;
    return {
      userId: decoded.sub,
      email: decoded.email,
      role: decoded.role,
      orgId: decoded.orgId,
    };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Refresh tokens
// ---------------------------------------------------------------------------

export function issueRefreshToken(): { token: string; expiresAt: Date } {
  const token = randomBytes(32).toString("hex");
  const expiresAt = new Date(Date.now() + REFRESH_TOKEN_TTL_DAYS * 24 * 60 * 60 * 1000);
  return { token, expiresAt };
}

export async function rotateRefreshToken(userId: string): Promise<{ refresh: string; access: string }> {
  const { token, expiresAt } = issueRefreshToken();
  await db.refreshToken.create({ data: { userId, token, expiresAt } });
  const user = await db.user.findUnique({ where: { id: userId } });
  if (!user) throw new Error("User not found while rotating refresh token");
  const access = signAccessToken({ userId: user.id, email: user.email, role: user.role });
  return { refresh: token, access };
}

export async function consumeRefreshToken(token: string): Promise<{ refresh: string; access: string } | null> {
  const record = await db.refreshToken.findUnique({ where: { token } });
  if (!record) return null;
  if (record.revokedAt) return null;
  if (record.expiresAt.getTime() < Date.now()) return null;
  await db.refreshToken.update({ where: { id: record.id }, data: { revokedAt: new Date() } });
  return rotateRefreshToken(record.userId);
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
  const isProd = process.env.NODE_ENV === "production";
  store.set(ACCESS_COOKIE, access, {
    httpOnly: true,
    secure: isProd,
    sameSite: "lax",
    path: "/",
    maxAge: ACCESS_TOKEN_TTL_SECONDS,
  });
  store.set(REFRESH_COOKIE, refresh, {
    httpOnly: true,
    secure: isProd,
    sameSite: "lax",
    path: "/api/auth/refresh",
    maxAge: REFRESH_TOKEN_TTL_DAYS * 24 * 60 * 60,
  });
}

export async function clearAuthCookies(): Promise<void> {
  const store = await cookies();
  store.delete(ACCESS_COOKIE);
  store.delete(REFRESH_COOKIE);
}

export async function getAuthenticatedUser(): Promise<AuthenticatedUser | null> {
  const store = await cookies();
  const access = store.get(ACCESS_COOKIE)?.value;
  if (!access) return null;
  const user = verifyAccessToken(access);
  if (user) return user;
  // Try refresh
  const refresh = store.get(REFRESH_COOKIE)?.value;
  if (!refresh) return null;
  const refreshed = await consumeRefreshToken(refresh);
  if (!refreshed) return null;
  await setAuthCookies(refreshed.access, refreshed.refresh);
  return verifyAccessToken(refreshed.access);
}

// ---------------------------------------------------------------------------
// API key auth (for the developer platform / programmatic access)
// ---------------------------------------------------------------------------

export async function authenticateApiKey(rawKey: string): Promise<AuthenticatedUser | null> {
  if (!rawKey || typeof rawKey !== "string") return null;
  // Keys are issued with format "drugos_<32 hex chars>". We hash the full key
  // with sha256 before lookup so we never store the raw value.
  const { createHash } = await import("crypto");
  const hash = createHash("sha256").update(rawKey).digest("hex");
  const key = await db.apiKey.findFirst({
    where: { hashedKey: hash, revokedAt: null },
    include: { user: true },
  });
  if (!key) return null;
  await db.apiKey.update({ where: { id: key.id }, data: { lastUsedAt: new Date() } });
  return {
    userId: key.user.id,
    email: key.user.email,
    role: key.user.role,
    orgId: key.organizationId,
  };
}

// ---------------------------------------------------------------------------
// Authorization helpers
// ---------------------------------------------------------------------------

export function requireRole(user: AuthenticatedUser | null, ...roles: string[]): boolean {
  if (!user) return false;
  if (roles.length === 0) return true;
  return roles.includes(user.role);
}

// ---------------------------------------------------------------------------
// V100 ROOT FIX (BUG #11, P0 CRITICAL): MFA / 2FA — TOTP verification +
// short-lived MFA challenge tokens.
//
// The previous login route verified the password and immediately issued
// access+refresh tokens WITHOUT checking `user.mfaEnabled`. 2FA was purely
// cosmetic — account takeover with password alone. The entire TOTP setup
// was dead code on the auth path.
//
// Root fix: after password verification, if `user.mfaEnabled === true`,
// the login route returns `{ error: "mfa_required", mfaToken: <challenge> }`
// instead of issuing access/refresh tokens. A separate
// `POST /api/auth/2fa/login-verify` endpoint accepts the challenge token
// + TOTP code, verifies the code against `user.mfaSecret`, and ONLY then
// issues the real access/refresh tokens.
// ---------------------------------------------------------------------------

export interface MfaChallengePayload {
  sub: string; // user id
  email: string;
  type: "mfa_challenge";
}

/** Issue a short-lived MFA challenge token (5 min TTL). */
export function signMfaChallengeToken(userId: string, email: string): string {
  const payload: MfaChallengePayload = { sub: userId, email, type: "mfa_challenge" };
  return jwt.sign(payload, JWT_SECRET, {
    issuer: JWT_ISSUER,
    expiresIn: MFA_CHALLENGE_TTL_SECONDS,
    algorithm: "HS256",
  });
}

/** Verify an MFA challenge token. Returns the userId or null. */
export function verifyMfaChallengeToken(token: string): { userId: string; email: string } | null {
  try {
    const decoded = jwt.verify(token, JWT_SECRET, {
      issuer: JWT_ISSUER,
      algorithms: ["HS256"],
    }) as MfaChallengePayload;
    if (!decoded || decoded.type !== "mfa_challenge" || !decoded.sub) return null;
    return { userId: decoded.sub, email: decoded.email };
  } catch {
    return null;
  }
}

/**
 * Verify a 6-digit TOTP code against a base32-encoded secret.
 * Implements RFC 6238 TOTP using HMAC-SHA1 (the standard TOTP algorithm).
 * Uses a ±1 time-step window to tolerate clock skew.
 *
 * @param token 6-digit TOTP code from the user's authenticator app.
 * @param secretBase32 Base32-encoded secret stored in `user.mfaSecret`.
 * @returns true if the code is valid within the current ±1 window.
 */
export function verifyTotp(token: string, secretBase32: string): boolean {
  if (!token || !secretBase32) return false;
  const code = token.replace(/\s/g, "");
  if (!/^\d{6}$/.test(code)) return false;
  const key = base32Decode(secretBase32);
  if (!key || key.length === 0) return false;
  // RFC 6238: 30-second time step, T0 = 0.
  const timeStep = 30;
  const now = Math.floor(Date.now() / 1000);
  const counter = Math.floor(now / timeStep);
  // Check ±1 window for clock skew.
  for (const offset of [-1, 0, 1]) {
    const expected = hotp(key, counter + offset);
    if (timingSafeEqual(Buffer.from(expected), Buffer.from(code))) {
      return true;
    }
  }
  return false;
}

/** HMAC-based One-Time Password (RFC 4226) — the building block of TOTP. */
function hotp(key: Buffer, counter: number): string {
  const buf = Buffer.alloc(8);
  // counter is a 64-bit big-endian integer
  buf.writeBigUInt64BE(BigInt(counter));
  const hmac = createHmac("sha1", key).update(buf).digest();
  const offset = hmac[hmac.length - 1] & 0x0f;
  const bin =
    ((hmac[offset] & 0x7f) << 24) |
    ((hmac[offset + 1] & 0xff) << 16) |
    ((hmac[offset + 2] & 0xff) << 8) |
    (hmac[offset + 3] & 0xff);
  const code = bin % 1_000_000;
  return code.toString().padStart(6, "0");
}

/** Minimal RFC 4648 base32 decoder (no padding required). */
function base32Decode(input: string): Buffer | null {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
  const clean = input.toUpperCase().replace(/=+$/, "").replace(/\s/g, "");
  if (!clean) return null;
  let bits = 0;
  let value = 0;
  const output: number[] = [];
  for (const ch of clean) {
    const idx = alphabet.indexOf(ch);
    if (idx === -1) return null;
    value = (value << 5) | idx;
    bits += 5;
    if (bits >= 8) {
      bits -= 8;
      output.push((value >> bits) & 0xff);
    }
  }
  return Buffer.from(output);
}
