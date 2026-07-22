/**
 * One-time setup token for 2FA enrollment.
 *
 * FE-071 ROOT FIX: /api/auth/2fa/setup returned the TOTP secret in
 * plaintext JSON. This is necessary for QR-code rendering, but if any XSS
 * exists anywhere in the app, the attacker can read the secret and
 * permanently compromise the user's 2FA — they can call /verify themselves
 * to persist it.
 *
 * Mitigation (this module): issue a short-lived, one-time-use setup token
 * bound to the user's session. The token is returned alongside the secret,
 * but:
 *   1. It can only be used ONCE. After /verify consumes it, a second
 *      attacker request with the same token is rejected.
 *   2. It expires after 5 minutes (TTL).
 *   3. It is bound to the userId — a stolen token cannot be used to enroll
 *      2FA for a different user.
 *
 * This does NOT fully prevent XSS-driven 2FA compromise (an attacker who
 * can read the response can also call /verify immediately), but it DOES:
 *   - Close the replay window (a token sniffed from logs cannot be reused).
 *   - Add a defense-in-depth layer on top of the CSP headers (which are
 *     the primary XSS mitigation).
 *
 * BE-078 ROOT FIX (LIMITED — MULTI-INSTANCE DEPLOYMENTS):
 * The pending-enrollment Map is per-process. In a multi-instance deploy
 * (K8s replicas, etc.), each instance has its own Map. An attacker who
 * sends the same setupToken to TWO instances simultaneously could
 * potentially race both verifications. The actual risk is LOW because:
 *   - The setupToken is bound to the user's authenticated session (only
 *     the legitimate user receives it from /api/auth/2fa/setup).
 *   - The attacker would need to BE the user (or have stolen their
 *     session) AND send the same token to two instances within the
 *     ~5-minute TTL — and the second enrollment would just overwrite
 *     the first (the user's authenticator app shows a different secret
 *     than the server, locking the user out — DoS, not account takeover).
 * The proper fix is to persist setup tokens in a shared store (Redis
 * SETNX, or Postgres with a unique constraint on tokenHash + usedAt IS
 * NULL). Until that's implemented, this module is documented as
 * single-instance only. Operators running multi-instance deploys MUST
 * set up Redis-backed 2FA setup (TODO: BE-078-multi-instance).
 *
 * FE-018 ROOT FIX (Team Member 14, v2 verification): The audit flagged that
 * "two-factor-setup-token.ts generates tokens with crypto.randomBytes(20)
 * but does not expire them". Inspecting the ACTUAL code: the TTL was already
 * 5 minutes (SETUP_TOKEN_TTL_MS = 5 * 60 * 1000) and the expiry was already
 * enforced (verify2faSetupToken rejects with "token_expired" when
 * entry.expiresAt < Date.now()). So the audit was either against a stale
 * version or missed the existing enforcement. This v2:
 *   1. Verifies the existing 5-minute TTL is enforced (it is).
 *   2. Adds a deterministic test helper `__fastForwardTimeForTests(ms)` so
 *      the regression test can verify expiry WITHOUT waiting 5 real minutes.
 *   3. Tightens the entropy from randomBytes(32) — the audit said 20 bytes
 *      (160 bits); the actual code already uses 32 bytes (256 bits), which
 *      exceeds the audit's recommendation. Documented here for clarity.
 *   4. The TTL is 5 minutes, not the audit's suggested 10 minutes — tighter
 *      is better for a setup token (the user is actively enrolling; 5 min
 *      is generous; 10 min extends the replay window unnecessarily).
 *
 * The token is a random 32-byte hex string. We store a SHA-256 hash of it
 * in memory (never the raw token). Lookup is O(1) via Map.
 */

import { createHash, randomBytes } from "crypto";

const SETUP_TOKEN_TTL_MS = 5 * 60 * 1000; // 5 minutes
const MAX_CONCURRENT_TOKENS = 10000; // bounded memory

interface PendingEnrollment {
  userId: string;
  secretHash: string; // sha256(secret) — we don't store the raw secret
  setupTokenHash: string; // sha256(setupToken)
  expiresAt: number; // ms epoch
  usedAt: number | null;
}

// Keyed by setupTokenHash for O(1) lookup. Value carries userId for the
// reverse check.
const pending = new Map<string, PendingEnrollment>();

// Periodic cleanup so the Map doesn't grow unboundedly.
let lastCleanup = Date.now();
const CLEANUP_INTERVAL_MS = 5 * 60 * 1000;

function maybeCleanup() {
  const now = Date.now();
  if (now - lastCleanup < CLEANUP_INTERVAL_MS) return;
  lastCleanup = now;
  for (const [hash, entry] of pending) {
    if (entry.expiresAt < now || entry.usedAt !== null) {
      pending.delete(hash);
    }
  }
}

function sha256(s: string): string {
  return createHash("sha256").update(s).digest("hex");
}

/**
 * Generate a fresh TOTP secret + a one-time setup token bound to `userId`.
 * The secret is NOT stored here — the caller returns it to the client.
 * We store only hashes (defense in depth: a memory dump can't recover
 * either secret).
 *
 * Returns:
 *   - secret: the raw base32 TOTP secret (caller returns to client for QR)
 *   - setupToken: the raw one-time token (caller returns to client)
 *
 * The client must send BOTH secret + setupToken to /api/auth/2fa/verify.
 */
export function issue2faSetupToken(userId: string, secret: string): {
  secret: string;
  setupToken: string;
  expiresAt: number;
} {
  maybeCleanup();

  // Bound memory: if we somehow have >MAX_CONCURRENT_TOKENS pending, evict
  // the oldest. This should never happen in normal use (5-min TTL + cleanup
  // keeps the Map tiny), but defense in depth.
  if (pending.size > MAX_CONCURRENT_TOKENS) {
    let oldestHash: string | null = null;
    let oldestTime = Infinity;
    for (const [hash, entry] of pending) {
      if (entry.expiresAt < oldestTime) {
        oldestTime = entry.expiresAt;
        oldestHash = hash;
      }
    }
    if (oldestHash) pending.delete(oldestHash);
  }

  const setupToken = randomBytes(32).toString("hex");
  // FE-018: use _now() so the test time-offset applies consistently.
  const expiresAt = _now() + SETUP_TOKEN_TTL_MS;
  const entry: PendingEnrollment = {
    userId,
    secretHash: sha256(secret),
    setupTokenHash: sha256(setupToken),
    expiresAt,
    usedAt: null,
  };
  pending.set(entry.setupTokenHash, entry);

  return { secret, setupToken, expiresAt };
}

export interface Verify2faSetupResult {
  ok: boolean;
  reason?: "token_not_found" | "token_used" | "token_expired" | "user_mismatch" | "secret_mismatch";
}

/**
 * Validate a setup token presented by /api/auth/2fa/verify. On success,
 * mark the token as used so it can never be replayed.
 *
 * Checks (in order):
 *   1. Token hash exists in the pending map.
 *   2. Token has not been used (usedAt === null).
 *   3. Token has not expired.
 *   4. The userId on the request matches the userId bound to the token.
 *   5. The secret on the request matches the secret hash bound to the token
 *      (defense in depth: prevents an attacker from substituting their own
 *      secret while reusing a stolen token).
 *
 * On success, marks the entry used and returns { ok: true }. The caller
 * then persists mfaSecret + mfaEnabled on the User row.
 */
export function verify2faSetupToken(
  userId: string,
  secret: string,
  setupToken: string
): Verify2faSetupResult {
  maybeCleanup();

  const tokenHash = sha256(setupToken);
  const entry = pending.get(tokenHash);
  if (!entry) {
    return { ok: false, reason: "token_not_found" };
  }
  if (entry.usedAt !== null) {
    return { ok: false, reason: "token_used" };
  }
  // FE-018: use _now() so the test time-offset applies. The expiry check
  // is the CRITICAL enforcement — without it, a token sniffed from logs
  // or email breaches would be valid forever, letting an attacker complete
  // 2FA setup years later and lock the user out of their account.
  if (entry.expiresAt < _now()) {
    // Evict expired entry.
    pending.delete(tokenHash);
    return { ok: false, reason: "token_expired" };
  }
  if (entry.userId !== userId) {
    return { ok: false, reason: "user_mismatch" };
  }
  if (entry.secretHash !== sha256(secret)) {
    return { ok: false, reason: "secret_mismatch" };
  }

  // Mark as used — one-time enforcement.
  entry.usedAt = _now();
  pending.set(tokenHash, entry);
  return { ok: true };
}

/**
 * Test-only helper: clear all pending tokens. Never call from production.
 */
export function __clear2faSetupTokensForTests(): void {
  pending.clear();
  lastCleanup = Date.now();
  // FE-018: also reset the time offset so the next test starts fresh.
  __timeOffsetMsForTests = 0;
}

// FE-018 ROOT FIX: deterministic time-offset for expiry regression tests.
// Tests cannot wait 5 real minutes for a token to expire. This offset is
// added to Date.now() inside `__nowForTests()` — but ONLY when set via
// `__fastForwardTimeForTests`. Production code paths use the real
// `Date.now()` directly via the `_now()` helper below.
let __timeOffsetMsForTests = 0;

/**
 * Test-only: fast-forward the module's clock by `ms` milliseconds. This
 * lets the regression test verify that an expired token is rejected
 * WITHOUT waiting the real 5-minute TTL. The offset is applied to BOTH
 * the `expiresAt` computation in `issue2faSetupToken` and the
 * expiry check in `verify2faSetupToken`, so the test sees consistent
 * behavior. Call `__clear2faSetupTokensForTests()` in `beforeEach` to
 * reset the offset between tests.
 *
 * NEVER call this from production code — it would let an attacker freeze
 * the clock and keep tokens alive forever.
 */
export function __fastForwardTimeForTests(ms: number): void {
  __timeOffsetMsForTests += ms;
}

/**
 * Internal: the current time, with the test offset applied. Used by both
 * `issue2faSetupToken` and `verify2faSetupToken` so they agree on what
 * "now" is. Production code has `__timeOffsetMsForTests = 0` so this is
 * just `Date.now()`.
 */
function _now(): number {
  return Date.now() + __timeOffsetMsForTests;
}
