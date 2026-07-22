/**
 * AES-256-GCM encryption-at-rest utility.
 *
 * Used for encrypting sensitive fields persisted to the database (currently
 * WebhookEndpoint.secretEncrypted). The encryption key is sourced from the
 * WEBHOOK_SECRET_KEY environment variable (must be a 32-byte base64 or hex
 * string). If the env var is missing in production we fail-fast — we do NOT
 * fall back to a hardcoded key, because that would defeat the entire point
 * of encryption-at-rest.
 *
 * Ciphertext format: "v1:<base64 iv>:<base64 ciphertext>:<base64 tag>"
 * The "v1" prefix lets us evolve the format later without breaking old rows.
 */

import { createCipheriv, createDecipheriv, randomBytes, createHash, createHmac, timingSafeEqual } from "crypto";

const KEY_ENV_VAR = "WEBHOOK_SECRET_KEY";
const PREFIX = "v1";

function getKey(): Buffer {
  const raw = process.env[KEY_ENV_VAR];
  if (!raw) {
    // BE-063 ROOT FIX (v115, MEDIUM): fail-closed for missing key.
    //
    // ROOT CAUSE: the previous code only threw when
    // `NODE_ENV === "production"`. If NODE_ENV was unset (a common
    // misconfiguration in staging / preview environments), the code
    // fell back to a HARDCODED dev key derived from a publicly-known
    // string. Anyone with repo access could decrypt any secret
    // encrypted with this key. The pattern was a landmine: a future
    // developer importing `encryptSecret` for a new feature in a
    // misconfigured environment would silently encrypt secrets with
    // a publicly-known key.
    //
    // ROOT FIX: invert the check. Instead of "throw only if
    // production", we "throw UNLESS explicitly in development or
    // test". This means:
    //   - NODE_ENV=production → throw (correct)
    //   - NODE_ENV=staging → throw (correct — staging needs a real key)
    //   - NODE_ENV=preview → throw (correct — preview needs a real key)
    //   - NODE_ENV=development → dev key (acceptable for local dev)
    //   - NODE_ENV=test → dev key (acceptable for unit tests)
    //   - NODE_ENV unset → throw (correct — fail-closed)
    const isDev = process.env.NODE_ENV === "development" || process.env.NODE_ENV === "test";
    if (!isDev) {
      throw new Error(
        `${KEY_ENV_VAR} must be set when NODE_ENV is not "development" or "test" ` +
        `(current NODE_ENV=${JSON.stringify(process.env.NODE_ENV)}). ` +
        `Generate a 32-byte key with: openssl rand -base64 32`
      );
    }
    // Dev-only deterministic key so the app doesn't crash during local dev.
    // NEVER used in production — the throw above gates this.
    return createHash("sha256").update("drugos-dev-only-webhook-key-not-for-production").digest();
  }
  // Accept either base64 or hex.
  let buf: Buffer;
  if (/^[0-9a-fA-F]{64}$/.test(raw)) {
    buf = Buffer.from(raw, "hex");
  } else {
    try {
      buf = Buffer.from(raw, "base64");
    } catch {
      throw new Error(`${KEY_ENV_VAR} must be valid base64 or hex`);
    }
  }
  if (buf.length !== 32) {
    throw new Error(
      `${KEY_ENV_VAR} must decode to exactly 32 bytes (got ${buf.length}). ` +
      `Generate with: openssl rand -base64 32`
    );
  }
  return buf;
}

/**
 * Encrypt a plaintext string using AES-256-GCM.
 * Returns "v1:<base64 iv>:<base64 ciphertext>:<base64 tag>".
 */
export function encryptSecret(plaintext: string): string {
  if (typeof plaintext !== "string" || plaintext.length === 0) {
    throw new Error("plaintext must be a non-empty string");
  }
  const key = getKey();
  const iv = randomBytes(12); // 96-bit IV is recommended for GCM
  const cipher = createCipheriv("aes-256-gcm", key, iv);
  const ciphertext = Buffer.concat([
    cipher.update(plaintext, "utf8"),
    cipher.final(),
  ]);
  const tag = cipher.getAuthTag();
  return [
    PREFIX,
    iv.toString("base64"),
    ciphertext.toString("base64"),
    tag.toString("base64"),
  ].join(":");
}

/**
 * Decrypt a ciphertext produced by encryptSecret().
 * Returns the original plaintext, or throws if the value is tampered/invalid.
 */
export function decryptSecret(stored: string): string {
  if (typeof stored !== "string" || stored.length === 0) {
    throw new Error("stored ciphertext is empty");
  }
  const parts = stored.split(":");
  if (parts.length !== 4 || parts[0] !== PREFIX) {
    throw new Error(
      `Unrecognized ciphertext format (expected "${PREFIX}:iv:ct:tag")`
    );
  }
  const [, ivB64, ctB64, tagB64] = parts;
  const key = getKey();
  const iv = Buffer.from(ivB64, "base64");
  const ct = Buffer.from(ctB64, "base64");
  const tag = Buffer.from(tagB64, "base64");
  const decipher = createDecipheriv("aes-256-gcm", key, iv);
  decipher.setAuthTag(tag);
  const plaintext = Buffer.concat([decipher.update(ct), decipher.final()]);
  return plaintext.toString("utf8");
}

/**
 * Returns true if a stored value looks like an encrypted secret (i.e. starts
 * with "v1:"). Used by migrations to detect rows that still hold plaintext.
 */
export function isEncryptedSecret(stored: string): boolean {
  return typeof stored === "string" && stored.startsWith(`${PREFIX}:`);
}

// ---------------------------------------------------------------------------
// FE-015 ROOT FIX (Team Member 14): Constant-time HMAC computation/verification.
//
// Previously this file had NO `verifyHmac()` helper at all — webhook routes
// that needed HMAC verification would have reached for `===` (string equality),
// which short-circuits on the first differing byte. An attacker who can
// measure response time can determine the expected HMAC byte-by-byte: for a
// 32-byte HMAC this is ~256 * 32 = 8192 requests — feasible in a few hours
// on a quiet endpoint. Once forged, the attacker can inject fake webhook
// payloads (e.g. fake Stripe events, fake GitHub push events) and trigger
// arbitrary state changes in the platform.
//
// This module provides:
//   - `computeHmac(key, message, algorithm?)` — returns a hex digest.
//   - `verifyHmac(key, message, expectedHex, algorithm?)` — CONSTANT-TIME
//     comparison via `crypto.timingSafeEqual`. Returns true iff the computed
//     HMAC equals `expectedHex` AND both buffers are the same length.
//
// The comparison is constant-time because:
//   1. We compute the HMAC fresh on every call (the expected value is the
//      attacker-controlled input; the computed value is the secret-derived
//      ground truth). The attacker cannot short-circuit the computation.
//   2. We compare the two buffers with `timingSafeEqual`, which iterates
//      ALL bytes before returning — there is no early exit.
//   3. We guard the length check explicitly because `timingSafeEqual`
//      throws on mismatched lengths (which itself is a timing side-channel
//      if not handled — but the length of the expected HMAC is fixed by
//      the algorithm, so a length mismatch already means the attacker
//      sent malformed input).
//
// Usage (webhook signature verification):
//   const expected = req.headers.get("x-hub-signature-256")?.replace("sha256=", "");
//   const body = await req.text();
//   const ok = verifyHmac(webhookSecret, body, expected ?? "", "sha256");
//   if (!ok) return NextResponse.json({ error: "invalid_signature" }, { status: 401 });
// ---------------------------------------------------------------------------

/**
 * Compute the HMAC of `message` using `key`. Returns a lowercase hex digest.
 *
 * Defaults to SHA-256 (the algorithm used by GitHub's x-hub-signature-256
 * and Stripe's Stripe-Signature headers). Supports any algorithm supported
 * by Node's `crypto.createHmac` (sha1, sha256, sha512, etc.).
 *
 * The key may be passed as a string (UTF-8 encoded) or a Buffer. The
 * message may also be a string or Buffer.
 */
export function computeHmac(
  key: string | Buffer,
  message: string | Buffer,
  algorithm: string = "sha256"
): string {
  const keyBuf = typeof key === "string" ? Buffer.from(key, "utf8") : key;
  const msgBuf = typeof message === "string" ? Buffer.from(message, "utf8") : message;
  return createHmac(algorithm, keyBuf).update(msgBuf).digest("hex");
}

/**
 * Verify that `expectedHex` matches the HMAC of `message` under `key`, using
 * a CONSTANT-TIME comparison to prevent timing attacks.
 *
 * Returns `true` if and only if:
 *   - `expectedHex` is a non-empty string,
 *   - the computed HMAC has the same byte length as `expectedHex` (i.e. the
 *     caller is using the correct algorithm — sha256 → 64 hex chars / 32
 *     bytes, sha1 → 40 hex chars / 20 bytes, etc.),
 *   - and the two buffers are byte-for-byte equal under `timingSafeEqual`.
 *
 * Returns `false` for any malformed input (empty, wrong length, non-hex).
 * This fail-closed behavior ensures a forged or truncated signature cannot
 * bypass verification by exploiting edge cases in the comparison.
 *
 * IMPORTANT: this function does NOT protect against length-extension
 * attacks on sha1/sha256 — but HMAC itself is immune to length-extension
 * by construction (the key is inner-and-outer-padded), so this is not a
 * concern. The constant-time comparison is the only timing side-channel
 * that matters here.
 */
export function verifyHmac(
  key: string | Buffer,
  message: string | Buffer,
  expectedHex: string,
  algorithm: string = "sha256"
): boolean {
  // Fail closed on malformed input — never throw, never leak whether the
  // expected value was empty vs. wrong-length vs. wrong-content.
  if (typeof expectedHex !== "string" || expectedHex.length === 0) {
    return false;
  }
  const computed = computeHmac(key, message, algorithm);
  const a = Buffer.from(computed, "utf8");
  const b = Buffer.from(expectedHex, "utf8");
  // Length mismatch: return false WITHOUT calling timingSafeEqual (it
  // throws on length mismatch). We do not early-return on the CONTENT —
  // only on the length, which is fixed by the algorithm and thus not a
  // secret. The attacker already knows which algorithm we use (it's in
  // the header name: x-hub-signature-256 → sha256).
  if (a.length !== b.length) {
    return false;
  }
  // Constant-time comparison — iterates all bytes before returning.
  return timingSafeEqual(a, b);
}
