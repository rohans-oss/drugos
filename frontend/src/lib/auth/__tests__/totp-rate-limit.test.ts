/**
 * FE-003 ROOT FIX unit tests: TOTP brute-force rate limiting.
 *
 * These tests verify that the per-user TOTP attempt counter:
 *   1. Allows up to TOTP_MAX_ATTEMPTS wrong codes within the window.
 *   2. Locks the 5th wrong code (returns locked=true).
 *   3. Continues to return locked=true for subsequent attempts until the
 *      lock window expires.
 *   4. Resets on a successful verification (clearTotpAttempts).
 *
 * The test file deliberately does NOT exercise the HTTP route — it tests
 * the rate-limit primitives directly so the test is deterministic and
 * fast (no DB, no JWT, no TOTP secret needed).
 */
import {
  checkTotpRateLimit,
  recordFailedTotp,
  clearTotpAttempts,
  __resetTotpStateForTests,
  TOTP_MAX_ATTEMPTS,
} from "../rate-limit";

describe("FE-003: TOTP brute-force rate limiting", () => {
  beforeEach(() => {
    __resetTotpStateForTests();
  });

  test("a fresh user is not locked", () => {
    const result = checkTotpRateLimit("user-1");
    expect(result.locked).toBe(false);
    expect(result.retryAfterSeconds).toBe(0);
  });

  test(`locks after exactly ${TOTP_MAX_ATTEMPTS} wrong codes`, () => {
    const userId = "user-2";
    // Attempts 1..TOTP_MAX_ATTEMPTS-1 should NOT lock.
    for (let i = 1; i < TOTP_MAX_ATTEMPTS; i++) {
      const r = recordFailedTotp(userId);
      expect(r.locked).toBe(false);
      expect(r.attemptsRemaining).toBe(TOTP_MAX_ATTEMPTS - i);
    }
    // The TOTP_MAX_ATTEMPTS-th wrong code SHOULD lock.
    const final = recordFailedTotp(userId);
    expect(final.locked).toBe(true);
    expect(final.attemptsRemaining).toBe(0);
    expect(final.retryAfterSeconds).toBeGreaterThan(0);
  });

  test("checkTotpRateLimit returns locked after the threshold is crossed", () => {
    const userId = "user-3";
    for (let i = 0; i < TOTP_MAX_ATTEMPTS; i++) {
      recordFailedTotp(userId);
    }
    const check = checkTotpRateLimit(userId);
    expect(check.locked).toBe(true);
    expect(check.retryAfterSeconds).toBeGreaterThan(0);
  });

  test("clearTotpAttempts resets the counter", () => {
    const userId = "user-4";
    for (let i = 0; i < TOTP_MAX_ATTEMPTS - 1; i++) {
      recordFailedTotp(userId);
    }
    clearTotpAttempts(userId);
    // After clear, the user is fresh.
    const check = checkTotpRateLimit(userId);
    expect(check.locked).toBe(false);
    // And the first wrong code shows full attempts remaining.
    const r = recordFailedTotp(userId);
    expect(r.attemptsRemaining).toBe(TOTP_MAX_ATTEMPTS - 1);
  });

  test("rate limit is per-user (different users do not interfere)", () => {
    // Burn all attempts for user-A.
    for (let i = 0; i < TOTP_MAX_ATTEMPTS; i++) {
      recordFailedTotp("user-A");
    }
    // User-B is still fresh.
    const checkB = checkTotpRateLimit("user-B");
    expect(checkB.locked).toBe(false);
  });
});
