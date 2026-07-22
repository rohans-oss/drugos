/**
 * FE-069 ROOT FIX tests: /api/rl per-user rate limiting.
 *
 * BE-027 ROOT FIX (Team Member 12): the CSV cache tests below were
 * removed because they tested rl-csv-cache.ts — a DUPLICATE cache module
 * that NO production route actually read from. rl-csv-cache.ts has been
 * DELETED; the single source of truth is now rl-ranker.ts (whose cache
 * is exercised by the new be-021-to-040 test suite). The rate-limit
 * tests below are independent of the cache module and remain unchanged.
 *
 * These tests run WITHOUT a database — they exercise the pure rate-limit
 * module directly.
 */

import {
  checkUserRateLimit,
  // FE-017: sync aliases so this existing test suite keeps working without
  // rewriting every beforeEach/test to be async.
  resetUserRateLimitSync as resetUserRateLimit,
  __clearAllUserRateLimitsForTestsSync as __clearAllUserRateLimitsForTests,
} from "@/lib/auth/per-user-rate-limit";

describe("FE-069: per-user rate limiting for /api/rl", () => {
  const USER_ID = "clxxxxxxxxxxxxxxxxxxxx01";

  beforeEach(() => {
    __clearAllUserRateLimitsForTests();
  });

  test("allows up to `max` requests within the window", () => {
    for (let i = 0; i < 60; i++) {
      const rl = checkUserRateLimit(USER_ID, { max: 60, windowSeconds: 60 });
      expect(rl.blocked).toBe(false);
    }
  });

  test("blocks the 61st request within the window with a positive retryAfterSeconds", () => {
    for (let i = 0; i < 60; i++) {
      checkUserRateLimit(USER_ID, { max: 60, windowSeconds: 60 });
    }
    const rl = checkUserRateLimit(USER_ID, { max: 60, windowSeconds: 60 });
    expect(rl.blocked).toBe(true);
    expect(rl.retryAfterSeconds).toBeGreaterThan(0);
    expect(rl.retryAfterSeconds).toBeLessThanOrEqual(60);
    expect(rl.remaining).toBe(0);
  });

  test("rate limit is per-user: a second user is not affected", () => {
    const USER_A = "clxxxxxxxxxxxxxxxxxxxx02";
    const USER_B = "clxxxxxxxxxxxxxxxxxxxx03";
    for (let i = 0; i < 60; i++) {
      checkUserRateLimit(USER_A, { max: 60, windowSeconds: 60 });
    }
    const aBlocked = checkUserRateLimit(USER_A, { max: 60, windowSeconds: 60 });
    const bOk = checkUserRateLimit(USER_B, { max: 60, windowSeconds: 60 });
    expect(aBlocked.blocked).toBe(true);
    expect(bOk.blocked).toBe(false);
  });

  test("resetUserRateLimit clears the user's bucket", () => {
    for (let i = 0; i < 60; i++) {
      checkUserRateLimit(USER_ID, { max: 60, windowSeconds: 60 });
    }
    expect(checkUserRateLimit(USER_ID, { max: 60, windowSeconds: 60 }).blocked).toBe(true);
    resetUserRateLimit(USER_ID);
    const after = checkUserRateLimit(USER_ID, { max: 60, windowSeconds: 60 });
    expect(after.blocked).toBe(false);
    expect(after.remaining).toBe(59);
  });
});

