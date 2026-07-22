/**
 * FE-006 ROOT FIX unit tests: per-user API proxy rate limiting.
 *
 * Verifies that the 6 public-API-proxy routes (drugs, diseases,
 * clinical-trials, literature, patents, safety) enforce a per-user
 * sliding-window rate limit so a single user cannot drain the platform's
 * NCBI / PatentsView / openFDA API quotas.
 */
import {
  checkUserApiRateLimit,
  recordUserApiRequest,
  __resetUserApiStateForTests,
  USER_API_RATE_LIMIT_PER_MINUTE,
} from "../rate-limit";

describe("FE-006: per-user API proxy rate limiting", () => {
  beforeEach(() => {
    __resetUserApiStateForTests();
  });

  test("a fresh user has the full quota", () => {
    const r = checkUserApiRateLimit("user-A");
    expect(r.blocked).toBe(false);
    expect(r.remaining).toBe(USER_API_RATE_LIMIT_PER_MINUTE);
  });

  test(`blocks after ${USER_API_RATE_LIMIT_PER_MINUTE} requests in the window`, () => {
    const userId = "user-B";
    // Record (LIMIT - 1) requests — should still be under the limit.
    for (let i = 0; i < USER_API_RATE_LIMIT_PER_MINUTE - 1; i++) {
      recordUserApiRequest(userId);
    }
    let check = checkUserApiRateLimit(userId);
    expect(check.blocked).toBe(false);
    expect(check.remaining).toBe(1);

    // The LIMIT-th request triggers the block on the NEXT check.
    recordUserApiRequest(userId);
    check = checkUserApiRateLimit(userId);
    expect(check.blocked).toBe(true);
    expect(check.retryAfterSeconds).toBeGreaterThan(0);
    expect(check.remaining).toBe(0);
  });

  test("rate limit is per-user", () => {
    // Burn user-A's quota.
    for (let i = 0; i < USER_API_RATE_LIMIT_PER_MINUTE; i++) {
      recordUserApiRequest("user-A");
    }
    // User-B is unaffected.
    const checkB = checkUserApiRateLimit("user-B");
    expect(checkB.blocked).toBe(false);
    expect(checkB.remaining).toBe(USER_API_RATE_LIMIT_PER_MINUTE);
  });
});
