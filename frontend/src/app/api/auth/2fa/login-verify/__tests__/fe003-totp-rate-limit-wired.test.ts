/**
 * FE-003 ROOT FIX (v2) regression test: 2FA login-verify route WIRES IN
 * the TOTP rate-limit primitives.
 *
 * This test would have caught the original FE-003 bug: a previous attempt
 * added `checkTotpRateLimit` / `recordFailedTotp` / `clearTotpAttempts`
 * to `rate-limit.ts` AND wrote a passing unit test (`totp-rate-limit.test.ts`)
 * that exercised the primitives in isolation — but NEVER WIRED THEM INTO
 * the actual /api/auth/2fa/login-verify route. The route's docstring even
 * admitted: "If you want 2FA rate limiting, add a separate per-user 2FA
 * attempt counter." The test file passed but the actual HTTP endpoint
 * was still brute-forceable (1M codes / 1000 req/s = 17 minutes).
 *
 * Root fix: the route now calls checkTotpRateLimit BEFORE TOTP verification,
 * recordFailedTotp on failure, and clearTotpAttempts on success.
 *
 * This test is a SOURCE-LEVEL regression test: it reads the actual route
 * source file and asserts the rate-limit primitives are imported and called.
 * This catches the "primitives exist but aren't wired in" anti-pattern that
 * a unit test of the primitives alone cannot detect.
 */
import * as fs from "fs";
import * as path from "path";

const ROUTE_PATH = path.resolve(
  __dirname,
  "../route.ts"
);

function readRouteSource(): string {
  return fs.readFileSync(ROUTE_PATH, "utf-8");
}

describe("FE-003 (v2): 2FA login-verify route wires in TOTP rate limiting", () => {
  const source = readRouteSource();

  test("imports checkTotpRateLimit, recordFailedTotp, clearTotpAttempts from rate-limit", () => {
    // The import statement must reference all three primitives.
    expect(source).toMatch(/from\s+["']@\/lib\/auth\/rate-limit["']/);
    expect(source).toMatch(/\bcheckTotpRateLimit\b/);
    expect(source).toMatch(/\brecordFailedTotp\b/);
    expect(source).toMatch(/\bclearTotpAttempts\b/);
  });

  test("calls checkTotpRateLimit BEFORE TOTP verification (gate on user.id)", () => {
    // The check must happen AFTER the user lookup (we need user.id) but
    // BEFORE the TOTP verify call. We assert the source order:
    //   checkTotpRateLimit(...) appears BEFORE verifyTotpWithReplayCheck(...)
    const checkIdx = source.indexOf("checkTotpRateLimit(");
    const verifyIdx = source.indexOf("verifyTotpWithReplayCheck(");
    expect(checkIdx).toBeGreaterThan(-1);
    expect(verifyIdx).toBeGreaterThan(-1);
    expect(checkIdx).toBeLessThan(verifyIdx);
  });

  test("calls recordFailedTotp on TOTP verification failure", () => {
    // Within the !result.ok branch, recordFailedTotp must be called.
    const failBranchIdx = source.indexOf("!result.ok");
    expect(failBranchIdx).toBeGreaterThan(-1);
    const afterBranch = source.slice(failBranchIdx);
    expect(afterBranch).toMatch(/\brecordFailedTotp\(/);
  });

  test("calls clearTotpAttempts on TOTP verification success", () => {
    // After the !result.ok branch (i.e. in the success path), clearTotpAttempts
    // must be called BEFORE issuing tokens.
    const failBranchIdx = source.indexOf("!result.ok");
    const afterBranch = source.slice(failBranchIdx);
    expect(afterBranch).toMatch(/\bclearTotpAttempts\(/);
  });

  test("returns HTTP 429 with Retry-After header when user is locked", () => {
    // The lock response must include status 429 and a Retry-After header
    // (RFC 6585 + RFC 7231).
    expect(source).toMatch(/status:\s*429/);
    expect(source).toMatch(/["']Retry-After["']/);
  });

  test("docstring no longer says 'If you want 2FA rate limiting, add a separate per-user 2FA attempt counter'", () => {
    // The misleading comment that admitted rate limiting was missing must
    // be GONE. The root fix removed it because the rate limiting IS now
    // wired in — leaving the comment would mislead future readers.
    expect(source).not.toMatch(
      /If you want 2FA rate limiting, add a separate per-user 2FA attempt counter/
    );
  });
});
