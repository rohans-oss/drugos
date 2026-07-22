/**
 * FE-071 ROOT FIX tests: 2FA setup one-time token.
 *
 * Verifies:
 *   1. issue2faSetupToken returns a setupToken alongside the secret.
 *   2. verify2faSetupToken accepts the correct (userId, secret, setupToken).
 *   3. A token cannot be reused (one-time enforcement).
 *   4. A token bound to user A cannot be used by user B (anti-replay across
 *      accounts).
 *   5. A token with a substituted secret is rejected (defense in depth).
 *   6. An attacker-supplied setupToken (random hex) is rejected with
 *      "token_not_found".
 */

import {
  issue2faSetupToken,
  verify2faSetupToken,
  __clear2faSetupTokensForTests,
} from "@/lib/auth/two-factor-setup-token";

describe("FE-071: 2FA setup one-time token", () => {
  const USER_A = "curusera000000000000000001";
  const USER_B = "curuserb000000000000000002";
  const SECRET_A = "JBSWY3DPEHPK3PXPAAAAAAAA"; // base32, 20 bytes

  beforeEach(() => {
    __clear2faSetupTokensForTests();
  });

  test("issue2faSetupToken returns a setupToken + the original secret", () => {
    const result = issue2faSetupToken(USER_A, SECRET_A);
    expect(result.secret).toBe(SECRET_A);
    expect(result.setupToken).toMatch(/^[0-9a-f]{64}$/); // 32-byte hex
    expect(result.expiresAt).toBeGreaterThan(Date.now());
  });

  test("verify2faSetupToken accepts the correct triple", () => {
    const { secret, setupToken } = issue2faSetupToken(USER_A, SECRET_A);
    const result = verify2faSetupToken(USER_A, secret, setupToken);
    expect(result.ok).toBe(true);
  });

  test("a token CANNOT be reused (one-time enforcement)", () => {
    const { secret, setupToken } = issue2faSetupToken(USER_A, SECRET_A);
    const first = verify2faSetupToken(USER_A, secret, setupToken);
    expect(first.ok).toBe(true);
    const second = verify2faSetupToken(USER_A, secret, setupToken);
    expect(second.ok).toBe(false);
    expect(second.reason).toBe("token_used");
  });

  test("a token bound to user A is rejected for user B", () => {
    const { secret, setupToken } = issue2faSetupToken(USER_A, SECRET_A);
    const result = verify2faSetupToken(USER_B, secret, setupToken);
    expect(result.ok).toBe(false);
    expect(result.reason).toBe("user_mismatch");
  });

  test("a token with a SUBSTITUTED secret is rejected (defense in depth)", () => {
    const { setupToken } = issue2faSetupToken(USER_A, SECRET_A);
    const attackerSecret = "KRSXG5BAONUGC4TFFYYYYYYY";
    const result = verify2faSetupToken(USER_A, attackerSecret, setupToken);
    expect(result.ok).toBe(false);
    expect(result.reason).toBe("secret_mismatch");
  });

  test("an attacker-forged random setupToken is rejected with token_not_found", () => {
    const forgedToken = "a".repeat(64); // 32-byte hex, never issued
    const result = verify2faSetupToken(USER_A, SECRET_A, forgedToken);
    expect(result.ok).toBe(false);
    expect(result.reason).toBe("token_not_found");
  });
});
