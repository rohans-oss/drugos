/**
 * FE-004 ROOT FIX (v2) regression test: password change MUST revoke sessions.
 *
 * This test would have caught the original FE-004 bug: the previous code
 * only called `revokeAllRefreshTokensForUser` INSIDE the `if (!audit.ok)`
 * branch — i.e. it revoked sessions ONLY when the audit log failed. The
 * common path (audit succeeds) left all outstanding refresh tokens valid
 * for 30 days, defeating the entire point of changing a password.
 *
 * Root fix: revocation now happens UNCONDITIONALLY after the password
 * update, plus `clearAuthCookies()` forces the current session to
 * re-authenticate with the new password.
 *
 * This is a SOURCE-LEVEL regression test that reads the actual route
 * source and asserts the unconditional revocation pattern.
 */
import * as fs from "fs";
import * as path from "path";

const ROUTE_PATH = path.resolve(__dirname, "../route.ts");

function readRouteSource(): string {
  return fs.readFileSync(ROUTE_PATH, "utf-8");
}

describe("FE-004 (v2): password change unconditionally revokes all sessions", () => {
  const source = readRouteSource();

  test("imports revokeAllRefreshTokensForUser and clearAuthCookies from auth/server", () => {
    expect(source).toMatch(/\brevokeAllRefreshTokensForUser\b/);
    expect(source).toMatch(/\bclearAuthCookies\b/);
  });

  test("calls revokeAllRefreshTokensForUser UNCONDITIONALLY after db.user.update", () => {
    // The revoke call must appear AFTER the passwordHash update and
    // OUTSIDE any `if (!audit.ok)` guard. We verify:
    //   1. db.user.update appears before revokeAllRefreshTokensForUser.
    //   2. revokeAllRefreshTokensForUser is NOT nested inside an
    //      `if (!audit.ok)` block.
    const updateIdx = source.indexOf("db.user.update");
    expect(updateIdx).toBeGreaterThan(-1);

    const afterUpdate = source.slice(updateIdx);
    const revokeIdx = afterUpdate.indexOf("revokeAllRefreshTokensForUser(");
    expect(revokeIdx).toBeGreaterThan(-1);

    // The revoke call must appear BEFORE the `if (!audit.ok)` check —
    // i.e. it's unconditional, not gated on audit failure.
    const auditCheckIdx = afterUpdate.indexOf("if (!audit.ok)");
    expect(auditCheckIdx).toBeGreaterThan(-1);
    expect(revokeIdx).toBeLessThan(auditCheckIdx);
  });

  test("calls clearAuthCookies() to force re-authentication", () => {
    // The cookie clear must happen AFTER the revoke call.
    const revokeIdx = source.indexOf("revokeAllRefreshTokensForUser(");
    const clearIdx = source.indexOf("clearAuthCookies()");
    expect(revokeIdx).toBeGreaterThan(-1);
    expect(clearIdx).toBeGreaterThan(-1);
    expect(clearIdx).toBeGreaterThan(revokeIdx);
  });

  test("does NOT have the old 'only revoke when audit fails' pattern", () => {
    // The old code had a comment like "We MUST revoke all sessions to
    // force re-login — otherwise an attacker..." INSIDE the !audit.ok
    // branch. The new code revokes unconditionally, so that comment
    // pattern should be gone.
    expect(source).not.toMatch(
      /The password WAS changed, but the audit log failed\. We MUST\s+revoke all sessions to force re-login/
    );
  });
});
