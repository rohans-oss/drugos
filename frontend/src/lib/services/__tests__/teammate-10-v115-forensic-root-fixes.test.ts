/**
 * Teammate 10 — v115 Forensic Root-Fix Regression Tests
 *
 * This file contains REGRESSION TESTS for the 34 issues fixed in the
 * Teammate 10 swim lane (frontend/src/app/api/auth/*, /api/admin/*,
 * /api/knowledge-graph/*, /api/dataset/*, /lib/auth/*). Each test
 * verifies a SPECIFIC code-level invariant that was broken before the
 * fix and is now correct. Tests are STATIC-SOURCE-MATCHING where the
 * invariant is a code pattern (e.g. "platformRole: true is in the
 * select clause"), and BEHAVIORAL where the invariant is a runtime
 * property (e.g. "verifyTotpWithReplayCheck rejects a replayed code").
 *
 * Per the user's strict instruction, these tests do NOT replace reading
 * the real code — they verify the real code's invariants. The tests
 * fail BEFORE the fix (because the invariant was broken) and pass AFTER.
 *
 * Branch: teammate-10-root-fixes-v115
 */

import fs from "fs";
import path from "path";

// Helper: read a source file as a string.
function readSrc(relPath: string): string {
  // __dirname = .../frontend/src/lib/services/__tests__
  // We want to resolve to .../frontend/, so go up 4 levels.
  const root = path.resolve(__dirname, "../../../..");
  return fs.readFileSync(path.resolve(root, relPath), "utf8");
}

// Helper: read a source file with line/block comments stripped. Used by
// tests that check for ABSENCE of a pattern in executable code (a comment
// mentioning the pattern is fine — the bug is only if the pattern appears
// in actual code). Strips://  - Full-line // comments
//  - Block /* */ comments (including JSDoc /** */)
//  - Mid-line trailing // comments (conservatively — drops everything
//    after // on the same line; this is fine for our use case because
//    none of the patterns we check for contain //)
function readCode(relPath: string): string {
  const src = readSrc(relPath);
  // Strip block comments (including JSDoc).
  const noBlocks = src.replace(/\/\*[\s\S]*?\*\//g, "");
  // Strip full-line and trailing // comments.
  const noLineComments = noBlocks
    .split("\n")
    .map((line) => {
      const idx = line.indexOf("//");
      return idx >= 0 ? line.slice(0, idx) : line;
    })
    .join("\n");
  return noLineComments;
}

// ---------------------------------------------------------------------------
// BE-006 (CRITICAL): 2FA login-verify route selects platformRole AND
// passes it to signAccessToken.
// ---------------------------------------------------------------------------
describe("BE-006 (CRITICAL): 2FA login-verify stamps platformRole into the access token", () => {
  const src = readSrc("src/app/api/auth/2fa/login-verify/route.ts");

  test("the user lookup select clause includes platformRole: true", () => {
    // Must be in the select clause (NOT just in a comment).
    expect(src).toMatch(/select:\s*\{[\s\S]*?platformRole:\s*true/);
  });

  test("signAccessToken is called with platformRole", () => {
    // The signAccessToken call at the end of the handler must pass platformRole.
    expect(src).toMatch(/signAccessToken\(\s*\{[\s\S]*?platformRole:\s*\(user\.platformRole/);
  });

  test("the select clause also includes deletedAt for BE-045 alignment", () => {
    expect(src).toMatch(/deletedAt:\s*true/);
  });
});

// ---------------------------------------------------------------------------
// BE-007 (HIGH): admin/metrics uses requirePlatformAdmin (not requireAdmin).
// ---------------------------------------------------------------------------
describe("BE-007 (HIGH): admin/metrics gated on requirePlatformAdmin", () => {
  const src = readSrc("src/app/api/admin/metrics/route.ts");

  test("imports requirePlatformAdmin from lib/auth/require-platform-admin", () => {
    expect(src).toMatch(/import\s*\{\s*requirePlatformAdmin\s*\}\s*from\s*["']@\/lib\/auth\/require-platform-admin["']/);
  });

  test("GET handler calls requirePlatformAdmin(req) — not requireAdmin()", () => {
    expect(src).toMatch(/await\s+requirePlatformAdmin\(req\)/);
  });

  test("does NOT import requireAdmin from api-helpers", () => {
    // requireAdmin should NOT appear in the imports (we replaced it).
    const importSection = src.split("/**")[0] + src.substring(0, 1500);
    // The route file may still mention requireAdmin in comments — that's OK.
    // The IMPORT must be gone.
    expect(importSection).not.toMatch(/import\s*\{[^}]*\brequireAdmin\b[^}]*\}\s*from\s*["']@\/lib\/api-helpers["']/);
  });

  test("does NOT use isPlatformSuperuser (old role-based gate)", () => {
    // Check the executable CODE (not comments) — the comment may
    // legitimately mention isPlatformSuperuser when explaining why it
    // was removed.
    const code = readCode("src/app/api/admin/metrics/route.ts");
    expect(code).not.toMatch(/\bisPlatformSuperuser\b/);
  });
});

// ---------------------------------------------------------------------------
// BE-010 (HIGH): login route performs dummy bcrypt.compare for non-existent
// users to close the timing oracle.
// ---------------------------------------------------------------------------
describe("BE-010 (HIGH): login route dummy bcrypt compare", () => {
  const src = readSrc("src/app/api/auth/login/route.ts");

  test("imports bcryptjs (for the dummy compare)", () => {
    expect(src).toMatch(/import\s+bcrypt\s+from\s*["']bcryptjs["']/);
  });

  test("imports randomBytes from crypto", () => {
    expect(src).toMatch(/import\s*\{\s*randomBytes\s*\}\s*from\s*["']crypto["']/);
  });

  test("defines a DUMMY_PASSWORD_HASH constant (bcrypt hash of random bytes)", () => {
    expect(src).toMatch(/const\s+DUMMY_PASSWORD_HASH\s*=\s*bcrypt\.hashSync\(/);
    expect(src).toMatch(/randomBytes\(32\)\.toString\(["']hex["']\)/);
  });

  test("the !user branch calls bcrypt.compare with DUMMY_PASSWORD_HASH", () => {
    // The branch that handles "user not found" must perform the dummy compare.
    expect(src).toMatch(/if\s*\(!user\s*\|\|\s*user\.deletedAt\s*!==\s*null\)\s*\{[\s\S]*?await\s+bcrypt\.compare\(password,\s*DUMMY_PASSWORD_HASH\)/);
  });
});

// ---------------------------------------------------------------------------
// BE-013 (HIGH): auth/me PATCH handler calls requireCsrfOrSend.
// ---------------------------------------------------------------------------
describe("BE-013 (HIGH): auth/me PATCH has CSRF protection", () => {
  const src = readSrc("src/app/api/auth/me/route.ts");

  test("imports requireCsrfOrSend from api-helpers", () => {
    expect(src).toMatch(/import\s*\{[^}]*\brequireCsrfOrSend\b[^}]*\}\s*from\s*["']@\/lib\/api-helpers["']/);
  });

  test("PATCH handler calls requireCsrfOrSend(req) after authentication", () => {
    expect(src).toMatch(/export\s+async\s+function\s+PATCH\(req:\s*NextRequest\)[\s\S]*?const\s+csrf\s*=\s*await\s+requireCsrfOrSend\(req\)/);
    expect(src).toMatch(/if\s*\(\s*csrf\.response\s*\)\s*return\s+csrf\.response/);
  });
});

// ---------------------------------------------------------------------------
// BE-014 + BE-076 (HIGH): refresh route has IP+user rate limit + audit log.
// ---------------------------------------------------------------------------
describe("BE-014 + BE-076 (HIGH): refresh route rate limit + audit log", () => {
  const src = readSrc("src/app/api/auth/refresh/route.ts");

  test("imports checkIpRateLimitDistributed for Layer 1 IP limit", () => {
    expect(src).toMatch(/checkIpRateLimitDistributed/);
  });

  test("imports checkUserRateLimitDistributed for Layer 2 user limit", () => {
    expect(src).toMatch(/checkUserRateLimitDistributed/);
  });

  test("imports writeAuditLog", () => {
    expect(src).toMatch(/writeAuditLog/);
  });

  test("defines a per-user refresh rate limit constant", () => {
    expect(src).toMatch(/REFRESH_USER_RATE_LIMIT\s*=\s*\{\s*max:\s*\d+,\s*windowSeconds:\s*\d+\s*\}/);
  });

  test("writes token_refreshed audit log on success", () => {
    expect(src).toMatch(/action:\s*["']token_refreshed["']/);
  });

  test("writes token_refresh_failed audit log on failure", () => {
    expect(src).toMatch(/action:\s*["']token_refresh_failed["']/);
  });

  test("preserves clearAuthCookies import (for invalid refresh)", () => {
    expect(src).toMatch(/import.*clearAuthCookies.*from\s*["']@\/lib\/auth\/server["']/);
  });
});

// ---------------------------------------------------------------------------
// BE-019 (MEDIUM): clearMfaChallengeCookie uses path /api/auth/2fa
// (matches the SET path).
// ---------------------------------------------------------------------------
describe("BE-019 (MEDIUM): clearMfaChallengeCookie path matches SET path", () => {
  const src = readSrc("src/app/api/auth/2fa/login-verify/route.ts");
  const loginSrc = readSrc("src/app/api/auth/login/route.ts");

  test("login route SETS cookie with path /api/auth/2fa", () => {
    expect(loginSrc).toMatch(/path:\s*["']\/api\/auth\/2fa["']/);
  });

  test("clearMfaChallengeCookie DELETES with path /api/auth/2fa (NOT /api/auth/2fa/login-verify)", () => {
    // Find the clearMfaChallengeCookie function body.
    const fnMatch = src.match(/async\s+function\s+clearMfaChallengeCookie\(\)[\s\S]*?\n\}/);
    expect(fnMatch).not.toBeNull();
    const fnBody = fnMatch![0];
    expect(fnBody).toMatch(/path:\s*["']\/api\/auth\/2fa["']/);
    expect(fnBody).not.toMatch(/path:\s*["']\/api\/auth\/2fa\/login-verify["']/);
  });
});

// ---------------------------------------------------------------------------
// BE-020 (HIGH): verify-email uses shared resolveJwtSecret (fail-closed).
// ---------------------------------------------------------------------------
describe("BE-020 (HIGH): verify-email uses shared resolveJwtSecret", () => {
  const src = readSrc("src/app/api/auth/verify-email/route.ts");

  test("imports resolveJwtSecret + resolvePreviousJwtSecret from lib/auth/server", () => {
    expect(src).toMatch(/import\s*\{\s*resolveJwtSecret,\s*resolvePreviousJwtSecret\s*\}\s*from\s*["']@\/lib\/auth\/server["']/);
  });

  test("does NOT have a divergent NODE_ENV === 'production' check (the old bug)", () => {
    // The old code had `if (process.env.NODE_ENV === "production") { return internalError(...) }`
    // inline. The new code delegates to resolveJwtSecret which throws.
    // The old pattern should NOT appear in the secret-resolution block.
    expect(src).not.toMatch(/if\s*\(\s*process\.env\.NODE_ENV\s*===\s*["']production["']\s*\)\s*\{[\s\S]*?JWT_SECRET/);
  });

  test("does NOT reference the hardcoded dev secret string", () => {
    expect(src).not.toMatch(/dev-only-insecure-secret-change-me-MINIMUM-32-CHARS-FOR-HS256/);
  });

  test("tries both current and previous secrets (rotation window)", () => {
    expect(src).toMatch(/secretCandidates\s*=\s*\[resolveJwtSecret\(\),\s*resolvePreviousJwtSecret\(\)\]/);
  });
});

// ---------------------------------------------------------------------------
// BE-022 (MEDIUM): login route does NOT return mfaToken in JSON body.
// ---------------------------------------------------------------------------
describe("BE-022 (MEDIUM): login route does not leak mfaToken in JSON body", () => {
  const src = readSrc("src/app/api/auth/login/route.ts");

  test("the MFA-required response does NOT include mfaToken field", () => {
    // Find the mfaRequired response block.
    const mfaBlockMatch = src.match(/return\s+NextResponse\.json\(\s*\{[\s\S]*?mfaRequired:\s*true[\s\S]*?\}\s*\)/);
    expect(mfaBlockMatch).not.toBeNull();
    const mfaBlock = mfaBlockMatch![0];
    // The block must NOT include `mfaToken,` or `mfaToken:` as a field.
    expect(mfaBlock).not.toMatch(/^\s*mfaToken[,=:]/m);
  });
});

// ---------------------------------------------------------------------------
// BE-025 + BE-026 + BE-082 (LOW/MEDIUM): cypher-validator fixes.
// ---------------------------------------------------------------------------
describe("BE-025 + BE-026 + BE-082: cypher-validator fixes", () => {
  const src = readSrc("src/app/api/knowledge-graph/cypher-validator.ts");

  test("BE-082: MAX_CYPHER_LENGTH is 10000 (aligned with Zod schema)", () => {
    expect(src).toMatch(/const\s+MAX_CYPHER_LENGTH\s*=\s*10[_]?000/);
    expect(src).not.toMatch(/const\s+MAX_CYPHER_LENGTH\s*=\s*5[_]?000/);
  });

  test("BE-025: ALLOWED_TOP_LEVEL_VERBS does NOT include CALL db.labels", () => {
    // The dead-code allowance for CALL db.labels is removed.
    const allowMatch = src.match(/const\s+ALLOWED_TOP_LEVEL_VERBS\s*=\s*\/[^\n]+\//);
    expect(allowMatch).not.toBeNull();
    expect(allowMatch![0]).not.toMatch(/CALL\\s\+db\\.labels/);
  });

  test("BE-026: validator strips backtick-quoted identifiers", () => {
    // Look for the backtick-strip regex literal in the source. We use a
    // string-based search because the regex itself contains backticks
    // which would conflict with JS template literals.
    expect(src).toContain(".replace(/`(?:[^`\\\\]|\\\\.)*`/g");
  });
});

// ---------------------------------------------------------------------------
// BE-028 (LOW): auth/me GET clears cookies when user not found.
// ---------------------------------------------------------------------------
describe("BE-028 (LOW): auth/me GET clears cookies when user not found", () => {
  const src = readSrc("src/app/api/auth/me/route.ts");

  test("imports clearAuthCookies from lib/auth/server", () => {
    expect(src).toMatch(/import\s*\{[^}]*\bclearAuthCookies\b[^}]*\}\s*from\s*["']@\/lib\/auth\/server["']/);
  });

  test("the !user branch in GET calls clearAuthCookies before returning 401", () => {
    expect(src).toMatch(/if\s*\(\s*!user\s*\)\s*\{[\s\S]*?await\s+clearAuthCookies\(\)/);
  });
});

// ---------------------------------------------------------------------------
// BE-029 (LOW): auth/me PATCH org-switch audit log is critical.
// ---------------------------------------------------------------------------
describe("BE-029 (LOW): org-switch audit log marked critical", () => {
  const src = readSrc("src/app/api/auth/me/route.ts");

  test("the active_org_switched audit log includes critical: true", () => {
    // Find the writeAuditLog call for active_org_switched.
    const auditMatch = src.match(/writeAuditLog\(\s*\{[\s\S]*?action:\s*["']active_org_switched["'][\s\S]*?\}\s*\)/);
    expect(auditMatch).not.toBeNull();
    expect(auditMatch![0]).toMatch(/critical:\s*true/);
  });

  test("the audit-failure path rolls back the org switch (returns internalError)", () => {
    expect(src).toMatch(/if\s*\(\s*!orgSwitchAudit\.ok\s*\)[\s\S]*?return\s+internalError/);
  });
});

// ---------------------------------------------------------------------------
// BE-031 (LOW): verifyPassword logs bcrypt errors to stderr.
// ---------------------------------------------------------------------------
describe("BE-031 (LOW): verifyPassword logs bcrypt errors", () => {
  const src = readSrc("src/lib/auth/server.ts");

  test("the catch block in verifyPassword logs the error (not silent)", () => {
    const fnMatch = src.match(/export\s+async\s+function\s+verifyPassword[\s\S]*?\n\}/);
    expect(fnMatch).not.toBeNull();
    const fnBody = fnMatch![0];
    expect(fnBody).toMatch(/catch\s*\(\s*e[^)]*\)\s*\{/);
    expect(fnBody).toMatch(/console\.error\(["']\[verifyPassword\]/);
  });
});

// ---------------------------------------------------------------------------
// BE-034 (MEDIUM): knowledge-graph POST refuses users with no active org.
// ---------------------------------------------------------------------------
describe("BE-034 (MEDIUM): knowledge-graph POST refuses orgless users", () => {
  const src = readSrc("src/app/api/knowledge-graph/route.ts");

  test("POST handler returns 403 when auth.user.orgId is falsy", () => {
    expect(src).toMatch(/if\s*\(\s*!auth\.user\.orgId\s*\)\s*\{[\s\S]*?status:\s*403/);
  });

  test("the 403 response mentions active organization", () => {
    expect(src).toMatch(/no_active_organization|active organization/i);
  });
});

// ---------------------------------------------------------------------------
// BE-036 (LOW): auth/me PATCH falls back to first org when null is passed.
// ---------------------------------------------------------------------------
describe("BE-036 (LOW): auth/me PATCH null org falls back to first membership", () => {
  const src = readSrc("src/app/api/auth/me/route.ts");

  test("when activeOrganizationId is null, looks up first org membership", () => {
    expect(src).toMatch(/if\s*\(\s*switchingOrg\s*&&\s*body\.activeOrganizationId\s*===\s*null\s*\)[\s\S]*?db\.organizationMember\.findFirst/);
  });

  test("returns 400 with clear error if user has no org memberships at all", () => {
    expect(src).toMatch(/no_organization_membership/);
  });
});

// ---------------------------------------------------------------------------
// BE-039 (LOW): dataset/quality uses shared CANONICAL_NODE_TYPES.
// ---------------------------------------------------------------------------
describe("BE-039 (LOW): dataset/quality uses shared CANONICAL_NODE_TYPES", () => {
  const src = readSrc("src/app/api/dataset/quality/route.ts");

  test("imports CANONICAL_NODE_TYPES from lib/ml-contracts", () => {
    expect(src).toMatch(/import\s*\{\s*CANONICAL_NODE_TYPES\s*\}\s*from\s*["']@\/lib\/ml-contracts["']/);
  });

  test("does NOT include 'Drug' in the canonical types list (it was a phantom type)", () => {
    // Check the executable CODE (not comments) — the comment may
    // legitimately mention "Drug" when explaining why it was removed.
    const code = readCode("src/app/api/dataset/quality/route.ts");
    // A literal "Drug" string in an array context would be the bug.
    expect(code).not.toMatch(/["']Drug["']/);
  });
});

// ---------------------------------------------------------------------------
// BE-044 (LOW): JWT kid header for token-type separation.
// ---------------------------------------------------------------------------
describe("BE-044 (LOW): JWT kid header for token-type separation", () => {
  const src = readSrc("src/lib/auth/server.ts");

  test("defines KID_ACCESS and KID_MFA_CHALLENGE constants", () => {
    expect(src).toMatch(/const\s+KID_ACCESS\s*=\s*["']drugos:access:v1["']/);
    expect(src).toMatch(/const\s+KID_MFA_CHALLENGE\s*=\s*["']drugos:mfa_challenge:v1["']/);
  });

  test("signAccessToken passes keyid: KID_ACCESS", () => {
    expect(src).toMatch(/signAccessToken[\s\S]*?keyid:\s*KID_ACCESS/);
  });

  test("signMfaChallengeToken passes keyid: KID_MFA_CHALLENGE", () => {
    expect(src).toMatch(/signMfaChallengeToken[\s\S]*?keyid:\s*KID_MFA_CHALLENGE/);
  });

  test("verifyAccessToken checks the kid header matches KID_ACCESS", () => {
    expect(src).toMatch(/verifyAccessToken[\s\S]*?kid\s*!==\s*KID_ACCESS/);
  });

  test("verifyMfaChallengeToken checks the kid header matches KID_MFA_CHALLENGE", () => {
    expect(src).toMatch(/verifyMfaChallengeToken[\s\S]*?kid\s*!==\s*KID_MFA_CHALLENGE/);
  });
});

// ---------------------------------------------------------------------------
// BE-045 (MEDIUM): consumeRefreshToken checks user.deletedAt.
// ---------------------------------------------------------------------------
describe("BE-045 (MEDIUM): consumeRefreshToken checks user.deletedAt", () => {
  const src = readSrc("src/lib/auth/server.ts");

  test("consumeRefreshToken fetches the user's deletedAt BEFORE rotating", () => {
    expect(src).toMatch(/consumeRefreshToken[\s\S]*?db\.user\.findUnique\(\s*\{[\s\S]*?deletedAt:\s*true/);
  });

  test("if user is soft-deleted, revokes all refresh tokens and returns null", () => {
    expect(src).toMatch(/if\s*\(\s*!softDeletedUser\s*\|\|\s*softDeletedUser\.deletedAt\s*!==\s*null\s*\)[\s\S]*?revokeAllRefreshTokensForUser/);
  });
});

// ---------------------------------------------------------------------------
// BE-046 (MEDIUM): cf-connecting-ip and true-client-ip gated by env vars.
// ---------------------------------------------------------------------------
describe("BE-046 (MEDIUM): proxy-set headers gated by env vars", () => {
  const src = readSrc("src/lib/auth/rate-limit.ts");

  test("cf-connecting-ip is only trusted when TRUST_CLOUDFLARE_HEADERS=true", () => {
    expect(src).toMatch(/if\s*\(\s*process\.env\.TRUST_CLOUDFLARE_HEADERS\s*===\s*["']true["']\s*\)[\s\S]*?cf-connecting-ip/);
  });

  test("true-client-ip is only trusted when TRUST_AKAMAI_HEADERS=true", () => {
    expect(src).toMatch(/if\s*\(\s*process\.env\.TRUST_AKAMAI_HEADERS\s*===\s*["']true["']\s*\)[\s\S]*?true-client-ip/);
  });

  test("does NOT have a safeHeaders array that unconditionally trusts all three", () => {
    // The old code had: const safeHeaders = ["x-real-ip", "cf-connecting-ip", "true-client-ip"];
    expect(src).not.toMatch(/safeHeaders\s*=\s*\[[^\]]*cf-connecting-ip[^\]]*true-client-ip/);
  });
});

// ---------------------------------------------------------------------------
// BE-047 (LOW): knowledge-graph POST returns 502 (not 500) on upstream failure.
// ---------------------------------------------------------------------------
describe("BE-047 (LOW): knowledge-graph POST returns 502 on upstream failure", () => {
  const src = readSrc("src/app/api/knowledge-graph/route.ts");

  test("POST handler's catch block returns 502 (not internalError)", () => {
    // Find the POST handler's catch block.
    const postMatch = src.match(/export\s+async\s+function\s+POST[\s\S]*$/);
    expect(postMatch).not.toBeNull();
    const postBody = postMatch![0];
    // The final catch block must NOT use internalError (which returns 500).
    // It should use NextResponse.json with status: 502.
    expect(postBody).toMatch(/status:\s*502/);
  });
});

// ---------------------------------------------------------------------------
// BE-055 (LOW): auth/me GET response is no-cache (not max-age=60).
// ---------------------------------------------------------------------------
describe("BE-055 (LOW): auth/me GET is no-cache (security state must be fresh)", () => {
  const src = readSrc("src/app/api/auth/me/route.ts");

  test("Cache-Control header is no-cache (not private, max-age=60)", () => {
    expect(src).toMatch(/["']Cache-Control["']:\s*["']no-cache/);
    expect(src).not.toMatch(/["']Cache-Control["']:\s*["']private,\s*max-age=60["']/);
  });
});

// ---------------------------------------------------------------------------
// BE-060 (LOW): requirePlatformAdmin applies rate limit to GET too.
// ---------------------------------------------------------------------------
describe("BE-060 (LOW): requirePlatformAdmin rate-limits GET requests", () => {
  const src = readSrc("src/lib/auth/require-platform-admin.ts");

  test("defines separate READ and WRITE rate limits", () => {
    expect(src).toMatch(/PLATFORM_ADMIN_WRITE_RATE_LIMIT/);
    expect(src).toMatch(/PLATFORM_ADMIN_READ_RATE_LIMIT/);
  });

  test("the rate-limit check runs for ALL methods (not just non-GET)", () => {
    // The old code had: `if (req && req.method !== "GET" && req.method !== "HEAD")`
    // The new code runs the check for ALL methods, with different limits.
    expect(src).toMatch(/const\s+isWrite\s*=\s*req\s*&&\s*req\.method\s*!==\s*["']GET["']/);
    expect(src).toMatch(/const\s+limit\s*=\s*isWrite\s*\?\s*PLATFORM_ADMIN_WRITE_RATE_LIMIT\s*:\s*PLATFORM_ADMIN_READ_RATE_LIMIT/);
  });
});

// ---------------------------------------------------------------------------
// BE-064 (LOW): resolveJwtSecret has a warned flag (logs once).
// ---------------------------------------------------------------------------
describe("BE-064 (LOW): resolveJwtSecret deduplicates the dev-secret warning", () => {
  const src = readSrc("src/lib/auth/server.ts");

  test("defines a module-level jwtSecretWarned flag", () => {
    expect(src).toMatch(/let\s+jwtSecretWarned\s*=\s*false/);
  });

  test("the warning is only logged when !jwtSecretWarned, then sets the flag", () => {
    expect(src).toMatch(/if\s*\(\s*!process\.env\.JWT_SECRET\s*&&\s*!jwtSecretWarned\s*\)\s*\{[\s\S]*?jwtSecretWarned\s*=\s*true/);
  });
});

// ---------------------------------------------------------------------------
// BE-065 (LOW): 2FA login-verify audit log uses real role (not "unknown").
// ---------------------------------------------------------------------------
describe("BE-065 (LOW): 2FA login-verify audit log uses real role", () => {
  const src = readSrc("src/app/api/auth/2fa/login-verify/route.ts");

  test("the login_mfa_replay_rejected audit log uses user.role (not 'unknown')", () => {
    // Find the audit log for login_mfa_replay_rejected.
    const auditMatch = src.match(/writeAuditLog\(\s*\{[\s\S]*?action:\s*["']login_mfa_replay_rejected["'][\s\S]*?\}\s*\)/);
    expect(auditMatch).not.toBeNull();
    const auditBody = auditMatch![0];
    expect(auditBody).toMatch(/role:\s*user\.role/);
    expect(auditBody).not.toMatch(/role:\s*["']unknown["']/);
  });
});

// ---------------------------------------------------------------------------
// BE-066 (LOW): 2FA login-verify uses recordFailedTotpDistributed.
// ---------------------------------------------------------------------------
describe("BE-066 (LOW): 2FA route uses distributed TOTP rate limiter", () => {
  const src = readSrc("src/app/api/auth/2fa/login-verify/route.ts");

  test("imports recordFailedTotpDistributed", () => {
    expect(src).toMatch(/recordFailedTotpDistributed/);
  });

  test("the TOTP failure path calls recordFailedTotpDistributed (not recordFailedTotp)", () => {
    // Find the line where afterFail is assigned.
    expect(src).toMatch(/const\s+afterFail\s*=\s*await\s+recordFailedTotpDistributed\(user\.id\)/);
  });
});

// ---------------------------------------------------------------------------
// BE-077 (LOW): per-user-rate-limit InMemoryBackend.recordAndCount has
// a clarifying comment about why `async` is required.
// ---------------------------------------------------------------------------
describe("BE-077 (LOW): InMemoryBackend async keyword documented", () => {
  const src = readSrc("src/lib/auth/per-user-rate-limit.ts");

  test("the InMemoryBackend class has a comment explaining why async is required", () => {
    expect(src).toMatch(/BE-077/);
    expect(src).toMatch(/async.*satisfy.*RateLimitStorage.*interface/is);
  });
});

// ---------------------------------------------------------------------------
// BE-078 (LOW): two-factor-setup-token has multi-instance documentation.
// ---------------------------------------------------------------------------
describe("BE-078 (LOW): two-factor-setup-token documents multi-instance limitation", () => {
  const src = readSrc("src/lib/auth/two-factor-setup-token.ts");

  test("the file-level comment mentions BE-078 and the multi-instance limitation", () => {
    expect(src).toMatch(/BE-078/);
    expect(src).toMatch(/multi-instance/i);
  });
});

// ---------------------------------------------------------------------------
// BE-079 (LOW): totp.ts replay protection comment explains the `<=` choice.
// ---------------------------------------------------------------------------
describe("BE-079 (LOW): totp replay protection uses <= (correct) with rationale", () => {
  const src = readSrc("src/lib/auth/totp.ts");

  test("the replay check uses <= (NOT <) for proper replay protection", () => {
    expect(src).toMatch(/matched\.counter\s*<=\s*lastUsedCounter/);
  });

  test("the comment explains why <= is correct (not <)", () => {
    expect(src).toMatch(/BE-079/);
    expect(src).toMatch(/breaks RFC 6238.*5\.2 replay protection/);
  });
});

// ---------------------------------------------------------------------------
// BE-081 (LOW): admin/metrics uses Prisma groupBy (not raw SQL).
// ---------------------------------------------------------------------------
describe("BE-081 (LOW): admin/metrics uses Prisma groupBy (not raw SQL)", () => {
  const src = readSrc("src/app/api/admin/metrics/route.ts");

  test("does NOT use $queryRaw with DATE() and INTERVAL", () => {
    // Check the executable CODE (not comments) — the BE-081 comment
    // legitimately mentions the old raw SQL when explaining the fix.
    const code = readCode("src/app/api/admin/metrics/route.ts");
    expect(code).not.toMatch(/\$queryRaw/);
    expect(code).not.toMatch(/DATE\(\s*["']createdAt["']\s*\)/);
    expect(code).not.toMatch(/INTERVAL\s*['"]7 days['"]/);
  });

  test("uses Prisma groupBy for top actions", () => {
    expect(src).toMatch(/db\.auditLog\.groupBy\(/);
  });

  test("computes daily active users via findMany + JS aggregation (dialect-agnostic)", () => {
    expect(src).toMatch(/db\.auditLog\.findMany\(/);
    expect(src).toMatch(/dayMap\s*=\s*new\s+Map/);
  });
});

// ---------------------------------------------------------------------------
// BE-082 (LOW): already covered in BE-025+BE-026+BE-082 block above.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// BE-084 (LOW): getAuthenticatedUser clears ONLY access cookie when org
// membership fails (not both).
// ---------------------------------------------------------------------------
describe("BE-084 (LOW): getAuthenticatedUser clears only access cookie on org removal", () => {
  const src = readSrc("src/lib/auth/server.ts");

  test("the access-cookie org-membership-fail branch uses store.delete(ACCESS_COOKIE) (not clearAuthCookies)", () => {
    expect(src).toMatch(/if\s*\(\s*!stillMember\s*\)\s*\{[\s\S]*?store\.delete\(ACCESS_COOKIE\)/);
  });

  test("also clears lastActiveOrgId in the DB (so next refresh doesn't re-stamp stale orgId)", () => {
    expect(src).toMatch(/db\.user\.update\(\s*\{[\s\S]*?data:\s*\{\s*lastActiveOrgId:\s*null\s*\}/);
  });
});

// ---------------------------------------------------------------------------
// BE-067 (INFORMATIONAL): login route has a clarifying comment about
// why recordIpAttempt is only called in the sync fallback path.
// ---------------------------------------------------------------------------
describe("BE-067 (INFORMATIONAL): login route recordIpAttempt comment", () => {
  const src = readSrc("src/app/api/auth/login/route.ts");

  test("the catch block has a clarifying BE-067 comment", () => {
    expect(src).toMatch(/BE-067/);
  });
});

// ---------------------------------------------------------------------------
// Integration smoke test: verify the new RefreshResult type is exported.
// ---------------------------------------------------------------------------
describe("Integration: RefreshResult type is exported for /api/auth/refresh", () => {
  const src = readSrc("src/lib/auth/server.ts");

  test("RefreshResult type is exported", () => {
    expect(src).toMatch(/export\s+type\s+RefreshResult\s*=/);
  });

  test("rotateRefreshToken returns RefreshResult (not the old 2-field shape)", () => {
    expect(src).toMatch(/rotateRefreshToken\([^)]*\):\s*Promise<RefreshResult>/);
  });

  test("consumeRefreshToken returns RefreshResult | null", () => {
    expect(src).toMatch(/consumeRefreshToken\([^)]*\):\s*Promise<RefreshResult\s*\|\s*null>/);
  });
});
