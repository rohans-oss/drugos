/**
 * FE-010 through FE-023 root-fix verification tests.
 * Team Member 13 — Frontend - API Routes (Critical Integration).
 */

import { readFileSync, existsSync } from "fs";
import { join } from "path";

const FRONTEND_ROOT = join(process.cwd(), "src");
const FRONTEND_PRISMA = join(process.cwd(), "prisma");

function readSrc(relPath: string): string {
  const full = join(FRONTEND_ROOT, relPath);
  if (!existsSync(full)) throw new Error(`file not found: ${full}`);
  return readFileSync(full, "utf8");
}
function readPrisma(relPath: string): string {
  const full = join(FRONTEND_PRISMA, relPath);
  if (!existsSync(full)) throw new Error(`file not found: ${full}`);
  return readFileSync(full, "utf8");
}

// FE-010: RL predictions must NOT be persisted as status="validated"
describe("FE-010: RL predictions persisted as 'predicted'", () => {
  const rlRoute = readSrc("app/api/rl/route.ts");
  const schema = readPrisma("schema.prisma");
  const rlRanker = readSrc("lib/services/rl-ranker.ts");

  test("rl/route.ts does NOT use status 'validated' for new RL hypotheses", () => {
    expect(rlRoute).toMatch(/status:\s*["']predicted["']/);
    // BE-028 ROOT FIX (Team Member 12): persistRlCandidates now uses
    // `tx.hypothesis.upsert(...)` inside a `db.$transaction` instead of
    // a standalone `db.hypothesis.create(...)`. The `create` branch of
    // the upsert still uses `status: "predicted"` — we verify that here.
    // Look for the upsert's `create:` block (which contains the status
    // assignment for NEW hypotheses).
    const upsertBlock = rlRoute.match(/tx\.hypothesis\.upsert\(\{[\s\S]*?\}\s*\)/);
    expect(upsertBlock).toBeTruthy();
    // The create branch must set status: "predicted" (NOT "validated").
    const createBranch = upsertBlock![0].match(/create:\s*\{[\s\S]*?(?:update:|}\s*\))/);
    expect(createBranch).toBeTruthy();
    expect(createBranch![0]).toMatch(/status:\s*["']predicted["']/);
    expect(createBranch![0]).not.toMatch(/status:\s*["']validated["']/);
  });
  test("rl/route.ts sets rlPredicted: true on persist", () => {
    expect(rlRoute).toMatch(/rlPredicted:\s*true/);
  });
  test("schema.prisma Hypothesis has rlPredicted Boolean field", () => {
    const hypBlock = schema.match(/model Hypothesis \{[\s\S]*?\}/);
    expect(hypBlock).toBeTruthy();
    expect(hypBlock![0]).toMatch(/rlPredicted\s+Boolean\s+@default\(false\)/);
    expect(hypBlock![0]).toMatch(/@@index\(\[rlPredicted\]\)/);
  });
  test("schema.prisma Hypothesis.status comment includes 'predicted'", () => {
    const hypBlock = schema.match(/model Hypothesis \{[\s\S]*?\}/);
    expect(hypBlock).toBeTruthy();
    expect(hypBlock![0]).toMatch(/draft \| predicted \| reviewing \| validated/);
  });
  test("rl-ranker syncRlOutputToHypotheses preserves validated/rejected", () => {
    expect(rlRanker).toMatch(/h\.status === "draft" \? "predicted" : h\.status/);
  });
  test("rl/route.ts persistRlCandidates does NOT overwrite validated/rejected", () => {
    expect(rlRoute).toMatch(/existing\.status === "validated" \|\| existing\.status === "rejected"/);
  });
});

// FE-011: Real CSRF protection
describe("FE-011: Real CSRF protection (double-submit cookie)", () => {
  const apiHelpers = readSrc("lib/api-helpers.ts");
  test("requireCsrfOrSend is NOT a passthrough NO-OP", () => {
    expect(apiHelpers).not.toMatch(/\/\/\s*TODO: implement full CSRF/i);
    expect(apiHelpers).toMatch(/timingSafeEqual/);
    expect(apiHelpers).toMatch(/csrf_missing|csrf_mismatch/);
  });
  test("requireCsrfOrSend accepts a NextRequest parameter", () => {
    expect(apiHelpers).toMatch(/export async function requireCsrfOrSend\(req: NextRequest\)/);
  });
  test("CSRF uses double-submit cookie pattern", () => {
    expect(apiHelpers).toMatch(/CSRF_COOKIE_NAME/);
    expect(apiHelpers).toMatch(/issueCsrfToken/);
    expect(apiHelpers).toMatch(/setCsrfCookie/);
  });
  test("CSRF exempts API-key auth (Bearer drugos_)", () => {
    expect(apiHelpers).toMatch(/Bearer/);
    expect(apiHelpers).toMatch(/drugos_/);
  });
  test("CSRF exempts unauthenticated requests (no session cookies)", () => {
    expect(apiHelpers).toMatch(/hasAccessCookie/);
    expect(apiHelpers).toMatch(/hasRefreshCookie/);
  });
  test("CSRF is applied to state-changing routes", () => {
    const routes = [
      "app/api/auth/logout/route.ts",
      "app/api/auth/password/route.ts",
      "app/api/auth/2fa/setup/route.ts",
      "app/api/auth/2fa/verify/route.ts",
      "app/api/auth/2fa/disable/route.ts",
      "app/api/auth/register/route.ts",
      "app/api/projects/route.ts",
      "app/api/projects/[id]/route.ts",
      "app/api/projects/[id]/comments/route.ts",
      "app/api/billing/subscription/route.ts",
      "app/api/evidence-package/route.ts",
      "app/api/rl/route.ts",
      "app/api/api-keys/route.ts",
      "app/api/api-keys/[id]/revoke/route.ts",
      "app/api/admin/users/route.ts",
      "app/api/notifications/[id]/read/route.ts",
    ];
    for (const r of routes) expect(readSrc(r)).toMatch(/requireCsrfOrSend/);
  });
  test("login route sets the CSRF cookie on success", () => {
    const login = readSrc("app/api/auth/login/route.ts");
    expect(login).toMatch(/issueCsrfToken/);
    expect(login).toMatch(/setCsrfCookie/);
  });
});

// FE-012: API key auth path
describe("FE-012: API key auth path in getAuthenticatedUser", () => {
  const server = readSrc("lib/auth/server.ts");
  test("getAuthenticatedUser checks Authorization: Bearer drugos_ header", () => {
    expect(server).toMatch(/authorization/i);
    expect(server).toMatch(/drugos_/);
  });
  test("getAuthenticatedUser calls authenticateApiKey when Bearer drugos_ is present", () => {
    expect(server).toMatch(/authenticateApiKey\(rawKey\)/);
  });
});

// FE-013: Admin IDOR guard
describe("FE-013: PATCH /api/admin/users cross-tenant IDOR guard", () => {
  const admin = readSrc("app/api/admin/users/route.ts");
  test("PATCH handler checks target user's org membership", () => {
    expect(admin).toMatch(/adminMemberships/);
    expect(admin).toMatch(/targetMemberships/);
    expect(admin).toMatch(/organizationId: \{\s*in:\s*adminOrgIds\s*\}/);
  });
  test("returns 404 when target user is in a different org", () => {
    expect(admin).toMatch(/admin_user_update_denied_cross_tenant/);
  });
  test("owner role bypasses the cross-tenant check", () => {
    expect(admin).toMatch(/auth\.user\.role !== "owner"/);
  });
});

// FE-014: api-keys userId filter
describe("FE-014: GET /api/api-keys userId filter", () => {
  const apiKeysRoute = readSrc("app/api/api-keys/route.ts");
  test("GET handler computes ownerFilter based on role", () => {
    expect(apiKeysRoute).toMatch(/ownerFilter/);
    expect(apiKeysRoute).toMatch(/auth\.user\.role === "admin" \|\| auth\.user\.role === "owner"/);
  });
  test("non-admins pass auth.user.userId as the filter", () => {
    expect(apiKeysRoute).toMatch(/:\s*auth\.user\.userId/);
  });
  test("listApiKeys is called with the ownerFilter", () => {
    expect(apiKeysRoute).toMatch(/listApiKeys\(auth\.user\.orgId,\s*ownerFilter\)/);
  });
});

// FE-015: No double "drugos_" prefix
describe("FE-015: API key display does not double-prefix", () => {
  const remaining = readSrc("components/drugos/remaining-screens.tsx");
  test("UI does NOT prepend drugos_ to k.prefix redundantly", () => {
    // FE-038 changed prefix to be 8 hex chars AFTER drugos_, so the UI
    // correctly renders "drugos_{k.prefix}…" where k.prefix is just hex.
    // The bug (FE-015) was when prefix INCLUDED "drugos_" and the UI
    // prepended another "drugos_" — producing "drugos_drugos_…".
    // With FE-038's fix, prefix is 8 hex chars, so "drugos_{prefix}" is correct.
    expect(remaining).toMatch(/drugos_\{k\.prefix\}/);
    // Verify the lib stores prefix as slice(7, 15) — 8 hex chars after drugos_
    const apiKeysLib = readSrc("lib/services/api-keys.ts");
    expect(apiKeysLib).toMatch(/rawKey\.slice\(7,\s*15\)/);
  });
});

// FE-016: MFA challenge token replay protection
describe("FE-016: MFA challenge token replay protection", () => {
  const server = readSrc("lib/auth/server.ts");
  const verify = readSrc("app/api/auth/2fa/login-verify/route.ts");
  const login = readSrc("app/api/auth/login/route.ts");
  const schema = readPrisma("schema.prisma");

  test("signMfaChallengeToken includes a jti", () => {
    expect(server).toMatch(/jti/);
    expect(server).toMatch(/randomBytes\(16\)\.toString\("hex"\)/);
  });
  test("MfaChallenge table exists in schema with jti @unique", () => {
    const mfaBlock = schema.match(/model MfaChallenge \{[\s\S]*?\}/);
    expect(mfaBlock).toBeTruthy();
    expect(mfaBlock![0]).toMatch(/jti\s+String\s+@unique/);
  });
  test("login-verify route rejects replayed jti (P2002)", () => {
    expect(verify).toMatch(/P2002/);
    expect(verify).toMatch(/login_mfa_replay_rejected/);
  });
  test("login route sets the mfa challenge token as an HttpOnly cookie", () => {
    expect(login).toMatch(/drugos_mfa_challenge/);
    expect(login).toMatch(/httpOnly:\s*true/);
    expect(login).toMatch(/sameSite:\s*"strict"/);
  });
  test("login-verify reads mfaToken from cookie first, falls back to body", () => {
    expect(verify).toMatch(/drugos_mfa_challenge/);
    expect(verify).toMatch(/body\.mfaToken/);
  });
});

// FE-017: Project endpoints enforce visibility + role
describe("FE-017: Project endpoints enforce visibility + role", () => {
  const projectRoute = readSrc("app/api/projects/[id]/route.ts");
  test("GET /api/projects/[id] checks project.visibility", () => {
    expect(projectRoute).toMatch(/project\.visibility === "private"/);
    expect(projectRoute).toMatch(/project\.visibility === "public"/);
  });
  test("POST /api/projects/[id] checks OrganizationMember.role", () => {
    expect(projectRoute).toMatch(/PROJECT_WRITE_ROLES/);
    expect(projectRoute).toMatch(/organizationMember\.findFirst/);
  });
  test("comments route ignores client-supplied authorName (FE-073)", () => {
    const commentsRoute = readSrc("app/api/projects/[id]/comments/route.ts");
    // FE-073 already fixed this — authorName is ignored, userId is passed to addComment
    expect(commentsRoute).toMatch(/addComment\(id,\s*auth\.user\.userId,\s*commentBody\)/);
    expect(commentsRoute).toMatch(/authorName.*intentionally ignored/i);
  });
});

// FE-018: recordSuccessfulLogin moved AFTER MFA verification
describe("FE-018: recordSuccessfulLogin not called before MFA", () => {
  const login = readSrc("app/api/auth/login/route.ts");
  const verify = readSrc("app/api/auth/2fa/login-verify/route.ts");

  test("login route does NOT call recordSuccessfulLogin in the mfaEnabled branch", () => {
    // Extract the MFA branch — from `if (user.mfaEnabled) {` to the
    // `mfaRequired: true` return. Verify recordSuccessfulLogin is NOT
    // CALLED (with parentheses) in that block. The word may appear in
    // comments explaining WHY it's not called — that's fine.
    const mfaStart = login.indexOf("if (user.mfaEnabled) {");
    expect(mfaStart).toBeGreaterThan(-1);
    const mfaEnd = login.indexOf("mfaRequired: true", mfaStart);
    expect(mfaEnd).toBeGreaterThan(mfaStart);
    const mfaBranch = login.slice(mfaStart, mfaEnd);
    // Match actual function CALL: `await recordSuccessfulLogin(` — the
    // `await` prefix distinguishes a real call from a comment that happens
    // to mention the function name with parens.
    expect(mfaBranch).not.toMatch(/await\s+recordSuccessfulLogin\s*\(/);
  });
  test("login-verify route calls recordSuccessfulLogin AFTER TOTP verifies", () => {
    const verifyTotpIdx = verify.indexOf("verifyTotpWithReplayCheck");
    const recordSuccessIdx = verify.indexOf("recordSuccessfulLogin(user.id)", verifyTotpIdx);
    expect(verifyTotpIdx).toBeGreaterThan(-1);
    expect(recordSuccessIdx).toBeGreaterThan(verifyTotpIdx);
  });
  test("login-verify route increments failedLoginCount on TOTP failure", () => {
    expect(verify).toMatch(/recordFailedLogin\(user\.id\)/);
  });
});

// FE-019: Single RL CSV parser
describe("FE-019: Consolidated RL CSV parser", () => {
  const rlRoute = readSrc("app/api/rl/route.ts");
  const rlRanker = readSrc("lib/services/rl-ranker.ts");

  test("rl/route.ts imports getRankedHypotheses from the lib", () => {
    expect(rlRoute).toMatch(/from ["']@\/lib\/services\/rl-ranker["']/);
    expect(rlRoute).toMatch(/getRankedHypotheses/);
  });
  test("rl/route.ts does NOT define its own parseRlCsv function", () => {
    expect(rlRoute).not.toMatch(/async function parseRlCsv/);
    expect(rlRoute).not.toMatch(/interface RlCandidate/);
  });
  test("rl-ranker.ts uses csv-parse/sync", () => {
    expect(rlRanker).toMatch(/from ["']csv-parse\/sync["']/);
  });
  test("rl-ranker.ts honors RL_OUTPUT_CSV_PATH as canonical env var", () => {
    expect(rlRanker).toMatch(/RL_OUTPUT_CSV_PATH/);
  });
  test("rl-ranker.ts honors RL_LOCAL_CSV as legacy alias", () => {
    expect(rlRanker).toMatch(/RL_LOCAL_CSV/);
  });
  test("RankedHypothesis includes full schema (overallScore, marketScore, etc.)", () => {
    expect(rlRanker).toMatch(/overallScore\?:\s*number/);
    expect(rlRanker).toMatch(/marketScore\?:\s*number/);
    expect(rlRanker).toMatch(/plausibilityScore\?:\s*number/);
  });
});

// FE-020: IPv6 support
describe("FE-020: IPv6 support in rate-limit getClientIp", () => {
  const rateLimit = readSrc("lib/auth/rate-limit.ts");
  test("uses a permissive IP regex that matches both IPv4 and IPv6", () => {
    expect(rateLimit).toMatch(/IPV4_OR_V6_RE/);
    expect(rateLimit).toMatch(/\[0-9a-fA-F\]/);
  });
  test("honors cf-connecting-ip and true-client-ip", () => {
    expect(rateLimit).toMatch(/cf-connecting-ip/);
    expect(rateLimit).toMatch(/true-client-ip/);
  });
  test("isValidIp rejects strings longer than 45 chars", () => {
    expect(rateLimit).toMatch(/s\.length > 45/);
  });
});

// FE-021: getAuthenticatedUser clears invalid cookies
describe("FE-021: getAuthenticatedUser clears invalid cookies", () => {
  const server = readSrc("lib/auth/server.ts");
  test("calls clearAuthCookies when both access and refresh fail", () => {
    expect(server).toMatch(/clearAuthCookies\(\)/);
  });
  test("the clear is conditional on (access || refresh) being present", () => {
    expect(server).toMatch(/if \(access \|\| refresh\)/);
  });
});

// FE-022: Dead requireRole removed
describe("FE-022: Dead boolean requireRole removed from server.ts", () => {
  const server = readSrc("lib/auth/server.ts");
  test("server.ts does NOT export the boolean requireRole", () => {
    expect(server).not.toMatch(/export function requireRole\([\s\S]*?\):\s*boolean/);
  });
  test("server.ts re-exports requireRole from @/lib/api-helpers", () => {
    expect(server).toMatch(/export \{[\s\S]*?requireRole[\s\S]*?\} from ["']@\/lib\/api-helpers["']/);
  });
});

// FE-023: Safety tier thresholds removed
describe("FE-023: Safety tier thresholds removed; disclaimer added", () => {
  const coreScreens = readSrc("components/drugos/core-screens.tsx");
  const safetyBadge = readSrc("components/drugos/safety-badge.tsx");
  const types = readSrc("lib/types.ts");

  test("core-screens does NOT compute safetyTier from safetyScore thresholds", () => {
    expect(coreScreens).not.toMatch(/safetyScore.*>=\s*0\.7\s*\?\s*['"]green['"]/);
  });
  test("RL candidates are mapped to safetyTier: 'unknown'", () => {
    expect(coreScreens).toMatch(/safetyTier:\s*['"]unknown['"] as const/);
  });
  test("SafetyTier type includes 'unknown'", () => {
    expect(types).toMatch(/'green' \| 'yellow' \| 'red' \| 'unknown'/);
  });
  test("SafetyBadge component renders 'unknown' tier", () => {
    expect(safetyBadge).toMatch(/unknown:/);
    expect(safetyBadge).toMatch(/Model score only/);
  });
  test("core-screens includes a patient-safety disclaimer banner", () => {
    expect(coreScreens).toMatch(/Patient-safety disclaimer/);
  });
});

// BEHAVIOR TESTS
describe("Behavior: computeOverallScore weights (FE-019)", () => {
  test("returns null when no signals are present", async () => {
    const { computeOverallScore } = await import("@/lib/services/rl-ranker");
    expect(computeOverallScore({})).toBeNull();
  });
  test("weighted blend: 0.4*gnn + 0.3*safety + 0.3*market", async () => {
    const { computeOverallScore } = await import("@/lib/services/rl-ranker");
    const result = computeOverallScore({ gnnScore: 1.0, safetyScore: 0.5, marketScore: 0.5 });
    expect(result).toBeCloseTo(0.7, 5);
  });
  test("falls back to policyProb when no per-dimension scores", async () => {
    const { computeOverallScore } = await import("@/lib/services/rl-ranker");
    expect(computeOverallScore({ policyProb: 0.42 })).toBeCloseTo(0.42, 5);
  });
});

describe("Behavior: CSRF token issuance (FE-011)", () => {
  test("issueCsrfToken returns a 64-char hex string", async () => {
    const { issueCsrfToken } = await import("@/lib/api-helpers");
    const token = issueCsrfToken();
    expect(token).toMatch(/^[0-9a-f]{64}$/);
  });
  test("issueCsrfToken is non-deterministic", async () => {
    const { issueCsrfToken } = await import("@/lib/api-helpers");
    expect(issueCsrfToken()).not.toBe(issueCsrfToken());
  });
});

describe("Behavior: MFA challenge token has jti (FE-016)", () => {
  test("signMfaChallengeToken returns a JWT with a jti claim", async () => {
    const { signMfaChallengeToken, verifyMfaChallengeToken } = await import("@/lib/auth/server");
    const token = signMfaChallengeToken({ userId: "test-user", email: "test@example.com" });
    expect(token).toBeTruthy();
    const jwt = await import("jsonwebtoken");
    const decoded = jwt.decode(token, { complete: true }) as { payload?: { jti?: string } } | null;
    expect(decoded?.payload?.jti).toBeTruthy();
    expect(decoded!.payload!.jti).toMatch(/^[0-9a-f]{32}$/);
    const verified = verifyMfaChallengeToken(token);
    expect(verified).toBeTruthy();
    expect(verified?.userId).toBe("test-user");
  });
  test("two successive tokens have different jtis", async () => {
    const { signMfaChallengeToken } = await import("@/lib/auth/server");
    const jwt = await import("jsonwebtoken");
    const t1 = signMfaChallengeToken({ userId: "u1", email: "a@b.com" });
    const t2 = signMfaChallengeToken({ userId: "u1", email: "a@b.com" });
    const d1 = jwt.decode(t1, { complete: true }) as { payload?: { jti?: string } };
    const d2 = jwt.decode(t2, { complete: true }) as { payload?: { jti?: string } };
    expect(d1.payload!.jti).not.toBe(d2.payload!.jti);
  });
});

describe("Behavior: IPv6 validation (FE-020)", () => {
  function makeReq(headers: Record<string, string>): any {
    const h = new Headers();
    for (const [k, v] of Object.entries(headers)) h.set(k, v);
    return { headers: h } as any;
  }
  test("IPv4 address is recognized", async () => {
    const { checkIpRateLimit } = await import("@/lib/auth/rate-limit");
    expect(checkIpRateLimit(makeReq({ "x-real-ip": "203.0.113.1" })).blocked).toBe(false);
  });
  test("IPv6 address is recognized (FE-020 fix)", async () => {
    const { checkIpRateLimit } = await import("@/lib/auth/rate-limit");
    expect(checkIpRateLimit(makeReq({ "x-real-ip": "2001:db8::1" })).blocked).toBe(false);
  });
  test("IPv6 loopback is recognized", async () => {
    const { checkIpRateLimit } = await import("@/lib/auth/rate-limit");
    expect(checkIpRateLimit(makeReq({ "x-real-ip": "::1" })).blocked).toBe(false);
  });
  test("cf-connecting-ip header is honored (Cloudflare)", async () => {
    const { checkIpRateLimit } = await import("@/lib/auth/rate-limit");
    expect(checkIpRateLimit(makeReq({ "cf-connecting-ip": "2001:db8::abcd" })).blocked).toBe(false);
  });
});
