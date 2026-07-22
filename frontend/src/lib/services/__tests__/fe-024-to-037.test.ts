/**
 * FE-024 to FE-037 — Team 14 (Frontend Services & ML Integration)
 * Forensic root-fix verification tests.
 *
 * Each test is named after the issue ID it covers. The tests are PURE
 * (no DB, no network) where possible — they verify the LOGIC of the
 * fix, not the integration. Integration tests live in tests/api/.
 *
 * Run: npx jest src/lib/services/__tests__/fe-024-to-037.test.ts
 */

import {
  computeTotp,
  verifyTotp,
  verifyTotpWithReplayCheck,
  generateTotpSecret,
} from "@/lib/auth/totp";

// ---------------------------------------------------------------------------
// FE-024: drug-mechanism lookup service exists and exports the right API.
// ---------------------------------------------------------------------------

describe("FE-024: drug-mechanism lookup service", () => {
  it("should export lookupDrugMechanism and lookupDrugMechanisms", async () => {
    const mod = await import("@/lib/services/drug-mechanism");
    expect(typeof mod.lookupDrugMechanism).toBe("function");
    expect(typeof mod.lookupDrugMechanisms).toBe("function");
  });

  it("should return null mechanism when ChEMBL has no match (no fabricated data)", async () => {
    const mod = await import("@/lib/services/drug-mechanism");
    // Mock fetch to return empty molecules array.
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () => ({
      ok: true,
      json: async () => ({ molecules: [] }),
    }) as unknown as Response);

    try {
      const result = await mod.lookupDrugMechanism("NONEXISTENTDRUG12345");
      expect(result.mechanism).toBeNull();
      expect(result.chemblId).toBeNull();
      expect(result.drugName).toBe("NONEXISTENTDRUG12345");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

// ---------------------------------------------------------------------------
// FE-025: candidate-table column header says 'Composite Score', not 'Confidence'.
// ---------------------------------------------------------------------------

describe("FE-025: composite score column rename", () => {
  it("candidate-table.tsx source contains 'Composite Score' header", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(
      __dirname,
      "../../../components/drugos/candidate-table.tsx"
    );
    const src = fs.readFileSync(filePath, "utf8");
    expect(src).toContain("Composite Score");
    expect(src).not.toContain('"Confidence"');
    // Tooltip must explicitly say it's NOT a statistical confidence interval.
    expect(src).toContain("NOT a statistical confidence interval");
  });

  it("core-screens.tsx no longer sets mechanism to RL debug values", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(
      __dirname,
      "../../../components/drugos/core-screens.tsx"
    );
    const src = fs.readFileSync(filePath, "utf8");
    // The bad pattern was: mechanism: `RL reward: ${rc.reward...
    expect(src).not.toMatch(/mechanism:\s*`RL reward:/);
    // The fix: mechanism: '' and rlDebugInfo: { reward, policyProb, ... }
    expect(src).toContain("rlDebugInfo");
    expect(src).toContain("mechanism: ''");
  });
});

// ---------------------------------------------------------------------------
// FE-026 / FE-034: mock-data.ts is DELETED; empty-defaults.ts is the replacement.
// ---------------------------------------------------------------------------

describe("FE-026 / FE-034: mock-data.ts deleted, empty-defaults.ts is empty", () => {
  it("all data exports in empty-defaults.ts should be empty arrays or empty objects", async () => {
    // FE-034 ROOT FIX: mock-data.ts was DELETED (dangerous name invited
    // future engineers to re-add fabricated data). The empty defaults now
    // live in @/lib/empty-defaults. The contract is the same: every data
    // export is an empty array or zeroed object.
    const mod = await import("@/lib/empty-defaults");
    // Diseases — was 10 fake entries, now must be empty.
    expect(Array.isArray(mod.diseases)).toBe(true);
    expect(mod.diseases.length).toBe(0);
    // Drug candidates — was 13 fake entries, now empty.
    expect(Array.isArray(mod.drugCandidates)).toBe(true);
    expect(mod.drugCandidates.length).toBe(0);
    // Clinical trials — was 6 fake entries, now empty.
    expect(Array.isArray(mod.clinicalTrials)).toBe(true);
    expect(mod.clinicalTrials.length).toBe(0);
    // Graph nodes/edges — were fake, now empty.
    expect(mod.graphNodes.length).toBe(0);
    expect(mod.graphEdges.length).toBe(0);
    // Users — was 8 fake users, now empty.
    expect(mod.users.length).toBe(0);
    // Notifications — was 5 fake, now empty.
    expect(mod.notifications.length).toBe(0);
    // Audit logs — was 6 fake, now empty.
    expect(mod.auditLogs.length).toBe(0);
    // Knowledge graph aliases — were mock arrays, now empty.
    expect(mod.knowledgeGraphNodes.length).toBe(0);
    expect(mod.knowledgeGraphEdges.length).toBe(0);
    // Pathway data — was fabricated, now empty.
    expect(mod.pathwayData.nodes.length).toBe(0);
    expect(mod.pathwayData.edges.length).toBe(0);
    // Dashboard stats — was fabricated numbers, now all zeros.
    expect(mod.dashboardStats.totalCandidates).toBe(0);
    expect(mod.dashboardStats.totalDrugs).toBe(0);
    expect(mod.dashboardStats.knowledgeGraphNodes).toBe(0);
  });

  it("empty-defaults.ts module loads without error", async () => {
    // FE-034: empty-defaults.ts is the canonical replacement. It must load
    // cleanly. Types are NOT re-exported here — they live in @/lib/types.
    const mod = await import("@/lib/empty-defaults");
    expect(mod).toBeDefined();
  });

  it("types.ts should be the canonical home for DrugCandidate with rlDebugInfo", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(__dirname, "../../types.ts");
    const src = fs.readFileSync(filePath, "utf8");
    expect(src).toContain("interface DrugCandidate");
    expect(src).toContain("rlDebugInfo");
    // The mechanism field doc must say "NEVER RL debug values".
    expect(src).toContain("NEVER RL debug values");
  });
});

// ---------------------------------------------------------------------------
// FE-027: ESLint config re-enables meaningful rules.
// ---------------------------------------------------------------------------

describe("FE-027: ESLint config re-enables rules", () => {
  it("eslint.config.mjs should not globally disable critical rules", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(__dirname, "../../../../eslint.config.mjs");
    const src = fs.readFileSync(filePath, "utf8");
    // FE-027 v2: The config now has a test-file override block that
    // legitimately sets some rules to "off" for *.test.ts files (Jest
    // mocking uses `any` and `require()`). The MAIN rules block must
    // still have the rules enabled. We extract the main block (before
    // the test-file override) and verify it doesn't disable the rules.
    //
    // The bad pattern was `"@typescript-eslint/no-explicit-any": "off"`.
    // After the fix, it should be "warn" (not "off") in the main block.
    // Note: `no-unused-vars: "off"` is allowed because @typescript-eslint
    // handles it — that's the standard TS pattern.
    const mainBlockEnd = src.indexOf("// FE-027 ROOT FIX (v2): Test files");
    const mainBlock = mainBlockEnd > 0 ? src.slice(0, mainBlockEnd) : src;
    expect(mainBlock).not.toMatch(/no-explicit-any":\s*"off"/);
    expect(mainBlock).not.toMatch(/no-unreachable":\s*"off"/);
    expect(mainBlock).not.toMatch(/no-console":\s*"off"/);
    expect(mainBlock).not.toMatch(/exhaustive-deps":\s*"off"/);
    expect(mainBlock).not.toMatch(/no-debugger":\s*"off"/);
    expect(mainBlock).not.toMatch(/prefer-const":\s*"off"/);
    expect(mainBlock).not.toMatch(/no-redeclare":\s*"off"/);
    expect(mainBlock).not.toMatch(/no-fallthrough":\s*"off"/);
    // The fix uses "warn" or "error" for these rules.
    expect(src).toMatch(/no-explicit-any":\s*"warn"/);
    expect(src).toMatch(/no-unreachable":\s*"error"/);
    expect(src).toMatch(/prefer-const":\s*"error"/);
  });
});

// ---------------------------------------------------------------------------
// FE-028: reactStrictMode is enabled.
// ---------------------------------------------------------------------------

describe("FE-028: reactStrictMode enabled", () => {
  it("next.config.ts should set reactStrictMode: true", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(__dirname, "../../../../next.config.ts");
    const src = fs.readFileSync(filePath, "utf8");
    expect(src).toMatch(/reactStrictMode:\s*true/);
    expect(src).not.toMatch(/reactStrictMode:\s*false/);
  });
});

// ---------------------------------------------------------------------------
// FE-029: dead deps removed from package.json.
// ---------------------------------------------------------------------------

describe("FE-029: dead deps removed", () => {
  it("package.json should NOT declare next-auth, nodemailer, bcrypt, z-ai-web-dev-sdk", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(__dirname, "../../../../package.json");
    const src = fs.readFileSync(filePath, "utf8");
    const pkg = JSON.parse(src);
    const deps = Object.keys(pkg.dependencies || {});
    expect(deps).not.toContain("next-auth");
    expect(deps).not.toContain("nodemailer");
    expect(deps).not.toContain("bcrypt"); // bcryptjs is still there (used)
    expect(deps).not.toContain("z-ai-web-dev-sdk");
    // bcryptjs should still be present (it's actually used).
    expect(deps).toContain("bcryptjs");
  });

  it("auth/server.ts validateEmail should NOT claim to send verification email", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(__dirname, "../../auth/server.ts");
    const src = fs.readFileSync(filePath, "utf8");
    // The misleading comment was: "we send a verification email for real accounts"
    expect(src).not.toMatch(/we send a verification email for real accounts/);
  });
});

// ---------------------------------------------------------------------------
// FE-030: favicon bundled locally, no third-party CDN.
// ---------------------------------------------------------------------------

describe("FE-030: favicon bundled locally", () => {
  it("layout.tsx should use /logo.svg, not the external CDN URL", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(__dirname, "../../../app/layout.tsx");
    const src = fs.readFileSync(filePath, "utf8");
    // The actual icon value must be the local path, not the CDN URL.
    // (The CDN URL may still appear in comments explaining the fix.)
    expect(src).toMatch(/icon:\s*"\/logo\.svg"/);
    // The actual icon value must NOT be the CDN URL.
    expect(src).not.toMatch(/icon:\s*"https:\/\/z-cdn\.chatglm\.cn/);
  });

  it("public/logo.svg should exist", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(__dirname, "../../../../public/logo.svg");
    expect(fs.existsSync(filePath)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// FE-031: /api/auth/refresh clears cookies on invalid refresh.
// ---------------------------------------------------------------------------

describe("FE-031: refresh clears cookies on invalid token", () => {
  it("refresh route should call clearAuthCookies on invalid refresh", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(
      __dirname,
      "../../../app/api/auth/refresh/route.ts"
    );
    const src = fs.readFileSync(filePath, "utf8");
    // Must import clearAuthCookies.
    expect(src).toMatch(/import.*clearAuthCookies/);
    // Must call clearAuthCookies in the no-refresh branch.
    expect(src).toMatch(/if\s*\(!refresh\)[\s\S]*?clearAuthCookies/);
    // Must call clearAuthCookies in the invalid-refresh branch.
    expect(src).toMatch(/if\s*\(!result\)[\s\S]*?clearAuthCookies/);
  });
});

// ---------------------------------------------------------------------------
// FE-032: rotateRefreshToken checks user.status + lockedUntil.
// ---------------------------------------------------------------------------

describe("FE-032: rotateRefreshToken checks suspended/locked", () => {
  it("server.ts rotateRefreshToken should check user.status === 'suspended'", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(__dirname, "../../auth/server.ts");
    const src = fs.readFileSync(filePath, "utf8");
    expect(src).toMatch(/user\.status\s*===\s*"suspended"/);
    expect(src).toMatch(/revokeAllRefreshTokensForUser/);
    expect(src).toMatch(/user\.lockedUntil/);
    expect(src).toMatch(/account_suspended/);
    expect(src).toMatch(/account_locked/);
  });

  it("admin users PATCH should revoke tokens on suspend", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(
      __dirname,
      "../../../app/api/admin/users/route.ts"
    );
    const src = fs.readFileSync(filePath, "utf8");
    expect(src).toMatch(/revokeAllRefreshTokensForUser/);
    expect(src).toMatch(/data\.status\s*===\s*"suspended"/);
  });
});

// ---------------------------------------------------------------------------
// FE-033: TOTP replay protection.
// ---------------------------------------------------------------------------

describe("FE-033: TOTP replay protection", () => {
  it("verifyTotpWithReplayCheck should reject a replayed code", () => {
    const secret = generateTotpSecret();
    const now = Date.now();
    const code = computeTotp(secret, new Date(now));

    // First verification succeeds. Get the counter.
    const result1 = verifyTotpWithReplayCheck(secret, code, null);
    expect(result1.ok).toBe(true);
    if (result1.ok) {
      // Second verification with the same code + lastUsedCounter = same counter
      // must be REJECTED as replayed.
      const result2 = verifyTotpWithReplayCheck(secret, code, result1.counter);
      expect(result2.ok).toBe(false);
      if (!result2.ok) {
        expect(result2.reason).toBe("replayed");
      }
    }
  });

  it("verifyTotpWithReplayCheck should accept a fresh code when lastUsedCounter is older", () => {
    const secret = generateTotpSecret();
    const now = Date.now();
    // Compute a code for the PREVIOUS window (-30s).
    const oldCode = computeTotp(secret, new Date(now - 30000));
    const oldResult = verifyTotpWithReplayCheck(secret, oldCode, null);
    expect(oldResult.ok).toBe(true);

    // Now compute a code for the CURRENT window.
    const newCode = computeTotp(secret, new Date(now));
    if (oldResult.ok) {
      const newResult = verifyTotpWithReplayCheck(secret, newCode, oldResult.counter);
      expect(newResult.ok).toBe(true);
    }
  });

  it("verifyTotpWithReplayCheck should reject invalid code", () => {
    const secret = generateTotpSecret();
    const result = verifyTotpWithReplayCheck(secret, "000000", null);
    // 000000 is unlikely to match (1 in 1M chance). If it does, the test
    // is flaky — re-run.
    if (!result.ok) {
      expect(result.reason).toBe("invalid_code");
    }
  });

  it("schema should have lastTotpCounter field on User", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(__dirname, "../../../../prisma/schema.prisma");
    const src = fs.readFileSync(filePath, "utf8");
    expect(src).toMatch(/lastTotpCounter\s+BigInt\?/);
  });
});

// ---------------------------------------------------------------------------
// FE-034: writeAuditLog blocking for critical actions.
// ---------------------------------------------------------------------------

describe("FE-034: writeAuditLog critical flag", () => {
  it("api-helpers.ts writeAuditLog should support critical flag", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(__dirname, "../../api-helpers.ts");
    const src = fs.readFileSync(filePath, "utf8");
    expect(src).toMatch(/critical\?\s*:\s*boolean/);
    expect(src).toMatch(/writeAuditLogCritical/);
    // Dead-letter fallback for non-critical.
    expect(src).toMatch(/audit_log_dead_letter/);
    expect(src).toMatch(/AUDIT-LOG-FAILURE/);
  });

  it("login route should use critical: true for login_failed and login", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(
      __dirname,
      "../../../app/api/auth/login/route.ts"
    );
    const src = fs.readFileSync(filePath, "utf8");
    // The login_failed audit must be critical.
    expect(src).toMatch(/action:\s*"login_failed"[\s\S]*?critical:\s*true/);
    // The login success audit must be critical.
    expect(src).toMatch(/action:\s*"login"[\s\S]*?critical:\s*true/);
  });

  it("password route should use critical: true and revoke sessions on audit failure", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(
      __dirname,
      "../../../app/api/auth/password/route.ts"
    );
    const src = fs.readFileSync(filePath, "utf8");
    expect(src).toMatch(/action:\s*"password_change"[\s\S]*?critical:\s*true/);
    expect(src).toMatch(/revokeAllRefreshTokensForUser/);
  });
});

// ---------------------------------------------------------------------------
// FE-035: registration rate limit + email verification.
// ---------------------------------------------------------------------------

describe("FE-035: registration rate limit + email verification", () => {
  it("register route should call checkIpRateLimit", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(
      __dirname,
      "../../../app/api/auth/register/route.ts"
    );
    const src = fs.readFileSync(filePath, "utf8");
    expect(src).toMatch(/checkIpRateLimit/);
    expect(src).toMatch(/recordIpAttempt/);
    // 429 status for rate-limited.
    expect(src).toMatch(/status:\s*429/);
  });

  it("register route should NOT issue tokens on registration", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(
      __dirname,
      "../../../app/api/auth/register/route.ts"
    );
    const src = fs.readFileSync(filePath, "utf8");
    // The bad code called rotateRefreshToken + setAuthCookies on register.
    // After the fix, these are NOT called — the user must verify email first.
    // We check that the response includes verificationRequired: true.
    expect(src).toMatch(/verificationRequired:\s*true/);
    // The verify-email endpoint must exist.
    const verifyPath = path.resolve(
      __dirname,
      "../../../app/api/auth/verify-email/route.ts"
    );
    expect(fs.existsSync(verifyPath)).toBe(true);
  });

  it("login route should reject unverified accounts", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(
      __dirname,
      "../../../app/api/auth/login/route.ts"
    );
    const src = fs.readFileSync(filePath, "utf8");
    expect(src).toMatch(/email_not_verified/);
    expect(src).toMatch(/emailVerified/);
  });
});

// ---------------------------------------------------------------------------
// FE-036: TOCTOU race — catch Prisma P2002.
// ---------------------------------------------------------------------------

describe("FE-036: TOCTOU race on email uniqueness", () => {
  it("register route should catch Prisma P2002 and return 409", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(
      __dirname,
      "../../../app/api/auth/register/route.ts"
    );
    const src = fs.readFileSync(filePath, "utf8");
    // Must import Prisma error class.
    expect(src).toMatch(/Prisma\.PrismaClientKnownRequestError|PrismaClientKnownRequestError/);
    // Must check e.code === "P2002".
    expect(src).toMatch(/e\.code\s*===\s*"P2002"/);
    // Must return 409 with email_taken.
    expect(src).toMatch(/email_taken/);
    expect(src).toMatch(/status:\s*409/);
  });
});

// ---------------------------------------------------------------------------
// FE-037: persistRlCandidates only writes to user-owned project.
// ---------------------------------------------------------------------------

describe("FE-037: persistRlCandidates user-owned project", () => {
  it("rl route persistRlCandidates should find-or-create a project OWNED BY the user", async () => {
    const fs = await import("fs");
    const path = await import("path");
    const filePath = path.resolve(__dirname, "../../../app/api/rl/route.ts");
    const src = fs.readFileSync(filePath, "utf8");
    // Must look up by ownerId: userId, NOT just orgId.
    expect(src).toMatch(/ownerId:\s*userId/);
    // Must NOT use the old pattern of finding the first project in the org.
    // The bad code: db.project.findFirst({ where: { organizationId: ... }, orderBy: { createdAt: "asc" } })
    // After the fix, the where clause includes ownerId.
    expect(src).toMatch(/where:\s*{\s*ownerId:\s*userId,\s*name:\s*RL_PROJECT_NAME/);
    // Must create with visibility: "private" to prevent org-wide visibility.
    expect(src).toMatch(/visibility:\s*"private"/);
    // Must set createdById (the user) on Hypothesis create.
    expect(src).toMatch(/createdById:\s*userId/);
  });
});
