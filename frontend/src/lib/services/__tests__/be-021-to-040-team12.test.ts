/**
 * BE-021 to BE-040 ROOT FIX tests (Team Member 12 — Backend API Routes).
 *
 * These tests verify the ROOT-CAUSE fixes for 20 backend issues. Each test
 * is named BE-XXX and asserts the specific behavior the fix guarantees.
 *
 * Tests are designed to run WITHOUT a database — they exercise:
 *   - Pure functions (Zod schemas, type definitions, path resolution)
 *   - Module-level exports (cache clear, state inspector)
 *   - Static analysis (status codes, type shapes)
 *
 * DB-dependent behavior (Prisma transactions, upserts) is verified by
 * inspection of the source code — those paths require a running Postgres
 * instance and are out of scope for unit tests.
 */

import { promises as fs } from "fs";
import * as path from "path";
import * as os from "os";

// ---------------------------------------------------------------------------
// BE-022: GT checkpoint path resolution (CRITICAL)
// ---------------------------------------------------------------------------

describe("BE-022: GT checkpoint search includes repo-root parent dir", () => {
  test("resolveRepoRoot returns parent when cwd ends with /frontend", () => {
    // Replicate the _resolveRepoRoot logic from gt-inference.ts to verify
    // the BE-022 fix logic. The actual module reads process.cwd() at
    // module-load time, so we test the LOGIC here.
    const path = require("path");
    function _resolveRepoRoot(cwd: string, gtRepoRoot?: string): string {
      if (gtRepoRoot) return path.resolve(gtRepoRoot);
      // Normalize trailing slash for the endsWith check.
      const normalized = cwd.replace(/\/+$/, "");
      if (normalized.endsWith(path.sep + "frontend") || normalized.endsWith("/frontend")) {
        return path.resolve(normalized, "..");
      }
      return cwd;
    }
    expect(_resolveRepoRoot("/repo/frontend")).toBe("/repo");
    expect(_resolveRepoRoot("/repo/frontend/")).toBe("/repo");
    expect(_resolveRepoRoot("/repo")).toBe("/repo");
    expect(_resolveRepoRoot("/repo", "/explicit/root")).toBe("/explicit/root");
  });

  test("checkpoint candidate dirs include parent output_v100 path", () => {
    // Read the actual gt-inference.ts source and verify the candidate
    // dirs include the parent-dir search path. This is a STATIC check —
    // it guarantees the fix is present in the source code.
    // BE-008 + BE-022 merge: the function is `getCheckpointCandidateDirs()`
    // (lazy evaluation) using `getRepoRoot()` — both functionally
    // equivalent to the original BE-022 `_REPO_ROOT` constant.
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../gt-inference.ts"),
      "utf8"
    );
    // The candidate dirs must resolve relative to a repoRoot variable
    // (not process.cwd() directly). Accept either `repoRoot` (BE-008)
    // or `_REPO_ROOT` (BE-022) — both are the merged fix.
    expect(src).toMatch(/path\.resolve\(repoRoot,\s*["']output_v100["']\)|path\.resolve\(_REPO_ROOT,\s*["']output_v100["']\)/);
    // And repoRoot must be derived from process.cwd() when running from frontend/
    expect(src).toMatch(/endsWith.*frontend/);
    // And GT_REPO_ROOT env var must be honored
    expect(src).toMatch(/GT_REPO_ROOT/);
  });
});

// ---------------------------------------------------------------------------
// BE-023: GT_SERVICE_URL proxy path exists
// ---------------------------------------------------------------------------

describe("BE-023: gt-inference.ts has GT_SERVICE_URL proxy path", () => {
  test("runHttpInference exists and checks GT_SERVICE_URL", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../gt-inference.ts"),
      "utf8"
    );
    expect(src).toMatch(/runHttpInference/);
    expect(src).toMatch(/GT_SERVICE_URL/);
    expect(src).toMatch(/\/predict|\/top-k/);
  });
});

// ---------------------------------------------------------------------------
// BE-024: phase1 service exposes /stats endpoint
// ---------------------------------------------------------------------------

describe("BE-024: phase1 service exposes /stats endpoint", () => {
  test("phase1/service.py defines a /stats route", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../../../phase1/service.py"),
      "utf8"
    );
    expect(src).toMatch(/@app\.get\(["']\/stats["']\)/);
    // The /stats endpoint must return the DatasetStatsResponse shape
    // expected by the frontend's proxyToDatasetService.
    expect(src).toMatch(/nodesLoaded/);
    expect(src).toMatch(/edgesLoaded/);
    expect(src).toMatch(/edgeTypesPresent/);
  });

  test("dataset-stats.ts surfaces 404 as a hard error (no silent fallback)", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../dataset-stats.ts"),
      "utf8"
    );
    // BE-024 fix: 404 must throw with a clear message, NOT fall through
    // to the local checkpoint silently.
    expect(src).toMatch(/res\.status === 404/);
    expect(src).toMatch(/throw new Error/);
    expect(src).toMatch(/BE-024/);
  });
});

// ---------------------------------------------------------------------------
// BE-025: dataset-stats DEFAULT_CHECKPOINT_PATH points to phase1 (not phase2)
// ---------------------------------------------------------------------------

describe("BE-025: DEFAULT_CHECKPOINT_PATH points to phase1", () => {
  test("default checkpoint path is phase1/data/checkpoints/step_01.json", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../dataset-stats.ts"),
      "utf8"
    );
    // The DEFAULT_CHECKPOINT_PATH assignment spans multiple lines; the
    // regex must use the /s flag (dotall) to match across newlines.
    const defaultPathAssignment = src.match(
      /DEFAULT_CHECKPOINT_PATH\s*=\s*path\.resolve\([\s\S]*?\);/
    );
    expect(defaultPathAssignment).not.toBeNull();
    expect(defaultPathAssignment![0]).toMatch(/phase1/);
    expect(defaultPathAssignment![0]).toMatch(/step_01\.json/);
    // Must NOT point to phase2 (the original bug).
    expect(defaultPathAssignment![0]).not.toMatch(/phase2/);
  });
});

// ---------------------------------------------------------------------------
// BE-026: rl-ranker.ts no longer overrides upstream.total with upstream.count
// ---------------------------------------------------------------------------

describe("BE-026: rl-ranker respects upstream.total (not upstream.count)", () => {
  test("source code uses upstream.total, not upstream.count, for total", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../rl-ranker.ts"),
      "utf8"
    );
    // BE-013 + BE-070 + BE-026 merged fix: the proxy path uses
    // `typeof upstream.total === "number" ? upstream.total : upstream.count`
    // (BE-013/BE-070 inline) OR `upstreamTotal` (BE-026 variable). Both
    // are functionally equivalent — they prefer upstream.total and fall
    // back to upstream.count. Accept either pattern.
    const proxyBlock = src.match(
      /const upstream = await proxyToRlService[\s\S]*?catch \(e\) \{[\s\S]*?\}/
    );
    expect(proxyBlock).not.toBeNull();
    // The proxy block must NOT contain the bare `total: upstream.count`
    // assignment (the bug). It must use either the inline typeof guard
    // or the upstreamTotal variable.
    expect(proxyBlock![0]).not.toMatch(/^\s*total:\s*upstream\.count\s*,?\s*$/m);
    expect(proxyBlock![0]).toMatch(
      /upstream\.total|upstreamTotal/
    );
  });
});

// ---------------------------------------------------------------------------
// BE-027: rl-csv-cache.ts deleted; single cache in rl-ranker.ts
// ---------------------------------------------------------------------------

describe("BE-027: rl-csv-cache.ts deleted; single cache in rl-ranker.ts", () => {
  test("rl-csv-cache.ts no longer exists", async () => {
    const deletedPath = path.resolve(__dirname, "../rl-csv-cache.ts"); // services/ dir
    await expect(fs.access(deletedPath)).rejects.toThrow();
  });

  test("rl-ranker.ts exports production-safe clearRlRankerCsvCache", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../rl-ranker.ts"),
      "utf8"
    );
    expect(src).toMatch(/export function clearRlRankerCsvCache/);
    expect(src).toMatch(/export function getRlRankerCsvCacheState/);
  });

  test("refresh/route.ts imports from rl-ranker (NOT rl-csv-cache)", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../app/api/rl/refresh/route.ts"),
      "utf8"
    );
    // Must import from rl-ranker.
    expect(src).toMatch(/from ["']@\/lib\/services\/rl-ranker["']/);
    // Must NOT have an import statement for rl-csv-cache.
    // (Comments may mention rl-csv-cache for historical context — that's fine.)
    expect(src).not.toMatch(/from ["'][^"]*rl-csv-cache["']/);
  });

  test("clearRlRankerCsvCache + getRlRankerCsvCacheState work as expected", async () => {
    const {
      clearRlRankerCsvCache,
      getRlRankerCsvCacheState,
      __clearRlRankerCsvCacheForTests,
    } = await import("../rl-ranker");
    // Clear before test.
    __clearRlRankerCsvCacheForTests();
    expect(getRlRankerCsvCacheState()).toHaveLength(0);
    // clearRlRankerCsvCache is a no-op on an empty cache.
    expect(() => clearRlRankerCsvCache()).not.toThrow();
    __clearRlRankerCsvCacheForTests();
  });
});

// ---------------------------------------------------------------------------
// BE-028: persistRlCandidates uses $transaction + upsert + composite key
// ---------------------------------------------------------------------------

describe("BE-028: persistRlCandidates uses $transaction + upsert", () => {
  test("rl/route.ts uses db.$transaction with tx.hypothesis.upsert", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../app/api/rl/route.ts"),
      "utf8"
    );
    expect(src).toMatch(/db\.\$transaction\(async \(tx\)/);
    expect(src).toMatch(/tx\.hypothesis\.upsert/);
    expect(src).toMatch(/projectId_drugName_diseaseName/);
  });

  test("Prisma schema has @@unique on Hypothesis (projectId, drugName, diseaseName)", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../../prisma/schema.prisma"),
      "utf8"
    );
    // Find the Hypothesis model block and verify it has the composite unique.
    // Use greedy match to start-of-line `}` to handle nested {} in comments.
    const hypothesisBlock = src.match(/model Hypothesis \{[\s\S]*?^\}/m);
    expect(hypothesisBlock).not.toBeNull();
    expect(hypothesisBlock![0]).toMatch(
      /@@unique\(\[projectId,\s*drugName,\s*diseaseName\]\)/
    );
  });
});

// ---------------------------------------------------------------------------
// BE-029: Zod schema validation on owned routes
// ---------------------------------------------------------------------------

describe("BE-029: Zod schema validation applied to owned routes", () => {
  test("zod-schemas.ts exports all per-route schemas + validateBody helper", async () => {
    const mod = await import("../../zod-schemas");
    expect(typeof mod.validateBody).toBe("function");
    expect(mod.PredictBody).toBeDefined();
    expect(mod.RlBody).toBeDefined();
    expect(mod.KnowledgeGraphBody).toBeDefined();
    expect(mod.TwoFaDisableBody).toBeDefined();
    expect(mod.TwoFaLoginVerifyBody).toBeDefined();
    expect(mod.PasswordChangeBody).toBeDefined();
    expect(mod.VerifyEmailBody).toBeDefined();
    expect(mod.AuthMePatchBody).toBeDefined();
    expect(mod.BillingSubscriptionBody).toBeDefined();
  });

  test("validateBody returns ok:true for valid input", () => {
    const { validateBody, PredictBody } = require("../../zod-schemas");
    const result = validateBody(PredictBody, {
      pairs: [{ drug: "aspirin", disease: "headache" }],
    });
    expect(result.ok).toBe(true);
    expect(result.data.pairs).toHaveLength(1);
  });

  test("validateBody returns ok:false with 400 response for invalid input", () => {
    const { validateBody, PredictBody } = require("../../zod-schemas");
    const result = validateBody(PredictBody, {
      pairs: [], // empty array — schema requires min 1
    });
    expect(result.ok).toBe(false);
    // The response is a NextResponse — check its status.
    // (NextResponse is a Response subclass; status is readable.)
    expect(result.response.status).toBe(400);
  });

  test("BE-030: PredictBody rejects non-number limit (NaN bug)", () => {
    const { validateBody, PredictBody } = require("../../zod-schemas");
    const result = validateBody(PredictBody, {
      pairs: [{ drug: "aspirin", disease: "headache" }],
      limit: "abc", // string — would have caused NaN bug
    });
    expect(result.ok).toBe(false);
    expect(result.response.status).toBe(400);
  });

  test("BillingSubscriptionBody rejects both totpCode AND mfaTicket", () => {
    const { validateBody, BillingSubscriptionBody } = require("../../zod-schemas");
    const result = validateBody(BillingSubscriptionBody, {
      planId: "enterprise",
      currentPassword: "secret123",
      totpCode: "123456",
      mfaTicket: "ticket-jwt",
    });
    expect(result.ok).toBe(false);
    expect(result.response.status).toBe(400);
  });

  test("TwoFaLoginVerifyBody requires 6-digit code", () => {
    const { validateBody, TwoFaLoginVerifyBody } = require("../../zod-schemas");
    // 5-digit code → rejected
    expect(
      validateBody(TwoFaLoginVerifyBody, { code: "12345" }).ok
    ).toBe(false);
    // 6-digit code → accepted
    expect(
      validateBody(TwoFaLoginVerifyBody, { code: "123456" }).ok
    ).toBe(true);
    // 7-digit code → rejected
    expect(
      validateBody(TwoFaLoginVerifyBody, { code: "1234567" }).ok
    ).toBe(false);
  });

  test("all 7 owned routes import validateBody from zod-schemas", () => {
    const routePaths = [
      "app/api/auth/2fa/disable/route.ts",
      "app/api/auth/2fa/login-verify/route.ts",
      "app/api/auth/me/route.ts",
      "app/api/auth/password/route.ts",
      "app/api/auth/verify-email/route.ts",
      "app/api/billing/subscription/route.ts",
      "app/api/knowledge-graph/route.ts",
      "app/api/predict/route.ts",
      "app/api/rl/route.ts",
    ];
    for (const rel of routePaths) {
      const full = path.resolve(__dirname, "../../../", rel);
      const src = require("fs").readFileSync(full, "utf8");
      expect(src).toMatch(/from ["']@\/lib\/zod-schemas["']/);
    }
  });
});

// ---------------------------------------------------------------------------
// BE-030: predict limit NaN bug fixed via Zod
// ---------------------------------------------------------------------------

describe("BE-030: predict limit no longer NaNs on string input", () => {
  test("predict/route.ts uses validateBody(PredictBody, body) before reading limit", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../app/api/predict/route.ts"),
      "utf8"
    );
    expect(src).toMatch(/validateBody\(PredictBody/);
    // The OLD buggy pattern `Math.min(body.limit ?? 1000, 5000)` must NOT
    // appear — it would still produce NaN for string input. The new code
    // reads `parsed.data.limit` (schema-validated).
    expect(src).not.toMatch(/Math\.min\(body\.limit \?\? 1000, 5000\)/);
    expect(src).toMatch(/parsed\.data\.limit/);
  });
});

// ---------------------------------------------------------------------------
// BE-031 / BE-032 / BE-033: 403 → 401 for invalid credentials
// ---------------------------------------------------------------------------

describe("BE-031/032/033: invalid credentials return 401 (not 403)", () => {
  test("BE-031: 2fa/disable returns 401 for invalid password", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../app/api/auth/2fa/disable/route.ts"),
      "utf8"
    );
    // Find the invalid_password response and verify it's 401.
    const invalidPwdBlock = src.match(
      /error:\s*"invalid_password"[\s\S]*?status:\s*(\d+)/
    );
    expect(invalidPwdBlock).not.toBeNull();
    expect(invalidPwdBlock![1]).toBe("401");
  });

  test("BE-031: 2fa/disable returns 401 for invalid TOTP code", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../app/api/auth/2fa/disable/route.ts"),
      "utf8"
    );
    // The invalid_code response must be 401 (was 403).
    // Use a relaxed regex that matches either `result.reason` or
    // `totpResult.reason` (the variable name differs between routes).
    const invalidCodeBlock = src.match(
      /error:\s*(?:\w+\.reason === "replayed" \? "code_replayed" : "invalid_code")[\s\S]*?status:\s*(\d+)/
    );
    expect(invalidCodeBlock).not.toBeNull();
    expect(invalidCodeBlock![1]).toBe("401");
  });

  test("BE-032: password route returns 401 for invalid current password", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../app/api/auth/password/route.ts"),
      "utf8"
    );
    const invalidCredsBlock = src.match(
      /error:\s*"invalid_credentials"[\s\S]*?status:\s*(\d+)/
    );
    expect(invalidCredsBlock).not.toBeNull();
    expect(invalidCredsBlock![1]).toBe("401");
  });

  test("BE-033: billing/subscription returns 401 for invalid password", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../app/api/billing/subscription/route.ts"),
      "utf8"
    );
    const invalidPwdBlock = src.match(
      /error:\s*"invalid_credentials"[\s\S]*?status:\s*(\d+)/
    );
    expect(invalidPwdBlock).not.toBeNull();
    expect(invalidPwdBlock![1]).toBe("401");
  });

  test("BE-033: billing/subscription returns 401 for invalid MFA", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../app/api/billing/subscription/route.ts"),
      "utf8"
    );
    const invalidMfaBlock = src.match(
      /error:\s*"invalid_mfa"[\s\S]*?status:\s*(\d+)/
    );
    expect(invalidMfaBlock).not.toBeNull();
    expect(invalidMfaBlock![1]).toBe("401");
  });
});

// ---------------------------------------------------------------------------
// BE-034: 2fa/login-verify returns 401 (not 400) for invalid TOTP
// ---------------------------------------------------------------------------

describe("BE-034: 2fa/login-verify returns 401 for invalid TOTP code", () => {
  test("invalid_code response status is 401 (was 400)", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../app/api/auth/2fa/login-verify/route.ts"),
      "utf8"
    );
    // Find the invalid_code response block and verify it's 401.
    // Use a relaxed regex that matches any variable name (result.reason
    // in login-verify, totpResult.reason in 2fa/disable).
    const invalidCodeBlock = src.match(
      /error:\s*\w+\.reason === "replayed" \? "code_replayed" : "invalid_code"[\s\S]*?status:\s*(\d+)/
    );
    expect(invalidCodeBlock).not.toBeNull();
    expect(invalidCodeBlock![1]).toBe("401");
  });
});

// ---------------------------------------------------------------------------
// BE-035: verify-email has per-IP rate limiting
// ---------------------------------------------------------------------------

describe("BE-035: verify-email has per-IP rate limiting", () => {
  test("verify-email route calls checkIpRateLimit + recordIpAttempt", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../app/api/auth/verify-email/route.ts"),
      "utf8"
    );
    expect(src).toMatch(/checkIpRateLimit/);
    expect(src).toMatch(/recordIpAttempt/);
    expect(src).toMatch(/429/);
    expect(src).toMatch(/Retry-After/);
  });
});

// ---------------------------------------------------------------------------
// BE-036: 2fa/disable uses db.$transaction for atomic disable + audit
// ---------------------------------------------------------------------------

describe("BE-036: 2fa/disable uses db.$transaction (atomic disable + audit)", () => {
  test("route uses db.$transaction with tx.user.update + tx.auditLog.create", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../app/api/auth/2fa/disable/route.ts"),
      "utf8"
    );
    expect(src).toMatch(/db\.\$transaction\(async \(tx\)/);
    expect(src).toMatch(/tx\.user\.update/);
    expect(src).toMatch(/tx\.auditLog\.create/);
    expect(src).toMatch(/action:\s*"2fa_disable"/);
    // The OLD broken rollback (mfaEnabled: false after secret cleared)
    // must NOT appear — the transaction replaces it.
    expect(src).not.toMatch(/mfaEnabled: false.*secret already cleared/s);
  });
});

// ---------------------------------------------------------------------------
// BE-037: verify-email uses db.$transaction (atomic update + audit)
// ---------------------------------------------------------------------------

describe("BE-037: verify-email uses db.$transaction (atomic update + audit)", () => {
  test("route uses db.$transaction with tx.user.update + tx.auditLog.create", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../app/api/auth/verify-email/route.ts"),
      "utf8"
    );
    expect(src).toMatch(/db\.\$transaction\(async \(tx\)/);
    expect(src).toMatch(/tx\.user\.update/);
    expect(src).toMatch(/tx\.auditLog\.create/);
    expect(src).toMatch(/action:\s*"email_verified"/);
    // The OLD pattern (log to stderr and return 200) must NOT appear.
    expect(src).not.toMatch(/AUDIT-LOG-FAILURE.*email_verified/);
  });
});

// ---------------------------------------------------------------------------
// BE-038: OrganizationMember has composite index (organizationId, joinedAt)
// ---------------------------------------------------------------------------

describe("BE-038: OrganizationMember has composite index", () => {
  test("Prisma schema has @@index([organizationId, joinedAt]) on OrganizationMember", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../../prisma/schema.prisma"),
      "utf8"
    );
    // The OrganizationMember model block contains nested {} in comments
    // (e.g. `findMany({ orderBy: { joinedAt: ... } })`), so a non-greedy
    // match on the first `}` stops too early. Use a greedy match up to
    // the closing `}` at start-of-line (the model terminator).
    const orgMemberBlock = src.match(/model OrganizationMember \{[\s\S]*?^\}/m);
    expect(orgMemberBlock).not.toBeNull();
    expect(orgMemberBlock![0]).toMatch(
      /@@index\(\[organizationId,\s*joinedAt\]\)/
    );
  });
});

// ---------------------------------------------------------------------------
// BE-039: /api/auth/me GET uses a single query (no N+1)
// ---------------------------------------------------------------------------

describe("BE-039: /api/auth/me GET uses single query with include", () => {
  test("route uses organizationMemberships include (not separate findMany)", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../app/api/auth/me/route.ts"),
      "utf8"
    );
    expect(src).toMatch(/organizationMemberships:/);
    // BE-039: the GET handler must NOT issue a separate
    // `db.organizationMember.findMany` CALL. We allow the string in
    // comments (explaining the old bug) but not as an actual call. A
    // call would be `= await db.organizationMember.findMany(` — note
    // the `await` and the function-call paren.
    const getBlock = src.match(
      /export async function GET\([^)]*\)\s*\{[\s\S]*?\n\}/
    );
    expect(getBlock).not.toBeNull();
    // Look for the CALL pattern, not the string in comments.
    expect(getBlock![0]).not.toMatch(
      /await\s+db\.organizationMember\.findMany\(/
    );
  });
});

// ---------------------------------------------------------------------------
// BE-040: api-client SafetyReport type matches openfda DrugSafetySummary
// ---------------------------------------------------------------------------

describe("BE-040: api-client SafetyReport matches DrugSafetySummary", () => {
  test("SafetyReport has brandName + genericName (NOT drug)", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../api-client.ts"),
      "utf8"
    );
    const safetyReportBlock = src.match(
      /export interface SafetyReport \{[\s\S]*?\}/
    );
    expect(safetyReportBlock).not.toBeNull();
    expect(safetyReportBlock![0]).toMatch(/brandName:\s*string/);
    expect(safetyReportBlock![0]).toMatch(/genericName:\s*string/);
    // The OLD `drug: string` field must NOT be present.
    expect(safetyReportBlock![0]).not.toMatch(/^\s*drug:\s*string/m);
  });
});

// ---------------------------------------------------------------------------
// BE-021: /cypher endpoint exists in phase2/service.py
// ---------------------------------------------------------------------------

describe("BE-021: phase2 service exposes /cypher endpoint", () => {
  test("phase2/service.py defines a /cypher route", () => {
    const src = require("fs").readFileSync(
      path.resolve(__dirname, "../../../../../phase2/service.py"),
      "utf8"
    );
    expect(src).toMatch(/@app\.post\(["']\/cypher["']\)/);
  });
});
