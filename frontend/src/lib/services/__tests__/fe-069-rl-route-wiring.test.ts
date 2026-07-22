/**
 * FE-069 ROOT FIX tests: /api/rl route handler wiring.
 *
 * These tests verify that the /api/rl GET and POST handlers ACTUALLY:
 *   1. Import and call checkUserRateLimit (the per-user rate limiter).
 *   2. Return 429 with Retry-After when the user exceeds 60 req/min.
 *   3. Call getRankedHypotheses (which internally caches the CSV — verified
 *      by fe-069-rl-ranker-cache.test.ts).
 *
 * The previous "fix" added the per-user-rate-limit.ts module AND a test for
 * that module, but NEVER WIRED IT INTO THE ROUTE HANDLER. The route had no
 * rate limiting. These tests catch exactly that class of "module exists but
 * is dead code" bug by exercising the route handler itself.
 *
 * Architecture note:
 *   - The CSV cache lives INSIDE rl-ranker.ts's readLocalCsv (TTL + mtime +
 *     fs.watch) — verified by fe-069-rl-ranker-cache.test.ts.
 *   - The rate limiter lives at the route boundary (this file's concern).
 *   - The route also has CSRF protection (requireCsrfOrSend) on POST, added
 *     by another agent. We mock it to always pass so we can test rate
 *     limiting in isolation.
 */

// --- Mocks must be set up BEFORE importing the route handlers. ---

jest.mock("next/headers", () => ({
  cookies: jest.fn(),
}));

// Mock the db module so we don't need a real Postgres/SQLite.
const dbMock = {
  organizationMember: { findFirst: jest.fn() },
  project: { findFirst: jest.fn(), create: jest.fn() },
  hypothesis: { findFirst: jest.fn(), create: jest.fn(), update: jest.fn() },
  auditLog: { create: jest.fn() },
};
jest.mock("@/lib/db", () => ({ db: dbMock }));

// Mock getAuthenticatedUser / requireAuth so we can simulate auth.
jest.mock("@/lib/auth/server", () => {
  const actual = jest.requireActual("@/lib/auth/server");
  return {
    ...actual,
    getAuthenticatedUser: jest.fn(),
  };
});

// Mock the CSRF guard so POST passes the CSRF check in tests. Also mock
// requireAuth, writeAuditLog, and internalError so the route can be
// exercised without a real DB or cookie store.
jest.mock("@/lib/api-helpers", () => {
  const actual = jest.requireActual("@/lib/api-helpers");
  return {
    ...actual,
    requireCsrfOrSend: jest.fn(async () => ({ response: null })),
    requireAuth: jest.fn(),
    writeAuditLog: jest.fn(async () => {}),
    internalError: jest.fn((msg: string) =>
      Response.json({ error: "internal_error", message: msg }, { status: 500 })
    ),
  };
});

// Mock rl-ranker so we don't touch disk; we only care that the route CALLS it.
jest.mock("@/lib/services/rl-ranker", () => ({
  getRankedHypotheses: jest.fn(),
}));

// Mock the per-user rate limiter. Default implementation delegates to the
// real one so the route truly exercises the limiter; tests override per-case.
//
// FE-017: the route now calls `checkUserRateLimitDistributed` (async) with a
// sync fallback to `checkUserRateLimit`. We mock BOTH so tests can override
// either. By default both delegate to the real implementations.
jest.mock("@/lib/auth/per-user-rate-limit", () => {
  const actual = jest.requireActual("@/lib/auth/per-user-rate-limit");
  return {
    ...actual,
    checkUserRateLimit: jest.fn(actual.checkUserRateLimit),
    checkUserRateLimitDistributed: jest.fn(actual.checkUserRateLimitDistributed),
  };
});

// Mock ml-stubs so checkRlAvailability returns a controlled value.
jest.mock("@/lib/services/ml-stubs", () => ({
  checkRlAvailability: jest.fn(() => ({
    available: false,
    service: "rl_ranker",
    description: "RL ranker service",
    reason: "not_configured",
  })),
}));

import { cookies } from "next/headers";
import { getAuthenticatedUser } from "@/lib/auth/server";
import { requireAuth, requireCsrfOrSend } from "@/lib/api-helpers";
import { POST, GET } from "@/app/api/rl/route";
import { NextRequest } from "next/server";
import { getRankedHypotheses } from "@/lib/services/rl-ranker";
import { checkUserRateLimit, checkUserRateLimitDistributed } from "@/lib/auth/per-user-rate-limit";

const AUTHED_USER = {
  userId: "curuser000000000000000001",
  email: "researcher@example.com",
  role: "researcher",
};

describe("FE-069: /api/rl route wiring — rate limit is actually called", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    jest.clearAllMocks();
    // Restore the real rate-limiter behavior by default (delegating mock).
    const actual = jest.requireActual("@/lib/auth/per-user-rate-limit");
    (checkUserRateLimit as jest.Mock).mockImplementation(actual.checkUserRateLimit);
    (checkUserRateLimit as jest.Mock).mockClear();
    // FE-017: the route now calls the async distributed limiter; restore
    // its real implementation too so tests that don't override it exercise
    // the real in-memory backend (REDIS_URL is unset in tests).
    (checkUserRateLimitDistributed as jest.Mock).mockImplementation(actual.checkUserRateLimitDistributed);
    (checkUserRateLimitDistributed as jest.Mock).mockClear();

    // Auth: simulate an authenticated user.
    (getAuthenticatedUser as jest.Mock).mockResolvedValue(AUTHED_USER);
    (requireAuth as jest.Mock).mockResolvedValue({
      user: AUTHED_USER,
      response: null,
    });
    // CSRF: always pass.
    (requireCsrfOrSend as jest.Mock).mockResolvedValue({ response: null });

    (cookies as unknown as jest.Mock).mockResolvedValue({
      get: jest.fn(() => undefined),
      set: jest.fn(),
      delete: jest.fn(),
    });

    // rl-ranker: return empty by default.
    (getRankedHypotheses as jest.Mock).mockResolvedValue({
      candidates: [],
      source: "none",
      generatedAt: new Date().toISOString(),
      count: 0,
      csvPath: "/tmp/x",
      note: "test",
    });

    // DB: no org membership → persistRlCandidates is a no-op.
    dbMock.organizationMember.findFirst.mockResolvedValue(null);

    process.env = { ...originalEnv };
    process.env.RL_OUTPUT_CSV_PATH = "/tmp/nonexistent-rl.csv";
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  test("POST /api/rl calls checkUserRateLimit with the authenticated userId", async () => {
    const req = new NextRequest("http://localhost/api/rl", {
      method: "POST",
      body: JSON.stringify({}),
      headers: { "Content-Type": "application/json" },
    });
    await POST(req);

    // FE-017: the route now calls the ASYNC distributed limiter. The sync
    // limiter is only a fallback. Either way, the userId must be passed.
    const limiterCalls = (checkUserRateLimitDistributed as jest.Mock).mock.calls.length
      + (checkUserRateLimit as jest.Mock).mock.calls.length;
    expect(limiterCalls).toBeGreaterThanOrEqual(1);
    const firstCall = (checkUserRateLimitDistributed as jest.Mock).mock.calls[0]
      ?? (checkUserRateLimit as jest.Mock).mock.calls[0];
    expect(firstCall[0]).toBe(AUTHED_USER.userId);
  });

  test("GET /api/rl calls checkUserRateLimit with the authenticated userId", async () => {
    // FE-033 ROOT FIX: GET now accepts a NextRequest so it can read sort +
    // pagination query params. Pass a minimal request.
    const req = new NextRequest("http://localhost/api/rl", { method: "GET" });
    await GET(req);

    const limiterCalls = (checkUserRateLimitDistributed as jest.Mock).mock.calls.length
      + (checkUserRateLimit as jest.Mock).mock.calls.length;
    expect(limiterCalls).toBeGreaterThanOrEqual(1);
    const firstCall = (checkUserRateLimitDistributed as jest.Mock).mock.calls[0]
      ?? (checkUserRateLimit as jest.Mock).mock.calls[0];
    expect(firstCall[0]).toBe(AUTHED_USER.userId);
  });

  test("POST /api/rl returns 429 with Retry-After when rate limit is exceeded", async () => {
    // Force the limiter to report "blocked". FE-017: the route calls the
    // async distributed limiter first, so mock THAT one.
    (checkUserRateLimitDistributed as jest.Mock).mockResolvedValue({
      blocked: true,
      retryAfterSeconds: 42,
      remaining: 0,
    });

    const req = new NextRequest("http://localhost/api/rl", {
      method: "POST",
      body: JSON.stringify({}),
      headers: { "Content-Type": "application/json" },
    });
    const res = await POST(req);

    expect(res.status).toBe(429);
    expect(res.headers.get("Retry-After")).toBe("42");
    const body = await res.json();
    expect(body.error).toBe("rate_limited");
    // CRITICAL: getRankedHypotheses must NOT have been called — the rate
    // limit gate must reject before any disk I/O happens.
    expect(getRankedHypotheses).not.toHaveBeenCalled();
  });

  test("GET /api/rl returns 429 with Retry-After when rate limit is exceeded", async () => {
    (checkUserRateLimitDistributed as jest.Mock).mockResolvedValue({
      blocked: true,
      retryAfterSeconds: 30,
      remaining: 0,
    });

    // FE-033: GET now takes a NextRequest.
    const req = new NextRequest("http://localhost/api/rl", { method: "GET" });
    const res = await GET(req);

    expect(res.status).toBe(429);
    expect(res.headers.get("Retry-After")).toBe("30");
    expect(getRankedHypotheses).not.toHaveBeenCalled();
  });

  test("POST /api/rl calls getRankedHypotheses when rate limit passes (proves the route uses the cached lib)", async () => {
    // FE-003 ROOT FIX: the route now returns 503 when source is "none"
    // AND no service URL is set. To test the wiring (that
    // getRankedHypotheses is called), we mock it to return a non-empty
    // candidate list with source: "local_csv" — simulating a real RL
    // output CSV was found.
    (getRankedHypotheses as jest.Mock).mockResolvedValue({
      candidates: [
        { drug: "metformin", disease: "breast cancer", rank: 1, gnnScore: 0.85 },
      ],
      source: "local_csv",
      generatedAt: new Date().toISOString(),
      count: 1,
      csvPath: "/tmp/x",
      note: "test",
    });
    const req = new NextRequest("http://localhost/api/rl", {
      method: "POST",
      body: JSON.stringify({ drug: "met" }),
      headers: { "Content-Type": "application/json" },
    });
    const res = await POST(req);
    expect(res.status).toBe(200);
    // CRITICAL: getRankedHypotheses must have been called with the user's
    // query — proving the route delegates to the lib (which caches) rather
    // than parsing the CSV inline.
    expect(getRankedHypotheses).toHaveBeenCalledTimes(1);
    expect((getRankedHypotheses as jest.Mock).mock.calls[0][0]).toMatchObject({
      drug: "met",
    });
  });

  test("GET /api/rl calls getRankedHypotheses when rate limit passes", async () => {
    // FE-003 ROOT FIX: mock non-empty candidates so the route returns 200
    // and the wiring assertion holds (otherwise the 503 fallback fires).
    (getRankedHypotheses as jest.Mock).mockResolvedValue({
      candidates: [
        { drug: "metformin", disease: "breast cancer", rank: 1, gnnScore: 0.85 },
      ],
      source: "local_csv",
      generatedAt: new Date().toISOString(),
      count: 1,
      csvPath: "/tmp/x",
      note: "test",
    });
    // FE-033: GET now takes a NextRequest. The route calls getRankedHypotheses
    // with sort/sortDir/offset/pageSize (default pageSize=50, offset=0).
    const req = new NextRequest("http://localhost/api/rl", { method: "GET" });
    const res = await GET(req);
    expect(res.status).toBe(200);
    expect(getRankedHypotheses).toHaveBeenCalledTimes(1);
    // The new call signature uses pageSize + offset (not `limit`). Default
    // pageSize is 50, offset is 0. sort/sortDir are undefined (the route
    // passes them through as undefined when not provided).
    expect((getRankedHypotheses as jest.Mock).mock.calls[0][0]).toMatchObject({
      pageSize: 50,
      offset: 0,
    });
  });

  test("rate limit is per-user: a second user is not blocked by the first user's requests", () => {
    const actual = jest.requireActual("@/lib/auth/per-user-rate-limit");
    (checkUserRateLimit as jest.Mock).mockImplementation(actual.checkUserRateLimit);
    const USER_A = "curaaaa000000000000000001";
    const USER_B = "curbbbb000000000000000002";
    // Exhaust USER_A's quota.
    for (let i = 0; i < 60; i++) {
      const rl = checkUserRateLimit(USER_A, { max: 60, windowSeconds: 60 });
      expect(rl.blocked).toBe(false);
    }
    // USER_A's 61st is blocked.
    const aBlocked = checkUserRateLimit(USER_A, { max: 60, windowSeconds: 60 });
    expect(aBlocked.blocked).toBe(true);
    // USER_B's first is NOT blocked.
    const bOk = checkUserRateLimit(USER_B, { max: 60, windowSeconds: 60 });
    expect(bOk.blocked).toBe(false);
  });
});
