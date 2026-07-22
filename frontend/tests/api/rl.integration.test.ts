/**
 * Integration tests for /api/rl (Issue 237).
 *
 * Verifies that the /api/rl route proxies correctly to the Phase 4
 * RL Hypothesis Ranker service (RL_SERVICE_URL/rank).
 *
 * Test strategy: same as predict.integration.test.ts — mock global.fetch,
 * import the route handler, assert proxying behavior.
 */

import { describe, it, expect, beforeEach, afterEach, jest } from "@jest/globals";

const originalFetch = global.fetch;

function buildMockRequest(
  body: unknown,
  method: "POST" | "GET" = "POST",
  searchParams: Record<string, string> = {},
): {
  method: string;
  headers: Record<string, string>;
  body: string;
  json: () => Promise<unknown>;
  nextUrl: { searchParams: URLSearchParams };
  signal: AbortSignal;
} {
  const url = new URL("http://localhost:3000/api/rl");
  for (const [k, v] of Object.entries(searchParams)) {
    url.searchParams.set(k, v);
  }
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: typeof body === "string" ? body : JSON.stringify(body || {}),
    json: async () => (typeof body === "string" ? JSON.parse(body) : body || {}),
    nextUrl: { searchParams: url.searchParams },
    signal: new AbortController().signal,
  };
}

// Mock auth + CSRF + rate limiter.
jest.mock("@/lib/api-helpers", () => ({
  requireAuth: async () => ({
    user: {
      userId: "test-user-1",
      email: "test@example.com",
      role: "researcher",
      orgId: "org-1",
    },
    response: null,
  }),
  requireCsrfOrSend: async () => ({ response: null }),
  internalError: (msg: string) =>
    new Response(JSON.stringify({ error: "internal_error", message: msg }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    }),
  writeAuditLog: async () => undefined,
}));

jest.mock("@/lib/zod-schemas", () => ({
  RlBody: {},
  validateBody: (_schema: unknown, body: unknown) => ({
    ok: true as const,
    data: body,
    response: null,
  }),
}));

jest.mock("@/lib/auth/per-user-rate-limit", () => ({
  checkUserRateLimitDistributed: async () => ({
    blocked: false,
    retryAfterSeconds: 0,
  }),
  checkUserRateLimit: () => ({ blocked: false, retryAfterSeconds: 0 }),
}));

jest.mock("@/lib/services/ml-stubs", () => ({
  checkRlAvailability: () => ({
    available: Boolean(process.env.RL_SERVICE_URL),
    service: "RL Hypothesis Ranker",
    description: "",
    reason: "",
  }),
}));

// Mock the db so persistRlCandidates doesn't fail.
jest.mock("@/lib/db", () => ({
  db: {
    hypothesis: {
      create: async () => ({ id: "test-hypothesis-1" }),
    },
    project: {
      findFirst: async () => ({ id: "test-project-1", organizationId: "org-1" }),
    },
  },
}));

describe("Issue 237: /api/rl proxies to RL_SERVICE_URL/rank", () => {
  beforeEach(() => {
    process.env.RL_SERVICE_URL = "http://localhost:8004";
  });

  afterEach(() => {
    global.fetch = originalFetch;
    delete process.env.RL_SERVICE_URL;
    jest.restoreAllMocks();
  });

  it("GET /api/rl proxies to RL_SERVICE_URL/rank and returns candidates", async () => {
    const mockResponse = {
      candidates: [
        {
          drug: "Aspirin",
          disease: "headache",
          rank: 1,
          reward: 0.85,
          gnnScore: 0.9,
          safetyScore: 0.8,
          marketScore: 0.7,
          overallScore: 0.82,
        },
        {
          drug: "Aspirin",
          disease: "fever",
          rank: 2,
          reward: 0.78,
          gnnScore: 0.85,
          safetyScore: 0.75,
          marketScore: 0.65,
          overallScore: 0.75,
        },
      ],
      source: "service",
      modelVersion: "rl_drug_ranker.py-v105",
      generatedAt: "2026-07-16T10:00:00Z",
      total: 2,
      page: 0,
      pageSize: 50,
      count: 2,
      csvPath: "/data/top_candidates_20260716.csv",
      backend: "csv",
    };

    const fetchMock = jest.fn<(input: string | URL, init?: RequestInit) => Promise<Response>>();
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify(mockResponse), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }) as Response,
    );
    global.fetch = fetchMock as unknown as typeof global.fetch;

    const { GET } = await import("@/app/api/rl/route");

    const req = buildMockRequest({}, "GET", { drug: "Aspirin" });
    const response = await GET(req as never);
    const body = await response.json();

    // Verify the route forwarded to /rank with the drug param.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url] = fetchMock.mock.calls[0];
    const calledUrl = String(url);
    expect(calledUrl).toContain("http://localhost:8004/rank");
    expect(calledUrl).toContain("drug=Aspirin");

    // Verify the response shape.
    expect(response.status).toBe(200);
    expect(body.candidates).toHaveLength(2);
    expect(body.candidates[0].drug).toBe("Aspirin");
    expect(body.candidates[0].rank).toBe(1);
    expect(body.source).toBe("rl_service");
    expect(body.total).toBe(2);
  });

  it("returns source:none when RL_SERVICE_URL is not set", async () => {
    delete process.env.RL_SERVICE_URL;

    const { GET } = await import("@/app/api/rl/route");

    const req = buildMockRequest({}, "GET");
    const response = await GET(req as never);

    // When neither the service URL nor a local CSV is available, the
    // route returns 503 with a service_not_deployed error.
    expect(response.status).toBe(503);
    const body = await response.json();
    expect(body.error).toBe("service_not_deployed");
    expect(body.reason).toContain("RL_SERVICE_URL");
  });

  it("retries on 5xx and surfaces the error after retries are exhausted", async () => {
    // Use mockImplementation for fresh Response objects on each retry.
    const fetchMock = jest.fn<(input: string | URL, init?: RequestInit) => Promise<Response>>();
    fetchMock.mockImplementation(async () =>
      new Response(JSON.stringify({ detail: "internal error" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }) as Response,
    );
    global.fetch = fetchMock as unknown as typeof global.fetch;

    const { GET } = await import("@/app/api/rl/route");

    const req = buildMockRequest({}, "GET");
    const response = await GET(req as never);

    // After retries, the route should surface a 500-level error.
    expect(response.status).toBeGreaterThanOrEqual(500);
    // Verify retries happened (maxRetries=3 for rl-ranker).
    expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(3);
  });
});
