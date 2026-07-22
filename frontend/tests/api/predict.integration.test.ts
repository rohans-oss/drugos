/**
 * Integration tests for /api/predict (Issue 236).
 *
 * Verifies that the /api/predict route proxies correctly to the Phase 3
 * Graph Transformer service (GT_SERVICE_URL/predict).
 *
 * Test strategy:
 *   1. Mock global.fetch to simulate the Python GT service's responses.
 *   2. Import the route handler directly and invoke it with a mock
 *      NextRequest.
 *   3. Assert the route proxies the request body, forwards the response,
 *      and handles errors correctly.
 *
 * These tests do NOT require a real Python service to be running —
 * they verify the PROXYING LOGIC, not the model inference. The model
 * inference is tested by the Python service's own test suite.
 */

import { describe, it, expect, beforeEach, afterEach, jest } from "@jest/globals";

// ---------------------------------------------------------------------------
// Mock the auth layer so we don't need a real DB / JWT.
// ---------------------------------------------------------------------------

// Store the original fetch so we can restore it after each test.
const originalFetch = global.fetch;

// Helper to build a mock NextRequest with auth context.
function buildMockRequest(body: unknown, method: "POST" | "GET" = "POST"): {
  method: string;
  headers: Record<string, string>;
  body: string;
  json: () => Promise<unknown>;
  nextUrl: { searchParams: URLSearchParams };
  signal: AbortSignal;
} {
  const url = new URL("http://localhost:3000/api/predict");
  if (method === "GET" && body && typeof body === "object") {
    const params = body as Record<string, string>;
    for (const [k, v] of Object.entries(params)) {
      url.searchParams.set(k, v);
    }
  }
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: typeof body === "string" ? body : JSON.stringify(body),
    json: async () => (typeof body === "string" ? JSON.parse(body) : body),
    nextUrl: { searchParams: url.searchParams },
    signal: new AbortController().signal,
  };
}

// Mock the api-helpers module to bypass auth.
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
  internalError: (msg: string) =>
    new Response(JSON.stringify({ error: "internal_error", message: msg }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    }),
  writeAuditLog: async () => undefined,
}));

// Mock the zod-schemas validateBody to always pass.
jest.mock("@/lib/zod-schemas", () => ({
  PredictBody: {},
  validateBody: (_schema: unknown, body: unknown) => ({
    ok: true as const,
    data: body,
    response: null,
  }),
}));

describe("Issue 236: /api/predict proxies to GT_SERVICE_URL/predict", () => {
  beforeEach(() => {
    // Set GT_SERVICE_URL before each test.
    process.env.GT_SERVICE_URL = "http://localhost:8003";
  });

  afterEach(() => {
    // Restore fetch and clear env.
    global.fetch = originalFetch;
    delete process.env.GT_SERVICE_URL;
    jest.restoreAllMocks();
  });

  it("POST /api/predict proxies the pairs to GT_SERVICE_URL/predict", async () => {
    // Mock fetch to simulate the GT service's /predict response.
    const mockResponse = {
      predictions: [
        { drug: "Aspirin", disease: "headache", score: 0.85, confidence: 0.7 },
        { drug: "Aspirin", disease: "fever", score: 0.78, confidence: 0.56 },
      ],
      source: "gt_checkpoint",
      modelVersion: "gt_v110",
      generatedAt: "2026-07-16T10:00:00Z",
      count: 2,
      checkpointPath: "/data/gt_checkpoint.pt",
      error_count: 0,
      error_rate: 0.0,
    };

    const fetchMock = jest.fn<(input: string | URL, init?: RequestInit) => Promise<Response>>();
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify(mockResponse), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }) as Response,
    );
    global.fetch = fetchMock as unknown as typeof global.fetch;

    // Import the route handler AFTER mocks are set up.
    const { POST } = await import("@/app/api/predict/route");

    const req = buildMockRequest({
      pairs: [
        { drug: "Aspirin", disease: "headache" },
        { drug: "Aspirin", disease: "fever" },
      ],
    });
    const response = await POST(req as never);
    const body = await response.json();

    // Verify the route forwarded the request to the GT service.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("http://localhost:8003/predict");
    expect(init?.method).toBe("POST");
    const forwardedBody = JSON.parse(init?.body as string);
    expect(forwardedBody.pairs).toHaveLength(2);

    // Verify the route returned the proxied response.
    expect(response.status).toBe(200);
    expect(body.predictions).toHaveLength(2);
    expect(body.predictions[0]).toEqual({
      drug: "Aspirin",
      disease: "headache",
      score: 0.85,
      confidence: 0.7,
    });
    expect(body.source).toBe("gt_checkpoint");
    expect(body.count).toBe(2);
  });

  it("returns source:none with a clear note when GT_SERVICE_URL is not set", async () => {
    delete process.env.GT_SERVICE_URL;

    const { POST } = await import("@/app/api/predict/route");

    const req = buildMockRequest({
      pairs: [{ drug: "Aspirin", disease: "headache" }],
    });
    const response = await POST(req as never);
    const body = await response.json();

    // The route returns 200 with source:"none" (not 500) — the request
    // succeeded; the service is just not configured.
    expect(response.status).toBe(200);
    expect(body.source).toBe("none");
    expect(body.predictions).toEqual([]);
    expect(body.count).toBe(0);
    expect(body.note).toContain("GT_SERVICE_URL is not set");
  });

  it("returns source:none when the GT service returns 503 (no checkpoint)", async () => {
    // Use mockImplementation so each retry gets a FRESH Response object
    // (Response bodies can only be read once — mockResolvedValue would
    // return the same consumed Response on retries).
    const fetchMock = jest.fn<(input: string | URL, init?: RequestInit) => Promise<Response>>();
    fetchMock.mockImplementation(async () =>
      new Response(
        JSON.stringify({ detail: "GT model unavailable: no checkpoint" }),
        { status: 503, headers: { "Content-Type": "application/json" } },
      ) as Response,
    );
    global.fetch = fetchMock as unknown as typeof global.fetch;

    const { POST } = await import("@/app/api/predict/route");

    const req = buildMockRequest({
      pairs: [{ drug: "Aspirin", disease: "headache" }],
    });
    const response = await POST(req as never);
    const body = await response.json();

    // 503 from the service = checkpoint not loaded. Surface as source:none
    // so the dashboard shows "model not trained yet" instead of a 502.
    expect(body.source).toBe("none");
    expect(body.note).toContain("no checkpoint is loaded");
  });

  it("retries on 5xx then surfaces 502 on final failure", async () => {
    // Use mockImplementation for fresh Response objects on each retry.
    const fetchMock = jest.fn<(input: string | URL, init?: RequestInit) => Promise<Response>>();
    fetchMock.mockImplementation(async () =>
      new Response(JSON.stringify({ detail: "internal error" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }) as Response,
    );
    global.fetch = fetchMock as unknown as typeof global.fetch;

    const { POST } = await import("@/app/api/predict/route");

    const req = buildMockRequest({
      pairs: [{ drug: "Aspirin", disease: "headache" }],
    });
    const response = await POST(req as never);

    // After retries are exhausted, the route should surface a 500-level
    // error (the gt-inference.ts throws MlServiceError on 5xx, which the
    // route catches and returns as internalError → 500).
    expect(response.status).toBeGreaterThanOrEqual(500);
    // Verify retries happened (maxRetries=2 for predict means 3 total attempts).
    expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(2);
  });
});
