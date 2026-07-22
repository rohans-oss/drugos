/**
 * Integration tests for /api/knowledge-graph (Issue 238).
 *
 * Verifies that the /api/knowledge-graph route proxies correctly to the
 * Phase 2 Knowledge Graph service (KG_SERVICE_URL/kg/stats, /query, /cypher).
 */

import { describe, it, expect, beforeEach, afterEach, jest } from "@jest/globals";

const originalFetch = global.fetch;

function buildMockRequest(
  searchParams: Record<string, string> = {},
  method: "GET" | "POST" = "GET",
  body?: unknown,
): {
  method: string;
  headers: Record<string, string>;
  body: string;
  json: () => Promise<unknown>;
  nextUrl: { searchParams: URLSearchParams };
  signal: AbortSignal;
} {
  const url = new URL("http://localhost:3000/api/knowledge-graph");
  for (const [k, v] of Object.entries(searchParams)) {
    url.searchParams.set(k, v);
  }
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : "",
    json: async () => body || {},
    nextUrl: { searchParams: url.searchParams },
    signal: new AbortController().signal,
  };
}

// Mock auth.
jest.mock("@/lib/api-helpers", () => ({
  requireAuth: async () => ({
    user: {
      userId: "test-user-1",
      email: "test@example.com",
      role: "data-scientist",
      orgId: "org-1",
    },
    response: null,
  }),
  requireRole: async (user: unknown, ..._roles: string[]) => ({
    user,
    response: null,
  }),
  internalError: (msg: string) =>
    new Response(JSON.stringify({ error: "internal_error", message: msg }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    }),
  writeAuditLog: async () => undefined,
}));

jest.mock("@/lib/zod-schemas", () => ({
  KnowledgeGraphBody: {},
  validateBody: (_schema: unknown, body: unknown) => ({
    ok: true as const,
    data: body,
    response: null,
  }),
}));

// Mock the cypher validator to always pass (we're testing the proxy,
// not the validator — the validator has its own unit tests).
jest.mock("@/app/api/knowledge-graph/cypher-validator", () => ({
  validateReadOnlyCypher: (_cypher: string) => ({ ok: true, reason: undefined }),
}));

describe("Issue 238: /api/knowledge-graph proxies to KG_SERVICE_URL", () => {
  beforeEach(() => {
    process.env.KG_SERVICE_URL = "http://localhost:8002";
  });

  afterEach(() => {
    global.fetch = originalFetch;
    delete process.env.KG_SERVICE_URL;
    jest.restoreAllMocks();
  });

  it("GET /api/knowledge-graph (no params) proxies to /kg/stats", async () => {
    const mockStats = {
      sources: [
        { name: "chembl", loaded: true, rows: 2_000_000 },
        { name: "drugbank", loaded: true, rows: 1532 },
      ],
      nodeCount: 1_500_000,
      edgeCount: 5_000_000,
      nodeTypeCounts: { Compound: 10000, Protein: 50000, Disease: 5000 },
      edgeTypeCounts: { "Compound->Protein": 200000 },
      nonCanonicalNodeCounts: { AdverseEvent: 91926 },
      source: "kg_service",
      generatedAt: "2026-07-16T10:00:00Z",
    };

    const fetchMock = jest.fn<(input: string | URL, init?: RequestInit) => Promise<Response>>();
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify(mockStats), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }) as Response,
    );
    global.fetch = fetchMock as unknown as typeof global.fetch;

    const { GET } = await import("@/app/api/knowledge-graph/route");

    const req = buildMockRequest();
    const response = await GET(req as never);
    const body = await response.json();

    // Verify the route called /kg/stats (NOT /stats — Issue 232 fix).
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("http://localhost:8002/kg/stats");

    // Verify the response shape.
    expect(response.status).toBe(200);
    expect(body.nodeCount).toBe(1_500_000);
    expect(body.edgeCount).toBe(5_000_000);
    expect(body.sources).toHaveLength(2);
    expect(body.source).toBe("kg_service");
  });

  it("GET /api/knowledge-graph?drug=X proxies to /query (POST)", async () => {
    const mockQueryResponse = {
      nodes: [
        { id: "drug:aspirin", label: "Aspirin", type: "Compound" },
        { id: "disease:headache", label: "Headache", type: "Disease" },
      ],
      edges: [
        { source: "drug:aspirin", target: "disease:headache", type: "treats" },
      ],
    };

    const fetchMock = jest.fn<(input: string | URL, init?: RequestInit) => Promise<Response>>();
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify(mockQueryResponse), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }) as Response,
    );
    global.fetch = fetchMock as unknown as typeof global.fetch;

    const { GET } = await import("@/app/api/knowledge-graph/route");

    const req = buildMockRequest({ drug: "Aspirin", limit: "50" });
    const response = await GET(req as never);
    const body = await response.json();

    // Verify the route called /query (POST) with the drug param.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("http://localhost:8002/query");
    expect(init?.method).toBe("POST");
    const forwardedBody = JSON.parse(init?.body as string);
    expect(forwardedBody.drug).toBe("Aspirin");
    expect(forwardedBody.limit).toBe(50);

    // Verify the response.
    expect(response.status).toBe(200);
    expect(body.nodes).toHaveLength(2);
    expect(body.edges).toHaveLength(1);
  });

  it("POST /api/knowledge-graph proxies to /cypher with the validated cypher", async () => {
    const mockCypherResponse = {
      records: [{ n: "Aspirin" }, { n: "Ibuprofen" }],
    };

    const fetchMock = jest.fn<(input: string | URL, init?: RequestInit) => Promise<Response>>();
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify(mockCypherResponse), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }) as Response,
    );
    global.fetch = fetchMock as unknown as typeof global.fetch;

    const { POST } = await import("@/app/api/knowledge-graph/route");

    const req = buildMockRequest(
      {},
      "POST",
      { cypher: "MATCH (n:Compound) RETURN n LIMIT 10", params: {} },
    );
    const response = await POST(req as never);
    const body = await response.json();

    // Verify the route called /cypher (POST).
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("http://localhost:8002/cypher");
    expect(init?.method).toBe("POST");
    const forwardedBody = JSON.parse(init?.body as string);
    expect(forwardedBody.cypher).toContain("MATCH (n:Compound)");

    // Verify the response.
    expect(response.status).toBe(200);
    expect(body.records).toHaveLength(2);
  });

  it("returns 503 when KG_SERVICE_URL is not set", async () => {
    delete process.env.KG_SERVICE_URL;

    const { GET } = await import("@/app/api/knowledge-graph/route");

    const req = buildMockRequest();
    const response = await GET(req as never);

    expect(response.status).toBe(503);
    const body = await response.json();
    expect(body.error).toBe("service_not_deployed");
    expect(body.reason).toContain("KG_SERVICE_URL");
  });

  it("rejects raw Cypher via GET (injection risk)", async () => {
    const { GET } = await import("@/app/api/knowledge-graph/route");

    const req = buildMockRequest({ cypher: "MATCH (n) RETURN n" });
    const response = await GET(req as never);

    expect(response.status).toBe(400);
    const body = await response.json();
    expect(body.error).toBe("bad_request");
    expect(body.message).toContain("Cypher-injection");
  });
});
