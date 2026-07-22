/**
 * Integration tests for /api/dataset (Issue 239).
 *
 * Verifies that the /api/dataset route reads from Phase 1 (via
 * PHASE1_SERVICE_URL/stats) — NOT from Phase 2's bridge summary.
 *
 * Issue 226 ROOT FIX: the previous version read
 * ../phase2/data/checkpoints/step_01.json (a Phase 2 artifact). The
 * new version proxies to PHASE1_SERVICE_URL/stats (the Phase 1
 * service's own /stats endpoint).
 */

import { describe, it, expect, beforeEach, afterEach, jest } from "@jest/globals";

const originalFetch = global.fetch;

function buildMockRequest(
  searchParams: Record<string, string> = {},
): {
  method: string;
  headers: Record<string, string>;
  body: string;
  json: () => Promise<unknown>;
  nextUrl: { searchParams: URLSearchParams };
  signal: AbortSignal;
} {
  const url = new URL("http://localhost:3000/api/dataset");
  for (const [k, v] of Object.entries(searchParams)) {
    url.searchParams.set(k, v);
  }
  return {
    method: "GET",
    headers: { "Content-Type": "application/json" },
    body: "",
    json: async () => ({}),
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

describe("Issue 239: /api/dataset reads from Phase 1 (PHASE1_SERVICE_URL/stats)", () => {
  beforeEach(() => {
    // Set the canonical env var (NOT the legacy DATASET_SERVICE_URL).
    process.env.PHASE1_SERVICE_URL = "http://localhost:8001";
    delete process.env.DATASET_SERVICE_URL;
  });

  afterEach(() => {
    global.fetch = originalFetch;
    delete process.env.PHASE1_SERVICE_URL;
    delete process.env.DATASET_SERVICE_URL;
    jest.restoreAllMocks();
  });

  it("GET /api/dataset proxies to PHASE1_SERVICE_URL/stats", async () => {
    const mockStats = {
      sources: [
        { name: "chembl", loaded: true, rowsLoaded: 2_000_000 },
        { name: "drugbank", loaded: true, rowsLoaded: 1532 },
        { name: "uniprot", loaded: true, rowsLoaded: 560_000 },
        { name: "string", loaded: false },
        { name: "disgenet", loaded: true, rowsLoaded: 90_000 },
        { name: "omim", loaded: true, rowsLoaded: 25_000 },
        { name: "pubchem", loaded: true, rowsLoaded: 110_000_000 },
      ],
      nodesLoaded: 561_532,
      edgesLoaded: 2_000_000,
      edgeTypesPresent: ["Compound->Protein", "Protein->Protein", "Gene->Disease"],
      pipelineVersion: "phase1-service-v1",
      schemaVersion: "1.0",
      bridgeVersion: null,
      backend: "phase1_service",
      warnings: [],
      errors: [],
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

    const { GET } = await import("@/app/api/dataset/route");

    const req = buildMockRequest();
    const response = await GET(req as never);
    const body = await response.json();

    // Verify the route called /stats on the Phase 1 service.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("http://localhost:8001/stats");

    // Verify the response shape.
    expect(response.status).toBe(200);
    expect(body.sources).toHaveLength(7);
    expect(body.sources[0].name).toBe("chembl");
    expect(body.sources[0].rowsLoaded).toBe(2_000_000);
    expect(body.nodesLoaded).toBe(561_532);
    expect(body.edgeTypesPresent).toContain("Compound->Protein");
    expect(body.backend).toBe("phase1_service");
  });

  it("honors DATASET_SERVICE_URL as a legacy alias", async () => {
    // Unset the canonical var, set the legacy alias.
    delete process.env.PHASE1_SERVICE_URL;
    process.env.DATASET_SERVICE_URL = "http://localhost:9001";

    const mockStats = {
      sources: [{ name: "chembl", loaded: true, rowsLoaded: 100 }],
      nodesLoaded: 100,
      edgesLoaded: 50,
      edgeTypesPresent: [],
      warnings: [],
      errors: [],
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

    const { GET } = await import("@/app/api/dataset/route");

    const req = buildMockRequest();
    const response = await GET(req as never);

    // Verify the route used the legacy alias URL.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("http://localhost:9001/stats");
    expect(response.status).toBe(200);
  });

  it("PHASE1_SERVICE_URL wins over DATASET_SERVICE_URL when both are set", async () => {
    process.env.PHASE1_SERVICE_URL = "http://localhost:8001";
    process.env.DATASET_SERVICE_URL = "http://localhost:9001";

    const mockStats = {
      sources: [],
      nodesLoaded: 0,
      edgesLoaded: 0,
      edgeTypesPresent: [],
      warnings: [],
      errors: [],
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

    const { GET } = await import("@/app/api/dataset/route");

    const req = buildMockRequest();
    await GET(req as never);

    // Verify the canonical URL was used (8001, not 9001).
    const [url] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("http://localhost:8001/stats");
  });

  it("returns status:no_data when neither env var is set", async () => {
    delete process.env.PHASE1_SERVICE_URL;
    delete process.env.DATASET_SERVICE_URL;

    const { GET } = await import("@/app/api/dataset/route");

    const req = buildMockRequest();
    const response = await GET(req as never);
    const body = await response.json();

    // The route returns 200 with status:"no_data" (not 503) — the request
    // succeeded; the service is just not configured.
    expect(response.status).toBe(200);
    expect(body.status).toBe("no_data");
    expect(body.source).toBe("none");
    expect(body.sources).toEqual([]);
    expect(body.note).toContain("PHASE1_SERVICE_URL is not set");
  });

  it("filters by source when ?source=chembl is provided", async () => {
    const mockStats = {
      sources: [
        { name: "chembl", loaded: true, rowsLoaded: 2_000_000 },
        { name: "drugbank", loaded: true, rowsLoaded: 1532 },
      ],
      nodesLoaded: 100,
      edgesLoaded: 50,
      edgeTypesPresent: [],
      warnings: [],
      errors: [],
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

    const { GET } = await import("@/app/api/dataset/route");

    const req = buildMockRequest({ source: "chembl" });
    const response = await GET(req as never);
    const body = await response.json();

    // Verify the route filtered the sources list.
    expect(response.status).toBe(200);
    expect(body.sources).toHaveLength(1);
    expect(body.sources[0].name).toBe("chembl");
  });
});
