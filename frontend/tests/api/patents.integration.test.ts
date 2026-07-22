/**
 * Task 258 integration test: /api/patents/search returns real USPTO data.
 *
 * ROOT FIX: this test exercises the route handler directly. We mock
 * `requireAuth` and the PatentsView service. It asserts:
 *
 *   1. Missing `q` returns 400 (Zod requires it).
 *   2. `q` shorter than 2 chars returns 400.
 *   3. `q` with path-traversal characters returns 400.
 *   4. A valid `q` passes Zod and reaches the service.
 *   5. The response shape is `{ items, total, paginated, pagesFetched }`.
 */

import { NextRequest } from "next/server";

const mockSearchPatents = jest.fn();
jest.mock("@/lib/services/patentsview", () => ({
  searchPatents: (...args: unknown[]) => mockSearchPatents(...args),
}));

jest.mock("@/lib/auth/server", () => ({
  getAuthenticatedUser: jest.fn().mockResolvedValue({
    userId: "test-user-id",
    email: "test@example.com",
    role: "viewer",
    orgId: "test-org-id",
  }),
  verifyAccessToken: jest.fn(),
  authenticateApiKey: jest.fn(),
}));

import { GET } from "@/app/api/patents/search/route";
import { __resetUserApiV2StateForTests } from "@/lib/auth/rate-limit";

function makeReq(url: string): NextRequest {
  return new NextRequest(new URL(url, "http://localhost:3000"), {
    method: "GET",
  });
}

describe("Task 258 — /api/patents/search integration", () => {
  beforeEach(() => {
    __resetUserApiV2StateForTests();
    mockSearchPatents.mockReset();
    mockSearchPatents.mockResolvedValue({
      total: 1,
      patents: [
        {
          patentNumber: "US12345678",
          title: "Test Patent",
          abstract: "Test abstract",
          grantDate: "2020-01-01",
          inventors: ["John Doe"],
          assignees: ["Test Corp"],
          cpcLabels: ["A61K"],
          url: "https://patents.google.com/patent/US12345678",
        },
      ],
      paginated: false,
      pagesFetched: 1,
    });
  });

  it("rejects when q is missing (Zod requires it)", async () => {
    const req = makeReq("http://localhost:3000/api/patents/search");
    const res = await GET(req);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("bad_request");
    expect(mockSearchPatents).not.toHaveBeenCalled();
  });

  it("rejects q shorter than 2 chars", async () => {
    const req = makeReq("http://localhost:3000/api/patents/search?q=a");
    const res = await GET(req);
    expect(res.status).toBe(400);
    expect(mockSearchPatents).not.toHaveBeenCalled();
  });

  it("rejects q with path-traversal characters", async () => {
    const req = makeReq("http://localhost:3000/api/patents/search?q=../../etc/passwd");
    const res = await GET(req);
    expect(res.status).toBe(400);
    expect(mockSearchPatents).not.toHaveBeenCalled();
  });

  it("accepts a valid q and proxies to PatentsView — returns items[]", async () => {
    const req = makeReq("http://localhost:3000/api/patents/search?q=aspirin");
    const res = await GET(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.items).toBeDefined();
    expect(body.items.length).toBe(1);
    expect(body.items[0].patentNumber).toBe("US12345678");
    expect(body.total).toBe(1);
    expect(body.paginated).toBe(false);
    expect(body.pagesFetched).toBe(1);
    expect(mockSearchPatents).toHaveBeenCalledWith({ query: "aspirin", limit: 20 });
  });

  it("accepts limit=abc (schema replaces with default 20)", async () => {
    const req = makeReq("http://localhost:3000/api/patents/search?q=aspirin&limit=abc");
    const res = await GET(req);
    expect(res.status).toBe(200);
    // The schema's clampedInt transform replaces "abc" with the default 20.
    expect(mockSearchPatents).toHaveBeenCalledWith({ query: "aspirin", limit: 20 });
  });

  it("clamps limit=999 to 100", async () => {
    const req = makeReq("http://localhost:3000/api/patents/search?q=aspirin&limit=999");
    const res = await GET(req);
    expect(res.status).toBe(200);
    expect(mockSearchPatents).toHaveBeenCalledWith({ query: "aspirin", limit: 100 });
  });

  it("surfaces the 'reason' field when PatentsView API key is missing", async () => {
    mockSearchPatents.mockResolvedValue({
      total: 0,
      patents: [],
      paginated: false,
      pagesFetched: 0,
      reason: "PATENTSVIEW_API_KEY not configured.",
    });
    const req = makeReq("http://localhost:3000/api/patents/search?q=aspirin");
    const res = await GET(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.reason).toMatch(/PATENTSVIEW_API_KEY/);
    expect(body.items).toEqual([]);
  });
});
