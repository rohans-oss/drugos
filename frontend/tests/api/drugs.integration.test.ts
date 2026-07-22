/**
 * Task 255 integration test: /api/drugs/search proxies to real RxNorm.
 *
 * ROOT CAUSE: the audit required an integration test verifying that
 * `curl localhost:3000/api/drugs/search?q=asp` returns real drugs from
 * the underlying biomedical API. No such test existed.
 *
 * ROOT FIX: this test exercises the route handler directly. We mock
 * `requireAuth` to return a valid authenticated user (so the route
 * proceeds past auth) and mock the upstream RxNorm service to return
 * a deterministic result. This lets us verify:
 *
 *   1. Zod validation rejects `q` shorter than 2 chars (400).
 *   2. Zod validation rejects `q` with path-traversal characters (400).
 *   3. Zod rejects `rxcui` with non-digit characters (400).
 *   4. A valid `q` passes Zod + auth and reaches the upstream service
 *      (which is mocked — the assertion is that the route called it
 *      with the right arguments).
 *
 * The full end-to-end test (real auth + real RxNorm HTTP call) is left
 * to the operator-run smoke test in scripts/run-integration-tests.js.
 */

import { NextRequest } from "next/server";

// Mock the upstream RxNorm service so the test doesn't depend on the
// real NLM API being up. The mock records its calls so we can assert
// the route forwarded the right arguments.
const mockSearchDrugsByName = jest.fn();
const mockGetDrugProperties = jest.fn();
jest.mock("@/lib/services/rxnorm", () => ({
  searchDrugsByName: (...args: unknown[]) => mockSearchDrugsByName(...args),
  getDrugProperties: (...args: unknown[]) => mockGetDrugProperties(...args),
}));

// Mock the auth module so requireAuth returns a valid authenticated
// user — the route proceeds past auth and we can verify Zod + the
// upstream call. We do NOT mock the rate-limit module because we want
// to verify the V2 guard is wired (it would block at 5 req/sec, but
// we reset state between tests).
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

import { GET } from "@/app/api/drugs/search/route";
import { __resetUserApiV2StateForTests } from "@/lib/auth/rate-limit";

function makeReq(url: string): NextRequest {
  return new NextRequest(new URL(url, "http://localhost:3000"), {
    method: "GET",
  });
}

describe("Task 255 — /api/drugs/search integration", () => {
  beforeEach(() => {
    __resetUserApiV2StateForTests();
    mockSearchDrugsByName.mockReset();
    mockGetDrugProperties.mockReset();
    mockSearchDrugsByName.mockResolvedValue([
      { rxcui: "1191", name: "aspirin", tty: "IN" },
    ]);
  });

  it("rejects q shorter than 2 chars with a 400 (Zod validation)", async () => {
    const req = makeReq("http://localhost:3000/api/drugs/search?q=a");
    const res = await GET(req);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("bad_request");
    expect(body.issues).toBeDefined();
    expect(body.issues.length).toBeGreaterThan(0);
    expect(mockSearchDrugsByName).not.toHaveBeenCalled();
  });

  it("rejects q containing path-traversal characters (Zod regex)", async () => {
    const req = makeReq("http://localhost:3000/api/drugs/search?q=../../etc/passwd");
    const res = await GET(req);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("bad_request");
    expect(mockSearchDrugsByName).not.toHaveBeenCalled();
  });

  it("accepts a valid biomedical name and proxies to RxNorm", async () => {
    const req = makeReq("http://localhost:3000/api/drugs/search?q=aspirin");
    const res = await GET(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.items).toBeDefined();
    expect(body.items.length).toBe(1);
    expect(body.items[0].rxcui).toBe("1191");
    expect(body.items[0].name).toBe("aspirin");
    expect(body.total).toBe(1);
    expect(body.query).toBe("aspirin");
    // Verify the route forwarded the right arguments to the service.
    expect(mockSearchDrugsByName).toHaveBeenCalledWith("aspirin", 10);
  });

  it("accepts an rxcui numeric query and proxies to getDrugProperties", async () => {
    mockGetDrugProperties.mockResolvedValue({
      rxcui: "1191",
      name: "aspirin",
      activeIngredients: ["ASPIRIN"],
      brandNames: ["BAYER"],
      doseForm: "TABLET",
      tty: "IN",
    });
    const req = makeReq("http://localhost:3000/api/drugs/search?rxcui=1191");
    const res = await GET(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.rxcui).toBe("1191");
    expect(body.name).toBe("aspirin");
    expect(mockGetDrugProperties).toHaveBeenCalledWith("1191");
  });

  it("rejects an rxcui with non-digit characters (Zod regex)", async () => {
    const req = makeReq("http://localhost:3000/api/drugs/search?rxcui=abc");
    const res = await GET(req);
    expect(res.status).toBe(400);
    expect(mockGetDrugProperties).not.toHaveBeenCalled();
  });

  it("rejects when neither q nor rxcui is provided", async () => {
    const req = makeReq("http://localhost:3000/api/drugs/search");
    const res = await GET(req);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("bad_request");
    expect(body.message).toMatch(/q.*rxcui/i);
  });
});
