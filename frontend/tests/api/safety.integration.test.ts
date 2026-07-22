/**
 * Task 256 integration test: /api/safety/[drug] returns real openFDA data.
 *
 * ROOT FIX: this test exercises the route handler directly. We mock
 * `requireAuth` to return a valid user and mock the openFDA service
 * to return a deterministic SafetyReport. The test verifies:
 *
 *   1. Invalid drug path params (path traversal, too short) return 400.
 *   2. A valid drug name passes Zod, reaches the service, and returns
 *      a SafetyReport with `brandName`/`genericName` (NOT `drug`).
 *   3. When the service returns null (drug not in openFDA), the route
 *      returns 404 with a clear message.
 */

import { NextRequest } from "next/server";

const mockGetDrugSafetySummary = jest.fn();
jest.mock("@/lib/services/openfda", () => ({
  getDrugSafetySummary: (...args: unknown[]) => mockGetDrugSafetySummary(...args),
  isOpenfdaApiKeyConfigured: jest.fn().mockReturnValue(true),
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

import { GET } from "@/app/api/safety/[drug]/route";
import { __resetUserApiV2StateForTests } from "@/lib/auth/rate-limit";

function makeReq(drug: string): NextRequest {
  const url = `http://localhost:3000/api/safety/${encodeURIComponent(drug)}`;
  return new NextRequest(new URL(url, "http://localhost:3000"), {
    method: "GET",
  });
}

describe("Task 256 — /api/safety/[drug] integration", () => {
  beforeEach(() => {
    __resetUserApiV2StateForTests();
    mockGetDrugSafetySummary.mockReset();
  });

  it("rejects a drug name shorter than 2 chars (Zod path validation)", async () => {
    const drug = "a";
    const req = makeReq(drug);
    const res = await GET(req, { params: Promise.resolve({ drug }) });
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("bad_request");
    expect(mockGetDrugSafetySummary).not.toHaveBeenCalled();
  });

  it("rejects a drug name with path-traversal characters", async () => {
    const drug = "../../etc/passwd";
    const req = makeReq(drug);
    const res = await GET(req, { params: Promise.resolve({ drug }) });
    expect(res.status).toBe(400);
    expect(mockGetDrugSafetySummary).not.toHaveBeenCalled();
  });

  it("accepts a valid drug name and proxies to openFDA — returns brandName/genericName (NOT drug)", async () => {
    mockGetDrugSafetySummary.mockResolvedValue({
      brandName: "ASPIRIN",
      genericName: "ASPIRIN",
      totalReports: 1234,
      seriousReports: 234,
      seriousReportsWithDeath: 12,
      topReactions: [{ term: "Nausea", count: 80 }],
      disclaimer: "Adverse event data is sourced from FAERS.",
    });
    const drug = "Aspirin";
    const req = makeReq(drug);
    const res = await GET(req, { params: Promise.resolve({ drug }) });
    expect(res.status).toBe(200);
    const body = await res.json();
    // Task 251: verify the response uses brandName/genericName, NOT drug.
    expect(body.brandName).toBe("ASPIRIN");
    expect(body.genericName).toBe("ASPIRIN");
    expect(body.drug).toBeUndefined(); // the old field name must NOT be present
    expect(body.totalReports).toBe(1234);
    expect(body.seriousReports).toBe(234);
    expect(body.seriousReportsWithDeath).toBe(12);
    expect(body.topReactions.length).toBe(1);
    expect(body.topReactions[0]).toEqual({ term: "Nausea", count: 80 });
    expect(body.disclaimer).toMatch(/FAERS/);
    expect(mockGetDrugSafetySummary).toHaveBeenCalledWith("Aspirin");
  });

  it("returns 404 when openFDA has no data for the drug", async () => {
    mockGetDrugSafetySummary.mockResolvedValue(null);
    const drug = "Aspirin";
    const req = makeReq(drug);
    const res = await GET(req, { params: Promise.resolve({ drug }) });
    expect(res.status).toBe(404);
    const body = await res.json();
    expect(body.error).toBe("not_found");
  });

  it("accepts a hyphenated drug name like St-John's Wort", async () => {
    mockGetDrugSafetySummary.mockResolvedValue({
      brandName: "ST JOHNS WORT",
      genericName: "ST JOHNS WORT",
      totalReports: 5,
      seriousReports: 0,
      seriousReportsWithDeath: 0,
      topReactions: [],
      disclaimer: "x",
    });
    const drug = "St-John's Wort";
    const req = makeReq(drug);
    const res = await GET(req, { params: Promise.resolve({ drug }) });
    expect(res.status).toBe(200);
    expect(mockGetDrugSafetySummary).toHaveBeenCalledWith("St-John's Wort");
  });
});
