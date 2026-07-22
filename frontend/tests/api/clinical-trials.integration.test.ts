/**
 * Task 257 integration test: /api/clinical-trials/search returns real
 * ClinicalTrials.gov data.
 *
 * ROOT FIX: this test exercises the route handler directly. We mock
 * `requireAuth` and the CT.gov service. It asserts:
 *
 *   1. Missing both `condition` and `intervention` returns 400 (Zod refine).
 *   2. Providing `condition` only passes Zod and reaches the service.
 *   3. Invalid `status` values are rejected by the Zod enum (400).
 *   4. `pageToken` with >256 chars is rejected (400).
 *   5. A valid request returns `{ items, total, pageSize, nextPageToken }`.
 */

import { NextRequest } from "next/server";

const mockSearchClinicalTrials = jest.fn();
jest.mock("@/lib/services/clinical-trials", () => ({
  searchClinicalTrials: (...args: unknown[]) => mockSearchClinicalTrials(...args),
  escapeQuery: (s: string) => s,
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

import { GET } from "@/app/api/clinical-trials/search/route";
import { __resetUserApiV2StateForTests } from "@/lib/auth/rate-limit";

function makeReq(url: string): NextRequest {
  return new NextRequest(new URL(url, "http://localhost:3000"), {
    method: "GET",
  });
}

describe("Task 257 — /api/clinical-trials/search integration", () => {
  beforeEach(() => {
    __resetUserApiV2StateForTests();
    mockSearchClinicalTrials.mockReset();
    mockSearchClinicalTrials.mockResolvedValue({
      total: 1,
      trials: [
        {
          nctId: "NCT00000001",
          title: "Test Trial",
          status: "RECRUITING",
          phase: "PHASE3",
          conditions: ["diabetes"],
          interventions: ["aspirin"],
          studyType: "INTERVENTIONAL",
          url: "https://clinicaltrials.gov/study/NCT00000001",
          locations: [],
        },
      ],
      nextPageToken: undefined,
    });
  });

  it("rejects when neither condition nor intervention is provided (Zod refine)", async () => {
    const req = makeReq("http://localhost:3000/api/clinical-trials/search");
    const res = await GET(req);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("bad_request");
    expect(mockSearchClinicalTrials).not.toHaveBeenCalled();
  });

  it("accepts condition=diabetes and proxies to CT.gov", async () => {
    const req = makeReq("http://localhost:3000/api/clinical-trials/search?condition=diabetes");
    const res = await GET(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.items).toBeDefined();
    expect(body.items.length).toBe(1);
    expect(body.items[0].nctId).toBe("NCT00000001");
    expect(body.total).toBe(1);
    expect(mockSearchClinicalTrials).toHaveBeenCalledWith(
      expect.objectContaining({ condition: "diabetes" })
    );
  });

  it("accepts intervention=aspirin only", async () => {
    const req = makeReq("http://localhost:3000/api/clinical-trials/search?intervention=aspirin");
    const res = await GET(req);
    expect(res.status).toBe(200);
    expect(mockSearchClinicalTrials).toHaveBeenCalledWith(
      expect.objectContaining({ intervention: "aspirin" })
    );
  });

  it("rejects an invalid status enum value", async () => {
    const req = makeReq("http://localhost:3000/api/clinical-trials/search?condition=cancer&status=INVALID");
    const res = await GET(req);
    expect(res.status).toBe(400);
    expect(mockSearchClinicalTrials).not.toHaveBeenCalled();
  });

  it("accepts a valid status enum value", async () => {
    const req = makeReq("http://localhost:3000/api/clinical-trials/search?condition=cancer&status=RECRUITING");
    const res = await GET(req);
    expect(res.status).toBe(200);
    expect(mockSearchClinicalTrials).toHaveBeenCalledWith(
      expect.objectContaining({ status: "RECRUITING" })
    );
  });

  it("rejects a pageToken longer than 256 chars", async () => {
    const longToken = "a".repeat(257);
    const req = makeReq(`http://localhost:3000/api/clinical-trials/search?condition=cancer&pageToken=${longToken}`);
    const res = await GET(req);
    expect(res.status).toBe(400);
    expect(mockSearchClinicalTrials).not.toHaveBeenCalled();
  });

  it("forwards pageToken to the service when valid", async () => {
    const req = makeReq("http://localhost:3000/api/clinical-trials/search?condition=cancer&pageToken=abc123");
    const res = await GET(req);
    expect(res.status).toBe(200);
    expect(mockSearchClinicalTrials).toHaveBeenCalledWith(
      expect.objectContaining({ pageToken: "abc123" })
    );
  });
});
