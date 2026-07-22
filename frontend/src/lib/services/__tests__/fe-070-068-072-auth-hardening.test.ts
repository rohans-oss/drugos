/**
 * FE-070 / FE-068 / FE-072 ROOT FIX tests: auth cookie + /api/auth/me.
 *
 * FE-070: ACCESS cookie uses SameSite=Strict; REFRESH cookie uses Lax.
 * FE-068: GET /api/auth/me returns 401 (not 404) when user is deleted.
 * FE-072: PATCH /api/auth/me rejects suspended users with 403.
 *
 * NOTE ON TEST STRATEGY:
 *   The test DB infrastructure in this repo is currently broken — the
 *   Prisma schema is set to `provider = "postgresql"` (FE-019 root fix)
 *   but `tests/api/env.ts` sets `DATABASE_URL = file:...test.db` (sqlite).
 *   Prisma rejects this combination, so EVERY DB-backed test in
 *   `src/lib/services/__tests__/` fails at module-import time. This is a
 *   pre-existing infrastructure issue documented in the PR; it is NOT
 *   caused by my FE-070/068/072 fixes.
 *
 *   To verify my fixes WITHOUT depending on the broken DB infra, these
 *   tests mock the `@/lib/db` module. The route handlers' auth/status
 *   logic is what we're verifying — the DB is just a lookup, which a mock
 *   faithfully simulates.
 */

// --- Mocks must be set up BEFORE importing the route handlers. ---

// next/headers cookies() is async — mock it.
jest.mock("next/headers", () => ({
  cookies: jest.fn(),
}));

// Mock the db module so we don't need a real Postgres/SQLite.
const dbMock = {
  user: {
    findUnique: jest.fn(),
    findMany: jest.fn(),
    update: jest.fn(),
    create: jest.fn(),
    deleteMany: jest.fn(),
  },
  organizationMember: {
    findMany: jest.fn(),
  },
  auditLog: {
    create: jest.fn(),
  },
};
jest.mock("@/lib/db", () => ({
  db: dbMock,
}));

// Mock getAuthenticatedUser so we can simulate various auth states.
jest.mock("@/lib/auth/server", () => {
  const actual = jest.requireActual("@/lib/auth/server");
  return {
    ...actual,
    getAuthenticatedUser: jest.fn(),
  };
});

import { cookies } from "next/headers";
import {
  setAuthCookies,
  ACCESS_COOKIE,
  REFRESH_COOKIE,
  getAuthenticatedUser,
} from "@/lib/auth/server";
import { GET, PATCH } from "@/app/api/auth/me/route";
import { NextRequest } from "next/server";

// --- FE-070: cookie SameSite policy ---

describe("FE-070: SameSite policy hardening", () => {
  beforeEach(() => {
    (cookies as unknown as jest.Mock).mockReset();
  });

  test("ACCESS cookie is set with SameSite=Strict (NOT Lax)", async () => {
    const setters: Record<string, any> = {};
    const store = {
      set: jest.fn((name: string, _val: string, opts: any) => {
        setters[name] = opts;
      }),
      delete: jest.fn(),
      get: jest.fn(),
    };
    (cookies as unknown as jest.Mock).mockResolvedValue(store);

    await setAuthCookies("access-token-xxx", "refresh-token-yyy");

    expect(setters[ACCESS_COOKIE]).toBeDefined();
    expect(setters[ACCESS_COOKIE].sameSite).toBe("strict");
    expect(setters[ACCESS_COOKIE].httpOnly).toBe(true);
    expect(setters[ACCESS_COOKIE].path).toBe("/");
    // CRITICAL: must NOT be lax (the pre-fix value).
    expect(setters[ACCESS_COOKIE].sameSite).not.toBe("lax");
  });

  test("REFRESH cookie is set with SameSite=Lax and path=/ (FE-050 alignment)", async () => {
    const setters: Record<string, any> = {};
    const store = {
      set: jest.fn((name: string, _val: string, opts: any) => {
        setters[name] = opts;
      }),
      delete: jest.fn(),
      get: jest.fn(),
    };
    (cookies as unknown as jest.Mock).mockResolvedValue(store);

    await setAuthCookies("access-token-xxx", "refresh-token-yyy");

    expect(setters[REFRESH_COOKIE]).toBeDefined();
    expect(setters[REFRESH_COOKIE].sameSite).toBe("lax");
    // FE-050 (merged from another agent): refresh cookie path was expanded
    // from /api/auth/refresh to / so the auto-refresh code in
    // getAuthenticatedUser() — called by every authenticated route — can
    // read it. The security trade-off is acceptable: HttpOnly + Secure +
    // SameSite=Lax covers the attack surface.
    expect(setters[REFRESH_COOKIE].path).toBe("/");
    expect(setters[REFRESH_COOKIE].httpOnly).toBe(true);
  });
});

// --- FE-068 + FE-072: /api/auth/me route behavior ---

describe("FE-068: GET /api/auth/me returns 401 (not 404) for deleted user", () => {
  beforeEach(() => {
    (getAuthenticatedUser as jest.Mock).mockReset();
    dbMock.user.findUnique.mockReset();
    dbMock.organizationMember.findMany.mockReset();
  });

  test("returns 401 when access token decodes but user no longer exists", async () => {
    (getAuthenticatedUser as jest.Mock).mockResolvedValue({
      userId: "curghost000000000000000001",
      email: "ghost@example.com",
      role: "researcher",
    });
    dbMock.user.findUnique.mockResolvedValue(null); // user deleted

    const res = await GET();
    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body.error).toBe("unauthorized");
    // CRITICAL: NOT 404 (the pre-fix value that leaked "valid token, deleted user").
    expect(res.status).not.toBe(404);
    expect(body.error).not.toBe("not_found");
  });

  test("returns 401 when no session (unauthenticated)", async () => {
    (getAuthenticatedUser as jest.Mock).mockResolvedValue(null);
    const res = await GET();
    expect(res.status).toBe(401);
  });

  test("returns 200 with user profile when user exists", async () => {
    const userRow = {
      id: "cur123",
      email: "fe068@example.com",
      name: "FE-068 Tester",
      role: "researcher",
      title: null,
      bio: null,
      status: "active",
      emailVerified: true,
      academicVerified: false,
      mfaEnabled: false,
      lastLoginAt: null,
      createdAt: new Date().toISOString(),
    };
    (getAuthenticatedUser as jest.Mock).mockResolvedValue({
      userId: userRow.id,
      email: userRow.email,
      role: "researcher",
    });
    dbMock.user.findUnique.mockResolvedValue(userRow);
    dbMock.organizationMember.findMany.mockResolvedValue([]);

    const res = await GET();
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.user.email).toBe("fe068@example.com");
  });
});

describe("FE-072: PATCH /api/auth/me blocks suspended users", () => {
  beforeEach(() => {
    (getAuthenticatedUser as jest.Mock).mockReset();
    dbMock.user.findUnique.mockReset();
    dbMock.user.update.mockReset();
    dbMock.auditLog.create.mockReset();
  });

  test("returns 403 for a suspended user (even with a valid 15-min access token)", async () => {
    const suspendedUser = { status: "suspended" };
    (getAuthenticatedUser as jest.Mock).mockResolvedValue({
      userId: "cursusp0000000000000000001",
      email: "suspended@example.com",
      role: "researcher",
    });
    dbMock.user.findUnique.mockResolvedValue(suspendedUser);

    const req = new NextRequest("http://localhost/api/auth/me", {
      method: "PATCH",
      body: JSON.stringify({ name: "Trying To Rename" }),
      headers: { "Content-Type": "application/json" },
    });
    const res = await PATCH(req);
    expect(res.status).toBe(403);
    const body = await res.json();
    expect(body.error).toBe("account_suspended");

    // CRITICAL: db.user.update must NOT have been called — the rename must
    // not persist for a suspended user.
    expect(dbMock.user.update).not.toHaveBeenCalled();
  });

  test("returns 200 for an active user updating their name", async () => {
    (getAuthenticatedUser as jest.Mock).mockResolvedValue({
      userId: "curactive00000000000000001",
      email: "active@example.com",
      role: "researcher",
    });
    dbMock.user.findUnique.mockResolvedValue({ status: "active" });
    dbMock.user.update.mockResolvedValue({
      id: "curactive00000000000000001",
      email: "active@example.com",
      name: "Renamed Active",
      role: "researcher",
      title: null,
      bio: null,
    });

    const req = new NextRequest("http://localhost/api/auth/me", {
      method: "PATCH",
      body: JSON.stringify({ name: "Renamed Active" }),
      headers: { "Content-Type": "application/json" },
    });
    const res = await PATCH(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.user.name).toBe("Renamed Active");
    expect(dbMock.user.update).toHaveBeenCalled();
  });

  test("returns 401 when access token decodes but user is deleted (FE-068 consistency on PATCH)", async () => {
    (getAuthenticatedUser as jest.Mock).mockResolvedValue({
      userId: "curdeleted00000000000000001",
      email: "ghost@example.com",
      role: "researcher",
    });
    dbMock.user.findUnique.mockResolvedValue(null);

    const req = new NextRequest("http://localhost/api/auth/me", {
      method: "PATCH",
      body: JSON.stringify({ name: "x" }),
      headers: { "Content-Type": "application/json" },
    });
    const res = await PATCH(req);
    expect(res.status).toBe(401);
    expect(dbMock.user.update).not.toHaveBeenCalled();
  });
});
