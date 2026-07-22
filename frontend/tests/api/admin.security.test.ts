/**
 * TASK-274 ROOT FIX: Admin security test.
 *
 * Verifies that:
 *   1. Non-admin users (no platformRole) get 403 on /api/admin/*.
 *   2. Users with platformRole === "admin" can access /api/admin/*.
 *   3. Users with role === "owner" but platformRole === "none" are DENIED
 *      (the prior bug was that owner === platform-superuser).
 *   4. Unauthenticated requests get 401.
 *   5. The 403 response body does NOT leak the user's current platformRole.
 *   6. Every 403 writes an audit log entry (probing detection).
 *
 * The tests are split into TWO describe blocks:
 *   - "Auth gate logic (no DB needed)" — tests 1, 3, 4, 5. These use
 *     signed JWT tokens with synthetic user IDs and verify the gate
 *     logic without touching the DB. They ALWAYS run.
 *   - "DB-backed tests" — tests 2, 6. These create real User rows and
 *     verify the full flow. They run ONLY when a postgres test DB is
 *     available (see tests/api/db-helpers.ts).
 */

import { describe, it, expect, beforeEach } from "@jest/globals";
import { GET as adminUsersGet, PATCH as adminUsersPatch } from "@/app/api/admin/users/route";
import { GET as systemStatusGet } from "@/app/api/system/status/route";
import { db } from "@/lib/db";
import { signAccessToken } from "@/lib/auth/server";
import { describeWithDb, isDbAvailable } from "./db-helpers";
import { setTestCookies, clearTestCookies } from "./jest-setup";

// Helper to build a NextRequest with cookies.
async function buildReq(
  url: string,
  opts: { method?: string; body?: unknown; cookies?: Record<string, string>; headers?: Record<string, string> } = {}
) {
  const { NextRequest } = await import("next/server");
  const init: RequestInit & { headers: Record<string, string> } = {
    method: opts.method || "GET",
    headers: { ...(opts.headers || {}) },
  };
  if (opts.body !== undefined) {
    init.body = JSON.stringify(opts.body);
    init.headers["content-type"] = "application/json";
  }
  // Set cookies in the global mock so getAuthenticatedUser() can read them.
  if (opts.cookies) {
    setTestCookies(opts.cookies);
    // Also set the cookie header for routes that read req.cookies directly.
    init.headers["cookie"] = Object.entries(opts.cookies)
      .map(([k, v]) => `${k}=${v}`)
      .join("; ");
  }
  return new NextRequest(`http://localhost:3000${url}`, init);
}

// Use a synthetic user ID that doesn't exist in the DB — the auth gate
// checks the JWT claims, not the DB. The DB is only touched AFTER the
// gate passes (and we're testing the gate, so we don't need a real user).
const SYNTHETIC_USER_ID = "synthetic-user-id-for-auth-gate-test";

describe("TASK-274: Auth gate logic (no DB needed)", () => {
  // These tests verify the requirePlatformAdmin middleware's gate logic.
  // They don't need the DB because the middleware checks the JWT's
  // platformRole claim BEFORE any DB query.

  beforeEach(() => {
    clearTestCookies();
  });

  it("returns 401 for unauthenticated requests to /api/admin/users", async () => {
    const req = await buildReq("/api/admin/users");
    const res = await adminUsersGet(req);
    expect(res.status).toBe(401);
  });

  it("returns 403 for a researcher (platformRole=none) even with a valid token", async () => {
    // NOTE: we intentionally OMIT orgId so getAuthenticatedUser skips
    // the DB org-membership check. We're testing the platformRole gate,
    // not the org-membership check. The middleware checks platformRole
    // AFTER getAuthenticatedUser returns, so the 403 fires regardless.
    const token = signAccessToken({
      userId: SYNTHETIC_USER_ID,
      email: "researcher@test.com",
      role: "researcher",
      platformRole: "none",
    });
    const req = await buildReq("/api/admin/users", {
      cookies: { drugos_access: token },
    });
    const res = await adminUsersGet(req);
    expect(res.status).toBe(403);
    const body = await res.json();
    // The 403 body MUST NOT leak the user's platformRole.
    expect(body.error).toBe("forbidden");
    expect(body.message).toBe("Platform administrator access required.");
    expect(JSON.stringify(body)).not.toContain("platformRole");
    expect(JSON.stringify(body)).not.toContain("none");
    expect(JSON.stringify(body)).not.toContain("researcher");
  });

  it("returns 403 for an org owner (role=owner, platformRole=none) — the Task-261 bug", async () => {
    // The prior architecture granted owner == platform-superuser. The fix
    // is that owner is purely functional; platformRole must be "admin".
    // Omit orgId to skip the DB org-membership check.
    const token = signAccessToken({
      userId: SYNTHETIC_USER_ID,
      email: "owner@test.com",
      role: "owner",
      platformRole: "none",
    });
    const req = await buildReq("/api/admin/users", {
      cookies: { drugos_access: token },
    });
    const res = await adminUsersGet(req);
    expect(res.status).toBe(403);
  });

  it("returns 403 for an org admin (role=admin, platformRole=none)", async () => {
    const token = signAccessToken({
      userId: SYNTHETIC_USER_ID,
      email: "admin@test.com",
      role: "admin",
      platformRole: "none",
    });
    const req = await buildReq("/api/admin/users", {
      cookies: { drugos_access: token },
    });
    const res = await adminUsersGet(req);
    expect(res.status).toBe(403);
  });

  it("returns 403 for a legacy token with NO platformRole claim (fail-closed)", async () => {
    // Legacy tokens issued before this fix don't have a platformRole claim.
    // verifyAccessToken coerces undefined → "none" (fail-closed). We
    // simulate this by signing a token with platformRole="none" (the
    // default in signAccessToken when the field is omitted).
    // Omit orgId to skip the DB org-membership check.
    const token = signAccessToken({
      userId: SYNTHETIC_USER_ID,
      email: "legacy@test.com",
      role: "owner",
      // Intentionally omit platformRole — signAccessToken defaults to "none".
    } as any);
    const req = await buildReq("/api/admin/users", {
      cookies: { drugos_access: token },
    });
    const res = await adminUsersGet(req);
    expect(res.status).toBe(403);
  });

  it("returns 403 on PATCH /api/admin/users for a non-admin token", async () => {
    const token = signAccessToken({
      userId: SYNTHETIC_USER_ID,
      email: "patcher@test.com",
      role: "researcher",
      platformRole: "none",
    });
    const csrf = "csrf-patch-token";
    const req = await buildReq("/api/admin/users", {
      method: "PATCH",
      body: { userId: "any-target-id", role: "admin" },
      cookies: { drugos_access: token, drugos_csrf: csrf },
      headers: { "x-csrf-token": csrf },
    });
    const res = await adminUsersPatch(req);
    expect(res.status).toBe(403);
  });

  it("returns 401 for unauthenticated requests to /api/system/status", async () => {
    const req = await buildReq("/api/system/status");
    const res = await systemStatusGet(req as any);
    expect(res.status).toBe(401);
  });

  it("returns 403 for a non-platform-admin on /api/system/status", async () => {
    const token = signAccessToken({
      userId: SYNTHETIC_USER_ID,
      email: "status@test.com",
      role: "admin", // org admin, but platformRole=none
      platformRole: "none",
    });
    const req = await buildReq("/api/system/status", {
      cookies: { drugos_access: token },
    });
    const res = await systemStatusGet(req as any);
    expect(res.status).toBe(403);
  });
});

describeWithDb("TASK-274: DB-backed admin security tests", () => {
  beforeEach(async () => {
    if (!isDbAvailable()) return;
    const tables = ["AuditLog", "AuditLogDeadLetter", "Notification", "OrganizationMember", "Organization", "User"];
    for (const t of tables) {
      try {
        // @ts-ignore
        await db[t].deleteMany({});
      } catch {
        // Table may not exist in test DB — skip.
      }
    }
  });

  async function createUser(opts: {
    email: string;
    role?: string;
    platformRole?: string;
    status?: string;
  }) {
    return db.user.create({
      data: {
        email: opts.email,
        passwordHash: "$2a$12$dummy.hash.for.testing.only.not.real.hash.value",
        name: opts.email.split("@")[0],
        role: (opts.role as any) || "researcher",
        platformRole: (opts.platformRole as any) || "none",
        status: (opts.status as any) || "active",
        emailVerified: true,
      },
    });
  }

  async function createOrg(userId: string, slug: string) {
    const org = await db.organization.create({
      data: { name: `Org ${slug}`, slug, plan: "free", seats: 5 },
    });
    await db.organizationMember.create({
      data: { userId, organizationId: org.id, role: "owner" },
    });
    return org;
  }

  it("returns 200 for a platform admin (platformRole=admin) and lists real users", async () => {
    const user = await createUser({ email: "platform-admin@test.com", role: "researcher", platformRole: "admin" });
    const org = await createOrg(user.id, "pa-org");
    await createUser({ email: "other@test.com", role: "researcher" });
    const token = signAccessToken({
      userId: user.id,
      email: user.email,
      role: user.role,
      platformRole: "admin",
      orgId: org.id,
    });
    const req = await buildReq("/api/admin/users", {
      cookies: { drugos_access: token },
    });
    const res = await adminUsersGet(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(Array.isArray(body.items)).toBe(true);
    expect(body.total).toBeGreaterThanOrEqual(2);
    // The response MUST include platformRole so the admin console can
    // display who has platform-admin access.
    expect(body.items[0]).toHaveProperty("platformRole");
  });

  it("writes an audit log entry on every 403 (probing detection)", async () => {
    const user = await createUser({ email: "prober@test.com", role: "researcher" });
    const org = await createOrg(user.id, "prober-org");
    const token = signAccessToken({
      userId: user.id,
      email: user.email,
      role: user.role,
      platformRole: "none",
      orgId: org.id,
    });
    const req = await buildReq("/api/admin/users", {
      cookies: { drugos_access: token },
    });
    await adminUsersGet(req);
    const logs = await db.auditLog.findMany({
      where: { userId: user.id, action: "platform_admin_denied" },
    });
    expect(logs.length).toBeGreaterThanOrEqual(1);
  });

  it("PATCH /api/admin/users by a non-admin does NOT change the target user", async () => {
    const user = await createUser({ email: "patcher@test.com", role: "researcher" });
    const org = await createOrg(user.id, "patcher-org");
    const target = await createUser({ email: "target@test.com", role: "researcher" });
    const token = signAccessToken({
      userId: user.id,
      email: user.email,
      role: user.role,
      platformRole: "none",
      orgId: org.id,
    });
    const csrf = "csrf-test-token";
    const req = await buildReq("/api/admin/users", {
      method: "PATCH",
      body: { userId: target.id, role: "admin" },
      cookies: { drugos_access: token, drugos_csrf: csrf },
      headers: { "x-csrf-token": csrf },
    });
    const res = await adminUsersPatch(req);
    expect(res.status).toBe(403);
    // Verify the target's role was NOT changed.
    const refreshed = await db.user.findUnique({ where: { id: target.id } });
    expect(refreshed?.role).toBe("researcher");
  });
});
