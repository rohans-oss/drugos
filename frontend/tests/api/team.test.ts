/**
 * TASK-278 ROOT FIX: Team membership test.
 *
 * Verifies that:
 *   1. /api/team returns real DB rows (wired to OrganizationMember).
 *   2. The route is scoped to the user's own org.
 *   3. The response includes the expected fields (id, name, email,
 *      role, orgRole, status, lastLoginAt, joinedAt).
 *   4. The route rejects users with no active org.
 *   5. Zod validation on query params (limit, offset) works.
 */

import { describe, it, expect, beforeEach } from "@jest/globals";
import { GET as teamGet } from "@/app/api/team/route";
import { db } from "@/lib/db";
import { signAccessToken } from "@/lib/auth/server";
import { describeWithDb } from "./db-helpers";

async function buildReq(
  url: string,
  opts: { cookies?: Record<string, string> } = {}
) {
  const { NextRequest } = await import("next/server");
  const init: RequestInit & { headers: Record<string, string> } = {
    method: "GET",
    headers: {},
  };
  if (opts.cookies) {
    init.headers["cookie"] = Object.entries(opts.cookies)
      .map(([k, v]) => `${k}=${v}`)
      .join("; ");
  }
  return new NextRequest(`http://localhost:3000${url}`, init);
}

async function createUser(opts: { email: string; role?: string; name?: string }) {
  return db.user.create({
    data: {
      email: opts.email,
      passwordHash: "$2a$12$dummy.hash.for.testing.only.not.real.hash.value",
      name: opts.name || opts.email.split("@")[0],
      role: (opts.role as any) || "researcher",
      platformRole: "none",
      status: "active",
      emailVerified: true,
    },
  });
}

async function createOrgWithMembers(slug: string, members: { user: any; role: string }[]) {
  const org = await db.organization.create({
    data: { name: `Org ${slug}`, slug, plan: "team", seats: 10 },
  });
  for (const m of members) {
    await db.organizationMember.create({
      data: { userId: m.user.id, organizationId: org.id, role: m.role as any },
    });
  }
  return org;
}

describeWithDb("TASK-278: Team membership — wired to real OrganizationMember table", () => {
  beforeEach(async () => {
    const tables = ["Notification", "AuditLog", "OrganizationMember", "Organization", "User"];
    for (const t of tables) {
      try {
        // @ts-ignore
        await db[t].deleteMany({});
      } catch {
        // skip
      }
    }
  });

  it("returns real DB rows for the caller's org", async () => {
    const owner = await createUser({ email: "owner@test.com", name: "Owner Name" });
    const member1 = await createUser({ email: "member1@test.com", name: "Member One" });
    const member2 = await createUser({ email: "member2@test.com", name: "Member Two" });
    const org = await createOrgWithMembers("team-org", [
      { user: owner, role: "owner" },
      { user: member1, role: "member" },
      { user: member2, role: "admin" },
    ]);
    const token = signAccessToken({
      userId: owner.id, email: owner.email, role: owner.role, platformRole: "none", orgId: org.id,
    });
    const req = await buildReq("/api/team", { cookies: { drugos_access: token } });
    const res = await teamGet(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.total).toBe(3);
    expect(body.items.length).toBe(3);
    // Verify the fields are present.
    const sample = body.items[0];
    expect(sample).toHaveProperty("id");
    expect(sample).toHaveProperty("name");
    expect(sample).toHaveProperty("email");
    expect(sample).toHaveProperty("role");
    expect(sample).toHaveProperty("orgRole");
    expect(sample).toHaveProperty("status");
    expect(sample).toHaveProperty("joinedAt");
    // The response should include members from BOTH the User table
    // (name, email, role, status) and the OrganizationMember table
    // (orgRole, joinedAt).
    const ownerEntry = body.items.find((i: any) => i.email === "owner@test.com");
    expect(ownerEntry.orgRole).toBe("owner");
    const member1Entry = body.items.find((i: any) => i.email === "member1@test.com");
    expect(member1Entry.orgRole).toBe("member");
  });

  it("scopes results to the caller's own org (no cross-tenant leak)", async () => {
    const userA = await createUser({ email: "a@test.com" });
    const userB = await createUser({ email: "b@test.com" });
    const orgA = await createOrgWithMembers("org-a", [{ user: userA, role: "owner" }]);
    const orgB = await createOrgWithMembers("org-b", [{ user: userB, role: "owner" }]);
    const token = signAccessToken({
      userId: userA.id, email: userA.email, role: userA.role, platformRole: "none", orgId: orgA.id,
    });
    const req = await buildReq("/api/team", { cookies: { drugos_access: token } });
    const res = await teamGet(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    // Org A's owner should only see org A's members (1 — just themselves).
    expect(body.total).toBe(1);
    expect(body.items[0].email).toBe("a@test.com");
    // And NOT see org B's members.
    const leaked = body.items.find((i: any) => i.email === "b@test.com");
    expect(leaked).toBeUndefined();
  });

  it("rejects users with no active org", async () => {
    const user = await createUser({ email: "noorg@test.com" });
    const token = signAccessToken({
      userId: user.id, email: user.email, role: user.role, platformRole: "none",
      // No orgId.
    });
    const req = await buildReq("/api/team", { cookies: { drugos_access: token } });
    const res = await teamGet(req);
    expect(res.status).toBe(400);
  });

  it("rejects invalid query params via Zod validation", async () => {
    const user = await createUser({ email: "zod@test.com" });
    const org = await createOrgWithMembers("zod-org", [{ user, role: "owner" }]);
    const token = signAccessToken({
      userId: user.id, email: user.email, role: user.role, platformRole: "none", orgId: org.id,
    });
    const req = await buildReq("/api/team?limit=abc", { cookies: { drugos_access: token } });
    const res = await teamGet(req);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("bad_request");
    expect(body).toHaveProperty("issues");
  });

  it("caps limit at 100 (Zod bounds)", async () => {
    const user = await createUser({ email: "cap@test.com" });
    const org = await createOrgWithMembers("cap-org", [{ user, role: "owner" }]);
    const token = signAccessToken({
      userId: user.id, email: user.email, role: user.role, platformRole: "none", orgId: org.id,
    });
    // limit=9999 should be capped at 100 by the Zod schema.
    const req = await buildReq("/api/team?limit=9999", { cookies: { drugos_access: token } });
    const res = await teamGet(req);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("bad_request");
  });

  it("returns 401 for unauthenticated requests", async () => {
    const req = await buildReq("/api/team");
    const res = await teamGet(req);
    expect(res.status).toBe(401);
  });
});
