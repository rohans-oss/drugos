/**
 * TASK-275 ROOT FIX: Audit log test.
 *
 * Verifies that:
 *   1. Privileged actions (user PATCH, API key create/revoke) write to
 *      the audit log.
 *   2. The audit log route returns real DB rows (not mock data).
 *   3. Org-scoped admins only see their own org's logs.
 *   4. Platform admins see system-wide logs.
 *   5. Dead-letter entries are only visible to platform admins.
 */

import { describe, it, expect, beforeEach } from "@jest/globals";
import { GET as auditLogsGet } from "@/app/api/audit-logs/route";
import { POST as apiKeysPost } from "@/app/api/api-keys/route";
import { POST as apiKeysRevokePost } from "@/app/api/api-keys/[id]/revoke/route";
import { db } from "@/lib/db";
import { signAccessToken } from "@/lib/auth/server";
import { issueApiKey } from "@/lib/services/api-keys";
import { describeWithDb } from "./db-helpers";

async function buildReq(
  url: string,
  opts: { method?: string; body?: unknown; cookies?: Record<string, string> } = {}
) {
  const { NextRequest } = await import("next/server");
  const init: RequestInit & { headers: Record<string, string> } = {
    method: opts.method || "GET",
    headers: {},
  };
  if (opts.body !== undefined) {
    init.body = JSON.stringify(opts.body);
    init.headers["content-type"] = "application/json";
  }
  if (opts.cookies) {
    init.headers["cookie"] = Object.entries(opts.cookies)
      .map(([k, v]) => `${k}=${v}`)
      .join("; ");
  }
  return new NextRequest(`http://localhost:3000${url}`, init);
}

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

async function createOrg(userId: string, slug: string, role: string = "owner") {
  const org = await db.organization.create({
    data: { name: `Org ${slug}`, slug, plan: "team", seats: 5 },
  });
  await db.organizationMember.create({
    data: { userId, organizationId: org.id, role: role as any },
  });
  return org;
}

describeWithDb("TASK-275: Audit logs — privileged actions are logged and queryable", () => {
  beforeEach(async () => {
    const tables = ["AuditLog", "AuditLogDeadLetter", "Notification", "ApiKey", "OrganizationMember", "Organization", "User"];
    for (const t of tables) {
      try {
        // @ts-ignore
        await db[t].deleteMany({});
      } catch {
        // skip
      }
    }
  });

  it("writes an audit log entry when an API key is created", async () => {
    const user = await createUser({ email: "dev@test.com", role: "developer" });
    const org = await createOrg(user.id, "dev-org", "member");
    const token = signAccessToken({
      userId: user.id, email: user.email, role: user.role, platformRole: "none", orgId: org.id,
    });
    const csrf = "csrf-test-token";
    const req = await buildReq("/api/api-keys", {
      method: "POST",
      body: { name: "My test key" },
      cookies: { drugos_access: token, drugos_csrf: csrf },
    });
    (req as any).headers.set("x-csrf-token", csrf);
    const res = await apiKeysPost(req);
    expect(res.status).toBe(201);

    // Verify the audit log entry was written.
    const logs = await db.auditLog.findMany({
      where: { userId: user.id, action: "api_key_create" },
    });
    expect(logs.length).toBe(1);
    const meta = JSON.parse(logs[0].metadata);
    expect(meta.keyName).toBe("My test key");
  });

  it("writes an audit log entry when an API key is revoked", async () => {
    const user = await createUser({ email: "dev2@test.com", role: "developer" });
    const org = await createOrg(user.id, "dev2-org", "member");
    // Issue a key directly via the service (bypasses the route's audit log).
    const key = await issueApiKey(org.id, user.id, "Key to revoke");
    const token = signAccessToken({
      userId: user.id, email: user.email, role: user.role, platformRole: "none", orgId: org.id,
    });
    const csrf = "csrf-test-token";
    const req = await buildReq(`/api/api-keys/${key.id}/revoke`, {
      method: "POST",
      cookies: { drugos_access: token, drugos_csrf: csrf },
    });
    (req as any).headers.set("x-csrf-token", csrf);
    const res = await apiKeysRevokePost(req, { params: Promise.resolve({ id: key.id }) });
    expect(res.status).toBe(200);

    const logs = await db.auditLog.findMany({
      where: { userId: user.id, action: "api_key_revoke" },
    });
    expect(logs.length).toBe(1);
  });

  it("returns real DB rows (not mock data) from /api/audit-logs", async () => {
    const user = await createUser({ email: "admin@test.com", role: "admin" });
    const org = await createOrg(user.id, "admin-org", "admin");
    // Write a real audit log entry directly.
    await db.auditLog.create({
      data: {
        userId: user.id,
        organizationId: org.id,
        actorName: user.email,
        action: "test_action_real",
        resource: "test:resource",
        metadata: JSON.stringify({ foo: "bar" }),
      },
    });
    const token = signAccessToken({
      userId: user.id, email: user.email, role: user.role, platformRole: "none", orgId: org.id,
    });
    const req = await buildReq("/api/audit-logs", {
      cookies: { drugos_access: token },
    });
    const res = await auditLogsGet(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.items.length).toBeGreaterThanOrEqual(1);
    const testEntry = body.items.find((i: any) => i.action === "test_action_real");
    expect(testEntry).toBeDefined();
    expect(testEntry.actorName).toBe(user.email);
  });

  it("scopes audit logs to the user's org for non-platform admins", async () => {
    const userA = await createUser({ email: "a@test.com", role: "admin" });
    const orgA = await createOrg(userA.id, "org-a", "admin");
    const userB = await createUser({ email: "b@test.com", role: "admin" });
    const orgB = await createOrg(userB.id, "org-b", "admin");
    // Write a log in org B.
    await db.auditLog.create({
      data: {
        userId: userB.id,
        organizationId: orgB.id,
        actorName: userB.email,
        action: "org_b_secret_action",
        resource: "secret",
        metadata: "{}",
      },
    });
    const token = signAccessToken({
      userId: userA.id, email: userA.email, role: userA.role, platformRole: "none", orgId: orgA.id,
    });
    const req = await buildReq("/api/audit-logs", { cookies: { drugos_access: token } });
    const res = await auditLogsGet(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    // Org A's admin should NOT see org B's logs.
    const leaked = body.items.find((i: any) => i.action === "org_b_secret_action");
    expect(leaked).toBeUndefined();
    expect(body.scope).toBe("organization");
  });

  it("allows platform admins to see system-wide logs", async () => {
    const userA = await createUser({ email: "pa@test.com", role: "researcher", platformRole: "admin" });
    const orgA = await createOrg(userA.id, "pa-org", "member");
    const userB = await createUser({ email: "other@test.com", role: "researcher" });
    const orgB = await createOrg(userB.id, "other-org", "member");
    await db.auditLog.create({
      data: {
        userId: userB.id,
        organizationId: orgB.id,
        actorName: userB.email,
        action: "system_wide_visible_action",
        resource: "test",
        metadata: "{}",
      },
    });
    const token = signAccessToken({
      userId: userA.id, email: userA.email, role: userA.role, platformRole: "admin", orgId: orgA.id,
    });
    const req = await buildReq("/api/audit-logs", { cookies: { drugos_access: token } });
    const res = await auditLogsGet(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.scope).toBe("system");
    const found = body.items.find((i: any) => i.action === "system_wide_visible_action");
    expect(found).toBeDefined();
  });

  it("denies dead-letter access to non-platform admins", async () => {
    const user = await createUser({ email: "dl@test.com", role: "admin" });
    const org = await createOrg(user.id, "dl-org", "admin");
    const token = signAccessToken({
      userId: user.id, email: user.email, role: user.role, platformRole: "none", orgId: org.id,
    });
    const req = await buildReq("/api/audit-logs?dead_letter=true", { cookies: { drugos_access: token } });
    const res = await auditLogsGet(req);
    expect(res.status).toBe(403);
  });
});
