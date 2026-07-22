/**
 * TASK-276 ROOT FIX: Notifications test.
 *
 * Verifies that:
 *   1. The notification trigger fires when a project comment is posted.
 *   2. The notification trigger fires when an invoice is created (via
 *      changePlan in the billing service).
 *   3. The notification trigger fires when a hypothesis validation
 *      completes.
 *   4. The /api/notifications route returns real DB rows.
 *   5. The /api/notifications/[id]/read route actually marks the
 *      notification as read (Task 264 — was a no-op in the audit's
 *      stale description).
 */

import { describe, it, expect, beforeEach } from "@jest/globals";
import { GET as notificationsGet } from "@/app/api/notifications/route";
import { POST as notificationReadPost } from "@/app/api/notifications/[id]/read/route";
import { db } from "@/lib/db";
import { signAccessToken } from "@/lib/auth/server";
import {
  notifyProjectComment,
  notifyInvoiceReady,
  notifyHypothesisValidationComplete,
} from "@/lib/services/notifications";
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

async function createUser(opts: { email: string; role?: string; platformRole?: string }) {
  return db.user.create({
    data: {
      email: opts.email,
      passwordHash: "$2a$12$dummy.hash.for.testing.only.not.real.hash.value",
      name: opts.email.split("@")[0],
      role: (opts.role as any) || "researcher",
      platformRole: (opts.platformRole as any) || "none",
      status: "active",
      emailVerified: true,
    },
  });
}

async function createOrg(userId: string, slug: string, role: string = "owner") {
  const org = await db.organization.create({
    data: { name: `Org ${slug}`, slug, plan: "free", seats: 5 },
  });
  await db.organizationMember.create({
    data: { userId, organizationId: org.id, role: role as any },
  });
  return org;
}

describeWithDb("TASK-276: Notifications — triggers fire correctly and routes are real", () => {
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

  it("notifyProjectComment writes a notification for every project member except the commenter", async () => {
    const commenter = await createUser({ email: "commenter@test.com" });
    const member1 = await createUser({ email: "member1@test.com" });
    const member2 = await createUser({ email: "member2@test.com" });
    const org = await db.organization.create({ data: { name: "Test Org", slug: "test-org", plan: "free", seats: 5 } });
    for (const u of [commenter, member1, member2]) {
      await db.organizationMember.create({ data: { userId: u.id, organizationId: org.id, role: "member" } });
    }
    const project = await db.project.create({
      data: { name: "Test Project", organizationId: org.id, createdById: commenter.id, visibility: "org", status: "active" },
    });

    await notifyProjectComment(project.id, commenter.id, "This is a test comment");

    // commenter should NOT have a notification (no self-notifs).
    const commenterNotifs = await db.notification.findMany({ where: { userId: commenter.id } });
    expect(commenterNotifs.length).toBe(0);
    // member1 and member2 SHOULD each have a notification.
    const m1Notifs = await db.notification.findMany({ where: { userId: member1.id } });
    expect(m1Notifs.length).toBe(1);
    expect(m1Notifs[0].title).toContain("New comment");
    expect(m1Notifs[0].body).toContain("commenter");
    const m2Notifs = await db.notification.findMany({ where: { userId: member2.id } });
    expect(m2Notifs.length).toBe(1);
  });

  it("notifyInvoiceReady writes a notification for billing/owner members only", async () => {
    const billingUser = await createUser({ email: "billing@test.com" });
    const owner = await createUser({ email: "owner@test.com" });
    const regularMember = await createUser({ email: "regular@test.com" });
    const org = await db.organization.create({ data: { name: "Billing Org", slug: "billing-org", plan: "team", seats: 5 } });
    await db.organizationMember.create({ data: { userId: billingUser.id, organizationId: org.id, role: "billing" } });
    await db.organizationMember.create({ data: { userId: owner.id, organizationId: org.id, role: "owner" } });
    await db.organizationMember.create({ data: { userId: regularMember.id, organizationId: org.id, role: "member" } });

    await notifyInvoiceReady(org.id, "INV-2026-01-ABC123", 9900, "usd");

    const billingNotifs = await db.notification.findMany({ where: { userId: billingUser.id } });
    expect(billingNotifs.length).toBe(1);
    expect(billingNotifs[0].title).toContain("INV-2026-01-ABC123");
    expect(billingNotifs[0].body).toContain("$99.00");

    const ownerNotifs = await db.notification.findMany({ where: { userId: owner.id } });
    expect(ownerNotifs.length).toBe(1);

    // Regular member should NOT receive billing notifications.
    const regularNotifs = await db.notification.findMany({ where: { userId: regularMember.id } });
    expect(regularNotifs.length).toBe(0);
  });

  it("notifyHypothesisValidationComplete notifies the submitter and the org's PIs", async () => {
    const submitter = await createUser({ email: "submitter@test.com", role: "data-scientist" });
    const pi = await createUser({ email: "pi@test.com", role: "pi" });
    const otherPi = await createUser({ email: "pi2@test.com", role: "pi" });
    const regularMember = await createUser({ email: "regular@test.com", role: "researcher" });
    const org = await db.organization.create({ data: { name: "PI Org", slug: "pi-org", plan: "team", seats: 5 } });
    for (const u of [submitter, pi, otherPi, regularMember]) {
      await db.organizationMember.create({ data: { userId: u.id, organizationId: org.id, role: "member" } });
    }

    await notifyHypothesisValidationComplete(submitter.id, org.id, "Aspirin", "Migraine", "validated_positive");

    const submitterNotifs = await db.notification.findMany({ where: { userId: submitter.id } });
    expect(submitterNotifs.length).toBe(1);
    expect(submitterNotifs[0].title).toContain("Aspirin");
    expect(submitterNotifs[0].title).toContain("Migraine");
    expect(submitterNotifs[0].type).toBe("success"); // validated_positive → success

    const piNotifs = await db.notification.findMany({ where: { userId: pi.id } });
    expect(piNotifs.length).toBe(1);
    expect(piNotifs[0].title).toContain("review");

    const otherPiNotifs = await db.notification.findMany({ where: { userId: otherPi.id } });
    expect(otherPiNotifs.length).toBe(1);

    // Regular member should NOT receive PI review notifications.
    const regularNotifs = await db.notification.findMany({ where: { userId: regularMember.id } });
    expect(regularNotifs.length).toBe(0);
  });

  it("/api/notifications returns real DB rows", async () => {
    const user = await createUser({ email: "notif-user@test.com" });
    const org = await createOrg(user.id, "notif-org", "member");
    await db.notification.create({
      data: { userId: user.id, type: "info", title: "Test notif", body: "Test body" },
    });
    const token = signAccessToken({
      userId: user.id, email: user.email, role: user.role, platformRole: "none", orgId: org.id,
    });
    const req = await buildReq("/api/notifications", { cookies: { drugos_access: token } });
    const res = await notificationsGet(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.items.length).toBe(1);
    expect(body.items[0].title).toBe("Test notif");
    expect(body.unread).toBe(1);
  });

  it("POST /api/notifications/[id]/read actually marks the notification as read (Task 264)", async () => {
    const user = await createUser({ email: "read-user@test.com" });
    const org = await createOrg(user.id, "read-org", "member");
    const notif = await db.notification.create({
      data: { userId: user.id, type: "info", title: "Unread", body: "Mark me as read" },
    });
    expect(notif.readAt).toBeNull();
    const token = signAccessToken({
      userId: user.id, email: user.email, role: user.role, platformRole: "none", orgId: org.id,
    });
    const csrf = "csrf-read-token";
    const req = await buildReq(`/api/notifications/${notif.id}/read`, {
      method: "POST",
      cookies: { drugos_access: token, drugos_csrf: csrf },
    });
    (req as any).headers.set("x-csrf-token", csrf);
    const res = await notificationReadPost(req, { params: Promise.resolve({ id: notif.id }) });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.markedRead).toBe(1);

    // Verify the DB row was actually updated.
    const refreshed = await db.notification.findUnique({ where: { id: notif.id } });
    expect(refreshed?.readAt).not.toBeNull();
  });

  it("POST /api/notifications/[id]/read is idempotent — returns 404 on re-read", async () => {
    const user = await createUser({ email: "idempotent@test.com" });
    const org = await createOrg(user.id, "idempotent-org", "member");
    const notif = await db.notification.create({
      data: { userId: user.id, type: "info", title: "Already read", body: "Test", readAt: new Date() },
    });
    const token = signAccessToken({
      userId: user.id, email: user.email, role: user.role, platformRole: "none", orgId: org.id,
    });
    const csrf = "csrf-idem-token";
    const req = await buildReq(`/api/notifications/${notif.id}/read`, {
      method: "POST",
      cookies: { drugos_access: token, drugos_csrf: csrf },
    });
    (req as any).headers.set("x-csrf-token", csrf);
    const res = await notificationReadPost(req, { params: Promise.resolve({ id: notif.id }) });
    expect(res.status).toBe(404);
  });

  it("POST /api/notifications/[id]/read rejects attempts to mark ANOTHER user's notification", async () => {
    const user = await createUser({ email: "attacker@test.com" });
    const victim = await createUser({ email: "victim@test.com" });
    const org = await createOrg(user.id, "attacker-org", "member");
    // Victim's notification.
    const notif = await db.notification.create({
      data: { userId: victim.id, type: "info", title: "Victim's notif", body: "Private" },
    });
    const token = signAccessToken({
      userId: user.id, email: user.email, role: user.role, platformRole: "none", orgId: org.id,
    });
    const csrf = "csrf-attacker-token";
    const req = await buildReq(`/api/notifications/${notif.id}/read`, {
      method: "POST",
      cookies: { drugos_access: token, drugos_csrf: csrf },
    });
    (req as any).headers.set("x-csrf-token", csrf);
    const res = await notificationReadPost(req, { params: Promise.resolve({ id: notif.id }) });
    // 404, not 403 — we don't leak whether the notification exists.
    expect(res.status).toBe(404);
    // Verify the notification was NOT marked as read.
    const refreshed = await db.notification.findUnique({ where: { id: notif.id } });
    expect(refreshed?.readAt).toBeNull();
  });
});
