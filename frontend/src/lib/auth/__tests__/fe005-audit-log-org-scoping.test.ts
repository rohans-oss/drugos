/**
 * FE-005 ROOT FIX (v2) regression test: AuditLog organizationId auto-population.
 *
 * This test would have caught the original FE-005 bug: the previous
 * writeAuditLog required every call site to pass `organizationId` explicitly,
 * but ZERO call sites did, so every AuditLog row was written with
 * `organizationId: null`. The /api/audit-logs route then filtered by
 * `organizationId: auth.user.orgId` and got an empty result for every
 * non-owner admin.
 *
 * Root fix: writeAuditLog now auto-populates organizationId from
 * `params.user.orgId` when the caller doesn't pass it explicitly.
 *
 * This test verifies:
 *   1. When the caller passes a user with orgId but NO explicit organizationId,
 *      the AuditLog row is written with organizationId = user.orgId.
 *   2. When the caller passes an explicit organizationId, it OVERRIDES
 *      the user's orgId (defense-in-depth for cross-org admin actions).
 *   3. When the caller passes neither, organizationId is null (system events).
 */
// Mock the db module BEFORE importing api-helpers. The factory creates
// the mocks inline so we don't reference outer-scope variables (which
// breaks under ts-jest's hoisting). We retrieve the mocks via
// jest.requireMock inside the test.
jest.mock("@/lib/db", () => {
  const create = jest.fn();
  const executeRaw = jest.fn();
  return {
    __esModule: true,
    db: {
      auditLog: { create },
      $executeRaw: executeRaw,
    },
  };
});

import { writeAuditLog } from "@/lib/api-helpers";

// Helper to retrieve the mocked db module's create function.
function getCreateMock(): jest.Mock {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const mod = require("@/lib/db") as {
    db: { auditLog: { create: jest.Mock } };
  };
  return mod.db.auditLog.create;
}

beforeEach(() => {
  const create = getCreateMock();
  create.mockReset();
  create.mockResolvedValue({});
});

describe("FE-005 (v2): writeAuditLog auto-populates organizationId from user.orgId", () => {
  test("populates organizationId from user.orgId when caller omits it", async () => {
    const create = getCreateMock();
    const result = await writeAuditLog({
      user: {
        userId: "user-1",
        email: "alice@org-a.com",
        role: "admin",
        orgId: "org-a",
      },
      action: "login",
      resource: "user:user-1",
    });

    expect(result.ok).toBe(true);
    expect(create).toHaveBeenCalledTimes(1);
    const call = create.mock.calls[0][0];
    expect(call.data.organizationId).toBe("org-a");
    // Also folded into metadata for redundancy / schemas without the column.
    const meta = JSON.parse(call.data.metadata);
    expect(meta.organizationId).toBe("org-a");
  });

  test("explicit organizationId param overrides user.orgId", async () => {
    const create = getCreateMock();
    const result = await writeAuditLog({
      user: {
        userId: "user-1",
        email: "alice@org-a.com",
        role: "owner",
        orgId: "org-a",
      },
      action: "admin_action",
      resource: "user:user-2",
      // An owner performing an action on a user in a DIFFERENT org
      // should attribute the audit log to the TARGET org, not their own.
      organizationId: "org-b",
    });

    expect(result.ok).toBe(true);
    const call = create.mock.calls[0][0];
    expect(call.data.organizationId).toBe("org-b");
  });

  test("null user (anonymous/system event) → null organizationId", async () => {
    const create = getCreateMock();
    const result = await writeAuditLog({
      user: null,
      action: "failed_login_unknown_email",
      resource: "email:nobody@example.com",
    });

    expect(result.ok).toBe(true);
    const call = create.mock.calls[0][0];
    expect(call.data.organizationId).toBeUndefined();
  });

  test("user without orgId → null organizationId (not a crash)", async () => {
    const create = getCreateMock();
    const result = await writeAuditLog({
      user: {
        userId: "user-no-org",
        email: "new@noregisteredorg.com",
        role: "researcher",
        // orgId intentionally omitted — a brand-new user who hasn't
        // been added to an Organization yet.
      },
      action: "register",
      resource: "user:user-no-org",
    });

    expect(result.ok).toBe(true);
    const call = create.mock.calls[0][0];
    expect(call.data.organizationId).toBeUndefined();
  });
});
