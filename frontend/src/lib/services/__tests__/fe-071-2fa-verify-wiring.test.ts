/**
 * FE-071 ROOT FIX tests: 2FA verify-side setup-token enforcement.
 *
 * These tests verify that /api/auth/2fa/verify ACTUALLY calls
 * verify2faSetupToken when the user is enrolling for the first time
 * (mfaEnabled === false). The previous "fix" added the
 * two-factor-setup-token.ts module AND a test for that module, but the
 * verify route NEVER IMPORTED OR CALLED verify2faSetupToken. This meant
 * an XSS attacker who stole the secret could call /verify themselves and
 * persist 2FA under their own control — permanent account compromise.
 *
 * These tests catch exactly that wiring gap by exercising the verify
 * route handler itself, not just the setup-token module.
 *
 * NOTE: Like fe-070-068-072-auth-hardening.test.ts, we mock @/lib/db and
 * next/headers cookies() to avoid the broken Prisma test DB infra.
 */

// --- Mocks must be set up BEFORE importing the route handlers. ---

jest.mock("next/headers", () => ({
  cookies: jest.fn(),
}));

const dbMock = {
  user: {
    findUnique: jest.fn(),
    update: jest.fn(),
    updateMany: jest.fn(),
  },
  auditLog: { create: jest.fn() },
};
jest.mock("@/lib/db", () => ({ db: dbMock }));

jest.mock("@/lib/auth/server", () => {
  const actual = jest.requireActual("@/lib/auth/server");
  return {
    ...actual,
    getAuthenticatedUser: jest.fn(),
  };
});

// Mock the CSRF guard so POST passes the CSRF check in tests.
jest.mock("@/lib/api-helpers", () => {
  const actual = jest.requireActual("@/lib/api-helpers");
  return {
    ...actual,
    requireCsrfOrSend: jest.fn(async () => ({ response: null })),
    badRequest: jest.fn((msg: string) =>
      Response.json({ error: "bad_request", message: msg }, { status: 400 })
    ),
    internalError: jest.fn((msg: string) =>
      Response.json({ error: "internal_error", message: msg }, { status: 500 })
    ),
    writeAuditLog: jest.fn(async () => {}),
  };
});

// Mock the TOTP verifier so we don't need a real TOTP code; we control
// whether verification "succeeds" per-test.
jest.mock("@/lib/auth/totp", () => ({
  verifyTotpWithReplayCheck: jest.fn(),
}));

import { getAuthenticatedUser } from "@/lib/auth/server";
import { verifyTotpWithReplayCheck } from "@/lib/auth/totp";
import { POST } from "@/app/api/auth/2fa/verify/route";
import { NextRequest } from "next/server";
import {
  issue2faSetupToken,
  __clear2faSetupTokensForTests,
} from "@/lib/auth/two-factor-setup-token";

const AUTHED_USER = {
  userId: "curuser000000000000000001",
  email: "researcher@example.com",
  role: "researcher",
};

const SECRET = "JBSWY3DPEHPK3PXPAAAAAAAA"; // base32, 20 bytes

function makeReq(body: Record<string, unknown>) {
  return new NextRequest("http://localhost/api/auth/2fa/verify", {
    method: "POST",
    body: JSON.stringify(body),
    headers: { "Content-Type": "application/json" },
  });
}

describe("FE-071: /api/auth/2fa/verify enforces one-time setup token on first-time enrollment", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (getAuthenticatedUser as jest.Mock).mockResolvedValue(AUTHED_USER);
    __clear2faSetupTokensForTests();
  });

  test("rejects first-time enrollment with NO setupToken (400 setup_token_required)", async () => {
    // User has 2FA disabled → this is a first-time enrollment.
    dbMock.user.findUnique.mockResolvedValue({
      id: AUTHED_USER.userId,
      mfaEnabled: false,
      mfaSecret: null,
      lastTotpCounter: null,
    });

    const res = await POST(makeReq({ secret: SECRET, code: "123456" }));

    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("setup_token_required");
    // CRITICAL: TOTP verification must NOT have been attempted — the
    // setup-token gate must reject before we touch TOTP.
    expect(verifyTotpWithReplayCheck).not.toHaveBeenCalled();
    // CRITICAL: the user's mfaSecret must NOT have been persisted.
    expect(dbMock.user.update).not.toHaveBeenCalled();
  });

  test("rejects first-time enrollment with a forged (never-issued) setupToken", async () => {
    dbMock.user.findUnique.mockResolvedValue({
      id: AUTHED_USER.userId,
      mfaEnabled: false,
      mfaSecret: null,
      lastTotpCounter: null,
    });

    const forgedToken = "a".repeat(64); // 32-byte hex, never issued
    const res = await POST(
      makeReq({ secret: SECRET, code: "123456", setupToken: forgedToken })
    );

    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("invalid_setup_token");
    expect(verifyTotpWithReplayCheck).not.toHaveBeenCalled();
    expect(dbMock.user.update).not.toHaveBeenCalled();
  });

  test("rejects first-time enrollment with a setupToken bound to a DIFFERENT user", async () => {
    dbMock.user.findUnique.mockResolvedValue({
      id: AUTHED_USER.userId,
      mfaEnabled: false,
      mfaSecret: null,
      lastTotpCounter: null,
    });

    // Issue a token for a DIFFERENT user.
    const OTHER_USER = "curother000000000000000099";
    const { setupToken } = issue2faSetupToken(OTHER_USER, SECRET);

    const res = await POST(
      makeReq({ secret: SECRET, code: "123456", setupToken })
    );

    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("invalid_setup_token");
    expect(verifyTotpWithReplayCheck).not.toHaveBeenCalled();
    expect(dbMock.user.update).not.toHaveBeenCalled();
  });

  test("rejects first-time enrollment with a SUBSTITUTED secret (defense in depth)", async () => {
    dbMock.user.findUnique.mockResolvedValue({
      id: AUTHED_USER.userId,
      mfaEnabled: false,
      mfaSecret: null,
      lastTotpCounter: null,
    });

    // Issue a token for SECRET, but the attacker substitutes a different secret.
    const { setupToken } = issue2faSetupToken(AUTHED_USER.userId, SECRET);
    const attackerSecret = "KRSXG5BAONUGC4TFFYYYYYYY";

    const res = await POST(
      makeReq({ secret: attackerSecret, code: "123456", setupToken })
    );

    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("invalid_setup_token");
    expect(verifyTotpWithReplayCheck).not.toHaveBeenCalled();
  });

  test("rejects replay: a setupToken cannot be used twice (one-time enforcement)", async () => {
    // First enrollment succeeds.
    dbMock.user.findUnique.mockResolvedValue({
      id: AUTHED_USER.userId,
      mfaEnabled: false,
      mfaSecret: null,
      lastTotpCounter: null,
    });
    (verifyTotpWithReplayCheck as jest.Mock).mockReturnValue({
      ok: true,
      counter: 12345,
    });
    dbMock.user.update.mockResolvedValue({});
    dbMock.user.updateMany.mockResolvedValue({ count: 1 });
    dbMock.auditLog.create.mockResolvedValue({});

    const { setupToken } = issue2faSetupToken(AUTHED_USER.userId, SECRET);
    const res1 = await POST(
      makeReq({ secret: SECRET, code: "123456", setupToken })
    );
    expect(res1.status).toBe(200);

    // Second attempt with the SAME setupToken must fail.
    dbMock.user.findUnique.mockResolvedValue({
      id: AUTHED_USER.userId,
      mfaEnabled: false, // Simulate the pre-enrollment state again
      mfaSecret: null,
      lastTotpCounter: null,
    });
    const res2 = await POST(
      makeReq({ secret: SECRET, code: "123456", setupToken })
    );
    expect(res2.status).toBe(400);
    const body2 = await res2.json();
    expect(body2.error).toBe("invalid_setup_token");
  });

  test("accepts first-time enrollment with a valid setupToken + correct TOTP code", async () => {
    dbMock.user.findUnique.mockResolvedValue({
      id: AUTHED_USER.userId,
      mfaEnabled: false,
      mfaSecret: null,
      lastTotpCounter: null,
    });
    (verifyTotpWithReplayCheck as jest.Mock).mockReturnValue({
      ok: true,
      counter: 12345,
    });
    dbMock.user.update.mockResolvedValue({});
    dbMock.user.updateMany.mockResolvedValue({ count: 1 });
    dbMock.auditLog.create.mockResolvedValue({});

    const { setupToken } = issue2faSetupToken(AUTHED_USER.userId, SECRET);
    const res = await POST(
      makeReq({ secret: SECRET, code: "123456", setupToken })
    );

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.enabled).toBe(true);
    expect(verifyTotpWithReplayCheck).toHaveBeenCalledTimes(1);
    // mfaSecret + mfaEnabled persisted.
    expect(dbMock.user.update).toHaveBeenCalledWith(
      expect.objectContaining({
        where: { id: AUTHED_USER.userId },
        data: expect.objectContaining({ mfaSecret: SECRET, mfaEnabled: true }),
      })
    );
  });

  test("does NOT require a setupToken for re-verification (mfaEnabled === true)", async () => {
    // User already has 2FA enabled — this is a re-verification, not enrollment.
    dbMock.user.findUnique.mockResolvedValue({
      id: AUTHED_USER.userId,
      mfaEnabled: true,
      mfaSecret: SECRET,
      lastTotpCounter: 10000,
    });
    (verifyTotpWithReplayCheck as jest.Mock).mockReturnValue({
      ok: true,
      counter: 10001,
    });
    dbMock.user.updateMany.mockResolvedValue({ count: 1 });

    const res = await POST(makeReq({ code: "123456" }));

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);
    // No setupToken was needed — the gate is only for first-time enrollment.
    // mfaSecret should NOT be re-persisted (only persisted when !mfaEnabled).
    expect(dbMock.user.update).not.toHaveBeenCalled();
  });
});
