import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import {
  verifyMfaChallengeToken,
  verifyTotp,
  signAccessToken,
  rotateRefreshToken,
  setAuthCookies,
} from "@/lib/auth/server";
import { badRequest, writeAuditLog } from "@/lib/api-helpers";

/**
 * V100 ROOT FIX (BUG #11, P0 CRITICAL): 2FA login verification endpoint.
 *
 * The login route now enforces MFA: when `user.mfaEnabled === true`, it
 * returns `{ error: "mfa_required", mfaToken: <challenge> }` instead of
 * issuing access/refresh tokens. This endpoint completes the second factor:
 *
 *   POST /api/auth/2fa/login-verify
 *   { "mfaToken": "<from login response>", "totpCode": "123456" }
 *
 * If the TOTP code is valid (±1 time-step window), the endpoint issues
 * the real access+refresh tokens and sets the auth cookies. If invalid,
 * it returns 401.
 */
interface VerifyBody {
  mfaToken?: string;
  totpCode?: string;
}

export async function POST(req: NextRequest) {
  let body: VerifyBody;
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON body");
  }
  const mfaToken = (body.mfaToken || "").trim();
  const totpCode = (body.totpCode || "").trim();
  if (!mfaToken) return badRequest("mfaToken is required");
  if (!totpCode) return badRequest("totpCode is required");

  const challenge = verifyMfaChallengeToken(mfaToken);
  if (!challenge) {
    return NextResponse.json(
      { error: "invalid_mfa_token", message: "MFA challenge token is invalid or expired." },
      { status: 401 }
    );
  }

  const user = await db.user.findUnique({ where: { id: challenge.userId } });
  if (!user || !user.mfaEnabled || !user.mfaSecret) {
    return NextResponse.json(
      { error: "mfa_not_enabled", message: "MFA is not enabled for this account." },
      { status: 400 }
    );
  }

  const valid = verifyTotp(totpCode, user.mfaSecret);
  if (!valid) {
    await writeAuditLog({
      user: { userId: user.id, email: user.email, role: user.role, orgId: undefined },
      action: "login_mfa_failed",
      resource: `user:${user.id}`,
    });
    return NextResponse.json(
      { error: "invalid_totp", message: "Invalid TOTP code." },
      { status: 401 }
    );
  }

  // TOTP valid — issue real tokens.
  const membership = await db.organizationMember.findFirst({
    where: { userId: user.id },
    orderBy: { joinedAt: "asc" },
  });
  await db.user.update({
    where: { id: user.id },
    data: { lastLoginAt: new Date() },
  });
  const tokens = await rotateRefreshToken(user.id);
  const access = signAccessToken({
    userId: user.id,
    email: user.email,
    role: user.role,
    orgId: membership?.organizationId,
  });
  await setAuthCookies(access, tokens.refresh);
  await writeAuditLog({
    user: { userId: user.id, email: user.email, role: user.role, orgId: membership?.organizationId },
    action: "login_mfa_success",
    resource: `user:${user.id}`,
  });

  return NextResponse.json({
    user: {
      id: user.id,
      email: user.email,
      name: user.name,
      role: user.role,
    },
    organizationId: membership?.organizationId,
  });
}
