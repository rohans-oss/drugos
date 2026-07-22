import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import {
  verifyPassword,
  signAccessToken,
  rotateRefreshToken,
  setAuthCookies,
  signMfaChallengeToken,
} from "@/lib/auth/server";
import { badRequest, writeAuditLog } from "@/lib/api-helpers";

interface LoginBody {
  email: string;
  password: string;
}

export async function POST(req: NextRequest) {
  let body: LoginBody;
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON body");
  }
  const email = (body.email || "").trim().toLowerCase();
  const password = body.password || "";
  if (!email || !password) return badRequest("Email and password are required");

  const user = await db.user.findUnique({ where: { email } });
  if (!user) {
    return NextResponse.json(
      { error: "invalid_credentials", message: "Invalid email or password" },
      { status: 401 }
    );
  }
  if (user.status === "suspended") {
    return NextResponse.json(
      { error: "account_suspended", message: "Account suspended. Contact your administrator." },
      { status: 403 }
    );
  }
  const ok = await verifyPassword(password, user.passwordHash);
  if (!ok) {
    return NextResponse.json(
      { error: "invalid_credentials", message: "Invalid email or password" },
      { status: 401 }
    );
  }

  // V100 ROOT FIX (BUG #11, P0 CRITICAL): enforce 2FA. The previous code
  // verified the password and immediately issued access+refresh tokens
  // WITHOUT checking `user.mfaEnabled`. 2FA was purely cosmetic — account
  // takeover with password alone. Root fix: if MFA is enabled, return
  // `mfa_required` with a short-lived challenge token instead of issuing
  // real tokens. The client must call POST /api/auth/2fa/login-verify
  // with the TOTP code to complete login.
  if (user.mfaEnabled && user.mfaSecret) {
    const mfaToken = signMfaChallengeToken(user.id, user.email);
    await writeAuditLog({
      user: { userId: user.id, email: user.email, role: user.role, orgId: undefined },
      action: "login_mfa_challenge",
      resource: `user:${user.id}`,
    });
    return NextResponse.json(
      {
        error: "mfa_required",
        message: "Multi-factor authentication is required. Provide your TOTP code.",
        mfaToken,
      },
      { status: 200 } // 200 not 401 — the password WAS correct, MFA is the next step
    );
  }

  // Find the user's primary organization (first membership).
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
    action: "login",
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
