import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { badRequest, internalError } from "@/lib/api-helpers";
// BE-029 ROOT FIX (Team Member 12): Zod-validated request body.
import { validateBody, VerifyEmailBody } from "@/lib/zod-schemas";
// BE-035 ROOT FIX (Team Member 12): per-IP rate limit on email verification.
// The email verification endpoint mutates state (sets emailVerified=true)
// and is reachable by unauthenticated callers. Although the JWT signature
// is computationally infeasible to brute-force, every state-mutating
// endpoint should be rate-limited as defense in depth (OWASP
// recommendation). We reuse the existing per-IP rate-limit primitives
// from the login flow — `checkIpRateLimit` + `recordIpAttempt`.
import { checkIpRateLimit, recordIpAttempt } from "@/lib/auth/rate-limit";
// BE-020 ROOT FIX: use the SHARED `resolveJwtSecret()` from lib/auth/server.ts
// instead of re-implementing the secret-resolution logic here. The shared
// resolver:
//   - throws in production if JWT_SECRET is missing or <32 chars (fail-closed);
//   - returns the loudly-logged dev-only fallback ONLY when NODE_ENV is
//     explicitly "development" or "test" (or unset, but NOT "production");
//   - is the SINGLE source of truth used by signAccessToken / signMfaChallengeToken /
//     signMfaTicket, so email-verify tokens are signed with the SAME secret
//     as access tokens (which they must be — the verify-email flow issues
//     a 24h token at registration that THIS route verifies).
//
// The previous implementation inlined a divergent resolver that only
// checked `process.env.NODE_ENV === "production"`. If NODE_ENV was UNSET
// (which is the default in many deploy environments — e.g. `node server.js`
// without NODE_ENV=production), the verify-email route fell back to the
// publicly-known dev secret. An attacker who reads the repo could forge
// email-verification tokens for ANY userId — verifying any email without
// access to the inbox — leading to account takeover via email-verification
// bypass. Using the shared resolver closes this hole.
import { resolveJwtSecret, resolvePreviousJwtSecret } from "@/lib/auth/server";
import jwt from "jsonwebtoken";

/**
 * POST /api/auth/verify-email
 * Body: { token: string }
 *
 * FE-035 ROOT FIX: Email verification flow.
 *
 * Previously, registration set emailVerified=false but never sent a
 * verification email (nodemailer was in package.json but never imported).
 * The flag was a lie — it was always false and never checked.
 *
 * Now the flow is:
 *   1. /api/auth/register creates the user with emailVerified=false,
 *      signs a 24h verification JWT, and "sends" it via EMAIL_SERVICE_URL
 *      (or logs to stderr in dev mode).
 *   2. The user clicks the link in the email, which POSTs the token here.
 *   3. We verify the JWT signature + expiry + type==='email_verify'.
 *   4. We mark the user's emailVerified=true.
 *   5. The user can now log in via /api/auth/login (which rejects
 *      unverified accounts).
 *
 * Security properties:
 *   - The token is signed with JWT_SECRET (same as access tokens).
 *   - The token expires in 24 hours.
 *   - The token type is 'email_verify', so it CANNOT be used as an
 *     access token (verifyAccessToken checks type==='access').
 *   - We do NOT issue access+refresh tokens here — the user must log in
 *     separately after verification.
 */
export async function POST(req: NextRequest) {
  // BE-035 ROOT FIX (Team Member 12): per-IP rate limit BEFORE any work.
  // Email verification mutates state (emailVerified=true) and is
  // reachable by unauthenticated callers. The JWT signature is
  // computationally infeasible to brute-force, but defense in depth
  // requires every state-mutating endpoint to be rate-limited. We reuse
  // the existing per-IP limiter from the login flow: same window
  // (IP_WINDOW_MINUTES), same max attempts (IP_MAX_ATTEMPTS).
  const ipLock = checkIpRateLimit(req);
  if (ipLock.blocked) {
    return NextResponse.json(
      {
        error: "rate_limited",
        message: "Too many verification attempts. Please try again later.",
        retryAfterSeconds: ipLock.retryAfterSeconds,
      },
      { status: 429, headers: { "Retry-After": String(ipLock.retryAfterSeconds) } }
    );
  }
  // Record the attempt up-front so a flood of requests is capped
  // regardless of whether each one succeeds or fails. (The login route
  // uses the same pattern — FE-056.)
  recordIpAttempt(req);

  let body: { token?: string };
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON body");
  }

  // BE-029 ROOT FIX: schema-validate the body. Rejects: missing token,
  // non-string token, tokens shorter than 10 chars (clearly malformed)
  // or longer than 8192 chars (DoS guard).
  const parsed = validateBody(VerifyEmailBody, body);
  if (!parsed.ok) return parsed.response;
  const token = parsed.data.token;

  // BE-020 ROOT FIX: use the SHARED `resolveJwtSecret()` so the secret-resolution
  // logic is identical to every other JWT path. The shared resolver throws in
  // production if JWT_SECRET is missing or too short (fail-closed), and only
  // returns the dev-only fallback when NODE_ENV is explicitly NOT "production"
  // (i.e. "development", "test", or unset). The previous implementation here
  // only checked `process.env.NODE_ENV === "production"`, which meant a deploy
  // with NODE_ENV unset fell back to the publicly-known dev secret — letting
  // attackers forge email-verification tokens. Using the shared resolver also
  // supports hot-rotation via JWT_SECRET_PREVIOUS (zero-downtime rotation).
  //
  // We try both the current AND previous secrets so a token signed just
  // before a rotation remains valid during the 24h email-verify TTL.
  const secretCandidates = [resolveJwtSecret(), resolvePreviousJwtSecret()].filter(
    (s): s is string => !!s
  );

  let decoded: { sub: string; email: string; type: string } | null = null;
  for (const secret of secretCandidates) {
    try {
      decoded = jwt.verify(token, secret, {
        issuer: "drugos",
        algorithms: ["HS256"],
      }) as { sub: string; email: string; type: string };
      break;
    } catch {
      // try next candidate (rotation window)
    }
  }
  if (!decoded) {
    return NextResponse.json(
      { error: "invalid_or_expired_token", message: "The verification link is invalid or has expired. Please request a new one." },
      { status: 400 }
    );
  }

  if (decoded.type !== "email_verify" || !decoded.sub) {
    return NextResponse.json(
      { error: "invalid_token", message: "Invalid token type." },
      { status: 400 }
    );
  }

  try {
    const user = await db.user.findUnique({ where: { id: decoded.sub } });
    if (!user) {
      // BE-064 ROOT FIX: Return the SAME generic error for user-not-found
      // as for all other token validation failures. The previous code
      // returned "not_found" which leaked that the user DID exist at some
      // point (vs. an invalid token). An attacker could use this to
      // enumerate which userIds have/have not registered.
      return NextResponse.json(
        { error: "invalid_or_expired_token", message: "The verification link is invalid or has expired. Please request a new one." },
        { status: 400 }
      );
    }
    if (user.email !== decoded.email) {
      // BE-064 ROOT FIX: Return the SAME generic "invalid_or_expired_token"
      // error for email mismatch instead of the specific "email_mismatch"
      // error. The previous specific error leaked that the user exists
      // (an attacker with a stolen token for email A trying it for email B
      // would see "email_mismatch" — confirming email A's user exists).
      return NextResponse.json(
        { error: "invalid_or_expired_token", message: "The verification link is invalid or has expired. Please request a new one." },
        { status: 400 }
      );
    }
    if (user.emailVerified) {
      // Idempotent — already verified. Return success.
      return NextResponse.json({ ok: true, alreadyVerified: true });
    }

    // BE-037 ROOT FIX (Team Member 12): wrap the email update + audit log
    // in a db.$transaction so they are ATOMIC. The previous code did two
    // separate writes: first `db.user.update({ emailVerified: true })`,
    // then `writeAuditLog({ critical: true })`. If the audit write
    // failed, the code logged to stderr but returned 200 — the user
    // could log in but the audit trail had a gap (FDA 21 CFR Part 11
    // compliance issue for security events).
    //
    // Root fix: inline the audit log creation inside a Prisma
    // transaction. If the audit row cannot be created, the transaction
    // rolls back the emailVerified update — the user must re-verify.
    // This is the proper trade-off for audit completeness: a brief
    // user-facing error is preferable to a silent compliance gap.
    try {
      await db.$transaction(async (tx) => {
        await tx.user.update({
          where: { id: user.id },
          data: { emailVerified: true },
        });
        await tx.auditLog.create({
          data: {
            userId: user.id,
            actorName: user.email,
            action: "email_verified",
            resource: `user:${user.id}`,
            // Note: AuditLog.organizationId is nullable and only set for
            // org-scoped events. Email verification happens BEFORE the
            // user has an active org session (they haven't logged in
            // yet), so we leave it null here. The first login after
            // verification will populate organizationId on subsequent
            // audit entries.
            metadata: JSON.stringify({
              critical: true,
              timestamp: new Date().toISOString(),
            }),
          } as any,
        });
      });
    } catch (e: unknown) {
      // Transaction failed — emailVerified was NOT updated (rollback).
      // Return an error so the user knows to re-verify.
      const msg = e instanceof Error ? e.message : String(e);
      console.error("[verify-email] atomic update+audit transaction failed:", msg);
      return internalError(
        "Email verification could not be completed — the audit log " +
          "could not be written. Please try again; if the problem " +
          "persists, request a new verification link."
      );
    }

    return NextResponse.json({ ok: true, alreadyVerified: false });
  } catch (e) {
    console.error("Email verification failed:", e);
    return internalError("Failed to verify email.");
  }
}
