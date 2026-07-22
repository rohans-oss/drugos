import { NextRequest, NextResponse } from "next/server";
import { consumeRefreshToken, setAuthCookies, clearAuthCookies } from "@/lib/auth/server";
import { writeAuditLog } from "@/lib/api-helpers";
// BE-014 + BE-076 ROOT FIX:
//
// BE-014: The previous refresh handler had NO rate limit. Each successful
// refresh ROTATES the token — `consumeRefreshToken` revokes the old token
// and issues a new one (creates a new RefreshToken DB row). An attacker
// who steals a refresh token could hammer this endpoint thousands of times
// per second, each call creating a new RefreshToken row. Effects:
//   (a) DB pollution — the RefreshToken table grows unboundedly (each row
//       has a 30-day TTL);
//   (b) the original victim's session is invalidated on the first attacker
//       call (token rotation revokes the old token), so the victim notices
//       and may change their password — but the attacker's new token is
//       valid for 30 days;
//   (c) CPU burn — each refresh calls db.user.findUnique +
//       db.refreshToken.update + db.refreshToken.create + JWT signing
//       (HS256), ~5-10ms per call, exhausts DB connection pool at
//       ~1000 req/s.
//
// BE-076: The previous handler wrote NO audit log entry for refreshes.
// Every successful and failed refresh is a security-relevant event — a
// refresh means a session was extended (or a stolen token was used to
// issue new tokens). FDA 21 CFR Part 11 requires audit trails for
// "creation, modification, and deletion of records" — a session extension
// is a modification of the user's authentication state. An attacker using
// a stolen refresh token would leave NO audit trail.
//
// ROOT FIX (both):
//   1. checkIpRateLimitDistributed BEFORE the refresh-token lookup — limits
//      unauthenticated brute-force (an attacker probing with random token
//      strings). Falls back to sync checkIpRateLimit if Redis is down
//      (defense in depth — same pattern as /api/auth/login).
//   2. checkUserRateLimitDistributed AFTER consumeRefreshToken succeeds —
//      10 refreshes per minute per user is generous (access token TTL is
//      15 min, so legitimate refresh happens once per 15 min).
//   3. writeAuditLog for every refresh — `token_refreshed` on success,
//      `token_refresh_failed` on failure. Include the user's identity
//      (from the consumed refresh token) and the IP/UA for forensics.
import {
  checkIpRateLimit,
  checkIpRateLimitDistributed,
  recordIpAttempt,
} from "@/lib/auth/rate-limit";
import { checkUserRateLimitDistributed } from "@/lib/auth/per-user-rate-limit";

// BE-014: per-user refresh limit. Access token TTL is 15 min, so a
// legitimate client refreshes at most once per 15 min. 10 refreshes/min
// is 150x the legitimate rate — generous enough to tolerate client-side
// retry storms (e.g. multiple tabs refreshing in parallel after a sleep),
// strict enough to prevent the DB-pollution / pool-exhaustion attack.
const REFRESH_USER_RATE_LIMIT = { max: 10, windowSeconds: 60 };

export async function POST(req: NextRequest) {
  // BE-014 Layer 1: per-IP rate limit (unauthenticated brute-force defense).
  // Same pattern as /api/auth/login — Redis-backed distributed limiter
  // when REDIS_URL is set, in-memory fallback otherwise.
  let ipCheck;
  try {
    ipCheck = await checkIpRateLimitDistributed(req);
  } catch (e) {
    console.error(
      "[RATE-LIMIT] distributed IP limiter failed, falling back to sync:",
      e
    );
    ipCheck = checkIpRateLimit(req);
    recordIpAttempt(req);
  }
  if (ipCheck.blocked) {
    return NextResponse.json(
      {
        error: "rate_limited",
        message: `Too many refresh attempts from this IP. Try again in ${Math.ceil(
          ipCheck.retryAfterSeconds / 60
        )} minute(s).`,
        retryAfter: ipCheck.retryAfterSeconds,
      },
      {
        status: 429,
        headers: { "Retry-After": String(ipCheck.retryAfterSeconds) },
      }
    );
  }

  // The refresh cookie is HttpOnly, so we read it via the cookies() helper.
  // We import dynamically to avoid next/headers SSR warnings outside of route
  // handlers.
  const { cookies } = await import("next/headers");
  const store = await cookies();
  const refresh = store.get("drugos_refresh")?.value;
  if (!refresh) {
    // FE-031 ROOT FIX: No refresh cookie at all — clear any stale access
    // cookie so the browser stops sending it on every subsequent request.
    await clearAuthCookies();
    // BE-076: Audit the failed refresh (no token presented). User identity
    // is unknown — log with user: null so the audit row captures the IP
    // and timestamp for forensic analysis.
    await writeAuditLog({
      user: null,
      action: "token_refresh_failed",
      resource: "auth:refresh",
      metadata: { reason: "no_refresh_token" },
    }).catch(() => {
      // Audit-log failure must not change the 401 response.
    });
    return NextResponse.json(
      { error: "no_refresh_token" },
      { status: 401 }
    );
  }
  const result = await consumeRefreshToken(refresh);
  if (!result) {
    // FE-031 ROOT FIX: The refresh token is invalid (revoked, expired, or
    // not found). Previously we returned 401 WITHOUT clearing cookies —
    // the browser kept sending the bad cookie on every subsequent request,
    // triggering a DB lookup and 401 every time. The user was effectively
    // locked out until they manually cleared cookies.
    //
    // Now we clear both cookies (access + refresh) so the client returns
    // to a clean state. The frontend's 401 handler will redirect to login.
    await clearAuthCookies();
    // BE-076: Audit the failed refresh (invalid token). User identity is
    // unknown because consumeRefreshToken returned null (we couldn't
    // decode the user from the token). The IP and UA are still recorded.
    await writeAuditLog({
      user: null,
      action: "token_refresh_failed",
      resource: "auth:refresh",
      metadata: { reason: "invalid_or_expired_token" },
    }).catch(() => {
      // Audit-log failure must not change the 401 response.
    });
    return NextResponse.json(
      { error: "invalid_refresh_token" },
      { status: 401 }
    );
  }

  // BE-014 Layer 2: per-USER rate limit (post-authentication). This
  // catches an attacker who has a VALID stolen refresh token — the IP
  // limit above doesn't apply if they're rotating IPs (botnet, Tor).
  // 10 refreshes/min is 150x the legitimate rate (access token TTL 15min
  // → 1 refresh per 15 min per session). The check runs AFTER
  // consumeRefreshToken so we have a userId to key on.
  const userRl = await checkUserRateLimitDistributed(
    result.userId,
    REFRESH_USER_RATE_LIMIT
  );
  if (userRl.blocked) {
    // Don't set the new cookies — the refresh is rejected. The old refresh
    // token was already revoked by consumeRefreshToken (rotation), so the
    // attacker loses their stolen token AND gets a 429.
    await writeAuditLog({
      user: {
        userId: result.userId,
        email: result.email,
        role: result.role,
        platformRole: result.platformRole,
      },
      action: "token_refresh_rate_limited",
      resource: "auth:refresh",
      metadata: { retryAfterSeconds: userRl.retryAfterSeconds },
    }).catch(() => {});
    return NextResponse.json(
      {
        error: "rate_limited",
        message: "Too many refresh attempts. Slow down.",
        retryAfter: userRl.retryAfterSeconds,
      },
      {
        status: 429,
        headers: { "Retry-After": String(userRl.retryAfterSeconds) },
      }
    );
  }

  await setAuthCookies(result.access, result.refresh);

  // BE-076: Audit the successful refresh. The user identity comes from
  // consumeRefreshToken's result (which decoded the user from the
  // presented refresh token). The IP and UA are auto-captured by
  // writeAuditLog via the request context (when available). Non-critical:
  // if the audit write fails, the refresh still succeeds — the user is
  // already authenticated and the action is non-destructive. We don't
  // want a DB outage to lock the user out of their session.
  await writeAuditLog({
    user: {
      userId: result.userId,
      email: result.email,
      role: result.role,
      platformRole: result.platformRole,
    },
    action: "token_refreshed",
    resource: "auth:refresh",
  }).catch(() => {
    // Best-effort — refresh must still succeed for the legitimate user.
  });

  return NextResponse.json({ ok: true });
}
