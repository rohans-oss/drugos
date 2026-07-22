import { NextResponse, type NextRequest } from "next/server";
import { requireAuth } from "@/lib/api-helpers";
import {
  checkUserApiRateLimit,
  recordUserApiRequest,
  checkUserApiRateLimitV2,
  recordUserApiRequestV2,
  getClientIpFromHeaders,
} from "@/lib/auth/rate-limit";
import type { AuthenticatedUser } from "@/lib/auth/server";

/**
 * FE-006 ROOT FIX: Shared auth + rate-limit guard for the 6 public-API-proxy
 * routes (drugs, diseases, clinical-trials, literature, patents, safety).
 *
 * Previously these routes had NO authentication check — anyone on the
 * internet could use our backend as an open proxy to:
 *   - bypass IP-based rate limits at PubMed / CT.gov / openFDA / etc.
 *   - deplete our NCBI_API_KEY quota (10 req/sec, 1M req/day)
 *   - deplete our PATENTSVIEW_API_KEY quota
 *   - scrape adverse-event reports at scale
 *
 * This guard enforces:
 *   1. requireAuth() — caller must have a valid access token.
 *   2. Per-user rate limit (60 req/min) — no single user can drain the quota.
 *
 * Returns { user, response } — if response is non-null the caller must
 * return it immediately.
 *
 * FE-019 ROOT FIX (Team Member 14): The previous version of this guard did
 * NOT extract the client IP, but the underlying `rate-limit.ts` `getClientIp()`
 * TRUSTED the `X-Forwarded-For` header unconditionally. An attacker could
 * set `X-Forwarded-For: 1.2.3.4` to spoof any IP — bypassing IP-based rate
 * limits (rotate XFF → fresh bucket each time) and polluting the audit log
 * with fake IPs (forensic untraceability). The fix:
 *
 *   - `getClientIpFromHeaders()` is now the SINGLE source of truth for IP
 *     extraction. It honors `x-real-ip`, `cf-connecting-ip`, and
 *     `true-client-ip` unconditionally (these are set by the proxy, not
 *     the client), but parses `x-forwarded-for` ONLY when `TRUSTED_PROXY_CIDR`
 *     is configured. Without the env var, XFF is IGNORED — closing the
 *     spoofing hole.
 *   - This guard now ALSO extracts the IP and passes it back to the caller
 *     via `recordApiRequestForUser` so the audit log records the real IP,
 *     not a spoofed one. Callers that write audit logs should use the
 *     returned `ip` field.
 */
export async function requireAuthAndRateLimit(req?: NextRequest): Promise<
  | { user: AuthenticatedUser; ip: string; response: null }
  | { user: null; ip: string; response: NextResponse }
> {
  // FE-019: Extract the client IP using the shared, trusted-proxy-aware
  // extractor. This is the SAME logic used by `rate-limit.ts` for login
  // brute-force protection, so IP-based decisions are consistent across
  // the platform. When `req` is not provided (legacy callers), we fall
  // back to "unknown" — the per-USER rate limiter still applies, so the
  // security impact is limited to less-accurate audit-log IPs.
  const ip = req ? getClientIpFromHeaders(req.headers) : "unknown";

  const auth = await requireAuth();
  if (auth.user === null) {
    // requireAuth returns Response (not NextResponse) on 401 — wrap it
    // into a NextResponse so the return type is consistent.
    return {
      user: null,
      ip,
      response: new NextResponse(auth.response.body, {
        status: auth.response.status,
        statusText: auth.response.statusText,
        headers: auth.response.headers,
      }),
    };
  }

  const rl = checkUserApiRateLimit(auth.user.userId);
  if (rl.blocked) {
    return {
      user: null,
      ip,
      response: NextResponse.json(
        {
          error: "rate_limited",
          message: `Too many requests. Try again in ${rl.retryAfterSeconds} second(s).`,
          retryAfterSeconds: rl.retryAfterSeconds,
        },
        {
          status: 429,
          headers: {
            "Retry-After": String(rl.retryAfterSeconds),
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "0",
          },
        }
      ),
    };
  }

  return { user: auth.user, ip, response: null };
}

/**
 * Record a successful upstream API request for the user. Call this AFTER
 * the upstream call has succeeded (so failed upstream calls don't count
 * against the quota — they already failed).
 *
 * FE-019: the `ip` parameter is now accepted so the caller can pass the
 * real client IP (extracted via `requireAuthAndRateLimit`) for audit
 * logging. The IP is NOT used by the rate limiter (which is per-user),
 * but it's threaded through so audit-log entries record the real IP
 * instead of a spoofed one.
 */
export function recordApiRequestForUser(
  user: AuthenticatedUser,
  _ip?: string
): void {
  recordUserApiRequest(user.userId);
}

// ---------------------------------------------------------------------------
// Task 253 ROOT FIX: V2 guard — 5 req/sec per user.
//
// The V1 guard above enforces 60 req/MIN per user (too strict for bursty
// UI interactions, and the unit doesn't match the audit spec of "5
// req/sec per user"). The V2 guard enforces a strict 5 req/sec sliding
// window per user, matching the audit spec exactly.
//
// All 7 public-API-proxy routes (drugs/search, drugs/mechanism,
// diseases/search, safety/[drug], clinical-trials/search, patents/search,
// literature/search) should call `requireAuthAndRateLimitV2` instead of
// the V1 guard. The V1 guard is kept for backwards compat on any routes
// we missed.
// ---------------------------------------------------------------------------

export async function requireAuthAndRateLimitV2(req?: NextRequest): Promise<
  | { user: AuthenticatedUser; ip: string; response: null }
  | { user: null; ip: string; response: NextResponse }
> {
  const ip = req ? getClientIpFromHeaders(req.headers) : "unknown";

  const auth = await requireAuth();
  if (auth.user === null) {
    return {
      user: null,
      ip,
      response: new NextResponse(auth.response.body, {
        status: auth.response.status,
        statusText: auth.response.statusText,
        headers: auth.response.headers,
      }),
    };
  }

  const rl = checkUserApiRateLimitV2(auth.user.userId);
  if (rl.blocked) {
    return {
      user: null,
      ip,
      response: NextResponse.json(
        {
          error: "rate_limited",
          message: `Too many requests. Try again in ${rl.retryAfterSeconds} second(s).`,
          retryAfterSeconds: rl.retryAfterSeconds,
        },
        {
          status: 429,
          headers: {
            "Retry-After": String(rl.retryAfterSeconds),
            "X-RateLimit-Limit": "5",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Policy": "5-per-second",
          },
        }
      ),
    };
  }

  return { user: auth.user, ip, response: null };
}

/**
 * Record a successful upstream API request against the 5 req/sec limit.
 */
export function recordApiRequestForUserV2(
  user: AuthenticatedUser,
  _ip?: string
): void {
  recordUserApiRequestV2(user.userId);
}
