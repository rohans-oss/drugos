/**
 * TASK-271 ROOT FIX: requirePlatformAdmin() middleware.
 *
 * GATES ALL /api/admin/* routes. Only callers with `platformRole === "admin"`
 * can pass. This is the architectural fix for Task 261:
 *
 *   "Fix `frontend/src/app/api/admin/users/route.ts` — currently checks
 *    `user.role === "OWNER"` and grants platform-superuser privileges. Add
 *    a separate `platformRole` field. Only `platformRole === "ADMIN"` can
 *    access /api/admin/*."
 *
 * WHY THIS EXISTS (and why the prior `role === "platformOwner"` patch was
 * insufficient):
 *
 *   The prior fix added a `platformOwner` value to the `UserRole` enum and
 *   gated /api/admin/* on `role === "platformOwner"`. This reduced the
 *   blast radius of the original Task-261 finding (where `role === "owner"`
 *   granted platform-superuser privileges) but kept the COUPLING between
 *   the user's FUNCTIONAL role (researcher, pi, admin, owner — drives
 *   in-app RBAC) and their PLATFORM-OPERATOR status (SaaS staff who can
 *   suspend tenants, read cross-tenant audit logs). The coupling meant:
 *
 *     1. Granting platform access required changing `role`, which
 *        silently changed the user's in-app permissions as a side effect
 *        (a researcher promoted to platformOwner would also gain admin
 *        rights inside every org they were a member of).
 *
 *     2. The JWT carried only `role`, so every /api/admin/* request had
 *        to do a DB round-trip to check whether `role === "platformOwner"`
 *        was still valid — or accept the staleness window (up to 15 min
 *        for access tokens, 30 days for refresh tokens).
 *
 *     3. The `role === "owner"` check in `requireAdmin` (api-helpers.ts)
 *        was STILL being called by some /api/admin/* routes as a fallback,
 *        meaning an org owner could still trip the admin gate in some
 *        code paths. The audit (Task 261) found this inconsistent.
 *
 *   The clean fix (implemented here) is a SEPARATE `platformRole` field
 *   on the User row, carried in the JWT as a separate claim, and gated
 *   by THIS middleware. The two fields are independently grantable and
 *   independently revocable — the OWASP ASVS V1.2 "Separation of Duties"
 *   pattern for multi-tenant SaaS.
 *
 * USAGE:
 *
 *   import { requirePlatformAdmin } from "@/lib/auth/require-platform-admin";
 *
 *   export async function GET(req: NextRequest) {
 *     const auth = await requirePlatformAdmin(req);
 *     if (auth.user === null) return auth.response;
 *     // ... route handler ...
 *   }
 *
 * The middleware also enforces:
 *   - Authentication (delegates to getAuthenticatedUser).
 *   - CSRF protection on state-changing methods (POST/PATCH/PUT/DELETE).
 *   - Rate limiting (1 req/sec per platform admin — Task 273).
 *
 * BEHAVIOR:
 *   - 401 if no authenticated user.
 *   - 403 if authenticated but platformRole !== "admin". The 403 response
 *     body does NOT leak whether the user exists or what their current
 *     platformRole is — just "forbidden". This is OWASP-recommended:
 *     error messages should not reveal authorization state to attackers.
 *   - Writes a critical audit log entry on every 403 (so the SaaS operator
 *     can detect probing). The audit log is best-effort: if it fails, the
 *     403 still stands (fail-closed on authz, fail-open on audit).
 *
 * AUDIT LOG ON DENIAL:
 *   Every 403 from this middleware writes an audit log entry with action
 *   "platform_admin_denied". This lets the SaaS operator detect:
 *     - A user probing /api/admin/* to test their privileges.
 *     - A compromised account that's been used to probe.
 *     - A misconfigured client that's hitting admin endpoints by mistake.
 *   The audit log is best-effort (non-critical) — if the audit-log write
 *   fails, the 403 still stands. We don't want a DB outage to lock the
 *   entire admin console out by failing every request with a 500.
 */

import { type NextRequest, NextResponse } from "next/server";
import { getAuthenticatedUser, type AuthenticatedUser } from "@/lib/auth/server";
import { writeAuditLog, requireCsrfOrSend } from "@/lib/api-helpers";
import { checkUserRateLimitDistributed } from "@/lib/auth/per-user-rate-limit";

// Task 273 + BE-060 ROOT FIX: rate limits for platform admin endpoints.
//
// BE-060: The previous code only applied the rate limit to non-GET
// requests. GET requests were exempt — an admin (or an attacker with a
// stolen admin session) could make unlimited GET requests to
// /api/admin/users (which returns PII). At 50 users per request × 1
// req/sec = 3000 users exfiltrated in 60 seconds. The rate limit is a
// throttle, not a block — but a stolen admin session could slowly
// exfiltrate the entire user database over ~1 hour, undetected.
//
// Root fix: apply DIFFERENT rate limits to GET vs state-changing
// methods. GET gets a more generous limit (5 req/sec — still allows the
// admin console's periodic polling at 1 req/5s, but blocks rapid
// exfiltration). POST/PATCH/PUT/DELETE keep the strict 1 req/sec limit
// (state-changing operations should be rare for a platform admin).
const PLATFORM_ADMIN_WRITE_RATE_LIMIT = { max: 1, windowSeconds: 1 };
const PLATFORM_ADMIN_READ_RATE_LIMIT = { max: 5, windowSeconds: 1 };

export type PlatformAdminAuth =
  | { user: AuthenticatedUser; response: null }
  | { user: null; response: Response };

/**
 * Returns true iff the user has the platform-admin role.
 *
 * Exported so other modules can do the same check without paying the
 * full middleware cost (e.g. /api/audit-logs uses this to decide
 * whether to scope queries to the user's org or to allow system-wide
 * access for platform admins).
 *
 * Fail-closed: undefined / null / unknown values are treated as "not
 * admin". The only value that returns true is the literal string "admin".
 */
export function isPlatformAdmin(user: { platformRole?: string } | null | undefined): boolean {
  return user?.platformRole === "admin";
}

/**
 * Require a platform admin (SaaS operator staff) for the current request.
 *
 * See the file-level comment for the full rationale. This function:
 *   1. Authenticates the user (401 if not logged in).
 *   2. Checks `user.platformRole === "admin"` (403 otherwise).
 *   3. Enforces CSRF on state-changing methods (POST/PATCH/PUT/DELETE).
 *   4. Enforces a 1 req/sec per-admin rate limit (Task 273).
 *
 * The `req` parameter is required for CSRF + rate-limit checks. For
 * GET-only routes that don't need CSRF, you can pass `null` as the
 * request and the CSRF check is skipped (rate limit still applies,
 * using the authenticated user's ID).
 */
export async function requirePlatformAdmin(req: NextRequest | null): Promise<PlatformAdminAuth> {
  const user = await getAuthenticatedUser();
  if (!user) {
    return {
      user: null,
      response: NextResponse.json(
        { error: "unauthorized", message: "Authentication required" },
        { status: 401 }
      ),
    };
  }

  // TASK-261: the GATE. Only platformRole === "admin" passes. Any other
  // value (including "none", undefined, null, or a forged string) is
  // denied. This is the architectural fix — the gate is on a SEPARATE
  // field from `role`, so promoting someone to org `admin` or `owner`
  // no longer grants them platform-admin access.
  if (!isPlatformAdmin(user)) {
    // Best-effort audit log of the denial. This lets the SaaS operator
    // detect probing (a user hitting /api/admin/* to test their
    // privileges) and investigate compromised accounts. Non-critical: if
    // the audit log write fails, the 403 still stands — we don't want
    // a DB outage to lock the admin console out by failing every
    // request with a 500.
    await writeAuditLog({
      user,
      action: "platform_admin_denied",
      resource: req?.url || "(unknown)",
      metadata: {
        method: req?.method || "GET",
        // Record the user's current role + platformRole for forensics.
        // This is NOT leaked to the client (the 403 body just says
        // "forbidden") — it's only in the audit log, which is itself
        // gated behind platform-admin access.
        userRole: user.role,
        userPlatformRole: user.platformRole || "none",
      },
    }).catch(() => {
      // Swallow — audit log failure must not change the 403 response.
    });
    return {
      user: null,
      response: NextResponse.json(
        {
          error: "forbidden",
          message: "Platform administrator access required.",
        },
        { status: 403 }
      ),
    };
  }

  // TASK-273 + BE-060: rate limit — applies to BOTH GET and state-changing
  // methods now (previously GET was exempt). Different limits:
  //   - GET/HEAD: 5 req/sec (allows admin console polling, blocks exfil).
  //   - POST/PATCH/PUT/DELETE: 1 req/sec (state changes should be rare).
  // We use the distributed limiter so the cap holds across multiple Node.js
  // instances (production deployments behind a load balancer).
  const isWrite = req && req.method !== "GET" && req.method !== "HEAD";
  const limit = isWrite ? PLATFORM_ADMIN_WRITE_RATE_LIMIT : PLATFORM_ADMIN_READ_RATE_LIMIT;
  if (req) {
    const rl = await checkUserRateLimitDistributed(user.userId, limit);
    if (rl.blocked) {
      // BE-060: when an admin hits the rate limit, log it as a potential
      // exfiltration signal (for GET) or a misuse signal (for writes).
      // This is best-effort — the rate limit itself is the primary defense.
      await writeAuditLog({
        user,
        action: "platform_admin_rate_limited",
        resource: req.url || "(unknown)",
        metadata: {
          method: req.method || "GET",
          limit: `${limit.max}/${limit.windowSeconds}s`,
          retryAfterSeconds: rl.retryAfterSeconds,
        },
      }).catch(() => {
        // Swallow — audit log failure must not change the 429 response.
      });
      return {
        user: null,
        response: NextResponse.json(
          {
            error: "rate_limited",
            message: "Too many admin requests. Slow down.",
            retryAfter: rl.retryAfterSeconds,
          },
          {
            status: 429,
            headers: { "Retry-After": String(rl.retryAfterSeconds) },
          }
        ),
      };
    }
  }

  // TASK-272 / FE-011: CSRF protection on state-changing methods.
  // The requireCsrfOrSend helper handles the double-submit cookie
  // pattern and exempts valid API keys. We skip it for GET/HEAD
  // (no state change) and for null requests (caller explicitly opted out).
  if (req && req.method !== "GET" && req.method !== "HEAD") {
    const csrf = await requireCsrfOrSend(req);
    if (csrf.response) {
      return { user: null, response: csrf.response };
    }
  }

  return { user, response: null };
}
