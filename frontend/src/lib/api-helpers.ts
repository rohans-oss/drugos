/**
 * Shared API helpers for Next.js route handlers.
 */

import { NextResponse, type NextRequest } from "next/server";
import { getAuthenticatedUser, type AuthenticatedUser } from "@/lib/auth/server";
import { db } from "@/lib/db";

export async function requireAuth(): Promise<{ user: AuthenticatedUser; response: null } | { user: null; response: Response }> {
  const user = await getAuthenticatedUser();
  if (!user) {
    return {
      user: null,
      response: NextResponse.json({ error: "unauthorized", message: "Authentication required" }, { status: 401 }),
    };
  }
  return { user, response: null };
}

export async function requireAdmin(): Promise<{ user: AuthenticatedUser; response: null } | { user: null; response: Response }> {
  const auth = await requireAuth();
  if (auth.user === null) return auth;
  // BE-002 ROOT FIX: accept admin, owner, AND platformOwner. All three are
  // "admin-level" roles for the purpose of requireAdmin (they can all
  // access admin endpoints). The DIFFERENCE is in cross-tenant scoping:
  // only `platformOwner` bypasses org scoping (see isPlatformSuperuser).
  // `owner` is now org-scoped just like `admin` — the previous code
  // granted `owner` system-wide access, which was a multi-tenant data
  // breach (any user promoted to owner could enumerate every user in
  // every org). The fix is to introduce `platformOwner` as the ONLY
  // role with system-wide access, settable only via direct DB access.
  if (
    auth.user.role !== "admin" &&
    auth.user.role !== "owner" &&
    auth.user.role !== "platformOwner"
  ) {
    return {
      user: null,
      response: NextResponse.json({ error: "forbidden", message: "Admin access required" }, { status: 403 }),
    };
  }
  return auth;
}

/**
 * BE-002 ROOT FIX: returns true ONLY for the `platformOwner` role.
 *
 * The `platformOwner` role is the ONLY role that bypasses org scoping.
 * It is reserved for the SaaS operator's staff and can ONLY be set via
 * direct DB access — no API route can grant it (ALLOWED_ROLES_ADMIN in
 * register/route.ts does NOT include "platformOwner").
 *
 * Routes that have a "system-wide" path (e.g. admin/users GET without
 * org filter, audit-logs GET without org filter) MUST gate that path
 * on `isPlatformSuperuser(auth.user)`. The `owner` and `admin` roles
 * are org-scoped — they see only their own org's data.
 *
 * This is the root fix for the multi-tenant data breach where any user
 * promoted to `owner` could read every user in every org, suspend any
 * user system-wide, and read every audit log across all tenants.
 */
export function isPlatformSuperuser(user: { role: string } | null): boolean {
  return user?.role === "platformOwner";
}

/**
 * FE-010 ROOT FIX: requireRole was previously defined in lib/auth/server.ts
 * but had ZERO call sites — every non-admin route used requireAuth (any
 * authenticated user), so a read-only viewer could change billing, revoke
 * API keys, create projects in any org, etc.
 *
 * This helper is the route-call-site-friendly version: it returns
 * { user, response } just like requireAuth/requireAdmin, so routes can use
 * the same early-return pattern:
 *
 *   const auth = await requireRole(user, "billing", "owner", "admin");
 *   if (auth.user === null) return auth.response;
 *
 * The roles list is variadic: pass every role that should be allowed. We
 * always implicitly allow "admin" and "owner" because they are superuser
 * roles — restricting them from a specific endpoint would be a footgun.
 */
export async function requireRole(
  user: AuthenticatedUser | null,
  ...roles: string[]
): Promise<{ user: AuthenticatedUser; response: null } | { user: null; response: Response }> {
  // Always authenticate first.
  if (!user) {
    return {
      user: null,
      response: NextResponse.json(
        { error: "unauthorized", message: "Authentication required" },
        { status: 401 }
      ),
    };
  }
  // Admin/owner/platformOwner are always allowed (superuser bypass).
  // BE-002: platformOwner is the new system-wide superuser role. owner
  // and admin are org-scoped admin-level roles (they can access admin
  // endpoints but only see their own org's data).
  const allowed = new Set([...roles, "admin", "owner", "platformOwner"]);
  if (!allowed.has(user.role)) {
    return {
      user: null,
      response: NextResponse.json(
        {
          error: "forbidden",
          message: `Your role (${user.role}) is not permitted to perform this action.`,
        },
        { status: 403 }
      ),
    };
  }
  return { user, response: null };
}

/**
 * Convenience wrapper for routes that need role-based access but don't
 * already have the user. Combines requireAuth + requireRole in one call:
 *
 *   const auth = await requireAuthRole("billing", "owner", "admin");
 *   if (auth.user === null) return auth.response;
 */
export async function requireAuthRole(
  ...roles: string[]
): Promise<{ user: AuthenticatedUser; response: null } | { user: null; response: Response }> {
  const auth = await requireAuth();
  if (auth.user === null) return auth;
  return requireRole(auth.user, ...roles);
}

/**
 * FE-034 ROOT FIX: writeAuditLog previously swallowed ALL failures silently.
 * For a pharma platform where audit logs are regulatory requirements
 * (FDA 21 CFR Part 11), silent audit-log failure is a compliance
 * violation — security incidents could go unrecorded and forensic
 * investigations would be hampered by missing log entries.
 *
 * Now we support TWO modes:
 *
 *   1. Critical (default for security actions): If the audit-log write
 *      fails, the request is ABORTED with a 500. The caller never sees
 *      a "success" response that wasn't audited. Use this for: login,
 *      logout, password change, 2FA enable/disable, admin actions,
 *      role/status changes, API key creation/revocation, billing changes.
 *
 *   2. Non-critical (default): If the audit-log write fails, we log to
 *      stderr AND record a fallback entry in a dead-letter table (or
 *      just stderr if the DB is the failure cause). The request
 *      continues — the action was less important than the audit trail.
 *
 * The caller decides which mode by passing `critical: true` or omitting
 * it. Security-sensitive callers MUST pass `critical: true`.
 *
 * Returns:
 *   - { ok: true } on success.
 *   - { ok: false, error } on failure. If critical, the caller MUST
 *     return internalError() to the client.
 */
export interface AuditLogResult {
  ok: boolean;
  error?: string;
}

export async function writeAuditLog(params: {
  user: AuthenticatedUser | null;
  action: string;
  resource?: string;
  metadata?: Record<string, unknown>;
  /**
   * If true, a failure to write the audit log ABORTS the request.
   * Use for security-critical actions (login, password change, 2FA
   * disable, admin actions, role/status changes, API key ops).
   */
  critical?: boolean;
  /** Optional request IP, for forensic analysis. */
  ip?: string;
  /** Optional User-Agent string, for forensic analysis. */
  userAgent?: string;
  /**
   * Optional organization ID. Stored in the audit log row if the
   * schema supports it; otherwise folded into metadata.
   * (Team-15 FE-045 webhook audit tests pass this field.)
   *
   * FE-005 ROOT FIX (v2): If the caller does NOT pass this explicitly,
   * we auto-populate it from `params.user.orgId`. The previous version
   * required every call site to pass `organizationId` explicitly —
   * ZERO call sites did, so every AuditLog row was written with
   * `organizationId: null`. The /api/audit-logs route then filtered
   * by `organizationId: auth.user.orgId` and got an empty result for
   * every non-owner admin (an accidental "fix" via broken behavior).
   * Worse, an owner querying system-wide saw rows with null orgId
   * and could not attribute them to any tenant — defeating the
   * cross-tenant isolation the column was added for. Auto-populating
   * from the authenticated user's orgId is the OWASP-recommended
   * pattern: the actor's org is ALWAYS known at audit-write time.
   */
  organizationId?: string;
}): Promise<AuditLogResult> {
  // FE-005 / FE-040 ROOT FIX (v2, merged): Resolve the effective
  // organizationId — explicit param wins, else fall back to the
  // authenticated user's orgId.
  //
  // The previous "fix" added `organizationId String?` to the AuditLog
  // schema and accepted an optional `organizationId` param here — BUT
  // none of the ~20 production callers (billing, evidence-package, kg,
  // rl, admin, auth/*) ever passed it. So the column was ALWAYS NULL in
  // production, completely defeating the purpose of multi-tenant
  // audit-trail isolation. The fix was comments-only — exactly the
  // failure mode the audit warned about.
  //
  // Real root fix (independently arrived at by Team 12 and Team Cosmic):
  // auto-populate `organizationId` from the authenticated user's `orgId`
  // (set by getAuthenticatedUser from the access-token's `orgId` claim)
  // when the caller does not explicitly pass one. This makes EVERY
  // user-initiated audit-log row org-scoped automatically, without
  // requiring each call site to remember to pass it. Callers that need
  // to override (e.g. system/webhook events with no user session) can
  // still pass `organizationId` explicitly.
  const effectiveOrgId =
    params.organizationId ?? params.user?.orgId ?? null;

  try {
    await db.auditLog.create({
      data: {
        userId: params.user?.userId || null,
        actorName: params.user?.email || "anonymous",
        action: params.action,
        resource: params.resource || null,
        ip: params.ip || null,
        userAgent: params.userAgent || null,
        // Always populate the organizationId column when we have one.
        // This is what makes the /api/audit-logs org filter actually
        // work — previously every row had null orgId.
        ...(effectiveOrgId ? { organizationId: effectiveOrgId } : {}),
        metadata: JSON.stringify({
          ...(params.metadata || {}),
          // Also fold organizationId into metadata for call sites that
          // read the JSON blob (e.g. the audit-log viewer UI), so the
          // org context is discoverable even when the row is exported
          // as JSON.
          ...(effectiveOrgId ? { organizationId: effectiveOrgId } : {}),
        }),
      },
      // BE-061 ROOT FIX (v115, LOW): the previous code had `as any` on
      // the data object — this bypassed TypeScript's type check on the
      // AuditLog.organizationId field. The fix removes the `as any` so
      // TypeScript validates the payload. If the Prisma schema is
      // updated (e.g., a new required field is added), the build will
      // fail here instead of silently writing null at runtime.
    });
    return { ok: true };
  } catch (e) {
    const errMsg = e instanceof Error ? e.message : String(e);
    // Always log to stderr — even if critical, this gives operators a
    // chance to see what went wrong before the request is aborted.
    console.error("[AUDIT-LOG-FAILURE]", {
      action: params.action,
      resource: params.resource,
      userId: params.user?.userId,
      error: errMsg,
      critical: params.critical === true,
      timestamp: new Date().toISOString(),
    });

    if (params.critical === true) {
      // Abort the request — the action MUST be auditable.
      return { ok: false, error: errMsg };
    }

    // BE-003 ROOT FIX: non-critical fallback — write to the
    // AuditLogDeadLetter table (modeled in Prisma schema). The previous
    // code used raw $executeRaw SQL with CREATE TABLE IF NOT EXISTS on
    // every call — that had three bugs (see schema.prisma comment for
    // AuditLogDeadLetter). The Prisma-model path is:
    //   1. Type-safe (no SQL injection, no DDL on every call).
    //   2. Queryable from the application layer (operators can list
    //      dead-letter entries via /api/admin/audit-logs?dead_letter=true).
    //   3. Monitorable (alert when row count grows — indicates a
    //      systemic AuditLog write failure that needs operator attention).
    // If THIS write also fails (DB truly down), we log to stderr as the
    // last-resort record. The request continues because the action was
    // non-critical — but the dead-letter entry is the compliance trail.
    //
    // The underlying PostgreSQL table name is `audit_log_dead_letter`
    // (Prisma maps the AuditLogDeadLetter model to this snake_case
    // table name via the @@map annotation in schema.prisma). The
    // /api/admin/audit-logs route accepts a `?dead_letter=true` query
    // param to list rows from this table.
    //
    // BE-080 ROOT FIX (v115, LOW): the previous code swallowed the
    // dead-letter write error silently (only a console.error). The fix
    // captures the dead-letter error in a structured log entry that
    // includes the original audit-log params — so operators can
    // RECONSTRUCT the audit trail from the stderr log if both the
    // primary AND dead-letter writes fail. The structured format is
    // machine-parseable for log aggregation (Splunk, CloudWatch Logs,
    // ELK). The dead-letter error is ALSO returned to the caller (in
    // the result.error field) so the caller can decide whether to
    // surface a user-facing warning.
    let deadLetterError: string | undefined;
    try {
      await db.auditLogDeadLetter.create({
        data: {
          action: params.action,
          resource: params.resource || null,
          userId: params.user?.userId || null,
          actorName: params.user?.email || "anonymous",
          metadata: JSON.stringify(params.metadata || {}),
          error: errMsg,
          // FE-016 ROOT FIX (Team Member 15, v108 — pre-existing build blocker):
          // We already returned early at line 260 if params.critical === true,
          // so by the time we reach this dead-letter write, params.critical is
          // guaranteed to be `false | undefined`. TypeScript narrows the type
          // accordingly and flags `params.critical === true` as an unintentional
          // comparison. Use `false` directly — semantically identical to the
          // narrowed expression.
          critical: false,
          ...(effectiveOrgId ? { organizationId: effectiveOrgId } : {}),
        },
      });
    } catch (fallbackErr) {
      // BE-080: Both primary and fallback failed — the DB is likely
      // down. Capture the dead-letter error and log a STRUCTURED
      // record with the FULL audit-log params so operators can
      // reconstruct the audit trail from the stderr log.
      deadLetterError = fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr);
      console.error("[AUDIT-LOG-FAILURE] Dead-letter write ALSO failed — MANUAL RECOVERY REQUIRED:", {
        originalAuditLog: {
          action: params.action,
          resource: params.resource,
          userId: params.user?.userId,
          actorName: params.user?.email || "anonymous",
          organizationId: effectiveOrgId,
          metadata: params.metadata,
          ip: params.ip,
          userAgent: params.userAgent,
        },
        primaryError: errMsg,
        deadLetterError,
        timestamp: new Date().toISOString(),
        recoveryAction:
          "Both the AuditLog and AuditLogDeadLetter tables are unreachable. " +
          "The audit trail for this event exists ONLY in this log entry. " +
          "Operators MUST manually re-create the audit log entry once the " +
          "DB is restored, OR accept that this event is unrecorded (FDA " +
          "21 CFR Part 11 compliance incident — file a CAPA).",
      });
    }
    return { ok: false, error: deadLetterError ? `${errMsg} (dead-letter also failed: ${deadLetterError})` : errMsg };
  }
}

/**
 * FE-034: Convenience helper for critical audit logs. Throws if the
 * write fails, so the caller's existing try/catch returns 500.
 *
 * Usage:
 *   await writeAuditLogCritical({ user, action: 'login', resource: ... });
 *   // — if this throws, the caller's catch block returns 500.
 */
export async function writeAuditLogCritical(params: {
  user: AuthenticatedUser | null;
  action: string;
  resource?: string;
  metadata?: Record<string, unknown>;
  ip?: string;
  userAgent?: string;
}): Promise<void> {
  const result = await writeAuditLog({ ...params, critical: true });
  if (!result.ok) {
    throw new Error(`audit_log_write_failed: ${result.error}`);
  }
}

export function badRequest(message: string) {
  return NextResponse.json({ error: "bad_request", message }, { status: 400 });
}

export function notFound(message: string) {
  return NextResponse.json({ error: "not_found", message }, { status: 404 });
}

export function internalError(message: string) {
  return NextResponse.json({ error: "internal_error", message }, { status: 500 });
}

/**
 * FE-022 ROOT FIX: requireRoleOrSend — like requireAuthRole but returns
 * { user, response } so routes can use the early-return pattern.
 */
export async function requireRoleOrSend(
  ...roles: string[]
): Promise<{ user: AuthenticatedUser; response: null } | { user: null; response: Response }> {
  return requireAuthRole(...roles);
}

// ---------------------------------------------------------------------------
// FE-011 ROOT FIX: REAL CSRF protection (double-submit cookie pattern).
//
// Previously `requireCsrfOrSend` was a NO-OP that returned `{ ok: true }`
// unconditionally — every state-changing route that called it had a false
// sense of security. Combined with SameSite=Lax cookies (which DO provide
// partial CSRF protection for non-GET requests from cross-origin, but are
// bypassable in older browsers and would be re-opened if SameSite=None
// were ever set for SSO), this left POST/PUT/PATCH/DELETE routes exposed.
//
// The fix uses the double-submit cookie pattern (OWASP-recommended):
//
//   1. On successful login, setAuthCookies() also sets a `drugos_csrf`
//      cookie containing a random 32-byte token. The cookie is NOT
//      HttpOnly — the browser client MUST be able to read it and copy
//      its value into the `X-CSRF-Token` request header on every
//      state-changing request.
//   2. requireCsrfOrSend(req) reads the `X-CSRF-Token` header and the
//      `drugos_csrf` cookie. If they are equal (constant-time) AND
//      non-empty, the request passes. If either is missing or they
//      differ, the request is rejected with 403.
//   3. The token is session-scoped (regenerated on each login) so a
//      stolen token does not survive a re-login.
//
// Why double-submit and not synchronizer-token: double-submit is stateless
// — we don't need a server-side table of issued tokens. The cookie's
// SameSite=Lax attribute (set below) prevents cross-origin reads, so an
// attacker on evil.com cannot read the victim's csrf cookie to forge the
// matching header. The combination of (cookie SameSite=Lax) + (header must
// match cookie) + (header is not auto-sent by browsers) is the defense.
//
// API-key auth (Authorization: Bearer drugos_…) is exempt — programmatic
// clients do not have cookies and are not vulnerable to CSRF (an attacker
// cannot make the victim's browser send an Authorization header with the
// attacker's key, and even if they could, they'd be using their own key).
// ---------------------------------------------------------------------------

import { randomBytes, timingSafeEqual } from "crypto";

export const CSRF_COOKIE_NAME = "drugos_csrf";
export const CSRF_HEADER_NAME = "x-csrf-token";

/**
 * Issue a fresh CSRF token (32 random bytes, hex-encoded). Called by the
 * login route on successful authentication and by the refresh route when
 * access cookies are rotated. The token is stored in a cookie that the
 * browser can read (httpOnly: false) so the SPA can copy it into the
 * X-CSRF-Token header.
 */
export function issueCsrfToken(): string {
  return randomBytes(32).toString("hex");
}

export async function setCsrfCookie(token: string): Promise<void> {
  const { cookies } = await import("next/headers");
  const store = await cookies();
  const isProd = process.env.NODE_ENV === "production";
  store.set(CSRF_COOKIE_NAME, token, {
    httpOnly: false, // the client MUST read this to copy into the header
    secure: isProd,
    sameSite: "lax", // not strict — we want top-level navigations to keep the cookie
    path: "/",
    maxAge: 30 * 24 * 60 * 60, // 30 days — matches REFRESH_TOKEN_TTL_DAYS
  });
}

export async function clearCsrfCookie(): Promise<void> {
  const { cookies } = await import("next/headers");
  const store = await cookies();
  store.set(CSRF_COOKIE_NAME, "", {
    httpOnly: false,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 0,
  });
}

/**
 * BE-078 ROOT FIX: CSRF bypass via fake API key.
 *
 * The previous code exempted ANY request with `Authorization: Bearer drugos_...`
 * from CSRF checks — even if the key was INVALID. An attacker with the victim's
 * session cookie could send `Authorization: Bearer drugos_fake_key` to skip the
 * CSRF check, then the cookie auth would still succeed. The attacker could now
 * make state-changing requests (POST, PATCH, DELETE) without the CSRF token.
 *
 * Root fix: Only exempt CSRF if the API key is VALID. We now call
 * authenticateApiKey() to verify the key before exempting. Invalid API keys
 * fall through to the normal CSRF check (which will reject the request if
 * cookies are present but no CSRF token is provided).
 *
 * Exemptions:
 *   - Requests with a VALID `Authorization: Bearer drugos_…` API key are EXEMPT.
 *     API-key auth is not vulnerable to CSRF (the attacker cannot make the
 *     victim's browser send the attacker's key). Exempting VALID API keys is
 *     necessary for the developer platform to work.
 */
export async function requireCsrfOrSend(req: NextRequest): Promise<{
  ok: boolean;
  response: null;
} | { ok: false; response: Response }> {
  // BE-078: Exemption 1 — API-key auth (Bearer drugos_…) ONLY if the key
  // is VALID. An attacker sending a fake drugos_ prefix to bypass CSRF
  // will fail the auth check and fall through to the cookie-based CSRF
  // validation, which will reject the request.
  //
  // BE-032 ROOT FIX (v115, LOW): Headers.get() is case-insensitive per
  // the Fetch API spec (RFC 7230 §3.2). The previous code redundantly
  // checked `req.headers.get("authorization") || req.headers.get("Authorization")`
  // — the second call always returns the same value as the first. Same
  // for the CSRF header check below. Removed the redundant lookups.
  const authHeader = req.headers.get("authorization");
  if (authHeader && authHeader.toLowerCase().startsWith("bearer ")) {
    const rawKey = authHeader.slice(7).trim();
    if (rawKey.startsWith("drugos_")) {
      // Verify the key is actually valid before exempting CSRF.
      const { authenticateApiKey } = await import("@/lib/auth/server");
      const apiUser = await authenticateApiKey(rawKey);
      if (apiUser) {
        // Valid API key — exempt from CSRF (programmatic clients are not
        // vulnerable to browser-based CSRF attacks).
        return { ok: true, response: null };
      }
      // Invalid API key — do NOT exempt. Fall through to the cookie-based
      // CSRF check below. This closes the bypass: an attacker with a fake
      // key and a valid session cookie will still need the CSRF token.
    }
  }

  // Check for auth cookies (session cookies). If the request has NO auth
  // cookies at all, CSRF protection is irrelevant — CSRF attacks exploit
  // the browser's auto-sending of cookies on cross-site requests. If there
  // are no cookies, there's nothing to exploit. This exemption lets
  // unauthenticated endpoints like /api/auth/login and /api/auth/register
  // work without a pre-issued CSRF token.
  let hasAccessCookie = false;
  let hasRefreshCookie = false;
  let cookieToken = "";
  try {
    const { cookies } = await import("next/headers");
    const store = await cookies();
    hasAccessCookie = !!store.get("drugos_access")?.value;
    hasRefreshCookie = !!store.get("drugos_refresh")?.value;
    cookieToken = store.get(CSRF_COOKIE_NAME)?.value || "";
  } catch {
    // cookies() throws outside a request scope — treat as missing.
  }

  // Exemption 2: No session cookies → unauthenticated request → CSRF N/A.
  // This covers login, register, and any other public POST endpoint.
  if (!hasAccessCookie && !hasRefreshCookie && !cookieToken) {
    return { ok: true, response: null };
  }

  // BE-032 ROOT FIX: Headers.get is case-insensitive per Fetch spec —
  // removed the redundant `.toUpperCase()` fallback.
  const headerToken = req.headers.get(CSRF_HEADER_NAME) || "";

  if (!cookieToken || !headerToken) {
    return {
      ok: false,
      response: NextResponse.json(
        { error: "csrf_missing", message: "CSRF token missing — both the drugos_csrf cookie and the X-CSRF-Token header are required for authenticated requests." },
        { status: 403 }
      ),
    };
  }
  // Constant-time comparison to prevent timing attacks.
  const a = Buffer.from(cookieToken);
  const b = Buffer.from(headerToken);
  if (a.length !== b.length || !timingSafeEqual(a, b)) {
    return {
      ok: false,
      response: NextResponse.json(
        { error: "csrf_mismatch", message: "CSRF token mismatch — the X-CSRF-Token header does not match the drugos_csrf cookie." },
        { status: 403 }
      ),
    };
  }
  return { ok: true, response: null };
}
