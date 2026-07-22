import { NextRequest, NextResponse } from "next/server";
import { getAuthenticatedUser, signAccessToken, setAuthCookies, clearAuthCookies } from "@/lib/auth/server";
import { db } from "@/lib/db";
import { badRequest, writeAuditLog, internalError, requireCsrfOrSend } from "@/lib/api-helpers";
// BE-029 ROOT FIX (Team Member 12): Zod-validated PATCH body.
import { validateBody, AuthMePatchBody } from "@/lib/zod-schemas";

/**
 * GET /api/auth/me — return the current user's profile + organization
 * memberships. Returns 401 if no valid session.
 *
 * FE-051 ROOT FIX: every authenticated page load triggers this endpoint,
 * which does db.user.findUnique + db.organizationMember.findMany. For a
 * SPA with frequent re-renders and a pharma platform with 1000 researchers
 * doing 100 page views/day each, that's 200K DB queries/day just for /me.
 * We add `Cache-Control: private, max-age=60` so browsers AND Next.js's
 * Data Cache cache the response for 60 seconds per user.
 *
 * `private` is critical: it forbids shared/CDN caches from storing the
 * response (which would leak one user's profile to others). Only the
 * user's own browser may cache it. `max-age=60` is short enough that
 * role/membership changes propagate within a minute, but long enough
 * to collapse the 100-page-views/day-per-user load into ~1 DB query/min.
 *
 * The PATCH handler below remains un-cached because it mutates state.
 */
export async function GET() {
  const authUser = await getAuthenticatedUser();
  if (!authUser) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  // BE-039 ROOT FIX (Team Member 12): the previous code did TWO DB
  // queries — `db.user.findUnique(...)` + `db.organizationMember.findMany(...)`.
  // For 1000 researchers doing 100 page views/day, that's 200K extra
  // queries/day just for /api/auth/me. We collapse them into ONE query
  // using Prisma's `include` to eager-load the user's org memberships.
  // The `include` performs a single JOIN on the server side, halving
  // the DB round-trips for every authenticated page load.
  //
  // FE-060 preservation: we still use `select` on the nested
  // `organization` to fetch only id/name/slug/plan (not the entire
  // Organization record).
  let user;
  try {
    user = await db.user.findUnique({
      where: { id: authUser.userId },
      select: {
        id: true,
        email: true,
        name: true,
        role: true,
        title: true,
        bio: true,
        status: true,
        emailVerified: true,
        academicVerified: true,
        mfaEnabled: true,
        lastLoginAt: true,
        createdAt: true,
        organizationMemberships: {
          select: {
            role: true,
            organization: {
              select: { id: true, name: true, slug: true, plan: true },
            },
          },
        },
      },
    });
  } catch (e: any) {
    if (e?.name === 'PrismaClientInitializationError' || e?.code === 'P1001') {
      return NextResponse.json(
        { error: "database_offline", message: "Cannot connect to the database. Ensure the Docker Postgres container is running." },
        { status: 503 }
      );
    }
    throw e;
  }
  if (!user) {
    // FE-068 ROOT FIX: Return 401 (not 404) when the access token decoded
    // successfully but the user no longer exists in the DB. Returning 404
    // leaked information: an attacker could distinguish "valid token for a
    // deleted user" (404) from "invalid token" (401). Treating both cases
    // as 401 collapses the side channel — the attacker learns nothing
    // about whether the user ever existed.
    //
    // BE-028 ROOT FIX: ALSO clear the auth cookies. The previous code
    // returned 401 WITHOUT clearing cookies, so the browser kept sending
    // the bad access cookie on every subsequent request → every request
    // hit the DB, found no user, returned 401. The user was locked out
    // until they manually cleared cookies. /api/auth/refresh handles this
    // correctly (FE-031) — /api/auth/me now does the same.
    await clearAuthCookies();
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  // BE-039: the memberships are now nested under `organizationMemberships`
  // (the implicit relation name from the User → OrganizationMember back-relation).
  // Defensive: if the Prisma client doesn't return the relation (e.g. an
  // older client version, or a test mock that doesn't populate it), fall
  // back to an empty array rather than throwing on `.map()`.
  const memberships = user.organizationMemberships ?? [];
  const body = {
    user: {
      id: user.id,
      email: user.email,
      name: user.name,
      role: user.role,
      title: user.title,
      bio: user.bio,
      status: user.status,
      emailVerified: user.emailVerified,
      academicVerified: user.academicVerified,
      mfaEnabled: user.mfaEnabled,
      lastLoginAt: user.lastLoginAt,
      createdAt: user.createdAt,
    },
    organizations: memberships.map((m) => ({
      id: m.organization.id,
      name: m.organization.name,
      slug: m.organization.slug,
      plan: m.organization.plan,
      role: m.role,
    })),
    activeOrganizationId: authUser.orgId || memberships[0]?.organization.id || null,
  };
  return NextResponse.json(body, {
    headers: {
      // BE-055 ROOT FIX: previously `Cache-Control: private, max-age=60`
      // was applied to a response body that included `mfaEnabled` and
      // `emailVerified`. After the user enabled/disabled 2FA (or verified
      // their email), the cached /me response still showed the OLD value
      // for up to 60 seconds — the UI said "2FA: Off" even though the DB
      // said "2FA: On". For a security setting, 60 seconds of stale data
      // is concerning: the UI may not prompt for 2FA on the next sensitive
      // action, or the user may re-attempt enrollment and get a confusing
      // "2FA is already enabled" error. Setting `no-cache` (with
      // `no-store` for defense in depth) forces the browser to re-fetch
      // /me on every page load — the extra DB query per page load is
      // acceptable for a pharma platform where the security state must
      // be accurate. The previous `private, max-age=60` was a performance
      // optimization that traded security-state correctness for ~1 DB
      // query/min savings — not worth it.
      "Cache-Control": "no-cache, no-store, must-revalidate",
      "Pragma": "no-cache",
      "Expires": "0",
    },
  });
}

/**
 * PATCH /api/auth/me — update the current user's profile fields.
 *
 * Only safe, user-editable fields are accepted: name, title, bio.
 * Email and role changes are NOT allowed here — they require admin
 * intervention or a separate verification flow.
 */
export async function PATCH(req: NextRequest) {
  const authUser = await getAuthenticatedUser();
  if (!authUser) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  // BE-013 ROOT FIX: CSRF protection on the PATCH handler. The previous
  // code did NOT call `requireCsrfOrSend(req)`, unlike EVERY other
  // state-changing /api/auth/* route (logout, password, 2fa/setup,
  // 2fa/verify, 2fa/disable, 2fa/login-verify, register). The PATCH
  // handler updates the user's profile AND switches their active
  // organization (which re-issues the access token with a new orgId,
  // changing what data they can access). This is a state-changing route.
  // The SameSite=Strict access cookie mitigates cross-site POSTs, but:
  //   (a) the route also accepts the refresh cookie (SameSite=Lax) which
  //       IS sent on top-level navigations;
  //   (b) the audit's defense-in-depth principle says CSRF tokens are
  //       required EVEN when SameSite is set, because SameSite can be
  //       bypassed in older browsers and future SSO flows may set
  //       SameSite=None;
  //   (c) the FE-011 fix's own comment says "Combined with CSRF tokens
  //       (FE-011) for defense-in-depth" — this route broke the
  //       defense-in-depth.
  // Adding the CSRF check makes the route consistent with every other
  // state-changing /api/auth/* route.
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  // FE-072 ROOT FIX: Suspended users must not be able to edit their profile.
  //
  // getAuthenticatedUser() only verifies the JWT signature — it does NOT
  // re-check the user's current status in the DB. So a user suspended by
  // an admin retains a valid access token for up to ACCESS_TOKEN_TTL (15
  // min) and could change their name/title/bio during that window. In a
  // pharma research setting, a suspended user changing their display name
  // to impersonate a colleague could cause real collaboration harm.
  //
  // Root fix: fetch the user's current status from the DB on every PATCH
  // and reject if status === "suspended". This is a defense-in-depth
  // measure alongside the longer-term fix (token revocation on suspension
  // via refresh-token revoke + access-token blacklist until expiry).
  const currentUser = await db.user.findUnique({
    where: { id: authUser.userId },
    select: { status: true },
  });
  if (!currentUser) {
    // FE-068 (same rationale as GET): treat deleted user as 401, not 404.
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  if (currentUser.status === "suspended") {
    return NextResponse.json(
      {
        error: "account_suspended",
        message: "Your account has been suspended. Profile changes are not permitted.",
      },
      { status: 403 }
    );
  }

  // BE-079 ROOT FIX: Added activeOrganizationId to the PATCH body so users
  // can switch their active org. Previously, activeOrganizationId was always
  // the first org (from JWT, set at login), with no way to switch. All
  // queries were scoped to the first org they joined, preventing multi-org
  // users from working across organizations.
  let body: { name?: string; title?: string; bio?: string; activeOrganizationId?: string };
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON body");
  }

  // BE-029 ROOT FIX: schema-validate the body. The AuthMePatchBody schema
  // enforces: name (1..200 chars if present), title (≤200), bio (≤2000),
  // activeOrganizationId (non-empty string OR null). The previous code's
  // `typeof body.name === "string"` check accepted ANY string length — a
  // 1MB name would have been written to the DB. The schema caps lengths
  // at parse time.
  const parsed = validateBody(AuthMePatchBody, body);
  if (!parsed.ok) return parsed.response;
  body = parsed.data as { name?: string; title?: string; bio?: string; activeOrganizationId?: string };

  // BE-079: If activeOrganizationId is provided, validate it BEFORE
  // updating any profile fields. The user must be a member of the
  // target org. Invalid orgId → 400 (don't partially update).
  if (body.activeOrganizationId !== undefined) {
    const targetOrgId = body.activeOrganizationId;
    // null means "clear active org" — allowed.
    if (targetOrgId !== null) {
      const membership = await db.organizationMember.findFirst({
        where: { userId: authUser.userId, organizationId: targetOrgId },
        select: { id: true },
      });
      if (!membership) {
        return NextResponse.json(
          { error: "forbidden", message: "You are not a member of the specified organization." },
          { status: 403 }
        );
      }
    }
  }

  const data: { name?: string; title?: string | null; bio?: string | null } = {};
  if (typeof body.name === "string") {
    // BE-029: schema already enforces 1..200 chars, but we still trim
    // and re-check post-trim in case the trimmed result is empty.
    const trimmed = body.name.trim();
    if (trimmed.length < 1) return badRequest("Name cannot be empty");
    data.name = trimmed;
  }
  if (typeof body.title === "string") {
    data.title = body.title.trim().slice(0, 200) || null;
  }
  if (typeof body.bio === "string") {
    data.bio = body.bio.trim().slice(0, 2000) || null;
  }

  // BE-079: Track whether we're doing an org switch (separate from profile update)
  const switchingOrg = body.activeOrganizationId !== undefined;

  // BE-036 ROOT FIX: When `activeOrganizationId: null` is passed, fall back
  // to the user's FIRST org membership (don't leave them orgless). The
  // previous code allowed `null` to mean "clear active org" — setting
  // `lastActiveOrgId = null` and issuing an access token with
  // `orgId: undefined`. Every org-scoped query (projects, hypotheses,
  // billing) would then 403 ("No active organization"). There was no UI
  // flow that would prompt the user to pick a new org — they were stuck
  // in a "no active org" state until they PATCHed again with a valid
  // orgId. The previous behavior was a footgun: a bug in the frontend
  // org switcher (or a confused API client) could leave the user
  // orgless. The fix uses the audit's suggested option 2: when null is
  // passed, look up the user's first org membership and switch to THAT
  // (instead of clearing). If the user genuinely has no org
  // memberships, we return 400 (they must be invited to an org first).
  if (switchingOrg && body.activeOrganizationId === null) {
    const firstMembership = await db.organizationMember.findFirst({
      where: { userId: authUser.userId },
      orderBy: { joinedAt: "asc" },
      select: { organizationId: true },
    });
    if (!firstMembership) {
      return NextResponse.json(
        {
          error: "no_organization_membership",
          message:
            "You have no organization memberships. Ask an administrator to invite you to an organization before you can use org-scoped features.",
        },
        { status: 400 }
      );
    }
    // Replace the null with the first membership's orgId — the rest of
    // the flow treats this as a regular org switch (with membership
    // validation already done via the findFirst above).
    body.activeOrganizationId = firstMembership.organizationId;
  }

  if (Object.keys(data).length === 0 && !switchingOrg) {
    // FE-054 ROOT FIX: HTTP PATCH semantics allow a no-op patch — the server
    // returns 200 with the current (unchanged) resource. Previously this
    // returned 400 "No updatable fields", which broke clients that send an
    // empty patch to refresh their cached profile. We now return the current
    // user resource with 200, matching RFC 5789 §2.1 ("If the server
    // receives a PATCH request with no body, the server MUST process it as
    // if the body was empty and apply no changes").
    const current = await db.user.findUnique({
      where: { id: authUser.userId },
      select: {
        id: true,
        email: true,
        name: true,
        role: true,
        title: true,
        bio: true,
      },
    });
    if (!current) {
      return NextResponse.json({ error: "not_found" }, { status: 404 });
    }
    return NextResponse.json({ user: current, noop: true });
  }

  const updated = await db.user.update({
    where: { id: authUser.userId },
    data,
    select: {
      id: true,
      email: true,
      name: true,
      role: true,
      title: true,
      bio: true,
    },
  });

  // BE-079 REAL ROOT FIX (v2): If the user is switching their active org,
  // PERSIST the new orgId to the User row (lastActiveOrgId) AND issue a new
  // access token with the updated orgId. The prior "fix" only issued a new
  // access token — it never persisted the orgId. So when the access token
  // expired (15 min), rotateRefreshToken issued a new one WITHOUT orgId
  // (because lastActiveOrgId was null), and the user lost their org context.
  // This was the exact "fix introduced a new bug" pattern the audit warned
  // about: the comment said "ensures subsequent requests are scoped to the
  // newly-selected org" but that was only true for 15 minutes.
  //
  // Real root fix: write lastActiveOrgId to the User row so the refresh
  // path (rotateRefreshToken in lib/auth/server.ts) can read it and keep
  // the orgId stable across token rotations. We also issue a new access
  // token immediately so the user doesn't have to wait for the next
  // refresh to see the new org scope.
  let newAccessToken: string | null = null;
  if (switchingOrg && body.activeOrganizationId !== undefined && body.activeOrganizationId !== null) {
    const newOrgId = body.activeOrganizationId;
    // BE-079 v2: Persist the new orgId to the User row. This is the
    // critical missing piece — without it, the org switch only lasts
    // until the access token expires (15 min).
    // BE-036: newOrgId is guaranteed non-null here (we replaced null
    // with the first membership's orgId above). The `|| null` fallback
    // is kept for defensive coding — if newOrgId is somehow empty
    // string, we don't want to persist that as a valid orgId.
    await db.user.update({
      where: { id: authUser.userId },
      data: { lastActiveOrgId: newOrgId || null },
    });
    newAccessToken = signAccessToken({
      userId: authUser.userId,
      email: authUser.email,
      role: authUser.role,
      platformRole: authUser.platformRole,
      orgId: newOrgId,
    });
    // Set the new access token cookie. We keep the existing refresh token
    // (org switching doesn't require re-authentication). When the refresh
    // token eventually rotates, rotateRefreshToken will read
    // lastActiveOrgId (which we just persisted) and issue the new access
    // token with the correct orgId.
    const { cookies } = await import("next/headers");
    const cookieStore = await cookies();
    const refreshToken = cookieStore.get("drugos_refresh")?.value;
    if (refreshToken) {
      await setAuthCookies(newAccessToken, refreshToken);
    }
    // BE-029 ROOT FIX: mark the org-switch audit log as CRITICAL. Switching
    // the active org changes the user's data access scope — they can now
    // read a different org's projects, hypotheses, and evidence packages.
    // This is a security-relevant action that should be audit-logged as
    // critical (so a failed audit-log write aborts the request, ensuring
    // the action is always auditable). Other security-relevant actions
    // (password change, 2FA disable, billing change) are already marked
    // critical. FDA 21 CFR Part 11 requires complete audit trails for
    // security events.
    const orgSwitchAudit = await writeAuditLog({
      user: { ...authUser, orgId: newOrgId },
      action: "active_org_switched",
      resource: `user:${authUser.userId}`,
      metadata: { previousOrgId: authUser.orgId, newOrgId: newOrgId },
      critical: true,
    });
    if (!orgSwitchAudit.ok) {
      // Critical audit failed — abort the request. We've already persisted
      // `lastActiveOrgId` and set the new access cookie, so we must roll
      // those back to maintain the invariant that the audit log captures
      // every successful org switch.
      await db.user.update({
        where: { id: authUser.userId },
        data: { lastActiveOrgId: authUser.orgId || null },
      }).catch(() => {
        // best-effort rollback
      });
      // Clear the new access cookie we just set (restore the old one).
      const { cookies } = await import("next/headers");
      const cookieStore = await cookies();
      const oldAccessToken = signAccessToken({
        userId: authUser.userId,
        email: authUser.email,
        role: authUser.role,
        platformRole: authUser.platformRole,
        orgId: authUser.orgId,
      });
      const refreshToken = cookieStore.get("drugos_refresh")?.value;
      if (refreshToken) {
        await setAuthCookies(oldAccessToken, refreshToken);
      }
      return internalError("Failed to record org switch in audit log. Org switch has been rolled back.");
    }
  }

  const actionTypes: string[] = [];
  if (Object.keys(data).length > 0) actionTypes.push("profile_update");
  if (switchingOrg) actionTypes.push("active_org_switched");

  await writeAuditLog({
    user: authUser,
    action: actionTypes.join(","),
    resource: `user:${updated.id}`,
    metadata: { fields: Object.keys(data), switchedOrg: switchingOrg },
  });

  return NextResponse.json({
    user: updated,
    // BE-079: Return the new activeOrganizationId so the client can
    // update its local state without re-fetching /api/auth/me.
    //
    // BE-041 NOTE: This field is intentionally included for browser
    // clients (cookie-based auth). The new access token itself is set as
    // an HttpOnly cookie via setAuthCookies — it is NOT in the JSON body.
    // Non-browser clients (curl, Postman, API keys) will NOT receive the
    // new access token from this endpoint; they should use the refresh
    // flow (/api/auth/refresh) to obtain a new access token after the
    // org switch. API-key auth reads orgId from the DB's lastActiveOrgId
    // (which we just updated), so non-browser clients using API keys do
    // NOT need to call PATCH /api/auth/me at all — they can just
    // directly call /api/projects and the API key auth path will pick up
    // the new orgId from the DB. This is documented for clarity; the
    // inconsistency is cosmetic.
    activeOrganizationId: switchingOrg
      ? body.activeOrganizationId
      : (authUser.orgId || null),
    orgSwitched: switchingOrg,
  });
}
