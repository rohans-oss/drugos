import { NextRequest, NextResponse } from "next/server";
import { requireAuth, badRequest, requireCsrfOrSend } from "@/lib/api-helpers";
import { createProject, listProjects } from "@/lib/services/projects";
import { db } from "@/lib/db";

/**
 * FE-044 ROOT FIX: project creation used to check `User.role` (a global
 * platform role like "researcher", "data-scientist", "pi", "business-dev").
 * But the correct granularity for org-scoped permissions is
 * `OrganizationMember.role` (owner | admin | member | viewer | billing).
 * A user with User.role="researcher" who has been DEMOTED to
 * OrgMember.role="viewer" (read-only) in their org could still create
 * projects in that org — a privilege escalation within the org.
 *
 * Root fix: GET and POST both:
 *   1. requireAuth (any logged-in user).
 *   2. Load the caller's OrganizationMember row for their active org.
 *   3. requireRole on the ORG-MEMBER role, not the USER role.
 *
 * Org-member roles that may create/list projects: owner, admin, member.
 * `viewer` and `billing` are read-only / financial-only and may NOT.
 *
 * This is intentionally a per-route fix — a full RBAC refactor would touch
 * every endpoint and is out of scope for this issue. The pattern is now
 * established for replication elsewhere.
 */

/** Org-member roles allowed to list/create projects. */
const PROJECT_ROLES = new Set(["owner", "admin", "member"]);

async function requireOrgProjectRole() {
  const auth = await requireAuth();
  if (auth.user === null) return { user: null, response: auth.response, member: null } as const;
  if (!auth.user.orgId) {
    return {
      user: null,
      response: badRequest("User has no active organization"),
      member: null,
    } as const;
  }
  const member = await db.organizationMember.findUnique({
    where: {
      userId_organizationId: {
        userId: auth.user.userId,
        organizationId: auth.user.orgId,
      },
    },
    select: { role: true },
  });
  if (!member) {
    return {
      user: null,
      response: NextResponse.json(
        { error: "forbidden", message: "You are not a member of this organization." },
        { status: 403 }
      ),
      member: null,
    } as const;
  }
  // Allow platform-level admin/owner as superusers.
  const isPlatformSuperuser = auth.user.role === "admin" || auth.user.role === "owner";
  if (!isPlatformSuperuser && !PROJECT_ROLES.has(member.role)) {
    return {
      user: null,
      response: NextResponse.json(
        {
          error: "forbidden",
          message: `Your organization role (${member.role}) is not permitted to manage projects. Required: owner, admin, or member.`,
        },
        { status: 403 }
      ),
      member: null,
    } as const;
  }
  return { user: auth.user, response: null, member } as const;
}

export async function GET() {
  const auth = await requireOrgProjectRole();
  if (auth.user === null) return auth.response;
  const projects = await listProjects(auth.user.orgId!);
  return NextResponse.json({ items: projects });
}

export async function POST(req: NextRequest) {
  // FE-011: CSRF protection on every state-changing route.
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  const auth = await requireOrgProjectRole();
  if (auth.user === null) return auth.response;
  let body: { name: string; description?: string; visibility?: "private" | "org" | "public"; tags?: string[] };
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON body");
  }
  if (!body.name || body.name.trim().length < 1) {
    return badRequest("Project name is required");
  }
  const project = await createProject({
    name: body.name.trim(),
    description: body.description,
    ownerId: auth.user.userId,
    organizationId: auth.user.orgId!,
    visibility: body.visibility,
    tags: body.tags,
  });
  return NextResponse.json(project, { status: 201 });
}
