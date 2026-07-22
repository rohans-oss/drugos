import { NextRequest, NextResponse } from "next/server";
import { requireAuth, notFound, requireCsrfOrSend } from "@/lib/api-helpers";
import { getProject, createHypothesis } from "@/lib/services/projects";
import { db } from "@/lib/db";

/**
 * FE-017 ROOT FIX: Project endpoints previously checked ONLY that the
 * project's orgId matched the user's orgId — they did NOT check the
 * Project.visibility field. A "private" project (intended to be visible
 * only to its owner) was readable by ANY org member. POST (add hypothesis)
 * had the same hole — any org member could add hypotheses to a project
 * they don't own. There was also no OrganizationMember.role check, so a
 * viewer (read-only) could mutate projects.
 *
 * Root fix:
 *   - GET: enforce visibility. Private → only owner + admin/owner roles.
 *          Org → any org member. Public → any authenticated user.
 *   - POST: require OrganizationMember.role in {owner, admin, member,
 *          developer, researcher, data-scientist, pi, business-dev}. Viewer
 *          and billing are read-only and CANNOT create hypotheses. Also
 *          enforce visibility (a non-owner cannot add hypotheses to a
 *          private project they can't even see).
 *
 * FE-011: CSRF protection applied to POST.
 */

// OrganizationMember roles that are allowed to create hypotheses / mutate
// project content. Viewer and billing are read-only by design.
const PROJECT_WRITE_ROLES = new Set([
  "owner",
  "admin",
  "member",
  "developer",
  "researcher",
  "data-scientist",
  "pi",
  "business-dev",
]);

export async function GET(_req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const auth = await requireAuth();
  if (auth.user === null) return auth.response;
  const { id } = await params;
  const project = await getProject(id);
  if (!project) return notFound("Project not found");

  // BE-061 ROOT FIX: The previous "public" visibility check returned the
  // project to ANY authenticated user without verifying org membership.
  // This meant a user in Org A could read any "public" project in Org B,
  // potentially leaking competitor research data if "public" was set by
  // mistake. The "public" visibility now means "visible to any member of
  // the owning organization" — NOT "visible to anyone on the internet".
  // Cross-org visibility must be explicitly enabled via a separate
  // "published" visibility level (future feature). For now, ALL projects
  // require org membership to read.
  if (project.organizationId !== auth.user.orgId) {
    return NextResponse.json({ error: "forbidden", message: "Project belongs to a different organization." }, { status: 403 });
  }
  // BE-061: Additional check — "public" projects are readable by any org
  // member; "private" projects are only readable by the owner + admin/owner
  // roles; "org" projects are readable by any org member (same as public).
  if (project.visibility === "private") {
    const isOwner = project.ownerId === auth.user.userId;
    const isAdminOrOwner = auth.user.role === "admin" || auth.user.role === "owner";
    if (!isOwner && !isAdminOrOwner) {
      return NextResponse.json(
        { error: "forbidden", message: "This project is private to its owner." },
        { status: 403 }
      );
    }
  }
  return NextResponse.json(project);
}

export async function POST(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  // FE-011: CSRF protection on every state-changing route.
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  const auth = await requireAuth();
  if (auth.user === null) return auth.response;
  const { id } = await params;
  const project = await db.project.findUnique({ where: { id } });
  if (!project) return notFound("Project not found");
  if (project.organizationId !== auth.user.orgId) {
    return NextResponse.json({ error: "forbidden" }, { status: 403 });
  }

  // FE-017: visibility check on write too.
  if (project.visibility === "private") {
    const isOwner = project.ownerId === auth.user.userId;
    const isAdminOrOwner = auth.user.role === "admin" || auth.user.role === "owner";
    if (!isOwner && !isAdminOrOwner) {
      return NextResponse.json(
        { error: "forbidden", message: "This project is private to its owner." },
        { status: 403 }
      );
    }
  }

  // FE-017: OrganizationMember.role check.
  if (auth.user.role !== "owner") {
    const membership = await db.organizationMember.findFirst({
      where: { userId: auth.user.userId, organizationId: project.organizationId },
      select: { role: true },
    });
    if (!membership || !PROJECT_WRITE_ROLES.has(membership.role)) {
      return NextResponse.json(
        {
          error: "forbidden",
          message: `Your organization role (${membership?.role || "none"}) cannot modify project content.`,
        },
        { status: 403 }
      );
    }
  }

  let body: { title: string; drugName: string; diseaseName: string; notes?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_request", message: "Invalid JSON" }, { status: 400 });
  }
  if (!body.title || !body.drugName || !body.diseaseName) {
    return NextResponse.json({ error: "bad_request", message: "title, drugName, diseaseName required" }, { status: 400 });
  }
  const hyp = await createHypothesis({
    projectId: id,
    title: body.title,
    drugName: body.drugName,
    diseaseName: body.diseaseName,
    createdById: auth.user.userId,
    notes: body.notes,
  });
  return NextResponse.json(hyp, { status: 201 });
}
