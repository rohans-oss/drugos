/**
 * Projects & collaboration service.
 * Backed by Prisma/SQLite. Real CRUD operations on projects, hypotheses,
 * comments, and activity feed.
 */

import { db } from "@/lib/db";

export interface CreateProjectInput {
  name: string;
  description?: string;
  ownerId: string;
  organizationId: string;
  visibility?: "private" | "org" | "public";
  tags?: string[];
}

export async function createProject(input: CreateProjectInput) {
  return db.project.create({
    data: {
      name: input.name,
      description: input.description || null,
      ownerId: input.ownerId,
      organizationId: input.organizationId,
      visibility: input.visibility || "private",
      tags: (input.tags || []).join(","),
    },
  });
}

export async function listProjects(orgId: string) {
  return db.project.findMany({
    where: { organizationId: orgId },
    orderBy: { updatedAt: "desc" },
    include: {
      _count: { select: { hypotheses: true, comments: true } },
    },
  });
}

export async function getProject(id: string) {
  return db.project.findUnique({
    where: { id },
    include: {
      hypotheses: { orderBy: { updatedAt: "desc" } },
      comments: { orderBy: { createdAt: "desc" }, take: 50 },
      activities: { orderBy: { createdAt: "desc" }, take: 50 },
    },
  });
}

export interface CreateHypothesisInput {
  projectId: string;
  title: string;
  drugName: string;
  diseaseName: string;
  createdById: string;
  notes?: string;
}

export async function createHypothesis(input: CreateHypothesisInput) {
  const hyp = await db.hypothesis.create({
    data: {
      projectId: input.projectId,
      title: input.title,
      drugName: input.drugName,
      diseaseName: input.diseaseName,
      createdById: input.createdById,
      notes: input.notes,
      status: "draft",
    },
  });
  // BE-050 ROOT FIX (v115, LOW): the previous code wrote `actorName: "system"`
  // to ProjectActivity — losing audit-trail attribution. For a pharma
  // platform where hypothesis ownership is a compliance requirement
  // (FDA 21 CFR Part 11 — "who did what when"), this was an audit trail
  // gap. The activity feed showed "system created hypothesis X" instead
  // of "Dr. Smith created hypothesis X".
  //
  // ROOT FIX: derive the actorName from the User table (same pattern as
  // `addComment` below). If the user has a name set, use it. Otherwise
  // fall back to their email. The fetch is ONE indexed lookup by id —
  // negligible cost compared to the hypothesis write itself.
  const creator = await db.user.findUnique({
    where: { id: input.createdById },
    select: { name: true, email: true },
  });
  const actorName = creator?.name || creator?.email || "unknown";
  await db.projectActivity.create({
    data: {
      projectId: input.projectId,
      type: "hypothesis_created",
      actorName,
      summary: `Created hypothesis "${input.title}" (${input.drugName} → ${input.diseaseName})`,
    },
  });
  return hyp;
}

export async function listHypotheses(projectId: string) {
  return db.hypothesis.findMany({
    where: { projectId },
    orderBy: { updatedAt: "desc" },
  });
}

/**
 * FE-073 ROOT FIX: Comment impersonation.
 *
 * Previously: addComment(projectId, authorName, body) stored whatever
 * `authorName` the caller provided. The /api/projects/[id]/comments route
 * passed `body.authorName || auth.user.email` — so a client sending
 * authorName="Dr. Smith (PI)" would post a damaging comment attributed to
 * "Dr. Smith (PI)" even though the actual commenter was someone else.
 *
 * Root fix: the service signature now takes `userId` (NOT `authorName`).
 * The authorName is ALWAYS derived from the User table at write time:
 *   - If User.name is set, use it.
 *   - Otherwise fall back to User.email.
 * The client-supplied authorName is silently ignored at every layer.
 *
 * The Comment model already has a `userId` field (nullable in schema) — we
 * now always populate it, so attribution is audit-traceable even if the
 * user later renames themselves.
 *
 * Backward-compat: the old `(projectId, authorName, body)` 3-arg signature
 * is preserved as a runtime guard — if a caller passes a string in the
 * second-arg position (legacy), we throw. This forces every call site to
 * migrate to the new signature; we never silently accept an
 * attacker-controlled name.
 */
export async function addComment(
  projectId: string,
  userId: string,
  body: string
) {
  // Defense-in-depth: if a legacy caller passes authorName as the 2nd arg
  // (which would be a display name like "Dr. Smith (PI)", not a userId),
  // reject loudly rather than silently storing an attacker-controlled name.
  //
  // UserIds in this system are CUIDs (24 lowercase alphanumeric chars,
  // e.g. "clxxxxxxxxxxxxxxxxxxxxxx") or UUIDs (36 chars with hyphens).
  // Display names contain spaces, punctuation, and/or uppercase letters.
  // We reject any string that:
  //   - is less than 20 chars (CUIDs are 24; UUIDs are 36)
  //   - contains chars other than lowercase alphanumeric or hyphens
  //     (names have spaces, periods, parentheses, commas, etc.)
  if (
    typeof userId !== "string" ||
    userId.length < 20 ||
    /[^a-z0-9-]/.test(userId)
  ) {
    throw new Error(
      "addComment: second argument must be a userId (cuid), not an authorName. " +
        "FE-073 root fix: authorName is now derived from the User table."
    );
  }

  const user = await db.user.findUnique({
    where: { id: userId },
    select: { name: true, email: true },
  });
  if (!user) {
    throw new Error(`addComment: user ${userId} not found`);
  }
  // Always derive authorName from the DB — never trust the caller.
  const authorName = user.name || user.email;

  const comment = await db.comment.create({
    data: { projectId, userId, authorName, body },
  });
  await db.projectActivity.create({
    data: {
      projectId,
      type: "comment_added",
      actorName: authorName,
      summary: body.slice(0, 120),
    },
  });
  return comment;
}

export async function listComments(projectId: string) {
  return db.comment.findMany({
    where: { projectId },
    orderBy: { createdAt: "desc" },
  });
}

export async function listProjectActivities(projectId: string) {
  return db.projectActivity.findMany({
    where: { projectId },
    orderBy: { createdAt: "desc" },
    take: 100,
  });
}
