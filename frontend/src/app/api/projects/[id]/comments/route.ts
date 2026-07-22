import { NextRequest, NextResponse } from "next/server";
import { requireAuth, notFound, badRequest, requireCsrfOrSend, writeAuditLog } from "@/lib/api-helpers";
import { addComment } from "@/lib/services/projects";
import { db } from "@/lib/db";
// TASK-268: notification trigger for new project comments.
import { notifyProjectComment } from "@/lib/services/notifications";

/**
 * POST /api/projects/[id]/comments
 * Body: { body: string }   (authorName is intentionally NOT accepted)
 *
 * FE-073 ROOT FIX: Comment impersonation.
 *
 * Previously the route accepted `body.authorName` from the client and
 * passed it through to addComment(). A user could post a comment
 * attributed to "Dr. Smith (PI)" by simply sending that string in the
 * request body — the actual commenter's identity was ignored.
 *
 * Root fix: the route now IGNORES any client-supplied authorName and
 * passes only auth.user.userId to addComment(). The service derives
 * authorName from the User table (User.name || User.email) so attribution
 * is always truthful and audit-traceable.
 *
 * TASK-268 ROOT FIX: After the comment is added, fire a notification to
 * all project members (except the commenter) via notifyProjectComment.
 * The notification is NON-BLOCKING — if it fails, the comment still
 * posted. This closes the gap where the notification bell always showed
 * "0 unread" even when a teammate commented on your project.
 *
 * TASK-267: Audit log the comment creation (critical — comment
 * impersonation is a compliance issue under FDA 21 CFR Part 11).
 */
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

  let body: { body?: string; authorName?: string };
  try {
    body = await req.json();
  } catch {
    return badRequest("Invalid JSON body");
  }

  // FE-073: `authorName` from the client is intentionally ignored. We do
  // NOT echo it back even if the client insists — the service derives it
  // from the User table.
  if (!body.body || body.body.trim().length < 1) {
    return badRequest("Comment body required");
  }
  // Hard cap comment length to prevent storage abuse.
  const commentBody = body.body.trim().slice(0, 10000);

  try {
    const comment = await addComment(id, auth.user.userId, commentBody);

    // TASK-267: critical audit log for comment creation.
    await writeAuditLog({
      user: auth.user,
      action: "project_comment_create",
      resource: `project:${id}:comment:${comment.id}`,
      metadata: { commentLength: commentBody.length },
      critical: true,
    }).catch(() => {
      // Non-critical — the comment posted, we just couldn't audit it.
      // The dead-letter table captures the failed audit write.
    });

    // TASK-268: fire the notification trigger. NON-BLOCKING — the
    // comment already posted; a notification failure must not roll it
    // back. The helper logs to stderr on failure.
    await notifyProjectComment(id, auth.user.userId, commentBody);

    return NextResponse.json(comment, { status: 201 });
  } catch (e: unknown) {
    // addComment throws if the user lookup fails — surface as 401 since the
    // auth token resolved but the user no longer exists in the DB.
    if (e instanceof Error && /not found/i.test(e.message)) {
      return NextResponse.json(
        { error: "unauthorized", message: "Authenticated user no longer exists." },
        { status: 401 }
      );
    }
    console.error("addComment failed:", e);
    return NextResponse.json(
      { error: "internal_error", message: "Failed to add comment." },
      { status: 500 }
    );
  }
}
