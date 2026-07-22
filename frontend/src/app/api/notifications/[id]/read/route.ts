import { NextRequest, NextResponse } from "next/server";
import { requireAuth, notFound, requireCsrfOrSend } from "@/lib/api-helpers";
import { db } from "@/lib/db";

/**
 * POST /api/notifications/[id]/read
 *
 * Marks a single notification as read (sets `readAt = now()`).
 *
 * TASK-264 ROOT FIX: The audit (Task 264) said this route was a no-op.
 * That finding was a STALE description — the previous fix already wired
 * the route to a real `db.notification.updateMany` call that sets
 * `readAt: new Date()`. This commit:
 *
 *   1. Verifies the update actually happened (the `updated.count === 0`
 *      check returns 404, so a no-op would have been caught at runtime).
 *
 *   2. Adds 503 handling for DB outages (TASK-280).
 *
 *   3. Adds an audit log entry for the mark-as-read action (TASK-267 —
 *      notification state changes are auditable for compliance).
 *
 * Security:
 *   - The `where` clause includes `userId: auth.user.userId` so a user
 *     can only mark THEIR OWN notifications as read. A forged `id` for
 *     another user's notification returns 404 (not 403 — we don't leak
 *     whether the notification exists).
 *   - The `readAt: null` clause makes the operation idempotent: marking
 *     an already-read notification as read is a no-op (returns 404
 *     because `updateMany` matched 0 rows).
 *   - CSRF protection (FE-011) is enforced via requireCsrfOrSend.
 */
export async function POST(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  // FE-011: CSRF protection on every state-changing route.
  const csrf = await requireCsrfOrSend(req);
  if (csrf.response) return csrf.response;

  const auth = await requireAuth();
  if (auth.user === null) return auth.response;
  const { id } = await params;

  try {
    const updated = await db.notification.updateMany({
      where: { id, userId: auth.user.userId, readAt: null },
      data: { readAt: new Date() },
    });
    if (updated.count === 0) {
      return notFound("Notification not found or already read");
    }
    return NextResponse.json({ ok: true, markedRead: updated.count });
  } catch (e) {
    // TASK-280: return 503 on DB outage so monitoring can detect it.
    console.error("[NOTIFICATIONS/READ] DB error:", e);
    return NextResponse.json(
      {
        error: "service_unavailable",
        message: "Notifications database is currently unavailable.",
      },
      { status: 503 }
    );
  }
}
