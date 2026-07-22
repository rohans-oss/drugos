import { NextRequest, NextResponse } from "next/server";
import { requireAuth } from "@/lib/api-helpers";
import { buildPaginatedResponse } from "@/lib/pagination";
import { db } from "@/lib/db";
// TASK-272: Zod validation on query params.
import { NotificationsQuery } from "@/lib/zod-schemas";

/**
 * GET /api/notifications
 *
 * Returns the authenticated user's notifications, wired to the REAL
 * Notification Prisma model (db.notification.findMany). Task 263's
 * "currently returns mock notifications" finding was a stale
 * description — the previous fix already wired the route to the real
 * table. This commit adds:
 *
 *   1. TASK-272: Zod validation on query params (limit, offset). The
 *      previous code used `parsePagination` which accepted NaN —
 *      `?limit=abc` would silently return all rows (no cap).
 *
 *   2. TASK-280: returns 503 when the notifications table is
 *      unreachable, so the monitoring layer can detect the outage.
 *
 * Query params (Zod-validated):
 *   - limit: max rows to return (default 50, capped at 100).
 *   - offset: pagination offset (default 0).
 *
 * BE-042 ROOT FIX (v115, LOW): the previous code used a single
 * `groupBy` on `readAt` to compute total + unread counts. Prisma's
 * `groupBy` on a nullable DateTime field produces ONE bucket per
 * distinct timestamp value — for a user with 10K read notifications,
 * that's 10K buckets returned from the DB. The query was O(N) in
 * distinct timestamps, defeating the "single round-trip" goal.
 *
 * ROOT FIX: replace the groupBy with TWO count queries — one for
 * the total count and one for the unread count (readAt IS NULL).
 * Two round-trips, but each is O(1) (PostgreSQL COUNT(*) uses an
 * index-only scan). This is dramatically cheaper than groupBy for
 * power users with many read notifications.
 */
export async function GET(req: NextRequest) {
  const auth = await requireAuth();
  if (auth.user === null) return auth.response;

  // TASK-272: validate query params with Zod. The schema coerces strings
  // to numbers and applies bounds. Falls back to defaults if params are
  // absent; returns 400 on validation failure.
  const parseResult = NotificationsQuery.safeParse({
    limit: req.nextUrl.searchParams.get("limit") ?? undefined,
    offset: req.nextUrl.searchParams.get("offset") ?? undefined,
  });
  if (!parseResult.success) {
    return NextResponse.json(
      {
        error: "bad_request",
        message: "Invalid query parameters.",
        issues: parseResult.error.issues.map((iss) => ({
          path: iss.path.join("."),
          message: iss.message,
        })),
      },
      { status: 400 }
    );
  }
  const { limit, offset } = parseResult.data;
  const page = { limit, offset };

  const where = { userId: auth.user.userId };

  try {
    // BE-042 ROOT FIX: two count queries instead of groupBy. Each
    // count is O(1) at the DB (index-only scan on userId) — much
    // cheaper than the previous groupBy which was O(N) in distinct
    // readAt timestamps.
    const [items, total, unread] = await Promise.all([
      db.notification.findMany({
        where,
        orderBy: { createdAt: "desc" },
        take: page.limit,
        skip: page.offset,
      }),
      db.notification.count({ where }),
      db.notification.count({ where: { ...where, readAt: null } }),
    ]);

    return NextResponse.json({ ...buildPaginatedResponse(items, total, page), unread });
  } catch (e) {
    // TASK-280: return 503 when the notifications table is unreachable
    // so the monitoring layer can detect the outage and alert.
    console.error("[NOTIFICATIONS] DB error:", e);
    return NextResponse.json(
      {
        error: "service_unavailable",
        message: "Notifications database is currently unavailable.",
      },
      { status: 503 }
    );
  }
}
