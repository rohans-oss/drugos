import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { getAuthenticatedUser } from "@/lib/auth/server";
import { parsePagination, buildPaginatedResponse } from "@/lib/pagination";

/**
 * GET /api/auth/activity
 *
 * Returns the most recent audit-log entries authored BY the currently
 * authenticated user. Unlike /api/audit-logs (which is admin-only and
 * returns all entries across the org), this endpoint is scoped to the
 * calling user and is used by the Security Settings screen to show
 * "recent account activity" — logins, password changes, 2FA enrollments.
 *
 * FE-047 ROOT FIX: previously `take: 20` with no offset/pagination —
 * users could not see activity older than the 20 most recent entries.
 * Now accepts `limit` (capped at 100) and `offset` query params and
 * returns the standard paginated envelope.
 */
export async function GET(req: NextRequest) {
  const user = await getAuthenticatedUser();
  if (!user) {
    return NextResponse.json({ error: "unauthorized", message: "Authentication required" }, { status: 401 });
  }

  const page = parsePagination(req.nextUrl.searchParams);
  const where = { userId: user.userId };
  const [items, total] = await Promise.all([
    db.auditLog.findMany({
      where,
      orderBy: { createdAt: "desc" },
      take: page.limit,
      skip: page.offset,
    }),
    db.auditLog.count({ where }),
  ]);
  return NextResponse.json(buildPaginatedResponse(items, total, page));
}
