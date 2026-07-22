import { NextRequest, NextResponse } from "next/server";
import { requireAuth, badRequest } from "@/lib/api-helpers";
import { db } from "@/lib/db";
// TASK-272: Zod validation on query params.
import { TeamQuery } from "@/lib/zod-schemas";

/**
 * GET /api/team — list the members of the current user's organization.
 *
 * Returns each member's name, email, role, status, and last-login timestamp.
 * Wired to the REAL OrganizationMember Prisma model
 * (db.organizationMember.findMany). Task 266's "currently returns mock
 * team data" finding was a stale description — the previous fix already
 * wired the route to the real table. This commit adds:
 *
 *   1. TASK-272: Zod validation on query params (limit, offset). The
 *      previous `parsePagination` accepted NaN — `?limit=abc` would
 *      silently return all rows.
 *
 *   2. TASK-280: returns 503 when the DB is unreachable, so the
 *      monitoring layer can detect the outage and alert.
 *
 * Security:
 *   - The `where` clause is scoped to `auth.user.orgId` — a user can
 *     only see members of their OWN org. A forged `?orgId=` query
 *     param is ignored (the route does not accept it).
 *   - Email is included — within an org, members have a legitimate
 *     need-to-know for collaboration. Cross-tenant access is gated by
 *     the orgId scoping.
 */
export async function GET(req: NextRequest) {
  const auth = await requireAuth();
  if (auth.user === null) return auth.response;
  if (!auth.user.orgId) return badRequest("No active organization");

  // TASK-272: validate query params with Zod.
  const parseResult = TeamQuery.safeParse({
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

  const where = { organizationId: auth.user.orgId };

  try {
    const [memberships, total] = await Promise.all([
      db.organizationMember.findMany({
        where,
        include: {
          user: {
            select: {
              id: true,
              email: true,
              name: true,
              role: true,
              title: true,
              bio: true,
              status: true,
              lastLoginAt: true,
              createdAt: true,
            },
          },
        },
        orderBy: { joinedAt: "asc" },
        take: limit,
        skip: offset,
      }),
      db.organizationMember.count({ where }),
    ]);

    const items = memberships.map((m) => ({
      id: m.user.id,
      name: m.user.name || m.user.email,
      email: m.user.email,
      role: m.user.role,
      orgRole: m.role, // owner | admin | member | viewer | billing
      title: m.user.title,
      bio: m.user.bio,
      status: m.user.status,
      lastLoginAt: m.user.lastLoginAt,
      joinedAt: m.joinedAt,
    }));

    return NextResponse.json({ items, total, limit, offset });
  } catch (e) {
    // TASK-280: return 503 on DB outage so monitoring can detect it.
    console.error("[TEAM] DB error:", e);
    return NextResponse.json(
      {
        error: "service_unavailable",
        message: "Team database is currently unavailable.",
      },
      { status: 503 }
    );
  }
}
