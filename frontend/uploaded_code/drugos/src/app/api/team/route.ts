import { NextResponse } from "next/server";
import { requireAuth, badRequest } from "@/lib/api-helpers";
import { db } from "@/lib/db";

/**
 * GET /api/team — list the members of the current user's organization.
 *
 * Returns each member's name, email, role, status, and last-login timestamp.
 * Useful for the Team Members page in the app shell.
 */
export async function GET() {
  const auth = await requireAuth();
  if (auth.user === null) return auth.response;
  if (!auth.user.orgId) return badRequest("No active organization");

  const memberships = await db.organizationMember.findMany({
    where: { organizationId: auth.user.orgId },
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
  });

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

  return NextResponse.json({ items, total: items.length });
}
