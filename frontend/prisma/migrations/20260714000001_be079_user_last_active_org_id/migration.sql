-- BE-079 REAL ROOT FIX (v2): Add lastActiveOrgId to User.
--
-- The prior BE-079 "fix" added a PATCH /api/auth/me endpoint that issued a
-- new access token with the requested orgId — but it did NOT persist the
-- orgId anywhere. When the access token expired (15 min), rotateRefreshToken
-- (lib/auth/server.ts) issued a new access token WITHOUT orgId (it only
-- passed userId/email/role to signAccessToken). The user lost their org
-- context entirely — every org-scoped query returned 403.
--
-- This affected ALL users after 15 minutes, not just those who switched
-- orgs. The fix was comments-only — exactly the failure mode the audit
-- warned about.
--
-- Real root fix: persist `lastActiveOrgId` on the User row. Set it at login
-- (from the first membership), update it when the user switches org via
-- PATCH /api/auth/me, and READ it in rotateRefreshToken so the refreshed
-- access token carries the correct orgId.
--
-- The column is nullable so existing User rows remain valid (they get
-- backfilled on next login). No foreign-key constraint is added because
-- the active org can be any org the user is a member of, and we don't
-- want a CASCADE DELETE to null out the field when an org is deleted
-- (the user should be re-prompted to pick a new active org instead).

ALTER TABLE "User" ADD COLUMN "lastActiveOrgId" TEXT;

-- Backfill: for every user, set lastActiveOrgId to their FIRST membership
-- (oldest by joinedAt). This matches the login route's behavior of picking
-- the first membership as the default active org. Users who switch orgs
-- later will have their lastActiveOrgId updated by PATCH /api/auth/me.
UPDATE "User" u
SET "lastActiveOrgId" = (
  SELECT om."organizationId"
  FROM "OrganizationMember" om
  WHERE om."userId" = u."id"
  ORDER BY om."joinedAt" ASC
  LIMIT 1
)
WHERE "lastActiveOrgId" IS NULL
  AND EXISTS (
    SELECT 1 FROM "OrganizationMember" om WHERE om."userId" = u."id"
  );
