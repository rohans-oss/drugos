-- TASK-261 / TASK-269 ROOT FIX: Introduce a SEPARATE `platformRole` column on
-- "User" that gates access to /api/admin/* (platform operator console).
--
-- WHY A SEPARATE COLUMN (not a new UserRole enum value):
--   The previous architecture overloaded `UserRole` for two distinct concerns:
--   (a) functional role inside the org (researcher, pi, billing, admin, owner)
--   (b) platform-operator status (SaaS staff).
--   Mixing them meant `role === 'owner'` was interpreted as "org owner" in
--   some routes and "platform superuser" in others — any user promoted to
--   `owner` could enumerate every user in every org and read every audit log
--   across all tenants (Task 261 audit finding). The prior `platformOwner`
--   enum-value patch reduced the blast radius but kept the coupling: an
--   `admin` (functional) could not be distinguished from a `platformOwner`
--   (operator) at the JWT level, and granting platform access required
--   changing `role`, which changed in-app permissions as a side effect.
--
--   The clean fix is a SEPARATE column. `role` keeps its existing meaning.
--   `platformRole` is the orthogonal operator flag: `none` (default) for
--   every regular user, `admin` for SaaS operator staff. The two fields are
--   independently grantable and independently revocable (OWASP ASVS V1.2
--   "Separation of Duties" for multi-tenant SaaS).
--
--   Only `platformRole === 'admin'` can access /api/admin/* routes — enforced
--   by the new `requirePlatformAdmin()` middleware. The `role === 'owner'` /
--   `platformOwner` checks in api-helpers.ts are KEPT for backwards
--   compatibility with org-scoped admin routes (/api/audit-logs, /api/team,
--   /api/billing/*) but NO LONGER grant access to /api/admin/*.
--
--   platformRole is settable ONLY via direct DB access by the SaaS operator.
--   No API route can grant it (the PATCH /api/admin/users Zod schema does
--   not accept `platformRole` in the request body). Fail-closed.
--
-- NON-BREAKING: This migration is purely additive (new enum type + new
-- nullable-with-default column + new index). Existing rows get
-- `platformRole = 'none'` automatically. No data backfill is required.

-- Create the PlatformRole enum type.
CREATE TYPE "PlatformRole" AS ENUM ('none', 'admin');

-- Add the column with a default of 'none' so every existing user is
-- immediately denied platform-admin access (fail-closed). The default
-- also applies to future INSERTs that omit the column.
ALTER TABLE "User" ADD COLUMN "platformRole" "PlatformRole" NOT NULL DEFAULT 'none';

-- Index for fast lookup of all platform admins
-- (SELECT * FROM "User" WHERE "platformRole" = 'admin').
CREATE INDEX "User_platformRole_idx" ON "User"("platformRole");
