-- FE-005 ROOT FIX: Add organizationId to AuditLog for multi-tenant isolation.
-- Previously the AuditLog table had no organizationId column, so an admin of
-- Org A could read every audit log system-wide by hitting GET /api/audit-logs
-- (no WHERE filter). This migration adds the column (nullable so existing
-- rows remain valid) and an index on it so per-tenant queries are not full
-- table scans.

ALTER TABLE "AuditLog" ADD COLUMN "organizationId" TEXT;

-- Index for per-tenant audit-log queries.
CREATE INDEX "AuditLog_organizationId_idx" ON "AuditLog"("organizationId");

-- Add foreign key constraint to Organization. Existing rows have NULL
-- organizationId, which is allowed (the column is nullable). New rows are
-- stamped with the actor's orgId by writeAuditLog.
ALTER TABLE "AuditLog"
  ADD CONSTRAINT "AuditLog_organizationId_fkey"
  FOREIGN KEY ("organizationId") REFERENCES "Organization"("id")
  ON DELETE SET NULL ON UPDATE CASCADE;
