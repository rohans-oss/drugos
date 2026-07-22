-- ============================================================================
-- Drug Repurposing ETL Platform — Rollback 008 Drug.is_globally_approved
-- Migration: 008_drug_is_globally_approved_rollback.sql
-- Description: Reverses the ALTER TABLE changes from
--              008_drug_is_globally_approved.sql.
--
-- ROOT-CAUSE FIX (audit P1-28): rollback sidecar for 008.
-- ============================================================================

BEGIN;

-- Drop the partial index created by 008.
DROP INDEX IF EXISTS idx_drugs_is_globally_approved;

-- Drop the column added by 008.
ALTER TABLE drugs DROP COLUMN IF EXISTS is_globally_approved;

-- v76 ROOT FIX (T-037 — schema_version rollback consistency):
-- Delete the schema_version row inserted by 008 (version=8). All
-- rollbacks now follow the same convention: delete the version row so
-- schema_version always reflects the CURRENT schema state. The
-- _migration_history table retains the full audit trail.
DELETE FROM schema_version WHERE version = 8;

COMMIT;
