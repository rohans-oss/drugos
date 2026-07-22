-- ============================================================================
-- Drug Repurposing ETL Platform — Rollback 006 Drug Withdrawn/Safety Columns
-- Migration: 006_drug_withdrawn_safety_columns_rollback.sql
-- Description: Reverses the ALTER TABLE changes from
--              006_drug_withdrawn_safety_columns.sql.
--
-- v22 ROOT FIX (audit P2-10 / Section 9): rollback sidecar for 006.
-- ============================================================================

BEGIN;

-- Drop indexes created by 006.
DROP INDEX IF EXISTS idx_drugs_is_withdrawn;
DROP INDEX IF EXISTS idx_drugs_clinical_status;
DROP INDEX IF EXISTS idx_drugs_cas_number;

-- Drop the trigger that 006 created (IF EXISTS guards make this safe).
DROP TRIGGER IF EXISTS trg_drugs_sync_withdrawn ON drugs;
-- v73 ROOT FIX (T-007 — rollback dropped a function name that was never
-- created):
--   The previous rollback called DROP FUNCTION on a name that migration
--   006 NEVER created — the actual function created by 006 (line 415)
--   is the trigger function whose name matches the trigger above. The
--   bogus DROP FUNCTION IF EXISTS silently no-op'd (because the named
--   function did not exist), and the REAL trigger function was left
--   orphaned in the schema after rollback — the trigger was dropped
--   (line 18) so the function was unreachable, but it polluted
--   ``pg_proc`` forever. A future migration that wanted to reuse the
--   function name for a different purpose would have collided with the
--   orphan.
--
--   ROOT FIX: drop the ACTUAL function name (matching the trigger name
--   above). ``IF EXISTS`` keeps the rollback idempotent — if 006 was
--   never applied (or the function was already dropped), the statement
--   is a no-op.
DROP FUNCTION IF EXISTS trg_drugs_sync_withdrawn() CASCADE;

-- Drop columns added by 006.
ALTER TABLE drugs DROP COLUMN IF EXISTS is_withdrawn;
ALTER TABLE drugs DROP COLUMN IF EXISTS clinical_status;
ALTER TABLE drugs DROP COLUMN IF EXISTS cas_number;
ALTER TABLE drugs DROP COLUMN IF EXISTS logp;
ALTER TABLE drugs DROP COLUMN IF EXISTS tpsa;
ALTER TABLE drugs DROP COLUMN IF EXISTS h_bond_donor_count;
ALTER TABLE drugs DROP COLUMN IF EXISTS h_bond_acceptor_count;
ALTER TABLE drugs DROP COLUMN IF EXISTS rotatable_bond_count;
ALTER TABLE drugs DROP COLUMN IF EXISTS heavy_atom_count;
ALTER TABLE drugs DROP COLUMN IF EXISTS complexity;
ALTER TABLE drugs DROP COLUMN IF EXISTS completeness_score;
ALTER TABLE drugs DROP COLUMN IF EXISTS groups;

-- v76 ROOT FIX (T-037 — schema_version rollback consistency):
-- Delete the schema_version row inserted by 006 (version=6). All
-- rollbacks now follow the same convention: delete the version row so
-- schema_version always reflects the CURRENT schema state. The
-- _migration_history table retains the full audit trail.
DELETE FROM schema_version WHERE version = 6;

COMMIT;
