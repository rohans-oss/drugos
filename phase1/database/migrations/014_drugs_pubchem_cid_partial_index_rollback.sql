-- ============================================================================
-- Rollback for Migration 014: drugs_pubchem_cid_partial_index
-- Drops the partial index added by 014_drugs_pubchem_cid_partial_index.sql.
-- Safe to run multiple times (uses IF EXISTS).
--
-- P1-042 ROOT FIX (v110):
--   1. Fix the header comment — the previous version said "Rollback for
--      Migration 013" but the filename is 014_*. The wrong number caused
--      confusion when searching for "rollback 013" (which would also match
--      this file's header).
--   2. Wrap in BEGIN/COMMIT for atomicity (matches all other rollbacks).
--   3. Delete the schema_version row so check_migrations() no longer
--      reports version 14 as applied after the rollback completes.
-- ============================================================================

BEGIN;

DROP INDEX IF EXISTS ix_drugs_pubchem_cid_null_inchikey;

-- P1-042 ROOT FIX (v110): delete the schema_version row.
DELETE FROM schema_version WHERE version = 14;

COMMIT;
