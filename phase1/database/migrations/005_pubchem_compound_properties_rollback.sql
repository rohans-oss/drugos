-- ============================================================================
-- Drug Repurposing ETL Platform — Rollback 005 PubChem Compound Properties
-- Migration: 005_pubchem_compound_properties_rollback.sql
-- Description: Drops the pubchem_compound_properties table and its indexes.
--
-- v22 ROOT FIX (audit P2-10 / Section 9): rollback sidecar for 005.
-- ============================================================================

BEGIN;

-- Drop indexes first (must precede table drop in some dialects).
DROP INDEX IF EXISTS idx_pubchem_props_cid;
DROP INDEX IF EXISTS idx_pubchem_props_inchikey;
DROP INDEX IF EXISTS idx_pubchem_props_is_deleted;
DROP INDEX IF EXISTS idx_pubchem_props_run_id;

-- Drop the table.
DROP TABLE IF EXISTS pubchem_compound_properties CASCADE;

-- v76 ROOT FIX (T-037 — schema_version rollback consistency):
-- Delete the schema_version row inserted by 005 (version=5). This
-- matches the convention adopted across ALL rollbacks (003/010/011
-- already delete; 002/004/005/006/007/008/009 now also delete). The
-- schema_version table is OWNED by 001; only this migration's row is
-- removed. The _migration_history table (run_migrations.py) retains
-- the full audit trail of apply/rollback events with timestamps.
DELETE FROM schema_version WHERE version = 5;

COMMIT;
