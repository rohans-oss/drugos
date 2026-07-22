-- ============================================================================
-- Rollback for Migration 018: standard_relation + is_homodimer
-- Reverts the columns + constraints + indexes added by 018.
-- Safe to run multiple times (uses IF EXISTS).
-- ============================================================================

BEGIN;

-- Drop the indexes first (depend on the columns).
DROP INDEX IF EXISTS idx_ppi_is_homodimer;
DROP INDEX IF EXISTS idx_dpi_standard_relation;

-- Drop the CHECK constraints.
ALTER TABLE protein_protein_interactions DROP CONSTRAINT IF EXISTS chk_ppi_homodimer_invariant;
ALTER TABLE drug_protein_interactions DROP CONSTRAINT IF EXISTS chk_dpi_standard_relation;

-- Drop the columns. SQLite <3.35 doesn't support DROP COLUMN; the migration
-- runner's per-statement try/except handles "no such column" as a no-op.
ALTER TABLE protein_protein_interactions DROP COLUMN IF EXISTS is_homodimer;
ALTER TABLE drug_protein_interactions DROP COLUMN IF EXISTS standard_relation;

-- P1-042 ROOT FIX (v110): delete the schema_version row.
DELETE FROM schema_version WHERE version = 18;

COMMIT;
