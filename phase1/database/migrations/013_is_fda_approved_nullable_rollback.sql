-- ===========================================================================
-- Migration 013 ROLLBACK: restore NOT NULL DEFAULT FALSE on is_fda_approved
-- ===========================================================================
-- WARNING: this rollback REVERTS the P1-049 / P1-046 patient-safety fix.
-- Any rows with is_fda_approved = NULL ("unknown FDA status") will be
-- coerced to FALSE before the NOT NULL constraint is restored — silently
-- reverting the v93 fix. Only run this rollback if you fully understand
-- the patient-safety implications.
-- ===========================================================================

BEGIN;

-- Coerce NULL → FALSE so the NOT NULL constraint can be restored.
UPDATE drugs SET is_fda_approved = FALSE WHERE is_fda_approved IS NULL;

-- Drop the CHECK constraint.
ALTER TABLE drugs DROP CONSTRAINT IF EXISTS chk_drugs_is_fda_approved;

-- Restore the DEFAULT.
ALTER TABLE drugs ALTER COLUMN is_fda_approved SET DEFAULT FALSE;

-- Restore NOT NULL.
ALTER TABLE drugs ALTER COLUMN is_fda_approved SET NOT NULL;

-- P1-042 ROOT FIX (v110): delete the schema_version row so check_migrations()
-- no longer reports version 13 as applied after the rollback completes.
DELETE FROM schema_version WHERE version = 13;

COMMIT;
