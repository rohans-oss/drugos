-- ===========================================================================
-- Migration 013: is_fda_approved nullable (P1-049 / P1-046 ROOT FIX)
-- ===========================================================================
-- P1-049 / P1-046 FORENSIC ROOT FIX (Team 4):
--
-- The previous schema (migration 001 line 264) was:
--     is_fda_approved BOOLEAN NOT NULL DEFAULT FALSE
--
-- The ChEMBL pipeline docstring (chembl_pipeline.py line 85) says
-- ``is_fda_approved`` should be ``None`` (unknown) until the FDA Orange
-- Book join runs (the v93 patient-safety fix). The v93 fix (per
-- chembl_pipeline.py line 468-472) sets ``is_fda_approved = None`` for
-- ``max_phase == 4`` drugs until an FDA Orange Book join runs.
--
-- But the DB column was ``NOT NULL`` — inserting ``None`` raised
-- ``IntegrityError``. The loader's ``_pre_validate_drugs`` coerced
-- ``None`` to ``False`` to satisfy the constraint, SILENTLY REVERTING
-- the v93 fix. EMA-only drugs (max_phase=4, not FDA-approved) were
-- stored as ``is_fda_approved=FALSE`` — same as a confirmed-not-approved
-- drug. Downstream RL ranker's FDA safety filter treated them identically.
--
-- ROOT FIX:
--   1. Drop the NOT NULL constraint (allow NULL = "unknown FDA status").
--   2. Drop the DEFAULT FALSE (the loader must explicitly insert NULL
--      when the status is unknown — no silent coercion).
--   3. Add a CHECK constraint that allows NULL, TRUE, or FALSE (SQLite
--      compatibility — SQLite stores BOOLEAN as INTEGER, so the CHECK
--      guards against 2, -1, etc.).
--
-- IMPORTANT — backfill semantics:
--   Existing rows with ``is_fda_approved = FALSE`` are NOT automatically
--   changed to NULL. The operator must decide whether existing FALSE
--   values represent "confirmed not FDA-approved" or "unknown" (the
--   pre-fix schema conflated them). The recommended backfill is:
--
--     UPDATE drugs SET is_fda_approved = NULL
--     WHERE is_fda_approved = FALSE
--       AND max_phase >= 4
--       AND source = 'chembl';
--
--   This reverts the silent coercion for EMA-only drugs (max_phase=4
--   from ChEMBL but no FDA Orange Book join). Drugs with ``source !=
--   'chembl'`` or ``max_phase < 4`` retain their FALSE value (those
--   sources do distinguish "not approved" from "unknown").
--
-- Rollback: see 013_is_fda_approved_nullable_rollback.sql.
-- ===========================================================================

BEGIN;

-- Step 1: drop the legacy CHECK constraint if it exists (idempotent).
-- The original migration 001 did NOT add a CHECK on is_fda_approved
-- (only NOT NULL DEFAULT FALSE), so there may be no constraint to drop.
-- We use IF EXISTS for idempotency.
ALTER TABLE drugs DROP CONSTRAINT IF EXISTS chk_drugs_is_fda_approved;

-- Step 2: drop the NOT NULL constraint.
-- PostgreSQL: ALTER COLUMN ... DROP NOT NULL.
-- SQLite does not support ALTER COLUMN DROP NOT NULL directly; the
-- application-layer migration runner (run_migrations.py) handles SQLite
-- via table rebuild. This migration is PostgreSQL-targeted; the SQLite
-- path is handled by the ORM (database/models/drug_model.py) which now
-- declares the column as nullable.
ALTER TABLE drugs ALTER COLUMN is_fda_approved DROP NOT NULL;

-- Step 3: drop the DEFAULT FALSE (the loader must explicitly insert NULL
-- when the status is unknown — no silent coercion).
ALTER TABLE drugs ALTER COLUMN is_fda_approved DROP DEFAULT;

-- Step 4: add the CHECK constraint (NULL, TRUE, or FALSE only).
-- Use a DO block so the migration is idempotent (safe to re-run).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_drugs_is_fda_approved'
          AND conrelid = 'drugs'::regclass
    ) THEN
        ALTER TABLE drugs
            ADD CONSTRAINT chk_drugs_is_fda_approved
            CHECK (is_fda_approved IS NULL OR is_fda_approved IN (TRUE, FALSE));
    END IF;
END $$;

-- Step 5: backfill — revert the silent coercion for EMA-only drugs.
-- ChEMBL-sourced drugs with max_phase >= 4 that were stored as FALSE
-- (because the NOT NULL constraint coerced NULL → FALSE) are set back
-- to NULL ("unknown FDA status"). The operator can re-run the FDA
-- Orange Book join to populate the correct value.
UPDATE drugs
SET is_fda_approved = NULL
WHERE is_fda_approved = FALSE
  AND max_phase >= 4
  AND source = 'chembl';

-- ===========================================================================
-- Step 6: Schema version metadata
-- ===========================================================================
-- P1-042 ROOT FIX (v110): the previous version of this migration was MISSING
-- the INSERT INTO schema_version row. The migration runner's
-- _is_migration_applied() check uses _migration_history (not schema_version),
-- so the migration was still tracked as applied. BUT check_migrations()
-- cross-references schema_version to confirm the DB is at the expected
-- version. Without the version=13 row, those checks reported
-- schema_version_matches=False even though all 13 migrations had been
-- applied — a false-negative that blocked CI gates.
-- ROOT FIX: add the INSERT with ON CONFLICT DO NOTHING for idempotency.
INSERT INTO schema_version (version, description)
VALUES (
    13,
    'P1-049 / P1-046 ROOT FIX: make drugs.is_fda_approved nullable so the ChEMBL '
    'pipeline can persist NULL = "unknown FDA status" (EMA-only drugs) instead of '
    'silently coercing to FALSE. Drop NOT NULL + DEFAULT FALSE; add CHECK allowing '
    'NULL / TRUE / FALSE. Backfill EMA-only drugs (max_phase>=4, source=chembl) to NULL.'
)
ON CONFLICT (version) DO NOTHING;

COMMIT;

-- ===========================================================================
-- Post-migration verification (P1-043 pattern — assert the CHECK exists).
-- Runs AFTER COMMIT so the verification sees the committed state.
-- ===========================================================================
DO $$
DECLARE
    _constraint_exists BOOLEAN;
    _is_nullable BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_drugs_is_fda_approved'
          AND conrelid = 'drugs'::regclass
    ) INTO _constraint_exists;

    IF NOT _constraint_exists THEN
        RAISE EXCEPTION 'P1-049 VERIFICATION FAILED: chk_drugs_is_fda_approved constraint missing after migration 013';
    END IF;

    SELECT is_nullable INTO _is_nullable
    FROM information_schema.columns
    WHERE table_name = 'drugs' AND column_name = 'is_fda_approved';

    IF _is_nullable IS NULL THEN
        RAISE EXCEPTION 'P1-049 VERIFICATION FAILED: drugs.is_fda_approved column not found';
    END IF;

    IF _is_nullable <> 'YES' THEN
        RAISE EXCEPTION 'P1-049 VERIFICATION FAILED: drugs.is_fda_approved must be nullable (is_nullable=YES), got %', _is_nullable;
    END IF;

    RAISE NOTICE 'P1-049 VERIFICATION PASSED: drugs.is_fda_approved is nullable and CHECK constraint exists';
END $$;
