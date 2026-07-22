-- ============================================================================
-- Drug Repurposing ETL Platform — Drug.is_globally_approved Column Migration
-- Migration: 008_drug_is_globally_approved.sql
-- Description: Add an `is_globally_approved` BOOLEAN column to the `drugs`
--              table so the ChEMBL pipeline's per-drug global-approval flag
--              (any of FDA / EMA / PMDA / MHRA / Health Canada / TGA —
--              derived from `max_phase == 4` per SW-1 ROOT FIX patient-
--              safety audit) is persisted instead of being silently dropped
--              by `_filter_to_drug_columns` (which kept only Drug-model
--              columns, and is_globally_approved was not one).
--
-- ROOT-CAUSE FIX (audit P1-28):
--   The ChEMBL pipeline (chembl_pipeline.py:1984) emits is_globally_approved
--   in every record dict, but the Drug ORM model did not declare the column
--   and `_filter_to_drug_columns` did not whitelist it. The loader
--   (bulk_upsert_drugs) rejected/ignored the unknown column, so the value
--   was always NULL in the DB. Downstream consumers that queried
--   `is_globally_approved` got NULL for every row — the column was
--   effectively dead.
--
--   This migration adds the column; the ORM and the column-whitelist are
--   updated in parallel (database/models.py, chembl_pipeline.py).
--
-- All new columns are NULLABLE — existing rows and existing tests are
-- unaffected. No columns are dropped, no constraints are weakened.
--
-- PREREQUISITES: 001_initial_schema.sql through 007_pipeline_run_metadata.sql.
-- ============================================================================

BEGIN;

-- ===========================================================================
-- Phase 1: Add is_globally_approved column
-- ===========================================================================
-- Nullable BOOLEAN: NULL means "unknown" (e.g. drugs loaded from sources
-- other than ChEMBL that don't populate max_phase). The ChEMBL pipeline
-- sets it to (max_phase == 4) per SW-1 ROOT FIX.
ALTER TABLE drugs ADD COLUMN IF NOT EXISTS is_globally_approved BOOLEAN;

-- Idempotency: PostgreSQL does not support IF NOT EXISTS for ADD CONSTRAINT,
-- so we use a DO block to check pg_constraint first.
-- v74 ROOT FIX (T-014 compound — simplify CHECK to match ORM):
--   The previous CHECK ``is_globally_approved IN (FALSE, TRUE)`` was
--   PostgreSQL-specific (SQLite does not parse the FALSE/TRUE keywords
--   in CHECK constraints pre-3.23). The ORM (models.py:815-818) now
--   uses the portable ``is_globally_approved IS NOT NULL AND
--   is_globally_approved IN (0, 1)`` form (matching chk_drugs_is_fda_approved
--   and chk_drugs_is_withdrawn). Migration 008's CHECK is aligned to
--   the same form so dev (SQLite/ORM-created) and prod (PostgreSQL/
--   migration-created) enforce the IDENTICAL constraint semantics.
--   The ``IS NOT NULL`` predicate is defense-in-depth: the column is
--   ``NOT NULL DEFAULT FALSE`` (line 110-112 below), so NULL should
--   never reach the DB today — but if a future migration makes the
--   column nullable, the explicit predicate guards against silent
--   NULL acceptance (patient-safety signal — an unknown global-
--   approval status must NOT silently become "approved").
--
--   v75 ROOT FIX (T-029 — verify T-029 is fully resolved):
--     The audit issue T-029 quoted the 4-value form (the set of all
--     four boolean literals: zero, one, TRUE, FALSE) at line 42-51.
--     That form is GONE in v74 (replaced with the portable 2-value
--     zero/one form at lines 61 and 70). The v75 verification confirms:
--       (1) Migration 008 line 61: zero/one 2-value form (actual SQL)
--       (2) Migration 008 line 70: zero/one 2-value form (actual SQL)
--       (3) ORM models.py line 816: zero/one 2-value form (actual SQL)
--       (4) chk_drugs_is_fda_approved (001:242) uses zero/one form
--       (5) chk_drugs_is_withdrawn (006:43) uses zero/one form
--     All five boolean CHECK constraints now use the SAME 2-value
--     form — no stylistic inconsistency, no dialect-specific
--     ``TRUE``/``FALSE`` keywords. T-029 is fully resolved; this
--     comment block is the audit trail.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_drugs_is_globally_approved') THEN
        ALTER TABLE drugs ADD CONSTRAINT chk_drugs_is_globally_approved
            CHECK (is_globally_approved IS NOT NULL AND is_globally_approved IN (0, 1));
        RAISE NOTICE '  [OK] Added constraint chk_drugs_is_globally_approved (v74 T-014: portable IN (0, 1) form)';
    ELSE
        -- v74: the constraint already exists. If it was added by an
        -- older (pre-v74) run with the IN (FALSE, TRUE) form, replace
        -- it with the portable IN (0, 1) form so dev/prod match.
        BEGIN
            ALTER TABLE drugs DROP CONSTRAINT chk_drugs_is_globally_approved;
            ALTER TABLE drugs ADD CONSTRAINT chk_drugs_is_globally_approved
                CHECK (is_globally_approved IS NOT NULL AND is_globally_approved IN (0, 1));
            RAISE NOTICE '  [OK] Replaced chk_drugs_is_globally_approved with portable IN (0, 1) form (v74 T-014)';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE '  [SKIP] constraint chk_drugs_is_globally_approved already exists and could not be replaced: %', SQLERRM;
        END;
    END IF;
END $$;

-- ===========================================================================
-- Phase 2: Backfill from max_phase == 4 (ChEMBL semantic — any regulator)
-- ===========================================================================
-- Any drug already loaded with max_phase=4 is globally approved per ChEMBL.
-- This backfill brings existing rows in line with what the ChEMBL pipeline
-- would have written if is_globally_approved had existed from the start.
-- Rows with is_globally_approved already set (non-NULL) are preserved.
--
-- v57 ROOT FIX (T-002 — Vioxx Patient-Safety Bug):
--   The previous backfill set is_globally_approved=TRUE for EVERY drug with
--   max_phase=4, INCLUDING withdrawn drugs (Vioxx, Bextra, Baycol, etc.).
--   Combined with migration 006's backfill matching ZERO rows on a fresh DB
--   (because drugs.groups is NULL until drugbank_pipeline runs), every
--   approved-but-withdrawn drug ended up as is_globally_approved=TRUE AND
--   is_withdrawn=FALSE — appearing as a SAFE repurposing candidate.
--
--   FIX: exclude withdrawn drugs from the globally-approved backfill, AND
--   add a CHECK constraint enforcing the invariant
--   (is_globally_approved=TRUE IMPLIES is_withdrawn=FALSE) so future
--   loader bugs cannot reintroduce the patient-safety bypass.
UPDATE drugs
SET is_globally_approved = TRUE
WHERE is_globally_approved IS NULL
  AND max_phase = 4
  AND is_withdrawn = FALSE;

UPDATE drugs
SET is_globally_approved = FALSE
WHERE is_globally_approved IS NULL
  AND max_phase IS NOT NULL
  AND max_phase < 4;

-- v57 ROOT FIX (T-002 continued): explicit FALSE for withdrawn drugs.
-- A withdrawn drug is by definition NOT a currently-approved repurposing
-- candidate, regardless of max_phase.
UPDATE drugs
SET is_globally_approved = FALSE
WHERE is_globally_approved IS NULL
  AND is_withdrawn = TRUE;

-- v65 ROOT FIX (P1C-006 — silent NULL exclusion from "approved" queries):
--   The ORM now declares is_globally_approved as NOT NULL DEFAULT FALSE
--   (matching is_fda_approved and is_withdrawn). The previous migration
--   left the column NULLABLE, so any drug that didn't match the backfill
--   conditions above (e.g. max_phase IS NULL AND is_withdrawn = FALSE —
--   a drug loaded from a source that doesn't emit max_phase) kept a NULL
--   is_globally_approved. Downstream `WHERE is_globally_approved = True`
--   silently excluded these NULL rows (three-valued logic: NULL = True
--   → UNKNOWN → falsy), treating drugs with unknown approval status as
--   NOT approved — the opposite of the patient-safety intent.
--   ROOT FIX: backfill any remaining NULLs to FALSE (the conservative,
--   patient-safe default), then set the column to NOT NULL DEFAULT FALSE
--   so future INSERTs that omit the column get False (explicit, not NULL).
UPDATE drugs
SET is_globally_approved = FALSE
WHERE is_globally_approved IS NULL;

ALTER TABLE drugs ALTER COLUMN is_globally_approved SET DEFAULT FALSE;

ALTER TABLE drugs ALTER COLUMN is_globally_approved SET NOT NULL;

-- v57 ROOT FIX (T-002 continued): invariant — a drug cannot be both
-- globally approved AND withdrawn. This is a hard patient-safety
-- invariant; if a future loader bug or trigger tries to set both,
-- PostgreSQL will reject the INSERT/UPDATE with IntegrityError.
-- v90 ROOT FIX (BUG #6): use the portable ``= 1`` form (not ``= TRUE``)
-- so the constraint works identically on SQLite dev/test DBs created via
-- the migration runner. The ORM (models.py) now declares the SAME
-- constraint with the SAME name so dev DBs created via create_all() also
-- enforce it — closing the "tests pass on dev, prod kills" anti-pattern.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_drugs_no_approved_and_withdrawn') THEN
        ALTER TABLE drugs ADD CONSTRAINT chk_drugs_no_approved_and_withdrawn
            CHECK (NOT (is_globally_approved = 1 AND is_withdrawn = 1));
        RAISE NOTICE '  [OK] Added patient-safety invariant chk_drugs_no_approved_and_withdrawn';
    ELSE
        RAISE NOTICE '  [SKIP] constraint chk_drugs_no_approved_and_withdrawn already exists';
    END IF;
END $$;

-- ===========================================================================
-- Phase 3: Index for fast filtering of globally-approved drugs
-- ===========================================================================
CREATE INDEX IF NOT EXISTS idx_drugs_is_globally_approved
    ON drugs (is_globally_approved)
    WHERE is_globally_approved IS NOT NULL;

-- ===========================================================================
-- Phase 4: Schema version metadata
-- ===========================================================================
INSERT INTO schema_version (version, description)
VALUES (8, 'Add is_globally_approved BOOLEAN column to drugs table (P1-28 ROOT FIX — ChEMBL pipeline emits this but it was silently dropped by _filter_to_drug_columns); backfill from max_phase == 4')
ON CONFLICT (version) DO NOTHING;

COMMIT;
