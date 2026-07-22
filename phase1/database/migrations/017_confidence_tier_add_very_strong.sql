-- ===========================================================================
-- Migration 017: add very_strong confidence tier (split strong band)
-- ===========================================================================
-- P1-004 ROOT FIX EXTENSION (Team-1 v102 -- add very_strong tier):
--
-- The original P1-004 fix (migration 012) collapsed Piñero's strong band
-- [0.3, 1.0] into a single "strong" tier. This lost the gradation between
-- a score of 0.31 (marginal evidence) and 0.95 (very strong, curated
-- multi-source). Downstream ML models that bin on confidence_tier weighted
-- them identically -- biasing the model toward lower-confidence edges.
--
-- ROOT FIX: split the strong band into:
--   "strong"      [0.3, 0.5)   -- strong evidence (lower half)
--   "very_strong" [0.5, 1.0]   -- very strong evidence (upper half; curated)
--
-- This migration:
--   1. Drops the old chk_gda_confidence_tier constraint (which only
--      allowed 'sub_weak', 'weak', 'strong').
--   2. Backfills existing rows: those with score >= 0.5 AND
--      confidence_tier = 'strong' are renamed to 'very_strong'.
--      Rows with score in [0.3, 0.5) keep the 'strong' label.
--   3. Re-adds chk_gda_confidence_tier with the new label set:
--      ('sub_weak', 'weak', 'strong', 'very_strong').
--
-- This migration is IDEMPOTENT -- safe to run multiple times. The
-- constraint drop uses IF EXISTS; the backfill UPDATE only touches rows
-- that need renaming; the constraint add uses IF NOT EXISTS via a DO block.
--
-- Rollback: see 017_confidence_tier_add_very_strong_rollback.sql.
-- ===========================================================================

BEGIN;

-- Step 1: drop the old constraint so the backfill can rename labels
-- without the CHECK rejecting the intermediate state.
ALTER TABLE gene_disease_associations DROP CONSTRAINT IF EXISTS chk_gda_confidence_tier;

-- Step 2: backfill existing rows.
-- Rows with score >= 0.5 AND confidence_tier = 'strong' become 'very_strong'.
-- Rows with score in [0.3, 0.5) keep the 'strong' label.
-- Rows with score < 0.3 are unaffected (they have 'sub_weak' or 'weak').
UPDATE gene_disease_associations
SET confidence_tier = 'very_strong'
WHERE confidence_tier = 'strong'
  AND score IS NOT NULL
  AND score >= 0.5;

-- Step 3: re-add the constraint with the new label set (includes very_strong).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_gda_confidence_tier'
          AND conrelid = 'gene_disease_associations'::regclass
    ) THEN
        ALTER TABLE gene_disease_associations
            ADD CONSTRAINT chk_gda_confidence_tier
            CHECK (confidence_tier IS NULL OR confidence_tier IN
                   ('sub_weak', 'weak', 'strong', 'very_strong'));
    END IF;
END $$;

-- ===========================================================================
-- Step 4: Schema version metadata
-- ===========================================================================
-- P1-042 ROOT FIX (v110): the previous version of this migration was MISSING
-- the INSERT INTO schema_version row. check_migrations() cross-references
-- schema_version; without the version=17 row, those checks reported
-- schema_version_matches=False even though the migration had been applied.
-- ROOT FIX: add the INSERT with ON CONFLICT DO NOTHING for idempotency.
INSERT INTO schema_version (version, description)
VALUES (
    17,
    'P1-004 v102 ROOT FIX EXTENSION: split Piñero strong band [0.3, 1.0] into '
    '"strong" [0.3, 0.5) and "very_strong" [0.5, 1.0]. Backfill rows with '
    'score >= 0.5 AND confidence_tier = strong to very_strong. Update CHECK '
    'constraint to allow the new label.'
)
ON CONFLICT (version) DO NOTHING;

COMMIT;

-- ===========================================================================
-- Post-migration verification (P1-004 v102 -- assert the CHECK exists
-- with the new label set). Runs AFTER COMMIT so the verification sees
-- the committed state.
-- ===========================================================================
DO $$
DECLARE
    _constraint_exists BOOLEAN;
    _constraint_def TEXT;
    _mislabelled_count INTEGER;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_gda_confidence_tier'
          AND conrelid = 'gene_disease_associations'::regclass
    ) INTO _constraint_exists;

    IF NOT _constraint_exists THEN
        RAISE EXCEPTION 'P1-004 v102 VERIFICATION FAILED: chk_gda_confidence_tier constraint missing after migration 017 -- the DB is in a half-migrated state. Manual intervention required.';
    END IF;

    -- Verify the constraint definition contains the new 'very_strong' label.
    SELECT pg_get_constraintdef(oid) INTO _constraint_def
    FROM pg_constraint
    WHERE conname = 'chk_gda_confidence_tier'
      AND conrelid = 'gene_disease_associations'::regclass;

    IF _constraint_def IS NULL THEN
        RAISE EXCEPTION 'P1-004 v102 VERIFICATION FAILED: could not read chk_gda_confidence_tier definition';
    END IF;

    IF _constraint_def NOT LIKE '%very_strong%' THEN
        RAISE EXCEPTION 'P1-004 v102 VERIFICATION FAILED: chk_gda_confidence_tier does not contain the new very_strong label. Got: %', _constraint_def;
    END IF;

    -- Verify NO rows with score >= 0.5 still have the 'strong' label
    -- (the backfill should have renamed them all to 'very_strong').
    SELECT COUNT(*) INTO _mislabelled_count
    FROM gene_disease_associations
    WHERE confidence_tier = 'strong'
      AND score IS NOT NULL
      AND score >= 0.5;

    IF _mislabelled_count > 0 THEN
        RAISE EXCEPTION 'P1-004 v102 VERIFICATION FAILED: gene_disease_associations still contains % rows with confidence_tier=''strong'' AND score >= 0.5 -- the migration 017 backfill did not complete. Manual intervention required.', _mislabelled_count;
    END IF;

    RAISE NOTICE 'P1-004 v102 VERIFICATION PASSED: chk_gda_confidence_tier exists with new label set (sub_weak, weak, strong, very_strong) and no rows with score >= 0.5 are mislabelled as strong.';
END $$;
