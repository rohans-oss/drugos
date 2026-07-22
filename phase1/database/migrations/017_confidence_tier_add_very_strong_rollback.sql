-- ===========================================================================
-- Rollback for Migration 017: add very_strong confidence tier
-- ===========================================================================
-- Reverts the very_strong split: rows with confidence_tier='very_strong'
-- are renamed back to 'strong', and the CHECK constraint is reverted
-- to the original 3-label set (sub_weak, weak, strong).
--
-- WARNING: rolling back this migration LOSES the gradation between
-- marginal (0.3-0.5) and very strong (0.5-1.0) evidence. Only roll
-- back if you have a specific, documented reason.
-- ===========================================================================

BEGIN;

ALTER TABLE gene_disease_associations DROP CONSTRAINT IF EXISTS chk_gda_confidence_tier;

-- Rename 'very_strong' rows back to 'strong'.
UPDATE gene_disease_associations
SET confidence_tier = 'strong'
WHERE confidence_tier = 'very_strong';

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
                   ('sub_weak', 'weak', 'strong'));
    END IF;
END $$;

-- P1-042 ROOT FIX (v110): delete the schema_version row so check_migrations()
-- no longer reports version 17 as applied after the rollback completes.
DELETE FROM schema_version WHERE version = 17;

COMMIT;

DO $$
BEGIN
    RAISE NOTICE 'P1-017 ROLLBACK: very_strong tier removed; rows renamed back to strong. CHECK constraint reverted to 3-label set (sub_weak, weak, strong).';
END $$;
