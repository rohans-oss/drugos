-- ===========================================================================
-- Rollback for Migration 012: confidence_tier label alignment
-- ===========================================================================
-- Reverts the P1-004 v100 rename: restores the old label set
-- ('weak', 'moderate', 'strong'). Use ONLY if a downstream consumer
-- depends on the legacy label set — the rollback re-introduces the
-- scientific mislabel bug (Piñero's [0.06, 0.3) band re-tagged as
-- 'moderate' instead of 'weak').
-- ===========================================================================

BEGIN;

ALTER TABLE gene_disease_associations DROP CONSTRAINT IF EXISTS chk_gda_confidence_tier;

-- Reverse the backfill.
UPDATE gene_disease_associations
SET confidence_tier = 'moderate'
WHERE confidence_tier = 'weak';

UPDATE gene_disease_associations
SET confidence_tier = 'weak'
WHERE confidence_tier = 'sub_weak';

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
                   ('weak', 'moderate', 'strong'));
    END IF;
END $$;

-- P1-042 ROOT FIX (v110): delete the schema_version row so check_migrations()
-- no longer reports version 12 as applied after the rollback completes.
DELETE FROM schema_version WHERE version = 12;

COMMIT;
