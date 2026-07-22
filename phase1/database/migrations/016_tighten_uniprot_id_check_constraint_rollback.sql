-- ===========================================================================
-- Rollback for Migration 016: tighten proteins.uniprot_id CHECK constraint
-- ===========================================================================
-- Reverts the strict CHECK (LENGTH IN (6, 10)) back to the relaxed
-- CHECK (LENGTH >= 4 AND LENGTH <= 10).
--
-- WARNING: rolling back this migration WEAKENS the database contract.
-- Only roll back if you have a specific, documented reason (e.g. a
-- legacy tool that writes 4-5 char accessions). Otherwise, leave the
-- strict CHECK in place -- it matches the ORM and the UniProt spec.
-- ===========================================================================

BEGIN;

ALTER TABLE proteins DROP CONSTRAINT IF EXISTS chk_proteins_uniprot_length;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_proteins_uniprot_length'
          AND conrelid = 'proteins'::regclass
    ) THEN
        ALTER TABLE proteins
            ADD CONSTRAINT chk_proteins_uniprot_length
            CHECK (LENGTH(uniprot_id) >= 4 AND LENGTH(uniprot_id) <= 10);
    END IF;
END $$;

-- P1-042 ROOT FIX (v110): delete the schema_version row so check_migrations()
-- no longer reports version 16 as applied after the rollback completes.
DELETE FROM schema_version WHERE version = 16;

COMMIT;

-- NOTE: rows that were quarantined (uniprot_id set to NULL) by the
-- forward migration are NOT restored. Their original (invalid) uniprot_id
-- values are lost. This is intentional -- the values were junk and
-- restoring them would re-introduce the bug.
DO $$
BEGIN
    RAISE NOTICE 'P1-016 ROLLBACK: chk_proteins_uniprot_length reverted to relaxed (LENGTH >= 4 AND <= 10). Quarantined rows are NOT restored -- their original uniprot_id values were junk and have been lost.';
END $$;
