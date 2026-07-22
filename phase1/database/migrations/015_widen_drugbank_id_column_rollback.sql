-- ============================================================================
-- ROLLBACK: Widen drugbank_id column to VARCHAR(64)
-- Migration: 013_widen_drugbank_id_column_rollback.sql
-- Description: Revert the P1-017 widening of ``drugs.drugbank_id`` and
--              ``entity_mapping.drugbank_id`` back to VARCHAR(10).
--
-- WARNING: This rollback will FAIL if any row has a drugbank_id longer
--          than 10 chars (e.g. a synthesized ``SYNTH-DB-...`` ID).
--          Quarantine or delete those rows BEFORE running this rollback.
-- ============================================================================

BEGIN;

-- Pre-rollback audit: warn if any row will be truncated.
DO $$
DECLARE
    _long_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO _long_count
    FROM drugs
    WHERE LENGTH(drugbank_id) > 10;
    IF _long_count > 0 THEN
        RAISE WARNING
            '  [AUDIT] % row(s) in drugs.drugbank_id have LENGTH > 10 — ' ||
            'these will be TRUNCATED by the rollback to VARCHAR(10). ' ||
            'Quarantine or delete them BEFORE committing this rollback.',
            _long_count;
    END IF;
END $$;

ALTER TABLE drugs
    ALTER COLUMN drugbank_id TYPE VARCHAR(10);

ALTER TABLE entity_mapping
    ALTER COLUMN drugbank_id TYPE VARCHAR(10);

DELETE FROM schema_version WHERE version = 15;

DO $$
BEGIN
    RAISE NOTICE '  [OK] Reverted drugs.drugbank_id + entity_mapping.drugbank_id to VARCHAR(10) (P1-017 rollback)';
END $$;

COMMIT;
