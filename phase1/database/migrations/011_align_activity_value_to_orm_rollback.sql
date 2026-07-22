-- ============================================================================
-- Rollback for migration 011_align_activity_value_to_orm.sql
-- ============================================================================
-- WARNING: rolling back this migration re-introduces the dev-vs-prod schema
-- drift that T-013 fixed (FLOAT vs NUMERIC(10,4)). Only roll back if you
-- have a downstream consumer that ABSOLUTELY requires FLOAT and you have
-- verified the consumer handles the precision loss. In normal operation,
-- DO NOT roll back — the NUMERIC(10,4) type is the source of truth
-- (matching the ORM models.py:1148).
-- ============================================================================

BEGIN;

ALTER TABLE drug_protein_interactions
    ALTER COLUMN activity_value TYPE FLOAT
    USING activity_value::double precision;

DELETE FROM schema_version WHERE version = 11;

COMMIT;
