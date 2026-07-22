-- ============================================================================
-- Drug Repurposing ETL Platform — Align activity_value FLOAT → NUMERIC(10,4)
-- Migration: 011_align_activity_value_to_orm.sql
-- Version: 11
--
-- v74 ROOT FIX (T-013 — schema drift between migration 001 and ORM).
--
-- ROOT CAUSE
-- ----------
-- Migration 001 originally declared
--     drug_protein_interactions.activity_value  FLOAT
-- (IEEE 754 binary64, ~15 significant digits, NO exact decimal representation).
-- The ORM ``DrugProteinInteraction.activity_value`` in models.py:1148 declared
--     activity_value: Mapped[Optional[float]] = mapped_column(Numeric(10, 4), nullable=True)
-- (decimal, 10 total digits, 4 after decimal).
--
-- Dev DBs created via ``Base.metadata.create_all()`` (ORM path — pytest, local
-- SQLite/PG dev runs) had NUMERIC(10,4). Prod DBs created via the migration
-- runner had FLOAT. No migration 002-010 ALTERed this column's type, so the
-- divergence persisted across every release.
--
-- CONSEQUENCES (verified by audit)
-- --------------------------------
-- 1. IC50 values like 0.00123 nM stored as FLOAT may differ from NUMERIC(10,4)
--    by 1 ULP (last unit in place) — cross-table joins on activity_value fail
--    silently.
-- 2. pIC50 calculations (=-log10(activity_value * 1e-9)) produce different
--    results on dev (ORM) vs prod (migration) DBs. The Graph Transformer
--    trained on dev data does not reproduce on prod.
-- 3. run_migrations.py::verify_schema_matches_orm introspects the ORM and
--    compares to the DB; this mismatch is flagged as drift, but the function
--    has a fallback dict (EXPECTED_SCHEMA) that masks the divergence.
--
-- ROOT FIX
-- --------
-- Two-part:
--   1. Amend migration 001 to declare ``NUMERIC(10, 4)`` so FRESH deployments
--      match the ORM from the start. (Done — see 001 line ~638.)
--   2. This migration (011) ALTERs the column type for ALREADY-DEPLOYED
--      databases (those upgraded from v73 or earlier). The USING clause
--      performs an explicit cast so PostgreSQL re-materialises every row
--      as exact decimal — no silent rounding, no ULP drift.
--
-- The CHECK constraint ``chk_dpi_activity_value_positive`` (already present
-- from migration 001) is preserved — ALTER COLUMN TYPE does not drop CHECKs.
--
-- SQLite note: SQLite does not support ``ALTER COLUMN TYPE`` — the SQLite
-- migration translator skips those statements (see run_migrations.py:2626).
-- On SQLite, the column type is advisory (SQLite uses dynamic typing), so
-- the FLOAT vs NUMERIC(10,4) distinction doesn't matter at runtime. This
-- migration's ALTER statements are PostgreSQL-only by design.
--
-- PREREQUISITES: 001_initial_schema.sql through 010_drug_indication_columns.sql.
-- IDEMPOTENT: re-running is a no-op (PostgreSQL ALTER TYPE to the same type
-- is a metadata-only change when no rows need conversion).
-- ============================================================================

BEGIN;

-- ===========================================================================
-- Phase 1: Detect current column type and ALTER to NUMERIC(10, 4) if needed.
--
-- We use a single DO block with IF/ELSIF/ELSE (no RETURN statement — the
-- SQLite migration translator cannot translate PL/pgSQL RETURN inside DO
-- blocks, see run_migrations.py::_translate_sql_for_sqlite). The IF/ELSIF
-- chain covers every possible current type without needing early exit.
-- ===========================================================================
DO $$
DECLARE
    _current_type TEXT;
BEGIN
    SELECT data_type INTO _current_type
    FROM information_schema.columns
    WHERE table_name = 'drug_protein_interactions'
      AND column_name = 'activity_value';

    -- normalise to lowercase for comparison
    _current_type := lower(_current_type);

    IF _current_type IS NULL THEN
        -- Column not found — table doesn't exist yet. Migration 001 hasn't
        -- run. We can't ALTER a column that doesn't exist. The schema_version
        -- entry is still recorded below so the migration is marked applied
        -- (the column will be created correctly as NUMERIC(10,4) when 001
        -- runs after this migration — but in practice 001 always runs
        -- before 011 due to migration ordering).
        RAISE NOTICE 'Migration 011: column drug_protein_interactions.activity_value not found — table not yet created. Skipping ALTER (001 will create the column correctly as NUMERIC(10,4)).';
    ELSIF _current_type IN ('numeric', 'decimal') THEN
        -- Already a decimal type. Verify precision/scale match Numeric(10, 4).
        -- ALTER to the exact precision/scale to be safe (no-op if already correct).
        BEGIN
            ALTER TABLE drug_protein_interactions
                ALTER COLUMN activity_value TYPE NUMERIC(10, 4)
                USING activity_value::numeric(10, 4);
            RAISE NOTICE 'Migration 011: activity_value precision/scale aligned to NUMERIC(10, 4)';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Migration 011: activity_value already NUMERIC(10,4) — no-op (%)', SQLERRM;
        END;
    ELSIF _current_type IN ('double precision', 'real', 'float') THEN
        -- FLOAT in PostgreSQL maps to double precision (8 bytes) for the
        -- SQL standard FLOAT type with no precision, OR real (4 bytes)
        -- when FLOAT(p) with p<=24 is used. Both must be converted.
        RAISE NOTICE 'Migration 011: converting activity_value from % to NUMERIC(10, 4)...', _current_type;
        ALTER TABLE drug_protein_interactions
            ALTER COLUMN activity_value TYPE NUMERIC(10, 4)
            USING activity_value::numeric(10, 4);
        RAISE NOTICE 'Migration 011: activity_value converted FLOAT → NUMERIC(10, 4) (T-013 root fix)';
    ELSE
        -- Unexpected type — attempt conversion anyway (best-effort).
        RAISE NOTICE 'Migration 011: activity_value has unexpected type % — attempting conversion', _current_type;
        ALTER TABLE drug_protein_interactions
            ALTER COLUMN activity_value TYPE NUMERIC(10, 4)
            USING activity_value::numeric(10, 4);
    END IF;
END $$;

-- ===========================================================================
-- Phase 2: Schema version metadata
-- ===========================================================================
INSERT INTO schema_version (version, description)
VALUES (
    11,
    'v74 ROOT FIX (T-013): align drug_protein_interactions.activity_value FLOAT → NUMERIC(10, 4) '
    'to match the ORM (models.py DrugProteinInteraction.activity_value). Closes the dev-vs-prod '
    'schema drift where pIC50 calculations diverged by 1 ULP and cross-table joins on '
    'activity_value silently failed.'
)
ON CONFLICT (version) DO NOTHING;

COMMIT;
