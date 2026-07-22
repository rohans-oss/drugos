-- ============================================================================
-- Drug Repurposing ETL Platform — Widen drugbank_id column to VARCHAR(64)
-- Migration: 015_widen_drugbank_id_column.sql
-- Description: Widen the ``drugs.drugbank_id`` column from VARCHAR(10) to
--              VARCHAR(64) to accommodate synthesized IDs (``SYNTH-DB-...``).
--
-- P1-017 ROOT FIX (Team-2 — synthesized IDs need a wider column):
--   The ``drugs.drugbank_id`` column was VARCHAR(10) — long enough for
--   real DrugBank IDs (``DB00945`` = 7 chars, ``DB123456`` = 8 chars)
--   but TOO SHORT for synthesized IDs:
--     * ``DBSYNTH{6 digits}`` = 13 chars (the OLD synthesized form,
--       now retired — was silently truncated or rejected at the DB level).
--     * ``SYNTH-DB-{8 hex}`` = 17 chars (the NEW synthesized form
--       from ``_synthesize_drugbank_id`` when an InChIKey is available).
--     * ``SYNTH-DB-M{6 digits}`` = 17 chars (the NEW synthesized form
--       for missing-InChIKey drugs).
--   The v50 open-data fallback (``_v50_downloaders.py``) was effectively
--   dead code for missing-InChIKey drugs because the DBSYNTH form didn't
--   fit in VARCHAR(10). ROOT FIX: widen the column to VARCHAR(64) — same
--   width as the ORM model's ``DRUGBANK_ID_LENGTH`` constant (updated in
--   this same P1-017 fix). VARCHAR(64) is generous: it accommodates the
--   current 17-char synthesized form plus any future expansion (e.g. a
--   longer hash or a different prefix) without another migration.
--
--   This migration is SAFE for existing data:
--     * PostgreSQL: ``ALTER TABLE ... ALTER COLUMN ... TYPE VARCHAR(64)``
--       is a metadata-only operation (no table rewrite) when widening a
--       varchar column. Existing indexes are automatically updated.
--     * SQLite: ``ALTER TABLE ... ALTER COLUMN ... TYPE VARCHAR(64)`` is
--       NOT supported (SQLite doesn't support ALTER COLUMN TYPE). The
--       migration runner uses the standard SQLite table-rebuild pattern
--       (create new table, copy data, drop old, rename) — see
--       ``run_migrations.py::_sqlite_alter_column_type``. OR the ORM's
--       ``Base.metadata.create_all()`` on a fresh SQLite DB creates the
--       column at VARCHAR(64) directly (no migration needed for dev/test).
--
-- PREREQUISITES: 001_initial_schema.sql through 012_confidence_tier_pinero_alignment.sql.
--
-- Domains addressed: SCI-1 (ID integrity), DQ (data quality),
--   ARCH (ORM-schema parity).
-- ============================================================================

BEGIN;

-- ===========================================================================
-- Phase 1: Widen the drugs.drugbank_id column
-- ===========================================================================
ALTER TABLE drugs
    ALTER COLUMN drugbank_id TYPE VARCHAR(64);

DO $$
BEGIN
    RAISE NOTICE '  [OK] Widened drugs.drugbank_id from VARCHAR(10) to VARCHAR(64) (P1-017)';
END $$;

-- ===========================================================================
-- Phase 2: Widen the entity_mapping.drugbank_id column (same fix)
-- ===========================================================================
-- entity_mapping.drugbank_id was also VARCHAR(10) in migration 001 (line 1225).
-- It stores the SAME drug IDs as drugs.drugbank_id, so it needs the same widening.
ALTER TABLE entity_mapping
    ALTER COLUMN drugbank_id TYPE VARCHAR(64);

DO $$
BEGIN
    RAISE NOTICE '  [OK] Widened entity_mapping.drugbank_id from VARCHAR(10) to VARCHAR(64) (P1-017)';
END $$;

-- ===========================================================================
-- Phase 3: Schema version metadata
-- ===========================================================================
INSERT INTO schema_version (version, description)
VALUES (
    15,
    'Widen drugbank_id columns (drugs + entity_mapping) from VARCHAR(10) to ' ||
    'VARCHAR(64) to accommodate SYNTH-DB- prefixed synthesized IDs (P1-017)'
)
ON CONFLICT (version) DO NOTHING;

COMMIT;
