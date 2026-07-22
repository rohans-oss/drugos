-- v49 ROOT FIX migration 010 — Drug indication columns
-- =====================================================================
-- v73 ROOT FIX (T-008 — atomicity + version-column type contract):
--   The previous 010 file was the ONLY migration in the chain (001-010)
--   that lacked a ``BEGIN;`` ... ``COMMIT;`` wrapper. Every other
--   migration is atomic by file convention. The PostgreSQL migration
--   runner (run_migrations.py:3880) opens an explicit ``engine.begin()``
--   per migration so the missing wrapper was tolerated there — but the
--   raw ``psql -f 010_drug_indication_columns.sql`` path used by
--   operators running migrations manually had NO transaction boundary,
--   so a partial failure (e.g. ``uq_gda_gene_id_disease_source`` index
--   creation succeeded but the ``indication`` column ADD failed) left
--   the schema half-applied with no ``schema_version`` row to record
--   the attempt. The next ``run_migrations()`` invocation saw no
--   version=10 row and re-ran 010, which then failed on the
--   already-existing index.
--
--   Additionally, the previous INSERT used the STRING literal ``'010'``
--   for the INTEGER ``schema_version.version`` column
--   (``001_initial_schema.sql:84`` declares ``version INTEGER PRIMARY``).
--   PostgreSQL silently coerces ``'010'::INTEGER = 10``, but every other
--   migration (001-009) uses bare INTEGER literals (1, 2, ..., 9). The
--   inconsistency breaks any future type change to ``schema_version``
--   (e.g. migrating to TEXT version strings): 010 would crash while
--   001-009 survive. ``MAX(version)`` could also behave unexpectedly
--   if mixed string/integer literals are present.
--
--   ROOT FIX: wrap the entire migration in ``BEGIN;`` ... ``COMMIT;``
--   (matching 001-009 file convention) and use the bare INTEGER literal
--   ``10`` (matching 001-009's integer convention).
--
-- ROOT CAUSE being fixed:
--   The Drug ORM had no `indication` column. The DrugBank pipeline
--   already produced `indication` in its drugs_df, but the loader's
--   `_filter_to_drug_columns` silently dropped it because the ORM had
--   no attribute. The Phase 2 bridge (`phase2/drugos_graph/phase1_bridge.py`)
--   needs this column to synthesize Compound-treats-Disease edges
--   (the TransE link-prediction target) when running in PostgreSQL mode.
--   Without it, PostgreSQL mode produced ZERO prediction-target edges.
--
-- COMPOUND CHAIN this closes:
--   Compound Chain 2 ("PostgreSQL Bridge Data Corruption"):
--     Drug ORM has no `indication` column
--     → bridge `indications` DataFrame is empty
--     → ZERO Compound-treats-Disease edges
--     → `positive_pairs_sufficient = False`
--     → V1 launch impossible in pure-PostgreSQL mode
--
-- ALSO adds:
--   `indication_source` (String(30)) — provenance tag
--     ('drugbank_xml' | 'chembl_max_phase' | 'rxnorm' | 'manual')
--   So downstream consumers (RL ranker, hypothesis explorer) can
--   filter indications by confidence.
--
--   Unique index on (gene_id, disease_id, source) for
--   gene_disease_associations — closes the bulk_upsert_gda fallback
--   chain (v49 Compound-5). Without this index, `ON CONFLICT
--   (gene_id, disease_id, source)` always raised an error and
--   fell back to row-by-row inserts (10-100× slower + ERROR spam).

BEGIN;

-- ─── 1. Drug indication columns ───────────────────────────────────────
ALTER TABLE drugs
    ADD COLUMN IF NOT EXISTS indication TEXT;
ALTER TABLE drugs
    ADD COLUMN IF NOT EXISTS indication_source VARCHAR(30);

-- Backfill comment (no data to backfill on a fresh DB; documented for
-- operators upgrading from v48):
--   UPDATE drugs SET indication = ..., indication_source = 'drugbank_xml'
--   WHERE id IN (SELECT id FROM drugs WHERE indication IS NULL);
-- The DrugBank pipeline's next run will populate these columns
-- automatically via the v49 `_filter_to_drug_columns` fix.

-- ─── 2. Unique index on (gene_id, disease_id, source) ────────────────
-- v49 ROOT FIX (Compound-5 — bulk_upsert_gda always falls back):
-- The v48 schema only had a unique constraint on
-- (gene_symbol, disease_id, source). When `gene_id` was populated
-- (NCBI Entrez Gene ID — the stable identifier), bulk_upsert_gda
-- used `ON CONFLICT (gene_id, disease_id, source)` which raised:
--   "there is no unique or exclusion constraint matching the ON
--    CONFLICT specification"
-- and fell back to row-by-row inserts. ROOT FIX: add a partial
-- unique index on (gene_id, disease_id, source) WHERE gene_id IS
-- NOT NULL. The partial index lets rows without gene_id (legacy
-- data) still coexist with rows that have it.

CREATE UNIQUE INDEX IF NOT EXISTS uq_gda_gene_id_disease_source
    ON gene_disease_associations (gene_id, disease_id, source)
    WHERE gene_id IS NOT NULL;

-- Also add a B-tree index on gene_id alone for fast lookup
-- (used by entity resolution cross-source joins).
-- v90 ROOT FIX (BUG #19 — P1 idx_gda_gene_id full vs partial conflict):
--   The previous code created a PARTIAL index named ``idx_gda_gene_id``
--   with ``WHERE gene_id IS NOT NULL``. But migration 004 (line 134)
--   already created a FULL index named ``idx_gda_gene_id`` (no WHERE
--   clause). PostgreSQL's ``CREATE INDEX IF NOT EXISTS`` is a NO-OP if
--   an index with the same name already exists — it does NOT replace
--   it. Migration 004 runs first (version 4 < version 10), creating the
--   FULL index. Migration 010's partial index creation was silently
--   skipped. The partial-index optimization (smaller index, faster
--   writes for NULL-gene_id rows) was never realised. ROOT FIX: use a
--   DIFFERENT name (``idx_gda_gene_id_partial``) for the partial index
--   so it actually gets created alongside the full index from migration
--   004. The full index (``idx_gda_gene_id``) remains for general
--   queries; the partial index (``idx_gda_gene_id_partial``) is an
--   optimization for queries that filter ``WHERE gene_id IS NOT NULL``.
CREATE INDEX IF NOT EXISTS idx_gda_gene_id_partial
    ON gene_disease_associations (gene_id)
    WHERE gene_id IS NOT NULL;

-- ─── 3. Schema version bump ──────────────────────────────────────────
-- v73 ROOT FIX (T-008): use INTEGER literal 10, NOT string '010'.
-- ``schema_version.version`` is declared INTEGER PRIMARY KEY in
-- migration 001 line 84. Every other migration (001-009) uses bare
-- INTEGER literals (1, 2, ..., 9). Using a STRING here would rely on
-- implicit PostgreSQL text-to-integer coercion, which breaks if the
-- column type is ever changed and is inconsistent with the rest of
-- the migration chain. ``applied_at`` is omitted — the column has a
-- DEFAULT NOW() (001 line 88) so it auto-populates.
INSERT INTO schema_version (version, description)
VALUES (10, 'v49: add drugs.indication + indication_source; add uq_gda_gene_id_disease_source partial unique index')
ON CONFLICT (version) DO NOTHING;

COMMIT;
