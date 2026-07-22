-- ============================================================================
-- Drug Repurposing ETL Platform — Add standard_relation + is_homodimer columns
-- Migration: 018_add_standard_relation_and_is_homodimer.sql
-- Version: 18
--
-- P1-051 ROOT FIX (v110): the audit's regression guard
-- (``tests/test_database_schema.py``) found that TWO columns declared in the
-- ORM (``database/models.py``) were MISSING from the SQL migration files:
--
--   1. ``drug_protein_interactions.standard_relation`` (String(5), nullable)
--      INT-003 ROOT FIX: ChEMBL's standard_relation carries censoring
--      semantics ('=', '<', '>', '~') that distinguish exact measurements
--      from bounds. Without this column, the RL ranker treats IC50 > 100uM
--      (weak binder) the same as IC50 = 1nM (potent inhibitor) — a
--      patient-safety-grade bug.
--
--   2. ``protein_protein_interactions.is_homodimer`` (Boolean, NOT NULL
--      DEFAULT FALSE)
--      v91 ROOT FIX (BUG #9): True when protein_a_id == protein_b_id
--      (self-interaction / homodimer). Biologically critical: EGFR
--      dimerization, p53 tetramerization. Without this column, the
--      Graph Transformer cannot distinguish homodimers from heterodimers,
--      biasing the link-prediction model.
--
-- ROOT FIX: add both columns via ALTER TABLE. Both ALTERs use IF NOT EXISTS
-- for idempotency. The CHECK constraint on standard_relation (matching the
-- ORM's chk_dpi_standard_relation) is added in a DO block for cross-dialect
-- portability. The is_homodimer column has the same server_default + CHECK
-- pattern as the other boolean columns (is_fda_approved, is_withdrawn, etc.)
-- per the v90 ROOT FIX (BUG #23) three-way boolean default unification.
--
-- PREREQUISITES: 001_initial_schema.sql through 017_confidence_tier_add_very_strong.sql.
-- IDEMPOTENT: safe to run multiple times (every statement uses IF NOT EXISTS
-- or IF EXISTS).
-- ============================================================================

BEGIN;

-- ===========================================================================
-- Phase 1: Add drug_protein_interactions.standard_relation
-- ===========================================================================
ALTER TABLE drug_protein_interactions
    ADD COLUMN IF NOT EXISTS standard_relation VARCHAR(5);

-- Add the CHECK constraint matching the ORM's chk_dpi_standard_relation.
-- Valid ChEMBL relations: '=', '<', '>', '~', '<=', '>=', '<<', '>>'.
-- NULL is allowed (some sources don't emit a relation).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_dpi_standard_relation'
          AND conrelid = 'drug_protein_interactions'::regclass
    ) THEN
        ALTER TABLE drug_protein_interactions
            ADD CONSTRAINT chk_dpi_standard_relation
            CHECK (
                standard_relation IS NULL
                OR standard_relation IN ('=', '<', '>', '~', '<=', '>=', '<<', '>>')
            );
        RAISE NOTICE 'P1-051: added chk_dpi_standard_relation';
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_dpi_standard_relation
    ON drug_protein_interactions (standard_relation)
    WHERE standard_relation IS NOT NULL;

-- ===========================================================================
-- Phase 2: Add protein_protein_interactions.is_homodimer
-- ===========================================================================
ALTER TABLE protein_protein_interactions
    ADD COLUMN IF NOT EXISTS is_homodimer BOOLEAN NOT NULL DEFAULT FALSE;

-- Backfill from existing data: if protein_a_id == protein_b_id, it's a homodimer.
UPDATE protein_protein_interactions
SET is_homodimer = TRUE
WHERE protein_a_id = protein_b_id AND is_homodimer = FALSE;

UPDATE protein_protein_interactions
SET is_homodimer = FALSE
WHERE protein_a_id != protein_b_id AND is_homodimer = TRUE;

-- Add the invariant CHECK matching the ORM's chk_ppi_homodimer_invariant.
-- Homodimer rows MUST have protein_a_id == protein_b_id; heterodimer rows
-- MUST have protein_a_id != protein_b_id. This is the v91 ROOT FIX (BUG #9)
-- defense-in-depth invariant.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_ppi_homodimer_invariant'
          AND conrelid = 'protein_protein_interactions'::regclass
    ) THEN
        ALTER TABLE protein_protein_interactions
            ADD CONSTRAINT chk_ppi_homodimer_invariant
            CHECK (
                (protein_a_id = protein_b_id AND is_homodimer = TRUE)
                OR (protein_a_id != protein_b_id AND is_homodimer = FALSE)
            );
        RAISE NOTICE 'P1-051: added chk_ppi_homodimer_invariant';
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_ppi_is_homodimer
    ON protein_protein_interactions (is_homodimer)
    WHERE is_homodimer = TRUE;

-- ===========================================================================
-- Phase 3: Schema version metadata
-- ===========================================================================
INSERT INTO schema_version (version, description)
VALUES (
    18,
    'P1-051 ROOT FIX (v110): add drug_protein_interactions.standard_relation (INT-003 '
    'censoring semantics) and protein_protein_interactions.is_homodimer (v91 BUG #9 '
    'homodimer flag). Both columns were declared in the ORM but missing from the SQL '
    'migrations — schema drift caught by tests/test_database_schema.py.'
)
ON CONFLICT (version) DO NOTHING;

COMMIT;
