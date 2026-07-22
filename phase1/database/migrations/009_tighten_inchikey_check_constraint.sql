-- ============================================================================
-- Drug Repurposing ETL Platform — Tighten InChIKey CHECK constraint
-- Migration: 009_tighten_inchikey_check_constraint.sql
-- Description: Replace the over-permissive chk_drugs_inchikey_format
--              constraint (which accepted TEST/OUTER/INNER/IK% prefixes)
--              with a strict version that mirrors the Python-side
--              ``INCHIKEY_REGEX`` in phase2/drugos_graph/config.py.
--
-- v28 ROOT FIX (audit TOP-17):
--   The Python validator ``validate_inchikey`` uses the strict 27-char
--   regex ``^[A-Z]{14}-[A-Z]{10}-[A-Z]$``. But the SQL CHECK constraint
--   in 001_initial_schema.sql (lines 225-236) ALSO accepted:
--     * TEST%      — arbitrary test fixtures
--     * OUTER%     — biology outer-membrane markers (no InChIKey equivalent)
--     * INNER%     — biology inner-membrane markers (no InChIKey equivalent)
--     * LIKE 'IK%' with LENGTH <= 30 — broad prefix match
--   This divergence meant biologics records (e.g. an "OUTER_MEMBRANE_P35"
--   identifier) were REJECTED by Python at the cleaning layer but ACCEPTED
--   by SQL at the database layer. The two layers silently disagreed, so
--   dev DBs (ORM-created via BasePipeline._ensure_directories auto-init)
--   accumulated rows that production DBs (migration-created) rejected —
--   exactly the "same schema everywhere" guarantee broken (audit P1-ER
--   finding 3, "InChIKey validation is dangerously permissive").
--
--   The new constraint mirrors Python EXACTLY for the canonical case
--   (27-char strict InChIKey) PLUS the SYNTH% escape hatch. SYNTH% is
--   retained because every dev fixture in tests/fixtures/ uses
--   SYNTH0001..SYNTH9999 as synthetic compound identifiers — these are
--   not chemistry and are clearly labelled as synthetic. TEST/OUTER/
--   INNER/IK% are removed because they have no equivalent on the Python
--   side and were the source of the silent divergence.
--
-- PREREQUISITES: 001_initial_schema.sql through 006_drug_withdrawn_safety_columns.sql.
--
-- Domains addressed: SCI-1 (InChIKey integrity), DQ (data quality),
--   ARCH (ORM-schema parity).
-- ============================================================================

BEGIN;

-- ===========================================================================
-- Phase 1: Pre-migration data audit
-- ===========================================================================
-- v28 ROOT FIX: report rows that WILL be invalid under the new constraint
-- so operators can fix or quarantine them BEFORE the constraint swap.
-- This is a NOTICE-only audit; it does not modify data. The subsequent
-- ALTER will FAIL if any row violates the new CHECK, so this block gives
-- operators a chance to clean up first.
-- v29 ROOT FIX: the audit predicate now uses the SAME regex as the
-- constraint (canonical 27-char OR tightened SYNTH[A-Z0-9-]{4,30}),
-- not just LENGTH=27.
-- v110 Task 32 root fix: SYNTH escape hatch tightened from '^SYNTH' to
-- '^SYNTH[A-Z0-9-]{4,30}$' (alphanumeric+dash only, 4-30 chars after SYNTH).
DO $$
DECLARE
    _bad_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO _bad_count
    FROM drugs
    WHERE NOT (
        inchikey ~ '^[A-Z]{14}-[A-Z]{10}-[A-Z]$'
        OR inchikey ~ '^SYNTH[A-Z0-9-]{4,30}$'
    );
    IF _bad_count > 0 THEN
        RAISE WARNING
            '  [AUDIT] % row(s) in drugs.inchikey will VIOLATE the new '
            'chk_drugs_inchikey_format constraint (canonical 27-char InChIKey '
            'regex ^[A-Z]{14}-[A-Z]{10}-[A-Z]$ OR SYNTH%% prefix only). '
            'These rows must be fixed or quarantined before this migration '
            'can complete.',
            _bad_count;
    ELSE
        RAISE NOTICE '  [OK] No rows in drugs.inchikey will violate the new constraint';
    END IF;
END $$;

-- ===========================================================================
-- Phase 2: Drop the over-permissive constraint
-- ===========================================================================
ALTER TABLE drugs DROP CONSTRAINT IF EXISTS chk_drugs_inchikey_format;
-- v42 FORENSIC ROOT FIX (P0-6): the previous ``RAISE NOTICE`` was a
-- STANDALONE statement OUTSIDE any DO $$ ... $$ block. RAISE is a PL/pgSQL
-- statement, not valid standalone SQL. PostgreSQL raised
-- ``syntax error at or near 'RAISE'`` the moment this migration was applied,
-- the tightened constraint never took effect, the schema_version row was
-- never inserted, and migration 009 was recorded as failed forever.
-- ROOT FIX: wrap the RAISE NOTICE in a DO dollar-dollar BEGIN ... END dollar-dollar semicolon block so it
-- is parsed as PL/pgSQL. This mirrors the pattern already used at line 51.
DO $$
BEGIN
    RAISE NOTICE '  [OK] Dropped old chk_drugs_inchikey_format (accepted TEST/OUTER/INNER/IK)';
END $$;

-- ===========================================================================
-- Phase 3: Add the tightened constraint
-- ===========================================================================
-- v57 ROOT FIX (T-003 — Scientific Correctness):
--   The previous "tightened" constraint was byte-identical to migration 001
--   (LENGTH=27 OR LIKE 'SYNTH%'). Any 27-char ASCII string passed
--   (e.g. 'TESTTESTTESTTESTTESTTESTTES'). The "tightening" was theatre.
--
--   FIX: use PostgreSQL's POSIX regex operator (~) to enforce the canonical
--   InChIKey format: ^[A-Z]{14}-[A-Z]{10}-[A-Z]$. This is the same regex
--   enforced in Python (cleaning._constants.INCHIKEY_REGEX).
--
--   PostgreSQL DOES support ~ (POSIX regex) — the previous comment claiming
--   "SQLite lacks the operator" was true for SQLite but irrelevant: this
--   migration only runs against PostgreSQL (SQLite is dev-only and uses
--   SQLAlchemy's CHECK emulation, which DOES support ~ via REGEXP operator
--   when the SQLite REGEXP function is registered, OR falls back to
--   LENGTH-based check).
--
--   For SQLite dev/test compatibility, we wrap the regex check in a
--   plpgsql DO block (PostgreSQL only) and provide a LENGTH-based
--   fallback for non-PostgreSQL dialects. The Python validator
--   (is_canonical_inchikey) is the AUTHORITATIVE source of truth and
--   catches 27-char gibberish BEFORE data reaches the DB.
ALTER TABLE drugs
    DROP CONSTRAINT IF EXISTS chk_drugs_inchikey_format;

DO $$
BEGIN
    -- PostgreSQL: use POSIX regex for strict InChIKey validation.
    -- v110 Task 32 root fix: tightened the SYNTH escape hatch from
    -- ``LIKE 'SYNTH%'`` (which accepted ANY string starting with SYNTH,
    -- including 'SYNTH_GARBAGE_123!!' with spaces/punctuation) to
    -- ``~ '^SYNTH[A-Z0-9-]{4,30}$'`` which requires SYNTH followed by
    -- 4-30 uppercase alphanumeric characters or dashes ONLY. This matches
    -- the actual synthetic ID formats used in the codebase:
    --   SYNTH0001..SYNTH9999           (test fixtures)
    --   SYNTH-DB-[0-9A-F]{8}           (drugbank synthesized drug IDs)
    --   SYNTH-DB-M\d{6}                (drugbank synthesized mixture IDs)
    -- Real InChIKeys follow the canonical 27-char IUPAC format
    -- (14 + dash + 10 + dash + 1), e.g. BQJCRHHNABKAKU-XKUOQXGZSA-N (aspirin).
    -- The audit's claim of "14-2-8" format is scientifically WRONG — the
    -- actual IUPAC spec is 14-10-1. The canonical regex below is correct.
    -- Ref: https://www.inchi-trust.org/technical-faq/#2
    ALTER TABLE drugs
        ADD CONSTRAINT chk_drugs_inchikey_format
        CHECK (
            inchikey ~ '^[A-Z]{14}-[A-Z]{10}-[A-Z]$'
            OR inchikey ~ '^SYNTH[A-Z0-9-]{4,30}$'
        );
    RAISE NOTICE '  [OK] Added tightened chk_drugs_inchikey_format (POSIX regex + tightened SYNTH escape hatch)';
EXCEPTION
    WHEN feature_not_supported OR syntax_error THEN
        -- P1-015 ROOT FIX (Team-2 — SQLite fallback now uses real REGEXP):
        --   The previous SQLite fallback used
        --   ``LENGTH(inchikey) = 27 AND SUBSTR(inchikey, 15, 1) = '-' AND
        --   SUBSTR(inchikey, 26, 1) = '-'`` — a weak check that accepted
        --   any 27-char string with hyphens at positions 15 and 26,
        --   including digits (``11111111111111-2222222222-3``), lowercase
        --   (``aaaaaaaaaaaaaa-bbbbbbbbbb-c``), and punctuation
        --   (``!!!!!!!!!!!!!!-!!!!!!!!!!-!``). Dev DBs (SQLite) accepted
        --   gibberish InChIKeys that prod PostgreSQL rejected — a
        --   dev/prod asymmetry footgun.
        --   ROOT FIX: ``database/connection.py`` now registers a SQLite
        --   REGEXP function via ``create_function`` (see
        --   ``_register_sqlite_regexp_function`` in
        --   ``_attach_lifecycle_events``). The migration runner
        --   (``run_migrations.py``) translates PostgreSQL's ``~``
        --   operator to SQLite's ``REGEXP`` operator. This fallback
        --   block is now ONLY reached when the migration is run on a
        --   SQLite engine that does NOT have the REGEXP function
        --   registered (e.g. a test that bypasses ``connection.py``).
        --   In that case, we use the SAME REGEXP form — if the function
        --   is not registered, the CHECK will raise on the first INSERT,
        --   surfacing the missing registration immediately (BY DESIGN).
        ALTER TABLE drugs
            ADD CONSTRAINT chk_drugs_inchikey_format
            CHECK (
                inchikey REGEXP '^[A-Z]{14}-[A-Z]{10}-[A-Z]$'
                OR inchikey REGEXP '^SYNTH[A-Z0-9-]{4,30}$'
            );
        RAISE NOTICE '  [OK] Added fallback chk_drugs_inchikey_format (SQLite REGEXP — identical semantics to PostgreSQL ~, including tightened SYNTH escape hatch)';
END $$;

COMMENT ON CONSTRAINT chk_drugs_inchikey_format ON drugs IS
    'InChIKey format: PostgreSQL POSIX regex ^[A-Z]{14}-[A-Z]{10}-[A-Z]$ '
    'OR SYNTH% prefix (synthetic test fixtures). v57 ROOT FIX (T-003) — '
    'replaces the byte-identical-to-001 LENGTH=27 backstop with actual '
    'canonical-format enforcement. The Python validator '
    '(cleaning._constants.is_canonical_inchikey) is the authoritative source.';

-- ===========================================================================
-- Phase 4: Schema version metadata
-- ===========================================================================
INSERT INTO schema_version (version, description)
VALUES (
    9,
    'Tighten chk_drugs_inchikey_format to mirror Python INCHIKEY_REGEX '
    '(27-char canonical OR SYNTH% only; removed TEST/OUTER/INNER/IK% clauses)'
)
ON CONFLICT (version) DO NOTHING;

COMMIT;
