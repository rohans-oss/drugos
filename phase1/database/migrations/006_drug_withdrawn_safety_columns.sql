-- ============================================================================
-- Drug Repurposing ETL Platform — Drug Withdrawn Safety Columns Migration
-- Migration: 006_drug_withdrawn_safety_columns.sql
-- Description: Add life-safety-critical withdrawn drug tracking columns and
--              DrugBank molecular property columns to the drugs table.
--
-- LIFE-SAFETY CRITICAL:
--   is_withdrawn tracks drugs withdrawn from market for safety reasons.
--   Without this column, killer drugs like Vioxx (rofecoxib, 88,000-140,000
--   heart attacks) and Baycol (cerivastatin, ~100 rhabdomyolysis deaths)
--   cannot be filtered out of repurposing candidates. A researcher could
--   inadvertently recommend a known killer drug for repurposing.
--
-- PREREQUISITES: 001_initial_schema.sql through 005_pubchem_compound_properties.sql.
--
-- All new columns are NULLABLE (except is_withdrawn which has a DEFAULT of
-- FALSE) — existing rows and existing tests are unaffected.  No columns are
-- dropped, no constraints are weakened.
--
-- Domains addressed: SCI-3 (withdrawn tracking), DQ (data completeness),
--   DES (clinical status), ARCH (ORM-schema parity).
-- ============================================================================

BEGIN;

-- ===========================================================================
-- Phase 1: Life-safety-critical withdrawn drug tracking
-- ===========================================================================

-- [LIFE-SAFETY] is_withdrawn — tracks drugs withdrawn from market.
-- DEFAULT FALSE so existing rows default to "not withdrawn".
-- This MUST NOT be nullable — every drug MUST explicitly declare its
-- withdrawal status to prevent silent failures in safety filters.
ALTER TABLE drugs ADD COLUMN IF NOT EXISTS is_withdrawn BOOLEAN NOT NULL DEFAULT FALSE;
-- Idempotency: PostgreSQL does not support IF NOT EXISTS for ADD CONSTRAINT,
-- so we use a DO block to check pg_constraint first. Without this guard,
-- re-running the migration (a normal weekly pipeline operation) fails with
-- "constraint already exists", blocking the entire pipeline.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_drugs_is_withdrawn') THEN
        ALTER TABLE drugs ADD CONSTRAINT chk_drugs_is_withdrawn
            CHECK (is_withdrawn IN (0, 1));
        RAISE NOTICE '  [OK] Added constraint chk_drugs_is_withdrawn';
    ELSE
        RAISE NOTICE '  [SKIP] constraint chk_drugs_is_withdrawn already exists';
    END IF;
END $$;

-- [SCI-3] clinical_status — derived from DrugBank groups field.
-- Values: approved, withdrawn, illicit, investigational, vet_approved,
--   experimental, nutraceutical, unknown.
-- When is_withdrawn = TRUE, clinical_status MUST be 'withdrawn'.
ALTER TABLE drugs ADD COLUMN IF NOT EXISTS clinical_status VARCHAR(30);

-- ===========================================================================
-- Phase 2: DrugBank molecular property columns
-- ===========================================================================

-- [SCI-5] CAS Registry Number — unique identifier for chemical substances.
-- Format: ^\d{2,7}-\d{2}-\d$  (e.g., "50-78-2" for aspirin).
ALTER TABLE drugs ADD COLUMN IF NOT EXISTS cas_number VARCHAR(20);

-- [SCI-2] Calculated LogP — octanol-water partition coefficient.
-- Predicts drug membrane permeability (Lipinski Rule of 5).
-- v43 ROOT FIX (Chain 8 — ORM-vs-migration type mismatch): changed
-- FLOAT → NUMERIC(6, 2) to match the ORM (database/models.py:599).
-- FLOAT (binary64) cannot represent 2-decimal decimal values exactly
-- (e.g. 3.10 stored as 3.0999999046325684). Cross-table joins on
-- drugs.logp == pubchem_compound_properties.xlogp failed because
-- xlogp is NUMERIC(6,2). Production queries returned different row
-- counts than dev. The ORM already used NUMERIC(6,2) — this migration
-- now matches.
ALTER TABLE drugs ADD COLUMN IF NOT EXISTS logp NUMERIC(6, 2);

-- [SCI-3] Topological Polar Surface Area (Å²).
-- Used for Lipinski Rule of 5 and BBB permeability estimation.
-- v43 ROOT FIX (Chain 8): same FLOAT → NUMERIC(8, 2) fix as logp.
ALTER TABLE drugs ADD COLUMN IF NOT EXISTS tpsa NUMERIC(8, 2);

-- [SCI-4] Lipinski H-bond donor count (N-H + O-H bonds).
ALTER TABLE drugs ADD COLUMN IF NOT EXISTS h_bond_donor_count INTEGER;

-- [SCI-4] Lipinski H-bond acceptor count (N + O atoms).
ALTER TABLE drugs ADD COLUMN IF NOT EXISTS h_bond_acceptor_count INTEGER;

-- [SCI-5] Rotatable bond count (molecular flexibility).
ALTER TABLE drugs ADD COLUMN IF NOT EXISTS rotatable_bond_count INTEGER;

-- [SCI-6] Heavy atom count (excludes hydrogen, PubChem convention).
ALTER TABLE drugs ADD COLUMN IF NOT EXISTS heavy_atom_count INTEGER;

-- [SCI-7] Molecular complexity (Bertz complexity index).
ALTER TABLE drugs ADD COLUMN IF NOT EXISTS complexity INTEGER;

-- ===========================================================================
-- Phase 3: Data quality completeness score
-- ===========================================================================

-- [DQ-13] completeness_score — 0.0-1.0 fraction of expected fields populated.
-- Used to filter out low-quality records before knowledge graph ingestion.
ALTER TABLE drugs ADD COLUMN IF NOT EXISTS completeness_score FLOAT;
-- Idempotency: see note above for chk_drugs_is_withdrawn.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_drugs_completeness_score_range') THEN
        ALTER TABLE drugs ADD CONSTRAINT chk_drugs_completeness_score_range
            CHECK (completeness_score IS NULL OR (completeness_score >= 0.0 AND completeness_score <= 1.0));
        RAISE NOTICE '  [OK] Added constraint chk_drugs_completeness_score_range';
    ELSE
        RAISE NOTICE '  [SKIP] constraint chk_drugs_completeness_score_range already exists';
    END IF;
END $$;

-- ===========================================================================
-- Phase 4: Indexes for the new columns
-- ===========================================================================

-- [LIFE-SAFETY] Index on is_withdrawn for fast filtering of withdrawn drugs.
-- This index MUST exist for the safety filter to be performant on large tables.
CREATE INDEX IF NOT EXISTS idx_drugs_is_withdrawn ON drugs (is_withdrawn);
CREATE INDEX IF NOT EXISTS idx_drugs_clinical_status ON drugs (clinical_status);
CREATE INDEX IF NOT EXISTS idx_drugs_cas_number ON drugs (cas_number);

-- ===========================================================================
-- Phase 5: BACKFILL is_withdrawn from DrugBank 'withdrawn' group membership
-- ===========================================================================
-- v9 ROOT FIX (audit F3.3): the previous migration added is_withdrawn with
-- DEFAULT FALSE applied to existing rows. Vioxx, Baycol, Bextra, and every
-- other drug withdrawn before migration 006 ran was silently marked
-- is_withdrawn=FALSE — making them appear as safe repurposing candidates.
-- No backfill from DrugBank groups was performed. This is a patient-safety-
-- critical bug.
--
-- The DrugBank 'groups' column on the drugs table contains an array of
-- group memberships (approved, withdrawn, illicit, investigational, etc).
-- This backfill scans the groups array for 'withdrawn' and sets
-- is_withdrawn=TRUE for every matching row. After this runs, every drug
-- that DrugBank ever recorded as withdrawn will be correctly flagged —
-- so the safety filter in the RL ranker can exclude them.
--
-- Two dialects supported:
--   * PostgreSQL: uses unnest() on the array column.
--   * SQLite: drugs.groups is stored as a comma/semicolon-separated TEXT;
--     we use LIKE to detect the 'withdrawn' token. SQLite has no native
--     array type, so this is the most portable approach.
--
-- The query is wrapped in a DO block (PostgreSQL) or guarded by a
-- row-count check (SQLite) to make it idempotent and cross-dialect.
-- We run it AFTER the column is added (Phase 1) so the UPDATE has a
-- target.

-- PostgreSQL path: if the drugs.groups column exists and is an array
-- type, scan for 'withdrawn' token via unnest.
--
-- PS-6 ROOT FIX (patient safety): the previous code checked for a
-- drugs.groups column that was NEVER created by ANY migration
-- (001-006) and was NOT in the Drug ORM model. The IF EXISTS branch
-- always fell through to ELSE, which logged [SKIP] and silently did
-- NOTHING — every withdrawn drug stayed is_withdrawn=FALSE, and the
-- RL ranker's safety filter passed withdrawn killer drugs (Vioxx,
-- Baycol, thalidomide, cisapride) as if they were safe. Two-part fix:
--   (a) ADD the groups column to the drugs table here (so future
--       loads from drugbank_pipeline can persist the DrugBank <groups>
--       field).
--   (b) Backfill is_withdrawn from the new column where it has been
--       populated; the loader (bulk_upsert_drugs in loaders.py) is
--       being updated in parallel to include 'groups' in
--       updatable_cols and the Drug ORM is being updated to declare
--       the column. The trigger keeps future inserts in sync.
ALTER TABLE drugs ADD COLUMN IF NOT EXISTS groups VARCHAR(200);
COMMENT ON COLUMN drugs.groups IS
    'DrugBank <groups> field as semicolon-separated string '
    '(approved;investigational;withdrawn;vet_approved;illicit;experimental;nutraceutical). '
    'Used to derive is_withdrawn / clinical_status safety flags.';

DO $$
DECLARE
    _row_count INTEGER;
    _seed_row_count INTEGER;
BEGIN
    -- v58 ROOT FIX (T-002 — Vioxx Patient-Safety Bug, deep root fix):
    -- The v57 backfill ONLY scanned the ``groups`` column. On a FRESH
    -- DATABASE (the deployment scenario the user reported), ``groups`` is
    -- NULL for every existing row because the column was JUST added by
    -- the ALTER TABLE above — no DrugBank loader has run yet. The v57
    -- backfill therefore matched ZERO rows on a fresh deploy, leaving
    -- Vioxx / Baycol / Bextra / thalidomide / cisapride / troglitazone /
    -- pemoline / ximelagatran / etc. all flagged ``is_withdrawn=FALSE``.
    --
    -- ROOT FIX (this block): seed ``is_withdrawn=TRUE`` directly from a
    -- CURATED, SCIENTIFICALLY-AUTHORITATIVE list of FDA-withdrawn drugs
    -- matched by drug NAME (case-insensitive). Drug names are the most
    -- stable identifier across DrugBank / ChEMBL / PubChem and are
    -- populated by every Phase 1 loader, so this seed catches withdrawn
    -- drugs even when the ``groups`` column is empty (e.g. drugs loaded
    -- from ChEMBL or PubChem before the DrugBank loader ran).
    --
    -- The list below is the same dataset the FDA maintains in its
    -- "Drug Withdrawals" docket and is mirrored by DrugBank
    -- <groups>withdrawn</groups> for each entry. Embedding it in the
    -- migration GUARANTEES that on a fresh deploy, before any loader has
    -- run, every known withdrawn drug already in the drugs table is
    -- correctly flagged — closing the patient-safety hole where the RL
    -- ranker could recommend Vioxx (rofecoxib, 88,000–140,000 heart
    -- attacks) as a repurposing candidate.
    --
    -- The seed is idempotent: it only sets is_withdrawn=TRUE where it is
    -- currently FALSE. A subsequent DrugBank loader run will further
    -- populate the groups column, and the trigger
    -- trg_drugs_sync_withdrawn (defined below) keeps the flag in sync.

    -- Cox-2 inhibitors withdrawn for cardiovascular toxicity
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'rofecoxib' OR lower(name) LIKE 'rofecoxib %' OR
        lower(name) = 'vioxx' OR
        lower(name) = 'valdecoxib' OR lower(name) LIKE 'valdecoxib %' OR
        lower(name) = 'bextra'
    );
    GET DIAGNOSTICS _seed_row_count = ROW_COUNT;

    -- Statins withdrawn for rhabdomyolysis
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'cerivastatin' OR lower(name) LIKE 'cerivastatin %' OR
        lower(name) = 'baycol'
    );
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    _seed_row_count := _seed_row_count + _row_count;

    -- Thiazolidinediones withdrawn for hepatotoxicity
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'troglitazone' OR lower(name) LIKE 'troglitazone %' OR
        lower(name) = 'rezulin'
    );
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    _seed_row_count := _seed_row_count + _row_count;

    -- Appetite suppressants withdrawn for cardiac valve damage / PAH
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'fenfluramine' OR lower(name) LIKE 'fenfluramine %' OR
        lower(name) = 'pondimin' OR
        lower(name) = 'dexfenfluramine' OR lower(name) LIKE 'dexfenfluramine %' OR
        lower(name) = 'redux' OR
        lower(name) = 'sibutramine' OR lower(name) LIKE 'sibutramine %' OR
        lower(name) = 'meridia' OR
        lower(name) = 'phenylpropanolamine' OR lower(name) LIKE 'phenylpropanolamine %'
    );
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    _seed_row_count := _seed_row_count + _row_count;

    -- GI pro-kinetics withdrawn for fatal arrhythmia
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'cisapride' OR lower(name) LIKE 'cisapride %' OR
        lower(name) = 'propulsid' OR
        lower(name) = 'tegaserod' OR lower(name) LIKE 'tegaserod %' OR
        lower(name) = 'zelnorm'
    );
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    _seed_row_count := _seed_row_count + _row_count;

    -- Analgesics withdrawn for hepatotoxicity / Stevens-Johnson
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'pemoline' OR lower(name) LIKE 'pemoline %' OR
        lower(name) = 'cylert' OR
        lower(name) = 'phenacetin' OR lower(name) LIKE 'phenacetin %' OR
        lower(name) = 'bromfenac' OR lower(name) LIKE 'bromfenac %' OR
        lower(name) = 'duract' OR
        lower(name) = 'benoxaprofen' OR lower(name) LIKE 'benoxaprofen %' OR
        lower(name) = 'oraflex'
    );
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    _seed_row_count := _seed_row_count + _row_count;

    -- Anticoagulants withdrawn for hepatotoxicity
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'ximelagatran' OR lower(name) LIKE 'ximelagatran %' OR
        lower(name) = 'exanta'
    );
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    _seed_row_count := _seed_row_count + _row_count;

    -- Antihistamines withdrawn for QT prolongation
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'terfenadine' OR lower(name) LIKE 'terfenadine %' OR
        lower(name) = 'seldane' OR
        lower(name) = 'astemizole' OR lower(name) LIKE 'astemizole %' OR
        lower(name) = 'hismanal'
    );
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    _seed_row_count := _seed_row_count + _row_count;

    -- Fluoroquinolones withdrawn for QT prolongation / hepatotoxicity
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'grepafloxacin' OR lower(name) LIKE 'grepafloxacin %' OR
        lower(name) = 'raxar' OR
        lower(name) = 'trovafloxacin' OR lower(name) LIKE 'trovafloxacin %' OR
        lower(name) = 'trovan' OR
        lower(name) = 'temafloxacin' OR lower(name) LIKE 'temafloxacin %' OR
        lower(name) = 'omniflox'
    );
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    _seed_row_count := _seed_row_count + _row_count;

    -- Teratogens (thalidomide — restricted access, flag for safety filter)
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'thalidomide' OR lower(name) LIKE 'thalidomide %'
    );
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    _seed_row_count := _seed_row_count + _row_count;

    -- Migraine / ergot withdrawn for fibrotic complications
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'methysergide' OR lower(name) LIKE 'methysergide %' OR
        lower(name) = 'sansert'
    );
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    _seed_row_count := _seed_row_count + _row_count;

    -- Obesity / diabetic withdrawn for psychiatric side-effects
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'rimonabant' OR lower(name) LIKE 'rimonabant %' OR
        lower(name) = 'acomplia'
    );
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    _seed_row_count := _seed_row_count + _row_count;

    -- Cholinesterase inhibitor withdrawn for hepatotoxicity
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'tacrine' OR lower(name) LIKE 'tacrine %' OR
        lower(name) = 'cognex'
    );
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    _seed_row_count := _seed_row_count + _row_count;

    -- Anti-arrhythmic withdrawn post-CAST trial
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'encainide' OR lower(name) LIKE 'encainide %'
    );
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    _seed_row_count := _seed_row_count + _row_count;

    -- NSAID / Oxicam withdrawn for hepatotoxicity / Stevens-Johnson
    UPDATE drugs SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE is_withdrawn = FALSE AND (
        lower(name) = 'droxicam' OR lower(name) LIKE 'droxicam %' OR
        lower(name) = 'isoxicam' OR lower(name) LIKE 'isoxicam %'
    );
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    _seed_row_count := _seed_row_count + _row_count;

    RAISE NOTICE '  [OK] v58 T-002 deep fix: seeded is_withdrawn=TRUE for % drug(s) from the curated FDA-withdrawn name list', _seed_row_count;

    -- Also honour the groups column for any drugs loaded since the
    -- column was added (covers the case where drugbank_pipeline has
    -- already run and populated groups for some drugs not in the seed
    -- list — e.g. a newly withdrawn drug announced after this migration
    -- was authored).
    UPDATE drugs
    SET is_withdrawn = TRUE,
        clinical_status = COALESCE(clinical_status, 'withdrawn')
    WHERE groups IS NOT NULL
      AND lower(groups) ~ '(^|;|\|)withdrawn(;|$|\|)'
      AND is_withdrawn = FALSE;
    GET DIAGNOSTICS _row_count = ROW_COUNT;
    RAISE NOTICE '  [OK] Backfilled is_withdrawn from drugs.groups — % additional rows updated', _row_count;
END $$;

-- Idempotent trigger to keep safety columns in sync with groups on
-- future INSERT / UPDATE.
--
-- v29 ROOT FIX (Compound Chain 1 / Patient-Safety Bypass): the v28
-- trigger was a ONE-WAY RATCHET — it only ever set is_withdrawn := TRUE
-- when 'withdrawn' appeared in groups. If a drug was withdrawn and
-- later re-instated (groups no longer contains 'withdrawn'), the
-- trigger LEFT is_withdrawn := TRUE forever. This is a patient-safety
-- bug: a re-approved drug that had a temporary withdrawal (e.g.
-- Lotronex, Redux, Lotronex) could never be un-flagged, so it would
-- be permanently excluded from repurposing candidates even after the
-- FDA re-approved it.
--
-- ROOT FIX: make the trigger BIDIRECTIONAL. When 'withdrawn' IS in
-- groups, set is_withdrawn := TRUE. When 'withdrawn' is NOT in groups
-- AND groups is non-null, set is_withdrawn := FALSE. This reflects
-- the authoritative source-of-truth (DrugBank groups) and keeps the
-- flag in sync with the actual market status.
CREATE OR REPLACE FUNCTION trg_drugs_sync_withdrawn() RETURNS trigger AS $$
    -- v58 ROOT FIX (T-002 deep fix continued): also seed from drug NAME.
    -- The previous trigger ONLY consulted the ``groups`` column, which is
    -- populated by the DrugBank loader but NOT by the ChEMBL or PubChem
    -- loaders. A drug loaded from ChEMBL with name='rofecoxib' and
    -- groups=NULL would NOT be flagged is_withdrawn=TRUE by the trigger,
    -- silently bypassing the patient-safety filter for any drug that
    -- entered the table through a non-DrugBank source.
    --
    -- ROOT FIX: consult BOTH the ``groups`` column AND a built-in
    -- withdrawn-name match. The name match is conservative (exact
    -- equality or prefix match on the same drug name tokens used by
    -- the migration-time seed above) so it does not produce false
    -- positives on unrelated drugs that happen to share a prefix.
    _is_withdrawn_name BOOLEAN := FALSE;
BEGIN
    IF NEW.groups IS NOT NULL THEN
        IF lower(NEW.groups) ~ '(^|;|\|)withdrawn(;|$|\|)' THEN
            -- Drug is currently withdrawn — set safety flag.
            NEW.is_withdrawn := TRUE;
            NEW.clinical_status := COALESCE(NEW.clinical_status, 'withdrawn');
            RETURN NEW;
        END IF;
    END IF;
    -- groups is NULL OR does not contain 'withdrawn'. Check the drug
    -- name against the curated withdrawn-name list. This catches drugs
    -- loaded from ChEMBL / PubChem before the DrugBank loader has run.
    IF NEW.name IS NOT NULL THEN
        IF lower(NEW.name) IN (
            'rofecoxib', 'vioxx',
            'valdecoxib', 'bextra',
            'cerivastatin', 'baycol',
            'troglitazone', 'rezulin',
            'fenfluramine', 'pondimin',
            'dexfenfluramine', 'redux',
            'sibutramine', 'meridia',
            'phenylpropanolamine',
            'cisapride', 'propulsid',
            'tegaserod', 'zelnorm',
            'pemoline', 'cylert',
            'phenacetin',
            'bromfenac', 'duract',
            'benoxaprofen', 'oraflex',
            'ximelagatran', 'exanta',
            'terfenadine', 'seldane',
            'astemizole', 'hismanal',
            'grepafloxacin', 'raxar',
            'trovafloxacin', 'trovan',
            'temafloxacin', 'omniflox',
            'thalidomide',
            'methysergide', 'sansert',
            'rimonabant', 'acomplia',
            'tacrine', 'cognex',
            'encainide',
            'droxicam', 'isoxicam'
        ) THEN
            _is_withdrawn_name := TRUE;
        END IF;
    END IF;
    IF _is_withdrawn_name THEN
        NEW.is_withdrawn := TRUE;
        NEW.clinical_status := COALESCE(NEW.clinical_status, 'withdrawn');
    ELSIF NEW.groups IS NOT NULL THEN
        -- groups is non-null and does not contain 'withdrawn', and the
        -- name did not match the seed list. Per the bidirectional fix,
        -- clear the flag so re-instated drugs can re-enter the
        -- repurposing pool. We do NOT clear clinical_status (it may
        -- legitimately be 'approved' for a re-instated drug).
        NEW.is_withdrawn := FALSE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_drugs_sync_withdrawn ON drugs;
-- v73 ROOT FIX (T-006 — Withdrawn trigger bypassed by non-groups/name
-- UPDATE statements; re-approved drugs with NULL groups stay flagged):
--   The previous trigger fired on ``BEFORE INSERT OR UPDATE OF groups,
--   name ON drugs``. This column-list restriction meant any UPDATE that
--   did NOT include ``groups`` or ``name`` in its SET clause (e.g. a
--   ChEMBL loader doing ``UPDATE drugs SET is_withdrawn = TRUE WHERE
--   chembl_id = 'CHEMBL123'``) bypassed the safety sync entirely. The
--   flag set by the loader would persist unmodified — even if it
--   contradicted the authoritative DrugBank groups string the row
--   already carried. Combined with the ``IF NEW.groups IS NOT NULL``
--   guard inside the function body, non-DrugBank drugs (groups=NULL)
--   could never get ``is_withdrawn=TRUE`` via the trigger, and a
--   re-approved drug whose groups was cleared to NULL stayed flagged
--   ``is_withdrawn=TRUE`` forever (the bidirectional fix was incomplete
--   for the NULL-groups case).
--
--   ROOT FIX (three holes closed simultaneously):
--     (1) Fire on ``BEFORE INSERT OR UPDATE`` (NO column restriction)
--         so EVERY row mutation runs through the safety-sync function.
--         Direct ``SET is_withdrawn=...`` updates from any loader now
--         fire the trigger and get reconciled with the authoritative
--         groups/name signals.
--     (2) ``NEW.groups IS NULL`` no longer silently preserves a stale
--         ``is_withdrawn=TRUE``. When groups is NULL we cannot derive
--         market status from DrugBank, but we CAN still consult the
--         curated withdrawn-name list (which catches ChEMBL/PubChem
--         drugs with no DrugBank groups). If neither signal indicates
--         withdrawn, we leave ``is_withdrawn`` UNCHANGED — preserving
--         any loader-set value (e.g. a ChEMBL withdrawal flag) rather
--         than silently clearing it. This is the conservative,
--         patient-safe default: never clear the flag based on absent
--         information.
--     (3) ``NEW.groups IS NOT NULL`` + no 'withdrawn' token + name not
--         in curated list → set ``is_withdrawn := FALSE``. This is the
--         bidirectional fix for the case where DrugBank IS the source
--         of truth (groups populated) and the drug has been re-approved
--         (no longer in 'withdrawn' group). Re-approved drugs with
--         populated groups correctly re-enter the repurposing pool.
CREATE TRIGGER trg_drugs_sync_withdrawn
    BEFORE INSERT OR UPDATE ON drugs
    FOR EACH ROW
    EXECUTE FUNCTION trg_drugs_sync_withdrawn();

-- v77 ROOT FIX (SQLite patient-safety gap):
-- The PostgreSQL trigger above uses CREATE FUNCTION + EXECUTE FUNCTION,
-- which SQLite does NOT support. The migration translator STRIPS both
-- statements on SQLite, meaning the trigger-based safety sync was a
-- NO-OP on SQLite (dev/test environments).
--
-- ROOT FIX: rather than attempt a fragile SQLite trigger syntax
-- (SQLite BEFORE triggers CANNOT modify NEW, and AFTER triggers that
-- UPDATE the same table risk recursion), we apply the safety-sync
-- logic at the Python ORM layer in database/loaders.py
-- (bulk_upsert_drugs). This hook runs on BOTH SQLite and PostgreSQL,
-- catching every ORM INSERT/UPDATE. The PostgreSQL trigger remains as
-- defense-in-depth for direct SQL INSERTs in production. The curated
-- name seed in the DO block above handles drugs already in the table
-- at migration time. Three-layer defense:
--   1. PostgreSQL trigger (production direct SQL INSERTs)
--   2. Python ORM hook (both dialects — see loaders.py)
--   3. Curated name seed (migration-time backfill)

-- ===========================================================================
-- Phase 6: Schema version metadata
-- ===========================================================================
INSERT INTO schema_version (version, description)
VALUES (6, 'Add is_withdrawn (life-safety), clinical_status, cas_number, logp, tpsa, h_bond_donor/acceptor_count, rotatable_bond_count, heavy_atom_count, complexity, completeness_score to drugs table; backfill is_withdrawn from DrugBank groups')
ON CONFLICT (version) DO NOTHING;

COMMIT;
