-- ============================================================================
-- Drug Repurposing ETL Platform — Rollback 009 Tighten InChIKey CHECK
-- Migration: 009_tighten_inchikey_check_constraint_rollback.sql
-- Description: Reverses the constraint swap from
--              009_tighten_inchikey_check_constraint.sql by restoring
--              the original over-permissive CHECK from
--              001_initial_schema.sql (lines 225-236).
--
-- v28 ROOT FIX (audit TOP-17): rollback sidecar for 009.
--
-- WARNING: restoring the old constraint RE-INTRODUCES the silent
-- Python-vs-SQL divergence that 009 fixed. Operators should only run
-- this rollback if they have an explicit reason (e.g. a downstream
-- consumer that depends on TEST/OUTER/INNER/IK% identifiers reaching
-- the database). Rolling back WITHOUT fixing the Python side will
-- re-open audit finding P1-ER-3.
--
-- NOTES:
--   - 009 did NOT modify any data rows; only the constraint. This
--     rollback therefore also does not touch data.
--   - v76 ROOT FIX (T-037): the schema_version row inserted by 009
--     (version=9) IS now removed, consistent with the convention
--     adopted across ALL rollbacks (003/010/011 already deleted;
--     002/004/005/006/007/008/009 now also delete). The previous
--     comment said the row was "NOT removed" — that was the old
--     inconsistent convention. The _migration_history table retains
--     the full audit trail of apply/rollback events.
-- ============================================================================

BEGIN;

-- Reverse the constraint swap (009 dropped the old, added the new).
-- Drop the tightened constraint and restore the original verbatim from
-- 001_initial_schema.sql so a rollback returns the schema to byte-exact
-- parity with the pre-009 state.
ALTER TABLE drugs DROP CONSTRAINT IF EXISTS chk_drugs_inchikey_format;

-- v73 ROOT FIX (T-005 — rollback restored a constraint that NEVER existed
-- in 001):
--   The previous rollback restored a CHECK clause containing extra
--   prefix-match branches (TEST, OUTER, INNER, and a length-capped IK
--   prefix). The accompanying comment claimed this was the "verbatim
--   text from 001_initial_schema.sql (lines 225-236)". That claim was
--   FALSE: 001_initial_schema.sql ONLY contains the canonical
--   ``LENGTH(inchikey) = 27 OR inchikey LIKE 'SYNTH%'`` form (verified
--   by reading 001 lines 309-313). The extra prefix branches existed
--   ONLY in early design drafts that were stripped BEFORE 001 was ever
--   applied to a real database.
--
--   Restoring a MORE permissive constraint than 001 ever had is a one-way
--   ratchet in the wrong direction: the prefix branches that 009's
--   audit predicate warned about would become ACCEPTED after rollback
--   — the opposite of "undo the change". Operators who rolled back 009
--   ended up with a schema that 001 itself never produced, and the
--   Python-vs-SQL divergence that 009 was designed to fix would be
--   WORSE than before 009 ran.
--
--   ROOT FIX: restore the EXACT constraint from 001_initial_schema.sql
--   lines 309-313 (canonical 27-char length OR SYNTH prefix only). This
--   is the true pre-009 state — byte-exact parity with 001. No
--   additional clauses.
ALTER TABLE drugs
    ADD CONSTRAINT chk_drugs_inchikey_format
    CHECK (
        LENGTH(inchikey) = 27
        OR inchikey LIKE 'SYNTH%'
    );

COMMENT ON CONSTRAINT chk_drugs_inchikey_format ON drugs IS
    'InChIKey format: original CHECK from 001_initial_schema.sql lines '
    '309-313 (27-char canonical OR SYNTH% prefix only). v73 ROOT FIX '
    '(T-005) — restores the EXACT 001 text, NOT the over-permissive '
    'TEST/OUTER/INNER/IK% variant the previous rollback incorrectly '
    'restored. v28 audit TOP-17 rollback: re-introduces the LENGTH=27 '
    'backstop (no regex) that 009 tightened — only run if a downstream '
    'consumer cannot tolerate the strict POSIX regex.';

-- v42 FORENSIC ROOT FIX (P0-7): the previous ``RAISE NOTICE`` was a
-- STANDALONE statement OUTSIDE any DO $$ ... $$ block. RAISE is a PL/pgSQL
-- statement, not valid standalone SQL, so the rollback crashed with
-- ``syntax error at or near 'RAISE'`` and operators could not roll back
-- migration 009. ROOT FIX: wrap the RAISE NOTICE in a DO block so it is
-- parsed as PL/pgSQL.
DO $$
BEGIN
    RAISE NOTICE '  [OK] Rolled back chk_drugs_inchikey_format to 001 original (over-permissive)';
END $$;

-- v76 ROOT FIX (T-037 — schema_version rollback consistency):
-- Delete the schema_version row inserted by 009 (version=9). All
-- rollbacks now follow the same convention: delete the version row so
-- schema_version always reflects the CURRENT schema state.
DELETE FROM schema_version WHERE version = 9;

COMMIT;
