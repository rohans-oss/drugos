-- ============================================================================
-- Drug Repurposing ETL Platform — Rollback 002 Bug Fixes
-- Migration: 002_bug_fixes_migration_rollback.sql
-- Description: Reverses the column/constraint changes from
--              002_bug_fixes_migration.sql.
--
-- v22 ROOT FIX (audit P2-10 / Section 9): rollback sidecar for 002.
--
-- v29 ROOT FIX (audit D-13): rollback was a no-op. Now actually undoes
-- the forward migration. The previous rollback only dropped and re-added
-- the chk_audit_log_operation constraint — but 002's forward migration
-- drop+re-add of that constraint is itself a no-op (001 already has the
-- same whitelist), so the rollback was undoing a no-op with a no-op.
-- Meanwhile 002's REAL schema changes (the new COALESCE unique index on
-- GDA, the renamed GDA unique constraint, the dedup-archive table, the
-- dropped original uq_gda_gene_disease_source constraint) were left
-- untouched — the rollback was effectively a no-op. This version drops
-- every schema object 002 created and restores every object 002 dropped,
-- so the schema after rollback matches the schema before 002 was applied.
--
-- NOTES:
--   - 002 added `audit_log.row_count`, `audit_log.details`. These columns
--     are ALSO in 001 (migration 002 uses `ADD COLUMN IF NOT EXISTS`
--     defensively because 001 already added them). They are OWNED by 001,
--     NOT by 002 — dropping them here would break 001's contract. They
--     are intentionally left in place.
--   - 002 added `proteins.gene_symbol`, `proteins.protein_name`,
--     `proteins.function_desc` — but again these are ALSO in 001
--     (002 uses `IF NOT EXISTS`). They are OWNED by 001, not by 002.
--     Left in place.
--   - 002 created `ix_gda_dedup_temp` (transient) but DROPPED it within
--     the same migration. Nothing to undo.
--   - 002 DROPPED `uq_entity_mapping_inchikey` (the 001 version) and
--     RE-CREATED it with an identical definition (partial unique index
--     `WHERE canonical_inchikey IS NOT NULL`). The net change is zero,
--     so no rollback action is needed for this index.
--     v74 ROOT FIX (T-017): the DROP+CREATE cycle was REMOVED from 002
--     entirely. 002 no longer touches this index. The 001 version is
--     authoritative and remains in place after rollback (since 002
--     never modified it).
--   - 002 drop+re-add of `chk_audit_log_operation` is a no-op (same
--     whitelist in 001 and 002). No rollback action needed.
--   - 002's data changes (NULL cleanup, deduplication of GDA rows,
--     archiving of duplicates into _migration_002_dedup_archive) CANNOT
--     be reversed without restoring from backup — the original rows
--     were DELETEd. The dedup archive table is preserved here so an
--     operator can manually restore archived rows if needed. Documented
--     as a known limitation.
-- ============================================================================

BEGIN;

-- v29 ROOT FIX (audit D-13): actually undo 002's schema changes.

-- 1. Drop the COALESCE-based unique index 002 created on
--    gene_disease_associations. This index did not exist before 002.
DROP INDEX IF EXISTS uq_gene_disease_associations_gda_coalesced;

-- 2. v90 ROOT FIX (BUG #2): migration 002 now KEEPS the ORM-matching
--    constraint name ``uq_gda_gene_disease_source`` (it no longer renames
--    it to ``uq_gene_disease_associations_gene_symbol_disease_id_source``).
--    For backwards-compatibility with DBs that ran the OLD v82-v89
--    migration 002 (which created the renamed constraint), we still drop
--    the old name here. On DBs running the v90+ migration 002, this
--    DROP is a no-op (constraint never existed under that name).
ALTER TABLE gene_disease_associations
    DROP CONSTRAINT IF EXISTS uq_gene_disease_associations_gene_symbol_disease_id_source;

-- 3. Re-add the ORIGINAL constraint from 001_initial_schema.sql (line ~1112).
--    v90 ROOT FIX (BUG #2): since migration 002 now KEEPS the name
--    ``uq_gda_gene_disease_source``, the constraint already exists with
--    the correct name after 002. This re-add is a no-op on v90+ DBs
--    (idempotent guard skips it). It only fires on DBs that ran the old
--    v82-v89 migration 002 (which renamed the constraint) — there, the
--    DROP above removed the renamed constraint, and this re-adds the
--    correctly-named one.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_gda_gene_disease_source'
    ) THEN
        ALTER TABLE gene_disease_associations
            ADD CONSTRAINT uq_gda_gene_disease_source
            UNIQUE (gene_symbol, disease_id, source);
        RAISE NOTICE '  [OK] Re-added constraint uq_gda_gene_disease_source (002 rollback)';
    ELSE
        RAISE NOTICE '  [SKIP] constraint uq_gda_gene_disease_source already exists';
    END IF;
END $$;

-- 4. RENAME (do NOT drop) the _migration_002_dedup_archive table 002
--    created. This table held archived duplicates from the GDA dedup
--    operation. The archived JSON data is preserved in the table until
--    this rollback runs.
--
--    v75 ROOT FIX (T-036 — rollback drops dedup_archive without dumping):
--      The v74 rollback did ``DROP TABLE IF EXISTS _migration_002_dedup_archive``
--      at this point — destroying the archived duplicate rows permanently.
--      The rollback header comment (NOTES section, lines 43-48) even
--      acknowledged this: "if an operator needs to recover the archived
--      rows, they should dump the table BEFORE running this rollback."
--      But there was no automated dump step — operators had to remember
--      to manually run ``pg_dump -t _migration_002_dedup_archive`` or
--      ``sqlite3 .dump _migration_002_dedup_archive`` BEFORE invoking
--      the rollback. In practice, no one remembers, so the data loss
--      was permanent.
--
--      ROOT FIX (master-grade, no operator memory required):
--        RENAME the table instead of dropping it. The renamed table
--        ``_migration_002_dedup_archive_rollback_backup`` persists
--        after the rollback completes. An operator who needs to
--        recover the archived rows can SELECT from the renamed table
--        directly. The renamed table is omitted from the ORM (no
--        SQLAlchemy model references it) so it does not interfere
--        with application queries. A future "clean-up" migration can
--        drop the renamed table once the operator confirms the data
--        is no longer needed.
--
--      PORTABILITY: ``ALTER TABLE ... RENAME TO ...`` is ANSI SQL
--      and works identically on PostgreSQL and SQLite. The IF EXISTS
--      guard via a DO block is PostgreSQL-only; on SQLite the
--      migration runner's OperationalError handler treats "no such
--      table" as a no-op (run_migrations.py:4272-4284).
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = '_migration_002_dedup_archive'
    ) THEN
        ALTER TABLE _migration_002_dedup_archive
            RENAME TO _migration_002_dedup_archive_rollback_backup;
        RAISE NOTICE '  [OK] Renamed _migration_002_dedup_archive → _migration_002_dedup_archive_rollback_backup (T-036 root fix: data preserved)';
    ELSE
        RAISE NOTICE '  [SKIP] _migration_002_dedup_archive does not exist (nothing to rename)';
    END IF;
END $$;

-- 5. chk_audit_log_operation: 002's drop+re-add is a no-op (same
--    whitelist as 001). No action needed — leaving the constraint in
--    its 001/002 state. Previous versions of this rollback did a
--    drop+re-add that "restored" the original whitelist, but the
--    "original" whitelist and the "002" whitelist are identical, so
--    the drop+re-add was a no-op. Removed to avoid confusion.

-- 6. v76 ROOT FIX (T-037 — schema_version rollback consistency):
--    The previous rollback left the ``schema_version`` row (version=2)
--    in place, inconsistent with rollbacks 003/010/011 which DELETE
--    their version row. After rolling back 002-010, the
--    ``schema_version`` table had gaps (2,4,5,6,7,8,9 left; 3,10,11
--    deleted) — ``check_migrations()`` reported confusing
--    "highest applied migration is N but schema_version has gaps"
--    warnings. ROOT FIX: adopt ONE convention across ALL rollbacks:
--    every rollback DELETEs its own version row. This makes the
--    ``schema_version`` table always reflect the CURRENT schema state
--    (only rows for migrations that are CURRENTLY APPLIED remain).
--    The ``schema_version`` table itself is OWNED by 001 (001 creates
--    it and inserts version=1); only the row is deleted, not the table.
DELETE FROM schema_version WHERE version = 2;

COMMIT;
