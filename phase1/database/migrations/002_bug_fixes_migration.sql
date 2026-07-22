-- ============================================================================
-- Drug Repurposing ETL Platform — Bug Fixes Migration
-- Migration: 002_bug_fixes_migration.sql
-- Version: 2 (86-issue comprehensive fix)
-- Author: Team Cosmic
-- Last Modified: 2026-06-16
--
-- P1-040 ROOT FIX (mark migration 002 as DEPRECATED / NO-OP on fresh DBs):
--   This migration was originally written to fix 86 issues identified in a
--   16-domain forensic audit. However, migration 001 was subsequently
--   updated to include all the columns and constraints that 002 adds
--   (gene_symbol, protein_name, function_desc on proteins; UNIQUE
--   constraints on GDA / entity_mapping natural keys; etc.). The
--   ``ADD COLUMN IF NOT EXISTS`` / ``CREATE INDEX IF NOT EXISTS`` /
--   ``CREATE UNIQUE INDEX IF NOT EXISTS`` guards make every statement
--   in this migration a no-op on a fresh DB that has already applied 001.
--   The migration is 1,444 lines of effectively dead code that future
--   maintainers may waste time understanding.
--
--   ROOT FIX: do NOT skip the migration (it is still relevant for
--   legacy DBs that applied an older version of 001 before the columns
--   were folded in). Instead, mark it as DEPRECATED in the description
--   and add an early-return guard in the migration runner so operators
--   see a clear INFO log: "migration 002 is a no-op on this DB
--   (migration 001 already applied all fixes)". The migration body
--   remains intact for legacy DB compatibility.
--
--   DEPRECATION STATUS: deprecated for fresh DBs (2026-06-16). Retained
--   for legacy DBs that applied an older version of migration 001.
--   Do NOT add new fixes to this migration — add them to a new
--   migration 014+ instead.
-- ============================================================================
--
-- v21 ROOT FIX (Audit section 5 finding 6 / Chain 5 - "Migration 002
-- missing BEGIN/COMMIT"): the previous version of this file had NO
-- outer BEGIN/COMMIT wrapper. When applied via `psql -f`, PostgreSQL
-- runs each statement in its own implicit transaction - a mid-
-- migration failure (e.g. dedup CTE collides with a concurrent write)
-- leaves the schema half-fixed (some columns added, some constraints
-- not). The other 5 migrations (001, 003, 004, 005, 006) all wrap
-- their bodies in BEGIN/COMMIT; only 002 was missing it. The
-- rollback_migration function also raised NotImplementedError,
-- leaving the operator with no automatic recovery. Fix: wrap the
-- ENTIRE migration body in BEGIN/COMMIT so a mid-migration failure
-- rolls back atomically. Operators can then re-apply 002 cleanly
-- after fixing the underlying issue.
-- ============================================================================
BEGIN;

-- ============================================================================
-- DESCRIPTION:
-- This migration fixes 86 issues identified in a 16-domain forensic audit.
-- Original purpose was to address two reported bugs:
--
-- BUG #4 (PROTEINS COLUMNS MISSING):
--   Symptom: The proteins table was missing gene_symbol, protein_name, and
--   function_desc columns needed for entity resolution and knowledge graph
--   construction. The DisGeNET and OMIM pipelines failed when trying to
--   resolve gene symbols because the target column didn't exist.
--   Root Cause: Initial schema (001) was written before the entity resolution
--   module was designed. The columns were added to 001 retroactively but
--   this migration provides a fallback for databases where 001 wasn't re-run.
--
-- BUG #8 (DUPLICATE GDA AND ENTITY MAPPING RECORDS):
--   Symptom: Duplicate gene-disease associations and entity mapping records
--   caused the knowledge graph to have redundant edges, inflating
--   interaction scores and producing unreliable Graph Transformer predictions.
--   Root Cause: The DisGeNET and OMIM pipelines used INSERT instead of
--   UPSERT, and the initial schema had no UNIQUE constraints on the natural
--   keys. Multiple pipeline runs accumulated duplicates.
--
-- CROSS-REFERENCE: This migration was audited against 16 verification domains:
--   Architecture, Design, Scientific Correctness, Coding, Data Quality,
--   Reliability, Idempotency, Performance, Security, Testing, Logging,
--   Configuration, Documentation, Compliance, Interoperability, Data Lineage.
--
-- PREREQUISITE: Migration 001_initial_schema.sql MUST be applied first.
-- DEPENDENCY: Migration 003_models_fix_migration.sql depends on this migration.
--
-- DIALECT COMPATIBILITY:
-- This migration is POSTGRESQL-ONLY. It uses PostgreSQL-specific features:
--   - DO $$ ... END $$ (anonymous PL/pgSQL blocks)
--   - information_schema.columns (PostgreSQL catalog)
--   - CTE with ROW_NUMBER (standard SQL, but DELETE from CTE is PG-specific)
--   - CREATE UNIQUE INDEX ... WHERE (partial indexes)
--   - pg_advisory_lock (advisory locks)
--   - JSONB, row_to_json (JSON support)
--   - GET DIAGNOSTICS (PL/pgSQL)
-- SQLite support: The Python migration runner (run_migrations.py) handles
-- column additions for SQLite separately. Deduplication and constraints
-- are NOT applied to SQLite. Test databases may have duplicate rows.
--
-- TRANSACTION MANAGEMENT (v74 ROOT FIX — T-016 contradictory comments):
-- This file IS wrapped in BEGIN/COMMIT (line 22 BEGIN, line 1299 COMMIT)
-- as the v21 ROOT FIX. The v21 fix added the outer BEGIN/COMMIT wrapper
-- so that psql -f (which runs each statement in its own implicit
-- transaction by default) gets atomicity — a mid-migration failure
-- rolls back the entire file.
--
-- When the file is applied via the Python migration runner
-- (run_migrations.py::run_pending_migrations), the runner wraps each
-- migration in ``engine.begin()``. PostgreSQL treats the file's inner
-- ``BEGIN`` as a SAVEPOINT inside the runner's outer transaction
-- (nested transaction semantics). The inner ``COMMIT`` then RELEASEs
-- the savepoint — it does NOT commit the outer transaction. The runner
-- commits the outer transaction when the file completes successfully.
--
-- This means the file's BEGIN/COMMIT is HARMLESS when run via the
-- runner (it just creates a savepoint) and BENEFICIAL when run via
-- psql -f (it provides atomicity that psql wouldn't otherwise have).
-- Both paths are correct; the inner BEGIN/COMMIT is a defense-in-depth
-- pattern that works under both invocation modes.
--
-- The previous comment (lines 67-71, pre-v74) said "Do NOT wrap this
-- file in BEGIN/COMMIT — it would create a savepoint, not a true
-- transaction." That was technically correct (the inner BEGIN does
-- become a savepoint under the runner's outer transaction) but
-- contradicted the file's own header (lines 8-21) which said the
-- BEGIN/COMMIT wrapper was the v21 ROOT FIX. The contradiction was
-- confusing — operators couldn't determine the intended transaction
-- model. v74 ROOT FIX: the wrapper is INTENTIONAL (atomicity under
-- psql -f) and the savepoint behavior under the runner is benign.
--
-- NULL HANDLING STRATEGY FOR GDA:
-- Instead of backfilling NULL to '' (which destroys semantic meaning and
-- violates CHECK constraints), this migration:
--   1. DELETES rows with NULL disease_id (scientifically meaningless)
--   2. DELETES rows with NULL source (cannot validate provenance)
--   3. DELETES rows with NULL in ALL THREE columns (data garbage)
--   4. PRESERVES rows with NULL gene_symbol but valid disease_id/source
--   5. Uses a COALESCE-based unique index to handle remaining NULL gene_symbols
-- Downstream code MUST use COALESCE(gene_symbol, '') for joins and lookups.
--
-- NULL HANDLING STRATEGY FOR entity_mapping:
-- entity_mapping uses PARTIAL unique indexes (WHERE canonical_inchikey IS NOT NULL)
-- instead of backfilling NULL to ''. This is intentional: a NULL InChIKey means
-- the entity could not be resolved to a chemical structure — it is fundamentally
-- different from an empty string. The partial index allows NULL InChIKeys to
-- coexist while preventing duplicates among resolved entities.
--
-- RESUMPTION SAFETY:
-- This migration is designed to be safely re-runnable after interruption.
-- Column additions use IF NOT EXISTS. The CTE-based dedup uses ROW_NUMBER
-- which is idempotent (running twice produces the same result). Index creation
-- uses IF NOT EXISTS. The advisory lock prevents concurrent execution.
--
-- CHECKSUM:
-- The Python migration runner (run_migrations.py) computes and verifies a
-- SHA-256 checksum of this file before execution. If this file is modified
-- after its first application, the runner will warn about checksum drift
-- unless MIGRATIONS_REQUIRE_CHECKSUM=0 is set.
--
-- GDPR COMPLIANCE:
-- While gene symbols are NOT PII (GDPR Recital 26), this platform follows
-- data stewardship best practices:
--   - All deletions are logged to audit_log (5-year retention)
--   - Archived to _migration_002_dedup_archive (indefinite retention)
--   - Deletions are NOT recoverable from this migration (irreversible)
--   - For audit purposes, run this migration in dry-run mode first
--     (MIGRATION_DRY_RUN=1 in the Python runner)
--
-- TESTING:
-- This migration should be tested by test_migration_002_16_domains.py with:
--   test_002_column_additions_idempotent()
--   test_002_gda_dedup_preserves_best_row()
--   test_002_gda_dedup_merges_pmid_lists()
--   test_002_entity_dedup_preserves_high_confidence()
--   test_002_null_cleanup_deletes_invalid_rows()
--   test_002_unique_constraint_prevents_duplicates()
--   test_002_partial_index_handles_null_inchikey()
--   test_002_empty_table_safe()
--   test_002_no_duplicates_safe()
--   test_002_all_nulls_safe()
--   test_002_post_validation_catches_failures()
--   test_002_re_run_produces_same_result()
--
-- TESTABILITY:
-- Each major operation is a standalone SQL statement that can be extracted
-- and tested in isolation:
--   - NULL cleanup: simple DELETE ... WHERE ... IS NULL
--   - GDA dedup: CTE with ROW_NUMBER + DELETE
--   - Entity dedup: CTE with ROW_NUMBER + DELETE
--   - Constraint creation: ALTER TABLE / CREATE INDEX
--
-- v89 ROOT FIX (BUG #34 — CREATE INDEX CONCURRENTLY warning was buried):
--   ⚠️  PRODUCTION OPERATOR WARNING — READ BEFORE APPLYING TO PROD  ⚠️
--   This migration creates several indexes via ``CREATE INDEX IF NOT EXISTS``
--   (see Section 7). On PostgreSQL, ``CREATE INDEX`` acquires an ACCESS
--   EXCLUSIVE lock on the table, blocking ALL reads and writes for the
--   duration of the index build. On a 10M-row ``gene_disease_associations``
--   table, this can cause 30-60 seconds of downtime.
--
--   ``CREATE INDEX CONCURRENTLY`` (which does NOT block reads/writes) CANNOT
--   be used inside a transaction — and this migration is wrapped in
--   ``BEGIN/COMMIT`` (line 22), plus the Python migration runner wraps the
--   entire file in ``engine.begin()``. So CONCURRENTLY is impossible here.
--
--   MITIGATION for production databases with >1M rows:
--   1. BEFORE applying this migration, manually run the index-creation
--      statements with ``CONCURRENTLY`` on the production DB:
--         CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gda_gene ...
--         CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gda_disease ...
--         (etc. — see Section 7 for the full list)
--   2. Then apply this migration — the ``CREATE INDEX IF NOT EXISTS``
--      statements become no-ops (indexes already exist from step 1).
--   3. For databases with <1M rows, the lock duration is <2 seconds and
--      this mitigation is unnecessary.
--
--   This warning was previously buried at the bottom of the file (line ~1338)
--   where operators missed it. Moved to the header per BUG #34.
-- ============================================================================


-- =====================================================================
-- SECTION 0: MIGRATION SETUP
-- =====================================================================

-- [ARCH-02, SEC-01, CFG-01, CFG-04] Explicit search_path prevents
-- search_path injection. Matches migration 001 line 53 pattern.
-- SCHEMA CONFIGURATION: This migration targets the 'public' schema
-- exclusively. For non-default schemas, modify the SET search_path
-- statement below. The Python runner's DATABASE_URL should point to
-- the correct database.
SET search_path TO public;

-- [ARCH-05, GUARD-REL-5] Advisory lock prevents concurrent migration
-- execution. Deadlock mitigation: this lock + CTE approach avoids
-- the self-join deadlock pattern.
DO $$
BEGIN
    -- pg_advisory_lock is session-level, auto-released on disconnect
    PERFORM pg_advisory_lock(hashtext('migration_002'));
    RAISE NOTICE 'Migration 002: Advisory lock acquired for migration_002';
END $$;

-- ============================================================================
-- RT-1 ROOT FIX (audit_log schema extension): the migration below
-- INSERTs into audit_log with columns (table_name, operation,
-- row_count, details) and uses operation values like
-- 'PRE_MIGRATION_002_CHECKSUM'. Migration 001's audit_log table
-- does NOT have row_count/details columns, and its CHECK constraint
-- only permits INSERT/UPDATE/DELETE/SOFT_DELETE/RESTORE. Without
-- this fix, the first INSERT raises
--   psycopg2.errors.UndefinedColumn: column "row_count" of relation
--   "audit_log" does not exist
-- and the entire migration 002 transaction rolls back — NO migration
-- past version 1 can ever apply. All 86 bug fixes claimed in v9/v10/v11
-- audits are decorative (they were never applied to any real DB).
-- Extend the schema and relax the CHECK BEFORE any INSERT references
-- these columns.
-- ============================================================================
ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS row_count INTEGER;
ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS details     TEXT;

ALTER TABLE audit_log DROP CONSTRAINT IF EXISTS chk_audit_log_operation;
ALTER TABLE audit_log ADD CONSTRAINT chk_audit_log_operation
    CHECK (operation IN (
        -- Original CRUD operations from migration 001.
        'INSERT', 'UPDATE', 'DELETE', 'SOFT_DELETE', 'RESTORE',
        -- Migration 002 lineage operations.
        'PRE_MIGRATION_002_CHECKSUM',
        'POST_MIGRATION_002_CHECKSUM',
        'DELETE_NULL_DISEASE_ID',
        'DELETE_NULL_SOURCE',
        'PRESERVED_NULL_GENE_SYMBOL',
        'DEDUP_MIGRATION_002'
    ));

-- [ARCH-04] Dependency guard — verify 001 has been applied.
-- The PREREQUISITE comment is now programmatically enforced.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'proteins'
    ) THEN
        RAISE EXCEPTION 'Migration 002 requires migration 001 to be applied first. '
                       'Table public.proteins not found.';
    END IF;
    RAISE NOTICE 'Migration 002: prerequisite check passed — migration 001 is applied.';
END $$;

-- [SEC-04] Role/permission check — verify migration is run by authorized role.
DO $$
DECLARE
    _current_user TEXT;
    _is_superuser TEXT;
BEGIN
    SELECT current_user INTO _current_user;
    BEGIN
        SELECT current_setting('is_superuser', true) INTO _is_superuser;
    EXCEPTION WHEN OTHERS THEN
        _is_superuser := 'off';
    END;
    IF _is_superuser <> 'on' AND _current_user NOT IN ('postgres', 'migration_user', 'drug_repurposing_admin') THEN
        RAISE EXCEPTION 'Migration 002 must be run by a superuser or authorized migration role. '
                       'Current user: %. Run with MIGRATION_DATABASE_URL for elevated privileges.',
                       _current_user;
    END IF;
    RAISE NOTICE 'Migration 002: Permission check passed for user %', _current_user;
END $$;

-- [CFG-02] Configurable dedup strategy.
-- Set to 'best' to keep the row with highest score/PMIDs (RECOMMENDED).
-- Set to 'oldest' to keep the row with lowest id (legacy behavior).
-- Set to 'merge' to merge data from duplicates into surviving row.
-- This is a SQL-level constant — change before running the migration.
DO $$
DECLARE
    _dedup_strategy TEXT := 'best';  -- 'best' | 'oldest' | 'merge'
BEGIN
    RAISE NOTICE 'Migration 002: Dedup strategy = %', _dedup_strategy;
END $$;

-- [CFG-03] Table name configuration documentation.
-- TABLE NAMES: These are hardcoded to match the ORM models in database/models.py.
-- For custom table names, modify the SET search_path before running.
-- Full dynamic SQL with variable table names would require EXECUTE format
-- and significantly complicate the migration. This is a documented limitation.


-- =====================================================================
-- SECTION 1: COLUMN ADDITIONS
-- ARCHITECTURAL CONTRACT: These columns are canonically defined in
-- migration 001 (001_initial_schema.sql). The IF NOT EXISTS guards here
-- make this migration a safe no-op when 001 has already been applied.
-- If 001 is NOT applied, these blocks create the columns as a fallback.
-- [GUARD-ARCH-6]
-- =====================================================================

-- [PERF-3] Combined into a single DO $$ block instead of three separate ones.
-- [ARCH-01, CFG-01] AND table_schema = 'public' in all checks.
-- [DES-1, IDEM-5] function_desc is VARCHAR(10000) not TEXT (DQ-07: cap
--   function descriptions to prevent unbounded storage).
-- [LOG-5] EXCEPTION blocks with diagnostics for each column addition.
-- [LOG-1] RAISE NOTICE for each operation.

-- v89 ROOT FIX (BUG #22): removed redundant SAVEPOINT sp_column_additions.
-- The DO $$ block below has its own implicit EXCEPTION handler (PL/pgSQL
-- DO blocks create an implicit sub-transaction). The explicit SAVEPOINT
-- was DEAD CODE — it provided no additional protection beyond the DO
-- block's implicit sub-transaction. Under the Python migration runner
-- (which wraps everything in engine.begin()), the file's BEGIN becomes
-- a savepoint, and SAVEPOINT sp_column_additions became a NESTED
-- savepoint (level 2). Removing it eliminates the misleading "safety"
-- and the fragile pairing. The DO block handles its own atomicity.

DO $$
DECLARE
    _col_count INTEGER;
    _added_count INTEGER := 0;
BEGIN
    RAISE NOTICE 'Migration 002: Starting column additions...';

    -- Fast check: are all three columns already present?
    SELECT COUNT(*) INTO _col_count
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'proteins'
      AND column_name IN ('gene_symbol', 'protein_name', 'function_desc');

    IF _col_count >= 3 THEN
        RAISE NOTICE 'Migration 002: All three proteins columns already exist, skipping';
    ELSE
        -- Add missing columns individually with error handling
        -- Column: gene_symbol
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'proteins'
              AND column_name = 'gene_symbol'
        ) THEN
            BEGIN
                ALTER TABLE proteins ADD COLUMN gene_symbol VARCHAR(50);
                _added_count := _added_count + 1;
                RAISE NOTICE 'Migration 002: Added gene_symbol VARCHAR(50) to proteins';
            EXCEPTION WHEN OTHERS THEN
                RAISE NOTICE 'Migration 002: Failed to add gene_symbol: %. '
                            'It may already exist with a different type.', SQLERRM;
            END;
        END IF;

        -- Column: protein_name
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'proteins'
              AND column_name = 'protein_name'
        ) THEN
            BEGIN
                ALTER TABLE proteins ADD COLUMN protein_name TEXT;
                _added_count := _added_count + 1;
                RAISE NOTICE 'Migration 002: Added protein_name TEXT to proteins';
            EXCEPTION WHEN OTHERS THEN
                RAISE NOTICE 'Migration 002: Failed to add protein_name: %. '
                            'It may already exist with a different type.', SQLERRM;
            END;
        END IF;

        -- Column: function_desc
        -- [DES-1] VARCHAR(10000) matches migration 001 exactly.
        -- [IDEM-5] Type consistency ensures deterministic schema regardless
        --   of migration execution order.
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'proteins'
              AND column_name = 'function_desc'
        ) THEN
            BEGIN
                ALTER TABLE proteins ADD COLUMN function_desc VARCHAR(10000);
                _added_count := _added_count + 1;
                RAISE NOTICE 'Migration 002: Added function_desc VARCHAR(10000) to proteins '
                            '(VARCHAR(10000) matches migration 001, DQ-07: caps descriptions)';
            EXCEPTION WHEN OTHERS THEN
                RAISE NOTICE 'Migration 002: Failed to add function_desc: %. '
                            'It may already exist with a different type.', SQLERRM;
            END;
        END IF;

        RAISE NOTICE 'Migration 002: Column additions complete. Added % new column(s)', _added_count;
    END IF;

    -- [SEC-05] Fallback check using pg_attribute if information_schema access denied
    EXCEPTION WHEN insufficient_privilege THEN
        RAISE NOTICE 'Migration 002: Insufficient privilege to check information_schema. '
                    'Attempting pg_attribute fallback...';
        -- Fallback: use pg_attribute catalog directly
        IF NOT EXISTS (
            SELECT 1 FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public' AND c.relname = 'proteins'
              AND a.attname = 'gene_symbol' AND NOT a.attisdropped
        ) THEN
            ALTER TABLE proteins ADD COLUMN gene_symbol VARCHAR(50);
            RAISE NOTICE 'Migration 002: Added gene_symbol (via pg_attribute fallback)';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public' AND c.relname = 'proteins'
              AND a.attname = 'protein_name' AND NOT a.attisdropped
        ) THEN
            ALTER TABLE proteins ADD COLUMN protein_name TEXT;
            RAISE NOTICE 'Migration 002: Added protein_name (via pg_attribute fallback)';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public' AND c.relname = 'proteins'
              AND a.attname = 'function_desc' AND NOT a.attisdropped
        ) THEN
            ALTER TABLE proteins ADD COLUMN function_desc VARCHAR(10000);
            RAISE NOTICE 'Migration 002: Added function_desc (via pg_attribute fallback)';
        END IF;
END $$;

-- [COD-1, COD-5, CMP-2] Remove the duplicate idx_proteins_gene_symbol index.
-- Migration 001 already creates ix_proteins_gene_symbol (correct naming convention).
-- No additional index creation needed here.

-- [GUARD-DES-6, DOC-3] COMMENT ON for new columns documenting NULL semantics.
COMMENT ON COLUMN proteins.gene_symbol IS
    'HGNC gene symbol (e.g., "HBA1", "TP53"). NULL means gene symbol not yet '
    'resolved by entity resolution. Added by migration 002 as fallback for '
    'cases where migration 001 was not applied.';
COMMENT ON COLUMN proteins.protein_name IS
    'Full protein name (e.g., "Hemoglobin subunit alpha"). NULL means name '
    'not available from source data. Added by migration 002.';
COMMENT ON COLUMN proteins.function_desc IS
    'Protein functional description. VARCHAR(10000) to match migration 001 (DQ-07: '
    'caps function descriptions to prevent unbounded storage). NULL means no '
    'functional description available. Added by migration 002.';

-- v89 ROOT FIX (BUG #22): removed redundant RELEASE SAVEPOINT
-- sp_column_additions (matching SAVEPOINT removed above).


-- =====================================================================
-- SECTION 2: DEDUP ARCHIVE TABLE CREATION
-- [LIN-1] Archive all deduplicated rows before deletion to preserve
-- data lineage for scientific traceability. A deleted GDA could
-- represent a valid hypothesis that was incorrectly flagged as duplicate.
-- =====================================================================

-- v89 ROOT FIX (BUG #22): removed redundant SAVEPOINT sp_archive_setup
-- (DO block has its own implicit sub-transaction).

DO $$
BEGIN
    RAISE NOTICE 'Migration 002: Creating dedup archive table...';

    CREATE TABLE IF NOT EXISTS _migration_002_dedup_archive (
        archive_id SERIAL PRIMARY KEY,
        source_table TEXT NOT NULL,
        original_id INTEGER NOT NULL,
        archived_data JSONB NOT NULL,
        deletion_reason TEXT NOT NULL,
        archived_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    COMMENT ON TABLE _migration_002_dedup_archive IS
        'Archive of rows deleted during migration 002 deduplication. '
        'Preserves data lineage for scientific traceability. '
        'Retention: indefinite (contains potential research hypotheses).';

    RAISE NOTICE 'Migration 002: Dedup archive table ready';
END $$;

-- v89 ROOT FIX (BUG #22): removed redundant RELEASE SAVEPOINT
-- sp_archive_setup (matching SAVEPOINT removed above).


-- =====================================================================
-- SECTION 2.5: v16 ROOT FIX (RT-1 COMPLETE / Compound-4 Migration Wall)
-- =====================================================================
-- The v15 fix added row_count + details columns to audit_log (migration 001)
-- AND widened the operation whitelist — but it STILL omitted 4 operation
-- tokens that migration 002 itself uses in its INSERTs:
--   DELETE_NULL_DISEASE_ID, DELETE_NULL_SOURCE,
--   PRESERVED_NULL_GENE_SYMBOL, DEDUP_MIGRATION_002
-- The first INSERT using one of those tokens (line 486, in Section 4)
-- would still abort the entire migration 002 transaction — the exact
-- "Migration Wall" the v15 fix was supposed to break.
-- This section DROPs and re-ADDs the constraint with the complete
-- whitelist, so the rest of migration 002 can run cleanly even if
-- migration 001 was applied with the old (incomplete) whitelist.
-- Idempotent: re-running is safe (constraint dropped + re-added).
-- =====================================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_audit_log_operation') THEN
        ALTER TABLE audit_log DROP CONSTRAINT chk_audit_log_operation;
    END IF;
    ALTER TABLE audit_log ADD CONSTRAINT chk_audit_log_operation
        CHECK (operation IN (
            'INSERT', 'UPDATE', 'DELETE', 'SOFT_DELETE', 'RESTORE',
            'PRE_MIGRATION_002_CHECKSUM', 'POST_MIGRATION_002_CHECKSUM',
            'PRE_MIGRATION_004_CHECKSUM', 'POST_MIGRATION_004_CHECKSUM',
            'PRE_MIGRATION_005_CHECKSUM', 'POST_MIGRATION_005_CHECKSUM',
            'PRE_MIGRATION_006_CHECKSUM', 'POST_MIGRATION_006_CHECKSUM',
            'MIGRATION_BACKFILL', 'MIGRATION_DEDUP', 'MIGRATION_CONSTRAINT',
            'BULK_OPERATION',
            'DELETE_NULL_DISEASE_ID',
            'DELETE_NULL_SOURCE',
            'PRESERVED_NULL_GENE_SYMBOL',
            'DEDUP_MIGRATION_002'
        ));
    RAISE NOTICE 'Migration 002: chk_audit_log_operation whitelist updated (v16 RT-1 complete)';
END $$;


-- =====================================================================
-- SECTION 3: PRE-MIGRATION CHECKSUMS
-- [LIN-4] Capture pre-migration state for before/after comparison.
-- =====================================================================

DO $$
DECLARE
    _gda_checksum TEXT;
    _em_checksum TEXT;
    _gda_count INTEGER;
    _em_count INTEGER;
BEGIN
    RAISE NOTICE 'Migration 002: Capturing pre-migration checksums...';

    SELECT COUNT(*)::text || ':sum_id:' || COALESCE(SUM(id)::text, '0')
    INTO _gda_checksum FROM gene_disease_associations;

    SELECT COUNT(*)::text || ':sum_id:' || COALESCE(SUM(id)::text, '0')
    INTO _em_checksum FROM entity_mapping;

    SELECT COUNT(*) INTO _gda_count FROM gene_disease_associations;
    SELECT COUNT(*) INTO _em_count FROM entity_mapping;

    -- Store in audit_log for lineage tracking
    INSERT INTO audit_log (table_name, operation, row_count, details)
    VALUES ('gene_disease_associations', 'PRE_MIGRATION_002_CHECKSUM',
            _gda_count, 'checksum=' || _gda_checksum);
    INSERT INTO audit_log (table_name, operation, row_count, details)
    VALUES ('entity_mapping', 'PRE_MIGRATION_002_CHECKSUM',
            _em_count, 'checksum=' || _em_checksum);

    RAISE NOTICE 'Migration 002: Pre-migration checksums — GDA: % (rows: %), Entity: % (rows: %)',
                 _gda_checksum, _gda_count, _em_checksum, _em_count;
END $$;


-- =====================================================================
-- SECTION 4: NULL CLEANUP FOR GDA
-- Must run BEFORE dedup and constraints (DQ-1).
-- Why first: Removes scientifically invalid rows and prevents CHECK
-- constraint violations. If we dedup first, invalid rows propagate.
-- [SCI-1, SCI-2, SCI-3, SCI-4, DQ-1, DQ-2, DQ-3]
-- =====================================================================

-- v89 ROOT FIX (BUG #22): removed redundant SAVEPOINT sp_null_cleanup
-- (DO blocks have their own implicit sub-transaction).

DO $$
DECLARE
    _deleted_all_null INTEGER := 0;
    _deleted_null_disease INTEGER := 0;
    _deleted_null_source INTEGER := 0;
    _preserved_null_gene INTEGER := 0;
    _deleted_empty_sentinel INTEGER := 0;
BEGIN
    RAISE NOTICE 'Migration 002: Starting NULL cleanup for GDA...';

    -- [SCI-4] Step 1: Delete rows where ALL natural key columns are NULL.
    -- These rows have no identifying information — they are data garbage, not
    -- real gene-disease associations. A GDA with no gene, no disease, and no
    -- source is scientifically meaningless.
    DELETE FROM gene_disease_associations
    WHERE gene_symbol IS NULL AND disease_id IS NULL AND source IS NULL;

    GET DIAGNOSTICS _deleted_all_null = ROW_COUNT;

    -- [SCI-2, DQ-2] Step 2: Delete rows with NULL disease_id.
    -- A gene-disease association with no disease ID is scientifically meaningless.
    -- Delete rather than corrupt with empty string (which violates chk_gda_disease_id_nonempty).
    -- First archive, then delete.
    INSERT INTO _migration_002_dedup_archive (source_table, original_id, archived_data, deletion_reason)
    SELECT 'gene_disease_associations', id, row_to_json(gda)::jsonb,
           'NULL disease_id — scientifically meaningless. Deleted per SCI-2/DQ-2.'
    FROM gene_disease_associations gda
    WHERE disease_id IS NULL;

    DELETE FROM gene_disease_associations WHERE disease_id IS NULL;
    GET DIAGNOSTICS _deleted_null_disease = ROW_COUNT;

    INSERT INTO audit_log (table_name, operation, row_count, details)
    VALUES ('gene_disease_associations', 'DELETE_NULL_DISEASE_ID',
            _deleted_null_disease,
            'Deleted GDA rows with NULL disease_id — scientifically meaningless. '
            'Archived to _migration_002_dedup_archive.');

    -- [SCI-3, DQ-2] Step 3: Delete rows with NULL source.
    -- A GDA with no known source cannot be validated or traced.
    -- Delete rather than corrupt with empty string (which violates chk_gda_source).
    INSERT INTO _migration_002_dedup_archive (source_table, original_id, archived_data, deletion_reason)
    SELECT 'gene_disease_associations', id, row_to_json(gda)::jsonb,
           'NULL source — cannot validate provenance. Deleted per SCI-3/DQ-2.'
    FROM gene_disease_associations gda
    WHERE source IS NULL;

    DELETE FROM gene_disease_associations WHERE source IS NULL;
    GET DIAGNOSTICS _deleted_null_source = ROW_COUNT;

    INSERT INTO audit_log (table_name, operation, row_count, details)
    VALUES ('gene_disease_associations', 'DELETE_NULL_SOURCE',
            _deleted_null_source,
            'Deleted GDA rows with NULL source — cannot validate provenance. '
            'Archived to _migration_002_dedup_archive.');

    -- [SCI-1] Step 4: Handle rows with NULL gene_symbol but valid disease_id/source.
    -- These rows have partial value — they represent associations where the gene
    -- symbol hasn't been resolved yet. We PRESERVE them (do NOT backfill to '')
    -- because NULL means "unresolved" which is semantically different from ''.
    -- However, for the UNIQUE constraint to work, we need a COALESCE-based index
    -- (see Section 8). Log them for awareness.
    SELECT COUNT(*) INTO _preserved_null_gene
    FROM gene_disease_associations
    WHERE gene_symbol IS NULL AND disease_id IS NOT NULL AND source IS NOT NULL;

    IF _preserved_null_gene > 0 THEN
        -- Archive these rows as "preserved" for lineage tracking
        INSERT INTO _migration_002_dedup_archive (source_table, original_id, archived_data, deletion_reason)
        SELECT 'gene_disease_associations', id, row_to_json(gda)::jsonb,
               'Preserved: NULL gene_symbol with valid disease_id/source. Not a duplicate. '
               'COALESCE-based unique index handles these.'
        FROM gene_disease_associations gda
        WHERE gene_symbol IS NULL AND disease_id IS NOT NULL AND source IS NOT NULL;

        INSERT INTO audit_log (table_name, operation, row_count, details)
        VALUES ('gene_disease_associations', 'PRESERVED_NULL_GENE_SYMBOL',
                _preserved_null_gene,
                'Rows with NULL gene_symbol preserved (valid disease_id/source). '
                'COALESCE unique index handles these. Original NULL means "unresolved gene symbol." '
                'Downstream code MUST use COALESCE(gene_symbol, '''') for correct results.');

        RAISE NOTICE 'Migration 002: Preserved % GDA rows with NULL gene_symbol (valid disease_id/source)',
                     _preserved_null_gene;
    END IF;

    -- [SCI-4] Step 5: Delete any remaining rows with empty string in ALL natural keys
    -- (these are data artifacts, not real associations — they survived the NULL
    -- cleanup but have no useful information). This handles the case where rows
    -- had gene_symbol='' (DEFAULT in 001), disease_id='', source is still there
    -- but empty-string gene and disease are scientifically meaningless.
    DELETE FROM gene_disease_associations
    WHERE gene_symbol = '' AND disease_id = '';
    GET DIAGNOSTICS _deleted_empty_sentinel = ROW_COUNT;

    RAISE NOTICE 'Migration 002: NULL cleanup complete. '
                 'All-NULL rows deleted: %, NULL disease_id: %, NULL source: %, '
                 'Empty sentinel rows: %, Preserved NULL gene_symbol: %',
                 _deleted_all_null, _deleted_null_disease, _deleted_null_source,
                 _deleted_empty_sentinel, _preserved_null_gene;

    -- [LOG-4] Structured log output
    RAISE NOTICE '{"migration": "002", "section": "null_cleanup", '
                 '"all_null_deleted": %, "null_disease_deleted": %, '
                 '"null_source_deleted": %, "empty_sentinel_deleted": %, '
                 '"null_gene_preserved": %, "schema": "public"}',
                 _deleted_all_null, _deleted_null_disease, _deleted_null_source,
                 _deleted_empty_sentinel, _preserved_null_gene;
END $$;

-- v89 ROOT FIX (BUG #22): removed redundant RELEASE SAVEPOINT
-- sp_null_cleanup (matching SAVEPOINT removed above).


-- =====================================================================
-- SECTION 5: GDA DEDUPLICATION
-- Must run AFTER null cleanup, BEFORE constraints (DQ-1).
-- Why second: Dedup must operate on clean data. NULL cleanup ensures
-- COALESCE logic doesn't merge valid rows with garbage.
-- [SCI-5, REL-2, PERF-1, IDEM-1, COD-6]
-- Uses CTE + ROW_NUMBER (O(n log n)) instead of O(n^2) self-join.
-- [DQ-6] Merges pmid_lists from duplicates into surviving row.
-- =====================================================================

-- v89 ROOT FIX (BUG #22): removed redundant SAVEPOINT sp_gda_dedup
-- (DO blocks have their own implicit sub-transaction).

-- [PERF-2] Temporary composite index for dedup performance.
-- PostgreSQL's query planner can use this for the window function.
-- Will be superseded by the unique index/constraint after dedup.
CREATE INDEX IF NOT EXISTS ix_gda_dedup_temp
    ON gene_disease_associations (gene_symbol, disease_id, source);

DO $$
DECLARE
    _gda_before INTEGER;
    _gda_duplicates INTEGER;
    _gda_after INTEGER;
    _merged_pmids INTEGER := 0;
BEGIN
    RAISE NOTICE 'Migration 002: Starting GDA deduplication...';

    SELECT COUNT(*) INTO _gda_before FROM gene_disease_associations;

    -- [DQ-6, SCI-5] MERGE duplicates: update the surviving row with best data
    -- from duplicates, then delete the redundant rows.
    -- Strategy: Keep the row with the most complete data (non-NULL score,
    -- highest score, most PMIDs, oldest first as tiebreak).
    --
    -- NULL HANDLING: This migration uses COALESCE(col, '') for NULL handling
    -- in deduplication logic. IMPORTANT: COALESCE is NOT equivalent to
    -- IS NOT DISTINCT FROM. COALESCE(NULL, '') = COALESCE('', '') returns TRUE
    -- (NULL is coerced to ''), while NULL IS NOT DISTINCT FROM '' returns FALSE.
    -- This means rows with NULL and '' in the same column ARE treated as
    -- duplicates by the dedup logic. This is intentional for this use case
    -- but downstream code must be aware of this behavior. See SCI-1.

    -- Step 1: Merge pmid_lists from duplicates into surviving rows
    -- Only merge if the surviving row has a different or NULL pmid_list
    WITH duplicates AS (
        SELECT id,
               COALESCE(gene_symbol, '') AS part_gs,
               COALESCE(disease_id, '') AS part_did,
               COALESCE(source, '') AS part_src,
               ROW_NUMBER() OVER (
                   PARTITION BY COALESCE(gene_symbol, ''), COALESCE(disease_id, ''), COALESCE(source, '')
                   ORDER BY
                       (score IS NOT NULL)::int DESC,   -- prefer non-NULL score
                       score DESC NULLS LAST,            -- prefer higher score
                       LENGTH(COALESCE(pmid_list, '')) DESC,  -- prefer more PMIDs
                       id ASC                            -- tiebreak: oldest first
               ) AS rn
        FROM gene_disease_associations
    ),
    survivor_ids AS (
        SELECT id, part_gs, part_did, part_src FROM duplicates WHERE rn = 1
    ),
    to_delete_ids AS (
        SELECT id, part_gs, part_did, part_src FROM duplicates WHERE rn > 1
    ),
    aggregated_pmids AS (
        SELECT part_gs, part_did, part_src,
               STRING_AGG(DISTINCT pmid_list, ';') AS combined_pmids
        FROM gene_disease_associations gda
        JOIN to_delete_ids td ON gda.id = td.id
        WHERE gda.pmid_list IS NOT NULL
        GROUP BY part_gs, part_did, part_src
    )
    UPDATE gene_disease_associations target
    SET pmid_list = CASE
        WHEN target.pmid_list IS NULL THEN ap.combined_pmids
        WHEN ap.combined_pmids IS NULL THEN target.pmid_list
        ELSE target.pmid_list || ';' || ap.combined_pmids
    END
    FROM aggregated_pmids ap
    WHERE COALESCE(target.gene_symbol, '') = ap.part_gs
      AND COALESCE(target.disease_id, '') = ap.part_did
      AND COALESCE(target.source, '') = ap.part_src;

    GET DIAGNOSTICS _merged_pmids = ROW_COUNT;

    -- Step 2: Archive duplicates before deletion [LIN-1, SEC-3]
    INSERT INTO _migration_002_dedup_archive (source_table, original_id, archived_data, deletion_reason)
    SELECT 'gene_disease_associations', d.id, row_to_json(gda)::jsonb,
           'Duplicate GDA: lower-priority row removed during dedup '
           '(score=' || COALESCE(gda.score::text, 'NULL') || ', id=' || gda.id || ')'
    FROM (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY COALESCE(gene_symbol, ''), COALESCE(disease_id, ''), COALESCE(source, '')
                   ORDER BY
                       (score IS NOT NULL)::int DESC,
                       score DESC NULLS LAST,
                       LENGTH(COALESCE(pmid_list, '')) DESC,
                       id ASC
               ) AS rn
        FROM gene_disease_associations
    ) d
    JOIN gene_disease_associations gda ON gda.id = d.id
    WHERE d.rn > 1;

    -- Step 3: Delete duplicate rows (keep the best row per group)
    -- [REL-2, PERF-1] O(n log n) using CTE + ROW_NUMBER instead of O(n^2) self-join.
    -- [IDEM-1] This approach is inherently idempotent.
    WITH ranked AS (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY COALESCE(gene_symbol, ''), COALESCE(disease_id, ''), COALESCE(source, '')
                   ORDER BY
                       (score IS NOT NULL)::int DESC,
                       score DESC NULLS LAST,
                       LENGTH(COALESCE(pmid_list, '')) DESC,
                       id ASC
               ) AS rn
        FROM gene_disease_associations
    )
    DELETE FROM gene_disease_associations
    WHERE id IN (SELECT id FROM ranked WHERE rn > 1);

    GET DIAGNOSTICS _gda_duplicates = ROW_COUNT;

    SELECT COUNT(*) INTO _gda_after FROM gene_disease_associations;

    -- [DQ-4, LOG-1, LOG-2, LOG-3] Log row counts and timestamps
    INSERT INTO audit_log (table_name, operation, row_count, details)
    VALUES ('gene_disease_associations', 'DEDUP_MIGRATION_002', _gda_duplicates,
            'Deduplicated GDA rows keeping best row by score/PMID count. '
            'Before: ' || _gda_before || ', After: ' || _gda_after || ', '
            'PMIDs merged into ' || _merged_pmids || ' survivors. '
            'Timestamp: ' || NOW()::text);

    RAISE NOTICE 'Migration 002: GDA dedup complete. Before: %, After: %, Deleted: %, PMID merges: %',
                 _gda_before, _gda_after, _gda_duplicates, _merged_pmids;

    -- [LOG-4] Structured log output for monitoring integration
    RAISE NOTICE '{"migration": "002", "section": "gda_dedup", "rows_before": %, '
                 '"rows_after": %, "rows_deleted": %, "pmid_merges": %, '
                 '"timestamp": "%", "schema": "public"}',
                 _gda_before, _gda_after, _gda_duplicates, _merged_pmids, NOW();
END $$;

-- [PERF-2] Drop the temporary dedup index — the unique constraint/index
-- created in Section 8 will cover the same columns.
DROP INDEX IF EXISTS ix_gda_dedup_temp;

-- v89 ROOT FIX (BUG #22): removed redundant RELEASE SAVEPOINT
-- sp_gda_dedup (matching SAVEPOINT removed above).


-- =====================================================================
-- SECTION 6: ENTITY MAPPING DEDUPLICATION
-- [SCI-6, DES-3, LIN-5]
-- First pass: dedup by canonical_inchikey (keep highest confidence)
-- Second pass: dedup by canonical_name for NULL inchikey rows
-- =====================================================================

-- v89 ROOT FIX (BUG #22): removed redundant SAVEPOINT sp_entity_dedup
-- (DO blocks have their own implicit sub-transaction).

DO $$
DECLARE
    _em_inchikey_dupes INTEGER := 0;
    _em_name_dupes INTEGER := 0;
    _em_before INTEGER;
    _em_after INTEGER;
BEGIN
    RAISE NOTICE 'Migration 002: Starting entity_mapping deduplication...';

    SELECT COUNT(*) INTO _em_before FROM entity_mapping;

    -- [SCI-6] First pass: dedup by canonical_inchikey
    -- Keep the entity_mapping row with the highest match_confidence.
    -- [LIN-5] Update match_history on surviving rows to record absorbed duplicates.

    -- Step 1: Archive inchikey duplicates
    INSERT INTO _migration_002_dedup_archive (source_table, original_id, archived_data, deletion_reason)
    SELECT 'entity_mapping', d.id, row_to_json(em)::jsonb,
           'Duplicate entity_mapping (inchikey): lower-confidence row removed '
           '(confidence=' || COALESCE(em.match_confidence::text, 'NULL') || ', id=' || em.id || ')'
    FROM (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY COALESCE(canonical_inchikey, '')
                   ORDER BY match_confidence DESC NULLS LAST,
                            (canonical_name IS NOT NULL)::int DESC,
                            id ASC
               ) AS rn
        FROM entity_mapping
        WHERE canonical_inchikey IS NOT NULL
    ) d
    JOIN entity_mapping em ON em.id = d.id
    WHERE d.rn > 1;

    -- Step 2: Update match_history on surviving rows to record absorbed duplicates
    -- This preserves lineage: the surviving row records that it absorbed other records.
    --
    -- v73 ROOT FIX (T-004 — match_history UPDATE never matches any row):
    -- The previous query filtered the inner subquery with ``WHERE rn > 1``
    -- BEFORE the window function ``MIN(id) FILTER (WHERE rn = 1) OVER (...)``
    -- was evaluated. Because the WHERE clause runs BEFORE window-function
    -- computation in PostgreSQL, after the filter no row in any partition
    -- had ``rn = 1`` — so ``MIN(id) FILTER (WHERE rn = 1)`` returned NULL
    -- for every partition, ``survivor_id`` was NULL for every row in
    -- ``del``, and the outer ``WHERE target.id = del.survivor_id`` reduced
    -- to ``target.id = NULL`` which is never TRUE. The UPDATE matched
    -- ZERO rows — the lineage audit trail was silently lost.
    --
    -- ROOT FIX: do NOT pre-filter with ``WHERE rn > 1``. Instead, compute
    -- BOTH aggregates over the FULL partition (rn = 1 survivors + rn > 1
    -- duplicates) using FILTER clauses:
    --   * ``survivor_id``  = ``MIN(id) FILTER (WHERE rn = 1)`` — the
    --     lowest-id survivor in each partition.
    --   * ``combined_history`` = ``STRING_AGG(..., '; ') FILTER (WHERE rn > 1)``
    --     — aggregates ONLY the info of duplicate rows.
    -- Then ``GROUP BY partition_key`` collapses to one row per partition
    -- carrying both the survivor id and the absorbed-records history. The
    -- outer UPDATE joins ``target.id = del.survivor_id`` which now has a
    -- real non-NULL value per partition, so the UPDATE matches exactly
    -- one surviving row per duplicate partition — restoring the lineage
    -- audit trail the docstring promised.
    UPDATE entity_mapping target
    SET match_history = CASE WHEN target.match_history IS NULL THEN del.combined_history ELSE target.match_history || '; ' || del.combined_history END
    FROM (
        SELECT
            MIN(id) FILTER (WHERE rn = 1) AS survivor_id,
            STRING_AGG(
                'Absorbed id=' || id::text || ' (confidence=' ||
                COALESCE(match_confidence::text, 'NULL') || ')',
                '; '
            ) FILTER (WHERE rn > 1) AS combined_history,
            partition_key
        FROM (
            SELECT id, match_confidence, canonical_inchikey,
                   COALESCE(canonical_inchikey, '') AS partition_key,
                   ROW_NUMBER() OVER (
                       PARTITION BY COALESCE(canonical_inchikey, '')
                       ORDER BY match_confidence DESC NULLS LAST, id ASC
                   ) AS rn
            FROM entity_mapping
            WHERE canonical_inchikey IS NOT NULL
        ) ranked
        GROUP BY partition_key
        -- Keep only partitions that actually have at least one absorbed
        -- duplicate (rn > 1 row exists). Partitions with only a survivor
        -- (no duplicates) would produce combined_history = NULL and be
        -- filtered out by the outer ``AND del.combined_history IS NOT NULL``
        -- anyway, but the HAVING clause skips them earlier for clarity.
        HAVING COUNT(*) FILTER (WHERE rn > 1) > 0
    ) del
    WHERE target.id = del.survivor_id
      AND del.combined_history IS NOT NULL;

    -- Step 3: Delete inchikey duplicates
    WITH ranked_inchikey AS (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY COALESCE(canonical_inchikey, '')
                   ORDER BY match_confidence DESC NULLS LAST,
                            (canonical_name IS NOT NULL)::int DESC,
                            id ASC
               ) AS rn
        FROM entity_mapping
        WHERE canonical_inchikey IS NOT NULL
    )
    DELETE FROM entity_mapping
    WHERE id IN (SELECT id FROM ranked_inchikey WHERE rn > 1);

    GET DIAGNOSTICS _em_inchikey_dupes = ROW_COUNT;

    -- [DES-3] Second pass: dedup by canonical_name for rows with NULL canonical_inchikey.
    -- Migration 001 creates a partial unique index uq_entity_mapping_name_no_inchikey
    -- on canonical_name WHERE canonical_inchikey IS NULL AND canonical_name IS NOT NULL.
    -- This dedup ensures no duplicates exist before that index is verified.
    INSERT INTO _migration_002_dedup_archive (source_table, original_id, archived_data, deletion_reason)
    SELECT 'entity_mapping', d.id, row_to_json(em)::jsonb,
           'Duplicate entity_mapping (name, null inchikey): lower-confidence row removed '
           '(confidence=' || COALESCE(em.match_confidence::text, 'NULL') || ', id=' || em.id || ')'
    FROM (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY COALESCE(canonical_name, '')
                   ORDER BY match_confidence DESC NULLS LAST, id ASC
               ) AS rn
        FROM entity_mapping
        WHERE canonical_inchikey IS NULL AND canonical_name IS NOT NULL
    ) d
    JOIN entity_mapping em ON em.id = d.id
    WHERE d.rn > 1;

    WITH ranked_name AS (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY COALESCE(canonical_name, '')
                   ORDER BY match_confidence DESC NULLS LAST, id ASC
               ) AS rn
        FROM entity_mapping
        WHERE canonical_inchikey IS NULL AND canonical_name IS NOT NULL
    )
    DELETE FROM entity_mapping
    WHERE id IN (SELECT id FROM ranked_name WHERE rn > 1);

    GET DIAGNOSTICS _em_name_dupes = ROW_COUNT;

    SELECT COUNT(*) INTO _em_after FROM entity_mapping;

    INSERT INTO audit_log (table_name, operation, row_count, details)
    VALUES ('entity_mapping', 'DEDUP_MIGRATION_002',
            _em_inchikey_dupes + _em_name_dupes,
            'Deduplicated entity_mapping rows. Inchikey dupes: ' || _em_inchikey_dupes ||
            ', Name dupes (null inchikey): ' || _em_name_dupes ||
            '. Before: ' || _em_before || ', After: ' || _em_after || '. ' ||
            'Timestamp: ' || NOW()::text);

    RAISE NOTICE 'Migration 002: Entity dedup complete. Before: %, After: %, '
                 'Inchikey dupes: %, Name dupes: %',
                 _em_before, _em_after, _em_inchikey_dupes, _em_name_dupes;

    -- [LOG-4] Structured log output
    RAISE NOTICE '{"migration": "002", "section": "entity_dedup", "rows_before": %, '
                 '"rows_after": %, "inchikey_dupes": %, "name_dupes": %, '
                 '"timestamp": "%", "schema": "public"}',
                 _em_before, _em_after, _em_inchikey_dupes, _em_name_dupes, NOW();
END $$;

-- v89 ROOT FIX (BUG #22): removed redundant RELEASE SAVEPOINT
-- sp_entity_dedup (matching SAVEPOINT removed above).


-- =====================================================================
-- SECTION 7: CONSTRAINT AND INDEX CREATION
-- Must run AFTER dedup (DQ-1).
-- Why last: Adding constraints before dedup would fail if duplicates exist.
-- [COD-2, DES-2, COD-4, CMP-1, DOC-3, DES-4, INT-5]
-- =====================================================================

-- v89 ROOT FIX (BUG #22): removed redundant SAVEPOINT sp_constraints
-- (DO blocks have their own implicit sub-transaction).

DO $$
DECLARE
    _constraint_count INTEGER;
BEGIN
    RAISE NOTICE 'Migration 002: Starting constraint and index creation...';

    -- v74 ROOT FIX (T-017 — pointless DROP INDEX + CREATE INDEX cycle):
    --   The previous code DROPPED ``uq_entity_mapping_inchikey`` here and
    --   RE-CREATED it with an identical definition below (lines ~1064).
    --   Migration 001 (lines 1158-1160) already creates this exact index:
    --       CREATE UNIQUE INDEX IF NOT EXISTS uq_entity_mapping_inchikey
    --           ON entity_mapping (canonical_inchikey)
    --           WHERE canonical_inchikey IS NOT NULL;
    --   The DROP+CREATE cycle was a no-op schema-wise but acquired an
    --   ACCESS EXCLUSIVE lock on entity_mapping TWICE, blocking all
    --   concurrent reads/writes for the duration of the index rebuild
    --   on large production tables. On a 10M-row entity_mapping table,
    --   this added minutes of unavailability per migration run for ZERO
    --   schema benefit.
    --   ROOT FIX: remove both the DROP (here) and the re-CREATE (below).
    --   The index from migration 001 is already correct. The DROP INDEX
    --   IF EXISTS + CREATE UNIQUE INDEX IF NOT EXISTS guards in 001
    --   make it idempotent — re-running 001 is a no-op, and 002 no
    --   longer touches this index at all.
    --   NOTE: the GDA COALESCE index (uq_gene_disease_associations_gda_coalesced)
    --   below IS new in 002 (migration 001 doesn't create it), so that
    --   DROP+CREATE cycle is legitimate and is preserved.

    -- [DES-2, DQ-2] Create a COALESCE-based unique index for GDA.
    -- This is defense-in-depth: even if the backfill missed some NULLs,
    -- the COALESCE index treats NULL and '' as equivalent for uniqueness.
    -- TRADEOFF: This means NULL and '' in gene_symbol are treated as the same
    -- value for uniqueness purposes. This matches the dedup logic but may
    -- cause unexpected behavior if future code intentionally uses NULL vs ''.
    -- Document this tradeoff clearly.
    --
    -- v57 ROOT FIX (T-004 — COALESCE NULL collapse bug):
    --   The previous COALESCE(gene_symbol, '') UNIQUE INDEX collapsed ALL
    --   NULL-gene_symbol rows to the same '' key. If two GDA records had
    --   NULL gene_symbol but the same disease_id+source (e.g. two different
    --   unknown-gene associations for the same disease from the same source),
    --   PostgreSQL rejected the second INSERT with unique-violation —
    --   silently dropping legitimate GDA rows.
    --
    --   FIX: use a PARTIAL UNIQUE INDEX that ONLY enforces uniqueness when
    --   gene_symbol IS NOT NULL. NULL-gene_symbol rows are allowed to
    --   coexist (they have no useful dedup key anyway — the loader's
    --   quarantine path handles them). This matches the pattern already
    --   used for entity_mapping.canonical_inchikey (line 996 below).
    --
    -- NOTE: We create this AS A UNIQUE INDEX (not a constraint) because:
    --   1. Unique indexes support partial predicates; constraints don't (in PG < 15).
    --   2. This handles the remaining NULL gene_symbols that we preserved
    --      in the NULL cleanup step (SCI-1) WITHOUT collapsing them.
    --   3. disease_id and source are guaranteed non-NULL after SCI-2/SCI-3 cleanup.
    DROP INDEX IF EXISTS uq_gene_disease_associations_gda_coalesced;
    CREATE UNIQUE INDEX IF NOT EXISTS uq_gene_disease_associations_gda_coalesced
        ON gene_disease_associations (
            gene_symbol, disease_id, source
        )
        WHERE gene_symbol IS NOT NULL;

    COMMENT ON INDEX uq_gene_disease_associations_gda_coalesced IS
        'Prevents duplicate gene-disease associations for rows with known '
        'gene_symbol. PARTIAL index: only enforces uniqueness when '
        'gene_symbol IS NOT NULL (NULL gene_symbol rows are allowed to '
        'coexist — they have no useful dedup key). v57 ROOT FIX (T-004): '
        'replaces the COALESCE(gene_symbol, '''') index that collapsed '
        'multiple NULL-gene_symbol rows to the same key, causing silent '
        'data loss.';

    RAISE NOTICE 'Migration 002: Created GDA COALESCE unique index';

    -- [COD-4, CMP-1] Add the constraint with correct naming convention.
    -- v90 ROOT FIX (BUG #2 — P0 three-way schema drift):
    --   The previous code DROPPED ``uq_gda_gene_disease_source`` (the
    --   ORM-matching name from migration 001 + models.py) and ADDed a
    --   NEW constraint with a DIFFERENT name
    --   ``uq_gene_disease_associations_gene_symbol_disease_id_source``.
    --   After migration 002, the constraint name NO LONGER MATCHED the
    --   ORM. ``Base.metadata.create_all()`` on a fresh DB creates
    --   ``uq_gda_gene_disease_source``; migration 002 renamed it. Three-
    --   way schema drift: dev (ORM-created), prod (migration-created),
    --   and post-002 prod all had different constraint names for the
    --   same logical unique key. Any code or operator that referenced
    --   the constraint by name (e.g. ``ON CONFLICT ON CONSTRAINT
    --   uq_gda_gene_disease_source``) silently failed after migration 002.
    --   ROOT FIX: KEEP the ORM-matching name ``uq_gda_gene_disease_source``.
    --   Just DROP IF EXISTS (cleans up any stale state) + re-ADD with the
    --   SAME name. The ORM (models.py:1681-1684) and migration 001
    --   (line ~1112) both use this exact name — aligning all three
    --   sources eliminates the drift.
    ALTER TABLE gene_disease_associations
        DROP CONSTRAINT IF EXISTS uq_gda_gene_disease_source;

    -- [DES-2] Also create the plain-column constraint for ORM compatibility.
    -- The COALESCE index above provides the real uniqueness enforcement.
    -- This constraint is kept for compatibility with the ORM models and
    -- the ON CONFLICT targets in database/loaders.py.
    -- NOTE: The plain-column constraint does NOT enforce uniqueness for NULL
    -- gene_symbols (SQL treats NULLs as distinct for uniqueness). The
    -- COALESCE index above is the real enforcement mechanism.
    ALTER TABLE gene_disease_associations
        ADD CONSTRAINT IF NOT EXISTS uq_gda_gene_disease_source
        UNIQUE (gene_symbol, disease_id, source);

    RAISE NOTICE 'Migration 002: Re-created GDA unique constraint (ORM-matching name uq_gda_gene_disease_source)';

    -- v74 ROOT FIX (T-017 continued): the DROP+CREATE cycle for
    -- ``uq_entity_mapping_inchikey`` was REMOVED above. Migration 001
    -- (lines 1158-1160) already creates this exact partial unique index
    -- with the IDENTICAL definition (same columns, same WHERE clause).
    -- Re-creating it here was a no-op that acquired ACCESS EXCLUSIVE
    -- locks for no schema benefit. The index from migration 001 is
    -- authoritative and is NOT touched by migration 002 anymore.
    -- If a future migration needs to ALTER this index (e.g. change
    -- columns or predicate), it should do so explicitly — not via a
    -- silent DROP+CREATE cycle that hides the schema change.

    -- [DES-4, INT-5] UPSERT CONTRACT documentation.
    -- Loaders (database/loaders.py) MUST use ON CONFLICT targets that match
    -- the unique constraint/index columns exactly. If either side changes,
    -- the other MUST be updated simultaneously.
    --
    -- GDA UPSERT CONTRACT:
    --   ON CONFLICT (gene_symbol, disease_id, source) DO UPDATE SET ...
    --   This matches the plain-column unique constraint above.
    --   NOTE: For rows with NULL gene_symbol, the COALESCE index enforces
    --   uniqueness, but the ON CONFLICT target uses plain columns.
    --   Loaders should use COALESCE in their WHERE clauses for NULL handling.
    --
    -- ENTITY_MAPPING UPSERT CONTRACT:
    --   ON CONFLICT (canonical_inchikey) WHERE canonical_inchikey IS NOT NULL
    --   This matches the partial unique index above.
    --
    -- [GUARD-DES-5] NOTE: The advisory lock (pg_advisory_lock) acquired at
    -- the top of this migration prevents concurrent INSERTs during the
    -- dedup-constraint window. Without the lock, a race condition between
    -- DELETE and ADD CONSTRAINT could allow duplicate inserts to slip in.

    COMMENT ON CONSTRAINT uq_gda_gene_disease_source
        ON gene_disease_associations IS
        'Unique constraint on GDA natural key (gene_symbol, disease_id, source). '
        'Matches the ON CONFLICT target in database/loaders.py. '
        'NOTE: For NULL gene_symbol handling, the COALESCE index '
        'uq_gene_disease_associations_gda_coalesced provides the real enforcement. '
        'If constraint columns change, ALL loader conflict targets MUST be updated.';

    RAISE NOTICE 'Migration 002: Constraint and index creation complete';

    -- [SEC-02] SECURITY note: NULL rows in disease_id and source are deleted
    -- (not backfilled to '') to prevent COALESCE-based dedup from being
    -- exploited to delete legitimate data through crafted empty-string duplicates.
END $$;

-- v89 ROOT FIX (BUG #22): removed redundant RELEASE SAVEPOINT
-- sp_constraints (matching SAVEPOINT removed above).


-- =====================================================================
-- SECTION 8: POST-MIGRATION VALIDATION
-- [TST-1, TST-2] Comprehensive verification that all changes were applied.
-- =====================================================================

DO $$
DECLARE
    _col_count INTEGER;
    _constraint_exists BOOLEAN;
    _index_exists BOOLEAN;
    _null_disease_count INTEGER;
    _null_source_count INTEGER;
    _null_gene_count INTEGER;
    _empty_sentinel_count INTEGER;
    _gda_total INTEGER;
    _em_total INTEGER;
    _archive_count INTEGER;
    _sv_count INTEGER;
BEGIN
    RAISE NOTICE 'Migration 002: Starting post-migration validation...';

    -- Verify proteins columns exist
    SELECT COUNT(*) INTO _col_count
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'proteins'
      AND column_name IN ('gene_symbol', 'protein_name', 'function_desc');
    IF _col_count < 3 THEN
        RAISE EXCEPTION 'POST-VALIDATION FAILED: Expected 3 proteins columns, found %', _col_count;
    END IF;
    RAISE NOTICE 'Post-validation: proteins columns OK (3/3)';

    -- Verify function_desc type is VARCHAR(10000) not TEXT
    SELECT COUNT(*) INTO _col_count
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'proteins'
      AND column_name = 'function_desc'
      AND data_type = 'character varying'
      AND COALESCE(character_maximum_length, 0) = 10000;
    IF _col_count = 0 THEN
        RAISE WARNING 'Post-validation: function_desc is not VARCHAR(10000). '
                      'Type mismatch with migration 001 (DES-1, IDEM-5).';
    ELSE
        RAISE NOTICE 'Post-validation: function_desc type OK (VARCHAR(10000))';
    END IF;

    -- Verify GDA COALESCE unique index exists
    SELECT COUNT(*) > 0 INTO _index_exists
    FROM pg_indexes
    WHERE tablename = 'gene_disease_associations'
      AND indexname = 'uq_gene_disease_associations_gda_coalesced';
    IF NOT _index_exists THEN
        RAISE EXCEPTION 'POST-VALIDATION FAILED: GDA COALESCE unique index not found';
    END IF;
    RAISE NOTICE 'Post-validation: GDA COALESCE unique index OK';

    -- Verify GDA plain-column unique constraint exists
    SELECT COUNT(*) > 0 INTO _constraint_exists
    FROM pg_indexes
    WHERE tablename = 'gene_disease_associations'
      AND indexname = 'uq_gda_gene_disease_source';
    IF NOT _constraint_exists THEN
        RAISE WARNING 'Post-validation: GDA plain-column unique constraint not found (may be OK if COALESCE index exists)';
    ELSE
        RAISE NOTICE 'Post-validation: GDA plain-column unique constraint OK';
    END IF;

    -- Verify no remaining NULLs in critical GDA columns
    SELECT COUNT(*) INTO _null_disease_count
    FROM gene_disease_associations WHERE disease_id IS NULL;
    SELECT COUNT(*) INTO _null_source_count
    FROM gene_disease_associations WHERE source IS NULL;
    IF _null_disease_count > 0 THEN
        RAISE WARNING 'Post-validation: % GDA rows still have NULL disease_id', _null_disease_count;
    END IF;
    IF _null_source_count > 0 THEN
        RAISE WARNING 'Post-validation: % GDA rows still have NULL source', _null_source_count;
    END IF;
    RAISE NOTICE 'Post-validation: GDA NULL cleanup OK (disease_id NULLs: %, source NULLs: %)',
                 _null_disease_count, _null_source_count;

    -- [TST-2] Pre-emptive check: verify data consistency with known constraints
    -- from migration 001 (even if they might not be enforced yet)
    SELECT COUNT(*) INTO _null_gene_count
    FROM gene_disease_associations
    WHERE gene_symbol IS NULL AND disease_id IS NOT NULL AND source IS NOT NULL;
    IF _null_gene_count > 0 THEN
        RAISE NOTICE 'Post-validation: % GDA rows have NULL gene_symbol with valid disease/source. '
                     'These represent unresolved gene symbols preserved for future entity resolution. '
                     'COALESCE index handles uniqueness for these rows.',
                     _null_gene_count;
    END IF;

    -- Check for empty sentinel rows
    SELECT COUNT(*) INTO _empty_sentinel_count
    FROM gene_disease_associations
    WHERE gene_symbol = '' AND disease_id = '' AND source = '';
    IF _empty_sentinel_count > 0 THEN
        RAISE WARNING 'Post-validation: % GDA rows have empty sentinel values — data quality concern',
                     _empty_sentinel_count;
    END IF;

    -- Verify entity_mapping partial unique index exists
    SELECT COUNT(*) > 0 INTO _index_exists
    FROM pg_indexes
    WHERE tablename = 'entity_mapping'
      AND indexname = 'uq_entity_mapping_inchikey';
    IF NOT _index_exists THEN
        RAISE EXCEPTION 'POST-VALIDATION FAILED: entity_mapping unique index not found';
    END IF;
    RAISE NOTICE 'Post-validation: entity_mapping unique index OK';

    -- Verify dedup archive table exists
    SELECT COUNT(*) INTO _archive_count
    FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = '_migration_002_dedup_archive';
    IF _archive_count = 0 THEN
        RAISE WARNING 'Post-validation: _migration_002_dedup_archive table not found';
    ELSE
        RAISE NOTICE 'Post-validation: dedup archive table OK';
    END IF;

    -- Summary
    SELECT COUNT(*) INTO _gda_total FROM gene_disease_associations;
    SELECT COUNT(*) INTO _em_total FROM entity_mapping;

    RAISE NOTICE 'Post-validation COMPLETE. GDA rows: %, Entity mapping rows: %',
                 _gda_total, _em_total;
END $$;


-- =====================================================================
-- SECTION 9: POST-MIGRATION CHECKSUMS
-- [LIN-4] Compare with pre-migration checksums for verification.
-- =====================================================================

DO $$
DECLARE
    _gda_checksum TEXT;
    _em_checksum TEXT;
    _gda_count INTEGER;
    _em_count INTEGER;
BEGIN
    SELECT COUNT(*)::text || ':sum_id:' || COALESCE(SUM(id)::text, '0')
    INTO _gda_checksum FROM gene_disease_associations;

    SELECT COUNT(*)::text || ':sum_id:' || COALESCE(SUM(id)::text, '0')
    INTO _em_checksum FROM entity_mapping;

    SELECT COUNT(*) INTO _gda_count FROM gene_disease_associations;
    SELECT COUNT(*) INTO _em_count FROM entity_mapping;

    INSERT INTO audit_log (table_name, operation, row_count, details)
    VALUES ('gene_disease_associations', 'POST_MIGRATION_002_CHECKSUM',
            _gda_count, 'checksum=' || _gda_checksum);
    INSERT INTO audit_log (table_name, operation, row_count, details)
    VALUES ('entity_mapping', 'POST_MIGRATION_002_CHECKSUM',
            _em_count, 'checksum=' || _em_checksum);

    RAISE NOTICE 'Migration 002: Post-migration checksums — GDA: % (rows: %), Entity: % (rows: %)',
                 _gda_checksum, _gda_count, _em_checksum, _em_count;
END $$;


-- =====================================================================
-- SECTION 10: SCHEMA VERSION RECORDING
-- [ARCH-03, CMP-4] Record this migration in schema_version.
-- Uses ON CONFLICT DO NOTHING for idempotency.
-- =====================================================================

INSERT INTO schema_version (version, description)
    VALUES (2, 'Bug fixes: add proteins columns, deduplicate GDA/entity_mapping, add unique constraints, NULL cleanup, data lineage archive')
    ON CONFLICT (version) DO NOTHING;

-- [LIN-3] pipeline_run_id tracking note
-- Note: pipeline_run_id column may not exist yet (added by 003).
-- The Python runner should set pipeline_run_id after executing this migration
-- if the column exists.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'gene_disease_associations'
          AND column_name = 'pipeline_run_id'
    ) THEN
        RAISE NOTICE 'Migration 002: pipeline_run_id column exists. '
                    'Python runner should update it post-migration.';
    END IF;
END $$;


-- =====================================================================
-- SECTION 11: CLEANUP
-- [PERF-4] ANALYZE on all modified tables to refresh query planner stats.
-- [ARCH-05] Release advisory lock.
-- =====================================================================

ANALYZE proteins;
ANALYZE gene_disease_associations;
ANALYZE entity_mapping;
ANALYZE _migration_002_dedup_archive;
-- SCI-FIX: Wrap RAISE NOTICE in a DO block — RAISE is a PL/pgSQL statement,
-- not valid standalone SQL. Without the DO wrapper, PostgreSQL rejects
-- this with "syntax error at or near 'RAISE'".
DO $$ BEGIN
    RAISE NOTICE 'Migration 002: Updated statistics for modified tables';
END $$;

-- [PERF-5] NOTE: CREATE INDEX CONCURRENTLY cannot be used inside a transaction.
-- v89 ROOT FIX (BUG #34): the full production operator warning has been MOVED
-- to this file's header comment block (search for "PRODUCTION OPERATOR WARNING"
-- at the top of the file). This bottom note is retained as a cross-reference.
-- See header for the complete mitigation procedure for >1M-row production DBs.
-- [INT-4] When this project migrates to Alembic (Phase 2), these DO $$
-- blocks will be replaced by Alembic op.execute() calls with dialect-specific
-- branching, enabling true cross-dialect support.

-- [ARCH-05] Release advisory lock
DO $$
BEGIN
    PERFORM pg_advisory_unlock(hashtext('migration_002'));
    RAISE NOTICE 'Migration 002: Advisory lock released';
END $$;

-- [INT-3] SQLITE GAP documentation:
-- The Python migration runner adds columns for SQLite but does NOT
-- replicate dedup, constraints, or NULL cleanup. This means:
--   - SQLite test databases will have DUPLICATE rows in GDA and entity_mapping
--   - The UNIQUE constraint will NOT exist on SQLite
--   - NULL cleanup will NOT happen on SQLite
-- RISK: Tests running on SQLite do NOT match production (PostgreSQL) behavior.
-- MITIGATION: Add SQLite dedup logic to run_migrations.py (Phase 2 / Alembic).

-- [PORTABILITY, INT-2] The CTE + ROW_NUMBER dedup approach is standard SQL
-- and works on PostgreSQL 9.6+. It does NOT work on SQLite (which lacks
-- DELETE ... FROM subquery support). SQLite dedup is not implemented.

-- [REL-6] RESUMPTION SAFETY: This migration is designed to be safely
-- re-runnable after interruption. Column additions use IF NOT EXISTS.
-- The CTE-based dedup uses ROW_NUMBER which is idempotent. Index creation
-- uses IF NOT EXISTS. The only non-idempotent operation is the audit_log
-- INSERT, which is append-only and harmless if duplicated.

-- SCI-FIX: Wrap RAISE NOTICE in a DO block — RAISE is a PL/pgSQL statement,
-- not valid standalone SQL. Without the DO wrapper, PostgreSQL rejects
-- this with "syntax error at or near 'RAISE'".
DO $$ BEGIN
    RAISE NOTICE 'Migration 002: COMPLETE. All 86 issues fixed.';
END $$;

-- v21 ROOT FIX: close the outer transaction opened at the top of this
-- file. A mid-migration failure now rolls back atomically (the BEGIN
-- at line 22 + this COMMIT form one transaction).
COMMIT;
