-- ============================================================================
-- Drug Repurposing ETL Platform — PipelineRun.metadata_json Column Migration
-- Migration: 007_pipeline_run_metadata.sql
-- Description: Add a `metadata_json` JSON column to the `pipeline_runs` table
--              so the rich per-run audit metadata that BasePipeline already
--              computes (run_id, correlation_id, triggered_by, source_version,
--              sha256_raw, sha256_cleaned, git_commit, seed, schema_version,
--              validation_errors, dq_metrics, record counts) is persisted
--              alongside the existing fixed columns instead of being silently
--              discarded by the PipelineRun(...) constructor.
--
-- ROOT-CAUSE FIX (audit P1-18):
--   BasePipeline._write_run_log (phase1/pipelines/base_pipeline.py) builds a
--   `metadata_json` dict containing run_id / sha256 / dq_metrics / git_commit /
--   seed / schema_version / validation_errors and passes it to _write_run_log.
--   But the PipelineRun ORM model did not declare a `metadata_json` column,
--   and the PipelineRun(...) constructor calls at lines ~4338 and ~4475 did
--   not pass `metadata_json=...`. The metadata was therefore silently
--   dropped on every successful run — operators could not query the DB for
--   "which git commit produced this run" or "what was the raw-input SHA-256"
--   without trawling log files.
--
-- This migration adds the column; the ORM and constructor are updated in
-- parallel (database/models.py, base_pipeline.py).
--
-- All new columns are NULLABLE — existing rows and existing tests are
-- unaffected. No columns are dropped, no constraints are weakened.
--
-- PREREQUISITES: 001_initial_schema.sql through 006_drug_withdrawn_safety_columns.sql.
-- Dialects: PostgreSQL (JSONB) and SQLite (JSON via TEXT).
--
-- v75 ROOT FIX (T-026 — PostgreSQL PL-pgSQL DO block fails on SQLite):
--   The v74 migration used a PL-pgSQL anonymous DO block (the
--   PostgreSQL-specific "DO dollar-quote" construct) with
--   information_schema.columns lookups and a JSONB-vs-TEXT exception
--   handler. On PostgreSQL this worked correctly. On SQLite (the
--   dev/test backend), the entire DO block is invalid PL-pgSQL:
--     * SQLite has no PL-pgSQL DO statement (it is PostgreSQL-only).
--     * SQLite has no information_schema catalog (it uses
--       sqlite_master and PRAGMA table_info).
--     * SQLite has no JSONB type (JSON values are stored as TEXT).
--   The _translate_sql_for_sqlite regex translator in
--   run_migrations.py (lines 2199-2418) attempted to strip the
--   PL-pgSQL wrapper and leave the inner SQL — but the inner SQL
--   contained "ALTER TABLE ... ADD COLUMN metadata_json JSONB" which
--   is itself invalid on SQLite. The translator's EXCEPTION WHEN
--   OTHERS stripping sometimes left a bare "ALTER TABLE ... JSONB"
--   that raised OperationalError near JSONB syntax error.
--   Worse: under the V18 CD-5 root-fix policy, translation failures
--   must FAIL HARD (run_migrations.py:3604-3646) — so a single bad
--   migration blocks the ENTIRE migration chain on SQLite.
--
--   ROOT FIX (master-grade, no translator dependency):
--     Replace the DO block with portable ANSI SQL that both dialects
--     execute natively, with NO PL-pgSQL and NO information_schema.
--     The migration is now two parts:
--       (1) ALTER TABLE pipeline_runs ADD COLUMN metadata_json TEXT
--           — works on BOTH SQLite (native TEXT) and PostgreSQL (TEXT
--           is universally accepted). The IF NOT EXISTS clause is
--           supported by PostgreSQL 9.6+ and SQLite 3.35+. Older
--           SQLite raises a syntax error on IF NOT EXISTS; the
--           migration runner catches OperationalError and treats
--           "duplicate column" as a no-op (run_migrations.py:4314).
--       (2) On PostgreSQL ONLY, upgrade the TEXT column to JSONB so
--           production gets the indexable, deduplicated JSON storage
--           that JSONB provides. The upgrade is guarded by a dialect
--           check at the migration runner level (NOT inside this SQL
--           file) — see run_migrations.py::_apply_postgres_only.
--           For SQLite, the column stays TEXT (the SQLAlchemy JSON
--           dialect serialises Python dicts to TEXT transparently).
--
--   This fix removes the dependency on _translate_sql_for_sqlite
--   for this migration. The migration is now portable SQL that both
--   dialects parse and execute without translation.
--   NOTE: this comment deliberately avoids the literal "DO dollar-dollar"
--   token so the translator's regex does not match it inside a comment.
-- ============================================================================

BEGIN;

-- ===========================================================================
-- Phase 1: Add metadata_json column (portable — works on SQLite + PostgreSQL)
-- ===========================================================================
-- Use TEXT on BOTH dialects. On PostgreSQL, the column is upgraded to JSONB
-- by the post-migration hook (run_migrations.py::_apply_postgres_only).
-- On SQLite, TEXT is the only option (SQLite has no native JSON column type;
-- the SQLAlchemy JSON dialect serialises Python dicts to TEXT transparently).
-- The IF NOT EXISTS guard makes this safe to re-run on both dialects.
-- SQLite <3.35 does not support IF NOT EXISTS on ADD COLUMN — the migration
-- runner's OperationalError handler treats "duplicate column" as a no-op.
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS metadata_json TEXT;

-- ===========================================================================
-- Phase 2: Schema version metadata
-- ===========================================================================
INSERT INTO schema_version (version, description)
VALUES (7, 'Add metadata_json JSON column to pipeline_runs so per-run audit metadata (run_id, sha256_raw, sha256_cleaned, git_commit, dq_metrics, validation_errors) is persisted instead of silently discarded')
ON CONFLICT (version) DO NOTHING;

COMMIT;
