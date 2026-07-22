-- ============================================================================
-- Drug Repurposing ETL Platform — Rollback 007 PipelineRun.metadata_json
-- Migration: 007_pipeline_run_metadata_rollback.sql
-- Description: Reverses the ALTER TABLE changes from
--              007_pipeline_run_metadata.sql.
--
-- ROOT-CAUSE FIX (audit P1-18): rollback sidecar for 007.
-- ============================================================================

BEGIN;

-- Drop the column added by 007.
ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS metadata_json;

-- v76 ROOT FIX (T-037 — schema_version rollback consistency):
-- Delete the schema_version row inserted by 007 (version=7). All
-- rollbacks now follow the same convention: delete the version row so
-- schema_version always reflects the CURRENT schema state.
DELETE FROM schema_version WHERE version = 7;

COMMIT;
