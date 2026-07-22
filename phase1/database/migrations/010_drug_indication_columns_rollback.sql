-- Rollback for migration 010 — Drug indication columns + GDA gene_id unique index
-- ====================================================================
-- Reverses migration 010. Safe to run on PostgreSQL 12+ and SQLite.
--
-- v73 ROOT FIX (T-008 — atomicity + version-column type contract):
--   The previous rollback (like the forward 010 migration) lacked a
--   ``BEGIN;`` ... ``COMMIT;`` wrapper, making it the only non-atomic
--   rollback in the 001-010 chain. A partial failure during manual
--   ``psql -f`` execution left the schema half-rolled-back (columns
--   dropped but the ``schema_version`` row not deleted — or vice
--   versa). Additionally, the ``DELETE FROM schema_version WHERE
--   version = '010'`` clause used a STRING literal for an INTEGER
--   column, inconsistent with the bare-INTEGER convention used by
--   every other rollback.
--
--   ROOT FIX: wrap in ``BEGIN;`` ... ``COMMIT;`` and use the INTEGER
--   literal ``10`` so the DELETE matches the row inserted by the
--   forward migration (which now also uses INTEGER 10 after the
--   T-008 fix).

BEGIN;

-- Drop the GDA indexes first (they depend on gene_disease_associations)
-- v90 ROOT FIX (BUG #19): the partial index was renamed from
-- idx_gda_gene_id to idx_gda_gene_id_partial in migration 010. Drop the
-- renamed partial index here. (The full idx_gda_gene_id from migration
-- 004 is NOT dropped here — it belongs to migration 004, not 010.)
DROP INDEX IF EXISTS idx_gda_gene_id_partial;
DROP INDEX IF EXISTS uq_gda_gene_id_disease_source;

-- Drop the drug indication columns
ALTER TABLE drugs DROP COLUMN IF EXISTS indication_source;
ALTER TABLE drugs DROP COLUMN IF EXISTS indication;

-- Remove the schema_version row. Use INTEGER literal 10 to match
-- the forward migration's INSERT (T-008 fix).
DELETE FROM schema_version WHERE version = 10;

COMMIT;
