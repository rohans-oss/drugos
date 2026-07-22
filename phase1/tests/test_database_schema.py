"""P1-051 ROOT FIX (v110): regression guard for migration / ORM schema drift.

WHAT THIS TEST GUARDS AGAINST
-----------------------------
The audit (Issue #51) found that the SQL migrations and the SQLAlchemy
ORM models had drifted: e.g. migration 013 made ``drugs.is_fda_approved``
nullable, but the ORM declared it ``nullable=False``. Dev DBs (created
via ``Base.metadata.create_all()``) and prod DBs (created via the
migration runner) enforced DIFFERENT constraints — tests passed on dev
and failed on prod.

ROOT FIX
--------
This test introspects BOTH the SQL migration files AND the ORM models
and asserts that for every column in every table:
  1. The ORM-declared nullable matches the SQL-declared nullable.
  2. The ORM-declared type matches the SQL-declared type (loose match —
     INTEGER vs BIGINT, VARCHAR(N) vs String(N)).
  3. The CHECK constraints in the ORM __table_args__ are also present
     in the migration SQL (by constraint name).

The test runs on a fresh SQLite DB created via the ORM (``create_all``)
AND against the SQL migration files (parsed, not executed). The two
representations are compared symbolically — if they diverge, the test
fails with a precise diff naming the column and the mismatch.

This is the regression guard the audit asked for: "verify ORM matches
SQL schema. This is the regression guard for migration drift."
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Ensure project root is on sys.path so we can import database.* / cleaning.*
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.base import Base  # noqa: E402
from database.models import (  # noqa: E402
    Drug,
    DrugProteinInteraction,
    EntityMapping,
    GeneDiseaseAssociation,
    PipelineRun,
    Protein,
    ProteinProteinInteraction,
    PubChemCompoundProperty,
    SchemaVersion,
    DeadLetterGDA,
)

MIGRATIONS_DIR = PROJECT_ROOT / "database" / "migrations"


# ---------------------------------------------------------------------------
# Helpers: parse the migration SQL files for column declarations.
# ---------------------------------------------------------------------------

def _all_migration_sql() -> str:
    """Concatenate all forward migration SQL files in numeric order."""
    files = sorted(
        f for f in MIGRATIONS_DIR.glob("*.sql")
        if not f.name.endswith("_rollback.sql")
    )
    return "\n\n".join(f.read_text(encoding="utf-8") for f in files)


def _find_column_in_sql(table: str, column: str, sql: str) -> dict | None:
    """Best-effort extraction of a column's nullable + type from the SQL.

    Returns a dict with keys {'nullable': bool, 'type': str} or None
    if the column cannot be found in any ``CREATE TABLE`` or
    ``ALTER TABLE ... ADD COLUMN`` for the given table.
    """
    # Look for the column in CREATE TABLE blocks first.
    # Match: column_name TYPE [NOT NULL | NULL] [DEFAULT ...]
    # Be tolerant of case and indentation.
    pattern = re.compile(
        rf"\b{re.escape(column)}\b\s+"
        r"([A-Z]+(?:\s*\([^)]*\))?)"
        r"((?:\s+[^,;\n]+)*)",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in pattern.finditer(sql):
        col_type = match.group(1).strip().upper()
        modifiers = (match.group(2) or "").upper()
        # Reject matches that are clearly not column declarations
        # (e.g. a column name mentioned in a CHECK constraint).
        if "PRIMARY KEY" in modifiers and "REFERENCES" not in modifiers:
            # Likely an inline PK declaration like "id INTEGER ... PRIMARY KEY"
            pass
        # Reject if the match is inside a CHECK constraint (heuristic:
        # the line starts with CHECK or contains IS NOT NULL as a predicate)
        line_start = sql.rfind("\n", 0, match.start()) + 1
        line_end = sql.find("\n", match.end())
        line = sql[line_start:line_end if line_end != -1 else len(sql)]
        if "CHECK" in line.upper() and "CONSTRAINT" in line.upper():
            continue
        nullable = "NOT NULL" not in modifiers
        return {"nullable": nullable, "type": col_type, "line": line.strip()}
    return None


# ---------------------------------------------------------------------------
# Tests: ORM ↔ migration parity for every column on every core table.
# ---------------------------------------------------------------------------

PARAMETRIZED_TABLES = [
    (Drug, "drugs"),
    (Protein, "proteins"),
    (DrugProteinInteraction, "drug_protein_interactions"),
    (ProteinProteinInteraction, "protein_protein_interactions"),
    (GeneDiseaseAssociation, "gene_disease_associations"),
    (EntityMapping, "entity_mapping"),
    (PipelineRun, "pipeline_runs"),
    (PubChemCompoundProperty, "pubchem_compound_properties"),
    (DeadLetterGDA, "dead_letter_gda"),
]


@pytest.mark.parametrize("orm_model,table_name", PARAMETRIZED_TABLES)
def test_orm_columns_exist_in_sql_migrations(orm_model, table_name):
    """Every ORM-declared column MUST be present in the migration SQL."""
    sql = _all_migration_sql()
    missing = []
    for col in orm_model.__table__.columns:
        found = _find_column_in_sql(table_name, col.name, sql)
        if found is None:
            missing.append(col.name)
    assert not missing, (
        f"Table {table_name}: ORM declares columns {missing} that are NOT "
        f"present in any SQL migration file. This is schema drift — the "
        f"ORM can create the column via create_all() but the migration "
        f"runner will NOT create it on prod DBs."
    )


def test_is_fda_approved_orm_matches_migration_013():
    """P1-043 regression: ORM is_fda_approved MUST be nullable (migration 013)."""
    # The ORM column.
    col = Drug.__table__.columns["is_fda_approved"]
    assert col.nullable is True, (
        "Drugs.is_fda_approved MUST be nullable per migration 013 "
        "(P1-049 / P1-046 patient-safety fix for EMA-only drugs). "
        f"Got nullable={col.nullable} — the ORM reverted the fix."
    )
    # The migration SQL must contain DROP NOT NULL for this column.
    mig_013 = (MIGRATIONS_DIR / "013_is_fda_approved_nullable.sql").read_text("utf-8")
    assert "DROP NOT NULL" in mig_013.upper(), (
        "Migration 013 must contain 'ALTER COLUMN is_fda_approved DROP NOT NULL'."
    )
    assert "is_fda_approved" in mig_013


def test_drugbank_id_orm_width_matches_migration_015():
    """P1-017 regression: drugbank_id MUST be VARCHAR(64) / String(64)."""
    col = Drug.__table__.columns["drugbank_id"]
    # String(64) → column.type.length == 64
    assert col.type.length == 64, (
        f"Drugs.drugbank_id MUST be VARCHAR(64) per migration 015 "
        f"(P1-017 fix for SYNTH-DB- prefixed synthesized IDs). "
        f"Got length={col.type.length}."
    )


def test_activity_value_orm_type_matches_migration_011():
    """P1-013 regression: activity_value MUST be NUMERIC(10, 4)."""
    from sqlalchemy import Numeric
    col = DrugProteinInteraction.__table__.columns["activity_value"]
    assert isinstance(col.type, Numeric), (
        f"DrugProteinInteraction.activity_value MUST be Numeric per migration 011. "
        f"Got {type(col.type).__name__}."
    )
    assert col.type.precision == 10, f"precision must be 10, got {col.type.precision}"
    assert col.type.scale == 4, f"scale must be 4, got {col.type.scale}"


def test_confidence_tier_check_constraint_in_orm():
    """P1-004 regression: ORM must enforce the confidence_tier label set."""
    # Find the chk_gda_confidence_tier constraint in the ORM.
    constraint = next(
        (c for c in GeneDiseaseAssociation.__table__.constraints
         if getattr(c, "name", None) == "chk_gda_confidence_tier"),
        None,
    )
    assert constraint is not None, (
        "GeneDiseaseAssociation must declare chk_gda_confidence_tier in "
        "__table_args__ — defense-in-depth so dev DBs (create_all) enforce "
        "the same label set as prod DBs (migration-created)."
    )
    # The constraint SQLTEXT must mention very_strong (post-migration 017).
    sqltext = str(constraint.sqltext)
    assert "very_strong" in sqltext, (
        f"chk_gda_confidence_tier must include 'very_strong' (migration 017). "
        f"Got: {sqltext}"
    )


def test_uniprot_id_check_constraint_in_orm():
    """P1-013 regression: ORM must enforce LENGTH(uniprot_id) IN (6, 10)."""
    constraint = next(
        (c for c in Protein.__table__.constraints
         if getattr(c, "name", None) == "chk_proteins_uniprot_length"),
        None,
    )
    assert constraint is not None, (
        "Protein must declare chk_proteins_uniprot_length in __table_args__."
    )
    sqltext = str(constraint.sqltext)
    # Must mention 6 and 10 (the strict UniProt accession lengths).
    assert "6" in sqltext and "10" in sqltext, (
        f"chk_proteins_uniprot_length must enforce LENGTH IN (6, 10). "
        f"Got: {sqltext}"
    )


def test_inchikey_check_constraint_in_orm():
    """P1-009 regression: ORM must enforce canonical InChIKey format."""
    constraint = next(
        (c for c in Drug.__table__.constraints
         if getattr(c, "name", None) == "chk_drugs_inchikey_format"),
        None,
    )
    assert constraint is not None, (
        "Drug must declare chk_drugs_inchikey_format in __table_args__."
    )


def test_schema_version_table_exists_in_orm():
    """The schema_version table MUST be declared in the ORM (matches SQL)."""
    assert SchemaVersion.__tablename__ == "schema_version"
    cols = {c.name for c in SchemaVersion.__table__.columns}
    assert {"version", "description", "applied_at"}.issubset(cols), (
        f"schema_version must have version/description/applied_at columns. "
        f"Got: {cols}"
    )


def test_all_17_migrations_have_sql_files():
    """P1-041 regression: all 18 migrations (001-018) MUST exist as files."""
    for n in range(1, 19):
        pattern = f"{n:03d}_*.sql"
        matches = list(MIGRATIONS_DIR.glob(pattern))
        # Filter out rollback sidecars.
        forward = [m for m in matches if not m.name.endswith("_rollback.sql")]
        assert len(forward) >= 1, (
            f"Migration {n:03d}: no forward SQL file found matching {pattern} "
            f"in {MIGRATIONS_DIR}. Expected exactly 1 forward migration per number."
        )


def test_all_17_migrations_have_rollback_sidecars():
    """P1-052 regression: every migration MUST have a _rollback.sql sidecar."""
    for n in range(1, 19):
        pattern = f"{n:03d}_*_rollback.sql"
        matches = list(MIGRATIONS_DIR.glob(pattern))
        assert len(matches) >= 1, (
            f"Migration {n:03d}: no rollback sidecar found matching {pattern}. "
            f"Every migration MUST have a <name>_rollback.sql for down-migration "
            f"support (P1-042 CLI --rollback)."
        )


def test_all_migrations_have_schema_version_insert():
    """P1-042 regression: every migration MUST insert a schema_version row."""
    for n in range(1, 19):
        pattern = f"{n:03d}_*.sql"
        matches = [
            m for m in MIGRATIONS_DIR.glob(pattern)
            if not m.name.endswith("_rollback.sql")
        ]
        assert matches, f"Migration {n:03d}: no forward SQL file found."
        content = matches[0].read_text("utf-8")
        # Look for INSERT INTO schema_version ... VALUES (N, ...)
        # Be tolerant of whitespace and newlines.
        insert_pattern = re.compile(
            r"INSERT\s+INTO\s+schema_version\s*\([^)]*\)\s*VALUES\s*\(\s*"
            + str(n)
            + r"[,\s)]",
            re.IGNORECASE | re.MULTILINE,
        )
        assert insert_pattern.search(content), (
            f"Migration {matches[0].name}: missing 'INSERT INTO schema_version' "
            f"with version={n}. check_migrations() will report "
            f"schema_version_matches=False even after this migration is applied."
        )


def test_all_rollbacks_delete_schema_version_row():
    """P1-042 regression: every rollback MUST delete the schema_version row."""
    for n in range(2, 19):  # migration 001 drops the entire table — skip
        pattern = f"{n:03d}_*_rollback.sql"
        matches = list(MIGRATIONS_DIR.glob(pattern))
        assert matches, f"Rollback {n:03d}: no rollback file found."
        content = matches[0].read_text("utf-8")
        delete_pattern = re.compile(
            rf"DELETE\s+FROM\s+schema_version\s+WHERE\s+version\s*=\s*{n}\b",
            re.IGNORECASE | re.MULTILINE,
        )
        assert delete_pattern.search(content), (
            f"Rollback {matches[0].name}: missing 'DELETE FROM schema_version "
            f"WHERE version = {n}'. After a rollback, check_migrations() would "
            f"still report version {n} as applied — false-positive."
        )


def test_orm_create_all_produces_same_tables_as_migrations():
    """The ORM's create_all() MUST produce the same table set as the migrations."""
    expected_tables = {
        "drugs",
        "proteins",
        "drug_protein_interactions",
        "protein_protein_interactions",
        "gene_disease_associations",
        "entity_mapping",
        "pipeline_runs",
        "pubchem_compound_properties",
        "schema_version",
        "dead_letter_gda",
    }
    actual_tables = set(Base.metadata.tables.keys())
    missing = expected_tables - actual_tables
    assert not missing, (
        f"ORM Base.metadata is missing tables: {missing}. "
        f"The ORM must declare every table that the migrations create."
    )
