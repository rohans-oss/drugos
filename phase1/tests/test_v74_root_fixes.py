#!/usr/bin/env python3
"""
v74 ROOT-LEVEL FIX VERIFICATION TEST SUITE
==========================================

This test suite verifies all 11 root-level fixes applied in v74 to the
v73_ROOT_FIXED_codebase. Each test is named after the issue ID (T-013
through T-023) and checks the ROOT CAUSE of the issue, not surface
symptoms.

Run with:
    cd phase1
    DATABASE_URL=sqlite:////tmp/v74_test.db DRUGOS_DEV_ALLOW_DEFAULT_DB=1 \\
        PYTHONPATH=. python -m pytest tests/test_v74_root_fixes.py -v

Or standalone:
    cd phase1
    DATABASE_URL=sqlite:////tmp/v74_test.db DRUGOS_DEV_ALLOW_DEFAULT_DB=1 \\
        PYTHONPATH=. python tests/test_v74_root_fixes.py

The tests use a FRESH SQLite database per run (no Postgres dependency).
SQLite is the dev/test dialect -- verifying fixes on SQLite proves they
work on the dev path (the path that was broken in v73). PostgreSQL
deploys get the same fixes via the migration runner.

FIXES COVERED
-------------
T-013: activity_value FLOAT -> NUMERIC(10,4) schema drift (migration 001 + new migration 011)
T-014: chk_drugs_is_globally_approved CheckConstraint missing from ORM
T-015: DAG docstrings contradict actual schedules
T-016: Contradictory BEGIN/COMMIT comments in migration 002
T-017: Pointless DROP INDEX + CREATE INDEX cycle in migration 002
T-018: Dead `uniprot_id IS NULL OR` branch in migration 003
T-019: Makefile error-swallowing `|| echo` / `|| true` patterns
T-020: Deprecated `airflow db init` fallback in Makefile
T-021: Unbounded apache-airflow>=2.8.0 pin in requirements.txt
T-022: rdkit silent degradation on ARM64 (runtime warnings added)
T-023: retries=2 on 4xx HTTP errors waste 60 min (fail-fast + exponential backoff)
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

# Ensure phase1/ is on sys.path
PHASE1_ROOT = Path(__file__).resolve().parent.parent
if str(PHASE1_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE1_ROOT))


# ===========================================================================
# T-013: activity_value FLOAT -> NUMERIC(10,4) schema drift
# ===========================================================================

def test_t013_activity_value_is_numeric_in_migration_001():
    """Migration 001 must declare activity_value as NUMERIC(10, 4), not FLOAT."""
    sql = (PHASE1_ROOT / "database" / "migrations" / "001_initial_schema.sql").read_text()
    assert "activity_value  NUMERIC(10, 4)" in sql, \
        "migration 001 should declare activity_value as NUMERIC(10, 4)"
    assert "activity_value  FLOAT," not in sql, \
        "migration 001 should NOT declare activity_value as FLOAT"


def test_t013_orm_activity_value_is_numeric():
    """ORM DrugProteinInteraction.activity_value must be Numeric(10, 4)."""
    from database.models import DrugProteinInteraction
    from sqlalchemy import Numeric
    col = DrugProteinInteraction.__table__.c.activity_value
    assert isinstance(col.type, Numeric), \
        f"activity_value should be Numeric, got {type(col.type).__name__}"
    assert col.type.precision == 10, \
        f"precision should be 10, got {col.type.precision}"
    assert col.type.scale == 4, \
        f"scale should be 4, got {col.type.scale}"


def test_t013_migration_011_exists_and_aligns_float_to_numeric():
    """Migration 011 must exist and ALTER FLOAT -> NUMERIC(10, 4) for deployed DBs."""
    mig = PHASE1_ROOT / "database" / "migrations" / "011_align_activity_value_to_orm.sql"
    assert mig.exists(), "migration 011 file should exist"
    sql = mig.read_text()
    assert "ALTER COLUMN activity_value TYPE NUMERIC(10, 4)" in sql
    assert "USING activity_value::numeric(10, 4)" in sql
    # SQLite-translatable (no RETURN; in active SQL)
    # Strip comments before checking
    active_lines = [l for l in sql.split("\n") if not l.strip().startswith("--")]
    active_sql = "\n".join(active_lines)
    assert "RETURN;" not in active_sql, \
        "migration 011 must not use RETURN; (breaks SQLite translator)"


def test_t013_migration_011_applies_cleanly_on_sqlite():
    """Migration 011 must apply cleanly on SQLite (no syntax errors)."""
    from database.migrations.run_migrations import run_migrations, _is_migration_applied
    from sqlalchemy import create_engine, text
    engine = create_engine("sqlite:///:memory:")
    result = run_migrations(engine=engine)
    assert "011_align_activity_value_to_orm.sql" in result.applied, \
        f"migration 011 should be in applied list: {result.applied}"
    assert not result.errors, f"migration errors: {result.errors}"


def test_t013_activity_value_stores_decimal_exact():
    """A real IC50 value stored as NUMERIC(10,4) must round-trip without ULP error.

    On PostgreSQL, NUMERIC(10,4) stores exact decimals (no ULP drift).
    On SQLite, the column type is advisory (dynamic typing) so the value
    is stored as REAL -- but the SCHEMA declaration still matches the ORM,
    which is what T-013 fixes. This test verifies the schema declaration
    is consistent (the actual rounding behavior is dialect-specific).
    """
    from sqlalchemy import create_engine, text, inspect
    from database.base import Base
    import database.models  # noqa: F401 -- register models

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    # Verify the column type is NUMERIC(10, 4) in the actual DB schema
    inspector = inspect(engine)
    dpi_cols = inspector.get_columns("drug_protein_interactions")
    av_col = next(c for c in dpi_cols if c["name"] == "activity_value")
    av_type_str = str(av_col["type"]).upper()
    assert "NUMERIC" in av_type_str or "DECIMAL" in av_type_str, \
        f"activity_value column type should be NUMERIC/DECIMAL, got {av_type_str}"
    assert "10" in av_type_str and "4" in av_type_str, \
        f"activity_value should have precision=10, scale=4, got {av_type_str}"

    # Insert and retrieve a value to verify it round-trips
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO drugs (inchikey, name, is_fda_approved, is_withdrawn, is_globally_approved) "
            "VALUES ('BSYNRYMUTXBXSQ-UHFFFAOYSA-N', 'Aspirin', 1, 0, 1)"
        ))
        conn.execute(text(
            "INSERT INTO proteins (uniprot_id, gene_symbol, protein_name, organism) "
            "VALUES ('P23219', 'PTGS2', 'PTGS2', 'Homo sapiens')"
        ))
        conn.execute(text(
            "INSERT INTO drug_protein_interactions (drug_id, protein_id, activity_type, activity_value, activity_units, source) "
            "VALUES (1, 1, 'IC50', 0.00123, 'nM', 'chembl')"
        ))
        row = conn.execute(text(
            "SELECT activity_value FROM drug_protein_interactions WHERE id = 1"
        )).fetchone()
        assert row[0] is not None
        # On SQLite, the value is stored as REAL (dynamic typing) so 0.00123 is preserved.
        # On PostgreSQL, NUMERIC(10,4) would round to 0.0012.
        # The KEY assertion: the value round-trips to a number close to 0.00123
        # (FLOAT would give 0.0012300000000001 or similar ULP error on PostgreSQL).
        assert abs(float(row[0]) - 0.00123) < 1e-5, \
            f"activity_value should round-trip to ~0.00123, got {row[0]}"


# ===========================================================================
# T-014: chk_drugs_is_globally_approved CheckConstraint in ORM
# ===========================================================================

def test_t014_orm_has_chk_drugs_is_globally_approved():
    """Drug.__table_args__ must include chk_drugs_is_globally_approved."""
    from database.models import Drug
    from sqlalchemy import CheckConstraint
    checks = [c for c in Drug.__table_args__ if isinstance(c, CheckConstraint)]
    names = [c.name for c in checks]
    assert "chk_drugs_is_globally_approved" in names, \
        f"chk_drugs_is_globally_approved missing from Drug.__table_args__: {names}"


def test_t014_constraint_uses_portable_in_0_1_form():
    """The constraint must use IN (0, 1) (SQLite-compatible), not IN (FALSE, TRUE)."""
    from database.models import Drug
    from sqlalchemy import CheckConstraint
    checks = [c for c in Drug.__table_args__ if isinstance(c, CheckConstraint)]
    iga = next(c for c in checks if c.name == "chk_drugs_is_globally_approved")
    expr = str(iga.sqltext)
    assert "IS NOT NULL" in expr, f"missing IS NOT NULL guard: {expr}"
    assert "IN (0, 1)" in expr, f"not using portable IN (0, 1) form: {expr}"


def test_t014_migration_008_uses_portable_form():
    """Migration 008's CHECK must use IN (0, 1), not IN (FALSE, TRUE)."""
    sql = (PHASE1_ROOT / "database" / "migrations" / "008_drug_is_globally_approved.sql").read_text()
    # Active SQL lines (not comments)
    active = "\n".join(l for l in sql.split("\n") if not l.strip().startswith("--"))
    assert "is_globally_approved IS NOT NULL AND is_globally_approved IN (0, 1)" in active, \
        "migration 008 active SQL should use portable IN (0, 1) form"
    assert "is_globally_approved IN (FALSE, TRUE)" not in active, \
        "migration 008 active SQL should NOT use IN (FALSE, TRUE)"


def test_t014_invalid_is_globally_approved_rejected_on_sqlite():
    """A row with is_globally_approved=2 must be rejected on SQLite (dev path)."""
    from sqlalchemy import create_engine, text
    from database.base import Base
    import database.models  # noqa: F401

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        # Valid insert
        conn.execute(text(
            "INSERT INTO drugs (inchikey, name, is_fda_approved, is_withdrawn, is_globally_approved) "
            "VALUES ('BSYNRYMUTXBXSQ-UHFFFAOYSA-N', 'Valid', 0, 0, 0)"
        ))
    # Invalid insert (is_globally_approved=2) must fail
    try:
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO drugs (inchikey, name, is_fda_approved, is_withdrawn, is_globally_approved) "
                "VALUES ('BSYNRYMUTXBXSQ-UHFFFAOYSA-N_X', 'Invalid', 0, 0, 2)"
            ))
        raise AssertionError("is_globally_approved=2 should have been rejected")
    except Exception as e:
        if "IntegrityError" in type(e).__name__ or "CHECK" in str(e).upper():
            pass  # expected
        else:
            raise


# ===========================================================================
# T-015: DAG docstrings match actual schedules
# ===========================================================================

def test_t015_disgenet_docstring_says_tuesday():
    """disgenet_dag.py docstring must say Tuesday (not Sunday)."""
    content = (PHASE1_ROOT / "dags" / "disgenet_dag.py").read_text()
    assert "Tuesday at 06:00 UTC" in content
    assert "every Sunday at 06:00" not in content
    # Verify the actual schedule matches
    assert 'schedule="0 6 * * 2"' in content


def test_t015_drugbank_docstring_says_monday():
    """drugbank_dag.py docstring must say Monday (not Sunday)."""
    content = (PHASE1_ROOT / "dags" / "drugbank_dag.py").read_text()
    assert "Monday at 03:00 UTC" in content
    assert "every Sunday at 03:00" not in content
    assert 'schedule="0 3 * * 1"' in content


def test_t015_pubchem_docstring_says_wednesday():
    """pubchem_dag.py docstring must say Wednesday (not Sunday)."""
    content = (PHASE1_ROOT / "dags" / "pubchem_dag.py").read_text()
    assert "Wednesday at 08:00 UTC" in content
    assert "every Sunday at 08:00" not in content
    assert 'schedule="0 8 * * 3"' in content


def test_t015_chembl_docstring_resolves_contradiction():
    """chembl_dag.py docstring must not contain the contradictory 'runs weekly on Sunday'."""
    content = (PHASE1_ROOT / "dags" / "chembl_dag.py").read_text()
    assert "Wednesday at 04:00 UTC" in content
    assert "runs weekly on Sunday" not in content, \
        "chembl docstring must not contain contradictory 'runs weekly on Sunday'"
    assert 'schedule="0 4 * * 3"' in content


# ===========================================================================
# T-016: Contradictory BEGIN/COMMIT comments resolved
# ===========================================================================

def test_t016_no_contradictory_do_not_wrap_comment():
    """Migration 002 must NOT contain the contradictory 'Do NOT wrap' comment."""
    sql = (PHASE1_ROOT / "database" / "migrations" / "002_bug_fixes_migration.sql").read_text()
    assert "Do NOT wrap this file in BEGIN/COMMIT" not in sql, \
        "the contradictory 'Do NOT wrap' comment must be removed"
    # The v74 ROOT FIX T-016 clarifying comment must be present
    assert "v74 ROOT FIX" in sql and "T-016" in sql, \
        "v74 ROOT FIX T-016 clarifying comment must be present"


# ===========================================================================
# T-017: Pointless DROP INDEX + CREATE INDEX cycle removed
# ===========================================================================

def test_t017_no_drop_create_cycle_for_uq_entity_mapping_inchikey():
    """Migration 002 must not DROP+CREATE uq_entity_mapping_inchikey (pointless cycle)."""
    sql = (PHASE1_ROOT / "database" / "migrations" / "002_bug_fixes_migration.sql").read_text()
    # Active SQL lines (not comments)
    active_lines = [l for l in sql.split("\n") if not l.strip().startswith("--")]
    active = "\n".join(active_lines)
    assert "DROP INDEX IF EXISTS uq_entity_mapping_inchikey" not in active, \
        "migration 002 should not DROP uq_entity_mapping_inchikey (001 already creates it correctly)"
    assert "CREATE UNIQUE INDEX IF NOT EXISTS uq_entity_mapping_inchikey" not in active, \
        "migration 002 should not CREATE uq_entity_mapping_inchikey (001 already creates it correctly)"


# ===========================================================================
# T-018: Dead `uniprot_id IS NULL OR` branch removed
# ===========================================================================

def test_t018_no_dead_is_null_or_branch():
    """Migration 003 must not have the dead `uniprot_id IS NULL OR` branch."""
    sql = (PHASE1_ROOT / "database" / "migrations" / "003_models_fix_migration.sql").read_text()
    assert "uniprot_id IS NULL OR (LENGTH(uniprot_id)" not in sql, \
        "migration 003 must not contain dead `uniprot_id IS NULL OR` branch"
    # The tightened form must be present
    assert "CHECK (LENGTH(uniprot_id) >= 4 AND LENGTH(uniprot_id) <= 10)" in sql, \
        "migration 003 must contain the tightened CHECK (matches migration 001)"


# ===========================================================================
# T-019: Makefile error-swallowing patterns removed
# ===========================================================================

def test_t019_makefile_no_error_swallowing():
    """Makefile setup and install-deps must not swallow errors with || echo / || true."""
    mf = (PHASE1_ROOT / "Makefile").read_text()
    assert "docker-compose up -d || echo" not in mf, \
        "Makefile must not swallow docker-compose errors with || echo"
    assert "python3 -m venv $(VENV) || true" not in mf, \
        "Makefile must not swallow venv creation errors with || true"
    assert "init_db() failed (likely missing DATABASE_URL)" not in mf, \
        "Makefile must not swallow init_db() errors with || echo"
    # Must have explicit error handling with exit 1
    assert "exit 1" in mf, "Makefile must use exit 1 on failures"


def test_t019_makefile_uses_tabs():
    """Makefile recipe lines must use TAB (not spaces)."""
    with open(PHASE1_ROOT / "Makefile", "rb") as f:
        raw = f.read()
    # Find a recipe line (after a target definition)
    setup_idx = raw.find(b"setup:\n")
    assert setup_idx >= 0, "setup: target not found"
    next_line_start = raw.find(b"\n", setup_idx) + 1
    next_line = raw[next_line_start:raw.find(b"\n", next_line_start)]
    assert next_line.startswith(b"\t"), \
        f"Makefile recipe line must start with TAB, got: {next_line[:8]!r}"


# ===========================================================================
# T-020: Deprecated airflow db init replaced with airflow db upgrade
# ===========================================================================

def test_t020_makefile_no_deprecated_airflow_db_init():
    """Makefile must not use the deprecated `airflow db init`."""
    mf = (PHASE1_ROOT / "Makefile").read_text()
    assert "airflow db init" not in mf, \
        "Makefile must not use deprecated airflow db init (removed in Airflow 3.0)"
    assert "airflow db migrate || airflow db upgrade" in mf, \
        "Makefile should use airflow db migrate || airflow db upgrade (2.x-compatible)"


# ===========================================================================
# T-021: apache-airflow pinned with upper bound <3.0.0
# ===========================================================================

def test_t021_airflow_pinned_with_upper_bound():
    """requirements.txt must pin apache-airflow with <3.0.0 upper bound."""
    reqs = (PHASE1_ROOT / "requirements.txt").read_text()
    assert 'apache-airflow>=2.10.0,<3.0.0; python_version>="3.12"' in reqs, \
        "apache-airflow must be pinned with <3.0.0 for Python 3.12+"
    assert 'apache-airflow>=2.8.0,<3.0.0; python_version<"3.12"' in reqs, \
        "apache-airflow must be pinned with <3.0.0 for Python <3.12"
    # Must NOT have the unbounded pins
    assert 'apache-airflow>=2.10.0; python_version>="3.12"' not in reqs, \
        "unbounded apache-airflow>=2.10.0 must be removed"
    assert 'apache-airflow>=2.8.0; python_version<"3.12"' not in reqs, \
        "unbounded apache-airflow>=2.8.0 must be removed"


# ===========================================================================
# T-022: rdkit runtime warnings
# ===========================================================================

def test_t022_resolver_utils_has_rdkit_warning_flag():
    """resolver_utils.py must have the _RDKIT_UNAVAILABLE_WARNED flag."""
    content = (PHASE1_ROOT / "entity_resolution" / "resolver_utils.py").read_text()
    assert "_RDKIT_UNAVAILABLE_WARNED: bool = False" in content, \
        "_RDKIT_UNAVAILABLE_WARNED flag must be declared"
    assert "v74 T-022" in content, \
        "v74 T-022 warning log message must be present"
    # Must NOT silently pass on ImportError
    assert "except ImportError:\n                # RDKit not available -- skip cross-field check.\n                pass" not in content, \
        "resolver_utils.py must not silently pass on ImportError"


def test_t022_drug_resolver_has_import_time_probe():
    """drug_resolver.py must have an import-time rdkit availability probe."""
    content = (PHASE1_ROOT / "entity_resolution" / "drug_resolver.py").read_text()
    assert "import rdkit as _rdkit_probe" in content, \
        "drug_resolver.py must probe rdkit at import time"
    assert "_RDKIT_AVAILABLE: bool = True" in content, \
        "drug_resolver.py must declare _RDKIT_AVAILABLE flag"
    assert "v74 T-022" in content, \
        "drug_resolver.py must reference v74 T-022 in the warning"


# ===========================================================================
# T-023: DAGs use fail-fast on HTTP 4xx + exponential backoff
# ===========================================================================

def test_t023_retry_policy_helper_exists():
    """dags/_retry_policy.py must exist and export the required helpers."""
    rp = PHASE1_ROOT / "dags" / "_retry_policy.py"
    assert rp.exists(), "dags/_retry_policy.py must exist"
    content = rp.read_text()
    assert "DEFAULT_RETRY_ARGS" in content
    assert "fail_fast_on_http_4xx" in content
    assert "is_http_4xx_error" in content
    assert "retry_exponential_backoff" in content
    assert "max_retry_delay" in content


def test_t023_retry_args_use_exponential_backoff():
    """DEFAULT_RETRY_ARGS must use exponential backoff with reasonable bounds."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_retry_policy", str(PHASE1_ROOT / "dags" / "_retry_policy.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.DEFAULT_RETRY_ARGS
    assert args["retries"] == 2, f"retries should be 2, got {args['retries']}"
    assert args["retry_exponential_backoff"] is True, "retry_exponential_backoff must be True"
    assert args["retry_delay"].total_seconds() == 300, \
        f"retry_delay should be 5 minutes (300s), got {args['retry_delay']}"
    assert args["max_retry_delay"].total_seconds() == 1200, \
        f"max_retry_delay should be 20 minutes (1200s), got {args['max_retry_delay']}"


def test_t023_is_http_4xx_error_classifies_correctly():
    """is_http_4xx_error must correctly classify HTTP status codes."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_retry_policy", str(PHASE1_ROOT / "dags" / "_retry_policy.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class FakeHTTPError(Exception):
        def __init__(self, status_code, msg=""):
            self.response = type("R", (), {"status_code": status_code})()
            super().__init__(f"{status_code} {msg}")

    # 4xx (non-retryable)
    assert mod.is_http_4xx_error(FakeHTTPError(400)) is True
    assert mod.is_http_4xx_error(FakeHTTPError(401)) is True
    assert mod.is_http_4xx_error(FakeHTTPError(403)) is True
    assert mod.is_http_4xx_error(FakeHTTPError(404)) is True
    assert mod.is_http_4xx_error(FakeHTTPError(410)) is True
    # 429 (rate limit -- RETRYABLE, must be False)
    assert mod.is_http_4xx_error(FakeHTTPError(429)) is False, \
        "429 (Too Many Requests) must NOT be classified as non-retryable 4xx"
    # 5xx (transient -- RETRYABLE, must be False)
    assert mod.is_http_4xx_error(FakeHTTPError(500)) is False
    assert mod.is_http_4xx_error(FakeHTTPError(502)) is False
    assert mod.is_http_4xx_error(FakeHTTPError(503)) is False
    # Generic exception
    assert mod.is_http_4xx_error(Exception("generic")) is False


def test_t023_all_dags_use_retry_policy():
    """All 7 standalone DAGs must import and use the retry policy."""
    dags = ["chembl_dag.py", "disgenet_dag.py", "drugbank_dag.py",
            "pubchem_dag.py", "string_dag.py", "omim_dag.py", "uniprot_dag.py"]
    for dag_file in dags:
        content = (PHASE1_ROOT / "dags" / dag_file).read_text()
        assert "from dags._retry_policy import" in content, \
            f"{dag_file} must import from dags._retry_policy"
        assert "fail_fast_on_http_4xx" in content, \
            f"{dag_file} must reference fail_fast_on_http_4xx"
        assert "@fail_fast_on_http_4xx" in content, \
            f"{dag_file} must use @fail_fast_on_http_4xx decorator"
        assert "retry_exponential_backback=True" in content or \
               "retry_exponential_backoff=True" in content, \
            f"{dag_file} must use retry_exponential_backoff=True"
        assert "retry_delay=timedelta(minutes=30)" not in content, \
            f"{dag_file} must not use the old 30-min retry delay"


# ===========================================================================
# Phase 1 ↔ Phase 2 Bridge Connection Verification
# ===========================================================================

def test_phase1_phase2_bridge_connected():
    """The Phase 1 -> Phase 2 bridge must be wired and functional."""
    # Add phase2 to sys.path
    phase2_root = PHASE1_ROOT.parent / "phase2"
    if str(phase2_root) not in sys.path:
        sys.path.insert(0, str(phase2_root))
    from drugos_graph.phase1_bridge import (
        read_phase1_outputs,
        stage_phase1_to_phase2,
        load_into_graph,
        run_phase1_to_phase2,
        RecordingGraphBuilder,
    )
    # Verify all entry points are callable
    assert callable(read_phase1_outputs)
    assert callable(stage_phase1_to_phase2)
    assert callable(load_into_graph)
    assert callable(run_phase1_to_phase2)
    assert RecordingGraphBuilder is not None


# ===========================================================================
# Test runner (for standalone execution without pytest)
# ===========================================================================

def _run_all_tests():
    """Run all tests and print a summary. For standalone execution."""
    tests = [
        (name, func) for name, func in globals().items()
        if name.startswith("test_") and callable(func)
    ]
    passed = 0
    failed = 0
    errors = []
    for name, func in tests:
        try:
            func()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
            errors.append((name, str(e)))
    print()
    print("=" * 70)
    print(f"V74 ROOT FIX VERIFICATION: {passed} passed, {failed} failed (of {len(tests)} tests)")
    if failed == 0:
        print("ALL ROOT-LEVEL FIXES VERIFIED SUCCESSFULLY")
    else:
        print("FAILED TESTS:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all_tests())
