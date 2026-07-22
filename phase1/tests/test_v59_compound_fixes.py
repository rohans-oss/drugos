#!/usr/bin/env python3
"""
v59 ROOT FIX verification tests.

These tests verify the compound-issue fixes that v57/v58 left open:
  1. disease_id contradiction (compound of P1C-001 gene_symbol fix)
  2. ORM CheckConstraint regex operator ``~`` breaking SQLite create_all
  3. SQLite migration runner multi-statement execution
  4. SQLite translator handling PostgreSQL-specific syntax
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # phase1/


class TestV59DiseaseIdContradiction:
    """v59 ROOT FIX: disease_id DEFAULT '' + CHECK <> '' contradiction.

    The v57 fix removed the contradiction for gene_symbol but left the
    IDENTICAL pattern in place for disease_id. This is the compound
    issue the user reported as "GDA schema contradiction crashes
    DisGeNET/OMIM inserts on PostgreSQL".
    """

    def test_migration_001_no_disease_id_default_empty(self):
        """Migration 001 MUST NOT have `disease_id VARCHAR(50) NOT NULL DEFAULT ''`."""
        mig = PROJECT_ROOT / "database" / "migrations" / "001_initial_schema.sql"
        sql = mig.read_text()
        assert "disease_id      VARCHAR(50) NOT NULL DEFAULT ''" not in sql, (
            "v59 ROOT FIX INCOMPLETE: migration 001 still has "
            "`disease_id VARCHAR(50) NOT NULL DEFAULT ''` -- this "
            "contradicts the CHECK (disease_id <> '') constraint and "
            "crashes INSERTs on PostgreSQL."
        )

    def test_migration_001_disease_id_check_preserved(self):
        """The CHECK (disease_id <> '') constraint MUST be preserved."""
        mig = PROJECT_ROOT / "database" / "migrations" / "001_initial_schema.sql"
        sql = mig.read_text()
        assert "CHECK (disease_id <> '')" in sql, (
            "v59 ROOT FIX REGRESSION: CHECK (disease_id <> '') was "
            "removed -- disease_id must be non-empty (scientifically "
            "meaningless to associate a gene with an empty disease ID)."
        )

    def test_orm_models_no_disease_id_server_default(self):
        """ORM models.py MUST NOT have server_default='' for disease_id."""
        models = PROJECT_ROOT / "database" / "models.py"
        sql = models.read_text()
        # Find the disease_id column definition in GeneDiseaseAssociation
        gda_start = sql.index("class GeneDiseaseAssociation")
        gda_section = sql[gda_start:]
        m = re.search(
            r'disease_id:\s*Mapped\[Optional\[str\]\]\s*=\s*mapped_column\([^)]+\)',
            gda_section, re.DOTALL,
        )
        assert m, "disease_id column definition not found in GeneDiseaseAssociation"
        assert 'server_default=""' not in m.group(0), (
            f"v59 ROOT FIX INCOMPLETE: ORM disease_id still has "
            f"server_default=\"\" -- contradicts the CHECK constraint. "
            f"Definition: {m.group(0)[:200]}"
        )

    def test_loaders_no_disease_id_empty_string_assignment(self):
        """loaders.py MUST NOT set df['disease_id'] = '' when column missing."""
        loaders = PROJECT_ROOT / "database" / "loaders.py"
        sql = loaders.read_text()
        # The actual code line should be df["disease_id"] = None (not "")
        # The only remaining mention of df["disease_id"] = "" should be
        # in COMMENT blocks explaining the v59 fix.
        lines = sql.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            # Check for actual code that sets disease_id to ""
            if 'df["disease_id"] = ""' in stripped:
                pytest.fail(
                    f"v59 ROOT FIX INCOMPLETE: loaders.py line {i} still "
                    f"sets df['disease_id'] = '' -- this crashes PostgreSQL "
                    f"(CHECK constraint rejects empty string). Line: {line}"
                )

    def test_loaders_disease_id_none_replacement(self):
        """loaders.py MUST set df['disease_id'] = None (not '') when missing."""
        loaders = PROJECT_ROOT / "database" / "loaders.py"
        sql = loaders.read_text()
        assert 'df["disease_id"] = None' in sql, (
            "v59 ROOT FIX INCOMPLETE: loaders.py should set "
            "df['disease_id'] = None (and quarantine) when the column "
            "is missing -- mirrors the gene_symbol fix pattern."
        )


class TestV59SQLiteRegexFix:
    """v59 ROOT FIX: ORM CheckConstraint regex operator ``~`` breaks
    SQLite Base.metadata.create_all(). Replaced with portable forms."""

    def test_orm_no_regex_operator_in_check_constraints(self):
        """ORM models.py MUST NOT use PostgreSQL ``~`` regex operator in
        CheckConstraint expressions (breaks SQLite create_all)."""
        models = PROJECT_ROOT / "database" / "models.py"
        sql = models.read_text()
        # Find all CheckConstraint expressions
        for m in re.finditer(r'CheckConstraint\(\s*["\'](.*?)["\']', sql, re.DOTALL):
            expr = m.group(1)
            # The ``~`` operator is PostgreSQL-only and breaks SQLite
            assert " ~ " not in expr and " ~'" not in expr and "~ '" not in expr, (
                f"v59 ROOT FIX INCOMPLETE: CheckConstraint still uses "
                f"PostgreSQL regex operator ``~``: {expr[:100]}. "
                f"This breaks SQLite Base.metadata.create_all()."
            )


class TestV59MigrationRunner:
    """v59 ROOT FIX: SQLite migration runner executes statements
    individually (not as one multi-statement call)."""

    def test_runner_splits_statements_for_sqlite(self):
        """The migration runner MUST split SQL into individual statements
        for SQLite (SQLite's execute() only allows one statement at a time)."""
        runner = PROJECT_ROOT / "database" / "migrations" / "run_migrations.py"
        sql = runner.read_text()
        assert "_split_sql_statements" in sql, (
            "v59 ROOT FIX INCOMPLETE: migration runner should use "
            "_split_sql_statements to split SQL for SQLite."
        )

    def test_runner_has_statement_execution_error_class(self):
        """The runner SHOULD have a StatementExecutionError class for
        per-statement error context."""
        runner = PROJECT_ROOT / "database" / "migrations" / "run_migrations.py"
        sql = runner.read_text()
        assert "class StatementExecutionError" in sql, (
            "v59 ROOT FIX INCOMPLETE: StatementExecutionError class "
            "not found -- needed for per-statement error reporting."
        )


class TestV59EnvExampleStringScore:
    """v59 ROOT FIX: .env.example STRING_MIN_COMBINED_SCORE=700 (not 400)."""

    def test_env_example_string_score_700(self):
        """Both .env.example files MUST ship STRING_MIN_COMBINED_SCORE=700."""
        for env_path in [
            PROJECT_ROOT / "config" / ".env.example",
            PROJECT_ROOT.parent / "phase2" / "drugos_graph" / ".env.example",
        ]:
            if not env_path.exists():
                continue
            content = env_path.read_text()
            # Find the STRING_MIN_COMBINED_SCORE line
            for line in content.split("\n"):
                if "STRING_MIN_COMBINED_SCORE=" in line and not line.strip().startswith("#"):
                    assert "700" in line, (
                        f"v59 ROOT FIX INCOMPLETE: {env_path} has "
                        f"STRING_MIN_COMBINED_SCORE != 700. Line: {line}"
                    )


class TestV59AirflowCatchup:
    """v59 ROOT FIX: All Airflow DAGs MUST have catchup=False."""

    def test_all_dags_have_catchup_false(self):
        """Every DAG file MUST have catchup=False to prevent 7×N backfill
        runs on first deploy."""
        dags_dir = PROJECT_ROOT / "dags"
        for dag_file in dags_dir.glob("*_dag.py"):
            content = dag_file.read_text()
            assert "catchup=False" in content, (
                f"v59 ROOT FIX INCOMPLETE: {dag_file.name} does not have "
                f"catchup=False -- daily backfill would cause 7×N runs on "
                f"first deploy."
            )


class TestV59ChemblV50Filename:
    """v59 ROOT FIX: ChEMBL v50 downloader writes chembl_activities.csv.gz
    (not chembl_activities_clean.csv) so DPI edge generation fires."""

    def test_chembl_v50_writes_csv_gz(self):
        """The ChEMBL pipeline MUST write/read chembl_activities.csv.gz
        (the filename clean() looks for) so DPI edge generation fires."""
        # The filename is in chembl_pipeline.py (not _v50_downloaders.py
        # which writes raw JSONL). The pipeline's clean() step reads
        # chembl_activities.csv.gz and writes chembl_activities_clean.csv.
        pipeline = PROJECT_ROOT / "pipelines" / "chembl_pipeline.py"
        content = pipeline.read_text()
        assert 'chembl_activities.csv.gz' in content, (
            "v59 ROOT FIX INCOMPLETE: chembl_pipeline.py does not reference "
            "chembl_activities.csv.gz -- DPI edge generation never fires."
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
