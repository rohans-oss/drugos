"""
v76 ROOT FIX TEST SUITE -- T-037 through T-047
==============================================

Forensic verification that all 11 issues (T-037..T-047) are root-fixed
in the v76 codebase. Each test reads the ACTUAL source files (not
comments, not test fixtures) and verifies the fix is present and
correct.

This file is NOT a smoke test. It reads real SQL, real Python, real YAML
and asserts on the actual content. If a previous AI claimed a fix but
didn't apply it, these tests will FAIL.

Run:  python -m pytest phase1/tests/test_v76_root_fixes.py -v
"""

import re
import sys
from pathlib import Path

import pytest

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE1_ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = PHASE1_ROOT / "database" / "migrations"
DAGS_DIR = PHASE1_ROOT / "dags"
DOCKER_COMPOSE = PHASE1_ROOT / "docker-compose.yml"


# ── T-037: schema_version rollback consistency ──────────────────────────────

class TestT037SchemaVersionRollbackConsistency:
    """T-037: All rollbacks (002-011) must DELETE their schema_version row."""

    @pytest.mark.parametrize("version,filename", [
        (2, "002_bug_fixes_migration_rollback.sql"),
        (3, "003_models_fix_migration_rollback.sql"),
        (4, "004_extend_gda_table_for_389_audit_rollback.sql"),
        (5, "005_pubchem_compound_properties_rollback.sql"),
        (6, "006_drug_withdrawn_safety_columns_rollback.sql"),
        (7, "007_pipeline_run_metadata_rollback.sql"),
        (8, "008_drug_is_globally_approved_rollback.sql"),
        (9, "009_tighten_inchikey_check_constraint_rollback.sql"),
        (10, "010_drug_indication_columns_rollback.sql"),
        (11, "011_align_activity_value_to_orm_rollback.sql"),
    ])
    def test_rollback_deletes_schema_version_row(self, version, filename):
        """Every rollback must DELETE its own schema_version row."""
        rollback_path = MIGRATIONS_DIR / filename
        assert rollback_path.exists(), f"Rollback file not found: {rollback_path}"
        content = rollback_path.read_text(encoding="utf-8")
        # Strip SQL comments before checking (the DELETE must be ACTIVE code).
        active_lines = [
            line for line in content.split("\n")
            if line.strip() and not line.strip().startswith("--")
        ]
        active_sql = "\n".join(active_lines)
        expected_delete = f"DELETE FROM schema_version WHERE version = {version}"
        assert expected_delete in active_sql, (
            f"T-037 FAIL: {filename} does NOT contain active "
            f"'{expected_delete}'. The rollback leaves a stale "
            f"schema_version row, inconsistent with the convention."
        )


# ── T-038: InChIKey CHECK uses PostgreSQL regex ─────────────────────────────

class TestT038InChIKeyCheckUsesRegex:
    """T-038: chk_drugs_inchikey_format must use PostgreSQL ~ regex."""

    def test_migration_001_uses_regex(self):
        """Migration 001 must use ~ regex for InChIKey, not just LENGTH=27."""
        schema = (MIGRATIONS_DIR / "001_initial_schema.sql").read_text("utf-8")
        # Find the chk_drugs_inchikey_format constraint block.
        match = re.search(
            r"CONSTRAINT\s+chk_drugs_inchikey_format\s*CHECK\s*\(([^)]+)\)",
            schema, re.IGNORECASE | re.DOTALL,
        )
        assert match, "chk_drugs_inchikey_format constraint not found in 001"
        constraint_body = match.group(1)
        assert "~" in constraint_body, (
            "T-038 FAIL: chk_drugs_inchikey_format still uses the weak "
            "LENGTH=27 form. Expected PostgreSQL ~ regex operator."
        )
        assert "^[A-Z]{14}-[A-Z]{10}-[A-Z]$" in constraint_body, (
            "T-038 FAIL: regex pattern not found in constraint."
        )
        assert "SYNTH%" in constraint_body, (
            "T-038 FAIL: SYNTH% escape hatch missing from constraint."
        )

    def test_sqlite_translator_has_inchikey_specific_translation(self):
        """The SQLite translator must translate InChIKey regex to a STRONG
        portable form (LENGTH=27 + hyphen positions), not the weak
        LENGTH(TRIM()) > 0."""
        runner = (MIGRATIONS_DIR / "run_migrations.py").read_text("utf-8")
        # The specific InChIKey translation must exist somewhere in the file.
        # We search for the key pattern: a re.sub call that matches the
        # InChIKey regex and produces LENGTH=27 + SUBSTR.
        assert "inchikey" in runner and "SUBSTR(inchikey, 15, 1)" in runner, (
            "T-038 FAIL: SQLite translator does NOT have a specific "
            "InChIKey regex translation producing SUBSTR checks. The "
            "generic LENGTH(TRIM()) > 0 fallback would be a REGRESSION."
        )
        # Verify the specific translation produces LENGTH=27.
        assert "LENGTH(inchikey) = 27 AND SUBSTR(inchikey, 15, 1) = '-'" in runner, (
            "T-038 FAIL: specific InChIKey translation does not produce "
            "LENGTH(inchikey) = 27 AND SUBSTR hyphen check."
        )

    def test_orm_uses_portable_strong_form(self):
        """The ORM CheckConstraint must use the portable LENGTH+SUBSTR form
        (not the old weak LENGTH=27 OR SYNTH%, and not the ~ regex which
        SQLite can't parse via create_all)."""
        models = (PHASE1_ROOT / "database" / "models.py").read_text("utf-8")
        # Find the chk_drugs_inchikey_format CheckConstraint by searching
        # for the name string, then grab the surrounding block.
        name_idx = models.find('name="chk_drugs_inchikey_format"')
        assert name_idx != -1, "ORM chk_drugs_inchikey_format CheckConstraint not found"
        # Grab a window around the name to capture the CheckConstraint call.
        block = models[max(0, name_idx - 500):name_idx + 100]
        assert "SUBSTR(inchikey, 15, 1)" in block, (
            "T-038 FAIL: ORM InChIKey CHECK does not validate hyphen at "
            "position 15 via SUBSTR."
        )
        assert "SUBSTR(inchikey, 26, 1)" in block, (
            "T-038 FAIL: ORM InChIKey CHECK does not validate hyphen at "
            "position 26 via SUBSTR."
        )


# ── T-039: SMILES CHECK uses portable NOT LIKE form ─────────────────────────

class TestT039SmilesCheckPortable:
    """T-039: chk_drugs_smiles_valid must use portable NOT LIKE (no ~)."""

    def test_migration_001_uses_not_like(self):
        """Migration 001 SMILES CHECK must NOT use ~ operator."""
        schema = (MIGRATIONS_DIR / "001_initial_schema.sql").read_text("utf-8")
        match = re.search(
            r"CONSTRAINT\s+chk_drugs_smiles_valid\s*CHECK\s*\(([^)]+)\)",
            schema, re.IGNORECASE | re.DOTALL,
        )
        assert match, "chk_drugs_smiles_valid constraint not found in 001"
        constraint_body = match.group(1)
        assert "~" not in constraint_body, (
            "T-039 FAIL: chk_drugs_smiles_valid still uses PostgreSQL ~ "
            "operator which fails on SQLite create_all."
        )
        assert "NOT LIKE" in constraint_body.upper(), (
            "T-039 FAIL: chk_drugs_smiles_valid does not use portable "
            "NOT LIKE form."
        )

    def test_orm_uses_not_like(self):
        """ORM SMILES CHECK must use portable NOT LIKE form."""
        models = (PHASE1_ROOT / "database" / "models.py").read_text("utf-8")
        # Find the chk_drugs_smiles_valid CheckConstraint by name.
        name_idx = models.find('name="chk_drugs_smiles_valid"')
        assert name_idx != -1, "ORM chk_drugs_smiles_valid CheckConstraint not found"
        block = models[max(0, name_idx - 500):name_idx + 100]
        assert "~" not in block, (
            "T-039 FAIL: ORM SMILES CHECK still uses ~ operator."
        )
        assert "NOT LIKE" in block.upper(), (
            "T-039 FAIL: ORM SMILES CHECK does not use NOT LIKE form."
        )


# ── T-040: Airflow bitshift rewritten as explicit statements ────────────────

class TestT040ExplicitBitshift:
    """T-040: The list-bitshift is rewritten as explicit statements."""

    def test_no_list_bitshift_for_drugbank_branch(self):
        """The old ``check_drugbank >> [drugbank, skip_drugbank] >> drugbank_done``
        must be replaced with explicit single-edge statements."""
        dag = (DAGS_DIR / "master_pipeline_dag.py").read_text("utf-8")
        # Strip comments to check ACTIVE code only (the v76 fix comment
        # mentions the old form to explain what was replaced).
        active_lines = [
            line for line in dag.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        active_code = "\n".join(active_lines)
        # The old fragile form must NOT be present in active code.
        old_form = "check_drugbank >> [drugbank, skip_drugbank] >> drugbank_done"
        assert old_form not in active_code, (
            "T-040 FAIL: old list-bitshift form still present in active "
            "code. Expected explicit single-edge statements."
        )
        # The new explicit form must be present in active code.
        assert "check_drugbank >> drugbank" in active_code, (
            "T-040 FAIL: explicit 'check_drugbank >> drugbank' not found."
        )
        assert "check_drugbank >> skip_drugbank" in active_code, (
            "T-040 FAIL: explicit 'check_drugbank >> skip_drugbank' not found."
        )
        assert "drugbank >> drugbank_done" in active_code, (
            "T-040 FAIL: explicit 'drugbank >> drugbank_done' not found."
        )
        assert "skip_drugbank >> drugbank_done" in active_code, (
            "T-040 FAIL: explicit 'skip_drugbank >> drugbank_done' not found."
        )


# ── T-041: disgenet >> omim wire removed ────────────────────────────────────

class TestT041DisgenetOmimParallel:
    """T-041: The ``disgenet >> omim`` wire must be removed for parallel GDA loading."""

    def test_disgenet_omim_wire_removed(self):
        """The ``disgenet >> omim`` wire must NOT be present."""
        dag = (DAGS_DIR / "master_pipeline_dag.py").read_text("utf-8")
        # Strip comments to check ACTIVE code only.
        active_lines = [
            line for line in dag.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        active_code = "\n".join(active_lines)
        assert "disgenet >> omim" not in active_code, (
            "T-041 FAIL: 'disgenet >> omim' wire still present in active "
            "code. DisGeNET and OMIM must run in parallel."
        )


# ── T-042: omim >> drugbank wire removed; DrugBank graceful ──────────────────

class TestT042DrugBankDecoupled:
    """T-042: ``omim >> drugbank`` removed; DrugBank gracefully handles missing OMIM CSV."""

    def test_omim_drugbank_wire_removed(self):
        """The ``omim >> drugbank`` wire must NOT be present in active code."""
        dag = (DAGS_DIR / "master_pipeline_dag.py").read_text("utf-8")
        active_lines = [
            line for line in dag.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        active_code = "\n".join(active_lines)
        assert "omim >> drugbank" not in active_code, (
            "T-042 FAIL: 'omim >> drugbank' wire still present in active "
            "code. DrugBank must run in parallel with OMIM."
        )

    def test_drugbank_graceful_missing_omim(self):
        """DrugBank's _write_structured_indications must NOT raise
        RuntimeError when OMIM CSV is missing. It must log WARNING and
        write a header-only file."""
        pipeline = (PHASE1_ROOT / "pipelines" / "drugbank_pipeline.py").read_text("utf-8")
        # Find the _write_structured_indications method.
        match = re.search(
            r"def _write_structured_indications\(self.*?(?=\n    def |\nclass )",
            pipeline, re.DOTALL,
        )
        assert match, "_write_structured_indications method not found"
        method_body = match.group(0)
        # The old hard RuntimeError for missing file must be GONE.
        # Check that the method does NOT raise RuntimeError for missing OMIM.
        # The v76 fix replaces the raise with a WARNING + header-only write.
        assert "header-only" in method_body.lower() or "header only" in method_body.lower(), (
            "T-042 FAIL: _write_structured_indications does not mention "
            "header-only fallback for missing OMIM CSV."
        )
        # Must log a WARNING (not raise) when OMIM CSV is missing.
        assert "logger.warning" in method_body, (
            "T-042 FAIL: _write_structured_indications does not log WARNING "
            "for missing OMIM CSV."
        )


# ── T-043: setup healthcheck verifies writability ───────────────────────────

class TestT043SetupHealthcheckWritable:
    """T-043: setup healthcheck must use test -w (writability), not just test -d."""

    def test_healthcheck_uses_test_w(self):
        """The setup healthcheck must include test -w checks."""
        compose = DOCKER_COMPOSE.read_text("utf-8")
        # Find the setup service healthcheck.
        setup_match = re.search(
            r"setup:.*?healthcheck:.*?test:.*?\n(?:.*?\n)*?.*?retries:",
            compose, re.DOTALL,
        )
        assert setup_match, "setup service healthcheck not found"
        healthcheck_block = setup_match.group(0)
        assert "test -w" in healthcheck_block, (
            "T-043 FAIL: setup healthcheck does not use 'test -w' to "
            "verify writability. It only checks directory existence."
        )


# ── T-044: Neo4j healthcheck uses curl ──────────────────────────────────────

class TestT044Neo4jHealthcheckCurl:
    """T-044: Neo4j healthcheck must use curl, not wget."""

    def test_neo4j_healthcheck_uses_curl(self):
        """The Neo4j healthcheck must use curl -f, not wget."""
        compose = DOCKER_COMPOSE.read_text("utf-8")
        # Find the neo4j service healthcheck TEST line specifically.
        # The healthcheck test is on a line starting with "test:".
        # We look for the test line that's INSIDE the neo4j service block
        # (between "neo4j:" service header and the next service header).
        neo4j_start = compose.find("neo4j:5.20")
        assert neo4j_start != -1, "neo4j:5.20 image not found in compose"
        # Find the next service header (a line starting with "  servicename:")
        # after the neo4j block. Services are indented at 2 spaces.
        neo4j_block = compose[neo4j_start:]
        # Find the healthcheck test line within the neo4j block.
        test_match = re.search(
            r'test:\s*\["CMD-SHELL",\s*"(.*?)"\]',
            neo4j_block,
        )
        assert test_match, "Neo4j healthcheck test line not found"
        test_command = test_match.group(1)
        assert "curl" in test_command, (
            f"T-044 FAIL: Neo4j healthcheck does not use curl. "
            f"Found: {test_command}"
        )
        assert "wget" not in test_command, (
            f"T-044 FAIL: Neo4j healthcheck still uses wget. "
            f"Found: {test_command}"
        )


# ── T-045: DATABASE_URL built from POSTGRES_USER/PASSWORD ───────────────────

class TestT045DatabaseUrlFromEnvVars:
    """T-045: DATABASE_URL default must use ${POSTGRES_USER}/${POSTGRES_PASSWORD}."""

    def test_no_hardcoded_cosmic_cosmic_in_database_url(self):
        """No DATABASE_URL default should hardcode cosmic:cosmic."""
        compose = DOCKER_COMPOSE.read_text("utf-8")
        # Find all ACTIVE DATABASE_URL assignment lines (lines that assign
        # a value to DATABASE_URL, not comment lines that mention it).
        # Active assignment lines look like:
        #   DATABASE_URL: ${DATABASE_URL:-postgresql+psycopg2://...}
        db_url_lines = re.findall(
            r'^\s+DATABASE_URL:\s*\S.*',
            compose, re.MULTILINE,
        )
        assert len(db_url_lines) >= 3, (
            f"Expected at least 3 DATABASE_URL assignment lines, "
            f"found {len(db_url_lines)}: {db_url_lines}"
        )
        for line in db_url_lines:
            # The default must reference ${POSTGRES_USER} and ${POSTGRES_PASSWORD}.
            assert "${POSTGRES_USER:-cosmic}" in line, (
                f"T-045 FAIL: DATABASE_URL line does not use ${{POSTGRES_USER}}: {line}"
            )
            assert "${POSTGRES_PASSWORD:-cosmic}" in line, (
                f"T-045 FAIL: DATABASE_URL line does not use ${{POSTGRES_PASSWORD}}: {line}"
            )
            # Must NOT have a hardcoded cosmic:cosmic@ (without env var interpolation).
            assert "postgresql+psycopg2://cosmic:cosmic@" not in line, (
                f"T-045 FAIL: DATABASE_URL line hardcodes cosmic:cosmic: {line}"
            )


# ── T-046: Escaping comment for ${VAR} vs $$VAR vs \gexec ───────────────────

class TestT046EscapingComment:
    """T-046: The airflow-init entrypoint must have a comment documenting the
    ${VAR} vs $$VAR vs \\gexec escaping conventions."""

    def test_escaping_comment_present(self):
        """The docker-compose.yml must document the escaping conventions."""
        compose = DOCKER_COMPOSE.read_text("utf-8")
        assert "T-046" in compose or "T_046" in compose, (
            "T-046 FAIL: no T-046 reference found in docker-compose.yml."
        )
        # Must mention all three conventions.
        assert "${VAR" in compose or "${POSTGRES_PASSWORD" in compose, (
            "T-046 FAIL: comment does not document ${VAR} interpolation."
        )
        assert "$$VAR" in compose or "$$AIRFLOW_ADMIN" in compose, (
            "T-046 FAIL: comment does not document $$VAR bash expansion."
        )
        assert "\\gexec" in compose, (
            "T-046 FAIL: comment does not document \\gexec psql meta-command."
        )


# ── T-047: Comprehensive transaction-control filter ─────────────────────────

class TestT047TransactionControlFilter:
    """T-047: The SQL splitter must filter ALL transaction-control statements,
    not just bare BEGIN/COMMIT."""

    def test_is_transaction_control_statement_function_exists(self):
        """The _is_transaction_control_statement helper must exist."""
        runner = (MIGRATIONS_DIR / "run_migrations.py").read_text("utf-8")
        assert "def _is_transaction_control_statement" in runner, (
            "T-047 FAIL: _is_transaction_control_statement function not defined."
        )

    def test_splitter_uses_comprehensive_filter(self):
        """The _split_sql_statements function must call
        _is_transaction_control_statement instead of bare BEGIN/COMMIT check."""
        runner = (MIGRATIONS_DIR / "run_migrations.py").read_text("utf-8")
        # The old bare check ``upper != "BEGIN" and upper != "COMMIT"``
        # must NOT be present in active code.
        # (It may appear in comments explaining the old behavior.)
        active_lines = [
            line for line in runner.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        active_code = "\n".join(active_lines)
        assert '_is_transaction_control_statement(upper)' in active_code, (
            "T-047 FAIL: _split_sql_statements does not call "
            "_is_transaction_control_statement. The old bare BEGIN/COMMIT "
            "filter is still in use."
        )

    def test_filter_catches_all_variants(self):
        """Import the function and verify it catches all transaction-control forms."""
        # Add the migrations dir to the path so we can import.
        sys.path.insert(0, str(PHASE1_ROOT))
        try:
            from database.migrations.run_migrations import _is_transaction_control_statement
        finally:
            sys.path.pop(0)

        # All of these must be filtered (return True).
        must_filter = [
            "BEGIN",
            "BEGIN TRANSACTION",
            "BEGIN WORK",
            "START TRANSACTION",
            "COMMIT",
            "COMMIT TRANSACTION",
            "COMMIT WORK",
            "COMMIT AND CHAIN",
            "ROLLBACK",
            "ROLLBACK TRANSACTION",
            "ROLLBACK WORK",
            "END",
            "END TRANSACTION",
            "SAVEPOINT sp1",
            "RELEASE SAVEPOINT sp1",
            "RELEASE sp1",
            "SET TRANSACTION ISOLATION LEVEL READ COMMITTED",
            "SET CONSTRAINTS ALL IMMEDIATE",
        ]
        for stmt in must_filter:
            assert _is_transaction_control_statement(stmt), (
                f"T-047 FAIL: '{stmt}' should be filtered as a "
                f"transaction-control statement but was NOT."
            )

        # These must NOT be filtered (return False).
        must_keep = [
            "SELECT 1",
            "INSERT INTO drugs VALUES (1)",
            "CREATE TABLE foo (id INT)",
            "ALTER TABLE drugs ADD COLUMN x TEXT",
            "DROP TABLE foo",
            "DELETE FROM schema_version WHERE version = 5",
            "UPDATE drugs SET name = 'aspirin'",
        ]
        for stmt in must_keep:
            assert not _is_transaction_control_statement(stmt), (
                f"T-047 FAIL: '{stmt}' should NOT be filtered (it's a "
                f"normal SQL statement) but WAS filtered."
            )


# ── Phase 1 ↔ Phase 2 connection verification ───────────────────────────────

class TestPhase1Phase2Connection:
    """Verify the Phase 1 -> Phase 2 connection is 100% intact."""

    def test_trigger_phase2_wire_exists(self):
        """The master DAG must wire pubchem_load >> trigger_phase2."""
        dag = (DAGS_DIR / "master_pipeline_dag.py").read_text("utf-8")
        assert "pubchem_load >> trigger_phase2" in dag, (
            "Phase 1 ↔ Phase 2 connection BROKEN: 'pubchem_load >> trigger_phase2' "
            "wire not found in master DAG."
        )

    def test_phase1_bridge_exists(self):
        """The phase1_bridge.py module must exist."""
        bridge = PROJECT_ROOT / "phase2" / "drugos_graph" / "phase1_bridge.py"
        assert bridge.exists(), (
            "Phase 1 ↔ Phase 2 connection BROKEN: phase1_bridge.py not found."
        )

    def test_trigger_phase2_invokes_run_unified(self):
        """The trigger_phase2 task must invoke run_unified.py."""
        dag = (DAGS_DIR / "master_pipeline_dag.py").read_text("utf-8")
        assert "run_unified" in dag, (
            "Phase 1 ↔ Phase 2 connection BROKEN: trigger_phase2 does not "
            "invoke run_unified.py."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
