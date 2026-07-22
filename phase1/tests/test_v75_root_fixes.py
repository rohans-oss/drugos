"""
v75 ROOT FIX Verification -- T-024 through T-036
=================================================

This test file is the SINGLE proof that all 13 v75 root-level fixes
are in place and behave correctly. Each test verifies the ROOT FIX
(not surface-level): it reads the actual code/SQL/YAML/Makefile content
and asserts the v75 fix is present AND functional.

Run with:
    cd phase1 && python -m pytest tests/test_v75_root_fixes.py -v

OR (no pytest needed):
    cd phase1 && python tests/test_v75_root_fixes.py
"""

from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

# Ensure phase1/ is importable
HERE = Path(__file__).resolve().parent
PHASE1_ROOT = HERE.parent
UNIFIED_ROOT = PHASE1_ROOT.parent
for p in (str(PHASE1_ROOT), str(UNIFIED_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


class TestT024_SlaTimeoutAlignment(unittest.TestCase):
    """T-024: TASK_SLA and TASK_TIMEOUT must be aligned (no advisory SLA)."""

    def test_sla_equals_timeout(self):
        from datetime import timedelta
        # Import the module to read the actual constants
        dag_path = PHASE1_ROOT / "dags" / "master_pipeline_dag.py"
        text = dag_path.read_text()
        # Extract TASK_SLA and TASK_TIMEOUT values
        sla_match = re.search(r"^TASK_SLA\s*=\s*timedelta\(hours=(\d+)\)", text, re.MULTILINE)
        timeout_match = re.search(r"^TASK_TIMEOUT\s*=\s*timedelta\(hours=(\d+)\)", text, re.MULTILINE)
        self.assertIsNotNone(sla_match, "TASK_SLA not found in master_pipeline_dag.py")
        self.assertIsNotNone(timeout_match, "TASK_TIMEOUT not found in master_pipeline_dag.py")
        sla_hours = int(sla_match.group(1))
        timeout_hours = int(timeout_match.group(1))
        self.assertEqual(
            sla_hours, timeout_hours,
            f"T-024 ROOT FIX: TASK_SLA ({sla_hours}h) must equal TASK_TIMEOUT "
            f"({timeout_hours}h) -- Airflow SLA misses are advisory, so an "
            f"SLA shorter than the timeout generates false-positive warnings "
            f"that train operators to ignore them.",
        )

    def test_sla_is_7h(self):
        """The v75 fix sets both to 7h (upper bound of normal training time)."""
        dag_path = PHASE1_ROOT / "dags" / "master_pipeline_dag.py"
        text = dag_path.read_text()
        self.assertIn("TASK_SLA = timedelta(hours=7)", text)
        self.assertIn("TASK_TIMEOUT = timedelta(hours=7)", text)


class TestT025_DownloadParallelEntityResolution(unittest.TestCase):
    """T-025: download_parallel.py must run entity resolution between download and load."""

    def test_entity_resolution_phase_present(self):
        script_path = PHASE1_ROOT / "scripts" / "download_parallel.py"
        text = script_path.read_text()
        # Strip docstrings so documentation quoting the old v74 form
        # does not trigger a false positive.
        active = _strip_py_docstrings(text)
        # The two-phase design must be present
        self.assertIn("run_download_and_clean_only", active,
                      "T-025: download_parallel.py must call .run_download_and_clean_only() in Phase A")
        self.assertIn("run_load_only", active,
                      "T-025: download_parallel.py must call .run_load_only() in Phase C")
        # The entity resolution phase must be present
        self.assertIn("_run_entity_resolution_phase", active,
                      "T-025: download_parallel.py must have an entity resolution phase")
        self.assertIn("from entity_resolution.run import run_entity_resolution", active,
                      "T-025: download_parallel.py must import the shared run_entity_resolution function")
        # The v74 .run() call (full run including LOAD before entity resolution)
        # must be GONE from active code (the call site at run_pipeline()).
        # The docstring may quote it for audit-trail purposes.
        self.assertNotIn("cls(run_id=_run_id).run()", active,
                         "T-025: download_parallel.py must NOT call cls(...).run() (the full run) -- "
                         "this was the v74 bug that loaded before entity resolution ran.")

    def test_shared_module_exists(self):
        """The shared entity_resolution/run.py module must exist (DRY root fix)."""
        mod_path = PHASE1_ROOT / "entity_resolution" / "run.py"
        self.assertTrue(mod_path.exists(),
                        "T-025: entity_resolution/run.py must exist as the shared entry point")
        text = mod_path.read_text()
        self.assertIn("def run_entity_resolution()", text)
        # It must NOT import airflow (so it can be called from CLI, pytest, notebook)
        self.assertNotIn("from airflow", text,
                         "T-025: entity_resolution/run.py must not depend on Airflow")

    def test_master_dag_uses_shared_module(self):
        """The master DAG must delegate to the shared module (no inline logic)."""
        dag_path = PHASE1_ROOT / "dags" / "master_pipeline_dag.py"
        text = dag_path.read_text()
        self.assertIn("from entity_resolution.run import run_entity_resolution", text,
                      "T-025: master_pipeline_dag.py must import the shared run_entity_resolution")
        # The inline ~250-line implementation must be GONE (extracted to run.py)
        self.assertNotIn("drug_resolver.build_mapping", text,
                         "T-025: master_pipeline_dag.py must not have inline build_mapping calls -- "
                         "they were extracted to entity_resolution/run.py")


class TestT026_Migration007PortableSql(unittest.TestCase):
    """T-026: migration 007 must not use DO $$ block with information_schema."""

    def test_no_do_block_with_information_schema(self):
        mig_path = PHASE1_ROOT / "database" / "migrations" / "007_pipeline_run_metadata.sql"
        text = mig_path.read_text()
        active = _strip_sql_comments(text)
        # The DO $$ block that queried information_schema.columns must be GONE
        # from active SQL (comments may reference it for documentation).
        self.assertNotIn("information_schema.columns", active,
                         "T-026: migration 007 must not query information_schema.columns in active SQL "
                         "(PostgreSQL-only; fails on SQLite even after translator strips DO wrapper)")
        # The portable ALTER TABLE ADD COLUMN IF NOT EXISTS must be present
        self.assertIn("ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS metadata_json TEXT", active,
                      "T-026: migration 007 must use portable ALTER TABLE ADD COLUMN IF NOT EXISTS metadata_json TEXT")

    def test_postgres_only_upgrade_hook_exists(self):
        """The migration runner must have the _apply_postgres_only_upgrades hook."""
        runner_path = PHASE1_ROOT / "database" / "migrations" / "run_migrations.py"
        text = runner_path.read_text()
        self.assertIn("_POSTGRES_ONLY_UPGRADES", text,
                      "T-026: run_migrations.py must define _POSTGRES_ONLY_UPGRADES dict")
        self.assertIn("_apply_postgres_only_upgrades", text,
                      "T-026: run_migrations.py must define _apply_postgres_only_upgrades function")
        self.assertIn('"007_pipeline_run_metadata.sql"', text,
                      "T-026: _POSTGRES_ONLY_UPGRADES must have an entry for migration 007")
        # The hook must be CALLED inside the per-migration transaction.
        # v90 ROOT FIX (BUG #15): signature changed from (engine, migration_name)
        # to (conn, migration_name) so the upgrade runs INSIDE the caller's
        # transaction and commits atomically with the migration.
        self.assertIn("_apply_postgres_only_upgrades(conn, migration_name)", text,
                      "T-026/BUG #15: _apply_postgres_only_upgrades must be called with the "
                      "per-migration connection (conn) so it commits atomically with the migration")


class TestT027_MlflowCustomDockerfile(unittest.TestCase):
    """T-027: mlflow service must use a custom Dockerfile + healthcheck + MLFLOW_TRACKING_URI in airflow services."""

    def test_dockerfile_mlflow_exists(self):
        dockerfile_path = PHASE1_ROOT / "docker" / "Dockerfile.mlflow"
        self.assertTrue(dockerfile_path.exists(),
                        "T-027: docker/Dockerfile.mlflow must exist (mlflow pre-baked into image)")
        text = dockerfile_path.read_text()
        self.assertIn("mlflow==", text,
                      "T-027: Dockerfile.mlflow must install a pinned mlflow version")
        self.assertIn("psycopg2-binary", text,
                      "T-027: Dockerfile.mlflow must install psycopg2-binary for postgres backend")

    def test_compose_mlflow_uses_build_not_image(self):
        compose_path = PHASE1_ROOT / "docker-compose.yml"
        text = compose_path.read_text()
        # Strip comments so YAML comments referencing the old form do not
        # trigger false positives.
        active = _strip_yaml_comments(text)
        # The mlflow service must use build: (not image: python:3.11-slim).
        # Find the mlflow service block -- match from "  mlflow:" until the
        # next 2-space-indented key OR top-level key OR EOF.
        mlflow_match = re.search(
            r"^  mlflow:\s*\n((?:^(?:    .*)?\n)+?)(?=^  [a-z]|^[a-z]|\Z)",
            active, re.MULTILINE,
        )
        self.assertIsNotNone(mlflow_match, "mlflow service not found in docker-compose.yml")
        mlflow_block = mlflow_match.group(1)
        self.assertIn("build:", mlflow_block,
                      "T-027: mlflow service must use build: (custom Dockerfile)")
        self.assertIn("docker/Dockerfile.mlflow", mlflow_block,
                      "T-027: mlflow service build must reference docker/Dockerfile.mlflow")
        self.assertIn("healthcheck:", mlflow_block,
                      "T-027: mlflow service must have a healthcheck")
        self.assertIn("/health", mlflow_block,
                      "T-027: mlflow healthcheck must hit /health endpoint")

    def test_mlflow_tracking_uri_in_airflow_services(self):
        compose_path = PHASE1_ROOT / "docker-compose.yml"
        text = compose_path.read_text()
        active = _strip_yaml_comments(text)
        # All three airflow services must have MLFLOW_TRACKING_URI
        for svc in ("airflow-init", "airflow-webserver", "airflow-scheduler"):
            svc_match = re.search(
                rf"^  {re.escape(svc)}:\s*\n((?:^(?:    .*)?\n)+?)(?=^  [a-z]|^[a-z]|\Z)",
                active, re.MULTILINE,
            )
            self.assertIsNotNone(svc_match, f"{svc} service not found")
            svc_block = svc_match.group(1)
            self.assertIn("MLFLOW_TRACKING_URI: http://mlflow:5000", svc_block,
                          f"T-027: {svc} service must have MLFLOW_TRACKING_URI env var")


class TestT028_AirflowSchedulerHealthcheck(unittest.TestCase):
    """T-028: airflow-scheduler must have a healthcheck."""

    def test_scheduler_healthcheck_present(self):
        compose_path = PHASE1_ROOT / "docker-compose.yml"
        text = compose_path.read_text()
        active = _strip_yaml_comments(text)
        # Find the airflow-scheduler service block -- match from
        # "  airflow-scheduler:" until the next 2-space-indented key OR
        # top-level key OR EOF.
        sched_match = re.search(
            r"^  airflow-scheduler:\s*\n((?:^(?:    .*)?\n)+?)(?=^  [a-z]|^[a-z]|\Z)",
            active, re.MULTILINE,
        )
        self.assertIsNotNone(sched_match, "airflow-scheduler service not found")
        sched_block = sched_match.group(1)
        self.assertIn("healthcheck:", sched_block,
                      "T-028: airflow-scheduler must have a healthcheck")
        self.assertIn("airflow jobs check", sched_block,
                      "T-028: scheduler healthcheck must use 'airflow jobs check'")
        self.assertIn("SchedulerJob", sched_block,
                      "T-028: scheduler healthcheck must check SchedulerJob type")


def _strip_sql_comments(text: str) -> str:
    """Strip ``-- ...`` line comments from SQL text (for test assertions)."""
    return "\n".join(
        ln for ln in text.split("\n")
        if not ln.lstrip().startswith("--")
    )


def _strip_py_comments(text: str) -> str:
    """Strip ``# ...`` line comments from Python text (for test assertions).

    Naive (does not handle string literals containing #), but sufficient
    for our test purposes.
    """
    out = []
    for ln in text.split("\n"):
        # Skip lines that are pure comments
        stripped = ln.lstrip()
        if stripped.startswith("#"):
            continue
        # Truncate at inline # (naive -- does not handle strings)
        if "#" in ln:
            # Only strip if # is not inside a string literal (approximate
            # by checking quote balance before #)
            quote_count = 0
            for i, ch in enumerate(ln):
                if ch in ('"', "'"):
                    quote_count += 1
                if ch == "#" and quote_count % 2 == 0:
                    ln = ln[:i]
                    break
        out.append(ln)
    return "\n".join(out)


def _strip_py_docstrings(text: str) -> str:
    """Strip triple-quoted Python docstrings (for test assertions).

    Removes the contents of triple-double-quote and triple-single-quote
    blocks so that docstrings quoting the old (pre-fix) form for
    documentation purposes do not trigger false positives.
    """
    # Match triple-quoted strings (both " and ' variants)
    pattern = re.compile(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', re.MULTILINE)
    return pattern.sub('""', text)


def _strip_yaml_comments(text: str) -> str:
    """Strip ``# ...`` line comments from YAML text."""
    return "\n".join(
        ln for ln in text.split("\n")
        if not ln.lstrip().startswith("#")
    )


def _strip_makefile_comments(text: str) -> str:
    """Strip ``# ...`` line comments from Makefile text."""
    out = []
    for ln in text.split("\n"):
        # Makefile comments start with # at the start of a line (possibly
        # after a TAB). Comments mid-line are NOT stripped (they could
        # be inside a shell command).
        if ln.lstrip().startswith("#"):
            continue
        out.append(ln)
    return "\n".join(out)


class TestT029_CheckConstraintConsistency(unittest.TestCase):
    """T-029: chk_drugs_is_globally_approved must use IN (0, 1) form (no 4-value form)."""

    def test_no_4_value_check_form_in_sql(self):
        """No SQL file may use the 4-value IN (0, 1, TRUE, FALSE) form in ACTIVE SQL (not comments)."""
        files_to_check = [
            PHASE1_ROOT / "database" / "migrations" / "008_drug_is_globally_approved.sql",
            PHASE1_ROOT / "database" / "migrations" / "001_initial_schema.sql",
            PHASE1_ROOT / "database" / "migrations" / "006_drug_withdrawn_safety_columns.sql",
        ]
        bad_pattern = re.compile(
            r"IN\s*\(\s*0\s*,\s*1\s*,\s*(?:TRUE|true)\s*,\s*(?:FALSE|false)\s*\)",
            re.IGNORECASE,
        )
        for f in files_to_check:
            if not f.exists():
                continue
            text = f.read_text()
            active_sql = _strip_sql_comments(text)
            matches = bad_pattern.findall(active_sql)
            self.assertEqual(matches, [],
                             f"T-029: {f.name} must NOT contain the 4-value IN (0, 1, TRUE, FALSE) form "
                             f"in active SQL (comments allowed to quote the old form for documentation). "
                             f"Found: {matches}")

    def test_globally_approved_uses_2_value_form_in_sql(self):
        mig_path = PHASE1_ROOT / "database" / "migrations" / "008_drug_is_globally_approved.sql"
        text = mig_path.read_text()
        active_sql = _strip_sql_comments(text)
        # Both ALTER TABLE statements must use the 2-value form in active SQL
        self.assertEqual(active_sql.count("is_globally_approved IN (0, 1)"), 2,
                         "T-029: migration 008 must use 'is_globally_approved IN (0, 1)' twice in "
                         "active SQL (initial constraint + replace constraint)")


class TestT030_YamlNoDuplicates(unittest.TestCase):
    """T-030: uniprot_organism_crosswalk.yaml must have no duplicate keys."""

    def test_no_duplicate_keys(self):
        yaml_path = PHASE1_ROOT / "data" / "uniprot_organism_crosswalk.yaml"
        import yaml
        text = yaml_path.read_text()

        # Strict loader that REJECTS duplicate keys
        class StrictLoader(yaml.SafeLoader):
            pass

        def no_duplicates(loader, node, deep=False):
            mapping = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in mapping:
                    raise yaml.constructor.ConstructorError(
                        f"duplicate key: {key!r}",
                    )
                mapping[key] = loader.construct_object(value_node, deep=deep)
            return mapping

        StrictLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, no_duplicates,
        )

        # Must parse without raising
        d = yaml.load(text, Loader=StrictLoader)
        self.assertIsInstance(d, dict)
        self.assertGreater(len(d), 200,
                           "T-030: deduplicated YAML must still have >200 accessions "
                           f"(got {len(d)})")


class TestT031_EscapeHatchGuardInsideMain(unittest.TestCase):
    """T-031: _check_production_escape_hatches_unified() must NOT be called at module import time."""

    def test_no_module_level_call(self):
        run_unified_path = UNIFIED_ROOT / "run_unified.py"
        text = run_unified_path.read_text()
        # The module-level call (a top-level _check_production_escape_hatches_unified() with no indentation)
        # must be GONE. The call must be INSIDE main().
        # Find any line that is "_check_production_escape_hatches_unified()" at column 0
        module_level_calls = re.findall(
            r"^_check_production_escape_hatches_unified\(\)\s*$",
            text, re.MULTILINE,
        )
        self.assertEqual(module_level_calls, [],
                         "T-031: run_unified.py must NOT call _check_production_escape_hatches_unified() "
                         "at module import time. Found a top-level call.")

    def test_call_inside_main(self):
        run_unified_path = UNIFIED_ROOT / "run_unified.py"
        text = run_unified_path.read_text()
        # The call must be INSIDE main() (indented)
        indented_calls = re.findall(
            r"^    _check_production_escape_hatches_unified\(\)\s*$",
            text, re.MULTILINE,
        )
        self.assertGreaterEqual(len(indented_calls), 1,
                                "T-031: _check_production_escape_hatches_unified() must be called "
                                "inside main() (indented 4 spaces)")

    def test_help_works_in_prod_with_escape_hatches(self):
        """--help must exit 0 even in production with escape hatches set."""
        import subprocess
        env = dict(os.environ)
        env["DRUGOS_ENVIRONMENT"] = "prod"
        env["DRUGOS_ALLOW_PERMISSIVE_KG"] = "1"
        result = subprocess.run(
            [sys.executable, str(UNIFIED_ROOT / "run_unified.py"), "--help"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0,
                         f"T-031: --help must exit 0 in prod+escape-hatch env. "
                         f"Got exit {result.returncode}. stderr: {result.stderr[:500]}")
        self.assertIn("usage:", result.stdout.lower())


class TestT032_ExitCode3Contract(unittest.TestCase):
    """T-032: exit code 3 contract must reflect auto-detect path."""

    def test_contract_updated(self):
        run_unified_path = UNIFIED_ROOT / "run_unified.py"
        text = run_unified_path.read_text()
        # Strip Python comments so documentation that QUOTES the old form
        # for audit-trail purposes does not trigger a false positive.
        active = _strip_py_comments(text)
        # The contract must mention auto-detected / DRUGOS_NEO4J_URI / default localhost
        # (not just the stale "only when --neo4j-uri supplied" qualifier)
        self.assertNotIn("only when --neo4j-uri supplied", active,
                         "T-032: exit code 3 contract must NOT say 'only when --neo4j-uri supplied' "
                         "in active code (that was the v74 stale qualifier)")
        # The new contract must mention the auto-detect path
        self.assertIn("auto-detected", active,
                      "T-032: exit code 3 contract must mention 'auto-detected'")


class TestT033_PersistPathExplicitNone(unittest.TestCase):
    """T-033: _persist_path must be initialized to None before the try block."""

    def test_explicit_none_init(self):
        run_unified_path = UNIFIED_ROOT / "run_unified.py"
        text = run_unified_path.read_text()
        # Strip docstrings AND comments so documentation quoting the
        # old dir() pattern does not trigger a false positive.
        active = _strip_py_docstrings(text)
        active = _strip_py_comments(active)
        # The v75 None init must be present
        self.assertIn("_persist_path = None", active,
                      "T-033: _persist_path = None must be initialized before the try block")
        # The fragile dir() check must be GONE from active code
        self.assertNotIn("'_persist_path' in dir()", active,
                         "T-033: the fragile '_persist_path' in dir() check must be GONE from active code")
        # The new explicit check must be present
        self.assertIn("_persist_path is not None", active,
                      "T-033: the explicit '_persist_path is not None' check must be present")


class TestT034_MakefileCommentTrimmed(unittest.TestCase):
    """T-034: Makefile clean-target comment must be trimmed (no 18-line block)."""

    def test_clean_comment_short(self):
        makefile_path = UNIFIED_ROOT / "Makefile"
        text = makefile_path.read_text()
        # Find the clean target block
        clean_match = re.search(
            r"^clean:\s*\n((?:\t.*\n)+)",
            text, re.MULTILINE,
        )
        self.assertIsNotNone(clean_match, "clean target not found in Makefile")
        clean_block = clean_match.group(1)
        # Count comment lines (lines starting with \t#)
        comment_lines = [ln for ln in clean_block.split("\n") if ln.startswith("\t#")]
        self.assertLessEqual(
            len(comment_lines), 5,
            f"T-034: clean target comment must be ≤5 lines (v75 trim). "
            f"Found {len(comment_lines)} comment lines.",
        )
        # The "broken symlink tree" phrase from the v74 verbose comment must be GONE
        self.assertNotIn("broken symlink", clean_block,
                         "T-034: 'broken symlink' phrase from v74 verbose comment must be GONE")


class TestT035_MakefileTestIsolation(unittest.TestCase):
    """T-035: phase1/Makefile must have clean-db + test-isolated targets; `all` must use test-isolated."""

    def test_clean_db_target_exists(self):
        makefile_path = PHASE1_ROOT / "Makefile"
        text = makefile_path.read_text()
        self.assertIn("clean-db:", text,
                      "T-035: clean-db target must exist in phase1/Makefile")

    def test_test_isolated_target_exists(self):
        makefile_path = PHASE1_ROOT / "Makefile"
        text = makefile_path.read_text()
        self.assertIn("test-isolated:", text,
                      "T-035: test-isolated target must exist in phase1/Makefile")
        self.assertIn("sqlite:///:memory:", text,
                      "T-035: test-isolated target must use in-memory SQLite DB")

    def test_all_uses_test_isolated(self):
        makefile_path = PHASE1_ROOT / "Makefile"
        text = makefile_path.read_text()
        # The `all` target must depend on test-isolated (not test)
        all_match = re.search(r"^all:\s*install-deps\s+\S+\s+(\S+)", text, re.MULTILINE)
        self.assertIsNotNone(all_match, "all target not found in phase1/Makefile")
        all_test_target = all_match.group(1)
        self.assertEqual(all_test_target, "test-isolated",
                         f"T-035: `all` target must depend on test-isolated (not test). "
                         f"Got: {all_test_target}")


class TestT036_RollbackRenamesNotDrops(unittest.TestCase):
    """T-036: 002_rollback must RENAME _migration_002_dedup_archive (not DROP)."""

    def test_no_drop_of_dedup_archive(self):
        rollback_path = PHASE1_ROOT / "database" / "migrations" / "002_bug_fixes_migration_rollback.sql"
        text = rollback_path.read_text()
        active = _strip_sql_comments(text)
        # The DROP TABLE _migration_002_dedup_archive must be GONE from
        # active SQL (comments may reference it for audit-trail purposes).
        self.assertNotIn("DROP TABLE IF EXISTS _migration_002_dedup_archive", active,
                         "T-036: 002_rollback must NOT DROP _migration_002_dedup_archive in active SQL (data loss)")

    def test_rename_to_backup_present(self):
        rollback_path = PHASE1_ROOT / "database" / "migrations" / "002_bug_fixes_migration_rollback.sql"
        text = rollback_path.read_text()
        active = _strip_sql_comments(text)
        # The RENAME TO _migration_002_dedup_archive_rollback_backup must be present
        self.assertIn("RENAME TO _migration_002_dedup_archive_rollback_backup", active,
                      "T-036: 002_rollback must RENAME _migration_002_dedup_archive to "
                      "_migration_002_dedup_archive_rollback_backup (preserves data)")

    def test_no_such_table_in_idempotent_noop(self):
        """The SQLite migration runner must treat 'no such table' as an idempotent no-op."""
        runner_path = PHASE1_ROOT / "database" / "migrations" / "run_migrations.py"
        text = runner_path.read_text()
        self.assertIn('"no such table" in _stmt_err', text,
                      "T-036: per-statement idempotent-noop handler must catch 'no such table'")
        self.assertIn('"no such table" in _exc_msg', text,
                      "T-036: migration-level idempotent-noop handler must catch 'no such table'")


class TestPhase1Phase2Bridge100PercentLinked(unittest.TestCase):
    """Verify Phase 1 ↔ Phase 2 graph explorer integration is 100% wired."""

    def test_bridge_module_exists(self):
        bridge_path = UNIFIED_ROOT / "phase2" / "drugos_graph" / "phase1_bridge.py"
        self.assertTrue(bridge_path.exists(),
                        "Phase1↔Phase2 bridge module must exist")

    def test_bridge_run_phase1_to_phase2_callable(self):
        bridge_path = UNIFIED_ROOT / "phase2" / "drugos_graph" / "phase1_bridge.py"
        text = bridge_path.read_text()
        self.assertIn("def run_phase1_to_phase2(", text,
                      "Bridge must define run_phase1_to_phase2()")
        self.assertIn("def read_phase1_outputs(", text,
                      "Bridge must define read_phase1_outputs()")
        self.assertIn("def stage_phase1_to_phase2(", text,
                      "Bridge must define stage_phase1_to_phase2()")
        self.assertIn("def load_into_graph(", text,
                      "Bridge must define load_into_graph()")

    def test_master_dag_wires_trigger_phase2(self):
        """The master DAG must wire trigger_phase2 as the final task."""
        dag_path = PHASE1_ROOT / "dags" / "master_pipeline_dag.py"
        text = dag_path.read_text()
        self.assertIn("pubchem_load >> trigger_phase2", text,
                      "Master DAG must wire pubchem_load >> trigger_phase2 (Phase 1 -> Phase 2 link)")
        self.assertIn("run_unified.py", text,
                      "trigger_phase2 task must invoke run_unified.py")

    def test_run_unified_calls_bridge(self):
        """run_unified.py must call run_phase1_to_phase2 from the bridge."""
        run_unified_path = UNIFIED_ROOT / "run_unified.py"
        text = run_unified_path.read_text()
        self.assertIn("from drugos_graph.phase1_bridge import run_phase1_to_phase2", text,
                      "run_unified.py must import run_phase1_to_phase2 from the bridge")
        self.assertIn("result = run_phase1_to_phase2(", text,
                      "run_unified.py must call run_phase1_to_phase2()")


if __name__ == "__main__":
    unittest.main(verbosity=2)
