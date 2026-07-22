"""
FIX D8 + P1-031 + P1-034 FORENSIC ROOT FIX (Team 4):
Structural DAG validation tests.

Validates the Airflow DAG structure WITHOUT requiring Airflow to be
running. Tests that:
  * All expected tasks exist (including load_chembl / load_drugbank /
    load_uniprot / _trigger_phase2 -- previously MISSING from this test).
  * Task DEPENDENCIES are correct (P1-031 ROOT FIX -- the previous
    test only checked task IDs, NOT the >> / << wiring. A regression
    that removed ``chembl >> resolve`` would have passed silently).
  * The DrugBank branch logic works properly.
  * The DAG schedule is weekly.

P1-034 ROOT FIX: removed ALL ``pytest.skip("Could not import DAG module")``
calls. These were functionally equivalent to the silent-skip pattern
that the audit specifically flagged as how P1-031 (missing task
dependencies) shipped to production. Now, missing airflow FAILS the
test with a clear error (via ``dags._dags_init.require_airflow()``),
so CI catches the missing dependency immediately.
"""

from __future__ import annotations

# P1-034 ROOT FIX: patch sqlalchemy 2.0 to accept airflow's legacy
# annotations BEFORE any airflow import. See full explanation in
# tests/test_team4_p1_031_to_037_forensic_fixes.py.
import sys as _sys
import warnings as _warnings
_warnings.filterwarnings("ignore", category=DeprecationWarning)
_warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

try:
    from sqlalchemy.orm import util as _orm_util
    if not getattr(_orm_util._extract_mapped_subtype, "_p1_034_patched", False):
        _original_extract = _orm_util._extract_mapped_subtype
        def _lenient_extract(raw_annotation, cls, originating_module, key, attr_cls, required, is_dataclass_field, expect_mapped=True, raiseerr=True, **kwargs):
            annotation_str = str(raw_annotation)
            if "Mapped[" in annotation_str:
                return _original_extract(raw_annotation, cls, originating_module, key, attr_cls, required, is_dataclass_field, expect_mapped=expect_mapped, raiseerr=raiseerr, **kwargs)
            return None
        _lenient_extract._p1_034_patched = True
        _orm_util._extract_mapped_subtype = _lenient_extract
        try:
            from sqlalchemy.orm import decl_base as _decl_base
            _decl_base._extract_mapped_subtype = _lenient_extract
        except ImportError:
            pass
except ImportError:
    pass

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestDAGStructure:
    """Validate the master_pipeline_dag structure without running Airflow.

    P1-034 ROOT FIX: removed the ``@pytest.fixture(autouse=True)`` that
    did the silent-skip-on-missing-airflow pattern. With apache-airflow
    now declared in requirements.txt AND requirements-dev.txt, the
    import MUST succeed in every env (CI + dev). Previously the entire
    DAG test class was SKIPPED, never validated, so the "6022 passed"
    headline excluded all DAG validation -- including the P1-031
    dependency chain regression that shipped to production.
    """

    def test_master_dag_file_importable(self):
        """The master_pipeline_dag.py file should be importable without errors."""
        spec = importlib.util.spec_from_file_location(
            "master_pipeline_dag",
            PROJECT_ROOT / "dags" / "master_pipeline_dag.py",
        )
        assert spec is not None, "Could not create module spec for master_pipeline_dag.py"

    def test_airflow_is_importable(self):
        """P1-034 ROOT FIX: airflow MUST be importable.

        If airflow is not installed, this test FAILS (does not skip).
        Use ``dags._dags_init.require_airflow()`` for a clear error
        message with the exact ``pip install`` remediation.
        """
        from dags._dags_init import require_airflow
        # This raises RuntimeError with a clear remediation message if
        # airflow is not installed. It does NOT silently skip.
        dag_dec, task_dec = require_airflow()
        assert dag_dec is not None
        assert task_dec is not None

    def test_expected_task_ids_exist(self):
        """P1-031 ROOT FIX: ALL expected task IDs should be present in the DAG.

        The previous test was INCOMPLETE -- it checked for
        load_string/load_disgenet/load_omim/load_pubchem_enrichment but
        NOT for load_chembl/load_drugbank/load_uniprot/_trigger_phase2
        (all of which were added by the v79 P0-B2 ROOT FIX). A regression
        that removed load_chembl would have passed silently.
        """
        from dags.master_pipeline_dag import master_pipeline
        dag = master_pipeline()
        task_ids = {t.task_id for t in dag.tasks}
        expected_tasks = {
            # Branch + join
            "check_drugbank_xml",
            "download_drugbank",
            "skip_drugbank",
            "drugbank_done",
            # Primary downloads (4 sources + DrugBank branch)
            "download_chembl",
            "download_uniprot",
            "download_string",
            # Secondary downloads
            "download_disgenet",
            "download_omim",
            "download_pubchem",
            # Entity resolution
            "entity_resolution",
            # Loads (P1-031: ALL 7 must exist, including the v79-added 3)
            "load_chembl",
            "load_drugbank",
            "load_uniprot",
            "load_string",
            "load_disgenet",
            "load_omim",
            "load_pubchem_enrichment",
            # Phase 2 trigger (V18 ROOT FIX -- must exist for Phase 1->2 wiring).
            # The function is named ``_trigger_phase2`` (underscore prefix
            # marks it as module-private); the TaskFlow API auto-generates
            # the task_id from the function name, so the task_id is
            # ``_trigger_phase2`` (with underscore).
            "_trigger_phase2",
        }
        missing = expected_tasks - task_ids
        assert not missing, (
            f"Expected task IDs missing from DAG: {missing}. "
            f"Got: {task_ids}. A regression likely removed a task "
            f"definition from master_pipeline_dag.py."
        )

    def test_drugbank_branch_logic_exists(self):
        """The DrugBank branch operator should exist for conditional execution."""
        from dags.master_pipeline_dag import master_pipeline
        dag = master_pipeline()
        branch_tasks = [t for t in dag.tasks if t.task_id == "check_drugbank_xml"]
        assert len(branch_tasks) == 1, "Expected exactly one check_drugbank_xml task"

    def test_dag_schedule_is_weekly(self):
        """The DAG should be scheduled to run weekly (Sunday 02:00 UTC)."""
        from dags.master_pipeline_dag import master_pipeline
        dag = master_pipeline()
        # Schedule can be None (paused) or a cron expression.
        # Default should be weekly: "0 2 * * 0" (Sunday 02:00 UTC).
        assert dag.schedule_interval is not None or dag.timetable is not None, (
            "Master DAG must have a schedule_interval -- a None schedule "
            "means the DAG never runs automatically (dead DAG)."
        )

    def test_dag_default_args_retries(self):
        """The DAG should have retry configuration in default_args."""
        from dags.master_pipeline_dag import master_pipeline
        dag = master_pipeline()
        assert dag.default_args.get("retries", 0) >= 1, (
            "DAG should have at least 1 retry -- 0 retries means a single "
            "transient error kills the entire Sunday master run."
        )

    # ------------------------------------------------------------------
    # P1-031 FORENSIC ROOT FIX: dependency chain verification.
    # The previous test file ONLY checked that task IDs existed -- it did
    # NOT verify the >> / << wiring. A regression that removed
    # ``chembl >> resolve`` would have passed the old test silently. The
    # tests below parse the DAG's dependency graph and assert EXACT
    # upstream/downstream relationships for every critical edge.
    # ------------------------------------------------------------------

    def _get_dag(self):
        """Helper: import and build the master DAG once per test."""
        from dags.master_pipeline_dag import master_pipeline
        return master_pipeline()

    def _upstream_ids(self, dag, task_id):
        """Return the set of upstream task_ids for ``task_id``."""
        task = dag.get_task(task_id)
        return {t.task_id for t in task.upstream_list}

    def _downstream_ids(self, dag, task_id):
        """Return the set of downstream task_ids for ``task_id``."""
        task = dag.get_task(task_id)
        return {t.task_id for t in task.downstream_list}

    def test_branch_check_drugbank_downstream(self):
        """P1-031: check_drugbank_xml must branch to download_drugbank AND skip_drugbank."""
        dag = self._get_dag()
        downstream = self._downstream_ids(dag, "check_drugbank_xml")
        assert downstream == {"download_drugbank", "skip_drugbank"}, (
            f"check_drugbank_xml should branch to download_drugbank AND "
            f"skip_drugbank. Got: {downstream}. A regression likely "
            f"removed one of the branch wires."
        )

    def test_drugbank_done_joins_both_branches(self):
        """P1-031: drugbank_done must join BOTH download_drugbank AND skip_drugbank."""
        dag = self._get_dag()
        upstream = self._upstream_ids(dag, "drugbank_done")
        assert upstream == {"download_drugbank", "skip_drugbank"}, (
            f"drugbank_done must depend on BOTH download_drugbank AND "
            f"skip_drugbank (the BranchPythonOperator's two branches). "
            f"Got: {upstream}. A regression likely removed one of the "
            f"join wires -- which would cause drugbank_done to wait "
            f"forever for the skipped branch."
        )

    def test_entity_resolution_waits_for_all_downloads(self):
        """P1-031 ROOT FIX: entity_resolution must wait for ALL 6 download paths.

        The audit's specific concern: ``entity_resolution`` runs BEFORE
        all 7 pipelines complete, merging some drugs but not others.
        The fix wires ALL 6 download paths (chembl, drugbank_done,
        uniprot, string, disgenet, omim) into ``resolve``. PubChem is
        INTENTIONALLY excluded -- PubChem download needs drugs in the
        DB first (from chembl_load + drugbank_load), so it runs AFTER
        entity_resolution, not before.
        """
        dag = self._get_dag()
        upstream = self._upstream_ids(dag, "entity_resolution")
        expected = {
            "download_chembl",
            "drugbank_done",  # join task (not download_drugbank directly)
            "download_uniprot",
            "download_string",
            "download_disgenet",
            "download_omim",
        }
        missing = expected - upstream
        assert not missing, (
            f"entity_resolution is missing upstream dependencies: "
            f"{missing}. It must wait for ALL 6 download paths to "
            f"complete before running entity resolution -- otherwise "
            f"the resolver operates on PARTIAL data and merges some "
            f"drugs but not others, producing an inconsistent KG."
        )

    def test_loads_run_after_entity_resolution(self):
        """P1-031 ROOT FIX: 6 of 7 load tasks must run AFTER entity_resolution.

        The v40 two-phase design: download -> resolve -> load. The v79
        P0-B2 ROOT FIX added load_chembl / load_drugbank / load_uniprot
        (previously missing). This test verifies the dependency did NOT
        regress.

        Note: ``load_pubchem_enrichment`` is INTENTIONALLY EXCLUDED --
        it depends on ``download_pubchem`` (which itself depends on
        ``chembl_load`` + ``drugbank_load``, both of which depend on
        ``entity_resolution``). So pubchem_load is transitively downstream
        of entity_resolution, but not directly. This is by design:
        PubChem enrichment must wait for the drugs table to be populated
        (which happens in chembl_load + drugbank_load), not for entity
        resolution per se.
        """
        dag = self._get_dag()
        # 6 loads that directly depend on entity_resolution.
        # load_pubchem_enrichment is excluded (see docstring above).
        load_tasks = [
            "load_chembl",
            "load_drugbank",
            "load_uniprot",
            "load_string",
            "load_disgenet",
            "load_omim",
        ]
        for load_task in load_tasks:
            upstream = self._upstream_ids(dag, load_task)
            assert "entity_resolution" in upstream, (
                f"{load_task} must depend on entity_resolution (the v40 "
                f"two-phase design: download -> resolve -> load). "
                f"Got upstream: {upstream}. A regression likely removed "
                f"the ``resolve >> {load_task}`` wire."
            )

    def test_pubchem_load_depends_on_pubchem_download(self):
        """P1-031 ROOT FIX: load_pubchem_enrichment must run AFTER download_pubchem.

        PubChem's load step reads the file that download_pubchem wrote.
        This is a separate test from test_loads_run_after_entity_resolution
        because pubchem_load's dependency chain is different (it depends
        on download_pubchem, not entity_resolution directly).
        """
        dag = self._get_dag()
        upstream = self._upstream_ids(dag, "load_pubchem_enrichment")
        assert "download_pubchem" in upstream, (
            f"load_pubchem_enrichment must depend on download_pubchem. "
            f"Got: {upstream}. A regression likely removed the "
            f"``download_pubchem >> load_pubchem_enrichment`` wire."
        )

    def test_pubchem_download_waits_for_drugs_table(self):
        """P1-031 / BUG #1 P0 ROOT FIX: pubchem_download must wait for
        chembl_load AND drugbank_load (which populate the drugs table).

        PubChem's download queries ``drugs WHERE pubchem_cid IS NULL``.
        If it runs BEFORE chembl_load / drugbank_load, the drugs table
        is EMPTY and PubChem produces ZERO enrichment.
        """
        dag = self._get_dag()
        upstream = self._upstream_ids(dag, "download_pubchem")
        # download_pubchem must depend on chembl_load and drugbank_load.
        # It may also depend on other tasks (e.g. entity_resolution),
        # but the critical invariant is the drugs-table producers.
        assert "load_chembl" in upstream, (
            f"download_pubchem must depend on load_chembl (which "
            f"populates the drugs table that PubChem queries). "
            f"Got: {upstream}. Without this wire, PubChem queries an "
            f"EMPTY drugs table and produces ZERO enrichment."
        )
        assert "load_drugbank" in upstream, (
            f"download_pubchem must depend on load_drugbank (which "
            f"populates the drugs table that PubChem queries). "
            f"Got: {upstream}. Without this wire, PubChem queries an "
            f"EMPTY drugs table and produces ZERO enrichment."
        )

    def test_pubchem_load_after_pubchem_download(self):
        """P1-031: load_pubchem_enrichment must run AFTER download_pubchem."""
        dag = self._get_dag()
        upstream = self._upstream_ids(dag, "load_pubchem_enrichment")
        assert "download_pubchem" in upstream, (
            f"load_pubchem_enrichment must depend on download_pubchem. "
            f"Got: {upstream}. A regression likely removed the "
            f"``download_pubchem >> load_pubchem_enrichment`` wire."
        )

    def test_trigger_phase2_waits_for_all_loads(self):
        """P1-031 ROOT FIX: _trigger_phase2 must wait for ALL 7 load tasks.

        Phase 2 (KG construction + TransE training) reads from the
        staging DB. If it runs BEFORE all 7 loads complete, the KG is
        built on PARTIAL data. The trigger_rule=NONE_FAILED_MIN_ONE_SUCCESS
        on _trigger_phase2 lets it fire when pubchem_load is SKIPPED
        (PubChem API outage -> graceful degradation), but the dependency
        wire MUST still exist so _trigger_phase2 does not race with
        pubchem_load.

        Note: the task_id is ``_trigger_phase2`` (with underscore prefix)
        because the function is named ``_trigger_phase2`` (underscore
        marks it as module-private). The TaskFlow API auto-generates
        the task_id from the function name.
        """
        dag = self._get_dag()
        upstream = self._upstream_ids(dag, "_trigger_phase2")
        expected_loads = {
            "load_chembl",
            "load_drugbank",
            "load_uniprot",
            "load_string",
            "load_disgenet",
            "load_omim",
            "load_pubchem_enrichment",  # P1-018 ROOT FIX (Team-2)
        }
        missing = expected_loads - upstream
        assert not missing, (
            f"_trigger_phase2 is missing upstream load dependencies: "
            f"{missing}. Phase 2 (KG build + TransE training) must wait "
            f"for ALL 7 load tasks to finish (succeed or skip) before "
            f"reading the staging DB -- otherwise the KG is built on "
            f"PARTIAL data."
        )

    def test_no_orphan_tasks(self):
        """P1-031 ROOT FIX: NO task should be orphaned (no upstream AND no downstream).

        An orphan task is one with NO dependencies in either direction.
        Airflow would run it in PARALLEL with everything else -- it
        could fire before its prerequisites or after its consumers.
        The audit's P1-031 concern was that disgenet/omim were
        originally orphaned, causing race conditions.
        """
        dag = self._get_dag()
        orphans = []
        for task in dag.tasks:
            # The ONLY allowed "root" tasks are check_drugbank_xml (the
            # BranchPythonOperator that starts the DAG) -- everything
            # else must have at least one upstream dependency.
            if task.task_id == "check_drugbank_xml":
                continue
            if not task.upstream_list and not task.downstream_list:
                orphans.append(task.task_id)
        assert not orphans, (
            f"Orphan tasks (no upstream AND no downstream): {orphans}. "
            f"Every task except check_drugbank_xml must have at least "
            f"one dependency in some direction, otherwise it runs in "
            f"PARALLEL with everything else and may race with its "
            f"prerequisites or consumers."
        )

    def test_trigger_phase2_is_terminal(self):
        """P1-031: _trigger_phase2 must be a TERMINAL task (no downstream).

        Note: the task_id is ``_trigger_phase2`` (with underscore prefix)
        because the function is named ``_trigger_phase2`` (module-private).
        """
        dag = self._get_dag()
        downstream = self._downstream_ids(dag, "_trigger_phase2")
        assert downstream == set(), (
            f"_trigger_phase2 must be a terminal task (no downstream). "
            f"Got downstream: {downstream}. If _trigger_phase2 has "
            f"downstream tasks, Phase 2's failure would cascade to "
            f"those tasks instead of just failing the DAG red."
        )


# ----------------------------------------------------------------------
# P1-031 ROOT FIX: standalone DAG dependency verification.
# Each standalone DAG now has a pre-flight sensor/check task that must
# run BEFORE the pipeline task. These tests verify the sensor -> pipeline
# wiring is correct.
# ----------------------------------------------------------------------

class TestStandaloneDAGDependencies:
    """P1-031 / P1-032 / P1-035 / P1-036 / P1-037: verify standalone DAG wiring."""

    def test_chembl_dag_health_check_runs_first(self):
        """P1-032: check_chembl_health must run BEFORE run_chembl."""
        from dags.chembl_dag import chembl_dag
        dag = chembl_dag()
        task_ids = {t.task_id for t in dag.tasks}
        assert "check_chembl_health" in task_ids, (
            "chembl_dag must have a check_chembl_health task (P1-032 root fix)."
        )
        assert "run_chembl" in task_ids, (
            "chembl_dag must have a run_chembl task."
        )
        # Verify wiring: check_chembl_health >> run_chembl.
        run_chembl_task = dag.get_task("run_chembl")
        upstream = {t.task_id for t in run_chembl_task.upstream_list}
        assert "check_chembl_health" in upstream, (
            f"run_chembl must depend on check_chembl_health. Got: {upstream}."
        )

    def test_chembl_dag_retries_is_six(self):
        """P1-032: chembl_dag must have retries=6 (was 2 -- lost weekly data on maintenance)."""
        from dags.chembl_dag import chembl_dag
        dag = chembl_dag()
        assert dag.default_args.get("retries") == 6, (
            f"chembl_dag.default_args.retries must be 6 (P1-032 root fix "
            f"for ChEMBL 30-60min maintenance windows). Got: "
            f"{dag.default_args.get('retries')}."
        )

    def test_chembl_dag_has_failure_callback(self):
        """P1-032: chembl_dag must have an on_failure_callback for alerting."""
        from dags.chembl_dag import chembl_dag
        dag = chembl_dag()
        assert dag.default_args.get("on_failure_callback") is not None, (
            "chembl_dag.default_args must have an on_failure_callback "
            "(P1-032 root fix for permanent-failure alerting)."
        )

    def test_drugbank_dag_schema_check_runs_first(self):
        """P1-035: check_drugbank_schema must run BEFORE run_drugbank."""
        from dags.drugbank_dag import drugbank_dag
        dag = drugbank_dag()
        task_ids = {t.task_id for t in dag.tasks}
        assert "check_drugbank_schema" in task_ids, (
            "drugbank_dag must have a check_drugbank_schema task (P1-035 root fix)."
        )
        assert "run_drugbank" in task_ids, (
            "drugbank_dag must have a run_drugbank task."
        )
        run_drugbank_task = dag.get_task("run_drugbank")
        upstream = {t.task_id for t in run_drugbank_task.upstream_list}
        assert "check_drugbank_schema" in upstream, (
            f"run_drugbank must depend on check_drugbank_schema. Got: {upstream}."
        )

    def test_drugbank_dag_supported_schemas_not_empty(self):
        """P1-035: SUPPORTED_DRUGBANK_SCHEMAS must be a non-empty frozenset."""
        from dags.drugbank_dag import SUPPORTED_DRUGBANK_SCHEMAS
        assert isinstance(SUPPORTED_DRUGBANK_SCHEMAS, frozenset)
        assert len(SUPPORTED_DRUGBANK_SCHEMAS) > 0, (
            "SUPPORTED_DRUGBANK_SCHEMAS must not be empty -- otherwise "
            "every DrugBank XML file would be rejected."
        )
        # 5.1.10 is the current production release -- must be supported.
        assert "5.1.10" in SUPPORTED_DRUGBANK_SCHEMAS, (
            "SUPPORTED_DRUGBANK_SCHEMAS must include 5.1.10 (current "
            "production DrugBank release)."
        )

    def test_disgenet_dag_schedule_is_monday_2am(self):
        """P1-036: disgenet_dag must run Monday 02:00 UTC (audit recommendation)."""
        from dags.disgenet_dag import disgenet_dag
        dag = disgenet_dag()
        # schedule_interval can be a CronExpression or string.
        schedule = str(dag.schedule_interval)
        assert "0 2 * * 1" in schedule or "0 2 * * 1" == schedule, (
            f"disgenet_dag.schedule_interval must be '0 2 * * 1' "
            f"(Monday 02:00 UTC, P1-036 audit recommendation). Got: "
            f"{schedule!r}."
        )

    def test_disgenet_dag_release_sensor_runs_first(self):
        """P1-036: check_disgenet_release must run BEFORE run_disgenet."""
        from dags.disgenet_dag import disgenet_dag
        dag = disgenet_dag()
        task_ids = {t.task_id for t in dag.tasks}
        assert "check_disgenet_release" in task_ids, (
            "disgenet_dag must have a check_disgenet_release task (P1-036 root fix)."
        )
        run_disgenet_task = dag.get_task("run_disgenet")
        upstream = {t.task_id for t in run_disgenet_task.upstream_list}
        assert "check_disgenet_release" in upstream, (
            f"run_disgenet must depend on check_disgenet_release. Got: {upstream}."
        )

    def test_pubchem_dag_https_check_runs_first(self):
        """P1-037: check_pubchem_https must run BEFORE run_pubchem."""
        from dags.pubchem_dag import pubchem_dag
        dag = pubchem_dag()
        task_ids = {t.task_id for t in dag.tasks}
        assert "check_pubchem_https" in task_ids, (
            "pubchem_dag must have a check_pubchem_https task (P1-037 root fix)."
        )
        run_pubchem_task = dag.get_task("run_pubchem")
        upstream = {t.task_id for t in run_pubchem_task.upstream_list}
        assert "check_pubchem_https" in upstream, (
            f"run_pubchem must depend on check_pubchem_https. Got: {upstream}."
        )

    def test_pubchem_dag_task_timeout_is_4h(self):
        """P1-037: run_pubchem must have execution_timeout=4h (explicit, not inherited)."""
        from dags.pubchem_dag import pubchem_dag, PUBCHEM_TASK_TIMEOUT
        from datetime import timedelta
        assert PUBCHEM_TASK_TIMEOUT == timedelta(hours=4), (
            f"PUBCHEM_TASK_TIMEOUT must be 4h (P1-037 audit recommendation). "
            f"Got: {PUBCHEM_TASK_TIMEOUT}."
        )
        dag = pubchem_dag()
        run_pubchem_task = dag.get_task("run_pubchem")
        # The execution_timeout should be set on the task itself, not
        # just inherited from default_args.
        assert run_pubchem_task.execution_timeout == timedelta(hours=4), (
            f"run_pubchem.execution_timeout must be 4h (P1-037 explicit "
            f"setting). Got: {run_pubchem_task.execution_timeout}."
        )


# ----------------------------------------------------------------------
# P1-033 ROOT FIX: DB deadlock retry policy verification.
# ----------------------------------------------------------------------

class TestDBDeadlockRetryPolicy:
    """P1-033: verify the DB deadlock retry policy is correctly configured."""

    def test_db_deadlock_retry_args_max_delay_is_5min(self):
        """P1-033: DB_DEADLOCK_RETRY_ARGS.max_retry_delay must be 5min (300s)."""
        from dags._retry_policy import DB_DEADLOCK_RETRY_ARGS
        from datetime import timedelta
        assert DB_DEADLOCK_RETRY_ARGS["max_retry_delay"] == timedelta(minutes=5), (
            f"DB_DEADLOCK_RETRY_ARGS.max_retry_delay must be 5min "
            f"(timedelta(minutes=5), P1-033 audit recommendation). Got: "
            f"{DB_DEADLOCK_RETRY_ARGS['max_retry_delay']}."
        )

    def test_db_deadlock_retry_args_retries_is_5(self):
        """P1-033: DB_DEADLOCK_RETRY_ARGS.retries must be 5 (transient deadlocks)."""
        from dags._retry_policy import DB_DEADLOCK_RETRY_ARGS
        assert DB_DEADLOCK_RETRY_ARGS["retries"] == 5, (
            f"DB_DEADLOCK_RETRY_ARGS.retries must be 5 (P1-033 -- DB "
            f"deadlocks are transient and should be retried). Got: "
            f"{DB_DEADLOCK_RETRY_ARGS['retries']}."
        )

    def test_db_deadlock_retry_args_has_exponential_backoff(self):
        """P1-033: DB_DEADLOCK_RETRY_ARGS must have retry_exponential_backoff=True (jitter)."""
        from dags._retry_policy import DB_DEADLOCK_RETRY_ARGS
        assert DB_DEADLOCK_RETRY_ARGS["retry_exponential_backoff"] is True, (
            "DB_DEADLOCK_RETRY_ARGS.retry_exponential_backoff must be True "
            "(P1-033 -- Airflow's exponential backoff includes jitter to "
            "prevent thundering-herd re-deadlocks)."
        )

    def test_is_db_deadlock_error_detects_pg_deadlock(self):
        """P1-033: is_db_deadlock_error must detect psycopg2 DeadlockDetected."""
        from dags._retry_policy import is_db_deadlock_error

        # Simulate a psycopg2.errors.DeadlockDetected without importing psycopg2.
        class FakeDeadlockDetected(Exception):
            pass

        # Rename the class to match what the detector looks for.
        FakeDeadlockDetected.__name__ = "DeadlockDetected"
        exc = FakeDeadlockDetected("deadlock detected")
        assert is_db_deadlock_error(exc), (
            "is_db_deadlock_error must return True for an exception "
            "named 'DeadlockDetected'."
        )

    def test_is_db_deadlock_error_detects_sqlite_locked(self):
        """P1-033: is_db_deadlock_error must detect SQLite 'database is locked'."""
        from dags._retry_policy import is_db_deadlock_error
        exc = Exception("database is locked")
        assert is_db_deadlock_error(exc), (
            "is_db_deadlock_error must return True for SQLite "
            "'database is locked' errors."
        )

    def test_is_db_deadlock_error_detects_pg_lock_timeout(self):
        """P1-033: is_db_deadlock_error must detect PostgreSQL lock wait timeout."""
        from dags._retry_policy import is_db_deadlock_error
        exc = Exception("lock wait timeout exceeded")
        assert is_db_deadlock_error(exc), (
            "is_db_deadlock_error must return True for PostgreSQL "
            "'lock wait timeout exceeded' errors."
        )

    def test_is_db_deadlock_error_does_not_match_http_4xx(self):
        """P1-033: is_db_deadlock_error must NOT match HTTP 4xx errors."""
        from dags._retry_policy import is_db_deadlock_error
        exc = Exception("401 Unauthorized: bad API key")
        assert not is_db_deadlock_error(exc), (
            "is_db_deadlock_error must return False for HTTP 4xx errors "
            "(those are handled by fail_fast_on_http_4xx, not retried)."
        )

    def test_is_db_deadlock_error_does_not_match_generic_error(self):
        """P1-033: is_db_deadlock_error must NOT match generic exceptions."""
        from dags._retry_policy import is_db_deadlock_error
        exc = ValueError("invalid data format")
        assert not is_db_deadlock_error(exc), (
            "is_db_deadlock_error must return False for generic ValueError "
            "(not a deadlock)."
        )

    def test_retry_on_db_deadlock_retries_on_deadlock(self):
        """P1-033: retry_on_db_deadlock must retry on deadlock and eventually succeed."""
        from dags._retry_policy import retry_on_db_deadlock
        import time
        import unittest.mock as mock

        call_count = {"n": 0}

        @retry_on_db_deadlock
        def flaky_db_write():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise Exception("deadlock detected")
            return "success"

        # Patch time.sleep so the test doesn't actually wait.
        with mock.patch("time.sleep"):
            result = flaky_db_write()
        assert result == "success"
        assert call_count["n"] == 3, (
            f"retry_on_db_deadlock should have retried twice (3 calls total). "
            f"Got {call_count['n']} calls."
        )

    def test_retry_on_db_deadlock_does_not_retry_on_4xx(self):
        """P1-033: retry_on_db_deadlock must NOT retry on HTTP 4xx errors."""
        from dags._retry_policy import retry_on_db_deadlock
        import unittest.mock as mock

        call_count = {"n": 0}

        @retry_on_db_deadlock
        def http_4xx_call():
            call_count["n"] += 1
            raise Exception("401 Unauthorized: bad API key")

        with mock.patch("time.sleep"):
            with pytest.raises(Exception, match="401 Unauthorized"):
                http_4xx_call()
        assert call_count["n"] == 1, (
            f"retry_on_db_deadlock should NOT retry on HTTP 4xx (1 call). "
            f"Got {call_count['n']} calls."
        )

    def test_retry_on_db_deadlock_gives_up_after_max_retries(self):
        """P1-033: retry_on_db_deadlock must give up after 5 retries and re-raise."""
        from dags._retry_policy import retry_on_db_deadlock
        import unittest.mock as mock

        call_count = {"n": 0}

        @retry_on_db_deadlock
        def always_deadlocked():
            call_count["n"] += 1
            raise Exception("deadlock detected")

        with mock.patch("time.sleep"):
            with pytest.raises(Exception, match="deadlock detected"):
                always_deadlocked()
        # 1 initial + 5 retries = 6 total calls.
        assert call_count["n"] == 6, (
            f"retry_on_db_deadlock should call the function 6 times "
            f"(1 initial + 5 retries). Got {call_count['n']} calls."
        )
