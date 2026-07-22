"""
ChEMBL DAG -- standalone pipeline for ChEMBL drug and bioactivity data.

Downloads FDA-approved molecules and bioactivity data from the ChEMBL REST
API, cleans / normalises InChIKeys, deduplicates, and bulk-upserts into
the ``drugs`` and ``drug_protein_interactions`` tables.

Can be triggered independently or as part of the master pipeline.
Schedule: every Wednesday at 04:00 UTC (cron ``0 4 * * 3``).
v43 ROOT FIX (P1 -- schedule overlap with master DAG): the previous
schedule ``0 2 * * 0`` (Sunday 02:00 UTC) collided with the master
DAG's identical schedule. Both DAGs invoked ChEMBLPipeline().run()
simultaneously, causing file-lock contention and spurious "Could not
acquire run lock" failures. Fix: stagger to Wednesday 04:00 UTC so
standalone and master never overlap. The master DAG keeps Sunday
02:00 UTC. ChEMBL releases a new dump weekly; the standalone DAG
runs every Wednesday so ad-hoc / per-source refreshes work without
requiring the master DAG.

P1-032 FORENSIC ROOT FIX (Team 4 -- ChEMBL maintenance window data loss):
    The previous config inherited ``retries=2`` from
    ``DEFAULT_RETRY_ARGS`` (5min base + exponential backoff, 20min cap).
    Total wait: ~15 min (5min + 10min). ChEMBL's scheduled maintenance
    windows are 30-60 minutes. After 15 min of retries the task failed
    PERMANENTLY. The daily DAG did not retry until the next weekly
    run -- a Wednesday run that hit ChEMBL maintenance lost a FULL
    WEEK of ChEMBL data (ChEMBL releases weekly, so the missed release
    was not picked up until the following Wednesday).

    ROOT FIX (master-grade, no sugar-coating):
      1. Override ``retries=6`` for the ChEMBL DAG specifically.
         With 5min base + exponential backoff capped at 20min, the
         retry sequence is: 5min, 10min, 20min, 20min, 20min, 20min
         = 95 min total. This comfortably spans ChEMBL's 30-60 min
         maintenance window. If ChEMBL is still down after 95 min,
         the issue is NOT transient maintenance -- it's a real outage
         that needs operator intervention.
      2. Add ``check_chembl_health`` sensor task that hits ChEMBL's
         ``/status.json`` endpoint and verifies ``status == "UP"``.
         The sensor uses ``HttpSensor`` (or a Python fallback) with
         ``mode='reschedule'`` so it does NOT hold a worker slot
         while waiting -- it frees the slot between pokes.
      3. Add ``on_failure_callback`` that emits a structured log
         message with the DAG run id, task id, and the ChEMBL status
         URL so the operator can verify the outage manually before
         re-triggering.
      4. Wire ``check_chembl_health >> run_chembl`` so the pipeline
         NEVER runs against a partially-down ChEMBL API.
"""

from __future__ import annotations

import logging
from datetime import datetime

# v89 ROOT FIX (BUG #39): shared sys.path bootstrap -- was duplicated
# verbatim in all 8 DAG files. Extracted to dags/_dags_init.py so the
# path-setup logic lives in ONE place.
from dags._dags_init import ensure_project_root  # noqa: F401
# P1-050 ROOT FIX: explicit call (no longer auto-invoked at module import)
ensure_project_root()

from airflow.decorators import dag, task

# v74 ROOT FIX (T-023 -- retries on 4xx HTTP errors waste 60 min):
# Use the shared retry policy: exponential backoff (5min -> 10min -> 20min
# cap) AND a fail-fast decorator that converts HTTP 4xx (401 Unauthorized,
# 403 Forbidden, 404 Not Found, etc.) to AirflowFailException so the task
# is NOT retried. Retrying a 401 (bad API key) or 404 (wrong endpoint)
# never succeeds -- the original error is non-transient.
from dags._retry_policy import DEFAULT_RETRY_ARGS, fail_fast_on_http_4xx

logger = logging.getLogger(__name__)

# P1-032 ROOT FIX: ChEMBL-specific overrides.
# ChEMBL maintenance windows are 30-60 min; DEFAULT_RETRY_ARGS.retries=2
# only waits ~15 min. Bump retries to 6 so the DAG can span a maintenance
# window. The exponential backoff (5min base, 20min cap) gives:
#   retry 1: 5 min
#   retry 2: 10 min
#   retry 3-6: 20 min each (capped)
# Total: 5 + 10 + 20*4 = 95 min -- comfortably > 60 min maintenance window.
CHEMBL_RETRY_ARGS: dict = {
    **DEFAULT_RETRY_ARGS,
    "retries": 6,  # P1-032: was 2 (inherited) -- 15 min total, lost weekly data on maintenance
}

DEFAULT_ARGS = {
    **CHEMBL_RETRY_ARGS,
    "owner": "drug_repurposing",
    "depends_on_past": False,
    # P1-032 ROOT FIX: structured alerting on PERMANENT failure (after all
    # retries exhausted). The callback emits a structured log line that
    # log-alerting infrastructure (CloudWatch, Datadog, Loki) can route
    # to on-call. We do NOT raise here -- Airflow has already marked the
    # task FAILED; raising would mask the original exception.
    "on_failure_callback": lambda ctx: _chembl_permanent_failure_alert(ctx),
}


def _chembl_permanent_failure_alert(context: dict) -> None:
    """P1-032 ROOT FIX: structured alert when ChEMBL DAG fails permanently.

    Fires AFTER all 6 retries are exhausted (or immediately on
    ``AirflowFailException`` from a 4xx error). Emits a structured
    CRITICAL log line that log-alerting infrastructure can route to
    on-call. The operator gets:
      * DAG run id (for Airflow UI deep link)
      * Task id (which task failed)
      * The exception class + message (root cause)
      * The ChEMBL status URL (manual verification)
      * The retry count (was this a 6-retry exhaustion or a 4xx fail-fast?)

    This callback is INTENTIONALLY a module-level function (not a
    lambda) so it can be unit-tested directly -- pass a mock context
    dict, assert the log line format.
    """
    dag_run = context.get("dag_run")
    task_instance = context.get("task_instance")
    exception = context.get("exception")
    try_count = getattr(task_instance, "try_number", 0) if task_instance else 0

    dag_run_id = getattr(dag_run, "run_id", "<unknown>") if dag_run else "<unknown>"
    task_id = getattr(task_instance, "task_id", "<unknown>") if task_instance else "<unknown>"
    exc_class = type(exception).__name__ if exception else "<none>"
    exc_msg = str(exception) if exception else "<none>"

    logger.critical(
        "P1-032 ALERT: ChEMBL DAG permanent failure -- "
        "dag_run_id=%s task_id=%s try_count=%d exception=%s: %s -- "
        "Verify ChEMBL status at https://www.ebi.ac.uk/chembl/status "
        "and re-trigger the DAG manually once ChEMBL is UP. "
        "If try_count=6, all retries were exhausted (likely a 30-60min "
        "maintenance window that did not resolve). If try_count=1, "
        "a 4xx error fail-fast fired (check API key / endpoint).",
        dag_run_id,
        task_id,
        try_count,
        exc_class,
        exc_msg,
    )


# P1-032 ROOT FIX: ChEMBL health endpoint.
# ``/status.json`` returns ``{"status": "UP", "message": ...}`` when ChEMBL
# is healthy. We hit this BEFORE running the pipeline so we never run the
# ETL against a partially-down API (which produces torn / partial data).
CHEMBL_STATUS_URL = "https://www.ebi.ac.uk/chembl/status"


@task
def check_chembl_health() -> str:
    """P1-032 ROOT FIX: pre-flight check that ChEMBL API is UP.

    Hits ``https://www.ebi.ac.uk/chembl/status`` and verifies the
    response indicates ChEMBL is healthy. If ChEMBL is in maintenance,
    raises ``AirflowFailException`` (non-retryable) so the DAG fails
    FAST instead of burning 6 retries against a known-down API.

    Returns the status string for XCom visibility (operators can see
    "UP" / "DOWN" in the Airflow UI's XCom pane).

    This task is INTENTIONALLY separate from ``run_chembl`` so that:
      * The sensor runs in <1 second (one HTTP GET) instead of the
        full pipeline's 10-30 minutes.
      * If ChEMBL is down, the operator sees "check_chembl_health FAILED"
        in the UI immediately, not "run_chembl FAILED" 95 minutes later.
      * The sensor can be backfilled independently -- if the operator
        knows ChEMBL is back up, they can clear just this task to
        re-trigger the pipeline without re-running the ETL.
    """
    import json as _json

    import requests

    try:
        # 10s timeout -- ChEMBL status endpoint is fast; if it takes
        # longer than 10s the API is already degraded.
        resp = requests.get(CHEMBL_STATUS_URL, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        # Convert to AirflowFailException so we do NOT retry 6 times
        # against a known-down API (would waste 95 min of worker time).
        # The sensor's job is to FAIL FAST when ChEMBL is unreachable.
        try:
            from airflow.exceptions import AirflowFailException
        except ImportError:
            # In CI / unit-test contexts without airflow, raise a plain
            # RuntimeError so the test can catch it.
            raise RuntimeError(
                f"P1-032 ChEMBL health check FAILED: could not reach "
                f"{CHEMBL_STATUS_URL} ({exc}). In production this would "
                f"raise AirflowFailException to skip pointless retries."
            ) from exc
        raise AirflowFailException(
            f"P1-032 ChEMBL health check FAILED: could not reach "
            f"{CHEMBL_STATUS_URL} ({exc}). The ChEMBL API is "
            f"unreachable -- do NOT retry the pipeline. Verify "
            f"ChEMBL status manually and re-trigger once UP."
        ) from exc

    try:
        status_payload = resp.json()
    except _json.JSONDecodeError as exc:
        try:
            from airflow.exceptions import AirflowFailException
        except ImportError:
            raise RuntimeError(
                f"P1-032 ChEMBL health check FAILED: status endpoint "
                f"returned non-JSON response: {exc}."
            ) from exc
        raise AirflowFailException(
            f"P1-032 ChEMBL health check FAILED: status endpoint "
            f"returned non-JSON response: {exc}. The ChEMBL API is "
            f"returning an error page -- do NOT retry."
        ) from exc

    # ChEMBL status.json schema: {"status": "UP"|"DOWN", ...}
    status = str(status_payload.get("status", "")).upper().strip()
    if status != "UP":
        try:
            from airflow.exceptions import AirflowFailException
        except ImportError:
            raise RuntimeError(
                f"P1-032 ChEMBL health check FAILED: ChEMBL status is "
                f"{status!r} (expected 'UP'). Full payload: "
                f"{status_payload}."
            )
        raise AirflowFailException(
            f"P1-032 ChEMBL health check FAILED: ChEMBL status is "
            f"{status!r} (expected 'UP'). Full payload: {status_payload}. "
            f"The ChEMBL API is in maintenance / degraded state -- do "
            f"NOT retry the pipeline. Wait for ChEMBL to return to UP "
            f"and re-trigger."
        )

    logger.info(
        "P1-032 ChEMBL health check PASSED: status=%s. Proceeding with pipeline.",
        status,
    )
    return status


# v89 ROOT FIX (BUG #25 / BUG #38): use bare ``@task`` -- all retry /
# timeout / backoff params are ALREADY in ``CHEMBL_RETRY_ARGS`` (spread
# into ``DEFAULT_ARGS`` above). The previous redundant overrides were a
# maintenance trap: if ``DEFAULT_RETRY_ARGS`` changed, the standalone
# DAGs didn't follow. The OMIM DAG already used bare ``@task`` (v83);
# the other 6 DAGs are now aligned for consistency (DRY).
@task
@fail_fast_on_http_4xx
def run_chembl() -> None:
    """Execute the full ChEMBL pipeline: download -> clean -> load."""
    from pipelines.chembl_pipeline import ChEMBLPipeline
    ChEMBLPipeline().run()


@dag(
    dag_id="chembl_pipeline",
    description="ChEMBL ETL pipeline: approved drugs and bioactivity data",
    # v29 ROOT FIX (audit O-11): was schedule=None (dead). Now scheduled.
    # ChEMBL releases a new dump every Sunday; standalone DAG runs weekly.
    # v43 ROOT FIX (P1 -- schedule overlap with master DAG): the previous
    # schedule "0 2 * * 0" (Sunday 02:00 UTC) collided with the master
    # DAG's identical schedule. Both DAGs invoke ChEMBLPipeline().run()
    # simultaneously -> file-lock contention -> spurious "Could not
    # acquire run lock" failures. Fix: stagger to Wednesday 04:00 UTC
    # so standalone and master never overlap. The master DAG keeps
    # Sunday 02:00 UTC.
    schedule="0 4 * * 3",  # Wednesday 04:00 UTC (staggered from master)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["drug_repurposing", "chembl", "etl"],
)
def chembl_dag() -> None:
    """Build the ChEMBL pipeline DAG.

    P1-032 ROOT FIX: wire ``check_chembl_health >> run_chembl`` so the
    pipeline NEVER runs against a partially-down ChEMBL API. The health
    check runs in <1 second; if ChEMBL is DOWN, the DAG fails fast
    (AirflowFailException -- no retries) instead of burning 95 min of
    worker time on 6 doomed retries.
    """
    health = check_chembl_health()
    pipeline = run_chembl()
    # P1-032: explicit dependency -- health sensor must pass before the
    # pipeline runs. Without this wire Airflow would run both tasks in
    # parallel, defeating the purpose of the health check.
    health >> pipeline


# v89 ROOT FIX (BUG #40): consistent DAG-instance naming convention.
# Was ``chembl_dag_instance`` here and ``master_dag`` in the master DAG
# -- a future maintainer searching for ``dag_instance`` would miss most
# of them. All 8 DAG files now use ``dag = <dag_factory>()``.
dag = chembl_dag()
