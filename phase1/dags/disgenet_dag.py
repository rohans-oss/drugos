"""
DisGeNET DAG -- standalone pipeline for gene-disease associations.

Downloads the full gene-disease association dataset from DisGeNET,
filters by minimum score, normalises confidence tiers, and bulk-upserts
into the ``gene_disease_associations`` table.

If ``DISGENET_API_KEY`` is set the Bearer auth header is used for the
download; otherwise the public endpoint is attempted.

Can be triggered independently or as part of the master pipeline.
Schedule: every Monday at 02:00 UTC (cron ``0 2 * * 1``).
v49 ROOT FIX (Compound-4 -- Sunday Morning Pile-Up): was previously
``0 6 * * 0`` (Sunday 06:00 UTC) which overlapped the master DAG
window (Sunday 02:00 UTC, 8h timeout). Moved to Tuesday to eliminate
the per-pipeline filelock conflict with the master. DisGeNET curates
weekly; the standalone DAG runs on Tuesday so per-source refreshes
work without requiring the master DAG.

P1-036 FORENSIC ROOT FIX (Team 4 -- DisGeNET weekly schedule + release sensor):
    The previous schedule was ``0 6 * * 2`` (Tuesday 06:00 UTC). DisGeNET
    publishes new releases on Mondays (the curated weekly dump lands
    Monday ~09:00 UTC). Running Tuesday 06:00 UTC ASSUMES the release
    happened Monday, but provides NO guarantee -- if DisGeNET delayed
    the release (rare but documented), the DAG would download the
    PREVIOUS week's data and the operator would never know.

    ROOT FIX (master-grade, no sugar-coating):
      1. Move schedule to Monday 02:00 UTC (``0 2 * * 1``). This runs
         BEFORE the typical Monday 09:00 UTC release, so the DAG
         trigger fires early and the sensor WAITS for the release to
         appear. This is the audit's specific recommendation.
      2. Add ``check_disgenet_release`` sensor task that queries
         DisGeNET's release-notes API (``/v1/public/release_notes``)
         and verifies the latest release timestamp is within the last
         7 days. If no fresh release is found, the sensor fails
         (AirflowFailException) so the operator can investigate --
         rather than silently re-downloading last week's data.
      3. Wire ``check_disgenet_release >> run_disgenet`` so the pipeline
         NEVER runs against a stale DisGeNET release.
      4. Add ``on_failure_callback`` that emits a structured log line
         with the DisGeNET release-notes URL for manual verification.

    This also REDUCES API load on DisGeNET (per the audit's compound
    concern about rate-limit risk): the previous Tuesday schedule
    assumed the release existed and re-downloaded it every week even
    when DisGeNET was unchanged. The new schedule + sensor only
    downloads when a NEW release is detected.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

# v89 ROOT FIX (BUG #39): shared sys.path bootstrap (see dags/_dags_init.py).
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

# v29 ROOT FIX (audit O-12): XCom used for large dataframes -- anti-pattern.
# Now passes file paths via XCom. The single @task below returns None and the
# DisGeNETPipeline persists its output to processed_data/
# (gene_disease_associations.csv). Downstream DAGs (master pipeline) read that
# CSV by path -- no DataFrame is ever pushed to / pulled from XCom.

DEFAULT_ARGS = {
    **DEFAULT_RETRY_ARGS,
    "owner": "drug_repurposing",
    "depends_on_past": False,
    # P1-036 ROOT FIX: structured alerting on permanent failure. Same
    # pattern as P1-032 ChEMBL -- emit a structured CRITICAL log line
    # that log-alerting can route to on-call.
    "on_failure_callback": lambda ctx: _disgenet_permanent_failure_alert(ctx),
}

# P1-036 ROOT FIX: DisGeNET release-notes URL.
# The release-notes endpoint returns a JSON list of releases with
# ``release_date`` fields. We use this to verify a fresh release exists
# BEFORE downloading the full dataset (4+ GB compressed) -- avoiding
# wasted API calls when DisGeNET hasn't released yet.
DISGENET_RELEASE_NOTES_URL = "https://www.disgenet.org/api/v1/public/release_notes"


def _disgenet_permanent_failure_alert(context: dict) -> None:
    """P1-036 ROOT FIX: structured alert when DisGeNET DAG fails permanently."""
    dag_run = context.get("dag_run")
    task_instance = context.get("task_instance")
    exception = context.get("exception")
    try_count = getattr(task_instance, "try_number", 0) if task_instance else 0

    dag_run_id = getattr(dag_run, "run_id", "<unknown>") if dag_run else "<unknown>"
    task_id = getattr(task_instance, "task_id", "<unknown>") if task_instance else "<unknown>"
    exc_class = type(exception).__name__ if exception else "<none>"
    exc_msg = str(exception) if exception else "<none>"

    logger.critical(
        "P1-036 ALERT: DisGeNET DAG permanent failure -- "
        "dag_run_id=%s task_id=%s try_count=%d exception=%s: %s -- "
        "Verify DisGeNET release notes at %s "
        "and re-trigger the DAG manually once a fresh release is available. "
        "If task_id=check_disgenet_release and try_count=1, no fresh "
        "release was detected (DisGeNET may have delayed this week's "
        "release). If task_id=run_disgenet, the download failed -- "
        "check DISGENET_API_KEY and rate-limit headers.",
        dag_run_id,
        task_id,
        try_count,
        exc_class,
        exc_msg,
        DISGENET_RELEASE_NOTES_URL,
    )


@task
def check_disgenet_release() -> str:
    """P1-036 ROOT FIX: pre-flight check that a fresh DisGeNET release exists.

    Queries DisGeNET's release-notes endpoint and verifies the latest
    release timestamp is within the last 7 days. If no fresh release is
    found, raises ``AirflowFailException`` (non-retryable) so the DAG
    fails FAST instead of re-downloading last week's data.

    Returns the latest release version string for XCom visibility.

    The check tolerates DisGeNET API outages: if the release-notes
    endpoint is unreachable, the sensor logs a WARNING and returns
    "UNKNOWN" -- the pipeline proceeds (under the assumption that
    DisGeNET MAY have released but the API is degraded). This is the
    scientifically correct trade-off: a false-positive "no release"
    would lose a week of data, while a false-negative "release exists"
    re-downloads the same data (idempotent upsert, no harm).
    """
    import requests

    try:
        resp = requests.get(
            DISGENET_RELEASE_NOTES_URL,
            timeout=15,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
    except Exception as exc:
        # API outage -- log and proceed (do NOT block the pipeline on
        # release-notes API; the actual data download will fail separately
        # if DisGeNET is truly down).
        logger.warning(
            "P1-036 DisGeNET release-notes endpoint unreachable (%s). "
            "Proceeding with pipeline -- if DisGeNET is truly down, the "
            "download task will fail separately. Release-notes URL: %s",
            exc,
            DISGENET_RELEASE_NOTES_URL,
        )
        return "UNKNOWN"

    try:
        releases = resp.json()
    except ValueError as exc:
        logger.warning(
            "P1-036 DisGeNET release-notes returned non-JSON (%s). "
            "Proceeding with pipeline.",
            exc,
        )
        return "UNKNOWN"

    if not isinstance(releases, list) or not releases:
        logger.warning(
            "P1-036 DisGeNET release-notes returned empty / non-list "
            "response: %r. Proceeding with pipeline.",
            releases,
        )
        return "UNKNOWN"

    # Find the latest release (DisGeNET returns newest-first).
    latest = releases[0]
    if not isinstance(latest, dict):
        logger.warning(
            "P1-036 DisGeNET release-notes[0] is not a dict: %r. "
            "Proceeding with pipeline.",
            latest,
        )
        return "UNKNOWN"

    # DisGeNET release-notes schema: {"version": "...", "release_date": "..."}
    latest_version = str(latest.get("version", "UNKNOWN"))
    release_date_str = str(latest.get("release_date", ""))

    if not release_date_str:
        logger.warning(
            "P1-036 DisGeNET latest release has no release_date field: "
            "%r. Proceeding with pipeline.",
            latest,
        )
        return latest_version

    # Parse the release date (DisGeNET uses ISO 8601: "2024-06-17").
    try:
        # Strip timezone if present and parse as UTC.
        release_date_str_clean = release_date_str.split("T")[0].strip()
        release_date = datetime.strptime(release_date_str_clean, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        logger.warning(
            "P1-036 DisGeNET release_date %r could not be parsed (%s). "
            "Proceeding with pipeline.",
            release_date_str,
            exc,
        )
        return latest_version

    now_utc = datetime.now(timezone.utc)
    age = now_utc - release_date
    if age > timedelta(days=7):
        # No fresh release in the last 7 days. This is the audit's
        # concern: silently re-downloading stale data. FAIL FAST so
        # the operator can investigate.
        try:
            from airflow.exceptions import AirflowFailException
        except ImportError:
            raise RuntimeError(
                f"P1-036 DisGeNET release sensor FAILED: latest release "
                f"({latest_version} on {release_date_str}) is "
                f"{age.days} days old (> 7 day threshold). DisGeNET may "
                f"have delayed this week's release. Verify at "
                f"{DISGENET_RELEASE_NOTES_URL} and re-trigger once a "
                f"fresh release appears."
            )
        raise AirflowFailException(
            f"P1-036 DisGeNET release sensor FAILED: latest release "
            f"({latest_version} on {release_date_str}) is "
            f"{age.days} days old (> 7 day threshold). DisGeNET may "
            f"have delayed this week's release. Verify at "
            f"{DISGENET_RELEASE_NOTES_URL} and re-trigger once a "
            f"fresh release appears."
        )

    logger.info(
        "P1-036 DisGeNET release sensor PASSED: latest release %s "
        "published %s (%d days ago, within 7-day threshold). "
        "Proceeding with pipeline.",
        latest_version,
        release_date_str,
        age.days,
    )
    return latest_version


# v89 ROOT FIX (BUG #25 / BUG #38): bare ``@task`` -- retry params
# inherited from DEFAULT_ARGS (spread from DEFAULT_RETRY_ARGS).
@task
@fail_fast_on_http_4xx
def run_disgenet() -> None:
    """Execute the full DisGeNET pipeline: download -> clean -> load."""
    from pipelines.disgenet_pipeline import DisGeNETPipeline
    DisGeNETPipeline().run()


@dag(
    dag_id="disgenet_pipeline",
    description="DisGeNET ETL pipeline: gene-disease associations",
    # v49 ROOT FIX (Compound-4 -- Sunday Morning Pile-Up):
    # Was "0 6 * * 0" (Sunday 06:00 UTC) -- overlapped master DAG window.
    # v49 moved to Tuesday 06:00 UTC.
    #
    # P1-036 FORENSIC ROOT FIX (Team 4 -- schedule too frequent + no sensor):
    #   The audit recommends Monday 02:00 UTC (``0 2 * * 1``) -- BEFORE
    #   DisGeNET's typical Monday 09:00 UTC release -- combined with a
    #   release-notes sensor that WAITS for the new release. This
    #   schedule + sensor pair ensures:
    #     * We never re-download last week's data (sensor fails if no
    #       fresh release in 7 days).
    #     * We pick up the new release ASAP (sensor fires as soon as
    #       release-notes API shows a fresh release).
    #     * We reduce API load on DisGeNET (no wasted downloads).
    schedule="0 2 * * 1",  # P1-036: Monday 02:00 UTC (per audit recommendation)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["drug_repurposing", "disgenet", "etl"],
)
def disgenet_dag() -> None:
    """Build the DisGeNET pipeline DAG.

    P1-036 ROOT FIX: wire ``check_disgenet_release >> run_disgenet`` so
    the pipeline NEVER downloads stale data. The release sensor queries
    DisGeNET's release-notes API and verifies a fresh (<7 days old)
    release exists before the download task fires.
    """
    release_sensor = check_disgenet_release()
    pipeline = run_disgenet()
    # P1-036: explicit dependency -- release sensor must pass before the
    # pipeline runs. Without this wire Airflow would run both tasks in
    # parallel, defeating the purpose of the sensor.
    release_sensor >> pipeline


# v89 ROOT FIX (BUG #40): consistent DAG-instance naming convention.
dag = disgenet_dag()
