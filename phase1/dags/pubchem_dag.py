"""
PubChem DAG -- standalone pipeline for PubChem drug enrichment.

Reads InChIKeys from the ``drugs`` table where ``pubchem_cid`` IS NULL,
batch-queries the PubChem PUG REST API for properties, and bulk-updates
the ``drugs`` table with retrieved molecular data.

Can be triggered independently or as part of the master pipeline.
Schedule: every Wednesday at 08:00 UTC (cron ``0 8 * * 3``).
v49 ROOT FIX (Compound-4 -- Sunday Morning Pile-Up): was previously
``0 8 * * 0`` (Sunday 08:00 UTC) which overlapped the master DAG
window (Sunday 02:00 UTC, 8h timeout). Moved to Wednesday to
eliminate the per-pipeline filelock conflict with the master.
PubChem updates compound properties continuously; the standalone DAG
runs every Wednesday so ad-hoc / per-source refreshes work without
requiring the master DAG.

P1-037 FORENSIC ROOT FIX (Team 4 -- PubChem FTP + no resume + 1h timeout):
    The audit's original description said the DAG downloads PubChem's
    CID-Synonym file via FTP. The current code already uses HTTPS
    (PUBCHEM_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pubchem/...")
    and the bulk downloader in ``pipelines/_v50_downloaders.py`` already
    supports HTTP Range for resume. So the HTTPS + resumable parts are
    already in place.

    HOWEVER, the audit's broader concern remains valid:
      1. The DAG's task timeout was inherited from DEFAULT_RETRY_ARGS
         (4h execution_timeout) but NOT explicitly set on the @task
         decorator -- a future change to DEFAULT_RETRY_ARGS could
         silently shorten it. The audit's specific recommendation is
         "increase the task timeout to 4 hours". Make it EXPLICIT.
      2. There is NO test that verifies the resumable download logic
         actually works -- if a future refactor removes the HTTP Range
         header handling, the bulk download would silently restart
         from byte 0 on every failure (wasting 4 GB of bandwidth per
         retry).
      3. The PubChem bulk download URL config (``PUBCHEM_FTP_BASE``)
         has a misleading name -- it's actually an HTTPS URL. A future
         maintainer might "fix" the name by changing it to a real FTP
         URL (``ftp://``), reintroducing the original audit issue.

    ROOT FIX (master-grade, no sugar-coating):
      1. Set ``execution_timeout=timedelta(hours=4)`` EXPLICITLY on the
         ``run_pubchem`` @task decorator. This is now self-documenting
         and immune to DEFAULT_RETRY_ARGS drift.
      2. Add ``check_pubchem_https`` pre-flight task that verifies the
         PubChem base URL is HTTPS (not FTP / plain HTTP). Fails fast
         if a misconfiguration introduces an FTP URL.
      3. Wire ``check_pubchem_https >> run_pubchem`` so the pipeline
         NEVER runs against an FTP URL.
      4. The resumable-download logic in ``_v50_downloaders.py`` is
         verified by a new regression test
         (``test_pubchem_resumable_download``) that mocks a partial
         download and asserts the Range header is sent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

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
# PubChemPipeline persists its output to processed_data/ (pubchem_enrichment.csv).
# Downstream DAGs (master pipeline) read that CSV by path -- no DataFrame is
# ever pushed to / pulled from XCom.

DEFAULT_ARGS = {
    **DEFAULT_RETRY_ARGS,
    "owner": "drug_repurposing",
    "depends_on_past": False,
}

# P1-037 ROOT FIX: explicit 4-hour task timeout for PubChem pipeline.
# The PubChem bulk SDF download is 4 GB compressed; even on a fast
# connection this takes 30-60 minutes, and the PUG REST enrichment loop
# (batched InChIKey queries) can take 2-3 hours for a full 10k-drug run.
# The audit's specific recommendation is "increase the task timeout to
# 4 hours". This constant makes the timeout self-documenting and immune
# to DEFAULT_RETRY_ARGS drift.
PUBCHEM_TASK_TIMEOUT = timedelta(hours=4)


@task
def check_pubchem_https() -> str:
    """P1-037 ROOT FIX: pre-flight check that PubChem base URL is HTTPS.

    Verifies ``PUBCHEM_FTP_BASE`` (the bulk-download URL) AND
    ``PUBCHEM_REST_BASE`` (the PUG REST API URL) are both HTTPS.
    A misconfiguration that introduces an ``ftp://`` or ``http://``
    URL would re-introduce the original audit issue (slow / unreliable
    FTP download, or unencrypted HTTP vulnerable to MITM).

    Returns the HTTPS-verified PubChem REST base URL for XCom visibility.

    Fails fast (AirflowFailException -- no retries) if either URL is
    not HTTPS. The operator must fix the env var / config and re-trigger.
    """
    try:
        from config.settings import PUBCHEM_FTP_BASE, PUBCHEM_REST_BASE
    except ImportError as exc:
        try:
            from airflow.exceptions import AirflowFailException
            raise AirflowFailException(
                f"P1-037 PubChem HTTPS check FAILED: could not import "
                f"PUBCHEM_FTP_BASE / PUBCHEM_REST_BASE from "
                f"config.settings ({exc})."
            ) from exc
        except ImportError:
            raise RuntimeError(
                f"P1-037 PubChem HTTPS check FAILED: could not import "
                f"PUBCHEM_FTP_BASE / PUBCHEM_REST_BASE from "
                f"config.settings ({exc})."
            ) from exc

    bad_urls: list[str] = []
    for label, url in (("PUBCHEM_FTP_BASE", PUBCHEM_FTP_BASE),
                        ("PUBCHEM_REST_BASE", PUBCHEM_REST_BASE)):
        if not url:
            bad_urls.append(f"{label} is empty")
            continue
        if not url.startswith("https://"):
            bad_urls.append(
                f"{label}={url!r} is NOT HTTPS (must start with "
                f"'https://'). FTP and plain HTTP are forbidden -- "
                f"FTP is slow / firewall-hostile and HTTP is MITM-vulnerable."
            )

    if bad_urls:
        msg = (
            "P1-037 PubChem HTTPS check FAILED: " + "; ".join(bad_urls) +
            ". Fix the env vars (PUBCHEM_FTP_BASE, PUBCHEM_REST_BASE) "
            "or config/settings.py and re-trigger."
        )
        try:
            from airflow.exceptions import AirflowFailException
            raise AirflowFailException(msg)
        except ImportError:
            raise RuntimeError(msg)

    logger.info(
        "P1-037 PubChem HTTPS check PASSED: PUBCHEM_FTP_BASE=%s, "
        "PUBCHEM_REST_BASE=%s. Both are HTTPS.",
        PUBCHEM_FTP_BASE,
        PUBCHEM_REST_BASE,
    )
    return PUBCHEM_REST_BASE


# v89 ROOT FIX (BUG #25 / BUG #38): bare ``@task`` -- retry params
# inherited from DEFAULT_ARGS (spread from DEFAULT_RETRY_ARGS).
# P1-037 ROOT FIX: set execution_timeout=4h EXPLICITLY on the @task
# decorator. The previous code inherited the 4h timeout from
# DEFAULT_RETRY_ARGS.execution_timeout -- a future change to
# DEFAULT_RETRY_ARGS could silently shorten it. The audit's specific
# recommendation is "increase the task timeout to 4 hours". This
# explicit setting makes the timeout self-documenting and immune to
# DEFAULT_RETRY_ARGS drift.
@task(execution_timeout=PUBCHEM_TASK_TIMEOUT)
@fail_fast_on_http_4xx
def run_pubchem() -> None:
    """Execute the full PubChem pipeline: download -> clean -> load.

    P1-037 ROOT FIX: ``execution_timeout=timedelta(hours=4)`` is set
    EXPLICITLY on the @task decorator (not inherited from
    DEFAULT_RETRY_ARGS). This is the audit's specific recommendation.
    The PubChem bulk SDF download (4 GB compressed) + the PUG REST
    enrichment loop (batched InChIKey queries for 10k drugs) can
    legitimately take 3-4 hours on a cold cache.
    """
    from pipelines.pubchem_pipeline import PubChemPipeline
    PubChemPipeline().run()


@dag(
    dag_id="pubchem_pipeline",
    description="PubChem ETL pipeline: drug enrichment via PUG REST API",
    # v49 ROOT FIX (Compound-4 -- Sunday Morning Pile-Up):
    # Was "0 8 * * 0" (Sunday 08:00 UTC) -- overlapped master DAG window.
    # Moved to Wednesday 08:00 UTC to eliminate the filelock conflict.
    #
    # P1-047 FORENSIC ROOT FIX (Team 4 -- ChEMBL/PubChem Wednesday overlap):
    # The previous schedule was ``0 8 * * 3`` (Wednesday 08:00 UTC), which
    # overlapped with ChEMBL's ``0 4 * * 3`` (Wednesday 04:00 UTC) -- both
    # write to the ``drugs`` table, causing DB contention. ROOT FIX: move
    # PubChem to Saturday 08:00 UTC (3 hours after STRING at Sat 05:00 --
    # no overlap). Saturday NEVER collides with the master's Sunday window.
    schedule="0 8 * * 6",  # Every Saturday at 08:00 UTC (P1-047 root fix)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["drug_repurposing", "pubchem", "etl"],
)
def pubchem_dag() -> None:
    """Build the PubChem pipeline DAG.

    P1-037 ROOT FIX: wire ``check_pubchem_https >> run_pubchem`` so the
    pipeline NEVER runs against an FTP or plain-HTTP URL. The HTTPS
    check fails fast (AirflowFailException -- no retries) if either
    PubChem URL is misconfigured.
    """
    https_check = check_pubchem_https()
    pipeline = run_pubchem()
    # P1-037: explicit dependency -- HTTPS check must pass before the
    # pipeline runs. Without this wire Airflow would run both tasks in
    # parallel, defeating the purpose of the HTTPS check.
    https_check >> pipeline


# v89 ROOT FIX (BUG #40): consistent DAG-instance naming convention.
dag = pubchem_dag()
