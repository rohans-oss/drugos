"""
UniProt DAG -- standalone pipeline for human reviewed (Swiss-Prot) protein data.

Downloads human reviewed proteins from the UniProt REST API using
cursor-based pagination, cleans and normalises records, and bulk-upserts
into the ``proteins`` table.

Can be triggered independently or as part of the master pipeline.
Schedule: 15th of every month at 04:00 UTC (cron ``0 4 15 * *``). UniProt's
Swiss-Prot human reviewed set updates monthly; the standalone DAG runs
on the 15th of every month so ad-hoc / per-source refreshes work without
requiring the master DAG.

v89 FORENSIC ROOT FIX (BUG #8 P1 -- Sunday Morning Pile-Up):
  Moved from ``0 4 1 * *`` (1st of month) to ``0 4 15 * *`` (15th of
  month). The 1st could fall on a Sunday, colliding with the master DAG
  (Sunday 02:00 UTC, up to 7h runtime). See omim_dag.py for full fix
  rationale. The 15th never systematically collides with Sunday.
"""

from __future__ import annotations

from datetime import datetime

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

# v29 ROOT FIX (audit O-12): XCom used for large dataframes -- anti-pattern.
# Now passes file paths via XCom. The single @task below returns None and the
# UniProtPipeline persists its output to processed_data/ (proteins.csv).
# Downstream DAGs (master pipeline) read that CSV by path -- no DataFrame is
# ever pushed to / pulled from XCom.

DEFAULT_ARGS = {
    **DEFAULT_RETRY_ARGS,
    "owner": "drug_repurposing",
    "depends_on_past": False,
}


# v89 ROOT FIX (BUG #25 / BUG #38): bare ``@task`` -- retry params
# inherited from DEFAULT_ARGS (spread from DEFAULT_RETRY_ARGS).
@task
@fail_fast_on_http_4xx
def run_uniprot() -> None:
    """Execute the full UniProt pipeline: download -> clean -> load."""
    from pipelines.uniprot_pipeline import UniProtPipeline
    UniProtPipeline().run()


@dag(
    dag_id="uniprot_pipeline",
    description="UniProt ETL pipeline: human reviewed protein data",
    # v29 ROOT FIX (audit O-11): was schedule=None (dead). Now scheduled.
    # UniProt's Swiss-Prot human reviewed set updates monthly; standalone DAG
    # runs so ad-hoc / per-source refreshes work without requiring the master DAG.
    #
    # P1-047 FORENSIC ROOT FIX (Team 4 -- DAG schedule collision):
    # The previous schedule was ``0 4 15 * *`` (15th of month at 04:00 UTC).
    # When the 15th fell on a Sunday, this DAG collided with the master
    # DAG (Sun 02:00 UTC) and OMIM/STRING DAGs (15th at 05:00/07:00 UTC).
    # All four DAGs write to the same ``proteins`` table -> DB contention.
    #
    # ROOT FIX: move to weekly Friday at 04:00 UTC. UniProt's loader is
    # idempotent. Friday NEVER collides with the master's Sunday window.
    schedule="0 4 * * 5",  # Every Friday at 04:00 UTC (P1-047 root fix)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["drug_repurposing", "uniprot", "etl"],
)
def uniprot_dag() -> None:
    """Build the UniProt pipeline DAG."""
    run_uniprot()


# v89 ROOT FIX (BUG #40): consistent DAG-instance naming convention.
dag = uniprot_dag()
