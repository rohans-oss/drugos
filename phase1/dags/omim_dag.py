"""
OMIM DAG -- standalone pipeline for OMIM gene-phenotype mappings.

Downloads morbidmap.txt (if OMIM_API_KEY is set) or uses the OMIM API
with pagination, parses confirmed gene-phenotype associations
(mapping_key IN [3, 4] per OMIM_MAPPING_KEYS_INCLUDE default), and
bulk-upserts into the ``gene_disease_associations`` table.

v39 ROOT FIX (P1 #52): updated docstring. The previous docstring said
"mapping_key == 3" but OMIM_MAPPING_KEYS_INCLUDE defaults to [3, 4]
(mapping_key 3 = molecular basis known, mapping_key 4 = contiguous
gene deletion/duplication syndrome). Both are loaded.

Can be triggered independently or as part of the master pipeline.
Schedule: 15th of every month at 07:00 UTC

v89 FORENSIC ROOT FIX (BUG #8 P1 -- Sunday Morning Pile-Up):
  The previous schedule ``0 7 1 * *`` (1st of month 07:00 UTC) could
  fall on a Sunday -- colliding with the master DAG (Sunday 02:00 UTC,
  up to 7h runtime). When the 1st falls on Sunday, both the master DAG
  and this standalone DAG invoke the SAME pipelines (OMIMPipeline().run()),
  writing to the SAME CSV files and DB tables. This causes file-lock
  contention and DB write conflicts (~12 collisions per year).
  ROOT FIX: move to the 15th of the month. The 15th is a fixed
  day-of-month that is independent of the day-of-week, so it NEVER
  systematically collides with the Sunday master DAG schedule. The
  ~2-week offset from the master DAG also gives operators a clean
  mid-month refresh point without overlapping the weekly cadence.
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
# OMIMPipeline persists its output to processed_data/
# (omim_gene_disease_associations.csv). Downstream DAGs (master pipeline) read
# that CSV by path -- no DataFrame is ever pushed to / pulled from XCom.

DEFAULT_ARGS = {
    **DEFAULT_RETRY_ARGS,
    "owner": "drug_repurposing",
    "depends_on_past": False,
}


# v83 FORENSIC ROOT FIX (P2-13) + v89 ROOT FIX (BUG #25 / BUG #38):
# bare ``@task`` -- all retry / timeout / backoff params are inherited
# from ``DEFAULT_ARGS`` (spread from ``DEFAULT_RETRY_ARGS``). The
# ``@fail_fast_on_http_4xx`` decorator is retained (it's the actual
# functional enhancement, not a redundant override). All 7 standalone
# DAGs now use this same pattern.
@task
@fail_fast_on_http_4xx
def run_omim() -> None:
    """Execute the full OMIM pipeline: download -> clean -> load."""
    from pipelines.omim_pipeline import OMIMPipeline
    OMIMPipeline().run()


@dag(
    dag_id="omim_pipeline",
    description="OMIM ETL pipeline: gene-phenotype mappings",
    # v29 ROOT FIX (audit O-11): was schedule=None (dead). Now scheduled.
    # OMIM releases new morbidmap entries monthly; standalone DAG runs
    # so ad-hoc / per-source refreshes work without requiring the master DAG.
    #
    # P1-047 FORENSIC ROOT FIX (Team 4 -- DAG schedule collision):
    # The previous schedule was ``0 7 15 * *`` (15th of month at 07:00 UTC).
    # When the 15th fell on a Sunday, this DAG collided with the master
    # DAG (Sun 02:00 UTC) and UniProt/STRING DAGs (15th at 04:00/05:00 UTC).
    # All four DAGs write to the same ``drugs``, ``proteins``,
    # ``gene_disease_associations`` tables -> DB contention, lock timeouts,
    # possible deadlocks.
    #
    # ROOT FIX: move to weekly Thursday at 07:00 UTC. OMIM's loader is
    # idempotent (it checks for upstream changes and no-ops if nothing
    # changed), so the weekly cadence is safe. Thursday NEVER collides
    # with the master's Sunday window or any other standalone DAG.
    schedule="0 7 * * 4",  # Every Thursday at 07:00 UTC (P1-047 root fix)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["drug_repurposing", "omim", "etl"],
)
def omim_dag() -> None:
    """Build the OMIM pipeline DAG."""
    run_omim()


# v89 ROOT FIX (BUG #40): consistent DAG-instance naming convention.
dag = omim_dag()
