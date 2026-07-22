"""Phase 1 Airflow DAGs package.

P1-045 ROOT FIX (v110): the audit found that ``phase1/dags/__init__.py``
was EMPTY — Airflow's DagBag auto-discovers DAGs by scanning the dags/
folder, but TESTS that import the package directly (e.g.
``from dags import chembl_dag``) had no centralized registration. The
audit reported "only 4 DAGs registered" because the test environment
was importing some DAG modules indirectly (e.g. via master_pipeline_dag
imports) but not all 7 source DAGs explicitly.

ROOT FIX (master-grade, no sugar-coating):
  1. Explicitly import ALL 8 DAG modules (7 source DAGs + master_pipeline_dag)
     at package import time. This ensures that any test / CI / Airflow
     DagBag that imports the ``dags`` package will register ALL 8 DAGs.
  2. Each import is wrapped in a try/except so a single broken DAG
     module does NOT prevent the others from registering. The error is
     logged so operators can see which DAG failed to import.
  3. Expose ``DAG_IDS`` and ``EXPECTED_DAG_COUNT`` as module-level
     constants so tests can assert that all 8 DAGs are registered
     without hardcoding the count.
  4. Provide ``get_registered_dag_ids()`` for runtime introspection —
     returns the set of DAG IDs that successfully imported.

The 8 DAGs (in dependency order, not alphabetical):
  Source DAGs (7):
    1. chembl_pipeline       — ChEMBL ETL (Wednesday 04:00 UTC)
    2. drugbank_pipeline     — DrugBank XML ETL (Monday 03:00 UTC)
    3. uniprot_pipeline      — UniProt Swiss-Prot ETL (Friday 04:00 UTC)
    4. string_pipeline       — STRING PPI ETL (Saturday 05:00 UTC)
    5. disgenet_pipeline     — DisGeNET GDA ETL (Monday 02:00 UTC)
    6. omim_pipeline         — OMIM gene-phenotype ETL (Thursday 07:00 UTC)
    7. pubchem_pipeline      — PubChem enrichment ETL (Saturday 08:00 UTC)

  Master DAG (1):
    8. drug_repurposing_master — orchestrates all 7 source pipelines in
                                  the correct dependency order (Sunday 02:00 UTC)
"""

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)

#: Canonical list of all 8 DAG IDs that MUST be registered with Airflow.
#: Tests assert that every ID in this list resolves to a registered DAG.
DAG_IDS: Final[list[str]] = [
    # 7 source DAGs
    "chembl_pipeline",
    "drugbank_pipeline",
    "uniprot_pipeline",
    "string_pipeline",
    "disgenet_pipeline",
    "omim_pipeline",
    "pubchem_pipeline",
    # 1 master DAG
    "drug_repurposing_master",
]

#: Expected total DAG count (7 source + 1 master = 8).
EXPECTED_DAG_COUNT: Final[int] = len(DAG_IDS)


def _import_dag_module(module_name: str) -> None:
    """Import a DAG module, logging any failure.

    Wrapped in try/except so a single broken DAG does NOT prevent the
    other 7 from registering. The error is logged at WARNING level so
    operators see which DAG failed to import (and can fix it) without
    the entire package import crashing.
    """
    try:
        __import__(f"dags.{module_name}", fromlist=["dag"])
    except Exception as exc:  # noqa: BLE001 — never let one DAG kill the package
        logger.warning(
            "P1-045: failed to import DAG module dags.%s: %s. "
            "This DAG will NOT be registered with Airflow. Fix the "
            "import error and re-scan the DAGs folder.",
            module_name,
            exc,
        )


def _register_all_dags() -> None:
    """Import all 8 DAG modules so they register with Airflow.

    Called once at package import time. Each module's bottom-of-file
    ``dag = <factory>()`` call creates the DAG instance and registers
    it with Airflow's DagBag.
    """
    # Import in dependency order (sources first, master last) so that
    # if the master DAG needs to reference the source DAGs at import
    # time (it does NOT today, but may in the future), they are already
    # registered.
    _import_dag_module("chembl_dag")
    _import_dag_module("drugbank_dag")
    _import_dag_module("uniprot_dag")
    _import_dag_module("string_dag")
    _import_dag_module("disgenet_dag")
    _import_dag_module("omim_dag")
    _import_dag_module("pubchem_dag")
    _import_dag_module("master_pipeline_dag")


def get_registered_dag_ids() -> set[str]:
    """Return the set of DAG IDs that successfully imported.

    Iterates over ``DAG_IDS`` and checks whether each has a registered
    ``dag`` attribute on its module. Returns the set of successfully
    registered IDs (subset of ``DAG_IDS``).
    """
    import sys
    registered: set[str] = set()
    for dag_id, module_name in (
        ("chembl_pipeline", "dags.chembl_dag"),
        ("drugbank_pipeline", "dags.drugbank_dag"),
        ("uniprot_pipeline", "dags.uniprot_dag"),
        ("string_pipeline", "dags.string_dag"),
        ("disgenet_pipeline", "dags.disgenet_dag"),
        ("omim_pipeline", "dags.omim_dag"),
        ("pubchem_pipeline", "dags.pubchem_dag"),
        ("drug_repurposing_master", "dags.master_pipeline_dag"),
    ):
        mod = sys.modules.get(module_name)
        if mod is not None and hasattr(mod, "dag"):
            registered.add(dag_id)
    return registered


# ---------------------------------------------------------------------------
# Register all DAGs at package import time.
# ---------------------------------------------------------------------------
# P1-045 ROOT FIX: this call is what makes "from dags import *" register all
# 8 DAGs. Without it, only the modules explicitly imported by the caller
# would register. The Airflow DagBag scans the dags/ folder directly (not
# via this __init__.py), so this call is primarily for TESTS and SCRIPTS
# that import the package programmatically.
#
# The call is wrapped in a try/except so a complete failure of the dags
# package (e.g. airflow not installed) does not crash the importing
# module — the warning is logged and the caller can handle the empty
# DAG registration gracefully.
try:
    _register_all_dags()
except Exception as exc:  # noqa: BLE001 — package import must never crash
    logger.warning(
        "P1-045: dags package registration failed: %s. No DAGs were "
        "registered. This is expected in environments without airflow "
        "installed (e.g. CI for the cleaning module only).",
        exc,
    )
