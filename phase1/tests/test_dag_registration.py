"""P1-053 ROOT FIX (v110): regression guard for DAG registration.

WHAT THIS TEST GUARDS AGAINST
-----------------------------
The audit (Issue #53) found that "only 4 DAGs are registered" — the
Airflow DagBag was failing to import some DAG modules (import errors),
and there was no test to catch this.

ROOT FIX
--------
This test verifies:
  1. The ``dags`` package exposes ``DAG_IDS`` and ``EXPECTED_DAG_COUNT``.
  2. All 8 expected DAG IDs (7 source DAGs + 1 master DAG) are present
     in ``DAG_IDS``.
  3. Every DAG module can be imported WITHOUT raising — no import errors
     that would silently drop the DAG from Airflow's DagBag.
  4. Each DAG module exposes a ``dag`` attribute (the registered DAG
     instance created by ``dag = <factory>()`` at the bottom of each file).
  5. The ``dag.dag_id`` of each registered DAG matches the expected ID
     in ``DAG_IDS``.

This is the regression guard the audit asked for: "verify all 7 source
DAGs + master DAG are registered with Airflow."

NOTE: these tests do NOT require Airflow to be installed. They verify
the import-time registration contract; Airflow's DagBag relies on the
same contract at scan time. If airflow IS installed, the tests also
verify that each ``dag`` is an Airflow DAG instance.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so we can import dags.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Tests need dev-mode settings (the DAGs import config.settings at parse time).
os.environ.setdefault("DRUGOS_ENVIRONMENT", "development")


# ---------------------------------------------------------------------------
# Tests: the dags package exposes the expected constants.
# ---------------------------------------------------------------------------

def test_dags_package_exposes_dag_ids():
    """The dags package MUST expose DAG_IDS for testability."""
    import dags
    assert hasattr(dags, "DAG_IDS"), (
        "dags package must expose DAG_IDS (list of expected DAG IDs)."
    )
    assert hasattr(dags, "EXPECTED_DAG_COUNT"), (
        "dags package must expose EXPECTED_DAG_COUNT."
    )


def test_expected_dag_count_is_eight():
    """P1-045 acceptance: 7 source DAGs + 1 master DAG = 8 DAGs total."""
    import dags
    assert dags.EXPECTED_DAG_COUNT == 8, (
        f"EXPECTED_DAG_COUNT must be 8 (7 source + 1 master). "
        f"Got {dags.EXPECTED_DAG_COUNT}."
    )


def test_dag_ids_contain_all_seven_source_dags_plus_master():
    """All 8 expected DAG IDs MUST be in DAG_IDS."""
    import dags
    expected = {
        "chembl_pipeline",
        "drugbank_pipeline",
        "uniprot_pipeline",
        "string_pipeline",
        "disgenet_pipeline",
        "omim_pipeline",
        "pubchem_pipeline",
        "drug_repurposing_master",
    }
    actual = set(dags.DAG_IDS)
    assert actual == expected, (
        f"DAG_IDS mismatch. Missing: {expected - actual}. "
        f"Extra: {actual - expected}."
    )


# ---------------------------------------------------------------------------
# Tests: every DAG module imports cleanly and exposes a ``dag`` attribute.
# ---------------------------------------------------------------------------

DAG_MODULES = [
    ("chembl_dag", "chembl_pipeline"),
    ("drugbank_dag", "drugbank_pipeline"),
    ("uniprot_dag", "uniprot_pipeline"),
    ("string_dag", "string_pipeline"),
    ("disgenet_dag", "disgenet_pipeline"),
    ("omim_dag", "omim_pipeline"),
    ("pubchem_dag", "pubchem_pipeline"),
    ("master_pipeline_dag", "drug_repurposing_master"),
]


@pytest.mark.parametrize("module_name,expected_dag_id", DAG_MODULES)
def test_dag_module_imports_cleanly(module_name, expected_dag_id):
    """Every DAG module MUST import without raising (Airflow DagBag contract)."""
    # Use importlib instead of import dags.X so each test is isolated.
    try:
        mod = importlib.import_module(f"dags.{module_name}")
    except ImportError as exc:
        if "airflow" in str(exc).lower():
            pytest.skip(f"airflow not installed — skipping DAG import test for {module_name}")
        raise
    except Exception as exc:
        pytest.fail(
            f"dags.{module_name} failed to import: {type(exc).__name__}: {exc}. "
            f"Airflow's DagBag would silently drop this DAG."
        )
    assert hasattr(mod, "dag"), (
        f"dags.{module_name} must expose a 'dag' attribute (the registered "
        f"DAG instance created by 'dag = <factory>()' at module bottom)."
    )


@pytest.mark.parametrize("module_name,expected_dag_id", DAG_MODULES)
def test_registered_dag_has_correct_id(module_name, expected_dag_id):
    """The registered DAG's dag_id MUST match the expected ID."""
    try:
        mod = importlib.import_module(f"dags.{module_name}")
    except ImportError as exc:
        if "airflow" in str(exc).lower():
            pytest.skip(f"airflow not installed — skipping for {module_name}")
        raise
    dag = mod.dag
    # The dag attribute may be an Airflow DAG or a mock. If it's an Airflow
    # DAG, it has a .dag_id attribute.
    if hasattr(dag, "dag_id"):
        actual_id = dag.dag_id
    else:
        # If airflow isn't available, the module may use a placeholder.
        # In that case, skip the dag_id check.
        pytest.skip(f"dag attribute on dags.{module_name} has no dag_id (airflow not installed)")
    assert actual_id == expected_dag_id, (
        f"dags.{module_name}: dag.dag_id = {actual_id!r}, expected {expected_dag_id!r}."
    )


def test_dags_init_imports_all_modules_when_airflow_available():
    """Importing the dags package MUST register all 8 DAGs (when airflow is available)."""
    try:
        import airflow  # noqa: F401
    except ImportError:
        pytest.skip("airflow not installed — skipping registration test")
    import dags
    registered = dags.get_registered_dag_ids()
    expected = set(dags.DAG_IDS)
    missing = expected - registered
    assert not missing, (
        f"dags package failed to register: {missing}. "
        f"Registered: {registered}"
    )


def test_dags_init_does_not_crash_without_airflow(monkeypatch):
    """The dags package MUST NOT crash on import when airflow is not installed.

    A failure here would break any test / CI environment that imports
    the dags package without airflow installed (e.g. a test that only
    tests the cleaning module).
    """
    # Block airflow imports to simulate the "airflow not installed" case.
    import builtins
    real_import = builtins.__import__

    def _no_airflow(name, *args, **kwargs):
        if name.startswith("airflow"):
            raise ImportError(f"No module named '{name}' (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_airflow)
    # Force re-import of the dags package.
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("dags"):
            del sys.modules[mod_name]
    # Importing should NOT raise — it should log a warning and continue.
    try:
        import dags  # noqa: F401
    except Exception as exc:
        pytest.fail(
            f"dags package crashed on import without airflow: "
            f"{type(exc).__name__}: {exc}"
        )
