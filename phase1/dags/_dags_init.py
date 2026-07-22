"""Shared sys.path bootstrap for all Airflow DAG files (v89 ROOT FIX BUG #39).

PROBLEM (BUG #39)
-----------------
The ``_PROJECT_ROOT`` sys.path insertion block was duplicated verbatim
in ALL 8 DAG files (master_pipeline_dag + 7 standalone DAGs):

    _PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

If the project structure ever changes (e.g. ``dags/`` moves), all 8
files must be updated in lock-step -- a classic copy-paste maintenance
hazard. A single missed update produces ``ImportError`` at DAG parse
time in only SOME DAGs, with no compile-time signal.

ROOT FIX (master-grade, no sugar-coating)
-----------------------------------------
Extract the bootstrap into THIS shared module. Every DAG file now
imports it AND explicitly calls it at module top:

    from dags._dags_init import ensure_project_root
    ensure_project_root()  # explicit side effect (P1-050 root fix)

P1-050 FORENSIC ROOT FIX (Team 4 -- hidden module-level side effect):
The previous version called ``ensure_project_root()`` at MODULE LEVEL
(no ``if __name__ == "__main__":`` guard). Any
``from dags._dags_init import ensure_project_root`` triggered the side
effect: ``sys.path.insert(0, _PROJECT_ROOT)``. This mutated ``sys.path``
for the ENTIRE process. Test isolation was broken -- a test that imported
``dags._dags_init`` to test ``ensure_project_root()`` directly modified
``sys.path`` for all subsequent tests in the same process. Tests that
asserted ``sys.path`` did NOT contain a specific path failed mysteriously
after a ``_dags_init`` import.

ROOT FIX: REMOVE the module-level call. Each DAG file MUST now
explicitly call ``ensure_project_root()`` at the top of its module body
(after the import). This makes the side effect EXPLICIT per-DAG, not
hidden in an imported module. The function itself is unchanged -- only
the auto-invocation is removed.

This module has NO Airflow dependency and can be imported in any
Python context (CI, pytest, manual inspection) WITHOUT side effects.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The Phase 1 project root (parent of the ``dags/`` directory). All
#: pipeline / config / entity_resolution imports resolve from here.
_PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent)


def ensure_project_root() -> str:
    """Insert ``_PROJECT_ROOT`` at the front of ``sys.path`` (idempotent).

    Returns the project-root string so callers can reuse it without
    recomputing the path.

    P1-050 ROOT FIX: this function is NO LONGER auto-invoked at module
    import time. Each DAG file MUST explicitly call it at the top of
    its module body. This makes the ``sys.path`` side effect explicit
    per-DAG rather than hidden in an imported module -- fixing the test
    isolation breakage.
    """
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    return _PROJECT_ROOT


def require_airflow() -> tuple:
    """P1-034 ROOT FIX: import airflow lazily, with a CLEAR error if missing.

    The audit found that DAG files do ``from airflow.decorators import
    dag, task`` at module top. If airflow is not installed (CI env that
    only tests cleaning logic), the import crashes with ``ModuleNotFoundError``
    -- but ``test_dag_structure.py`` used to ``pytest.importorskip('airflow')``
    which SILENTLY SKIPPED all DAG structure validation. Result: DAG
    structure bugs (like P1-031 missing task dependencies) went
    undetected in CI.

    This helper provides a SINGLE place where the airflow import is
    resolved. It returns the ``(dag, task)`` tuple on success. On
    failure, it raises ``RuntimeError`` with a clear remediation message
    naming:
      * The exact ``pip install`` command (``pip install apache-airflow>=2.8.0``)
      * The requirements file path (``requirements.txt``)
      * The reason the import is required (DAG files need it)

    DAG files SHOULD continue to use ``from airflow.decorators import
    dag, task`` directly (top-level import is fine NOW that airflow is
    in requirements.txt). This helper is for TESTS and SCRIPTS that
    want to verify airflow is importable before proceeding, with a
    better error than ``ModuleNotFoundError``.

    Usage in tests::

        from dags._dags_init import require_airflow
        dag, task = require_airflow()  # raises RuntimeError if airflow missing

    Returns
    -------
    tuple
        ``(dag, task)`` -- the two TaskFlow API decorators.

    Raises
    ------
    RuntimeError
        If ``apache-airflow`` is not installed. The error message
        includes the exact ``pip install`` remediation.
    """
    try:
        from airflow.decorators import dag, task
        return dag, task
    except ImportError as exc:
        raise RuntimeError(
            "P1-034: apache-airflow is NOT installed. The Drug Repurposing "
            "platform's DAG files (phase1/dags/*.py) require "
            "apache-airflow>=2.8.0 at import time. Without it, DAG "
            "structure validation tests are SILENTLY SKIPPED -- which "
            "is how DAG bugs like P1-031 (missing task dependencies) "
            "shipped to production. Remediation: run "
            "'pip install -r requirements.txt' (which pins "
            "apache-airflow>=2.8.0) OR 'pip install apache-airflow>=2.8.0'. "
            "Then re-run the tests. Original ImportError: "
            f"{exc}"
        ) from exc


# P1-050 FORENSIC ROOT FIX (Team 4): the module-level
# ``ensure_project_root()`` call has been REMOVED. Each DAG file must
# now explicitly call ``ensure_project_root()`` at the top of its module
# body (after the import). This makes the ``sys.path`` side effect
# explicit per-DAG, not hidden in an imported module.
