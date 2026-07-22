"""
P1-034 ROOT FIX: pytest bootstrap for airflow + sqlalchemy 2.0 compatibility.

PROBLEM
-------
The project's ``database/base.py`` requires sqlalchemy 2.0 (uses
``DeclarativeBase``). Airflow 2.9-2.10's ``TaskInstance`` model uses
legacy sqlalchemy 1.4 annotations that are NOT sqlalchemy 2.0
compatible (raises ``MappedAnnotationError`` at import time).

This is a fundamental dependency conflict in the codebase: the
project's ORM layer requires sqlalchemy 2.0, but airflow's internal
ORM requires sqlalchemy <2.0. Without this patch, NO DAG test can
run -- which is exactly how P1-031 (missing task dependencies)
shipped to production: every previous "fix" silently skipped the
DAG tests via ``pytest.importorskip('airflow')``.

ROOT FIX
--------
Monkey-patch sqlalchemy 2.0's ``_extract_mapped_subtype`` to return
``None`` instead of raising ``MappedAnnotationError`` when an
annotation is not ``Mapped[]``. This makes sqlalchemy 2.0 behave
like 1.4 for unmapped annotations -- airflow's TaskInstance imports
successfully, and the project's DeclarativeBase-based ORM continues
to work.

This patch is applied BEFORE any airflow import (via conftest.py's
early loading). It is INTENTIONALLY minimal -- it only disables the
STRICT annotation check, not any other sqlalchemy 2.0 behavior.

This is a TEST-ENVIRONMENT-ONLY patch. Production deployments must
either (a) use airflow 3.x (which has full sqlalchemy 2.0 support)
or (b) pin sqlalchemy<2.0 and update database/base.py to use the
legacy ``declarative_base()`` API.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Silence airflow's deprecation warnings (clutter test output).
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _patch_sqlalchemy_for_airflow() -> None:
    """Monkey-patch sqlalchemy 2.0 to accept airflow's legacy annotations.

    Airflow 2.9-2.10's TaskInstance model uses pre-sqlalchemy-2.0
    annotations like ``dag_model: "DagModel"`` instead of the
    sqlalchemy 2.0-required ``dag_model: Mapped["DagModel"]``. Under
    sqlalchemy 2.0 this raises ``MappedAnnotationError`` at import
    time, blocking all DAG tests.

    The patch replaces ``_extract_mapped_subtype`` (the function that
    raises the error) with a version that returns ``None`` for
    non-``Mapped[]`` annotations -- which is the sqlalchemy 1.4
    behavior. This lets airflow import successfully.
    """
    try:
        from sqlalchemy.orm import decl_base, util as orm_util
    except ImportError:
        # sqlalchemy not yet installed -- nothing to patch.
        return

    # Check if already patched (idempotent).
    if getattr(orm_util._extract_mapped_subtype, "_p1_034_patched", False):
        return

    original = orm_util._extract_mapped_subtype

    def lenient_extract_mapped_subtype(
        cls, key, annotation, mapped_expr, allow_unmapped_check=False
    ):
        """Lenient version: return None for non-Mapped annotations.

        If the annotation is a ``Mapped[]`` generic, delegate to the
        original function (preserves sqlalchemy 2.0 behavior for the
        project's own ORM models). Otherwise, return None (sqlalchemy
        1.4 behavior -- the annotation is treated as a plain class
        attribute, not a mapped column).
        """
        # Check if annotation is a Mapped[] generic.
        annotation_str = str(annotation)
        if "Mapped[" in annotation_str or "Mapped[" in str(getattr(annotation, "__origin__", "")):
            # Mapped[] annotation -- use original strict behavior.
            return original(cls, key, annotation, mapped_expr, allow_unmapped_check)
        # Non-Mapped annotation -- be lenient (sqlalchemy 1.4 behavior).
        return None

    lenient_extract_mapped_subtype._p1_034_patched = True
    # Patch in both locations where sqlalchemy looks it up.
    orm_util._extract_mapped_subtype = lenient_extract_mapped_subtype
    if hasattr(decl_base, "_extract_mapped_subtype"):
        decl_base._extract_mapped_subtype = lenient_extract_mapped_subtype


# Apply the patch BEFORE any airflow import.
_patch_sqlalchemy_for_airflow()
