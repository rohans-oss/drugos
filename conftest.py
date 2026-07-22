# =============================================================================
# conftest.py (repo root) — P1-011 v113 ROOT FIX
# =============================================================================
# Bootstraps sys.path so that BOTH bare imports (`from database.connection
# import ...`) AND absolute imports (`from phase1.database.connection import
# ...`) work from every test, regardless of the test's location or the
# current working directory.
#
# Previously, tests that did `from phase1.database.connection import get_engine`
# could fail with `ModuleNotFoundError: No module named '_circuit_breaker'`
# if phase1/__init__.py hadn't run yet. This conftest.py runs BEFORE any
# test collection (pytest guarantees this), inserting both the repo root
# and phase1/ into sys.path so every import style resolves.
# =============================================================================
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_PHASE1_ROOT = _REPO_ROOT / "phase1"

for _p in (_REPO_ROOT, _PHASE1_ROOT):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)
