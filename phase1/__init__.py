"""Phase 1 -- Data Ingestion Pipelines package.

v43 ROOT FIX (Chain 7 -- bare-import packaging): the Phase 1 modules use
bare imports like ``from cleaning.normalizer import ...`` which only
work when ``phase1/`` is on ``sys.path``. Previously this required the
operator to either ``cd phase1/`` first OR run via ``run_unified.py``
(which bootstraps sys.path). Importing ``phase1.pipelines.chembl``
from outside the package raised ``ModuleNotFoundError: No module named
'cleaning'``.

This ``__init__.py`` bootstraps ``sys.path`` so that bare imports work
from ANY current working directory. Operators can now do::

    from phase1.pipelines.chembl_pipeline import ChEMBLPipeline

without setting up sys.path themselves.

The bootstrap is idempotent -- it only inserts ``phase1/`` into
``sys.path`` if it's not already there.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Idempotent sys.path bootstrap for bare imports.
_PHASE1_ROOT = Path(__file__).resolve().parent
_PHASE1_ROOT_STR = str(_PHASE1_ROOT)
if _PHASE1_ROOT_STR not in sys.path:
    sys.path.insert(0, _PHASE1_ROOT_STR)

# Also add the project root so `from phase1.xxx import ...` works.
_PROJECT_ROOT = _PHASE1_ROOT.parent
_PROJECT_ROOT_STR = str(_PROJECT_ROOT)
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)
