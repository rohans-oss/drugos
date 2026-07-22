"""Phase 2 -- DrugOS Knowledge Graph package.

v43 ROOT FIX (Chain 7 -- bare-import packaging): the Phase 2 modules
under ``drugos_graph/`` use absolute imports like
``from drugos_graph.kg_builder import ...`` which require ``phase2/``
to be on ``sys.path``. Previously this required the operator to either
``cd phase2/`` first OR run via ``run_unified.py`` (which bootstraps
sys.path).

This ``__init__.py`` bootstraps ``sys.path`` so that imports work from
ANY current working directory. Operators can now do::

    from phase2.drugos_graph.kg_builder import DrugOSGraphBuilder

without setting up sys.path themselves.

The bootstrap is idempotent -- it only inserts paths into ``sys.path``
if they're not already there.

P2-061 ROOT FIX (Teammate 4, forensic, root-level): the previous code
UNCONDITIONALLY inserted ``phase2/`` and the project root into
``sys.path`` at import time. This pollutes the global Python path —
if ``phase2`` is imported as a submodule of a package (e.g.
``import autonomous_drug_repurposing.phase2`` from a downstream
consumer), the ``sys.path.insert`` calls add paths that may shadow
other packages or cause import conflicts in the consumer's
environment. ROOT FIX: only bootstrap sys.path when:
  1. ``phase2`` is imported as a TOP-LEVEL package (not a submodule),
     detected by checking ``__name__ == "phase2"`` (not
     ``"autonomous_drug_repurposing.phase2"``).
  2. The paths are not already on sys.path (idempotent — unchanged).
This preserves the dev/CI ergonomics (``from drugos_graph.x import y``
works from the repo root) without polluting the path when phase2 is
used as a library.
"""
from __future__ import annotations

import sys
from pathlib import Path

# P2-061 ROOT FIX: only bootstrap sys.path when phase2 is imported as
# a TOP-LEVEL package. When imported as a submodule (e.g.
# ``autonomous_drug_repurposing.phase2``), skip the bootstrap — the
# parent package is responsible for path setup.
if __name__ == "phase2":
    # Idempotent sys.path bootstrap.
    _PHASE2_ROOT = Path(__file__).resolve().parent
    _PHASE2_ROOT_STR = str(_PHASE2_ROOT)
    if _PHASE2_ROOT_STR not in sys.path:
        sys.path.insert(0, _PHASE2_ROOT_STR)

    # Also add the project root so `from phase2.xxx import ...` works.
    _PROJECT_ROOT = _PHASE2_ROOT.parent
    _PROJECT_ROOT_STR = str(_PROJECT_ROOT)
    if _PROJECT_ROOT_STR not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT_STR)
