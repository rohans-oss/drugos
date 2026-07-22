"""DrugOS Graph — Path Configuration (extracted from config.py).

v108 ROOT FIX (ISSUE-P2-056): this module is the SECOND real extraction
from the 8400-line ``config.py`` monolith. The path constants are MOVED
here (not copied), and ``config.py`` imports them back so
``from .config import RAW_DIR`` continues to work.

The path constants are leaf definitions (Path objects derived from
``__file__`` and environment variables) with no dependencies on other
config sections, so they can be extracted cleanly without circular
imports.

NOTE: ``Path(__file__)`` in this module resolves to the same directory
as ``config.py`` (both live in ``phase2/drugos_graph/``), so the
``_PROJECT_ROOT`` computation is identical regardless of which file
defines it.

Consumers should prefer importing from this module directly:
    from .config_paths import RAW_DIR, PROCESSED_DIR
The legacy ``from .config import RAW_DIR`` continues to work via
re-export in ``config.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─── Project Root Resolution ─────────────────────────────────────────────────
# Resolves the project root directory. Honors the DRUGOS_PROJECT_ROOT
# environment variable (for production deployments); falls back to the
# parent of this package directory (for development); finally falls back
# to the current working directory (for pip-installed packages).

_DRUGOS_ROOT_ENV = os.environ.get("DRUGOS_PROJECT_ROOT", "")
if _DRUGOS_ROOT_ENV:
    _PROJECT_ROOT = Path(_DRUGOS_ROOT_ENV).resolve()
else:
    # For development: use parent of package dir
    # For installed packages: use current working directory
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if not (_PROJECT_ROOT / "data").exists():
        # Likely installed via pip — use cwd
        _PROJECT_ROOT = Path.cwd()

# Fixes audit issue 1.1 — backward-compat alias
# Deprecated: use _PROJECT_ROOT for internal derivation; PROJECT_ROOT kept for backward compat
PROJECT_ROOT: Path = _PROJECT_ROOT

DATA_DIR = _PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# FIX TOP-12: Phase 2's ``PROCESSED_DIR`` is for Phase 2's OWN outputs.
# Phase 2 READS Phase 1's outputs via the bridge from
# ``PHASE1_PROCESSED_DIR`` (phase1/processed_data/).
PHASE1_PROCESSED_DIR: Path = Path(
    os.environ.get(
        "DRUGOS_PHASE1_PROCESSED_DIR",
        str(_PROJECT_ROOT.parent / "phase1" / "processed_data"),
    )
)

# FIX TOP-3: ``RESULTS_PERSIST_PATH`` is the on-disk JSON file where
# ``run_full_pipeline`` writes its full results dict.
RESULTS_PERSIST_PATH: Path = PROCESSED_DIR / "pipeline_results.json"

# Fixes audit issue 1.2 — KG_DIR renamed to KG_EXPORT_DIR (clearer intent)
KG_EXPORT_DIR = DATA_DIR / "kg_exports"

# Backward-compat alias — KG_DIR still importable
KG_DIR: Path = KG_EXPORT_DIR

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
LOGS_DIR = _PROJECT_ROOT / "logs"


__all__ = [
    "_PROJECT_ROOT",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "PHASE1_PROCESSED_DIR",
    "RESULTS_PERSIST_PATH",
    "KG_EXPORT_DIR",
    "KG_DIR",
    "EMBEDDINGS_DIR",
    "LOGS_DIR",
]
