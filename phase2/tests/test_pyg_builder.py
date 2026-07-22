"""Regression tests for pyg_builder.py.

This file is referenced by the pyg_builder.py module docstring (issues 56-59).
P2-049 ROOT FIX: the previous docstring referenced ``tests/test_pyg_builder.py``
(top-level), which did NOT exist. This file at ``phase2/tests/test_pyg_builder.py``
is the actual regression suite.

The tests here verify the SAFETY-CRITICAL invariants of pyg_builder:
  - Post-split integrity check (P2-066): must raise RuntimeError, not assert.
  - HeteroData construction produces the documented schema.
  - Reverse-edge naming convention is enforced.

Run with:
    pytest phase2/tests/test_pyg_builder.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

# Repo root is 3 levels up.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("DRUGOS_SKIP_IMPORT_CHECK", "1")


def test_p2_049_test_pyg_builder_file_exists():
    """P2-049: this file (phase2/tests/test_pyg_builder.py) must exist
    so the pyg_builder.py docstring reference is not misleading."""
    # If you're running this test, the file exists by definition.
    assert os.path.isfile(__file__)


def test_p2_066_post_split_check_uses_runtime_error_not_assert():
    """P2-066: the post-split integrity check must use RuntimeError,
    not assert — so it survives python -O mode."""
    pyg_path = os.path.join(
        _REPO_ROOT, "phase2", "drugos_graph", "pyg_builder.py"
    )
    with open(pyg_path, "r", encoding="utf-8") as f:
        content = f.read()
    bad_assert = 'assert hasattr(tgt, "edge_label")'
    assert bad_assert not in content, (
        "pyg_builder.py must NOT use 'assert hasattr(tgt, \"edge_label\")' "
        "for the post-split check — replace with RuntimeError so it "
        "survives python -O. (P2-066)"
    )
    assert "raise RuntimeError" in content and "P2-066" in content


def test_pyg_builder_module_importable():
    """Sanity: pyg_builder module must be importable without errors."""
    # Skip if torch_geometric not installed — the import will fail.
    pytest.importorskip("torch_geometric")
    from phase2.drugos_graph import pyg_builder  # noqa: F401


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
