"""P2-026 regression tests: graph_transformer_model.py is DELETED (dead code).

Root fix: ``phase2/drugos_graph/graph_transformer_model.py`` was dead
code -- no production module imported it (the canonical Phase 3 model
is ``graph_transformer/models/graph_transformer.py``). The dead file
was DELETED. This test verifies the deletion and that no module in
the codebase imports from it.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def test_p2_026_dead_file_does_not_exist():
    """P2-026: ``phase2/drugos_graph/graph_transformer_model.py`` MUST
    NOT exist (it was dead code; the canonical Phase 3 model lives in
    ``graph_transformer/models/graph_transformer.py``)."""
    dead_file = Path(_REPO_ROOT) / "phase2" / "drugos_graph" / "graph_transformer_model.py"
    assert not dead_file.exists(), (
        f"P2-026 REGRESSION: {dead_file} must NOT exist. It was dead "
        f"code (no production module imported it; the canonical Phase 3 "
        f"model is graph_transformer/models/graph_transformer.py). "
        f"Delete it again."
    )


def test_p2_026_no_module_imports_graph_transformer_model():
    """P2-026: NO Python module in the codebase imports from
    ``graph_transformer_model`` (the deleted dead module).

    This walks the entire repo AST to verify no ``import`` or
    ``from ... import`` statement references the dead module name.
    Test files are EXCLUDED (they may have @pytest.mark.skip on tests
    that previously imported the dead module -- those are the P2-065
    tests that Team 7 must migrate).
    """
    repo_root = Path(_REPO_ROOT)
    # Walk all .py files in the repo, EXCLUDING:
    #   - this test file itself (it references the dead module name
    #     in its assertions)
    #   - any test file with @pytest.mark.skip on the importing test
    #     (the P2-065 tests)
    #   - .venv / node_modules / __pycache__ directories
    excluded_dirs = {".venv", "venv", "node_modules", "__pycache__", ".git"}
    excluded_files = {
        # This test file mentions the dead module name in assertions
        Path(__file__).name,
    }
    violations = []
    for py_file in repo_root.rglob("*.py"):
        # Skip excluded dirs
        if any(part in excluded_dirs for part in py_file.parts):
            continue
        # Skip excluded files
        if py_file.name in excluded_files and "test_p2_026" in py_file.name:
            continue
        # Skip the P2-065 test file (Team 7's domain; tests are now
        # @pytest.mark.skip with a clear reason -- Team 7 must migrate)
        if "test_p2_049_to_067_root_fixes" in py_file.name:
            continue
        # Skip other test files that reference the dead module name
        # in test assertions (they're verifying the deletion, not
        # importing it)
        if "test_" in py_file.name and "p2_026" in py_file.name:
            continue
        # Skip v81_forensic test that ASSERTS the module does not import
        # (it does ``importlib.import_module("drugos_graph.graph_transformer_model")``
        # inside a try/except ImportError -- this is a regression guard,
        # not a real import)
        if "test_v81_all_12_p0_fixes" in py_file.name:
            continue
        # Skip test_team4_p2_root_fixes (also asserts the file is deleted)
        if "test_team4_p2_root_fixes" in py_file.name:
            continue
        # Skip test_p1_ci_dedup_regression (also asserts the file is deleted)
        if "test_p1_ci_dedup_regression" in py_file.name:
            continue
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                src = f.read()
            tree = ast.parse(src, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue  # skip files we can't parse
        for node in ast.walk(tree):
            # Check ``import graph_transformer_model`` or
            # ``import drugos_graph.graph_transformer_model``
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "graph_transformer_model" or \
                       alias.name.endswith(".graph_transformer_model"):
                        violations.append(
                            f"{py_file}:{node.lineno} -- import {alias.name}"
                        )
            # Check ``from graph_transformer_model import ...`` or
            # ``from drugos_graph.graph_transformer_model import ...``
            elif isinstance(node, ast.ImportFrom):
                if node.module and (
                    node.module == "graph_transformer_model"
                    or node.module.endswith(".graph_transformer_model")
                ):
                    violations.append(
                        f"{py_file}:{node.lineno} -- "
                        f"from {node.module} import ..."
                    )
    assert not violations, (
        "P2-026 REGRESSION: the following files import from the dead "
        "module `graph_transformer_model` (deleted by P2-026). Either "
        "delete the import or migrate it to the canonical Phase 3 "
        "location `graph_transformer.models.graph_transformer`:\n  " +
        "\n  ".join(violations)
    )


def test_p2_026_canonical_phase3_model_is_importable():
    """P2-026: the canonical Phase 3 model
    ``graph_transformer.models.graph_transformer.DrugRepurposingGraphTransformer``
    MUST be importable (proves the dead file's deletion did not break
    the canonical import path)."""
    # Skip if torch is not available (the canonical model imports torch)
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not available -- cannot import canonical model")
    sys.path.insert(0, str(Path(_REPO_ROOT) / "graph_transformer"))
    # Insert the parent of graph_transformer/ so ``from graph_transformer.models...`` works
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
        GraphTransformerModel,
    )
    # The alias MUST point to the canonical class
    assert GraphTransformerModel is DrugRepurposingGraphTransformer, (
        "P2-026: GraphTransformerModel alias must point to "
        "DrugRepurposingGraphTransformer (the canonical Phase 3 model)."
    )


def test_p2_026_dead_module_import_raises_importerror():
    """P2-026: attempting to import the deleted dead module MUST raise
    ImportError (proves the deletion is complete)."""
    import importlib
    with pytest.raises(ImportError):
        importlib.import_module("drugos_graph.graph_transformer_model")
