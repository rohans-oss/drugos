"""Phase 2 test suite conftest -- pre-imports torch_geometric to avoid
circular import issues during test collection.

v61 ROOT FIX (torch_geometric 2.8.0 circular import):
When pytest collects tests from multiple files that import
``torch_geometric`` submodules, the first import triggers
``torch_geometric/__init__.py`` to execute. If a test module imports a
torch_geometric submodule (e.g. ``torch_geometric.data``) while
``__init__.py`` is still executing, the partial ``torch_geometric``
module doesn't yet have all its attributes set, raising:

    AttributeError: partially initialized module 'torch_geometric' has
    no attribute 'typing' (most likely due to a circular import)

ROOT FIX: pre-import ``torch_geometric`` (the full package) at the TOP
of this conftest.py, which pytest loads BEFORE collecting any test
module. This ensures ``torch_geometric/__init__.py`` fully executes
once, setting all attributes on the ``torch_geometric`` module, so
subsequent test-module imports of submodules don't hit the partial-
initialization window.
"""
import torch_geometric  # noqa: F401 -- pre-import to avoid circular import
import torch_geometric.typing  # noqa: F401
import torch_geometric.data  # noqa: F401
import torch_geometric.transforms  # noqa: F401
