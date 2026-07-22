"""
Graph Transformer package for the Autonomous Drug Repurposing Platform.

Phase 3 core AI engine: reads the biomedical knowledge graph and predicts
drug-disease therapeutic interaction scores.

FIX vs original codebase:
  - **B8 (import hell)**: this package is now a *proper* installable Python
    package. Internal imports use relative paths (``from .data import ...``)
    so the package can be imported from anywhere as
    ``from graph_transformer.models.graph_transformer import
    DrugRepurposingGraphTransformer`` without any sys.path hackery.
  - **B7 (dual DEFAULT_FEATURE_DIMS)**: the single source of truth for
    feature dimensions now lives in ``graph_transformer.data`` and is
    re-exported from every sub-module that needs it. There is no longer a
    second, divergent copy in ``models.graph_transformer``.
"""
from __future__ import annotations

# ROOT FIX (FORENSIC-AUDIT-I37): aligned version with rl package.
# The previous version "2.0.0" mismatched rl's "4.0.0", suggesting
# independent versioning. Both packages are now versioned together
# as "4.1.0" (V4 + forensic audit fixes). The bridge checks that
# both packages have compatible versions.
__version__ = "4.1.0"
__schema_version__ = "4.1.0"

# Re-export the most-used symbols so callers can do
#   from graph_transformer import DrugRepurposingGraphTransformer
# without having to know the internal layout.
#
# ROOT FIX (E14): removed compliance_note and self_check from the
# re-exports and __all__. These were never used externally (E14 finding)
# and only polluted the API surface. They remain available via
# graph_transformer.data.compliance_note / self_check for anyone who
# needs them.
#
# ROOT FIX (E20): added LABEL_LEAKING_EDGES to the re-exports and
# __all__. This is a critical constant used by the bridge and trainer,
# and should be accessible from the top-level package.
from .data import (
    DEFAULT_FEATURE_DIMS,
    EDGE_TYPES,
    NODE_TYPES,
    V1_AUC_THRESHOLD,
    LABEL_LEAKING_EDGES,  # E20 fix: add to public API
)
from .models.graph_transformer import DrugRepurposingGraphTransformer
from .training.trainer import GraphTransformerTrainer

__all__ = [
    "DrugRepurposingGraphTransformer",
    "GraphTransformerTrainer",
    "DEFAULT_FEATURE_DIMS",
    "EDGE_TYPES",
    "NODE_TYPES",
    "V1_AUC_THRESHOLD",
    "LABEL_LEAKING_EDGES",  # E20 fix: in __all__
    "__version__",
    "__schema_version__",
]
