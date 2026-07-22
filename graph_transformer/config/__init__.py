"""Configuration helpers for the Graph Transformer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# v91 FORENSIC ROOT FIX: when `phase1/database/connection.py` does
# `from config import settings`, Python can accidentally find THIS file
# (graph_transformer/config/__init__.py) instead of phase1/config/settings.py
# -- a name collision between phase1/config/ and graph_transformer/config/.
# When loaded as top-level `config`, the relative import `from ..data` goes
# beyond the top-level package and raises ImportError. The fix: try the
# relative import first (normal case when imported as graph_transformer.config),
# fall back to an absolute import (defensive case when imported as top-level).
try:
    from ..data import (
        DEFAULT_EDGE_TYPES,
        DEFAULT_FEATURE_DIMS,
        DEFAULT_NODE_TYPES,
        LABEL_LEAKING_EDGES,
    )
except ImportError:
    from graph_transformer.data import (
        DEFAULT_EDGE_TYPES,
        DEFAULT_FEATURE_DIMS,
        DEFAULT_NODE_TYPES,
        LABEL_LEAKING_EDGES,
    )


@dataclass
class GTConfig:
    """Configuration for the Graph Transformer model + trainer.

    Used by ``DrugRepurposingGraphTransformer.from_config`` (B6 fix:
    from_config now requires feature_dims to be set explicitly; this
    dataclass provides a sane default).

    Attributes:
        feature_dims: Dict mapping node type to feature dim. Defaults
            to ``DEFAULT_FEATURE_DIMS``.
        embedding_dim: Unified embedding dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        edge_types: List of (src, rel, tgt) edge type tuples.
        node_types: List of node type strings.
        ffn_hidden_dim: Hidden dim for FFN in each layer.
        dropout: General dropout rate.
        attention_dropout: Attention score dropout rate.
        link_predictor_hidden_dims: Hidden dims for the link predictor.
        link_predictor_dropout: Dropout for the link predictor.
        exclude_edges: Set of edge types to exclude during forward
            (label leakage prevention).
    """

    feature_dims: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_FEATURE_DIMS))
    embedding_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    edge_types: List[Tuple[str, str, str]] = field(
        default_factory=lambda: list(DEFAULT_EDGE_TYPES)
    )
    node_types: List[str] = field(
        default_factory=lambda: list(DEFAULT_NODE_TYPES)
    )
    ffn_hidden_dim: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    # ROOT FIX (FORENSIC-AUDIT-I11): changed default from None to [256, 128]
    # to match the model's internal default. The previous None caused
    # inconsistency: GTConfig() -> None, but the model's constructor
    # converts None to [256, 128] internally. Three different values
    # across the codebase (None, [256, 128], [64, 32]) was confusing.
    # Now GTConfig and the model agree on [256, 128] by default; the
    # bridge explicitly overrides to [64, 32] for the demo graph.
    link_predictor_hidden_dims: Optional[List[int]] = field(
        default_factory=lambda: [256, 128]
    )
    link_predictor_dropout: float = 0.2
    exclude_edges: set = field(
        default_factory=lambda: set(LABEL_LEAKING_EDGES)
    )
