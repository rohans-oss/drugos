"""
Model architectures module for the Graph Transformer.

Provides:
- NodeTypeProjection: Projects raw node features to a unified embedding space.
- GraphTransformerLayer: Single transformer layer with heterogeneous attention.
- DrugDiseaseLinkPredictor: MLP head for scoring drug-disease pairs.
- DrugRepurposingGraphTransformer: Full model combining all components.
"""
from __future__ import annotations

from .embeddings import NodeTypeEmbedding, NodeTypeProjection
from .layers import GraphTransformerLayer
from .link_predictor import DrugDiseaseLinkPredictor
from .graph_transformer import DrugRepurposingGraphTransformer

__all__ = [
    "NodeTypeProjection",
    "NodeTypeEmbedding",
    "GraphTransformerLayer",
    "DrugDiseaseLinkPredictor",
    "DrugRepurposingGraphTransformer",
]
