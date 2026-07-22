"""DrugOS Graph Module -- KG Embedding Model Protocol
=====================================================
Defines the structural interface (Protocol) for any knowledge graph
embedding model used in the DrugOS pipeline.

Why a Protocol?
  * ``TransEModel`` (Week 2 baseline) and the Phase 3 Graph Transformer
    both produce entity/relation embeddings and scores. This Protocol
    lets ``train_transe``, ``predict_drug_candidates``, and evaluation
    code accept ANY model that conforms to the interface -- enabling
    drop-in replacement without changing downstream consumers.
  * Fixes A1.6 -- transe_model.py was not interchangeable with future models.

Interoperability:
  * Any model implementing this Protocol can be passed to
    ``train_transe(model=..., ...)``, ``predict_drug_candidates(model=..., ...)``,
    and ``evaluate_link_prediction(...)`` without code changes.

P2-028 ROOT FIX (Team 8 -- Protocol was aspirational):
  The previous version of this file defined only ``KGEmbeddingModel``,
  whose required attributes (``entity_embeddings``,
  ``relation_embeddings``, ``normalize_entity_embeddings``,
  ``num_total_entities``) match ``TransEModel``'s API but NOT
  ``DrugRepurposingGraphTransformer``'s API. The GraphTransformer
  (Phase 3 production model) uses a DIFFERENT forward() signature
  (``node_features``, ``edge_indices``, ``drug_indices``,
  ``disease_indices``) and does NOT expose ``entity_embeddings`` /
  ``relation_embeddings`` / ``normalize_entity_embeddings`` /
  ``num_total_entities``. The Protocol was therefore ASPIRATIONAL --
  it claimed GraphTransformer conformed but it did not.

  ROOT FIX: define a SEPARATE ``DrugRepurposingModel`` Protocol that
  matches ``DrugRepurposingGraphTransformer``'s ACTUAL API
  (``forward``, ``forward_logits``, ``score_direction``, ``save``,
  ``load``). ``KGEmbeddingModel`` is kept for TransE-style KGE models
  (homogeneous embedding tables + (h, r, t) scoring). The two Protocols
  reflect the two distinct model families in the pipeline:

    - ``KGEmbeddingModel`` -- TransE / DistMult / ComplEx (homogeneous).
      Used by ``train_transe`` and ``predict_drug_candidates``.
    - ``DrugRepurposingModel`` -- HGT / GraphTransformer (heterogeneous).
      Used by the Phase 3 training loop in
      ``graph_transformer/training/trainer.py`` and the inference path
      in ``graph_transformer/inference/``.

  Both Protocols share a common ``score_direction`` property so
  ``compute_auc`` and the evaluation path can read the direction from
  EITHER model type via the SAME attribute name.

  A CI test (``tests/test_p2_028_model_protocol_real.py``) verifies
  that:
    - ``TransEModel`` satisfies ``KGEmbeddingModel`` (runtime_checkable
      isinstance).
    - ``DrugRepurposingGraphTransformer`` satisfies
      ``DrugRepurposingModel`` (runtime_checkable isinstance).
    - Both classes expose ``score_direction`` returning one of
      ``"lower_better"`` / ``"higher_better"``.

Fixes: A1.6, I15.13, P2-028 (Team 8).
"""

from __future__ import annotations

from typing import Dict, Optional, Protocol, Tuple, runtime_checkable

import torch


@runtime_checkable
class KGEmbeddingModel(Protocol):
    """Structural interface for HOMOGENEOUS KG embedding models.

    Implemented by ``TransEModel`` (and future DistMult / ComplEx
    variants). These models share a single entity embedding table and
    a single relation embedding table, and score a triple
    ``(head, relation, tail)`` via a single forward call that takes
    index tensors.

    P2-028 ROOT FIX (Team 8): this Protocol is now SCOPED to
    homogeneous KGE models. The Phase 3 GraphTransformer does NOT
    satisfy it (different forward signature, no entity_embeddings
    property). Use ``DrugRepurposingModel`` for the GraphTransformer.

    ``@runtime_checkable`` enables ``isinstance(model, KGEmbeddingModel)``
    checks at runtime (for validation guards, not for branching logic).

    Attributes:
        entity_embeddings: An ``nn.Embedding`` whose ``.weight`` tensor
            has shape ``(num_entities, embedding_dim)``.
        relation_embeddings: An ``nn.Embedding`` whose ``.weight`` tensor
            has shape ``(num_relations, embedding_dim)``.

    Methods:
        forward(head_indices, rel_indices, tail_indices) -> Tensor:
            Compute plausibility scores for triples.
        normalize_entity_embeddings() -> None:
            Normalize entity embeddings (TransE-specific convention).
    """

    @property
    def entity_embeddings(self) -> torch.nn.Embedding:
        """Entity embedding lookup table: (num_entities, embedding_dim)."""
        ...

    @property
    def relation_embeddings(self) -> torch.nn.Embedding:
        """Relation embedding lookup table: (num_relations, embedding_dim)."""
        ...

    def forward(
        self,
        head_indices: torch.Tensor,
        rel_indices: torch.Tensor,
        tail_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute plausibility score for each (h, r, t) triple.

        Args:
            head_indices: Entity index tensor for triple heads.
            rel_indices: Relation index tensor.
            tail_indices: Entity index tensor for triple tails.

        Returns:
            Tensor of shape ``(batch_size,)`` with one score per triple.
            Convention varies by model: TransE uses L2 distance (lower=better);
            the Phase 3 Graph Transformer may use dot product (higher=better).
        """
        ...

    def normalize_entity_embeddings(self) -> None:
        """Normalize entity embeddings to unit L2 norm (TransE convention).

        Called after each optimizer step. Models that do not require
        normalization (e.g., DistMult, ComplEx) can implement this as
        a no-op.
        """
        ...

    # v43 ROOT FIX (P1 -- score_direction not in Protocol): the Protocol
    # did not declare score_direction, so a model that doesn't declare
    # it silently defaults to "lower_better" via getattr in train_transe.
    # HGT (higher_better) would fail the assertion at training time --
    # but only at training time, not at Protocol compliance time. Adding
    # score_direction to the Protocol makes the contract explicit.
    @property
    def score_direction(self) -> str:
        """Scoring convention: 'lower_better' (TransE) or 'higher_better' (HGT).

        - 'lower_better': score = -||h+r-t|| (TransE, Bordes 2013).
          Lower score = more plausible triple.
        - 'higher_better': score = sigmoid(W·[h||r||t]) (HGT, Graph
          Transformer). Higher score = more plausible triple.

        train_transe uses this to decide the loss function and the AUC
        direction. predict_drug_candidates uses it to decide topk
        direction (largest=False for lower_better, largest=True for
        higher_better).
        """
        ...

    # v102 ROOT FIX (P2-039): num_total_entities not in Protocol.
    #
    # The previous train_transe code at line 2219 used
    # ``getattr(model, "num_total_entities", None)`` with a fallback to
    # ``model.entity_embeddings.num_embeddings``. The comment claimed
    # this was "for HGT" — but neither TransE NOR HGT was REQUIRED to
    # expose num_total_entities. HGT did (added in v43), TransE did
    # not. The getattr was therefore dead code for TransE: it always
    # fell through to the fallback. This is misleading for maintainers:
    # a future developer adding a new heterogeneous model would see
    # ``getattr(model, "num_total_entities", None)`` and assume the
    # existing code path uses it — but the contract was undocumented.
    #
    # ROOT FIX: add ``num_total_entities`` to the KGEmbeddingModel
    # Protocol as a REQUIRED property. Document that:
    #   - Homogeneous models (TransE): returns entity_embeddings.num_embeddings
    #     (the single entity table's row count).
    #   - Heterogeneous models (HGT): returns the SUM of all node-type
    #     entity counts (the cross-type index space used for negative
    #     sampling).
    # This makes the contract explicit and forward-compatible: any new
    # model that implements KGEmbeddingModel MUST expose
    # num_total_entities, and the train_transe code path will use it.
    @property
    def num_total_entities(self) -> int:
        """Total entity count across ALL node/entity tables.

        - Homogeneous models (TransE): equals
          ``entity_embeddings.num_embeddings`` (single entity table).
        - Heterogeneous models (HGT): equals
          ``sum(self._node_counts.values())`` — the SUM of all node-
          type counts. Used for index-range validation in train_transe
          and for negative sampling space sizing.

        Callers that need the TOTAL entity count MUST use this property
        instead of ``entity_embeddings.num_embeddings`` (which returns
        only the FIRST node type's count on heterogeneous models, leading
        to out-of-range negative samples and index-range validation
        failures — see P2-039 root cause).
        """
        ...


@runtime_checkable
class DrugRepurposingModel(Protocol):
    """Structural interface for HETEROGENEOUS graph models used in the
    DrugOS drug-repurposing pipeline.

    P2-028 ROOT FIX (Team 8): this Protocol was added because the
    existing ``KGEmbeddingModel`` Protocol required attributes
    (``entity_embeddings``, ``relation_embeddings``,
    ``normalize_entity_embeddings``, ``num_total_entities``) that the
    Phase 3 ``DrugRepurposingGraphTransformer`` does NOT expose. The
    GraphTransformer uses a different forward() signature (heterogeneous
    node features + edge indices) and has its own ``save`` / ``load``
    methods. The old Protocol was therefore ASPIRATIONAL for the
    GraphTransformer -- it claimed conformance but did not enforce it.

    This Protocol matches ``DrugRepurposingGraphTransformer``'s ACTUAL
    API surface (verified against
    ``graph_transformer/models/graph_transformer.py``). It is the
    contract used by:
      - ``graph_transformer/training/trainer.py`` (training loop)
      - ``graph_transformer/inference/`` (inference helpers)
      - The Phase 3 -> Phase 4 RL bridge (``gt_rl_bridge.py``)
      - The evaluation path in ``evaluation.py`` (reads
        ``score_direction`` to set AUC direction)

    Implementers MUST provide:
      - ``forward`` -- full forward pass returning CALIBRATED probabilities.
      - ``forward_logits`` -- forward pass returning RAW logits (for
        ``BCEWithLogitsLoss`` training).
      - ``score_direction`` -- property returning ``"higher_better"``
        (graph transformers output logits -> sigmoid; higher = more
        plausible drug-disease pair).
      - ``save`` -- save model state + config to a file path.
      - ``load`` -- classmethod that reconstructs the model from a
        saved checkpoint.

    ``@runtime_checkable`` enables ``isinstance(model,
    DrugRepurposingModel)`` checks at runtime. The CI test
    ``tests/test_p2_028_model_protocol_real.py`` verifies that
    ``DrugRepurposingGraphTransformer`` satisfies this Protocol.
    """

    @property
    def score_direction(self) -> str:
        """Scoring convention. For DrugRepurposingModel: always
        ``'higher_better'``.

        The link predictor outputs logits -> sigmoid probabilities.
        Higher score = more plausible drug-disease pair. The eval path
        (``compute_auc`` in evaluation.py) reads this to set
        ``higher_is_better=True`` for AUC computation.
        """
        ...

    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_indices: Dict[Tuple[str, str, str], torch.Tensor],
        drug_indices: torch.Tensor,
        disease_indices: torch.Tensor,
        edge_weights: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        exclude_edges: Optional[set] = None,
        apply_temperature: bool = True,
    ) -> torch.Tensor:
        """Full forward pass: encode + predict, returning CALIBRATED probabilities.

        This is the INFERENCE entry point. The returned tensor contains
        sigmoid-calibrated probabilities in [0, 1] suitable for ranking
        drug-disease pairs and for the RL ranker's ``gnn_score`` input.

        For TRAINING (where ``BCEWithLogitsLoss`` expects raw logits),
        use ``forward_logits`` instead to avoid the
        ``sigmoid -> BCELoss`` NaN bomb.

        Args:
            node_features: Dict mapping node type to (N_t, D_t) tensor.
            edge_indices: Dict mapping (src, rel, tgt) to (2, E) tensor.
            drug_indices: (N,) tensor of drug node indices.
            disease_indices: (N,) tensor of disease node indices.
            edge_weights: Optional per-edge-type weights.
            exclude_edges: Optional set of edges to exclude for THIS
                call only. If None, uses the model's stored config.
            apply_temperature: If True (default), apply the calibrated
                temperature scaling via the link predictor.

        Returns:
            (N,) tensor of calibrated probabilities in [0, 1].
        """
        ...

    def forward_logits(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_indices: Dict[Tuple[str, str, str], torch.Tensor],
        drug_indices: torch.Tensor,
        disease_indices: torch.Tensor,
        edge_weights: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        exclude_edges: Optional[set] = None,
    ) -> torch.Tensor:
        """Forward pass returning RAW LOGITS (for BCEWithLogitsLoss).

        This is the TRAINING entry point. It avoids the
        ``sigmoid -> BCELoss`` NaN bomb from the original code.

        Args:
            node_features: Dict mapping node type to (N_t, D_t) tensor.
            edge_indices: Dict mapping (src, rel, tgt) to (2, E) tensor.
            drug_indices: (N,) tensor of drug node indices.
            disease_indices: (N,) tensor of disease node indices.
            edge_weights: Optional per-edge-type weights.
            exclude_edges: Optional set of edges to exclude for THIS
                call only. If None, uses the model's stored config.

        Returns:
            (N,) tensor of raw logits (unbounded real numbers).
        """
        ...

    def save(self, path: str) -> None:
        """Save model state dict + FULL config to a file path.

        The saved checkpoint must include ALL config fields
        (``feature_dims``, ``embedding_dim``, ``num_layers``,
        ``num_heads``, ``edge_types``, ``node_types``,
        ``exclude_edges``, ``ffn_hidden_dim``, ``dropout``,
        ``attention_dropout``, ``link_predictor_hidden_dims``,
        ``link_predictor_dropout``) so ``load`` can reconstruct the
        model exactly without relying on global defaults that may
        have changed since the checkpoint was written.

        Args:
            path: File path for the checkpoint (e.g. ``models/hgt_best.pt``).
        """
        ...

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DrugRepurposingModel":
        """Load model from a checkpoint saved by ``save``.

        Uses the config saved in the checkpoint to reconstruct the
        model exactly. Must use ``torch.load(..., weights_only=True)``
        when the PyTorch version supports it (security fix -- prevents
        arbitrary code execution via malicious checkpoints).

        Args:
            path: Path to the checkpoint file.
            device: Device to load the model onto (``"cpu"``, ``"cuda"``).

        Returns:
            A reconstructed model instance with loaded weights.
        """
        ...


__all__: list[str] = ["KGEmbeddingModel", "DrugRepurposingModel"]