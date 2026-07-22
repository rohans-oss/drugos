"""
Node embedding modules for the Graph Transformer.

Projects heterogeneous node features (drugs, proteins, pathways, diseases,
clinical_outcomes) into a unified embedding space with type distinctions.

FIX vs original codebase (B8):
  Internal imports use relative paths. No functional change vs original
  -- this module was already correct, only its import style was broken.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class NodeTypeEmbedding(nn.Module):
    """Learnable per-node-type embedding vector.

    Each of the 5 node types (drug, protein, pathway, disease,
    clinical_outcome) gets a unique learnable vector that is added to
    the projected node features.

    ROOT FIX (B5): this class is now used EXTERNALLY via
    ``DrugRepurposingGraphTransformer.get_node_type_embeddings()``,
    which exposes the learned type embeddings for downstream consumers
    (dashboard visualization, model inspection, etc.). Previously it
    was only used internally by ``NodeTypeProjection`` and exported in
    ``models/__init__.py`` without any external caller -- making the
    export "API surface pollution". The V11 fix wires it into the
    public API of the main model class.

    P3-004 ROOT FIX (Team Member 9, v104 — UNKNOWN NODE-TYPE FALLBACK):
        At inference time, if the model receives a graph with a NEW
        node type (e.g., 'variant' for pharmacogenomics added after
        the model is trained), the lookup
        ``self.embeddings(node_type_indices)`` raises IndexError because
        ``node_type_indices`` contains a value >= ``num_node_types``.
        This blocks KG growth — the model cannot be deployed without
        retraining.

        ROOT FIX: add a fallback 'unknown' embedding (slot index
        ``num_node_types``, accessible via the ``UNKNOWN_TYPE_IDX``
        class attribute). In ``forward()``, CLAMP out-of-range indices
        to the unknown slot and log a WARNING (once per instance, to
        avoid log spam). The unknown embedding is initialized to ZERO
        with a small learnable bias, so it does not perturb trained
        representations but can be learned if the model is later
        fine-tuned on a graph that includes the new type.

        This graceful degradation allows Phase 2 to grow (add new node
        types) without blocking Phase 3 inference — the new type's
        nodes will pass through with neutral embeddings until the model
        is retrained.

    Args:
        num_node_types: Number of distinct node types.
        embedding_dim: Dimension of the type embedding.
    """

    # P3-004 ROOT FIX v104: the unknown-type slot is at index num_node_types
    # (i.e., one slot past the last trained type). UNKNOWN_TYPE_IDX is a
    # CLASS-level constant so callers can introspect the fallback index
    # without instantiating the class.
    UNKNOWN_TYPE_IDX: int = -1  # resolved per-instance in __init__

    def __init__(self, num_node_types: int = 5, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_node_types = num_node_types
        # P3-004 ROOT FIX v104: allocate one EXTRA slot for the 'unknown'
        # type fallback. Total embedding table size = num_node_types + 1.
        #
        # P3-022 ROOT FIX (v114 forensic, SCIENTIFIC-ML CORRECTNESS): the
        # previous implementation initialized the unknown slot to ZERO and
        # re-zeroed it after _init_weights. The slot IS learnable
        # (``requires_grad=True`` by default), but during fine-tuning on a
        # graph that includes the new node type, the slot starts at zero.
        # Zero is a SADDLE POINT for embeddings: the gradient signal for
        # the unknown slot comes ONLY from nodes of the new type, and if
        # those nodes are rare (e.g., a new "variant" node type with only
        # 10 instances in a 100K-node graph), the gradient signal is
        # weak. The slot may NEVER escape the zero basin -- the
        # downstream linear layer's gradient w.r.t. a zero embedding is
        # also zero for that embedding's row. This is the standard "dead
        # neuron" problem for embeddings.
        #
        # ROOT FIX: initialize the unknown slot to a SMALL RANDOM value
        # (nn.init.normal_ with std=0.02, the BERT/GPT initialization).
        # This breaks the zero-gradient symmetry: the embedding starts
        # non-zero, so downstream linear layers produce non-zero outputs,
        # so the gradient w.r.t. the embedding is non-zero, so the
        # embedding CAN learn. The perturbation to trained representations
        # is < 2% (vs Xavier init's std=0.125) -- negligible for trained
        # types but crucial for the unknown type to escape the zero basin.
        self.UNKNOWN_TYPE_IDX = num_node_types  # instance-level resolution
        self.embeddings = nn.Embedding(num_node_types + 1, embedding_dim)
        # P3-022 ROOT FIX (v114): initialize unknown slot to SMALL RANDOM
        # (std=0.02) instead of ZERO. This breaks the zero-gradient saddle
        # point so fine-tuning on a graph with new node types can actually
        # learn the unknown slot's embedding. The 0.02 std is the
        # BERT/GPT initialization (Devlin et al. 2018, Radford et al.
        # 2019) -- small enough to not perturb trained representations
        # but large enough to escape the zero basin.
        with torch.no_grad():
            nn.init.normal_(self.embeddings.weight[self.UNKNOWN_TYPE_IDX], mean=0.0, std=0.02)
        # Track whether we've warned about unknown-type fallback (per instance).
        self._unknown_warned: bool = False
        # P3-030 ROOT FIX (v107): track the most-recent forward call's
        # out-of-range mask so callers (NodeTypeProjection, the model,
        # and downstream service /predict endpoints) can mark predictions
        # involving unknown-type nodes with a ``degraded=True`` flag.
        # The audit's mandate: "Do not silently serve degraded predictions."
        # The previous code only logged a WARNING once per instance, then
        # forgot — the model kept emitting neutral (zero) type embeddings
        # for the unknown nodes with no way for the caller to know which
        # predictions were affected. This attribute is reset to None at
        # the start of every forward() call and set to the (num_nodes,)
        # boolean mask of out-of-range indices if any were detected. A
        # caller can check ``last_unknown_mask`` after forward() to mark
        # the affected node positions, then propagate that flag up to the
        # link predictor's per-pair output (any pair touching an unknown-
        # type node is ``degraded=True``).
        self.last_unknown_mask: Optional[torch.Tensor] = None
        # P3-030: track the count of distinct unknown-type indices seen
        # in the most-recent forward. Useful for callers that want to
        # log "N nodes of unknown type" without re-computing the mask.
        self.last_unknown_count: int = 0

    def forward(self, node_type_indices: torch.Tensor) -> torch.Tensor:
        """Look up type embeddings.

        P3-004 ROOT FIX v104: out-of-range indices (>= num_node_types)
        are CLAMPED to the unknown-type slot (index num_node_types) and
        a WARNING is logged (once per instance). This prevents the
        IndexError that would otherwise crash inference when Phase 2
        adds a new node type after the model is trained.

        Args:
            node_type_indices: Long tensor of shape (num_nodes,).
                Values in [0, num_node_types-1] are looked up normally.
                Values >= num_node_types are clamped to the unknown
                slot (index num_node_types).

        Returns:
            Tensor of shape (num_nodes, embedding_dim).
        """
        # P3-004 ROOT FIX v104: detect out-of-range indices and clamp
        # them to the unknown-type slot. This is the ROOT FIX for the
        # IndexError crash that blocked KG growth.
        # P3-030 ROOT FIX (v107): RECORD the out-of-range mask in
        # ``self.last_unknown_mask`` so callers can mark affected
        # predictions as ``degraded=True`` instead of silently serving
        # neutral embeddings. The mask is reset on every forward() call
        # (so a caller reading it after forward() sees the mask for THAT
        # call only, not a stale one from a prior call).
        out_of_range_mask = node_type_indices >= self.num_node_types
        # P3-030: always reset, then populate only if there are unknowns.
        self.last_unknown_mask = None
        self.last_unknown_count = 0
        if out_of_range_mask.any():
            num_oob = int(out_of_range_mask.sum().item())
            self.last_unknown_mask = out_of_range_mask
            self.last_unknown_count = num_oob
            if not self._unknown_warned:
                logger.warning(
                    f"NodeTypeEmbedding.forward() received {num_oob} "
                    f"node-type indices >= num_node_types "
                    f"({self.num_node_types}). These will be CLAMPED to "
                    f"the 'unknown' type slot (index "
                    f"{self.UNKNOWN_TYPE_IDX}), which is initialized to "
                    f"SMALL RANDOM (std=0.02, P3-022 v114 ROOT FIX). "
                    f"The model will produce APPROXIMATE embeddings for "
                    f"these nodes; fine-tuning on a graph that includes "
                    f"the new types will LEARN the unknown slot's "
                    f"embedding (the small random init breaks the "
                    f"zero-gradient saddle point that previously trapped "
                    f"the slot at zero forever). This is graceful "
                    f"degradation (P3-004 ROOT FIX v104 + P3-022 v114) "
                    f"— Phase 2 can add new node types without crashing "
                    f"Phase 3 inference. To get full fidelity, RETRAIN "
                    f"the model on a graph that includes the new types. "
                    f"P3-030 v107: callers can read "
                    f"``self.last_unknown_mask`` after forward() to mark "
                    f"predictions involving these nodes as "
                    f"``degraded=True``. (This warning is emitted ONCE "
                    f"per NodeTypeEmbedding instance.)"
                )
                self._unknown_warned = True
            # Clamp to the unknown slot. torch.clamp preserves dtype
            # and device, and is differentiable (no gradient through
            # the index selection — Embedding lookup itself is the
            # differentiable op).
            node_type_indices = node_type_indices.clamp(
                max=self.UNKNOWN_TYPE_IDX
            )
        return self.embeddings(node_type_indices)

    def _reset_unknown_slot(self) -> None:
        """Re-initialize the unknown-type slot (index ``UNKNOWN_TYPE_IDX``).

        FORENSIC ROOT FIX (audit Issue 133): this method is called by
        ``DrugRepurposingGraphTransformer.__init__`` AFTER
        ``self.apply(self._init_weights)``. The model's ``_init_weights``
        re-initializes ALL ``nn.Embedding`` modules (including this one)
        with Xavier_normal_. That re-init OVERWRITES the small-random
        init this class's ``__init__`` wrote to the unknown-type slot,
        breaking the unknown-type contract.

        P3-022 ROOT FIX (v114): re-initialize the unknown slot to SMALL
        RANDOM (std=0.02) instead of ZERO. The previous re-zeroing
        caused the unknown slot to be stuck at the zero saddle point
        during fine-tuning on graphs with new node types -- the
        downstream linear layer's gradient w.r.t. a zero embedding is
        zero, so the slot could never learn. Small random init breaks
        the symmetry, enabling fine-tuning to actually learn the new
        type's embedding.
        """
        with torch.no_grad():
            nn.init.normal_(
                self.embeddings.weight[self.UNKNOWN_TYPE_IDX],
                mean=0.0, std=0.02,
            )

    def was_degraded(self) -> bool:
        """Return True if the most-recent forward() saw any unknown-type index.

        P3-030 ROOT FIX (v107): convenience accessor for callers that
        want a single boolean instead of inspecting
        ``last_unknown_mask``. Returns False if forward() has not been
        called yet, or if the most-recent call had no unknown types.
        """
        return self.last_unknown_count > 0


class NodeTypeProjection(nn.Module):
    """Per-node-type linear projection into a unified embedding space.

    Handles the heterogeneous feature dimensions of different node types
    (e.g., drugs may have 1024-dim Morgan fingerprints while proteins
    have 768-dim ESM-2 embeddings).

    P3-009 ROOT FIX (Team Member 9, v104 — FREEZE PRETRAINED EMBEDDINGS):
        When Phase 2's pre-trained TransE (Bordes et al. 2013) embeddings
        are loaded into the per-type projection layers via
        ``load_pretrained_embeddings()``, the GNN's optimizer would
        normally update them during training. For small graphs (demo,
        pilot), the GNN's gradient signal is noisy and OVERWRITES the
        TransE signal — wasting the TransE pre-training (which took
        hours).

        ROOT FIX: add a ``freeze_pretrained`` flag (default True). When
        frozen, ``load_pretrained_embeddings()`` sets
        ``requires_grad=False`` on the loaded projection's weight and
        bias, so the optimizer skips them entirely. The frozen
        projection acts as a fixed feature extractor that maps raw
        node features into the TransE-learned embedding space, and the
        GNN learns only the attention/FFN weights on top.

        The frozen types are tracked in ``self._frozen_types`` so a
        future caller can introspect which types are frozen. A CI test
        verifies that ``requires_grad=False`` is set on frozen types.

    Args:
        feature_dims: Dict mapping node type name to raw feature dimension.
        embedding_dim: Target embedding dimension for all types.
        feature_norm: Type of normalization ('none' or 'layer').
            'batch' was REMOVED in P3-029 ROOT FIX (v107): the previous
            ``_SafeBatchNorm1d`` wrapper + ``feature_norm="batch"`` branch
            was dead code on every default construction path
            (``DrugRepurposingGraphTransformer.__init__`` does not pass
            ``feature_norm`` and the bridge's ``build_model`` does not
            expose it). 100+ lines of dead code misled reviewers into
            thinking BatchNorm was active. If a future caller needs
            BatchNorm, re-add it WITH tests AND wire it through the
            model's constructor (do NOT restore the dead public-API
            surface that was never invoked). Passing 'batch' now raises
            ValueError to make the removal explicit.
        freeze_pretrained: Default freeze policy for
            ``load_pretrained_embeddings()``. If True (default), loaded
            embeddings are frozen. If False, they are trainable. Can
            be overridden per-call via the ``freeze`` parameter of
            ``load_pretrained_embeddings()``.
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        embedding_dim: int = 128,
        feature_norm: str = "none",
        freeze_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dims = dict(feature_dims)
        self.embedding_dim = embedding_dim
        self.feature_norm = feature_norm
        # P3-009 ROOT FIX v104: default freeze policy for loaded pretrained
        # embeddings. Can be overridden per-call.
        self._default_freeze_pretrained: bool = bool(freeze_pretrained)
        # P3-009 ROOT FIX v104: track which node types have frozen
        # pretrained embeddings. Public for introspection (CI test checks
        # this set).
        self._frozen_types: set = set()

        # Per-type linear projections
        self.projections: Dict[str, nn.Module] = {}
        for node_type, dim in feature_dims.items():
            if dim <= 0:
                raise ValueError(
                    f"feature_dims['{node_type}'] must be positive, got {dim}"
                )
            proj = nn.Linear(dim, embedding_dim)
            self.add_module(f"proj_{node_type}", proj)
            self.projections[node_type] = proj

        # Optional normalization layers.
        # P3-029 ROOT FIX (v107): the ``feature_norm="batch"`` branch and
        # the ``_SafeBatchNorm1d`` wrapper have been REMOVED. They were
        # dead code on every default construction path (no caller passed
        # ``feature_norm="batch"``), and the 100+ lines of "RETAINED for
        # public API option" comments misled reviewers into thinking
        # BatchNorm was active. The audit's recommendation was to delete
        # the dead code and re-add it with tests if a real need arises.
        # We now raise ValueError on ``feature_norm="batch"`` so a caller
        # who relied on the dead API gets a clear error instead of silent
        # acceptance followed by an untrained-BatchNorm fallback.
        self.norms: Dict[str, nn.Module] = {}
        if feature_norm == "layer":
            for node_type in feature_dims:
                norm = nn.LayerNorm(embedding_dim)
                self.add_module(f"norm_{node_type}", norm)
                self.norms[node_type] = norm
        elif feature_norm == "batch":
            raise ValueError(
                "P3-029 ROOT FIX (v107): feature_norm='batch' is no longer "
                "supported. The _SafeBatchNorm1d wrapper + 'batch' branch "
                "were dead code on every default construction path "
                "(DrugRepurposingGraphTransformer does not pass feature_norm "
                "and the bridge's build_model does not expose it). 100+ "
                "lines of dead code misled reviewers. If you genuinely "
                "need BatchNorm, re-add it WITH tests AND wire it through "
                "DrugRepurposingGraphTransformer.__init__. Use "
                "feature_norm='layer' (the standard pre-norm choice for "
                "deep transformers, used by GPT-2/LLaMA) or 'none'."
            )
        elif feature_norm not in ("none",):
            raise ValueError(
                f"feature_norm must be 'none' or 'layer', got {feature_norm!r}. "
                f"(P3-029 v107: 'batch' was removed as dead code.)"
            )

        # Learnable node type embeddings
        self.node_type_embedding = NodeTypeEmbedding(
            num_node_types=len(feature_dims),
            embedding_dim=embedding_dim,
        )

        # Node type name to index mapping.
        # ROOT FIX (FORENSIC-AUDIT-I09): use INSERTION ORDER instead of
        # sorted (alphabetical) order. The previous code used
        # ``enumerate(sorted(feature_dims.keys()))``, which means the type
        # index depends on alphabetical order. If a user passes feature_dims
        # in a different order (e.g., {"drug": 128, "protein": 64} vs
        # {"protein": 64, "drug": 128}), the type indices change, breaking
        # checkpoint compatibility. Python 3.7+ guarantees dict insertion
        # order, so using insertion order is deterministic AND stable
        # across different dict construction orders (as long as the user
        # constructs the dict consistently).
        self._type_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(feature_dims.keys())
        }

    # ------------------------------------------------------------------
    # P3-009 ROOT FIX v104: load + freeze pretrained embeddings
    # ------------------------------------------------------------------
    def load_pretrained_embeddings(
        self,
        node_type: str,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        freeze: Optional[bool] = None,
    ) -> None:
        """Load pre-trained TransE (or other) embeddings into a projection.

        P3-009 ROOT FIX v104: replaces the per-type linear projection's
        randomly-initialized weight with a pre-trained tensor (e.g., from
        Phase 2's TransE training). Optionally freezes the projection
        so the GNN's optimizer does NOT update it during training.

        Why freeze: TransE pre-training (Bordes et al. 2013) takes hours
        on a biomedical KG. If the GNN's optimizer is allowed to update
        the projection weights, the noisy GNN gradient signal on small
        graphs OVERWRITES the TransE signal — wasting the pre-training.
        Freezing preserves the TransE signal; the GNN learns only the
        attention/FFN weights on top.

        When NOT to freeze: on large production graphs (1M+ pairs), the
        GNN gradient signal is strong enough that fine-tuning the
        projection is beneficial. Pass ``freeze=False`` in that case.

        Args:
            node_type: Node type name (must be in ``feature_dims``).
            weight: Tensor of shape ``(embedding_dim, feature_dim)`` —
                the pre-trained projection weight. The shape must match
                ``nn.Linear.weight`` convention (out_features, in_features).
            bias: Optional tensor of shape ``(embedding_dim,)`` — the
                pre-trained projection bias. If None, the bias is left
                at its random initialization.
            freeze: If True, set ``requires_grad=False`` on the
                projection's weight (and bias if provided). If False,
                leave them trainable. If None, use the constructor's
                ``freeze_pretrained`` default.

        Raises:
            ValueError: If ``node_type`` is not in ``feature_dims``, or
                if ``weight`` shape does not match the projection's
                expected shape.
        """
        if node_type not in self.projections:
            raise ValueError(
                f"Unknown node type '{node_type}'. Known types: "
                f"{list(self.projections.keys())}. Add '{node_type}' to "
                f"feature_dims at construction time before loading "
                f"pretrained embeddings."
            )
        proj = self.projections[node_type]
        expected_w_shape = proj.weight.shape  # (out_features, in_features)
        if tuple(weight.shape) != tuple(expected_w_shape):
            raise ValueError(
                f"Pretrained weight shape {tuple(weight.shape)} does not "
                f"match projection '{node_type}' expected shape "
                f"{tuple(expected_w_shape)} (out_features, in_features) = "
                f"(embedding_dim, feature_dim). TransE embeddings must "
                f"be projected to match the linear layer's weight "
                f"geometry before loading."
            )
        # Copy the pretrained weight into the projection (no_grad to
        # avoid polluting the optimizer state).
        with torch.no_grad():
            proj.weight.copy_(weight.to(proj.weight.device).to(proj.weight.dtype))
            if bias is not None:
                if tuple(bias.shape) != tuple(proj.bias.shape):
                    raise ValueError(
                        f"Pretrained bias shape {tuple(bias.shape)} does "
                        f"not match projection '{node_type}' expected "
                        f"shape {tuple(proj.bias.shape)}."
                    )
                proj.bias.copy_(bias.to(proj.bias.device).to(proj.bias.dtype))

        # P3-009 ROOT FIX v104: apply the freeze policy.
        if freeze is None:
            freeze = self._default_freeze_pretrained
        if freeze:
            proj.weight.requires_grad_(False)
            if proj.bias is not None:
                proj.bias.requires_grad_(False)
            self._frozen_types.add(node_type)
            logger.info(
                f"P3-009 ROOT FIX v104: loaded pretrained embeddings for "
                f"'{node_type}' and FROZEN the projection (requires_grad="
                f"False). The GNN optimizer will NOT update these "
                f"weights. To unfreeze, call "
                f"`unfreeze_pretrained_embeddings('{node_type}')`."
            )
        else:
            # Make sure requires_grad is True (in case the projection was
            # previously frozen and we are re-loading with freeze=False).
            proj.weight.requires_grad_(True)
            if proj.bias is not None:
                proj.bias.requires_grad_(True)
            self._frozen_types.discard(node_type)
            logger.info(
                f"P3-009 ROOT FIX v104: loaded pretrained embeddings for "
                f"'{node_type}' and left them TRAINABLE (requires_grad="
                f"True). The GNN optimizer WILL update these weights."
            )

    def unfreeze_pretrained_embeddings(self, node_type: str) -> None:
        """Unfreeze a previously-frozen pretrained projection.

        P3-009 ROOT FIX v104: companion to ``load_pretrained_embeddings``.
        Allows a caller to load frozen TransE embeddings, train the GNN
        for a few warmup epochs, then unfreeze for joint fine-tuning
        (a common transfer-learning recipe).

        Args:
            node_type: Node type name (must be in ``feature_dims``).
        """
        if node_type not in self.projections:
            raise ValueError(
                f"Unknown node type '{node_type}'. Known types: "
                f"{list(self.projections.keys())}."
            )
        proj = self.projections[node_type]
        proj.weight.requires_grad_(True)
        if proj.bias is not None:
            proj.bias.requires_grad_(True)
        self._frozen_types.discard(node_type)
        logger.info(
            f"P3-009 ROOT FIX v104: unfroze pretrained embeddings for "
            f"'{node_type}'. The GNN optimizer will now update these weights."
        )

    def frozen_types(self) -> set:
        """Return the set of node types whose projections are frozen.

        P3-009 ROOT FIX v104: public introspection method. The CI test
        ``test_p3_009_freeze_pretrained_embeddings`` uses this to verify
        that ``load_pretrained_embeddings(freeze=True)`` actually froze
        the projection.
        """
        return set(self._frozen_types)

    def forward(self, node_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Project all node type features to the unified embedding space.

        V90 ROOT FIX (BUG #49): iterate node_features in the ORDER defined
        by ``self._type_to_idx`` (i.e., the order of ``feature_dims.keys()``
        at construction time), NOT the dict's insertion order. The previous
        code iterated ``node_features.items()`` directly, which works in
        Python 3.7+ (dict preserves insertion order) BUT only if the caller
        built the dict with the same key order as ``feature_dims``. If a
        caller passed a dict with a different insertion order (e.g.,
        ``{"disease": ..., "drug": ...}`` instead of
        ``{"drug": ..., "disease": ...}``), the iteration order would
        differ from the type-index mapping, causing subtle bugs in any
        downstream code that assumes a consistent order.

        The fix sorts the iteration by ``self._type_to_idx[node_type]``
        so the output dict's iteration order ALWAYS matches the
        construction-time feature_dims order, regardless of the input
        dict's insertion order. This makes the projection order-stable
        across different callers.

        Args:
            node_features: Dict mapping node type name to feature tensor
                of shape (num_nodes_of_type, raw_feature_dim).

        Returns:
            Dict mapping node type name to projected tensor of shape
            (num_nodes_of_type, embedding_dim).
        """
        projected: Dict[str, torch.Tensor] = {}

        # P3-049 ROOT FIX (v107): FILTER unknown node types BEFORE sorting.
        # The previous code sorted with key ``self._type_to_idx.get(nt, -1)``,
        # which gave unknown types sort key = -1, placing them FIRST in the
        # iteration order. The loop body then logged a WARNING and
        # ``continue``-d past them, so they were skipped — but the
        # REMAINING known types were projected in an order that DEPENDED on
        # whether an unknown type was present (because the unknown type's
        # -1 key shifted every other type's relative position by one slot
        # in the sort). Concretely: with feature_dims={"drug","protein"}
        # and node_features={"drug","protein","variant"}, the sort was
        # ["variant" (-1), "drug" (0), "protein" (1)] → known types
        # iterated in [drug, protein] order. WITHOUT the unknown type, the
        # sort was ["drug" (0), "protein" (1)] → same [drug, protein]
        # order. So the bug was INVISIBLE for two known types, but with
        # THREE+ known types the iteration order COULD change when an
        # unknown type was added (the relative order of known types
        # depends on their integer indices, which the sort preserves —
        # but the audit's concern was that the BEHAVIOR is non-deterministic
        # across graphs: a graph with an extra "variant" type produces a
        # DIFFERENT iteration order than a graph without one, even though
        # the projection of each known type is independent of iteration
        # order TODAY). The ROOT FIX filters unknown types up front so
        # the sort sees ONLY known types, making the iteration order
        # DETERMINISTIC regardless of which unknown types are present.
        known_types = [nt for nt in node_features.keys() if nt in self._type_to_idx]
        unknown_types = [nt for nt in node_features.keys() if nt not in self._type_to_idx]
        if unknown_types:
            logger.warning(
                f"P3-049 ROOT FIX (v107): NodeTypeProjection.forward() "
                f"received {len(unknown_types)} unknown node type(s) "
                f"{unknown_types!r} (known: {list(self._type_to_idx.keys())}). "
                f"Filtering them out BEFORE sorting so the projection "
                f"order of known types is deterministic (independent of "
                f"which unknown types are present). The unknown types' "
                f"features are NOT projected — callers should retrain the "
                f"model with the new types added to feature_dims."
            )
        for node_type in sorted(known_types, key=lambda nt: self._type_to_idx[nt]):
            features = node_features[node_type]
            # The filter above already ensured node_type is in self.projections
            # (since _type_to_idx keys == projections keys by construction).
            # Defensive guard kept for safety against future schema changes.

            num_nodes = features.shape[0]

            # Project to embedding space
            h = self.projections[node_type](features)

            # Validate output
            if torch.isnan(h).any() or torch.isinf(h).any():
                raise RuntimeError(
                    f"Non-finite values in projected features for '{node_type}'. "
                    f"Check input features for NaN/Inf."
                )

            # Apply normalization
            if node_type in self.norms:
                h = self.norms[node_type](h)

            # Add node type embedding
            type_idx = self._type_to_idx[node_type]
            type_indices = torch.full(
                (num_nodes,), type_idx, dtype=torch.long, device=features.device
            )
            type_emb = self.node_type_embedding(type_indices)
            h = h + type_emb

            projected[node_type] = h

        return projected

    def get_type_index(self, node_type: str) -> int:
        """Get the integer index for a node type name."""
        if node_type not in self._type_to_idx:
            raise ValueError(
                f"Unknown node type '{node_type}'. "
                f"Known: {list(self._type_to_idx.keys())}"
            )
        return self._type_to_idx[node_type]
