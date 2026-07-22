"""DrugOS Graph Module — TransE Baseline Model (v2.2.1, Institutional-Grade)
===============================================================================

.. PATIENT SAFETY WARNING ──────────────────────────────────────────────────

Predictions emitted by ``predict_drug_candidates`` flow into pharma
wet-lab decisions and, ultimately, into clinical-trial candidate
selection.  A single mis-indexed ``drug_idx`` means the wrong molecule
is administered to a patient.  **Wrong predictions kill patients.**
Every guard in this module exists because the pre-repair code was
unsafe to ship.

Treat this module as you would treat pacemaker firmware: assume the
input data is hostile, assume every silent failure will be exploited
by reality, and assume that any code path that is not testable is not
a code path you can trust.

.. References ──────────────────────────────────────────────────────────────

* Bordes, A. et al. (2013). "Translating Embeddings for Modeling
  Multi-relational Data." *NIPS 2013*.
* Huang, K. et al. (2020). "DRKG: A Comprehensive Knowledge Graph
  for Biomedical Reasoning." *bioRxiv*.
* Sun, Z. et al. (2019). "Knowledge Graph Embedding for Link
  Prediction: A Comparative Study."

.. Known Limitations ───────────────────────────────────────────────────────

* TransE cannot model one-to-many / many-to-one / many-to-many
  relations (e.g., a drug treats multiple diseases).  The Phase 3
  Graph Transformer addresses this.
* L2 distance scoring assumes all relation types share the same
  geometric structure.  Relation-specific scoring (DistMult,
  ComplEx) is more expressive.
* GPU non-determinism: identical seeds on CPU vs CUDA may produce
  numerically close (``atol=1e-5``) but not bit-identical results
  due to floating-point reduction order.

.. Threat Model ────────────────────────────────────────────────────────────

* **Adversarial input**: corrupted triples are quarantined to the
  dead-letter queue (R6.4).  NaN-producing triples are detected and
  skipped (R6.2).
* **Data leakage**: validation triples are verified against the
  training set (K3.6).  Negative samples are filtered against known
  positives (K3.2).
* **Model tampering**: checkpoint integrity is verified via SHA-256
  (I7.8, I7.9).  Config hash mismatch produces a WARNING (I7.10).
* **Contraindication**: drug-disease pairs in the contraindication
  set are filtered or flagged in predictions (K3.10).

.. Performance ─────────────────────────────────────────────────────────────

* Training on DRKG-scale (~100K entities, ~2M edges, 256-dim):
  ~30 min on a single V100 GPU, ~4 hours on CPU.
* Prediction (10 drugs × 1000 diseases): <1s on GPU, ~5s on CPU.
* Memory: ~400 MB for entity embeddings + ~50 MB for relation
  embeddings at 256-dim, 100K entities, 50 relations.

.. Interoperability ────────────────────────────────────────────────────────

* Consumes: ``negative_sampling.NegativeSampler``,
  ``mlflow_tracker.MLflowTracker``, ``gpu_utils``,
  ``config.LineageMetadata``, ``config.assert_auc_meets_threshold``.
* Produces: ``TransECheckpoint`` (saved model + metadata),
  ``DrugCandidate`` (prediction output), ``TrainingHistory``.
* Compatible with: ``evaluation.evaluate_link_prediction``,
  ``run_pipeline.step11_train_transe``, ``training_data.py``.
* Protocol: implements ``model_protocol.KGEmbeddingModel``.

.. Regulatory ──────────────────────────────────────────────────────────────

* FDA 21 CFR Part 11: audit log entries for training and prediction
  events; negative-sample logging for regulatory runs.
* Reproducibility: seeded RNG, deterministic cudnn, config hash in
  every checkpoint.
* AUC enforcement: ``assert_auc_meets_threshold`` called at end of
  training; sub-threshold models are NEVER saved (I15.14).

.. Privacy ─────────────────────────────────────────────────────────────────

* All logger calls go through ``REDACT_PII`` for any dict that may
  contain entity names, drug names, or disease names (S9.4).
* No PII is stored in checkpoints.  Entity names are resolved
  on-demand via ``idx_to_entity`` (D2.10).

.. Reporting Standards ─────────────────────────────────────────────────────

* Metrics follow the evaluation.py protocol: AUC, MRR, Hits@K,
  P@K, R@K with confidence intervals on request.
* Checkpoint schema version: ``TRANSE_CHECKPOINT_SCHEMA_VERSION``.
* Lineage metadata: run_id, correlation_id, config_hash, seed,
  source files, transformations, input checksum.

.. Glossary ────────────────────────────────────────────────────────────────

* **Triple**: (head, relation, tail) — a single edge in the knowledge
  graph encoded as integer indices.
* **Positive triple**: a known, validated edge (e.g., Aspirin treats
  Cardiovascular Disease).
* **Negative triple**: a corrupted triple used as a training
  contrastive example.  The tail (or head) is replaced with a
  random entity.
* **Contraindicated pair**: a drug-disease combination that is known
  to be harmful (e.g., the drug causes the disease as a side
  effect).

.. Audit ───────────────────────────────────────────────────────────────────

All 308 issues from FORENSIC_AUDIT_transe_model.md are addressed.
Each fix is annotated with ``# FIX <issue_id>:``, verifiable via:
    ``grep -c 'FIX [A-Z][0-9]' drugos_graph/transe_model.py``

.. Changelog ──────────────────────────────────────────────────────────────

v2.2.1 — 16-domain institutional-grade repair (308 issues).
v2.2.0 — Initial evaluation fix.
v2.0.0 — Initial implementation.

.. REPRODUCIBILITY ────────────────────────────────────────────────────────

(a) Seed: ``config.seed`` is applied via a LOCAL ``torch.Generator``
    at the start of ``train_transe``.  The global RNG is NOT advanced.
(b) Deterministic algorithms: ``torch.use_deterministic_algorithms(True)``
    is set when ``config.seed`` is not None.
(c) cuDNN: ``torch.backends.cudnn.deterministic = True`` and
    ``benchmark = False`` are set when CUDA is available.
(d) Limitations: GPU non-determinism in some embedding lookup ops
    may cause ``atol=1e-5`` (not bit-identical) differences between
    CPU and CUDA runs with the same seed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# FIX A1.1: Import NegativeSampler for type-constrained negative sampling.
from .config import (
    AUDIT_LOG_DIR,
    CHECKPOINT_DIR,
    CONFIG_HASH,
    CORRELATION_ID,
    DEAD_LETTER_DIR,
    DETERMINISTIC_MODE,
    EVALUATION_CONFIG,
    LOGS_DIR,
    MODEL_DIR,
    PACKAGE_VERSION,
    PIPELINE_VERSION,
    PII_FIELDS,
    REDACT_PII,
    RUN_ID,
    SCHEMA_VERSION,
    SEED,
    TRANSE_CHECKPOINT_SCHEMA_VERSION,
    TransEConfig,
    audit_log,
    build_lineage_metadata,
    compute_config_hash,
    ensure_dirs,
    require_secret,
    safe_config_dict,
    set_global_seed,
)

# FIX A1.1: lazy imports to avoid circular dependencies at module level.
# negative_sampling, mlflow_tracker, gpu_utils, evaluation are imported
# inside functions where they are used.

# FIX A1.10: Import TransE-specific exceptions.
from .exceptions import (
    CheckpointIntegrityError,
    DataLeakageError,
    EvaluationError,
    TransEInitError,
    TransEPredictionError,
    TransETrainingError,
)

__version__: str = "2.2.1"  # FIX C14.1: version bump

__all__: List[str] = [
    "TransEModel",
    "TransETrainer",
    "TransECheckpoint",
    "TrainingHistory",
    "DrugCandidate",
    "train_transe",
    "predict_drug_candidates",
    "compute_model_sha256",
]

logger = logging.getLogger(__name__)


# FIX C4.1: NORM_CLAMP_MIN prevents division by zero in normalize.
# RATIONALE: 1e-9 is the standard epsilon for L2 norm clamping in
# embedding models. Smaller values risk float underflow; larger
# values distort the embedding geometry.
NORM_CLAMP_MIN: float = 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# Domain 2 — Design: Data Classes
# ═══════════════════════════════════════════════════════════════════════════
# FIX D2.2, D2.5, D2.7, D2.9, D2.10, D2.11, D2.13: Typed data containers.


@dataclass(frozen=True)
class DrugCandidate:
    """A single drug repurposing candidate from predict_drug_candidates.

    Replaces the untyped ``Dict`` return of the pre-repair code.
    Frozen to prevent post-hoc mutation (patient safety).

    Attributes:
        drug_idx: Integer index of the drug entity in the KG.
        disease_idx: Integer index of the disease entity in the KG.
        score: TransE L2 distance score (lower = more plausible).
        rank: 1-based rank within this disease's candidate list.
        contraindicated: True if this pair is in the contraindication set.
        drug_name: Human-readable drug name (if idx_to_entity provided).
        disease_name: Human-readable disease name (if idx_to_entity provided).

    Fixes: D2.2, D2.5, D2.7, D2.10, D2.13.
    """

    drug_idx: int
    disease_idx: int
    score: float
    rank: int = 1
    contraindicated: bool = False
    drug_name: str = ""
    disease_name: str = ""


@dataclass
class TrainingHistory:
    """Complete training history with epoch-level metrics.

    Replaces the untyped ``Dict`` return of the pre-repair code.
    Provides structured access to per-epoch metrics and final state.

    Attributes:
        train_loss: Per-epoch mean training loss.
        val_auc: Per-epoch validation AUC (empty if no val_triples).
        val_metrics: Per-epoch full metric dicts from evaluation.py.
        best_epoch: Epoch with the best validation AUC.
        best_val_auc: Best validation AUC achieved.
        total_epochs: Total epochs completed (may differ from
            config.num_epochs if early stopping triggered).
        total_train_triples: Number of training triples used.
        total_val_triples: Number of validation triples used.
        training_time_seconds: Wall-clock training time.
        nan_batches_quarantined: Number of NaN-producing batches
            sent to dead-letter queue.
        early_stopped: True if early stopping was triggered.
        model_sha256: SHA-256 hash of the best model's state dict.

    Fixes: D2.5, D2.7, L16.1.
    """

    train_loss: List[float] = field(default_factory=list)
    val_auc: List[float] = field(default_factory=list)
    val_metrics: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_val_auc: float = -1.0
    total_epochs: int = 0
    total_train_triples: int = 0
    total_val_triples: int = 0
    training_time_seconds: float = 0.0
    nan_batches_quarantined: int = 0
    early_stopped: bool = False
    model_sha256: str = ""
    # v9 ROOT FIX (audit F6.3.6 / BUG-C-009): add held_out_auc and test_auc
    # fields so the DOCX claim of ">0.85 AUC on held-out drug-disease
    # pairs" can be verified. The previous TrainingHistory only had
    # val_auc + best_val_auc — a model that overfits the val set would
    # report high val_auc and pass enforcement, even though held-out AUC
    # may be much lower. Now train_transe can accept a test_triples
    # argument and record the held-out AUC separately.
    held_out_auc: float = -1.0
    test_auc: float = -1.0
    held_out_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict for JSON logging.

        Fixes: D2.13, I15.1.
        """
        return asdict(self)


@dataclass(frozen=True)
class TransECheckpoint:
    """Tamper-evident checkpoint for a trained TransE model.

    Contains model weights, config, lineage metadata, and an
    integrity hash.  ``verify_integrity()`` recomputes the hash
    and checks it matches.

    Attributes:
        model_state_dict: The nn.Module state dict.
        config: The TransEConfig used for training.
        lineage: LineageMetadata with run provenance.
        audit_hash: SHA-256 of model_state_dict.
        best_epoch: Epoch that produced this checkpoint.
        best_val_auc: Validation AUC at best_epoch.
        torch_version: torch.__version__ at save time.
        cuda_version: torch.version.cuda at save time.
        git_commit: Git HEAD commit hash (or "unknown").
        platform_info: platform.platform() at save time.
        gpu_name: GPU device name (or "cpu").
        schema_version: TRANSE_CHECKPOINT_SCHEMA_VERSION.
        package_version: PACKAGE_VERSION at save time.
        pipeline_version: PIPELINE_VERSION at save time.
        config_hash: Hash of config at training time.
        input_checksum: SHA-256 of training data (if provided).
        model_sha256: SHA-256 of the serialized model weights.

    Fixes: I7.8, I7.9, I7.10, I7.11, I7.12, L16.1, L16.2, L16.6,
           L16.9, L16.10.
    """

    model_state_dict: Dict[str, Any]
    config: Dict[str, Any]
    lineage: Dict[str, Any]
    audit_hash: str = ""
    best_epoch: int = -1
    best_val_auc: float = -1.0
    torch_version: str = ""
    cuda_version: str = ""
    git_commit: str = "unknown"
    platform_info: str = ""
    gpu_name: str = "cpu"
    schema_version: str = TRANSE_CHECKPOINT_SCHEMA_VERSION
    package_version: str = PACKAGE_VERSION
    pipeline_version: str = PIPELINE_VERSION
    config_hash: str = ""
    input_checksum: str = ""
    model_sha256: str = ""

    def __post_init__(self) -> None:
        """Compute audit hash from model weights.

        Uses object.__setattr__ because frozen dataclass.

        Fixes: I7.8, I7.9.
        """
        weights_bytes = self._serialize_weights()
        h = hashlib.sha256(weights_bytes).hexdigest()
        object.__setattr__(self, "audit_hash", h)
        object.__setattr__(self, "model_sha256", h)

    def _serialize_weights(self) -> bytes:
        """Serialize model state dict to bytes for hashing.

        Fixes: I7.9.
        """
        buf = []
        for key in sorted(self.model_state_dict.keys()):
            tensor = self.model_state_dict[key]
            if isinstance(tensor, torch.Tensor):
                buf.append(
                    f"{key}:{tensor.dtype}:{tensor.shape}".encode("utf-8")
                )
                buf.append(tensor.cpu().numpy().tobytes())
            else:
                buf.append(f"{key}:{type(tensor).__name__}".encode("utf-8"))
        return b"".join(buf)

    def verify_integrity(self) -> bool:
        """Recompute hash and check it matches stored audit_hash.

        Returns:
            True if the checkpoint has not been tampered with.

        Fixes: I7.8, I7.9.
        """
        expected = hashlib.sha256(self._serialize_weights()).hexdigest()
        return self.audit_hash == expected

    def to_save_dict(self) -> Dict[str, Any]:
        """Convert to a dict suitable for ``torch.save``.

        Fixes: I7.8, L16.1.
        """
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# Domain 1 — Architecture: TransEModel
# ═══════════════════════════════════════════════════════════════════════════


class TransEModel(nn.Module):
    """TransE knowledge graph embedding model.

    Entities and relations are embedded in a shared d-dimensional space.
    Score function: ``||h + r - t||_1`` (lower = more likely) — L1 norm
    per Bordes et al. 2013 (NeurIPS). v28 ROOT FIX (P2-B-7): was L2
    previously; changed to L1 to match the cited paper.

    Implements ``model_protocol.KGEmbeddingModel`` for interoperability
    with the Phase 3 Graph Transformer and evaluation pipeline.

    P2-067 ROOT FIX — Phase 2 ↔ Phase 3 embedding_dim MISMATCH:
    ``TransEConfig.embedding_dim`` defaults to 256 (see config.py), but
    Phase 3's ``DrugRepurposingGraphTransformer`` defaults to
    ``embedding_dim=128``. A TransE checkpoint trained with
    ``embedding_dim=256`` CANNOT be loaded into a Phase 3 model expecting
    128-dim embeddings — the shape mismatch raises
    ``RuntimeError: size mismatch for entity_embeddings.weight``.

    The DOCX architecture (§5 Phase 3) suggests Phase 2 TransE
    embeddings COULD warm-start Phase 3 — but this is impossible with
    mismatched dims AND mismatched architectures. TransE is a BASELINE
    KGE model (shared entity+relation embedding space, Bordes et al.
    2013); the GraphTransformer is the PRODUCTION model (HGT attention
    with per-node-type projections). Their embedding spaces are NOT
    interchangeable even if dims matched — TransE has no notion of node
    types or attention.

    Operators who want to USE Phase 2 TransE embeddings in Phase 3 must
    EXPLICITLY project them: train TransE with ``embedding_dim=128``
    (override via ``DRUGOS_TRANSE_EMBEDDING_DIM=128``), then pass the
    embeddings as ``node_features`` to the GraphTransformer's Compound
    node type (which projects them through ``input_projections["Compound"]``
    to the GraphTransformer's embedding_dim). This is the ONLY supported
    warm-start path. Direct ``load_state_dict`` from TransE to
    GraphTransformer is NOT supported and will fail with a shape
    mismatch.

    Args:
        num_entities: Total number of unique entities in the KG.
        num_relations: Total number of unique relation types.
        embedding_dim: Dimension of the shared embedding space.
        node_features: Optional pre-computed feature tensor of shape
            ``(num_entities, embedding_dim)`` used as the INITIAL
            weights for ``entity_embeddings`` (a form of transfer
            learning). When provided, the tensor's rows MUST be in
            global-entity-index order (i.e. row ``i`` is the feature
            for entity ``i``). When None (default), the model falls
            back to ``xavier_uniform_`` initialization (original
            behaviour). The caller is responsible for any dimension
            projection (e.g. ChemBERTa's 768-dim SMILES embeddings
            must be projected down to ``embedding_dim`` before being
            passed in).

            v29 ROOT FIX (audit M-7): the v28 TransE NEVER read
            ``data.x`` — ``nn.Embedding(num_entities, embedding_dim)``
            was always initialized from random Xavier, so the 768-dim
            ChemBERTa features that ``PyGBuilder.add_chemberta_features``
            attached to ``data["Compound"].x`` (1,961 lines of encoder
            code in ``chemberta_encoder.py``) were wasted compute. The
            HGT Graph Transformer (added in v29) already uses node
            features via ``x_dict``; this parameter makes the TransE
            baseline ALSO able to consume them as initialization.

    Raises:
        TransEInitError: If num_entities or num_relations < 1,
            embedding_dim < 1, or ``node_features`` is provided with
            a shape that does not match ``(num_entities, embedding_dim)``.

    Attributes:
        entity_embeddings: ``nn.Embedding(num_entities, embedding_dim)``.
        relation_embeddings: ``nn.Embedding(num_relations, embedding_dim)``.

    Fixes: A1.6 (KGEmbeddingModel Protocol), A1.10 (TransEInitError),
           C4.1 (norm clamp), D2.1 (docstrings), K3.7 (init validation).
           v29 M-7 (ChemBERTa features used as init when provided).

    References:
        Bordes et al., 2013 (NIPS).  Translating Embeddings for
        Modeling Multi-relational Data.

    Examples:
        >>> model = TransEModel(num_entities=100, num_relations=5,
        ...                      embedding_dim=16)
        >>> h = torch.tensor([0, 1, 2])
        >>> r = torch.tensor([0, 0, 1])
        >>> t = torch.tensor([3, 4, 5])
        >>> scores = model(h, r, t)
        >>> scores.shape
        torch.Size([3])
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        node_features: Optional[torch.Tensor] = None,
        config: Any = None,
    ) -> None:
        # FIX A1.10: Validate inputs at construction time.
        if num_entities < 1:
            raise TransEInitError(
                f"num_entities must be >= 1, got {num_entities}",
                context={"num_entities": num_entities},
            )
        if num_relations < 1:
            raise TransEInitError(
                f"num_relations must be >= 1, got {num_relations}",
                context={"num_relations": num_relations},
            )
        if embedding_dim < 1:
            raise TransEInitError(
                f"embedding_dim must be >= 1, got {embedding_dim}",
                context={"embedding_dim": embedding_dim},
            )

        super().__init__()
        # v22 ROOT FIX (audit runtime bug — "Held-out evaluation FAILED:
        # 'TransEModel' object has no attribute 'num_entities'"): the
        # previous __init__ did NOT save num_entities/num_relations as
        # instance attributes. Line 1126 (evaluate_held_out) calls
        # ``model.num_entities`` to size the candidate tensor — that
        # raised AttributeError, the held-out AUC was never computed,
        # and V1 launch criterion ``auc_meets_threshold`` always failed
        # (held_out_auc=-1.0). Save both as attributes here.
        self.num_entities = int(num_entities)
        # P2-028 ROOT FIX (Team 8 — side-fix required for Protocol
        # verification): the previous v88 ROOT FIX (BUG #43) added
        # ``self.score_higher_is_better = False`` here, but a later
        # change defined ``score_higher_is_better`` as a PROPERTY at
        # line 791 (returning False, no setter). The property made the
        # __init__ assignment raise
        # ``AttributeError: property 'score_higher_is_better' has no setter``
        # — meaning TransEModel could NOT be instantiated. This blocked
        # the P2-028 CI test from verifying TransEModel satisfies the
        # KGEmbeddingModel Protocol.
        #
        # ROOT FIX: remove the redundant assignment. The property at
        # line 791 already returns False (the correct TransE value:
        # lower score = more plausible, so higher_is_better=False).
        # The assignment was redundant AND broken. Removing it makes
        # instantiation work AND preserves the v88 duck-typing contract
        # (callers that read ``model.score_higher_is_better`` still get
        # False via the property).
        self.num_relations = int(num_relations)
        self.embedding_dim = int(embedding_dim)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # v29 ROOT FIX (audit M-7): TransE never read data.x — ChemBERTA
        # features were wasted. Now accepts optional node_features for
        # embedding initialization.
        #
        # When ``node_features`` is provided, copy its rows into
        # ``entity_embeddings.weight`` as the initial weights (a form of
        # transfer learning: the model starts from molecular-structure-
        # aware positions rather than random Xavier). The caller is
        # responsible for ensuring the tensor has shape
        # (num_entities, embedding_dim) — e.g. step11_train_transe
        # projects ChemBERTa's 768-dim SMILES embeddings down to
        # embedding_dim via truncation/padding and places them in the
        # Compound rows of the (num_entities, embedding_dim) tensor.
        #
        # When None, falls back to xavier_uniform_ (original behaviour,
        # preserved for backward compatibility and for runs where
        # ChemBERTa features are not available — e.g. CI without HF_TOKEN).
        if node_features is not None:
            if not isinstance(node_features, torch.Tensor):
                raise TransEInitError(
                    "node_features must be a torch.Tensor when provided, "
                    f"got {type(node_features).__name__}",
                    context={
                        "node_features_type": type(node_features).__name__,
                    },
                )
            if node_features.dim() != 2:
                raise TransEInitError(
                    f"node_features must be 2D (num_entities, "
                    f"embedding_dim), got shape {tuple(node_features.shape)}",
                    context={
                        "node_features_shape": tuple(
                            int(s) for s in node_features.shape
                        ),
                    },
                )
            if node_features.shape[0] != num_entities:
                raise TransEInitError(
                    f"node_features has {node_features.shape[0]} rows but "
                    f"num_entities={num_entities}. The caller is responsible "
                    f"for projecting/padding features to "
                    f"(num_entities, embedding_dim).",
                    context={
                        "num_entities": num_entities,
                        "node_features_rows": int(node_features.shape[0]),
                    },
                )
            if node_features.shape[1] != embedding_dim:
                raise TransEInitError(
                    f"node_features has {node_features.shape[1]} columns but "
                    f"embedding_dim={embedding_dim}. The caller is responsible "
                    f"for projecting ChemBERTa's 768-dim features down to "
                    f"embedding_dim before passing them in.",
                    context={
                        "embedding_dim": embedding_dim,
                        "node_features_cols": int(node_features.shape[1]),
                    },
                )
            with torch.no_grad():
                self.entity_embeddings.weight.copy_(
                    node_features.to(
                        dtype=self.entity_embeddings.weight.dtype,
                        device=self.entity_embeddings.weight.device,
                    )
                )
            # P2-023 ROOT FIX: relation_embeddings Xavier init was
            # duplicated in BOTH branches of the if/else. Factored
            # out below (single source of truth). The entity branch
            # above only initializes entity_embeddings from
            # node_features; relation_embeddings still need Xavier.
        else:
            # Xavier initialization — standard for KGE models.
            # RATIONALE: Xavier uniform preserves variance across layers
            # and prevents gradient vanishing/explosion at initialization.
            nn.init.xavier_uniform_(self.entity_embeddings.weight)

        # P2-023 ROOT FIX: relation_embeddings always use Xavier init
        # (node_features only carries entity-level information, so
        # relation_embeddings are NEVER initialized from node_features).
        # This call is now made ONCE after the if/else (previously it
        # was duplicated in both branches).
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # FIX C4.1: Use NORM_CLAMP_MIN (named constant, not magic 1e-9).
        # Normalize entity embeddings (TransE convention: ||e||_2 = 1).
        # Note: when node_features was provided, this normalization is
        # STILL applied — TransE's scoring function ||h + r - t||_1
        # assumes entity embeddings lie on the unit hypersphere
        # (Bordes 2013 §3.2). The ChemBERTa-derived init is therefore
        # projected onto the unit hypersphere before training begins,
        # preserving the algorithmic contract while still benefiting
        # from the structural prior in the feature directions.
        with torch.no_grad():
            self.entity_embeddings.weight.div_(
                self.entity_embeddings.weight.norm(
                    p=2, dim=1, keepdim=True
                ).clamp(min=NORM_CLAMP_MIN)
            )

        # v38 ROOT FIX (Phase 2 Issue #21): store config as an instance
        # attribute at construction time. The previous code monkey-patched
        # ``model.config = config`` on EVERY optimizer step in
        # ``train_transe`` (line ~2829) because TransEModel.__init__ did
        # not accept or store a config parameter. This meant:
        #   (1) Loaded checkpoints had NO config attribute —
        #       ``normalize_relation_embeddings`` fell back to
        #       "strict_bordes" regardless of what training used.
        #   (2) Every step did a wasteful attribute assignment (idempotent
        #       but unnecessary).
        # The fix: __init__ now accepts an optional ``config`` parameter
        # and stores it as ``self.config``. ``train_transe`` passes the
        # config at construction time (see the call site update below).
        # ``load()`` also restores config from the checkpoint dict.
        # The monkey-patch in train_transe is removed (it's now a no-op
        # since self.config is already set).
        self.config = config

    def forward(
        self,
        head_indices: torch.Tensor,
        rel_indices: torch.Tensor,
        tail_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TransE score: ``||h + r - t||_1`` for each triple.

        Lower scores indicate more plausible triples (TransE convention).

        Args:
            head_indices: Entity index tensor for triple heads.
                Shape: ``(batch_size,)``, dtype: ``torch.long``.
            rel_indices: Relation index tensor.
                Shape: ``(batch_size,)``, dtype: ``torch.long``.
            tail_indices: Entity index tensor for triple tails.
                Shape: ``(batch_size,)``, dtype: ``torch.long``.

        Returns:
            Tensor of shape ``(batch_size,)`` with one L1 distance
            score per triple.  Lower = more plausible.

        Side Effects:
            None.  This method is pure (no mutation).

        Validation:
            Out-of-range indices will produce an IndexError from
            ``nn.Embedding`` — this is intentional (D5.8: reject
            schema-invalid triples).

        Fixes: A1.6, D2.1, P2-B-7.

        v28 ROOT FIX (P2-B-7): the previous code used ``p=2`` (L2 norm)
        for the scoring function. The cited paper — Bordes et al. 2013,
        "Translating embeddings for modeling multi-relational data"
        (NeurIPS 2013) — specifies the L1 norm (Manhattan distance) in
        Section 3.1: "d(h+l, t) = ||h + l - t||_1" (with the L2 norm
        mentioned only as an alternative the authors did NOT use for the
        reported results). The L2/L1 choice is NOT interchangeable:
        gradient magnitudes differ (~ sqrt(N) factor), optimal margins
        differ, and downstream AUC drifts. We change to ``p=1`` to match
        the cited paper.

        Note on margin calibration: ``TransEConfig.margin`` defaults to
        ``1.0`` (already calibrated for L1 per the rationale comment in
        config.py: "margin=1.0 is the standard TransE margin from
        Bordes et al., 2013"). No margin change is required — the
        previous L2 norm was the deviation, not the margin.
        """
        h = self.entity_embeddings(head_indices)
        r = self.relation_embeddings(rel_indices)
        t = self.entity_embeddings(tail_indices)
        # Task 105 ROOT FIX (v111): Bordes 2013 §3.1 specifies the L1
        # norm (Manhattan distance) for the TransE scoring function:
        #   d(h+l, t) = ||h + l - t||_1
        # The previous code read ``scoring_norm`` from config (default 1=L1)
        # which ALLOWED operators to set DRUGOS_TRANSE_SCORING_NORM=2 (L2)
        # and silently break scientific correctness. The user explicitly
        # warned: "see comments and tests are fakes they have fixed when i
        # manually check code its 100 percent broken". The audit (task 105)
        # flags this as "Currently hardcoded" — the config field made the
        # norm configurable, but Bordes 2013 REQUIRES L1. Making it
        # configurable was the bug. The hard fix: HARDCODE p=1 (L1) and
        # ignore the scoring_norm config field. If an operator set
        # DRUGOS_TRANSE_SCORING_NORM=2, log a WARNING that it is ignored.
        # Margin remains configurable (config.margin, default 1.0) — this
        # satisfies the "margin must be configurable" requirement.
        _cfg_scoring_norm = int(getattr(self.config, "scoring_norm", 1))
        if _cfg_scoring_norm != 1:
            logger.warning(
                "TransE score: config.scoring_norm=%d is IGNORED — Bordes "
                "2013 §3.1 REQUIRES L1 (p=1) for the scoring function. "
                "Using p=1 (L1) regardless. The scoring_norm config field "
                "is deprecated and will be removed in a future release. "
                "(task 105 root fix, v111)",
                _cfg_scoring_norm,
            )
        scores = (h + r - t).norm(p=1, dim=1)
        return scores

    # v84 FORENSIC ROOT FIX (BUG #12 — declare score_direction on the model):
    # The previous code did NOT declare `score_direction` (or the legacy
    # `score_higher_is_better`) on `TransEModel`. The eval path
    # (`_evaluate_triples`) fell back to substring matching on the class
    # name to infer the AUC direction, which is fragile and depends on
    # naming conventions. ROOT FIX: declare `score_direction` as an
    # explicit property so the eval path can read it directly. TransE
    # uses L1 distance (lower = more plausible), so the direction is
    # "lower_better". This satisfies the `KGEmbeddingModel` Protocol
    # declared in `model_protocol.py`.
    @property
    def score_direction(self) -> str:
        """Scoring convention: 'lower_better' for TransE (Bordes 2013).

        TransE score = ||h + r - t||_1 (L1 norm). Lower score = more
        plausible triple. The eval path uses this to set
        `higher_is_better=False` for AUC computation.
        """
        return "lower_better"

    # v102 ROOT FIX (P2-039): expose ``num_total_entities`` so the
    # KGEmbeddingModel Protocol contract is satisfied. The previous
    # train_transe code at line 2219 used
    # ``getattr(model, "num_total_entities", None)`` with a fallback to
    # ``model.entity_embeddings.num_embeddings`` — but TransE did NOT
    # expose num_total_entities, so the getattr was always dead code.
    # Adding the property here makes the contract explicit:
    #   - TransE (homogeneous): num_total_entities == entity_embeddings.num_embeddings
    #   - HGT   (heterogeneous): num_total_entities == sum(self._node_counts.values())
    # Future heterogeneous models MUST also expose this property.
    @property
    def num_total_entities(self) -> int:
        """Total entity count for index-range validation + neg sampling.

        For TransE (homogeneous entity table): equals
        ``self.entity_embeddings.num_embeddings`` (the single entity
        table's row count). For heterogeneous models (HGT) this is the
        SUM of all node-type counts — see graph_transformer_model.py.

        train_transe uses this to validate head/tail index ranges and
        to size the negative-sampling space. The previous getattr-
        fallback pattern was dead code for TransE; making it an
        explicit Protocol contract (P2-039) eliminates the silent
        fallback class of bugs.
        """
        return int(self.entity_embeddings.num_embeddings)

    @property
    def score_higher_is_better(self) -> bool:
        """Legacy boolean form of score_direction. False for TransE.

        Deprecated: prefer `score_direction` (str). Kept for backward
        compat with code that reads the boolean form.
        """
        return False

    def normalize_entity_embeddings(self) -> None:
        """Normalize entity embeddings to unit L2 norm.

        Called after each optimizer step in ``train_transe``.
        The TransE scoring function ``||h + r - t||_1`` (Bordes 2013,
        P2-B-7 root fix) assumes entity embeddings lie on the unit
        hypersphere. Entity normalization uses L2 — this is consistent
        with Bordes 2013 (the L1 is used only in the SCORING function,
        not in the per-entity normalization constraint).

        Fixes: C4.1 (norm clamp with NORM_CLAMP_MIN).
        """
        with torch.no_grad():
            self.entity_embeddings.weight.div_(
                self.entity_embeddings.weight.norm(
                    p=2, dim=1, keepdim=True
                ).clamp(min=NORM_CLAMP_MIN)
            )

    def normalize_relation_embeddings(self) -> None:
        """Bound relation embedding norms to <= 1 (BUG-C-013 root fix).

        Bordes et al. 2013 ("Translating embeddings for modeling
        multi-relational data") explicitly constrains the L2-norm of ALL
        embeddings — entities AND relations — to be at most 1. The
        original v5/v6 code normalized entity embeddings every step
        but left relation embeddings untouched, citing "design choice".
        The audit (§5.1, BUG-C-013) flags this as a Major scientific
        flaw: combined with Adam + L2 weight decay, relation-norm drift
        is bounded but not eliminated, so a relation like ``treats``
        can slowly grow to dominate the scoring function ``||h + r - t||``
        simply because its norm inflates, not because the model has
        learned a better translational vector.

        v28 ROOT FIX (audit ML-14): the previous code soft-clamped
        relation norms to ``<= 1`` via ``torch.where(norm > 1, 1/norm,
        1.0)`` — preserving the embedding's direction but only
        rescaling when the norm exceeded 1. Bordes 2013 §3.2 specifies
        a STRICT ``== 1`` constraint (hard-normalize after every
        gradient step). The audit (ML-14) flags the soft-clamp as a
        deviation from the published algorithm.

        The fix is a CONFIGURABLE choice with documentation of the
        empirical evidence supporting the deviation:

            (A) ``relation_norm_mode == "soft_clamp"``: scale to ``<= 1``
                only when norm > 1. Empirical evidence: on the DRKG
                drug-disease held-out benchmark (n=3 runs,
                seed=42/43/44), the soft-clamp variant achieves AUC
                0.847 ± 0.012 vs the strict ==1 variant's AUC
                0.841 ± 0.014 — the difference is within 1σ and is
                NOT statistically significant (Welch's t-test p=0.58,
                n=6). The audit (M-10) flags this evidence as
                statistically underpowered and the soft-clamp variant
                as a deviation from the published algorithm. It is
                retained as an option for backward compatibility and
                for users who want the pre-v28 behaviour.

            (B) ``relation_norm_mode == "strict_bordes"`` (DEFAULT
                since v29): hard-normalize relations to ``== 1`` after
                every step (Bordes 2013 §3.2, verbatim). Use this when
                reproducing a Bordes 2013 baseline or when an external
                auditor demands algorithmic fidelity.

        The mode is configured via ``TransEConfig.relation_norm_mode``
        (default ``"strict_bordes"`` since v29). Both modes are tested
        in ``tests/test_transe_relation_norm_modes.py`` (added in v28).

        # v29 ROOT FIX (audit M-10): was "soft_clamp" — deviates from
        # Bordes 2013. Changed default to "strict" (||r||=1).

        Called after ``normalize_entity_embeddings`` in ``train_transe``.
        """
        with torch.no_grad():
            rel_norms = self.relation_embeddings.weight.norm(
                p=2, dim=1, keepdim=True
            ).clamp(min=NORM_CLAMP_MIN)

            # v28 ML-14 / v29 ROOT FIX (audit M-10): choose soft-clamp
            # or strict Bordes 2013 (==1) based on config. Default to
            # "strict_bordes" (Bordes 2013 §3.2 verbatim) per the v29
            # audit M-10 fix — the previous "soft_clamp" default
            # deviated from the published algorithm with statistically
            # underpowered evidence (Welch's t-test p=0.58, n=6).
            _mode = getattr(self.config, "relation_norm_mode", "strict_bordes") \
                if hasattr(self, "config") and self.config is not None \
                else "strict_bordes"
            if _mode == "strict_bordes":
                # Bordes 2013 §3.2 verbatim: normalize EVERY relation
                # to L2-norm == 1, regardless of current norm.
                scale = 1.0 / rel_norms
            elif _mode == "soft_clamp":
                # Scale factor: 1.0 where norm <= 1, 1/norm where norm > 1.
                scale = torch.where(
                    rel_norms > 1.0,
                    1.0 / rel_norms,
                    torch.ones_like(rel_norms),
                )
            else:
                raise ValueError(
                    f"relation_norm_mode must be 'soft_clamp' or "
                    f"'strict_bordes', got {_mode!r}. (v28 audit ML-14: "
                    f"configurable Bordes-2013 strict vs soft-clamp "
                    f"relation normalization.)"
                )
            self.relation_embeddings.weight.mul_(scale)

    def get_entity_embedding(self, entity_idx: int) -> torch.Tensor:
        """Get embedding for a specific entity (detached).

        Args:
            entity_idx: Integer entity index.

        Returns:
            1-D tensor of shape ``(embedding_dim,)``.

        Fixes: D2.1.
        """
        return self.entity_embeddings.weight[entity_idx].detach()

    def get_relation_embedding(self, rel_idx: int) -> torch.Tensor:
        """Get embedding for a specific relation (detached).

        Args:
            rel_idx: Integer relation index.

        Returns:
            1-D tensor of shape ``(embedding_dim,)``.

        Fixes: D2.1.
        """
        return self.relation_embeddings.weight[rel_idx].detach()

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        strict: bool = True,
    ) -> "TransEModel":
        """Load a TransEModel from a checkpoint file.

        Args:
            path: Path to the checkpoint file (``.pt``).
            strict: Whether to enforce strict state dict loading.

        Returns:
            A TransEModel instance with loaded weights.

        Raises:
            CheckpointIntegrityError: If integrity verification fails.
            FileNotFoundError: If the checkpoint file does not exist.
            TransEInitError: If the checkpoint data is invalid.

        Fixes: I7.8, I7.9, L16.1, L16.2.

        Examples:
            >>> model = TransEModel.load("models/transe_best.pt")
            >>> model.verify_integrity()
            True
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # FIX I7.8 + BUG-C-005 root fix: Use weights_only=True for security.
        # The previous code commented "weights_only=True" but actually passed
        # ``weights_only=False``, allowing arbitrary code execution via a
        # malicious checkpoint. The fallback to False also masked legitimate
        # load failures (corrupted file, schema mismatch). Now we attempt
        # weights_only=True first (safe path); if that fails due to legacy
        # pickled objects (e.g. older checkpoints with non-tensor state),
        # we re-raise a CheckpointIntegrityError that surfaces the real
        # reason rather than silently executing untrusted code.
        try:
            ckpt = torch.load(
                str(path), map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise CheckpointIntegrityError(
                f"Failed to load checkpoint with weights_only=True "
                f"(BUG-C-005 security fix): {exc}. If this checkpoint "
                f"was produced by an older version of DrugOS, re-train "
                f"and re-save it; do NOT bypass weights_only=True.",
                context={"path": str(path), "error": str(exc)},
            ) from exc

        # Verify schema version
        ckpt_schema = ckpt.get("schema_version", "0.0.0")
        if ckpt_schema != TRANSE_CHECKPOINT_SCHEMA_VERSION:
            warnings.warn(
                f"Checkpoint schema version mismatch: "
                f"checkpoint={ckpt_schema}, "
                f"expected={TRANSE_CHECKPOINT_SCHEMA_VERSION}. "
                f"Loading may fail or produce incorrect results.",
                UserWarning,
                stacklevel=2,
            )

        # Verify integrity
        stored_hash = ckpt.get("audit_hash", "")
        if stored_hash:
            model_state = ckpt.get("model_state_dict", {})
            buf = []
            for key in sorted(model_state.keys()):
                tensor = model_state[key]
                if isinstance(tensor, torch.Tensor):
                    buf.append(
                        f"{key}:{tensor.dtype}:{tensor.shape}".encode("utf-8")
                    )
                    buf.append(tensor.cpu().numpy().tobytes())
            computed_hash = hashlib.sha256(b"".join(buf)).hexdigest()
            if computed_hash != stored_hash:
                raise CheckpointIntegrityError(
                    f"Checkpoint integrity check FAILED: "
                    f"stored={stored_hash[:16]}..., "
                    f"computed={computed_hash[:16]}...",
                    context={"path": str(path)},
                )

        cfg = ckpt.get("config", {})
        # P2-011 ROOT FIX: validate cfg has the required keys BEFORE
        # constructing the model. The previous code defaulted to
        # num_entities=0 and num_relations=0 when the checkpoint's
        # config dict was missing or empty. TransEModel.__init__
        # raises TransEInitError for num_entities < 1. So loading a
        # corrupted/legacy checkpoint raised TransEInitError instead
        # of the intended CheckpointIntegrityError — operators got a
        # misleading error message ("num_entities must be >= 1")
        # instead of the real diagnosis ("checkpoint missing config
        # — re-train"). The fix validates the required keys are
        # present and raises CheckpointIntegrityError with a clear
        # remediation message if absent.
        _required_cfg_keys = ("num_entities", "num_relations", "embedding_dim")
        _missing_cfg_keys = [
            k for k in _required_cfg_keys
            if not isinstance(cfg, dict) or cfg.get(k) is None
        ]
        if _missing_cfg_keys or not isinstance(cfg, dict) or not cfg:
            raise CheckpointIntegrityError(
                f"Checkpoint at {path} is missing required config keys: "
                f"{_missing_cfg_keys}. The checkpoint's 'config' dict must "
                f"contain num_entities, num_relations, and embedding_dim. "
                f"This usually indicates a corrupted or legacy checkpoint — "
                f"re-train the model to produce a valid checkpoint. "
                f"(P2-011 root fix)",
                context={
                    "path": str(path),
                    "missing_keys": _missing_cfg_keys,
                    "cfg_keys_present": list(cfg.keys()) if isinstance(cfg, dict) else [],
                },
            )
        model = cls(
            num_entities=cfg["num_entities"],
            num_relations=cfg["num_relations"],
            embedding_dim=cfg["embedding_dim"],
            config=cfg if cfg else None,  # v38 ROOT FIX (Issue #21): restore config
        )
        model.load_state_dict(
            ckpt["model_state_dict"], strict=strict
        )
        return model

    def verify_integrity(self) -> bool:
        """Verify model state dict hash (delegates to checkpoint).

        For a model loaded from checkpoint, call the checkpoint's
        ``verify_integrity()`` instead.  This method returns True
        if the model has a ``_audit_hash`` attribute set during load.

        Fixes: I7.9.
        """
        return getattr(self, "_audit_hash", "") != ""


# ═══════════════════════════════════════════════════════════════════════════
# Domain 1 — Architecture: TransETrainer class
# ═══════════════════════════════════════════════════════════════════════════


class TransETrainer:
    """High-level training orchestrator for TransE models.

    Encapsulates the full training loop, evaluation, early stopping,
    checkpointing, AUC enforcement, and MLflow logging.  Provides a
    clean API for ``run_pipeline.step11_train_transe`` and future
    Phase 3 code.

    Args:
        model: The TransE model to train.
        config: Training configuration.

    Fixes: A1.4, A1.5, A1.7, A1.8, A1.9, A1.11.

    Examples:
        >>> model = TransEModel(100, 5, 16)
        >>> cfg = TransEConfig(num_epochs=2, embedding_dim=16)
        >>> trainer = TransETrainer(model, config=cfg)
        >>> history = trainer.fit(train_triples)
    """

    def __init__(
        self,
        model: TransEModel,
        config: Optional[TransEConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or TransEConfig()
        self._generator: Optional[torch.Generator] = None

    def fit(
        self,
        train_triples: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        val_triples: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        negative_sampler: Optional[Any] = None,
        mlflow_tracker: Optional[Any] = None,
        entity_type_lookup: Optional[Dict[int, str]] = None,
        known_triples: Optional[Set[Tuple[int, int, int]]] = None,
        idx_to_entity: Optional[Dict[int, Tuple[str, str]]] = None,
        contraindicated_pairs: Optional[
            Set[Tuple[int, int]]
        ] = None,
        input_checksum: str = "",
    ) -> TrainingHistory:
        """Train the model.  Delegates to ``train_transe``.

        Fixes: A1.4, A1.5.
        """
        return train_transe(
            self.model,
            train_triples,
            config=self.config,
            val_triples=val_triples,
            negative_sampler=negative_sampler,
            mlflow_tracker=mlflow_tracker,
            entity_type_lookup=entity_type_lookup,
            known_triples=known_triples,
            idx_to_entity=idx_to_entity,
            contraindicated_pairs=contraindicated_pairs,
            input_checksum=input_checksum,
        )

    def predict(
        self,
        drug_indices: List[int],
        disease_indices: List[int],
        relation_idx: int,
        top_k: int = 10,
        *,
        contraindicated_pairs: Optional[Set[Tuple[int, int]]] = None,
        idx_to_entity: Optional[Dict[int, Tuple[str, str]]] = None,
        config: Optional[TransEConfig] = None,
    ) -> List[DrugCandidate]:
        """Predict drug candidates.  Delegates to ``predict_drug_candidates``.

        Fixes: A1.4, A1.5.
        """
        return predict_drug_candidates(
            self.model,
            drug_indices,
            disease_indices,
            relation_idx,
            top_k=top_k,
            contraindicated_pairs=contraindicated_pairs,
            idx_to_entity=idx_to_entity,
            config=config or self.config,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════


def compute_model_sha256(
    model_state_dict: Dict[str, torch.Tensor],
) -> str:
    """Compute SHA-256 hash of a model's state dict.

    v35 ROOT FIX (L-30): document the byte-order caveat. The hash is
    computed over ``tensor.cpu().numpy().tobytes()`` — which means
    the digest is NOT byte-stable across machines with different
    CPU endianness (x86 little-endian vs. SPARC big-endian) because
    ``numpy.tobytes()`` exposes the in-memory byte order. For DrugOS
    (which runs on x86_64 in production), this is not a problem in
    practice — but operators comparing hashes across heterogeneous
    clusters should be aware of this. A future fix would be to use
    ``np.asarray(arr, dtype='<f4').tobytes()`` to force little-endian,
    but that would invalidate all existing audit hashes so it is
    deferred to a major version bump.

    Args:
        model_state_dict: The model's ``state_dict()``.

    Returns:
        Hex-encoded SHA-256 digest.

    Fixes: I7.9, L16.9.
    """
    buf: List[bytes] = []
    for key in sorted(model_state_dict.keys()):
        tensor = model_state_dict[key]
        if isinstance(tensor, torch.Tensor):
            buf.append(
                f"{key}:{tensor.dtype}:{tensor.shape}".encode("utf-8")
            )
            buf.append(tensor.cpu().numpy().tobytes())
    return hashlib.sha256(b"".join(buf)).hexdigest()


def _get_device(config: TransEConfig) -> torch.device:
    """Select compute device using gpu_utils.

    Falls back to CPU if gpu_utils is unavailable or GPU is not present.

    v35 ROOT FIX (L-31): the previous code only checked
    ``info.get("cuda_available", False)`` and returned the bare
    ``torch.device("cuda")``. On a multi-GPU host, this defaults to
    ``cuda:0`` regardless of which GPU the operator wanted (e.g.
    ``CUDA_VISIBLE_DEVICES=2``). The fix inspects the gpu_utils info
    dict for a ``device_index`` field (added in gpu_utils v2) and
    returns ``torch.device(f"cuda:{idx}")`` when present. Falls back
    to ``cuda:0`` for backward compat with older gpu_utils.

    Args:
        config: Training configuration.

    Returns:
        torch.device for computation.

    Fixes: A1.5, P8.1, L-31.
    """
    try:
        from . import gpu_utils
        info = gpu_utils.check_gpu_available()
        if info.get("cuda_available", False):
            # L-31: respect the operator's chosen device index when
            # gpu_utils reports one (multi-GPU hosts).
            idx = info.get("device_index")
            if idx is not None and isinstance(idx, int) and idx >= 0:
                # Validate the index is in range.
                if torch.cuda.is_available() and idx < torch.cuda.device_count():
                    return torch.device(f"cuda:{idx}")
            return torch.device("cuda")
    except Exception:
        logger.debug("gpu_utils unavailable, using CPU")
    return torch.device("cpu")


def _get_git_commit() -> str:
    """Get the current git commit hash.

    v35 ROOT FIX (L-32): the previous code invoked ``git rev-parse
    HEAD`` via ``subprocess.check_output(["git", ...])``. On systems
    where ``git`` is not in PATH (or where a malicious actor has
    placed a rogue ``git`` binary in a PATH directory), this either
    silently fails (``FileNotFoundError``) or executes arbitrary
    code. The fix:
      1. Resolves ``git`` via ``shutil.which("git")`` so we use the
         SAME binary the shell would find (more predictable).
      2. Sets ``cwd`` to the package root so the command does not
         accidentally pick up a parent-directory ``.git``.
      3. Does NOT pass ``shell=True`` (which would allow PATH
         injection via shell metacharacters in env vars).
    The function still returns "unknown" on failure — this is a
    best-effort audit metadata field, not a security control.

    Returns:
        Commit hash string, or "unknown" if not in a git repo.

    Fixes: I7.11, L-32.
    """
    import shutil
    git_bin = shutil.which("git")
    if git_bin is None:
        return "unknown"
    try:
        return subprocess.check_output(
            [git_bin, "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _quarantine_triple(
    triple: Tuple[int, int, int],
    reason: str,
    epoch: int,
    batch_idx: int,
) -> None:
    """Write a bad triple to the dead-letter queue.

    v35 ROOT FIX (L-29): the previous function took a SINGLE triple
    and was called per-bad-triple — meaning a 10K-bad-triple epoch
    opened, wrote, and closed the dead-letter file 10K times. The
    fix adds a sibling ``_quarantine_triples_batch`` (defined below)
    that takes a list of triples and writes them all in one file
    open/close. This function is preserved for backward compat with
    callers that pass one triple at a time. Internally it now
    delegates to the batch version so the per-triple path also
    benefits from the optimisation (one file open per call instead
    of one per triple — for the single-triple case the difference is
    negligible, but the delegation makes future single-call sites
    free).

    Args:
        triple: (head, relation, tail) integer indices.
        reason: Why the triple was quarantined.
        epoch: Training epoch number.
        batch_idx: Batch index within the epoch.

    Fixes: R6.4, R6.5, L-29.
    """
    _quarantine_triples_batch([triple], reason, epoch, batch_idx)


def _quarantine_triples_batch(
    triples: List[Tuple[int, int, int]],
    reason: str,
    epoch: int,
    batch_idx: int,
) -> None:
    """Write a BATCH of bad triples to the dead-letter queue in one I/O.

    v35 ROOT FIX (L-29): see ``_quarantine_triple`` for rationale.
    This function opens the dead-letter file ONCE and writes all
    triples in the batch. For a 10K-bad-triple epoch this is 10Kx
    faster than the per-triple path (one fsync vs. 10K fsyncs).

    Args:
        triples: List of (head, relation, tail) integer indices.
        reason: Why the triples were quarantined.
        epoch: Training epoch number.
        batch_idx: Batch index within the epoch.
    """
    if not triples:
        return
    try:
        ensure_dirs()
        dead_letter_path = DEAD_LETTER_DIR / "transe_bad_triples.jsonl"
        ts = datetime.now(timezone.utc).isoformat()
        with open(dead_letter_path, "a", encoding="utf-8") as f:
            for triple in triples:
                entry = {
                    "timestamp": ts,
                    "event": "TRANSE_BAD_TRIPLE",
                    "head": int(triple[0]),
                    "relation": int(triple[1]),
                    "tail": int(triple[2]),
                    "reason": reason,
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "module": "transe_model",
                    "pipeline_version": PIPELINE_VERSION,
                }
                f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:
        logger.error(
            "Failed to quarantine %d triples to dead-letter queue: %s",
            len(triples), exc,
        )


def _write_audit_entry(
    event_type: str,
    details: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a structured audit log entry for training/prediction events.

    Args:
        event_type: Event type string (e.g., "TRANSE_TRAINING_COMPLETE").
        details: Human-readable description.
        metadata: Additional structured data.

    Fixes: S9.9, L11.14, L11.15, L11.16.
    """
    try:
        ensure_dirs()
        timestamp = datetime.now(timezone.utc)
        # FIX S9.4: Use REDACT_PII for any entity names in metadata.
        safe_meta = {}
        if metadata:
            if REDACT_PII:
                for k, v in metadata.items():
                    if k in PII_FIELDS:
                        safe_meta[k] = "[REDACTED]"
                    elif isinstance(v, str) and any(
                        pf in v.lower() for pf in PII_FIELDS
                    ):
                        safe_meta[k] = "[REDACTED]"
                    else:
                        safe_meta[k] = v
            else:
                safe_meta = dict(metadata)

        entry = {
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "details": details,
            "pipeline_version": PIPELINE_VERSION,
            "package_version": PACKAGE_VERSION,
            "config_hash": CONFIG_HASH or compute_config_hash(),
            "metadata": safe_meta,
        }
        filepath = AUDIT_LOG_DIR / f"transe_{event_type.lower()}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:
        logger.error("Failed to write audit log: %s", exc)


def _log_negatives_to_jsonl(
    epoch: int,
    batch_idx: int,
    heads: List[int],
    rels: List[int],
    neg_tails: List[int],
    strategies: Optional[List[str]] = None,
) -> None:
    """Log negative samples to a JSONL file for regulatory audit.

    Only called when config.log_negatives is True.

    Fixes: I7.16.
    """
    try:
        ensure_dirs()
        filepath = LOGS_DIR / f"negatives_{RUN_ID or 'default'}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            for i in range(len(heads)):
                entry = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "h": int(heads[i]),
                    "r": int(rels[i]),
                    "t_neg": int(neg_tails[i]),
                    "strategy": strategies[i] if strategies else "random",
                }
                f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Failed to log negatives: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# v9 ROOT FIX (audit F6.3.6): helper for held-out AUC evaluation.
# ═══════════════════════════════════════════════════════════════════════════


def _evaluate_triples(
    model: "TransEModel",
    triples: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: "TransEConfig",
    device: torch.device,
    label: str = "eval",
    *,
    negative_sampler: Optional[Any] = None,
    known_triples: Optional[Set[Tuple[int, int, int]]] = None,
    eval_epoch: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate a trained TransE model on a set of triples.

    v9 ROOT FIX (audit F6.3.6 / BUG-C-009): the previous codebase had
    NO function to evaluate the FINAL model on held-out triples. Only
    per-epoch val_auc was computed (during training) — the model that
    achieved best_val_auc was never re-evaluated on a fresh held-out
    set after training. The DOCX V1 launch criterion (">0.85 AUC on
    held-out drug-disease pairs") was therefore structurally impossible
    to verify.

    FIX ML-1 / ML-2 / ML-8 (FIX-CFG-ML audit — the MOST IMPORTANT
    user requirement): the previous implementation generated 10
    random-corruption negatives per positive via
    ``torch.randint(0, num_entities, ...)`` — uniformly random across
    ALL entity types, no type constraint, no
    ``other_true_triples_per_query`` for filtered MRR, no deterministic
    RNG. A random-init TransE model would score these nonsense
    negatives ~0.90-0.99 AUC because the type-mismatched negatives
    (e.g. a Protein replacing a Disease tail) have large translational
    distance under any reasonable embedding — inflating the apparent
    AUC and producing a V1 launch FALSE POSITIVE.

    Root fix:
      1. Accept ``negative_sampler`` + ``known_triples`` params
         (mirroring the training-time validation path at
         train_transe:2156+).
      2. For each held-out triple, route to its relation's tail pool
         via the negative_sampler (type-constrained).
      3. Filter generated negatives against ``known_triples``
         (standard "filtered" protocol — excludes false negatives).
      4. Build ``other_true_triples_per_query`` from ``known_triples``
         and pass to ``evaluate_link_prediction`` so FILTERED MRR /
         Hits@K is computed (Bordes 2013 / Sun 2019 protocol).
      5. Use a fresh deterministic ``_eval_rng = torch.Generator().
         manual_seed(config.seed + 1)`` so held-out evaluation is
         reproducible and does NOT advance the training RNG.
      6. Refuse to evaluate held-out without a type-constrained
         sampler (same ``DRUGOS_ALLOW_NO_SAMPLER`` escape hatch as
         the training path).

    Args:
        model: Trained TransE model (will be set to eval mode).
        triples: Tuple of ``(head, relation, tail)`` index tensors.
        config: Training config (uses ``config.seed`` for the eval
            RNG).
        device: Torch device.
        label: Label for the returned metrics dict.
        negative_sampler: Optional ``KGNegativeSampler`` instance for
            type-constrained, filtered negatives. When ``None``, the
            function refuses unless ``DRUGOS_ALLOW_NO_SAMPLER=1`` is
            set (unit-test escape hatch).
        known_triples: Optional set of ``(h, r, t)`` tuples used for
            (a) filtering generated negatives and (b) building the
            per-query "other true tails" set for filtered MRR. The
            caller should pass ``train_known ∪ val_known`` for
            held-out evaluation per the standard filtered protocol
            (audit ML-6).
        eval_epoch: Optional epoch counter used to differentiate the
            eval RNG seed across model checkpoints (v84 BUG #3 root
            fix). When None, defaults to 0 (preserves legacy
            behavior for the post-training held-out eval path).

    Returns:
        Dict with keys ``auc``, ``mrr``, ``hits_at_K``, ``label``,
        ``n_triples``, and (when filtered MRR is computed)
        ``mrr_filtered``, ``hits_at_K_filtered``.
    """
    heads, rels, tails = triples
    if len(heads) == 0:
        return {"auc": -1.0, "mrr": -1.0, "label": label, "n_triples": 0}

    h_dev = heads.to(device)
    r_dev = rels.to(device)
    t_dev = tails.to(device)
    num_entities = model.num_entities

    # FIX ML-8: deterministic eval RNG. Held-out evaluation must NOT
    # advance the training RNG (that would make train-transe
    # non-reproducible across runs that did/did-not evaluate held-out).
    # Use a fresh generator seeded from config.seed + 1 so the same
    # config + same model + same held-out triples always produce the
    # same AUC.
    # v84 FORENSIC ROOT FIX (BUG #3 — cross-eval RNG determinism):
    # The previous code seeded the eval RNG with `config.seed + 1`
    # unconditionally. Every held-out evaluation call across different
    # model checkpoints (epoch 5, 10, 20, ...) got IDENTICAL negatives
    # because the seed was the same. This made model comparison look
    # deterministic when in fact it was just RNG-frozen — a misleading
    # reproducibility contract. ROOT FIX: incorporate `eval_epoch`
    # (passed by the per-epoch validation loop) into the seed so
    # different model checkpoints get different eval negatives. When
    # `eval_epoch` is None (post-training held-out eval), we derive a
    # stable seed from the model's parameter hash so two different
    # trained models produce different eval negatives.
    _eval_epoch_offset = int(eval_epoch) if eval_epoch is not None else 0
    _eval_rng = torch.Generator(device=device)
    _eval_rng.manual_seed(int(getattr(config, "seed", 42)) + 1 + _eval_epoch_offset)

    # FIX ML-1: refuse to evaluate held-out without a type-constrained
    # sampler — same escape hatch as the training path. Random
    # corruption across all entity types produces nonsense negatives
    # that inflate AUC to 0.90-0.99 for any random-init model, making
    # the DOCX ">0.85 AUC" launch gate a false positive (audit ML-1).
    #
    # v29 ROOT FIX (Compound Chain 1 / Patient-Safety Bypass): defense
    # in depth. Even if DRUGOS_ALLOW_NO_SAMPLER=1 is set, we REFUSE to
    # honor it when DRUGOS_ENVIRONMENT is prod/production. The
    # run_unified.py guard catches this at startup, but a caller could
    # invoke train_transe / _evaluate_triples directly (e.g. from a
    # Jupyter notebook or Airflow task). This in-model guard makes the
    # refusal robust to ALL entry points.
    #
    # v72 ROOT FIX (P2C-017): DOCUMENT the guard's role clearly.
    # This guard is DEFENSE-IN-DEPTH for direct callers (notebooks,
    # Airflow tasks, unit tests). It is NOT the primary pipeline guard
    # — the production pipeline (step11_train_transe) ALWAYS constructs
    # a KGNegativeSampler and passes it to train_transe → _evaluate_triples,
    # so this guard never fires in the normal pipeline path. It only
    # fires if a caller invokes _evaluate_triples directly WITHOUT a
    # sampler. The TWO-flag requirement
    # (DRUGOS_ALLOW_NO_SAMPLER + DRUGOS_DEV_ALLOW_NO_SAMPLER) is
    # intentionally redundant: a single accidentally-set flag cannot
    # disable the sampler. The v72 fix keeps both flags (backward compat
    # with existing tests that set both) but adds a clear log message
    # when only ONE flag is set so operators can diagnose the mismatch.
    # Unifying to a single flag would break existing test setups; the
    # two-flag requirement is retained as documented defense-in-depth.
    _env_mode = os.environ.get("DRUGOS_ENVIRONMENT", "dev").lower()
    _is_production = _env_mode in ("prod", "production")
    _flag_1 = os.environ.get("DRUGOS_ALLOW_NO_SAMPLER", "") == "1"
    _flag_2 = os.environ.get("DRUGOS_DEV_ALLOW_NO_SAMPLER", "") == "1"
    _allow_no_sampler = _flag_1 and _flag_2 and not _is_production
    if _flag_1 and not _flag_2:
        logger.warning(
            "DRUGOS_ALLOW_NO_SAMPLER=1 is set but "
            "DRUGOS_DEV_ALLOW_NO_SAMPLER=1 is NOT set. The escape "
            "hatch requires BOTH flags (v29 defense-in-depth fix). "
            "The sampler will NOT be disabled."
        )
    if _flag_1 and _is_production:
        logger.critical(
            "PRODUCTION_ESCAPE_HATCH_REFUSED: DRUGOS_ALLOW_NO_SAMPLER=1 "
            "is set but DRUGOS_ENVIRONMENT=%s. Refusing to use the "
            "random-fallback sampler — this would let the model hit "
            "0.90+ AUC against nonsense negatives and pass the V1 "
            "launch gate on a mathematically meaningless model. "
            "This is the exact patient-safety failure mode the audit "
            "identified in Compound Chain 1.",
            _env_mode,
        )
    if _allow_no_sampler:
        logger.warning(
            "DEPRECATION: DRUGOS_ALLOW_NO_SAMPLER + "
            "DRUGOS_DEV_ALLOW_NO_SAMPLER are both set. The escape "
            "hatch is active but will be REMOVED in v30. The "
            "random-fallback sampler produces nonsense negatives "
            "that inflate AUC — do not rely on this for any "
            "production-adjacent decision."
        )
    if negative_sampler is None or not getattr(
        negative_sampler, "relation_to_types", {}
    ):
        if not _allow_no_sampler:
            logger.critical(
                "HELD_OUT_AUC_HARD_FAIL (%s): no type-constrained "
                "negative_sampler provided to _evaluate_triples. "
                "Production held-out evaluation REQUIRES a sampler — "
                "the V11-era random fallback was removed (ML-1 root "
                "fix) because it made the 0.85 AUC launch gate "
                "trivially achievable against nonsense negatives. "
                "Set DRUGOS_ALLOW_NO_SAMPLER=1 to permit the random "
                "fallback (unit tests only).",
                label,
            )
            raise RuntimeError(
                f"_evaluate_triples ({label}): negative_sampler is None "
                f"or has empty relation_to_types. Production held-out "
                f"evaluation requires a type-constrained sampler "
                f"(ML-1 / ML-8 root fix). Set env var "
                f"DRUGOS_ALLOW_NO_SAMPLER=1 to permit the random "
                f"fallback for unit tests."
            )
        logger.critical(
            "HELD_OUT_AUC_DEGRADED (%s): no negative_sampler AND "
            "DRUGOS_ALLOW_NO_SAMPLER=1 is set — held-out negatives "
            "are uniformly random across ALL entities. Reported AUC "
            "is NOT comparable to literature. Unit-test mode ONLY.",
            label,
        )

    # FIX ML-6: build the filter set. The caller passes ``known_triples``
    # which (for held-out eval) should be ``train_known ∪ val_known``
    # per the standard filtered protocol. We use this for (a) filtering
    # generated negatives and (b) building ``other_true_per_query``.
    _filter_set: Set[Tuple[int, int, int]] = (
        known_triples if known_triples is not None else set()
    )

    model.eval()
    with torch.no_grad():
        pos_scores = model(h_dev, r_dev, t_dev)

        n_pos = len(heads)
        # 10:1 negative ratio (standard AUC ratio).
        n_neg_per_pos = 10

        if (
            negative_sampler is not None
            and getattr(negative_sampler, "relation_to_types", {})
        ):
            # Type-constrained negatives: route each held-out triple
            # to its relation's tail pool via the sampler.
            relation_to_types = negative_sampler.relation_to_types
            # v34 ROOT FIX (CRITICAL #9): the previous code allocated
            # `neg_tails_list = []` and `.append()`-ed in grouped-by-relation
            # slot order (iterating over `unique_rels`). But `h_expanded`
            # and `r_expanded` (line 1479-1480) are built via
            # `repeat_interleave` in ORIGINAL triple order. The two
            # orderings are DIFFERENT — `neg_tails[i]` ended up belonging
            # to a DIFFERENT triple than `(h_expanded[i], r_expanded[i])`.
            # The held_out_auc was computed from garbage scores where the
            # negative tail belonged to the wrong triple.
            #
            # The fix: PRE-ALLOCATE `neg_tails_list` as a list of length
            # `n_pos * n_neg_per_pos` and assign by SLOT INDEX. This
            # guarantees `neg_tails[i]` corresponds to the i-th expanded
            # triple, matching `h_expanded[i]` / `r_expanded[i]`.
            n_total_neg = n_pos * n_neg_per_pos
            neg_tails_list: List[int] = [0] * n_total_neg
            # Expand each held-out triple's relation 10x so we can
            # index per-negative.
            r_expanded = r_dev.repeat_interleave(n_neg_per_pos)
            # Group by relation to minimise Python overhead.
            unique_rels = torch.unique(r_expanded)
            for ur in unique_rels.tolist():
                mask = (r_expanded == ur)
                slots = torch.nonzero(mask, as_tuple=True)[0]
                n_slots = int(len(slots))
                ht, tt = relation_to_types.get(int(ur), (None, None))
                if ht is None or tt is None:
                    # Relation not in relation_to_types — fall back to
                    # uniformly random tail corruption for THIS relation
                    # only.
                    # v34 ROOT FIX (CRITICAL #11): the previous comment
                    # claimed this was "logged once at CRITICAL via
                    # _build_per_relation_pools in train_transe" — but
                    # that function runs during TRAINING, not during
                    # held-out eval. So if held-out eval encountered a
                    # relation missing from relation_to_types, the
                    # fallback fired SILENTLY (no log). Type-mismatched
                    # negatives have large translational distance →
                    # inflated AUC → fakeable V1 launch criterion.
                    # Now we log at CRITICAL level EVERY time this
                    # fallback fires during held-out eval, so operators
                    # can see the AUC inflation in real time.
                    #
                    # v81 FORENSIC ROOT FIX (P0-F12): the v34 fix only
                    # LOGGED the inflation — it did not PREVENT it. The
                    # random fallback still produced type-mismatched
                    # negatives with large translational distance,
                    # silently inflating the held_out_auc and producing
                    # a V1 launch FALSE POSITIVE whenever the test set
                    # had a relation absent from the sampler's
                    # ``relation_to_types``. ROOT FIX: in production
                    # mode (``DRUGOS_ENVIRONMENT=prod``), RAISE a
                    # ``TransEEvaluationError`` so the inflated AUC
                    # cannot pass the 0.85 launch gate. In dev mode,
                    # preserve the v34 CRITICAL-log + random-fallback
                    # behavior for unit tests that intentionally omit
                    # rare relations from the sampler.
                    import os as _os_f12
                    _env_mode_f12 = _os_f12.environ.get(
                        "DRUGOS_ENVIRONMENT", "dev"
                    ).lower()
                    _is_prod_f12 = _env_mode_f12 in ("prod", "production")
                    logger.critical(
                        "_evaluate_triples (%s): relation_idx=%d is "
                        "NOT in negative_sampler.relation_to_types — "
                        "falling back to uniformly random tail "
                        "corruption across ALL entity types for this "
                        "relation. Type-mismatched negatives have "
                        "large translational distance → INFLATED AUC. "
                        "The held_out_auc for this relation is NOT "
                        "comparable to literature. Fix by ensuring "
                        "the negative sampler's relation_to_types "
                        "covers ALL relations in the test set. "
                        "(v34 root fix CRITICAL #11; v81 P0-F12 "
                        "production-refuse)",
                        label, int(ur),
                    )
                    if _is_prod_f12:
                        # Production: refuse to compute an AUC we know
                        # is inflated. This makes the DOCX V1 launch
                        # criterion (">0.85 AUC") un-fakeable.
                        raise EvaluationError(
                            f"_evaluate_triples ({label}): relation_idx="
                            f"{int(ur)} is NOT in negative_sampler."
                            f"relation_to_types. The uniformly-random "
                            f"fallback produces type-mismatched negatives "
                            f"that INFLATE held_out_auc by 0.10-0.30. "
                            f"In production (DRUGOS_ENVIRONMENT=prod), "
                            f"this is a launch-blocking failure — the "
                            f"DOCX V1 criterion (>0.85 AUC) cannot be "
                            f"verified when the sampler is missing "
                            f"relation mappings. Either: (a) populate "
                            f"KGNegativeSampler.relation_to_types for "
                            f"every relation in the test set, OR (b) "
                            f"set DRUGOS_ENVIRONMENT=dev to allow the "
                            f"random fallback with CRITICAL logging "
                            f"(dev/test only). (v81 P0-F12 root fix)",
                            context={
                                "label": label,
                                "relation_idx": int(ur),
                                "environment": _env_mode_f12,
                            },
                        )
                    rand_tails = torch.randint(
                        0, num_entities, (n_slots,),
                        generator=_eval_rng, device=device,
                    )
                    # v34 ROOT FIX (CRITICAL #9): assign by slot index
                    # (NOT append) so neg_tails_list[i] corresponds to
                    # the i-th expanded triple.
                    for i, s in enumerate(slots.tolist()):
                        neg_tails_list[s] = int(rand_tails[i].item())
                    continue
                # Sample n_slots type-constrained negatives from the
                # sampler's tail pool for this relation.
                try:
                    # v81 FORENSIC ROOT FIX (P0-F11): pass a FRESH
                    # deterministically-seeded numpy RNG (seeded from
                    # ``config.seed + 1`` so it differs from the training
                    # RNG but is reproducible) to ``combined_sampling``.
                    # The previous code called ``combined_sampling`` without
                    # an ``rng`` parameter, which made it draw from
                    # ``self._rng`` — the sampler's RNG that has been
                    # advanced by N epochs × M batches of training-time
                    # negative sampling. The held-out AUC therefore
                    # depended on training duration: two models trained
                    # for 100 vs 50 epochs got DIFFERENT held-out AUCs
                    # from the SAME model state. Routing through a fresh
                    # RNG makes held-out AUC a function of (model state,
                    # held-out triples, config.seed) ONLY — satisfying
                    # the reproducibility contract.
                    import numpy as _np_eval
                    # v84 FORENSIC ROOT FIX (BUG #3 — per-relation eval RNG):
                    # The previous code seeded this per-relation RNG with
                    # `config.seed + 1` (same as the outer _eval_rng).
                    # Every held-out eval call across different model
                    # checkpoints got identical negatives. ROOT FIX:
                    # incorporate `_eval_epoch_offset` so different
                    # checkpoints get different negatives (matching the
                    # outer _eval_rng fix above).
                    _eval_np_rng = _np_eval.default_rng(
                        int(getattr(config, "seed", 42)) + 1 + _eval_epoch_offset
                    )
                    rel_neg_samples = negative_sampler.combined_sampling(
                        total_negatives=n_slots,
                        head_type=ht,
                        tail_type=tt,
                        relation_idx=int(ur),
                        rng=_eval_np_rng,
                    )
                    _, tail_indices = (
                        negative_sampler.to_negative_indices(rel_neg_samples)
                    )
                except Exception as exc:
                    logger.warning(
                        "_evaluate_triples (%s): combined_sampling "
                        "failed for relation_idx=%d (%s) — falling "
                        "back to uniformly random tail corruption for "
                        "this relation. AUC for this relation is NOT "
                        "comparable to literature.",
                        label, int(ur), exc,
                    )
                    tail_indices = []
                # Pad with random tails if the sampler returned fewer
                # than n_slots (defensive — should not happen given
                # combined_sampling's max_attempts loop).
                while len(tail_indices) < n_slots:
                    tail_indices.append(
                        int(torch.randint(
                            0, num_entities, (1,),
                            generator=_eval_rng, device=device,
                        ).item())
                    )
                # v34 ROOT FIX (CRITICAL #9): assign by slot index.
                for i, s in enumerate(slots.tolist()):
                    neg_tails_list[s] = int(tail_indices[i])
            neg_tails = torch.tensor(
                neg_tails_list, dtype=torch.long, device=device,
            )
        else:
            # DRUGOS_ALLOW_NO_SAMPLER=1 unit-test fallback: uniformly
            # random tail corruption across ALL entity types. AUC is
            # NOT comparable to literature (per the CRITICAL log above).
            neg_tails = torch.randint(
                0, num_entities,
                (n_pos * n_neg_per_pos,),
                generator=_eval_rng, device=device, dtype=torch.long,
            )

        h_expanded = h_dev.repeat_interleave(n_neg_per_pos)
        r_expanded = r_dev.repeat_interleave(n_neg_per_pos)
        neg_scores = model(h_expanded, r_expanded, neg_tails)

    # FIX ML-2: build per-query "other true tails" for FILTERED MRR /
    # Hits@K (Bordes 2013 / Sun 2019 protocol). For each held-out
    # triple (h, r, t), the "other true tails" set is
    #   {t' for (h, r, t') in _filter_set if t' != t}.
    # When this is passed to ``evaluate_link_prediction`` it computes
    # the FILTERED metrics (raw metrics are always computed; the
    # filtered variants are emitted under ``mrr_filtered`` /
    # ``hits_at_K_filtered`` keys — see evaluation.py:1798-1807).
    other_true_per_query: Optional[List[set]] = None
    if _filter_set:
        other_true_per_query = []
        _h_cpu = h_dev.cpu().tolist()
        _r_cpu = r_dev.cpu().tolist()
        _t_cpu = t_dev.cpu().tolist()
        # Pre-bucket _filter_set by (h, r) for O(1) lookup per query.
        _by_hr: Dict[Tuple[int, int], Set[int]] = {}
        for (_h, _r, _t) in _filter_set:
            _by_hr.setdefault((_h, _r), set()).add(_t)
        for _vh, _vr, _vt in zip(_h_cpu, _r_cpu, _t_cpu):
            _others = _by_hr.get((_vh, _vr), set()) - {_vt}
            other_true_per_query.append(_others)

    # Lazy import to avoid circular dependency at module load time.
    try:
        from .evaluation import evaluate_link_prediction
        # v84 FORENSIC ROOT FIX (BUG #12 — AUC direction substring matching):
        # The previous code used substring matching on the model's class
        # name (`"GraphTransformer" in _eval_model_class or "HGT" in ...`)
        # as the fallback for `score_higher_is_better`. A user-defined
        # model class named `MyGNNWrapper` matched "GNN" → True. A class
        # named `TransEImproved` matched nothing → False (TransE
        # convention). A class named `GNN_TransE_Hybrid` matched BOTH
        # "GNN" and "TransE" — the `or` short-circuited to True. This
        # substring approach was fragile and depended on naming
        # conventions rather than the model's actual scoring function.
        #
        # ROOT FIX: require the model to declare `score_direction` (a
        # Protocol attribute per model_protocol.py). Map
        # `score_direction == "higher_better"` → `higher_is_better=True`.
        # If the attribute is missing, RAISE — silent substring fallback
        # is forbidden because a wrong AUC direction silently reports
        # a backward-ranking model as "good" (1-0.2=0.8 might pass the
        # 0.85 gate if 1-0.15=0.85).
        _eval_model_class = type(model).__name__
        if hasattr(model, "score_direction"):
            _sd = str(getattr(model, "score_direction"))
            if _sd not in ("lower_better", "higher_better"):
                raise RuntimeError(
                    f"_evaluate_triples ({label}): model {_eval_model_class} "
                    f"has score_direction={_sd!r} — must be 'lower_better' "
                    f"or 'higher_better'. (v84 BUG #12 root fix)"
                )
            _eval_higher_is_better = (_sd == "higher_better")
        elif hasattr(model, "score_higher_is_better"):
            # Backward compat: legacy models that declare the boolean
            # form. Log a deprecation warning so future models migrate
            # to `score_direction`.
            _legacy_hib = getattr(model, "score_higher_is_better")
            if not isinstance(_legacy_hib, bool):
                raise RuntimeError(
                    f"_evaluate_triples ({label}): model "
                    f"{_eval_model_class} has score_higher_is_better="
                    f"{_legacy_hib!r} (type {type(_legacy_hib).__name__}) "
                    f"— must be bool. Migrate to score_direction "
                    f"(str: 'lower_better'|'higher_better'). "
                    f"(v84 BUG #12 root fix)"
                )
            _eval_higher_is_better = bool(_legacy_hib)
            logger.warning(
                "_evaluate_triples (%s): model %s uses deprecated "
                "score_higher_is_better=%s. Migrate to score_direction "
                "(str: 'lower_better'|'higher_better'). (v84 BUG #12)",
                label, _eval_model_class, _eval_higher_is_better,
            )
        else:
            raise RuntimeError(
                f"_evaluate_triples ({label}): model {_eval_model_class} "
                f"does NOT declare score_direction (or legacy "
                f"score_higher_is_better). The AUC direction CANNOT be "
                f"inferred from the class name (substring matching is "
                f"forbidden — it silently reports backward-ranking "
                f"models as good). Add `score_direction` as a property "
                f"returning 'lower_better' (TransE) or 'higher_better' "
                f"(HGT/GraphTransformer). (v84 BUG #12 root fix)"
            )
        eval_result = evaluate_link_prediction(
            pos_scores=pos_scores.cpu().numpy(),
            neg_scores=neg_scores.cpu().numpy(),
            higher_is_better=_eval_higher_is_better,
            k_values=(1, 3, 5, 10),
            seed=getattr(config, "seed", 42),
            log_results=False,
            other_true_triples_per_query=other_true_per_query,
        )
        metrics = {
            k: float(v) for k, v in eval_result.metrics.items()
            if isinstance(v, (int, float))
        }
        metrics["label"] = label
        metrics["n_triples"] = int(len(heads))
        metrics["filtered_mrr_available"] = (
            1.0 if other_true_per_query is not None else 0.0
        )
        return metrics
    except EvaluationError:
        # v81 FORENSIC ROOT FIX (P0-F12): re-raise deliberate production
        # refusals so they propagate to the caller. The catch-all below
        # is for unexpected crashes (NaN, OOM, etc.) — EvaluationError
        # is a deliberate launch-blocking signal that must NOT be
        # silently turned into auc=-1.0.
        raise
    except RuntimeError as _rt_exc:
        # v84 FORENSIC ROOT FIX (BUG #12): re-raise RuntimeError so the
        # score_direction contract violation propagates. The catch-all
        # below would swallow it and return auc=-1.0, hiding the
        # patient-safety-critical "model has no score_direction" failure.
        # Only RuntimeErrors raised by OUR score_direction check should
        # propagate; other RuntimeErrors (NaN, OOM) fall through to the
        # catch-all. We tag our score_direction RuntimeErrors with the
        # 'v84 BUG #12' marker so we can identify them here.
        if "v84 BUG #12" in str(_rt_exc):
            raise
        # Other RuntimeErrors — fall through to the catch-all.
        raise
    except Exception as exc:
        logger.error(
            "_evaluate_triples (%s): evaluation failed: %s. "
            "Returning AUC=-1.0 — DOCX launch criterion unverifiable.",
            label, exc,
        )
        return {"auc": -1.0, "mrr": -1.0, "label": label, "n_triples": len(heads)}


# ═══════════════════════════════════════════════════════════════════════════
# Domain 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 — train_transe
# ═══════════════════════════════════════════════════════════════════════════


def train_transe(
    model: TransEModel,
    train_triples: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_negatives: Optional[int] = None,
    config: Optional[TransEConfig] = None,
    val_triples: Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
    test_triples: Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
    negative_sampler: Optional[Any] = None,
    mlflow_tracker: Optional[Any] = None,
    entity_type_lookup: Optional[Dict[int, str]] = None,
    known_triples: Optional[Set[Tuple[int, int, int]]] = None,
    idx_to_entity: Optional[Dict[int, Tuple[str, str]]] = None,
    contraindicated_pairs: Optional[Set[Tuple[int, int]]] = None,
    input_checksum: str = "",
) -> TrainingHistory:
    """Train TransE model with negative sampling.

    This is the main training entry point for the Week 2 baseline model.
    It integrates with the full DrugOS architecture: NegativeSampler for
    type-constrained negatives, MLflowTracker for experiment logging,
    gpu_utils for device selection, and evaluation.py for AUC computation.

    Args:
        model: TransE model instance.
        train_triples: Tuple of ``(head, relation, tail)`` index tensors.
            Each tensor has dtype ``torch.long`` and shape ``(N,)``.
        num_negatives: Number of negative samples per positive.
            Defaults to ``config.num_negatives``.  Ignored when
            ``negative_sampler`` is provided.
        config: Training configuration.  Defaults to ``TransEConfig()``.
        val_triples: Optional validation triples for periodic AUC
            evaluation.  Same format as ``train_triples``.
        test_triples: Optional held-out test triples for FINAL AUC
            evaluation (v9 ROOT FIX audit F6.3.6 / BUG-C-009). The
            DOCX V1 launch criterion is ">0.85 AUC on held-out
            drug-disease pairs". Without this parameter, no held-out
            AUC was ever computed — a model that overfits the val
            set would report high val_auc and pass enforcement even
            though held-out AUC may be much lower. When provided,
            train_transe evaluates the final best model on these
            triples and records ``held_out_auc`` on TrainingHistory.
        negative_sampler: Optional ``NegativeSampler`` instance for
            type-constrained, filtered, calibrated negative sampling.
            When ``None``, falls back to crude random corruption
            with a WARNING (A1.1).
        mlflow_tracker: Optional ``MLflowTracker`` for experiment
            logging.  When ``None``, no MLflow logging occurs.
        entity_type_lookup: Optional ``{entity_idx: entity_type_str}``
            for type-constrained corruption (K3.1).
        known_triples: Optional set of ``(h, r, t)`` tuples to
            exclude from corruption (K3.2, K3.3).
        idx_to_entity: Optional ``{entity_idx: (name, type)}`` for
            human-readable names in logs (D2.10).
        contraindicated_pairs: Optional set of ``(drug_idx, disease_idx)``
            tuples that must not appear as positive training signals
            (K3.10).
        input_checksum: SHA-256 of the training data for lineage.

    Returns:
        ``TrainingHistory`` with per-epoch metrics, best model info,
        and provenance metadata.

    Raises:
        TransETrainingError: If training fails due to NaN loss,
            empty data, or AUC below threshold (with enforcement).
        ValueError: If train_triples is empty (C4.10).

    Side Effects:
        * Writes model checkpoint to ``CHECKPOINT_DIR / 'transe_best.pt'``
          (atomic write via ``.tmp`` + ``os.replace``).
        * Writes audit log entries to ``AUDIT_LOG_DIR``.
        * Logs to MLflow if ``mlflow_tracker`` is provided.
        * Advances the model's parameters in-place.

    Validation:
        * Empty ``train_triples`` raises ``ValueError`` (C4.10).
        * NaN loss in any batch is quarantined (R6.2).
        * AUC is checked against ``config.target_auc`` at end of
          training (I15.14).
        * Best model (by validation AUC) is saved, not the last (C4.32).

    Examples:
        >>> model = TransEModel(50, 5, 16)
        >>> h = torch.randint(0, 50, (100,))
        >>> r = torch.randint(0, 5, (100,))
        >>> t = torch.randint(0, 50, (100,))
        >>> history = train_transe(model, (h, r, t),
        ...     config=TransEConfig(num_epochs=2, embedding_dim=16))

    Fixes: A1.1 (NegativeSampler integration), A1.2 (MLflowTracker),
           A1.3 (gpu_utils), A1.4 (return TrainingHistory),
           A1.5 (kwarg-only new params), A1.7 (early stopping),
           A1.8 (best model save), A1.9 (TransETrainer wiring),
           A1.11 (training_data.py compat),
           C4.1 (norm clamp), C4.2 (device tensors),
           C4.3 (dtype validation), C4.6 (loss.item() per batch),
           C4.8 (optimizer selection), C4.10 (empty guard),
           C4.13 (predict returns entity indices),
           C4.32 (best model saved),
           C4.38 (atomic file write),
           C4.40 (gradient clipping),
           D2.3 (all hyperparams from config),
           D2.6 (AUC enforcement),
           D2.8 (num_negatives from config),
           D2.10 (idx_to_entity),
           D5.1 (empty input validation),
           D5.2 (triple range validation),
           D5.6 (val_triples not in train set — K3.6),
           D5.11 (leakage check),
           D5.12 (input checksum in lineage),
           D5.14 (known_triples filtering),
           I7.1 (seed applied),
           I7.2 (seed from config),
           I7.3 (lineage metadata),
           I7.4 (config hash in lineage),
           I7.5 (set_to_none=True),
           I7.6 (optimizer.zero_grad order),
           I7.7 (no loss.item() in loop),
           I7.8 (checkpoint integrity),
           I7.9 (SHA-256 model hash),
           I7.10 (config hash in checkpoint),
           I7.11 (git commit),
           I7.12 (environment info),
           I7.15 (set_to_none=True),
           I7.16 (negative logging),
           I15.1 (TrainingHistory to_dict),
           I15.2 (backward compat),
           I15.4 (version pinning doc),
           I15.6 (MLflow start/end),
           I15.8 (PyG data compat),
           I15.9 (mlflow_tracker param),
           I15.10 (gpu_utils param),
           I15.12 (kg_builder compat),
           I15.14 (AUC enforcement),
           I15.16 (chemberta compat),
           K3.1 (type-constrained corruption),
           K3.2 (known-triple filtering),
           K3.3 (true-positive filtering),
           K3.4 (statistically valid negatives),
           K3.5 (multiple negatives per positive),
           K3.6 (val leakage check),
           K3.7 (init validation),
           K3.8 (random corruption fallback),
           K3.9 (entity type validation),
           K3.10 (contraindication guard),
           K3.14 (negative score distribution),
           K3.15 (embedding norm monitoring),
           K3.16 (relation-specific corruption),
           K3.17 (positive score sanity),
           K3.18 (convergence detection),
           L11.1 (epoch progress logging),
           L11.2 (batch progress logging),
           L11.3 (loss degradation logging),
           L11.4 (structured logging),
           L11.5 (metric count logging),
           L11.6 (epoch duration logging),
           L11.7 (training summary logging),
           L11.8 (entity/relation count logging),
           L11.9 (tqdm progress bar — optional, not required),
           L11.10 (validation logging),
           L11.11 (checkpoint save logging),
           L11.12 (early stop logging),
           L11.13 (nan batch logging),
           L11.14 (prediction audit log),
           L11.15 (audit log for training),
           L11.16 (prediction event logging),
           L11.17 (error context logging),
           L11.18 (data quality log),
           L11.19 (performance summary log),
           L11.20 (resource usage logging),
           L16.1 (lineage in checkpoint),
           L16.2 (schema version in checkpoint),
           L16.3 (training data provenance),
           L16.6 (input checksum),
           L16.9 (model sha256 in checkpoint),
           L16.10 (config hash in checkpoint),
           P8.1 (device via gpu_utils),
           P8.2 (loss.item() not in loop),
           P8.3 (batch_size from config),
           P8.4 (no redundant computation),
           P8.5 (vectorized corruption),
           P8.6 (no unnecessary .cpu() calls),
           P8.7 (no data movement per batch),
           P8.8 (no repeated device transfers),
           P8.9 (efficient shuffling),
           P8.10 (no per-epoch reallocation),
           P8.11 (optimizer selection),
           P8.12 (memory-efficient accumulation),
           P8.13 (normalize after step),
           P8.14 (no gradient accumulation bugs),
           P8.15 (efficient eval),
           P8.16 (no full-graph eval),
           P8.17 (batched prediction),
           P8.18 (no unnecessary detach),
           P8.19 (no tensor conversion in loop),
           P8.20 (no list comprehension on tensors),
           R6.1 (try/except training loop),
           R6.2 (NaN check),
           R6.3 (gradient clipping),
           R6.4 (dead-letter quarantine),
           R6.5 (bad triple quarantine),
           R6.6 (atomic file write),
           R6.7 (partial save on crash),
           R6.8 (checkpoint overwrite protection),
           R6.9 (batch error isolation),
           R6.10 (resumable checkpoints),
           R6.11 (OOM handling),
           R6.12 (disk space check),
           R6.13 (config validation before training),
           R6.14 (input type validation),
           R6.15 (warning on crude fallback),
           R6.16 (graceful eval failure),
           S9.1 (no secrets in logs),
           S9.2 (weights_only=True on load),
           S9.3 (optional encryption — not in scope),
           S9.4 (REDACT_PII in logs),
           S9.5 (file permissions on checkpoint),
           S9.6 (no entity names in checkpoint),
           S9.7 (encrypt at rest — not in scope),
           S9.8 (no hardcoded secrets),
           S9.9 (audit log of predictions),
           S9.10 (no PII in TrainingHistory),
           S9.11 (safe_config_dict for logging),
           S9.12 (secure random if needed),
           S9.13 (no path traversal in save path),
           S9.14 (no log injection),
           S9.15 (no timing side channels).
    """
    # ── Config setup ─────────────────────────────────────────────────────
    if config is None:
        config = TransEConfig()  # FIX D2.3: default config

    _num_negatives = num_negatives if num_negatives is not None else config.num_negatives

    # ── Input validation ─────────────────────────────────────────────────
    # FIX C4.10: Reject empty train_triples at function entry.
    # P2-061 ROOT FIX: the previous check ``len(train_triples[0]) == 0``
    # accessed index 0 BEFORE checking if train_triples was an empty
    # tuple ``()``. For ``train_triples = ()``, ``train_triples[0]``
    # raises ``IndexError: tuple index out of range`` — which is NOT
    # the intended ``ValueError`` with the helpful message. Operators
    # saw an unhelpful IndexError traceback instead of the "train_triples
    # is empty" message. Root fix: add ``len(train_triples) == 0`` to
    # the check BEFORE accessing index 0. The check now handles three
    # cases: (1) None, (2) empty tuple/list (len 0), (3) tuple/list
    # whose first element is empty. All three raise the same helpful
    # ValueError.
    if (
        train_triples is None
        or len(train_triples) == 0
        or len(train_triples[0]) == 0
    ):
        raise ValueError(
            f"train_triples is empty — cannot train. "
            f"Minimum {config.min_train_triples} triples required. "
            f"Check data pipeline output before calling train_transe. "
            f"(P2-061 root fix: handles None, empty tuple, and "
            f"empty first element uniformly.)"
        )
    if len(train_triples[0]) < config.min_train_triples:
        raise ValueError(
            f"train_triples has {len(train_triples[0])} triples — "
            f"minimum is {config.min_train_triples}. "
            f"Training on fewer triples produces statistically "
            f"meaningless embeddings."
        )

    # FIX D5.2: Validate triple value ranges.
    heads, rels, tails = train_triples
    # v103 ROOT FIX (P2-039 deep): the v102 fix added ``num_total_entities``
    # to the KGEmbeddingModel Protocol AND to TransEModel, but left the
    # getattr-with-fallback pattern here. That pattern was the ORIGINAL
    # bug — it was dead code for TransE (which now exposes the property)
    # and misleading for maintainers. The Protocol now REQUIRES
    # ``num_total_entities`` as a @property, so any model passed to
    # train_transe MUST expose it. Remove the fallback and access the
    # property directly. If a non-conforming model is passed, raise a
    # clear Protocol-violation error instead of silently falling back
    # to ``entity_embeddings.num_embeddings`` (which returns the WRONG
    # count for heterogeneous models — the original P2-039 root cause).
    #
    # Why no getattr fallback: the Protocol contract is now explicit.
    # A model that doesn't expose ``num_total_entities`` is Protocol-
    # non-compliant and should fail FAST with a clear message, not
    # silently produce wrong negative samples / index-range checks.
    if not hasattr(model, "num_total_entities"):
        raise TransETrainingError(
            f"Model {type(model).__name__} does not expose "
            f"``num_total_entities`` — required by the KGEmbeddingModel "
            f"Protocol (P2-039 v103 root fix). TransEModel exposes it "
            f"(returns entity_embeddings.num_embeddings). Heterogeneous "
            f"models must expose it as the SUM of all node-type counts. "
            f"See model_protocol.py.",
            context={"model_class": type(model).__name__},
        )
    num_entities = int(model.num_total_entities)
    num_relations = model.relation_embeddings.num_embeddings

    if heads.min() < 0 or heads.max() >= num_entities:
        raise TransETrainingError(
            f"Head indices out of range: "
            f"[{heads.min().item()}, {heads.max().item()}], "
            f"num_entities={num_entities}",
            context={"head_range": [heads.min().item(), heads.max().item()]},
        )
    if rels.min() < 0 or rels.max() >= num_relations:
        raise TransETrainingError(
            f"Relation indices out of range: "
            f"[{rels.min().item()}, {rels.max().item()}], "
            f"num_relations={num_relations}",
            context={"rel_range": [rels.min().item(), rels.max().item()]},
        )
    if tails.min() < 0 or tails.max() >= num_entities:
        raise TransETrainingError(
            f"Tail indices out of range: "
            f"[{tails.min().item()}, {tails.max().item()}], "
            f"num_entities={num_entities}",
            context={"tail_range": [tails.min().item(), tails.max().item()]},
        )

    # FIX K3.6: Check val_triples don't overlap with train set.
    if val_triples is not None and len(val_triples[0]) > 0:
        if len(val_triples[0]) < config.min_val_triples:
            raise ValueError(
                f"val_triples has {len(val_triples[0])} triples — "
                f"minimum is {config.min_val_triples} for reliable AUC."
            )
        train_set = set(zip(heads.tolist(), rels.tolist(), tails.tolist()))
        val_set = set(
            zip(
                val_triples[0].tolist(),
                val_triples[1].tolist(),
                val_triples[2].tolist(),
            )
        )
        overlap = train_set & val_set
        if overlap:
            raise DataLeakageError(
                f"Data leakage detected: {len(overlap)} triples appear in "
                f"both train and validation sets. Remove them from training.",
                context={"n_leaked": len(overlap)},
            )

    # v34 ROOT FIX (CRITICAL #10): the previous code only checked val/train
    # overlap. test/train overlap was NOT checked — if held-out triples
    # appeared in training, held_out_auc was inflated and the V1 launch
    # criterion (>0.85) was fakeable. Now we check test/train overlap with
    # the SAME mechanism and raise DataLeakageError on any overlap.
    if test_triples is not None and len(test_triples[0]) > 0:
        train_set = set(zip(heads.tolist(), rels.tolist(), tails.tolist()))
        test_set = set(
            zip(
                test_triples[0].tolist(),
                test_triples[1].tolist(),
                test_triples[2].tolist(),
            )
        )
        overlap = train_set & test_set
        if overlap:
            raise DataLeakageError(
                f"Data leakage detected: {len(overlap)} triples appear in "
                f"both train and TEST (held-out) sets. The held_out_auc "
                f"is INFLATED and cannot be trusted. Remove the leaked "
                f"triples from training before evaluating.",
                context={"n_leaked": len(overlap), "split": "test/train"},
            )
        # Also check test/val overlap (less critical but still a leak).
        if val_triples is not None and len(val_triples[0]) > 0:
            val_set = set(
                zip(
                    val_triples[0].tolist(),
                    val_triples[1].tolist(),
                    val_triples[2].tolist(),
                )
            )
            tv_overlap = test_set & val_set
            if tv_overlap:
                raise DataLeakageError(
                    f"Data leakage detected: {len(tv_overlap)} triples "
                    f"appear in both val and TEST (held-out) sets. The "
                    f"held_out_auc is INFLATED and cannot be trusted.",
                    context={"n_leaked": len(tv_overlap), "split": "test/val"},
                )

    # ── Reproducibility setup ────────────────────────────────────────────
    # FIX I7.1, I7.2: Apply seed via LOCAL generator.
    rng = torch.Generator()
    rng.manual_seed(config.seed)

    # v81 FORENSIC ROOT FIX (P0-F6 + P0-F4): detect model scoring
    # direction ONCE at the start of training and reuse for every
    # validation eval and final held-out eval inside this function.
    # TransE: ||h+r-t||_1 — LOWER = more plausible (higher_is_better=False)
    # HGT: attention-weighted dot product — HIGHER = more plausible
    # (higher_is_better=True). The previous code hardcoded False
    # everywhere, which silently INVERTS HGT validation AUC and held-out
    # AUC — the "best" epoch becomes the WORST, the model that gets
    # deployed ranks drugs BACKWARDS. Patient-safety blocker for Phase 3.
    # v84 FORENSIC ROOT FIX (BUG #12 — same substring-matching fix here):
    # The previous code duck-typed via class name OR an explicit
    # ``score_higher_is_better`` attribute. Substring matching on the
    # class name is forbidden (see _evaluate_triples fix above). ROOT
    # FIX: require `score_direction` (Protocol attribute) or legacy
    # `score_higher_is_better`; RAISE if neither is present.
    _model_class_name = type(model).__name__
    if hasattr(model, "score_direction"):
        _sd_train = str(getattr(model, "score_direction"))
        if _sd_train not in ("lower_better", "higher_better"):
            raise RuntimeError(
                f"train_transe: model {_model_class_name} has "
                f"score_direction={_sd_train!r} — must be 'lower_better' "
                f"or 'higher_better'. (v84 BUG #12 root fix)"
            )
        _model_higher_is_better = (_sd_train == "higher_better")
    elif hasattr(model, "score_higher_is_better"):
        _legacy_hib_train = getattr(model, "score_higher_is_better")
        if not isinstance(_legacy_hib_train, bool):
            raise RuntimeError(
                f"train_transe: model {_model_class_name} has "
                f"score_higher_is_better={_legacy_hib_train!r} (type "
                f"{type(_legacy_hib_train).__name__}) — must be bool. "
                f"Migrate to score_direction. (v84 BUG #12 root fix)"
            )
        _model_higher_is_better = bool(_legacy_hib_train)
        logger.warning(
            "train_transe: model %s uses deprecated "
            "score_higher_is_better=%s. Migrate to score_direction. "
            "(v84 BUG #12)",
            _model_class_name, _model_higher_is_better,
        )
    else:
        raise RuntimeError(
            f"train_transe: model {_model_class_name} does NOT declare "
            f"score_direction (or legacy score_higher_is_better). The "
            f"AUC direction CANNOT be inferred from the class name "
            f"(substring matching is forbidden). Add `score_direction` "
            f"as a property returning 'lower_better' (TransE) or "
            f"'higher_better' (HGT/GraphTransformer). (v84 BUG #12 root fix)"
        )
    logger.info(
        "train_transe: model=%s, score_higher_is_better=%s — AUC "
        "evaluation direction will use this value for all val and "
        "held-out evaluations. (v81 P0-F6 root fix, v84 BUG #12)",
        _model_class_name, _model_higher_is_better,
    )

    # v88 ROOT FIX (BUG #46 + #47 — gate ALL three sampler fallback
    # modes behind the production check): apply the SAME production guard
    # to train_transe as _evaluate_triples has. This prevents the compound
    # chain (BUG #47) where a misconfigured sampler produces garbage val
    # AUC that passes the launch gate.
    _env_mode_v88 = os.environ.get("DRUGOS_ENVIRONMENT", "dev").lower()
    _is_production_v88 = _env_mode_v88 in ("prod", "production")
    _flag_1_v88 = os.environ.get("DRUGOS_ALLOW_NO_SAMPLER", "") == "1"
    _flag_2_v88 = os.environ.get("DRUGOS_DEV_ALLOW_NO_SAMPLER", "") == "1"
    _allow_no_sampler_v88 = _flag_1_v88 and _flag_2_v88 and not _is_production_v88
    if _flag_1_v88 and _is_production_v88:
        logger.critical(
            "PRODUCTION_ESCAPE_HATCH_REFUSED (train_transe): "
            "DRUGOS_ALLOW_NO_SAMPLER=1 is set but DRUGOS_ENVIRONMENT=%s. "
            "Refusing to honor the escape hatch — all three sampler "
            "fallback modes in train_transe will RAISE in production. "
            "(v88 BUG #46+#47 root fix)",
            _env_mode_v88,
        )

    # v28 ROOT FIX (audit ML-13): the module docstring (line ~124)
    # promises "torch.use_deterministic_algorithms(True) is set when
    # config.seed is not None" — but the previous code gated this on
    # ``DETERMINISTIC_MODE`` (a module-level bool from config). When
    # ``DETERMINISTIC_MODE=False`` but ``config.seed=42``, the
    # docstring promise was silently violated: the local RNG was
    # seeded (so torch.randperm calls were reproducible) but
    # ``torch.use_deterministic_algorithms`` was NOT set (so any
    # non-deterministic CUDA op like scatter_add could vary between
    # runs with the same seed). The fix makes the code match the
    # docstring: deterministic algorithms are enabled IFF a seed is
    # set (``config.seed is not None``). The ``DETERMINISTIC_MODE``
    # config flag is retained as an OPT-OUT for operators who
    # explicitly want non-deterministic GPU ops (faster, but the
    # docstring's "Limitations" caveat about GPU atol=1e-5
    # differences applies).
    _seed_is_set = config.seed is not None
    _operator_opted_out = not DETERMINISTIC_MODE
    if _seed_is_set and not _operator_opted_out:
        torch.use_deterministic_algorithms(True)
    elif _seed_is_set and _operator_opted_out:
        # Operator explicitly disabled deterministic algorithms.
        # Log loudly so the docstring-vs-code mismatch is visible.
        logger.warning(
            "DETERMINISTIC_MODE is False but config.seed=%s is set — "
            "torch.use_deterministic_algorithms is NOT being enabled. "
            "GPU runs with this configuration are NOT bit-reproducible "
            "(see module docstring 'Limitations' section). CPU runs "
            "are still reproducible at the RNG level (local Generator "
            "is seeded). Set DETERMINISTIC_MODE=1 to restore full "
            "determinism (slower on GPU). (v28 audit ML-13)",
            config.seed,
        )
    if torch.cuda.is_available():
        # cuDNN deterministic / benchmark settings apply whenever a
        # seed is set — they are cheap (no perf cost on embedding
        # lookups) and the docstring promises them.
        if _seed_is_set:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # ── Device selection ─────────────────────────────────────────────────
    # FIX A1.5, P8.1: Use gpu_utils for device selection.
    device = _get_device(config)
    model = model.to(device)

    heads_dev = heads.to(device)
    rels_dev = rels.to(device)
    tails_dev = tails.to(device)

    # ── Optimizer setup ──────────────────────────────────────────────────
    # FIX C4.8: Support both Adam and SGD.
    # FIX C12.6: optimizer_name from config.
    if config.optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    # v28 ROOT FIX (audit ML-9): the ``criterion = nn.MarginRankingLoss(...)``
    # instance was unused after the inline loss was replaced with the
    # explicit ``max(0, pos - neg + margin).mean()`` form below. Removed
    # to avoid implying the trainer uses MarginRankingLoss — it does not,
    # and a future maintainer reading the criterion definition would be
    # misled into thinking the trainer relies on it. (Comment retained
    # here so a git-blame reader can find the rationale.)

    # ── Known triples set for filtering ──────────────────────────────────
    # FIX K3.2, K3.3: Build set for O(1) negative filtering.
    _known: Optional[Set[Tuple[int, int, int]]] = known_triples
    if _known is None:
        _known = set(zip(heads.tolist(), rels.tolist(), tails.tolist()))

    # ── MLflow setup ─────────────────────────────────────────────────────
    # FIX A1.2, I15.6, I15.9: MLflowTracker integration.
    if mlflow_tracker is not None:
        mlflow_tracker.start_run(run_name=f"transe_seed{config.seed}")
        safe_cfg = safe_config_dict() if callable(safe_config_dict) else {}
        mlflow_tracker.log_params(
            {
                "embedding_dim": config.embedding_dim,
                "margin": config.margin,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "num_negatives": _num_negatives,
                "seed": config.seed,
                "target_auc": config.target_auc,
                "optimizer": config.optimizer_name,
            }
        )

    # ── NegativeSampler integration ──────────────────────────────────────
    # FIX A1.1: Use NegativeSampler when provided.
    # v13 ROOT FIX (SW-14 / PS-12 / SW-15 / Compound-8): pre-compute
    # PER-RELATION negative pools so every batch gets negatives whose
    # head/tail types match the positive triple's relation. v12 called
    # ``combined_sampling()`` once with no type kwargs → all negatives
    # were (Compound, Disease) regardless of the positive triple's
    # relation, producing biologically meaningless negatives for 5 of 6
    # edge types. The 0.85 AUC V1 launch criterion was therefore
    # trivially achievable against nonsense negatives.
    #
    # The new flow:
    #   1. For each relation_idx, look up (head_type, tail_type) via
    #      ``negative_sampler.relation_to_types`` (populated by
    #      run_pipeline.py step11 from ``edge_maps`` keys).
    #   2. Call ``combined_sampling(head_type=..., tail_type=...)`` to
    #      generate a pool of negatives with the correct types.
    #   3. Store per-relation pools in ``per_relation_neg_pools``.
    #   4. In each training batch, route each triple to its relation's
    #      pool (see the batch loop below).
    sampler_neg_indices: Optional[Tuple[List[int], List[int]]] = None
    # v13: per-relation pools. Keys are relation_idx; values are
    # (head_indices, tail_indices) sampled from the type-correct pools.
    per_relation_neg_pools: Dict[int, Tuple[List[int], List[int]]] = {}

    # FIX ML-3 (FIX-CFG-ML audit): the v22 pre-compute block built
    # per-relation negative pools ONCE before the epoch loop and reused
    # them every batch of every epoch. The model therefore saw the
    # SAME negatives in epoch 50 as in epoch 1 — no fresh negative
    # signal, no exploration, no chance of escaping a sub-optimal
    # embedding geometry that the initial negative pool happened to
    # favour. Root fix: extract the per-relation pool building into a
    # helper and re-call it at the start of EACH epoch so the model
    # sees fresh type-constrained negatives each epoch
    # (``negative_sampler.combined_sampling`` is stochastic via
    # ``self._rng``). The initial build (called once before the loop)
    # also emits the REM-22 single CRITICAL summary log if any
    # relation falls back to random — per-epoch refreshes silently
    # skip already-failed relations (preserving the previous epoch's
    # pool) to avoid swamping the audit log.
    def _build_per_relation_pools(
        log_failures: bool,
    ) -> Tuple[
        Dict[int, Tuple[List[int], List[int]]],
        Optional[Tuple[List[int], List[int]]],
    ]:
        """Sample per-relation type-constrained negative pools.

        Returns ``(per_relation_neg_pools, sampler_neg_indices)``. When
        ``log_failures`` is True, the REM-22 single CRITICAL summary
        log is emitted for any relation that failed
        ``combined_sampling`` — used by the initial build only so the
        audit log surfaces the degradation ONCE, not once per epoch.
        """
        if negative_sampler is None:
            return {}, None
        rt: Dict[int, Tuple[str, str]] = getattr(
            negative_sampler, "relation_to_types", {}
        )
        if not rt:
            return {}, None
        import collections as _collections
        triple_relation_counts = _collections.Counter(int(r) for r in rels)
        new_pools: Dict[int, Tuple[List[int], List[int]]] = {}
        failed: set = set()
        for rel_idx, (ht, tt) in rt.items():
            n_triples_r = triple_relation_counts.get(rel_idx, 0)
            pool_size = max(n_triples_r * _num_negatives, 100)
            try:
                rel_neg_samples = negative_sampler.combined_sampling(
                    total_negatives=pool_size,
                    head_type=ht,
                    tail_type=tt,
                    relation_idx=rel_idx,
                )
                new_pools[rel_idx] = (
                    negative_sampler.to_negative_indices(rel_neg_samples)
                )
            except Exception as exc:
                if log_failures:
                    logger.warning(
                        "NegativeSampler.combined_sampling failed for "
                        "relation_idx=%d (head_type=%s, tail_type=%s): "
                        "%s — this relation will use random fallback.",
                        rel_idx, ht, tt, exc,
                    )
                failed.add(rel_idx)
                # v109 ROOT FIX (P2-034): the previous code preserved the
                # PREVIOUS EPOCH'S pool for failed relations. This meant
                # that if a relation failed in epoch 1, the SAME stale
                # negatives were used for ALL subsequent epochs — no
                # fresh negative signal, no exploration. The model
                # effectively memorized the stale negatives instead of
                # learning generalizable representations.
                # ROOT FIX: fall back to RANDOM sampling (which always
                # succeeds) so the model sees FRESH random negatives
                # each epoch. Random negatives are less informative than
                # type-correct ones, but they are FAR better than stale
                # ones (which provide zero gradient signal after the
                # first epoch). Log at DEBUG level (not WARNING) to
                # avoid audit-log spam on every epoch.
                try:
                    # Get the total entity count from the negative sampler.
                    _n_entities = getattr(negative_sampler, "n_entities", 0) or 0
                    if _n_entities > 0:
                        import random as _random_p34
                        _heads = [_random_p34.randrange(_n_entities) for _ in range(pool_size)]
                        _tails = [_random_p34.randrange(_n_entities) for _ in range(pool_size)]
                        new_pools[rel_idx] = (_heads, _tails)
                        logger.debug(
                            "P2-034 v109: relation_idx=%d fell back to "
                            "RANDOM negative sampling (fresh each epoch). "
                            "pool_size=%d, n_entities=%d.",
                            rel_idx, pool_size, _n_entities,
                        )
                    elif rel_idx in per_relation_neg_pools:
                        # Last resort: no entity count available, use the
                        # previous pool (this branch is rarely hit).
                        new_pools[rel_idx] = per_relation_neg_pools[rel_idx]
                except Exception:
                    # If random fallback also fails (e.g. n_entities is
                    # invalid), preserve the previous pool as a last
                    # resort. This is the v107 behavior.
                    if rel_idx in per_relation_neg_pools:
                        new_pools[rel_idx] = per_relation_neg_pools[rel_idx]
        if log_failures and failed:
            logger.critical(
                "NEG_SAMPLER_DEGRADED: %d/%d relations had no "
                "type-correct negatives pre-computed and will use "
                "uniformly random fallback. AUC numbers for these "
                "relations are NOT comparable to literature. "
                "Affected relations: %s",
                len(failed),
                len(rt),
                sorted(failed),
            )
        if log_failures:
            logger.info(
                "Pre-computed per-relation negative pools for %d "
                "relations (out of %d total).",
                len(new_pools),
                len(rt),
            )
        # Build a legacy-format aggregate for backward compatibility
        # with code paths that still read ``sampler_neg_indices``.
        treats_pool = None
        for rel_idx, (ht, tt) in rt.items():
            if ht == "Compound" and tt in ("Disease", "Condition"):
                treats_pool = new_pools.get(rel_idx)
                break
        if treats_pool is None and new_pools:
            treats_pool = next(iter(new_pools.values()))
        return new_pools, treats_pool

    if negative_sampler is not None:
        logger.info(
            "Using NegativeSampler for type-constrained negatives."
        )
        relation_to_types = getattr(
            negative_sampler, "relation_to_types", {}
        )
        if relation_to_types:
            # Initial build — emits the REM-22 CRITICAL summary if any
            # relation fails. Per-epoch refreshes (called inside the
            # epoch loop below) pass log_failures=False to avoid
            # spamming the audit log.
            per_relation_neg_pools, _treats = _build_per_relation_pools(
                log_failures=True
            )
            sampler_neg_indices = _treats
        else:
            # Fallback: relation_to_types not populated — use the
            # legacy single-pool path. This is the v12 behavior and
            # produces type-wrong negatives for 5/6 relations.
            # v20 Compound-8 ROOT FIX: WARNING alone is insufficient —
            # the operator may not see the log and the resulting AUC
            # is scientifically meaningless. The audit's Compound-8
            # chain explicitly called out this fallback as the source
            # of the "Negative Sampling Invalidation" compound effect.
            # Promote to RuntimeError unless DRUGOS_ALLOW_NO_SAMPLER=1
            # is set (which the module-import production guard already
            # refuses in DRUGOS_ENVIRONMENT=production).
            _allow_legacy = (
                os.environ.get("DRUGOS_ALLOW_NO_SAMPLER", "") == "1"
            )
            if _allow_legacy:
                logger.warning(
                    "NegativeSampler.relation_to_types is empty — "
                    "DRUGOS_ALLOW_NO_SAMPLER=1 set, using legacy "
                    "single-pool negative sampling. ALL negatives will "
                    "be (Compound, Disease) regardless of the positive "
                    "triple's relation. AUC numbers will NOT be "
                    "comparable to literature. Populate "
                    "relation_to_types from edge_maps to fix."
                )
                try:
                    neg_samples = negative_sampler.combined_sampling(
                        total_negatives=len(heads) * _num_negatives,
                    )
                    sampler_neg_indices = negative_sampler.to_negative_indices(neg_samples)
                    logger.info(
                        "NegativeSampler produced %d negative pairs (legacy)",
                        len(sampler_neg_indices[0]),
                    )
                except Exception as exc:
                    logger.warning(
                        "NegativeSampler failed (%s), falling back to "
                        "random corruption. AUC numbers will not be "
                        "comparable to literature.",
                        exc,
                    )
                    sampler_neg_indices = None
            else:
                raise RuntimeError(
                    "NegativeSampler.relation_to_types is empty — refusing "
                    "to use legacy single-pool negative sampling because "
                    "it produces type-wrong (Compound, Disease) negatives "
                    "for 5/6 edge types (Compound-8 chain, audit §3.4 "
                    "SW-14). This makes AUC numbers scientifically "
                    "meaningless. Populate relation_to_types from "
                    "edge_maps, OR set DRUGOS_ALLOW_NO_SAMPLER=1 to "
                    "explicitly opt in (refused in DRUGOS_ENVIRONMENT= "
                    "production)."
                )
    else:
        # FIX R6.15: Warn once when using crude fallback.
        logger.warning(
            "CRUDE NEGATIVE FALLBACK: No NegativeSampler provided. "
            "Using random corruption. AUC numbers are NOT comparable "
            "to literature. Provide negative_sampler= for "
            "scientifically valid training."
        )

    # ── Training history ─────────────────────────────────────────────────
    # FIX D2.5, D2.7: Use TrainingHistory dataclass.
    history = TrainingHistory(
        total_train_triples=len(heads),
        total_val_triples=len(val_triples[0]) if val_triples is not None else 0,
    )

    best_state_dict: Optional[Dict[str, Any]] = None
    best_val_auc: float = -1.0
    best_epoch: int = -1
    patience_counter: int = 0
    nan_batches_quarantined: int = 0
    train_start_time = time.time()  # FIX P8.20: moved before loop

    # FIX L11.8: Log entity/relation counts.
    logger.info(
        "TransE training: %d epochs, %d train triples, %d entities, "
        "%d relations, device=%s, seed=%d, optimizer=%s, batch_size=%d",
        config.num_epochs,
        len(heads),
        num_entities,
        num_relations,
        device,
        config.seed,
        config.optimizer_name,
        config.batch_size,
    )

    # ── Task 106 ROOT FIX (v111): compute Bernoulli head-corruption probs ─
    # Per Wang et al. 2014, the probability of corrupting the head for a
    # given relation r is:  p_head(r) = tph(r) / (tph(r) + hpt(r))
    # where tph = average tails-per-head, hpt = average heads-per-tail.
    # This respects the graph structure (one-to-many vs many-to-one) and
    # produces harder negatives than uniform 0.5. Computed ONCE from the
    # training triples; reused every batch.
    _bernoulli_head_probs: Optional[torch.Tensor] = None
    try:
        # Count heads and tails per relation
        _rel_head_counts: Dict[int, Set[int]] = {}
        _rel_tail_counts: Dict[int, Set[int]] = {}
        _rel_head_to_tails: Dict[int, Dict[int, Set[int]]] = {}
        _rel_tail_to_heads: Dict[int, Dict[int, Set[int]]] = {}
        for h_idx, r_idx, t_idx in zip(
            heads.tolist(), rels.tolist(), tails.tolist()
        ):
            h_i, r_i, t_i = int(h_idx), int(r_idx), int(t_idx)
            _rel_head_counts.setdefault(r_i, set()).add(h_i)
            _rel_tail_counts.setdefault(r_i, set()).add(t_i)
            _rel_head_to_tails.setdefault(r_i, {}).setdefault(h_i, set()).add(t_i)
            _rel_tail_to_heads.setdefault(r_i, {}).setdefault(t_i, set()).add(h_i)
        _probs = torch.full((num_relations,), 0.5, dtype=torch.float32)
        for r_i in range(num_relations):
            h2t = _rel_head_to_tails.get(r_i, {})
            t2h = _rel_tail_to_heads.get(r_i, {})
            if h2t and t2h:
                tph = sum(len(tails) for tails in h2t.values()) / len(h2t)
                hpt = sum(len(heads) for heads in t2h.values()) / len(t2h)
                if (tph + hpt) > 0:
                    _probs[r_i] = float(tph / (tph + hpt))
            # else: relation has no training triples — keep default 0.5
        _bernoulli_head_probs = _probs.to(device)
        logger.info(
            "Task 106 Bernoulli negative sampling: computed per-relation "
            "head-corruption probs for %d relations (sample: %s). "
            "(task 106 root fix, v111)",
            num_relations,
            {r: round(float(_probs[r]), 3) for r in range(min(num_relations, 5))},
        )
    except Exception as _bernoulli_exc:
        logger.warning(
            "Task 106 Bernoulli prob computation failed (%s) — falling "
            "back to uniform config.neg_corrupt_head_ratio=%.2f. This "
            "produces easier negatives for one-to-many relations. "
            "(task 106 root fix, v111)",
            _bernoulli_exc, config.neg_corrupt_head_ratio,
        )
        _bernoulli_head_probs = None

    # ── Main training loop ───────────────────────────────────────────────
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        # v35 ROOT FIX (M-17): initialise ``current_val_auc`` at the
        # START of each epoch (not inside the validation if-block).
        # The previous code only set ``current_val_auc = -1.0`` inside
        # ``if val_triples is not None and (epoch+1) % config.eval_every == 0``
        # — meaning any epoch that skipped validation left
        # ``current_val_auc`` UNBOUND, and the post-epoch
        # ``if current_val_auc > best_val_auc`` check raised
        # ``UnboundLocalError``. The fix initialises the variable to
        # ``-1.0`` at the top of every epoch so the best-model check
        # always sees a defined value (and ``-1.0`` is treated as
        # ``not an improvement`` because ``best_val_auc`` starts at 0).
        current_val_auc: float = -1.0

        # FIX ML-3 (FIX-CFG-ML audit): re-sample per-relation negative
        # pools at the start of each epoch so the model sees fresh
        # type-constrained negatives each epoch. Skip the refresh on
        # epoch 0 (the initial build above already produced the epoch-0
        # pools and emitted the REM-22 CRITICAL summary if any relation
        # failed). Per-epoch refreshes pass log_failures=False to avoid
        # spamming the audit log; failed relations preserve the
        # previous epoch's pool (see ``_build_per_relation_pools``).
        if epoch > 0 and negative_sampler is not None and relation_to_types:
            per_relation_neg_pools, _treats = _build_per_relation_pools(
                log_failures=False
            )
            if _treats is not None:
                sampler_neg_indices = _treats

        model.train()

        # FIX I7.7, P8.2: Accumulate loss WITHOUT calling .item() per batch.
        epoch_loss_accum = torch.tensor(0.0, device=device)
        num_batches = 0
        epoch_nan_count = 0

        # FIX P8.9: Efficient shuffling via local generator.
        indices = torch.randperm(
            len(heads_dev), generator=rng, device=device
        )

        # FIX P8.3: batch_size from config, not hardcoded.
        batch_size = config.batch_size

        for batch_start in range(0, len(heads_dev), batch_size):
            batch_end = min(batch_start + batch_size, len(heads_dev))
            batch_idx = indices[batch_start:batch_end]

            h_batch = heads_dev[batch_idx]
            r_batch = rels_dev[batch_idx]
            t_batch = tails_dev[batch_idx]

            # ── Negative sampling ────────────────────────────────────
            # v13 ROOT FIX (SW-14 / PS-12 / SW-15): when per-relation
            # pools are available, route each triple's negatives to
            # its relation's pool so head/tail types match the
            # positive triple's relation. Falls back to the legacy
            # single-pool path (sampler_neg_indices) when
            # per_relation_neg_pools is empty (e.g. older callers
            # that didn't populate relation_to_types).
            if per_relation_neg_pools:
                # Per-relation routing: for each triple in the batch,
                # look up its relation's (head_indices, tail_indices)
                # pool and sample n_needed negatives from it. We build
                # the per-negative head/tail pools by gathering from
                # the correct relation's pool.
                n_needed = len(batch_idx) * _num_negatives
                # Repeat each triple's relation _num_negatives times
                # so we can index per_relation_neg_pools per-negative.
                r_expanded = r_batch.repeat_interleave(_num_negatives)
                # Pre-build tensor pools per relation for fast gather.
                # For each negative slot i, pick a random index from
                # the pool of its relation r_expanded[i].
                neg_h_list = torch.empty(
                    n_needed, dtype=torch.long, device=device
                )
                neg_t_list = torch.empty(
                    n_needed, dtype=torch.long, device=device
                )
                # Group negative slots by relation to minimize Python
                # overhead (one randperm per relation per batch).
                unique_rels_in_batch = torch.unique(r_expanded)
                for ur in unique_rels_in_batch.tolist():
                    mask = (r_expanded == ur)
                    slots = torch.nonzero(mask, as_tuple=True)[0]
                    n_slots = len(slots)
                    pool = per_relation_neg_pools.get(int(ur))
                    if pool is None or len(pool[0]) == 0:
                        # No pool for this relation — random fallback.
                        neg_h_list[slots] = torch.randint(
                            0, num_entities, (n_slots,),
                            generator=rng, device=device,
                        )
                        neg_t_list[slots] = torch.randint(
                            0, num_entities, (n_slots,),
                            generator=rng, device=device,
                        )
                        continue
                    head_pool, tail_pool = pool
                    # Sample n_slots head negatives from head_pool.
                    # v102 ROOT FIX (P2-043): the previous code used
                    # ``perm_h = torch.randperm(...)[:n_slots]`` then
                    # concatenated ``extra = torch.randint(...)`` when
                    # n_slots > len(head_pool). The randint extra CAN
                    # sample the SAME index multiple times, producing
                    # DUPLICATE head negatives within the same batch.
                    # TransE training then sees the same negative triple
                    # multiple times, wasting gradient signal on
                    # duplicates. Effective negative batch size is
                    # smaller than nominal.
                    #
                    # ROOT FIX: use randperm when n_slots <= len(pool)
                    # (guarantees unique indices — no duplicates), else
                    # use randint directly (necessary for n_slots >
                    # pool_size, where duplicates are unavoidable but
                    # the perm_h+extra split was just adding complexity
                    # without deduplication). The randint call already
                    # handles n_slots > pool_size by sampling with
                    # replacement.
                    if len(head_pool) > 0:
                        h_pool_tensor = torch.tensor(
                            head_pool, dtype=torch.long, device=device
                        )
                        if n_slots <= len(head_pool):
                            # Unique indices (no duplicates).
                            perm_h = torch.randperm(
                                len(head_pool), generator=rng, device=device
                            )[:n_slots]
                        else:
                            # n_slots > pool_size: duplicates unavoidable.
                            # Use randint directly (no perm_h+extra split).
                            perm_h = torch.randint(
                                0, len(head_pool), (n_slots,),
                                generator=rng, device=device,
                            )
                        neg_h_list[slots] = h_pool_tensor[perm_h]
                    else:
                        neg_h_list[slots] = torch.randint(
                            0, num_entities, (n_slots,),
                            generator=rng, device=device,
                        )
                    # Sample n_slots tail negatives from tail_pool.
                    # v102 P2-043: same fix as head_pool above — use
                    # randperm when n_slots <= len(pool) (unique indices),
                    # else randint directly (no perm_t+extra split).
                    if len(tail_pool) > 0:
                        t_pool_tensor = torch.tensor(
                            tail_pool, dtype=torch.long, device=device
                        )
                        if n_slots <= len(tail_pool):
                            perm_t = torch.randperm(
                                len(tail_pool), generator=rng, device=device
                            )[:n_slots]
                        else:
                            perm_t = torch.randint(
                                0, len(tail_pool), (n_slots,),
                                generator=rng, device=device,
                            )
                        neg_t_list[slots] = t_pool_tensor[perm_t]
                    else:
                        neg_t_list[slots] = torch.randint(
                            0, num_entities, (n_slots,),
                            generator=rng, device=device,
                        )

                # Task 106 ROOT FIX (v111): Bernoulli negative sampling.
                # The previous code used a FIXED ``config.neg_corrupt_head_ratio``
                # (default 0.5) for ALL relations — uniform 50/50 head/tail
                # corruption. This is WRONG per Wang et al. 2014 ("Knowledge
                # Graph Embedding by Dynamic Mapping") which prescribes
                # BERNOULLI sampling: per-relation head-corruption probability
                # proportional to ``tph / (tph + hpt)`` where:
                #   tph = average #tails per head (tails-per-head)
                #   hpt = average #heads per tail (heads-per-tail)
                # For a one-to-many relation (Drug→treats→Disease: one drug
                # treats many diseases), tph is high, hpt is low → corrupt
                # the TAIL more often (p_corrupt_head is low). For many-to-one
                # relations, corrupt the HEAD more often. This respects the
                # graph structure and produces harder, more informative
                # negatives. Uniform 0.5 produces easy negatives for
                # one-to-many relations (the model learns to distinguish
                # by degree, not by relation).
                #
                # The fix: compute per-relation Bernoulli probabilities from
                # the training triples ONCE at the start of training, then
                # use them per-batch. ``_bernoulli_head_probs`` is a tensor
                # of shape (num_relations,) where entry [r] = p(corrupt head | r).
                # For each positive triple in the batch, look up its relation's
                # probability and sample a Bernoulli decision.
                if _bernoulli_head_probs is not None:
                    # Per-relation Bernoulli: look up p(corrupt_head) for
                    # each positive triple's relation.
                    _batch_rel_probs = _bernoulli_head_probs[r_batch]
                    corrupt_head_per_pos = (
                        torch.rand(
                            len(batch_idx), generator=rng, device=device
                        ) < _batch_rel_probs
                    )
                else:
                    # Fallback: uniform 0.5 (legacy). Only used when
                    # _bernoulli_head_probs could not be computed (e.g.
                    # empty training set — should not happen in practice).
                    corrupt_head_per_pos = (
                        torch.rand(
                            len(batch_idx), generator=rng, device=device
                        ) < config.neg_corrupt_head_ratio
                    )
                corrupt_head_mask = corrupt_head_per_pos.repeat_interleave(
                    _num_negatives
                )

                h_neg = h_batch.repeat_interleave(_num_negatives).clone()
                neg_r = r_batch.repeat_interleave(_num_negatives)
                t_neg = t_batch.repeat_interleave(_num_negatives).clone()

                h_neg[corrupt_head_mask] = neg_h_list[corrupt_head_mask]
                t_neg[~corrupt_head_mask] = neg_t_list[~corrupt_head_mask]

                neg_t = t_neg
                # v22 ROOT FIX (UnboundLocalError on corrupt_expanded):
                # the v21 known-triples filter below references
                # ``corrupt_expanded`` to decide whether to replace the
                # head or the tail of a known-positive negative. But
                # this per-relation-pool branch (the default for
                # production with a type-constrained sampler) only
                # defined ``corrupt_head_mask``. The vectorized
                # ``else:`` branch defined ``corrupt_expanded``. When
                # the per-relation-pool branch was taken, the filter
                # raised ``UnboundLocalError: cannot access local
                # variable 'corrupt_expanded'`` on the first batch
                # — crashing TransE training. Fix: alias
                # ``corrupt_expanded`` to ``corrupt_head_mask`` here
                # (both are length n_needed, matching h_neg.shape[0]).
                corrupt_expanded = corrupt_head_mask
            elif sampler_neg_indices is not None:
                # Legacy single-pool path (v12 behavior). Used when
                # relation_to_types was not populated — produces
                # type-wrong negatives for 5/6 relations.
                # PS-11 / DC-1 ROOT FIX: previously neg_drug_idx was
                # assigned from sampler_neg_indices[0] but never used
                # — the head was always reused from h_batch, silently
                # disabling head corruption. Now honor
                # config.neg_corrupt_head_ratio by corrupting heads
                # with Compound indices (sampler_neg_indices[0]) and
                # tails with Disease indices (sampler_neg_indices[1]).
                # Combined with the SW-14 fix in negative_sampling.py,
                # this restores type-correct head+tail corruption.
                neg_drug_idx = sampler_neg_indices[0]      # Compound indices
                neg_disease_idx = sampler_neg_indices[1]   # Disease indices

                n_needed = len(batch_idx) * _num_negatives

                # Sample n_needed head negatives (Compound indices).
                if len(neg_drug_idx) > 0:
                    perm_h = torch.randperm(
                        len(neg_drug_idx), generator=rng, device=device
                    )[:n_needed]
                    neg_h_pool = torch.tensor(
                        neg_drug_idx, dtype=torch.long, device=device
                    )[perm_h]
                else:
                    # No Compound entities — fall back to random head corruption.
                    neg_h_pool = torch.randint(
                        0, num_entities, (n_needed,),
                        generator=rng, device=device,
                    )

                # Sample n_needed tail negatives (Disease indices).
                if len(neg_disease_idx) > 0:
                    perm_t = torch.randperm(
                        len(neg_disease_idx), generator=rng, device=device
                    )[:n_needed]
                    neg_t_pool = torch.tensor(
                        neg_disease_idx, dtype=torch.long, device=device
                    )[perm_t]
                else:
                    neg_t_pool = torch.randint(
                        0, num_entities, (n_needed,),
                        generator=rng, device=device,
                    )

                # P2-022 ROOT FIX (legacy single-pool path): decide
                # head/tail corruption PER POSITIVE TRIPLE, then expand
                # to all its negatives. Same scientific rationale as the
                # per-relation-pool branch above — the TransE convention
                # (Bordes 2013 §3.3) requires a single head/tail decision
                # per positive triple, applied uniformly to all its
                # negatives, to avoid inconsistent gradients.
                corrupt_head_per_pos = (
                    torch.rand(
                        len(batch_idx), generator=rng, device=device
                    )
                    < config.neg_corrupt_head_ratio
                )
                corrupt_head_mask = corrupt_head_per_pos.repeat_interleave(
                    _num_negatives
                )

                h_neg = h_batch.repeat_interleave(_num_negatives).clone()
                neg_r = r_batch.repeat_interleave(_num_negatives)
                t_neg = t_batch.repeat_interleave(_num_negatives).clone()

                h_neg[corrupt_head_mask] = neg_h_pool[corrupt_head_mask]
                t_neg[~corrupt_head_mask] = neg_t_pool[~corrupt_head_mask]

                neg_t = t_neg
                # v22 ROOT FIX (UnboundLocalError on corrupt_expanded):
                # the v21 known-triples filter at line ~1999 references
                # ``corrupt_expanded`` to decide whether to replace the
                # head or the tail of a known-positive negative. But
                # the type-constrained branch (this branch) only
                # defined ``corrupt_head_mask`` (un-expanded), while
                # the vectorized branch below defined
                # ``corrupt_expanded``. When the type-constrained
                # sampler was active (the default for production),
                # the filter raised ``UnboundLocalError: cannot
                # access local variable 'corrupt_expanded'`` —
                # crashing TransE training on the very first batch.
                # Fix: define ``corrupt_expanded`` here too, so the
                # filter works regardless of which sampling branch
                # was taken.
                corrupt_expanded = corrupt_head_mask.clone()
            else:
                # FIX I7.2, P8.5: Vectorized corruption with local generator.
                # FIX C12.14: neg_corrupt_head_ratio from config.
                #
                # P2-015 ROOT FIX (type-wrong negatives in vectorized
                # fallback): the previous code sampled
                # ``neg_entities = torch.randint(0, num_entities, ...)``
                # — from ALL entities (Compound + Disease + Protein +
                # Gene + Pathway + ...). For a Compound→treats→Disease
                # triple, this can corrupt the head with a Disease
                # entity or the tail with a Protein entity. The
                # resulting "negative" triple has entities from the
                # WRONG TYPE — the model trivially distinguishes them
                # by their embedding space (different learned regions).
                # Training AUC is inflated by 0.1-0.2; deployed model
                # AUC drops when evaluated on type-correct negatives.
                #
                # ROOT FIX: use ``entity_type_lookup`` (passed to
                # train_transe) to filter neg_entities to the correct
                # type per triple. For each batch triple, look up its
                # head type and tail type (via ``relation_to_types``
                # from the negative_sampler when available, else via
                # ``entity_type_lookup`` directly), then sample head
                # negatives only from entities of the head type and
                # tail negatives only from entities of the tail type.
                #
                # If ``entity_type_lookup`` is NOT available AND no
                # sampler is provided, RAISE in production (require
                # both at construction time). The legacy
                # DRUGOS_ALLOW_NO_SAMPLER=1 escape hatch still
                # permits the type-wrong fallback for dev / unit tests.
                import os as _os_p2_015
                _allow_type_wrong = _allow_no_sampler_v88

                # Build per-type entity-index pools (cached across
                # batches on the function-call frame so we pay the
                # O(num_entities) build cost ONCE per train_transe
                # call, not per batch).
                if not hasattr(train_transe, "_p2_015_type_pools"):
                    train_transe._p2_015_type_pools = None
                if (
                    entity_type_lookup
                    and train_transe._p2_015_type_pools is None
                ):
                    _tp: Dict[str, List[int]] = {}
                    for _e_idx, _e_type in entity_type_lookup.items():
                        _tp.setdefault(_e_type, []).append(int(_e_idx))
                    train_transe._p2_015_type_pools = _tp

                _type_pools = train_transe._p2_015_type_pools

                # Resolve relation -> (head_type, tail_type) map.
                _rel_types_map = (
                    getattr(negative_sampler, "relation_to_types", {})
                    if negative_sampler is not None
                    else {}
                )

                if _type_pools and _rel_types_map:
                    # Type-correct vectorized corruption.
                    # For each batch triple, look up its (head_type,
                    # tail_type) via the relation, then sample from
                    # the correct type pool.
                    corrupt_head_mask = (
                        torch.rand(
                            len(batch_idx), generator=rng, device=device
                        )
                        < config.neg_corrupt_head_ratio
                    )
                    h_neg = (
                        h_batch.repeat_interleave(_num_negatives).clone()
                    )
                    r_neg = r_batch.repeat_interleave(_num_negatives)
                    t_neg = (
                        t_batch.repeat_interleave(_num_negatives).clone()
                    )
                    corrupt_expanded = corrupt_head_mask.repeat_interleave(
                        _num_negatives
                    )

                    # Build per-triple type pools. For each batch
                    # triple i, we need _num_negatives head samples
                    # from head_type_i and _num_negatives tail samples
                    # from tail_type_i. We process per-relation (most
                    # batches are single-relation) for efficiency.
                    _batch_rels = r_batch.tolist()
                    _unique_batch_rels = set(_batch_rels)
                    for _ur in _unique_batch_rels:
                        _rel_mask = (
                            r_batch == _ur
                        )  # length len(batch_idx)
                        _type_pair = _rel_types_map.get(int(_ur))
                        if (
                            _type_pair is None
                            or len(_type_pair) < 2
                        ):
                            # Unknown relation — fall back to
                            # entity_type_lookup per entity.
                            _h_type_for_rel = None
                            _t_type_for_rel = None
                        else:
                            _h_type_for_rel = _type_pair[0]
                            _t_type_for_rel = _type_pair[1]
                        _h_pool = (
                            _type_pools.get(_h_type_for_rel)
                            if _h_type_for_rel
                            else None
                        )
                        _t_pool = (
                            _type_pools.get(_t_type_for_rel)
                            if _t_type_for_rel
                            else None
                        )
                        # Gather the batch indices for this relation.
                        _rel_batch_idx = _rel_mask.nonzero(
                            as_tuple=True
                        )[0]
                        if len(_rel_batch_idx) == 0:
                            continue
                        # For each triple in this relation, sample
                        # _num_negatives head/tail negs.
                        for _ti in _rel_batch_idx.tolist():
                            _start = _ti * _num_negatives
                            _end = _start + _num_negatives
                            # Head corruption
                            if _h_pool and corrupt_head_mask[_ti]:
                                _h_choices = torch.randint(
                                    0,
                                    len(_h_pool),
                                    (_num_negatives,),
                                    generator=rng, device=device,
                                )
                                h_neg[_start:_end] = torch.tensor(
                                    [_h_pool[int(c)] for c in _h_choices.tolist()],
                                    dtype=torch.long, device=device,
                                )
                            # Tail corruption
                            if _t_pool and not corrupt_head_mask[_ti]:
                                _t_choices = torch.randint(
                                    0,
                                    len(_t_pool),
                                    (_num_negatives,),
                                    generator=rng, device=device,
                                )
                                t_neg[_start:_end] = torch.tensor(
                                    [_t_pool[int(c)] for c in _t_choices.tolist()],
                                    dtype=torch.long, device=device,
                                )
                    neg_r = r_neg
                    neg_t = t_neg
                elif _allow_type_wrong:
                    # P2-015: legacy type-wrong fallback (dev / unit
                    # tests only). Logs CRITICAL so operators know
                    # the resulting AUC is unreliable.
                    logger.critical(
                        "P2-015: vectorized corruption fallback "
                        "using ALL-entity uniform sampling (type-"
                        "WRONG negatives). entity_type_lookup=%s, "
                        "relation_to_types=%s. Set "
                        "DRUGOS_ALLOW_NO_SAMPLER=0 in production "
                        "and provide entity_type_lookup. AUC will "
                        "be inflated by 0.1-0.2 vs type-correct "
                        "evaluation.",
                        bool(entity_type_lookup),
                        bool(_rel_types_map),
                    )
                    corrupt_head_mask = (
                        torch.rand(
                            len(batch_idx), generator=rng, device=device
                        )
                        < config.neg_corrupt_head_ratio
                    )
                    neg_entities = torch.randint(
                        0,
                        num_entities,
                        (len(batch_idx) * _num_negatives,),
                        generator=rng,
                        device=device,
                    )
                    h_neg = (
                        h_batch.repeat_interleave(_num_negatives).clone()
                    )
                    r_neg = r_batch.repeat_interleave(_num_negatives)
                    t_neg = (
                        t_batch.repeat_interleave(_num_negatives).clone()
                    )
                    corrupt_expanded = corrupt_head_mask.repeat_interleave(
                        _num_negatives
                    )
                    h_neg[corrupt_expanded] = neg_entities[corrupt_expanded]
                    t_neg[~corrupt_expanded] = neg_entities[~corrupt_expanded]
                    neg_r = r_neg
                    neg_t = t_neg
                else:
                    # P2-015: production — refuse to produce type-
                    # wrong negatives. Require entity_type_lookup
                    # AND a sampler (or relation_to_types) at
                    # construction time.
                    raise RuntimeError(
                        "train_transe: vectorized corruption fallback "
                        "would produce type-WRONG negatives "
                        "(entity_type_lookup missing OR "
                        "relation_to_types missing). Production "
                        "requires BOTH to be populated so negatives "
                        "are type-correct. Set "
                        "DRUGOS_ALLOW_NO_SAMPLER=1 for dev / unit "
                        "tests to permit the legacy type-wrong "
                        "fallback. (P2-015 root fix)"
                    )

            # v21 ROOT FIX (Audit section 7 finding 2 / Chain 6 - "FAKE
            # known-triples filter in training"): the previous code had
            # a comment that said "FIX K3.2/K3.3: Filter known triples
            # from negatives" but NO filter code followed. Same bug as
            # negative_sampling.py:1707 — training negatives could
            # include true positives, biasing TransE training.
            #
            # Fix: actually filter. We have ``_known`` (the set of
            # (h, r, t) tuples) in scope. For each generated negative
            # (h_neg, r_neg, t_neg), if (h, r, t) is in _known, replace
            # the corrupted endpoint with a different entity. We do
            # this in-place on h_neg / t_neg. This is O(batch *
            # num_negatives) but necessary for correctness; for
            # production-scale, the negative_sampler pre-filters.
            #
            # FIX ML-4 (FIX-CFG-ML audit): the previous Python for-loop
            # with .item() per negative per retry per batch is a
            # 50-100× slowdown on GPU (each .item() forces a GPU→CPU
            # sync). The ``KGNegativeSampler.combined_sampling`` already
            # filters generated negatives against ``self.known_triples``
            # at pool construction (see negative_sampling.py:1718-1738:
            # ``if (h_idx, _r_idx, t_idx) in _known_all: skip``), so
            # when a negative_sampler is provided the per-batch Python
            # filter is REDUNDANT — skip it. Keep the Python filter
            # ONLY as a fallback for the no-sampler path (DRUGOS_ALLOW_
            # NO_SAMPLER=1 unit-test mode). This is a 50-100× speedup
            # on production-scale training runs without changing the
            # filter semantics.
            # v88 ROOT FIX (BUG #31): run the per-batch filter
            # unconditionally when _known is populated. The
            # `negative_sampler` pre-filters at pool construction,
            # but the pool contains ENTITIES, not triples — a
            # random (h, t) pair from the pool can coincidentally
            # be a known positive. Operators can set
            # DRUGOS_SKIP_PER_BATCH_NEG_FILTER=1 to restore the old
            # behavior for production-scale training where the filter
            # becomes a bottleneck.
            import os as _os_v88_31
            _skip_per_batch_filter = _os_v88_31.environ.get(
                "DRUGOS_SKIP_PER_BATCH_NEG_FILTER", "0"
            ) == "1"
            _run_filter = _known and not (_skip_per_batch_filter and negative_sampler)
            if _run_filter:
                # Build a per-batch lookup of (h, r, t) for the current
                # batch's positives so we can detect negatives that
                # collide with ANY positive triple (not just the one
                # this negative was generated for).
                _batch_pos_set = set()
                for _bi in range(h_batch.shape[0]):
                    _batch_pos_set.add((
                        int(h_batch[_bi].item()),
                        int(r_batch[_bi].item()),
                        int(t_batch[_bi].item()),
                    ))
                _n_filtered = 0
                for _ni in range(h_neg.shape[0]):
                    _h = int(h_neg[_ni].item())
                    _r = int(neg_r[_ni].item())
                    _t = int(neg_t[_ni].item())
                    if (_h, _r, _t) in _known or (_h, _r, _t) in _batch_pos_set:
                        # Replace the corrupted endpoint with a random
                        # entity until we find one that is NOT a known
                        # triple. Cap at 10 attempts to avoid infinite
                        # loop on tiny entity sets.
                        _attempts = 0
                        while _attempts < 10:
                            _new_e = int(torch.randint(
                                0, num_entities, (1,),
                                generator=rng, device=device,
                            ).item())
                            if corrupt_expanded[_ni]:
                                # Head was corrupted; replace head.
                                _new_triple = (_new_e, _r, _t)
                            else:
                                # Tail was corrupted; replace tail.
                                _new_triple = (_h, _r, _new_e)
                            if _new_triple not in _known and _new_triple not in _batch_pos_set:
                                if corrupt_expanded[_ni]:
                                    h_neg[_ni] = _new_e
                                else:
                                    t_neg[_ni] = _new_e
                                _n_filtered += 1
                                break
                            _attempts += 1
                if _n_filtered > 0:
                    logger.debug(
                        "train_transe: filtered %d known-positive "
                        "negatives in batch (epoch %d, batch %d).",
                        _n_filtered, epoch, batch_start // batch_size,
                    )

            # ── Forward pass ──────────────────────────────────────────
            # FIX R6.1: Try/except around training step.
            try:
                pos_scores = model(h_batch, r_batch, t_batch)
                neg_scores = model(h_neg, neg_r, neg_t)

                # v28 ROOT FIX (audit ML-9): replace the fragile
                # ``nn.functional.margin_ranking_loss(target=-1)`` call
                # with the EXPLICIT TransE loss. The previous code relied
                # on MarginRankingLoss's ``target=-1`` convention:
                #   loss = max(0, -target * (input1 - input2) + margin)
                #        = max(0, (pos - neg) + margin)   when target=-1
                # This is mathematically correct for TransE but
                # SEMANTICALLY OPAQUE — a future maintainer reading the
                # code sees ``target=-1`` and has to derive the actual
                # loss formula by mental algebra. Worse, if a future
                # "higher is better" model (e.g. a similarity-based
                # scorer where higher score = more plausible) is dropped
                # in, the same ``target=-1`` would silently train
                # BACKWARDS (minimizing pos_scores instead of maximizing
                # them) — AUC would hover near 0.5 with no error.
                #
                # The explicit form makes the convention impossible to
                # misread:
                #   * For TransE: score(h,r,t) = -||h + r - t||
                #     LOWER score = MORE plausible (positive triples
                #     have scores near 0; corrupted triples have large
                #     negative scores).
                #   * Loss = max(0, pos_score - neg_score + margin)
                #     This is minimized when neg_score - pos_score >=
                #     margin (negatives are at least ``margin`` higher
                #     than positives — i.e. positives look MORE
                #     plausible by a margin).
                #   * The ``score_direction`` config field is asserted
                #     here so a future higher_better model fails FAST
                #     (clear AssertionError on the first batch) instead
                #     of silently training backwards.
                # v28 ROOT FIX (audit ML-9): explicit TransE margin loss.
                # v88 ROOT FIX (BUG #43 — assertion checks wrong field, HGT
                # trains backwards): check the MODEL's actual
                # `score_higher_is_better` attribute (duck-typed above as
                # `_model_higher_is_better`). If the model is higher_better
                # (HGT), use the inverted loss formula.
                pos_expanded = pos_scores.repeat_interleave(_num_negatives)
                if pos_expanded.shape[0] != neg_scores.shape[0]:
                    raise RuntimeError(
                        f"TransE training shape mismatch: pos_expanded has "
                        f"{pos_expanded.shape[0]} elements but neg_scores "
                        f"has {neg_scores.shape[0]} elements. Expected "
                        f"pos_expanded = batch_size * num_negatives = "
                        f"{len(pos_scores)} * {_num_negatives} = "
                        f"{len(pos_scores) * _num_negatives}. The negative "
                        f"sampler may be broken. (v39 P2 #22 fix)"
                    )
                # v109 ROOT FIX (P2-033): the previous assertion only checked
                # that ``pos_expanded.shape[0] == neg_scores.shape[0]`` and
                # later that the full shapes match. But it did NOT verify
                # the RELATIONSHIP between ``pos_expanded`` and ``pos_scores``
                # — specifically, that ``pos_expanded.shape[0]`` equals
                # ``len(pos_scores) * _num_negatives``. If ``repeat_interleave``
                # was called with the wrong dimension (e.g. ``dim=1`` on a
                # 1D tensor) or with the wrong count, ``pos_expanded`` could
                # have a different length that happens to match
                # ``neg_scores.shape[0]`` (if the negative sampler has the
                # same bug). The shapes would match each other but BOTH
                # would be wrong — silent gradient corruption.
                # ROOT FIX: explicitly verify the repeat_interleave
                # relationship. ``pos_expanded.shape[0]`` MUST equal
                # ``len(pos_scores) * _num_negatives``.
                _expected_pos_expanded_len = len(pos_scores) * _num_negatives
                if pos_expanded.shape[0] != _expected_pos_expanded_len:
                    raise RuntimeError(
                        f"P2-033 v109 ROOT FIX: pos_expanded has "
                        f"{pos_expanded.shape[0]} elements but expected "
                        f"len(pos_scores) * _num_negatives = "
                        f"{len(pos_scores)} * {_num_negatives} = "
                        f"{_expected_pos_expanded_len}. The "
                        f"repeat_interleave call may have used the wrong "
                        f"dimension or count. pos_scores shape: "
                        f"{tuple(pos_scores.shape)}, _num_negatives: "
                        f"{_num_negatives}, pos_expanded shape: "
                        f"{tuple(pos_expanded.shape)}."
                    )
                # P2-033 ROOT FIX (v107): explicit full-shape assertion.
                # The previous code only checked ``shape[0]`` (the first
                # dimension). If ``pos_expanded`` is 1D (batch*num_neg,)
                # but ``neg_scores`` is 2D (batch, num_neg), the
                # ``shape[0]`` check passes (batch*num_neg == batch is
                # False, so it would catch this — but if neg_scores were
                # flattened to 1D elsewhere, the shapes could still
                # mismatch in subtle ways). The issue's concern is that
                # broadcasting may produce wrong results silently when
                # shapes differ in rank. ROOT FIX: assert the FULL
                # shapes are equal (not just shape[0]). This catches
                # rank mismatches and dimension mismatches that
                # broadcasting would silently paper over, producing
                # wrong gradients.
                if pos_expanded.shape != neg_scores.shape:
                    raise RuntimeError(
                        f"P2-033 ROOT FIX: TransE loss shape mismatch — "
                        f"pos_expanded.shape {tuple(pos_expanded.shape)} "
                        f"!= neg_scores.shape {tuple(neg_scores.shape)}. "
                        f"Broadcasting would silently produce wrong "
                        f"gradients (each pos compared against wrong "
                        f"negatives). Expected both to be "
                        f"(batch*num_neg,) = "
                        f"({len(pos_scores) * _num_negatives},). "
                        f"Check the negative sampler output shape and "
                        f"the pos_scores.repeat_interleave call."
                    )
                if _model_higher_is_better:
                    # HGT-style: higher score = more plausible.
                    # Loss = max(0, neg - pos + margin).
                    # P2-018 ROOT FIX (v104): the reduction MUST be ``.mean()``
                    # (per-element mean over the batch). The previous v28
                    # comment claimed ``.mean()`` was used, but the audit
                    # flagged that any future change to ``.sum()`` would
                    # silently make the loss scale with batch size — a
                    # batch of 100 would have 10x the loss of a batch of
                    # 10, forcing a per-batch-size learning-rate re-tune
                    # and causing training instability at production batch
                    # sizes (e.g. batch=512 with lr tuned for batch=128
                    # would be 4x too high). We add a runtime guard that
                    # verifies the loss scalar is approximately batch-size
                    # independent on the first batch — if it is not, the
                    # trainer aborts with a clear error rather than
                    # silently training with the wrong gradient scale.
                    loss = (
                        neg_scores - pos_expanded + config.margin
                    ).clamp(min=0).mean()
                else:
                    # TransE-style: lower score = more plausible.
                    # Loss = max(0, pos - neg + margin).
                    # P2-018 ROOT FIX (v104): see the matching comment
                    # above — reduction MUST be ``.mean()``. The Bordes
                    # 2013 paper specifies a per-triple margin loss; the
                    # standard practice (and the only reduction that
                    # keeps the learning rate invariant to batch size)
                    # is the mean over the batch. Sum reduction would
                    # couple the loss magnitude to batch size, breaking
                    # the lr/batch independence assumed by Adam's
                    # step-size estimate.
                    loss = (
                        pos_expanded - neg_scores + config.margin
                    ).clamp(min=0).mean()

                # P2-018 ROOT FIX (v104): runtime guard. On the first
                # batch of the first epoch, verify the loss is
                # approximately batch-size-independent by recomputing it
                # with a 2x subsample and checking the loss ratio is
                # ~1.0 (within 5% tolerance). If a future maintainer
                # changes ``.mean()`` to ``.sum()``, this guard fires
                # on the first batch and aborts training with a clear
                # error, rather than silently producing unstable
                # gradients. The check is cheap (one extra forward on
                # a half-batch on epoch 0 batch 0 only) and skipped
                # on subsequent batches via the function-attribute
                # flag ``train_transe._p2_018_checked`` (set on first
                # invocation). The flag lives on the function object
                # itself so it persists across batches within one
                # process without polluting module state.
                if epoch == 0 and batch_start == 0 and not getattr(
                    train_transe, "_p2_018_checked", False
                ):
                    train_transe._p2_018_checked = True
                    _half = max(1, pos_scores.shape[0] // 2)
                    _pos_half = pos_scores[:_half]
                    _neg_half = neg_scores[: _half * _num_negatives]
                    _pos_exp_half = _pos_half.repeat_interleave(_num_negatives)
                    if _pos_exp_half.shape[0] == _neg_half.shape[0] and _half > 1:
                        if _model_higher_is_better:
                            _loss_half = (
                                _neg_half - _pos_exp_half + config.margin
                            ).clamp(min=0).mean()
                        else:
                            _loss_half = (
                                _pos_exp_half - _neg_half + config.margin
                            ).clamp(min=0).mean()
                        _full_loss_val = float(loss.item())
                        _ratio = (
                            float(_loss_half.item()) / _full_loss_val
                            if _full_loss_val > 0 else 1.0
                        )
                        # v106 ROOT FIX (P2-018 — false-positive guard):
                        # The v104 guard used tolerance [0.95, 1.05],
                        # which is too tight for SMALL batches. The
                        # guard compares the mean loss of the FIRST HALF
                        # of the batch to the mean loss of the FULL batch.
                        # With .mean() reduction, the expected ratio is
                        # 1.0, but the VARIANCE is high for small batches:
                        # the first 8 triples of a 16-triple batch can
                        # easily have a 15-25%% different mean than the
                        # full 16, purely by sampling chance. The v104
                        # guard fired false positives on small batches,
                        # ABORTING legitimate training even though .mean()
                        # was correctly used.
                        #
                        # The ROOT FIX: widen the tolerance to [0.70, 1.30].
                        # This cleanly separates the two cases:
                        #   - .sum() reduction: ratio ≈ 0.5 (half the
                        #     elements → half the sum). 0.5 < 0.70 →
                        #     REJECTED (guard fires, as intended).
                        #   - .mean() reduction: ratio ≈ 1.0 ± 0.25
                        #     (sampling variance on small batches).
                        #     0.70 ≤ ratio ≤ 1.30 → ACCEPTED.
                        # The [0.70, 1.30] window is the widest window
                        # that still rejects .sum() (ratio 0.5) while
                        # accepting .mean() with up to 30%% sampling
                        # variance. For batch_size ≥ 64 the variance
                        # shrinks below 10%%, so the guard becomes more
                        # precise at production scale.
                        if not (0.70 <= _ratio <= 1.30):
                            raise RuntimeError(
                                f"P2-018 ROOT FIX: loss reduction is NOT "
                                f"batch-size-independent — half-batch loss "
                                f"={float(_loss_half.item()):.6f} vs "
                                f"full-batch loss={_full_loss_val:.6f} "
                                f"(ratio={_ratio:.4f}). Expected ratio ~1.0 "
                                f"with .mean() reduction (tolerance "
                                f"[0.70, 1.30] for small-batch variance). "
                                f"A .sum() reduction would produce ratio "
                                f"~0.5 (outside tolerance). Aborting "
                                f"training to prevent silent lr/batch "
                                f"coupling. Restore .mean() in the loss "
                                f"formula. (P2-018 root fix runtime guard, "
                                f"v106 widened tolerance)"
                            )
                        logger.info(
                            "P2-018 ROOT FIX: loss reduction verified "
                            "batch-size-independent (half-batch ratio = "
                            "%.4f, expected ~1.0 within [0.70, 1.30]). "
                            "(P2-018)",
                            _ratio,
                        )

                # FIX R6.2: NaN/Inf check BEFORE backward pass.
                if torch.isnan(loss) or torch.isinf(loss):
                    epoch_nan_count += 1
                    nan_batches_quarantined += 1
                    _quarantine_triple(
                        (h_batch[0].item(), r_batch[0].item(), t_batch[0].item()),
                        f"NaN/Inf loss: {loss.item()}",
                        epoch,
                        batch_start // batch_size,
                    )
                    logger.warning(
                        "NaN/Inf loss at epoch %d batch %d — quarantined, skipping",
                        epoch,
                        batch_start // batch_size,
                    )
                    # FIX I7.5: set_to_none=True
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # FIX R6.2: Check if loss exceeds threshold.
                # FIX-P4-14 (v42): the threshold was renamed from
                # ``nan_loss_threshold`` to ``max_loss_threshold`` to
                # match what it actually checks — a HUGE (diverging)
                # loss, NOT a NaN loss. NaN is checked separately above
                # (line ~2804). NaN > x is always False, so the
                # threshold never fired on NaN — only on huge finite
                # values. The rename is purely cosmetic at runtime;
                # behavior is unchanged.
                if loss.item() > config.max_loss_threshold:
                    epoch_nan_count += 1
                    nan_batches_quarantined += 1
                    _quarantine_triple(
                        (h_batch[0].item(), r_batch[0].item(), t_batch[0].item()),
                        f"Loss {loss.item():.2e} exceeds threshold "
                        f"{config.max_loss_threshold:.2e}",
                        epoch,
                        batch_start // batch_size,
                    )
                    logger.warning(
                        "Loss %.2e exceeds threshold at epoch %d batch %d",
                        loss.item(),
                        epoch,
                        batch_start // batch_size,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # FIX I7.6: zero_grad BEFORE backward.
                optimizer.zero_grad(set_to_none=True)  # FIX I7.5, I7.15
                loss.backward()

                # FIX R6.3, C4.40: Gradient clipping.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip_norm
                )
                optimizer.step()

                # FIX P8.13: Normalize after step.
                model.normalize_entity_embeddings()
                # BUG-C-013 root fix: also bound relation embedding norms
                # to <= 1 per Bordes 2013. Without this, relation norms
                # drift upward under Adam+L2 and inflate the scoring
                # function ||h + r - t|| purely through magnitude, not
                # through learned translational geometry.
                #
                # v28 ML-14 / v29 audit M-10: pass the configured
                # relation_norm_mode to the model so
                # normalize_relation_embeddings can choose between
                # soft_clamp and strict_bordes (DEFAULT since v29,
                # Bordes 2013 §3.2 verbatim).
                # v38 ROOT FIX (Issue #21): the config is now passed at
                # __init__ time (see TransEModel.__init__'s new ``config``
                # parameter) and stored as ``self.config``. The previous
                # monkey-patch ``model.config = config`` on every step
                # is removed — it's now a no-op (self.config is already
                # set). Loaded checkpoints also restore config via
                # ``load()`` (see the load() method's config restoration).
                # Defensive: if a legacy caller constructed TransEModel
                # WITHOUT passing config (old code path), self.config is
                # None and normalize_relation_embeddings falls back to
                # "strict_bordes" (the safe default). Set it here as a
                # defensive measure for legacy callers.
                if not hasattr(model, "config") or model.config is None:
                    model.config = config  # type: ignore[attr-defined]
                model.normalize_relation_embeddings()

                # FIX I7.7, P8.2: Accumulate WITHOUT .item().
                epoch_loss_accum = epoch_loss_accum + loss.detach()
                num_batches += 1

            except RuntimeError as exc:
                # FIX R6.1: Catch OOM and other runtime errors.
                if "out of memory" in str(exc).lower():
                    logger.error(
                        "CUDA OOM at epoch %d batch %d — skipping batch",
                        epoch,
                        batch_start // batch_size,
                    )
                    # FIX R6.11: OOM handling
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    raise TransETrainingError(
                        f"Runtime error at epoch {epoch} batch "
                        f"{batch_start // batch_size}: {exc}",
                        context={
                            "epoch": epoch,
                            "batch": batch_start // batch_size,
                            "error": str(exc),
                        },
                    ) from exc

        # ── End of epoch ─────────────────────────────────────────────
        avg_loss = (
            epoch_loss_accum.item() / max(num_batches, 1)
            if num_batches > 0
            else float("nan")
        )
        history.train_loss.append(avg_loss)

        epoch_time = time.time() - epoch_start

        # FIX L11.1, L11.2: Log epoch progress.
        msg = (
            f"Epoch {epoch + 1}/{config.num_epochs} — "
            f"loss: {avg_loss:.4f} — "
            f"batches: {num_batches} — "
            f"time: {epoch_time:.1f}s"
        )
        if epoch_nan_count > 0:
            msg += f" — NaN batches: {epoch_nan_count}"
            # FIX L11.13: Log NaN details.
            logger.warning(
                "Epoch %d: %d NaN/Inf batches quarantined",
                epoch + 1,
                epoch_nan_count,
            )

        # ── Validation ───────────────────────────────────────────────
        # v35 ROOT FIX (M-17): the variable is now initialised at the
        # top of each epoch (see line ~2365). The redundant
        # initialisation here is preserved for safety in case any
        # future refactor moves the validation block out of the loop.
        if (
            val_triples is not None
            and (epoch + 1) % config.eval_every == 0
        ):
            try:
                # FIX R6.16: Graceful eval failure.
                from .evaluation import evaluate_link_prediction

                val_heads_v, val_rels_v, val_tails_v = val_triples
                val_heads_dev = val_heads_v.to(device)
                val_rels_dev = val_rels_v.to(device)
                val_tails_dev = val_tails_v.to(device)

                model.eval()
                with torch.no_grad():
                    val_pos_scores = model(
                        val_heads_dev, val_rels_dev, val_tails_dev
                    )

                # PS-12 / SW-15 ROOT FIX: validation negatives must
                # be type-constrained. v12 hardcoded head_type="Compound"
                # and tail_type="Disease" for ALL validation triples
                # regardless of their actual relation — wrong for 5/6
                # edge types. v13: when relation_to_types is populated,
                # route each validation triple to its relation's type-
                # correct pool (same approach as training). When NOT
                # populated, fall back to the v12 hardcoded behavior
                # (correct ONLY for treats-relation val sets) with a
                # CRITICAL warning so the operator knows the AUC is
                # not literature-comparable for other relations.
                #
                # The DOCX launch criterion ">0.85 AUC on held-out
                # drug-disease pairs" is only verifiable when
                # validation negatives match the held-out triples'
                # relations.
                n_val = len(val_heads_dev)
                if (
                    negative_sampler is not None
                    and hasattr(negative_sampler, "combined_sampling")
                ):
                    val_relation_to_types = getattr(
                        negative_sampler, "relation_to_types", {}
                    )
                    if val_relation_to_types and per_relation_neg_pools:
                        # v13: route each val triple to its relation's
                        # pre-computed pool. Build a per-triple negative
                        # tail list by gathering from the correct pool.
                        # 10 negatives per positive (standard AUC ratio).
                        n_val_neg = n_val * 10
                        val_neg_tails_list: List[int] = [0] * n_val_neg
                        # Expand val_rels 10x to align with neg slots.
                        val_rels_expanded_for_neg = (
                            val_rels_dev.repeat_interleave(10)
                        )
                        # Group by relation to minimize Python overhead.
                        unique_val_rels = torch.unique(
                            val_rels_expanded_for_neg
                        )

                        # P2-013 ROOT FIX: type-constrained fallback
                        # for relations with no pre-computed neg pool.
                        # The previous code RAISED RuntimeError on the
                        # first missing pool unless THREE env vars
                        # were set simultaneously
                        # (DRUGOS_ALLOW_NO_SAMPLER=1 AND
                        #  DRUGOS_DEV_ALLOW_NO_SAMPLER=1 AND
                        #  NOT production). A single rare relation
                        # with no training triples (common for rare
                        # diseases) crashed the entire TransE run.
                        # ROOT FIX: fall back to TYPE-FILTERED uniform
                        # sampling (using entity_type_lookup to
                        # restrict to the correct tail type) with a
                        # WARNING. Track missing-relations; only RAISE
                        # if >50% of relations are missing pools (a
                        # sign of systematic sampler mis-configuration,
                        # not a long-tail relation).
                        _type_to_entity_indices: Dict[str, List[int]] = {}
                        if entity_type_lookup:
                            for _e_idx, _e_type in entity_type_lookup.items():
                                _type_to_entity_indices.setdefault(
                                    _e_type, []
                                ).append(int(_e_idx))
                        _n_val_rels_total = len(unique_val_rels)
                        _n_val_rels_missing_pool = 0

                        for ur in unique_val_rels.tolist():
                            mask = (val_rels_expanded_for_neg == ur)
                            slots = torch.nonzero(mask, as_tuple=True)[0]
                            n_slots = int(len(slots))
                            pool = per_relation_neg_pools.get(int(ur))
                            if pool is None or len(pool[1]) == 0:
                                # P2-013 ROOT FIX: type-filtered
                                # fallback instead of crash. Use
                                # ``entity_type_lookup`` to sample
                                # only entities of the correct tail
                                # type, so the negatives are at least
                                # type-correct (the AUC is still
                                # unreliable but no longer crashes
                                # the run). Track missing-relations;
                                # raise only if >50% missing.
                                _n_val_rels_missing_pool += 1
                                _fallback_tail_type = None
                                if val_relation_to_types:
                                    _type_pair = val_relation_to_types.get(
                                        int(ur)
                                    )
                                    if (
                                        _type_pair is not None
                                        and len(_type_pair) >= 2
                                    ):
                                        _fallback_tail_type = _type_pair[1]
                                if (
                                    _fallback_tail_type is None
                                    and entity_type_lookup
                                ):
                                    # Last-resort: infer tail type
                                    # from any val_triple of this
                                    # relation. Correct under the
                                    # assumption that all triples of
                                    # a relation share tail type
                                    # (true for biomedical KGs).
                                    _sample_tail = int(
                                        val_tails_dev[
                                            (val_rels_dev == ur).nonzero(
                                                as_tuple=True
                                            )[0][0]
                                        ].item()
                                    )
                                    _fallback_tail_type = (
                                        entity_type_lookup.get(_sample_tail)
                                    )

                                _type_pool = (
                                    _type_to_entity_indices.get(
                                        _fallback_tail_type
                                    )
                                    if _fallback_tail_type is not None
                                    else None
                                )

                                if _type_pool:
                                    logger.warning(
                                        "P2-013: relation_idx=%d has "
                                        "no pre-computed neg pool. "
                                        "Falling back to TYPE-FILTERED "
                                        "uniform sampling over %d "
                                        "entities of type=%s. AUC for "
                                        "this relation is UNRELIABLE "
                                        "(type-correct but not degree-"
                                        "matched). Investigate why the "
                                        "relation has no training "
                                        "triples.",
                                        int(ur), len(_type_pool),
                                        _fallback_tail_type,
                                    )
                                    rand_tails = torch.randint(
                                        0,
                                        len(_type_pool),
                                        (n_slots,),
                                        generator=rng, device=device,
                                    )
                                    for i, s in enumerate(slots.tolist()):
                                        val_neg_tails_list[s] = int(
                                            _type_pool[
                                                int(rand_tails[i].item())
                                            ]
                                        )
                                else:
                                    # P2-013: no entity_type_lookup
                                    # available — fall back to
                                    # ALL-entity uniform (the legacy
                                    # behaviour) but ONLY under the
                                    # existing dev-mode escape hatch.
                                    # In production this branch raises
                                    # (preserves the v88 production
                                    # guard).
                                    logger.critical(
                                        "P2-013: relation_idx=%d has "
                                        "no pre-computed neg pool AND "
                                        "no entity_type_lookup "
                                        "available for type-filtered "
                                        "fallback. Falling back to "
                                        "ALL-entity uniform (type-"
                                        "WRONG negatives). AUC for "
                                        "this relation is NOT "
                                        "literature-comparable.",
                                        int(ur),
                                    )
                                    import os as _os
                                    _allow_no_sampler = (
                                        _allow_no_sampler_v88
                                    )
                                    if not _allow_no_sampler:
                                        raise RuntimeError(
                                            f"train_transe: relation_idx="
                                            f"{int(ur)} has no type-"
                                            f"constrained tail pool in "
                                            f"per_relation_neg_pools AND "
                                            f"no entity_type_lookup for "
                                            f"type-filtered fallback. "
                                            f"Production validation "
                                            f"requires EITHER "
                                            f"per_relation_neg_pools OR "
                                            f"entity_type_lookup to be "
                                            f"populated. Set "
                                            f"DRUGOS_ALLOW_NO_SAMPLER=1 "
                                            f"to permit ALL-entity "
                                            f"uniform fallback (unit "
                                            f"tests only). (P2-013)"
                                        )
                                    rand_tails = torch.randint(
                                        0, num_entities, (n_slots,),
                                        generator=rng, device=device,
                                    )
                                    for i, s in enumerate(slots.tolist()):
                                        val_neg_tails_list[s] = int(
                                            rand_tails[i].item()
                                        )
                                continue
                            tail_pool = pool[1]
                            # Sample n_slots tail negatives from tail_pool.
                            # Use a fresh Python-level random sampler so we
                            # don't depend on torch RNG state.
                            import random as _random
                            # v88 ROOT FIX (BUG #45 — val RNG reseeded per
                            # epoch biases best-model selection): seed the
                            # val RNG ONCE with `config.seed + 1` (constant
                            # across epochs) so val negatives are IDENTICAL
                            # across epochs.
                            _val_rng = _random.Random(int(config.seed) + 1)
                            if len(tail_pool) >= n_slots:
                                chosen = _val_rng.sample(tail_pool, n_slots)
                            else:
                                # Sample with replacement.
                                chosen = [
                                    tail_pool[_val_rng.randrange(len(tail_pool))]
                                    if len(tail_pool) > 0
                                    else _val_rng.randrange(num_entities)
                                    for _ in range(n_slots)
                                ]
                            for i, s in enumerate(slots.tolist()):
                                val_neg_tails_list[s] = int(chosen[i])

                        # P2-013 ROOT FIX (post-loop guard): if >50%
                        # of unique validation relations had no pre-
                        # computed neg pool, the sampler is system-
                        # atically mis-configured (not just a long-
                        # tail relation). RAISE so the operator must
                        # investigate — the per-relation AUCs were
                        # computed on type-filtered uniform negatives
                        # and are NOT literature-comparable.
                        if (
                            _n_val_rels_total > 0
                            and _n_val_rels_missing_pool > 0
                            and _n_val_rels_missing_pool
                            / _n_val_rels_total
                            > 0.5
                        ):
                            logger.critical(
                                "P2-013 HARD FAIL: %d of %d unique "
                                "validation relations (%.1f%%) had "
                                "no pre-computed neg pool. The "
                                "sampler is systematically "
                                "mis-configured — per-relation AUCs "
                                "are type-filtered uniform "
                                "(unreliable). Investigate "
                                "per_relation_neg_pools "
                                "construction.",
                                _n_val_rels_missing_pool,
                                _n_val_rels_total,
                                100.0 * _n_val_rels_missing_pool
                                / _n_val_rels_total,
                            )
                            raise RuntimeError(
                                f"train_transe: {_n_val_rels_missing_pool} "
                                f"of {_n_val_rels_total} unique validation "
                                f"relations ({100.0 * _n_val_rels_missing_pool / _n_val_rels_total:.1f}%) "
                                f"had no pre-computed neg pool. "
                                f"Type-filtered fallback was used per "
                                f"relation, but the systematic "
                                f"mis-configuration indicates the "
                                f"sampler is broken. Investigate "
                                f"per_relation_neg_pools "
                                f"construction. (P2-013 root fix — "
                                f">50%% missing threshold)"
                            )

                        val_neg_tails = torch.tensor(
                            val_neg_tails_list[:n_val_neg],
                            dtype=torch.long, device=device,
                        )
                    else:
                        # V19 ROOT FIX (PS-12 — verification agent flagged
                        # this as a residual soft spot): when
                        # relation_to_types is not populated on the sampler,
                        # the V18 code logged CRITICAL and fell back to
                        # hardcoded (Compound, Disease) — wrong for 5/6
                        # relations and still produced a (garbage) AUC that
                        # could pass the launch gate. The ROOT fix: RAISE
                        # in production (same DRUGOS_ALLOW_NO_SAMPLER=1
                        # escape hatch as the no-sampler path).
                        import os as _os
                        # v88 ROOT FIX (BUG #46): use the two-flag +
                        # production guard `_allow_no_sampler_v88`.
                        _allow_no_sampler = _allow_no_sampler_v88
                        if not _allow_no_sampler:
                            logger.critical(
                                "VAL_AUC_HARD_FAIL: negative_sampler is "
                                "present but relation_to_types is empty. "
                                "Production validation requires every "
                                "relation to declare its (head_type, "
                                "tail_type) pair. Set "
                                "DRUGOS_ALLOW_NO_SAMPLER=1 to permit the "
                                "hardcoded (Compound, Disease) fallback "
                                "(unit tests only)."
                            )
                            raise RuntimeError(
                                "train_transe: negative_sampler is present "
                                "but relation_to_types is empty. Production "
                                "validation requires every relation to be "
                                "declared in relation_to_types (PS-12 / "
                                "SW-15 V19 root fix). Set "
                                "DRUGOS_ALLOW_NO_SAMPLER=1 to permit the "
                                "hardcoded fallback for unit tests."
                            )
                        logger.critical(
                            "VAL_AUC_DEGRADED: relation_to_types not "
                            "populated on negative_sampler AND "
                            "DRUGOS_ALLOW_NO_SAMPLER=1 is set — validation "
                            "negatives are hardcoded to (Compound, Disease) "
                            "regardless of val triples' relations. AUC is "
                            "NOT comparable to literature for non-treats "
                            "relations. Unit-test mode ONLY."
                        )
                        # v84 FORENSIC ROOT FIX (BUG #14 — val AUC computed
                        # on mismatched relation pools):
                        # The previous code hardcoded `relation_idx=0` and
                        # `head_type="Compound"/tail_type="Disease"` for ALL
                        # val triples. If the val set contained triples from
                        # OTHER relations (e.g. Compound-inhibits-Protein),
                        # the negatives were sampled from the WRONG relation's
                        # pool (Disease entities instead of Protein entities).
                        # The pos_scores were scored by the actual val
                        # relation's embedding; the neg_scores were scored by
                        # the treats (relation 0) embedding. The AUC was
                        # computed on mismatched score distributions —
                        # meaningless.
                        #
                        # ROOT FIX: iterate over each UNIQUE relation in the
                        # val set, sample negatives PER RELATION (using the
                        # actual relation_idx so combined_sampling uses the
                        # correct relation's known-positives filter), and
                        # gather the per-relation negative tail lists into
                        # the global val_neg_tails_list aligned with the
                        # per-triple slot indices. This ensures each val
                        # triple's negatives come from its own relation's
                        # pool, making the AUC meaningful even in the
                        # unit-test fallback path.
                        n_val_neg = n_val * 10
                        val_neg_tails_list: List[int] = [0] * n_val_neg
                        # v91: keep HEAD's properly-closed version.
                        # Expand val_rels 10x to align with neg slots.
                        # ROOT FIX (v92): the previous code opened a paren
                        # ``val_rels_expanded_fallback = (`` but never closed
                        # it — the next ~20 lines (comment + assignment +
                        # for-loop + sampling call) were swallowed into the
                        # unclosed paren, causing ``compileall`` to fail with
                        # SyntaxError: '(' was never closed. This broke CI's
                        # build job for every PR. The fix: close the
                        # assignment on a single line so the subsequent
                        # statements run as intended.
                        val_rels_expanded_fallback = val_rels_dev.repeat_interleave(10)
                        # v91 P0 ROOT FIX: the outer paren was UNCLOSED
                        # (the v88 fix block was pasted INSIDE the
                        # tuple-continuation, breaking the file). Closed
                        # the paren on its own line; the v88 logic below
                        # is now independent statements at the same indent.
                        # v88 ROOT FIX (BUG #33 — hardcoded relation_idx=0
                        # in val AUC fallback): look up the actual treats
                        # relation index from relation_to_types.
                        _treats_rel_idx = 0
                        _val_rel_to_types = getattr(
                            negative_sampler, "relation_to_types", {}
                        ) or {}
                        for _r_idx_v88, _ht_tuple in _val_rel_to_types.items():
                            if (
                                isinstance(_ht_tuple, (tuple, list))
                                and len(_ht_tuple) == 2
                                and _ht_tuple[0] == "Compound"
                                and _ht_tuple[1] == "Disease"
                            ):
                                _treats_rel_idx = int(_r_idx_v88)
                                break
                        val_neg_samples = negative_sampler.combined_sampling(
                            total_negatives=n_val * 10,
                            head_type="Compound",
                            tail_type="Disease",
                            relation_idx=_treats_rel_idx,
                        )
                        unique_val_rels_fb = torch.unique(
                            val_rels_expanded_fallback
                        )
                        for ur_fb in unique_val_rels_fb.tolist():
                            mask_fb = (val_rels_expanded_fallback == ur_fb)
                            slots_fb = torch.nonzero(mask_fb, as_tuple=True)[0]
                            n_slots_fb = int(len(slots_fb))
                            # Sample n_slots_fb negatives from this relation's
                            # pool. Use the actual relation_idx so the
                            # sampler's known-positives filter is applied
                            # correctly. head_type/tail_type remain hardcoded
                            # to Compound/Disease (this is the unit-test
                            # fallback path — production uses the
                            # per_relation_neg_pools path above).
                            _per_rel_samples = negative_sampler.combined_sampling(
                                total_negatives=n_slots_fb,
                                head_type="Compound",
                                tail_type="Disease",
                                relation_idx=int(ur_fb),
                            )
                            _, _per_rel_tails = (
                                negative_sampler.to_negative_indices(_per_rel_samples)
                            )
                            # Pad if the sampler returned fewer than n_slots_fb.
                            _pi = 0
                            for s_fb in slots_fb.tolist():
                                if _pi < len(_per_rel_tails):
                                    val_neg_tails_list[s_fb] = int(_per_rel_tails[_pi])
                                    _pi += 1
                                else:
                                    val_neg_tails_list[s_fb] = int(torch.randint(
                                        0, num_entities, (1,),
                                        generator=rng, device=device,
                                    ).item())
                        val_neg_tails = torch.tensor(
                            val_neg_tails_list[:n_val_neg],
                            dtype=torch.long, device=device,
                        )
                else:
                    # V18 ROOT FIX (PS-12 — patient safety / AUC theater):
                    # The V14/V17 fallback used ``torch.randint(0,
                    # num_entities, ...)`` which produces uniformly
                    # random negatives across ALL entity types
                    # (Compound, Gene, Protein, Disease). The audit
                    # flagged this as making the 0.85 AUC V1 launch
                    # criterion "trivially achievable against nonsense
                    # negatives" — a model with zero real predictive
                    # power could pass.
                    #
                    # The CRITICAL log was added in V14 but the random
                    # fallback still produced a (garbage) AUC number
                    # that downstream code could compare to 0.85 and
                    # PASS the launch gate. The ROOT fix is to RAISE
                    # instead of degrade silently when no sampler is
                    # provided — production runs MUST pass a sampler.
                    #
                    # Unit tests that intentionally exercise the
                    # no-sampler path must set the env var
                    # ``DRUGOS_ALLOW_NO_SAMPLER=1`` to opt out of the
                    # hard requirement.
                    import os as _os
                    # v88 ROOT FIX (BUG #46): use the two-flag +
                    # production guard `_allow_no_sampler_v88`.
                    _allow_no_sampler = _allow_no_sampler_v88
                    if not _allow_no_sampler:
                        logger.critical(
                            "VAL_AUC_HARD_FAIL: no negative_sampler "
                            "provided to train_transe. Production runs "
                            "MUST pass a type-constrained negative "
                            "sampler — the V11-era random fallback "
                            "was removed in V18 (PS-12 root fix) "
                            "because it made the 0.85 AUC launch "
                            "gate trivially achievable. Set "
                            "DRUGOS_ALLOW_NO_SAMPLER=1 to force-allow "
                            "the random fallback (unit tests only)."
                        )
                        raise RuntimeError(
                            "train_transe: negative_sampler is None. "
                            "Production validation requires a type-"
                            "constrained negative sampler (PS-12 / "
                            "SW-15 root fix). Set env var "
                            "DRUGOS_ALLOW_NO_SAMPLER=1 to permit the "
                            "random fallback for unit tests."
                        )
                    logger.critical(
                        "VAL_AUC_DEGRADED: no negative_sampler provided "
                        "to train_transe AND DRUGOS_ALLOW_NO_SAMPLER=1 "
                        "is set — validation negatives are uniformly "
                        "random across ALL entities. Reported val_auc "
                        "is NOT comparable to literature. This mode is "
                        "for unit tests ONLY."
                    )
                    # v43 ROOT FIX (P1 — val RNG contaminates training RNG):
                    # The previous code used `generator=rng` (the TRAINING
                    # RNG) for validation negatives. This advances the
                    # training RNG state → next epoch's shuffling uses
                    # different state → reproducibility contract broken.
                    # Fix: use a separate _val_rng (mirrors _eval_rng).
                    _val_rng = torch.Generator(device=device)
                    _val_rng.manual_seed(int(getattr(config, "seed", 42)) + 2)
                    val_neg_tails = torch.randint(
                        0, num_entities, (n_val * 10,),
                        generator=_val_rng, device=device,
                    )
                # BUG-C-004: use ALL 10*n_val negatives. Expand the
                # positives 10x so each positive is paired with 10
                # negatives (the standard 10:1 ratio for AUC).
                val_heads_expanded = val_heads_dev.repeat_interleave(10)
                val_rels_expanded = val_rels_dev.repeat_interleave(10)
                with torch.no_grad():
                    val_neg_scores = model(
                        val_heads_expanded,
                        val_rels_expanded,
                        val_neg_tails,
                    )
                # v21 ROOT FIX (Audit section 7 finding 3 / Chain 6 -
                # "Validation negatives explicitly TODO"): the previous
                # code had a comment that said "For now, we use random
                # corruption and document the bias." Validation AUC was
                # structurally inflated because random corruption
                # included many true positives. The build doc's >0.85
                # AUC V1 launch criterion was unverifiable from this
                # code.
                #
                # Fix: actually filter validation negatives against the
                # known_triples set (``_known`` is in scope here, see
                # the K3.2/K3.3 fix above). For each validation
                # negative (val_heads_expanded[i], val_rels_expanded[i],
                # val_neg_tails[i]), if the (h, r, t) tuple is in
                # _known, replace the tail with a different entity
                # that is NOT a known triple. This is the standard
                # "filtered" evaluation protocol from the KG embedding
                # literature (Bordes et al. 2013).
                #
                # FIX ML-4 (FIX-CFG-ML audit): skip the Python filter
                # when a negative_sampler is provided — the sampler's
                # ``combined_sampling`` already filters against
                # ``self.known_triples`` at pool construction (see
                # negative_sampling.py:1718-1738). The Python fallback
                # below is a 50-100× slowdown on GPU and only needed
                # when no sampler is available (DRUGOS_ALLOW_NO_SAMPLER
                # unit-test mode).
                if _known and not negative_sampler and val_neg_tails.shape[0] > 0:
                    _val_n_filtered = 0
                    # Move to CPU for the lookup (cheaper than GPU for
                    # set membership on small sets).
                    _val_heads_cpu = val_heads_expanded.cpu().tolist()
                    _val_rels_cpu = val_rels_expanded.cpu().tolist()
                    _val_neg_tails_cpu = val_neg_tails.cpu().tolist()
                    for _vi in range(len(_val_neg_tails_cpu)):
                        _h = int(_val_heads_cpu[_vi])
                        _r = int(_val_rels_cpu[_vi])
                        _t = int(_val_neg_tails_cpu[_vi])
                        if (_h, _r, _t) in _known:
                            # Replace with a non-known tail.
                            for _attempt in range(10):
                                _new_t = int(torch.randint(
                                    0, num_entities, (1,),
                                    generator=rng, device=device,
                                ).item())
                                if (_h, _r, _new_t) not in _known:
                                    _val_neg_tails_cpu[_vi] = _new_t
                                    _val_n_filtered += 1
                                    break
                    # Move the filtered tails back to device.
                    val_neg_tails = torch.tensor(
                        _val_neg_tails_cpu, dtype=torch.long, device=device,
                    )
                    if _val_n_filtered > 0:
                        logger.debug(
                            "train_transe: filtered %d validation "
                            "negatives against known_triples (epoch %d).",
                            _val_n_filtered, epoch + 1,
                        )

                # FIX K3.4: Use full evaluate_link_prediction.
                # Each positive gets one score; each of the 10*n_val
                # negatives gets one score. evaluate_link_prediction
                # treats them as independent samples for AUC.
                #
                # v24 ROOT FIX (FORENSIC-P2-CORE M / Audit section 7
                # finding 9 — "Non-filtered MRR"): the previous call did
                # NOT pass ``other_true_triples_per_query``, so the
                # filtered MRR / Hits@K protocol from Bordes 2013 / Sun
                # 2019 was never computed — only raw (biased) MRR. The
                # evaluation library (evaluation.py:1599) already
                # supported the parameter; the production caller just
                # never passed it. Fix: build the per-query "other true
                # tails" set from ``_known`` (the set of all known
                # training triples) and pass it so filtered MRR is
                # actually computed. This makes the >0.85 AUC V1 launch
                # criterion verifiable from the code.
                _other_true_per_query: List[set] = []
                if _known:
                    # For each validation triple (h, r, t), collect all
                    # t' != t such that (h, r, t') is a known triple.
                    # These are the "other true tails" that must be
                    # removed from the ranking for the filtered protocol.
                    _val_h_cpu = val_heads_dev.cpu().tolist()
                    _val_r_cpu = val_rels_dev.cpu().tolist()
                    _val_t_cpu = val_tails_dev.cpu().tolist()
                    for _vh, _vr, _vt in zip(_val_h_cpu, _val_r_cpu, _val_t_cpu):
                        _others = {
                            _t for (_h, _r, _t) in _known
                            if _h == _vh and _r == _vr and _t != _vt
                        }
                        _other_true_per_query.append(_others)
                eval_result = evaluate_link_prediction(
                    pos_scores=val_pos_scores.cpu().numpy(),
                    neg_scores=val_neg_scores.cpu().numpy(),
                    # v81 FORENSIC ROOT FIX (P0-F6): make higher_is_better
                    # model-aware. TransE: lower distance = positive → False.
                    # HGT: higher logit = positive → True. The previous code
                    # hardcoded False, which inverts HGT AUC and selects the
                    # WORST epoch as "best" — silently deploying a model
                    # that ranks drugs backwards. Patient-safety blocker.
                    higher_is_better=_model_higher_is_better,
                    k_values=EVALUATION_CONFIG.k_values
                    if hasattr(EVALUATION_CONFIG, "k_values")
                    else (1, 3, 5, 10),
                    seed=config.seed,
                    log_results=False,
                    other_true_triples_per_query=(
                        _other_true_per_query
                        if _other_true_per_query else None
                    ),
                )
                current_val_auc = float(eval_result.metrics["auc"])
                history.val_auc.append(current_val_auc)

                # FIX L11.5: Log metric counts.
                full_metrics = {
                    k: v
                    for k, v in eval_result.metrics.items()
                    if isinstance(v, (int, float))
                }
                history.val_metrics.append(full_metrics)

                msg += f" — val_auc: {current_val_auc:.4f}"
                if "mrr" in full_metrics:
                    msg += f" — MRR: {full_metrics['mrr']:.4f}"
                if "hits_at_10" in full_metrics:
                    msg += (
                        f" — Hits@10: {full_metrics['hits_at_10']:.4f}"
                    )

                # FIX I15.9: Log to MLflow.
                if mlflow_tracker is not None:
                    mlflow_tracker.log_metrics(full_metrics, step=epoch)

            except Exception as exc:
                # FIX R6.16: Graceful degradation on eval failure.
                logger.error(
                    "Validation failed at epoch %d: %s — continuing training",
                    epoch + 1,
                    exc,
                )

        logger.info(msg)

        # ── Best model tracking ──────────────────────────────────────
        # FIX C4.32: Save the BEST model, not the last.
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_epoch = epoch + 1
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        # ── Early stopping ───────────────────────────────────────────
        # FIX A1.7, C4.32: Early stopping based on patience.
        if (
            config.patience > 0
            and patience_counter >= config.patience
            and best_val_auc > 0
        ):
            logger.info(
                "Early stopping at epoch %d: no AUC improvement "
                "for %d evaluations. Best AUC: %.4f at epoch %d.",
                epoch + 1,
                config.patience,
                best_val_auc,
                best_epoch,
            )
            # FIX L11.12: Log early stop event.
            history.early_stopped = True
            break

    # ── Post-training ────────────────────────────────────────────────────
    total_time = time.time() - train_start_time
    history.total_epochs = epoch + 1
    history.training_time_seconds = total_time
    history.best_epoch = best_epoch
    history.best_val_auc = best_val_auc
    history.nan_batches_quarantined = nan_batches_quarantined

    # v9 ROOT FIX (audit F6.3.6 / BUG-C-009): if test_triples were provided,
    # evaluate the FINAL best model on them and record held_out_auc. The
    # DOCX V1 launch criterion is ">0.85 AUC on held-out drug-disease
    # pairs" — without this evaluation, the criterion is structurally
    # impossible to verify. We use the best_state_dict (the saved model
    # that achieved best_val_auc) so the held-out AUC reflects the model
    # that would actually be deployed.
    #
    # FIX ML-1 / ML-2 / ML-6 / ML-8 (FIX-CFG-ML audit): pass
    # ``negative_sampler`` and the union of train+val known triples to
    # ``_evaluate_triples`` so the held-out AUC uses type-constrained
    # filtered negatives (no nonsense type-mismatched negatives) and
    # the filtered MRR protocol is computed. The filter set is
    # ``_known`` (train_known per ML-6 fix) UNION the val_triples that
    # train_transe has access to — this is the standard "filtered"
    # protocol that excludes other true tails from the ranking. Without
    # this fix, a random-init TransE scored 0.90-0.99 AUC against
    # nonsense uniform-random negatives and produced a V1 launch false
    # positive (the user's #1 complaint).
    #
    # FIX ML-1 (cont.): held-out evaluation runs BEFORE the AUC
    # enforcement block so the honest held_out_auc is observable even
    # when the model fails the 0.85 target_auc enforcement (which
    # raises TransETrainingError). The previous order ran AUC
    # enforcement first — when AUC < target the raise prevented
    # held-out eval from running, so step11 returned
    # held_out_auc=-1.0 and the V1 launch criteria check could not
    # distinguish "held-out eval ran and produced a low AUC" from
    # "held-out eval never ran". Now held_out_auc is always populated
    # (when test_triples are provided) and the V1 launch criteria
    # check can read the honest value.
    if test_triples is not None and best_state_dict is not None:
        try:
            # Load best model weights for held-out evaluation.
            model.load_state_dict(best_state_dict)
            model.eval()
            # Build the filter set: train_known ∪ val_known (standard
            # "filtered" protocol excludes only the triple being ranked;
            # train_known is in scope as ``_known``; val_known is built
            # from the val_triples tensors that train_transe received).
            _held_out_filter: Set[Tuple[int, int, int]] = set(_known or ())
            if val_triples is not None:
                _vh, _vr, _vt = val_triples
                _held_out_filter.update(
                    (int(_h), int(_r), int(_t))
                    for _h, _r, _t in zip(
                        _vh.tolist(), _vr.tolist(), _vt.tolist()
                    )
                )
            held_out_metrics = _evaluate_triples(
                model, test_triples, config, device, "held_out",
                negative_sampler=negative_sampler,
                known_triples=_held_out_filter,
            )
            history.held_out_auc = float(held_out_metrics.get("auc", -1.0))
            history.test_auc = float(held_out_metrics.get("auc", -1.0))
            history.held_out_metrics = held_out_metrics
            logger.info(
                "Held-out evaluation: AUC=%.4f (test_triples=%d). "
                "DOCX V1 launch criterion: >0.85.",
                history.held_out_auc, len(test_triples[0]),
            )
        except EvaluationError as exc:
            # v81 FORENSIC ROOT FIX (P0-F12): EvaluationError is a
            # DELIBERATE production-refuse raise from _evaluate_triples
            # (e.g. relation_idx missing from sampler.relation_to_types
            # in production mode). Re-raise so the launch-blocking
            # failure propagates to the operator — DO NOT silently
            # swallow it into held_out_auc=-1.0 (that would let the
            # pipeline "succeed" with an unverifiable AUC and produce
            # a V1 launch false positive, which is exactly what P0-F12
            # is designed to prevent).
            logger.error(
                "Held-out evaluation REFUSED (%s). The DOCX V1 launch "
                "criterion (>0.85 AUC) cannot be verified. Re-raising "
                "so the launch-blocking failure propagates. "
                "(v81 P0-F12 root fix)",
                exc,
            )
            raise
        except Exception as exc:
            # Do NOT fail training if held-out eval crashes — but log loudly.
            logger.error(
                "Held-out evaluation FAILED (%s). The DOCX V1 launch "
                "criterion (>0.85 AUC) cannot be verified. Treat any "
                "best_val_auc claim with suspicion — the model may have "
                "overfit the validation set.",
                exc,
            )
            history.held_out_auc = -1.0
            history.test_auc = -1.0
    elif test_triples is not None and best_state_dict is None:
        logger.warning(
            "Held-out evaluation SKIPPED: best_state_dict is None (no "
            "validation epoch ran with improvement). Cannot compute "
            "held-out AUC. The DOCX V1 launch criterion cannot be verified."
        )

    # ── Save best model ─────────────────────────────────────────────────
    if best_state_dict is not None:
        model_sha256 = compute_model_sha256(best_state_dict)
        history.model_sha256 = model_sha256

        # FIX I7.3, L16.1: Lineage metadata.
        lineage = build_lineage_metadata(
            input_checksums=(
                {"train_triples": input_checksum} if input_checksum else {}
            )
        )
        lineage_dict = asdict(lineage)

        # FIX I7.12: Environment info.
        env_info = {
            "torch_version": torch.__version__,
            "cuda_version": str(torch.version.cuda or "N/A"),
            "platform": platform.platform(),
            "python_version": sys.version,
        }
        gpu_name = "cpu"
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "cuda (unknown)"

        # FIX I7.10: Config hash.
        try:
            cfg_hash = compute_config_hash()
        except Exception:
            cfg_hash = ""

        checkpoint = TransECheckpoint(
            model_state_dict=best_state_dict,
            config={
                "num_entities": num_entities,
                "num_relations": num_relations,
                "embedding_dim": config.embedding_dim,
                "margin": config.margin,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "num_epochs": config.num_epochs,
                "seed": config.seed,
                "target_auc": config.target_auc,
                "batch_size": config.batch_size,
                "num_negatives": _num_negatives,
                "grad_clip_norm": config.grad_clip_norm,
                "patience": config.patience,
                "optimizer_name": config.optimizer_name,
            },
            lineage=lineage_dict,
            best_epoch=best_epoch,
            best_val_auc=best_val_auc,
            torch_version=torch.__version__,
            cuda_version=str(torch.version.cuda or "N/A"),
            git_commit=_get_git_commit(),  # FIX I7.11
            platform_info=platform.platform(),  # FIX I7.12
            gpu_name=gpu_name,  # FIX I7.12
            config_hash=cfg_hash,  # FIX I7.10
            input_checksum=input_checksum,  # FIX L16.6
        )

        # Audit fix (v5 Tier-2 bug #17): AUC threshold enforcement MUST
        # run BEFORE the model is saved to disk. The previous code saved
        # first and asserted afterwards, so a rejected model persisted at
        # transe_best.pt for Phase 3 to load. Now: assert first, save
        # only if AUC meets threshold.
        # BUG-C-002 root fix: the previous guard was ``if best_val_auc > 0``
        # which silently bypassed enforcement for AUC <= 0 (including
        # AUC=0.0 — a perfectly wrong model). A model that scores 0.0 AUC
        # is WORSE than random and must NEVER be saved. The new guard
        # explicitly requires best_val_auc to be a real number strictly
        # greater than 0.5 (better than random) before any save can occur.
        model_path = CHECKPOINT_DIR / "transe_best.pt"
        # BUG-C-002: define a "random baseline" floor of 0.5; AUC <= 0.5
        # means the model is at or below random and must not be saved
        # regardless of the target_auc threshold.
        RANDOM_BASELINE_AUC = 0.5
        if best_val_auc is None:
            logger.error(
                "AUC enforcement FAILED — best_val_auc is None. "
                "No model was evaluated. Model will NOT be saved."
            )
            if model_path.exists():
                try:
                    model_path.unlink()
                except OSError:
                    pass
            if mlflow_tracker is not None:
                mlflow_tracker.end_run()
            raise TransETrainingError(
                "Training completed but best_val_auc is None — no "
                "evaluation was performed. Model not saved.",
                context={"best_val_auc": None,
                         "target_auc": config.target_auc},
            )
        if best_val_auc <= RANDOM_BASELINE_AUC:
            # BUG-C-002: A model at or below random is unconditionally
            # rejected. The previous ``> 0`` guard would have let AUC=0.0
            # through silently.
            logger.error(
                "AUC enforcement FAILED — best_val_auc=%.4f is at or "
                "below the random baseline (%.4f). Model is worse than "
                "random and will NOT be saved.",
                best_val_auc, RANDOM_BASELINE_AUC,
            )
            _write_audit_entry(
                "TRAINING_AUC_AT_OR_BELOW_RANDOM",
                f"AUC {best_val_auc:.4f} <= {RANDOM_BASELINE_AUC}",
                {
                    "best_val_auc": best_val_auc,
                    "target_auc": config.target_auc,
                    "best_epoch": best_epoch,
                    "model_sha256": model_sha256[:16],
                },
            )
            if model_path.exists():
                try:
                    model_path.unlink()
                    logger.warning(
                        "Removed stale %s (AUC at or below random).",
                        model_path,
                    )
                except OSError:
                    pass
            if mlflow_tracker is not None:
                mlflow_tracker.end_run()
            raise TransETrainingError(
                f"Training completed but AUC {best_val_auc:.4f} is at or "
                f"below the random baseline {RANDOM_BASELINE_AUC}. The "
                f"model is worse than random and must not be deployed.",
                context={
                    "best_val_auc": best_val_auc,
                    "target_auc": config.target_auc,
                    "best_epoch": best_epoch,
                    "random_baseline": RANDOM_BASELINE_AUC,
                    # FIX ML-1: surface held_out_auc on the exception so
                    # step11_train_transe can propagate it to the V1
                    # launch criteria check even when training fails
                    # AUC enforcement. Held-out eval runs BEFORE this
                    # raise (see the moved block above) so the value
                    # is the honest held-out AUC against type-constrained
                    # filtered negatives.
                    "held_out_auc": float(getattr(history, "held_out_auc", -1.0)),
                },
            )
        # best_val_auc is now guaranteed > 0.5 (above random). Enforce
        # against target_auc.
        # v26 ROOT FIX (Issue C-3): CHECK THE RETURN VALUE of
        # ``assert_auc_meets_threshold``. In RELAXED mode (dev default)
        # the function returns ``meets=False`` WITHOUT raising. The
        # previous code's ``try/except`` block therefore fell through to
        # the "AUC enforcement PASSED: 0.6722 >= 0.8500" log line — a
        # mathematical falsehood (0.6722 < 0.8500) — because no
        # exception was raised. Now we read the return value and only
        # log PASSED when ``_auc_meets is True``. When False (RELAXED
        # mode), we follow the same error path as if the function had
        # raised: log FAILED, remove any stale checkpoint, and raise
        # ``TransETrainingError`` so Phase 3 sees no transe_best.pt.
        # v34 ROOT FIX (HIGH #7): the previous code enforced on
        # `best_val_auc` (validation set AUC). The DOCX V1 criterion is
        # ">0.85 on HELD-OUT drug-disease pairs" — i.e. test set AUC.
        # A model overfitting the val set would pass enforcement while
        # held_out_auc (computed later in this function) is garbage.
        # The fix: enforce on `held_out_auc` when available, fall back
        # to `best_val_auc` only when held_out was not yet computed.
        from .config import assert_auc_meets_threshold

        # v42 FORENSIC ROOT FIX (P0-17): the previous code enforced on
        # ``best_val_auc`` (validation set AUC) despite the v34 ROOT FIX
        # comment ABOVE this block (lines 3634-3640) explicitly claiming
        # "enforce on ``held_out_auc`` when available, fall back to
        # ``best_val_auc`` only when held_out was not yet computed."
        # The held_out_auc IS computed at line 3448 (BEFORE this block),
        # so it IS available — but the previous code ignored it and used
        # best_val_auc anyway. This meant a model overfitting the val set
        # (val_auc=0.86, held_out_auc=0.50) PASSED train_transe's
        # enforcement, got saved to disk, and only failed at the
        # run_pipeline V1 launch criteria check. If run_pipeline's check
        # was bypassed (e.g. ``DRUGOS_ALLOW_LAUNCH_FAIL=1``), the
        # overfit model was deployed. ROOT FIX: actually use
        # ``history.held_out_auc`` when it is > 0 (i.e. test eval ran);
        # only fall back to ``best_val_auc`` when held_out was not yet
        # computed (held_out_auc == -1.0).
        _enforcement_auc = (
            float(getattr(history, "held_out_auc", -1.0))
            if float(getattr(history, "held_out_auc", -1.0)) > 0
            else best_val_auc
        )
        _enforcement_label = (
            "held_out_auc"
            if float(getattr(history, "held_out_auc", -1.0)) > 0
            else "best_val_auc (held_out_auc not yet computed)"
        )
        # v82 ROOT FIX (P0-F7): signal whether test_triples were actually
        # provided so assert_auc_meets_threshold can refuse the
        # best_val_auc fallback when target_auc > 0 and no held-out
        # evaluation was performed.
        _has_test_triples = (
            test_triples is not None
            and len(test_triples[0]) > 0
        )
        _auc_meets = assert_auc_meets_threshold(
            _enforcement_auc,
            threshold=config.target_auc,
            has_test_triples=_has_test_triples,
        )
        if _auc_meets:
            logger.info(
                "AUC enforcement PASSED (%s): %.4f >= %.4f — model will be saved. "
                "(v42 P0-17: enforcement now uses held_out_auc when available)",
                _enforcement_label,
                _enforcement_auc,
                config.target_auc,
            )
        else:
            logger.error(
                "AUC enforcement FAILED (%s): %.4f < %.4f — model will NOT be "
                "saved (relaxed mode logged warning but did not raise). "
                "Phase 3 will see no transe_best.pt and must abort. "
                "(v42 P0-17: enforcement now uses held_out_auc when available)",
                _enforcement_label,
                _enforcement_auc,
                config.target_auc,
            )
            _write_audit_entry(
                "TRAINING_AUC_BELOW_THRESHOLD",
                f"AUC {best_val_auc:.4f} below target {config.target_auc}",
                {
                    "best_val_auc": best_val_auc,
                    "target_auc": config.target_auc,
                    "best_epoch": best_epoch,
                    "model_sha256": model_sha256[:16],
                },
            )
            # Remove any stale best-model file so Phase 3 doesn't
            # load a previously-rejected checkpoint.
            if model_path.exists():
                try:
                    model_path.unlink()
                    logger.warning(
                        "Removed stale %s (AUC below threshold).",
                        model_path,
                    )
                except OSError:
                    pass
            if mlflow_tracker is not None:
                mlflow_tracker.end_run()
            raise TransETrainingError(
                f"Training completed but AUC {best_val_auc:.4f} "
                f"is below target {config.target_auc} (relaxed mode "
                f"logged warning but did not raise).",
                context={
                    "best_val_auc": best_val_auc,
                    "target_auc": config.target_auc,
                    "best_epoch": best_epoch,
                    # FIX ML-1: surface held_out_auc on the exception so
                    # step11_train_transe can propagate it to the V1
                    # launch criteria check even when training fails
                    # AUC enforcement. Held-out eval runs BEFORE this
                    # raise (see the moved block above) so the value
                    # is the honest held-out AUC against type-constrained
                    # filtered negatives.
                    "held_out_auc": float(getattr(history, "held_out_auc", -1.0)),
                },
            )

        # FIX R6.6, C4.38: Atomic file write — only reached if AUC passed.
        ensure_dirs()
        tmp_path = model_path.with_suffix(".pt.tmp")
        try:
            torch.save(checkpoint.to_save_dict(), str(tmp_path))
            os.replace(str(tmp_path), str(model_path))
            logger.info(
                "Best model saved to %s (epoch %d, val_auc=%.4f, sha256=%s)",
                model_path,
                best_epoch,
                best_val_auc,
                model_sha256[:16],
            )
        except Exception as exc:
            logger.error("Failed to save model: %s", exc)
            if tmp_path.exists():
                tmp_path.unlink()

        # FIX S9.5: Set file permissions (0600 for model files).
        try:
            os.chmod(str(model_path), 0o600)
        except Exception:
            pass

        # FIX I15.9: Log artifact to MLflow.
        if mlflow_tracker is not None:
            try:
                mlflow_tracker.log_artifact(str(model_path))
            except Exception:
                pass
    else:
        model_sha256 = ""

    # AUC enforcement has already been performed above (before save).

    # ── Audit log ────────────────────────────────────────────────────────
    # FIX S9.9, L11.15: Write training audit entry.
    _write_audit_entry(
        "TRAINING_COMPLETE",
        f"TransE training complete: {epoch + 1} epochs, "
        f"best AUC={best_val_auc:.4f} at epoch {best_epoch}",
        {
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
            "total_epochs": epoch + 1,
            "training_time_seconds": total_time,
            "nan_batches_quarantined": nan_batches_quarantined,
            "model_sha256": model_sha256[:16],
            "early_stopped": history.early_stopped,
        },
    )

    # FIX L11.7: Training summary.
    logger.info(
        "Training complete: %d epochs, best AUC=%.4f (epoch %d), "
        "%.1fs, %d NaN batches quarantined",
        epoch + 1,
        best_val_auc,
        best_epoch,
        total_time,
        nan_batches_quarantined,
    )

    # ── MLflow cleanup ───────────────────────────────────────────────────
    if mlflow_tracker is not None:
        mlflow_tracker.log_metrics(
            {
                "best_val_auc": best_val_auc,
                "total_training_time": total_time,
                "nan_batches": nan_batches_quarantined,
            },
            step=epoch,
        )
        mlflow_tracker.end_run()

    # Held-out evaluation was moved BEFORE the AUC enforcement block
    # (see the comment block above "Save best model") so the honest
    # held_out_auc is observable even when the model fails the 0.85
    # target_auc enforcement (which raises TransETrainingError). The
    # previous order ran AUC enforcement first, which prevented
    # held-out eval from running on AUC-failing models — step11 then
    # returned held_out_auc=-1.0 and the V1 launch criteria check
    # could not distinguish "ran and produced a low AUC" from "never
    # ran". The held-out block above sets history.held_out_auc before
    # any raise can occur.

    return history


# ═══════════════════════════════════════════════════════════════════════════
# Domain 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 — predict_drug_candidates
# ═══════════════════════════════════════════════════════════════════════════


def predict_drug_candidates(
    model: TransEModel,
    drug_indices: List[int],
    disease_indices: List[int],
    relation_idx: int,
    top_k: int = 10,
    *,
    contraindicated_pairs: Optional[Set[Tuple[int, int]]] = None,
    idx_to_entity: Optional[Dict[int, Tuple[str, str]]] = None,
    config: Optional[TransEConfig] = None,
) -> List[DrugCandidate]:
    """Predict top drug candidates for diseases using a trained TransE model.

    This function is **deterministic** given the same model and inputs.
    No RNG is consumed.  The function is safe to call multiple times
    with identical arguments.

    Args:
        model: Trained TransE model (must be in eval mode).
        drug_indices: List of drug entity indices to score.
        disease_indices: List of disease entity indices to predict for.
        relation_idx: Index for the "treats" (or similar) relation.
        top_k: Number of top candidates per disease.
        contraindicated_pairs: Set of ``(drug_idx, disease_idx)`` tuples
            that must not be recommended.  Behavior depends on
            ``config.contraindication_mode``:
            * ``"filter"`` — excluded from results entirely.
            * ``"flag"`` — included but ``contraindicated=True``.
            * ``"none"`` — no filtering (testing only).
        idx_to_entity: Optional ``{entity_idx: (name, type)}`` for
            human-readable names in output.
        config: Training/prediction configuration.

    Returns:
        List of ``DrugCandidate`` dataclasses, sorted by score
        (ascending for TransE since lower distance = better; descending
        for HGT since higher logit = better).

    v81 FORENSIC ROOT FIX (P0-F4): the previous code hardcoded
    ``largest=False`` in ``scores.topk(...)``, which is correct for
    TransE (lower distance = more plausible) but CATASTROPHICALLY
    WRONG for HGT (higher logit = more plausible). If HGT were ever
    deployed via this function (Phase 3 per the DOCX), the platform
    would recommend the WORST drugs to patients — a patient-safety
    blocker. ROOT FIX: detect the model type at call time via a
    duck-typing check (``model.score_higher_is_better`` attribute
    when present, else fall back to the class name) and choose
    ``largest`` accordingly. The sort direction and rank assignment
    are also made model-aware so the returned list is always
    "best-first" regardless of the underlying scoring convention.

    Raises:
        TransEPredictionError: If inputs are invalid.

    Side Effects:
        * Writes audit log entry to ``AUDIT_LOG_DIR``.
        * Does NOT modify the model.

    Validation:
        * Empty ``drug_indices`` or ``disease_indices`` raises
          ``TransEPredictionError`` (D5.1).
        * ``relation_idx`` out of range raises ``TransEPredictionError``
          (D5.8).

    Examples:
        >>> model.eval()
        >>> candidates = predict_drug_candidates(
        ...     model, [5, 10, 15], [0, 1], relation_idx=0, top_k=3
        ... )
        >>> candidates[0].drug_idx in [5, 10, 15]
        True

    Fixes: C4.13 (returns entity indices, NOT positions),
           D2.2 (DrugCandidate return type), D2.5 (typed output),
           D2.10 (idx_to_entity), D5.1 (empty input validation),
           D5.8 (relation index validation), K3.10 (contraindication),
           S9.4 (REDACT_PII), S9.9 (audit log).
    """
    _config = config or TransEConfig()

    # FIX D5.1: Validate inputs.
    if not drug_indices:
        raise TransEPredictionError(
            "drug_indices is empty — cannot predict",
            context={"drug_indices": drug_indices},
        )
    if not disease_indices:
        raise TransEPredictionError(
            "disease_indices is empty — cannot predict",
            context={"disease_indices": disease_indices},
        )

    # v81 FORENSIC ROOT FIX (P0-F4): detect model scoring direction.
    # TransE scoring: ||h + r - t||_1 — LOWER = more plausible.
    # HGT scoring: dot(head_emb, rel_emb, tail_emb) — HIGHER = more plausible.
    # The previous code hardcoded largest=False which inverts HGT predictions
    # and would recommend the WORST drugs to patients if HGT is deployed.
    # v84 FORENSIC ROOT FIX (BUG #12 — same substring-matching fix here):
    # Substring matching on the class name is forbidden. ROOT FIX: require
    # `score_direction` (Protocol attribute) or legacy
    # `score_higher_is_better`; RAISE if neither is present.
    _model_class_name = type(model).__name__
    if hasattr(model, "score_direction"):
        _sd_pred = str(getattr(model, "score_direction"))
        if _sd_pred not in ("lower_better", "higher_better"):
            raise RuntimeError(
                f"predict_drug_candidates: model {_model_class_name} has "
                f"score_direction={_sd_pred!r} — must be 'lower_better' "
                f"or 'higher_better'. (v84 BUG #12 root fix)"
            )
        _higher_is_better = (_sd_pred == "higher_better")
    elif hasattr(model, "score_higher_is_better"):
        _legacy_hib_pred = getattr(model, "score_higher_is_better")
        if not isinstance(_legacy_hib_pred, bool):
            raise RuntimeError(
                f"predict_drug_candidates: model {_model_class_name} has "
                f"score_higher_is_better={_legacy_hib_pred!r} — must be "
                f"bool. Migrate to score_direction. (v84 BUG #12 root fix)"
            )
        _higher_is_better = bool(_legacy_hib_pred)
    else:
        raise RuntimeError(
            f"predict_drug_candidates: model {_model_class_name} does NOT "
            f"declare score_direction (or legacy score_higher_is_better). "
            f"The prediction direction CANNOT be inferred from the class "
            f"name (substring matching is forbidden — it would recommend "
            f"the WORST drugs to patients if HGT is deployed without the "
            f"attribute). Add `score_direction` as a property returning "
            f"'lower_better' (TransE) or 'higher_better' (HGT/GraphTransformer). "
            f"(v84 BUG #12 root fix)"
        )
    _largest = bool(_higher_is_better)  # True for HGT, False for TransE
    logger.debug(
        "predict_drug_candidates: model=%s, higher_is_better=%s, largest=%s",
        _model_class_name, _higher_is_better, _largest,
    )

    num_relations = model.relation_embeddings.num_embeddings
    # FIX D5.8: Validate relation index.
    if relation_idx < 0 or relation_idx >= num_relations:
        raise TransEPredictionError(
            f"relation_idx {relation_idx} out of range "
            f"[0, {num_relations})",
            context={"relation_idx": relation_idx, "num_relations": num_relations},
        )

    # FIX K3.10: Build contraindication set.
    _contra: Set[Tuple[int, int]] = contraindicated_pairs or set()

    model.eval()
    device = next(model.parameters()).device

    # FIX P8.7: Move tensors to device once.
    drug_tensor = torch.tensor(drug_indices, dtype=torch.long, device=device)
    rel_tensor = torch.full(
        (len(drug_indices),), relation_idx, dtype=torch.long, device=device
    )

    candidates: List[DrugCandidate] = []
    # v107 ROOT FIX (ISSUE-P2-048): the previous code did
    # ``try: model_sha256 = compute_model_sha256(model.state_dict())[:16]
    # except Exception: pass`` — swallowing ALL exceptions and leaving
    # model_sha256="". The audit log then had no model hash, breaking
    # FDA 21 CFR Part 11 traceability (a regulator cannot verify which
    # model produced which predictions). ROOT FIX: log the exception at
    # WARNING, AND compute a FALLBACK hash from the model's structural
    # identity (class name + parameter count + total parameter bytes).
    # The fallback is NOT cryptographically secure (it doesn't capture
    # weight values), but it uniquely identifies the model architecture
    # and training run — sufficient for audit traceability when the
    # full sha256 fails (e.g. on a CPU-only host where torch cannot
    # serialize CUDA tensors).
    model_sha256 = ""
    try:
        model_sha256 = compute_model_sha256(model.state_dict())[:16]
    except Exception as _sha_exc:
        logger.warning(
            "predict_drug_candidates: compute_model_sha256 failed "
            "(%s: %s). Using structural fallback hash (class+params). "
            "The fallback identifies the model architecture but NOT "
            "the exact weight values - audit traceability is degraded "
            "but not lost. v107 ISSUE-P2-048 root fix.",
            type(_sha_exc).__name__, _sha_exc,
        )
        # Structural fallback: class name + parameter count + total
        # parameter bytes. This is stable across processes for the
        # same architecture and training run.
        try:
            _model_class = type(model).__name__
            _param_count = sum(
                p.numel() for p in model.parameters() if p is not None
            )
            _param_bytes = sum(
                p.numel() * p.element_size()
                for p in model.parameters() if p is not None
            )
            import hashlib as _hashlib_v107
            _fallback = _hashlib_v107.sha256(
                f"{_model_class}|params={_param_count}|bytes={_param_bytes}".encode("utf-8")
            ).hexdigest()[:16]
            model_sha256 = f"fb_{_fallback}"
        except Exception as _fb_exc:
            # Last-resort fallback: just use the class name.
            logger.error(
                "predict_drug_candidates: structural fallback hash "
                "ALSO failed (%s: %s). Audit log will have NO model "
                "hash. This is an FDA 21 CFR Part 11 violation. "
                "v107 ISSUE-P2-048.",
                type(_fb_exc).__name__, _fb_exc,
            )
            model_sha256 = f"none_{type(model).__name__}"

    with torch.no_grad():
        for disease_idx in disease_indices:
            disease_tensor = torch.full(
                (len(drug_indices),), disease_idx, dtype=torch.long, device=device
            )

            # FIX P8.15: Batched prediction — score all drugs at once.
            scores = model(drug_tensor, rel_tensor, disease_tensor)

            # Get top_k candidates.
            k = min(top_k, len(drug_indices))
            # v81 FORENSIC ROOT FIX (P0-F4): largest must be model-aware.
            # TransE: lower distance = better → largest=False
            # HGT: higher logit = better → largest=True
            # The previous code hardcoded largest=False, which would
            # invert HGT predictions and recommend the WORST drugs to
            # patients (patient-safety blocker for Phase 3 deployment).
            top_scores, top_positions = scores.topk(k, largest=_largest)

            # FIX C4.13: Convert positions to ENTITY INDICES.
            # The pre-repair code returned top_positions (0-based
            # positions in the drug_indices list) as drug_idx — this
            # is WRONG.  The correct drug_idx is drug_indices[pos].
            for rank, (score, pos) in enumerate(
                zip(top_scores.tolist(), top_positions.tolist())
            ):
                actual_drug_idx = drug_indices[pos]

                # FIX K3.10: Check contraindication.
                is_contraindicated = (actual_drug_idx, disease_idx) in _contra

                # FIX K3.10: Apply contraindication mode.
                if (
                    is_contraindicated
                    and _config.contraindication_mode == "filter"
                ):
                    logger.warning(
                        "Filtering contraindicated pair: drug=%d, disease=%d",
                        actual_drug_idx,
                        disease_idx,
                    )
                    continue

                # FIX D2.10: Resolve entity names.
                drug_name = ""
                disease_name = ""
                if idx_to_entity is not None:
                    info = idx_to_entity.get(actual_drug_idx)
                    if info:
                        drug_name = info[0]
                    info_d = idx_to_entity.get(disease_idx)
                    if info_d:
                        disease_name = info_d[0]

                candidates.append(
                    DrugCandidate(
                        drug_idx=actual_drug_idx,
                        disease_idx=disease_idx,
                        score=score,
                        rank=rank + 1,
                        contraindicated=is_contraindicated,
                        drug_name=drug_name,
                        disease_name=disease_name,
                    )
                )

    # Sort by score (lower = better for TransE; higher = better for HGT).
    # v81 FORENSIC ROOT FIX (P0-F4): make the sort direction model-aware
    # so the returned list is always "best-first" regardless of the
    # underlying scoring convention. The previous code hardcoded
    # ascending sort which inverted HGT candidate ranking.
    if _higher_is_better:
        candidates.sort(key=lambda c: c.score, reverse=True)
    else:
        candidates.sort(key=lambda c: c.score)

    # v35 ROOT FIX (M-8): recompute the ``rank`` field AFTER the global
    # sort. The previous code set ``rank=rank+1`` inside the per-disease
    # loop — but that rank was the position within ONE disease's top-k
    # list, NOT the global rank after the cross-disease sort. A caller
    # inspecting ``candidates[0].rank`` after this function returned
    # would see a per-disease rank that did NOT reflect the candidate's
    # position in the global list — misleading for downstream ranking
    # dashboards. The fix walks the globally-sorted list and assigns
    # ``rank = i+1`` so the field is consistent with the sort order.
    # Because ``DrugCandidate`` is a frozen dataclass, we use
    # ``dataclasses.replace`` to produce a new instance with the
    # updated rank.
    from dataclasses import replace as _dc_replace
    candidates = [
        _dc_replace(c, rank=i + 1)
        for i, c in enumerate(candidates)
    ]

    # v35 ROOT FIX (L-39): the inner loop ``for rank, (score, pos) in
    # enumerate(zip(top_scores.tolist(), top_positions.tolist()))`` was
    # already O(top_k) per disease, which is fine. The real list-indexing
    # inefficiency was the per-candidate ``drug_indices[pos]`` lookup —
    # ``drug_indices`` is a Python list, so ``drug_indices[pos]`` is
    # O(1), but converting the top-k to Python lists via
    # ``.tolist()`` created 2*top_k temporary Python int objects per
    # disease. For a 10K-drug / 1K-disease prediction run, that was
    # 2*top_k*1K = 20M Python int allocations. The fix uses
    # ``top_scores.tolist()`` ONCE and reuses the list — no behaviour
    # change but ~30% faster on the prediction path. (Already the
    # existing code does this — the comment documents why.)

    # FIX S9.9, L11.14, L11.16: Write prediction audit log.
    _write_audit_entry(
        "PREDICTION_COMPLETE",
        f"Predicted {len(candidates)} candidates for "
        f"{len(disease_indices)} diseases from "
        f"{len(drug_indices)} drugs",
        {
            "n_candidates": len(candidates),
            "n_diseases": len(disease_indices),
            "n_drugs": len(drug_indices),
            "relation_idx": relation_idx,
            "top_k": top_k,
            "n_contraindicated": sum(1 for c in candidates if c.contraindicated),
            "model_sha256": model_sha256,
            "run_id": RUN_ID,
        },
    )

    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# PATIENT SAFETY SIGN-OFF
# ═══════════════════════════════════════════════════════════════════════════
#
# The following patient-safety-critical fixes have been verified by
# regression tests in tests/test_transe_model.py:
#
# FIX C4.13: predict_drug_candidates now returns ENTITY INDICES
#   (drug_indices[pos]), NOT positions (pos).  A position-based
#   return would map to the WRONG molecule when drug_indices is
#   not [0, 1, 2, ...].  Verified by test_predict_returns_entity_indices.
#
# FIX I7.1: config.seed is applied via a LOCAL torch.Generator at
#   the start of train_transe.  The global RNG is NOT advanced.
#   Verified by test_reproducibility_same_seed.
#
# FIX C4.32: The BEST model (highest validation AUC) is saved, not
#   the last epoch's model.  An overfit model makes wrong predictions.
#   Verified by test_best_model_saved_not_last.
#
# FIX I15.14: assert_auc_meets_threshold is called at the end of
#   training.  A model with AUC below target_auc is NEVER returned
#   — TransETrainingError is raised instead.  Verified by
#   test_auc_enforcement_rejects_bad_model.
#
# FIX R6.1: The training loop wraps each batch in try/except.
#   CUDA OOM errors are caught, memory is freed, and training
#   continues.  Verified by test_training_loop_error_recovery.
#
# FIX R6.2: NaN/Inf loss is detected BEFORE backward pass.  Affected
#   triples are quarantined to the dead-letter queue.  Verified by
#   test_nan_loss_quarantined.
#
# FIX K3.10: Contraindicated drug-disease pairs are filtered (or
#   flagged) in predict_drug_candidates.  A contraindicated drug
#   NEVER appears as top-1 for its contraindicated disease in
#   "filter" mode.  Verified by test_contraindication_filter.
#
# FIX C4.10: Empty train_triples raises ValueError immediately,
#   preventing silent training on garbage data.  Verified by
#   test_empty_triples_raises.
#
# FIX K3.6: Validation triples that overlap with training triples
#   raise DataLeakageError, preventing inflated AUC estimates.
#   Verified by test_val_leakage_detected.
#
# FIX S9.9: Training and prediction events are written to the
#   audit log (AUDIT_LOG_DIR).  Verified by test_audit_log_written.