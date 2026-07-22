"""Canonical Phase 3 model checkpoint format — the SINGLE source of truth.

TASK 324 ROOT FIX (forensic, root-level, no surface fix):
  The Phase 3 trainer (``graph_transformer/training/trainer.py``) saves a
  checkpoint .pt file via ``torch.save(checkpoint, path)``. The checkpoint
  is a Python dict. This module defines the EXACT set of keys, value
  types, and constraints that the dict MUST satisfy.

  Both the writer (trainer) and readers (inference service, Phase 4 env)
  import this contract. Any change to the checkpoint format is a
  compile-time error on both sides — the contract consistency test
  (Task 330) verifies the trainer's checkpoint dict matches this contract.

Checkpoint format (v3 — self-contained)
---------------------------------------
The checkpoint .pt file contains a single Python dict with the following
keys (REQUIRED unless marked OPTIONAL):

    {
      # --- Model weights ---
      "model_state_dict": Dict[str, Tensor],     # the trained model weights
      "best_state_dict": Dict[str, Tensor],      # OPTIONAL — best-val-auc weights
                                                  # (skipped if no epoch improved val AUC)

      # --- Optimizer state (for resuming training) ---
      "optimizer_state_dict": Dict[str, Any],

      # --- Training metrics ---
      "best_val_auc": float,                      # best validation AUC achieved
      "best_val_loss": float,                     # val loss at the best AUC epoch
      "best_epoch": int,                          # epoch index of best_val_auc
      "history": List[Dict[str, float]],          # per-epoch training history

      # --- Graph schema (lightweight summary) ---
      "graph_schema": {
        "node_types": List[str],                  # e.g. ["drug", "protein", ...]
        "feature_dims": Dict[str, int],           # node_type -> feature dim
        "edge_types": List[List[str]],            # list of [src, rel, dst]
      },

      # --- Package versioning ---
      "package_version": str,                     # graph_transformer.__version__
      "schema_version": str,                      # graph_transformer.__schema_version__

      # --- Self-contained model reconstruction (audit Issue 124) ---
      "model_class_name": str,                    # qualified class name for dispatch
      "hyperparams": Dict[str, Any],              # cls(**hyperparams) reconstructs arch

      # --- Self-contained graph tensors (audit Issue 139) ---
      "node_features": Dict[str, Tensor],         # node_type -> feature tensor
      "edge_indices": Dict[Tuple[str,str,str], Tensor],  # edge_type -> edge_index
      "node_maps": Dict[str, Dict[str, int]],     # node_type -> {name -> index}

      # --- Name resolution (for HTTP API) ---
      "drug_names": List[str],                    # ordered list of drug names
      "disease_names": List[str],                 # ordered list of disease names
      "known_pairs": List[Tuple[str, str]],       # known (drug, disease) pairs to filter

      # --- Training metadata (provenance) ---
      "training_metadata": Dict[str, Any],        # who/when/what trained this model
    }

Why this contract exists
------------------------
A bug in the checkpoint format = wrong graph = wrong prediction = a
pharma partner tests the wrong drug on a real patient = patient harm.
This contract makes checkpoint format changes a HARD ERROR on both
writer and reader sides, preventing silent corruption.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


# =============================================================================
# Checkpoint format version
# =============================================================================
# Bump when the checkpoint schema changes. Readers MUST reject checkpoints
# with incompatible versions.
CHECKPOINT_FORMAT_VERSION: str = "3"

# =============================================================================
# REQUIRED checkpoint keys — must be present in every checkpoint
# =============================================================================
# The trainer's save_checkpoint() MUST include all of these. Readers MAY
# assume they are present (use direct indexing, not .get()).
#
# NOTE: ``training_metadata`` is intentionally OPTIONAL (not in this list)
# because the current trainer doesn't produce a provenance sub-dict. It's
# documented in CHECKPOINT_OPTIONAL_KEYS below as a TODO for a future PR
# — the contract test verifies the REQUIRED keys are present, and a
# WARNING is logged if training_metadata is missing.
CHECKPOINT_REQUIRED_KEYS: Tuple[str, ...] = (
    "model_state_dict",         # the trained weights (last epoch)
    "optimizer_state_dict",     # for resuming training
    "best_val_auc",             # primary model-quality metric
    "best_val_loss",            # secondary metric
    "best_epoch",               # which epoch was best
    "history",                  # per-epoch training log
    "graph_schema",             # lightweight graph summary
    "package_version",          # graph_transformer.__version__
    "schema_version",           # graph_transformer.__schema_version__
    "model_class_name",         # for inference service dispatch
    "hyperparams",              # to reconstruct model architecture
    "node_features",            # actual graph tensors
    "edge_indices",             # actual graph tensors
    "node_maps",                # name -> index lookups
    "drug_names",               # for HTTP API name resolution
    "disease_names",            # for HTTP API name resolution
    "known_pairs",              # to filter known pairs from top-k
)

# =============================================================================
# OPTIONAL checkpoint keys — may be present, readers must use .get()
# =============================================================================
# ``best_state_dict``: only present if some epoch improved val AUC.
# ``training_metadata``: provenance sub-dict (trained_at, trained_by, etc.).
#   The current trainer doesn't produce this — it's a documented TODO.
#   Readers should use .get("training_metadata", {}) and log a warning
#   if missing. The contract test verifies the REQUIRED keys are present;
#   a missing training_metadata is a soft warning, not a hard error.
CHECKPOINT_OPTIONAL_KEYS: Tuple[str, ...] = (
    "best_state_dict",          # only present if some epoch improved val AUC
    "training_metadata",        # provenance sub-dict (TODO — not yet produced)
)

# All keys (required + optional).
CHECKPOINT_ALL_KEYS: Tuple[str, ...] = CHECKPOINT_REQUIRED_KEYS + CHECKPOINT_OPTIONAL_KEYS

# =============================================================================
# Per-key value type spec (for runtime validation)
# =============================================================================
CHECKPOINT_KEY_TYPES: Dict[str, str] = {
    "model_state_dict": "dict",
    "best_state_dict": "dict",
    "optimizer_state_dict": "dict",
    "best_val_auc": "float",
    "best_val_loss": "float",
    "best_epoch": "int",
    "history": "list",
    "graph_schema": "dict",
    "package_version": "str",
    "schema_version": "str",
    "model_class_name": "str",
    "hyperparams": "dict",
    "node_features": "dict",
    "edge_indices": "dict",
    "node_maps": "dict",
    "drug_names": "list",
    "disease_names": "list",
    "known_pairs": "list",
    "training_metadata": "dict",
}

# =============================================================================
# Model class name defaults
# =============================================================================
# The default production model class. The inference service uses this if
# the checkpoint's ``model_class_name`` is missing or unparseable.
DEFAULT_MODEL_CLASS_NAME: str = "DrugRepurposingGraphTransformer"

# All model class names the inference service knows how to dispatch to.
# Adding a new model variant = append to this tuple + register in service.py.
SUPPORTED_MODEL_CLASS_NAMES: Tuple[str, ...] = (
    "DrugRepurposingGraphTransformer",
    "graph_transformer.models.graph_transformer.DrugRepurposingGraphTransformer",
)

# =============================================================================
# graph_schema sub-dict required keys
# =============================================================================
GRAPH_SCHEMA_REQUIRED_KEYS: Tuple[str, ...] = (
    "node_types",       # List[str] — Phase 3 canonical node types
    "feature_dims",     # Dict[str, int] — node_type -> feature tensor dim
    "edge_types",       # List[List[str]] — list of [src, rel, dst]
)

# =============================================================================
# hyperparams sub-dict required keys
# =============================================================================
# These are the architecture params the inference service needs to
# reconstruct the model via cls(**hyperparams).
HYPERPARAMS_REQUIRED_KEYS: Tuple[str, ...] = (
    "embedding_dim",        # int — node embedding dimension
    "num_layers",           # int — number of transformer layers
    "num_heads",            # int — attention heads per layer
    "dropout",              # float — dropout rate
    "edge_dim",             # int — edge feature dimension
    "hidden_dim",           # int — hidden layer dimension in MLP head
)

# =============================================================================
# training_metadata sub-dict required keys
# =============================================================================
# Provenance: who trained this, when, on what data, for how long.
TRAINING_METADATA_REQUIRED_KEYS: Tuple[str, ...] = (
    "trained_at",           # ISO 8601 timestamp
    "trained_by",           # username / agent ID
    "training_duration_s",  # float — wall-clock seconds
    "epochs_completed",     # int — actual epochs run (may differ from config)
    "early_stopped",        # bool — True if early stopping triggered
    "gpu_used",             # bool — True if GPU was used
    "dataset_version",      # str — Phase 1 data version hash
)


# =============================================================================
# Checkpoint validator
# =============================================================================


def validate_checkpoint_dict(
    checkpoint: Dict[str, Any],
    *,
    strict: bool = True,
) -> List[str]:
    """Validate a checkpoint dict against the contract.

    Returns a list of error messages. Empty list = valid checkpoint.

    Parameters
    ----------
    checkpoint : dict
        The checkpoint dict to validate (typically loaded via torch.load).
    strict : bool, default True
        If True, unknown keys trigger errors. If False, unknown keys
        trigger warnings.

    Returns
    -------
    list of str
        Empty if valid; otherwise a list of human-readable error messages.
    """
    errors: List[str] = []

    if not isinstance(checkpoint, dict):
        return [f"Checkpoint must be a dict, got {type(checkpoint).__name__}."]

    # Check 1: all required keys present.
    for key in CHECKPOINT_REQUIRED_KEYS:
        if key not in checkpoint:
            errors.append(
                f"Missing required checkpoint key {key!r}. "
                f"Required keys: {list(CHECKPOINT_REQUIRED_KEYS)}."
            )

    # Check 2: no unknown keys (in strict mode).
    if strict:
        for key in checkpoint.keys():
            if key not in CHECKPOINT_ALL_KEYS:
                errors.append(
                    f"WARNING: Unknown checkpoint key {key!r}. "
                    f"Known keys: {list(CHECKPOINT_ALL_KEYS)}."
                )

    # Check 3: model_class_name is supported.
    cls_name = checkpoint.get("model_class_name")
    if cls_name is not None:
        # Accept either the short name or the qualified name.
        if (cls_name not in SUPPORTED_MODEL_CLASS_NAMES
                and not any(cls_name.endswith(s) for s in SUPPORTED_MODEL_CLASS_NAMES)):
            errors.append(
                f"model_class_name {cls_name!r} is not in the supported list "
                f"{list(SUPPORTED_MODEL_CLASS_NAMES)}. Register the new class "
                f"in graph_transformer/contracts/phase3_schema.py "
                f"SUPPORTED_MODEL_CLASS_NAMES."
            )

    # Check 4: graph_schema sub-dict.
    graph_schema = checkpoint.get("graph_schema")
    if graph_schema is not None and isinstance(graph_schema, dict):
        for key in GRAPH_SCHEMA_REQUIRED_KEYS:
            if key not in graph_schema:
                errors.append(
                    f"graph_schema missing required key {key!r}. "
                    f"Required: {list(GRAPH_SCHEMA_REQUIRED_KEYS)}."
                )

    # Check 5: hyperparams sub-dict.
    hyperparams = checkpoint.get("hyperparams")
    if hyperparams is not None and isinstance(hyperparams, dict):
        # In lenient mode, only warn about missing hyperparams (some
        # legacy checkpoints may not have all of them).
        for key in HYPERPARAMS_REQUIRED_KEYS:
            if key not in hyperparams:
                errors.append(
                    f"WARNING: hyperparams missing key {key!r}. "
                    f"Required: {list(HYPERPARAMS_REQUIRED_KEYS)}. "
                    f"The inference service will fall back to the model "
                    f"class's default for this parameter."
                )

    # Check 6: training_metadata sub-dict.
    training_metadata = checkpoint.get("training_metadata")
    if training_metadata is not None and isinstance(training_metadata, dict):
        for key in TRAINING_METADATA_REQUIRED_KEYS:
            if key not in training_metadata:
                errors.append(
                    f"WARNING: training_metadata missing key {key!r}. "
                    f"Required: {list(TRAINING_METADATA_REQUIRED_KEYS)}."
                )

    # Check 7: best_val_auc is a non-negative float.
    best_auc = checkpoint.get("best_val_auc")
    if best_auc is not None:
        if not isinstance(best_auc, (int, float)):
            errors.append(
                f"best_val_auc must be a float, got {type(best_auc).__name__}."
            )
        elif best_auc < 0.0 or best_auc > 1.0:
            errors.append(
                f"best_val_auc must be in [0, 1], got {best_auc}. "
                f"AUC > 1.0 indicates a bug in the trainer's metric computation."
            )

    # P3-036 ROOT FIX (v113 forensic): validate best_val_loss.
    # The previous validator checked best_val_auc is in [0, 1] but did
    # NOT check best_val_loss. best_val_loss is a BCEWithLogitsLoss
    # value, which is always non-negative (negative log likelihood). A
    # corrupted checkpoint could have best_val_loss = -1.0 (sign error)
    # or best_val_loss = NaN (numerical instability) and the validator
    # would accept it. Downstream consumers (the service, the bridge)
    # would display "best val loss = -1.0" or "best val loss = NaN" to
    # the operator, losing confidence in the platform's quality metrics.
    best_loss = checkpoint.get("best_val_loss")
    if best_loss is not None:
        import math as _math
        if not isinstance(best_loss, (int, float)):
            errors.append(
                f"best_val_loss must be a float, got {type(best_loss).__name__}."
            )
        elif _math.isnan(best_loss):
            errors.append(
                f"best_val_loss is NaN -- indicates numerical instability "
                f"during training (e.g., exploding gradients, fp16 overflow). "
                f"The checkpoint is corrupt; retrain with gradient clipping "
                f"and/or a lower learning rate."
            )
        elif _math.isinf(best_loss):
            errors.append(
                f"best_val_loss is Inf -- indicates a training divergence "
                f"(loss overflowed to +Inf). The checkpoint is corrupt; "
                f"retrain with gradient clipping and/or a lower learning rate."
            )
        elif best_loss < 0.0:
            errors.append(
                f"best_val_loss must be non-negative (BCEWithLogitsLoss is a "
                f"negative log likelihood), got {best_loss}. A negative loss "
                f"indicates a sign error in the trainer's loss computation."
            )

    # P3-036 ROOT FIX (v113): also validate best_epoch is a non-negative int.
    best_epoch = checkpoint.get("best_epoch")
    if best_epoch is not None:
        if not isinstance(best_epoch, int) or isinstance(best_epoch, bool):
            errors.append(
                f"best_epoch must be an int, got {type(best_epoch).__name__}."
            )
        elif best_epoch < 0:
            errors.append(
                f"best_epoch must be non-negative, got {best_epoch}. "
                f"A negative epoch index indicates a bug in the trainer's "
                f"checkpoint-selection logic."
            )

    return errors


def is_valid_checkpoint(checkpoint: Dict[str, Any]) -> bool:
    """Return True if ``checkpoint`` satisfies the contract (no errors)."""
    return not validate_checkpoint_dict(checkpoint, strict=True)
