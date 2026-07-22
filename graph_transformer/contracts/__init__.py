"""graph_transformer.contracts package — Phase 3 model checkpoint contract.

TASK 324 ROOT FIX (forensic, root-level):
  This package defines the canonical Phase 3 model checkpoint format.
  Phase 3 (writer: ``graph_transformer.training.trainer.GraphTransformerTrainer``)
  saves a checkpoint .pt file. Phase 3 inference service (reader:
  ``graph_transformer.service``) and Phase 4 RL env (reader:
  ``rl.env``) load this checkpoint.

  Previously the checkpoint format was defined INLINE in the trainer's
  ``save_checkpoint()`` method — readers had to reverse-engineer the
  schema from the writer's source code. When the trainer added a new key
  (e.g. ``best_state_dict``, ``hyperparams``), readers silently broke
  until someone noticed.

  This module extracts the checkpoint format into a CONTRACT that all
  sides import. Any change to the format is a compile-time error on
  both sides — the contract consistency test (Task 330) verifies the
  trainer's checkpoint dict matches this contract.
"""
from __future__ import annotations

from graph_transformer.contracts.phase3_schema import (
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_REQUIRED_KEYS,
    CHECKPOINT_OPTIONAL_KEYS,
    CHECKPOINT_ALL_KEYS,
    CHECKPOINT_KEY_TYPES,
    DEFAULT_MODEL_CLASS_NAME,
    SUPPORTED_MODEL_CLASS_NAMES,
    TRAINING_METADATA_REQUIRED_KEYS,
    GRAPH_SCHEMA_REQUIRED_KEYS,
    HYPERPARAMS_REQUIRED_KEYS,
    validate_checkpoint_dict,
    is_valid_checkpoint,
)

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CHECKPOINT_REQUIRED_KEYS",
    "CHECKPOINT_OPTIONAL_KEYS",
    "CHECKPOINT_ALL_KEYS",
    "CHECKPOINT_KEY_TYPES",
    "DEFAULT_MODEL_CLASS_NAME",
    "SUPPORTED_MODEL_CLASS_NAMES",
    "TRAINING_METADATA_REQUIRED_KEYS",
    "GRAPH_SCHEMA_REQUIRED_KEYS",
    "HYPERPARAMS_REQUIRED_KEYS",
    "validate_checkpoint_dict",
    "is_valid_checkpoint",
]
