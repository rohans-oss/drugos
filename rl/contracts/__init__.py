"""rl.contracts package — Phase 4 schema contracts.

TASK 325 ROOT FIX (forensic, root-level):
  This package defines the canonical Phase 4 output schema:
    - ``validated_hypotheses.csv`` format (the final writeback file that
      closes the data flywheel — feeds validated predictions back to
      Phase 3 trainer as new labeled data points).

  Both Phase 4 (writer: ``rl.validate``, ``rl.rl_drug_ranker``) and
  Phase 3 (reader: ``graph_transformer.training.trainer`` for retraining)
  import from this package.

SH-002 ROOT FIX: removed `OUTCOME_INCONCLUSIVE` from the import list.
  That constant was an artifact of the old drift (3-value enum) and is
  NOT a valid outcome. The canonical 4-value enum (validated_positive,
  validated_negative, validated_toxic, invalidated) is imported instead
  via OUTCOME_NEGATIVE and OUTCOME_INVALIDATED.
"""
from __future__ import annotations

from rl.contracts.phase4_schema import (
    VALIDATED_HYPOTHESES_FILENAME,
    VALIDATED_HYPOTHESES_COLUMNS,
    VALIDATED_HYPOTHESES_REQUIRED_COLUMNS,
    VALIDATED_HYPOTHESES_OPTIONAL_COLUMNS,
    VALIDATED_HYPOTHESES_DTYPES,
    OUTCOME_VALUES,
    OUTCOME_POSITIVE,
    OUTCOME_TOXIC,
    # SH-002 ROOT FIX: OUTCOME_INCONCLUSIVE removed (not a valid outcome).
    OUTCOME_NEGATIVE,
    OUTCOME_INVALIDATED,
    VALIDATED_BY_VALUES,
    ColumnSpec as Phase4ColumnSpec,
    OUTCOME_TO_LABEL,
    LABEL_TO_OUTCOME,
    is_valid_outcome,
    is_validated_by,
    validate_validated_hypotheses_row,
    validate_validated_hypotheses_dataframe,
)

__all__ = [
    "VALIDATED_HYPOTHESES_FILENAME",
    "VALIDATED_HYPOTHESES_COLUMNS",
    "VALIDATED_HYPOTHESES_REQUIRED_COLUMNS",
    "VALIDATED_HYPOTHESES_OPTIONAL_COLUMNS",
    "VALIDATED_HYPOTHESES_DTYPES",
    "OUTCOME_VALUES",
    "OUTCOME_POSITIVE",
    "OUTCOME_TOXIC",
    # SH-002 ROOT FIX: OUTCOME_INCONCLUSIVE removed.
    "OUTCOME_NEGATIVE",
    "OUTCOME_INVALIDATED",
    "VALIDATED_BY_VALUES",
    "Phase4ColumnSpec",
    "OUTCOME_TO_LABEL",
    "LABEL_TO_OUTCOME",
    "is_valid_outcome",
    "is_validated_by",
    "validate_validated_hypotheses_row",
    "validate_validated_hypotheses_dataframe",
]
