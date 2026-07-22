"""
Shared schema for validated_hypotheses — re-exports from canonical location.

DEPRECATED: this module is kept for backward compatibility with code that
imports from ``common.validated_hypotheses_schema``. The CANONICAL source
of truth is now ``shared.contracts.writeback``. New code should import
directly from there:

    from shared.contracts.writeback import (
        CANONICAL_VALIDATED_CSV,
        OUTCOME_COL,
        OUTCOME_VALIDATED_POSITIVE,
        OUTCOME_VALIDATED_TOXIC,
        POSITIVE_OUTCOMES,
        PENALTY_OUTCOMES,
    )

This file re-exports every public name from shared.contracts.writeback so
existing imports continue to work without modification. The actual schema
definitions live in shared/contracts/writeback.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make shared.contracts.writeback importable when common/ is imported
# standalone (e.g., by tests that manipulate sys.path). This is defensive —
# if shared/ is already importable (normal case), the insert is a no-op.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Re-export EVERYTHING from the canonical contract.
from shared.contracts.writeback import *  # noqa: F401,F403,E402
from shared.contracts.writeback import (  # noqa: F401,E402  (explicit for IDE)
    CANONICAL_VALIDATED_CSV,
    LEGACY_RL_VALIDATED_CSV,
    get_validated_csv_path,
    ensure_csv_dir,
    OUTCOME_COL,
    DRUG_COL,
    DISEASE_COL,
    TIMESTAMP_COL,
    VALIDATED_BY_COL,
    VALIDATION_STUDY_ID_COL,
    NOTES_COL,
    ORIGINAL_GT_SCORE_COL,
    ORIGINAL_RL_RANK_COL,
    WRITEBACK_VERSION_COL,
    WRITEBACK_CSV_COLUMNS,
    REQUIRED_COLUMNS,
    OUTCOME_VALIDATED_POSITIVE,
    OUTCOME_VALIDATED_TOXIC,
    OUTCOME_VALIDATED_NEGATIVE,
    OUTCOME_INVALIDATED,
    VALID_OUTCOMES,
    POSITIVE_OUTCOMES,
    TOXIC_OUTCOMES,
    NEGATIVE_OUTCOMES,
    BONUS_OUTCOMES,
    PENALTY_OUTCOMES,
    VALIDATED_POSITIVE_BONUS,
    VALIDATED_TOXIC_PENALTY,
    EDGE_VALIDATED_TREATS,
    EDGE_VALIDATED_TOXIC_FOR,
    EDGE_VALIDATED_NEGATIVE_FOR,
    edge_label_for_outcome,
    NEO4J_DRUG_LABEL_PREFERRED,
    NEO4J_DRUG_LABEL_LEGACY,
    NEO4J_DRUG_LABELS,
    NEO4J_DISEASE_LABEL,
    NEO4J_DRUG_ID_PROP,
    NEO4J_DRUG_NAME_PROP,
    NEO4J_DISEASE_ID_PROP,
    NEO4J_DISEASE_NAME_PROP,
    ATOMIC_WRITE_TMP_SUFFIX,
    ATOMIC_WRITE_FSYNC,
    WRITEBACK_VERSION,
)
