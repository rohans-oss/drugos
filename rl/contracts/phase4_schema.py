"""Canonical Phase 4 output schema — DELEGATES to shared.contracts.writeback.

SH-002 + SH-003 ROOT FIX (forensic, root-level):
  Previously this module REDEFINED the outcome enum and CSV column names,
  causing CONTRACT DRIFT against ``shared/contracts/writeback.py``:

    - SH-002: this module declared 3 outcomes (validated_positive,
      validated_toxic, validated_inconclusive) while the shared contract
      declared 4 (validated_positive, validated_toxic,
      validated_negative, invalidated). The 3-value set could never
      represent `validated_negative` or `invalidated` — Phase 3's
      trainer would silently drop those rows during retraining,
      corrupting the data flywheel.

    - SH-003: this module declared CSV columns (drug_id, disease_id,
      drug_name, disease_name, score, ...) that did NOT match the
      shared contract's columns (drug, disease, outcome, validated_by,
      validation_study_id, validated_at, notes, original_gt_score,
      original_rl_rank, writeback_version). The Phase 4 writer
      (phase4/writeback.py) writes the SHARED schema to disk, so any
      reader using this module's ColumnSpec list would fail to find
      the required columns and reject every row.

  ROOT FIX: this module now DELEGATES the canonical outcome enum and
  CSV column list to ``shared.contracts.writeback`` (the AUTHORITATIVE
  source per its docstring and per actual usage by both the writer
  ``phase4/writeback.py`` and the reader
  ``graph_transformer/training/trainer.py``). The ``ColumnSpec`` dataclass
  and validators are KEPT (they're useful for runtime row validation),
  but they're rebuilt from the SHARED column list so they can never drift
  again.

  Backward-compat aliases are preserved so existing imports
  (``OUTCOME_POSITIVE``, ``OUTCOME_VALUES``, ``ColumnSpec``) keep working.
  The misleading ``OUTCOME_INCONCLUSIVE`` constant is REMOVED — it never
  existed in the canonical contract and was the root cause of the drift.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Make shared.contracts.writeback importable when rl/contracts is imported
# standalone (e.g., by tests that manipulate sys.path). Defensive — if
# shared/ is already importable (normal case), the insert is a no-op.
# -----------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# -----------------------------------------------------------------------------
# CANONICAL IMPORTS — single source of truth.
# -----------------------------------------------------------------------------
# SH-002/SH-003 ROOT FIX: import outcomes and column names from the
# shared contract. ANY change to the outcome enum or column list MUST
# be made in shared/contracts/writeback.py — this module mirrors it.
from shared.contracts.writeback import (  # noqa: E402
    CANONICAL_VALIDATED_CSV,
    LEGACY_RL_VALIDATED_CSV,
    DRUG_COL,
    DISEASE_COL,
    OUTCOME_COL,
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
    WRITEBACK_VERSION,
    get_validated_csv_path,
    ensure_csv_dir,
    get_writer_path,
    get_reader_path,
)


# =============================================================================
# Canonical filename (re-exported for back-compat)
# =============================================================================
VALIDATED_HYPOTHESES_FILENAME: str = os.path.basename(CANONICAL_VALIDATED_CSV)


# =============================================================================
# Outcome enum — DELEGATES to shared contract (SH-002 ROOT FIX)
# =============================================================================
# Backward-compat aliases. These point to the SAME string objects as the
# shared contract — there is no possibility of drift.
OUTCOME_POSITIVE: str = OUTCOME_VALIDATED_POSITIVE
OUTCOME_TOXIC: str = OUTCOME_VALIDATED_TOXIC
OUTCOME_NEGATIVE: str = OUTCOME_VALIDATED_NEGATIVE      # NEW (was missing)
OUTCOME_INVALIDATED: str = OUTCOME_INVALIDATED           # NEW (was missing)

# SH-002 ROOT FIX: was previously a 3-tuple missing
# `validated_negative` and `invalidated`. Now mirrors the shared
# contract's 4-value set EXACTLY.
OUTCOME_VALUES: Tuple[str, ...] = tuple(VALID_OUTCOMES)

# Outcome -> human-readable label (for display in the frontend).
OUTCOME_TO_LABEL: Dict[str, str] = {
    OUTCOME_VALIDATED_POSITIVE: "Validated Positive",
    OUTCOME_VALIDATED_TOXIC: "Validated Toxic",
    OUTCOME_VALIDATED_NEGATIVE: "Validated Negative",
    OUTCOME_INVALIDATED: "Invalidated",
}

# Reverse lookup.
LABEL_TO_OUTCOME: Dict[str, str] = {v: k for k, v in OUTCOME_TO_LABEL.items()}


# =============================================================================
# Validated-by enum — the allowed values for the ``validated_by`` column
# =============================================================================
# The validator ID encodes WHO validated the hypothesis and HOW. The
# prefix determines the trust level (wet_lab > clinical_study > literature).
VALIDATED_BY_VALUES: Tuple[str, ...] = (
    "wet_lab",          # in-vitro / in-vivo wet-lab validation
    "clinical_study",   # human clinical trial
    "literature",       # published paper support
    "expert_review",    # domain-expert manual review
    "automated",        # automated pipeline (e.g. literature search)
)


# =============================================================================
# ColumnSpec — typed specification for one validated_hypotheses column
# =============================================================================


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single validated_hypotheses column.

    Attributes
    ----------
    name : str
        Canonical column name (case-sensitive, exact match required).
    dtype : str
        Pandas dtype hint: ``"string"``, ``"float64"``, ``"int64"``,
        ``"bool"``, ``"object"``.
    nullable : bool
        True if the column may contain NULL/NaN values.
    description : str
        Human-readable description.
    """

    name: str
    dtype: str = "string"
    nullable: bool = False
    description: str = ""


# =============================================================================
# REQUIRED columns — DERIVED from shared contract (SH-003 ROOT FIX)
# =============================================================================
# SH-003 ROOT FIX: the previous hardcoded list (drug_id, disease_id,
# drug_name, disease_name, score, outcome, validated_by, validated_at)
# DID NOT match the shared contract's column names (drug, disease,
# outcome, validated_by, validation_study_id, validated_at, notes,
# original_gt_score, original_rl_rank, writeback_version). The Phase 4
# writer writes the SHARED schema, so any reader using the old list
# would fail to find required columns.
#
# We now BUILD the ColumnSpec list from the shared contract's
# WRITEBACK_CSV_COLUMNS so they can never drift. The dtype map is
# sourced from shared.contracts.writeback.WRITEBACK_DTYPES.
_SHARED_DTYPES: Dict[str, str] = {
    DRUG_COL: "string",
    DISEASE_COL: "string",
    OUTCOME_COL: "string",
    VALIDATED_BY_COL: "string",
    VALIDATION_STUDY_ID_COL: "string",
    TIMESTAMP_COL: "string",
    NOTES_COL: "string",
    ORIGINAL_GT_SCORE_COL: "float64",
    ORIGINAL_RL_RANK_COL: "int64",
    WRITEBACK_VERSION_COL: "string",
}

_SHARED_NULLABLE: Dict[str, bool] = {
    DRUG_COL: False,
    DISEASE_COL: False,
    OUTCOME_COL: False,
    VALIDATED_BY_COL: False,
    VALIDATION_STUDY_ID_COL: True,
    TIMESTAMP_COL: False,
    NOTES_COL: True,
    ORIGINAL_GT_SCORE_COL: True,
    ORIGINAL_RL_RANK_COL: True,
    WRITEBACK_VERSION_COL: False,
}

_SHARED_DESCRIPTIONS: Dict[str, str] = {
    DRUG_COL: "Drug identifier (canonical name from Phase 1).",
    DISEASE_COL: "Disease identifier (canonical name from Phase 1).",
    OUTCOME_COL: (
        "One of: validated_positive | validated_toxic | "
        "validated_negative | invalidated."
    ),
    VALIDATED_BY_COL: (
        "Validator ID prefix (wet_lab | clinical_study | literature | "
        "expert_review | automated)."
    ),
    VALIDATION_STUDY_ID_COL: "Optional study identifier (e.g. NCT number).",
    TIMESTAMP_COL: "ISO 8601 timestamp of validation (UTC).",
    NOTES_COL: "Free-text notes from the validator.",
    ORIGINAL_GT_SCORE_COL: "Original Graph Transformer score [0, 1].",
    ORIGINAL_RL_RANK_COL: "Original RL rank (1-indexed).",
    WRITEBACK_VERSION_COL: "Writeback schema version (semver).",
}


def _build_column_specs() -> Tuple[ColumnSpec, ...]:
    """Build ColumnSpec list from the shared contract's column list.

    This GUARANTEES that this module's ColumnSpec list matches the
    shared contract's WRITEBACK_CSV_COLUMNS — they're built from the
    SAME source list.
    """
    specs: List[ColumnSpec] = []
    for col_name in WRITEBACK_CSV_COLUMNS:
        specs.append(
            ColumnSpec(
                name=col_name,
                dtype=_SHARED_DTYPES.get(col_name, "string"),
                nullable=_SHARED_NULLABLE.get(col_name, True),
                description=_SHARED_DESCRIPTIONS.get(col_name, ""),
            )
        )
    return tuple(specs)


# Required columns (non-nullable per shared contract).
VALIDATED_HYPOTHESES_REQUIRED_COLUMNS: Tuple[ColumnSpec, ...] = tuple(
    spec for spec in _build_column_specs() if not spec.nullable
)

# Optional columns (nullable per shared contract).
VALIDATED_HYPOTHESES_OPTIONAL_COLUMNS: Tuple[ColumnSpec, ...] = tuple(
    spec for spec in _build_column_specs() if spec.nullable
)

# All columns in canonical order (mirrors shared.WRITEBACK_CSV_COLUMNS).
VALIDATED_HYPOTHESES_COLUMNS: Tuple[ColumnSpec, ...] = _build_column_specs()

# Flat list of column names (for pandas usecols / dtype construction).
# SH-003 ROOT FIX: this now EXACTLY matches shared.WRITEBACK_CSV_COLUMNS.
VALIDATED_HYPOTHESES_COLUMN_NAMES: Tuple[str, ...] = tuple(
    c.name for c in VALIDATED_HYPOTHESES_COLUMNS
)

# Dtype dict for pandas.read_csv(dtype=...).
VALIDATED_HYPOTHESES_DTYPES: Dict[str, str] = {
    c.name: c.dtype for c in VALIDATED_HYPOTHESES_COLUMNS
}


# =============================================================================
# Validators
# =============================================================================


def is_valid_outcome(value: str) -> bool:
    """Return True if ``value`` is a valid outcome enum value.

    SH-002 ROOT FIX: now checks against the 4-value shared contract
    (was previously 3 values, missing validated_negative + invalidated).
    """
    return value in OUTCOME_VALUES


def is_validated_by(value: str) -> bool:
    """Return True if ``value`` is a valid validated_by prefix.

    Accepts both bare prefixes (``"wet_lab"``) and qualified IDs
    (``"wet_lab:partner_A"``) — the prefix before the first colon is
    what matters.
    """
    if not isinstance(value, str):
        return False
    prefix = value.split(":", 1)[0]
    return prefix in VALIDATED_BY_VALUES


def validate_validated_hypotheses_row(row: Dict[str, Any]) -> List[str]:
    """Validate a single validated_hypotheses row (as a dict).

    Returns a list of error messages. Empty list = valid row.
    """
    errors: List[str] = []

    # Check 1: all required columns present and non-null.
    for col in VALIDATED_HYPOTHESES_REQUIRED_COLUMNS:
        if col.name not in row:
            errors.append(f"Missing required column {col.name!r}.")
        elif row[col.name] is None:
            errors.append(f"Required column {col.name!r} is null.")
        elif isinstance(row[col.name], str) and not row[col.name].strip():
            errors.append(f"Required column {col.name!r} is empty.")

    # Check 2: outcome is a valid enum value.
    outcome = row.get(OUTCOME_COL)
    if outcome is not None and not is_valid_outcome(outcome):
        errors.append(
            f"outcome {outcome!r} is not valid. "
            f"Must be one of: {list(OUTCOME_VALUES)}."
        )

    # Check 3: validated_by is a valid prefix.
    vb = row.get(VALIDATED_BY_COL)
    if vb is not None and not is_validated_by(vb):
        errors.append(
            f"validated_by {vb!r} is not valid. "
            f"Prefix must be one of: {list(VALIDATED_BY_VALUES)}."
        )

    # Check 4: original_gt_score is in [0, 1] (if present).
    score = row.get(ORIGINAL_GT_SCORE_COL)
    if score is not None and score != "":
        try:
            score_f = float(score)
            if score_f < 0.0 or score_f > 1.0:
                errors.append(
                    f"original_gt_score {score_f} is out of range [0, 1]."
                )
        except (TypeError, ValueError):
            errors.append(f"original_gt_score {score!r} is not a valid float.")

    # Check 5: validated_at is ISO 8601 parseable.
    validated_at = row.get(TIMESTAMP_COL)
    if validated_at is not None:
        try:
            from datetime import datetime
            datetime.fromisoformat(str(validated_at).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            errors.append(
                f"validated_at {validated_at!r} is not a valid ISO 8601 timestamp."
            )

    return errors


def validate_validated_hypotheses_dataframe(df: Any) -> List[str]:
    """Validate a pandas DataFrame of validated_hypotheses rows.

    Returns a list of error messages. Empty list = valid DataFrame.
    """
    errors: List[str] = []

    if df is None:
        return ["DataFrame is None."]
    if not hasattr(df, "columns"):
        return [f"Expected a pandas DataFrame, got {type(df).__name__}."]

    actual_cols = set(df.columns)

    # Check 1: all required columns present.
    for col in VALIDATED_HYPOTHESES_REQUIRED_COLUMNS:
        if col.name not in actual_cols:
            errors.append(f"Missing required column {col.name!r}.")

    # Check 2: no unknown columns (strict).
    known_cols = set(VALIDATED_HYPOTHESES_COLUMN_NAMES)
    for col in actual_cols:
        if col not in known_cols:
            errors.append(
                f"WARNING: Unknown column {col!r}. "
                f"Known columns: {list(VALIDATED_HYPOTHESES_COLUMN_NAMES)}."
            )

    # Check 3: outcome column only contains valid enum values.
    if OUTCOME_COL in actual_cols:
        invalid = df.loc[
            df[OUTCOME_COL].notna() & ~df[OUTCOME_COL].isin(OUTCOME_VALUES),
            OUTCOME_COL,
        ]
        if len(invalid) > 0:
            unique_invalid = invalid.unique().tolist()
            errors.append(
                f"outcome column contains {len(invalid)} rows with invalid values: "
                f"{unique_invalid}. Must be one of: {list(OUTCOME_VALUES)}."
            )

    # Check 4: validated_by column only contains valid prefixes.
    if VALIDATED_BY_COL in actual_cols:
        invalid_mask = df[VALIDATED_BY_COL].notna() & ~df[VALIDATED_BY_COL].apply(
            lambda x: is_validated_by(x) if isinstance(x, str) else False
        )
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            unique_invalid = df.loc[invalid_mask, VALIDATED_BY_COL].unique().tolist()
            errors.append(
                f"validated_by column contains {invalid_count} rows with invalid "
                f"prefixes: {unique_invalid}. Prefix must be one of: "
                f"{list(VALIDATED_BY_VALUES)}."
            )

    # Check 5: original_gt_score column in [0, 1] (if present).
    if ORIGINAL_GT_SCORE_COL in actual_cols:
        scores = df[ORIGINAL_GT_SCORE_COL].dropna()
        if len(scores) > 0:
            try:
                scores_f = scores.astype(float)
                bad = scores_f[(scores_f < 0.0) | (scores_f > 1.0)]
                if len(bad) > 0:
                    errors.append(
                        f"original_gt_score column contains {len(bad)} rows outside [0, 1]. "
                        f"Min={scores_f.min()}, Max={scores_f.max()}."
                    )
            except (TypeError, ValueError):
                errors.append("original_gt_score column contains non-numeric values.")

    return errors


# =============================================================================
# __all__ — explicit export list
# =============================================================================
__all__ = [
    # Filename
    "VALIDATED_HYPOTHESES_FILENAME",
    # Outcomes (delegated to shared)
    "OUTCOME_POSITIVE",
    "OUTCOME_TOXIC",
    "OUTCOME_NEGATIVE",
    "OUTCOME_INVALIDATED",
    "OUTCOME_VALUES",
    "OUTCOME_TO_LABEL",
    "LABEL_TO_OUTCOME",
    # Validated-by
    "VALIDATED_BY_VALUES",
    # Columns
    "ColumnSpec",
    "VALIDATED_HYPOTHESES_REQUIRED_COLUMNS",
    "VALIDATED_HYPOTHESES_OPTIONAL_COLUMNS",
    "VALIDATED_HYPOTHESES_COLUMNS",
    "VALIDATED_HYPOTHESES_COLUMN_NAMES",
    "VALIDATED_HYPOTHESES_DTYPES",
    # Validators
    "is_valid_outcome",
    "is_validated_by",
    "validate_validated_hypotheses_row",
    "validate_validated_hypotheses_dataframe",
    # Re-exports from shared (for back-compat with code that imported
    # these from rl.contracts.phase4_schema)
    "CANONICAL_VALIDATED_CSV",
    "LEGACY_RL_VALIDATED_CSV",
    "DRUG_COL",
    "DISEASE_COL",
    "OUTCOME_COL",
    "TIMESTAMP_COL",
    "VALIDATED_BY_COL",
    "VALIDATION_STUDY_ID_COL",
    "NOTES_COL",
    "ORIGINAL_GT_SCORE_COL",
    "ORIGINAL_RL_RANK_COL",
    "WRITEBACK_VERSION_COL",
    "WRITEBACK_CSV_COLUMNS",
    "REQUIRED_COLUMNS",
    "OUTCOME_VALIDATED_POSITIVE",
    "OUTCOME_VALIDATED_TOXIC",
    "OUTCOME_VALIDATED_NEGATIVE",
    "OUTCOME_INVALIDATED",
    "VALID_OUTCOMES",
    "POSITIVE_OUTCOMES",
    "TOXIC_OUTCOMES",
    "NEGATIVE_OUTCOMES",
    "BONUS_OUTCOMES",
    "PENALTY_OUTCOMES",
    "WRITEBACK_VERSION",
    "get_validated_csv_path",
    "ensure_csv_dir",
    "get_writer_path",
    "get_reader_path",
]
