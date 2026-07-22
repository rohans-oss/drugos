"""rl.validate — Scientific Validation (P4-008 modular wrapper + audit #200).

P4-008 ROOT FIX: modular wrapper around the validation functions:
validate_input_schema, validate_environment, preprocess_data,
generate_data_quality_report, ScientificFailureError, check_alert_conditions.
See rl/env.py for the full P4-008 rationale.

Audit #200 ROOT FIX: also exports ``run_scientific_validation_gate``
which runs the FULL scientific validation gate (DOCX §8 V1 launch
criteria):
  - gt_test_auc: GT model AUC on held-out test set (prod threshold: 0.85)
  - rl_auc: RL agent's ranking AUC (must be > 0.5 = better than random)
  - kp_recovery: at least N% of known positives in top-N candidates
  - literature: at least N top predictions supported by PubMed literature

The function returns a dict with ``overall_pass`` (bool) and a per-check
breakdown. The CLI's ``validate`` subcommand calls this and EXITS
NON-ZERO on failure, so CI/Airflow can detect a failing gate and block
deployment.

The previous ``validate.py`` only re-exported schema-validation helpers
(validate_input_schema, etc.) — there was NO function that ran the full
gate and returned a pass/fail verdict. Operators had to write custom
scripts to check launch readiness.

Callers can now import:
    from rl.validate import run_scientific_validation_gate
"""
from __future__ import annotations

from .rl_drug_ranker import (
    validate_input_schema,
    validate_environment,
    preprocess_data,
    generate_data_quality_report,
    validate_canonical_ids,
    ScientificFailureError,
    PipelineMetrics,
    check_alert_conditions,
    # Data dictionary + schema (used for validation)
    DATA_DICTIONARY,
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
    # Audit #200: full scientific validation gate
    run_scientific_validation_gate,
)

__all__ = [
    "validate_input_schema",
    "validate_environment",
    "preprocess_data",
    "generate_data_quality_report",
    "validate_canonical_ids",
    "ScientificFailureError",
    "PipelineMetrics",
    "check_alert_conditions",
    "DATA_DICTIONARY",
    "INPUT_SCHEMA",
    "OUTPUT_SCHEMA",
    # Audit #200
    "run_scientific_validation_gate",
]
