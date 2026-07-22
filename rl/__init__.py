"""
RL-Driven Hypothesis Ranker package -- Team Cosmic (Phase 4).

V4 ROOT FIX (B-F9): ``rl/`` is now a proper installable Python package,
not a single-file module imported via ``sys.path.insert`` hackery.

This eliminates:
  - The bridge's ``sys.path.insert(0, rl_dir)`` mutation (a global
    side-effect that could shadow other ``rl_drug_ranker.py`` files
    on the user's system).
  - The inability to ``pip install`` the RL module.
  - The inability to type-check across module boundaries with mypy.

Phase 3 (``graph_transformer``) and Phase 4 (``rl``) are now
structurally symmetric: both are proper packages, both are imported
via ordinary Python import statements, and the bridge imports Phase 4
exactly the same way it imports Phase 3 -- ``from rl.rl_drug_ranker
import ...``.

This is the structural foundation for "Phase 3 <-> Phase 4 100%
connected": the two phases now share a single import graph with no
path manipulation, no module shadowing, and no implicit global state.
"""
from __future__ import annotations

# ROOT FIX (FORENSIC-AUDIT-I37): aligned version with graph_transformer package.
# Both packages are now versioned together as "4.1.0".
__version__ = "4.1.0"
__schema_version__ = "4.1.0"

# Re-export the most-used symbols so callers can do
#   from rl import PipelineConfig, run_pipeline, KNOWN_POSITIVES
# without having to know the internal layout.
# P4-029 ROOT FIX: added the missing re-exports that the audit found
# were defined in rl_drug_ranker.py but NOT exposed at the package
# level. Callers who did ``from rl import safe_load_input`` got
# ImportError, forcing them to use the longer ``from rl.rl_drug_ranker
# import safe_load_input`` form. This broke the package's public API
# contract and made the rl package feel half-built. The fix adds all
# missing public symbols to both the import statement and __all__.
# P4-004 ROOT FIX: KNOWN_POSITIVES and VALIDATED_HYPOTHESES are now
# _LazyList proxy objects in rl_drug_ranker. They CAN be eagerly
# imported here (the proxy is just a thin wrapper — the CSV read only
# fires on first iteration/len/indexing, NOT at import time). This
# preserves the public API (`from rl import KNOWN_POSITIVES`) without
# re-introducing the import-time CSV read.
from .rl_drug_ranker import (
    # Configuration
    RewardConfig,
    PipelineConfig,
    DEFAULT_CONFIG,
    # P0 fix: scientific failure exception
    ScientificFailureError,
    # Constants — KNOWN_POSITIVES and VALIDATED_HYPOTHESES are _LazyList
    # proxies (P4-004): importing them does NOT trigger the CSV read.
    KNOWN_POSITIVES,
    VALIDATED_HYPOTHESES,
    WITHDRAWN_DRUGS,
    CONTROLLED_SUBSTANCES,
    REQUIRED_COLUMNS,
    FEATURE_COLS,
    # Core classes
    RewardFunction,
    DrugRankingEnv,
    RankedCandidate,
    PipelineMetrics,
    # Functions
    compute_reward,
    validate_input_schema,
    preprocess_data,
    generate_fake_data,
    generate_data_quality_report,
    train_agent,
    evaluate_agent,
    compute_auc,
    extract_policy_prob_high,
    literature_crosscheck,
    check_known_positive_recovery,
    save_results,
    run_pipeline,
    # P4-029: previously-missing re-exports (public API completeness)
    # P4-004: also re-export the lazy-load helpers so callers can
    # explicitly trigger/reload the caches.
    get_known_positives,
    get_validated_hypotheses,
    reload_known_positives,
    reload_validated_hypotheses,
    load_validated_hypotheses,
    merge_results,
    safe_load_input,
    split_data,
    setup_logging,
    validate_environment,
    get_device,
    display_top_candidates,
    compute_file_hash,
    sanitize_string,
    flag_controlled_substances,
    redact_proprietary_ids,
    compute_output_hmac,
    save_provenance_metadata,
    check_for_pii,
    log_audit_event,
    generate_output_filename,
    get_secret,
    check_alert_conditions,
    # Data dictionary + schema
    DATA_DICTIONARY,
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
    # P4-005: per-tenant reward weights
    load_reward_weights_for_tenant,
    save_reward_weights_for_tenant,
    apply_tenant_reward_weights,
    DEFAULT_REWARD_WEIGHTS_DIR,
)

# P4-004: KNOWN_POSITIVES and VALIDATED_HYPOTHESES are exposed via the
# package-level __getattr__ below (lazy). They are listed in __all__ so
# `from rl import *` still works, but they are NOT eagerly imported.
__all__ = [
    "RewardConfig",
    "PipelineConfig",
    "DEFAULT_CONFIG",
    "ScientificFailureError",
    "KNOWN_POSITIVES",
    "VALIDATED_HYPOTHESES",  # lazy via __getattr__
    "get_known_positives",
    "get_validated_hypotheses",
    "reload_known_positives",
    "reload_validated_hypotheses",
    "WITHDRAWN_DRUGS",
    "CONTROLLED_SUBSTANCES",
    "REQUIRED_COLUMNS",
    "FEATURE_COLS",
    "RewardFunction",
    "DrugRankingEnv",
    "RankedCandidate",
    "PipelineMetrics",
    "compute_reward",
    "validate_input_schema",
    "preprocess_data",
    "generate_fake_data",
    "generate_data_quality_report",
    "train_agent",
    "evaluate_agent",
    "compute_auc",
    "extract_policy_prob_high",
    "literature_crosscheck",
    "check_known_positive_recovery",
    "save_results",
    "run_pipeline",
    # P4-029: previously-missing symbols (public API completeness)
    "load_validated_hypotheses",
    "merge_results",
    "safe_load_input",
    "split_data",
    "setup_logging",
    "validate_environment",
    "get_device",
    "display_top_candidates",
    "compute_file_hash",
    "sanitize_string",
    "flag_controlled_substances",
    "redact_proprietary_ids",
    "compute_output_hmac",
    "save_provenance_metadata",
    "check_for_pii",
    "log_audit_event",
    "generate_output_filename",
    "get_secret",
    "check_alert_conditions",
    "DATA_DICTIONARY",
    "INPUT_SCHEMA",
    "OUTPUT_SCHEMA",
    # P4-005: per-tenant reward weights
    "load_reward_weights_for_tenant",
    "save_reward_weights_for_tenant",
    "apply_tenant_reward_weights",
    "DEFAULT_REWARD_WEIGHTS_DIR",
    "__version__",
    "__schema_version__",
]
