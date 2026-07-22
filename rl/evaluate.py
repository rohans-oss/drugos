"""rl.evaluate — Evaluation & AUC (P4-008 modular wrapper + audit #199).

P4-008 ROOT FIX: modular wrapper around the evaluation functions:
evaluate_agent, compute_auc, extract_policy_prob_high,
check_known_positive_recovery, literature_crosscheck, display_top_candidates,
split_data. See rl/env.py for the full P4-008 rationale.

Audit #199 ROOT FIX: also exports ``produce_evaluation_report`` which
produces a complete JSON-serializable evaluation report with:
  - AUC (from compute_auc)
  - KP recovery (from check_known_positive_recovery)
  - Top-N candidates with per-candidate feature values
  - Model info (class, device, policy arch)
  - Timestamp (timezone-aware UTC)

The previous ``evaluate_agent`` only returned a ``List[RankedCandidate]``.
Most callers just printed AUC and discarded the rest. The DOCX §8 V1
launch contract requires all of: AUC, KP recovery, top-N candidates,
AND per-candidate feature values (for the dashboard's Hypothesis Detail
View). ``produce_evaluation_report`` assembles all of that in one call.

Callers can now import:
    from rl.evaluate import evaluate_agent, compute_auc, produce_evaluation_report
"""
from __future__ import annotations

from .rl_drug_ranker import (
    evaluate_agent,
    compute_auc,
    extract_policy_prob_high,
    check_known_positive_recovery,
    literature_crosscheck,
    display_top_candidates,
    split_data,
    # Audit #199: full JSON evaluation report
    produce_evaluation_report,
)

__all__ = [
    "evaluate_agent",
    "compute_auc",
    "extract_policy_prob_high",
    "check_known_positive_recovery",
    "literature_crosscheck",
    "display_top_candidates",
    "split_data",
    # Audit #199
    "produce_evaluation_report",
]
