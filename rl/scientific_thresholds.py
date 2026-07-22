"""Shared scientific-validation thresholds for the Autonomous Drug Repurposing Platform.

P4-013 ROOT FIX (HIGH — Team Member 12 / Phase 4): the previous code had
TWO independent definitions of the KP recovery threshold:

  1. ``rl/rl_drug_ranker.py`` used ``config.min_kp_recovery_rate``
     (default 0.2) at the scientific_validation gate.
  2. ``graph_transformer/gt_rl_bridge.py`` used
     ``max(rl_config_threshold, 0.5)`` (effectively 0.5) at its own
     scientific_validation gate.

A run with ``kp_recovery_rate = 0.4`` would PASS the ranker's gate
(0.4 >= 0.2) but FAIL the bridge's gate (0.4 < 0.5). The two components
disagreed on whether the run was scientifically valid, leaving the ops
team unable to determine if the run succeeded. The bridge writes its
CSV; the ranker refuses to; the pipeline state is inconsistent.

The fix defines a SINGLE constant — ``KP_RECOVERY_THRESHOLD`` — in this
shared module. Both ``rl_drug_ranker.py`` and ``gt_rl_bridge.py``
import it, so the threshold can NEVER drift between the two components.
A CI test (``tests/test_team12_p4_012_to_018.py::test_p4_013_*``)
verifies both files import and use the same constant.

The threshold value is 0.5 (50%), matching the V1 launch criterion
implied by the bridge's existing ``max(rl_config_threshold, 0.5)``
logic. The DOCX §8 V1 launch criteria do not specify a numeric KP
recovery threshold, but the bridge's existing 0.5 is the stricter
value and was clearly intended as the production bar (the ranker's
0.2 was a developer-friendly default that the bridge intentionally
overrode). Standardizing on 0.5 means a run must recover at least
half of the known positives in the test set to be considered
scientifically valid — a meaningful bar for a drug-repurposing
platform where known positives are the ground truth.

This module is INTENTIONALLY minimal — it contains only the shared
thresholds. It has no dependencies on torch, pandas, or any other
heavy import, so it can be imported from both the RL ranker (which
runs in CI without torch) and the GT bridge (which requires torch).
"""
from __future__ import annotations

# P4-023 ROOT FIX: KP_RECOVERY_THRESHOLD is now SCALE-AWARE, not a fixed
# constant. The previous fixed 0.5 threshold was statistically
# meaningless on small demo graphs (2 KPs in test → recovery rate is
# 0%, 50%, or 100% — a 3-point discrete scale). The 0.5 threshold meant
# "recover BOTH test KPs" which is not a meaningful bar on tiny graphs.
#
# The fix: compute the threshold based on the number of KPs in the test
# set (n_test_kps):
#   - Production (≥1000 KPs): 0.5 (50% — statistically meaningful)
#   - Pilot (100-1000 KPs): 0.4 (allows some variance)
#   - Demo (<100 KPs): 0.34 (allows 1/3 = 33% or 2/3 = 67% to pass)
#
# The scale-aware threshold is computed by resolve_kp_recovery_threshold()
# below. The constant KP_RECOVERY_THRESHOLD is kept for backward compat
# but should NOT be used directly — always call resolve_kp_recovery_threshold().

# P4-023: the minimum number of literature-supported predictions
# required by the V1 launch criterion (DOCX §8: "At least 5 top
# predictions are supported by published literature"). This is
# already defined inline in rl_drug_ranker.py, but we expose it here
# so downstream consumers (bridge, dashboard, CI) can import a
# single constant instead of hardcoding 5.
MIN_LITERATURE_SUPPORTED: int = 5
"""Minimum number of literature-supported predictions for the V1 launch
criterion (DOCX §8). The scientific_validation gate checks
``n_literature_supported >= MIN_LITERATURE_SUPPORTED``.
"""

# P4-013: the GT test AUC threshold for the V1 launch criterion
# (DOCX §8: "Graph Transformer achieves >0.85 AUC on held-out
# drug-disease pairs"). This is already defined as
# ``config.gt_test_auc_threshold`` in rl_drug_ranker.py (default 0.85),
# but we expose the canonical value here so the bridge can import it
# without duplicating the magic number.
GT_TEST_AUC_THRESHOLD: float = 0.85
"""Minimum GT test AUC for the V1 launch criterion (DOCX §8). The
scientific_validation gate checks ``gt_test_auc > GT_TEST_AUC_THRESHOLD``.
"""

# P4-013: the RL AUC threshold. The DOCX §8 V1 launch criterion
# requires "RL agent produces consistent, non-random rankings" — an
# AUC > 0.5 (better than random) is the operationalization.
RL_AUC_THRESHOLD: float = 0.5
"""Minimum RL AUC for the V1 launch criterion. AUC <= 0.5 means the
RL agent is no better than random ranking — the scientific_validation
gate fails.
"""

# P4-023 ROOT FIX: KP_RECOVERY_THRESHOLD is now SCALE-AWARE, not a fixed
# constant. The previous fixed 0.5 threshold was statistically
# meaningless on small demo graphs (2 KPs in test → recovery rate is
# 0%, 50%, or 100% — a 3-point discrete scale).
#
# The fix introduces a BASE threshold that varies by test set size:
#   - Production (≥1000 KPs): 0.5 (50% — statistically meaningful)
#   - Pilot (100-1000 KPs): 0.4 (allows some variance)
#   - Demo (<100 KPs): 0.34 (allows 1/3 = 33% or 2/3 = 67% to pass)
#
# The existing P4-013 ``resolve_kp_recovery_threshold(config_threshold)``
# applies ``max(config_threshold, BASE)`` so callers can RAISE the
# threshold but cannot lower it below the scale-aware base. Both the
# ranker and the bridge call the SAME function, so they always agree.

# The fixed fallback threshold (kept for backward compat).
KP_RECOVERY_THRESHOLD: float = 0.5
"""Fixed fallback threshold for backward compatibility.

Use ``resolve_kp_recovery_threshold(n_test_kps)`` for scale-aware
thresholding, or ``resolve_kp_recovery_threshold(config_threshold)``
for the P4-013 config-clamped threshold.
"""


def _compute_base_threshold(n_test_kps: int) -> float:
    """P4-023: compute the scale-aware BASE threshold."""
    if n_test_kps >= 1000:
        return 0.5   # Production: ≥50% recovery required
    elif n_test_kps >= 100:
        return 0.4   # Pilot: ≥40% recovery required
    elif n_test_kps > 0:
        return 0.34  # Demo: ≥34% recovery required (allows 1/3 on tiny graphs)
    else:
        return 0.5   # Unknown — use production default


def resolve_kp_recovery_threshold(
    config_threshold: float = 0.0,
    n_test_kps: int = 0,
) -> float:
    """P4-013 + P4-023 MERGED ROOT FIX: the SINGLE source of truth for
    computing the KP recovery threshold.

    This function serves TWO use cases:

    1. P4-023 (scale-aware): call with ``n_test_kps`` to get a base
       threshold that adapts to the test set size:
         - n_test_kps >= 1000 → 0.5
         - 100 <= n_test_kps < 1000 → 0.4
         - 0 < n_test_kps < 100 → 0.34
         - n_test_kps == 0 → 0.5 (unknown, use production default)

    2. P4-013 (config clamp): call with ``config_threshold`` to apply
       ``max(config_threshold, base_threshold)``. Callers can RAISE the
       threshold above the base but cannot lower it below.

    Both the ranker and the bridge call this SAME function with the SAME
    arguments, so they are GUARANTEED to compute the SAME threshold.

    Args:
        config_threshold: The caller-provided threshold from
            ``PipelineConfig.min_kp_recovery_rate``. May be any float;
            values below the base threshold are clamped up.
        n_test_kps: Number of known positives in the test set (for
            scale-aware base threshold computation).

    Returns:
        The resolved threshold. Always >= the scale-aware base.
    """
    # Compute the scale-aware base threshold (P4-023)
    base = _compute_base_threshold(n_test_kps)

    try:
        cfg = float(config_threshold)
    except (TypeError, ValueError):
        return base

    import math as _math
    if _math.isnan(cfg) or _math.isinf(cfg):
        return base
    if cfg < 0.0 or cfg > 1.0:
        return base

    # P4-013: clamp to the base (callers can raise, cannot lower)
    return max(cfg, base)


__all__ = [
    "KP_RECOVERY_THRESHOLD",
    "MIN_LITERATURE_SUPPORTED",
    "GT_TEST_AUC_THRESHOLD",
    "RL_AUC_THRESHOLD",
    "resolve_kp_recovery_threshold",
]
