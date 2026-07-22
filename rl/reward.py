"""rl.reward — Reward Function & Config (P4-008/P4-021 modular wrapper).

P4-021 ROOT FIX (Team Member 9, REAL EXTRACTION STEP):
The column constants (FEATURE_COLS, REQUIRED_COLUMNS) are now imported
from rl/constants.py (the self-contained constants module), NOT from
the 9000-line monolith. This is the FIRST real extraction step toward
P4-021's goal of actual decoupling.

The RewardConfig and RewardFunction classes still live in
rl_drug_ranker.py because they have deep dependencies on the
withdrawn-drug sets, pandas/numpy, and the column constants. A full
extraction is planned post-v105 when CI coverage is higher.

Callers can import:
    from rl.reward import RewardFunction, RewardConfig, compute_reward
    from rl.reward import load_reward_weights_for_tenant, apply_tenant_reward_weights
"""
from __future__ import annotations

# P4-021: import CONSTANTS from rl/constants.py (self-contained, no monolith dep).
from .constants import FEATURE_COLS, REQUIRED_COLUMNS

# P4-021: RewardConfig + RewardFunction still come from the monolith (they have
# deep interdependencies on the withdrawn-drug sets, pandas/numpy, etc.).
from .rl_drug_ranker import (
    RewardConfig,
    RewardFunction,
    compute_reward,
    # P4-005: per-tenant reward weights
    load_reward_weights_for_tenant,
    save_reward_weights_for_tenant,
    apply_tenant_reward_weights,
    DEFAULT_REWARD_WEIGHTS_DIR,
    # Constants used by the reward function (still from monolith — these are
    # scientific guardrail sets, not column names, so they stay with the
    # reward logic until RewardFunction is extracted).
    WITHDRAWN_DRUGS,
    INDICATION_WITHDRAWN_DRUGS,
    CONTROLLED_SUBSTANCES,
    DEFAULT_PROPRIETARY_PREFIXES,
)

__all__ = [
    "RewardConfig",
    "RewardFunction",
    "compute_reward",
    "load_reward_weights_for_tenant",
    "save_reward_weights_for_tenant",
    "apply_tenant_reward_weights",
    "DEFAULT_REWARD_WEIGHTS_DIR",
    "FEATURE_COLS",
    "REQUIRED_COLUMNS",
    "WITHDRAWN_DRUGS",
    "INDICATION_WITHDRAWN_DRUGS",
    "CONTROLLED_SUBSTANCES",
    "DEFAULT_PROPRIETARY_PREFIXES",
]
