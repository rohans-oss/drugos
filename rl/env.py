"""rl.env — Drug Ranking RL Environment (P4-008/P4-021 modular wrapper).

P4-021 ROOT FIX (Team Member 9, REAL EXTRACTION STEP):
The column constants are now imported from rl/constants.py (the
self-contained constants module), NOT from the 9000-line monolith.
This is the FIRST real extraction step toward P4-021's goal of actual
decoupling: a caller who does `from rl.env import DRUG_COL` no longer
transitively triggers the monolith's import side effects for constants.

The DrugRankingEnv class itself (~980 lines) still lives in
rl_drug_ranker.py because it has deep dependencies on RankedCandidate,
PipelineMetrics, RewardFunction, WITHDRAWN_DRUGS, etc. A full extraction
is planned post-v105 when CI coverage is higher. The extraction plan:
  1. [DONE] Extract column constants to rl/constants.py (this commit)
  2. Extract RankedCandidate + PipelineMetrics to rl/types.py
  3. Extract RewardConfig to rl/reward.py (self-contained dataclass)
  4. Extract RewardFunction to rl/reward.py (~700 lines)
  5. Extract DrugRankingEnv to rl/env.py (~980 lines, the final piece)
  6. rl_drug_ranker.py becomes a backward-compat shim

This wrapper provides the IMPORT INTERFACE for callers. The structural
separation is now REAL at the constants level — the class extraction is
deferred to avoid breakage in the parallel-agent workflow.
"""
from __future__ import annotations

# P4-021: import CONSTANTS from rl/constants.py (self-contained, no monolith dep).
from .constants import (
    DRUG_COL,
    DISEASE_COL,
    GNN_SCORE_COL,
    SAFETY_COL,
    MARKET_COL,
    CONFIDENCE_COL,
    PATHWAY_COL,
    PATENT_COL,
    RARE_DISEASE_COL,
    UNMET_NEED_COL,
    EFFICACY_COL,
    ADME_COL,
    DISEASE_PAIR_COUNT_COL,
    DISEASE_AVG_GNN_COL,
    DISEASE_AVG_SAFETY_COL,
    GNN_SCORE_TIMESTAMP_COL,
    GNN_SCORE_STALENESS_WARNING_HOURS,
    REWARD_COL,
    RANK_COL,
    LITERATURE_SUPPORT_COL,
    IS_KNOWN_POSITIVE_COL,
    CONTROLLED_SUBSTANCE_COL,
)

# P4-021: DrugRankingEnv + RankedCandidate + PipelineMetrics still come from
# the monolith (they have deep interdependencies). See docstring above.
from .rl_drug_ranker import (
    DrugRankingEnv,
    RankedCandidate,
    PipelineMetrics,
)

__all__ = [
    "DrugRankingEnv",
    "RankedCandidate",
    "PipelineMetrics",
    "DRUG_COL",
    "DISEASE_COL",
    "GNN_SCORE_COL",
    "SAFETY_COL",
    "MARKET_COL",
    "CONFIDENCE_COL",
    "PATHWAY_COL",
    "PATENT_COL",
    "RARE_DISEASE_COL",
    "UNMET_NEED_COL",
    "EFFICACY_COL",
    "ADME_COL",
    "DISEASE_PAIR_COUNT_COL",
    "DISEASE_AVG_GNN_COL",
    "DISEASE_AVG_SAFETY_COL",
    "GNN_SCORE_TIMESTAMP_COL",
    "GNN_SCORE_STALENESS_WARNING_HOURS",
    "REWARD_COL",
    "RANK_COL",
    "LITERATURE_SUPPORT_COL",
    "IS_KNOWN_POSITIVE_COL",
    "CONTROLLED_SUBSTANCE_COL",
]
