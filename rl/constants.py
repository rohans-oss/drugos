"""rl.constants — Column constants for the RL ranker (P4-021 REAL extraction).

P4-021 ROOT FIX (Team Member 9): the previous code had ALL column constants
defined inline in the 9,361-line rl_drug_ranker.py monolith. The wrapper
modules (rl/env.py, rl/reward.py, etc.) were re-export shims that imported
FROM the monolith — meaning any change to the monolith risked breaking ALL
wrappers simultaneously, and the "modular separation" was cosmetic.

This file is the FIRST REAL extraction step toward P4-021's goal of actual
decoupling. It moves the SELF-CONTAINED column constants (no dependencies on
torch, pandas, or any class) into their own module. Both rl_drug_ranker.py
AND the wrapper modules import from here, so:

  1. A change to a column name lives in ONE place (this file).
  2. The wrapper modules no longer transitively depend on the monolith
     for constants — they can be imported without triggering the 9000-line
     monolith's import side effects.
  3. This proves the modular structure is REAL, not cosmetic.

TASK 328 ROOT FIX (forensic, root-level):
  The 6 CANONICAL RL feature names (gnn_score, safety_score, market_score,
  efficacy_score, patent_score, adme_score) are now imported from
  shared/contracts/feature_names.py — the SINGLE source of truth shared
  with Phase 3's bridge (which writes these features to the RL input CSV).
  This eliminates the schema drift where Phase 3 writes ``gnn_score`` but
  Phase 4 reads ``gt_score`` (or vice versa) — the contract makes any
  rename a compile-time error on both sides.

  The non-canonical constants (CONFIDENCE_COL, PATHWAY_COL, etc.) remain
  defined here because they are Phase 4-internal (not written by Phase 3).

Future extraction steps (post-v105, when CI coverage is higher):
  - rl/types.py — RankedCandidate, PipelineMetrics dataclasses
  - rl/reward.py — RewardConfig + RewardFunction (move from monolith)
  - rl/env.py — DrugRankingEnv (move from monolith)
  - rl_drug_ranker.py becomes a backward-compat shim

This file is INTENTIONALLY minimal — only constants, no logic, no imports
beyond stdlib typing + the shared contract.
"""
from __future__ import annotations

from typing import List

# =============================================================================
# TASK 328 ROOT FIX: import the 6 CANONICAL RL feature names from the
# shared contract. Both Phase 3 bridge (writer) and Phase 4 env (reader)
# import from the same module — eliminating the schema drift that
# previously caused silent zero-feature bugs.
# =============================================================================
try:
    from shared.contracts.feature_names import (
        FEATURE_GNN_SCORE as GNN_SCORE_COL,
        FEATURE_SAFETY_SCORE as SAFETY_COL,
        FEATURE_MARKET_SCORE as MARKET_COL,
        FEATURE_EFFICACY_SCORE as EFFICACY_COL,
        FEATURE_PATENT_SCORE as PATENT_COL,
        FEATURE_ADME_SCORE as ADME_COL,
    )
    _FEATURE_NAMES_FROM_CONTRACT = True
except ImportError:
    # Degraded mode: shared.contracts.feature_names not importable.
    # Fall back to hardcoded values that match the contract. The contract
    # consistency test will fail in CI, surfacing the misconfiguration.
    GNN_SCORE_COL: str = "gnn_score"
    SAFETY_COL: str = "safety_score"
    MARKET_COL: str = "market_score"
    EFFICACY_COL: str = "efficacy_score"
    PATENT_COL: str = "patent_score"
    ADME_COL: str = "adme_score"
    _FEATURE_NAMES_FROM_CONTRACT = False

# ============================================================================
# CORE IDENTIFIER COLUMNS
# ============================================================================
DRUG_COL: str = "drug"
DISEASE_COL: str = "disease"

# ============================================================================
# FEATURE COLUMNS (observed by the RL agent)
# ============================================================================
# The 6 canonical features (above) are imported from the shared contract.
# The remaining feature columns are Phase 4-internal (not written by
# Phase 3) so they stay defined here.
CONFIDENCE_COL: str = "confidence"
PATHWAY_COL: str = "pathway_score"

# PATENT_COL semantics: For REPURPOSING, OFF-patent = better (cheaper,
# generic availability, no IP blocking by original manufacturer).
# (PATENT_COL is imported from the shared contract above.)

RARE_DISEASE_COL: str = "rare_disease_flag"

# Renamed from existing_drugs_score: previous name was actively misleading.
UNMET_NEED_COL: str = "unmet_need_score"

# (EFFICACY_COL and ADME_COL are imported from the shared contract above.)

# ============================================================================
# DISEASE-CONTEXT FEATURES (added at runtime by the env)
# ============================================================================
DISEASE_PAIR_COUNT_COL: str = "disease_pair_count"
DISEASE_AVG_GNN_COL: str = "disease_avg_gnn"
DISEASE_AVG_SAFETY_COL: str = "disease_avg_safety"

# ============================================================================
# GNN SCORE STALENESS TRACKING (P4-007)
# ============================================================================
# The input CSV may include this column (ISO 8601 format) to indicate when
# the gnn_score was computed by the Phase 3 GT model. The DrugRankingEnv
# checks the timestamp at init time and logs a WARNING if the gnn_score is
# stale (>24h old), because the GT model may have been retrained since then.
GNN_SCORE_TIMESTAMP_COL: str = "gnn_score_timestamp"
GNN_SCORE_STALENESS_WARNING_HOURS: float = 24.0

# ============================================================================
# OPTIONAL CANONICAL-IDENTIFIER COLUMNS
# ============================================================================
SOURCE_DB_COL: str = "source_database"
DRUG_CANONICAL_COL: str = "drug_inchikey"
DISEASE_CANONICAL_COL: str = "disease_mesh_id"

# ============================================================================
# OUTPUT COLUMN CONSTANTS
# ============================================================================
REWARD_COL: str = "reward"
RANK_COL: str = "rank"
LITERATURE_SUPPORT_COL: str = "literature_support"
IS_KNOWN_POSITIVE_COL: str = "is_known_positive"
CONTROLLED_SUBSTANCE_COL: str = "controlled_substance"

# ============================================================================
# DEFAULT FEATURE COLUMNS
# ============================================================================
# The environment may EXTEND this list with disease context features at runtime.
FEATURE_COLS: List[str] = [
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
]

REQUIRED_COLUMNS: List[str] = FEATURE_COLS + [DRUG_COL, DISEASE_COL]

__all__ = [
    # Core identifiers
    "DRUG_COL", "DISEASE_COL",
    # Feature columns
    "GNN_SCORE_COL", "SAFETY_COL", "MARKET_COL", "CONFIDENCE_COL",
    "PATHWAY_COL", "PATENT_COL", "RARE_DISEASE_COL", "UNMET_NEED_COL",
    "EFFICACY_COL", "ADME_COL",
    # Disease context
    "DISEASE_PAIR_COUNT_COL", "DISEASE_AVG_GNN_COL", "DISEASE_AVG_SAFETY_COL",
    # GNN staleness
    "GNN_SCORE_TIMESTAMP_COL", "GNN_SCORE_STALENESS_WARNING_HOURS",
    # Canonical IDs
    "SOURCE_DB_COL", "DRUG_CANONICAL_COL", "DISEASE_CANONICAL_COL",
    # Output
    "REWARD_COL", "RANK_COL", "LITERATURE_SUPPORT_COL",
    "IS_KNOWN_POSITIVE_COL", "CONTROLLED_SUBSTANCE_COL",
    # Aggregates
    "FEATURE_COLS", "REQUIRED_COLUMNS",
]
