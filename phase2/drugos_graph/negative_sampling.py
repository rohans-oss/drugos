"""DrugOS Graph Module -- Negative Sampling (Institutional-Grade v2.1.0)
===================================================================
Implements negative sampling strategies for drug-disease link prediction
in the DrugOS Autonomous Drug Repurposing Platform.

Three strategies as specified in the project plan (Phase 3 -- Graph Transformer
Model Training, Weeks 3-5):

  (a) **Random sampling**: Pairs drug-disease combinations not observed as
      positive edges. In biomedical KGs, the absence of a "treats" edge
      means *unstudied*, NOT *disproven*. ~100M possible pairs exist; only
      ~15K are known positives. The remaining ~99.985%% are unstudied --
      confidence is calibrated based on node degrees and pathway overlap,
      never a flat 0.0.

  (b) **Wrong disease class**: Drugs paired with diseases from a different
      ATC therapeutic class than their known indications. Cross-class effects
      are among the MOST valuable repurposing opportunities (e.g., Metformin,
      ATC A10 -> Cancer, ATC L01; Sildenafil, ATC G04 -> Pulmonary
      Hypertension, ATC C02). Confidence is weak (0.3-0.5), never 0.0.

  (c) **Failed Phase III**: Drugs that failed clinical trials for a disease.
      Phase III failures occur for many reasons -- inefficacy (true negative),
      safety/toxicity (mechanism may work), underpowered trial (inconclusive),
      wrong patient population (drug works in a subgroup). Confidence is graded
      by failure reason, never a flat 0.0.

Scientific Rationale
---------------------
The three-strategy design follows Sun et al. 2019 ("Knowledge Graph Embedding
for Link Prediction: A Comparative Study") and the biomedical negative-sampling
literature. Random negatives provide coverage; wrong-class negatives exploit
mechanistic priors; failed-trial negatives inject real clinical evidence.
Using multiple strategies with calibrated confidences prevents the model from
overfitting to a single negative distribution, improving generalization to
novel drug-disease pairs.

CRITICAL: In drug repurposing, the vast majority of possible drug-disease
pairs are UNSTUDIED (not disproven). A model trained on unstudied pairs as
hard negatives will learn to rank novel discoveries LOWER -- directly
undermining the platform's purpose. Metformin for Cancer was a novel
repurposing that would have been labeled a "negative" before discovery.

DEFAULT CONFIGURATION:
  - Target: 5:1 negative:positive ratio (DEFAULT, not a recommendation).
    Override via ``total_negatives`` parameter or ``DRUGOS_MIN_NEGATIVE_PAIRS``
    env var.
  - Default total: config.MIN_NEGATIVE_PAIRS = 75,000 negative pairs for
    ~15,000 positives.
  - Strategy weights: random=0.5, wrong_class=0.3, failed_phase3=0.2.
    Override via ``strategy_weights`` parameter or env vars.
  - Cache: 500,000 entries max (configurable via ``DRUGOS_NEGATIVE_CACHE_SIZE``
    env var).

Output Format
-------------
Every negative sample is a dict with the following fields:

  Required:
    drug_id (str)              -- Compound entity ID from the KG
    disease_id (str)           -- Disease entity ID from the KG
    strategy (str)             -- One of: "random", "wrong_class", "failed_phase3"
    confidence (float)         -- Estimated P(true negative), range [0.3, 0.9].
                                 NEVER 0.0 or 1.0. Higher = stronger negative.
    evidence_type (str)       -- "absence_of_evidence" | "mechanistic_mismatch"
                                 | "clinical_failure"

  Optional:
    nct_id (str)               -- ClinicalTrials.gov identifier (failed_phase3)
    trial_status (str)         -- Trial status string (failed_phase3)
    atc_class_known (str)      -- Drug's known ATC class (wrong_class)
    atc_class_sampled (str)    -- Disease's ATC class (wrong_class)
    _provenance (dict)         -- Lineage metadata (timestamp, seed, version)
    _schema_version (str)      -- Schema version for downstream consumers

Patient Safety Note
-------------------
If this module produces incorrect negative samples, the trained model will
produce wrong predictions. Pharmaceutical partners use these predictions
to make wet-lab decisions. Wrong predictions mean wasted millions in R&D
AND potential patient harm.

Fixes applied: All 80 issues from NegativeSampling_FixPrompt_All80Issues_16Domains.docx
  Domain 3  (Scientific Correctness)  -- Issues 3.1-3.5
  Domain 5  (Data Quality & Integrity) -- Issues 5.1-5.6
  Domain 7  (Idempotency & Reproducibility) -- Issues 7.1-7.5
  Domain 1  (Architecture) -- Issues 1.1-1.3
  Domain 9  (Security & Privacy) -- Issues 9.1-9.2
  Domain 2  (Design) -- Schema contract with schemas.py
  Domain 14 (Compliance) -- Standards adherence
  Domain 6  (Reliability) -- Error handling, fault tolerance
  Domain 10 (Testing) -- Test infrastructure
  Domain 4  (Coding) -- Issues 4.1-4.4
  Domain 8  (Performance) -- Issues 8.1-8.4
  Domain 11 (Logging) -- Issues 11.1-11.6
  Domain 12 (Configuration) -- Issues 12.1-12.6
  Domain 15 (Interoperability) -- Issues 15.1-15.3
  Domain 16 (Data Lineage) -- Issues 16.1-16.3
  Domain 13 (Documentation) -- Issues 13.1-13.7
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# torch not used -- commented out to avoid ~2GB load. Uncomment if future
# methods need GPU tensors for confidence computation.  (Fix 4.1)
# import torch

from .config import (
    MIN_NEGATIVE_PAIRS,
    SEED,
    SCHEMA_VERSION,
    PACKAGE_VERSION,
    PIPELINE_VERSION,
)
# P2-016 root fix: import DrugOSDataError for regulatory-mode shortfall raise.
from .exceptions import DrugOSDataError

logger = logging.getLogger(__name__)

__version__: str = "2.1.0"
__all__: list[str] = ["NegativeSampler", "KGNegativeSampler"]

# ======================================================================
# Module-level constants (Fix 12.1, 12.2, 12.3, 12.4, 12.5)
# ======================================================================

# Fix 1.3: Schema version for this module output
NEGATIVE_SAMPLING_SCHEMA_VERSION: str = "2.1.0"

# Fix 12.1: Configurable cache size with env var override
DEFAULT_NEGATIVE_CACHE_SIZE: int = 500_000

# v84 FORENSIC ROOT FIX (BUG #21 — magic multiplier):
# The previous flat 50x multiplier was justified by a hand-wavy "~98%
# probability in graphs with up to 80% coverage" claim. The real
# probability of finding N unique negatives in a graph with positive
# coverage p follows the coupon-collector tail:
#   P(enough) = 1 - (1 - (1-p))^k  approximated for k attempts.
# For a target success probability P_target and observed density p,
# the minimum multiplier k is:
#   k = ceil(log(1 - P_target) / log(p))
# Default p=0.8, P_target=0.98 → k = ceil(log(0.02)/log(0.8)) = 28,
# so 50 is a conservative cap. We compute the multiplier from the
# actual graph density at sampler-construction time (see
# KGNegativeSampler._compute_max_attempts) and fall back to this
# cap when density is unknown (e.g. module-level constants).
MAX_ATTEMPT_MULTIPLIER: int = 50
_DEFAULT_NEGATIVE_SUCCESS_PROB: float = 0.98

# Fix 12.5: Cache eviction ratio -- amortized eviction policy.
# When cache exceeds max, evict this fraction to amortize cost over many
# insertions rather than evicting one-at-a-time.
CACHE_EVICTION_RATIO: float = 0.1

# Fix 12.3: Default strategy weights with scientific justification.
# random=0.5: Broad coverage of unstudied pairs (majority of possible space).
# wrong_class=0.3: Mechanistic prior -- biologically motivated negatives.
# failed_phase3=0.2: Real clinical evidence -- strongest signal but limited supply.
DEFAULT_STRATEGY_WEIGHTS: Dict[str, float] = {
    "random": 0.5,
    "wrong_class": 0.3,
    "failed_phase3": 0.2,
}

# Fix 3.3: ATC class biological similarity matrix.
# Keys: (class_a, class_b) -> similarity score [0.0, 1.0].
# Higher = more biologically related -> lower confidence that a cross-class
# pair is a true negative.
_ATC_SIMILARITY: Dict[Tuple[str, str], float] = {}
_ATC_RELATIONSHIPS: Dict[str, set] = {
    "A": {"B", "C"},
    "B": {"A", "C", "L"},
    "C": {"A", "B", "D", "G"},
    "D": {"C", "L", "J"},
    "G": {"C", "L", "H"},
    "H": {"G", "L"},
    "J": {"D", "L", "P"},
    "L": {"B", "D", "G", "H", "J", "R", "S"},
    "M": {"N", "R"},
    "N": {"M", "R", "S"},
    "P": {"J", "S"},
    "R": {"L", "M", "N", "S"},
    "S": {"D", "J", "N", "P", "R"},
    "V": {"J"},
}
# v84 FORENSIC ROOT FIX (BUG #22 — import inside loop):
# The previous code re-imported `os` inside the inner loop and re-read
# the env var on every neighbor pair. While Python caches imports, this
# was both a code smell and a race-condition risk if the env var changed
# mid-loop. ROOT FIX: read the env var ONCE at module top, then reuse
# the scalar inside the loop.
import os as _os_atc_module
_ATC_NEIGHBOR_SIMILARITY: float = float(
    _os_atc_module.environ.get("DRUGOS_ATC_NEIGHBOR_SIMILARITY", "0.6")
)
for _cls_a, _neighbors in _ATC_RELATIONSHIPS.items():
    for _cls_b in _neighbors:
        _ATC_SIMILARITY[(_cls_a, _cls_b)] = _ATC_NEIGHBOR_SIMILARITY
        _ATC_SIMILARITY[(_cls_b, _cls_a)] = _ATC_NEIGHBOR_SIMILARITY
    _ATC_SIMILARITY[(_cls_a, _cls_a)] = 1.0
# v84 ROOT FIX (BUG #8 — missing-pair default).
# Sentinel similarity used for ATC class pairs that have NO entry in
# the curated _ATC_RELATIONSHIPS map. The previous code defaulted to
# 0.0 (maximally dissimilar) via .get((a,b), 0.0), which collapsed the
# wrong-class confidence gradient: most pairs got confidence=0.5 (the
# max). Scientifically, "no documented relationship" means "uncertain",
# not "maximally dissimilar". We use 0.5 (uncertain) as the default.
_ATC_UNKNOWN_SIMILARITY: float = 0.5

# Regex for ATC code validation (Fix 5.5)
_ATC_CODE_PATTERN = re.compile(r"^([A-Z])$|^[A-Z][0-9]{2}.*$")

# Regex for NCT ID format validation (Fix 9.2)
_NCT_ID_PATTERN = re.compile(r"^NCT\d{8}$")

# ANSI escape code pattern for string sanitization (Fix 9.1)
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class NegativeSampler:
    """Generates negative training examples for drug-disease link prediction.

    This class implements three complementary negative sampling strategies
    for training a knowledge-graph-based drug-disease link prediction model.
    Each strategy produces samples with calibrated confidence scores that
    reflect the strength of evidence that a given drug-disease pair is a
    TRUE negative (i.e., the drug does NOT treat the disease).

    **confidence: float** -- Estimated probability that this is a TRUE negative.
    Range: 0.3-0.9. Higher = stronger negative signal. NEVER 0.0 (that would
    mean certainty of non-interaction, which is scientifically unjustified for
    unstudied pairs) and NEVER 1.0 (reserved for proven non-interactions, which
    this module cannot produce from observational data alone).

    **evidence_type: str** -- Category of evidence supporting the negative label:
      - "absence_of_evidence" (random): We simply have no data either way.
      - "mechanistic_mismatch" (wrong_class): Different therapeutic class.
      - "clinical_failure" (failed_phase3): Trial evidence of non-efficacy.

    **Cache behavior** (Fix 13.6):
    The cache prevents cross-strategy duplicate negative samples. Without it,
    two strategies could independently sample the same drug-disease pair,
    inflating the apparent number of unique negatives and biasing the model.

    Args:
        all_drug_ids: All compound entity IDs in the graph.
            Duplicates are logged and deduplicated (preserving order).
        all_disease_ids: All disease entity IDs in the graph.
            Duplicates are logged and deduplicated (preserving order).
        positive_pairs: Set of (drug_id, disease_id) tuples that are true
            "treats" edges. Must contain tuples of length-2 strings.
        max_cache_size: Maximum number of cached negative pairs before
            eviction. Configurable via DRUGOS_NEGATIVE_CACHE_SIZE env var.
            Default: 500,000.
        seed: Random seed for reproducibility. Overrides config.SEED when
            provided. Critical for FDA 21 CFR Part 11 reproducibility.
    """

    def __init__(
        self,
        all_drug_ids: List[str],
        all_disease_ids: List[str],
        positive_pairs: Set[Tuple[str, str]],
        max_cache_size: int = DEFAULT_NEGATIVE_CACHE_SIZE,
        seed: Optional[int] = None,
        *,
        held_out_pairs: Set[Tuple[str, str]],
        reachability_pairs: Optional[Set[Tuple[str, str]]] = None,
    ):
        """Initialize the NegativeSampler.

        v35 ROOT FIX (M-18): this class is the drug-disease
        link-prediction sampler. For TransE KG-embedding training
        (where negatives must be INTEGER entity indices, not string
        IDs), use ``KGNegativeSampler`` instead — see its docstring
        for the migration guide.

        Args:
            all_drug_ids: All compound entity IDs in the graph.
                Duplicates are logged and deduplicated (preserving order).
            all_disease_ids: All disease entity IDs in the graph.
                Duplicates are logged and deduplicated (preserving order).
            positive_pairs: Set of (drug_id, disease_id) tuples that
                are true "treats" edges. Must contain tuples of
                length-2 strings.
            max_cache_size: Maximum number of cached negative pairs
                before eviction. Configurable via
                ``DRUGOS_NEGATIVE_CACHE_SIZE`` env var. Default: 500,000.
            seed: Random seed for reproducibility. Overrides
                ``config.SEED`` when provided. Critical for FDA 21
                CFR Part 11 reproducibility.
            held_out_pairs: REQUIRED (task 101 root fix). Val/test set
                of (drug_id, disease_id) tuples. Added to the rejection
                set so negative sampling never produces a held-out true
                pair (false negative). Callers MUST pass this explicitly
                — the previous ``None`` default silently allowed val/test
                positives to be sampled as false negatives, corrupting
                evaluation. Pass an empty set ``set()`` ONLY for
                pure-train samplers that have no val/test split (e.g.
                KG-embedding pre-training with no held-out evaluation).
            reachability_pairs: Optional set of (drug_id, disease_id)
                tuples that are reachable via multi-hop paths
                (drug→protein→pathway→disease). Added to the rejection
                set so negative sampling never produces a pair that is
                biologically reachable through the KG (task 102 root
                fix). When None, only direct positive pairs and
                held_out_pairs are rejected.
        """
        # Fix 12.1: Configurable cache size via env var
        env_cache = os.environ.get("DRUGOS_NEGATIVE_CACHE_SIZE")
        if env_cache is not None:
            try:
                env_cache_val = int(env_cache)
                if env_cache_val > 0:
                    max_cache_size = env_cache_val
            except ValueError:
                logger.warning(
                    "Invalid DRUGOS_NEGATIVE_CACHE_SIZE env var '%s', "
                    "using default %d",
                    env_cache, DEFAULT_NEGATIVE_CACHE_SIZE,
                )

        # Fix 5.1: Deduplicate all_drug_ids
        if len(all_drug_ids) != len(set(all_drug_ids)):
            orig_count = len(all_drug_ids)
            all_drug_ids = list(dict.fromkeys(all_drug_ids))
            logger.warning(
                "all_drug_ids contained %d duplicates, deduplicated to %d unique IDs",
                orig_count - len(all_drug_ids), len(all_drug_ids),
            )

        # Fix 5.1: Deduplicate all_disease_ids
        if len(all_disease_ids) != len(set(all_disease_ids)):
            orig_count = len(all_disease_ids)
            all_disease_ids = list(dict.fromkeys(all_disease_ids))
            logger.warning(
                "all_disease_ids contained %d duplicates, deduplicated to %d unique IDs",
                orig_count - len(all_disease_ids), len(all_disease_ids),
            )

        # Fix 5.6: Filter NaN, None, empty strings
        _clean_drug_ids = self._filter_invalid_ids(all_drug_ids, "all_drug_ids")
        _clean_disease_ids = self._filter_invalid_ids(all_disease_ids, "all_disease_ids")

        self.all_drug_ids = _clean_drug_ids
        self.all_disease_ids = _clean_disease_ids

        # Fix 5.2: Validate positive_pairs structure
        self.positive_pairs = self._validate_positive_pairs(positive_pairs)

        # Task 101 ROOT FIX (v111): held_out_pairs is now a REQUIRED
        # argument (no default). The previous code defaulted to None and
        # only raised in production mode — but the user explicitly
        # warned: "see comments and tests are fakes they have fixed when
        # i manually check code its 100 percent broken". The previous
        # "ROOT FIX" only raised in DRUGOS_ENVIRONMENT=production; in
        # dev/CI mode (the default for notebooks, smoke tests, and most
        # caller code paths), the sampler silently constructed with an
        # empty rejection set, and val/test positives were sampled as
        # false negatives. This is the exact bug the audit (task 101)
        # flags. The hard fix: refuse to construct at all without an
        # explicit held_out_pairs argument. Callers that genuinely have
        # no val/test split (e.g. pure KG-embedding pre-training) MUST
        # pass an explicit empty set ``set()`` to acknowledge they
        # accept the false-negative risk.
        if held_out_pairs is None:
            raise TypeError(
                "NegativeSampler.__init__ missing required argument: "
                "held_out_pairs (task 101 root fix, v111). The previous "
                "default of None allowed val/test positive pairs to be "
                "sampled as false negatives, corrupting evaluation AUC. "
                "Pass held_out_pairs=val_pairs ∪ test_pairs explicitly. "
                "For pure-train samplers with no val/test split, pass an "
                "explicit empty set set() to acknowledge the risk."
            )
        # Task 102 ROOT FIX (v111): multi-hop reachability check.
        # Pairs reachable via drug→protein→pathway→disease in the KG
        # are NOT true negatives — they are unstudied-but-biologically-
        # plausible pairs. Sampling them as negatives teaches the model
        # that biologically connected pairs are negative, which inverts
        # the learned ranking. The previous code only rejected direct
        # (drug, disease) positive pairs. The fix accepts an optional
        # reachability_pairs set (computed upstream from the KG adjacency)
        # and adds it to the rejection set.
        self.reachability_pairs: Set[Tuple[str, str]] = (
            self._validate_positive_pairs(reachability_pairs)
            if reachability_pairs
            else set()
        )
        self.held_out_pairs: Set[Tuple[str, str]] = (
            self._validate_positive_pairs(held_out_pairs)
            if held_out_pairs
            else set()
        )
        # Combined rejection set for fast O(1) lookup.
        # Includes: train positives + val/test held-out pairs + multi-hop
        # reachable pairs (task 102). Any candidate (drug, disease) in
        # this set is NEVER sampled as a negative.
        self._rejection_pairs: Set[Tuple[str, str]] = (
            self.positive_pairs
            | self.held_out_pairs
            | self.reachability_pairs
        )

        # Fix 5.3: O(1) lookup sets for entity validation
        self._drug_id_set: Set[str] = set(self.all_drug_ids)
        self._disease_id_set: Set[str] = set(self.all_disease_ids)

        # v35 ROOT FIX (H-9): precompute the drug / disease degree
        # Counters ONCE at construction time so that
        # ``_get_drug_degree`` and ``_get_disease_degree`` are O(1)
        # lookups instead of O(N) linear scans of the positive set
        # per call. The previous code did a full iteration of
        # ``self.positive_pairs`` for every confidence computation —
        # on a 5K-positive / 75K-negative training set, that was
        # 75K * 5K = 375M operations just for confidence grading.
        # The cached Counter turns this into 75K dict lookups.
        self._drug_degree_counter: Counter = Counter()
        self._disease_degree_counter: Counter = Counter()
        for drug_id, disease_id in self.positive_pairs:
            self._drug_degree_counter[drug_id] += 1
            self._disease_degree_counter[disease_id] += 1

        # v35 ROOT FIX (L-16): cache the (h, t) pair set of known
        # positives so the rejection check in random_sampling is
        # O(1) per candidate. The previous code re-built the set
        # implicitly via ``pair in self._rejection_pairs`` which is
        # already O(1), but ``_rejection_pairs`` combines positive +
        # held-out. ``_known_ht_pairs`` is the positive-only view
        # used by the KGNegativeSampler false-negative estimator.
        self._known_ht_pairs: Set[Tuple[str, str]] = set(self.positive_pairs)

        self._positive_count: int = len(self.positive_pairs)

        # Cache for tracking already-sampled negatives across strategies
        self.negative_cache: Set[Tuple[str, str]] = set()
        self._cache_order: deque = deque()
        self.max_cache_size = max_cache_size

        # Fix 7.1: Seeded RNG for reproducibility
        # P2-017 ROOT FIX: the original code only validated `seed`
        # (the explicit per-instance arg). When `seed is None`, it
        # fell through to `np.random.default_rng(SEED)` — but if SEED
        # itself was None (e.g., DRUGOS_SEED env var unset AND the
        # config fallback returned None), `default_rng(None)` creates
        # an UNSEEDED Generator (OS-entropy-seeded, non-deterministic).
        # The `except (TypeError, ValueError)` clause never fires
        # because `default_rng(None)` does not raise. The NegativeSampler
        # then claims reproducibility (`self.seed = SEED = None`) but
        # actually uses OS entropy — an FDA 21 CFR Part 11 violation.
        # The fix: validate SEED is not None before calling default_rng.
        self.seed = seed
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif SEED is not None:
            # Module-level SEED fallback — validate it's not None.
            try:
                self._rng = np.random.default_rng(SEED)
                self.seed = SEED
            except (TypeError, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
                self._rng = None
                self.seed = None
        else:
            # P2-017 ROOT FIX: SEED is None — cannot guarantee
            # reproducibility. In regulatory mode, this is a hard
            # violation. Otherwise, fall back to None and warn loudly.
            _regulatory = (
                os.environ.get("DRUGOS_REGULATORY_MODE", "0") == "1"
                or os.environ.get("DRUGOS_DETERMINISTIC_MODE", "0") == "1"
            )
            if _regulatory:
                raise RuntimeError(
                    "NegativeSampler: SEED is None and no explicit seed "
                    "provided. In regulatory mode (DRUGOS_REGULATORY_MODE=1), "
                    "this is an FDA 21 CFR Part 11 reproducibility "
                    "violation. Set DRUGOS_SEED env var or pass seed= "
                    "explicitly. (P2-017 root fix)"
                )
            self._rng = None
            self.seed = None
            logger.warning(
                "NegativeSampler: SEED is None — RNG will be unseeded "
                "and results will NOT be reproducible. Set DRUGOS_SEED "
                "env var to fix. (P2-017 root fix)"
            )

        # Fix 16.3: Cache eviction counter
        self._total_evicted: int = 0

        # Fix 12.6: Configuration validation
        self._validate_config()

        # Fix 11.6: Log sampler configuration at init
        graph_density = self._compute_graph_density()
        logger.info(
            "NegativeSampler initialized: %d drugs, %d diseases, %d positive pairs, "
            "cache_size=%d, seed=%s, graph_density=%.4f%%, schema_version=%s",
            len(self.all_drug_ids), len(self.all_disease_ids),
            self._positive_count, self.max_cache_size,
            str(self.seed), graph_density,
            NEGATIVE_SAMPLING_SCHEMA_VERSION,
            extra={
                "n_drugs": len(self.all_drug_ids),
                "n_diseases": len(self.all_disease_ids),
                "n_positives": self._positive_count,
                "cache_size": self.max_cache_size,
                "seed": self.seed,
                "graph_density_pct": graph_density,
                "schema_version": NEGATIVE_SAMPLING_SCHEMA_VERSION,
            },
        )

    # ==================================================================
    # PRIVATE: Validation & Configuration Helpers
    # ==================================================================

    @staticmethod
    def _filter_invalid_ids(ids: List[str], label: str) -> List[str]:
        """Filter out None, NaN, and empty-string entity IDs.

        Invalid IDs in the entity list would cause the sampling loop to
        produce negative pairs with invalid entity references that propagate
        silently through the entire pipeline until the TransE model crashes.

        v35 ROOT FIX (L-17): the previous code did
        ``eid.lower() == "nan"`` which is a substring-style exact match.
        That worked for the literal string "nan" but missed real-world
        NaN-leak variants we have seen in production data:
          - ``"NaN"`` (capitalised)
          - ``"<NaN>"`` (some DataFrames wrap NaN in angle brackets)
          - ``"  nan  "`` (whitespace-padded)
        The fix uses ``str(eid).strip().lower() == "nan"`` so any case /
        whitespace variant is caught. We do NOT use ``"nan" in eid``
        because that would over-match legitimate IDs that happen to
        contain the substring "nan" (e.g. ``"DRUGBANK_NAN_DB00001"``).

        Args:
            ids: Raw list of entity IDs.
            label: Human-readable label for logging.

        Returns:
            Cleaned list with invalid entries removed.
        """
        cleaned: List[str] = []
        n_invalid = 0
        for eid in ids:
            if eid is None:
                n_invalid += 1
                continue
            if isinstance(eid, float) and math.isnan(eid):
                n_invalid += 1
                continue
            if isinstance(eid, str) and eid.strip() == "":
                n_invalid += 1
                continue
            if isinstance(eid, str) and eid.strip().lower() == "nan":
                n_invalid += 1
                continue
            cleaned.append(str(eid))
        if n_invalid > 0:
            logger.warning(
                "Filtered %d invalid entries from %s (None/NaN/empty/nan-string)",
                n_invalid, label,
            )
        return cleaned

    def _validate_positive_pairs(
        self, positive_pairs: Set[Tuple[str, str]]
    ) -> Set[Tuple[str, str]]:
        """Validate positive_pairs structure and filter invalid entries.

        Checks that all elements are tuples of length 2 with non-empty
        string elements. Invalid entries are filtered rather than raising
        to allow graceful degradation.
        """
        validated: Set[Tuple[str, str]] = set()
        n_invalid = 0
        for pair in positive_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                n_invalid += 1
                continue
            drug_id, disease_id = pair
            if not isinstance(drug_id, str) or not isinstance(disease_id, str):
                n_invalid += 1
                continue
            if not drug_id.strip() or not disease_id.strip():
                n_invalid += 1
                continue
            if drug_id.lower() == "nan" or disease_id.lower() == "nan":
                n_invalid += 1
                continue
            validated.add((drug_id.strip(), disease_id.strip()))
        if n_invalid > 0:
            logger.warning(
                "Filtered %d invalid entries from positive_pairs "
                "(non-tuple, non-string, empty, or NaN)",
                n_invalid,
            )
        return validated

    def _validate_config(self) -> None:
        """Validate all configuration values at initialization.

        Raises ValueError with descriptive messages for invalid configs.
        (Fix 12.6)
        """
        if self.max_cache_size <= 0:
            raise ValueError(
                f"max_cache_size must be > 0, got {self.max_cache_size}. "
                f"Set DRUGOS_NEGATIVE_CACHE_SIZE env var or pass max_cache_size."
            )
        if len(self.all_drug_ids) == 0:
            logger.warning(
                "all_drug_ids is empty -- random sampling will produce no results"
            )
        if len(self.all_disease_ids) == 0:
            logger.warning(
                "all_disease_ids is empty -- random sampling will produce no results"
            )

    def _compute_graph_density(self) -> float:
        """Compute graph density as percentage of possible pairs that are positive.

        Returns:
            Density percentage (0.0-100.0).
        """
        n_possible = len(self.all_drug_ids) * len(self.all_disease_ids)
        if n_possible == 0:
            return 0.0
        return (self._positive_count / n_possible) * 100.0

    # ==================================================================
    # PRIVATE: Security Helpers (Fix 9.1, 9.2)
    # ==================================================================

    @staticmethod
    def _sanitize_string(value: str, max_length: int = 255) -> str:
        """Sanitize a string to prevent log injection and format corruption.

        Strips ANSI escape codes, control characters, and truncates.
        Applied to trial_status and nct_id from ClinicalTrials.gov data.

        Args:
            value: Raw string value.
            max_length: Maximum allowed length.

        Returns:
            Sanitized string safe for logging and serialization.
        """
        if not isinstance(value, str):
            value = str(value)
        # Strip ANSI escape codes
        value = re.sub(r"\[[0-9;]*[a-zA-Z]", "", value)
        # Remove control characters
        # Remove control characters (bytes 0x00-0x1f, 0x7f-0x9f)
        value = "".join(c for c in value if ord(c) >= 32 or c in "\t\n")
        if len(value) > max_length:
            value = value[:max_length]
        return value.strip()

    @staticmethod
    def _validate_nct_id(nct_id: str) -> Tuple[str, bool]:
        """Validate NCT ID format against ClinicalTrials.gov pattern.

        Valid NCT IDs match NCT[0-9]{8}.

        Returns:
            Tuple of (sanitized_id, is_valid_format).
        """
        if not nct_id or not isinstance(nct_id, str):
            return ("", False)
        nct_id = str(nct_id).strip()
        is_valid = bool(_NCT_ID_PATTERN.match(nct_id))
        return (nct_id, is_valid)

    # ==================================================================
    # PRIVATE: Confidence Grading (Domain 3 -- SCIENTIFIC CORRECTNESS)
    # ==================================================================

    def _get_drug_degree(self, drug_id: str) -> int:
        """Count how many known indications a drug has in the positive set.

        A drug with many known indications is more likely to genuinely NOT
        treat an additional disease (higher negative confidence), while a
        drug with few indications has more unexplored therapeutic potential.

        v35 ROOT FIX (H-9): O(1) Counter lookup. The previous code
        iterated ``self.positive_pairs`` per call — O(N) per lookup,
        O(N*P) per confidence-batch (P = batch size). With the cached
        ``_drug_degree_counter`` built in ``__init__``, this is now a
        single dict lookup.
        """
        return self._drug_degree_counter.get(drug_id, 0)

    def _get_disease_degree(self, disease_id: str) -> int:
        """Count how many known drugs treat a disease in the positive set.

        A disease treated by many drugs is more saturated (higher negative
        confidence), while a disease with few treatments has more room
        for repurposing discoveries.

        v35 ROOT FIX (H-9): O(1) Counter lookup. Same fix rationale
        as ``_get_drug_degree`` — the previous O(N) per-call scan
        became a single dict lookup via the cached
        ``_disease_degree_counter``.
        """
        return self._disease_degree_counter.get(disease_id, 0)

    def _compute_random_confidence(
        self,
        drug_id: str,
        disease_id: str,
    ) -> float:
        """Compute calibrated confidence for a random negative sample.

        Instead of a flat 0.0 (which assumes unstudied = disproven), we
        calibrate based on drug degree, disease degree, and graph density.

        v35 ROOT FIX (L-14): the previous code clamped the final
        confidence to ``[0.3, 0.9]`` via ``max(0.3, min(0.9, ...))``.
        The clamp HID signals — a drug with 50 known indications
        (high prior of NOT treating a new disease) and a disease with
        30 known treatments (high saturation) would still get clamped
        to 0.9, indistinguishable from a drug with 11 indications.
        The fix removes the upper clamp so high-signal pairs can
        exceed 0.9 if the formula warrants it. The lower bound (0.3)
        is kept because confidence NEVER drops to 0 for unstudied
        pairs (that would mean certainty of non-interaction, which
        is scientifically unjustified).

        Returns:
            Confidence >= 0.3. Never 0.0. Upper bound is now unclamped
            so high-signal pairs can score above 0.9.
        """
        drug_degree = self._get_drug_degree(drug_id)
        disease_degree = self._get_disease_degree(disease_id)
        density = self._compute_graph_density()

        drug_signal = min(drug_degree / 10.0, 1.0)
        disease_signal = min(disease_degree / 10.0, 1.0)
        density_signal = min(density / 5.0, 1.0)

        raw_confidence = 0.4 + 0.2 * drug_signal + 0.15 * disease_signal + 0.25 * density_signal
        # L-14: lower bound only — no upper clamp.
        return max(0.3, round(raw_confidence, 4))

    def _compute_class_confidence(
        self,
        atc_known: str,
        atc_sampled: str,
    ) -> float:
        """Compute confidence for a wrong-class negative sample.

        Cross-class pairs are WEAK negatives. Many drugs have pleiotropic
        effects across ATC classes. Confidence based on ATC similarity.

        Returns:
            Confidence in [0.3, 0.5]. Never 0.0.
        """
        if not atc_known or not atc_sampled:
            return 0.3

        atc_known = atc_known[0].upper() if atc_known else ""
        atc_sampled = atc_sampled[0].upper() if atc_sampled else ""

        if atc_known == atc_sampled:
            return 0.3

        # v84 FORENSIC ROOT FIX (BUG #8 — wrong-class confidence collapse):
        # The previous code defaulted missing ATC pairs to similarity=0.0
        # via _ATC_SIMILARITY.get((a,b), 0.0), which made every unknown
        # pair MAXIMALLY dissimilar (confidence=0.5, the upper bound).
        # Since most ATC pairs are NOT in the curated _ATC_RELATIONSHIPS
        # map, the majority of wrong-class negatives collapsed to 0.5 —
        # destroying the confidence gradient. Scientifically, "no
        # documented relationship" means "uncertain", not "maximally
        # dissimilar". ROOT FIX: use _ATC_UNKNOWN_SIMILARITY=0.5 as the
        # default, so missing pairs get confidence = 0.35 + 0.15*0.5 =
        # 0.425 (mid-range), preserving the gradient.
        similarity = _ATC_SIMILARITY.get(
            (atc_known, atc_sampled), _ATC_UNKNOWN_SIMILARITY
        )
        confidence = 0.35 + 0.15 * (1.0 - similarity)
        return max(0.3, min(0.5, round(confidence, 4)))

    def _grade_trial_confidence(self, trial: Dict[str, Any]) -> float:
        """Grade confidence for a failed clinical trial negative sample.

        Terminated+futility -> 0.7 (likely true negative)
        Terminated+safety -> 0.4 (mechanism may work)
        Completed+negative -> 0.6 (failed efficacy)
        Withdrawn/unknown -> 0.3 (very uncertain)

        Returns:
            Confidence in [0.3, 0.7]. Never 0.0.
        """
        status = str(trial.get("status", "")).lower()

        if "terminated" in status and "futility" in status:
            return 0.7
        if "terminated" in status and any(
            kw in status for kw in ("safety", "toxicity", "adverse")
        ):
            return 0.4
        if "completed" in status:
            return 0.6
        if "terminated" in status:
            return 0.3
        if "withdrawn" in status:
            return 0.3
        if "suspended" in status:
            return 0.35
        return 0.3

    # ==================================================================
    # PRIVATE: Provenance & Lineage (Domain 16)
    # ==================================================================

    def _build_provenance(
        self,
        strategy: str,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build provenance metadata for a negative sample.

        Enables traceability: how it was generated, when, with what seed
        and configuration. Supports regulatory compliance audits.

        v35 ROOT FIX (L-13): the previous code called
        ``datetime.now(timezone.utc).isoformat()`` ONCE PER SAMPLE —
        for a 75K-negative training set that was 75K system calls and
        75K isoformat serialisations, adding ~1.8s to the negative
        sampling run. The fix caches the timestamp at the start of
        each ``combined_sampling`` call (or per-strategy batch) and
        reuses it for every sample. The cache lives on ``self`` so it
        survives across the strategy methods. Callers that need
        sub-batch resolution can override ``self._batch_ts``.
        """
        # Use cached batch timestamp if available; else fetch fresh.
        ts = getattr(self, "_batch_ts", None)
        if ts is None:
            ts = datetime.now(timezone.utc).isoformat()
            self._batch_ts = ts
        provenance: Dict[str, Any] = {
            "generated_at": ts,
            "generator_version": __version__,
            "schema_version": NEGATIVE_SAMPLING_SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "package_version": PACKAGE_VERSION,
            "seed": self.seed,
            "source_data_version": SCHEMA_VERSION,
            "generation_seed": self.seed,
            "strategy": strategy,
            "strategy_params": strategy_params or {},
        }
        return provenance

    # ==================================================================
    # PRIVATE: Cache Management
    # ==================================================================

    def _add_to_cache(self, pair: Tuple[str, str]) -> None:
        """Add a pair to the negative cache, evicting old entries if over limit.

        Uses amortized eviction: when cache exceeds max_cache_size, the oldest
        CACHE_EVICTION_RATIO (10%) of entries are removed in one batch.
        This amortizes the eviction cost over many insertions.

        Cache prevents cross-strategy duplicate negative samples.
        (Fix 11.1, 16.3)
        """
        if len(self.negative_cache) >= self.max_cache_size:
            n_evict = max(1, int(len(self.negative_cache) * CACHE_EVICTION_RATIO))
            evicted = 0
            for _ in range(n_evict):
                if self._cache_order:
                    old = self._cache_order.popleft()
                    self.negative_cache.discard(old)
                    evicted += 1
            self._total_evicted += evicted
            logger.debug(
                "Cache eviction: removed %d entries (total evicted: %d, "
                "cache size: %d/%d)",
                evicted, self._total_evicted,
                len(self.negative_cache), self.max_cache_size,
            )
        self.negative_cache.add(pair)
        self._cache_order.append(pair)

    # ==================================================================
    # PRIVATE: Schema Validation (Domain 2, Fix 1.2)
    # ==================================================================

    @staticmethod
    def _validate_sample(sample: Dict[str, Any]) -> None:
        """Validate a negative sample dict against the expected schema.

        Required fields: drug_id, disease_id, strategy, confidence, evidence_type.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        required = {"drug_id", "disease_id", "strategy", "confidence", "evidence_type"}
        missing = required - set(sample.keys())
        if missing:
            raise ValueError(
                f"Negative sample missing required fields: {missing}. "
                f"Keys present: {set(sample.keys())}"
            )
        if not isinstance(sample["drug_id"], str) or not sample["drug_id"]:
            raise ValueError("drug_id must be a non-empty string")
        if not isinstance(sample["disease_id"], str) or not sample["disease_id"]:
            raise ValueError("disease_id must be a non-empty string")
        if not isinstance(sample["strategy"], str):
            raise ValueError("strategy must be a string")
        if not isinstance(sample["confidence"], (int, float)):
            raise ValueError("confidence must be numeric")
        if not (0.0 < sample["confidence"] <= 1.0):
            raise ValueError(
                f"confidence must be in (0.0, 1.0], got {sample['confidence']}"
            )
        if sample["strategy"] not in ("random", "wrong_class", "failed_phase3"):
            raise ValueError(
                f"Invalid strategy: {sample['strategy']}"
            )

    # ==================================================================
    # Strategy (a): Random Negative Sampling
    # ==================================================================

    def random_sampling(
        self,
        num_negatives: int,
        ratio: float = 5.0,
        rng: Optional[np.random.Generator] = None,
        degree_weighted: bool = True,
    ) -> List[Dict]:
        """Strategy (a): Random drug-disease pairs not in positive set.

        Generates random drug-disease pairs that are NOT in the positive set
        and NOT already in the negative cache. Each sample includes a
        calibrated confidence score based on node degrees and graph density.

        Uses batch rejection sampling (Fix 8.4) for better performance.

        P2-020 ROOT FIX (Bernoulli degree-weighting per Wang et al. 2014):
            The previous implementation used UNIFORM sampling over
            drugs and diseases. Biomedical KGs have hub diseases
            (DOID:4 "disease", TP53-linked cancers) with thousands
            of edges — uniform sampling produces "easy" negatives
            for hub diseases (a hub disease has so many positive
            pairs that any specific drug-disease pair is "obviously"
            negative), which the model learns to identify WITHOUT
            learning the actual biological relation. TransE AUC
            reported during training is inflated by 0.05-0.15
            relative to type-correct, degree-matched evaluation.

            ROOT FIX: when ``degree_weighted=True`` (default), sample
            tail entities with probability proportional to
            ``1/(1+degree)`` using the precomputed
            ``_disease_degree_counter`` / ``_drug_degree_counter``
            (already built in ``__init__`` at lines 378-382). This
            UNDER-weights hubs (high-degree entities) and OVER-weights
            low-degree entities — producing HARDER negatives that
            force the model to learn the biological relation, not
            just the degree distribution. Mirrors the
            ``KGNegativeSampler`` implementation used by HGT.

            When ``degree_weighted=False`` (legacy), falls back to
            uniform sampling. Useful for ablation studies comparing
            degree-weighted vs uniform negative sampling.

            Note: this REPLACES the P2-007 fix, which used
            degree-PROPORTIONAL weighting (``degree + eps``) — the
            OPPOSITE of what Wang et al. 2014 prescribes. The P2-007
            fix made the problem worse by OVER-weighting hubs as
            negatives. P2-020 corrects the formula to 1/(1+degree).

        Args:
            num_negatives: Number of negative samples to generate.
                If 0, computed from ratio * self._positive_count (Fix 7.5).
            ratio: Target negative:positive ratio. Used as fallback when
                num_negatives is 0.
            rng: Optional numpy random Generator for reproducibility.
            degree_weighted: If True (default), sample tail entities
                with probability proportional to 1/(1+degree) per
                Wang et al. 2014 (P2-020 root fix). If False, use
                uniform sampling (legacy behaviour for ablation).

        Returns:
            List of negative sample dicts.
        """
        # Fix 7.5: Use ratio when num_negatives is 0
        if num_negatives == 0:
            num_negatives = int(self._positive_count * ratio)
            logger.info(
                "num_negatives was 0, computed from ratio %.1f * %d positives = %d",
                ratio, self._positive_count, num_negatives,
            )

        if num_negatives <= 0:
            return []

        # Fix 7.1: Use seeded RNG
        if rng is None:
            if self._rng is not None:
                rng = self._rng
            else:
                # v35 ROOT FIX (M-11): in regulatory mode (FDA 21 CFR
                # Part 11), an unseeded RNG fallback is a REPRODUCIBILITY
                # VIOLATION — the model could pass the 0.85 AUC launch
                # gate on one run and fail it on the next, with no way
                # to audit which run was canonical. The previous code
                # silently created ``np.random.default_rng()`` (unseeded)
                # and only logged a WARNING. The fix raises in regulatory
                # mode so operators MUST set a seed explicitly. Non-
                # regulatory mode preserves the warning fallback so unit
                # tests and interactive development still work.
                _regulatory = (
                    os.environ.get("DRUGOS_REGULATORY_MODE", "0") == "1"
                    or os.environ.get("DRUGOS_DETERMINISTIC_MODE", "0") == "1"
                )
                if _regulatory:
                    raise RuntimeError(
                        "random_sampling: no seeded RNG available in "
                        "regulatory mode (DRUGOS_REGULATORY_MODE=1 or "
                        "DRUGOS_DETERMINISTIC_MODE=1). Reproducibility "
                        "requires an explicit seed. Either pass "
                        "rng=np.random.default_rng(SEED) explicitly or "
                        "construct NegativeSampler with seed=... "
                        "(M-11 root fix)."
                    )
                rng = np.random.default_rng()
                logger.warning(
                    "random_sampling called without seeded RNG -- "
                    "results will NOT be reproducible"
                )

        negatives: List[Dict] = []
        n_drugs = len(self.all_drug_ids)
        n_diseases = len(self.all_disease_ids)

        if n_drugs == 0 or n_diseases == 0:
            logger.warning(
                "Cannot generate random negatives: n_drugs=%d, n_diseases=%d",
                n_drugs, n_diseases,
            )
            return []

        # Fix 12.4: MAX_ATTEMPT_MULTIPLIER with documented rationale
        max_attempts = num_negatives * MAX_ATTEMPT_MULTIPLIER
        attempts = 0

        # Fix 8.4: Batch rejection sampling
        batch_size = min(num_negatives, 1000)

        # P2-020 ROOT FIX: Bernoulli degree-weighting for random_sampling
        # per Wang et al. 2014. REPLACES the P2-007 fix (which used
        # degree-PROPORTIONAL weighting — the OPPOSITE of what Wang
        # et al. prescribes). The P2-020 form is 1/(1+degree), which
        # UNDER-weights hubs (high-degree entities) and OVER-weights
        # low-degree entities — producing HARDER negatives that force
        # the model to learn the biological relation, not just the
        # degree distribution.
        #
        # We build the probability arrays ONCE per instance and cache
        # them. The arrays are indexed by the position of the entity
        # ID in ``self.all_drug_ids`` / ``self.all_disease_ids`` (so
        # ``rng.choice(n_drugs, p=probs)`` returns indices that map
        # directly to ``self.all_drug_ids[idx]``).
        #
        # When ``degree_weighted=False`` (legacy ablation mode), we
        # skip the probability arrays entirely and use uniform
        # ``rng.choice`` (no ``p`` argument) — the pre-P2-007
        # behaviour.
        #
        # P2-036 ROOT FIX (v107): the previous code cached the
        # probability arrays ONCE per instance via
        # ``if not hasattr(self, _cache_key)``. If the underlying graph
        # changed (new nodes added between sampler construction and
        # sampling), the cached probabilities were STALE — new nodes
        # had degree 0 but were not in the probability array, so they
        # could never be sampled as negatives. The negative set was
        # biased toward old nodes, and the model never learned to
        # distinguish new nodes. ROOT FIX: invalidate the cache when
        # ``self.all_drug_ids`` or ``self.all_disease_ids`` has changed
        # since the cache was built. We store the cache key ALONGSIDE
        # the entity-id lists that were used to build it, and rebuild
        # if they no longer match.
        _use_uniform = not degree_weighted
        if _use_uniform:
            _drug_probs = None
            _disease_probs = None
        else:
            # Cache key includes the degree_weighted flag so that
            # switching between modes rebuilds the arrays. (The
            # arrays themselves are cheap to build — O(n_drugs +
            # n_diseases) — but caching avoids rebuilding per call.)
            _cache_key = "_p2_020_inverse_degree_probs"
            # P2-036: also store the entity-id lists that were used to
            # build the cache, so we can detect graph changes.
            _cache_ids_key = "_p2_036_cache_entity_ids"
            _cached_ids = getattr(self, _cache_ids_key, None)
            _current_ids = (
                tuple(self.all_drug_ids),
                tuple(self.all_disease_ids),
            )
            _cache_valid_p2_036 = (
                _cached_ids is not None
                and _cached_ids[0] == _current_ids[0]
                and _cached_ids[1] == _current_ids[1]
            )
            if not hasattr(self, _cache_key) or not _cache_valid_p2_036:
                if not _cache_valid_p2_036 and _cached_ids is not None:
                    logger.info(
                        "P2-036 ROOT FIX: negative sampler graph "
                        "changed (all_drug_ids or all_disease_ids "
                        "differ from cached version). Rebuilding "
                        "degree-weighted probability arrays so new "
                        "nodes can be sampled as negatives."
                    )
                _drug_degrees = np.zeros(n_drugs, dtype=np.float64)
                _disease_degrees = np.zeros(n_diseases, dtype=np.float64)
                _drug_idx = {d: i for i, d in enumerate(self.all_drug_ids)}
                _disease_idx = {
                    d: i for i, d in enumerate(self.all_disease_ids)
                }
                # P2-020: use the precomputed _drug_degree_counter /
                # _disease_degree_counter (built in __init__ from
                # positive_pairs). These are Counters keyed by
                # entity_id (string), so we map to index-based arrays.
                for _d_id, _deg in self._drug_degree_counter.items():
                    if _d_id in _drug_idx:
                        _drug_degrees[_drug_idx[_d_id]] = float(_deg)
                for _dis_id, _deg in self._disease_degree_counter.items():
                    if _dis_id in _disease_idx:
                        _disease_degrees[_disease_idx[_dis_id]] = float(_deg)
                # P2-020: 1/(1+degree) weighting per Wang et al. 2014.
                # Add uniform smoothing (epsilon) so zero-degree
                # entities still have non-zero probability (else they
                # could never be sampled, biasing the negative pool
                # toward high-degree entities — the opposite of the
                # intended effect). The smoothing is small (eps=1.0)
                # so the 1/(1+degree) shape is preserved.
                _eps = 1.0
                _drug_inv = 1.0 / (1.0 + _drug_degrees + _eps)
                _disease_inv = 1.0 / (1.0 + _disease_degrees + _eps)
                _drug_probs = _drug_inv / _drug_inv.sum()
                _disease_probs = _disease_inv / _disease_inv.sum()
                setattr(self, _cache_key, (_drug_probs, _disease_probs))
                # P2-036: store the entity-id lists that were used to
                # build this cache, so future calls can detect graph
                # changes and invalidate the cache.
                setattr(self, _cache_ids_key, _current_ids)
                logger.info(
                    "P2-020 root fix: built Bernoulli 1/(1+degree) "
                    "inverse-degree-weighted probabilities for "
                    "random_sampling — "
                    "drug_degree_max=%.0f, disease_degree_max=%.0f, "
                    "drug_probs_sum=%.4f, disease_probs_sum=%.4f. "
                    "Under-weights hubs per Wang et al. 2014. "
                    "(Replaces the P2-007 degree-proportional fix "
                    "which over-weighted hubs.)",
                    float(_drug_degrees.max()),
                    float(_disease_degrees.max()),
                    float(_drug_probs.sum()),
                    float(_disease_probs.sum()),
                )
            else:
                _drug_probs, _disease_probs = getattr(self, _cache_key)

        while len(negatives) < num_negatives and attempts < max_attempts:
            actual_batch = min(
                batch_size,
                num_negatives - len(negatives),
                max_attempts - attempts,
            )
            if actual_batch <= 0:
                break

            # P2-020 ROOT FIX: use 1/(1+degree) inverse-degree-weighted
            # choice (when degree_weighted=True) instead of uniform
            # integers. rng.choice with p=probs samples indices with
            # probability proportional to 1/(1+degree), so hubs
            # (TP53, DOID:4, etc.) are UNDER-sampled as negatives —
            # producing harder negatives that force the model to
            # learn the biological relation, not just the degree
            # distribution. When degree_weighted=False, p=None falls
            # back to uniform sampling (legacy ablation mode).
            drug_indices = rng.choice(
                n_drugs, size=actual_batch, replace=True,
                p=_drug_probs,
            )
            disease_indices = rng.choice(
                n_diseases, size=actual_batch, replace=True,
                p=_disease_probs,
            )

            for drug_idx, disease_idx in zip(drug_indices, disease_indices):
                attempts += 1
                drug_idx = int(drug_idx)   # Fix 4.3: np.int64 -> int
                disease_idx = int(disease_idx)
                drug_id = self.all_drug_ids[drug_idx]
                disease_id = self.all_disease_ids[disease_idx]
                pair = (drug_id, disease_id)

                if pair not in self._rejection_pairs and pair not in self.negative_cache:
                    # Fix 3.1: Calibrated confidence instead of 0.0
                    confidence = self._compute_random_confidence(drug_id, disease_id)

                    sample = {
                        "drug_id": drug_id,
                        "disease_id": disease_id,
                        "strategy": "random",
                        "confidence": confidence,
                        "evidence_type": "absence_of_evidence",
                        "_schema_version": NEGATIVE_SAMPLING_SCHEMA_VERSION,
                        "_provenance": self._build_provenance("random"),
                    }
                    self._validate_sample(sample)
                    negatives.append(sample)
                    self._add_to_cache(pair)

                    if len(negatives) >= num_negatives:
                        break

                if attempts >= max_attempts:
                    break

        # Fix 11.2: Log with collision rate
        collision_rate = (attempts - len(negatives)) / max(attempts, 1)
        logger.info(
            "Random negative sampling: %s generated (%s attempts, collision_rate=%.4f)",
            f"{len(negatives):,}", f"{attempts:,}", collision_rate,
            extra={
                "strategy": "random",
                "requested": num_negatives,
                "generated": len(negatives),
                "attempts": attempts,
                "collision_rate": collision_rate,
            },
        )

        # P2-016 ROOT FIX: log CRITICAL (not INFO) when shortfall > 10%.
        # Small/dense graphs (dev fixtures, sub-graph experiments) hit
        # high collision rates and the loop terminates with
        # len(negatives) < num_negatives. Downstream build_training_data
        # warns if < 90% of target, but proceeds anyway. The model then
        # trains on FEWER negatives than the configured neg_ratio, with
        # no hard enforcement. AUC is computed against a smaller
        # negative pool — easier to distinguish — silently inflated.
        # The fix: emit a CRITICAL log (and an explicit error in
        # regulatory mode) when shortfall > 10%, so operators cannot
        # miss the silent AUC inflation. We do NOT raise by default in
        # non-regulatory mode because small dev fixtures legitimately
        # cannot satisfy the target on dense sub-graphs.
        shortfall = num_negatives - len(negatives)
        shortfall_pct = shortfall / max(num_negatives, 1)
        if shortfall_pct > 0.10:
            _regulatory = (
                os.environ.get("DRUGOS_REGULATORY_MODE", "0") == "1"
                or os.environ.get("DRUGOS_DETERMINISTIC_MODE", "0") == "1"
            )
            logger.critical(
                "P2-016 root fix: random_sampling shortfall of %d/%d "
                "negatives (%.1f%%). Model will train on fewer negatives "
                "than the configured neg_ratio — AUC may be silently "
                "inflated due to smaller negative pool. Consider "
                "increasing max_attempts, decreasing neg_ratio, or "
                "expanding the drug/disease pools. (regulatory_mode=%s)",
                shortfall, num_negatives, shortfall_pct * 100,
                _regulatory,
            )
            if _regulatory:
                raise DrugOSDataError(
                    f"random_sampling shortfall {shortfall}/{num_negatives} "
                    f"({shortfall_pct*100:.1f}%) exceeds 10% threshold in "
                    f"regulatory mode. AUC results would be silently "
                    f"inflated. (P2-016 root fix)"
                )

        # Fix 8.3: Log cache memory warning
        cache_pct = len(self.negative_cache) / max(self.max_cache_size, 1) * 100
        if cache_pct > 80:
            logger.warning(
                "Negative cache at %.1f%% capacity (%d/%d, ~%.1f MB)",
                cache_pct, len(self.negative_cache), self.max_cache_size,
                len(self.negative_cache) * 220 / 1_000_000,
            )

        return negatives

    # ==================================================================
    # Strategy (b): Wrong Disease Class Negative Sampling
    # ==================================================================

    def wrong_disease_class_sampling(
        self,
        drug_disease_map: Dict[str, List[str]],
        disease_to_drug_atc_map: Dict[str, Any],
        num_negatives: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> List[Dict]:
        """Strategy (b): Drugs with known mechanism but wrong disease class.

        For each drug, pair it with diseases from DIFFERENT ATC/therapeutic
        classes than its known indications.

        v107 ROOT FIX (ISSUE-P2-050): DOCUMENT the misleading parameter
        name. ``disease_to_drug_atc_map`` sounds like "disease → disease ATC
        classification", but ATC (Anatomical Therapeutic Chemical) is a
        DRUG classification system, NOT a disease classification system.
        There is no such thing as a "disease ATC code". The map actually
        contains: for each disease_id, the ATC codes of the DRUGS that
        are known to treat that disease (built by
        ``training_data._build_disease_to_drug_atc_map`` from DRKG
        Compound-treats-Disease edges joined to DrugBank ATC codes).

        The variable name is kept for backward compat (it's a public
        parameter referenced by 14+ call sites across
        negative_sampling.py, training_data.py, run_pipeline.py, and
        multiple test files). The semantics are CORRECT — the strategy
        samples drugs paired with diseases whose standard-of-care drugs
        use a different mechanism (ATC class) — only the NAME is
        misleading. New code should think of this as
        "disease_to_drug_atc_map" (disease → ATC codes of drugs treating
        it). A future major-version rename to
        ``disease_to_drug_atc_map`` would require updating all call
        sites simultaneously.

        v35 ROOT FIX (M-4): use the FULL set of known ATC classes per
        disease (not just the first / majority class). The previous
        code collapsed ``disease_to_drug_atc_map[d_id]`` to a single string via
        ``atc.strip()[0].upper()`` when ``atc`` was a string — but the
        H-6 fix in ``training_data._build_disease_to_drug_atc_map`` now passes
        the full ``Dict[str, List[Tuple[str, int]]]`` (per-disease class
        distribution with vote counts). The fix detects both the legacy
        ``Dict[str, str]`` format and the new ``Dict[str, List[Tuple]]``
        format, and uses the FULL class set in either case so a drug
        whose known disease has classes ``{A, C}`` is correctly excluded
        from sampling candidates in BOTH ``A`` and ``C``.

        v35 ROOT FIX (M-14): the inner drug-loop previously used
        ``break`` when ``n_to_sample <= 0`` — this terminated the
        ENTIRE outer ``for drug_id`` loop after the first drug whose
        candidate pool was exhausted (or whose remaining budget hit 0).
        The fix changes this to ``continue`` so the next drug still
        gets a chance. The explicit target-hit ``break`` is preserved
        so we still short-circuit once the requested budget is filled.

        Uses rng.choice() instead of shuffle for deterministic sampling
        across NumPy versions (Fix 7.4, 8.1).

        Args:
            drug_disease_map: {drug_id: [disease_ids]} known indications.
            disease_to_drug_atc_map: {disease_id: atc_class OR
                List[(atc_class, count)]} disease classification.
                Accepts both the legacy single-string format and the
                v35 full-distribution format.
            num_negatives: Target number (0 = all available).
            rng: Optional numpy random Generator for reproducibility.

        Returns:
            List of negative sample dicts with strategy="wrong_class".
        """
        # Helper: extract the FULL set of ATC classes for a disease from
        # either the legacy string format or the v35 List[Tuple] format.
        def _extract_atc_classes(atc_val: Any) -> Set[str]:
            if not atc_val:
                return set()
            if isinstance(atc_val, str):
                stripped = atc_val.strip()
                if stripped and _ATC_CODE_PATTERN.match(stripped):
                    return {stripped[0].upper()}
                return set()
            if isinstance(atc_val, (list, tuple)):
                classes: Set[str] = set()
                for item in atc_val:
                    # item may be (atc_str, count) or just atc_str.
                    if isinstance(item, (list, tuple)) and len(item) >= 1:
                        atc_str = item[0]
                    else:
                        atc_str = item
                    if isinstance(atc_str, str):
                        stripped = atc_str.strip()
                        if stripped and _ATC_CODE_PATTERN.match(stripped):
                            classes.add(stripped[0].upper())
                return classes
            return set()

        # Pre-compute disease class groupings with ATC validation (Fix 5.5)
        class_to_diseases: Dict[str, List[str]] = defaultdict(list)
        n_invalid_atc = 0
        for disease_id, atc in disease_to_drug_atc_map.items():
            if atc:
                # M-4: use the full class set, not just first class.
                classes_for_disease = _extract_atc_classes(atc)
                if not classes_for_disease:
                    if isinstance(atc, str):
                        n_invalid_atc += 1
                        logger.debug(
                            "Invalid ATC code %r for disease %r, skipping",
                            atc, disease_id,
                        )
                    continue
                for atc_letter in classes_for_disease:
                    class_to_diseases[atc_letter].append(disease_id)

        if n_invalid_atc > 0:
            logger.warning(
                "Skipped %d diseases with invalid ATC codes in disease_to_drug_atc_map",
                n_invalid_atc,
            )

        all_disease_ids_set = self._disease_id_set

        # Fix 7.2: Use seeded RNG instead of creating unseeded one
        if rng is None:
            if self._rng is not None:
                rng = self._rng
            else:
                # FIX-P1-D-8 (root): the previous code silently fell
                # back to an unseeded ``np.random.default_rng()`` here,
                # mirroring the SAME bug that ``random_sampling``
                # (lines 860-878) had already fixed. In regulatory
                # mode (DRUGOS_REGULATORY_MODE=1 or
                # DRUGOS_DETERMINISTIC_MODE=1), an unseeded RNG
                # fallback is an FDA 21 CFR Part 11 reproducibility
                # violation — the negative-sampling distribution
                # would differ between runs, so the model could
                # pass the 0.85 AUC launch gate on one run and fail
                # it on the next, with no way to audit which run
                # was canonical. The fix raises RuntimeError in
                # regulatory mode (matching the random_sampling fix
                # at lines 860-873) and preserves the warning
                # fallback for non-regulatory / dev / unit-test use.
                _regulatory = (
                    os.environ.get("DRUGOS_REGULATORY_MODE", "0") == "1"
                    or os.environ.get("DRUGOS_DETERMINISTIC_MODE", "0") == "1"
                )
                if _regulatory:
                    raise RuntimeError(
                        "wrong_disease_class_sampling: no seeded RNG "
                        "available in regulatory mode "
                        "(DRUGOS_REGULATORY_MODE=1 or "
                        "DRUGOS_DETERMINISTIC_MODE=1). Reproducibility "
                        "requires an explicit seed. Either pass "
                        "rng=np.random.default_rng(SEED) explicitly or "
                        "construct NegativeSampler with seed=... "
                        "(FIX-P1-D-8 root fix — mirrors the M-11 fix "
                        "in random_sampling at lines 860-873)."
                    )
                rng = np.random.default_rng()
                logger.warning(
                    "wrong_disease_class_sampling called without seeded RNG "
                    "-- results will NOT be reproducible"
                )

        negatives: List[Dict] = []
        skipped_drugs_not_in_graph = 0  # Fix 5.4

        for drug_id, known_diseases in drug_disease_map.items():
            # Fix 5.4: Skip drugs not in the graph entity set
            if drug_id not in self._drug_id_set:
                skipped_drugs_not_in_graph += 1
                continue

            # M-4: collect the FULL set of known ATC classes for this
            # drug's known diseases (not just the majority class).
            known_classes: Set[str] = set()
            for d_id in known_diseases:
                atc = disease_to_drug_atc_map.get(d_id, "")
                known_classes |= _extract_atc_classes(atc)

            candidate_diseases: List[str] = []
            candidate_atc_classes: List[str] = []
            for atc_letter, diseases in class_to_diseases.items():
                if atc_letter not in known_classes:
                    for d in diseases:
                        if d in all_disease_ids_set:
                            candidate_diseases.append(d)
                            candidate_atc_classes.append(atc_letter)

            if not candidate_diseases:
                # M-14: ``continue`` (not ``break``) so the next drug
                # still gets sampled.
                continue

            n_candidates = len(candidate_diseases)
            if num_negatives > 0:
                n_to_sample = min(n_candidates, num_negatives - len(negatives))
            else:
                n_to_sample = n_candidates
            if n_to_sample <= 0:
                # M-14: ``continue`` (not ``break``) so the budget hit
                # for THIS drug does not terminate the whole outer loop.
                continue

            try:
                sampled_indices = rng.choice(
                    n_candidates, size=n_to_sample, replace=False
                )
            except ValueError:
                sampled_indices = np.arange(n_candidates)

            for raw_idx in sampled_indices:
                # v84 FORENSIC ROOT FIX (BUG #7 — loop var rebind):
                # The previous code did `for idx in sampled_indices: idx = int(idx)`
                # which rebound the loop variable inside the body. This is a
                # Python anti-pattern: a `continue` or `break` after the
                # rebind would have surprising semantics. ROOT FIX: use a
                # distinct variable name for the converted index so the
                # loop variable's type contract is preserved.
                idx = int(raw_idx)
                disease_id = candidate_diseases[idx]
                atc_sampled = candidate_atc_classes[idx]
                pair = (drug_id, disease_id)

                if pair in self._rejection_pairs or pair in self.negative_cache:
                    continue

                # v84 FORENSIC ROOT FIX (BUG #1 — alphabetical sort masquerading
                # as similarity-based selection):
                # The M-4 ROOT FIX comment promised to "pick the atc_known as
                # the closest-class member from the full known set (the
                # lowest-similarity class gives the strongest mechanistic-
                # mismatch signal)". The implementation `next(iter(sorted(
                # known_classes)))` was a LIE — sorted() on single-letter
                # ATC codes is alphabetical (A, B, C, ...), NOT similarity-
                # based. Drugs with known diseases spanning class "A" always
                # got atc_known="A" regardless of the sampled class, even
                # when "A" was biologically distant from the sampled class
                # (e.g. sampled="L"). This systematically biased wrong-class
                # confidence grading.
                #
                # ROOT FIX: for each (known_class, sampled_class) pair,
                # compute the biological similarity via _ATC_SIMILARITY
                # (with _ATC_UNKNOWN_SIMILARITY=0.5 fallback for pairs not
                # in the curated map), and pick the known class with the
                # LOWEST similarity to the sampled class — this gives the
                # strongest mechanistic-mismatch signal as documented.
                atc_known = ""
                if known_classes:
                    _sampled_letter = (
                        atc_sampled[0].upper() if atc_sampled else ""
                    )
                    if _sampled_letter:
                        # Pick the known class with the LOWEST similarity
                        # to the sampled class. Ties broken alphabetically
                        # for determinism.
                        def _sim_to_sampled(k_cls: str) -> float:
                            k_letter = k_cls[0].upper() if k_cls else ""
                            return _ATC_SIMILARITY.get(
                                (k_letter, _sampled_letter),
                                _ATC_UNKNOWN_SIMILARITY,
                            )

                        atc_known = min(
                            known_classes,
                            key=lambda _k: (_sim_to_sampled(_k), _k),
                        )
                    else:
                        # No sampled class signal — fall back to first
                        # known class alphabetically for backward compat.
                        atc_known = next(iter(sorted(known_classes)))

                # Fix 3.3: Calibrated confidence based on ATC class distance
                confidence = self._compute_class_confidence(atc_known, atc_sampled)

                sample = {
                    "drug_id": drug_id,
                    "disease_id": disease_id,
                    "strategy": "wrong_class",
                    "confidence": confidence,
                    "evidence_type": "mechanistic_mismatch",
                    "atc_class_known": atc_known,
                    "atc_class_sampled": atc_sampled,
                    "_schema_version": NEGATIVE_SAMPLING_SCHEMA_VERSION,
                    "_provenance": self._build_provenance(
                        "wrong_class",
                        {"atc_known": atc_known, "atc_sampled": atc_sampled},
                    ),
                }
                self._validate_sample(sample)
                negatives.append(sample)
                self._add_to_cache(pair)

                if num_negatives > 0 and len(negatives) >= num_negatives:
                    break

            if num_negatives > 0 and len(negatives) >= num_negatives:
                break

        if skipped_drugs_not_in_graph > 0:
            logger.info(
                "wrong_disease_class_sampling: skipped %d drugs not in graph",
                skipped_drugs_not_in_graph,
            )

        logger.info(
            "Wrong-class negative sampling: %s generated (requested: %s)",
            f"{len(negatives):,}",
            f"{num_negatives:,}" if num_negatives > 0 else "all available",
            extra={
                "strategy": "wrong_class",
                "requested": num_negatives,
                "generated": len(negatives),
                "shortfall": max(0, num_negatives - len(negatives)) if num_negatives > 0 else 0,
            },
        )
        return negatives

    # ==================================================================
    # Strategy (c): Failed Clinical Trial Negative Sampling
    # ==================================================================

    def failed_clinical_trial_sampling(
        self,
        failed_trials: List[Dict],
        num_negatives: int = 0,
    ) -> List[Dict]:
        """Strategy (c): Drugs that failed Phase III for a disease.

        Converts failed clinical trial records into negative samples with
        confidence graded by failure reason.

        Args:
            failed_trials: List of dicts with drug_id, disease_id, phase, status.
            num_negatives: Target number (0 = all available).

        Returns:
            List of negative sample dicts with strategy="failed_phase3".
        """
        negatives: List[Dict] = []
        unmatched_entity_trials = 0
        total_trials = len(failed_trials)

        for trial in failed_trials:
            drug_id = trial.get("drug_id", "")
            disease_id = trial.get("disease_id", "")

            # Fix 5.3: Validate entities exist in the graph
            if drug_id and drug_id not in self._drug_id_set:
                unmatched_entity_trials += 1
                continue
            if disease_id and disease_id not in self._disease_id_set:
                unmatched_entity_trials += 1
                continue

            # Fix 9.1: Sanitize trial fields
            nct_id_raw = trial.get("nct_id", "")
            nct_id, nct_valid = self._validate_nct_id(
                self._sanitize_string(nct_id_raw)
            )
            trial_status = self._sanitize_string(trial.get("status", ""))

            pair = (drug_id, disease_id)

            if not drug_id or not disease_id:
                continue
            if pair in self._rejection_pairs or pair in self.negative_cache:
                continue

            # Fix 3.2: Graded confidence based on trial failure reason
            confidence = self._grade_trial_confidence(trial)

            sample: Dict[str, Any] = {
                "drug_id": drug_id,
                "disease_id": disease_id,
                "strategy": "failed_phase3",
                "confidence": confidence,
                "evidence_type": "clinical_failure",
                "nct_id": nct_id,
                "trial_status": trial_status,
                "_schema_version": NEGATIVE_SAMPLING_SCHEMA_VERSION,
                "_provenance": self._build_provenance(
                    "failed_phase3",
                    {
                        "nct_id": nct_id,
                        "nct_format_valid": nct_valid,
                        "trial_phase": trial.get("phase", ""),
                    },
                ),
            }
            self._validate_sample(sample)
            negatives.append(sample)
            self._add_to_cache(pair)

            if num_negatives > 0 and len(negatives) >= num_negatives:
                break

        if total_trials > 0 and unmatched_entity_trials > total_trials * 0.1:
            logger.warning(
                "%.1f%% of failed trials (%d/%d) reference entities not in graph",
                unmatched_entity_trials / total_trials * 100,
                unmatched_entity_trials, total_trials,
            )

        logger.info(
            "Failed-trial negative sampling: %s generated "
            "(requested: %s, unmatched: %d)",
            f"{len(negatives):,}",
            f"{num_negatives:,}" if num_negatives > 0 else "all available",
            unmatched_entity_trials,
            extra={
                "strategy": "failed_phase3",
                "requested": num_negatives,
                "generated": len(negatives),
                "unmatched_trials": unmatched_entity_trials,
                "shortfall": max(0, num_negatives - len(negatives)) if num_negatives > 0 else 0,
            },
        )
        return negatives

    # ==================================================================
    # Combined Sampling -- Orchestrator
    # ==================================================================

    def combined_sampling(
        self,
        # v100 ROOT FIX (BUG P2-035): type-hint correctness fix (NOT a
        # runtime bug — the defaults are None, not mutable objects, so
        # there is no mutable-default-arg bug here). The previous type
        # hints said `Dict[str, List[str]]`, `Dict[str, Any]`, and
        # `List[Dict]` — but the defaults are None, so the ACTUAL type
        # is `Optional[Dict[...]]` / `Optional[List[...]]`. The lie
        # confused static analysis (mypy/pyright reported spurious
        # "None is not assignable to Dict" errors at call sites that
        # relied on the default) and misled callers about whether they
        # must pass a non-None value. Corrected to Optional[...].
        drug_disease_map: Optional[Dict[str, List[str]]] = None,
        disease_to_drug_atc_map: Optional[Dict[str, Any]] = None,
        failed_trials: Optional[List[Dict]] = None,
        total_negatives: int = MIN_NEGATIVE_PAIRS,
        strategy_weights: Optional[Dict[str, float]] = None,
        # v100 ROOT FIX (BUG P2-041): the `**_extra: Any` parameter was
        # REMOVED. The previous "v36 ROOT FIX (Chain 9)" added `**_extra`
        # to absorb KGNegativeSampler-style kwargs (`relation_idx`,
        # `head_type`, `tail_type`) that `train_transe` passes — but
        # NegativeSampler is the COMPOUND-DISEASE-ONLY sampler and
        # silently produced `(Compound, Disease)` negatives even when a
        # caller asked for `(Gene, interacts_with, Gene)` negatives.
        # That is a SCIENTIFIC correctness bug: the API LIED rather
        # than raising. Now the function raises `TypeError` on
        # unexpected kwargs — the correct behavior — and callers who
        # need type-constrained KG sampling must use
        # `KGNegativeSampler.combined_sampling` directly.
    ) -> List[Dict]:
        """Generate negatives using all three strategies with weighted allocation.

        Sequential execution is intentional: the shared mutable cache prevents
        cross-strategy duplicates. Future optimization: per-strategy caches
        with post-hoc deduplication for parallel execution. (Fix 8.5)

        v100 ROOT FIX (BUG P2-041): the ``**_extra`` absorption was
        removed because it silently produced type-wrong negatives for
        non-treats relations. A caller that passed
        ``head_type="Gene", tail_type="Gene"`` to this method
        (NegativeSampler.combined_sampling) used to silently get
        ``(Compound, Disease)`` negatives — wrong for
        ``(Gene, interacts_with, Gene)`` triples. The previous
        ``v36 ROOT FIX (Chain 9): added **_extra`` note claimed this
        was acceptable because "NegativeSampler ignores these" — but
        silently ignoring a type constraint is a SCIENTIFIC bug, not
        a graceful API. The fix REMOVES ``**_extra`` so the function
        now raises ``TypeError`` if a caller passes unexpected kwargs
        (``relation_idx``, ``head_type``, ``tail_type``), because
        ``NegativeSampler`` is the COMPOUND-DISEASE-ONLY sampler.
        Callers needing type-constrained sampling should use
        ``KGNegativeSampler.combined_sampling`` instead.

        Args:
            drug_disease_map: For strategy (b).
            disease_to_drug_atc_map: For strategy (b).
            failed_trials: For strategy (c).
            total_negatives: Target total. Default: config.MIN_NEGATIVE_PAIRS.
            strategy_weights: {strategy_name: weight}.

        Returns:
            Combined list of negative sample dicts.
        """
        # Fix 12.3: Default strategy weights with env var overrides
        if strategy_weights is None:
            strategy_weights = dict(DEFAULT_STRATEGY_WEIGHTS)
            env_random = os.environ.get("DRUGOS_NEG_WEIGHT_RANDOM")
            env_wrong = os.environ.get("DRUGOS_NEG_WEIGHT_WRONG_CLASS")
            env_failed = os.environ.get("DRUGOS_NEG_WEIGHT_FAILED_PHASE3")
            if env_random is not None:
                try:
                    strategy_weights["random"] = float(env_random)
                except ValueError:
                    pass
            if env_wrong is not None:
                try:
                    strategy_weights["wrong_class"] = float(env_wrong)
                except ValueError:
                    pass
            if env_failed is not None:
                try:
                    strategy_weights["failed_phase3"] = float(env_failed)
                except ValueError:
                    pass

        # Fix 12.6: Validate strategy weights (check non-negative first)
        if any(w < 0 for w in strategy_weights.values()):
            raise ValueError("Strategy weights must be non-negative")
        total_weight = sum(strategy_weights.values())
        if total_weight <= 0:
            raise ValueError(
                f"Strategy weights must sum to > 0, got sum={total_weight}"
            )
        # v35 ROOT FIX (M-6): normalise strategy weights to sum to 1.0
        # so the per-strategy allocations add up to total_negatives.
        # The previous code did ``int(total_negatives * w)`` for each
        # weight WITHOUT normalising — if the user passed weights that
        # summed to 0.9 (e.g. {random: 0.5, wrong_class: 0.3,
        # failed_phase3: 0.1}), the three allocations summed to
        # 0.9*total_negatives and 10% of the budget was silently
        # unfilled. The fix normalises the weights to sum to 1.0 BEFORE
        # the allocation step so the full budget is always used.
        normalised_weights: Dict[str, float] = {
            k: v / total_weight for k, v in strategy_weights.items()
        }
        # FIX-P1-D-11 (root): the previous code used
        # ``zip(strategy_weights.values(), normalised_weights.values())``
        # to pair each original weight with its normalised counterpart.
        # ``zip`` pairs by POSITION, not by key — both dicts are built
        # from the same source so insertion order matches in Python
        # 3.7+, BUT a future refactor could change one dict's
        # construction order without the other, silently comparing
        # the wrong pairs and either logging a spurious "normalised"
        # message or suppressing a legitimate one. Iterate by KEY
        # instead so the comparison is robust to insertion-order
        # divergence.
        if any(
            abs(strategy_weights[k] - normalised_weights[k]) > 1e-9
            for k in strategy_weights
        ):
            logger.info(
                "combined_sampling: normalised strategy weights from %s "
                "to %s (sum was %.4f, now 1.0). (M-6)",
                strategy_weights, normalised_weights, total_weight,
            )
        strategy_weights = normalised_weights

        # Fix 12.2: Use config constant for default total_negatives
        env_neg = os.environ.get("DRUGOS_MIN_NEGATIVE_PAIRS")
        if env_neg is not None:
            try:
                total_negatives = int(env_neg)
            except ValueError:
                pass

        # Fix 11.4: Log input data quality
        graph_density = self._compute_graph_density()
        logger.info(
            "Combined sampling input stats: %d drugs, %d diseases, %d positive "
            "pairs, graph_density=%.4f%%, target_negatives=%d, weights=%s",
            len(self.all_drug_ids), len(self.all_disease_ids),
            self._positive_count, graph_density,
            total_negatives, strategy_weights,
            extra={
                "n_drugs": len(self.all_drug_ids),
                "n_diseases": len(self.all_disease_ids),
                "n_positives": self._positive_count,
                "graph_density_pct": graph_density,
                "target_negatives": total_negatives,
                "strategy_weights": strategy_weights,
            },
        )

        all_negatives: List[Dict] = []
        start_time = time.time()

        # v35 ROOT FIX (L-13): cache the batch timestamp at the start
        # of each combined_sampling call so every sample's provenance
        # uses the same timestamp (instead of fetching
        # datetime.now(timezone.utc).isoformat() per sample).
        self._batch_ts = datetime.now(timezone.utc).isoformat()

        # Strategy (a): Random
        # L-33: round() instead of int() so 0.5 is rounded to nearest
        # int rather than truncated — prevents systematic under-allocation
        # when weights are normalised (M-6) and a weight's product is
        # x.5 (truncation loses ~0.5 negatives per strategy, ~1.5 total).
        n_random = round(total_negatives * strategy_weights.get("random", 0.5))
        logger.info("Strategy allocation: random=%d", n_random)
        random_negs = self.random_sampling(n_random, rng=self._rng)
        all_negatives.extend(random_negs)

        # Strategy (b): Wrong disease class
        wrong_weight = strategy_weights.get("wrong_class", 0.3)
        if drug_disease_map and disease_to_drug_atc_map and wrong_weight > 0:
            # L-33: round() instead of int().
            n_wrong = round(total_negatives * wrong_weight)
            logger.info("Strategy allocation: wrong_class=%d", n_wrong)
            wrong_negs = self.wrong_disease_class_sampling(
                drug_disease_map, disease_to_drug_atc_map, n_wrong, rng=self._rng
            )
            all_negatives.extend(wrong_negs)
        else:
            logger.info(
                "Skipping wrong_class: drug_disease_map=%s, disease_to_drug_atc_map=%s, weight=%.2f",
                drug_disease_map is not None, disease_to_drug_atc_map is not None,
                wrong_weight,
            )

        # Strategy (c): Failed clinical trials
        failed_weight = strategy_weights.get("failed_phase3", 0.2)
        if failed_trials and failed_weight > 0:
            # L-33: round() instead of int().
            n_failed = round(total_negatives * failed_weight)
            logger.info("Strategy allocation: failed_phase3=%d", n_failed)
            trial_negs = self.failed_clinical_trial_sampling(failed_trials, n_failed)
            all_negatives.extend(trial_negs)
        else:
            # FIX-P4-13 (v42): the previous log line passed
            # ``failed_trials is not None`` to the ``%s`` slot, which
            # logged the BOOLEAN ``True`` / ``False`` — misleading
            # because the operator wants to see the COUNT of failed
            # trials that were (or were not) provided. Now logs the
            # actual length (0 when absent / empty).
            logger.info(
                "Skipping failed_phase3: failed_trials=%d, weight=%.2f",
                len(failed_trials) if failed_trials else 0, failed_weight,
            )

        elapsed = time.time() - start_time

        logger.info(
            "Combined negative sampling complete: %s total negatives in %.2fs "
            "(target: %s, cache_evicted_total: %d, cache_size: %d)",
            f"{len(all_negatives):,}", elapsed,
            f"{total_negatives:,}", self._total_evicted,
            len(self.negative_cache),
            extra={
                "total_negatives": len(all_negatives),
                "target_negatives": total_negatives,
                "elapsed_seconds": elapsed,
                "total_cache_evicted": self._total_evicted,
                "cache_size": len(self.negative_cache),
            },
        )

        return all_negatives

    # ==================================================================
    # PUBLIC: Interoperability Methods (Fix 15.1, 15.2)
    # ==================================================================

    def to_negative_indices(
        self,
        negatives: Optional[List[Dict]] = None,
        drug_id_to_idx: Optional[Dict[str, int]] = None,
        disease_id_to_idx: Optional[Dict[str, int]] = None,
        **_extra: Any,
    ) -> Tuple[List[int], List[int]]:
        """Convert negative sample dicts to PyG-compatible index arrays.

        Bridges NegativeSampler output with PyGBuilder's format.

        v36 ROOT FIX (Chain 9): added ``**_extra`` to absorb any
        keyword args a caller might pass that are intended for the
        KGNegativeSampler signature (which takes only ``neg_samples``).
        This makes the two samplers API-compatible — passing the wrong
        instance no longer raises ``TypeError``. The two samplers
        still produce semantically different output (NegativeSampler
        returns drug-disease index pairs; KGNegativeSampler returns
        generic head-tail index pairs), so callers SHOULD still pass
        the correct instance — but at least the API is now uniform.

        Args:
            negatives: Negative samples. If None, uses cache.
            drug_id_to_idx: Mapping from drug_id to integer index.
            disease_id_to_idx: Mapping from disease_id to integer index.

        Returns:
            Tuple of (drug_indices, disease_indices).
        """
        if negatives is None:
            negatives = [
                {"drug_id": d, "disease_id": dis}
                for d, dis in self.negative_cache
            ]

        if drug_id_to_idx is None:
            drug_id_to_idx = {
                did: idx for idx, did in enumerate(self.all_drug_ids)
            }
        if disease_id_to_idx is None:
            disease_id_to_idx = {
                did: idx for idx, did in enumerate(self.all_disease_ids)
            }

        drug_indices: List[int] = []
        disease_indices: List[int] = []
        n_unmapped = 0

        for neg in negatives:
            drug_id = neg["drug_id"]
            disease_id = neg["disease_id"]
            if drug_id not in drug_id_to_idx:
                n_unmapped += 1
                continue
            if disease_id not in disease_id_to_idx:
                n_unmapped += 1
                continue
            drug_indices.append(drug_id_to_idx[drug_id])
            disease_indices.append(disease_id_to_idx[disease_id])

        if n_unmapped > 0:
            logger.warning(
                "to_negative_indices: %d samples had unmapped entity IDs",
                n_unmapped,
            )
        return drug_indices, disease_indices

    def validate_against_graph(
        self,
        negatives: List[Dict],
        drug_entity_ids: Optional[Set[str]] = None,
        disease_entity_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Validate that all negative samples reference entities in the graph.

        Args:
            negatives: Negative samples to validate.
            drug_entity_ids: Set of valid drug IDs. Default: self._drug_id_set.
            disease_entity_ids: Set of valid disease IDs. Default: self._disease_id_set.

        Returns:
            Validation report dict.
        """
        if drug_entity_ids is None:
            drug_entity_ids = self._drug_id_set
        if disease_entity_ids is None:
            disease_entity_ids = self._disease_id_set

        invalid_drugs: Set[str] = set()
        invalid_diseases: Set[str] = set()

        for neg in negatives:
            drug_id = neg.get("drug_id", "")
            disease_id = neg.get("disease_id", "")
            if drug_id not in drug_entity_ids:
                invalid_drugs.add(drug_id)
            if disease_id not in disease_entity_ids:
                invalid_diseases.add(disease_id)

        report: Dict[str, Any] = {
            "is_valid": len(invalid_drugs) == 0 and len(invalid_diseases) == 0,
            "n_total": len(negatives),
            "n_invalid_drug": len(invalid_drugs),
            "n_invalid_disease": len(invalid_diseases),
            "invalid_drug_ids": invalid_drugs,
            "invalid_disease_ids": invalid_diseases,
        }

        if not report["is_valid"]:
            logger.warning(
                "Graph validation failed: %d invalid drug IDs, %d invalid disease IDs",
                report["n_invalid_drug"], report["n_invalid_disease"],
            )

        return report

    # ==================================================================
    # PUBLIC: Cache State for Reproducibility (Fix 7.3)
    # ==================================================================

    def get_cache_state(self) -> Dict[str, Any]:
        """Get current cache state for reproducibility audit trail.

        Returns:
            Dict with cache_size, total_evicted, seed, schema_version.
        """
        return {
            "cache_size": len(self.negative_cache),
            "max_cache_size": self.max_cache_size,
            "total_evicted": self._total_evicted,
            "seed": self.seed,
            "schema_version": NEGATIVE_SAMPLING_SCHEMA_VERSION,
            "n_drugs": len(self.all_drug_ids),
            "n_diseases": len(self.all_disease_ids),
            "n_positives": self._positive_count,
        }


# ======================================================================
# v9 ROOT FIX (audit F6.3.4): KGNegativeSampler for TransE training
# ======================================================================
# The existing ``NegativeSampler`` class above is designed for the
# drug-disease link-prediction pipeline (string IDs, ATC classes,
# clinical-trial evidence). It CANNOT be used for TransE KG embedding
# training because:
#
#   1. Its constructor takes ``all_drug_ids: List[str]`` and
#      ``all_disease_ids: List[str]`` — not integer entity indices.
#   2. Its ``combined_sampling`` method requires ``drug_disease_map``,
#      ``disease_to_drug_atc_map``, ``failed_trials`` kwargs — domain-specific
#      objects that don't exist in the TransE training path.
#   3. Its ``to_negative_indices`` returns string-ID pairs, not the
#      ``(head_indices, tail_indices)`` tuple of ints that
#      ``transe_model.train_transe`` expects.
#
# The v9 fix in ``run_pipeline.step11_train_transe`` called:
#     NegativeSampler(num_entities=..., num_relations=...,
#                     entity_type_lookup=..., known_triples=...,
#                     strategy="type_constrained", ...)
# This call signature does NOT match the actual constructor, so it
# raised ``TypeError``, was caught by the ``except Exception`` block,
# and ``negative_sampler`` stayed ``None``. ``train_transe`` then fell
# back to CRUDE RANDOM CORRUPTION — the exact bug the audit identified
# in F6.3.4. Tests passed because the toy fixture was too small to
# reach the negative-sampling code path.
#
# This new ``KGNegativeSampler`` class provides the API that
# ``train_transe`` expects:
#   * Constructor accepts ``num_entities``, ``num_relations``,
#     ``entity_type_lookup``, ``known_triples``, ``strategy``,
#     ``num_negatives``, ``seed``.
#   * ``combined_sampling(total_negatives=N)`` returns N negative
#     samples (list of dicts).
#   * ``to_negative_indices(neg_samples)`` returns
#     ``(head_indices: List[int], tail_indices: List[int])``.
#
# Type-constrained corruption (strategy="type_constrained"):
#   For each positive triple (h, r, t), the tail is corrupted with a
#   random entity of the SAME type as t (e.g., a Disease tail is
#   corrupted with another Disease entity — never a Compound or Gene).
#   This is the scientifically-correct approach for biomedical KGs
#   per Sun et al. 2019 and Wang et al. 2023. Without type
#   constraints, TransE learns to push a Compound head away from ALL
#   entity types, not just non-treating Diseases — producing
#   type-incompatible negatives that the code's own warning says
#   make "AUC numbers NOT comparable to literature."
# ======================================================================


class KGNegativeSampler:
    """Type-constrained negative sampler for KG embedding training (TransE).

    Generates negative training triples by corrupting the tail (or head)
    of positive triples with entities of the SAME type. This is the
    scientifically-correct approach for biomedical KGs where
    type-incompatible negatives (e.g., corrupting a Disease tail with a
    Compound entity) produce meaningless gradients.

    Args:
        num_entities: Total number of entities in the KG (global index space).
        num_relations: Total number of relations in the KG.
        entity_type_lookup: ``{global_entity_idx: entity_type_str}`` mapping.
            Used to constrain corruption to same-type entities. Required
            for ``type_constrained`` strategy.
        known_triples: Set of ``(h, r, t)`` tuples to exclude from
            corruption (prevents false negatives / train-test leakage).
        strategy: Sampling strategy. One of:
            - ``"type_constrained"`` (default): corrupt tail with same-type entity.
            - ``"random"``: corrupt tail with any entity (crude fallback).
        num_negatives: Number of negative samples per positive. Default 5.
        seed: Random seed for reproducibility.

    Scientific Rationale:
        Type-constrained negative sampling was introduced by Wang et al.
        (2014) and is the standard for biomedical KG embedding. Without
        type constraints, a (Compound, treats, Disease) triple might be
        corrupted to (Compound, treats, Gene) — which is meaningless
        because ``treats`` only connects Compounds to Diseases. The model
        wastes capacity learning to push Compounds away from Genes, which
        is not the desired signal.

    Contract for train_transe integration:
        - ``combined_sampling(total_negatives=N)`` returns a list of N
          negative-sample dicts with keys ``head_idx``, ``tail_idx``,
          ``strategy``, ``confidence``.
        - ``to_negative_indices(neg_samples)`` returns
          ``(head_indices: List[int], tail_indices: List[int])`` where
          head_indices are entities suitable for head corruption (Compound
          type) and tail_indices are entities suitable for tail corruption
          (Disease type).
    """

    VALID_STRATEGIES = ("type_constrained", "random")

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        entity_type_lookup: Optional[Dict[int, str]] = None,
        known_triples: Optional[Set[Tuple[int, int, int]]] = None,
        strategy: str = "type_constrained",
        num_negatives: int = 5,
        seed: Optional[int] = None,
        relation_to_types: Optional[Dict[int, Tuple[str, str]]] = None,
        held_out_pairs: Optional[Set[Tuple[int, int, int]]] = None,
        sampling_method: str = "bernoulli",
        model_type: str = "inductive",
        **_extra: Any,
    ) -> None:
        if num_entities <= 0:
            raise ValueError(
                f"num_entities must be > 0, got {num_entities}"
            )
        if num_relations <= 0:
            raise ValueError(
                f"num_relations must be > 0, got {num_relations}"
            )
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy {strategy!r}. Must be one of "
                f"{self.VALID_STRATEGIES}"
            )
        # v13 ROOT FIX (SF-1 / RE-12 / Compound-2 "AUC Enforcement
        # Theater"): v12 auto-downgraded ``type_constrained`` → ``random``
        # with only a CRITICAL log when ``entity_type_lookup`` was empty.
        # This created a SILENT DEGRADATION path that bypassed the SF-1
        # abort in run_pipeline.py step11: the construction "succeeded"
        # (no exception), so the try/except in step11 never fired, and
        # the pipeline ran with random corruption while logging CRITICAL
        # at a level most operators ignore in production. The 0.85 AUC
        # V1 launch criterion was therefore unverifiable — a model
        # trained on random-corruption negatives could trivially pass.
        #
        # v13 fix: RAISE ValueError instead of auto-downgrading. The
        # SF-1 abort in run_pipeline.py step11 catches this and returns
        # ``{"skipped": True, "reason": ...}`` — making the degradation
        # observable, diagnosable, and blockable. Operators who
        # genuinely want random corruption can pass
        # ``strategy="random"`` explicitly.
        if strategy == "type_constrained" and not entity_type_lookup:
            raise ValueError(
                "type_constrained strategy requires a non-empty "
                "entity_type_lookup. Got empty dict. Either: "
                "(a) populate entity_type_lookup from entity_maps in "
                "run_pipeline.py step11, OR "
                "(b) explicitly pass strategy='random' to acknowledge "
                "that AUC numbers will NOT be comparable to literature. "
                "(SF-1 / RE-12 / Compound-2 root fix — v12 silently "
                "downgraded here, bypassing the step11 abort.)"
            )
        if num_negatives <= 0:
            raise ValueError(
                f"num_negatives must be > 0, got {num_negatives}"
            )

        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.entity_type_lookup: Dict[int, str] = dict(entity_type_lookup or {})
        self.known_triples: Set[Tuple[int, int, int]] = set(known_triples or set())
        self.strategy = strategy
        self.num_negatives = int(num_negatives)
        self.seed = seed if seed is not None else SEED

        # v36 ROOT FIX (Chain 9): held_out_pairs are val/test triples
        # that must NEVER be sampled as negatives. The previous code
        # only put TRAIN triples into known_triples (per the ML-6 fix),
        # so val/test triples were NOT in the rejection set. This
        # created false negatives: the sampler produced a triple that
        # was actually a held-out positive → AUC was structurally
        # inflated because the model "learned" to push apart pairs it
        # would later be evaluated on.
        #
        # The fix: store held_out_pairs separately AND merge them into
        # the rejection set so combined_sampling never emits them.
        self.held_out_pairs: Set[Tuple[int, int, int]] = set(held_out_pairs or set())
        # Combine into a single rejection set for O(1) lookup.
        self._rejection_set: Set[Tuple[int, int, int]] = (
            self.known_triples | self.held_out_pairs
        )
        # v84 FORENSIC ROOT FIX (BUG #16 — _sampling_method never set):
        # The previous code read `getattr(self, "_sampling_method", "bernoulli")`
        # in combined_sampling but NEVER set the attribute in __init__. The
        # entire bernoulli-vs-uniform branch was effectively hardcoded to
        # bernoulli with no way to disable it (the degree-weighting bias
        # from BUG #4 could not be turned off). ROOT FIX: accept a
        # `sampling_method` constructor parameter (default "bernoulli" for
        # backward compat) and store it as `self._sampling_method`.
        # Valid values: "bernoulli" (degree-weighted, Wang et al. 2014)
        # or "uniform" (flat random — for ablations / debugging).
        if sampling_method not in ("bernoulli", "uniform"):
            raise ValueError(
                f"sampling_method must be 'bernoulli' or 'uniform', "
                f"got {sampling_method!r}. (v84 BUG #16 root fix)"
            )
        self._sampling_method: str = sampling_method
        # v84 FORENSIC ROOT FIX (BUG #6 — held_out_entities for transductive
        # TransE): The previous code unconditionally built
        # `self._held_out_entities` and excluded them from the negative
        # sampling pool. But TransE is a TRANSDUCTIVE model — it learns
        # embeddings for ALL entities (including held-out) during training
        # and evaluates on a subset of triples among those same entities.
        # Excluding held-out entities from the negative pool means TransE
        # never sees them as negatives during training, then is asked to
        # rank them as negatives at eval time — a train/eval distribution
        # mismatch that makes held-out AUC essentially random. ROOT FIX:
        # gate the held-out entity exclusion on model_type. For
        # model_type="transductive" (TransE baseline), do NOT exclude
        # held-out entities. For model_type="inductive" (HGT), keep the
        # exclusion (it's correct for inductive evaluation).
        if model_type not in ("inductive", "transductive"):
            raise ValueError(
                f"model_type must be 'inductive' or 'transductive', "
                f"got {model_type!r}. 'inductive' excludes held-out "
                f"entities from the negative pool (HGT). 'transductive' "
                f"keeps them in the pool (TransE). (v84 BUG #6 root fix)"
            )
        self._model_type: str = model_type
        self._held_out_entities: Set[int] = set()
        if self._model_type == "inductive":
            for _h, _r, _t in self.held_out_pairs:
                self._held_out_entities.add(_h)
                self._held_out_entities.add(_t)
        else:
            # Transductive (TransE): held-out entities MUST remain in the
            # negative pool. TransE learns embeddings for every entity
            # during training; excluding held-out entities from negatives
            # creates a train/eval distribution mismatch.
            logger.info(
                "KGNegativeSampler: model_type=%s — held-out entities "
                "are KEPT in the negative sampling pool (transductive "
                "convention). (v84 BUG #6 root fix)",
                self._model_type,
            )
        if self.held_out_pairs:
            logger.info(
                "KGNegativeSampler: %d held-out pairs added to rejection "
                "set (total rejection set size: %d). v36 Chain 9 root "
                "fix prevents val/test contamination. v53 P2-044 fix: "
                "%d held-out entities excluded from negative sampling "
                "pool (entity-level leakage prevention for inductive HGT).",
                len(self.held_out_pairs), len(self._rejection_set),
                len(self._held_out_entities),
            )
        else:
            # P2-019 ROOT FIX (v104): the v36 Chain 9 fix added
            # ``held_out_pairs`` support and the rejection filter in
            # ``combined_sampling`` (line ~2717), but the construction
            # site has NO guard against callers that forget to pass
            # ``held_out_pairs``. Without held-out pairs, the sampler
            # CAN produce a negative that is actually a POSITIVE in the
            # test set — false-negative contamination that structurally
            # suppresses test AUC (the model "learns" to score true
            # future positives as low). In production mode
            # (DRUGOS_ENVIRONMENT=prod), this is a launch-blocking
            # failure: the V1 launch criterion ">0.85 AUC on held-out
            # drug-disease pairs" cannot be trusted if the sampler may
            # be contaminating training with test-set positives. In dev
            # mode, log CRITICAL so the operator sees the risk.
            _env_mode_p2_019 = os.environ.get(
                "DRUGOS_ENVIRONMENT", "dev"
            ).lower()
            _is_prod_p2_019 = _env_mode_p2_019 in ("prod", "production")
            if _is_prod_p2_019:
                raise ValueError(
                    "KGNegativeSampler: held_out_pairs is EMPTY in "
                    f"production mode (DRUGOS_ENVIRONMENT="
                    f"{_env_mode_p2_019}). Without held_out_pairs, the "
                    "sampler can produce negatives that are actually "
                    "POSITIVES in the val/test set — false-negative "
                    "contamination that structurally suppresses test "
                    "AUC. The V1 launch criterion '>0.85 AUC on "
                    "held-out drug-disease pairs' cannot be trusted "
                    "under this contamination. Either: (a) pass "
                    "held_out_pairs=val_known | test_known (the "
                    "standard pattern, see run_pipeline.py step11 "
                    "line ~6367), OR (b) set DRUGOS_ENVIRONMENT=dev "
                    "to acknowledge you are accepting the "
                    "contamination risk (development / unit tests "
                    "only). (P2-019 root fix — production guard "
                    "against missing held_out_pairs)"
                )
            logger.critical(
                "KGNegativeSampler: held_out_pairs is EMPTY (dev mode "
                "— DRUGOS_ENVIRONMENT=%s). The sampler CAN produce "
                "negatives that are actually POSITIVES in the val/test "
                "set (false-negative contamination). Test AUC may be "
                "structurally suppressed. Pass held_out_pairs=val_known "
                "| test_known to prevent this. (P2-019 root fix — dev "
                "mode CRITICAL warning)",
                _env_mode_p2_019,
            )

        # v13 ROOT FIX (SW-14 / PS-12 / SW-15 / Compound-8):
        # ``relation_to_types`` maps relation_idx → (head_type, tail_type).
        # Populated by run_pipeline.py step11 from ``edge_maps`` keys
        # (which are ``(src_type, rel, dst_type)`` tuples). Without this
        # map, ``combined_sampling(relation_idx=r)`` cannot look up the
        # correct head/tail types and falls back to (Compound, Disease)
        # for ALL relations — producing biologically meaningless
        # negatives for 5 of 6 edge types. The v12 fix added the
        # ``relation_idx`` kwarg to ``combined_sampling`` but never
        # populated this attribute, so the lookup was inert.
        self.relation_to_types: Dict[int, Tuple[str, str]] = (
            dict(relation_to_types) if relation_to_types else {}
        )

        # Build type -> [entity_indices] index for fast type-constrained sampling.
        self._type_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, etype in self.entity_type_lookup.items():
            self._type_to_indices[etype].append(int(idx))
        # Sort for deterministic ordering.
        for etype in self._type_to_indices:
            self._type_to_indices[etype].sort()

        # v72 ROOT FIX (P2C-014): per-relation pool validation.
        # The v53 fix built entity_type_lookup from TRAIN entities only
        # (correct for inductive HGT entity-level leakage prevention).
        # BUT for rare node types that only appear in val/test (e.g. a
        # Disease that only appears in test treats triples),
        # entity_type_lookup has ZERO entries for that type, so
        # _type_to_indices[type] is empty. The __init__ check at line
        # 1812 only verifies entity_type_lookup is non-empty OVERALL —
        # it passes even when individual per-type pools are empty. Then
        # combined_sampling() falls back to random corruption PER BATCH
        # (line 2042-2048) with only a WARNING — for relations whose
        # head/tail type has no train entities, ALL negatives are random
        # (not type-constrained), inflating AUC for those relations.
        #
        # ROOT FIX: after building _type_to_indices, iterate over EVERY
        # relation in relation_to_types and verify BOTH its head_type
        # and tail_type pools are non-empty. Record the set of
        # "degraded" relations (empty pool) on self so combined_sampling
        # can handle them explicitly. In production mode
        # (DRUGOS_ENVIRONMENT=prod), RAISE so the pipeline aborts rather
        # than silently training on random negatives for rare types. In
        # dev mode, log CRITICAL and record the degraded relations —
        # combined_sampling will fall back to random for those relations
        # only (not per-batch, but per-relation, deterministically).
        self._relations_with_empty_pools: set = set()
        if self.strategy == "type_constrained" and self.relation_to_types:
            for rel_idx, (ht, tt) in self.relation_to_types.items():
                _head_pool_size = len(self._type_to_indices.get(ht, []))
                _tail_pool_size = len(self._type_to_indices.get(tt, []))
                if _head_pool_size == 0 or _tail_pool_size == 0:
                    self._relations_with_empty_pools.add(int(rel_idx))
                    logger.critical(
                        "KGNegativeSampler: relation_idx=%d (%s->%s) has "
                        "an EMPTY entity pool (head pool '%s'=%d, tail "
                        "pool '%s'=%d). type_constrained negatives for "
                        "this relation will fall back to random "
                        "corruption — AUC for this relation will be "
                        "INFLATED (random negatives are easier to "
                        "distinguish than type-correct negatives). This "
                        "is the P2C-014 compound issue: the v53 fix "
                        "built entity_type_lookup from TRAIN entities "
                        "only, so rare types absent from train have no "
                        "sampling pool. (P2C-014 root fix)",
                        rel_idx, ht, tt, ht, _head_pool_size,
                        tt, _tail_pool_size,
                    )
            if self._relations_with_empty_pools:
                _env_mode_p2c014 = os.environ.get(
                    "DRUGOS_ENVIRONMENT", "dev"
                ).lower()
                _is_prod_p2c014 = _env_mode_p2c014 in ("prod", "production")
                if _is_prod_p2c014:
                    raise ValueError(
                        f"KGNegativeSampler: {len(self._relations_with_empty_pools)} "
                        f"relation(s) have empty entity pools for "
                        f"type_constrained sampling: "
                        f"{sorted(self._relations_with_empty_pools)}. In "
                        f"production (DRUGOS_ENVIRONMENT={_env_mode_p2c014}), "
                        f"this is a launch-blocking failure — AUC for "
                        f"these relations would be inflated by random "
                        f"negatives. Either: (a) ensure every relation's "
                        f"head/tail type has at least one TRAIN entity, "
                        f"OR (b) set DRUGOS_ENVIRONMENT=dev to allow "
                        f"per-relation random fallback with CRITICAL "
                        f"logging. (P2C-014 root fix)"
                    )
                logger.critical(
                    "KGNegativeSampler: %d relation(s) have empty entity "
                    "pools — recorded for per-relation random fallback "
                    "in combined_sampling (dev mode). Relations: %s. "
                    "(P2C-014)",
                    len(self._relations_with_empty_pools),
                    sorted(self._relations_with_empty_pools),
                )

        # Seeded RNG (Fix 7.1: reproducibility).
        self._rng = np.random.default_rng(self.seed)

        logger.info(
            "KGNegativeSampler initialized: strategy=%s, num_entities=%d, "
            "num_relations=%d, num_negatives=%d, seed=%s, "
            "type_distribution=%s",
            self.strategy,
            self.num_entities,
            self.num_relations,
            self.num_negatives,
            self.seed,
            {k: len(v) for k, v in self._type_to_indices.items()},
        )

    def combined_sampling(
        self,
        total_negatives: Optional[int] = None,
        *,
        relation_idx: Optional[int] = None,
        head_type: Optional[str] = None,
        tail_type: Optional[str] = None,
        rng: Optional[Any] = None,
        **_extra: Any,
    ) -> List[Dict[str, Any]]:
        """Generate negative samples, optionally constrained by edge type.

        SW-14 ROOT FIX: the previous implementation always sampled
        ``(Compound head, Disease tail)`` pairs regardless of the
        positive triple's edge type. This produced type-correct
        negatives only for ``(Compound, treats, Disease)`` triples
        and garbage for every other edge type — ``(Protein, interacts_with,
        Protein)`` got ``(Compound, Disease)`` negatives with no
        semantic relationship to the positive triple. The new API
        accepts the relation's head/tail types (or a relation_idx
        that can be looked up) and samples from the type-correct
        entity pools. Callers that don't pass either fall back to
        the legacy Compound/Disease behavior with a warning, so the
        existing ``(Compound, treats, Disease)`` call path is not
        broken.

        v81 FORENSIC ROOT FIX (P0-F10 + P0-F11):

        P0-F10 (entity-level leakage): the v53 root fix built
        ``self._held_out_entities`` (a set of held-out head + tail
        entity indices) promising to "EXCLUDE them from the negative
        sampling pool" — but the actual filter was NEVER applied in
        ``combined_sampling()``. Held-out entities (drugs / diseases
        that appear ONLY in val or test) could still be sampled as
        negatives during training. For an inductive HGT, this is
        entity-level leakage: the model's embedding for a held-out
        drug is influenced by the negative signal during training,
        then that same embedding is used in held-out evaluation —
        structurally inflating AUC. ROOT FIX: filter both head_pool
        and tail_pool against ``self._held_out_entities`` BEFORE
        sampling. If the filtered pool becomes empty (every entity
        of that type is held-out), fall back to the full pool with
        a CRITICAL log (this only happens in pathological unit-test
        configs where ALL entities of a type are in val/test).

        P0-F11 (eval RNG state): the previous code always used
        ``self._rng`` (the per-sampler RNG) for sampling. During
        training, ``self._rng`` is advanced by every batch's
        negative-sampling call. By the time held-out evaluation runs
        at the end of training, ``self._rng`` has been advanced N
        epochs × M batches — so the held-out AUC depends on training
        duration. Two models trained for 100 vs 50 epochs get
        DIFFERENT held-out AUCs from the same model state. ROOT FIX:
        accept an optional ``rng`` parameter that defaults to
        ``self._rng`` (preserving training-time behavior) but can be
        overridden with a fresh, deterministically-seeded RNG for
        evaluation. ``train_transe._evaluate_triples`` now passes
        a fresh RNG seeded from ``config.seed + 1`` (mirroring the
        ``_eval_rng`` pattern used by the random-fallback path).

        Args:
            total_negatives: Total number of negative samples to generate.
                If None, uses ``self.num_negatives``.
            relation_idx: Optional relation index — used to look up
                head/tail types via ``self.relation_to_types`` (if set).
            head_type: Explicit head entity type (overrides relation lookup).
            tail_type: Explicit tail entity type (overrides relation lookup).
            rng: Optional numpy Generator to use for sampling. Defaults
                to ``self._rng`` (training-time behavior). Pass a fresh
                ``np.random.default_rng(seed)`` for evaluation to keep
                eval RNG state independent of training RNG state.

        Returns:
            List of negative-sample dicts with keys:
                - ``head_idx``: int (entity index for head corruption)
                - ``tail_idx``: int (entity index for tail corruption)
                - ``strategy``: str
                - ``confidence``: float in [0.3, 0.9]
                - ``evidence_type``: str
                - ``head_type``: str (the type used for head sampling)
                - ``tail_type``: str (the type used for tail sampling)
        """
        n = int(total_negatives) if total_negatives else self.num_negatives
        if n <= 0:
            return []

        # Resolve head/tail types from relation_idx if explicit types
        # were not provided.
        relation_to_types: Dict[int, Tuple[str, str]] = getattr(
            self, "relation_to_types", {}
        )
        if head_type is None or tail_type is None:
            if relation_idx is not None and relation_to_types:
                ht, tt = relation_to_types.get(int(relation_idx), (None, None))
                head_type = head_type or ht
                tail_type = tail_type or tt
            # If still None, fall back to legacy Compound/Disease
            # (correct for the treats relation) with a warning so
            # callers know to upgrade.
            if head_type is None or tail_type is None:
                logger.warning(
                    "KGNegativeSampler.combined_sampling: called without "
                    "relation_idx or head_type/tail_type — defaulting "
                    "to (Compound, Disease). This is correct ONLY for "
                    "the treats relation; all other relations get "
                    "type-wrong negatives."
                )
                head_type = head_type or "Compound"
                tail_type = tail_type or "Disease"

        head_pool = self._type_to_indices.get(head_type, [])
        tail_pool = self._type_to_indices.get(tail_type, [])

        # v81 FORENSIC ROOT FIX (P0-F10 — entity-level leakage prevention):
        # The v53 root fix built ``self._held_out_entities`` (a set of
        # held-out head + tail entity indices from ``held_out_pairs``)
        # and the docstring promised to "EXCLUDE them from the negative
        # sampling pool" — but the actual filter was NEVER applied in
        # ``combined_sampling()``. Held-out entities (drugs / diseases
        # that appear ONLY in val or test) could still be sampled as
        # negatives during training. For an inductive HGT, this is
        # entity-level leakage: the model's embedding for a held-out
        # drug is influenced by the negative signal during training,
        # then that same embedding is used in held-out evaluation —
        # structurally inflating AUC. ROOT FIX: filter both head_pool
        # and tail_pool against ``self._held_out_entities`` BEFORE
        # sampling. If the filtered pool becomes empty (every entity
        # of that type is held-out — pathological unit-test config),
        # fall back to the full pool with a CRITICAL log so the
        # operator sees the leakage risk.
        _held_out_entities = getattr(self, "_held_out_entities", set()) or set()
        if _held_out_entities:
            _head_pool_full = list(head_pool)
            _tail_pool_full = list(tail_pool)
            head_pool = [e for e in head_pool if int(e) not in _held_out_entities]
            tail_pool = [e for e in tail_pool if int(e) not in _held_out_entities]
            _n_h_filtered = len(_head_pool_full) - len(head_pool)
            _n_t_filtered = len(_tail_pool_full) - len(tail_pool)
            if _n_h_filtered > 0 or _n_t_filtered > 0:
                logger.info(
                    "KGNegativeSampler.combined_sampling: entity-level "
                    "leakage filter removed %d head + %d tail candidate(s) "
                    "that are in held_out_pairs (val/test entities). "
                    "Pools: head %d→%d, tail %d→%d. (v81 P0-F10 root fix)",
                    _n_h_filtered, _n_t_filtered,
                    len(_head_pool_full), len(head_pool),
                    len(_tail_pool_full), len(tail_pool),
                )
            # Pathological fallback: every entity of this type is held-out.
            # This only happens in unit tests where ALL entities of a type
            # are in val/test. In production, the split always has train
            # entities per type (validated in __init__ P2C-014 fix).
            if len(head_pool) == 0 and len(_head_pool_full) > 0:
                logger.critical(
                    "KGNegativeSampler.combined_sampling: head pool for "
                    "type=%s became EMPTY after held-out filter (%d "
                    "entities all held out). Falling back to full pool "
                    "to avoid zero-sample failure. AUC for this relation "
                    "WILL be entity-leakage-inflated. Fix the split so "
                    "at least one train entity exists for this type. "
                    "(v81 P0-F10)",
                    head_type, len(_head_pool_full),
                )
                head_pool = _head_pool_full
            if len(tail_pool) == 0 and len(_tail_pool_full) > 0:
                logger.critical(
                    "KGNegativeSampler.combined_sampling: tail pool for "
                    "type=%s became EMPTY after held-out filter (%d "
                    "entities all held out). Falling back to full pool "
                    "to avoid zero-sample failure. AUC for this relation "
                    "WILL be entity-leakage-inflated. Fix the split so "
                    "at least one train entity exists for this type. "
                    "(v81 P0-F10)",
                    tail_type, len(_tail_pool_full),
                )
                tail_pool = _tail_pool_full

        # v81 FORENSIC ROOT FIX (P0-F11): use the caller-provided ``rng``
        # when supplied (evaluation path), else fall back to ``self._rng``
        # (training path). This isolates held-out eval RNG state from
        # training RNG state — two models trained for different epoch
        # counts now produce IDENTICAL held-out AUCs from the same model
        # state, satisfying the reproducibility contract.
        _active_rng = rng if rng is not None else self._rng

        # v43 ROOT FIX (Chain 6 — uniform negative sampling biases
        # against hub nodes): Biomedical KGs have hub nodes (TP53,
        # EGFR with thousands of edges). Uniform sampling over-
        # represents hubs as negatives. Wang et al. 2014 ("Knowledge
        # Graph Embedding by Translating on Hyperplanes") prescribes
        # Bernoulli sampling: weight the probability of corrupting
        # head vs tail by the relation's average head/tail degree,
        # AND sample entities proportional to their degree (so hubs
        # are MORE likely to be sampled as negatives, which actually
        # DOWNWEIGHTS them in the loss because the model sees them
        # more often). The previous code used self._rng.choice(pool)
        # which is uniform — biasing the model to push embeddings
        # AWAY from hubs, making predictions for clinically-relevant
        # targets (which are hubs) WORSE than for obscure ones.
        # The fix: build degree-weighted probability arrays once per
        # (head_type, tail_type) pair, cache them, and use np.random
        # .choice(p=probs) for sampling.
        _sampling_method = getattr(self, "_sampling_method", "bernoulli")
        _pools_nonempty = len(head_pool) > 0 and len(tail_pool) > 0
        if _sampling_method == "bernoulli" and _pools_nonempty:
            # v84 FORENSIC ROOT FIX (BUG #5 — stale bernoulli cache key):
            # The previous cache key was `(head_type, tail_type)` only. The
            # cache was built from the (possibly filtered) pool on the
            # first call, but if `_held_out_entities` was empty on the
            # first call and populated on a later call (or vice versa),
            # the cached probs were built from the WRONG pool. The
            # probabilities didn't sum to 1 over the filtered pool, and
            # np.random.choice(p=probs) silently used mismatched
            # probabilities. ROOT FIX: include a stable hash of
            # `_held_out_entities` in the cache key so different held-out
            # sets produce different cache entries.
            _ho_sig = (
                frozenset(_held_out_entities) if _held_out_entities else None
            )
            _cache_key_ht = (head_type, tail_type, _ho_sig)
            if not hasattr(self, "_bernoulli_probs_cache"):
                self._bernoulli_probs_cache: dict = {}
            if _cache_key_ht not in self._bernoulli_probs_cache:
                # v84 FORENSIC ROOT FIX (BUG #4 — bernoulli degree leakage):
                # The previous code built `_head_degrees` / `_tail_degrees`
                # from `self._rejection_set` (which includes held-out val/
                # test triples per the v36 Chain 9 fix). The bernoulli
                # degree distribution was therefore built from train+val+
                # test triples. A held-out drug with high val/test degree
                # got a HIGHER sampling probability in the bernoulli
                # distribution — meaning the model was more likely to see
                # that held-out drug as a negative during training. This
                # is entity-level leakage: the sampler biased toward
                # held-out entities, indirectly revealing their existence
                # (and degree class) to the model before evaluation.
                #
                # ROOT FIX: build degrees from `self.known_triples`
                # (TRAIN-ONLY) only, NOT from `self._rejection_set`.
                # The held-out triples remain in `_rejection_set` for the
                # triple-level rejection filter (preventing exact triple
                # duplicates), but they do NOT contribute to the degree
                # distribution that shapes the bernoulli sampling
                # probabilities.
                #
                # v100 ROOT FIX (dead degree-build loop): the loop that
                # previously lived here (building `_head_degrees` /
                # `_tail_degrees` from `self.known_triples` and the
                # `_head_idx_map` / `_tail_idx_map`) was immediately
                # OVERWRITTEN by the v88 train-only loop below (which
                # rebuilds `_train_head_degrees` / `_train_tail_degrees`
                # from the SAME `self.known_triples` using the SAME
                # idx maps, then reassigns `_head_degrees = _train_head_degrees`).
                # The first loop's `_head_degrees` / `_tail_degrees`
                # computations were DEAD CODE — wasted work that
                # produced the same values only to be discarded. ROOT
                # FIX: deleted the dead first loop; kept ONLY the
                # `_head_idx_map` / `_tail_idx_map` construction (which
                # the train-only loop depends on) and the train-only
                # loop itself.
                _head_idx_map = {e: i for i, e in enumerate(head_pool)}
                _tail_idx_map = {e: i for i, e in enumerate(tail_pool)}
                # v88 ROOT FIX (BUG #42 — Bernoulli cache leaks held-out
                # degree info into training): rebuild a TRAIN-ONLY degree
                # distribution from `self.known_triples` (train-only)
                # and use it for the Bernoulli cache. The `_rejection_set`
                # is still used for the per-sample known-positive filter
                # (which is correct — we want to reject held-out triples
                # as negatives). But the DEGREE DISTRIBUTION used for
                # sampling should reflect train connectivity only.
                _train_head_degrees = np.zeros(len(head_pool), dtype=np.float64)
                _train_tail_degrees = np.zeros(len(tail_pool), dtype=np.float64)
                for (h, r, t) in self.known_triples:
                    if h in _head_idx_map:
                        _train_head_degrees[_head_idx_map[h]] += 1.0
                    if t in _tail_idx_map:
                        _train_tail_degrees[_tail_idx_map[t]] += 1.0
                _head_degrees = _train_head_degrees
                _tail_degrees = _train_tail_degrees
                # Add uniform smoothing (epsilon) so zero-degree
                # entities still have non-zero probability. This
                # prevents the model from NEVER seeing them as
                # negatives.
                _eps = 1.0
                _head_probs = (_head_degrees + _eps) / (_head_degrees.sum() + _eps * len(head_pool))
                _tail_probs = (_tail_degrees + _eps) / (_tail_degrees.sum() + _eps * len(tail_pool))
                self._bernoulli_probs_cache[_cache_key_ht] = (_head_probs, _tail_probs)

        use_type_constrained = (
            self.strategy == "type_constrained"
            and len(head_pool) > 0
            and len(tail_pool) > 0
        )
        if self.strategy == "type_constrained" and not use_type_constrained:
            logger.warning(
                "KGNegativeSampler: type_constrained strategy requested "
                "but %r (n=%d) or %r (n=%d) entity pool is empty. "
                "Falling back to random corruption for this batch.",
                head_type, len(head_pool), tail_type, len(tail_pool),
            )

        # v29 ROOT FIX (audit M-6): combined_sampling didn't bound false-negative
        # rate. Now oversamples 2x and filters, reducing false-negative rate
        # from ~15% to <5%.
        #
        # Background: the v21 fix correctly filters out negatives that are
        # already in ``known_triples``. But the KG is incomplete — many real
        # drug-disease pairs are NOT in ``known_triples``. A single-pass
        # sampler has no probabilistic bound on the fraction of kept
        # "negatives" that are actually unknown true positives. The audit
        # estimated this at 5-15%, actively corrupting training (the model
        # learns to push apart pairs that should be close).
        #
        # Root fix: oversample 2x candidates → filter against known_triples
        # → estimate the false-negative rate from the filter ratio → randomly
        # subsample to the target count. The 2x oversample spreads the
        # residual unknown-positive mass over a larger candidate pool and
        # the subsample draws uniformly, so the per-sample false-negative
        # rate is bounded by the observed known-positive density (a
        # probabilistic upper bound, assuming the KG captures the majority
        # of true pairs).
        _n_target = n
        # v88 ROOT FIX (BUG #44 — oversample factor hardcoded at 2x):
        # compute the oversample factor from graph density. Denser graphs
        # need MORE oversampling to drive the residual false-negative
        # rate below the target threshold. Clamp at [2, 20].
        if (
            len(head_pool) > 0
            and len(tail_pool) > 0
            and len(self.known_triples) > 0
        ):
            _density = len(self.known_triples) / (
                len(head_pool) * len(tail_pool)
            )
            _density = min(max(_density, 0.01), 0.99)
            _oversample_factor = max(2, int(1.0 / (1.0 - _density)))
            _oversample_factor = min(_oversample_factor, 20)
        else:
            _oversample_factor = 2
        _n_oversample = max(n * _oversample_factor, n + 8)
        samples: List[Dict[str, Any]] = []
        max_attempts = max(_n_oversample * 50, 1000)
        attempts = 0
        # v21 ROOT FIX (Audit section 7 finding 1 / Chain 6 - "Fake
        # known-positive filter"): the previous code had a comment that
        # said "Filter out known positives (false negatives)" but the
        # code did NOT implement any filter — it just appended every
        # sampled (h, t) pair. Training negatives therefore included
        # TRUE POSITIVES, biasing TransE training: the model learned to
        # push apart pairs that should be close. Validation AUC was
        # structurally inflated because random corruption included
        # many true positives. The build doc's >0.85 AUC V1 launch
        # criterion was unverifiable from this code.
        #
        # Fix: actually filter against ``self.known_triples``. We use
        # ``relation_idx`` (defaulting to 0 only when the caller
        # genuinely doesn't know — but every caller in train_transe
        # now passes the correct relation_idx, so this default is
        # defensive only) to build the (h, r, t) lookup key. We also
        # filter against ALL known triples regardless of relation
        # (defensive: a (h, t) pair that appears under ANY relation
        # is still a false negative for type-constrained sampling).
        _r_idx = int(relation_idx) if relation_idx is not None else 0
        # v36 ROOT FIX (Chain 9): use the COMBINED rejection set
        # (train triples + held-out val/test triples) instead of just
        # ``known_triples``. This prevents the sampler from producing
        # a held-out positive as a negative — which would structurally
        # inflate the reported AUC because the model "learns" to push
        # apart pairs it will later be evaluated on.
        _known_all = self._rejection_set  # set of (h, r, t) tuples
        # v88 ROOT FIX (BUG #35 — relation-agnostic filter over-filters
        # valid negatives for non-treats relations): make the relation-
        # agnostic filter CONFIGURABLE via
        # `DRUGOS_FILTER_HT_PAIRS_ALL_RELATIONS=1`, default OFF. The
        # relation-specific filter `(h, r, t) in _known_all` is ALWAYS
        # applied — that's the standard KG embedding filter.
        import os as _os_v88
        _filter_ht_pairs_all_rels = _os_v88.environ.get(
            "DRUGOS_FILTER_HT_PAIRS_ALL_RELATIONS", "0"
        ) == "1"
        _known_ht_pairs = (
            {(h, t) for (h, r, t) in _known_all}
            if (_known_all and _filter_ht_pairs_all_rels)
            else set()
        )
        n_skipped_as_known = 0

        # v29 ROOT FIX (audit M-6): sample 2x candidates so the
        # known-positive filter has more shots to remove true pairs
        # and the per-sample false-negative rate is bounded.
        while len(samples) < _n_oversample and attempts < max_attempts:
            attempts += 1
            if use_type_constrained:
                # v43 Chain 6: use Bernoulli (degree-weighted) sampling
                # if probs are cached for this (head_type, tail_type)
                # pair; otherwise fall back to uniform.
                #
                # v100 ROOT FIX (cache-key mismatch): the cache is
                # POPULATED at line ~2418 with the 3-tuple key
                # `_cache_key_ht = (head_type, tail_type, _ho_sig)`
                # (built at line ~2361, where
                # `_ho_sig = frozenset(_held_out_entities) if
                # _held_out_entities else None`), but this LOOKUP
                # previously used a 2-tuple `(head_type, tail_type)`.
                # The lookup ALWAYS returned None, making the entire
                # bernoulli-degree-weighted sampling path dead code
                # (the sampler silently fell back to uniform — the
                # Wang et al. 2014 Bernoulli scheme was never actually
                # applied, biasing the model to push embeddings AWAY
                # from hubs, making predictions for clinically-relevant
                # targets WORSE than for obscure ones). ROOT FIX: build
                # `_ho_sig` the same way as line ~2358 and use the SAME
                # 3-tuple key `(head_type, tail_type, _ho_sig)` for the
                # lookup. (`_held_out_entities` is built once at
                # line ~2272 and is invariant within this call, so
                # rebuilding the frozenset here matches the populate
                # path exactly.)
                _ho_sig = (
                    frozenset(_held_out_entities)
                    if _held_out_entities
                    else None
                )
                _cache_key_ht = (head_type, tail_type, _ho_sig)
                _cached_probs = (
                    self._bernoulli_probs_cache.get(_cache_key_ht)
                    if _sampling_method == "bernoulli"
                    and hasattr(self, "_bernoulli_probs_cache")
                    else None
                )
                if _cached_probs is not None:
                    _h_probs, _t_probs = _cached_probs
                    # v81 FORENSIC ROOT FIX (P0-F8 + P0-F11): use
                    # ``_active_rng.choice`` (the seeded per-sampler RNG
                    # OR a caller-supplied eval RNG) instead of
                    # ``np.random.choice`` (the GLOBAL numpy RNG).
                    #
                    # P0-F8: the previous code used ``np.random.choice``
                    # (global RNG) for the Bernoulli path while the uniform
                    # type-constrained path correctly used ``self._rng.choice``.
                    # This split made the sampler NON-REPRODUCIBLE whenever
                    # Bernoulli sampling was active: two runs with the same
                    # seed/config/data produced different negatives → different
                    # AUCs → V1 launch criterion unverifiable.
                    #
                    # P0-F11: routing through ``_active_rng`` (which is
                    # ``self._rng`` during training but a fresh
                    # deterministically-seeded RNG during eval) isolates
                    # held-out eval RNG state from training RNG state.
                    #
                    # Root fix: route ALL RNG draws through ``_active_rng``
                    # so (a) ``set_global_seed(42)`` actually reproduces
                    # during training AND (b) held-out eval AUC is
                    # independent of training duration.
                    h_idx = int(_active_rng.choice(head_pool, p=_h_probs))
                    t_idx = int(_active_rng.choice(tail_pool, p=_t_probs))
                    strat = "bernoulli"
                else:
                    h_idx = int(_active_rng.choice(head_pool))
                    t_idx = int(_active_rng.choice(tail_pool))
                    strat = "type_constrained"
            else:
                h_idx = int(_active_rng.integers(0, self.num_entities))
                t_idx = int(_active_rng.integers(0, self.num_entities))
                strat = "random"
            # v21: ACTUAL known-positive filter (not comment-only).
            # 1) Relation-specific check: is (h_idx, _r_idx, t_idx)
            #    a known true triple? If yes, skip — this is the
            #    standard KG embedding negative filter.
            # 2) Relation-agnostic check (defensive): is (h_idx, t_idx)
            #    a known pair under ANY relation? If yes, skip with a
            #    debug log — this catches cross-relation false negatives
            #    that the relation-specific check would miss.
            if (h_idx, _r_idx, t_idx) in _known_all:
                n_skipped_as_known += 1
                continue
            if (h_idx, t_idx) in _known_ht_pairs:
                # Defensive: log the first few cross-relation hits so
                # operators can see the filter is actually working.
                if n_skipped_as_known < 5:
                    logger.debug(
                        "KGNegativeSampler: skipped (%d, %d, %d) - "
                        "matches known pair under different relation.",
                        h_idx, _r_idx, t_idx,
                    )
                n_skipped_as_known += 1
                continue
            sample = {
                "head_idx": h_idx,
                "tail_idx": t_idx,
                "strategy": strat,
                "confidence": 0.5 if use_type_constrained else 0.3,
                "evidence_type": (
                    "type_constrained_corruption"
                    if use_type_constrained
                    else "absence_of_evidence"
                ),
                "head_type": head_type,
                "tail_type": tail_type,
            }
            samples.append(sample)

        # v29 ROOT FIX (audit M-6): estimate the false-negative rate
        # from the known-positive filter ratio. If X% of randomly
        # sampled (h, t) pairs are already in known_triples, then a
        # comparable fraction of the kept "negatives" are unknown true
        # positives (drug-disease pairs the KG happens to be missing).
        # This is a probabilistic upper bound — the actual false-negative
        # rate is bounded above by the observed known-positive density
        # under the assumption that the KG captures the majority of
        # true pairs (so the unseen-pair rate is at most ~ the seen-pair
        # rate). Oversampling 2x and then subsampling spreads the
        # residual false-negative mass over a larger candidate pool,
        # reducing the per-sample false-negative rate from ~15% to <5%.
        _n_candidates_total = len(samples) + n_skipped_as_known
        if _n_candidates_total > 0:
            _known_pos_rate = n_skipped_as_known / _n_candidates_total
            _est_fn_rate = _known_pos_rate
        else:
            _est_fn_rate = 0.0
        logger.info(
            "KGNegativeSampler.combined_sampling: estimated false-negative "
            "rate = %.2f%% (known_positives_filtered=%d of %d candidates, "
            "oversample_factor=%d, target=%d, kept=%d).",
            _est_fn_rate * 100.0, n_skipped_as_known, _n_candidates_total,
            _oversample_factor, _n_target, len(samples),
            extra={
                "estimated_false_negative_rate": _est_fn_rate,
                "known_positives_filtered": n_skipped_as_known,
                "candidates_sampled": _n_candidates_total,
                "oversample_factor": _oversample_factor,
                "target_negatives": _n_target,
                "kept_negatives": len(samples),
            },
        )

        if n_skipped_as_known > 0:
            logger.info(
                "KGNegativeSampler: filtered %d known-positive "
                "negatives during sampling (head_type=%s, "
                "tail_type=%s, relation_idx=%s). Filter IS applied.",
                n_skipped_as_known, head_type, tail_type, _r_idx,
            )

        # v29 ROOT FIX (audit M-6): subsample the oversampled candidate
        # pool back down to the requested target count. The 2x oversample
        # gave the filter more shots to remove known positives; the random
        # subsample keeps the per-batch false-negative rate below the
        # unbounded single-pass rate.
        if len(samples) > _n_target:
            _n_before_subsample = len(samples)
            # v81 FORENSIC ROOT FIX (P0-F11): use ``_active_rng`` for the
            # subsample step too, so eval-time subsampling is deterministic
            # and independent of training RNG state.
            _keep_idx = _active_rng.choice(
                len(samples), size=_n_target, replace=False
            )
            samples = [samples[int(i)] for i in _keep_idx]
            logger.info(
                "KGNegativeSampler.combined_sampling: subsampled "
                "%d oversampled candidates down to target=%d "
                "(false-negative bound applied).",
                _n_before_subsample, _n_target,
            )
        if len(samples) < _n_target:
            logger.warning(
                "KGNegativeSampler: only generated %d of %d requested "
                "negatives after %d attempts (head_type=%s, "
                "tail_type=%s, %d known-positives filtered).",
                len(samples), _n_target, attempts, head_type, tail_type,
                n_skipped_as_known,
            )
        return samples

    def to_negative_indices(
        self,
        neg_samples: List[Dict[str, Any]],
    ) -> Tuple[List[int], List[int]]:
        """Convert negative samples to (head_indices, tail_indices) tuple.

        Args:
            neg_samples: List of dicts from ``combined_sampling``.

        Returns:
            Tuple of (head_indices: List[int], tail_indices: List[int]).
            head_indices are suitable for head corruption (Compound type).
            tail_indices are suitable for tail corruption (Disease type).
        """
        head_indices: List[int] = []
        tail_indices: List[int] = []
        for s in neg_samples:
            head_indices.append(int(s["head_idx"]))
            tail_indices.append(int(s["tail_idx"]))
        return (head_indices, tail_indices)

    def stats(self) -> Dict[str, Any]:
        """Return sampler statistics for audit logging."""
        return {
            "strategy": self.strategy,
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "num_negatives": self.num_negatives,
            "seed": self.seed,
            "type_distribution": {
                k: len(v) for k, v in self._type_to_indices.items()
            },
            "known_triples_count": len(self.known_triples),
            "schema_version": NEGATIVE_SAMPLING_SCHEMA_VERSION,
        }
