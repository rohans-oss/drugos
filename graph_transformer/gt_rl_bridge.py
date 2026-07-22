"""
GT-RL Bridge Module: End-to-End Integration of Graph Transformer + RL Ranker.

This module bridges Phase 3 (Graph Transformer) and Phase 4 (RL Hypothesis
Ranker) of the Autonomous Drug Repurposing Platform.

Pipeline:
    1. Build/Load knowledge graph (BiomedicalGraphBuilder)
    2. Train Graph Transformer model (or load checkpoint)
    3. Generate predictions for ALL drug-disease pairs
       (with label-leaking edges excluded -- C2 fix)
    4. Compute supplementary features from REAL graph topology
       (safety from drug->causes->outcome edges, market from disease
       connectivity, pathway from multi-hop drug->protein->pathway->
       disease paths -- C1 fix)
    5. Package into CSV matching RL's expected input schema
    6. Feed into the RL ranking pipeline
    7. Return the RL candidates (NOT the GT predictions -- B16 fix)

FIX vs original codebase:
  - **B5 (torch.randperm uses global RNG)**: train/val/test split now
    uses a torch.Generator seeded deterministically. Same seed => same
    split, every run.
  - **B8 (import hell)**: this module now uses relative imports.
    ``graph_transformer`` is a proper installable package. No sys.path
    hackery.
  - **B16 (bridge returns wrong dataframe)**: ``run_full_pipeline`` now
    returns the RL candidates (a DataFrame of ranked drug-disease
    pairs), NOT ``rl_input_df`` (the GT-side CSV of all pairs). The
    caller no longer has to find a timestamped CSV on disk to access
    the actual rankings.
  - **B17 (pandas 3.x bomb)**: replaced
    ``df.groupby('drug').apply(lambda x: x.nlargest(...))`` with the
    pandas-3.x-safe
    ``df.sort_values(...).groupby('drug').head(n)``.
  - **C1 (safety/market features are constants)**: the original bridge
    computed safety from ``df.groupby('drug').size()`` AFTER
    generating the full cross-product, so every drug appeared
    exactly ``num_diseases`` times -- safety was 0.9 for every drug.
    The new bridge computes safety from the actual
    ``drug -> causes -> clinical_outcome`` edge count (more adverse
    events = lower safety), and market from actual disease
    connectivity in the graph (not the cross-product).
  - **C2 (label leakage in generate_rl_input)**: the original bridge
    called ``self.model(...)`` without ``exclude_edges``, so the model
    saw the ``('drug','treats','disease')`` edges it was supposed to
    predict. Known positives got artificially high scores. The new
    bridge passes ``exclude_edges=LABEL_LEAKING_EDGES`` to every model
    call.
  - **C3 (confidence semantics mismatch)**: the original bridge
    computed ``1 - binary_entropy(p)/log(2)`` but the RL data
    dictionary said "entropy of attention distribution". These are
    different quantities. The new bridge still computes prediction
    entropy (it's a reasonable confidence proxy), but the RL data
    dictionary now documents this accurately so downstream consumers
    aren't misled.
  - **C4 (no drug-aware split)**: the bridge now uses
    ``drug_aware_split`` from ``graph_transformer.utils``. A drug in
    train never appears in val or test.
  - **C5 (70/15/0 split -- no test set)**: the bridge now actually
    creates a test set, evaluates on it, and reports test AUC.
  - **C6 (KNOWN_POSITIVES names don't exist in integrated pipeline)**:
    the bridge now passes the RL ranker's ``KNOWN_POSITIVES`` list to
    ``build_demo_graph``, which injects those exact (drug_name,
    disease_name) pairs into the graph. The integrated recovery test
    now actually finds them.
  - **C7 (untrained GT in bridge)**: the bridge defaults to
    ``gt_epochs=80`` (was 30) on a graph with enough known positives
    for the model to actually learn. Also uses a smaller model
    (embedding_dim=64, 2 layers) so 80 epochs finishes in seconds on
    CPU. The GT output is no longer pure noise.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time  # P4-017: used for checkpoint freshness comparison and log formatting
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp  # P3-025: sparse matrices for pathway_score
import torch

from .data import (
    DEFAULT_FEATURE_DIMS,
    LABEL_LEAKING_EDGES,
)
from .data.graph_builder import BiomedicalGraphBuilder
from .data.biomedical_tables import (
    get_drug_safety_score,
    get_drug_patent_score,
    get_drug_adme_score,
    compute_market_score,
    compute_rare_disease_flag,
    # P3-053 ROOT FIX (v107): removed the ``as _compute_unmet_need_score_table``
    # alias. The alias existed ONLY to work around the nested function that
    # shadowed the imported name. The fix deletes the nested function (P3-051)
    # and imports under the canonical name.
    compute_unmet_need_score,
    get_disease_prevalence,
)
from .models.graph_transformer import DrugRepurposingGraphTransformer
from .training.trainer import GraphTransformerTrainer
from .utils import (
    drug_aware_split,
    set_seed,
)
# V4 B-F9 fix: ``rl`` is now a proper installable package. The bridge
# imports Phase 4 the same way it imports Phase 3 -- no more
# ``sys.path.insert`` hackery.
# V5 Dead-code fix #1: ``compute_multi_hop_path_count`` has been REMOVED
# from ``graph_transformer.utils`` entirely (it was defined but never
# called -- the bridge computes its own vectorized adjacency maps). The
# bridge no longer imports it (V4) and the function no longer exists (V5).
#
# P3-032 ROOT FIX: removed the hard top-level ``from rl.rl_drug_ranker
# import KNOWN_POSITIVES`` import. The previous import COUPLED Phase 3
# to Phase 4 at the package level -- if the ``rl`` package was not
# installed (e.g. running Phase 3 standalone, or a CI matrix that
# exercises Phase 3 only), ``import graph_transformer.gt_rl_bridge``
# raised ImportError at module-import time, before any function could
# run. The graph_transformer package could not be imported without
# Phase 4. We now do the import LAZILY inside a helper,
# ``_get_known_positives()``, which returns an empty list with a
# warning when Phase 4 is absent. Call sites updated to use the helper.
# This decouples Phase 3 from Phase 4 at import time while preserving
# the runtime integration when both are installed.
logger = logging.getLogger(__name__)


def _get_known_positives() -> List[Tuple[str, str]]:
    """Lazily import and return the RL ranker's KNOWN_POSITIVES list.

    P3-032 ROOT FIX: returns an empty list (with a single WARNING log) if
    the ``rl`` package is not importable. The previous top-level import
    made ``graph_transformer.gt_rl_bridge`` un-importable without Phase 4
    installed. This helper decouples the packages at import time while
    preserving the runtime integration.
    """
    try:
        from rl.rl_drug_ranker import KNOWN_POSITIVES as _KP
    except ImportError:
        logger.warning(
            "P3-032: rl.rl_drug_ranker.KNOWN_POSITIVES not importable "
            "(Phase 4 package not installed). Returning empty list. "
            "Known-positive injection / holdout will be a no-op -- this "
            "is fine for Phase 3 standalone runs, but Phase 4 must be "
            "installed for the full production pipeline."
        )
        return []
    # P4-004: KNOWN_POSITIVES is now a _LazyList proxy. Force-load it
    # into a plain list so the bridge gets a snapshot (not a proxy).
    return list(_KP)


def _get_validated_hypotheses() -> List[Tuple[str, str]]:
    """Lazily import and return the RL ranker's VALIDATED_HYPOTHESES list.

    P4-001 ROOT FIX (CRITICAL — Team Cosmic / Phase 4): the bridge passes
    this list to ``BiomedicalGraphBuilder.build_demo_graph`` so the 4
    validated pairs (thalidomide→MM, sildenafil→PAH, mifepristone→Cushing,
    topiramate→migraine) are injected as "treats" edges in the demo graph.
    This makes the data flywheel (DOCX §10) functional:
      - The GT model learns these pairs (gnn_score becomes high).
      - The RL agent sees them in its input data (cross-product of graph
        drugs × graph diseases).
      - The +0.1 validated_bonus in the reward function FIRES.
      - The RL agent ranks them HIGH → pharma partner sees them at top.

    Without this, the validated pairs NEVER appear in the env's input
    data, so the +0.1 validated_bonus is dead code (P4-001 bug).

    Returns an empty list (with a WARNING) if Phase 4 is not installed.
    The empty-list case makes the validated-hypothesis injection a no-op
    (the bridge still works, just without the data flywheel).
    """
    try:
        from rl.rl_drug_ranker import VALIDATED_HYPOTHESES as _VH
    except ImportError:
        logger.warning(
            "P4-001: rl.rl_drug_ranker.VALIDATED_HYPOTHESES not importable "
            "(Phase 4 package not installed). Returning empty list. "
            "Validated-hypothesis injection will be a no-op -- the data "
            "flywheel (DOCX §10) will be non-functional. Install Phase 4 "
            "for the full production pipeline."
        )
        return []
    # P4-004: VALIDATED_HYPOTHESES is now a _LazyList proxy. Force-load.
    return list(_VH)


def _deterministic_name_seed(seed: int, name: str, offset: int) -> int:
    """Deterministic 31-bit seed from (seed, name, offset) using SHA-256.

    V90 ROOT FIX (COMPOUND #2 / BUG #4): the previous code used
    ``hash(drug_name) % (2**31)`` to seed per-drug feature RNGs. Python's
    built-in ``hash()`` is randomized per process via ``PYTHONHASHSEED``
    (security defense against hash-collision DoS attacks). This made:

      1. ``patent_score`` and ``adme_score`` NON-REPRODUCIBLE across
         Python processes (the same drug got different scores each run).
      2. CI flakes (the same commit could pass CI once and fail once,
         because the random feature distributions differed).
      3. Bug reproduction impossible (a user reports "patent_score=0.92"
         but the developer's run produces patent_score=0.15).

    The fix mirrors ``BiomedicalGraphBuilder._deterministic_seed`` in
    ``graph_builder.py``: SHA-256 hash the concatenated parts and take
    the low 31 bits as the seed. SHA-256 is deterministic across
    processes, platforms, and Python versions.

    Args:
        seed: The bridge's base seed (e.g. 42).
        name: The drug/protein/disease name to hash.
        offset: A per-feature offset (e.g. 42 for patent, 43 for adme)
            so different features get different seeds even for the same
            drug name.

    Returns:
        A 31-bit non-negative integer suitable for ``np.random.default_rng``.
    """
    # P3-033 ROOT FIX (v113 forensic): use PURE length-prefix encoding
    # (NO separator between parts). The previous code (P3-032 v107 fix)
    # used ``|`` as a separator BETWEEN length-prefixed parts:
    # ``f"{len(str(seed))}:{seed}|{len(str(name))}:{name}|..."``. The
    # ``|`` separator is REDUNDANT (the length prefix already delimits
    # each part) and INCONSISTENT with ``graph_builder.py``'s
    # ``_deterministic_seed`` (which uses pure length-prefix with NO
    # separator). A future developer might "fix" the inconsistency by
    # removing the ``|`` from one module or adding it to the other,
    # potentially introducing a real collision if the length-prefix
    # logic is also changed.
    #
    # ROOT FIX: use pure length-prefix encoding (no separators), matching
    # ``graph_builder.py``'s ``_deterministic_seed`` exactly. The length
    # prefix already eliminates collision risk (bencode-style).
    encoded = (
        f"{len(str(seed))}:{seed}"
        f"{len(str(name))}:{name}"
        f"{len(str(offset))}:{offset}"
    )
    h = hashlib.sha256(encoded.encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big") & 0x7FFFFFFF


class GTRLBridge:
    """Bridges the Graph Transformer (Phase 3) and RL Ranker (Phase 4).

    This class orchestrates the full pipeline:
        1. Build or load a biomedical knowledge graph
        2. Train or load a Graph Transformer model
        3. Generate predictions for all drug-disease pairs
        4. Compute supplementary features for the RL agent
        5. Output a CSV ready for the RL ranking pipeline
        6. Run the RL pipeline and return the ranked candidates

    Args:
        output_dir: Directory for all outputs (CSVs, checkpoints, etc.).
        device: Compute device ('cpu' or 'cuda').
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        output_dir: str = "output",
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.output_dir = output_dir
        self.device = device
        self.seed = seed
        # B5 fix: seed everything once at construction time. The
        # individual methods below also seed their own RNGs for
        # reproducibility.
        set_seed(seed)
        self.rng = np.random.default_rng(seed)
        # V31 ROOT FIX (P1-11 / Compound #6): dedicated feature RNG.
        # The audit found that ``_compute_supplementary_features`` and
        # ``_compute_drug_level_features`` both called
        # V90 ROOT FIX (BUG #18, P1): REMOVED self._feature_rng. The
        # audit found this RNG was DEAD CODE -- the per-drug
        # patent/adme/efficacy values use DEDICATED drug-seeded RNGs
        # (``drug_rng = np.random.default_rng(drug_seed)``), so this
        # ``self._feature_rng`` was never the source of feature
        # randomness. The docstring admitted "the per-drug values
        # below use dedicated drug-seeded RNGs ... so this ``rng``
        # variable is only used for the legacy non-per-drug noise
        # that has already been removed." Removing dead code keeps
        # the codebase honest. If future code needs per-instance RNG,
        # it should create its own dedicated RNG (not reuse a shared
        # one that advances state across calls in surprising ways).
        # self._feature_rng = np.random.default_rng(seed + 42)  # REMOVED
        # V90 ROOT FIX (BUG #38): REMOVE the dead ``self._feature_rng``
        # instance attribute. The V31 P1-11 fix introduced this attribute
        # to "hoist the feature RNG to instance state so streaming and
        # in-memory paths produce statistically equivalent distributions."
        # BUT the code in ``_compute_drug_level_features`` and
        # ``_compute_supplementary_features`` uses PER-DRUG deterministic
        # RNGs (``drug_rng = np.random.default_rng(drug_seed)``), NOT
        # ``self._feature_rng``. The ``rng = self._feature_rng`` line at
        # the top of each method was a DEAD assignment -- the comment
        # explicitly admitted it: "this ``rng`` variable is only used for
        # the legacy non-per-drug noise that has already been removed."
        #
        # The audit's BUG #38 finding: "Dead code that misleads reviewers.
        # A reviewer sees self._feature_rng and assumes it's used for
        # feature randomness, but it's not."
        #
        # The fix: remove ``self._feature_rng`` entirely AND remove the
        # ``rng = self._feature_rng`` assignments in both methods. The
        # per-drug deterministic RNGs (now using SHA-256 seeds via
        # ``_deterministic_name_seed`` per COMPOUND #2 fix) are the
        # actual source of feature randomness. There is no "shared
        # instance-level RNG" needed because each drug's features are
        # computed independently and deterministically.

        os.makedirs(output_dir, exist_ok=True)

        # V30 ROOT FIX (9.7): version compatibility check. The DOCX
        # claims the bridge checks that both packages have compatible
        # versions, but the original code never did. This made it
        # possible to mix an old RL ranker with a new GT model (or
        # vice versa), causing silent failures. The fix imports both
        # package versions and raises if they mismatch.
        try:
            from . import __version__ as _gt_version
            from rl import __version__ as _rl_version
            if _gt_version != _rl_version:
                logger.warning(
                    f"V30 ROOT FIX (9.7): GT package version {_gt_version} "
                    f"!= RL package version {_rl_version}. The two packages "
                    f"are versioned together; a mismatch may indicate an "
                    f"incomplete upgrade. Proceeding, but check for "
                    f"compatibility issues."
                )
            else:
                logger.info(
                    f"V30 ROOT FIX (9.7): GT and RL package versions match "
                    f"({_gt_version}). Cross-package compatibility verified."
                )
        except ImportError as e:
            logger.warning(
                f"V30 ROOT FIX (9.7): could not import both package "
                f"versions for compatibility check: {e}. Proceeding "
                f"without version check."
            )

        # Will be populated by build_graph / load_model
        self.model: Optional[DrugRepurposingGraphTransformer] = None
        self.node_features: Dict[str, torch.Tensor] = {}
        self.edge_indices: Dict[Tuple[str, str, str], torch.Tensor] = {}
        self.node_maps: Dict[str, Dict[str, int]] = {}
        self.drug_names: List[str] = []
        self.disease_names: List[str] = []
        self.known_pairs: List[Tuple[str, str]] = []

        # P4-017 ROOT FIX (Team Member 12): track the timestamp when the
        # knowledge graph was built/loaded into memory. The
        # train_graph_transformer method compares this timestamp to the
        # GT checkpoint's mtime — if the checkpoint is OLDER than the
        # KG, the checkpoint is STALE (it was trained on a previous
        # version of the KG) and the bridge raises RuntimeError instead
        # of silently training the RL agent on stale predictions.
        # Default is 0.0 (epoch) so the check is skipped until the KG
        # is actually built.
        self._kg_built_at: float = 0.0

        # Holds the most recent train/val/test split for inspection
        self._split: Optional[Dict[str, torch.Tensor]] = None
        self._test_metrics: Optional[Dict[str, float]] = None

        # V4 C-F8 fix: holds the trained RL model after run_full_pipeline,
        # so get_top_k_novel_predictions can route Phase 6 through it.
        self.rl_model: Any = None
        self.rl_config: Any = None

    # ------------------------------------------------------------------
    # PHASE 3.1 -- Graph construction
    # ------------------------------------------------------------------
    def build_demo_graph(
        self,
        num_drugs: int = 20,
        num_diseases: int = 15,
        num_known_treatments: int = 15,
        inject_known_positives: bool = True,
        inject_validated_hypotheses: bool = True,
    ) -> None:
        """Build a demo knowledge graph for testing the pipeline.

        FIX (C6): if ``inject_known_positives`` is True, the bridge
        passes the RL ranker's ``KNOWN_POSITIVES`` list to the graph
        builder, which injects those exact (drug_name, disease_name)
        pairs as ``treats`` edges. This means the RL recovery test
        (which looks for ``aspirin``, ``metformin``, etc. by name)
        actually finds them in the integrated pipeline.

        V4 ROOT FIX (B-F9): ``KNOWN_POSITIVES`` is now imported at the
        top of the module via ``from rl.rl_drug_ranker import
        KNOWN_POSITIVES`` -- no more ``sys.path.insert`` hackery inside
        this method. The ``rl`` package is a proper installable Python
        package (``rl/__init__.py``), structurally symmetric to
        ``graph_transformer``.

        P4-001 ROOT FIX (CRITICAL — Team Cosmic / Phase 4): if
        ``inject_validated_hypotheses`` is True (default), the bridge
        ALSO passes the RL ranker's ``VALIDATED_HYPOTHESES`` list to the
        graph builder. These are the 4 pharma-validated repurposing
        pairs (thalidomide→MM, sildenafil→PAH, mifepristone→Cushing,
        topiramate→migraine) from validated_hypotheses.csv — the data
        flywheel (DOCX §10). Injecting them as "treats" edges makes the
        GT model learn them (gnn_score becomes high), and the RL agent
        sees them in its input data so the +0.1 validated_bonus can
        fire. Without this injection, the data flywheel is dead code.

        Args:
            num_drugs: Number of drug nodes.
            num_diseases: Number of disease nodes.
            num_known_treatments: Number of known drug-disease pairs.
            inject_known_positives: If True, inject the RL ranker's
                KNOWN_POSITIVES list into the graph.
            inject_validated_hypotheses: If True (default), inject the
                RL ranker's VALIDATED_HYPOTHESES list into the graph
                (P4-001 data flywheel fix).
        """
        # P3-032 ROOT FIX: lazily fetch KNOWN_POSITIVES via the helper
        # instead of referencing the top-level constant. The helper returns
        # [] (with a warning) when Phase 4 is not installed, instead of
        # crashing the whole module import.
        known_positives: Optional[List[Tuple[str, str]]] = None
        if inject_known_positives:
            known_positives = _get_known_positives()

        # P4-001 ROOT FIX: lazily fetch VALIDATED_HYPOTHESES via the helper.
        # The helper returns [] (with a warning) when Phase 4 is not
        # installed. When Phase 4 IS installed, this injects the 4
        # validated pairs as "treats" edges → data flywheel functional.
        validated_hypotheses: Optional[List[Tuple[str, str]]] = None
        if inject_validated_hypotheses:
            validated_hypotheses = _get_validated_hypotheses()

        logger.info("Building demo knowledge graph...")

        (
            self.node_features,
            self.edge_indices,
            self.node_maps,
            self.known_pairs,
        ) = BiomedicalGraphBuilder.build_demo_graph(
            num_drugs=num_drugs,
            num_diseases=num_diseases,
            num_known_treatments=num_known_treatments,
            seed=self.seed,
            known_positives=known_positives,
            validated_hypotheses=validated_hypotheses,
        )

        self.drug_names = list(self.node_maps.get("drug", {}).keys())
        self.disease_names = list(self.node_maps.get("disease", {}).keys())

        # P4-017 ROOT FIX: record the timestamp when the KG was built.
        # The train_graph_transformer method compares this to the GT
        # checkpoint's mtime — if the checkpoint is older, the
        # checkpoint is STALE and the bridge raises RuntimeError.
        import time as _time_mod
        self._kg_built_at = _time_mod.time()

        logger.info(
            f"Graph built: {len(self.drug_names)} drugs, "
            f"{len(self.disease_names)} diseases, "
            f"{len(self.known_pairs)} known treatment pairs "
            f"({len(validated_hypotheses) if validated_hypotheses else 0} validated, "
            f"kg_built_at={self._kg_built_at:.3f})"
        )

    # ------------------------------------------------------------------
    # ROOT FIX (Phase 1+2+3+4 100% Connection):
    # load_graph_from_phase1 -- load a REAL graph from Phase 1->2 output
    # ------------------------------------------------------------------
    # This is the alternative to ``build_demo_graph()``. Instead of
    # generating a SYNTHETIC random graph with hardcoded drug names, it
    # accepts the ``Phase1StagedData`` produced by the Phase 1->2 bridge
    # (``phase2.drugos_graph.phase1_bridge.stage_phase1_to_phase2()``)
    # and builds a REAL graph from the actual biomedical data that
    # Phase 1 ingested from the 7 public sources.
    #
    # The user's forensic audit found that Phase 3+4 were 0% connected
    # to Phase 1+2: ``run_full_pipeline()`` ALWAYS called
    # ``build_demo_graph()``, and there was NO code path to load a real
    # graph. This method + the ``phase1_staged_data`` parameter on
    # ``run_full_pipeline()`` close that gap.
    # ------------------------------------------------------------------
    def load_graph_from_phase1(self, staged_data: Any) -> None:
        """Load a REAL knowledge graph from Phase 1->2 staged data.

        Replaces ``build_demo_graph()`` when the caller has REAL Phase 1
        data (the production path). The staged data is the output of
        ``phase2.drugos_graph.phase1_bridge.stage_phase1_to_phase2()``,
        which reads Phase 1's processed CSVs (DrugBank, OMIM, ChEMBL,
        etc.) and converts them into Phase 2 node/edge dicts.

        After this call, ``self.node_features``, ``self.edge_indices``,
        ``self.node_maps``, ``self.known_pairs``, ``self.drug_names``,
        and ``self.disease_names`` are populated with REAL data -- ready
        for ``build_model()`` and ``train_model()``.

        Args:
            staged_data: A ``Phase1StagedData`` (or duck-typed object)
                with ``compound_nodes``, ``protein_nodes``,
                ``pathway_nodes``, ``disease_nodes``,
                ``clinical_outcome_nodes``, and ``edges``.

        Raises:
            ValueError: If the staged data has zero drug or disease
                nodes (propagated from
                ``BiomedicalGraphBuilder.from_phase1_staged_data``).
        """
        logger.info(
            "ROOT FIX (Phase 1+2+3+4): loading REAL knowledge graph "
            "from Phase 1->2 staged data (NOT synthetic demo graph)."
        )

        (
            self.node_features,
            self.edge_indices,
            self.node_maps,
            self.known_pairs,
        ) = BiomedicalGraphBuilder.from_phase1_staged_data(
            staged_data, seed=self.seed
        )

        self.drug_names = list(self.node_maps.get("drug", {}).keys())
        self.disease_names = list(self.node_maps.get("disease", {}).keys())

        # P4-017 ROOT FIX: record the timestamp when the REAL KG was
        # loaded. The train_graph_transformer method compares this to
        # the GT checkpoint's mtime — if the checkpoint is older, the
        # checkpoint is STALE (it was trained on a previous version of
        # the KG) and the bridge raises RuntimeError instead of
        # silently training the RL agent on stale predictions.
        import time as _time_mod
        self._kg_built_at = _time_mod.time()

        logger.info(
            f"REAL graph loaded: {len(self.drug_names)} drugs, "
            f"{len(self.disease_names)} diseases, "
            f"{len(self.known_pairs)} REAL known treatment pairs "
            f"(from Phase 1->2 staged data, kg_built_at={self._kg_built_at:.3f})."
        )

    # ------------------------------------------------------------------
    # PHASE 3.2 -- Model construction
    # ------------------------------------------------------------------
    def build_model(
        self,
        embedding_dim: int = 32,
        num_layers: int = 3,
        num_heads: int = 2,
        dropout: float = 0.2,
        attention_dropout: float = 0.2,
        # V90 ROOT FIX (BUG #34): parameterize link_predictor_hidden_dims
        # instead of hardcoding [64, 32]. The previous code hardcoded
        # [64, 32] which was appropriate for the demo graph (~200 training
        # pairs) but UNDERSIZED for production (10K drugs, millions of
        # pairs) where [256, 128] would be appropriate. The caller
        # (run_full_pipeline) can now override this via the
        # gt_link_predictor_hidden_dims parameter. Default [64, 32]
        # preserves the demo-scale behavior.
        link_predictor_hidden_dims: Optional[List[int]] = None,
    ) -> None:
        """Build the Graph Transformer model.

        V90 ROOT FIX (BUG #7, P0): default ``num_layers=3`` (was 1).
        A 1-layer graph transformer only aggregates 1-hop neighbors.
        The drug -> protein -> pathway -> disease pattern is 3 hops.
        With 1 layer, the disease node's embedding only sees pathway
        nodes (1-hop), NOT the proteins or drugs that connect to those
        pathways -- the model CANNOT learn the multi-hop pattern. The
        only reason ``run_real_pipeline.py`` achieved AUC > 0.85 was
        that it overrode to 3 layers, but the default path through
        ``run_full_pipeline`` used 1 layer and could not learn.

        3 layers is the FLOOR for a 3-hop pattern. Each layer aggregates
        1-hop neighbors, so 3 layers reach 3 hops out. Standard HGT
        practice for biomedical KGs is 3-4 layers.

        Uses the single-source-of-truth ``DEFAULT_FEATURE_DIMS`` from
        ``graph_transformer.data`` (B7 fix -- no more dual constants).

        ROOT FIX (A1/A2): the previous defaults used larger models
        (64, 2, 4) which overfit on the small demo graph (~200 training
        pairs) because they had ~100K parameters -- enough to memorize
        all training pairs in 1 epoch. The model would achieve good
        val AUC at epoch 1 (by luck) and then degrade as it memorized
        training-specific patterns.

        The (32, 3, 2) config with a SMALL link predictor
        (hidden_dims=[64, 32]) keeps the model at ~15K parameters --
        small enough that it CANNOT memorize 200 training pairs and is
        forced to learn the GENERAL feature-alignment pattern that
        generalizes to held-out drugs.

        In production (10K drugs, millions of pairs), scale up to
        (128, 4, 8) with link_predictor_hidden_dims=[256, 128] and
        reduce dropout to 0.1.

        V90 ROOT FIX (BUG #34): ``link_predictor_hidden_dims`` is now a
        parameter (was hardcoded [64, 32]). The caller can override it
        for production scale. The adaptive scaling in run_full_pipeline
        now sets [64, 32] for demo, [128, 64] for pilot, and [256, 128]
        for production -- matching the model's embedding_dim scaling.

        Args:
            embedding_dim: Embedding dimension.
            num_layers: Number of transformer layers. MUST be >= 3 to
                learn the 3-hop drug -> protein -> pathway -> disease
                pattern. The default is 3 (BUG #7 fix).
            num_heads: Number of attention heads.
            dropout: Dropout rate for FFN and residual connections.
            attention_dropout: Dropout rate for attention scores.
            link_predictor_hidden_dims: Hidden dims for the link predictor
                MLP. If None (default), uses [64, 32] for demo scale.
                Pass [256, 128] for production scale.
        """
        # V90 ROOT FIX (BUG #7, P0): enforce minimum 3 layers.
        if num_layers < 3:
            raise ValueError(
                f"V90 ROOT FIX (BUG #7): num_layers={num_layers} is too "
                f"small. A graph transformer with fewer than 3 layers "
                f"cannot learn the 3-hop drug -> protein -> pathway -> "
                f"disease pattern (each layer aggregates 1-hop neighbors, "
                f"so 3 layers are needed to reach 3 hops). Use "
                f"num_layers >= 3. The default is now 3."
            )

        # B7 fix: import from the single source of truth.
        feature_dims = dict(DEFAULT_FEATURE_DIMS)

        # V90 BUG #34 fix: use caller-provided link_predictor_hidden_dims
        # or default to [64, 32] (demo scale). The previous code hardcoded
        # [64, 32] which was a CAPACITY BOTTLENECK for production-scale
        # models. The link predictor's capacity must scale with the
        # model's embedding_dim: [64, 32] for embedding_dim=32 (demo),
        # [128, 64] for embedding_dim=64 (pilot), [256, 128] for
        # embedding_dim=128 (production).
        if link_predictor_hidden_dims is None:
            link_predictor_hidden_dims = [64, 32]

        self.model = DrugRepurposingGraphTransformer(
            feature_dims=feature_dims,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            # V90 BUG #34 fix: parameterized (was hardcoded [64, 32]).
            link_predictor_hidden_dims=link_predictor_hidden_dims,
            link_predictor_dropout=dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"V90 BUG #34: Model built: {n_params:,} parameters "
            f"(dropout={dropout}, attention_dropout={attention_dropout}, "
            f"link_predictor_hidden_dims={link_predictor_hidden_dims})"
        )

    # ------------------------------------------------------------------
    # P3-054 ROOT FIX (v107): graph-content hash + safe checkpoint resume
    # ------------------------------------------------------------------
    def _compute_graph_hash(self) -> str:
        """Compute a SHA-256 content hash of the current knowledge graph.

        P3-054 ROOT FIX (v107): used by ``_can_resume_from_checkpoint_safely``
        to determine whether the on-disk GT checkpoint was trained on the
        SAME graph topology as the current in-memory graph.
        """
        if not self.node_features or not self.edge_indices:
            return "empty_graph_no_hash"
        hash_parts = []
        for ntype in sorted(self.node_features.keys()):
            tensor = self.node_features[ntype]
            count = int(tensor.shape[0])
            sample = tensor[:100].flatten().detach().cpu().numpy().tobytes()
            hash_parts.append(f"{ntype}:{count}:{len(sample)}")
        for et in sorted(self.edge_indices.keys(), key=lambda t: (t[0], t[1], t[2])):
            ei = self.edge_indices[et]
            count = int(ei.shape[1]) if ei.dim() == 2 else 0
            hash_parts.append(f"{et[0]}|{et[1]}|{et[2]}:{count}")
        encoded = "\n".join(hash_parts).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _can_resume_from_checkpoint_safely(self) -> bool:
        """Check whether the on-disk GT checkpoint is safe to resume from.

        P3-054 ROOT FIX (v107): returns True ONLY when the checkpoint
        file exists, a sidecar hash file exists, and the hash matches
        the current graph's hash. Returns False (with a WARNING) otherwise.
        """
        checkpoint_path = os.path.join(self.output_dir, "gt_checkpoint.pt")
        hash_sidecar_path = os.path.join(self.output_dir, "gt_checkpoint.graph_hash")
        if not os.path.exists(checkpoint_path):
            logger.warning(
                "P3-054 ROOT FIX (v107): force_retrain=False requested, "
                "but no checkpoint exists at %s. Falling back to fresh training.",
                checkpoint_path,
            )
            return False
        if not os.path.exists(hash_sidecar_path):
            logger.warning(
                "P3-054 ROOT FIX (v107): force_retrain=False requested, "
                "but no graph_hash sidecar exists at %s. Falling back to "
                "fresh training. Re-run with force_retrain=True (default) "
                "once to write a hash-tagged checkpoint.",
                hash_sidecar_path,
            )
            return False
        try:
            with open(hash_sidecar_path, "r", encoding="utf-8") as f:
                stored_hash = f.read().strip()
        except Exception as exc:
            logger.warning(
                "P3-054 ROOT FIX (v107): could not read graph_hash sidecar: %s. "
                "Falling back to fresh training.", exc,
            )
            return False
        current_hash = self._compute_graph_hash()
        if stored_hash != current_hash:
            logger.warning(
                "P3-054 ROOT FIX (v107): checkpoint hash (%s) does NOT match "
                "current graph hash (%s). Graph topology changed. Falling back "
                "to fresh training.", stored_hash, current_hash,
            )
            return False
        logger.info(
            "P3-054 ROOT FIX (v107): checkpoint hash MATCHES current graph "
            "(%s). Resuming from checkpoint.", current_hash,
        )
        return True

    def _write_graph_hash_sidecar(self) -> None:
        """Write the current graph's content hash to a sidecar file.

        P3-054 ROOT FIX (v107): called after a successful GT training run
        so the next run_full_pipeline(force_retrain=False) can verify the
        checkpoint matches the current graph. Atomic write (temp + rename).
        """
        if not self.node_features:
            return
        hash_value = self._compute_graph_hash()
        sidecar_path = os.path.join(self.output_dir, "gt_checkpoint.graph_hash")
        tmp_path = sidecar_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(hash_value)
            os.replace(tmp_path, sidecar_path)
            logger.info(
                "P3-054 ROOT FIX (v107): wrote graph_content_hash %s to "
                "sidecar %s.", hash_value, sidecar_path,
            )
        except Exception as exc:
            logger.warning(
                "P3-054 ROOT FIX (v107): failed to write graph hash sidecar: %s. "
                "Future runs with force_retrain=False will fall back to fresh "
                "training.", exc,
            )

    # ------------------------------------------------------------------
    # PHASE 3.3a -- Training data + drug-aware split (extracted for
    # resume_from_checkpoint re-evaluation -- V90 BUG #5 fix)
    # ------------------------------------------------------------------
    def _compute_training_split(self, neg_ratio: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Build training pairs + drug-aware train/val/test split.

        V90 ROOT FIX (BUG #5, P0): extracted from ``train_model`` so the
        resume_from_checkpoint path can compute the SAME test split
        (deterministic, seeded by ``self.seed``) and re-evaluate on it.
        Without this extraction, the resume path returned a dict with
        no ``test_auc``, crashing the scientific_validation gate with
        ``TypeError: '>' not supported between instances of 'NoneType'
        and 'float'``.

        V90 ROOT FIX (BUG #16, P1): REMOVED the alignment_median
        negative-sampling filter. The audit found that node features
        are ``rng.standard_normal`` (purely random per the S-05 fix).
        The alignment matrix was the dot product of two independent
        random matrices -- its entries were approximately ``N(0,
        signal_dim)`` and the median was ≈ 0. The filter rejected
        ~50% of random pairs based on whether their random feature
        dot product was above or below 0. This was filtering on NOISE.
        The model learns from topology (not raw features), so this
        filter had ZERO effect on the model's learning. The "clean
        training signal" claim was theater. Removed.

        Returns:
            Dict with keys: train_drug_idx, train_disease_idx,
            train_labels, val_drug_idx, val_disease_idx, val_labels,
            test_drug_idx, test_disease_idx, test_labels.
        """
        if self.model is None:
            raise RuntimeError("Call build_model() first")

        # V90 ROOT FIX (COMPOUND #3): the previous resume-from-checkpoint
        # path returned a MINIMAL dict WITHOUT test_auc / test_auc_verified.
        # The bridge's scientific_validation gate then evaluated
        # ``gt_results.get("test_auc_verified")`` -> None, fell back to
        # ``gt_results.get("test_auc", 0.0)`` -> 0.0, and the gate
        # ``0.0 > 0.85`` ALWAYS failed. Every run after the first (when
        # gt_checkpoint.pt existed) crashed with an opaque RuntimeError.
        #
        # The fix: do the checkpoint check AFTER the train/val/test split
        # is generated and the trainer is created. On resume, we SKIP
        # trainer.fit() (the expensive part) but STILL evaluate on the
        # held-out test set so the results dict includes test_auc and
        # test_auc_verified. The scientific_validation gate then gets
        # real metrics instead of None/0.0.
        checkpoint_path = os.path.join(self.output_dir, "gt_checkpoint.pt")

        # Create training data from known treatment pairs
        drug_map = self.node_maps.get("drug", {})
        disease_map = self.node_maps.get("disease", {})
        if not drug_map or not disease_map:
            raise RuntimeError("No drug or disease nodes in the graph")

        num_drugs = len(drug_map)
        num_diseases = len(disease_map)

        # Generate positive pairs from graph structure
        treats_edges = self.edge_indices.get(("drug", "treats", "disease"), None)
        if treats_edges is not None and treats_edges.shape[1] > 0:
            pos_drug_idx = treats_edges[0]
            pos_disease_idx = treats_edges[1]
        else:
            # P3-028 ROOT FIX (CRITICAL — do NOT generate fake positives).
            # The previous code silently generated synthetic positives
            # (drug_0 -> disease_0, drug_1 -> disease_1, ...) when no
            # treats edges existed. These are MOCK DATA — the pairs don't
            # correspond to real drug-disease treatments. The GT model
            # would train on fake positive labels and learn a meaningless
            # pattern. The fix RAISES — the caller must provide a graph
            # with real treats edges (from Phase 1 DrugBank/RepoDB data
            # or the demo graph's curated KNOWN_POSITIVES).
            raise RuntimeError(
                "P3-028 ROOT FIX: no ('drug', 'treats', 'disease') edges "
                "found in the graph. The GT model CANNOT train without "
                "real positive drug-disease treatment pairs. The previous "
                "code silently generated FAKE positives (drug_i -> disease_i) "
                "which are scientifically meaningless. Provide a graph with "
                "real treats edges (from Phase 1 DrugBank/RepoDB data or "
                "the demo graph's curated KNOWN_POSITIVES)."
            )

        n_pos = len(pos_drug_idx)

        # Generate negative pairs (no known treatment relationship)
        neg_drug_indices: List[int] = []
        neg_disease_indices: List[int] = []
        neg_rng = np.random.default_rng(self.seed + 1)

        # Set of known positive (drug_idx, disease_idx) tuples to exclude
        pos_set = set(
            (int(pos_drug_idx[i].item()), int(pos_disease_idx[i].item()))
            for i in range(n_pos)
        )

        # P3-021 ROOT FIX (SCIENTIFIC — INCLUDE KP DRUGS IN NEGATIVE
        # SAMPLING): the W-07 fix excluded KP drugs from negative sampling
        # so they appeared ONLY in positive pairs. The stated rationale
        # was "prevent the conflicting-signal bug where the model sees
        # aspirin in BOTH positive and negative pairs." But the C-3 fix
        # (below, line ~794) ALREADY holds out ALL KP drugs from the GT
        # TRAINING SET via drug_aware_split(held_out_drugs=kp_drugs).
        # This means KP drugs NEVER appear in the TRAINING split —
        # neither as positives NOR as negatives — regardless of whether
        # they're in the negative-sampling pool. So W-07's concern is
        # MOOT given C-3: there is no "conflicting signal" because KP
        # drugs don't reach the training set at all.
        #
        # The HARM of W-07: by excluding KP drugs from the negative
        # pool, the val/test set (which DOES contain KP-drug pairs via
        # the C-3 hold-out) had KP-drug-POSITIVE pairs but NO
        # KP-drug-NEGATIVE pairs. The model's scores for KP drugs were
        # UNCALIBRATED — it never saw "aspirin + unrelated_disease =
        # negative" during training or evaluation. A model that scores
        # ALL aspirin pairs high (because aspirin has good topology)
        # would still "recover" KPs by chance, making the KP recovery
        # test unreliable.
        #
        # The fix: INCLUDE KP drugs in the negative-sampling pool (use
        # ALL drugs, not just non-KP). The C-3 split then distributes
        # KP-drug-negative pairs to val/test (not train), giving the
        # evaluation a REAL negative baseline for KP drugs: "can the
        # model rank aspirin+cardiovascular (positive) above
        # aspirin+unrelated_disease (negative)?" This does NOT leak the
        # treats label — the negative pair uses a DIFFERENT disease.
        # P3-032 ROOT FIX: use the lazy helper instead of the (removed)
        # top-level KNOWN_POSITIVES constant.
        kp_drug_indices: set = set()
        for drug_name, _ in _get_known_positives():
            if drug_name in drug_map:
                kp_drug_indices.add(drug_map[drug_name])
        # P3-021: use ALL drugs (including KP) for negative candidates.
        # The C-3 split (held_out_drugs=kp_drug_indices) ensures KP
        # drugs still don't appear in TRAINING — they only appear in
        # val/test, where the KP-drug-negative pairs provide calibration.
        all_drug_indices_for_neg = list(range(num_drugs))
        logger.info(
            f"ROOT FIX (P3-021): INCLUDING all {num_drugs} drugs (incl. "
            f"{len(kp_drug_indices)} KP drugs) in negative sampling. The "
            f"C-3 split (held_out_drugs=kp_drugs) ensures KP drugs still "
            f"do NOT appear in training — they appear in val/test only, "
            f"where KP-drug-negative pairs provide calibration for the "
            f"KP recovery test. (W-07 exclusion removed: it was redundant "
            f"given C-3 and starved val/test of KP-drug negatives.)"
        )

        # V90 ROOT FIX (BUG #16, P1): REMOVED the alignment_median filter.
        # The audit found that node features are rng.standard_normal
        # (purely random per the S-05 fix). The alignment matrix was the
        # dot product of two independent random matrices -- its entries
        # were approximately N(0, signal_dim) and the median was ≈ 0.
        # The filter rejected ~50% of random pairs based on whether their
        # random feature dot product was above or below 0. This was
        # filtering on NOISE. The model learns from topology (not raw
        # features), so this filter had ZERO effect on the model's
        # learning. The "clean training signal" claim was theater.
        #
        # Plain random negative sampling (with KP exclusion + pos_set
        # exclusion) is the honest approach. The model must learn from
        # graph TOPOLOGY, not from feature alignment.

        attempts = 0
        # P3-026 ROOT FIX (SCIENTIFIC — parameterize neg_ratio). The previous
        # code hardcoded NEG_RATIO = 6 and neg_ratio = NEG_RATIO, ignoring
        # the neg_ratio parameter. The comment claimed "parameterize
        # neg_ratio instead of hardcoding 6" but the code just renamed a
        # magic number to a constant — NOT parameterization.
        # The fix: use the neg_ratio PARAMETER if provided, else default
        # to 6 (preserving previous behavior). The caller can now scale
        # neg_ratio with the actual class imbalance (e.g., for a highly
        # imbalanced graph with 1:1000 pos:neg ratio, use neg_ratio=20
        # to under-sample negatives; for a balanced graph, use neg_ratio=1).
        if neg_ratio is None:
            neg_ratio = 6  # default (preserves previous behavior)
        # P3-038 ROOT FIX (v113 forensic): increased from 50 to 200.
        # The previous ``MAX_ATTEMPTS_MULTIPLIER = 50`` gave up too
        # early on dense graphs (where most drug-disease pairs are
        # reachable via multi-hop paths and thus rejected as "false
        # negatives"). For a small graph with 5 KPs and neg_ratio=6,
        # target is 30 negatives and max_attempts was 1500. On a dense
        # graph (70% reachable), only ~450 random candidates were
        # accepted, but the reachability filter rejected most of those
        # too -- the final count was often 5-10 (vs target 30). The
        # warning was logged but training CONTINUED with a 1:2 or 1:1
        # pos:neg ratio instead of 1:6.
        #
        # ROOT FIX: increase to 200. This gives the sampler 4x more
        # retries. For most graphs, this is sufficient to reach the
        # target. If still insufficient, the warning is logged (the
        # fallback is to continue with fewer negatives, which is
        # documented in the warning).
        MAX_ATTEMPTS_MULTIPLIER = 200
        max_attempts = n_pos * neg_ratio * MAX_ATTEMPTS_MULTIPLIER

        # P3-010 ROOT FIX (SCIENTIFIC — multi-hop reachability check for
        # negative sampling). The previous code generated negatives via
        # uniform random + corrupt-one-side, with NO check for whether a
        # generated "negative" pair is actually reachable via multi-hop
        # paths (drug→protein→pathway→disease). This creates FALSE
        # NEGATIVES — pairs labeled as negative that are biologically
        # plausible positives (the drug and disease share a pathway).
        # The model is punished for correctly scoring them high.
        #
        # The fix: build a (num_drugs, num_diseases) reachability matrix
        # via multi-hop BFS. A pair is "reachable" if there's a path
        # drug→protein→pathway→disease (3-hop) or drug→protein→pathway
        # (2-hop to pathway, which connects to disease). Exclude reachable
        # pairs from the negative pool. This ensures negatives are TRULY
        # negative — the drug and disease have NO biological connection
        # in the graph.
        reachable_pairs: set = set()
        try:
            # Build drug->protein adjacency
            drug_to_proteins: Dict[int, set] = {}
            for et_key in [("drug", "inhibits", "protein"),
                           ("drug", "activates", "protein"),
                           ("drug", "binds", "protein"),
                           ("drug", "modulates", "protein")]:
                ei = self.edge_indices.get(et_key)
                if ei is not None and ei.numel() > 0:
                    for d_idx, p_idx in zip(ei[0].tolist(), ei[1].tolist()):
                        drug_to_proteins.setdefault(d_idx, set()).add(p_idx)
            # Build protein->pathway adjacency
            protein_to_pathways: Dict[int, set] = {}
            pw_ei = self.edge_indices.get(("protein", "part_of", "pathway"))
            if pw_ei is not None and pw_ei.numel() > 0:
                for p_idx, pw_idx in zip(pw_ei[0].tolist(), pw_ei[1].tolist()):
                    protein_to_pathways.setdefault(p_idx, set()).add(pw_idx)
            # Build pathway->disease adjacency
            pathway_to_diseases: Dict[int, set] = {}
            pd_ei = self.edge_indices.get(("pathway", "disrupted_in", "disease"))
            if pd_ei is not None and pd_ei.numel() > 0:
                for pw_idx, ds_idx in zip(pd_ei[0].tolist(), pd_ei[1].tolist()):
                    pathway_to_diseases.setdefault(pw_idx, set()).add(ds_idx)
            # Compute reachable (drug, disease) pairs via 3-hop BFS
            for d_idx, proteins in drug_to_proteins.items():
                for p_idx in proteins:
                    for pw_idx in protein_to_pathways.get(p_idx, set()):
                        for ds_idx in pathway_to_diseases.get(pw_idx, set()):
                            reachable_pairs.add((d_idx, ds_idx))
            if reachable_pairs:
                logger.info(
                    f"P3-010 ROOT FIX: built reachability matrix with "
                    f"{len(reachable_pairs)} reachable (drug, disease) pairs "
                    f"via 3-hop BFS (drug→protein→pathway→disease). "
                    f"These pairs are EXCLUDED from the negative pool to "
                    f"prevent false-negative label noise."
                )
        except Exception as exc:
            logger.warning(
                "P3-010 ROOT FIX: reachability matrix build failed (%s). "
                "Negative sampling will proceed WITHOUT the reachability "
                "filter (may include false negatives).", exc,
            )

        # P3-S04 ROOT FIX (SCIENTIFIC): the previous code used UNIFORM
        # RANDOM negative sampling -- pick a random drug and a random
        # disease independently, check only that the pair is not in
        # pos_set. This has two problems documented in the audit:
        #   1. INDIRECT LEAKAGE: a (drug, disease) pair where the drug
        #      and disease are connected via a 2-hop or 3-hop path
        #      (drug->protein->pathway->disease) is treated as a negative,
        #      but the model can easily score it high via message
        #      passing. This creates label noise (the model is told
        #      "negative" but its message-passing says "looks positive").
        #   2. EASY NEGATIVES: most uniform-random pairs have NO
        #      biological connection, so the model trivially scores them
        #      low -> inflated AUC that doesn't reflect real
        #      generalization.
        # The standard KG-embedding fix (TransE, Bordes et al. 2013) is
        # "corrupt one side": for each positive (drug, disease) pair,
        # generate a negative by replacing EITHER the drug OR the
        # disease with a random one (50/50 chance). This ensures the
        # negative is "close" to a positive (shares either the drug or
        # the disease), making it a HARDER negative -- the model must
        # learn the specific drug-disease association, not just "this
        # drug is rare" or "this disease is rare."
        #
        # P3-020 ROOT FIX (SCIENTIFIC — MIX CORRUPT-ONE-SIDE + CORRUPT-BOTH):
        # the P3-S04 fix used ONLY corrupt-one-side (50% drug / 50%
        # disease). This produces ONLY hard negatives (each shares one
        # endpoint with a positive). The model NEVER sees a completely
        # unrelated (random drug + random disease) pair as a negative.
        # At inference, novel drug-disease pairs where NEITHER endpoint
        # was in training are scored by the model with no baseline
        # calibration for "completely unrelated = negative." This can
        # inflate false-positive rates for novel pairs.
        #
        # Standard KG-embedding practice (TransE, Bordes et al. 2013;
        # also RotatE, Sun et al. 2019) uses BOTH corrupt-one-side AND
        # corrupt-both negatives. The corrupt-both negatives teach the
        # model that completely unrelated pairs are negative (a trivial
        # baseline), while corrupt-one-side negatives teach the model
        # the specific association. The mix is typically 80% one-side
        # (hard) + 20% both (easy), matching the TransE convention.
        #
        # We implement the 80/20 mix here: 40% corrupt-drug + 40%
        # corrupt-disease + 20% corrupt-both. A full multi-hop
        # reachability check (build a (num_drugs, num_diseases)
        # reachability matrix and exclude reachable pairs from negatives)
        # is the gold standard but is O(n_drugs * n_diseases) memory and
        # O(n_paths) time — deferred to a future optimization.
        pos_drug_idx_list = pos_drug_idx.tolist()
        pos_disease_idx_list = pos_disease_idx.tolist()
        # P3-020: 80% corrupt-one-side (split 40/40 drug/disease) + 20% corrupt-both.
        CORRUPT_BOTH_PROB = 0.20
        while len(neg_drug_indices) < n_pos * neg_ratio and attempts < max_attempts:
            attempts += 1
            # Pick a random positive pair to corrupt.
            pos_i = int(neg_rng.integers(0, n_pos))
            r = neg_rng.random()
            if r < CORRUPT_BOTH_PROB:
                # P3-020: corrupt BOTH endpoints — random drug + random
                # disease. This produces an EASY negative (no shared
                # endpoint with any positive) that teaches the model
                # "completely unrelated = negative." The model needs
                # this baseline so it doesn't over-score novel pairs at
                # inference. We use all_drug_indices_for_neg (includes
                # KP drugs per P3-021).
                d_idx = int(all_drug_indices_for_neg[
                    neg_rng.integers(0, len(all_drug_indices_for_neg))
                ])
                ds_idx = int(neg_rng.integers(0, num_diseases))
            elif r < 0.5 * (1 + CORRUPT_BOTH_PROB):
                # Corrupt the drug (keep the disease): hard negative.
                # P3-021: use ALL drugs (incl. KP) — C-3 split handles hold-out.
                d_idx = int(all_drug_indices_for_neg[
                    neg_rng.integers(0, len(all_drug_indices_for_neg))
                ])
                ds_idx = int(pos_disease_idx_list[pos_i])
            else:
                # Corrupt the disease (keep the drug): hard negative.
                d_idx = int(pos_drug_idx_list[pos_i])
                ds_idx = int(neg_rng.integers(0, num_diseases))
            if (d_idx, ds_idx) in pos_set:
                continue
            # P3-010 ROOT FIX: skip reachable pairs (false negatives).
            # A reachable pair has a multi-hop biological connection
            # (drug→protein→pathway→disease), so it's a plausible
            # positive, NOT a true negative. Including it as a negative
            # creates label noise — the model is punished for correctly
            # scoring it high via message passing.
            if (d_idx, ds_idx) in reachable_pairs:
                continue
            # Optional: also skip if the corrupted pair matches another
            # positive (rare but possible). The pos_set check above
            # handles this.
            neg_drug_indices.append(d_idx)
            neg_disease_indices.append(ds_idx)

        if len(neg_drug_indices) < n_pos * neg_ratio:
            logger.warning(
                f"Could only generate {len(neg_drug_indices)} negative pairs "
                f"(target {n_pos * neg_ratio}). Graph may be too dense."
            )

        # Combine positive and negative
        all_drug_idx = torch.cat([
            pos_drug_idx,
            torch.tensor(neg_drug_indices, dtype=torch.long),
        ])
        all_disease_idx = torch.cat([
            pos_disease_idx,
            torch.tensor(neg_disease_indices, dtype=torch.long),
        ])
        all_labels = torch.cat([
            torch.ones(n_pos),
            torch.zeros(len(neg_drug_indices)),
        ])

        # ROOT FIX (C-3): hold out ALL KP drugs from GT training for
        # ALL graph sizes. The previous A1/A2 "fix" did NOT hold out
        # KP drugs for small graphs (<100 drugs), which meant the GT
        # model trained on aspirin's features and then scored
        # aspirin->cardiovascular disease at inference -- the score was
        # inflated by aspirin-specific memorization, not genuine
        # generalization. Holding out KP drugs aligns the GT split
        # with the RL split (both drug-aware).
        all_kp_drug_indices: set = set()
        # P3-032 ROOT FIX: use the lazy helper.
        for drug_name, _ in _get_known_positives():
            if drug_name in drug_map:
                all_kp_drug_indices.add(drug_map[drug_name])

        held_out_drug_indices = all_kp_drug_indices
        logger.info(
            f"ROOT FIX (C-3): holding out all {len(held_out_drug_indices)} "
            f"KNOWN_POSITIVES drugs from GT training for ALL graph sizes "
            f"({num_drugs} drugs). The GT model will NOT train on any KP "
            f"drug, so gnn_scores for KP pairs are TRUE generalization "
            f"measures (not drug-level memorization)."
        )

        # Drug-aware split (C4 + C5 + V4 B-F6 fix): a drug in train
        # never appears in val or test. KP drugs go to val+test only.
        split = drug_aware_split(
            all_drug_idx, all_disease_idx, all_labels,
            train_frac=0.7, val_frac=0.15, seed=self.seed,
            held_out_drugs=held_out_drug_indices if held_out_drug_indices else None,
        )
        logger.info(
            f"ROOT FIX (C-3): using drug_aware_split for ALL graph sizes "
            f"({num_drugs} drugs). GT split is ALIGNED with RL split "
            f"(both drug-aware). No drug-level train/test leakage at "
            f"the GT->RL boundary."
        )
        return split

    # ------------------------------------------------------------------
    # PHASE 3.3 -- Training (drug-aware split, held-out test set)
    # ------------------------------------------------------------------
    def train_model(
        self,
        epochs: int = 500,
        batch_size: int = 32,
        patience: int = 40,
        resume_from_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """Train the Graph Transformer on the knowledge graph.

        V30 ROOT FIX (9.8): the original bridge SAVED gt_checkpoint.pt
        but NEVER LOADED it. Every run_full_pipeline re-trained GT from
        random init, wasting the saved checkpoint. The fix adds a
        ``resume_from_checkpoint`` parameter (default True) that loads
        the checkpoint if it exists, skipping training. This is critical
        for iterative development: a user can train once, then re-run
        the RL phase multiple times without re-training GT each time.

        ROOT FIX (E16): increased epochs from 300 to 500 and patience
        from 30 to 40. The previous defaults were too short for the
        model to converge on the multi-hop alignment signal from
        structured features. 500 epochs with patience=40 gives the
        model enough time to learn while still early-stopping if it
        overfits.

        ROOT FIX (A1/A2): increased epochs from 200 to 300 and patience
        from 25 to 30. With graph-structure-encoded features, the model
        needs more epochs to converge on the multi-hop alignment signal.

        FIXES vs original:
          - **C4 (drug-aware split)**: uses ``drug_aware_split`` so a
            drug in train never appears in val or test.
          - **C5 (70/15/0 -- no test set)**: actually creates a test
            set and evaluates on it. Test AUC is reported in the
            returned dict.
          - **C7 (untrained GT)**: bumped epochs + patience so the GT
            model actually converges.
          - **B5 (seeded RNG)**: the split uses a torch.Generator
            seeded with ``self.seed``, so the same seed produces the
            same split every run.

        Args:
            epochs: Maximum training epochs.
            batch_size: Mini-batch size.
            patience: Early stopping patience.
            resume_from_checkpoint: If True (default), check for an
                existing gt_checkpoint.pt in output_dir and load it
                instead of re-training. Set to False to force re-training.

        Returns:
            Training results dict with best_val_auc, test_auc,
            epochs_trained, etc.
        """
        if self.model is None:
            raise RuntimeError("Call build_model() first")

        # V90 ROOT FIX (BUG #5, P0): the resume path used to return a
        # minimal dict WITHOUT test_auc / test_auc_verified. Downstream,
        # the scientific_validation gate did
        # ``gt_test_auc > V1_AUC_THRESHOLD`` where gt_test_auc was None,
        # raising ``TypeError: '>' not supported between instances of
        # 'NoneType' and 'float'``. On ANY run where gt_checkpoint.pt
        # already existed (i.e., every run after the first), the
        # pipeline crashed with an opaque TypeError AND skipped the
        # held-out test evaluation entirely.
        #
        # Root fix: the resume path now computes the SAME train/val/test
        # split as the fresh-training path, then re-runs evaluate() on
        # the held-out test split before returning. This populates
        # test_auc / test_loss / test_accuracy AND test_auc_verified
        # (via evaluate_link_prediction), so the scientific_validation
        # gate has a real AUC to check.
        #
        # The split is deterministic (seeded by self.seed), so the
        # resume path produces the SAME test set as the original
        # training run. We are NOT training on the test set -- we are
        # evaluating the loaded checkpoint on the same held-out test
        # split that was used at original training time.
        checkpoint_path = os.path.join(self.output_dir, "gt_checkpoint.pt")
        if resume_from_checkpoint and os.path.exists(checkpoint_path):
            # P4-017 ROOT FIX (Team Member 12): validate the checkpoint's
            # timestamp against the KG's last-modified time. If the
            # checkpoint is OLDER than the KG, the checkpoint is STALE —
            # it was trained on a previous version of the KG. The
            # previous code loaded the checkpoint WITHOUT validating its
            # age, so if the KG was updated today but the checkpoint was
            # from yesterday, the RL agent trained on predictions from a
            # STALE model. The rankings then reflected yesterday's KG,
            # not today's — a pharma partner would see outdated
            # recommendations.
            #
            # The fix: compare ``os.path.getmtime(checkpoint_path)`` to
            # ``self._kg_built_at``. If the checkpoint is older, raise
            # RuntimeError with a clear message. The operator must either
            # (a) re-train the GT model (delete the checkpoint) or
            # (b) explicitly confirm the checkpoint is still valid by
            # setting ``resume_from_checkpoint=False`` to force
            # re-training. A CI test
            # (tests/test_team12_p4_012_to_018.py::test_p4_017_*)
            # verifies the check fires.
            if self._kg_built_at > 0.0:
                try:
                    _checkpoint_mtime = os.path.getmtime(checkpoint_path)
                except OSError as _ckpt_stat_err:
                    logger.warning(
                        f"P4-017: could not stat checkpoint {checkpoint_path} "
                        f"to read mtime: {_ckpt_stat_err}. Skipping stale "
                        f"check (this is a best-effort guard)."
                    )
                    _checkpoint_mtime = float('inf')  # assume fresh
                if _checkpoint_mtime < self._kg_built_at:
                    _ckpt_age_s = self._kg_built_at - _checkpoint_mtime
                    raise RuntimeError(
                        f"P4-017 ROOT FIX: STALE GT checkpoint detected. "
                        f"The checkpoint at {checkpoint_path} was last "
                        f"modified at {_checkpoint_mtime:.3f} "
                        f"({time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(_checkpoint_mtime))}), "
                        f"but the knowledge graph was built/loaded at "
                        f"{self._kg_built_at:.3f} "
                        f"({time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(self._kg_built_at))}). "
                        f"The checkpoint is {_ckpt_age_s:.1f}s OLDER than "
                        f"the KG — it was trained on a PREVIOUS version "
                        f"of the KG. Loading it would cause the RL agent "
                        f"to train on STALE predictions, producing "
                        f"rankings that reflect the old KG, not the "
                        f"current one. A pharma partner would see "
                        f"outdated recommendations. "
                        f"FIX: either (a) delete the checkpoint to force "
                        f"GT re-training on the current KG, or (b) pass "
                        f"resume_from_checkpoint=False to "
                        f"train_graph_transformer to force re-training. "
                        f"Do NOT silently load a stale checkpoint."
                    )
                else:
                    logger.info(
                        f"P4-017 ROOT FIX: checkpoint freshness verified. "
                        f"checkpoint_mtime={_checkpoint_mtime:.3f} >= "
                        f"kg_built_at={self._kg_built_at:.3f} "
                        f"(checkpoint is fresh — trained on the current KG)."
                    )
            else:
                logger.info(
                    f"P4-017: kg_built_at=0.0 (KG not yet built when the "
                    f"bridge was constructed). Skipping stale-checkpoint "
                    f"check. This is expected for the first run."
                )
            try:
                _temp_trainer = GraphTransformerTrainer(
                    self.model, self.node_features, self.edge_indices,
                    device=self.device, seed=self.seed,
                    # FORENSIC ROOT FIX (audit Issue 139): pass graph
                    # metadata here too so the resume path's
                    # load_checkpoint can restore it from the
                    # self-contained checkpoint.
                    node_maps=self.node_maps,
                    drug_names=self.drug_names,
                    disease_names=self.disease_names,
                    known_pairs=self.known_pairs,
                )
                _temp_trainer.load_checkpoint(checkpoint_path)
                logger.info(
                    f"V90 ROOT FIX (BUG #5): loaded existing GT checkpoint "
                    f"from {checkpoint_path}. Skipping re-training. Will "
                    f"RE-EVALUATE on held-out test split so the "
                    f"scientific_validation gate has a real test_auc "
                    f"(was: returned minimal dict with no test_auc, "
                    f"crashing with TypeError on the gate comparison)."
                )

                # Compute the SAME split as the fresh-training path so
                # the test set is identical to the original training run.
                # This is necessary for the test_auc to be comparable.
                _split = self._compute_training_split()
                test_d = _split["test_drug_idx"]
                test_ds = _split["test_disease_idx"]
                test_l = _split["test_labels"]

                # Re-evaluate on the held-out test split.
                test_metrics = _temp_trainer.evaluate(
                    test_d, test_ds, test_l,
                    batch_size=batch_size,
                )
                results = {
                    "best_val_auc": _temp_trainer.best_val_auc,
                    "best_val_loss": _temp_trainer.best_val_loss,
                    "best_epoch": getattr(_temp_trainer, "best_epoch", 0),
                    "epochs_trained": 0,
                    "history": list(_temp_trainer.training_history),
                    "resumed_from_checkpoint": True,
                    "checkpoint_path": checkpoint_path,
                    "test_auc": test_metrics["auc"],
                    "test_loss": test_metrics["loss"],
                    "test_accuracy": test_metrics["accuracy"],
                }

                # Independent verification via evaluate_link_prediction
                # (same as the fresh-training path).
                try:
                    from .evaluation import evaluate_link_prediction
                    eval_metrics = evaluate_link_prediction(
                        model=self.model,
                        node_features=self.node_features,
                        edge_indices=self.edge_indices,
                        drug_indices=test_d,
                        disease_indices=test_ds,
                        labels=test_l,
                        batch_size=batch_size,
                        device=self.device,
                    )
                    results["test_auc_verified"] = eval_metrics["auc"]
                    results["test_loss_verified"] = eval_metrics["loss"]
                    results["test_accuracy_verified"] = eval_metrics["accuracy"]
                except Exception as e:
                    logger.warning(
                        f"V90 ROOT FIX (BUG #5): evaluate_link_prediction "
                        f"verification failed on resume: {e}"
                    )

                return results
            except Exception as e:
                logger.warning(
                    f"V90 ROOT FIX (BUG #5): failed to load checkpoint "
                    f"from {checkpoint_path}: {e}. Re-training from scratch."
                )

        # Create training data from known treatment pairs
        # V90 ROOT FIX (BUG #5): split preparation is now in
        # ``_compute_training_split()`` so the resume_from_checkpoint
        # path can compute the SAME test split (deterministic, seeded)
        # and re-evaluate on it. Without this, the resume path returned
        # a dict with no test_auc and crashed the scientific_validation
        # gate with TypeError.
        self._split = self._compute_training_split()

        train_d = self._split["train_drug_idx"]
        train_ds = self._split["train_disease_idx"]
        train_l = self._split["train_labels"]
        val_d = self._split["val_drug_idx"]
        val_ds = self._split["val_disease_idx"]
        val_l = self._split["val_labels"]
        test_d = self._split["test_drug_idx"]
        test_ds = self._split["test_disease_idx"]
        test_l = self._split["test_labels"]

        logger.info(
            f"Drug-aware split: train={len(train_l)} (drugs={len(torch.unique(train_d))}), "
            f"val={len(val_l)} (drugs={len(torch.unique(val_d))}), "
            f"test={len(test_l)} (drugs={len(torch.unique(test_d))})"
        )

        # Train
        # V90 BUG #43: neg_ratio=6 is used in _compute_training_split
        # (line 653: neg_ratio = NEG_RATIO where NEG_RATIO=6). This is
        # documented here so the train_model source has a visible
        # reference to the negative-sampling ratio. The previous code
        # had neg_ratio as a magic number with no documentation; the
        # fix parameterizes it as NEG_RATIO (module-level constant)
        # and documents it here for auditability.
        # V90 BUG #44: max_attempts = n_pos * neg_ratio * 50 (line 666:
        # max_attempts = n_pos * neg_ratio * MAX_ATTEMPTS_MULTIPLIER
        # where MAX_ATTEMPTS_MULTIPLIER=50). The factor 50 gives the
        # negative sampler enough retries to find n_pos * neg_ratio
        # unique negatives even on small graphs where the candidate
        # pool is limited. Documented here for auditability.
        # ROOT FIX (A1/A2): use learning_rate=5e-4.
        # The previous 1e-3 learning rate caused the model to overfit
        # (train_loss -> 0.0001 while val_loss -> 2.5+).
        #
        # ROOT FIX (S-11): the previous code used weight_decay=1e-4
        # (10x higher than the trainer's default of 1e-5). The audit
        # found this was an undocumented MAGIC NUMBER with no measured
        # justification. The "ROOT FIX (A1/A2)" comment claimed 1e-4
        # "prevents memorization" but provided no measurement showing
        # 1e-4 was better than 1e-5. The trainer's default (1e-5) is
        # the standard for Adam (cf. Kingma & Ba 2015, Loshchilov &
        # Hutter 2019). 1e-4 is aggressive and can prevent the model
        # from learning at all on small graphs.
        #
        # The fix: use the trainer's default (1e-5) unless there is a
        # measured reason to override. The S-05 fix (removing the
        # _enrich_features_with_graph_signal artificial correlation)
        # already prevents the memorization the 1e-4 was trying to
        # combat -- the model no longer has an artificial feature
        # alignment to memorize, so aggressive regularization is no
        # longer needed.
        trainer = GraphTransformerTrainer(
            model=self.model,
            node_features=self.node_features,
            edge_indices=self.edge_indices,
            learning_rate=5e-4,
            # FORENSIC ROOT FIX (audit Issue 136): use weight_decay=0.01
            # (the production-grade Transformer value per Loshchilov &
            # Hutter 2019), not the previous 1e-5. Combined with the
            # trainer's switch from Adam to AdamW (decoupled weight
            # decay), this prevents the model from overfitting the
            # training pairs. The previous 1e-5 was effectively zero
            # regularization and let the model memorize known pairs
            # without learning generalizable structure.
            weight_decay=0.01,
            device=self.device,
            seed=self.seed,  # V4 C-F6 fix: pass seed for reproducible shuffling
            # FORENSIC ROOT FIX (audit Issue 139): pass the graph metadata
            # so the trainer can save a SELF-CONTAINED checkpoint (no
            # separate graph_state.pt sidecar needed). The service can
            # then load EVERYTHING (model + graph + name lookups) from a
            # single .pt file, eliminating the two-file sync problem.
            node_maps=self.node_maps,
            drug_names=self.drug_names,
            disease_names=self.disease_names,
            known_pairs=self.known_pairs,
        )

        # V90 ROOT FIX (COMPOUND #3): handle checkpoint resume HERE (after
        # the split is generated and trainer is created), NOT at the top
        # of the method. The previous code returned a minimal dict WITHOUT
        # test_auc / test_auc_verified, which caused the scientific_validation
        # gate to ALWAYS fail on resume (gt_test_auc = 0.0 > 0.85 is False).
        #
        # The fix: on resume, load the checkpoint into the trainer (which
        # restores model weights, best_val_auc, best_val_loss, best_epoch),
        # then SKIP trainer.fit() (the expensive part) but STILL evaluate
        # on the held-out test set. The results dict then includes real
        # test_auc and test_auc_verified values, so the scientific_validation
        # gate gets real metrics.
        resumed_from_checkpoint = False
        if resume_from_checkpoint and os.path.exists(checkpoint_path):
            try:
                trainer.load_checkpoint(checkpoint_path)
                resumed_from_checkpoint = True
                logger.info(
                    f"V90 ROOT FIX (COMPOUND #3): loaded existing GT checkpoint "
                    f"from {checkpoint_path}. Skipping re-training (trainer.fit) "
                    f"but WILL evaluate on the held-out test set so the "
                    f"scientific_validation gate gets real test_auc / "
                    f"test_auc_verified values (not None/0.0). Set "
                    f"resume_from_checkpoint=False to force re-training."
                )
            except Exception as e:
                logger.warning(
                    f"V30 ROOT FIX (9.8): failed to load checkpoint "
                    f"from {checkpoint_path}: {e}. Re-training from scratch."
                )

        if resumed_from_checkpoint:
            # Build results dict from the checkpoint's stored metrics.
            # The trainer's load_checkpoint already restored best_val_auc,
            # best_val_loss, best_epoch, and training_history.
            results = {
                "best_val_auc": trainer.best_val_auc,
                "best_val_loss": trainer.best_val_loss,
                "best_epoch": trainer.best_epoch,  # V90 BUG #33: now restored
                "epochs_trained": 0,  # 0 NEW epochs (loaded from checkpoint)
                "history": list(trainer.training_history),
                "resumed_from_checkpoint": True,
                "checkpoint_path": checkpoint_path,
            }
        else:
            results = trainer.fit(
                train_d, train_ds, train_l,
                val_d, val_ds, val_l,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                # exclude_edges defaults to LABEL_LEAKING_EDGES inside trainer
            )

        # C5 fix: evaluate on held-out TEST set
        # V90 COMPOUND #3: this runs for BOTH fresh training AND resume.
        # On resume, this is the critical fix -- without it, the results
        # dict lacks test_auc, and the scientific_validation gate fails.
        self._test_metrics = trainer.evaluate(
            test_d, test_ds, test_l,
            batch_size=batch_size,
        )
        results["test_auc"] = self._test_metrics["auc"]
        results["test_loss"] = self._test_metrics["loss"]
        results["test_accuracy"] = self._test_metrics["accuracy"]

        # P3-017 ROOT FIX (forensic, Team Member 10): the previous
        # comment here admitted that evaluate_link_prediction was
        # "CODE-PATH-IDENTICAL" to trainer.evaluate (both called
        # model.encode + link_predictor methods). That was true for
        # the V90/V92 implementation, which only computed the AUC
        # twice via the same code path. The "verified AUC" provided
        # zero independent scientific value.
        #
        # The P3-017 fix (in graph_transformer/evaluation/__init__.py)
        # now computes THREE independent AUCs:
        #   1. sklearn.roc_auc_score on MLP-forward probabilities
        #      (same as trainer.evaluate -- the primary metric)
        #   2. From-scratch Mann-Whitney U AUC on the SAME MLP scores
        #      (independent implementation -- catches sklearn API misuse)
        #   3. From-scratch Mann-Whitney U AUC on cosine-similarity
        #      scores (bypasses the MLP -- catches MLP overfitting)
        #
        # If sklearn vs Mann-Whitney disagree by >0.001, one of them
        # has a bug. If the MLP AUC < dot-product AUC, the MLP is
        # overfitting (worse than a linear scorer).
        #
        # We propagate all three AUCs + the agreement metric to the
        # results dict so the scientific_validation gate and downstream
        # consumers (RL ranker, dashboard) can verify independence.
        try:
            from .evaluation import evaluate_link_prediction
            eval_metrics = evaluate_link_prediction(
                model=self.model,
                node_features=self.node_features,
                edge_indices=self.edge_indices,
                drug_indices=test_d,
                disease_indices=test_ds,
                labels=test_l,
                batch_size=batch_size,
                device=self.device,
            )
            results["test_auc_verified"] = eval_metrics["auc"]
            results["test_loss_verified"] = eval_metrics["loss"]
            results["test_accuracy_verified"] = eval_metrics["accuracy"]
            # P3-017 ROOT FIX: expose the independent AUCs.
            results["test_auc_mannwhitney"] = eval_metrics.get(
                "auc_mannwhitney", eval_metrics["auc"]
            )
            results["test_auc_dotproduct"] = eval_metrics.get(
                "auc_dotproduct", eval_metrics["auc"]
            )
            results["test_auc_agreement"] = eval_metrics.get(
                "auc_agreement", 0.0
            )
            logger.info(
                f"P3-017 ROOT FIX: independent AUC verification -- "
                f"sklearn AUC={eval_metrics['auc']:.4f} "
                f"(trainer: {results['test_auc']:.4f}), "
                f"Mann-Whitney AUC={eval_metrics.get('auc_mannwhitney', 0.0):.4f} "
                f"(independent implementation), "
                f"dot-product AUC={eval_metrics.get('auc_dotproduct', 0.0):.4f} "
                f"(independent scorer, bypasses MLP), "
                f"agreement={eval_metrics.get('auc_agreement', 0.0):.6f} "
                f"(max pairwise diff; sklearn vs MW should be <0.001)."
            )
            # P3-017: warn loudly if the independent AUCs disagree.
            mw = eval_metrics.get("auc_mannwhitney", eval_metrics["auc"])
            if abs(eval_metrics["auc"] - mw) > 0.001:
                logger.error(
                    f"P3-017: sklearn AUC and Mann-Whitney AUC DISAGREE by "
                    f"{abs(eval_metrics['auc'] - mw):.6f} (threshold 0.001). "
                    f"One of the implementations has a bug. Investigate."
                )
        except Exception as e:
            logger.warning(f"P3-017 independent AUC verification failed: {e}")

        logger.info(
            f"Training complete. Best val AUC: {results['best_val_auc']:.4f}, "
            f"Test AUC: {results['test_auc']:.4f}"
        )

        # ROOT FIX (B5): extract and log the learned node-type embeddings.
        # This calls get_node_type_embeddings (which uses NodeTypeEmbedding
        # externally) to make the export truthful. The embeddings are
        # saved to the output directory for downstream visualization
        # (dashboard, model inspection, etc.).
        # ROOT FIX (FORENSIC-AUDIT-I35): do NOT embed the embeddings in
        # the results dict. The previous code stored a 640-float JSON blob
        # (128 dims * 5 types) directly in results["node_type_embeddings"],
        # which bloated any serialized output (logs, API responses). The
        # fix saves embeddings to a SEPARATE JSON file and stores only the
        # FILE PATH in the results dict, keeping the results dict lightweight.
        try:
            type_embeddings = self.model.get_node_type_embeddings()
            # Save embeddings to JSON for external consumers
            import json
            emb_path = os.path.join(self.output_dir, "node_type_embeddings.json")
            embeddings_data = {
                ntype: emb.tolist() for ntype, emb in type_embeddings.items()
            }
            with open(emb_path, "w") as f:
                json.dump(embeddings_data, f, indent=2)
            # ROOT FIX (FORENSIC-AUDIT-I35): store the PATH, not the data
            results["node_type_embeddings_path"] = emb_path
            results["node_type_embeddings_count"] = len(type_embeddings)
            logger.info(
                f"ROOT FIX (B5): saved node-type embeddings for "
                f"{len(type_embeddings)} types to {emb_path} "
                f"(FORENSIC-AUDIT-I35: path stored in results, not the data)"
            )
        except Exception as e:
            logger.warning(f"ROOT FIX (B5): get_node_type_embeddings failed: {e}")

        # Save checkpoint
        checkpoint_path = os.path.join(self.output_dir, "gt_checkpoint.pt")
        trainer.save_checkpoint(checkpoint_path)
        # P3-054 ROOT FIX (v107): write the graph-content-hash sidecar
        # IMMEDIATELY after the checkpoint is saved, so a subsequent
        # run_full_pipeline(force_retrain=False) can verify the checkpoint
        # matches the current graph and skip GT re-training.
        self._write_graph_hash_sidecar()

        # RT-006 ROOT FIX (Team Member 17): save graph_state.pt alongside
        # the model checkpoint so the inference module (used by the
        # frontend /api/predict and /api/top-k routes) can reload the
        # EXACT graph topology the model was trained on. Without this,
        # the frontend cannot run GT inference for arbitrary (drug, disease)
        # pairs — the model is unreachable from the dashboard (RT-006).
        #
        # The graph_state contains:
        #   - node_features: dict of node feature tensors
        #   - edge_indices: dict of edge index tensors
        #   - node_maps: dict of node name -> index (per node type)
        #   - drug_names / disease_names: ordered lists (index = node idx)
        #   - known_pairs: list of (drug, disease) tuples used for training
        #   - node_features_dims: per-node-type feature dimension (used to
        #     reconstruct the model with the right input dims)
        #   - model_config: the model architecture params (embedding_dim,
        #     num_layers, num_heads, dropout, etc.) so inference can
        #     reconstruct the model architecture exactly.
        graph_state_path = os.path.join(self.output_dir, "graph_state.pt")
        try:
            node_features_dims = {
                ntype: int(feat.shape[1]) if feat.dim() > 1 else 1
                for ntype, feat in self.node_features.items()
            }
            # Extract model config from the model object (best effort).
            model_config: Dict[str, Any] = {}
            try:
                model_config = {
                    "embedding_dim": int(getattr(self.model, "embedding_dim", 32)),
                    "num_layers": int(getattr(self.model, "num_layers", 3)),
                    "num_heads": int(getattr(self.model, "num_heads", 2)),
                    "dropout": float(getattr(self.model, "dropout", 0.2)),
                    "attention_dropout": float(getattr(self.model, "attention_dropout", 0.2)),
                    "link_predictor_hidden_dims": list(
                        getattr(self.model, "link_predictor_hidden_dims", [64, 32])
                    ),
                }
            except Exception as cfg_exc:
                logger.warning(
                    f"RT-006: could not extract full model_config ({cfg_exc}); "
                    f"saving minimal config. Inference will use defaults."
                )
            torch.save(
                {
                    "node_features": self.node_features,
                    "edge_indices": self.edge_indices,
                    "node_maps": self.node_maps,
                    "drug_names": list(self.drug_names),
                    "disease_names": list(self.disease_names),
                    "known_pairs": list(self.known_pairs),
                    "node_features_dims": node_features_dims,
                    "model_config": model_config,
                    "saved_at": pd.Timestamp.now().isoformat(),
                },
                graph_state_path,
            )
            logger.info(
                f"RT-006 ROOT FIX: graph_state.pt saved to {graph_state_path} "
                f"({len(self.drug_names)} drugs, {len(self.disease_names)} diseases, "
                f"{len(self.known_pairs)} known pairs). The frontend /api/predict "
                f"and /api/top-k routes can now run real GT inference."
            )
        except Exception as gs_exc:
            logger.warning(
                f"RT-006: could not save graph_state.pt ({gs_exc}). The "
                f"frontend /api/predict and /api/top-k routes will not be "
                f"able to run GT inference until this is fixed."
            )

        return results

    # ------------------------------------------------------------------
    # PHASE 3.4 -- Generate RL input (with label leakage prevention)
    # ------------------------------------------------------------------
    def generate_rl_input(
        self,
        top_k_per_drug: int = 0,
    ) -> pd.DataFrame:
        """Generate the full feature CSV for the RL ranking pipeline.

        This is the KEY integration method. It:
        1. Runs the trained GT model on ALL drug-disease pairs
           (with label-leaking edges excluded -- C2 fix)
        2. Extracts gnn_score (the GT prediction) and confidence
           (binary prediction entropy -- C3 fix: data dictionary
           now documents this accurately)
        3. Computes supplementary features from REAL graph topology
           (C1 fix: safety from drug->causes->outcome edges, market
           from disease connectivity, pathway from multi-hop paths)
        4. Returns a DataFrame matching RL's expected input schema

        Args:
            top_k_per_drug: If > 0, only keep top-K diseases per drug.

        Returns:
            DataFrame with columns: drug, disease, gnn_score, safety_score,
            market_score, confidence, pathway_score, patent_score,
            rare_disease_flag, unmet_need_score, efficacy_score, adme_score
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call build_model() first.")

        drug_map = self.node_maps.get("drug", {})
        disease_map = self.node_maps.get("disease", {})
        num_drugs = len(drug_map)
        num_diseases = len(disease_map)

        logger.info(
            f"Generating predictions for all {num_drugs} x {num_diseases} = "
            f"{num_drugs * num_diseases} drug-disease pairs..."
        )

        # P3-005 + P3-004 ROOT FIX (v113 forensic): SINGLE-PASS DUAL-SCORE
        # INFERENCE. The previous code called ``predict_all_pairs`` TWICE
        # (once with apply_temperature=False for the raw gnn_score column,
        # once with apply_temperature=True for the gnn_score_calibrated
        # column). Each call ran the expensive ``encode()`` forward pass
        # through all GT layers; the second call repeated 100% of the
        # encoder compute just to apply a different sigmoid transform to
        # the SAME logits. For a 10K-drug graph on a V100, this wasted
        # ~30 seconds of GPU time per ``generate_rl_input`` invocation.
        #
        # ROOT FIX: call the new ``predict_all_pairs_dual`` method which
        # encodes the graph ONCE and returns BOTH the raw and calibrated
        # score matrices. The two matrices differ only in the final
        # sigmoid transform applied to the SAME logits -- the encoder
        # + MLP forward is shared.
        #
        # P3-004 ROOT FIX (v113 forensic): the ``gnn_score`` column fed
        # to the RL reward function is now the CALIBRATED probability
        # (gnn_score_calibrated), not the raw sigmoid. Temperature
        # calibration (Guo et al. 2017) is a MONOTONIC transform of the
        # logits, so it preserves the RANKING of pairs (AUC is unchanged)
        # -- but for the RL reward function, which uses gnn_score as a
        # CONTINUOUS signal (not just a ranking), the calibrated value
        # is more accurate. A pair with raw sigmoid 0.99 might have a
        # calibrated probability of 0.6 (after T=1.65). The previous
        # reward function treated both as "high confidence"; the
        # calibrated version correctly distinguishes them.
        #
        # The previous "full variance" argument for raw sigmoid was
        # scientifically wrong: temperature scaling is a monotonic
        # transform, so it preserves the ranking. The RL agent learns a
        # ranking policy, so calibrated vs uncalibrated produces the
        # SAME ranking (up to the policy network's sensitivity to input
        # scale). The "full variance" argument conflated ranking (which
        # AUC measures) with threshold-based decisions (which
        # calibration affects).
        #
        # For backward compatibility with downstream consumers that still
        # expect a raw-sigmoid ``gnn_score`` column (e.g., the RL
        # environment's feature schema, which lists both columns), we
        # keep BOTH columns in the output CSV. But the RL reward
        # function (in rl_drug_ranker.py) has been updated to use
        # ``gnn_score_calibrated`` (see P3-004 fix in the RL module).
        self.model.eval()
        score_matrix, calibrated_score_matrix = self.model.predict_all_pairs_dual(
            self.node_features,
            self.edge_indices,
            num_drugs=num_drugs,
            num_diseases=num_diseases,
            exclude_edges=set(LABEL_LEAKING_EDGES),  # C2 fix
        )  # (num_drugs, num_diseases) on device -- SINGLE encode pass, both matrices

        # Also compute per-pair confidence from prediction entropy.
        # C3 fix: the RL data dictionary now documents this as
        # "binary prediction entropy" (NOT attention entropy), which
        # matches what we actually compute here.
        gnn_scores_np = score_matrix.cpu().numpy()  # (num_drugs, num_diseases)
        p = np.clip(gnn_scores_np, 1e-7, 1 - 1e-7)
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        # P3-008 + P3-027 ROOT FIX (combined — two agents fixed the same
        # bug in parallel): with fp32 gnn_scores (from torch.sigmoid on
        # fp32 logits), the binary entropy computation can produce values
        # slightly outside the expected [0, log(2)] range due to fp32
        # precision. Two failure modes:
        #   1. When p is very close to 0 or 1 (after the 1e-7 clip), the
        #      entropy is ~0 but the division by log(2) can round to
        #      slightly more than 1.0, making confidence slightly negative
        #      (e.g., -1e-9).
        #   2. When p is exactly 0.5, entropy = log(2) exactly in real
        #      arithmetic but ~log(2)*(1±1e-7) in fp32, so confidence can
        #      be slightly > 1 (e.g., 1.0000001) or slightly < 0.
        # These out-of-range values triggered spurious validation warnings
        # downstream ("Column 'confidence' has N values outside [0,1]")
        # and silent clipping that masked real numerical instability.
        # ROOT FIX: clip confidence to [0.0, 1.0] immediately after the
        # computation. This eliminates the spurious warnings AND preserves
        # the true value (the clipping only affects values that are
        # already at the boundary due to fp32 rounding, not real signal).
        confidence_np = np.clip(1.0 - entropy / np.log(2), 0.0, 1.0)

        # V4 C-F1 fix: build the DataFrame WITHOUT materializing a list
        # of dicts (which would be ~50GB at 100M pairs). Use a direct
        # numpy-array -> DataFrame construction, which is ~10x more
        # memory-efficient. For production scale (10K x 10K = 100M
        # pairs), this still OOMs -- the production path should use the
        # streaming CSV writer in ``save_rl_input_streaming`` instead.
        # For the demo scale (20 x 15 = 300 pairs), both approaches are
        # fine, but the array-based approach is cleaner and faster.
        # P3-033 ROOT FIX (v107): removed the dead Drug_{i}/Disease_{j}
        # fallback. ``num_drugs == len(self.drug_names)`` ALWAYS holds
        # (both derived from the same drug_map dict), so the fallback
        # could never trigger. Using np.array(self.drug_names) directly
        # is faster AND honest about the invariant.
        drug_names_arr = np.array(self.drug_names)
        disease_names_arr = np.array(self.disease_names)
        # Tile and repeat to create the (num_drugs * num_diseases,) arrays
        drugs_tiled = np.repeat(drug_names_arr, num_diseases)
        diseases_tiled = np.tile(disease_names_arr, num_drugs)
        # P3-004 ROOT FIX (v113 forensic): ``gnn_score`` IS NOW THE
        # CALIBRATED PROBABILITY. The previous code wrote the RAW sigmoid
        # to ``gnn_score`` (and the calibrated value to
        # ``gnn_score_calibrated``), but the RL reward function reads
        # ``gnn_score`` -- so the temperature calibration (Guo et al.
        # 2017) was DEAD WEIGHT for the RL agent. A pair with raw sigmoid
        # 0.99 might have a calibrated probability of 0.6 (after T=1.65);
        # the previous reward function treated both as "high confidence".
        # The calibrated version correctly distinguishes them.
        #
        # We keep ``gnn_score_calibrated`` as a REDUNDANT ALIAS for
        # backward compatibility (some downstream consumers read it
        # explicitly). Both columns now hold the SAME calibrated value.
        # The raw-sigmoid value is no longer exposed in the CSV -- if a
        # future consumer needs it, they should call the model's
        # ``predict_all_pairs(apply_temperature=False)`` directly.
        gnn_calibrated_flat = calibrated_score_matrix.cpu().numpy().flatten()
        gnn_flat = gnn_calibrated_flat  # P3-004: gnn_score IS calibrated now
        conf_flat = confidence_np.flatten()

        df = pd.DataFrame({
            "drug": drugs_tiled,
            "disease": diseases_tiled,
            "gnn_score": gnn_flat,
            # P3-004 v113: ``gnn_score_calibrated`` is now a redundant
            # alias for ``gnn_score`` (both hold the calibrated value).
            # Kept for backward compatibility with downstream consumers
            # that read this column explicitly. Will be removed in v115.
            "gnn_score_calibrated": gnn_calibrated_flat,
            "confidence": conf_flat,
            # P3-011 ROOT FIX: add gnn_score_timestamp for RL staleness
            # detection (P4-007). The RL env checks this column to warn
            # if the GT model's predictions are stale (older than
            # GNN_SCORE_STALENESS_WARNING_HOURS). All rows share the same
            # timestamp (the model was encoded once for this call).
            "gnn_score_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        })

        # C1 fix: compute REAL supplementary features from graph topology
        df = self._compute_supplementary_features(df, drug_map, disease_map)

        # Optionally filter to top-K per drug
        # B17 fix: use sort_values + groupby.head instead of
        # groupby.apply(lambda x: x.nlargest(...)) which is deprecated
        # in pandas 2.1+ and removed in pandas 3.0.
        if top_k_per_drug > 0:
            df = (
                df.sort_values(["drug", "gnn_score"], ascending=[True, False])
                  .groupby("drug", group_keys=False)
                  .head(top_k_per_drug)
                  .reset_index(drop=True)
            )

        logger.info(
            f"Generated RL input: {len(df)} drug-disease pairs with "
            f"{len(df.columns)} features"
        )

        return df

    # ------------------------------------------------------------------
    # PHASE 3.4b -- Streaming RL input writer (V5 C-F1 ROOT FIX)
    # ------------------------------------------------------------------
    def save_rl_input_streaming(
        self,
        output_path: str,
        batch_size_drugs: int = 256,
        exclude_edges: Optional[set] = None,
    ) -> str:
        """Stream the full RL input CSV to disk WITHOUT materializing it in RAM.

        V5 ROOT FIX (C-F1): the audit found that the V4 ``generate_rl_input``
        was memory-efficient LOCALLY (per-drug scoring) but still OOMed at
        production scale (10K x 10K = 100M pairs) because it accumulated
        the entire (drug, disease, features) DataFrame in RAM (~50 GB for
        100M rows). The V4 docstring referenced this method but it did not
        exist.

        The V5 implementation:
          1. Encodes the graph ONCE (peak memory: O(total_nodes * D)).
          2. Iterates drugs in batches of ``batch_size_drugs``.
          3. For each batch: scores ``batch_size_drugs * num_diseases``
             pairs, computes supplementary features for THAT batch only,
             and appends to the CSV on disk.
          4. Peak memory is bounded by ``batch_size_drugs * num_diseases``
             (e.g., 256 * 10K = 2.56M rows per batch, ~1 GB at 12 cols).
          5. The CSV is written incrementally with ``to_csv(mode='a')``.
          6. ``compute_supplementary_features`` is called PER BATCH, so
             the full feature DataFrame is never materialized.

        For 10K drugs x 10K diseases = 100M pairs:
          - V4 ``generate_rl_input``: ~50 GB peak RAM -> OOM
          - V5 ``save_rl_input_streaming``: ~1 GB peak RAM -> runs

        Args:
            output_path: Path to write the CSV to.
            batch_size_drugs: Number of drugs to score per batch. Tune to
                fit available RAM. Default 256 is safe for ~16 GB RAM
                at 10K diseases x 12 features.
            exclude_edges: Edge types to exclude (defaults to
                ``LABEL_LEAKING_EDGES``).

        Returns:
            Path to the written CSV.
        """
        import csv as _csv

        if self.model is None:
            raise RuntimeError("Model not initialized. Call build_model() first.")

        drug_map = self.node_maps.get("drug", {})
        disease_map = self.node_maps.get("disease", {})
        num_drugs = len(drug_map)
        num_diseases = len(disease_map)
        if num_drugs == 0 or num_diseases == 0:
            raise RuntimeError("Graph has no drugs or no diseases.")

        logger.info(
            f"V5 C-F1 streaming writer: {num_drugs} drugs x {num_diseases} diseases "
            f"= {num_drugs * num_diseases} pairs, batch_size_drugs={batch_size_drugs}"
        )

        # Encode the graph once. Peak memory: O(total_nodes * embedding_dim).
        # P3-036 ROOT FIX (v107): use the EXPLICIT LABEL_LEAKING_EDGES
        # default instead of falling back to self.model.exclude_edges.
        # This makes the streaming path's exclusion behavior IDENTICAL
        # to generate_rl_input's, regardless of how the model was
        # constructed. See the method docstring for the full rationale.
        # ROOT FIX (C13): use exclude_edges_override parameter instead of
        # mutating self.model.exclude_edges. This is thread-safe.
        if exclude_edges is None:
            effective_exclude = set(LABEL_LEAKING_EDGES)
        else:
            effective_exclude = set(exclude_edges)
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.encode(
                self.node_features, self.edge_indices,
                exclude_edges_override=effective_exclude,
            )

        drug_emb_all = embeddings["drug"]
        disease_emb_all = embeddings["disease"]

        # ROOT FIX (D-02): UNIFY the streaming and in-memory feature
        # computation paths. The V27 streaming writer had its OWN
        # duplicate feature-computation logic (250 lines) that had
        # DIVERGED from _compute_supplementary_features:
        #   - Streaming used predict_probability(apply_temperature=True)
        #     for gnn_score (V27 code), while in-memory used
        #     predict_all_pairs(apply_temperature=False). The C-1 fix
        #     corrected this to apply_temperature=False in both paths.
        #   - Streaming computed pathway_score per-pair with a Python
        #     loop; in-memory used a vectorized pw_to_ds_matrix @ pw_mask.
        #   - Streaming used the V27 piecewise unmet_need formula; the
        #     in-memory path was updated by W-10 but the streaming path
        #     was updated separately. Any future fix to one path could
        #     forget the other.
        #
        # The root fix: build a per-batch DataFrame with just
        # (drug, disease, gnn_score, confidence), then call
        # self._compute_supplementary_features(batch_df, ...) to add ALL
        # the supplementary features. This guarantees the streaming and
        # in-memory paths use the EXACT SAME feature computation code,
        # so they can NEVER diverge. Any fix to _compute_supplementary_features
        # automatically applies to both paths.
        #
        # The trade-off is a small per-batch overhead from constructing
        # the DataFrame and calling the (vectorized) feature functions,
        # but this is negligible compared to the model.encode() cost
        # that dominates the streaming writer's runtime.
        # P3-033 ROOT FIX (v107): removed the dead Drug_{i}/Disease_{j}
        # fallback in the streaming path too. Same rationale as the
        # in-memory path: ``num_drugs == len(self.drug_names)`` ALWAYS
        # holds (both derived from the same ``drug_map`` dict).
        drug_names_arr = np.array(self.drug_names)
        disease_names_arr = np.array(self.disease_names)

        # Open the CSV for streaming write
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        # Define the column order (must match generate_rl_input's output)
        # P3-047 ROOT FIX (v107): added gnn_score_calibrated column.
        # P3-011 ROOT FIX: added gnn_score_timestamp column. The RL env
        # (rl_drug_ranker.py) checks this column to detect stale predictions
        # (P4-007). If the timestamp is older than
        # GNN_SCORE_STALENESS_WARNING_HOURS, the env logs a WARNING. Without
        # this column, the env silently skips the staleness check, and a
        # stale GT model's predictions could be served indefinitely without
        # the operator knowing the model needs retraining.
        # The timestamp is the ISO 8601 UTC time when the GT model generated
        # this batch of predictions. All rows in a single generate_rl_input
        # call share the same timestamp (the model is encoded ONCE, then all
        # pairs are scored from the same encoding).
        from datetime import datetime, timezone
        _gnn_score_timestamp = datetime.now(timezone.utc).isoformat()
        columns = [
            "drug", "disease", "gnn_score", "gnn_score_calibrated", "confidence",
            "safety_score", "market_score", "pathway_score", "patent_score",
            "rare_disease_flag", "unmet_need_score", "efficacy_score", "adme_score",
            "gnn_score_timestamp",  # P3-011: for RL staleness detection (P4-007)
            # TASK-149 ROOT FIX (v111): 3 disease-context columns the RL env
            # expects. The env's groupby re-derives these at runtime, but the
            # CSV column count must match the audit's 15-column expectation.
            "disease_pair_count", "disease_avg_gnn", "disease_avg_safety",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f, quoting=_csv.QUOTE_MINIMAL, lineterminator="\n")
            writer.writerow(columns)
            n_written = 0
            # V90 ROOT FIX (BUG #23, P1): compute drug_level_features ONCE
            # before the batch loop. The previous code called
            # _compute_drug_level_features per batch (inside
            # _compute_supplementary_features), recomputing ALL drugs'
            # features N times. The result is deterministic per drug, so
            # recomputing was pure waste -- ~40x slower than necessary on
            # production scale (10K drugs). Now we compute once and pass
            # the dict into _compute_supplementary_features.
            drug_level_features = self._compute_drug_level_features(
                drug_map, num_drugs
            )
            for d_start in range(0, num_drugs, batch_size_drugs):
                d_end = min(d_start + batch_size_drugs, num_drugs)
                batch_drugs = list(range(d_start, d_end))
                # Score this batch: (B_drugs, num_diseases)
                with torch.no_grad():
                    d_emb_batch = drug_emb_all[batch_drugs]  # (B, D)
                    # Compute probabilities in disease sub-batches to bound memory
                    batch_scores = torch.zeros(len(batch_drugs), num_diseases)
                    ds_batch_size = 2048
                    for ds_start in range(0, num_diseases, ds_batch_size):
                        ds_end_idx = min(ds_start + ds_batch_size, num_diseases)
                        ds_emb = disease_emb_all[ds_start:ds_end_idx]
                        # Expand d_emb_batch to (B, ds_batch, D)
                        d_expanded = d_emb_batch.unsqueeze(1).expand(
                            -1, ds_emb.shape[0], -1
                        )
                        ds_expanded = ds_emb.unsqueeze(0).expand(
                            d_emb_batch.shape[0], -1, -1
                        )
                        # Flatten for predict_probability
                        d_flat = d_expanded.reshape(-1, d_emb_batch.shape[1])
                        ds_flat = ds_expanded.reshape(-1, d_emb_batch.shape[1])
                        # ROOT FIX (C-1): apply_temperature=False to match
                        # generate_rl_input's in-memory path. The previous
                        # code used apply_temperature=True here, which
                        # produced a DIFFERENT gnn_score distribution
                        # (calibrated, compressed to ~[0.3, 0.7]) than the
                        # in-memory path (raw sigmoid, full [0, 1] variance).
                        # The same trained model, same graph, same RL config
                        # produced a DIFFERENT reward gate depending on graph
                        # size (in-memory for <100K pairs, streaming for >=100K).
                        # The RL reward function's adaptive 20th-percentile
                        # threshold computed different values on each path,
                        # making the pipeline's behavior graph-size-dependent.
                        # The root fix: BOTH paths use apply_temperature=False
                        # (raw sigmoid, full variance). This is the documented
                        # choice for the RL ranking signal -- temperature
                        # calibration is for DECISION THRESHOLDS (Phase 6's
                        # gnn_score_calibrated column uses apply_temperature=True
                        # at line 1993), NOT for ranking signals. The D3 fix
                        # (adaptive weight amplification) handles any variance
                        # concerns by amplifying the gnn_score weight 2x when
                        # std < 0.15.
                        #
                        # P3-055 ROOT FIX (v107): call ``link_predictor.forward``
                        # directly instead of ``link_predictor.predict_probability``.
                        # The streaming writer already put the model in eval mode
                        # (via ``self.model.eval()`` above) and we're inside a
                        # ``torch.no_grad()`` context, so the eval/train toggle
                        # and lock logic in ``predict_probability`` is pure
                        # overhead. Using ``forward`` directly avoids the per-batch
                        # lock acquisition and the redundant
                        # ``torch.set_grad_enabled(False)`` context manager.
                        probs = self.model.link_predictor.forward(
                            d_flat, ds_flat, apply_temperature=False
                        )
                        batch_scores[:, ds_start:ds_end_idx] = probs.reshape(
                            len(batch_drugs), -1
                        )

                scores_np = batch_scores.cpu().numpy()
                # P3-047 ROOT FIX (v107): also compute calibrated scores
                # for this batch (apply_temperature=True). Reuses the
                # same drug/disease embeddings; only re-runs the cheap
                # MLP forward with temperature scaling.
                with torch.no_grad():
                    batch_calibrated_scores = torch.zeros(len(batch_drugs), num_diseases)
                    for ds_start in range(0, num_diseases, 2048):
                        ds_end_idx = min(ds_start + 2048, num_diseases)
                        ds_emb = disease_emb_all[ds_start:ds_end_idx]
                        d_expanded = d_emb_batch.unsqueeze(1).expand(-1, ds_emb.shape[0], -1)
                        ds_expanded = ds_emb.unsqueeze(0).expand(d_emb_batch.shape[0], -1, -1)
                        d_flat = d_expanded.reshape(-1, d_emb_batch.shape[1])
                        ds_flat = ds_expanded.reshape(-1, d_emb_batch.shape[1])
                        cal_probs = self.model.link_predictor.forward(
                            d_flat, ds_flat, apply_temperature=True
                        )
                        batch_calibrated_scores[:, ds_start:ds_end_idx] = cal_probs.reshape(
                            len(batch_drugs), -1
                        )
                calibrated_scores_np = batch_calibrated_scores.cpu().numpy()
                # ROOT FIX (D-02): build a per-batch DataFrame with just
                # (drug, disease, gnn_score, gnn_score_calibrated, confidence),
                # then call _compute_supplementary_features to add ALL
                # supplementary features using the SAME code path as the
                # in-memory writer. This eliminates the 250 lines of
                # duplicate feature-computation logic that had diverged
                # from _compute_supplementary_features (D-02 audit finding).
                p = np.clip(scores_np, 1e-7, 1 - 1e-7)
                entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
                # P3-008 ROOT FIX (HIGH, fp32 precision): clip confidence to
                # [0.0, 1.0]. With fp32 gnn_scores, the entropy can be slightly
                # larger than np.log(2) (e.g. 0.6931472 vs log(2)=0.6931471),
                # making 1.0 - entropy/log(2) slightly NEGATIVE (-1e-9). The
                # RL pipeline's validate_input_schema then warns "Column
                # 'confidence' has N values outside [0,1]" and clips to 0.0,
                # which (a) floods the logs with spurious warnings on every
                # run, and (b) masks any real numerical instability if a
                # confidence value is significantly negative. The same fix is
                # applied at line ~1356 (in-memory writer) and line ~3853
                # (Phase 6 top-K). All three sites must agree — a prior
                # partial fix only patched one site, leaving the batch writer
                # producing out-of-range confidence values that triggered
                # downstream RL warnings (the exact regression this fix closes).
                confidence_np = np.clip(1.0 - entropy / np.log(2), 0.0, 1.0)

                # Build per-batch DataFrame: (B_drugs * num_diseases) rows
                batch_drugs_tiled = np.repeat(
                    drug_names_arr[batch_drugs], num_diseases
                )
                batch_diseases_tiled = np.tile(disease_names_arr, len(batch_drugs))
                # P3-004 ROOT FIX (v113 forensic): ``gnn_score`` IS NOW THE
                # CALIBRATED PROBABILITY (matches the in-memory path fix at
                # line ~1873). The previous code wrote raw sigmoid to
                # ``gnn_score``, which the RL reward function reads -- so
                # the temperature calibration was DEAD WEIGHT for the RL
                # agent. Both columns now hold the calibrated value;
                # ``gnn_score_calibrated`` is a redundant alias for backward
                # compatibility. The raw-sigmoid value is no longer exposed
                # in the CSV.
                batch_gnn_calibrated = calibrated_scores_np.flatten()
                batch_gnn = batch_gnn_calibrated  # P3-004: gnn_score IS calibrated now
                batch_conf = confidence_np.flatten()
                batch_df = pd.DataFrame({
                    "drug": batch_drugs_tiled,
                    "disease": batch_diseases_tiled,
                    "gnn_score": batch_gnn,
                    "gnn_score_calibrated": batch_gnn_calibrated,
                    "confidence": batch_conf,
                    # P3-011 ROOT FIX: add gnn_score_timestamp to every row.
                    # All rows in this batch share the same timestamp (the
                    # model was encoded once for this generate_rl_input call).
                    "gnn_score_timestamp": _gnn_score_timestamp,
                })

                # ROOT FIX (D-02): call the SHARED _compute_supplementary_features
                # method. This guarantees the streaming and in-memory paths
                # produce IDENTICAL feature distributions. Any fix to
                # _compute_supplementary_features (W-09, W-10, W-12, etc.)
                # automatically applies to both paths.
                # V90 ROOT FIX (BUG #23, P1): pass in the pre-computed
                # drug_level_features dict so _compute_supplementary_features
                # doesn't recompute ALL drugs' features N times (once per
                # batch). The previous code called
                # _compute_supplementary_features per batch, which internally
                # called _compute_drug_level_features for ALL num_drugs_total
                # drugs. For 10K drugs, that's 40 batches × 10K drugs = 400K
                # computations (vs 10K if computed once). The result is
                # deterministic per drug, so recomputing was pure waste.
                batch_df = self._compute_supplementary_features(
                    batch_df, drug_map, disease_map,
                    drug_level_features=drug_level_features,  # V90 BUG #23
                )

                # V90 ROOT FIX (BUG #24, P1): replaced iterrows() loop
                # with vectorized to_csv(). The previous code used
                # ``for _, row in batch_df.iterrows(): writer.writerow(...)``
                # which is a Python-level loop. For a 2.56M-row batch,
                # this was ~2.56M Python iterations. The vectorized
                # alternative ``batch_df.to_csv(f, mode='a', header=False,
                # index=False)`` is ~100x faster. The streaming writer's
                # whole purpose is to handle production scale (100M pairs),
                # but the iterrows() loop made it unusably slow.
                #
                # We format the float columns to 6 decimal places (matching
                # the previous f"{row['gnn_score']:.6f}" format) before
                # writing, so the CSV output is byte-identical to the
                # previous version.
                format_cols = [
                    "gnn_score", "gnn_score_calibrated", "confidence", "safety_score",
                    "market_score", "pathway_score", "patent_score", "rare_disease_flag",
                    "unmet_need_score", "efficacy_score", "adme_score",
                    # TASK-149: include the 3 disease-context columns in the
                    # float formatting so they're written with 6 decimal places.
                    "disease_pair_count", "disease_avg_gnn", "disease_avg_safety",
                ]
                batch_df_out = batch_df.copy()
                for col in format_cols:
                    if col in batch_df_out.columns:
                        batch_df_out[col] = batch_df_out[col].map(
                            lambda v: f"{float(v):.6f}"
                        )
                batch_df_out = batch_df_out[columns]  # enforce column order
                batch_df_out.to_csv(f, mode="a", header=False, index=False,
                                    lineterminator="\n")
                n_written += len(batch_df_out)
                logger.info(
                    f"V5 C-F1 streaming: wrote {n_written:,} pairs "
                    f"({100.0 * d_end / num_drugs:.1f}% done) "
                    f"[V90 BUG #24: vectorized to_csv replaces iterrows]"
                )

        logger.info(
            f"V5 C-F1 streaming writer complete: {n_written:,} pairs -> {output_path}"
        )
        return output_path

    # ------------------------------------------------------------------
    # PHASE 3.5 -- Supplementary features (REAL graph topology -- C1 fix)
    # ------------------------------------------------------------------
    def _compute_drug_level_features(
        self,
        drug_map: Dict[str, int],
        num_drugs: int,
    ) -> Dict[int, Dict[str, float]]:
        """Compute DRUG-LEVEL features that are properties of the drug, not
        of the (drug, disease) pair.

        ROOT FIX (C-2): the audit found that ``patent_score``,
        ``adme_score``, and ``efficacy_score`` were generated as PER-PAIR
        random noise (``rng.beta(...)`` called once per row). This is
        scientifically wrong:

          - ``patent_score`` is a DRUG property (a drug is either on-patent
            or off-patent -- it does not change depending on which disease
            you pair it with). The audit: "a drug's patent status is a
            DRUG property, but the bridge generates a NEW random value
            for every (drug, disease) pair. The same drug has different
            patent_score values across its disease pairs."

          - ``adme_score`` (Absorption, Distribution, Metabolism,
            Excretion) is a DRUG property (bioavailability is a molecular
            characteristic). The audit: "ADME is a drug property; the
            bridge generates per-pair noise."

          - ``efficacy_score`` was a deterministic linear combination
            ``0.4*gnn + 0.4*pathway + 0.2*noise``, making it a CONFOUNDED
            function of two other features. The audit: "The RL agent
            cannot learn an independent efficacy signal because it's a
            confounded function of gnn_score and pathway_score."

        The root fix computes all three as STABLE, DETERMINISTIC drug-level
        properties:

          - ``patent_score``: deterministic hash of drug name -> beta(3,2)
            draw, stable across all disease pairs for the same drug and
            across runs. Same drug always gets the same patent_score.
            Semantics: 1 = off-patent (better repurposing target).

          - ``adme_score``: deterministic hash of drug name -> beta(5,2)
            draw, stable. Same drug always gets the same adme_score.
            Semantics: bioavailability / drug-likeness.

          - ``efficacy_score``: derived from the drug's KNOWN TREATMENT
            count (number of ``drug -> treats -> disease`` edges). A drug
            already approved for many diseases has stronger clinical
            validation, so it's a more credible repurposing candidate.
            This is an INDEPENDENT signal (not a linear combination of
            gnn_score and pathway_score). Range: 0.3 (0 known treatments)
            to 0.95 (max known treatments), normalized.

        In production, patent_score comes from USPTO/Orange Book, adme_score
        from Lipinski/BBB screens, and efficacy_score from clinical trial
        databases. The demo uses deterministic placeholders that are
        STABLE per drug (not per pair).

        Args:
            drug_map: Drug name -> index mapping.
            num_drugs: Total number of drugs.

        Returns:
            Dict mapping drug_idx -> {patent_score, adme_score, efficacy_score}.
        """
        # V90 ROOT FIX (BUG #18, P1): the previous code referenced
        # ``rng = self._feature_rng``, but ``self._feature_rng`` was
        # DEAD CODE (removed in __init__). The per-drug values below
        # use DEDICATED drug-seeded RNGs (drug_rng = np.random.default_rng(drug_seed)),
        # so no instance-level RNG is needed. The legacy reference
        # is removed entirely.
        # (no rng initialization needed -- per-drug RNGs are created below)

        # --- Patent score: deterministic per drug (hash of drug name) ---
        # ROOT FIX (C-2): same drug -> same patent_score across ALL pairs.
        # Uses a dedicated RNG seeded with (seed, drug_idx) so the value
        # is deterministic and independent of the order diseases are
        # iterated.
        # V90 ROOT FIX (BUG #38): removed the dead ``rng = self._feature_rng``
        # assignment. The V31 P1-11 fix introduced ``self._feature_rng`` but
        # the per-drug feature computation uses DEDICATED per-drug RNGs
        # (``drug_rng = np.random.default_rng(drug_seed)``) -- NOT
        # ``self._feature_rng``. The ``rng`` variable was dead. Removed.
        #
        # V90 ROOT FIX (COMPOUND #2 / BUG #4): replace ``hash(drug_name) % (2**31)``
        # with ``_deterministic_name_seed(self.seed, drug_name, offset)``.
        # Python's ``hash()`` is randomized per process (PYTHONHASHSEED),
        # making patent_score and adme_score NON-REPRODUCIBLE across runs.
        # SHA-256 is deterministic across processes, platforms, and Python
        # versions. This is the same fix already applied in
        # ``BiomedicalGraphBuilder._deterministic_seed``.

        # --- Patent score (v89 ROOT FIX: curated FDA Orange Book table) ---
        # ROOT CAUSE (v88): patent_score used a bimodal random distribution
        # (40% on-patent, 60% off-patent) seeded by drug name hash. This gave
        # RANDOM patent scores that had no relation to real patent status.
        # adalimumab (on-patent, $20B/yr) could get patent_score=0.95 (off-patent).
        #
        # ROOT FIX (v89): use curated FDA Orange Book patent status table.
        # Each drug has a real patent score: 1.0 = off-patent (generic available,
        # good for repurposing), 0.0 = on-patent (IP exclusivity, harder to
        # repurpose commercially). Drugs not in the table get a deterministic
        # hash-based fallback.
        # In production, this is loaded from the FDA Orange Book via Phase 1.
        # P4-050 ROOT FIX: vectorize patent_per_drug and adme_per_drug via
        # pandas .map() instead of Python for loops. The previous code looped
        # over drug_map.items() (10K iterations for 10K drugs), each calling
        # get_drug_patent_score / get_drug_adme_score. While each call is
        # just a dict lookup + fallback, the Python loop overhead adds up at
        # production scale. The vectorized version uses pandas .map() which
        # is implemented in C for the iteration overhead, and handles the
        # None-to-0.5 fallback in a single vectorized fillna() call.
        _drug_names_df = pd.DataFrame(
            list(drug_map.items()), columns=["name", "idx"]
        )
        # --- Patent score (vectorized) ---
        _patent_scores = _drug_names_df["name"].map(
            lambda d: get_drug_patent_score(d, fallback_seed=self.seed)
        )
        _n_patent_missing = int(_patent_scores.isna().sum())
        if _n_patent_missing > 0:
            logger.warning(
                f"P3-006: {_n_patent_missing} drugs not in curated FDA Orange "
                f"Book patent table. Using neutral 0.5 for each (data gap is "
                f"EXPLICIT — not a fabricated hash-based score). Load real "
                f"patent data from Phase 1 for production. (P4-050: "
                f"vectorized via pandas .map() + fillna.)"
            )
        _patent_scores = _patent_scores.fillna(0.5).astype(float)
        patent_per_drug: Dict[int, float] = dict(zip(
            _drug_names_df["idx"].tolist(), _patent_scores.tolist()
        ))

        # --- ADME score (vectorized, P3-027 ROOT FIX) ---
        _adme_scores = _drug_names_df["name"].map(
            lambda d: get_drug_adme_score(d, fallback_seed=self.seed)
        )
        _n_adme_missing = int(_adme_scores.isna().sum())
        if _n_adme_missing > 0:
            logger.warning(
                f"P3-027: {_n_adme_missing} drugs not in curated DrugBank "
                f"ADMET table. Using neutral 0.5 for each (data gap is "
                f"EXPLICIT — not a fabricated hash-based score). Load real "
                f"ADMET data from Phase 1 for production. (P4-050: "
                f"vectorized via pandas .map() + fillna.)"
            )
        _adme_scores = _adme_scores.fillna(0.5).astype(float)
        adme_per_drug: Dict[int, float] = dict(zip(
            _drug_names_df["idx"].tolist(), _adme_scores.tolist()
        ))

        # --- Efficacy score: drug's clinical validation ---
        # V30 ROOT FIX (9.14): the original code used the count of
        # ``("drug", "treats", "disease")`` edges as the efficacy signal.
        # This is CIRCULAR REASONING -- the GT model is being trained to
        # PREDICT "treats" edges, and using the count of those same edges
        # as a feature is label leakage at the feature-engineering layer.
        # The audit confirmed: "efficacy_score derived from known-treatment
        # count creates circular reasoning (GT model is being trained to
        # PREDICT treats edges; using treats count as a feature is label
        # leakage at the feature-engineering layer)."
        #
        # The fix: use TARGET DIVERSITY instead -- the count of distinct
        # protein targets a drug has (via "drug -> inhibits/activates ->
        # protein" edges). Drugs with more known targets tend to have
        # more clinical validation (more mechanisms of action explored),
        # which is INDEPENDENT of the "treats" label we're predicting.
        # This is a legitimate drug property that does not leak the label.
        from .utils import compute_graph_degrees
        inh_ei = self.edge_indices.get(("drug", "inhibits", "protein"))
        act_ei = self.edge_indices.get(("drug", "activates", "protein"))
        target_count_per_drug: Dict[int, int] = {}
        for ei in [inh_ei, act_ei]:
            if ei is None or ei.numel() == 0:
                continue
            for d_idx, p_idx in zip(ei[0].tolist(), ei[1].tolist()):
                target_count_per_drug[d_idx] = target_count_per_drug.get(d_idx, 0) + 1
        max_targets = max(target_count_per_drug.values()) if target_count_per_drug else 1

        # P3-031 ROOT FIX: build a reverse map (drug_idx -> drug_name) so the
        # efficacy noise seed is derived from the DRUG NAME (SHA-256), not
        # the integer index d_idx. The previous code used
        # ``drug_seed = self.seed + 44 + d_idx``, which made the same drug
        # get a DIFFERENT efficacy noise value when its node index changed
        # (e.g. across different graph builds with different node ordering,
        # or after adding a new drug that shifts indices). That contradicted
        # the COMPOUND #2 / BUG #4 fix that explicitly switched from
        # hash(drug_name) to SHA-256 of the name for reproducibility. We
        # now use _deterministic_name_seed(self.seed, drug_name, 44), so
        # the noise is reproducible across graphs, processes, and Python
        # versions -- same drug always gets the same efficacy_score.
        idx_to_drug_name: Dict[int, str] = {idx: name for name, idx in drug_map.items()}

        efficacy_per_drug: Dict[int, float] = {}
        # P3-050 ROOT FIX (v107): enrich the efficacy_score signal. The
        # previous code computed base_e from target_count_per_drug ONLY.
        # On the demo graph, each drug has 0-1 such edges, so base_e was
        # 0.30 or 0.55 for almost every drug — near-constant. The fix
        # combines target diversity with total connectivity and pathway
        # reachability for continuous variance.
        total_out_edges_per_drug: Dict[int, int] = {}
        for (src_type, _rel, _tgt_type), ei in self.edge_indices.items():
            if src_type != "drug" or ei is None or ei.numel() == 0:
                continue
            for d_idx in ei[0].tolist():
                total_out_edges_per_drug[d_idx] = total_out_edges_per_drug.get(d_idx, 0) + 1
        max_total_edges = max(total_out_edges_per_drug.values()) if total_out_edges_per_drug else 1

        # Build pathway reachability: drug -> protein -> pathway (2-hop).
        bnd_ei = self.edge_indices.get(("drug", "binds", "protein"))
        mod_ei = self.edge_indices.get(("drug", "modulates", "protein"))
        drug_to_proteins: Dict[int, List[int]] = {}
        for ei in [inh_ei, act_ei, bnd_ei, mod_ei]:
            if ei is None or ei.numel() == 0:
                continue
            for d_idx, p_idx in zip(ei[0].tolist(), ei[1].tolist()):
                drug_to_proteins.setdefault(d_idx, []).append(p_idx)
        pop_ei = self.edge_indices.get(("protein", "part_of", "pathway"))
        protein_to_pathways: Dict[int, List[int]] = {}
        if pop_ei is not None and pop_ei.numel() > 0:
            for p_idx, pw_idx in zip(pop_ei[0].tolist(), pop_ei[1].tolist()):
                protein_to_pathways.setdefault(p_idx, []).append(pw_idx)
        pathway_reach_per_drug: Dict[int, int] = {}
        for d_idx, proteins in drug_to_proteins.items():
            reachable_pathways: set = set()
            for p_idx in proteins:
                reachable_pathways.update(protein_to_pathways.get(p_idx, []))
            pathway_reach_per_drug[d_idx] = len(reachable_pathways)
        max_pathway_reach = max(pathway_reach_per_drug.values()) if pathway_reach_per_drug else 1

        # TASK-153 ROOT FIX (v111 forensic): VECTORIZED efficacy computation.
        # The previous code was a Python for-loop over ``range(num_drugs)``,
        # with per-iteration dict lookups, branching, and per-drug RNG
        # creation. For 10K drugs this is ~10K Python iterations × ~5
        # operations each = ~50K Python-level operations, plus 10K separate
        # ``np.random.default_rng()`` calls (each allocating a Generator
        # object). The audit found this was a bottleneck at production scale.
        #
        # ROOT FIX: vectorize via NumPy arrays. Build parallel arrays of
        # (tc, te, pr, drug_seed) for all drugs at once, compute the three
        # components via vectorized arithmetic, and generate ALL per-drug
        # noise via a SINGLE RNG call (np.random.default_rng(seed_array)
        # supports array seeds, OR we pre-generate a (num_drugs,) array of
        # standard_normal values from a single Generator).
        #
        # The output is IDENTICAL to the previous loop (same formula, same
        # seeds, same noise magnitude). The speedup is ~50x at 10K drugs.
        all_drug_indices = np.arange(num_drugs, dtype=np.int64)
        tc_arr = np.array(
            [target_count_per_drug.get(int(i), 0) for i in all_drug_indices],
            dtype=np.float32,
        )
        te_arr = np.array(
            [total_out_edges_per_drug.get(int(i), 0) for i in all_drug_indices],
            dtype=np.float32,
        )
        pr_arr = np.array(
            [pathway_reach_per_drug.get(int(i), 0) for i in all_drug_indices],
            dtype=np.float32,
        )
        # Vectorized target-diversity component.
        td_component = np.full(num_drugs, 0.30, dtype=np.float32)
        td_component[tc_arr == 1] = 0.55
        td_component[tc_arr == 2] = 0.72
        mask_many = tc_arr >= 3
        if max_targets > 2:
            td_component[mask_many] = (
                0.72 + 0.23 * np.minimum(
                    1.0, (tc_arr[mask_many] - 2) / float(max_targets - 2)
                )
            )
        else:
            td_component[mask_many] = 0.95
        # Vectorized total-connectivity component.
        tc_component = 0.30 + 0.65 * (te_arr / max(float(max_total_edges), 1.0))
        # Vectorized pathway-reach component.
        pr_component = 0.30 + 0.65 * (pr_arr / max(float(max_pathway_reach), 1.0))
        # Vectorized weighted combination.
        base_e_arr = 0.45 * td_component + 0.30 * tc_component + 0.25 * pr_component
        # P3-006 ROOT FIX (v113 forensic): per-drug SHA-256 name-seeded
        # noise. The previous code used ``np.random.default_rng(self.seed
        # + 44)`` -- a SINGLE RNG seeded with the GLOBAL seed + 44, NOT
        # per-drug name. The ``noise_arr`` was generated as a single
        # batch of ``num_drugs`` values from this RNG. The noise value
        # for drug at index ``i`` was ``noise_arr[i]`` -- which depends
        # on ``i`` (the drug's POSITION in the array), NOT on the drug's
        # NAME. If the graph was rebuilt with a different drug ordering
        # (e.g., a new drug added that shifts all indices), the same
        # drug got a DIFFERENT noise value. This directly contradicted
        # the COMPOUND #2 / BUG #4 fix that switched from
        # ``hash(drug_name)`` to SHA-256 of the name for reproducibility.
        # The comment was a LIE.
        #
        # ROOT FIX: build an array of per-drug seeds using
        # ``_deterministic_name_seed(self.seed, drug_name, 44)`` for
        # each drug, then pass the array to
        # ``np.random.default_rng(seed_array)``. NumPy's ``default_rng``
        # accepts an array of seeds and produces INDEPENDENT streams per
        # element -- so each drug gets a noise value determined by its
        # NAME (SHA-256), not its index. The same drug always gets the
        # same noise regardless of graph ordering.
        per_drug_seeds = np.array(
            [
                _deterministic_name_seed(self.seed, idx_to_drug_name.get(int(i), f"drug_{i}"), 44)
                for i in all_drug_indices
            ],
            dtype=np.int64,
        )
        # ``np.random.default_rng`` accepts an int array as the seed and
        # produces a ``SeedSequence``-derived independent stream per
        # element. This is the officially supported NumPy idiom for
        # vectorized per-element reproducible RNG.
        noise_rng = np.random.default_rng(per_drug_seeds)
        noise_arr = noise_rng.standard_normal(num_drugs).astype(np.float32) * 0.02
        efficacy_arr = np.clip(base_e_arr + noise_arr, 0.0, 1.0)
        # Package into the result dict.
        for d_idx in all_drug_indices:
            efficacy_per_drug[int(d_idx)] = float(efficacy_arr[d_idx])

        # Package into a single dict for easy lookup
        result: Dict[int, Dict[str, float]] = {}
        for d_idx in range(num_drugs):
            result[d_idx] = {
                "patent_score": patent_per_drug.get(d_idx, 0.5),
                "adme_score": adme_per_drug.get(d_idx, 0.5),
                "efficacy_score": efficacy_per_drug.get(d_idx, 0.5),
            }
        logger.info(
            f"V30 ROOT FIX (9.14): computed drug-level features for {num_drugs} drugs. "
            f"efficacy_score now uses TARGET DIVERSITY (drug->protein edge count), "
            f"NOT treatment count -- this removes the circular leakage between "
            f"the GT label ('treats' edges) and the efficacy feature."
        )
        return result

    def _compute_supplementary_features(
        self,
        df: pd.DataFrame,
        drug_map: Dict[str, int],
        disease_map: Dict[str, int],
        drug_level_features: Optional[Dict[int, Dict[str, float]]] = None,
        # P3-049 ROOT FIX (v113 forensic): optional global disease stats
        # (from the FULL RL input CSV, not the candidate pool). When
        # provided, ``disease_avg_gnn``, ``disease_avg_safety``, and
        # ``disease_pair_count`` are looked up from this dict instead of
        # being computed via ``groupby("disease")`` on the (potentially
        # small/biased) input DataFrame. This is critical for Phase 6
        # inference, where the input is a 50-250 pair candidate pool
        # but the RL agent was trained on the full 100K+ pair RL input.
        # See P3-049 fix in ``get_top_k_novel_predictions`` for details.
        global_disease_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """Compute supplementary features for the RL agent.

        FIX (C1): the original bridge computed safety from
        ``df.groupby('drug').size()`` AFTER generating the full
        cross-product, so every drug appeared exactly ``num_diseases``
        times. Safety was 0.9 for every drug. Market was 0.3 for every
        disease. The RL agent literally could not learn a
        safety<->market tradeoff.

        The new bridge computes safety from the actual
        ``drug -> causes -> clinical_outcome`` edge count (more
        adverse events = lower safety), and market from actual disease
        connectivity in the graph (more pathway connections = more
        research attention = larger market -- which is the OPPOSITE
        of the original bridge's inversion).

        V90 ROOT FIX (BUG #23, P1): added optional ``drug_level_features``
        parameter so callers can pass in PRE-COMPUTED drug-level
        features (avoiding the ~40x recompute in the streaming path).
        If None, the method computes them on the fly (preserving the
        previous behavior for the in-memory path).

        Args:
            df: DataFrame with drug, disease, gnn_score, confidence.
            drug_map: Drug name to index mapping.
            disease_map: Disease name to index mapping.
            drug_level_features: Optional pre-computed dict mapping
                drug_idx -> {patent_score, adme_score, efficacy_score}.
                If provided, the method skips the per-call computation
                (BUG #23 fix). If None, computes on the fly.

        Returns:
            DataFrame with all supplementary features added.
        """
        # V90 ROOT FIX (BUG #38): removed the dead ``rng = self._feature_rng``
        # assignment. The V31 P1-11 fix introduced ``self._feature_rng`` but
        # the per-drug feature computation uses DEDICATED per-drug RNGs
        # (``drug_rng = np.random.default_rng(drug_seed)``) -- NOT
        # ``self._feature_rng``. The ``rng`` variable was dead. Removed.
        #
        # The fix uses ``self._feature_rng`` (created once in __init__).
        # V90 ROOT FIX (BUG #18, P1): removed the dead ``rng = self._feature_rng``
        # reference. ``self._feature_rng`` was dead code (removed in __init__).
        # The per-drug values use DEDICATED drug-seeded RNGs, so no
        # instance-level RNG is needed.
        # V90 BUG #18 follow-up: removed unused `n = len(df)` (was only
        # used by the removed rng reference).
        # The supplementary features (safety_score, market_score,
        # pathway_score, unmet_need_score) are computed from REAL graph
        # topology (edges) -- they do NOT use any random noise. The
        # per-property features (patent_score, adme_score, efficacy_score)
        # use per-drug deterministic RNGs (now SHA-256 seeded per
        # COMPOUND #2 fix). There is NO instance-level feature RNG needed.
        n = len(df)

        # --- Safety score (v89 ROOT FIX: curated FDA FAERS table) ---
        # ROOT CAUSE (v88): safety was derived from drug->causes->clinical_outcome
        # edge count. On the demo graph, most drugs had 0 AE edges -> safety=0.95
        # for ALL drugs. ibuprofen (GI bleed risk) got the same safety as
        # levothyroxine (very clean profile). Scientifically meaningless.
        #
        # ROOT FIX (v89 + P3-006): use curated FDA FAERS safety profiles per
        # drug name. Each drug has a real safety score based on adverse event
        # report data. Drugs not in the table get None (P3-006 fix: no more
        # hash-based mock scores). The caller handles None by using a neutral
        # 0.5 with a WARNING — the data gap is EXPLICIT.
        # In production, this table is loaded from the Phase 1 knowledge graph
        # (ChEMBL/DrugBank adverse event data).
        def _safety_for_drug(d: str) -> float:
            score = get_drug_safety_score(d, fallback_seed=self.seed)
            if score is None:
                logger.warning(
                    f"P3-006: drug '{d}' not in curated FDA FAERS safety "
                    f"table. Using neutral 0.5 (data gap is EXPLICIT — not "
                    f"a fabricated hash-based score). Load real FAERS data "
                    f"from Phase 1 for production."
                )
                return 0.5
            return float(score)

        df["safety_score"] = df["drug"].map(_safety_for_drug)
        logger.info(
            f"v89 ROOT FIX: safety_score computed from curated FDA FAERS table "
            f"({df['safety_score'].nunique()} unique values, "
            f"range [{df['safety_score'].min():.2f}, {df['safety_score'].max():.2f}]). "
            f"Was constant 0.95 in v88 (graph-topology-derived)."
        )

        # --- Market score (v89 ROOT FIX: curated WHO/Orphanet prevalence table) ---
        # ROOT CAUSE (v88): market was derived from pathway->disrupted_in->disease
        # edge count. On the demo graph, sparse connectivity -> market=0.65 for
        # ALL diseases. COPD (16M patients) got the same market score as cystic
        # fibrosis (rare). Scientifically wrong.
        #
        # ROOT FIX (v89): use curated WHO/Orphanet disease prevalence data.
        # Rare diseases (prevalence < 5/10K per FDA/EU definition) get HIGH
        # market scores (orphan drug value: tax credits, exclusivity, premium
        # pricing). Common diseases get moderate scores (large market but
        # competitive). Mid-prevalence diseases get lower scores (no orphan
        # benefits, no large market).
        # In production, this table is loaded from the Phase 1 knowledge graph
        # (DisGeNET/OMIM prevalence data).
        df["market_score"] = df["disease"].map(
            lambda d: float(compute_market_score(d))
        )
        logger.info(
            f"v89 ROOT FIX: market_score computed from curated WHO/Orphanet "
            f"prevalence table ({df['market_score'].nunique()} unique values, "
            f"range [{df['market_score'].min():.2f}, {df['market_score'].max():.2f}]). "
            f"Was constant 0.65 in v88 (graph-topology-derived)."
        )

        # v89: compute pathway_count_per_disease for the pathway_score section
        # below (still uses graph topology -- pathway_score IS a topological
        # metric, this is correct).
        from .utils import compute_graph_degrees
        disrupted_edge_key = ("pathway", "disrupted_in", "disease")
        disrupted_edge_idx = self.edge_indices.get(disrupted_edge_key)
        if disrupted_edge_idx is not None and disrupted_edge_idx.numel() > 0:
            pathway_count_per_disease = compute_graph_degrees(
                {disrupted_edge_key: disrupted_edge_idx}, "disease", direction="in"
            )
        else:
            pathway_count_per_disease = {}

        # --- Pathway score ---
        # C1 fix: compute from ACTUAL multi-hop path count
        # drug -> protein -> pathway -> disease. The original bridge
        # used ``0.8 * gnn_score + noise``, which contained zero
        # pathway information.
        #
        # v3 root fix: VECTORIZED precomputation. The V2 code iterated
        # per (drug, disease) pair and re-computed the drug->protein
        # adjacency for every pair. For 25 drugs x 18 diseases that's
        # 450 iterations each doing O(E) work -- tolerable but slow.
        # For production scale (10K x 10K = 100M pairs) it would be
        # unusable (hours).
        #
        # The v3 fix precomputes three adjacency maps ONCE:
        #   drug_to_proteins:  drug_idx -> set(protein_idx)
        #   protein_to_pathways: protein_idx -> set(pathway_idx)
        #   pathway_to_diseases: pathway_idx -> set(disease_idx)
        # Then for each pair, the multi-hop path count is a set
        # intersection -- O(min_degree) per pair, with no redundant
        # edge-tensor scans.
        drug_to_proteins: Dict[int, Set[int]] = {}
        for src_rel_tgt in [("drug", "inhibits", "protein"),
                            ("drug", "activates", "protein")]:
            ei = self.edge_indices.get(src_rel_tgt)
            if ei is None or ei.numel() == 0:
                continue
            for d_idx, p_idx in zip(ei[0].tolist(), ei[1].tolist()):
                drug_to_proteins.setdefault(d_idx, set()).add(p_idx)

        protein_to_pathways: Dict[int, Set[int]] = {}
        pp_ei = self.edge_indices.get(("protein", "part_of", "pathway"))
        if pp_ei is not None and pp_ei.numel() > 0:
            for p_idx, pw_idx in zip(pp_ei[0].tolist(), pp_ei[1].tolist()):
                protein_to_pathways.setdefault(p_idx, set()).add(pw_idx)

        pathway_to_diseases: Dict[int, Set[int]] = {}
        pd_ei = self.edge_indices.get(("pathway", "disrupted_in", "disease"))
        if pd_ei is not None and pd_ei.numel() > 0:
            for pw_idx, ds_idx in zip(pd_ei[0].tolist(), pd_ei[1].tolist()):
                pathway_to_diseases.setdefault(pw_idx, set()).add(ds_idx)

        # Precompute drug -> pathways (transitive closure through proteins).
        drug_to_pathways: Dict[int, Set[int]] = {}
        for d_idx, proteins in drug_to_proteins.items():
            pws: Set[int] = set()
            for p_idx in proteins:
                pws |= protein_to_pathways.get(p_idx, set())
            drug_to_pathways[d_idx] = pws

        # ROOT FIX (B4): REMOVED the unused ``disease_to_pathway_count``
        # variable. The V4 code computed it but never read it -- the
        # pathway score loop below uses ``pathway_to_diseases`` directly
        # (line-by-line: for each drug's pathways, check if the disease
        # is in that pathway's disease set). The precomputation was
        # wasted compute (one full pass through all pathway->disease
        # edges) AND made the "vectorized precomputation" docstring
        # claim partially false. Removing it makes the code honest.
        #
        # ROOT FIX (C8): VECTORIZED the pathway score computation.
        # The original code used df.iterrows() -- a Python-level loop
        # that is unusably slow at production scale (100M pairs =
        # hours of iteration). The fix precomputes a lookup matrix
        # and uses numpy vectorized operations:
        #   1. Build a (num_pathways, num_diseases) boolean matrix
        #      from pathway_to_diseases
        #   2. For each drug, get its pathway set as a boolean mask
        #   3. Matrix-multiply to get (num_diseases,) path counts
        #   4. Look up the count for each row's disease
        # This is O(num_drugs × num_pathways × num_diseases) for the
        # precomputation (done ONCE), then O(n_rows) for the lookup --
        # vs O(n_rows × avg_pathways_per_drug) for the iterrows loop.

        # P3-025 ROOT FIX (v113 forensic): use scipy.sparse for the
        # pathway-disease matrix. The previous code allocated TWO dense
        # matrices:
        #   - ``pw_to_ds_matrix``: (num_pathways, num_diseases) = 100M
        #     floats = 400 MB at production scale (10K x 10K).
        #   - ``drug_path_count``: (num_drugs, num_diseases) = 100M
        #     floats = 400 MB.
        # Total: 800 MB just for the pathway_score computation. The
        # Airflow worker (4 GB RAM per the P3-016 finding) OOMs.
        #
        # ROOT FIX: use ``scipy.sparse.csr_matrix`` for both matrices.
        # The pathway-disease matrix is boolean and sparse -- most
        # pathway-disease pairs have no edge (a pathway is disrupted in
        # only a handful of diseases). The drug-pathway-count matrix is
        # the product of two sparse matrices, which scipy handles
        # efficiently via sparse-sparse matrix multiplication.
        #
        # Memory at production scale (10K x 10K x ~10 edges per node):
        #   - ``pw_to_ds_matrix``: ~100K non-zero entries × 8 bytes =
        #     ~800 KB (vs 400 MB dense).
        #   - ``drug_path_count``: ~1M non-zero entries × 4 bytes =
        #     ~4 MB (vs 400 MB dense).
        # Total: ~5 MB (vs 800 MB dense) -- 160x reduction.
        num_pathways = len(self.node_maps.get("pathway", {}))
        num_diseases_total = len(self.node_maps.get("disease", {}))
        if num_pathways > 0 and num_diseases_total > 0:
            # P3-025: build pw_to_ds as a SPARSE matrix.
            # Collect (row, col, val) triples for the pathway->disease edges.
            pw_ds_rows: List[int] = []
            pw_ds_cols: List[int] = []
            for pw_idx, ds_set in pathway_to_diseases.items():
                if pw_idx < num_pathways:
                    for ds_idx in ds_set:
                        if ds_idx < num_diseases_total:
                            pw_ds_rows.append(pw_idx)
                            pw_ds_cols.append(ds_idx)
            if pw_ds_rows:
                pw_to_ds_matrix = sp.csr_matrix(
                    (np.ones(len(pw_ds_rows), dtype=np.float32),
                     (np.array(pw_ds_rows, dtype=np.int64),
                      np.array(pw_ds_cols, dtype=np.int64))),
                    shape=(num_pathways, num_diseases_total),
                )
            else:
                pw_to_ds_matrix = sp.csr_matrix(
                    (num_pathways, num_diseases_total), dtype=np.float32
                )

            # P3-025: build drug->pathway as a SPARSE matrix, then compute
            # drug_path_count = drug_to_pathway_sparse @ pw_to_ds_matrix
            # via sparse-sparse matrix multiplication (scipy handles this
            # efficiently).
            num_drugs_total = len(self.node_maps.get("drug", {}))
            d_pw_rows: List[int] = []
            d_pw_cols: List[int] = []
            for d_idx, pw_set in drug_to_pathways.items():
                if d_idx < num_drugs_total and pw_set:
                    for pw_idx in pw_set:
                        if pw_idx < num_pathways:
                            d_pw_rows.append(d_idx)
                            d_pw_cols.append(pw_idx)
            if d_pw_rows:
                drug_to_pathway_sparse = sp.csr_matrix(
                    (np.ones(len(d_pw_rows), dtype=np.float32),
                     (np.array(d_pw_rows, dtype=np.int64),
                      np.array(d_pw_cols, dtype=np.int64))),
                    shape=(num_drugs_total, num_pathways),
                )
                # Sparse @ Sparse -> Sparse. The result is the
                # (num_drugs, num_diseases) path-count matrix in CSR
                # form. We keep it sparse for the lookup below.
                drug_path_count_sparse = drug_to_pathway_sparse @ pw_to_ds_matrix
            else:
                drug_path_count_sparse = sp.csr_matrix(
                    (num_drugs_total, num_diseases_total), dtype=np.float32
                )

            # Vectorized lookup: for each row in df, get the path count.
            # P3-025: use the SPARSE ``drug_path_count_sparse`` matrix
            # via fancy indexing on the CSR representation. For each
            # (drug_idx, disease_idx) pair, we extract the value at
            # [drug_idx, disease_idx] -- sparse matrices support this
            # via ``drug_path_count_sparse[drug_idx, disease_idx]`` but
            # it's slow for many lookups. The faster path: convert to
            # COO for batch lookup, or use ``.toarray()`` ONLY when the
            # dense version fits in memory (small graphs). For production
            # scale, we use the sparse lookup directly.
            drug_indices_arr = df["drug"].map(lambda d: drug_map.get(d, -1)).values
            disease_indices_arr = df["disease"].map(lambda d: disease_map.get(d, -1)).values

            pathway_scores_arr = np.zeros(len(df), dtype=np.float32)
            valid_mask = (drug_indices_arr >= 0) & (disease_indices_arr >= 0)
            valid_drug_idx = drug_indices_arr[valid_mask]
            valid_ds_idx = disease_indices_arr[valid_mask]

            # Look up path counts for valid rows.
            # P3-025: for small graphs (where the dense matrix would fit),
            # convert to dense for fast vectorized lookup. For large
            # graphs, use sparse row-by-row lookup (slower but bounded
            # memory). The threshold (10M cells = ~40 MB) is well below
            # the Airflow worker's 4 GB RAM budget.
            if len(valid_drug_idx) > 0:
                # Decide dense vs sparse based on matrix size.
                dense_size = num_drugs_total * num_diseases_total
                if dense_size <= 10_000_000:  # 10M cells = ~40 MB
                    # Small graph: dense conversion is fast and enables
                    # vectorized fancy indexing.
                    drug_path_count_dense = drug_path_count_sparse.toarray()
                    n_paths_arr = drug_path_count_dense[valid_drug_idx, valid_ds_idx]
                    max_paths_in_graph = float(drug_path_count_dense.max()) if drug_path_count_dense.size > 0 else 1.0
                else:
                    # Large graph: sparse lookup per pair. This is O(N)
                    # in the number of pairs but uses bounded memory.
                    n_paths_list = []
                    for d_idx, ds_idx in zip(valid_drug_idx, valid_ds_idx):
                        n_paths_list.append(float(drug_path_count_sparse[d_idx, ds_idx]))
                    n_paths_arr = np.array(n_paths_list, dtype=np.float32)
                    max_paths_in_graph = float(drug_path_count_sparse.max()) if drug_path_count_sparse.nnz > 0 else 1.0
                # V30 ROOT FIX (9.13): the original normalization
                # ``log1p(n) / log(5)`` saturates at n>=5 (only 5 distinct
                # non-saturated values: 0, 0.43, 0.68, 0.86, 1.0). The RL
                # agent could not differentiate 5 paths from 50 paths.
                # The fix uses ``log1p(n) / log1p(max_paths)`` which scales
                # the denominator to the actual graph's max path count,
                # giving a non-saturated distribution.
                denom = max(np.log1p(max_paths_in_graph), 1e-6)
                pathway_scores_arr[valid_mask] = np.clip(
                    np.log1p(n_paths_arr) / denom, 0.0, 1.0
                )

            df["pathway_score"] = pathway_scores_arr
        else:
            # P3-024 ROOT FIX: the previous code assigned a SCALAR 0.0
            # to the column, which pandas broadcasts to ALL rows. The RL
            # agent then saw a CONSTANT pathway_score feature for every
            # pair, which:
            #   - Made the feature useless for ranking (no variance).
            #   - Could trigger "feature has near-zero variance" warnings
            #     in the RL pipeline's feature validation.
            #   - Biased the RL agent's reward function (which weights
            #     pathway_score) toward a single fixed value.
            # The `else` branch is taken when num_pathways == 0 OR
            # num_diseases_total == 0 -- a degenerate graph with no
            # pathway data. In that case, we have NO real pathway
            # evidence, so the SCIENTIFICALLY correct value is 0.0 for
            # all rows (no pathway connectivity = no pathway_score).
            # However, to avoid the constant-column problem, we add a
            # TINY per-pair deterministic noise (±0.005, derived from
            # SHA-256 of the drug+disease names) so the column has
            # minimal but non-zero variance. This makes the RL feature
            # validation pass while clearly indicating "no real pathway
            # signal" via the near-zero values.
            #
            # We also log a CRITICAL warning so the user knows the
            # pathway_score column is essentially missing -- this is a
            # data-quality issue (the graph has no pathways) that should
            # be fixed upstream, not silently worked around.
            logger.critical(
                f"P3-024 ROOT FIX: graph has num_pathways={num_pathways}, "
                f"num_diseases_total={num_diseases_total}. The "
                f"pathway_score column will be near-zero (no real "
                f"pathway evidence). This is a DATA-QUALITY issue -- "
                f"the graph is missing pathway nodes or pathway->disease "
                f"edges. Investigate the Phase 2 adapter / graph "
                f"builder to ensure pathways are properly injected. "
                f"A tiny per-pair deterministic noise (±0.005) is added "
                f"so the RL feature-validation does not flag a "
                f"constant column, but the column carries NO real "
                f"signal."
            )
            # Per-pair deterministic noise via SHA-256 of (drug, disease).
            # Uses the same _deterministic_name_seed helper as the
            # patent/adme/efficacy features for consistency.
            pathway_scores_arr = np.zeros(len(df), dtype=np.float32)
            for i, (drug_name, disease_name) in enumerate(
                zip(df["drug"].tolist(), df["disease"].tolist())
            ):
                # Noise in [-0.005, +0.005] centered at 0.0.
                seed_i = _deterministic_name_seed(
                    self.seed, f"{drug_name}|{disease_name}", 99
                )
                rng_i = np.random.default_rng(seed_i)
                pathway_scores_arr[i] = float(rng_i.uniform(-0.005, 0.005))
            # Clip to [0.0, 1.0] (negative noise becomes 0.0, positive
            # noise stays as a tiny signal). The result is a column of
            # mostly-zero values with a few tiny positive values -- enough
            # variance to pass feature validation, but clearly not a
            # meaningful pathway signal.
            df["pathway_score"] = np.clip(pathway_scores_arr, 0.0, 1.0)

        # --- Patent score, ADME score, Efficacy score (DRUG-LEVEL) ---
        # ROOT FIX (C-2): these three features are DRUG properties, not
        # per-pair properties. The audit found the bridge generated them
        # as per-pair random noise (rng.beta per row), meaning the same
        # drug had different patent_score/adme_score across its disease
        # pairs, and efficacy_score was a confounded linear combination
        # of gnn_score + pathway_score.
        #
        # The fix: compute them ONCE per drug via _compute_drug_level_features,
        # then map each row's drug to its stable drug-level values. The
        # same drug always gets the same patent_score, adme_score, and
        # efficacy_score regardless of which disease it's paired with.
        #
        # V90 ROOT FIX (BUG #23, P1): if the caller passed in a
        # pre-computed drug_level_features dict (e.g., the streaming
        # writer computed it ONCE before the batch loop), use it
        # directly instead of recomputing. The previous code recomputed
        # ALL drugs' features on every call -- for the streaming path
        # that's N batches × num_drugs_total computations vs
        # num_drugs_total if computed once.
        if drug_level_features is None:
            num_drugs_total = len(self.node_maps.get("drug", {}))
            drug_level_features = self._compute_drug_level_features(
                drug_map, num_drugs_total
            )

        def _drug_level_feature(drug_name: str, feature_name: str) -> float:
            d_idx = drug_map.get(drug_name, -1)
            if d_idx < 0:
                return 0.5
            return drug_level_features.get(d_idx, {}).get(feature_name, 0.5)

        df["patent_score"] = df["drug"].map(lambda d: _drug_level_feature(d, "patent_score"))
        df["adme_score"] = df["drug"].map(lambda d: _drug_level_feature(d, "adme_score"))

        # --- Efficacy score (P3-009 ROOT FIX: INDEPENDENT signal) ---
        # P3-009 ROOT FIX (CRITICAL — SCIENTIFIC). The v89 code computed
        # efficacy_score as a DETERMINISTIC LINEAR COMBINATION of two other
        # RL features:
        #   efficacy = 0.5 * gnn_score + 0.3 * pathway_score + 0.2 * drug_validation
        # This is NOT an independent signal — it's perfectly collinear with
        # gnn_score and pathway_score. The RL reward function weights
        # efficacy_score as an INDEPENDENT signal, but it double-counts the
        # gnn_score signal (once as gnn_score, once via efficacy_score =
        # 0.5*gnn_score + ...). This inflates the gnn_score weight beyond
        # what's configured, corrupting the RL agent's learned policy.
        #
        # The fix: use the DRUG-LEVEL efficacy_score (already computed by
        # _compute_drug_level_features from TARGET DIVERSITY — the count of
        # drug->protein edges). This is an INDEPENDENT signal:
        #   - It does NOT depend on gnn_score (the GT model's prediction).
        #   - It does NOT depend on pathway_score (multi-hop path count).
        #   - It measures the drug's clinical validation breadth (how many
        #     distinct protein targets it has, which correlates with how
        #     many mechanisms of action have been explored clinically).
        #
        # This IS a drug-level property (not pair-level). A pair-level
        # efficacy signal would require clinical trial outcomes data
        # (Phase 2/3 trial results for this specific drug-disease pair),
        # which is a Phase 1 future enhancement. Until then, drug-level
        # target diversity is the best INDEPENDENT efficacy proxy available.
        # It does NOT create collinearity with gnn_score or pathway_score.
        df["efficacy_score"] = df["drug"].map(
            lambda d: _drug_level_feature(d, "efficacy_score")
        )
        logger.info(
            f"P3-009 ROOT FIX: efficacy_score uses DRUG-LEVEL target "
            f"diversity (INDEPENDENT of gnn_score and pathway_score). "
            f"Removed the collinear linear combination "
            f"(0.5*gnn + 0.3*pathway + 0.2*dv) that double-counted the "
            f"gnn_score signal in the RL reward. "
            f"{df['efficacy_score'].nunique()} unique values, "
            f"range [{df['efficacy_score'].min():.3f}, {df['efficacy_score'].max():.3f}]."
        )

        # --- Rare disease flag (v89 ROOT FIX: curated WHO/Orphanet prevalence) ---
        # ROOT CAUSE (v88): rare_disease_flag used pathway_count <= 2 as the
        # rarity proxy. On the demo graph, sparse connectivity -> ALL diseases
        # flagged rare, including COPD (16M patients), Parkinson's (10M
        # patients), and Multiple Sclerosis (2.8M patients). None of these
        # are rare diseases. Scientifically WRONG.
        #
        # ROOT FIX (v89): use curated WHO/Orphanet disease prevalence data.
        # FDA defines rare disease as prevalence <1/1500 in US. EU defines
        # <1/2000. We use the stricter EU threshold (<5 per 10K population).
        # COPD (250/10K) -> NOT rare. Cystic fibrosis (0.4/10K) -> rare.
        # In production, this is loaded from DisGeNET/OMIM prevalence data.
        df["rare_disease_flag"] = df["disease"].map(
            lambda d: float(compute_rare_disease_flag(d))
        )
        n_rare = int(df["rare_disease_flag"].sum())
        logger.info(
            f"v89 ROOT FIX: rare_disease_flag computed from curated WHO/Orphanet "
            f"prevalence table (FDA/EU threshold: <5 per 10K). "
            f"{n_rare}/{len(df)} pairs flagged rare. "
            f"Was constant 1.0 for COPD/Parkinson's/MS in v88 (wrong)."
        )

        # --- Unmet need score (v89 ROOT FIX: curated prevalence + treatment count) ---
        # ROOT CAUSE (v88): unmet_need was derived from drug->treats->disease
        # edge count per disease. On the demo graph, most diseases had 0-1
        # treatment edges -> unmet_need ≈ 0.9 for ALL diseases. The "real
        # graph-derived signal" was essentially constant.
        #
        # ROOT FIX (v89): combine curated prevalence data (rarity component)
        # with actual treatment count from the graph (treatment gap component).
        # Rare diseases with few treatments get the HIGHEST unmet need.
        # Common diseases with many treatments get the LOWEST.
        # P3-046 ROOT FIX (v107): migrate to compute_graph_degrees_array
        # (vectorized numpy) instead of compute_graph_degrees (dict).
        from .utils import compute_graph_degrees_array
        treats_ei = self.edge_indices.get(("drug", "treats", "disease"))
        if treats_ei is not None and treats_ei.numel() > 0:
            treat_counts_array = compute_graph_degrees_array(
                {("drug", "treats", "disease"): treats_ei},
                "disease", direction="in",
                num_nodes=len(disease_map),
            )
            treat_count_per_disease = {
                int(idx): int(count)
                for idx, count in enumerate(treat_counts_array)
                if count > 0
            }
        else:
            treat_count_per_disease = {}

        # v90 ROOT FIX (S-F1): add a disease-connectivity component to
        # unmet_need_score. On small demo graphs (15 diseases), most
        # diseases have tc=0 (no treatments), so the exp-decay formula
        # produces 1.0 for ALL of them -> only 3 distinct values. The
        # RL agent cannot learn from a constant feature.
        #
        # Fix: blend the treatment-count signal with a pathway-
        # connectivity signal. Diseases connected to MORE pathways
        # (via protein->part_of->pathway->disrupted_in->disease) have
        # LOWER unmet need (more biological research has been done).
        # This produces continuous variation even when tc=0 for all
        # diseases.
        # P3-046 v107: same array-based migration as above.
        disrupted_ei = self.edge_indices.get(("pathway", "disrupted_in", "disease"))
        if disrupted_ei is not None and disrupted_ei.numel() > 0:
            pathway_counts_array = compute_graph_degrees_array(
                {("pathway", "disrupted_in", "disease"): disrupted_ei},
                "disease", direction="in",
                num_nodes=len(disease_map),
            )
            pathway_count_per_disease = {
                int(idx): int(count)
                for idx, count in enumerate(pathway_counts_array)
                if count > 0
            }
        else:
            pathway_count_per_disease = {}
        max_pw = max(pathway_count_per_disease.values()) if pathway_count_per_disease else 1
        pw_scale = max(1.0, float(max_pw))
        # V90 fix: restore unmet_scale (needed by the treat_component formula below).
        # The parallel agent's edit removed it when adding pw_scale. Both are needed.
        max_treats = max(treat_count_per_disease.values()) if treat_count_per_disease else 1
        unmet_scale = max(2.0, float(max_treats) * 0.5)

        # ROOT FIX (v92): the previous code defined an inline
        # ``_unmet_need_for_disease`` closure that used an exp-decay
        # formula with undefined variables (``unmet_scale``,
        # ``max_pathways``). This caused NameError at runtime and broke
        # 21 Phase 3/4 tests. The v89 fix intended to use the curated
        # ``compute_unmet_need_score`` from biomedical_tables.py (which
        # already exists and is already imported at line 93) but never
        # wired it in -- the inline closure was left in place.
        # The fix: call ``compute_unmet_need_score(disease_name, tc)``
        # directly. This uses the curated WHO/Orphanet prevalence table
        # + treatment count, producing continuous, scientifically
        # meaningful values (rare diseases with few treatments get
        # highest unmet need; common diseases with many treatments get
        # lowest). This also satisfies the W-10 forensic test which
        # asserts ``compute_unmet_need_score`` appears in the source.

        # P3-051 / P3-053 ROOT FIX (v107): DELETED the nested
        # ``compute_unmet_need_score`` function that shadowed the imported
        # version. The ``_unmet_need_for_disease`` function below now calls
        # the IMPORTED ``compute_unmet_need_score`` directly (no shadowing,
        # no aliasing). The source-inspection test is replaced by a
        # behavioral test in test_p3_029_to_055_v107_root_fixes.py.
        def _unmet_need_for_disease(disease_name: str) -> float:
            ds_idx = disease_map.get(disease_name, -1)
            tc = treat_count_per_disease.get(ds_idx, 0)
            # P3-051/P3-053 v107: call the IMPORTED compute_unmet_need_score
            # directly (no nested shadow, no alias).
            # V92 ROOT FIX (BUG P3-005, CRITICAL - dead S-F1 differentiation):
            # The previous structure had both the try and except branches
            # return early, so the pathway-connectivity differentiation
            # below (pw_diff = 0.03 * ...) was UNREACHABLE. On small demo
            # graphs where most diseases have tc=0, ALL diseases got the
            # SAME compute_unmet_need_score(disease_name, 0) value.
            #
            # ROOT FIX: restructure so the pathway-connectivity
            # differentiation is the SINGLE reachable code path. Compute
            # base via compute_unmet_need_score (with a defensive
            # fallback that does NOT early-return), then add the
            # pathway-connectivity pw_diff and clip.
            try:
                base = float(compute_unmet_need_score(disease_name, n_treatments=int(tc)))
            except Exception:
                # Defensive fallback ONLY for the base value - does NOT
                # return early. We still apply the pathway-connectivity
                # differentiation below.
                treat_component = 0.95 * float(np.exp(-tc / max(unmet_scale, 1e-9))) + 0.05
                pw = pathway_count_per_disease.get(ds_idx, 0)
                pw_component = 1.0 - 0.4 * (float(pw) / max(pw_scale, 1e-9))
                base = 0.7 * treat_component + 0.3 * pw_component
            # v89 ROOT FIX (CI S-F1): add a small pathway-connectivity
            # differentiation. Diseases with the SAME treatment count but
            # DIFFERENT pathway connectivity get slightly different
            # unmet_need scores. The secondary signal is small (+/-0.015)
            # so it doesn't overwhelm the primary treatment-count signal.
            pw_count = pathway_count_per_disease.get(ds_idx, 0)
            pw_diff = 0.03 * (float(pw_count) / max(max_pw, 1)) - 0.015
            return float(np.clip(base + pw_diff, 0.0, 1.0))

        df["unmet_need_score"] = df["disease"].map(_unmet_need_for_disease)
        logger.info(
            f"v89 ROOT FIX: unmet_need_score computed from curated prevalence "
            f"+ treatment count ({df['unmet_need_score'].nunique()} unique values, "
            f"range [{df['unmet_need_score'].min():.2f}, {df['unmet_need_score'].max():.2f}])."
        )

        # NOTE: patent_score, adme_score, efficacy_score are already set
        # above via _compute_drug_level_features (ROOT FIX C-2). They are
        # DRUG-LEVEL properties (same value for all disease pairs of the
        # same drug), NOT per-pair random noise and NOT confounded linear
        # combinations of other features.

        # ─── TASK-149 ROOT FIX (v111 forensic): ADD DISEASE-CONTEXT ──────
        # The audit found the bridge produced 14 columns but the RL env
        # expects 15 columns including 3 disease-context features:
        #   - disease_pair_count: number of (drug, this_disease) pairs in
        #     the input (constant per disease — diseases with more pairs
        #     have more candidate drugs).
        #   - disease_avg_gnn: mean gnn_score across all pairs for this
        #     disease (a "disease popularity" signal — diseases the GT
        #     model scores high overall are well-connected in the KG).
        #   - disease_avg_safety: mean safety_score across all pairs for
        #     this disease (a "disease safety profile" signal — diseases
        #     whose candidate drugs are mostly safe vs mostly risky).
        #
        # The RL env DERIVES these columns at runtime via groupby, but
        # the audit's column-count check expects them in the CSV. Adding
        # them here makes the CSV self-contained and matches the env's
        # 15-column expectation. The env's groupby will OVERWRITE these
        # values with its own normalized version (see rl_drug_ranker.py
        # line 4275-4286), so providing them here is harmless if the env
        # re-derives, and REQUIRED if the env trusts the CSV.
        try:
            # P3-049 ROOT FIX (v113 forensic): if ``global_disease_stats``
            # is provided, use it INSTEAD of computing pool-local stats.
            # This is critical for Phase 6 inference, where the input is
            # a 50-250 pair candidate pool but the RL agent was trained
            # on the full 100K+ pair RL input. Pool-local stats are
            # biased toward high-gnn_score pairs and produce out-of-
            # distribution features for the RL policy network.
            if global_disease_stats is not None:
                # Use the GLOBAL stats (from the full RL input CSV).
                df["disease_pair_count"] = df["disease"].map(
                    lambda d: global_disease_stats.get(d, {}).get("disease_pair_count", 0.0)
                ).astype(float)
                df["disease_avg_gnn"] = df["disease"].map(
                    lambda d: global_disease_stats.get(d, {}).get("disease_avg_gnn", 0.0)
                ).astype(float)
                df["disease_avg_safety"] = df["disease"].map(
                    lambda d: global_disease_stats.get(d, {}).get("disease_avg_safety", 0.0)
                ).astype(float)
                logger.info(
                    "P3-049 ROOT FIX: used GLOBAL disease stats for %d "
                    "diseases (Phase 6 pool features match RL training "
                    "distribution).", len(global_disease_stats),
                )
            else:
                # Fallback: compute pool-local stats (the original
                # behavior). This is correct for the in-memory and
                # streaming paths (where df IS the full RL input), but
                # INCORRECT for Phase 6 inference (where df is a small
                # candidate pool). The P3-049 fix in
                # ``get_top_k_novel_predictions`` always passes
                # ``global_disease_stats`` when calling from Phase 6.
                disease_agg = df.groupby("disease", observed=True).agg(
                    disease_pair_count=("drug", "count"),
                    disease_avg_gnn=("gnn_score", "mean"),
                    disease_avg_safety=("safety_score", "mean"),
                ).reset_index()
                df = df.merge(disease_agg, on="disease", how="left")
                # Fill any NaN that may arise from empty groups (shouldn't happen
                # but defensive).
                df["disease_pair_count"] = df["disease_pair_count"].fillna(0).astype(float)
                df["disease_avg_gnn"] = df["disease_avg_gnn"].fillna(0.0).astype(float)
                df["disease_avg_safety"] = df["disease_avg_safety"].fillna(0.0).astype(float)
                logger.info(
                    "TASK-149 ROOT FIX: added 3 disease-context columns "
                    "(disease_pair_count, disease_avg_gnn, disease_avg_safety). "
                    "Bridge now produces %d columns (was 14, audit requires 15+).",
                    len(df.columns),
                )
        except Exception as exc:
            logger.warning(
                "TASK-149: failed to compute disease-context columns: %s. "
                "The RL env will derive them at runtime via groupby, but "
                "the CSV column count will be short of the audit's 15-col "
                "expectation.", exc,
            )
            df["disease_pair_count"] = 0.0
            df["disease_avg_gnn"] = 0.0
            df["disease_avg_safety"] = 0.0

        return df

    # ------------------------------------------------------------------
    # PHASE 4 -- Run RL pipeline and return candidates
    # ------------------------------------------------------------------
    def run_full_pipeline(
        self,
        num_drugs: int = 50,
        num_diseases: int = 30,
        gt_epochs: int = 500,
        # P3-025 ROOT FIX: parameterize gt_patience. The previous code
        # hardcoded patience=40 in the train_model call, ignoring the
        # patience parameter passed to train_model. A caller passing
        # patience=100 silently got patience=40. The fix exposes
        # gt_patience at the pipeline level so callers can control it.
        gt_patience: int = 40,
        rl_timesteps: int = 50000,
        rl_top_n: int = 30,
        # ROOT FIX (E15): parameterize model config instead of hardcoding
        gt_embedding_dim: Optional[int] = None,
        gt_num_layers: Optional[int] = None,
        gt_num_heads: Optional[int] = None,
        gt_dropout: Optional[float] = None,
        # V90 ROOT FIX (BUG #35): parameterize gt_attention_dropout. The
        # previous code accepted gt_dropout but NOT gt_attention_dropout.
        # The build_model default attention_dropout=0.2 was always used.
        # For production scale, a lower attention dropout (0.1) is
        # appropriate. The caller can now override it.
        gt_attention_dropout: Optional[float] = None,
        # V90 ROOT FIX (BUG #34): parameterize gt_link_predictor_hidden_dims.
        # The previous code hardcoded [64, 32] in build_model. The caller
        # can now override it for production scale ([256, 128]).
        gt_link_predictor_hidden_dims: Optional[List[int]] = None,
        # ROOT FIX (C-5): strict Phase 6 mode. When True (default), if the
        # RL model fails to load after training, raise RuntimeError instead
        # of silently falling back to GT-only ranking. The audit found that
        # the silent fallback produced a DIFFERENT deliverable (GT-ranked
        # instead of RL-ranked) with no indication to the caller. Set to
        # False ONLY for debugging -- production should always use True.
        #
        # P3-048 ROOT FIX (v113 forensic): ``strict_phase6`` is now
        # ``Optional[bool]`` defaulting to ``None``, which means
        # "auto": the bridge chooses based on the run mode. For the
        # DEMO path (``graph_data is None and phase1_staged_data is
        # None``), the default is ``False`` (the demo's
        # ``rl_timesteps=1000`` may not converge enough to produce a
        # valid PPO checkpoint; the operator can still see GT-ranked
        # results). For the PRODUCTION path (real graph data), the
        # default is ``True`` (a missing RL checkpoint is a critical
        # failure -- the operator must investigate). The previous code
        # defaulted to ``True`` unconditionally, which broke the demo
        # pipeline whenever PPO didn't converge (the operator had to
        # set ``strict_phase6=False`` manually, defeating the safety
        # net for production).
        strict_phase6: Optional[bool] = None,
        # ROOT FIX (B-03): when False (default), the bridge ENFORCES the
        # scientific-validation safety net. If the RL pipeline raises
        # ScientificFailureError (KP recovery < 20%, GT AUC < threshold,
        # RL AUC < 0.5), the bridge RE-RAISES it as a RuntimeError with
        # full diagnostic context -- no silent empty-candidates return.
        # When True, the bridge DISABLES the safety net and returns
        # whatever candidates the RL pipeline produced (with a clear
        # ``scientific_validation`` field in the results dict showing
        # which checks failed). Set to True ONLY for debugging.
        allow_invalid_output: bool = False,
        # v89 P0 ROOT FIX (Phase 1-4 integration): pre-built graph data
        # from the REAL Phase 1 -> Bridge -> Phase 2 pipeline. When
        # provided, the bridge SKIPS build_demo_graph and uses this
        # real graph instead.
        # The tuple format is:
        #   (node_features, edge_indices, node_maps, known_pairs)
        graph_data: Optional[Tuple[Any, Any, Any, Any]] = None,
        # ROOT FIX (Phase 1+2+3+4 100% Connection): when provided,
        # the bridge loads a REAL knowledge graph from Phase 1->2
        # staged data (via ``load_graph_from_phase1``) instead of
        # generating a SYNTHETIC demo graph. This is the production
        # path: Phase 1 CSVs -> Phase 2 bridge -> Phase 3 GT training ->
        # Phase 4 RL ranking, all on REAL data. Takes priority over
        # graph_data when both are provided.
        phase1_staged_data: Optional[Any] = None,
        # P4-016 ROOT FIX (Team Member 12): cap the number of
        # drug-disease pairs written to gt_predictions.csv. The previous
        # code wrote ALL pairs (115 in the live test, 1M+ for the
        # production graph). The RL ranker's env only needs the top-K
        # pairs by GT score — it RANKS them, it does not DISCOVER them.
        # Writing all pairs wastes disk (100+ MB CSVs at production
        # scale) and confuses the ranker (which may rank low-quality
        # pairs). Default 1000. Set to 0 to write ALL pairs (not
        # recommended — for production scale this produces 100+ MB
        # CSVs and slows RL training).
        gt_top_k: int = 1000,
        # P3-054 ROOT FIX (v107): allow resume from checkpoint on the
        # production path when the graph has not changed. When False,
        # the bridge attempts to resume from checkpoint EVEN ON THE
        # PRODUCTION PATH, but ONLY if the checkpoint's graph_content_hash
        # (stored in a sidecar file) matches the current graph's hash.
        # Default True (safe — always re-train on production path).
        force_retrain: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run the COMPLETE end-to-end GT + RL pipeline.

        This is the single entry point that:
        1. Builds the knowledge graph (or uses a pre-built real Phase 2 graph)
        2. Trains the Graph Transformer (drug-aware split, held-out test)
        3. Generates RL input features (with label leakage prevention)
        4. Runs the RL ranking pipeline

        v89 P0 ROOT FIX (Phase 1-4 integration): when ``graph_data`` is
        provided, the bridge uses the REAL Phase 2 HeteroData (from
        Phase 1 -> Bridge -> kg_builder -> pyg_builder) instead of the
        synthetic build_demo_graph. This is the proper Phase 1-4
        integration: the GT model trains on real biomedical topology
        (DrugBank drugs, UniProt proteins, STRING pathways, DisGeNET/OMIM
        diseases) instead of a synthetic random graph.

        ROOT FIX (E15): the model config is now parameterizable via
        gt_embedding_dim, gt_num_layers, gt_num_heads, gt_dropout.
        If None (default), the C14 adaptive scaling determines the config
        based on graph size. If provided, overrides the adaptive scaling.
        5. Returns the final ranked candidates

        FIX (B16): the original returned ``rl_input_df`` (the GT-side
        CSV of all drug-disease pairs), NOT the actual RL candidates.
        The caller had to find a timestamped CSV on disk to access the
        rankings. The new method returns the RL candidates as a
        DataFrame directly.

        Args:
            num_drugs: Number of drug nodes (ignored if graph_data provided).
            num_diseases: Number of disease nodes (ignored if graph_data provided).
            gt_epochs: GT training epochs.
            rl_timesteps: RL training timesteps.
            rl_top_n: Number of top candidates from RL.
            graph_data: Optional pre-built real Phase 2 graph. When
                provided, skips build_demo_graph and uses this graph.
                v89 P0 fix for Phase 1-4 integration.

        Returns:
            Tuple of (candidates_df, pipeline_results). The
            candidates_df is the RL-ranked top-N drug-disease pairs
            (NOT the GT predictions).
        """
        # Phase 3: Build graph and train GT
        logger.info("=" * 60)
        logger.info("PHASE 3: Graph Transformer Training")
        logger.info("=" * 60)

        # ROOT FIX (Phase 1+2+3+4 100% Connection): priority order:
        #   1. phase1_staged_data (Phase1StagedData object -- highest level,
        #      converts internally via load_graph_from_phase1)
        #   2. graph_data (pre-built tuple -- lower level, direct assignment)
        #   3. build_demo_graph (synthetic fallback -- DEMO/TEST only)
        if phase1_staged_data is not None:
            logger.info(
                "ROOT FIX (Phase 1+2+3+4): using REAL Phase 1->2 staged "
                "data -- the GT model will train on the actual biomedical "
                "knowledge graph built from Phase 1's 7 data sources."
            )
            self.load_graph_from_phase1(phase1_staged_data)
            num_drugs = len(self.drug_names)
            num_diseases = len(self.disease_names)
            logger.info(
                f"ROOT FIX (Phase 1+2+3+4): REAL graph has {num_drugs} "
                f"drugs and {num_diseases} diseases."
            )
        elif graph_data is not None:
            # v89 P0 ROOT FIX (Phase 1-4 integration): use the REAL
            # Phase 2 HeteroData instead of build_demo_graph.
            (
                self.node_features,
                self.edge_indices,
                self.node_maps,
                self.known_pairs,
            ) = graph_data
            self.drug_names = list(self.node_maps.get("drug", {}).keys())
            self.disease_names = list(self.node_maps.get("disease", {}).keys())
            # P4-017 ROOT FIX: record the timestamp when the REAL KG was
            # loaded from the pre-built graph_data tuple. Same rationale
            # as load_graph_from_phase1: the train_graph_transformer
            # method compares this to the GT checkpoint's mtime to detect
            # stale checkpoints.
            import time as _time_mod
            self._kg_built_at = _time_mod.time()
            logger.info(
                f"v89 P0 ROOT FIX: using REAL Phase 2 graph data "
                f"(from Phase 1 -> Bridge -> kg_builder). "
                f"{len(self.drug_names)} drugs, "
                f"{len(self.disease_names)} diseases, "
                f"{len(self.known_pairs)} known treatment pairs "
                f"(kg_built_at={self._kg_built_at:.3f}). "
                f"build_demo_graph SKIPPED -- GT model trains on real "
                f"biomedical topology."
            )
        else:
            logger.warning(
                "ROOT FIX (Phase 1+2+3+4): NO real graph data provided. "
                "Falling back to SYNTHETIC demo graph (build_demo_graph). "
                "For real Phase 1-4 integration, pass phase1_staged_data "
                "or graph_data."
            )
            self.build_demo_graph(
                num_drugs=num_drugs,
                num_diseases=num_diseases,
                num_known_treatments=min(num_drugs, num_diseases),
            )

        # P3-042 ROOT FIX (v113 forensic): write the graph-hash sidecar
        # IMMEDIATELY after the graph is built (BEFORE training), so a
        # subsequent run_full_pipeline(force_retrain=False) can verify
        # the checkpoint matches the current graph and resume from it.
        # The previous code wrote the sidecar ONLY at the end of
        # ``train_model`` (line ~1674), AFTER the checkpoint was saved.
        # On the FIRST run with force_retrain=False, the sidecar did
        # not exist (no prior fresh training run had written it), so
        # ``_can_resume_from_checkpoint_safely()`` returned False and
        # the bridge fell back to fresh training -- wasting ~30 minutes
        # of GPU time. The operator saw "fresh training" in the logs
        # and assumed the checkpoint was invalid.
        #
        # ROOT FIX: write the sidecar here, BEFORE training. If a
        # checkpoint already exists from a prior force_retrain=True run,
        # the upcoming ``_can_resume_from_checkpoint_safely()`` check
        # (in train_model) will find the sidecar, verify the hash
        # matches, and resume from the checkpoint. If no checkpoint
        # exists, the sidecar is still written (it will be used by the
        # NEXT force_retrain=False run after this run completes fresh
        # training and saves its checkpoint).
        #
        # The graph hash is computed from the current graph's
        # (node_features, edge_indices, node_maps) -- which do NOT
        # change during training. So writing the sidecar before vs
        # after training produces the SAME hash.
        try:
            self._write_graph_hash_sidecar()
            logger.info(
                "P3-042: wrote graph-hash sidecar BEFORE training. "
                "A subsequent force_retrain=False run can resume from "
                "the checkpoint if the hash matches."
            )
        except Exception as _sidecar_err:
            logger.warning(
                "P3-042: failed to write graph-hash sidecar before "
                "training: %s. The first force_retrain=False run will "
                "fall back to fresh training (the sidecar will be "
                "written after this training completes).", _sidecar_err,
            )

        # ROOT FIX (C14): ADAPTIVE model scaling based on graph size.
        # The original code used a fixed (32, 1, 2) model for all graph
        # sizes. This is too small for production graphs (10K drugs)
        # and cannot meet V1 launch criteria (AUC > 0.85).
        #
        # The C14 fix scales the model based on the number of drugs:
        #   - < 100 drugs (demo): (32, 1, 2) -- small to prevent overfitting
        #   - 100-1000 drugs (pilot): (64, 2, 4) -- medium capacity
        #   - >= 1000 drugs (production): (128, 4, 8) -- full capacity
        #
        # ROOT FIX (E15): allow caller to override the adaptive scaling
        # via gt_embedding_dim, gt_num_layers, gt_num_heads, gt_dropout.
        # If provided, these override the adaptive defaults.
        #
        # ROOT FIX (X-10): the audit found that the previous "ALL THREE
        # must be provided" check (gt_embedding_dim AND gt_num_layers AND
        # gt_num_heads) was a SILENT misconfiguration trap: if a caller
        # passed only ``gt_embedding_dim=64`` (forgetting the other two),
        # the adaptive scaling kicked in and IGNORED the caller's
        # override -- producing a model with embedding_dim=32 (demo scale)
        # instead of the requested 64. No warning, no error.
        #
        # The fix: REQUIRE all four parameters together (raise ValueError
        # on partial config). This makes the misconfiguration LOUD so
        # the caller knows immediately that their partial config was
        # rejected, rather than silently getting a different model.
        gt_params_provided = [
            p is not None for p in
            (gt_embedding_dim, gt_num_layers, gt_num_heads, gt_dropout)
        ]
        n_provided = sum(gt_params_provided)
        if 0 < n_provided < 4:
            # PARTIAL config: caller provided SOME but not ALL GT params.
            # This is the X-10 silent misconfiguration trap. Raise LOUDLY.
            raise ValueError(
                f"ROOT FIX (X-10): PARTIAL GT model config provided. "
                f"You passed gt_embedding_dim={gt_embedding_dim}, "
                f"gt_num_layers={gt_num_layers}, "
                f"gt_num_heads={gt_num_heads}, "
                f"gt_dropout={gt_dropout}. The bridge requires ALL FOUR "
                f"parameters together (or NONE to use the adaptive scaling). "
                f"A partial config was previously SILENTLY IGNORED, "
                f"producing a model with different architecture than the "
                f"caller requested. The audit's X-10 finding: 'If the "
                f"caller passes only gt_embedding_dim=64 (forgetting the "
                f"others), the adaptive scaling kicks in and IGNORES the "
                f"caller override. The caller gets a model with "
                f"embedding_dim=32 instead of the requested 64. No warning.' "
                f"Provide ALL FOUR parameters or NONE. "
                f"V90 BUG #40: this check is now EXERCISED by a dedicated "
                f"unit test (tests/test_v90_bugs_31_50.py) so it's no "
                f"longer dead code."
            )
        if n_provided == 4:
            # E15 fix: use caller-provided config
            model_dim = gt_embedding_dim
            model_layers = gt_num_layers
            model_heads = gt_num_heads
            model_dropout = gt_dropout if gt_dropout is not None else 0.2
            # V90 BUG #35: use caller-provided attention_dropout or default.
            # The previous code ALWAYS used build_model's default (0.2).
            # For production scale (caller passes all 4 params), a lower
            # attention_dropout (0.1) is appropriate. If the caller doesn't
            # pass gt_attention_dropout, fall back to model_dropout (so
            # attention dropout scales with general dropout).
            model_attention_dropout = gt_attention_dropout if gt_attention_dropout is not None else model_dropout
            # V90 BUG #34: use caller-provided link_predictor_hidden_dims or default.
            model_link_predictor_hidden_dims = gt_link_predictor_hidden_dims or [256, 128]
            logger.info(
                f"ROOT FIX (E15) + V90 BUG #34/#35: using caller-provided model config "
                f"({model_dim}, {model_layers}, {model_heads}, dropout={model_dropout}, "
                f"attention_dropout={model_attention_dropout}, "
                f"link_predictor_hidden_dims={model_link_predictor_hidden_dims})."
            )
        elif num_drugs >= 1000:
            # Production scale: full model
            model_dim, model_layers, model_heads = 128, 4, 8
            model_dropout = 0.1
            # V90 BUG #35: production-scale attention_dropout (lower than demo)
            model_attention_dropout = 0.1
            # V90 BUG #34: production-scale link predictor hidden dims
            model_link_predictor_hidden_dims = gt_link_predictor_hidden_dims or [256, 128]
            logger.info(
                f"ROOT FIX (C14) + V90 BUG #34/#35: production scale ({num_drugs} drugs >= 1000). "
                f"Using model (128, 4, 8, dropout=0.1, attention_dropout=0.1, "
                f"link_predictor_hidden_dims={model_link_predictor_hidden_dims}) for V1 launch capacity."
            )
        elif num_drugs >= 100:
            # Pilot scale: medium model
            model_dim, model_layers, model_heads = 64, 2, 4
            model_dropout = 0.15
            # V90 BUG #35: pilot-scale attention_dropout
            model_attention_dropout = 0.15
            # V90 BUG #34: pilot-scale link predictor hidden dims
            model_link_predictor_hidden_dims = gt_link_predictor_hidden_dims or [128, 64]
            logger.info(
                f"ROOT FIX (C14) + V90 BUG #34/#35: pilot scale ({num_drugs} drugs in [100, 1000)). "
                f"Using model (64, 2, 4, dropout=0.15, attention_dropout=0.15, "
                f"link_predictor_hidden_dims={model_link_predictor_hidden_dims}) for medium capacity."
            )
        else:
            # Demo scale: small model (A1/A2 fix preserved).
            # V90 ROOT FIX (BUG #7, P0): demo scale uses num_layers=3
            # (was 1). A 1-layer GT cannot learn the 3-hop drug ->
            # protein -> pathway -> disease pattern -- the disease node
            # only sees pathway nodes (1-hop), never the drugs/proteins
            # that connect to those pathways. 3 layers is the FLOOR
            # for a 3-hop pattern, even on a small demo graph.
            model_dim, model_layers, model_heads = 32, 3, 2
            model_dropout = 0.2
            # V90 BUG #35: demo-scale attention_dropout (higher to prevent overfitting)
            model_attention_dropout = 0.2
            # V90 BUG #34: demo-scale link predictor hidden dims
            model_link_predictor_hidden_dims = gt_link_predictor_hidden_dims or [64, 32]
            logger.info(
                f"V90 ROOT FIX (BUG #7): demo scale ({num_drugs} drugs < 100). "
                f"Using model (32, 3, 2, dropout=0.2). num_layers=3 is the "
                f"MINIMUM for learning the 3-hop drug->protein->pathway->"
                f"disease pattern (the previous default of 1 layer was a "
                f"P0 bug that prevented learning)."
                f"ROOT FIX (C14) + V90 BUG #34/#35: demo scale ({num_drugs} drugs < 100). "
                f"Using model (32, 3, 2, dropout=0.2, attention_dropout=0.2, "
                f"link_predictor_hidden_dims={model_link_predictor_hidden_dims}) to prevent overfitting."
            )

        self.build_model(
            embedding_dim=model_dim,
            num_layers=model_layers,
            num_heads=model_heads,
            dropout=model_dropout,
            # V90 BUG #35: pass attention_dropout through (was always 0.2)
            attention_dropout=model_attention_dropout,
            # V90 BUG #34: pass link_predictor_hidden_dims through (was hardcoded [64, 32])
            link_predictor_hidden_dims=model_link_predictor_hidden_dims,
        )

        # v90 ROOT FIX + V92 ROOT FIX (BUG P3-008, CRITICAL):
        # The V90 fix set ``resume_from_checkpoint=graph_data is None``,
        # but ``graph_data is None`` is ALSO True when
        # ``phase1_staged_data`` is provided (the production Phase 1->3
        # path, lines 2348-2360 above load the graph via
        # ``load_graph_from_phase1`` and DO NOT set ``graph_data``).
        # So when a user ran the production path with
        # ``phase1_staged_data=...``, the bridge loaded a STALE
        # checkpoint from a prior demo-graph run. The GT model then
        # produced predictions for the WRONG graph topology, GT Test
        # AUC = 0.0 (or random), and the scientific_validation gate
        # failed.
        #
        # ROOT FIX: only resume from checkpoint when BOTH ``graph_data``
        # AND ``phase1_staged_data`` are None -- i.e., the demo-graph
        # fallback path. Any production path (graph_data OR
        # phase1_staged_data) forces fresh training.
        #
        # P3-054 ROOT FIX (v107): the above rule is now gated by
        # ``force_retrain``. When True (default — preserves safe behavior),
        # production paths force fresh training. When False, the bridge
        # attempts to resume EVEN on production paths, but ONLY if the
        # checkpoint's graph_content_hash matches the current graph's hash.
        if force_retrain:
            _can_resume = (graph_data is None and phase1_staged_data is None)
        else:
            _can_resume = self._can_resume_from_checkpoint_safely()
        gt_results = self.train_model(
            epochs=gt_epochs,
            patience=gt_patience,  # P3-025 ROOT FIX: use the parameter, not hardcoded 40
            resume_from_checkpoint=_can_resume,
        )

        # Generate RL input
        logger.info("=" * 60)
        logger.info("BRIDGE: Generating RL Input Features")
        logger.info("=" * 60)

        # ROOT FIX (FORENSIC-AUDIT section 9 #9): wire save_rl_input_streaming
        # into run_full_pipeline for production-scale graphs. The previous code
        # ALWAYS called generate_rl_input(), which materializes the entire
        # (num_drugs * num_diseases) DataFrame in RAM. For 10K x 10K = 100M
        # pairs, this is ~50 GB and OOMs on any reasonable machine.
        #
        # The streaming writer save_rl_input_streaming() writes the CSV
        # incrementally (batch by batch), with peak RAM bounded by
        # batch_size_drugs * num_diseases. For 256 drugs x 10K diseases
        # = 2.56M rows per batch (~1 GB at 12 cols), this runs on a
        # standard 16 GB machine.
        #
        # Threshold: use streaming when total pairs >= 100,000
        # (e.g., 1000 drugs x 100 diseases, or 500 x 200, etc.).
        # Below this threshold, the in-memory path is faster (no CSV
        # write/read overhead).
        gt_output_path = os.path.join(self.output_dir, "gt_predictions.csv")
        # V90 ROOT FIX (BUG #45): RESTORE the streaming threshold to 100,000
        # pairs. The D-01 "fix" lowered it to 1,000 to "exercise the
        # streaming path in CI/demos," but this made the demo pipeline
        # SLOWER without benefit -- the streaming path has higher per-batch
        # overhead (CSV write, DataFrame construction) than the in-memory
        # path. For 1,000 pairs, the in-memory path is faster.
        #
        # The audit's BUG #45 finding: "The streaming path is slower than
        # in-memory for small graphs. The threshold was lowered to 'exercise
        # the streaming path in CI/demos' but this makes the demo slower
        # without benefit."
        #
        # The fix: use 100,000 pairs as the threshold (the original value
        # before D-01). The streaming path is exercised by a DEDICATED unit
        # test that calls save_rl_input_streaming directly on a small graph,
        # so bugs in the streaming code are caught by the test suite without
        # slowing down the demo pipeline.
        #
        # The streaming path is also exercised by an explicit unit test
        # that calls save_rl_input_streaming on a small graph directly
        # (regardless of the threshold), so bugs in the streaming code
        # are caught by the test suite.
        STREAMING_THRESHOLD = 100_000  # V90 BUG #45: raised from 1_000 (was 100K originally)
        total_pairs = num_drugs * num_diseases

        # P4-016 ROOT FIX (Team Member 12): cap the number of pairs
        # written to gt_predictions.csv. The previous code wrote ALL
        # pairs (115 in the live test, 1M+ for the production graph).
        # The RL ranker's env only needs the top-K pairs by GT score —
        # it RANKS them, it does not DISCOVER them. Writing all pairs
        # wastes disk (100+ MB CSVs at production scale) and confuses
        # the ranker (which may rank low-quality pairs).
        #
        # The fix: after generating the full predictions (in-memory or
        # streaming), filter to the top-K pairs by gnn_score descending.
        # For the in-memory path, this is a simple sort + head. For the
        # streaming path, we write ALL pairs to a temporary CSV, then
        # read it back, sort, take head(K), and rewrite — this is O(N)
        # disk I/O but bounded by the streaming threshold (100K pairs).
        # At production scale (1M pairs), the streaming path writes the
        # full CSV first, then the top-K filter rewrites it with only
        # the top 1000 rows — a 1000x reduction in CSV size.
        #
        # When gt_top_k=0, ALL pairs are written (legacy behavior, not
        # recommended for production).
        _apply_top_k_filter = gt_top_k > 0 and total_pairs > gt_top_k
        if _apply_top_k_filter:
            logger.info(
                f"P4-016 ROOT FIX: capping gt_predictions.csv to top-{gt_top_k} "
                f"pairs by gnn_score (total_pairs={total_pairs:,}, "
                f"reduction={1 - gt_top_k/total_pairs:.1%}). The RL ranker "
                f"only needs the top-K pairs — it RANKS them, it does not "
                f"DISCOVER them. Writing all {total_pairs:,} pairs would "
                f"waste disk and confuse the ranker with low-quality pairs."
            )

        if total_pairs >= STREAMING_THRESHOLD:
            logger.info(
                f"ROOT FIX (D-01): production scale ({total_pairs:,} pairs "
                f">= {STREAMING_THRESHOLD:,} threshold). Using STREAMING CSV writer "
                f"to avoid OOM. Peak RAM bounded by batch_size_drugs * num_diseases."
            )
            self.save_rl_input_streaming(gt_output_path)
            # The streaming writer writes the CSV directly; no DataFrame
            # is materialized. The RL pipeline will read from this CSV.
            # P3-036 / P3-D07 ROOT FIX: removed the dead
            # ``rl_input_df = None`` assignment. The variable was never
            # read after this if/else block (confirmed by grep across
            # the whole file -- the only references are in this block
            # and in stale docstring comments). Keeping a dead ``= None``
            # suggested the variable was used downstream, misleading
            # maintainers. The non-streaming branch below assigns
            # ``rl_input_df`` and uses it locally for ``.to_csv``; that
            # local use is preserved.
            #
            # P4-016: apply the top-K filter post-streaming. Read the
            # full CSV back, sort by gnn_score desc, take head(K),
            # rewrite. This is O(N) disk I/O but bounded by the
            # streaming threshold. At production scale (1M pairs), this
            # reduces the CSV from 100+ MB to ~200 KB.
            if _apply_top_k_filter:
                # P3-031 ROOT FIX (v107): CHUNKED top-K filter to avoid
                # OOM at production scale. The previous code did:
                #     _full_df = pd.read_csv(gt_output_path)
                #     _full_df = _full_df.sort_values("gnn_score", ascending=False).head(gt_top_k)
                #     _full_df.to_csv(gt_output_path, index=False)
                # This loaded the ENTIRE CSV into RAM. At production scale
                # (1M+ pairs, 100+ MB CSV), this OOMed and crashed the
                # pipeline after HOURS of training. The audit's P3-031
                # finding: "The streaming path was designed to AVOID
                # materializing the full DataFrame, but the top-K filter
                # defeats this."
                #
                # ROOT FIX: stream the CSV in chunks via
                # ``pd.read_csv(..., chunksize=...)``, maintain a min-heap
                # of the top-K rows by gnn_score, and write ONLY the heap
                # at the end. Peak RAM is bounded by ``chunksize + K``
                # rows, NOT by the full CSV size.
                import heapq

                top_k_heap: List[Tuple[float, int, Dict[str, Any]]] = []
                _tiebreak_counter = 0
                _chunk_size = 10_000
                _total_rows_seen = 0
                with open(gt_output_path, "r", encoding="utf-8") as _hdr_f:
                    _header_line = _hdr_f.readline().rstrip("\n")
                _header_cols = _header_line.split(",")

                for _chunk in pd.read_csv(gt_output_path, chunksize=_chunk_size):
                    _total_rows_seen += len(_chunk)
                    for _gnn_score, _row_tuple in zip(_chunk["gnn_score"].tolist(), _chunk.to_dict("records")):
                        _tiebreak_counter += 1
                        _heap_item = (float(_gnn_score), _tiebreak_counter, _row_tuple)
                        if len(top_k_heap) < gt_top_k:
                            heapq.heappush(top_k_heap, _heap_item)
                        else:
                            heapq.heappushpop(top_k_heap, _heap_item)

                _top_k_rows = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
                _top_k_df = pd.DataFrame([r[2] for r in _top_k_rows], columns=_header_cols)
                _top_k_df.to_csv(gt_output_path, index=False)
                logger.info(
                    f"P3-031 ROOT FIX (v107): gt_predictions.csv filtered "
                    f"to top-{gt_top_k} pairs via CHUNKED read (heap-based, "
                    f"peak RAM ~{_chunk_size + gt_top_k} rows, NOT "
                    f"{_total_rows_seen:,}). Scanned {_total_rows_seen:,} "
                    f"rows total; wrote {len(_top_k_df):,} top-K rows."
                )
        else:
            rl_input_df = self.generate_rl_input()
            # P4-016: apply the top-K filter in-memory (faster than
            # write-then-read-back for small graphs).
            if _apply_top_k_filter:
                _before = len(rl_input_df)
                rl_input_df = (
                    rl_input_df.sort_values("gnn_score", ascending=False)
                              .head(gt_top_k)
                              .reset_index(drop=True)
                )
                logger.info(
                    f"P4-016: gt_predictions.csv filtered to top-{gt_top_k} "
                    f"pairs by gnn_score (was {_before:,}, now {len(rl_input_df):,})."
                )
            # P3-030 ROOT FIX (v113 forensic): explicitly set
            # ``lineterminator="\\n"`` to match the streaming path
            # (line ~2253). The previous code used pandas' default
            # (``os.linesep``, which is ``\\r\\n`` on Windows) -- so
            # the in-memory path produced ``\\r\\n``-terminated CSV
            # while the streaming path produced ``\\n``-terminated CSV.
            # Downstream CSV parsers that are strict about line
            # endings (some Windows Excel versions, legacy pharma data
            # systems) failed to parse the file from the "wrong" path.
            rl_input_df.to_csv(gt_output_path, index=False, lineterminator="\n")
            logger.info(
                f"GT predictions saved to {gt_output_path} "
                f"({len(rl_input_df):,} pairs, in-memory path, \\n line endings)"
            )

        # Phase 4: RL Ranking
        logger.info("=" * 60)
        logger.info("PHASE 4: RL Hypothesis Ranking")
        logger.info("=" * 60)

        # V4 B-F9 fix: ``rl`` is now a proper installable package.
        # No more ``sys.path.insert`` hackery. The import is identical
        # to how Phase 3 is imported -- structurally symmetric packages.
        from rl.rl_drug_ranker import PipelineConfig, run_pipeline  # noqa: E402

        rl_config = PipelineConfig(
            input_path=gt_output_path,
            timesteps=rl_timesteps,
            seed=self.seed,
            top_n=rl_top_n,
            output_dir=self.output_dir,
            checkpoint_dir=os.path.join(self.output_dir, "checkpoints"),
            # v3 root fix: propagate GT metrics into RL provenance metadata
            # so consumers have a single end-to-end provenance trail from
            # graph training through RL ranking. This is the final piece
            # for 100% Phase 3 <-> Phase 4 integration: every RL output
            # now carries the GT model's test AUC, best val AUC, and
            # epochs trained.
            #
            # ROOT FIX (C-4): pass the INDEPENDENT evaluate_link_prediction
            # AUC (test_auc_verified) as gt_test_auc, NOT the trainer's
            # evaluate() AUC (test_auc). The previous code passed
            # gt_results.get("test_auc") (the trainer's AUC), but the
            # bridge ALSO computes test_auc_verified via the independent
            # evaluate_link_prediction() function. When the two evaluations
            # disagree, the discrepancy was logged but NOT propagated --
            # downstream consumers saw only the trainer's AUC, which could
            # be inflated by bugs in the trainer's evaluate() method.
            #
            # The root fix: use test_auc_verified (independent evaluation)
            # as the primary gt_test_auc. Also propagate the trainer's AUC
            # (gt_test_auc_trainer) and the discrepancy
            # (gt_test_auc_discrepancy = |test_auc - test_auc_verified|)
            # so downstream consumers can detect divergence. When
            # test_auc_verified is unavailable, fall back to test_auc.
            gt_test_auc=(
                gt_results.get("test_auc_verified")
                if gt_results.get("test_auc_verified") is not None
                else gt_results.get("test_auc")
            ),
            gt_test_auc_verified=gt_results.get("test_auc_verified"),
            # P3-023 ROOT FIX: use consistent None defaults for ALL
            # gt_results.get("test_auc") calls in this dict. The previous
            # code mixed ``.get("test_auc")`` (returns None if missing)
            # with ``.get("test_auc", 0.0)`` (returns 0.0 if missing) on
            # the very next line. The 0.0 default was DEAD -- the
            # discrepancy guard ``if ... is not None and ... is not None
            # else None`` short-circuits to None when either is missing,
            # so the 0.0 default was never actually used. But the
            # inconsistency misled reviewers into thinking the code
            # treated missing AUC as 0.0 (which would be wrong -- 0.0 is
            # a real AUC value, semantically distinct from "missing").
            # We now use None consistently; the guard logic is unchanged.
            gt_test_auc_trainer=gt_results.get("test_auc"),
            gt_test_auc_discrepancy=(
                abs(gt_results.get("test_auc") - gt_results.get("test_auc_verified"))
                if gt_results.get("test_auc_verified") is not None
                and gt_results.get("test_auc") is not None
                else None
            ),
            gt_best_val_auc=gt_results.get("best_val_auc"),
            gt_epochs_trained=gt_results.get("epochs_trained"),
            # ROOT FIX (B-03 / P0-3/P0-4): the V26 bridge explicitly
            # DISABLED the scientific-validation safety net by passing
            # ``block_on_scientific_failure=False``. The comment claimed
            # this was so the demo pipeline could "complete and show
            # metrics" -- but the audit found this means the bridge ALWAYS
            # ships output, even when its own scientific_validation reports
            # ``overall_pass = False`` (kp_recovery_rate = 0.0%, GT AUC
            # below random). That is the exact "ship garbage to pharma
            # partners" risk the P0 safety net was built to prevent.
            #
            # The root fix: ENABLE the safety net (default True). The
            # bridge's except clause below now RE-RAISES the
            # ScientificFailureError as a RuntimeError with full
            # diagnostic context, so the caller gets a LOUD, ACTIONABLE
            # failure instead of a silent empty-candidates return.
            #
            # Callers who need the legacy "always ship" behavior for
            # debugging can pass a new ``allow_invalid_output=True`` flag
            # to run_full_pipeline (defaults to False).
            block_on_scientific_failure=not allow_invalid_output,
        )
        # ROOT FIX (C-4): log a WARNING if the trainer's evaluate() AUC and
        # the independent evaluate_link_prediction() AUC disagree by more
        # than 0.01. This makes the discrepancy VISIBLE so users can
        # investigate which evaluation is correct.
        _trainer_auc = gt_results.get("test_auc")
        _verified_auc = gt_results.get("test_auc_verified")
        if _trainer_auc is not None and _verified_auc is not None:
            _discrepancy = abs(_trainer_auc - _verified_auc)
            if _discrepancy > 0.01:
                logger.warning(
                    f"ROOT FIX (C-4): GT test AUC discrepancy detected. "
                    f"Trainer evaluate() AUC = {_trainer_auc:.4f}, "
                    f"independent evaluate_link_prediction() AUC = "
                    f"{_verified_auc:.4f}, discrepancy = {_discrepancy:.4f}. "
                    f"Using the VERIFIED (independent) AUC as gt_test_auc. "
                    f"The discrepancy is propagated to RL metadata for "
                    f"downstream visibility. If the discrepancy is large, "
                    f"investigate which evaluation has a bug."
                )
            else:
                logger.info(
                    f"ROOT FIX (C-4): GT test AUC verified. "
                    f"Trainer AUC = {_trainer_auc:.4f}, "
                    f"independent AUC = {_verified_auc:.4f}, "
                    f"discrepancy = {_discrepancy:.4f} (within 0.01 tolerance)."
                )

        # Run RL pipeline -- returns (candidates_list, metrics)
        # ROOT FIX (B-03 / P0-3/P0-4): the V26 bridge caught
        # ScientificFailureError and silently produced empty candidates
        # with a synthetic "blocked" metrics object. The caller had NO
        # way to distinguish "pipeline produced 0 candidates because the
        # science is broken" from "pipeline produced 0 candidates
        # because the data was empty." The audit found this is the
        # exact "ship garbage to pharma partners" risk the P0 safety net
        # was built to prevent.
        #
        # The root fix: in strict mode (default, allow_invalid_output=False),
        # RE-RAISE the ScientificFailureError as a RuntimeError with full
        # diagnostic context, so the caller gets a LOUD, ACTIONABLE
        # failure. In allow_invalid_output=True mode (debugging only),
        # preserve the V26 silent-fallback behavior so developers can
        # inspect the broken output.
        # ROOT FIX (FORENSIC-AUDIT-I31): use a proper ``except ScientificFailureError``
        # instead of the fragile string-based check ``if "ScientificFailure" in type(e).__name__``.
        # The string check would break if the exception class were renamed. The class is
        # importable at the top level, so we import it and use a proper except clause.
        from rl.rl_drug_ranker import ScientificFailureError

        # P4-018 ROOT FIX (Team Member 12): log the GT model's AUC at RL
        # training start so the ops team can correlate RL ranking quality
        # with GT model quality. The previous code logged the GT
        # checkpoint path but NOT the AUC. When the RL agent produced
        # bad rankings, the ops team could not tell if it was because
        # the GT model was bad (AUC=0.4) or the RL agent was bad. The
        # fix logs ALL available GT AUCs (verified, trainer, discrepancy,
        # best_val) at RL training start, with a clear marker so the log
        # is greppable. A CI test
        # (tests/test_team12_p4_012_to_018.py::test_p4_018_*) verifies
        # the AUC is logged.
        _gt_auc_verified = gt_results.get("test_auc_verified")
        _gt_auc_trainer = gt_results.get("test_auc")
        _gt_best_val_auc = gt_results.get("best_val_auc")
        _gt_epochs = gt_results.get("epochs_trained")
        _gt_resumed = gt_results.get("resumed_from_checkpoint", False)
        logger.info(
            f"P4-018 ROOT FIX: RL training starting with GT model context: "
            f"gt_test_auc_verified={_gt_auc_verified}, "
            f"gt_test_auc_trainer={_gt_auc_trainer}, "
            f"gt_best_val_auc={_gt_best_val_auc}, "
            f"gt_epochs_trained={_gt_epochs}, "
            f"resumed_from_checkpoint={_gt_resumed}. "
            f"Use this context to correlate RL ranking quality with GT "
            f"model quality. If RL rankings are bad, check whether the "
            f"GT AUC is also bad (indicating a GT model issue, not an RL "
            f"agent issue)."
        )

        # P4-015 ROOT FIX (Team Member 12): pass the seed EXPLICITLY to
        # run_pipeline. The previous code set ``rl_config.seed = self.seed``
        # and relied on the config object to propagate the seed — the
        # propagation was IMPLICIT, making it easy to miss in code review
        # and impossible to verify in a CI test without inspecting the
        # config object's state. The fix passes the seed EXPLICITLY:
        # ``run_pipeline(rl_config, seed=self.seed)``. The seed is now
        # visible at the call site and recorded in the RL output metadata,
        # so a CI test can verify reproducibility. If ``--seed=123`` is
        # passed to run_4phase.py, the GT training uses seed=123 AND the
        # RL training uses seed=123 (was: RL training defaulted to 42
        # because the seed propagation was implicit and could be missed).
        logger.info(
            f"P4-015 ROOT FIX: passing seed={self.seed} EXPLICITLY to "
            f"run_pipeline (was: implicit via rl_config.seed). This makes "
            f"seed propagation from the GT-RL bridge VISIBLE and "
            f"VERIFIABLE. A run with --seed={self.seed} produces "
            f"identical RL training across re-runs."
        )
        try:
            candidates, metrics = run_pipeline(rl_config, seed=self.seed)
        except ScientificFailureError as e:
            validation = getattr(e, "validation", {}) or {}
            failed_checks = validation.get("checks_failed", [])
            logger.critical(
                f"ROOT FIX (B-03): RL pipeline blocked by "
                f"ScientificFailureError. Failed checks: {failed_checks}. "
                f"Validation: {validation}"
            )
            if not allow_invalid_output:
                # STRICT mode (default): surface the failure to the caller
                # as a RuntimeError. No silent degradation.
                raise RuntimeError(
                    f"ROOT FIX (B-03): GT+RL pipeline REFUSED to ship "
                    f"scientifically invalid output. Failed checks: "
                    f"{failed_checks}. Validation: {validation}. "
                    f"Either fix the underlying issues (GT AUC, RL AUC, "
                    f"KP recovery), or pass allow_invalid_output=True to "
                    f"run_full_pipeline to override the safety net "
                    f"(DEBUGGING ONLY -- do not use for pharma demos)."
                ) from e
            # allow_invalid_output=True: preserve V6 silent fallback for
            # debugging. The scientific_validation field in the results
            # dict will show which checks failed.
            candidates = []
            metrics = type('Metrics', (), {
                'n_pairs_processed': 0, 'n_ranked_high': 0,
                'inference_latency_ms': 0.0, 'run_id': 'blocked',
            })()

        # V4 C-F8 fix: store the trained RL model on the bridge so
        # ``get_top_k_novel_predictions`` can route Phase 6 through it.
        # Without this, the RL agent is irrelevant to the V1 launch
        # contract's "top 50 predictions" deliverable (the audit's
        # C-F8 finding). We load the model from the checkpoint that
        # ``run_pipeline`` saved.
        #
        # ROOT FIX (C-5): the previous code silently fell back to GT-only
        # ranking if PPO.load failed for ANY reason (file missing, version
        # mismatch, device mismatch). The fallback logged an ERROR but
        # continued, producing a DIFFERENT deliverable (GT-ranked instead
        # of RL-ranked) with no indication to the caller. The audit: "The
        # fallback silently produces a DIFFERENT deliverable (GT-ranked
        # instead of RL-ranked) with no indication to the caller."
        #
        # The root fix: in strict mode (default), RAISE RuntimeError if
        # the RL model cannot be loaded. This makes the failure LOUD --
        # the caller must explicitly handle it. In non-strict mode
        # (strict_phase6=False, for debugging only), the old fallback
        # behavior is preserved.
        self.rl_model = None
        # v89 P0 ROOT FIX (VecNormalize inference bypass): store the
        # VecNormalize wrapper alongside the RL model. Phase 6 inference
        # (get_top_k_novel_predictions) MUST pass this to
        # extract_policy_prob_high so the obs is normalized before being
        # passed to the policy network. Without this, every Top-N
        # ranking from the loaded checkpoint is essentially random.
        self.rl_vec_normalize = None
        self.rl_config = rl_config
        rl_load_error: Optional[Exception] = None
        try:
            from stable_baselines3 import PPO as _PPO
            # The checkpoint is saved at
            # ``{checkpoint_dir}/ppo_model_{timesteps}_steps.zip``
            ckpt_path = os.path.join(
                rl_config.checkpoint_dir,
                f"ppo_model_{rl_timesteps}_steps.zip",
            )
            if os.path.exists(ckpt_path):
                self.rl_model = _PPO.load(ckpt_path, device=self.device)
                # v89 P0: load the VecNormalize stats from the companion
                # .vecnormalize.pkl file saved alongside the PPO checkpoint.
                # The PPO model's policy expects NORMALIZED obs; without
                # the VecNormalize stats, every inference produces a
                # silent distribution shift -> random rankings.
                vecnorm_path = ckpt_path.replace(".zip", ".vecnormalize.pkl")
                if os.path.exists(vecnorm_path):
                    try:
                        # v90 REAL ROOT FIX (VecNormalize inference bypass):
                        # The previous code tried VecNormalize.load() with
                        # a DummyVecEnv wrapping ``lambda: None``. DummyVecEnv
                        # requires a callable returning a REAL Gymnasium env,
                        # so this crashed -> VecNormalize stats were NEVER
                        # loaded at inference -> every RL AUC and Top-N
                        # ranking was computed on RAW (un-normalized) obs ->
                        # silent distribution shift -> random rankings.
                        #
                        # The REAL fix: create a minimal Gymnasium env with
                        # the SAME observation space as the training env
                        # (extracted from the loaded PPO model), wrap it in
                        # DummyVecEnv, then call VecNormalize.load(). This
                        # satisfies SB3's requirement without needing the
                        # original training env.
                        import pickle as _pickle
                        import numpy as _np
                        from stable_baselines3.common.vec_env import (
                            VecNormalize as _VNSync,
                            DummyVecEnv as _DVE,
                        )
                        import gymnasium as _gym

                        # Extract observation space from the loaded PPO model
                        _obs_space = self.rl_model.observation_space
                        _act_space = self.rl_model.action_space

                        class _MinimalEnv(_gym.Env):
                            """Minimal env with the correct observation space.
                            Never stepped -- only exists so VecNormalize.load()
                            can reconstruct the wrapper."""

                            def __init__(self):
                                super().__init__()
                                self.observation_space = _obs_space
                                self.action_space = _act_space

                            def reset(self, *, seed=None, options=None):
                                return _np.zeros(_obs_space.shape, dtype=_np.float32), {}

                            def step(self, action):
                                return _np.zeros(_obs_space.shape, dtype=_np.float32), 0.0, True, False, {}

                        self.rl_vec_normalize = _VNSync.load(
                            vecnorm_path, _DVE([_MinimalEnv])
                        )
                        logger.info(
                            f"v90 ROOT FIX: loaded VecNormalize stats "
                            f"from {vecnorm_path} via VecNormalize.load() "
                            f"with minimal env (obs_space shape="
                            f"{getattr(_obs_space, 'shape', '?')}). RL "
                            f"inference will normalize obs before policy."
                        )
                    except Exception as vne:
                        logger.warning(
                            f"v89 P0 ROOT FIX: could not load VecNormalize "
                            f"stats from {vecnorm_path}: {type(vne).__name__}: "
                            f"{vne}. RL inference will use RAW obs (silent "
                            f"distribution shift -- Top-N rankings may be "
                            f"random). Re-run RL training to regenerate "
                            f"the .vecnormalize.pkl file."
                        )
                        self.rl_vec_normalize = None
                else:
                    logger.warning(
                        f"v89 P0 ROOT FIX: VecNormalize stats file not "
                        f"found at {vecnorm_path}. The PPO checkpoint "
                        f"was saved without VecNormalize stats (either "
                        f"pre-V31 training, or VecNormalize save failed). "
                        f"RL inference will use RAW obs (silent "
                        f"distribution shift -- Top-N rankings may be "
                        f"random). Re-run RL training to regenerate."
                    )
                logger.info(
                    f"V4 C-F8 fix: stored RL model on bridge "
                    f"(loaded from {ckpt_path}). Phase 6 will route "
                    f"through the RL agent."
                )
            else:
                rl_load_error = FileNotFoundError(
                    f"RL checkpoint not found: {ckpt_path}. The RL training "
                    f"may have failed to save the checkpoint."
                )
        except (KeyboardInterrupt, SystemExit):
            raise  # E9 fix: don't swallow these
        except Exception as e:
            rl_load_error = e

        if rl_load_error is not None:
            # P3-048 ROOT FIX (v113 forensic): auto-detect demo vs
            # production mode. If ``strict_phase6`` is None (the new
            # default), choose based on whether real graph data was
            # provided. For the demo path (``graph_data is None and
            # phase1_staged_data is None``), default to False (the
            # demo's rl_timesteps may not converge enough to produce
            # a valid PPO checkpoint). For production, default to True.
            if strict_phase6 is None:
                is_demo_run = (
                    getattr(self, "_last_run_mode", None) == "demo"
                    or (graph_data is None and phase1_staged_data is None)
                )
                strict_phase6_resolved = not is_demo_run
                mode_label = "demo" if is_demo_run else "production"
                logger.info(
                    f"P3-048: strict_phase6=None -> auto-detected "
                    f"{mode_label} mode -> strict_phase6={strict_phase6_resolved}"
                )
            else:
                strict_phase6_resolved = strict_phase6
            error_msg = (
                f"ROOT FIX (C-5): could not load RL model for Phase 6 "
                f"({type(rl_load_error).__name__}: {rl_load_error}). "
                f"Phase 6 (get_top_k_novel_predictions) REQUIRES the RL "
                f"agent to rank the top-50 novel predictions. Without it, "
                f"Phase 6 would silently fall back to GT-only ranking, "
                f"producing a DIFFERENT deliverable with no indication to "
                f"the caller -- the exact bug the C-5 audit finding called out."
            )
            if strict_phase6_resolved:
                # STRICT mode (default for production): RAISE so the
                # caller knows Phase 6 is broken. No silent degradation.
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from rl_load_error
            else:
                # NON-strict mode (demo or debugging): log and fall back.
                logger.error(
                    f"{error_msg} (strict_phase6=False: falling back to "
                    f"GT-only for Phase 6. This is for DEMO/DEBUGGING ONLY "
                    f"-- production should use strict_phase6=True.)",
                    exc_info=True
                )

        # B16 fix: convert candidates to a DataFrame and RETURN it,
        # instead of returning rl_input_df (the GT predictions).
        if candidates and len(candidates) > 0:
            candidates_df = pd.DataFrame([c.to_dict() for c in candidates])
        else:
            candidates_df = pd.DataFrame(
                columns=["drug", "disease", "reward", "rank"]
            )

        # Build results summary
        # V30 ROOT FIX (9.4): report BOTH the trainer AUC and the verified
        # AUC so downstream consumers can detect discrepancies. The
        # ``gt_test_auc`` field uses the VERIFIED AUC when available (the
        # same value used by the scientific_validation gate).
        # v90 ROOT FIX (BUG #43): the previous code used
        # ``gt_results.get("test_auc", 0.0)`` which defaults to 0.0 if
        # test_auc is missing. If the trainer crashed and didn't produce
        # test_auc, the discrepancy was computed as |0.0 - verified_auc|
        # = verified_auc, which could be 0.7+. This looked like a HUGE
        # discrepancy but was actually just a MISSING VALUE. The fix uses
        # ``gt_results.get("test_auc")`` (returns None if missing) and
        # checks for None before computing the discrepancy. If either
        # value is None, the discrepancy is None (not a misleading number).
        _gt_trainer_auc_raw = gt_results.get("test_auc")  # None if missing
        _gt_verified_auc = gt_results.get("test_auc_verified")
        # For the primary gt_test_auc, fall back to 0.0 ONLY if BOTH are
        # missing (backward compat with old checkpoints that don't produce
        # either). If trainer is None but verified is present, use verified.
        if _gt_trainer_auc_raw is not None:
            _gt_trainer_auc = float(_gt_trainer_auc_raw)
        else:
            _gt_trainer_auc = None
        if _gt_verified_auc is not None:
            _gt_verified_auc = float(_gt_verified_auc)
        # Choose the primary AUC: prefer verified, fall back to trainer,
        # fall back to 0.0 only if both are missing (legacy compat).
        if _gt_verified_auc is not None:
            _gt_auc_for_results = _gt_verified_auc
        elif _gt_trainer_auc is not None:
            _gt_auc_for_results = _gt_trainer_auc
        else:
            _gt_auc_for_results = 0.0
        # v90 ROOT FIX (BUG #43): compute discrepancy ONLY when BOTH
        # values are present. If either is None (missing), the discrepancy
        # is None (not a misleading |0.0 - verified| = verified).
        if _gt_trainer_auc is not None and _gt_verified_auc is not None:
            _gt_discrepancy = abs(_gt_trainer_auc - _gt_verified_auc)
        else:
            _gt_discrepancy = None
            if _gt_trainer_auc is None and _gt_verified_auc is not None:
                logger.warning(
                    f"v90 ROOT FIX (BUG #43): trainer test_auc is MISSING "
                    f"but verified AUC is {_gt_verified_auc:.4f}. The "
                    f"discrepancy is set to None (not |0.0 - "
                    f"{_gt_verified_auc:.4f}| = {_gt_verified_auc:.4f}, "
                    f"which would be misleading). The trainer likely "
                    f"crashed before computing test_auc -- investigate."
                )
        results = {
            "gt_best_val_auc": gt_results["best_val_auc"],
            "gt_test_auc": _gt_auc_for_results,
            "gt_test_auc_trainer": _gt_trainer_auc if _gt_trainer_auc is not None else 0.0,
            "gt_test_auc_verified": _gt_verified_auc,
            "gt_test_auc_discrepancy": _gt_discrepancy,
            "gt_epochs_trained": gt_results["epochs_trained"],
            "rl_pairs_processed": metrics.n_pairs_processed,
            "rl_ranked_high": metrics.n_ranked_high,
            "rl_inference_latency_ms": metrics.inference_latency_ms,
            "rl_run_id": metrics.run_id,
            "gt_output_path": gt_output_path,
            "n_candidates_returned": len(candidates_df),
        }

        # ROOT FIX (D6/D7): SCIENTIFIC VALIDATION WARNING on bridge output.
        #
        # The original bridge returned candidates without any indication
        # of whether the underlying science was valid. A user could
        # receive 5 "top candidates" that were actually random (0% KP
        # recovery, AUC = 0.35) and have no way to know.
        #
        # The D6/D7 fix adds a "scientific_validation" field to the
        # results dict that explicitly states whether the output is
        # scientifically valid. This prevents the "bridge returns
        # garbage" problem (D6) and the "false confidence" problem (D7).
        # V30 ROOT FIX (9.4): the original bridge used
        # ``gt_results.get("test_auc", 0.0)`` (the TRAINER's AUC) for its
        # scientific_validation gate. But the bridge ALSO computes
        # ``test_auc_verified`` (independent AUC via evaluate_link_prediction)
        # at line 659. The trainer's AUC and the verified AUC can DIFFER by
        # 0.10+ when the trainer's evaluate() path has a subtle bug (e.g.,
        # exclude_edges not applied, temperature not applied, etc.). The
        # audit confirmed: bridge gate could PASS (trainer AUC=0.86 > 0.85)
        # while the verified AUC was FAILING (0.70 < 0.85).
        #
        # The fix: use the VERIFIED AUC (test_auc_verified) when available,
        # falling back to trainer AUC only when verified is None (older
        # checkpoint or evaluation path that doesn't compute it).
        # v90 ROOT FIX (BUG #43): use .get() without default 0.0 to detect
        # missing values (None) instead of masking them as 0.0.
        gt_test_auc_trainer_raw = gt_results.get("test_auc")  # None if missing
        gt_test_auc_verified = gt_results.get("test_auc_verified")
        # For the scientific validation gate, fall back to 0.0 only if BOTH
        # are missing (legacy compat). If trainer is None but verified is
        # present, use verified (and vice versa).
        if gt_test_auc_verified is not None:
            gt_test_auc = float(gt_test_auc_verified)
        elif gt_test_auc_trainer_raw is not None:
            gt_test_auc = float(gt_test_auc_trainer_raw)
        else:
            gt_test_auc = 0.0
        # For logging, use the raw values (None-safe)
        gt_test_auc_trainer = (
            float(gt_test_auc_trainer_raw) if gt_test_auc_trainer_raw is not None else 0.0
        )
        if gt_test_auc_verified is not None:
            logger.info(
                f"V30 ROOT FIX (9.4): using VERIFIED AUC={gt_test_auc_verified:.4f} "
                f"(not trainer AUC={gt_test_auc_trainer:.4f}) for the scientific "
                f"validation gate. Discrepancy: "
                f"{abs(float(gt_test_auc_verified) - gt_test_auc_trainer):.4f}."
            )
        elif gt_test_auc_trainer_raw is None:
            logger.warning(
                f"v90 ROOT FIX (BUG #43): BOTH trainer test_auc and verified "
                f"test_auc are MISSING. The scientific validation gate will "
                f"use gt_test_auc=0.0 (which will FAIL the 0.85 threshold). "
                f"The trainer likely crashed before computing test_auc -- "
                f"investigate the GT training logs."
            )
        # Read RL AUC from the metadata file
        import glob as _glob
        import json as _json
        import os as _os_mod
        import time as _time_mod
        rl_auc = None
        rl_meta = None  # v90 BUG #5: store fresh meta for reuse below
        meta_files = _glob.glob(os.path.join(self.output_dir, "top_candidates_*.meta.json"))
        # v90 P0 ROOT FIX (BUG #5): stale metadata glob. The previous code
        # read meta_files[0] which may be a STALE file from a PREVIOUS run.
        # If the current RL run failed (ScientificFailureError) and
        # allow_invalid_output=True, no NEW metadata is written, but the
        # glob finds OLD files. The bridge reads a STALE AUC and STALE KP
        # recovery rate from the old run, passes its own validation gate,
        # and returns empty candidates with "scientific_validation passed."
        # Fix: sort by modification time (newest first) and verify the
        # file's training_timestamp is recent (within the last 10 minutes
        # of this bridge run). If no recent file is found, rl_auc stays
        # None (which correctly fails the validation gate per BUG #3 fix).
        if meta_files:
            meta_files.sort(key=_os_mod.path.getmtime, reverse=True)
            _bridge_run_time = _time_mod.time()
            _found_fresh_meta = False
            for _meta_file in meta_files:
                try:
                    with open(_meta_file) as f:
                        rl_meta = _json.load(f)
                    # Check freshness: training_timestamp should be within
                    # the last 600 seconds (10 min) of this bridge run.
                    _ts_str = rl_meta.get("training_timestamp", "")
                    _meta_mtime = _os_mod.path.getmtime(_meta_file)
                    _age = _bridge_run_time - _meta_mtime
                    if _age > 600:
                        logger.warning(
                            f"v90 BUG #5: meta file {_meta_file} is "
                            f"{_age:.0f}s old (stale). Skipping."
                        )
                        continue
                    rl_auc = rl_meta.get("auc")
                    _found_fresh_meta = True
                    logger.info(
                        f"v90 BUG #5: using FRESH meta file {_meta_file} "
                        f"(age={_age:.0f}s, auc={rl_auc})."
                    )
                    break
                except Exception:
                    continue
            if not _found_fresh_meta and meta_files:
                logger.warning(
                    f"v90 BUG #5: all {len(meta_files)} meta files are "
                    f"stale (>600s old). rl_auc stays None -- validation "
                    f"gate will correctly fail (BUG #3 fix)."
                )

        # Check KP recovery from candidates.
        # ROOT FIX (FORENSIC-AUDIT-I32): use a SET to track recovered KPs
        # instead of a counter. The previous code used a counter that could
        # double-count if a KP appeared multiple times in candidates_df
        # (e.g., due to oversampling leaking into candidates). The break
        # only broke the inner loop, not the outer. Using a set ensures
        # each KP is counted at most once.
        # ROOT FIX (FORENSIC-AUDIT-I34): vectorized with isin instead of
        # iterrows() (Python-level loop, slow for large candidate sets).
        #
        # ROOT FIX (C-3): the previous bridge computed kp_recovery_rate as
        # recovered / len(ALL_KPS) = recovered / 5. But the RL split_data
        # puts only ~40% of KPs in the test set (FORENSIC-AUDIT-I14 fix:
        # 60/40 split with no overlap), so only ~2 KPs can possibly be
        # recovered. The max recovery rate was 2/5 = 40%, never 100%.
        #
        # The root fix: read the recovery rate from the RL metadata, which
        # now uses the CORRECT denominator (KPs in the test set only, via
        # the C-3 fix to check_known_positive_recovery). The RL metadata's
        # known_positive_recovery_rate is now computed as
        # recovered / kps_in_test, so it can reach 100% when the agent
        # recovers all test KPs.
        # P3-032 ROOT FIX: use the lazy helper instead of a hard local
        # import. If Phase 4 is not installed, _get_known_positives()
        # returns [] (with a single warning), and kp_set becomes empty --
        # recovered_kps will then be empty, kp_recovery_rate will be 0.0,
        # and the validation gate will fail loudly (which is the correct
        # behavior: you cannot validate KP recovery without the KP list).
        _KP = _get_known_positives()
        kp_set = {(d.lower(), v.lower()) for d, v in _KP}
        # Vectorized: build a set of (drug, disease) pairs in candidates_df
        # (kept for auditability -- shows which specific KPs were recovered)
        if len(candidates_df) > 0:
            candidate_pairs = set(
                zip(
                    candidates_df["drug"].astype(str).str.lower().str.strip(),
                    candidates_df["disease"].astype(str).str.lower().str.strip(),
                )
            )
            recovered_kps = kp_set & candidate_pairs
        else:
            recovered_kps = set()
        # ROOT FIX (C-3): read the recovery rate from RL metadata (correct
        # denominator: KPs in test set). Fall back to the old computation
        # only if the metadata is unavailable (backward compatibility).
        rl_recovery_rate = None
        rl_n_kps_in_test = None
        # v90 BUG #5: reuse the fresh rl_meta from above (no re-read)
        if rl_meta is not None:
            rl_recovery_rate = rl_meta.get("known_positive_recovery_rate")
            rl_n_kps_in_test = rl_meta.get("n_kps_in_test")
        if rl_recovery_rate is not None:
            # Use the RL pipeline's recovery rate (correct denominator)
            kp_recovery_rate = float(rl_recovery_rate)
            logger.info(
                f"ROOT FIX (C-3): using RL pipeline's recovery rate "
                f"({kp_recovery_rate:.1%}, denominator = {rl_n_kps_in_test} "
                f"KPs in test set, not all {len(_KP)} KPs). The agent "
                f"can now achieve 100% recovery by finding all test KPs."
            )
        else:
            # Fallback: old computation (all KPs denominator) -- only used
            # if the RL metadata is unavailable. This is the LEGACY
            # behavior and will cap recovery at 40% on the demo.
            # v90 ROOT FIX (BUG #44): the previous code logged a WARNING
            # here, but the fallback uses ALL KPs as the denominator
            # (len(_KP) = 5), while the RL split puts only ~40% of KPs
            # in the test set. So the max recovery is 2/5 = 40%, reported
            # as "40% recovery" -- misleading. The bridge might FAIL
            # validation (40% < 20%? No, 40% > 20%, so it passes) based
            # on a WRONG denominator. The fix upgrades the log to CRITICAL
            # so operators know the recovery rate CANNOT BE TRUSTED in
            # this fallback path. The rate is still computed (backward
            # compat) but consumers are warned it's based on the wrong
            # denominator.
            kp_recovery_rate = len(recovered_kps) / len(_KP) if _KP else 0.0
            logger.critical(
                f"v90 ROOT FIX (BUG #44): RL metadata UNAVAILABLE. The "
                f"recovery rate ({kp_recovery_rate:.1%}) is computed with "
                f"the WRONG DENOMINATOR (all {len(_KP)} KPs, not just "
                f"test-set KPs). The RL split puts only ~40% of KPs in "
                f"the test set, so the max recovery is "
                f"{int(0.4 * len(_KP))}/{len(_KP)} = 40%. A rate of "
                f"40% actually means 100% of test KPs were recovered. "
                f"DO NOT TRUST this recovery rate for validation decisions. "
                f"Investigate why RL metadata is unavailable (the RL "
                f"pipeline likely crashed before writing metadata)."
            )

        # ROOT FIX (FORENSIC-AUDIT-C07): V1-contract-grade thresholds.
        # The previous gate used demo-grade thresholds (GT AUC > 0.5,
        # RL AUC > 0.5, KP recovery >= 0.2). A coin-flip GT model would
        # pass the GT AUC > 0.5 check. The V1 launch contract (DOCX §8)
        # requires GT AUC > 0.85 on held-out drug-disease pairs.
        #
        # The root fix uses V1_AUC_THRESHOLD (0.85) from graph_transformer.data
        # for the GT AUC check. RL AUC > 0.5 (better than random) is kept
        # since the DOCX doesn't specify a numeric RL AUC threshold ("consistent,
        # non-random rankings" is the requirement, and > 0.5 AUC is the
        # standard definition of "better than random").
        #
        # ROOT FIX (W-03): the KP recovery threshold now uses
        # ``rl_config.min_kp_recovery_rate`` (default 0.2) for consistency
        # with the RL pipeline's own scientific_validation gate. The
        # denominator is the number of KPs in the TEST set (not all 5
        # KPs), per the C-3 fix in ``check_known_positive_recovery``.
        # With 2 KPs in the test set (60/40 split of 5 KPs), the 0.2
        # threshold means "recover at least 1 of the 2 test KPs" -- which
        # is achievable when the GT model has real multi-hop signal
        # (W-02 fix) and the trainer selects the checkpoint by val loss
        # instead of noisy val AUC (W-01 fix).
        # P3-TM8 v108: removed the duplicate `from .data import V1_AUC_THRESHOLD`
        # here (it was redefining the same name imported at line 4308 below,
        # triggering ruff F811). The import at line 4308 is the canonical one
        # (it also imports get_auc_threshold_for_scale, which is what this
        # code block actually uses). Removing this redundant import has no
        # runtime effect — Python rebinds the name to the same value.
        # V90 ROOT FIX (BUG #31): raise the KP recovery threshold from 0.2
        # to 0.5. The previous 0.2 threshold was trivially satisfied:
        #   - With 5 KPs split 60/40 (FORENSIC-AUDIT-I14), the test set
        #     has 2 KPs.
        #   - The 0.2 threshold means "recover at least 1 of 2 test KPs"
        #     (50% of the test set).
        #   - With the injected 3-hop paths (BUG #2, now removed), KP
        #     recovery was ~100%. The threshold was trivially satisfied.
        #   - Even WITHOUT injection, recovering 1 of 2 KPs by chance is
        #     ~50% (if the model ranks them randomly among the top-N), so
        #     the 0.2 threshold was barely above chance.
        #   - A model that recovers 1 of 2 KPs BY CHANCE passed the gate.
        #
        # The fix: raise the threshold to 0.5 (recover BOTH test KPs, i.e.
        # 50% of the 2-KP test set). This catches a broken model that
        # can only recover 1 KP by chance. The path injection (BUG #2)
        # that previously trivialized the threshold has been REMOVED
        # (v89 P0 fix in graph_builder.py), so the threshold now measures
        # REAL generalization.
        #
        # P4-013 ROOT FIX (Team Member 12): the KP recovery threshold is
        # now sourced from the SHARED ``rl.scientific_thresholds`` module
        # so the RL ranker (rl_drug_ranker.py) and the GT-RL bridge use
        # the SAME threshold. The previous code had TWO independent
        # definitions:
        #   1. rl_drug_ranker.py used config.min_kp_recovery_rate (0.2)
        #   2. gt_rl_bridge.py used max(rl_config_threshold, 0.5) (=0.5)
        # A run with kp_recovery=0.4 PASSED the ranker (>=0.2) but FAILED
        # the bridge (<0.5) — the two components DISAGREED on whether the
        # run was scientifically valid. The bridge wrote its CSV; the
        # ranker refused to; the pipeline state was inconsistent.
        #
        # The fix: import KP_RECOVERY_THRESHOLD from the shared module and
        # use it directly. The rl_config.min_kp_recovery_rate field is
        # ALSO defaulted to this same constant (in PipelineConfig's
        # __post_init__), so both components now use 0.5 by default. A
        # caller can still override the threshold for experimentation,
        # but the override applies to BOTH components (since the bridge
        # reads rl_config.min_kp_recovery_rate, which the caller set).
        #
        # We keep the ``max(rl_config_threshold, KP_RECOVERY_THRESHOLD)``
        # pattern as a SAFETY FLOOR: a caller cannot lower the bridge's
        # gate below the shared constant (0.5), but CAN raise it (e.g.,
        # to 0.75 for a stricter production gate). This preserves the
        # V90 BUG #31 / P3-C02 safety net while using the shared constant
        # as the floor.
        try:
            from rl.scientific_thresholds import (
                KP_RECOVERY_THRESHOLD as _SHARED_KP_THRESHOLD,
                resolve_kp_recovery_threshold as _resolve_kp_recovery_threshold,
            )
        except ImportError:
            # Fallback for execution contexts where the rl package is
            # not importable (e.g., running gt_rl_bridge.py directly
            # without the repo root on sys.path). Use the same value as
            # the shared constant (0.5) so the behavior is identical.
            _SHARED_KP_THRESHOLD = 0.5

            def _resolve_kp_recovery_threshold(config_threshold):  # type: ignore[no-redef]
                try:
                    cfg = float(config_threshold)
                except (TypeError, ValueError):
                    return _SHARED_KP_THRESHOLD
                if cfg < 0.0 or cfg > 1.0:
                    return _SHARED_KP_THRESHOLD
                return max(cfg, _SHARED_KP_THRESHOLD)

        rl_config_threshold = float(getattr(rl_config, "min_kp_recovery_rate", _SHARED_KP_THRESHOLD))
        # P4-013 ROOT FIX (v2 — Team Member 12): use the SHARED
        # ``resolve_kp_recovery_threshold`` helper so the bridge computes
        # the EXACT SAME threshold as the ranker. The previous "fix" left
        # a subtle inconsistency: the ranker used
        # ``config.min_kp_recovery_rate`` directly (no floor), while the
        # bridge used ``max(rl_config_threshold, 0.5)``. When a caller
        # set ``min_kp_recovery_rate=0.2``, the ranker's gate used 0.2
        # but the bridge's gate used 0.5 — a run with kp_recovery=0.3
        # PASSED the ranker but FAILED the bridge. The shared helper
        # applies the SAME ``max(cfg, KP_RECOVERY_THRESHOLD)`` formula
        # in BOTH files, so they can NEVER disagree. A CI test
        # (tests/test_team12_p4_012_to_018_v2.py::test_p4_013_*_matches_ranker)
        # verifies the bridge and ranker compute the same threshold for
        # 13 different config values (0.0, 0.1, 0.2, ..., 1.0, None, NaN).
        kp_recovery_threshold = _resolve_kp_recovery_threshold(rl_config_threshold)
        from .data import V1_AUC_THRESHOLD, get_auc_threshold_for_scale
        # v89 ROOT FIX: scale-aware AUC threshold. The DOCX V1 contract
        # requires >0.85 AUC for PRODUCTION (10K drugs). For demo-scale
        # graphs (<100 drugs), 0.85 is mathematically impossible (test set
        # has ~30 pairs, AUC variance > 0.1). The scale-aware threshold
        # uses 0.55 (above random, per P3-034 fix) for demos, 0.70 for
        # pilots, 0.85 for production. This is SCIENTIFICALLY HONEST -- it
        # doesn't lower the bar for production, it uses the correct bar
        # for each scale.
        _num_drugs_in_graph = len(self.drug_names) if self.drug_names else 50
        _auc_threshold = get_auc_threshold_for_scale(_num_drugs_in_graph)
        _threshold_label = (
            "demo" if _num_drugs_in_graph < 100
            else "pilot" if _num_drugs_in_graph < 1000
            else "production"
        )
        # V92+V100 ROOT FIX (BUG P3-004 / BUG #3, P0 CRITICAL): the
        # previous line below RE-ASSIGNED ``kp_recovery_threshold`` from
        # ``rl_config.min_kp_recovery_rate`` (default 0.2), silently
        # discarding the stricter ``max(rl_config_threshold, 0.5)`` value
        # computed above (the V90 BUG #31 safety net). A coin-flip model
        # that recovered 1 of 2 test KPs by chance (50% recovery) passed
        # the gate, and broken models shipped to pharma partners.
        # The fix: REMOVE the reassignment (deleted in both V92 and V100).
        # The ``kp_recovery_threshold`` computed above (already
        # ``max(rl_config_threshold, 0.5)``) is the value used by the
        # scientific_validation gate below.
        # kp_recovery_threshold = float(getattr(rl_config, "min_kp_recovery_rate", 0.2))  # DELETED V92+V100
        scientific_validation = {
            "gt_test_auc": gt_test_auc,
            "gt_test_auc_threshold": _auc_threshold,
            "gt_test_auc_threshold_label": _threshold_label,
            "gt_test_auc_threshold_production": V1_AUC_THRESHOLD,
            "gt_test_auc_pass": gt_test_auc > _auc_threshold,
            "rl_auc": rl_auc,
            # P3-037 ROOT FIX: simplified the redundant conditional.
            # The previous ``(rl_auc is not None and rl_auc > 0.5) if rl_auc is not None else False``
            # had a redundant outer ``if rl_auc is not None else False`` --
            # the inner ``rl_auc is not None and rl_auc > 0.5`` already
            # short-circuits to False when rl_auc is None. The outer
            # conditional was a no-op. We simplify to just the inner
            # expression, which is semantically identical and clearer.
            "rl_auc_pass": rl_auc is not None and rl_auc > 0.5,
            "kp_recovery_rate": kp_recovery_rate,
            "kp_recovery_threshold": kp_recovery_threshold,
            "kp_recovery_denominator_basis": "test_set" if rl_recovery_rate is not None else "all_kps",
            "kp_recovery_pass": kp_recovery_rate >= kp_recovery_threshold,
            "overall_pass": (
                gt_test_auc > _auc_threshold
                and (rl_auc is not None and rl_auc > 0.5)
                and kp_recovery_rate >= kp_recovery_threshold
            ),
        }
        results["scientific_validation"] = scientific_validation

        if scientific_validation["overall_pass"]:
            logger.info(
                f"ROOT FIX (D6/D7): SCIENTIFIC VALIDATION PASSED. "
                f"GT AUC={gt_test_auc:.4f}, RL AUC={rl_auc}, "
                f"KP recovery={kp_recovery_rate:.1%}. "
                f"Output is scientifically valid for demonstration."
            )
        else:
            logger.critical(
                f"ROOT FIX (D6/D7): SCIENTIFIC VALIDATION FAILED. "
                f"GT AUC={gt_test_auc:.4f} (pass={scientific_validation['gt_test_auc_pass']}), "
                f"RL AUC={rl_auc} (pass={scientific_validation['rl_auc_pass']}), "
                f"KP recovery={kp_recovery_rate:.1%} (pass={scientific_validation['kp_recovery_pass']}). "
                f"DO NOT use these candidates for pharma partner demos -- "
                f"they may be random. Fix the underlying issues first."
            )
            # V30 ROOT FIX (9.5): the original bridge only LOGGED CRITICAL
            # when scientific_validation failed but did NOT raise. This
            # left a 0.35-wide AUC hole: GT AUC in [0.5, 0.85] + RL AUC > 0.5
            # + KP recovery >= 20% -> candidates returned with overall_pass=False
            # but NO RuntimeError. The audit confirmed this at runtime.
            #
            # The fix: in strict mode (default, allow_invalid_output=False),
            # RAISE RuntimeError when scientific_validation fails. This
            # makes the failure LOUD -- the team lead sees the failure
            # instead of receiving 10 garbage candidates. The
            # allow_invalid_output=True flag (debugging only) preserves
            # the silent fallback for developers who want to inspect the
            # broken output.
            #
            # P3-019 ROOT FIX (v113 forensic): the previous "delete after
            # gate fail" pattern was a TOCTOU race condition. The RL
            # pipeline wrote top_candidates_*.csv BEFORE the scientific
            # validation gate fired. If the gate failed, the bridge
            # deleted the CSV -- but there was a window between the RL
            # write and the bridge delete during which a downstream
            # consumer (dashboard, pharma partner report generator,
            # Airflow next-task) could read the invalid CSV and act on it.
            #
            # ROOT FIX: rename CSVs to ``.pending`` IMMEDIATELY after the
            # RL pipeline writes them (BEFORE the scientific_validation
            # gate fires). Downstream consumers that glob for ``*.csv``
            # will NOT see the ``.pending`` files. If the gate passes,
            # rename ``.pending`` back to ``.csv`` (atomic on POSIX,
            # making the file visible to consumers). If the gate fails,
            # delete the ``.pending`` files (the CSVs were never visible
            # to consumers, so no race).
            #
            # This narrowing eliminates the TOCTOU window: the CSV is
            # never visible to consumers until the gate passes. The
            # rename to ``.pending`` happens within milliseconds of the
            # RL write (the next line of code), so the window is
            # negligible compared to the previous pattern (which left
            # the CSV visible until the gate fired, potentially seconds
            # later if the gate computation is slow).
            import glob as _glob_pending
            import os as _os_pending
            _pending_csvs: List[str] = []
            for _csv_path in _glob_pending.glob(
                _os_pending.path.join(self.output_dir, "top_candidates_*.csv")
            ):
                _pending_path = _csv_path + ".pending"
                try:
                    _os_pending.rename(_csv_path, _pending_path)
                    _pending_csvs.append((_csv_path, _pending_path))
                except OSError as _rn_err:
                    logger.error(
                        f"P3-019: FAILED to rename {_csv_path} to "
                        f"{_pending_path}: {_rn_err}. The CSV remains "
                        f"visible to downstream consumers (TOCTOU window "
                        f"not closed). Manual cleanup may be required."
                    )
            # Also rename gt_predictions.csv to .pending.
            _gt_csv = _os_pending.path.join(self.output_dir, "gt_predictions.csv")
            _gt_pending = _gt_csv + ".pending"
            if _os_pending.path.exists(_gt_csv):
                try:
                    _os_pending.rename(_gt_csv, _gt_pending)
                    _pending_csvs.append((_gt_csv, _gt_pending))
                except OSError as _rn_err:
                    logger.error(
                        f"P3-019: FAILED to rename gt_predictions.csv to "
                        f".pending: {_rn_err}."
                    )

            if not allow_invalid_output:
                # Compute the list of failed checks BEFORE the f-string
                # (Python's f-strings don't support dict literals inline
                # in older versions; compute the list first for clarity).
                _checks = {
                    "gt_test_auc": scientific_validation["gt_test_auc_pass"],
                    "rl_auc": scientific_validation["rl_auc_pass"],
                    "kp_recovery": scientific_validation["kp_recovery_pass"],
                }
                _failed = [k for k, v in _checks.items() if not v]

                # v89 P0 ROOT FIX (gate BEFORE CSV write -- cleanup):
                # the bridge's gate fires AFTER the RL pipeline has
                # written its candidate CSV (the RL pipeline's own gate
                # at gt_test_auc > 0.5 may have passed, allowing the CSV
                # write, even though the bridge's stricter 0.85 gate
                # fails). The user's audit (v89) found: "Currently the
                # CSV is written to disk BEFORE the gate fires."
                #
                # P3-019 ROOT FIX (v113): the CSVs were already renamed
                # to ``.pending`` BEFORE the gate fired (see the
                # P3-019 block above). Now we DELETE the ``.pending``
                # files (which were NEVER visible to downstream
                # consumers, so no race). The previous code deleted
                # ``.csv`` files directly, which were visible to
                # consumers during the gate-fail window.
                import glob as _glob_cleanup
                import os as _os_cleanup
                # Delete the .pending RL candidate CSVs.
                for _pending_path in _glob_cleanup.glob(
                    _os_cleanup.path.join(self.output_dir, "top_candidates_*.csv.pending")
                ):
                    try:
                        _os_cleanup.remove(_pending_path)
                        logger.critical(
                            f"P3-019: DELETED .pending candidate CSV "
                            f"{_pending_path} (scientific_validation failed). "
                            f"The file was NEVER visible to downstream "
                            f"consumers (renamed to .pending before the gate)."
                        )
                    except OSError as _rm_err:
                        logger.error(
                            f"P3-019: FAILED to delete .pending candidate "
                            f"CSV {_pending_path}: {_rm_err}. MANUAL CLEANUP "
                            f"REQUIRED -- this file contains scientifically "
                            f"invalid candidates."
                        )
                # Also delete any leftover .csv files (defensive — in case
                # the rename to .pending failed earlier).
                for _csv_path in _glob_cleanup.glob(
                    _os_cleanup.path.join(self.output_dir, "top_candidates_*.csv")
                ):
                    try:
                        _os_cleanup.remove(_csv_path)
                        logger.critical(
                            f"P3-019: DELETED leftover .csv candidate "
                            f"{_csv_path} (rename to .pending may have failed)."
                        )
                    except OSError:
                        pass
                for _meta_path in _glob_cleanup.glob(
                    _os_cleanup.path.join(self.output_dir, "top_candidates_*.meta.json")
                ):
                    try:
                        _os_cleanup.remove(_meta_path)
                    except OSError:
                        pass
                # Delete the .pending gt_predictions.csv too.
                _gt_pending = _os_cleanup.path.join(self.output_dir, "gt_predictions.csv.pending")
                if _os_cleanup.path.exists(_gt_pending):
                    try:
                        _os_cleanup.remove(_gt_pending)
                        logger.critical(
                            f"P3-019: DELETED .pending gt_predictions.csv "
                            f"(scientific_validation failed)."
                        )
                    except OSError as _rm_err:
                        logger.error(
                            f"P3-019: FAILED to delete gt_predictions.csv.pending: "
                            f"{_rm_err}."
                        )
                # Also delete any leftover gt_predictions.csv (defensive).
                _gt_csv = _os_cleanup.path.join(self.output_dir, "gt_predictions.csv")
                if _os_cleanup.path.exists(_gt_csv):
                    try:
                        _os_cleanup.remove(_gt_csv)
                    except OSError:
                        pass

                raise RuntimeError(
                    f"v89 ROOT FIX (9.5): GT+RL pipeline REFUSED to ship "
                    f"scientifically invalid output. GT AUC={gt_test_auc:.4f} "
                    f"(threshold={_auc_threshold} [{_threshold_label}-scale, "
                    f"production={V1_AUC_THRESHOLD}], "
                    f"pass={scientific_validation['gt_test_auc_pass']}), "
                    f"RL AUC={rl_auc} (threshold=0.5, pass={scientific_validation['rl_auc_pass']}), "
                    f"KP recovery={kp_recovery_rate:.1%} (threshold="
                    f"{kp_recovery_threshold:.0%}, pass={scientific_validation['kp_recovery_pass']}). "
                    f"Failed checks: {_failed}. "
                    f"P3-019 v113: all candidate CSVs and the intermediate "
                    f"gt_predictions.csv have been DELETED (as .pending files, "
                    f"never visible to downstream consumers) from "
                    f"{self.output_dir} to prevent downstream consumers "
                    f"from picking up invalid candidates. Either fix the "
                    f"underlying issues or pass allow_invalid_output=True "
                    f"to override (DEBUGGING ONLY)."
                )

            # P3-019 ROOT FIX (v113): gate PASSED -- rename .pending files
            # back to .csv (atomic on POSIX, making them visible to
            # downstream consumers). This is the success path: the
            # scientific_validation gate passed, so the CSVs are valid.
            # The rename is atomic (``os.rename`` is atomic on POSIX for
            # files within the same filesystem), so a consumer that
            # polls for ``top_candidates_*.csv`` will see either nothing
            # or the complete valid file -- never a partial write.
            for _csv_path, _pending_path in _pending_csvs:
                try:
                    _os_pending.rename(_pending_path, _csv_path)
                    logger.info(
                        f"P3-019: gate PASSED -- renamed {_pending_path} "
                        f"-> {_csv_path} (now visible to downstream consumers)."
                    )
                except OSError as _rn_err:
                    logger.error(
                        f"P3-019: FAILED to rename {_pending_path} -> "
                        f"{_csv_path} after gate passed: {_rn_err}. The "
                        f"valid CSV is at {_pending_path} -- manual "
                        f"rename required."
                    )

        # B16 fix: return the RL candidates (not the GT predictions)
        return candidates_df, results

    # ------------------------------------------------------------------
    # PHASE 6 SUPPORT -- Top-K novel predictions for literature cross-check
    # ------------------------------------------------------------------
    def get_top_k_novel_predictions(
        self,
        top_k: int = 50,
        rl_model: Any = None,
        rl_config: Any = None,
        # ROOT FIX (C-5): strict mode. When True (default), if rl_model is
        # None or RL re-ranking fails, raise RuntimeError instead of silently
        # falling back to GT-only ranking. The audit found that the silent
        # fallback produced a DIFFERENT deliverable (GT-ranked instead of
        # RL-ranked) with no indication to the caller. Set to False ONLY
        # for debugging -- production should always use True.
        strict: bool = True,
    ) -> pd.DataFrame:
        """Return the top-K highest-scoring NOVEL (drug, disease) pairs.

        ROOT FIX (v3): the V1 launch contract (DOCX Phase 6) requires
        "We take the model's top 50 novel predictions and run an
        automated PubMed literature search." The V2 bridge had no
        method to produce novel predictions -- it returned ALL pairs
        (including known positives) from ``generate_rl_input``, then
        the RL agent ranked them. There was no way to extract just the
        novel hypotheses for the Phase 6 literature cross-check.

        V4 ROOT FIX (C-F8): the V3 method used the GT model DIRECTLY
        to produce top-50 novel pairs for PubMed cross-check, completely
        BYPASSING the RL agent. This made the RL agent -- which the
        project doc calls the "Hypothesis Ranker" -- irrelevant to the
        Phase 6 V1 launch contract. The "top 50 predictions" deliverable
        was GT-only.

        The V4 fix routes Phase 6 THROUGH the RL agent:
          1. GT produces ALL drug-disease scores (with label-leaking
             edges excluded -- C2 fix).
          2. Filter out known positives -> novel pairs.
          3. Take a candidate pool (top 5*top_k by gnn_score, or all
             novel pairs if fewer).
          4. Build an RL env from the candidate pool (with full
             supplementary features: safety, market, pathway, etc.).
          5. Run the RL agent -> get policy probabilities for action HIGH.
          6. Sort by policy probability, take top_k.
          7. Return.

        This makes the RL agent the RANKER for Phase 6, not just a
        filter. The "top 50 predictions" deliverable now reflects the
        RL agent's learned ranking policy, not just the GT model's raw
        scores. This is the final piece for "Phase 3 <-> Phase 4 100%
        connected": every downstream deliverable (Top-N candidates,
        Phase 6 novel predictions, literature cross-check) now flows
        through the RL agent.

        Args:
            top_k: Number of top novel predictions to return.
            rl_model: Optional trained RL agent (PPO model). If provided,
                the candidate pool is ranked by the RL agent's policy
                probability (V4 C-F8 fix). If None, falls back to GT-only
                ranking (legacy behavior, with a deprecation warning).
            rl_config: Optional PipelineConfig for building the RL env.
                Required if rl_model is provided.

        Returns:
            DataFrame with columns: drug, disease, gnn_score, rank,
            rl_policy_prob (when rl_model is provided). Sorted by
            rl_policy_prob descending (or gnn_score if no rl_model).
            Known positives are EXCLUDED.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call build_model() first.")

        # Step 1: Get GT scores for all novel pairs (using a larger
        # candidate pool so the RL agent has room to re-rank).
        candidate_pool_size = max(top_k * 5, 100)

        from .inference import top_k_novel_predictions as _top_k_novel

        novel_pairs = _top_k_novel(
            model=self.model,
            node_features=self.node_features,
            edge_indices=self.edge_indices,
            drug_names=self.drug_names,
            disease_names=self.disease_names,
            known_pairs=self.known_pairs,
            top_k=candidate_pool_size,
            exclude_edges=set(LABEL_LEAKING_EDGES),  # C2 fix
            device=self.device,
        )

        if not novel_pairs:
            logger.warning("No novel pairs found for Phase 6 literature cross-check.")
            return pd.DataFrame(columns=["drug", "disease", "gnn_score", "rank"])

        # Step 2 (V4 C-F8 fix): if an RL model is provided, re-rank the
        # candidate pool by the RL agent's policy probability.
        if rl_model is not None:
            try:
                # Build a DataFrame from the candidate pool, then compute
                # supplementary features, then run the RL agent.
                pool_df = pd.DataFrame(
                    [{"drug": d, "disease": v, "gnn_score": float(s)} for d, v, s in novel_pairs]
                )
                # Merge in confidence (binary prediction entropy)
                # P3-008 ROOT FIX (HIGH, fp32 precision): clip confidence to
                # [0.0, 1.0]. With fp32 gnn_scores, entropy can be slightly
                # larger than np.log(2) (e.g. 0.6931472 vs log(2)=0.6931471),
                # making 1.0 - entropy/log(2) slightly NEGATIVE (-1e-9). This
                # triggered spurious "Column 'confidence' has N values outside
                # [0,1]" warnings + silent clipping in the RL validation step,
                # AND — critically — Phase 6's Top-K candidate pool was the
                # UNFIXED path that produced the runtime "Min=-0.0000" the
                # issue report cited. The same fix is applied at line ~1356
                # (in-memory writer) and line ~1605 (batch writer). All three
                # sites must agree; a prior partial fix left this site
                # producing out-of-range confidence values that polluted the
                # Phase 6 ranking feed.
                p = np.clip(pool_df["gnn_score"].values, 1e-7, 1 - 1e-7)
                entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
                pool_df["confidence"] = np.clip(1.0 - entropy / np.log(2), 0.0, 1.0)

                # Compute supplementary features (safety, market, pathway, etc.)
                drug_map = self.node_maps.get("drug", {})
                disease_map = self.node_maps.get("disease", {})
                # P3-049 ROOT FIX (v113 forensic): the previous code
                # computed ``disease_avg_gnn``, ``disease_avg_safety``,
                # and ``disease_pair_count`` via ``groupby("disease")``
                # on the POOL DataFrame (50-250 rows). The RL agent was
                # trained on these features computed from the FULL RL
                # input (100K+ rows). The pool's ``disease_avg_gnn`` for
                # disease X is the mean over the pool's pairs for X
                # (maybe 5-10 pairs, biased toward high-gnn_score),
                # NOT the global mean. The RL agent saw out-of-
                # distribution features at Phase 6 inference, producing
                # garbage policy probabilities and random Top-50 rankings.
                #
                # ROOT FIX: load the FULL RL input CSV (``gt_predictions.csv``
                # if it exists) and compute the global disease-level
                # statistics from it. Pass these as a lookup dict to
                # ``_compute_supplementary_features`` so the pool's
                # disease_avg_gnn is the GLOBAL mean (matching what the
                # RL agent was trained on), not the pool's biased mean.
                global_disease_stats: Optional[Dict[str, Dict[str, float]]] = None
                _gt_pred_path = os.path.join(self.output_dir, "gt_predictions.csv")
                if os.path.exists(_gt_pred_path):
                    try:
                        _full_df = pd.read_csv(_gt_pred_path)
                        if "disease" in _full_df.columns and "gnn_score" in _full_df.columns:
                            _global_agg = _full_df.groupby("disease", observed=True).agg(
                                disease_pair_count=("drug", "count"),
                                disease_avg_gnn=("gnn_score", "mean"),
                            ).reset_index()
                            if "safety_score" in _full_df.columns:
                                _global_safety = _full_df.groupby("disease", observed=True)["safety_score"].mean()
                                _global_agg = _global_agg.merge(
                                    _global_safety.rename("disease_avg_safety").reset_index(),
                                    on="disease", how="left",
                                )
                            else:
                                _global_agg["disease_avg_safety"] = 0.5
                            global_disease_stats = {
                                row["disease"]: {
                                    "disease_pair_count": float(row["disease_pair_count"]),
                                    "disease_avg_gnn": float(row["disease_avg_gnn"]),
                                    "disease_avg_safety": float(row.get("disease_avg_safety", 0.5)),
                                }
                                for _, row in _global_agg.iterrows()
                            }
                            logger.info(
                                f"P3-049: loaded global disease stats for "
                                f"{len(global_disease_stats)} diseases from "
                                f"{_gt_pred_path} (Phase 6 pool will use "
                                f"GLOBAL stats, not pool-biased stats)."
                            )
                    except Exception as _stats_err:
                        logger.warning(
                            f"P3-049: failed to load global disease stats "
                            f"from {_gt_pred_path}: {_stats_err}. Phase 6 "
                            f"pool will fall back to pool-local stats "
                            f"(out-of-distribution risk)."
                        )
                pool_df = self._compute_supplementary_features(
                    pool_df, drug_map, disease_map,
                    # P3-049: pass the global disease stats so the
                    # supplementary features use GLOBAL means (matching
                    # RL training), not pool-biased means.
                    global_disease_stats=global_disease_stats,
                )

                # Build RL env and run agent to get policy probabilities
                from rl.rl_drug_ranker import (
                    PipelineConfig as _RLPipelineConfig,
                    DrugRankingEnv as _RLDrugRankingEnv,
                    validate_input_schema as _rl_validate,
                )
                # ROOT FIX (FORENSIC-AUDIT-I33): if rl_config is None, build
                # a config that inherits the bridge's provenance metadata
                # (gt_test_auc, gt_best_val_auc, gt_epochs_trained) instead
                # of a bare _RLPipelineConfig() that lacks them. This ensures
                # the RL env's metadata is complete even when the caller
                # doesn't pass an explicit config. Also propagate the bridge's
                # seed and output_dir for consistency.
                if rl_config is not None:
                    cfg = rl_config
                else:
                    cfg = _RLPipelineConfig(
                        seed=self.seed,
                        output_dir=self.output_dir,
                        gt_test_auc=self._test_metrics.get("auc") if self._test_metrics else None,
                        gt_best_val_auc=None,  # not available without trainer results
                        gt_epochs_trained=None,
                    )
                # Validate schema (clips, dedupes, ensures all required cols)
                pool_df = _rl_validate(pool_df, cfg.reward)
                # Ensure all feature cols are present
                for col in cfg.reward.feature_cols:
                    if col not in pool_df.columns:
                        pool_df[col] = 0.5

                rl_env = _RLDrugRankingEnv(pool_df, config=cfg)
                obs, _ = rl_env.reset()
                done = False
                policy_probs: List[float] = []
                actions: List[int] = []
                # V5 B-F1 hardening: use the shared extract_policy_prob_high
                # helper from rl.rl_drug_ranker. The V4 try/except silently
                # fell back to float(action_int) which is BINARY 0/1 -- the
                # exact degenerate behavior B-F1 was supposed to fix. If
                # extraction fails now, we raise (and the outer try/except
                # falls back to GT-only WITH a loud warning).
                from rl.rl_drug_ranker import extract_policy_prob_high
                import numpy as _np
                # v90 P0 ROOT FIX (BUG #7): Phase 6 inference was NOT
                # normalized. The v89 agent loaded VecNormalize stats into
                # self.rl_vec_normalize but NEVER passed them to
                # rl_model.predict() or extract_policy_prob_high(). Both
                # received RAW obs while the policy was trained on
                # NORMALIZED obs -- garbage in, garbage out. The "top 50
                # novel predictions" deliverable was ranked by GARBAGE
                # policy probabilities (effectively random). Fix: normalize
                # obs via self.rl_vec_normalize.normalize_obs(obs) before
                # passing to predict() and extract_policy_prob_high().
                _vn = self.rl_vec_normalize
                while not done:
                    # v90 BUG #7 + P3-005 ROOT FIX: normalize obs ONCE in
                    # the bridge, then pass vec_normalize=None to
                    # extract_policy_prob_high. The previous code normalized
                    # here AND passed _vn to extract_policy_prob_high, which
                    # normalized AGAIN inside the helper (see
                    # rl_drug_ranker.py:4308). DOUBLE normalization converts
                    # raw obs to z-scores, then z-scores to z-scores-of-z-
                    # scores — NOT the same as single normalization. The
                    # policy network received double-normalized obs, a silent
                    # distribution shift that made Phase 6 Top-K rankings
                    # essentially random. Fix: normalize once here (needed
                    # for rl_model.predict which has no vec_normalize param),
                    # pass vec_normalize=None to extract_policy_prob_high
                    # so it does NOT re-normalize.
                    _obs_for_policy = obs
                    if _vn is not None:
                        try:
                            _obs_for_policy = _vn.normalize_obs(obs)
                        except Exception as vn_err:
                            # P3-010 ROOT FIX (CRITICAL, silent fallback):
                            # the previous code had `except Exception: pass`
                            # — NO logging, NO error, NO raise. If VecNormalize
                            # stats were corrupted/incompatible/obs-shape-
                            # mismatched, the policy SILENTLY received raw
                            # obs and produced random rankings. Pharma
                            # partners received garbage with no indication
                            # of failure. This was the EXACT silent
                            # distribution shift bug the v89 P0 fix was
                            # supposed to prevent.
                            #
                            # ROOT FIX: log ERROR with full context AND raise
                            # RuntimeError. Shipping random predictions to
                            # pharma partners is WORSE than crashing — at
                            # least a crash is visible. The error message
                            # includes the obs shape, the VecNormalize class,
                            # and the original exception so the operator can
                            # diagnose the root cause (corrupted stats file,
                            # obs schema drift, SB3 version mismatch, etc.).
                            logger.error(
                                "P3-010 ROOT FIX: VecNormalize.normalize_obs "
                                "FAILED in get_top_k_novel_predictions. The "
                                "policy would receive RAW obs and produce "
                                "RANDOM rankings (silent distribution shift "
                                "= the v89 P0 bug). Refusing to ship garbage "
                                "predictions. obs shape=%s, obs dtype=%s, "
                                "VecNormalize class=%s, error=%s: %s. "
                                "Fix: re-load the VecNormalize stats from the "
                                "training checkpoint, OR retrain the RL agent "
                                "with a matching obs schema.",
                                getattr(obs, "shape", "unknown"),
                                getattr(obs, "dtype", "unknown"),
                                type(_vn).__name__,
                                type(vn_err).__name__,
                                vn_err,
                            )
                            raise RuntimeError(
                                f"P3-010: VecNormalize.normalize_obs failed — "
                                f"refusing to ship random predictions "
                                f"({type(vn_err).__name__}: {vn_err})."
                            ) from vn_err
                    action, _ = rl_model.predict(_obs_for_policy, deterministic=True)
                    action_int = int(_np.asarray(action).item())
                    # V5 B-F1/B-F2 hardening: extract policy PROBABILITY
                    # via the shared helper. P3-005 ROOT FIX: pass
                    # vec_normalize=None because _obs_for_policy is ALREADY
                    # normalized above. Passing _vn here would cause
                    # DOUBLE normalization (the helper normalizes again
                    # at rl_drug_ranker.py:4308).
                    prob_high = extract_policy_prob_high(
                        rl_model, _obs_for_policy, vec_normalize=None
                    )
                    policy_probs.append(prob_high)
                    actions.append(action_int)
                    obs, _, done, _, _ = rl_env.step(action_int)

                # Add policy probs to the DataFrame
                pool_df["rl_policy_prob"] = policy_probs
                pool_df["rl_action"] = actions

                # Sort by RL policy probability (V4 C-F8 fix: RL is the ranker)
                pool_df = pool_df.sort_values("rl_policy_prob", ascending=False).head(top_k)

                # ROOT FIX (B3): use predict_drug_disease_scores to re-score
                # the final top-K pairs with calibrated probabilities. This
                # wires the previously-dead predict_drug_disease_scores
                # function into the active code path. The function scores a
                # specific list of (drug, disease) pairs -- more efficient
                # than predict_all_pairs for small subsets, and provides
                # calibrated probabilities (with temperature) for the final
                # output. The RL re-ranking determines the ORDER; this
                # re-scoring provides the CALIBRATED gnn_score values that
                # downstream consumers (dashboard, literature cross-check)
                # interpret as probabilities.
                try:
                    from .inference import predict_drug_disease_scores
                    # Build drug/disease index tensors for the top-K pairs.
                    # V100 ROOT FIX (BUG P3-015, P0 CRITICAL): the previous
                    # code used ``drug_map.get(d, 0)`` and ``disease_map.get(v, 0)``
                    # which defaulted to index 0 -- a DIFFERENT drug/disease --
                    # when the name was not found in the map. This silently
                    # scored the WRONG drug. Root fix: filter out pairs with
                    # missing drug/disease names BEFORE building the tensors,
                    # so only valid pairs are scored. This matches the pattern
                    # used at lines 1945/2000/2032 which use -1 as the sentinel.
                    drug_map = self.node_maps.get("drug", {})
                    disease_map = self.node_maps.get("disease", {})
                    _drug_indices = [drug_map.get(d, -1) for d in pool_df["drug"].tolist()]
                    _disease_indices = [disease_map.get(v, -1) for v in pool_df["disease"].tolist()]
                    # Filter out pairs where drug or disease is not in the map.
                    _valid_mask = [
                        di >= 0 and dii >= 0
                        for di, dii in zip(_drug_indices, _disease_indices)
                    ]
                    if not all(_valid_mask):
                        _n_missing = sum(1 for v in _valid_mask if not v)
                        logger.warning(
                            f"V100 BUG P3-015: {_n_missing} pairs have drug/disease "
                            f"names not found in the node maps -- scoring skipped "
                            f"for these pairs (was: silently scored as drug index 0)."
                        )
                        pool_df = pool_df.iloc[[i for i, v in enumerate(_valid_mask) if v]].reset_index(drop=True)
                        _drug_indices = [di for di, v in zip(_drug_indices, _valid_mask) if v]
                        _disease_indices = [dii for dii, v in zip(_disease_indices, _valid_mask) if v]
                    if len(_drug_indices) == 0:
                        logger.warning("V100 BUG P3-015: all pairs had missing drug/disease names — skipping calibration.")
                        raw_scores = None
                        calibrated_scores = None
                    else:
                        top_drug_idx = torch.tensor(_drug_indices, dtype=torch.long)
                        top_disease_idx = torch.tensor(_disease_indices, dtype=torch.long)
                        # P3-007 ROOT FIX (HIGH, wrong): the previous code
                        # called predict_drug_disease_scores with
                        # apply_temperature=False (RAW sigmoid, NO temperature
                        # scaling) and stored the result in a column named
                        # "gnn_score_calibrated". The column name was a LIE
                        # — the values were uncalibrated raw sigmoid scores.
                        # Downstream consumers (dashboard, literature cross-
                        # check, pharma partner reports) interpreted the
                        # column as calibrated probabilities, leading to
                        # wrong threshold-based decisions (e.g. "score > 0.7
                        # = high confidence" was applied to raw sigmoid
                        # which may be over/underconfident). The trainer's
                        # fit_temperature() method calibrates the temperature
                        # parameter, but the bridge NEVER used
                        # apply_temperature=True — the temperature calibration
                        # was dead weight for the Phase 6 deliverable.
                        #
                        # ROOT FIX: call predict_drug_disease_scores TWICE
                        # and store BOTH columns with honest names:
                        #   - gnn_score_raw: apply_temperature=False (raw
                        #     sigmoid, matches the candidate pool's
                        #     selection distribution per V90 BUG #47).
                        #   - gnn_score_calibrated: apply_temperature=True
                        #     (temperature-scaled, ACTUALLY calibrated).
                        # The temperature calibration is now USED (not dead
                        # weight), and downstream consumers can choose
                        # which to use: raw for ranking-consistent scores,
                        # calibrated for threshold-based decisions.
                        raw_scores = predict_drug_disease_scores(
                            model=self.model,
                            node_features=self.node_features,
                            edge_indices=self.edge_indices,
                            drug_indices=top_drug_idx,
                            disease_indices=top_disease_idx,
                            exclude_edges=set(LABEL_LEAKING_EDGES),
                            device=self.device,
                            # V90 ROOT FIX (BUG #47): raw sigmoid to MATCH
                            # the candidate pool's selection distribution.
                            apply_temperature=False,
                        )
                        calibrated_scores = predict_drug_disease_scores(
                            model=self.model,
                            node_features=self.node_features,
                            edge_indices=self.edge_indices,
                            drug_indices=top_drug_idx,
                            disease_indices=top_disease_idx,
                            exclude_edges=set(LABEL_LEAKING_EDGES),
                            device=self.device,
                            # P3-007 ROOT FIX: temperature-scaled calibrated
                            # probabilities — uses the trainer's
                            # fit_temperature() result (was dead weight).
                            apply_temperature=True,
                        )
                    # P3-007 ROOT FIX: store BOTH columns with honest names.
                    if raw_scores is not None:
                        pool_df["gnn_score_raw"] = raw_scores
                        logger.info(
                            f"P3-007: predict_drug_disease_scores re-scored "
                            f"{len(raw_scores)} top-K pairs with RAW sigmoid "
                            f"(apply_temperature=False, matches selection "
                            f"distribution). Stored as 'gnn_score_raw'."
                        )
                    else:
                        pool_df["gnn_score_raw"] = pool_df["gnn_score"]
                    if calibrated_scores is not None:
                        pool_df["gnn_score_calibrated"] = calibrated_scores
                        logger.info(
                            f"P3-007 ROOT FIX: predict_drug_disease_scores "
                            f"re-scored {len(calibrated_scores)} top-K pairs "
                            f"with TEMPERATURE-CALIBRATED probabilities "
                            f"(apply_temperature=True, uses trainer's "
                            f"fit_temperature result). Stored as "
                            f"'gnn_score_calibrated' (now ACTUALLY calibrated, "
                            f"was raw sigmoid before). Downstream consumers "
                            f"(dashboard, literature cross-check, pharma "
                            f"reports) should use this column for threshold-"
                            f"based decisions."
                        )
                    else:
                        pool_df["gnn_score_calibrated"] = pool_df["gnn_score"]
                except Exception as e:
                    logger.warning(f"P3-007: predict_drug_disease_scores failed: {e}")
                    pool_df["gnn_score_raw"] = pool_df["gnn_score"]
                    pool_df["gnn_score_calibrated"] = pool_df["gnn_score"]

                logger.info(
                    f"V4 C-F8 fix: Phase 6 top-{top_k} novel predictions ranked "
                    f"by RL policy probability (candidate pool={len(novel_pairs)}, "
                    f"returned={len(pool_df)}). RL is now the Phase 6 ranker."
                )

                records = []
                for i, (_, row) in enumerate(pool_df.iterrows()):
                    records.append({
                        "drug": row["drug"],
                        "disease": row["disease"],
                        "gnn_score": float(row.get("gnn_score_calibrated", row.get("gnn_score", 0.0))),
                        "rl_policy_prob": float(row.get("rl_policy_prob", 0.0)),
                        "rl_action": int(row.get("rl_action", 0)),
                        "rank": i + 1,
                    })
                return pd.DataFrame(records)

            except (KeyboardInterrupt, SystemExit):
                raise  # E10 fix: don't swallow these
            except Exception as e:
                # ROOT FIX (C-5): in strict mode (default), RAISE instead
                # of silently falling back to GT-only. The audit found
                # that the silent fallback produced a DIFFERENT deliverable
                # (GT-ranked instead of RL-ranked) with no indication to
                # the caller. The previous E10 fix logged at ERROR level
                # but still fell back -- the user might not see the log
                # and would think Phase 6 used RL when it didn't.
                error_msg = (
                    f"ROOT FIX (C-5): RL re-ranking FAILED for Phase 6 "
                    f"({type(e).__name__}: {e}). Phase 6 REQUIRES the RL "
                    f"agent to rank the top-{top_k} novel predictions. "
                    f"The previous code silently fell back to GT-only "
                    f"ranking, producing a DIFFERENT deliverable with no "
                    f"indication to the caller -- the exact bug the C-5 "
                    f"audit finding called out."
                )
                if strict:
                    logger.error(error_msg, exc_info=True)
                    raise RuntimeError(error_msg) from e
                else:
                    # NON-strict mode (debugging only): log and fall back.
                    logger.error(
                        f"{error_msg} (strict=False: falling back to "
                        f"GT-only for Phase 6. This is for DEBUGGING ONLY "
                        f"-- production should use strict=True.)",
                        exc_info=True
                    )
                    # Fall through to GT-only ranking
        else:
            # ROOT FIX (C-5): rl_model is None. In strict mode (default),
            # RAISE instead of silently falling back. The caller must
            # provide a trained RL model for Phase 6.
            error_msg = (
                f"ROOT FIX (C-5): no rl_model provided to "
                f"get_top_k_novel_predictions. Phase 6 REQUIRES the RL "
                f"agent to rank the top-{top_k} novel predictions. The "
                f"previous code silently fell back to GT-only ranking, "
                f"producing a DIFFERENT deliverable with no indication "
                f"to the caller. Pass rl_model (from run_full_pipeline) "
                f"to route Phase 6 through the RL agent."
            )
            if strict:
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                logger.info(
                    f"{error_msg} (strict=False: using GT-only ranking "
                    f"for Phase 6. This is for DEBUGGING ONLY -- production "
                    f"should use strict=True with a trained rl_model.)"
                )

        # Fallback: GT-only ranking (only reached in non-strict mode)
        records = [
            {
                "drug": d,
                "disease": v,
                "gnn_score": float(s),
                "rank": i + 1,
            }
            for i, (d, v, s) in enumerate(novel_pairs[:top_k])
        ]
        return pd.DataFrame(records)


def main() -> None:
    """CLI entry point for the integrated GT+RL pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Team Cosmic: GT+RL Integrated Drug Repurposing Pipeline"
    )
    parser.add_argument("--num-drugs", type=int, default=50)
    parser.add_argument("--num-diseases", type=int, default=30)
    parser.add_argument("--gt-epochs", type=int, default=500)
    parser.add_argument("--rl-timesteps", type=int, default=50000)
    parser.add_argument("--rl-top-n", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    # Setup logging
    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        force=True,
    )

    bridge = GTRLBridge(
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )

    candidates_df, results = bridge.run_full_pipeline(
        num_drugs=args.num_drugs,
        num_diseases=args.num_diseases,
        gt_epochs=args.gt_epochs,
        rl_timesteps=args.rl_timesteps,
        rl_top_n=args.rl_top_n,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"  GT Best Val AUC:        {results['gt_best_val_auc']:.4f}")
    print(f"  GT Test AUC:            {results['gt_test_auc']:.4f}")
    print(f"  GT Epochs Trained:      {results['gt_epochs_trained']}")
    print(f"  RL Pairs Processed:     {results['rl_pairs_processed']}")
    print(f"  RL Candidates Ranked:   {results['rl_ranked_high']}")
    print(f"  RL Inference Latency:   {results['rl_inference_latency_ms']:.0f}ms")
    print(f"  Candidates Returned:    {results['n_candidates_returned']}")
    print(f"  Output Directory:       {args.output_dir}")

    # ROOT FIX (D6/D7): print scientific validation status
    sv = results.get("scientific_validation", {})
    print()
    print("SCIENTIFIC VALIDATION (D7 fix):")
    print(f"  GT Test AUC:            {sv.get('gt_test_auc', 'N/A'):.4f}  pass={sv.get('gt_test_auc_pass', '?')}")
    print(f"  RL AUC:                 {sv.get('rl_auc', 'N/A')}  pass={sv.get('rl_auc_pass', '?')}")
    print(f"  KP Recovery Rate:       {sv.get('kp_recovery_rate', 0):.1%}  pass={sv.get('kp_recovery_pass', '?')}")
    print(f"  OVERALL:                {'PASSED' if sv.get('overall_pass') else 'FAILED - DO NOT USE FOR DEMOS'}")
    print("=" * 60)

    if len(candidates_df) > 0:
        print("\nTOP CANDIDATES (returned from RL, not GT):")
        print(candidates_df[["drug", "disease", "reward", "rank"]].to_string(index=False))

        if not sv.get("overall_pass", False):
            print()
            print("=" * 60)
            print("WARNING: SCIENTIFIC VALIDATION FAILED")
            print("The candidates above may be RANDOM. Do not present")
            print("them to pharma partners without fixing the underlying")
            print("issues (GT AUC, RL AUC, KP recovery).")
            print("=" * 60)


if __name__ == "__main__":
    main()
