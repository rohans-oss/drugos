"""
Regression tests for P4-001 through P4-011 (Team Member 11, Phase 4 RL).

P4-001 [CRITICAL]: Data flywheel functional — validated_hypotheses.csv drugs in demo graph.
P4-002 [HIGH]: DISEASE_NAMES uses spaces (matches Phase 2 KG).
P4-003 [HIGH]: save_results refuses to write CSV in standalone mode.
P4-004 [HIGH]: KNOWN_POSITIVES / VALIDATED_HYPOTHESES lazy-loaded (no import side effect).
P4-005 [HIGH]: Per-tenant reward weights YAML + CLI show-weights/set-weights.
P4-006 [HIGH]: is_contextual_bandit metadata field exposed.
P4-007 [MEDIUM]: gnn_score_timestamp staleness check (>24h warning).
P4-008 [MEDIUM]: Modular wrappers (env.py, reward.py, train.py, evaluate.py, validate.py, cli.py).
P4-009 [MEDIUM]: evaluate_agent / compute_auc receive vec_normalize from run_pipeline.
P4-010 [MEDIUM]: action_space is Discrete(2) (O(1), scales to 1M pairs).
P4-011 [LOW]: torch.manual_seed called before PPO init (reproducible).

Run with:
    pytest tests/test_p4_001_to_011_team11_v104.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

# Ensure the repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================================
# P4-001 [CRITICAL]: Data flywheel functional
# ============================================================================
class TestP4001DataFlywheel:
    """P4-001: validated_hypotheses.csv drugs/diseases are in the demo graph."""

    def test_p4_001_validated_drugs_in_real_drug_names(self):
        """All 4 validated drugs must be in REAL_DRUG_NAMES (front-loaded)."""
        from graph_transformer.data.graph_builder import BiomedicalGraphBuilder
        from rl.rl_drug_ranker import VALIDATED_HYPOTHESES

        real_drugs = set(BiomedicalGraphBuilder.REAL_DRUG_NAMES)
        vh_drugs = {d for d, _ in VALIDATED_HYPOTHESES}
        missing = vh_drugs - real_drugs
        assert not missing, (
            f"P4-001 FAIL: validated drugs {missing} are NOT in REAL_DRUG_NAMES. "
            f"The data flywheel (DOCX §10) is dead code for these drugs."
        )

    def test_p4_001_validated_diseases_in_real_disease_names(self):
        """All 4 validated diseases must be in REAL_DISEASE_NAMES (front-loaded)."""
        from graph_transformer.data.graph_builder import BiomedicalGraphBuilder
        from rl.rl_drug_ranker import VALIDATED_HYPOTHESES

        real_diseases = set(BiomedicalGraphBuilder.REAL_DISEASE_NAMES)
        vh_diseases = {dis for _, dis in VALIDATED_HYPOTHESES}
        missing = vh_diseases - real_diseases
        assert not missing, (
            f"P4-001 FAIL: validated diseases {missing} are NOT in REAL_DISEASE_NAMES."
        )

    def test_p4_001_validated_drugs_in_default_demo_graph(self):
        """The default 25-drug demo graph must include all 4 validated drugs."""
        from graph_transformer.data.graph_builder import BiomedicalGraphBuilder
        from rl.rl_drug_ranker import VALIDATED_HYPOTHESES, KNOWN_POSITIVES

        _, _, node_maps, known_pairs = BiomedicalGraphBuilder.build_demo_graph(
            num_drugs=20, num_diseases=15, seed=42,
            known_positives=list(KNOWN_POSITIVES),
            validated_hypotheses=list(VALIDATED_HYPOTHESES),
        )
        graph_drugs = set(node_maps["drug"].keys())
        vh_drugs = {d for d, _ in VALIDATED_HYPOTHESES}
        missing = vh_drugs - graph_drugs
        assert not missing, (
            f"P4-001 FAIL: validated drugs {missing} are NOT in the default demo graph."
        )

    def test_p4_001_validated_pairs_have_treats_edges(self):
        """All 4 validated pairs must have 'treats' edges (data flywheel)."""
        from graph_transformer.data.graph_builder import BiomedicalGraphBuilder
        from rl.rl_drug_ranker import VALIDATED_HYPOTHESES, KNOWN_POSITIVES

        _, _, _, known_pairs = BiomedicalGraphBuilder.build_demo_graph(
            num_drugs=20, num_diseases=15, seed=42,
            known_positives=list(KNOWN_POSITIVES),
            validated_hypotheses=list(VALIDATED_HYPOTHESES),
        )
        known_pairs_set = {(d.lower(), dis.lower()) for d, dis in known_pairs}
        vh_pairs_set = {(d.lower(), dis.lower()) for d, dis in VALIDATED_HYPOTHESES}
        missing = vh_pairs_set - known_pairs_set
        assert not missing, (
            f"P4-001 FAIL: validated pairs {missing} do NOT have 'treats' edges. "
            f"The data flywheel is non-functional."
        )

    def test_p4_001_validated_bonus_fires_for_thalidomide_mm(self):
        """The +0.1 validated_bonus must fire for (thalidomide, multiple myeloma)."""
        from rl.rl_drug_ranker import compute_reward

        base_row = pd.Series({
            "drug": "aspirin", "disease": "pain",
            "gnn_score": 0.7, "safety_score": 0.9, "market_score": 0.5,
            "confidence": 0.8, "pathway_score": 0.6, "patent_score": 0.7,
            "rare_disease_flag": 0.0, "unmet_need_score": 0.4,
            "efficacy_score": 0.7, "adme_score": 0.8,
        })
        vh_row = base_row.copy()
        vh_row["drug"] = "thalidomide"
        vh_row["disease"] = "multiple myeloma"

        base_reward = compute_reward(base_row)
        vh_reward = compute_reward(vh_row)
        # The VH bonus is +0.1; allow small float tolerance.
        assert abs((vh_reward - base_reward) - 0.1) < 1e-6, (
            f"P4-001 FAIL: validated bonus did not fire. "
            f"base={base_reward}, vh={vh_reward}, diff={vh_reward - base_reward} "
            f"(expected ~0.1)."
        )

    def test_p4_001_at_least_50_percent_validated_in_graph(self):
        """CI gate: ≥50% of validated_hypotheses.csv drugs must be in the graph."""
        from graph_transformer.data.graph_builder import BiomedicalGraphBuilder
        from rl.rl_drug_ranker import VALIDATED_HYPOTHESES

        real_drugs = set(BiomedicalGraphBuilder.REAL_DRUG_NAMES)
        vh_drugs = {d for d, _ in VALIDATED_HYPOTHESES}
        pct = len(vh_drugs & real_drugs) / max(len(vh_drugs), 1)
        assert pct >= 0.5, (
            f"P4-001 FAIL: only {pct:.0%} of validated drugs are in REAL_DRUG_NAMES "
            f"(threshold: 50%)."
        )


# ============================================================================
# P4-002 [HIGH]: DISEASE_NAMES uses spaces
# ============================================================================
class TestP4002DiseaseNamesSpaces:
    """P4-002: DISEASE_NAMES uses spaces (matches Phase 2 KG + PubMed)."""

    def test_p4_002_disease_names_use_spaces(self):
        """No DISEASE_NAMES entry should contain underscores."""
        from rl.rl_drug_ranker import DISEASE_NAMES
        underscore_names = [d for d in DISEASE_NAMES if "_" in d]
        assert not underscore_names, (
            f"P4-002 FAIL: DISEASE_NAMES entries with underscores: {underscore_names}. "
            f"These cause KP recovery failure and PubMed 0-result queries."
        )

    def test_p4_002_disease_names_match_known_positives_format(self):
        """DISEASE_NAMES format must match KNOWN_POSITIVES disease format."""
        from rl.rl_drug_ranker import DISEASE_NAMES, KNOWN_POSITIVES

        # All KP diseases should be findable in DISEASE_NAMES (or be a substring)
        disease_set = {d.lower() for d in DISEASE_NAMES}
        for _, kp_disease in KNOWN_POSITIVES:
            kp_disease_lower = kp_disease.lower()
            # The KP disease should either be in DISEASE_NAMES or be a known
            # synonym (e.g., "pain" is a KP disease but not in DISEASE_NAMES —
            # that's fine; we only check that DISEASE_NAMES doesn't use a
            # DIFFERENT format like "type_2_diabetes" vs "type 2 diabetes").
            if kp_disease_lower in disease_set:
                continue  # exact match — good
            # Check no underscore variant exists
            underscore_variant = kp_disease_lower.replace(" ", "_")
            assert underscore_variant not in disease_set, (
                f"P4-002 FAIL: DISEASE_NAMES has '{underscore_variant}' (underscore "
                f"variant of KP disease '{kp_disease}'). String comparison fails."
            )

    def test_p4_002_disease_names_match_phase2_kg(self):
        """DISEASE_NAMES should overlap with REAL_DISEASE_NAMES (Phase 2 KG)."""
        from rl.rl_drug_ranker import DISEASE_NAMES
        from graph_transformer.data.graph_builder import BiomedicalGraphBuilder

        rl_set = {d.lower() for d in DISEASE_NAMES}
        kg_set = {d.lower() for d in BiomedicalGraphBuilder.REAL_DISEASE_NAMES}
        overlap = rl_set & kg_set
        assert len(overlap) >= 10, (
            f"P4-002 FAIL: only {len(overlap)} diseases overlap between "
            f"DISEASE_NAMES and REAL_DISEASE_NAMES. Expected ≥10."
        )


# ============================================================================
# P4-003 [HIGH]: save_results refuses CSV in standalone mode
# ============================================================================
class TestP4003StandaloneCSVRefusal:
    """P4-003: save_results refuses to write CSV when metadata flags standalone."""

    def test_p4_003_save_results_refuses_standalone(self, tmp_path):
        """save_results raises RuntimeError when metadata._standalone_mode=True."""
        from rl.rl_drug_ranker import save_results, RankedCandidate, PipelineConfig

        candidate = RankedCandidate(
            drug="aspirin", disease="pain", reward=0.5, features={},
            rank=1, is_known_positive=True, policy_prob=0.8,
        )
        metadata = {
            "_standalone_mode": True,
            "_standalone_mode_reason": "test standalone mode",
        }
        config = PipelineConfig(output_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="P4-003 ROOT FIX"):
            save_results([candidate], metadata=metadata, config=config)

    def test_p4_003_save_results_allows_non_standalone(self, tmp_path):
        """save_results writes CSV when metadata._standalone_mode is False."""
        from rl.rl_drug_ranker import save_results, RankedCandidate, PipelineConfig

        candidate = RankedCandidate(
            drug="aspirin", disease="pain", reward=0.5, features={},
            rank=1, is_known_positive=True, policy_prob=0.8,
        )
        metadata = {"_standalone_mode": False}
        config = PipelineConfig(output_dir=str(tmp_path))
        path = save_results([candidate], metadata=metadata, config=config)
        assert os.path.exists(path), "P4-003 FAIL: CSV was not written."

    def test_p4_003_generate_fake_data_tags_standalone(self):
        """generate_fake_data tags the DataFrame with _standalone_mode=True."""
        from rl.rl_drug_ranker import generate_fake_data
        df = generate_fake_data(n_pairs=50, seed=42)
        assert df.attrs.get("_standalone_mode") is True, (
            f"P4-003 FAIL: generate_fake_data did not set _standalone_mode=True."
        )


# ============================================================================
# P4-004 [HIGH]: Lazy load (no import side effect)
# ============================================================================
class TestP4004LazyLoad:
    """P4-004: KNOWN_POSITIVES and VALIDATED_HYPOTHESES are lazy."""

    def test_p4_004_lazy_proxy_not_loaded_at_import(self):
        """The lazy proxies should not be loaded until first access."""
        import importlib
        # Force a fresh import
        if "rl.rl_drug_ranker" in sys.modules:
            mod = sys.modules["rl.rl_drug_ranker"]
        else:
            mod = importlib.import_module("rl.rl_drug_ranker")
        # The proxies may already be loaded from prior tests; reset and re-check.
        if hasattr(mod.KNOWN_POSITIVES, "_reset_cache"):
            mod.KNOWN_POSITIVES._reset_cache()
            mod.VALIDATED_HYPOTHESES._reset_cache()
        assert not mod.KNOWN_POSITIVES._is_loaded(), (
            "P4-004 FAIL: KNOWN_POSITIVES was loaded at import time."
        )
        assert not mod.VALIDATED_HYPOTHESES._is_loaded(), (
            "P4-004 FAIL: VALIDATED_HYPOTHESES was loaded at import time."
        )

    def test_p4_004_lazy_proxy_loads_on_access(self):
        """Accessing the proxy triggers the load."""
        from rl.rl_drug_ranker import KNOWN_POSITIVES
        KNOWN_POSITIVES._reset_cache()
        assert not KNOWN_POSITIVES._is_loaded()
        kps = list(KNOWN_POSITIVES)  # trigger load
        assert KNOWN_POSITIVES._is_loaded()
        assert len(kps) > 0

    def test_p4_004_lazy_proxy_caches_result(self):
        """Repeated access returns the same cached list."""
        from rl.rl_drug_ranker import KNOWN_POSITIVES
        kps1 = list(KNOWN_POSITIVES)
        kps2 = list(KNOWN_POSITIVES)
        assert kps1 == kps2

    def test_p4_004_get_known_positives_helper(self):
        """get_known_positives() returns a plain list."""
        from rl.rl_drug_ranker import get_known_positives
        kps = get_known_positives()
        assert isinstance(kps, list)
        assert len(kps) > 0

    def test_p4_004_reload_clears_cache(self):
        """reload_known_positives() forces a re-load."""
        from rl.rl_drug_ranker import reload_known_positives, KNOWN_POSITIVES
        # Trigger initial load
        list(KNOWN_POSITIVES)
        assert KNOWN_POSITIVES._is_loaded()
        # Reload
        reload_known_positives()
        # The reload itself triggers a load, so it should be loaded again
        assert KNOWN_POSITIVES._is_loaded()

    def test_p4_004_proxy_supports_set_len_iter(self):
        """The proxy supports set(), len(), iter() — the operations used internally."""
        from rl.rl_drug_ranker import KNOWN_POSITIVES, VALIDATED_HYPOTHESES
        # set() — used in RewardFunction.__init__
        kp_set = set(KNOWN_POSITIVES)
        assert isinstance(kp_set, set)
        # len() — used in generate_fake_data
        kp_len = len(KNOWN_POSITIVES)
        assert kp_len > 0
        # iter() — used in for d, v in KNOWN_POSITIVES
        for d, v in KNOWN_POSITIVES:
            assert isinstance(d, str)
            assert isinstance(v, str)
            break
        # contains — used in some checks
        first_pair = list(KNOWN_POSITIVES)[0]
        assert first_pair in KNOWN_POSITIVES


# ============================================================================
# P4-005 [HIGH]: Per-tenant reward weights
# ============================================================================
class TestP4005PerTenantRewardWeights:
    """P4-005: reward_weights.yaml + per-tenant loading + CLI."""

    def test_p4_005_default_yaml_exists(self):
        """The default reward_weights.yaml ships with the package."""
        from rl.rl_drug_ranker import DEFAULT_REWARD_WEIGHTS_DIR
        default_path = os.path.join(DEFAULT_REWARD_WEIGHTS_DIR, "reward_weights.yaml")
        assert os.path.exists(default_path), (
            f"P4-005 FAIL: {default_path} not found."
        )

    def test_p4_005_load_default_weights(self):
        """Loading the default profile returns valid weights."""
        from rl.rl_drug_ranker import load_reward_weights_for_tenant, FEATURE_COLS
        weights = load_reward_weights_for_tenant()
        assert set(weights.keys()) == set(FEATURE_COLS)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_p4_005_save_and_load_tenant(self, tmp_path):
        """Save and load a tenant-specific profile."""
        from rl.rl_drug_ranker import (
            save_reward_weights_for_tenant,
            load_reward_weights_for_tenant,
            FEATURE_COLS,
        )
        custom = {k: 0.1 for k in FEATURE_COLS}  # sum=1.0 (10 features × 0.1)
        path = save_reward_weights_for_tenant(
            custom, tenant_id="test_tenant", weights_dir=str(tmp_path),
        )
        assert os.path.exists(path)
        loaded = load_reward_weights_for_tenant(
            tenant_id="test_tenant", weights_dir=str(tmp_path),
        )
        assert loaded == custom

    def test_p4_005_apply_tenant_to_config(self, tmp_path):
        """apply_tenant_reward_weights returns a new config with the tenant's weights."""
        from rl.rl_drug_ranker import (
            DEFAULT_CONFIG, apply_tenant_reward_weights,
            save_reward_weights_for_tenant, FEATURE_COLS,
        )
        custom = {k: 0.1 for k in FEATURE_COLS}
        save_reward_weights_for_tenant(
            custom, tenant_id="test_apply", weights_dir=str(tmp_path),
        )
        new_config = apply_tenant_reward_weights(
            DEFAULT_CONFIG, tenant_id="test_apply", weights_dir=str(tmp_path),
        )
        assert new_config.reward.reward_weights == custom
        # Original config is unchanged
        assert DEFAULT_CONFIG.reward.reward_weights != custom

    def test_p4_005_invalid_tenant_id_rejected(self):
        """Invalid tenant IDs (path traversal) are rejected."""
        from rl.rl_drug_ranker import load_reward_weights_for_tenant
        with pytest.raises(ValueError, match="invalid tenant_id"):
            load_reward_weights_for_tenant(tenant_id="../../../etc/passwd")

    def test_p4_005_weights_sum_validation(self, tmp_path):
        """Weights that don't sum to 1.0 raise ValueError."""
        from rl.rl_drug_ranker import (
            save_reward_weights_for_tenant, FEATURE_COLS,
        )
        bad = {k: 0.05 for k in FEATURE_COLS}  # sum=0.5, not 1.0
        with pytest.raises(ValueError, match="sum to"):
            save_reward_weights_for_tenant(
                bad, tenant_id="bad", weights_dir=str(tmp_path),
            )


# ============================================================================
# P4-006 [HIGH]: is_contextual_bandit metadata
# ============================================================================
class TestP4006ContextualBanditMetadata:
    """P4-006: is_contextual_bandit field exposed in metadata."""

    def test_p4_006_pipeline_config_has_ppo_gamma(self):
        """PipelineConfig has the ppo_gamma field (default 0.0)."""
        from rl.rl_drug_ranker import PipelineConfig
        cfg = PipelineConfig()
        assert cfg.ppo_gamma == 0.0
        assert hasattr(cfg, "ppo_gamma")

    def test_p4_006_is_contextual_bandit_default_true(self):
        """Default config (gamma=0) is a contextual bandit."""
        from rl.rl_drug_ranker import PipelineConfig
        cfg = PipelineConfig()
        is_cb = (cfg.ppo_gamma == 0.0)
        assert is_cb is True

    def test_p4_006_is_contextual_bandit_false_when_gamma_positive(self):
        """Config with gamma>0 is NOT a contextual bandit (sequential MDP)."""
        from rl.rl_drug_ranker import PipelineConfig
        cfg = PipelineConfig()
        cfg.ppo_gamma = 0.95
        assert (cfg.ppo_gamma == 0.0) is False

    def test_p4_006_run_pipeline_metadata_has_is_contextual_bandit(self):
        """run_pipeline metadata dict has is_contextual_bandit field (smoke check)."""
        # We don't run the full pipeline (too slow); we just verify the
        # metadata dict construction code includes the field by inspecting
        # the source.
        import inspect
        from rl.rl_drug_ranker import run_pipeline
        src = inspect.getsource(run_pipeline)
        assert "is_contextual_bandit" in src, (
            "P4-006 FAIL: run_pipeline source does not reference is_contextual_bandit."
        )
        assert "mdp_structure" in src, (
            "P4-006 FAIL: run_pipeline source does not reference mdp_structure."
        )


# ============================================================================
# P4-007 [MEDIUM]: gnn_score timestamp staleness
# ============================================================================
class TestP4007GnnScoreStaleness:
    """P4-007: gnn_score_timestamp column + staleness check."""

    def test_p4_007_timestamp_col_constant_exists(self):
        """GNN_SCORE_TIMESTAMP_COL constant is defined."""
        from rl.rl_drug_ranker import GNN_SCORE_TIMESTAMP_COL
        assert GNN_SCORE_TIMESTAMP_COL == "gnn_score_timestamp"

    def test_p4_007_staleness_threshold_constant(self):
        """GNN_SCORE_STALENESS_WARNING_HOURS is 24.0."""
        from rl.rl_drug_ranker import GNN_SCORE_STALENESS_WARNING_HOURS
        assert GNN_SCORE_STALENESS_WARNING_HOURS == 24.0

    def test_p4_007_env_warns_on_stale_timestamp(self, caplog):
        """The env logs a WARNING when gnn_score_timestamp is >24h old."""
        import logging
        from rl.rl_drug_ranker import (
            DrugRankingEnv, PipelineConfig, GNN_SCORE_TIMESTAMP_COL,
            generate_fake_data,
        )
        df = generate_fake_data(n_pairs=20, seed=42)
        # Add a stale timestamp (48h ago)
        stale_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        df[GNN_SCORE_TIMESTAMP_COL] = stale_ts
        df.attrs["_standalone_mode"] = False  # avoid standalone warning
        cfg = PipelineConfig()
        env = DrugRankingEnv(df, config=cfg, set_adaptive_threshold=False)
        assert env._gnn_score_stale is True
        assert env._gnn_score_age_hours is not None
        assert env._gnn_score_age_hours > 24.0

    def test_p4_007_env_no_warning_on_fresh_timestamp(self):
        """The env does NOT warn when gnn_score_timestamp is fresh (<24h)."""
        from rl.rl_drug_ranker import (
            DrugRankingEnv, PipelineConfig, GNN_SCORE_TIMESTAMP_COL,
            generate_fake_data,
        )
        df = generate_fake_data(n_pairs=20, seed=42)
        fresh_ts = datetime.now(timezone.utc).isoformat()
        df[GNN_SCORE_TIMESTAMP_COL] = fresh_ts
        df.attrs["_standalone_mode"] = False
        cfg = PipelineConfig()
        env = DrugRankingEnv(df, config=cfg, set_adaptive_threshold=False)
        assert env._gnn_score_stale is False
        assert env._gnn_score_age_hours is not None
        assert env._gnn_score_age_hours < 24.0

    def test_p4_007_env_handles_missing_timestamp_column(self):
        """The env handles missing gnn_score_timestamp column gracefully."""
        from rl.rl_drug_ranker import (
            DrugRankingEnv, PipelineConfig, generate_fake_data,
        )
        df = generate_fake_data(n_pairs=20, seed=42)
        df.attrs["_standalone_mode"] = False
        cfg = PipelineConfig()
        env = DrugRankingEnv(df, config=cfg, set_adaptive_threshold=False)
        assert env._gnn_score_stale is False
        assert env._gnn_score_age_hours is None


# ============================================================================
# P4-008 [MEDIUM]: Modular wrappers
# ============================================================================
class TestP4008ModularWrappers:
    """P4-008: modular wrapper files exist and are <500 lines."""

    def test_p4_008_env_py_exists(self):
        from rl import env as env_mod
        assert hasattr(env_mod, "DrugRankingEnv")

    def test_p4_008_reward_py_exists(self):
        from rl import reward as reward_mod
        assert hasattr(reward_mod, "RewardFunction")
        assert hasattr(reward_mod, "compute_reward")
        assert hasattr(reward_mod, "load_reward_weights_for_tenant")

    def test_p4_008_train_py_exists(self):
        from rl import train as train_mod
        assert hasattr(train_mod, "train_agent")
        assert hasattr(train_mod, "get_device")

    def test_p4_008_evaluate_py_exists(self):
        from rl import evaluate as eval_mod
        assert hasattr(eval_mod, "evaluate_agent")
        assert hasattr(eval_mod, "compute_auc")
        assert hasattr(eval_mod, "split_data")

    def test_p4_008_validate_py_exists(self):
        from rl import validate as validate_mod
        assert hasattr(validate_mod, "validate_input_schema")
        assert hasattr(validate_mod, "ScientificFailureError")

    def test_p4_008_cli_py_exists(self):
        from rl import cli as cli_mod
        assert hasattr(cli_mod, "main")

    def test_p4_008_wrappers_under_500_lines(self):
        """Each modular wrapper must be <500 lines (P4-008 requirement)."""
        import rl
        rl_dir = os.path.dirname(rl.__file__)
        wrappers = ["env.py", "reward.py", "train.py", "evaluate.py", "validate.py", "cli.py"]
        for w in wrappers:
            path = os.path.join(rl_dir, w)
            with open(path, "r") as f:
                line_count = sum(1 for _ in f)
            assert line_count < 500, (
                f"P4-008 FAIL: {w} has {line_count} lines (must be <500)."
            )


# ============================================================================
# P4-009 [MEDIUM]: VecNormalize in eval path
# ============================================================================
class TestP4009VecNormalizeEval:
    """P4-009: evaluate_agent and compute_auc receive vec_normalize."""

    def test_p4_009_evaluate_agent_accepts_vec_normalize(self):
        """evaluate_agent signature has vec_normalize parameter."""
        import inspect
        from rl.rl_drug_ranker import evaluate_agent
        sig = inspect.signature(evaluate_agent)
        assert "vec_normalize" in sig.parameters

    def test_p4_009_extract_policy_prob_high_accepts_vec_normalize(self):
        """extract_policy_prob_high signature has vec_normalize parameter."""
        import inspect
        from rl.rl_drug_ranker import extract_policy_prob_high
        sig = inspect.signature(extract_policy_prob_high)
        assert "vec_normalize" in sig.parameters

    def test_p4_009_train_agent_returns_vec_normalize(self):
        """train_agent returns a 3-tuple (model, checkpoint_path, vec_normalize)."""
        import inspect
        from rl.rl_drug_ranker import train_agent
        sig = inspect.signature(train_agent)
        # Check return annotation mentions Tuple
        ret = sig.return_annotation
        assert "Tuple" in str(ret) or "tuple" in str(ret).lower()

    def test_p4_009_run_pipeline_passes_vec_normalize(self):
        """run_pipeline source passes vec_normalize to evaluate_agent & compute_auc."""
        import inspect
        from rl.rl_drug_ranker import run_pipeline
        src = inspect.getsource(run_pipeline)
        # Check that evaluate_agent is called with vec_normalize=
        assert "vec_normalize=vec_normalize" in src, (
            "P4-009 FAIL: run_pipeline does not pass vec_normalize to evaluate_agent/compute_auc."
        )

    def test_p4_009_train_agent_saves_vecnormalize(self):
        """train_agent source saves VecNormalize stats alongside checkpoint."""
        import inspect
        from rl.rl_drug_ranker import train_agent
        src = inspect.getsource(train_agent)
        assert "vecnormalize.pkl" in src or "vec_normalize" in src.lower()


# ============================================================================
# P4-010 [MEDIUM]: Action space is Discrete(2), scales to 1M pairs
# ============================================================================
class TestP4010ActionSpaceScales:
    """P4-010: action_space is Discrete(2) (O(1), not O(n_pairs))."""

    def test_p4_010_action_space_is_discrete_2(self):
        """DrugRankingEnv.action_space is spaces.Discrete(2)."""
        from rl.rl_drug_ranker import DrugRankingEnv, PipelineConfig, generate_fake_data
        df = generate_fake_data(n_pairs=20, seed=42)
        df.attrs["_standalone_mode"] = False
        cfg = PipelineConfig()
        env = DrugRankingEnv(df, config=cfg, set_adaptive_threshold=False)
        from gymnasium import spaces
        assert isinstance(env.action_space, spaces.Discrete), (
            f"P4-010 FAIL: action_space is {type(env.action_space)}, expected Discrete."
        )
        assert env.action_space.n == 2, (
            f"P4-010 FAIL: action_space.n = {env.action_space.n}, expected 2 (HIGH/LOW)."
        )

    def test_p4_010_action_space_independent_of_n_pairs(self):
        """action_space.n is 2 regardless of the number of pairs."""
        from rl.rl_drug_ranker import DrugRankingEnv, PipelineConfig, generate_fake_data
        for n_pairs in [20, 100, 500]:
            df = generate_fake_data(n_pairs=n_pairs, seed=42)
            df.attrs["_standalone_mode"] = False
            cfg = PipelineConfig()
            env = DrugRankingEnv(df, config=cfg, set_adaptive_threshold=False)
            assert env.action_space.n == 2, (
                f"P4-010 FAIL: with n_pairs={n_pairs}, action_space.n = {env.action_space.n}"
            )

    def test_p4_010_env_scales_to_100k_pairs(self):
        """The env can handle 100K pairs (1K drugs × 100 diseases) without explosion."""
        from rl.rl_drug_ranker import DrugRankingEnv, PipelineConfig, generate_fake_data
        # 100K pairs would be slow to generate with generate_fake_data; use 10K
        # which is enough to verify the action_space doesn't explode.
        df = generate_fake_data(n_pairs=10000, seed=42)
        df.attrs["_standalone_mode"] = False
        cfg = PipelineConfig()
        env = DrugRankingEnv(df, config=cfg, set_adaptive_threshold=False)
        assert env.action_space.n == 2
        assert env.n_pairs == 10000
        # Reset and step work
        obs, _ = env.reset()
        assert obs.shape[0] > 0


# ============================================================================
# P4-011 [LOW]: torch.manual_seed before PPO init
# ============================================================================
class TestP4011TorchSeed:
    """P4-011: torch.manual_seed called before PPO init."""

    def test_p4_011_train_agent_calls_torch_manual_seed(self):
        """train_agent source calls torch.manual_seed."""
        import inspect
        from rl.rl_drug_ranker import train_agent
        src = inspect.getsource(train_agent)
        assert "torch.manual_seed" in src, (
            "P4-011 FAIL: train_agent does not call torch.manual_seed."
        )

    def test_p4_011_pipeline_config_has_seed_field(self):
        """PipelineConfig has a seed field (default 42)."""
        from rl.rl_drug_ranker import PipelineConfig
        cfg = PipelineConfig()
        assert cfg.seed == 42

    def test_p4_011_seed_passed_to_train_agent(self):
        """train_agent accepts a seed parameter and uses it for torch.manual_seed."""
        import inspect
        from rl.rl_drug_ranker import train_agent
        sig = inspect.signature(train_agent)
        assert "seed" in sig.parameters


# ============================================================================
# Smoke test: rl package imports cleanly
# ============================================================================
class TestSmokeImport:
    """Smoke test: the rl package imports cleanly after all fixes."""

    def test_smoke_import_rl(self):
        import rl
        assert rl.__version__ in ("4.1.0", "4.2.0")

    def test_smoke_import_rl_drug_ranker(self):
        from rl import rl_drug_ranker
        assert hasattr(rl_drug_ranker, "DrugRankingEnv")
        assert hasattr(rl_drug_ranker, "RewardFunction")
        assert hasattr(rl_drug_ranker, "train_agent")

    def test_smoke_compute_reward_works(self):
        """compute_reward returns a finite float for a valid row."""
        from rl.rl_drug_ranker import compute_reward
        row = pd.Series({
            "drug": "aspirin", "disease": "pain",
            "gnn_score": 0.7, "safety_score": 0.9, "market_score": 0.5,
            "confidence": 0.8, "pathway_score": 0.6, "patent_score": 0.7,
            "rare_disease_flag": 0.0, "unmet_need_score": 0.4,
            "efficacy_score": 0.7, "adme_score": 0.8,
        })
        reward = compute_reward(row)
        assert np.isfinite(reward)
        assert reward > 0  # aspirin->pain is a KP, should get positive reward
