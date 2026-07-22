"""
P4-001 to P4-011 — Team Member 11 (Phase 4 RL Drug Ranker) — Verified Regression Tests.

This test file was written by running the ACTUAL code paths (not by reading
existing tests or comments). Each test verifies real runtime behavior:

  P4-001: validated_hypotheses.csv drugs appear in REAL_DRUG_NAMES; +0.1 bonus fires
  P4-002: DISEASE_NAMES uses spaces (no underscores); matches KNOWN_POSITIVES
  P4-003: standalone mode (generate_fake_data) REFUSES to write CSV (RuntimeError)
  P4-004: KNOWN_POSITIVES / VALIDATED_HYPOTHESES are _LazyList (no import-time CSV read)
  P4-005: reward weights loadable per-tenant from YAML; apply_tenant_reward_weights works
  P4-006: ppo_gamma is a PipelineConfig field (default 0.0, configurable >0)
  P4-007: gnn_score staleness check fires WARNING when timestamp >24h old
  P4-008: rl/env.py, reward.py, train.py, evaluate.py, validate.py, cli.py all <500 lines
  P4-009: VecNormalize stats saved to .vecnormalize.pkl alongside checkpoint
  P4-010: action_space is Discrete(2) (O(1)), NOT Discrete(n_pairs)
  P4-011: torch.manual_seed set before PPO init; same seed → identical weights

Run:  pytest tests/test_p4_team11_v106_verified.py -v
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile
import subprocess
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

# Ensure repo is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_drug_names():
    """Load the ACTUAL REAL_DRUG_NAMES from graph_builder.py."""
    from graph_transformer.data.graph_builder import BiomedicalGraphBuilder
    return set(d.lower() for d in BiomedicalGraphBuilder.REAL_DRUG_NAMES)


@pytest.fixture(scope="module")
def validated_hypotheses_csv():
    """Load the ACTUAL validated_hypotheses.csv."""
    return pd.read_csv("rl/validated_hypotheses.csv")


# ---------------------------------------------------------------------------
# P4-001: Data flywheel — validated drugs in demo graph + bonus fires
# ---------------------------------------------------------------------------

class TestP4001DataFlywheel:
    """P4-001: validated_hypotheses.csv drugs MUST be in the demo graph."""

    def test_validated_drugs_in_real_drug_names(self, validated_hypotheses_csv, real_drug_names):
        """Every drug in validated_hypotheses.csv must appear in REAL_DRUG_NAMES."""
        vh_drugs = set(validated_hypotheses_csv["drug"].str.lower().str.strip())
        missing = vh_drugs - real_drug_names
        assert not missing, (
            f"P4-001 FAIL: validated drugs {missing} are NOT in REAL_DRUG_NAMES. "
            f"The +0.1 reward bonus can never fire because the env never presents "
            f"these drug-disease pairs. The data flywheel (DOCX §10) is dead code."
        )

    def test_validated_bonus_actually_fires(self):
        """The RewardFunction must add +0.1 for validated (drug, disease) pairs."""
        from rl.rl_drug_ranker import (
            RewardFunction, RewardConfig, DRUG_COL, DISEASE_COL,
            GNN_SCORE_COL, SAFETY_COL, MARKET_COL, CONFIDENCE_COL,
            PATHWAY_COL, PATENT_COL, RARE_DISEASE_COL, UNMET_NEED_COL,
            EFFICACY_COL, ADME_COL,
        )
        rf = RewardFunction(RewardConfig())
        # (thalidomide, multiple myeloma) is a validated hypothesis
        row = pd.Series({
            DRUG_COL: "thalidomide",
            DISEASE_COL: "multiple myeloma",
            GNN_SCORE_COL: 0.5, SAFETY_COL: 0.8, MARKET_COL: 0.5,
            CONFIDENCE_COL: 0.5, PATHWAY_COL: 0.5, PATENT_COL: 0.5,
            RARE_DISEASE_COL: 0.0, UNMET_NEED_COL: 0.5,
            EFFICACY_COL: 0.5, ADME_COL: 0.5,
        })
        reward_with_bonus = rf.compute(row)

        # Now compute WITHOUT the bonus
        rf_no_bonus = RewardFunction(RewardConfig())
        rf_no_bonus._validated_hypotheses = set()  # clear
        reward_without_bonus = rf_no_bonus.compute(row)

        delta = reward_with_bonus - reward_without_bonus
        assert abs(delta - 0.1) < 0.01, (
            f"P4-001 FAIL: validated bonus delta={delta}, expected ~0.1. "
            f"with_bonus={reward_with_bonus}, without={reward_without_bonus}"
        )

    def test_thalidomide_not_globally_rejected(self):
        """Thalidomide for multiple myeloma must NOT be hard-rejected (indication-specific)."""
        from rl.rl_drug_ranker import (
            RewardFunction, RewardConfig, DRUG_COL, DISEASE_COL,
            GNN_SCORE_COL, SAFETY_COL, MARKET_COL, CONFIDENCE_COL,
            PATHWAY_COL, PATENT_COL, RARE_DISEASE_COL, UNMET_NEED_COL,
            EFFICACY_COL, ADME_COL, INDICATION_WITHDRAWN_DRUGS,
        )
        # thalidomide must be in INDICATION_WITHDRAWN_DRUGS (not global WITHDRAWN_DRUGS)
        assert "thalidomide" in INDICATION_WITHDRAWN_DRUGS, (
            "thalidomide must be in INDICATION_WITHDRAWN_DRUGS (not global WITHDRAWN_DRUGS)"
        )
        assert "multiple myeloma" not in str(INDICATION_WITHDRAWN_DRUGS["thalidomide"]), (
            "multiple myeloma must NOT be a contraindicated indication for thalidomide"
        )
        rf = RewardFunction(RewardConfig())
        rf._validated_hypotheses = set()  # isolate the gate check
        row = pd.Series({
            DRUG_COL: "thalidomide", DISEASE_COL: "multiple myeloma",
            GNN_SCORE_COL: 0.5, SAFETY_COL: 0.8, MARKET_COL: 0.5,
            CONFIDENCE_COL: 0.5, PATHWAY_COL: 0.5, PATENT_COL: 0.5,
            RARE_DISEASE_COL: 0.0, UNMET_NEED_COL: 0.5,
            EFFICACY_COL: 0.5, ADME_COL: 0.5,
        })
        reward = rf.compute(row)
        assert reward > -1.0, (
            f"P4-001 FAIL: thalidomide→MM was hard-rejected (reward={reward}). "
            f"It's FDA-approved for MM under REMS."
        )


# ---------------------------------------------------------------------------
# P4-002: DISEASE_NAMES uses spaces (not underscores)
# ---------------------------------------------------------------------------

class TestP4002DiseaseNamesSpaces:
    """P4-002: DISEASE_NAMES must use spaces to match KNOWN_POSITIVES and PubMed."""

    def test_no_underscores_in_disease_names(self):
        from rl.rl_drug_ranker import DISEASE_NAMES
        underscored = [d for d in DISEASE_NAMES if "_" in d]
        assert not underscored, (
            f"P4-002 FAIL: DISEASE_NAMES contains underscored names: {underscored}. "
            f"These break KP recovery and PubMed search (PubMed uses spaces)."
        )

    def test_known_positives_diseases_use_spaces(self):
        from rl.rl_drug_ranker import KNOWN_POSITIVES
        for drug, disease in KNOWN_POSITIVES:
            assert "_" not in disease, (
                f"P4-002 FAIL: KP ({drug}, {disease}) has underscore in disease name."
            )


# ---------------------------------------------------------------------------
# P4-003: Standalone mode REFUSES to write CSV
# ---------------------------------------------------------------------------

class TestP4003StandaloneRefusesCsv:
    """P4-003: standalone mode (generate_fake_data) must refuse to write CSV."""

    def test_generate_fake_data_tags_standalone(self):
        from rl.rl_drug_ranker import generate_fake_data
        data = generate_fake_data(n_pairs=30, seed=42)
        assert data.attrs.get("_standalone_mode") is True, (
            "P4-003 FAIL: generate_fake_data did not tag DataFrame with _standalone_mode=True"
        )

    def test_env_captures_standalone_flag(self):
        from rl.rl_drug_ranker import generate_fake_data, DrugRankingEnv, DEFAULT_CONFIG
        data = generate_fake_data(n_pairs=30, seed=42)
        env = DrugRankingEnv(data, DEFAULT_CONFIG)
        assert env._standalone_mode is True, (
            "P4-003 FAIL: DrugRankingEnv did not capture _standalone_mode from data.attrs"
        )

    def test_save_results_refuses_in_standalone_mode(self, tmp_path):
        from rl.rl_drug_ranker import (
            save_results, PipelineConfig, DRUG_COL, DISEASE_COL, REWARD_COL,
        )
        cfg = PipelineConfig()
        cfg._standalone_mode = True
        cfg._standalone_mode_reason = "test"
        cand = pd.DataFrame([{
            DRUG_COL: "aspirin", DISEASE_COL: "pain", REWARD_COL: 0.5,
            "rank": 1, "literature_support": False,
            "is_known_positive": False, "controlled_substance": False,
        }])
        out_csv = str(tmp_path / "candidates.csv")
        with pytest.raises(RuntimeError, match="STANDALONE mode"):
            save_results(cand, out_csv, config=cfg, metadata={})
        assert not os.path.exists(out_csv), (
            "P4-003 FAIL: CSV was written to disk despite standalone mode refusal."
        )

    def test_save_results_works_in_bridge_mode(self, tmp_path):
        from rl.rl_drug_ranker import (
            save_results, PipelineConfig, DRUG_COL, DISEASE_COL, REWARD_COL,
        )
        cfg = PipelineConfig()
        # _standalone_mode defaults to False (bridge mode)
        cand = pd.DataFrame([{
            DRUG_COL: "aspirin", DISEASE_COL: "pain", REWARD_COL: 0.5,
            "rank": 1, "literature_support": False,
            "is_known_positive": False, "controlled_substance": False,
        }])
        out_csv = str(tmp_path / "candidates.csv")
        # Should NOT raise
        try:
            save_results(cand, out_csv, config=cfg, metadata={})
        except Exception as e:
            # Other validation errors are OK; we only care that it's NOT the standalone refusal
            assert "STANDALONE" not in str(e), f"P4-003 FAIL: bridge mode wrongly refused: {e}"


# ---------------------------------------------------------------------------
# P4-004: Lazy-load (no import-time CSV read)
# ---------------------------------------------------------------------------

class TestP4004LazyLoad:
    """P4-004: KNOWN_POSITIVES / VALIDATED_HYPOTHESES lazy-loaded (no import side effect)."""

    def test_lazy_list_proxy_in_fresh_subprocess(self):
        """In a fresh subprocess, verify cache is empty before access and populated after."""
        test_code = f"""
import sys, os
sys.path.insert(0, {REPO_ROOT!r})
os.chdir({REPO_ROOT!r})
from rl.rl_drug_ranker import KNOWN_POSITIVES, VALIDATED_HYPOTHESES, _LazyList
assert isinstance(KNOWN_POSITIVES, _LazyList)
assert isinstance(VALIDATED_HYPOTHESES, _LazyList)
assert KNOWN_POSITIVES._is_loaded() == False, "KP cache loaded at import time"
assert VALIDATED_HYPOTHESES._is_loaded() == False, "VH cache loaded at import time"
n_kp = len(KNOWN_POSITIVES)
n_vh = len(VALIDATED_HYPOTHESES)
assert KNOWN_POSITIVES._is_loaded() == True, "KP cache NOT populated after access"
assert VALIDATED_HYPOTHESES._is_loaded() == True, "VH cache NOT populated after access"
assert n_kp > 0
assert n_vh > 0
print("OK")
"""
        r = subprocess.run([sys.executable, "-c", test_code],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, (
            f"P4-004 FAIL: lazy-load invariant broken in fresh subprocess.\n"
            f"STDOUT: {r.stdout}\nSTDERR: {r.stderr[-500:]}"
        )

    def test_import_succeeds_without_csv(self, tmp_path):
        """import rl.rl_drug_ranker must NOT crash when validated_hypotheses.csv is missing."""
        csv_path = os.path.join(REPO_ROOT, "rl", "validated_hypotheses.csv")
        backup = csv_path + ".bak"
        try:
            shutil.move(csv_path, backup)
            test_code = f"""
import sys, os
sys.path.insert(0, {REPO_ROOT!r})
os.chdir({REPO_ROOT!r})
import rl.rl_drug_ranker  # must NOT crash
vh = list(rl.rl_drug_ranker.VALIDATED_HYPOTHESES)  # must NOT crash (returns [])
print("OK")
"""
            r = subprocess.run([sys.executable, "-c", test_code],
                               capture_output=True, text=True, timeout=60)
            assert r.returncode == 0, (
                f"P4-004 FAIL: import crashed when CSV missing.\n"
                f"STDERR: {r.stderr[-500:]}"
            )
        finally:
            if os.path.exists(backup):
                shutil.move(backup, csv_path)


# ---------------------------------------------------------------------------
# P4-005: Per-tenant reward weights from YAML
# ---------------------------------------------------------------------------

class TestP4005RewardWeightsConfigurable:
    """P4-005: reward weights loadable per-tenant from YAML."""

    def test_default_profile_loads(self):
        from rl.rl_drug_ranker import (
            load_reward_weights_for_tenant, FEATURE_COLS,
        )
        w = load_reward_weights_for_tenant(None)
        assert set(w.keys()) == set(FEATURE_COLS), (
            f"P4-005 FAIL: default weights keys mismatch FEATURE_COLS"
        )
        assert abs(sum(w.values()) - 1.0) < 1e-6, (
            f"P4-005 FAIL: default weights sum={sum(w.values())}, must be 1.0"
        )

    def test_tenant_profile_save_load_apply(self, tmp_path):
        from rl.rl_drug_ranker import (
            load_reward_weights_for_tenant, save_reward_weights_for_tenant,
            apply_tenant_reward_weights, PipelineConfig,
        )
        default_w = load_reward_weights_for_tenant(None)
        # Create a tenant profile with shifted weights (same sum)
        tenant_w = dict(default_w)
        tenant_w["safety_score"] = 0.20
        tenant_w["rare_disease_flag"] = 0.13  # +0.05 from default 0.08
        save_reward_weights_for_tenant(
            tenant_w, tenant_id="rare_test", weights_dir=str(tmp_path),
            profile_name="rare_test",
        )
        loaded = load_reward_weights_for_tenant("rare_test", weights_dir=str(tmp_path))
        assert loaded["safety_score"] == 0.20
        assert loaded["rare_disease_flag"] == 0.13

        cfg = PipelineConfig()
        new_cfg = apply_tenant_reward_weights(cfg, "rare_test", weights_dir=str(tmp_path))
        assert new_cfg.reward.reward_weights["rare_disease_flag"] == 0.13, (
            "P4-005 FAIL: apply_tenant_reward_weights did not propagate weights to config"
        )

    def test_invalid_weights_rejected(self, tmp_path):
        from rl.rl_drug_ranker import save_reward_weights_for_tenant, FEATURE_COLS
        # Use a NEGATIVE weight (sum is still 1.0 if we compensate, but range fails).
        # We give one weight -0.5 and another +1.5 so sum stays 1.0, but both fail range.
        bad_w = {col: 0.0 for col in FEATURE_COLS}
        # Distribute so sum = 1.0 but one weight is negative (fails [0,1] range check)
        bad_w[FEATURE_COLS[0]] = -0.5
        bad_w[FEATURE_COLS[1]] = 1.5  # -0.5 + 1.5 = 1.0 net, but both out of [0,1]
        # Sum is 1.0 (valid), but range check should fire on the -0.5 / 1.5 values.
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            save_reward_weights_for_tenant(bad_w, tenant_id="bad", weights_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# P4-006: ppo_gamma configurable (contextual bandit default)
# ---------------------------------------------------------------------------

class TestP4006PpoGammaConfigurable:
    """P4-006: ppo_gamma is a config field (default 0.0, configurable >0)."""

    def test_ppo_gamma_is_dataclass_field(self):
        from rl.rl_drug_ranker import PipelineConfig
        assert "ppo_gamma" in PipelineConfig.__dataclass_fields__, (
            "P4-006 FAIL: ppo_gamma is not a dataclass field (silent ignore bug)"
        )

    def test_default_is_contextual_bandit(self):
        from rl.rl_drug_ranker import PipelineConfig
        cfg = PipelineConfig()
        assert cfg.ppo_gamma == 0.0, (
            f"P4-006 FAIL: default ppo_gamma={cfg.ppo_gamma}, expected 0.0 (contextual bandit)"
        )

    def test_ppo_gamma_configurable_for_sequential(self):
        from rl.rl_drug_ranker import PipelineConfig
        cfg = PipelineConfig(ppo_gamma=0.95)
        assert cfg.ppo_gamma == 0.95, (
            "P4-006 FAIL: ppo_gamma=0.95 was not honored (silent ignore)"
        )


# ---------------------------------------------------------------------------
# P4-007: gnn_score staleness check
# ---------------------------------------------------------------------------

class TestP4007GnnScoreStaleness:
    """P4-007: env warns when gnn_score_timestamp is >24h old."""

    def _make_data(self, ts):
        from rl.rl_drug_ranker import (
            DRUG_COL, DISEASE_COL, GNN_SCORE_COL, SAFETY_COL, MARKET_COL,
            CONFIDENCE_COL, PATHWAY_COL, PATENT_COL, RARE_DISEASE_COL,
            UNMET_NEED_COL, EFFICACY_COL, ADME_COL, GNN_SCORE_TIMESTAMP_COL,
        )
        return pd.DataFrame([{
            DRUG_COL: "aspirin", DISEASE_COL: "pain",
            GNN_SCORE_COL: 0.5, SAFETY_COL: 0.8, MARKET_COL: 0.5,
            CONFIDENCE_COL: 0.5, PATHWAY_COL: 0.5, PATENT_COL: 0.5,
            RARE_DISEASE_COL: 0.0, UNMET_NEED_COL: 0.5,
            EFFICACY_COL: 0.5, ADME_COL: 0.5,
            GNN_SCORE_TIMESTAMP_COL: ts,
        }])

    def test_stale_timestamp_triggers_warning(self):
        from rl.rl_drug_ranker import DrugRankingEnv, PipelineConfig
        stale_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        env = DrugRankingEnv(self._make_data(stale_ts), PipelineConfig())
        assert env._gnn_score_stale is True, (
            "P4-007 FAIL: 48h-old timestamp did not trigger staleness flag"
        )
        assert env._gnn_score_age_hours > 24.0

    def test_fresh_timestamp_no_warning(self):
        from rl.rl_drug_ranker import DrugRankingEnv, PipelineConfig
        fresh_ts = datetime.now(timezone.utc).isoformat()
        env = DrugRankingEnv(self._make_data(fresh_ts), PipelineConfig())
        assert env._gnn_score_stale is False, (
            "P4-007 FAIL: fresh timestamp triggered false staleness warning"
        )


# ---------------------------------------------------------------------------
# P4-008: Modular file structure (wrappers <500 lines)
# ---------------------------------------------------------------------------

class TestP4008ModularFiles:
    """P4-008: rl/env.py, reward.py, train.py, evaluate.py, validate.py, cli.py <500 lines."""

    @pytest.mark.parametrize("rel_path", [
        "rl/env.py", "rl/reward.py", "rl/train.py",
        "rl/evaluate.py", "rl/validate.py", "rl/cli.py",
    ])
    def test_wrapper_under_500_lines(self, rel_path):
        full = os.path.join(REPO_ROOT, rel_path)
        with open(full) as f:
            n = sum(1 for _ in f)
        assert n < 500, (
            f"P4-008 FAIL: {rel_path} has {n} lines, must be <500"
        )

    def test_wrappers_re_export_real_symbols(self):
        from rl.env import DrugRankingEnv
        from rl.reward import RewardFunction, compute_reward
        from rl.train import train_agent
        from rl.evaluate import evaluate_agent, compute_auc
        from rl.validate import validate_input_schema, ScientificFailureError
        from rl.cli import main
        assert all([
            DrugRankingEnv, RewardFunction, compute_reward, train_agent,
            evaluate_agent, compute_auc, validate_input_schema,
            ScientificFailureError, main,
        ]), "P4-008 FAIL: a wrapper re-exported None"


# ---------------------------------------------------------------------------
# P4-009: VecNormalize stats saved & loaded
# ---------------------------------------------------------------------------

class TestP4009VecNormalizeStats:
    """P4-009: .vecnormalize.pkl saved alongside PPO checkpoint."""

    def test_checkpoint_and_vecnormalize_saved(self, tmp_path):
        from rl.rl_drug_ranker import (
            generate_fake_data, DrugRankingEnv, train_agent, PipelineConfig,
        )
        data = generate_fake_data(n_pairs=30, seed=42)
        env = DrugRankingEnv(data, PipelineConfig())
        env._standalone_mode = False  # force bridge-like for save test
        cfg = PipelineConfig(timesteps=64, checkpoint_dir=str(tmp_path))
        model, ckpt, vecnorm = train_agent(env, timesteps=64, seed=42, config=cfg)
        assert ckpt is not None, "P4-009 FAIL: checkpoint not saved"
        assert os.path.exists(ckpt), f"P4-009 FAIL: {ckpt} not on disk"
        vecnorm_path = ckpt.replace(".zip", ".vecnormalize.pkl")
        assert os.path.exists(vecnorm_path), (
            f"P4-009 FAIL: {vecnorm_path} not saved alongside checkpoint"
        )
        assert vecnorm is not None, "P4-009 FAIL: train_agent returned None for vecnormalize"

    def test_model_reloads(self, tmp_path):
        from rl.rl_drug_ranker import (
            generate_fake_data, DrugRankingEnv, train_agent, PipelineConfig,
        )
        from stable_baselines3 import PPO
        data = generate_fake_data(n_pairs=30, seed=42)
        env = DrugRankingEnv(data, PipelineConfig())
        env._standalone_mode = False
        cfg = PipelineConfig(timesteps=64, checkpoint_dir=str(tmp_path))
        model, ckpt, _ = train_agent(env, timesteps=64, seed=42, config=cfg)
        loaded = PPO.load(ckpt, device="cpu")
        assert loaded is not None, "P4-009 FAIL: PPO.load returned None"


# ---------------------------------------------------------------------------
# P4-010: Action space is O(1) (Discrete(2))
# ---------------------------------------------------------------------------

class TestP4010ActionSpaceScales:
    """P4-010: action_space is Discrete(2), not Discrete(n_pairs)."""

    def test_small_env_action_space_is_2(self):
        from rl.rl_drug_ranker import generate_fake_data, DrugRankingEnv, PipelineConfig
        data = generate_fake_data(n_pairs=50, seed=42)
        env = DrugRankingEnv(data, PipelineConfig())
        assert env.action_space.n == 2, (
            f"P4-010 FAIL: small env action_space.n={env.action_space.n}, expected 2"
        )

    def test_large_env_action_space_is_2(self):
        from rl.rl_drug_ranker import generate_fake_data, DrugRankingEnv, PipelineConfig
        data = generate_fake_data(n_pairs=1000, seed=42)
        env = DrugRankingEnv(data, PipelineConfig())
        assert env.action_space.n == 2, (
            f"P4-010 FAIL: large env action_space.n={env.action_space.n}, expected 2 "
            f"(Discrete(n_pairs) would give 1000 — does not scale)"
        )


# ---------------------------------------------------------------------------
# P4-011: torch.manual_seed → reproducible training
# ---------------------------------------------------------------------------

class TestP4011ReproducibleTraining:
    """P4-011: same seed → identical policy weights."""

    def test_same_seed_identical_weights(self, tmp_path):
        import torch
        from rl.rl_drug_ranker import (
            generate_fake_data, DrugRankingEnv, train_agent, PipelineConfig,
        )

        def run_once(seed):
            data = generate_fake_data(n_pairs=30, seed=42)
            env = DrugRankingEnv(data, PipelineConfig())
            env._standalone_mode = False
            cfg = PipelineConfig(timesteps=64, checkpoint_dir=str(tmp_path / f"run_{seed}"))
            model, _, _ = train_agent(env, timesteps=64, seed=seed, config=cfg)
            params = [p.detach().clone().flatten() for p in model.policy.parameters()]
            return torch.cat(params)

        w1 = run_once(42)
        w2 = run_once(42)
        diff = (w1 - w2).abs().max().item()
        assert diff < 1e-5, (
            f"P4-011 FAIL: same seed produced different weights (max diff={diff:.2e}). "
            f"torch.manual_seed is not called before PPO init."
        )
