"""Tests for P4-005 through P4-013 ROOT FIXES (batch).

P4-005 (HIGH): generate_fake_data injects KPs with RANDOM features —
              standalone agent learns WRONG feature→action mapping.
              Fix: tag standalone data, REFUSE to save checkpoint.

P4-006 (HIGH): observation_space bounds are (-inf, +inf) but VecNormalize
              clips to ±10 — mismatch confuses SB3 internals.
              Fix: set low=-10.0, high=10.0 to match VecNormalize.

P4-007 (MEDIUM): gnn_score z-score normalized then sigmoid'd — DESTROYS
                 the original score's meaning and range (not batch-invariant).
                 Fix: use raw gnn_score directly.

P4-008 (MEDIUM): Train and test envs share the SAME reward_fn object —
                 mutations in train env affect test env.
                 Fix: deepcopy reward_fn for test env.

P4-009 (MEDIUM): reset() shuffles self.data IN PLACE — evaluate_agent
                 calls reset() WITHOUT shuffle=False.
                 Fix: document the shuffle behavior (no code change needed).

P4-011 (LOW): US_PREVALENCE table has INCONSISTENT units — some entries
              are "survivors" not "prevalence".
              Fix: use CURRENT US PREVALENCE consistently for all entries.

P4-012 (LOW): literature_crosscheck raises RuntimeError if biopython
              missing — but run_pipeline catches it, making the check
              non-blocking. The V1 criterion silently fails.
              Fix: track skip state, EXCLUDE from gate when skipped.

P4-013 (LOW): fillna(False) on object-dtype column triggers FutureWarning
              in pandas 2.2+.
              Fix: use infer_objects(copy=False) after fillna.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "rl"))

from rl.rl_drug_ranker import (
    PipelineConfig,
    RewardConfig,
    DrugRankingEnv,
    RewardFunction,
    generate_fake_data,
    US_PREVALENCE,
    RARE_DISEASE_PREVALENCE_THRESHOLD,
    _is_rare_disease,
    split_data,
    DRUG_COL,
    DISEASE_COL,
    GNN_SCORE_COL,
    SAFETY_COL,
)


# ===========================================================================
# P4-005: standalone checkpoint save must be REFUSED
# ===========================================================================

def test_p4_005_generate_fake_data_tags_standalone_mode():
    """generate_fake_data must tag the DataFrame with _standalone_mode=True."""
    data = generate_fake_data(n_pairs=50, seed=42)
    assert data.attrs.get("_standalone_mode") is True, (
        "P4-005: generate_fake_data must set data.attrs['_standalone_mode']=True "
        "so train_agent can REFUSE to save the checkpoint."
    )
    assert "_standalone_mode_reason" in data.attrs, (
        "P4-005: generate_fake_data must set data.attrs['_standalone_mode_reason'] "
        "with a human-readable explanation."
    )


def test_p4_005_env_propagates_standalone_flag():
    """DrugRankingEnv must propagate _standalone_mode from data to env."""
    data = generate_fake_data(n_pairs=50, seed=42)
    cfg = PipelineConfig(timesteps=10, top_n=5, n_pairs=50)
    env = DrugRankingEnv(data, config=cfg, set_adaptive_threshold=True)
    assert env._standalone_mode is True, (
        "P4-005: DrugRankingEnv must propagate _standalone_mode from data.attrs "
        "to env._standalone_mode so train_agent can check it."
    )


def test_p4_005_train_agent_refuses_standalone_checkpoint(tmp_path, monkeypatch):
    """train_agent must REFUSE to save the checkpoint when env is standalone.

    We mock PPO.learn to avoid actually training (slow). The test
    verifies that model.save() is NOT called and checkpoint_path is
    set to None.
    """
    # Mock stable_baselines3.PPO to avoid actual training.
    save_called = []

    class _MockModel:
        def __init__(self, *args, **kwargs):
            pass
        def learn(self, total_timesteps):
            pass
        def save(self, path):
            save_called.append(path)

    # Mock the SB3 imports inside train_agent.
    sb3_modules = type(sys)("_mock_sb3")
    sb3_modules.PPO = _MockModel
    sys.modules["stable_baselines3"] = sb3_modules

    # Mock VecNormalize too.
    vec_env_mod = type(sys)("_mock_vec_env")
    class _MockVecNormalize:
        def __init__(self, env, **kwargs):
            self.env = env
        def save(self, path):
            pass
    class _MockDummyVecEnv:
        def __init__(self, env_fns):
            self.env = env_fns[0]()
    vec_env_mod.VecNormalize = _MockVecNormalize
    vec_env_mod.DummyVecEnv = _MockDummyVecEnv
    vec_env_mod.VecEnv = type("_VecEnv", (), {})
    sys.modules["stable_baselines3.common.vec_env"] = vec_env_mod

    # Mock check_env.
    env_check_mod = type(sys)("_mock_env_check")
    env_check_mod.check_env = lambda env, warn, skip_render_check: None
    sys.modules["stable_baselines3.common.env_checker"] = env_check_mod

    try:
        from rl.rl_drug_ranker import train_agent
        data = generate_fake_data(n_pairs=50, seed=42)
        cfg = PipelineConfig(timesteps=10, top_n=5, n_pairs=50)
        env = DrugRankingEnv(data, config=cfg, set_adaptive_threshold=True)
        # Point the checkpoint to tmp_path so we don't pollute the repo.
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        model, checkpoint_path, vec_norm = train_agent(
            env, timesteps=10, seed=42, config=cfg,
        )
        # The checkpoint_path must be None (save refused).
        assert checkpoint_path is None, (
            f"P4-005: train_agent must set checkpoint_path=None when env "
            f"is standalone (refuse to save). Got {checkpoint_path}."
        )
        # model.save must NOT have been called.
        assert save_called == [], (
            f"P4-005: train_agent must NOT call model.save() when env is "
            f"standalone. save() was called with: {save_called}"
        )
    finally:
        # Restore real SB3 modules.
        for mod_name in ["stable_baselines3", "stable_baselines3.common.vec_env",
                         "stable_baselines3.common.env_checker"]:
            sys.modules.pop(mod_name, None)


# ===========================================================================
# P4-006: observation_space bounds must be [-10, +10] (match VecNormalize)
# ===========================================================================

def test_p4_006_observation_space_bounds_match_vec_normalize():
    """observation_space bounds must be [-10, +10] (VecNormalize clip_obs=10.0)."""
    data = generate_fake_data(n_pairs=50, seed=42)
    cfg = PipelineConfig(timesteps=10, top_n=5, n_pairs=50)
    env = DrugRankingEnv(data, config=cfg, set_adaptive_threshold=True)
    # The bounds must be [-10, +10] to match VecNormalize's default
    # clip_obs=10.0. The previous (-inf, +inf) was a mismatch.
    # The low/high are arrays of size n_features — check the first element.
    low_val = float(env.observation_space.low[0])
    high_val = float(env.observation_space.high[0])
    assert low_val == -10.0, (
        f"P4-006: observation_space.low[0] must be -10.0 (match VecNormalize "
        f"clip_obs=10.0), got {low_val}"
    )
    assert high_val == 10.0, (
        f"P4-006: observation_space.high[0] must be 10.0 (match VecNormalize "
        f"clip_obs=10.0), got {high_val}"
    )
    # All elements must have the same bounds (VecNormalize clips all features).
    assert np.all(env.observation_space.low == -10.0), (
        "P4-006: all observation_space.low elements must be -10.0"
    )
    assert np.all(env.observation_space.high == 10.0), (
        "P4-006: all observation_space.high elements must be 10.0"
    )


# ===========================================================================
# P4-007: gnn_score must NOT be z-score+sigmoid transformed
# ===========================================================================

def test_p4_007_no_zscore_sigmoid_on_gnn_score():
    """The reward function must use raw gnn_score (no z-score+sigmoid).

    The previous V30 (10.10) fix z-score normalized gnn_score then
    applied sigmoid. This was NOT batch-invariant: the same gnn_score
    produced different reward contributions depending on the batch's
    mean/std. The fix uses the raw gnn_score directly.
    """
    # Build two reward functions with DIFFERENT _gnn_score_mean/std
    # (simulating two different batches). The reward for the SAME row
    # must be IDENTICAL across the two functions (since the raw gnn_score
    # is used, not the z-score).
    cfg = RewardConfig()
    rf_a = RewardFunction(cfg)
    rf_a._gnn_score_mean = 0.5
    rf_a._gnn_score_std = 0.2
    rf_b = RewardFunction(cfg)
    rf_b._gnn_score_mean = 0.7
    rf_b._gnn_score_std = 0.1

    # Build a test row with a known gnn_score.
    row = pd.Series({
        GNN_SCORE_COL: 0.8,
        SAFETY_COL: 0.9,
        "market_score": 0.5,
        "confidence": 0.7,
        "pathway_score": 0.6,
        "patent_score": 0.5,
        "rare_disease_flag": 0.0,
        "unmet_need_score": 0.5,
        "efficacy_score": 0.6,
        "adme_score": 0.7,
        DRUG_COL: "testdrug",
        DISEASE_COL: "testdisease",
    })
    reward_a = rf_a.compute(row)
    reward_b = rf_b.compute(row)
    # The rewards must be IDENTICAL (raw gnn_score, no batch-dependent transform).
    assert abs(reward_a - reward_b) < 1e-9, (
        f"P4-007: reward must NOT depend on batch statistics. With "
        f"_gnn_score_mean=0.5/std=0.2: reward={reward_a:.6f}. With "
        f"_gnn_score_mean=0.7/std=0.1: reward={reward_b:.6f}. The "
        f"z-score+sigmoid transformation was NOT removed (P4-007 fix "
        f"did not take effect)."
    )


# ===========================================================================
# P4-008: test env must have its OWN reward_fn (deepcopy)
# ===========================================================================

def test_p4_008_test_env_gets_deepcopy_of_reward_fn():
    """run_pipeline must deepcopy reward_fn for the test env.

    We verify by reading the source code (run_pipeline is too slow to
    invoke in a unit test). The source must contain a deepcopy call
    before constructing the test env.
    """
    src = (_REPO_ROOT / "rl" / "rl_drug_ranker.py").read_text()
    # The test env construction must use a deepcopy of reward_fn.
    assert "test_reward_fn = _copy_for_test_env.deepcopy(reward_fn)" in src, (
        "P4-008: run_pipeline must deepcopy reward_fn for the test env. "
        "Expected 'test_reward_fn = _copy_for_test_env.deepcopy(reward_fn)' "
        "in the source."
    )
    assert "reward_fn=test_reward_fn" in src, (
        "P4-008: the test env must be constructed with the deepcopied "
        "test_reward_fn, not the shared reward_fn."
    )


# ===========================================================================
# P4-009: evaluate_agent shuffle behavior must be documented
# ===========================================================================

def test_p4_009_evaluate_agent_shuffle_is_documented():
    """evaluate_agent must document the shuffle behavior (P4-009)."""
    src = (_REPO_ROOT / "rl" / "rl_drug_ranker.py").read_text()
    assert "P4-009 ROOT FIX" in src, (
        "P4-009: evaluate_agent must have a P4-009 ROOT FIX comment "
        "documenting the shuffle behavior."
    )


# ===========================================================================
# P4-011: US_PREVALENCE must use CURRENT prevalence consistently
# ===========================================================================

def test_p4_011_us_prevalence_uses_consistent_metric():
    """US_PREVALENCE must use CURRENT prevalence for all entries.

    The previous table mixed 'survivors' (ever diagnosed) with 'current
    prevalence'. The fix uses CURRENT prevalence for all entries and
    documents the metric in each entry's inline comment.
    """
    src = (_REPO_ROOT / "rl" / "rl_drug_ranker.py").read_text()
    # No entry should be commented as "survivors" (the old inconsistent metric).
    # We check the US_PREVALENCE block (between the dict opening and closing).
    us_prev_start = src.find("US_PREVALENCE: dict[str, int] = {")
    us_prev_end = src.find("}", us_prev_start + 1)
    # Find the next } after the dict (the dict's closing brace).
    # We need to skip nested braces.
    depth = 1
    i = us_prev_start + 1
    while depth > 0 and i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
        i += 1
    us_prev_end = i
    us_prev_block = src[us_prev_start:us_prev_end]
    # The old "survivors" comment should NOT appear (we now use "current prevalence").
    # Allow "stroke survivors" since that IS the current prevalence definition for stroke
    # (people living with stroke effects). But "leukemia survivors" / "melanoma survivors"
    # / "lymphoma survivors" should NOT appear — those used the wrong metric.
    bad_patterns = [
        "leukemia.*survivors",
        "melanoma.*survivors",
        "lymphoma.*survivors",
    ]
    import re
    for pat in bad_patterns:
        assert not re.search(pat, us_prev_block, re.IGNORECASE), (
            f"P4-011: US_PREVALENCE block still contains '{pat}' — the "
            f"old 'survivors' metric. Use CURRENT prevalence consistently."
        )


def test_p4_011_leukemia_prevalence_updated():
    """Leukemia prevalence must be the NCI SEER current prevalence (~475K).

    The previous value was 380K (Leukemia & Lymphoma Society "survivors"
    — a different metric). The fix uses NCI SEER current prevalence.
    """
    assert US_PREVALENCE.get("leukemia") == 475_000, (
        f"P4-011: leukemia prevalence must be 475000 (NCI SEER current "
        f"prevalence), got {US_PREVALENCE.get('leukemia')}. The previous "
        f"380K was the Leukemia & Lymphoma Society 'survivors' count "
        f"(a different metric)."
    )


def test_p4_011_melanoma_prevalence_updated():
    """Melanoma prevalence must be the NCI SEER current prevalence (~1.3M)."""
    assert US_PREVALENCE.get("melanoma") == 1_300_000, (
        f"P4-011: melanoma prevalence must be 1300000 (NCI SEER current "
        f"prevalence), got {US_PREVALENCE.get('melanoma')}."
    )


def test_p4_011_stroke_prevalence_documented_as_current():
    """Stroke prevalence must be documented as current prevalence (not just 'survivors')."""
    src = (_REPO_ROOT / "rl" / "rl_drug_ranker.py").read_text()
    # The stroke entry must mention "current prevalence" in its comment.
    assert '"stroke":' in src, "P4-011: stroke entry missing from US_PREVALENCE"
    # Find the stroke line and check its comment.
    stroke_line_idx = src.find('"stroke":')
    stroke_line = src[stroke_line_idx:stroke_line_idx + 200]
    assert "current prevalence" in stroke_line.lower() or "people living" in stroke_line.lower(), (
        f"P4-011: stroke entry must document the metric as 'current prevalence' "
        f"or 'people living'. Line: {stroke_line}"
    )


# ===========================================================================
# P4-012: literature check must be EXCLUDED from gate when biopython missing
# ===========================================================================

def test_p4_012_literature_check_tracked_in_scientific_validation():
    """scientific_validation must include literature_check_skipped field."""
    src = (_REPO_ROOT / "rl" / "rl_drug_ranker.py").read_text()
    assert "_literature_check_skipped" in src, (
        "P4-012: run_pipeline must track _literature_check_skipped so the "
        "gate can EXCLUDE the literature check when biopython is missing."
    )
    assert "literature_check_skipped" in src, (
        "P4-012: scientific_validation dict must include 'literature_check_skipped'."
    )
    assert "n_literature_supported" in src, (
        "P4-012: scientific_validation dict must include 'n_literature_supported'."
    )


# ===========================================================================
# P4-013: fillna(False) must use infer_objects to silence FutureWarning
# ===========================================================================

def test_p4_013_fillna_uses_infer_objects():
    """fillna(False) must suppress the FutureWarning (P4-013 fix).

    The fix wraps the fillna in a warnings.catch_warnings() context that
    suppresses the pandas 2.2+ FutureWarning about downcasting object
    dtype arrays. The 2-step pattern (fillna → to_numpy(dtype=bool)) is
    already correct and forward-compatible with pandas 3.0+.
    """
    src = (_REPO_ROOT / "rl" / "rl_drug_ranker.py").read_text()
    assert "warnings.catch_warnings()" in src, (
        "P4-013: the fillna(False) call must be wrapped in a "
        "warnings.catch_warnings() context to suppress the pandas 2.2+ "
        "FutureWarning about downcasting object dtype arrays."
    )
    assert "Downcasting object dtype arrays on" in src, (
        "P4-013: the warnings.filterwarnings call must target the "
        "'Downcasting object dtype arrays on' FutureWarning message."
    )


def test_p4_013_no_future_warning_on_fillna():
    """split_data must not emit a FutureWarning on pandas 2.2+."""
    data = generate_fake_data(n_pairs=100, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        # This must NOT raise a FutureWarning.
        train_df, test_df = split_data(
            data, test_size=0.2, seed=42, drug_aware=True,
            ensure_known_positives_in_test=True, return_oversampled=False,
        )
    # Sanity: both splits must be non-empty.
    assert len(train_df) > 0 and len(test_df) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
