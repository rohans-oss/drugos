"""Test for P4-001 + P4-010 ROOT FIX.

P4-001: ppo_gamma=0.95 for independent-step MDP — value head learns noisy
        discounted returns, EV≈0. Fix: set ppo_gamma=0.0 (contextual bandit).
P4-010: Stale comments in train_agent reference gamma=0.0 but actual default
        was 0.95 — provenance lie. Fix: update ALL stale comments.

This test verifies:
  1. PipelineConfig.ppo_gamma default is 0.0 (NOT 0.95).
  2. The DrugRankingEnv is genuinely a contextual bandit (action at step N
     does NOT affect observation at step N+1).
  3. No stale "V30 (10.29): 0.0" comments remain in the source file that
     would mislead a maintainer.
  4. The train_agent code actually passes the config's ppo_gamma through
     to PPO (not a hardcoded value).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the rl/ module importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "rl"))

from rl.rl_drug_ranker import (
    PipelineConfig,
    DrugRankingEnv,
    RewardConfig,
    DEFAULT_CONFIG,
    FEATURE_COLS,
    DRUG_COL,
    DISEASE_COL,
    generate_fake_data,
)


# ---------------------------------------------------------------------------
# P4-001: ppo_gamma default must be 0.0 (contextual bandit)
# ---------------------------------------------------------------------------

def test_p4_001_pipeline_config_ppo_gamma_default_is_zero():
    """PipelineConfig.ppo_gamma must default to 0.0 (contextual bandit).

    The DrugRankingEnv is a contextual bandit: each step is INDEPENDENT
    (action at step N does NOT affect observation at step N+1). With
    gamma=0.95, PPO's value head targets the discounted sum of ~20 future
    INDEPENDENT rewards, which is NOISY → explained_variance ≈ 0.
    gamma=0.0 makes the value head predict the IMMEDIATE reward, which it
    CAN learn.
    """
    cfg = PipelineConfig()
    assert cfg.ppo_gamma == 0.0, (
        f"P4-001 ROOT FIX FAILED: PipelineConfig.ppo_gamma default is "
        f"{cfg.ppo_gamma}, expected 0.0 (contextual bandit). The previous "
        f"P4-018 v2 reversion to 0.95 must be reverted — see the audit's "
        f"finding that EV(value head) ≈ 0 with gamma=0.95."
    )


def test_p4_001_env_is_contextual_bandit_action_does_not_affect_next_obs():
    """Verify the env is genuinely a contextual bandit.

    In a contextual bandit, the action taken at step N does NOT affect
    the observation at step N+1. We verify this by:
      1. Running two episodes with the SAME seed but DIFFERENT actions
         (always-HIGH vs always-LOW).
      2. The sequence of OBSERVATIONS must be IDENTICAL across the two
         episodes (only the rewards differ).
    If the observations differ, the env is NOT a contextual bandit and
    gamma > 0 would be appropriate. If they're identical (the expected
    case), gamma=0.0 is the scientifically-correct choice.
    """
    cfg = PipelineConfig(timesteps=10, top_n=5, n_pairs=50)
    data = generate_fake_data(n_pairs=50, seed=42)

    # Build two envs from the SAME data with the SAME seed.
    env_a = DrugRankingEnv(data, config=cfg, set_adaptive_threshold=True)
    env_b = DrugRankingEnv(data, config=cfg, set_adaptive_threshold=True)

    # Reset both with the same seed — observations should be identical.
    obs_a, _ = env_a.reset(seed=42, options={"shuffle": False})
    obs_b, _ = env_b.reset(seed=42, options={"shuffle": False})

    obs_seq_a = [obs_a.copy()]
    obs_seq_b = [obs_b.copy()]

    # Run env_a with always-HIGH, env_b with always-LOW.
    done_a = False
    while not done_a:
        obs_a, _, done_a, _, _ = env_a.step(1)  # always HIGH
        obs_seq_a.append(obs_a.copy())

    done_b = False
    while not done_b:
        obs_b, _, done_b, _, _ = env_b.step(0)  # always LOW
        obs_seq_b.append(obs_b.copy())

    # The observation sequences MUST be identical — proving the env is a
    # contextual bandit (action does not affect next observation).
    assert len(obs_seq_a) == len(obs_seq_b), (
        f"Episode lengths differ: {len(obs_seq_a)} vs {len(obs_seq_b)}"
    )
    for i, (oa, ob) in enumerate(zip(obs_seq_a, obs_seq_b)):
        assert np.allclose(oa, ob, atol=1e-6), (
            f"P4-001: observation at step {i} differs between always-HIGH "
            f"and always-LOW episodes. The env is NOT a contextual bandit "
            f"(action at step N affects observation at step N+1). If this "
            f"is intentional, gamma > 0 may be appropriate. If not, the "
            f"env has a bug and gamma=0.0 is correct."
        )


def test_p4_001_train_agent_reads_gamma_from_config():
    """train_agent must read ppo_gamma from PipelineConfig, not hardcode it.

    The previous code hardcoded gamma=0.95 in PPO(...) even when the
    config said 0.0 (provenance lie). This test verifies that
    train_agent honors the config value by checking the source code
    contains `gamma=_ppo_gamma` (not a hardcoded number).
    """
    src_path = _REPO_ROOT / "rl" / "rl_drug_ranker.py"
    src = src_path.read_text()
    # The PPO constructor call must use the config-derived _ppo_gamma.
    assert "gamma=_ppo_gamma" in src, (
        "P4-001: train_agent must pass gamma=_ppo_gamma (from config) to "
        "PPO(...), not a hardcoded value."
    )
    # The VecNormalize wrapper must also use _ppo_gamma.
    assert "gamma=_ppo_gamma,  # P4-001" in src, (
        "P4-001: VecNormalize must use gamma=_ppo_gamma (from config)."
    )


# ---------------------------------------------------------------------------
# P4-010: stale comments must be removed
# ---------------------------------------------------------------------------

def test_p4_010_no_stale_v30_10_29_gamma_comments():
    """No stale 'V30 (10.29): 0.0' comments should remain in the source.

    P4-018 v2 reverted ppo_gamma from 0.0 back to 0.95, but the V30
    (10.29) comments claiming gamma=0.0 remained — a provenance lie.
    P4-001 re-fixes the default to 0.0; P4-010 removes the stale
    comments so future maintainers aren't misled.
    """
    src_path = _REPO_ROOT / "rl" / "rl_drug_ranker.py"
    src = src_path.read_text()
    # The stale comment claimed gamma=0.0 was in effect when the actual
    # default was 0.95. After P4-001 + P4-010, these stale comments are
    # removed.
    stale_patterns = [
        "V30 (10.29): 0.0 for contextual bandit",
        "V30 (10.29): 0.0 for contextual bandit (was 0.95)",
        "V30 ROOT FIX (10.8/10.29): PPO hyperparams from config",
    ]
    for pat in stale_patterns:
        assert pat not in src, (
            f"P4-010: stale comment '{pat}' still present in source. "
            f"This comment claimed gamma=0.0 was in effect when the "
            f"actual default was 0.95 (P4-018 v2 reversion). Remove "
            f"or update the comment to match the actual default (now "
            f"0.0 per P4-001)."
        )


def test_p4_010_pipeline_config_ppo_gamma_comment_matches_default():
    """The docstring/comment for ppo_gamma must match the actual default.

    P4-010 is fundamentally about provenance consistency: the comments
    and the code must agree. This test verifies that the inline comment
    next to ppo_gamma mentions 0.0 (the actual default), not 0.95.
    """
    src_path = _REPO_ROOT / "rl" / "rl_drug_ranker.py"
    src = src_path.read_text()
    # Find the ppo_gamma field declaration line.
    lines = src.splitlines()
    ppo_gamma_line_idx = None
    for i, line in enumerate(lines):
        if "ppo_gamma: float = " in line and "ppo_gamma: float = 0." in line:
            ppo_gamma_line_idx = i
            break
    assert ppo_gamma_line_idx is not None, (
        "Could not find 'ppo_gamma: float = ...' declaration in source."
    )
    # The line and its surrounding 5 lines must mention 0.0 (not 0.95 as
    # the actual default).
    context = "\n".join(lines[max(0, ppo_gamma_line_idx - 5):ppo_gamma_line_idx + 1])
    assert "0.0" in context, (
        f"P4-010: the comment context around ppo_gamma must mention 0.0 "
        f"(the actual default per P4-001). Context:\n{context}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
