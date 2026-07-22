"""Test for P4-002 ROOT FIX (CRITICAL).

P4-002: EV(always-HIGH) = +0.120 is POSITIVE — PPO collapses to
        always-HIGH. With high_action_bonus=5.0, bad_high_penalty_scale=0.30,
        correct_rejection_reward=0.05, low_action_penalty=1.0, and ~15%
        good pairs (avg reward 0.5):
          EV(always-HIGH) = 0.15*(0.5*5.0) + 0.85*(-1.0*0.30) = +0.120
          EV(always-LOW)  = 0.15*(-0.5*1.0) + 0.85*(0.05) = -0.0325
        PPO's value head is dead (P4-001), so the agent cannot learn to
        discriminate — it collapses to always-HIGH (the positive-EV
        default). The RL ranker adds NO ranking signal.

        Fix: raise bad_high_penalty_scale from 0.30 to 1.0 (full penalty
        for false HIGH). New EV(always-HIGH) = -0.475 (strongly negative),
        forcing PPO to discriminate.

This test verifies:
  1. RewardConfig.bad_high_penalty_scale default is 1.0 (NOT 0.30).
  2. The EV(always-HIGH) is NEGATIVE with the new defaults.
  3. The EV(always-LOW) is more negative than EV(always-HIGH) is positive
     (so PPO has a gradient toward discrimination, not toward either
     extreme).
  4. The EV(perfect) > EV(always-HIGH) and EV(perfect) > EV(always-LOW)
     (perfect play is the global optimum).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "rl"))

from rl.rl_drug_ranker import (
    RewardConfig,
    DrugRankingEnv,
    PipelineConfig,
    generate_fake_data,
    DRUG_COL,
    DISEASE_COL,
)


def _ev_analysis(cfg: RewardConfig, good_rate: float = 0.15, avg_good_reward: float = 0.5):
    """Compute EV(always-HIGH), EV(always-LOW), EV(perfect) for the reward config.

    Reward table (from DrugRankingEnv.step):
      action=HIGH, reward>0: final = reward * high_action_bonus
      action=HIGH, reward<=0: final = reward * bad_high_penalty_scale  (reward is negative)
      action=LOW,  reward>0: final = -reward * low_action_penalty
      action=LOW,  reward<=0: final = |reward| * correct_rejection_reward

    A "bad" pair has reward = -1.0 (the default for pairs that fail the
    reward function's gates). A "good" pair has reward = avg_good_reward
    (0.5 is the demo average).
    """
    good_high = avg_good_reward * cfg.high_action_bonus
    bad_high = -1.0 * cfg.bad_high_penalty_scale
    good_low = -avg_good_reward * cfg.low_action_penalty
    bad_low = 1.0 * cfg.correct_rejection_reward

    ev_always_high = good_rate * good_high + (1 - good_rate) * bad_high
    ev_always_low = good_rate * good_low + (1 - good_rate) * bad_low
    ev_perfect = good_rate * good_high + (1 - good_rate) * bad_low
    return ev_always_high, ev_always_low, ev_perfect


def test_p4_002_bad_high_penalty_scale_default_is_one():
    """RewardConfig.bad_high_penalty_scale must default to 1.0.

    The previous value 0.30 made EV(always-HIGH) POSITIVE (+0.120), so
    PPO collapsed to always-HIGH. The fix raises it to 1.0 (full penalty
    for false HIGH), making EV(always-HIGH) = -0.475 (strongly negative).
    """
    cfg = RewardConfig()
    assert cfg.bad_high_penalty_scale == 1.0, (
        f"P4-002 ROOT FIX FAILED: RewardConfig.bad_high_penalty_scale "
        f"default is {cfg.bad_high_penalty_scale}, expected 1.0. The "
        f"previous value 0.30 made EV(always-HIGH) = +0.120 (positive), "
        f"so PPO collapsed to always-HIGH. The fix sets it to 1.0 so "
        f"EV(always-HIGH) = -0.475 (strongly negative), forcing PPO to "
        f"discriminate."
    )


def test_p4_002_ev_always_high_is_negative():
    """EV(always-HIGH) must be NEGATIVE with the default reward config.

    With bad_high_penalty_scale=1.0 (the P4-002 fix), high_action_bonus=5.0,
    correct_rejection_reward=0.05, low_action_penalty=1.0, ~15% good pairs,
    avg good reward=0.5:
      EV(always-HIGH) = 0.15*(0.5*5.0) + 0.85*(-1.0*1.0)
                      = 0.375 - 0.85 = -0.475
    This is STRONGLY NEGATIVE, so PPO cannot default to always-HIGH.
    """
    cfg = RewardConfig()
    ev_high, ev_low, ev_perfect = _ev_analysis(cfg)
    assert ev_high < 0, (
        f"P4-002: EV(always-HIGH) = {ev_high:.4f} must be NEGATIVE. "
        f"With bad_high_penalty_scale={cfg.bad_high_penalty_scale}, "
        f"high_action_bonus={cfg.high_action_bonus}, "
        f"correct_rejection_reward={cfg.correct_rejection_reward}, "
        f"low_action_penalty={cfg.low_action_penalty}, the EV is "
        f"POSITIVE — PPO will collapse to always-HIGH. Raise "
        f"bad_high_penalty_scale to 1.0 (the P4-002 fix)."
    )


def test_p4_002_ev_perfect_greater_than_ev_always_high():
    """EV(perfect) > EV(always-HIGH) — PPO has a gradient toward discrimination.

    If EV(perfect) <= EV(always-HIGH), PPO has no incentive to learn to
    discriminate (always-HIGH is at least as good as perfect play).
    """
    cfg = RewardConfig()
    ev_high, ev_low, ev_perfect = _ev_analysis(cfg)
    assert ev_perfect > ev_high, (
        f"P4-002: EV(perfect)={ev_perfect:.4f} must be > EV(always-HIGH)"
        f"={ev_high:.4f}. If perfect play is not strictly better than "
        f"always-HIGH, PPO has no gradient toward discrimination."
    )
    # The gap should be substantial (not just barely better).
    gap = ev_perfect - ev_high
    assert gap > 0.3, (
        f"P4-002: the gap between EV(perfect) ({ev_perfect:.4f}) and "
        f"EV(always-HIGH) ({ev_high:.4f}) is only {gap:.4f}, which is too "
        f"small for PPO to reliably climb. Expected gap > 0.3."
    )


def test_p4_002_ev_perfect_greater_than_ev_always_low():
    """EV(perfect) > EV(always-LOW) — PPO has a gradient away from always-LOW too."""
    cfg = RewardConfig()
    ev_high, ev_low, ev_perfect = _ev_analysis(cfg)
    assert ev_perfect > ev_low, (
        f"P4-002: EV(perfect)={ev_perfect:.4f} must be > EV(always-LOW)"
        f"={ev_low:.4f}. If perfect play is not strictly better than "
        f"always-LOW, PPO has no gradient away from always-LOW."
    )


def test_p4_002_step_rewards_match_ev_model():
    """Verify the actual step() rewards match the EV model.

    Build a controlled dataset with a KNOWN mix of good pairs (reward > 0)
    and bad pairs (reward = -1.0 from failing the safety gate). The reward
    function's gates (safety_hard_reject, gnn_hard_reject) return -1.0 for
    pairs that fail. We construct pairs that fail the safety gate (safety
    below safety_hard_reject) to get the -1.0 reward, and pairs that pass
    all gates to get a positive reward.

    With bad_high_penalty_scale=1.0 (the P4-002 fix), always-HIGH on a
    mix of 85% bad pairs (reward=-1.0) and 15% good pairs (reward=+0.5)
    should yield a NEGATIVE average reward.
    """
    cfg = PipelineConfig(timesteps=10, top_n=5, n_pairs=50)
    # Build a controlled dataset: 5 good pairs (safety > safety_warning)
    # and 25 bad pairs (safety < safety_hard_reject, which makes reward=-1.0).
    # The reward function returns -1.0 for pairs failing the safety gate.
    n_good = 5
    n_bad = 25
    rows = []
    for i in range(n_good):
        # Good pair: high gnn_score (passes gnn gate), high safety (passes
        # safety gate), moderate other features → positive reward.
        rows.append({
            DRUG_COL: f"good_drug_{i}",
            DISEASE_COL: f"good_disease_{i}",
            "gnn_score": 0.8,
            "safety_score": 0.9,  # above safety_warning (0.7)
            "market_score": 0.5,
            "confidence": 0.7,
            "pathway_score": 0.6,
            "patent_score": 0.5,
            "rare_disease_flag": 0.0,
            "unmet_need_score": 0.5,
            "efficacy_score": 0.6,
            "adme_score": 0.7,
        })
    for i in range(n_bad):
        # Bad pair: low safety (fails safety gate) → reward = -1.0.
        rows.append({
            DRUG_COL: f"bad_drug_{i}",
            DISEASE_COL: f"bad_disease_{i}",
            "gnn_score": 0.8,  # passes gnn gate
            "safety_score": 0.05,  # below safety_hard_reject (0.1) → reward=-1.0
            "market_score": 0.5,
            "confidence": 0.7,
            "pathway_score": 0.6,
            "patent_score": 0.5,
            "rare_disease_flag": 0.0,
            "unmet_need_score": 0.5,
            "efficacy_score": 0.6,
            "adme_score": 0.7,
        })
    data = pd.DataFrame(rows)
    env = DrugRankingEnv(data, config=cfg, set_adaptive_threshold=True)

    # Reset without shuffle so we can iterate deterministically.
    obs, _ = env.reset(seed=42, options={"shuffle": False})

    # Run always-HIGH and collect rewards.
    high_rewards = []
    done = False
    while not done:
        obs, reward, done, _, _ = env.step(1)  # always HIGH
        high_rewards.append(reward)

    # Reset and run always-LOW.
    obs, _ = env.reset(seed=42, options={"shuffle": False})
    low_rewards = []
    done = False
    while not done:
        obs, reward, done, _, _ = env.step(0)  # always LOW
        low_rewards.append(reward)

    avg_high = float(np.mean(high_rewards))
    avg_low = float(np.mean(low_rewards))

    # With 5 good pairs (reward ~0.5*5.0=2.5 each) and 25 bad pairs
    # (reward=-1.0*1.0=-1.0 each), avg_high should be:
    #   (5*2.5 + 25*(-1.0)) / 30 = (12.5 - 25) / 30 = -0.4167
    # This is NEGATIVE, proving the P4-002 fix makes always-HIGH unprofitable.
    assert avg_high < 0, (
        f"P4-002: avg reward for always-HIGH = {avg_high:.4f} must be "
        f"NEGATIVE. With bad_high_penalty_scale=1.0 (the P4-002 fix), "
        f"the false-HIGH penalty dominates the true-HIGH bonus when bad "
        f"pairs are common. If avg_high is positive, "
        f"bad_high_penalty_scale is still too low (the P4-002 fix did "
        f"not take effect)."
    )
    # Sanity: always-LOW should be near zero or slightly negative (the
    # good pairs penalize LOW, the bad pairs reward LOW slightly).
    # With 5 good pairs (-0.5 each) and 25 bad pairs (+0.05 each):
    #   (5*(-0.5) + 25*0.05) / 30 = (-2.5 + 1.25) / 30 = -0.0417
    # Slightly negative — PPO should learn to discriminate (say HIGH on
    # good pairs, LOW on bad pairs) for the +0.4175/pair perfect-play EV.
    assert avg_low > avg_high, (
        f"P4-002: avg always-LOW ({avg_low:.4f}) should be > avg always-HIGH "
        f"({avg_high:.4f}). If always-HIGH beats always-LOW, PPO collapses "
        f"to always-HIGH (the P4-002 bug)."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
