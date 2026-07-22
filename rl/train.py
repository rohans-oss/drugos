"""rl.train — PPO Training (P4-008 modular wrapper).

P4-008 ROOT FIX: modular wrapper around the train_agent function,
get_device helper, and the PPO hyperparameter config fields. See
rl/env.py for the full P4-008 rationale.

Callers can now import:
    from rl.train import train_agent, get_device
"""
from __future__ import annotations

from .rl_drug_ranker import (
    train_agent,
    get_device,
    # PipelineConfig carries the PPO hyperparams (ppo_learning_rate,
    # ppo_gamma, ppo_ent_coef, ppo_clip_range, ppo_net_arch, etc.)
    PipelineConfig,
    DEFAULT_CONFIG,
)

__all__ = [
    "train_agent",
    "get_device",
    "PipelineConfig",
    "DEFAULT_CONFIG",
]
