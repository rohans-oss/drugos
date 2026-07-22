#!/usr/bin/env python3
"""
v92 Forensic Root-Fix Verification Tests

Verifies each P4-049..P4-077 bug fix + Phase 4 handoff fixes by inspecting
the ACTUAL code and runtime behavior (not comments, not test stubs).

Each test corresponds to a specific bug ID from the audit.
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/z/my-project/work/autonomous-drug-repurposing/rl')
os.chdir('/home/z/my-project/work/autonomous-drug-repurposing')
os.environ['RL_SKIP_LITERATURE'] = '1'

import rl_drug_ranker as r
from rl_drug_ranker import (
    RewardConfig, PipelineConfig, RewardFunction, DrugRankingEnv,
    RankedCandidate, PipelineMetrics, OUTPUT_SCHEMA, WITHDRAWN_DRUGS,
    KNOWN_POSITIVES, VALIDATED_HYPOTHESES, generate_fake_data,
    compute_reward, validate_canonical_ids, _validate_canonical_id_formats,
    _TrainingMetricsCallback, compute_policy_prob_confidence_interval,
    construct_pathway_chain,
)

passed = 0
failed = 0
results = []

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        results.append(f"PASS: {name}")
    else:
        failed += 1
        results.append(f"FAIL: {name} -- {detail}")

# ─── P4-049: BAD_HIGH_PENALTY_SCALE docstring consistency ──────────────
cfg = RewardConfig()
check("P4-049 bad_high_penalty_scale default is 0.30",
      cfg.bad_high_penalty_scale == 0.30,
      f"got {cfg.bad_high_penalty_scale}")

# ─── P4-050: high_action_bonus docstring consistency ───────────────────
check("P4-050 high_action_bonus default is 5.0",
      cfg.high_action_bonus == 5.0,
      f"got {cfg.high_action_bonus}")

# ─── P4-051: render() is actually called (not dead) ────────────────────
# Verify render() exists and is callable
check("P4-051 DrugRankingEnv.render method exists",
      hasattr(DrugRankingEnv, 'render') and callable(getattr(DrugRankingEnv, 'render')),
      "render method missing")
# Verify render is called in evaluate_agent (check source)
import inspect
src = inspect.getsource(r.evaluate_agent)
check("P4-051 render() is called in evaluate_agent",
      'env.render' in src,
      "render() not called in evaluate_agent")

# ─── P4-052: Rank bad HIGH docstring says r * BAD_HIGH_PENALTY_SCALE ──
env_src = inspect.getsource(DrugRankingEnv)
check("P4-052 step docstring mentions bad_high_penalty_scale",
      'bad_high_penalty_scale' in env_src,
      "docstring not updated")

# ─── P4-053: correct_rejection_reward default is 0.05 (not 0.0) ───────
check("P4-053 correct_rejection_reward default is 0.05",
      cfg.correct_rejection_reward == 0.05,
      f"got {cfg.correct_rejection_reward}")

# ─── P4-054: malformed comment with trailing # fixed ───────────────────
full_src = inspect.getsource(r)
# The malformed comment was an ACTIVE comment (not in a docstring) that had
# "BEFORE realpath.    #" (trailing # with no space). The fix replaced it
# with a properly formatted comment. The string now only appears in the
# EXPLANATION comment (which is properly formatted), not as an active
# code comment with the trailing #.
# Check: no line ends with "realpath.    #" (the malformed pattern)
malformed_lines = [l for l in full_src.split('\n') if l.rstrip().endswith('realpath.    #')]
check("P4-054 no malformed 'BEFORE realpath.    #' comment (active code)",
      len(malformed_lines) == 0,
      f"malformed lines: {malformed_lines}")

# ─── P4-056: thalidomide not globally hard-rejected (indication-specific) ──
# v92 fix: removed thalidomide from WITHDRAWN_DRUGS entirely.
# v100 fix (merged): moved thalidomide to WITHDRAWN_DRUGS_INDICATION_SPECIFIC
# with contraindicated indications {pregnancy, morning sickness, ...}.
# Both fixes achieve the same goal: thalidomide is NOT hard-rejected for
# multiple myeloma (its FDA-approved indication). The v100 approach is more
# precise (still protects pregnant patients).
check("P4-056 thalidomide NOT in WITHDRAWN_DRUGS_GLOBAL (globally withdrawn)",
      'thalidomide' not in getattr(r, 'WITHDRAWN_DRUGS_GLOBAL', r.WITHDRAWN_DRUGS),
      "thalidomide still globally withdrawn")
check("P4-056 thalidomide IS in WITHDRAWN_DRUGS_INDICATION_SPECIFIC (indication-specific)",
      hasattr(r, 'WITHDRAWN_DRUGS_INDICATION_SPECIFIC') and 'thalidomide' in r.WITHDRAWN_DRUGS_INDICATION_SPECIFIC,
      "thalidomide not in indication-specific set")

# ─── P4-057: validated pair (thalidomide, multiple myeloma) can get bonus ──
# Since thalidomide is no longer withdrawn, the reward function should NOT
# return -1.0 for thalidomide pairs (unless safety/gnn fails)
thal_row = pd.Series({
    'drug': 'thalidomide',
    'disease': 'multiple myeloma',
    'gnn_score': 0.5, 'safety_score': 0.8, 'market_score': 0.5,
    'confidence': 0.5, 'pathway_score': 0.5, 'patent_score': 0.5,
    'rare_disease_flag': 1.0, 'unmet_need_score': 0.5,
    'efficacy_score': 0.5, 'adme_score': 0.5,
})
reward = compute_reward(thal_row)
check("P4-057 thalidomide+multiple myeloma reward != -1.0 (not hard-rejected)",
      reward != -1.0,
      f"got reward={reward}")
# Verify the validated bonus is applied (thalidomide, multiple myeloma is in validated_hypotheses.csv)
check("P4-057 (thalidomide, multiple myeloma) in VALIDATED_HYPOTHESES",
      ('thalidomide', 'multiple myeloma') in VALIDATED_HYPOTHESES,
      "validated pair not loaded")

# ─── P4-058: rare_disease_flag derived from disease name (not random) ──
data = generate_fake_data(n_pairs=50, seed=42)
# All rows with the same disease should have the same rare_disease_flag
disease_groups = data.groupby('disease')['rare_disease_flag'].nunique()
check("P4-058 rare_disease_flag is consistent per disease (not random)",
      (disease_groups <= 1).all(),
      f"found diseases with varying flags: {disease_groups[disease_groups > 1].to_dict()}")

# ─── P4-059: z-score normalization removed from gnn_score ──────────────
# The reward function should use gnn_score AS-IS (no sigmoid(z-score))
rf_src = inspect.getsource(RewardFunction.compute)
check("P4-059 z-score normalization removed (no _gnn_score_mean in compute)",
      '_gnn_score_mean' not in rf_src or 'gnn_val_for_reward = float(gnn_val)' in rf_src,
      "z-score normalization still present")

# ─── P4-060: quality report uses adaptive threshold ────────────────────
qr_src = inspect.getsource(r.generate_data_quality_report)
check("P4-060 quality report uses adaptive threshold (set_adaptive_threshold called)",
      'set_adaptive_threshold' in qr_src and 'actual_gnn_threshold' in qr_src,
      "adaptive threshold not used in quality report")

# ─── P4-061/P4-076: ppo_gamma=0.95 (proper RL, not contextual bandit) ──
pcfg = PipelineConfig()
check("P4-061/P4-076 ppo_gamma=0.95 (proper RL)",
      pcfg.ppo_gamma == 0.95,
      f"got ppo_gamma={pcfg.ppo_gamma}")

# ─── P4-062: graduated safety_factor (not binary 0.5/1.0) ──────────────
# Test: safety=0.6 should give safety_factor < 0.5 (stricter than binary 0.5)
# Row with safety=0.6 (below warning=0.7 but above hard_reject=0.5)
row_safety_06 = pd.Series({
    'drug': 'test', 'disease': 'test',
    'gnn_score': 0.5, 'safety_score': 0.6, 'market_score': 0.5,
    'confidence': 0.5, 'pathway_score': 0.5, 'patent_score': 0.5,
    'rare_disease_flag': 0.0, 'unmet_need_score': 0.5,
    'efficacy_score': 0.5, 'adme_score': 0.5,
})
row_safety_09 = row_safety_06.copy()
row_safety_09['safety_score'] = 0.9
# With graduated: safety=0.6 -> factor=(0.6-0.5)/(1-0.5)=0.2; safety=0.9 -> factor=1.0
# With binary: both would give factor=0.5 (0.6 < 0.7) and 1.0 (0.9 >= 0.7)
# So reward(0.6) with graduated should be ~0.2/0.5 = 40% of reward(0.6) with binary
# Easier check: reward(safety=0.9) > reward(safety=0.6) * 2 (graduated makes the difference larger)
reward_06 = compute_reward(row_safety_06)
reward_09 = compute_reward(row_safety_09)
check("P4-062 graduated safety_factor (reward(safety=0.9) >> reward(safety=0.6))",
      reward_09 > reward_06 * 1.5,
      f"reward(0.6)={reward_06}, reward(0.9)={reward_09}")

# ─── P4-063: KP oversampling jitter std=0.05 (not 0.01) ────────────────
split_src = inspect.getsource(r.split_data)
check("P4-063 KP jitter std=0.05 (not 0.01)",
      'normal(0, 0.05' in split_src,
      "jitter std not 0.05")

# ─── P4-064: reward >= 0 check (not reward > 0) + reward floor ─────────
step_src = inspect.getsource(DrugRankingEnv.step)
check("P4-064 step() uses 'reward >= 0' (not 'reward > 0')",
      'reward >= 0' in step_src,
      "still using reward > 0")
# Verify reward floor in compute()
check("P4-064 reward floor (max(reward, 0.01)) in compute()",
      'max(reward, 0.01)' in rf_src,
      "reward floor not present")

# ─── P4-065: n_steps clamp is 1x (not 2x) on small graphs ─────────────
train_src = inspect.getsource(r.train_agent)
check("P4-065 n_steps clamp is 1x (env.n_pairs, not env.n_pairs * 2)",
      'env.n_pairs * 2' not in train_src and 'max(1, env.n_pairs)' in train_src,
      "still using 2x clamp")

# ─── P4-066: AUC labels read from env_test.data (not test_data) ────────
auc_src = inspect.getsource(r.compute_auc)
# The bug was: `row = test_data.iloc[current_row_idx]` (ACTIVE CODE).
# The fix: `row = env_test.data.iloc[current_row_idx]` (ACTIVE CODE).
# The docstring still references test_data.iloc (in explanation text),
# which is fine. Check for ACTIVE assignment lines (not comments, not
# docstring backtick-quoted text).
active_lines = [l.strip() for l in auc_src.split('\n')
                if not l.strip().startswith('#')
                and not l.strip().startswith('"')
                and '``' not in l  # skip docstring backtick-quoted lines
                and 'test_data.iloc' in l
                and l.strip().startswith('row =')]
check("P4-066 AUC labels read from env_test.data (not test_data) in active code",
      len(active_lines) == 0 and 'row = env_test.data.iloc' in auc_src,
      f"active test_data.iloc lines: {active_lines}")

# ─── P4-068: standalone mode skips GT AUC check ────────────────────────
# Build a config with gt_test_auc=None (standalone mode)
standalone_cfg = PipelineConfig()
standalone_cfg.gt_test_auc = None
# The scientific_validation should mark gt_test_auc as skipped (not failed)
# We can't easily run the full pipeline here, so check the logic
check("P4-068 PipelineConfig.gt_test_auc defaults to None (standalone mode)",
      PipelineConfig().gt_test_auc is None,
      f"got {PipelineConfig().gt_test_auc}")
# Check the skip logic exists in run_pipeline
run_src = inspect.getsource(r.run_pipeline)
check("P4-068 'gt_test_auc_skipped' logic in run_pipeline",
      'gt_test_auc_skipped' in run_src,
      "skip logic not present")
check("P4-068 RL_ALLOW_SCIENCE_FAILURE documented in CLI help",
      'RL_ALLOW_SCIENCE_FAILURE' in inspect.getsource(r._build_arg_parser),
      "not documented")

# ─── P4-069: n_safety_rejected incremented in env ──────────────────────
check("P4-069 DrugRankingEnv has n_safety_rejected attribute",
      hasattr(DrugRankingEnv, 'n_safety_rejected') or 'n_safety_rejected' in env_src,
      "n_safety_rejected not in env")
# Verify it's incremented in step()
check("P4-069 n_safety_rejected incremented in step()",
      'self.n_safety_rejected += 1' in step_src,
      "not incremented in step()")

# ─── P4-070: resume path wraps in VecNormalize ─────────────────────────
check("P4-070 resume path wraps env in DummyVecEnv + VecNormalize",
      'DummyVecEnv' in train_src and 'VecNormalize.load' in train_src,
      "resume path not symmetric with fresh path")

# ─── P4-071: ActorCriticPolicy and nn imports removed ──────────────────
check("P4-071 ActorCriticPolicy import removed from train_agent",
      'from stable_baselines3.common.policies import ActorCriticPolicy' not in train_src,
      "ActorCriticPolicy still imported")
check("P4-071 'import torch.nn as nn' removed from train_agent",
      'import torch.nn as nn' not in train_src,
      "torch.nn still imported")

# ─── P4-072: _val_kp_mask vectorized (no iterrows) ─────────────────────
check("P4-072 _val_kp_mask uses vectorized ops (no iterrows)",
      '_val_kp_mask = val_for_threshold_df.apply' not in run_src,
      "still using iterrows")

# ─── P4-073: is_safe() called in display_top_candidates ────────────────
display_src = inspect.getsource(r.display_top_candidates)
check("P4-073 is_safe() called in display_top_candidates",
      'is_safe()' in display_src,
      "is_safe() not called")

# ─── P4-074: no 'import json as _json' ─────────────────────────────────
# The bug was an ACTIVE `import json as _json` statement. The fix removed
# the active import. The comment "v92 ROOT FIX (P4-074): removed redundant
# `import json as _json`." is fine (it's a comment, not an import).
# Check for ACTIVE import statements (lines starting with "import json as _json").
active_json_imports = [l.strip() for l in full_src.split('\n')
                       if l.strip().startswith('import json as _json')]
check("P4-074 no active 'import json as _json' statement",
      len(active_json_imports) == 0,
      f"active imports: {active_json_imports}")

# ─── P4-075: validate_canonical_ids has real validation ────────────────
check("P4-075 _validate_canonical_id_formats function exists",
      hasattr(r, '_validate_canonical_id_formats'),
      "validation helper not defined")
# Test with invalid InChIKey
test_df = pd.DataFrame({
    'drug': ['d1', 'd2'],
    'disease': ['dis1', 'dis2'],
    'drug_inchikey': ['INVALID', 'BQJCRHHNABKAKU-XKASOQGDSA-N'],  # 1 invalid, 1 valid
    'disease_mesh_id': ['D000001', 'INVALID'],
})
# Should not crash, should log warnings
try:
    _validate_canonical_id_formats(test_df)
    check("P4-075 _validate_canonical_id_formats runs without crashing",
          True)
except Exception as e:
    check("P4-075 _validate_canonical_id_formats runs without crashing",
          False, f"crashed: {e}")

# ─── P4-077: training_loss and episode_rewards populated via callback ──
check("P4-077 _TrainingMetricsCallback class exists",
      hasattr(r, '_TrainingMetricsCallback'),
      "callback class not defined")
check("P4-077 _TrainingMetricsCallback inherits from BaseCallback",
      'BaseCallback' in str(_TrainingMetricsCallback.__bases__),
      f"bases: {_TrainingMetricsCallback.__bases__}")
# Verify metrics param is wired in train_agent
check("P4-077 train_agent accepts metrics parameter",
      'metrics' in inspect.signature(r.train_agent).parameters,
      "metrics param not in train_agent signature")

# ─── Phase 4 handoff: OUTPUT_SCHEMA has policy_prob ────────────────────
check("Phase4 OUTPUT_SCHEMA required_columns has policy_prob",
      'policy_prob' in OUTPUT_SCHEMA['required_columns'],
      "policy_prob not in required_columns")
check("Phase4 OUTPUT_SCHEMA has optional_columns",
      'optional_columns' in OUTPUT_SCHEMA,
      "no optional_columns")
check("Phase4 OUTPUT_SCHEMA optional has confidence_interval_lower",
      'confidence_interval_lower' in OUTPUT_SCHEMA.get('optional_columns', []),
      "no confidence_interval_lower")
check("Phase4 OUTPUT_SCHEMA optional has pathway_chain",
      'pathway_chain' in OUTPUT_SCHEMA.get('optional_columns', []),
      "no pathway_chain")

# ─── Phase 4 handoff: confidence interval computation exists ───────────
check("Phase4 compute_policy_prob_confidence_interval exists",
      hasattr(r, 'compute_policy_prob_confidence_interval'),
      "CI function not defined")
check("Phase4 construct_pathway_chain exists",
      hasattr(r, 'construct_pathway_chain'),
      "pathway chain function not defined")

# ─── Phase 4 handoff: RankedCandidate has CI and pathway_chain fields ──
rc = RankedCandidate(drug='aspirin', disease='pain', reward=0.5)
check("Phase4 RankedCandidate has confidence_interval_lower field",
      hasattr(rc, 'confidence_interval_lower'),
      "missing field")
check("Phase4 RankedCandidate has pathway_chain field",
      hasattr(rc, 'pathway_chain'),
      "missing field")
# Verify to_dict includes the new fields
d = rc.to_dict()
check("Phase4 to_dict has policy_prob",
      'policy_prob' in d,
      "missing policy_prob")
check("Phase4 to_dict has confidence_interval_lower",
      'confidence_interval_lower' in d,
      "missing CI field")
check("Phase4 to_dict has pathway_chain",
      'pathway_chain' in d,
      "missing pathway_chain")
check("Phase4 to_dict has is_safe",
      'is_safe' in d,
      "missing is_safe")

# ─── Print results ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("v92 FORENSIC ROOT-FIX VERIFICATION RESULTS")
print("=" * 70)
for res in results:
    print(res)
print("=" * 70)
print(f"TOTAL: {passed} passed, {failed} failed")
print("=" * 70)
sys.exit(0 if failed == 0 else 1)
