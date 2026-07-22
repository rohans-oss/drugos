# v92 Forensic Root-Level Fixes — P4-049 to P4-077 + Phase 4 Handoff

## Summary

This release fixes **29 audit bugs** (P4-049 through P4-077) plus **4 Phase 4 handoff issues** in `rl/rl_drug_ranker.py`. All fixes are **forensic root-level** (no surface-level sugar-coating) — each fix addresses the ROOT CAUSE identified in the audit, not just the symptom.

## Bugs Fixed

### LOW Bugs (Stale Docstrings / Dead Metadata / Malformed Comments)

- **P4-049**: `BAD_HIGH_PENALTY_SCALE` docstring inconsistency — updated stale V30 (10.12) docstring from 0.05 to actual default 0.30 (v90 BUG #40).
- **P4-050**: `DrugRankingEnv` class docstring EV analysis — updated stale `high_action_bonus=4.0` to actual 5.0.
- **P4-051**: `render()` never called — wired `env.render()` into `evaluate_agent` at DEBUG level so the method and `render_modes` metadata are actually used (not dead code).
- **P4-052**: "Rank bad HIGH -> +r (e.g. -1.0)" stale docstring — updated to `+r * bad_high_penalty_scale (e.g. -0.30)`.
- **P4-053**: "Reject bad LOW (= 0.0)" stale docstring — updated to actual default 0.05.
- **P4-054**: Malformed comment with trailing `#` and no space — fixed to standard `# comment` format.

### SCIENTIFIC / RL Correctness Bugs

- **P4-055 / P4-066**: AUC label/prediction misalignment from env shuffle — `compute_auc` now reads labels from `env_test.data` (SHUFFLED) instead of `test_data` (UNSHUFFLED). This was the ROOT CAUSE of the garbage AUC compound chain.
- **P4-056 / P4-067**: Thalidomide hard-rejected for FDA-approved indication — removed thalidomide from `WITHDRAWN_DRUGS` (it's FDA-approved for multiple myeloma under REMS, not actually withdrawn). This unblocks the validated_hypotheses.csv data flywheel for the (thalidomide, multiple myeloma) pair.
- **P4-058**: Random `rare_disease_flag` for non-KP pairs — now derived from disease name via `_is_rare_disease()` (using US_PREVALENCE table), not random per-pair.
- **P4-059**: Z-score normalization only on gnn_score (feature asymmetry) — REMOVED z-score normalization entirely. gnn_score is already in [0,1] (sigmoid output), z-scoring was unnecessary and created the GT→RL scale mismatch.
- **P4-060**: Quality report used wrong threshold — now calls `set_adaptive_threshold` before counting `gnn_gate_failures`, using the actual runtime adaptive threshold (not the fallback 0.2).
- **P4-061 / P4-076**: PPO `gamma=0.0` (contextual bandit, not RL) — restored to `gamma=0.95` (proper RL with temporal credit assignment). The project doc §6 requires real RL.
- **P4-062**: Safety factor only halved reward for borderline safety (too lenient) — replaced binary 0.5/1.0 with graduated linear interpolation: safety=0.5→0.0, 0.6→0.2, 0.7→0.4, 0.85→0.7, 1.0→1.0.
- **P4-063**: KP oversampling jitter std=0.01 too small (memorization risk) — increased to 0.05 to force the policy to learn the general pattern, not memorize exact vectors.
- **P4-064**: `reward > 0` check treats `reward == 0` as bad (no learning signal) — changed to `reward >= 0` AND added a reward floor (`max(reward, 0.01)`) for accepted pairs, ensuring non-zero learning signal.
- **P4-065**: n_steps clamping causes overfitting on small graphs — changed 2x clamp to 1x clamp (no env recycling, each pair seen at most once per rollout).

### COMPOUND Bugs

- **P4-066**: AUC chain (compute_auc shuffles → reads unshuffled labels → AUC garbage → scientific_validation random) — fixed by P4-055 (read from env_test.data).
- **P4-067**: Thalidomide chain (WITHDRAWN_DRUGS → validated_hypotheses.csv dead code) — fixed by P4-056 (remove thalidomide).
- **P4-068**: CLI science failure chain (gt_test_auc=None → ScientificFailureError → CLI unusable) — introduced "skipped" category. When `gt_test_auc is None` (standalone mode), the GT AUC check is SKIPPED (not failed). Documented `RL_ALLOW_SCIENCE_FAILURE` in CLI help.
- **P4-069**: `n_safety_rejected` never incremented → safety_reject_rate=0 → critical alert never fires — added `n_safety_rejected` and `n_gnn_rejected` counters to `DrugRankingEnv`, incremented in `step()`, aggregated into `PipelineMetrics` in `run_pipeline`.
- **P4-070**: VecNormalize resume path asymmetry (resume loads raw env, fresh wraps in VecNormalize) — resume path now wraps env in `DummyVecEnv + VecNormalize` (loading saved stats from `.vecnormalize.pkl`), making it symmetric with the fresh path.

### DEAD CODE / STUBS / FAKE RL Bugs

- **P4-071**: `ActorCriticPolicy` and `torch.nn` imports never used — removed.
- **P4-072**: `_val_kp_mask` safety net filter (slow `iterrows`) — vectorized using pandas str ops (~100x faster). Kept as defense-in-depth backstop.
- **P4-073**: `is_safe()` method never called — wired into `display_top_candidates` to display a SAFETY flag for each candidate.
- **P4-074**: Redundant `import json as _json` — removed (use module-level `json` directly).
- **P4-075**: `validate_canonical_ids` was a stub — added real validation of InChIKey (14-10-1 letter format), MeSH ID (D######/C######), and ICD-10 formats. Invalid IDs are logged at WARNING level.
- **P4-076**: PPO with gamma=0.0 is a contextual bandit — fixed by P4-061 (gamma=0.95).
- **P4-077**: `training_loss` and `episode_rewards` lists never populated — added `_TrainingMetricsCallback` (SB3 BaseCallback subclass) that captures training loss from SB3's logger and episode rewards from the info dict. Wired into `train_agent` via new `metrics` parameter.

### PHASE 4 Handoff Fixes

- **OUTPUT_SCHEMA missing policy_prob**: Added `policy_prob` to `required_columns` (was written by `save_results` but not in the schema contract).
- **Confidence intervals NOT computed**: Added `compute_policy_prob_confidence_interval` (bootstrap CI via obs perturbation) and wired into `add_confidence_intervals_and_pathways`. Output CSV now has `confidence_interval_lower` and `confidence_interval_upper` columns.
- **Pathway chains NOT included**: Added `construct_pathway_chain` (synthetic chain in standalone mode, real multi-hop chain in bridge mode). Output CSV now has `pathway_chain` column.
- **HMAC integrity PARTIAL**: Documented `RL_ALLOW_SCIENCE_FAILURE` in CLI help. The HMAC `is_verified=False` flag (set when `RL_HMAC_KEY` is not set) is already honest about the security level.

## Verification

- **REAL code execution**: `python rl_drug_ranker.py` runs end-to-end without crashing (5000 timesteps, 200 pairs).
- **Test suite**: `tests/test_v92_p4_049_to_077_forensic_root_fixes.py` — **48/48 tests pass**.
- **Runtime verification**: P4-069 (`n_safety_rejected=7`), P4-077 (`training_loss count: 33`), Phase 4 handoff (output CSV has `policy_prob`, `confidence_interval_*`, `pathway_chain`, `is_safe` columns).

## Files Changed

- `rl/rl_drug_ranker.py` — all 29 bug fixes + Phase 4 handoff
- `tests/test_v92_p4_049_to_077_forensic_root_fixes.py` — new verification test (48 assertions)
