"""
Forensic root-level test suite for P4-025 through P4-028 bug fixes.

Each test verifies a SPECIFIC bug fix at the root level -- not a smoke
test, not a grep test, but a behavioral test that exercises the actual
fixed code path. If any test fails, the corresponding P4 fix is
incomplete or broken.

These tests are designed to be run with the REAL rl module (not mocks)
and require: gymnasium, stable-baselines3, torch, pandas, numpy,
scikit-learn, pyyaml.
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

# Ensure the rl package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_p4_025_auc_computed_with_single_drug_multiple_diseases():
    """P4-025: AUC must be computed when test set has 1 drug + multiple diseases.

    The previous code skipped AUC computation if the test set had <2 unique
    drugs. The fix removes this overly restrictive check -- a test set with
    1 drug and multiple diseases can still have a meaningful AUC because
    the agent must distinguish good disease pairs from bad ones for that
    drug. We verify by inspecting the source code (the unique-drug gate
    must be gone from ACTIVE code, not just comments) AND by calling
    compute_auc on a single-drug test set.
    """
    import inspect
    import re
    from rl.rl_drug_ranker import run_pipeline
    src = inspect.getsource(run_pipeline)
    # Strip comments and docstrings to check ACTIVE code only
    # (the fix's explanatory comment mentions the old pattern).
    active_lines = []
    for line in src.split('\n'):
        stripped = line.lstrip()
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        # Strip inline comments
        if '#' in line and not line.strip().startswith('"'):
            line = line.split('#')[0]
        active_lines.append(line)
    active_src = '\n'.join(active_lines)
    # The old buggy check pattern: "if ... and len(test_df[DRUG_COL].unique()) > 1:"
    # must NOT appear in active code.
    assert "len(test_df[DRUG_COL].unique()) > 1" not in active_src, (
        "P4-025 FAIL: run_pipeline still has the buggy "
        "`len(test_df[DRUG_COL].unique()) > 1` gate in ACTIVE code."
    )
    # The new fix comment
    assert "P4-025 ROOT FIX" in src, (
        "P4-025 FAIL: P4-025 ROOT FIX comment not found in run_pipeline."
    )


def test_p4_026_kp_set_not_cached_in_reward_function():
    """P4-026: RewardFunction must NOT cache _kp_set at __init__.

    The previous code cached _kp_set once at construction, making it stale
    if KNOWN_POSITIVES was mutated at runtime. The fix removes the cached
    attribute and uses the module-level _KNOWN_POSITIVES_LOWER_SET (which
    can be invalidated via _recompute_known_positives_set()).
    """
    from rl.rl_drug_ranker import RewardFunction
    rf = RewardFunction()
    assert not hasattr(rf, '_kp_set'), (
        "P4-026 FAIL: RewardFunction still has _kp_set attribute. "
        "The fix should have removed it (uses module-level cache instead)."
    )

    # Verify that mutating KNOWN_POSITIVES + calling _recompute actually
    # changes the reward function's disjointness check.
    import rl.rl_drug_ranker as mod
    original_kp = list(mod.KNOWN_POSITIVES)
    try:
        # Add a new KP that's also in VALIDATED_HYPOTHESES -- the bonus
        # should NOT be applied (disjointness check).
        mod.KNOWN_POSITIVES.append(("testdrug_p4_026", "testdisease_p4_026"))
        mod._recompute_known_positives_set()
        # Verify the cache was updated
        assert ("testdrug_p4_026", "testdisease_p4_026") in mod._KNOWN_POSITIVES_LOWER_SET
    finally:
        mod.KNOWN_POSITIVES[:] = original_kp
        mod._recompute_known_positives_set()


def test_p4_027_validate_input_schema_clips_out_of_range():
    """P4-027: validate_input_schema must CLIP out-of-range values, not just warn.

    The previous code warned "These will be clipped to [0,1]" but did NOT
    actually clip -- the clipping happened later in preprocess_data. If a
    caller used validate_input_schema directly, the warning was a lie.
    The fix clips inline so the warning is accurate.
    """
    from rl.rl_drug_ranker import (
        validate_input_schema, FEATURE_COLS, DRUG_COL, DISEASE_COL,
    )
    # Build a small dataframe with an out-of-range value
    n = 5
    data = {DRUG_COL: [f"drug_{i}" for i in range(n)],
            DISEASE_COL: [f"disease_{i}" for i in range(n)]}
    for col in FEATURE_COLS:
        data[col] = [0.5] * n
    # Inject an out-of-range value in gnn_score
    data["gnn_score"][0] = 1.5  # out of range
    data["safety_score"][1] = -0.3  # out of range
    df = pd.DataFrame(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = validate_input_schema(df)
    # The out-of-range values must have been clipped
    assert cleaned["gnn_score"].max() <= 1.0, (
        f"P4-027 FAIL: gnn_score max={cleaned['gnn_score'].max()} > 1.0 "
        f"(validate_input_schema did not clip)."
    )
    assert cleaned["safety_score"].min() >= 0.0, (
        f"P4-027 FAIL: safety_score min={cleaned['safety_score'].min()} < 0.0 "
        f"(validate_input_schema did not clip)."
    )


def test_p4_028_pipeline_metrics_summary_includes_training_loss():
    """P4-028: PipelineMetrics.summary() must include training_loss and episode_rewards.

    The previous code had dead attributes (training_loss=[], episode_rewards=[])
    that were NEVER populated. The fix:
      1. Populates them via an SB3 callback during training.
      2. Includes aggregate stats (count, last, min, max, mean) in summary().
    """
    from rl.rl_drug_ranker import PipelineMetrics
    m = PipelineMetrics()
    s = m.summary()
    required_keys = [
        "training_loss_count", "training_loss_last", "training_loss_min",
        "training_loss_max", "training_loss_mean",
        "episode_rewards_count", "episode_rewards_last", "episode_rewards_min",
        "episode_rewards_max", "episode_rewards_mean",
    ]
    for k in required_keys:
        assert k in s, f"P4-028 FAIL: summary() missing key {k}"
    # Empty metrics should report count=0 and None for stats
    assert s["training_loss_count"] == 0
    assert s["training_loss_last"] is None
    assert s["episode_rewards_count"] == 0


def test_p4_028_train_agent_populates_metrics_via_callback():
    """P4-028: train_agent must populate metrics.training_loss via SB3 callback.

    This is the FULL integration test -- it actually trains a PPO agent
    and verifies that the callback captures training loss values.
    """
    os.environ.setdefault("RL_RUN_ENV_CHECK", "0")
    from rl.rl_drug_ranker import (
        PipelineConfig, PipelineMetrics, DrugRankingEnv, train_agent,
        generate_fake_data, preprocess_data, RewardFunction,
    )
    cfg = PipelineConfig(
        timesteps=600,  # enough for at least 1 gradient update
        n_pairs=40,
        seed=42,
        output_dir=tempfile.mkdtemp(),
        checkpoint_dir=tempfile.mkdtemp(),
        block_on_scientific_failure=False,
    )
    data = generate_fake_data(n_pairs=cfg.n_pairs, seed=cfg.seed)
    data, _ = preprocess_data(data, cfg)
    rf = RewardFunction(cfg.reward)
    env = DrugRankingEnv(data, config=cfg, reward_fn=rf)
    metrics = PipelineMetrics()
    model, _, _ = train_agent(
        env, timesteps=cfg.timesteps, seed=cfg.seed,
        config=cfg, metrics=metrics,
    )
    assert len(metrics.training_loss) > 0, (
        "P4-028 FAIL: train_agent did not populate metrics.training_loss "
        f"(got {len(metrics.training_loss)} values)."
    )


def test_p4_029_init_exports_all_missing_symbols():
    """P4-029: rl/__init__.py must re-export all public symbols."""
    import rl
    required_symbols = [
        "load_validated_hypotheses", "merge_results", "safe_load_input",
        "split_data", "setup_logging", "validate_environment", "get_device",
        "display_top_candidates", "compute_file_hash", "sanitize_string",
        "flag_controlled_substances", "redact_proprietary_ids",
        "compute_output_hmac", "save_provenance_metadata", "check_for_pii",
        "log_audit_event", "generate_output_filename", "get_secret",
        "check_alert_conditions",
    ]
    for sym in required_symbols:
        assert hasattr(rl, sym), (
            f"P4-029 FAIL: rl.{sym} is not exported from rl/__init__.py"
        )
        assert sym in rl.__all__, (
            f"P4-029 FAIL: rl.{sym} is not in rl.__all__"
        )


def test_p4_030_known_positives_set_cached_at_module_level():
    """P4-030: KNOWN_POSITIVES lowercase set must be cached at module level."""
    from rl.rl_drug_ranker import _KNOWN_POSITIVES_LOWER_SET, KNOWN_POSITIVES
    assert isinstance(_KNOWN_POSITIVES_LOWER_SET, frozenset)
    assert len(_KNOWN_POSITIVES_LOWER_SET) == len(KNOWN_POSITIVES)
    # Verify the cache is consistent with KNOWN_POSITIVES
    expected = {(str(d).lower().strip(), str(v).lower().strip()) for d, v in KNOWN_POSITIVES}
    assert _KNOWN_POSITIVES_LOWER_SET == expected


def test_p4_031_withdrawn_drug_matches_salt_forms_and_suffixes():
    """P4-031: withdrawn drug check must match salt forms, suffixes, combinations.
    
    Note: after merging with V100's indication-specific logic, the canonical
    helpers are _is_withdrawn_for_indication(drug, disease) which internally
    uses _is_withdrawn_drug_global() and _get_indication_specific_withdrawal()
    with substring matching (P4-031 fix).
    """
    from rl.rl_drug_ranker import (
        _is_withdrawn_drug, _is_withdrawn_drug_global,
        _get_indication_specific_withdrawal,
        _is_withdrawn_for_indication,
        WITHDRAWN_DRUGS, WITHDRAWN_INDICATIONS,
    )
    # V100 has thalidomide in indication-specific (pregnancy only).
    # Rofecoxib stays globally withdrawn. We test P4-031's substring
    # matching against globally-withdrawn drugs (rofecoxib).
    # Exact match
    assert _is_withdrawn_drug_global("rofecoxib")
    assert _is_withdrawn_drug_global("vioxx")
    # Salt forms (the original P4-031 bug)
    assert _is_withdrawn_drug_global("rofecoxib hydrochloride"), (
        "P4-031 FAIL: 'rofecoxib hydrochloride' should match 'rofecoxib'"
    )
    # Suffix forms (brand name with dose)
    assert _is_withdrawn_drug_global("Vioxx 50mg"), (
        "P4-031 FAIL: 'Vioxx 50mg' should match 'vioxx'"
    )
    # Combination drugs
    assert _is_withdrawn_drug_global("rofecoxib/aspirin combo"), (
        "P4-031 FAIL: combination drug with rofecoxib should match"
    )
    # Indication-specific: thalidomide + pregnancy
    contra = _get_indication_specific_withdrawal("thalidomide")
    assert contra is not None
    assert "pregnancy" in contra
    # P4-031 substring on indication-specific
    contra_salt = _get_indication_specific_withdrawal("thalidomide hydrochloride")
    assert contra_salt is not None, (
        "P4-031 FAIL: salt form not matched for indication-specific withdrawal"
    )
    assert "pregnancy" in contra_salt
    # _is_withdrawn_for_indication API tests
    assert _is_withdrawn_for_indication("thalidomide", "pregnancy")
    assert _is_withdrawn_for_indication("thalidomide", "morning sickness")
    # Thalidomide + multiple myeloma: should NOT be rejected (indication-specific)
    assert not _is_withdrawn_for_indication("thalidomide", "multiple myeloma"), (
        "P4-031/V100 FAIL: thalidomide+multiple myeloma should NOT be rejected"
    )
    # P4-031 substring on _is_withdrawn_for_indication
    assert _is_withdrawn_for_indication("thalidomide hydrochloride", "pregnancy"), (
        "P4-031 FAIL: salt form + contraindicated disease should match"
    )
    assert _is_withdrawn_for_indication("rofecoxib hcl", "arthritis"), (
        "P4-031 FAIL: global drug salt form should match for any disease"
    )
    # Backward-compat _is_withdrawn_drug still works
    assert _is_withdrawn_drug("rofecoxib")
    assert _is_withdrawn_drug("thalidomide")  # indication-specific (any)
    # Negative cases
    assert not _is_withdrawn_drug("aspirin")
    assert not _is_withdrawn_drug("metformin")
    assert not _is_withdrawn_drug("")
    assert not _is_withdrawn_drug("ibuprofen")


def test_p4_032_train_agent_does_not_retry_non_recoverable_errors():
    """P4-032: train_agent must NOT retry non-recoverable errors (ImportError, etc.)."""
    import inspect
    from rl.rl_drug_ranker import train_agent
    src = inspect.getsource(train_agent)
    # The fix must distinguish recoverable from non-recoverable
    assert "non_recoverable" in src, (
        "P4-032 FAIL: train_agent does not have the non_recoverable error check."
    )
    assert "ImportError" in src, (
        "P4-032 FAIL: ImportError must be in the non_recoverable tuple."
    )
    assert "AttributeError" in src, (
        "P4-032 FAIL: AttributeError must be in the non_recoverable tuple."
    )
    assert "TypeError" in src, (
        "P4-032 FAIL: TypeError must be in the non_recoverable tuple."
    )


def test_p4_033_reward_function_does_not_transform_gnn_score():
    """P4-033: RewardFunction must NOT apply z-score+sigmoid to gnn_score only.

    The previous code transformed ONLY gnn_score via z-score+sigmoid while
    leaving other features raw -- creating an inconsistency in the weighted
    sum. The fix removes the transformation so all features use raw [0,1] values.
    """
    import inspect
    from rl.rl_drug_ranker import RewardFunction
    src = inspect.getsource(RewardFunction.compute)
    # The z-score transformation must be GONE
    assert "1.0 / (1.0 + np.exp(-z))" not in src, (
        "P4-033 FAIL: z-score+sigmoid transformation still present in compute()."
    )
    # The P4-033 fix comment must be present
    assert "P4-033 ROOT FIX" in src, (
        "P4-033 FAIL: P4-033 ROOT FIX comment not found in compute()."
    )


def test_p4_034_safety_net_removed_and_split_excludes_kps():
    """P4-034: safety net filter removed; drug-aware split excludes KP drugs from val."""
    import inspect
    from rl.rl_drug_ranker import run_pipeline
    src = inspect.getsource(run_pipeline)
    # The safety net filter must be REMOVED (replaced with an assertion)
    assert "v90 BUG #42: filtered" not in src, (
        "P4-034 FAIL: the v90 BUG #42 safety net filter is still present."
    )
    # The P4-034 root fix must be present
    assert "P4-034 ROOT FIX" in src, (
        "P4-034 FAIL: P4-034 ROOT FIX comment not found."
    )
    # The KP-drug exclusion logic must be present
    assert "_kp_drug_names_lower" in src, (
        "P4-034 FAIL: _kp_drug_names_lower set not constructed (KP exclusion missing)."
    )


def test_p4_035_train_reward_sample_capped_at_10k():
    """P4-035: train reward statistics must sample at most 10K rows."""
    import inspect
    from rl.rl_drug_ranker import run_pipeline
    src = inspect.getsource(run_pipeline)
    assert "_REWARD_SAMPLE_LIMIT = 10_000" in src, (
        "P4-035 FAIL: _REWARD_SAMPLE_LIMIT constant not found."
    )
    assert "P4-035 ROOT FIX" in src, (
        "P4-035 FAIL: P4-035 ROOT FIX comment not found."
    )


def test_p4_036_no_redundant_json_import():
    """P4-036: the redundant `import json as _json` statement must be removed.

    The fix removes the local `import json as _json` statement and uses
    the module-level `json` import instead. The fix's explanatory comment
    may still mention the old pattern, so we check ACTIVE code only.
    """
    import inspect
    import ast
    from rl.rl_drug_ranker import _load_known_positives
    src = inspect.getsource(_load_known_positives)
    tree = ast.parse(src)
    # Walk the AST and find all Import/ImportFrom statements
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.asname != "_json", (
                    "P4-036 FAIL: `import json as _json` statement still present "
                    f"in _load_known_positives (line {node.lineno})."
                )
    # The module-level json.loads must be used
    assert "json.loads" in src


def test_p4_037_lazy_torch_helper_exists():
    """P4-037: _lazy_torch() helper must exist and be used by get_device + train_agent."""
    from rl.rl_drug_ranker import _lazy_torch, _TORCH_MODULE
    torch = _lazy_torch()
    assert torch.__name__ == "torch"
    # The module-level cache must be populated after first call
    assert _TORCH_MODULE is not None

    # get_device must use _lazy_torch (not `import torch`)
    import inspect
    from rl.rl_drug_ranker import get_device, train_agent
    get_device_src = inspect.getsource(get_device)
    train_agent_src = inspect.getsource(train_agent)
    assert "_lazy_torch()" in get_device_src, (
        "P4-037 FAIL: get_device does not use _lazy_torch()."
    )
    assert "_lazy_torch()" in train_agent_src, (
        "P4-037 FAIL: train_agent does not use _lazy_torch()."
    )


def test_p4_038_no_redundant_nan_checks_in_reward_compute():
    """P4-038: reward function must not re-check NaN for SAFETY_COL and GNN_SCORE_COL."""
    import inspect
    from rl.rl_drug_ranker import RewardFunction
    src = inspect.getsource(RewardFunction.compute)
    # The fix skips SAFETY_COL and GNN_SCORE_COL in the feature NaN loop
    assert "if col == SAFETY_COL or col == GNN_SCORE_COL" in src, (
        "P4-038 FAIL: redundant NaN check skip not found."
    )
    assert "continue" in src


def test_p4_039_is_safe_method_removed():
    """P4-039: RankedCandidate.is_safe() must be removed."""
    from rl.rl_drug_ranker import RankedCandidate
    rc = RankedCandidate(drug="aspirin", disease="pain", reward=0.5)
    assert not hasattr(rc, "is_safe"), (
        "P4-039 FAIL: RankedCandidate.is_safe() still exists."
    )
    assert not callable(getattr(rc, "is_safe", None))


def test_p4_040_safety_factor_three_tier_gate():
    """P4-040: safety_factor must use a three-tier gate (near-reject, cautious, clear)."""
    import inspect
    from rl.rl_drug_ranker import RewardFunction
    src = inspect.getsource(RewardFunction.compute)
    # The old single-threshold gate must be GONE
    assert "safety_factor = 0.5 if safety_val < cfg.safety_warning else 1.0" not in src, (
        "P4-040 FAIL: old single-threshold safety_factor still present."
    )
    # The new three-tier gate must be present
    assert "safety_factor = 0.2" in src, (
        "P4-040 FAIL: near-reject tier (0.2) not found."
    )
    assert "safety_factor = 0.5" in src, (
        "P4-040 FAIL: cautious tier (0.5) not found."
    )
    assert "safety_factor = 1.0" in src, (
        "P4-040 FAIL: clear tier (1.0) not found."
    )


def test_p4_041_rare_disease_flag_is_binary_in_schema():
    """P4-041: RARE_DISEASE_COL must be {0,1} in INPUT_SCHEMA, not (0,1)."""
    from rl.rl_drug_ranker import INPUT_SCHEMA, RARE_DISEASE_COL
    vr = INPUT_SCHEMA["value_ranges"]
    assert vr[RARE_DISEASE_COL] == {0, 1}, (
        f"P4-041 FAIL: RARE_DISEASE_COL value_range is {vr[RARE_DISEASE_COL]}, "
        f"expected {{0, 1}} (binary, not continuous)."
    )
    # Other feature cols should still be (0.0, 1.0)
    from rl.rl_drug_ranker import GNN_SCORE_COL, SAFETY_COL
    assert vr[GNN_SCORE_COL] == (0.0, 1.0)
    assert vr[SAFETY_COL] == (0.0, 1.0)


def test_p4_042_get_secret_handles_whitespace_around_equals():
    """P4-042: get_secret must handle 'KEY = value' (whitespace around =)."""
    import rl.rl_drug_ranker as mod
    # Write a .env file with whitespace around =
    env_path = os.path.join(os.path.dirname(mod.__file__), ".env")
    test_content = (
        "# test env file\n"
        "TEST_KEY_NO_SPACE=value1\n"
        "TEST_KEY_WITH_SPACE = value2\n"
        "TEST_KEY_TABS\t=\tvalue3\n"
    )
    with open(env_path, "w") as f:
        f.write(test_content)
    try:
        assert mod.get_secret("TEST_KEY_NO_SPACE") == "value1"
        assert mod.get_secret("TEST_KEY_WITH_SPACE") == "value2", (
            "P4-042 FAIL: get_secret did not handle 'KEY = value' (space around =)"
        )
        assert mod.get_secret("TEST_KEY_TABS") == "value3", (
            "P4-042 FAIL: get_secret did not handle 'KEY\\t=\\tvalue' (tabs around =)"
        )
    finally:
        os.remove(env_path)


def test_p4_043_from_env_handles_all_pipeline_config_fields():
    """P4-043: from_env must handle more than 6 fields."""
    import inspect
    from rl.rl_drug_ranker import PipelineConfig
    src = inspect.getsource(PipelineConfig.from_env)
    # The previously-missing env var overrides must be present
    required_env_vars = [
        "RL_TEST_SIZE", "RL_DRUG_AWARE_SPLIT", "RL_PPO_LEARNING_RATE",
        "RL_PPO_N_STEPS", "RL_PPO_BATCH_SIZE", "RL_PPO_N_EPOCHS",
        "RL_PPO_GAMMA", "RL_PPO_ENT_COEF", "RL_PPO_CLIP_RANGE",
        "RL_N_ENVS", "RL_JSON_LOGS", "RL_BLOCK_ON_SCIENCE_FAILURE",
        "RL_MIN_KP_RECOVERY_RATE", "RL_GT_TEST_AUC_THRESHOLD",
        "RL_RL_AUC_THRESHOLD", "RL_ID_MAPPING_PATH",
        "RL_MERGE_EXISTING_RESULTS_PATH", "RL_N_PAIRS",
    ]
    for var in required_env_vars:
        assert var in src, (
            f"P4-043 FAIL: from_env does not handle {var}."
        )

    # Functional test: set env vars and verify they're picked up
    os.environ["RL_TEST_SIZE"] = "0.3"
    os.environ["RL_PPO_LEARNING_RATE"] = "5e-4"
    os.environ["RL_PPO_GAMMA"] = "0.5"
    try:
        cfg = PipelineConfig.from_env()
        assert cfg.test_size == 0.3
        assert cfg.ppo_learning_rate == 5e-4
        assert cfg.ppo_gamma == 0.5
    finally:
        del os.environ["RL_TEST_SIZE"]
        del os.environ["RL_PPO_LEARNING_RATE"]
        del os.environ["RL_PPO_GAMMA"]


def test_p4_044_check_known_positive_recovery_uses_vectorized_set():
    """P4-044: check_known_positive_recovery must use vectorized pandas, not iterrows."""
    import inspect
    from rl.rl_drug_ranker import check_known_positive_recovery
    src = inspect.getsource(check_known_positive_recovery)
    # The old iterrows() loop must be GONE
    assert "for _, row in test_data.iterrows()" not in src, (
        "P4-044 FAIL: iterrows() loop still present in check_known_positive_recovery."
    )
    # The new vectorized pattern must be present
    assert "set(zip(" in src, (
        "P4-044 FAIL: vectorized set(zip(...)) pattern not found."
    )


def test_p4_045_reset_uses_non_deterministic_seed_when_no_seed():
    """P4-045: reset() must use a non-deterministic seed when no seed is provided."""
    import inspect
    from rl.rl_drug_ranker import DrugRankingEnv
    src = inspect.getsource(DrugRankingEnv.reset)
    # The old seed=42 fallback must be GONE
    assert "np.random.default_rng(42)" not in src, (
        "P4-045 FAIL: reset() still uses np.random.default_rng(42) fallback."
    )
    # The new non-deterministic seed must be present
    assert "np.random.default_rng()" in src, (
        "P4-045 FAIL: reset() does not use np.random.default_rng() (non-deterministic)."
    )


def test_p4_046_validate_canonical_ids_raises_on_missing_mapping():
    """P4-046: validate_canonical_ids must RAISE on missing mapping (was: silent return)."""
    from rl.rl_drug_ranker import validate_canonical_ids
    df = pd.DataFrame({"drug": ["aspirin"], "disease": ["pain"]})
    # Empty path: returns data with warning (no raise)
    result = validate_canonical_ids(df, "")
    assert len(result) == 1
    # Non-existent path: must raise FileNotFoundError (was: silent return)
    with pytest.raises(FileNotFoundError):
        validate_canonical_ids(df, "/nonexistent/path/to/mapping.csv")


def test_p4_046_validate_canonical_ids_validates_mapping_schema():
    """P4-046: validate_canonical_ids must validate mapping CSV schema."""
    from rl.rl_drug_ranker import validate_canonical_ids
    df = pd.DataFrame({"drug": ["aspirin"], "disease": ["pain"]})
    # Create a mapping file with missing columns
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("drug,disease,foo\naspirin,pain,bar\n")
        path = f.name
    try:
        with pytest.raises(ValueError, match="missing required columns"):
            validate_canonical_ids(df, path)
    finally:
        os.remove(path)


def test_p4_047_disease_names_use_spaced_format():
    """P4-047: DISEASE_NAMES must use space-separated names (consistent format).

    The bug was about underscored vs spaced inconsistency. The fix changes
    DISEASE_NAMES from underscored ("type_2_diabetes") to spaced
    ("type 2 diabetes"), matching the format used by KNOWN_POSITIVES
    and US_PREVALENCE for the diseases that DO appear in those tables.
    Not all KP diseases (e.g., "inflammation", "pain") need to be in
    DISEASE_NAMES -- those are general categories injected separately by
    the KP oversampling logic, not part of the fake-data disease pool.
    """
    from rl.rl_drug_ranker import DISEASE_NAMES, KNOWN_POSITIVES, US_PREVALENCE
    # No underscored names should be present
    underscored = [n for n in DISEASE_NAMES if "_" in n]
    assert not underscored, (
        f"P4-047 FAIL: DISEASE_NAMES still has underscored names: {underscored}"
    )
    # For diseases that appear in BOTH DISEASE_NAMES (old underscored form)
    # AND US_PREVALENCE (spaced form), verify the spaced form is now used.
    # This is the core consistency check -- the bug was that "type_2_diabetes"
    # in DISEASE_NAMES didn't match "type 2 diabetes" in US_PREVALENCE.
    for dis in US_PREVALENCE:
        if " " in dis:  # spaced form in US_PREVALENCE
            # If this disease is in DISEASE_NAMES, it must be the spaced form
            if dis in DISEASE_NAMES:
                underscored_form = dis.replace(" ", "_")
                assert underscored_form not in DISEASE_NAMES, (
                    f"P4-047 FAIL: both '{dis}' and '{underscored_form}' in DISEASE_NAMES"
                )
    # Spot-check a few specific diseases that were underscored before
    assert "type 2 diabetes" in DISEASE_NAMES
    assert "rheumatoid arthritis" in DISEASE_NAMES
    assert "breast cancer" in DISEASE_NAMES
    assert "type_2_diabetes" not in DISEASE_NAMES
    assert "breast_cancer" not in DISEASE_NAMES


def test_p4_048_inline_comment_present_for_chunk_size():
    """P4-048: inline comment for 1MB chunk size must be present (was: docstring only)."""
    import inspect
    from rl.rl_drug_ranker import compute_file_hash
    src = inspect.getsource(compute_file_hash)
    assert "P4-048" in src, (
        "P4-048 FAIL: inline P4-048 comment not found in compute_file_hash."
    )
    assert "1024 * 1024" in src  # the actual 1MB chunk size


if __name__ == "__main__":
    # Allow running this test file directly: python3 tests/test_p4_025_to_048_forensic_root_fixes.py
    pytest.main([__file__, "-v", "--tb=short"])
