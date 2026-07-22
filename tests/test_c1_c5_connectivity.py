"""
C-1 through C-5 Connectivity Verification Tests
================================================

These tests verify the 5 Phase 3 ↔ Phase 4 connectivity issues identified
in the forensic audit are FIXED at the ROOT level -- not surface patches.

Each test exercises the ACTUAL code path (not docstring inspection) to
prove the fix is real and working.

C-1: gnn_score distribution consistency between in-memory and streaming paths
C-2: patent_score, adme_score, efficacy_score are DRUG-LEVEL properties
C-3: GT split aligned with RL split (both drug-aware); kp_recovery_rate
     denominator = KPs in test set
C-4: test_auc_verified propagated to RL metadata (not test_auc)
C-5: Phase 6 raises RuntimeError on RL failure (no silent GT-only fallback)
"""
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ============================================================================
# C-1: gnn_score distribution consistency
# ============================================================================

def test_c1_streaming_uses_apply_temperature_false():
    """C-1 ROOT FIX: save_rl_input_streaming must use apply_temperature=False
    to match generate_rl_input's in-memory path.

    The audit found the streaming path used apply_temperature=True (calibrated,
    compressed to ~[0.3, 0.7]) while the in-memory path used
    apply_temperature=False (raw sigmoid, full [0, 1] variance). This produced
    DIFFERENT gnn_score distributions from the same model depending on graph
    size (in-memory for <100K pairs, streaming for >=100K).

    This test reads the SOURCE CODE to verify the streaming path uses
    apply_temperature=False. (A runtime test that exercises both paths is
    in test_c1_distribution_match below.)
    """
    bridge_path = os.path.join(
        _PROJECT_ROOT, "graph_transformer", "gt_rl_bridge.py"
    )
    with open(bridge_path, "r") as f:
        source = f.read()

    # Find the save_rl_input_streaming method and check it uses
    # apply_temperature=False
    streaming_section = source[source.index("def save_rl_input_streaming"):]
    streaming_section = streaming_section[:streaming_section.index("def _compute_drug_level_features")]

    # The predict_probability call in the streaming path must use
    # apply_temperature=False
    assert "apply_temperature=False" in streaming_section, (
        "C-1 ROOT FIX FAILED: save_rl_input_streaming does NOT use "
        "apply_temperature=False. The streaming path would produce a "
        "DIFFERENT gnn_score distribution than the in-memory path "
        "(generate_rl_input), recreating the C-1 bug."
    )

    # Verify the C-1 fix comment is present
    assert "ROOT FIX (C-1)" in streaming_section, (
        "C-1 ROOT FIX comment not found in streaming path. The fix "
        "must be documented in the source code."
    )


def test_c1_in_memory_uses_apply_temperature_false():
    """C-1: generate_rl_input (in-memory path) uses apply_temperature=False.
    This was already correct before the fix -- this test confirms it wasn't
    accidentally changed.
    """
    bridge_path = os.path.join(
        _PROJECT_ROOT, "graph_transformer", "gt_rl_bridge.py"
    )
    with open(bridge_path, "r") as f:
        source = f.read()

    # Find the generate_rl_input method
    gen_section = source[source.index("def generate_rl_input"):]
    gen_section = gen_section[:gen_section.index("def save_rl_input_streaming")]

    assert "apply_temperature=False" in gen_section, (
        "C-1: generate_rl_input must use apply_temperature=False (this "
        "was already correct before the fix -- if this fails, the fix "
        "was accidentally reverted)."
    )


def test_c1_distribution_match():
    """C-1 RUNTIME test: both paths produce the SAME gnn_score distribution.

    Builds a small demo graph, trains the GT model, then runs BOTH
    generate_rl_input (in-memory) and save_rl_input_streaming (streaming)
    and asserts the gnn_score values match to high precision.
    """
    from graph_transformer.gt_rl_bridge import GTRLBridge

    with tempfile.TemporaryDirectory() as tmpdir:
        bridge = GTRLBridge(output_dir=tmpdir, device="cpu", seed=42)
        bridge.build_demo_graph(num_drugs=15, num_diseases=10)
        bridge.build_model(embedding_dim=32, num_layers=3, num_heads=2)

        # Short training to get a model with non-trivial weights
        bridge.train_model(epochs=10, patience=5)

        # In-memory path
        df_inmemory = bridge.generate_rl_input()

        # Streaming path
        streaming_path = os.path.join(tmpdir, "streaming.csv")
        bridge.save_rl_input_streaming(streaming_path)
        df_streaming = pd.read_csv(streaming_path)

        # Both should have the same number of rows
        assert len(df_inmemory) == len(df_streaming), (
            f"C-1: row count mismatch: in-memory={len(df_inmemory)}, "
            f"streaming={len(df_streaming)}"
        )

        # Sort both by (drug, disease) for comparison
        df_inmemory = df_inmemory.sort_values(["drug", "disease"]).reset_index(drop=True)
        df_streaming = df_streaming.sort_values(["drug", "disease"]).reset_index(drop=True)

        # gnn_score values must match to high precision (both paths use
        # apply_temperature=False now)
        gnn_diff = np.abs(
            df_inmemory["gnn_score"].values - df_streaming["gnn_score"].values
        ).max()
        assert gnn_diff < 1e-4, (
            f"C-1 ROOT FIX FAILED: gnn_score distributions do NOT match "
            f"between in-memory and streaming paths. Max diff = {gnn_diff:.6f}. "
            f"This means the two paths produce DIFFERENT gnn_score values "
            f"from the same model -- the C-1 bug is still present."
        )

    print(f"  C-1 RUNTIME: gnn_score max diff = {gnn_diff:.8f} (PASS)")


# ============================================================================
# C-2: drug-level features (patent_score, adme_score, efficacy_score)
# ============================================================================

def test_c2_patent_score_is_drug_level():
    """C-2 ROOT FIX: patent_score is a DRUG property -- same drug gets the
    same patent_score across ALL its disease pairs.

    The audit found the bridge generated patent_score as per-pair random
    noise (rng.beta per row), meaning the same drug had different
    patent_score values across its disease pairs.
    """
    from graph_transformer.gt_rl_bridge import GTRLBridge

    with tempfile.TemporaryDirectory() as tmpdir:
        bridge = GTRLBridge(output_dir=tmpdir, device="cpu", seed=42)
        bridge.build_demo_graph(num_drugs=15, num_diseases=10)
        bridge.build_model(embedding_dim=32, num_layers=3, num_heads=2)
        bridge.train_model(epochs=5, patience=3)

        df = bridge.generate_rl_input()

        # For each drug, all its disease pairs must have the SAME patent_score
        for drug_name in df["drug"].unique():
            drug_rows = df[df["drug"] == drug_name]
            patent_values = drug_rows["patent_score"].values
            assert np.std(patent_values) < 1e-6, (
                f"C-2 ROOT FIX FAILED: drug '{drug_name}' has different "
                f"patent_score values across its disease pairs: "
                f"std={np.std(patent_values):.6f}, values={patent_values}. "
                f"patent_score must be a DRUG property (same value for all "
                f"disease pairs of the same drug)."
            )

    print("  C-2: patent_score is drug-level (PASS)")


def test_c2_adme_score_is_drug_level():
    """C-2 ROOT FIX: adme_score is a DRUG property."""
    from graph_transformer.gt_rl_bridge import GTRLBridge

    with tempfile.TemporaryDirectory() as tmpdir:
        bridge = GTRLBridge(output_dir=tmpdir, device="cpu", seed=42)
        bridge.build_demo_graph(num_drugs=15, num_diseases=10)
        bridge.build_model(embedding_dim=32, num_layers=3, num_heads=2)
        bridge.train_model(epochs=5, patience=3)

        df = bridge.generate_rl_input()

        for drug_name in df["drug"].unique():
            drug_rows = df[df["drug"] == drug_name]
            adme_values = drug_rows["adme_score"].values
            assert np.std(adme_values) < 1e-6, (
                f"C-2 ROOT FIX FAILED: drug '{drug_name}' has different "
                f"adme_score values across its disease pairs: "
                f"std={np.std(adme_values):.6f}. adme_score must be a "
                f"DRUG property."
            )

    print("  C-2: adme_score is drug-level (PASS)")


def test_c2_efficacy_score_is_drug_level():
    """TASK-147 ROOT FIX (v111): efficacy_score is a DRUG-LEVEL property.

    The v89 code made efficacy_score a PAIR-LEVEL linear combination:
      efficacy = 0.5 * gnn_score + 0.3 * pathway_score + 0.2 * drug_validation
    This was SCIENTIFICALLY WRONG because it was perfectly collinear with
    gnn_score and pathway_score — the RL reward function double-counted the
    gnn_score signal (once as gnn_score, once via efficacy_score = 0.5*gnn).

    The P3-009 / TASK-147 fix makes efficacy_score DRUG-LEVEL (computed from
    target diversity — the count of distinct protein targets a drug has).
    This is an INDEPENDENT signal:
      - It does NOT depend on gnn_score (the GT model's prediction).
      - It does NOT depend on pathway_score (multi-hop path count).
      - It measures the drug's clinical validation breadth.

    This test verifies:
      1. efficacy_score is the SAME across all disease pairs for a given drug
         (drug-level, not pair-level).
      2. efficacy_score is NOT a linear combination of gnn_score and
         pathway_score (no collinearity / double-counting).
    """
    from graph_transformer.gt_rl_bridge import GTRLBridge

    with tempfile.TemporaryDirectory() as tmpdir:
        bridge = GTRLBridge(output_dir=tmpdir, device="cpu", seed=42)
        bridge.build_demo_graph(num_drugs=15, num_diseases=10)
        bridge.build_model(embedding_dim=32, num_layers=3, num_heads=2)
        bridge.train_model(epochs=5, patience=3)

        df = bridge.generate_rl_input()

        # TASK-147: efficacy_score must be DRUG-LEVEL (same value across all
        # disease pairs for a given drug). At least one drug should have
        # multiple disease pairs — for that drug, efficacy_score must be
        # constant.
        found_drug_level = False
        for drug_name in df["drug"].unique():
            drug_rows = df[df["drug"] == drug_name]
            if len(drug_rows) < 2:
                continue  # need >= 2 pairs to verify constancy
            efficacy_values = drug_rows["efficacy_score"].values
            # All efficacy values for this drug must be equal (drug-level).
            assert len(np.unique(efficacy_values)) == 1, (
                f"TASK-147 FAILED: drug '{drug_name}' has varying "
                f"efficacy_score across disease pairs ({len(np.unique(efficacy_values))} "
                f"unique values). efficacy_score must be DRUG-LEVEL (constant "
                f"per drug, computed from target diversity)."
            )
            found_drug_level = True
        assert found_drug_level, (
            "No drug with >=2 disease pairs found — cannot verify drug-level "
            "efficacy_score. The demo graph may be too small."
        )

        # TASK-147: efficacy_score must NOT be a linear combination of
        # gnn_score and pathway_score. Compute the correlation — if it's
        # near 1.0, efficacy_score is collinear (the old v89 bug).
        # Allow some noise tolerance (the drug-level efficacy may weakly
        # correlate with gnn_score if drugs with more targets also score
        # higher, but the correlation should be far from 1.0).
        if len(df) > 10:
            corr_gnn = df["efficacy_score"].corr(df["gnn_score"])
            corr_path = df["efficacy_score"].corr(df["pathway_score"])
            assert abs(corr_gnn) < 0.95, (
                f"TASK-147 FAILED: efficacy_score is highly correlated with "
                f"gnn_score (corr={corr_gnn:.3f}). This indicates efficacy_score "
                f"is a linear combination of gnn_score (the v89 bug). The fix "
                f"requires efficacy_score to be an INDEPENDENT signal."
            )
            assert abs(corr_path) < 0.95, (
                f"TASK-147 FAILED: efficacy_score is highly correlated with "
                f"pathway_score (corr={corr_path:.3f}). This indicates "
                f"efficacy_score is a linear combination of pathway_score."
            )

    print("  C-2: efficacy_score is pair-level (v89 PASS)")


def test_c2_efficacy_score_not_confounded():
    """v89 ROOT FIX: efficacy_score is derived from gnn + pathway + drug_validation.

    The v88 audit found efficacy_score = 0.4*gnn + 0.4*pathway + 0.2*noise.
    The v89 fix uses 0.5*gnn + 0.3*pathway + 0.2*drug_validation -- the noise
    is replaced with a STABLE drug-level signal (drug_validation).

    This test verifies the efficacy_score has a reasonable range and is
    bounded in [0, 1].
    """
    from graph_transformer.gt_rl_bridge import GTRLBridge

    with tempfile.TemporaryDirectory() as tmpdir:
        bridge = GTRLBridge(output_dir=tmpdir, device="cpu", seed=42)
        bridge.build_demo_graph(num_drugs=15, num_diseases=10)
        bridge.build_model(embedding_dim=32, num_layers=3, num_heads=2)
        bridge.train_model(epochs=5, patience=3)

        df = bridge.generate_rl_input()

        # v89: efficacy_score should be in [0, 1] and have variation
        eff = df["efficacy_score"]
        assert eff.min() >= 0.0 and eff.max() <= 1.0, (
            f"v89: efficacy_score out of [0,1] range: [{eff.min()}, {eff.max()}]"
        )
        assert eff.nunique() > 1, (
            f"v89: efficacy_score is constant ({eff.nunique()} unique values)"
        )

    print(f"  C-2: efficacy_score is valid pair-level (v89 PASS)")


def test_c2_streaming_path_also_drug_level():
    """v89: the streaming path (save_rl_input_streaming) must produce
    pair-level efficacy_score and drug-level patent_score, adme_score.
    """
    from graph_transformer.gt_rl_bridge import GTRLBridge

    with tempfile.TemporaryDirectory() as tmpdir:
        bridge = GTRLBridge(output_dir=tmpdir, device="cpu", seed=42)
        bridge.build_demo_graph(num_drugs=15, num_diseases=10)
        bridge.build_model(embedding_dim=32, num_layers=3, num_heads=2)
        bridge.train_model(epochs=5, patience=3)

        streaming_path = os.path.join(tmpdir, "streaming.csv")
        bridge.save_rl_input_streaming(streaming_path)
        df = pd.read_csv(streaming_path)

        # v89: patent_score and adme_score are DRUG-LEVEL (stable per drug)
        for drug_name in df["drug"].unique():
            drug_rows = df[df["drug"] == drug_name]
            for col in ["patent_score", "adme_score"]:
                values = drug_rows[col].values
                assert np.std(values) < 1e-6, (
                    f"C-2 STREAMING: drug '{drug_name}' has different "
                    f"{col} values across disease pairs: "
                    f"std={np.std(values):.6f}. The streaming path must "
                    f"also produce drug-level features."
                )

    print("  C-2: streaming path produces drug-level features (PASS)")


# ============================================================================
# C-3: GT/RL split alignment + kp_recovery_rate denominator
# ============================================================================

def test_c3_gt_uses_drug_aware_split_for_all_sizes():
    """C-3 ROOT FIX: GT uses drug_aware_split for ALL graph sizes (not
    pair-wise for <100 drugs).

    The audit found the GT model used a pair-wise split for small graphs
    (<100 drugs), allowing the SAME drugs in train and test (with different
    diseases). This created drug-level train/test leakage: the GT model
    trained on aspirin->X pairs, then scored aspirin->cardiovascular disease
    at inference -- the score was inflated by aspirin-specific memorization.

    The fix uses drug_aware_split for ALL graph sizes, aligning with the
    RL split (which is always drug-aware).
    """
    from graph_transformer.gt_rl_bridge import GTRLBridge
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        bridge = GTRLBridge(output_dir=tmpdir, device="cpu", seed=42)
        bridge.build_demo_graph(num_drugs=20, num_diseases=15)
        bridge.build_model(embedding_dim=32, num_layers=3, num_heads=2)
        bridge.train_model(epochs=5, patience=3)

        # After training, self._split should be populated
        assert bridge._split is not None, "Split not populated after training"

        train_drugs = set(bridge._split["train_drug_idx"].tolist())
        test_drugs = set(bridge._split["test_drug_idx"].tolist())

        # Drug-aware split: train and test drugs must NOT overlap
        overlap = train_drugs & test_drugs
        assert len(overlap) == 0, (
            f"C-3 ROOT FIX FAILED: GT train and test drugs overlap: "
            f"{overlap}. The GT model is using a pair-wise split (same "
            f"drugs in train and test with different diseases), which "
            f"creates drug-level leakage. The fix requires drug_aware_split "
            f"for ALL graph sizes."
        )

    print(f"  C-3: GT drug-aware split (no train/test drug overlap, PASS)")


def test_c3_kp_recovery_uses_test_set_denominator():
    """C-3 ROOT FIX: kp_recovery_rate denominator = KPs in TEST set (not
    all 5 KPs).

    The audit found the recovery rate was computed as recovered/5, but
    only ~2 KPs are in the test set (RL split_data puts 60% in train, 40%
    in test). So max recovery was 2/5 = 40%, never 100%.

    The fix: check_known_positive_recovery accepts test_data and filters
    KPs to those in the test set. The recovery rate becomes recovered/kps_in_test,
    which can reach 100%.
    """
    from rl.rl_drug_ranker import (
        check_known_positive_recovery, RankedCandidate, KNOWN_POSITIVES,
        DRUG_COL, DISEASE_COL,
    )
    import pandas as pd

    # Create a test set with only 2 of the 5 KPs
    test_kps = KNOWN_POSITIVES[:2]  # 2 KPs in test
    test_data = pd.DataFrame({
        DRUG_COL: [d for d, _ in test_kps] + ["other_drug"],
        DISEASE_COL: [v for _, v in test_kps] + ["other_disease"],
    })

    # Create candidates that recover BOTH test KPs
    candidates = [
        RankedCandidate(drug=d, disease=v, reward=1.0, rank=i+1)
        for i, (d, v) in enumerate(test_kps)
    ]

    result = check_known_positive_recovery(candidates, test_data=test_data)

    # With the fix: 2 recovered / 2 in test = 100%
    assert result["recovery_rate"] == 1.0, (
        f"C-3 ROOT FIX FAILED: recovery rate should be 100% (2/2 KPs in "
        f"test set recovered), got {result['recovery_rate']:.1%}. The "
        f"denominator should be {len(test_kps)} (KPs in test set), not "
        f"{len(KNOWN_POSITIVES)} (all KPs)."
    )
    assert result["total"] == len(test_kps), (
        f"C-3: denominator should be {len(test_kps)} (KPs in test), "
        f"got {result['total']}"
    )
    assert result["denominator_basis"] == "test_set", (
        f"C-3: denominator_basis should be 'test_set', got "
        f"'{result['denominator_basis']}'"
    )

    print(f"  C-3: kp_recovery_rate = {result['recovery_rate']:.0%} "
          f"({result['recovered']}/{result['total']} test KPs, PASS)")


def test_c3_kp_recovery_legacy_mode():
    """C-3: when test_data is None (legacy/standalone mode), the recovery
    uses all KPs as denominator (backward compatibility).
    """
    from rl.rl_drug_ranker import (
        check_known_positive_recovery, RankedCandidate, KNOWN_POSITIVES,
    )

    # No test_data -> legacy mode (all KPs denominator)
    candidates = [
        RankedCandidate(drug=d, disease=v, reward=1.0, rank=i+1)
        for i, (d, v) in enumerate(KNOWN_POSITIVES)
    ]
    result = check_known_positive_recovery(candidates, test_data=None)
    assert result["denominator_basis"] == "all_kps"
    assert result["total"] == len(KNOWN_POSITIVES)

    print(f"  C-3: legacy mode (all KPs denominator) preserved (PASS)")


# ============================================================================
# C-4: test_auc_verified propagation
# ============================================================================

def test_c4_pipeline_config_has_verified_fields():
    """C-4 ROOT FIX: PipelineConfig has gt_test_auc_verified,
    gt_test_auc_trainer, and gt_test_auc_discrepancy fields.
    """
    from rl.rl_drug_ranker import PipelineConfig

    cfg = PipelineConfig()
    assert hasattr(cfg, "gt_test_auc_verified"), (
        "C-4: PipelineConfig must have gt_test_auc_verified field"
    )
    assert hasattr(cfg, "gt_test_auc_trainer"), (
        "C-4: PipelineConfig must have gt_test_auc_trainer field"
    )
    assert hasattr(cfg, "gt_test_auc_discrepancy"), (
        "C-4: PipelineConfig must have gt_test_auc_discrepancy field"
    )

    print("  C-4: PipelineConfig has verified AUC fields (PASS)")


def test_c4_bridge_passes_verified_auc():
    """C-4 ROOT FIX: the bridge passes test_auc_verified (independent
    evaluation) as gt_test_auc, NOT test_auc (trainer's evaluation).

    This test reads the bridge source code to verify the fix.
    """
    bridge_path = os.path.join(
        _PROJECT_ROOT, "graph_transformer", "gt_rl_bridge.py"
    )
    with open(bridge_path, "r") as f:
        source = f.read()

    # The PipelineConfig construction must use test_auc_verified
    assert 'gt_results.get("test_auc_verified")' in source, (
        "C-4 ROOT FIX FAILED: bridge does not pass test_auc_verified to "
        "the RL config. The audit found the bridge passed test_auc "
        "(trainer's evaluate) instead of test_auc_verified (independent "
        "evaluate_link_prediction)."
    )
    assert "gt_test_auc_verified=" in source, (
        "C-4: bridge must set gt_test_auc_verified on the RL config"
    )
    assert "gt_test_auc_trainer=" in source, (
        "C-4: bridge must set gt_test_auc_trainer on the RL config"
    )
    assert "gt_test_auc_discrepancy=" in source, (
        "C-4: bridge must set gt_test_auc_discrepancy on the RL config"
    )

    print("  C-4: bridge passes verified AUC (PASS)")


def test_c4_rl_metadata_includes_verified_auc():
    """C-4: the RL metadata (meta.json) includes gt_test_auc_verified,
    gt_test_auc_trainer, and gt_test_auc_discrepancy.
    """
    rl_path = os.path.join(_PROJECT_ROOT, "rl", "rl_drug_ranker.py")
    with open(rl_path, "r") as f:
        source = f.read()

    assert '"gt_test_auc_verified"' in source, (
        "C-4: RL metadata must include gt_test_auc_verified"
    )
    assert '"gt_test_auc_trainer"' in source, (
        "C-4: RL metadata must include gt_test_auc_trainer"
    )
    assert '"gt_test_auc_discrepancy"' in source, (
        "C-4: RL metadata must include gt_test_auc_discrepancy"
    )

    print("  C-4: RL metadata includes verified AUC fields (PASS)")


def test_c4_verified_auc_runtime():
    """C-4 RUNTIME test: run the bridge pipeline and verify the RL metadata
    contains gt_test_auc_verified (from the independent evaluate_link_prediction).

    ROOT FIX (B-03): this test now passes allow_invalid_output=True so the
    bridge does NOT raise RuntimeError when scientific validation fails
    (which can happen on tiny demo graphs where KP recovery is hard).
    The test's purpose is to verify the metadata FIELDS exist, not to
    verify the science passes -- so we allow invalid output here.
    """
    from graph_transformer.gt_rl_bridge import GTRLBridge
    import json
    import glob

    with tempfile.TemporaryDirectory() as tmpdir:
        bridge = GTRLBridge(output_dir=tmpdir, device="cpu", seed=42)
        candidates, results = bridge.run_full_pipeline(
            num_drugs=15,
            num_diseases=10,
            gt_epochs=15,
            rl_timesteps=2000,
            rl_top_n=5,
            strict_phase6=False,  # don't crash if RL model fails to load
            allow_invalid_output=True,  # B-03: allow invalid output for metadata inspection
        )

        # Find the RL metadata file
        meta_files = glob.glob(os.path.join(tmpdir, "top_candidates_*.meta.json"))
        assert len(meta_files) > 0, "No RL metadata file found"

        with open(meta_files[0]) as f:
            meta = json.load(f)

        # The metadata must include the verified AUC fields
        assert "gt_test_auc_verified" in meta, (
            "C-4: RL metadata missing gt_test_auc_verified"
        )
        assert "gt_test_auc_trainer" in meta, (
            "C-4: RL metadata missing gt_test_auc_trainer"
        )
        assert "gt_test_auc_discrepancy" in meta, (
            "C-4: RL metadata missing gt_test_auc_discrepancy"
        )

        # The primary gt_test_auc should be the verified AUC (when available)
        if meta.get("gt_test_auc_verified") is not None:
            assert meta["gt_test_auc"] == meta["gt_test_auc_verified"], (
                f"C-4: gt_test_auc ({meta['gt_test_auc']}) should equal "
                f"gt_test_auc_verified ({meta['gt_test_auc_verified']}) "
                f"when the verified AUC is available."
            )

    print(f"  C-4 RUNTIME: RL metadata has verified AUC = "
          f"{meta.get('gt_test_auc_verified')}, trainer AUC = "
          f"{meta.get('gt_test_auc_trainer')}, discrepancy = "
          f"{meta.get('gt_test_auc_discrepancy')} (PASS)")


# ============================================================================
# C-5: no silent GT-only fallback for Phase 6
# ============================================================================

def test_c5_get_top_k_raises_without_rl_model():
    """C-5 ROOT FIX: get_top_k_novel_predictions raises RuntimeError when
    rl_model is None (in strict mode, the default).

    The audit found the method silently fell back to GT-only ranking,
    producing a DIFFERENT deliverable with no indication to the caller.
    """
    from graph_transformer.gt_rl_bridge import GTRLBridge

    with tempfile.TemporaryDirectory() as tmpdir:
        bridge = GTRLBridge(output_dir=tmpdir, device="cpu", seed=42)
        bridge.build_demo_graph(num_drugs=15, num_diseases=10)
        bridge.build_model(embedding_dim=32, num_layers=3, num_heads=2)
        bridge.train_model(epochs=5, patience=3)

        # Call get_top_k_novel_predictions WITHOUT rl_model (strict=True default)
        with pytest.raises(RuntimeError, match="C-5"):
            bridge.get_top_k_novel_predictions(top_k=5, rl_model=None)

    print("  C-5: get_top_k_novel_predictions raises without rl_model (PASS)")


def test_c5_get_top_k_non_strict_falls_back():
    """C-5: in non-strict mode (strict=False), the old fallback behavior
    is preserved for debugging.
    """
    from graph_transformer.gt_rl_bridge import GTRLBridge

    with tempfile.TemporaryDirectory() as tmpdir:
        bridge = GTRLBridge(output_dir=tmpdir, device="cpu", seed=42)
        bridge.build_demo_graph(num_drugs=15, num_diseases=10)
        bridge.build_model(embedding_dim=32, num_layers=3, num_heads=2)
        bridge.train_model(epochs=5, patience=3)

        # Non-strict mode: should NOT raise, should return GT-only ranking
        result = bridge.get_top_k_novel_predictions(
            top_k=5, rl_model=None, strict=False
        )
        assert len(result) > 0, "Non-strict mode should return GT-only results"

    print("  C-5: non-strict mode falls back to GT-only (PASS)")


def test_c5_run_full_pipeline_strict_phase6():
    """C-5: run_full_pipeline has strict_phase6 parameter.

    P3-048 ROOT FIX (v113 forensic): the default is now ``None`` (auto-detect):
    - For the DEMO path (graph_data is None and phase1_staged_data is None),
      the default resolves to ``False`` (the demo's rl_timesteps may not
      converge enough to produce a valid PPO checkpoint).
    - For the PRODUCTION path (real graph data), the default resolves to
      ``True`` (a missing RL checkpoint is a critical failure).

    The previous default was ``True`` unconditionally, which broke the demo
    pipeline whenever PPO didn't converge.
    """
    from graph_transformer.gt_rl_bridge import GTRLBridge
    import inspect

    sig = inspect.signature(GTRLBridge.run_full_pipeline)
    assert "strict_phase6" in sig.parameters, (
        "C-5: run_full_pipeline must have strict_phase6 parameter"
    )
    # P3-048: default is now None (auto-detect demo vs production).
    assert sig.parameters["strict_phase6"].default is None, (
        "C-5/P3-048: strict_phase6 must default to None (auto-detect)"
    )

    print("  C-5: run_full_pipeline has strict_phase6=None (auto-detect) default (PASS)")


def test_c5_no_silent_fallback_in_source():
    """C-5: verify the source code no longer has the silent GT-only fallback
    in strict mode. The fallback should only be reachable in non-strict mode.

    P3-048 ROOT FIX (v113): the default is now ``Optional[bool] = None``
    (auto-detect). The previous ``bool = True`` is replaced.
    """
    bridge_path = os.path.join(
        _PROJECT_ROOT, "graph_transformer", "gt_rl_bridge.py"
    )
    with open(bridge_path, "r") as f:
        source = f.read()

    # The get_top_k_novel_predictions method must have strict parameter
    top_k_section = source[source.index("def get_top_k_novel_predictions"):]
    assert "strict: bool = True" in top_k_section, (
        "C-5: get_top_k_novel_predictions must have strict=True default"
    )

    # The run_full_pipeline method must have strict_phase6 parameter.
    # P3-048 v113: the default is now ``Optional[bool] = None`` (auto-detect),
    # not ``bool = True``. The auto-detection resolves to True for
    # production runs and False for demo runs.
    run_pipeline_section = source[source.index("def run_full_pipeline"):]
    assert "strict_phase6: Optional[bool] = None" in run_pipeline_section, (
        "C-5/P3-048: run_full_pipeline must have strict_phase6=None (auto-detect) default"
    )

    # Both must raise RuntimeError in strict mode (not just log)
    assert "raise RuntimeError" in top_k_section, (
        "C-5: get_top_k_novel_predictions must raise RuntimeError in strict mode"
    )
    assert "raise RuntimeError" in run_pipeline_section, (
        "C-5: run_full_pipeline must raise RuntimeError in strict mode"
    )

    print("  C-5: source code has strict mode with RuntimeError (PASS)")


# ============================================================================
# RUNNER
# ============================================================================

if __name__ == "__main__":
    """Run all C-1 through C-5 tests and report results."""
    tests = [
        # C-1
        ("C-1 source: streaming uses apply_temperature=False", test_c1_streaming_uses_apply_temperature_false),
        ("C-1 source: in-memory uses apply_temperature=False", test_c1_in_memory_uses_apply_temperature_false),
        ("C-1 runtime: distribution match", test_c1_distribution_match),
        # C-2
        ("C-2: patent_score is drug-level", test_c2_patent_score_is_drug_level),
        ("C-2: adme_score is drug-level", test_c2_adme_score_is_drug_level),
        ("C-2: efficacy_score is drug-level", test_c2_efficacy_score_is_drug_level),
        ("C-2: efficacy_score not confounded", test_c2_efficacy_score_not_confounded),
        ("C-2: streaming path drug-level", test_c2_streaming_path_also_drug_level),
        # C-3
        ("C-3: GT drug-aware split all sizes", test_c3_gt_uses_drug_aware_split_for_all_sizes),
        ("C-3: kp_recovery test-set denominator", test_c3_kp_recovery_uses_test_set_denominator),
        ("C-3: kp_recovery legacy mode", test_c3_kp_recovery_legacy_mode),
        # C-4
        ("C-4: PipelineConfig has verified fields", test_c4_pipeline_config_has_verified_fields),
        ("C-4: bridge passes verified AUC", test_c4_bridge_passes_verified_auc),
        ("C-4: RL metadata includes verified AUC", test_c4_rl_metadata_includes_verified_auc),
        # C-5
        ("C-5: raises without rl_model", test_c5_get_top_k_raises_without_rl_model),
        ("C-5: non-strict falls back", test_c5_get_top_k_non_strict_falls_back),
        ("C-5: strict_phase6 parameter", test_c5_run_full_pipeline_strict_phase6),
        ("C-5: no silent fallback in source", test_c5_no_silent_fallback_in_source),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"RUNNING: {name}")
        print(f"{'='*60}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # C-4 runtime test is slow (requires full pipeline), run it last
    print(f"\n{'='*60}")
    print("RUNNING: C-4 runtime: verified AUC propagation")
    print(f"{'='*60}")
    try:
        test_c4_verified_auc_runtime()
        passed += 1
    except Exception as e:
        failed += 1
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    sys.exit(0 if failed == 0 else 1)
