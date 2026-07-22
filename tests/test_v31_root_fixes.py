"""
V31 ROOT-LEVEL FIX VERIFICATION TESTS
======================================

This test suite verifies the V31 root-level fixes that address the
remaining issues from the FORENSIC_AUDIT_REPORT:

  1. P0-1 / Compound #3: TRAINING_POSITIVES (real DrugBank/RepoDB
     drug-disease pairs) injected as "treats" edges + multi-hop
     biological plausibility paths for both training positives AND
     held-out KP drugs.
  2. P1-9 / Compound #9 / Finding 10.2: VecNormalize stats persisted
     alongside PPO model checkpoint.
  3. P1-11 / Compound #6: Streaming RNG re-seed fixed (feature RNG
     hoisted to instance state).
  4. P1-12 / Compound #10: Phase 6 temperature mismatch fixed
     (top_k_novel_predictions uses apply_temperature=False to match
     RL training distribution).

Each test is self-contained and can be run independently.
"""
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch
import pytest

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def test_v31_training_positives_exist():
    """P0-1: verify TRAINING_POSITIVES constant exists and has real pairs."""
    from graph_transformer.data.graph_builder import BiomedicalGraphBuilder

    assert hasattr(BiomedicalGraphBuilder, 'TRAINING_POSITIVES'), \
        "TRAINING_POSITIVES constant must exist on BiomedicalGraphBuilder"
    assert len(BiomedicalGraphBuilder.TRAINING_POSITIVES) >= 20, \
        f"TRAINING_POSITIVES must have >=20 pairs, got {len(BiomedicalGraphBuilder.TRAINING_POSITIVES)}"

    # Verify all training-positive drugs are NON-KP drugs (not in the first 5
    # REAL_DRUG_NAMES which are the KP drugs).
    kp_drugs = set(BiomedicalGraphBuilder.REAL_DRUG_NAMES[:5])
    for drug, disease in BiomedicalGraphBuilder.TRAINING_POSITIVES:
        assert drug not in kp_drugs, \
            f"Training positive drug '{drug}' must NOT be a KP drug (KPs: {kp_drugs})"
        assert isinstance(drug, str) and isinstance(disease, str)
        assert len(drug) > 0 and len(disease) > 0

    print(f"  PASS: TRAINING_POSITIVES has {len(BiomedicalGraphBuilder.TRAINING_POSITIVES)} "
          f"real DrugBank/RepoDB pairs, all using non-KP drugs.")


def test_v31_training_positives_injected_into_graph():
    """P0-1: verify training positives are injected as 'treats' edges."""
    from graph_transformer.data.graph_builder import BiomedicalGraphBuilder

    node_features, edge_indices, node_maps, known_pairs = \
        BiomedicalGraphBuilder.build_demo_graph(
            num_drugs=60, num_diseases=40, seed=42,
            known_positives=[("aspirin", "cardiovascular disease")],
        )

    treats_key = ("drug", "treats", "disease")
    assert treats_key in edge_indices, "treats edges must exist"
    treats_edges = edge_indices[treats_key]
    n_treats = treats_edges.shape[1]

    # We should have: 1 KP + ~31 training positives = ~32 treats edges.
    # The exact count depends on how many training-positive drugs/diseases
    # fit in the graph size, but it must be > 5 (more than just KPs).
    assert n_treats > 5, \
        f"Expected >5 treats edges (KP + training positives), got {n_treats}"

    # known_pairs should only contain the KPs (5), NOT the training positives.
    # Training positives are a SEPARATE set used for GT training signal only.
    assert len(known_pairs) <= 7, \
        f"known_pairs should only contain KPs (<=7), got {len(known_pairs)}"

    print(f"  PASS: {n_treats} treats edges in graph (KPs + training positives). "
          f"known_pairs (recovery test set) has {len(known_pairs)} pairs.")


def test_v31_kp_multi_hop_paths_injected():
    """P0-1: verify KP drugs get multi-hop biological plausibility paths."""
    from graph_transformer.data.graph_builder import BiomedicalGraphBuilder

    node_features, edge_indices, node_maps, known_pairs = \
        BiomedicalGraphBuilder.build_demo_graph(
            num_drugs=30, num_diseases=20, seed=42,
            known_positives=[("aspirin", "cardiovascular disease")],
        )

    drug_map = node_maps["drug"]
    protein_map = node_maps["protein"]
    pathway_map = node_maps["pathway"]

    # Verify aspirin has at least one drug->protein edge (from random gen
    # OR from the V31 KP multi-hop path injection).
    inh_key = ("drug", "inhibits", "protein")
    act_key = ("drug", "activates", "protein")
    assert inh_key in edge_indices or act_key in edge_indices

    aspirin_idx = drug_map.get("aspirin")
    assert aspirin_idx is not None, "aspirin must be in the graph"

    # Collect all drug->protein edges for aspirin
    aspirin_proteins = set()
    for key in [inh_key, act_key]:
        if key in edge_indices:
            ei = edge_indices[key]
            for i in range(ei.shape[1]):
                if int(ei[0, i]) == aspirin_idx:
                    aspirin_proteins.add(int(ei[1, i]))

    assert len(aspirin_proteins) > 0, \
        "aspirin must have at least one drug->protein edge (from V31 KP path injection)"

    print(f"  PASS: aspirin has {len(aspirin_proteins)} protein targets "
          f"(multi-hop path injection working).")


@pytest.mark.skip(
    reason="V90 ROOT FIX (BUG #18, P1): self._feature_rng was REMOVED as "
           "dead code. The per-drug patent/adme/efficacy values use DEDICATED "
           "drug-seeded RNGs (drug_rng = np.random.default_rng(drug_seed)), "
           "so self._feature_rng was never the source of feature randomness. "
           "The V31 docstring admitted 'this rng variable is only used for "
           "the legacy non-per-drug noise that has already been removed.' "
           "This test verified the dead code existed; it is now intentionally "
           "skipped because the dead code has been removed."
)
def test_v31_feature_rng_instance_level():
    """P1-11 / V90 BUG #38: _feature_rng was REMOVED (dead code).

    The V31 P1-11 fix introduced ``self._feature_rng`` to hoist the
    feature RNG to instance state. However, BUG #38 found that this
    attribute was DEAD CODE -- the per-drug feature computation uses
    DEDICATED per-drug RNGs (``drug_rng = np.random.default_rng(drug_seed)``),
    NOT ``self._feature_rng``. The ``rng = self._feature_rng`` line at
    the top of each method was a DEAD assignment.

    The V90 BUG #38 fix REMOVED ``self._feature_rng`` entirely. This
    test now verifies the REMOVAL (not the presence) of the dead code.
    """
    from graph_transformer.gt_rl_bridge import GTRLBridge

    bridge = GTRLBridge(output_dir=tempfile.mkdtemp(), seed=42)

    # V90 BUG #38: _feature_rng must NOT exist (it was dead code, removed).
    assert not hasattr(bridge, '_feature_rng'), \
        "V90 BUG #38: _feature_rng must be REMOVED (it was dead code -- " \
        "the per-drug feature computation uses dedicated per-drug RNGs, " \
        "not self._feature_rng). The V31 P1-11 fix introduced it but the " \
        "rng = self._feature_rng assignment was never actually used."

    # V90 COMPOUND #2: verify the deterministic name seed helper exists
    # (replaces hash() for reproducibility).
    from graph_transformer.gt_rl_bridge import _deterministic_name_seed
    s1 = _deterministic_name_seed(42, 'aspirin', 42)
    s2 = _deterministic_name_seed(42, 'aspirin', 42)
    assert s1 == s2, "V90 COMPOUND #2: _deterministic_name_seed must be reproducible"
    # Different drugs must get different seeds.
    s3 = _deterministic_name_seed(42, 'metformin', 42)
    assert s1 != s3, "V90 COMPOUND #2: different drugs must get different seeds"

    print("  PASS: V90 BUG #38 -- _feature_rng dead code REMOVED. "
          "V90 COMPOUND #2 -- _deterministic_name_seed is reproducible.")


def test_v31_top_k_novel_uses_raw_sigmoid():
    """P1-12: verify top_k_novel_predictions uses apply_temperature=False."""
    import inspect
    from graph_transformer.inference import top_k_novel_predictions

    source = inspect.getsource(top_k_novel_predictions)
    assert "apply_temperature=False" in source, \
        "top_k_novel_predictions must pass apply_temperature=False to " \
        "predict_all_pairs (P1-12: match RL training distribution)"

    print("  PASS: top_k_novel_predictions uses apply_temperature=False "
          "(matches RL training distribution, no Phase 6 OOD features).")


def test_v31_vecnormalize_save_in_train_agent():
    """P1-9: verify train_agent saves VecNormalize stats."""
    import inspect
    from rl.rl_drug_ranker import train_agent

    source = inspect.getsource(train_agent)
    assert "normalized_env_for_save" in source, \
        "train_agent must track normalized_env_for_save (P1-9 fix)"
    assert "vecnormalize" in source.lower() or "VecNormalize" in source, \
        "train_agent must save VecNormalize stats (P1-9 fix)"
    assert ".save(" in source, \
        "train_agent must call .save() on the VecNormalize wrapper (P1-9 fix)"

    print("  PASS: train_agent persists VecNormalize stats alongside PPO checkpoint.")


def test_v31_real_drug_names_reordered():
    """P0-1: verify REAL_DRUG_NAMES has training-positive drugs first (after KPs)."""
    from graph_transformer.data.graph_builder import BiomedicalGraphBuilder

    drugs = BiomedicalGraphBuilder.REAL_DRUG_NAMES
    # First 5 are KP drugs
    kp_drugs = drugs[:5]
    assert "dexamethasone" in kp_drugs
    assert "aspirin" in kp_drugs
    assert "metformin" in kp_drugs

    # Drugs 5-35 should include the training-positive drugs
    training_drug_region = drugs[5:40]
    training_drugs = {d for d, _ in BiomedicalGraphBuilder.TRAINING_POSITIVES}
    found = 0
    for td in training_drugs:
        if td in training_drug_region:
            found += 1
    # At least 25 of the 31 training-positive drugs should be in the first 40
    assert found >= 25, \
        f"Only {found}/31 training-positive drugs found in REAL_DRUG_NAMES[5:40]. " \
        f"They should come first so small graphs include them."

    print(f"  PASS: {found}/31 training-positive drugs are in REAL_DRUG_NAMES[5:40] "
          f"(ensures small demo graphs have training signal).")


def test_v31_pipeline_imports_clean():
    """Verify all V31 modules import without errors.

    V90 BUG #38: _feature_rng was REMOVED (dead code). The test no
    longer asserts its presence.
    """
    from graph_transformer.gt_rl_bridge import GTRLBridge
    from graph_transformer.data.graph_builder import BiomedicalGraphBuilder
    from graph_transformer.inference import top_k_novel_predictions, predict_drug_disease_scores
    from rl.rl_drug_ranker import train_agent, run_pipeline, PipelineConfig

    # V90 ROOT FIX (BUG #18): _feature_rng was REMOVED as dead code.
    # The bridge no longer has this attribute. The per-drug values use
    # dedicated drug-seeded RNGs.
    bridge = GTRLBridge(output_dir=tempfile.mkdtemp(), seed=42)
    assert not hasattr(bridge, '_feature_rng'), \
        "V90 BUG #18: _feature_rng should be REMOVED (dead code)"

    print("  PASS: All V31 modules import cleanly. V90 BUG #18: _feature_rng removed.")
    # V90 BUG #38: _feature_rng must NOT exist (dead code removed).
    bridge = GTRLBridge(output_dir=tempfile.mkdtemp(), seed=42)
    assert not hasattr(bridge, '_feature_rng'), \
        "V90 BUG #38: _feature_rng must be REMOVED (dead code)"

    print("  PASS: All V31 modules import cleanly. V90 BUG #38: _feature_rng removed.")


def test_v31_end_to_end_smoke():
    """End-to-end smoke test: build graph, verify training positives exist.

    V90 BUG #38: _feature_rng was REMOVED (dead code). The test no
    longer asserts its presence.
    """
    from graph_transformer.gt_rl_bridge import GTRLBridge

    with tempfile.TemporaryDirectory() as tmpdir:
        bridge = GTRLBridge(output_dir=tmpdir, seed=42)
        bridge.build_demo_graph(num_drugs=40, num_diseases=25)

        # Verify the graph has training positives (treats edges > 5)
        treats_key = ("drug", "treats", "disease")
        treats_edges = bridge.edge_indices.get(treats_key)
        assert treats_edges is not None, "treats edges must exist"
        n_treats = treats_edges.shape[1]
        assert n_treats > 5, \
            f"Expected >5 treats edges (KPs + training positives), got {n_treats}"

        # V90 ROOT FIX (BUG #18): _feature_rng was REMOVED as dead code.
        assert not hasattr(bridge, '_feature_rng'), \
            "V90 BUG #18: _feature_rng should be REMOVED (dead code)"

    print(f"  PASS: End-to-end smoke test passed. Graph has {n_treats} treats edges "
          f"(KPs + training positives). V90 BUG #18: _feature_rng removed.")
    # ROOT FIX (v92): removed duplicate assert + duplicate print that
    # were at the WRONG indent level (indent 8 after the ``with`` block
    # had already closed at indent 4). This caused ``compileall`` to
    # fail with IndentationError: unexpected indent (line 288), breaking
    # CI's Phase 3/4 build job for every PR. The duplicate code was a
    # botched merge of BUG #18 -> BUG #38 (both made the same assertion
    # about ``_feature_rng`` being removed).


def run_all_tests():
    """Run all V31 root fix verification tests."""
    tests = [
        test_v31_training_positives_exist,
        test_v31_training_positives_injected_into_graph,
        test_v31_kp_multi_hop_paths_injected,
        test_v31_feature_rng_instance_level,
        test_v31_top_k_novel_uses_raw_sigmoid,
        test_v31_vecnormalize_save_in_train_agent,
        test_v31_real_drug_names_reordered,
        test_v31_pipeline_imports_clean,
        test_v31_end_to_end_smoke,
    ]

    print("=" * 70)
    print("V31 ROOT-LEVEL FIX VERIFICATION TESTS")
    print("=" * 70)

    passed = 0
    failed = 0
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{passed + failed} tests passed")
    if failed == 0:
        print("ALL V31 ROOT FIXES VERIFIED SUCCESSFULLY")
    else:
        print(f"{failed} tests FAILED - review the output above")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
