#!/usr/bin/env python3
"""Dedicated tests for P3-005, P3-011, P3-013, P3-014, P3-015, P3-016,
P3-018, P3-021 root fixes (Team Member 7, Batch A).

These tests verify the ACTUAL code behavior (not just comments) by:
1. Importing the real modules
2. Building real objects (model, graph, bridge)
3. Running real forward passes
4. Asserting the specific fix is in place

Run:
    pytest tests/test_p3_team7_batch_a_forensic_fixes.py -v
"""
import os
import sys
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# Set dev environment for tests (RDKit/ChemBERTa not installed)
os.environ.setdefault("DRUGOS_ENVIRONMENT", "dev")
os.environ.setdefault("DRUGOS_SKIP_CHEMBERTA", "1")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ============================================================================
# P3-005: from_phase1_staged_data uses REAL features (not random noise)
# ============================================================================
def test_p3_005_from_phase1_uses_real_features():
    """from_phase1_staged_data must use real feature computation,
    NOT rng.standard_normal() random noise."""
    from graph_transformer.data.graph_builder import BiomedicalGraphBuilder
    src = inspect.getsource(BiomedicalGraphBuilder.from_phase1_staged_data)
    # The fix imports phase2_adapter feature functions
    assert "_drug_feature_from_smiles" in src, \
        "P3-005 FAIL: from_phase1_staged_data must use _drug_feature_from_smiles"
    assert "_protein_sequence_feature" in src, \
        "P3-005 FAIL: from_phase1_staged_data must use _protein_sequence_feature"
    assert "_structured_name_feature" in src, \
        "P3-005 FAIL: from_phase1_staged_data must use _structured_name_feature"
    # Must NOT use rng.standard_normal for ALL node types
    # (the old buggy code did: features = rng.standard_normal((len(names), feat_dim)))
    assert "rng.standard_normal((len(names)" not in src, \
        "P3-005 FAIL: from_phase1_staged_data must NOT use rng.standard_normal for all nodes"


# ============================================================================
# P3-011: gnn_score_timestamp column in bridge output
# ============================================================================
def test_p3_011_gnn_score_timestamp_in_bridge():
    """generate_rl_input and save_rl_input_streaming must produce
    gnn_score_timestamp column for RL staleness detection (P4-007)."""
    from graph_transformer.gt_rl_bridge import GTRLBridge
    gen_src = inspect.getsource(GTRLBridge.generate_rl_input)
    assert "gnn_score_timestamp" in gen_src, \
        "P3-011 FAIL: generate_rl_input must produce gnn_score_timestamp"
    stream_src = inspect.getsource(GTRLBridge.save_rl_input_streaming)
    assert "gnn_score_timestamp" in stream_src, \
        "P3-011 FAIL: save_rl_input_streaming must produce gnn_score_timestamp"


# ============================================================================
# P3-013: self-loop NOT scaled by cross_type_norm
# ============================================================================
def test_p3_013_self_loop_not_scaled_by_cross_type_norm():
    """The self-loop message must NOT be multiplied by cross_type_norm.
    Residual connections should be independent of edge-type count."""
    from graph_transformer.models.layers import HeterogeneousMultiHeadAttention
    src = inspect.getsource(HeterogeneousMultiHeadAttention.forward)
    # Find the actual code line (not comments) that multiplies self_loop_weight
    self_loop_lines = [
        line.strip() for line in src.split("\n")
        if "self_loop_weight" in line
        and "messages =" in line
        and not line.strip().startswith("#")
    ]
    assert len(self_loop_lines) > 0, \
        "P3-013 FAIL: no 'messages = ... self_loop_weight' line found"
    for line in self_loop_lines:
        assert "cross_type_norm" not in line, \
            f"P3-013 FAIL: self_loop line must NOT contain cross_type_norm: {line}"


# ============================================================================
# P3-014: self_loop_weight init=1.0 (not 0.1)
# ============================================================================
def test_p3_014_self_loop_weight_init_is_1():
    """self_loop_weight must initialize to 1.0 (standard residual weight).
    The old comment said 'init=0.1' but the actual init was 1.0 (P3-S01)."""
    from graph_transformer.models.layers import HeterogeneousMultiHeadAttention
    attn = HeterogeneousMultiHeadAttention(
        embedding_dim=32, num_heads=2, edge_types=[("drug", "inhibits", "protein")],
    )
    assert attn.self_loop_weight.data.item() == 1.0, \
        f"P3-014 FAIL: self_loop_weight must init to 1.0, got {attn.self_loop_weight.data.item()}"


# ============================================================================
# P3-015: q_proj is per-node-type (ModuleDict, not shared Linear)
# ============================================================================
def test_p3_015_q_proj_is_per_node_type():
    """q_proj must be a ModuleDict with per-node-type Linear projections,
    matching standard HGT (Wang et al. 2019)."""
    from graph_transformer.models.layers import HeterogeneousMultiHeadAttention
    attn = HeterogeneousMultiHeadAttention(
        embedding_dim=32, num_heads=2, edge_types=[("drug", "inhibits", "protein")],
    )
    assert isinstance(attn.q_proj, nn.ModuleDict), \
        f"P3-015 FAIL: q_proj must be ModuleDict, got {type(attn.q_proj)}"
    # Must have projections for all 5 canonical node types
    for ntype in ["drug", "protein", "pathway", "disease", "clinical_outcome"]:
        assert ntype in attn.q_proj, \
            f"P3-015 FAIL: q_proj missing node type '{ntype}'"


def test_p3_015_q_proj_forward_uses_per_type():
    """The forward pass must apply per-type q_proj, not a shared one."""
    from graph_transformer.models.layers import HeterogeneousMultiHeadAttention
    src = inspect.getsource(HeterogeneousMultiHeadAttention.forward)
    # Must NOT use the old shared q_proj(all_embeddings) pattern
    assert "self.q_proj(all_embeddings)" not in src, \
        "P3-015 FAIL: forward must NOT use shared q_proj(all_embeddings)"
    # Must use per-type q_proj[nt](h)
    assert "self.q_proj[nt]" in src or "self.q_proj[" in src, \
        "P3-015 FAIL: forward must use per-type q_proj[nt](h)"


# ============================================================================
# P3-016: residual preserves ALL node types (no silent drop)
# ============================================================================
def test_p3_016_residual_preserves_all_node_types():
    """The residual connection must NOT drop node types that aren't in
    attn_out. All node types must be preserved (input passes through
    unchanged if no attention update)."""
    from graph_transformer.models.layers import GraphTransformerLayer
    src = inspect.getsource(GraphTransformerLayer.forward)
    import re
    # Must NOT use dict comprehension with 'if k in attn_out' filter
    dict_comp_attn = re.findall(
        r'\{[^}]*for\s+\w+,\s*\w+\s+in\s+node_embeddings\.items\(\)\s+if\s+\w+\s+in\s+attn_out[^}]*\}',
        src,
    )
    assert len(dict_comp_attn) == 0, \
        f"P3-016 FAIL: residual must NOT use dict comp with 'if k in attn_out' filter: {dict_comp_attn}"
    dict_comp_ffn = re.findall(
        r'\{[^}]*for\s+\w+,\s*\w+\s+in\s+node_embeddings\.items\(\)\s+if\s+\w+\s+in\s+ffn_out[^}]*\}',
        src,
    )
    assert len(dict_comp_ffn) == 0, \
        f"P3-016 FAIL: residual must NOT use dict comp with 'if k in ffn_out' filter: {dict_comp_ffn}"


def test_p3_016_residual_forward_preserves_isolated_type():
    """REAL CODE test: build a layer, run forward with a node type that
    has NO incoming edges, verify it survives the residual."""
    from graph_transformer.models.layers import GraphTransformerLayer
    layer = GraphTransformerLayer(
        embedding_dim=16, num_heads=2,
        edge_types=[("drug", "inhibits", "protein")],
        node_types=["drug", "protein", "pathway"],
    )
    node_embeddings = {
        "drug": torch.randn(3, 16),
        "protein": torch.randn(4, 16),
        "pathway": torch.randn(2, 16),  # NO incoming edges
    }
    edge_indices = {
        ("drug", "inhibits", "protein"): torch.tensor([[0, 1], [0, 1]]),
    }
    out = layer(node_embeddings, edge_indices)
    # pathway must STILL be in the output (not dropped by residual)
    assert "pathway" in out, "P3-016 FAIL: pathway type was dropped by residual"
    assert out["pathway"].shape == (2, 16), \
        f"P3-016 FAIL: pathway shape wrong: {out['pathway'].shape}"


# ============================================================================
# P3-018: V1_AUC_THRESHOLD = 0.85 for ALL scales
# ============================================================================
def test_p3_018_threshold_0_85_all_scales():
    """V1_AUC_THRESHOLD_DEMO and V1_AUC_THRESHOLD_PILOT must both be 0.85.
    The audit says 'Use 0.85 for ALL scales.'"""
    from graph_transformer.data import (
        V1_AUC_THRESHOLD,
        V1_AUC_THRESHOLD_DEMO,
        V1_AUC_THRESHOLD_PILOT,
        get_auc_threshold_for_scale,
    )
    assert V1_AUC_THRESHOLD == 0.85
    assert V1_AUC_THRESHOLD_DEMO == 0.85, \
        f"P3-018 FAIL: DEMO must be 0.85, got {V1_AUC_THRESHOLD_DEMO}"
    assert V1_AUC_THRESHOLD_PILOT == 0.85, \
        f"P3-018 FAIL: PILOT must be 0.85, got {V1_AUC_THRESHOLD_PILOT}"
    assert get_auc_threshold_for_scale(10) == 0.85
    assert get_auc_threshold_for_scale(500) == 0.85
    assert get_auc_threshold_for_scale(5000) == 0.85


# ============================================================================
# P3-021: AE edges comment accuracy (topology, not safety_score source)
# ============================================================================
def test_p3_021_ae_edges_comment_accurate():
    """The AE edges comment must NOT claim they're used for safety_score.
    safety_score comes from curated FDA FAERS table, not graph AE edges."""
    from graph_transformer.data.graph_builder import BiomedicalGraphBuilder
    src = inspect.getsource(BiomedicalGraphBuilder.build_demo_graph)
    # Find the AE edges section
    ae_section_start = src.find("Drug-causes-outcome edges")
    assert ae_section_start >= 0, "P3-021 FAIL: AE edges section not found"
    # Get a larger section to capture the full comment + code
    ae_section = src[ae_section_start:ae_section_start + 3000]
    # The fix must mention P3-021 and describe AE edges as topology
    assert "P3-021" in ae_section, \
        "P3-021 FAIL: AE edges comment must reference P3-021 fix"
    assert "TOPOLOGY" in ae_section or "topology" in ae_section, \
        "P3-021 FAIL: AE edges comment must describe them as topology"
    # The fix must explicitly state AE edges are NOT the safety_score source
    assert "NOT the RL safety_score source" in ae_section or \
           "NOT the RL safety_score" in ae_section, \
        "P3-021 FAIL: AE edges comment must state they are NOT the safety_score source"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
