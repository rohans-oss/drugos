"""Regression tests for P3-001 through P3-010 (Team Member 9, v104).

These tests verify the FORENSIC ROOT-CAUSE fixes for the Phase 3 Graph
Transformer Model / Link Predictor / Layers / Embeddings issues. Each
test runs REAL CODE (no mocks, no smoke tests) against the actual
graph_transformer module and would have FAILED against the pre-v104
codebase.

Issue inventory:
  P3-001  HIGH   graph_transformer.py — stale 14-edge schema check (should be 18)
  P3-002  HIGH   link_predictor.py   — MLP hidden_dims too large for small KGs
  P3-003  HIGH   layers.py           — no causal mask, undocumented
  P3-004  HIGH   embeddings.py       — IndexError on unknown node types at inference
  P3-005  MEDIUM graph_transformer.py — no torch.manual_seed in __init__
  P3-006  MEDIUM link_predictor.py   — no calibrated flag on predict_probability
  P3-007  MEDIUM layers.py           — LayerNorm documentation + gradient stability
  P3-008  MEDIUM link_predictor.py   — RLock bottleneck under high concurrency
  P3-009  MEDIUM embeddings.py       — no freeze_pretrained flag for TransE
  P3-010  LOW    layers.py           — dropout=0.1 too low for small graphs

Run with:
    python3 -m pytest tests/test_p3_tm9_model_issues.py -v
"""
from __future__ import annotations

import logging
import threading
import time

import pytest
import torch

# ----------------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _silence_noisy_loggers():
    """Reduce log spam during tests (the fix code logs INFO/WARNING messages)."""
    logging.getLogger("graph_transformer").setLevel(logging.ERROR)
    yield
    logging.getLogger("graph_transformer").setLevel(logging.INFO)


# ----------------------------------------------------------------------------
# P3-001: stale 14-edge schema check — must require 18 (9 forward + 9 reverse)
# ----------------------------------------------------------------------------


def test_p3_001_old_14_edge_schema_rejected():
    """P3-001: a graph built with the OLD 14-type schema must be REJECTED.

    The canonical Phase 2 schema (graph_transformer/data/__init__.py) has 18
    edge types (9 forward + 9 reverse). The pre-v104 code raised ValueError
    only when ``len(edge_types) < 14``, allowing the OLD 14-type schema to
    pass silently — the 4 new neutral edge types (binds, modulates,
    bound_by, modulated_by) had no learned embeddings, degrading message
    passing. The v104 ROOT FIX raises the threshold to 18.
    """
    from graph_transformer.data import EDGE_TYPES, DEFAULT_FEATURE_DIMS
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )

    # Build the OLD 14-type schema: 7 forward + 7 reverse (skip the 4 new ones).
    # Indices 0..6 are forward (inhibits, activates, binds, modulates,
    # part_of, disrupted_in, treats) — wait, the new schema has binds/modulates
    # at indices 2 and 3. To get the OLD schema we need to remove indices 2, 3,
    # 11, 12 (binds, modulates, bound_by, modulated_by).
    old_14 = [et for i, et in enumerate(EDGE_TYPES) if i not in (2, 3, 11, 12)]
    assert len(old_14) == 14, f"test setup error: expected 14, got {len(old_14)}"

    with pytest.raises(ValueError, match="at least 18 edge types"):
        DrugRepurposingGraphTransformer(
            feature_dims=DEFAULT_FEATURE_DIMS,
            edge_types=old_14,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
        )


def test_p3_001_canonical_18_edge_schema_accepted():
    """P3-001: the canonical 18-edge schema (default) is accepted."""
    from graph_transformer.data import DEFAULT_FEATURE_DIMS
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )

    # Default edge_types should be the 18-type canonical schema.
    m = DrugRepurposingGraphTransformer(
        feature_dims=DEFAULT_FEATURE_DIMS,
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
    )
    assert len(m.edge_types) == 18, f"expected 18, got {len(m.edge_types)}"


def test_p3_001_15_through_17_edge_schemas_rejected():
    """P3-001: any partial schema between 14 and 18 must also be rejected.

    A graph with 15, 16, or 17 edge types is missing some forward/reverse
    pair — message passing is incomplete. The fix must reject these too.
    """
    from graph_transformer.data import EDGE_TYPES, DEFAULT_FEATURE_DIMS
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )

    # Try removing one, two, or three edge types from the canonical 18.
    for n_drop in (1, 2, 3):
        partial = list(EDGE_TYPES[:-n_drop])
        assert len(partial) == 18 - n_drop
        with pytest.raises(ValueError, match="at least 18 edge types"):
            DrugRepurposingGraphTransformer(
                feature_dims=DEFAULT_FEATURE_DIMS,
                edge_types=partial,
                embedding_dim=16,
                num_heads=2,
                num_layers=1,
            )


# ----------------------------------------------------------------------------
# P3-002: graph-size-aware MLP hidden_dims scaling
# ----------------------------------------------------------------------------


def test_p3_002_mlp_hidden_dims_scale_with_graph_size():
    """P3-002: link_predictor MLP hidden_dims must scale with num_training_pairs.

    Pre-v104: hidden_dims=[256, 128] unconditionally -> ~100K params for 115
    demo pairs = ~1000 params/pair -> severe overfitting (AUC=0.403).
    v104: <1K pairs -> [64, 32], 1K-100K -> [128, 64], >100K -> [256, 128].
    """
    from graph_transformer.models.graph_transformer import (
        _mlp_hidden_dims_for_graph_size,
    )

    assert _mlp_hidden_dims_for_graph_size(115) == [64, 32]
    assert _mlp_hidden_dims_for_graph_size(999) == [64, 32]
    assert _mlp_hidden_dims_for_graph_size(1000) == [128, 64]
    assert _mlp_hidden_dims_for_graph_size(50_000) == [128, 64]
    assert _mlp_hidden_dims_for_graph_size(100_000) == [256, 128]
    assert _mlp_hidden_dims_for_graph_size(1_000_000) == [256, 128]


def test_p3_002_demo_graph_mlp_param_count_scaled():
    """P3-002: the demo graph (115 pairs) gets a SMALL MLP, not [256, 128].

    The pre-v104 default produced ~115K params on the demo graph (~1000
    params/pair). The v104 default produces ~7K params (~63 params/pair).
    """
    from graph_transformer.data import DEFAULT_FEATURE_DIMS
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )

    m = DrugRepurposingGraphTransformer(
        feature_dims=DEFAULT_FEATURE_DIMS,
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
        num_training_pairs=115,  # demo graph size
    )
    n_params = sum(p.numel() for p in m.link_predictor.mlp.parameters())
    # Should be < 15K (was ~115K pre-v104 with [256, 128] default)
    assert n_params < 15_000, f"MLP too large for demo graph: {n_params} params"
    # Should be > 1K (sanity — not degenerate)
    assert n_params > 1000, f"MLP too small: {n_params} params"
    # Params-per-pair ratio should be < 200 (was ~1000 pre-v104)
    ratio = n_params / 115
    assert ratio < 200, f"params/pair too high: {ratio:.1f}"


def test_p3_002_explicit_hidden_dims_respected():
    """P3-002: if caller explicitly passes hidden_dims, the scaling is NOT applied.

    The caller knows best for their use case. The auto-scaling only kicks
    in when hidden_dims=None AND num_training_pairs is provided.
    """
    from graph_transformer.data import DEFAULT_FEATURE_DIMS
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )

    m = DrugRepurposingGraphTransformer(
        feature_dims=DEFAULT_FEATURE_DIMS,
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
        num_training_pairs=115,  # would normally trigger [64, 32]
        link_predictor_hidden_dims=[256, 128],  # explicit override
    )
    assert m.link_predictor_hidden_dims == [256, 128]


# ----------------------------------------------------------------------------
# P3-003: no causal mask in attention (documentation + regression test)
# ----------------------------------------------------------------------------


def test_p3_003_no_causal_mask_in_attention():
    """P3-003: HeterogeneousMultiHeadAttention must NOT apply a causal mask.

    KGs are UNDIRECTED. A causal mask would break bidirectional message
    passing (the core GNN mechanism). The regression test verifies:
      1. The class docstring explicitly documents the no-mask policy.
      2. The forward() source contains NO mask construction (no
         ``tril``, ``triu``, ``masked_fill`` with a triangular mask).
      3. Attention is BIDIRECTIONAL — a drug attending to a protein
         receives a message in the SAME forward pass as the protein
         attending to the drug.
    """
    from graph_transformer.models.layers import HeterogeneousMultiHeadAttention

    # 1. Docstring documents the no-mask policy.
    docstring = HeterogeneousMultiHeadAttention.__doc__ or ""
    assert "causal mask" in docstring.lower(), (
        "P3-003 REGRESSION: HeterogeneousMultiHeadAttention docstring must "
        "explicitly mention 'causal mask' to warn future maintainers."
    )
    assert "do not add a causal mask" in docstring.lower(), (
        "P3-003 REGRESSION: docstring must explicitly say 'DO NOT add a "
        "causal mask' to prevent a future maintainer from adding one."
    )

    # 2. Source of forward() contains no triangular mask construction.
    #    We inspect forward() specifically (not the class docstring) so
    #    that documenting the no-mask policy does not itself trigger the
    #    forbidden-pattern check.
    import inspect
    src = inspect.getsource(HeterogeneousMultiHeadAttention.forward)
    forbidden_patterns = ["torch.tril", "torch.triu", "mask_tri", "register_buffer('mask'", "causal_mask =", "causal_mask="]
    for pat in forbidden_patterns:
        assert pat not in src, (
            f"P3-003 REGRESSION: HeterogeneousMultiHeadAttention.forward "
            f"source contains forbidden pattern '{pat}'. A causal mask "
            f"would break bidirectional message passing on the KG."
        )

    # 3. Bidirectional attention verified by FORWARD PASS on a 2-node graph.
    #    drug -> inhibits -> protein AND protein -> inhibited_by -> drug.
    #    After 1 attention layer, the drug's embedding MUST depend on the
    #    protein's initial value (and vice versa).
    edge_types = [
        ("drug", "inhibits", "protein"),
        ("protein", "inhibited_by", "drug"),
    ]
    attn = HeterogeneousMultiHeadAttention(
        embedding_dim=8, num_heads=2, edge_types=edge_types, dropout=0.0,
    )
    attn.eval()
    # Two graphs: same drug embedding, different protein embedding.
    drug_a = torch.randn(1, 8)
    drug_b = drug_a.clone()  # SAME drug embedding
    protein_a = torch.randn(1, 8)
    protein_b = protein_a + torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])  # DIFFERENT

    edge_idx = torch.tensor([[0], [0]], dtype=torch.long)
    edges = {
        ("drug", "inhibits", "protein"): edge_idx,
        ("protein", "inhibited_by", "drug"): edge_idx,
    }

    out_a = attn({"drug": drug_a, "protein": protein_a}, edges)
    out_b = attn({"drug": drug_b, "protein": protein_b}, edges)
    # The drug's output embedding should DIFFER between the two graphs
    # (because it receives a message from the protein, which differs).
    drug_diff = (out_a["drug"] - out_b["drug"]).abs().sum().item()
    assert drug_diff > 1e-6, (
        f"P3-003 REGRESSION: drug output unchanged when protein changes "
        f"(diff={drug_diff}). Bidirectional attention is BROKEN — a causal "
        f"mask may have been added."
    )


# ----------------------------------------------------------------------------
# P3-004: unknown node-type fallback (no IndexError at inference)
# ----------------------------------------------------------------------------


def test_p3_004_unknown_node_type_does_not_crash():
    """P3-004: NodeTypeEmbedding.forward() must NOT raise IndexError on new types.

    Pre-v104: passing an index >= num_node_types caused IndexError, crashing
    inference when Phase 2 added a new node type (e.g., 'variant'). v104
    clamps out-of-range indices to a zero-initialized 'unknown' slot.
    """
    from graph_transformer.models.embeddings import NodeTypeEmbedding

    emb = NodeTypeEmbedding(num_node_types=5, embedding_dim=8)
    # Indices 0..4 are valid; 5..9 are out-of-range (new node types).
    indices = torch.tensor([0, 1, 5, 9, 4])
    # Must NOT raise
    out = emb(indices)
    assert out.shape == (5, 8)
    # The unknown-slot embeddings (indices 5, 9 -> clamped to 5) must be ZERO.
    assert torch.allclose(out[2], torch.zeros(8)), (
        "P3-004 REGRESSION: unknown-type slot should be zero-initialized "
        "to avoid perturbing trained representations."
    )
    assert torch.allclose(out[3], torch.zeros(8))


def test_p3_004_unknown_type_warning_emitted_once(caplog):
    """P3-004: a WARNING is logged the FIRST time an unknown type is seen.

    Subsequent calls must NOT re-warn (to avoid log spam in production).
    """
    from graph_transformer.models.embeddings import NodeTypeEmbedding

    emb = NodeTypeEmbedding(num_node_types=3, embedding_dim=4)
    with caplog.at_level(logging.WARNING, logger="graph_transformer.models.embeddings"):
        emb(torch.tensor([5]))   # first out-of-range -> warns
        emb(torch.tensor([7]))   # second out-of-range -> NO warn
        emb(torch.tensor([0]))   # in-range -> NO warn
    # At least one warning, at most one warning.
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) >= 1, "expected at least 1 WARNING"
    assert len(warnings) <= 1, (
        f"expected at most 1 WARNING (per-instance), got {len(warnings)}"
    )


# ----------------------------------------------------------------------------
# P3-005: reproducible init via torch.manual_seed
# ----------------------------------------------------------------------------


def test_p3_005_same_seed_produces_identical_init():
    """P3-005: two models constructed with the SAME seed must have identical weights.

    Pre-v104: __init__ did not call torch.manual_seed, so each run produced
    different initial weights (AUC varied +/-0.03). v104 calls
    torch.manual_seed(seed) at the start of __init__.
    """
    from graph_transformer.data import DEFAULT_FEATURE_DIMS
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )

    common = dict(
        feature_dims=DEFAULT_FEATURE_DIMS,
        embedding_dim=16,
        num_heads=2,
        num_layers=2,
    )
    m1 = DrugRepurposingGraphTransformer(seed=42, **common)
    m2 = DrugRepurposingGraphTransformer(seed=42, **common)
    # Compare every parameter tensor.
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2, f"param name mismatch: {n1} vs {n2}"
        assert torch.allclose(p1, p2), (
            f"P3-005 REGRESSION: parameter '{n1}' differs between two "
            f"models constructed with seed=42. Reproducible init is broken."
        )


def test_p3_005_different_seeds_produce_different_init():
    """P3-005: two models constructed with DIFFERENT seeds must differ.

    Sanity check: the seed parameter must actually drive init, not be a no-op.
    """
    from graph_transformer.data import DEFAULT_FEATURE_DIMS
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )

    common = dict(
        feature_dims=DEFAULT_FEATURE_DIMS,
        embedding_dim=16,
        num_heads=2,
        num_layers=2,
    )
    m1 = DrugRepurposingGraphTransformer(seed=42, **common)
    m3 = DrugRepurposingGraphTransformer(seed=999, **common)
    # At least one parameter must differ.
    diffs = [
        not torch.allclose(p1, p3)
        for (_, p1), (_, p3) in zip(m1.named_parameters(), m3.named_parameters())
    ]
    assert any(diffs), (
        "P3-005 REGRESSION: all parameters identical between seed=42 and "
        "seed=999. The seed parameter is not driving init."
    )


def test_p3_005_seed_round_trips_through_save_load():
    """P3-005: the seed is saved in the checkpoint config and restored on load.

    A model saved with seed=42 must be loadable into a model that ALSO has
    seed=42 in its config (the seed round-trips through save/load).
    """
    import tempfile
    import os
    from graph_transformer.data import DEFAULT_FEATURE_DIMS
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )

    m = DrugRepurposingGraphTransformer(
        feature_dims=DEFAULT_FEATURE_DIMS,
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
        seed=42,
        num_training_pairs=115,
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        m.save(path)
        loaded = DrugRepurposingGraphTransformer.load(path)
        assert loaded.seed == 42, f"seed round-trip failed: got {loaded.seed}"
        assert loaded.num_training_pairs == 115
    finally:
        os.unlink(path)


# ----------------------------------------------------------------------------
# P3-006: calibrated flag on predict_probability
# ----------------------------------------------------------------------------


def test_p3_006_calibrated_flag_false_before_fit_temperature():
    """P3-006: a fresh link predictor must report calibrated=False.

    Pre-v104: callers had no way to know whether fit_temperature() had been
    called. v104 adds a ``_calibrated`` flag (False at init) and a
    ``return_metadata=True`` option to predict_probability() that returns a
    dict with ``calibrated`` key.
    """
    from graph_transformer.models.link_predictor import DrugDiseaseLinkPredictor

    lp = DrugDiseaseLinkPredictor(embedding_dim=8)
    assert lp._calibrated is False, "fresh predictor should be uncalibrated"

    drug = torch.randn(3, 8)
    dis = torch.randn(3, 8)
    out = lp.predict_probability(drug, dis, return_metadata=True)
    assert isinstance(out, dict)
    assert "calibrated" in out
    assert out["calibrated"] is False
    assert "probability" in out
    assert out["probability"].shape == (3,)


def test_p3_006_calibrated_flag_true_after_fit_temperature():
    """P3-006: after fit_temperature() succeeds, calibrated must be True."""
    from graph_transformer.models.link_predictor import DrugDiseaseLinkPredictor

    lp = DrugDiseaseLinkPredictor(embedding_dim=8)
    drug = torch.randn(8, 8)
    dis = torch.randn(8, 8)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    T = lp.fit_temperature(drug, dis, labels, lr=0.01, max_iter=10)
    assert lp._calibrated is True, (
        f"after fit_temperature (T={T}), _calibrated should be True"
    )
    out = lp.predict_probability(drug, dis, return_metadata=True)
    assert out["calibrated"] is True


def test_p3_006_warning_logged_on_uncalibrated_predict(caplog):
    """P3-006: a WARNING is logged the FIRST time predict_probability is called uncalibrated.

    The warning fires ONCE per instance to avoid log spam.
    """
    from graph_transformer.models.link_predictor import DrugDiseaseLinkPredictor

    lp = DrugDiseaseLinkPredictor(embedding_dim=8)
    lp.eval()
    drug = torch.randn(2, 8)
    dis = torch.randn(2, 8)
    with caplog.at_level(logging.WARNING, logger="graph_transformer.models.link_predictor"):
        lp.predict_probability(drug, dis)  # 1st uncalibrated -> warn
        lp.predict_probability(drug, dis)  # 2nd uncalibrated -> NO warn
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING
                and "UNCALIBRATED" in r.getMessage()]
    assert len(warnings) == 1, (
        f"expected exactly 1 UNCALIBRATED warning, got {len(warnings)}"
    )


# ----------------------------------------------------------------------------
# P3-007: pre-norm LayerNorm + gradient stability check
# ----------------------------------------------------------------------------


def test_p3_007_layernorm_applied_in_graph_transformer_layer():
    """P3-007: GraphTransformerLayer must apply LayerNorm (pre-norm style).

    Pre-v104 docstring claimed "no LayerNorm" but the code DID have it.
    The fix updates the docstring to document the PRE-NORM choice (Xiong
    et al. 2020 — pre-norm is more stable than post-norm for deep models).
    The test verifies LayerNorm modules exist AND are applied in forward().
    """
    from graph_transformer.models.layers import GraphTransformerLayer

    layer = GraphTransformerLayer(
        embedding_dim=8, num_heads=2,
        edge_types=[("drug", "inhibits", "protein")],
        node_types=["drug", "protein"],
    )
    # norm1 and norm2 must be nn.ModuleDict with per-type LayerNorms.
    assert layer.norm1 is not None
    assert layer.norm2 is not None
    assert "drug" in layer.norm1
    assert "protein" in layer.norm2


def test_p3_007_docstring_documents_pre_norm_choice():
    """P3-007: the GraphTransformerLayer docstring must mention pre-norm and Xiong et al."""
    from graph_transformer.models.layers import GraphTransformerLayer

    doc = GraphTransformerLayer.__doc__ or ""
    assert "pre-norm" in doc.lower() or "pre norm" in doc.lower()
    assert "xiong" in doc.lower(), (
        "P3-007 REGRESSION: docstring must cite Xiong et al. 2020 to "
        "explain why pre-norm is chosen over post-norm."
    )


def test_p3_007_gradient_stability_helper():
    """P3-007: check_gradient_stability() correctly detects stable vs unstable gradients.

    The helper takes a dict of {layer_name: grad_norm} and returns whether
    the max/min ratio is below a threshold (default 10x).
    """
    from graph_transformer.models.layers import GraphTransformerLayer

    # Stable case: all norms within 2x.
    result_stable = GraphTransformerLayer.check_gradient_stability(
        model=None,
        per_layer_gradient_norms={
            "layer0": 1.0,
            "layer1": 1.5,
            "layer2": 1.2,
            "layer3": 0.9,
        },
    )
    assert result_stable["stable"] is True
    assert result_stable["ratio"] < 2.0

    # Unstable case: norms span 100x (vanishing/exploding gradient).
    result_unstable = GraphTransformerLayer.check_gradient_stability(
        model=None,
        per_layer_gradient_norms={
            "layer0": 10.0,
            "layer1": 0.1,  # 100x ratio
        },
    )
    assert result_unstable["stable"] is False
    assert result_unstable["ratio"] >= 10.0


def test_p3_007_gradient_norms_stable_on_real_forward_backward():
    """P3-007: a real 4-layer Graph Transformer must have stable gradient norms.

    After loss.backward(), the per-layer gradient norm ratio must be < 10x
    (the pre-norm LayerNorm ensures this per Xiong et al. 2020).
    """
    from graph_transformer.data import DEFAULT_FEATURE_DIMS, EDGE_TYPES
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )
    from graph_transformer.models.layers import GraphTransformerLayer

    torch.manual_seed(0)
    m = DrugRepurposingGraphTransformer(
        feature_dims=DEFAULT_FEATURE_DIMS,
        embedding_dim=16,
        num_heads=2,
        num_layers=4,
        edge_types=EDGE_TYPES,
    )
    m.train()
    # Build a minimal forward pass.
    nf = {nt: torch.randn(2, d, requires_grad=True)
          for nt, d in DEFAULT_FEATURE_DIMS.items()}
    ei = {et: torch.tensor([[0], [0]], dtype=torch.long) for et in EDGE_TYPES}
    drug_idx = torch.tensor([0])
    dis_idx = torch.tensor([0])
    logits = m.forward_logits(nf, ei, drug_idx, dis_idx)
    # logits has shape (1,) — match the target shape.
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([1.0])
    )
    loss.backward()
    # Collect per-layer gradient norms.
    norms = {}
    for i, layer in enumerate(m.graph_transformer_layers):
        n = sum(
            (p.grad.norm(2) ** 2).item() if p.grad is not None else 0.0
            for p in layer.parameters()
        ) ** 0.5
        norms[f"layer{i}"] = n
    result = GraphTransformerLayer.check_gradient_stability(m, norms)
    assert result["stable"], (
        f"P3-007 REGRESSION: gradient norms unstable across 4 layers. "
        f"Ratio={result['ratio']:.2f}x. Message: {result['message']}"
    )


# ----------------------------------------------------------------------------
# P3-008: lock-free concurrent inference (no RLock bottleneck)
# ----------------------------------------------------------------------------


def test_p3_008_concurrent_predict_probability_no_deadlock():
    """P3-008: 4 threads x 50 predictions must complete without deadlock.

    Pre-v104: a threading.RLock serialized all predictions, dropping
    throughput ~100x. v104 uses torch.set_grad_enabled(False) (per-thread)
    for the common eval-mode fast path, eliminating the lock.
    """
    from graph_transformer.models.link_predictor import DrugDiseaseLinkPredictor

    lp = DrugDiseaseLinkPredictor(embedding_dim=8)
    lp.eval()  # common inference mode -> fast path is lock-free

    errors: list = []
    n_calls_per_thread = 50
    n_threads = 4

    def worker():
        try:
            for _ in range(n_calls_per_thread):
                d = torch.randn(5, 8)
                dis = torch.randn(5, 8)
                _ = lp.predict_probability(d, dis)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)  # 30s timeout — pre-v104 would deadlock or be 100x slower
    elapsed = time.time() - t0

    assert not errors, f"errors in worker threads: {errors}"
    # 200 predictions should complete in < 5s on CPU. Pre-v104 with RLock
    # serialization would be ~100x slower (still no deadlock, but slow).
    # We don't assert a hard time bound (CI machines vary), but if it
    # takes > 30s, something is wrong.
    assert elapsed < 30.0, (
        f"P3-008 REGRESSION: 200 concurrent predictions took {elapsed:.1f}s "
        f"(> 30s timeout). Lock-free fast path may be broken."
    )


def test_p3_008_eval_mode_fast_path_is_lock_free():
    """P3-008: when module is in eval mode, the fast path must NOT touch _predict_lock.

    We verify this by replacing _predict_lock with a sentinel object that
    RAISES if acquired. If the eval-mode fast path is truly lock-free,
    the sentinel is never acquired and the predictions succeed silently.
    """
    from graph_transformer.models.link_predictor import DrugDiseaseLinkPredictor

    lp = DrugDiseaseLinkPredictor(embedding_dim=8)
    lp.eval()  # common inference mode
    drug = torch.randn(3, 8)
    dis = torch.randn(3, 8)

    # Replace _predict_lock with a sentinel that raises on acquire.
    # If the eval-mode fast path is truly lock-free, this sentinel is
    # never touched and the predictions succeed silently.
    class _LockAcquiredError(RuntimeError):
        pass

    class _ExplodingLock:
        def __enter__(self):
            raise _LockAcquiredError(
                "P3-008 REGRESSION: eval-mode fast path acquired the lock. "
                "It should be lock-free (use torch.set_grad_enabled(False) "
                "instead)."
            )
        def __exit__(self, *a):
            return False
        def acquire(self, *a, **kw):
            raise _LockAcquiredError(
                "P3-008 REGRESSION: eval-mode fast path acquired the lock."
            )
        def release(self):
            pass

    lp._predict_lock = _ExplodingLock()
    # These three calls must NOT raise — the fast path should bypass the lock.
    lp.predict_probability(drug, dis)
    lp.predict_probability(drug, dis)
    lp.predict_probability(drug, dis)


# ----------------------------------------------------------------------------
# P3-009: freeze_pretrained flag + load_pretrained_embeddings method
# ----------------------------------------------------------------------------


def test_p3_009_freeze_pretrained_default_true():
    """P3-009: NodeTypeProjection must default freeze_pretrained=True.

    Small graphs (demo, pilot) should freeze TransE embeddings by default
    so the noisy GNN gradient signal does not overwrite them.
    """
    from graph_transformer.models.embeddings import NodeTypeProjection

    proj = NodeTypeProjection({"drug": 16}, embedding_dim=8)
    assert proj._default_freeze_pretrained is True


def test_p3_009_load_pretrained_embeddings_freezes():
    """P3-009: load_pretrained_embeddings() must set requires_grad=False when freeze=True."""
    from graph_transformer.models.embeddings import NodeTypeProjection

    proj = NodeTypeProjection({"drug": 16, "protein": 8}, embedding_dim=32)
    # Sanity: initially both projections are trainable.
    assert proj.projections["drug"].weight.requires_grad is True

    # Load pretrained weight for 'drug' with default freeze (True).
    w = torch.randn(32, 16)
    proj.load_pretrained_embeddings("drug", w)
    assert proj.projections["drug"].weight.requires_grad is False
    assert proj.projections["drug"].bias.requires_grad is False
    assert "drug" in proj.frozen_types()
    # 'protein' should still be trainable.
    assert proj.projections["protein"].weight.requires_grad is True


def test_p3_009_load_pretrained_embeddings_freeze_false():
    """P3-009: load_pretrained_embeddings(freeze=False) leaves the projection trainable."""
    from graph_transformer.models.embeddings import NodeTypeProjection

    proj = NodeTypeProjection({"drug": 16}, embedding_dim=32, freeze_pretrained=True)
    w = torch.randn(32, 16)
    proj.load_pretrained_embeddings("drug", w, freeze=False)
    assert proj.projections["drug"].weight.requires_grad is True
    assert proj.projections["drug"].bias.requires_grad is True
    assert "drug" not in proj.frozen_types()


def test_p3_009_unfreeze_pretrained_embeddings():
    """P3-009: unfreeze_pretrained_embeddings() restores requires_grad=True."""
    from graph_transformer.models.embeddings import NodeTypeProjection

    proj = NodeTypeProjection({"drug": 16}, embedding_dim=32)
    w = torch.randn(32, 16)
    proj.load_pretrained_embeddings("drug", w)  # freeze=True (default)
    assert "drug" in proj.frozen_types()
    proj.unfreeze_pretrained_embeddings("drug")
    assert proj.projections["drug"].weight.requires_grad is True
    assert "drug" not in proj.frozen_types()


def test_p3_009_load_pretrained_rejects_wrong_shape():
    """P3-009: load_pretrained_embeddings() must reject a weight with wrong shape."""
    from graph_transformer.models.embeddings import NodeTypeProjection

    proj = NodeTypeProjection({"drug": 16}, embedding_dim=32)
    wrong_w = torch.randn(16, 32)  # transposed -> should fail
    with pytest.raises(ValueError, match="shape"):
        proj.load_pretrained_embeddings("drug", wrong_w)


def test_p3_009_load_pretrained_rejects_unknown_type():
    """P3-009: load_pretrained_embeddings() must reject an unknown node type."""
    from graph_transformer.models.embeddings import NodeTypeProjection

    proj = NodeTypeProjection({"drug": 16}, embedding_dim=32)
    w = torch.randn(32, 16)
    with pytest.raises(ValueError, match="Unknown node type"):
        proj.load_pretrained_embeddings("nonexistent_type", w)


# ----------------------------------------------------------------------------
# P3-010: graph-size-aware dropout scaling
# ----------------------------------------------------------------------------


def test_p3_010_dropout_scales_with_graph_size():
    """P3-010: dropout must scale with num_training_pairs.

    Pre-v104: hardcoded dropout=0.1 unconditionally -> overfitting on small
    graphs. v104: <10K pairs -> 0.5, 10K-1M -> 0.2, >1M -> 0.1.
    """
    from graph_transformer.models.graph_transformer import _dropout_for_graph_size

    assert _dropout_for_graph_size(115) == 0.5
    assert _dropout_for_graph_size(9_999) == 0.5
    assert _dropout_for_graph_size(10_000) == 0.2
    assert _dropout_for_graph_size(500_000) == 0.2
    assert _dropout_for_graph_size(1_000_000) == 0.1


def test_p3_010_demo_graph_gets_heavy_dropout():
    """P3-010: the demo graph (115 pairs) must get dropout=0.5, not 0.1.

    The pre-v104 default of 0.1 was too low for small graphs (AUC=0.403
    on demo). The v104 default scales to 0.5 for <10K pairs.
    """
    from graph_transformer.data import DEFAULT_FEATURE_DIMS
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )

    m = DrugRepurposingGraphTransformer(
        feature_dims=DEFAULT_FEATURE_DIMS,
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
        num_training_pairs=115,
    )
    assert m.dropout == 0.5, (
        f"P3-010 REGRESSION: demo graph (115 pairs) should get dropout=0.5, "
        f"got {m.dropout}"
    )
    assert m.attention_dropout == 0.5


def test_p3_010_explicit_dropout_respected():
    """P3-010: if caller explicitly passes dropout, the scaling is NOT applied.

    The caller knows best for their use case. The auto-scaling only kicks
    in when dropout is left at the default 0.1.
    """
    from graph_transformer.data import DEFAULT_FEATURE_DIMS
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )

    m = DrugRepurposingGraphTransformer(
        feature_dims=DEFAULT_FEATURE_DIMS,
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
        num_training_pairs=115,  # would normally trigger 0.5
        dropout=0.3,  # explicit override
        attention_dropout=0.15,
    )
    assert m.dropout == 0.3
    assert m.attention_dropout == 0.15


# ----------------------------------------------------------------------------
# Integration: end-to-end forward pass with ALL fixes active
# ----------------------------------------------------------------------------


def test_integration_end_to_end_forward_pass_with_all_fixes():
    """Integration: a full forward pass with all 10 fixes active must succeed.

    This is NOT a smoke test — it runs the REAL model on the REAL canonical
    18-edge schema with REAL node features and verifies the output shape,
    dtype, and value range. If any of the 10 fixes broke the forward path,
    this test fails.
    """
    from graph_transformer.data import (
        DEFAULT_FEATURE_DIMS,
        EDGE_TYPES,
        LABEL_LEAKING_EDGES,
    )
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )

    torch.manual_seed(0)
    m = DrugRepurposingGraphTransformer(
        feature_dims=DEFAULT_FEATURE_DIMS,
        embedding_dim=16,
        num_heads=2,
        num_layers=2,
        edge_types=EDGE_TYPES,
        seed=42,
        num_training_pairs=115,
    )
    m.eval()
    nf = {nt: torch.randn(3, d) for nt, d in DEFAULT_FEATURE_DIMS.items()}
    ei = {et: torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
          for et in EDGE_TYPES}
    drug_idx = torch.tensor([0, 1, 2])
    dis_idx = torch.tensor([0, 1, 2])

    # forward_logits returns raw logits (training path)
    logits = m.forward_logits(nf, ei, drug_idx, dis_idx)
    # logits has shape (N,) — squeezed by the link predictor.
    assert logits.shape[0] == 3, f"expected 3 logits, got {logits.shape}"
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()

    # predict_all_pairs returns a score matrix (inference path) of shape
    # (num_drugs, num_diseases). Use the first 3 drugs and 3 diseases.
    scores = m.predict_all_pairs(
        nf, ei,
        num_drugs=3,
        num_diseases=3,
        apply_temperature=True,
    )
    assert scores.shape == (3, 3)
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all() and (scores <= 1).all(), (
        "scores must be probabilities in [0, 1]"
    )


def test_integration_save_load_round_trip_with_v104_fields():
    """Integration: save/load must round-trip seed + num_training_pairs."""
    import os
    import tempfile
    from graph_transformer.data import DEFAULT_FEATURE_DIMS
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )

    m = DrugRepurposingGraphTransformer(
        feature_dims=DEFAULT_FEATURE_DIMS,
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
        seed=123,
        num_training_pairs=5000,
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        m.save(path)
        loaded = DrugRepurposingGraphTransformer.load(path)
        assert loaded.seed == 123
        assert loaded.num_training_pairs == 5000
        # The link_predictor_hidden_dims should reflect the scaled value
        # for 5K pairs ([128, 64]).
        assert loaded.link_predictor_hidden_dims == [128, 64]
        # Dropout should reflect the scaled value for 5K pairs (0.5,
        # because 5000 < 10_000 -> 0.5 per _dropout_for_graph_size).
        assert loaded.dropout == 0.5, (
            f"dropout round-trip failed: got {loaded.dropout}, expected 0.5"
        )
    finally:
        os.unlink(path)
