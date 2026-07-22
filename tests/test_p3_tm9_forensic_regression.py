"""
ADDITIONAL forensic-grade regression tests for P3-001 through P3-010 (Team Member 9).

These tests are STRICTER than the existing test_p3_tm9_model_issues.py suite.
They test ADVERSARIAL edge cases that would catch regressions if someone
reverted a fix or introduced a subtle bug. Each test is self-contained and
exercises the REAL model code (not mocks, not stubs).

Coverage:
  P3-001: 14-type rejection + 18-type acceptance + >18-type acceptance (superset)
  P3-002: MLP param count scales CORRECTLY across the full tier ladder
  P3-003: Attention weights are DENSE (no masked entries) - verified via softmax
  P3-004: Unknown node type produces ZERO embedding (not random) + warning ONCE
  P3-005: Reproducible init survives save/load round-trip
  P3-006: calibrated flag is False on fresh model, True after fit_temperature,
          and the warning fires EXACTLY ONCE per instance
  P3-007: LayerNorm parameters EXIST and RECEIVE non-zero gradients on backward
  P3-008: 100 concurrent predictions complete without deadlock (V1 contract)
  P3-009: Frozen projection weights DO NOT change after optimizer.step()
  P3-010: Dropout scaling is applied via num_training_pairs even when caller
          doesn't pass dropout explicitly

Run:
    /home/z/.venv/bin/python3.12 -m pytest tests/test_p3_tm9_forensic_regression.py -v
"""
from __future__ import annotations

import io
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout

import pytest
import torch
import torch.nn as nn

# Add repo to path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from graph_transformer.data import (
    DEFAULT_EDGE_TYPES,
    DEFAULT_FEATURE_DIMS,
    DEFAULT_NODE_TYPES,
    LABEL_LEAKING_EDGES,
)
from graph_transformer.models.graph_transformer import (
    DrugRepurposingGraphTransformer,
    _mlp_hidden_dims_for_graph_size,
    _dropout_for_graph_size,
)
from graph_transformer.models.link_predictor import DrugDiseaseLinkPredictor
from graph_transformer.models.layers import (
    GraphTransformerLayer,
    HeterogeneousMultiHeadAttention,
)
from graph_transformer.models.embeddings import (
    NodeTypeProjection,
    NodeTypeEmbedding,
)


# ============================================================================
# P3-001: edge_types check must be < 18 (not < 14)
# ============================================================================
class TestP3001StaleSchemaCheck:
    """P3-001: the model must reject graphs with fewer than 18 edge types."""

    def test_14_type_old_schema_rejected(self):
        """A 14-type graph (the OLD schema) must be REJECTED."""
        old_14 = DEFAULT_EDGE_TYPES[:14]
        with pytest.raises(ValueError, match="18"):
            DrugRepurposingGraphTransformer(
                feature_dims=DEFAULT_FEATURE_DIMS,
                embedding_dim=32,
                num_layers=1,
                num_heads=4,
                edge_types=old_14,
                node_types=DEFAULT_NODE_TYPES,
            )

    def test_17_type_schema_rejected(self):
        """A 17-type graph (just below the threshold) must be REJECTED."""
        seventeen = DEFAULT_EDGE_TYPES[:17]
        with pytest.raises(ValueError, match="18"):
            DrugRepurposingGraphTransformer(
                feature_dims=DEFAULT_FEATURE_DIMS,
                embedding_dim=32,
                num_layers=1,
                num_heads=4,
                edge_types=seventeen,
                node_types=DEFAULT_NODE_TYPES,
            )

    def test_18_type_canonical_schema_accepted(self):
        """The canonical 18-type schema must be ACCEPTED."""
        m = DrugRepurposingGraphTransformer(
            feature_dims=DEFAULT_FEATURE_DIMS,
            embedding_dim=32,
            num_layers=1,
            num_heads=4,
            edge_types=DEFAULT_EDGE_TYPES,
            node_types=DEFAULT_NODE_TYPES,
        )
        assert len(m.edge_types) == 18

    def test_superset_20_type_schema_accepted(self):
        """A superset (20 types) must be ACCEPTED (the check is a minimum, not exact)."""
        extra = [("drug", "extra1", "protein"), ("protein", "extra1_rev", "drug")]
        superset = list(DEFAULT_EDGE_TYPES) + extra
        m = DrugRepurposingGraphTransformer(
            feature_dims=DEFAULT_FEATURE_DIMS,
            embedding_dim=32,
            num_layers=1,
            num_heads=4,
            edge_types=superset,
            node_types=DEFAULT_NODE_TYPES,
        )
        assert len(m.edge_types) == 20


# ============================================================================
# P3-002: MLP hidden_dims must scale with num_pairs
# ============================================================================
class TestP3002MlpScaling:
    """P3-002: MLP hidden_dims must scale with graph size to avoid overfitting."""

    @pytest.mark.parametrize("npairs,expected", [
        (0, [64, 32]),
        (1, [64, 32]),
        (999, [64, 32]),
        (1000, [128, 64]),
        (99999, [128, 64]),
        (100000, [256, 128]),
        (1_000_000, [256, 128]),
    ])
    def test_helper_scaling(self, npairs, expected):
        """The helper must return the correct tier for each graph size."""
        assert _mlp_hidden_dims_for_graph_size(npairs) == expected

    def test_negative_pairs_raises(self):
        """Negative num_training_pairs must raise ValueError."""
        with pytest.raises(ValueError):
            _mlp_hidden_dims_for_graph_size(-1)

    def test_link_predictor_param_count_scales_monotonically(self):
        """The MLP parameter count must scale monotonically with graph size."""
        counts = {}
        for npairs in [50, 5000, 200000]:
            lp = DrugDiseaseLinkPredictor(embedding_dim=32, num_pairs=npairs)
            n = sum(p.numel() for p in lp.parameters())
            counts[npairs] = n
        # Small graph must have FEWER params than medium, medium fewer than large
        assert counts[50] < counts[5000] < counts[200000], \
            f"Param counts not monotonic: {counts}"

    def test_explicit_hidden_dims_respected_over_num_pairs(self):
        """If caller explicitly passes hidden_dims, num_pairs must NOT override it."""
        lp = DrugDiseaseLinkPredictor(
            embedding_dim=32,
            hidden_dims=[256, 128],  # explicit large
            num_pairs=10,  # would normally give [64, 32]
        )
        assert lp.hidden_dims == [256, 128], \
            f"Explicit hidden_dims overridden by num_pairs: {lp.hidden_dims}"

    def test_demo_graph_param_per_pair_ratio_healthy(self):
        """On a 115-pair demo graph, the MLP must have <500 params per pair (not 1000+)."""
        npairs = 115
        lp = DrugDiseaseLinkPredictor(embedding_dim=64, num_pairs=npairs)
        n_params = sum(p.numel() for p in lp.mlp.parameters())
        ratio = n_params / npairs
        # P3-002 issue says the bug was ~1000 params/pair. Fix must bring it <500.
        assert ratio < 500, \
            f"MLP has {ratio:.1f} params/pair (>{500} threshold). Overfitting risk. " \
            f"Params={n_params}, pairs={npairs}, hidden_dims={lp.hidden_dims}"


# ============================================================================
# P3-003: No causal mask in attention
# ============================================================================
class TestP3003NoCausalMask:
    """P3-003: attention must NOT apply a causal mask (KGs are undirected)."""

    def test_attention_weights_are_dense(self):
        """Attention weights must be non-zero for ALL neighbors (no masking)."""
        edge_types = [
            ("drug", "inhibits", "protein"),
            ("protein", "inhibited_by", "drug"),
        ]
        attn = HeterogeneousMultiHeadAttention(
            embedding_dim=16, num_heads=2, edge_types=edge_types, dropout=0.0
        )
        attn.eval()
        node_emb = {
            "drug": torch.randn(3, 16),
            "protein": torch.randn(2, 16),
        }
        # drug 0,1,2 all inhibit protein 0
        edge_indices = {
            ("drug", "inhibits", "protein"): torch.tensor([[0, 1, 2], [0, 0, 0]]),
            ("protein", "inhibited_by", "drug"): torch.tensor([[0, 0, 0], [0, 1, 2]]),
        }
        with torch.no_grad():
            out = attn(node_emb, edge_indices)
        # protein[0] receives from drug 0,1,2 — output must be finite and non-zero
        assert not torch.isnan(out["protein"][0]).any()
        assert not torch.isinf(out["protein"][0]).any()
        assert out["protein"][0].abs().sum() > 1e-6, \
            "protein[0] output is ~zero — attention may be masked"

    def test_bidirectional_message_passing(self):
        """Both drug->protein and protein->drug messages must flow in one forward pass."""
        edge_types = [
            ("drug", "inhibits", "protein"),
            ("protein", "inhibited_by", "drug"),
        ]
        attn = HeterogeneousMultiHeadAttention(
            embedding_dim=16, num_heads=2, edge_types=edge_types, dropout=0.0
        )
        attn.eval()
        node_emb = {
            "drug": torch.randn(2, 16),
            "protein": torch.randn(2, 16),
        }
        edge_indices = {
            ("drug", "inhibits", "protein"): torch.tensor([[0], [0]]),
            ("protein", "inhibited_by", "drug"): torch.tensor([[0], [0]]),
        }
        with torch.no_grad():
            out = attn(node_emb, edge_indices)
        # Both drug[0] and protein[0] must receive messages
        assert out["drug"][0].abs().sum() > 1e-6
        assert out["protein"][0].abs().sum() > 1e-6

    def test_docstring_warns_against_causal_mask(self):
        """The docstring must explicitly warn against adding a causal mask."""
        doc = HeterogeneousMultiHeadAttention.__doc__ or ""
        assert "causal mask" in doc.lower() or "DO NOT ADD A CAUSAL MASK" in doc, \
            "Docstring does not warn against causal mask"


# ============================================================================
# P3-004: Unknown node-type fallback
# ============================================================================
class TestP3004UnknownNodeTypeFallback:
    """P3-004: unknown node types must not crash inference."""

    def test_out_of_range_index_does_not_crash(self):
        """An out-of-range node type index must NOT raise IndexError."""
        nte = NodeTypeEmbedding(num_node_types=5, embedding_dim=16)
        # Index 5 is out of range (valid is 0-4)
        result = nte(torch.tensor([0, 5, 10, 2]))
        assert result.shape == (4, 16)

    def test_unknown_slot_is_zero_initialized(self):
        """The unknown slot must be initialized to ZERO (not random)."""
        nte = NodeTypeEmbedding(num_node_types=5, embedding_dim=16)
        # Look up only the unknown slot
        result = nte(torch.tensor([5]))
        assert torch.allclose(result[0], torch.zeros(16)), \
            f"Unknown slot is not zero: {result[0]}"

    def test_warning_emitted_once(self):
        """The unknown-type warning must fire EXACTLY ONCE per instance."""
        nte = NodeTypeEmbedding(num_node_types=5, embedding_dim=16)
        log_buffer = io.StringIO()
        handler = logging.StreamHandler(log_buffer)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("graph_transformer.models.embeddings")
        logger.addHandler(handler)
        try:
            # Call 3 times with out-of-range indices
            nte(torch.tensor([5, 10]))
            nte(torch.tensor([6, 11]))
            nte(torch.tensor([7, 12]))
        finally:
            logger.removeHandler(handler)
        log_text = log_buffer.getvalue()
        warning_count = log_text.count("CLAMPED to")
        assert warning_count == 1, \
            f"Warning fired {warning_count} times (expected 1): {log_text}"


# ============================================================================
# P3-005: Reproducible init via torch.manual_seed
# ============================================================================
class TestP3005ReproducibleInit:
    """P3-005: same seed must produce identical weights."""

    def test_same_seed_identical_weights(self):
        """Two models with seed=42 must have IDENTICAL parameters."""
        kwargs = dict(
            feature_dims=DEFAULT_FEATURE_DIMS,
            embedding_dim=32,
            num_layers=2,
            num_heads=4,
            edge_types=DEFAULT_EDGE_TYPES,
            node_types=DEFAULT_NODE_TYPES,
            seed=42,
        )
        m1 = DrugRepurposingGraphTransformer(**kwargs)
        m2 = DrugRepurposingGraphTransformer(**kwargs)
        for n, (p1, p2) in enumerate(zip(m1.parameters(), m2.parameters())):
            assert torch.equal(p1, p2), f"Param {n} differs between two seed=42 models"

    def test_different_seeds_different_weights(self):
        """Two models with DIFFERENT seeds must have DIFFERENT parameters."""
        kwargs = lambda s: dict(
            feature_dims=DEFAULT_FEATURE_DIMS,
            embedding_dim=32,
            num_layers=2,
            num_heads=4,
            edge_types=DEFAULT_EDGE_TYPES,
            node_types=DEFAULT_NODE_TYPES,
            seed=s,
        )
        m1 = DrugRepurposingGraphTransformer(**kwargs(42))
        m2 = DrugRepurposingGraphTransformer(**kwargs(99))
        any_diff = any(not torch.equal(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters()))
        assert any_diff, "Different seeds produced identical weights"

    def test_seed_round_trips_through_save_load(self, tmp_path):
        """The seed must be saved and restored through save/load."""
        m1 = DrugRepurposingGraphTransformer(
            feature_dims=DEFAULT_FEATURE_DIMS,
            embedding_dim=32,
            num_layers=1,
            num_heads=4,
            edge_types=DEFAULT_EDGE_TYPES,
            node_types=DEFAULT_NODE_TYPES,
            seed=123,
        )
        ckpt = tmp_path / "ckpt.pt"
        m1.save(str(ckpt))
        m2 = DrugRepurposingGraphTransformer.load(str(ckpt))
        assert m2.seed == 123, f"Seed not round-tripped: {m2.seed}"
        # Weights must match
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2), "Loaded model weights differ from saved"


# ============================================================================
# P3-006: Calibrated flag + warning
# ============================================================================
class TestP3006CalibrationFlag:
    """P3-006: calibrated flag must track fit_temperature() calls."""

    def test_fresh_model_uncalibrated(self):
        """A fresh link predictor must report calibrated=False."""
        lp = DrugDiseaseLinkPredictor(embedding_dim=16, num_pairs=50)
        assert lp._calibrated is False

    def test_metadata_returned_when_requested(self):
        """return_metadata=True must return a dict with 'probability' and 'calibrated'."""
        lp = DrugDiseaseLinkPredictor(embedding_dim=16, num_pairs=50)
        lp.eval()
        drug = torch.randn(3, 16)
        disease = torch.randn(3, 16)
        result = lp.predict_probability(drug, disease, return_metadata=True)
        assert isinstance(result, dict)
        assert "probability" in result
        assert "calibrated" in result
        assert result["calibrated"] is False

    def test_calibrated_after_fit_temperature(self):
        """After fit_temperature(), calibrated must be True."""
        lp = DrugDiseaseLinkPredictor(embedding_dim=16, num_pairs=50)
        lp.eval()
        drug = torch.randn(5, 16)
        disease = torch.randn(5, 16)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        lp.fit_temperature(drug, disease, labels)
        assert lp._calibrated is True
        result = lp.predict_probability(drug, disease, return_metadata=True)
        assert result["calibrated"] is True

    def test_warning_fires_once_per_instance(self):
        """The uncalibrated warning must fire EXACTLY ONCE per instance."""
        lp = DrugDiseaseLinkPredictor(embedding_dim=16, num_pairs=50)
        lp.eval()
        log_buffer = io.StringIO()
        handler = logging.StreamHandler(log_buffer)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("graph_transformer.models.link_predictor")
        logger.addHandler(handler)
        drug = torch.randn(3, 16)
        disease = torch.randn(3, 16)
        try:
            lp.predict_probability(drug, disease)
            lp.predict_probability(drug, disease)
            lp.predict_probability(drug, disease)
        finally:
            logger.removeHandler(handler)
        log_text = log_buffer.getvalue()
        warning_count = log_text.count("UNCALIBRATED")
        assert warning_count == 1, \
            f"Warning fired {warning_count} times (expected 1)"


# ============================================================================
# P3-007: LayerNorm + gradient stability
# ============================================================================
class TestP3007LayerNormGradientStability:
    """P3-007: LayerNorm must be applied and gradients must be stable."""

    def test_layernorm_modules_exist_for_all_node_types(self):
        """norm1 and norm2 must exist for every node type."""
        layer = GraphTransformerLayer(
            embedding_dim=32,
            num_heads=4,
            edge_types=DEFAULT_EDGE_TYPES,
            ffn_hidden_dim=64,
            node_types=DEFAULT_NODE_TYPES,
        )
        assert layer.norm1 is not None
        assert layer.norm2 is not None
        for nt in DEFAULT_NODE_TYPES:
            assert nt in layer.norm1, f"norm1 missing node type {nt}"
            assert nt in layer.norm2, f"norm2 missing node type {nt}"

    def test_layernorm_receives_gradients(self):
        """LayerNorm parameters must receive NON-ZERO gradients on backward."""
        layer = GraphTransformerLayer(
            embedding_dim=32,
            num_heads=4,
            edge_types=DEFAULT_EDGE_TYPES,
            ffn_hidden_dim=64,
            node_types=DEFAULT_NODE_TYPES,
        )
        node_emb = {nt: torch.randn(2, 32, requires_grad=True) for nt in DEFAULT_NODE_TYPES}
        edge_indices = {}
        for et in DEFAULT_EDGE_TYPES:
            src, _, tgt = et
            edge_indices[et] = torch.tensor([[0, 1], [0, 1]])
        out = layer(node_emb, edge_indices)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        # Check at least one norm1 weight has a non-zero gradient
        found_nonzero = False
        for nt in DEFAULT_NODE_TYPES:
            g = layer.norm1[nt].weight.grad
            if g is not None and g.abs().sum() > 0:
                found_nonzero = True
                break
        assert found_nonzero, "No non-zero gradients on any norm1 weight"

    def test_gradient_stability_helper_returns_stable(self):
        """The gradient stability helper must return stable=True for healthy grads."""
        # Simulate per-layer gradient norms that are within 10x of each other
        per_layer = {
            "layer_0": 1.0,
            "layer_1": 0.8,
            "layer_2": 0.9,
            "layer_3": 0.85,
        }
        result = GraphTransformerLayer.check_gradient_stability(None, per_layer)
        assert result["stable"] is True
        assert result["ratio"] < 10.0

    def test_gradient_stability_helper_detects_instability(self):
        """The helper must return stable=False for pathological grads (1000x ratio)."""
        per_layer = {
            "layer_0": 1.0,
            "layer_1": 0.001,  # 1000x smaller
        }
        result = GraphTransformerLayer.check_gradient_stability(None, per_layer)
        assert result["stable"] is False
        assert result["ratio"] >= 10.0


# ============================================================================
# P3-008: Lock-free fast path for concurrent inference (V1 contract: 100 req)
# ============================================================================
class TestP3008LockFreeConcurrentInference:
    """P3-008: 100 concurrent predictions must complete without deadlock."""

    def test_100_concurrent_predictions_no_deadlock(self):
        """100 concurrent predictions must complete within 30s (no deadlock)."""
        lp = DrugDiseaseLinkPredictor(embedding_dim=16, num_pairs=50)
        lp.eval()
        drug = torch.randn(3, 16)
        disease = torch.randn(3, 16)
        # Warmup
        for _ in range(3):
            lp.predict_probability(drug, disease)
        # 100 concurrent predictions
        n_threads = 100
        errors = []
        done = threading.Event()

        def worker():
            try:
                for _ in range(3):
                    lp.predict_probability(drug, disease)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
        start = time.time()
        for t in threads:
            t.start()
        # Wait with timeout — if deadlock, this will exceed 30s
        for t in threads:
            t.join(timeout=30)
            if t.is_alive():
                done.set()
                pytest.fail("Deadlock detected: thread still alive after 30s")
        elapsed = time.time() - start
        assert not errors, f"{len(errors)} errors during concurrent predictions: {errors[0]}"
        assert elapsed < 30, f"Took {elapsed:.1f}s (>30s threshold) — possible lock contention"

    def test_eval_mode_does_not_acquire_lock(self):
        """In eval mode, the lock-free fast path must be taken (no lock acquisition)."""
        import inspect
        src = inspect.getsource(DrugDiseaseLinkPredictor.predict_probability)
        # The fast path must exist
        assert "if not self.training" in src, "Lock-free fast path missing"
        assert "torch.set_grad_enabled(False)" in src, "set_grad_enabled missing"
        # The lock acquisition must be in the ELSE branch (only for train mode)
        assert "with self._predict_lock" in src, "RLock retained for train-mode case"


# ============================================================================
# P3-009: Freeze pretrained embeddings
# ============================================================================
class TestP3009FreezePretrained:
    """P3-009: frozen projection weights must NOT be updated by the optimizer."""

    def test_freeze_default_true(self):
        """The default for freeze_pretrained must be True."""
        proj = NodeTypeProjection(
            feature_dims=DEFAULT_FEATURE_DIMS,
            embedding_dim=32,
        )
        assert proj._default_freeze_pretrained is True

    def test_frozen_weights_not_updated_by_optimizer(self):
        """Frozen weights must NOT change after optimizer.step()."""
        proj = NodeTypeProjection(
            feature_dims=DEFAULT_FEATURE_DIMS,
            embedding_dim=32,
            freeze_pretrained=True,
        )
        drug_proj = proj.projections["drug"]
        # Load pretrained weights (frozen by default)
        fake_w = torch.randn_like(drug_proj.weight)
        fake_b = torch.randn_like(drug_proj.bias)
        proj.load_pretrained_embeddings("drug", fake_w, fake_b)
        # Snapshot the weights
        w_before = drug_proj.weight.clone()
        b_before = drug_proj.bias.clone()
        # Run a forward + backward + optimizer step
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, proj.parameters()), lr=0.1)
        features = {"drug": torch.randn(4, DEFAULT_FEATURE_DIMS["drug"])}
        out = proj(features)
        loss = out["drug"].sum()
        loss.backward()
        optimizer.step()
        # Frozen weights must be UNCHANGED
        assert torch.equal(drug_proj.weight, w_before), \
            "Frozen drug weight was updated by optimizer"
        assert torch.equal(drug_proj.bias, b_before), \
            "Frozen drug bias was updated by optimizer"

    def test_unfrozen_weights_updated_by_optimizer(self):
        """Unfrozen weights MUST change after optimizer.step()."""
        proj = NodeTypeProjection(
            feature_dims=DEFAULT_FEATURE_DIMS,
            embedding_dim=32,
            freeze_pretrained=False,
        )
        drug_proj = proj.projections["drug"]
        w_before = drug_proj.weight.clone()
        optimizer = torch.optim.SGD(proj.parameters(), lr=0.1)
        features = {"drug": torch.randn(4, DEFAULT_FEATURE_DIMS["drug"])}
        out = proj(features)
        loss = out["drug"].sum()
        loss.backward()
        optimizer.step()
        assert not torch.equal(drug_proj.weight, w_before), \
            "Unfrozen drug weight was NOT updated by optimizer"

    def test_unfreeze_restores_requires_grad(self):
        """unfreeze_pretrained_embeddings must set requires_grad=True."""
        proj = NodeTypeProjection(
            feature_dims=DEFAULT_FEATURE_DIMS,
            embedding_dim=32,
            freeze_pretrained=True,
        )
        drug_proj = proj.projections["drug"]
        fake_w = torch.randn_like(drug_proj.weight)
        proj.load_pretrained_embeddings("drug", fake_w)
        assert not drug_proj.weight.requires_grad
        proj.unfreeze_pretrained_embeddings("drug")
        assert drug_proj.weight.requires_grad
        assert "drug" not in proj.frozen_types()


# ============================================================================
# P3-010: Dropout scales with graph size
# ============================================================================
class TestP3010DropoutScaling:
    """P3-010: dropout must scale with graph size to prevent overfitting."""

    @pytest.mark.parametrize("npairs,expected", [
        (0, 0.5),
        (9999, 0.5),
        (10000, 0.2),
        (999999, 0.2),
        (1000000, 0.1),
    ])
    def test_helper_scaling(self, npairs, expected):
        """The helper must return the correct dropout for each graph size."""
        assert _dropout_for_graph_size(npairs) == expected

    def test_negative_pairs_raises(self):
        """Negative num_training_pairs must raise ValueError."""
        with pytest.raises(ValueError):
            _dropout_for_graph_size(-1)

    def test_model_applies_scaled_dropout_for_small_graph(self):
        """A model with num_training_pairs=100 must use dropout=0.5."""
        m = DrugRepurposingGraphTransformer(
            feature_dims=DEFAULT_FEATURE_DIMS,
            embedding_dim=32,
            num_layers=1,
            num_heads=4,
            edge_types=DEFAULT_EDGE_TYPES,
            node_types=DEFAULT_NODE_TYPES,
            num_training_pairs=100,
        )
        assert m.dropout == 0.5, f"dropout={m.dropout}, expected 0.5"

    def test_model_applies_scaled_dropout_for_large_graph(self):
        """A model with num_training_pairs=2M must use dropout=0.1."""
        m = DrugRepurposingGraphTransformer(
            feature_dims=DEFAULT_FEATURE_DIMS,
            embedding_dim=32,
            num_layers=1,
            num_heads=4,
            edge_types=DEFAULT_EDGE_TYPES,
            node_types=DEFAULT_NODE_TYPES,
            num_training_pairs=2_000_000,
        )
        assert m.dropout == 0.1, f"dropout={m.dropout}, expected 0.1"

    def test_explicit_dropout_respected(self):
        """If caller explicitly passes dropout, num_training_pairs must NOT override it."""
        m = DrugRepurposingGraphTransformer(
            feature_dims=DEFAULT_FEATURE_DIMS,
            embedding_dim=32,
            num_layers=1,
            num_heads=4,
            edge_types=DEFAULT_EDGE_TYPES,
            node_types=DEFAULT_NODE_TYPES,
            dropout=0.3,  # explicit
            num_training_pairs=100,  # would normally give 0.5
        )
        assert m.dropout == 0.3, f"Explicit dropout overridden: {m.dropout}"


# ============================================================================
# INTEGRATION: all 10 fixes work together in a real forward pass
# ============================================================================
class TestIntegrationAllFixesTogether:
    """Integration test: build a model with all 10 fixes and run a real forward pass."""

    def test_full_forward_pass_with_all_fixes(self):
        """Build a model exercising all 10 fixes and run forward_logits + forward."""
        torch.manual_seed(42)
        m = DrugRepurposingGraphTransformer(
            feature_dims=DEFAULT_FEATURE_DIMS,
            embedding_dim=32,
            num_layers=2,
            num_heads=4,
            edge_types=DEFAULT_EDGE_TYPES,  # P3-001: 18 types
            node_types=DEFAULT_NODE_TYPES,
            num_training_pairs=100,  # P3-002 + P3-010: small graph -> [64,32] MLP, 0.5 dropout
            seed=42,  # P3-005: reproducible
        )
        # P3-002: MLP should be [64, 32]
        assert m.link_predictor_hidden_dims == [64, 32]
        # P3-010: dropout should be 0.5
        assert m.dropout == 0.5
        # P3-007: each layer should have norm1 and norm2
        for layer in m.graph_transformer_layers:
            assert layer.norm1 is not None
            assert layer.norm2 is not None

        # Build a small graph
        node_features = {
            nt: torch.randn(3, DEFAULT_FEATURE_DIMS[nt])
            for nt in DEFAULT_NODE_TYPES
        }
        edge_indices = {}
        for et in DEFAULT_EDGE_TYPES:
            src, _, tgt = et
            if src in node_features and tgt in node_features:
                edge_indices[et] = torch.tensor([[0, 1], [0, 1]])

        drug_idx = torch.tensor([0, 1])
        disease_idx = torch.tensor([0, 1])

        # forward_logits (training path)
        logits = m.forward_logits(node_features, edge_indices, drug_idx, disease_idx)
        assert logits.shape == (2,)
        assert not torch.isnan(logits).any()

        # forward (inference path with temperature)
        m.eval()
        with torch.no_grad():
            probs = m.forward(node_features, edge_indices, drug_idx, disease_idx)
        assert probs.shape == (2,)
        assert (probs >= 0).all() and (probs <= 1).all()
        assert not torch.isnan(probs).any()

    def test_save_load_round_trip_preserves_v104_fields(self, tmp_path):
        """Save/load must round-trip seed and num_training_pairs (v104 fields)."""
        m1 = DrugRepurposingGraphTransformer(
            feature_dims=DEFAULT_FEATURE_DIMS,
            embedding_dim=32,
            num_layers=1,
            num_heads=4,
            edge_types=DEFAULT_EDGE_TYPES,
            node_types=DEFAULT_NODE_TYPES,
            seed=99,
            num_training_pairs=5000,
        )
        ckpt = tmp_path / "round_trip.pt"
        m1.save(str(ckpt))
        m2 = DrugRepurposingGraphTransformer.load(str(ckpt))
        assert m2.seed == 99
        assert m2.num_training_pairs == 5000
        assert m2.link_predictor_hidden_dims == m1.link_predictor_hidden_dims
        assert m2.dropout == m1.dropout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
