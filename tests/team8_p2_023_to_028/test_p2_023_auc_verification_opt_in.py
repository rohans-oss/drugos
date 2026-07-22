"""P2-023 regression tests: AUC verification is opt-in (not per-call).

Root fix: ``EvaluationConfig.verify_sklearn_agreement`` defaults to
``False`` (was ``True``). The slow O(n_pos * n_neg) Mann-Whitney U
cross-check is no longer run on every ``compute_auc`` call. Operators
who want the cross-check call ``verify_auc_against_manual`` ONCE at
end of training.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

# Ensure phase2 is importable
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_PHASE2_ROOT = os.path.join(_REPO_ROOT, "phase2")
if _PHASE2_ROOT not in sys.path:
    sys.path.insert(0, _PHASE2_ROOT)


def test_p2_023_default_verify_sklearn_agreement_is_false():
    """P2-023: ``verify_sklearn_agreement`` MUST default to False.

    The previous default ``True`` caused ``compute_auc`` to call
    ``_manual_auc`` (O(n_pos * n_neg)) on every AUC computation,
    adding ~30 minutes per epoch on 100K x 100K eval sets.
    """
    # Ensure the env var is NOT set so we get the production default
    # (not a test-environment override).
    saved = os.environ.pop("DRUGOS_VERIFY_SKLEARN_AUC", None)
    try:
        from drugos_graph.config import EVALUATION_CONFIG
        assert EVALUATION_CONFIG.verify_sklearn_agreement is False, (
            "P2-023 REGRESSION: verify_sklearn_agreement must default to "
            "False. The previous True default caused compute_auc to call "
            "_manual_auc (O(n_pos * n_neg)) on every AUC computation, "
            "adding ~30 minutes per epoch on 100K x 100K eval sets. "
            "Operators who want the cross-check call "
            "verify_auc_against_manual ONCE at end of training."
        )
    finally:
        if saved is not None:
            os.environ["DRUGOS_VERIFY_SKLEARN_AUC"] = saved


def test_p2_023_env_var_reenables_per_call_verification():
    """P2-023: setting ``DRUGOS_VERIFY_SKLEARN_AUC=1`` re-enables the
    per-call cross-check (backwards compatibility for debugging)."""
    os.environ["DRUGOS_VERIFY_SKLEARN_AUC"] = "1"
    try:
        # Force a fresh import to pick up the env var
        import importlib
        import drugos_graph.config as _cfg
        importlib.reload(_cfg)
        assert _cfg.EVALUATION_CONFIG.verify_sklearn_agreement is True, (
            "P2-023: DRUGOS_VERIFY_SKLEARN_AUC=1 must re-enable per-call "
            "verification for backwards-compatible debugging."
        )
    finally:
        del os.environ["DRUGOS_VERIFY_SKLEARN_AUC"]
        # Reload to restore the default state for other tests
        import importlib
        import drugos_graph.config as _cfg
        importlib.reload(_cfg)


def test_p2_023_verify_auc_against_manual_helper_exists():
    """P2-023: the new ``verify_auc_against_manual`` helper MUST be
    exposed in ``evaluation.__all__`` and callable."""
    from drugos_graph import evaluation
    assert "verify_auc_against_manual" in evaluation.__all__, (
        "P2-023 REGRESSION: verify_auc_against_manual must be in "
        "evaluation.__all__ so operators can import it for end-of-"
        "training verification."
    )
    assert callable(evaluation.verify_auc_against_manual), (
        "P2-023 REGRESSION: verify_auc_against_manual must be callable."
    )


def test_p2_023_helper_returns_dict_with_passes_flag():
    """P2-023: ``verify_auc_against_manual`` returns a dict with
    ``sklearn_auc``, ``manual_auc``, ``abs_delta``, ``passes``."""
    from drugos_graph.evaluation import verify_auc_against_manual
    # 100 positives with higher scores than 100 negatives -- AUC = 1.0
    pos = np.linspace(0.5, 1.0, 100)
    neg = np.linspace(0.0, 0.49, 100)
    result = verify_auc_against_manual(pos, neg, higher_is_better=True)
    assert isinstance(result, dict)
    assert set(result.keys()) >= {
        "sklearn_auc", "manual_auc", "abs_delta", "atol", "passes",
        "n_pos", "n_neg",
    }, f"P2-023: unexpected result keys: {set(result.keys())}"
    assert result["passes"] is True, (
        f"P2-023: sklearn and manual AUC must agree on a separable "
        f"dataset. sklearn={result['sklearn_auc']}, "
        f"manual={result['manual_auc']}, delta={result['abs_delta']}"
    )
    assert result["n_pos"] == 100
    assert result["n_neg"] == 100
    assert result["abs_delta"] < 1e-8


def test_p2_023_helper_raises_on_no_direction_source():
    """P2-023: the helper MUST raise when no direction source is
    provided (mirrors compute_auc P2-007 behaviour)."""
    from drugos_graph.evaluation import (
        verify_auc_against_manual,
        EvaluationInputError,
    )
    pos = np.array([0.1, 0.2, 0.3])
    neg = np.array([0.4, 0.5, 0.6])
    with pytest.raises(EvaluationInputError) as exc_info:
        verify_auc_against_manual(pos, neg)
    msg = str(exc_info.value)
    assert "P2-023" in msg or "no_direction_source" in msg, (
        f"P2-023: helper must raise with a clear P2-023 message. "
        f"Got: {msg}"
    )


def test_p2_023_compute_auc_is_fast_on_large_eval_set():
    """P2-023 BENCHMARK: ``compute_auc`` on 50K x 50K MUST complete in
    <5 seconds. Before the fix, the per-call Mann-Whitney cross-check
    made this take ~30 minutes."""
    from drugos_graph.evaluation import compute_auc
    # 50K positives + 50K negatives = 100K total scores
    rng = np.random.default_rng(42)
    pos = rng.uniform(0.5, 1.0, 50_000)
    neg = rng.uniform(0.0, 0.5, 50_000)
    # Ensure env var is not set so we use the fast default path
    saved = os.environ.pop("DRUGOS_VERIFY_SKLEARN_AUC", None)
    try:
        t0 = time.time()
        auc = compute_auc(pos, neg, higher_is_better=True)
        elapsed = time.time() - t0
    finally:
        if saved is not None:
            os.environ["DRUGOS_VERIFY_SKLEARN_AUC"] = saved
    # AUC should be ~1.0 (positives are all > 0.5, negatives all < 0.5)
    assert 0.99 < auc <= 1.0, f"P2-023: unexpected AUC: {auc}"
    # The fast path MUST complete in <5 seconds on 50K x 50K
    # (sklearn's O(n log n) on 100K elements takes <1s on modern hardware;
    # 5s leaves generous headroom for CI runners)
    assert elapsed < 5.0, (
        f"P2-023 REGRESSION: compute_auc took {elapsed:.2f}s on 50K x 50K "
        f"(must be <5s). The per-call Mann-Whitney cross-check is likely "
        f"still enabled -- verify that "
        f"EvaluationConfig.verify_sklearn_agreement defaults to False."
    )
