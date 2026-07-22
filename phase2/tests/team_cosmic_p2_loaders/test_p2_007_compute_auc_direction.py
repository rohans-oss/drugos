"""Team Cosmic — Phase 2 Loaders — Regression test for P2-007.

compute_auc default higher_is_better=False silently INVERTS HGT/GraphTransformer AUC

Severity: CRITICAL  |  Category: Scientific  |  File: phase2/drugos_graph/evaluation.py  |  Line: 1028-1031

This test exercises the ACTUAL production code path (not comments, not smoke
tests) and verifies the behaviour contract from the issue's "Fix:" section.
If a previous "ROOT FIX" comment was a lie, this test will FAIL.
"""
from __future__ import annotations

import os
import re
import sys
import inspect
import random

import numpy as np
import pytest

# Pre-import torch_geometric to dodge the circular-import trap (see conftest.py)
import torch_geometric  # noqa: F401
import torch_geometric.typing  # noqa: F401
import torch_geometric.data  # noqa: F401
import torch_geometric.transforms  # noqa: F401


def test_p2_007_compute_auc_infers_direction_from_model():
    """CRITICAL: HGT AUC must NOT be inverted when caller passes model=."""
    from drugos_graph.evaluation import compute_auc, EvaluationInputError

    # TransE: lower distance = more plausible
    pos_low = np.array([0.05, 0.10, 0.15, 0.20])
    neg_low = np.array([0.80, 0.85, 0.90, 0.95])
    # HGT: higher score = more plausible
    pos_high = np.array([0.85, 0.90, 0.92, 0.95])
    neg_high = np.array([0.05, 0.10, 0.12, 0.15])

    # Explicit direction works for both
    assert compute_auc(pos_low, neg_low, higher_is_better=False) > 0.99
    assert compute_auc(pos_high, neg_high, higher_is_better=True) > 0.99

    # HGT model with score_direction='higher_better' must NOT invert
    class FakeHGT:
        score_direction = "higher_better"
    auc_from_model = compute_auc(pos_high, neg_high, model=FakeHGT())
    assert auc_from_model > 0.99, (
        f"HGT AUC was INVERTED via model.score_direction: got {auc_from_model} "
        f"(patient safety blocker — model would rank drugs BACKWARDS)"
    )

    # TransE model with score_direction='lower_better'
    class FakeTransE:
        score_direction = "lower_better"
    assert compute_auc(pos_low, neg_low, model=FakeTransE()) > 0.99

    # No direction → must RAISE (not silently default to False)
    old = os.environ.pop("DRUGOS_ALLOW_DEFAULT_AUC_DIRECTION", None)
    try:
        with pytest.raises(EvaluationInputError):
            compute_auc(pos_low, neg_low)
    finally:
        if old is not None:
            os.environ["DRUGOS_ALLOW_DEFAULT_AUC_DIRECTION"] = old

    # Invalid direction string raises
    with pytest.raises(EvaluationInputError):
        compute_auc(pos_low, neg_low, model_score_direction="sideways")

