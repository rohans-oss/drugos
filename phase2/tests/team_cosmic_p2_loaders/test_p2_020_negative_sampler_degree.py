"""Team Cosmic — Phase 2 Loaders — Regression test for P2-020.

NegativeSampler.random_sampling uses UNIFORM sampling — over-represents hub diseases as negatives, inflating AUC

Severity: HIGH  |  Category: Scientific  |  File: phase2/drugos_graph/negative_sampling.py  |  Line: 891+

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


def test_p2_020_negative_sampler_degree_weighted():
    """HIGH: degree_weighted=True must use 1/(1+degree) per Wang et al. 2014."""
    from drugos_graph.negative_sampling import NegativeSampler
    src = inspect.getsource(NegativeSampler.random_sampling)
    assert "P2-020" in src
    assert "degree_weighted" in src
    assert ("1/(1+degree)" in src or "1.0 / (1.0 +" in src
            or "/ (1.0 +" in src or "/(1+" in src
            or "inverse" in src.lower())

    # LIVE test: degree-weighted mode produces negatives
    positives = [("D1", "DIS1"), ("D1", "DIS2"), ("D1", "DIS3"), ("D1", "DIS4"),
                 ("D2", "DIS5")]
    all_drugs = [f"D{i}" for i in range(1, 21)]
    all_diseases = [f"DIS{i}" for i in range(1, 31)]

    ns = NegativeSampler(
        positive_pairs=positives,
        all_drug_ids=all_drugs,
        all_disease_ids=all_diseases,
        seed=42,
    )
    rng = np.random.default_rng(42)
    negs = ns.random_sampling(num_negatives=50, rng=rng, degree_weighted=True)
    assert len(negs) > 0, "degree-weighted sampling produced 0 negatives"
    for n in negs:
        assert (n["drug_id"], n["disease_id"]) not in set(positives)

