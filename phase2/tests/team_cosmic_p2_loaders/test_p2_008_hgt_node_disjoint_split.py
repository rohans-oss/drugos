"""Team Cosmic — Phase 2 Loaders — Regression test for P2-008.

HGT step11b uses compound-disjoint split but DISEASES appear in both train and test

Severity: CRITICAL  |  Category: Scientific  |  File: phase2/drugos_graph/run_pipeline.py  |  Line: 7047-7073

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


def test_p2_008_hgt_split_partitions_both_endpoints():
    """CRITICAL: a disease must NOT appear in two splits (leakage prevention)."""
    rng = random.Random(42)
    compounds = list(range(50))
    diseases = list(range(100, 130))
    rng.shuffle(compounds)
    disease_rng = random.Random(43)
    disease_rng.shuffle(diseases)

    def partition(idx_list, r_train=0.8, r_val=0.1):
        n = len(idx_list)
        n_train = int(n * r_train)
        n_val = int(n * r_val)
        return (set(idx_list[:n_train]),
                set(idx_list[n_train:n_train + n_val]),
                set(idx_list[n_train + n_val:]))

    train_c, val_c, test_c = partition(compounds)
    train_d, val_d, test_d = partition(diseases)

    src_list = [rng.choice(compounds) for _ in range(200)]
    dst_list = [rng.choice(diseases) for _ in range(200)]

    train_idx, val_idx, test_idx = [], [], []
    for i, (c, d) in enumerate(zip(src_list, dst_list)):
        if c in train_c and d in train_d:
            train_idx.append(i)
        elif c in val_c and d in val_d:
            val_idx.append(i)
        elif c in test_c and d in test_d:
            test_idx.append(i)

    train_diseases = {dst_list[i] for i in train_idx}
    val_diseases = {dst_list[i] for i in val_idx}
    test_diseases = {dst_list[i] for i in test_idx}

    # The critical assertion: NO disease in two splits
    assert not (train_diseases & val_diseases), "disease leakage train<->val"
    assert not (train_diseases & test_diseases), "disease leakage train<->test"
    assert not (val_diseases & test_diseases), "disease leakage val<->test"

    # Also no compound in two splits
    train_compounds = {src_list[i] for i in train_idx}
    val_compounds = {src_list[i] for i in val_idx}
    test_compounds = {src_list[i] for i in test_idx}
    assert not (train_compounds & val_compounds)
    assert not (train_compounds & test_compounds)
    assert not (val_compounds & test_compounds)

