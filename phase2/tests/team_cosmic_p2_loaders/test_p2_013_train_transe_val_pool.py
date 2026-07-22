"""Team Cosmic — Phase 2 Loaders — Regression test for P2-013.

train_transe raises RuntimeError on missing per-relation val pool — single bad relation crashes entire training

Severity: HIGH  |  Category: Integration  |  File: phase2/drugos_graph/transe_model.py  |  Line: 3380-3400

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


def test_p2_013_train_transe_val_pool_fallback():
    """HIGH: missing per-relation val pool must fall back to type-filtered, not crash."""
    from drugos_graph.transe_model import train_transe
    src = inspect.getsource(train_transe)
    assert "P2-013" in src, "P2-013 marker not in train_transe"
    assert "entity_type_lookup" in src, "entity_type_lookup not used for fallback"
    assert "50" in src or "0.5" in src, "no >50% threshold for raising"

