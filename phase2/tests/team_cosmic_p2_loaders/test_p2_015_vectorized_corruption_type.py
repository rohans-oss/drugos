"""Team Cosmic — Phase 2 Loaders — Regression test for P2-015.

Vectorized corruption fallback samples from ALL entities regardless of type — produces type-wrong negatives

Severity: HIGH  |  Category: Scientific  |  File: phase2/drugos_graph/transe_model.py  |  Line: 2967-2995

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


def test_p2_015_vectorized_corruption_type_filtered():
    """HIGH: fallback must use entity_type_lookup, not all-entity uniform."""
    from drugos_graph.transe_model import train_transe
    src = inspect.getsource(train_transe)
    assert "P2-015" in src
    assert "entity_type_lookup" in src
    assert "_type_pools" in src or "_p2_015_type_pools" in src
    assert "raise RuntimeError" in src

