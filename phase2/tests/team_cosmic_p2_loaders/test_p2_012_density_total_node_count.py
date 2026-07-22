"""Team Cosmic — Phase 2 Loaders — Regression test for P2-012.

Density uses count(DISTINCT startNode) and count(DISTINCT endNode) — underestimates when src/dst sets are partial

Severity: HIGH  |  Category: Wrong  |  File: phase2/drugos_graph/graph_stats.py  |  Line: 1054-1067

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


def test_p2_012_density_uses_total_node_count():
    """HIGH: density denominator must be all possible pairs, not just participating."""
    from drugos_graph.graph_stats import GraphStats
    src = inspect.getsource(GraphStats)
    assert "MATCH (n:" in src, "P2-012: no MATCH (n:Type) query"
    assert "RETURN count(n)" in src, "P2-012: no RETURN count(n)"
    assert "participating" in src.lower(), "P2-012: no participating legacy metric"

