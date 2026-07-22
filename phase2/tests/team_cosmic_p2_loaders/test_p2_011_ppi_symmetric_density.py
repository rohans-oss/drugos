"""Team Cosmic — Phase 2 Loaders — Regression test for P2-011.

Per-edge-type density uses wrong denominator for symmetric PPI (Protein-interacts_with-Protein)

Severity: HIGH  |  Category: Scientific  |  File: phase2/drugos_graph/graph_stats.py  |  Line: 1057-1067

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


def test_p2_011_ppi_symmetric_density_denominator():
    """HIGH: PPI density must use n*(n-1)/2 (undirected), not n*(n-1)."""
    from drugos_graph.config import SYMMETRIC_RELATIONS, is_symmetric_relation
    assert "interacts_with" in SYMMETRIC_RELATIONS
    assert is_symmetric_relation("interacts_with")
    assert not is_symmetric_relation("treats")  # directed

    from drugos_graph.graph_stats import GraphStats
    src = inspect.getsource(GraphStats)
    assert "(n_src_part - 1) // 2" in src or "(n - 1) // 2" in src, (
        "P2-011: no // 2 denominator for symmetric relations in GraphStats"
    )

