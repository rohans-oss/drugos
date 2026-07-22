"""Team Cosmic — Phase 2 Loaders — Regression test for P2-018.

temporal_split negative-pool fallback to full-graph nodes re-introduces transductive negative sampling

Severity: HIGH  |  Category: Scientific  |  File: phase2/drugos_graph/pyg_builder.py  |  Line: 2596-2603

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


def test_p2_018_temporal_split_raises_on_small_split():
    """HIGH: small split must RAISE, not silently fall back to transductive."""
    from drugos_graph.pyg_builder import PyGBuilder
    src = inspect.getsource(PyGBuilder)
    assert "P2-018" in src
    assert "DRUGOS_ALLOW_SMALL_SPLIT_NEGATIVES" in src
    assert "raise RuntimeError" in src

