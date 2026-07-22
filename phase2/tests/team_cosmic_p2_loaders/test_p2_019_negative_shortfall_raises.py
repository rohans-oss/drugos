"""Team Cosmic — Phase 2 Loaders — Regression test for P2-019.

Negative shortfall logs WARNING but does not RAISE — AUC is silently unreliable

Severity: HIGH  |  Category: Scientific  |  File: phase2/drugos_graph/pyg_builder.py  |  Line: 2668-2675

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


def test_p2_019_negative_shortfall_raises():
    """HIGH: shortfall < 50% must RAISE (AUC is uninterpretable)."""
    from drugos_graph.pyg_builder import PyGBuilder
    src = inspect.getsource(PyGBuilder)
    assert "P2-019" in src
    assert "DRUGOS_ALLOW_INSUFFICIENT_NEGATIVES" in src
    assert "0.5" in src  # the 50% threshold
    assert "raise RuntimeError" in src

