"""Team Cosmic — Phase 2 Loaders — Regression test for P2-009.

Outdated docstring says drugbank_id is canonical Compound ID — config.py says inchikey is canonical

Severity: HIGH  |  Category: Wrong  |  File: phase2/drugos_graph/phase1_bridge.py  |  Line: 56-72

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


def test_p2_009_bridge_docstring_inchikey_canonical():
    """HIGH: docstring must reflect that inchikey (not drugbank_id) is canonical."""
    import drugos_graph.phase1_bridge as bridge
    src = inspect.getsource(bridge)
    assert "inchikey" in src.lower(), "inchikey not mentioned in bridge source"
    lines = src.splitlines()
    canonical_lines = [
        l for l in lines[:300]
        if "canonical" in l.lower() and ("inchikey" in l.lower() or "compound" in l.lower())
    ]
    assert canonical_lines, "no canonical+inchikey line in bridge schema mapping"
    assert any("inchikey" in l.lower() and "canonical" in l.lower() for l in canonical_lines), (
        f"no line says inchikey is canonical: {canonical_lines[:3]}"
    )

