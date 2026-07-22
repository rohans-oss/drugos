"""Team Cosmic — Phase 2 Loaders — Regression test for P2-010.

Compound ID pattern accepts CIDm00002244 and CIDs00002244 (STITCH) but no longer accepts them after lowercasing

Severity: HIGH  |  Category: Scientific  |  File: phase2/drugos_graph/kg_builder.py  |  Line: 241

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


def test_p2_010_compound_id_pattern_case_insensitive_stitch():
    """HIGH: STITCH CIDm/CIDs IDs must pass after upstream .upper()."""
    from drugos_graph.kg_builder import ID_PATTERNS
    pat = ID_PATTERNS["Compound"]
    cases = [
        ("CIDm00002244", True),   # canonical STITCH form (lowercase m)
        ("CIDs00002244", True),   # canonical STITCH form (lowercase s)
        ("CIDM00002244", True),   # after upstream .upper()
        ("CIDS00002244", True),   # after upstream .upper()
        ("CidM00002244", True),   # mixed case
        ("DB00001", True),        # DrugBank
        ("CHEMBL123", True),      # ChEMBL
        ("RZVAJINKQORUOD-UHFFFAOYSA-N", True),  # InChIKey
        ("NAME:aspirin", False),  # rejected catch-all
        ("garbage", False),
        ("CIDM", False),          # no digits
    ]
    for s, expected in cases:
        got = bool(re.match(pat, s))
        assert got == expected, f"P2-010: {s!r} got={got} expected={expected}"

