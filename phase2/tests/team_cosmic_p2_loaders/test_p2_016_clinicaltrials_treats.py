"""Team Cosmic — Phase 2 Loaders — Regression test for P2-016.

rel_type='treats' only when primary_outcome_met is True — but most trials never report primary_outcome_met

Severity: HIGH  |  Category: Scientific  |  File: phase2/drugos_graph/clinicaltrials_loader.py  |  Line: 3799-3805

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


def test_p2_016_clinicaltrials_treats_criteria():
    """HIGH: Completed + unknown outcome must NOT be skipped (treats, not tested_for)."""
    from drugos_graph.clinicaltrials_loader import (
        _classify_trial_confidence,
        _TRIAL_SKIP,
    )
    # Completed + None must NOT be skipped (P2-016 fix)
    result = _classify_trial_confidence("Completed", None)
    assert result is not _TRIAL_SKIP, (
        "Completed + None outcome was SKIPPED — loses 70% of completed trials"
    )
    assert _classify_trial_confidence("Completed", True) is not _TRIAL_SKIP
    assert _classify_trial_confidence("Completed", False) is not _TRIAL_SKIP
    assert _classify_trial_confidence("Unknown status", None) is _TRIAL_SKIP

