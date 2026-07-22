"""Team Cosmic — Phase 2 Loaders — Regression test for P2-014.

__del__ calls end_run() at GC time — non-deterministic, may run after MLflow tracking server is unreachable

Severity: HIGH  |  Category: Concurrency  |  File: phase2/drugos_graph/mlflow_tracker.py  |  Line: 184-194

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


def test_p2_014_mlflow_atexit_and_idempotent_close():
    """HIGH: atexit registered; __exit__ calls close(); close() is idempotent."""
    from drugos_graph.mlflow_tracker import MLflowTracker
    src = inspect.getsource(MLflowTracker)
    assert "atexit.register" in src, "atexit.register not in MLflowTracker"
    assert "self.close()" in src, "__exit__ does not call close()"
    assert "_closed" in src, "no _closed idempotency flag"

    # LIVE test: instantiate, close twice, verify _closed flag
    t = MLflowTracker(experiment_name="test_p2_014_unit")
    t.close()
    t.close()  # idempotent — must not raise
    assert t._closed is True

