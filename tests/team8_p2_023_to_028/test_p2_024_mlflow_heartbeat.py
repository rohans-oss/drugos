"""P2-024 regression tests: MLflow heartbeat thread handles SIGKILL.

Root fix: a daemon heartbeat thread updates ``drugos.heartbeat_ts`` on
the MLflow run every 30 seconds. After SIGKILL (which atexit cannot
catch), the heartbeat stops; ops can compare the heartbeat timestamp
to the current time to detect stale RUNNING runs.
"""
from __future__ import annotations

import os
import sys
import time

import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_PHASE2_ROOT = os.path.join(_REPO_ROOT, "phase2")
if _PHASE2_ROOT not in sys.path:
    sys.path.insert(0, _PHASE2_ROOT)


def test_p2_024_heartbeat_constants_exist():
    """P2-024: the heartbeat constants MUST be defined at module level."""
    from drugos_graph import mlflow_tracker
    assert hasattr(mlflow_tracker, "HEARTBEAT_TAG_NAME"), (
        "P2-024 REGRESSION: HEARTBEAT_TAG_NAME must be defined."
    )
    assert mlflow_tracker.HEARTBEAT_TAG_NAME == "drugos.heartbeat_ts", (
        f"P2-024: unexpected HEARTBEAT_TAG_NAME: "
        f"{mlflow_tracker.HEARTBEAT_TAG_NAME}"
    )
    assert hasattr(mlflow_tracker, "HEARTBEAT_PID_TAG_NAME"), (
        "P2-024 REGRESSION: HEARTBEAT_PID_TAG_NAME must be defined."
    )
    assert mlflow_tracker.HEARTBEAT_PID_TAG_NAME == "drugos.heartbeat_pid"
    assert hasattr(mlflow_tracker, "DEFAULT_HEARTBEAT_INTERVAL_SECONDS")
    assert hasattr(mlflow_tracker, "DEFAULT_HEARTBEAT_STALE_THRESHOLD_SECONDS")


def test_p2_024_tracker_has_heartbeat_state_attributes():
    """P2-024: ``MLflowTracker`` instances MUST expose the heartbeat
    state attributes (interval, thread, stop event, counts)."""
    from drugos_graph.mlflow_tracker import MLflowTracker
    t = MLflowTracker(heartbeat_interval=0)
    assert hasattr(t, "_heartbeat_interval"), "P2-024: missing _heartbeat_interval"
    assert hasattr(t, "_heartbeat_stale_threshold"), "P2-024: missing _heartbeat_stale_threshold"
    assert hasattr(t, "_heartbeat_thread"), "P2-024: missing _heartbeat_thread"
    assert hasattr(t, "_heartbeat_stop"), "P2-024: missing _heartbeat_stop"
    assert hasattr(t, "last_heartbeat_ts"), "P2-024: missing last_heartbeat_ts"
    assert hasattr(t, "heartbeat_count"), "P2-024: missing heartbeat_count"
    assert hasattr(t, "heartbeat_failure_count"), "P2-024: missing heartbeat_failure_count"
    t.close()


def test_p2_024_heartbeat_writes_to_local_log_without_mlflow():
    """P2-024: when MLflow is not installed, the heartbeat MUST append
    to ``_local_log`` so the heartbeat trail is still available."""
    from drugos_graph.mlflow_tracker import MLflowTracker
    # mlflow is intentionally not installed in this test env
    t = MLflowTracker(heartbeat_interval=1)
    assert t.mlflow is None, (
        "P2-024: this test requires mlflow to NOT be installed so the "
        "fallback path is exercised."
    )
    # Start a run with a 1-second heartbeat interval
    t.start_run("p2_024_test")
    try:
        # Wait long enough for at least 2 heartbeat iterations
        time.sleep(2.5)
        # The local log MUST contain at least 2 heartbeat entries
        heartbeats = [e for e in t._local_log if isinstance(e, dict) and e.get("type") == "heartbeat"]
        assert len(heartbeats) >= 2, (
            f"P2-024: expected >=2 heartbeat entries in _local_log, "
            f"got {len(heartbeats)}. Full log: {t._local_log}"
        )
        assert t.heartbeat_count >= 2, (
            f"P2-024: heartbeat_count must be >=2 after 2.5s, got "
            f"{t.heartbeat_count}"
        )
        assert t.last_heartbeat_ts is not None, (
            "P2-024: last_heartbeat_ts must be set after the first heartbeat"
        )
        # The heartbeat timestamp MUST be recent (within the last 5s)
        now = time.time()
        assert now - t.last_heartbeat_ts < 5.0, (
            f"P2-024: last heartbeat was {now - t.last_heartbeat_ts:.1f}s "
            f"ago -- the thread may not be running."
        )
    finally:
        t.close()


def test_p2_024_close_stops_heartbeat_thread():
    """P2-024: ``close()`` MUST stop the heartbeat thread (join within
    5 seconds) so it doesn't write a spurious heartbeat AFTER end_run."""
    from drugos_graph.mlflow_tracker import MLflowTracker
    t = MLflowTracker(heartbeat_interval=1)
    t.start_run("p2_024_close_test")
    # Wait for at least one heartbeat
    time.sleep(1.5)
    assert t._heartbeat_thread is not None
    assert t._heartbeat_thread.is_alive(), (
        "P2-024: heartbeat thread must be alive before close()"
    )
    t.close()
    # After close, the thread MUST have exited (or be in the process of
    # exiting -- daemon threads may take a moment to terminate)
    # The close() method joins with timeout=5s, so by the time close()
    # returns, the thread should be done.
    assert t._heartbeat_thread is None, (
        "P2-024: close() must set _heartbeat_thread to None after joining"
    )
    assert t._closed is True


def test_p2_024_heartbeat_disabled_when_interval_zero():
    """P2-024: setting ``heartbeat_interval=0`` disables the heartbeat
    (useful for unit tests). The thread MUST NOT start."""
    from drugos_graph.mlflow_tracker import MLflowTracker
    t = MLflowTracker(heartbeat_interval=0)
    t.start_run("p2_024_disabled_test")
    try:
        # Give the thread time to NOT start
        time.sleep(0.2)
        assert t._heartbeat_thread is None, (
            "P2-024: heartbeat_interval=0 must NOT start the thread"
        )
        assert t.heartbeat_count == 0
    finally:
        t.close()


def test_p2_024_heartbeat_count_increments_over_time():
    """P2-024: ``heartbeat_count`` MUST increment over time while the
    thread is running (proves the loop is actually executing)."""
    from drugos_graph.mlflow_tracker import MLflowTracker
    t = MLflowTracker(heartbeat_interval=1)
    t.start_run("p2_024_increment_test")
    try:
        time.sleep(0.5)
        first_count = t.heartbeat_count
        # The first heartbeat is immediate (no interval wait), so
        # first_count should be >= 1 by now.
        assert first_count >= 1, (
            f"P2-024: expected first heartbeat by 0.5s, got count={first_count}"
        )
        time.sleep(2.0)
        second_count = t.heartbeat_count
        # After 2 more seconds with 1s interval, we expect at least 2 more
        # heartbeats (allowing for scheduling jitter)
        assert second_count > first_count, (
            f"P2-024: heartbeat_count did not increment over 2s "
            f"(first={first_count}, second={second_count}). The loop "
            f"is not running."
        )
    finally:
        t.close()
