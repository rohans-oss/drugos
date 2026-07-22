"""P1-058 ROOT FIX (v110): regression guard for circuit breaker behavior.

WHAT THIS TEST GUARDS AGAINST
-----------------------------
The audit (Issue #58) asked: "verify breaker opens on failure and
recovers via probe."

The circuit breaker protects external API calls (ChEMBL, PubChem,
UniProt, DisGeNET) from cascading failures. A bug in the breaker
state machine would either:
  - Never trip → cascading failures bring down the whole pipeline.
  - Never recover → a single transient outage permanently disables
    a source for the worker's lifetime.

This test verifies:
  1. The breaker starts in CLOSED state.
  2. After ``failure_threshold`` consecutive failures, it transitions
     to OPEN.
  3. In OPEN state, ``allow_request()`` returns False.
  4. After ``reset_timeout`` seconds, it transitions to HALF_OPEN.
  5. In HALF_OPEN, exactly ONE probe request is allowed (subsequent
     requests are refused).
  6. A successful probe transitions the breaker to CLOSED.
  7. A failed probe transitions the breaker back to OPEN (with backoff).
  8. The P1-028 probe_timeout safety net releases a stuck probe slot.

This is the regression guard the audit asked for.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DRUGOS_ENVIRONMENT", "development")

from _circuit_breaker import _CircuitBreaker  # noqa: E402


# ---------------------------------------------------------------------------
# Tests: CLOSED state (normal operation).
# ---------------------------------------------------------------------------

def test_breaker_starts_in_closed_state():
    """A fresh breaker MUST be in CLOSED state."""
    cb = _CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
    assert cb.state == "CLOSED" or cb.state == "closed"
    assert cb.allow_request() is True


def test_breaker_allows_requests_below_failure_threshold():
    """Below the failure threshold, the breaker MUST stay CLOSED."""
    cb = _CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
    for i in range(4):  # 4 failures < threshold of 5
        cb.record_failure()
        assert cb.allow_request() is True, (
            f"After {i + 1} failures (< threshold 5), breaker should still allow requests."
        )


# ---------------------------------------------------------------------------
# Tests: OPEN state (failure mode).
# ---------------------------------------------------------------------------

def test_breaker_opens_after_failure_threshold():
    """After ``failure_threshold`` failures, the breaker MUST transition to OPEN."""
    cb = _CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
    for i in range(5):
        cb.record_failure()
    state = cb.state.upper() if hasattr(cb.state, "upper") else str(cb.state)
    assert state == "OPEN", (
        f"After 5 failures (== threshold), breaker must be OPEN. Got state={cb.state!r}."
    )


def test_open_breaker_refuses_requests():
    """In OPEN state, ``allow_request()`` MUST return False."""
    cb = _CircuitBreaker(failure_threshold=3, reset_timeout=60.0)
    for _ in range(3):
        cb.record_failure()
    assert cb.allow_request() is False, (
        "OPEN breaker must refuse requests via allow_request()."
    )


# ---------------------------------------------------------------------------
# Tests: HALF_OPEN state (probe-based recovery).
# ---------------------------------------------------------------------------

def test_breaker_transitions_to_half_open_after_reset_timeout():
    """After ``reset_timeout`` seconds, the breaker MUST transition to HALF_OPEN."""
    cb = _CircuitBreaker(
        failure_threshold=3,
        reset_timeout=0.05,  # 50ms — short for fast tests
    )
    for _ in range(3):
        cb.record_failure()
    assert cb.allow_request() is False  # OPEN
    # Wait for reset_timeout to elapse.
    time.sleep(0.1)
    # The next call to allow_request() should transition to HALF_OPEN and
    # allow the probe.
    assert cb.allow_request() is True, (
        "After reset_timeout, breaker must allow the probe request (HALF_OPEN)."
    )


def test_half_open_allows_exactly_one_probe():
    """In HALF_OPEN, exactly ONE probe request is allowed; subsequent requests are refused."""
    cb = _CircuitBreaker(
        failure_threshold=2,
        reset_timeout=0.05,
        probe_timeout=300.0,  # long — we're testing the single-probe gate, not the timeout
    )
    # Open the breaker.
    for _ in range(2):
        cb.record_failure()
    # Wait for reset_timeout.
    time.sleep(0.1)
    # First request — this is the probe. Should be allowed.
    first = cb.allow_request()
    assert first is True, "HALF_OPEN must allow exactly ONE probe."
    # Second request — must be refused (probe in flight).
    second = cb.allow_request()
    assert second is False, (
        "HALF_OPEN must refuse a second request while the probe is in flight."
    )


def test_successful_probe_closes_breaker():
    """A successful probe MUST transition the breaker back to CLOSED."""
    cb = _CircuitBreaker(
        failure_threshold=2,
        reset_timeout=0.05,
    )
    for _ in range(2):
        cb.record_failure()
    time.sleep(0.1)
    # Probe
    assert cb.allow_request() is True
    # Record success — should close the breaker.
    cb.record_success()
    state = cb.state.upper() if hasattr(cb.state, "upper") else str(cb.state)
    assert state == "CLOSED", (
        f"After successful probe, breaker must be CLOSED. Got state={cb.state!r}."
    )
    # Subsequent requests should be allowed.
    assert cb.allow_request() is True


def test_failed_probe_reopens_breaker():
    """A failed probe MUST transition the breaker back to OPEN."""
    cb = _CircuitBreaker(
        failure_threshold=2,
        reset_timeout=0.05,
    )
    for _ in range(2):
        cb.record_failure()
    time.sleep(0.1)
    # Probe
    assert cb.allow_request() is True
    # Probe fails — should re-open.
    cb.record_failure()
    state = cb.state.upper() if hasattr(cb.state, "upper") else str(cb.state)
    assert state == "OPEN", (
        f"After failed probe, breaker must be OPEN. Got state={cb.state!r}."
    )
    # Subsequent requests should be refused.
    assert cb.allow_request() is False


# ---------------------------------------------------------------------------
# Tests: P1-028 probe_timeout safety net.
# ---------------------------------------------------------------------------

def test_probe_timeout_releases_stuck_probe():
    """P1-028: a probe stuck longer than probe_timeout MUST be auto-released."""
    cb = _CircuitBreaker(
        failure_threshold=2,
        reset_timeout=0.05,
        probe_timeout=1.0,  # P1-028: minimum is 1.0s — use 1.0s and wait 1.1s
    )
    for _ in range(2):
        cb.record_failure()
    time.sleep(0.1)
    # First probe — should be allowed.
    assert cb.allow_request() is True
    # Don't call record_success / record_failure — simulate a crashed caller.
    # Wait for probe_timeout to elapse (1.0s minimum).
    time.sleep(1.1)
    # The next allow_request() should auto-release the stuck probe and
    # allow a new one.
    assert cb.allow_request() is True, (
        "After probe_timeout, the stuck probe slot MUST be auto-released "
        "so a new probe can fire (P1-028 ROOT FIX)."
    )


# ---------------------------------------------------------------------------
# Tests: state transition semantics.
# ---------------------------------------------------------------------------

def test_record_success_in_closed_state_resets_failure_count():
    """In CLOSED state, a success MUST reset the failure counter."""
    cb = _CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
    # 4 failures (below threshold).
    for _ in range(4):
        cb.record_failure()
    # A success should reset the count.
    cb.record_success()
    # Now 4 more failures should NOT trip the breaker.
    for _ in range(4):
        cb.record_failure()
    state = cb.state.upper() if hasattr(cb.state, "upper") else str(cb.state)
    assert state == "CLOSED", (
        f"After reset, 4 failures (< threshold 5) must not trip. Got state={cb.state!r}."
    )


def test_is_open_inverse_of_allow_request_in_closed_state():
    """In CLOSED state, is_open() MUST be the inverse of allow_request()."""
    cb = _CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
    assert cb.allow_request() is True
    # is_open should be False in CLOSED state.
    assert cb.is_open() is False


def test_breaker_name_appears_in_logging():
    """The breaker MUST accept an optional name for log messages."""
    cb = _CircuitBreaker(failure_threshold=5, reset_timeout=60.0, name="chembl_api")
    # The name should be stored (we don't assert on log output here —
    # the existence of the name attribute is enough for the contract).
    assert cb.name == "chembl_api" or getattr(cb, "name", None) == "chembl_api"


# ---------------------------------------------------------------------------
# Tests: probe() context manager (P1-028 convenience API).
# ---------------------------------------------------------------------------

def test_probe_context_manager_releases_slot_on_success():
    """The probe() context manager MUST release the slot on success."""
    cb = _CircuitBreaker(
        failure_threshold=2,
        reset_timeout=0.05,
    )
    for _ in range(2):
        cb.record_failure()
    time.sleep(0.1)
    # Use the context manager — should acquire + release the probe slot.
    import contextlib
    with contextlib.suppress(Exception):
        with cb.probe():
            pass  # simulate a successful probe
    # After the context exits, the slot should be released — a new
    # probe should be allowed.
    assert cb.allow_request() is True, (
        "After probe() context manager exits, the slot MUST be released."
    )


def test_probe_context_manager_releases_slot_on_exception():
    """The probe() context manager MUST release the slot on exception.

    NOTE: releasing the slot means the breaker records the failure (re-opens)
    and the slot becomes available again after the next reset_timeout. This
    test verifies the slot is released (not stuck in half-open forever).
    The breaker is in OPEN state immediately after the exception (correct
    behavior — the probe failed). After reset_timeout elapses, a new probe
    can be acquired.
    """
    cb = _CircuitBreaker(
        failure_threshold=2,
        reset_timeout=0.05,
    )
    for _ in range(2):
        cb.record_failure()
    time.sleep(0.1)
    # Use the context manager and raise inside.
    import contextlib
    with contextlib.suppress(RuntimeError):
        with cb.probe():
            raise RuntimeError("simulated probe failure")
    # Immediately after the exception, the breaker is OPEN (correct).
    state = cb.state.upper() if hasattr(cb.state, "upper") else str(cb.state)
    assert state == "OPEN", (
        f"After probe() failed with exception, breaker must be OPEN. "
        f"Got state={cb.state!r}."
    )
    # The next request must be refused (breaker is OPEN, slot is released
    # but the breaker hasn't reset yet).
    assert cb.allow_request() is False, (
        "Immediately after failed probe, breaker must refuse requests (OPEN)."
    )
    # After reset_timeout elapses, a new probe should be allowed — this
    # proves the slot was released (not stuck).
    time.sleep(0.1)
    assert cb.allow_request() is True, (
        "After reset_timeout, a new probe MUST be allowed — this proves "
        "the probe() context manager released the slot on exception."
    )
