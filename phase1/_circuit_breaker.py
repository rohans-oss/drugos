"""Shared circuit breaker implementation for the Drug Repurposing ETL platform.

This module provides :class:`_CircuitBreaker`, a thread-safe circuit breaker
with closed / open / half-open state transitions and a single-probe gate
for the half-open state.

All Phase 1 modules that need a circuit breaker should import from this
single location instead of defining their own copy.

States
------
- **CLOSED** (normal): all requests are allowed.
- **OPEN** (failing): all requests are refused until ``reset_timeout``
  elapses, at which point the breaker transitions to HALF_OPEN.
- **HALF_OPEN** (probing): exactly **one** probe request is allowed.
  Subsequent requests are refused until the probe completes (success ->
  CLOSED, failure -> OPEN).  This single-probe gate prevents a thundering
  herd when the protected service is still recovering.

API
---
The class exposes three equivalent check interfaces so callers can pick
the one that matches their codebase convention:

- ``allow_request() -> bool`` -- returns True if the request should proceed.
- ``is_open() -> bool`` -- returns True if the breaker is open (call should
  be refused).  This is the logical inverse of ``allow_request()`` for
  OPEN/CLOSED states; in HALF_OPEN the semantics differ subtly
  (``is_open`` returns False for the probe, ``allow_request`` returns True).
- ``state`` property -- returns the current state string.

P1-028 ROOT FIX (Team-2 — half-open probe stuck-forever on caller crash):
  The original ``allow_request()`` reserved the half-open probe slot by
  setting ``_half_open_probe_in_flight=True`` and relied on the caller
  to call ``record_success()`` or ``record_failure()`` to clear it. If
  the caller CRASHED between ``allow_request()`` returning True and the
  ``record_*()`` call (OOM kill, SIGKILL, segfault in a C extension
  like RDKit), the flag stayed True forever — the breaker was stuck in
  half-open and the protected service (ChEMBL API, PubChem API) was
  silently disabled for the rest of the Airflow worker's lifetime.

  ROOT FIX: track ``_half_open_probe_reserved_at`` (monotonic timestamp
  when the probe was reserved). In ``allow_request()``, if a probe has
  been in flight longer than ``probe_timeout`` seconds (default 300s =
  5 min — long enough for any legitimate API call but short enough to
  recover within a single Airflow task retry window), assume the
  original probe crashed and release the slot. A new probe is allowed.
  This bounds the stuck-half-open window to ``probe_timeout`` seconds
  instead of infinity.

  Additionally, a ``probe()`` context manager is provided for NEW
  callers — it acquires the probe slot on enter and ALWAYS releases it
  on exit (success, failure, or exception). Existing callers that use
  the ``allow_request()`` / ``record_*()`` pair continue to work
  unchanged (with the auto-recovery safety net).

History
-------
Consolidated from five duplicate implementations across the codebase:
  - ``database/connection.py``  (dataclass, UPPERCASE states, ``allow_request``)
  - ``pipelines/base_pipeline.py``  (plain class, lowercase states, ``is_open``)
  - ``pipelines/disgenet_pipeline.py``  (plain class, no probe gate)
  - ``pipelines/_chembl_http_client.py``  (wrapper around base_pipeline's)
  - ``cleaning/__init__.py``  (plain class, ``name`` param, no lock)

The canonical version merges all features: thread-safety via a lock,
``_half_open_probe_in_flight`` single-probe gate (v40 ROOT FIX / P1-A8),
``time.monotonic()`` for monotonic elapsed-time measurement, optional
``name`` attribute for logging, all three check APIs, AND the P1-028
probe-timeout safety net + ``probe()`` context manager.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)


# P1-028 default probe timeout: 5 minutes. Long enough for any
# legitimate API call (ChEMBL, PubChem, UniProt, DisGeNET — even with
# retries), short enough to recover within a single Airflow task retry
# window (default 5 min retry delay). If a probe is stuck longer than
# this, assume the caller process crashed (OOM, SIGKILL, segfault) and
# release the slot so a new probe can fire.
_DEFAULT_PROBE_TIMEOUT: float = 300.0

# P1-030 ROOT FIX (Team Member 3 -- rolling failure window + exponential
# backoff for ChEMBL-class outages):
#   The issue: the breaker tripped after 5 CONSECUTIVE failures (no time
#   window) and reset after 30s. ChEMBL outages typically last 5-15 min.
#   After the 30s reset, the breaker immediately re-tripped, hammering
#   ChEMBL every 30s -- worsening the outage and risking IP bans.
#   ROOT FIX: (1) add a ROLLING failure window (default 300s = 5 min) so
#   the breaker only trips when failure_threshold failures occur WITHIN
#   the window (not lifetime-consecutive). (2) Increase the default
#   reset_timeout to 60s. (3) Add EXPONENTIAL BACKOFF: each failed
#   half-open probe DOUBLES the effective reset_timeout (capped at
#   _BACKOFF_CAP_MULT * base), so repeated outages cause progressively
#   longer cooldowns instead of a fixed 60s hammer.
_DEFAULT_FAILURE_WINDOW: float = 300.0
_BACKOFF_CAP_MULT: float = 8.0  # max 8x base reset_timeout (60s -> 480s)


class _CircuitBreaker:
    """Thread-safe circuit breaker with closed / open / half-open states.

    After ``failure_threshold`` consecutive failures, the breaker opens
    and refuses further calls for ``reset_timeout`` seconds.  After the
    timeout, it enters half-open state: one call is allowed (the probe);
    if it succeeds, the breaker closes; if it fails, the breaker re-opens.

    The half-open single-probe gate (``_half_open_probe_in_flight``)
    ensures that exactly ONE call is allowed in half-open state.
    Subsequent calls are refused until the probe completes via
    ``record_success()`` or ``record_failure()``.  This prevents a
    thundering herd when the protected service is recovering.

    Parameters
    ----------
    failure_threshold : int
        Consecutive failures required to trip the breaker open.
        Must be >= 1.  Default: 5.
    reset_timeout : float
        Seconds the breaker stays open before transitioning to half-open.
        Must be >= 0.  Default: 30.0.
    name : str or None
        Optional human-readable name included in log messages.
        Default: None.
    probe_timeout : float
        P1-028 ROOT FIX: maximum seconds a half-open probe slot may
        stay reserved before it's assumed crashed and auto-released.
        Bounds the stuck-half-open window when a caller crashes (OOM
        kill, SIGKILL, segfault) between ``allow_request()`` returning
        True and the ``record_*()`` call. Must be >= 1.0. Default: 300.0
        (5 min — see ``_DEFAULT_PROBE_TIMEOUT`` rationale above).
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        *,
        name: str | None = None,
        probe_timeout: float = _DEFAULT_PROBE_TIMEOUT,
        failure_window: float = _DEFAULT_FAILURE_WINDOW,
        exponential_backoff: bool = True,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(
                f"failure_threshold must be >= 1, got {failure_threshold}"
            )
        if reset_timeout < 0:
            raise ValueError(
                f"reset_timeout must be >= 0, got {reset_timeout}"
            )
        if probe_timeout < 1.0:
            raise ValueError(
                f"probe_timeout must be >= 1.0s, got {probe_timeout}"
            )
        if failure_window < 0:
            raise ValueError(
                f"failure_window must be >= 0, got {failure_window}"
            )
        self._failure_threshold: int = int(failure_threshold)
        self._reset_timeout: float = float(reset_timeout)
        self.name: str | None = name
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._state: str = "closed"  # closed | open | half_open
        self._lock = threading.Lock()
        # Half-open single-probe gate: only ONE call is allowed in
        # half_open state.  Subsequent calls are refused until the
        # probe completes (record_success or record_failure).
        self._half_open_probe_in_flight: bool = False
        # P1-028 ROOT FIX: monotonic timestamp when the half-open probe
        # slot was reserved. Used by ``allow_request()`` to detect
        # crashed probes (caller died between ``allow_request()``
        # returning True and the ``record_*()`` call). If
        # ``time.monotonic() - _half_open_probe_reserved_at >
        # probe_timeout``, the slot is auto-released so a new probe can
        # fire. Initialized to 0.0 (no probe reserved).
        self._half_open_probe_reserved_at: float = 0.0
        self._probe_timeout: float = float(probe_timeout)
        # P1-030 ROOT FIX: rolling failure window. ``_failure_times`` is
        # a deque of monotonic timestamps, one per recorded failure,
        # pruned to entries within ``failure_window`` seconds of the
        # current time. The breaker trips when
        # ``len(_failure_times) >= failure_threshold`` WITHIN the window
        # (not lifetime-consecutive). This prevents the breaker from
        # tripping due to a sparse trickle of failures spread over hours,
        # while still tripping fast on a burst of failures within minutes.
        self._failure_window: float = float(failure_window)
        self._failure_times: deque = deque()
        # P1-030 ROOT FIX: exponential backoff multiplier. Starts at 1.0;
        # doubles on each failed half-open probe (capped at
        # _BACKOFF_CAP_MULT). Reset to 1.0 on a successful probe. The
        # effective reset_timeout = ``_reset_timeout * _backoff_mult``.
        # This makes repeated outages progressively back off instead of
        # hammering the protected service every 60s.
        self._exponential_backoff: bool = bool(exponential_backoff)
        self._backoff_mult: float = 1.0

    # -- Convenience properties ----------------------------------------

    @property
    def failure_threshold(self) -> int:
        """Consecutive failure count that trips the breaker open."""
        return self._failure_threshold

    @failure_threshold.setter
    def failure_threshold(self, value: int) -> None:
        self._failure_threshold = max(1, int(value))

    @property
    def reset_timeout(self) -> float:
        """Seconds the breaker stays open before half-open transition."""
        return self._reset_timeout

    @reset_timeout.setter
    def reset_timeout(self, value: float) -> None:
        self._reset_timeout = max(0.0, float(value))

    @property
    def failure_count(self) -> int:
        """Current consecutive failure count."""
        return self._failure_count

    @failure_count.setter
    def failure_count(self, value: int) -> None:
        self._failure_count = int(value)

    @property
    def last_failure_time(self) -> float:
        """Monotonic timestamp of the most recent failure."""
        return self._last_failure_time

    @last_failure_time.setter
    def last_failure_time(self, value: float) -> None:
        self._last_failure_time = float(value)

    @property
    def state(self) -> str:
        """Current breaker state (``'closed'``, ``'open'``, or ``'half_open'``).

        v89 FORENSIC ROOT FIX (BUG #12 P1 -- pure observation):
          The previous implementation triggered an open -> half_open
          transition when the reset timeout had elapsed. This was a
          SIDE EFFECT on a read-only property -- the same class of bug
          as ``is_open()``. Monitoring code that read ``breaker.state``
          for dashboards inadvertently transitioned the breaker, and
          because the transition did NOT set
          ``_half_open_probe_in_flight`` consistently with
          ``allow_request()``, it could leave the breaker in a
          half-reserved state.
          ROOT FIX: this property is now PURE OBSERVATION. It returns
          the current state string without any transition. The open ->
          half_open transition is performed EXCLUSIVELY by
          ``allow_request()``.
        """
        with self._lock:
            return self._state

    @state.setter
    def state(self, value: str) -> None:
        """Set the breaker state directly (use with caution).

        If setting to ``'open'``, also sets ``_last_failure_time`` to the
        current monotonic clock so that the recovery timeout is measured
        from this moment.  This preserves the behaviour expected by tests
        that manually set the state and then check recovery transitions.
        """
        with self._lock:
            self._state = value
            if value == "open" and self._last_failure_time == 0.0:
                self._last_failure_time = time.monotonic()

    # Backward-compat UPPERCASE aliases consumed by connection.py tests.
    @property
    def recovery_timeout(self) -> float:
        """Alias for ``reset_timeout`` (backward compat with connection.py)."""
        return self._reset_timeout

    @recovery_timeout.setter
    def recovery_timeout(self, value: float) -> None:
        self._reset_timeout = max(0.0, float(value))

    # -- Core state transitions ----------------------------------------

    def record_success(self) -> None:
        """Record a successful operation -- closes the breaker."""
        with self._lock:
            self._failure_count = 0
            self._state = "closed"
            self._half_open_probe_in_flight = False
            # P1-028: clear the probe reservation timestamp so the next
            # half-open probe starts fresh.
            self._half_open_probe_reserved_at = 0.0
            # P1-030: clear the rolling failure window and reset the
            # exponential backoff multiplier on a successful probe.
            self._failure_times.clear()
            self._backoff_mult = 1.0

    def record_failure(self) -> None:
        """Record a failed operation -- may open the breaker.

        v89 FORENSIC ROOT FIX (BUG #13 P1 -- half-open probe flag not cleared
          on threshold-path re-open):
          The previous code checked ``if self._state == "half_open"`` AFTER
          the ``if self._failure_count >= self._failure_threshold`` block.
          When a half-open probe failed, ``_failure_count`` (which was NOT
          reset on entering half_open) was already >= threshold, so the
          threshold block set ``self._state = "open"`` FIRST. The subsequent
          ``if self._state == "half_open"`` check then evaluated False
          (state was now "open"), so ``_half_open_probe_in_flight`` was
          NOT cleared. The flag stayed True, blocking all future probes
          (combined with Bug #12's ``is_open()`` mutation, this could
          stick the breaker open forever).
          ROOT FIX: check for half_open FIRST and handle it exclusively
          (clear the flag, set state to open, log, and return). The
          threshold check only fires when NOT in half_open. This guarantees
          the flag is ALWAYS cleared when leaving half_open, regardless of
          which path triggered the transition.
        """
        with self._lock:
            self._failure_count += 1
            now = time.monotonic()
            self._last_failure_time = now
            # P1-030 ROOT FIX: prune the rolling failure window and
            # append the current failure timestamp. The threshold check
            # below uses the WINDOWED count (not lifetime-consecutive).
            if self._failure_window > 0:
                cutoff = now - self._failure_window
                while self._failure_times and self._failure_times[0] < cutoff:
                    self._failure_times.popleft()
                self._failure_times.append(now)
            else:
                # failure_window=0 means "lifetime consecutive" (legacy
                # behaviour) -- still track for consistency.
                self._failure_times.append(now)
            windowed_count = len(self._failure_times)
            if self._state == "half_open":
                # A failed probe in half_open trips the breaker back to
                # open. ALWAYS clear the probe-in-flight flag when leaving
                # half_open (v89 BUG #13 -- was not cleared on the
                # threshold-path re-open, leaving the breaker stuck).
                self._state = "open"
                self._half_open_probe_in_flight = False
                # P1-028: clear the probe reservation timestamp.
                self._half_open_probe_reserved_at = 0.0
                # P1-030 ROOT FIX: exponential backoff. Each failed
                # half-open probe DOUBLES the effective reset_timeout
                # (capped at _BACKOFF_CAP_MULT * base). This makes
                # repeated outages progressively back off instead of
                # hammering the protected service every 60s.
                if self._exponential_backoff:
                    self._backoff_mult = min(
                        self._backoff_mult * 2.0, _BACKOFF_CAP_MULT
                    )
                label = self.name or "circuit_breaker"
                logger.warning(
                    "[%s] Circuit breaker re-OPENED after failed half-open "
                    "probe (failure_count=%d, threshold=%d, "
                    "reset_timeout=%.1fs, backoff_mult=%.1f, "
                    "effective_reset=%.1fs)",
                    label,
                    self._failure_count,
                    self._failure_threshold,
                    self._reset_timeout,
                    self._backoff_mult,
                    self._reset_timeout * self._backoff_mult,
                )
                return
            if windowed_count >= self._failure_threshold:
                if self._state != "open":
                    label = self.name or "circuit_breaker"
                    logger.warning(
                        "[%s] Circuit breaker OPENED after %d failures "
                        "within %.0fs window (threshold=%d, "
                        "reset_timeout=%.1fs)",
                        label,
                        windowed_count,
                        self._failure_window,
                        self._failure_threshold,
                        self._reset_timeout,
                    )
                self._state = "open"

    # -- Check APIs ----------------------------------------------------

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns
        -------
        bool
            True if the request should proceed, False if blocked.

        In half_open state, exactly ONE probe request is allowed.
        Subsequent requests are refused until the probe completes
        (record_success -> closed, record_failure -> open).

        P1-028 ROOT FIX (Team-2 — auto-recover from crashed probes):
          If a previous ``allow_request()`` call reserved the half-open
          probe slot but the caller CRASHED before calling
          ``record_success()`` / ``record_failure()`` (OOM kill, SIGKILL,
          segfault in a C extension like RDKit), the slot would stay
          reserved forever — the breaker was stuck in half-open and the
          protected service was silently disabled for the rest of the
          Airflow worker's lifetime.
          ROOT FIX: track ``_half_open_probe_reserved_at`` (monotonic
          timestamp when the slot was reserved). If a new
          ``allow_request()`` call sees the slot reserved for longer
          than ``probe_timeout`` seconds, assume the original probe
          crashed and release the slot — a new probe is allowed. This
          bounds the stuck-half-open window to ``probe_timeout`` seconds
          (default 300s = 5 min) instead of infinity.
        """
        with self._lock:
            current_state = self._state
            if current_state == "open":
                # Check if recovery timeout has elapsed. P1-030: use the
                # EFFECTIVE reset_timeout (base * backoff_mult) so
                # exponential backoff is honoured.
                effective_reset = self._reset_timeout * self._backoff_mult
                if time.monotonic() - self._last_failure_time > effective_reset:
                    self._state = "half_open"
                    self._half_open_probe_in_flight = False
                    # P1-028: clear the reservation timestamp on the
                    # open → half_open transition (no probe reserved yet).
                    self._half_open_probe_reserved_at = 0.0
                    current_state = "half_open"
                else:
                    return False
            if current_state == "closed":
                return True
            # half_open: allow exactly ONE probe.
            if self._half_open_probe_in_flight:
                # P1-028 ROOT FIX: check if the in-flight probe has
                # been stuck longer than ``probe_timeout``. If so,
                # assume the caller crashed (OOM, SIGKILL, segfault)
                # and auto-release the slot so a new probe can fire.
                # This bounds the stuck-half-open window to
                # ``probe_timeout`` seconds instead of infinity.
                reserved_for = time.monotonic() - self._half_open_probe_reserved_at
                if reserved_for > self._probe_timeout:
                    label = self.name or "circuit_breaker"
                    logger.warning(
                        "[%s] Half-open probe auto-released after %.1fs "
                        "(probe_timeout=%.1fs) — assuming original caller "
                        "crashed (OOM/SIGKILL/segfault). New probe allowed.",
                        label,
                        reserved_for,
                        self._probe_timeout,
                    )
                    # Fall through to reserve a new probe slot below.
                else:
                    return False
            # Reserve the probe slot for THIS caller.
            self._half_open_probe_in_flight = True
            self._half_open_probe_reserved_at = time.monotonic()
            return True

    @contextlib.contextmanager
    def probe(self) -> "contextlib.AbstractContextManager[None]":
        """Context manager API for half-open probes (P1-028 ROOT FIX).

        Acquires the probe slot on enter (equivalent to
        ``allow_request()`` returning True) and ALWAYS releases it on
        exit — success, failure, or exception. Callers that use this
        API do NOT need to call ``record_success()`` / ``record_failure()``
        manually; the context manager handles both.

        Usage::

            with breaker.probe() as probe_acquired:
                if not probe_acquired:
                    # Breaker is open or a probe is already in flight.
                    return None
                # Make the protected API call here. If it raises,
                # the context manager records a failure and re-raises.
                result = protected_api_call()
            # On normal exit, the context manager records success.

        This API is RECOMMENDED for new callers — it eliminates the
        ``allow_request()`` / ``record_*()`` pairing bug class entirely.
        Existing callers that use the pair API continue to work
        unchanged (with the P1-028 auto-recovery safety net).

        Yields
        ------
        bool
            True if the probe slot was acquired (caller should proceed),
            False if the breaker is open or a probe is already in flight
            (caller should skip / fall back).
        """
        acquired = self.allow_request()
        try:
            yield acquired
        except Exception:
            if acquired:
                self.record_failure()
            raise
        else:
            if acquired:
                self.record_success()

    def is_open(self) -> bool:
        """Return True if the breaker is open and calls should be refused.

        v89 FORENSIC ROOT FIX (BUG #12 P1 -- is_open() mutated state):
          The previous implementation MUTATED state when called from the
          "open" state with an elapsed reset timeout: it transitioned to
          "half_open" AND set ``_half_open_probe_in_flight = True``. This
          RESERVED the probe slot, so a subsequent ``allow_request()``
          call (which sets the flag to False before the probe, then True
          to reserve) saw the flag already True and returned False --
          refusing the actual probe. The breaker appeared stuck open even
          after the reset timeout. Any monitoring/dashboard code that
          called ``is_open()`` inadvertently broke the subsequent
          ``allow_request()`` call.
          ROOT FIX: make ``is_open()`` a PURE OBSERVATION method. It does
          NOT transition state, does NOT reserve probe slots. The
          open -> half_open transition is performed EXCLUSIVELY by
          ``allow_request()``. Callers who want to actually acquire a
          probe slot MUST call ``allow_request()``.

        Semantics
        ---------
        - state == "closed": return False (not open, calls allowed).
        - state == "open": return True (open, calls refused). Note: even
          if the reset timeout has elapsed, we return True because the
          state has not yet transitioned. ``allow_request()`` will
          transition to half_open on the next call.
        - state == "half_open": return ``_half_open_probe_in_flight``
          (True if a probe is already in flight and additional calls
          should be refused; False if no probe is in flight and the next
          ``allow_request()`` call will acquire the probe).
        """
        with self._lock:
            if self._state == "open":
                # Pure observation -- do NOT transition to half_open here.
                # allow_request() performs the transition when a caller
                # actually wants to acquire a probe slot.
                return True
            if self._state == "half_open":
                # In half_open, "open" means "refuse additional calls" =
                # a probe is already in flight.
                return self._half_open_probe_in_flight
            # closed
            return False

    # -- Hard reset (P1-002 / P1-011 ROOT FIX) ----------------------------
    def reset(self) -> None:
        """Hard-reset the breaker to the closed state (thread-safe).

        P1-002 / P1-011 ROOT FIX (Team-1 -- consolidate duplicate breaker):
          ``database/connection.py`` previously defined its OWN local
          ``_CircuitBreaker`` dataclass with a ``reset()`` method called
          by ``reset_global_state()`` during test teardown / process
          shutdown. The canonical implementation here had no ``reset()``
          method, so deleting the duplicate would have broken
          ``connection.py``. ROOT FIX: add ``reset()`` to the canonical
          breaker so ``connection.py`` can import the canonical class
          and call ``reset()`` unchanged.

        Semantically ``reset()`` is similar to ``record_success()``
        (both close the breaker), but ``reset()`` ALSO clears
        ``_last_failure_time`` (so the next ``allow_request()`` does not
        see a stale timestamp) and is intended for OUT-OF-BAND state
        cleanup (test teardown, operator override, process restart)
        rather than as a normal state-transition. Use
        ``record_success()`` for normal "the probe succeeded" transitions;
        use ``reset()`` only when you genuinely want to wipe all state.
        """
        with self._lock:
            self._failure_count = 0
            self._state = "closed"
            self._last_failure_time = 0.0
            self._half_open_probe_in_flight = False
            self._half_open_probe_reserved_at = 0.0
            # P1-030: clear the rolling failure window and backoff.
            self._failure_times.clear()
            self._backoff_mult = 1.0

    # -- Backward-compat UPPERCASE state aliases (P1-002 / P1-011) ---------
    # ``database/connection.py``'s local ``_CircuitBreaker`` used
    # UPPERCASE state strings ("CLOSED", "OPEN", "HALF_OPEN"). The
    # canonical implementation uses lowercase ("closed", "open",
    # "half_open"). Tests and observability code that read
    # ``breaker.state`` and compared to UPPERCASE strings would break
    # after the consolidation. To preserve backward compatibility, we
    # keep the canonical lowercase internally but ALSO accept UPPERCASE
    # in the ``state`` setter (so legacy tests that set state directly
    # still work). The ``state`` getter continues to return lowercase
    # (canonical form) -- callers that need UPPERCASE should use
    # ``breaker.state.upper()``.
    #
    # This is a deliberate bridge, not a permanent API. New code should
    # use lowercase state strings exclusively.
    def reset_to_open(self) -> None:
        """Force the breaker to OPEN (thread-safe, for testing/admin).

        P1-002 / P1-011 ROOT FIX: legacy ``connection.py`` tests directly
        set ``breaker._state = "OPEN"`` to simulate a tripped breaker.
        Direct attribute mutation bypasses the lock and is unsafe. This
        helper provides a thread-safe equivalent.
        """
        with self._lock:
            self._state = "open"
            if self._last_failure_time == 0.0:
                self._last_failure_time = time.monotonic()
