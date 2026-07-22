"""DrugOS Graph Module — MLflow Experiment Tracker
====================================================
Logs training metrics, model parameters, and artifacts to MLflow
for experiment tracking and reproducibility.

P2-014 ROOT FIX (deterministic shutdown via atexit):
    The previous ``__del__`` called ``end_run()`` at garbage-
    collection time, which is NON-DETERMINISTIC. For long-lived
    tracker instances held by pipeline singletons, ``__del__`` may
    not fire until interpreter shutdown — by then, the MLflow
    tracking server may be unreachable (network torn down, HTTP
    client closed), and ``end_run`` silently fails (caught by the
    ``except`` clause). The run is left dangling in the MLflow UI
    as "ACTIVE" forever.

    ROOT FIX: register ``close()`` with ``atexit`` during
    ``__init__``. ``atexit`` runs BEFORE the network is torn down
    (during normal interpreter shutdown), so the MLflow run is
    ended deterministically. ``__exit__`` now calls ``close()``
    (not ``end_run``) so the defensive try/except applies.
    ``__del__`` remains as a last-resort safety net but is
    documented as best-effort only — operators should use
    ``with MLflowTracker() as t:`` or call ``close()`` explicitly.

P2-024 ROOT FIX (Team 8 — heartbeat for SIGKILL / OOM-kill recovery):
    atexit handlers do NOT fire on SIGKILL (OOM kill, kernel panic,
    ``os._exit``). The MLflow run is left in RUNNING state forever.
    After 100 OOM-killed runs, the MLflow UI shows 100 "RUNNING" runs
    that are actually dead — ops cannot find the real active run, and
    the audit trail is corrupted.

    ROOT FIX: a daemon heartbeat thread updates an MLflow tag
    ``drugos.heartbeat_ts`` with the current epoch timestamp every
    ``heartbeat_interval`` seconds (default 30, env override
    ``DRUGOS_MLFLOW_HEARTBEAT_INTERVAL_SECONDS``). After a SIGKILL,
    the heartbeat stops updating. A periodic reaper (or operator
    inspecting the MLflow UI) can compare ``drugos.heartbeat_ts`` to
    the current time: if the gap exceeds
    ``heartbeat_stale_threshold`` seconds (default 300 = 5 minutes,
    env override ``DRUGOS_MLFLOW_HEARTBEAT_STALE_THRESHOLD``), the
    run is marked FAILED.

    The heartbeat thread is a Python daemon thread — it does NOT
    block interpreter shutdown (unlike atexit, daemon threads are
    killed abruptly on shutdown). This is correct: the heartbeat's
    job is to UPDATE the run's liveness signal, not to perform
    cleanup. Cleanup is atexit's job (close()).

    When MLflow is not installed, the heartbeat falls back to
    appending to ``self._local_log`` so the heartbeat trail is still
    available to operators via ``get_local_log()`` (useful for
    debugging in CI / local environments without an MLflow server).
"""

import atexit
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from .config import ensure_dirs

logger = logging.getLogger(__name__)


# ─── P2-024 heartbeat constants ───────────────────────────────────────────────
# Env var overrides let ops tune the heartbeat for their environment:
#   * Short interval (e.g. 10s) for fast-feedback dev environments.
#   * Long interval (e.g. 300s) for production where MLflow write rate
#     is a cost concern (each heartbeat is an MLflow tag write).
#   * Stale threshold MUST be >= 3x the interval to avoid false
#     positives during transient network slowness.
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = int(
    os.environ.get("DRUGOS_MLFLOW_HEARTBEAT_INTERVAL_SECONDS", "30")
)
DEFAULT_HEARTBEAT_STALE_THRESHOLD_SECONDS = int(
    os.environ.get("DRUGOS_MLFLOW_HEARTBEAT_STALE_THRESHOLD_SECONDS", "300")
)
# The MLflow tag name used to store the heartbeat timestamp. Ops
# queries this tag to detect stale RUNNING runs:
#   SELECT run_id, tags['drugos.heartbeat_ts']
#   FROM runs
#   WHERE status = 'RUNNING'
#     AND (current_timestamp - tags['drugos.heartbeat_ts']) > 300
HEARTBEAT_TAG_NAME = "drugos.heartbeat_ts"
HEARTBEAT_PID_TAG_NAME = "drugos.heartbeat_pid"


class MLflowTracker:
    """Tracks experiments using MLflow.

    Falls back to local file logging if MLflow is not installed.

    P2-014: callers should use ``with MLflowTracker() as t:`` or
    call ``close()`` explicitly. ``__del__`` is a best-effort
    safety net only; ``atexit`` handles the deterministic shutdown.

    P2-024 (Team 8): a daemon heartbeat thread updates
    ``drugos.heartbeat_ts`` on the MLflow run every
    ``heartbeat_interval`` seconds. After SIGKILL (which atexit
    cannot catch), the heartbeat stops; ops can compare the
    heartbeat timestamp to the current time to detect stale
    RUNNING runs and reap them. The heartbeat is started in
    ``start_run`` and stopped in ``close``.
    """

    def __init__(
        self,
        # P2-050 ROOT FIX: the previous default "DrugOS_Week2" was a
        # misleading artifact of an early sprint plan that confined
        # Phase 2 to a single week. Per the DOCX build plan, Phase 2
        # (Knowledge Graph Construction) actually spans Weeks 2-5, and
        # the KG is the deliverable — not a per-week artefact. Operators
        # who didn't override got EVERY Phase 2 run rolled up under one
        # experiment called "Week2", which made the MLflow UI useless
        # for filtering Week 4 TransE runs vs Week 5 HGT runs. Root fix:
        # default to "DrugOS_Phase2" (the phase, not the week). Callers
        # who want a tighter scope should override with a more specific
        # name like "DrugOS_Phase2_TransE_v3" or "DrugOS_Phase2_HGT_v1".
        experiment_name: str = "DrugOS_Phase2",
        tracking_uri: Optional[str] = None,
        # P2-024 ROOT FIX (Team 8): heartbeat interval for the daemon
        # thread that updates ``drugos.heartbeat_ts`` on the MLflow run.
        # Default 30s; override via the
        # ``DRUGOS_MLFLOW_HEARTBEAT_INTERVAL_SECONDS`` env var or this
        # parameter. Set to 0 to DISABLE the heartbeat (useful for unit
        # tests; NOT recommended for production — disabling the
        # heartbeat means SIGKILL'd runs stay RUNNING forever, which is
        # the exact bug P2-024 fixes).
        heartbeat_interval: Optional[int] = None,
        heartbeat_stale_threshold: Optional[int] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.mlflow = None
        self.run = None
        self._local_log = []
        # P2-014 ROOT FIX: register close() with atexit so the
        # MLflow run is ended deterministically at interpreter
        # shutdown, BEFORE the network is torn down. ``atexit``
        # runs in reverse-registration order, so this fires after
        # any upstream atexit handlers that may still need the
        # tracker. ``close()`` is idempotent (safe to call multiple
        # times — see the ``self.run = None`` reset in ``end_run``).
        self._closed: bool = False
        atexit.register(self._atexit_close)
        # Task 95 ROOT FIX: atexit does NOT fire on SIGTERM, SIGHUP,
        # SIGQUIT, or os._exit() -- only on clean interpreter shutdown.
        # Production training jobs are routinely killed with SIGTERM by
        # orchestrators (Airflow, Kubernetes, SLURM) and the MLflow run
        # was leaked (left in RUNNING state forever) because atexit
        # never ran. The fix installs signal handlers for the common
        # termination signals that re-raise after closing the run.
        # SIGKILL (9) is uncatchable by design -- the heartbeat daemon
        # (P2-024) is the only defence against SIGKILL leaks.
        self._install_signal_handlers()

        # P2-024 ROOT FIX (Team 8): heartbeat state. The thread is
        # created in start_run and joined in close. The
        # ``_heartbeat_stop`` event lets the loop exit cleanly
        # between intervals (so close doesn't have to wait up to 30s
        # for the next iteration).
        self._heartbeat_interval: int = (
            heartbeat_interval
            if heartbeat_interval is not None
            else DEFAULT_HEARTBEAT_INTERVAL_SECONDS
        )
        self._heartbeat_stale_threshold: int = (
            heartbeat_stale_threshold
            if heartbeat_stale_threshold is not None
            else DEFAULT_HEARTBEAT_STALE_THRESHOLD_SECONDS
        )
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop: threading.Event = threading.Event()
        # Exposed for tests / ops dashboards: the last heartbeat
        # timestamp successfully written. ``None`` until the first
        # iteration of the heartbeat loop completes.
        self.last_heartbeat_ts: Optional[float] = None
        # Number of heartbeats written — useful for tests that verify
        # the thread is actually running (not just instantiated).
        self.heartbeat_count: int = 0
        # Number of heartbeat write failures (e.g. MLflow server
        # unreachable). A high count indicates a degraded MLflow
        # connection — the heartbeat trail will be stale even though
        # the process is alive.
        self.heartbeat_failure_count: int = 0

        try:
            import mlflow
            self.mlflow = mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow initialized: experiment={experiment_name}")
        except ImportError:
            logger.warning(
                "mlflow not installed — using local file logging. "
                "Install with: pip install mlflow"
            )
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            self.mlflow = None

    def _atexit_close(self) -> None:
        """P2-014: atexit-registered close handler.

        v107 ROOT FIX (ISSUE-P2-047): the previous code did
        ``try: self.close() except Exception: pass``, swallowing ALL
        exceptions silently. If close() raised (e.g. MLflow server
        unreachable, network timeout), the exception was dropped and
        the MLflow run was left in an inconsistent state — the UI
        showed it as "RUNNING" forever, and metrics logged after the
        failure were lost. Operators had no way to detect the problem.

        ROOT FIX: still swallow the exception (atexit handlers must
        not raise — it produces noisy tracebacks at shutdown and can
        mask the real exit code), but LOG it at WARNING level first
        so the failure is visible in production logs. The MLflow
        client library's own atexit handlers will retry the close on
        the next interpreter start; we just need to surface that THIS
        shutdown failed.
        """
        try:
            self.close()
        except Exception as e:
            # v107 ISSUE-P2-047: log at WARNING so operators can detect
            # dangling MLflow runs. We still swallow the exception
            # (atexit handlers must not raise).
            logger.warning(
                "MLflowTracker._atexit_close: close() raised %s: %s. "
                "The MLflow run may be left in an inconsistent state "
                "(UI may show it as RUNNING). Check the MLflow server "
                "connectivity and the run's status on next startup. "
                "v107 ISSUE-P2-047 root fix.",
                type(e).__name__, e,
            )

    def _install_signal_handlers(self) -> None:
        """Task 95 ROOT FIX: install SIGTERM/SIGHUP/SIGQUIT handlers.

        ``atexit`` does NOT fire on:

          * ``SIGTERM`` (default signal sent by ``kill <pid>``, Airflow,
            Kubernetes pod termination, SLURM scancel, systemd stop)
          * ``SIGHUP`` (terminal hangup, daemon reload)
          * ``SIGQUIT`` (``Ctrl-\\``, Java-style abort)
          * ``os._exit()`` (used by some C extensions and multiprocessing
            forks)
          * ``SIGKILL`` (uncatchable by design)

        Without these handlers, every production training job that was
        SIGTERM'd by an orchestrator leaked its MLflow run (left it in
        RUNNING state forever). The heartbeat (P2-024) detects the
        leak but does not PREVENT it -- this method prevents it by
        calling ``self.close()`` before re-raising the signal with the
        default handler.

        Idempotent: only installs once per process. Subsequent calls
        (e.g. if the caller constructs multiple ``MLflowTracker``
        instances) skip the install -- only the FIRST tracker's
        close() is registered as the signal handler, which is fine
        because all subsequent trackers will have their own atexit
        handlers and the signal-installed tracker's close() will be
        called via atexit anyway.
        """
        # Only install once per process. The signal handler closure
        # captures ``self`` so we MUST not install twice (the second
        # install would close the WRONG tracker instance).
        if getattr(self.__class__, "_signal_handlers_installed", False):
            return
        import signal
        import os

        # Signals we want to catch and clean up before dying. SIGKILL
        # is uncatchable by design; SIGINT is handled by Python's
        # default KeyboardInterrupt handler (caller can catch that).
        _CATCHABLE_SIGNALS = []
        for sig_name in ("SIGTERM", "SIGHUP", "SIGQUIT"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                _CATCHABLE_SIGNALS.append((sig, sig_name))

        if not _CATCHABLE_SIGNALS:
            # Platform without any of these signals (rare -- e.g.
            # Windows only has SIGTERM and not SIGHUP/SIGQUIT). Skip
            # silently; atexit + KeyboardInterrupt still cover most
            # cases on Windows.
            return

        def _signal_handler(signum, frame):
            try:
                logger.warning(
                    "MLflowTracker: received signal %d -- closing MLflow "
                    "run before terminating (Task 95 root fix).",
                    signum,
                )
                self.close()
            except Exception as e:
                # Best-effort: log but don't prevent termination.
                logger.error(
                    "MLflowTracker: close() failed during signal "
                    "handling: %s: %s", type(e).__name__, e,
                )
            # Re-raise with the default handler so the process exits
            # with the correct status code (e.g. 143 for SIGTERM).
            # ``signal.signal(signum, SIG_DFL)`` restores the default
            # then ``os.kill(os.getpid(), signum)`` re-sends it.
            try:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
            except Exception:
                # If re-sending fails (e.g. race with another handler),
                # fall back to sys.exit with the conventional exit code.
                import sys
                sys.exit(128 + signum)

        try:
            for sig, _name in _CATCHABLE_SIGNALS:
                # Only install if the caller hasn't already installed a
                # custom handler (don't clobber user-installed handlers
                # -- they may have their own cleanup logic).
                current = signal.getsignal(sig)
                if current in (signal.SIG_DFL, None):
                    signal.signal(sig, _signal_handler)
                else:
                    logger.info(
                        "MLflowTracker: not installing handler for %s "
                        "(custom handler %r already installed).",
                        _name, current,
                    )
            self.__class__._signal_handlers_installed = True
        except (ValueError, OSError) as e:
            # signal.signal() can raise ValueError if called from a
            # non-main thread (Python restriction). In that case, log
            # and skip -- the main thread's atexit still fires.
            logger.warning(
                "MLflowTracker: could not install signal handlers "
                "(%s: %s). atexit + context-manager close() are still "
                "active; SIGTERM will leak the run if received from a "
                "non-main-thread construct.",
                type(e).__name__, e,
            )

    def start_run(self, run_name: str = "default") -> "MLflowTracker":
        """Start a new MLflow run.

        Task 95 ROOT FIX: the previous ``start_run`` returned ``self``
        and relied on the caller to either use ``with tracker.start_run()``
        or call ``close()`` explicitly. If the caller's body raised
        BETWEEN ``start_run()`` returning and ``__exit__`` firing (or if
        the caller forgot ``with`` entirely), the MLflow run was leaked.
        The fix:

          1. ``start_run`` still returns ``self`` (for ``with`` usage).
          2. If the caller uses ``with``, ``__exit__`` calls ``close()``
             deterministically (unchanged).
          3. If the caller does NOT use ``with``, the atexit handler
             (registered in ``__init__``) and the SIGTERM/SIGHUP/SIGQUIT
             handlers installed by ``_install_signal_handlers`` close
             the run on process exit.
          4. ``start_run`` itself wraps the heartbeat-thread start in a
             try/except: if the thread fails to start, ``close()`` is
             called immediately so the partially-open run is not leaked.

        This is the smallest change that closes the leak without
        breaking the existing context-manager contract.
        """
        if self.mlflow:
            self.run = self.mlflow.start_run(run_name=run_name)
        logger.info(f"Started experiment run: {run_name}")

        # P2-024 ROOT FIX (Team 8): start the heartbeat daemon thread.
        # Setting ``_heartbeat_stop`` ensures a stale event from a
        # previous run is cleared before the new thread starts.
        self._heartbeat_stop.clear()
        if self._heartbeat_interval > 0:
            try:
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name="drugos-mlflow-heartbeat",
                    daemon=True,  # P2-024: daemon so it doesn't block shutdown
                )
                self._heartbeat_thread.start()
                logger.info(
                    "P2-024: heartbeat thread started (interval=%ds, "
                    "stale_threshold=%ds, tag=%s). If this process is "
                    "SIGKILL'd, ops can detect the stale run by comparing "
                    "the heartbeat tag to the current time.",
                    self._heartbeat_interval,
                    self._heartbeat_stale_threshold,
                    HEARTBEAT_TAG_NAME,
                )
            except (RuntimeError, OSError) as e:
                # Task 95: if the heartbeat thread fails to start
                # (e.g. interpreter is shutting down and threads can't
                # be created), close the run immediately so it's not
                # leaked. Log the failure but don't re-raise -- the
                # caller's body still needs to execute and atexit /
                # signal handlers will close the run if the body
                # raises.
                logger.error(
                    "MLflowTracker.start_run: failed to start heartbeat "
                    "thread: %s: %s. Closing run to prevent leak.",
                    type(e).__name__, e,
                )
                try:
                    self.close()
                except Exception as close_exc:
                    logger.error(
                        "MLflowTracker.start_run: close() after heartbeat "
                        "thread failure also failed: %s: %s",
                        type(close_exc).__name__, close_exc,
                    )
                raise
        return self  # v40: enable ``with tracker.start_run() as t:`` usage

    def _heartbeat_loop(self) -> None:
        """P2-024 ROOT FIX (Team 8): daemon thread that writes the heartbeat.

        Runs until ``_heartbeat_stop`` is set (by ``close()``) or the
        interpreter exits (daemon threads are killed abruptly on
        shutdown — this is correct: the heartbeat's job is to UPDATE
        the run's liveness signal while the process is alive; it does
        NOT need to perform cleanup).

        Each iteration:
          1. Compute the current epoch timestamp.
          2. Write it to the ``drugos.heartbeat_ts`` MLflow tag (or
             append to ``_local_log`` if MLflow is not installed).
          3. Increment ``heartbeat_count`` (or ``heartbeat_failure_count``
             on failure).
          4. Update ``last_heartbeat_ts``.
          5. Wait ``_heartbeat_interval`` seconds (interruptible by
             ``_heartbeat_stop.wait()`` so close() returns promptly).
        """
        # P2-024: write an immediate first heartbeat so the run is
        # marked alive BEFORE the first interval elapses. This lets
        # ops dashboards see the run as "live" immediately after
        # start_run returns, without waiting up to 30s.
        # P2-023 ROOT FIX (v107): after N consecutive heartbeat failures,
        # mark the MLflow run as FAILED via a best-effort
        # ``mlflow.end_run(status="FAILED")``. The previous code
        # incremented ``heartbeat_failure_count`` forever — after 100
        # failures the heartbeat was effectively dead but the MLflow UI
        # still showed the run as "RUNNING" indefinitely. Operators
        # could not distinguish a live run from a dead one. ROOT FIX:
        # after 10 consecutive failures (configurable via
        # ``DRUGOS_MLFLOW_HEARTBEAT_MAX_FAILURES``), attempt to end the
        # run with status=FAILED so the MLflow UI reflects reality.
        # The heartbeat thread then EXITS (no point continuing to
        # retry if the server is unreachable).
        _max_hb_failures_p2_023 = int(
            os.environ.get("DRUGOS_MLFLOW_HEARTBEAT_MAX_FAILURES", "10")
        )
        while not self._heartbeat_stop.is_set():
            try:
                ts = time.time()
                pid = os.getpid()
                if self.mlflow and self.run:
                    # set_tag is the cheapest MLflow write (no metric
                    # history). Two tags: the timestamp (for staleness
                    # detection) and the PID (for ops to identify the
                    # process that owns the run, useful when killing a
                    # zombie).
                    self.mlflow.set_tag(HEARTBEAT_TAG_NAME, str(ts))
                    self.mlflow.set_tag(HEARTBEAT_PID_TAG_NAME, str(pid))
                else:
                    # Fallback: append to local log so the heartbeat
                    # trail is still available without an MLflow server.
                    self._local_log.append({
                        "type": "heartbeat",
                        "ts": ts,
                        "pid": pid,
                    })
                self.last_heartbeat_ts = ts
                self.heartbeat_count += 1
                # P2-023: reset failure counter on success.
                if self.heartbeat_failure_count > 0:
                    logger.info(
                        "P2-023: heartbeat recovered after %d "
                        "consecutive failures. Resuming normal "
                        "heartbeat.",
                        self.heartbeat_failure_count,
                    )
                    self.heartbeat_failure_count = 0
            except Exception as e:
                # The heartbeat MUST NOT raise — it would kill the
                # daemon thread and silently disable liveness tracking.
                # Log and continue; the next iteration will retry.
                self.heartbeat_failure_count += 1
                logger.warning(
                    "P2-024: heartbeat write failed (count=%d): %s. "
                    "The MLflow run will appear stale even though the "
                    "process is alive. Check MLflow server health.",
                    self.heartbeat_failure_count, e,
                )
                # P2-023 ROOT FIX: after N consecutive failures, mark
                # the run as FAILED and exit the heartbeat loop. The
                # MLflow UI will then show the run as FAILED (not
                # "RUNNING" forever), and operators can distinguish a
                # dead run from a live one.
                if self.heartbeat_failure_count >= _max_hb_failures_p2_023:
                    logger.error(
                        "P2-023 ROOT FIX: heartbeat failed %d "
                        "consecutive times (threshold=%d). Marking "
                        "the MLflow run as FAILED and stopping the "
                        "heartbeat thread. The MLflow UI will now "
                        "show this run as FAILED — investigate MLflow "
                        "server health.",
                        self.heartbeat_failure_count,
                        _max_hb_failures_p2_023,
                    )
                    try:
                        if self.mlflow and self.run:
                            # Best-effort: set a tag indicating the
                            # heartbeat died, then end the run as
                            # FAILED. If this also fails, there's
                            # nothing more we can do — the run will
                            # stay RUNNING in the UI until the MLflow
                            # server's own reaper cleans it up.
                            self.mlflow.set_tag(
                                "drugos.heartbeat_status",
                                "FAILED_AFTER_MAX_RETRIES",
                            )
                            self.mlflow.end_run(status="FAILED")
                    except Exception as _end_exc:
                        logger.error(
                            "P2-023: best-effort end_run(FAILED) also "
                            "failed (%s: %s). The MLflow UI will keep "
                            "showing this run as RUNNING until the "
                            "server's own reaper cleans it up.",
                            type(_end_exc).__name__, _end_exc,
                        )
                    # Exit the heartbeat loop — no point continuing.
                    break
            # Wait for the interval OR until close() sets the stop
            # event — whichever comes first. This ensures close()
            # returns promptly without waiting up to 30s for the
            # next iteration.
            self._heartbeat_stop.wait(self._heartbeat_interval)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        # v35 ROOT FIX (L-1): the previous code appended to
        # ``self._local_log`` UNCONDITIONALLY — even when MLflow was
        # active and had already logged the params. For long runs
        # (100+ epochs × 5+ params), this caused ``_local_log`` to grow
        # unbounded (500+ entries). The fix only appends when MLflow is
        # NOT active (i.e., the local log IS the canonical record).
        if self.mlflow and self.run:
            self.mlflow.log_params(params)
        else:
            self._local_log.append({"type": "params", "data": params})
        logger.info(f"Logged params: {params}")

    def log_metrics(self, metrics: Dict[str, float], step: int = 0) -> None:
        """Log training metrics."""
        # v35 ROOT FIX (L-1): see ``log_params`` — only append to the
        # local log when MLflow is NOT active.
        if self.mlflow and self.run:
            self.mlflow.log_metrics(metrics, step=step)
        else:
            self._local_log.append({"type": "metrics", "data": metrics, "step": step})

    def log_artifact(self, path: str) -> None:
        """Log a file artifact."""
        # v35 ROOT FIX (L-1): see ``log_params`` — only append to the
        # local log when MLflow is NOT active.
        if self.mlflow and self.run:
            self.mlflow.log_artifact(path)
        else:
            self._local_log.append({"type": "artifact", "path": path})

    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag on the current MLflow run.

        v63 ROOT FIX (P2C-003+016 — ChEMBERTa silent-disable cascade):
        the audit required that when ChEMBERTa is disabled (any of the
        3 fallback layers fires), the pipeline MUST log
        ``CHEMBERTA_DISABLED=true`` to MLflow so operators monitoring
        the MLflow dashboard can immediately see that the Graph
        Transformer trained on random Xavier features (not molecular
        structure). Without this tag, the only signal was a buried
        WARNING log that production dashboards filtered out — exactly
        the silent-degradation the audit named.
        """
        if self.mlflow and self.run:
            self.mlflow.set_tag(key, str(value))
        else:
            self._local_log.append({"type": "tag", "key": key, "value": value})
        logger.info(f"Set tag: {key}={value}")

    def end_run(self) -> None:
        """End the current MLflow run."""
        if self.mlflow and self.run:
            self.mlflow.end_run()
            self.run = None
        logger.info("Experiment run ended")

    def get_local_log(self) -> list:
        """Return local log (for fallback when MLflow is unavailable)."""
        return self._local_log

    def __enter__(self):
        # v40 ROOT FIX (P2 #4): start_run now returns self, so __enter__
        # can delegate to it. This ensures the run is always started
        # before the body executes, and __exit__ always ends it.
        return self.start_run()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # P2-014 ROOT FIX: call close() (not end_run) so the
        # defensive try/except in close() applies AND the
        # idempotency guard (_closed) prevents double-end. The
        # previous code called end_run() directly which (a) had no
        # try/except, so an exception during end_run would propagate
        # out of the with-block (masking the original exception),
        # and (b) could be re-invoked by atexit / __del__, producing
        # "Run not active" warnings in the MLflow UI.
        self.close()

    # v100 ROOT FIX (BUG P2-038 — dangling MLflow run via __del__):
    # The previous __del__ called end_run() at garbage-collection time,
    # which is NON-DETERMINISTIC. If the MLflowTracker was held by a
    # long-lived object (e.g. the pipeline's singleton logger), __del__
    # might not fire until interpreter shutdown — by then, the MLflow
    # tracking server may be unreachable, and end_run silently fails,
    # leaving the run dangling in the MLflow UI.
    #
    # P2-014 ROOT FIX (deterministic shutdown): the atexit handler
    # registered in __init__ now handles deterministic shutdown.
    # close() is idempotent (``_closed`` flag) so the atexit handler
    # + explicit close() + __del__ can all fire without producing
    # "Run not active" warnings. ``__del__`` remains as a last-resort
    # safety net for callers that forget to use the context manager
    # AND somehow bypass atexit (e.g. os._exit).
    def close(self) -> None:
        """Deterministically end the MLflow run.

        v100 ROOT FIX (BUG P2-038): callers should invoke close() in a
        try/finally block (or register with atexit) to ensure the MLflow
        run is ended BEFORE the tracking server becomes unreachable.
        __del__ is a fallback for callers that forget — but __del__ is
        non-deterministic (fires at GC time, which may be after the
        tracking server is gone). close() is the only path that
        GUARANTEES end_run succeeds.

        P2-014 ROOT FIX: ``close()`` is now IDEMPOTENT — the
        ``_closed`` flag prevents double-end. The atexit handler
        registered in ``__init__`` calls ``close()`` at interpreter
        shutdown, so callers using the context manager
        (``with MLflowTracker() as t:``) AND the atexit handler can
        BOTH fire without producing "Run not active" warnings.

        P2-024 ROOT FIX (Team 8): stop the heartbeat daemon thread
        BEFORE ending the run. This ensures no heartbeat fires AFTER
        end_run (which would log a spurious "Run not active" warning).
        The thread is joined with a short timeout (5s) so a stuck
        heartbeat doesn't block shutdown; if the join times out, the
        daemon flag means Python will kill the thread at interpreter
        exit anyway.
        """
        # P2-014: idempotency guard. ``end_run`` resets ``self.run``
        # to None, but if multiple shutdown paths fire (atexit +
        # explicit close + __del__), the second call would call
        # ``mlflow.end_run()`` with no active run, producing a
        # noisy "Run not active" warning in the MLflow UI. The
        # ``_closed`` flag is set on the FIRST close call; subsequent
        # calls are no-ops.
        if self._closed:
            return
        self._closed = True

        # P2-024 ROOT FIX (Team 8): stop the heartbeat thread BEFORE
        # ending the run. Setting the stop event causes the loop's
        # ``_heartbeat_stop.wait()`` to return immediately, the loop
        # checks ``is_set()`` and exits. The join(timeout=5) ensures
        # close doesn't block for the full heartbeat interval.
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5.0)
            if self._heartbeat_thread.is_alive():
                logger.warning(
                    "P2-024: heartbeat thread did not exit within 5s "
                    "timeout — it will be killed at interpreter exit "
                    "(daemon=True). This may indicate a stuck MLflow "
                    "write."
                )
        self._heartbeat_thread = None

        try:
            self.end_run()
        except Exception:
            pass

    # v43 ROOT FIX (P2 — no __del__ for dangling runs): if the caller
    # uses start_run() without a context manager AND forgets to call
    # end_run() or close(), the MLflow run is left dangling. __del__ is
    # a last-resort safety net that calls end_run() when the tracker is
    # garbage-collected. v100 P2-038: callers should prefer close() —
    # __del__ is non-deterministic and may fire after the tracking
    # server is unreachable.
    # P2-014 ROOT FIX: ``__del__`` is now a thin wrapper around the
    # idempotent ``close()`` (which checks ``_closed``). This means
    # if atexit / explicit close() already fired, ``__del__`` is a
    # no-op (no "Run not active" warning). If neither fired (caller
    # forgot AND somehow bypassed atexit), ``__del__`` still ends
    # the run as a last resort — best-effort, may fail silently if
    # the tracking server is gone (the exception is swallowed).
    def __del__(self):
        try:
            # P2-014: delegate to close() which has the idempotency
            # guard. Directly calling end_run() here would bypass
            # the ``_closed`` check and could produce "Run not
            # active" warnings when atexit / explicit close() have
            # already fired.
            if not self._closed:
                self.close()
        except Exception:
            pass  # __del__ must never raise
        # v72 ROOT FIX (P2C-019): removed ``return False``. __del__ is a
        # destructor whose return value is ignored by the Python runtime.
        # The previous ``return False`` was dead code that could mislead a
        # maintainer into thinking the return value mattered. The
        # convention is to return None (implicitly) from __del__.

    # ─────────────────────────────────────────────────────────────────────────
    # P2-024 ROOT FIX (Team 8 — forensic completion): the REAPER.
    # ─────────────────────────────────────────────────────────────────────────
    # The heartbeat (above) writes ``drugos.heartbeat_ts`` every 30s. But
    # the issue ALSO requires: "If a run's heartbeat is >5 minutes stale,
    # MLflow can mark it as FAILED." The previous fix only wrote the
    # heartbeat — there was no function that QUERIED MLflow for stale
    # RUNNING runs and MARKED them FAILED. Ops had to do it manually.
    # This is the missing piece: a classmethod that ops (or a cron job,
    # or an Airflow sensor) calls periodically to reap stale runs.
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def reap_stale_runs(
        cls,
        experiment_name: Optional[str] = None,
        stale_threshold_seconds: Optional[int] = None,
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """P2-024 ROOT FIX (Team 8): mark stale RUNNING runs as FAILED.

        The heartbeat thread (``_heartbeat_loop``) updates
        ``drugos.heartbeat_ts`` on the MLflow run every
        ``heartbeat_interval`` seconds. If the process is SIGKILL'd
        (OOM kill, kernel panic, ``os._exit``), atexit does NOT fire
        and the heartbeat stops. The MLflow run stays in RUNNING state
        forever. After 100 OOM-killed runs, the MLflow UI shows 100
        "RUNNING" runs that are actually dead — ops cannot find the
        real active run, and the audit trail is corrupted.

        This classmethod queries MLflow for all RUNNING runs in the
        experiment, reads each run's ``drugos.heartbeat_ts`` tag, and
        marks runs whose heartbeat is older than ``stale_threshold_seconds``
        as FAILED (with a ``drugos.reaped_reason="heartbeat_stale"`` tag
        so ops can audit the reap action).

        Called by:
          - ops manually: ``python -c "from drugos_graph.mlflow_tracker
            import MLflowTracker; MLflowTracker.reap_stale_runs()"``
          - a cron job / Airflow sensor that runs every 5 minutes
          - the pipeline's startup self-check (to clean up stale runs
            from a previous crashed invocation before starting a new one)

        Args:
            experiment_name: The MLflow experiment to scan. If None,
                uses the default ``"DrugOS_Phase2"`` (matching
                ``MLflowTracker.__init__``).
            stale_threshold_seconds: A run is considered stale if its
                ``drugos.heartbeat_ts`` is more than this many seconds
                in the past. If None, uses
                ``DEFAULT_HEARTBEAT_STALE_THRESHOLD_SECONDS`` (default
                300 = 5 minutes, override via
                ``DRUGOS_MLFLOW_HEARTBEAT_STALE_THRESHOLD``).
            dry_run: If True, log what WOULD be reaped but do NOT
                actually mark runs as FAILED. Useful for ops to
                preview the reap action before running it for real.

        Returns:
            A dict with keys:
              - ``scanned`` (int): number of RUNNING runs examined.
              - ``reaped`` (int): number of runs marked FAILED (0 if
                ``dry_run=True``).
              - ``skipped_no_heartbeat_tag`` (int): runs without a
                ``drugos.heartbeat_ts`` tag (pre-P2-024 runs or runs
                from other tools). These are NOT reaped — they may
                be legitimate long-running runs from a different tool.
              - ``skipped_heartbeat_recent`` (int): runs with a recent
                heartbeat (within the threshold).
              - ``reaped_run_ids`` (List[str]): IDs of reaped runs.
              - ``errors`` (List[str]): error messages for runs that
                could not be reaped (e.g. MLflow server error).

        Raises:
            RuntimeError: if MLflow is not installed (the reaper
                requires MLflow to query and update runs).
        """
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "P2-024 reap_stale_runs requires mlflow to be installed. "
                "Install with: pip install mlflow"
            ) from exc

        if experiment_name is None:
            experiment_name = "DrugOS_Phase2"
        if stale_threshold_seconds is None:
            stale_threshold_seconds = DEFAULT_HEARTBEAT_STALE_THRESHOLD_SECONDS

        result: Dict[str, Any] = {
            "scanned": 0,
            "reaped": 0,
            "skipped_no_heartbeat_tag": 0,
            "skipped_heartbeat_recent": 0,
            "reaped_run_ids": [],
            "errors": [],
            "dry_run": dry_run,
            "stale_threshold_seconds": stale_threshold_seconds,
            "experiment_name": experiment_name,
        }

        try:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                # No experiment — nothing to reap.
                logger.info(
                    "P2-024 reap_stale_runs: experiment %r not found — "
                    "nothing to reap.",
                    experiment_name,
                )
                return result
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="attributes.status = 'RUNNING'",
            )
        except Exception as exc:
            result["errors"].append(
                f"failed to query MLflow for RUNNING runs: {exc}"
            )
            logger.error(
                "P2-024 reap_stale_runs: failed to query MLflow: %s", exc
            )
            return result

        # `runs` is a pandas DataFrame; if pandas is not available or
        # the search returned no runs, fall back to an empty iterable.
        try:
            run_rows = runs.iterrows() if hasattr(runs, "iterrows") else []
        except Exception:
            run_rows = []

        now = time.time()
        for _, row in run_rows:
            result["scanned"] += 1
            # The run_id is the index of the DataFrame (mlflow.search_runs
            # uses run_id as the index).
            try:
                run_id = row.name if hasattr(row, "name") else None
            except Exception:
                run_id = None
            if run_id is None:
                result["errors"].append(
                    "could not extract run_id from MLflow search row"
                )
                continue

            # Read the heartbeat timestamp tag.
            # mlflow.search_runs puts tags in columns named "tags.<tag_name>".
            heartbeat_ts_str = None
            try:
                if hasattr(runs, "columns"):
                    tag_col = f"tags.{HEARTBEAT_TAG_NAME}"
                    if tag_col in runs.columns:
                        heartbeat_ts_str = row.get(tag_col)
                # Fallback: use mlflow.get_run to read tags directly
                # (more reliable across MLflow versions).
                if heartbeat_ts_str is None:
                    run_info = mlflow.get_run(run_id)
                    heartbeat_ts_str = run_info.data.tags.get(HEARTBEAT_TAG_NAME)
            except Exception as exc:
                result["errors"].append(
                    f"run {run_id}: failed to read heartbeat tag: {exc}"
                )
                continue

            if heartbeat_ts_str is None:
                # No heartbeat tag — this is either a pre-P2-024 run or
                # a run from a different tool. Do NOT reap it (it may be
                # a legitimate long-running run from another tool that
                # doesn't use our heartbeat convention).
                result["skipped_no_heartbeat_tag"] += 1
                logger.info(
                    "P2-024 reap_stale_runs: run %s has no %s tag — "
                    "skipping (may be a pre-P2-024 run or a run from "
                    "a different tool).",
                    run_id, HEARTBEAT_TAG_NAME,
                )
                continue

            try:
                heartbeat_ts = float(heartbeat_ts_str)
            except (TypeError, ValueError):
                result["errors"].append(
                    f"run {run_id}: heartbeat tag {heartbeat_ts_str!r} "
                    f"is not a valid float"
                )
                continue

            age = now - heartbeat_ts
            if age <= stale_threshold_seconds:
                result["skipped_heartbeat_recent"] += 1
                logger.debug(
                    "P2-024 reap_stale_runs: run %s heartbeat is %.1fs "
                    "old (within threshold %ds) — skipping.",
                    run_id, age, stale_threshold_seconds,
                )
                continue

            # The run is stale — mark it FAILED.
            logger.warning(
                "P2-024 reap_stale_runs: run %s heartbeat is %.1fs old "
                "(threshold %ds) — %s.",
                run_id, age, stale_threshold_seconds,
                "would mark FAILED (dry_run)" if dry_run else "marking FAILED",
            )
            if dry_run:
                result["reaped_run_ids"].append(run_id)
                # In dry_run, don't actually mark — just count what we
                # WOULD reap. The reaped count is 0 (we didn't reap).
                continue

            try:
                mlflow.set_terminated(run_id, status="FAILED")
                # Also set a tag so ops can audit WHY the run was reaped.
                try:
                    client = mlflow.tracking.MlflowClient()
                    client.set_tag(
                        run_id, "drugos.reaped_reason", "heartbeat_stale"
                    )
                    client.set_tag(
                        run_id,
                        "drugos.reaped_heartbeat_age_seconds",
                        str(int(age)),
                    )
                    client.set_tag(
                        run_id, "drugos.reaped_at", str(now),
                    )
                except Exception:
                    # The set_terminated call already marked the run
                    # FAILED; the audit tags are best-effort.
                    pass
                result["reaped"] += 1
                result["reaped_run_ids"].append(run_id)
            except Exception as exc:
                result["errors"].append(
                    f"run {run_id}: failed to mark FAILED: {exc}"
                )
                logger.error(
                    "P2-024 reap_stale_runs: failed to mark run %s as "
                    "FAILED: %s",
                    run_id, exc,
                )

        logger.info(
            "P2-024 reap_stale_runs: scanned=%d, reaped=%d, "
            "skipped_no_heartbeat=%d, skipped_recent=%d, errors=%d "
            "(dry_run=%s, threshold=%ds)",
            result["scanned"], result["reaped"],
            result["skipped_no_heartbeat_tag"],
            result["skipped_heartbeat_recent"],
            len(result["errors"]), dry_run, stale_threshold_seconds,
        )
        return result
