"""P2-027 regression tests: setup_logging uses a named logger (not basicConfig).

Root fix: ``phase2/drugos_graph/utils.py`` now exposes a proper
``setup_logging()`` function that uses a NAMED logger
``drugos.phase2`` with a ``FileHandler`` (writing to
``${DRUGOS_LOG_DIR:-/var/log/drugos}/phase2.log``). This is immune
to Airflow's root-logger configuration (the bug P2-027 fixes).
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_PHASE2_ROOT = os.path.join(_REPO_ROOT, "phase2")
if _PHASE2_ROOT not in sys.path:
    sys.path.insert(0, _PHASE2_ROOT)


def test_p2_027_setup_logging_is_exported():
    """P2-027: ``setup_logging`` MUST be in ``utils.__all__``."""
    from drugos_graph import utils
    assert "setup_logging" in utils.__all__, (
        "P2-027 REGRESSION: setup_logging must be in utils.__all__"
    )
    assert callable(utils.setup_logging), (
        "P2-027: setup_logging must be callable"
    )


def test_p2_027_constants_are_exported():
    """P2-027: the logging constants MUST be exported."""
    from drugos_graph import utils
    for name in [
        "PHASE2_LOGGER_NAME",
        "PHASE2_DEFAULT_LOG_DIR",
        "PHASE2_DEFAULT_LOG_FILE",
        "PHASE2_LOG_FORMAT",
        "PHASE2_DATE_FORMAT",
    ]:
        assert name in utils.__all__, f"P2-027: {name} must be in __all__"
        assert hasattr(utils, name), f"P2-027: {name} must be defined"


def test_p2_027_setup_logging_returns_named_logger():
    """P2-027: ``setup_logging`` MUST return the ``drugos.phase2`` named
    logger (NOT the root logger)."""
    from drugos_graph.utils import setup_logging, PHASE2_LOGGER_NAME
    logger = setup_logging(attach_file=False)  # CI-safe: no file
    assert logger.name == PHASE2_LOGGER_NAME, (
        f"P2-027: setup_logging must return the named logger "
        f"'{PHASE2_LOGGER_NAME}', got '{logger.name}'. Named loggers "
        f"are immune to Airflow's root-logger config (the bug P2-027 "
        f"fixes). basicConfig mutates the ROOT logger, which Airflow "
        f"overrides."
    )
    # Clean up
    for h in list(logger.handlers):
        logger.removeHandler(h)


def test_p2_027_logger_does_not_propagate_to_root():
    """P2-027: the ``drugos.phase2`` logger MUST have ``propagate=False``
    so its records do NOT reach the root logger (which Airflow controls)."""
    from drugos_graph.utils import setup_logging, PHASE2_LOGGER_NAME
    logger = setup_logging(attach_file=False)
    assert logger.propagate is False, (
        "P2-027: the drugos.phase2 logger must have propagate=False so "
        "its records are NOT routed to the root logger (which Airflow "
        "controls). If propagate=True, Airflow's root handler would "
        "duplicate the records into Airflow's worker log -- the exact "
        "bug P2-027 fixes."
    )
    for h in list(logger.handlers):
        logger.removeHandler(h)


def test_p2_027_file_handler_writes_to_log_dir():
    """P2-027: when ``attach_file=True`` and the log dir is writable,
    a ``FileHandler`` MUST be attached that writes to
    ``${log_dir}/phase2.log``."""
    from drugos_graph.utils import setup_logging, PHASE2_LOGGER_NAME
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "phase2.log"
        assert not log_path.exists()
        logger = setup_logging(
            level="INFO",
            log_dir=tmpdir,
            log_file="phase2.log",
            attach_stream=False,
            attach_file=True,
        )
        # Log a unique message we can grep for
        marker = "P2_027_FILE_HANDLER_TEST_MARKER_42"
        logger.info(marker)
        # Flush all handlers
        for h in logger.handlers:
            h.flush()
        # The log file MUST exist and contain the marker
        assert log_path.exists(), (
            f"P2-027: log file {log_path} must exist after setup_logging "
            f"with attach_file=True"
        )
        content = log_path.read_text(encoding="utf-8")
        assert marker in content, (
            f"P2-027: marker {marker!r} not found in log file. The "
            f"FileHandler is not routing records to the file. Content:\n"
            f"{content}"
        )
        # Clean up
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass


def test_p2_027_setup_logging_is_idempotent():
    """P2-027: calling ``setup_logging`` multiple times MUST NOT add
    duplicate handlers (the function is idempotent -- it removes
    previously-attached handlers before adding new ones)."""
    from drugos_graph.utils import setup_logging
    logger = setup_logging(attach_file=False, attach_stream=True)
    first_count = len(logger.handlers)
    assert first_count >= 1, "P2-027: first call must add at least one handler"
    # Second call -- should NOT duplicate
    logger = setup_logging(attach_file=False, attach_stream=True)
    second_count = len(logger.handlers)
    assert second_count == first_count, (
        f"P2-027: setup_logging is not idempotent -- first call added "
        f"{first_count} handlers, second call left {second_count}. "
        f"Duplicate handlers cause duplicate log lines."
    )
    # Third call -- still no duplication
    logger = setup_logging(attach_file=False, attach_stream=True)
    third_count = len(logger.handlers)
    assert third_count == first_count
    # Clean up
    for h in list(logger.handlers):
        logger.removeHandler(h)


def test_p2_027_respects_log_level_env_var():
    """P2-027: ``DRUGOS_LOG_LEVEL`` env var MUST set the log level."""
    from drugos_graph.utils import setup_logging
    os.environ["DRUGOS_LOG_LEVEL"] = "WARNING"
    try:
        logger = setup_logging(attach_file=False)
        assert logger.level == logging.WARNING, (
            f"P2-027: DRUGOS_LOG_LEVEL=WARNING must set logger level to "
            f"WARNING, got {logging.getLevelName(logger.level)}"
        )
    finally:
        del os.environ["DRUGOS_LOG_LEVEL"]
        for h in list(logger.handlers):
            logger.removeHandler(h)


def test_p2_027_falls_back_to_stream_when_log_dir_not_writable():
    """P2-027: when the log directory is not writable (e.g. /var/log in
    CI), the function MUST skip the file handler and only attach the
    stream handler (so it doesn't crash in restricted environments)."""
    from drugos_graph.utils import setup_logging
    # Use a path that definitely cannot be created (under /proc which
    # is read-only on Linux)
    logger = setup_logging(
        log_dir="/proc/cannot_create_this_directory_drugos_p2_027",
        log_file="phase2.log",
        attach_stream=True,
        attach_file=True,
    )
    # The stream handler MUST be attached even if the file handler failed
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    assert has_stream, (
        "P2-027: stream handler must be attached even when the log dir "
        "is not writable (so the function works in CI/containers)."
    )
    # The function MUST NOT raise (graceful fallback)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


def test_p2_027_does_not_call_basic_config():
    """P2-027: ``setup_logging`` MUST NOT call ``logging.basicConfig``
    (which mutates the ROOT logger and is overridden by Airflow)."""
    # Mock basicConfig to detect any call
    original = logging.basicConfig
    called = {"count": 0}

    def mock_basicConfig(*args, **kwargs):
        called["count"] += 1
        return original(*args, **kwargs)

    logging.basicConfig = mock_basicConfig
    try:
        from drugos_graph.utils import setup_logging
        logger = setup_logging(attach_file=False)
        assert called["count"] == 0, (
            f"P2-027: setup_logging must NOT call logging.basicConfig "
            f"(it was called {called['count']} times). basicConfig "
            f"mutates the ROOT logger which Airflow overrides -- the "
            f"exact bug P2-027 fixes."
        )
        for h in list(logger.handlers):
            logger.removeHandler(h)
    finally:
        logging.basicConfig = original
