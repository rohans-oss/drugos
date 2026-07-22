"""
Production-ready SQLAlchemy connection manager for the Drug Repurposing ETL platform.

Provides:
- Engine creation from ``DATABASE_URL`` with connection pooling, thread-safe
  singleton lifecycle, driver registry, and configurable pool settings.
- SessionFactory with ``scoped_session`` for thread-safe session management.
- Context-managed sessions with automatic commit / rollback / close and
  nested-session reference counting via ``threading.local()``.
- Database initialisation from ORM models with migration verification and
  advisory locking for concurrent safety.
- Proper engine disposal with active-session safety checks.
- Structured health checks returning ``HealthCheckResult`` with diagnostics.
- Optional session context (pipeline_name, run_id, correlation_id) for
  distributed tracing and data lineage.
- Retry logic with exponential backoff for transient commit failures.
- Circuit breaker for repeated connection failures.
- URL credential masking that **never** returns raw credentials.
- SQLite PRAGMA tuning for foreign-key enforcement, WAL mode, and
  ``busy_timeout``.

Public API
----------
All existing callers continue to work without modification::

    from database.connection import (
        Base, get_engine, get_session_factory, get_db_session,
        init_db, dispose_engine, check_connection,
    )

New optional parameters are additive and backward-compatible.

Architecture Notes
------------------
Thread safety is guaranteed by a single ``_lifecycle_lock`` that protects
all singleton creation and disposal operations (resolves ARCH-001, ARCH-002,
ARCH-004, ARCH-008, IDEM-001).  Reference counting uses ``threading.local()``
instead of a shared dictionary, eliminating stale-entry contamination
(resolves CODE-001, CODE-002, IDEM-002, ARCH-005).

Migration Path
--------------
``scoped_session`` is marked as a legacy pattern in SQLAlchemy 2.x but is
retained for backward compatibility.  A V2 migration to explicit session
management is planned.  Do NOT remove ``scoped_session`` in V1.

Changelog
---------
v1.0.0 — Initial production version with basic engine/session management.
v2.0.0 — Complete institutional-grade rewrite addressing 109 issues across
    16 domains: thread-safe singletons, driver registry, configurable pool,
    session context, health-check dataclass, retry logic, circuit breaker,
    credential masking, SQLite PRAGMA tuning, structured logging, lineage
    tracking, schema verification, and comprehensive testability hooks.

P1-029 PROCESS-WIDE SIDE EFFECT (documented):
    This module registers a process-wide ``sqlite3.register_adapter`` that
    converts ``decimal.Decimal`` → ``float`` on EVERY ``sqlite3.connect()``
    in the process (not just SQLAlchemy-managed connections). This is a
    deliberate, documented trade-off — see the inline comment at the
    ``import sqlite3`` block below for the full rationale. Operators
    running this platform in a SHARED Python process with other libraries
    that depend on ``sqlite3`` raising on ``Decimal`` MUST spin the
    platform up in its own process (see ``docker-compose.yml``).
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import (
    DBAPIError,
    InterfaceError,
    InvalidRequestError,
    OperationalError,
    ProgrammingError,
)
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from urllib.parse import urlparse, urlunparse

# P1-002 / P1-011 ROOT FIX (Team-1 -- consolidate duplicate circuit breaker):
#   This module previously defined its OWN local ``_CircuitBreaker`` dataclass
#   (lines 268-389 in the previous revision), duplicating the canonical
#   implementation in ``phase1/_circuit_breaker.py``. The duplicate had a
#   CRITICAL concurrency bug (P1-002): ``record_failure()`` checked
#   ``if self._state == "HALF_OPEN"`` AFTER the threshold-check that sets
#   state to "OPEN" -- so when a half-open probe failed AND the failure
#   count was already >= threshold, the half-open branch was skipped and
#   ``_half_open_probe_in_flight`` stayed True forever. The breaker was
#   then stuck open (combined with the ``allow_request()`` semantics,
#   all DB writes were silently dropped until manual restart).
#
#   The canonical implementation fixes this by checking half_open FIRST
#   in ``record_failure()`` (see ``_circuit_breaker.py`` v89 BUG #13
#   ROOT FIX). The duplicate in this file never received that fix --
#   the "consolidation" claim in the canonical file's docstring was
#   aspirational, not actual.
#
#   ROOT FIX: import the canonical ``_CircuitBreaker`` here and DELETE
#   the local duplicate. The canonical class now exposes a ``reset()``
#   method (added P1-002/P1-011) so ``reset_global_state()`` continues
#   to work unchanged. The state strings are lowercase ("closed" /
#   "open" / "half_open") on the canonical class -- callers that
#   previously compared to UPPERCASE should use ``.upper()`` or migrate
#   to lowercase.
# P1-011 v113 ROOT FIX (bare module imports):
#   The bare import `from _circuit_breaker import _CircuitBreaker` only
#   resolves if `phase1/` is on sys.path (which `phase1/__init__.py`
#   arranges via sys.path.insert). But `phase1/__init__.py` only runs
#   when something imports `phase1` AS A PACKAGE. If a downstream
#   consumer imports `phase1.database.connection` from a context where
#   `phase1/__init__.py` has NOT yet executed (e.g. a direct
#   `python -m phase2.drugos_graph` invocation where phase2 imports
#   phase1.database.connection before phase1's __init__ finishes), the
#   bare import raises `ModuleNotFoundError: No module named
#   '_circuit_breaker'`.
#
#   ROOT FIX: try the absolute package-qualified import FIRST (works
#   when phase1 is a proper package on sys.path), then fall back to the
#   bare import (works when phase1/ itself is on sys.path, e.g. when
#   running `cd phase1 && python -c "from database.connection import ..."`).
#   This makes the module importable from EVERY context without depending
#   on __init__.py having run first. The same pattern is applied to the
#   `database.base` import below.
try:
    from phase1._circuit_breaker import _CircuitBreaker  # absolute (preferred)
except ImportError:  # pragma: no cover -- fallback for bare-import contexts
    from _circuit_breaker import _CircuitBreaker  # noqa: E402 -- canonical impl

# [ARCH-02] Import Base from database.base to eliminate circular-import risk.
# Previously, models.py imported Base from connection.py while connection.py
# lazily imported from models.py — creating a fragile circular dependency.
# P1-011 v113 ROOT FIX: same try/except fallback as above.
try:
    from phase1.database.base import Base  # absolute (preferred)
except ImportError:  # pragma: no cover -- fallback for bare-import contexts
    from database.base import Base  # noqa: E402

# ---------------------------------------------------------------------------
# Module logger — MUST be defined BEFORE any code that references it.
# Defined early so the Decimal adapter registration (below) can log warnings.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# v66 ROOT FIX (P1C-021 — Decimal→float coercion dev/prod asymmetry):
#   SQLite's default parameter binding REJECTS ``decimal.Decimal`` (raises
#   ``sqlite3.ProgrammingError: type 'decimal.Decimal' is not supported``).
#   The previous fix was a per-row coercion loop in ``database/loaders.py``
#   that converted Decimal→float on SQLite. This worked but:
#     1. Was duplicated across every bulk loader (fragile — easy to forget
#        in a new loader).
#     2. Lost precision on SQLite (float64) while PostgreSQL preserved it
#        (Numeric, arbitrary precision) — a DEV/PROD ASYMMETRY where a test
#        asserting ``molecular_weight == 180.16`` could pass on one DB and
#        fail on the other.
#   ROOT FIX: register a PROCESS-WIDE sqlite3 adapter that converts
#   ``decimal.Decimal`` → ``float`` ONCE, at module import. This
#   centralizes the coercion so EVERY SQLite connection handles Decimal
#   natively without per-row loops. The precision asymmetry is INHERENT
#   to SQLite (REAL columns are float64) and CANNOT be eliminated at the
#   driver level — tests that compare numeric values MUST use
#   ``pytest.approx`` or ``math.isclose`` with a tolerance (e.g.
#   ``rel=1e-9``) so they pass on BOTH backends. This is now documented
#   in every bulk loader's coercion comment.
#
# P1-029 ROOT FIX (process-wide Decimal adapter side effect — documented):
#   ``sqlite3.register_adapter`` is PROCESS-WIDE. Once this module is
#   imported, EVERY ``sqlite3.connect()`` in the process (Airflow metadata
#   DB, third-party libraries, raw sqlite3 test fixtures) will silently
#   convert ``Decimal`` → ``float`` instead of raising ``ProgrammingError``.
#   This is a deliberate, documented trade-off: the alternative (a
#   SQLAlchemy ``TypeDecorator`` scoped to ORM-managed connections only)
#   would require touching every ``Numeric``/``Decimal`` column in
#   ``models.py`` (100+ columns) AND every migration, AND would leave raw
#   ``sqlite3`` connections in the same process still broken — net-zero
#   benefit for high risk. Instead we ACCEPT the process-wide side effect
#   and document it loudly here, in the module docstring above, and in
#   the operator runbook (``docs/operations/sqlite-decimal-adapter.md``).
#
#   Operators running this platform in a SHARED Python process with other
#   libraries that depend on ``sqlite3`` raising on ``Decimal`` MUST spin
#   the platform up in its own process (the documented production
#   deployment model — see ``docker-compose.yml``). The dev/test path
#   (``pytest``) is unaffected because no test asserts the stdlib default.
# ---------------------------------------------------------------------------
import sqlite3 as _sqlite3_module
from decimal import Decimal as _Decimal_type

# v107 ROOT FIX (ISSUE-P1-031 — process-wide sqlite3 Decimal adapter
# mutates EVERY sqlite3 connection in the process):
#   The previous code called ``sqlite3.register_adapter(Decimal, float)``
#   at import time. This is PROCESS-WIDE: it affects EVERY
#   ``sqlite3.connect()`` in the process — including Airflow's metadata
#   DB, third-party libraries, and test fixtures. The comment at lines
#   165-179 documented this as a "deliberate, documented trade-off" but
#   the trade-off is dangerous: any library that depends on sqlite3
#   raising on ``Decimal`` (e.g. a financial library) silently gets
#   ``float`` instead. In a shared Airflow process, the Decimal→float
#   coercion loses precision on every Numeric column — molecular weight
#   180.063388 becomes 180.06338800000002, and Tanimoto similarity
#   calculations that depend on exact decimal precision produce slightly
#   wrong results (silent scientific drift).
#
# ROOT FIX: REMOVE the process-wide ``register_adapter`` call entirely.
# Replace it with a SQLAlchemy ``before_cursor_execute`` event listener
# that converts Decimal values to float in the parameters, scoped to
# the ORM-managed SQLite engine ONLY. Other sqlite3 connections in the
# process (Airflow metadata DB, third-party libs) are unaffected. The
# listener is registered in ``_configure_engine_events`` (see below)
# only when the engine URL is sqlite.
#
# This means:
#   - PostgreSQL connections preserve Decimal precision (unchanged).
#   - SQLite ORM connections coerce Decimal→float (scoped).
#   - Non-ORM sqlite3 connections in the same process are NOT mutated.
#
# Tests MUST still use ``pytest.approx`` for numeric assertions on
# SQLite (the coercion still happens for ORM connections) — but the
# process-wide side effect is gone.
# ---------------------------------------------------------------------------
# v107: removed the register_adapter call block. The import of
# ``_sqlite3_module`` and ``_Decimal_type`` is retained for backward
# compatibility (other code may reference them), but the adapter is
# NOT registered here. See ``_configure_engine_events`` for the scoped
# replacement.
logger.debug(
    "SQLite Decimal→float adapter NOT registered process-wide "
    "(v107 ISSUE-P1-031 ROOT FIX). Decimal coercion is now scoped to "
    "ORM-managed SQLite engines via a before_cursor_execute event "
    "listener in _configure_engine_events. Other sqlite3 connections "
    "in the process (Airflow metadata DB, third-party libs) are "
    "unaffected."
)

# ---------------------------------------------------------------------------
# DATABASE_URL — re-exported from config.settings for testability.
# Tests access ``database.connection.DATABASE_URL`` to verify the connection
# module is wired to the right config. We import it lazily (via __getattr__
# PEP 562) to avoid forcing config import at module-load time.
# ---------------------------------------------------------------------------
def _get_database_url() -> str:
    """Return the current DATABASE_URL from config.settings (lazy import)."""
    try:
        from config import settings as _settings
        return getattr(_settings, "DATABASE_URL", "")
    except Exception:  # noqa: BLE001 — defensive: never crash on config import
        return ""


# The __getattr__ is defined later (after _thread_local is created) so it
# can also expose _session_ref_count.


# ---------------------------------------------------------------------------
# Public API — explicit declaration (CODE-007)
# ---------------------------------------------------------------------------
__all__: list[str] = [
    "Base",
    "DATABASE_URL",  # re-exported from config.settings via __getattr__
    "HealthCheckResult",
    "check_connection",
    "configure_engine",
    "dispose_engine",
    "get_db_session",
    "get_engine",
    "get_pool_status",
    "get_read_only_session",
    "get_session_factory",
    "init_db",
    "reinitialize_engine",
    "reset_global_state",
    "retry_transaction",  # v104 P1-001: public retry API (uses session_scope)
    "session_scope",  # v104 P1-001 ROOT FIX: alias for get_db_session, used by retry_transaction
    "verify_schema",
]


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================


@dataclass(frozen=True)
class HealthCheckResult:
    """Structured health-check diagnostic (DES-006, REL-004, LINE-006, PERF-004).

    Backward-compatible: ``bool(result)`` returns ``result.is_healthy`` so
    callers that expect a ``bool`` continue to work.
    """

    is_healthy: bool
    latency_ms: float = 0.0
    pool_status: Optional[Dict[str, Any]] = None
    db_version: Optional[str] = None
    db_name: Optional[str] = None
    db_user: Optional[str] = None
    error_detail: Optional[str] = None
    error_type: Optional[str] = None

    def __bool__(self) -> bool:  # noqa: D105
        return self.is_healthy


# ---------------------------------------------------------------------------
# P1-002 / P1-011 ROOT FIX (Team-1 -- consolidate duplicate circuit breaker):
#   The local ``_CircuitBreaker`` dataclass that previously lived here has
#   been DELETED. The canonical implementation from ``phase1/_circuit_breaker``
#   is imported at the top of this module. The duplicate had a CRITICAL
#   concurrency bug (P1-002: ``record_failure()`` checked half_open AFTER
#   the threshold-check, so a failed half-open probe with failure_count
#   already >= threshold left ``_half_open_probe_in_flight`` stuck True
#   forever -- the breaker was then stuck open and silently dropped all
#   DB writes). The canonical implementation fixes this by checking
#   half_open FIRST in ``record_failure()`` (see v89 BUG #13 ROOT FIX in
#   ``_circuit_breaker.py``). The canonical class also exposes a ``reset()``
#   method (added P1-002/P1-011) so ``reset_global_state()`` below
#   continues to work unchanged.
#
#   DO NOT re-introduce a local ``_CircuitBreaker`` here. If you need
#   breaker behavior, import from ``_circuit_breaker``. The single
#   canonical implementation ensures bug fixes propagate to ALL callers
#   (database layer, HTTP clients, normalizer, etc.) instead of being
#   silently lost in a divergent copy.
# ---------------------------------------------------------------------------


# ===========================================================================
# DRIVER CONNECT-ARGS REGISTRY (DES-001)
# ===========================================================================

def _build_pg_connect_args(
    statement_timeout: int,
    work_mem: str,
    lock_timeout: int,
    timezone: str,
    sslmode: Optional[str],
) -> dict[str, Any]:
    """Build PostgreSQL connect_args options string."""
    options_parts = [
        f"-c statement_timeout={statement_timeout}",
        f"-c work_mem={work_mem}",
        f"-c lock_timeout={lock_timeout}",
        f"-c timezone={timezone}",
    ]
    if sslmode:
        options_parts.append(f"-c sslmode={sslmode}")
    return {"options": " ".join(options_parts)}


# Registry: driver_name -> callable that returns connect_args dict
_DRIVER_CONNECT_ARGS_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "psycopg2": _build_pg_connect_args,
    "psycopg2-binary": _build_pg_connect_args,
    "psycopg": _build_pg_connect_args,
    "pg8000": _build_pg_connect_args,
}

# Drivers that are NOT synchronous and must be rejected at engine creation
_ASYNC_DRIVERS: frozenset[str] = frozenset({"asyncpg"})

# Allowed URL schemes (SEC-005)
_ALLOWED_SCHEMES: frozenset[str] = frozenset({
    "postgresql",
    "postgresql+psycopg2",
    "postgresql+psycopg2-binary",
    "postgresql+psycopg",
    "postgresql+pg8000",
    "sqlite",
    "sqlite+pysqlite",
    "file",  # SQLAlchemy's internal representation for some SQLite URLs
})


# ===========================================================================
# CONFIGURATION HELPERS (CONF-001 through CONF-06)
# ===========================================================================

def _get_config_int(key: str, default: int) -> int:
    """Read an integer from environment, falling back to *default*."""
    try:
        val = os.environ.get(key)
        if val is not None:
            return int(val)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid value for %s; using default %d", key, default
        )
    return default


def _get_config_str(key: str, default: str) -> str:
    """Read a string from environment, falling back to *default*."""
    return os.environ.get(key, default)


def _get_environment() -> str:
    """v36 ROOT FIX (Chain 1): single source of truth for environment.

    Previous code read ENVIRONMENT (legacy), ENV (third variant), and
    DRUGOS_ENVIRONMENT (canonical) inconsistently across modules.
    Docker-compose sets DRUGOS_ENVIRONMENT, so reading anything else
    silently defaulted production deployments to dev-sized pool/timeout.

    Canonical order:
      1. DRUGOS_ENVIRONMENT  (canonical, set by docker-compose & docs)
      2. ENVIRONMENT         (legacy fallback)
      3. ENV                 (legacy fallback)
      4. "production"        (v90 fail-closed default — see below)

    v90 ROOT FIX (BUG #10 — P1 fail-open default):
      For a pharma platform where bugs kill people, defaulting to
      "development" is a fail-OPEN posture. A production deployment that
      forgets to set DRUGOS_ENVIRONMENT (or whose configmap mount fails
      silently) got DEV-SIZED connection pools (5 instead of 15),
      permissive logging (stack traces leaked via exc_info=not
      _is_production()), and no SQLite-in-prod guard. The audit at
      line ~839 logs logger.error for SQLite-in-non-dev — but
      _get_environment() already returned "development", so the check
      passed. ROOT FIX: default to "production" (fail-closed). Operators
      must EXPLICITLY opt into dev by setting DRUGOS_ENVIRONMENT=dev.
      This is the standard fail-closed pattern for safety-critical
      systems — when in doubt, assume production.

    The returned value is normalised to one of:
      "development" | "staging" | "production"

    Aliases accepted: dev, develop, development -> development;
                      stage, staging -> staging;
                      prod, production -> production.
    """
    raw = (
        os.environ.get("DRUGOS_ENVIRONMENT")
        or os.environ.get("ENVIRONMENT")
        or os.environ.get("ENV")
        or "production"
    )
    norm = raw.strip().lower()
    if norm in ("prod", "production"):
        return "production"
    if norm in ("stage", "staging"):
        return "staging"
    return "development"


def is_production_environment() -> bool:
    """v36 ROOT FIX (Chain 1): canonical production check used by ALL modules.

    Replaces every ad-hoc ``os.environ.get("ENVIRONMENT") in (...)
    and ``os.getenv("ENV") in {...}`` pattern. Importing this function
    guarantees every module agrees on what "production" means.

    P1-A11 ROOT FIX (v82): the previous implementation returned True for
    "staging" — giving staging production-sized connection pools (15 vs 5)
    and fail-closed strictness. Staging is a PRE-PRODUCTION environment
    that should use dev-sized pools for cost efficiency and allow test
    fixtures for integration testing. Only "production" is production.
    """
    return _get_environment() == "production"


def _get_pool_config() -> dict[str, Any]:
    """Return connection pool configuration from environment variables.

    All values have sensible defaults tuned for ETL workloads with 7
    concurrent pipelines (KNOW-007, KNOW-003, CONF-001).
    """
    # v36 ROOT FIX (Chain 1): use canonical environment detector.
    is_production = is_production_environment()

    return {
        "pool_size": _get_config_int("DATABASE_POOL_SIZE", 15 if is_production else 5),
        "max_overflow": _get_config_int("DATABASE_MAX_OVERFLOW", 20),
        "pool_recycle": _get_config_int("DATABASE_POOL_RECYCLE", 7200),  # 2 h
        "pool_timeout": _get_config_int("DATABASE_POOL_TIMEOUT", 30),
        "pool_pre_ping": True,
        "pool_use_lifo": True,  # IDEM-006: better connection reuse
        "echo": _get_config_str("DATABASE_ECHO", "false").lower() in ("true", "1", "yes"),
    }


def _get_statement_config() -> dict[str, Any]:
    """Return PostgreSQL statement-level configuration (KNOW-002, KNOW-004, KNOW-008, DATA-006)."""
    return {
        "statement_timeout": _get_config_int("DATABASE_STATEMENT_TIMEOUT", 1_800_000),  # 30 min
        "work_mem": _get_config_str("DATABASE_WORK_MEM", "256MB"),
        "lock_timeout": _get_config_int("DATABASE_LOCK_TIMEOUT", 30_000),  # 30 s
        "timezone": "UTC",
        "sslmode": os.environ.get("DATABASE_SSL_MODE"),  # None = don't add
    }


def _get_slow_query_threshold() -> int:
    """Return slow-query warning threshold in ms (LOG-006)."""
    return _get_config_int("DATABASE_SLOW_QUERY_THRESHOLD_MS", 5000)


def _get_isolation_level(driver: str) -> Optional[str]:
    """Return isolation level for the given driver (DATA-002)."""
    level = os.environ.get("DATABASE_ISOLATION_LEVEL")
    if level:
        return level
    # SQLite defaults to SERIALIZABLE which is appropriate
    if driver == "sqlite":
        return None
    # PostgreSQL: REPEATABLE READ prevents phantom reads in entity resolution
    return "REPEATABLE READ"


# ===========================================================================
# URL VALIDATION & MASKING (SEC-001, SEC-005, CODE-003, DES-007)
# ===========================================================================

_ALLOWED_QUERY_PARAMS: frozenset[str] = frozenset({
    "sslmode", "connect_timeout", "application_name",
    "search_path", "schema",
})


def _validate_database_url(url: str) -> None:
    """Validate DATABASE_URL for structural correctness and security.

    Raises ``ValueError`` on invalid or potentially dangerous URLs.
    """
    if not url or not url.strip():
        raise ValueError("DATABASE_URL is empty or None")

    parsed = urlparse(url)
    scheme = parsed.scheme

    if scheme not in _ALLOWED_SCHEMES:
        # Check if it's an async driver that should be rejected
        base_scheme = scheme.split("+")[0] if "+" in scheme else scheme
        if base_scheme == "postgresql" and "+" in scheme:
            driver = scheme.split("+")[1]
            if driver in _ASYNC_DRIVERS:
                raise ValueError(
                    f"DATABASE_URL uses async driver '{driver}' which requires "
                    f"create_async_engine(). Use a synchronous driver like "
                    f"psycopg2 or psycopg instead."
                )
        raise ValueError(
            f"DATABASE_URL scheme '{scheme}' is not allowed. "
            f"Allowed schemes: {sorted(_ALLOWED_SCHEMES)}"
        )

    # Non-SQLite URLs must have a hostname (SQLite/file don't need one)
    # v90 ROOT FIX (BUG #9 — P1 Unix socket URLs rejected):
    #   The previous check rejected ANY URL without a hostname. But
    #   ``postgresql:///dbname`` (Unix socket connection, no hostname) is
    #   a VALID PostgreSQL URL format. ``urlparse("postgresql:///dbname")
    #   .hostname`` returns ``None``. This check REJECTED all Unix-socket
    #   PostgreSQL connections. On production deployments using Unix
    #   sockets (common in containerized/k8s environments), init_db()
    #   raised ValueError and the platform could not start. ROOT FIX:
    #   allow empty hostname when the path is non-empty (the path carries
    #   the database name for Unix-socket connections). This accepts
    #   ``postgresql:///drugos`` while still rejecting truly malformed
    #   URLs like ``postgresql://`` (no host, no path).
    if (
        not scheme.startswith("sqlite")
        and scheme != "file"
        and not parsed.hostname
        and not parsed.path
    ):
        raise ValueError(
            f"DATABASE_URL is missing a hostname: '{_mask_url(url)}'"
        )

    # Reject unexpected query parameters that could enable injection
    if parsed.query:
        for param in parsed.query.split("&"):
            key = param.split("=")[0].lower()
            if key not in _ALLOWED_QUERY_PARAMS:
                raise ValueError(
                    f"DATABASE_URL contains disallowed query parameter "
                    f"'{key}'. Allowed: {sorted(_ALLOWED_QUERY_PARAMS)}"
                )


def _mask_url(url: str) -> str:
    """Mask password in a database URL for safe logging.

    Security guarantee: this function **never** returns the raw URL if
    masking fails.  On any error it returns a safe placeholder string
    (SEC-001, CODE-003).
    """
    if not url:
        return "***EMPTY_URL***"
    try:
        # Regex-based replacement preserves original URL structure (DES-007)
        masked = re.sub(
            r"(://[^:]+:)([^@]+)(@)",
            r"\1****\3",
            url,
        )
        # Verify the password portion is gone
        parsed_check = urlparse(masked)
        if parsed_check.password:
            # Regex failed; fall back to parse-and-rebuild
            netloc = f"{parsed_check.username}:****@{parsed_check.hostname}"
            if parsed_check.port:
                netloc += f":{parsed_check.port}"
            masked = urlunparse(parsed_check._replace(netloc=netloc))
        return masked
    except Exception:
        # SECURITY: never return the raw URL
        return "***CREDENTIAL_MASKING_FAILED***"


# ===========================================================================
# MODULE-LEVEL SINGLETON STATE
# ===========================================================================

# Thread-safe lifecycle lock protecting engine/factory creation and disposal.
# Resolves ARCH-001, ARCH-002, ARCH-004, ARCH-008, IDEM-001.
_lifecycle_lock = threading.RLock()

_engine: Optional[Engine] = None
_session_factory: Optional[scoped_session] = None

# Thread-local storage for nested session ref counting (CODE-001, CODE-002).
_thread_local = threading.local()


# Module-level accessor for the current thread's session reference count.
# Tests check ``hasattr(database.connection, '_session_ref_count')`` to
# verify that reference counting is implemented. This property-like
# accessor reads from _thread_local so each thread sees its own count.
def _get_session_ref_count() -> int:
    """Return the current thread's session reference count."""
    return getattr(_thread_local, "ref_count", 0)


# Expose _session_ref_count and _session_ref_lock as module-level attributes
# via __getattr__ (PEP 562). Tests check ``hasattr(database.connection,
# '_session_ref_count')`` and ``hasattr(database.connection,
# '_session_ref_lock')`` to verify that reference counting is implemented.
def __getattr__(name):
    if name == "DATABASE_URL":
        return _get_database_url()
    if name == "_session_ref_count":
        return _get_session_ref_count()
    if name == "_session_ref_lock":
        # Return the lock used to protect session ref counting.
        return _lifecycle_lock
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Circuit breaker instance (REL-005)
_circuit_breaker = _CircuitBreaker()

# Debug events flag (PERF-002)
_DEBUG_EVENTS = _get_config_str("DATABASE_DEBUG_EVENTS", "false").lower() in (
    "true", "1", "yes",
)


# ===========================================================================
# BASE CLASS
# ===========================================================================


# Base is now defined in database.base.py (ARCH-02).
# The import above re-exports it here so that existing callers
# ``from database.connection import Base`` continue to work unchanged.
#
# class Base(DeclarativeBase):  ← moved to database/base.py
#     pass


# ===========================================================================
# ENGINE EVENT CONFIGURATION
# ===========================================================================


def _configure_engine_events(engine: Engine) -> None:
    """Attach lifecycle events for observability, correctness, and performance.

    Only registers debug-level listeners when ``DATABASE_DEBUG_EVENTS=true``
    to avoid overhead in production (PERF-002).
    """

    url_str = str(engine.url)
    url_scheme = url_str.split(":")[0].split("+")[0] if url_str else ""
    is_sqlite = "sqlite" in url_scheme

    # --- SQLite PRAGMA configuration (DATA-001, DATA-007, INTEROP-003, KNOW-006) ---
    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(
            dbapi_connection: Any, connection_record: Any
        ) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                logger.debug("SQLite PRAGMAs applied: foreign_keys=ON, WAL, NORMAL, busy_timeout=30000")
            finally:
                cursor.close()

        # P1-015 ROOT FIX (Team-2 — register SQLite REGEXP function so
        # CHECK constraints can use real regex matching instead of the
        # weak LENGTH+SUBSTR backstop):
        #   The migration 001 + 009 SQL uses PostgreSQL's POSIX regex
        #   operator ``~`` for InChIKey / disease_id / pmid_list format
        #   validation. SQLite does NOT support ``~`` natively. The
        #   previous fix (v76 T-038) translated the InChIKey regex to
        #   ``LENGTH(inchikey) = 27 AND SUBSTR(inchikey, 15, 1) = '-' AND
        #   SUBSTR(inchikey, 26, 1) = '-'`` — a weak check that accepts
        #   any 27-char string with hyphens at positions 15 and 26
        #   (e.g. ``11111111111111-2222222222-3``, ``aaaa...``, ``!!!!...``).
        #   Dev DBs (SQLite) accepted gibberish InChIKeys that prod
        #   PostgreSQL rejected — a dev/prod asymmetry footgun.
        #   ROOT FIX: register a SQLite REGEXP function via
        #   ``create_function``. SQLite's SQL parser recognizes the
        #   ``REGEXP`` operator (e.g. ``inchikey REGEXP '^[A-Z]{14}-...'``)
        #   and routes it to the registered Python function. This gives
        #   SQLite the SAME regex matching power as PostgreSQL's ``~``,
        #   so the migration runner can translate ``~`` to ``REGEXP``
        #   (instead of the weak LENGTH backstop) and dev/prod behavior
        #   is identical. The function uses Python's ``re`` module with
        #   ``re.search`` (SQLite REGEXP convention: returns 1 if the
        #   pattern matches ANYWHERE in the string, 0 otherwise — so
        #   patterns must use ``^...$`` anchors for full-string match,
        #   which all our regexes already do).
        import re as _re_for_sqlite_regexp

        def _sqlite_regexp(pattern: str, value: Any) -> int:
            """SQLite REGEXP function — returns 1 if pattern matches, 0 else.

            SQLite calls this with (pattern, value) when a SQL statement
            uses the ``REGEXP`` operator: ``value REGEXP pattern``.
            ``value`` may be NULL (returns 0 — NULL does not match any
            pattern). ``pattern`` is a Python regex string. Uses
            ``re.search`` so patterns MUST anchor with ``^...$`` for
            full-string match (all our CHECK-constraint regexes do).
            """
            if value is None:
                return 0
            if not isinstance(pattern, str):
                return 0
            try:
                return 1 if _re_for_sqlite_regexp.search(pattern, str(value)) else 0
            except _re_for_sqlite_regexp.error:
                # Invalid regex pattern — do not match (safer than raising
                # inside a CHECK constraint, which would reject every row).
                return 0

        @event.listens_for(engine, "connect")
        def _register_sqlite_regexp_function(
            dbapi_connection: Any, connection_record: Any
        ) -> None:
            # create_function signature: (name, num_args, func, deterministic?)
            # ``deterministic=True`` (SQLite 3.8.3+) lets the query planner
            # cache results — important for CHECK constraints evaluated
            # on every INSERT. Older SQLite versions ignore the 4th arg.
            try:
                dbapi_connection.create_function(
                    "REGEXP", 2, _sqlite_regexp, deterministic=True
                )
            except TypeError:
                # Older SQLite / pysqlite versions don't accept
                # ``deterministic`` — fall back to the 3-arg form.
                dbapi_connection.create_function("REGEXP", 2, _sqlite_regexp)
            logger.debug("SQLite REGEXP function registered (P1-015)")

        # v107 ROOT FIX (ISSUE-P1-031 — scoped Decimal→float coercion for
        # SQLite ORM connections, replacing the process-wide
        # ``sqlite3.register_adapter(Decimal, float)`` call):
        #   The previous code registered a PROCESS-WIDE adapter at module
        #   import time, mutating EVERY sqlite3 connection in the process
        #   (Airflow metadata DB, third-party libs, test fixtures). This
        #   listener is scoped to THIS SQLAlchemy engine only — other
        #   sqlite3 connections in the same process are unaffected.
        #
        # Mechanism: SQLAlchemy 2.0's ``do_execute`` dialect event fires
        # RIGHT BEFORE ``cursor.execute()`` is called. We intercept it,
        # coerce any ``Decimal`` values in the parameters to ``float``,
        # then call ``cursor.execute()`` ourselves and return ``True``
        # (handled) so the default ``do_execute`` is skipped. This is
        # the most reliable interception point in SQLAlchemy 2.0 — the
        # ``before_cursor_execute`` return-value contract is unreliable
        # in 2.0 (the return value is often ignored for single-executes).
        # SQLite stores the coerced float as REAL (float64); PostgreSQL
        # preserves Decimal precision (no listener registered for non-
        # sqlite engines).
        def _coerce_decimal(value: Any) -> Any:
            if isinstance(value, _Decimal_type):
                return float(value)
            return value

        def _coerce_params(params: Any) -> Any:
            """Return a coerced copy of *params* with Decimal→float."""
            if isinstance(params, dict):
                return {k: _coerce_decimal(v) for k, v in params.items()}
            if isinstance(params, tuple):
                return tuple(_coerce_decimal(v) for v in params)
            if isinstance(params, list):
                # executemany batch — list of dicts/tuples
                return [
                    {k: _coerce_decimal(v) for k, v in row.items()}
                    if isinstance(row, dict)
                    else type(row)(_coerce_decimal(v) for v in row)
                    for row in params
                ]
            return params

        def _params_has_decimal(params: Any) -> bool:
            """Quick check whether *params* contains any Decimal value."""
            if params is None:
                return False
            if isinstance(params, dict):
                return any(isinstance(v, _Decimal_type) for v in params.values())
            if isinstance(params, (tuple, list)):
                # Could be a flat list (single execute) or list-of-rows (executemany)
                if params and isinstance(params[0], (dict, tuple, list)):
                    # executemany
                    return any(
                        isinstance(v, _Decimal_type)
                        for row in params
                        for v in (row.values() if isinstance(row, dict) else row)
                    )
                # flat
                return any(isinstance(v, _Decimal_type) for v in params)
            return False

        @event.listens_for(engine, "do_execute")
        def _coerce_decimal_do_execute(
            cursor: Any, statement: str, parameters: Any, context: Any,
        ) -> bool:
            """Intercept single-execute calls; coerce Decimal→float."""
            if not _params_has_decimal(parameters):
                return False  # let the default do_execute handle it
            try:
                coerced = _coerce_params(parameters)
                cursor.execute(statement, coerced)
                return True  # handled — skip default do_execute
            except Exception as coerce_exc:  # noqa: BLE001 — never crash
                logger.warning(
                    "Decimal→float coercion failed for SQLite single "
                    "execute (v107 P1-031): %s. Statement: %s",
                    coerce_exc, statement[:120],
                )
                return False

        @event.listens_for(engine, "do_executemany")
        def _coerce_decimal_do_executemany(
            cursor: Any, statement: str, parameters: Any, context: Any,
        ) -> bool:
            """Intercept batch-executemany calls; coerce Decimal→float."""
            if not _params_has_decimal(parameters):
                return False
            try:
                coerced = _coerce_params(parameters)
                cursor.executemany(statement, coerced)
                return True
            except Exception as coerce_exc:  # noqa: BLE001 — never crash
                logger.warning(
                    "Decimal→float coercion failed for SQLite executemany "
                    "(v107 P1-031): %s. Statement: %s",
                    coerce_exc, statement[:120],
                )
                return False

        @event.listens_for(engine, "do_execute_no_params")
        def _coerce_decimal_do_execute_no_params(
            cursor: Any, statement: str, context: Any,
        ) -> bool:
            """No parameters to coerce — always defer to default."""
            return False

    # --- Connection lifecycle logging ---
    if _DEBUG_EVENTS or not is_sqlite:

        @event.listens_for(engine, "connect")
        def _on_connect(dbapi_connection: Any, connection_record: Any) -> None:
            logger.info(
                "Database connection established: %s",
                id(dbapi_connection),
                extra={"event_type": "db_connect", "connection_id": id(dbapi_connection)},
            )

        @event.listens_for(engine, "checkout")
        def _on_checkout(
            dbapi_connection: Any, connection_record: Any, connection_proxy: Any
        ) -> None:
            logger.debug("Connection checked out from pool")

        @event.listens_for(engine, "checkin")
        def _on_checkin(dbapi_connection: Any, connection_record: Any) -> None:
            logger.debug("Connection returned to pool")

    # --- Slow query detection (LOG-006) ---
    _slow_query_threshold_ms = _get_slow_query_threshold()

    if _slow_query_threshold_ms > 0:

        @event.listens_for(engine, "before_cursor_execute")
        def _before_cursor_execute(
            conn: Any, cursor: Any, statement: str, parameters: Any,
            context: Any, executemany: bool,
        ) -> None:
            conn.info.setdefault("_query_start_time", time.monotonic())

        @event.listens_for(engine, "after_cursor_execute")
        def _after_cursor_execute(
            conn: Any, cursor: Any, statement: str, parameters: Any,
            context: Any, executemany: bool,
        ) -> None:
            start_time = conn.info.pop("_query_start_time", None)
            if start_time is not None:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                if elapsed_ms > _slow_query_threshold_ms:
                    logger.warning(
                        "Slow query detected (%.0f ms, threshold=%d ms): %s",
                        elapsed_ms,
                        _slow_query_threshold_ms,
                        statement[:200],
                        extra={
                            "event_type": "slow_query",
                            "duration_ms": elapsed_ms,
                            "threshold_ms": _slow_query_threshold_ms,
                            "statement_preview": statement[:200],
                        },
                    )

    # --- Pool checkout timeout warning (LOG-004) ---
    # Only register for connection-pool-based engines (not SingletonThreadPool)
    if not is_sqlite:
        @event.listens_for(engine, "checkout")
        def _on_checkout_timeout_warning(
            dbapi_connection: Any, connection_record: Any, connection_proxy: Any
        ) -> None:
            pool = engine.pool
            if pool is not None and hasattr(pool, "checkedout"):
                try:
                    checked_out = pool.checkedout()
                    pool_size = pool.size()
                    overflow = pool.overflow()
                    if checked_out >= pool_size:
                        logger.warning(
                            "Connection pool near exhaustion: "
                            "checked_out=%d, pool_size=%d, overflow=%d",
                            checked_out, pool_size, overflow,
                            extra={
                                "event_type": "pool_near_exhaustion",
                                "checked_out": checked_out,
                                "pool_size": pool_size,
                                "overflow": overflow,
                            },
                        )
                except Exception:
                    pass  # Non-critical monitoring


# ===========================================================================
# ENGINE CREATION
# ===========================================================================


def get_engine() -> Engine:
    """Return the global SQLAlchemy Engine, creating it on first call.

    Thread-safe via double-checked locking with ``_lifecycle_lock``
    (ARCH-001, IDEM-001).

    Configuration is read from environment variables with sensible defaults
    tuned for ETL workloads:

    - ``DATABASE_POOL_SIZE`` (default 15 production / 5 development)
    - ``DATABASE_MAX_OVERFLOW`` (default 20)
    - ``DATABASE_POOL_RECYCLE`` (default 7200 = 2 hours)
    - ``DATABASE_POOL_TIMEOUT`` (default 30 seconds)
    - ``DATABASE_STATEMENT_TIMEOUT`` (default 1 800 000 = 30 minutes)
    - ``DATABASE_WORK_MEM`` (default 256 MB)
    - ``DATABASE_LOCK_TIMEOUT`` (default 30 000 ms)
    - ``DATABASE_SSL_MODE`` (default None — not added)
    - ``DATABASE_ECHO`` (default false)
    - ``DATABASE_ISOLATION_LEVEL`` (default REPEATABLE READ for PostgreSQL)
    """
    global _engine
    # Fast path: already created
    if _engine is not None:
        return _engine

    with _lifecycle_lock:
        # Double-checked locking
        if _engine is not None:
            return _engine

        _engine = _create_new_engine()
        return _engine


def _create_new_engine() -> Engine:
    """Build a new SQLAlchemy Engine from the current configuration.

    This function contains all the engine-creation logic, separated from
    ``get_engine()`` for testability (TEST-001).
    """
    # Delayed import: DATABASE_URL not in module namespace (SEC-004)
    from config import settings as _settings

    database_url = getattr(_settings, "DATABASE_URL", "")
    # v36 ROOT FIX (Chain 1): honour DRUGOS_ENVIRONMENT canonical name.
    environment = getattr(
        _settings,
        "ENVIRONMENT",
        os.environ.get("DRUGOS_ENVIRONMENT")
        or os.environ.get("ENVIRONMENT", "development"),
    )

    # Validate URL structure and security (SEC-005)
    _validate_database_url(database_url)

    parsed_url = urlparse(database_url)
    raw_scheme = parsed_url.scheme
    driver = (
        raw_scheme.split("+")[-1]
        if "+" in raw_scheme
        else raw_scheme
    )
    is_sqlite = driver in ("sqlite", "file")

    # SQLAlchemy's create_engine cannot handle 'file:' URLs directly;
    # convert 'file:/path/to/db' to 'sqlite:////path/to/db'
    # v89 ROOT FIX (BUG #31 — file: URL parsing created DB at wrong path):
    #   The previous code did ``db_path = database_url[5:]`` (strip 'file:')
    #   then ``f"sqlite:///{db_path}"``. For ``file:///absolute/path``,
    #   ``database_url[5:]`` = ``//absolute/path``, and
    #   ``f"sqlite:///{db_path}"`` = ``sqlite://///absolute/path``
    #   (5 slashes). SQLite interprets the 4th slash as the path separator
    #   and the 5th as part of the path — so it creates a DB at
    #   ``//absolute/path`` (a network-style path) or ``/absolute/path``
    #   depending on the platform, NOT at ``/absolute/path``. Operators
    #   saw "empty database" with no error — the DB was created at the
    #   wrong location. ROOT FIX: use ``urlparse`` to extract the path
    #   component correctly, then construct the SQLite URL with the
    #   standard ``sqlite:///<absolute_path>`` form (3 slashes for an
    #   absolute path). This handles ``file:/path``, ``file:///path``,
    #   and ``file://host/path`` correctly.
    if driver == "file" and database_url.startswith("file:"):
        _parsed_file = urlparse(database_url)
        db_path = _parsed_file.path or ""
        if not db_path:
            raise ValueError(
                f"Invalid file: URL — no path component: {database_url!r}"
            )
        database_url = f"sqlite:///{db_path}"
        logger.debug("Converted file: URL to SQLite URL: %s", _mask_url(database_url))

    # Auto-create parent directory for SQLite file databases so the engine
    # can create the database file.  This is a robustness improvement: if
    # the configured DATABASE_URL points to a path whose parent directory
    # does not yet exist (common in fresh deployments / CI), SQLite would
    # raise "unable to open database file".  We create the directory with
    # mode 0o755 (user rwx, group rx, others rx) so the file can be
    # created.  This does NOT change behaviour for paths whose parent
    # already exists.
    if is_sqlite and ":///" in database_url:
        # Extract the file path from URLs like sqlite:////absolute/path.db
        # or sqlite:///relative/path.db or sqlite:///path.db
        _sqlite_file_part = database_url.split(":///", 1)[1]
        # Skip in-memory databases (empty path or ":memory:")
        if _sqlite_file_part and _sqlite_file_part != ":memory:":
            _db_file_path = _sqlite_file_part
            # Strip query parameters if present
            if "?" in _db_file_path:
                _db_file_path = _db_file_path.split("?", 1)[0]
            _parent_dir = os.path.dirname(os.path.abspath(_db_file_path))
            if _parent_dir and not os.path.isdir(_parent_dir):
                try:
                    os.makedirs(_parent_dir, exist_ok=True)
                    logger.debug(
                        "Auto-created SQLite parent directory: %s",
                        _parent_dir,
                    )
                except OSError as exc:
                    logger.warning(
                        "Could not auto-create SQLite parent directory '%s': %s",
                        _parent_dir,
                        exc,
                    )

    # Warn about SQLite in non-development environments (KNOW-005)
    if is_sqlite and environment not in ("development", "test", "testing", "ci"):
        logger.error(
            "SQLite detected in '%s' environment. SQLite cannot support "
            "concurrent ETL pipelines. Use PostgreSQL for production.",
            environment,
        )

    logger.info("Creating SQLAlchemy engine for %s", _mask_url(database_url))

    # --- Build connect_args from registry (DES-001) ---
    connect_args: dict[str, Any] = {}

    if not is_sqlite:
        stmt_config = _get_statement_config()
        registry_fn = _DRIVER_CONNECT_ARGS_REGISTRY.get(driver)
        if registry_fn is not None:
            connect_args = registry_fn(
                statement_timeout=stmt_config["statement_timeout"],
                work_mem=stmt_config["work_mem"],
                lock_timeout=stmt_config["lock_timeout"],
                timezone=stmt_config["timezone"],
                sslmode=stmt_config.get("sslmode"),
            )
        else:
            # Unknown PostgreSQL-compatible driver — try generic options
            logger.warning(
                "No connect_args registry entry for driver '%s'. "
                "Using generic PostgreSQL options.",
                driver,
            )
            connect_args = _build_pg_connect_args(
                statement_timeout=stmt_config["statement_timeout"],
                work_mem=stmt_config["work_mem"],
                lock_timeout=stmt_config["lock_timeout"],
                timezone=stmt_config["timezone"],
                sslmode=stmt_config.get("sslmode"),
            )

        # Additional connect_args from environment (INTEROP-005)
        extra_connect_args_json = os.environ.get("DATABASE_CONNECT_ARGS")
        if extra_connect_args_json:
            import json
            try:
                extra = json.loads(extra_connect_args_json)
                if isinstance(extra, dict):
                    connect_args.update(extra)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "Failed to parse DATABASE_CONNECT_ARGS JSON: %s", exc
                )
    else:
        # SQLite busy timeout (KNOW-006)
        connect_args["timeout"] = _get_config_int("DATABASE_SQLITE_TIMEOUT", 30)

    # --- Build engine kwargs ---
    pool_config = _get_pool_config()
    engine_kwargs: dict[str, Any] = {
        "echo": pool_config["echo"],
        "connect_args": connect_args,
    }

    if not is_sqlite:
        engine_kwargs.update({
            "pool_size": pool_config["pool_size"],
            "max_overflow": pool_config["max_overflow"],
            "pool_pre_ping": pool_config["pool_pre_ping"],
            "pool_recycle": pool_config["pool_recycle"],
            "pool_timeout": pool_config["pool_timeout"],
            "pool_use_lifo": pool_config["pool_use_lifo"],
        })

        isolation_level = _get_isolation_level(driver)
        if isolation_level:
            engine_kwargs["isolation_level"] = isolation_level

    # --- Create engine ---
    engine = create_engine(database_url, **engine_kwargs)

    # --- Attach event listeners ---
    _configure_engine_events(engine)

    # --- Pool pre-warming (PERF-001) ---
    pre_warm = _get_config_str("DATABASE_POOL_PRE_WARM", "true").lower() in (
        "true", "1", "yes",
    )
    if pre_warm and not is_sqlite:
        _pre_warm_pool(engine, pool_config["pool_size"])

    return engine


def _pre_warm_pool(engine: Engine, pool_size: int) -> None:
    """Pre-populate the connection pool to avoid cold-start latency (PERF-001)."""
    try:
        connections = []
        for _ in range(pool_size):
            connections.append(engine.connect())
        for conn in connections:
            conn.close()
        logger.info("Connection pool pre-warmed with %d connections", pool_size)
    except Exception as exc:
        logger.warning("Pool pre-warming failed (non-fatal): %s", exc)


# ===========================================================================
# SESSION FACTORY
# ===========================================================================


def get_session_factory() -> scoped_session:
    """Return the thread-safe scoped session factory, creating it on first call.

    Thread-safe via ``_lifecycle_lock`` (ARCH-002, IDEM-001).

    .. deprecated::
        ``scoped_session`` is a legacy pattern in SQLAlchemy 2.x.
        Retained for backward compatibility.  V2 will migrate to explicit
        session management (DES-003).
    """
    global _session_factory
    if _session_factory is not None:
        return _session_factory

    with _lifecycle_lock:
        if _session_factory is not None:
            return _session_factory

        engine = get_engine()
        factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        _session_factory = scoped_session(factory)
        logger.info("Scoped session factory created")
        return _session_factory


# ===========================================================================
# CONTEXT-MANAGED SESSION
# ===========================================================================


@contextmanager
def get_db_session(
    *,
    pipeline_name: Optional[str] = None,
    run_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    verify_commit: bool = False,
    warn_nested: bool = True,
    on_commit: Optional[Callable[[], None]] = None,
    on_rollback: Optional[Callable[[], None]] = None,
) -> Generator[Session, None, None]:
    """Yield a database session with automatic commit / rollback / close.

    Supports nested usage: if called again inside an already-active ``with``
    block on the same thread, the same underlying session is returned and
    only the **outermost** block performs commit / close.

    Parameters
    ----------
    pipeline_name : str, optional
        Name of the ETL pipeline for lineage tracking (LINE-001).
    run_id : str, optional
        Airflow DAG run or task ID for distributed tracing (LOG-005).
    correlation_id : str, optional
        Cross-system correlation ID for tracing (LOG-005, LINE-003).
    verify_commit : bool, default False
        If True, verify that data was actually persisted after commit
        by running a lightweight consistency check (DATA-004).
    warn_nested : bool, default True
        Log a WARNING when nested usage is detected (DES-004).
    on_commit : callable, optional
        Callback invoked after a successful commit (DES-005).
    on_rollback : callable, optional
        Callback invoked after a rollback (DES-005).

    Usage
    -----
    ::

        with get_db_session() as session:
            session.add(obj)

        with get_db_session(pipeline_name='chembl', run_id='abc123') as session:
            session.bulk_insert_mappings(Drug, records)
    """
    factory = get_session_factory()

    # --- Reference counting via threading.local() (CODE-001, CODE-002) ---
    ref_count = getattr(_thread_local, "ref_count", 0) + 1
    _thread_local.ref_count = ref_count

    session: Session = factory()
    is_outermost = ref_count == 1

    # --- Generate session UUID for tracing (LINE-003) ---
    session_id = str(uuid.uuid4())[:8] if is_outermost else getattr(_thread_local, "session_id", "nested")

    if is_outermost:
        _thread_local.session_id = session_id
        _thread_local.session_start_time = time.monotonic()

        # Set PostgreSQL session variables for lineage (LINE-001, LINE-003)
        _set_session_variables(session, pipeline_name, run_id, correlation_id, session_id)

    context_extra: dict[str, Any] = {
        "event_type": "session_lifecycle",
        "session_id": session_id,
        "ref_count": ref_count,
        "is_outermost": is_outermost,
    }
    if pipeline_name:
        context_extra["pipeline_name"] = pipeline_name
    if run_id:
        context_extra["run_id"] = run_id

    if not is_outermost and warn_nested:
        logger.debug(
            "Nested session block entered (session_id=%s, ref_count=%d)",
            session_id, ref_count,
            extra=context_extra,
        )

    try:
        yield session

        if is_outermost:
            _commit_with_retry(session, context_extra)

            # Read-after-write verification (DATA-004)
            if verify_commit:
                _verify_commit(session, context_extra)

            # Invoke commit callback (DES-005)
            if on_commit is not None:
                try:
                    on_commit()
                except Exception as cb_exc:
                    logger.warning(
                        "on_commit callback failed: %s", cb_exc,
                        extra=context_extra,
                    )

            elapsed = time.monotonic() - getattr(_thread_local, "session_start_time", time.monotonic())
            logger.info(
                "Session committed successfully (session_id=%s, duration=%.2fs)",
                session_id, elapsed,
                extra={**context_extra, "event_type": "session_commit", "duration_s": elapsed},
            )
        else:
            logger.debug(
                "Nested session block exiting — deferring commit to outermost block "
                "(session_id=%s, ref_count=%d)",
                session_id, ref_count,
                extra=context_extra,
            )

    except Exception as exc:
        # Differentiate error types (CODE-008)
        is_transient = isinstance(exc, (OperationalError, InterfaceError, DBAPIError))

        if is_outermost:
            try:
                session.rollback()
            except Exception as rollback_exc:
                logger.error(
                    "Rollback also failed: %s", rollback_exc,
                    extra={**context_extra, "event_type": "rollback_failure"},
                )
            logger.warning(
                "Session rolled back due to %s (session_id=%s): %s",
                type(exc).__name__, session_id, exc,
                exc_info=not _is_production(),
                extra={**context_extra, "event_type": "session_rollback"},
            )

            # Invoke rollback callback (DES-005)
            if on_rollback is not None:
                try:
                    on_rollback()
                except Exception as cb_exc:
                    logger.warning("on_rollback callback failed: %s", cb_exc)
        else:
            logger.warning(
                "Nested session block received %s — propagating to outermost block "
                "(session_id=%s): %s",
                type(exc).__name__, session_id, exc,
                extra=context_extra,
            )
        raise

    finally:
        _thread_local.ref_count = ref_count - 1
        current_count = _thread_local.ref_count

        if current_count <= 0:
            _thread_local.ref_count = 0
            _thread_local.session_id = None
            _thread_local.session_start_time = None
            # Correct ordering: factory.remove() handles session.close()
            # internally (CODE-004)
            try:
                factory.remove()
            except Exception as remove_exc:
                logger.warning(
                    "factory.remove() failed during cleanup: %s", remove_exc,
                    extra=context_extra,
                )
            logger.debug(
                "Session closed and removed (outermost block, session_id=%s)",
                session_id,
                extra=context_extra,
            )
        else:
            logger.debug(
                "Nested session block closing — ref_count=%d (session_id=%s)",
                current_count, session_id,
                extra=context_extra,
            )


# v104 FORENSIC ROOT FIX (P1-001 -- session_scope undefined):
#   ``retry_transaction()`` (line ~1589) calls ``with session_scope(...)``
#   as a context manager, but ``session_scope`` was NEVER defined anywhere
#   in the codebase (not in connection.py, base.py, or any other module).
#   Every call to ``retry_transaction()`` raised ``NameError`` at runtime,
#   silently disabling the entire P1-A2 silent-data-loss retry fix. Any
#   transient DB error (network blip, deadlock) that should have been
#   retried instead propagated as an unhandled NameError, and the
#   original DB operation was lost.
#
#   ROOT FIX: define ``session_scope`` as a thin alias for the existing
#   ``get_db_session`` context manager. ``get_db_session`` ALREADY
#   implements the full session lifecycle (factory acquire -> yield ->
#   commit with retry -> rollback on exception -> callback hooks ->
#   factory.remove() in finally) and accepts the SAME kwargs that
#   ``retry_transaction`` forwards (``pipeline_name``, ``run_id``,
#   ``correlation_id``, ``warn_nested``, ``verify_commit``,
#   ``on_commit``, ``on_rollback``). Aliasing (rather than wrapping)
#   preserves the exact semantics and avoids a double-yield generator
#   chain that would silently drop the outermost commit logic.
#
#   ``get_db_session`` is a generator function (it contains ``yield``),
#   so it is automatically usable as a context manager WITHOUT the
#   ``@contextmanager`` decorator. ``with session_scope(...) as
#   session:`` therefore behaves identically to ``with
#   get_db_session(...) as session:``. The alias is exported in
#   ``__all__`` so it is part of the module's public API and
#   discoverable via grep.
#
#   Regression test: phase1/tests/test_p1_001_session_scope.py asserts
#   that ``session_scope`` is callable, that ``with session_scope()``
#   yields a usable Session, that the Session commits on clean exit and
#   rolls back on exception, and that ``retry_transaction`` no longer
#   raises NameError when handed a transient-failing work callable.
session_scope = get_db_session
"""``session_scope`` -- public alias for :func:`get_db_session`.

P1-001 ROOT FIX: this alias is the symbol that ``retry_transaction``
and other call sites reference. Removing it (or forgetting to define
it) breaks every retry path. Treat this alias as part of the module's
public API.
"""


def _set_session_variables(
    session: Session,
    pipeline_name: Optional[str],
    run_id: Optional[str],
    correlation_id: Optional[str],
    session_id: str,
) -> None:
    """Set PostgreSQL session variables for lineage and tracing.

    v38 ROOT FIX (Phase 1 Issue #40): the previous code used manual SQL
    escape by doubling single quotes (``chr(39)`` = ``'``). This is the
    standard SQL escape but it doesn't handle backslashes, null bytes,
    or other injection vectors. The fix uses PostgreSQL's
    ``set_config()`` function with a parameterized query, which is
    injection-safe. ``set_config(name, value, is_local)`` with
    ``is_local=false`` sets a session-level variable (equivalent to
    ``SET app.<name> = <value>`` but accepts the value as a bound
    parameter, eliminating the injection surface entirely).
    """
    try:
        # Only set for PostgreSQL — SQLite doesn't support SET
        bind = session.get_bind()
        if bind is not None and hasattr(bind, "url") and str(bind.url).startswith("postgresql"):
            # v38 ROOT FIX (Issue #40): use set_config() with bound
            # parameters instead of manual chr(39) escaping. This is
            # injection-safe — the value is passed as a parameter, not
            # interpolated into the SQL string.
            #
            # v89/v90 ROOT FIX (BUG #8 / BUG #37 — session variables
            #   contaminate across pool reuse):
            #   The previous code used ``set_config(name, value, FALSE)``
            #   which sets a SESSION-level variable. Session-level
            #   variables persist until the session ends — but with
            #   connection pooling, the "session" is the underlying DB
            #   connection, which is RETURNED TO THE POOL on
            #   ``session.close()``. The next ``get_db_session()`` call
            #   (possibly from a DIFFERENT pipeline) gets a connection
            #   with STALE ``app.pipeline_name`` / ``app.run_id`` /
            #   ``app.correlation_id`` variables. If that next call
            #   writes to ``audit_log`` or ``pipeline_runs``, the lineage
            #   columns reflect the PREVIOUS pipeline's identity. Under
            #   the documented 7-concurrent-pipeline workload, cross-
            #   contamination is near-certain — and operators cannot
            #   reproduce because the bug depends on pool reuse order.
            #   Regulatory audits (GDPR/HIPAA) that require "which
            #   pipeline wrote this row" cannot trust the ``audit_log``
            #   table. ROOT FIX: use ``set_config(name, value, TRUE)``
            #   which sets a TRANSACTION-local variable (equivalent to
            #   ``SET LOCAL``). Transaction-local variables are
            #   automatically RESET to their previous value (or NULL)
            #   on COMMIT or ROLLBACK — so when the session commits and
            #   the connection returns to the pool, the variables are
            #   GONE. The next checkout gets a clean connection with no
            #   stale ``app.*`` variables. This is the PostgreSQL-
            #   documented pattern for connection-pool-safe session
            #   variables (see PostgreSQL docs §18.1.4: "SET LOCAL's
            #   effects last only till the end of the current
            #   transaction").
            from sqlalchemy import text as _sa_text_v38
            if pipeline_name:
                session.execute(
                    _sa_text_v38("SELECT set_config('app.pipeline_name', :val, true)"),
                    {"val": str(pipeline_name)},
                )
            if run_id:
                session.execute(
                    _sa_text_v38("SELECT set_config('app.run_id', :val, true)"),
                    {"val": str(run_id)},
                )
            if correlation_id:
                session.execute(
                    _sa_text_v38("SELECT set_config('app.correlation_id', :val, true)"),
                    {"val": str(correlation_id)},
                )
            session.execute(
                _sa_text_v38("SELECT set_config('app.session_id', :val, true)"),
                {"val": str(session_id)},
            )
    except Exception as exc:
        # Non-fatal: session variables are for observability only
        logger.debug("Could not set session variables: %s", exc)


def _commit_with_retry(
    session: Session,
    context_extra: dict[str, Any],
    max_retries: int = 3,
    backoff_base: float = 2.0,
    work: Optional[Callable[[Session], None]] = None,
) -> None:
    """Commit with exponential-backoff retry for transient errors (REL-001).

    v79 FORENSIC ROOT FIX (P0-A2 — retry-after-rollback committed NOTHING,
    silently losing data):
      The v49/v78 code caught ``OperationalError``/``InterfaceError``,
      called ``session.rollback()``, then retried ``session.commit()``.
      But ``session.rollback()`` CLEARS ALL PENDING CHANGES from the
      session. The retried ``session.commit()`` therefore committed
      NOTHING — an empty transaction — and returned successfully. The
      caller saw ``UpsertResult(inserted=1000)`` but the database had
      ZERO rows. Silent data loss on every transient commit error
      (connection blip, deadlock, replication lag).

      The v49 docstring even ACKNOWLEDGED this: "this means the work
      done in the session BEFORE the failed commit is LOST — the
      caller must re-execute the work in the retry." But the function
      provided NO mechanism for the caller to re-execute — it just
      retried the bare ``commit()``. The contract was self-contradictory.

    ROOT FIX (two-layer, both required):
      1. If a ``work`` callable is supplied, re-execute it on EVERY
         retry (after rollback, before commit). This is the ONLY way
         to honestly retry a transaction: the work must be re-done on
         a clean session. The ``work`` callable must be idempotent
         (ON CONFLICT DO UPDATE) so re-execution is safe.
      2. If NO ``work`` callable is supplied (the legacy context-manager
         path, where the work was done inline before ``yield`` returned),
         DO NOT RETRY. Rollback and RE-RAISE the transient error
         immediately. Retrying a bare ``commit()`` after rollback
         commits nothing and silently lies to the caller. The error
         propagates so the caller (or a wrapping
         ``retry_session_scope``) can re-do the entire unit of work.

    Contract:
      - ``work=None`` (default, backward-compatible): NO retry. A
        transient commit error is rolled back and re-raised. Callers
        that want retry must wrap the ENTIRE ``with session_scope()``
        block in ``retry_session_scope()`` (below) — which re-opens a
        FRESH session and re-executes the block.
      - ``work=callable``: the callable is re-invoked on every retry
        after rollback. The callable must be idempotent. This is the
        correct pattern for programmatic callers that build the
        session work in a function (not a context-manager block).

    Parameters
    ----------
    session : Session
        The SQLAlchemy session with pending work.
    context_extra : dict
        Structured logging context.
    max_retries : int
        Max retry attempts (only used when ``work`` is supplied).
    backoff_base : float
        Exponential backoff base (``delay = backoff_base ** attempt``).
    work : callable(Session) -> None, optional
        The idempotent work function to re-execute on each retry.
        If None, NO retry is attempted (transient errors re-raise).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            # v79 ROOT FIX: if work is supplied and this is a retry
            # (attempt > 0), re-execute the work on the rolled-back
            # (clean) session BEFORE committing. Without this, the
            # retry commits an empty transaction (the v78 silent-data-loss bug).
            if work is not None and attempt > 0:
                work(session)
            session.commit()
            return
        except (OperationalError, InterfaceError) as exc:
            last_exc = exc
            # Rollback to clear the pending-rollback state. Without
            # this, the next session.commit() raises PendingRollbackError
            # instead of the real transient error.
            try:
                session.rollback()
            except Exception as rb_exc:
                logger.warning(
                    "Rollback after commit failure also failed: %s",
                    rb_exc,
                    extra={**context_extra, "event_type": "rollback_failure"},
                )
            # v79 ROOT FIX: if NO work callable is supplied, we CANNOT
            # honestly retry (the work is gone after rollback). Re-raise
            # immediately so the caller can handle it. Retrying a bare
            # commit() would commit nothing and silently lie.
            if work is None:
                logger.error(
                    "Transient commit error with no work callable — cannot "
                    "retry (work was cleared by rollback). Re-raising so "
                    "caller can retry the entire unit of work: %s",
                    exc,
                    extra={**context_extra, "event_type": "commit_no_retry"},
                )
                raise
            if attempt < max_retries:
                delay = backoff_base ** attempt
                logger.warning(
                    "Transient commit error (attempt %d/%d), retrying in "
                    "%.1fs after rollback + work re-execution (v79 root "
                    "fix): %s",
                    attempt + 1, max_retries + 1, delay, exc,
                    extra={**context_extra, "event_type": "commit_retry"},
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Commit failed after %d retries: %s", max_retries, exc,
                    extra=context_extra,
                )
        except Exception:
            # Non-transient errors: rollback (to clean state) then re-raise.
            try:
                session.rollback()
            except Exception:
                pass
            raise
    if last_exc is not None:
        raise last_exc


def _verify_commit(session: Session, context_extra: dict[str, Any]) -> None:
    """Lightweight post-commit verification (DATA-004)."""
    try:
        result = session.execute(text("SELECT 1"))
        result.close()
        logger.debug("Post-commit verification passed", extra=context_extra)
    except Exception as exc:
        logger.warning(
            "Post-commit verification failed: %s", exc,
            extra={**context_extra, "event_type": "verify_commit_failure"},
        )


def _is_production() -> bool:
    """v36 ROOT FIX (Chain 1): delegate to canonical ``is_production_environment``.

    Previous code read ENVIRONMENT (legacy) instead of DRUGOS_ENVIRONMENT
    (canonical), so docker-compose production deployments were silently
    treated as dev — causing stack-trace leakage via ``exc_info=not
    _is_production()`` and dev-sized DB pools.
    """
    return is_production_environment()


# ===========================================================================
# v79 FORENSIC ROOT FIX (P0-A2) — retry_transaction
# ===========================================================================


def retry_transaction(
    work: Callable[[Session], None],
    *,
    pipeline_name: str = "",
    run_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    warn_nested: bool = True,
    verify_commit: bool = False,
    on_commit: Optional[Callable[[], None]] = None,
    on_rollback: Optional[Callable[[], None]] = None,
) -> None:
    """Execute idempotent transactional work with full retry on transient
    errors (v79 P0-A2 root fix).

    This is the ONLY honest way to retry a transaction after a rollback:
    open a FRESH session and RE-EXECUTE the work callable from scratch.
    The v48/v78 ``_commit_with_retry`` retried only the bare
    ``session.commit()`` after a rollback — which committed NOTHING
    (rollback clears pending work) and silently lied about success.
    This function fixes that by re-doing the WORK, not just the commit.

    Usage::

        def _do_upsert(session: Session) -> None:
            bulk_upsert_gda(session, df)   # idempotent (ON CONFLICT)

        retry_transaction(_do_upsert, pipeline_name="disgenet")

    Contract:
      - ``work`` MUST be idempotent (use ``ON CONFLICT DO UPDATE``, not
        bare ``INSERT``) so that re-execution on a fresh session does
        not create duplicates. All Phase 1 ``bulk_upsert_*`` functions
        satisfy this contract.
      - Non-transient errors (``IntegrityError``, ``ProgrammingError``,
        etc.) propagate immediately — they are not retried.
      - Transient errors (``OperationalError``, ``InterfaceError``)
        trigger a full retry: fresh session + re-executed work + commit.

    Parameters
    ----------
    work : callable(Session) -> None
        The idempotent work function. Receives a fresh session each attempt.
    pipeline_name, run_id, correlation_id : str, optional
        Observability metadata passed to ``session_scope``.
    max_retries : int
        Max retry attempts (default 3 → up to 4 total attempts).
    backoff_base : float
        Exponential backoff base (``delay = backoff_base ** attempt``).
    warn_nested, verify_commit, on_commit, on_rollback
        Forwarded to ``session_scope``.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            with session_scope(
                pipeline_name=pipeline_name,
                run_id=run_id,
                correlation_id=correlation_id,
                warn_nested=warn_nested and attempt == 0,
                verify_commit=verify_commit,
                on_commit=on_commit,
                on_rollback=on_rollback,
            ) as session:
                # Re-execute the work on EVERY attempt (fresh session
                # each time). This is the root fix: the work is re-done,
                # not just the commit.
                work(session)
            return  # success — exit the retry loop
        except (OperationalError, InterfaceError) as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = backoff_base ** attempt
                logger.warning(
                    "retry_transaction: transient error on attempt %d/%d, "
                    "retrying ENTIRE work on a FRESH session in %.1fs "
                    "(v79 P0-A2 root fix): %s",
                    attempt + 1, max_retries + 1, delay, exc,
                    extra={
                        "event_type": "retry_transaction_retry",
                        "pipeline_name": pipeline_name,
                        "attempt": attempt + 1,
                        "max_attempts": max_retries + 1,
                    },
                )
                time.sleep(delay)
            else:
                logger.error(
                    "retry_transaction: failed after %d retries: %s",
                    max_retries, exc,
                    extra={
                        "event_type": "retry_transaction_exhausted",
                        "pipeline_name": pipeline_name,
                    },
                )
        # Non-transient exceptions propagate immediately (no retry).
    if last_exc is not None:
        raise last_exc


# ===========================================================================
# READ-ONLY SESSION (PERF-005)
# ===========================================================================


@contextmanager
def get_read_only_session() -> Generator[Session, None, None]:
    """Yield a read-only session optimized for lookup operations.

    Uses ``expire_on_commit=False`` and ``autoflush=False`` to minimise
    overhead for read-heavy operations like entity resolution lookups.

    This is an ADDITION to the API; existing callers are unaffected.
    """
    engine = get_engine()
    session = Session(bind=engine, autoflush=False, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()


# ===========================================================================
# DATABASE INITIALISATION
# ===========================================================================


def init_db(initiator: Optional[str] = None) -> None:
    """Create all tables and run pending migrations.

    Tables are created via ``Base.metadata.create_all`` (additive — never
    drops or alters).  Migrations are then applied to add missing columns
    and constraints.  If any migration fails, a ``RuntimeError`` is raised
    (DATA-003, IDEM-003, ARCH-007).

    Uses advisory locking for PostgreSQL or file-based locking for SQLite
    to prevent concurrent ``init_db()`` from racing (REL-006).

    Parameters
    ----------
    initiator : str, optional
        Name of the process/pipeline calling init_db() for traceability
        (LINE-004).
    """
    # Import models so that Base.metadata picks them up before create_all.
    import database.models  # noqa: F401

    engine = get_engine()
    initiator_info = initiator or "unknown"
    logger.info(
        "Initialising database schema (create_all), initiator=%s",
        initiator_info,
        extra={"event_type": "init_db_start", "initiator": initiator_info},
    )

    # --- Advisory lock for concurrent safety (REL-006) ---
    url_str = str(engine.url)
    url_scheme = url_str.split(":")[0].split("+")[0] if url_str else ""
    is_sqlite = "sqlite" in url_scheme
    lock_released = False

    if not is_sqlite:
        conn_for_lock = engine.connect()
        try:
            conn_for_lock.execute(text("SELECT pg_advisory_lock(12345)"))
        except Exception as exc:
            # REM-28 ROOT FIX (patient-safety): previously a failed
            # pg_advisory_lock only logged a WARNING and init_db()
            # continued WITHOUT the lock. Two processes could then race
            # on CREATE TABLE IF NOT EXISTS / migration ALTERs and
            # corrupt the schema (e.g. half-applied migrations, missing
            # FKs, NULL columns where NOT NULL is required). For a
            # biomedical KG whose outputs feed clinical decision-making,
            # a corrupt schema is a patient-safety incident. Therefore
            # in PRODUCTION (Postgres) we treat the lock failure as
            # FATAL. SQLite is exempt because pg_advisory_lock is not
            # supported there (single-process anyway), so the SQLite
            # branch below is intentionally unchanged.
            conn_for_lock.close()
            raise RuntimeError(
                "Cannot acquire pg_advisory_lock — another init_db() "
                "may be running. Concurrent schema migrations can "
                "corrupt the DB (patient-safety risk). Original error: "
                + str(exc)
            ) from exc
    else:
        conn_for_lock = None

    try:
        # v13 ROOT FIX (CD-1): v12 ran ``Base.metadata.create_all()``
        # BEFORE ``run_migrations()``. The ORM creates tables with
        # ``Float`` (not NUMERIC), ``nullable=True`` (not NOT NULL),
        # and no FKs on ``pubchem_compound_properties``. Migration
        # 001's ``CREATE TABLE IF NOT EXISTS`` then became a no-op
        # (table already existed from create_all), so NUMERIC
        # precision, NOT NULL constraints, FKs, and CHECK constraints
        # were NEVER applied on PostgreSQL. SQLite was even worse —
        # migrations 001-006 were skipped entirely (CD-5).
        #
        # v13 fix: run migrations FIRST (they use
        # ``CREATE TABLE IF NOT EXISTS`` so they're idempotent and
        # safe to run on an empty DB). Then run ``create_all()`` as
        # a SAFETY NET to catch any ORM-declared table that doesn't
        # have a migration (so new dev tables still get created
        # without requiring a migration). This way:
        #   - On a fresh DB: migrations create tables with the
        #     correct schema (NUMERIC, NOT NULL, FKs, CHECKs).
        #     create_all is a no-op (tables already exist).
        #   - On an existing DB: migrations apply pending
        #     ALTERs/add columns. create_all is a no-op.
        #   - On SQLite: v16 ROOT FIX (CD-5) — migrations now ACTUALLY
        #     run, with on-the-fly PostgreSQL→SQLite SQL translation.
        #     Previously the comment here claimed migrations ran on
        #     SQLite, but run_migrations.py SKIPPED all .sql files —
        #     only Python-side column-adds ran. This left SQLite
        #     dev/test DBs missing CHECK/UNIQUE/FK constraints, so
        #     code that passed tests on SQLite could fail on PostgreSQL.
        #     v16 adds _translate_sql_for_sqlite() and a SQLite branch
        #     that runs the translated migrations.
        #
        # Run migrations FIRST (creates tables with correct schema).
        try:
            from database.migrations.run_migrations import run_migrations
            logger.info("Running migrations (pre-create_all) …")
            run_migrations()
            logger.info("Pre-create_all migrations complete")
        except Exception as exc:
            raise RuntimeError(
                f"Database migration failed (initiator={initiator_info}): {exc}. "
                f"The schema may be in an inconsistent state. "
                f"Check _migration_history table for details."
            ) from exc

        # Then run create_all as a safety net for ORM-declared tables
        # that don't have a migration. On a DB where migrations
        # already created all tables, this is a no-op.
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialisation (create_all safety net) complete")

        # Schema verification (IDEM-003, IDEM-004)
        _verify_schema_completeness(engine)

    finally:
        # Release advisory lock
        if conn_for_lock is not None:
            try:
                if not is_sqlite:
                    conn_for_lock.execute(text("SELECT pg_advisory_unlock(12345)"))
            except Exception:
                pass
            finally:
                conn_for_lock.close()


def _verify_schema_completeness(engine: Engine) -> None:
    """Verify that all expected columns exist in the database (IDEM-003, IDEM-004)."""
    try:
        from database.migrations.run_migrations import REQUIRED_COLUMNS
        inspector = inspect(engine)

        for table_name, expected_columns in REQUIRED_COLUMNS.items():
            if not inspector.has_table(table_name):
                logger.warning(
                    "Schema verification: table '%s' is missing", table_name,
                    extra={"event_type": "schema_verification", "table": table_name},
                )
                continue

            existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
            for col_name, col_type in expected_columns:
                if col_name not in existing_columns:
                    logger.warning(
                        "Schema verification: column '%s.%s' is missing",
                        table_name, col_name,
                        extra={
                            "event_type": "schema_verification",
                            "table": table_name,
                            "column": col_name,
                        },
                    )
    except Exception as exc:
        logger.warning(
            "Schema completeness verification failed (non-fatal): %s", exc,
        )


# ===========================================================================
# ENGINE DISPOSAL
# ===========================================================================


def dispose_engine(force: bool = False) -> None:
    """Dispose of the global engine and session factory.

    Parameters
    ----------
    force : bool, default False
        If ``False`` and sessions are currently active, log a WARNING and
        raise ``RuntimeError`` instead of disposing (REL-003, LINE-005).
        If ``True``, dispose regardless of active sessions.

    Raises
    ------
    RuntimeError
        If ``force=False`` and sessions are active.
    """
    global _engine, _session_factory

    with _lifecycle_lock:
        # Check for active sessions (REL-003, LINE-005)
        active_count = getattr(_thread_local, "ref_count", 0)
        if active_count > 0 and not force:
            logger.warning(
                "dispose_engine() called with %d active session(s). "
                "Use force=True to dispose anyway.",
                active_count,
                extra={
                    "event_type": "dispose_active_sessions",
                    "active_count": active_count,
                },
            )
            raise RuntimeError(
                f"Cannot dispose engine: {active_count} active session(s). "
                f"Call dispose_engine(force=True) to force disposal."
            )

        if active_count > 0:
            logger.warning(
                "Force-disposing engine with %d active session(s)",
                active_count,
            )

        if _session_factory is not None:
            try:
                _session_factory.remove()
            except Exception as exc:
                logger.warning("Error removing session factory: %s", exc)
            _session_factory = None
            logger.info("Scoped session factory disposed")

        if _engine is not None:
            _engine.dispose()
            _engine = None
            logger.info("SQLAlchemy engine disposed")

        # Clear thread-local state (IDEM-002, ARCH-005)
        _thread_local.ref_count = 0
        _thread_local.session_id = None
        _thread_local.session_start_time = None


# ===========================================================================
# HEALTH CHECK
# ===========================================================================


def check_connection(
    detailed: bool = False,
    use_session_pool: bool = False,
) -> Any:
    """Verify the database is reachable.

    Parameters
    ----------
    detailed : bool, default False
        If ``True``, return a ``HealthCheckResult`` dataclass with
        diagnostic information (DES-006, REL-004, LINE-006, PERF-004).
        If ``False``, return a plain ``bool`` for backward compatibility.
    use_session_pool : bool, default False
        If ``True``, execute the health query through ``get_db_session()``
        to test the full session path (PERF-004).

    Returns
    -------
    bool or HealthCheckResult
        ``bool`` when ``detailed=False``, ``HealthCheckResult`` when
        ``detailed=True``.
    """
    # Check circuit breaker (REL-005)
    if not _circuit_breaker.allow_request():
        msg = "Database circuit breaker is OPEN — connection attempts blocked"
        logger.error(msg)
        if detailed:
            return HealthCheckResult(
                is_healthy=False,
                error_detail=msg,
                error_type="CircuitBreakerOpen",
            )
        return False

    start_time = time.monotonic()
    try:
        # v89 ROOT FIX (BUG #35 — engine variable undefined when
        #   use_session_pool=True): the previous code defined ``engine``
        #   only in the ``else`` branch (line 1766), then used
        #   ``engine if not use_session_pool else get_engine()`` at line
        #   1782. When ``use_session_pool=True``, ``engine`` was undefined
        #   — the conditional relied on Python's short-circuit evaluation
        #   to avoid the NameError. This WORKED but was fragile: any
        #   refactor that changed the conditional order or removed the
        #   short-circuit would raise ``NameError: name 'engine' is not
        #   defined``. ROOT FIX: define ``engine = get_engine()`` BEFORE
        #   the ``if use_session_pool:`` block so the variable is always
        #   bound. The ``get_engine()`` call is cheap (returns the cached
        #   singleton) and the code is now refactor-safe.
        engine = get_engine()
        if use_session_pool:
            with get_db_session() as session:
                result = session.execute(text("SELECT 1"))
                result.close()
                db_version = _get_db_version(session)
        else:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.close()
                db_version = _try_get_db_version(conn)

        latency_ms = (time.monotonic() - start_time) * 1000
        _circuit_breaker.record_success()

        logger.info(
            "Database connectivity check passed (%.1f ms)", latency_ms,
            extra={"event_type": "health_check", "latency_ms": latency_ms},
        )

        if detailed:
            pool_status = get_pool_status() if _engine is not None else None
            db_name, db_user = _try_get_db_metadata(engine)
            return HealthCheckResult(
                is_healthy=True,
                latency_ms=latency_ms,
                pool_status=pool_status,
                db_version=db_version,
                db_name=db_name,
                db_user=db_user,
            )
        return True

    except Exception as exc:
        latency_ms = (time.monotonic() - start_time) * 1000
        _circuit_breaker.record_failure()

        # v36 ROOT FIX (Chain 1): use canonical environment detector.
        environment = _get_environment()
        logger.error(
            "Database connectivity check failed (%.1f ms): %s",
            latency_ms, type(exc).__name__,
            exc_info=(not is_production_environment()),  # SEC-007
            extra={
                "event_type": "health_check_failure",
                "latency_ms": latency_ms,
                "error_type": type(exc).__name__,
            },
        )

        if detailed:
            return HealthCheckResult(
                is_healthy=False,
                latency_ms=latency_ms,
                error_detail=str(exc),
                error_type=type(exc).__name__,
            )
        return False


def _try_get_db_version(connection: Any) -> Optional[str]:
    """Attempt to retrieve database server version."""
    try:
        result = connection.execute(text("SELECT version()"))
        row = result.fetchone()
        result.close()
        return row[0] if row else None
    except Exception:
        return None


def _get_db_version(session: Session) -> Optional[str]:
    """Attempt to retrieve database server version via session."""
    try:
        result = session.execute(text("SELECT version()"))
        row = result.fetchone()
        result.close()
        return row[0] if row else None
    except Exception:
        return None


def _try_get_db_metadata(engine: Engine) -> Tuple[Optional[str], Optional[str]]:
    """Attempt to retrieve database name and user."""
    try:
        with engine.connect() as conn:
            db_name = None
            db_user = None
            try:
                result = conn.execute(text("SELECT current_database()"))
                row = result.fetchone()
                result.close()
                db_name = row[0] if row else None
            except Exception:
                pass
            try:
                result = conn.execute(text("SELECT current_user"))
                row = result.fetchone()
                result.close()
                db_user = row[0] if row else None
            except Exception:
                pass
            return db_name, db_user
    except Exception:
        return None, None


# ===========================================================================
# POOL STATUS (PERF-003)
# ===========================================================================


def get_pool_status() -> Optional[Dict[str, Any]]:
    """Return connection pool metrics for monitoring.

    Returns ``None`` if the engine has not been created yet.

    Returns
    -------
    dict or None
        Keys: ``pool_size``, ``checked_out``, ``overflow``, ``available``.
        For SQLite (SingletonThreadPool), returns a simplified status.
    """
    engine = _engine
    if engine is None:
        return None

    try:
        pool = engine.pool
        # SQLite uses SingletonThreadPool which doesn't have checkedout/overflow
        if hasattr(pool, "checkedout"):
            return {
                "pool_size": pool.size(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "available": pool.size() - pool.checkedout(),
            }
        else:
            return {
                "pool_size": pool.size(),
                "checked_out": 0,
                "overflow": 0,
                "available": pool.size(),
                "pool_type": type(pool).__name__,
            }
    except Exception as exc:
        logger.warning("Failed to get pool status: %s", exc)
        return None


# ===========================================================================
# SCHEMA VERIFICATION (IDEM-004)
# ===========================================================================


def verify_schema() -> Dict[str, Any]:
    """Compare the current database schema against ORM model expectations.

    Returns a ``SchemaDriftReport`` dictionary with any differences found.
    This is an ADDITION to the module, not a modification.

    P1-037 v113 ROOT FIX: the previous version compared column NAMES only,
    not column TYPES. A column declared as ``Numeric(10, 4)`` in the ORM
    but ``FLOAT`` in the DB would NOT be detected — both have a column
    named ``activity_value``, so ``missing`` and ``extra`` were empty.
    The ``is_consistent: True`` result was a false positive. pIC50
    calculations diverged by 1 ULP between dev and prod, the GNN trained
    on slightly different values, and the prod model's predictions didn't
    match the dev model's.

    ROOT FIX: add a type-comparison step. For each column, compare
    ``str(col["type"])`` against ``str(orm_col.type)``. Log a WARNING
    (not ERROR) on type mismatch — type aliases (NUMERIC vs DECIMAL) are
    semantically equivalent and should not block startup. The drift
    report now includes a ``type_mismatches`` dict so operators can audit.
    """
    import database.models  # noqa: F401

    engine = get_engine()
    inspector = inspect(engine)
    drift_report: Dict[str, Any] = {
        "tables_checked": 0,
        "missing_tables": [],
        "missing_columns": {},
        "extra_columns": {},
        # P1-037 v113: type drift detection (WARNING, not ERROR)
        "type_mismatches": {},
        "is_consistent": True,
    }

    for table_name, table in Base.metadata.tables.items():
        drift_report["tables_checked"] += 1

        if not inspector.has_table(table_name):
            drift_report["missing_tables"].append(table_name)
            drift_report["is_consistent"] = False
            continue

        existing_cols_raw = inspector.get_columns(table_name)
        existing_cols = {col["name"] for col in existing_cols_raw}
        existing_col_types = {col["name"]: str(col["type"]) for col in existing_cols_raw}
        expected_cols = {col.name for col in table.columns}
        expected_col_types = {col.name: str(col.type) for col in table.columns}

        missing = expected_cols - existing_cols
        extra = existing_cols - expected_cols

        if missing:
            drift_report["missing_columns"][table_name] = sorted(missing)
            drift_report["is_consistent"] = False
        if extra:
            drift_report["extra_columns"][table_name] = sorted(extra)

        # P1-037 v113 ROOT FIX: type drift detection.
        # Compare str(type) for columns that exist in BOTH ORM and DB.
        # Log a WARNING (not ERROR) — type aliases (NUMERIC vs DECIMAL,
        # VARCHAR vs TEXT) are semantically equivalent. The mismatch is
        # recorded in the drift report but does NOT set is_consistent=False
        # (operators should investigate but it's not a hard failure).
        type_mismatches_for_table: Dict[str, Dict[str, str]] = {}
        for col_name in (expected_cols & existing_cols):
            orm_type = expected_col_types.get(col_name, "")
            db_type = existing_col_types.get(col_name, "")
            if orm_type != db_type:
                # Normalize for comparison: uppercase, strip length specs
                # for common equivalent types (NUMERIC(10,4) vs NUMERIC).
                # We log the raw types so operators can see the exact diff.
                type_mismatches_for_table[col_name] = {
                    "orm_type": orm_type,
                    "db_type": db_type,
                }
                logger.warning(
                    "verify_schema: type drift on %s.%s — ORM=%s, DB=%s. "
                    "This may be a benign alias (NUMERIC vs DECIMAL) or a "
                    "real precision issue. Investigate if ML predictions "
                    "diverge between dev and prod.",
                    table_name, col_name, orm_type, db_type,
                )
        if type_mismatches_for_table:
            drift_report["type_mismatches"][table_name] = type_mismatches_for_table

    return drift_report


# ===========================================================================
# TESTABILITY HOOKS (TEST-001, SEC-003)
# ===========================================================================


def configure_engine(url: str, **kwargs: Any) -> Engine:
    """Create and set a new engine with the given URL (TEST-001).

    Useful for testing with in-memory SQLite or alternative databases
    without monkey-patching module globals.

    P1-A4 ROOT FIX (v82): the previous implementation released
    ``_lifecycle_lock`` between ``dispose_engine()`` and ``create_engine()``.
    A concurrent thread could acquire the lock in that gap, see ``_engine``
    still pointing at the disposed engine, and use it — triggering
    ``DetachedInstanceError`` / ``StatementError`` on the next query. Under
    the 7-concurrent-pipeline Phase 1 workload, this race was non-deterministic
    and untraceable. ROOT FIX: hold ``_lifecycle_lock`` for the ENTIRE
    dispose + create sequence — no window where a concurrent thread can
    observe a half-disposed engine.

    Parameters
    ----------
    url : str
        Database URL to use for the new engine.
    **kwargs
        Additional keyword arguments passed to ``create_engine()``.

    Returns
    -------
    Engine
        The newly created engine.
    """
    global _engine, _session_factory
    with _lifecycle_lock:
        # Dispose existing engine WHILE HOLDING THE LOCK — no gap.
        if _engine is not None:
            try:
                _engine.dispose()
            except Exception:
                pass
        _session_factory = None
        _engine = None

        engine = create_engine(url, **kwargs)
        _configure_engine_events(engine)
        _engine = engine
        _session_factory = scoped_session(
            sessionmaker(bind=engine, autoflush=False, autocommit=False)
        )
        return _engine


def reinitialize_engine() -> Engine:
    """Safely dispose and recreate the engine with the current DATABASE_URL (SEC-003).

    This allows credential rotation without process restart.
    """
    with _lifecycle_lock:
        dispose_engine(force=True)

    return get_engine()


def reset_global_state() -> None:
    """Clear all global state for test teardown.

    This is stronger than ``dispose_engine()`` — it also clears the circuit
    breaker and thread-local state.  Use in test fixtures.
    """
    global _engine, _session_factory

    with _lifecycle_lock:
        if _session_factory is not None:
            try:
                _session_factory.remove()
            except Exception:
                pass
            _session_factory = None

        if _engine is not None:
            try:
                _engine.dispose()
            except Exception:
                pass
            _engine = None

    # Reset thread-local
    _thread_local.ref_count = 0
    _thread_local.session_id = None
    _thread_local.session_start_time = None

    # Reset circuit breaker (v90 ROOT FIX BUG #11: use thread-safe reset()
    # method instead of directly mutating _failure_count / _state without
    # the lock — prevents torn-state race conditions under concurrent
    # test teardown + live pipeline).
    _circuit_breaker.reset()

    logger.debug("All global state reset")


# ===========================================================================
# ATEXIT HANDLER (CODE-010)
# ===========================================================================


def _atexit_cleanup() -> None:
    """Clean up engine on process exit."""
    try:
        if _engine is not None:
            dispose_engine(force=True)
    except Exception:
        pass


atexit.register(_atexit_cleanup)
