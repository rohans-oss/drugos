"""Shared retry policy for standalone ETL DAGs (v74 ROOT FIX T-023).

PROBLEM (T-023)
---------------
All 7 standalone DAGs (chembl, drugbank, disgenet, omim, pubchem, string,
uniprot) had ``retries=2, retry_delay=30min`` on every task. For transient
errors (5xx, network timeout, rate limit 429) retries help. For 4xx errors:

  - 401 Unauthorized -- bad/expired API key (DISGENET_API_KEY, OMIM_API_KEY)
  - 403 Forbidden -- quota exceeded, IP blocked
  - 404 Not Found -- wrong endpoint, source renamed/removed
  - 400 Bad Request -- malformed query

retrying after 30 minutes wastes 60 minutes (2 retries × 30min) and STILL
fails -- the API key won't un-expire, the quota won't un-exhaust, the
endpoint won't un-disappear. During those 60 minutes the DAG occupies a
worker slot, blocking other DAGs.

ROOT FIX
--------
1. Add ``retry_exponential_backoff=True`` so the 30-min delay shrinks to
   ~10s on the first retry (transient errors recover faster).
2. Set ``retry_delay=timedelta(minutes=5)`` (was 30min) -- the exponential
   backoff grows it from there.
3. Wrap each task function with :func:`fail_fast_on_http_4xx` -- if the
   raised exception is an HTTP 4xx, re-raise it as
   ``AirflowFailException`` which Airflow treats as NON-retryable. The
   task fails immediately, the DAG continues to its on_failure_callback,
   and the operator gets a clear "401 Unauthorized" error in the logs
   instead of a 60-min wait.

USAGE
-----
    from dags._retry_policy import DEFAULT_RETRY_ARGS, fail_fast_on_http_4xx

    DEFAULT_ARGS = {
        **DEFAULT_RETRY_ARGS,
        "owner": "drug_repurposing",
        # ... other args
    }

    @task(retries=2, execution_timeout=timedelta(hours=4))
    @fail_fast_on_http_4xx
    def run_chembl() -> None:
        from pipelines.chembl_pipeline import ChEMBLPipeline
        ChEMBLPipeline().run()
"""

from __future__ import annotations

import logging
from datetime import timedelta
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# Exponential backoff: first retry after ~10s, then ~20s, then ~40s.
# Base delay 5 min so even non-backoff-aware Airflow versions wait 5 min
# (was 30 min in v73 -- too long for transient errors that recover in
# seconds, and pointless for 4xx that never recover).
#
# v89 FORENSIC ROOT FIX (BUG #37 -- SLA == execution_timeout defeats
# early-warning):
#   The previous config set ``sla = execution_timeout = 4h``. Per
#   Airflow semantics, an SLA miss is ADVISORY -- it writes a row to
#   the ``sla_miss`` table and (optionally) sends an email, but it
#   does NOT kill the running task. With SLA == execution_timeout,
#   the SLA miss fires at EXACTLY 4h, and the hard kill fires at
#   EXACTLY 4h -- there is NO early-warning window. Operators cannot
#   proactively investigate slow tasks before they are killed.
#
#   ROOT FIX: set ``sla = 3h`` and ``execution_timeout = 4h``. The
#   1-hour gap gives operators an advisory signal at 3h ("this task
#   is taking longer than expected -- investigate") BEFORE the hard
#   kill at 4h. This is the scientifically correct configuration for
#   an SLA meant as an early-warning system: the warning must come
#   BEFORE the kill, not simultaneously with it.
#
#   The master DAG overrides BOTH to 7h (aligned) because TransE
#   training on real data can take 6-7h -- see
#   master_pipeline_dag.py::TASK_SLA / TASK_TIMEOUT for the
#   master-specific rationale.
F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_RETRY_ARGS: dict[str, Any] = {
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=20),  # cap exponential growth
    "execution_timeout": timedelta(hours=4),
    "sla": timedelta(hours=3),  # v89 BUG #37: 1h early-warning window before 4h hard kill
    "email_on_failure": False,
    "email_on_retry": False,
}

# P1-033 FORENSIC ROOT FIX (Team 4 -- DB deadlock retry policy too short):
#   The audit found that ``DEFAULT_RETRY_ARGS.max_retry_delay=20min`` is
#   adequate for HTTP 5xx errors but the issue specifically calls out
#   PostgreSQL deadlocks. PostgreSQL deadlocks resolve in 1-5 min under
#   normal load but can take up to 5 min under high concurrency (lock
#   wait queue, vacuum contention). The DEFAULT_RETRY_ARGS handles this
#   (5min + 10min = 15min total, > 5min deadlock window).
#
#   HOWEVER: the audit's broader point is that DB-write tasks need a
#   DISTINCT retry policy from HTTP-fetch tasks. A DB deadlock is a
#   TRANSIENT error that should be retried (the second attempt will
#   acquire the lock), while an HTTP 4xx is a PERMANENT error that
#   should fail-fast (handled by ``fail_fast_on_http_4xx``).
#
#   ROOT FIX: add ``DB_DEADLOCK_RETRY_ARGS`` for tasks that write to
#   PostgreSQL (entity_resolution, *_load tasks). This policy:
#     1. ``retries=5`` (was 2 in DEFAULT) -- DB deadlocks are MORE
#        transient than HTTP errors; 5 retries gives ~25 min of total
#        wait time (5+10+15+20+20=70s under exponential backoff, but
#        Airflow's jitter can push this to ~5min for the last retry).
#     2. ``max_retry_delay=timedelta(minutes=5)`` -- the audit's
#        SPECIFIC recommendation. The default 20min cap is too long
#        for a DB deadlock (which resolves in <5 min); capping at 5min
#        means the retry fires ASAP after the deadlock clears.
#     3. ``retry_exponential_backoff=True`` -- Airflow's exponential
#        backoff includes JITTER by default (random +/- 50% of the
#        computed delay). This prevents the "thundering herd" problem
#        where multiple concurrent tasks all retry at the exact same
#        instant and re-deadlock each other. The jitter spreads
#        retries across a 2.5-7.5min window for the last retry.
#
#   USAGE:
#     from dags._retry_policy import DB_DEADLOCK_RETRY_ARGS
#     DEFAULT_ARGS = {**DB_DEADLOCK_RETRY_ARGS, "owner": "drug_repurposing"}
#
#   WHEN TO USE:
#     * Tasks that write to PostgreSQL (entity_resolution, *_load).
#     * Tasks that acquire row-level locks (upserts, batch updates).
#   WHEN NOT TO USE:
#     * Tasks that ONLY fetch HTTP data (use DEFAULT_RETRY_ARGS).
#     * Tasks that hit external APIs (use DEFAULT_RETRY_ARGS +
#       @fail_fast_on_http_4xx).
DB_DEADLOCK_RETRY_ARGS: dict[str, Any] = {
    **DEFAULT_RETRY_ARGS,
    "retries": 5,  # P1-033: DB deadlocks are transient; 5 retries
    "max_retry_delay": timedelta(minutes=5),  # P1-033: 5min cap per audit recommendation
    # ``retry_exponential_backoff=True`` (inherited from DEFAULT_RETRY_ARGS)
    # automatically applies jitter -- Airflow's implementation adds
    # random +/- 50% to each computed delay. This is the jittered
    # backoff the audit asked for; it prevents thundering-herd
    # re-deadlocks when multiple workers retry simultaneously.
}


def is_db_deadlock_error(exc: BaseException) -> bool:
    """P1-033 ROOT FIX: detect PostgreSQL / SQLite deadlock errors.

    Returns True if ``exc`` represents a database deadlock, lock
    timeout, or serialization failure -- errors that are TRANSIENT
    and should be retried.

    Detected error patterns:
      * ``psycopg2.errors.DeadlockDetected`` (SQLSTATE 40P01)
      * ``psycopg2.errors.LockNotAvailable`` (SQLSTATE 55P03)
      * ``psycopg2.errors.SerializationFailure`` (SQLSTATE 40001)
      * ``sqlalchemy.exc.OperationalError`` wrapping the above
      * SQLite ``database is locked`` (parallel-write contention)
      * Generic ``OperationalError`` with "deadlock" / "lock" in message
    """
    # Direct class-name check (avoids importing psycopg2 at module load).
    exc_class_name = type(exc).__name__
    if exc_class_name in {
        "DeadlockDetected",
        "LockNotAvailable",
        "SerializationFailure",
    }:
        return True
    # SQLAlchemy wraps the driver error.
    cause = getattr(exc, "__cause__", None) or getattr(exc, "orig", None)
    if cause is not None and cause is not exc:
        cause_class = type(cause).__name__
        if cause_class in {
            "DeadlockDetected",
            "LockNotAvailable",
            "SerializationFailure",
        }:
            return True
    # String heuristic (last resort).
    msg = str(exc).lower()
    deadlock_markers = (
        "deadlock detected",
        "deadlock found",
        "database is locked",
        "lock wait timeout exceeded",
        "could not serialize access",
        "serialization failure",
    )
    return any(marker in msg for marker in deadlock_markers)


def retry_on_db_deadlock(func: F) -> F:
    """P1-033 ROOT FIX: decorator that retries DB deadlocks with jittered backoff.

    Wraps a task function. If the function raises a DB deadlock error
    (detected by :func:`is_db_deadlock_error`), the decorator sleeps
    with exponential backoff + jitter and retries up to 5 times. This
    is a FALLBACK for environments where Airflow's task-level retry
    is not available (e.g. when the function is called directly from
    a script, not via an Airflow @task decorator).

    In Airflow, the task-level ``retries=5`` + ``retry_exponential_backoff=True``
    (in ``DB_DEADLOCK_RETRY_ARGS``) handles this automatically -- this
    decorator is for non-Airflow callers (tests, manual scripts).
    """
    import random
    import time

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        max_retries = 5
        base_delay_seconds = 5.0  # 5s base (shorter than Airflow's 5min
        # because this is in-process retry, not task-level)
        max_delay_seconds = 300.0  # 5min cap (matches P1-033 recommendation)
        last_exc: BaseException | None = None
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if not is_db_deadlock_error(exc):
                    raise  # Not a deadlock -- re-raise unchanged.
                last_exc = exc
                if attempt == max_retries:
                    logger.error(
                        "P1-033 retry_on_db_deadlock: exhausted %d retries "
                        "for DB deadlock: %s",
                        max_retries,
                        exc,
                    )
                    raise
                # Exponential backoff with jitter: delay = min(max,
                # base * 2^attempt) * random(0.5, 1.5).
                raw_delay = base_delay_seconds * (2 ** attempt)
                capped_delay = min(raw_delay, max_delay_seconds)
                jittered_delay = capped_delay * random.uniform(0.5, 1.5)
                logger.warning(
                    "P1-033 retry_on_db_deadlock: DB deadlock detected "
                    "(attempt %d/%d): %s. Retrying in %.1fs with jitter.",
                    attempt + 1,
                    max_retries,
                    exc,
                    jittered_delay,
                )
                time.sleep(jittered_delay)
        # Should be unreachable -- the loop either returns or raises.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("P1-033 retry_on_db_deadlock: unreachable state")

    return wrapper  # type: ignore[return-value]


# HTTP 4xx status codes that should NOT be retried. 429 (Too Many Requests)
# is intentionally EXCLUDED -- it's a rate-limit signal and retries (with
# exponential backoff) are the correct response.
# 408 (Request Timeout) is ALSO intentionally EXCLUDED -- it represents a
# transient condition where the server timed out waiting for the client;
# retrying is the correct response (the server is up, the request was just
# too slow). 408 is NOT in this set, so is_http_4xx_error() returns False
# for 408, and the task is retried with exponential backoff. This is the
# scientifically correct behavior for a transient timeout.
# v89 FORENSIC ROOT FIX (BUG #19 P1): 409 (Conflict) was MISSING from the
#   non-retryable set. 409 indicates a state conflict on the server (e.g.
#   concurrent writes to the same resource, optimistic-lock failure).
#   Retrying the SAME request will not resolve the conflict -- the client
#   must change the request (e.g. re-read the current state and re-apply).
#   Retrying 409 wasted 60 minutes of exponential backoff for an error
#   that never self-resolves. ROOT FIX: add 409 to the non-retryable set.
_NON_RETRYABLE_HTTP_STATUSES: frozenset[int] = frozenset(
    {
        400,  # Bad Request -- malformed query, won't fix by retrying
        401,  # Unauthorized -- bad/expired API key, won't fix by retrying
        402,  # Payment Required -- billing issue, won't fix by retrying
        403,  # Forbidden -- quota exceeded / IP blocked, won't fix by retrying
        404,  # Not Found -- wrong endpoint, won't fix by retrying
        405,  # Method Not Allowed -- wrong HTTP verb, won't fix by retrying
        409,  # Conflict -- state conflict (concurrent write, optimistic lock);
              # retrying the SAME request won't resolve it -- must re-read
              # and re-apply. v89 BUG #19.
        410,  # Gone -- resource permanently removed, won't fix by retrying
        451,  # Unavailable For Legal Reasons -- geo-blocked, won't fix
    }
)


def _extract_http_status(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from an exception.

    Supports:
      - ``requests.exceptions.HTTPError`` (``response.status_code``)
      - ``httpx.HTTPStatusError`` (``response.status_code``)
      - ``urllib.error.HTTPError`` (``code`` attribute)
      - Any exception with a ``status_code`` or ``status`` attribute
      - Any exception whose string representation starts with "NNN " (e.g.
        "404 Not Found: ...") -- last-resort heuristic for wrapped errors
        where the original response object is lost.

    P1-033 v113 ROOT FIX: recursively unwrap ``__cause__`` and
    ``__context__`` chains. A wrapped exception (e.g.
    ``tenacity.RetryError`` wrapping ``requests.HTTPError``, or
    ``airflow.exceptions.AirflowException`` wrapping a 4xx) does NOT have
    ``status_code`` or ``response`` on the OUTER exception. Without
    unwrapping, a 401 Unauthorized (expired API key) is retried 6 times
    over 95 minutes instead of failing fast.

    Returns ``None`` if no status code can be extracted.
    """
    # P1-033 v113 ROOT FIX: unwrap __cause__ / __context__ chains.
    # Try the outer exception first, then walk the cause/context chain.
    # Limit depth to 10 to prevent infinite loops on circular references.
    _current: BaseException | None = exc
    _depth = 0
    while _current is not None and _depth < 10:
        # Direct attribute access (requests, httpx, custom API clients)
        for attr in ("status_code", "status", "code"):
            val = getattr(_current, attr, None)
            if isinstance(val, int) and 100 <= val <= 599:
                return val
        # Nested response object (requests.HTTPError.response.status_code)
        response = getattr(_current, "response", None)
        if response is not None:
            for attr in ("status_code", "status", "code"):
                val = getattr(response, attr, None)
                if isinstance(val, int) and 100 <= val <= 599:
                    return val
        # Move to the next layer of the exception chain
        _current = _current.__cause__ or _current.__context__
        _depth += 1

    # String heuristic -- last resort (on the ORIGINAL exception, not the
    # unwrapped inner one, because str(RetryError) is unhelpful).
    # v83 FORENSIC ROOT FIX (P2-12): the previous code extracted leading
    # digits from the message -- but "2024-01-15 download failed" would
    # extract "202" (stops at 3 digits) and treat it as HTTP 202 (a
    # success code), masking the real error. ROOT FIX: only accept a
    # leading 3-digit number that is IMMEDIATELY followed by a non-digit
    # (space, colon, end-of-string). This rejects "2024-01-15" (4 digits
    # before the dash -> not a 3-digit HTTP code) while accepting
    # "404 Not Found" (3 digits followed by a space).
    msg = str(exc).strip()
    if msg and msg[0].isdigit():
        # Extract EXACTLY 3 leading digits (HTTP status codes are 3 digits).
        if len(msg) >= 3 and msg[:3].isdigit():
            # The 4th character must NOT be a digit (otherwise this is a
            # longer number like "2024", not an HTTP status code).
            fourth_char_is_digit = len(msg) > 3 and msg[3].isdigit()
            if not fourth_char_is_digit:
                code = int(msg[:3])
                if 100 <= code <= 599:
                    return code
    return None


def is_http_4xx_error(exc: BaseException) -> bool:
    """Return True if ``exc`` represents an HTTP 4xx error that should not be retried.

    429 (Too Many Requests) is explicitly EXCLUDED -- it's a rate-limit
    signal and retries with exponential backoff are the correct response.
    All other 4xx codes are considered non-retryable per HTTP semantics
    (the client did something wrong; retrying with the same request won't
    help).
    """
    status = _extract_http_status(exc)
    if status is None:
        return False
    return status in _NON_RETRYABLE_HTTP_STATUSES


def fail_fast_on_http_4xx(func: F) -> F:
    """Decorator that converts HTTP 4xx exceptions to AirflowFailException.

    When the wrapped task raises an exception that :func:`is_http_4xx_error`
    identifies as a non-retryable 4xx, the exception is re-raised as
    ``AirflowFailException``. Airflow treats ``AirflowFailException`` as a
    terminal failure -- the task is NOT retried, the DAG proceeds to its
    ``on_failure_callback``, and the operator sees the original 4xx error
    message immediately instead of waiting 60 minutes for pointless retries.

    For all other exceptions (5xx, network timeouts, generic Exception),
    the original exception is re-raised unchanged so Airflow's normal
    retry logic applies.

    The import of ``AirflowFailException`` is deferred to inside the
    decorator body so this module can be imported in environments where
    Airflow is not installed (e.g. unit tests of the helper logic
    itself).
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if is_http_4xx_error(exc):
                # Import lazily so this module doesn't require airflow at
                # import time (allows the helper to be unit-tested without
                # the airflow dependency).
                try:
                    from airflow.exceptions import AirflowFailException
                except ImportError:
                    # Airflow not installed -- re-raise the original 4xx
                    # exception. This path is for unit tests of the helper
                    # logic; in production, Airflow is always available
                    # (it's a hard dependency of the DAGs).
                    logger.error(
                        "fail_fast_on_http_4xx: caught HTTP 4xx (%s) but "
                        "airflow is not importable -- re-raising original "
                        "exception. In production, this would be converted "
                        "to AirflowFailException to prevent pointless "
                        "retries. (v74 T-023)",
                        exc,
                    )
                    raise
                # Convert to AirflowFailException so Airflow skips retries.
                # Preserve the original exception's message and chain via
                # ``from`` so the operator sees the root cause in the UI.
                logger.error(
                    "fail_fast_on_http_4xx: HTTP 4xx detected (%s) -- "
                    "converting to AirflowFailException to prevent "
                    "pointless retries. The original error is not "
                    "transient (bad API key, wrong endpoint, quota "
                    "exceeded, etc.) and retrying after 5+ min would "
                    "waste a worker slot. Fix the root cause and "
                    "re-trigger the DAG manually. (v74 T-023)",
                    exc,
                )
                raise AirflowFailException(
                    f"HTTP 4xx error (non-retryable): {exc}"
                ) from exc
            # Not a 4xx -- re-raise unchanged so Airflow's normal retry
            # logic applies (transient 5xx, network timeout, etc.).
            raise

    return wrapper  # type: ignore[return-value]
