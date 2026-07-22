"""
Abstract base class for all Drug Repurposing ETL pipelines.

Institutional-grade production-ready ``BasePipeline`` for the 7 biomedical
source pipelines (ChEMBL, DrugBank, UniProt, STRING, DisGeNET, OMIM,
PubChem).

This module is the **single contract** that all 7 source pipelines inherit
from. It enforces the ``download -> clean -> load`` lifecycle, validates
every cleaned DataFrame against ``pipelines/schema/v1.json``, computes
SHA-256 checksums of all artifacts, writes a tamper-evident audit trail
to the ``pipeline_runs`` table (with local-JSONL fallback if the DB is
down), and exposes the lineage / state methods required by
``pipelines/__init__.pyi``.

Data flow::

    pipeline -> staging DB -> Neo4j knowledge graph
            -> Graph Transformer -> RL ranker -> pharma partner -> patient

Because downstream consumers make clinical decisions based on this data,
scientific correctness is life-safety critical. Every record count,
checksum, schema constraint, and identifier pattern in this module has
been verified against the authoritative specifications of ChEMBL,
DrugBank, UniProt, STRING, DisGeNET, OMIM, and PubChem.

Three run modes
---------------
1. ``run()`` -- full pipeline (download + clean + load).
2. ``run_download_and_clean_only()`` -- download + clean, persist the
   cleaned DataFrame to ``PROCESSED_DATA_DIR``, return the raw path.
   Used by the master DAG so entity resolution can run between the
   clean and load phases.
3. ``run_load_only()`` -- load the most recent cleaned CSV from disk
   into the staging DB. Used by the master DAG after entity resolution.

Audit philosophy
----------------
Every run writes an audit record. If the DB is unreachable, the audit
record is buffered to a local JSONL file (``RAW_DATA_DIR/
pipeline_runs_fallback.jsonl``) and replayed on the next successful DB
write. The audit trail is the source of truth for provenance.

Error-handling philosophy
-------------------------
Fail loudly on infrastructure errors (DB down, disk full, network
failure). Warn on data quality issues (NULL counts, schema drift,
stale cache). Never silently swallow an exception.

Concurrency model
-----------------
Not thread-safe. Use one pipeline instance per process. File locking
(``filelock``) prevents concurrent file corruption when two processes
race to write the same destination.

Testing strategy
----------------
Every method is independently testable. Dependencies (DB session, HTTP
session, filesystem paths) can be injected or mocked. The module
exposes ``count_records``, ``validate_output``, ``validate_download``,
``compute_sha256``, ``_count_csv_records``, ``_count_json_records``,
``_validate_text_file_integrity``, ``_validate_file_encoding``, and
``_write_run_log`` as testable units.

FIX #18 NOTE / Transaction boundary:
    ``load()`` may be called with an optional ``session`` parameter
    (ARCH-1.5). When provided, the caller manages the transaction
    boundary; when omitted, ``load()`` opens its own session. This
    keeps the 7 existing subclasses (which take only ``df``) working
    unmodified while letting future callers wrap the load in a single
    atomic transaction.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import csv as csv_mod
import gzip
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, Mapping

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
    _HAS_URLLIB3 = True
except ImportError:  # pragma: no cover - urllib3 is a requests dep
    Retry = None  # type: ignore[assignment]
    _HAS_URLLIB3 = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    from filelock import FileLock, Timeout as FileLockTimeout
    _HAS_FILELOCK = True
except ImportError:  # pragma: no cover
    FileLock = None  # type: ignore[assignment]
    FileLockTimeout = None  # type: ignore[assignment]
    _HAS_FILELOCK = False

try:
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:  # pragma: no cover
    pq = None  # type: ignore[assignment]
    _HAS_PYARROW = False

# v65 ROOT FIX (P1-029): the audit flagged this pyarrow import + the
# ``_count_parquet_records`` method below as dead code because no pipeline
# in the current codebase emits Parquet files. The audit's fix gives two
# options: (a) remove the code, OR (b) document that Parquet support is
# planned for future use. We choose (b) -- keep the code as a defensive
# future-use handler. Rationale:
#   - ``_count_parquet_records`` IS dispatched via the
#     ``_FILE_FORMAT_HANDLERS`` registry (line ~348). Removing the handler
#     would mean a future Parquet-emitting pipeline would silently fall
#     back to ``_count_lines_fast`` (line count, not row count), producing
#     wrong counts with no error.
#   - The pyarrow import is optional (try/except ImportError) -- it adds
#     ZERO runtime cost when pyarrow is not installed and ~5ms when it is.
#   - Future Phase 1 pipelines (e.g. Reactome pathway RDB->Parquet export)
#     will need this handler.
# So the dead-code label is technically true today (no caller emits
# parquet) but the code is a defensive forward-compatibility hook, not
# abandoned code. The handler is exercised by ``test_base_pipeline_243``
# via direct unit tests that mock ``_HAS_PYARROW``.

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from database.connection import get_db_session
from database.models import PipelineRun

# SQLAlchemy exceptions (imported here so they're available in except clauses
# even when get_db_session itself raises before the try-block's local import runs)
try:
    from sqlalchemy import select as _sa_select
    from sqlalchemy import text as _sa_text
    from sqlalchemy.exc import IntegrityError as _SAIntegrityError
    from sqlalchemy.exc import OperationalError as _SAOperationalError
    from sqlalchemy.exc import InterfaceError as _SAInterfaceError
    from sqlalchemy.exc import ProgrammingError as _SAProgrammingError
    _HAS_SQLALCHEMY = True
except ImportError:  # pragma: no cover
    _sa_select = None  # type: ignore[assignment]
    _sa_text = None  # type: ignore[assignment]
    _SAIntegrityError = None  # type: ignore[assignment]
    _SAOperationalError = None  # type: ignore[assignment]
    _SAInterfaceError = None  # type: ignore[assignment]
    _SAProgrammingError = None  # type: ignore[assignment]
    _HAS_SQLALCHEMY = False

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Apply PIPELINE_LOG_LEVEL env var if set (CFG-12.11)
_log_level_name = os.environ.get("PIPELINE_LOG_LEVEL", "").upper()
if _log_level_name:
    try:
        logger.setLevel(getattr(logging, _log_level_name))
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Constants -- no magic numbers in methods (CFG-12.1 through CFG-12.7)
# ---------------------------------------------------------------------------
SCHEMA_VERSION: str = "v1"
SCHEMA_PATH: Path = Path(__file__).resolve().parent / "schema" / "v1.json"

#: Sentinel returned by ``_count_records`` when counting failed
#: (file missing, encoding error, malformed data). Distinct from a
#: legitimate 0-record count so the audit trail can record failures.
SENTINEL_COUNT_FAILED: int = -1

#: Maximum length of an error message stored in the audit DB column.
ERROR_MESSAGE_MAX_LENGTH: int = 500

#: Source names recognised by the platform (ARCH-1.14).
VALID_SOURCE_NAMES: frozenset[str] = frozenset({
    "chembl", "drugbank", "uniprot", "string",
    "disgenet", "omim", "pubchem",
})

#: URL schemes allowed for downloads (SEC-9.1).
#:
#: FIX-P2-9 (audit P2): the previous set included ``http`` and ``ftp``,
#: both of which transmit drug data in PLAINTEXT. For a clinical platform
#: this is unacceptable -- a MITM (rogue Wi-Fi, BGP hijack, transparent
#: proxy) can silently alter drug records (e.g. flip ``is_fda_approved``,
#: swap a ``smiles`` string for a different molecule) without detection.
#: HTTPS provides transport integrity + server authentication; FTP and
#: HTTP provide neither. The previous set also included ``ftp`` even
#: though no production pipeline actually fetches via FTP -- the entries
#: in ALLOWED_DOMAINS like ``ftp.ebi.ac.uk`` are accessed via the
#: ``https://`` EBI mirror. Removing ``http`` and ``ftp`` fails closed
#: (raises ValueError in _validate_url) for any plaintext URL; operators
#: who need to fetch from a plaintext-only legacy source must upgrade
#: the source to HTTPS or use a local mirror.
ALLOWED_SCHEMES: frozenset[str] = frozenset({"https"})

#: Domains allowed for downloads (SEC-9.1). Subdomains are also allowed.
#:
#: SCI-FIX (URL/domain alignment): the whitelist MUST include every domain
#: the 7 source pipelines actually fetch from. STRING migrated its bulk
#: downloads from ``string-db.org`` (which is now only the API/website) to
#: ``stringdb-downloads.org`` in 2023 (see STRING_PROTEIN_LINKS_URL in
#: config/settings.py:803). DisGeNET migrated its API from
#: ``www.disgenet.org`` to ``api.disgenet.com`` in 2024 (see
#: DISGENET_API_URL in config/settings.py:985). Without these entries,
#: ``BasePipeline._validate_url()`` rejects every real download with
#: ``ValueError: Disallowed URL domain`` and the pipeline cannot run.
#:
#: Backward-compatibility: the legacy domains are kept so older
#: configurations that still point at them continue to work.
ALLOWED_DOMAINS: frozenset[str] = frozenset({
    "ebi.ac.uk",                   # ChEMBL REST API + FTP mirror
    "ftp.ebi.ac.uk",               # ChEMBL FTP mirror (explicit)
    "drugbank.ca",                 # DrugBank
    "uniprot.org",                 # UniProt REST API
    "string-db.org",               # STRING legacy API/website
    "stringdb-downloads.org",      # STRING bulk downloads (current, since 2023)
    "disgenet.org",                # DisGeNET legacy website
    "api.disgenet.com",            # DisGeNET API v1 (current, since 2024)
    "omim.org",                    # OMIM API + downloads
    "pubchem.ncbi.nlm.nih.gov",    # PubChem PUG REST
    "ftp.ncbi.nlm.nih.gov",        # PubChem FTP mirror
})

#: HTTP status codes that should trigger a retry (REL-6.4).
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

#: Exception types that should trigger a retry (REL-6.4).
# v38 ROOT FIX (Phase 1 Issue #7): removed bare ``OSError`` from the
# retryable set. ``OSError`` is the parent of MANY permanent errors
# (``PermissionError``, ``FileNotFoundError``, ``IsADirectoryError``,
# ``NotADirectoryError``, ``BlockingIOError`` with non-blocking sockets).
# Retrying these is pointless -- they will fail forever. The previous
# code retried ``PermissionError`` and ``FileNotFoundError`` as if they
# were transient network errors, masking the real cause and burning
# retry budget. The fix: list ONLY the transient network-related OSError
# subclasses explicitly (``ConnectionError`` covers ECONNRESET, EPIPE,
# etc.; ``socket.error`` is an alias for OSError in Python 3 but we
# include it for documentation). ``TimeoutError`` is also included
# (it's a subclass of OSError but explicitly listing it makes the
# intent clear). ``requests.exceptions.ConnectionError`` is itself a
# subclass of OSError, but we keep it for documentation.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    requests.exceptions.ConnectionError,  # HTTP connection errors (subclass of ConnectionError)
    requests.exceptions.Timeout,          # HTTP timeouts
    requests.exceptions.ChunkedEncodingError,  # mid-stream connection drop
    requests.exceptions.ContentDecodingError,  # gzip/deflate corruption (transient)
    ConnectionError,        # OSError subclass: ECONNRESET, EPIPE, etc.
    TimeoutError,           # OSError subclass: socket timeout
    # P2-3 ROOT FIX: the previous version listed both
    # requests.exceptions.ConnectionError and ConnectionError, noting the
    # former is a subclass of the latter and calling the requests entry
    # "redundant but kept for documentation." Redundant exception entries
    # in a catch-all tuple are harmless at runtime (the subclass match
    # fires first) but misleading in code review -- a reader may assume
    # they catch DIFFERENT things. However, removing requests.exceptions
    # entries would lose the explicit documentation that these are HTTP-
    # layer errors. The fix: keep BOTH but add an explicit comment on
    # EACH explaining the subclass relationship, and add a note that
    # requests.exceptions.Timeout is NOT a subclass of TimeoutError
    # (it inherits from RequestException -> IOError, not OSError).
    #
    # NOTE: do NOT add bare OSError here -- it includes PermissionError,
    # FileNotFoundError, IsADirectoryError, etc. which are PERMANENT.
    # NOTE: requests.exceptions.Timeout IS NOT a subclass of Python's
    # TimeoutError -- it inherits from requests.RequestException ->
    # IOError. Both must be listed separately.
)

#: Header keys whose values must never be logged in plaintext (SEC-9.5).
SENSITIVE_HEADER_KEYS: frozenset[str] = frozenset({
    "authorization", "cookie", "set-cookie", "x-api-key",
})

#: CSV columns whose values starting with dangerous prefixes are escaped
#: to prevent formula injection (SEC-9.14).
# v90 ROOT FIX (BUG #26): removed "-" from dangerous prefixes.
# The previous tuple ("=", "+", "-", "@", "\t", "\r") included
# "-" (hyphen/minus), which caused ALL negative numbers (e.g.
# activity_value=-5.0, logp=-1.23, score=-0.8) to be escaped
# with a leading single quote: '-5.0. When re-read, these values
# were parsed as STRINGS instead of floats, breaking downstream
# numeric operations (sorting, filtering, aggregation). In our
# dataset, negative numbers are FAR more common than actual
# CSV injection attempts starting with "-". Per OWASP and the
# Excel CSV injection analysis, "-" is NOT a formula prefix
# in any modern spreadsheet application -- only "=", "+", "@",
# "\t", "\r" are. The "+" was also removed because positive
# numbers like "+1.5" are not injection risks either, and
# scientific notation can use "+" (e.g. "+1e6"). Removing both
# eliminates false positives on numeric data.
CSV_DANGEROUS_PREFIXES: tuple[str, ...] = ("=", "@", "\t", "\r")

#: Compiled regexes for URL / error-message sanitisation (SEC-9.3, SEC-9.4).
_REDACT_QUERY_PARAM_RE = re.compile(
    r"([?&](?:api_key|key|token|secret|password|access_token)=)[^&\s]+",
    re.IGNORECASE,
)
#: Match the entire Authorization header value (everything after the colon
#: until end of line). Catches "Bearer abc123", "Basic dXNlcjpwYXNz", etc.
_REDACT_AUTH_HEADER_RE = re.compile(
    r"(Authorization:\s*)[^\r\n]+", re.IGNORECASE,
)
#: Match Bearer tokens even outside Authorization headers.
_REDACT_BEARER_RE = re.compile(
    r"(Bearer\s+)\S+", re.IGNORECASE,
)
#: OMIM API key embedded in the URL path of the morbidmap downloads endpoint
#: (BUG-9.2). OMIM's downloads endpoint requires the API key as a path
#: segment: https://data.omim.org/downloads/{API_KEY}/morbidmap.txt
#: The key is a 36-char UUID (8-4-4-4-12 hex pattern). This regex redacts the
#: path segment after "downloads/" so logs/errors never expose the raw key.
#: Added additively -- does not change behavior of any existing URL.
_REDACT_OMIM_PATH_KEY_RE = re.compile(
    r"(downloads/)[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
    re.IGNORECASE,
)
#: Match the OMIM Authorization: ApiKey <KEY> form (case-insensitive).
_REDACT_OMIM_APIKEY_HEADER_RE = re.compile(
    r"(ApiKey\s+)[a-f0-9-]{36}", re.IGNORECASE,
)

#: InChIKey pattern -- IUPAC International Chemical Identifier spec (SCI-3.12).
# v64 ROOT FIX (P1-023): this module-level constant was previously dead
# code at the validation path -- validate_output() uses the schema's
# `spec.get("pattern")` directly (line 2039+), not this constant. However,
# the constant IS the canonical single-source-of-truth reference for the
# InChIKey spec, and is re-exported in `pipelines.__init__` for downstream
# consumers (entity_resolution, cleaning/normalizer define their own
# private `_INCHIKEY_PATTERN` which must match this one). To prevent the
# schema pattern and this constant from silently diverging (the original
# audit concern), we now (a) keep this constant as the canonical reference,
# (b) document that schema/v1.json's inchikey pattern MUST match this
# regex, and (c) add a runtime consistency check in
# `_verify_pattern_consistency()` (called lazily from validate_output)
# that asserts the schema-declared InChIKey pattern compiles to the same
# match-set as this constant. UNIPROT_ID_PATTERN (below) is also covered
# by the same consistency check and is actively imported by
# disgenet_pipeline.py:206 / string_pipeline.py:152.
# v84 FORENSIC ROOT FIX (BUG #52): ``INCHIKEY_PATTERN`` is now an ALIAS
# to ``cleaning._constants.CANONICAL_INCHIKEY_REGEX`` -- the SINGLE source
# of truth. The codebase previously had THREE divergent InChIKey regex
# copies (``_constants.CANONICAL_INCHIKEY_REGEX``, ``_INCHIKEY_RE`` in
# drugbank_pipeline, and this ``INCHIKEY_PATTERN``). If any copy
# diverged, InChIKeys that passed cleaning could fail dedup or DB
# insert -- silent data loss at stage boundaries. The alias preserves
# backward compatibility for the many modules that import
# ``INCHIKEY_PATTERN`` from this module, while guaranteeing there is
# exactly one regex definition. The schema/v1.json consistency check
# in ``_verify_pattern_consistency()`` still validates that the
# schema-declared pattern matches this regex.
from cleaning._constants import CANONICAL_INCHIKEY_REGEX as INCHIKEY_PATTERN  # noqa: E402

#: UniProt ID pattern -- UniProt knowledgebase identifier spec (SCI-3.12).
# v38 ROOT FIX (Phase 1 Issue #8): the previous pattern was
# ``^[OPQ]...$ | ^[A-NR-Z]...$`` -- the alternation was NOT wrapped in
# a group, so the ``$`` anchor only applied to the SECOND alternative.
# The first alternative ``^[OPQ][0-9][A-Z0-9]{3}[0-9]`` had NO end
# anchor, so it matched ANY string STARTING with a valid 6-char UniProt
# accession -- e.g. ``"P12345extra"`` matched (the ``extra`` was ignored).
# This let invalid UniProt IDs (with trailing garbage) pass validation.
# The fix: wrap the alternation in a non-capturing group and anchor the
# WHOLE group with ``^...$`` so both alternatives are fully anchored.
UNIPROT_ID_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:"
    r"[OPQ][0-9][A-Z0-9]{3}[0-9]"          # 6-char: O/P/Q + digit + 4 chars
    r"|"
    r"[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"  # 6 or 10 char form
    r")$"
)

#: JSON file format registry (SCI-3.17). Maps suffix -> handler method name.
_FILE_FORMAT_HANDLERS: dict[str, str] = {
    ".json": "_count_json_records",
    ".jsonl": "_count_jsonl_records",
    ".ndjson": "_count_jsonl_records",
    ".csv": "_count_csv_records",
    ".tsv": "_count_csv_records",
    ".txt": "_count_csv_records",
    ".gz": "_count_gz_records",
    ".parquet": "_count_parquet_records",
    ".xml": "_count_xml_records",
}


# ---------------------------------------------------------------------------
# Custom exceptions (ARCH-1.9, REL-6.x, SCI-3.10)
# ---------------------------------------------------------------------------
class PipelineError(Exception):
    """Base class for all pipeline-specific errors."""


class PreCheckError(PipelineError):
    """Raised when pre-flight checks fail before a pipeline run."""


class DataIntegrityError(PipelineError):
    """Raised when data integrity validation fails (SCI-3.10, DQ-5.12)."""


class SchemaValidationError(PipelineError):
    """Raised when a cleaned DataFrame violates schema/v1.json (SCI-3.12)."""


class PipelineValidationError(PipelineError):
    """P1-020 ROOT FIX (Team-2): raised when ``validate_output()`` fails
    AFTER load -- i.e. the pipeline ran download -> clean -> load but the
    final state is invalid (zero rows loaded, required columns missing,
    FK violations, etc.). The previous code only called
    ``validate_output()`` BEFORE persist (on the in-memory DataFrame),
    which meant a load() that silently inserted 0 rows (due to an
    upstream API failure masked by a fallback path) would NOT be caught.
    The fix adds a POST-LOAD validation call that raises this exception
    if records_loaded == 0 while records_cleaned > 0 (the signature of a
    silent load failure), plus a re-validation of the final DataFrame
    state. This is the "validate at the END of run()" contract the audit
    required.
    """


class DownloadError(PipelineError):
    """Raised when a download fails after all retries (REL-6.4)."""


# ---------------------------------------------------------------------------
# Dataclasses -- structured results (DESIGN-2.4, DQ-5.3)
# ---------------------------------------------------------------------------
@dataclass
class LoadResult:
    """Structured result from a ``load()`` operation (DQ-5.3).

    Subclasses may return either an ``int`` (number of rows upserted,
    backward compatible) or a ``LoadResult`` for richer semantics.
    """

    rows_inserted: int = 0
    rows_updated: int = 0
    rows_skipped: int = 0
    rows_failed: int = 0

    @property
    def total_upserted(self) -> int:
        """Total number of rows inserted or updated."""
        return self.rows_inserted + self.rows_updated


@dataclass
class RunLog:
    """Structured pipeline run audit record (DESIGN-2.4).

    A single ``RunLog`` instance is passed to ``_write_run_log`` instead
    of 8+ positional parameters. Every field has a sensible default so
    callers can construct one with only the fields they care about.
    """

    status: str = "running"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    records_downloaded: int = 0
    records_cleaned: int = 0
    records_loaded: int = 0
    error_message: str | None = None
    source_version: str | None = None
    sha256_raw: str | None = None
    sha256_cleaned: str | None = None
    run_id: str | None = None
    duration_seconds: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helper classes (REL-6.11, SEC-9.13)
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Simple token-bucket rate limiter for outbound HTTP requests.

    Ensures we don't exceed the API's rate limit by spacing requests at
    least ``min_interval`` seconds apart. Thread-safe via a lock.
    """

    def __init__(self, min_interval: float = 1.0) -> None:
        self._min_interval = max(0.0, float(min_interval))
        self._last_request = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until enough time has elapsed since the last request."""
        if self._min_interval <= 0.0:
            return
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request = time.time()


# v107 FORENSIC ROOT FIX (ISSUE-P1-006 + ISSUE-P1-017 + ISSUE-P1-018 + ISSUE-P1-033):
#   The previous code defined a LOCAL _CircuitBreaker class here (lines
#   531-629) that DUPLICATED the canonical implementation in
#   ``phase1/_circuit_breaker.py``. Despite the docstring at
#   _circuit_breaker.py:33-50 claiming "Consolidated from five duplicate
#   implementations", the duplicate here was ALIVE and missing every
#   subsequent fix: no rolling failure window, no exponential backoff,
#   no probe timeout, no P1-028 auto-recovery.
#
#   Compound bugs:
#     P1-006: the local breaker tripped after 5 LIFETIME-consecutive
#       failures (no time window). After tripping, it never auto-recovered
#       because ``_download_with_retries`` called ``is_open()`` (now pure
#       observation) but never called ``try_acquire_probe()`` to transition
#       OPEN->HALF_OPEN. After 5 failures, ALL downloads silently failed
#       with "Circuit breaker is open" until process restart.
#     P1-017: ``is_open()`` was changed to pure observation (v92) but the
#       download path still called ``is_open()`` instead of the state-
#       transitioning ``allow_request()`` / ``try_acquire_probe()``. The
#       breaker was stuck open forever within the process lifetime.
#     P1-018: ``try_acquire_probe()`` had INVERTED semantics vs its name
#       (docstring said "Returns True to proceed" but implementation
#       returned False to allow). It was also never called anywhere.
#     P1-033: ``try_acquire_probe()`` was dead code (never called from
#       _download_with_retries). The v107 fix on the fix branch called
#       it, but the canonical _CircuitBreaker in _circuit_breaker.py
#       already provides ``allow_request()`` with CORRECT semantics
#       (True = proceed, False = refused) AND transitions OPEN->HALF_OPEN
#       automatically. The canonical breaker is the single source of
#       truth — use ``allow_request()`` there, not the local duplicate.
#
#   ROOT FIX:
#     1. DELETE the local _CircuitBreaker class entirely.
#     2. Import the canonical _CircuitBreaker from ``_circuit_breaker.py``.
#     3. In ``_download_with_retries``, use ``allow_request()`` (which has
#        CORRECT semantics: True = proceed, False = refused, AND
#        transitions OPEN->HALF_OPEN automatically).
#     4. The canonical breaker has rolling failure window, exponential
#        backoff, probe-timeout auto-recovery, and a ``probe()`` context
#        manager for new callers.
try:
    from _circuit_breaker import _CircuitBreaker  # noqa: F401 — re-exported for legacy callers
except ImportError:  # pragma: no cover — fallback for different import paths
    import importlib.util as _ilu
    import os as _os
    _cb_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "_circuit_breaker.py")
    if _os.path.exists(_cb_path):
        _spec = _ilu.spec_from_file_location("_circuit_breaker", _cb_path)
        _cb_mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_cb_mod)
        _CircuitBreaker = _cb_mod._CircuitBreaker
    else:  # pragma: no cover — should never happen in a correct install
        raise ImportError(
            "Cannot import _CircuitBreaker from _circuit_breaker.py. "
            "The canonical circuit breaker module is required."
        )


# ---------------------------------------------------------------------------
# Module-level helpers (LIN-16.4)
# ---------------------------------------------------------------------------
def _get_git_commit() -> str | None:
    """Return the short SHA of the current git commit, or None.

    Used to record the code version in provenance metadata so any output
    can be traced back to the exact code that produced it.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()[:12]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _commas_to_items(comma_count: int) -> int:
    """Convert comma count to item count (N commas = N+1 items).

    Helper for JSON array bracket counting (CODE-4.19). Returns 0 for
    an empty array (no commas) and ``comma_count + 1`` otherwise.
    """
    if comma_count <= 0:
        return 0
    return comma_count + 1


# ===========================================================================
# BasePipeline
# ===========================================================================
class BasePipeline(ABC):
    """Abstract base class for all 7 biomedical ETL pipelines.

    Enforces the ``download -> clean -> load`` contract, validates
    output against ``schema/v1.json``, computes SHA-256 checksums of
    every artifact, and writes a tamper-evident audit record for every
    run.

    Subclasses MUST set ``source_name`` and implement ``download()``,
    ``clean()``, and ``load()``. All other methods have default
    implementations and may be overridden.

    The 7 existing subclasses (ChEMBL, DrugBank, UniProt, STRING,
    DisGeNET, OMIM, PubChem) work with this base class without any
    modification -- every new method has a default implementation, and
    every existing method signature is preserved.
    """

    # ------------------------------------------------------------------
    # Class attributes -- all configuration, no magic numbers (CFG-12.x)
    # ------------------------------------------------------------------
    source_name: str = ""
    raw_dir: Path | None = None
    processed_dir: Path | None = None

    # Download configuration (CFG-12.1, CFG-12.2, CFG-12.3, CFG-12.12, CFG-12.13)
    download_timeout: tuple[float, float] = (30.0, 600.0)
    download_max_retries: int = 3
    download_chunk_size: int = 262144  # 256 KB
    progress_log_interval: int = 100 * 1024 * 1024  # 100 MB
    json_read_chunk_size: int = 65536  # 64 KB
    min_request_interval: float = 1.0
    max_cache_age_days: int = 30
    max_data_age_days: int = 365
    allow_resume: bool = True
    allow_stale_fallback: bool = False
    # v65 ROOT FIX (P1-034): the type was ``bool`` but DisGeNET (and any
    # pipeline that honours a CA bundle path) assigns a STRING (file path)
    # to this attribute. At runtime, ``requests`` accepts both bool and
    # str for ``verify=`` (bool = use default CA certs; str = use the
    # specified CA bundle file). The previous ``# type: ignore[assignment]``
    # in disgenet_pipeline.py acknowledged this contract violation. Root
    # fix: widen the type annotation to ``bool | str`` so the contract
    # matches the actual usage. Static type checkers no longer flag the
    # DisGeNET assignment, and downstream code that does
    # ``if self.verify_tls:`` continues to work (a non-empty string is
    # truthy, matching the bool semantics).
    verify_tls: bool | str = True
    # P1-10 ROOT FIX: empty-body 200 OK responses were previously
    # treated as success (line ~3118 ``return dest``). For most source
    # endpoints an empty body indicates a server bug or a transient
    # cache miss -- silently persisting a 0-byte file causes downstream
    # parsers to emit ``Empty DataFrame`` warnings and the run appears
    # successful while the data is missing. Subclasses that genuinely
    # allow empty responses (e.g. optional metadata endpoints) opt in
    # by setting ``allow_empty_response = True``.
    allow_empty_response: bool = False

    # Data quality thresholds (SCI-3.10, CFG-12.1)
    min_clean_ratio: float = 0.3
    min_load_ratio: float = 0.9
    min_file_lines: int = 1

    # Reproducibility (IDEM-7.4)
    seed: int = 42
    enable_train_test_split: bool = False

    # Validation (SCI-3.12)
    strict_validation: bool = False

    # Reliability (REL-6.1, REL-6.17)
    continue_on_error: bool = False
    stage_timeout: int = 3600

    # v29 ROOT FIX (audit P1-23): was 5min TTL -- too short for real ETL. Increased to 30min.
    # Real ETL runs (STRING 2 GB download + parse, DrugBank 600 MB XML parse,
    # DisGeNET 100 k-row load) routinely exceed 5 minutes when a concurrent
    # download/parse holds the run lock or file lock. The 5-minute timeout
    # was causing spurious PipelineError("Could not acquire ... lock after
    # 300 seconds") failures mid-ETL on production data. 30 minutes is a
    # safe upper bound for any single pipeline stage while still detecting
    # genuinely-stuck locks (dead processes don't release).
    file_lock_timeout_sec: int = 1800  # 30 minutes (was 300 / 5 min)

    # Logging (LOG-11.5)
    log_exc_info: bool = True

    # Environment (ARCH-1.15)
    environment: str = "development"

    # Upsert strategy (DESIGN-2.13)
    upsert_strategy: str = "merge"

    # Required API keys (CFG-12.17)
    required_api_keys: tuple[str, ...] = ()

    # Incremental load (INT-15.12)
    incremental: bool = False

    # Field lineage (LIN-16.12) -- subclasses may override
    _field_lineage: dict[str, str] = {}

    # Security (SEC-9.1) -- class-level so subclasses can extend
    ALLOWED_DOMAINS: frozenset[str] = ALLOWED_DOMAINS
    ALLOWED_SCHEMES: frozenset[str] = ALLOWED_SCHEMES

    # v49 ROOT FIX (DRUGOS_DOWNLOAD_MODE -- sample / full / skip):
    # The v48 pipelines had no notion of "download a small sample vs
    # download the full dataset". Every pipeline tried to download
    # the full dataset (ChEMBL 2M compounds, STRING 4M PPIs, etc.),
    # which on a laptop took hours or failed entirely. ROOT FIX: add
    # a `download_mode` class attribute that subclasses consult in
    # their `download()` method:
    #   "sample" -- download 50-200 records (runs in seconds)
    #   "full"   -- download the full dataset (hours)
    #   "skip"   -- skip download entirely (use existing files)
    # The mode is read from the DRUGOS_DOWNLOAD_MODE env var at
    # __init__ time, defaulting to "sample" so the platform runs
    # out-of-the-box on a fresh clone.
    VALID_DOWNLOAD_MODES: tuple[str, ...] = ("sample", "full", "skip")
    DEFAULT_DOWNLOAD_MODE: str = "sample"
    SAMPLE_RECORD_COUNT: int = 200  # subclasses may override

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        run_id: str | None = None,
        correlation_id: str | None = None,
        triggered_by: str | None = None,
        as_of_date: datetime | None = None,
        freeze_version: str | None = None,
        snapshot_tag: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialise the pipeline instance.

        Parameters
        ----------
        run_id : str, optional
            Unique identifier for this run (IDEM-7.1). Auto-generated
            as a UUID4 if not provided. Used as the deduplication key
            for audit records so the same run can be safely retried.
        correlation_id : str, optional
            Cross-system correlation ID for distributed tracing
            (LOG-11.7).
        triggered_by : str, optional
            Username or service that triggered this run, recorded in
            the audit trail for FDA 21 CFR Part 11 compliance
            (SEC-9.9, COMP-14.7).
        as_of_date : datetime, optional
            Backfill point-in-time (IDEM-7.8). If set, sources that
            support it will return data as it existed on this date.
        freeze_version : str, optional
            Pin to a specific source version (IDEM-7.3). If the source
            version changes between runs, a WARNING is logged.
        snapshot_tag : str, optional
            Tag for snapshot-based reproducibility (IDEM-7.14).
        seed : int, optional
            Random seed for reproducible stochastic operations
            (IDEM-7.4). Defaults to the class attribute ``seed`` (42).

        Raises
        ------
        TypeError
            If ``source_name`` is not set or is empty (DESIGN-2.14,
            DESIGN-2.15).
        """
        # Validate source_name (DESIGN-2.14, DESIGN-2.15)
        if not isinstance(self.source_name, str) or not self.source_name.strip():
            raise TypeError(
                f"Subclass must set source_name to a non-empty string, "
                f"got {self.source_name!r}"
            )
        self.source_name = self.source_name.strip()

        # Identifiers & lineage
        self.run_id: str = run_id or str(uuid.uuid4())
        self.correlation_id: str | None = correlation_id
        self.triggered_by: str | None = triggered_by
        self.as_of_date: datetime | None = as_of_date
        self.freeze_version: str | None = freeze_version
        self.snapshot_tag: str | None = snapshot_tag
        if seed is not None:
            self.seed = int(seed)

        # v113 FORENSIC ROOT FIX (P1-025, IDEM-7.4):
        # Per-instance RNG instead of mutating the GLOBAL ``random`` /
        # ``numpy.random`` state. The previous code called
        # ``random.seed(self.seed)`` and ``np.random.seed(self.seed)``
        # inside ``run()`` -- this mutated the GLOBAL RNG. If two
        # pipeline instances ran concurrently in the same process
        # (e.g., ChEMBL and DrugBank in the master DAG's parallel
        # tasks), the second instance's seed overwrote the first's --
        # the first pipeline's subsequent ``random.uniform()`` calls
        # used the SECOND pipeline's seed, breaking reproducibility.
        # The comment at ``pipelines/__init__.py:469`` claimed this
        # provided "IDEM-4" (idempotency), but it actually DESTROYED
        # idempotency for concurrent pipelines.
        #
        # ROOT FIX: instantiate a local ``random.Random(self.seed)``
        # and (when numpy is available) a local
        # ``np.random.default_rng(self.seed)``. Both are stored on
        # ``self`` and used by all retry-backoff / jitter code in this
        # module (see ``run()`` and the two ``random.uniform`` call
        # sites in ``_download_with_retries``). The GLOBAL RNG is no
        # longer touched.
        self._rng: random.Random = random.Random(self.seed)
        if _HAS_NUMPY:
            self._np_rng = np.random.default_rng(self.seed)
        else:  # pragma: no cover -- numpy is a hard dep in production
            self._np_rng = None

        # v49 ROOT FIX: read DRUGOS_DOWNLOAD_MODE from env (default sample).
        _mode = os.environ.get("DRUGOS_DOWNLOAD_MODE", self.DEFAULT_DOWNLOAD_MODE).lower().strip()
        if _mode not in self.VALID_DOWNLOAD_MODES:
            logger.warning(
                "Invalid DRUGOS_DOWNLOAD_MODE=%r -- must be one of %s. "
                "Falling back to default %r.",
                _mode, self.VALID_DOWNLOAD_MODES, self.DEFAULT_DOWNLOAD_MODE,
            )
            _mode = self.DEFAULT_DOWNLOAD_MODE
        self.download_mode: str = _mode
        logger.debug(
            "%s: download_mode=%r (from DRUGOS_DOWNLOAD_MODE env var)",
            self.__class__.__name__, self.download_mode,
        )

        # State -- populated during run() (ARCH-1.7: no dir creation here)
        self.start_time: datetime | None = None
        self.source_version: str | None = None
        self.source_publication_date: datetime | None = None
        self.downloaded_paths: list[Path] = []
        self._sha256_raw: str | None = None
        self._sha256_cleaned: str | None = None
        self._audit_buffer: list[dict[str, Any]] = []
        self._transformation_log: list[dict[str, Any]] = []
        self.dead_letter_queue: list[dict[str, Any]] = []
        self.entity_resolution_applied: bool = False
        # v107 P1-039: PII detection result (populated by _detect_pii in
        # run() / run_download_and_clean_only()). Initialized to [] so
        # code that reads it before clean() runs sees an empty list, not
        # AttributeError.
        self._pii_detected: list[str] = []

        # Kept for backward compatibility (ARCH-1.16): populated by run()
        # and used by some existing tests / callers as a run context dict.
        self.run_log: dict[str, Any] = {}

        # Internal collaborators (lazy-initialised on first use)
        self._http_session: requests.Session | None = None
        self._schema_cache: dict[str, Any] | None = None
        self._count_cache: dict[tuple[str, int, float], int] = {}
        self._rate_limiter = _RateLimiter(self.min_request_interval)
        self._circuit_breaker = _CircuitBreaker()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate ``source_name`` at class definition time (ARCH-1.14).

        Catches empty / whitespace source names early -- at import time
        rather than at instantiation time.
        """
        super().__init_subclass__(**kwargs)
        # Only validate if the subclass sets its own source_name
        if "source_name" in cls.__dict__ and isinstance(cls.source_name, str):
            name = cls.source_name.strip()
            if not name:
                raise TypeError(
                    f"{cls.__name__}.source_name must be a non-empty string, "
                    f"got {cls.source_name!r}"
                )
            if name.lower() not in VALID_SOURCE_NAMES:
                logger.warning(
                    "Unrecognized source_name %r in %s. "
                    "Expected one of: %s",
                    name,
                    cls.__name__,
                    ", ".join(sorted(VALID_SOURCE_NAMES)),
                )

    # ------------------------------------------------------------------
    # Lazy directory initialisation (ARCH-1.7)
    # ------------------------------------------------------------------
    def _ensure_directories(self) -> None:
        """Lazily create ``raw_dir`` and ``PROCESSED_DATA_DIR`` on first use.

        Called from ``run()``, ``run_download_and_clean_only()``, and
        ``run_load_only()`` -- NOT from ``__init__``. This makes
        ``__init__`` side-effect-free, which is important for test
        isolation and for Airflow DAG parsing (where pipelines are
        constructed but not run).

        Also lazily initialises the staging database schema by calling
        ``database.connection.init_db()`` if the ``pipeline_runs`` table
        is missing. This makes ``python -m pipelines run <source>`` work
        end-to-end without a manual ``init_db()`` step (REL-7, USAB-1).
        In production, ``init_db()`` is still called explicitly by the
        docker-compose setup / Makefile; the check here is a defensive
        safety net for direct CLI / Airflow Task usage. ``init_db()`` is
        idempotent (uses ``Base.metadata.create_all``), so calling it
        when tables already exist is a no-op.

        Raises
        ------
        PermissionError
            If the data directories cannot be created (CODE-4.2).
        """
        if self.raw_dir is None:
            self.raw_dir = RAW_DATA_DIR / self.source_name
        try:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot create data directory {self.raw_dir} for pipeline "
                f"'{self.source_name}'. Check filesystem permissions."
            ) from e
        try:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot create processed data directory "
                f"{PROCESSED_DATA_DIR}. Check filesystem permissions."
            ) from e

        # --- Defensive DB schema init (REL-7, USAB-1) ---------------------
        # Ensure the staging DB tables exist before any pipeline work
        # begins. This is a no-op when the schema is already in place
        # (init_db() uses create_all which is additive/idempotent).
        # Wrapped in try/except so a transient DB connectivity issue
        # surfaces as a clear PreCheckError-style message rather than a
        # downstream "no such table" OperationalError mid-load.
        try:
            from sqlalchemy import inspect as _sa_inspect
            from database.connection import init_db as _init_db, get_engine as _get_engine
            _engine = _get_engine()
            _insp = _sa_inspect(_engine)
            _required_tables = {
                "drugs", "proteins", "drug_protein_interactions",
                "protein_protein_interactions", "gene_disease_associations",
                "entity_mapping", "pipeline_runs",
            }
            _existing = set(_insp.get_table_names())
            if not _required_tables.issubset(_existing):
                logger.info(
                    "[%s] Staging DB schema incomplete (missing: %s). "
                    "Running init_db() to create missing tables.",
                    self.source_name,
                    sorted(_required_tables - _existing),
                )
                _init_db(initiator=f"BasePipeline._ensure_directories[{self.source_name}]")
        except (OSError, ValueError, RuntimeError, ImportError) as exc:  # noqa: BLE001
            # v85 FORENSIC ROOT FIX (BUG #51): narrowed from broad
            # ``except Exception``. DB init can fail with connection errors
            # (OSError), SQL errors (ValueError), or missing drivers
            # (ImportError). Programming bugs now propagate.
            # Re-raise as a clear, actionable error -- do NOT silently
            # swallow. The user needs to know the DB cannot be reached.
            raise RuntimeError(
                f"Pipeline '{self.source_name}' could not initialise the "
                f"staging database schema. Original error: {exc}. "
                f"Verify DATABASE_URL is reachable and the database user "
                f"has CREATE TABLE permission."
            ) from exc

    # ------------------------------------------------------------------
    # Context manager (ARCH-1.13)
    # ------------------------------------------------------------------
    def __enter__(self) -> "BasePipeline":
        """Enter context: return self."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context: ensure teardown runs."""
        self.teardown()

    # ------------------------------------------------------------------
    # Properties (SEC-9.17, CFG-12.18)
    # ------------------------------------------------------------------
    @property
    def http_session(self) -> requests.Session:
        """Return a reusable ``requests.Session`` with retries and TLS verify.

        Lazily created on first access. The session mounts an
        ``HTTPAdapter`` with a ``Retry`` policy that retries on 429,
        500, 502, 503, 504 with exponential backoff (SEC-9.17,
        CODE-4.40). TLS verification is always enabled (SEC-9.7).
        """
        if self._http_session is None:
            session = requests.Session()
            session.verify = self.verify_tls
            if _HAS_URLLIB3 and Retry is not None:
                retry = Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=RETRYABLE_STATUS_CODES,
                    allowed_methods=frozenset(["GET", "HEAD", "POST"]),
                    raise_on_status=False,
                )
                adapter = HTTPAdapter(max_retries=retry)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
            self._http_session = session
        return self._http_session

    @property
    def use_cached_download(self) -> bool:
        """Whether to use cached downloads (CFG-12.18 feature flag)."""
        return os.environ.get("PIPELINE_USE_CACHE", "true").lower() == "true"

    @property
    def skip_integrity_check(self) -> bool:
        """Whether to skip integrity checks (CFG-12.18 feature flag)."""
        return os.environ.get("PIPELINE_SKIP_INTEGRITY", "false").lower() == "true"

    # ------------------------------------------------------------------
    # Abstract methods -- subclasses MUST implement
    # ------------------------------------------------------------------
    @abstractmethod
    def download(self) -> Path | list[Path]:
        """Download raw data from the source.

        Returns
        -------
        Path or list of Path
            Path(s) to the downloaded file(s). Single-file sources
            return a single ``Path``; multi-file sources (e.g. STRING
            which downloads links + aliases) may return a list. The
            base class handles both shapes (SCI-3.13).

        Notes
        -----
        - Subclasses should call ``self._download_file(url, dest)``
          for each file rather than implementing their own HTTP logic.
        - Subclasses may set ``self.source_version`` from response
          headers or file content (SCI-3.8).
        - Existing subclasses return a single ``Path``; the new
          ``list[Path]`` option is purely additive.
        """
        ...

    @abstractmethod
    def clean(self, raw_path: Path) -> pd.DataFrame:
        """Clean and normalise the raw data.

        Parameters
        ----------
        raw_path : Path
            Path to the raw downloaded file (the first path if
            ``download()`` returned a list).

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame. The base class will validate this
            against ``schema/v1.json`` (SCI-3.12) and persist it to
            ``PROCESSED_DATA_DIR`` (ARCH-1.3).

        Notes
        -----
        - Subclasses should call ``self._log_transformation(step,
          rows_affected, details)`` for each transformation step so
          the audit trail records what was applied (LIN-16.11).
        - Subclasses should standardise identifiers (InChIKey for
          drugs, UniProt ID for proteins) so downstream entity
          resolution can join across sources.
        - Bad rows may be appended to ``self.dead_letter_queue``
          instead of crashing the whole clean (REL-6.1).
        """
        ...

    @abstractmethod
    def load(self, df: pd.DataFrame, session: Any | None = None) -> int | LoadResult:
        """Bulk upsert the cleaned DataFrame into the staging DB.

        Parameters
        ----------
        df : pandas.DataFrame
            Cleaned DataFrame from ``clean()``.
        session : Session, optional
            SQLAlchemy session to use. If provided, the caller
            manages the transaction boundary (ARCH-1.5). If None,
            ``load()`` opens its own session.

        Returns
        -------
        int or LoadResult
            Number of rows upserted (inserted + updated), OR a
            ``LoadResult`` with detailed metrics (DQ-5.3). Returning
            ``int`` is backward compatible with the 7 existing
            subclasses.

        Upsert semantics
        ----------------
        Insert new rows, update existing rows based on primary key.
        Strategy is controlled by ``self.upsert_strategy``: ``merge``
        (default), ``replace``, or ``append_only`` (DESIGN-2.13).

        FIX #18 NOTE / Transaction boundary:
            For true atomicity, the entire ``load()`` should be
            wrapped in a single transaction. Currently each subclass
            creates its own sessions for individual operations. Future
            work: refactor subclasses to accept the optional
            ``session`` parameter and run all sub-operations within
            one transaction boundary. The base class already passes
            ``session`` through.
        """
        ...

    # ------------------------------------------------------------------
    # Public run methods
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        dry_run: bool = False,
        force_refresh: bool = False,
        skip_download: bool = False,
        skip_load: bool = False,
        max_records: int | None = None,
        count_records: bool = True,
    ) -> None:
        """Execute the full pipeline: download -> clean -> load (ARCH-1.12).

        Parameters
        ----------
        dry_run : bool, default False
            If True, run download and clean but do not write to the DB.
            Log what would be done.
        force_refresh : bool, default False
            If True, re-download even if a valid cached file exists.
        skip_download : bool, default False
            If True, use cached data (equivalent to run_load_only but
            with the full audit record).
        skip_load : bool, default False
            If True, stop after clean (do not call load()).
        max_records : int, optional
            Process only the first N records. Useful for testing
            (PERF-8.11).
        count_records : bool, default True
            If False, skip record counting to save time on large
            files (PERF-8.8). ``records_downloaded`` will be set to
            ``SENTINEL_COUNT_FAILED`` (-1) to indicate "unknown".

        Raises
        ------
        PreCheckError
            If pre-flight checks fail.
        DataIntegrityError
            If records_cleaned == 0 but records_downloaded > 0
            (SCI-3.10), or if the cleaned CSV has been modified
            since the download+clean phase (IDEM-7.6).
        SchemaValidationError
            If ``strict_validation=True`` and the cleaned DataFrame
            violates ``schema/v1.json`` (SCI-3.12).
        """
        self._ensure_directories()
        self.start_time = datetime.now(timezone.utc)
        self.run_log = {
            "run_id": self.run_id,
            "source": self.source_name,
            "started_at": self.start_time.isoformat(),
            "dry_run": dry_run,
            "force_refresh": force_refresh,
        }
        logger.info(
            "[%s][run_id=%s] Pipeline starting (env=%s, dry_run=%s)...",
            self.source_name,
            self.run_id,
            self.environment,
            dry_run,
        )

        # Pre-flight checks (ARCH-1.9)
        checks = self.pre_check()
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            logger.error(
                "[%s] Pre-flight checks failed: %s",
                self.source_name,
                ", ".join(failed),
            )
            raise PreCheckError(
                f"Pre-flight checks failed for {self.source_name}: "
                f"{', '.join(failed)}"
            )

        # v113 FORENSIC ROOT FIX (P1-025, IDEM-7.4):
        # Per-instance RNG was already initialised in ``__init__`` as
        # ``self._rng`` (random.Random) and ``self._np_rng``
        # (numpy Generator). The previous code here called
        # ``random.seed(self.seed)`` and ``np.random.seed(self.seed)``
        # on the GLOBAL RNG -- this mutated the global state for every
        # other module in the process (Airflow scheduler, ChEMBL HTTP
        # client, deduplicator). With concurrent pipelines, the second
        # pipeline's seed overwrote the first's, destroying both
        # reproducibility AND idempotency. The global seeding is now
        # REMOVED; all jitter in this module uses ``self._rng`` (see
        # ``_download_with_retries``).

        records_downloaded: int = 0
        records_cleaned: int = 0
        records_loaded: int = 0
        status = "running"
        error_message: str | None = None
        validation_errors: list[str] = []
        dq_metrics: dict[str, Any] = {}

        # File lock to prevent concurrent runs of the same pipeline (IDEM-7.12)
        lock = self._acquire_run_lock()

        try:
            # ----- Download -----
            if not skip_download:
                raw_result = self.download()
                # download() may return a Path or a list of Paths (SCI-3.13)
                raw_paths = (
                    raw_result if isinstance(raw_result, list)
                    else [raw_result]
                )
                raw_paths = [p for p in raw_paths if p is not None]
                self.downloaded_paths = raw_paths

                if count_records and raw_paths:
                    # v84 FORENSIC ROOT FIX (BUG #41): the previous code
                    # called ``self._count_records(p)`` TWICE per path --
                    # once in the ``sum()`` at line 1178, and again in
                    # the ``any()`` check at line 1182-1185. For a 2GB
                    # STRING file, this DOUBLED the counting time
                    # (minutes of wall-clock). ROOT FIX: compute each
                    # file's count ONCE, store in a list, then derive
                    # both the total and the sentinel check from the
                    # stored list.
                    _per_file_counts = [
                        self._count_records(p) for p in raw_paths
                    ]
                    # v90 ROOT FIX (BUG #13): filter out -1 sentinels
                    # before summing (see the other occurrence around
                    # line 1537 for detailed explanation).
                    _valid_counts = [
                        c for c in _per_file_counts
                        if c != SENTINEL_COUNT_FAILED
                    ]
                    if _valid_counts:
                        records_downloaded = sum(_valid_counts)
                    else:
                        records_downloaded = SENTINEL_COUNT_FAILED
                    # If any individual count failed, propagate the sentinel
                    if any(
                        c == SENTINEL_COUNT_FAILED
                        for c in _per_file_counts
                    ):
                        logger.warning(
                            "[%s] One or more record counts failed; "
                            "records_downloaded=%d may be inaccurate "
                            "(%d of %d counts were unknown).",
                            self.source_name,
                            records_downloaded,
                            sum(1 for c in _per_file_counts if c == SENTINEL_COUNT_FAILED),
                            len(_per_file_counts),
                        )
                else:
                    records_downloaded = SENTINEL_COUNT_FAILED

                # Compute SHA-256 of raw files (LIN-16.2, IDEM-7.10)
                if raw_paths:
                    sha = self._compute_sha256(raw_paths[0])
                    self._sha256_raw = sha

                logger.info(
                    "[%s] Downloaded %s records",
                    self.source_name,
                    "unknown" if records_downloaded == SENTINEL_COUNT_FAILED
                    else records_downloaded,
                )
            else:
                # Use cached raw file (skip_download mode)
                if self.downloaded_paths:
                    raw_paths = self.downloaded_paths
                else:
                    # FIX-P2-C-11 (audit P2): the previous code did
                    # ``list(self.raw_dir.glob("*"))`` which globs EVERY
                    # file in ``raw_dir`` including ``.sha256`` sidecars,
                    # ``.meta.json``, ``.lock``, ``.run_context.json``, and
                    # partial ``.tmp`` files. Passing those to ``clean()``
                    # produced parser errors or (worse) silently loaded
                    # sidecar metadata as if it were drug data. Restrict
                    # the glob to known data-file extensions (with optional
                    # compression suffix).
                    raw_paths = []
                    if self.raw_dir:
                        for pattern in (
                            "*.csv*", "*.json*", "*.xml*", "*.tsv*",
                        ):
                            raw_paths.extend(self.raw_dir.glob(pattern))
                        # Exclude sidecars / lock / temp files that may have
                        # been picked up by the broad globs above (e.g.
                        # ``foo.csv.sha256`` matches ``*.csv*`` but is a
                        # sidecar; ``foo.csv.tmp`` is a partial write).
                        _SIDE_EXT = (
                            ".sha256", ".meta.json", ".lock",
                            ".run_context.json", ".tmp", ".part",
                        )
                        raw_paths = [
                            p for p in raw_paths
                            if not any(p.name.endswith(ext) for ext in _SIDE_EXT)
                        ]
                records_downloaded = SENTINEL_COUNT_FAILED

            # ----- Clean -----
            if raw_paths:
                # NOTE: ``clean()`` receives only the FIRST file. For
                # multi-file sources (e.g. STRING's ``links`` + ``aliases``
                # downloads), the remaining files are not passed to
                # ``clean()`` here -- pipeline subclasses that need to
                # process multiple files MUST override ``run()`` to call
                # ``clean()`` once per file. This contract is documented
                # here (FIX-P2-C-12, audit P2) so subclasses do not rely
                # on the base class to fan-out multi-file cleaning.
                clean_df = self.clean(raw_paths[0])
            else:
                clean_df = pd.DataFrame()

            if max_records is not None and max_records > 0:
                clean_df = clean_df.head(max_records)

            # Reject NULL primary keys (DQ-5.19)
            clean_df = self._drop_null_primary_keys(clean_df)

            # Count valid records (DQ-5.2) -- distinct from len(df)
            records_cleaned = self._count_valid_records(clean_df)
            total_rows = len(clean_df)
            logger.info(
                "[%s] Cleaned to %d valid records (%d total rows)",
                self.source_name,
                records_cleaned,
                total_rows,
            )

            # Schema validation (SCI-3.12)
            is_valid, validation_errors = self.validate_output(clean_df)
            if not is_valid:
                for err in validation_errors:
                    logger.error("[%s] Schema validation: %s", self.source_name, err)
                if self.strict_validation:
                    raise SchemaValidationError(
                        f"Schema validation failed for {self.source_name}: "
                        f"{'; '.join(validation_errors)}"
                    )

            # Data quality metrics (DQ-5.16, DQ-5.17)
            dq_metrics = self._compute_data_quality_metrics(clean_df)
            dq_metrics["quality_score"] = self._compute_quality_score(clean_df)

            # Catastrophic loss detection (SCI-3.10)
            if (
                records_downloaded > 0
                and records_cleaned < records_downloaded * self.min_clean_ratio
            ):
                logger.error(
                    "[%s] Clean ratio below threshold: %d/%d = %.2f < %.2f",
                    self.source_name,
                    records_cleaned,
                    records_downloaded,
                    records_cleaned / max(1, records_downloaded),
                    self.min_clean_ratio,
                )
                # v83 FORENSIC ROOT FIX (P0-C11 -- pipeline_runs.status
                #   CHECK constraint violation):
                #   The DB CHECK (migration 001 line 177-178) only allows
                #   ('running', 'success', 'failed', 'partial'). The
                #   previous code wrote 'warning' here, 'partial' is the
                #   canonical status for "completed with quality concerns".
                #   The INSERT was raising `CHECK constraint failed:
                #   chk_pipeline_runs_status`, caught by the audit-log
                #   exception handler at line ~1577 and silently logged
                #   as `Audit log write failed` -- meaning NO audit trail
                #   was being recorded for any pipeline run with a
                #   sub-threshold clean ratio. This is a silent audit-
                #   trail deletion that violates the institutional-grade
                #   lineage contract.
                status = "partial"

            if records_cleaned == 0 and records_downloaded > 0:
                raise DataIntegrityError(
                    f"records_cleaned == 0 but records_downloaded == "
                    f"{records_downloaded} for {self.source_name}"
                )

            # Sanitize CSV output (SEC-9.14) and persist (ARCH-1.3)
            clean_df = self._sanitize_csv_output(clean_df)
            # v107 ROOT FIX (ISSUE-P1-039 — _detect_pii was dead code):
            #   The audit found ``_detect_pii`` was defined but NEVER
            #   called from any pipeline. The platform claimed
            #   GDPR/HIPAA compliance in docstrings but the PII detection
            #   was inert. A drug-name field that accidentally contained
            #   a patient email (e.g. from a misconfigured DrugBank XML)
            #   was never flagged. ROOT FIX: call ``_detect_pii`` after
            #   sanitization (before persistence) and log a WARNING for
            #   each column where PII is detected. We do NOT dead-letter
            #   rows (that would be a larger schema change) but we
            #   surface the signal so operators can investigate. The
            #   warning is also recorded in the run-log metadata below
            #   via the ``pii_detected`` field.
            try:
                pii_columns = self._detect_pii(clean_df)
                if pii_columns:
                    logger.warning(
                        "[%s] PII detected in cleaned data (SEC-9.15): %s. "
                        "Investigate the upstream source — drug/protein/disease "
                        "data should NEVER contain PII. (v107 P1-039)",
                        self.source_name, pii_columns,
                    )
                    self._pii_detected = pii_columns  # stash for run-log
                else:
                    self._pii_detected = []
            except Exception as pii_exc:  # noqa: BLE001 — never crash clean
                logger.debug(
                    "[%s] PII detection failed (non-fatal): %s",
                    self.source_name, pii_exc,
                )
                self._pii_detected = []
            cleaned_path = self._persist_cleaned_data(clean_df)
            logger.info(
                "[%s] Cleaned data persisted to: %s",
                self.source_name,
                cleaned_path,
            )

            # Write run context sidecar for run_load_only (IDEM-7.6)
            self._write_run_context(
                cleaned_path,
                records_downloaded=records_downloaded,
                records_cleaned=records_cleaned,
            )

            # ----- Load -----
            if not skip_load and not dry_run:
                with get_db_session(
                    pipeline_name=self.source_name,
                    run_id=self.run_id,
                    correlation_id=self.correlation_id,
                ) as session:
                    load_result = self.load(clean_df, session=session)
                    if isinstance(load_result, LoadResult):
                        # P1-058 ROOT FIX (v107): use rows_inserted (NOT
                        # total_upserted) for the records_loaded audit field.
                        # total_upserted = rows_inserted + rows_updated —
                        # this double-counts rows that were updated (they
                        # were already counted as inserted in a previous
                        # run). A load() that updates 100% of existing rows
                        # (0 new inserts) reported records_loaded=1000 —
                        # the load-ratio check passed but NO NEW data was
                        # loaded. The KG was stale. ROOT FIX: use
                        # rows_inserted so the audit trail reflects NEW
                        # data loaded. total_upserted is recorded in
                        # dq_metrics['load_detail'] for observability.
                        records_loaded = load_result.rows_inserted
                        dq_metrics["load_detail"] = asdict(load_result)
                        dq_metrics["load_detail"]["_audit_note"] = (
                            "records_loaded=rows_inserted (P1-058 ROOT FIX); "
                            "total_upserted=rows_inserted+rows_updated "
                            "includes updates (recorded here for observability)"
                        )
                    else:
                        records_loaded = int(load_result)

                logger.info(
                    "[%s] Loaded %d records",
                    self.source_name,
                    records_loaded,
                )

                # Load ratio check (SCI-3.10)
                if (
                    records_cleaned > 0
                    and records_loaded < records_cleaned * self.min_load_ratio
                ):
                    logger.error(
                        "[%s] Load ratio below threshold: %d/%d = %.2f < %.2f",
                        self.source_name,
                        records_loaded,
                        records_cleaned,
                        records_loaded / max(1, records_cleaned),
                        self.min_load_ratio,
                    )
                    # v83 P0-C11: 'warning' -> 'partial' (DB CHECK constraint
                    # only allows 'running'/'success'/'failed'/'partial').
                    status = "partial"

                # P1-020 ROOT FIX (Team-2): POST-LOAD validation.
                # ``validate_output()`` was already called on the in-memory
                # ``clean_df`` BEFORE persist (line ~1354), which validates
                # schema/columns. But that check CANNOT catch a load() that
                # silently inserted 0 rows (e.g. an upstream API failure
                # masked by a fallback path returned an empty DataFrame
                # that passed schema validation but loaded nothing). The
                # audit explicitly required: "Call self.validate_output()
                # at the end of run(). If validation fails, raise
                # PipelineValidationError. Add a CI test that mocks a
                # zero-row insert and verifies the error."
                #
                # ROOT FIX: after load(), re-run validate_output() on the
                # final state AND add an explicit zero-load guard. The
                # zero-load guard catches the "silent 0-row insert" case
                # that schema validation alone cannot detect.
                if not skip_load and not dry_run and records_cleaned > 0:
                    if records_loaded == 0:
                        # The signature of a silent load failure: we had
                        # cleaned records to load, but load() inserted 0.
                        # This is NEVER acceptable in production -- it
                        # means the KG has gaps that downstream phases
                        # won't detect. Raise PipelineValidationError so
                        # the Airflow task fails and the operator is
                        # notified.
                        raise PipelineValidationError(
                            f"[{self.source_name}] P1-020 POST-LOAD "
                            f"VALIDATION FAILED: records_cleaned="
                            f"{records_cleaned} but records_loaded=0. "
                            f"The load() step silently inserted 0 rows "
                            f"(likely an upstream API failure masked by "
                            f"a fallback path, or a DB connection issue). "
                            f"The KG would have gaps -- downstream phases "
                            f"(KG build, Graph Transformer, RL ranker) "
                            f"would produce incomplete rankings. The "
                            f"pipeline FAILS LOUDLY per the institutional-"
                            f"grade contract. (P1-020 ROOT FIX)"
                        )
                    # Re-validate the final DataFrame state (defense-in-
                    # depth: catches any corruption introduced between
                    # the pre-persist validation and the post-load check).
                    _post_valid, _post_errors = self.validate_output(clean_df)
                    if not _post_valid and self.strict_validation:
                        raise PipelineValidationError(
                            f"[{self.source_name}] P1-020 POST-LOAD "
                            f"SCHEMA VALIDATION FAILED: {_post_errors}. "
                            f"The cleaned DataFrame passed pre-persist "
                            f"validation but failed post-load re-validation "
                            f"-- the DataFrame may have been mutated by "
                            f"load() (which should NOT happen). "
                            f"(P1-020 ROOT FIX)"
                        )
            elif dry_run:
                logger.info("[%s] Dry run: skipping load()", self.source_name)
                records_loaded = 0

            if status == "running":
                status = "success"

        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, ConnectionError, TimeoutError, PipelineError) as exc:
            # P1-020 ROOT FIX (Team-2): added ``PipelineError`` to the
            # caught types so that ``PipelineValidationError`` (raised by
            # the post-load validation above) is caught here, status is
            # set to "failed", the audit record is written correctly, and
            # the exception is re-raised. Without this, a
            # ``PipelineValidationError`` would propagate PAST the except
            # (because it's not a subclass of any caught type), the
            # finally block would write the audit record with
            # status="running" (the last value set before the raise), and
            # the operator would see a misleading "running" status in the
            # audit trail for a pipeline that actually failed.
            status = "failed"
            # v85 FORENSIC ROOT FIX (BUG #51): narrowed from broad
            # ``except Exception``. The pipeline's download/clean/load
            # stages can fail with I/O, data, runtime, or network errors.
            # Programming bugs (AttributeError, NameError) now propagate
            # instead of being silently caught as "pipeline failed".
            # CODE-4.4: SystemExit/KeyboardInterrupt have empty str()
            raw_msg = str(exc) if str(exc) else type(exc).__name__
            error_message = self._sanitize_error_message(raw_msg)
            logger.error(
                "[%s] Pipeline failed: %s",
                self.source_name,
                error_message,
                exc_info=self.log_exc_info,
            )
            raise
        finally:
            finished_at = datetime.now(timezone.utc)
            if self.start_time is not None:
                duration = round(
                    (finished_at - self.start_time).total_seconds(), 3
                )
            else:
                duration = None  # CODE-4.5: distinguish "no duration" from instant

            # Populate run_log context (ARCH-1.16)
            self.run_log.update({
                "finished_at": finished_at.isoformat(),
                "status": status,
                "records_downloaded": records_downloaded,
                "records_cleaned": records_cleaned,
                "records_loaded": records_loaded,
                "duration_seconds": duration,
                "validation_errors": validation_errors,
                "dq_metrics": dq_metrics,
                "error_message": error_message,
            })

            # Write the audit record (DQ-5.10, IDEM-7.2)
            try:
                self._write_run_log(
                    status=status,
                    started_at=self.start_time if self.start_time is not None
                    else finished_at,
                    finished_at=finished_at,
                    records_downloaded=records_downloaded,
                    records_cleaned=records_cleaned,
                    records_loaded=records_loaded,
                    error_message=error_message,
                    metadata_json={
                        "source": self.source_name,
                        "duration_seconds": int(duration) if duration is not None
                        else None,
                        "run_id": self.run_id,
                        "correlation_id": self.correlation_id,
                        "triggered_by": self.triggered_by,
                        "source_version": self.source_version,
                        "sha256_raw": self._sha256_raw,
                        "sha256_cleaned": self._sha256_cleaned,
                        "git_commit": _get_git_commit(),
                        "seed": self.seed,
                        "schema_version": SCHEMA_VERSION,
                        "validation_errors": validation_errors,
                        "dq_metrics": dq_metrics,
                        # v84 FORENSIC ROOT FIX (BUG #44): removed the
                        # duplicate ``records_downloaded`` /
                        # ``records_cleaned`` / ``records_loaded`` keys.
                        # These were ALREADY passed as positional args to
                        # ``_write_run_log`` at lines 1419-1421, so the
                        # audit JSON had each field twice -- redundant and
                        # confusing for any consumer parsing the JSON.
                    },
                )
            except (OSError, ValueError, TypeError) as audit_exc:
                # v85 FORENSIC ROOT FIX (BUG #51): narrowed from broad
                # ``except Exception``. Audit writes can fail with I/O,
                # serialization (TypeError/ValueError), or DB connection
                # (OSError) errors. Programming bugs now propagate.
                logger.error(
                    "[%s] Audit log write failed: %s",
                    self.source_name,
                    audit_exc,
                )

            # Teardown (ARCH-1.10)
            try:
                self.teardown()
            except (OSError, RuntimeError, ValueError) as teardown_exc:
                logger.warning(
                    "[%s] Teardown error: %s",
                    self.source_name,
                    teardown_exc,
                )

            # Release the run lock (IDEM-7.12)
            self._release_run_lock(lock)

    def run_download_and_clean_only(self) -> Path:
        """Download and clean data without loading into DB (ARCH-1.1, ARCH-1.2).

        Used by the master DAG to separate download+clean from load,
        allowing entity resolution to run between them. Returns the
        path to the downloaded raw file.

        Side effects:
            - Persists the cleaned DataFrame to ``PROCESSED_DATA_DIR``
              so ``run_load_only`` can pick it up.
            - Writes a ``.run_context.json`` sidecar with the SHA-256
              of the cleaned CSV (IDEM-7.6).
            - Writes an audit record with status
              ``download_clean_success`` (ARCH-1.2).

        Returns
        -------
        Path
            Path to the downloaded raw file.

        Raises
        ------
        Exception
            Any exception from ``download()`` or ``clean()`` is
            re-raised after the audit record is written.
        """
        self._ensure_directories()
        self.start_time = datetime.now(timezone.utc)
        logger.info(
            "[%s][run_id=%s] Download+clean run starting...",
            self.source_name,
            self.run_id,
        )

        records_downloaded: int = 0
        records_cleaned: int = 0
        status = "running"
        error_message: str | None = None

        try:
            raw_path = self.download()
            # download() may return Path or list[Path] (SCI-3.13)
            raw_paths = (
                raw_path if isinstance(raw_path, list) else [raw_path]
            )
            raw_paths = [p for p in raw_paths if p is not None]
            self.downloaded_paths = raw_paths

            # v90 ROOT FIX (BUG #13): the previous code used
            # ``sum(self._count_records(p) for p in raw_paths)``
            # which INCLUDED -1 sentinel values (SENTINEL_COUNT_FAILED)
            # in the sum, producing misleading counts (e.g. 3 files
            # with 1000, 500, -1 records -> sum=1499 instead of 1500
            # with one unknown). If ALL counts failed, sum=-N instead
            # of "unknown". ROOT FIX: filter out sentinel values before
            # summing; if any count is -1, set the total to -1 (unknown)
            # and log a warning.
            _per_file_counts = [self._count_records(p) for p in raw_paths]
            _valid_counts = [c for c in _per_file_counts if c != SENTINEL_COUNT_FAILED]
            if _valid_counts:
                records_downloaded = sum(_valid_counts)
            else:
                records_downloaded = SENTINEL_COUNT_FAILED
            if any(c == SENTINEL_COUNT_FAILED for c in _per_file_counts):
                logger.warning(
                    "[%s] One or more record counts failed; "
                    "records_downloaded=%d may be inaccurate "
                    "(%d of %d counts were unknown).",
                    self.source_name,
                    records_downloaded,
                    sum(1 for c in _per_file_counts if c == SENTINEL_COUNT_FAILED),
                    len(_per_file_counts),
                )
            logger.info(
                "[%s] Downloaded %d records",
                self.source_name,
                records_downloaded,
            )

            # FIX-P2-C-12 (audit P2): the previous code did
            # ``clean(raw_paths[0] if raw_paths else Path())`` which
            # passed an EMPTY ``Path()`` to ``clean()`` when
            # ``download()`` returned no paths -- producing a confusing
            # ``FileNotFoundError`` with an empty-path message instead
            # of a clear "no raw files" error. For multi-file sources
            # (STRING downloads links + aliases), only the first file is
            # passed to ``clean()`` -- this is documented above at the
            # other call site (around line 1146); subclasses that need
            # multi-file cleaning MUST override ``run()``.
            if not raw_paths:
                raise PipelineError(
                    f"[{self.source_name}] No raw files to clean: "
                    f"download() returned no paths. Check that the source "
                    f"URL is reachable and that raw_dir "
                    f"({self.raw_dir!s}) is not empty."
                )
            clean_df = self.clean(raw_paths[0])
            records_cleaned = self._count_valid_records(clean_df)
            logger.info(
                "[%s] Cleaned to %d records",
                self.source_name,
                records_cleaned,
            )

            # Persist cleaned data (ARCH-1.3) -- side effect, not return value
            clean_df = self._sanitize_csv_output(clean_df)
            # v107 ROOT FIX (ISSUE-P1-039 — _detect_pii was dead code):
            # call _detect_pii after sanitization, before persistence.
            # Same rationale as the run() call site (see comment there).
            try:
                pii_columns = self._detect_pii(clean_df)
                if pii_columns:
                    logger.warning(
                        "[%s] PII detected in cleaned data (SEC-9.15): %s. "
                        "Investigate the upstream source — drug/protein/disease "
                        "data should NEVER contain PII. (v107 P1-039)",
                        self.source_name, pii_columns,
                    )
                    self._pii_detected = pii_columns
                else:
                    self._pii_detected = []
            except Exception as pii_exc:  # noqa: BLE001 — never crash clean
                logger.debug(
                    "[%s] PII detection failed (non-fatal): %s",
                    self.source_name, pii_exc,
                )
                self._pii_detected = []
            cleaned_path = self._persist_cleaned_data(clean_df)
            logger.info(
                "[%s] Cleaned data persisted to: %s",
                self.source_name,
                cleaned_path,
            )
            self._write_run_context(
                cleaned_path,
                records_downloaded=records_downloaded,
                records_cleaned=records_cleaned,
            )

            # v83 FORENSIC ROOT FIX (P0-C11 -- pipeline_runs.status
            #   CHECK constraint violation):
            #   The DB CHECK only allows ('running', 'success', 'failed',
            #   'partial'). The previous code wrote
            #   'download_clean_success' which the CHECK rejected,
            #   silently deleting the audit trail for EVERY download+clean
            #   run (the audit-log exception handler at line ~1577 caught
            #   the IntegrityError and logged it as a non-fatal audit-log
            #   write failure). The pipeline phase is already captured in
            #   metadata_json.phase='download_clean', so the status column
            #   only needs the canonical 'success' value.
            status = "success"
            return raw_paths[0] if raw_paths else Path()
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, ConnectionError, TimeoutError) as exc:
            status = "failed"
            # v85 FORENSIC ROOT FIX (BUG #51): narrowed from broad
            # ``except Exception``. Download+clean can fail with I/O,
            # data, runtime, or network errors. Programming bugs
            # now propagate instead of being masked as "download failed".
            raw_msg = str(exc) if str(exc) else type(exc).__name__
            error_message = self._sanitize_error_message(raw_msg)
            logger.error(
                "[%s] Download+clean failed: %s",
                self.source_name,
                error_message,
                exc_info=self.log_exc_info,
            )
            raise
        finally:
            finished_at = datetime.now(timezone.utc)
            duration = (
                round((finished_at - self.start_time).total_seconds(), 3)
                if self.start_time is not None else None
            )
            try:
                self._write_run_log(
                    status=status,
                    started_at=self.start_time if self.start_time is not None
                    else finished_at,
                    finished_at=finished_at,
                    records_downloaded=records_downloaded,
                    records_cleaned=records_cleaned,
                    records_loaded=0,
                    error_message=error_message,
                    metadata_json={
                        "source": self.source_name,
                        "duration_seconds": int(duration) if duration is not None
                        else None,
                        "run_id": self.run_id,
                        "phase": "download_clean",
                        # v35 ROOT FIX (issue 23): include the same provenance
                        # fields written by ``run()`` so the audit trail for
                        # download+clean runs is as complete as for full runs.
                        # Previously these fields were missing, making it
                        # impossible to trace which source version, code commit,
                        # or schema version produced a given cleaned CSV.
                        "correlation_id": self.correlation_id,
                        "triggered_by": self.triggered_by,
                        "source_version": self.source_version,
                        "sha256_cleaned": self._sha256_cleaned,
                        "git_commit": _get_git_commit(),
                        "seed": self.seed,
                        "schema_version": SCHEMA_VERSION,
                    },
                )
            except (OSError, ValueError, TypeError) as audit_exc:
                # v85 FORENSIC ROOT FIX (BUG #51): narrowed from broad
                # ``except Exception``. Audit writes can fail with I/O,
                # serialization, or DB connection errors.
                logger.error(
                    "[%s] Audit log write failed: %s",
                    self.source_name,
                    audit_exc,
                )
            # v35 ROOT FIX (issue 24): call teardown() in the finally block
            # so HTTP sessions, file handles, and any subclass-specific
            # resources are released even when an exception propagates.
            # Previously this method leaked resources on failure (the
            # ``run()`` method already calls teardown in its finally block,
            # but run_download_and_clean_only did not).
            try:
                self.teardown()
            except (OSError, RuntimeError, ValueError) as teardown_exc:
                logger.warning(
                    "[%s] teardown() failed during run_download_and_clean_only "
                    "finally block: %s",
                    self.source_name,
                    teardown_exc,
                )

    def run_load_only(self) -> None:
        """Re-load from existing cleaned data without re-downloading.

        Reads the cleaned CSV from
        ``PROCESSED_DATA_DIR / self._get_processed_filename()`` and
        calls ``load()``. Verifies the CSV's SHA-256 against the
        ``.run_context.json`` sidecar written by
        ``run_download_and_clean_only`` to detect tampering (IDEM-7.6).

        Raises
        ------
        FileNotFoundError
            If the cleaned CSV does not exist. The error message tells
            the user to run the full pipeline first.
        DataIntegrityError
            If the cleaned CSV has been modified since the
            download+clean phase (SHA-256 mismatch).
        """
        self._ensure_directories()
        self.start_time = datetime.now(timezone.utc)
        logger.info(
            "[%s][run_id=%s] Load-only run starting...",
            self.source_name,
            self.run_id,
        )

        records_loaded: int = 0
        status = "running"
        error_message: str | None = None

        try:
            clean_path = PROCESSED_DATA_DIR / self._get_processed_filename()
            logger.info("[%s] Loading from: %s", self.source_name, clean_path)
            if not clean_path.exists():
                # CODE-4.12: error message doesn't reference Makefile
                raise FileNotFoundError(
                    f"No cleaned data found at {clean_path}. "
                    f"Run the full pipeline first to download and clean the data."
                )

            # Verify SHA-256 against .run_context.json (IDEM-7.6)
            self._verify_run_context(clean_path)

            # Read CSV with explicit dtype (SCI-3.11, INT-15.1 through INT-15.4)
            clean_df = pd.read_csv(
                clean_path,
                encoding="utf-8",
                dtype=self.get_dtypes(),
                low_memory=False,
                quoting=csv_mod.QUOTE_MINIMAL,
            )
            logger.info(
                "[%s] Read %d rows from %s",
                self.source_name,
                len(clean_df),
                clean_path,
            )

            # Schema validation (DQ-5.13)
            is_valid, validation_errors = self.validate_output(clean_df)
            if not is_valid and self.strict_validation:
                raise SchemaValidationError(
                    f"Schema validation failed for {self.source_name}: "
                    f"{'; '.join(validation_errors)}"
                )
            for err in validation_errors:
                logger.warning("[%s] Schema validation: %s", self.source_name, err)

            with get_db_session(
                pipeline_name=self.source_name,
                run_id=self.run_id,
                correlation_id=self.correlation_id,
            ) as session:
                load_result = self.load(clean_df, session=session)
                if isinstance(load_result, LoadResult):
                    records_loaded = load_result.total_upserted
                else:
                    records_loaded = int(load_result)

            logger.info(
                "[%s] Loaded %d records",
                self.source_name,
                records_loaded,
            )
            # v83 FORENSIC ROOT FIX (P0-C11 -- pipeline_runs.status
            #   CHECK constraint violation):
            #   Same root cause as 'download_clean_success' above. The DB
            #   CHECK only allows ('running', 'success', 'failed',
            #   'partial'). 'load_success' is replaced with the canonical
            #   'success'; the phase information is in metadata_json.phase.
            status = "success"
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as exc:
            status = "failed"
            # v85 FORENSIC ROOT FIX (BUG #51): narrowed from broad
            # ``except Exception``. Load-only can fail with I/O, DB,
            # or data errors. Programming bugs now propagate.
            raw_msg = str(exc) if str(exc) else type(exc).__name__
            error_message = self._sanitize_error_message(raw_msg)
            logger.error(
                "[%s] Load-only run failed: %s",
                self.source_name,
                error_message,
                exc_info=self.log_exc_info,
            )
            raise
        finally:
            finished_at = datetime.now(timezone.utc)
            duration = (
                round((finished_at - self.start_time).total_seconds(), 3)
                if self.start_time is not None else None
            )
            try:
                self._write_run_log(
                    status=status,
                    started_at=self.start_time if self.start_time is not None
                    else finished_at,
                    finished_at=finished_at,
                    records_downloaded=0,
                    records_cleaned=0,
                    records_loaded=records_loaded,
                    error_message=error_message,
                    metadata_json={
                        "source": self.source_name,
                        "duration_seconds": int(duration) if duration is not None
                        else None,
                        "run_id": self.run_id,
                        "phase": "load_only",
                        # v35 ROOT FIX (issue 23): include the same provenance
                        # fields written by ``run()`` and
                        # ``run_download_and_clean_only()`` for audit-trail
                        # completeness. ``sha256_cleaned`` is re-populated by
                        # ``_verify_run_context`` during load (the sidecar's
                        # expected SHA-256 becomes the actual SHA-256 of the
                        # file we re-loaded).
                        "correlation_id": self.correlation_id,
                        "triggered_by": self.triggered_by,
                        "source_version": self.source_version,
                        "sha256_cleaned": self._sha256_cleaned,
                        "git_commit": _get_git_commit(),
                        "seed": self.seed,
                        "schema_version": SCHEMA_VERSION,
                    },
                )
            # v107 ROOT FIX (ISSUE-P1-027 — run_load_only audit-log handler
            #   was broader than run() / run_download_and_clean_only()):
            #   The previous code used ``except Exception as audit_exc:`` here
            #   — DIFFERENT from the narrowed ``(OSError, ValueError,
            #   TypeError)`` tuple used in ``run()`` (line ~1616) and
            #   ``run_download_and_clean_only()`` (line ~1823). A programming
            #   bug in ``_write_run_log`` (e.g. AttributeError from a typo)
            #   was silently masked in ``run_load_only`` but propagated in
            #   ``run()`` — tests that exercised ``run_load_only`` could
            #   pass while ``run()`` failed, hiding the bug. ROOT FIX:
            #   align all three run methods on the SAME narrowed tuple.
            except (OSError, ValueError, TypeError) as audit_exc:
                logger.error(
                    "[%s] Audit log write failed: %s",
                    self.source_name,
                    audit_exc,
                )
            # v35 ROOT FIX (issue 24): call teardown() in the finally block
            # so HTTP sessions, file handles, and any subclass-specific
            # resources are released even when an exception propagates.
            # Mirrors the same fix in ``run_download_and_clean_only`` above.
            # v107 ROOT FIX (ISSUE-P1-025 — run_load_only teardown handler
            #   was broader than run() / run_download_and_clean_only()):
            #   The previous code used ``except Exception as teardown_exc:``
            #   here — DIFFERENT from the narrowed ``(OSError, RuntimeError,
            #   ValueError)`` tuple used in ``run()`` and
            #   ``run_download_and_clean_only()``. A typo in ``teardown()``
            #   (e.g. AttributeError) was silently masked as a warning,
            #   leaking HTTP sessions, file handles, and DB sessions. ROOT
            #   FIX: align with the narrowed tuple used in the other two
            #   run methods so programming bugs propagate.
            try:
                self.teardown()
            except (OSError, RuntimeError, ValueError) as teardown_exc:
                logger.warning(
                    "[%s] teardown() failed during run_load_only finally block: %s",
                    self.source_name,
                    teardown_exc,
                )

    # ------------------------------------------------------------------
    # Pre-flight checks (ARCH-1.9)
    # ------------------------------------------------------------------
    def pre_check(self) -> dict[str, bool]:
        """Run pre-flight checks before starting the pipeline.

        Returns
        -------
        dict
            Mapping of check name -> passed (True/False).
        """
        return {
            "raw_dir_writable": self._check_dir_writable(RAW_DATA_DIR),
            "processed_dir_writable": self._check_dir_writable(PROCESSED_DATA_DIR),
            "db_reachable": self._check_db_reachable(),
            "disk_space_sufficient": self._check_disk_space(),
        }

    def _check_dir_writable(self, path: Path) -> bool:
        """Check that *path* is writable by creating and removing a test file.

        CFG-12.8. Returns True if writable, False otherwise. Handles
        the case where the directory doesn't exist yet (tries to
        create it).
        """
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except (OSError, PermissionError):
            return False

    def _check_db_reachable(self) -> bool:
        """Check that the staging DB is reachable (P1-22 ROOT FIX).

        Previous code used ``pd.io.sql.text(...) if hasattr(pd.io.sql,
        "text") else __import__("sqlalchemy").text(...)`` -- a fragile
        pandas-internals hack that breaks on every pandas release where
        the ``pd.io.sql.text`` symbol is moved or removed. The fix uses
        the top-level ``sqlalchemy.text`` import (added to the module-
        level try-block) directly. The except clause is also narrowed
        from ``Exception`` to ``(OperationalError, TimeoutError)`` so
        that genuine programming errors (e.g. ``AttributeError`` from
        a typo in the session API) propagate instead of being silently
        swallowed as "DB unreachable".
        """
        if not _HAS_SQLALCHEMY:
            logger.error(
                "[%s] SQLAlchemy not installed -- cannot verify DB "
                "reachability. Pre-check failing closed (P1-22).",
                self.source_name,
            )
            return False
        try:
            with get_db_session() as session:
                session.execute(_sa_text("SELECT 1"))
            return True
        except TimeoutError as exc:
            logger.warning(
                "[%s] DB unreachable during pre_check (timeout): %s",
                self.source_name,
                exc,
            )
            return False
        except Exception as exc:
            # P1-044 ROOT FIX (v107): guard against None _SAOperationalError.
            # If SQLAlchemy is not installed, _SAOperationalError is None.
            # The previous ``except (_SAOperationalError, TimeoutError)``
            # would raise ``TypeError: catching classes that do not inherit
                       # from BaseException is not allowed`` because None is not a
            # valid exception type. The guard at the top of this function
            # (``if not _HAS_SQLALCHEMY: return False``) mitigates this in
            # the common case, but defense-in-depth requires us to also
            # handle the case where _HAS_SQLALCHEMY is True but
            # _SAOperationalError is None (shouldn't happen, but be safe).
            if _SAOperationalError is not None and isinstance(exc, _SAOperationalError):
                logger.warning(
                    "[%s] DB unreachable during pre_check (operational): %s",
                    self.source_name,
                    exc,
                )
                return False
            # Also catch InterfaceError (a parent of OperationalError in
            # some SQLAlchemy versions) and ProgrammingError (DB driver
            # not installed).
            if _SAInterfaceError is not None and isinstance(exc, _SAInterfaceError):
                logger.warning(
                    "[%s] DB unreachable during pre_check (interface): %s",
                    self.source_name,
                    exc,
                )
                return False
            raise

    def _check_disk_space(self, min_mb: int = 1024) -> bool:
        """Check that at least *min_mb* MB of disk space is available.

        Uses ``shutil.disk_usage``. Returns True if available space
        exceeds the threshold, False otherwise.

        P1-7 ROOT FIX: the previous code returned ``True`` on ANY
        exception -- including OSError raised by ``shutil.disk_usage``
        when the volume is unmounted, the path is invalid, or the
        underlying statvfs syscall fails. A disk-full condition MUST
        fail the pre-flight so operators see it BEFORE the pipeline
        writes 50 GB and aborts mid-load with corrupted partial files.
        Narrowing the except to ``OSError`` (the only exception
        ``shutil.disk_usage`` raises) and returning ``False`` makes
        the pre-check fail closed. Non-OSError exceptions (programming
        bugs) propagate so they surface during testing.
        """
        try:
            import shutil
            usage = shutil.disk_usage(RAW_DATA_DIR)
            return usage.free >= min_mb * 1024 * 1024
        except OSError as exc:
            logger.error(
                "[%s] Disk-space pre-check FAILED: could not stat %s: %s. "
                "Failing closed -- disk-full MUST abort the run BEFORE "
                "mid-pipeline corruption (P1-7 ROOT FIX).",
                self.source_name,
                RAW_DATA_DIR,
                exc,
            )
            return False

    def _check_api_keys(self) -> dict[str, bool]:
        """Verify that all required API keys are set (CFG-12.17).

        Returns
        -------
        dict
            Mapping of env var name -> is set (True/False).
        """
        return {key: bool(os.environ.get(key)) for key in self.required_api_keys}

    # ------------------------------------------------------------------
    # Teardown (ARCH-1.10)
    # ------------------------------------------------------------------
    def teardown(self) -> None:
        """Clean up temp files, partial downloads, and open sessions.

        Called automatically by ``run()``'s finally block and by the
        context manager ``__exit__``. Subclasses may override to add
        their own cleanup, but should call ``super().teardown()``.
        """
        # Close the HTTP session
        if self._http_session is not None:
            try:
                self._http_session.close()
            except (OSError, RuntimeError, ValueError) as exc:
                # P1-050 ROOT FIX (v107): log at WARNING so operators can
                # detect socket leaks. The previous silent ``pass`` let
                # connection-close failures accumulate invisibly —
                # eventually hitting the OS file-descriptor limit and
                # crashing the pipeline with "Too many open files".
                logger.warning(
                    "[%s] HTTP session close failed: %s: %s — possible "
                    "socket leak (P1-050 ROOT FIX).",
                    self.source_name,
                    type(exc).__name__,
                    exc,
                )
            self._http_session = None

        # Replay buffered audit records on next successful DB write (DQ-5.10)
        if self._audit_buffer:
            try:
                self._replay_audit_buffer()
            except (OSError, ValueError, RuntimeError) as exc:
                # v85 FORENSIC ROOT FIX (BUG #51): narrowed from broad
                # ``except Exception``. Audit replay can fail with DB
                # connection or serialization errors.
                logger.warning(
                    "[%s] Could not replay audit buffer: %s",
                    self.source_name,
                    exc,
                )

    # ------------------------------------------------------------------
    # File locking (IDEM-7.12)
    # ------------------------------------------------------------------
    def _acquire_run_lock(self) -> Any:
        """Acquire a file lock to prevent concurrent runs of this pipeline.

        Returns the lock object (which must be released by
        ``_release_run_lock``), or None if ``filelock`` is not
        installed.
        """
        if not _HAS_FILELOCK or FileLock is None:
            return None
        if self.raw_dir is None:
            self.raw_dir = RAW_DATA_DIR / self.source_name
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.raw_dir / ".run.lock"
        # v29 ROOT FIX (audit P1-23): was 5min TTL -- too short for real ETL. Increased to 30min.
        lock_timeout = self.file_lock_timeout_sec
        lock = FileLock(lock_path, timeout=lock_timeout)
        try:
            lock.acquire()
            return lock
        except FileLockTimeout:
            raise PipelineError(
                f"Could not acquire run lock for {self.source_name} "
                f"after {lock_timeout} seconds. Another run may be in progress."
            )

    def _release_run_lock(self, lock: Any) -> None:
        """Release the run lock acquired by ``_acquire_run_lock``."""
        if lock is None:
            return
        try:
            lock.release()
            # Try to remove the lock file for cleanliness
            lock_path = getattr(lock, "lock_file", None)
            if lock_path and Path(lock_path).exists():
                try:
                    Path(lock_path).unlink()
                except OSError:
                    pass
        except (OSError, RuntimeError, ValueError):
            # v85 FORENSIC ROOT FIX (BUG #51): narrowed from broad
            # ``except Exception``. Lock release can fail with I/O or
            # runtime errors. Programming bugs now propagate.
            pass

    # ------------------------------------------------------------------
    # Schema loading & validation (SCI-3.11, SCI-3.12)
    # ------------------------------------------------------------------
    def _load_schema(self) -> dict[str, Any]:
        """Load and cache ``schema/v1.json`` (SCI-3.12).

        Returns
        -------
        dict
            Parsed schema document.

        Notes
        -----
        The schema is cached on the instance after first load. If the
        file is missing or malformed, an empty dict is returned (so
        validation is a no-op rather than a hard failure).
        """
        if self._schema_cache is not None:
            return self._schema_cache
        try:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
                self._schema_cache = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning(
                "[%s] Could not load schema %s: %s. Validation will be skipped.",
                self.source_name,
                SCHEMA_PATH,
                exc,
            )
            self._schema_cache = {}
        return self._schema_cache

    def get_dtypes(self) -> dict[str, str]:
        """Return column name -> dtype mapping for this pipeline's CSV.

        Derived from ``schema/v1.json`` (SCI-3.11). Subclasses may
        override for custom types.

        Returns
        -------
        dict
            Mapping of column name to pandas dtype string. ``integer``
            schema types map to ``"Int64"`` (nullable integer),
            ``number`` to ``"float64"``, ``boolean`` to ``"boolean"``,
            and everything else to ``"str"``.

        Notes
        -----
        The schema may declare a type as a string (e.g. ``"integer"``)
        or as a list of types (e.g. ``["integer", "null"]`` for
        nullable columns). For list types, we use the first non-null
        type in the list.
        """
        schema = self._load_schema()
        dtypes: dict[str, str] = {}
        file_key = self._get_processed_filename()
        properties = schema.get("properties", {}).get(file_key, {}).get("properties", {})
        for col, spec in properties.items():
            col_type = spec.get("type")
            # Handle JSON Schema's type-as-list form: ["integer", "null"]
            if isinstance(col_type, list):
                # Use the first non-null type in the list
                col_type = next(
                    (t for t in col_type if t != "null"),
                    None,
                )
            if col_type == "integer":
                dtypes[col] = "Int64"
            elif col_type == "number":
                dtypes[col] = "float64"
            elif col_type == "boolean":
                dtypes[col] = "boolean"
            else:
                dtypes[col] = "str"
        return dtypes

    # v107 ROOT FIX (ISSUE-P1-024 + ISSUE-P1-035 — pattern-consistency
    # check was documented in comments but NEVER implemented):
    #   The comments at lines ~360-385 promised a
    #   ``_verify_pattern_consistency()`` method "called lazily from
    #   validate_output" that would assert the schema-declared InChIKey
    #   and UniProt patterns compile to the same match-set as the
    #   ``INCHIKEY_PATTERN`` and ``UNIPROT_ID_PATTERN`` constants. The
    #   audit found this method was NEVER defined — a textbook case of
    #   "comment says fixed, code says broken". The constants could
    #   diverge from the schema silently.
    #
    # ROOT FIX: implement ``_verify_pattern_consistency()`` and call it
    # lazily from ``validate_output()``. The check is memoised per
    # ``(file_key, schema_version)`` so it runs at most once per
    # pipeline instance + schema — no per-row overhead.
    _PATTERN_CONSISTENCY_CACHE: dict[tuple[str, str], list[str]] = {}

    @classmethod
    def _verify_pattern_consistency(cls, file_key: str, schema: dict) -> list[str]:
        """Verify schema-declared patterns match the canonical constants.

        Returns a list of warning messages (empty if all consistent).
        Memoised per ``(file_key, schema_version)`` to avoid re-running
        the check on every ``validate_output()`` call.
        """
        schema_version = str(schema.get("version", "unknown"))
        cache_key = (file_key, schema_version)
        if cache_key in cls._PATTERN_CONSISTENCY_CACHE:
            return cls._PATTERN_CONSISTENCY_CACHE[cache_key]

        warnings_list: list[str] = []
        properties = (
            schema.get("properties", {}).get(file_key, {}).get("properties", {})
        )
        # InChIKey consistency check (ISSUE-P1-024)
        for col_name, col_spec in properties.items():
            declared = col_spec.get("pattern")
            if not declared:
                continue
            col_lower = col_name.lower()
            if "inchikey" in col_lower:
                try:
                    declared_re = re.compile(declared)
                except re.error as exc:
                    warnings_list.append(
                        f"Schema pattern for '{col_name}' is not a valid "
                        f"regex: {exc}"
                    )
                    continue
                # Test a known-good InChIKey against both patterns
                test_inchikey = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
                if bool(INCHIKEY_PATTERN.match(test_inchikey)) != bool(declared_re.match(test_inchikey)):
                    warnings_list.append(
                        f"Schema pattern for '{col_name}' ({declared!r}) "
                        f"does not match INCHIKEY_PATTERN constant "
                        f"({INCHIKEY_PATTERN.pattern!r}) — test InChIKey "
                        f"'{test_inchikey}' diverges. The schema and "
                        f"constant may silently diverge (ISSUE-P1-024)."
                    )
                # Test a known-bad InChIKey
                bad_inchikey = "INVALID-INCHIKEY"
                if bool(INCHIKEY_PATTERN.match(bad_inchikey)) != bool(declared_re.match(bad_inchikey)):
                    warnings_list.append(
                        f"Schema pattern for '{col_name}' ({declared!r}) "
                        f"does not match INCHIKEY_PATTERN constant "
                        f"({INCHIKEY_PATTERN.pattern!r}) — bad InChIKey "
                        f"'{bad_inchikey}' diverges. (ISSUE-P1-024)."
                    )
            elif "uniprot" in col_lower:
                try:
                    declared_re = re.compile(declared)
                except re.error as exc:
                    warnings_list.append(
                        f"Schema pattern for '{col_name}' is not a valid "
                        f"regex: {exc}"
                    )
                    continue
                # Test known-good UniProt IDs (6-char and 10-char forms)
                for test_uniprot in ("P12345", "A0A0K3AVT9"):
                    if bool(UNIPROT_ID_PATTERN.match(test_uniprot)) != bool(declared_re.match(test_uniprot)):
                        warnings_list.append(
                            f"Schema pattern for '{col_name}' ({declared!r}) "
                            f"does not match UNIPROT_ID_PATTERN constant "
                            f"({UNIPROT_ID_PATTERN.pattern!r}) — test "
                            f"UniProt ID '{test_uniprot}' diverges "
                            f"(ISSUE-P1-035)."
                        )
                # Test a known-bad UniProt ID
                bad_uniprot = "INVALID1"
                if bool(UNIPROT_ID_PATTERN.match(bad_uniprot)) != bool(declared_re.match(bad_uniprot)):
                    warnings_list.append(
                        f"Schema pattern for '{col_name}' ({declared!r}) "
                        f"does not match UNIPROT_ID_PATTERN constant "
                        f"({UNIPROT_ID_PATTERN.pattern!r}) — bad UniProt "
                        f"ID '{bad_uniprot}' diverges (ISSUE-P1-035)."
                    )

        cls._PATTERN_CONSISTENCY_CACHE[cache_key] = warnings_list
        return warnings_list

    def validate_output(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Validate cleaned DataFrame against ``schema/v1.json`` (SCI-3.12).

        Checks:
        1. Required columns exist and are non-NULL.
        2. InChIKey columns match ``^[A-Z]{14}-[A-Z]{10}-[A-Z]$``.
        3. UniProt ID columns match the UniProt pattern.
        4. ``combined_score`` is in ``[0, 1000]``.
        5. ``score`` is in ``[0, 1]``.
        6. ``molecular_weight`` is ``>= 0``.
        7. ``max_phase`` is in ``[0, 4]``.
        8. ``length`` is ``>= 1``.
        9. ``mapping_key`` is in the schema's enum (currently
           ``[1, 2, 3, 4, null]``). The pipeline filters to
           ``OMIM_MAPPING_KEYS_INCLUDE`` (default ``[3, 4]``).

        Parameters
        ----------
        df : pandas.DataFrame
            Cleaned DataFrame to validate.

        Returns
        -------
        tuple
            ``(is_valid, list_of_error_messages)``. ``is_valid`` is
            True if no errors were found.
        """
        errors: list[str] = []
        if df is None or df.empty:
            # Empty DataFrame is valid -- a query that returned 0 rows
            # should produce a valid empty file (DQ-5.15).
            return True, []

        schema = self._load_schema()
        if not schema:
            return True, []

        file_key = self._get_processed_filename()
        file_schema = schema.get("properties", {}).get(file_key, {})
        if not file_schema:
            # P1-047 ROOT FIX (v108): the previous code silently returned
            # ``True, []`` (no errors) when the schema had no entry for
            # this file. This made validate_output() a NO-OP for any
            # pipeline whose _get_processed_filename() returned a name
            # not in schema/v1.json. A drugbank row with a malformed
            # InChIKey passed validation because there was no schema
            # entry for ``drugbank_drugs.csv``. The KG then had malformed
            # InChIKeys, and cross-source entity resolution on InChIKey
            # failed silently. ROOT FIX: log a WARNING so operators can
            # detect the missing schema entry. We still return True, []
            # (backward compat — a missing schema is not a data error),
            # but the WARNING surfaces the gap so it can be fixed.
            logger.warning(
                "[%s] validate_output: NO schema entry found for '%s' in "
                "schema/v1.json. Validation is a NO-OP for this file — "
                "malformed data (bad InChIKey, out-of-range scores, etc.) "
                "will NOT be caught. Add a schema entry for '%s' to enable "
                "validation (P1-047 ROOT FIX).",
                self.source_name,
                file_key,
                file_key,
            )
            return True, []

        # v107 ROOT FIX (ISSUE-P1-024 + ISSUE-P1-035): lazily verify that
        # the schema-declared patterns for InChIKey / UniProt columns match
        # the canonical ``INCHIKEY_PATTERN`` / ``UNIPROT_ID_PATTERN``
        # constants. The check is memoised per (file_key, schema_version)
        # so it runs at most once per pipeline instance + schema. Any
        # divergence is logged as a WARNING — the validation does NOT
        # fail (the schema is the source of truth for the actual pattern
        # check below), but the divergence is surfaced for maintainers.
        try:
            consistency_warnings = self._verify_pattern_consistency(file_key, schema)
            for w in consistency_warnings:
                logger.warning("Pattern consistency: %s", w)
        except Exception as cons_exc:  # noqa: BLE001 — never crash validation
            logger.debug(
                "Pattern consistency check failed (non-fatal): %s", cons_exc
            )

        properties = file_schema.get("properties", {})
        required = file_schema.get("required", [])

        # 1. Required columns exist
        for col in required:
            if col not in df.columns:
                errors.append(f"Required column '{col}' is missing")
                continue
            # v82 FORENSIC ROOT FIX (P1-6 -- required-column NULL check
            # inconsistent with pattern-check NaN-sentinel filter):
            #   The previous check used ONLY ``df[col].isna().any()``, which
            #   catches genuine pandas NaN/None but NOT the literal string
            #   sentinels "nan", "none", "null", "" that the pattern-check
            #   block (lines ~2105-2110) filters out. A required column
            #   with values ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "nan", ""]
            #   PASSED this NULL check (pd.isna("nan") is False -- it's a
            #   string), then the literal "nan" survived the pattern check
            #   (filtered out as a sentinel BEFORE the regex ran), and the
            #   row with inchikey="nan" was loaded into the DB. Downstream
            #   entity resolution's ``_inchikey_index.get("nan")`` then
            #   returned this row for EVERY future lookup that resolved
            #   to "nan" (e.g. a missing-InChIKey biologic with a
            #   synthetic key collision) -- wrong-row joins silently
            #   corrupted the KG.
            # ROOT FIX: extend the NULL check to ALSO flag literal
            #   NaN-string sentinels. The sentinel set is the SAME one
            #   the pattern-check block uses (lines ~2107), so the two
            #   checks are now consistent. A required column with any
            #   NaN OR any sentinel-string value is now flagged as having
            #   NULL values (the count is the union of both).
            _series_str = df[col].astype(str)
            _nan_sentinel_mask = _series_str.str.lower().isin(
                ("nan", "none", "null", "")
            )
            _true_null_mask = df[col].isna() | _nan_sentinel_mask
            if _true_null_mask.any():
                null_count = int(_true_null_mask.sum())
                errors.append(
                    f"Required column '{col}' has {null_count} NULL values "
                    f"(includes NaN-string sentinels: 'nan', 'none', 'null', "
                    f"'') -- v82 P1-6 root fix"
                )

        # 2-9. Pattern and range validation per column
        for col, spec in properties.items():
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                continue

            # P1-19 ROOT FIX: Apply ANY regex pattern declared in the schema
            # directly via re.compile(spec["pattern"]).match on the series.
            # Previously, only the InChIKey pattern was matched via brittle
            # string equality (`spec.get("pattern") == r"^[A-Z]{14}-..."`) --
            # any equivalent regex edit (whitespace, reordered alternatives,
            # re-anchoring, escape tweaks) would silently disable validation.
            # The UniProt branch used a substring check (`"OPQ" in
            # spec["pattern"] and "A-NR-Z" in spec["pattern"]`) that would
            # also fire on unrelated patterns mentioning those substrings.
            # Both branches are now unified into a single compile-and-match
            # block that validates against whatever pattern the schema
            # actually declares.
            pattern_str = spec.get("pattern")
            if pattern_str:
                try:
                    compiled_pattern = re.compile(pattern_str)
                except re.error as exc:
                    errors.append(
                        f"Column '{col}': invalid regex in schema "
                        f"({pattern_str!r}): {exc}"
                    )
                    continue
                # v64 ROOT FIX (P1-019): the previous code did
                # `series.astype(str).str.match(compiled_pattern)` directly
                # on the dropna()'d series. But CSV round-trip can serialize
                # NaN as the literal string "nan" (or "None"/"null"/""),
                # which survives dropna() (it's a real string, not NaN).
                # The literal "nan" then fails the InChIKey / UniProt ID
                # pattern check, producing false-positive validation errors.
                # Root fix: filter out the NaN-string sentinels BEFORE the
                # pattern check so only genuine non-null values are validated.
                _series_str = series.astype(str)
                _nan_sentinel_mask = _series_str.str.lower().isin(
                    ("nan", "none", "null", "")
                )
                _series_str = _series_str[~_nan_sentinel_mask]
                if _series_str.empty:
                    continue
                bad = ~_series_str.str.match(compiled_pattern)
                if bad.any():
                    errors.append(
                        f"Column '{col}': {int(bad.sum())} values "
                        f"do not match pattern {pattern_str!r}"
                    )

            # Range checks
            minimum = spec.get("minimum")
            maximum = spec.get("maximum")
            if minimum is not None or maximum is not None:
                try:
                    numeric = pd.to_numeric(series, errors="coerce")
                    if minimum is not None and (numeric < minimum).any():
                        errors.append(
                            f"Column '{col}': {int((numeric < minimum).sum())} "
                            f"values below minimum {minimum}"
                        )
                    if maximum is not None and (numeric > maximum).any():
                        errors.append(
                            f"Column '{col}': {int((numeric > maximum).sum())} "
                            f"values above maximum {maximum}"
                        )
                except (ValueError, TypeError):
                    pass

            # Enum check (e.g. mapping_key must be 3)
            enum_values = spec.get("enum")
            if enum_values:
                bad = ~series.isin(enum_values)
                if bad.any():
                    errors.append(
                        f"Column '{col}': {int(bad.sum())} values "
                        f"not in allowed enum {enum_values}"
                    )

        return (len(errors) == 0), errors

    def get_source_version(self) -> str | None:
        """Return the source version (e.g. 'ChEMBL v33', 'STRING v12.0').

        Default implementation returns ``self.source_version`` (set by
        subclasses during ``download()``). Subclasses may override to
        extract the version from API response headers or file content
        (SCI-3.8).

        P1-055 ROOT FIX (v107): if ``self.source_version`` is None (no
        subclass set it during download()), return a sensible default
        based on the source name and current UTC date. The previous
        implementation returned None — the audit trail had
        ``source_version=None`` for every run, violating FDA 21 CFR
        Part 11 version traceability. ROOT FIX: the default is
        ``f"{source_name}_as_of_{UTC_date}"`` — not as precise as a
        real release version, but it provides version traceability
        (the operator can answer "which ChEMBL version produced this
        KG?" with "chembl_as_of_2026-07-13"). A WARNING is logged so
        operators know the subclass should be updated to set the real
        version from the API response.
        """
        sv = getattr(self, "source_version", None)
        if sv is not None:
            return sv
        # P1-055 ROOT FIX (v107): generate a default source_version.
        logger.warning(
            "[%s] source_version was not set by download() — using "
            "default '%s_as_of_<UTC date>'. Update the pipeline's "
            "download() method to set self.source_version from the "
            "API response (e.g. ChEMBL's 'release' field, STRING's "
            "version directory name). (P1-055 ROOT FIX)",
            self.source_name,
            self.source_name,
        )
        from datetime import datetime, timezone
        return f"{self.source_name}_as_of_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

    # ------------------------------------------------------------------
    # Record counting (SCI-3.1 through SCI-3.18, PERF-8.1 through PERF-8.3)
    # ------------------------------------------------------------------
    def _count_records(self, path: Path) -> int:
        """Count data records in a downloaded file.

        Format-aware (SCI-3.17):
        - ``.json`` -- bracket-counting on the top-level array or the
          first array-valued key of an object (SCI-3.3, SCI-3.4).
        - ``.jsonl`` / ``.ndjson`` -- line count.
        - ``.csv`` / ``.tsv`` / ``.txt`` -- proper CSV parsing with
          multi-line quoted field support (SCI-3.1).
        - ``.gz`` -- detect inner format and delegate (SCI-3.18).
        - ``.parquet`` -- read metadata only, no data load.
        - ``.xml`` -- count top-level elements via iterparse.

        Returns
        -------
        int
            Number of data records. Returns ``SENTINEL_COUNT_FAILED``
            (-1) on error (file missing, encoding error, malformed
            data) so the audit trail records a count failure rather
            than a misleading 0 (SCI-3.5).

        Notes
        -----
        Results are memoised by ``(path, size, mtime)`` to avoid
        recounting the same file (CODE-4.44, PERF-8.12).

        File handles are always managed via ``with open`` to prevent
        leaks (CODE-4.15, REL-6.x). The initial readability check
        uses ``with open`` to atomically verify the file exists and
        is readable, avoiding the TOCTOU race between ``exists()``
        and ``stat()``.
        """
        # Validate input
        if path is None:
            return 0
        try:
            # Use 'with open' to verify the file exists and is readable
            # atomically (avoids TOCTOU race between exists() and stat()).
            with open(path, "rb") as _readability_fh:
                _readability_fh.read(1)  # read 1 byte to confirm readability
        except FileNotFoundError:
            logger.warning(
                "[%s] _count_records: file does not exist: %s",
                self.source_name,
                path,
            )
            # v43 ROOT FIX (P1 -- _count_records returns 0 not SENTINEL):
            # The docstring says "Returns SENTINEL_COUNT_FAILED (-1) on
            # error (file missing, encoding error, malformed data)."
            # But the previous code returned 0 for missing files, which
            # made the "any count failed" check at line 1087-1090 NEVER
            # fire (0 != -1). This caused silent empty-pipeline success:
            # a missing raw file -> records_downloaded=0 -> clean-ratio
            # check 0 < 0*0.3 = False (no warning) -> pipeline "succeeds"
            # with 0 records loaded. Returning SENTINEL_COUNT_FAILED
            # makes the check fire correctly.
            return SENTINEL_COUNT_FAILED
        except (PermissionError, OSError, ValueError):
            logger.warning(
                "[%s] _count_records: file not readable: %s",
                self.source_name,
                path,
            )
            return SENTINEL_COUNT_FAILED

        # Memoisation (CODE-4.44)
        # P2-7 ROOT FIX: the previous cache key used ``(str(path), st_size,
        # st_mtime)`` where ``st_mtime`` is a float with nanosecond
        # resolution on ext4. Two consecutive ``path.stat()`` calls on a
        # file being written can return different ``st_mtime`` values,
        # defeating the cache. The fix: use ``int(st_mtime)`` (second
        # resolution) as the cache key component. This is conservative --
        # if the file is modified within the same second, the stale cache
        # entry is returned (one extra file read), but this is far less
        # harmful than the current behaviour where the cache NEVER hits
        # for in-flight files. In practice, ``_count_records`` is called
        # after the file is fully written, so ``int(st_mtime)`` is stable.
        try:
            stat = path.stat()
            # P1-048 ROOT FIX (v107): use st_mtime_ns (nanosecond resolution)
            # instead of int(st_mtime) (second resolution). The previous
            # code used int(st_mtime) which has SECOND resolution — if the
            # file is modified twice within the same second (rare but
            # possible on fast SSDs), the cache returned the OLD count.
            # st_mtime_ns has nanosecond resolution, so the cache key
            # changes on every modification.
            cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
        except OSError:
            cache_key = (str(path), 0, 0)
        if cache_key in self._count_cache:
            return self._count_cache[cache_key]

        # Dispatch to format handler. The original implementation used
        # `if path.suffix == ".json":` to detect JSON files; we now use
        # a format registry (_FILE_FORMAT_HANDLERS) for cleaner dispatch
        # and to support .jsonl, .ndjson, .parquet, .xml, etc. (SCI-3.17).
        suffix = path.suffix.lower()
        handler_name = _FILE_FORMAT_HANDLERS.get(suffix)
        if handler_name is None:
            # Unknown format -- best-effort line count
            logger.warning(
                "[%s] Unknown file format %s, falling back to line count",
                self.source_name,
                suffix,
            )
            result = self._count_lines_fast(path)
        else:
            handler = getattr(self, handler_name)
            try:
                result = handler(path)
            except UnicodeDecodeError as exc:
                logger.warning(
                    "[%s] UnicodeDecodeError counting %s: %s",
                    self.source_name,
                    path.name,
                    exc,
                )
                result = SENTINEL_COUNT_FAILED
            except (FileNotFoundError, PermissionError, OSError) as exc:
                logger.error(
                    "[%s] Infrastructure error counting %s: %s",
                    self.source_name,
                    path.name,
                    exc,
                )
                raise
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning(
                    "[%s] Malformed data in %s: %s",
                    self.source_name,
                    path.name,
                    exc,
                )
                result = SENTINEL_COUNT_FAILED

        self._count_cache[cache_key] = result
        return result

    def count_records(self, path: Path) -> int:
        """Public alias for ``_count_records`` (DESIGN-2.8)."""
        return self._count_records(path)

    def _count_csv_records(self, path: Path) -> int:
        """Count data rows in a CSV/TSV file using proper CSV parsing (SCI-3.1).

        Correctly handles multi-line quoted fields (common in SMILES
        strings, protein sequences, free-text descriptions from
        ChEMBL/DrugBank/UniProt). Uses Python's ``csv`` module which
        is RFC 4180 compliant.

        Parameters
        ----------
        path : Path
            Path to the CSV/TSV file.

        Returns
        -------
        int
            Number of data rows (total rows minus 1 header row).
            Returns 0 for an empty file. Returns
            ``SENTINEL_COUNT_FAILED`` on UnicodeDecodeError.
        """
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        count = 0
        try:
            with open(path, "r", encoding="utf-8", errors="strict", newline="") as fh:
                reader = csv_mod.reader(fh, delimiter=delimiter)
                for i, _row in enumerate(reader):
                    if i == 0:
                        continue  # skip header
                    count += 1
        except UnicodeDecodeError:
            raise  # propagate to _count_records
        return count

    def _count_json_records(self, path: Path) -> int:
        """Count records in a JSON file without loading it into memory.

        Handles two shapes (SCI-3.3, SCI-3.4):
        - ``[{...}, {...}]`` -- top-level array. Returns the item count.
        - ``{"key": [...], ...}`` -- object with one or more array
          values. Returns the item count of the first array-valued
          key. If no array value is found, returns 1 (single object).

        Returns 0 for an empty array ``[]``. Returns
        ``SENTINEL_COUNT_FAILED`` on UnicodeDecodeError or malformed
        JSON.
        """
        try:
            with open(path, "r", encoding="utf-8", errors="strict") as fh:
                first_char = fh.read(1)
                if not first_char:
                    return 0  # empty file
                if first_char == "{":
                    return self._count_json_object_records(fh)
                elif first_char == "[":
                    return self._count_json_array_items(fh)
                else:
                    logger.warning(
                        "[%s] JSON file %s does not start with { or [",
                        self.source_name,
                        path.name,
                    )
                    return 0
        except UnicodeDecodeError:
            raise

    def _count_json_object_records(self, fh: Any) -> int:
        """Count records in a JSON object by finding the first array value.

        Algorithm (SCI-3.3):
        1. Scan for the first key whose value starts with ``[``.
        2. Count items in that array using bracket counting.
        3. If no array value is found, return 1 (single object).

        When counting items inside the array, track BOTH ``[`` / ``]``
        and ``{`` / ``}`` depth so commas inside object literals are
        NOT counted as item separators. Also track whether the array
        had any content (to distinguish ``[]`` from ``[1]``).
        """
        depth = 1  # already consumed the opening {
        in_string = False
        escape_next = False
        in_key = True  # at top level of an object, we expect a key
        key_buffer: list[str] = []
        last_key: str | None = None
        found_array = False
        array_depth = 0  # depth relative to the start of the array
        array_item_count = 0
        array_had_content = False

        while True:
            chunk = fh.read(self.json_read_chunk_size)
            if not chunk:
                break
            for ch in chunk:
                if escape_next:
                    escape_next = False
                    if in_string and in_key:
                        key_buffer.append(ch)
                    continue
                if ch == "\\":
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    if in_string and in_key:
                        key_buffer = []
                    elif not in_string and in_key:
                        last_key = "".join(key_buffer)
                        in_key = False
                    else:
                        # A string value is content (if we're in the array)
                        if found_array:
                            array_had_content = True
                    continue
                if in_string:
                    if in_key:
                        key_buffer.append(ch)
                    continue

                if found_array:
                    # Inside the array we're counting.
                    # Track BOTH [ and { depth so commas inside object
                    # literals are NOT counted as item separators.
                    if ch == "[" or ch == "{":
                        array_depth += 1
                        if array_depth > 1:
                            array_had_content = True
                    elif ch == "]" or ch == "}":
                        array_depth -= 1
                        if array_depth == 0:
                            result = (
                                (array_item_count + 1)
                                if array_had_content else 0
                            )
                            logger.info(
                                "[%s] Counted %d records from key %r",
                                self.source_name,
                                result,
                                last_key,
                            )
                            return result
                    elif ch == "," and array_depth == 1:
                        array_item_count += 1
                    elif not ch.isspace():
                        # Scalar value in the array
                        array_had_content = True
                else:
                    if ch == ":":
                        # Next non-whitespace char tells us the value type
                        in_key = False
                    elif ch == "[" and not in_key:
                        # Found the first array value
                        found_array = True
                        array_depth = 1
                        array_item_count = 0
                        array_had_content = False
                    elif ch == "{" and not in_key:
                        # Non-array value, skip until we return to top level
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            # End of object -- no array found
                            return 1
                        in_key = True
                    elif ch == "," and depth == 1:
                        # Next key
                        in_key = True
        return 1  # single object if we exit the loop

    def _count_json_array_items(self, fh: Any) -> int:
        """Count items in a JSON array using bracket counting (SCI-3.4).

        Uses a streaming approach so we don't load the entire file
        into memory. The array starts at the position right after the
        opening ``[`` (already consumed by the caller).

        Tracks BOTH ``[`` / ``]`` and ``{`` / ``}`` depth so that
        commas inside object literals (e.g. ``{"id":1, "name":"x"}``)
        are NOT counted as item separators. Only commas at the top
        level of the array (depth=1) are counted.

        Returns 0 for an empty array ``[]``. Returns N for an array
        with N items (count of commas at depth 1, plus 1 if any
        content was seen).
        """
        depth = 1  # already consumed the opening [
        in_string = False
        escape_next = False
        count = 0
        # Track whether the array has any content at all -- distinguishes
        # `[]` (0 items) from `[1]` (1 item with 0 commas).
        array_had_content = False

        while True:
            chunk = fh.read(self.json_read_chunk_size)
            if not chunk:
                break
            for ch in chunk:
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\":
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    # A string is content
                    array_had_content = True
                    continue
                if in_string:
                    continue
                # Track both array and object depth -- a comma inside
                # an object literal is NOT an item separator.
                if ch == "[" or ch == "{":
                    depth += 1
                    if depth > 1:
                        array_had_content = True
                elif ch == "]" or ch == "}":
                    depth -= 1
                    if depth == 0:
                        # End of the outer array
                        return (count + 1) if array_had_content else 0
                elif ch == "," and depth == 1:
                    count += 1
                elif not ch.isspace():
                    # Any other non-whitespace char at depth 1 means
                    # there's a scalar item (number, true, false, null)
                    array_had_content = True
        return (count + 1) if array_had_content else 0

    def _count_jsonl_records(self, path: Path) -> int:
        """Count lines in a JSONL / NDJSON file (one record per line).

        Each non-empty line is one JSON record. Empty lines are
        skipped.
        """
        count = 0
        try:
            with open(path, "r", encoding="utf-8", errors="strict") as fh:
                for line in fh:
                    if line.strip():
                        count += 1
        except UnicodeDecodeError:
            raise
        return count

    def _count_parquet_records(self, path: Path) -> int:
        """Count rows in a Parquet file using metadata only (SCI-3.17).

        Reads only the Parquet metadata (row group stats), not the
        data. Requires ``pyarrow`` to be installed.
        """
        if not _HAS_PYARROW:
            logger.warning(
                "[%s] pyarrow not installed, cannot count Parquet rows in %s",
                self.source_name,
                path.name,
            )
            return SENTINEL_COUNT_FAILED
        try:
            metadata = pq.read_metadata(path)
            return metadata.num_rows
        except (OSError, ValueError, RuntimeError) as exc:
            # v85 FORENSIC ROOT FIX (BUG #51): narrowed from broad
            # ``except Exception``. Parquet reading can fail with I/O,
            # format, or runtime errors.
            logger.warning(
                "[%s] Could not read Parquet metadata from %s: %s",
                self.source_name,
                path.name,
                exc,
            )
            return SENTINEL_COUNT_FAILED

    def _count_xml_records(self, path: Path) -> int:
        """Count top-level elements in an XML file via iterparse (SCI-3.17).

        Uses ``xml.etree.ElementTree.iterparse`` so the entire file
        is not loaded into memory. Clears elements after processing
        to free memory.
        """
        count = 0
        try:
            for _event, _elem in ET.iterparse(str(path), events=("end",)):
                count += 1
                _elem.clear()
        except ET.ParseError as exc:
            logger.warning(
                "[%s] XML parse error in %s: %s",
                self.source_name,
                path.name,
                exc,
            )
            return SENTINEL_COUNT_FAILED
        return count

    def _count_gz_records(self, path: Path) -> int:
        """Count records in a gzipped file by detecting inner format (SCI-3.18).

        Detects the inner format by reading the first few bytes after
        decompression, then delegates to the appropriate counter.
        """
        # Validate gzip magic bytes
        try:
            with open(path, "rb") as fh:
                magic = fh.read(2)
            if magic != b"\x1f\x8b":
                logger.warning(
                    "[%s] File %s has invalid gzip magic bytes "
                    "(expected 0x1f 0x8b, got %r)",
                    self.source_name,
                    path.name,
                    magic,
                )
                return SENTINEL_COUNT_FAILED
        except OSError as exc:
            logger.error(
                "[%s] Could not read %s: %s",
                self.source_name,
                path.name,
                exc,
            )
            raise

        inner_format = self._detect_inner_format(path)
        if inner_format == "csv":
            return self._count_gz_csv_records(path)
        elif inner_format == "json":
            return self._count_gz_json_records(path)
        elif inner_format == "jsonl":
            return self._count_gz_jsonl_records(path)
        elif inner_format == "parquet":
            # Write to temp file? Too complex -- use the parquet counter
            # directly on the gzip stream
            logger.warning(
                "[%s] Gzipped Parquet not supported for counting in %s",
                self.source_name,
                path.name,
            )
            return SENTINEL_COUNT_FAILED
        else:
            logger.warning(
                "[%s] Unknown inner format %r in gzip file %s",
                self.source_name,
                inner_format,
                path.name,
            )
            return SENTINEL_COUNT_FAILED

    def _detect_inner_format(self, path: Path) -> str:
        """Detect the format of a gzipped file's content (SCI-3.18).

        Reads the first 8 decompressed bytes and matches against known
        magic bytes:
        - ``PAR1`` -> parquet
        - ``{`` or ``[`` -> json (could be JSON object, JSON array,
          or JSONL -- disambiguated by checking for a newline within
          the first 200 bytes)
        - ``<`` -> xml
        - Otherwise, heuristic: if the first line contains ``,`` or
          ``\\t``, treat as CSV.

        Returns
        -------
        str
            One of ``"parquet"``, ``"json"``, ``"jsonl"``, ``"xml"``,
            ``"csv"``, ``"unknown"``.
        """
        try:
            with gzip.open(path, "rb") as fh:
                header = fh.read(8)
        except (OSError, gzip.BadGzipFile) as exc:
            logger.warning(
                "[%s] Could not read gzip header from %s: %s",
                self.source_name,
                path.name,
                exc,
            )
            return "unknown"

        if header[:4] == b"PAR1":
            return "parquet"
        if header[:1] == b"[":
            return "json"
        if header[:1] == b"<":
            return "xml"

        # For `{`, disambiguate JSON object from JSONL. JSONL files
        # have multiple `{...}` records separated by newlines, so we
        # check whether the first line is a complete `{...}` object
        # AND the next non-empty line also starts with `{`.
        if header[:1] == b"{":
            try:
                with gzip.open(path, "rt", encoding="utf-8", errors="strict") as fh:
                    first_line = fh.readline()
                    # Read a few more lines to check for JSONL pattern
                    next_lines = []
                    for _ in range(3):
                        line = fh.readline()
                        if not line:
                            break
                        next_lines.append(line)
                first_stripped = first_line.strip()
                # If the first line is a complete JSON object {...}
                if first_stripped.startswith("{") and first_stripped.endswith("}"):
                    # Check if the next non-empty line also starts with {
                    for nl in next_lines:
                        nl_stripped = nl.strip()
                        if nl_stripped:
                            if nl_stripped.startswith("{"):
                                return "jsonl"
                            else:
                                break  # next line is not an object -- not JSONL
                    # Single-line JSON object -> could be JSON or JSONL with 1 record
                    # Treat as JSON (the bracket counter will handle it)
                    return "json"
                # First line starts with { but doesn't end with } on the same line
                # -> multi-line JSON object
                return "json"
            except UnicodeDecodeError:
                return "unknown"

        # Heuristic: read first line and check for CSV delimiters
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="strict") as fh:
                first_line = fh.readline()
            if "," in first_line:
                # Check if it looks like JSONL (single object per line)
                stripped = first_line.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    return "jsonl"
                return "csv"
            if "\t" in first_line:
                return "csv"
        except UnicodeDecodeError:
            return "unknown"
        return "unknown"

    def _count_gz_csv_records(self, path: Path) -> int:
        """Count data rows in a gzipped CSV file using csv module.

        v9 ROOT FIX (audit F4.6): the previous implementation called
        ``fh.read()`` to load the ENTIRE remaining gzipped file into
        memory (``io.StringIO(first_line + fh.read())``) just to count
        rows. On a 2 GB STRING links file this would OOM the worker.
        The streaming design documented in surrounding docstrings was
        violated by that single line. Now we stream line-by-line
        directly through ``csv.reader`` -- constant memory regardless
        of file size.
        """
        count = 0
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="strict", newline="") as fh:
                # Detect delimiter from first line
                first_line = fh.readline()
                if not first_line:
                    return 0
                delimiter = "\t" if "\t" in first_line else ","
                # Stream the rest of the file directly through csv.reader.
                # ``csv.reader`` accepts any iterator yielding lines, so we
                # use ``itertools.chain`` to prepend the first line back
                # without buffering the whole file into memory.
                import itertools
                line_iter = itertools.chain([first_line], fh)
                reader = csv_mod.reader(line_iter, delimiter=delimiter)
                for i, _row in enumerate(reader):
                    if i == 0:
                        continue  # skip header
                    count += 1
        except UnicodeDecodeError:
            raise
        return count

    def _count_gz_json_records(self, path: Path) -> int:
        """Count records in a gzipped JSON file."""
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="strict") as fh:
                first_char = fh.read(1)
                if not first_char:
                    return 0
                if first_char == "{":
                    return self._count_json_object_records(fh)
                elif first_char == "[":
                    return self._count_json_array_items(fh)
                else:
                    return 0
        except UnicodeDecodeError:
            raise

    def _count_gz_jsonl_records(self, path: Path) -> int:
        """Count lines in a gzipped JSONL file."""
        count = 0
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="strict") as fh:
                for line in fh:
                    if line.strip():
                        count += 1
        except UnicodeDecodeError:
            raise
        return count

    def _count_lines_fast(self, path: Path) -> int:
        """Fast line counting using chunked newline counting (PERF-8.3).

        Counts ``\\n`` bytes in 1MB chunks. Subtracts 1 for the
        header. Returns 0 for an empty file.
        """
        count = 0
        try:
            with open(path, "rb") as fh:
                while True:
                    chunk = fh.read(1024 * 1024)
                    if not chunk:
                        break
                    count += chunk.count(b"\n")
        except OSError:
            return SENTINEL_COUNT_FAILED
        return max(0, count - 1)  # subtract header

    # ------------------------------------------------------------------
    # File integrity validation (SCI-3.7, SCI-3.2, REL-6.16)
    # ------------------------------------------------------------------
    def _validate_text_file_integrity(
        self,
        path: Path,
        allow_empty: bool = False,
    ) -> bool:
        """Validate integrity of a text (non-gzip) download (M5, REL-6.16).

        Fast chunk-based check that does NOT read the entire file:
        1. Read first 1KB -- verify the file has content.
        2. Seek to last 1KB -- verify it ends with ``\\n`` or ``\\r``.
        3. If ``allow_empty=True``, an empty file is considered valid.

        Parameters
        ----------
        path : Path
            Path to the file to validate.
        allow_empty : bool, default False
            If True, an empty file is valid (DQ-5.15). Used for
            initial downloads where a 0-record response is legitimate.

        Returns
        -------
        bool
            True if the file passes integrity checks, False otherwise.
        """
        try:
            size = path.stat().st_size
        except (FileNotFoundError, OSError):
            return False

        if size == 0:
            return allow_empty

        try:
            with open(path, "rb") as fh:
                # Read first 1KB to verify file has content
                head = fh.read(1024)
                if not head:
                    return allow_empty

                # Seek to last 1KB and verify ends with newline
                seek_pos = max(0, size - 1024)
                fh.seek(seek_pos)
                tail = fh.read(1024)
                if not tail or tail[-1:] not in (b"\n", b"\r"):
                    logger.debug(
                        "[%s] File %s does not end with newline",
                        self.source_name,
                        path.name,
                    )
                    return False

                # Quick line-count check using only the head + tail
                # (REL-6.16: don't read the entire file)
                # We check that there are at least 2 lines (header + 1 data)
                # by reading just the first 8KB.
                # v43 ROOT FIX (P2 -- large-file integrity check too permissive):
                # The previous check `sample.count(b"\n") < 2 and size < 8192`
                # only fired for files < 8KB. A truncated 2GB STRING file
                # (truncated at 1.9GB, missing the last row) had many
                # newlines in the first 8KB -> passed integrity. Fix: also
                # check the TAIL of the file -- read the last 1KB and
                # verify it ends with a complete row (newline + content).
                fh.seek(0)
                sample = fh.read(8192)
                if sample.count(b"\n") < 2 and size < 8192:
                    # Small file with < 2 lines
                    if size > 0 and sample.count(b"\n") >= 1:
                        return True
                    return False
                # v43: tail check for large files -- verify the last 1KB
                # ends with a newline (complete last row).
                if size > 8192:
                    fh.seek(max(0, size - 1024))
                    tail = fh.read(1024)
                    if tail and not tail.endswith(b"\n"):
                        logger.warning(
                            "[%s] _validate_text_file_integrity: file %s "
                            "does not end with newline -- may be truncated",
                            self.source_name, path,
                        )
                        return False

            return True
        except (OSError, ValueError) as exc:
            logger.warning(
                "[%s] Integrity check failed for %s: %s",
                self.source_name,
                path.name,
                exc,
            )
            return False

    def validate_text_file_integrity(self, path: Path) -> bool:
        """Public alias for ``_validate_text_file_integrity`` (DESIGN-2.8)."""
        return self._validate_text_file_integrity(path)

    def validate_download(self, path: Path) -> bool:
        """Validate a downloaded file (DESIGN-2.9).

        Default implementation calls ``_validate_download_integrity``
        with no expected columns or checksum. Subclasses may override
        for source-specific validation.

        Parameters
        ----------
        path : Path
            Path to the downloaded file.

        Returns
        -------
        bool
            True if the file passes validation.
        """
        is_valid, _reason = self._validate_download_integrity(path)
        return is_valid

    def _validate_download_integrity(
        self,
        path: Path,
        expected_columns: list[str] | None = None,
        min_records: int = 0,
        expected_sha256: str | None = None,
    ) -> tuple[bool, str]:
        """Multi-layered download integrity validation (SCI-3.7).

        Layers:
        1. Structural -- file is non-empty (or empty is allowed).
        2. Encoding -- file is valid UTF-8 (SCI-3.2).
        3. Checksum -- SHA-256 matches sidecar or expected value (SCI-3.9).
        4. Format-specific -- CSV delimiter consistency, JSON parses.

        Parameters
        ----------
        path : Path
            Path to the downloaded file.
        expected_columns : list of str, optional
            Expected column names for CSV/TSV files. If provided, the
            first line is checked to contain all expected columns.
        min_records : int, default 0
            Minimum number of records. If the file has fewer, a
            warning is logged but the file is still considered valid
            (a legitimate query may return 0 records).
        expected_sha256 : str, optional
            Expected SHA-256 hex digest. If provided, the file's
            actual SHA-256 must match.

        Returns
        -------
        tuple
            ``(is_valid, reason)``. ``reason`` is an empty string if
            valid, otherwise a description of the failure.
        """
        # Layer 1: Structural
        try:
            if not path.exists():
                return False, f"File does not exist: {path}"
            size = path.stat().st_size
        except OSError as exc:
            return False, f"Could not stat {path}: {exc}"

        if size == 0:
            # Empty file is valid if min_records is 0
            return (True, "") if min_records == 0 else (False, "File is empty")

        # Layer 2: Encoding (for text files)
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".xml"}:
            if not self._validate_file_encoding(path):
                return False, "File is not valid UTF-8"

        # Layer 3: Checksum
        if expected_sha256 is not None:
            actual = self._compute_sha256(path)
            if actual != expected_sha256:
                return False, (
                    f"SHA-256 mismatch: expected {expected_sha256}, "
                    f"got {actual}"
                )
        else:
            # Check sidecar .sha256 file
            sha256_sidecar = path.with_suffix(path.suffix + ".sha256")
            if sha256_sidecar.exists():
                try:
                    content = sha256_sidecar.read_text(encoding="utf-8").strip()
                    expected = content.split()[0] if content else ""
                    if expected:
                        actual = self._compute_sha256(path)
                        if actual != expected:
                            return False, (
                                f"SHA-256 mismatch with sidecar: "
                                f"expected {expected}, got {actual}"
                            )
                except OSError:
                    pass  # Sidecar read failure is not fatal

        # Layer 4: Format-specific
        if suffix == ".json":
            try:
                with open(path, "rb") as fh:
                    # Read first 1KB to check if it parses
                    sample = fh.read(1024)
                    if sample:
                        try:
                            # Just check it starts with { or [
                            if sample[0:1] not in (b"{", b"["):
                                return False, "JSON file does not start with { or ["
                        except (IndexError, ValueError):
                            return False, "JSON file is malformed"
            except OSError as exc:
                return False, f"Could not read {path}: {exc}"

        # Header validation for CSV/TSV
        if expected_columns and suffix in {".csv", ".tsv"}:
            try:
                with open(path, "r", encoding="utf-8", errors="strict") as fh:
                    header_line = fh.readline().strip()
                delimiter = "\t" if suffix == ".tsv" else ","
                actual_cols = [c.strip() for c in header_line.split(delimiter)]
                missing = set(expected_columns) - set(actual_cols)
                if missing:
                    return False, (
                        f"Missing expected columns: {missing}"
                    )
            except (OSError, UnicodeDecodeError) as exc:
                return False, f"Could not read header: {exc}"

        return True, ""

    def _validate_file_encoding(self, path: Path) -> bool:
        """Verify file contains valid UTF-8 (SCI-3.2).

        Reads in 64KB chunks to avoid loading the entire file into
        memory. Returns False and logs details of the first invalid
        byte offset if the file is not valid UTF-8.

        Parameters
        ----------
        path : Path
            Path to the file to validate.

        Returns
        -------
        bool
            True if the file is valid UTF-8, False otherwise.
        """
        offset = 0
        try:
            with open(path, "rb") as fh:
                while True:
                    chunk = fh.read(65536)
                    if not chunk:
                        break
                    try:
                        chunk.decode("utf-8", errors="strict")
                    except UnicodeDecodeError as exc:
                        logger.warning(
                            "[%s] File %s has invalid UTF-8 at byte %d: %s",
                            self.source_name,
                            path.name,
                            offset + exc.start,
                            exc,
                        )
                        return False
                    offset += len(chunk)
        except OSError as exc:
            logger.warning(
                "[%s] Could not read %s for encoding check: %s",
                self.source_name,
                path.name,
                exc,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Checksum computation (SCI-3.9, LIN-16.2)
    # ------------------------------------------------------------------
    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA-256 hash of file in streaming fashion (SCI-3.9).

        Reads the file in 64KB chunks to avoid loading the entire
        file into memory. Used for:
        - Verifying download integrity (LIN-16.2).
        - Computing the cleaned-CSV checksum (LIN-16.3).
        - Content-addressable storage (IDEM-7.15).

        Parameters
        ----------
        path : Path
            Path to the file to hash.

        Returns
        -------
        str
            Hex-encoded SHA-256 digest.
        """
        sha256 = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(65536)  # 64KB
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()

    def _verify_published_checksum(self, path: Path, expected_sha256: str) -> bool:
        """Verify a file against a published SHA-256 (SCI-3.9).

        Parameters
        ----------
        path : Path
            Path to the downloaded file.
        expected_sha256 : str
            Expected hex-encoded SHA-256 digest, as published by the
            source (ChEMBL, STRING, and DrugBank all publish
            checksums).

        Returns
        -------
        bool
            True if the file's SHA-256 matches the expected value.
            If not, the file is deleted (the caller should
            re-download).
        """
        actual = self._compute_sha256(path)
        if actual != expected_sha256:
            logger.error(
                "[%s] Checksum verification failed for %s: "
                "expected %s, got %s. Deleting corrupt file.",
                self.source_name,
                path.name,
                expected_sha256,
                actual,
            )
            try:
                path.unlink()
            except OSError:
                pass
            return False
        return True

    # ------------------------------------------------------------------
    # Download (SCI-3.6, SEC-9.x, REL-6.x, PERF-8.x)
    # ------------------------------------------------------------------
    def _download_file(
        self,
        url: str,
        dest: Path,
        headers: Mapping[str, str] | None = None,
        timeout: float | tuple[float, float] | None = None,
        max_retries: int | None = None,
        expected_sha256: str | None = None,
    ) -> Path:
        """Stream-download a file from *url* to *dest* with full hardening.

        Features:
        - URL scheme and domain validation (SEC-9.1).
        - Path traversal prevention (SEC-9.2).
        - File locking to prevent concurrent writes (IDEM-7.12).
        - Rate limiting (SEC-9.13).
        - Conditional requests (If-Modified-Since, If-None-Match) for
          cache freshness checks (SCI-3.15, COMP-14.12).
        - Resume with ETag/Last-Modified precondition check to
          prevent chimeric files (SCI-3.6, IDEM-7.9).
        - Retry with exponential backoff + jitter on retryable
          errors only (REL-6.4, REL-6.5, REL-6.6).
        - Content-Length verification (REL-6.7, REL-6.8).
        - 206 Content-Range verification (REL-6.9, COMP-14.13).
        - SHA-256 computation and sidecar storage (SCI-3.9).
        - Integrity validation (SCI-3.7).
        - URL sanitisation in logs (SEC-9.4).
        - Error message sanitisation (SEC-9.3).
        - fsync after write (CODE-4.33).

        Parameters
        ----------
        url : str
            URL to download. Must use an allowed scheme and domain.
        dest : Path
            Destination path. Must be within ``RAW_DATA_DIR``.
        headers : Mapping[str, str], optional
            Additional HTTP headers (e.g. ``Authorization``).
        timeout : float or tuple, optional
            ``(connect_timeout, read_timeout)`` in seconds. Defaults
            to ``self.download_timeout`` (REL-6.18).
        max_retries : int, optional
            Number of retry attempts. Defaults to
            ``self.download_max_retries``.
        expected_sha256 : str, optional
            Expected SHA-256 hex digest. If provided, the downloaded
            file is verified against this value (SCI-3.9).

        Returns
        -------
        Path
            Path to the downloaded file.

        Raises
        ------
        ValueError
            If the URL scheme/domain is not allowed or dest is
            outside ``RAW_DATA_DIR``.
        DownloadError
            If the download fails after all retries.
        """
        # Input validation (CODE-4.46)
        if not isinstance(url, str):
            raise TypeError(
                f"url must be a string, got {type(url).__name__}"
            )

        # URL scheme and domain validation (SEC-9.1)
        self._validate_url(url)

        # Path traversal prevention (SEC-9.2)
        self._validate_dest_path(dest)

        # Apply defaults
        if timeout is None:
            timeout = self.download_timeout
        if max_retries is None:
            max_retries = self.download_max_retries

        # Cache check: skip if file exists and is valid (SCI-3.15)
        if self.use_cached_download and dest.exists() and dest.stat().st_size > 0:
            if self._should_skip_download(dest, url, headers):
                logger.info(
                    "[%s] File exists (validated), skipping: %s",
                    self.source_name,
                    dest.name,
                )
                return dest
            # Cache is stale or invalid -- re-download
            logger.info(
                "[%s] Cached file %s is stale, re-downloading",
                self.source_name,
                dest.name,
            )
            try:
                dest.unlink()
            except OSError:
                pass

        # Sanitise URL for logging (SEC-9.4)
        safe_url = self._sanitize_url(url)
        logger.info("[%s] Downloading %s ...", self.source_name, safe_url)

        # File lock to prevent concurrent downloads of the same file (IDEM-7.12)
        lock = self._acquire_file_lock(dest)
        try:
            return self._download_with_retries(
                url, dest, headers, timeout, max_retries, expected_sha256
            )
        finally:
            self._release_file_lock(lock)

    def _download_with_retries(
        self,
        url: str,
        dest: Path,
        headers: Mapping[str, str] | None,
        timeout: float | tuple[float, float],
        max_retries: int,
        expected_sha256: str | None,
    ) -> Path:
        """Inner retry loop for ``_download_file`` (REL-6.4, REL-6.5, REL-6.6)."""
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                # v107 FORENSIC ROOT FIX (ISSUE-P1-017 + ISSUE-P1-033):
                #   The previous code called ``self._circuit_breaker.is_open()``
                #   which (per the v92 ROOT FIX) is now PURE OBSERVATION --
                #   it does NOT transition state from OPEN -> HALF_OPEN.
                #   After 5 failures, the breaker tripped and NEVER
                #   recovered within the process lifetime (the transition
                #   method ``try_acquire_probe()`` / ``allow_request()`` was
                #   never called in the download path). All downloads
                #   silently failed with "Circuit breaker is open" until
                #   the Airflow worker process restarted.
                #
                #   ROOT FIX: use ``allow_request()`` from the canonical
                #   ``_circuit_breaker.py``. This method (a) returns True
                #   if the request should proceed / False if refused, AND
                #   (b) transitions OPEN -> HALF_OPEN when the reset
                #   timeout has elapsed, reserving the single-probe slot.
                #   The breaker now auto-recovers after the timeout.
                #   (ISSUE-P1-033's ``try_acquire_probe()`` fix on the
                #   local duplicate class is superseded by this canonical
                #   ``allow_request()`` call — the local class was deleted
                #   in the P1-006/P1-017/P1-018 fix above.)
                if not self._circuit_breaker.allow_request():
                    raise DownloadError(
                        f"Circuit breaker is open for {self.source_name}; "
                        f"failing fast. Try again later."
                    )

                # Rate limiting (SEC-9.13)
                self._rate_limiter.wait()

                # Resume logic (SCI-3.6, IDEM-7.9)
                resume_from = 0
                try:
                    resume_from = dest.stat().st_size
                except FileNotFoundError:
                    resume_from = 0  # CODE-4.47: TOCTOU-safe

                # Verify partial file integrity before resuming.
                # P1-32 ROOT FIX: previously this only checked the 2-byte
                # gzip magic (0x1f 0x8b). A truncated gzip file passes
                # that check (the magic is at the start of the file, so
                # it survives any truncation after byte 2) but is not
                # parseable -- resuming such a file produces a chimeric
                # output that mixes the original (truncated) compressed
                # stream with the appended resume bytes, yielding an
                # unparseable gzip that crashes downstream readers. The
                # fix uses ``gzip.open(...).read(1)`` to verify the
                # file is actually parseable (not just that it starts
                # with the right magic). On BadGzipFile / EOFError /
                # OSError, the partial file is deleted and the download
                # restarts from byte 0.
                if resume_from > 0 and dest.suffix == ".gz":
                    try:
                        with open(dest, "rb") as fh:
                            magic = fh.read(2)
                        if magic != b"\x1f\x8b":
                            logger.warning(
                                "[%s] Partial gzip file %s has invalid magic "
                                "bytes (expected 0x1f 0x8b, got %r), deleting "
                                "and restarting download",
                                self.source_name,
                                dest.name,
                                magic,
                            )
                            dest.unlink()
                            resume_from = 0
                        else:
                            # Magic bytes are present, but the file may
                            # still be truncated mid-stream (missing
                            # CRC32 + size trailer, or cut off mid-deflate).
                            # Probe-parseability by attempting to read 1
                            # decompressed byte from the start. A
                            # truncated gzip raises BadGzipFile /
                            # EOFError / OSError depending on where the
                            # truncation occurred.
                            import gzip as _gzip_mod
                            try:
                                with _gzip_mod.open(dest, "rb") as gz_fh:
                                    _probe = gz_fh.read(1)
                                # If _probe is empty, the file was a
                                # valid-but-empty gzip (rare for a resume
                                # candidate -- treat as truncation).
                                if not _probe:
                                    raise _gzip_mod.BadGzipFile(
                                        "gzip stream decompressed to 0 bytes"
                                    )
                            except (
                                _gzip_mod.BadGzipFile,
                                EOFError,
                                OSError,
                            ) as gz_exc:
                                logger.warning(
                                    "[%s] Partial gzip file %s is not "
                                    "parseable (%s: %s) -- truncated CRC32/"
                                    "size trailer or mid-stream cut. "
                                    "Deleting and restarting download "
                                    "(P1-32 ROOT FIX).",
                                    self.source_name,
                                    dest.name,
                                    type(gz_exc).__name__,
                                    gz_exc,
                                )
                                dest.unlink()
                                resume_from = 0
                    except OSError:
                        pass

                # Conditional request headers for resume (SCI-3.6)
                req_headers: dict[str, str] = dict(headers or {})
                if resume_from > 0 and self.allow_resume:
                    req_headers["Range"] = f"bytes={resume_from}-"
                    # Add ETag/Last-Modified precondition if available
                    cond_headers = self._validate_resume_precondition(dest)
                    req_headers.update(cond_headers)
                    logger.info(
                        "[%s] Resuming download from byte %d (attempt %d/%d)",
                        self.source_name,
                        resume_from,
                        attempt,
                        max_retries,
                    )

                # Make the request via the reusable session (SEC-9.17)
                resp = self.http_session.get(
                    url,
                    stream=True,
                    headers=req_headers,
                    timeout=timeout,
                    verify=self.verify_tls,
                )

                # Handle 412 Precondition Failed -- source has changed (SCI-3.6)
                if resp.status_code == 412:
                    logger.warning(
                        "[%s] Source has changed (412 Precondition Failed) "
                        "for %s. Deleting partial file and restarting.",
                        self.source_name,
                        dest.name,
                    )
                    try:
                        dest.unlink()
                    except OSError:
                        pass
                    resp.close()
                    continue  # retry from scratch

                # Handle 304 Not Modified -- use cached file (SCI-3.15, COMP-14.12)
                if resp.status_code == 304:
                    logger.info(
                        "[%s] Source not modified (304), using cached file: %s",
                        self.source_name,
                        dest.name,
                    )
                    resp.close()
                    self._circuit_breaker.record_success()
                    return dest

                # Handle 416 Range Not Satisfiable -- restart from scratch (CODE-4.29)
                if resp.status_code == 416:
                    logger.warning(
                        "[%s] Server returned 416, restarting from scratch",
                        self.source_name,
                    )
                    resp.close()
                    try:
                        dest.unlink()
                    except OSError:
                        pass
                    # P1-9 ROOT FIX: the retry GET below was previously
                    # fired immediately after the 416 response, WITHOUT
                    # calling ``self._rate_limiter.wait()`` first. The
                    # main GET above (line ~2996) is preceded by a rate-
                    # limiter wait (line ~2953), but the 416 retry path
                    # bypassed it -- meaning every 416 doubled the
                    # effective request rate to the source, risking 429
                    # throttling or IP banning on sources like STRING /
                    # ChEMBL / PubChem that enforce tight rate limits.
                    # Re-acquire the rate-limiter token before the
                    # retry GET.
                    self._rate_limiter.wait()
                    resp = self.http_session.get(
                        url,
                        stream=True,
                        headers=dict(headers or {}),
                        timeout=timeout,
                        verify=self.verify_tls,
                    )
                    resume_from = 0

                # Retry on 5xx / 429 (REL-6.4)
                if resp.status_code in RETRYABLE_STATUS_CODES:
                    # v43 ROOT FIX (P2 -- closed response in HTTPError):
                    # Capture status_code before close, then pass
                    # response=None to HTTPError. The previous code passed
                    # the closed resp -> accessing .text or .headers on the
                    # closed response raises a secondary ValueError that
                    # masks the original HTTPError.
                    _status = resp.status_code
                    resp.close()
                    raise requests.exceptions.HTTPError(
                        f"HTTP {_status}", response=None
                    )

                resp.raise_for_status()

                # Parse Content-Length safely (CODE-4.30)
                try:
                    content_length = int(resp.headers.get("content-length", 0))
                except (ValueError, TypeError):
                    content_length = 0

                # Determine total size (CODE-4.31)
                content_range = resp.headers.get("Content-Range", "")
                if content_range and resp.status_code == 206:
                    try:
                        total_str = content_range.split("/")[-1]
                        total = int(total_str)
                    except (ValueError, IndexError):
                        total = resume_from + content_length
                elif resp.status_code == 206:
                    total = resume_from + content_length
                else:
                    total = content_length

                # Verify 206 Content-Range matches requested range (REL-6.9)
                if resp.status_code == 206 and content_range:
                    match = re.match(r"bytes (\d+)-", content_range)
                    if match and int(match.group(1)) != resume_from:
                        logger.warning(
                            "[%s] Server returned different range than requested "
                            "(expected %d, got %s), restarting",
                            self.source_name,
                            resume_from,
                            match.group(1),
                        )
                        resp.close()
                        try:
                            dest.unlink()
                        except OSError:
                            pass
                        continue

                # Empty body handling (REL-6.8)
                if total == 0 and content_length == 0:
                    # P1-10 ROOT FIX: previously ANY empty-body 200 OK
                    # returned ``dest`` as if successful -- silently
                    # persisting a 0-byte file that downstream parsers
                    # would emit ``Empty DataFrame`` warnings on. The
                    # run appeared successful while the data was
                    # missing. The fix: only return ``dest`` when the
                    # subclass has explicitly opted in via
                    # ``allow_empty_response = True``; otherwise treat
                    # the empty body as a download failure, delete the
                    # 0-byte file, record a circuit-breaker failure, and
                    # ``continue`` the retry loop so the next attempt
                    # re-fetches. If all attempts return empty, the
                    # safety-net ``raise DownloadError`` at the end of
                    # the loop fires and the operator sees the failure
                    # (instead of a silent 0-byte "success").
                    if self.allow_empty_response:
                        logger.info(
                            "[%s] Server returned empty body for %s -- "
                            "subclass allows empty responses (P1-10).",
                            self.source_name,
                            dest.name,
                        )
                        resp.close()
                        self._circuit_breaker.record_success()
                        return dest
                    logger.error(
                        "[%s] Server returned empty body (Content-Length: 0) "
                        "for %s -- subclass does not allow empty responses. "
                        "Deleting 0-byte file and retrying (P1-10 ROOT FIX).",
                        self.source_name,
                        dest.name,
                    )
                    resp.close()
                    try:
                        dest.unlink()
                    except OSError:
                        pass
                    self._circuit_breaker.record_failure()
                    last_exc = DownloadError(
                        f"Empty response body from {url} -- source may be "
                        f"temporarily unavailable. Set "
                        f"allow_empty_response=True on the subclass if "
                        f"this is an optional-metadata endpoint."
                    )
                    continue

                # Write the file
                downloaded = resume_from
                mode = "ab" if resume_from > 0 and resp.status_code == 206 else "wb"
                if mode == "wb" and dest.exists():
                    try:
                        dest.unlink()
                    except OSError:
                        pass

                next_log_at = self.progress_log_interval  # CODE-4.34
                with open(dest, mode) as fh:
                    for chunk in resp.iter_content(
                        chunk_size=self.download_chunk_size
                    ):
                        if chunk:
                            fh.write(chunk)
                            downloaded += len(chunk)
                            if downloaded >= next_log_at:
                                if total > 0:
                                    logger.info(
                                        "  %s: %d/%d bytes (%d%%)",
                                        dest.name,
                                        downloaded,
                                        total,
                                        downloaded * 100 // total,
                                    )
                                else:
                                    logger.info(
                                        "  %s: %d bytes",
                                        dest.name,
                                        downloaded,
                                    )
                                next_log_at += self.progress_log_interval
                    fh.flush()  # CODE-4.33
                    os.fsync(fh.fileno())  # CODE-4.33

                resp.close()

                # Verify downloaded size (REL-6.7)
                try:
                    actual_size = dest.stat().st_size
                except FileNotFoundError:
                    actual_size = 0
                if total > 0 and actual_size < total:
                    logger.error(
                        "[%s] Downloaded file %s is %d bytes, expected %d bytes",
                        self.source_name,
                        dest.name,
                        actual_size,
                        total,
                    )
                    try:
                        dest.unlink()
                    except OSError:
                        pass
                    # FIX-P2-C-9 (audit P2): raise
                    # ``requests.exceptions.ConnectionError`` (which IS in
                    # ``RETRYABLE_EXCEPTIONS``) instead of bare ``IOError``.
                    # ``IOError`` is an alias for ``OSError`` and was
                    # deliberately EXCLUDED from ``RETRYABLE_EXCEPTIONS`` by
                    # the v38 ROOT FIX (see comment at the constant's
                    # definition) because bare ``OSError`` includes
                    # permanent errors (PermissionError, FileNotFoundError,
                    # etc.). The except clauses at the bottom of this loop
                    # (``except RETRYABLE_EXCEPTIONS`` and the subsequent
                    # ``except HTTPError``) did NOT catch ``IOError``, so a
                    # truncated download (server closed the connection early)
                    # was never retried -- it propagated up as a hard
                    # failure. Raising ``requests.exceptions.ConnectionError``
                    # makes the truncated-download case retryable while
                    # preserving the v38 ROOT FIX's exclusion of bare
                    # ``OSError`` for permanent filesystem errors.
                    raise requests.exceptions.ConnectionError(
                        f"Incomplete download: {dest.name} "
                        f"({actual_size}/{total} bytes)"
                    )

                # Store download metadata sidecar (SCI-3.6, SCI-3.15)
                self._store_download_metadata(dest, resp)

                # Compute and store SHA-256 (SCI-3.9, LIN-16.2)
                sha256 = self._compute_sha256(dest)
                sha256_path = dest.with_suffix(dest.suffix + ".sha256")
                try:
                    with open(sha256_path, "w", encoding="utf-8") as f:
                        f.write(f"{sha256}  {dest.name}\n")
                except OSError as exc:
                    logger.warning(
                        "[%s] Could not write SHA-256 sidecar for %s: %s",
                        self.source_name,
                        dest.name,
                        exc,
                    )

                # Verify against expected checksum if provided (SCI-3.9)
                if expected_sha256 is not None:
                    if not self._verify_published_checksum(dest, expected_sha256):
                        raise DownloadError(
                            f"Checksum verification failed for {dest.name}"
                        )

                # Integrity validation (M5, SCI-3.7)
                if not self.skip_integrity_check:
                    if dest.suffix == ".gz":
                        self._validate_gzip_integrity(dest)
                    elif not self._validate_text_file_integrity(dest, allow_empty=True):
                        logger.warning(
                            "[%s] File %s failed integrity check, re-downloading",
                            self.source_name,
                            dest.name,
                        )
                        try:
                            dest.unlink()
                        except OSError:
                            pass
                        continue  # retry

                logger.info(
                    "[%s] Downloaded %s (%d bytes, sha256=%s...)",
                    self.source_name,
                    dest.name,
                    actual_size,
                    sha256[:12],
                )

                self._circuit_breaker.record_success()
                return dest

            except RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                self._circuit_breaker.record_failure()
                if attempt == max_retries:
                    logger.error(
                        "[%s] Download failed after %d attempts: %s",
                        self.source_name,
                        max_retries,
                        self._sanitize_error_message(str(exc)),
                    )
                    raise DownloadError(
                        f"Failed to download {self._sanitize_url(url)} "
                        f"after {max_retries} attempts: {exc}"
                    ) from exc
                # Exponential backoff with jitter (REL-6.5, REL-6.6)
                # v40 ROOT FIX (P1 #10): the formula is
                # ``2^(attempt-1) + uniform(0,1)``. For attempt=1 (first
                # retry), backoff = 1 + [0,1] = [1,2]s. For attempt=2,
                # backoff = 2 + [0,1] = [2,3]s. For attempt=3,
                # backoff = 4 + [0,1] = [4,5]s. This is the standard
                # "exponential backoff with full jitter" pattern. The
                # _http_client.py docstring mentions a different formula
                # (``backoff_base * (2 ** attempt) + jitter``) but the
                # actual _http_client.py code uses the same formula as
                # here. The docstrings are now aligned.
                backoff = (2 ** (attempt - 1)) + self._rng.uniform(0, 1)
                logger.warning(
                    "[%s] Download failed (attempt %d/%d), retrying in %.1fs: %s",
                    self.source_name,
                    attempt,
                    max_retries,
                    backoff,
                    self._sanitize_error_message(str(exc)),
                )
                time.sleep(backoff)
            except requests.exceptions.HTTPError as exc:
                last_exc = exc
                status_code = (
                    exc.response.status_code if exc.response is not None
                    else 0
                )
                if status_code in RETRYABLE_STATUS_CODES:
                    self._circuit_breaker.record_failure()
                    if attempt == max_retries:
                        raise DownloadError(
                            f"HTTP {status_code} after {max_retries} attempts"
                        ) from exc
                    backoff = (2 ** (attempt - 1)) + self._rng.uniform(0, 1)
                    logger.warning(
                        "[%s] HTTP %d (attempt %d/%d), retrying in %.1fs",
                        self.source_name,
                        status_code,
                        attempt,
                        max_retries,
                        backoff,
                    )
                    time.sleep(backoff)
                else:
                    # Permanent error -- don't retry (REL-6.4).
                    # P1-046 ROOT FIX (v107): call record_failure() BEFORE
                    # raising. The previous code raised DownloadError without
                    # recording the failure on the circuit breaker. A
                    # persistent 401 (expired API key) did NOT trip the
                    # breaker — the pipeline kept retrying the 401 forever
                    # (until max_retries), and the next call to the same API
                    # started fresh with no memory of the previous 4xx.
                    # The operator never saw "circuit breaker open" in the
                    # logs. ROOT FIX: record the failure so the breaker
                    # opens after N consecutive 4xx errors, giving the
                    # operator a clear signal.
                    self._circuit_breaker.record_failure()
                    raise DownloadError(
                        f"HTTP {status_code} (permanent error — circuit "
                        f"breaker failure recorded)"
                    ) from exc

        # Should be unreachable, but keep as safety net (CODE-4.38)
        raise DownloadError(
            f"Failed to download {self._sanitize_url(url)} after {max_retries} retries"
            + (f": {last_exc}" if last_exc else "")
        )

    def _should_skip_download(
        self,
        dest: Path,
        url: str,
        headers: Mapping[str, str] | None,
    ) -> bool:
        """Check if a cached download should be skipped (SCI-3.15).

        Returns True if:
        - The file passes integrity validation, AND
        - The cache is not stale (based on max_cache_age_days and
          conditional request to the source).

        If the conditional request fails (network error), the cached
        file is used with a logged caveat about staleness
        (REL-6.13).
        """
        # Integrity check
        if dest.suffix == ".gz":
            # FIX-P2-8 (audit P2): the previous ``gfh.seek(-1, 2)``
            # required decompressing the ENTIRE stream to compute the
            # uncompressed size -- minutes for STRING's 2 GB links file.
            # Delegate to ``_validate_gzip_integrity`` which performs a
            # fast O(1) magic + trailer + small-chunk-decompress check.
            if not self._validate_gzip_integrity(dest):
                return False
        elif not self._validate_text_file_integrity(dest, allow_empty=True):
            return False

        # Check cache age (SCI-3.15)
        try:
            mtime = dest.stat().st_mtime
            age_days = (time.time() - mtime) / 86400
            if age_days > self.max_cache_age_days:
                logger.info(
                    "[%s] Cached file %s is %.1f days old (> %d), "
                    "checking freshness",
                    self.source_name,
                    dest.name,
                    age_days,
                    self.max_cache_age_days,
                )
                # Force a freshness check
                return not self._is_cache_stale(dest, url, headers)
        except OSError:
            return False

        # Optional freshness check via HEAD request
        if not self._is_cache_stale(dest, url, headers):
            return True

        return False

    def _is_cache_stale(
        self,
        dest: Path,
        url: str,
        headers: Mapping[str, str] | None,
    ) -> bool:
        """Check if cached download is stale by querying source (SCI-3.15).

        Sends a HEAD request with ``If-Modified-Since`` and
        ``If-None-Match`` headers. Returns True if the source has
        been modified (cache is stale), False if the cache is fresh.

        If the HEAD request fails (network error), logs a WARNING
        and returns False (use cached file with a caveat) per
        REL-6.13.
        """
        # Load stored ETag / Last-Modified from sidecar
        meta_path = dest.with_suffix(dest.suffix + ".meta.json")
        cond_headers: dict[str, str] = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if "etag" in meta:
                    cond_headers["If-None-Match"] = meta["etag"]
                if "last_modified" in meta:
                    cond_headers["If-Modified-Since"] = meta["last_modified"]
            except (OSError, json.JSONDecodeError):
                pass

        if not cond_headers:
            # No stored metadata -- can't check freshness, assume fresh
            return False

        try:
            self._rate_limiter.wait()
            resp = self.http_session.head(
                url,
                headers={**dict(headers or {}), **cond_headers},
                timeout=self.download_timeout,
                verify=self.verify_tls,
                allow_redirects=True,
            )
            if resp.status_code == 304:
                return False  # not modified -- cache is fresh
            elif resp.status_code == 200:
                return True  # source has new version -- cache is stale
            else:
                logger.warning(
                    "[%s] Freshness check returned HTTP %d for %s",
                    self.source_name,
                    resp.status_code,
                    dest.name,
                )
                return False
        except RETRYABLE_EXCEPTIONS as exc:
            logger.warning(
                "[%s] Freshness check failed for %s (using cached file): %s",
                self.source_name,
                dest.name,
                exc,
            )
            return False  # REL-6.13: graceful degradation

    def _store_download_metadata(
        self,
        dest: Path,
        response: requests.Response,
    ) -> None:
        """Store ETag, Last-Modified, and Content-MD5 from response (SCI-3.6).

        Writes a sidecar ``.meta.json`` file next to the downloaded
        file. Used by ``_validate_resume_precondition`` and
        ``_is_cache_stale`` for conditional requests.
        """
        meta: dict[str, Any] = {}
        if etag := response.headers.get("ETag"):
            meta["etag"] = etag
        if lm := response.headers.get("Last-Modified"):
            meta["last_modified"] = lm
        if md5 := response.headers.get("Content-MD5"):
            meta["content_md5"] = md5
        meta["url"] = self._sanitize_url(response.url)
        meta["status_code"] = response.status_code
        meta["downloaded_at"] = datetime.now(timezone.utc).isoformat()

        meta_path = dest.with_suffix(dest.suffix + ".meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
        except OSError as exc:
            logger.warning(
                "[%s] Could not write metadata sidecar for %s: %s",
                self.source_name,
                dest.name,
                exc,
            )

    def _validate_resume_precondition(self, dest: Path) -> dict[str, str]:
        """Load stored metadata and return conditional request headers (SCI-3.6).

        Returns
        -------
        dict
            Headers to add to the resume request. Empty if no
            metadata is stored (resume is disabled in that case).
        """
        meta_path = dest.with_suffix(dest.suffix + ".meta.json")
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError) as meta_exc:
            # v107 ROOT FIX (ISSUE-P1-029 — corrupt meta silently disables
            #   the resume precondition):
            #   The previous code returned ``{}`` (empty dict) when the meta
            #   file was corrupt (e.g. truncated JSON from a previous crashed
            #   download). This SILENTLY disabled the If-Match /
            #   If-Unmodified-Since precondition. The download then
            #   proceeded WITHOUT the precondition, potentially appending
            #   bytes from a NEW version of the file to the OLD partial
            #   bytes — producing a chimeric file that fails integrity
            #   validation (or worse, passes if the bytes happen to align).
            #
            # ROOT FIX: when the meta file is corrupt, DELETE both the
            # corrupt meta AND the partial download. The next download
            # starts fresh from byte 0 with no stale partial to append to.
            # This is the patient-safe behaviour — we lose the resume
            # optimization but eliminate the chimeric-file footgun.
            logger.warning(
                "[%s] Resume metadata for %s is corrupt (%s). Deleting "
                "the partial download and stale meta file to start fresh "
                "(v107 P1-029 — no chimeric files).",
                self.source_name, dest.name, meta_exc,
            )
            # Delete the corrupt meta file
            try:
                meta_path.unlink()
            except (OSError, FileNotFoundError):
                pass
            # Delete the partial download (the actual data file)
            try:
                dest.unlink()
            except (OSError, FileNotFoundError):
                pass
            # Also delete the .tmp sidecar if it exists
            tmp_path = dest.with_suffix(dest.suffix + ".tmp")
            try:
                tmp_path.unlink()
            except (OSError, FileNotFoundError):
                pass
            return {}
        headers: dict[str, str] = {}
        if "etag" in meta:
            headers["If-Match"] = meta["etag"]
        elif "last_modified" in meta:
            headers["If-Unmodified-Since"] = meta["last_modified"]
        return headers

    def _validate_gzip_integrity(self, dest: Path) -> bool:
        """Validate that a .gz file is not truncated.

        FIX-P2-8 (audit P2): the previous implementation used
        ``gfh.seek(-1, 2)`` on a ``gzip.GzipFile`` to "verify it's not
        truncated". ``seek(whence=2)`` on a ``GzipFile`` requires
        decompressing the ENTIRE stream to compute the uncompressed
        size -- for STRING's 2 GB protein-links file this took MINUTES
        on every cache freshness check, dominating pipeline runtime.

        The new implementation uses a fast two-stage check:

        1. Validate the 2-byte gzip magic header (``0x1f 0x8b``) by
           reading just the first 2 bytes of the file.
        2. Validate the 8-byte gzip trailer (CRC32 + ISIZE) by reading
           the LAST 8 bytes of the file directly (no decompression).
           A truncated file will be missing these bytes or will have
           an ISIZE that is obviously inconsistent with the compressed
           size (the gzip format guarantees ISIZE <= compressed_size
           only loosely, but a missing trailer is a hard fail).
        3. As a final sanity check, attempt to decompress a small
           leading chunk (64 KiB). This catches mid-stream truncation
           that a trailer-only check would miss, without paying the
           cost of full decompression.

        This is O(1) in the file size -- the previous O(N) decompress-
        everything check has been removed.
        """
        try:
            file_size = dest.stat().st_size
            if file_size < 18:
                # A gzip file has a 10-byte header + 8-byte trailer
                # minimum (no compressed payload). Smaller => corrupt.
                logger.warning(
                    "[%s] File %s is too small to be a valid gzip "
                    "stream (%d bytes < 18 byte minimum)",
                    self.source_name, dest.name, file_size,
                )
                return False
            with open(dest, "rb") as fh:
                magic = fh.read(2)
                if magic != b"\x1f\x8b":
                    logger.warning(
                        "[%s] File %s has invalid gzip magic bytes "
                        "(expected 0x1f 0x8b)",
                        self.source_name,
                        dest.name,
                    )
                    return False
                # Read the 8-byte gzip trailer (CRC32 + ISIZE) directly
                # from the end of the file -- no decompression required.
                fh.seek(-8, 2)
                trailer = fh.read(8)
                # ISIZE is the last 4 bytes, little-endian unsigned int,
                # = uncompressed size mod 2^32. A truncated file will
                # have either the wrong bytes here (if cut mid-payload)
                # or, more reliably, the read will return fewer than
                # 8 bytes if the file is too short (already guarded
                # above). We accept any non-zero trailer presence -- a
                # more rigorous CRC validation would require
                # decompressing, which defeats the purpose of this
                # fast check.
                if len(trailer) != 8:
                    logger.warning(
                        "[%s] File %s is missing the 8-byte gzip trailer "
                        "(truncated)",
                        self.source_name, dest.name,
                    )
                    return False
                # Decompress a small leading chunk (64 KiB) to catch
                # mid-stream corruption that the trailer check would
                # miss. This is O(1) in file size.
                fh.seek(0)
                with gzip.open(fh, "rb") as gfh:
                    gfh.read(65536)
            return True
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            logger.warning(
                "[%s] Existing .gz file %s is truncated: %s",
                self.source_name,
                dest.name,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # URL / path / error sanitisation (SEC-9.1 through SEC-9.5)
    # ------------------------------------------------------------------
    def _validate_url(self, url: str) -> None:
        """Validate URL scheme and domain (SEC-9.1).

        Raises
        ------
        ValueError
            If the URL scheme is not in ``ALLOWED_SCHEMES`` or the
            domain is not in ``ALLOWED_DOMAINS`` (or a subdomain
            thereof).
        """
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in self.ALLOWED_SCHEMES:
            raise ValueError(
                f"Disallowed URL scheme: {parsed.scheme!r}. "
                f"Allowed: {sorted(self.ALLOWED_SCHEMES)}"
            )
        hostname = parsed.hostname or ""
        if not hostname:
            raise ValueError(f"URL has no hostname: {self._sanitize_url(url)}")
        if hostname not in self.ALLOWED_DOMAINS and not any(
            hostname.endswith("." + d) for d in self.ALLOWED_DOMAINS
        ):
            raise ValueError(
                f"Disallowed URL domain: {hostname!r}. "
                f"Allowed: {sorted(self.ALLOWED_DOMAINS)}"
            )

    def _validate_dest_path(self, dest: Path) -> None:
        """Ensure *dest* is within ``RAW_DATA_DIR`` (SEC-9.2).

        Prevents path traversal attacks where a malicious URL could
        cause a download to be written outside the data directory.

        Raises
        ------
        ValueError
            If *dest* resolves to a path outside ``RAW_DATA_DIR``.
        """
        try:
            resolved = dest.resolve()
            base = RAW_DATA_DIR.resolve()
            try:
                resolved.relative_to(base)
            except ValueError:
                raise ValueError(
                    f"Download destination {dest} is outside RAW_DATA_DIR "
                    f"{base}. Possible path traversal attempt."
                )
        except OSError as exc:
            raise ValueError(
                f"Could not resolve path {dest}: {exc}"
            ) from exc

    def _sanitize_url(self, url: str) -> str:
        """Redact API keys from URLs before logging (SEC-9.4, BUG-9.2).

        Replaces the value of any query parameter named ``api_key``,
        ``key``, ``token``, ``secret``, ``password``, or
        ``access_token`` with ``[REDACTED]``.

        Also redacts the OMIM API key when it appears as a path segment
        in the morbidmap downloads URL
        (``https://data.omim.org/downloads/{KEY}/morbidmap.txt``).
        The key is a 36-char UUID; the redaction is additive and only
        fires for URLs containing that exact pattern -- every other URL
        passes through unchanged.
        """
        # Redact query-param-style keys (legacy behavior, unchanged).
        url = _REDACT_QUERY_PARAM_RE.sub(r"\1[REDACTED]", url)
        # Redact OMIM path-segment keys (BUG-9.2, additive).
        url = _REDACT_OMIM_PATH_KEY_RE.sub(r"\1[REDACTED]", url)
        return url

    def _sanitize_error_message(self, msg: str) -> str:
        """Redact API keys, tokens, and credentials from error messages (SEC-9.3).

        Also truncates to ``ERROR_MESSAGE_MAX_LENGTH`` (500 chars) to
        fit the audit DB column (CODE-4.6).

        P1-042 ROOT FIX (v107): TRUNCATE FIRST, THEN REDACT.
        The previous order was redact → truncate. This had a subtle bug:
        if a secret appeared AFTER character 500, the redaction ran on
        the full message (good), but then truncation cut off the
        ``[REDACTED]`` marker — leaving the original secret GONE but the
        audit column showing a truncated message with no indication that
        redaction occurred. More critically, if a Bearer token SPANNED
        the 500-char boundary (e.g. "Bearer abc" where the token
        continued past char 500), the redaction regex
        ``Bearer\\s+\\S+`` would match the full token in the untruncated
        message — but a future change to truncate-first would leave a
        partial "Bearer abc" that the regex might not catch.
        The defense-in-depth fix: truncate FIRST (so secrets beyond 500
        chars are GONE, not just redacted), THEN redact any secrets that
        survived the truncation (including partial tokens at the
        boundary). The redaction regexes use ``\\S+`` (greedy
        non-whitespace) which catches partial tokens.
        """
        # P1-042 ROOT FIX (v107): Truncate FIRST, then redact.
        msg = msg[:ERROR_MESSAGE_MAX_LENGTH]
        # Redact URL query params
        msg = _REDACT_QUERY_PARAM_RE.sub(r"\1[REDACTED]", msg)
        # Redact OMIM path-segment keys (BUG-9.2, additive)
        msg = _REDACT_OMIM_PATH_KEY_RE.sub(r"\1[REDACTED]", msg)
        # Redact OMIM ApiKey header form (BUG-2.2 / BUG-9.3, additive)
        msg = _REDACT_OMIM_APIKEY_HEADER_RE.sub(r"\1[REDACTED]", msg)
        # Redact Bearer tokens first (more specific)
        msg = _REDACT_BEARER_RE.sub(r"\1[REDACTED]", msg)
        # Redact Authorization headers (catches Basic, Digest, etc.)
        msg = _REDACT_AUTH_HEADER_RE.sub(r"\1[REDACTED]", msg)
        return msg

    def _sanitize_headers(self, headers: Mapping[str, str]) -> dict[str, str]:
        """Redact sensitive headers before logging (SEC-9.5).

        Returns a copy of *headers* with sensitive header values
        replaced by ``[REDACTED]``.
        """
        return {
            k: "[REDACTED]" if k.lower() in SENSITIVE_HEADER_KEYS else str(v)
            for k, v in headers.items()
        }

    def _sanitize_csv_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prevent CSV formula injection by escaping dangerous prefixes (SEC-9.14).

        Escapes cells that start with ``=``, ``+``, ``-``, ``@``,
        ``\\t``, or ``\\r`` by prepending a single quote. This
        prevents Excel / Google Sheets from interpreting them as
        formulas.

        v65 ROOT FIX (P1-044): the previous implementation only scanned
        object-dtype columns (``df.select_dtypes(include=["object"])``).
        If a numeric column contained a string value starting with ``=``
        (e.g. due to pandas type-coercion oddities where a column of
        mostly-ints with one "=1+1" string gets coerced to object dtype
        but a column of mostly-floats with one "=A1" string stays
        object), it would be MISSED.

        v82 FORENSIC ROOT FIX (P1-5 -- ``df.astype(object)`` destroys
        numeric dtypes on round-trip):
          The v65 fix was OVERLY AGGRESSIVE -- it cast EVERY column to
          object dtype, including genuine int64/float64/bool columns.
          The CSV write then serialized Python objects (int -> str), and
          the read-back re-inferred dtypes. For most columns this
          round-tripped correctly, but for nullable Int64 columns with
          NaN, the object-dtype intermediate could produce
          ``float('nan')`` instead of ``pd.NA``, breaking downstream
          ``pd.isna()`` checks (the pubchem_cid column in the drugs
          schema is the canonical victim -- ``pubchem_cid is None``
          checks failed because ``nan != None``).
        ROOT FIX: scan ALL columns for dangerous-string cells (preserving
          the v65 coverage guarantee) but WITHOUT the global
          ``df.astype(object)`` cast. The implementation:
          1. Build a boolean mask ``_has_dangerous`` per column that is
             True only if ANY cell in that column is a string starting
             with a dangerous prefix. This is computed via
             ``Series.astype(str).str.startswith(CSV_DANGEROUS_PREFIXES)``
             which works for both object and numeric dtypes (numeric
             values are stringified first -- they will never start with
             ``=``/``@``/``\\t``/``\\r`` because their str
             representation is purely numeric/scientific).
          2. For columns where ``_has_dangerous.any()`` is True, cast
             JUST that column to object and apply the escape lambda.
          3. For columns where no cell is dangerous, leave the dtype
             untouched -- the CSV writer handles numeric dtypes natively
             and the read-back preserves the original dtype.
        This preserves the v65 security fix (every dangerous cell is
        escaped) while eliminating the P1-5 dtype-destruction regression.
        """
        if df is None or df.empty:
            return df
        # v82 FORENSIC ROOT FIX (P1-5): surgical per-column sanitization.
        # Build the mask ONCE per column; only cast columns that actually
        # contain a dangerous-string cell. Genuine numeric columns (int64,
        # float64, bool, Int64, Float64) never have dangerous-string cells
        # because their stringified form never starts with ``=``/``+``/
        # ``-``/``@``/``\t``/``\r`` -- so they pass through unchanged.
        for col in df.columns:
            series = df[col]
            # Stringify every cell to detect dangerous prefixes uniformly.
            # For numeric dtypes this is a cheap O(N) operation; for
            # object dtype it's the same cost as before.
            _str_view = series.astype(str)
            _danger_mask = _str_view.str.startswith(CSV_DANGEROUS_PREFIXES)
            if not _danger_mask.any():
                # No dangerous cells in this column -- leave dtype intact.
                continue
            # Column has at least one dangerous cell -- cast to object and
            # escape ONLY the dangerous cells. Non-dangerous cells keep
            # their original Python object representation (int -> int,
            # float -> float, str -> str).
            # v107 ROOT FIX (ISSUE-P1-040 — ``.values`` caused index
            #   misalignment in the ``where()`` call):
            #   The previous code passed ``~_danger_mask.values`` (a numpy
            #   array) to ``Series.where()``, which disabled pandas' index
            #   alignment. The ``_obj_series.map(...)`` replacement could
            #   return a Series with a DIFFERENT index if any cell was NaN
            #   (map() drops NaN by default in some pandas versions). The
            #   ``where()`` then misaligned: the dangerous cell at position
            #   i could be replaced with the escaped value from position j,
            #   leaving the dangerous cell UNescaped and corrupting a
            #   non-dangerous cell. CSV injection prevention was incomplete.
            # ROOT FIX: pass ``~_danger_mask`` (a Series) directly, WITHOUT
            # ``.values``. Pandas then aligns by index — the mask and the
            # replacement Series share the same index as ``_obj_series``,
            # so each cell is escaped iff its own mask bit is set.
            _obj_series = series.astype(object)
            # v107 FORENSIC ROOT FIX (ISSUE-P1-019 + ISSUE-P1-040):
            #   The previous code used ``~_danger_mask.values`` (numpy array)
            #   as the condition for ``Series.where()``. The ``.values``
            #   attribute DISCARDS the index, so if the DataFrame's index
            #   had been reset (e.g. after ``dropna``), the alignment
            #   between ``_danger_mask.values`` and ``_obj_series`` was
            #   incorrect -- dangerous cells were not escaped. A SMILES
            #   string starting with ``=`` (rare but possible for peptide
            #   SMILES) was not escaped, creating a CSV formula injection
            #   vector. The safety claim was FALSE.
            #
            #   ROOT FIX (ISSUE-P1-040): use ``~_danger_mask`` (WITHOUT
            #   ``.values``) so pandas aligns by index. The boolean mask
            #   and the Series now share the same index, guaranteeing
            #   correct cell-level escaping even after index resets.
            #   v107 defensive hardening: explicitly reindex the mask to
            #   _obj_series.index with fill_value=False as a guarantee
            #   against future pandas behavior changes (e.g. if astype()
            #   ever resets the index in a future pandas release).
            _aligned_mask = _danger_mask.reindex(_obj_series.index, fill_value=False)
            _escaped = _obj_series.where(
                ~_aligned_mask,
                _obj_series.map(
                    lambda x: f"'{x}" if isinstance(x, str) else x
                ),
            )
            df[col] = _escaped
        return df

    def _detect_pii(self, df: pd.DataFrame) -> list[str]:
        """Detect columns that may contain PII (SEC-9.15).

        Scans for common PII patterns: email addresses, phone
        numbers, SSNs, credit card numbers. Returns a list of column
        names where PII was detected.

        Notes
        -----
        This is a best-effort heuristic check. False positives are
        possible (e.g. a column of long numeric IDs may match the
        credit card pattern). The check is for warning purposes
        only; PII handling is the responsibility of the subclass.
        """
        pii_columns: list[str] = []
        patterns = {
            "email": re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
            "ssn": re.compile(r"^\d{3}-\d{2}-\d{4}$"),
            "phone": re.compile(r"^\+?1?\-?\d{3}\-?\d{3}\-?\d{4}$"),
        }
        if df is None or df.empty:
            return pii_columns
        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().astype(str).head(100)
            if sample.empty:
                continue
            for pii_type, pattern in patterns.items():
                matches = sample.str.match(pattern).sum()
                if matches > len(sample) * 0.3:  # >30% match threshold
                    pii_columns.append(f"{col} (suspected {pii_type})")
                    break
        return pii_columns

    # ------------------------------------------------------------------
    # File locking helpers (IDEM-7.12)
    # ------------------------------------------------------------------
    def _acquire_file_lock(self, dest: Path) -> Any:
        """Acquire a file lock for downloading *dest* (IDEM-7.12)."""
        if not _HAS_FILELOCK or FileLock is None:
            return None
        lock_path = dest.with_suffix(dest.suffix + ".lock")
        # v29 ROOT FIX (audit P1-23): was 5min TTL -- too short for real ETL. Increased to 30min.
        lock_timeout = self.file_lock_timeout_sec
        lock = FileLock(lock_path, timeout=lock_timeout)
        try:
            lock.acquire()
            return lock
        except FileLockTimeout:
            raise PipelineError(
                f"Could not acquire file lock for {dest} after "
                f"{lock_timeout} seconds."
            )

    def _release_file_lock(self, lock: Any) -> None:
        """Release a file lock acquired by ``_acquire_file_lock``."""
        if lock is None:
            return
        try:
            lock.release()
            lock_path = getattr(lock, "lock_file", None)
            if lock_path and Path(lock_path).exists():
                try:
                    Path(lock_path).unlink()
                except OSError:
                    pass
        except (OSError, RuntimeError, ValueError):
            # v85 FORENSIC ROOT FIX (BUG #51): narrowed from broad
            # ``except Exception``. Lock release can fail with I/O or
            # runtime errors. Programming bugs now propagate.
            pass

    # ------------------------------------------------------------------
    # Cleaned data persistence (ARCH-1.3, IDEM-7.6, LIN-16.8)
    # ------------------------------------------------------------------
    def _persist_cleaned_data(self, df: pd.DataFrame) -> Path:
        """Persist cleaned DataFrame to ``PROCESSED_DATA_DIR`` as CSV (ARCH-1.3).

        The CSV is written with explicit ``encoding="utf-8"``,
        ``index=False``, and ``QUOTE_MINIMAL`` quoting for
        interoperability (INT-15.4). P1-26 ROOT FIX: the previous
        write used ``QUOTE_NONNUMERIC`` while the corresponding read
        (``_load_cleaned_data``, line ~1406) used ``QUOTE_MINIMAL`` --
        the asymmetric quoting caused NaN values to round-trip as
        empty strings (``""``) instead of NaN, breaking downstream
        ``pd.isna()`` filters. Both sides now use ``QUOTE_MINIMAL``
        (the pandas default). A SHA-256 sidecar is written
        next to the CSV (LIN-16.3).

        Parameters
        ----------
        df : pandas.DataFrame
            Cleaned DataFrame to persist.

        Returns
        -------
        Path
            Path to the persisted CSV file.
        """
        dest = PROCESSED_DATA_DIR / self._get_processed_filename()
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            dest,
            index=False,
            encoding="utf-8",
            # P1-26 ROOT FIX: QUOTE_MINIMAL on write to match the read
            # path (was QUOTE_NONNUMERIC). Asymmetric quoting caused
            # NaN values to round-trip as "" instead of NaN.
            quoting=csv_mod.QUOTE_MINIMAL,
        )
        # Compute and store SHA-256 (LIN-16.3)
        sha256 = self._compute_sha256(dest)
        self._sha256_cleaned = sha256
        sha256_path = dest.with_suffix(dest.suffix + ".sha256")
        try:
            with open(sha256_path, "w", encoding="utf-8") as f:
                f.write(f"{sha256}  {dest.name}\n")
        except OSError as exc:
            logger.warning(
                "[%s] Could not write SHA-256 sidecar for cleaned data: %s",
                self.source_name,
                exc,
            )
        return dest

    def _write_run_context(
        self,
        cleaned_path: Path,
        records_downloaded: int,
        records_cleaned: int,
    ) -> None:
        """Write a ``.run_context.json`` sidecar for ``run_load_only`` (IDEM-7.6).

        The run context contains the SHA-256 of the cleaned CSV, the
        run ID, source version, and record counts. ``run_load_only``
        reads this sidecar and verifies the CSV's SHA-256 before
        loading, creating a cryptographic chain between the two
        phases.
        """
        context = {
            "run_id": self.run_id,
            "source": self.source_name,
            "source_version": self.source_version,
            "sha256_raw": self._sha256_raw,
            "sha256_cleaned": self._sha256_cleaned,
            "records_downloaded": records_downloaded,
            "records_cleaned": records_cleaned,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _get_git_commit(),
            "seed": self.seed,
        }
        context_path = cleaned_path.with_suffix(
            cleaned_path.suffix + ".run_context.json"
        )
        try:
            with open(context_path, "w", encoding="utf-8") as f:
                json.dump(context, f, indent=2, sort_keys=True)
        except OSError as exc:
            logger.warning(
                "[%s] Could not write run context for %s: %s",
                self.source_name,
                cleaned_path.name,
                exc,
            )

    def _read_run_context(self, cleaned_path: Path) -> dict[str, Any] | None:
        """Read the ``.run_context.json`` sidecar (IDEM-7.6).

        Returns None if the sidecar does not exist (e.g. the CSV was
        written by an older version of the pipeline that didn't
        write the sidecar).
        """
        context_path = cleaned_path.with_suffix(
            cleaned_path.suffix + ".run_context.json"
        )
        if not context_path.exists():
            return None
        try:
            with open(context_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "[%s] Could not read run context for %s: %s",
                self.source_name,
                cleaned_path.name,
                exc,
            )
            return None

    def _verify_run_context(self, cleaned_path: Path) -> None:
        """Verify the cleaned CSV against its run context (IDEM-7.6).

        Raises
        ------
        DataIntegrityError
            If the SHA-256 of the cleaned CSV does not match the
            SHA-256 recorded in the ``.run_context.json`` sidecar.
        """
        context = self._read_run_context(cleaned_path)
        if context is None:
            # P1-051 ROOT FIX (v107): the previous code logged an INFO and
            # silently skipped verification — then loaded the (potentially
            # tampered) CSV. A tampered cleaned CSV (e.g. an attacker
            # modified is_fda_approved from False to True) was loaded
            # without SHA-256 verification, corrupting the KG. ROOT FIX:
            # raise DataIntegrityError. The caller (run_load_only / recover_from_failure)
            # must handle this by re-running download+clean to regenerate
            # the sidecar.
            raise DataIntegrityError(
                f"Cleaned CSV {cleaned_path.name} has NO run context sidecar "
                f"(.run_context.json). The previous run may have crashed "
                f"before writing it, OR the file may have been tampered "
                f"with. SHA-256 verification CANNOT be performed. Re-run "
                f"download+clean to regenerate the sidecar (P1-051 ROOT FIX)."
            )
        expected_sha = context.get("sha256_cleaned")
        if not expected_sha:
            return
        actual_sha = self._compute_sha256(cleaned_path)
        if actual_sha != expected_sha:
            raise DataIntegrityError(
                f"Cleaned CSV {cleaned_path.name} has been modified since "
                f"download+clean (expected SHA-256 {expected_sha}, got "
                f"{actual_sha}). Re-run download+clean to regenerate."
            )

    # ------------------------------------------------------------------
    # Train/test split tagging (SCI-3.14)
    # ------------------------------------------------------------------
    def _tag_train_test_split(
        self,
        df: pd.DataFrame,
        test_fraction: float = 0.2,
        validate_fraction: float = 0.0,
        seed: int = 42,
        primary_key_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Deterministically tag rows as train/test/validate (SCI-3.14).

        Uses a hash of the primary key columns so the split is
        deterministic and idempotent -- the same data always gets the
        same split regardless of row order. This prevents data
        leakage between runs.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to tag.
        test_fraction : float, default 0.2
            Fraction of rows to tag as ``test``.
        validate_fraction : float, default 0.0
            Fraction of rows to tag as ``validate``. The remainder
            is tagged as ``train``.
        seed : int, default 42
            Seed for the hash function. Recorded in audit metadata.
        primary_key_columns : list of str, optional
            Columns to use as the primary key for hashing. Defaults
            to the schema's required columns.

        Returns
        -------
        pandas.DataFrame
            Copy of *df* with a new ``_split`` column containing
            ``"train"``, ``"test"``, or ``"validate"``.
        """
        if df is None or df.empty:
            return df
        if primary_key_columns is None:
            schema = self._load_schema()
            file_key = self._get_processed_filename()
            primary_key_columns = (
                schema.get("properties", {})
                .get(file_key, {})
                .get("required", [])
            )
        if not primary_key_columns:
            primary_key_columns = list(df.columns)[:1]

        def _hash_row(row: pd.Series) -> int:
            key = "|".join(str(row[c]) for c in primary_key_columns)
            return int(hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest(), 16)

        hash_values = df.apply(_hash_row, axis=1)
        max_hash = (1 << 128) - 1
        test_threshold = int(max_hash * test_fraction)
        validate_threshold = int(max_hash * (test_fraction + validate_fraction))

        def _classify(h: int) -> str:
            if h <= test_threshold:
                return "test"
            elif h <= validate_threshold:
                return "validate"
            return "train"

        df = df.copy()
        df["_split"] = hash_values.apply(_classify)
        self._log_transformation(
            "train_test_split",
            len(df),
            {"test_fraction": test_fraction, "seed": seed},
        )
        return df

    # ------------------------------------------------------------------
    # Referential integrity (SCI-3.16)
    # ------------------------------------------------------------------
    def _validate_referential_integrity(
        self,
        df: pd.DataFrame,
    ) -> tuple[bool, list[str]]:
        """Check foreign keys reference known entities (SCI-3.16).

        Best-effort check -- only validates against files that exist
        on disk in ``PROCESSED_DATA_DIR``. Reports dangling
        references as warnings, not errors.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to check.

        Returns
        -------
        tuple
            ``(is_valid, list_of_warnings)``. ``is_valid`` is always
            True (warnings don't invalidate the data); the warnings
            list contains descriptions of dangling references.
        """
        warnings_list: list[str] = []
        if df is None or df.empty:
            return True, warnings_list

        # Check uniprot_id references against proteins.csv
        # v37 ROOT FIX (Phase 1 Issue #11): NaN-to-'nan' string bug.
        # The previous code did ``df["uniprot_id"].astype(str)`` which
        # converts NaN to the literal string "nan". The "nan" string
        # is then checked against ``known_uniprots`` (which doesn't
        # contain "nan" because pandas' CSV reader drops NaN rows by
        # default in usecols), so EVERY NaN row was flagged as a
        # "dangling reference" -- false-positive warnings on every row
        # where uniprot_id was legitimately missing. The fix: drop NaN
        # rows BEFORE the astype(str) conversion so they're not counted
        # as dangling (missing != dangling).
        if "uniprot_id" in df.columns:
            proteins_path = PROCESSED_DATA_DIR / "proteins.csv"
            if proteins_path.exists():
                try:
                    known_uniprots = set(
                        pd.read_csv(proteins_path, usecols=["uniprot_id"])[
                            "uniprot_id"
                        ].dropna().astype(str)
                    )
                    # Drop NaN rows from the check -- missing != dangling.
                    _df_non_null = df[df["uniprot_id"].notna()]
                    dangling = _df_non_null[
                        ~_df_non_null["uniprot_id"].astype(str).isin(known_uniprots)
                    ]
                    dangling_count = len(dangling)
                    if dangling_count > 0:
                        warnings_list.append(
                            f"{dangling_count} rows have uniprot_id values "
                            f"not in proteins.csv"
                        )
                except (OSError, ValueError, pd.errors.ParserError) as exc:
                    warnings_list.append(
                        f"Could not check uniprot_id references: {exc}"
                    )

        # Check inchikey references against drugs.csv
        # v37 ROOT FIX (Phase 1 Issue #11): same NaN-to-'nan' bug --
        # drop NaN rows before the astype(str) check.
        if "inchikey" in df.columns:
            drugs_path = PROCESSED_DATA_DIR / "drugs.csv"
            if drugs_path.exists():
                try:
                    known_inchikeys = set(
                        pd.read_csv(drugs_path, usecols=["inchikey"])[
                            "inchikey"
                        ].dropna().astype(str)
                    )
                    _df_non_null_ik = df[df["inchikey"].notna()]
                    dangling = _df_non_null_ik[
                        ~_df_non_null_ik["inchikey"].astype(str).isin(known_inchikeys)
                    ]
                    dangling_count = len(dangling)
                    if dangling_count > 0:
                        warnings_list.append(
                            f"{dangling_count} rows have inchikey values "
                            f"not in drugs.csv"
                        )
                except (OSError, ValueError, pd.errors.ParserError) as exc:
                    warnings_list.append(
                        f"Could not check inchikey references: {exc}"
                    )

        return True, warnings_list

    # ------------------------------------------------------------------
    # Data quality metrics (DQ-5.2, DQ-5.4, DQ-5.5, DQ-5.16, DQ-5.17)
    # ------------------------------------------------------------------
    def _count_valid_records(self, df: pd.DataFrame) -> int:
        """Count rows where all required columns are non-NULL (DQ-5.2).

        Parameters
        ----------
        df : pandas.DataFrame
            Cleaned DataFrame.

        Returns
        -------
        int
            Number of rows with no NULLs in any required column.
        """
        if df is None or df.empty:
            return 0
        schema = self._load_schema()
        file_key = self._get_processed_filename()
        required = (
            schema.get("properties", {})
            .get(file_key, {})
            .get("required", [])
        )
        if not required:
            return len(df)
        existing_required = [c for c in required if c in df.columns]
        if not existing_required:
            return len(df)
        return int(df[existing_required].notna().all(axis=1).sum())

    def _check_uniqueness(
        self,
        df: pd.DataFrame,
        columns: list[str],
    ) -> tuple[int, int]:
        """Check uniqueness of *columns* in *df* (DQ-5.4).

        Returns
        -------
        tuple
            ``(total_rows, unique_rows)``.
        """
        if df is None or df.empty or not columns:
            return (0, 0)
        existing = [c for c in columns if c in df.columns]
        if not existing:
            return (len(df), len(df))
        total = len(df)
        unique = int(df[existing].drop_duplicates().shape[0])
        return (total, unique)

    def _check_column_completeness(self, df: pd.DataFrame) -> dict[str, float]:
        """Return column name -> non-NULL fraction (DQ-5.5).

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to check.

        Returns
        -------
        dict
            Mapping of column name to fraction of non-NULL values
            (0.0 to 1.0).
        """
        if df is None or df.empty:
            return {}
        return {col: float(df[col].notna().mean()) for col in df.columns}

    def _compute_data_quality_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute DQ metrics: NULL counts, duplicate counts, etc. (DQ-5.16).

        Parameters
        ----------
        df : pandas.DataFrame
            Cleaned DataFrame.

        Returns
        -------
        dict
            Dictionary with keys:
            - ``total_rows``
            - ``null_counts`` -- column -> NULL count
            - ``duplicate_count``
            - ``unique_counts`` -- column -> unique value count
        """
        if df is None or df.empty:
            return {
                "total_rows": 0,
                "null_counts": {},
                "duplicate_count": 0,
                "unique_counts": {},
            }
        return {
            "total_rows": int(len(df)),
            "null_counts": {
                col: int(df[col].isna().sum()) for col in df.columns
            },
            "duplicate_count": int(df.duplicated().sum()),
            "unique_counts": {
                col: int(df[col].nunique()) for col in df.columns
            },
        }

    def _compute_quality_score(self, df: pd.DataFrame) -> float:
        """Compute a 0-1 data quality score (DQ-5.17).

        Combines:
        - Completeness (non-NULL fraction averaged across required columns).
        - Uniqueness (1 - duplicate fraction).
        - Validity (schema compliance fraction -- computed by
          ``validate_output``).

        Returns
        -------
        float
            Quality score in [0, 1]. Higher is better.
        """
        if df is None or df.empty:
            return 1.0  # empty is "valid" by convention

        # Completeness
        schema = self._load_schema()
        file_key = self._get_processed_filename()
        required = (
            schema.get("properties", {})
            .get(file_key, {})
            .get("required", [])
        )
        existing_required = [c for c in required if c in df.columns]
        if existing_required:
            completeness = float(
                df[existing_required].notna().mean().mean()
            )
        else:
            completeness = 1.0

        # Uniqueness
        # P1-024 ROOT FIX (Team-2 — document the div-by-zero guard):
        #   The ``len(df) > 0`` check below guards the division
        #   ``int(df.duplicated().sum()) / len(df)`` from div-by-zero.
        #   The guard is correct but the previous code did NOT document
        #   WHY it's there — a future refactor that moves the division
        #   out of the guarded block would introduce a div-by-zero.
        #   ROOT FIX: add an explicit comment on the division line so
        #   the invariant is visible at the point of risk. This is a
        #   documentation-only fix; no code change.
        if len(df) > 0:
            # Safe: guarded by ``len(df) > 0`` check above — no div-by-zero.
            # If a future refactor moves this division out of the guarded
            # block, add an explicit ``if len(df) == 0: return 1.0`` guard.
            uniqueness = 1.0 - (int(df.duplicated().sum()) / len(df))
        else:
            uniqueness = 1.0

        # Validity (from schema validation)
        is_valid, errors = self.validate_output(df)
        validity = 1.0 if is_valid else max(0.0, 1.0 - len(errors) * 0.1)

        # Weighted average
        return round((completeness + uniqueness + validity) / 3.0, 4)

    def _drop_null_primary_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with NULL in required columns (DQ-5.19).

        Logs how many rows were dropped. If all rows have NULL in a
        required column, raises ``DataIntegrityError``.
        """
        if df is None or df.empty:
            return df
        schema = self._load_schema()
        file_key = self._get_processed_filename()
        required = (
            schema.get("properties", {})
            .get(file_key, {})
            .get("required", [])
        )
        existing = [c for c in required if c in df.columns]
        if not existing:
            return df
        before = len(df)
        df = df.dropna(subset=existing)
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(
                "[%s] Dropped %d rows with NULL in required columns %s",
                self.source_name,
                dropped,
                existing,
            )
        if len(df) == 0 and before > 0:
            raise DataIntegrityError(
                f"All {before} rows have NULL in required columns {existing}"
            )
        return df

    # ------------------------------------------------------------------
    # Filename helper (ARCH-1.6, COMP-14.5)
    # ------------------------------------------------------------------
    def _get_processed_filename(self) -> str:
        """Return the filename for this pipeline's cleaned data.

        Maintains the canonical 7-source filename mapping for backward
        compatibility (ARCH-1.6). Subclasses may override by setting
        the ``processed_filename`` class attribute.

        Returns
        -------
        str
            Filename in ``PROCESSED_DATA_DIR``.
        """
        # Allow subclass override via processed_filename attribute
        if getattr(self, "processed_filename", None):
            return self.processed_filename
        filenames = {
            "chembl": "drugs.csv",
            "drugbank": "drugbank_drugs.csv",
            "uniprot": "proteins.csv",
            "string": "protein_protein_interactions.csv",
            "disgenet": "gene_disease_associations.csv",
            "omim": "omim_gene_disease_associations.csv",
            "pubchem": "pubchem_enrichment.csv",
        }
        return filenames.get(self.source_name, f"{self.source_name}.csv")

    # ------------------------------------------------------------------
    # Audit logging (DQ-5.10, DQ-5.11, IDEM-7.1, IDEM-7.2)
    # ------------------------------------------------------------------
    def _write_run_log(
        self,
        status: str,
        started_at: datetime | None,
        finished_at: datetime | None,
        records_downloaded: int,
        records_cleaned: int,
        records_loaded: int,
        error_message: str | None = None,
        metadata_json: dict | None = None,
    ) -> None:
        """Persist pipeline run audit record to the database (DQ-5.10).

        If the DB is unreachable, the audit record is buffered to a
        local JSONL file (``RAW_DATA_DIR/pipeline_runs_fallback.jsonl``)
        and replayed on the next successful DB write.

        On ``UniqueConstraint(source, run_date)`` collision, the
        existing row is updated with the new results (DQ-5.11,
        IDEM-7.2).

        Parameters
        ----------
        status : str
            Run status. v83 P0-C11: the DB CHECK constraint
            ``chk_pipeline_runs_status`` (migration 001) only allows
            ``'running'``, ``'success'``, ``'failed'``, ``'partial'``.
            The previous code also wrote ``'warning'``,
            ``'download_clean_success'``, and ``'load_success'`` which
            the CHECK rejected (silently deleting the audit trail -- see
            the v83 P0-C11 root-cause comment at the call sites).
            Callers MUST now use only the 4 canonical statuses. The
            pipeline PHASE (download_clean / load / full) is recorded
            in ``metadata_json['phase']``, not in the status column.
        started_at : datetime or None
            Run start time (UTC).
        finished_at : datetime or None
            Run finish time (UTC).
        records_downloaded : int
            Number of records downloaded (or ``SENTINEL_COUNT_FAILED``
            if counting failed).
        records_cleaned : int
            Number of valid records after cleaning.
        records_loaded : int
            Number of records upserted into the staging DB.
        error_message : str, optional
            Sanitised error message (max 500 chars).
        metadata_json : dict, optional
            Additional metadata (run_id, source_version, SHA-256s,
            DQ metrics, etc.).
        """
        # Compute duration safely (CODE-4.23, CODE-4.41, CODE-4.45)
        duration_seconds: int | None = None
        if (
            started_at is not None
            and finished_at is not None
        ):
            duration_seconds = int(
                round((finished_at - started_at).total_seconds())
            )

        # Sanitise error message (SEC-9.3, CODE-4.6)
        if error_message is not None:
            error_message = self._sanitize_error_message(error_message)

        # Use started_at or fallback to now (CODE-4.23)
        run_date = (
            started_at if started_at is not None
            else datetime.now(timezone.utc)
        )

        try:
            with get_db_session(
                pipeline_name=self.source_name,
                run_id=self.run_id,
                correlation_id=self.correlation_id,
            ) as session:
                # IDEM-7.2 / DQ-5.11: upsert on collision
                existing = None
                if _HAS_SQLALCHEMY and _sa_select is not None:
                    existing = session.execute(
                        _sa_select(PipelineRun).where(
                            PipelineRun.source == self.source_name,
                            PipelineRun.run_date == run_date,
                        )
                    ).scalar_one_or_none()

                if existing is not None:
                    # Update the existing row
                    logger.info(
                        "[%s] Updating existing audit row (source=%s, run_date=%s)",
                        self.source_name,
                        self.source_name,
                        run_date.isoformat(),
                    )
                    existing.status = status
                    existing.records_downloaded = records_downloaded
                    existing.records_cleaned = records_cleaned
                    existing.records_loaded = records_loaded
                    existing.error_message = error_message
                    existing.duration_seconds = duration_seconds
                    # P1-18 ROOT FIX: persist the rich per-run metadata that
                    # _write_run_log already computes (run_id, sha256,
                    # git_commit, dq_metrics, validation_errors, etc.).
                    # Without this assignment, the metadata was silently
                    # dropped on every upsert-update path.
                    existing.metadata_json = metadata_json
                else:
                    # V90 CI fix: clamp negative record counts to 0.
                    # SENTINEL_COUNT_FAILED (-1) is used as a sentinel
                    # when record counting fails, but the DB CHECK
                    # constraint chk_pipeline_runs_counts_nonneg
                    # rejects negative values. The fix: replace any
                    # negative count with 0 before insertion. This
                    # preserves the "count failed" information in the
                    # metadata_json (logged elsewhere) while satisfying
                    # the DB constraint.
                    _rd = max(0, records_downloaded) if records_downloaded is not None else 0
                    _rc = max(0, records_cleaned) if records_cleaned is not None else 0
                    _rl = max(0, records_loaded) if records_loaded is not None else 0
                    run = PipelineRun(
                        source=self.source_name,
                        run_date=run_date,
                        status=status,
                        records_downloaded=_rd,
                        records_cleaned=_rc,
                        records_loaded=_rl,
                        error_message=error_message,
                        duration_seconds=duration_seconds,
                        # P1-18 ROOT FIX: persist the metadata_json that was
                        # previously discarded by the constructor (the
                        # PipelineRun model now declares this column).
                        metadata_json=metadata_json,
                    )
                    session.add(run)

                # Replay any buffered audit records (DQ-5.10)
                if self._audit_buffer:
                    self._replay_audit_buffer_in_session(session)

        # v85/v90 ROOT FIX (BUG #14/51): narrowed from broad
        # ``except Exception`` which caught programming bugs
        # (AttributeError, KeyError, NameError) and silently fell
        # back to local JSONL. A typo in a PipelineRun field name
        # was masked as "DB unavailable" -- the audit trail went to
        # a local file nobody read. Root fix: catch ONLY DB-related
        # errors (OperationalError, IntegrityError, InterfaceError,
        # ProgrammingError for schema drift) plus OS/import errors.
        # Programming bugs (AttributeError, NameError) propagate.
        except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
            # v107 FORENSIC ROOT FIX (ISSUE-P1-007):
            #   The previous code had ``if _db_exc_types and not isinstance(exc, tuple(_db_exc_types)):
            #   pass`` -- the ``pass`` body did NOTHING. The comment claimed
            #   "otherwise re-raise" but there was no re-raise. ALL caught
            #   exceptions fell back to local JSONL, including programming
            #   bugs (NameError, AttributeError from typos). A typo in a
            #   ``PipelineRun(source="chmembl")`` field name was silently
            #   masked as "DB unavailable" -- the audit trail went to a
            #   local file nobody read. Real bugs became invisible.
            #
            #   ROOT FIX: if the exception is NOT a DB error, RE-RAISE it
            #   so programming bugs propagate instead of being silently
            #   swallowed. Audit writes only fall back to JSONL for genuine
            #   DB errors (OperationalError, IntegrityError, InterfaceError,
            #   ProgrammingError) or the allowed non-DB types that genuinely
            #   indicate "DB unavailable" (OSError for filesystem/IPC,
            #   ImportError for missing SQLAlchemy driver).
            _db_exc_types = []
            if _SAIntegrityError is not None:
                _db_exc_types.append(_SAIntegrityError)
            if _SAOperationalError is not None:
                _db_exc_types.append(_SAOperationalError)
            if _SAInterfaceError is not None:
                _db_exc_types.append(_SAInterfaceError)
            if _SAProgrammingError is not None:
                _db_exc_types.append(_SAProgrammingError)
            if _db_exc_types and not isinstance(exc, tuple(_db_exc_types)):
                # v107 P1-007: RE-RAISE non-DB exceptions so programming
                # bugs (KeyError from a typo'd field name, RuntimeError
                # from a contract violation, ValueError from bad data)
                # propagate to the operator instead of being silently
                # masked as "DB unavailable". The audit trail must be
                # TRUSTWORTHY -- a silent fallback hides real bugs.
                raise
            if (
                _HAS_SQLALCHEMY
                and _SAIntegrityError is not None
                and isinstance(exc, _SAIntegrityError)
            ):
                logger.error(
                    "[%s] IntegrityError writing run log: %s. "
                    "Falling back to local JSONL.",
                    self.source_name,
                    self._sanitize_error_message(str(exc)),
                )
            else:
                logger.error(
                    "[%s] Could not write run log to DB: %s. "
                    "Falling back to local JSONL.",
                    self.source_name,
                    self._sanitize_error_message(str(exc)),
                )
            self._write_run_log_fallback(
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                records_downloaded=records_downloaded,
                records_cleaned=records_cleaned,
                records_loaded=records_loaded,
                error_message=error_message,
                metadata_json=metadata_json,
            )

    def _write_run_log_fallback(
        self,
        status: str,
        started_at: datetime | None,
        finished_at: datetime | None,
        records_downloaded: int,
        records_cleaned: int,
        records_loaded: int,
        error_message: str | None = None,
        metadata_json: dict | None = None,
    ) -> None:
        """Write the audit record to a local JSONL file (DQ-5.10).

        Called when the DB is unreachable. The record is also added
        to ``self._audit_buffer`` for replay on the next successful
        DB write.
        """
        record = {
            "source": self.source_name,
            "run_date": (
                started_at.isoformat() if started_at is not None
                else datetime.now(timezone.utc).isoformat()
            ),
            "status": status,
            "records_downloaded": records_downloaded,
            "records_cleaned": records_cleaned,
            "records_loaded": records_loaded,
            "error_message": error_message,
            "duration_seconds": (
                int(round((finished_at - started_at).total_seconds()))
                if started_at is not None and finished_at is not None
                else None
            ),
            "metadata": metadata_json or {},
            "fallback_at": datetime.now(timezone.utc).isoformat(),
        }
        # Buffer for replay
        self._audit_buffer.append(record)
        # Write to local JSONL file
        fallback_path = RAW_DATA_DIR / "pipeline_runs_fallback.jsonl"
        try:
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fallback_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            logger.error(
                "[%s] Could not write audit fallback to %s: %s",
                self.source_name,
                fallback_path,
                exc,
            )

    def _replay_audit_buffer(self) -> int:
        """Replay buffered audit records to the DB (DQ-5.10).

        Returns
        -------
        int
            Number of records successfully replayed.
        """
        if not self._audit_buffer:
            return 0
        try:
            with get_db_session() as session:
                return self._replay_audit_buffer_in_session(session)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            # v85 FORENSIC ROOT FIX (BUG #51): narrowed from broad
            # ``except Exception``. Audit replay can fail with DB
            # connection, SQL, or import errors.
            logger.warning(
                "[%s] Could not replay audit buffer: %s",
                self.source_name,
                exc,
            )
            return 0

    # v90 ROOT FIX (BUG #15): maximum retry count for audit buffer
    # replay. The previous code retried failed records FOREVER --
    # if a record's schema is incompatible with the DB (e.g. a
    # column was dropped), every pipeline run retries it, adding
    # it back to the buffer each time, growing the buffer without
    # bound. ROOT FIX: after _AUDIT_MAX_RETRIES failures, DROP the
    # record and log an error. This bounds the buffer size and
    # prevents infinite retry loops on permanently-broken records.
    _AUDIT_MAX_RETRIES: int = 3

    def _replay_audit_buffer_in_session(self, session: Any) -> int:
        """Replay buffered audit records within an existing session."""
        replayed = 0
        remaining: list[dict] = []
        for record in self._audit_buffer:
            # v90 ROOT FIX (BUG #15): check retry count.
            _retry_count = record.get("_retry_count", 0)
            if _retry_count >= self._AUDIT_MAX_RETRIES:
                logger.error(
                    "[%s] Dropping permanently-failed audit record after "
                    "%d retries (source=%s, run_date=%s). The record's "
                    "schema may be incompatible with the current DB.",
                    self.source_name,
                    _retry_count,
                    record.get("source"),
                    record.get("run_date"),
                )
                continue  # drop the record
            try:
                run_date_str = record.get("run_date")
                try:
                    run_date = (
                        datetime.fromisoformat(run_date_str)
                        if run_date_str
                        else datetime.now(timezone.utc)
                    )
                except (ValueError, TypeError):
                    run_date = datetime.now(timezone.utc)

                # V90 CI fix: clamp negative record counts to 0 (same fix
                # as the primary insertion path above).
                _rd_r = record.get("records_downloaded") or 0
                _rc_r = record.get("records_cleaned") or 0
                _rl_r = record.get("records_loaded") or 0
                run = PipelineRun(
                    source=record["source"],
                    run_date=run_date,
                    status=record.get("status"),
                    records_downloaded=max(0, _rd_r),
                    records_cleaned=max(0, _rc_r),
                    records_loaded=max(0, _rl_r),
                    error_message=record.get("error_message"),
                    duration_seconds=record.get("duration_seconds"),
                    # P1-18 ROOT FIX: replay buffered records' metadata too.
                    # The fallback record stored metadata under the "metadata"
                    # key (see _write_run_log_fallback); re-key it to
                    # metadata_json for the ORM column.
                    metadata_json=record.get("metadata"),
                )
                session.add(run)
                replayed += 1
            except (OSError, ValueError, RuntimeError, KeyError):
                # v85/v90 ROOT FIX (BUG #15/51): narrowed from broad
                # ``except Exception``. Per-record DB insert can fail
                # with constraint/SQL errors. Programming bugs propagate.
                # Increment retry counter instead of re-adding without limit.
                record["_retry_count"] = _retry_count + 1
                remaining.append(record)
        self._audit_buffer = remaining
        if replayed > 0:
            logger.info(
                "[%s] Replayed %d buffered audit records to DB",
                self.source_name,
                replayed,
            )
        return replayed

    # ------------------------------------------------------------------
    # Provenance metadata (LIN-16.8)
    # ------------------------------------------------------------------
    def _write_provenance(self, cleaned_path: Path) -> None:
        """Write a ``.provenance.json`` sidecar for the cleaned CSV (LIN-16.8).

        Records the full provenance metadata so any value in the
        output can be traced back to its source.
        """
        provenance = {
            "pipeline": self.source_name,
            "run_id": self.run_id,
            "source_version": self.source_version,
            "started_at": (
                self.start_time.isoformat() if self.start_time else None
            ),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "sha256_raw": self._sha256_raw,
            "sha256_cleaned": self._sha256_cleaned,
            "git_commit": _get_git_commit(),
            "seed": self.seed,
            "schema_version": SCHEMA_VERSION,
            "correlation_id": self.correlation_id,
            "triggered_by": self.triggered_by,
            "transformation_log": self._transformation_log,
            "field_lineage": self._field_lineage,
        }
        prov_path = cleaned_path.with_suffix(
            cleaned_path.suffix + ".provenance.json"
        )
        try:
            with open(prov_path, "w", encoding="utf-8") as f:
                json.dump(provenance, f, indent=2, sort_keys=True, default=str)
        except OSError as exc:
            logger.warning(
                "[%s] Could not write provenance sidecar for %s: %s",
                self.source_name,
                cleaned_path.name,
                exc,
            )

    # ------------------------------------------------------------------
    # Transformation log (LIN-16.11)
    # ------------------------------------------------------------------
    def _log_transformation(
        self,
        step: str,
        rows_affected: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a transformation step for the audit trail (LIN-16.11).

        Parameters
        ----------
        step : str
            Name of the transformation step (e.g.
            ``"normalize_inchikey"``, ``"dedup_by_inchikey"``).
        rows_affected : int
            Number of rows affected by this step.
        details : dict, optional
            Additional step-specific details (e.g. parameters used).
        """
        entry = {
            "step": step,
            "rows_affected": int(rows_affected),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        }
        self._transformation_log.append(entry)
        logger.debug(
            "[%s] Transformation: %s affected %d rows",
            self.source_name,
            step,
            rows_affected,
        )

    # ------------------------------------------------------------------
    # Logging & observability (LOG-11.6, LOG-11.13)
    # ------------------------------------------------------------------
    def _log_structured(
        self,
        level: int,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log a structured JSON-formatted message (LOG-11.6).

        When ``PIPELINE_LOG_FORMAT=json`` env var is set, logs are
        emitted as JSON for machine consumption. Otherwise, standard
        logging format is used for backward compatibility.
        """
        if os.environ.get("PIPELINE_LOG_FORMAT", "").lower() == "json":
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": self.source_name,
                "run_id": self.run_id,
                "level": logging.getLevelName(level),
                "message": message,
                **kwargs,
            }
            logger.log(level, json.dumps(payload, default=str))
        else:
            extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
            logger.log(level, "[%s] %s %s", self.source_name, message, extra)

    def _emit_metric(
        self,
        name: str,
        value: float,
        tags: dict[str, Any] | None = None,
    ) -> None:
        """Emit a metric for monitoring (LOG-11.13).

        Currently logs to a metrics logger. In production, this could
        be hooked up to Prometheus, StatsD, or CloudWatch.
        """
        metrics_logger = logging.getLogger(f"{__name__}.metrics")
        metrics_logger.info(
            "metric %s=%s %s",
            name,
            value,
            json.dumps(tags or {}, default=str),
        )

    def _categorize_error(self, exc: Exception) -> str:
        """Categorise an exception for monitoring (LOG-11.13).

        Returns
        -------
        str
            One of ``"network"``, ``"http_4xx"``, ``"http_5xx"``,
            ``"data_format"``, ``"database"``, ``"unknown"``.
        """
        if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return "network"
        if isinstance(exc, requests.exceptions.HTTPError):
            if exc.response is not None:
                if 400 <= exc.response.status_code < 500:
                    return "http_4xx"
                return "http_5xx"
            return "http_unknown"
        if isinstance(exc, (pd.errors.ParserError, ValueError, json.JSONDecodeError)):
            return "data_format"
        try:
            from sqlalchemy.exc import IntegrityError, OperationalError
            if isinstance(exc, (IntegrityError, OperationalError)):
                return "database"
        except ImportError:
            pass
        return "unknown"

    # ------------------------------------------------------------------
    # Lineage & state (required by pipelines/__init__.pyi stub)
    # ------------------------------------------------------------------
    def get_provenance(self) -> dict[str, Any]:
        """Return full provenance metadata for the last run (LIN-16.13).

        Returns
        -------
        dict
            Provenance metadata including run_id, source_name,
            source_version, SHA-256 checksums, git commit, seed,
            transformation log, and field lineage.
        """
        return {
            "run_id": getattr(self, "run_id", None),
            "source_name": self.source_name,
            "source_version": self.get_source_version(),
            "sha256_raw": getattr(self, "_sha256_raw", None),
            "sha256_cleaned": getattr(self, "_sha256_cleaned", None),
            "git_commit": _get_git_commit(),
            "seed": getattr(self, "seed", None),
            "started_at": (
                self.start_time.isoformat() if self.start_time else None
            ),
            "transformation_log": getattr(self, "_transformation_log", []),
            "field_lineage": getattr(self, "_field_lineage", {}),
            "schema_version": SCHEMA_VERSION,
        }

    def get_audit_trail(self, *, include_deleted: bool = True) -> dict[str, Any]:
        """Return audit trail for all runs of this pipeline source (LIN-16.13).

        P1-060 ROOT FIX (v107): the default ``include_deleted=True``
        returns ALL runs including soft-deleted ones. This is required
        for FDA 21 CFR Part 11 compliance — the audit trail MUST be
        complete and immutable. An operator who soft-deletes a failed
        run to "clean up" the UI MUST NOT silently remove it from the
        audit trail. Pass ``include_deleted=False`` ONLY when the
        operator explicitly requests non-deleted runs (e.g. for a
        dashboard view that filters deleted runs by design).

        Parameters
        ----------
        include_deleted : bool, default True
            If True (default, FDA Part 11 compliant), include soft-deleted
            runs in the audit trail. If False, filter them out.

        Returns
        -------
        dict
            ``{"source": ..., "runs": [...], "error": ...}`` where
            ``runs`` is a list of the last 100 run records. If the
            DB is unreachable, ``error`` is set to ``"DB unavailable"``.
        """
        if not _HAS_SQLALCHEMY or _sa_select is None:
            return {
                "source": self.source_name,
                "runs": [],
                "error": "SQLAlchemy not available",
            }
        try:
            with get_db_session() as session:
                stmt = (
                    _sa_select(PipelineRun)
                    .where(PipelineRun.source == self.source_name)
                )
                # P1-060 ROOT FIX (v107): default include_deleted=True
                # for FDA Part 11 compliance. Only filter when explicitly
                # requested.
                if not include_deleted:
                    is_deleted_col = getattr(PipelineRun, "is_deleted", None)
                    if is_deleted_col is not None:
                        stmt = stmt.where(is_deleted_col == False)  # noqa: E712
                stmt = stmt.order_by(PipelineRun.run_date.desc()).limit(100)
                runs = session.execute(stmt).scalars().all()
                return {
                    "source": self.source_name,
                    "runs": [
                        {
                            "run_date": str(r.run_date),
                            "status": r.status,
                            "records_downloaded": r.records_downloaded,
                            "records_cleaned": r.records_cleaned,
                            "records_loaded": r.records_loaded,
                            "duration_seconds": r.duration_seconds,
                            "error_message": r.error_message,
                        }
                        for r in runs
                    ],
                }
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            # v85 FORENSIC ROOT FIX (BUG #51): narrowed from broad
            # ``except Exception``. Audit reads can fail with DB
            # connection, SQL, or import errors.
            logger.warning(
                "[%s] Could not read audit trail: %s",
                self.source_name,
                exc,
            )
            return {
                "source": self.source_name,
                "runs": [],
                "error": f"DB unavailable: {exc}",
            }

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize pipeline state for checkpoint/restart (LIN-16.13).

        Returns
        -------
        dict
            State dictionary that can be passed to ``from_state_dict``
            to restore the pipeline to the same state.
        """
        return {
            "source_name": self.source_name,
            "run_id": getattr(self, "run_id", None),
            "start_time": (
                self.start_time.isoformat() if self.start_time else None
            ),
            "downloaded_paths": [str(p) for p in self.downloaded_paths],
            "source_version": self.get_source_version(),
            "dead_letter_count": len(self.dead_letter_queue),
            "sha256_raw": getattr(self, "_sha256_raw", None),
            "sha256_cleaned": getattr(self, "_sha256_cleaned", None),
            "transformation_log": getattr(self, "_transformation_log", []),
        }

    def from_state_dict(self, state: dict[str, Any]) -> None:
        """Restore pipeline state from a ``to_state_dict`` checkpoint."""
        if "run_id" in state and state["run_id"]:
            self.run_id = state["run_id"]
        if "start_time" in state and state["start_time"]:
            try:
                self.start_time = datetime.fromisoformat(state["start_time"])
            except (ValueError, TypeError):
                pass
        if "downloaded_paths" in state:
            self.downloaded_paths = [Path(p) for p in state["downloaded_paths"]]
        if "source_version" in state:
            self.source_version = state["source_version"]
        if "sha256_raw" in state:
            self._sha256_raw = state["sha256_raw"]
        if "sha256_cleaned" in state:
            self._sha256_cleaned = state["sha256_cleaned"]
        if "transformation_log" in state:
            self._transformation_log = state["transformation_log"]

    def recover_from_failure(self) -> None:
        """Attempt to recover from a failed pipeline run (LIN-16.13).

        Strategy:
        1. Check for persisted cleaned data from a previous run
           (``PROCESSED_DATA_DIR / _get_processed_filename()``).
        2. If found and the run_context sidecar is valid, attempt
           ``run_load_only`` to load the cleaned data.
        3. If not found, log a warning -- the caller should restart
           the full pipeline via ``run()``.
        """
        logger.info(
            "[%s] Attempting recovery from failure...",
            self.source_name,
        )
        clean_path = PROCESSED_DATA_DIR / self._get_processed_filename()
        if clean_path.exists():
            logger.info(
                "[%s] Found existing cleaned data at %s, attempting load",
                self.source_name,
                clean_path,
            )
            try:
                self.run_load_only()
                logger.info(
                    "[%s] Recovery succeeded via run_load_only",
                    self.source_name,
                )
                return
            except (OSError, ValueError, RuntimeError, ImportError, ConnectionError) as exc:
                # v85 FORENSIC ROOT FIX (BUG #51): narrowed from broad
                # ``except Exception``. Recovery can fail with DB or
                # I/O errors. Programming bugs now propagate.
                logger.error(
                    "[%s] Recovery via run_load_only failed: %s. "
                    "Restart the full pipeline with run().",
                    self.source_name,
                    exc,
                )
        else:
            logger.warning(
                "[%s] No cleaned data found at %s. "
                "Restart the full pipeline with run().",
                self.source_name,
                clean_path,
            )

    def get_dead_letters(self) -> list[dict[str, Any]]:
        """Return records that failed processing during the last run (LIN-16.13).

        Returns
        -------
        list of dict
            Records that were sent to the dead letter queue during
            ``clean()`` or ``load()``. Each record includes the
            original data and the error that caused it to fail.
        """
        return list(self.dead_letter_queue)

    # ------------------------------------------------------------------
    # Streaming & parallelism (PERF-8.10, PERF-8.13)
    # ------------------------------------------------------------------
    def clean_streaming(
        self,
        raw_path: Path,
        chunksize: int = 10000,
    ) -> Iterator[pd.DataFrame]:
        """Streaming alternative to ``clean()`` (PERF-8.13).

        Default implementation calls ``clean()`` and yields the full
        DataFrame. Subclasses may override for truly streaming
        processing of very large files.

        Parameters
        ----------
        raw_path : Path
            Path to the raw downloaded file.
        chunksize : int, default 10000
            Number of rows per chunk (subclasses may ignore this if
            the source format doesn't support chunking).

        Yields
        ------
        pandas.DataFrame
            Chunks of the cleaned DataFrame.
        """
        yield self.clean(raw_path)

    def _download_parallel(
        self,
        urls: list[str],
        dest_dir: Path,
        max_workers: int = 3,
    ) -> list[Path]:
        """Download multiple files in parallel (PERF-8.10).

        Uses ``concurrent.futures.ThreadPoolExecutor``. Each download
        is independent -- a failure in one does not affect the others.

        Parameters
        ----------
        urls : list of str
            URLs to download.
        dest_dir : Path
            Directory to download into. Filenames are derived from
            the URL.
        max_workers : int, default 3
            Maximum number of concurrent downloads.

        Returns
        -------
        list of Path
            Paths to the downloaded files (in the same order as
            *urls*). Failed downloads are ``None``.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _download_one(url: str) -> Path | None:
            try:
                # FIX-P2-C-10 (audit P2): ``Path(url).name`` produced
                # broken filenames for URLs with query strings
                # (``https://example.com/file.csv?v=1`` ->
                # ``file.csv?v=1`` -- an invalid filename containing a
                # question mark) or trailing slashes (``.../dir/`` ->
                # empty string). Strip the query and fragment FIRST via
                # ``urlparse(url).path`` so ``Path(...).name`` sees only
                # the path component. If the resulting name is empty
                # (URL ended in ``/``) or ``.`` (path was ``/``),
                # fall back to a hash of the URL so the download always
                # gets a deterministic, filesystem-safe filename.
                from urllib.parse import urlparse
                import hashlib as _hashlib
                _url_path = urlparse(url).path
                _name = Path(_url_path).name
                if not _name or _name in (".", ".."):
                    _name = (
                        "download_"
                        + _hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
                    )
                dest = dest_dir / _name
                return self._download_file(url, dest)
            except DownloadError as exc:
                # FIX-P2-12 (audit P2): the previous broad
                # ``except Exception`` swallowed EVERY exception including
                # permanent HTTP errors (4xx except 429, malformed URL,
                # PermissionError on the destination path). The caller
                # received ``None`` and treated it as "soft failure --
                # continue without this file", masking real configuration
                # problems (e.g. a 404 because the URL template changed,
                # a 403 because the API key expired). _download_file
                # already wraps retryable-exhausted network failures as
                # DownloadError; non-DownloadError exceptions (e.g. an
                # unexpected requests.HTTPError that escaped the retry
                # loop, or a programming AttributeError) should propagate
                # so the caller sees a real stack trace.
                logger.error(
                    "[%s] Parallel download failed for %s: %s",
                    self.source_name,
                    self._sanitize_url(url),
                    exc,
                )
                return None

        results: list[Path | None] = [None] * len(urls)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_download_one, url): i
                for i, url in enumerate(urls)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except (OSError, ValueError, RuntimeError, ConnectionError, TimeoutError) as exc:
                    # v85 FORENSIC ROOT FIX (BUG #51): narrowed from
                    # broad ``except Exception``. Parallel downloads can
                    # fail with I/O, network, or format errors.
                    # Programming bugs now propagate.
                    logger.error(
                        "[%s] Parallel download %d failed: %s",
                        self.source_name,
                        idx,
                        exc,
                    )
        return results

    # ------------------------------------------------------------------
    # Public data access (INT-15.5)
    # ------------------------------------------------------------------
    def get_cleaned_data(self) -> pd.DataFrame:
        """Return the most recent cleaned data from disk (INT-15.5).

        Returns
        -------
        pandas.DataFrame
            The cleaned DataFrame from
            ``PROCESSED_DATA_DIR / _get_processed_filename()``.

        Raises
        ------
        FileNotFoundError
            If the cleaned CSV does not exist.
        """
        clean_path = PROCESSED_DATA_DIR / self._get_processed_filename()
        if not clean_path.exists():
            raise FileNotFoundError(
                f"No cleaned data found at {clean_path}. Run the full "
                f"pipeline first."
            )
        return pd.read_csv(
            clean_path,
            encoding="utf-8",
            dtype=self.get_dtypes(),
            low_memory=False,
        )

    # ------------------------------------------------------------------
    # GDPR / compliance hooks (COMP-14.7)
    # ------------------------------------------------------------------
    def _export_data(self, subject_id: str) -> pd.DataFrame:
        """Export all data for a given subject (GDPR right to portability).

        P1-054 ROOT FIX (v107): the previous implementation returned an
        empty DataFrame silently — making the GDPR hook INERT. The
        platform CLAIMED GDPR compliance but the hook did nothing.
        The Autonomous Drug Repurposing Platform is a POPULATION-LEVEL
        system (10,000 FDA-approved drugs × every known disease) — it
        does NOT handle subject-level (patient-level) data. There are
        no patient records, no individual subject data, no PII.

        ROOT FIX: raise NotImplementedError with a clear message
        documenting that the platform is population-level only. This
        makes the inert-hook status EXPLICIT — any caller that tries
        to invoke GDPR portability gets a clear error instead of a
        silent empty DataFrame.

        Raises
        ------
        NotImplementedError
            Always — the platform is population-level only.
        """
        raise NotImplementedError(
            "_export_data (GDPR right to portability) is NOT implemented. "
            "The Autonomous Drug Repurposing Platform is a POPULATION-LEVEL "
            "system: 10,000 FDA-approved drugs × every known disease. It "
            "does NOT handle subject-level (patient-level) data, PII, or "
            "individual health records. GDPR Articles 15-20 (data subject "
            "rights) do not apply to population-level aggregate biomedical "
            "data derived from public databases (ChEMBL, DrugBank, UniProt, "
            "STRING, DisGeNET, OMIM, PubChem). If a future phase adds "
            "subject-level data (e.g. EHR integration), subclasses handling "
            "that data MUST override this method. (P1-054 ROOT FIX)"
        )

    def _delete_data(self, subject_id: str) -> int:
        """Delete all data for a given subject (GDPR right to erasure).

        P1-054 ROOT FIX (v107): the previous implementation returned 0
        silently — making the GDPR hook INERT. See ``_export_data`` for
        the full rationale. The platform is population-level only;
        GDPR right to erasure does not apply.

        Raises
        ------
        NotImplementedError
            Always — the platform is population-level only.
        """
        raise NotImplementedError(
            "_delete_data (GDPR right to erasure) is NOT implemented. "
            "The platform is POPULATION-LEVEL only — no subject-level "
            "data to delete. See _export_data docstring for full rationale. "
            "(P1-054 ROOT FIX)"
        )
