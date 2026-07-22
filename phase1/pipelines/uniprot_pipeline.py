"""UniProt Pipeline -- institutional-grade production-ready ETL for proteins.

This module implements ``UniProtPipeline``, the data ingestion pipeline that
downloads human-reviewed (Swiss-Prot) protein records from the UniProt REST
API, cleans and normalizes them, and bulk-upserts them into the ``proteins``
table of the staging database.

It is part of the Autonomous Drug Repurposing Platform (Team Cosmic /
VentureLab) and feeds the Knowledge Graph (Phase 2), the Graph Transformer
(Phase 3), and the RL Hypothesis Ranker (Phase 4).  Because downstream
consumers make clinical decisions based on this data, scientific correctness
is life-safety critical.

------------------------------------------------------------------------
Why this file exists (inception)
------------------------------------------------------------------------
The previous version of ``uniprot_pipeline.py`` (384 lines) had 346 issues
spanning 16 quality domains.  Five of them were FATAL -- they silently
destroyed or corrupted data:

* F1 -- ``load()`` did not accept ``session=``, raising ``TypeError`` on every
  ``run()`` call (no protein data ever loaded).
* F2 -- Sequences were truncated to 10 000 chars, silently destroying titin
  (~34 350 aa) and MUC16 (~14 507 aa).
* F3 -- The TSV header was not skipped on subsequent pages, creating phantom
  ``"Entry"`` rows in the cleaned dataset.
* F4 -- ``gene_name`` stored a protein name, not a gene symbol -- every
  downstream GDA join silently failed.
* F5 -- Downloads were non-atomic; a crash mid-download left a partial file
  that was silently reused forever.

Every fix in this file is traceable to one of the 346 issue IDs documented
in ``UNIPROT_PIPELINE_346_ISSUES_FIX_PROMPT.md``.

------------------------------------------------------------------------
Data flow
------------------------------------------------------------------------
::

    UniProt REST API  ->  raw TSV (atomic write)
                      ->  cleaned DataFrame (full sequence, validated)
                      ->  proteins.csv  (schema-v1 compliant)
                      ->  bulk_upsert_proteins(session, df)
                      ->  proteins table  ->  Neo4j graph
                                        ->  Graph Transformer
                                        ->  RL ranker
                                        ->  pharma partner -> patient

------------------------------------------------------------------------
Usage examples
------------------------------------------------------------------------
::

    # Full pipeline (download + clean + load)
    from pipelines.uniprot_pipeline import UniProtPipeline
    UniProtPipeline().run()

    # Download + clean only (used by the master DAG so entity resolution
    # can run between clean and load).
    UniProtPipeline().run_download_and_clean_only()

    # Load only (after entity resolution).
    UniProtPipeline().run_load_only()

    # Dependency-injected (for tests).
    from unittest.mock import MagicMock
    UniProtPipeline(
        http_client=MagicMock(),
        db_session_factory=MagicMock(),
        loader=MagicMock(),
    )

------------------------------------------------------------------------
Changelog
------------------------------------------------------------------------
v2.0.0 (2025-03-05) -- Institutional-grade rewrite addressing 346 issues
    across 16 domains.  See ``UNIPROT_PIPELINE_346_ISSUES_FIX_PROMPT.md``.

v1.0.0 -- Initial implementation (384 lines, deprecated).

------------------------------------------------------------------------
License
------------------------------------------------------------------------
MIT -- Team Cosmic / VentureLab.  See the project LICENSE file for details.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import pandas as pd
import requests
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError, IntegrityError  # v85 FORENSIC ROOT FIX (BUG #51)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from cleaning._constants import (
    normalize_uniprot_id,  # v29 ROOT FIX (audit P1-24)
    normalize_gene_symbol,  # v29 ROOT FIX (audit P1-24)
)
from cleaning.missing_values import handle_missing_protein_fields
from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR, UNIPROT_RELEASE
from database.connection import get_db_session
from database.loaders import UpsertResult, bulk_upsert_proteins
from pipelines.base_pipeline import BasePipeline, DownloadError, LoadResult


# ---------------------------------------------------------------------------
# v83 COMP-6 ROOT FIX: stale-cursor detection helper.
# ---------------------------------------------------------------------------
def _is_stale_cursor_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is a UniProt 4xx error indicating a stale cursor.

    UniProt cursor URLs (the ``Link`` header's ``rel="next"`` URL) expire
    after ~15 minutes. A resumed download that reuses an expired cursor
    receives HTTP 400 (Bad Request -- "invalid cursor") or 404 (Not Found
    -- "cursor not found"). These are non-retryable 4xx per R13, so the
    pipeline would be stuck without stale-cursor recovery.

    This helper inspects the exception (and any chained cause) for an
    HTTP status code in {400, 404} and returns True if found. It is
    intentionally NARROW -- only 400/404 trigger recovery. Other 4xx
    codes (401, 403, 429) indicate different problems (auth, quota,
    rate-limit) that stale-cursor recovery would not fix.
    """
    # Walk the exception chain (exc -> __cause__ -> __context__) to find
    # an HTTP status code. The DownloadError raised by _fetch_page wraps
    # the original HTTPError via `from exc`, so __cause__ has the
    # response object.
    seen: set[int] = set()  # guard against cycles
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        # Direct status_code attribute (requests.HTTPError).
        for attr in ("status_code", "status", "code"):
            val = getattr(current, attr, None)
            if isinstance(val, int) and val in (400, 404):
                return True
        # Nested response object.
        response = getattr(current, "response", None)
        if response is not None:
            for attr in ("status_code", "status", "code"):
                val = getattr(response, attr, None)
                if isinstance(val, int) and val in (400, 404):
                    return True
        # String heuristic -- DownloadError messages include "HTTP NNN".
        msg = str(current)
        if "HTTP 400" in msg or "HTTP 404" in msg:
            return True
        # Walk the chain.
        current = current.__cause__ or current.__context__
    return False


# ---------------------------------------------------------------------------
# Module metadata (DOC16-DOC20)
# ---------------------------------------------------------------------------
__all__ = ["UniProtPipeline"]
__version__ = "2.0.0"
__author__ = "Team Cosmic / VentureLab"
__license__ = "MIT"

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases (DOC14)
# ---------------------------------------------------------------------------
UniProtId = str          # e.g. "P69905" -- 6- or 10-char Swiss-Prot accession
GeneSymbol = str         # e.g. "HBA1" -- HGNC-canonical uppercase gene symbol
AminoAcidSequence = str  # e.g. "MVLSPADKTN..." -- IUPAC one-letter codes

# ---------------------------------------------------------------------------
# Compiled regex patterns (S3, S8, S9, S20, S21)
# ---------------------------------------------------------------------------
# UniProt accession pattern (matches schema/v1.json).  Two alternative forms:
#   * 6-char:   [OPQ][0-9][A-Z0-9]{3}[0-9]               e.g. P69905, Q8WXI7
#   * 10-char:  [A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2} e.g. A0A024RBG1
_UNIPROT_ACCESSION_RE: re.Pattern[str] = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$"
    r"|"
    r"^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)

# HGNC gene symbol: uppercase letter, then 0-49 alphanumeric/hyphen chars.
# v35 ROOT FIX: import from cleaning._constants (single source of truth).
# v42 ROOT FIX (P1-A-8): this is a HUMAN pipeline (queries
# organism_id:9606 by default). The previous import used
# ``CANONICAL_NON_HUMAN_GENE_SYMBOL_REGEX`` which allows Title-Case
# symbols (e.g. ``Tp53`` for mouse) -- too permissive for human data.
# The HUMAN form ``CANONICAL_HGNC_GENE_SYMBOL_REGEX`` is uppercase-only
# and rejects non-human symbols, preventing cross-species contamination
# of the human protein table.
from cleaning._constants import CANONICAL_HGNC_GENE_SYMBOL_REGEX as _HGNC_SYMBOL_RE

# STRING cross-reference ID: <taxid>.ENSP<digits>, e.g. "9606.ENSP00000357607".
# v35 ROOT FIX: accept ANY taxonomy ID (not just 9606). The UniProt
# pipeline ingests both human and non-human proteins; hard-coding 9606
# silently dropped every non-human STRING cross-reference (e.g. mouse
# 10090.ENSP00000XXXXX). The cleaning-layer validator
# (``resolver_utils._STRING_ID_RE``) already accepts any taxid.
_STRING_ID_RE: re.Pattern[str] = re.compile(r"^\d+\.ENSP\d+$")

# Valid amino-acid characters: 20 standard + ambiguity codes B J O U X Z +
# stop * + alignment gap "-" (v35 root fix: gap char included for
# consistency with cleaning._constants.CANONICAL_AA_SEQUENCE_REGEX and
# database.models._SEQUENCE_RE -- without it, aligned sequences with gaps
# would pass the DB CHECK but fail this pipeline validator).
_VALID_AA_PATTERN: re.Pattern[str] = re.compile(
    r"^[ACDEFGHIKLMNPQRSTVWYBJOUXZ\*\-]+$"
)

# EC number suffix in protein names, e.g. "EC 1.11.1.6" -- strict format.
_EC_NUMBER_RE: re.Pattern[str] = re.compile(r"\s*EC\s+[\d]+(?:\.[\d]+){1,3}\s*$")

# {ECO:...} evidence tags -- UniProt uses these to cite literature sources.
_ECO_TAG_RE: re.Pattern[str] = re.compile(r"\s*\{ECO:[^}]*\}")

# Parenthetical content (handles nested parens via manual scan; see below).
_PAREN_OPEN_RE: re.Pattern[str] = re.compile(r"\(")

# ---------------------------------------------------------------------------
# UniProt CC sub-section markers (S5, S6, C16) -- when ANY of these appears,
# everything from that marker onward belongs to a different sub-section and
# must be truncated from the function description.
# ---------------------------------------------------------------------------
_SUBSECTION_MARKERS: tuple[str, ...] = (
    "ACTIVITY REGULATION:",
    "ALTERNATIVE PRODUCTS:",
    "BIOTECHNOLOGY:",
    "CAUTION:",
    "CATALYTIC ACTIVITY:",
    "COFACTOR:",
    "DEVELOPMENTAL STAGE:",
    "DISEASE:",
    "DISRUPTION PHENOTYPE:",
    "DOMAIN:",
    "ENZYME REGULATION:",
    "FUNCTION:",          # second occurrence after the leading "FUNCTION: "
    "INDUCTION:",
    "INTERACTION:",
    "MASS SPECTROMETRY:",
    "MISCELLANEOUS:",
    "PATHWAY:",
    "PHARMACEUTICAL:",
    "POLYMORPHISM:",
    "PTM:",
    "SEQUENCE SIMILARITY:",
    "SITES:",
    "SIMILARITY:",
    "SUBCELLULAR LOCATION:",
    "SUBUNIT:",
    "TISSUE SPECIFICITY:",
    "TOXIC DOSE:",
)

# ---------------------------------------------------------------------------
# DATA_DICTIONARY (DOC3) -- full per-column documentation embedded in code.
# ---------------------------------------------------------------------------
DATA_DICTIONARY: dict[str, dict[str, Any]] = {
    "uniprot_id": {
        "type": "str",
        "description": "UniProt Swiss-Prot accession (e.g. P69905). Primary key.",
        "pattern": r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$",
        "source": "UniProt 'Entry' column",
        "required": True,
    },
    "gene_symbol": {
        "type": "str | None",
        "description": "HGNC gene symbol (e.g. HBA1) or non-human Title-Case symbol "
                       "(e.g. Tp53 for mouse). Used for GDA resolution.",
        "pattern": r"^[A-Za-z][A-Za-z0-9\-]{0,49}$",
        "source": "UniProt 'Gene Names (primary)' column",
        "required": False,
    },
    "gene_name": {
        "type": "None",
        "description": "DEPRECATED -- always None.  Use protein_name_canonical "
                       "for canonical names and gene_symbol for gene symbols.",
        "source": "N/A",
        "required": False,
        "deprecated": True,
    },
    "protein_name": {
        "type": "str | None",
        "description": "Full protein name from UniProt (synonyms in parentheses).",
        "source": "UniProt 'Protein names' column",
        "required": False,
    },
    "protein_name_canonical": {
        "type": "str | None",
        "description": "Canonical protein name (parenthetical synonyms, ECO tags, "
                       "and EC numbers stripped).",
        "source": "Derived from protein_name via _extract_canonical_name().",
        "required": False,
    },
    "organism": {
        "type": "str",
        "description": "Organism name (always 'Homo sapiens' for this pipeline).",
        "source": "UniProt 'Organism' column",
        "required": True,
    },
    "length": {
        "type": "int | None",
        "description": "Protein sequence length in amino acids (UniProt-reported).",
        "source": "UniProt 'Length' column",
        "required": False,
        "valid_range": "1-100000",
    },
    "sequence": {
        "type": "str | None",
        "description": "Full amino-acid sequence. NOT truncated -- titin (~34 350 aa) is stored in full.",
        "source": "UniProt 'Sequence' column",
        "required": False,
        "valid_chars": "ACDEFGHIKLMNPQRSTVWYBJOUXZ*-",
    },
    "function_desc": {
        "type": "str | None",
        "description": "Function description (FUNCTION: prefix and sub-section markers stripped).",
        "source": "UniProt 'Function [CC]' column",
        "required": False,
    },
    "string_id": {
        "type": "str | None",
        "description": "First valid STRING ID (format: <taxid>.ENSP<digits>, e.g. 9606.ENSP00000357607).",
        "source": "UniProt 'Cross-reference (STRING)' column",
        "required": False,
        "pattern": r"^\d+\.ENSP\d+$",
    },
    "all_string_ids": {
        "type": "str | None",
        "description": "Semicolon-separated list of ALL valid STRING IDs for the protein.",
        "source": "UniProt 'Cross-reference (STRING)' column",
        "required": False,
    },
}

# ---------------------------------------------------------------------------
# EXPECTED_OUTPUT_COLUMNS (D2-12) -- the cleaned DataFrame MUST contain at
# least these columns.  Extra columns (lineage flags, _source, etc.) are
# tolerated.
# ---------------------------------------------------------------------------
EXPECTED_OUTPUT_COLUMNS: frozenset[str] = frozenset({
    "uniprot_id",
    "gene_symbol",
    "gene_name",
    "protein_name",
    "protein_name_canonical",
    "organism",
    "length",
    "sequence",
    "function_desc",
    "string_id",
    "all_string_ids",
})

# Expected raw TSV column names from UniProt REST API (used for schema-version guard).
_EXPECTED_TSV_COLUMNS: frozenset[str] = frozenset({
    "Entry",
    "Gene Names",
    "Gene Names (primary)",
    "Protein names",
    "Organism",
    "Length",
    "Sequence",
    "Cross-reference (STRING)",
    "Function [CC]",
})

# Columns critical to load() -- if missing after rename, raise immediately.
_CRITICAL_COLUMNS: tuple[str, ...] = ("uniprot_id",)

# CSV cells starting with these characters are vulnerable to formula injection
# when the CSV is opened in Excel / Sheets (SEC4 / C27).
_CSV_DANGEROUS_PREFIXES: tuple[str, ...] = ("=", "+", "-", "@", "\t", "\r")

# UniProt domains allowed for the REST API and Link-header URLs (SEC1 / SEC8).
_ALLOWED_DOMAINS: frozenset[str] = frozenset({
    "rest.uniprot.org",
    "www.uniprot.org",
    "uniprot.org",
})

# Maximum response body size we accept from UniProt (SEC6).  UniProt's
# page size cap is 500 records; a TSV page is well under 5 MB.  100 MB is
# an extremely generous ceiling that still prevents OOM from a malformed
# / malicious response.
_MAX_RESPONSE_BYTES: int = 100 * 1024 * 1024


# ===========================================================================
# UniProtPipeline
# ===========================================================================
class UniProtPipeline(BasePipeline):
    """Institutional-grade UniProt REST API pipeline for human reviewed proteins.

    Downloads Swiss-Prot human proteins from the UniProt REST API, cleans
    and normalizes the data, validates scientific correctness, and
    bulk-upserts into the ``proteins`` table.

    Class Attributes
    ----------------
    source_name : str
        Pipeline identifier (``"uniprot"``); validated by ``BasePipeline``.
    uniprot_search_url : str
        Base URL for the UniProt REST search endpoint.
    uniprot_query : str
        UniProt query string (default: human reviewed proteins).
    uniprot_fields : list[str]
        Fields to request from the UniProt API.  ``ft_domain`` is
        intentionally excluded -- domain extraction is not implemented and
        requesting the field would waste ~5% of API bandwidth (S13).
    page_size : int
        Number of records per page (1-500, UniProt hard cap).
    max_retries : int
        Maximum retry attempts per page fetch.
    base_retry_delay : float
        Base delay in seconds for exponential backoff.
    max_retry_after_wait : int
        Maximum seconds to wait for a Retry-After header (SEC7).
    consecutive_retry_after_limit : int
        Max consecutive Retry-After responses before breaking the retry loop.

    Notes
    -----
    * All sequences are stored in full -- titin (~34 350 aa) is preserved
      (F2).  Truncation was a FATAL silent data-corruption bug.
    * ``gene_name`` is deprecated and set to ``None`` (F4).  Use
      ``gene_symbol`` for gene symbols and ``protein_name_canonical`` for
      canonical protein names.
    * The pipeline is idempotent: deterministic sort before dedup, atomic
      file write, SHA-256 checksum sidecar, content-hash logging for
      duplicate detection.
    * The pipeline is reproducible: ``self.seed`` is honored, the UniProt
      release is recorded in the provenance sidecar, and the same input
      always produces the same output.

    Examples
    --------
    >>> pipeline = UniProtPipeline()
    >>> pipeline.run()  # doctest: +SKIP
    """

    # ---------------------------------------------------------------------
    # Class attributes (A10, A12, A13, CFG1-CFG4)
    # ---------------------------------------------------------------------
    source_name: str = "uniprot"

    # UniProt REST API endpoint (CFG2).
    uniprot_search_url: str = "https://rest.uniprot.org/uniprotkb/search"

    # Query for human (taxonomy 9606) reviewed (Swiss-Prot) proteins (CFG3).
    # S23: isoform support is not yet implemented -- this query returns
    # canonical entries only.  Adding "AND (isoform:true)" would require
    # a separate download pass and is tracked as a TODO.
    # S24: natural variants (ft_variant) are not requested; variant
    # annotations would be critical for drug repurposing but require
    # extraction logic that is not yet implemented.
    # v41 ROOT FIX (P1 #29): organism_id is now env-overridable via
    # DRUGOS_UNIPROT_ORGANISM_ID (default 9606 = Homo sapiens). This
    # allows non-human protein studies without subclassing.
    # v42 ROOT FIX (P1-A-11): this used to be a class attribute
    # evaluated at class-definition time, so env-var changes after
    # import were ignored. The query is now constructed in __init__
    # (instance attribute) and the organism ID is validated:
    # ``org = os.environ.get(...) or '9606'`` (empty-string env var
    # v65 ROOT FIX (P1-038): the previous class attribute
    # ``uniprot_query: str = f"organism_id:{os.environ.get(...)} ..."`` was
    # evaluated ONCE at class definition time (module import). If
    # ``DRUGOS_UNIPROT_ORGANISM_ID`` was set AFTER import (e.g. in an
    # Airflow DAG that sets env vars before task execution), the class
    # attribute still had the OLD value. The ``__init__`` method
    # re-constructs the query, but only if ``UNIPROT_QUERY`` env var is
    # NOT set -- so the stale class attribute would be permanently used
    # whenever ``UNIPROT_QUERY`` was set.
    #
    # Root fix: replace the class attribute with a ``@property`` that
    # reads the env var at ACCESS time. This way:
    #   - ``pipeline.uniprot_query`` always reflects the CURRENT env var
    #     value (no stale state).
    #   - The ``UNIPROT_QUERY`` env var (full override) still takes
    #     precedence and is also read at access time.
    #   - The ``__init__`` method no longer needs to set
    #     ``self.uniprot_query`` -- the property handles it. We keep the
    #     ``__init__`` assignment as a private ``_uniprot_query_override``
    #     for explicit per-instance overrides (e.g. tests that want to
    #     pin a specific query without env var manipulation).
    _uniprot_query_override: Optional[str] = None

    @property
    def uniprot_query(self) -> str:
        """UniProt REST API search query (read at access time).

        Reads environment variables at access time so changes after
        module import (e.g. Airflow DAG setting env vars before task
        execution) are honoured. Precedence:
          1. ``self._uniprot_query_override`` (per-instance override,
             typically set by tests).
          2. ``UNIPROT_QUERY`` env var (full query override).
          3. Constructed from ``DRUGOS_UNIPROT_ORGANISM_ID`` env var
             (default 9606 = Homo sapiens) + ``AND reviewed:true``.
        """
        if self._uniprot_query_override is not None:
            return self._uniprot_query_override
        env_query = os.environ.get("UNIPROT_QUERY")
        if env_query:
            return env_query
        org = os.environ.get("DRUGOS_UNIPROT_ORGANISM_ID", "9606") or "9606"
        return f"organism_id:{org} AND reviewed:true"

    @uniprot_query.setter
    def uniprot_query(self, value: Optional[str]) -> None:
        """Allow ``self.uniprot_query = ...`` to set the per-instance override.

        Preserves backward compatibility with the previous class-attribute
        pattern where ``__init__`` did ``self.uniprot_query = ...``.
        Setting to ``None`` clears the override and reverts to env-var
        resolution.
        """
        self._uniprot_query_override = value

    # Fields requested from UniProt (CFG4, S13, S18).  ``ft_domain`` is
    # intentionally excluded until domain extraction is implemented.
    uniprot_fields: list[str] = [
        "accession",
        "gene_primary",
        "gene_names",
        "protein_name",
        "organism_name",
        "length",
        "sequence",
        "xref_string",   # S18: specific STRING xref field (not generic 'xref')
        "cc_function",
    ]

    # Pagination & retry tuning (CFG1, CFG7, C35).
    page_size: int = 500
    max_retries: int = 5
    base_retry_delay: float = 10.0           # seconds
    max_retry_after_wait: int = 300          # seconds (SEC7 / C43)
    consecutive_retry_after_limit: int = 3   # C8

    # BasePipeline attribute overrides (A12).
    min_request_interval: float = 0.5        # UniProt asks for self-throttling
    download_timeout: tuple[float, float] = (30.0, 600.0)
    download_max_retries: int = 5            # A13: coordinate with max_retries
    max_cache_age_days: int = 30             # CFG19
    verify_tls: bool = True                  # SEC2 / CFG15
    min_clean_ratio: float = 0.3
    min_load_ratio: float = 0.9
    stage_timeout: int = 3600                # CFG23 (R7)

    # File-permission mask for output files (SEC10 / SEC14).
    _SECURE_FILE_MODE: int = 0o600

    # ---------------------------------------------------------------------
    # __init__ (D2-5 dependency injection, CFG5/CFG8/CFG9 config)
    # ---------------------------------------------------------------------
    def __init__(
        self,
        *,
        http_client: Optional[requests.Session] = None,
        db_session_factory: Optional[Callable[..., Any]] = None,
        loader: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ``UniProtPipeline`` with optional dependency injection.

        Parameters
        ----------
        http_client : requests.Session | None
            Pre-configured HTTP client for API requests.  If *None*, a
            default ``requests.Session`` is created lazily on first use
            (A4 -- connection pooling).
        db_session_factory : callable | None
            Factory for DB sessions.  If *None*, ``get_db_session`` is used.
        loader : callable | None
            Bulk upsert function.  If *None*, ``bulk_upsert_proteins`` is
            used.  This is the dependency-injection seam used by tests.
        **kwargs
            Forwarded to ``BasePipeline.__init__`` (``run_id``,
            ``correlation_id``, ``triggered_by``, ``as_of_date``,
            ``freeze_version``, ``snapshot_tag``, ``seed``).

        Raises
        ------
        ValueError
            If any configuration value is invalid (CFG5).
        """
        super().__init__(**kwargs)

        # Dependency injection seams (D2-5).  Tests inject mocks here.
        self._http_session: Optional[requests.Session] = http_client
        self._db_session_factory: Callable[..., Any] = (
            db_session_factory or get_db_session
        )
        self._loader: Callable[..., Any] = loader or bulk_upsert_proteins

        # Per-run state (I3, I10 -- reset at the start of each download).
        self._consecutive_retry_after: int = 0
        self._total_retries: int = 0
        self._force_refresh: bool = False

        # v65 ROOT FIX (P1-038): the ``uniprot_query`` class attribute is
        # now a @property that reads env vars at access time. The previous
        # __init__ logic that set ``self.uniprot_query = ...`` based on
        # ``DRUGOS_UNIPROT_ORGANISM_ID`` and ``UNIPROT_QUERY`` is now
        # redundant -- the property handles both. We ONLY set
        # ``_uniprot_query_override`` if a test or caller explicitly
        # passes a query via constructor kwargs (handled below).
        #
        # The previous v42 ROOT FIX (P1-A-11) "construct at instance
        # creation time" was a partial fix -- it still cached the value at
        # __init__ time, so env-var changes AFTER __init__ (but BEFORE
        # the first API call) were ignored. The property approach is the
        # complete root fix: the query is fresh on every access.

        # Override class attributes from environment variables (CFG9).
        # Note: UNIPROT_QUERY env var is now read by the property at
        # access time, so we don't need to set it here.
        env = os.environ
        if url := env.get("UNIPROT_SEARCH_URL"):
            self.uniprot_search_url = url
        if ps := env.get("UNIPROT_PAGE_SIZE"):
            try:
                self.page_size = int(ps)
            except ValueError:
                logger.warning(
                    "[%s] Invalid UNIPROT_PAGE_SIZE=%r; using default %d",
                    self.source_name, ps, self.page_size,
                )
        if mr := env.get("UNIPROT_MAX_RETRIES"):
            try:
                self.max_retries = int(mr)
            except ValueError:
                logger.warning(
                    "[%s] Invalid UNIPROT_MAX_RETRIES=%r; using default %d",
                    self.source_name, mr, self.max_retries,
                )
        if brd := env.get("UNIPROT_BASE_RETRY_DELAY"):
            try:
                self.base_retry_delay = float(brd)
            except ValueError:
                logger.warning(
                    "[%s] Invalid UNIPROT_BASE_RETRY_DELAY=%r; using default %s",
                    self.source_name, brd, self.base_retry_delay,
                )

        # Pick up UNIPROT_RELEASE from settings.py (CFG8).
        try:
            if UNIPROT_RELEASE and UNIPROT_RELEASE != "current_release":
                self.source_version = UNIPROT_RELEASE
        except (ValueError, TypeError, AttributeError):  # pragma: no cover -- defensive  # v85 FORENSIC ROOT FIX (BUG #51)
            pass

        # Validate configuration (CFG5).
        self._validate_config()

    # ---------------------------------------------------------------------
    # Config validation (CFG5)
    # ---------------------------------------------------------------------
    def _validate_config(self) -> None:
        """Validate pipeline configuration.

        Raises
        ------
        ValueError
            If any configuration value is out of range or malformed.
        """
        if not 1 <= self.page_size <= 500:
            raise ValueError(
                f"page_size must be 1-500 (UniProt hard cap), got {self.page_size}"
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be >= 0, got {self.max_retries}"
            )
        if self.base_retry_delay <= 0:
            raise ValueError(
                f"base_retry_delay must be > 0, got {self.base_retry_delay}"
            )
        if self.max_retry_after_wait <= 0:
            raise ValueError(
                f"max_retry_after_wait must be > 0, got {self.max_retry_after_wait}"
            )
        if self.consecutive_retry_after_limit < 1:
            raise ValueError(
                f"consecutive_retry_after_limit must be >= 1, "
                f"got {self.consecutive_retry_after_limit}"
            )
        if not self.uniprot_search_url.startswith("https://"):
            raise ValueError(
                f"uniprot_search_url must use HTTPS, got {self.uniprot_search_url!r}"
            )
        if not self.uniprot_fields:
            raise ValueError("uniprot_fields must be a non-empty list")

    # ---------------------------------------------------------------------
    # effective_raw_dir property (A2)
    # ---------------------------------------------------------------------
    @property
    def effective_raw_dir(self) -> Path:
        """Return the effective raw data directory (A2).

        Falls back to ``RAW_DATA_DIR / self.source_name`` when
        ``self.raw_dir`` is *None* (e.g. when ``download()`` is called
        directly, before ``BasePipeline.run()`` has initialized it).
        """
        # BasePipeline may have set self.raw_dir already; respect it.
        existing = getattr(self, "raw_dir", None)
        if existing is not None:
            return Path(existing)
        return RAW_DATA_DIR / self.source_name

    # ---------------------------------------------------------------------
    # processed_dir property (D2-3)
    # ---------------------------------------------------------------------
    @property
    def processed_dir(self) -> Path:
        """Return the processed data directory (D2-3, C25)."""
        return PROCESSED_DATA_DIR

    # ---------------------------------------------------------------------
    # User-Agent (SEC3)
    # ---------------------------------------------------------------------
    @property
    def _user_agent(self) -> str:
        """User-Agent header value (SEC3)."""
        return (
            f"DrugRepurposingPlatform/{__version__} "
            f"(TeamCosmic; python-requests/{requests.__version__})"
        )

    # ---------------------------------------------------------------------
    # HTTP session (A4, R22, P13)
    # ---------------------------------------------------------------------
    def _get_http_session(self) -> requests.Session:
        """Get or create the HTTP session for connection reuse (A4).

        Returns
        -------
        requests.Session
            A session with ``Accept`` and ``User-Agent`` headers preset
            and ``verify`` set to ``self.verify_tls`` (SEC2).
        """
        if self._http_session is None:
            self._http_session = requests.Session()
            self._http_session.headers.update({
                "Accept": "text/tab-separated-values",
                "User-Agent": self._user_agent,
            })
            self._http_session.verify = self.verify_tls
        return self._http_session

    # ---------------------------------------------------------------------
    # URL validation (SEC1, SEC8)
    # ---------------------------------------------------------------------
    @classmethod
    def _validate_url(cls, url: str) -> str:
        """Validate that *url* points at an allowed UniProt domain (SEC1/SEC8).

        Prevents Server-Side Request Forgery (SSRF) via a malicious
        ``Link`` header by checking the scheme and hostname against an
        allow-list.

        Parameters
        ----------
        url : str
            URL to validate.

        Returns
        -------
        str
            The validated URL.

        Raises
        ------
        ValueError
            If the URL's scheme is not ``http``/``https`` or its hostname
            is not in ``_ALLOWED_DOMAINS``.
        """
        if not url or not isinstance(url, str):
            raise ValueError("URL must be a non-empty string")
        parsed = urlparse(url)
        if parsed.scheme not in ("https", "http"):
            raise ValueError(
                f"Invalid URL scheme: {parsed.scheme!r} "
                f"(expected 'https' or 'http')"
            )
        hostname = (parsed.hostname or "").lower()
        if hostname not in _ALLOWED_DOMAINS:
            raise ValueError(
                f"URL domain {hostname!r} not in allowed domains: "
                f"{sorted(_ALLOWED_DOMAINS)}. Possible SSRF attempt."
            )
        return url

    # ---------------------------------------------------------------------
    # Pre-flight check (A7)
    # ---------------------------------------------------------------------
    def pre_check(self) -> dict[str, bool]:
        """UniProt-specific pre-flight checks (A7).

        Verifies:
        1. UniProt API is reachable (HEAD request).
        2. Sufficient disk space for ~500 MB of raw + staged data.
        3. The raw data directory is writable.

        Returns
        -------
        dict[str, bool]
            Mapping of check name -> pass/fail.  The base ``run()``
            considers the pre-check failed if any value is *False*.
        """
        checks: dict[str, bool] = {}

        # Check API reachability (HEAD request, 10s timeout).
        try:
            resp = requests.head(
                self.uniprot_search_url,
                timeout=10,
                headers={"User-Agent": self._user_agent},
                verify=self.verify_tls,
            )
            # 5xx = server down; 4xx = our request is malformed.
            # Either way we cannot proceed safely.
            checks["api_reachable"] = resp.status_code < 500
            if not checks["api_reachable"]:
                logger.error(
                    "[%s] UniProt API returned HTTP %d in pre_check",
                    self.source_name, resp.status_code,
                )
        except requests.exceptions.RequestException as exc:
            logger.error(
                "[%s] UniProt API unreachable in pre_check: %s",
                self.source_name, exc,
                exc_info=getattr(self, "log_exc_info", True),
            )
            checks["api_reachable"] = False

        # Check disk space (need at least 500 MB).
        raw_dir = self.effective_raw_dir
        try:
            raw_dir.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(raw_dir)
            checks["disk_space"] = usage.free >= 500 * 1024 * 1024
            if not checks["disk_space"]:
                logger.error(
                    "[%s] Insufficient disk space: %.1f MB free (need >= 500 MB)",
                    self.source_name, usage.free / (1024 * 1024),
                )
        except OSError as exc:
            logger.error(
                "[%s] Cannot access raw_dir %s: %s",
                self.source_name, raw_dir, exc,
            )
            checks["disk_space"] = False

        # Check raw_dir is writable.
        try:
            test_file = raw_dir / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            checks["raw_dir_writable"] = True
        except OSError:
            checks["raw_dir_writable"] = False

        logger.info(
            "[%s] pre_check results: %s",
            self.source_name, checks,
            extra=self._log_context(),
        )
        return checks

    # ---------------------------------------------------------------------
    # download() -- atomic, paginated, with checksum + checkpoint (F3, F5)
    # ---------------------------------------------------------------------
    # v80 FORENSIC ROOT FIX (P0-C1 + P0-C8): the v50 downloader returns
    # files in three different formats (.jsonl / .dat.gz / .csv) but
    # ``clean()`` expects the raw UniProt REST TSV schema (tab-separated,
    # columns: Entry, Gene Names, Gene Names (primary), Protein names,
    # Organism, Length, Sequence, Cross-reference (STRING), Function [CC]).
    # The helper below normalizes all three formats to that TSV schema so
    # clean() works unchanged in every v50 mode. This also removes the
    # previous broken .jsonl->.csv branch (which produced dict-repr columns
    # that clean() could not parse -- P0-C8 dead code).
    def _normalize_v50_to_raw_tsv(self, prot_path: Path) -> Path:
        """Normalize any v50 downloader output to the raw UniProt TSV schema.

        Handles three input formats:
          1. ``.jsonl`` -- UniProt REST JSON (sample mode). Flattens the
             nested JSON into the 9 expected TSV columns.
          2. ``.dat.gz`` -- Swiss-Prot DAT format, gzipped (full / skip
             mode). Parses the line-oriented DAT records.
          3. ``.csv``   -- embedded-sample fallback (already-cleaned schema
             with columns like uniprot_id, gene_symbol, protein_name).
             Maps the embedded schema back to the raw TSV schema.

        The output is written to ``self.effective_raw_dir /
        uniprot_human_reviewed.tsv`` (overwriting any stale file) and
        returned. The original v50 file is preserved for audit.

        Raises
        ------
        DownloadError
            If the input format is unrecognized or parsing fails.
        """
        import csv as _csv
        import gzip as _gzip
        import json as _json

        out_path = self.effective_raw_dir / "uniprot_human_reviewed.tsv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # TSV header -- must match _EXPECTED_TSV_COLUMNS exactly so the
        # schema-version guard in clean() passes.
        TSV_HEADER = [
            "Entry",
            "Gene Names",
            "Gene Names (primary)",
            "Protein names",
            "Organism",
            "Length",
            "Sequence",
            "Cross-reference (STRING)",
            "Function [CC]",
        ]

        suffix = prot_path.suffix.lower()
        # Also detect .gz wrapping (e.g. .dat.gz).
        is_gz = suffix == ".gz"
        if is_gz:
            # Inner suffix: strip .gz, then look at what remains.
            inner = prot_path.with_suffix("").suffix.lower()
        else:
            inner = suffix

        logger.info(
            "[%s] Normalizing v50 output %s (format=%s) -> %s",
            self.source_name, prot_path.name,
            "dat.gz" if is_gz and inner == ".dat" else inner.lstrip("."),
            out_path.name,
        )

        rows_written = 0
        with open(out_path, "w", encoding="utf-8", newline="\n") as out_fh:
            writer = _csv.writer(out_fh, delimiter="\t", quoting=_csv.QUOTE_MINIMAL)
            writer.writerow(TSV_HEADER)

            # ─── Format 1: JSONL (sample mode REST JSON) ───────────────
            if suffix == ".jsonl":
                with open(prot_path, "r", encoding="utf-8") as in_fh:
                    for line in in_fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = _json.loads(line)
                        except _json.JSONDecodeError as exc:
                            logger.warning(
                                "[%s] Skipping malformed JSONL line: %s",
                                self.source_name, exc,
                            )
                            continue
                        row = self._flatten_uniprot_rest_json(rec)
                        writer.writerow(row)
                        rows_written += 1

            # ─── Format 2: .dat.gz (Swiss-Prot DAT, gzipped) ───────────
            elif is_gz and inner == ".dat":
                with _gzip.open(prot_path, "rt", encoding="utf-8") as in_fh:
                    for rec in self._iter_uniprot_dat_records(in_fh):
                        row = self._flatten_uniprot_dat_record(rec)
                        writer.writerow(row)
                        rows_written += 1

            # ─── Format 3: .csv (embedded-sample fallback) ─────────────
            elif suffix == ".csv":
                # Embedded samples use the already-cleaned schema:
                # uniprot_id, uniprot_ac, protein_name, gene_symbol,
                # gene_name, organism, protein_length, function.
                # Map back to the raw TSV schema.
                import pandas as _pd
                df = _pd.read_csv(prot_path)
                for _, df_row in df.iterrows():
                    entry = str(df_row.get("uniprot_id") or df_row.get("uniprot_ac") or "")
                    gene_primary = str(df_row.get("gene_symbol") or "")
                    gene_names = str(df_row.get("gene_name") or gene_primary)
                    protein_name = str(df_row.get("protein_name") or "")
                    organism = str(df_row.get("organism") or "Homo sapiens")
                    length = str(df_row.get("protein_length") or df_row.get("length") or "")
                    sequence = str(df_row.get("sequence") or "")
                    string_xref = str(df_row.get("string_id") or "")
                    function_desc = str(df_row.get("function") or df_row.get("function_desc") or "")
                    writer.writerow([
                        entry, gene_names, gene_primary, protein_name,
                        organism, length, sequence, string_xref, function_desc,
                    ])
                    rows_written += 1

            else:
                raise DownloadError(
                    f"Unrecognized v50 UniProt output format: {prot_path.name} "
                    f"(suffix={suffix!r}, inner={inner!r}). Expected one of: "
                    f".jsonl, .dat.gz, .csv."
                )

        logger.info(
            "[%s] Normalized %d records to raw TSV: %s",
            self.source_name, rows_written, out_path,
        )
        if rows_written == 0:
            raise DownloadError(
                f"v50 UniProt normalization produced 0 records from "
                f"{prot_path}. The source file may be empty or corrupt."
            )
        return out_path

    @staticmethod
    def _flatten_uniprot_rest_json(rec: dict) -> list:
        """Flatten one UniProt REST JSON record to the 9-column TSV row.

        The UniProt REST JSON schema is deeply nested. This helper
        extracts the canonical fields without raising on missing keys
        (returns empty strings for absent fields so clean()'s null
        handling applies uniformly).
        """
        def _safe_str(v):
            if v is None:
                return ""
            if isinstance(v, (list, dict)):
                return ""
            return str(v)

        # Entry (primary accession).
        entry = _safe_str(rec.get("primaryAccession"))

        # Gene Names (primary) + Gene Names (all).
        genes = rec.get("genes") or []
        gene_primary = ""
        gene_names_parts: list[str] = []
        for g in genes:
            if not isinstance(g, dict):
                continue
            gn = g.get("geneName") or {}
            if isinstance(gn, dict):
                val = gn.get("value")
                if val:
                    if not gene_primary:
                        gene_primary = str(val)
                    gene_names_parts.append(str(val))
            for syn in (g.get("synonyms") or []):
                if isinstance(syn, dict):
                    sv = syn.get("value")
                    if sv:
                        gene_names_parts.append(str(sv))
        gene_names = " ".join(gene_names_parts)

        # Protein names (recommended full name; fall back to submitted names).
        protein_name = ""
        pdesc = rec.get("proteinDescription") or {}
        if isinstance(pdesc, dict):
            rec_name = pdesc.get("recommendedName") or {}
            if isinstance(rec_name, dict):
                fn = rec_name.get("fullName") or {}
                if isinstance(fn, dict):
                    protein_name = _safe_str(fn.get("value"))
            if not protein_name:
                # Fall back to submitted names (first one).
                sub_names = pdesc.get("submittedNames") or []
                if isinstance(sub_names, list) and sub_names:
                    first = sub_names[0]
                    if isinstance(first, dict):
                        fn = first.get("fullName") or {}
                        if isinstance(fn, dict):
                            protein_name = _safe_str(fn.get("value"))

        # Organism (scientific name).
        organism = ""
        org = rec.get("organism") or {}
        if isinstance(org, dict):
            organism = _safe_str(org.get("scientificName"))

        # Length + Sequence.
        seq_obj = rec.get("sequence") or {}
        if not isinstance(seq_obj, dict):
            seq_obj = {}
        length = _safe_str(seq_obj.get("length"))
        sequence = _safe_str(seq_obj.get("value"))

        # Cross-reference (STRING).
        string_xref = ""
        for xref in (rec.get("uniProtKBCrossReferences") or []):
            if isinstance(xref, dict) and xref.get("database") == "STRING":
                string_xref = _safe_str(xref.get("id"))
                break

        # Function [CC].
        function_desc = ""
        for comment in (rec.get("comments") or []):
            if not isinstance(comment, dict):
                continue
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts") or []
                parts = []
                for t in texts:
                    if isinstance(t, dict):
                        parts.append(_safe_str(t.get("value")))
                function_desc = " ".join(p for p in parts if p)
                break

        return [
            entry, gene_names, gene_primary, protein_name,
            organism, length, sequence, string_xref, function_desc,
        ]

    @staticmethod
    def _iter_uniprot_dat_records(in_fh) -> Iterator[dict]:
        """Yield UniProt DAT records as dicts keyed by 2-letter type code.

        Each DAT record starts with ``ID   `` and ends with ``//``. Fields
        are 2-letter type codes (ID, AC, DE, GN, OS, SQ, DR, CC, etc.).
        Multi-line fields are joined; the SQ sequence is concatenated
        from the lines following the ``SQ   SEQUENCE`` header.
        """
        current: dict = {}
        in_sequence = False
        seq_parts: list[str] = []
        for line in in_fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line == "//":
                if current:
                    if seq_parts:
                        current["SQ_sequence"] = "".join(seq_parts)
                    yield current
                current = {}
                in_sequence = False
                seq_parts = []
                continue
            # DAT format: lines start with a 2-letter type code, then
            # 3 spaces, then the value. Some lines are continuation
            # lines (start with spaces).
            if line[:2].isalpha() and line[2:5] == "   ":
                key = line[:2]
                value = line[5:]
                if key == "SQ":
                    in_sequence = True
                    # SQ line contains metadata (length, molecular weight, etc.)
                    current.setdefault("SQ", []).append(value)
                    seq_parts = []
                elif in_sequence and key != "  ":
                    # We left the sequence block.
                    in_sequence = False
                    if seq_parts:
                        current["SQ_sequence"] = "".join(seq_parts)
                        seq_parts = []
                    current.setdefault(key, []).append(value)
                else:
                    current.setdefault(key, []).append(value)
            elif in_sequence:
                # Sequence continuation lines contain only the sequence
                # (with spaces every 10 chars). Strip spaces.
                seq_parts.append(line.replace(" ", ""))
            else:
                # Continuation of the previous field.
                if current:
                    last_key = next(reversed(current))
                    current[last_key][-1] = current[last_key][-1] + " " + line.strip()
        # Flush trailing record (file without final //).
        if current:
            if seq_parts:
                current["SQ_sequence"] = "".join(seq_parts)
            yield current

    @staticmethod
    def _flatten_uniprot_dat_record(rec: dict) -> list:
        """Flatten one parsed DAT record to the 9-column TSV row.

        DAT field reference:
          ID  -- identification (entry name, data class, length)
          AC  -- accession numbers (semicolon-separated; first is primary)
          DE  -- description (protein names)
          GN  -- gene names
          OS  -- organism species
          SQ  -- sequence header (length, MW, CRC64)
          SQ_sequence -- the actual sequence (set by _iter_uniprot_dat_records)
          DR  -- database cross-references
          CC  -- comments (including FUNCTION)
        """
        def _first(values):
            if not values:
                return ""
            return str(values[0]).strip()

        # AC: first accession is primary.
        ac_lines = rec.get("AC") or []
        entry = ""
        if ac_lines:
            first_ac = str(ac_lines[0]).strip()
            entry = first_ac.split(";")[0].strip()

        # DE: parse "RecName: Full=<name>;".
        de_lines = rec.get("DE") or []
        protein_name = ""
        for line in de_lines:
            line = str(line)
            if "RecName:" in line and "Full=" in line:
                # Extract the Full= value up to the next semicolon.
                after = line.split("Full=", 1)[-1]
                protein_name = after.split(";", 1)[0].strip()
                break
        if not protein_name and de_lines:
            # Fallback: use first DE line stripped of leading tags.
            protein_name = str(de_lines[0]).strip().rstrip(";")

        # GN: parse "Name=<symbol>;".
        gn_lines = rec.get("GN") or []
        gene_primary = ""
        gene_names_parts: list[str] = []
        for line in gn_lines:
            line = str(line)
            if "Name=" in line:
                after = line.split("Name=", 1)[-1]
                sym = after.split(";", 1)[0].strip()
                if sym and not gene_primary:
                    gene_primary = sym
                if sym:
                    gene_names_parts.append(sym)
            elif "Synonyms=" in line:
                after = line.split("Synonyms=", 1)[-1]
                sym = after.split(";", 1)[0].strip()
                if sym:
                    gene_names_parts.append(sym)
        gene_names = " ".join(gene_names_parts)

        # OS: organism scientific name.
        os_lines = rec.get("OS") or []
        organism = _first(os_lines)
        if organism and "(" in organism:
            # OS line format: "Homo sapiens (Human)."
            organism = organism.split("(")[0].strip().rstrip(".")

        # SQ: length from "SQ   SEQUENCE  <len> AA; ...".
        sq_lines = rec.get("SQ") or []
        length = ""
        if sq_lines:
            sq_text = str(sq_lines[0])
            # Format: "SEQUENCE 599 AA; 68996 MW;  <crc64> CRC64;"
            import re as _re
            m = _re.search(r"SEQUENCE\s+(\d+)\s+AA", sq_text)
            if m:
                length = m.group(1)
        sequence = rec.get("SQ_sequence") or ""

        # DR: STRING cross-reference.
        string_xref = ""
        for line in (rec.get("DR") or []):
            line = str(line)
            if line.startswith("STRING;"):
                # Format: "STRING; P23219."
                parts = line.split(";")
                if len(parts) >= 2:
                    string_xref = parts[1].strip().rstrip(".")
                    break

        # CC: FUNCTION comment.
        function_desc = ""
        cc_lines = rec.get("CC") or []
        in_function = False
        func_parts: list[str] = []
        for line in cc_lines:
            line = str(line)
            if line.startswith("-!- FUNCTION:"):
                in_function = True
                func_parts.append(line.split("FUNCTION:", 1)[-1].strip())
            elif line.startswith("-!-"):
                # New comment block -- stop FUNCTION collection.
                in_function = False
            elif in_function:
                func_parts.append(line.strip())
        function_desc = " ".join(p for p in func_parts if p)

        return [
            entry, gene_names, gene_primary, protein_name,
            organism, length, sequence, string_xref, function_desc,
        ]

    def download(self) -> Path:
        """Download human-reviewed proteins from the UniProt REST API.

        v50 ROOT FIX: now delegates to `pipelines._v50_downloaders.download_uniprot_full`
        which handles BOTH sample mode (8 proteins via REST) AND full mode
        (streams uniprot_sprot.dat.gz ~500MB from the public FTP -- no login).

        Uses cursor-based pagination via the ``Link`` header.  Handles
        HTTP 429 rate-limiting with exponential backoff + jitter (C41,
        C42).  Writes the TSV atomically via a ``.tmp`` file + rename
        (F5) and computes a SHA-256 sidecar (I4).

        Returns
        -------
        Path
            Path to the downloaded raw TSV file.

        Raises
        ------
        DownloadError
            If the download fails after all retries.
        """
        # v50 ROOT FIX: delegate to the unified downloader.
        try:
            from pipelines._v50_downloaders import download_uniprot_full
            if self.raw_dir is None:
                from config.settings import RAW_DATA_DIR
                self.raw_dir = RAW_DATA_DIR / "uniprot"
            downloaded = download_uniprot_full(self.raw_dir)
            prot_path = downloaded.get("proteins")
            if prot_path and prot_path.exists():
                # v80 FORENSIC ROOT FIX (P0-C1 + P0-C8):
                # ``clean()`` reads the file with ``pd.read_csv(sep="\t")``
                # and expects the RAW UniProt REST TSV schema (columns:
                # Entry, Gene Names, Gene Names (primary), Protein names,
                # Organism, Length, Sequence, Cross-reference (STRING),
                # Function [CC]). However the v50 downloader returns ONE
                # of three formats depending on mode / fallback path:
                #
                #   1. ``.jsonl`` (sample mode)   -- REST JSON, deeply nested
                #   2. ``.dat.gz`` (full/skip)    -- Swiss-Prot DAT, gzipped
                #   3. ``.csv`` (embedded fallback) -- already-cleaned schema
                #
                # The previous code only handled .jsonl -> .csv conversion
                # (and the conversion was BROKEN: ``pd.DataFrame(records)``
                # on nested JSON produced dict-repr columns that
                # ``to_csv(index=False)`` wrote as Python repr strings;
                # then ``clean()`` tried to read the comma-separated CSV
                # with ``sep="\t"`` and crashed). The .dat.gz and embedded
                # .csv paths were returned as-is and also crashed clean().
                #
                # ROOT FIX: normalize ALL three formats to the raw UniProt
                # TSV schema (tab-separated, with the 9 expected columns)
                # via ``_normalize_v50_to_raw_tsv()``. The normalized file
                # is written next to the source as
                # ``uniprot_human_reviewed.tsv`` so clean() can read it
                # unchanged. The original v50 file is preserved for audit.
                normalized_tsv = self._normalize_v50_to_raw_tsv(prot_path)
                return normalized_tsv
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            # v84 FORENSIC ROOT FIX (BUG #31): narrowed from broad
            # ``except Exception``. The previous code caught ALL failures
            # from ``_normalize_v50_to_raw_tsv`` -- including programming
            # bugs (AttributeError, KeyError) -- and silently fell back to
            # the v49 path. A bug in the v50 normalization was masked as
            # "v50 failed, using v49" -- the pipeline ALWAYS fell back to
            # v49, which may be stale or broken, with no visible warning.
            # ROOT FIX: catch ONLY the expected I/O, value, and parse
            # errors. Programming bugs propagate so they surface during
            # development instead of silently degrading to v49 forever.
            logger.warning(
                "[%s] v50 downloader failed (%s) -- falling back to v49 path",
                self.source_name, exc,
            )

        # I3 / I10 -- reset per-run instance state at the start of each download.
        self._consecutive_retry_after = 0
        self._total_retries = 0
        # Note: we do NOT clear dead_letter_queue here -- it is owned by
        # BasePipeline and is drained by teardown().

        output_path = self.effective_raw_dir / "uniprot_human_reviewed.tsv"

        # I8 / D2-2 -- honor force_refresh.
        if self._force_refresh and output_path.exists():
            logger.info(
                "[%s] force_refresh=True -- deleting cached file: %s",
                self.source_name, output_path,
            )
            try:
                output_path.unlink()
            except OSError as exc:
                logger.warning(
                    "[%s] Could not delete cached file %s: %s",
                    self.source_name, output_path, exc,
                )
            # Also delete checksum sidecar.
            checksum_path = output_path.with_suffix(".tsv.sha256")
            if checksum_path.exists():
                try:
                    checksum_path.unlink()
                except OSError:
                    pass

        # F5 / I1 / I4 -- validate cached file before reuse.
        if self._is_raw_file_valid(output_path):
            logger.info(
                "[%s] Valid cached file exists: %s",
                self.source_name, output_path,
            )
            return output_path

        # L11-L14 -- log the download configuration at the start.
        logger.info(
            "[%s] Download configuration: url=%s, query=%s, fields=%s, "
            "page_size=%d, max_retries=%d",
            self.source_name,
            self.uniprot_search_url,
            self.uniprot_query,
            self.uniprot_fields,
            self.page_size,
            self.max_retries,
            extra=self._log_context(),
        )

        # SEC1 -- validate the search URL before fetching.
        self._validate_url(self.uniprot_search_url)

        fields_str = ",".join(self.uniprot_fields)
        params: Optional[dict[str, Any]] = {
            "query": self.uniprot_query,
            "format": "tsv",
            "fields": fields_str,
            "size": self.page_size,
        }

        total_records = 0
        url: Optional[str] = self.uniprot_search_url
        header_written = False
        expected_total: Optional[int] = None
        page_num = 0

        # v21 ROOT FIX (Audit section 6 finding 5 - "Checkpoint writer
        # without reader"): _write_checkpoint is called after every page
        # (line 931) but _read_checkpoint was NEVER CALLED. Large
        # downloads always restarted from page 1 on failure. Honest
        # docstring admitted: "End-to-end resume is not yet wired into
        # download()." Fix: read the checkpoint at the start of
        # download(); if it exists AND the operator has set
        # DRUGOS_UNIPROT_RESUME=1, skip ahead to the saved cursor URL
        # and resume. Default is OFF (resume is opt-in) because the
        # temp file from the previous attempt is discarded on failure
        # and we'd be re-writing from page N to a fresh temp file
        # (which is fine but operators should know).
        #
        # v80 FORENSIC ROOT FIX (P0-C6 -- silent data loss on resume):
        #   The previous implementation opened ``tmp_path`` with mode
        #   ``"w"`` (truncate) UNCONDITIONALLY, even when resuming from
        #   a checkpoint. So pages 1..N-1 that were already fetched in
        #   the previous (failed) run were DISCARDED the moment the new
        #   run opened the temp file. Only page N onward was written.
        #   The final TSV therefore contained ONLY the tail of the
        #   dataset. Operators saw "100K proteins loaded" but reality
        #   was "last 20K only" -- a silent partial-data-load that
        #   corrupted every downstream KG edge / TransE training run
        #   for weeks without any error signal.
        #
        #   ROOT FIX: when resuming, open the temp file in APPEND mode
        #   (``"a"``) AND set ``header_written=True`` if the temp file
        #   already exists with content (the previous run already wrote
        #   the TSV header on page 1). When NOT resuming, behave as
        #   before (``"w"`` truncate, ``header_written=False``). This
        #   preserves the previously-fetched pages so the final TSV
        #   contains the COMPLETE dataset (pages 1..N from the previous
        #   run + pages N..end from this run).
        import os as _os
        is_resuming = False
        # v83 COMP-6: initialize saved_page so the stale-cursor guard
        # below never hits NameError on a non-resume path (Python's
        # short-circuit `and` protects the access, but being explicit
        # is safer for future maintainers).
        saved_page = 0
        saved_total = 0
        if _os.environ.get("DRUGOS_UNIPROT_RESUME", "") == "1":
            ckpt = self._read_checkpoint()
            if ckpt is not None and ckpt.get("cursor_url"):
                saved_page = int(ckpt.get("page_num", 0))
                saved_total = int(ckpt.get("total_records", 0))
                saved_url = ckpt["cursor_url"]
                if saved_url:
                    logger.info(
                        "[%s] DRUGOS_UNIPROT_RESUME=1: resuming from "
                        "checkpoint (page %d, %d records previously "
                        "fetched). Cursor URL: %s",
                        self.source_name, saved_page, saved_total,
                        saved_url[:80] + "..." if len(saved_url) > 80
                        else saved_url,
                    )
                    url = saved_url
                    page_num = saved_page
                    total_records = saved_total
                    is_resuming = True
                    # When resuming, the temp file may already contain
                    # pages 1..N-1 from the previous (failed) run. We
                    # will open it in APPEND mode below and preserve
                    # the existing header (set header_written=True so
                    # the first page of this run does NOT re-write the
                    # header -- that would produce a duplicate header
                    # row in the middle of the file). If the temp file
                    # does NOT exist (e.g. operator deleted it), we
                    # fall back to fresh-write semantics: header_written
                    # stays False and the first page writes the header.
                    # The actual header_written decision is made AFTER
                    # we know whether tmp_path exists (see below).

        # F5 - write to a temp file first, rename atomically on success.
        tmp_path = output_path.with_suffix(".tsv.tmp")

        # Ensure parent directory exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # v80 P0-C6 ROOT FIX: decide open-mode + header_written based on
        # whether we are resuming AND whether the temp file already has
        # content from the previous run.
        tmp_exists_with_content = (
            tmp_path.exists() and tmp_path.stat().st_size > 0
        )
        if is_resuming and tmp_exists_with_content:
            # Resume + existing temp file: append, skip header re-write.
            open_mode = "a"
            header_written = True
            logger.info(
                "[%s] P0-C6: resuming with existing temp file (%d bytes) "
                "-- appending pages, header preserved",
                self.source_name, tmp_path.stat().st_size,
            )
        else:
            # Fresh start (or resume with no existing temp file): truncate.
            open_mode = "w"
            header_written = False
            if is_resuming and not tmp_exists_with_content:
                logger.warning(
                    "[%s] P0-C6: DRUGOS_UNIPROT_RESUME=1 but temp file %s "
                    "does not exist (was deleted?). Starting fresh from "
                    "checkpoint cursor URL -- pages 1..N-1 will be re-fetched.",
                    self.source_name, tmp_path,
                )

        start_time = time.monotonic()

        # v83 COMP-6 ROOT FIX: stale-cursor recovery flag. If resuming
        # and the first page fetch fails with a 4xx (expired cursor),
        # we delete the checkpoint and restart from page 1. This flag
        # ensures we only attempt recovery ONCE (to avoid infinite loop
        # if the initial URL also fails for a different reason).
        _stale_cursor_recovered = False
        # v83 COMP-6: track whether this is the first iteration of a
        # resumed run. The first fetch after a resume is the one that
        # can fail with a stale cursor (the saved cursor URL expired).
        _is_first_resume_fetch = is_resuming

        try:
            with open(tmp_path, open_mode, encoding="utf-8", newline="\n") as fh:
                while url:
                    page_num += 1
                    try:
                        response = self._fetch_page(url, params)
                    except DownloadError as _fetch_exc:
                        # v83 COMP-6 ROOT FIX: detect stale cursor on
                        # the FIRST page of a resumed run. UniProt
                        # cursors expire after ~15 minutes; if the
                        # previous run failed mid-way and the operator
                        # re-triggers with DRUGOS_UNIPROT_RESUME=1,
                        # the saved cursor URL is now stale -> HTTP 400
                        # (invalid cursor) or 404 (cursor not found).
                        # The 4xx is non-retryable per R13, so without
                        # this recovery the pipeline is STUCK until the
                        # operator manually deletes download_checkpoint.json.
                        #
                        # ROOT FIX: if we are resuming AND this is the
                        # first fetch AND we haven't already recovered,
                        # delete the checkpoint + restart from the
                        # initial search URL (page 1). Log loudly so
                        # the operator sees the self-heal in the DAG log.
                        if (
                            _is_first_resume_fetch
                            and not _stale_cursor_recovered
                            and _is_stale_cursor_error(_fetch_exc)
                        ):
                            logger.warning(
                                "[%s] COMP-6 ROOT FIX: resumed cursor "
                                "URL returned 4xx (stale cursor) -- "
                                "deleting checkpoint and restarting "
                                "from page 1. The previous run's "
                                "cursor expired (UniProt cursors "
                                "expire after ~15 min). Original "
                                "error: %s",
                                self.source_name, _fetch_exc,
                            )
                            self._delete_checkpoint()
                            _stale_cursor_recovered = True
                            _is_first_resume_fetch = False
                            # Reset to fresh-start state.
                            url = self.uniprot_search_url
                            params = {
                                "query": self.uniprot_query,
                                "format": "tsv",
                                "fields": fields_str,
                                "size": self.page_size,
                            }
                            page_num = 0
                            total_records = 0
                            expected_total = None
                            # Truncate the temp file so we start fresh.
                            fh.seek(0)
                            fh.truncate()
                            header_written = False
                            is_resuming = False  # we're a fresh run now
                            continue
                        # Not a stale-cursor case, or already recovered
                        # -- re-raise to the outer except.
                        raise

                    # First fetch succeeded -- clear the flag so subsequent
                    # 4xx errors are NOT treated as stale-cursor cases.
                    _is_first_resume_fetch = False

                    # DQ13 -- capture the total result count from the response.
                    x_total = response.headers.get("X-Total-Results")
                    if x_total and expected_total is None:
                        try:
                            expected_total = int(x_total)
                            logger.info(
                                "[%s] UniProt reports %d total results",
                                self.source_name, expected_total,
                            )
                        except ValueError:
                            pass

                    # R12 / R15 / R16 -- validate Content-Type.
                    content_type = response.headers.get("Content-Type", "")
                    if (
                        "text/tab-separated-values" not in content_type
                        and "text/plain" not in content_type
                    ):
                        logger.warning(
                            "[%s] Unexpected Content-Type on page %d: %s",
                            self.source_name, page_num, content_type,
                        )

                    # L5 -- warn on empty response body instead of silent break.
                    text = (response.text or "").strip()
                    if not text:
                        logger.warning(
                            "[%s] Empty response body on page %d",
                            self.source_name, page_num,
                        )
                        break

                    # C2 -- use splitlines() to handle \r\n and \n consistently.
                    lines = text.splitlines()
                    if not lines:
                        break

                    # F3 / C1 -- correctly skip the re-emitted TSV header on
                    # subsequent pages.  UniProt re-emits the header on every
                    # cursor page.
                    if not header_written:
                        # First page: write header + data.
                        fh.write(lines[0] + "\n")
                        header_written = True
                        data_lines = lines[1:]
                    else:
                        # Subsequent pages: skip the re-emitted header row.
                        # v83 FORENSIC ROOT FIX (P2-10): the previous code
                        # checked ``lines[0].startswith("Entry\t") or
                        # lines[0] == "Entry"``. The second condition
                        # (``== "Entry"`` exactly, no tab) is dead code --
                        # UniProt TSV always has multiple columns
                        # (``Entry\tEntry Name\t...``), so a bare ``"Entry"``
                        # with no tab is impossible. ROOT FIX: removed the
                        # dead branch; keep only the ``startswith("Entry\t")``
                        # check which is the real UniProt header signature.
                        if lines[0].startswith("Entry\t"):
                            data_lines = lines[1:]
                        else:
                            # No header on this page -- keep all lines.
                            data_lines = lines

                    # C4 -- filter out blank lines (some pages emit a trailing
                    # blank line which would create a phantom "" uniprot_id row).
                    data_lines = [ln for ln in data_lines if ln.strip()]

                    # P3 -- bulk write instead of line-by-line.
                    if data_lines:
                        fh.write("\n".join(data_lines) + "\n")

                    total_records += len(data_lines)
                    logger.info(
                        "[%s] Page %d: fetched %d proteins (total: %d)",
                        self.source_name, page_num, len(data_lines),
                        total_records,
                        extra=self._log_context(),
                    )

                    # R8 -- write a checkpoint after each page so we can
                    # resume from cursor if needed (not yet implemented
                    # end-to-end, but the checkpoint is written for diagnosis).
                    next_url = self._parse_link_header(
                        response.headers.get("Link", "")
                    )
                    self._write_checkpoint(next_url or "", page_num, total_records)

                    # Cursor URL already has all params embedded (C33).
                    url = next_url
                    params = None

            # DQ13 -- validate total count.
            if expected_total is not None and total_records != expected_total:
                logger.warning(
                    "[%s] Record count mismatch: fetched %d, UniProt reported "
                    "%d total results.  This may indicate a pagination bug or "
                    "data changed mid-fetch (I15).",
                    self.source_name, total_records, expected_total,
                )

            # F5 -- atomic rename.  Only after the full download succeeds.
            tmp_path.replace(output_path)

            # v83 COMP-6 ROOT FIX: delete the checkpoint on success so
            # the next run does not reuse a stale cursor URL. The
            # previous code NEVER deleted the checkpoint -- on a failed-
            # then-resumed run, the stale cursor caused HTTP 400 and the
            # pipeline was stuck. Deleting here guarantees the next run
            # starts fresh from page 1 unless the operator explicitly
            # sets DRUGOS_UNIPROT_RESUME=1 AND the checkpoint was
            # written by a FAILED (not successful) previous run.
            self._delete_checkpoint()

        except (OSError, PermissionError) as exc:
            # R24 / R25 -- disk full or permission denied.
            logger.error(
                "[%s] OS error during download: %s (disk full or permission denied?)",
                self.source_name, exc,
                exc_info=getattr(self, "log_exc_info", True),
            )
            # SEC17 -- securely delete the partial temp file.
            self._secure_delete(tmp_path)
            raise DownloadError(f"OS error during download: {exc}") from exc
        except (OSError, RuntimeError, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
            # Any other failure: clean up the temp file.
            self._secure_delete(tmp_path)
            raise

        # F5 / I4 -- write a SHA-256 sidecar so subsequent runs can verify
        # the cached file's integrity.
        self._write_checksum(output_path)

        # SEC10 / SEC14 -- restrict file permissions to owner-only.
        self._set_secure_permissions(output_path)

        elapsed = time.monotonic() - start_time
        logger.info(
            "[%s] Downloaded %d total protein records to %s in %.2fs "
            "(total retries: %d)",
            self.source_name, total_records, output_path,
            elapsed, self._total_retries,
            extra={**self._log_context(), "duration_seconds": elapsed,
                   "total_records": total_records},
        )

        return output_path

    # ---------------------------------------------------------------------
    # _fetch_page() -- exponential backoff + jitter, rate limiter (C41, C42)
    # ---------------------------------------------------------------------
    def _fetch_page(
        self, url: str, params: Optional[dict[str, Any]] = None,
    ) -> requests.Response:
        """Fetch a single page from UniProt with retry on rate-limiting.

        Uses exponential backoff with jitter (C41, C42) and raises
        ``DownloadError`` on exhaustion (C39).  Distinguishes 4xx
        (permanent -- do not retry) from 5xx (transient -- retry) per R13.

        Parameters
        ----------
        url : str
            URL to fetch.  Validated against the allow-list (SEC1).
        params : dict | None
            Query parameters.  Used only for the first page; subsequent
            pages use the cursor URL embedded in the ``Link`` header
            (C33 -- ``params=None`` on subsequent calls).

        Returns
        -------
        requests.Response
            Successful response (HTTP 200, no Retry-After).

        Raises
        ------
        DownloadError
            If all retries are exhausted (C39, C40).
        ValueError
            If the URL is not in the allowed domains (SEC1).
        """
        import random

        # SEC1 -- validate URL before fetching.
        self._validate_url(url)

        last_exception: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # A5 / R21 -- be polite to the API; rate-limit before each call.
                if getattr(self, "_rate_limiter", None) is not None:
                    try:
                        self._rate_limiter.wait()
                    except (OSError, RuntimeError, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
                        # Rate limiter should never raise, but be defensive.
                        pass

                # A4 / R22 / P13 -- reuse the HTTP session for connection pooling.
                session = self._get_http_session()
                # v42 ROOT FIX (P1-A-12): pass the FULL tuple
                # ``download_timeout = (connect, read) = (30.0, 600.0)``
                # rather than just the read timeout. The previous
                # ``timeout=self.download_timeout[1]`` passed a single
                # float (600.0), which requests interprets as BOTH the
                # connect AND read timeout -- connect timeout became
                # 600s instead of 30s, so a hung TCP handshake wasted
                # 10 minutes per page (instead of failing fast at 30s).
                resp = session.get(
                    url,
                    params=params,
                    timeout=self.download_timeout,  # CFG16 -- (connect, read) tuple
                )

                # SEC6 -- cap response body size to prevent OOM.
                content_length = resp.headers.get("Content-Length")
                if content_length:
                    try:
                        size = int(content_length)
                        if size > _MAX_RESPONSE_BYTES:
                            raise DownloadError(
                                f"Response too large: {size} bytes "
                                f"(max: {_MAX_RESPONSE_BYTES})"
                            )
                    except ValueError:
                        pass

                if resp.status_code == 429:
                    # C41 -- exponential backoff.
                    # FIX-P2-C-13 (audit P2): the previous code computed
                    # ``delay = self.base_retry_delay * (2 ** (attempt - 1))``
                    # with NO cap. With ``base_retry_delay=10`` and
                    # ``max_retries=5``, attempt 5 slept
                    # ``10 * 2^4 = 160s`` + up to ``80s`` jitter = ``240s``.
                    # A 4-minute single-thread sleep stalls the pipeline far
                    # longer than necessary and exceeds the configured
                    # ``max_retry_after_wait`` (300s) once jitter is added.
                    # Cap the BASE delay at ``max_retry_after_wait`` BEFORE
                    # adding jitter so the worst case stays bounded.
                    delay = min(
                        self.base_retry_delay * (2 ** (attempt - 1)),
                        self.max_retry_after_wait,
                    )
                    # C42 -- random jitter (0 to 50% of delay).
                    jitter = random.uniform(0, delay * 0.5)
                    total_delay = delay + jitter
                    self._total_retries += 1
                    logger.warning(
                        "[%s] Rate-limited by UniProt (HTTP 429), sleeping "
                        "%.1fs (attempt %d/%d)",
                        self.source_name, total_delay,
                        attempt, self.max_retries,
                        extra=self._log_context(),
                    )
                    time.sleep(total_delay)
                    continue

                # R13 -- 4xx is permanent (our request is malformed).  Don't retry.
                if 400 <= resp.status_code < 500:
                    resp.raise_for_status()

                # 5xx is transient -- raise_for_status will raise, then we retry.
                resp.raise_for_status()

                # Handle Retry-After header (UniProt sometimes returns 200 +
                # Retry-After for heavy queries).
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    self._consecutive_retry_after += 1
                    # C8 -- break the loop after N consecutive Retry-Afters.
                    if self._consecutive_retry_after > self.consecutive_retry_after_limit:
                        logger.warning(
                            "[%s] %d consecutive Retry-After headers. "
                            "Returning response to prevent infinite loop.",
                            self.source_name, self._consecutive_retry_after,
                        )
                        self._consecutive_retry_after = 0
                        return resp

                    wait = self._parse_retry_after(retry_after)  # C6 / SEC7
                    self._total_retries += 1
                    logger.info(
                        "[%s] UniProt Retry-After: %ds (consecutive: %d)",
                        self.source_name, wait, self._consecutive_retry_after,
                    )
                    time.sleep(wait)
                    continue

                # Success -- reset the consecutive-retry counter.
                self._consecutive_retry_after = 0
                return resp

            except requests.exceptions.RequestException as exc:
                last_exception = exc

                # v29 ROOT FIX (audit P1-15): 4xx errors (except 429) are
                # permanent client errors -- retrying wastes API quota. Only
                # retry 5xx and network errors.  Although the 4xx branch above
                # calls raise_for_status() (which raises HTTPError), that
                # HTTPError is caught here by RequestException and would be
                # retried 5× without this guard.
                if isinstance(exc, requests.exceptions.HTTPError):
                    resp_exc = getattr(exc, "response", None)
                    status = getattr(resp_exc, "status_code", None)
                    if (status is not None
                            and 400 <= status < 500
                            and status != 429):
                        logger.warning(
                            "[%s] HTTP %d -- permanent client error, not "
                            "retrying: %s",
                            self.source_name, status, exc,
                            extra=self._log_context(),
                        )
                        raise DownloadError(
                            f"HTTP {status} permanent client error "
                            f"(not retried): {exc}"
                        ) from exc

                if attempt == self.max_retries:
                    # C39 -- raise DownloadError, not RuntimeError.
                    raise DownloadError(
                        f"Failed to fetch UniProt page after "
                        f"{self.max_retries} retries: {exc}"
                    ) from exc

                # C41 -- exponential backoff.
                delay = self.base_retry_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, delay * 0.5)
                total_delay = delay + jitter
                self._total_retries += 1
                logger.warning(
                    "[%s] Request failed: %s, retrying in %.1fs (attempt %d/%d)",
                    self.source_name, exc, total_delay,
                    attempt, self.max_retries,
                    exc_info=getattr(self, "log_exc_info", True),
                )
                time.sleep(total_delay)

        # C40 -- all retries exhausted without a return.
        raise DownloadError(
            f"Failed to fetch UniProt page after {self.max_retries} retries"
            + (f": {last_exception}" if last_exception else "")
        )

    # ---------------------------------------------------------------------
    # _parse_link_header() -- URL-validated Link parsing (C5, C32, C34, SEC8)
    # ---------------------------------------------------------------------
    @staticmethod
    def _parse_link_header(link_header: Optional[str]) -> Optional[str]:
        """Extract the ``next`` URL from a ``Link`` header (C5, SEC8).

        Validates the URL domain to prevent SSRF (SEC1/SEC8).  Handles
        the rare case where a comma appears inside a URL by using a
        regex that anchors on ``<...>`` boundaries (C34).  Tolerates
        arbitrary whitespace around the ``;`` separator (C5).

        Parameters
        ----------
        link_header : str | None
            Raw ``Link`` header value.

        Returns
        -------
        str | None
            The ``next`` URL, or *None* if not present or rejected.
        """
        if not link_header or not isinstance(link_header, str):
            return None
        # C34/C5 -- match <URL> ; rel="next" with the URL enclosed in angle
        # brackets.  Allow arbitrary whitespace between '>' and ';' and
        # between ';' and 'rel='.  This correctly handles commas inside
        # URLs (rare but possible).
        for match in re.finditer(
            r'<([^>]+)>\s*;\s*rel="next"', link_header,
        ):
            url = match.group(1)
            try:
                UniProtPipeline._validate_url(url)
            except ValueError:
                logger.warning(
                    "Link header URL rejected (domain not allowed): %s",
                    url[:100],
                )
                return None
            return url
        return None

    # ---------------------------------------------------------------------
    # _parse_retry_after() -- delta-seconds and HTTP-date (C6, SEC7, R1)
    # ---------------------------------------------------------------------
    def _parse_retry_after(self, retry_after: str) -> int:
        """Parse a ``Retry-After`` header value into seconds (C6, SEC7).

        Handles both delta-seconds (``"120"``) and HTTP-date format
        (``"Wed, 21 Oct 2025 07:28:00 GMT"``).  Caps the wait at
        ``self.max_retry_after_wait`` to prevent a malicious server from
        stalling the pipeline indefinitely (SEC7 / C43).

        Parameters
        ----------
        retry_after : str
            The ``Retry-After`` header value.

        Returns
        -------
        int
            Seconds to wait, capped at ``max_retry_after_wait`` and
            floored at 0.
        """
        # Try delta-seconds first.
        try:
            wait = int(retry_after)
        except (ValueError, TypeError):
            # C6 -- try HTTP-date format.
            try:
                dt = parsedate_to_datetime(retry_after)
                if dt is not None:
                    now = datetime.now(timezone.utc)
                    if dt.tzinfo is None:
                        from datetime import timezone as _tz
                        dt = dt.replace(tzinfo=_tz.utc)
                    wait = max(0, int((dt - now).total_seconds()))
                else:
                    wait = int(self.base_retry_delay)
            except (ValueError, TypeError, OverflowError):
                logger.warning(
                    "[%s] Unparseable Retry-After header: %r. "
                    "Using default wait of %.1fs.",
                    self.source_name, retry_after, self.base_retry_delay,
                )
                wait = int(self.base_retry_delay)

        # SEC7 / C43 -- cap at maximum.
        if wait > self.max_retry_after_wait:
            logger.warning(
                "[%s] Retry-After value %ds exceeds maximum %ds. Capping.",
                self.source_name, wait, self.max_retry_after_wait,
            )
            wait = self.max_retry_after_wait

        return max(0, wait)

    # ---------------------------------------------------------------------
    # _is_raw_file_valid() (F5, I1, I4, CFG19)
    # ---------------------------------------------------------------------
    def _is_raw_file_valid(self, path: Path) -> bool:
        """Check whether a cached raw file is valid for reuse (F5, I4).

        A file is valid if:
        1. It exists and has non-zero size.
        2. It has at least 2 lines (header + ≥ 1 data row) -- guards
           against partial downloads where only the header was written.
        3. It is not older than ``max_cache_age_days`` (CFG19).
        4. Its SHA-256 checksum matches the stored checksum (if a
           ``.sha256`` sidecar exists; I4).

        Parameters
        ----------
        path : Path
            Path to the cached raw TSV.

        Returns
        -------
        bool
            *True* if the file is valid and can be reused.
        """
        try:
            if not path.exists() or path.stat().st_size == 0:
                return False
        except OSError:
            return False

        # Check minimum row count (header + at least 1 data row).
        try:
            with open(path, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            if line_count < 2:
                logger.warning(
                    "[%s] Cached file has < 2 lines (%s). Re-downloading.",
                    self.source_name, path,
                )
                return False
        except (OSError, UnicodeDecodeError):
            return False

        # CFG19 -- check age.
        try:
            file_age_days = (time.time() - path.stat().st_mtime) / 86400
            if file_age_days > self.max_cache_age_days:
                logger.info(
                    "[%s] Cached file is %d days old (max: %d). Re-downloading.",
                    self.source_name, int(file_age_days), self.max_cache_age_days,
                )
                return False
        except OSError:
            return False

        # I4 -- check SHA-256 if a sidecar exists.
        checksum_path = path.with_suffix(path.suffix + ".sha256")
        if checksum_path.exists():
            try:
                stored_hash = checksum_path.read_text(encoding="utf-8").strip().split()[0]
                actual_hash = self._compute_sha256(path)
                if actual_hash != stored_hash:
                    logger.warning(
                        "[%s] SHA-256 mismatch for %s. Re-downloading.",
                        self.source_name, path,
                    )
                    return False
            except (OSError, IndexError, ValueError):
                logger.warning(
                    "[%s] Could not verify checksum for %s. Re-downloading.",
                    self.source_name, path,
                )
                return False

        return True

    # ---------------------------------------------------------------------
    # _compute_sha256() (L21)
    # ---------------------------------------------------------------------
    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute the SHA-256 hexdigest of *path* (64 KB streaming).

        Parameters
        ----------
        path : Path
            File to hash.

        Returns
        -------
        str
            64-character lowercase hex SHA-256 digest.
        """
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()

    # ---------------------------------------------------------------------
    # _write_checksum() (I4, L21)
    # ---------------------------------------------------------------------
    def _write_checksum(self, path: Path) -> None:
        """Write a SHA-256 checksum sidecar for *path* (I4).

        Sidecar filename: ``<path>.sha256`` (so ``.tsv`` -> ``.tsv.sha256``).
        Sidecar format: ``<hexdigest>  <filename>\\n`` (the standard
        ``sha256sum`` format so ``sha256sum -c`` works).
        """
        try:
            digest = self._compute_sha256(path)
            checksum_path = path.with_suffix(path.suffix + ".sha256")
            checksum_path.write_text(
                f"{digest}  {path.name}\n", encoding="utf-8",
            )
            self._set_secure_permissions(checksum_path)
            logger.info(
                "[%s] Wrote SHA-256 checksum: %s (digest: %s)",
                self.source_name, checksum_path, digest,
            )
        except OSError as exc:
            logger.warning(
                "[%s] Could not write checksum sidecar for %s: %s",
                self.source_name, path, exc,
            )

    # ---------------------------------------------------------------------
    # _stage_raw_file() (A9, SEC15)
    # ---------------------------------------------------------------------
    def _stage_raw_file(self, raw_path: Path) -> Path:
        """Copy the raw download to an immutable staging area (A9, SEC15).

        The raw TSV is both a download artifact and the input to
        ``clean()``.  This method creates an immutable copy with a
        SHA-256 checksum so that ``clean()`` always reads the same data,
        even if the original raw file is modified or deleted between runs.

        Parameters
        ----------
        raw_path : Path
            Path to the raw downloaded TSV.

        Returns
        -------
        Path
            Path to the staged copy.
        """
        staged_dir = self.effective_raw_dir / "staged"
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_path = staged_dir / raw_path.name

        if staged_path.exists():
            # Verify checksum -- if match, skip the copy.
            try:
                raw_hash = self._compute_sha256(raw_path)
                staged_hash = self._compute_sha256(staged_path)
                if raw_hash == staged_hash:
                    return staged_path
            except OSError:
                pass  # fall through to re-copy

        try:
            shutil.copy2(raw_path, staged_path)
            logger.info(
                "[%s] Staged raw file: %s -> %s",
                self.source_name, raw_path, staged_path,
            )
        except OSError as exc:
            logger.warning(
                "[%s] Could not stage raw file %s: %s",
                self.source_name, raw_path, exc,
            )
            return raw_path
        return staged_path

    # ---------------------------------------------------------------------
    # _write_checkpoint() / _read_checkpoint() (R8)
    # ---------------------------------------------------------------------
    def _write_checkpoint(
        self, cursor_url: str, page_num: int, total_records: int,
    ) -> None:
        """Write a checkpoint file for resume support (R8).

        The checkpoint records the next-page cursor URL, the current
        page number, and the running record count.  End-to-end resume
        is not yet wired into ``download()`` (the temp file is discarded
        on failure), but the checkpoint is written for diagnosis and
        future implementation.

        Parameters
        ----------
        cursor_url : str
            The next-page cursor URL (empty string if no more pages).
        page_num : int
            Current page number (1-indexed).
        total_records : int
            Records fetched so far.
        """
        checkpoint = {
            "cursor_url": cursor_url,
            "page_num": page_num,
            "total_records": total_records,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": getattr(self, "run_id", None),
        }
        checkpoint_path = self.effective_raw_dir / "download_checkpoint.json"
        try:
            checkpoint_path.write_text(
                json.dumps(checkpoint, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            logger.debug(
                "[%s] Could not write checkpoint: %s",
                self.source_name, exc,
            )

    def _read_checkpoint(self) -> Optional[dict[str, Any]]:
        """Read the last checkpoint, if any (R8).

        Returns
        -------
        dict | None
            Checkpoint dict, or *None* if no checkpoint exists or it is
            unparseable.
        """
        checkpoint_path = self.effective_raw_dir / "download_checkpoint.json"
        if not checkpoint_path.exists():
            return None
        try:
            return json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError, OSError):
            return None

    def _delete_checkpoint(self) -> None:
        """Delete the checkpoint file (COMP-6 ROOT FIX).

        Called after a successful download so the next run does not
        reuse a stale cursor URL. The previous code NEVER deleted the
        checkpoint -- even on full success -- so the checkpoint file
        persisted with the LAST ``next_url`` (empty string on success,
        but a valid cursor URL if the download FAILED mid-way). On the
        next run with ``DRUGOS_UNIPROT_RESUME=1``, the stale cursor
        was reused; UniProt returns HTTP 400 (invalid cursor) for
        expired cursors, which is a non-retryable 4xx -> the pipeline
        was stuck until the operator manually deleted the checkpoint
        file.

        ROOT FIX: delete the checkpoint on success. Combined with the
        stale-cursor detection in ``download()`` (which deletes the
        checkpoint + restarts from page 1 if the resumed cursor
        returns 4xx), the pipeline self-heals instead of getting stuck.
        """
        checkpoint_path = self.effective_raw_dir / "download_checkpoint.json"
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(
                    "[%s] Deleted checkpoint after successful download "
                    "(COMP-6 ROOT FIX): %s",
                    self.source_name, checkpoint_path.name,
                )
        except OSError as exc:
            # Non-fatal -- the checkpoint will be overwritten on the
            # next run's first page write. Log at DEBUG so operators
            # can diagnose permission issues if they arise.
            logger.debug(
                "[%s] Could not delete checkpoint %s: %s "
                "(non-fatal -- will be overwritten on next run)",
                self.source_name, checkpoint_path.name, exc,
            )

    # ---------------------------------------------------------------------
    # clean() -- full cleaning pipeline (F2, F3, F4, S1-S25, DQ1-DQ25, I2, I6)
    # ---------------------------------------------------------------------
    def clean(self, raw_path: Path) -> pd.DataFrame:
        """Clean and normalize UniProt protein data.

        Implements the full cleaning pipeline:

        1. Read the TSV with explicit dtypes (C29, C30, C31).
        2. Validate that expected TSV columns are present (D2-8, INT14).
        3. Rename columns to match the ``proteins`` table schema.
        4. Validate that critical columns were renamed successfully (C11).
        5. Extract ``gene_symbol`` from ``gene_names`` if missing (S3).
        6. Extract ``protein_name_canonical`` from ``protein_name`` (F4, S4).
        7. Set ``gene_name = None`` (deprecated -- F4).
        8. Clean ``function_desc`` (S5, S6, S7, S16, S17).
        9. Extract ``string_id`` and ``all_string_ids`` (S8, S9).
        10. Validate ``uniprot_id`` format (S20, DQ1).
        11. Validate ``length`` range (DQ14).
        12. Validate ``sequence`` characters (S21, DQ10).
        13. Cross-validate ``length`` vs ``len(sequence)`` (S11, DQ4).
        14. Detect & log duplicate ``uniprot_id``s with content hash (DQ2, I14).
        15. Sort by ``uniprot_id`` for deterministic dedup (I2, I6).
        16. Drop rows with null ``uniprot_id`` (DQ19 -- dead-letter).
        17. Drop duplicate ``uniprot_id``s (keep first).
        18. Validate organism -- log non-Homo sapiens records (S10, DQ5).
        19. Handle missing protein fields via ``handle_missing_protein_fields``
            with ``organism_fill_mode="strict"`` (S10).
        20. Ensure all required output columns exist (F4, C48, DQ18).
        21. Add lineage columns (LIN2, LIN7, LIN8).
        22. Compute DQ metrics (DQ20, L23).
        23. Sanitize for CSV formula injection (SEC4, C27).

        Parameters
        ----------
        raw_path : Path
            Path to the raw TSV file from ``download()``.

        Returns
        -------
        pd.DataFrame
            Cleaned protein DataFrame.  The base class ``run()``
            persists this to ``proteins.csv`` via
            ``_persist_cleaned_data()`` (A3 -- we do NOT write the CSV
            ourselves).
        """
        with self._timed_operation("clean"):
            # ---------- Step 1: read TSV (C29, C30, C31) ----------
            try:
                df = pd.read_csv(
                    raw_path,
                    sep="\t",
                    dtype=str,                       # read everything as string
                    na_values=["", "null", "None", "N/A", "NaN"],
                    keep_default_na=True,
                    encoding="utf-8",
                )
            except (pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
                # L7 -- wrap read failure with file context.
                raise DownloadError(
                    f"Failed to read UniProt TSV {raw_path}: {exc}"
                ) from exc

            logger.info(
                "[%s] Loaded %d raw protein records from %s",
                self.source_name, len(df), raw_path,
                extra=self._log_context(),
            )

            # L19 -- log raw vs cleaned ratio at the end.
            raw_count = len(df)
            self._log_null_counts(df, stage="raw")

            # ---------- Step 2: validate TSV columns (D2-8, INT14) ----------
            actual_columns = set(df.columns)
            missing = _EXPECTED_TSV_COLUMNS - actual_columns
            if missing:
                logger.warning(
                    "[%s] Expected TSV columns not found: %s. "
                    "Available: %s. This may indicate a UniProt API change.",
                    self.source_name, sorted(missing), sorted(actual_columns),
                )

            # INT14 -- log unknown (future) columns gracefully.
            extra_columns = actual_columns - _EXPECTED_TSV_COLUMNS
            if extra_columns:
                logger.info(
                    "[%s] UniProt returned unexpected columns (future API "
                    "addition?): %s. These will be dropped during cleaning.",
                    self.source_name, sorted(extra_columns),
                )

            # ---------- Step 3: rename columns ----------
            column_map: dict[str, str] = {
                "Entry": "uniprot_id",
                "Gene Names": "gene_names",
                "Gene Names (primary)": "gene_symbol",
                "Protein names": "protein_name",
                "Organism": "organism",
                "Length": "length",
                "Sequence": "sequence",
                # S18 -- UniProt REST uses "Cross-reference (STRING)" for xref_string.
                "Cross-reference (STRING)": "string_xref",
                "Function [CC]": "function_desc",
            }
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

            # ---------- Step 4: validate critical columns (C11) ----------
            for col in _CRITICAL_COLUMNS:
                if col not in df.columns:
                    raise ValueError(
                        f"Critical column '{col}' not found after rename. "
                        f"Available columns: {list(df.columns)}. "
                        f"This usually means UniProt's TSV column names have "
                        f"changed. Check UNIPROT_FIELDS and column_map."
                    )

            # L6 -- warn for ALL missing important columns, not just gene_symbol.
            self._log_missing_columns(df)

            # ---------- Step 5: gene_symbol (S3) ----------
            # If 'gene_symbol' wasn't mapped (column missing), extract from
            # 'gene_names' (first token).  Also validate HGNC format.
            if "gene_symbol" not in df.columns or df["gene_symbol"].isna().all():
                if "gene_names" in df.columns:
                    logger.warning(
                        "[%s] 'Gene Names (primary)' column not found in TSV. "
                        "Extracting gene_symbol from 'Gene Names' (first token).",
                        self.source_name,
                    )
                    df["gene_symbol"] = df["gene_names"].apply(
                        lambda x: str(x).split()[0]
                        if pd.notna(x) and str(x).strip() else None
                    )

            if "gene_symbol" in df.columns:
                df["gene_symbol"] = df["gene_symbol"].apply(self._validate_gene_symbol)

            # ---------- Step 6 & 7: protein_name_canonical + gene_name=None (F4) ----------
            if "protein_name" in df.columns:
                df["protein_name_canonical"] = df["protein_name"].apply(
                    self._extract_canonical_name
                )
            else:
                df["protein_name_canonical"] = None

            # F4 -- gene_name is DEPRECATED.  Set to None to stop the data
            # corruption.  Downstream code MUST use protein_name_canonical
            # (for canonical names) or gene_symbol (for gene symbols).
            df["gene_name"] = None
            self._log_transformation(
                "deprecate_gene_name", raw_count, raw_count,
                {"reason": "gene_name set to None (deprecated). "
                           "Use protein_name_canonical / gene_symbol."},
            )

            # ---------- Step 8: clean function_desc (S5, S6, S7, S16, S17) ----------
            if "function_desc" in df.columns:
                before_func = df["function_desc"].notna().sum()
                df["function_desc"] = df["function_desc"].apply(self._clean_function_desc)
                after_func = df["function_desc"].notna().sum()
                self._log_transformation(
                    "clean_function_desc", before_func, after_func,
                    {"reason": "Stripped FUNCTION: prefix, ECO tags, sub-section markers."},
                )
            else:
                df["function_desc"] = None

            # ---------- Step 9: extract string_id + all_string_ids (S8, S9) ----------
            if "string_xref" in df.columns:
                df["string_id"] = df["string_xref"].apply(self._extract_string_id)
                df["all_string_ids"] = df["string_xref"].apply(
                    self._extract_all_string_ids
                )
            else:
                df["string_id"] = None
                df["all_string_ids"] = None

            # DQ12 -- log duplicate string_ids.
            if "string_id" in df.columns:
                dup_string = df[df["string_id"].notna()]["string_id"].duplicated().sum()
                if dup_string > 0:
                    logger.info(
                        "[%s] %d duplicate string_ids found (multiple proteins "
                        "mapping to the same STRING ID). This is expected for "
                        "isoforms but may indicate a data issue.",
                        self.source_name, dup_string,
                    )

            # ---------- Step 10: validate uniprot_id format (S20, DQ1) ----------
            if "uniprot_id" in df.columns:
                invalid_mask = df["uniprot_id"].apply(
                    lambda x: pd.notna(x)
                    and isinstance(x, str)
                    and not _UNIPROT_ACCESSION_RE.match(x)
                )
                invalid_count = int(invalid_mask.sum())
                # v83 FORENSIC ROOT FIX (P2-11): capture the PRE-quarantine
                # total + invalid counts so ``_compute_dq_metrics`` can
                # report a MEANINGFUL ``validity_uniprot_id_raw`` metric.
                # The previous code computed ``validity_uniprot_id`` on the
                # POST-quarantine DataFrame (after invalid records were
                # removed at this step), so the metric was ALWAYS 1.0 --
                # meaningless. ROOT FIX: store the raw counts as instance
                # attributes; ``_compute_dq_metrics`` reads them to compute
                # a real validity ratio (valid / total_raw).
                self._uniprot_id_raw_total = len(df)
                self._uniprot_id_raw_invalid = invalid_count
                if invalid_count > 0:
                    invalid_ids = df.loc[invalid_mask, "uniprot_id"].head(10).tolist()
                    logger.warning(
                        "[%s] %d records have invalid UniProt accession format: %s",
                        self.source_name, invalid_count, invalid_ids,
                    )
                    # DQ19 -- quarantine invalid records.
                    if hasattr(self, "dead_letter_queue"):
                        invalid_df = df[invalid_mask].copy()
                        for _, row in invalid_df.iterrows():
                            self._quarantine_record(
                                row.to_dict(), "invalid_uniprot_id_format"
                            )
                    df = df[~invalid_mask].copy()

            # ---------- Step 11: validate length range (DQ14) ----------
            if "length" in df.columns:
                # Coerce to nullable Int64.
                df["length"] = pd.to_numeric(df["length"], errors="coerce").astype("Int64")
                invalid_length = df[
                    df["length"].notna() & (
                        (df["length"] < 1) | (df["length"] > 100000)
                    )
                ]
                if len(invalid_length) > 0:
                    logger.warning(
                        "[%s] %d records have length outside [1, 100000]: %s",
                        self.source_name, len(invalid_length),
                        invalid_length[["uniprot_id", "length"]].head(5).to_dict("records"),
                    )
                    # Set out-of-range lengths to None.
                    df.loc[
                        df["length"].notna() & (
                            (df["length"] < 1) | (df["length"] > 100000)
                        ),
                        "length",
                    ] = pd.NA

            # ---------- Step 12: validate sequence characters (S21, DQ10) ----------
            if "sequence" in df.columns:
                df["sequence"] = df["sequence"].apply(self._validate_sequence)

            # ---------- Step 13: cross-validate length vs sequence (S11, DQ4) ----------
            if "length" in df.columns and "sequence" in df.columns:
                mismatch_mask = df.apply(
                    lambda r: (
                        pd.notna(r["length"])
                        and isinstance(r["sequence"], str)
                        and int(r["length"]) != len(r["sequence"])
                    ),
                    axis=1,
                )
                mismatch_count = int(mismatch_mask.sum())
                if mismatch_count > 0:
                    mismatch_ids = df.loc[mismatch_mask, "uniprot_id"].head(5).tolist()
                    logger.warning(
                        "[%s] %d proteins have length != len(sequence). "
                        "This may indicate API or pipeline corruption. "
                        "First mismatching accessions: %s",
                        self.source_name, mismatch_count, mismatch_ids,
                    )

            # ---------- Step 14: detect & log duplicate uniprot_ids (DQ2, I14) ----------
            if "uniprot_id" in df.columns:
                dup_count = int(df["uniprot_id"].duplicated().sum())
                if dup_count > 0:
                    logger.warning(
                        "[%s] %d duplicate uniprot_ids found in raw data. "
                        "UniProt human reviewed should have ZERO duplicates. "
                        "This may indicate a pagination bug or API regression.",
                        self.source_name, dup_count,
                    )
                    dup_ids = df[df["uniprot_id"].duplicated(keep=False)][
                        "uniprot_id"
                    ].unique().tolist()[:10]
                    logger.warning(
                        "[%s] Duplicate accessions (first 10): %s",
                        self.source_name, dup_ids,
                    )
                    # I14 -- log content hash for duplicates with different sequences.
                    self._log_duplicate_content_hash(df)

            # ---------- Step 15 & 16: deterministic sort + dedup (I2, I6) ----------
            if "uniprot_id" in df.columns:
                df = df.sort_values("uniprot_id").reset_index(drop=True)

            before_null_filter = len(df)
            df = df[df["uniprot_id"].notna() & (df["uniprot_id"] != "")].copy()
            dropped_null = before_null_filter - len(df)
            if dropped_null > 0:
                self._log_transformation(
                    "drop_null_uniprot_id", before_null_filter, len(df),
                    {"dropped": dropped_null},
                )

            # ---------- Step 17: drop duplicates (I2) ----------
            before_dedup = len(df)
            df = df.drop_duplicates(subset=["uniprot_id"], keep="first").copy()
            if before_dedup - len(df) > 0:
                self._log_transformation(
                    "dedup_uniprot_id", before_dedup, len(df),
                    {"removed": before_dedup - len(df)},
                )

            # ---------- Step 18: validate organism (S10, DQ5) ----------
            # SCI-FIX (organism normalization): UniProt's REST API returns
            # the organism field as ``"Homo sapiens (Human)"`` (with the
            # common name in parentheses), while the original strict check
            # required an exact match against ``"Homo sapiens"``. As a
            # result EVERY record was being flagged as "non-Homo sapiens"
            # -- a false-positive that polluted the audit log and risked
            # downstream code paths treating genuine human proteins as
            # non-human. The fix normalises the organism field by:
            #   1. Stripping the parenthetical common-name suffix.
            #   2. Whitespace-trimming.
            #   3. Falling back to "Homo sapiens" for blanks (since the
            #      query is organism_id:9606, we are confident these are
            #      human -- see S10 note below).
            # After normalisation, the strict "Homo sapiens" comparison
            # works correctly and genuine non-human records (if any slip
            # through) still raise the warning.
            if "organism" in df.columns:
                import re as _re
                def _normalise_organism(val: object) -> object:
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        return val
                    s = str(val).strip()
                    if not s:
                        return s
                    # Strip a trailing parenthetical, e.g.
                    # "Homo sapiens (Human)" -> "Homo sapiens"
                    s = _re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
                    return s
                df["organism"] = df["organism"].map(_normalise_organism)

                non_human = df[
                    df["organism"].notna()
                    & (df["organism"] != "")
                    & (df["organism"] != "Homo sapiens")
                ]
                if len(non_human) > 0:
                    logger.warning(
                        "[%s] %d records have non-Homo sapiens organism: %s",
                        self.source_name, len(non_human),
                        non_human["organism"].unique().tolist()[:5],
                    )
                # S10 -- fill missing organism with "Homo sapiens" only because
                # the query is organism_id:9606 (we are confident these are human).
                # The handle_missing_protein_fields(strict) call below will
                # additionally verify nothing fishy is going on.
                df.loc[
                    df["organism"].isna() | (df["organism"] == ""),
                    "organism",
                ] = "Homo sapiens"

            # ---------- Step 19: handle missing protein fields (S10) ----------
            # F2 -- Sequences MUST be stored in full (titin ~34 350 aa).
            # ``handle_missing_protein_fields`` truncates at ``_MAX_SEQUENCE_LENGTH``
            # (default 10 000), which would silently destroy long proteins.
            # The cleaning module's ``_MAX_SEQUENCE_LENGTH`` is module-level state
            # that can be modified by other tests (e.g.
            # ``test_cleaning_init_16_domains.py::test_lazy_import_does_not_load_submodules``
            # re-imports the module, resetting the constant).  We therefore
            # CANNOT rely on temporarily raising the cap -- the function may
            # still see the OLD module's value.
            #
            # Solution: do sequence missing-value handling ourselves (F2/S21)
            # and call ``handle_missing_protein_fields`` with the ``sequence``
            # column removed so it cannot truncate.  We then restore the
            # (already-validated) sequence column afterwards.
            #
            # v83 FORENSIC ROOT FIX (P1-7): the previous code restored the
            # sequence column via ``df["sequence"] = _sequence_col.values``
            # -- POSITIONAL assignment. If ``handle_missing_protein_fields``
            # DROPPED rows (e.g. missing organism in strict mode), the
            # ``.values`` array was LONGER than ``df``, and pandas silently
            # truncated it (or raised on length mismatch). If it REORDERED
            # rows (unlikely but possible), sequences got misaligned with
            # their uniprot_id -- a silent, life-safety-critical corruption.
            # ROOT FIX: restore by INDEX ALIGNMENT via ``.reindex(df.index)``.
            # If a row was dropped, its sequence is gone (correct). If rows
            # were reordered, sequences follow their original rows (correct).
            _sequence_col = None
            _sequence_index = None
            if "sequence" in df.columns:
                _sequence_col = df["sequence"].copy()
                _sequence_index = df.index.copy()
                df = df.drop(columns=["sequence"])

            # Call handle_missing_protein_fields for organism / gene_name /
            # function_desc handling only.
            df = handle_missing_protein_fields(
                df,
                organism_fill_mode="strict",
                # add_truncation_marker is irrelevant because sequence is gone.
                add_truncation_marker=False,
            )

            # Restore the sequence column by INDEX ALIGNMENT (v83 P1-7).
            # ``_sequence_col`` is a Series indexed by the ORIGINAL df.index.
            # ``.reindex(df.index)`` aligns by label -- rows that survived
            # ``handle_missing_protein_fields`` get their original sequence;
            # dropped rows are absent from df.index and contribute nothing.
            if _sequence_col is not None:
                df["sequence"] = _sequence_col.reindex(df.index)
            else:
                df["sequence"] = None

            # ---------- Step 20: ensure all required output columns ----------
            df = self._ensure_protein_columns(df)

            # ---------- Step 21: lineage columns (LIN2, LIN7, LIN8) ----------
            df["_source"] = "uniprot"
            df["_source_version"] = getattr(self, "source_version", None)
            df["_source_row_index"] = range(len(df))
            df["_protein_name_was_canonicalized"] = (
                df["protein_name"].fillna("") != df["protein_name_canonical"].fillna("")
            )
            if "function_desc" in df.columns:
                df["_function_desc_was_cleaned"] = df["function_desc"].notna()
            else:
                df["_function_desc_was_cleaned"] = False
            if "all_string_ids" in df.columns:
                df["_string_id_is_subset"] = (
                    df["all_string_ids"].notna()
                    & df["all_string_ids"].astype(str).str.contains(";", na=False)
                )
            else:
                df["_string_id_is_subset"] = False

            # ---------- Step 22: DQ metrics (DQ20, L23) ----------
            dq_metrics = self._compute_dq_metrics(df)

            # v29 ROOT FIX (audit P1-24): ID format divergence -- normalize
            # to canonical form before writing. UniProt accessions and gene
            # symbols are uppercased + stripped. This guarantees downstream
            # joins against STRING (uniprot_id), DisGeNET (gene_symbol),
            # OMIM (gene_symbol), and DrugBank interactions (uniprot_id)
            # succeed regardless of which source wrote the value. Some
            # UniProt TSV fields ship lowercase accessions for historical
            # display reasons; without this normalization, a PPI edge from
            # STRING (``"P23219"``) would NOT join with a protein from
            # UniProt (``"p23219"``).
            if len(df) > 0:
                if "uniprot_id" in df.columns:
                    df["uniprot_id"] = df["uniprot_id"].apply(
                        lambda x: normalize_uniprot_id(x)
                        if pd.notna(x) and x != "" else x
                    )
                if "gene_symbol" in df.columns:
                    df["gene_symbol"] = df["gene_symbol"].apply(
                        lambda x: normalize_gene_symbol(x)
                        if pd.notna(x) and x != "" else x
                    )

            # ---------- Step 23: sanitize for CSV (SEC4, C27) ----------
            # v83 FORENSIC ROOT FIX (P1-12): the previous code called
            # ``_sanitize_dataframe_for_csv(df)`` on the DataFrame that is
            # BOTH written to CSV AND loaded to DB. The sanitizer prepends
            # ``'`` to any string starting with ``=``, ``+``, ``-``, ``@``,
            # ``\t``, ``\r`` -- but legitimate UniProt protein names CAN
            # start with ``-`` or ``+`` (e.g. chemokine fragment names,
            # charge-tagged peptide names), and ``function_desc`` entries
            # can start with ``-`` (e.g. "-Catalytic activity:..."). The
            # ``'`` prefix was written to the CSV AND loaded to the DB,
            # corrupting ``protein_name`` / ``protein_name_canonical`` /
            # ``function_desc``.
            # ROOT FIX: do NOT mutate the in-memory DataFrame. Instead,
            # override ``_persist_cleaned_data`` to sanitize a COPY only
            # at CSV-write time. The DB load receives the unsanitized
            # (correct) DataFrame. The CSV gets the sanitized (Excel-safe)
            # copy. The ``'`` prefix never reaches the DB.
            # (Sanitization moved to ``_persist_cleaned_data`` override below.)

            # Final null-count log (L19) -- raw vs cleaned ratio.
            self._log_null_counts(df, stage="clean")
            logger.info(
                "[%s] Clean complete: %d raw -> %d cleaned (ratio: %.4f, "
                "DQ score: %.4f)",
                self.source_name, raw_count, len(df),
                (len(df) / raw_count) if raw_count > 0 else 0.0,
                dq_metrics.get("quality_score", 0.0),
                extra=self._log_context(),
            )

            return df

    # ---------------------------------------------------------------------
    # _extract_canonical_name() -- nested parens, ECO, EC numbers (S4, S14, S15, C14, C15)
    # ---------------------------------------------------------------------
    def _extract_canonical_name(self, protein_name: Optional[str]) -> Optional[str]:
        """Extract the canonical protein name (S4, S14, S15, C14, C15).

        Strips:
        * ``{ECO:...}`` evidence tags (anywhere in the string).
        * Parenthetical content (handles nested parentheses via manual scan).
        * Trailing EC numbers (e.g. ``"Catalase EC 1.11.1.6"``).

        Parameters
        ----------
        protein_name : str | None
            Raw protein name from UniProt.

        Returns
        -------
        str | None
            Canonical name, or *None* if input is *None*, empty, or
            contains only parenthetical content (S15).
        """
        if not protein_name or not isinstance(protein_name, str):
            return None
        if not protein_name.strip():
            return None

        # S14 -- strip {ECO:...} tags first (before paren removal, so the
        # paren-stripping regex doesn't get confused by braces).
        cleaned = _ECO_TAG_RE.sub("", protein_name).strip()
        if not cleaned:
            return None

        # S4 -- strip nested parentheses via manual scan (regex can't handle
        # arbitrary nesting).  We keep everything before the first UNMATCHED
        # open paren.
        result_chars: list[str] = []
        depth = 0
        for ch in cleaned:
            if ch == "(":
                depth += 1
                if depth == 1:
                    # First open paren -- stop appending (but continue scanning
                    # to track depth so nested closes don't end the loop early).
                    continue
            elif ch == ")":
                if depth > 0:
                    depth -= 1
                continue
            elif depth == 0:
                result_chars.append(ch)

        canonical = "".join(result_chars).strip()

        # C15 -- strip trailing EC number (strict format: EC + 2-4 dotted ints).
        canonical = _EC_NUMBER_RE.sub("", canonical).strip()

        # S15 -- if everything was in parens, return None (not "").
        if not canonical:
            return None

        return canonical

    # ---------------------------------------------------------------------
    # _clean_function_desc() -- case-insensitive, earliest marker (S5, S6, S7, S16, S17, C16, C17, C18)
    # ---------------------------------------------------------------------
    def _clean_function_desc(self, desc: Optional[str]) -> Optional[str]:
        """Strip ``FUNCTION:`` prefix, sub-section markers, and ECO tags (S5-S7).

        UniProt's ``Function [CC]`` field looks like::

            "FUNCTION: Catalyzes the reaction {ECO:0000256|HAMAP-Rule:MF_00234}.
             CATALYTIC ACTIVITY: ... SUBUNIT: ..."

        We want only the function prose.  Steps:
        1. Strip the leading ``FUNCTION:`` / ``Function:`` prefix (S16 --
           case-insensitive).
        2. Find the EARLIEST sub-section marker (S6) and truncate there.
        3. Remove ALL ``{ECO:...}`` evidence tags (S7, C18 -- both inline
           and trailing).

        Parameters
        ----------
        desc : str | None
            Raw function description.

        Returns
        -------
        str | None
            Cleaned description, or *None* if input is empty / all
            stripped (S15).
        """
        if not desc or not isinstance(desc, str):
            return None
        if not desc.strip():
            return None

        cleaned = desc.strip()

        # S16 -- case-insensitive FUNCTION: prefix strip (only the first
        # occurrence; S17).
        for prefix in ("FUNCTION: ", "Function: ", "FUNCTION:", "Function:"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break

        # S6 -- find the EARLIEST sub-section marker (any order) and truncate.
        earliest_idx = len(cleaned)
        for marker in _SUBSECTION_MARKERS:
            idx = cleaned.find(marker)
            # S5 -- idx >= 0 (NOT idx > 0) so a marker at position 0 also matches.
            if 0 <= idx < earliest_idx:
                earliest_idx = idx
        if earliest_idx < len(cleaned):
            cleaned = cleaned[:earliest_idx].strip()

        # S7 / C18 -- remove ALL {ECO:...} evidence tags (inline + trailing).
        cleaned = _ECO_TAG_RE.sub("", cleaned).strip()

        return cleaned if cleaned else None

    # ---------------------------------------------------------------------
    # _extract_string_id() -- first valid STRING ID (S8, S9, C19)
    # ---------------------------------------------------------------------
    def _extract_string_id(self, xref: Optional[str]) -> Optional[str]:
        """Extract the first valid STRING ID from a cross-reference field (S8, S9).

        UniProt STRING xrefs look like::

            "9606.ENSP00000357607; 9606.ENSP00000412345;"

        Multiple IDs may be present (one per isoform).  This function
        returns the FIRST valid one.  All IDs are also stored in the
        ``all_string_ids`` column via ``_extract_all_string_ids()``.

        Parameters
        ----------
        xref : str | None
            Raw cross-reference string from UniProt.

        Returns
        -------
        str | None
            First valid STRING ID, or *None*.
        """
        if not xref or not isinstance(xref, str):
            return None

        # C19 -- iterate through all parts (handles leading semicolon).
        parts = [p.strip() for p in xref.split(";") if p.strip()]
        if not parts:
            return None

        # S9 -- validate format.
        valid_ids = [p for p in parts if _STRING_ID_RE.match(p)]
        if not valid_ids:
            logger.debug(
                "[%s] No valid STRING IDs found in xref: %s",
                self.source_name, xref[:100],
            )
            return None

        # S8 -- log when multiple IDs are present (some are discarded from
        # the primary column but kept in all_string_ids).
        if len(valid_ids) > 1:
            logger.debug(
                "[%s] Multiple STRING IDs found: %s. Using first: %s",
                self.source_name, valid_ids, valid_ids[0],
            )

        return valid_ids[0]

    # ---------------------------------------------------------------------
    # _extract_all_string_ids() -- semicolon-joined list (S8)
    # ---------------------------------------------------------------------
    @staticmethod
    def _extract_all_string_ids(xref: Optional[str]) -> Optional[str]:
        """Return a semicolon-joined list of ALL valid STRING IDs (S8).

        Parameters
        ----------
        xref : str | None
            Raw cross-reference string from UniProt.

        Returns
        -------
        str | None
            ``"9606.ENSP00000357607;9606.ENSP00000412345"`` or *None*.
        """
        if not xref or not isinstance(xref, str):
            return None
        parts = [p.strip() for p in xref.split(";") if p.strip()]
        valid = [p for p in parts if _STRING_ID_RE.match(p)]
        return ";".join(valid) if valid else None

    # ---------------------------------------------------------------------
    # _validate_gene_symbol() (S3, DQ9, DQ25)
    # ---------------------------------------------------------------------
    @staticmethod
    def _validate_gene_symbol(value: Any) -> Optional[str]:
        """Validate and normalize a gene symbol (S3, DQ9, DQ25).

        Strips whitespace and uppercases.  Returns *None* if the value
        is empty or does not match the HGNC pattern
        ``^[A-Z][A-Z0-9\\-]{0,49}$``.

        Parameters
        ----------
        value : Any
            Raw gene symbol value (may be NaN, str, etc.).

        Returns
        -------
        str | None
            Validated gene symbol, or *None*.
        """
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if not value:
            return None
        # Some UniProt gene names include synonyms separated by spaces;
        # the canonical symbol is the first token.
        if " " in value:
            value = value.split()[0]
        value = value.upper()
        if not _HGNC_SYMBOL_RE.match(value):
            return None
        return value

    # ---------------------------------------------------------------------
    # _validate_sequence() (S21, DQ10, C24, C57)
    # ---------------------------------------------------------------------
    def _validate_sequence(self, s: Any) -> Optional[str]:
        """Validate an amino-acid sequence (S21, DQ10, C24, C57).

        Non-string values (NaN, float, bytes) are converted to *None*.
        Strings containing invalid characters are logged and set to
        *None* (do not raise -- we want the pipeline to continue).

        Parameters
        ----------
        s : Any
            Raw sequence value.

        Returns
        -------
        str | None
            Validated sequence, or *None*.
        """
        if not isinstance(s, str):
            return None
        if not s:
            return None
        if not _VALID_AA_PATTERN.match(s):
            logger.warning(
                "[%s] Invalid sequence characters detected (length=%d), "
                "setting to None",
                self.source_name, len(s),
            )
            return None
        return s

    # ---------------------------------------------------------------------
    # _ensure_protein_columns() (F4, C48, DQ18)
    # ---------------------------------------------------------------------
    @staticmethod
    def _ensure_protein_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required output columns exist with proper defaults (F4, C48, DQ18).

        After Fix F4, the column set includes ``protein_name_canonical``
        and ``length``.  Defaults are *None* (not empty string) for
        consistency (C50).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to ensure columns on.

        Returns
        -------
        pd.DataFrame
            DataFrame with all required columns (may be the same object).
        """
        required_defaults: dict[str, Any] = {
            "uniprot_id": "",
            "gene_symbol": None,
            "gene_name": None,
            "protein_name": None,
            "protein_name_canonical": None,
            "organism": None,             # S10: default None, NOT "Homo sapiens"
            "length": None,               # C48 / DQ18
            "sequence": None,
            "function_desc": None,
            "string_id": None,
            "all_string_ids": None,
        }
        for col, default in required_defaults.items():
            if col not in df.columns:
                df[col] = default
        return df

    # ---------------------------------------------------------------------
    # _log_null_counts() (DQ3, L19)
    # ---------------------------------------------------------------------
    def _log_null_counts(self, df: pd.DataFrame, stage: str = "clean") -> None:
        """Log NULL and empty-string counts for all columns (DQ3, L19).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to profile.
        stage : str
            Pipeline stage for log context (``"raw"``, ``"clean"``, ``"load"``).
        """
        if df is None or len(df) == 0:
            logger.info("[%s] %s-stage: empty DataFrame", self.source_name, stage)
            return
        null_counts = df.isnull().sum()
        # Also count empty strings for object columns.
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                empty_count = int((df[col] == "").sum())
                if empty_count > 0:
                    null_counts[f"{col}(empty)"] = empty_count
            except (TypeError, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
                pass
        non_zero = null_counts[null_counts > 0]
        if len(non_zero) > 0:
            logger.info(
                "[%s] %s-stage NULL/empty counts: %s",
                self.source_name, stage, non_zero.to_dict(),
            )
        else:
            logger.info(
                "[%s] %s-stage: zero NULLs across all columns",
                self.source_name, stage,
            )

    # ---------------------------------------------------------------------
    # _log_missing_columns() (L6)
    # ---------------------------------------------------------------------
    def _log_missing_columns(self, df: pd.DataFrame) -> None:
        """Warn for ALL missing important columns, not just gene_symbol (L6).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame after the rename step.
        """
        important_columns: list[tuple[str, str]] = [
            ("uniprot_id", "critical"),
            ("gene_symbol", "important"),
            ("protein_name", "important"),
            ("sequence", "important"),
            ("organism", "important"),
            ("function_desc", "useful"),
            ("string_xref", "useful"),
            ("length", "useful"),
        ]
        for col_name, importance in important_columns:
            if col_name not in df.columns:
                level = (
                    logging.ERROR if importance == "critical"
                    else logging.WARNING if importance == "important"
                    else logging.INFO
                )
                logger.log(
                    level,
                    "[%s] Column '%s' not found in UniProt TSV (importance: %s)",
                    self.source_name, col_name, importance,
                )

    # ---------------------------------------------------------------------
    # _log_duplicate_content_hash() (I14)
    # ---------------------------------------------------------------------
    def _log_duplicate_content_hash(self, df: pd.DataFrame) -> None:
        """Log a content hash for duplicate uniprot_ids with different sequences (I14).

        If the same ``uniprot_id`` appears multiple times with DIFFERENT
        sequences, that is a real data-integrity problem -- we want to
        know about it.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to check.
        """
        if "uniprot_id" not in df.columns or "sequence" not in df.columns:
            return
        dup_mask = df["uniprot_id"].duplicated(keep=False)
        if not dup_mask.any():
            return
        dup_df = df[dup_mask]
        for uid in dup_df["uniprot_id"].unique():
            subset = dup_df[dup_df["uniprot_id"] == uid]
            hashes = subset["sequence"].apply(
                lambda s: hashlib.md5(
                    str(s).encode("utf-8", errors="replace")
                ).hexdigest() if pd.notna(s) else "NULL"
            )
            if hashes.nunique() > 1:
                logger.warning(
                    "[%s] Duplicate uniprot_id %r with DIFFERENT sequences "
                    "(content hash mismatch): %s",
                    self.source_name, uid, hashes.tolist(),
                )

    # ---------------------------------------------------------------------
    # _compute_dq_metrics() (DQ20, L23)
    # ---------------------------------------------------------------------
    def _compute_dq_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute data-quality metrics for the cleaned DataFrame (DQ20, L23).

        Returns a dict with completeness, validity, uniqueness, and
        consistency metrics for downstream monitoring.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned DataFrame.

        Returns
        -------
        dict[str, Any]
            Metrics dict with a ``quality_score`` field in [0.0, 1.0].
        """
        metrics: dict[str, Any] = {}
        total = len(df)
        metrics["total_records"] = total

        if total == 0:
            metrics["quality_score"] = 0.0
            return metrics

        # Completeness: non-null, non-empty fraction per column.
        for col in ("uniprot_id", "gene_symbol", "sequence", "protein_name"):
            if col in df.columns:
                valid = df[col].notna() & (df[col] != "")
                metrics[f"completeness_{col}"] = float(valid.sum()) / total

        # Validity: uniprot_id pattern compliance.
        # v83 FORENSIC ROOT FIX (P2-11): the previous code computed
        # ``validity_uniprot_id`` on the POST-quarantine DataFrame (after
        # invalid records were removed at Step 10), so the metric was
        # ALWAYS 1.0 -- meaningless. ROOT FIX: use the PRE-quarantine raw
        # counts (captured at Step 10) to compute a MEANINGFUL validity
        # ratio. The post-quarantine metric is retained as
        # ``validity_uniprot_id_post_quarantine`` for backward compat
        # (it's always 1.0, but downstream dashboards may reference it).
        if "uniprot_id" in df.columns:
            valid_ids = df["uniprot_id"].apply(
                lambda x: bool(_UNIPROT_ACCESSION_RE.match(x))
                if pd.notna(x) and isinstance(x, str) else False
            )
            metrics["validity_uniprot_id_post_quarantine"] = float(valid_ids.sum()) / total
            # v83 P2-11: meaningful RAW validity (pre-quarantine).
            raw_total = getattr(self, "_uniprot_id_raw_total", total)
            raw_invalid = getattr(self, "_uniprot_id_raw_invalid", 0)
            if raw_total > 0:
                metrics["validity_uniprot_id_raw"] = (
                    float(raw_total - raw_invalid) / float(raw_total)
                )
            else:
                metrics["validity_uniprot_id_raw"] = 1.0
            # Backward-compat alias (always 1.0 post-quarantine).
            metrics["validity_uniprot_id"] = metrics["validity_uniprot_id_post_quarantine"]

        # Uniqueness.
        if "uniprot_id" in df.columns:
            metrics["uniqueness_uniprot_id"] = 1.0 - (
                float(df["uniprot_id"].duplicated().sum()) / total
            )

        # Consistency: length vs sequence.
        if "length" in df.columns and "sequence" in df.columns:
            consistent = df.apply(
                lambda r: (
                    pd.isna(r["length"])
                    or not isinstance(r["sequence"], str)
                    or int(r["length"]) == len(r["sequence"])
                ),
                axis=1,
            )
            metrics["consistency_length_sequence"] = float(consistent.sum()) / total

        # Overall quality score = mean of the four core dimensions.
        score_components = [
            metrics.get("completeness_uniprot_id", 1.0),
            metrics.get("validity_uniprot_id", 1.0),
            metrics.get("uniqueness_uniprot_id", 1.0),
            metrics.get("consistency_length_sequence", 1.0),
        ]
        metrics["quality_score"] = sum(score_components) / len(score_components)

        logger.info(
            "[%s] Data quality metrics: %s",
            self.source_name,
            {k: f"{v:.4f}" if isinstance(v, float) else v
             for k, v in metrics.items()},
        )
        return metrics

    # ---------------------------------------------------------------------
    # _write_provenance_sidecar() (S25, LIN3, LIN9-LIN20, SEC16, SEC20, COMP1, COMP4)
    # ---------------------------------------------------------------------
    def _write_provenance_sidecar(
        self,
        raw_path: Path,
        cleaned_path: Path,
        record_count: int,
    ) -> None:
        """Write a provenance metadata sidecar JSON file (S25, LIN3-LIN20).

        The sidecar is named ``<cleaned_filename>.provenance.json`` and
        records the full provenance of the cleaned dataset: pipeline
        name and version, UniProt release, input/output SHA-256
        checksums, record counts, timestamp, run_id, correlation_id,
        triggered_by (FDA 21 CFR Part 11 -- COMP1), and the query /
        fields used.

        Parameters
        ----------
        raw_path : Path
            Path to the raw input file.
        cleaned_path : Path
            Path to the cleaned output file.
        record_count : int
            Number of records in the cleaned output.
        """
        def _sha256(p: Path) -> Optional[str]:
            if not p.exists():
                return None
            try:
                return self._compute_sha256(p)
            except OSError:
                return None

        provenance = {
            "pipeline": self.source_name,
            "pipeline_version": __version__,
            "schema_version": "v1",
            "run_id": getattr(self, "run_id", None),
            "correlation_id": getattr(self, "correlation_id", None),
            "triggered_by": getattr(self, "triggered_by", None),  # SEC20 / COMP1
            "uniprot_release": getattr(self, "source_version", None) or UNIPROT_RELEASE,
            # P1-016 ROOT FIX (Team-2): add a "release fingerprint" field
            # so downstream consumers (KG build, Graph Transformer, RL
            # ranker) can verify that two runs used the SAME UniProt
            # release. The fingerprint is ``release||raw_sha256`` -- a
            # collision-free identifier for "the exact bytes used". If
            # two runs have the same fingerprint, they are byte-identical
            # at the UniProt level. If they differ, downstream phases
            # can detect the drift and invalidate cached embeddings.
            "release_fingerprint": (
                f"{getattr(self, 'source_version', None) or UNIPROT_RELEASE}"
                f"||{_sha256(raw_path) or 'unknown'}"
            ),
            "release_is_pinned": (
                (getattr(self, "source_version", None) or UNIPROT_RELEASE)
                != "current_release"
            ),
            "query": self.uniprot_query,
            "fields": list(self.uniprot_fields),
            "raw_file": str(raw_path),
            "raw_sha256": _sha256(raw_path),
            "cleaned_file": str(cleaned_path),
            "cleaned_sha256": _sha256(cleaned_path),
            "record_count": record_count,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed": getattr(self, "seed", None),
            "as_of_date": str(getattr(self, "as_of_date", None)),  # LIN17
            "freeze_version": getattr(self, "freeze_version", None),  # LIN18
            "snapshot_tag": getattr(self, "snapshot_tag", None),     # LIN19
            "environment": getattr(self, "environment", "development"),
        }

        sidecar_path = cleaned_path.with_suffix(
            cleaned_path.suffix + ".provenance.json"
        )
        try:
            sidecar_path.write_text(
                json.dumps(provenance, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            self._set_secure_permissions(sidecar_path)
            logger.info(
                "[%s] Wrote provenance sidecar: %s",
                self.source_name, sidecar_path,
            )
        except OSError as exc:
            logger.warning(
                "[%s] Could not write provenance sidecar: %s",
                self.source_name, exc,
            )

    # ---------------------------------------------------------------------
    # load() -- accepts session=, returns LoadResult (F1, A1, D2-1, D2-4, C22, C23)
    # ---------------------------------------------------------------------
    def load(
        self,
        df: pd.DataFrame,
        *,
        session: Optional[Session] = None,
    ) -> LoadResult:
        """Bulk upsert cleaned protein data into the database (F1, D2-1).

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned protein DataFrame from ``clean()``.
        session : Session | None
            Optional SQLAlchemy session.  If *None*, a new session is
            created via ``get_db_session()``.  When the caller
            (``BasePipeline.run()``) provides a session, it is reused
            so the load participates in the caller's transaction
            boundary (A11, I11).

        Returns
        -------
        LoadResult
            Structured result with ``rows_inserted``, ``rows_updated``,
            ``rows_skipped``, ``rows_failed`` counts (C22, C23).

        Raises
        ------
        ValueError
            If required column ``uniprot_id`` is missing from *df* (C20).
        """
        # C20 -- validate that required columns are present.
        missing_required = [c for c in _CRITICAL_COLUMNS if c not in df.columns]
        if missing_required:
            raise ValueError(
                f"Cannot load proteins: required columns missing from "
                f"DataFrame: {missing_required}. Available columns: "
                f"{list(df.columns)}"
            )

        # Build the load DataFrame -- only include columns that exist on the
        # Protein model (D2-9, INT17, DQ17).  This prevents IntegrityError
        # from extra columns like `length` or `protein_name_canonical`
        # that are in the cleaned CSV (for schema compliance / downstream
        # use) but not on the DB table.
        load_columns = self._get_load_columns()
        load_df = df[[c for c in load_columns if c in df.columns]].copy()

        own_session = session is None
        # v29 ROOT FIX (audit P1-6): the previous code did
        #   session = self._db_session_factory()
        # which returns a context manager (get_db_session is a
        # @contextmanager). The context manager was NEVER entered --
        # ``session`` was the context manager, not the Session, so
        # every subsequent session.add() / session.commit() crashed
        # with AttributeError when load() was called standalone.
        # Also, the finally block only called session.close() -- it
        # never called __exit__(), so the commit never happened and
        # ALL loaded data was silently rolled back when load() ran
        # standalone.
        _session_cm = None
        if own_session:
            try:
                _session_cm = self._db_session_factory()  # C21 -- factory
                session = _session_cm.__enter__()  # v29: capture the Session
            except (OperationalError, IntegrityError, OSError) as exc:  # v85 FORENSIC ROOT FIX (BUG #51)
                logger.error(
                    "[%s] Failed to create DB session: %s",
                    self.source_name, exc,
                    exc_info=getattr(self, "log_exc_info", True),
                )
                raise

        try:
            with self._timed_operation("load"):
                result = self._loader(session, load_df)

                # C22 -- convert UpsertResult -> LoadResult.
                # The loader may be a real callable returning UpsertResult,
                # or a MagicMock (in tests).  Handle both.
                if isinstance(result, UpsertResult):
                    load_result = LoadResult(
                        rows_inserted=result.inserted,
                        rows_updated=result.updated,
                        rows_skipped=result.quarantined,
                        rows_failed=result.failed,
                    )
                    logger.info(
                        "[%s] Upserted proteins: total=%d inserted=%d "
                        "updated=%d quarantined=%d failed=%d",
                        self.source_name,
                        result.total_input, result.inserted,
                        result.updated, result.quarantined, result.failed,
                        extra=self._log_context(),
                    )
                    return load_result

                # Fallback for int return (backward-compat) or mocks.
                try:
                    count = int(result)
                except (TypeError, ValueError):
                    count = 0
                logger.info(
                    "[%s] Loaded %d proteins (legacy return type)",
                    self.source_name, count,
                )
                return LoadResult(rows_inserted=count)

        except (OSError, RuntimeError, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
            if own_session and session is not None:
                try:
                    session.rollback()
                except (OSError, RuntimeError, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
                    pass
            raise
        finally:
            # v29 ROOT FIX (audit P1-6): call __exit__ on the context
            # manager so it commits (on success) or rolls back (on
            # error). The previous code only called session.close(),
            # which silently rolled back ALL loaded data when load()
            # ran standalone.
            if own_session and _session_cm is not None:
                import sys as _sys
                _exc_info = _sys.exc_info()
                try:
                    _session_cm.__exit__(*_exc_info)
                except (OSError, RuntimeError, ValueError):  # noqa: BLE001 -- cleanup must not mask  # v85 FORENSIC ROOT FIX (BUG #51)
                    pass

    # ---------------------------------------------------------------------
    # _get_load_columns() (D2-9, INT17, DQ17)
    # ---------------------------------------------------------------------
    def _get_load_columns(self) -> list[str]:
        """Get the columns to load into the proteins table (D2-9, INT17).

        Derived from the ``Protein`` model's column list, intersected
        with the columns the pipeline produces.  Falls back to a
        hardcoded list if the model can't be imported.

        Returns
        -------
        list[str]
            Column names to send to ``bulk_upsert_proteins``.
        """
        try:
            from database.models import Protein
            model_cols = [c.name for c in Protein.__table__.columns]
            # Filter out SQLAlchemy-internal columns and mixin-managed columns
            # that the loader will set itself (id, created_at, updated_at,
            # is_deleted, deleted_at).
            skip = {"id", "created_at", "updated_at", "is_deleted", "deleted_at"}
            return [c for c in model_cols if c not in skip]
        except ImportError:
            # Fallback -- keep in sync with database/models.py.
            # v83 FORENSIC ROOT FIX (P1-11): the previous fallback list was
            # missing ``length``, ``protein_name_canonical``, and
            # ``all_string_ids`` -- three columns that EXPECTED_OUTPUT_COLUMNS
            # (line ~319) guarantees. If the fallback fired (e.g.
            # database.models not importable in a test environment), these
            # columns were SILENTLY DROPPED from the load DataFrame. The
            # ``protein_name_canonical`` column is the F4 fix for the
            # "gene_name stored a protein name" bug -- dropping it loses the
            # canonical name and re-introduces the corruption. ROOT FIX:
            # add the three missing columns to the fallback list so the
            # fallback matches the real model's column set.
            return [
                "uniprot_id", "gene_name", "gene_symbol", "protein_name",
                "protein_name_canonical", "organism", "sequence", "length",
                "function_desc", "string_id", "all_string_ids",
            ]

    # ---------------------------------------------------------------------
    # teardown() (A8)
    # ---------------------------------------------------------------------
    def teardown(self) -> None:
        """Clean up resources after a pipeline run (A8).

        Closes the HTTP session if it was created and flushes any
        pending dead-letter-queue records to disk.
        """
        try:
            if self._http_session is not None:
                try:
                    self._http_session.close()
                except (OSError, RuntimeError, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
                    pass
                self._http_session = None
        finally:
            # R4 / DQ19 -- flush dead-letter queue to disk.
            try:
                self._flush_dead_letter_queue()
            except (OSError, RuntimeError, ValueError) as exc:  # v85 FORENSIC ROOT FIX (BUG #51)
                logger.debug(
                    "[%s] DLQ flush failed in teardown: %s",
                    self.source_name, exc,
                )
            # Call super.teardown() if it exists (it closes the base
            # class's HTTP session, etc.).
            try:
                super().teardown()
            except (OSError, RuntimeError, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
                pass
            logger.info("[%s] teardown complete", self.source_name)

    # ---------------------------------------------------------------------
    # _sanitize_csv_value() / _sanitize_dataframe_for_csv() (SEC4, C27)
    # ---------------------------------------------------------------------
    @classmethod
    def _sanitize_csv_value(cls, value: Any) -> Any:
        """Sanitize a single value to prevent CSV formula injection (SEC4, C27).

        If *value* is a non-empty string starting with a dangerous prefix
        (``=``, ``+``, ``-``, ``@``, ``\\t``, ``\\r``), prepend a single
        quote to neutralize the formula.

        Parameters
        ----------
        value : Any
            Value to sanitize.

        Returns
        -------
        Any
            Sanitized value.
        """
        if isinstance(value, str) and value:
            if value.startswith(_CSV_DANGEROUS_PREFIXES):
                return "'" + value
        return value

    def _sanitize_dataframe_for_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply CSV formula injection prevention to all string columns (SEC4).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to sanitize.

        Returns
        -------
        pd.DataFrame
            Sanitized DataFrame (a copy).
        """
        if df is None or len(df) == 0:
            return df
        df = df.copy()
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].apply(self._sanitize_csv_value)
        return df

    # ---------------------------------------------------------------------
    # _persist_cleaned_data() override (v83 P1-12 root fix)
    # ---------------------------------------------------------------------
    def _persist_cleaned_data(self, df: pd.DataFrame) -> Path:
        """Override the base class CSV writer to apply CSV-only sanitization.

        v83 FORENSIC ROOT FIX (P1-12): the base class writes the DataFrame
        to CSV as-is. The UniProt pipeline needs CSV formula-injection
        protection (SEC4, C27) for Excel safety, but the previous code
        mutated the IN-MEMORY DataFrame (in ``clean()``) before it was
        loaded to the DB -- corrupting ``protein_name`` /
        ``protein_name_canonical`` / ``function_desc`` with leading ``'``
        for any value starting with ``-`` / ``+`` / ``@``.

        ROOT FIX: sanitize a COPY of the DataFrame ONLY at CSV-write time.
        The caller's ``df`` (returned from ``clean()`` and later passed to
        ``load()``) is NEVER mutated -- the DB receives unsanitized (correct)
        data. The CSV receives the sanitized (Excel-safe) copy. The ``'``
        prefix never reaches the DB.
        """
        sanitized_df = self._sanitize_dataframe_for_csv(df)
        return super()._persist_cleaned_data(sanitized_df)

    # ---------------------------------------------------------------------
    # _log_transformation() (LIN1, LIN5, L9)
    # ---------------------------------------------------------------------
    def _log_transformation(
        self,
        transformation: str,
        record_count_before: int,
        record_count_after: int,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a transformation step for data lineage (LIN1, LIN5, L9).

        Parameters
        ----------
        transformation : str
            Name of the transformation (e.g. ``"dedup_uniprot_id"``).
        record_count_before : int
            Number of records before the transformation.
        record_count_after : int
            Number of records after the transformation.
        details : dict | None
            Additional details about the transformation.
        """
        logger.info(
            "[%s] Transformation: %s | records: %d -> %d | delta: %d",
            self.source_name,
            transformation,
            record_count_before,
            record_count_after,
            record_count_after - record_count_before,
            extra={
                **self._log_context(),
                "transformation": transformation,
                "record_count_before": record_count_before,
                "record_count_after": record_count_after,
                **(details or {}),
            },
        )

    # ---------------------------------------------------------------------
    # _log_context() (L2, L3, L4)
    # ---------------------------------------------------------------------
    def _log_context(self) -> dict[str, Any]:
        """Return structured logging context for this pipeline run (L2, L3, L4).

        Returns
        -------
        dict[str, Any]
            Context dict with pipeline name, run_id, correlation_id,
            triggered_by, and environment.
        """
        return {
            "pipeline": self.source_name,
            "run_id": getattr(self, "run_id", None),
            "correlation_id": getattr(self, "correlation_id", None),
            "triggered_by": getattr(self, "triggered_by", None),
            "environment": getattr(self, "environment", "development"),
        }

    # ---------------------------------------------------------------------
    # _timed_operation() (L8, L16, L17, L18)
    # ---------------------------------------------------------------------
    @contextlib.contextmanager
    def _timed_operation(self, operation: str) -> Iterator[None]:
        """Context manager that logs the duration of an operation (L8, L16-L18).

        Parameters
        ----------
        operation : str
            Name of the operation (e.g. ``"download"``, ``"clean"``, ``"load"``).
        """
        start = time.monotonic()
        logger.info(
            "[%s] Starting %s",
            self.source_name, operation,
            extra=self._log_context(),
        )
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            logger.info(
                "[%s] Finished %s in %.2fs",
                self.source_name, operation, elapsed,
                extra={**self._log_context(), "duration_seconds": elapsed,
                       "operation": operation},
            )

    # ---------------------------------------------------------------------
    # _secure_delete() (SEC17)
    # ---------------------------------------------------------------------
    def _secure_delete(self, path: Path) -> None:
        """Securely delete a file by overwriting before removal (SEC17).

        Overwrites the file with zeros, fsyncs, then unlinks.  Falls
        back to a regular ``unlink()`` if the secure overwrite fails.

        Parameters
        ----------
        path : Path
            File to delete.
        """
        if not path.exists():
            return
        try:
            size = path.stat().st_size
            with open(path, "wb") as f:
                f.write(b"\x00" * min(size, 10 * 1024 * 1024))  # cap at 10 MB
                f.flush()
                os.fsync(f.fileno())
            path.unlink()
        except OSError as exc:
            logger.debug(
                "[%s] Secure delete failed for %s: %s; falling back to unlink",
                self.source_name, path, exc,
            )
            try:
                path.unlink()
            except OSError:
                pass

    # ---------------------------------------------------------------------
    # _set_secure_permissions() (SEC10, SEC14)
    # ---------------------------------------------------------------------
    def _set_secure_permissions(self, path: Path) -> None:
        """Set file permissions to owner-only read/write (SEC10, SEC14).

        Parameters
        ----------
        path : Path
            File to secure.
        """
        try:
            os.chmod(path, self._SECURE_FILE_MODE)
        except OSError:
            # On Windows or read-only filesystems, chmod may fail -- that's OK.
            logger.debug(
                "[%s] Could not set permissions on %s",
                self.source_name, path,
            )

    # ---------------------------------------------------------------------
    # _quarantine_record() (DQ19, R4)
    # ---------------------------------------------------------------------
    def _quarantine_record(self, record: dict[str, Any], reason: str) -> None:
        """Add a record to the dead-letter queue with a rejection reason (DQ19, R4).

        Parameters
        ----------
        record : dict
            The rejected record.
        reason : str
            Why the record was rejected.
        """
        if not hasattr(self, "dead_letter_queue"):
            self.dead_letter_queue: list[dict[str, Any]] = []
        record_copy = dict(record)
        record_copy["_rejection_reason"] = reason
        record_copy["_rejected_at"] = datetime.now(timezone.utc).isoformat()
        record_copy["_pipeline"] = self.source_name
        self.dead_letter_queue.append(record_copy)
        logger.debug(
            "[%s] Quarantined record: %s (reason: %s)",
            self.source_name,
            record_copy.get("uniprot_id", "?"),
            reason,
        )

    # ---------------------------------------------------------------------
    # _flush_dead_letter_queue() (DQ19, R4, L20)
    # ---------------------------------------------------------------------
    def _flush_dead_letter_queue(self) -> None:
        """Write the dead-letter queue to disk as JSONL (DQ19, R4, L20).

        File: ``<effective_raw_dir>/dead_letter_queue.jsonl``.
        Each line is a JSON object representing one rejected record.
        """
        queue = getattr(self, "dead_letter_queue", None)
        if not queue:
            return
        dlq_path = self.effective_raw_dir / "dead_letter_queue.jsonl"
        try:
            with open(dlq_path, "a", encoding="utf-8") as f:
                for record in queue:
                    f.write(json.dumps(record, default=str) + "\n")
            logger.info(
                "[%s] Flushed %d records to dead-letter queue: %s",
                self.source_name, len(queue), dlq_path,
            )
            queue.clear()
        except OSError as exc:
            logger.warning(
                "[%s] Could not flush dead-letter queue: %s",
                self.source_name, exc,
            )

    # ---------------------------------------------------------------------
    # _redact_log_message() (SEC13, SEC11)
    # ---------------------------------------------------------------------
    @staticmethod
    def _redact_log_message(msg: str) -> str:
        """Redact sensitive information from a log message (SEC11, SEC13).

        Strips ``api_key=...`` query parameters from URLs.

        Parameters
        ----------
        msg : str
            Log message.

        Returns
        -------
        str
            Redacted message.
        """
        return re.sub(
            r"(api[_-]?key=)[^&\s]+", r"\1[REDACTED]", msg, flags=re.IGNORECASE,
        )

    # ---------------------------------------------------------------------
    # _cleanup_old_raw_files() (COMP2)
    # ---------------------------------------------------------------------
    def _cleanup_old_raw_files(self) -> None:
        """Delete raw files older than the retention period (COMP2).

        Reads ``UNIPROT_RAW_RETENTION_DAYS`` env var (default 90 days).
        """
        try:
            retention_days = int(os.environ.get(
                "UNIPROT_RAW_RETENTION_DAYS", "90"
            ))
        except ValueError:
            retention_days = 90

        raw_dir = self.effective_raw_dir
        if not raw_dir.exists():
            return

        now = time.time()
        for path in raw_dir.glob("uniprot_human_reviewed.tsv*"):
            try:
                age_days = (now - path.stat().st_mtime) / 86400
                if age_days > retention_days:
                    logger.info(
                        "[%s] Deleting old raw file: %s (%d days old)",
                        self.source_name, path, int(age_days),
                    )
                    path.unlink()
            except OSError as exc:
                logger.warning(
                    "[%s] Could not delete old raw file %s: %s",
                    self.source_name, path, exc,
                )

    # ---------------------------------------------------------------------
    # _check_dependency_versions() (INT1, INT10)
    # ---------------------------------------------------------------------
    @staticmethod
    def _check_dependency_versions() -> None:
        """Verify that library versions meet minimum requirements (INT1, INT10).

        Raises
        ------
        RuntimeError
            If a required library version is too old.
        """
        try:
            from packaging.version import Version
        except ImportError:
            # packaging is not always available -- skip the check.
            return

        min_pandas = Version("1.5.0")
        min_requests = Version("2.28.0")

        if Version(pd.__version__) < min_pandas:
            raise RuntimeError(
                f"pandas >= {min_pandas} required, got {pd.__version__}"
            )
        if Version(requests.__version__) < min_requests:
            raise RuntimeError(
                f"requests >= {min_requests} required, got {requests.__version__}"
            )

    # ---------------------------------------------------------------------
    # _verify_model_sync() (INT17, INT18, D2-9)
    # ---------------------------------------------------------------------
    def _verify_model_sync(self) -> bool:
        """Verify that load_columns is in sync with the Protein model (INT17, INT18).

        Returns
        -------
        bool
            *True* if in sync, *False* (with warnings) otherwise.
        """
        try:
            from database.models import Protein
            model_columns = {c.name for c in Protein.__table__.columns}
            load_cols = set(self._get_load_columns())
            missing_in_load = model_columns - load_cols - {
                "id", "created_at", "updated_at", "is_deleted", "deleted_at",
            }
            extra_in_load = load_cols - model_columns
            if missing_in_load:
                logger.warning(
                    "[%s] Protein model has columns not in load_columns: %s",
                    self.source_name, missing_in_load,
                )
            if extra_in_load:
                logger.warning(
                    "[%s] load_columns has columns not in Protein model: %s",
                    self.source_name, extra_in_load,
                )
            return not (missing_in_load or extra_in_load)
        except ImportError:
            return True  # Can't verify without model.

    # ---------------------------------------------------------------------
    # _api_key (SEC9, INT20)
    # ---------------------------------------------------------------------
    @property
    def _api_key(self) -> Optional[str]:
        """UniProt API key, if configured (SEC9, INT20).

        UniProt does not require an API key for public use, but for
        high-volume usage providing an email is recommended.
        """
        return os.environ.get("UNIPROT_API_KEY")


# ---------------------------------------------------------------------------
# Backward-compatibility module-level constants (A10 + backward compat).
#
# Per the A10 fix, all tunable parameters now live as class attributes on
# ``UniProtPipeline``.  However, existing code (tests, downstream consumers,
# the ``pipelines`` package's lazy-import registry in ``pipelines/__init__.py``)
# still imports these names from the module level.  We expose them here as
# aliases so the public module-level API is preserved (no breaking changes).
#
# Downstream consumers should prefer the class attributes for new code.
# These aliases will be removed in v3.0.
#
# v80 FORENSIC ROOT FIX (P0-C4): ``uniprot_query`` is a ``@property`` on
# ``UniProtPipeline`` (see lines 466-485). Accessing it via the class
# (``UniProtPipeline.uniprot_query``) returns the property DESCRIPTOR
# object -- NOT a string. The previous assignment
# ``UNIPROT_QUERY: str = UniProtPipeline.uniprot_query`` therefore bound
# ``UNIPROT_QUERY`` to a ``property`` object, and any downstream code
# that called ``UNIPROT_QUERY.replace(...)``, ``UNIPROT_QUERY.split(...)``,
# or used it as a requests params value crashed with
# ``AttributeError: 'property' object has no attribute 'replace'`` (or
# was silently passed as the literal string ``"<property object at ...>"``
# in HTTP query params -- a far worse failure mode for a biomedical
# pipeline because the request appeared to succeed but queried nothing).
#
# ROOT FIX: compute the default query string at module-import time using
# the SAME env-var precedence as the property (``UNIPROT_QUERY`` env var
# -> ``DRUGOS_UNIPROT_ORGANISM_ID`` env var -> default 9606). This produces
# a real ``str`` (not a descriptor) while still honouring runtime env
# overrides that were set BEFORE import. Callers who need the live,
# access-time env-var resolution MUST use ``UniProtPipeline().uniprot_query``
# (the instance property) -- the module-level constant is a snapshot for
# backward compatibility only.
# ---------------------------------------------------------------------------
UNIPROT_SEARCH_URL: str = UniProtPipeline.uniprot_search_url
UNIPROT_FIELDS: list[str] = list(UniProtPipeline.uniprot_fields)


def _resolve_uniprot_query_at_import_time() -> str:
    """Resolve the default UniProt query string at module-import time.

    Mirrors the precedence of :pyattr:`UniProtPipeline.uniprot_query`:
      1. ``UNIPROT_QUERY`` env var (full query override)
      2. ``DRUGOS_UNIPROT_ORGANISM_ID`` env var (default ``"9606"``)
         + ``AND reviewed:true``
    Returns a real ``str`` (never a property descriptor).
    """
    _env_full = os.environ.get("UNIPROT_QUERY")
    if _env_full and isinstance(_env_full, str) and _env_full.strip():
        return _env_full.strip()
    _org = os.environ.get("DRUGOS_UNIPROT_ORGANISM_ID", "9606") or "9606"
    if not isinstance(_org, str):
        _org = "9606"
    _org = _org.strip() or "9606"
    return f"organism_id:{_org} AND reviewed:true"


UNIPROT_QUERY: str = _resolve_uniprot_query_at_import_time()
PAGE_SIZE: int = UniProtPipeline.page_size
MAX_RETRIES: int = UniProtPipeline.max_retries
BASE_RETRY_DELAY: float = UniProtPipeline.base_retry_delay

# Extend __all__ to include the backward-compat aliases.
__all__ = list(__all__) + [
    "UNIPROT_SEARCH_URL",
    "UNIPROT_FIELDS",
    "UNIPROT_QUERY",
    "PAGE_SIZE",
    "MAX_RETRIES",
    "BASE_RETRY_DELAY",
]
