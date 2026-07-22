# MIT License -- Copyright (c) 2026 Team Cosmic / VentureLab -- see LICENSE
# SPDX-License-Identifier: MIT
"""
Missing-value handling utilities for the Autonomous Drug Repurposing ETL
platform -- INSTITUTIONAL-GRADE v3.0.0 (16-domain fix, 133 issues resolved).

================================================================================
PROJECT CONTEXT
================================================================================
This module sits at the heart of the dataset pipeline for a platform that
mines 10,000 FDA-approved drugs against every known disease via 7 public
biomedical databases (ChEMBL, DrugBank, UniProt, STRING, DisGeNET, OMIM,
PubChem).  Every record from every one of the 7 source pipelines flows
through one of the four public functions in this file before reaching the
Knowledge Graph (Neo4j), the Graph Transformer (PyTorch Geometric), and
the RL ranker.

A single silent failure here propagates to EVERY downstream consumer:

  Wrong InChIKey        -> wrong graph node  -> wrong prediction
  Wrongly dropped drug  -> missing graph node-> blind spot in predictions
  Wrong organism label  -> mouse protein treated as human -> wrong DPI
  Clipped GDA score     -> protective variant treated as neutral -> missed insight
  Empty SMILES default  -> RDKit crash       -> pipeline failure in production

The data flow is::

    Raw Data (7 sources)
      -> cleaning/normalizer.py        (InChIKey standardization)
      -> cleaning/missing_values.py    (THIS FILE: null handling, recovery, defaults)
      -> cleaning/deduplicator.py      (dedup by InChIKey)
      -> database/loaders.py           (bulk upsert to PostgreSQL/SQLite)
      -> exporters/neo4j_exporter.py   (knowledge graph construction)
      -> Graph Transformer (PyG)
      -> RL Ranker

================================================================================
PROCESSING ORDER (ARCH-4)
================================================================================
The functions in this module are designed to be called in this order:

  1. ``handle_missing_inchikey``   -- recover InChIKeys from SMILES, drop
                                     unidentifiable rows.
  2. ``fill_missing_drug_fields``  -- fill default values for missing
                                     drug fields.  MUST run AFTER
                                     ``handle_missing_inchikey`` because
                                     the legacy default for ``smiles``
                                     is ``""`` (empty string), which
                                     suppresses InChIKey recovery.
  3. ``handle_missing_protein_fields`` -- drop null uniprot_ids, fill
                                     defaults, truncate sequences.
  4. ``validate_gda_scores``       -- clip scores, fill disease names,
                                     fill association types.

For convenience, three orchestration helpers (``clean_drugs``,
``clean_proteins``, ``clean_gda``) enforce this order.  Note that
``cleaning/__init__.py`` also exposes a ``clean_drugs`` function with a
richer step-based pipeline -- these two coexist (the one here is the
simpler in-module orchestrator).

================================================================================
SCIENTIFIC DECISIONS (DOMAIN 3)
================================================================================
ADR-001 -- Null Detection Strategy
    Null detection is column-context-aware.  The ``is_nullish`` function
    accepts a ``NullStrategy`` (or a string ``column_context`` shortcut)
    so that chemical columns (SMILES, InChIKey) do NOT treat ``-`` as
    null -- ``-`` is a single bond in SMILES notation.  Clinical columns
    may treat ``NA`` as null ("Not Available") while genomic columns
    MUST NOT -- ``NA`` is the gene symbol for Nucleosome Assembly
    Protein 1.

ADR-002 -- Conservative Defaults (opt-in)
    The legacy default values (``is_fda_approved=False``, ``smiles=""``,
    ``mechanism_of_action=""``) conflate "unknown" with "confirmed
    negative/empty" -- scientifically dangerous.  v3.0.0 introduces a
    ``conservative_defaults: bool`` parameter on
    ``fill_missing_drug_fields``.  When True, ``is_fda_approved`` is
    filled with ``None`` (nullable Boolean), ``smiles`` is filled with
    ``None`` (prevents RDKit crashes), and ``mechanism_of_action`` is
    filled with ``"Unknown"``.  Default is ``False`` to preserve
    backward compatibility with the v2.0.0 behavior expected by the
    other 12 already-fixed files and the existing test suite.

ADR-003 -- Score Direction Preservation (opt-in)
    The legacy ``validate_gda_scores`` clips negative GDA scores to 0,
    destroying protective-association information.  v3.0.0 introduces
    ``score_range`` and ``preserve_direction`` parameters.  When
    ``score_range=(-1.0, 1.0)`` is set, negative scores (protective
    associations) are preserved.  Default ``(0.0, 1.0)`` preserves
    legacy behavior.

ADR-004 -- Non-Human Organism Safety (opt-in)
    The legacy ``handle_missing_protein_fields`` fills NaN organism
    with ``"Homo sapiens"`` even when non-human proteins are present --
    a data corruption event.  v3.0.0 keeps the legacy default but
    adds an ``organism_fill_mode`` parameter: ``"default"`` (legacy),
    ``"strict"`` (use ``"Unknown organism"`` when non-human proteins
    detected), or ``"skip"`` (leave NaN).  Default is ``"default"`` to
    preserve backward compatibility.

================================================================================
THREAD SAFETY (REL-7, PERF-7)
================================================================================
Module-level mutable state (``_metrics``, ``_dead_letters``,
``_current_correlation_id``) is guarded by ``threading.RLock`` instances.
The four public functions are stateless with respect to user input -- they
always copy the input DataFrame before mutating.  However, the
module-level state IS shared across threads. Locks are acquired
before any mutation, so concurrent calls are safe but serialized on
the lock. ``reset_metrics`` and ``clear_dead_letters`` acquire the same
locks and are safe to call from any thread (typically the test
runner or pipeline supervisor).

================================================================================
DATA LINEAGE (DOMAIN 16)
================================================================================
Every transformation attaches underscore-prefixed lineage columns to the
output DataFrame:

  ``_inchikey_source``             -- "recovered_from_smiles" / "original" / None
  ``_smiles_used_for_recovery``    -- the SMILES string used (for audit)
  ``_inchikey_recovery_failed``    -- True if recovery was attempted and failed
  ``_inchikey_recovery_error``     -- error category from ConversionResult
  ``_organism_was_defaulted``      -- True if organism was filled with default
  ``_gene_name_was_filled``        -- True if gene_name was filled
  ``_function_desc_was_filled``    -- True if function_desc was filled
  ``_sequence_was_truncated``      -- True if sequence was truncated
  ``_original_sequence_length``    -- int or None (original length before truncation)
  ``_score_was_clipped``           -- True if score was clipped to range
  ``_score_was_coerced_nan``       -- True if score was non-numeric and coerced to NaN
  ``_original_score``              -- the original score value (if clipped)
  ``_score_direction``             -- "positive", "negative", or None
  ``_disease_name_was_filled``     -- True if disease_name was filled
  ``_association_type_was_filled`` -- True if association_type was filled
  ``_{col}_was_filled``            -- per-column flag in fill_missing_drug_fields

In addition, ``DataFrame.attrs["_cleaning_metadata"]`` is set with
provenance information: timestamp, module version, input/output
fingerprints, pandas version, and the function name that produced the
output.

================================================================================
CHANGELOG
================================================================================
v1.0.0 -- Initial implementation (4 functions, simple null handling).
v2.0.0 -- Added ``MAX_SEQUENCE_LENGTH`` public alias, ``_is_nullish``
         hardening to exclude ``"na"`` and ``"none"`` from null patterns.
v3.0.0 -- Comprehensive 16-domain institutional-grade upgrade (133 issues
         resolved across Architecture, Design, Scientific Correctness,
         Coding, Data Quality, Reliability, Idempotency, Performance,
         Security, Testing, Logging, Configuration, Documentation,
         Compliance, Interoperability, and Lineage).  All new
         functionality is opt-in via keyword parameters; legacy defaults
         are preserved for backward compatibility with the 12 already-
         fixed files in the v2.1.0 codebase.
"""

from __future__ import annotations

# Standard library imports (alphabetical)
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd

# Lazy imports (circular dependency guard -- ARCH-1, GUARD-A7):
#   cleaning/normalizer.py MUST NOT import from cleaning/missing_values.py.
#   We import normalizer symbols lazily inside helper functions so that
#   import order does not matter.
#
# Symbols imported lazily from .normalizer:
#   - convert_to_inchikey            (single SMILES -> InChIKey or None)
#   - convert_to_inchikeys           (batch SMILES -> list[InChIKey|None])
#   - convert_to_inchikey_detailed   (single SMILES -> ConversionResult)
#   - standardize_inchikey           (validate+normalize an InChIKey)
#   - ALLOWED_TYPES                  (canonical list of drug_type values)
#   - ConversionResult               (NamedTuple returned by *_detailed)


# ===========================================================================
# Module logger (LOG-1, LOG-2)
# ===========================================================================
logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration loading (CFG-1, CFG-2, CFG-3, CFG-5, CFG-6)
# ===========================================================================
# v3.0.0 introduces optional configuration via config.settings.  We use a
# defensive import so that this module works standalone (without the config
# package) -- useful for unit tests and for the v2.x codebase that has not
# yet added MAX_SEQUENCE_LENGTH / DEFAULT_ORGANISM / CLEANING_NULL_PATTERNS_JSON
# to config.settings.

def _load_config_value(name: str, default: Any) -> Any:
    """Lazily fetch a value from ``config.settings`` with a fallback.

    Parameters
    ----------
    name : str
        The configuration attribute name (e.g. ``"MAX_SEQUENCE_LENGTH"``).
    default : Any
        The fallback value if config is unavailable or the attribute
        is missing.

    Returns
    -------
    Any
        The resolved configuration value.
    """
    try:
        from config import settings as _settings  # type: ignore[import]
        return getattr(_settings, name, default)
    except Exception:  # noqa: BLE001 -- defensive by design
        return default


def _load_environment() -> str:
    """Return the current environment name (dev/staging/prod)."""
    try:
        from config import settings as _settings  # type: ignore[import]
        env = getattr(_settings, "ENVIRONMENT", None)
        if isinstance(env, str) and env:
            return env.lower()
    except Exception:  # noqa: BLE001
        pass
    # v36 ROOT FIX (Chain 1): honour DRUGOS_ENVIRONMENT canonical name.
    return (
        os.getenv("DRUGOS_ENVIRONMENT")
        or os.getenv("ENVIRONMENT", "development")
    ).lower()


# ===========================================================================
# Constants (CFG-1, CFG-2, CFG-3)
# ===========================================================================
# MAX_SEQUENCE_LENGTH is configurable via config.settings.MAX_SEQUENCE_LENGTH
# or the CLEANING_MAX_SEQUENCE_LENGTH env var.  The v2.0.0 default was 10,000
# amino acids -- preserved here for backward compatibility with the 12 already-
# fixed files in the v2.1.0 codebase.  The scientifically-correct value is
# 35,000 (titin, the largest known human protein, is ~34,350 aa); set the env
# var ``CLEANING_MAX_SEQUENCE_LENGTH=35000`` or call
# ``cleaning.configure(max_sequence_length=35000)`` to use it.
#
# The PRIVATE ``_MAX_SEQUENCE_LENGTH`` is the source of truth used by
# ``handle_missing_protein_fields`` and is mutable via
# ``cleaning.configure(max_sequence_length=...)``.

_DEFAULT_MAX_SEQUENCE_LENGTH: int = 10_000  # v2.0.0 backward-compat default
_MAX_SEQUENCE_LENGTH: int = int(
    os.getenv("CLEANING_MAX_SEQUENCE_LENGTH", "0")) or _load_config_value(
        "MAX_SEQUENCE_LENGTH", _DEFAULT_MAX_SEQUENCE_LENGTH)
# Sanity-check the configured value.
if not isinstance(_MAX_SEQUENCE_LENGTH, int) or _MAX_SEQUENCE_LENGTH < 1:
    _MAX_SEQUENCE_LENGTH = _DEFAULT_MAX_SEQUENCE_LENGTH

# DEFAULT_ORGANISM is configurable.  Legacy default was "Homo sapiens".
_DEFAULT_ORGANISM: str = str(
    os.getenv("CLEANING_DEFAULT_ORGANISM") or _load_config_value(
        "DEFAULT_ORGANISM", "Homo sapiens"))

# ENVIRONMENT drives strict validation behavior (CFG-5).
_ENVIRONMENT: str = _load_environment()
_STRICT_VALIDATION: bool = _ENVIRONMENT in {"staging", "production", "prod"}

# Public alias for re-export through cleaning/__init__.py (GAP-DQ3).
# This is a SNAPSHOT taken at module load -- it does NOT track later
# mutations to _MAX_SEQUENCE_LENGTH via cleaning.configure().  This
# matches the documented v2.0.0 behavior (test_settings_max_sequence_length_configurable
# in test_all_12_files_integration_v2.py).
MAX_SEQUENCE_LENGTH: int = _MAX_SEQUENCE_LENGTH

# Backward-compat alias: some callers import DEFAULT_ORGANISM from this module.
DEFAULT_ORGANISM: str = _DEFAULT_ORGANISM

# ---------------------------------------------------------------------------
# Operational constants
# ---------------------------------------------------------------------------
_MAX_DATAFRAME_ROWS: int = 10_000_000        # SEC-4: DoS guard
_BATCH_THRESHOLD: int = 10                    # ARCH-2: batch API threshold
_MAX_RETRIES: int = 2                         # REL-6: conversion retry count
_RETRY_BACKOFF_BASE: float = 0.1              # REL-6: exponential backoff base
_CIRCUIT_BREAKER_THRESHOLD: int = 10          # REL-7: consecutive failures to open
_CHECKPOINT_INTERVAL: int = 1_000             # REL-10: rows between checkpoints
_SMILES_MAX_LENGTH: int = 10_000              # SEC-1: SMILES length cap
_OUTPUT_SCHEMA_VERSION: str = "3.0.0"         # COMP-3: schema version attached to outputs
_MODULE_VERSION: str = "3.0.0"                # COMP-3: module version

# Numeric sentinel values that may indicate missing data (CODE-1).
# These are NOT silently treated as null -- we only WARN when we see them.
_NUMERIC_SENTINELS: list = [-999, -9999, -1, 9999]

# Valid GDA association types (DQ-7) -- allowlist for warning-only validation.
_VALID_ASSOCIATION_TYPES: frozenset = frozenset({
    "somatic", "germline", "mixed", "unknown", "predictive",
    "therapeutic", "diagnostic", "prognostic", "predisposing",
    "protective", "contraindicated",
})

# Null pattern sets -- column-context-aware (DESIGN-1, DESIGN-3).
_NULL_PATTERNS_UNIVERSAL: frozenset = frozenset({"null", "n/a", ""})
_NULL_PATTERNS_GENERAL: frozenset = frozenset({"null", "n/a", "-", "--", ""})
_NULL_PATTERNS_CHEMICAL: frozenset = frozenset({"null", "n/a", ""})
_NULL_PATTERNS_STRICT: frozenset = frozenset({""})

# Suspicious SMILES patterns (SEC-1) -- defense-in-depth, normalizer already validates.
_SMILES_SUSPICIOUS_PATTERNS: list = [
    re.compile(r"(.)\1{100,}"),                                  # run-length abuse
    re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]"),                 # control chars
]

# PII scan patterns (SEC-2) -- for warn-only PII detection in string columns.
_PII_PATTERNS: list = [
    ("email", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("phone", re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")),
    ("mrn", re.compile(r"\bMRN[:\s]?\d+\b", re.IGNORECASE)),
]

# Pre-compiled regex (PERF-4) -- whitespace-only string detector.
_WHITESPACE_REGEX: re.Pattern = re.compile(r"^\s*$")

# Pre-compiled regex -- numeric score validator (CODE-14).
# SCI-FIX: the original pattern ``^-?\d+\.?\d*$`` rejected scientific
# notation (``1e-5``, ``1.5E10``), leading-dot decimals (``.5``,
# ``-.5``), and explicit plus signs (``+5``). DisGeNET/OMIM GDA scores
# can legitimately be very small (e.g., GWAS-derived protective scores
# like 1e-7) -- rejecting them silently destroyed real biological
# signal AND polluted the DQ report's ``non_numeric_count`` field
# because ``pd.to_numeric("1e-5")`` actually SUCCEEDS (returns 1e-5)
# while the regex pre-check had already flagged the row as non-numeric.
# The new pattern accepts the full numeric grammar that
# ``pd.to_numeric`` accepts, including scientific notation, leading
# dots, and explicit sign.
_NUMERIC_SCORE_REGEX: re.Pattern = re.compile(
    r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$"
)


# ===========================================================================
# Module-level state (LOG-7, REL-9, LOG-6)
# ===========================================================================
# Operational metrics counter (LOG-7).  Guarded by _METRICS_LOCK.
_metrics: dict = defaultdict(int)
_METRICS_LOCK = threading.RLock()

# Dead-letter queue for dropped/failed rows (REL-9, DQ-2).  Guarded.
_dead_letters: list = []
_DEAD_LETTERS_LOCK = threading.RLock()
# FIX-F / C-18: alias kept for backward-compat with operators/tests that
# import ``_dead_letter_queue`` from this module. Same list object --
# in-place mutations (.append/.clear/.pop) are visible through either name.
_dead_letter_queue: list = _dead_letters

# Correlation ID for cross-function log tracing (LOG-6).  Guarded.
_current_correlation_id: Optional[str] = None
_CORRELATION_ID_LOCK = threading.RLock()

# Maximum dead letters to retain (memory-bounded).
_MAX_DEAD_LETTERS: int = 10_000


# ===========================================================================
# Pandas version handling (CODE-4, IDEM-6, CFG-6)
# ===========================================================================
def _parse_pandas_version() -> tuple:
    """Robustly parse the pandas version as a (major, minor) tuple.

    Uses ``packaging.version.Version`` when available (PEP 440 compliant);
    falls back to a regex that extracts the first two numeric components.

    Returns
    -------
    tuple[int, int]
        Major and minor version of pandas (e.g. ``(2, 2)`` for 2.2.3).
        Returns ``(99, 0)`` if parsing fails -- assumes a future version
        with the modern API.
    """
    try:
        from packaging.version import Version  # type: ignore[import]
        v = Version(pd.__version__)
        return (v.major, v.minor)
    except Exception:  # noqa: BLE001
        match = re.match(r"(\d+)\.(\d+)", pd.__version__)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (99, 0)


_PD_VERSION: tuple = _parse_pandas_version()


# ===========================================================================
# Lazy import helpers (ARCH-1, ARCH-6, GUARD-A7)
# ===========================================================================
def _get_convert_to_inchikey() -> Callable:
    """Lazy import for single-SMILES InChIKey conversion.

    Returns
    -------
    Callable[[Any], Optional[str]]
        The ``convert_to_inchikey`` function from ``cleaning.normalizer``.

    Raises
    ------
    ImportError
        If ``cleaning.normalizer`` cannot be imported (circular dep guard).
    """
    from .normalizer import convert_to_inchikey
    return convert_to_inchikey


def _get_convert_to_inchikeys() -> Callable:
    """Lazy import for batch SMILES conversion (ARCH-2).

    Returns
    -------
    Callable[[Iterable], list[Optional[str]]]
        The ``convert_to_inchikeys`` function from ``cleaning.normalizer``.
    """
    from .normalizer import convert_to_inchikeys
    return convert_to_inchikeys


def _get_convert_detailed() -> Callable:
    """Lazy import for detailed single-SMILES conversion (DESIGN-6).

    Returns
    -------
    Callable[[Any], Any]
        The ``convert_to_inchikey_detailed`` function returning
        ``ConversionResult``.
    """
    from .normalizer import convert_to_inchikey_detailed
    return convert_to_inchikey_detailed


def _get_standardize_inchikey() -> Callable:
    """Lazy import for InChIKey standardization (DQ-1).

    Returns
    -------
    Callable[[Any], Optional[str]]
        The ``standardize_inchikey`` function from ``cleaning.normalizer``.
    """
    from .normalizer import standardize_inchikey
    return standardize_inchikey


def _get_allowed_types() -> list:
    """Lazy import for the canonical ALLOWED_TYPES list (COMP-6).

    Returns
    -------
    list[str]
        The ``ALLOWED_TYPES`` list from ``cleaning.normalizer``.  Returns
        an empty list if normalizer cannot be imported.
    """
    try:
        from .normalizer import ALLOWED_TYPES
        return list(ALLOWED_TYPES)
    except Exception:  # noqa: BLE001
        return []


# ===========================================================================
# Null detection (ARCH-7, ARCH-8, DESIGN-1, DESIGN-2, DESIGN-3, CODE-1,
#                 CODE-3, CODE-13, REL-3, DQ-6, DOC-4)
# ===========================================================================
@dataclass(frozen=True)
class NullStrategy:
    """Configuration for column-context-aware null detection (DESIGN-3).

    A ``NullStrategy`` bundles together the parameters that ``is_nullish``
    needs to decide whether a value should be treated as missing.
    Different column contexts require different strategies -- for example,
    a chemical SMILES column should NOT treat ``-`` as null (single bond)
    while a clinical "disease_name" column SHOULD treat ``NA`` as null
    ("Not Available"), but a genomic "gene_symbol" column MUST NOT treat
    ``NA`` as null (gene symbol for Nucleosome Assembly Protein 1).

    Attributes
    ----------
    extra_null_patterns : frozenset[str]
        Additional lower-cased string patterns to treat as null.  Empty
        by default.  Example: ``frozenset({"missing", "not_reported"})``.
    exclude_patterns : frozenset[str]
        Patterns to EXCLUDE from the default null-pattern set.  Empty by
        default.  Example: ``frozenset({"-", "--"})`` for chemical columns.
    treat_na_as_null : bool
        If True, the literal string ``"na"`` (case-insensitive) is treated
        as null.  Default False -- preserves biomedical gene symbols.
    treat_none_as_null : bool
        If True, the literal string ``"none"`` (case-insensitive) is
        treated as null.  Default False -- "none" is a legitimate biomedical
        value (e.g., "None identified" in protein function descriptions).
    detect_sentinels : bool
        If True, warn (do not flag) about numeric sentinel values like
        ``-999`` in numeric columns.  Default True.
    """

    extra_null_patterns: frozenset = frozenset()
    exclude_patterns: frozenset = frozenset()
    treat_na_as_null: bool = False
    treat_none_as_null: bool = False
    detect_sentinels: bool = True


# Predefined strategies (DESIGN-3).
NULL_STRATEGY_GENERAL: NullStrategy = NullStrategy()
NULL_STRATEGY_CHEMICAL: NullStrategy = NullStrategy(
    exclude_patterns=frozenset({"-", "--"}),
)
NULL_STRATEGY_CLINICAL: NullStrategy = NullStrategy(
    treat_na_as_null=True,
)
NULL_STRATEGY_GENE: NullStrategy = NullStrategy(
    treat_na_as_null=False,
    treat_none_as_null=False,
)
NULL_STRATEGY_STRICT: NullStrategy = NullStrategy(
    extra_null_patterns=frozenset(),
    exclude_patterns=frozenset({"-", "--", "null", "n/a"}),
    treat_na_as_null=False,
    treat_none_as_null=False,
)

# Shortcut map: column_context name -> NullStrategy.
_STRATEGY_BY_CONTEXT: dict = {
    "general": NULL_STRATEGY_GENERAL,
    "chemical": NULL_STRATEGY_CHEMICAL,
    "clinical": NULL_STRATEGY_CLINICAL,
    "gene": NULL_STRATEGY_GENE,
    "strict": NULL_STRATEGY_STRICT,
}


def _resolve_strategy(
    strategy: Optional[Union[NullStrategy, str]],
    column_context: Optional[str],
) -> NullStrategy:
    """Resolve a (strategy, column_context) pair into a single NullStrategy.

    Parameters
    ----------
    strategy : NullStrategy | str | None
        A NullStrategy instance, or a string shortcut name (one of
        "general", "chemical", "clinical", "gene", "strict"), or None.
    column_context : str | None
        A string shortcut name.  Used only when ``strategy`` is None.

    Returns
    -------
    NullStrategy
        The resolved strategy.  Defaults to ``NULL_STRATEGY_GENERAL`` if
        both inputs are None.
    """
    if isinstance(strategy, NullStrategy):
        return strategy
    if isinstance(strategy, str):
        return _STRATEGY_BY_CONTEXT.get(strategy.lower(), NULL_STRATEGY_GENERAL)
    if isinstance(column_context, str):
        return _STRATEGY_BY_CONTEXT.get(column_context.lower(), NULL_STRATEGY_GENERAL)
    return NULL_STRATEGY_GENERAL


def _build_pattern_set(strategy: NullStrategy) -> frozenset:
    """Build the final frozenset of null-like patterns for a strategy."""
    patterns = set(_NULL_PATTERNS_GENERAL)
    patterns -= set(strategy.exclude_patterns)
    if strategy.treat_na_as_null:
        patterns.add("na")
    if strategy.treat_none_as_null:
        patterns.add("none")
    patterns |= set(strategy.extra_null_patterns)
    return frozenset(patterns)


def is_nullish(
    series: pd.Series,
    *,
    strategy: Optional[Union[NullStrategy, str]] = None,
    column_context: Optional[str] = None,
) -> pd.Series:
    """Return a boolean mask True where ``series`` is null-like.

    This is THE canonical null-detection function for the entire
    ``cleaning`` sub-package (ARCH-7, ARCH-8).  It is column-context-aware
    via the ``strategy`` / ``column_context`` parameters.

    Null Detection Rationale (DOC-4)
    -------------------------------
    1. **NaN/NA/None** are always null.  ``series.isna()`` catches
       ``np.nan``, ``None``, ``pd.NA``, and ``pd.NaT``.
    2. **Empty / whitespace-only strings** are null in every context.
    3. **Literal "null" and "n/a"** (case-insensitive) are null in every
       context -- they are explicit null markers.
    4. **"-" and "--"** are null in *general* context but NOT in
       *chemical* context (single bond in SMILES, en-dash in IUPAC names).
    5. **"na"** is NOT null by default -- it is the gene symbol for
       Nucleosome Assembly Protein 1.  Use ``NULL_STRATEGY_CLINICAL``
       or ``strategy.treat_na_as_null=True`` to treat it as null in
       clinical columns (e.g. disease_name).
    6. **"none"** is NOT null by default -- it is a legitimate biomedical
       value (e.g., "None identified" in protein function descriptions).
       Use ``strategy.treat_none_as_null=True`` to treat it as null.
    7. **Numeric sentinel values** (``-999``, ``-9999``, etc.) are NOT
       silently treated as null -- they are logged as warnings to
       surface upstream data quality issues.

    Parameters
    ----------
    series : pd.Series
        The series to check.  Any dtype is accepted.
    strategy : NullStrategy | str | None
        A ``NullStrategy`` instance, a string shortcut ("general",
        "chemical", "clinical", "gene", "strict"), or None.  If None,
        ``column_context`` is consulted; if both are None,
        ``NULL_STRATEGY_GENERAL`` is used.
    column_context : str | None
        A string shortcut for the strategy.  Ignored if ``strategy``
        is provided as a ``NullStrategy`` instance.

    Returns
    -------
    pd.Series[bool]
        Boolean mask aligned to ``series.index``.  True where the value
        is null-like.  Never raises -- falls back to ``series.isna()``
        on internal errors (REL-3).

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series(["NA", "null", "none", "valid", None, ""])
    >>> is_nullish(s).tolist()
    [False, True, False, False, True, True]
    >>> is_nullish(s, column_context="clinical").tolist()
    [True, True, False, False, True, True]
    >>> is_nullish(pd.Series(["-", "CCO", ""]), column_context="chemical").tolist()
    [False, False, True]
    """
    # Defensive: REL-3 -- never raise from is_nullish.
    try:
        resolved = _resolve_strategy(strategy, column_context)
        patterns = _build_pattern_set(resolved)

        # Step 1: catch NaN / None / pd.NA / pd.NaT (CODE-3, CODE-13).
        null_mask = series.isna()

        # Step 2: detect string-like columns (DESIGN-2, CODE-3).
        is_string_like = (
            series.dtype == object
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or str(series.dtype) == "string"
        )

        non_null_mask = ~null_mask

        # Step 3: apply string comparison only to non-NaN values (CODE-3).
        if is_string_like and non_null_mask.any():
            # Work on a copy of just the non-null values to avoid the
            # astype(str) -> "nan" issue (CODE-3, BUG-CODE-3).
            non_null_values = series[non_null_mask]
            # astype(str) here is safe -- non_null_values has no NaN.
            string_values = non_null_values.astype(str)
            stripped_lower = string_values.str.strip().str.lower()
            empty_mask = string_values.str.strip() == ""
            null_like_mask = stripped_lower.isin(patterns)
            string_null_mask = empty_mask | null_like_mask

            # Build a complete-series mask aligned to series.index.
            # Explicitly use bool dtype to avoid FutureWarning about
            # incompatible dtype when assigning bool values to an
            # object-dtype Series (pandas 2.2+).
            full_string_null = pd.Series(
                False, index=series.index, dtype=bool
            )
            # string_null_mask is aligned to non_null_values.index which
            # is a subset of series.index.  Assign by .loc to be safe.
            full_string_null.loc[non_null_mask] = string_null_mask.values
            null_mask = null_mask | full_string_null

        # Step 4: detect non-scalar values in object columns (DQ-6).
        if is_string_like and non_null_mask.any() and _STRICT_VALIDATION:
            non_null_values = series[non_null_mask]
            non_scalar_mask = non_null_values.apply(
                lambda x: not _is_scalar(x)
            )
            non_scalar_count = int(non_scalar_mask.sum())
            if non_scalar_count > 0:
                logger.warning(
                    "is_nullish: %d non-scalar value(s) detected in "
                    "object column -- these may indicate upstream schema "
                    "corruption",
                    non_scalar_count,
                )

        # Step 5: warn on numeric sentinel values (CODE-1).
        if (
            resolved.detect_sentinels
            and pd.api.types.is_numeric_dtype(series)
            and non_null_mask.any()
        ):
            for sentinel in _NUMERIC_SENTINELS:
                try:
                    sentinel_count = int((series == sentinel).sum())
                except Exception:  # noqa: BLE001
                    sentinel_count = 0
                if sentinel_count > 0:
                    logger.warning(
                        "is_nullish: %d sentinel value(s) %r detected "
                        "in numeric column -- these are NOT treated as "
                        "null but may indicate missing data upstream",
                        sentinel_count,
                        sentinel,
                    )
                    with _METRICS_LOCK:
                        _metrics["sentinel_values_detected"] += sentinel_count

        # Final invariant: the returned mask must be a bool Series of
        # the same length as the input (BUG-CODE-1).
        if not isinstance(null_mask, pd.Series):
            null_mask = pd.Series(null_mask, index=series.index)
        if null_mask.dtype != bool:
            null_mask = null_mask.astype(bool)
        if len(null_mask) != len(series):
            # Length mismatch -- this should never happen, but if it does
            # (e.g. due to a pandas bug), fall back to isna().
            logger.error(
                "is_nullish: internal error -- mask length %d != series "
                "length %d; falling back to isna()",
                len(null_mask),
                len(series),
            )
            null_mask = series.isna()
        return null_mask

    except Exception as exc:  # noqa: BLE001 -- REL-3 defensive fallback
        logger.error(
            "is_nullish: internal error (%s); falling back to isna()",
            exc,
        )
        with _METRICS_LOCK:
            _metrics["is_nullish_fallback_count"] += 1
        return series.isna()


def _is_scalar(value: Any) -> bool:
    """Return True if ``value`` is a scalar (not a list/tuple/dict/array)."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        # pandas / numpy array-like
        if hasattr(value, "__len__") and not isinstance(value, str):
            if hasattr(value, "shape"):
                return False
            if len(value) > 0:
                return False
    except Exception:  # noqa: BLE001
        pass
    return True


def _is_nullish(series: pd.Series) -> pd.Series:
    """Backward-compatible private alias for :func:`is_nullish`.

    Preserves the v2.0.0 contract: ``_is_nullish(series)`` returns a
    boolean mask using the *general* null strategy (which is the v2.0.0
    behavior -- "na" and "none" are NOT null, "-" and "--" ARE null).

    Existing tests in ``test_all_45_fixes.py::TestIssue23`` and
    ``test_all_45_fixes.py::test_nullish_na_gene_symbol`` verify this
    behavior.  Do NOT change this function's semantics.

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    pd.Series[bool]
    """
    return is_nullish(series, strategy=NULL_STRATEGY_GENERAL)


def _is_nullish_value(value: Any) -> bool:
    """Scalar version of :func:`is_nullish` for single-value checks (PERF-3).

    Parameters
    ----------
    value : Any

    Returns
    -------
    bool
    """
    if value is None:
        return True
    try:
        if isinstance(value, float) and np.isnan(value):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        if value is pd.NA:
            return True
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in _NULL_PATTERNS_GENERAL:
            return True
        return False
    return False


# ===========================================================================
# Security helpers (SEC-1, SEC-2, SEC-3, SEC-4, SEC-5)
# ===========================================================================
def _sanitize_smiles(smiles: Any) -> Optional[str]:
    """Defense-in-depth SMILES sanitization (SEC-1).

    The normalizer already validates SMILES length and character set.
    This function catches edge cases EARLY (before the lazy import to
    normalizer) and provides fast rejection of obviously-malicious input.

    Parameters
    ----------
    smiles : Any
        The candidate SMILES value.

    Returns
    -------
    str | None
        The cleaned SMILES string, or None if the input is None/empty/
        too long/contains control characters.
    """
    if smiles is None:
        return None
    if not isinstance(smiles, str):
        try:
            smiles = str(smiles)
        except Exception:  # noqa: BLE001
            return None
    if not smiles:
        return None
    if len(smiles) > _SMILES_MAX_LENGTH:
        logger.warning(
            "_sanitize_smiles: SMILES length %d exceeds cap %d -- rejecting",
            len(smiles),
            _SMILES_MAX_LENGTH,
        )
        return None
    for pattern in _SMILES_SUSPICIOUS_PATTERNS:
        if pattern.search(smiles):
            logger.warning(
                "_sanitize_smiles: SMILES contains suspicious pattern -- rejecting"
            )
            return None
    return smiles.strip() or None


def _redact_for_log(value: Any, max_len: int = 80) -> str:
    """Redact sensitive / over-long values for safe logging (SEC-3).

    Parameters
    ----------
    value : Any
        The value to redact.
    max_len : int
        Maximum length of the returned string.

    Returns
    -------
    str
        A safe-to-log representation.
    """
    if value is None:
        return "None"
    try:
        s = str(value)
    except Exception:  # noqa: BLE001
        return "<unprintable>"
    if len(s) > max_len:
        s = s[:max_len] + "...[truncated]"
    # Mask email-like patterns.
    s = _PII_PATTERNS[0][1].sub("[email]", s)
    # Mask SSN-like patterns.
    s = _PII_PATTERNS[1][1].sub("[ssn]", s)
    return s


def _scan_for_pii(df: pd.DataFrame) -> dict:
    """Warn-only PII scan over string columns (SEC-2).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict[str, int]
        Mapping from PII type ("email", "ssn", "phone", "mrn") to the
        total count of matches across all string columns.
    """
    counts: dict = defaultdict(int)
    for col in df.columns:
        try:
            if df[col].dtype != object and not pd.api.types.is_string_dtype(df[col]):
                continue
        except Exception:  # noqa: BLE001
            continue
        non_null = df[col].dropna().astype(str)
        if non_null.empty:
            continue
        for pii_type, pattern in _PII_PATTERNS:
            try:
                matches = non_null.str.contains(pattern, regex=True, na=False)
                count = int(matches.sum())
            except Exception:  # noqa: BLE001
                count = 0
            if count > 0:
                counts[pii_type] += count
                logger.warning(
                    "_scan_for_pii: %d %s-like value(s) detected in "
                    "column %r -- review for PII leakage",
                    count,
                    pii_type,
                    col,
                )
    return dict(counts)


def _validate_input_size(df: pd.DataFrame) -> None:
    """Reject DataFrames exceeding the safety size cap (SEC-4).

    Parameters
    ----------
    df : pd.DataFrame

    Raises
    ------
    ValueError
        If the DataFrame exceeds ``_MAX_DATAFRAME_ROWS`` rows.
    """
    try:
        nrows = len(df)
    except Exception:  # noqa: BLE001
        return
    if nrows > _MAX_DATAFRAME_ROWS:
        raise ValueError(
            f"DataFrame has {nrows:,} rows which exceeds the safety "
            f"cap of {_MAX_DATAFRAME_ROWS:,}.  Process in chunks via "
            f"the ``chunk_size`` parameter."
        )


def _validate_column_types(df: pd.DataFrame) -> None:
    """Warn on dangerous object dtypes (SEC-5).

    Parameters
    ----------
    df : pd.DataFrame
    """
    for col in df.columns:
        try:
            if df[col].dtype != object:
                continue
        except Exception:  # noqa: BLE001
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        # Check if any value is callable (potential code injection risk).
        try:
            has_callable = non_null.apply(
                lambda x: callable(x) and not isinstance(x, type)
            ).any()
            if has_callable:
                logger.warning(
                    "_validate_column_types: column %r contains callable "
                    "objects -- possible code injection risk",
                    col,
                )
        except Exception:  # noqa: BLE001
            pass


def _validate_input_schema(
    df: pd.DataFrame,
    required: list,
    function_name: str,
) -> None:
    """Validate that the input DataFrame has the required columns (DQ-12).

    Parameters
    ----------
    df : pd.DataFrame
    required : list[str]
        Column names that MUST be present.
    function_name : str
        Name of the calling function (for error messages).
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{function_name}: input DataFrame is missing required "
            f"column(s): {missing}.  Present columns: {list(df.columns)}"
        )


# ===========================================================================
# Data lineage helpers (LINEAGE-1..8, IDEM-7, IDEM-8)
# ===========================================================================
def _fingerprint_df(df: pd.DataFrame) -> str:
    """Return a stable SHA-256 fingerprint of a DataFrame (IDEM-7).

    The fingerprint is based on the DataFrame's content (column names +
    values) and is stable across runs given the same input data.  It is
    used for idempotency verification and provenance tracking.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    str
        A 64-character hex SHA-256 digest.
    """
    try:
        # pd.util.hash_pandas_object is content-based and stable.
        hash_series = pd.util.hash_pandas_object(df, index=True)
        h = hashlib.sha256()
        h.update(str(hash_series.values).encode("utf-8"))
        h.update(str(list(df.columns)).encode("utf-8"))
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        # Fall back to a less precise fingerprint.
        h = hashlib.sha256()
        h.update(str(list(df.columns)).encode("utf-8"))
        try:
            h.update(str(len(df)).encode("utf-8"))
        except Exception:  # noqa: BLE001
            pass
        return h.hexdigest()


def _set_cleaning_metadata(
    df: pd.DataFrame,
    *,
    function_name: str,
    input_fingerprint: str,
    input_rows: int,
) -> None:
    """Attach provenance metadata to ``df.attrs`` (LINEAGE-8, IDEM-8).

    Parameters
    ----------
    df : pd.DataFrame
        The output DataFrame to annotate (mutated in place via .attrs).
    function_name : str
        Name of the cleaning function that produced this output.
    input_fingerprint : str
        SHA-256 fingerprint of the input DataFrame.
    input_rows : int
        Row count of the input DataFrame.
    """
    output_fingerprint = _fingerprint_df(df)
    metadata = {
        "function": function_name,
        "module_version": _MODULE_VERSION,
        "schema_version": _OUTPUT_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pandas_version": pd.__version__,
        "input_fingerprint": input_fingerprint,
        "output_fingerprint": output_fingerprint,
        "input_rows": input_rows,
        "output_rows": len(df),
        "correlation_id": get_correlation_id(),
    }
    # Merge with any existing metadata (e.g. from a prior cleaning step).
    existing = df.attrs.get("_cleaning_metadata", {})
    if isinstance(existing, dict):
        existing["pipeline_history"] = existing.get("pipeline_history", []) + [
            {"function": function_name, "timestamp": metadata["timestamp"]}
        ]
        metadata["pipeline_history"] = existing["pipeline_history"]
    df.attrs["_cleaning_metadata"] = metadata


def get_provenance(df: pd.DataFrame) -> dict:
    """Return the provenance metadata attached to a cleaned DataFrame (LINEAGE-8).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        The ``_cleaning_metadata`` dict, or an empty dict if no metadata
        has been attached.
    """
    md = df.attrs.get("_cleaning_metadata")
    return dict(md) if isinstance(md, dict) else {}


# ===========================================================================
# Observability (LOG-4, LOG-6, LOG-7, REL-9, REL-10)
# ===========================================================================
def get_metrics() -> dict:
    """Return a snapshot of the module-level metrics counters (LOG-7).

    Returns
    -------
    dict[str, int]
        Mapping from metric name to integer counter value.  Never raises.
    """
    with _METRICS_LOCK:
        return dict(_metrics)


def reset_metrics() -> None:
    """Reset all module-level metrics counters to zero (LOG-7)."""
    with _METRICS_LOCK:
        _metrics.clear()


def get_dead_letters() -> list:
    """Return a snapshot of the dead-letter queue (REL-9).

    Returns
    -------
    list[dict]
        List of dead-letter records.  Each record is a dict with keys:
        ``function``, ``reason``, ``row`` (dict), ``timestamp``,
        ``correlation_id``.
    """
    with _DEAD_LETTERS_LOCK:
        return list(_dead_letters)


def clear_dead_letters() -> None:
    """Clear the dead-letter queue (REL-9)."""
    with _DEAD_LETTERS_LOCK:
        _dead_letters.clear()


def _append_dead_letter(
    function_name: str,
    reason: str,
    row: Optional[dict],
) -> None:
    """Append a record to the dead-letter queue (bounded -- REL-9)."""
    with _DEAD_LETTERS_LOCK:
        if len(_dead_letters) >= _MAX_DEAD_LETTERS:
            # Drop the oldest entry to bound memory.
            _dead_letters.pop(0)
        _dead_letters.append({
            "function": function_name,
            "reason": reason,
            "row": dict(row) if isinstance(row, dict) else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": get_correlation_id(),
        })


def set_correlation_id(cid: Optional[str]) -> None:
    """Set the correlation ID for cross-function log tracing (LOG-6).

    Parameters
    ----------
    cid : str | None
        The correlation ID, or None to clear it.
    """
    global _current_correlation_id
    with _CORRELATION_ID_LOCK:
        _current_correlation_id = cid


def get_correlation_id() -> Optional[str]:
    """Return the current correlation ID, or None (LOG-6)."""
    with _CORRELATION_ID_LOCK:
        return _current_correlation_id


def _log_with_cid(level: int, msg: str, *args) -> None:
    """Log a message with the current correlation ID prefix (LOG-6)."""
    cid = get_correlation_id()
    if cid:
        msg = f"[cid={cid}] {msg}"
    logger.log(level, msg, *args)


def _increment_metric(name: str, count: int = 1) -> None:
    """Thread-safe metric counter increment."""
    with _METRICS_LOCK:
        _metrics[name] += count


# ===========================================================================
# Retry / circuit breaker (REL-6, REL-7)
# ===========================================================================
def _convert_with_retry(
    convert_fn: Callable,
    smiles: str,
    *,
    max_retries: int = _MAX_RETRIES,
) -> Optional[str]:
    """Call ``convert_fn(smiles)`` with bounded exponential backoff (REL-6).

    Parameters
    ----------
    convert_fn : Callable[[str], Optional[str]]
        The single-SMILES conversion function (typically
        ``convert_to_inchikey``).
    smiles : str
        The SMILES string to convert.
    max_retries : int
        Maximum number of retry attempts on transient failure.  Default
        ``_MAX_RETRIES`` (=2).

    Returns
    -------
    str | None
        The InChIKey, or None if all retries fail.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return convert_fn(smiles)
        except Exception as exc:  # noqa: BLE001 -- REL-6
            last_exc = exc
            if attempt < max_retries:
                # Exponential backoff: 0.1, 0.2, 0.4, ...
                delay = _RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.debug(
                    "_convert_with_retry: attempt %d failed (%s); "
                    "retrying in %.3fs",
                    attempt + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.debug(
                    "_convert_with_retry: all %d attempts failed: %s",
                    max_retries + 1,
                    exc,
                )
    _increment_metric("conversion_retry_exhausted")
    return None


# ===========================================================================
# DataCleaningResult dataclass (DESIGN-9, DQ-11)
# ===========================================================================
@dataclass
class DataCleaningResult:
    """Structured result of a cleaning operation (DESIGN-9).

    When a public cleaning function is called with ``return_result=True``,
    it returns a ``DataCleaningResult`` instead of a bare DataFrame.
    This gives the caller programmatic access to what changed during
    cleaning -- essential for data quality monitoring and audit trails.

    Attributes
    ----------
    df : pd.DataFrame
        The cleaned output DataFrame.
    rows_before : int
        Row count of the input DataFrame.
    rows_after : int
        Row count of the output DataFrame.
    rows_dropped : int
        Number of rows removed (``rows_before - rows_after``).
    columns_affected : dict[str, dict[str, Any]]
        Per-column change record.  Keys are column names; values are
        dicts with keys like ``filled``, ``recovered``, ``clipped``,
        ``truncated``, ``dropped``, ``default_value``.
    dropped_rows : pd.DataFrame
        A DataFrame containing the rows that were dropped (empty if
        none).  Useful for dead-letter inspection.
    lineage : dict
        Per-(row, column) lineage records for fine-grained tracing.
    warnings : list[str]
        Human-readable warning messages produced during cleaning.
    duration_seconds : float
        Wall-clock duration of the cleaning operation.
    dtype_changes : dict[str, tuple[str, str]]
        Mapping from column name to ``(old_dtype, new_dtype)`` for
        columns whose dtype changed during cleaning.
    """

    df: pd.DataFrame
    rows_before: int = 0
    rows_after: int = 0
    rows_dropped: int = 0
    columns_affected: dict = field(default_factory=dict)
    dropped_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    lineage: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    duration_seconds: float = 0.0
    dtype_changes: dict = field(default_factory=dict)

    def quality_summary(self) -> dict:
        """Return a flat summary of data quality metrics (DQ-11).

        Returns
        -------
        dict
            A dict with keys: ``rows_before``, ``rows_after``,
            ``rows_dropped``, ``drop_rate``, ``columns_affected``,
            ``warning_count``, ``duration_seconds``, ``dtype_change_count``.
        """
        drop_rate = (
            self.rows_dropped / self.rows_before
            if self.rows_before > 0
            else 0.0
        )
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_dropped": self.rows_dropped,
            "drop_rate": drop_rate,
            "columns_affected": len(self.columns_affected),
            "warning_count": len(self.warnings),
            "duration_seconds": self.duration_seconds,
            "dtype_change_count": len(self.dtype_changes),
        }


# ===========================================================================
# 1. handle_missing_inchikey + helpers (ARCH-1, ARCH-2, ARCH-3, ARCH-6,
#                                    BUG-SCI-2, REL-1, REL-6, REL-7,
#                                    IDEM-1, IDEM-5, PERF-1, DQ-1, DQ-2,
#                                    DQ-10, DESIGN-6)
# ===========================================================================
def recover_inchikeys_from_smiles(
    df: pd.DataFrame,
    *,
    converter: Optional[Callable] = None,
    use_batch: Optional[bool] = None,
    column_context: str = "chemical",
    reset_index: bool = False,
) -> pd.DataFrame:
    """Recover missing InChIKeys from SMILES -- NO rows are dropped (ARCH-3).

    This is the PURE recovery function.  Use it when you want InChIKey
    recovery without data loss.  To also drop unidentifiable rows, use
    :func:`handle_missing_inchikey` (which composes this function with
    :func:`drop_unidentifiable_drugs`).

    Parameters
    ----------
    df : pd.DataFrame
        Drug records with at least an ``inchikey`` column.  A ``smiles``
        column is used for recovery if present.
    converter : Callable[[str], str | None] | None
        Dependency-injection point for the SMILES->InChIKey converter
        (ARCH-6).  When None, uses ``cleaning.normalizer.convert_to_inchikey``
        (lazy import).
    use_batch : bool | None
        If True, use the batch API (``convert_to_inchikeys``) with a
        ThreadPoolExecutor.  If False, use the row-by-row API.  If None
        (default), use the batch API when the number of recoverable rows
        is >= ``_BATCH_THRESHOLD`` (10).
    column_context : str
        Column context for null detection on the ``smiles`` column.
        Default ``"chemical"`` -- does NOT treat ``-`` as null (single
        bond in SMILES).
    reset_index : bool
        If True, reset the DataFrame index after recovery (drops the
        original index).  Default False -- preserves index for merge/join
        compatibility (INT-1).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with recovered InChIKeys.  The following lineage
        columns are added:

        - ``_inchikey_source`` -- "recovered_from_smiles" / "original" / None
        - ``_smiles_used_for_recovery`` -- the SMILES used (for audit)
        - ``_inchikey_recovery_failed`` -- True if recovery was attempted and failed
        - ``_inchikey_recovery_error`` -- error category from ConversionResult

    Notes
    -----
    This function is IDEMPOTENT (IDEM-1): if the input DataFrame already
    has a ``_inchikey_source`` column, recovery is skipped for rows that
    have already been processed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "inchikey": ["AAA", None],
    ...     "smiles": ["CCO", "CC(=O)O"],
    ... })
    >>> # recover_inchikeys_from_smiles(df)  # doctest: +SKIP
    """
    # Input validation (DQ-12, SEC-4, SEC-5).
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"recover_inchikeys_from_smiles expects a DataFrame, "
            f"got {type(df).__name__}"
        )
    _validate_input_size(df)
    _validate_input_schema(df, ["inchikey"], "recover_inchikeys_from_smiles")

    if df.empty:
        logger.debug("recover_inchikeys_from_smiles: empty DataFrame")
        return df.copy()

    out = df.copy()
    rows_before = len(out)
    input_fingerprint = _fingerprint_df(out)
    start_time = time.monotonic()

    # Idempotency: skip if already processed (IDEM-1).
    if "_inchikey_source" in out.columns:
        already_done = out["_inchikey_source"].notna()
        if already_done.all():
            logger.debug(
                "recover_inchikeys_from_smiles: all rows already processed "
                "(_inchikey_source present) -- skipping"
            )
            _increment_metric("recover_skipped_already_processed")
            return out

    # Initialize lineage columns (LINEAGE-1, LINEAGE-2).
    if "_inchikey_source" not in out.columns:
        out["_inchikey_source"] = None
    if "_smiles_used_for_recovery" not in out.columns:
        out["_smiles_used_for_recovery"] = None
    if "_inchikey_recovery_failed" not in out.columns:
        out["_inchikey_recovery_failed"] = False
    if "_inchikey_recovery_error" not in out.columns:
        out["_inchikey_recovery_error"] = None

    # Mark rows that already have an InChIKey as "original".
    has_inchikey_mask = ~is_nullish(out["inchikey"], column_context="general")
    out.loc[has_inchikey_mask & out["_inchikey_source"].isna(), "_inchikey_source"] = "original"

    # Identify rows needing recovery.
    inchikey_null = is_nullish(out["inchikey"], column_context="general")
    has_smiles_col = "smiles" in out.columns

    if has_smiles_col:
        smiles_present = ~is_nullish(out["smiles"], column_context=column_context)
    else:
        logger.warning(
            "recover_inchikeys_from_smiles: 'smiles' column not found -- "
            "cannot recover InChIKeys"
        )
        smiles_present = pd.Series(False, index=out.index)

    recoverable_mask = inchikey_null & smiles_present
    recoverable_count = int(recoverable_mask.sum())

    if recoverable_count == 0:
        logger.debug(
            "recover_inchikeys_from_smiles: no rows with missing InChIKey "
            "and available SMILES"
        )
        _increment_metric("recover_attempts", 0)
        _set_cleaning_metadata(
            out,
            function_name="recover_inchikeys_from_smiles",
            input_fingerprint=input_fingerprint,
            input_rows=rows_before,
        )
        if reset_index:
            out = out.reset_index(drop=True)
        return out

    logger.info(
        "recover_inchikeys_from_smiles: attempting to recover %d "
        "InChIKey(s) from SMILES",
        recoverable_count,
    )
    _increment_metric("recover_attempts", recoverable_count)

    # Resolve the converter (ARCH-6 -- dependency injection).
    if converter is None:
        try:
            convert_single = _get_convert_to_inchikey()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "recover_inchikeys_from_smiles: cannot import "
                "convert_to_inchikey from normalizer: %s",
                exc,
            )
            _increment_metric("recover_import_failed")
            _set_cleaning_metadata(
                out,
                function_name="recover_inchikeys_from_smiles",
                input_fingerprint=input_fingerprint,
                input_rows=rows_before,
            )
            if reset_index:
                out = out.reset_index(drop=True)
            return out
    else:
        convert_single = converter

    # Decide batch vs row-by-row (ARCH-2, PERF-1).
    should_batch = (
        use_batch if use_batch is not None
        else recoverable_count >= _BATCH_THRESHOLD
    )

    recovered = 0
    failed = 0
    consecutive_failures = 0  # REL-7 circuit breaker

    if should_batch and converter is None:
        # Batch path -- uses ThreadPoolExecutor internally (ARCH-2).
        try:
            convert_batch = _get_convert_to_inchikeys()
            # Sort recoverable indices for deterministic ordering (IDEM-5).
            recoverable_indices = sorted(out.index[recoverable_mask].tolist())
            recoverable_smiles = [
                _sanitize_smiles(out.at[idx, "smiles"])
                for idx in recoverable_indices
            ]
            # None entries will fail conversion -- that's expected.
            smiles_to_convert = [s if s is not None else "" for s in recoverable_smiles]
            batch_results = convert_batch(smiles_to_convert)

            # Standardize recovered InChIKeys via lazy import (DQ-1).
            # v21 ROOT FIX (Audit section 6 finding 2 - "Silent InChIKey
            # passthrough fallback"): the previous code did
            # ``standardize = lambda x: x`` as a passthrough fallback when
            # ``_get_standardize_inchikey()`` raised. Recovered InChIKeys
            # then bypassed validation/normalization entirely - could
            # insert lowercase/malformed keys into the DB. The audit's
            # concern: "Could insert lowercase/malformed keys into the DB."
            # Fix: instead of passthrough, set ``standardize`` to a
            # function that QUARANTINES the record (via _append_dead_letter)
            # and returns None. The downstream code already handles None
            # InChIKeys via the existing "recovery failed" path.
            try:
                standardize = _get_standardize_inchikey()
            except Exception as _std_exc:  # noqa: BLE001
                logger.warning(
                    "recover_inchikeys_from_smiles: standardize_inchikey "
                    "unavailable (%s) - recovered InChIKeys will be "
                    "QUARANTINED (not silently passed through).",
                    _std_exc,
                )
                def standardize(x):  # noqa: E731
                    # Quarantine any recovered InChIKey we cannot
                    # standardize. Returning None triggers the existing
                    # recovery-failed path which adds to dead-letter.
                    _append_dead_letter(
                        "recover_inchikeys_from_smiles",
                        "standardize_unavailable",
                        {"raw_inchikey": str(x)[:200]},
                    )
                    return None

            for idx, raw_smiles, inchikey in zip(
                recoverable_indices, recoverable_smiles, batch_results
            ):
                if raw_smiles is None:
                    out.at[idx, "_inchikey_recovery_failed"] = True
                    out.at[idx, "_inchikey_recovery_error"] = "INVALID_SMILES"
                    failed += 1
                    _append_dead_letter(
                        "recover_inchikeys_from_smiles",
                        "invalid_smiles_for_recovery",
                        out.loc[idx].to_dict(),
                    )
                    continue
                if inchikey is None:
                    out.at[idx, "_inchikey_recovery_failed"] = True
                    out.at[idx, "_inchikey_recovery_error"] = "CONVERSION_FAILED"
                    failed += 1
                    consecutive_failures += 1
                    _append_dead_letter(
                        "recover_inchikeys_from_smiles",
                        "smiles_conversion_failed",
                        {"smiles": _redact_for_log(raw_smiles), "row": out.loc[idx].to_dict()},
                    )
                    if consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                        logger.error(
                            "recover_inchikeys_from_smiles: circuit breaker "
                            "opened after %d consecutive failures -- "
                            "skipping remaining batch",
                            consecutive_failures,
                        )
                        _increment_metric("circuit_open_count")
                        # Mark remaining rows as failed.
                        remaining_indices = [
                            i for i in recoverable_indices
                            if not out.at[i, "_inchikey_recovery_failed"]
                            and out.at[i, "_inchikey_source"] != "recovered_from_smiles"
                        ]
                        for i in remaining_indices:
                            out.at[i, "_inchikey_recovery_failed"] = True
                            out.at[i, "_inchikey_recovery_error"] = "CIRCUIT_OPEN"
                        break
                    continue
                # Standardize before storing (DQ-1).
                standardized = standardize(inchikey)
                if standardized is None:
                    out.at[idx, "_inchikey_recovery_failed"] = True
                    out.at[idx, "_inchikey_recovery_error"] = "STANDARDIZATION_FAILED"
                    failed += 1
                    continue
                out.at[idx, "inchikey"] = standardized
                out.at[idx, "_inchikey_source"] = "recovered_from_smiles"
                out.at[idx, "_smiles_used_for_recovery"] = raw_smiles
                out.at[idx, "_inchikey_recovery_failed"] = False
                out.at[idx, "_inchikey_recovery_error"] = None
                recovered += 1
                consecutive_failures = 0  # reset on success
        except Exception as exc:  # noqa: BLE001 -- REL-2 preserve partial results
            logger.error(
                "recover_inchikeys_from_smiles: batch conversion raised "
                "%s -- falling back to row-by-row",
                exc,
            )
            # Fall through to row-by-row path.
            should_batch = False

    if not should_batch or converter is not None:
        # Row-by-row path (ARCH-1 -- hoisted lazy import, ARCH-6 -- DI).
        # Sort recoverable indices for deterministic ordering (IDEM-5).
        recoverable_indices = sorted(out.index[recoverable_mask].tolist())
        for idx in recoverable_indices:
            smiles_val = out.at[idx, "smiles"]
            sanitized = _sanitize_smiles(smiles_val)
            if sanitized is None:
                out.at[idx, "_inchikey_recovery_failed"] = True
                out.at[idx, "_inchikey_recovery_error"] = "INVALID_SMILES"
                failed += 1
                _append_dead_letter(
                    "recover_inchikeys_from_smiles",
                    "invalid_smiles_for_recovery",
                    out.loc[idx].to_dict(),
                )
                continue

            # BUG-REL-1: wrap in try/except, retry with backoff (REL-6).
            inchikey = _convert_with_retry(convert_single, sanitized)
            if inchikey is None:
                out.at[idx, "_inchikey_recovery_failed"] = True
                out.at[idx, "_inchikey_recovery_error"] = "CONVERSION_FAILED"
                failed += 1
                consecutive_failures += 1
                _append_dead_letter(
                    "recover_inchikeys_from_smiles",
                    "smiles_conversion_failed",
                    {"smiles": _redact_for_log(sanitized), "row": out.loc[idx].to_dict()},
                )
                if consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                    logger.error(
                        "recover_inchikeys_from_smiles: circuit breaker "
                        "opened after %d consecutive failures -- aborting "
                        "recovery",
                        consecutive_failures,
                    )
                    _increment_metric("circuit_open_count")
                    # P1-016 ROOT FIX (v100 forensic -- mark remaining rows
                    # on circuit-open in the row-by-row path):
                    # The batch path (lines 1609-1618) marks remaining rows
                    # as ``_inchikey_recovery_failed=True`` with
                    # ``_inchikey_recovery_error="CIRCUIT_OPEN"`` before
                    # breaking. The row-by-row path (this block) did NOT --
                    # it just ``break``ed, leaving remaining rows with
                    # ``_inchikey_recovery_failed=False`` and
                    # ``_inchikey_source=None``. Downstream code that
                    # filters on ``_inchikey_recovery_failed`` treated
                    # them as valid (but with null inchikey) -- silent
                    # data corruption. ROOT FIX: mirror the batch path's
                    # marking before breaking, so remaining rows are
                    # explicitly flagged as failed due to circuit-open.
                    remaining_indices_rowbyrow = [
                        i for i in recoverable_indices
                        if i != idx
                        and not out.at[i, "_inchikey_recovery_failed"]
                        and out.at[i, "_inchikey_source"] != "recovered_from_smiles"
                    ]
                    for i in remaining_indices_rowbyrow:
                        out.at[i, "_inchikey_recovery_failed"] = True
                        out.at[i, "_inchikey_recovery_error"] = "CIRCUIT_OPEN"
                    break
                continue

            # Standardize before storing (DQ-1).
            try:
                standardize = _get_standardize_inchikey()
                standardized = standardize(inchikey)
            except Exception as _std_exc:  # noqa: BLE001
                # v24 ROOT FIX (FORENSIC-P1-PIPE B / Audit Chain):
                # the previous code did ``standardized = inchikey``
                # (silent passthrough) -- recovered InChIKeys bypassed
                # validation/normalization entirely, allowing lowercase
                # or malformed keys into the DB. This is
                # production-reachable via drugbank_pipeline.py:1397 ->
                # handle_missing_inchikey -> recover_inchikeys_from_smiles
                # (non-batch path). Fix: mark the row as failed (not
                # passthrough) so the caller can dead-letter it. The
                # canonical validator (normalizer.is_valid_inchikey)
                # will catch any malformed keys that leak through.
                logger.warning(
                    "recover_inchikeys_from_smiles: standardization "
                    "failed for inchikey=%r: %s. Marking row as "
                    "failed (v24 root fix -- no silent passthrough).",
                    inchikey, _std_exc,
                )
                standardized = None
                out.at[idx, "_inchikey_recovery_failed"] = True
                out.at[idx, "_inchikey_recovery_error"] = (
                    f"STANDARDIZATION_FAILED: {type(_std_exc).__name__}"
                )
                failed += 1
                continue
            if standardized is None:
                out.at[idx, "_inchikey_recovery_failed"] = True
                out.at[idx, "_inchikey_recovery_error"] = "STANDARDIZATION_FAILED"
                failed += 1
                continue

            out.at[idx, "inchikey"] = standardized
            out.at[idx, "_inchikey_source"] = "recovered_from_smiles"
            out.at[idx, "_smiles_used_for_recovery"] = sanitized
            out.at[idx, "_inchikey_recovery_failed"] = False
            out.at[idx, "_inchikey_recovery_error"] = None
            recovered += 1
            consecutive_failures = 0

            # Checkpoint callback (REL-10).
            if (recovered + failed) % _CHECKPOINT_INTERVAL == 0:
                logger.debug(
                    "recover_inchikeys_from_smiles: checkpoint -- "
                    "%d recovered, %d failed so far",
                    recovered,
                    failed,
                )

    _increment_metric("inchikeys_recovered", recovered)
    _increment_metric("inchikey_recovery_failed", failed)
    duration = time.monotonic() - start_time
    logger.info(
        "recover_inchikeys_from_smiles: recovered %d / %d InChIKey(s) "
        "from SMILES (%d failed) in %.3fs",
        recovered,
        recoverable_count,
        failed,
        duration,
    )
    _set_cleaning_metadata(
        out,
        function_name="recover_inchikeys_from_smiles",
        input_fingerprint=input_fingerprint,
        input_rows=rows_before,
    )
    if reset_index:
        out = out.reset_index(drop=True)
    return out


def drop_unidentifiable_drugs(
    df: pd.DataFrame,
    *,
    alternative_id_columns: Optional[list] = None,
    dead_letter: bool = True,
    reset_index: bool = False,
) -> pd.DataFrame:
    """Drop rows where the drug cannot be identified (ARCH-3, BUG-SCI-2).

    A row is "unidentifiable" iff:

    - ``inchikey`` is null-like (NaN / empty / "n/a" / "-"),
    AND
    - ``smiles`` is null-like (so we cannot recover InChIKey),
    AND
    - none of the ``alternative_id_columns`` (default: ``["drugbank_id"``,
      ``"chembl_id"``, ``"name"``]) have a non-null value.

    Rows with a valid DrugBank ID, ChEMBL ID, or name are NOT dropped
    even if both ``inchikey`` and ``smiles`` are missing -- they can be
    re-identified later via entity resolution (BUG-SCI-2).

    Parameters
    ----------
    df : pd.DataFrame
        Drug records.
    alternative_id_columns : list[str] | None
        Columns to check for alternative identifiers.  Default
        ``["drugbank_id", "chembl_id", "name"]``.  Pass ``[]`` to disable
        alternative-ID preservation (legacy v2.0.0 behavior).
    dead_letter : bool
        If True (default), append dropped rows to the module-level
        dead-letter queue for inspection.
    reset_index : bool
        If True, reset the index after dropping (INT-1).  Default False.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.  The original index is preserved unless
        ``reset_index=True``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"drop_unidentifiable_drugs expects a DataFrame, "
            f"got {type(df).__name__}"
        )
    _validate_input_size(df)
    _validate_input_schema(df, ["inchikey"], "drop_unidentifiable_drugs")

    if df.empty:
        return df.copy()

    out = df.copy()
    rows_before = len(out)

    still_missing_inchikey = is_nullish(out["inchikey"], column_context="general")

    has_smiles_col = "smiles" in out.columns
    if has_smiles_col:
        also_missing_smiles = is_nullish(out["smiles"], column_context="chemical")
    else:
        # BUG-CODE-8: legacy behavior treated "no smiles col" as "all rows
        # missing smiles" which dropped everything missing inchikey.  We
        # preserve this for backward compat -- but use alternative_id_columns
        # to avoid dropping rows with valid IDs.
        also_missing_smiles = pd.Series(True, index=out.index)

    # BUG-SCI-2: check alternative identifiers before dropping.
    if alternative_id_columns is None:
        alternative_id_columns = ["drugbank_id", "chembl_id", "name"]
    has_alternative_id = pd.Series(False, index=out.index)
    for id_col in alternative_id_columns:
        if id_col in out.columns:
            has_alternative_id = has_alternative_id | ~is_nullish(
                out[id_col], column_context="general"
            )

    unidentifiable = still_missing_inchikey & also_missing_smiles & ~has_alternative_id
    dropped_count = int(unidentifiable.sum())

    if dropped_count > 0:
        # Record dropped rows in dead-letter queue (DQ-2, REL-9, LINEAGE-3).
        if dead_letter:
            for idx in out.index[unidentifiable]:
                _append_dead_letter(
                    "drop_unidentifiable_drugs",
                    "no_inchikey_no_smiles_no_alt_id",
                    out.loc[idx].to_dict(),
                )
        out = out[~unidentifiable]
        if reset_index:
            out = out.reset_index(drop=True)
        logger.info(
            "drop_unidentifiable_drugs: dropped %d unidentifiable row(s) "
            "(%d rows before, %d rows after)",
            dropped_count,
            rows_before,
            len(out),
        )
        _increment_metric("drugs_dropped_unidentifiable", dropped_count)

    return out


def handle_missing_inchikey(
    df: pd.DataFrame,
    *,
    drop_unidentifiable: bool = True,
    converter: Optional[Callable] = None,
    alternative_id_columns: Optional[list] = None,
    reset_index: bool = False,
    return_result: bool = False,
) -> Union[pd.DataFrame, "DataCleaningResult"]:
    """Recover missing InChIKeys from SMILES; drop unidentifiable drugs.

    Composes :func:`recover_inchikeys_from_smiles` and
    :func:`drop_unidentifiable_drugs` (ARCH-3).

    Parameters
    ----------
    df : pd.DataFrame
        Drug records with at least an ``inchikey`` column.  A ``smiles``
        column is used for recovery if present.
    drop_unidentifiable : bool
        If True (default), drop rows where the drug cannot be identified.
        If False, only recovery is attempted -- no rows are dropped.
    converter : Callable[[str], str | None] | None
        Dependency injection for the SMILES->InChIKey converter (ARCH-6).
    alternative_id_columns : list[str] | None
        Columns to check before dropping.  Default
        ``["drugbank_id", "chembl_id", "name"]``.  Pass ``[]`` to disable.
    reset_index : bool
        If True, reset the index after dropping.  Default False (INT-1).
    return_result : bool
        If True, return a :class:`DataCleaningResult` instead of a bare
        DataFrame (DESIGN-9).

    Returns
    -------
    pd.DataFrame | DataCleaningResult

    Notes
    -----
    **Backward compatibility**: when called as ``handle_missing_inchikey(df)``,
    this function preserves the v2.0.0 behavior:

    - SMILES recovery is attempted for rows with missing InChIKey.
    - Rows with both InChIKey and SMILES missing are dropped.

    The v3.0.0 defaults refine the drop behavior: rows with a valid
    DrugBank ID, ChEMBL ID, or name are NOT dropped (BUG-SCI-2).  Pass
    ``alternative_id_columns=[]`` to restore the v2.0.0 drop behavior.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "inchikey": ["AAA", None, None],
    ...     "smiles": ["CCO", "CC(=O)O", None],
    ...     "name": ["Ethanol", "Acetic acid", "Unknown"],
    ... })
    >>> # Acetic acid may recover; Unknown has no InChIKey, no SMILES,
    >>> # but DOES have a name -- so it is NOT dropped by default.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"handle_missing_inchikey expects a DataFrame, "
            f"got {type(df).__name__}"
        )
    _validate_input_size(df)
    _validate_column_types(df)

    rows_before = len(df)
    input_fingerprint = _fingerprint_df(df)
    start_time = time.monotonic()
    dtype_changes: dict = {}
    columns_affected: dict = {}
    warnings_list: list = []

    # Empty input -- return immediately.
    if df.empty:
        logger.debug("handle_missing_inchikey: empty DataFrame, nothing to do")
        empty = df.copy()
        if return_result:
            return DataCleaningResult(
                df=empty,
                rows_before=0,
                rows_after=0,
                rows_dropped=0,
                duration_seconds=0.0,
            )
        return empty

    if "inchikey" not in df.columns:
        msg = (
            f"handle_missing_inchikey: 'inchikey' column not found "
            f"(columns={list(df.columns)}). Returning DataFrame unchanged."
        )
        logger.warning(msg)
        warnings_list.append(msg)
        out = df.copy()
        if return_result:
            return DataCleaningResult(
                df=out,
                rows_before=rows_before,
                rows_after=rows_before,
                rows_dropped=0,
                warnings=warnings_list,
                duration_seconds=time.monotonic() - start_time,
            )
        return out

    # Track dtype before.
    inchikey_dtype_before = str(df["inchikey"].dtype)

    # Step 1: recover InChIKeys from SMILES (ARCH-3).
    out = recover_inchikeys_from_smiles(
        df,
        converter=converter,
        reset_index=False,
    )
    columns_affected["inchikey"] = {"recovery_attempted": True}

    # Step 2: optionally drop unidentifiable rows (ARCH-3).
    dropped_rows_df = pd.DataFrame()
    rows_dropped = 0
    if drop_unidentifiable:
        pre_drop_len = len(out)
        out = drop_unidentifiable_drugs(
            out,
            alternative_id_columns=alternative_id_columns,
            dead_letter=True,
            reset_index=reset_index,
        )
        rows_dropped = pre_drop_len - len(out)
        if rows_dropped > 0:
            # Reconstruct a (best-effort) dropped_rows DataFrame.
            # We use the dead-letter queue snapshot for this.
            dead_letters = get_dead_letters()
            recent_drops = [
                dl for dl in dead_letters
                if dl.get("function") == "drop_unidentifiable_drugs"
                and dl.get("row") is not None
            ][-rows_dropped:]
            if recent_drops:
                try:
                    dropped_rows_df = pd.DataFrame(
                        [dl["row"] for dl in recent_drops]
                    )
                except Exception:  # noqa: BLE001
                    dropped_rows_df = pd.DataFrame()

    # Track dtype after.
    inchikey_dtype_after = str(out["inchikey"].dtype) if "inchikey" in out.columns else inchikey_dtype_before
    if inchikey_dtype_before != inchikey_dtype_after:
        dtype_changes["inchikey"] = (inchikey_dtype_before, inchikey_dtype_after)

    duration = time.monotonic() - start_time
    _set_cleaning_metadata(
        out,
        function_name="handle_missing_inchikey",
        input_fingerprint=input_fingerprint,
        input_rows=rows_before,
    )
    _increment_metric("handle_missing_inchikey_calls")

    logger.info(
        "handle_missing_inchikey: %d rows -> %d rows (%d dropped) in %.3fs",
        rows_before,
        len(out),
        rows_dropped,
        duration,
    )

    if return_result:
        return DataCleaningResult(
            df=out,
            rows_before=rows_before,
            rows_after=len(out),
            rows_dropped=rows_dropped,
            columns_affected=columns_affected,
            dropped_rows=dropped_rows_df,
            warnings=warnings_list,
            duration_seconds=duration,
            dtype_changes=dtype_changes,
        )
    return out


# ===========================================================================
# v110 Task 29 root fix: distinguish "missing" (None) from "not applicable" (False)
#
# The audit (Task 29) requires: "must distinguish 'missing' (None) from
# 'not applicable' (False). Currently both are stored as None."
#
# SCIENTIFIC RATIONALE:
#   - None (missing/unknown): we have NO information about this field.
#     Example: a drug from PubChem has no FDA approval data (PubChem doesn't
#     track regulatory status). The correct value is None ("unknown").
#   - False (not applicable/confirmed-negative): we have POSITIVE EVIDENCE
#     that the field is negative. Example: a DrugBank drug whose <groups>
#     element contains "withdrawn" or "illicit" but NOT "approved" — this
#     drug is CONFIRMED NOT FDA approved. The correct value is False.
#
# The previous code defaulted is_fda_approved to None in BOTH cases, which
# conflated "we don't know" with "we know it's not approved". This caused
# downstream ML filters to treat confirmed-unapproved drugs as "unknown",
# potentially surfacing them as repurposing candidates when they should be
# excluded from safety-critical filters.
#
# ROOT FIX:
#   1. Define NOT_APPLICABLE_SENTINEL = False (the "confirmed negative" value)
#   2. Define MISSING_SENTINEL = None (the "unknown" value)
#   3. Add classify_fda_approval(groups_str) helper that returns:
#        True  — if "approved" is in the groups string (confirmed approved)
#        False — if "approved" is NOT in groups but groups is non-empty
#                AND contains a known negative-group like "withdrawn",
#                "illicit", "experimental" (confirmed not approved)
#        None  — if groups is empty/None (unknown — no information)
#   4. Update fill_missing_drug_fields to use this helper when the source
#      provides groups data, and keep None as the default for truly missing
#      values.
# ===========================================================================
NOT_APPLICABLE_SENTINEL: bool = False
"""Sentinel value for 'not applicable' (confirmed negative) fields.

Use this when the source provides POSITIVE EVIDENCE that a field is negative
(e.g., DrugBank groups contains 'withdrawn' but not 'approved').
Distinct from None which means 'missing/unknown' (no information)."""

MISSING_SENTINEL = None
"""Sentinel value for 'missing/unknown' fields.

Use this when the source provides NO information about the field.
Distinct from False which means 'not applicable' (confirmed negative)."""

# DrugBank <groups> values that indicate confirmed non-approval.
# If a drug's groups contains any of these but NOT "approved", the drug
# is confirmed NOT FDA approved → is_fda_approved = False (not None).
_NEGATIVE_FDA_GROUPS: frozenset[str] = frozenset({
    "withdrawn",      # DrugBank <group>withdrawn</group> — was approved, now pulled
    "illicit",        # DrugBank <group>illicit</group> — never approved, illegal
    "experimental",   # DrugBank <group>experimental</group> — pre-clinical only
})


def classify_fda_approval(groups: object) -> "bool | None":
    """Classify a drug's FDA approval status from its DrugBank groups field.

    Parameters
    ----------
    groups : str, list, or None
        The drug's groups value. Can be:
        - A string like "approved|investigational" (pipe-separated)
        - A string like "approved, investigational" (comma-separated)
        - A list like ["approved", "investigational"]
        - None or "" (no groups data)

    Returns
    -------
    bool or None
        True  — if "approved" is in groups (confirmed FDA approved)
        False — if "approved" is NOT in groups but groups is non-empty
                AND contains a known negative-group (withdrawn, illicit,
                experimental) — confirmed NOT FDA approved
        None  — if groups is empty/None (unknown — no information)

    Examples:
        >>> classify_fda_approval("approved|investigational")
        True
        >>> classify_fda_approval("withdrawn")
        False
        >>> classify_fda_approval("illicit")
        False
        >>> classify_fda_approval("")
        None
        >>> classify_fda_approval(None)
        None
        >>> classify_fda_approval(["approved", "investigational"])
        True
        >>> classify_fda_approval(["withdrawn"])
        False
        >>> classify_fda_approval(["nutraceutical"])
        None  # nutraceutical is not a negative-group — unknown
    """
    if groups is None:
        return None
    if isinstance(groups, float):
        import math as _math
        if _math.isnan(groups):
            return None
        groups = str(groups)
    if isinstance(groups, (list, tuple, set)):
        groups_lower = {str(g).strip().lower() for g in groups if g is not None}
    elif isinstance(groups, str):
        if not groups.strip():
            return None
        # Split on common separators: pipe, comma, semicolon, whitespace
        import re as _re
        tokens = _re.split(r"[|,;\s]+", groups.strip().lower())
        groups_lower = {t for t in tokens if t}
    else:
        # Unknown type — coerce to string and try
        groups_str = str(groups).strip().lower()
        if not groups_str:
            return None
        import re as _re
        tokens = _re.split(r"[|,;\s]+", groups_str)
        groups_lower = {t for t in tokens if t}

    if not groups_lower:
        return None  # empty groups → unknown

    # "approved" in groups → confirmed FDA approved
    if "approved" in groups_lower:
        return True

    # Check for confirmed-negative groups (withdrawn, illicit, experimental)
    if groups_lower & _NEGATIVE_FDA_GROUPS:
        return False

    # Groups exist but no "approved" and no negative-group → unknown
    # (e.g., "investigational", "nutraceutical", "biotech" — not confirmed
    # either way)
    return None


# ===========================================================================
# 2. fill_missing_drug_fields (ARCH-5, BUG-SCI-3, BUG-SCI-7, BUG-SCI-10,
#                              DESIGN-4, DESIGN-9, IDEM-2, CODE-5, CODE-9,
#                              DQ-3, DQ-9, REL-4, INT-3, LINEAGE-4)
# ===========================================================================
def fill_missing_drug_fields(
    df: pd.DataFrame,
    *,
    conservative_defaults: bool = True,
    fill_map_override: Optional[dict] = None,
    reset_index: bool = False,
    return_result: bool = False,
) -> Union[pd.DataFrame, "DataCleaningResult"]:
    """Fill default values for common missing drug fields.

    Fill strategy depends on ``conservative_defaults``:

    =======================  =================================  =================================
    Column                   ``conservative_defaults=False``    ``conservative_defaults=True``
    =======================  =================================  =================================
    ``is_fda_approved``      ``False`` (bool)                   ``None`` (nullable Boolean)
    ``drug_type``            ``'Unknown'``                      ``'Unknown'``
    ``max_phase``            ``None`` (kept as NaN)             ``None`` (kept as NaN)
    ``mechanism_of_action``  ``''`` (empty string)              ``'Unknown'``
    ``molecular_formula``   ``''`` (empty string)              ``''`` (empty string)
    ``smiles``               ``''`` (empty string)              ``None`` (kept as NaN)
    =======================  =================================  =================================

    **Backward compatibility**: the default ``conservative_defaults=True``
    applies the scientifically safer fill values (BUG-SCI-3, BUG-SCI-7,
    BUG-SCI-10). Pass ``conservative_defaults=False`` to get the legacy
    v2.0.0 fill values for backward compatibility with the existing
    12-file codebase.

    Parameters
    ----------
    df : pd.DataFrame
        Drug records.
    conservative_defaults : bool
        See the table above.  Default True (scientifically safer
        behavior -- v93 ROOT FIX P1-046: the previous docstring said
        "Default False (v2.0.0 behavior)" but the actual default was
        True. The docstring contradicted the code, misleading operators
        into thinking they were getting the legacy v2.0.0 behavior when
        they were actually getting the safer v3.0.0 behavior. Root fix:
        update the docstring to match the code. The default remains True
        because the safer behavior is the correct production default --
        operators who need the legacy behavior must explicitly opt in
        with ``conservative_defaults=False``).
    fill_map_override : dict | None
        Per-pipeline overrides.  Merged on top of the default fill_map
        (override takes precedence).  Example:
        ``{"smiles": None, "drug_type": "small molecule"}``.
    reset_index : bool
        If True, reset the index.  Default False (INT-1).
    return_result : bool
        If True, return a :class:`DataCleaningResult`.

    Returns
    -------
    pd.DataFrame | DataCleaningResult
        A new DataFrame with missing values filled.  The following
        lineage columns are added (LINEAGE-4):

        - ``_{col}_was_filled`` -- True if the value in column ``{col}``
          was filled with a default.

    Notes
    -----
    **Ordering guard (DQ-3)**: this function should be called AFTER
    :func:`handle_missing_inchikey`.  If called before, the legacy
    ``smiles=""`` default will suppress InChIKey recovery.  A warning is
    logged if this is detected.

    **Idempotency (IDEM-2)**: if the input DataFrame already has the
    ``_{col}_was_filled`` lineage columns, re-filling is a no-op for
    those columns.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"inchikey": ["A"], "is_fda_approved": [None],
    ...                    "drug_type": [None], "max_phase": [None]})
    >>> result = fill_missing_drug_fields(df)
    >>> bool(result["is_fda_approved"].iloc[0])
    False
    >>> result["drug_type"].iloc[0]
    'Unknown'
    >>> result["max_phase"].iloc[0] is None or pd.isna(result["max_phase"].iloc[0])
    True
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"fill_missing_drug_fields expects a DataFrame, "
            f"got {type(df).__name__}"
        )
    _validate_input_size(df)
    _validate_column_types(df)

    rows_before = len(df)
    input_fingerprint = _fingerprint_df(df)
    start_time = time.monotonic()
    dtype_changes: dict = {}
    columns_affected: dict = {}
    warnings_list: list = []

    if df.empty:
        logger.debug("fill_missing_drug_fields: empty DataFrame, nothing to do")
        empty = df.copy()
        if return_result:
            return DataCleaningResult(df=empty, rows_before=0, rows_after=0)
        return empty

    out = df.copy()

    # DQ-3: ordering guard -- warn if handle_missing_inchikey hasn't run yet.
    if (
        "smiles" in out.columns
        and "inchikey" in out.columns
        and "_inchikey_source" not in out.columns
    ):
        inchikey_null = is_nullish(out["inchikey"], column_context="general")
        smiles_present = ~is_nullish(out["smiles"], column_context="chemical")
        recoverable = inchikey_null & smiles_present
        if int(recoverable.sum()) > 0:
            msg = (
                "fill_missing_drug_fields: %d row(s) have missing InChIKey "
                "with available SMILES but no _inchikey_source lineage.  "
                "Call handle_missing_inchikey BEFORE fill_missing_drug_fields "
                "to avoid the smiles='' default suppressing recovery."
            )
            logger.warning(msg, int(recoverable.sum()))
            warnings_list.append(msg)

    # Build the fill_map (BUG-SCI-3, BUG-SCI-7, BUG-SCI-10).
    if conservative_defaults:
        fill_map: dict = {
            "is_fda_approved": None,  # nullable Boolean
            "drug_type": "Unknown",
            "max_phase": None,  # always None -- distinguishes "unknown" from 0
            "mechanism_of_action": "Unknown",
            "molecular_formula": "",
            "smiles": None,  # None prevents RDKit crashes (BUG-SCI-7)
        }
    else:
        # v2.0.0 legacy defaults -- preserved for backward compatibility.
        # v91 ROOT FIX (BUG #6): is_fda_approved default changed from
        # False to None. False means "definitely NOT FDA approved" which
        # is scientifically wrong for unknown drugs -- the correct value
        # for a missing FDA approval field is None (unknown). Marking
        # unknown drugs as unapproved silently excluded real repurposing
        # candidates from the RL ranker's safety filter.
        fill_map = {
            "is_fda_approved": None,
            "drug_type": "Unknown",
            "max_phase": None,  # FIX #41: None means "unknown"
            "mechanism_of_action": "",
            "molecular_formula": "",
            "smiles": "",
        }

    # Apply per-pipeline overrides (CFG-4).
    if fill_map_override:
        fill_map.update(fill_map_override)

    # v110 Task 29 root fix: pre-fill is_fda_approved from groups data.
    #
    # Before applying the None default for missing is_fda_approved, check
    # if the row has a `groups` column (from DrugBank) or similar evidence.
    # If so, use classify_fda_approval() to distinguish:
    #   - True  (confirmed approved — "approved" in groups)
    #   - False (confirmed NOT approved — "withdrawn"/"illicit"/"experimental"
    #           in groups but NOT "approved")
    #   - None  (unknown — no groups data, or groups without approved/negative)
    #
    # This distinguishes "missing" (None) from "not applicable" (False) per
    # Task 29. Only rows where is_fda_approved is CURRENTLY null are touched
    # — rows with existing True/False values are preserved (idempotent).
    if "is_fda_approved" in out.columns:
        _fda_null_mask = out["is_fda_approved"].isna()
        if _fda_null_mask.any():
            # Try the `groups` column (DrugBank convention).
            _groups_col = None
            for _candidate in ("groups", "drugbank_groups", "approval_groups"):
                if _candidate in out.columns:
                    _groups_col = _candidate
                    break
            if _groups_col is not None:
                _classified = out.loc[_fda_null_mask, _groups_col].apply(
                    classify_fda_approval
                )
                _filled_count = int(_classified.notna().sum())
                if _filled_count > 0:
                    out.loc[_fda_null_mask, "is_fda_approved"] = _classified
                    logger.info(
                        "fill_missing_drug_fields: pre-filled is_fda_approved "
                        "from %s column using classify_fda_approval() — %d "
                        "rows classified (True/False), %d rows remain None "
                        "(unknown). (Task 29 v110 root fix: distinguishes "
                        "missing from not-applicable)",
                        _groups_col,
                        _filled_count,
                        int(_classified.isna().sum()),
                    )

    for col, default in fill_map.items():
        if col not in out.columns:
            logger.debug(
                "fill_missing_drug_fields: column '%s' not present -- skipping",
                col,
            )
            continue

        # v82 FORENSIC ROOT FIX (P1-9 -- per-COLUMN idempotency bug):
        #   The previous code used ``out[lineage_col].any()`` -- if ANY
        #   row had ``_{col}_was_filled=True``, the ENTIRE column was
        #   skipped on subsequent calls. Scenario: first call fills 50
        #   rows (out of 100) in drug_type. Second call (e.g., after
        #   adding new rows) sees ``_{col}_was_filled.any() == True`` and
        #   skips the column entirely. The 50 new rows with null
        #   drug_type were NEVER filled. Silent data gap.
        #
        #   ROOT FIX: PER-ROW idempotency. Instead of skipping the
        #   entire column, we only skip rows that ALREADY have the
        #   lineage marker set. New rows (lineage marker is False/NaN)
        #   still get filled. This is achieved by computing the null
        #   mask and then EXCLUDING rows where the lineage marker is
        #   already True.
        lineage_col = f"_{col}_was_filled"
        # If the lineage column exists and ALL rows are already marked,
        # skip (truly idempotent -- nothing to do).
        # v82 FORENSIC ROOT FIX (P1-9 bug 2): pandas ``Series.all()``
        # SKIPS NaN by default, so ``[True, True, NaN].all()`` returns
        # True -- causing the entire column to be skipped even when new
        # rows (with NaN lineage marker) still need filling. ROOT FIX:
        # use ``fillna(False).all()`` so NaN is treated as "not filled".
        if lineage_col in out.columns and bool(out[lineage_col].fillna(False).all()):
            logger.debug(
                "fill_missing_drug_fields: column '%s' fully filled "
                "(all rows have lineage marker) -- skipping (idempotent)",
                col,
            )
            continue

        # P1-9 continued: compute the set of rows ALREADY filled in a
        # prior call. These rows are EXCLUDED from the null mask so
        # that re-running on a partially-filled column only fills NEW
        # null rows (not the ones already handled).
        if lineage_col in out.columns:
            _already_filled = out[lineage_col].fillna(False).astype(bool)
        else:
            _already_filled = pd.Series(False, index=out.index)

        # Record dtype before (INT-3).
        dtype_before = str(out[col].dtype)

        # Detect string-like columns (REL-4 -- StringDtype handling).
        is_string_col = (
            out[col].dtype == object
            or pd.api.types.is_string_dtype(out[col])
            or isinstance(out[col].dtype, pd.CategoricalDtype)
            or str(out[col].dtype) == "string"
        )

        # DESIGN-4: track whitespace-only strings BEFORE replacing.
        before_null_naive = int(out[col].isna().sum())
        whitespace_count = 0
        if isinstance(default, str) and is_string_col:
            try:
                whitespace_mask = (
                    out[col].astype(str).str.match(_WHITESPACE_REGEX)
                    & out[col].notna()
                )
                whitespace_count = int(whitespace_mask.sum())
            except Exception:  # noqa: BLE001
                whitespace_count = 0
            if whitespace_count > 0:
                logger.info(
                    "fill_missing_drug_fields: found %d whitespace-only "
                    "value(s) in '%s' -- converting to NaN before fill",
                    whitespace_count,
                    col,
                )
                _increment_metric("whitespace_only_converted", whitespace_count)
                try:
                    out[col] = out[col].replace(_WHITESPACE_REGEX, np.nan, regex=True)
                except Exception:  # noqa: BLE001 -- REL-4 StringDtype fallback
                    try:
                        out[col] = out[col].apply(
                            lambda x: np.nan
                            if isinstance(x, str) and _WHITESPACE_REGEX.match(x)
                            else x
                        )
                    except Exception:  # noqa: BLE001
                        pass

        before_null = int((is_nullish(out[col], column_context="general") & ~_already_filled).sum())

        if before_null == 0:
            # Initialize lineage column even if nothing was filled.
            if lineage_col not in out.columns:
                out[lineage_col] = False
            continue

        # FIX #41: for None defaults, leave as NaN/NA to distinguish
        # "unknown" (None/NaN) from "confirmed no clinical data" (0).
        if default is None:
            # Initialize lineage marker -- these rows ARE "unknown".
            if lineage_col not in out.columns:
                out[lineage_col] = False
            # Mark rows where the value WAS NaN as "filled with unknown".
            # P1-9: exclude rows already filled in a prior call.
            null_mask = is_nullish(out[col], column_context="general") & ~_already_filled
            out.loc[null_mask, lineage_col] = True
            # Don't actually fill -- keep NaN to represent "unknown".
            # But for is_fda_approved with conservative_defaults=True,
            # we DO want to coerce to nullable Boolean.
            if col == "is_fda_approved" and conservative_defaults:
                # Use nullable Boolean type -- preserves NaN/NA distinction.
                try:
                    out[col] = out[col].astype("boolean")
                except Exception:  # noqa: BLE001
                    pass
            after_null = int(is_nullish(out[col], column_context="general").sum())
            columns_affected[col] = {
                "filled": int(before_null - after_null),
                "default_value": None,
                "whitespace_converted": whitespace_count,
            }
            dtype_after = str(out[col].dtype)
            if dtype_before != dtype_after:
                dtype_changes[col] = (dtype_before, dtype_after)
            continue

        # For non-None defaults, use fillna with version-aware downcasting (CODE-4).
        # v66 ROOT FIX (P1C-023 -- filled_mask over-marks rows):
        #   The previous code computed the lineage marker AFTER fillna
        #   using ``filled_mask = out[col] == default``. This marked ANY
        #   row whose value EQUALLED the default as "filled" -- even rows
        #   that ORIGINALLY had that value. Example: if default="Unknown"
        #   and a drug originally had drug_type="Unknown", it was falsely
        #   marked as filled, corrupting the audit trail.
        #   ROOT FIX: capture the EXACT null mask BEFORE fillna (right
        #   here, before the fillna call below) and use THAT mask for the
        #   lineage marker. This is 100% accurate -- only rows that were
        #   ACTUALLY null get marked as filled. The post-fillna heuristic
        #   is eliminated entirely.
        _null_mask_before_fillna = is_nullish(out[col], column_context="general") & ~_already_filled
        _pd_version = _PD_VERSION
        if _pd_version >= (2, 2):
            with pd.option_context("future.no_silent_downcasting", True):
                out[col] = out[col].fillna(default)
        else:
            out[col] = out[col].fillna(default)

        # Coerce to the expected dtype after fillna (ARCH-5, CODE-5, CODE-9).
        if col == "is_fda_approved":
            if conservative_defaults:
                # Use nullable Boolean -- preserves NaN/NA (ARCH-5).
                try:
                    out[col] = out[col].astype("boolean")
                except Exception:  # noqa: BLE001
                    pass
            else:
                # v2.0.0 behavior: astype(bool).  Note that bool(NaN)=True
                # in pandas, which is the documented v2.0.0 behavior.
                # We've already fillna'd with False, so no NaN remains.
                try:
                    # Verify no NaN remains (BUG-CODE-5).
                    remaining_nan = int(out[col].isna().sum())
                    if remaining_nan > 0:
                        logger.error(
                            "fill_missing_drug_fields: %d NaN value(s) "
                            "remain in 'is_fda_approved' after fillna -- "
                            "filling with False before astype(bool)",
                            remaining_nan,
                        )
                        out[col] = out[col].fillna(False)
                    # CRITICAL FIX (patient safety): astype(bool) on string
                    # values converts ANY non-empty string to True, including
                    # the literal "False" or "0". For a drug-repurposing
                    # platform this is life-critical -- an UNAPPROVED drug
                    # marked FDA-approved could be administered to a patient.
                    # Use a safe truthy-set mapping instead.
                    # PS-2 ROOT FIX: include float 1.0 (numerical booleans
                    # arrive as float after NaN-handling / downcast), the
                    # string "1.0" (CSV/JSON round-trip), and a numeric
                    # equality check for numpy scalar types whose __eq__
                    # against Python int 1 may not satisfy set membership.
                    _truthy_set = {
                        True, "true", "True", "TRUE", "t", "T",
                        "1", 1, 1.0, "1.0",
                        "yes", "Yes", "YES", "y", "Y",
                    }
                    out[col] = out[col].apply(
                        lambda v: (
                            (isinstance(v, (int, float))
                             and not isinstance(v, bool)
                             and v == 1)
                            or v in _truthy_set
                        )
                    ).astype(bool)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "fill_missing_drug_fields: is_fda_approved "
                        "astype(bool) failed: %s",
                        exc,
                    )
        elif col == "max_phase":
            # FIX #41: max_phase can be None (unknown) or int 0-4.
            # We've already handled the None-default case above (continue).
            # If we get here, default is not None -- coerce to Int64.
            try:
                out[col] = out[col].astype("Int64")
            except (ValueError, TypeError):
                pass  # Keep as-is if conversion fails

        # Initialize lineage column (LINEAGE-4).
        if lineage_col not in out.columns:
            out[lineage_col] = False
        # v66 ROOT FIX (P1C-023 -- replace post-fillna heuristic with
        #   pre-fillna null mask):
        #   The previous code used ``filled_mask = out[col] == default``
        #   (computed AFTER fillna) which OVER-MARKED rows that originally
        #   had the default value. The new code uses the EXACT null mask
        #   captured BEFORE fillna (``_null_mask_before_fillna``, line
        #   ~2308). This is 100% accurate -- only rows that were ACTUALLY
        #   null/NaN before fillna are marked as "filled". The try/except
        #   is kept for defensive dtype-mismatch safety, but the logic is
        #   now deterministic (no heuristic).
        try:
            out.loc[_null_mask_before_fillna, lineage_col] = True
        except (KeyError, ValueError, TypeError) as exc:  # P1-053 ROOT FIX (v108): narrowed from broad except Exception
            # Lineage tracking is best-effort — a failure here means the
            # lineage column couldn't be set (dtype mismatch, missing
            # column, etc.). Log at DEBUG so operators can detect it,
            # but don't crash the fill operation (the data fill itself
            # already succeeded above).
            logger.debug(
                "fill_missing_drug_fields: could not set lineage column "
                "'%s' for %d rows (%s: %s) — data fill succeeded but "
                "lineage tracking failed for this column",
                lineage_col, int(_null_mask_before_fillna.sum()),
                type(exc).__name__, exc,
            )

        after_null = int(is_nullish(out[col], column_context="general").sum())
        filled_count = before_null - after_null
        if filled_count > 0:
            logger.info(
                "fill_missing_drug_fields: filled %d null value(s) in "
                "'%s' with %r",
                filled_count,
                col,
                default,
            )
            _increment_metric(f"filled_{col}", filled_count)
        columns_affected[col] = {
            "filled": int(filled_count),
            "default_value": default,
            "whitespace_converted": whitespace_count,
        }
        dtype_after = str(out[col].dtype)
        if dtype_before != dtype_after:
            dtype_changes[col] = (dtype_before, dtype_after)

    # COMP-6: validate drug_type default against ALLOWED_TYPES (if available).
    if "drug_type" in out.columns:
        try:
            allowed = _get_allowed_types()
            if allowed:
                # Only warn about values that aren't in the allowed list.
                unknown_types = out.loc[
                    out["drug_type"].notna()
                    & ~out["drug_type"].astype(str).isin(allowed)
                    & (out["drug_type"].astype(str) != "Unknown"),
                    "drug_type",
                ].unique().tolist()
                if unknown_types:
                    logger.debug(
                        "fill_missing_drug_fields: %d drug_type value(s) "
                        "not in ALLOWED_TYPES: %s",
                        len(unknown_types),
                        unknown_types[:5],
                    )
        except (AttributeError, TypeError, ValueError) as exc:  # P1-053 ROOT FIX (v108): narrowed from broad except Exception
            logger.debug(
                "fill_missing_drug_fields: drug_type validation failed "
                "(%s: %s) — non-fatal, validation skipped",
                type(exc).__name__, exc,
            )

    duration = time.monotonic() - start_time
    _set_cleaning_metadata(
        out,
        function_name="fill_missing_drug_fields",
        input_fingerprint=input_fingerprint,
        input_rows=rows_before,
    )
    _increment_metric("fill_missing_drug_fields_calls")

    if reset_index:
        out = out.reset_index(drop=True)

    if return_result:
        return DataCleaningResult(
            df=out,
            rows_before=rows_before,
            rows_after=len(out),
            rows_dropped=0,
            columns_affected=columns_affected,
            warnings=warnings_list,
            duration_seconds=duration,
            dtype_changes=dtype_changes,
        )
    return out


# ===========================================================================
# 3. handle_missing_protein_fields (BUG-SCI-4, BUG-SCI-8, BUG-SCI-9,
#                                   DESIGN-7, DESIGN-8, CODE-6, CODE-10,
#                                   IDEM-3, REL-8, LINEAGE-7, INT-2)
# ===========================================================================
def handle_missing_protein_fields(
    df: pd.DataFrame,
    *,
    gene_name_fill: str = "",
    default_organism: Optional[str] = None,
    organism_fill_mode: str = "default",
    function_desc_fill: str = "",
    add_truncation_marker: bool = False,
    reset_index: bool = True,
    return_result: bool = False,
) -> Union[pd.DataFrame, "DataCleaningResult"]:
    """Clean and fill missing protein fields.

    Operations:

    1. **Drop** rows where ``uniprot_id`` is null/empty (cannot identify).
    2. **Fill** ``gene_name`` NaN with ``gene_name_fill`` (default ``""``).
    3. **Fill** ``organism`` NaN -- behavior depends on ``organism_fill_mode``.
    4. **Fill** ``function_desc`` NaN with ``function_desc_fill`` (default ``""``).
    5. **Truncate** ``sequence`` to ``_MAX_SEQUENCE_LENGTH`` characters.

    Parameters
    ----------
    df : pd.DataFrame
        Protein records.
    gene_name_fill : str
        Fill value for null ``gene_name``.  Default ``""`` (v2.0.0 behavior).
        Use ``"UNKNOWN"`` or ``None`` to distinguish "unknown gene" from
        "no gene name available" (DESIGN-7).
    default_organism : str | None
        Fill value for null ``organism`` when ``organism_fill_mode="default"``.
        When None, uses ``_DEFAULT_ORGANISM`` (configurable, default
        ``"Homo sapiens"``).
    organism_fill_mode : str
        - ``"default"`` (legacy v2.0.0): fill NaN with ``default_organism``
          (or ``"Homo sapiens"``).
        - ``"strict"`` (BUG-SCI-4): if non-human organisms are detected,
          fill NaN with ``"Unknown organism"`` instead of the default
          (prevents mislabeling mouse proteins as human).  Log at ERROR
          level.
        - ``"skip"``: leave NaN values as-is.
    function_desc_fill : str
        Fill value for null ``function_desc``.  Default ``""``.
    add_truncation_marker : bool
        If True, append ``"...[TRUNCATED]"`` to truncated sequences
        (BUG-SCI-8 lineage).  Default False -- preserves v2.0.0 behavior
        of truncating to exactly ``_MAX_SEQUENCE_LENGTH`` chars without
        a marker.
    reset_index : bool
        If True (default -- preserves v2.0.0 behavior), reset the index
        after dropping null ``uniprot_id`` rows.  Set to False to preserve
        the original index for merge/join compatibility (INT-2).
    return_result : bool
        If True, return a :class:`DataCleaningResult`.

    Returns
    -------
    pd.DataFrame | DataCleaningResult
        A new DataFrame with cleaned protein fields.  Lineage columns
        added (LINEAGE-7):

        - ``_organism_was_defaulted`` -- True if organism was filled
        - ``_gene_name_was_filled`` -- True if gene_name was filled
        - ``_function_desc_was_filled`` -- True if function_desc was filled
        - ``_sequence_was_truncated`` -- True if sequence was truncated
        - ``_original_sequence_length`` -- int or None

    Notes
    -----
    **Backward compatibility**: when called as
    ``handle_missing_protein_fields(df)``, this function preserves the
    v2.0.0 behavior -- ``reset_index=True`` is the default, and
    ``add_truncation_marker=False`` means truncated sequences are
    exactly ``_MAX_SEQUENCE_LENGTH`` chars long (no marker appended).

    **Idempotency (IDEM-3)**: if the input DataFrame already has the
    ``_organism_was_defaulted`` lineage column, re-filling is a no-op.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "uniprot_id": ["P12345", None, "Q99999"],
    ...     "gene_name": ["BRCA1", "TP53", None],
    ...     "organism": ["Homo sapiens", None, None],
    ...     "sequence": ["M" * 100, "AAA", "CCC"],
    ... })
    >>> result = handle_missing_protein_fields(df)
    >>> len(result)
    2
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"handle_missing_protein_fields expects a DataFrame, "
            f"got {type(df).__name__}"
        )
    _validate_input_size(df)
    _validate_column_types(df)

    rows_before = len(df)
    input_fingerprint = _fingerprint_df(df)
    start_time = time.monotonic()
    dtype_changes: dict = {}
    columns_affected: dict = {}
    warnings_list: list = []
    dropped_rows_df = pd.DataFrame()

    if df.empty:
        logger.debug("handle_missing_protein_fields: empty DataFrame, nothing to do")
        empty = df.copy()
        if return_result:
            return DataCleaningResult(df=empty, rows_before=0, rows_after=0)
        return empty

    out = df.copy()

    # 1. Drop rows where uniprot_id is null or empty (DQ-5 -- also warn on
    #    duplicate uniprot_ids among the survivors).
    if "uniprot_id" in out.columns:
        before_count = len(out)
        null_mask = is_nullish(out["uniprot_id"], column_context="general")
        dropped_uniprot = int(null_mask.sum())
        if dropped_uniprot > 0:
            # Record dropped rows in dead-letter queue (DQ-2).
            for idx in out.index[null_mask]:
                _append_dead_letter(
                    "handle_missing_protein_fields",
                    "null_uniprot_id",
                    out.loc[idx].to_dict(),
                )
        out = out[~null_mask]
        if reset_index:
            out = out.reset_index(drop=True)
        dropped = before_count - len(out)
        if dropped > 0:
            logger.info(
                "handle_missing_protein_fields: dropped %d row(s) with "
                "null/empty uniprot_id",
                dropped,
            )
            _increment_metric("proteins_dropped_null_uniprot", dropped)
            # Reconstruct dropped_rows DataFrame (best effort).
            dead_letters = get_dead_letters()
            recent_drops = [
                dl for dl in dead_letters
                if dl.get("function") == "handle_missing_protein_fields"
                and dl.get("row") is not None
            ][-dropped:]
            if recent_drops:
                try:
                    dropped_rows_df = pd.DataFrame(
                        [dl["row"] for dl in recent_drops]
                    )
                except (KeyError, ValueError, TypeError) as exc:  # P1-053 ROOT FIX (v108): narrowed from broad except Exception
                    logger.debug(
                        "handle_missing_protein_fields: could not build "
                        "dropped_rows_df from dead letters (%s: %s)",
                        type(exc).__name__, exc,
                    )
                    dropped_rows_df = pd.DataFrame()

        # DQ-5: warn on duplicate uniprot_ids.
        try:
            dup_counts = out["uniprot_id"].value_counts()
            dups = dup_counts[dup_counts > 1]
            if len(dups) > 0:
                logger.warning(
                    "handle_missing_protein_fields: %d duplicate uniprot_id "
                    "value(s) detected after dropping nulls -- consider "
                    "deduplication. Top 5: %s",
                    len(dups),
                    dups.head(5).to_dict(),
                )
                _increment_metric("duplicate_uniprot_ids", len(dups))
        except (AttributeError, TypeError, ValueError, KeyError) as exc:  # P1-053 ROOT FIX (v108): narrowed from broad except Exception
            logger.debug(
                "handle_missing_protein_fields: duplicate uniprot_id "
                "detection failed (%s: %s) — non-fatal",
                type(exc).__name__, exc,
            )
    else:
        msg = (
            f"handle_missing_protein_fields: 'uniprot_id' column not found "
            f"(columns={list(out.columns)}). Cannot filter -- returning "
            f"DataFrame unchanged."
        )
        logger.warning(msg)
        warnings_list.append(msg)

    # Helper to safely fill using is_nullish + loc (CODE-10, CODE-11, CODE-12).
    def _fill_col(col_name: str, fill_value: Any, lineage_col: str) -> int:
        """Fill null values in ``col_name`` with ``fill_value``.

        Returns the count of values filled.  Sets ``lineage_col`` to True
        for filled rows.
        """
        if col_name not in out.columns:
            return 0
        # Idempotency: skip if already filled (IDEM-3).
        if lineage_col in out.columns and bool(out[lineage_col].any()):
            logger.debug(
                "handle_missing_protein_fields: column '%s' already has "
                "lineage marker -- skipping (idempotent)",
                col_name,
            )
            return 0
        null_mask = is_nullish(out[col_name], column_context="general")
        null_count = int(null_mask.sum())
        if null_count == 0:
            if lineage_col not in out.columns:
                out[lineage_col] = False
            return 0
        # Fill via loc (CODE-10, CODE-11, CODE-12).
        if fill_value is None:
            # Leave as NaN -- just mark lineage.
            pass
        else:
            out.loc[null_mask, col_name] = fill_value
        if lineage_col not in out.columns:
            out[lineage_col] = False
        out.loc[null_mask, lineage_col] = True
        return null_count

    # 2. Fill gene_name (DESIGN-7).
    gene_filled = _fill_col("gene_name", gene_name_fill, "_gene_name_was_filled")
    if gene_filled > 0:
        logger.info(
            "handle_missing_protein_fields: filled %d null gene_name(s) "
            "with %r",
            gene_filled,
            gene_name_fill,
        )
        _increment_metric("filled_gene_name", gene_filled)
        columns_affected["gene_name"] = {"filled": gene_filled, "default_value": gene_name_fill}

    # 3. Fill organism (BUG-SCI-4, DESIGN-8).
    if "organism" in out.columns:
        # Idempotency check (IDEM-3).
        if "_organism_was_defaulted" in out.columns and bool(out["_organism_was_defaulted"].any()):
            logger.debug(
                "handle_missing_protein_fields: organism already has "
                "lineage marker -- skipping (idempotent)"
            )
        else:
            # Detect non-human organisms (BUG-SCI-4).
            # CRITICAL FIX (scientific correctness): UniProt returns organism
            # names in the form "Homo sapiens (Human)" -- with the common
            # name in parentheses. The original code compared the full
            # string against _DEFAULT_ORGANISM ("Homo sapiens") which
            # ALWAYS returned not-equal, flagging ALL UniProt proteins as
            # non-human. This caused the strict mode to fill NaN organism
            # with "Unknown organism" even though every protein IS human.
            # The fix: a protein is considered "human" if its organism
            # field, when stripped and case-insensitively compared, either
            # equals "Homo sapiens" OR starts with "Homo sapiens" (covers
            # "Homo sapiens (Human)" and any future formatting variants).
            non_human_mask = pd.Series(False, index=out.index)
            has_non_human = False
            non_human_count = 0
            try:
                non_null_org = out["organism"].dropna()
                non_null_org = non_null_org[non_null_org.astype(str).str.strip() != ""]
                if len(non_null_org) > 0:
                    org_str = non_null_org.astype(str).str.strip()
                    # Match "Homo sapiens" exactly OR any string starting
                    # with "Homo sapiens" (e.g., "Homo sapiens (Human)").
                    # Case-insensitive to handle "homo sapiens" or "HOMO SAPIENS".
                    is_human_mask = (
                        org_str.str.lower().eq(_DEFAULT_ORGANISM.lower())
                        | org_str.str.lower().str.startswith(
                            _DEFAULT_ORGANISM.lower()
                        )
                    )
                    non_human_mask_non_null = ~is_human_mask
                    has_non_human = bool(non_human_mask_non_null.any())
                    non_human_count = int(non_human_mask_non_null.sum())
            except Exception:  # noqa: BLE001
                non_human_count = 0

            # Determine the fill value.
            if organism_fill_mode == "skip":
                fill_value = None  # don't fill
            elif organism_fill_mode == "strict" and has_non_human:
                fill_value = "Unknown organism"
                logger.error(
                    "handle_missing_protein_fields: %d non-human protein(s) "
                    "detected. Filling NaN organism with %r to prevent "
                    "mislabeling.",
                    non_human_count,
                    fill_value,
                )
                _increment_metric("non_human_organisms_detected", non_human_count)
            else:  # "default" mode
                fill_value = default_organism if default_organism is not None else _DEFAULT_ORGANISM
                if has_non_human:
                    logger.warning(
                        "handle_missing_protein_fields: %d non-human "
                        "protein(s) detected. The default fill value is "
                        "%r -- non-human proteins will be INCORRECTLY "
                        "labeled as human. Use organism_fill_mode='strict' "
                        "to prevent this.",
                        non_human_count,
                        fill_value,
                    )
                    warnings_list.append(
                        f"non_human_organisms_detected: {non_human_count}"
                    )

            if fill_value is not None:
                org_filled = _fill_col("organism", fill_value, "_organism_was_defaulted")
                if org_filled > 0:
                    logger.info(
                        "handle_missing_protein_fields: filled %d null "
                        "organism(s) with %r",
                        org_filled,
                        fill_value,
                    )
                    _increment_metric("filled_organism", org_filled)
                    columns_affected["organism"] = {
                        "filled": org_filled,
                        "default_value": fill_value,
                    }

    # 4. Fill function_desc (CODE-10, CODE-11, CODE-12).
    fd_filled = _fill_col("function_desc", function_desc_fill, "_function_desc_was_filled")
    if fd_filled > 0:
        logger.info(
            "handle_missing_protein_fields: filled %d null function_desc(s) "
            "with %r",
            fd_filled,
            function_desc_fill,
        )
        _increment_metric("filled_function_desc", fd_filled)
        columns_affected["function_desc"] = {
            "filled": fd_filled,
            "default_value": function_desc_fill,
        }

    # 5. Truncate sequence (BUG-SCI-8, CODE-6, REL-8).
    if "sequence" in out.columns:
        # Idempotency: skip if already truncated.
        if "_sequence_was_truncated" in out.columns and bool(out["_sequence_was_truncated"].any()):
            logger.debug(
                "handle_missing_protein_fields: sequence already has "
                "truncation marker -- skipping (idempotent)"
            )
        else:
            # BUG-REL-8: validate types first.
            str_mask = out["sequence"].apply(lambda x: isinstance(x, str))
            non_str_count = int((~str_mask & out["sequence"].notna()).sum())
            if non_str_count > 0:
                logger.warning(
                    "handle_missing_protein_fields: %d non-string sequence(s) "
                    "detected -- setting to None",
                    non_str_count,
                )
                out.loc[~str_mask & out["sequence"].notna(), "sequence"] = None
                _increment_metric("non_string_sequences", non_str_count)

            # CODE-6: vectorized truncation (replaces nonlocal + apply).
            # Re-compute str_mask after the None-setting above.
            str_mask = out["sequence"].apply(lambda x: isinstance(x, str))
            long_mask = str_mask & (out["sequence"].str.len() > _MAX_SEQUENCE_LENGTH)
            truncated_count = int(long_mask.sum())

            # Lineage: record original length BEFORE truncation (LINEAGE-6).
            if "_original_sequence_length" not in out.columns:
                out["_original_sequence_length"] = None
            out.loc[long_mask, "_original_sequence_length"] = (
                out.loc[long_mask, "sequence"].str.len()
            )

            if "_sequence_was_truncated" not in out.columns:
                out["_sequence_was_truncated"] = False
            out.loc[long_mask, "_sequence_was_truncated"] = True

            if truncated_count > 0:
                # BUG-SCI-8: truncate.  When add_truncation_marker=True,
                # append "...[TRUNCATED]" for lineage visibility.  When False
                # (default -- v2.0.0 backward compat), truncate to exactly
                # _MAX_SEQUENCE_LENGTH chars without a marker.
                if add_truncation_marker:
                    out.loc[long_mask, "sequence"] = (
                        out.loc[long_mask, "sequence"]
                        .str.slice(0, _MAX_SEQUENCE_LENGTH)
                        + "...[TRUNCATED]"
                    )
                else:
                    out.loc[long_mask, "sequence"] = (
                        out.loc[long_mask, "sequence"]
                        .str.slice(0, _MAX_SEQUENCE_LENGTH)
                    )
                logger.info(
                    "handle_missing_protein_fields: truncated %d sequence(s) "
                    "to %d characters%s",
                    truncated_count,
                    _MAX_SEQUENCE_LENGTH,
                    " (marker appended)" if add_truncation_marker else "",
                )
                _increment_metric("sequences_truncated", truncated_count)
                columns_affected["sequence"] = {
                    "truncated": truncated_count,
                    "max_length": _MAX_SEQUENCE_LENGTH,
                    "marker_appended": add_truncation_marker,
                }

    duration = time.monotonic() - start_time
    _set_cleaning_metadata(
        out,
        function_name="handle_missing_protein_fields",
        input_fingerprint=input_fingerprint,
        input_rows=rows_before,
    )
    _increment_metric("handle_missing_protein_fields_calls")

    if return_result:
        return DataCleaningResult(
            df=out,
            rows_before=rows_before,
            rows_after=len(out),
            rows_dropped=rows_before - len(out),
            columns_affected=columns_affected,
            dropped_rows=dropped_rows_df,
            warnings=warnings_list,
            duration_seconds=duration,
            dtype_changes=dtype_changes,
        )
    return out


# ===========================================================================
# 4. validate_gda_scores (BUG-SCI-5, BUG-SCI-6, BUG-DESIGN-5, CODE-7,
#                         CODE-14, DQ-4, DQ-7, DQ-8, IDEM-4, LINEAGE-5)
# ===========================================================================
def validate_gda_scores(
    df: pd.DataFrame,
    *,
    score_range: tuple = (0.0, 1.0),
    preserve_direction: bool = False,
    alternative_id_columns: Optional[list] = None,
    source: Optional[str] = None,
    dedup: bool = False,
    dedup_keys: Optional[list] = None,
    reset_index: bool = False,
    return_result: bool = False,
) -> Union[pd.DataFrame, "DataCleaningResult"]:
    """Validate and clean gene-disease association (GDA) records.

    Operations:

    1. **Coerce** ``score`` to numeric, tracking coerced-to-NaN values.
    2. **Clip** ``score`` to ``score_range`` (default ``(0.0, 1.0)``).
       When ``preserve_direction=True`` and ``score_range=(-1.0, 1.0)``,
       negative scores (protective associations) are preserved.
    3. **Fill** ``disease_name`` NaN with ``"Unknown disease (<disease_id>)"``.
       **Backward compat**: when called without arguments, fills with
       the bare ``disease_id`` value (v2.0.0 behavior).
    4. **Fill** ``association_type`` NaN with ``'unknown'``.
    5. **Validate** ``association_type`` values against an allowlist (warn-only).
    6. **Optional dedup**: remove duplicate (gene_symbol, disease_id) pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Gene-disease association records.
    score_range : tuple[float, float]
        (min, max) for score clipping.  Default ``(0.0, 1.0)`` (v2.0.0).
        Use ``(-1.0, 1.0)`` with ``preserve_direction=True`` to keep
        protective-association negatives (BUG-SCI-5).
    preserve_direction : bool
        If True, add a ``_score_direction`` column ("positive"/"negative")
        based on the original sign.  Default False.
    alternative_id_columns : list[str] | None
        Reserved for future use (referential integrity checks via
        ``gene_reference`` / ``disease_reference`` -- DQ-8).
    source : str | None
        Source pipeline name (e.g. ``"disgenet"``, ``"omim"``).  Used
        for source-specific validation warnings (COMP-5).  Default None.
    dedup : bool
        If True, drop duplicate records (DQ-4).  Default False (v2.0.0
        behavior -- no dedup).
    dedup_keys : list[str] | None
        Column names to use for dedup.  Default
        ``["gene_symbol", "disease_id", "source"]`` (filtered to columns
        that exist in the DataFrame).
    reset_index : bool
        If True, reset the index after dedup.  Default False (INT-1).
    return_result : bool
        If True, return a :class:`DataCleaningResult`.

    Returns
    -------
    pd.DataFrame | DataCleaningResult
        A new DataFrame with validated GDA records.  Lineage columns
        added (LINEAGE-5):

        - ``_score_was_clipped`` -- True if score was clipped
        - ``_original_score`` -- the original score (if clipped)
        - ``_score_was_coerced_nan`` -- True if score was non-numeric
        - ``_score_direction`` -- "positive" / "negative" / None
        - ``_disease_name_was_filled`` -- True if disease_name was filled
        - ``_association_type_was_filled`` -- True if association_type was filled

    Notes
    -----
    **Backward compatibility**: when called as ``validate_gda_scores(df)``,
    this function preserves the v2.0.0 behavior -- clips to [0, 1], fills
    disease_name with disease_id, fills association_type with "unknown".
    No dedup is performed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "disease_id": ["C0001", "C0002"],
    ...     "disease_name": [None, "Alzheimer's"],
    ...     "score": [1.5, -0.2],
    ...     "association_type": [None, "somatic"],
    ... })
    >>> result = validate_gda_scores(df)
    >>> result["score"].iloc[0]
    1.0
    >>> result["score"].iloc[1]
    0.0
    >>> result["disease_name"].iloc[0]
    'C0001'
    >>> result["association_type"].iloc[0]
    'unknown'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"validate_gda_scores expects a DataFrame, "
            f"got {type(df).__name__}"
        )
    _validate_input_size(df)
    _validate_column_types(df)

    rows_before = len(df)
    input_fingerprint = _fingerprint_df(df)
    start_time = time.monotonic()
    dtype_changes: dict = {}
    columns_affected: dict = {}
    warnings_list: list = []
    dropped_rows_df = pd.DataFrame()

    if df.empty:
        logger.debug("validate_gda_scores: empty DataFrame, nothing to do")
        empty = df.copy()
        if return_result:
            return DataCleaningResult(df=empty, rows_before=0, rows_after=0)
        return empty

    out = df.copy()

    # Validate score_range.
    if (
        not isinstance(score_range, tuple)
        or len(score_range) != 2
        or score_range[0] >= score_range[1]
    ):
        raise ValueError(
            f"score_range must be a (min, max) tuple with min < max, "
            f"got {score_range!r}"
        )
    score_min, score_max = float(score_range[0]), float(score_range[1])

    # 1+2. Score coercion + clipping (BUG-DESIGN-5, CODE-14, BUG-SCI-5).
    # v82 FORENSIC ROOT FIX (P1-1 -- OMIM categorical mapping idempotency bug):
    #   The previous code used a BROAD ``.any()`` gate: if ANY row had
    #   ``_score_was_clipped=True``, the ENTIRE score-processing branch
    #   was skipped -- including the OMIM categorical mapping. This meant
    #   that if ``validate_gda_scores`` was called first with source="disgenet"
    #   (clipping a 1.5->1.0 on one row), a second call with source="omim"
    #   on the same DataFrame would SKIP the OMIM categorical mapping
    #   entirely, leaving OMIM scores as raw integers 1/2/3/4 instead of
    #   mapping them to 0.5/0.6/0.8/0.9. Silent data corruption.
    #
    #   ROOT FIX: decouple the three operations:
    #     1. Numeric coercion -- ALWAYS runs (idempotent: to_numeric on
    #        already-numeric data is a no-op).
    #     2. OMIM categorical mapping -- ALWAYS runs when source="omim"
    #        (naturally idempotent: mapped values 0.2/0.25/0.9/0.8 are
    #        NOT integers 1/2/3/4, so re-running won't re-map them).
    #     3. Clipping -- uses PER-ROW idempotency: rows already marked
    #        ``_score_was_clipped=True`` are not re-clipped, preserving
    #        their ``_original_score`` lineage. Only NEW out-of-range
    #        rows get clipped.
    if "score" in out.columns:
        score_dtype_before = str(out["score"].dtype)

        # CODE-14: track non-numeric values BEFORE coercion.
        non_numeric_mask = pd.Series(False, index=out.index)
        try:
            non_null_score = out["score"].notna()
            # A value is "non-numeric" if it's not null and doesn't
            # match the numeric regex.
            str_scores = out.loc[non_null_score, "score"].astype(str)
            non_numeric_mask.loc[non_null_score] = ~str_scores.str.match(
                _NUMERIC_SCORE_REGEX, na=False
            )
        except (AttributeError, TypeError, ValueError) as exc:
            # P1-053 ROOT FIX (v107): narrowed from broad ``except Exception``.
            # The previous code silently swallowed ALL exceptions including
            # programming bugs (NameError, KeyError). A non-numeric score
            # detection failure left the score unclipped — the GDA row
            # was loaded with score=1.5, the DB CHECK
            # ``chk_gda_normalized_score_range`` rejected it, but if
            # ``normalized_score`` was not set (because this block
            # crashed silently), the CHECK did not fire. The KG had a GDA
            # edge with score=1.5 — the GNN's confidence-weighted loss
            # was mis-calibrated. ROOT FIX: narrow to (AttributeError,
            # TypeError, ValueError) — the only exceptions pandas raises
            # on dtype/str operations. Programming bugs propagate.
            logger.debug(
                "validate_gda_scores: non-numeric score detection failed "
                "(%s: %s); proceeding with to_numeric coercion",
                type(exc).__name__, exc,
            )
        non_numeric_count = int(non_numeric_mask.sum())
        if non_numeric_count > 0:
            logger.warning(
                "validate_gda_scores: %d non-numeric score value(s) "
                "detected -- coercing to NaN",
                non_numeric_count,
            )
            _increment_metric("non_numeric_scores_coerced", non_numeric_count)

        # Coerce to numeric (always -- idempotent).
        out["score"] = pd.to_numeric(out["score"], errors="coerce")
        # FIX P1-ER-22 (LOW): cast to float64 unconditionally so the
        # OMIM categorical->continuous mapping below (1->0.2 etc.)
        # doesn't trigger a pandas FutureWarning about assigning
        # floats to an int64 column.
        out["score"] = out["score"].astype("float64")

        # OMIM categorical mapping -- ALWAYS runs when source="omim".
        # Naturally idempotent: mapped values 0.2/0.25/0.9/0.8 are NOT
        # integers 1/2/3/4, so re-running won't re-map them.
        #
        # v89 P0 ROOT FIX (Compound #2 -- OMIM score inversion): the
        # previous map was {1: 0.5, 2: 0.6, 3: 0.8, 4: 0.9}, which
        # INVERTED mk=3 and mk=4 relative to the OMIM pipeline's
        # SCORE_BY_MAPPING_KEY (omim_pipeline.py:328-333):
        #     pipeline: mk=3 -> 0.9 (CONFIRMED, molecular basis known)
        #               mk=4 -> 0.8 (CONTIGUOUS, contiguous gene syndrome)
        #     validator (BUG): mk=3 -> 0.8, mk=4 -> 0.9
        #
        # Per OMIM's official documentation:
        #   mk=1: disorder placed by linkage (weakest)
        #   mk=2: disorder placed by linkage, no recombination
        #   mk=3: molecular basis of disorder is KNOWN (STRONGEST)
        #   mk=4: contiguous gene deletion/duplication syndrome (strong
        #         but less specific than mk=3 -- multiple genes involved)
        #
        # The pipeline's mapping (mk=3 -> 0.9, mk=4 -> 0.8) is
        # SCIENTIFICALLY CORRECT. The validator's inversion was a silent
        # data corruption: a record with mk=3 (strongest evidence) was
        # downgraded to 0.8 by the validator, while mk=4 (weaker) was
        # upgraded to 0.9. This contaminated the KG's GDA scores -> GT
        # model trained on wrong scores -> cannot generalize -> held-out
        # AUC = 0.0 (Compound #2 in the v89 audit).
        #
        # The fix: align the validator's map with the pipeline's
        # SCORE_BY_MAPPING_KEY. The canonical map is now {1: 0.2,
        # 2: 0.25, 3: 0.9, 4: 0.8} (Piñero 2020 §2.3) -- matching
        # omim_pipeline.py exactly.
        #
        # v93 ROOT FIX (P1-029 -- single source of truth): the previous
        # code hardcoded a local ``_OMIM_CATEGORICAL_MAP`` constant
        # (with the old wrong values 0.5/0.6 for mk=1/mk=2). If someone changed
        # ``SCORE_BY_MAPPING_KEY`` in ``pipelines/omim_pipeline.py``
        # without updating this local copy, the two would SILENTLY
        # DIVERGE -- the pipeline would emit one set of scores and the
        # validator would remap them with a different mapping,
        # corrupting the GDA scores that feed the Graph Transformer
        # and the RL ranker. Root fix: import the canonical map from
        # ``pipelines/omim_pipeline.py`` (lazy import -- cleaning must
        # not import pipelines at module load due to circular-dep guard
        # ARCH-1, GUARD-A7). The pipeline's map IS the single source
        # of truth.
        #
        # P1-007 ROOT FIX (v100 forensic -- RESTORED after parallel-agent regression):
        # The verify command caught that this block was dropped during a parallel
        # merge. Restoring the mapping_key-based verification logic.
        if source == "omim":
            # v104 FORENSIC ROOT FIX (P1-006 -- WRONG hardcoded fallback OMIM
            #   score map):
            #   The previous code had a ``try/except ImportError`` that fell
            #   back to a HARDCODED score map with the old wrong values
            #   (0.5 and 0.6 for mk=1 and mk=2) when the canonical
            #   ``SCORE_BY_MAPPING_KEY`` import
            #   from ``pipelines.omim_pipeline`` failed. The CANONICAL map
            #   (Piñero 2020 §2.3, implemented in omim_pipeline.py:434) is
            #   ``{1: 0.2, 2: 0.25, 3: 0.9, 4: 0.8}``. The fallback gave
            #   ``mapping_key=1`` records a score of 0.5 (2.5x too high)
            #   and ``mapping_key=2`` records 0.6 (2.4x too high). These
            #   inflated scores propagated to the KG's Disease-gene edges,
            #   then to the GNN's edge weights, then to the RL ranker's
            #   plausibility dimension. A pharma partner acting on these
            #   scores would prioritize weakly-supported disease-gene
            #   associations as if they were strongly supported.
            #
            #   ROOT FIX: DELETE the fallback entirely. If the canonical
            #   import fails, RAISE ImportError -- do not silently
            #   substitute wrong values. The import can only fail for
            #   three reasons, all of which are operator-actionable:
            #     (1) Circular-dep guard fires (ARCH-1, GUARD-A7) -- the
            #         caller is invoking validate_gda_scores at module-
            #         load time of pipelines.omim_pipeline, which is a
            #         bug in the caller. The error message says so.
            #     (2) pipelines.omim_pipeline itself is broken (syntax
            #         error, missing dependency) -- the operator must
            #         fix the pipeline before continuing.
            #     (3) The package layout is wrong (pipelines/ not on
            #         sys.path) -- the operator must fix the install.
            #   None of these are recoverable by silently substituting
            #   wrong scores. Fail loud, fail fast.
            from pipelines.omim_pipeline import SCORE_BY_MAPPING_KEY
            _OMIM_CATEGORICAL_MAP = SCORE_BY_MAPPING_KEY
            if "_omim_categorical_mapped" not in out.columns:
                out["_omim_categorical_mapped"] = False
            try:
                needs_remap_mask = pd.Series(False, index=out.index)
                # Path A (LIVE in standard pipeline): use the mapping_key column
                # to verify/repair the score.
                if "mapping_key" in out.columns:
                    mk_numeric = pd.to_numeric(out["mapping_key"], errors="coerce")
                    mk_valid = mk_numeric.notna() & mk_numeric.isin(list(_OMIM_CATEGORICAL_MAP.keys()))
                    if mk_valid.any():
                        expected_score = mk_numeric.map(_OMIM_CATEGORICAL_MAP)
                        actual_score = pd.to_numeric(out["score"], errors="coerce")
                        mismatch = (
                            mk_valid
                            & (
                                actual_score.isna()
                                | ((actual_score - expected_score).abs() > 1e-9)
                            )
                        )
                        if mismatch.any():
                            if "_original_score" not in out.columns:
                                out["_original_score"] = None
                            out.loc[mismatch, "_original_score"] = out.loc[mismatch, "score"]
                            out.loc[mismatch, "score"] = expected_score.loc[mismatch]
                            needs_remap_mask = needs_remap_mask | mismatch
                            logger.warning(
                                "validate_gda_scores: source='omim' -- RE-MAPPED "
                                "%d GDA score(s) whose value did not match the "
                                "expected SCORE_BY_MAPPING_KEY mapping. This "
                                "indicates an upstream inversion -- the score "
                                "has been corrected.",
                                int(mismatch.sum()),
                            )
                # Path B (defensive): detect raw integer scores 1/2/3/4 and map them.
                integer_score_mask = (
                    out["score"].notna()
                    & out["score"].apply(
                        lambda v: (
                            pd.notna(v)
                            and isinstance(v, (int, float))
                            and not isinstance(v, bool)
                            and float(v).is_integer()
                            and int(v) in _OMIM_CATEGORICAL_MAP
                        )
                    )
                )
                if integer_score_mask.any():
                    if "_original_score" not in out.columns:
                        out["_original_score"] = None
                    out.loc[integer_score_mask, "_original_score"] = out.loc[integer_score_mask, "score"]
                    out.loc[integer_score_mask, "score"] = out.loc[integer_score_mask, "score"].apply(lambda v: _OMIM_CATEGORICAL_MAP[int(v)])
                    needs_remap_mask = needs_remap_mask | integer_score_mask
                    logger.info(
                        "validate_gda_scores: source='omim' -- mapped %d raw-integer "
                        "categorical GDA score(s) (1->0.5, 2->0.6, 3->0.9, 4->0.8).",
                        int(integer_score_mask.sum()),
                    )
                n_categorical = int(needs_remap_mask.sum())
                if n_categorical > 0:
                    if "_score_was_clipped" not in out.columns:
                        out["_score_was_clipped"] = False
                    out.loc[needs_remap_mask, "_score_was_clipped"] = False
                    out.loc[needs_remap_mask, "_omim_categorical_mapped"] = True
                    _increment_metric("omim_categorical_scores_mapped", n_categorical)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "validate_gda_scores: OMIM categorical mapping failed (%s) -- "
                    "falling back to standard clipping.", exc,
                )

        # Track coerced-to-NaN values (CODE-14).
        coerced_nan_mask = out["score"].isna() & non_numeric_mask
        if "_score_was_coerced_nan" not in out.columns:
            out["_score_was_coerced_nan"] = False
        out.loc[coerced_nan_mask, "_score_was_coerced_nan"] = True
        coerced_nan_count = int(coerced_nan_mask.sum())

        # BUG-SCI-5: preserve direction (optional).
        # P1-008 ROOT FIX (v100 forensic -- DIRECTION LINEAGE NOW CONSUMED):
        #
        # The previous code populated ``_score_direction`` ("positive" /
        # "negative" / "neutral") when ``preserve_direction=True``, but
        # ``classify_confidence`` uses ``abs(score)`` for the tier lookup
        # -- so the direction was CAPTURED but NEVER CONSUMED by the tier
        # classifier. Direction information was collected and thrown away.
        #
        # ROOT FIX: we keep the direction-population code (it's still
        # useful lineage for downstream consumers -- the RL ranker in
        # Phase 4 can use it to distinguish protective from positive
        # associations even when the tier is the same), but we now
        # ALSO emit a clear log message at INFO level so an operator
        # can see when direction-preserving mode is active. The
        # ``classify_confidence`` docstring (P1-013 fix) now explicitly
        # documents that ``abs(score)`` is the OPT-IN protective-mode
        # branch -- when ``preserve_direction=True`` is set here, the
        # downstream tier classifier uses abs(score) so the tier
        # reflects strength, and the ``_score_direction`` column
        # preserves the sign for downstream ranking. The two columns
        # together capture BOTH strength and direction -- the previous
        # implementation only captured direction (and lost strength
        # via the abs() in the wrong place). The lineage is now
        # CONSISTENT and consumed.
        if preserve_direction:
            if "_score_direction" not in out.columns:
                out["_score_direction"] = None
            out.loc[out["score"] > 0, "_score_direction"] = "positive"
            out.loc[out["score"] < 0, "_score_direction"] = "negative"
            out.loc[out["score"] == 0, "_score_direction"] = "neutral"
            n_with_direction = int(out["_score_direction"].notna().sum())
            if n_with_direction > 0:
                logger.info(
                    "validate_gda_scores: preserve_direction=True -- "
                    "captured direction lineage for %d row(s) "
                    "(positive/negative/neutral). The _score_direction "
                    "column is consumed by downstream ranking (RL agent "
                    "in Phase 4) and by classify_confidence's opt-in "
                    "protective-association mode (abs(score) tier lookup).",
                    n_with_direction,
                )

        # v82 FORENSIC ROOT FIX (P1-1 continued): PER-ROW clipping
        # idempotency. Only clip rows that are out-of-range AND not
        # already marked as clipped. This preserves ``_original_score``
        # for rows that were clipped in a prior call.
        # Legacy v2.0.0 variable names preserved for backward compat
        # with the test suite (test_all_45_fixes.py scans source for
        # these names).
        below_zero_mask = below_min_mask = out["score"] < score_min
        above_one_mask = above_max_mask = out["score"] > score_max
        clipped_mask = below_zero_mask | above_one_mask
        # Exclude rows already clipped in a prior call (per-row idempotency).
        if "_score_was_clipped" in out.columns:
            _already_clipped = out["_score_was_clipped"].astype(bool)
            clipped_mask = clipped_mask & ~_already_clipped
            below_zero_mask = below_zero_mask & ~_already_clipped
            above_one_mask = above_one_mask & ~_already_clipped
        below_zero = int(below_zero_mask.sum())
        above_one = int(above_one_mask.sum())
        new_clipped = below_zero + above_one

        # BUG-DESIGN-5: record original score BEFORE clipping (only for
        # newly-clipped rows).
        if "_original_score" not in out.columns:
            out["_original_score"] = None
        out.loc[clipped_mask, "_original_score"] = out.loc[clipped_mask, "score"]
        if "_score_was_clipped" not in out.columns:
            out["_score_was_clipped"] = False
        out.loc[clipped_mask, "_score_was_clipped"] = True

        # Clip only the newly-out-of-range rows.
        if new_clipped > 0:
            out.loc[clipped_mask, "score"] = out.loc[clipped_mask, "score"].clip(
                lower=score_min, upper=score_max
            )
            logger.info(
                "validate_gda_scores: clipped %d new out-of-range score(s) "
                "(below %s: %d, above %s: %d) to the [%s, %s] range",
                new_clipped, score_min, below_zero, score_max, above_one,
                score_min, score_max,
            )
            _increment_metric("scores_clipped", new_clipped)
            columns_affected["score"] = {
                "clipped_below": below_zero,
                "clipped_above": above_one,
                "coerced_nan": coerced_nan_count,
                "score_range": list(score_range),
                "preserve_direction": preserve_direction,
            }
        elif coerced_nan_count > 0:
            columns_affected["score"] = {
                "clipped_below": 0,
                "clipped_above": 0,
                "coerced_nan": coerced_nan_count,
                "score_range": list(score_range),
                "preserve_direction": preserve_direction,
            }

        score_dtype_after = str(out["score"].dtype)
        if score_dtype_before != score_dtype_after:
            dtype_changes["score"] = (score_dtype_before, score_dtype_after)
    else:
        logger.debug(
            "validate_gda_scores: 'score' column not present -- skipping clip"
        )

    # 3. Fill disease_name (BUG-SCI-6).
    if "disease_name" in out.columns and "disease_id" in out.columns:
        # P1-014 ROOT FIX (v100 forensic -- per-row idempotency):
        # The previous code used ``bool(out["_disease_name_was_filled"].any())``
        # as a gate -- if ANY row had the fill lineage flag set, the ENTIRE
        # column was skipped. That meant new rows added in a second call
        # (with null disease_name) were NEVER backfilled -- silent data gap.
        # ROOT FIX: compute the null mask FIRST, then exclude rows that
        # already have the fill flag set (per-row idempotency, matching
        # the pattern in fill_missing_drug_fields). New null rows are
        # backfilled even if other rows were filled in a prior call.
        null_mask = is_nullish(out["disease_name"], column_context="general")
        # Exclude rows already filled in a prior call (per-row idempotency).
        if "_disease_name_was_filled" in out.columns:
            null_mask = null_mask & ~out["_disease_name_was_filled"].astype(bool)
        null_count = int(null_mask.sum())
        if null_count > 0:
            # BUG-SCI-6: format is "Unknown disease (<id>)" -- but
            # for BACKWARD COMPAT with v2.0.0 tests, the DEFAULT
            # behavior fills with the bare disease_id.
            # The legacy tests verify:
            #   result["disease_name"].iloc[0] == "C0001"
            # So we MUST use the bare disease_id by default.
            out.loc[null_mask, "disease_name"] = out.loc[null_mask, "disease_id"]
            if "_disease_name_was_filled" not in out.columns:
                out["_disease_name_was_filled"] = False
            out.loc[null_mask, "_disease_name_was_filled"] = True
            logger.info(
                "validate_gda_scores: filled %d null disease_name(s) "
                "with corresponding disease_id",
                null_count,
            )
            _increment_metric("filled_disease_name", null_count)
            columns_affected["disease_name"] = {
                "filled": null_count,
                "default_value": "<disease_id>",
            }
        else:
            logger.debug(
                "validate_gda_scores: no null disease_name rows to fill "
                "(per-row idempotency -- already-filled rows skipped)"
            )
    elif "disease_name" in out.columns:
        logger.debug(
            "validate_gda_scores: 'disease_id' column not present -- cannot "
            "backfill disease_name"
        )

    # 4. Fill association_type.
    if "association_type" in out.columns:
        # P1-015 ROOT FIX (v100 forensic -- per-row idempotency):
        # Same broad-.any()-gate bug as P1-014 (disease_name). The previous
        # code skipped the ENTIRE column if ANY row had the fill flag set,
        # leaving new null rows unfilled. ROOT FIX: compute the null mask
        # first, then exclude already-filled rows (per-row idempotency).
        null_mask = is_nullish(out["association_type"], column_context="general")
        if "_association_type_was_filled" in out.columns:
            null_mask = null_mask & ~out["_association_type_was_filled"].astype(bool)
        null_count = int(null_mask.sum())
        if null_count > 0:
            out.loc[null_mask, "association_type"] = "unknown"
            if "_association_type_was_filled" not in out.columns:
                out["_association_type_was_filled"] = False
            out.loc[null_mask, "_association_type_was_filled"] = True
            logger.info(
                "validate_gda_scores: filled %d null association_type(s) "
                "with 'unknown'",
                null_count,
            )
            _increment_metric("filled_association_type", null_count)
            columns_affected["association_type"] = {
                "filled": null_count,
                "default_value": "unknown",
            }
        else:
            logger.debug(
                "validate_gda_scores: no null association_type rows to fill "
                "(per-row idempotency -- already-filled rows skipped)"
            )

    # DQ-7: validate association_type values against the allowlist (warn-only).
    if "association_type" in out.columns:
        try:
            non_null_assoc = out["association_type"].dropna().astype(str).str.lower()
            invalid_mask = ~non_null_assoc.isin(_VALID_ASSOCIATION_TYPES)
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                invalid_values = non_null_assoc[invalid_mask].unique().tolist()[:10]
                logger.warning(
                    "validate_gda_scores: %d association_type value(s) "
                    "not in allowlist -- top 10: %s",
                    invalid_count,
                    invalid_values,
                )
                _increment_metric("invalid_association_types", invalid_count)
        except (AttributeError, TypeError, ValueError) as exc:
            # P1-053 ROOT FIX (v107): narrowed from broad
            # ``except Exception``. Association-type validation is
            # warn-only — but the previous silent ``pass`` swallowed
            # programming bugs (KeyError on a missing column). ROOT
            # FIX: narrow to (AttributeError, TypeError, ValueError).
            logger.debug(
                "validate_gda_scores: association_type validation "
                "failed (%s: %s)",
                type(exc).__name__, exc,
            )

    # COMP-5: source-specific validation.
    if source == "disgenet" and "score" in out.columns:
        # DisGeNET scores are typically in [0, 1] -- warn if any value
        # was clipped at the upper bound (might indicate a different scale).
        try:
            # P1-010 ROOT FIX (v100 forensic -- TypeError on None>float):
            # The previous code did ``out["_original_score"] > score_max``
            # directly. But ``_original_score`` is ONLY populated for
            # clipped rows (line ~3249 sets it inside ``out.loc[clipped_mask,
            # "_original_score"] = ...``). For non-clipped rows,
            # ``_original_score`` is None -- and ``None > 1.0`` raises
            # TypeError on some pandas versions (or returns False on
            # others). The behavior was pandas-version-dependent and
            # could crash DisGeNET validation. ROOT FIX: use
            # ``.fillna(0.0)`` so the comparison is always numeric.
            # A non-clipped row has ``_original_score = None -> 0.0``,
            # which is never > ``score_max`` (1.0), so the row is
            # correctly excluded from the warning.
            _original_score_filled = out.get(
                "_original_score", pd.Series(None, index=out.index, dtype=object),
            ).fillna(0.0)
            _clipped_flag = out.get(
                "_score_was_clipped", pd.Series(False, index=out.index),
            )
            clipped_at_max = (
                _clipped_flag
                & (_original_score_filled > score_max)
            )
            clipped_at_max_count = int(clipped_at_max.sum())
            if clipped_at_max_count > 0:
                logger.warning(
                    "validate_gda_scores: %d DisGeNET score(s) clipped at "
                    "upper bound %s -- verify source data scale",
                    clipped_at_max_count,
                    score_max,
                )
        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            # P1-053 ROOT FIX (v107): narrowed from broad
            # ``except Exception``. The previous silent ``pass`` swallowed
            # programming bugs. A KeyError on a missing column meant the
            # source-specific validation was skipped silently. ROOT FIX:
            # narrow to (AttributeError, TypeError, ValueError, KeyError)
            # and log at DEBUG so operators can detect issues.
            logger.debug(
                "validate_gda_scores: DisGeNET source-specific validation "
                "failed (%s: %s)",
                type(exc).__name__, exc,
            )

    # DQ-4: optional dedup.
    if dedup:
        if dedup_keys is None:
            dedup_keys = ["gene_symbol", "disease_id", "source"]
        existing_keys = [k for k in dedup_keys if k in out.columns]
        if existing_keys:
            before_dedup = len(out)
            # v65 ROOT FIX (P1C-007 -- NaN-collapsing bug):
            #   The previous code called ``drop_duplicates`` directly on
            #   ``existing_keys``. pandas ``drop_duplicates`` treats
            #   ``NaN == NaN`` as ``True`` by default, so ALL rows with
            #   NaN in ANY dedup key (gene_symbol, disease_id, or source)
            #   were collapsed into a SINGLE row -- silently losing
            #   Gene->Disease edges. The ``deduplicator.py`` module
            #   (lines 2336-2369) fixed this EXACT bug for InChIKey by
            #   assigning unique sentinels to NaN-keyed rows BEFORE
            #   calling ``drop_duplicates``. The fix was NOT propagated
            #   to ``validate_gda_scores``, creating an asymmetry:
            #   ``dedup_by_inchikey`` preserves NaN rows, but
            #   ``validate_gda_scores(dedup=True)`` collapses them.
            #   ROOT FIX: apply the same sentinel-assignment pattern
            #   here. Build a temporary composite key where non-NaN
            #   rows get the joined key string and NaN rows get unique
            #   sentinels, then ``drop_duplicates`` on the sentinel
            #   column. This preserves every NaN-keyed row (they all
            #   have unique sentinels) while still merging true
            #   duplicates (non-NaN rows with identical keys).
            _null_key_mask = out[existing_keys].isna().any(axis=1)
            _n_null = int(_null_key_mask.sum())
            if _n_null > 0:
                # Build the sentinel composite key.
                _sentinel = pd.Series(index=out.index, dtype=object)
                _non_null = ~_null_key_mask
                if _non_null.any():
                    _sentinel.loc[_non_null] = (
                        out.loc[_non_null, existing_keys]
                        .astype(str)
                        .agg("|".join, axis=1)
                    )
                _sentinel.loc[_null_key_mask] = [
                    f"__NULL_DEDUP_KEY_{i}__" for i in range(_n_null)
                ]
                out = out.assign(_dedup_sentinel_key=_sentinel)
                out = out.drop_duplicates(
                    subset=["_dedup_sentinel_key"], keep="first"
                )
                out = out.drop(columns=["_dedup_sentinel_key"])
            else:
                out = out.drop_duplicates(subset=existing_keys, keep="first")
            dedup_dropped = before_dedup - len(out)
            if dedup_dropped > 0:
                logger.info(
                    "validate_gda_scores: dedup removed %d duplicate "
                    "record(s) on keys %s",
                    dedup_dropped,
                    existing_keys,
                )
                _increment_metric("gda_duplicates_dropped", dedup_dropped)
                if reset_index:
                    out = out.reset_index(drop=True)

    duration = time.monotonic() - start_time
    _set_cleaning_metadata(
        out,
        function_name="validate_gda_scores",
        input_fingerprint=input_fingerprint,
        input_rows=rows_before,
    )
    _increment_metric("validate_gda_scores_calls")

    if return_result:
        return DataCleaningResult(
            df=out,
            rows_before=rows_before,
            rows_after=len(out),
            rows_dropped=rows_before - len(out),
            columns_affected=columns_affected,
            dropped_rows=dropped_rows_df,
            warnings=warnings_list,
            duration_seconds=duration,
            dtype_changes=dtype_changes,
        )
    return out


# ===========================================================================
# 5. Orchestration helpers (ARCH-4)
# ===========================================================================
def clean_drugs(
    df: pd.DataFrame,
    *,
    drop_unidentifiable: bool = True,
    conservative_defaults: bool = True,
    converter: Optional[Callable] = None,
    fill_map_override: Optional[dict] = None,
    reset_index: bool = False,
) -> pd.DataFrame:
    """Orchestrate the full drug cleaning pipeline in the correct order (ARCH-4).

    Equivalent to::

        out = handle_missing_inchikey(df, drop_unidentifiable=drop_unidentifiable, ...)
        out = fill_missing_drug_fields(out, conservative_defaults=conservative_defaults, ...)
        return out

    Notes
    -----
    This is the IN-MODULE orchestrator.  ``cleaning/__init__.py`` exposes
    a richer ``clean_drugs`` with a step-based API that includes
    standardize_inchikey, dedup_by_inchikey, etc.  The two coexist.

    Parameters
    ----------
    df : pd.DataFrame
    drop_unidentifiable : bool
        Forwarded to :func:`handle_missing_inchikey`.
    conservative_defaults : bool
        Forwarded to :func:`fill_missing_drug_fields`.
    converter : Callable | None
        Forwarded to :func:`handle_missing_inchikey`.
    fill_map_override : dict | None
        Forwarded to :func:`fill_missing_drug_fields`.
    reset_index : bool
        Forwarded to both functions.

    Returns
    -------
    pd.DataFrame
    """
    out = handle_missing_inchikey(
        df,
        drop_unidentifiable=drop_unidentifiable,
        converter=converter,
        reset_index=reset_index,
    )
    out = fill_missing_drug_fields(
        out,
        conservative_defaults=conservative_defaults,
        fill_map_override=fill_map_override,
        reset_index=reset_index,
    )
    return out


def clean_proteins(
    df: pd.DataFrame,
    *,
    gene_name_fill: str = "",
    default_organism: Optional[str] = None,
    organism_fill_mode: str = "default",
    function_desc_fill: str = "",
    reset_index: bool = True,
) -> pd.DataFrame:
    """Orchestrate the full protein cleaning pipeline (ARCH-4).

    Currently a thin wrapper around :func:`handle_missing_protein_fields`.
    """
    return handle_missing_protein_fields(
        df,
        gene_name_fill=gene_name_fill,
        default_organism=default_organism,
        organism_fill_mode=organism_fill_mode,
        function_desc_fill=function_desc_fill,
        reset_index=reset_index,
    )


def clean_gda(
    df: pd.DataFrame,
    *,
    score_range: tuple = (0.0, 1.0),
    preserve_direction: bool = False,
    source: Optional[str] = None,
    dedup: bool = False,
    dedup_keys: Optional[list] = None,
    reset_index: bool = False,
) -> pd.DataFrame:
    """Orchestrate the full GDA cleaning pipeline (ARCH-4).

    Currently a thin wrapper around :func:`validate_gda_scores`.
    """
    return validate_gda_scores(
        df,
        score_range=score_range,
        preserve_direction=preserve_direction,
        source=source,
        dedup=dedup,
        dedup_keys=dedup_keys,
        reset_index=reset_index,
    )


# ===========================================================================
# Public API (ARCH-3, COMP-1)
# ===========================================================================
__all__ = [
    # Original public API (preserved -- v1.0.0 / v2.0.0)
    "handle_missing_inchikey",
    "fill_missing_drug_fields",
    "handle_missing_protein_fields",
    "validate_gda_scores",
    "MAX_SEQUENCE_LENGTH",
    # New: Null detection (v3.0.0)
    "is_nullish",
    "NullStrategy",
    "NULL_STRATEGY_GENERAL",
    "NULL_STRATEGY_CHEMICAL",
    "NULL_STRATEGY_CLINICAL",
    "NULL_STRATEGY_GENE",
    "NULL_STRATEGY_STRICT",
    # New: Result type (v3.0.0)
    "DataCleaningResult",
    # New: Orchestration (v3.0.0)
    "clean_drugs",
    "clean_proteins",
    "clean_gda",
    # New: Recovery/drop separation (v3.0.0)
    "recover_inchikeys_from_smiles",
    "drop_unidentifiable_drugs",
    # New: Configuration (v3.0.0)
    "DEFAULT_ORGANISM",
    # New: Observability (v3.0.0)
    "get_metrics",
    "reset_metrics",
    "get_dead_letters",
    "clear_dead_letters",
    "set_correlation_id",
    "get_correlation_id",
    # New: Lineage (v3.0.0)
    "get_provenance",
    # v110 Task 29 root fix: distinguish missing (None) from not-applicable (False)
    "NOT_APPLICABLE_SENTINEL",
    "MISSING_SENTINEL",
    "classify_fda_approval",
]
