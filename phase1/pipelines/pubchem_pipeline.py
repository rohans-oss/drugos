# SPDX-License-Identifier: MIT
# (c) 2024-2026 Autonomous Drug Repurposing Platform -- Team Cosmic / VentureLab
# See LICENSE file for full text.
"""
PubChem enrichment pipeline -- institutional-grade production-ready rewrite.

This module implements ``PubChemPipeline``, a child of :class:`BasePipeline`
that enriches existing ``drugs`` table rows with physicochemical properties
fetched from PubChem PUG REST.  The output feeds the Phase 3 Graph
Transformer's molecular fingerprinting.

Life-safety context
-------------------
This pipeline processes chemical structure data that feeds a machine-learning
model predicting drug-disease relationships.  The model's predictions are
reviewed by pharmaceutical researchers who may launch clinical trials based
on them.  If the stereochemistry is wrong, the trial tests the wrong
enantiomer.  If the salt form is wrong, the trial tests the wrong
formulation.  If the molecular weight is wrong, the mass-spec calibration is
wrong.  If the CID is wrong, the entire entity-resolution chain is wrong.
Every one of these errors can kill people.  This file is therefore written
to the institutional-grade standard mandated by
``PUBCHEM_PIPELINE_MASTER_FIX_PROMPT.md`` (187 audit findings, 131 unique
issues across 16 verification domains).

Data flow
---------
``download()``
    1. Query ``drugs`` via the SQLAlchemy ORM (``select(Drug.inchikey).where(
       Drug.pubchem_cid.is_(None), Drug.inchikey.isnot(None),
       Drug.is_deleted == False)``).  Filter invalid InChIKey format in
       Python (SQLite fallback) and at the SQL level (PostgreSQL regex).
       ``ORDER BY inchikey ASC`` for determinism.  ``LIMIT`` when
       ``PUBCHEM_PIPELINE_MAX_RECORDS`` is set.
    2. Stream results to ``raw_data/pubchem/inchikeys_to_lookup.txt`` with
       a header comment and SHA-256 sidecar.
    3. Batch into groups of ``PUBCHEM_PIPELINE_BATCH_SIZE`` (default 95).
       POST each batch to PubChem PUG REST.  Save each batch's raw JSON
       response to ``raw_data/pubchem/pubchem_responses/batch_NNNN.json``
       with a SHA-256 sidecar -- supports replay without re-hitting PubChem.
    4. Cache the InChIKey list for ``PUBCHEM_PIPELINE_CACHE_TTL_SECONDS``
       (default 1 hour).  ``force_refresh=True`` always re-queries.

``clean(raw_path)``
    Pure transformation -- NO HTTP.  Loads the raw JSON archive produced by
    ``download()``, parses each response, validates the InChIKey matches
    the request, deduplicates by InChIKey (lowest CID wins), sanitizes
    empty strings to ``None``, validates numeric ranges, converts floats
    to ``Decimal``, extracts protonation state from the InChIKey, parses
    isotope labels from the SMILES, computes formal charge, and returns
    a DataFrame with the canonical column order.  Failures go to
    ``self.dead_letter_queue`` with reason codes.

``load(df, session=None)``
    Uses the passed session (caller-managed transaction boundary).
    Calls the existing ``bulk_update_drugs_from_pubchem`` (updates
    ``drugs.pubchem_cid``, ``molecular_formula``, ``molecular_weight``,
    ``smiles`` where ``pubchem_cid IS NULL``) and the new
    ``bulk_upsert_pubchem_compound_properties`` (persists all 15+
    physicochemical properties + lineage columns to the new
    ``pubchem_compound_properties`` table created by migration 005).
    Returns a :class:`LoadResult`.

Configuration
-------------
All tunables live in ``config/settings.py``:
``PUBCHEM_PIPELINE_BATCH_SIZE`` (default 95 -- 5% safety margin under
PubChem's 100-identifier hard limit), ``PUBCHEM_PIPELINE_MIN_BACKOFF``
(2.0s), ``PUBCHEM_PIPELINE_MAX_BACKOFF`` (32.0s),
``PUBCHEM_PIPELINE_READ_TIMEOUT`` (30.0s),
``PUBCHEM_PIPELINE_CACHE_TTL_SECONDS`` (3600),
``PUBCHEM_PIPELINE_CONCURRENCY`` (1 = sequential),
``PUBCHEM_PIPELINE_FETCH_SYNONYMS`` (False),
``PUBCHEM_PIPELINE_FETCH_CAS`` (False),
``PUBCHEM_PIPELINE_SPLIT_RETRY_MAX`` (20),
``PUBCHEM_PIPELINE_MAX_RECORDS`` (None = unlimited),
``PUBCHEM_CIRCUIT_BREAKER_THRESHOLD`` (5),
``PUBCHEM_CIRCUIT_BREAKER_RESET_SECONDS`` (60.0),
``PUBCHEM_PIPELINE_PROPERTIES`` (the 15 PubChem property names).
Connection / retry / API key reuse the ``ENTITY_RESOLUTION_PUBCHEM_*``
settings -- single source of truth.

Scientific caveats (see docs/pipelines/pubchem.md for full details)
-------------------------------------------------------------------
* **Stereochemistry**: ``canonical_smiles`` (no stereo) and
  ``isomeric_smiles`` (with stereo, isotopes, charge) are SEPARATE
  columns.  The Graph Transformer MUST use ``isomeric_smiles`` for
  fingerprinting.  (R)-thalidomide and (S)-thalidomide must remain
  distinguishable.
* **XLogP** is a PubChem XLogP3 QSAR prediction, NOT experimental logP.
  The ``xlogp_source = 'pubchem_xlogp3'`` flag makes this explicit.
* **TPSA** is calculated from the 2D structure, not measured.
* **CID** is the standardized (parent) CID -- two different salt forms of
  the same drug share the same parent CID.
* **HeavyAtomCount** excludes hydrogen (PubChem convention).
* **HBondDonorCount / HBondAcceptorCount** are Lipinski-style counts.
* **molecular_weight** is average MW using natural-abundance atomic
  weights.  **exact_mass** is monoisotopic mass -- use this for
  mass-spectrometry.

Schema contract
---------------
``pipelines/schema/v1.json#pubchem_enrichment.csv`` is the canonical
contract for this pipeline's output.  The ``clean()`` method's
``COLUMN_ORDER`` is the authoritative column ordering; the schema lists
the same columns (no more, no less).

Failure modes
-------------
* HTTP 4xx (except 429): permanent failure -- dead-letter, no retry.
* HTTP 429 / 5xx: transient -- retry with jittered backoff, respect
  ``Retry-After`` header, circuit breaker opens after 5 consecutive
  failures.
* InChIKey mismatch (response ≠ request): dead-letter with reason
  ``inchikey_mismatch``, do NOT store the response InChIKey.
* Invalid InChIKey format (regex fails): dead-letter with reason
  ``invalid_inchikey_format``.
* Out-of-range values (e.g. molecular_weight < 0): dead-letter with
  reason ``range_violation_<field>``, field set to None.
* Empty strings from PubChem: converted to ``None`` (SQL NULL) before
  persistence -- never stored as ``""``.

References
----------
* PubChem PUG REST: https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest
* InChIKey spec: https://www.inchi-trust.org/technical-faq/
* Lipinski Rule of 5: Lipinski CA et al., Adv Drug Deliv Rev 1997.
"""

from __future__ import annotations

# Standard library imports -- alphabetical.
import csv
import email.utils
import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Optional

# Third-party.
import pandas as pd
import requests

# Project imports.
from config.settings import (
    ENTITY_RESOLUTION_PUBCHEM_API_KEY,
    ENTITY_RESOLUTION_PUBCHEM_CA_BUNDLE,
    ENTITY_RESOLUTION_PUBCHEM_CALL_DELAY,
    ENTITY_RESOLUTION_PUBCHEM_CERT_PEM,
    ENTITY_RESOLUTION_PUBCHEM_KEY_PEM,
    ENTITY_RESOLUTION_PUBCHEM_MAX_RETRIES,
    ENTITY_RESOLUTION_PUBCHEM_REST_BASE,
    ENTITY_RESOLUTION_PUBCHEM_STRICT_SALT_FORM,
    ENTITY_RESOLUTION_PUBCHEM_TIMEOUT,
    OPERATOR_ID,
    OTEL_ENABLED,
    PIPELINE_CONTACT_EMAIL,
    PROCESSED_DATA_DIR,
    PROMETHEUS_ENABLED,
    PUBCHEM_CIRCUIT_BREAKER_RESET_SECONDS,
    PUBCHEM_CIRCUIT_BREAKER_THRESHOLD,
    PUBCHEM_PIPELINE_BATCH_SIZE,
    PUBCHEM_PIPELINE_CACHE_TTL_SECONDS,
    PUBCHEM_PIPELINE_CONCURRENCY,
    PUBCHEM_PIPELINE_FETCH_CAS,
    PUBCHEM_PIPELINE_FETCH_SYNONYMS,
    PUBCHEM_PIPELINE_MAX_BACKOFF,
    PUBCHEM_PIPELINE_MAX_RECORDS,
    PUBCHEM_PIPELINE_MIN_BACKOFF,
    PUBCHEM_PIPELINE_PROPERTIES,
    PUBCHEM_PIPELINE_READ_TIMEOUT,
    PUBCHEM_PIPELINE_RAW_RESPONSE_RETENTION_DAYS,
    PUBCHEM_PIPELINE_SPLIT_RETRY_MAX,
    PUBCHEM_REST_BASE,
    RDKIT_AVAILABLE,
)
from database.connection import get_db_session
from database.loaders import (
    UpsertResult,
    bulk_update_drugs_from_pubchem,
    bulk_upsert_pubchem_compound_properties,
)
from database.models import Drug
from pipelines.base_pipeline import BasePipeline, LoadResult
from sqlalchemy import select
# v29 ROOT FIX (audit P1-24): canonical ID normalization at the OUTPUT
# boundary so PubChem's InChIKey + CID columns join cleanly with ChEMBL
# and DrugBank regardless of the case / format PubChem's PUG-REST returned.
from cleaning._constants import normalize_inchikey, normalize_pubchem_cid

# Optional OpenTelemetry -- only imported when OTEL_ENABLED=True.
if OTEL_ENABLED:  # pragma: no cover -- exercised only in instrumented envs
    try:
        from opentelemetry import trace

        _tracer = trace.get_tracer(__name__)
    except ImportError:  # pragma: no cover
        _tracer = None
else:
    _tracer = None

# Optional Prometheus metrics -- only imported when PROMETHEUS_ENABLED=True.
if PROMETHEUS_ENABLED:  # pragma: no cover
    try:
        from prometheus_client import Counter as PromCounter
        from prometheus_client import Gauge as PromGauge
        from prometheus_client import Histogram as PromHistogram

        _PUBCHEM_BATCHES_TOTAL = PromCounter(
            "pubchem_batches_total",
            "Total PubChem batches processed",
            ["status"],
        )
        _PUBCHEM_RETRIES_TOTAL = PromCounter(
            "pubchem_retries_total",
            "Total PubChem retries",
            ["status_code"],
        )
        _PUBCHEM_RECORDS_LOADED = PromGauge(
            "pubchem_records_loaded",
            "PubChem records loaded into DB",
        )
        _PUBCHEM_API_LATENCY = PromHistogram(
            "pubchem_api_latency_seconds",
            "PubChem API latency",
            ["endpoint"],
        )
    except ImportError:  # pragma: no cover
        _PUBCHEM_BATCHES_TOTAL = None
        _PUBCHEM_RETRIES_TOTAL = None
        _PUBCHEM_RECORDS_LOADED = None
        _PUBCHEM_API_LATENCY = None
else:
    _PUBCHEM_BATCHES_TOTAL = None
    _PUBCHEM_RETRIES_TOTAL = None
    _PUBCHEM_RECORDS_LOADED = None
    _PUBCHEM_API_LATENCY = None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants -- these are NOT configuration knobs (those live in
# settings.py).  These are immutable scientific / protocol constants.
# ---------------------------------------------------------------------------

# v41 ROOT FIX (P1 #24): use the CANONICAL InChIKey regex from _constants
# instead of defining a 7th local copy. All modules should import from
# cleaning._constants to ensure there is exactly ONE definition.
from cleaning._constants import CANONICAL_INCHIKEY_REGEX as INCHIKEY_RE  # noqa: E402

# PubChem PUG REST fault response shape -- error responses have a ``Fault`` key.
# Detecting this prevents mis-parsing an error as a valid PropertyTable (INT-13).
PUBCHEM_FAULT_KEY = "Fault"

# HTTP status codes that are PERMANENT failures -- never retried (REL-2, DESIGN-12).
# 429 is NOT in this set (it's a rate-limit signal, retried with Retry-After).
PERMANENT_STATUS: frozenset[int] = frozenset(
    {400, 401, 403, 404, 405, 406, 410, 422}
)

# HTTP status codes that are TRANSIENT -- retried with jittered backoff (REL-2).
TRANSIENT_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

# Network exceptions treated as retryable (REL-11).
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ContentDecodingError,
)

# Range validation for fetched numeric properties (SCI-17, DQ-4).
# Out-of-range values are dead-lettered with reason "range_violation_<field>"
# and the field is set to None before persistence.
RANGES: dict[str, tuple[float, float]] = {
    "molecular_weight": (0.0, 100_000.0),       # Da; >10K likely a protein
    "exact_mass": (0.0, 100_000.0),
    "xlogp": (-5.0, 15.0),
    "tpsa": (0.0, 500.0),
    "complexity": (0.0, 10_000.0),
    "h_bond_donor_count": (0.0, 50.0),
    "h_bond_acceptor_count": (0.0, 50.0),
    "rotatable_bond_count": (0.0, 100.0),
    "heavy_atom_count": (1.0, 500.0),
    "pubchem_cid": (1.0, 1e12),
    "formal_charge": (-50.0, 50.0),
}

# Canonical column order for the cleaned DataFrame and CSV (COMP-10, INT-8).
# ``clean()`` reindexes the DataFrame to this order before returning.
COLUMN_ORDER: tuple[str, ...] = (
    # Identity
    "inchikey",
    "pubchem_cid",
    # Structural
    "molecular_formula",
    "molecular_weight",
    "exact_mass",
    "canonical_smiles",
    "isomeric_smiles",
    "inchi",
    "iupac_name",
    "cas_number",
    # Physicochemical (predicted/calculated -- see source flags)
    "xlogp",
    "xlogp_source",
    "tpsa",
    "tpsa_source",
    "complexity",
    # Counts
    "h_bond_donor_count",
    "h_bond_acceptor_count",
    "rotatable_bond_count",
    "heavy_atom_count",
    "formal_charge",
    "isotope_info",
    "salt_form",
    "protonation_state",
    # Lineage (Domain 16)
    "source",
    "source_id",
    "source_version",
    "download_date",
    "download_method",
    "pipeline_run_id",
    "input_checksum",
    "transformations",
    "as_of_date",
    # FIX-P1-B-4 (audit P1): per-record lineage stamped by
    # ``_parse_pubchem_response`` (lines ~2718-2721). The previous
    # ``COLUMN_ORDER`` omitted these two columns, so the
    # ``df.reindex(columns=list(COLUMN_ORDER))`` call at the end of
    # ``clean()`` silently DROPPED them. Adding them here preserves the
    # lineage through reindexing so callers (and the DB loader) can
    # trace each record back to its source batch + response SHA-256.
    "_source_batch_idx",
    "_source_response_sha256",
)

# Backward-compat column renames -- when reading legacy CSVs produced by
# the old pipeline, rename to the new schema (COMP-8).
COLUMN_RENAMES: dict[str, str] = {
    "hbond_donor_count": "h_bond_donor_count",
    "hbond_acceptor_count": "h_bond_acceptor_count",
    "smiles": "isomeric_smiles",  # legacy singular smiles -> isomeric
}

# String values that are treated as NULL by ``_sanitize_string`` (SCI-18, DQ-3).
NULL_STRING_VALUES: frozenset[str] = frozenset(
    {"", "nan", "none", "null", "n/a", "unknown", "-"}
)

# V19 ROOT FIX (PS-1 / SW-2 -- patient safety, scientific correctness):
# The InChIKey's last character is NOT a 4-state protonation flag. Per the
# official InChI Trust technical FAQ (https://www.inchi-trust.org/technical-faq/),
# the InChIKey structure is:
#   • chars 1-14: connectivity hash
#   • chars 16-25: remaining-layers hash (stereo /b/t, isotope /i, charge /q,
#     proton /p -- all hashed together)
#   • char 27 (last): version flag -- 'S' = Standard InChI, 'N' = Non-standard
#     InChI. ONLY those two values are spec-defined.
#
# V18 (and the V11 audit's own parenthetical recommendation) BOTH misread
# the standard: they treated the last char as a 4-state protonation flag
# (N/M/P/S). Because real-world InChIKeys almost always end in 'S'
# (Standard), V18's mapping labeled virtually every drug as
# `salt_form="salt_form"` -- including plain neutral molecules like aspirin,
# caffeine, and paracetamol. This is the patient-safety residual the
# V19 forensic re-audit flagged.
#
# Real protonation state is encoded in the InChI string's `/q` (formal
# charge) and `/p` (proton balance) layers -- NOT in the InChIKey. The
# V19 fix:
#   1. Adds `_extract_inchikey_version_flag()` -- returns 'S' or 'N' (the
#      only spec-defined values). Stored as `inchikey_version_flag`.
#   2. Adds `_extract_protonation_from_inchi()` -- parses the InChI string's
#      `/p` and `/q` layers to derive the actual protonation state.
#      Returns one of: 'neutral', 'protonated', 'deprotonated', 'zwitterion',
#      'salt_form' (multi-component), or None when the InChI is unavailable.
#   3. `_extract_salt_form()` is updated to take BOTH inchikey and inchi --
#      it derives salt_form from the InChI when available, and returns None
#      (NOT a fabricated 4-state mapping) when only the InChIKey is present.
#   4. `_extract_protonation_state()` is updated to take BOTH inchikey and
#      inchi -- same logic.
INCHIKEY_VERSION_FLAGS: frozenset[str] = frozenset({"S", "N"})

# Default timeout for the synonym / CAS lookup endpoint (separate from the
# main batch endpoint because it's per-CID, not per-batch).
_SYNONYM_LOOKUP_TIMEOUT: tuple[float, float] = (10.0, 15.0)


# ``__all__`` -- explicit exports (DOC-11).  Prevents ``from
# pipelines.pubchem_pipeline import *`` from exposing internal helpers.
__all__ = [
    "PubChemPipeline",
    "INCHIKEY_RE",
    "COLUMN_ORDER",
    "COLUMN_RENAMES",
    "RANGES",
    "PERMANENT_STATUS",
    "TRANSIENT_STATUS",
    "PubChemPipelineError",
    "PubChemUnreachableError",
    "PubChemResponseSchemaError",
]


# ---------------------------------------------------------------------------
# Pipeline-specific exceptions (REL-9).
# ---------------------------------------------------------------------------


class PubChemPipelineError(Exception):
    """Base exception for PubChem pipeline failures."""


class PubChemUnreachableError(PubChemPipelineError):
    """Raised when the first 3 batches all fail with ConnectionError.

    Indicates PubChem is completely unreachable (DNS failure, firewall
    block, regional outage).  The Airflow DAG should alert and retry later
    rather than burning through 100 more batches.
    """


class PubChemResponseSchemaError(PubChemPipelineError):
    """Raised when PubChem returns a response with an unexpected schema.

    E.g., the ``PropertyTable.Properties`` key is missing -- indicates
    PubChem has changed its API response format.  Failing fast prevents
    silent data corruption.
    """


# ---------------------------------------------------------------------------
# Helper functions (module-level, free of side effects -- easier to unit test).
# ---------------------------------------------------------------------------


def _sanitize_string(value: Any) -> Optional[str]:
    """Convert a value to a clean string or ``None`` (SCI-18, DQ-3).

    ``""``, whitespace-only strings, and the literal sentinel values in
    :data:`NULL_STRING_VALUES` become ``None``.  Non-string inputs are
    stringified first (then re-checked).  Non-empty strings are stripped
    of leading/trailing whitespace.

    Why this matters: PubChem occasionally returns ``""`` for a field
    (e.g., ``MolecularFormula: ""``).  The legacy pipeline stored ``""``
    and the loader's ``COALESCE(:field, drugs.field)`` SQL treated ``""``
    as non-NULL -- **silently overwriting existing real data with empty
    strings across the entire drugs table**.  This is silent data
    corruption.  Converting to ``None`` makes ``COALESCE(NULL, existing)``
    preserve existing data.
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value
    elif isinstance(value, bool):
        # Booleans are not strings -- reject explicitly to avoid
        # ``str(True) == "True"`` being persisted as a chemical name.
        return None
    else:
        s = str(value)
    stripped = s.strip()
    if stripped.lower() in NULL_STRING_VALUES:
        return None
    return stripped


def _extract_inchikey_version_flag(inchikey: Optional[str]) -> Optional[str]:
    """V19 ROOT FIX (PS-1): extract the InChIKey's version flag (last char).

    Per the InChI Trust standard, the last char of an InChIKey is a
    2-value version flag: ``'S'`` = Standard InChI, ``'N'`` = Non-standard
    InChI. No other values are spec-defined.

    Returns ``'S'``, ``'N'``, or ``None`` if the InChIKey is malformed.
    """
    if not isinstance(inchikey, str) or not INCHIKEY_RE.match(inchikey):
        return None
    last = inchikey[-1]
    if last not in INCHIKEY_VERSION_FLAGS:
        return None
    return last


# Regex to extract the /p (proton balance) and /q (formal charge) layers
# from an InChI string. InChI layers are slash-separated, e.g.:
#   InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12-5/h2-5H,1H3,(H,11,12)/p-1
#                                                                 ^^^^^
#                                                                 /p layer
# /p<N> means N protons were REMOVED (deprotonated, net negative).
# /p+N would mean protons ADDED but the InChI spec uses /p-<N> for both
# directions; positive /p values are not emitted (the convention is:
# /p-1 = -1 proton = deprotonated; absence of /p = neutral).
# /q<N> means formal charge N (e.g. /q+1 = +1 cation, /q-1 = -1 anion).
_INCHI_PROTON_LAYER_RE = re.compile(r"/p(-?\d+)")
_INCHI_CHARGE_LAYER_RE = re.compile(r"/q([+-]?\d+)")


def _extract_protonation_from_inchi(inchi: Optional[str]) -> Optional[str]:
    """V19 ROOT FIX (PS-1): derive the actual protonation state from the
    InChI string's ``/p`` (proton balance) and ``/q`` (formal charge) layers.

    Returns one of:
      - ``'neutral'``       -- no /p layer, no /q layer (or /q0)
      - ``'protonated'``    -- /q with positive charge (e.g. /q+1, amine salt)
      - ``'deprotonated'``  -- /p with negative value (e.g. /p-1, carboxylate)
                              OR /q with negative charge (e.g. /q-1)
      - ``'zwitterion'``    -- /q0 explicitly AND /p present (internal balance)
      - ``'salt_form'``     -- InChI has multiple disconnected components
                              (detected by counting '.' in the formula layer)
                              AND any component has /q≠0 -- this is the only
                              true "salt" case per IUPAC definition
      - ``None``            -- InChI string unavailable / unparseable

    Scientific basis:
      - InChI /p layer: https://www.inchi-trust.org/technical-faq/
        "Protons removed (negative /p) or added (positive /p) relative to
        the neutral parent structure as supplied."
      - InChI /q layer: formal charge on the entire structure.
      - Multi-component InChI (salt vs covalent): the /formula layer
        contains a '.' separator (e.g. "C6H8O6.HCl" -> ascorbate HCl salt).
    """
    if not isinstance(inchi, str) or not inchi.strip():
        return None
    inchi = inchi.strip()
    if not inchi.startswith("InChI="):
        return None

    # Extract /p and /q layers (only the LAST occurrence matters per spec).
    p_match = None
    for m in _INCHI_PROTON_LAYER_RE.finditer(inchi):
        p_match = m
    q_match = None
    for m in _INCHI_CHARGE_LAYER_RE.finditer(inchi):
        q_match = m

    p_val = int(p_match.group(1)) if p_match else 0
    q_val = int(q_match.group(1)) if q_match else 0

    # Detect multi-component InChI (salt / coordination complex).
    # InChI structure: "InChI=1S/<formula>/<connections>/<hydrogens>/<q>/<p>"
    # When split by "/", [0]="InChI=1S", [1]=formula, [2]=connections, etc.
    # Multi-component formulas contain a '.' (e.g. "C6H8O6.H3N" or "C6H8O6.ClH").
    formula_layer = ""
    try:
        # Strip the "InChI=1S/" or "InChI=1/" prefix.
        after_prefix = inchi.split("/", 2)
        if len(after_prefix) >= 2:
            formula_layer = after_prefix[1]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        # v84 FORENSIC ROOT FIX (BUG #32): narrowed from broad
        # ``except Exception  # noqa: BLE001 -- defensive parsing``.
        # The previous code caught ALL exceptions -- including
        # programming bugs (AttributeError, IndexError, RuntimeError)
        # -- and silently set ``formula_layer = ""``. A bug in the
        # InChI splitting logic was masked, causing the protonation-
        # state detector to silently classify every multi-component
        # InChI as "neutral" -- corrupting the Drug salt/protonation
        # signal the RL ranker uses for safety filtering.
        # ROOT FIX: catch ONLY the expected parsing exceptions. Any
        # other exception (programming bug) propagates so it surfaces
        # immediately instead of silently misclassifying compounds.
        formula_layer = ""
    is_multi_component = "." in formula_layer

    # Decision tree (root-level, scientifically grounded):
    if is_multi_component and q_val != 0:
        # Multiple disconnected components with a net non-zero formal charge
        # -> ionic salt (e.g. NaCl, procaine-HCl).
        return "salt_form"
    if p_val < 0:
        # Protons removed -> deprotonated (carboxylate, phenolate, etc.).
        return "deprotonated"
    if p_val > 0:
        # Protons added -> protonated (ammonium, protonated heterocycle).
        return "protonated"
    if q_val > 0:
        # Net positive charge without /p -> cation (e.g. quaternary ammonium).
        return "protonated"
    if q_val < 0:
        # Net negative charge without /p -> anion (e.g. sulfate, deprotonated enolate).
        return "deprotonated"
    if p_match is not None and q_val == 0:
        # /p present but /q=0 -> internal proton transfer (zwitterion).
        return "zwitterion"
    # No /p, no /q (or /q0) -> neutral.
    return "neutral"


def _extract_protonation_state(
    inchikey: Optional[str],
    inchi: Optional[str] = None,
) -> Optional[str]:
    """Extract the protonation state (V19 ROOT FIX for PS-1 / SW-2).

    Per the V19 forensic re-audit, the InChIKey's last char is a 2-value
    version flag (S/N), NOT a 4-state protonation flag. Real protonation
    state is derived from the InChI string's ``/p`` and ``/q`` layers.

    Args:
        inchikey: The InChIKey (used only for validation/logging context).
        inchi:    The full InChI string (preferred source of protonation info).

    Returns:
        One of ``{'neutral', 'protonated', 'deprotonated', 'zwitterion',
        'salt_form'}`` or ``None`` when neither the InChI nor a valid
        InChIKey is available.

    V18 BACKWARD-COMPAT NOTE:
        The V18 4-state N/M/P/S mapping is REMOVED. Callers that previously
        received 'N'/'M'/'P'/'S' from this function now receive
        'neutral'/'deprotonated'/'protonated'/'salt_form' (the full word) --
        or None when the InChI is unavailable. The previous behavior of
        returning 'S' for virtually every drug (because real InChIKeys
        almost always end in 'S' = Standard) was a patient-safety bug.
    """
    # Preferred path: derive from the InChI string.
    if inchi is not None:
        return _extract_protonation_from_inchi(inchi)
    # Fallback: InChI string unavailable. We can NOT derive protonation
    # from the InChIKey alone -- the last char is a version flag, not a
    # protonation flag. Return None rather than fabricating a wrong label.
    if inchikey is not None:
        logger.debug(
            "PS-1 V19: inchi string unavailable for inchikey=%s; cannot "
            "derive protonation state from key alone (last char is a "
            "version flag, not a protonation flag). Returning None.",
            inchikey,
        )
    return None


def _extract_salt_form(
    inchikey: Optional[str],
    inchi: Optional[str] = None,
) -> Optional[str]:
    """Derive a human-readable salt form (V19 ROOT FIX for PS-1 / SW-2).

    Per the V19 forensic re-audit, the InChIKey's last char is NOT a
    salt indicator. Salt form is derived from the InChI string by
    detecting multiple disconnected components (the '.' separator in
    the formula layer) combined with non-zero formal charge.

    Args:
        inchikey: The InChIKey (used only for validation/logging context).
        inchi:    The full InChI string (preferred source).

    Returns:
        - ``'salt_form'``      when the InChI has multiple charged components
                               (true ionic salt -- e.g. NaCl, procaine·HCl)
        - ``'neutral'``        when the InChI is single-component, no /p, /q=0
        - ``'protonated'``     when /p>0 or /q>0 (cation)
        - ``'deprotonated'``   when /p<0 or /q<0 (anion)
        - ``'zwitterion'``     when /p present and /q=0
        - ``None``             when the InChI is unavailable (we will NOT
                               fabricate a label from the InChIKey's version
                               flag -- that was the V18 patient-safety bug)

    V18 BACKWARD-COMP NOTE:
        The V18 4-state mapping (N->neutral, M->deprotonated, P->protonated,
        S->salt_form) is REMOVED. Real-world InChIKeys almost always end
        in 'S' (Standard), so V18 labeled plain neutral molecules like
        aspirin as "salt_form" -- selecting wrong formulations for wet-lab
        trial. V19 returns None when the InChI is unavailable, which is
        safer than a fabricated label.
    """
    p = _extract_protonation_state(inchikey, inchi)
    return p  # same taxonomy; caller can switch on the same words


def _extract_isotope_info(smiles: Optional[str]) -> Optional[str]:
    """Parse isotope labels from a SMILES string (SCI-14).

    Returns a JSON-serialised dict like ``'{"F": 18, "C": 11}'`` or
    ``None`` when the SMILES contains no isotope annotations.  Recognises
    the OpenSMILES isotope syntax ``[<n><Element>]`` (e.g. ``[18F]``,
    ``[13C]``, ``[2H]``/``[D]``, ``[3H]``/``[T]``).
    """
    if not isinstance(smiles, str) or not smiles:
        return None
    # Find all ``[<digits><Letter(s)>]`` tokens.
    matches = re.findall(r"\[(\d{1,3})([A-Z][a-z]?)\]", smiles)
    # Handle deuterium / tritium shorthand.
    if "[D]" in smiles or "[2H]" in smiles:
        matches.append(("2", "H"))
    if "[T]" in smiles or "[3H]" in smiles:
        matches.append(("3", "H"))
    if not matches:
        return None
    isotope_map: dict[str, int] = {}
    for n_str, element in matches:
        try:
            n = int(n_str)
        except ValueError:
            continue
        # Keep the highest isotope label per element (defensive).
        if element not in isotope_map or n > isotope_map[element]:
            isotope_map[element] = n
    if not isotope_map:
        return None
    return json.dumps(isotope_map, sort_keys=True)


def _extract_formal_charge(smiles: Optional[str]) -> Optional[int]:
    """Parse the formal charge from a SMILES string (SCI-15).

    Uses RDKit when available (authoritative).  Falls back to a SMILES
    token heuristic that counts ``+n`` / ``-n`` annotations inside atom
    brackets.  Returns ``None`` if the SMILES is empty or unparseable.
    """
    if not isinstance(smiles, str) or not smiles:
        return None
    if RDKIT_AVAILABLE:
        try:  # pragma: no cover -- exercised only when RDKit is installed
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return int(Chem.GetFormalCharge(mol))
        except (ImportError, ValueError, RuntimeError):  # noqa: BLE001 -- RDKit errors are opaque  # v85 FORENSIC ROOT FIX (BUG #51)
            pass
    # Heuristic fallback -- count +/- inside atom brackets.
    # ``[NH4+]`` -> +1; ``[Cl-]`` -> -1; ``[Ca+2]`` -> +2; ``[O-2]`` -> -2.
    total = 0
    found = False
    for token in re.findall(r"\[([^\]]+)\]", smiles):
        m = re.search(r"([+-])(\d*)$", token)
        if m:
            found = True
            sign = 1 if m.group(1) == "+" else -1
            n = int(m.group(2)) if m.group(2) else 1
            total += sign * n
    # v64 ROOT FIX (P1-020): return None (not 0) when no charge tokens
    # were found. The previous `return total if found else 0` conflated
    # "neutral molecule (genuine 0)" with "SMILES unparseable, defaulting
    # to 0". A molecule with an unparseable SMILES would get
    # formal_charge=0 (neutral), which is incorrect if the molecule is
    # actually charged but the SMILES parser failed. Root fix: return
    # None so the caller can distinguish "unknown" from "neutral" and
    # dead-letter the row rather than silently loading a wrong value.
    # Callers that need a bool/int default should coerce None themselves.
    return total if found else None


# ---------------------------------------------------------------------------
# PubChemPipeline
# ---------------------------------------------------------------------------


class PubChemPipeline(BasePipeline):
    """Institutional-grade PubChem enrichment pipeline.

    See the module docstring for the data flow diagram and scientific
    caveats.  Configuration is via ``config/settings.py`` env vars
    (``PUBCHEM_PIPELINE_*`` and reused ``ENTITY_RESOLUTION_PUBCHEM_*``).
    """

    source_name = "pubchem"
    processed_filename = "pubchem_enrichment.csv"

    # ------------------------------------------------------------------
    # Construction & config validation
    # ------------------------------------------------------------------

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the PubChem pipeline.

        Args:
            *args, **kwargs: passed through to ``BasePipeline.__init__``.
                Supported kwargs (see BasePipeline): run_id, correlation_id,
                triggered_by, as_of_date, freeze_version, snapshot_tag, seed.

        Raises:
            PubChemPipelineError: if a critical config is invalid.
        """
        super().__init__(*args, **kwargs)

        # Pull all settings into instance attributes so tests can override.
        # (ARCH-7, CONF-1 ... CONF-9 -- no module-level config constants.)
        self.batch_size: int = PUBCHEM_PIPELINE_BATCH_SIZE
        self.max_retries: int = ENTITY_RESOLUTION_PUBCHEM_MAX_RETRIES
        self.min_backoff: float = PUBCHEM_PIPELINE_MIN_BACKOFF
        self.max_backoff: float = PUBCHEM_PIPELINE_MAX_BACKOFF
        self.rate_limit_interval: float = ENTITY_RESOLUTION_PUBCHEM_CALL_DELAY
        self.connect_timeout: float = ENTITY_RESOLUTION_PUBCHEM_TIMEOUT
        self.read_timeout: float = PUBCHEM_PIPELINE_READ_TIMEOUT
        self.timeout: tuple[float, float] = (
            self.connect_timeout,
            self.read_timeout,
        )
        self.cache_ttl_seconds: int = PUBCHEM_PIPELINE_CACHE_TTL_SECONDS
        self.concurrency: int = max(1, PUBCHEM_PIPELINE_CONCURRENCY)
        self.fetch_synonyms: bool = PUBCHEM_PIPELINE_FETCH_SYNONYMS
        self.fetch_cas: bool = PUBCHEM_PIPELINE_FETCH_CAS
        # FIX-P1-B-2 (audit P1 -- ARCH-3 violation): ``clean()`` must be a
        # pure transformation with NO HTTP calls. The previous code called
        # ``_fetch_cas_for_cid`` inside ``clean()`` whenever
        # ``fetch_cas=True``, which (a) broke the architecture contract,
        # (b) made ``clean()`` non-idempotent -- re-running it produced
        # different ``cas_number`` values as PubChem's synonym table
        # evolved. The fix: CAS enrichment is gated behind
        # ``_allow_clean_http`` (default False) inside ``clean()`` and
        # moved to a dedicated ``enrich_cas()`` method that operators
        # call after ``clean()`` when they explicitly want live HTTP
        # enrichment. ``clean()`` emits a one-shot warning when
        # ``fetch_cas=True`` but ``_allow_clean_http=False`` so operators
        # know CAS enrichment was deliberately skipped during clean.
        self._allow_clean_http: bool = False
        self.split_retry_max: int = PUBCHEM_PIPELINE_SPLIT_RETRY_MAX
        self.max_records: Optional[int] = PUBCHEM_PIPELINE_MAX_RECORDS
        self.raw_response_retention_days: int = (
            PUBCHEM_PIPELINE_RAW_RESPONSE_RETENTION_DAYS
        )
        self.circuit_breaker_threshold: int = PUBCHEM_CIRCUIT_BREAKER_THRESHOLD
        self.circuit_breaker_reset_seconds: float = (
            PUBCHEM_CIRCUIT_BREAKER_RESET_SECONDS
        )
        self.pubchem_properties: list[str] = list(PUBCHEM_PIPELINE_PROPERTIES)
        self.rest_base: str = PUBCHEM_REST_BASE or ENTITY_RESOLUTION_PUBCHEM_REST_BASE
        self.api_key: Optional[str] = ENTITY_RESOLUTION_PUBCHEM_API_KEY
        self.ca_bundle: Optional[str] = ENTITY_RESOLUTION_PUBCHEM_CA_BUNDLE
        self.cert_pem: Optional[str] = ENTITY_RESOLUTION_PUBCHEM_CERT_PEM
        self.key_pem: Optional[str] = ENTITY_RESOLUTION_PUBCHEM_KEY_PEM
        self.strict_salt_form: bool = ENTITY_RESOLUTION_PUBCHEM_STRICT_SALT_FORM
        self.contact_email: str = PIPELINE_CONTACT_EMAIL
        self.operator_id: Optional[str] = OPERATOR_ID

        # Validate (CONF-8, CONF-12).
        self._validate_config()

        # Per-run mutable state -- reset at the start of ``download()``.
        self._api_call_count: int = 0
        self._retry_count: int = 0
        self._split_count: int = 0
        self._batch_count: int = 0
        self._avg_latency: float = 0.0
        self._total_latency: float = 0.0
        self._accumulated_records: dict[str, dict] = {}
        self._access_timestamp: Optional[datetime] = None
        self._input_checksum: Optional[str] = None
        self._force_refresh: bool = False
        self._consecutive_connection_failures: int = 0

        # User-Agent header -- required by PubChem ToS for automated clients.
        self._user_agent: str = (
            f"DrugRepurposingPlatform/1.0 (contact: {self.contact_email})"
        )

        logger.info(
            "[%s] PubChemPipeline initialised -- batch_size=%d, max_retries=%d, "
            "backoff=%.1f..%.1fs, timeout=(%.1f, %.1f), concurrency=%d, "
            "circuit_breaker=%d/%.1fs, fetch_cas=%s, fetch_synonyms=%s, "
            "rdkit=%s",
            self.source_name,
            self.batch_size,
            self.max_retries,
            self.min_backoff,
            self.max_backoff,
            self.connect_timeout,
            self.read_timeout,
            self.concurrency,
            self.circuit_breaker_threshold,
            self.circuit_breaker_reset_seconds,
            self.fetch_cas,
            self.fetch_synonyms,
            RDKIT_AVAILABLE,
        )

    def _validate_config(self) -> None:
        """Validate configuration values at construction time (CONF-8).

        Raises ``PubChemPipelineError`` if any critical config is invalid.
        Logs a warning for non-critical issues.
        """
        errors: list[str] = []
        if not (0 < self.batch_size <= 100):
            errors.append(
                f"PUBCHEM_PIPELINE_BATCH_SIZE must be in (0, 100], "
                f"got {self.batch_size}"
            )
        if self.max_retries < 0:
            errors.append(f"max_retries must be >= 0, got {self.max_retries}")
        if self.min_backoff <= 0:
            errors.append(f"min_backoff must be > 0, got {self.min_backoff}")
        if self.max_backoff < self.min_backoff:
            errors.append(
                f"max_backoff ({self.max_backoff}) must be >= min_backoff "
                f"({self.min_backoff})"
            )
        if self.rate_limit_interval < 0.2:
            # P1-16 ROOT FIX: Previously this only logged a warning, so a
            # misconfigured env var (e.g. PUBCHEM_CALL_DELAY=0.0) would
            # silently let the worker hammer PubChem at >5 req/sec and get
            # its IP banned for 24 hours. Now it raises PubChemPipelineError
            # so the operator is forced to fix the config before any HTTP
            # traffic is sent.
            raise PubChemPipelineError(
                f"rate_limit_interval={self.rate_limit_interval:.3f} is below "
                f"PubChem's 5 req/sec limit (0.2s) -- refusing to start: this "
                f"would get the worker IP banned for 24 hours. Set "
                f"ENTITY_RESOLUTION_PUBCHEM_CALL_DELAY >= 0.2."
            )
        if not self.rest_base or not self.rest_base.startswith(
            ("http://", "https://")
        ):
            errors.append(
                f"PUBCHEM_REST_BASE must be a valid HTTP(S) URL, got "
                f"{self.rest_base!r}"
            )
        if self.concurrency < 1:
            errors.append(f"concurrency must be >= 1, got {self.concurrency}")
        if self.cache_ttl_seconds < 0:
            errors.append(
                f"cache_ttl_seconds must be >= 0, got {self.cache_ttl_seconds}"
            )
        if self.circuit_breaker_threshold < 1:
            errors.append(
                f"circuit_breaker_threshold must be >= 1, "
                f"got {self.circuit_breaker_threshold}"
            )
        # v83 FORENSIC ROOT FIX (P2-18): if ``ca_bundle`` is set to an
        # EMPTY STRING (misconfiguration), the ``verify=(self.ca_bundle
        # if self.ca_bundle else True)`` ternary silently falls back to
        # ``verify=True`` (system CA store) instead of the intended CA
        # bundle -- a TLS-interception bypass that defeats the purpose of
        # pinning a custom CA bundle. ROOT FIX: validate that if
        # ``ca_bundle`` is set, it is non-empty AND points to an existing
        # readable file. Empty / non-existent / unreadable -> raise.
        if self.ca_bundle is not None:
            ca_str = str(self.ca_bundle).strip()
            if not ca_str:
                raise PubChemPipelineError(
                    "ENTITY_RESOLUTION_PUBCHEM_CA_BUNDLE is set to an empty "
                    "string -- this silently falls back to verify=True (system "
                    "CA store), bypassing the intended custom CA bundle. Set "
                    "it to a valid file path or unset it to use the system CA store."
                )
            ca_path = Path(ca_str)
            if not ca_path.exists():
                raise PubChemPipelineError(
                    f"ENTITY_RESOLUTION_PUBCHEM_CA_BUNDLE points to a "
                    f"non-existent file: {ca_str}. Fix the path or unset "
                    f"the env var to use the system CA store."
                )
            if not ca_path.is_file():
                raise PubChemPipelineError(
                    f"ENTITY_RESOLUTION_PUBCHEM_CA_BUNDLE is not a regular "
                    f"file: {ca_str} (could be a directory). Fix the path."
                )
        if errors:
            raise PubChemPipelineError(
                "PubChem pipeline config validation failed:\n  - "
                + "\n  - ".join(errors)
            )

    # ------------------------------------------------------------------
    # Public API: download
    # ------------------------------------------------------------------

    def download(self) -> Path:
        """Query the ``drugs`` table for InChIKeys needing enrichment and
        fetch their properties from PubChem PUG REST.

        Steps
        -----
        1. Resolve ``raw_dir`` (base class lazy-inits it).
        2. Check the cache for ``inchikeys_to_lookup.txt`` -- return early
           if fresh (within ``cache_ttl_seconds``) and SHA-256 sidecar
           verifies.  ``force_refresh`` bypasses the cache.
        3. Query ``drugs`` via the ORM (ARCH-12): soft-delete filter,
           InChIKey format filter, ORDER BY inchikey ASC, optional LIMIT.
        4. Stream to ``inchikeys_to_lookup.txt`` with a header comment,
           write the SHA-256 sidecar.
        5. Fetch PubChem responses in batches via ``_fetch_all_batches()``.
           Each batch's raw JSON is archived to
           ``raw_dir/pubchem_responses/batch_NNNN.json`` with its own
           SHA-256 sidecar -- supports replay without re-hitting PubChem.

        Returns
        -------
        Path
            Path to ``raw_dir / "inchikeys_to_lookup.txt"``.

        Side effects
        ------------
        * Writes ``inchikeys_to_lookup.txt`` and ``.sha256`` sidecar.
        * Writes ``pubchem_responses/batch_NNNN.json`` and per-batch
          ``.sha256`` sidecars.
        * Sets ``self.source_version`` to a string of the form
          ``"pubchem_pug_rest_as_of_<ISO 8601 UTC>"``.
        * Sets ``self._input_checksum`` to the SHA-256 of the lookup file.
        * Appends to ``self.dead_letter_queue`` for any failed InChIKey.
        * Records the access timestamp on ``self._access_timestamp``.
        """
        # v50 ROOT FIX: delegate to the unified downloader.
        try:
            from pipelines._v50_downloaders import download_pubchem_full
            if self.raw_dir is None:
                self._ensure_directories()
            downloaded = download_pubchem_full(self.raw_dir)
            enrichment_path = downloaded.get("enrichment")
            if enrichment_path and enrichment_path.exists():
                return enrichment_path
        except (OSError, ValueError, ConnectionError, TimeoutError) as exc:  # v85 FORENSIC ROOT FIX (BUG #51)
            logger.warning(
                "[%s] v50 downloader failed (%s) -- falling back to v49 path",
                self.source_name, exc,
            )

        # Lazy-init raw_dir (the base class inits it on first call to
        # ``run()`` / ``run_load_only()`` / ``run_download_and_clean_only()``
        # -- but ``download()`` can be called directly by tests).
        if self.raw_dir is None:
            self._ensure_directories()
        assert self.raw_dir is not None  # noqa: S101 -- invariant after _ensure

        # Reset per-run mutable state.
        self._api_call_count = 0
        self._retry_count = 0
        self._split_count = 0
        self._batch_count = 0
        self._avg_latency = 0.0
        self._total_latency = 0.0
        self._accumulated_records = {}
        self._access_timestamp = None
        self._consecutive_connection_failures = 0
        # Note: dead_letter_queue is shared with the base class -- do not
        # reset it here; ``run()`` controls its lifecycle.

        dest = self.raw_dir / "inchikeys_to_lookup.txt"

        # Cache check (CODE-15, DQ-14, IDEM-1, IDEM-2).
        if (
            not self._force_refresh
            and dest.exists()
            and self.cache_ttl_seconds > 0
        ):
            age_seconds = (
                datetime.now(timezone.utc)
                - datetime.fromtimestamp(dest.stat().st_mtime, tz=timezone.utc)
            ).total_seconds()
            if age_seconds < self.cache_ttl_seconds:
                if self._verify_sha256_sidecar(dest):
                    self._input_checksum = self._compute_sha256(dest)
                    logger.info(
                        "[%s] Using cached InChIKey list: path=%s, age=%ds, "
                        "size=%d bytes, sha256=%s",
                        self.source_name,
                        dest,
                        int(age_seconds),
                        dest.stat().st_size,
                        self._input_checksum,
                    )
                    # Still need to fetch PubChem responses -- they are
                    # not cached between runs (they could change if
                    # PubChem updates).
                    self._fetch_all_batches(dest)
                    return dest
                logger.warning(
                    "[%s] Cached file %s failed SHA-256 verification -- "
                    "re-querying",
                    self.source_name,
                    dest,
                )
            else:
                logger.info(
                    "[%s] Cached file stale (age=%ds > ttl=%ds) -- re-querying",
                    self.source_name,
                    int(age_seconds),
                    self.cache_ttl_seconds,
                )

        # ORM-based query (ARCH-12, DQ-7, DQ-8, DQ-10, IDEM-3, SEC-11, SEC-13).
        inchikeys: list[str] = []
        with get_db_session(
            pipeline_name=self.source_name,
            run_id=self.run_id,
            correlation_id=self.correlation_id,
        ) as session:
            stmt = (
                select(Drug.inchikey)
                .where(Drug.pubchem_cid.is_(None))
                .where(Drug.inchikey.isnot(None))
                .where(Drug.is_deleted == False)  # noqa: E712 -- ORM filter
                .order_by(Drug.inchikey.asc())
            )
            if self.max_records is not None and self.max_records > 0:
                stmt = stmt.limit(self.max_records)
            # Use yield_per for streaming on PostgreSQL -- on SQLite the
            # hint is ignored (PERF-10, PERF-11).
            for row in session.execute(stmt).yield_per(1000):
                ik = row.inchikey
                if ik and INCHIKEY_RE.match(ik):
                    inchikeys.append(ik)

        # Deduplicate preserving order (DQ-1).
        original_count = len(inchikeys)
        inchikeys = list(dict.fromkeys(inchikeys))
        if len(inchikeys) < original_count:
            logger.info(
                "[%s] Deduplicated %d -> %d InChIKeys",
                self.source_name,
                original_count,
                len(inchikeys),
            )

        # Write the lookup file with a header line (DQ-12, CODE-14).
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(
                f"# inchikeys_to_lookup -- generated "
                f"{datetime.now(timezone.utc).isoformat()} by "
                f"PubChemPipeline run_id={self.run_id}\n"
            )
            for ik in inchikeys:
                fh.write(ik + "\n")

        # Write SHA-256 sidecar (LIN-14, DQ-15, IDEM-2).
        self._input_checksum = self._compute_sha256(dest)
        sha256_path = dest.with_suffix(dest.suffix + ".sha256")
        with open(sha256_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"{self._input_checksum}  {dest.name}\n")

        if not inchikeys:
            logger.warning(
                "[%s] No drugs require PubChem enrichment -- skipping run. "
                "Run ChEMBL/DrugBank pipelines before PubChem.",
                self.source_name,
            )
        else:
            logger.info(
                "[%s] Found %d InChIKeys without PubChem CID -- fetching",
                self.source_name,
                len(inchikeys),
            )

        # Fetch PubChem responses (HTTP I/O -- happens in download(), NOT
        # in clean() per ARCH-3).
        self._fetch_all_batches(dest)

        return dest

    # ------------------------------------------------------------------
    # v80 FORENSIC ROOT FIX (P0-C3): v50 enrichment-CSV cleaner.
    # ------------------------------------------------------------------

    def _clean_v50_enrichment_csv(self, csv_path: Path) -> pd.DataFrame:
        """Parse the v50 ``pubchem_enrichment.csv`` into a COLUMN_ORDER DataFrame.

        The v50 downloader writes a CSV with columns: inchikey,
        pubchem_cid, canonical_smiles, xlogp, tpsa,
        h_bond_donor_count, h_bond_acceptor_count, rotatable_bond_count.
        This helper reads that CSV, validates each row, fills the
        remaining COLUMN_ORDER columns with sensible defaults / None,
        and returns a DataFrame with the full schema so downstream
        consumers (DB loader, entity resolution) work unchanged.

        Parameters
        ----------
        csv_path : Path
            Path to ``pubchem_enrichment.csv`` (v50 downloader output).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns in :data:`COLUMN_ORDER`.
        """
        from datetime import datetime, timezone

        logger.info(
            "[%s] Parsing v50 enrichment CSV: %s",
            self.source_name, csv_path,
        )
        try:
            raw_df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        except (pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
            logger.error(
                "[%s] Failed to read v50 enrichment CSV %s: %s",
                self.source_name, csv_path, exc,
            )
            return pd.DataFrame(columns=list(COLUMN_ORDER))

        if raw_df.empty:
            logger.warning(
                "[%s] v50 enrichment CSV is empty: %s",
                self.source_name, csv_path,
            )
            return pd.DataFrame(columns=list(COLUMN_ORDER))

        # Validate required columns.
        required_v50_cols = {"inchikey"}
        missing = required_v50_cols - set(raw_df.columns)
        if missing:
            logger.error(
                "[%s] v50 enrichment CSV missing required columns %s -- "
                "available: %s",
                self.source_name, missing, list(raw_df.columns),
            )
            return pd.DataFrame(columns=list(COLUMN_ORDER))

        now_iso = datetime.now(timezone.utc).isoformat()
        source_version = self.source_version or "pubchem_pug_rest_v50"
        run_id = self.run_id or "v50_run"

        records: list[dict] = []
        invalid_count = 0
        for idx, row in raw_df.iterrows():
            inchikey = str(row.get("inchikey") or "").strip()
            if not inchikey or not INCHIKEY_RE.match(inchikey):
                # Dead-letter invalid InChIKeys.
                invalid_count += 1
                self.dead_letter_queue.append({
                    "inchikey": inchikey,
                    "reason": "invalid_inchikey_in_v50_csv",
                    "row_idx": int(idx) if isinstance(idx, int) else None,
                    "timestamp": now_iso,
                })
                continue

            # Parse pubchem_cid (may be empty for unmatched InChIKeys).
            cid_raw = str(row.get("pubchem_cid") or "").strip()
            pubchem_cid: Optional[int] = None
            if cid_raw and cid_raw not in ("", "nan", "None"):
                try:
                    pubchem_cid = int(float(cid_raw))
                except (ValueError, TypeError):
                    pass

            # Parse numeric fields (xlogp, tpsa, counts).
            def _parse_float(v) -> Optional[float]:
                s = str(v or "").strip()
                if not s or s in ("nan", "None", "N/A", ""):
                    return None
                try:
                    return float(s)
                except (ValueError, TypeError):
                    return None

            def _parse_int(v) -> Optional[int]:
                s = str(v or "").strip()
                if not s or s in ("nan", "None", "N/A", ""):
                    return None
                try:
                    return int(float(s))
                except (ValueError, TypeError):
                    return None

            record = {
                # Identity
                "inchikey": inchikey,
                "pubchem_cid": pubchem_cid,
                # Structural (not provided by v50 CSV -- leave None)
                "molecular_formula": None,
                "molecular_weight": None,
                "exact_mass": None,
                "canonical_smiles": str(row.get("canonical_smiles") or "").strip() or None,
                "isomeric_smiles": None,
                "inchi": None,
                "iupac_name": None,
                "cas_number": None,
                # Physicochemical
                "xlogp": _parse_float(row.get("xlogp")),
                "xlogp_source": "pubchem_pug_rest" if _parse_float(row.get("xlogp")) is not None else None,
                "tpsa": _parse_float(row.get("tpsa")),
                "tpsa_source": "pubchem_pug_rest" if _parse_float(row.get("tpsa")) is not None else None,
                "complexity": None,
                # Counts
                "h_bond_donor_count": _parse_int(row.get("h_bond_donor_count")),
                "h_bond_acceptor_count": _parse_int(row.get("h_bond_acceptor_count")),
                "rotatable_bond_count": _parse_int(row.get("rotatable_bond_count")),
                "heavy_atom_count": None,
                "formal_charge": None,
                "isotope_info": None,
                "salt_form": None,
                "protonation_state": None,
                # Lineage
                "source": "pubchem",
                "source_id": str(pubchem_cid) if pubchem_cid is not None else "",
                "source_version": source_version,
                "download_date": now_iso,
                "download_method": "pug_rest_v50",
                "pipeline_run_id": run_id,
                "input_checksum": "",
                "transformations": "v50_csv_parse",
                "as_of_date": now_iso,
                "_source_batch_idx": 0,
                "_source_response_sha256": "",
            }
            records.append(record)

        if invalid_count > 0:
            logger.warning(
                "[%s] v50 enrichment CSV: %d / %d rows had invalid "
                "InChIKeys -- dead-lettered",
                self.source_name, invalid_count, len(raw_df),
            )

        # Build the output DataFrame with the full COLUMN_ORDER schema.
        out_df = pd.DataFrame(records, columns=list(COLUMN_ORDER))
        logger.info(
            "[%s] v50 enrichment CSV parsed: %d valid records, %d dead-lettered",
            self.source_name, len(out_df), invalid_count,
        )

        self._log_transformation(
            "v50_csv_parse",
            len(raw_df),
            {"valid": len(out_df), "invalid": invalid_count},
        )
        return out_df

    # ------------------------------------------------------------------
    # Public API: clean
    # ------------------------------------------------------------------

    def clean(self, raw_path: Path) -> pd.DataFrame:
        """Parse raw PubChem data into a cleaned DataFrame.

        Pure transformation -- NO HTTP calls (ARCH-3, ARCH-4).  Reads the
        raw archive produced by ``download()`` and parses each record
        into structured rows.

        v80 FORENSIC ROOT FIX (P0-C3 -- PubChemPipelineError in v50 mode):
          The v50 downloader returns ``pubchem_enrichment.csv`` -- a
          CSV with columns: inchikey, pubchem_cid, canonical_smiles,
          xlogp, tpsa, h_bond_donor_count, h_bond_acceptor_count,
          rotatable_bond_count. But the previous clean() expected
          ``raw_path`` to be an InChIKey LIST file
          (``inchikeys_to_lookup.txt``) AND expected a companion
          ``pubchem_responses/batch_*.json`` archive to exist. Neither
          exists in v50 mode -> clean() crashed with FileNotFoundError
          or returned an empty DataFrame, blocking the KG build.

          ROOT FIX: detect the input format from the file extension
          and content. If ``raw_path`` is a CSV with the enrichment
          schema (v50 mode), parse it directly via a new
          ``_clean_v50_enrichment_csv()`` helper. If it's the legacy
          InChIKey list + JSON archive (v49 mode), use the original
          parsing path. Both paths produce a DataFrame with the same
          ``COLUMN_ORDER`` schema so downstream consumers are unchanged.

        Parameters
        ----------
        raw_path : Path
            Either:
              - ``pubchem_enrichment.csv`` (v50 mode), OR
              - ``inchikeys_to_lookup.txt`` (v49 mode, with companion
                ``pubchem_responses/batch_*.json`` archive).

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame with columns in :data:`COLUMN_ORDER`.
            Empty DataFrame if no records were found.

        Side effects
        ------------
        * Appends invalid records to ``self.dead_letter_queue``.
        * Appends per-step transformations to ``self._transformation_log``.
        * Does NOT write any files -- the base class persists the returned
          DataFrame via ``_persist_cleaned_data()`` (ARCH-4, COMP-3,
          COMP-4, LIN-6, LIN-7, PERF-12).
        """
        # ARCH-3: clean() does NO HTTP.  Use ``responses`` library or a
        # network sentinel in tests to verify no requests are made.
        # FIX-P1-B-2 (audit P1 -- ARCH-3): when ``fetch_cas=True`` but the
        # operator has not opted into HTTP during clean (the default,
        # ``_allow_clean_http=False``), CAS enrichment is skipped inside
        # ``clean()``. Emit a one-shot warning so operators know CAS
        # enrichment was deliberately deferred to ``enrich_cas()``.
        if self.fetch_cas and not self._allow_clean_http:
            logger.warning(
                "[%s] fetch_cas=True but _allow_clean_http=False -- CAS "
                "enrichment is SKIPPED inside clean() to preserve ARCH-3 "
                "(pure transformation). Call enrich_cas(df) after clean() "
                "to perform live PubChem synonym lookup.",
                self.source_name,
            )

        # IDEM-5: seed RNG for reproducible jitter (used only if a
        # downstream consumer calls random from clean).
        if self.seed is not None:
            random.seed(self.seed & 0xFFFFFFFF)

        # v80 P0-C3: detect input format. The v50 downloader returns a
        # CSV with the enrichment schema; the v49 path returns an
        # InChIKey list + JSON archive. Dispatch to the right parser.
        if not raw_path.exists():
            raise FileNotFoundError(f"raw_path does not exist: {raw_path}")

        suffix = raw_path.suffix.lower()
        if suffix == ".csv":
            # v50 mode: raw_path is the enrichment CSV.
            logger.info(
                "[%s] v50 mode: parsing enrichment CSV %s",
                self.source_name, raw_path.name,
            )
            return self._clean_v50_enrichment_csv(raw_path)

        # v49 mode: raw_path is an InChIKey list file. Load it and the
        # companion JSON archive.
        logger.info(
            "[%s] v49 mode: parsing InChIKey list %s + JSON archive",
            self.source_name, raw_path.name,
        )
        with open(raw_path, "r", encoding="utf-8") as fh:
            requested_inchikeys: list[str] = []
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                requested_inchikeys.append(line)

        # COMP-8: apply legacy column renames (no-op for fresh CSVs).
        # (This is a backward-compat shim for any CSVs produced by the
        # old pipeline version -- it doesn't apply to the raw_path file,
        # which is an InChIKey list, not a CSV.)

        # Locate the raw JSON archive directory (ARCH-3).
        responses_dir = self.raw_dir / "pubchem_responses"
        if not responses_dir.exists():
            logger.warning(
                "[%s] No raw response archive at %s -- clean() will return "
                "an empty DataFrame",
                self.source_name,
                responses_dir,
            )
            return pd.DataFrame(columns=list(COLUMN_ORDER))

        response_files = sorted(responses_dir.glob("batch_*.json"))
        if not response_files:
            logger.warning(
                "[%s] No batch_*.json files in %s -- clean() will return "
                "an empty DataFrame",
                self.source_name,
                responses_dir,
            )
            return pd.DataFrame(columns=list(COLUMN_ORDER))

        # DQ-16: validate the raw_path file content -- every non-comment
        # line must be a valid InChIKey.
        # P1-025 ROOT FIX (Team-2 — document the div-by-zero invariant):
        #   The division ``len(invalid_in_raw) / len(requested_inchikeys)``
        #   below is guarded from div-by-zero by the LIST-COMPREHENSION
        #   invariant: if ``requested_inchikeys`` is EMPTY, the list
        #   comprehension ``[ik for ik in requested_inchikeys if ...]``
        #   yields an EMPTY list → ``invalid_in_raw`` is empty → the
        #   ``if invalid_in_raw:`` guard is False → the division is
        #   NEVER reached. So ``len(requested_inchikeys) > 0`` whenever
        #   the division executes. This is a subtle invariant — a future
        #   refactor that changes the list comprehension (e.g. to a
        #   generator or a filter) could break it. ROOT FIX: document
        #   the invariant here so future maintainers know the division
        #   is safe BY CONSTRUCTION, not by an explicit length check.
        #   This is a documentation-only fix; no code change.
        # INVARIANT: ``len(requested_inchikeys) > 0`` whenever the
        # division below executes, because:
        #   1. ``invalid_in_raw`` is built from a list comprehension
        #      over ``requested_inchikeys``.
        #   2. If ``requested_inchikeys`` is empty, ``invalid_in_raw``
        #      is also empty (list comprehension over empty = empty).
        #   3. The ``if invalid_in_raw:`` guard then evaluates False,
        #      so the division is NEVER reached.
        # Therefore: div-by-zero is impossible. If a future refactor
        # breaks this invariant, add an explicit
        # ``if not requested_inchikeys: return`` guard BEFORE this block.
        invalid_in_raw = [
            ik for ik in requested_inchikeys if not INCHIKEY_RE.match(ik)
        ]
        if invalid_in_raw:
            # Safe: see INVARIANT above — len(requested_inchikeys) > 0 here.
            invalid_pct = (len(invalid_in_raw) / len(requested_inchikeys)) * 100
            for ik in invalid_in_raw[:50]:
                self.dead_letter_queue.append(
                    {
                        "inchikey": ik,
                        "reason": "invalid_inchikey_in_raw_file",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            logger.warning(
                "[%s] %d InChIKeys in raw file failed format validation "
                "(%.1f%%) -- dead-lettered",
                self.source_name,
                len(invalid_in_raw),
                invalid_pct,
            )
            if invalid_pct > 50:
                raise PubChemPipelineError(
                    f"{invalid_pct:.1f}% of InChIKeys in {raw_path} are "
                    f"invalid -- refusing to process garbage input"
                )
            requested_inchikeys = [
                ik for ik in requested_inchikeys if INCHIKEY_RE.match(ik)
            ]

        self._log_transformation(
            "load_inchikeys",
            len(requested_inchikeys),
            {"source_file": str(raw_path)},
        )

        # Parse each batch file and accumulate records.
        all_records: list[dict] = []
        for batch_idx, batch_file in enumerate(response_files):
            try:
                with open(batch_file, "r", encoding="utf-8") as fh:
                    batch_data = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                logger.error(
                    "[%s] Could not read batch file %s: %s -- dead-lettering "
                    "all InChIKeys in this batch",
                    self.source_name,
                    batch_file,
                    exc,
                )
                # Compute the SHA-256 of the file for the dead-letter entry.
                try:
                    batch_sha = self._compute_sha256(batch_file)
                except (OSError, ValueError):  # noqa: BLE001  # v85 FORENSIC ROOT FIX (BUG #51)
                    batch_sha = None
                # Best-effort: dead-letter any InChIKey whose batch_idx
                # matches this file's index.  The mapping is approximate
                # because we don't have the original InChIKey list per
                # batch stored separately -- but the SHA-256 lets an
                # analyst trace back to the file.
                self.dead_letter_queue.append(
                    {
                        "reason": "batch_file_unreadable",
                        "batch_idx": batch_idx,
                        "batch_file": str(batch_file),
                        "batch_sha256": batch_sha,
                        "error": str(exc),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                continue

            # Compute the SHA-256 of the batch file for lineage (LIN-8).
            batch_sha256 = self._compute_sha256(batch_file)

            # Determine which InChIKeys were in this batch.  We use the
            # batch_idx to slice the requested_inchikeys list -- this
            # requires the batches to be processed in order, which they
            # are (sorted glob).
            batch_start = batch_idx * self.batch_size
            batch_end = batch_start + self.batch_size
            batch_requested = requested_inchikeys[batch_start:batch_end]

            records = self._parse_pubchem_response(
                batch_data,
                batch_requested,
                batch_idx=batch_idx,
                batch_sha256=batch_sha256,
            )
            all_records.extend(records)

        self._log_transformation(
            "parse_all_batches",
            len(all_records),
            {
                "batches_processed": len(response_files),
                "dead_lettered": len(self.dead_letter_queue),
            },
        )

        # Build the DataFrame with the canonical column order (COMP-10, INT-8).
        if not all_records:
            logger.warning(
                "[%s] No PubChem records parsed -- returning empty DataFrame",
                self.source_name,
            )
            return pd.DataFrame(columns=list(COLUMN_ORDER))

        df = pd.DataFrame.from_records(all_records)
        # Reindex to the canonical column order -- missing columns become
        # all-NaN, extra columns are dropped.
        df = df.reindex(columns=list(COLUMN_ORDER))

        # DQ-19: check for duplicate pubchem_cids (informational).
        if "pubchem_cid" in df.columns:
            cid_dupes = df[df["pubchem_cid"].notna()].duplicated(
                subset=["pubchem_cid"], keep=False
            )
            if cid_dupes.any():
                dupe_cids = df.loc[cid_dupes, "pubchem_cid"].unique().tolist()
                logger.warning(
                    "[%s] %d duplicate pubchem_cids found: %s",
                    self.source_name,
                    len(dupe_cids),
                    dupe_cids[:10],
                )

        # DQ-5: per-column NULL counts.
        null_counts = df.isnull().sum().to_dict()
        for col, cnt in null_counts.items():
            pct = (cnt / len(df) * 100) if len(df) else 0
            logger.info(
                "[%s] NULL count for %s: %d (%.1f%%)",
                self.source_name,
                col,
                cnt,
                pct,
            )

        # LOG-8: summary log.
        logger.info(
            "[%s] PubChem API calls: %d (batches=%d, retries=%d, splits=%d, "
            "avg_latency=%.2fs) -- %d records parsed, %d dead-lettered",
            self.source_name,
            self._api_call_count,
            self._batch_count,
            self._retry_count,
            self._split_count,
            self._avg_latency,
            len(df),
            len(self.dead_letter_queue),
        )

        # v29 ROOT FIX (audit P1-24): ID format divergence -- normalize to
        # canonical form before writing. ``inchikey`` is uppercased +
        # stripped; ``pubchem_cid`` is coerced to a plain Python ``int``
        # with leading zeros stripped. This guarantees downstream joins
        # against ChEMBL (InChIKey) and DrugBank (InChIKey via crosswalk)
        # succeed regardless of which source wrote the value. PubChem's
        # PUG-REST occasionally returns lowercase InChIKeys (e.g.
        # ``"bsynrymutxbxsq-uhfffaoysa-n"``) and zero-padded CIDs as
        # strings (e.g. ``"0002244"``); without this normalization, a
        # drug record from PubChem would NOT join with the same drug from
        # ChEMBL, silently dropping the PubChem property set from the
        # knowledge graph.
        if len(df) > 0:
            if "inchikey" in df.columns:
                df["inchikey"] = df["inchikey"].apply(
                    lambda x: normalize_inchikey(x)
                    if pd.notna(x) and x != "" else x
                )
            if "pubchem_cid" in df.columns:
                df["pubchem_cid"] = df["pubchem_cid"].apply(
                    lambda x: normalize_pubchem_cid(x)
                    if pd.notna(x) else x
                )

        return df

    # ------------------------------------------------------------------
    # Public API: enrich_cas
    # ------------------------------------------------------------------

    def enrich_cas(
        self,
        df: pd.DataFrame,
        *,
        force: bool = False,
    ) -> pd.DataFrame:
        """Live CAS-number enrichment via PubChem synonym lookup.

        FIX-P1-B-2 (audit P1 -- ARCH-3): the previous ``clean()``
        implementation made HTTP calls to PubChem whenever
        ``fetch_cas=True``, violating the architecture contract that
        ``clean()`` is a pure transformation. CAS enrichment is now
        segregated into this dedicated method so that:

        * ``clean()`` remains idempotent and HTTP-free.
        * Operators can choose whether / when to issue the live HTTP
          calls (rate-limit, retry, and circuit-breaker behaviour is
          inherited from ``_fetch_cas_for_cid``).

        Parameters
        ----------
        df : pandas.DataFrame
            Cleaned DataFrame produced by ``clean()``. Must contain
            ``pubchem_cid`` and ``inchikey`` columns.
        force : bool, default False
            When True, re-fetch CAS for rows that already have a
            ``cas_number`` value. When False (default), only rows with
            ``cas_number`` missing/empty are enriched.

        Returns
        -------
        pandas.DataFrame
            The same DataFrame with ``cas_number`` populated where
            PubChem returned a synonym. Rows that already had a CAS
            number are left untouched unless ``force=True``.

        Side effects
        ------------
        * Issues one HTTP request per row needing enrichment via
          ``_fetch_cas_for_cid`` (rate-limited per ``rate_limit_interval``).
        * Mutates ``self._api_call_count`` (informational).
        """
        if df.empty or not self.fetch_cas:
            return df
        if "pubchem_cid" not in df.columns:
            logger.warning(
                "[%s] enrich_cas: pubchem_cid column missing -- skipping",
                self.source_name,
            )
            return df

        # Temporarily enable HTTP for the lookup helper path. We do NOT
        # mutate self._allow_clean_http (which gates the call from inside
        # ``clean()``); instead we call ``_fetch_cas_for_cid`` directly.
        enriched = 0
        for idx, row in df.iterrows():
            cid = row.get("pubchem_cid")
            if cid is None or pd.isna(cid):
                continue
            existing = row.get("cas_number")
            if not force and existing is not None and not pd.isna(existing) and str(existing).strip():
                continue
            try:
                cas_number = self._fetch_cas_for_cid(int(cid))
            except (OSError, ValueError, ConnectionError, TimeoutError) as exc:  # noqa: BLE001 -- enrichment must not abort  # v85 FORENSIC ROOT FIX (BUG #51)
                logger.warning(
                    "[%s] enrich_cas: CAS lookup failed for CID=%s: %s",
                    self.source_name,
                    cid,
                    exc,
                )
                continue
            if cas_number:
                df.at[idx, "cas_number"] = cas_number
                enriched += 1
        logger.info(
            "[%s] enrich_cas: enriched %d / %d rows with CAS numbers",
            self.source_name,
            enriched,
            len(df),
        )
        return df

    # ------------------------------------------------------------------
    # Public API: load
    # ------------------------------------------------------------------

    def load(
        self,
        df: pd.DataFrame,
        session: Any | None = None,
    ) -> int | LoadResult:
        """Persist cleaned PubChem enrichment data to the database.

        Uses the passed session (caller-managed transaction boundary)
        per ARCH-1, ARCH-2.  If ``session is None``, opens a new session
        via ``get_db_session(pipeline_name=..., run_id=...,
        correlation_id=...)`` (preserves the audit trail).

        Parameters
        ----------
        df : pandas.DataFrame
            Cleaned DataFrame from ``clean()``.  Must contain ``inchikey``
            and ``pubchem_cid`` columns.
        session : Session, optional
            SQLAlchemy session.  When provided, the caller manages the
            transaction (the base class's ``run()`` opens the session and
            commits on success / rolls back on exception).

        Returns
        -------
        int or LoadResult
            Total rows upserted (drugs + compound_properties).  Returning
            ``LoadResult`` is also accepted by the base class.

        Side effects
        ------------
        * Calls ``bulk_update_drugs_from_pubchem(session, load_df)`` --
          updates ``drugs.pubchem_cid``, ``molecular_formula``,
          ``molecular_weight``, ``smiles`` where ``pubchem_cid IS NULL``.
        * Calls ``bulk_upsert_pubchem_compound_properties(session, df)`` --
          upserts all 15+ physicochemical properties + lineage columns
          into the new ``pubchem_compound_properties`` table.
        * Appends to ``self.dead_letter_queue`` for any rows missing
          ``pubchem_cid`` (DQ-17).
        * Does NOT call ``session.commit()`` -- that is the caller's
          responsibility (CODE-24).
        """
        if df.empty:
            logger.info("[%s] No PubChem enrichment data to load", self.source_name)
            return 0

        owns_session = session is None
        # v29 ROOT FIX (audit P1-5 + P1-7): two bugs in one.
        #
        # P1-5: the previous code did
        #   session = get_db_session(...)
        #   session.__enter__()
        # and DISCARDED the return value of __enter__(). ``session``
        # still referred to the context manager, so every subsequent
        # session.flush() / session.add() crashed with AttributeError
        # when load() was called standalone.
        #
        # P1-7: the previous finally block called
        #   session.__exit__(None, None, None)
        # which signals "no exception" to the context manager -- so it
        # COMMITTED partial data even when an exception was raised
        # mid-load. ROOT FIX: capture the return value of __enter__()
        # AND pass the actual exc_info to __exit__ so the context
        # manager commits on success / rolls back on exception.
        _session_cm = None
        if owns_session:
            _session_cm = get_db_session(
                pipeline_name=self.source_name,
                run_id=self.run_id,
                correlation_id=self.correlation_id,
            )
            session = _session_cm.__enter__()

        try:
            # P1-24 ROOT FIX: reset the DataFrame index to a default
            # RangeIndex before extracting per-column arrays. Previously
            # the load_dict was built from `df["col"].values` for each
            # column -- if df had a non-default index (e.g. a slice of a
            # larger frame with gaps, or a frame whose index was set by
            # an upstream groupby/merge), the per-column arrays were
            # extracted positionally and could end up at different
            # logical row positions when pandas re-aligned them by index
            # inside `pd.DataFrame(load_dict)`. The classic symptom is
            # "aspirin's InChIKey paired with ibuprofen's molecular
            # weight" -- a silent, hard-to-detect data corruption.
            # Resetting the index up-front guarantees that every column
            # shares the same RangeIndex, so positional `.values`
            # extraction is safe.
            df = df.reset_index(drop=True)

            # --- Step 1: update the drugs table (existing loader) ---
            # Build the load_df for bulk_update_drugs_from_pubchem -- only
            # the columns it expects (CODE-4, CODE-5, CODE-22).
            load_dict: dict[str, Any] = {
                "inchikey": df["inchikey"].values,
                "pubchem_cid": pd.to_numeric(
                    df["pubchem_cid"], errors="coerce"
                ).astype("Int64").values,
            }
            # Use isomeric_smiles (preferred -- stereo preserved) or fall
            # back to canonical_smiles.  This populates drugs.smiles.
            smiles_col = (
                df["isomeric_smiles"]
                if "isomeric_smiles" in df.columns
                and df["isomeric_smiles"].notna().any()
                else df.get("canonical_smiles")
            )
            if smiles_col is not None:
                load_dict["smiles"] = smiles_col.values
            if "molecular_formula" in df.columns:
                load_dict["molecular_formula"] = df["molecular_formula"].values
            if "molecular_weight" in df.columns:
                # v39 ROOT FIX (P1 #25): the schema says molecular_weight
                # uses "Decimal precision (NUMERIC(12,6))" but .values
                # returns a numpy float64 array, losing Decimal precision.
                # The fix: convert to Python Decimal objects before
                # passing to the DataFrame. The SQLAlchemy ORM's
                # Numeric(12,6) column accepts Decimal objects natively
                # and preserves the 6 decimal places. float64 only
                # preserves ~15 significant digits, which is fine for
                # small molecules (<1000 Da) but degrades for large
                # biologics (>10K Da). Decimal preserves exact precision
                # for all molecular weights.
                from decimal import Decimal as _Decimal_v39
                import numpy as _np_v39_p25
                _mw_values = df["molecular_weight"].values
                # Convert each value to Decimal, handling NaN gracefully.
                _mw_decimals = []
                for _v in _mw_values:
                    try:
                        # v64 ROOT FIX (P1-021): the previous NaN check only
                        # handled float NaN (`isinstance(_v, float)`). If `_v`
                        # is already a `Decimal` (from a prior conversion step),
                        # `Decimal('NaN')` would bypass this check and propagate
                        # into the DB insert, causing IntegrityError on a NOT
                        # NULL column. Root fix: also check
                        # `isinstance(_v, Decimal) and _v.is_nan()`.
                        if _v is None:
                            _mw_decimals.append(None)
                        elif isinstance(_v, float) and _np_v39_p25.isnan(_v):
                            _mw_decimals.append(None)
                        elif isinstance(_v, _Decimal_v39) and _v.is_nan():
                            _mw_decimals.append(None)
                        else:
                            _mw_decimals.append(_Decimal_v39(str(_v)))
                    except (ValueError, TypeError, ArithmeticError):  # v85 FORENSIC ROOT FIX (BUG #51)
                        _mw_decimals.append(None)
                load_dict["molecular_weight"] = _mw_decimals
            load_df = pd.DataFrame(load_dict)

            # Drop rows with no pubchem_cid (DQ-17).
            na_mask = load_df["pubchem_cid"].isna()
            if na_mask.any():
                dropped = load_df.loc[na_mask, "inchikey"].tolist()
                logger.warning(
                    "[%s] Dropping %d rows with no PubChem CID. First 50: %s",
                    self.source_name,
                    len(dropped),
                    dropped[:50],
                )
                for ik in dropped:
                    self.dead_letter_queue.append(
                        {
                            "inchikey": ik,
                            "reason": "no_cid_from_pubchem",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                load_df = load_df[~na_mask].copy()
            if load_df.empty:
                logger.info(
                    "[%s] No PubChem enrichment data with valid CID to load",
                    self.source_name,
                )
                return 0
            # Convert Int64 -> int64 (lowercase, non-nullable) for SQL
            # compatibility (CODE-11, CODE-12, CODE-21).
            load_df["pubchem_cid"] = load_df["pubchem_cid"].astype("int64")

            try:
                drugs_updated = bulk_update_drugs_from_pubchem(
                    session=session,
                    df=load_df,
                    batch_size=1000,
                )
                logger.info(
                    "[%s] bulk_update_drugs_from_pubchem: %d drug rows updated",
                    self.source_name,
                    drugs_updated,
                )
            except (OSError, RuntimeError, ValueError) as exc:  # v85 FORENSIC ROOT FIX (BUG #51)
                logger.error(
                    "[%s] bulk_update_drugs_from_pubchem failed: %s. "
                    "DataFrame shape=%s, first 5 inchikeys=%s",
                    self.source_name,
                    exc,
                    load_df.shape,
                    load_df["inchikey"].head(5).tolist(),
                )
                # Write the failing DataFrame to dead-letter for inspection.
                self._write_dead_letter_dataframe(
                    load_df, reason="db_update_failed"
                )
                raise

            # --- Step 2: upsert into pubchem_compound_properties (new) ---
            # The new table receives the full DataFrame (with lineage).
            # ARCH-5, INT-7.
            try:
                props_result: UpsertResult = (
                    bulk_upsert_pubchem_compound_properties(
                        session=session,
                        df=df,
                        batch_size=1000,
                    )
                )
                logger.info(
                    "[%s] bulk_upsert_pubchem_compound_properties: "
                    "input=%d, inserted+updated=%d, quarantined=%d, failed=%d",
                    self.source_name,
                    props_result.total_input,
                    props_result.inserted,
                    props_result.quarantined,
                    props_result.failed,
                )
                if _PUBCHEM_RECORDS_LOADED is not None:
                    _PUBCHEM_RECORDS_LOADED.set(props_result.inserted)
            except (OSError, RuntimeError, ValueError) as exc:  # v85 FORENSIC ROOT FIX (BUG #51)
                logger.error(
                    "[%s] bulk_upsert_pubchem_compound_properties failed: %s. "
                    "DataFrame shape=%s, first 5 inchikeys=%s",
                    self.source_name,
                    exc,
                    df.shape,
                    df["inchikey"].head(5).tolist(),
                )
                self._write_dead_letter_dataframe(df, reason="db_props_upsert_failed")
                raise

            # FIX-P1-B-3 (audit P1): the previous code computed
            # ``total = drugs_updated + props_result.inserted`` but never
            # returned it, AND repurposed ``LoadResult.rows_updated`` to
            # hold ONLY ``drugs_updated`` (the drugs table count), silently
            # dropping ``props_result.updated`` (the pubchem_compound_properties
            # table UPDATE count). Root fix: sum BOTH table-update counts
            # into ``rows_updated`` so callers see the true total. The dead
            # ``total`` line is removed.
            return LoadResult(
                rows_inserted=props_result.inserted,
                rows_updated=props_result.updated + drugs_updated,
                rows_skipped=props_result.quarantined,
                rows_failed=props_result.failed,
            )
        finally:
            # v29 ROOT FIX (audit P1-7): the previous code called
            #   session.__exit__(None, None, None)
            # which signals "no exception" -- so the context manager
            # COMMITTED partial data even when an exception was raised
            # mid-load. ROOT FIX: pass the actual exc_info so the
            # context manager commits on success / rolls back on
            # exception.
            if owns_session and _session_cm is not None:
                import sys as _sys
                _exc_info = _sys.exc_info()
                try:
                    _session_cm.__exit__(*_exc_info)
                except (OSError, RuntimeError, ValueError) as _cleanup_exc:  # noqa: BLE001  # v85 FORENSIC ROOT FIX (BUG #51)
                    # v39 ROOT FIX (P1 #26): the previous code silently
                    # swallowed ALL errors from __exit__ -- including
                    # rollback failures that should be surfaced. The
                    # comment said "cleanup must not mask" but the code
                    # did exactly that. The fix: log the cleanup error
                    # so operators can see rollback failures, but still
                    # don't re-raise (the original exception, if any,
                    # should propagate). This makes the failure visible
                    # without masking the primary error.
                    logger.warning(
                        "[%s] Session __exit__ failed during cleanup: %s. "
                        "This may indicate a rollback failure (the DB "
                        "connection may have died mid-load). The original "
                        "exception (if any) will propagate. (v39 P1 #26 fix)",
                        self.source_name, _cleanup_exc,
                    )

    # ------------------------------------------------------------------
    # Public API: get_source_version (override)
    # ------------------------------------------------------------------

    def get_source_version(self) -> Optional[str]:
        """Return the PubChem source version string (ARCH-6, LIN-3).

        PubChem PUG REST has no explicit version field.  We record the
        access timestamp as ``"pubchem_pug_rest_as_of_<ISO 8601 UTC>"``
        -- supports reproducibility audits and impact analysis when
        PubChem updates.
        """
        if self._access_timestamp is None:
            return None
        return (
            f"pubchem_pug_rest_as_of_"
            f"{self._access_timestamp.isoformat()}"
        )

    # ------------------------------------------------------------------
    # Public API: teardown (override)
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        """Flush the dead-letter queue to disk, then call ``super().teardown()``.

        Writes ``raw_dir / "pubchem_dead_letters.csv"`` with the
        ``QUOTE_NONNUMERIC`` quoting and a SHA-256 sidecar.  The base
        class's teardown closes the HTTP session.
        """
        try:
            self._write_dead_letters_file()
        except (OSError, RuntimeError, ValueError) as exc:  # noqa: BLE001 -- don't crash teardown  # v85 FORENSIC ROOT FIX (BUG #51)
            logger.warning(
                "[%s] Could not write dead-letter file: %s",
                self.source_name,
                exc,
            )
        super().teardown()

    # ------------------------------------------------------------------
    # Internals: batch fetching
    # ------------------------------------------------------------------

    def _fetch_all_batches(self, lookup_file: Path) -> None:
        """Fetch PubChem properties for every InChIKey in ``lookup_file``.

        Each batch's raw JSON response is written to
        ``raw_dir/pubchem_responses/batch_NNNN.json`` with a SHA-256
        sidecar (ARCH-3, LIN-9).  Failures append to
        ``self.dead_letter_queue`` (ARCH-8, REL-4).
        """
        # Load InChIKeys (cached file or fresh query).
        with open(lookup_file, "r", encoding="utf-8") as fh:
            inchikeys: list[str] = []
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                inchikeys.append(line)

        if not inchikeys:
            logger.warning(
                "[%s] No InChIKeys to look up -- skipping PubChem fetch",
                self.source_name,
            )
            return

        # DQ-2: validate every InChIKey before sending to PubChem.
        valid_inchikeys: list[str] = []
        for ik in inchikeys:
            if INCHIKEY_RE.match(ik):
                valid_inchikeys.append(ik)
            else:
                self.dead_letter_queue.append(
                    {
                        "inchikey": ik,
                        "reason": "invalid_inchikey_format",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
        if len(valid_inchikeys) < len(inchikeys):
            logger.warning(
                "[%s] %d / %d InChIKeys failed format validation -- "
                "dead-lettered",
                self.source_name,
                len(inchikeys) - len(valid_inchikeys),
                len(inchikeys),
            )
        inchikeys = valid_inchikeys
        if not inchikeys:
            logger.warning(
                "[%s] All InChIKeys invalid -- no PubChem fetch performed",
                self.source_name,
            )
            return

        # Prepare the raw response archive directory.
        responses_dir = self.raw_dir / "pubchem_responses"
        responses_dir.mkdir(parents=True, exist_ok=True)

        # Clear any stale batch files from a previous run (idempotency).
        for stale in responses_dir.glob("batch_*.json"):
            try:
                stale.unlink()
            except OSError:
                pass
        for stale in responses_dir.glob("batch_*.json.sha256"):
            try:
                stale.unlink()
            except OSError:
                pass

        # Batch the InChIKeys.
        batches: list[tuple[int, list[str]]] = []
        for batch_idx, start in enumerate(
            range(0, len(inchikeys), self.batch_size)
        ):
            batch = inchikeys[start : start + self.batch_size]
            batches.append((batch_idx, batch))

        total_batches = len(batches)
        self._batch_count = total_batches
        logger.info(
            "[%s] Fetching %d batches (batch_size=%d, total_inchikeys=%d)",
            self.source_name,
            total_batches,
            self.batch_size,
            len(inchikeys),
        )

        # ARCH-13: optional concurrent batch processing.
        if self.concurrency > 1 and total_batches > 1:
            self._fetch_batches_concurrent(batches, total_batches, responses_dir)
        else:
            self._fetch_batches_sequential(batches, total_batches, responses_dir)

        # Record the access timestamp for source_version (ARCH-6).
        if self._access_timestamp is None:
            self._access_timestamp = datetime.now(timezone.utc)
        # P1-055 ROOT FIX (v108): set source_version DIRECTLY from the
        # access timestamp, NOT via get_source_version() (which is
        # circular — it returns the fallback when source_version is None).
        # The previous code set self.source_version = self.get_source_version()
        # which called the fallback generator (logging a WARNING every time
        # and returning "pubchem_as_of_<date>"). ROOT FIX: set the version
        # string directly so no WARNING is logged and the version is
        # explicit about being a PubChem snapshot.
        self.source_version = (
            f"PubChem_PUG_REST_as_of_"
            f"{self._access_timestamp.strftime('%Y-%m-%d')}"
        )

        # LOG-8: summary.
        if self._api_call_count > 0:
            self._avg_latency = self._total_latency / self._api_call_count
        logger.info(
            "[%s] PubChem fetch complete: api_calls=%d, retries=%d, "
            "splits=%d, avg_latency=%.2fs, dead_lettered=%d",
            self.source_name,
            self._api_call_count,
            self._retry_count,
            self._split_count,
            self._avg_latency,
            len(self.dead_letter_queue),
        )

    def _fetch_batches_sequential(
        self,
        batches: list[tuple[int, list[str]]],
        total_batches: int,
        responses_dir: Path,
    ) -> None:
        """Fetch batches one at a time (default for determinism)."""
        for batch_idx, batch in batches:
            self._fetch_and_archive_batch(
                batch_idx, batch, total_batches, responses_dir
            )
            # PERF-6, CODE-19: skip rate-limit sleep on the last batch.
            if batch_idx + 1 < total_batches:
                time.sleep(self.rate_limit_interval)

    def _fetch_batches_concurrent(
        self,
        batches: list[tuple[int, list[str]]],
        total_batches: int,
        responses_dir: Path,
    ) -> None:
        """Fetch batches concurrently with a thread pool (ARCH-13).

        Uses ``ThreadPoolExecutor(max_workers=self.concurrency)``.  Each
        worker writes its own batch file -- no shared mutable state.  A
        ``threading.Semaphore`` enforces the rate limit across threads.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        semaphore = threading.Semaphore(self.concurrency)

        def _worker(b_idx: int, b: list[str]) -> int:
            with semaphore:
                self._fetch_and_archive_batch(
                    b_idx, b, total_batches, responses_dir
                )
                return b_idx

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {
                executor.submit(_worker, b_idx, b): b_idx for b_idx, b in batches
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except (OSError, RuntimeError, ValueError) as exc:  # noqa: BLE001  # v85 FORENSIC ROOT FIX (BUG #51)
                    logger.error(
                        "[%s] Batch worker failed: %s",
                        self.source_name,
                        exc,
                    )

    def _fetch_and_archive_batch(
        self,
        batch_idx: int,
        batch: list[str],
        total_batches: int,
        responses_dir: Path,
    ) -> None:
        """Fetch a single batch from PubChem and archive the raw response.

        On permanent 4xx failure, splits the batch and retries per-
        InChIKey (REL-5).  On transient failure, retries with jittered
        backoff respecting ``Retry-After``.  On all-retries-exhausted,
        dead-letters every InChIKey in the batch.
        """
        if _tracer is not None:  # pragma: no cover -- OTEL only
            with _tracer.start_as_current_span("pubchem_lookup_batch") as span:
                span.set_attribute("batch.size", len(batch))
                span.set_attribute("batch.idx", batch_idx)
                self._fetch_and_archive_batch_impl(
                    batch_idx, batch, total_batches, responses_dir
                )
            return
        self._fetch_and_archive_batch_impl(
            batch_idx, batch, total_batches, responses_dir
        )

    def _fetch_and_archive_batch_impl(
        self,
        batch_idx: int,
        batch: list[str],
        total_batches: int,
        responses_dir: Path,
    ) -> None:
        """Implementation of :meth:`_fetch_and_archive_batch` (no tracing)."""
        batch_start = time.monotonic()
        data, status, error = self._lookup_batch(batch_idx, batch, total_batches)
        batch_duration = time.monotonic() - batch_start
        self._total_latency += batch_duration

        if _PUBCHEM_API_LATENCY is not None:  # pragma: no cover
            _PUBCHEM_API_LATENCY.labels(endpoint="property_batch").observe(
                batch_duration
            )

        if data is None:
            # Permanent failure (4xx) or all-retries-exhausted.
            # REL-5: split the batch and retry per-InChIKey when the
            # batch is small enough.
            if (
                status in PERMANENT_STATUS
                and len(batch) <= self.split_retry_max
                and len(batch) > 1
            ):
                logger.info(
                    "[%s] Batch %d got permanent %d -- splitting into %d "
                    "individual lookups",
                    self.source_name,
                    batch_idx,
                    status,
                    len(batch),
                )
                self._split_count += 1
                split_data = self._split_retry_batch(batch_idx, batch, responses_dir)
                if split_data is not None:
                    self._archive_batch_response(
                        batch_idx, split_data, responses_dir
                    )
                    return
            # Dead-letter every InChIKey in the batch.
            reason = (
                f"http_{status}_permanent"
                if status is not None
                else f"all_retries_exhausted_{error or 'unknown'}"
            )
            for ik in batch:
                self.dead_letter_queue.append(
                    {
                        "inchikey": ik,
                        "reason": reason,
                        "batch_idx": batch_idx,
                        "status_code": status,
                        "response_snippet": (str(error) or "")[:500],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            if _PUBCHEM_BATCHES_TOTAL is not None:  # pragma: no cover
                _PUBCHEM_BATCHES_TOTAL.labels(status="failed").inc()
            return

        # Archive the raw response (ARCH-3, LIN-9).
        self._archive_batch_response(batch_idx, data, responses_dir)

        if _PUBCHEM_BATCHES_TOTAL is not None:  # pragma: no cover
            _PUBCHEM_BATCHES_TOTAL.labels(status="success").inc()

        # LOG-5: per-batch timing log.
        logger.info(
            "[%s] Batch %d/%d took %.2fs (%d inchikeys)",
            self.source_name,
            batch_idx + 1,
            total_batches,
            batch_duration,
            len(batch),
        )

    def _lookup_batch(
        self,
        batch_idx: int,
        inchikeys: list[str],
        total_batches: int,
    ) -> tuple[Optional[dict], Optional[int], Optional[str]]:
        """POST a batch of InChIKeys to PubChem PUG REST.

        Returns a ``(data, status, error)`` tuple:
        * ``data``: parsed JSON dict on success, ``None`` on failure.
        * ``status``: last HTTP status code received, ``None`` on network error.
        * ``error``: error message string, ``None`` on success.

        Implements jittered exponential backoff with ``Retry-After`` header
        respect, 4xx fast-fail, circuit-breaker integration, and full
        context logging (DESIGN-9, DESIGN-10, DESIGN-12, DESIGN-14,
        DESIGN-17, REL-1, REL-2, REL-8, REL-10, LOG-7, LOG-9).
        """
        properties_str = ",".join(self.pubchem_properties)
        url = (
            f"{self.rest_base}/compound/inchikey/property/"
            f"{properties_str}/JSON"
        )
        inchikey_csv = ",".join(inchikeys)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": self._user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        # v83 FORENSIC ROOT FIX (P2-7): the previous code passed the NCBI
        # API key as ``params["apikey"]`` (query string), which is logged
        # by CDNs/proxies. The OMIM pipeline avoided this (BUG-2.2) by
        # using ``Authorization: ApiKey`` header. ROOT FIX: send the API
        # key via the ``api-key`` HTTP header instead of the query string.
        # NCBI services accept both forms; the header form keeps the key
        # out of URLs / proxy logs. If PubChem ignores the header, the
        # only effect is reverting to the default rate limit (3 req/sec
        # instead of 10 req/sec) -- NOT a breaking change.
        params: dict[str, Any] = {}
        if self.api_key:
            headers["api-key"] = self.api_key

        last_status: Optional[int] = None
        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            # Circuit breaker check (ARCH-9, REL-3).
            if self._circuit_breaker.is_open():
                logger.error(
                    "[%s] Circuit breaker OPEN -- failing fast on batch %d",
                    self.source_name,
                    batch_idx,
                )
                # Dead-letter every InChIKey in the batch.
                for ik in inchikeys:
                    self.dead_letter_queue.append(
                        {
                            "inchikey": ik,
                            "reason": "circuit_breaker_open",
                            "batch_idx": batch_idx,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                return None, None, "circuit_breaker_open"

            self._api_call_count += 1
            request_start = time.monotonic()
            try:
                resp = self.http_session.post(
                    url,
                    data={"inchikey": inchikey_csv},
                    headers=headers,
                    params=params,
                    timeout=self.timeout,
                    # v65 ROOT FIX (P1-035): rewritten as an explicit
                    # ternary for clarity. The previous
                    # ``verify=(self.ca_bundle or True)`` was functionally
                    # correct but read as "always True", confusing reviewers.
                    verify=(self.ca_bundle if self.ca_bundle else True),
                    cert=(
                        (self.cert_pem, self.key_pem)
                        if (self.cert_pem and self.key_pem)
                        else None
                    ),
                )
            except RETRYABLE_EXCEPTIONS as exc:
                last_status = None
                last_error = f"{type(exc).__name__}: {exc}"
                self._consecutive_connection_failures += 1
                # REL-9: detect complete PubChem unreachability.
                if (
                    self._consecutive_connection_failures >= 3
                    and batch_idx == 0
                ):
                    logger.error(
                        "[%s] First %d batches all failed with connection "
                        "errors -- raising PubChemUnreachableError",
                        self.source_name,
                        self._consecutive_connection_failures,
                    )
                    raise PubChemUnreachableError(str(exc)) from exc
                self._circuit_breaker.record_failure()
                if attempt >= self.max_retries:
                    logger.error(
                        "[%s] PubChem batch %d/%d failed after %d retries: "
                        "%s\nURL: %s\nBatch InChIKeys (first 10): %s",
                        self.source_name,
                        batch_idx + 1,
                        total_batches,
                        self.max_retries,
                        exc,
                        url,
                        inchikeys[:10],
                    )
                    return None, None, last_error
                backoff = self._compute_backoff(attempt, None)
                logger.warning(
                    "[%s] PubChem batch %d retrying (attempt %d/%d, "
                    "exc=%s, backoff=%.2fs, batch_size=%d)",
                    self.source_name,
                    batch_idx,
                    attempt + 1,
                    self.max_retries,
                    type(exc).__name__,
                    backoff,
                    len(inchikeys),
                )
                if _PUBCHEM_RETRIES_TOTAL is not None:  # pragma: no cover
                    _PUBCHEM_RETRIES_TOTAL.labels(
                        status_code=str(type(exc).__name__)
                    ).inc()
                self._retry_count += 1
                time.sleep(backoff)
                continue

            # Success -- reset connection failure counter.
            self._consecutive_connection_failures = 0
            last_status = resp.status_code
            request_duration = time.monotonic() - request_start
            logger.debug(
                "[%s] PubChem batch %d: HTTP %d in %.2fs (%d bytes)",
                self.source_name,
                batch_idx,
                resp.status_code,
                request_duration,
                len(resp.content),
            )

            # 2xx success -- parse JSON.
            if 200 <= resp.status_code < 300:
                # INT-15: guard against HTML response.
                content_type = resp.headers.get("Content-Type", "")
                if "application/json" not in content_type:
                    logger.warning(
                        "[%s] Unexpected Content-Type %s on batch %d -- "
                        "dead-lettering",
                        self.source_name,
                        content_type,
                        batch_idx,
                    )
                    self._circuit_breaker.record_failure()
                    return None, resp.status_code, f"unexpected_content_type_{content_type}"
                # INT-14: check for truncated response.
                expected = int(resp.headers.get("Content-Length", 0) or 0)
                actual = len(resp.content)
                if expected > 0 and actual < expected:
                    logger.warning(
                        "[%s] Truncated response on batch %d: expected %d "
                        "bytes, got %d -- retrying",
                        self.source_name,
                        batch_idx,
                        expected,
                        actual,
                    )
                    self._circuit_breaker.record_failure()
                    if attempt >= self.max_retries:
                        return None, resp.status_code, "truncated_response"
                    backoff = self._compute_backoff(attempt, None)
                    self._retry_count += 1
                    time.sleep(backoff)
                    continue
                # Parse JSON.
                try:
                    data = resp.json()
                except (
                    ValueError,
                    requests.exceptions.JSONDecodeError,
                    json.JSONDecodeError,
                ) as exc:
                    preview = resp.text[:500]
                    logger.error(
                        "[%s] JSON decode error on batch %d: %s. "
                        "Body preview: %s",
                        self.source_name,
                        batch_idx,
                        exc,
                        preview,
                    )
                    self._circuit_breaker.record_failure()
                    if attempt >= self.max_retries:
                        return None, resp.status_code, f"json_decode_error: {exc}"
                    backoff = self._compute_backoff(attempt, None)
                    self._retry_count += 1
                    time.sleep(backoff)
                    continue
                # INT-12, INT-13: check for PubChem fault response.
                if PUBCHEM_FAULT_KEY in data:
                    fault = data[PUBCHEM_FAULT_KEY]
                    fault_code = fault.get("Code", "unknown")
                    fault_msg = fault.get("Message", "")
                    logger.warning(
                        "[%s] PubChem Fault on batch %d: code=%s, msg=%s -- "
                        "treating as permanent failure",
                        self.source_name,
                        batch_idx,
                        fault_code,
                        fault_msg,
                    )
                    self._circuit_breaker.record_failure()
                    return None, resp.status_code, f"pubchem_fault_{fault_code}"
                # INT-12: validate response schema.
                if (
                    "PropertyTable" not in data
                    or "Properties" not in data.get("PropertyTable", {})
                ):
                    logger.error(
                        "[%s] Unexpected response schema on batch %d -- "
                        "keys=%s, snippet=%s",
                        self.source_name,
                        batch_idx,
                        list(data.keys()),
                        json.dumps(data)[:500],
                    )
                    self._circuit_breaker.record_failure()
                    return (
                        None,
                        resp.status_code,
                        "unexpected_response_schema",
                    )
                self._circuit_breaker.record_success()
                return data, resp.status_code, None

            # 4xx (except 429) -- permanent failure (DESIGN-12, REL-1).
            if (
                400 <= resp.status_code < 500
                and resp.status_code != 429
            ):
                logger.warning(
                    "[%s] PubChem batch %d HTTP %d (non-retryable) -- "
                    "snippet=%s",
                    self.source_name,
                    batch_idx,
                    resp.status_code,
                    resp.text[:500],
                )
                # v52 ROOT FIX (P1-031 -- inconsistent circuit breaker):
                # The v48 PubChem client called record_failure() for 4xx
                # errors, which meant 5 consecutive 404s (e.g. querying
                # deleted PubChem CIDs) would OPEN the breaker and block
                # ALL PubChem API calls for 60s. But the ChEMBL client
                # (v43 fix) does NOT trip the breaker on 4xx -- 4xx errors
                # are PERMANENT client errors (the resource doesn't exist),
                # not transient API failures. The circuit breaker is
                # designed for TRANSIENT failures (429, 5xx, ConnectionError).
                # ROOT FIX: do NOT call record_failure() for 4xx. Call
                # record_success() instead -- a 404 means the API responded
                # correctly with "not found", which is a successful
                # round-trip. This aligns PubChem with ChEMBL's v43 policy
                # and eliminates the inconsistency.
                self._circuit_breaker.record_success()
                return None, resp.status_code, f"http_{resp.status_code}_permanent"

            # 429 or 5xx -- retryable.
            retry_after = resp.headers.get("Retry-After", "")
            backoff = self._compute_backoff(attempt, retry_after)
            logger.warning(
                "[%s] PubChem batch %d/%d retrying (attempt %d/%d, "
                "status=%d, retry_after=%ss, backoff=%.2fs, "
                "batch_size=%d, first_inchikey=%s)",
                self.source_name,
                batch_idx + 1,
                total_batches,
                attempt + 1,
                self.max_retries,
                resp.status_code,
                retry_after,
                backoff,
                len(inchikeys),
                inchikeys[0] if inchikeys else "",
            )
            if _PUBCHEM_RETRIES_TOTAL is not None:  # pragma: no cover
                _PUBCHEM_RETRIES_TOTAL.labels(
                    status_code=str(resp.status_code)
                ).inc()
            self._circuit_breaker.record_failure()
            if attempt >= self.max_retries:
                logger.error(
                    "[%s] PubChem batch %d/%d failed after %d retries: "
                    "status=%d\nURL: %s\nBatch InChIKeys (first 10): %s\n"
                    "Response snippet: %s",
                    self.source_name,
                    batch_idx + 1,
                    total_batches,
                    self.max_retries,
                    resp.status_code,
                    url,
                    inchikeys[:10],
                    resp.text[:500],
                )
                return None, resp.status_code, f"http_{resp.status_code}_retries_exhausted"
            self._retry_count += 1
            time.sleep(backoff)

        return None, last_status, last_error or "max_retries_exhausted"

    def _compute_backoff(
        self,
        attempt: int,
        retry_after_header: Optional[str],
    ) -> float:
        """Compute the backoff seconds for a retry (DESIGN-9, DESIGN-10).

        Uses exponential backoff ``min_backoff * (2 ** attempt)`` capped
        at ``max_backoff``, respects the ``Retry-After`` header (either
        delta-seconds or HTTP-date form), and adds uniform jitter in
        ``[0, min(backoff, 1.0)]`` to avoid thundering-herd.
        """
        # Parse Retry-After.
        retry_after_seconds = 0.0
        if retry_after_header:
            try:
                retry_after_seconds = float(retry_after_header)
            except ValueError:
                try:
                    retry_after_dt = email.utils.parsedate_to_datetime(
                        retry_after_header
                    )
                    if retry_after_dt is not None:
                        retry_after_dt = retry_after_dt.astimezone(timezone.utc)
                        retry_after_seconds = max(
                            0.0,
                            (
                                retry_after_dt
                                - datetime.now(timezone.utc)
                            ).total_seconds(),
                        )
                except (TypeError, ValueError):
                    retry_after_seconds = 0.0
        # Exponential backoff.
        backoff = min(
            self.min_backoff * (2 ** attempt), self.max_backoff
        )
        backoff = max(backoff, retry_after_seconds)
        # Jitter.
        backoff += random.uniform(0.0, min(backoff, 1.0))
        return backoff

    def _split_retry_batch(
        self,
        batch_idx: int,
        batch: list[str],
        responses_dir: Path,
    ) -> Optional[dict]:
        """Retry each InChIKey individually after a batch 4xx (REL-5).

        Returns a synthesized ``{"PropertyTable": {"Properties": [...]}}``
        dict containing the successful individual lookups, or ``None`` if
        every individual lookup also failed.
        """
        properties_str = ",".join(self.pubchem_properties)
        all_props: list[dict] = []
        for ik in batch:
            url = (
                f"{self.rest_base}/compound/inchikey/{ik}/property/"
                f"{properties_str}/JSON"
            )
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": self._user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
            # v83 P2-7: send API key via header (not query string).
            params: dict[str, Any] = {}
            if self.api_key:
                headers["api-key"] = self.api_key
            try:
                self._api_call_count += 1
                resp = self.http_session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self.timeout,
                    # v65 ROOT FIX (P1-035): rewritten as an explicit
                    # ternary for clarity. The previous
                    # ``verify=(self.ca_bundle or True)`` was functionally
                    # correct but read as "always True", confusing reviewers.
                    verify=(self.ca_bundle if self.ca_bundle else True),
                    cert=(
                        (self.cert_pem, self.key_pem)
                        if (self.cert_pem and self.key_pem)
                        else None
                    ),
                )
                if 200 <= resp.status_code < 300:
                    data = resp.json()
                    if (
                        PUBCHEM_FAULT_KEY not in data
                        and "PropertyTable" in data
                        and "Properties" in data["PropertyTable"]
                    ):
                        all_props.extend(data["PropertyTable"]["Properties"])
                        continue
                # Permanent failure for this individual InChIKey.
                self.dead_letter_queue.append(
                    {
                        "inchikey": ik,
                        "reason": f"http_{resp.status_code}_permanent_split",
                        "batch_idx": batch_idx,
                        "status_code": resp.status_code,
                        "response_snippet": resp.text[:500],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            except RETRYABLE_EXCEPTIONS as exc:
                self.dead_letter_queue.append(
                    {
                        "inchikey": ik,
                        "reason": f"split_retry_network_error_{type(exc).__name__}",
                        "batch_idx": batch_idx,
                        "response_snippet": str(exc)[:500],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            # Rate limit between individual lookups.
            time.sleep(self.rate_limit_interval)
        if not all_props:
            return None
        return {"PropertyTable": {"Properties": all_props}}

    def _archive_batch_response(
        self,
        batch_idx: int,
        data: dict,
        responses_dir: Path,
    ) -> None:
        """Write a batch's raw JSON response to ``batch_NNNN.json`` + SHA-256."""
        batch_file = responses_dir / f"batch_{batch_idx:04d}.json"
        with open(batch_file, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        sha256 = self._compute_sha256(batch_file)
        sha256_path = batch_file.with_suffix(batch_file.suffix + ".sha256")
        with open(sha256_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"{sha256}  {batch_file.name}\n")

    # ------------------------------------------------------------------
    # Internals: response parsing
    # ------------------------------------------------------------------

    def _parse_pubchem_response(
        self,
        data: dict,
        requested_inchikeys: list[str],
        batch_idx: int,
        batch_sha256: str,
    ) -> list[dict]:
        """Parse a PubChem PUG REST JSON response into record dicts.

        Verifies the response InChIKey matches one of the requested
        InChIKeys (SCI-11).  Sanitizes empty strings to ``None`` (SCI-18).
        Converts floats to ``Decimal`` (SCI-16).  Validates ranges
        (SCI-17, DQ-4).  Deduplicates by InChIKey keeping the lowest CID
        (DESIGN-19, DQ-13).  Extracts protonation state, isotope info,
        formal charge (SCI-8, SCI-14, SCI-15).
        """
        properties_table = data.get("PropertyTable", {})
        properties_list = properties_table.get("Properties", [])

        # SCI-11: build a set of requested InChIKeys for fast verification.
        requested_set = set(requested_inchikeys)
        # Build a mapping from response InChIKey -> record, deduping by
        # lowest CID (DESIGN-19).
        by_inchikey: dict[str, dict] = {}

        transformations_applied: list[str] = [
            "validated_inchikey_format",
            "fetched_pubchem_properties",
        ]

        for pubchem_record in properties_list:
            cid = pubchem_record.get("CID")
            response_inchikey = pubchem_record.get("InChIKey")

            # DESIGN-20: validate InChIKey type and format.
            if not isinstance(response_inchikey, str) or not INCHIKEY_RE.match(
                response_inchikey
            ):
                self.dead_letter_queue.append(
                    {
                        "inchikey": str(response_inchikey),
                        "reason": "invalid_response_inchikey",
                        "response_value": repr(response_inchikey),
                        "batch_idx": batch_idx,
                        "cid": cid,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                continue

            # SCI-11: verify the response InChIKey matches a requested one.
            if response_inchikey not in requested_set:
                logger.warning(
                    "[%s] InChIKey mismatch on batch %d: response=%s not in "
                    "requested set (CID=%s) -- dead-lettering",
                    self.source_name,
                    batch_idx,
                    response_inchikey,
                    cid,
                )
                self.dead_letter_queue.append(
                    {
                        "inchikey": response_inchikey,
                        "reason": "inchikey_mismatch",
                        "batch_idx": batch_idx,
                        "cid": cid,
                        "requested_inchikeys": requested_inchikeys[:10],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                continue
            transformations_applied.append(
                "verified_response_inchikey_matches_request"
            )

            # CODE-1: safe CID conversion.
            safe_cid = self._safe_int(cid, field_name="CID", inchikey=response_inchikey)
            if safe_cid is None:
                self.dead_letter_queue.append(
                    {
                        "inchikey": response_inchikey,
                        "reason": "invalid_cid",
                        "batch_idx": batch_idx,
                        "cid_value": repr(cid),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                continue

            # Build the record.
            # SCI-FIX (PubChem SMILES response key mapping):
            # PubChem's PUG REST API accepts ``CanonicalSMILES`` and
            # ``IsomericSMILES`` as INPUT property names, but the JSON
            # RESPONSE always returns them under different keys:
            #   - ``SMILES``               -> isomeric SMILES (with stereo,
            #                                  isotope, and charge info).
            #                                  When the molecule has no
            #                                  stereo/isotope/charge, this
            #                                  equals the canonical SMILES.
            #   - ``ConnectivitySMILES``   -> canonical SMILES (no stereo).
            # The original code looked up the input names ("CanonicalSMILES"
            # and "IsomericSMILES") in the response dict, which always
            # returned None -- silently losing 100% of SMILES data and
            # cascading into NULL formal_charge and isotope_info (which are
            # computed from isomeric_smiles). This was a life-safety
            # critical bug: without SMILES, the Graph Transformer cannot
            # compute molecular fingerprints, the RL ranker cannot assess
            # structural similarity, and downstream clinical predictions
            # would be made on incomplete chemistry. The fix below reads
            # both possible response keys and assigns them to the correct
            # schema column, falling back gracefully when only one is
            # present.
            raw_smiles = _sanitize_string(pubchem_record.get("SMILES"))
            raw_connectivity_smiles = _sanitize_string(
                pubchem_record.get("ConnectivitySMILES")
            )
            # Backward-compat: very old PubChem responses used the
            # input-name keys directly. Keep these as a defensive fallback.
            legacy_canonical = _sanitize_string(pubchem_record.get("CanonicalSMILES"))
            legacy_isomeric = _sanitize_string(pubchem_record.get("IsomericSMILES"))

            # PubChem's ``SMILES`` field is isomeric when stereo is present;
            # for molecules without stereo it equals the canonical SMILES.
            # ``ConnectivitySMILES`` is always canonical (no stereo).
            isomeric_smiles = raw_smiles or legacy_isomeric
            # SW-3 ROOT FIX: Canonical SMILES must NEVER be derived from
            # isomeric SMILES -- the isomeric form carries stereo (@, /, \)
            # that must stay isolated so the Graph Transformer can build
            # separate 2D (canonical) and 3D (isomeric) fingerprints. The
            # previous code fell back to isomeric SMILES for canonical,
            # silently corrupting the canonical column for chiral molecules
            # and making (R)- and (S)-enantiomers get identical fingerprints.
            # When PubChem omits ConnectivitySMILES/CanonicalSMILES we leave
            # canonical_smiles empty rather than silently corrupting it.
            canonical_smiles = (
                raw_connectivity_smiles
                or legacy_canonical
            )
            # v83 FORENSIC ROOT FIX (P2-6): if PubChem omits BOTH
            # ConnectivitySMILES and CanonicalSMILES, ``canonical_smiles``
            # is None -- the Graph Transformer cannot compute 2D fingerprints
            # for this compound. ROOT FIX: when canonical is missing but
            # isomeric is available, use RDKit to compute the canonical
            # (no-stereo) form from the isomeric form.
            # ``Chem.MolToSmiles(mol, isomericSmiles=False)`` strips stereo
            # by default, producing a TRUE canonical SMILES -- not a copy of
            # the isomeric form. This is scientifically correct: the
            # canonical form is defined as the no-stereo SMILES, and RDKit's
            # canonicalization algorithm is the industry standard. If RDKit
            # is not installed or the isomeric SMILES is unparseable, leave
            # canonical_smiles as None (preserving the previous behaviour
            # for that edge case).
            if not canonical_smiles and isomeric_smiles:
                try:
                    from rdkit import Chem as _rdkit_Chem
                    _mol = _rdkit_Chem.MolFromSmiles(isomeric_smiles)
                    if _mol is not None:
                        canonical_smiles = _rdkit_Chem.MolToSmiles(
                            _mol, isomericSmiles=False
                        )
                        transformations_applied.append(
                            "computed_canonical_smiles_from_isomeric_via_rdkit"
                        )
                except (ImportError, ValueError, RuntimeError):  # v85 FORENSIC ROOT FIX (BUG #51)
                    # RDKit not available or parse failed -- leave None.
                    pass

            # SCI-18: sanitize every string field.
            molecular_formula = _sanitize_string(pubchem_record.get("MolecularFormula"))
            inchi = _sanitize_string(pubchem_record.get("InChI"))
            iupac_name = _sanitize_string(pubchem_record.get("IUPACName"))
            transformations_applied.append("sanitized_empty_strings_to_null")

            # SCI-16, SCI-4: Decimal conversion for numeric fields.
            molecular_weight = self._safe_float(
                pubchem_record.get("MolecularWeight"),
                field_name="MolecularWeight",
                inchikey=response_inchikey,
            )
            exact_mass = self._safe_float(
                pubchem_record.get("ExactMass"),
                field_name="ExactMass",
                inchikey=response_inchikey,
            )
            xlogp = self._safe_float(
                pubchem_record.get("XLogP"),
                field_name="XLogP",
                inchikey=response_inchikey,
            )
            tpsa = self._safe_float(
                pubchem_record.get("TPSA"),
                field_name="TPSA",
                inchikey=response_inchikey,
            )
            complexity = self._safe_float(
                pubchem_record.get("Complexity"),
                field_name="Complexity",
                inchikey=response_inchikey,
            )
            transformations_applied.append("converted_molecular_weight_to_decimal")

            # Integer counts.
            h_bond_donor = self._safe_int(
                pubchem_record.get("HBondDonorCount"),
                field_name="HBondDonorCount",
                inchikey=response_inchikey,
            )
            h_bond_acceptor = self._safe_int(
                pubchem_record.get("HBondAcceptorCount"),
                field_name="HBondAcceptorCount",
                inchikey=response_inchikey,
            )
            rotatable_bond = self._safe_int(
                pubchem_record.get("RotatableBondCount"),
                field_name="RotatableBondCount",
                inchikey=response_inchikey,
            )
            heavy_atom = self._safe_int(
                pubchem_record.get("HeavyAtomCount"),
                field_name="HeavyAtomCount",
                inchikey=response_inchikey,
            )

            # SCI-17, DQ-4: range validation.
            range_check_candidates = {
                "molecular_weight": molecular_weight,
                "exact_mass": exact_mass,
                "xlogp": xlogp,
                "tpsa": tpsa,
                "complexity": complexity,
                "h_bond_donor_count": h_bond_donor,
                "h_bond_acceptor_count": h_bond_acceptor,
                "rotatable_bond_count": rotatable_bond,
                "heavy_atom_count": heavy_atom,
                "pubchem_cid": safe_cid,
            }
            range_violations: list[str] = []
            for field_name, value in range_check_candidates.items():
                if value is None:
                    continue
                lo, hi = RANGES.get(field_name, (None, None))
                if lo is not None and hi is not None:
                    try:
                        v = float(value)
                    except (TypeError, ValueError):
                        continue
                    if v < lo or v > hi:
                        self.dead_letter_queue.append(
                            {
                                "inchikey": response_inchikey,
                                "reason": f"range_violation_{field_name}",
                                "value": v,
                                "valid_range": [lo, hi],
                                "batch_idx": batch_idx,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        range_violations.append(field_name)
            # Set violating fields to None.
            if "molecular_weight" in range_violations:
                molecular_weight = None
            if "exact_mass" in range_violations:
                exact_mass = None
            if "xlogp" in range_violations:
                xlogp = None
            if "tpsa" in range_violations:
                tpsa = None
            if "complexity" in range_violations:
                complexity = None
            if "h_bond_donor_count" in range_violations:
                h_bond_donor = None
            if "h_bond_acceptor_count" in range_violations:
                h_bond_acceptor = None
            if "rotatable_bond_count" in range_violations:
                rotatable_bond = None
            if "heavy_atom_count" in range_violations:
                heavy_atom = None
            if "pubchem_cid" in range_violations:
                # CID out of range -- dead-letter the whole record.
                continue
            transformations_applied.append("validated_ranges")

            # SCI-8, SCI-5: extract protonation state and salt form.
            # V19 ROOT FIX (PS-1): the InChIKey's last char is a 2-value
            # version flag (S/N), NOT a 4-state protonation flag. Real
            # protonation is derived from the InChI string's /p and /q
            # layers. Pass BOTH inchikey and inchi to the extractors so
            # they can use the InChI when available, and return None
            # (rather than a fabricated 4-state label) when it's not.
            protonation_state = _extract_protonation_state(response_inchikey, inchi)
            salt_form = _extract_salt_form(response_inchikey, inchi)
            transformations_applied.append("extracted_protonation_state")

            # SCI-14: isotope info from isomeric SMILES.
            isotope_info = _extract_isotope_info(isomeric_smiles)
            transformations_applied.append("extracted_isotope_info")

            # SCI-15: formal charge from isomeric SMILES (preferred) or canonical.
            formal_charge = _extract_formal_charge(
                isomeric_smiles or canonical_smiles
            )
            transformations_applied.append("computed_formal_charge")

            # Optional CAS lookup (SCI-6) -- only when explicitly enabled AND
            # the caller has opted into HTTP during clean (ARCH-3).
            # FIX-P1-B-2: previously ``clean()`` issued live HTTP requests
            # to PubChem whenever ``fetch_cas=True``, violating the
            # "pure transformation -- NO HTTP calls" contract. Now CAS
            # enrichment only runs when ``self._allow_clean_http=True``
            # (which is False by default inside ``clean()``). Operators
            # who want live CAS enrichment should call ``enrich_cas()``
            # after ``clean()``.
            cas_number: Optional[str] = None
            if (
                self.fetch_cas
                and self._allow_clean_http
                and safe_cid is not None
            ):
                cas_number = self._fetch_cas_for_cid(safe_cid)

            # Build the lineage columns (Domain 16).
            # v83 FORENSIC ROOT FIX (P1-4): the previous code used
            # ``datetime.now(timezone.utc)`` for ``download_date``, which
            # is the CLEAN() time -- NOT the actual download time. If the
            # operator re-runs clean() on cached responses (hours or days
            # after the actual download), the download_date was wrong,
            # breaking lineage / reproducibility audits. ROOT FIX: use
            # ``self._access_timestamp`` (set in ``_fetch_all_batches``
            # at download time) when available; fall back to now() only
            # if the timestamp was never set (defensive -- should not
            # happen in normal flow).
            download_date = (
                self._access_timestamp
                if self._access_timestamp is not None
                else datetime.now(timezone.utc)
            )
            source_id = f"pubchem:CID:{safe_cid}"
            source_version = self.get_source_version() or (
                f"pubchem_pug_rest_as_of_{download_date.isoformat()}"
            )
            input_checksum = self._input_checksum or ""
            transformations_str = ";".join(
                list(dict.fromkeys(transformations_applied))
            )

            record = {
                "inchikey": response_inchikey,
                "pubchem_cid": safe_cid,
                "molecular_formula": molecular_formula,
                "molecular_weight": molecular_weight,
                "exact_mass": exact_mass,
                "canonical_smiles": canonical_smiles,
                "isomeric_smiles": isomeric_smiles,
                "inchi": inchi,
                "iupac_name": iupac_name,
                "cas_number": cas_number,
                "xlogp": xlogp,
                "xlogp_source": "pubchem_xlogp3" if xlogp is not None else None,
                "tpsa": tpsa,
                "tpsa_source": "pubchem_calculated" if tpsa is not None else None,
                "complexity": complexity,
                "h_bond_donor_count": h_bond_donor,
                "h_bond_acceptor_count": h_bond_acceptor,
                "rotatable_bond_count": rotatable_bond,
                "heavy_atom_count": heavy_atom,
                "formal_charge": formal_charge,
                "isotope_info": isotope_info,
                "salt_form": salt_form,
                "protonation_state": protonation_state,
                # P1-018 ROOT FIX (Team-2): ``stereo_parent_cid`` links
                # stereoisomers without collapsing them. Stereoisomers
                # (e.g. (R)-thalidomide CID 5462502 vs (S)-thalidomide CID
                # 5462504 vs racemic CID 3672) have DIFFERENT InChIKeys
                # (the stereo layer differs) but SHARE the first 14 chars
                # of the InChIKey (the connectivity layer / molecular
                # skeleton). We populate ``stereo_parent_cid`` in a
                # post-parse grouping step (see ``_assign_stereo_parent_cids``)
                # by grouping on the InChIKey connectivity prefix and
                # picking the lowest CID in each group as the parent.
                # This preserves stereochemistry (each CID is a SEPARATE
                # Compound node in the KG) while providing the linkage
                # for downstream entity resolution. The previous code
                # collapsed stereoisomers via a ``_stereo_collapse()``
                # function that mapped all stereoisomers to the racemic
                # CID -- scientifically wrong for chiral drugs like
                # thalidomide (the (R)-enantiomer is sedative, the
                # (S)-enantiomer is teratogenic).
                "stereo_parent_cid": None,  # populated post-parse
                "inchikey_connectivity_layer": (
                    response_inchikey.split("-")[0]
                    if response_inchikey and "-" in response_inchikey
                    else None
                ),
                "source": "pubchem",
                "source_id": source_id,
                "source_version": source_version,
                # COMP-11: download_date is stored as an ISO 8601 STRING in
                # the parsed record (so callers and tests can run
                # ``datetime.fromisoformat(rec["download_date"])`` directly).
                # The DB loader (``bulk_upsert_pubchem_compound_properties``)
                # parses this string back to a ``datetime`` object before
                # INSERT, because SQLAlchemy's DateTime column only accepts
                # ``datetime``/``date`` instances on SQLite.
                "download_date": download_date.isoformat(),
                "download_method": "pug_rest_batch",
                "pipeline_run_id": str(self.run_id),
                "input_checksum": input_checksum,
                "transformations": transformations_str,
                "as_of_date": (
                    self.as_of_date.isoformat()
                    if self.as_of_date is not None
                    else None
                ),
            }

            # DESIGN-19, DQ-13: dedupe by InChIKey, keep lowest CID.
            if response_inchikey in by_inchikey:
                existing_cid = by_inchikey[response_inchikey]["pubchem_cid"]
                new_cid = safe_cid
                logger.info(
                    "[%s] Duplicate CID for inchikey=%s: existing=%d, "
                    "new=%d -- keeping lowest",
                    self.source_name,
                    response_inchikey,
                    existing_cid,
                    new_cid,
                )
                if new_cid < existing_cid:
                    by_inchikey[response_inchikey] = record
            else:
                by_inchikey[response_inchikey] = record

        transformations_applied.append("deduplicated_by_inchikey_lowest_cid")
        self._log_transformation(
            "parse_pubchem_response",
            len(properties_list),
            {
                "batch_idx": batch_idx,
                "input_records": len(properties_list),
                "output_records": len(by_inchikey),
                "dropped_records": len(properties_list) - len(by_inchikey),
                "drop_reasons": dict(
                    Counter(
                        r.get("reason", "unknown")
                        for r in self.dead_letter_queue
                        if r.get("batch_idx") == batch_idx
                    )
                ),
                "batch_sha256": batch_sha256,
            },
        )

        # LIN-8: stamp source_batch_idx and source_response_sha256.
        for record in by_inchikey.values():
            record["_source_batch_idx"] = batch_idx
            record["_source_response_sha256"] = batch_sha256

        # P1-018 ROOT FIX (Team-2): assign ``stereo_parent_cid`` to each
        # record by grouping on the InChIKey connectivity layer (first 14
        # chars). Stereoisomers share this prefix; the lowest CID in each
        # group is the parent. This does NOT collapse stereoisomers --
        # each remains a separate Compound node -- but provides the
        # linkage for downstream entity resolution and KG build.
        self._assign_stereo_parent_cids(list(by_inchikey.values()))

        return list(by_inchikey.values())

    @staticmethod
    def _assign_stereo_parent_cids(records: list[dict[str, Any]]) -> None:
        """P1-018 ROOT FIX: assign ``stereo_parent_cid`` to each record.

        Groups records by ``inchikey_connectivity_layer`` (first 14 chars
        of the InChIKey -- the molecular skeleton, shared by stereoisomers).
        Within each group, the lowest CID is the ``stereo_parent_cid``.
        Mutates records in place.

        For thalidomide:
          CID 5462502 (R) -> inchikey_connectivity_layer = "CPALVAVH...X"
          CID 5462504 (S) -> inchikey_connectivity_layer = "CPALVAVH...X"
          CID 3672 (rac)  -> inchikey_connectivity_layer = "CPALVAVH...X"
        All three get ``stereo_parent_cid = 3672`` (the lowest CID).
        Each remains a SEPARATE Compound node in the KG -- the
        stereochemistry (isomeric_smiles) is preserved. The GNN can
        learn DISTINCT embeddings for (R)- vs (S)-thalidomide, which is
        life-safety-critical (the (S)-enantiomer is teratogenic).
        """
        # Group CIDs by connectivity layer.
        groups: dict[str, list[int]] = {}
        for rec in records:
            layer = rec.get("inchikey_connectivity_layer")
            cid = rec.get("pubchem_cid")
            if layer and cid is not None:
                groups.setdefault(layer, []).append(int(cid))
        # Assign parent (lowest CID in each group).
        for rec in records:
            layer = rec.get("inchikey_connectivity_layer")
            cid = rec.get("pubchem_cid")
            if layer and cid is not None and layer in groups:
                rec["stereo_parent_cid"] = min(groups[layer])
            else:
                # No connectivity layer or singleton -- parent is itself.
                rec["stereo_parent_cid"] = cid

    # ------------------------------------------------------------------
    # Internals: type conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(
        value: Any,
        field_name: str = "",
        inchikey: str = "",
    ) -> Optional[Decimal]:
        """Convert a value to ``Decimal`` (SCI-16, DESIGN-8, CODE-25, CODE-26).

        Returns ``None`` for null-like inputs, booleans (CODE-25), NaN
        floats (CODE-26), NaN strings, and unparseable values.  Uses
        ``Decimal(str(value))`` to avoid binary-float artifacts (e.g.
        ``Decimal(180.063388)`` gives ``180.06338800000002``;
        ``Decimal(str(180.063388))`` gives ``180.063388``).  Quantizes
        to 6 decimal places with ``ROUND_HALF_UP``.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            logger.warning(
                "[pubchem] _safe_float: %s returned boolean %r for "
                "inchikey=%s -- rejecting",
                field_name,
                value,
                inchikey,
            )
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in NULL_STRING_VALUES:
                return None
        try:
            d = Decimal(str(value))
            return d.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        except (InvalidOperation, ValueError, TypeError) as exc:
            logger.warning(
                "[pubchem] _safe_float: cannot convert %s=%r to Decimal "
                "for inchikey=%s: %s",
                field_name,
                value,
                inchikey,
                exc,
            )
            return None

    @staticmethod
    def _safe_int(
        value: Any,
        field_name: str = "",
        inchikey: str = "",
    ) -> Optional[int]:
        """Convert a value to ``int`` (DESIGN-18, CODE-1, CODE-25, CODE-26).

        Returns ``None`` for null-like inputs, booleans (CODE-25), NaN
        floats (CODE-26), NaN strings, and unparseable values.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            logger.warning(
                "[pubchem] _safe_int: %s returned boolean %r for "
                "inchikey=%s -- rejecting",
                field_name,
                value,
                inchikey,
            )
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in NULL_STRING_VALUES:
                return None
        try:
            # Use Decimal to handle "1.0" -> 1 robustly.
            return int(Decimal(str(value)))
        except (InvalidOperation, ValueError, TypeError) as exc:
            logger.warning(
                "[pubchem] _safe_int: cannot convert %s=%r to int for "
                "inchikey=%s: %s",
                field_name,
                value,
                inchikey,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Internals: CAS lookup (optional, SCI-6)
    # ------------------------------------------------------------------

    def _fetch_cas_for_cid(self, cid: int) -> Optional[str]:
        """Fetch the CAS Registry Number for a PubChem CID (SCI-6).

        Uses the ``/pug/compound/cid/{cid}/synonyms/JSON`` endpoint.
        The CAS number is the first synonym matching
        ``^\\d{2,7}-\\d{2}-\\d$``.  Returns ``None`` on any failure
        (no CAS in synonyms, network error, etc.).
        """
        url = f"{self.rest_base}/compound/cid/{cid}/synonyms/JSON"
        headers = {
            "Accept": "application/json",
            "User-Agent": self._user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        # v83 P2-7: send API key via header (not query string).
        params: dict[str, Any] = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        try:
            self._api_call_count += 1
            resp = self.http_session.get(
                url,
                headers=headers,
                params=params,
                timeout=_SYNONYM_LOOKUP_TIMEOUT,
                # v65 ROOT FIX (P1-035): rewritten as an explicit ternary.
                # The previous ``verify=(self.ca_bundle or True)`` was
                # functionally correct but read as "always True".
                verify=(self.ca_bundle if self.ca_bundle else True),
                cert=(
                    (self.cert_pem, self.key_pem)
                    if (self.cert_pem and self.key_pem)
                    else None
                ),
            )
            if not (200 <= resp.status_code < 300):
                return None
            data = resp.json()
            synonyms = (
                data.get("InformationList", {})
                .get("Information", [{}])[0]
                .get("Synonym", [])
            )
            cas_re = re.compile(r"^\d{2,7}-\d{2}-\d$")
            for syn in synonyms:
                if isinstance(syn, str) and cas_re.match(syn):
                    return syn
        except RETRYABLE_EXCEPTIONS as exc:
            logger.debug(
                "[%s] CAS lookup for CID %d failed: %s",
                self.source_name,
                cid,
                exc,
            )
        except (ValueError, KeyError, IndexError):
            return None
        return None

    # ------------------------------------------------------------------
    # Internals: file utilities
    # ------------------------------------------------------------------

    def _verify_sha256_sidecar(self, path: Path) -> bool:
        """Verify a file against its ``.sha256`` sidecar (DQ-15, IDEM-2)."""
        sha256_path = path.with_suffix(path.suffix + ".sha256")
        if not sha256_path.exists():
            return False
        try:
            with open(sha256_path, "r", encoding="utf-8") as fh:
                line = fh.read().strip()
            # Format: ``<sha256>  <filename>``
            expected = line.split()[0]
            actual = self._compute_sha256(path)
            return expected == actual
        except OSError:
            return False

    def _write_dead_letters_file(self) -> None:
        """Write ``self.dead_letter_queue`` to ``pubchem_dead_letters.csv``.

        Uses ``QUOTE_NONNUMERIC`` quoting, UTF-8 encoding, Unix line
        endings, and writes a SHA-256 sidecar (ARCH-8, REL-4).  No-op
        when the queue is empty.
        """
        if not self.dead_letter_queue:
            return
        if self.raw_dir is None:
            self._ensure_directories()
        assert self.raw_dir is not None  # noqa: S101
        dest = self.raw_dir / "pubchem_dead_letters.csv"
        # Collect all keys across all dead-letter entries -- different
        # failure paths produce different keys.
        all_keys: set[str] = set()
        for entry in self.dead_letter_queue:
            all_keys.update(entry.keys())
        # Sort for determinism (COMP-10).
        fieldnames = sorted(all_keys)
        with open(dest, "w", encoding="utf-8", newline="\n") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_NONNUMERIC,
                extrasaction="ignore",
            )
            writer.writeheader()
            for entry in self.dead_letter_queue:
                # Stringify values -- DictWriter with QUOTE_NONNUMERIC
                # requires all values to be str or numbers.
                row = {
                    k: ("" if v is None else str(v))
                    for k, v in entry.items()
                }
                writer.writerow(row)
        sha256 = self._compute_sha256(dest)
        sha256_path = dest.with_suffix(dest.suffix + ".sha256")
        with open(sha256_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"{sha256}  {dest.name}\n")
        # SEC-7: set file permissions to 0o600 (owner-only).
        try:
            os.chmod(dest, 0o600)
            os.chmod(sha256_path, 0o600)
        except OSError:
            pass  # Windows / non-POSIX
        logger.info(
            "[%s] Wrote %d dead-letter records to %s (sha256=%s)",
            self.source_name,
            len(self.dead_letter_queue),
            dest,
            sha256,
        )

    def _write_dead_letter_dataframe(
        self, df: pd.DataFrame, reason: str
    ) -> None:
        """Append a failing DataFrame to the dead-letter queue (CODE-23).

        Used when ``load()`` catches an exception from a loader -- the
        failing rows are preserved for inspection.
        """
        for _, row in df.iterrows():
            self.dead_letter_queue.append(
                {
                    "inchikey": row.get("inchikey"),
                    "reason": reason,
                    "pubchem_cid": row.get("pubchem_cid"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )


# ---------------------------------------------------------------------------
# Backward-compat module-level aliases (DOC-4, COMP-8, prompt Section 3
# rule: "No deletion of existing imports, constants, or helper functions --
# extend them, rename them safely (with backward-compat aliases if exported),
# but do not remove.").
#
# The legacy pubchem_pipeline.py exposed these as module-level constants.
# Other modules (and tests) import them via
# ``pipelines/__init__.py:_SYMBOL_MAP``.  The institutional-grade rewrite
# moves the configuration into ``config/settings.py`` and instance
# attributes -- but the module-level aliases are retained for backward
# compatibility.  They read from the same settings so the values are
# always in sync.
# ---------------------------------------------------------------------------

# The 15 PubChem properties fetched per CID.  Mirror of
# ``settings.PUBCHEM_PIPELINE_PROPERTIES``.
PUBCHEM_PROPERTIES: list[str] = list(PUBCHEM_PIPELINE_PROPERTIES)

# Legacy batch size -- now ``settings.PUBCHEM_PIPELINE_BATCH_SIZE``.
BATCH_SIZE: int = PUBCHEM_PIPELINE_BATCH_SIZE

# Legacy max retries -- now ``settings.ENTITY_RESOLUTION_PUBCHEM_MAX_RETRIES``.
MAX_RETRIES: int = ENTITY_RESOLUTION_PUBCHEM_MAX_RETRIES

# Legacy min backoff -- now ``settings.PUBCHEM_PIPELINE_MIN_BACKOFF``.
MIN_BACKOFF: float = PUBCHEM_PIPELINE_MIN_BACKOFF

# Legacy max backoff -- now ``settings.PUBCHEM_PIPELINE_MAX_BACKOFF``.
MAX_BACKOFF: float = PUBCHEM_PIPELINE_MAX_BACKOFF

# Legacy rate-limit interval -- now
# ``settings.ENTITY_RESOLUTION_PUBCHEM_CALL_DELAY``.
RATE_LIMIT_INTERVAL: float = ENTITY_RESOLUTION_PUBCHEM_CALL_DELAY
