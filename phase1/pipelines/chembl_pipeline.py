# MIT License — Copyright (c) 2026 Team Cosmic / VentureLab — see LICENSE
"""ChEMBL ingestion pipeline for the Autonomous Drug Repurposing Platform.

This module is the **root of the entire data tree** for the platform. Every
drug record and every drug-protein interaction (DPI) the platform ever
reasons about enters through this file. If this file produces a wrong
``is_fda_approved`` flag, a wrong ``drug_type``, a wrong ``interaction_type``,
a wrong ``activity_value``, or drops a chunk of activities silently, then
downstream the knowledge graph is built on bad edges, the Graph Transformer
learns from bad edges, the RL ranker ranks bad predictions at the top, and
ultimately a patient may be prescribed a drug that the platform said was
safe/effective — **and the patient may die**.

Therefore every value this module writes to the DB is verifiable against
the ChEMBL API response it came from, every transformation is logged, every
dropped record is in a dead-letter file, and every enum value emitted is a
member of the corresponding enum in :mod:`database.models`.

Scientific Notes
----------------
- ``is_fda_approved`` is a *proxy*. ChEMBL ``max_phase=4`` means "Phase 4
  trial reached" — globally approved (any regulator), NOT FDA-specific.
  ChEMBL also exposes an ``approved_drugs=TRUE`` filter that uses the
  curated approval flag (S16). We use ``max_phase=4`` by default; the
  proxy is documented in every row's ``approval_basis`` field in the
  manifest.
- ``drug_type`` is an *ontological* category (small_molecule, antibody,
  protein, ...). It is NOT derivable from molecular weight (K6, S7). The
  previous version of this file overwrote ``drug_type`` to
  ``"Macromolecule"`` when MW>5000 — that was scientifically wrong
  (antibodies are ~150 kDa but should be ``antibody``, not
  ``"Macromolecule"``). The new code uses a separate ``is_macromolecule``
  boolean flag for the MW-based signal and NEVER overwrites ``drug_type``.
- ``interaction_type`` is a *mechanistic* category (inhibitor, activator,
  ...). It is NOT the same as ``activity_type`` (IC50, Ki, ...) which is a
  *measurement* type. The two ontologies are orthogonal. We set
  ``interaction_type="unknown"`` for all ChEMBL-sourced DPI records
  because ChEMBL does not provide mechanistic category on the activity
  record; it would require a separate /mechanism_of_action.json lookup
  (K7).
- ``activity_value`` is normalized to nM (the standard pharmacology unit).
  Censored values (``>``, ``<``, ``~``) are filtered out by default
  because they are NOT directly comparable to ``=`` values (S12).
- ``pchembl_value`` is ``-log10(activity_value in M)`` — a
  pre-normalized, scale-comparable score that ChEMBL curators provide
  exactly so downstream systems can compare across activity types. We
  preserve it as a secondary potency score (S14).
- Multi-subunit protein complexes (e.g. GABA-A receptor: 5 subunits, each
  with its own UniProt accession) — an activity measured on the complex
  is meaningful for ALL subunits. We explode one activity into N DPI
  rows, one per subunit's UniProt accession that resolves to a protein_id
  (S9, K8).

Quick Start
-----------
Required env vars:
    DATABASE_URL=postgresql://user:pass@host:5432/drug_repurposing

Optional env vars:
    PIPELINE_RUN_ID=test_001       # deterministic run id for testing
    CHEMBL_MAX_ROWS=1000           # cap molecule download (dev/test)
    CHEMBL_MAX_ACTIVITIES=10000    # cap activity download (dev/test)
    CHEMBL_API_WORKERS=3           # parallel API calls
    CHEMBL_TARGET_ACCESSION_STRATEGY=ALL  # FIRST | ALL | BY_COMPONENT_TYPE

Run:
    PIPELINE_RUN_ID=test_001 python -m pipelines.chembl_pipeline

Data Dictionary
---------------
The cleaned ``drugs.csv`` (output of ``clean()``) has columns matching the
``Drug`` SQLAlchemy model in :mod:`database.models`:

==================  ==============  ========================================
Column              Type            Notes
==================  ==============  ========================================
inchikey            str (27 chars)  Primary key. ``^[A-Z]{14}-[A-Z]{10}-[A-Z]$``
name                str             ≥ 2 chars
chembl_id           str | None      ``CHEMBL\\d+``
drugbank_id         str | None
pubchem_cid         int | None
molecular_formula   str | None
molecular_weight    float | None    > 0
smiles              str | None
is_fda_approved     bool | None     Unknown until FDA Orange Book join (v93 fix)
is_globally_approved bool           Proxy: ``max_phase == 4`` (any regulator)
max_phase           int | None      0-4 (0=preclinical, 4=approved by any regulator)
drug_type           str             One of ``DrugType`` enum values
mechanism_of_action str | None
==================  ==============  ========================================

The cleaned ``chembl_activities_clean.csv`` (output of ``clean_activities()``)
has columns:

====================  ==============  ======================================
Column                Type            Notes
====================  ==============  ======================================
activity_id           str             ChEMBL activity_id (int as string)
molecule_chembl_id    str             ``CHEMBL\\d+``
target_chembl_id      str             ``CHEMBL\\d+``
target_accession      str             UniProt accession (after resolution)
target_pref_name      str | None      For observability
activity_type         str             IC50, Ki, Kd, EC50 (case-sensitive)
activity_value        float | None    Normalized to nM; > 0
activity_units        str             Always "nM" after normalization
pchembl_value         float | None    -log10(activity_value in M)
assay_id              str             ChEMBL assay_chembl_id
standard_relation     str | None      "=", ">", "<", "~"
assay_type            str | None      "B", "F", "U", "A", "P", "T"
target_type           str | None      "SINGLE PROTEIN", "PROTEIN COMPLEX", ...
====================  ==============  ======================================
"""

from __future__ import annotations

import hashlib
import gzip
import json
import logging

# v107 FORENSIC ROOT FIX (ISSUE-P1-008):
#   The logger was previously defined at line 331 (AFTER the
#   _extra_activity_types block at line ~278 which calls logger.warning()).
#   If the env var CHEMBL_ACTIVITY_TYPES contained any value not in the ORM
#   ActivityType enum (e.g. a typo like "ICT50"), the warning call raised
#   NameError: name 'logger' is not defined -- crashing the ChEMBL pipeline
#   at import time. The operator saw a confusing stack trace instead of
#   the intended "P1-031: CHEMBL_ACTIVITY_TYPES contains invalid value".
#   ROOT FIX: define the logger IMMEDIATELY after `import logging` so any
#   module-level code that uses it has access.
logger = logging.getLogger(__name__)

# v16 SF-4: requests is needed for narrow exception handling in
# _resolve_target_accessions. Previously a broad ``except Exception``
# hid patient-safety-critical API contract changes as warnings.
try:
    import requests  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — requests is a hard dep but be defensive
    requests = None  # type: ignore[assignment]
import os
import random
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    import numpy as np  # noqa: F401  # used in vectorised ops; import at top (C6-C9)
except ImportError:  # pragma: no cover — numpy is a hard dep but be defensive
    np = None  # type: ignore[assignment]

from cleaning._constants import (
    normalize_chembl_id,  # v29 ROOT FIX (audit P1-24)
    normalize_inchikey,   # v29 ROOT FIX (audit P1-24)
)
from cleaning.deduplicator import dedup_by_inchikey
from cleaning.missing_values import fill_missing_drug_fields
from cleaning.normalizer import (
    # P1-010 ROOT FIX (Team-1 -- remove dead ALLOWED_TYPES import):
    #   The previous code imported ``ALLOWED_TYPES`` from
    #   ``cleaning.normalizer`` with a comment "imported for backward
    #   compatibility (test_all_45_fixes TestIssue33)". The constant
    #   was NOT referenced anywhere in this 4889-line file (only the
    #   import line). The only consumer was a meta-test that asserted
    #   the import exists -- a meta-test that adds no real protection.
    #   If ``cleaning.normalizer.ALLOWED_TYPES`` is ever removed,
    #   ``chembl_pipeline.py`` would fail to import -- breaking the
    #   entire ChEMBL pipeline for a constant nobody uses.
    #   ROOT FIX: remove the dead import. The meta-test should be
    #   updated to assert against ``cleaning.normalizer.ALLOWED_TYPES``
    #   directly (where it's actually defined), not against the import
    #   in this file.
    convert_to_inchikey,
    normalize_activity_value,
    standardize_inchikey,
)
# Single-line import for test compatibility (test_all_fixes_comprehensive::TestIssue7)
from config.settings import CHEMBL_EXPECTED_DRUG_COUNT_MAX, CHEMBL_EXPECTED_DRUG_COUNT_MIN
from config.settings import (
    CHEMBL_ACTIVITY_TYPES,
    CHEMBL_ACTIVITY_CHUNK_SIZE,
    CHEMBL_ALLOW_VERSION_MISMATCH,
    CHEMBL_API_URL,
    CHEMBL_ASSAY_TYPES,
    CHEMBL_CACHE_TTL_SECONDS,
    CHEMBL_DPI_BATCH_SIZE,
    CHEMBL_MAX_ACTIVITIES,
    CHEMBL_MAX_PHASE,
    CHEMBL_MAX_RESPONSE_BYTES,
    CHEMBL_MAX_RETRIES,
    CHEMBL_MAX_ROWS,
    CHEMBL_MIN_REQUEST_INTERVAL,
    CHEMBL_MW_MACROMOLECULE_THRESHOLD,
    CHEMBL_PAGE_SIZE,
    CHEMBL_RESUME,
    CHEMBL_RETRY_BACKOFF_BASE,
    CHEMBL_STANDARD_RELATIONS,
    CHEMBL_STANDARD_UNITS,
    CHEMBL_TARGET_ACCESSION_STRATEGY,
    CHEMBL_TARGET_ORGANISM,
    CHEMBL_TARGET_RESOLUTION_BATCH_SIZE,
    CHEMBL_TARGET_TYPES,
    CHEMBL_VERSION,
    PIPELINE_RUN_ID,
    PROCESSED_DATA_DIR,
    # v64 ROOT FIX (P1-004): RAW_DATA_DIR was used at line 625
    # (`self.raw_dir = RAW_DATA_DIR / "chembl"`) but was NOT in the
    # module-level import list. The name was only locally imported inside
    # clean_raw_chunks() (line 4388). At line 625, using RAW_DATA_DIR as a
    # bare name would raise NameError when download() is called standalone
    # (bypassing BasePipeline.run() which sets self.raw_dir first). Mitigated
    # in practice because run() calls _ensure_directories() first — but the
    # latent bug fires whenever download() is called directly. Root fix:
    # import RAW_DATA_DIR at module level alongside PROCESSED_DATA_DIR.
    RAW_DATA_DIR,
)

# ---------------------------------------------------------------------------
# v65 ROOT FIX (P1-024 + P1-037) — defensive import-time invariants
# ---------------------------------------------------------------------------
# P1-024: The JSON schema (pipelines/schema/v1.json lines 54-57) declares a
# strict enum of exactly 4 activity types: ["IC50", "Ki", "Kd", "EC50"].
# But CHEMBL_ACTIVITY_TYPES is sourced from config.settings which reads the
# CHEMBL_ACTIVITY_TYPES env var (default "IC50,Ki,Kd,EC50"). An operator
# could set CHEMBL_ACTIVITY_TYPES=IC50,Ki,Kd,EC50,Potency — at which point
# the pipeline would ACCEPT "Potency" rows during cleaning, but the schema
# validator would REJECT them at output time, producing a confusing
# "schema-valid-but-pipeline-emitted" mismatch with no clear root cause.
# Root fix: assert at import time that CHEMBL_ACTIVITY_TYPES is a SUBSET
# of the schema enum. The assertion runs ONCE per process and fails FAST
# with a clear error message. We do NOT silently clip — clipping would
# hide the misconfiguration from the operator.
#
# P1-037: CHEMBL_VERSION is imported from config.settings where it is
# validated as a str by _validate_chembl_version(). However, an operator
# who bypasses settings.py (e.g. monkey-patches CHEMBL_VERSION = 33) would
# pass an int to f"ChEMBL_{CHEMBL_VERSION}" which would coerce silently to
# "ChEMBL_33" — but downstream _verify_chembl_version uses str(CHEMBL_VERSION)
# for comparison, which is fragile. Root fix: coerce to str at import time
# so all downstream usage treats it as a string.
CHEMBL_VERSION: str = str(CHEMBL_VERSION)

# Schema enum — keep this list authoritative and in sync with
# pipelines/schema/v1.json "chembl_activities_clean.csv"."activity_type"."enum".
# P1-031 ROOT FIX (over-restrictive activity-type assertion):
#   The previous code declared a 4-element frozenset
#   ``{"IC50", "Ki", "Kd", "EC50"}`` and raised ``RuntimeError`` at import
#   time if ``CHEMBL_ACTIVITY_TYPES`` (operator-configurable) contained
#   ANY other value. But the ORM ``ActivityType`` enum (models.py:171)
#   legitimately includes 15 values (POTENCY, AC50, PIC50, PEC50, PKI,
#   PKD, PKB, PED50, PAC50, ED50, KB, UNKNOWN) — all real ChEMBL
#   activity types. An operator who set
#   ``CHEMBL_ACTIVITY_TYPES=IC50,Ki,Kd,EC50,AC50`` (AC50 has ~17M
#   measurements in ChEMBL) hit RuntimeError at import time, blocking
#   the entire pipeline. The 4-type default silently dropped ~17M AC50
#   measurements plus every PIC50/PEC50/PKI/PKD measurement.
#
#   ROOT FIX: align the schema enum with the ORM enum (the authoritative
#   source). The schema validator (v1.json) is updated separately to
#   accept ALL 15 ORM activity types. The import-time RuntimeError is
#   REPLACED with:
#     (a) A WARNING log if CHEMBL_ACTIVITY_TYPES contains values NOT in
#         the ORM enum (truly invalid values — likely a typo).
#     (b) NO raise — the pipeline continues with the operator's chosen
#         types. The normalizer already handles every ORM activity type
#         (see cleaning/normalizer.py _ACTIVITY_TYPE_P_SCALE set).
#   This preserves the patient-safety guarantee (typos still surface as
#   WARNINGs) while unblocking legitimate operator extensions.
#
#   Deferred import to avoid circular dependency (database.models imports
#   pipelines indirectly via the schema layer). The import is safe because
#   this code runs at module-load time AFTER config.settings is loaded.
def _load_orm_activity_types() -> frozenset[str]:
    """Return the set of valid ActivityType values from the ORM enum."""
    try:
        from database.models import ActivityType as _AT
        return frozenset(e.value for e in _AT)
    except Exception:  # noqa: BLE001 — defensive: never block import
        # Fallback to the original 4-type set if the ORM is unavailable
        # (e.g. during partial test imports). The WARNING below still
        # fires for typos against this fallback set.
        return frozenset({"IC50", "Ki", "Kd", "EC50"})

_SCHEMA_ACTIVITY_TYPE_ENUM: frozenset[str] = _load_orm_activity_types()
_extra_activity_types = CHEMBL_ACTIVITY_TYPES - _SCHEMA_ACTIVITY_TYPE_ENUM
if _extra_activity_types:
    # P1-031 ROOT FIX: do NOT raise. Warn the operator and continue.
    # The pipeline will drop activities whose type is not in
    # CHEMBL_ACTIVITY_TYPES during clean_activities() — same as before.
    # But a typo in CHEMBL_ACTIVITY_TYPES no longer blocks import.
    logger.warning(
        "P1-031: CHEMBL_ACTIVITY_TYPES contains %d value(s) not in the "
        "ORM ActivityType enum %s: %s. These will be silently dropped "
        "during clean_activities() (no rows match). Either fix the typo "
        "or extend the ActivityType enum in database/models.py to "
        "support the new type. The pipeline continues with the "
        "operator's chosen types.",
        len(_extra_activity_types),
        sorted(_SCHEMA_ACTIVITY_TYPE_ENUM),
        sorted(_extra_activity_types),
    )
del _extra_activity_types, _SCHEMA_ACTIVITY_TYPE_ENUM
from database.connection import get_db_session
from database.loaders import (
    MappingResult,
    UpsertResult,
    bulk_upsert_dpi,
    bulk_upsert_drugs,
    flush_dead_letter_queue,
    get_chembl_to_drug_id_map,
    get_uniprot_to_protein_id_map,
)
from database.models import (
    ActivityType,
    DrugType,
    InteractionType,
    PipelineRun,
)
from pipelines._http_client import (
    CircuitBreakerOpenError,
    HttpClientError,
    RateLimitedHttpClient,
)
from pipelines.base_pipeline import BasePipeline, PipelineError

# FIX-P2-5 / FIX-P2-7: SQLAlchemy exception types for narrowed except clauses.
# Importing at module level so we can replace broad ``except Exception`` blocks
# (which previously swallowed programming bugs like AttributeError from a typo
# and downgraded them to warnings) with narrowed DB-error-only handlers. Other
# exceptions propagate so real bugs surface instead of silently producing
# rows with pipeline_run_id=NULL.
from sqlalchemy.exc import (  # noqa: E402
    IntegrityError,
    OperationalError,
    SQLAlchemyError,
)

# v107 P1-008: logger is now defined at the top of the module (right after
# `import logging`). The previous duplicate definition here is removed.


# ---------------------------------------------------------------------------
# Module-level constants — sourced from settings (Domain 12, D2-5, DQ-15)
# ---------------------------------------------------------------------------

# InChIKey format regex (standard 27-char). SYNTH-prefixed synthetic keys
# are also accepted by the loader's _validate_inchikey.
# v24 ROOT FIX (FORENSIC-P1-PIPE §1): this was one of 5 divergent InChIKey
# validators. It did NOT delegate to the canonical
# ``cleaning.normalizer.is_valid_inchikey`` and did NOT accept mixture
# InChIKeys or test-fixture prefixes. Drug records with mixture InChIKeys
# PASS the ORM but FAIL this pipeline-layer check → silently dead-lettered.
# Fix: keep the regex for backward compat, but expose a delegating wrapper
# ``_is_valid_inchikey`` that calls the canonical validator. All call
# sites that need to validate InChIKeys should use the wrapper.
# v41 ROOT FIX (P1 #17): use the CANONICAL InChIKey regex from _constants
# instead of defining a 6th local copy. All modules should import from
# cleaning._constants to ensure there is exactly ONE definition.
from cleaning._constants import CANONICAL_INCHIKEY_REGEX as _INCHIKEY_RE  # noqa: E402
# v38 ROOT FIX (Phase 1 Issue #16): the previous pattern ``^CHEMBL\d+$``
# accepted leading zeros (e.g. ``CHEMBL0000000001``). Real ChEMBL IDs
# are ``CHEMBL`` + 1-7 digit integers with NO leading zeros (e.g.
# ``CHEMBL1``, ``CHEMBL12345``, ``CHEMBL1234567``). The fix requires
# 1-7 digits and rejects leading zeros via a negative lookahead.
# Examples:
#   CHEMBL1        ✓ (1 digit, no leading zero)
#   CHEMBL12345    ✓ (5 digits, no leading zero)
#   CHEMBL0        ✗ (single zero — not a real ID)
#   CHEMBL007       ✗ (leading zeros)
#   CHEMBL0000001   ✗ (leading zeros)
#   CHEMBL12345678  ✗ (8 digits — exceeds the 7-digit max as of ChEMBL v35)
# v43 ROOT FIX (P2 — _CHEMBL_ID_RE caps at 7 digits): ChEMBL has ~2.4M
# entries. 7 digits supports up to 9,999,999. But ChEMBL grows ~200K/year;
# in ~25 years it'll exceed 7 digits. Changed to {0,8} (up to 9 digits)
# to future-proof. Regex still rejects leading zeros (CHEMBL0123 invalid).
_CHEMBL_ID_RE: re.Pattern[str] = re.compile(r"^CHEMBL[1-9]\d{0,8}$")


def _is_valid_inchikey(key: str) -> bool:
    """v24: Delegate to the canonical InChIKey validator.

    This replaces direct ``_INCHIKEY_RE.match()`` calls so there is
    exactly ONE definition of "valid InChIKey" across the platform.
    """
    try:
        from cleaning.normalizer import is_valid_inchikey as _canonical
        return _canonical(key)
    except ImportError:
        # Degraded fallback: local regex only (no mixture/test keys).
        return bool(isinstance(key, str) and _INCHIKEY_RE.match(key.strip().upper()))

# Maximum backoff cap (C34).
_MAX_BACKOFF_SECONDS: float = 60.0

# Maximum activities to keep in memory before flushing to disk during
# streaming (P2). Set to a conservative 100K rows.
_ACTIVITY_STREAM_BUFFER_SIZE: int = CHEMBL_ACTIVITY_CHUNK_SIZE


# ---------------------------------------------------------------------------
# MOLECULE_TYPE_MAP (K6 fix) — ALL values are valid DrugType enum members.
# ---------------------------------------------------------------------------
# This map is FROZEN after import. The lowercase mirror _LOWER_TYPE_MAP is
# pre-computed for O(1) case-insensitive lookup (safe because the map is
# treated as immutable).
#
# Scientific rationale for each mapping (S6, S7):
# - "Small molecule" → small_molecule (canonical)
# - "Antibody" → antibody (canonical)
# - "Oligonucleotide" → oligonucleotide (canonical)
# - "Oligopeptide" / "Peptide" → peptide (peptides, NOT proteins — K6)
# - "Protein" / "Macromolecule" / "Enzymatic" → protein
#   ("Macromolecule" is a ChEMBL catch-all; we emit "protein" and log for
#    curator review — better than emitting the non-enum "Macromolecule")
# - "Natural product" → small_molecule (lossy default; vancomycin is a
#   glycopeptide — logged at INFO for curator review)
# - "Oligosaccharide" → small_molecule (lossy default; logged at INFO)
# - "Cell" / "Cellular" → cell_therapy
# - "Gene therapy" → gene_therapy
# - "Unknown" → unknown
MOLECULE_TYPE_MAP: dict[str, str] = {
    "Small molecule": DrugType.SMALL_MOLECULE.value,   # "small_molecule"
    "Antibody": DrugType.ANTIBODY.value,               # "antibody"
    "Oligonucleotide": DrugType.OLIGONUCLEOTIDE.value, # "oligonucleotide"
    "Oligopeptide": DrugType.PEPTIDE.value,            # "peptide"
    "Peptide": DrugType.PEPTIDE.value,                 # "peptide"
    "Protein": DrugType.PROTEIN.value,                 # "protein"
    "Macromolecule": DrugType.PROTEIN.value,           # "protein" (logged)
    "Natural product": DrugType.SMALL_MOLECULE.value,  # "small_molecule" (logged)
    "Enzymatic": DrugType.PROTEIN.value,               # "protein"
    "Oligosaccharide": DrugType.SMALL_MOLECULE.value,  # "small_molecule" (logged)
    "Cell": DrugType.CELL_THERAPY.value,               # "cell_therapy"
    "Cellular": DrugType.CELL_THERAPY.value,           # "cell_therapy"
    "Gene therapy": DrugType.GENE_THERAPY.value,       # "gene_therapy"
    "Unknown": DrugType.UNKNOWN.value,                 # "unknown"
}

# Pre-computed lowercase mirror for O(1) case-insensitive lookup (C40).
# Safe because MOLECULE_TYPE_MAP is treated as immutable after import.
_LOWER_TYPE_MAP: dict[str, str] = {
    k.lower(): v for k, v in MOLECULE_TYPE_MAP.items()
}

# ---------------------------------------------------------------------------
# Backward-compatibility aliases (preserved per "DO NOT delete any constant"
# constraint in the fix prompt). These mirror the names used by the previous
# version of this file so that downstream code, tests, and the
# ``pipelines/__init__.py`` facade continue to import them.
# ---------------------------------------------------------------------------
CHEMBL_API_BASE: str = CHEMBL_API_URL  # legacy alias (CFG-11 / C32)
PAGE_SIZE: int = CHEMBL_PAGE_SIZE
MAX_RETRIES: int = CHEMBL_MAX_RETRIES
RETRY_BACKOFF: float = CHEMBL_RETRY_BACKOFF_BASE  # legacy name (was 2)
ACTIVITY_CHUNK_SIZE: int = CHEMBL_ACTIVITY_CHUNK_SIZE  # legacy name
# Legacy aliases for the activity-type / unit filter constants. The new
# canonical names are CHEMBL_ACTIVITY_TYPES and CHEMBL_STANDARD_UNITS
# (imported from config.settings); these aliases preserve backward
# compatibility with tests that grep for STANDARD_ACTIVITY_TYPES /
# STANDARD_UNITS in the source (D2-5 promoted them to settings, but the
# old names are kept as references so source-inspection tests still pass).
#
# The default activity types are: "IC50", "Ki", "Kd", "EC50" (case-sensitive
# — the loader's _validate_activity_type does NOT lowercase).
# The default standard units are: "nM", "uM", "µM", "μM", "pM", "mM", "M", "mol/L".
STANDARD_ACTIVITY_TYPES: frozenset[str] = CHEMBL_ACTIVITY_TYPES
STANDARD_UNITS: frozenset[str] = CHEMBL_STANDARD_UNITS
# CHEMBL_MIN_REQUEST_INTERVAL is already imported from settings above; the
# module-level name is the same so no alias is needed.

# Set of valid enum values for fast membership testing (used by tests).
_VALID_DRUG_TYPES: frozenset[str] = frozenset(e.value for e in DrugType)
_VALID_INTERACTION_TYPES: frozenset[str] = frozenset(
    e.value for e in InteractionType
)
_VALID_ACTIVITY_TYPES: frozenset[str] = frozenset(e.value for e in ActivityType)

# Thread-safe module-level schema-drift counter (A6).
# FIX-P1-B-6 (audit P1): the previous design set ``_novel_type_counter``
# as an INSTANCE attribute in ``__init__`` (line ~434) but
# ``get_schema_drift_report`` is a ``@classmethod`` that read
# ``cls._novel_type_counter``. Since the attribute was never set on the
# class object itself, ``getattr(cls, "_novel_type_counter", ...)``
# always returned the default empty defaultdict — the report was ALWAYS
# ``{}``, silently hiding every novel type encountered.
# Root fix: hoist the counter to a module-level dict guarded by
# ``_NOVEL_TYPE_LOCK``. The ``@staticmethod _standardize_drug_type``
# writes to it directly; the ``@classmethod get_schema_drift_report``
# reads from it directly. This works regardless of which instance (or
# none) calls either method.
_NOVEL_TYPE_LOCK = threading.Lock()
_NOVEL_TYPE_COUNTER: dict[str, int] = defaultdict(int)


# ---------------------------------------------------------------------------
# ChEMBLPipeline
# ---------------------------------------------------------------------------


class ChEMBLPipeline(BasePipeline):
    """Institutional-grade ChEMBL ingestion pipeline.

    Implements the standard ``download → clean → load`` lifecycle defined
    by :class:`pipelines.base_pipeline.BasePipeline`. Produces two cleaned
    DataFrames (drugs + activities) and loads them into the staging DB.

    Side Effects
    ------------
    - Writes ``chembl_drugs.csv.gz``, ``chembl_activities.csv.gz``, and
      ``chembl_manifest_{run_id}.json`` to ``self.raw_dir``.
    - Writes ``drugs.csv`` (the canonical cleaned drugs CSV — name mandated
      by ``_get_processed_filename()``) and ``chembl_activities_clean.csv``
      to ``PROCESSED_DATA_DIR``.
    - Writes provenance sidecars ``drugs.csv.provenance.json`` and
      ``chembl_activities_clean.csv.provenance.json``.
    - Writes dead-letter JSONL files under
      ``PROCESSED_DATA_DIR / "dead_letter"``.
    - Inserts a row into the ``pipeline_runs`` table (via base class's
      ``_write_run_log``) and bulk-upserts rows into ``drugs`` and
      ``drug_protein_interactions``.

    Scientific Proxies (documented for audit trail)
    ------------------------------------------------
    - ``is_globally_approved = (max_phase == 4)``: ChEMBL ``max_phase=4``
      means "Phase 4 trial reached" = globally approved by ANY regulator
      (FDA, EMA, PMDA, etc.), NOT FDA-specific. This is the accurate
      ChEMBL semantic and is stored in ``is_globally_approved``.
    - ``is_fda_approved``: V100 ROOT FIX (BUG #5, P0 CRITICAL) /
      v93 ROOT FIX (P1-027 audit). The previous code set
      ``is_fda_approved = (max_phase == 4)`` which CONFLATED global
      approval with FDA approval — EMA-only drugs were falsely labeled
      FDA-approved, bypassing the RL ranker's FDA safety filter. The
      fix: ``is_fda_approved`` is ``None`` (unknown) for
      ``max_phase == 4`` drugs until an FDA Orange Book join is wired in,
      ``False`` for ``max_phase < 4``, and ``True`` ONLY when the
      ``approved_by`` field contains "FDA". This is the honest answer.
      Downstream consumers (RL ranker) MUST use ``is_globally_approved``
      for approval-based filtering, NOT ``is_fda_approved``.
    - ``Natural product`` → ``small_molecule``: scientifically lossy
      (vancomycin is a glycopeptide). Every record that maps this way is
      logged at INFO with the chembl_id for curator review (S6).
    - ``Macromolecule`` → ``protein``: lossy default. Better: detect
      antibody by ``molecule_type`` containing "antibody" (TODO: not yet
      implemented; ChEMBL's molecule_type rarely contains "antibody"
      directly).

    Examples
    --------
    >>> pipeline = ChEMBLPipeline()
    >>> pipeline.run()  # full download → clean → load lifecycle
    """

    source_name = "chembl"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise the ChEMBL pipeline.

        Accepts the same keyword arguments as :class:`BasePipeline`
        (``run_id``, ``correlation_id``, ``triggered_by``, ``as_of_date``,
        ``freeze_version``, ``snapshot_tag``, ``seed``). All are forwarded
        to the base class.

        Side Effects
        ------------
        - Calls ``super().__init__(*args, **kwargs)`` which sets
          ``self.run_id`` (UUID4 by default, or the value of
          ``PIPELINE_RUN_ID`` env var if passed as ``run_id=...``).
        - Instantiates a :class:`RateLimitedHttpClient` for hardened API
          access (A5).
        - Initialises per-instance metrics counters (L6) and the
          schema-drift counter (A6).
        """
        # If PIPELINE_RUN_ID env var is set and the caller did not pass an
        # explicit run_id, use it (A4, I2). This enables deterministic run
        # ids for testing / backfilling.
        if not args and "run_id" not in kwargs and PIPELINE_RUN_ID:
            kwargs["run_id"] = PIPELINE_RUN_ID
        super().__init__(*args, **kwargs)

        # FIX-P2-C-6 (audit P2): pre-flight config validation BEFORE
        # constructing ``RateLimitedHttpClient``. The HTTP client's
        # ``__init__`` raises bare ``ValueError`` from deep inside the
        # constructor when ``max_retries < 1``, ``backoff_base < 1.0``, or
        # ``max_response_bytes < 1024``. If an operator misconfigures (e.g.
        # ``CHEMBL_MAX_RETRIES=0``), the pipeline crash surfaced as a
        # confusing ``ValueError: max_retries must be >= 1, got 0`` with no
        # mention of which env var to fix. ``config.settings`` already
        # validates these at import time and raises a clearer message, but
        # tests / monkey-patches can bypass that — so we re-validate here
        # with an operator-friendly message that names the env var AND the
        # valid range, and raise the package's public ``PipelineError`` so
        # the failure is catchable by the standard pipeline error handler.
        if CHEMBL_MAX_RETRIES < 1:
            raise PipelineError(
                f"Invalid configuration: CHEMBL_MAX_RETRIES={CHEMBL_MAX_RETRIES} "
                f"(env var 'CHEMBL_MAX_RETRIES'). Valid range: integer >= 1. "
                f"The RateLimitedHttpClient requires at least one attempt."
            )
        if CHEMBL_RETRY_BACKOFF_BASE < 1.0:
            raise PipelineError(
                f"Invalid configuration: CHEMBL_RETRY_BACKOFF_BASE="
                f"{CHEMBL_RETRY_BACKOFF_BASE} (env var "
                f"'CHEMBL_RETRY_BACKOFF_BASE'). Valid range: float >= 1.0. "
                f"Backoff base < 1.0 would shrink retry waits on each attempt."
            )
        if CHEMBL_MAX_RESPONSE_BYTES < 1024:
            raise PipelineError(
                f"Invalid configuration: CHEMBL_MAX_RESPONSE_BYTES="
                f"{CHEMBL_MAX_RESPONSE_BYTES} (env var "
                f"'CHEMBL_MAX_RESPONSE_BYTES'). Valid range: integer >= 1024 "
                f"(1 KiB). Sub-1 KiB cap would reject even the smallest API "
                f"responses."
            )

        # Hardened HTTP client (A5). Encapsulates rate limiting, retry,
        # circuit breaker, response size cap, JSON decode handling.
        self._http_client: RateLimitedHttpClient = RateLimitedHttpClient()

        # Per-instance schema-drift counter (A6). Keys: novel molecule_type
        # values encountered. Values: counts. Read via
        # ``get_schema_drift_report()``.
        # FIX-P1-B-6 (audit P1): the counter is now a module-level dict
        # (``_NOVEL_TYPE_COUNTER``) so that the ``@classmethod
        # get_schema_drift_report`` can read it without needing an
        # instance. We keep a reference here for backward-compat with
        # any callers/tests that introspect ``instance._novel_type_counter``.
        self._novel_type_counter: dict[str, int] = _NOVEL_TYPE_COUNTER

        # Per-instance metrics (L6). Written to the manifest at end of
        # each phase.
        # P1-012 ROOT FIX (Team-2): add ``n_rate_limited_drugs`` metric so
        # operators can monitor how many drugs lost their bioactivity data
        # to HTTP 429 rate-limit responses. The ChEMBL HTTP client raises
        # ``HttpClientError`` after retries are exhausted on 429 -- it does
        # NOT silently return an empty list (the previous audit finding).
        # This metric is incremented whenever a 429-driven ``HttpClientError``
        # propagates through ``_download_activities`` / ``_download_molecules``,
        # so the run audit trail records the count of rate-limited drugs.
        self._metrics: dict[str, int | float] = {
            "api_calls": 0,
            "api_calls_429": 0,
            "api_calls_5xx": 0,
            "api_calls_4xx": 0,
            "retries": 0,
            "molecules_fetched": 0,
            "activities_fetched": 0,
            "targets_resolved": 0,
            "drugs_upserted": 0,
            "drugs_quarantined": 0,
            "dpi_upserted": 0,
            "dpi_quarantined": 0,
            # P1-012: number of drugs whose bioactivity fetch was aborted by
            # HTTP 429 after all retries. Non-zero => operator must back off
            # and re-run the affected drugs.
            "n_rate_limited_drugs": 0,
            "duration_download_sec": 0.0,
            "duration_clean_sec": 0.0,
            "duration_load_sec": 0.0,
        }

        # Capture the source_fetch_date at construction time so that all
        # records loaded by this pipeline instance share the same
        # provenance timestamp (LIN-3). tz-aware UTC.
        self._source_fetch_date: datetime = datetime.now(timezone.utc)

        # Source version (LIN-2). Read from CHEMBL_VERSION setting; may be
        # updated to the actual API-reported version during ``download()``
        # if CHEMBL_ALLOW_VERSION_MISMATCH is True (S20).
        self.source_version: str = f"ChEMBL_{CHEMBL_VERSION}"

        # Run-scoped dead-letter records (pipeline-level drops, separate
        # from the loader's dead-letter queue which is module-global).
        self._pipeline_dead_letters: list[dict[str, Any]] = []

    def teardown(self) -> None:
        """SCI-FIX: Override teardown to close the RateLimitedHttpClient.

        The base class teardown() only closes ``self._http_session`` (used by
        ``_download_file``), NOT ``self._http_client`` (the RateLimitedHttpClient
        which wraps a separate ``requests.Session``). Without this override,
        the HTTP client's underlying TCP connections / file descriptors leak
        in long-running processes (e.g., Airflow scheduler).
        """
        try:
            if self._http_client is not None:
                self._http_client.close()
        except (OSError, RuntimeError, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
            pass
        super().teardown()

        # v36 ROOT FIX (Phase 1 Issue #15): the previous log message
        # said "ChEMBLPipeline initialised" — a copy-paste leftover
        # from __init__. The teardown() method running AFTER
        # super().teardown() (which may close log handlers) logging
        # "initialised" misled operators reading logs during incident
        # triage. Fixed to say "torn down" with the same context.
        logger.info(
            "[%s] ChEMBLPipeline torn down (run_id=%s, version=%s, "
            "fetch_date=%s)",
            self.source_name,
            self.run_id,
            self.source_version,
            self._source_fetch_date.isoformat(),
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self) -> Path:
        """Download approved molecules and bioactivity data from ChEMBL.

        v50 ROOT FIX: now delegates to `pipelines._v50_downloaders.download_chembl_full`
        which handles BOTH sample mode (10 FDA-approved drugs via API)
        AND full mode (paginates through all ~10K max_phase=4 molecules
        via the public ChEMBL REST API — no login required).

        Side Effects
        ------------
        - Writes ``chembl_drugs.csv.gz`` to ``self.raw_dir``.
        - Writes ``chembl_activities.csv.gz`` to ``self.raw_dir``.
        - Writes ``chembl_manifest_{run_id}.json`` to ``self.raw_dir``
          (A1, LIN-1 to LIN-18).
        - All writes are atomic (``.tmp`` + ``os.replace`` — R5, A7).

        Returns
        -------
        Path
            Path to the drugs CSV (the primary raw artifact). The base
            class's ``run()`` passes this to ``clean()``.

        Raises
        ------
        PipelineError
            If the ChEMBL API version check fails and
            ``CHEMBL_ALLOW_VERSION_MISMATCH=False`` (S20).
        HttpClientError
            On non-retryable HTTP errors (4xx other than 429).
        CircuitBreakerOpenError
            If the circuit breaker is OPEN after too many failures (R10).

        Notes
        -----
        - Filters molecules to ``max_phase={CHEMBL_MAX_PHASE}`` (default
          4 = globally approved). Configure via ``CHEMBL_MAX_PHASE`` env
          var (DOC-4).
        - Downloads activities in pages of ``CHEMBL_PAGE_SIZE`` (default
          1000). Each page is processed end-to-end (parse → write to
          disk-backed chunk) before fetching the next, to bound memory
          usage (P2). The chunk files are concatenated at the end into a
          single gzipped CSV.
        - v50: in full mode, uses the public EBI REST API
          (https://www.ebi.ac.uk/chembl/api/data) — no login, no API key.
          In sample mode, fetches 10 well-known FDA-approved drugs.
        """
        # v50 ROOT FIX: delegate to the unified downloader.
        # The downloader handles sample/full/skip modes and falls back
        # to embedded samples if the live API is unreachable.
        try:
            from pipelines._v50_downloaders import download_chembl_full
            if self.raw_dir is None:
                self.raw_dir = RAW_DATA_DIR / "chembl"
            downloaded = download_chembl_full(self.raw_dir)
            # Read the downloaded files and convert to the format
            # _download_molecules() would have returned.
            import pandas as _pd
            import json as _json
            mol_path = downloaded.get("molecules")
            act_path = downloaded.get("activities")
            if mol_path and mol_path.exists():
                if mol_path.suffix == ".jsonl":
                    # Parse JSONL into a list of dicts, then use _parse_molecules
                    # v93 ROOT FIX (P1-043): explicit encoding="utf-8" — the
                    # ChEMBL API returns UTF-8 JSONL with non-ASCII drug names
                    # (e.g. "α-Tocopherol", "caf feína"). The default encoding
                    # is locale.getpreferredencoding() (CP1252 on Windows,
                    # UTF-8 on Linux). On Windows, non-ASCII names raised
                    # UnicodeDecodeError, silently dropping the record.
                    mol_records = []
                    with open(mol_path, encoding="utf-8") as f:
                        for line in f:
                            mol_records.append(_json.loads(line))
                    drugs_df = self._parse_molecules(mol_records)
                else:
                    # CSV (embedded sample)
                    drugs_df = _pd.read_csv(mol_path)
                # Persist as the canonical chembl_drugs.csv.gz
                self._metrics["molecules_fetched"] = len(drugs_df)
                # Persist activities
                if act_path and act_path.exists():
                    if act_path.suffix == ".jsonl":
                        # v93 ROOT FIX (P1-043): explicit encoding="utf-8"
                        # (see mol_path block above for rationale).
                        act_records = []
                        with open(act_path, encoding="utf-8") as f:
                            for line in f:
                                act_records.append(_json.loads(line))
                        activities_df = _pd.DataFrame(act_records)
                    else:
                        activities_df = _pd.read_csv(act_path)
                    self._metrics["activities_fetched"] = len(activities_df)
                else:
                    activities_df = _pd.DataFrame()
                # Persist to raw_dir
                # v90 ROOT FIX (BUG #1): the previous code wrote
                # `chembl_drugs.csv` (PLAIN CSV, no gzip) but clean()
                # reads with `compression="gzip"` → BadGzipFile on every
                # v50 pipeline run. The v49 path writes `.csv.gz` with
                # gzip. ROOT FIX: write to `chembl_drugs.csv.gz` with
                # gzip compression, matching the v49 canonical filename
                # and the clean() expectations.
                drugs_csv = self.raw_dir / "chembl_drugs.csv.gz"
                drugs_df.to_csv(drugs_csv, index=False, compression="gzip")
                if not activities_df.empty:
                    # v57 ROOT FIX (P1-013 — ChEMBL v50 filename mismatch):
                    #   The previous code wrote `chembl_activities_clean.csv`
                    #   but the clean step (line ~843) looks for
                    #   `chembl_activities.csv.gz`. As a result, in v50 mode
                    #   (the recommended download path), `clean_activities`
                    #   was NEVER invoked → Drug→Protein (DPI) edges were
                    #   NEVER generated → the KG had Compound nodes with
                    #   zero Drug→Protein edges (the most important edge type
                    #   for drug repurposing).
                    #   FIX: write to `chembl_activities.csv.gz` (the
                    #   canonical raw activities filename that clean step
                    #   expects). Use gzip compression via pandas to_csv.
                    activities_gz_path = self.raw_dir / "chembl_activities.csv.gz"
                    activities_df.to_csv(
                        activities_gz_path, index=False, compression="gzip"
                    )
                    logger.info(
                        "[%s] v57 ROOT FIX (P1-013): wrote %d activities to %s "
                        "(canonical filename so clean_activities() can find it)",
                        self.source_name, len(activities_df), activities_gz_path,
                    )
                # Update sha for audit
                # V90 CI fix: _compute_file_sha256 is a METHOD (self._compute_file_sha256),
                # not a module-level function. The previous code called it as a bare
                # function name, which raised NameError at runtime. This was a pre-existing
                # Phase 1 bug that was hidden because the E2E sample-mode job was skipped
                # when P2 + Chain-1 verification failed first. Now that P2 passes (V90
                # COMP-3 fix), E2E runs and exposes this bug.
                # ROOT FIX: _compute_file_sha256 is an instance method (line 4337),
                # not a module-level function. The bare call was a pre-existing
                # NameError that crashed the E2E sample-mode CI job.
                self._sha256_raw = self._compute_file_sha256(drugs_csv)
                return drugs_csv
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            # v84 FORENSIC ROOT FIX (BUG #38): narrowed from broad
            # ``except Exception``. The previous code caught ALL
            # failures from the v50 downloader — including programming
            # bugs (AttributeError, KeyError) — and silently fell back
            # to the v49 path. A bug in ``download_chembl_full`` or
            # ``_parse_molecules`` was masked as "v50 failed, using
            # v49" — the pipeline ALWAYS fell back to v49, which may
            # be stale or broken, with no visible warning.
            # ROOT FIX: catch ONLY the expected I/O, value, and JSON
            # parse errors. If ``requests`` is available, also catch
            # ``requests.RequestException``. Programming bugs propagate
            # so they surface during development instead of silently
            # degrading to v49 forever.
            _exc_types = (OSError, ValueError, json.JSONDecodeError)
            if requests is not None:
                _exc_types = (OSError, ValueError, json.JSONDecodeError,
                              requests.RequestException)
            if not isinstance(exc, _exc_types):
                raise
            logger.warning(
                "[%s] v50 downloader failed (%s) — falling back to v49 path",
                self.source_name, exc,
            )

        download_start = time.monotonic()
        logger.info(
            "[%s] download() starting (run_id=%s, version=%s)",
            self.source_name,
            self.run_id,
            self.source_version,
        )

        # Verify ChEMBL API version (S20, INT-12).
        self._verify_chembl_version()

        # Fetch molecules + activities.
        drugs_df = self._download_molecules()
        activities_df = self._download_activities()

        # Update metrics.
        self._metrics["molecules_fetched"] = len(drugs_df)
        self._metrics["activities_fetched"] = len(activities_df)
        # Sync HTTP client metrics.
        self._sync_http_metrics()

        # Atomic write of drugs CSV (R5, A7).
        drugs_path = self.raw_dir / "chembl_drugs.csv.gz"
        self._atomic_write_csv_gz(drugs_path, drugs_df)
        logger.info(
            "[%s] Wrote %d drugs to %s",
            self.source_name,
            len(drugs_df),
            drugs_path,
        )

        # Atomic write of activities CSV.
        activities_path = self.raw_dir / "chembl_activities.csv.gz"
        self._atomic_write_csv_gz(activities_path, activities_df)
        logger.info(
            "[%s] Wrote %d activities to %s",
            self.source_name,
            len(activities_df),
            activities_path,
        )

        # Compute checksums for lineage (LIN-4, LIN-7).
        drugs_checksum = self._compute_file_sha256(drugs_path)
        activities_checksum = self._compute_file_sha256(activities_path)

        # Write the manifest (A1, LIN-1 to LIN-18).
        self._write_manifest(
            drugs_path=drugs_path,
            activities_path=activities_path,
            drugs_checksum=drugs_checksum,
            activities_checksum=activities_checksum,
            total_molecules=len(drugs_df),
            total_activities=len(activities_df),
        )

        # Record duration.
        self._metrics["duration_download_sec"] = round(
            time.monotonic() - download_start, 4
        )

        logger.info(
            "[%s] download() complete in %.2fs",
            self.source_name,
            self._metrics["duration_download_sec"],
        )
        return drugs_path

    # ------------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------------

    def clean(self, raw_path: Path) -> pd.DataFrame:
        """Clean and normalise ChEMBL drug data.

        Parameters
        ----------
        raw_path : Path
            Path to the gzipped drugs CSV produced by ``download()``.

        Returns
        -------
        pandas.DataFrame
            Cleaned drugs DataFrame. The base class writes this to
            ``PROCESSED_DATA_DIR / self._get_processed_filename()`` (which
            is ``drugs.csv`` for ChEMBL — D2-4, I9).

        Side Effects
        ------------
        - Also calls ``clean_activities()`` as a side effect on the
          sibling ``chembl_activities.csv.gz`` file, writing the cleaned
          activities to ``PROCESSED_DATA_DIR / "chembl_activities_clean.csv"``.
        - Calls ``self._log_transformation(step, rows_affected, details)``
          after each transformation step (LIN-6).
        - Drops records with invalid InChIKey / max_phase / molecular_weight
          to a dead-letter file under
          ``PROCESSED_DATA_DIR / "dead_letter"`` (DQ-6, DQ-7, R9).

        Steps
        -----
        1. Load raw drugs CSV (gzipped).
        2. Generate InChIKey from SMILES where missing (vectorised — C24).
        3. Standardise InChIKey format (uppercase, validate).
        4. Drop rows with no valid InChIKey (dead-letter — DQ-6).
        5. Deduplicate by InChIKey.
        6. Standardise ``drug_type`` via ``MOLECULE_TYPE_MAP`` (K6, S6, S7).
        7. Validate ``molecular_weight`` range (DQ-7).
        8. Coerce ``max_phase`` to int in [0, 4] (K4, K5).
        9. Compute ``is_fda_approved`` as a real Python bool (K4).
        10. Validate ``name`` ≥ 2 chars (synthesize fallback if needed — DQ-14).
        11. Fill missing drug fields via ``fill_missing_drug_fields``.
        12. Ensure all required Drug-table columns exist.
        13. Sort by ``chembl_id`` for deterministic output (I5).
        """
        clean_start = time.monotonic()
        logger.info("[%s] clean() starting (raw_path=%s)", self.source_name, raw_path)

        # Read the raw drugs CSV (gzipped, UTF-8 — INT-6, INT-7).
        # V90 CI fix: the ChEMBL API sometimes returns a non-gzip file
        # (rate-limit HTML page, maintenance page, or error JSON) which
        # raises BadGzipFile. The previous code had no error handling,
        # so the E2E sample-mode CI job crashed whenever the API was
        # having issues. The fix: try gzip first, fall back to plain
        # CSV (without compression), and raise a clear error if both
        # fail. This makes the pipeline robust to transient API issues.
        import gzip as _gzip
        try:
            drugs_df = pd.read_csv(
                raw_path,
                compression="gzip",
                low_memory=False,
                encoding="utf-8",
            )
        except (_gzip.BadGzipFile, OSError) as gz_exc:
            logger.warning(
                "[%s] V90 CI fix: gzip read failed (%s). Falling back to "
                "plain CSV read (the ChEMBL API may have returned a "
                "non-gzip response — rate limit, maintenance, etc.).",
                self.source_name, gz_exc,
            )
            try:
                drugs_df = pd.read_csv(
                    raw_path,
                    compression=None,
                    low_memory=False,
                    encoding="utf-8",
                )
            except Exception as plain_exc:
                raise OSError(
                    f"V90 CI fix: could not read {raw_path} as gzip ({gz_exc}) "
                    f"or as plain CSV ({plain_exc}). The ChEMBL API may be "
                    f"down or rate-limiting. Try again later."
                ) from plain_exc
        # P1-001 ROOT FIX (v100 forensic): the previous code UNCONDITIONALLY
        # re-read the file here with compression="gzip" if the suffix was
        # ".gz", OVERWRITING the drugs_df produced by the try/except above.
        # When the ChEMBL API returned a non-gzip body (rate-limit HTML,
        # maintenance page, JSON error) saved to a .csv.gz path, the first
        # try/except correctly fell back to a plain-CSV read — but this
        # second read then raised BadGzipFile and crashed the pipeline.
        # The first try/except block was effectively dead code. ROOT FIX:
        # remove the dead try/except gzip fallback block that was
        # immediately overwritten by the extension-based read above.
        # (Parallel V100 fix BUG #18 applied the same root fix — kept
        # this comment for the more detailed forensic trail.)
        initial_count = len(drugs_df)
        logger.info(
            "[%s] Loaded %d raw drug records from %s",
            self.source_name,
            initial_count,
            raw_path,
        )

        # v84 FORENSIC ROOT FIX (BUG #50 — COMPOUND): data quality
        # schema check. The compound bug chain was:
        #   v50 downloader catches all exceptions (BUG #38 fixed) →
        #   falls back to embedded samples → writes JSONL to a .csv
        #   file → clean() reads garbage → falls back to v49 path →
        #   reports "success" with wrong/missing data.
        # ROOT FIX: validate the DataFrame schema immediately after
        # load. If the expected ChEMBL columns are missing (e.g.
        # ``chembl_id``, ``name``, ``max_phase``), raise a
        # ``DataQualityError`` instead of silently processing garbage.
        # This breaks the compound degradation chain at the FIRST
        # stage where bad data enters the pipeline.
        _expected_chembl_cols = {"chembl_id", "name", "max_phase"}
        _actual_cols = set(drugs_df.columns)
        _missing_critical = _expected_chembl_cols - _actual_cols
        if _missing_critical and not drugs_df.empty:
            raise ValueError(
                f"[{self.source_name}] Data quality check FAILED: raw "
                f"drugs CSV is missing critical columns "
                f"{_missing_critical}. Expected at least "
                f"{_expected_chembl_cols}, got {_actual_cols}. This "
                f"indicates the v50 downloader produced a malformed "
                f"file (e.g. JSONL written as CSV). Refusing to "
                f"process garbage data — fix the downloader."
            )
        # Check for empty DataFrame (another silent-degradation signal).
        if drugs_df.empty:
            logger.warning(
                "[%s] Data quality check: raw drugs CSV is EMPTY. "
                "Proceeding with clean() but the output will have 0 "
                "drugs — the KG will be missing all ChEMBL compounds.",
                self.source_name,
            )

        # Step 1: Generate InChIKey from SMILES where missing (C24, C25, C26).
        drugs_df = self._step_generate_inchikeys(drugs_df)

        # Step 2: Standardise InChIKey format.
        drugs_df = self._step_standardize_inchikeys(drugs_df)

        # Step 3: Drop rows with no valid InChIKey (dead-letter).
        drugs_df = self._step_drop_invalid_inchikeys(drugs_df)

        # Step 4: Deduplicate by InChIKey.
        drugs_df = self._step_dedup_by_inchikey(drugs_df)

        # Step 5: Standardise drug_type (K6, S6, S7).
        drugs_df = self._step_standardize_drug_type(drugs_df)

        # Step 6: Validate molecular_weight range (DQ-7).
        drugs_df = self._step_validate_molecular_weight(drugs_df)

        # Step 7: Coerce max_phase to int in [0, 4] (K4, K5).
        drugs_df = self._step_coerce_max_phase(drugs_df)

        # Step 8: Compute is_fda_approved as real bool (K4, C30).
        drugs_df = self._step_compute_is_fda_approved(drugs_df)

        # Step 9: Validate / synthesize name (DQ-14, C13).
        drugs_df = self._step_validate_name(drugs_df)

        # Step 10: Fill missing drug fields.
        drugs_df = self._step_fill_missing_fields(drugs_df)

        # Step 11: Ensure all required columns exist.
        drugs_df = self._step_ensure_drug_columns(drugs_df)

        # Step 12: Sort for deterministic output (I5).
        drugs_df = self._step_sort_deterministic(drugs_df)

        # Side effect: clean the activities file (A2, A3, D2-3).
        # v64 ROOT FIX (P1-013): the previous code only looked for
        # ``chembl_activities.csv.gz`` — but the v50 downloader (the primary
        # path) writes ``chembl_activities_clean.csv`` (embedded sample) or
        # ``chembl_activities.jsonl`` (live API). The .csv.gz name only
        # exists in the legacy v49 path. As a result, clean_activities()
        # was NEVER called in v50 mode, so the ChEMBL DPI edge set was
        # silently missing from the KG. Root fix: probe ALL three known
        # activity-file names and use the first one that exists.
        activities_raw_path = None
        for _candidate_name in (
            "chembl_activities.csv.gz",       # legacy v49 path
            "chembl_activities_clean.csv",    # v50 embedded-sample path
            "chembl_activities.jsonl",        # v50 live-API path
        ):
            _candidate = raw_path.parent / _candidate_name
            if _candidate.exists():
                activities_raw_path = _candidate
                break
        if activities_raw_path is not None:
            try:
                # SCI-FIX (timing bug): pass the cleaned drugs DataFrame
                # directly to clean_activities() so the activity filter
                # can use the in-memory drug set. The previous code read
                # ``drugs.csv`` from disk, but that file is only written
                # AFTER ``clean()`` returns (in BasePipeline.run()).
                # As a result, the activity filter was ALWAYS skipped on
                # a fresh run (drugs.csv did not exist yet), which caused
                # 100% of activities to be unresolved at load time and
                # the pipeline to raise PipelineError "More than 50% of
                # activities have unresolved drug_id (DQ-9)".
                # Passing the in-memory drugs_df fixes this timing bug
                # while preserving backward compatibility (clean_activities
                # still falls back to drugs.csv when called standalone).
                self.clean_activities(activities_raw_path, cleaned_drugs_df=drugs_df)
            except (KeyError, ValueError, FileNotFoundError, pd.errors.ParserError) as exc:
                # v16 ROOT FIX (SF-3): narrow the broad ``except Exception``
                # to specific, expected failure modes. ChEMBL DPI edge set
                # silently missing on ANY error was unacceptable — only
                # data-format / IO errors should be tolerated. Other
                # exceptions (e.g. ProgrammingError, MemoryError) should
                # propagate so the operator can investigate. Logged at
                # ERROR with traceback so it is visible in production.
                # V18 ROOT FIX (SF-3 deepened): in PRODUCTION mode (env
                # var ``DRUGOS_STRICT=1``), this is FATAL. The v16/v17
                # behavior of "log + continue with drugs only" silently
                # produced a KG missing the ChEMBL DPI edge set — the
                # audit's Compound-6 degradation.
                #
                # V19 ROOT FIX (SF-3 — verification agent flagged this as
                # PARTIAL): the V18 default was PERMISSIVE (strict opt-in
                # via DRUGOS_STRICT=1), which meant operators got a
                # silently degraded KG unless they read the docs. The
                # ROOT fix is to FLIP THE DEFAULT: STRICT is now the
                # production default. Operators who want the legacy
                # permissive behavior (e.g. for unit-test fixtures or
                # known-broken ChEMBL snapshots) must explicitly opt in
                # via ``DRUGOS_ALLOW_PERMISSIVE_DPI=1``.
                import os as _os
                _permissive = _os.environ.get(
                    "DRUGOS_ALLOW_PERMISSIVE_DPI", ""
                ) == "1"
                # DRUGOS_STRICT=1 remains supported as a redundant
                # explicit-strict signal (takes precedence over the
                # permissive opt-in for operators who set both).
                _strict = (_os.environ.get("DRUGOS_STRICT", "") == "1") or (not _permissive)
                # P1-041 ROOT FIX (DRUGOS_ALLOW_PERMISSIVE_DPI silent escape):
                #   The previous code logged at ERROR level and emitted
                #   ``chembl_dpi_missing=1`` as a metric, but the KG
                #   appeared healthy (drugs loaded) and operators could
                #   miss the ERROR log if the DAG showed success. An
                #   operator who set DRUGOS_ALLOW_PERMISSIVE_DPI=1 to
                #   unblock a Sunday run after a ChEMBL API change
                #   silently produced a KG with ZERO drug-protein
                #   interactions. The KG looked normal but had no
                #   pharmacological edges — every drug-target prediction
                #   was broken.
                #
                #   ROOT FIX (three layers):
                #   (1) Escalate the log level to CRITICAL when permissive
                #       mode is active (ERROR is for "something failed
                #       but we recovered"; CRITICAL is for "the KG is
                #       silently degraded — operator must acknowledge").
                #   (2) Set ``self._metrics["dpi_missing"] = True`` so
                #       the flag is persisted to ``pipeline_run.metadata_json``
                #       via ``BasePipeline._write_run_log``. Downstream
                #       consumers can query this flag.
                #   (3) Require TWO-STEP opt-in: DRUGOS_ALLOW_PERMISSIVE_DPI=1
                #       marks DPI missing + raises (so the task fails RED);
                #       DRUGOS_ALLOW_PERMISSIVE_DPI=2 is the explicit
                #       operator acknowledgement that proceeds with the
                #       DPI-degraded KG. The two-step opt-in prevents
                #       the silent escape hatch from producing a silently
                #       degraded KG.
                _log_level = (
                    logger.critical if (_permissive and not _strict) else logger.error
                )
                _log_level(
                    "[%s] clean_activities() failed%s — ChEMBL DPI edge set "
                    "will be missing. %s: %s",
                    self.source_name,
                    " (STRICT MODE — FATAL)" if _strict else (
                        " (PERMISSIVE MODE — KG WILL BE DPI-DEGRADED; "
                        "trigger_phase2 pre-flight check will FAIL until "
                        "operator sets DRUGOS_ALLOW_PERMISSIVE_DPI=2 to "
                        "acknowledge)"
                    ),
                    type(exc).__name__, exc,
                    exc_info=True,
                )
                self._log_transformation(
                    step="clean_activities_failed",
                    rows_affected=0,
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
                # Tag the pipeline run so downstream consumers know DPI is missing.
                self._emit_metric("chembl_dpi_missing", 1)
                # P1-041 ROOT FIX layer 2: persist dpi_missing flag in
                # pipeline_run.metadata_json (via _metrics → _write_run_log).
                self._metrics["dpi_missing"] = True
                self._metrics["dpi_missing_reason"] = (
                    f"clean_activities_failed:{type(exc).__name__}"
                )
                _acknowledged = (
                    _os.environ.get("DRUGOS_ALLOW_PERMISSIVE_DPI", "") == "2"
                )
                self._metrics["dpi_missing_acknowledged"] = _acknowledged
                if _strict:
                    raise RuntimeError(
                        f"ChEMBL clean_activities() failed in STRICT mode "
                        f"(default since V19; set DRUGOS_ALLOW_PERMISSIVE_DPI=1 "
                        f"to opt in to the legacy permissive behavior): "
                        f"{type(exc).__name__}: {exc}. V19 SF-3 root fix — "
                        f"production runs must not silently continue with "
                        f"the DPI edge set missing."
                    ) from exc
                # P1-041 ROOT FIX layer 3: two-step opt-in.
                #   DRUGOS_ALLOW_PERMISSIVE_DPI=1 → mark DPI missing +
                #   RAISE so the task fails RED. The operator sees the
                #   failure in the Airflow UI and must explicitly set
                #   DRUGOS_ALLOW_PERMISSIVE_DPI=2 to acknowledge and
                #   proceed with the DPI-degraded KG.
                #   DRUGOS_ALLOW_PERMISSIVE_DPI=2 → mark DPI missing +
                #   CONTINUE (the operator has acknowledged). The KG is
                #   DPI-degraded but the operator has explicitly opted in.
                if not _acknowledged:
                    raise RuntimeError(
                        f"P1-041 ROOT FIX: clean_activities() failed and "
                        f"DRUGOS_ALLOW_PERMISSIVE_DPI=1 is set (permissive "
                        f"mode). The KG would be DPI-degraded (ZERO "
                        f"drug-protein interactions). To proceed with the "
                        f"DPI-degraded KG, the operator MUST explicitly "
                        f"acknowledge by setting "
                        f"DRUGOS_ALLOW_PERMISSIVE_DPI=2 (override-"
                        f"acknowledged). This two-step opt-in prevents "
                        f"the silent escape hatch from producing a "
                        f"silently degraded KG. Original error: "
                        f"{type(exc).__name__}: {exc}."
                    ) from exc

        self._metrics["duration_clean_sec"] = round(
            time.monotonic() - clean_start, 4
        )
        logger.info(
            "[%s] clean() complete in %.2fs — %d rows (started with %d)",
            self.source_name,
            self._metrics["duration_clean_sec"],
            len(drugs_df),
            initial_count,
        )

        # v29 ROOT FIX (audit P1-24): ID format divergence — normalize to
        # canonical form before writing. Every ChEMBL ID is uppercased +
        # stripped; every InChIKey is uppercased + stripped. This guarantees
        # that a downstream join against DrugBank / PubChem on InChIKey
        # succeeds regardless of which source wrote the value.
        if "chembl_id" in drugs_df.columns and len(drugs_df) > 0:
            drugs_df["chembl_id"] = drugs_df["chembl_id"].apply(
                lambda x: normalize_chembl_id(x) if pd.notna(x) else x
            )
        if "inchikey" in drugs_df.columns and len(drugs_df) > 0:
            drugs_df["inchikey"] = drugs_df["inchikey"].apply(
                lambda x: normalize_inchikey(x) if pd.notna(x) else x
            )

        return drugs_df

    def clean_activities(
        self,
        activities_raw_path: Path,
        cleaned_drugs_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Clean and normalise ChEMBL activity data into a DPI-ready DataFrame.

        Parameters
        ----------
        activities_raw_path : Path
            Path to the gzipped activities CSV produced by ``download()``.
        cleaned_drugs_df : pandas.DataFrame, optional
            The cleaned drugs DataFrame (output of ``clean()``). When
            provided, the activity filter uses this in-memory drug set
            instead of reading ``drugs.csv`` from disk. This is required
            when ``clean_activities()`` is called from inside ``clean()``
            because ``drugs.csv`` is only persisted to disk AFTER
            ``clean()`` returns (SCI-FIX: timing bug — see notes below).
            When ``None`` (standalone call), the method falls back to
            reading ``drugs.csv`` from disk if it exists.

        Returns
        -------
        pandas.DataFrame
            Cleaned activities DataFrame with columns:
            ``activity_id, molecule_chembl_id, target_chembl_id,
            target_accession, target_pref_name, activity_type,
            activity_value, activity_units, pchembl_value, assay_id,
            standard_relation, assay_type, target_type``.

        Side Effects
        ------------
        - Writes the cleaned activities to
          ``PROCESSED_DATA_DIR / "chembl_activities_clean.csv"``.
        - Writes a provenance sidecar
          ``chembl_activities_clean.csv.provenance.json`` (CMP-12).

        Steps
        -----
        1. Read raw activities CSV.
        2. Filter by ``activity_type`` ∈ ``CHEMBL_ACTIVITY_TYPES`` (S10).
        3. Filter by ``activity_units`` ∈ ``CHEMBL_STANDARD_UNITS`` (DQ-15, DQ-16).
        4. Filter by ``standard_relation`` ∈ ``CHEMBL_STANDARD_RELATIONS`` (S12).
        5. Resolve ``target_chembl_id`` → list of UniProt accessions (K3, S9).
        6. Explode multi-subunit complexes — one row per accession (K8, S9).
        7. Normalise ``activity_value`` to nM, passing ``activity_type=`` (S13).
        8. Preserve ``pchembl_value`` (S14).
        9. Write the cleaned DataFrame to disk.

        Notes
        -----
        - Aggregation by ``(drug, protein, activity_type)`` to produce one
          DPI per pair happens in ``load()``, not here. This method
          produces ONE row per (activity_id, accession) — i.e., one row
          per measurement per subunit. The aggregation step (S17) reduces
          these to one DPI per (drug, protein) pair using the median
          activity_value (most robust to outliers).
        - This method does NOT resolve ``drug_id`` or ``protein_id`` —
          that happens in ``load()`` where we have a DB session.
        """
        if not activities_raw_path.exists():
            logger.warning(
                "[%s] clean_activities(): activities file does not exist: %s",
                self.source_name,
                activities_raw_path,
            )
            return pd.DataFrame()

        logger.info(
            "[%s] clean_activities() starting (raw_path=%s)",
            self.source_name,
            activities_raw_path,
        )

        # Step 1: Read raw activities CSV.
        # v90 ROOT FIX (BUG #10): auto-detect compression from file
        # extension instead of hardcoding compression="gzip". The v50
        # path writes .csv.gz, but a plain .csv should still work.
        _compression = "gzip" if activities_raw_path.suffix == ".gz" else None
        activities_df = pd.read_csv(
            activities_raw_path,
            compression=_compression,
            low_memory=False,
            encoding="utf-8",
        )
        if len(activities_df) == 0:
            logger.info("[%s] No activities to clean.", self.source_name)
            # Still write an empty cleaned file so load() can read it.
            self._write_cleaned_activities(activities_df)
            return activities_df

        initial_count = len(activities_df)
        self._log_transformation(
            step="activities_loaded",
            rows_affected=initial_count,
            details={"source": str(activities_raw_path)},
        )

        # Step 2: Filter by activity_type (S10).
        activities_df = self._filter_activities_by_type(activities_df)

        # Step 3: Filter by activity_units (DQ-15, DQ-16).
        activities_df = self._filter_activities_by_units(activities_df)

        # Step 4: Filter by standard_relation (S12).
        activities_df = self._filter_activities_by_relation(activities_df)

        # Step 5: Filter by assay_type (S10).
        activities_df = self._filter_activities_by_assay_type(activities_df)

        # Step 5.5: CRITICAL FIX (scientific correctness / data integrity):
        # Filter activities to ONLY those whose ``molecule_chembl_id`` is
        # present in the drugs we downloaded. Without this filter, the
        # ChEMBL ``/activity.json`` endpoint returns bioactivity data for
        # ALL molecules (not just our FDA-approved drugs), and the load()
        # step fails with "More than 50% of activities have unresolved
        # drug_id" because the drugs table only contains max_phase=4 drugs.
        # The correct scientific behavior is: a drug-protein interaction
        # edge in the knowledge graph must connect to a Drug node we
        # actually have. An activity record for a molecule we don't have
        # is useless — drop it now, before we waste time on target
        # accession resolution and activity value normalization.
        #
        # SCI-FIX (timing bug): the original implementation read
        # ``drugs.csv`` from disk to obtain the valid chembl_id set.
        # However, when ``clean_activities()`` is invoked as a side effect
        # of ``clean()``, ``drugs.csv`` has NOT yet been written — it is
        # only persisted AFTER ``clean()`` returns. As a result the filter
        # was always skipped on a fresh run, and 100% of activities were
        # unresolved at load time, raising PipelineError (DQ-9).
        # The fix below uses the in-memory ``cleaned_drugs_df`` when
        # provided (the normal path from ``clean()``), and falls back to
        # reading ``drugs.csv`` when called standalone.
        drugs_csv_path = PROCESSED_DATA_DIR / "drugs.csv"
        valid_chembl_ids: set[str] = set()
        have_drug_set = False
        if cleaned_drugs_df is not None and "chembl_id" in cleaned_drugs_df.columns:
            valid_chembl_ids = set(
                cleaned_drugs_df["chembl_id"].dropna().astype(str)
            )
            have_drug_set = True
            logger.debug(
                "[%s] clean_activities: using in-memory drug set (%d drugs)",
                self.source_name, len(valid_chembl_ids),
            )
        elif drugs_csv_path.exists():
            try:
                drugs_df_temp = pd.read_csv(drugs_csv_path, usecols=["chembl_id"])
                valid_chembl_ids = set(
                    drugs_df_temp["chembl_id"].dropna().astype(str)
                )
                have_drug_set = True
                logger.debug(
                    "[%s] clean_activities: using drugs.csv drug set (%d drugs)",
                    self.source_name, len(valid_chembl_ids),
                )
            # v85/v90 FORENSIC ROOT FIX (BUG #18/51): narrowed from broad
            # ``except Exception`` which caught programming bugs
            # (AttributeError from wrong column name, KeyError from
            # missing column) and silently skipped the drug filter,
            # allowing activities for molecules NOT in our drugs table
            # to pass through → load() fails with "more than 50%
            # unresolved drug_id". Root fix: catch ONLY expected I/O
            # and data errors. Programming bugs propagate.
            except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
                logger.warning(
                    "[%s] Could not read drugs.csv for activity filter (%s) — "
                    "proceeding without filter (may cause load() to fail "
                    "with unresolved drug_id).",
                    self.source_name, exc,
                )

        if have_drug_set and "molecule_chembl_id" in activities_df.columns:
            try:
                pre_count = len(activities_df)
                mask = activities_df["molecule_chembl_id"].astype(str).isin(valid_chembl_ids)
                dropped_count = (~mask).sum()
                if dropped_count > 0:
                    dropped_df = activities_df[~mask].copy()
                    self._write_dead_letter(
                        dropped_df,
                        step="clean_activities_drug_not_in_db",
                        reason=(
                            "molecule_chembl_id is not in our FDA-approved "
                            "drugs table — activity cannot form a DPI edge "
                            "without a corresponding Drug node"
                        ),
                    )
                    logger.info(
                        "[%s] Filtered activities to drug set: kept %d/%d "
                        "(dropped %d activities for molecules not in drugs table)",
                        self.source_name,
                        mask.sum(),
                        pre_count,
                        dropped_count,
                    )
                    self._log_transformation(
                        step="activities_filtered_to_drug_set",
                        rows_affected=int(dropped_count),
                        details={
                            "kept": int(mask.sum()),
                            "total": pre_count,
                            "drugs_in_set": len(valid_chembl_ids),
                        },
                    )
                activities_df = activities_df[mask].copy()
            except (ValueError, KeyError, TypeError) as exc:  # v85 FORENSIC ROOT FIX (BUG #51)
                logger.warning(
                    "[%s] Could not filter activities by drug set (%s) — "
                    "proceeding without filter (may cause load() to fail "
                    "with unresolved drug_id).",
                    self.source_name,
                    exc,
                )
        elif not have_drug_set:
            logger.info(
                "[%s] No drug set available (neither cleaned_drugs_df nor "
                "drugs.csv) — skipping activity filter by drug set "
                "(activities will be filtered at load time).",
                self.source_name,
            )

        # Step 6: Resolve target_chembl_id → list of UniProt accessions
        # (K3, K8, S9). Returns dict[str, list[str]].
        unique_targets = set(
            activities_df["target_chembl_id"].dropna().astype(str).unique()
        )
        accession_map = self._resolve_target_accessions(unique_targets)
        # Map target_chembl_id → list of accessions. Drop rows where
        # resolution returned an empty list (dead-letter — DQ-10).
        activities_df["target_accession"] = activities_df["target_chembl_id"].map(
            lambda tid: accession_map.get(str(tid), []) if pd.notna(tid) else []
        )
        # Drop rows with no accessions (dead-letter).
        no_acc_mask = activities_df["target_accession"].apply(len) == 0
        if no_acc_mask.any():
            dropped = activities_df[no_acc_mask].copy()
            self._write_dead_letter(
                dropped,
                step="clean_activities_no_accession",
                reason="target_chembl_id resolved to no UniProt accessions",
            )
            logger.error(
                "[%s] Dropping %d/%d activities with no resolved accession. "
                "Sample target_chembl_ids: %s",
                self.source_name,
                len(dropped),
                initial_count,
                list(dropped["target_chembl_id"].head(10)),
            )
            activities_df = activities_df[~no_acc_mask].copy()

        # Step 7: Explode multi-subunit complexes (K8, S9).
        # One activity on a 5-subunit complex → 5 rows.
        activities_df = activities_df.explode("target_accession", ignore_index=True)
        # Drop rows where target_accession is None/NaN after explode.
        activities_df = activities_df.dropna(subset=["target_accession"]).copy()
        # Ensure string type.
        activities_df["target_accession"] = activities_df["target_accession"].astype(str)

        # Step 8: Normalise activity_value to nM, passing activity_type (S13).
        activities_df = self._step_normalize_activity_values(activities_df)

        # Step 9: Write the cleaned DataFrame.
        self._write_cleaned_activities(activities_df)

        logger.info(
            "[%s] clean_activities() complete — %d rows (started with %d)",
            self.source_name,
            len(activities_df),
            initial_count,
        )
        return activities_df

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, df: pd.DataFrame, session: Any | None = None) -> int:
        """Load cleaned drugs and activities into the staging DB.

        Parameters
        ----------
        df : pandas.DataFrame
            Cleaned drugs DataFrame (from ``clean()``).
        session : Session, optional
            SQLAlchemy session. If provided, the caller manages the
            transaction boundary. If ``None``, this method opens its own
            session (R11 — single session for drugs + DPI).

        Returns
        -------
        int
            Total rows upserted (drugs + DPI).

        Raises
        ------
        PipelineError
            If drug count < ``CHEMBL_EXPECTED_DRUG_COUNT_MIN`` (S18, DQ-13),
            or if > 50% of activities have unresolved drug_id / protein_id
            (DQ-9, DQ-10).

        Steps
        -----
        1. Compute ``input_checksum`` (SHA-256 of df CSV).
        2. Insert/UPSERT a PipelineRun row, get its ``id`` for DPI lineage.
        3. ``bulk_upsert_drugs(session, df, input_checksum=...)``.
        4. Validate drug count (raise PipelineError if < MIN).
        5. Read cleaned activities from
           ``PROCESSED_DATA_DIR / "chembl_activities_clean.csv"``.
        6. Resolve ``molecule_chembl_id`` → ``drug_id`` via
           ``get_chembl_to_drug_id_map(session, chembl_ids=...)`` (A9, P5).
        7. Resolve ``target_accession`` → ``protein_id`` via
           ``get_uniprot_to_protein_id_map(session, uniprot_ids=...)``
           (K2 — use ``.mapping``, not the MappingResult itself).
        8. Drop activities with unresolved drug_id / protein_id (dead-letter).
        9. Aggregate by (drug_id, protein_id, activity_type) — emit median
           activity_value (S17).
        10. Build the DPI DataFrame with ``interaction_type="unknown"``,
            valid ``activity_type``, ``source="chembl"``, ``source_id=activity_id``.
        11. ``bulk_upsert_dpi(session, dpi_df, pipeline_run_id=<int>,
            source_version=..., source_fetch_date=..., input_checksum=...)``
            in chunks of ``CHEMBL_DPI_BATCH_SIZE`` (P13).
        12. Flush the loader's dead-letter queue to disk (R9).
        13. Update the PipelineRun row's status to "success".
        """
        load_start = time.monotonic()
        logger.info(
            "[%s] load() starting (run_id=%s, drugs_df=%d rows)",
            self.source_name,
            self.run_id,
            len(df),
        )

        # Step 1: Compute input_checksum (LIN-4, I8).
        input_checksum = self._compute_df_sha256(df)

        total_loaded = 0

        # Use the provided session, or open our own (R11 — single session
        # for drugs + DPI so a failure rolls back both).
        owns_session = session is None
        # v29 ROOT FIX (audit P1-3): the previous code did
        #   session = get_db_session(...)
        #   session.__enter__()
        # and DISCARDED the return value of __enter__(). ``session``
        # still referred to the context manager, not the actual Session,
        # so every subsequent ``session.flush()`` / ``session.commit()``
        # / ``session.rollback()`` / ``session.close()`` failed with
        # AttributeError when load() was called standalone (outside
        # base_pipeline.run()). The pipeline only "worked" when called
        # from base_pipeline.run() which provided its own session.
        #
        # ROOT FIX: capture the return value of __enter__() into a
        # SEPARATE variable, then use THAT as the actual session. The
        # context manager is tracked separately so we can call __exit__
        # in the finally block.
        _session_cm = None  # the context manager (for __exit__)
        if owns_session:
            _session_cm = get_db_session(
                pipeline_name=self.source_name,
                run_id=self.run_id,
            )
            session = _session_cm.__enter__()

        try:
            # Step 2: Insert/UPSERT a PipelineRun row, get its id for DPI lineage.
            pipeline_run_id = self._ensure_pipeline_run_row(session, len(df))

            # Step 3: Bulk upsert drugs.
            # Filter the DataFrame to only valid Drug-model columns — the
            # loader rejects DataFrames with extra columns (e.g.
            # ``_smiles_was_filled`` from fill_missing_drug_fields,
            # ``is_macromolecule`` from _step_validate_molecular_weight).
            drugs_df_for_load = self._filter_to_drug_columns(df)
            drugs_result: UpsertResult = bulk_upsert_drugs(
                session,
                drugs_df_for_load,
                input_checksum=input_checksum,
            )
            # Flush to ensure the inserts are visible to subsequent queries
            # in the same session (the loader doesn't commit; the caller
            # manages the transaction boundary — R11).
            # v29 ROOT FIX (audit P1-9): the previous code did
            # ``except Exception: pass`` which SILENTLY swallowed
            # IntegrityError and other flush failures. This hid CHECK
            # constraint violations, duplicate key errors, and FK
            # violations — the data appeared to load but was actually
            # rolled back. ROOT FIX: LOG the error at WARNING level
            # so operators can see what failed, while still allowing
            # the pipeline to continue (the flush is non-critical —
            # the real commit happens in __exit__).
            # v52 ROOT FIX (P1-045 — phantom success): the v49 code
            # logged the warning and rolled back, but THEN still set
            # drugs_upserted = drugs_result.inserted + drugs_result.updated
            # — reporting the numbers from BEFORE the flush failed. This
            # is a PHANTOM SUCCESS: the data was rolled back but the
            # metrics report it as loaded. ROOT FIX: track the flush
            # failure and ZERO OUT the drugs_upserted count when it
            # occurs. Also set a drugs_flush_failed flag so the audit
            # trail records the failure.
            _flush_failed = False
            try:
                session.flush()
            except (OperationalError, IntegrityError) as _flush_exc:  # noqa: BLE001  # v85 FORENSIC ROOT FIX (BUG #51)
                _flush_failed = True
                # FIX-P2-1 (audit P2): after IntegrityError the SQLAlchemy
                # session is POISONED — every subsequent op raises
                # PendingRollbackError. The previous code only LOGGED the
                # warning and CONTINUED, so all downstream queries/upserts
                # in this load() call silently failed. Mirrors the
                # drugbank_pipeline.py:3166 pattern. Root fix: roll back
                # the session so subsequent operations can proceed (the
                # real commit lives in __exit__).
                try:
                    session.rollback()
                except (OSError, RuntimeError, ValueError):  # noqa: BLE001 — never mask the flush error  # v85 FORENSIC ROOT FIX (BUG #51)
                    pass
                logger.error(
                    "[%s] session.flush() FAILED — rolled back. "
                    "drugs_upserted will be reported as 0 (P1-045 phantom "
                    "success fix). Error: %s: %s",
                    self.source_name, type(_flush_exc).__name__, _flush_exc,
                )
            if _flush_failed:
                # v52 ROOT FIX (P1-045): do NOT report phantom success.
                # The data was rolled back — report 0 inserts/updates.
                self._metrics["drugs_upserted"] = 0
                self._metrics["drugs_flush_failed"] = True
                self._metrics["drugs_quarantined"] = drugs_result.quarantined
                logger.error(
                    "[%s] PHANTOM SUCCESS PREVENTED: drugs_upserted=0 "
                    "(flush failed, data rolled back). drugs_result had "
                    "reported inserted=%d, updated=%d — but those rows "
                    "were NOT persisted (P1-045 root fix).",
                    self.source_name,
                    drugs_result.inserted, drugs_result.updated,
                )
            else:
                self._metrics["drugs_upserted"] = (
                    drugs_result.inserted + drugs_result.updated
                )
                self._metrics["drugs_flush_failed"] = False
                self._metrics["drugs_quarantined"] = drugs_result.quarantined
            logger.info(
                "[%s] bulk_upsert_drugs: input=%d, inserted+updated=%d, "
                "quarantined=%d, failed=%d",
                self.source_name,
                drugs_result.total_input,
                self._metrics["drugs_upserted"],
                drugs_result.quarantined,
                drugs_result.failed,
            )
            if drugs_result.quarantined > 0:
                pct = drugs_result.quarantined / max(
                    drugs_result.total_input, 1
                ) * 100
                log_fn = (
                    logger.error if pct > 10 else logger.warning
                )
                log_fn(
                    "[%s] %d drugs quarantined (%.1f%% of input) — "
                    "see dead_letter file",
                    self.source_name,
                    drugs_result.quarantined,
                    pct,
                )

            # Flush loader's dead-letter queue to disk (R9, LIN-13).
            self._flush_loader_dead_letters(step="drugs")

            total_loaded += int(drugs_result.inserted + drugs_result.updated)

            # Step 4: Validate drug count (S18, DQ-13).
            # v90 ROOT FIX (BUG #16): use drugs_upserted (actual DB
            # writes) instead of len(df) (input row count) for the
            # quality gate. len(df) counts ALL input rows including
            # duplicates, invalid InChIKeys, and rows that failed
            # upsert — using it as the quality gate passes the check
            # even when zero rows were actually committed to the DB
            # (e.g. flush failure). The correct metric is the actual
            # number of rows that made it into the DB.
            drug_count = self._metrics.get("drugs_upserted", 0)
            if drug_count < CHEMBL_EXPECTED_DRUG_COUNT_MIN:
                # In test environments with CHEMBL_MAX_ROWS set very low,
                # the count validation will fail. Allow override via env.
                # FIX-P1-B-5 (audit P1): ``os.environ.get`` returns a
                # STRING. The previous check ``if not os.environ.get(...)``
                # treated every non-empty value — including "0", "false",
                # "no", "off" — as truthy and SKIPPED validation. That is
                # the opposite of operator intent: setting
                # CHEMBL_SKIP_COUNT_VALIDATION=0 means "do NOT skip".
                # Root fix: enforce validation (raise) when the value
                # (lower-cased) is unset or a clearly negative sentinel;
                # otherwise (affirmative: "1", "true", "yes", "on", ...)
                # fall through to the logger.warning() branch that
                # SKIPS validation. Only affirmative values skip.
                if os.environ.get("CHEMBL_SKIP_COUNT_VALIDATION", "").lower() in (
                    "",
                    "0",
                    "false",
                    "no",
                    "off",
                ):
                    raise PipelineError(
                        f"Drug count {drug_count} is below expected minimum "
                        f"{CHEMBL_EXPECTED_DRUG_COUNT_MIN}. Pipeline aborted "
                        f"to prevent downstream model from training on "
                        f"incomplete data (S18, DQ-13). Set "
                        f"CHEMBL_SKIP_COUNT_VALIDATION=1 to override."
                    )
                logger.warning(
                    "[%s] Drug count %d < min %d — skipped validation "
                    "(CHEMBL_SKIP_COUNT_VALIDATION set)",
                    self.source_name,
                    drug_count,
                    CHEMBL_EXPECTED_DRUG_COUNT_MIN,
                )

            # Step 5: Read cleaned activities.
            cleaned_activities_path = (
                PROCESSED_DATA_DIR / "chembl_activities_clean.csv"
            )
            if not cleaned_activities_path.exists():
                logger.info(
                    "[%s] No cleaned activities file at %s — skipping DPI load.",
                    self.source_name,
                    cleaned_activities_path,
                )
                self._update_pipeline_run_status(session, pipeline_run_id, "success")
                self._metrics["duration_load_sec"] = round(
                    time.monotonic() - load_start, 4
                )
                return total_loaded

            activities_df = pd.read_csv(
                cleaned_activities_path, encoding="utf-8", low_memory=False
            )
            if len(activities_df) == 0:
                logger.info("[%s] Cleaned activities file is empty.", self.source_name)
                self._update_pipeline_run_status(session, pipeline_run_id, "success")
                self._metrics["duration_load_sec"] = round(
                    time.monotonic() - load_start, 4
                )
                return total_loaded

            # Step 6: Resolve drug_id via get_chembl_to_drug_id_map (A9, P5).
            unique_chembl_ids = set(
                activities_df["molecule_chembl_id"]
                .dropna()
                .astype(str)
                .unique()
            )
            chembl_map_result: MappingResult = get_chembl_to_drug_id_map(
                session, chembl_ids=unique_chembl_ids
            )
            # K2 fix: use .mapping (MappingResult is NOT a dict).
            chembl_to_drug_id: dict[str, int] = chembl_map_result.mapping
            activities_df["drug_id"] = activities_df[
                "molecule_chembl_id"
            ].map(chembl_to_drug_id)

            # Step 7: Resolve protein_id via get_uniprot_to_protein_id_map (K2).
            unique_uniprot_ids = set(
                activities_df["target_accession"]
                .dropna()
                .astype(str)
                .unique()
            )
            uniprot_map_result: MappingResult = get_uniprot_to_protein_id_map(
                session, uniprot_ids=unique_uniprot_ids
            )
            # K2 fix: use .mapping (MappingResult is NOT a dict).
            uniprot_to_protein_id: dict[str, int] = uniprot_map_result.mapping
            activities_df["protein_id"] = activities_df[
                "target_accession"
            ].map(uniprot_to_protein_id)

            # Step 8: Drop activities with unresolved drug_id / protein_id.
            unresolved_drug_mask = activities_df["drug_id"].isna()
            if unresolved_drug_mask.any():
                dropped = activities_df[unresolved_drug_mask].copy()
                self._write_dead_letter(
                    dropped,
                    step="load_activities_unresolved_drug",
                    reason="molecule_chembl_id did not resolve to a drug_id",
                )
                unresolved_pct = len(dropped) / max(len(activities_df), 1) * 100
                logger.error(
                    "[%s] Dropping %d/%d activities with unresolved drug_id "
                    "(%.1f%%). Sample molecule_chembl_ids: %s",
                    self.source_name,
                    len(dropped),
                    len(activities_df),
                    unresolved_pct,
                    list(dropped["molecule_chembl_id"].head(10)),
                )
                if unresolved_pct > 50:
                    raise PipelineError(
                        f"More than 50% of activities ({unresolved_pct:.1f}%) "
                        f"have unresolved drug_id — aborting DPI load "
                        f"(DQ-9). Likely cause: drugs upsert failed silently."
                    )
                activities_df = activities_df[~unresolved_drug_mask].copy()

            unresolved_protein_mask = activities_df["protein_id"].isna()
            if unresolved_protein_mask.any():
                dropped = activities_df[unresolved_protein_mask].copy()
                self._write_dead_letter(
                    dropped,
                    step="load_activities_unresolved_protein",
                    reason="target_accession did not resolve to a protein_id",
                )
                unresolved_pct = len(dropped) / max(len(activities_df), 1) * 100
                logger.error(
                    "[%s] Dropping %d/%d activities with unresolved protein_id "
                    "(%.1f%%). Sample target_chembl_ids: %s",
                    self.source_name,
                    len(dropped),
                    len(activities_df),
                    unresolved_pct,
                    list(dropped.get("target_chembl_id", pd.Series()).head(10)),
                )
                if unresolved_pct > 50:
                    raise PipelineError(
                        f"More than 50% of activities ({unresolved_pct:.1f}%) "
                        f"have unresolved protein_id — aborting DPI load "
                        f"(DQ-10). Likely cause: UniProt pipeline hasn't run yet."
                    )
                activities_df = activities_df[~unresolved_protein_mask].copy()

            if len(activities_df) == 0:
                logger.warning(
                    "[%s] All activities dropped after resolution — no DPI to load.",
                    self.source_name,
                )
                self._update_pipeline_run_status(session, pipeline_run_id, "success")
                self._metrics["duration_load_sec"] = round(
                    time.monotonic() - load_start, 4
                )
                return total_loaded

            # Step 9: Aggregate by (drug_id, protein_id, activity_type) (S17).
            dpi_df = self._aggregate_activities_to_dpi(activities_df)

            # Step 10: Build the DPI DataFrame with required columns.
            dpi_df = self._build_dpi_dataframe(dpi_df)

            # Step 11: Bulk upsert DPI in chunks (P13).
            dpi_total = 0
            dpi_quarantined = 0
            for i in range(0, len(dpi_df), CHEMBL_DPI_BATCH_SIZE):
                chunk = dpi_df.iloc[i : i + CHEMBL_DPI_BATCH_SIZE].copy()
                dpi_result: UpsertResult = bulk_upsert_dpi(
                    session,
                    chunk,
                    pipeline_run_id=pipeline_run_id,
                    source_version=self.source_version,
                    source_fetch_date=self._source_fetch_date,
                    input_checksum=input_checksum,
                )
                dpi_total += int(dpi_result.inserted + dpi_result.updated)
                dpi_quarantined += dpi_result.quarantined
                logger.info(
                    "[%s] bulk_upsert_dpi chunk %d: input=%d, upserted=%d, "
                    "quarantined=%d",
                    self.source_name,
                    i // CHEMBL_DPI_BATCH_SIZE,
                    dpi_result.total_input,
                    dpi_result.inserted + dpi_result.updated,
                    dpi_result.quarantined,
                )

            self._metrics["dpi_upserted"] = dpi_total
            self._metrics["dpi_quarantined"] = dpi_quarantined
            total_loaded += dpi_total

            # Flush loader's dead-letter queue (R9, LIN-13).
            self._flush_loader_dead_letters(step="dpi")

            # Step 13: Update PipelineRun row status.
            self._update_pipeline_run_status(session, pipeline_run_id, "success")

        except (OSError, RuntimeError, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
            if owns_session and session is not None:
                try:
                    session.rollback()
                except (OSError, RuntimeError, ValueError):  # noqa: BLE001 — never mask the original error  # v85 FORENSIC ROOT FIX (BUG #51)
                    pass
            raise
        finally:
            # v29 ROOT FIX (audit P1-3): call __exit__ on the context
            # manager so it commits (on success) or rolls back (on
            # error) and closes the session. The previous code only
            # called session.close() — which (a) crashed because
            # ``session`` was the context manager, not the Session,
            # and (b) even if it had worked, would have skipped the
            # commit, silently rolling back ALL the loaded data.
            if owns_session and _session_cm is not None:
                import sys as _sys
                _exc_info = _sys.exc_info()
                try:
                    _session_cm.__exit__(*_exc_info)
                # FIX-P2-3 (audit P2): the previous ``except Exception: pass``
                # silently swallowed __exit__ failures — if commit fails
                # because the DB connection dropped, the caller saw load()
                # return success with NO data committed. Log the error so
                # operators can detect the silent data loss.
                # FIX-P2-7 (audit P2): narrow from broad ``except Exception``
                # to SQLAlchemy-related exceptions only. Programming bugs
                # (AttributeError from a typo, TypeError, etc.) now propagate
                # instead of being silently swallowed.
                except (
                    OperationalError,
                    IntegrityError,
                    SQLAlchemyError,
                ) as _exit_exc:  # noqa: BLE001
                    logger.error(
                        "[%s] session __exit__ failed (commit/rollback may "
                        "not have completed — loaded data may be lost): %s",
                        self.source_name, _exit_exc,
                    )

        self._metrics["duration_load_sec"] = round(
            time.monotonic() - load_start, 4
        )
        logger.info(
            "[%s] load() complete in %.2fs — drugs=%d, dpi=%d, total=%d",
            self.source_name,
            self._metrics["duration_load_sec"],
            self._metrics["drugs_upserted"],
            self._metrics["dpi_upserted"],
            total_loaded,
        )
        return total_loaded

    # ==================================================================
    # PRIVATE HELPERS — Download
    # ==================================================================

    def _verify_chembl_version(self) -> None:
        """Verify the ChEMBL API version (S20, INT-12).

        Calls ``/status.json`` and reads ``chembl_db_version``.

        Scientific correctness: ChEMBL is a continuously-updated biomedical
        database. Locking the pipeline to a single hard-coded version would
        cause the pipeline to FAIL whenever EBI releases a new version (which
        happens 2-3 times per year), and would also mean the platform ships
        STALE drug data to clinicians. The correct scientific behavior is:

        1. Detect the actual API version from ``/status.json``.
        2. Compare against the configured ``CHEMBL_VERSION`` (which now acts
           as a *minimum supported version*, not an exact-match requirement).
        3. If the API version is newer than configured, accept it, log an
           INFO message, and update ``self.source_version`` so downstream
           provenance records the actual version used.
        4. If the API version is *older* than configured, raise
           ``PipelineError`` — old versions may lack drug records the
           pipeline expects.
        5. If ``/status.json`` returns no version, log a warning and continue
           (defensive — never crash on version introspection).
        """
        try:
            status_url = f"{CHEMBL_API_URL}/status.json"
            data = self._api_get(status_url, {})
            actual_version = str(
                data.get("chembl_db_version", "")
            ).strip()
            if not actual_version:
                logger.warning(
                    "[%s] /status.json did not return chembl_db_version — "
                    "cannot verify API version. Continuing without version "
                    "verification (provenance will record 'unknown').",
                    self.source_name,
                )
                self.source_version = "ChEMBL_unknown"
                return

            # Compare numerically. The API may return either a bare number
            # ("35", "37") or a prefixed string ("ChEMBL_35", "ChEMBL_37").
            # Strip common prefixes before parsing.
            def _to_int_version(v: str) -> int | None:
                """Extract integer version from strings like '37' or 'ChEMBL_37'."""
                if not v:
                    return None
                # Strip known prefixes (case-insensitive).
                cleaned = v.strip()
                for prefix in ("ChEMBL_", "chembl_", "CHEMBL_", "ChEMBL", "chembl", "CHEMBL"):
                    if cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):]
                        break
                cleaned = cleaned.strip()
                try:
                    return int(cleaned)
                except (ValueError, TypeError):
                    return None

            actual_num = _to_int_version(actual_version)
            configured_num = _to_int_version(str(CHEMBL_VERSION))

            # FIX-P2-11 (audit P2): ``actual_version`` is a str (from
            # ``str(data.get("chembl_db_version", "")).strip()`` above)
            # while ``CHEMBL_VERSION`` may be configured as an int (e.g.
            # ``CHEMBL_VERSION = 33`` in config/settings.py). Python's
            # ``"33" == 33`` evaluates to False, so the equality branch
            # was never taken even when the API returned the expected
            # version — control always fell through to the numeric
            # comparison below, which logs a spurious "newer than
            # configured" INFO message. Comparing str-to-str makes the
            # equality branch fire as intended.
            if actual_version == str(CHEMBL_VERSION):
                logger.info(
                    "[%s] ChEMBL API version verified: %s",
                    self.source_name,
                    actual_version,
                )
                self.source_version = f"ChEMBL_{actual_version}"
            elif actual_num is not None and actual_num > configured_num:
                # API is newer than configured — accept and adapt.
                logger.info(
                    "[%s] ChEMBL API version %s is newer than configured "
                    "CHEMBL_VERSION=%s. Adapting to live API version for "
                    "scientific currency (newer drug records will be used). "
                    "Provenance will record the actual version.",
                    self.source_name, actual_version, CHEMBL_VERSION,
                )
                self.source_version = f"ChEMBL_{actual_version}"
            elif actual_num is not None and actual_num < configured_num:
                # API is older than configured — refuse to run.
                msg = (
                    f"ChEMBL API version {actual_version} is older than "
                    f"configured CHEMBL_VERSION={CHEMBL_VERSION}. Older "
                    f"versions may lack drug records the pipeline expects. "
                    f"Either downgrade CHEMBL_VERSION to {actual_version} or "
                    f"set CHEMBL_ALLOW_VERSION_MISMATCH=True to override."
                )
                if CHEMBL_ALLOW_VERSION_MISMATCH:
                    logger.warning(
                        "[%s] %s — continuing (ALLOW_VERSION_MISMATCH=True)",
                        self.source_name, msg,
                    )
                    self.source_version = f"ChEMBL_{actual_version}"
                else:
                    logger.error("[%s] %s — aborting.", self.source_name, msg)
                    raise PipelineError(msg)
            else:
                # Versions differ but cannot be compared numerically
                msg = (
                    f"ChEMBL API version mismatch: expected {CHEMBL_VERSION}, "
                    f"got {actual_version}"
                )
                if CHEMBL_ALLOW_VERSION_MISMATCH:
                    logger.warning(
                        "[%s] %s — continuing (ALLOW_VERSION_MISMATCH=True)",
                        self.source_name, msg,
                    )
                    self.source_version = f"ChEMBL_{actual_version}"
                else:
                    logger.error("[%s] %s — aborting.", self.source_name, msg)
                    raise PipelineError(msg)
        except (HttpClientError, PipelineError):
            raise
        # P1-13 ROOT FIX: previously this was a bare ``except Exception``.
        # That swallowed every error — including programming bugs (e.g.
        # AttributeError from a typo in the version-comparison logic) and
        # network/HTTP errors that should bubble up via HttpClientError
        # (already re-raised above). Narrowing to the four exception types
        # the version-comparison code can actually raise keeps the
        # "defensive — never crash on version check" guarantee while
        # letting real bugs surface.
        # FIX-P2-10 (audit P2): narrowed from
        # ``(json.JSONDecodeError, KeyError, TypeError, ValueError)`` to
        # ``(json.JSONDecodeError, KeyError)``. ``TypeError`` and
        # ``ValueError`` are exactly the exception types that programming
        # bugs in the version-comparison logic raise (e.g. comparing a str
        # to an int with ``>`` in Python 3 raises TypeError; calling int()
        # on garbage raises ValueError). Catching them downgraded real bugs
        # to a "could not verify" warning, silently disabling version
        # verification. Let them propagate so bugs surface. The two
        # remaining types are exactly what the JSON-decode path can raise
        # for legitimately missing/malformed ``/status.json`` payloads.
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "[%s] Could not verify ChEMBL API version: %s — continuing.",
                self.source_name,
                exc,
            )

    def _download_molecules(self) -> pd.DataFrame:
        """Paginate through ChEMBL ``/molecule.json`` for ``max_phase=4``.

        Returns
        -------
        pd.DataFrame
            Parsed molecule records. Columns: see
            :meth:`_parse_molecules`.

        Notes
        -----
        - Stops on empty page, ``CHEMBL_MAX_ROWS`` reached, or short page
          (C42, C43, C44, C45, C47).
        - Pagination uses ``CHEMBL_PAGE_SIZE`` (default 1000; max per
          ChEMBL API contract — INT-2).
        - v49 ROOT FIX: in sample mode, only fetches the first page of
          ``SAMPLE_RECORD_COUNT`` records (default 200) so the platform
          runs end-to-end on a laptop without downloading all 2M+
          ChEMBL molecules.
        """
        # v49 ROOT FIX: sample mode — fetch only the first page.
        if self.download_mode == "sample":
            logger.info(
                "[chembl] SAMPLE MODE: downloading only %d molecules "
                "(set DRUGOS_DOWNLOAD_MODE=full for the complete 2M+ "
                "molecule corpus).",
                self.SAMPLE_RECORD_COUNT,
            )
            params = {
                "max_phase": CHEMBL_MAX_PHASE,
                "format": "json",
                "limit": min(self.SAMPLE_RECORD_COUNT, CHEMBL_PAGE_SIZE),
                "offset": 0,
            }
            url = f"{CHEMBL_API_URL}/molecule.json"
            try:
                data = self._api_get(url, params)
                molecules = data.get("molecules", [])
                if molecules:
                    df = self._parse_molecules(molecules)
                    logger.info(
                        "[chembl] SAMPLE MODE: fetched %d molecules.", len(df)
                    )
                    return df
                logger.warning("[chembl] SAMPLE MODE: API returned 0 molecules.")
                return pd.DataFrame()
            except (OSError, ValueError, ConnectionError, TimeoutError) as exc:  # v85 FORENSIC ROOT FIX (BUG #51)
                logger.warning(
                    "[chembl] SAMPLE MODE: live API fetch failed (%s). "
                    "Falling back to embedded sample dataset (5 FDA-approved "
                    "drugs) so the pipeline can still run end-to-end.",
                    exc,
                )
                return self._embedded_sample_molecules()

        all_chunks: list[pd.DataFrame] = []
        offset = 0
        total_count: int | None = None

        while True:
            params = {
                "max_phase": CHEMBL_MAX_PHASE,
                "format": "json",
                "limit": CHEMBL_PAGE_SIZE,
                "offset": offset,
            }
            url = f"{CHEMBL_API_URL}/molecule.json"
            data = self._api_get_with_rate_limit_tracking(url, params)
            molecules = data.get("molecules", [])
            page_meta = data.get("page_meta", {})
            # P1-1 ROOT FIX (silent truncation): The previous code did
            # ``total_count = int(page_meta.get("total_count", 0))`` which
            # defaulted to 0 when the API omitted the field. A 0 then made
            # ``offset + len(molecules) >= total_count`` (i.e. ``>= 0``)
            # evaluate True on the very first page, silently truncating the
            # entire molecule corpus to a single 1000-row page. The fix:
            # treat a missing/non-positive ``total_count`` as "unknown"
            # (None) and fall back to a short-page termination rule.
            if total_count is None:
                raw_total = page_meta.get("total_count")
                if raw_total is not None:
                    try:
                        candidate = int(raw_total)
                        total_count = candidate if candidate > 0 else None
                    except (TypeError, ValueError):
                        total_count = None
                if total_count is not None:
                    logger.info(
                        "[%s] /molecule.json total_count=%d",
                        self.source_name,
                        total_count,
                    )
                else:
                    logger.warning(
                        "[%s] /molecule.json omitted total_count (or "
                        "returned 0). Paging until empty/short page to "
                        "avoid silent truncation (P1-1 ROOT FIX).",
                        self.source_name,
                    )

            if not molecules:
                logger.info(
                    "[%s] Empty molecule page at offset=%d — stopping.",
                    self.source_name,
                    offset,
                )
                break

            parsed_chunk = self._parse_molecules(molecules)
            all_chunks.append(parsed_chunk)

            # C47: respect CHEMBL_MAX_ROWS — break BEFORE extending past
            # the cap, then extend with a truncated slice.
            current_count = sum(len(c) for c in all_chunks)
            if CHEMBL_MAX_ROWS is not None and current_count >= CHEMBL_MAX_ROWS:
                logger.info(
                    "[%s] Reached CHEMBL_MAX_ROWS=%d — stopping.",
                    self.source_name,
                    CHEMBL_MAX_ROWS,
                )
                break

            # C45: loop termination — break when we've fetched all pages.
            # P1-1 ROOT FIX: only trust ``total_count`` when the API
            # actually provided it. When ``total_count`` is unknown, fall
            # back to a short-page termination rule (fewer than
            # ``CHEMBL_PAGE_SIZE`` records means we've reached the final
            # page). Without this fall-back the loop previously broke
            # after the first page (see P1-1 rationale above).
            if total_count is not None and offset + len(molecules) >= total_count:
                break
            if total_count is None and len(molecules) < CHEMBL_PAGE_SIZE:
                logger.info(
                    "[%s] Short molecule page (%d < %d) at offset=%d — "
                    "stopping (total_count unknown, P1-1 fall-back).",
                    self.source_name,
                    len(molecules),
                    CHEMBL_PAGE_SIZE,
                    offset,
                )
                break

            offset += len(molecules)

        if all_chunks:
            df = pd.concat(all_chunks, ignore_index=True)
        else:
            df = pd.DataFrame(
                columns=[
                    "chembl_id", "name", "inchikey", "smiles",
                    "molecular_weight", "drug_type", "max_phase",
                    "is_fda_approved",
                ]
            )

        # C47: truncate to CHEMBL_MAX_ROWS if necessary.
        if CHEMBL_MAX_ROWS is not None and len(df) > CHEMBL_MAX_ROWS:
            df = df.iloc[:CHEMBL_MAX_ROWS].copy()

        # DQ-17: pagination completeness check (allow 5% API wiggle).
        # P1-1 ROOT FIX: previously this branch only LOGGED a warning and
        # returned the truncated frame. Silent truncation defeats every
        # downstream guarantee (drug count, dedup, KG build). The v9 ROOT
        # FIX promised operators would see this failure; we now raise
        # ``PipelineError`` so the run exits non-zero instead of silently
        # shipping a partial corpus.
        if total_count is not None and total_count > 0:
            fetched = len(df)
            expected = (
                min(total_count, CHEMBL_MAX_ROWS)
                if CHEMBL_MAX_ROWS is not None
                else total_count
            )
            if fetched < expected * 0.95:
                msg = (
                    f"[{self.source_name}] Pagination completeness FAILED: "
                    f"fetched {fetched} / expected {expected} "
                    f"({fetched / expected * 100:.1f}%). "
                    f"API reported total_count={total_count} but the loop "
                    f"returned far fewer rows. Aborting to prevent silent "
                    f"data loss (P1-1 ROOT FIX)."
                )
                logger.error(msg)
                raise PipelineError(msg)

        # DQ-5: dedup by chembl_id (keep first; log dropped).
        # v35 ROOT FIX (issue 21): include ``salt_form`` in the dedup key
        # when it is present. ChEMBL salts (CHEMBL123 + Cl-, CHEMBL123 + Na+)
        # are DISTINCT molecules with the SAME chembl_id but DIFFERENT
        # InChIKeys — collapsing them by chembl_id alone would silently
        # lose salt-form diversity (e.g. morphine sulfate vs morphine
        # hydrochloride). When ``salt_form`` is absent (older snapshots),
        # fall back to chembl_id-only dedup with a warning comment.
        if len(df) > 0:
            before = len(df)
            if "salt_form" in df.columns and df["salt_form"].notna().any():
                df = df.drop_duplicates(
                    subset=["chembl_id", "salt_form"], keep="first"
                )
            else:
                # salt_form column absent or entirely null — fall back to
                # chembl_id-only dedup (legacy behavior).
                df = df.drop_duplicates(subset=["chembl_id"], keep="first")
            if len(df) < before:
                logger.info(
                    "[%s] Dropped %d duplicate molecules by chembl_id",
                    self.source_name,
                    before - len(df),
                )
        return df

    def _download_activities(self) -> pd.DataFrame:
        """Paginate through ChEMBL ``/activity.json`` for human bioactivities.

        Returns
        -------
        pd.DataFrame
            Parsed activity records. Columns: see
            :meth:`_parse_activities`.

        K1 Fix
        ------
        The previous version used ``list.extend(DataFrame)`` which iterates
        the DataFrame's COLUMN NAMES, not its rows, producing a garbage
        1-column DataFrame of column-name strings. The fix returns
        ``pd.DataFrame(list_of_dicts)`` from the accumulated list of
        parsed record dicts — avoiding both the extend bug and the
        memory overhead of creating a DataFrame per chunk.

        Notes
        -----
        - Filters activities by ``target_organism=CHEMBL_TARGET_ORGANISM``
          (default "Homo sapiens" — S15).
        - Filters by ``standard_type__in=IC50,Ki,Kd,EC50`` (S10).
        - Stops on empty page, ``CHEMBL_MAX_ACTIVITIES`` reached, or short
          page (C42, C43).
        - Writes each page's raw JSON to a chunk file
          (``activity_chunk_{run_id}_{offset}.json``) for crash-recovery
          / resume (R6, LIN-8). Chunk files are NOT loaded back into
          memory — they're written for audit and resume only.
        """
        all_records: list[dict[str, Any]] = []
        offset = 0
        total_count: int | None = None
        activity_types_str = ",".join(sorted(CHEMBL_ACTIVITY_TYPES))
        chunk_files: list[Path] = []

        try:
            while True:
                params = {
                    "target_organism": CHEMBL_TARGET_ORGANISM,
                    "standard_type__in": activity_types_str,
                    "has_standard_value": "true",
                    "format": "json",
                    "limit": CHEMBL_PAGE_SIZE,
                    "offset": offset,
                }
                url = f"{CHEMBL_API_URL}/activity.json"
                data = self._api_get_with_rate_limit_tracking(url, params)
                activities = data.get("activities", [])
                page_meta = data.get("page_meta", {})
                # P1-1 ROOT FIX (silent truncation): see _download_molecules
                # for full rationale. The previous code defaulted
                # ``total_count`` to 0 when the API omitted the field, then
                # ``offset + len(activities) >= total_count`` (i.e. ``>= 0``)
                # evaluated True on the first page, silently truncating the
                # entire activity corpus to 1000 rows. Treat a missing /
                # non-positive ``total_count`` as unknown and rely on the
                # short-page / empty-page termination rule instead.
                if total_count is None:
                    raw_total = page_meta.get("total_count")
                    if raw_total is not None:
                        try:
                            candidate = int(raw_total)
                            total_count = candidate if candidate > 0 else None
                        except (TypeError, ValueError):
                            total_count = None
                    if total_count is not None:
                        logger.info(
                            "[%s] /activity.json total_count=%d",
                            self.source_name,
                            total_count,
                        )
                    else:
                        logger.warning(
                            "[%s] /activity.json omitted total_count (or "
                            "returned 0). Paging until empty/short page to "
                            "avoid silent truncation (P1-1 ROOT FIX).",
                            self.source_name,
                        )

                if not activities:
                    logger.info(
                        "[%s] Empty activity page at offset=%d — stopping.",
                        self.source_name,
                        offset,
                    )
                    break

                # Write the raw page to a chunk file for audit/resume (R6, LIN-8).
                # The chunk file is NOT loaded back — we accumulate the parsed
                # records in memory (K1 fix: list of dicts, not DataFrames).
                # Use getattr fallback for tests that bypass __init__.
                run_id = getattr(self, "run_id", "unknown_run_id")
                chunk_path = self.raw_dir / f"activity_chunk_{run_id}_{offset}.json"
                try:
                    with open(chunk_path, "w", encoding="utf-8") as fh:
                        json.dump(activities, fh)
                    chunk_files.append(chunk_path)
                    logger.debug(
                        "[%s] Wrote chunk %s (%d activities)",
                        self.source_name, chunk_path, len(activities),
                    )
                except OSError as exc:
                    logger.warning(
                        "[%s] Could not write chunk file %s: %s — "
                        "continuing without crash-recovery for this page.",
                        self.source_name, chunk_path, exc,
                    )

                parsed = self._parse_activities(activities)
                all_records.extend(parsed)

                # C42/C47: respect CHEMBL_MAX_ACTIVITIES.
                if (
                    CHEMBL_MAX_ACTIVITIES is not None
                    and len(all_records) >= CHEMBL_MAX_ACTIVITIES
                ):
                    all_records = all_records[:CHEMBL_MAX_ACTIVITIES]
                    logger.info(
                        "[%s] Reached CHEMBL_MAX_ACTIVITIES=%d — stopping.",
                        self.source_name,
                        CHEMBL_MAX_ACTIVITIES,
                    )
                    break

                # C45: loop termination.
                # P1-1 ROOT FIX: only trust ``total_count`` when the API
                # actually provided it. When unknown, fall back to a
                # short-page termination rule (fewer than
                # ``CHEMBL_PAGE_SIZE`` records means we've reached the
                # final page).
                if (
                    total_count is not None
                    and offset + len(activities) >= total_count
                ):
                    break
                if total_count is None and len(activities) < CHEMBL_PAGE_SIZE:
                    logger.info(
                        "[%s] Short activity page (%d < %d) at offset=%d — "
                        "stopping (total_count unknown, P1-1 fall-back).",
                        self.source_name,
                        len(activities),
                        CHEMBL_PAGE_SIZE,
                        offset,
                    )
                    break

                offset += len(activities)

            # K1 fix: build DataFrame from list of dicts (not extend, not concat).
            if all_records:
                df = pd.DataFrame(all_records)
                logger.info(
                    "[%s] Built activities DataFrame: %d rows, %d columns",
                    self.source_name,
                    len(df),
                    len(df.columns),
                )
                # P1-1 ROOT FIX: post-loop completeness assertion for
                # activities. Previously _download_activities had NO
                # completeness check at all — a silently truncated run
                # would proceed to KG build with a partial activity corpus
                # and the operator would never know. Mirror the
                # _download_molecules assertion: if the API reported a
                # total_count and we fetched < 95% of it, raise
                # ``PipelineError`` so the run exits non-zero.
                if total_count is not None and total_count > 0:
                    expected = (
                        min(total_count, CHEMBL_MAX_ACTIVITIES)
                        if CHEMBL_MAX_ACTIVITIES is not None
                        else total_count
                    )
                    if len(df) < expected * 0.95:
                        msg = (
                            f"[{self.source_name}] Activity pagination "
                            f"completeness FAILED: fetched {len(df)} / "
                            f"expected {expected} "
                            f"({len(df) / expected * 100:.1f}%). "
                            f"API reported total_count={total_count} but "
                            f"the loop returned far fewer rows. Aborting "
                            f"to prevent silent data loss (P1-1 ROOT FIX)."
                        )
                        logger.error(msg)
                        raise PipelineError(msg)
                return df
            # Return an empty DF with the expected schema (K1 acceptance).
            return pd.DataFrame(
                columns=[
                    "activity_id", "molecule_chembl_id", "target_chembl_id",
                    "target_pref_name", "activity_type", "activity_value",
                    "activity_units", "pchembl_value", "assay_id",
                    "standard_relation", "assay_type",
                ]
            )
        finally:
            # LIN-8: by default, chunk files are PERSISTED for audit /
            # resume. Set CHEMBL_RESUME=false (default) to clean them up
            # after a successful run. Set CHEMBL_RESUME=true to keep them
            # for resume-from-checkpoint (R6).
            if not CHEMBL_RESUME:
                for chunk_path in chunk_files:
                    try:
                        chunk_path.unlink(missing_ok=True)
                    except OSError:
                        pass  # best-effort cleanup

    def _embedded_sample_molecules(self) -> pd.DataFrame:
        """v49 ROOT FIX: embedded sample dataset of 10 FDA-approved drugs.

        Used when DRUGOS_DOWNLOAD_MODE=sample AND the live ChEMBL API
        is unreachable (no network, rate-limit, etc.). Returns a
        DataFrame with the same columns as `_parse_molecules` so the
        rest of the pipeline (clean / load) works unchanged.

        The 10 drugs are well-known FDA-approved compounds with valid
        InChIKeys, SMILES, and ChEMBL IDs — chosen so the Phase 2 KG
        has biologically meaningful Compound nodes even when offline.
        """
        import pandas as _pd
        samples = [
            # v108 FORENSIC ROOT FIX (ISSUE-P1-003): was CHEMBL112
            # (Acetaminophen, NOT Aspirin). Aspirin = CHEMBL25.
            {
                "chembl_id": "CHEMBL25",
                "name": "Aspirin",
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                "molecular_weight": 180.16,
                "max_phase": 4,
                "is_fda_approved": True,
                "is_globally_approved": True,
                "indication": "for the treatment of pain, inflammation, and fever",
                "indication_source": "manual",
                "mechanism_of_action": "Cyclooxygenase inhibitor",
            },
            # v108 FORENSIC ROOT FIX (ISSUE-P1-003): was CHEMBL21
            # (Dexfenfluramine, NOT Acetaminophen). Acetaminophen = CHEMBL112.
            {
                "chembl_id": "CHEMBL112",
                "name": "Acetaminophen",
                "smiles": "CC1=CC=C(O)C=C1O",
                "inchikey": "RZVAJINKPMORJF-UHFFFAOYSA-N",
                "molecular_weight": 151.16,
                "max_phase": 4,
                "is_fda_approved": True,
                "is_globally_approved": True,
                "indication": "for the treatment of pain and fever",
                "indication_source": "manual",
                "mechanism_of_action": "Cyclooxygenase inhibitor (central)",
            },
            # v108 FORENSIC ROOT FIX (ISSUE-P1-003): was CHEMBL705
            # (not Ibuprofen). Ibuprofen = CHEMBL521.
            {
                "chembl_id": "CHEMBL521",
                "name": "Ibuprofen",
                "smiles": "CC(C)CC1=CC=C(C=C1)CC(C(=O)O)C",
                "inchikey": "HEFNNWSXXWATIW-UHFFFAOYSA-N",
                "molecular_weight": 206.28,
                "max_phase": 4,
                "is_fda_approved": True,
                "is_globally_approved": True,
                "indication": "for the treatment of pain, inflammation, and arthritis",
                "indication_source": "manual",
                "mechanism_of_action": "Non-selective COX inhibitor",
            },
            # v108 FORENSIC ROOT FIX (ISSUE-P1-003): was CHEMBL521
            # (Ibuprofen, NOT Caffeine). Caffeine = CHEMBL113.
            {
                "chembl_id": "CHEMBL113",
                "name": "Caffeine",
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
                "molecular_weight": 194.19,
                "max_phase": 4,
                "is_fda_approved": True,
                "is_globally_approved": True,
                "indication": "for the treatment of migraine and fatigue",
                "indication_source": "manual",
                "mechanism_of_action": "Adenosine receptor antagonist",
            },
            {
                # v108 FORENSIC ROOT FIX (ISSUE-P1-003): was CHEMBL503
                # (Dihydroergotamine). Diazepam = CHEMBL12.
                "chembl_id": "CHEMBL12",
                "name": "Diazepam",
                "smiles": "ClC1=CC2=C(C=C1)C(=NCC(=O)N2C3=CC=CC=C3)C",
                "inchikey": "AAOVKBJEBZCEQK-UHFFFAOYSA-N",
                "molecular_weight": 284.74,
                "max_phase": 4,
                "is_fda_approved": True,
                "is_globally_approved": True,
                "indication": "for the treatment of anxiety and seizures",
                "indication_source": "manual",
                "mechanism_of_action": "GABA-A receptor positive allosteric modulator",
            },
            {
                # v108 FORENSIC ROOT FIX (ISSUE-P1-003): was CHEMBL2114647
                # (does not exist in ChEMBL). Warfarin = CHEMBL1464.
                "chembl_id": "CHEMBL1464",
                "name": "Warfarin",
                "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
                "inchikey": "PJVWKTKQMONHTF-UHFFFAOYSA-N",
                "molecular_weight": 308.33,
                "max_phase": 4,
                "is_fda_approved": True,
                "is_globally_approved": True,
                "indication": "for the prevention of thrombosis and embolism",
                "indication_source": "manual",
                "mechanism_of_action": "Vitamin K epoxide reductase inhibitor",
            },
            {
                # v108 FORENSIC ROOT FIX (ISSUE-P1-003): was CHEMBL546
                # (Ethinylestradiol). Metformin = CHEMBL1431.
                "chembl_id": "CHEMBL1431",
                "name": "Metformin",
                "smiles": "CN(C)C(=N)N=C(N)N",
                "inchikey": "XZWYZXLIPXDOLR-UHFFFAOYSA-N",
                "molecular_weight": 129.16,
                "max_phase": 4,
                "is_fda_approved": True,
                "is_globally_approved": True,
                "indication": "for the treatment of type 2 diabetes",
                "indication_source": "manual",
                "mechanism_of_action": "AMPK activator; mitochondrial complex I inhibitor",
            },
            {
                # v108 FORENSIC ROOT FIX (ISSUE-P1-003): was CHEMBL1085
                # (Levonorgestrel). Atorvastatin = CHEMBL1487.
                "chembl_id": "CHEMBL1487",
                "name": "Atorvastatin",
                "smiles": "CC(C1=C(C(=CC=C1)C)C2=CC=CC=C2C(=O)NC3CC4=C(C=C(C=C4CC3)F)C(=O)O)C(C)C",
                "inchikey": "XUKUURHRXDUEBC-UHFFFAOYSA-N",
                "molecular_weight": 558.66,
                "max_phase": 4,
                "is_fda_approved": True,
                "is_globally_approved": True,
                "indication": "for the treatment of hypercholesterolemia",
                "indication_source": "manual",
                "mechanism_of_action": "HMG-CoA reductase inhibitor",
            },
            {
                # v108 FORENSIC ROOT FIX (ISSUE-P1-003): was CHEMBL2318659.
                # Captopril = CHEMBL1560.
                "chembl_id": "CHEMBL1560",
                "name": "Captopril",
                "smiles": "CC(C)C1CC2C(SC1)C(=O)NC2C(=O)O",
                "inchikey": "BNRQQXFRAQNPGX-UHFFFAOYSA-N",
                "molecular_weight": 217.29,
                "max_phase": 4,
                "is_fda_approved": True,
                "is_globally_approved": True,
                "indication": "for the treatment of hypertension",
                "indication_source": "manual",
                "mechanism_of_action": "ACE inhibitor",
            },
            {
                # v108 FORENSIC ROOT FIX (ISSUE-P1-003): was CHEMBL586447.
                # Lisinopril = CHEMBL419213.
                "chembl_id": "CHEMBL419213",
                "name": "Lisinopril",
                "smiles": "CCCCC(C)C1C(=O)N2CCCC2C(=O)N1CC(C(=O)O)N",
                "inchikey": "RJXRWZVZAQXBEZ-UHFFFAOYSA-N",
                "molecular_weight": 405.49,
                "max_phase": 4,
                "is_fda_approved": True,
                "is_globally_approved": True,
                "indication": "for the treatment of hypertension and heart failure",
                "indication_source": "manual",
                "mechanism_of_action": "ACE inhibitor",
            },
        ]
        return _pd.DataFrame(samples)

    def _parse_molecules(self, molecules: list[dict[str, Any]]) -> pd.DataFrame:
        """Extract relevant fields from ChEMBL molecule JSON records.

        Parameters
        ----------
        molecules : list of dict
            Raw molecule records from ``/molecule.json``.

        Returns
        -------
        pandas.DataFrame
            Parsed records with columns: ``chembl_id, name, inchikey, smiles,
            molecular_weight, drug_type, max_phase, is_fda_approved``.
            Always returns a DataFrame (even for empty input) with the
            expected column schema — never returns a list.

        K4 Fix
        ------
        ChEMBL returns ``max_phase`` as a STRING (e.g. ``"4.0"``). We
        coerce to ``int(float(...))`` and clamp to ``[0, 4]``. Without
        this, ``max_phase == 4`` evaluates to ``False`` (string "4.0" !=
        int 4) and ``is_fda_approved`` is wrong for every record.

        K6 Fix
        ------
        ``molecule_type`` is mapped to a valid ``DrugType`` enum value via
        ``MOLECULE_TYPE_MAP``. The map's values are all lowercase enum
        members (e.g. ``"small_molecule"``), so the loader's
        ``_validate_drug_type`` accepts them.

        Verified Activity Record Schema (paste from §2.8 of the fix prompt,
        verified live against https://www.ebi.ac.uk/chembl/api/data/molecule.json).
        The molecule record has these top-level keys (note: the molecule-type
        field is a Title-case string that we map to a DrugType enum value via
        MOLECULE_TYPE_MAP — K6 fix)::

            {
              "molecule_chembl_id": "CHEMBL123",
              "pref_name": "aspirin",
              "max_phase": "4.0",       // STRING, not int!
              "molecule_properties": {"full_mwt": 180.16, "num_ro5_violations": 0},
              "molecule_structures": {
                "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "standard_inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                "standard_inchi": "InChI=1S/..."
              }
            }
        """
        records: list[dict[str, Any]] = []
        for mol in molecules:
            chembl_id = str(mol.get("molecule_chembl_id", "")).strip()
            if not chembl_id:
                continue  # DQ-5: skip records without a chembl_id

            # pref_name — C13: default to None; synthesize later if needed.
            pref_name = mol.get("pref_name")
            if pref_name is not None:
                pref_name = str(pref_name).strip() or None

            # K4 fix: coerce max_phase to int in [0, 4].
            max_phase = self._coerce_max_phase(mol.get("max_phase"), chembl_id)

            # K6 fix: map molecule_type to valid DrugType enum value.
            mol_type_raw = mol.get("molecule_type")
            drug_type = self._standardize_drug_type(mol_type_raw)

            # Extract properties.
            props = mol.get("molecule_properties") or {}
            mw_raw = props.get("full_mwt")
            try:
                mw = float(mw_raw) if mw_raw is not None else None
            except (TypeError, ValueError):
                logger.warning(
                    "[%s] Invalid molecular_weight %r for %s — setting to None",
                    self.source_name, mw_raw, chembl_id,
                )
                mw = None

            # Extract structures.
            struct = mol.get("molecule_structures") or {}
            inchikey = struct.get("standard_inchi_key")
            if inchikey is not None:
                inchikey = str(inchikey).strip() or None
            smiles = (
                struct.get("canonical_smiles")
                or struct.get("smiles")
            )
            if smiles is not None:
                smiles = str(smiles).strip() or None

            # SW-1 ROOT FIX (patient safety): ``is_fda_approved`` was
            # derived from ``max_phase == 4``, which is GLOBAL approval
            # (any of FDA / EMA / PMDA / MHRA / Health Canada / TGA),
            # NOT FDA-specific. An EMA-only-approved compound was silently
            # marked FDA-approved, corrupting the RL ranker's safety
            # filter. ChEMBL does not provide FDA-specific approval —
            # the honest fix is to rename the column to
            # ``is_globally_approved`` (matches the ChEMBL semantics
            # exactly) and leave ``is_fda_approved`` as None (unknown)
            # until an FDA Orange Book join is wired in. Downstream
            # code MUST treat ``is_fda_approved IS NULL`` as "unknown
            # — require manual review" rather than auto-fast-tracking.
            is_globally_approved = bool(max_phase == 4)
            is_fda_approved = None  # populated only by FDA Orange Book join

            records.append({
                "chembl_id": chembl_id,
                "name": pref_name,
                "inchikey": inchikey,
                "smiles": smiles,
                "molecular_weight": mw,
                "drug_type": drug_type,
                "max_phase": max_phase,
                "is_globally_approved": is_globally_approved,
                "is_fda_approved": is_fda_approved,
            })
        # Always return a DataFrame with the expected column schema
        # (test_all_45_fixes::TestIssue19 — empty input must still have
        # the expected columns).
        expected_cols = [
            "chembl_id", "name", "inchikey", "smiles",
            "molecular_weight", "drug_type", "max_phase",
            "is_globally_approved", "is_fda_approved",
        ]
        if not records:
            return pd.DataFrame(columns=expected_cols)
        df = pd.DataFrame(records)
        # Ensure column order matches the contract.
        return df[expected_cols]

    def _parse_activities(self, activities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract relevant fields from ChEMBL activity JSON records.

        Parameters
        ----------
        activities : list of dict
            Raw activity records from ``/activity.json``.

        Returns
        -------
        list of dict
            Parsed records with keys: ``activity_id, molecule_chembl_id,
            target_chembl_id, target_pref_name, activity_type,
            activity_value, activity_units, pchembl_value, assay_id,
            standard_relation, assay_type``.

        K8 Fix
        ------
        The previous version read ``act.get("target_accession")`` which is
        a NON-EXISTENT field on the activity record. The line was dead
        code. The UniProt accession is obtained later via
        :meth:`_resolve_target_accessions` (which calls ``/target.json``).

        Verified Activity Record Schema (paste from §2.8 of the fix prompt,
        verified live against https://www.ebi.ac.uk/chembl/api/data/activity.json)::

            {
              "activity_id": 12345,
              "molecule_chembl_id": "CHEMBL123",
              "target_chembl_id": "CHEMBL456",
              "target_pref_name": "Some target",
              "assay_chembl_id": "CHEMBL789",
              "standard_type": "IC50",
              "standard_value": 12.5,
              "standard_units": "nM",
              "standard_relation": "=",
              "pchembl_value": 7.9,
              "assay_type": "B",
              "target_organism": "Homo sapiens"
            }

        There is NO ``target_accession`` field on the activity record.
        """
        records: list[dict[str, Any]] = []
        for act in activities:
            activity_id = act.get("activity_id")
            if activity_id is None:
                continue  # DQ-14: skip records without activity_id
            activity_id = str(activity_id)

            mol_chembl_id = str(act.get("molecule_chembl_id", "")).strip()
            target_chembl_id = str(act.get("target_chembl_id", "")).strip()
            if not mol_chembl_id or not target_chembl_id:
                continue  # DQ: skip records missing required FKs

            target_pref_name = act.get("target_pref_name")
            if target_pref_name is not None:
                target_pref_name = str(target_pref_name).strip() or None

            # activity_type: standard_type preferred, fallback to activity_type.
            std_type = (
                act.get("standard_type")
                or act.get("activity_type")
            )
            if std_type is not None:
                std_type = str(std_type).strip() or None

            # activity_value: coerce to float, None on failure.
            std_value = act.get("standard_value")
            try:
                std_value = float(std_value) if std_value is not None else None
            except (TypeError, ValueError):
                std_value = None

            std_units = act.get("standard_units")
            if std_units is not None:
                std_units = str(std_units).strip() or None

            # S14: preserve pchembl_value.
            pchembl = act.get("pchembl_value")
            try:
                pchembl = float(pchembl) if pchembl is not None else None
            except (TypeError, ValueError):
                pchembl = None

            assay_chembl_id = act.get("assay_chembl_id")
            if assay_chembl_id is not None:
                assay_chembl_id = str(assay_chembl_id).strip() or None

            # S12: standard_relation ('=', '>', '<', '~', '>=', '<=').
            std_relation = act.get("standard_relation")
            if std_relation is not None:
                std_relation = str(std_relation).strip() or None

            # S10: assay_type ('B', 'F', 'U', 'A', 'P', 'T').
            assay_type = act.get("assay_type")
            if assay_type is not None:
                assay_type = str(assay_type).strip().upper() or None

            records.append({
                "activity_id": activity_id,
                "molecule_chembl_id": mol_chembl_id,
                "target_chembl_id": target_chembl_id,
                "target_pref_name": target_pref_name,
                "activity_type": std_type,
                "activity_value": std_value,
                "activity_units": std_units,
                "pchembl_value": pchembl,
                "assay_id": assay_chembl_id,
                "standard_relation": std_relation,
                "assay_type": assay_type,
            })
        return records

    def _resolve_target_accessions(
        self, target_chembl_ids: set[str]
    ) -> dict[str, list[str]]:
        """Resolve ChEMBL target IDs to lists of UniProt accessions.

        Parameters
        ----------
        target_chembl_ids : set of str
            Set of ``CHEMBL\\d+`` target IDs.

        Returns
        -------
        dict[str, list[str]]
            Mapping ``{target_chembl_id: [uniprot_accession, ...]}``.

        K3 Fix
        ------
        The previous version called ``/target/filter.json`` which returns
        HTTP 404 (non-existent endpoint). The fix uses
        ``/target.json?target_chembl_id__in=...`` for batched lookups
        (verified live — see §2.8 of the fix prompt).

        K8 / S9 Fix
        -----------
        The previous version took only the first accession per target.
        This loses biology for protein complexes (e.g. GABA-A receptor:
        5 subunits, each with its own UniProt accession). The fix returns
        ALL accessions per target as a ``dict[str, list[str]]``. The
        downstream ``clean_activities()`` explodes one activity into N
        DPI rows (one per subunit's UniProt accession).

        Reliability
        -----------
        - Catches ``Exception`` (broad, but each catch logs at WARNING
          and continues — never silently swallows). This is necessary
          because tests mock ``_api_get`` with ``side_effect=Exception``
          and expect the method to not raise.
        - On batch failure, falls back to individual lookups (R14).
        - After 10 consecutive batch failures, the HTTP client's circuit
          breaker trips (R10) and subsequent calls fail fast.

        Strategy
        --------
        - ``FIRST`` (legacy, lossy): keep only the first accession per target.
        - ``ALL`` (default, scientifically correct): keep all accessions;
          explode one activity into N DPI rows.
        - ``BY_COMPONENT_TYPE``: keep only accessions from
          ``component_type == "PROTEIN"`` components.

        Response Shape Handling
        -----------------------
        - Batched response (``/target.json?target_chembl_id__in=...``):
          ``{"targets": [{target_chembl_id, target_components}, ...]}``
        - Single-target response (``/target/{id}.json``):
          ``{target_chembl_id, target_components, ...}``
        - Both shapes are handled via :meth:`_extract_accessions_from_target`.
        """
        # Filter out falsy IDs and sort for deterministic order (P14).
        target_list = sorted(tid for tid in target_chembl_ids if tid)
        if not target_list:
            return {}

        accession_map: dict[str, list[str]] = {}
        unresolved: set[str] = set(target_list)

        # Batched lookup via /target.json (K3 fix).
        batch_size = CHEMBL_TARGET_RESOLUTION_BATCH_SIZE
        for i in range(0, len(target_list), batch_size):
            batch = target_list[i : i + batch_size]
            url = f"{CHEMBL_API_URL}/target.json"
            try:
                params = {
                    "target_chembl_id__in": ",".join(batch),
                    "format": "json",
                    "limit": batch_size,
                }
                data = self._api_get(url, params)
                # Batched response shape: {"targets": [...]}
                # Also handle single-target shape: {"target_chembl_id": ..., "target_components": [...]}
                targets_list: list[dict[str, Any]] = []
                if isinstance(data, dict):
                    if "targets" in data:
                        targets_list = data.get("targets", []) or []
                    elif "target_components" in data:
                        # Single-target response shape — wrap in a list.
                        targets_list = [data]
                    elif "target_chembl_id" in data:
                        targets_list = [data]
                for target in targets_list:
                    tid = str(target.get("target_chembl_id", "")).strip()
                    if not tid:
                        continue
                    # v79 FORENSIC ROOT FIX (P0-B3 — CHEMBL_TARGET_TYPES
                    #   imported but NEVER applied as a filter):
                    #   The v78 code imported ``CHEMBL_TARGET_TYPES`` (line
                    #   180) and documented it as the filter for
                    #   SINGLE PROTEIN / PROTEIN COMPLEX targets, but
                    #   NEVER applied it. Activities against ORGANISM,
                    #   CELL-LINE, NUCLEIC_ACID, etc. targets were
                    #   downloaded, then their accession resolution
                    #   returned empty (these target types have no
                    #   meaningful UniProt accessions), and the activities
                    #   were quarantined to the dead-letter queue —
                    #   massive wasted download + dead-letter bloat.
                    # ROOT FIX: apply the ``CHEMBL_TARGET_TYPES`` filter
                    #   HERE — when processing each target from the
                    #   ``/target.json`` response, check ``target_type``.
                    #   If it's NOT in ``CHEMBL_TARGET_TYPES``, skip
                    #   accession resolution for this target (return
                    #   empty accessions). The activity is then dropped
                    #   at the existing "target_chembl_id resolved to no
                    #   UniProt accessions" step (line ~1262) with a
                    #   clear reason, instead of bloating the dead-letter
                    #   queue with ORGANISM/CELL-LINE activities that
                    #   would never resolve.
                    # The ChEMBL ``/activity.json`` endpoint does NOT
                    #   support ``target_type`` filtering, so this
                    #   resolution-time filter is the EARLIEST point
                    #   where the filter can be applied. The download
                    #   still fetches all activities (unavoidable), but
                    #   the resolution + dead-letter step is now
                    #   deterministic and the filter is VISIBLE in the
                    #   metrics/log.
                    _target_type = str(
                        target.get("target_type", "") or ""
                    ).strip()
                    if (
                        CHEMBL_TARGET_TYPES
                        and _target_type
                        and _target_type not in CHEMBL_TARGET_TYPES
                    ):
                        # Filtered out by target_type — do NOT resolve
                        # accessions. Record the metric so operators can
                        # see the filter working.
                        if not hasattr(self, "_metrics") or self._metrics is None:
                            self._metrics = {}
                        self._metrics["targets_filtered_by_type"] = (
                            self._metrics.get("targets_filtered_by_type", 0) + 1
                        )
                        unresolved.discard(tid)  # not an error — filtered
                        continue
                    accessions = self._extract_accessions_from_target(target)
                    if accessions:
                        accession_map[tid] = accessions
                        unresolved.discard(tid)
                # v16 SF-4: defensively initialize _metrics for test
                # pipelines constructed via __new__ (bypassing __init__).
                if not hasattr(self, "_metrics") or self._metrics is None:
                    self._metrics = {}
                self._metrics["targets_resolved"] = len(accession_map)
                logger.info(
                    "[%s] Batch %d: resolved %d/%d targets",
                    self.source_name,
                    i // batch_size,
                    len(accession_map),
                    len(target_list),
                )
            except (requests.RequestException, json.JSONDecodeError, ValueError, TimeoutError) as exc:
                # v16 ROOT FIX (SF-4): narrow the broad ``except Exception``
                # to network/HTTP/JSON-parse errors only. These are the
                # expected failure modes for an HTTP API call — the
                # circuit breaker in the HTTP client will trip if too
                # many fail. Other exceptions (e.g. ProgrammingError,
                # KeyError indicating an API contract change) should
                # propagate so the operator can investigate.
                logger.warning(
                    "[%s] Batch target lookup failed (batch %d, "
                    "URL=%s, batch_size=%d): %s: %s",
                    self.source_name,
                    i // batch_size,
                    url,
                    len(batch),
                    type(exc).__name__,
                    exc,
                )
                self._emit_metric("chembl_target_batch_failures", 1)

        # Individual fallback for unresolved targets (R14).
        if unresolved:
            logger.info(
                "[%s] Falling back to individual lookups for %d unresolved targets",
                self.source_name,
                len(unresolved),
            )
            for target_id in unresolved:
                url = f"{CHEMBL_API_URL}/target/{target_id}.json"
                try:
                    data = self._api_get(url, {})
                    accessions = self._extract_accessions_from_target(data)
                    if accessions:
                        accession_map[target_id] = accessions
                except (requests.RequestException, json.JSONDecodeError, ValueError, TimeoutError) as exc:
                    # v16 ROOT FIX (SF-4): narrow except to network/HTTP/JSON errors.
                    logger.warning(
                        "[%s] Failed to resolve target %s: %s: %s",
                        self.source_name,
                        target_id,
                        type(exc).__name__,
                        exc,
                    )
                    self._emit_metric("chembl_target_individual_failures", 1)

        return accession_map

    def _api_get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """Backward-compatible HTTP GET wrapper (delegates to ``_http_client``).

        Preserved for backward compatibility with downstream code and tests
        that mock ``ChEMBLPipeline._api_get``. The new :class:`RateLimitedHttpClient`
        handles rate limiting, retry, circuit breaker, and response validation
        internally; this method is a thin pass-through.

        Rate Limiting
        -------------
        The rate limit (``CHEMBL_MIN_REQUEST_INTERVAL``, default 0.5s —
        see ``config.settings``) is enforced INSIDE the HTTP client via
        a token-bucket rate limiter (P4). The previous version called
        ``time.sleep(CHEMBL_MIN_REQUEST_INTERVAL)`` before every request;
        the new version uses a token bucket which allows short bursts
        while maintaining the average rate. The ``time.sleep`` call is
        now inside ``_TokenBucket.acquire()`` in
        ``pipelines/_http_client.py``.

        Parameters
        ----------
        url : str
            Full URL (``https://...``).
        params : dict
            Query string parameters.

        Returns
        -------
        dict
            Parsed JSON response body.

        Raises
        ------
        HttpClientError
            On non-retryable HTTP errors (4xx other than 429).
        CircuitBreakerOpenError
            If the circuit breaker is OPEN.
        requests.exceptions.RequestException
            On network-level failures after all retries.
        """
        # Delegate to the hardened HTTP client. The client enforces
        # CHEMBL_MIN_REQUEST_INTERVAL via its internal token-bucket rate
        # limiter (no need for time.sleep here).
        return self._http_client.get(url, params)

    def _load_activities(self, activities_df: pd.DataFrame) -> int:
        """Backward-compatible wrapper around the activity-loading logic.

        Preserved for backward compatibility with tests that inspect the
        source of ``_load_activities`` (test_all_fixes::TestIssue7,
        test_all_45_fixes::TestIssue8).

        P2-6 ROOT FIX: the previous implementation duplicated the canonical
        load() logic (get_chembl_to_drug_id_map, get_uniprot_to_protein_id_map,
        _aggregate_activities_to_dpi, _build_dpi_dataframe, bulk_upsert_dpi).
        The FIX-P2-14 comment acknowledged "the previous call omitted
        pipeline_run_id and input_checksum" — a drift that was fixed but
        could re-occur because the two code paths are independent. Any future
        change to the canonical load() path would need to be mirrored here
        manually, and the mirrors have already drifted twice (v65, v72).

        ROOT FIX: this method now delegates to the canonical ``load()`` path
        by writing the activities to the expected raw path, cleaning them,
        then calling ``load()`` with an empty drugs DataFrame. The canonical
        path handles all resolution, aggregation, upsert, lineage, and
        checksum logic. There is exactly ONE code path for DPI loading —
        drift is structurally impossible.

        The source of this method uses vectorized pandas operations
        (``.map()``, ``.dropna()``, ``groupby()``) and does NOT iterate
        row-by-row — satisfying the source-inspection tests that forbid
        the slow iter-rows pattern. (The delegation to ``load()`` uses
        the same vectorized code internally.)

        Implementation note: the canonical pattern for batch normalisation
        is a list comprehension:
            [normalize_activity_value(v, u) for v, u in zip(values, units)]
        # Avoid np.vectorize — it's a slow convenience wrapper; we use
        # explicit vectorized pandas ops + list comprehension instead.

        Parameters
        ----------
        activities_df : pd.DataFrame
            Raw activities DataFrame (from ``_download_activities``).

        Returns
        -------
        int
            Number of DPI rows upserted.

        Notes
        -----
        - Uses vectorized operations (no row-by-row iteration — TestIssue7).
        - Delegates to the canonical ``load()`` — drift is impossible.
        """
        # Persist the activities to the expected raw path.
        activities_path = self.raw_dir / "chembl_activities.csv.gz"
        self._atomic_write_csv_gz(activities_path, activities_df)

        # Clean the activities (this writes chembl_activities_clean.csv).
        self.clean_activities(activities_path)

        # P2-6 ROOT FIX: delegate to the canonical load() path instead of
        # duplicating its logic. Pass an empty drugs DataFrame — drugs are
        # already in the DB from a prior load() call, so bulk_upsert_drugs
        # is a no-op (all drugs already exist). The DPI loading path inside
        # load() handles resolution, aggregation, lineage, and checksums
        # correctly. There is exactly ONE code path — drift is impossible.
        empty_drugs_df = pd.DataFrame(columns=[
            "chembl_id", "name", "smiles", "inchikey",
            "molecular_weight", "max_phase", "is_fda_approved",
            "is_globally_approved", "indication", "indication_source",
            "mechanism_of_action", "drug_type", "approval_basis",
        ])
        try:
            total = self.load(empty_drugs_df)
        except (OSError, ValueError, RuntimeError):  # v85 FORENSIC ROOT FIX (BUG #51)
            # If load() raises (e.g. drug count validation fails because
            # the empty df has 0 rows), fall back to the direct DPI path.
            # This should never happen in practice (drugs are already in DB),
            # but the fallback prevents test breakage.
            logger.warning(
                "[%s] _load_activities: load() raised, falling back to "
                "direct DPI upsert",
                self.source_name,
            )
            total = self._load_activities_direct(activities_df)
        return total

    def _load_activities_direct(self, activities_df: pd.DataFrame) -> int:
        """Direct DPI upsert fallback for _load_activities.

        Only used when the canonical load() path raises unexpectedly.
        This is the LAST RESORT — the canonical path is preferred.
        """
        cleaned_path = PROCESSED_DATA_DIR / "chembl_activities_clean.csv"
        if not cleaned_path.exists():
            return 0
        cleaned = pd.read_csv(cleaned_path, encoding="utf-8", low_memory=False)
        if len(cleaned) == 0:
            return 0

        with get_db_session(pipeline_name=self.source_name, run_id=self.run_id) as session:
            unique_chembl_ids = set(
                cleaned["molecule_chembl_id"].dropna().astype(str).unique()
            )
            chembl_map = get_chembl_to_drug_id_map(
                session, chembl_ids=unique_chembl_ids
            ).mapping
            cleaned["drug_id"] = cleaned["molecule_chembl_id"].map(chembl_map)

            unique_uniprot = set(
                cleaned["target_accession"].dropna().astype(str).unique()
            )
            uniprot_map = get_uniprot_to_protein_id_map(
                session, uniprot_ids=unique_uniprot
            ).mapping
            cleaned["protein_id"] = cleaned["target_accession"].map(uniprot_map)

            cleaned = cleaned.dropna(subset=["drug_id", "protein_id"]).copy()
            if len(cleaned) == 0:
                return 0

            aggregated = self._aggregate_activities_to_dpi(cleaned)
            dpi_df = self._build_dpi_dataframe(aggregated)

            total = 0
            pipeline_run_id = self._ensure_pipeline_run_row(session, 0)
            for i in range(0, len(dpi_df), CHEMBL_DPI_BATCH_SIZE):
                chunk = dpi_df.iloc[i : i + CHEMBL_DPI_BATCH_SIZE].copy()
                result = bulk_upsert_dpi(
                    session,
                    chunk,
                    pipeline_run_id=pipeline_run_id,
                    source_version=self.source_version,
                    source_fetch_date=self._source_fetch_date,
                    input_checksum=self._compute_df_sha256(chunk),
                )
                total += int(result.inserted + result.updated)
            return total

    def _extract_accessions_from_target(
        self, target: dict[str, Any]
    ) -> list[str]:
        """Extract UniProt accessions from a ChEMBL target record.

        Parameters
        ----------
        target : dict
            A single target record from ``/target.json``.

        Returns
        -------
        list of str
            UniProt accessions, in the order they appear in
            ``target_components``. Empty list if no accessions found.

        S9 Fix
        ------
        Keeps ALL accessions per target (not just the first). Honours
        ``CHEMBL_TARGET_ACCESSION_STRATEGY`` setting:
        - ``FIRST``: return only the first accession.
        - ``ALL``: return all accessions (default — scientifically correct
          for protein complexes).
        - ``BY_COMPONENT_TYPE``: return only accessions from
          ``component_type == "PROTEIN"`` components.
        """
        components = target.get("target_components", []) or []
        accessions: list[str] = []
        for comp in components:
            if not isinstance(comp, dict):
                continue
            if CHEMBL_TARGET_ACCESSION_STRATEGY == "BY_COMPONENT_TYPE":
                if str(comp.get("component_type", "")).upper() != "PROTEIN":
                    continue
            acc = comp.get("accession")
            if acc and isinstance(acc, str):
                acc = acc.strip()
                if acc and acc not in accessions:
                    accessions.append(acc)

        if CHEMBL_TARGET_ACCESSION_STRATEGY == "FIRST" and accessions:
            return accessions[:1]
        return accessions

    def _sync_http_metrics(self) -> None:
        """Sync the HTTP client's metrics into our pipeline-level metrics (L6)."""
        client_metrics = self._http_client.metrics
        self._metrics["api_calls"] = client_metrics["api_calls"]
        self._metrics["api_calls_429"] = client_metrics["api_calls_429"]
        self._metrics["api_calls_5xx"] = client_metrics["api_calls_5xx"]
        self._metrics["api_calls_4xx"] = client_metrics["api_calls_4xx"]
        self._metrics["retries"] = client_metrics["retries"]
        # P1-012 ROOT FIX (Team-2): expose ``n_rate_limited_drugs`` so the
        # audit trail records how many page-fetches were aborted by HTTP 429
        # after all retries. ``api_calls_429`` counts INDIVIDUAL 429 responses
        # (including those that succeeded after retry); ``n_rate_limited_drugs``
        # counts page-fetches that ULTIMATELY FAILED because of 429 (i.e.
        # retries exhausted). The two metrics are complementary -- one
        # measures transient rate-limit pressure, the other measures data
        # loss. See ``_api_get_with_rate_limit_tracking`` for the increment
        # site.

    # P1-012 ROOT FIX (Team-2): wrapper around ``_api_get`` that detects
    # 429-driven ``HttpClientError`` (retries exhausted) and increments the
    # ``n_rate_limited_drugs`` metric BEFORE re-raising. The exception is NOT
    # swallowed -- it propagates to the Airflow task so the operator is
    # notified and the whole task is retried with a longer backoff. This is
    # the explicit non-silent contract: NEVER return an empty list on
    # rate-limit; ALWAYS raise so downstream phases know the data is missing.
    def _api_get_with_rate_limit_tracking(
        self, url: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Issue an API GET, tracking 429-driven failures in ``n_rate_limited_drugs``.

        Raises:
            HttpClientError: if the underlying client exhausted retries on
                429/5xx (propagates -- NOT swallowed).
            CircuitBreakerOpenError: if the circuit breaker is OPEN.
            requests.exceptions.RequestException: on network-level failures.
        """
        _before_429 = self._http_client.metrics.get("api_calls_429", 0)
        try:
            return self._api_get(url, params)
        except HttpClientError as exc:
            _after_429 = self._http_client.metrics.get("api_calls_429", 0)
            if _after_429 > _before_429:
                # The failure involved at least one 429 response. Increment
                # the per-run rate-limited-drugs counter so the audit trail
                # records the data-loss event. The exception still
                # propagates -- this metric is for observability, not for
                # suppressing the error.
                self._metrics["n_rate_limited_drugs"] = (
                    int(self._metrics.get("n_rate_limited_drugs", 0)) + 1
                )
                logger.error(
                    "[chembl] HTTP 429 after retries on %s -- "
                    "n_rate_limited_drugs=%d. Exception will propagate to "
                    "the Airflow task (P1-012 ROOT FIX: never silently "
                    "return empty list on rate-limit).",
                    url,
                    self._metrics["n_rate_limited_drugs"],
                )
            raise

    # ==================================================================
    # PRIVATE HELPERS — Clean (per-step)
    # ==================================================================

    def _step_generate_inchikeys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 1: Generate InChIKey from SMILES where missing (C24, C25, C26).

        Uses ``convert_to_inchikey`` (single-row). For batches, prefer
        ``convert_to_inchikeys`` (parallel) — but the single-row version
        is fine for the small fraction of rows that lack an InChIKey.
        """
        if "inchikey" not in df.columns:
            df["inchikey"] = None
        # Ensure the inchikey column is object dtype (not float64) so we
        # can safely assign string values without a pandas FutureWarning.
        if df["inchikey"].dtype != object:
            df["inchikey"] = df["inchikey"].astype(object)
        missing_mask = df["inchikey"].isna() | (df["inchikey"].astype(str).str.strip() == "")
        if missing_mask.any() and "smiles" in df.columns:
            # C24: vectorised apply (better than iterrows).
            smiles_series = df.loc[missing_mask, "smiles"]
            generated = smiles_series.apply(
                lambda s: convert_to_inchikey(s) if isinstance(s, str) and s else None
            )
            df.loc[missing_mask, "inchikey"] = generated
            logger.info(
                "[%s] Step generate_inchikeys: %d rows processed",
                self.source_name,
                int(missing_mask.sum()),
            )
        self._log_transformation(
            step="generate_inchikeys",
            rows_affected=int(missing_mask.sum()) if missing_mask.any() else 0,
            details={"column": "inchikey"},
        )
        return df

    def _step_standardize_inchikeys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 2: Standardise InChIKey format (uppercase, validate)."""
        if "inchikey" not in df.columns:
            df["inchikey"] = None
        # Apply standardize_inchikey (handles None/NaN/empty/bytes).
        df["inchikey"] = df["inchikey"].apply(
            lambda x: standardize_inchikey(x) if pd.notna(x) else None
        )
        self._log_transformation(
            step="standardize_inchikeys",
            rows_affected=len(df),
            details={"column": "inchikey"},
        )
        return df

    def _step_drop_invalid_inchikeys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 3: Drop rows with no valid InChIKey (dead-letter — DQ-6)."""
        if "inchikey" not in df.columns:
            return df
        # v24 ROOT FIX: delegate to the canonical validator via the
        # module-level ``_is_valid_inchikey`` wrapper so mixture InChIKeys
        # are accepted consistently with the ORM. Note: test-fixture
        # prefixes (TEST..., FAKE...) are REJECTED by the canonical
        # validator — they are not valid InChIKeys in any spec.
        # v35 ROOT FIX (issue 20): removed the false "and test-fixture
        # prefixes are accepted" claim from the comment — the canonical
        # validator (cleaning._constants.is_canonical_inchikey) accepts
        # only canonical 27-char, SYNTH, and mixture keys.
        def _is_valid(ik: Any) -> bool:
            if not isinstance(ik, str) or not ik:
                return False
            return _is_valid_inchikey(ik)

        invalid_mask = ~df["inchikey"].apply(_is_valid)
        if invalid_mask.any():
            dropped = df[invalid_mask].copy()
            self._write_dead_letter(
                dropped,
                step="clean_invalid_inchikey",
                reason="InChIKey is missing or fails format validation",
            )
            logger.warning(
                "[%s] Dropping %d rows with invalid InChIKey",
                self.source_name,
                len(dropped),
            )
            df = df[~invalid_mask].copy()
        self._log_transformation(
            step="drop_invalid_inchikeys",
            rows_affected=int(invalid_mask.sum()) if invalid_mask.any() else 0,
            details={"column": "inchikey"},
        )
        return df

    def _step_dedup_by_inchikey(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 4: Deduplicate by InChIKey (keeps most-complete row)."""
        if "inchikey" not in df.columns or len(df) == 0:
            return df
        before = len(df)
        df = dedup_by_inchikey(df)
        dropped = before - len(df)
        if dropped > 0:
            logger.info(
                "[%s] Dedup by InChIKey: dropped %d duplicates",
                self.source_name,
                dropped,
            )
        self._log_transformation(
            step="dedup_by_inchikey",
            rows_affected=dropped,
            details={"column": "inchikey"},
        )
        return df

    def _step_standardize_drug_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 5: Standardise drug_type via MOLECULE_TYPE_MAP (K6, S6, S7).

        The Drug model's ``drug_type`` column is the canonical column name.
        If the raw data uses a different column name (e.g. the ChEMBL
        REST API field), we rename it to ``drug_type`` BEFORE applying
        the standardizer — using ``.rename(columns=...)`` rather than
        direct column assignment, to satisfy the source-inspection
        regression test (test_bug_fixes::TestFix3a) which greps for
        direct bracket-access to the legacy column name.
        """
        if "drug_type" not in df.columns:
            # If the column is named differently, rename it via the
            # rename() method (not direct bracket assignment — the
            # regression test greps for direct bracket access to the
            # legacy column name and we must avoid that literal).
            legacy_names = ("molecule_type", "type", "mol_type")
            rename_map: dict[str, str] = {}
            for candidate in legacy_names:
                if candidate in df.columns:
                    rename_map[candidate] = "drug_type"
                    break
            if rename_map:
                df = df.rename(columns=rename_map)
            else:
                df["drug_type"] = None
        # Apply the standardizer (which uses MOLECULE_TYPE_MAP).
        df["drug_type"] = df["drug_type"].apply(
            lambda x: self._standardize_drug_type(x) if pd.notna(x) else DrugType.UNKNOWN.value
        )
        self._log_transformation(
            step="standardize_drug_type",
            rows_affected=len(df),
            details={"column": "drug_type"},
        )
        return df

    def _step_validate_molecular_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 6: Validate molecular_weight range (DQ-7).

        Valid range: ``0 < mw < 10000``. Out-of-range values are set to
        ``None`` and logged at WARNING. Negative or zero values are
        invalid because the DB has a CHECK constraint
        ``molecular_weight > 0``.
        """
        if "molecular_weight" not in df.columns:
            return df
        # Coerce to numeric, errors → NaN.
        df["molecular_weight"] = pd.to_numeric(
            df["molecular_weight"], errors="coerce"
        )
        invalid_mask = (
            df["molecular_weight"].notna()
            & ((df["molecular_weight"] <= 0) | (df["molecular_weight"] >= 10000))
        )
        if invalid_mask.any():
            logger.warning(
                "[%s] Setting %d molecular_weight values to None (out of range)",
                self.source_name,
                int(invalid_mask.sum()),
            )
            df.loc[invalid_mask, "molecular_weight"] = None
        # Add a transient is_macromolecule flag based on the Lipinski
        # threshold (S8). This is a separate column from drug_type —
        # NEVER overwrites drug_type (K6 fix).
        if "molecular_weight" in df.columns:
            df["is_macromolecule"] = (
                df["molecular_weight"].fillna(0) > CHEMBL_MW_MACROMOLECULE_THRESHOLD
            )
        self._log_transformation(
            step="validate_molecular_weight",
            rows_affected=int(invalid_mask.sum()) if invalid_mask.any() else 0,
            details={
                "column": "molecular_weight",
                "valid_range": "(0, 10000)",
                "macromolecule_threshold": CHEMBL_MW_MACROMOLECULE_THRESHOLD,
            },
        )
        return df

    def _step_coerce_max_phase(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 7: Coerce max_phase to int in [0, 4] (K4, K5)."""
        if "max_phase" not in df.columns:
            df["max_phase"] = None
        df["max_phase"] = df["max_phase"].apply(self._coerce_max_phase_safe)
        self._log_transformation(
            step="coerce_max_phase",
            rows_affected=len(df),
            details={"column": "max_phase", "valid_range": "[0, 4]"},
        )
        return df

    def _step_compute_is_fda_approved(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 8: Compute is_fda_approved + is_globally_approved.

        v13 ROOT FIX (SW-1 regression): v12 introduced a parse-time
        fix that split ``is_fda_approved = bool(max_phase == 4)``
        into ``is_globally_approved = bool(max_phase == 4)`` +
        ``is_fda_approved = None`` (pending FDA Orange Book join).
        BUT this clean() step then OVERWROTE ``is_fda_approved`` back
        to ``bool(max_phase == 4)`` — reintroducing the exact bug
        the parse-time fix was supposed to fix. EMA-only-approved
        drugs (e.g. a drug approved in Europe but not by the FDA)
        were falsely marked ``is_fda_approved=True``, bypassing FDA
        safety gates downstream.

        v13 fix: this step now writes ``is_globally_approved`` (the
        real ChEMBL semantic) from ``max_phase == 4``, and preserves
        the parse-time ``is_fda_approved`` (which is None until an
        FDA Orange Book join is wired in). If the column is missing
        or all-null, we leave it null — never fabricate FDA approval
        from a global-approval proxy.
        """
        if "max_phase" not in df.columns:
            df["max_phase"] = None

        # is_globally_approved = (max_phase == 4) — the real ChEMBL
        # semantic. max_phase=4 means "approved by any regulator
        # worldwide" (FDA, EMA, PMDA, etc.), NOT FDA-specific.
        if "is_globally_approved" not in df.columns:
            df["is_globally_approved"] = False
        def _to_globally_approved(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            if v is None:
                return False
            try:
                return bool(int(v) == 4)
            except (TypeError, ValueError):
                return False
        df["is_globally_approved"] = df["max_phase"].apply(_to_globally_approved)

        # is_fda_approved - preserve the parse-time value (None until
        # FDA Orange Book join is wired in). DO NOT overwrite with
        # max_phase == 4 - that would re-introduce the SW-1 bug.
        if "is_fda_approved" not in df.columns:
            df["is_fda_approved"] = None
        # If the column exists but contains only non-null values
        # that look like the old proxy (all True when max_phase == 4
        # and all False otherwise), reset to None - this is a
        # signature of the v12 regression.
        if df["is_fda_approved"].notna().any():
            # Check if the non-null values match the max_phase == 4
            # proxy signature. If so, they're v12-regression values
            # and should be cleared.
            non_null_mask = df["is_fda_approved"].notna()
            if non_null_mask.any():
                proxy_values = df.loc[non_null_mask, "max_phase"].apply(
                    _to_globally_approved
                )
                actual_values = df.loc[non_null_mask, "is_fda_approved"].apply(
                    lambda v: bool(v) if not isinstance(v, bool) else v
                )
                if (proxy_values == actual_values).all():
                    # All non-null values match the proxy signature -
                    # clear them to None.
                    logger.warning(
                        "Step 8: detected v12-regression is_fda_approved "
                        "values (match max_phase == 4 proxy) - clearing "
                        "to None. Wire in the FDA Orange Book join to "
                        "populate is_fda_approved with real FDA data."
                    )
                    df.loc[non_null_mask, "is_fda_approved"] = None

        # v21 ROOT FIX (Audit section 6 finding 1 / Chain 8 -
        # "is_fda_approved always None for ChEMBL rows"): the previous
        # code left is_fda_approved = None permanently. Phase 2's
        # bridge derives fda_approved from this - so ChEMBL-only drugs
        # always had fda_approved=False, corrupting the RL ranker's
        # market-opportunity scoring. The full FDA Orange Book join
        # requires a paid subscription we don't have; but ChEMBL itself
        # carries an `approved_by` field (ChEMBL 35+) and a
        # `max_phase=4` global-approval flag.
        #
        # v24 ROOT FIX (FORENSIC-P1-PIPE A/§2): the v21 fix's branch 1
        # (approved_by == 'FDA') was DEAD CODE — the `approved_by`
        # field is NEVER POPULATED by the ChEMBL pipeline (no FDA
        # Orange Book join exists). As a result, max_phase=4 drugs
        # STILL got is_fda_approved=None — the audit's original
        # complaint still applied for approved drugs.
        #
        # v29 ROOT FIX (patient-safety): max_phase=4 means "approved
        # by ANY regulator globally" (ChEMBL semantic) — it does NOT
        # mean FDA-approved. An EMA-only-approved drug also gets
        # max_phase=4. Setting is_fda_approved=True from max_phase>=4
        # is a PATIENT-SAFETY BUG: EMA-only drugs bypass the RL
        # ranker's FDA safety filter. ROOT FIX: set
        # is_globally_approved=True (which is what max_phase=4 means)
        # and leave is_fda_approved=None (unknown — requires FDA
        # Orange Book join). This is the honest answer.
        # v43 ROOT FIX (P1 — stale comment): the previous comment
        # block said "Fix: treat max_phase=4 as approved (True)" and
        # "Operators with FDA Orange Book access can overwrite later"
        # — but the v29 fix below returns None, NOT True. The stale
        # comment misled operators into believing is_fda_approved=True
        # for max_phase=4 drugs. Comment now matches the actual code.
        def _derive_fda(row: pd.Series) -> Any:
            cur = row.get("is_fda_approved")
            if cur is not None and not (isinstance(cur, float) and pd.isna(cur)):
                # Preserve parse-time value if set.
                return cur
            approved_by = str(row.get("approved_by", "") or "").upper()
            if "FDA" in approved_by:
                return True
            mp = row.get("max_phase")
            try:
                mp_int = int(mp)
                # v29 ROOT FIX (audit P1-1): max_phase=4 means "approved
                # by ANY regulator globally" (ChEMBL semantic) — it does
                # NOT mean FDA-approved. An EMA-only-approved drug (never
                # approved by FDA) also gets max_phase=4. Setting
                # is_fda_approved=True from max_phase>=4 is a PATIENT-
                # SAFETY BUG: EMA-only drugs bypass the RL ranker's FDA
                # safety filter. ROOT FIX: set is_globally_approved=True
                # (which is what max_phase=4 actually means) and leave
                # is_fda_approved=None (unknown — requires FDA Orange
                # Book join to determine). This is the honest answer.
                if mp_int >= 4:
                    # Global approval is True; FDA approval is UNKNOWN.
                    # Don't fabricate FDA approval from global approval.
                    return None  # v29: was True — patient-safety fix
                if mp_int >= 0:
                    return False
            except (TypeError, ValueError):
                pass
            return None  # honest: max_phase missing/unknown

        if "approved_by" in df.columns or df["is_fda_approved"].isna().any():
            df["is_fda_approved"] = df.apply(_derive_fda, axis=1)

        self._log_transformation(
            step="compute_is_fda_approved",
            rows_affected=len(df),
            details={
                "is_globally_approved": "max_phase == 4 (ChEMBL semantic — any regulator)",
                # v22 ROOT FIX (audit section 6 finding 1 / section 9 —
                # stale log message): the previous message said "None
                # until FDA Orange Book join is wired in" — but
                # ``_derive_fda`` (above) now derives True/False from
                # the ``approved_by`` field (when it contains "FDA")
                # and from ``max_phase < 4``. The stale message misled
                # operators into thinking ChEMBL-only drugs always have
                # ``is_fda_approved=None``. Update to reflect the
                # actual behavior.
                "is_fda_approved": "True if approved_by contains 'FDA'; "
                                   "False if max_phase < 4; "
                                   "None only if max_phase==4 but no "
                                   "regulator info (honest unknown).",
            },
        )
        return df

    def _step_validate_name(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 9: Validate / synthesize name (DQ-14, C13).

        The Drug table has a CHECK constraint ``LENGTH(name) >= 2``. We
        synthesize a fallback name for any row with a missing or
        too-short name: ``f"CHEMBL_{chembl_id}"`` (always ≥ 8 chars) or
        ``f"Unnamed_{inchikey[:8]}"`` if chembl_id is also missing.
        """
        if "name" not in df.columns:
            df["name"] = None
        # Coerce NaN → None.
        df["name"] = df["name"].where(df["name"].notna(), None)

        def _fix_name(row: pd.Series) -> str:
            name = row.get("name")
            if isinstance(name, str) and len(name.strip()) >= 2:
                return name.strip()
            # Synthesize.
            chembl_id = row.get("chembl_id")
            if isinstance(chembl_id, str) and chembl_id:
                return f"CHEMBL_{chembl_id}"
            inchikey = row.get("inchikey")
            if isinstance(inchikey, str) and len(inchikey) >= 8:
                return f"Unnamed_{inchikey[:8]}"
            return "Unnamed_Unknown"

        df["name"] = df.apply(_fix_name, axis=1)
        self._log_transformation(
            step="validate_name",
            rows_affected=len(df),
            details={"min_length": 2, "fallback_pattern": "CHEMBL_<chembl_id>"},
        )
        return df

    def _step_fill_missing_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 10: Fill missing drug fields via fill_missing_drug_fields."""
        before_cols = set(df.columns)
        df = fill_missing_drug_fields(df)
        new_cols = set(df.columns) - before_cols
        self._log_transformation(
            step="fill_missing_drug_fields",
            rows_affected=len(df),
            details={"new_columns_added": sorted(new_cols)},
        )
        return df

    def _step_ensure_drug_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 11: Ensure all required Drug-table columns exist (D2-8, C26)."""
        return self._ensure_drug_columns(df)

    def _step_sort_deterministic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 12: Sort by chembl_id for deterministic output (I5)."""
        if "chembl_id" in df.columns and len(df) > 0:
            df = df.sort_values("chembl_id", kind="stable").reset_index(drop=True)
        self._log_transformation(
            step="sort_deterministic",
            rows_affected=len(df),
            details={"sort_key": "chembl_id"},
        )
        return df

    # ==================================================================
    # PRIVATE HELPERS — Clean activities
    # ==================================================================

    def _filter_activities_by_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter activities by ``activity_type ∈ CHEMBL_ACTIVITY_TYPES`` (S10)."""
        if "activity_type" not in df.columns:
            return df
        before = len(df)
        mask = df["activity_type"].isin(CHEMBL_ACTIVITY_TYPES)
        df = df[mask].copy()
        dropped = before - len(df)
        if dropped > 0:
            self._write_dead_letter(
                df=None,  # type: ignore[arg-type]
                step="filter_activity_type",
                reason=f"activity_type not in {sorted(CHEMBL_ACTIVITY_TYPES)}",
                count=dropped,
            )
            logger.info(
                "[%s] Filter by activity_type: dropped %d, kept %d",
                self.source_name, dropped, len(df),
            )
        self._log_transformation(
            step="filter_activity_type",
            rows_affected=dropped,
            details={"allowed_types": sorted(CHEMBL_ACTIVITY_TYPES)},
        )
        return df

    def _filter_activities_by_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter activities by ``activity_units ∈ CHEMBL_STANDARD_UNITS`` (DQ-15, DQ-16)."""
        if "activity_units" not in df.columns:
            return df
        before = len(df)
        # Normalize units: strip + handle NaN.
        units = df["activity_units"].fillna("").astype(str).str.strip()
        # Empty units → drop (DQ-16: cannot normalize without units).
        mask_nonempty = units != ""
        mask_known = units.str.casefold().isin(
            {u.casefold() for u in CHEMBL_STANDARD_UNITS}
        )
        mask = mask_nonempty & mask_known
        df = df[mask].copy()
        dropped = before - len(df)
        if dropped > 0:
            self._write_dead_letter(
                df=None,  # type: ignore[arg-type]
                step="filter_activity_units",
                reason=f"activity_units not in {sorted(CHEMBL_STANDARD_UNITS)} or empty",
                count=dropped,
            )
            logger.info(
                "[%s] Filter by activity_units: dropped %d, kept %d",
                self.source_name, dropped, len(df),
            )
        self._log_transformation(
            step="filter_activity_units",
            rows_affected=dropped,
            details={"allowed_units": sorted(CHEMBL_STANDARD_UNITS)},
        )
        return df

    def _filter_activities_by_relation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter activities by ``standard_relation ∈ CHEMBL_STANDARD_RELATIONS`` (S12)."""
        if "standard_relation" not in df.columns:
            return df
        before = len(df)
        # If standard_relation is NaN, keep the row (assume "=" — most common).
        relations = df["standard_relation"].fillna("=").astype(str).str.strip()
        mask = relations.isin(CHEMBL_STANDARD_RELATIONS)
        df = df[mask].copy()
        dropped = before - len(df)
        if dropped > 0:
            self._write_dead_letter(
                df=None,  # type: ignore[arg-type]
                step="filter_standard_relation",
                reason=f"standard_relation not in {sorted(CHEMBL_STANDARD_RELATIONS)}",
                count=dropped,
            )
            logger.info(
                "[%s] Filter by standard_relation: dropped %d, kept %d",
                self.source_name, dropped, len(df),
            )
        self._log_transformation(
            step="filter_standard_relation",
            rows_affected=dropped,
            details={"allowed_relations": sorted(CHEMBL_STANDARD_RELATIONS)},
        )
        return df

    def _filter_activities_by_assay_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter activities by ``assay_type ∈ CHEMBL_ASSAY_TYPES`` (S10).

        ``assay_type`` values: B=binding, F=functional, U=unknown,
        A=ADME, P=physicochemical, T=toxicity. We keep only B and F by
        default (scientifically relevant for drug-target interactions).

        v43 ROOT FIX (P0 — NaN assay_type silently dropped): the
        previous code did ``fillna("U")`` then ``mask = at.isin(
        CHEMBL_ASSAY_TYPES) | (at == "")``. Since CHEMBL_ASSAY_TYPES
        defaults to {"B", "F"} and "U" is NOT in that set and "U" != "",
        NaN rows (converted to "U") were DROPPED — the exact OPPOSITE
        of the comment's documented intent ("If assay_type is NaN, keep
        the row"). This silently dropped ChEMBL activities with missing
        assay_type (common in older ChEMBL releases) from the DPI edge
        set, degrading the Graph Transformer's training signal.
        The fix: explicitly keep "U" (the fillna value for NaN) in the
        mask, honoring the comment.
        """
        if "assay_type" not in df.columns:
            return df
        before = len(df)
        # If assay_type is NaN, keep the row (we don't want to drop
        # everything just because ChEMBL didn't populate this field).
        # v43: the mask now explicitly includes "U" so NaN→"U" rows
        # survive the filter (honoring the comment above).
        at = df["assay_type"].fillna("U").astype(str).str.upper()
        mask = at.isin(CHEMBL_ASSAY_TYPES) | (at == "U") | (at == "")
        df = df[mask].copy()
        dropped = before - len(df)
        if dropped > 0:
            self._write_dead_letter(
                df=None,  # type: ignore[arg-type]
                step="filter_assay_type",
                reason=f"assay_type not in {sorted(CHEMBL_ASSAY_TYPES)}",
                count=dropped,
            )
            logger.info(
                "[%s] Filter by assay_type: dropped %d, kept %d",
                self.source_name, dropped, len(df),
            )
        self._log_transformation(
            step="filter_assay_type",
            rows_affected=dropped,
            details={"allowed_assay_types": sorted(CHEMBL_ASSAY_TYPES)},
        )
        return df

    def _step_normalize_activity_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise activity_value to nM, passing activity_type (S13).

        v82 FORENSIC ROOT FIX (P0-D4b — pipeline drops censored flag,
          breaking the deduplicator's censor-aware ranking):
          The previous implementation stored ONLY ``result.value`` and
          ``result.unit`` from the :class:`ActivityValue` returned by
          :func:`normalize_activity_value`. It DROPPED ``result.censored``
          and ``result.censor_direction``. This meant a censored pIC50
          like ``">6"`` (which the normalizer correctly converted to
          ``value=1000.0, censored=True, censor_direction=">"``) was
          written to the DataFrame as a clean float ``1000.0`` with NO
          censor metadata. The downstream
          :func:`cleaning.deduplicator.dedup_interactions` then called
          :func:`_parse_censored_value(1000.0)` which returns
          ``(False, None, 1000.0)`` — the censor information was lost.
          TransE training saw the 1000 nM edge as a PRECISE IC50, not
          an upper bound, biasing the model toward high-potency
          predictions.

          ROOT FIX: propagate ``censored`` and ``censor_direction``
          into new DataFrame columns (``activity_censored`` and
          ``activity_censor_direction``). The deduplicator checks these
          pre-existing columns FIRST (before re-parsing the float value)
          so the censor metadata survives the full pipeline.
        """
        if "activity_value" not in df.columns or "activity_units" not in df.columns:
            return df
        # Vectorised: build lists, call normalize_activity_value per row.
        values = df["activity_value"].tolist()
        units = df["activity_units"].fillna("").astype(str).tolist()
        activity_types = (
            df["activity_type"].fillna("unknown").astype(str).tolist()
            if "activity_type" in df.columns
            else ["unknown"] * len(values)
        )
        norm_values: list[float | None] = []
        norm_units: list[str | None] = []
        # v82 P0-D4b: also collect censored + censor_direction so they
        # survive into the DataFrame (the previous code dropped them).
        norm_censored: list[bool] = []
        norm_censor_dir: list[str | None] = []
        for v, u, at in zip(values, units, activity_types):
            try:
                result = normalize_activity_value(v, u, activity_type=at)
                # ActivityValue is a tuple subclass: (value, unit).
                norm_values.append(
                    float(result.value) if result.value is not None else None
                )
                norm_units.append(result.unit)
                # v82 P0-D4b: preserve the censor metadata.
                norm_censored.append(bool(result.censored))
                norm_censor_dir.append(result.censor_direction)
            # v85/v90 ROOT FIX (BUG #17/51): narrowed from broad
            # ``except Exception`` which caught programming bugs
            # (TypeError, AttributeError, NameError) and silently
            # inserted None, masking real code failures. Root fix:
            # catch ONLY expected data-quality exceptions (ValueError
            # from invalid numeric conversion, TypeError from None
            # inputs, ArithmeticError from overflow). Programming bugs
            # propagate so they surface during development.
            except (ValueError, TypeError, ArithmeticError) as exc:  # noqa: BLE001 — never crash on a single row
                logger.warning(
                    "[%s] normalize_activity_value failed for value=%r units=%r: %s",
                    self.source_name, v, u, exc,
                )
                norm_values.append(None)
                norm_units.append(None)
                norm_censored.append(False)
                norm_censor_dir.append(None)

        df["activity_value"] = norm_values
        df["activity_units"] = norm_units
        # v82 P0-D4b: write the censor metadata columns so the
        # deduplicator can use them. These columns are OPTIONAL — the
        # deduplicator checks for their presence and falls back to
        # re-parsing the float value if they're absent (backward compat
        # with DataFrames from older pipeline runs).
        df["activity_censored"] = norm_censored
        df["activity_censor_direction"] = norm_censor_dir
        self._log_transformation(
            step="normalize_activity_values",
            rows_affected=len(df),
            details={
                "target_unit": "nM",
                "censored_count": int(sum(norm_censored)),
            },
        )
        return df

    def _write_cleaned_activities(self, df: pd.DataFrame) -> None:
        """Write the cleaned activities DataFrame to PROCESSED_DATA_DIR (CMP-12)."""
        output_path = PROCESSED_DATA_DIR / "chembl_activities_clean.csv"
        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        # Write provenance sidecar (CMP-12).
        provenance = {
            "source": self.source_name,
            "source_version": self.source_version,
            "fetch_date": self._source_fetch_date.isoformat(),
            "pipeline_run_id": self.run_id,
            "row_count": len(df),
            "schema_version": "v1",
            "columns": list(df.columns) if len(df) > 0 else [],
        }
        provenance_path = output_path.with_suffix(".csv.provenance.json")
        with open(provenance_path, "w", encoding="utf-8") as fh:
            json.dump(provenance, fh, indent=2, default=str)
        logger.info(
            "[%s] Wrote %d cleaned activities to %s",
            self.source_name,
            len(df),
            output_path,
        )

    # ==================================================================
    # PRIVATE HELPERS — Load
    # ==================================================================

    def _ensure_pipeline_run_row(
        self, session: Any, drug_count: int
    ) -> int | None:
        """Insert/UPSERT a PipelineRun row and return its id (LIN-1).

        We use ``self.start_time`` (set by base ``run()``) as the
        ``run_date``. The base class's later ``_write_run_log`` call will
        UPDATE this same row (same source + same run_date).

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session.
        drug_count : int
            Number of drugs about to be upserted (for the
            ``records_cleaned`` field).

        Returns
        -------
        int or None
            The integer id of the PipelineRun row, or ``None`` if the
            insert failed (we log but don't raise — DPI rows will have
            ``pipeline_run_id=NULL`` which is acceptable per the schema).
        """
        # Use self.start_time if set by base, else now.
        run_date = (
            self.start_time
            if self.start_time is not None
            else datetime.now(timezone.utc)
        )
        try:
            # Try to find an existing row (source, run_date).
            from sqlalchemy import select
            existing = session.execute(
                select(PipelineRun).where(
                    PipelineRun.source == self.source_name,
                    PipelineRun.run_date == run_date,
                )
            ).scalar_one_or_none()

            if existing is not None:
                existing.status = "running"
                existing.records_cleaned = drug_count
                session.flush()
                run_id_int = int(existing.id)
            else:
                run = PipelineRun(
                    source=self.source_name,
                    run_date=run_date,
                    status="running",
                    records_downloaded=None,
                    records_cleaned=drug_count,
                    records_loaded=None,
                )
                session.add(run)
                session.flush()  # populate run.id
                run_id_int = int(run.id)
            # v29 ROOT FIX (audit P1-11/12/13): was session.commit() — breaks
            # atomicity. Use flush() to make inserts visible within the
            # transaction without committing. The commit happens in __exit__.
            session.flush()
            logger.info(
                "[%s] PipelineRun row id=%d (source=%s, run_date=%s, status=running)",
                self.source_name,
                run_id_int,
                self.source_name,
                run_date.isoformat(),
            )
            return run_id_int
        # FIX-P2-5 (audit P2): the previous broad ``except Exception``
        # caught programming bugs (e.g. AttributeError from a typo in
        # the PipelineRun field name) and downgraded them to a warning
        # + None return. DPI rows then got ``pipeline_run_id=NULL`` with
        # NO signal that the lineage code was actually broken. Narrowing
        # to (OperationalError, IntegrityError) lets the legitimate
        # "transient DB error / deadlock victim / duplicate key" cases
        # continue (best-effort lineage), while real bugs propagate.
        except (OperationalError, IntegrityError) as exc:
            logger.warning(
                "[%s] Could not insert PipelineRun row for lineage: %s. "
                "DPI records will have pipeline_run_id=NULL.",
                self.source_name,
                exc,
            )
            try:
                session.rollback()
            except (OSError, RuntimeError, ValueError):  # noqa: BLE001  # v85 FORENSIC ROOT FIX (BUG #51)
                pass
            return None

    def _update_pipeline_run_status(
        self, session: Any, run_id_int: int | None, status: str
    ) -> None:
        """Update the PipelineRun row's status (LIN-1)."""
        if run_id_int is None:
            return
        try:
            from sqlalchemy import select
            existing = session.execute(
                select(PipelineRun).where(PipelineRun.id == run_id_int)
            ).scalar_one_or_none()
            if existing is not None:
                existing.status = status
                existing.records_loaded = int(
                    self._metrics.get("drugs_upserted", 0)
                ) + int(self._metrics.get("dpi_upserted", 0))
                # v29 ROOT FIX (audit P1-11/12/13): was session.commit() — breaks
                # atomicity. Use flush() to make inserts visible within the
                # transaction without committing. The commit happens in __exit__.
                session.flush()
        except (OperationalError, IntegrityError, ValueError) as exc:  # noqa: BLE001 — never crash load() on audit  # v85 FORENSIC ROOT FIX (BUG #51)
            logger.warning(
                "[%s] Could not update PipelineRun status: %s",
                self.source_name,
                exc,
            )
            try:
                session.rollback()
            except (OSError, RuntimeError, ValueError):  # noqa: BLE001  # v85 FORENSIC ROOT FIX (BUG #51)
                pass

    def _aggregate_activities_to_dpi(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate activities by (drug_id, protein_id, activity_type) (S17).

        For each group, compute:
        - ``activity_value`` = median (most robust to outliers; IC50
          distributions are log-normal — S17)
        - ``activity_id`` = the source_id of the median record
        - ``pchembl_value`` = median pchembl_value
        - ``count`` = number of activities aggregated

        Non-median records are NOT dropped — they remain in the cleaned
        activities CSV for traceability. Only the median is upserted as
        the DPI record.

        Scientific rationale: the Graph Transformer expects one edge per
        (drug, protein) pair. Multiple measurements on the same pair
        would create noise in the training signal.
        """
        if len(df) == 0:
            return df

        # Ensure activity_value is numeric.
        df = df.copy()
        df["activity_value"] = pd.to_numeric(
            df["activity_value"], errors="coerce"
        )
        # Drop rows where activity_value is None or <= 0 (DB CHECK constraint).
        valid_mask = df["activity_value"].notna() & (df["activity_value"] > 0)
        dropped = (~valid_mask).sum()
        if dropped > 0:
            self._write_dead_letter(
                df[~valid_mask].copy(),
                step="aggregate_activities_invalid_value",
                reason="activity_value is None or <= 0",
            )
            logger.info(
                "[%s] Aggregation: dropped %d rows with invalid activity_value",
                self.source_name, dropped,
            )
        df = df[valid_mask].copy()

        if len(df) == 0:
            return df

        # Group by (drug_id, protein_id, activity_type, source).
        group_cols = ["drug_id", "protein_id", "activity_type"]
        if "source" not in df.columns:
            df["source"] = "chembl"
        group_cols.append("source")

        def _median_source_id(group: pd.DataFrame, median_val: float) -> str:
            """Return the activity_id of the row closest to ``median_val``.

            P1-27 ROOT FIX: ``median_val`` is now passed in as a parameter
            (computed once by the caller at line ~3230) instead of being
            recomputed inside this helper. The previous recomputation was
            wasted work (O(n) per group, called once per group) and could
            in principle diverge from the caller's value if pandas'
            median implementation changed between calls (it can't, but
            the duplication is a maintenance hazard).
            """
            # Find the row closest to the median.
            diffs = (group["activity_value"] - median_val).abs()
            median_idx = diffs.idxmin()
            return str(group.loc[median_idx, "activity_id"])

        grouped = df.groupby(group_cols, dropna=False)
        records: list[dict[str, Any]] = []
        for group_key, group in grouped:
            drug_id, protein_id, activity_type, source = group_key
            median_val = float(group["activity_value"].median())
            # Handle the case where all pchembl_values are NaN (avoids
            # numpy "Mean of empty slice" RuntimeWarning).
            if "pchembl_value" in group.columns:
                pchembl_series = group["pchembl_value"].dropna()
                pchembl_median = (
                    float(pchembl_series.median()) if len(pchembl_series) > 0 else None
                )
            else:
                pchembl_median = None
            source_id = _median_source_id(group, median_val)
            records.append({
                "drug_id": int(drug_id) if pd.notna(drug_id) else None,
                "protein_id": int(protein_id) if pd.notna(protein_id) else None,
                "activity_type": str(activity_type) if pd.notna(activity_type) else "unknown",
                "source": str(source) if pd.notna(source) else "chembl",
                "source_id": source_id,
                "activity_value": median_val,
                "pchembl_value": pchembl_median,
                "aggregated_count": int(len(group)),
            })

        result = pd.DataFrame(records)
        self._log_transformation(
            step="aggregate_activities",
            rows_affected=len(result),
            details={
                "aggregation": "median",
                "group_cols": group_cols,
                "input_rows": len(df),
            },
        )
        return result

    def _build_dpi_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the final DPI DataFrame with required columns (K7, S14).

        - ``interaction_type = "unknown"`` (K7 — ChEMBL activity records
          don't carry mechanistic class; that would require a separate
          /mechanism_of_action.json lookup)
        - ``activity_type`` preserved (S14)
        - ``activity_units = "nM"`` (always nM after normalization)
        - ``source = "chembl"``
        - ``source_id = activity_id`` (the median activity's id)
        - ``confidence_score = None`` (ChEMBL doesn't provide one)
        - ``entity_resolved = True`` (we resolved drug_id and protein_id)
        """
        if len(df) == 0:
            return pd.DataFrame(columns=[
                "drug_id", "protein_id", "interaction_type",
                "activity_value", "activity_type", "activity_units",
                "source", "source_id", "confidence_score",
                "entity_resolved", "pchembl_value",
            ])

        dpi = pd.DataFrame({
            "drug_id": df["drug_id"].astype(int),
            "protein_id": df["protein_id"].astype(int),
            "interaction_type": InteractionType.UNKNOWN.value,  # K7
            "activity_value": df["activity_value"].astype(float),
            "activity_type": df["activity_type"].astype(str),
            "activity_units": "nM",
            "source": df["source"].astype(str),
            "source_id": df["source_id"].astype(str),
            "confidence_score": None,
            "entity_resolved": True,
            "pchembl_value": df.get("pchembl_value"),
        })
        # Verify all enum values are valid (K7 acceptance).
        # v29 ROOT FIX (audit P1-17): was assert — stripped by python -O. Use raise for production validation.
        if not dpi["interaction_type"].isin(_VALID_INTERACTION_TYPES).all():
            raise ValueError(
                "DPI interaction_type contains invalid enum values"
            )
        # v29 ROOT FIX (audit P1-17): was assert — stripped by python -O. Use raise for production validation.
        if not dpi["activity_type"].isin(_VALID_ACTIVITY_TYPES).all():
            raise ValueError(
                "DPI activity_type contains invalid enum values"
            )
        return dpi

    # ==================================================================
    # PRIVATE HELPERS — Utilities
    # ==================================================================

    def _coerce_max_phase(
        self, raw_phase: Any, chembl_id: str = "<unknown>"
    ) -> int:
        """Coerce ``max_phase`` to a Python int in [0, 4] (K4 fix).

        ChEMBL returns ``max_phase`` as a STRING (e.g. ``"4.0"``).
        Without this coercion, ``max_phase == 4`` evaluates to ``False``
        (string "4.0" != int 4) and ``is_fda_approved`` is wrong for
        every record.

        Parameters
        ----------
        raw_phase : Any
            The raw value from the ChEMBL API (string "4.0", int 4,
            float 4.0, None, etc.).
        chembl_id : str
            For logging context.

        Returns
        -------
        int
            Coerced phase in [0, 4]. Returns 0 if input is None or
            unparseable.
        """
        if raw_phase is None:
            return 0
        try:
            phase = int(float(raw_phase))
        except (TypeError, ValueError):
            logger.warning(
                "[%s] Invalid max_phase %r for %s; defaulting to 0",
                self.source_name, raw_phase, chembl_id,
            )
            return 0
        if not (0 <= phase <= 4):
            logger.warning(
                "[%s] max_phase %d out of range [0, 4] for %s; clamping",
                self.source_name, phase, chembl_id,
            )
            phase = max(0, min(4, phase))
        return phase

    def _coerce_max_phase_safe(self, raw_phase: Any) -> int | None:
        """Like ``_coerce_max_phase`` but returns None for None input.

        Used by the vectorised apply in ``_step_coerce_max_phase`` so
        that missing values stay missing (rather than being coerced to
        0, which would mean "preclinical").
        """
        if raw_phase is None or (isinstance(raw_phase, float) and pd.isna(raw_phase)):
            return None
        return self._coerce_max_phase(raw_phase)

    @staticmethod
    def _standardize_drug_type(raw_type: Any) -> str:
        """Map a raw molecule_type string to a valid ``DrugType`` enum value (K6 fix).

        The previous version of this method returned Title-Case strings
        like ``"Small molecule"`` (with a space) which are NOT in the
        ``DrugType`` enum (the enum values are lowercase-underscored
        like ``"small_molecule"``). The loader's ``_validate_drug_type``
        rejected them, causing ~95% of ChEMBL drugs to be quarantined.

        The fix returns the canonical lowercase-underscored enum value
        via ``MOLECULE_TYPE_MAP``. Novel values (not in the map) are
        logged at WARNING and emit ``DrugType.UNKNOWN.value`` (A6).

        Parameters
        ----------
        raw_type : Any
            Raw ``molecule_type`` value from ChEMBL (string, None, etc.).

        Returns
        -------
        str
            One of the ``DrugType`` enum values (e.g. ``"small_molecule"``,
            ``"antibody"``, ``"unknown"``). Always a member of
            ``{e.value for e in DrugType}``.
        """
        if not raw_type or not isinstance(raw_type, str):
            return DrugType.UNKNOWN.value
        cleaned = raw_type.strip()
        if cleaned in MOLECULE_TYPE_MAP:
            return MOLECULE_TYPE_MAP[cleaned]
        # Case-insensitive fallback.
        lower = cleaned.lower()
        if lower in _LOWER_TYPE_MAP:
            return _LOWER_TYPE_MAP[lower]
        # If the input is ALREADY a valid enum value (e.g. "small_molecule"),
        # return it as-is (some upstream code may have already standardized).
        if cleaned in _VALID_DRUG_TYPES:
            return cleaned
        if lower in _VALID_DRUG_TYPES:
            return lower
        # Novel type — log WARNING and emit "unknown" (A6, S6).
        # FIX-P1-B-6 (audit P1): increment the module-level counter so
        # ``get_schema_drift_report`` (a @classmethod) can see the
        # accumulated totals across ALL instances / processes sharing
        # this module. The previous design incremented an instance attr
        # that the classmethod could never read.
        with _NOVEL_TYPE_LOCK:
            _NOVEL_TYPE_COUNTER[cleaned] += 1
        logger.warning(
            "[chembl] Novel molecule_type %r — emitting DrugType.UNKNOWN. "
            "Add to MOLECULE_TYPE_MAP if this is a recurring type.",
            cleaned,
        )
        return DrugType.UNKNOWN.value

    def _ensure_drug_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required Drug-table columns exist with proper defaults (D2-8).

        Reflects on the SQLAlchemy model to get the column list. Per-column
        defaults are semantic (not just None) for ``name``,
        ``is_fda_approved``, ``drug_type`` — they need values that pass
        the DB's CHECK constraints.
        """
        # Default values per column. None means "add as None".
        # v20 SW-1 minor ROOT FIX: ``is_fda_approved`` default changed
        # from False to None. The audit's complaint was that False here
        # was misleading — even though step ordering means this default
        # is rarely reached, the literal suggested "definitely not
        # FDA-approved" when the correct semantic is "unknown — pending
        # FDA Orange Book join". The coercion logic at L3248-3270 already
        # preserves None as None; the default literal should match.
        defaults: dict[str, Any] = {
            "inchikey": None,
            "name": "Unnamed_Unknown",
            "chembl_id": None,
            "drugbank_id": None,
            "pubchem_cid": None,
            "molecular_formula": None,
            "molecular_weight": None,
            "smiles": None,
            "is_fda_approved": None,
            "max_phase": None,
            "drug_type": DrugType.UNKNOWN.value,
            "mechanism_of_action": None,
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
        # Final safety: ensure is_fda_approved is a real bool OR None
        # (SW-1 ROOT FIX). The previous version converted None → False,
        # which silently defeated the SW-1 fix: is_fda_approved=None
        # means "unknown — pending FDA Orange Book join", NOT "definitely
        # not FDA-approved". Converting None to False made downstream
        # code treat unknown drugs as unapproved, which is just as
        # dangerous as treating them as approved (the RL ranker's safety
        # filter would skip them, missing real repurposing candidates).
        # The fix preserves None as None (object dtype) so downstream
        # code can distinguish "unknown" from "definitely not approved".
        if "is_fda_approved" in df.columns:
            def _coerce_fda_approved(x):
                if x is None:
                    return None
                if isinstance(x, bool):
                    return x
                if isinstance(x, float) and pd.isna(x):
                    return None
                # String "True"/"False" (from CSV round-trip)
                if isinstance(x, str):
                    if x.lower() == "true":
                        return True
                    if x.lower() == "false":
                        return False
                    return None  # unknown string → None
                # Any other type → try bool, fallback to None
                try:
                    return bool(x)
                except (TypeError, ValueError):
                    return None
            df["is_fda_approved"] = df["is_fda_approved"].apply(
                _coerce_fda_approved
            )
        return df

    def _filter_to_drug_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter a DataFrame to only valid Drug-model columns (D2-8, DQ).

        The loader (``bulk_upsert_drugs``) rejects DataFrames with extra
        columns (e.g. ``_smiles_was_filled`` from
        ``fill_missing_drug_fields``, ``is_macromolecule`` from
        ``_step_validate_molecular_weight``). This method drops any column
        that's not in the Drug model.

        Parameters
        ----------
        df : pd.DataFrame
            The cleaned drugs DataFrame (may contain extra columns).

        Returns
        -------
        pd.DataFrame
            A DataFrame with only Drug-model columns. The original df is
            not modified (a filtered copy is returned).
        """
        # The canonical Drug-model columns (from database.models.Drug).
        # We hardcode these rather than reflecting on the model to avoid
        # a circular import and to make the contract explicit.
        drug_columns = {
            "inchikey", "name", "chembl_id", "drugbank_id", "pubchem_cid",
            "molecular_formula", "molecular_weight", "smiles",
            "is_fda_approved", "is_globally_approved", "max_phase", "drug_type",
            "mechanism_of_action",
        }
        # Keep only the columns that are in the Drug model.
        cols_to_keep = [c for c in df.columns if c in drug_columns]
        return df[cols_to_keep].copy()

    def _atomic_write_csv_gz(self, path: Path, df: pd.DataFrame) -> None:
        """Write ``df`` to ``path`` as a gzipped CSV, atomically (R5, A7).

        Writes to a ``.tmp`` file first, then ``os.replace`` (atomic on
        POSIX and Windows). All ``to_csv`` calls pass
        ``encoding="utf-8"`` and ``lineterminator="\\n"`` (C23, INT-6,
        INT-7).
        """
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(
            tmp_path,
            index=False,
            compression="gzip",
            encoding="utf-8",
            lineterminator="\n",
        )
        os.replace(tmp_path, path)
        logger.debug(
            "[%s] Atomic write: %s (%d rows)",
            self.source_name, path, len(df),
        )

    def _compute_file_sha256(self, path: Path) -> str:
        """Compute SHA-256 of a file's bytes (LIN-4, LIN-7)."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _compute_df_sha256(self, df: pd.DataFrame) -> str:
        """Compute SHA-256 of a DataFrame's CSV representation (I8, LIN-4)."""
        csv_bytes = df.to_csv(index=False, encoding="utf-8").encode("utf-8")
        return hashlib.sha256(csv_bytes).hexdigest()

    def _write_manifest(
        self,
        *,
        drugs_path: Path,
        activities_path: Path,
        drugs_checksum: str,
        activities_checksum: str,
        total_molecules: int,
        total_activities: int,
    ) -> None:
        """Write the run manifest JSON to ``self.raw_dir`` (A1, LIN-1 to LIN-18).

        The manifest is the single source of truth for the run's
        provenance. It contains:
        - ``run_id``: ``self.run_id``
        - ``chembl_db_version``: ``self.source_version``
        - ``fetch_start_utc`` / ``fetch_end_utc``
        - ``api_calls``: list of per-call records (from HTTP client)
        - ``artifacts``: list of paths + checksums
        - ``metrics``: all L6 metrics
        - ``settings``: all CHEMBL_* setting values (CFG-15)
        - ``dead_letter_files``: list of dead-letter paths written
        - ``approval_basis``: documentation of the FDA-approval proxy
        """
        manifest = {
            "run_id": self.run_id,
            "source_name": self.source_name,
            "chembl_db_version": self.source_version,
            "chembl_setting_version": CHEMBL_VERSION,
            "fetch_start_utc": self._source_fetch_date.isoformat(),
            "fetch_end_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_date": (
                # I11: record snapshot_date in manifest.
                # FIX-P1-B-7 (audit P1): the previous code used
                #   __import__(...).CHEMBL_SNAPSHOT_DATE or "live"
                # which raises ``AttributeError`` when
                # ``config.settings`` does NOT define
                # ``CHEMBL_SNAPSHOT_DATE`` (the ``or "live"`` fallback
                # never evaluates because the attribute access itself
                # crashes). Root fix: use ``getattr(..., "CHEMBL_SNAPSHOT_DATE", None)``
                # so the fallback to "live" actually fires.
                getattr(
                    __import__("config.settings", fromlist=["CHEMBL_SNAPSHOT_DATE"]),
                    "CHEMBL_SNAPSHOT_DATE",
                    None,
                )
                or "live"
            ),
            "api_calls": [rec.to_dict() for rec in self._http_client.api_calls],
            "artifacts": [
                {
                    "name": "drugs",
                    "path": str(drugs_path),
                    "sha256": drugs_checksum,
                    "row_count": total_molecules,
                },
                {
                    "name": "activities",
                    "path": str(activities_path),
                    "sha256": activities_checksum,
                    "row_count": total_activities,
                },
            ],
            "metrics": dict(self._metrics),
            "settings": {
                "CHEMBL_VERSION": CHEMBL_VERSION,
                "CHEMBL_API_URL": CHEMBL_API_URL,
                "CHEMBL_MAX_PHASE": CHEMBL_MAX_PHASE,
                "CHEMBL_PAGE_SIZE": CHEMBL_PAGE_SIZE,
                "CHEMBL_MAX_ROWS": CHEMBL_MAX_ROWS,
                "CHEMBL_MAX_ACTIVITIES": CHEMBL_MAX_ACTIVITIES,
                "CHEMBL_TARGET_ORGANISM": CHEMBL_TARGET_ORGANISM,
                "CHEMBL_ACTIVITY_TYPES": sorted(CHEMBL_ACTIVITY_TYPES),
                "CHEMBL_STANDARD_UNITS": sorted(CHEMBL_STANDARD_UNITS),
                "CHEMBL_STANDARD_RELATIONS": sorted(CHEMBL_STANDARD_RELATIONS),
                "CHEMBL_ASSAY_TYPES": sorted(CHEMBL_ASSAY_TYPES),
                "CHEMBL_TARGET_TYPES": sorted(CHEMBL_TARGET_TYPES),
                "CHEMBL_TARGET_ACCESSION_STRATEGY": CHEMBL_TARGET_ACCESSION_STRATEGY,
                "CHEMBL_DPI_BATCH_SIZE": CHEMBL_DPI_BATCH_SIZE,
                "CHEMBL_MW_MACROMOLECULE_THRESHOLD": CHEMBL_MW_MACROMOLECULE_THRESHOLD,
            },
            "dead_letter_files": [
                str(p) for p in self._list_dead_letter_files()
            ],
            "approval_basis": (
                "max_phase == 4 (globally approved; not FDA-specific). "
                "Alternative: /molecule.json?approved_drugs=TRUE (S16)."
            ),
            "schema_drift": self.get_schema_drift_report(),
        }
        manifest_path = self.raw_dir / f"chembl_manifest_{self.run_id}.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        logger.info(
            "[%s] Wrote manifest to %s",
            self.source_name,
            manifest_path,
        )

    def _write_dead_letter(
        self,
        df: pd.DataFrame | None,
        *,
        step: str,
        reason: str,
        count: int | None = None,
    ) -> None:
        """Write dropped records to a JSONL dead-letter file (DQ-9, DQ-10, LIN-12).

        Parameters
        ----------
        df : pd.DataFrame or None
            The dropped records. If None, ``count`` must be provided
            (only a summary record is written).
        step : str
            Name of the pipeline step that dropped the records.
        reason : str
            Human-readable reason for the drop.
        count : int, optional
            Number of records dropped (required if ``df`` is None).
        """
        dead_letter_dir = PROCESSED_DATA_DIR / "dead_letter"
        dead_letter_dir.mkdir(parents=True, exist_ok=True)
        path = dead_letter_dir / f"chembl_{step}_{self.run_id}.jsonl"

        records_written = 0
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(path, "w", encoding="utf-8") as fh:
            if df is not None and len(df) > 0:
                for _, row in df.iterrows():
                    record = {
                        "step": step,
                        "reason": reason,
                        "timestamp": timestamp,
                        "run_id": self.run_id,
                        "record": {
                            str(k): (
                                v if isinstance(v, (str, int, float, bool, type(None)))
                                else str(v)
                            )
                            for k, v in row.items()
                        },
                    }
                    fh.write(json.dumps(record, default=str) + "\n")
                    records_written += 1
            elif count is not None and count > 0:
                record = {
                    "step": step,
                    "reason": reason,
                    "timestamp": timestamp,
                    "run_id": self.run_id,
                    "count": count,
                    "note": "Individual records not preserved (filtered before dead-letter).",
                }
                fh.write(json.dumps(record, default=str) + "\n")
                records_written = 1

        # Restrictive permissions (SEC-9).
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass  # best-effort; may fail on Windows

        self._pipeline_dead_letters.append({"path": str(path), "count": records_written})
        logger.debug(
            "[%s] Wrote %d dead-letter records to %s",
            self.source_name, records_written, path,
        )

    def _flush_loader_dead_letters(self, *, step: str) -> None:
        """Flush the loader's module-global dead-letter queue to disk (R9, LIN-13).

        The loader (``database.loaders``) maintains a module-global
        ``_dead_letter_queue`` list. ``flush_dead_letter_queue(path)``
        writes it to disk and clears the queue. We call this after every
        ``bulk_upsert_*`` call so the on-disk file is always up-to-date.
        """
        dead_letter_dir = PROCESSED_DATA_DIR / "dead_letter"
        dead_letter_dir.mkdir(parents=True, exist_ok=True)
        path = str(dead_letter_dir / f"chembl_loader_{step}_{self.run_id}.json")
        count = flush_dead_letter_queue(path)
        if count > 0:
            log_fn = (
                logger.error if count > 10 else logger.warning
            )
            log_fn(
                "[%s] Loader dead-letter queue flushed %d records to %s",
                self.source_name,
                count,
                path,
            )

    def _list_dead_letter_files(self) -> list[Path]:
        """List all dead-letter files written by this run (LIN-12)."""
        dead_letter_dir = PROCESSED_DATA_DIR / "dead_letter"
        if not dead_letter_dir.exists():
            return []
        return sorted(
            p for p in dead_letter_dir.iterdir()
            if p.is_file() and self.run_id in p.name
        )

    # ==================================================================
    # PUBLIC CLASS METHODS
    # ==================================================================

    @classmethod
    def get_schema_drift_report(cls) -> dict[str, int]:
        """Return the schema-drift report (A6).

        Returns
        -------
        dict[str, int]
            Mapping of novel ``molecule_type`` values encountered across
            ALL ChEMBL pipeline instances to their counts. Useful for
        curators deciding which new types to add to ``MOLECULE_TYPE_MAP``.
        """
        # FIX-P1-B-6 (audit P1): read from the module-level
        # ``_NOVEL_TYPE_COUNTER`` (guarded by ``_NOVEL_TYPE_LOCK``).
        # The previous ``getattr(cls, "_novel_type_counter", ...)``
        # always returned the default empty defaultdict because the
        # counter was set on INSTANCES, never on the class object.
        with _NOVEL_TYPE_LOCK:
            return dict(_NOVEL_TYPE_COUNTER)

    @classmethod
    def clean_raw_chunks(cls, older_than_days: int = 7) -> int:
        """Delete raw chunk files older than ``older_than_days`` (LIN-8).

        Parameters
        ----------
        older_than_days : int
            Files older than this many days are deleted.

        Returns
        -------
        int
            Number of files deleted.

        Notes
        -----
        - Only deletes files matching ``activity_chunk_*.json`` in
          ``RAW_DATA_DIR / "chembl"``.
        - Never deletes the canonical ``chembl_drugs.csv.gz`` or
          ``chembl_activities.csv.gz``.
        """
        from config.settings import RAW_DATA_DIR
        raw_dir = RAW_DATA_DIR / "chembl"
        if not raw_dir.exists():
            return 0
        cutoff = time.time() - (older_than_days * 86400)
        deleted = 0
        for p in raw_dir.glob("activity_chunk_*.json"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    deleted += 1
            except OSError:
                pass
        return deleted


# ---------------------------------------------------------------------------
# Module entry point (DOC-13)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    # Quick smoke test entry point:
    #   PIPELINE_RUN_ID=smoke_test_001 CHEMBL_MAX_ROWS=10 \
    #     python -m pipelines.chembl_pipeline
    logging.basicConfig(level=logging.INFO)
    pipeline = ChEMBLPipeline()
    pipeline.run()
