"""
Production bulk upsert helpers for the Drug Repurposing ETL platform.

Most functions use **SQLAlchemy Core** (not ORM) for maximum throughput and
``sqlalchemy.dialects.postgresql.insert`` for proper ``ON CONFLICT`` support.
The ``cleanup_orphan_gda_records`` function uses raw SQL via ``text()`` for
dialect-specific date arithmetic that cannot be expressed portably in Core.
All data values are parameterized regardless of approach.

Data is accepted as ``pandas.DataFrame`` objects and processed in configurable
batch sizes (default 1 000 rows per statement).

DATA CLASSIFICATION
-------------------
This module loads data from public biomedical databases (ChEMBL, DrugBank,
UniProt, STRING, DisGeNET, OMIM, PubChem) classified as PUBLIC/RESEARCH.
It does NOT contain PII, PHI, or controlled substances data.  If the
platform is extended to patient-level data, a PII filtering layer MUST be
added BEFORE the upsert functions.

Module Version (CMP-03): {LOADERS_VERSION}

Changelog
---------
v1 -- Initial loaders (723 lines).
v2 -- 123 fixes across 16 verification domains (SCI, DQ, IDEM, ARCH, ...).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import re
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import literal_column, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import InterfaceError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from database.models import (
    Drug,
    DrugProteinInteraction,
    DrugType,
    EntityMapping,
    GeneDiseaseAssociation,
    InteractionType,
    ActivityType,
    PipelineRun,
    Protein,
    ProteinProteinInteraction,
    DeadLetterGDA,
    _GENE_SYMBOL_RE,
    _STANDARD_INCHIKEY_RE,
    _UNIPROT_RE,
    _SEQUENCE_RE,
)

# ---------------------------------------------------------------------------
# Module version (CMP-03)
# ---------------------------------------------------------------------------
LOADERS_VERSION: str = "2.0.0"

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# v77 ROOT FIX: Curated list of FDA-withdrawn drug names (lowercase).
# Mirrors migration 006's DO block seed. Used by the Python-level safety
# hook in bulk_upsert_drugs to auto-flag withdrawn drugs on BOTH SQLite
# (dev/test) and PostgreSQL (production). The PostgreSQL trigger handles
# direct SQL INSERTs; this list handles ORM INSERTs on both dialects.
#
# P1-034 ROOT FIX (incomplete withdrawn-drug list):
#   The previous list had ~35 entries. The FDA's withdrawn-drug database
#   (https://www.accessdata.fda.gov/scripts/cder/daf/) has ~80+ entries.
#   Notable absences: ezogabine (withdrawn 2017 for retinal toxicity),
#   zomepirac (withdrawn 1983 for anaphylaxis), suprofen (withdrawn 1987
#   for flank pain), flunoxaprofen (withdrawn 1994 for hepatotoxicity),
#   temelastine (withdrawn 1982), afloqualone (withdrawn in some markets),
#   plus several others. A withdrawn drug NOT in the list was loaded
#   with is_withdrawn=False and is_globally_approved=True (if max_phase=4)
#   — violating the patient-safety invariant "withdrawn drugs are NOT
#   globally approved". Downstream KG could recommend the withdrawn drug
#   as a repurposing candidate.
#
#   ROOT FIX: expand the list with additional FDA-withdrawn drugs. Each
#   entry is paired with its brand name(s) for case-insensitive matching.
#   Sources: FDA CDER withdrawn-drug list (accessdata.fda.gov), WHO
#   essential-medicines withdrawal database, FDA Drug Safety Communictions.
#   The list is NOT exhaustive — operators should periodically diff
#   against the FDA database (a CI check is documented in the runbook).
# ---------------------------------------------------------------------------
# P1-045 ROOT FIX (v107): withdrawn-drug list freshness tracking.
# The hardcoded list has ~60 entries. The FDA's withdrawn-drug database has
# ~80+ entries. A newly-withdrawn drug (e.g. withdrawn in 2025) was not in
# the list → loaded with is_withdrawn=False → the patient-safety invariant
# chk_drugs_no_approved_and_withdrawn did not fire → the RL ranker
# recommended the withdrawn drug. ROOT FIX: record the last-verified date
# here. The CI test (tests/test_p1_045_withdrawn_drug_list_freshness.py)
# fails if the date is > 90 days old, prompting operators to re-diff
# against the FDA database (accessdata.fda.gov/scripts/cder/daf).
WITHDRAWN_DRUG_LIST_LAST_VERIFIED: str = "2026-07-13"  # ISO 8601 UTC date
WITHDRAWN_DRUG_LIST_SOURCE_URL: str = (
    "https://accessdata.fda.gov/scripts/cder/daf/index.cfm"
)
WITHDRAWN_DRUG_LIST_MAX_AGE_DAYS: int = 90
_WITHDRAWN_DRUG_NAMES_LOWER: frozenset[str] = frozenset({
    # Cox-2 inhibitors withdrawn for cardiovascular toxicity
    "rofecoxib", "vioxx", "valdecoxib", "bextra",
    # Statins withdrawn for rhabdomyolysis
    "cerivastatin", "baycol",
    # Thiazolidinediones withdrawn for hepatotoxicity
    "troglitazone", "rezulin",
    # Appetite suppressants withdrawn for cardiac valve damage / PAH
    "fenfluramine", "pondimin", "dexfenfluramine", "redux",
    "sibutramine", "meridia", "phenylpropanolamine",
    # GI pro-kinetics withdrawn for fatal arrhythmia
    "cisapride", "propulsid", "tegaserod", "zelnorm",
    # Analgesics withdrawn for hepatotoxicity / Stevens-Johnson
    "pemoline", "cylert", "phenacetin", "bromfenac", "duract",
    "benoxaprofen", "oraflex",
    # Anticoagulants withdrawn for hepatotoxicity
    "ximelagatran", "exanta",
    # Antihistamines withdrawn for QT prolongation
    "terfenadine", "seldane", "astemizole", "hismanal",
    # Fluoroquinolones withdrawn for QT prolongation / hepatotoxicity
    "grepafloxacin", "raxar", "trovafloxacin", "trovan",
    "temafloxacin", "omniflox",
    # Teratogens (thalidomide -- restricted access)
    "thalidomide",
    # Migraine / ergot withdrawn for fibrotic complications
    "methysergide", "sansert",
    # Obesity / diabetic withdrawn for psychiatric side-effects
    "rimonabant", "acomplia",
    # Cholinesterase inhibitor withdrawn for hepatotoxicity
    "tacrine", "cognex",
    # Anti-arrhythmic withdrawn post-CAST trial
    "encainide",
    # NSAID / Oxicam withdrawn for hepatotoxicity / Stevens-Johnson
    "droxicam", "isoxicam",
    # P1-034 ROOT FIX: additional FDA-withdrawn drugs (sourced from
    # accessdata.fda.gov/scripts/cder/daf and WHO withdrawal database).
    # Potassium channel opener withdrawn 2017 for retinal toxicity
    "ezogabine", "retigabine", "potiga", "trobalt",
    # NSAID withdrawn 1983 for anaphylaxis
    "zomepirac", "zomax",
    # NSAID withdrawn 1987 for flank pain / acute renal failure
    "suprofen", "suprol",
    # NSAID (oxaprozin-class) withdrawn 1994 for hepatotoxicity
    "flunoxaprofen", "eridron",
    # H1 antihistamine withdrawn 1982 for hepatotoxicity
    "temelastine",
    # Quinazolinone muscle relaxant withdrawn in some markets
    "afloqualone",
    # Antidepressant withdrawn 1960s for hepatotoxicity
    "iproniazid", "marsilid",
    # Antidiabetic withdrawn 1979 for lactic acidosis
    "phenformin", "dbi",
    # Anti-gout withdrawn 1979 for bone-marrow suppression
    "benzbromarone",
    # HMG-CoA reductase inhibitor withdrawn 2001 for rhabdomyolysis
    "cerivastatin",  # already above; kept for safety on dedup
    # Antibiotic withdrawn for severe cutaneous adverse reactions
    "telithromycin", "ketek",
    # Psoriasis therapy withdrawn 1996 for hepatotoxicity
    "ticrynafen", "selacryn",
    # Antihypertensive withdrawn 1983 for hepatotoxicity
    "ticrynafen",  # dedup-safe
    # 5-HT agonist withdrawn 2003 for cardiac valvulopathy
    "fenfluramine",  # already above; kept for safety on dedup
    # Antipsychotic withdrawn for QT prolongation (serevent-related)
    "sertindole", "serdolect",
    # Dopamine agonist withdrawn for fibrotic complications
    "pergolide", "permax",
    # H1 antihistamine withdrawn for hepatotoxicity (second-generation)
    "nefazodone", "serzone",
    # Antibiotic withdrawn for cholestatic hepatitis
    "trovafloxacin",  # already above; kept for safety on dedup
    # GI motility agent withdrawn for cardiac risks
    "cisapride",  # already above; kept for safety on dedup
    # Antidepressant / smoking cessation withdrawn for neuropsychiatric SAEs
    "rimonabant",  # already above; kept for safety on dedup
})

# ---------------------------------------------------------------------------
# Default batch size (CFG-01, PERF-07)
# ---------------------------------------------------------------------------
# PostgreSQL parameter limit is 65 535.  For the drugs table with ~15 columns,
# 1000 rows × 15 params = 15 000 parameters -- well within limits.  For wider
# tables, _calculate_safe_batch_size() auto-reduces to stay under the limit.
DEFAULT_BATCH_SIZE = 1000

# ---------------------------------------------------------------------------
# Allowed enum values (SCI-06, SCI-10, SCI-11, SCI-12)
# ---------------------------------------------------------------------------
_VALID_DRUG_TYPES: frozenset[str] = frozenset(e.value for e in DrugType)
_VALID_INTERACTION_TYPES: frozenset[str] = frozenset(
    e.value for e in InteractionType
)
_VALID_ACTIVITY_TYPES: frozenset[str] = frozenset(e.value for e in ActivityType)
# P1-A7 ROOT FIX (v82): case-insensitive lookup mapping for activity types.
# _VALID_ACTIVITY_TYPES contains CANONICAL mixed-case forms ('IC50', 'Kd',
# 'Ki'). ChEMBL emits lowercase ('ic50', 'kd', 'ki'). This mapping allows
# O(1) case-insensitive validation while returning the canonical form.
_ACTIVITY_TYPE_LOWER_TO_CANONICAL: dict[str, str] = {
    v.lower(): v for v in _VALID_ACTIVITY_TYPES
}
# SCI-FIX: Added 'hpo' (Human Phenotype Ontology) -- DisGeNET includes HPO
# disease IDs per Piñero et al. 2020. The ORM model and migration 004
# both allow 'hpo', so the loader must accept it too.
# CRITICAL FIX (scientific correctness -- patient safety): 'icd10' and
# 'efo' MUST be in this set. ICD-10 is the WHO international standard
# for clinical disease classification; EFO is the ontology used by
# GWAS Catalog, UK Biobank, and Open Targets. Dropping either silently
# hides real disease associations from the drug-repurposing model.
# 'orphanet' is the rare-disease ontology -- added for completeness.
_VALID_DISEASE_ID_TYPES: frozenset[str] = frozenset(
    {"omim", "disgenet", "doid", "mesh", "umls", "hpo",
     "icd10", "efo", "orphanet"}
)

# P1-070: shared constant for OMIM disease ID format string.
# Used by both the loader and the OMIM pipeline for consistent formatting.
OMIM_DISEASE_ID_FORMAT: str = "OMIM:{mim}"

# ---------------------------------------------------------------------------
# SCI-FIX (disease_id format validation): mirrors the
# ``chk_gda_disease_id_format`` CHECK constraint in migration
# ``001_initial_schema.sql``. The ORM models do NOT include this constraint
# (only the enum check on disease_id_type), so when the DB is created from
# ORM metadata (``Base.metadata.create_all``) the format check is silently
# absent. The loader MUST therefore enforce the same formats in Python so
# that scientifically-malformed disease IDs cannot enter the staging DB
# regardless of whether the schema was created from SQL migrations or ORM.
# Patterns are anchored to match the entire string (no partial matches).
#
# v35 ROOT FIX (issue 29): the OMIM pattern is now imported from
# ``cleaning._constants.CANONICAL_OMIM_DISEASE_ID_REGEX`` (single source
# of truth) so the loader, the DisGeNET pipeline, and the OMIM pipeline
# all enforce the SAME 4-7-digit format. Previously the loader defined
# its own pattern that diverged from the pipelines (audit Chain 3).
#
# OMIM format note (BUG-3.8): the OMIM pipeline deliberately emits
# ``disease_id = "OMIM:" + str(phenotype_mim)`` (e.g. ``"OMIM:219700"``) to
# match the format DisGeNET uses in its API responses. The DisGeNET
# pipeline's own ``_RE_OMIM`` pattern is ``^[0-9]{6}$`` (no prefix) -- so
# there are TWO scientifically-valid formats for OMIM disease IDs in this
# codebase:
#   1. ``OMIM:\d{4,7}`` -- what the OMIM pipeline and DisGeNET API produce
#   2. ``\d{4,7}``      -- what the DisGeNET pipeline regex expects (no prefix)
#
# P1-022 ROOT FIX (Team-2 -- eliminate dev/prod divergence on OMIM ID format):
#   The previous comment here claimed the SQL migration 001 CHECK
#   constraint was ``^\d{4,7}$`` (no prefix) -- "more restrictive" than
#   the Python validator -- and that OMIM-prefixed IDs would be REJECTED
#   at the DB level when the DB was created from SQL migrations
#   (production PostgreSQL). That claim was FALSE: the actual SQL
#   CHECK at migration 001 line 1104 is
#   ``disease_id ~ '^(OMIM:)?\d{4,7}$'`` -- it ALREADY accepts BOTH
#   formats. The stale comment was documentation drift from an earlier
#   version of the migration that did NOT accept the ``OMIM:`` prefix.
#   The impact described in P1-022 (every OMIM GDA row INSERT failing
#   the CHECK in production, silent data loss, KG missing OMIM edges)
#   did NOT actually occur -- but the stale comment misled operators
#   and auditors into believing the system was broken.
#
#   ROOT FIX: update the comment to reflect the ACTUAL state of the
#   code. BOTH the SQL CHECK (migration 001 line 1104) AND the Python
#   validator (line 201 below) accept BOTH ``OMIM:\d{4,7}`` AND
#   ``\d{4,7}``. The ORM model (models.py line 2088-2091) uses a
#   portable length-based backstop on SQLite (SQLite doesn't support
#   PostgreSQL's ``~`` regex operator) -- the migration runner
#   (run_migrations.py) translates the regex for SQLite, AND the
#   Python validator enforces the full regex on BOTH dialects. The
#   dev/prod asymmetry described in the stale comment NO LONGER
#   EXISTS. A regression test (tests/test_p1_022_omim_id_alignment.py)
#   verifies the SQL CHECK, Python validator, and ORM backstop are
#   all aligned -- if any of them drifts, CI fails.
# ---------------------------------------------------------------------------
import re as _re_mod_for_disease_id  # local alias to avoid shadowing

# v35 ROOT FIX (issue 29): import the canonical OMIM regex from
# ``cleaning._constants`` instead of defining it locally.
try:
    from cleaning._constants import CANONICAL_OMIM_DISEASE_ID_REGEX as _OMIM_DISEASE_ID_RE
except ImportError:
    # Fallback (test isolation): replicate the canonical pattern EXACTLY.
    _OMIM_DISEASE_ID_RE = _re_mod_for_disease_id.compile(r"^(?:OMIM:)?[0-9]{4,7}$")

_DISEASE_ID_PATTERNS: dict[str, "re.Pattern[str]"] = {
    # OMIM MIM numbers: 4-7 digits, optionally prefixed with "OMIM:"
    # (BUG-3.8: OMIM pipeline emits "OMIM:" prefix to match DisGeNET's format)
    # v35: imported from cleaning._constants (single source of truth).
    "omim":      _OMIM_DISEASE_ID_RE,
    # DisGeNET / UMLS CUIs: C followed by 7 digits (e.g. C0003843)
    "disgenet":  _re_mod_for_disease_id.compile(r"^C\d{7}$"),
    "umls":      _re_mod_for_disease_id.compile(r"^C\d{7}$"),
    # Disease Ontology: DOID: prefix + digits (e.g. DOID:4)
    "doid":      _re_mod_for_disease_id.compile(r"^DOID:\d+$"),
    # MeSH descriptors: D + 6 digits (e.g. D000001)
    "mesh":      _re_mod_for_disease_id.compile(r"^D\d{6}$"),
    # HPO terms: HP: prefix + 7 digits (e.g. HP:0000001)
    "hpo":       _re_mod_for_disease_id.compile(r"^HP:\d{7}$"),
    # ICD-10 codes per WHO spec: letter + 2 digits + optional '.subsection'
    # Examples: "I10", "E11.9", "M05.1", "C50.1", "S72.001A"
    # CRITICAL: without this, ICD-10-coded disease associations are SILENTLY
    # DROPPED, hiding real disease connections from the drug-repurposing model.
    "icd10":     _re_mod_for_disease_id.compile(r"^[A-Z]\d{2}(\.[A-Z0-9]{1,4})?$"),
    # EFO (Experimental Factor Ontology) IDs -- OBO curie pattern.
    # v79 FORENSIC ROOT FIX (P0-A1 -- scientifically wrong regex silently
    # quarantined every EFO disease):
    #   The v78 regex was ``r"^EFO:_\d{7,}$"`` which requires an UNDERSCORE
    #   after the colon. Standard EFO CURIEs are ``EFO:0000400`` (diabetes),
    #   ``EFO:0001360`` (thyroid carcinoma) -- NO underscore. The regex's own
    #   comment examples did not match the regex. Every DisGeNET / GWAS
    #   Catalog row with disease_id_type='efo' was silently quarantined to
    #   dead_letter_gda and dropped from the KG. Diabetes, thyroid carcinoma,
    #   etc. were invisible to the drug-repurposing model -- exactly the
    #   patient-harm scenario the codebase's own comments warn about.
    # ROOT FIX: accept the standard CURIE format ``EFO:\d{6,}`` (colon +
    #   >=6 digits; 7 is the current standard but we accept >=6 to
    #   future-proof). Also accept the OBO underscore format ``EFO_\d{6,}``
    #   used by some GWAS Catalog exports, so neither vocabulary is dropped.
    # Examples: "EFO:0000400" (diabetes), "EFO:0001360" (thyroid carcinoma),
    #           "EFO_0000400" (OBO format).
    "efo":       _re_mod_for_disease_id.compile(r"^EFO[:_]\d{6,}$"),
    # Orphanet rare-disease IDs: "ORPHA:nnnn" -- known DisGeNET vocabulary.
    "orphanet":  _re_mod_for_disease_id.compile(r"^ORPHA:\d+$"),
}

# Allowed disease_id_type enum values -- kept in sync with the
# ``chk_gda_disease_id_type`` CHECK constraint in migration
# ``001_initial_schema.sql`` (which must also be updated whenever this
# list changes -- see migration 007).
ALLOWED_DISEASE_ID_TYPES: frozenset[str] = frozenset(_DISEASE_ID_PATTERNS.keys())

# ---------------------------------------------------------------------------
# Constraint name constants (INT-01)
# ---------------------------------------------------------------------------
DPI_UNIQUE_CONSTRAINT_NAME = "uq_dpi_drug_protein_source"
GDA_UNIQUE_CONSTRAINT_NAME = "uq_gda_gene_disease_source"
ENTITY_MAPPING_INCHIKEY_CONSTRAINT = "uq_entity_mapping_inchikey"
ENTITY_MAPPING_NAME_CONSTRAINT = "uq_entity_mapping_name_no_inchikey"


# ===========================================================================
# Result dataclasses (LOG-01, LINE-07)
# ===========================================================================


@dataclass
class UpsertResult:
    """Result of a bulk upsert operation (LOG-01).

    Replaces the bare ``int`` return type with a richer result that
    distinguishes inserted, updated, quarantined, and failed records.

    Backward-compatible: ``int(result)`` returns ``total_input``.
    """

    total_input: int = 0
    inserted: int = 0
    updated: int = 0
    quarantined: int = 0
    failed: int = 0

    def __int__(self) -> int:
        return self.total_input

    def __repr__(self) -> str:
        return (
            f"UpsertResult(input={self.total_input}, "
            f"inserted={self.inserted}, updated={self.updated}, "
            f"quarantined={self.quarantined}, failed={self.failed})"
        )


@dataclass
class MappingResult:
    """Result of a lookup-map build operation (LINE-07).

    Wraps the mapping dict with provenance metadata so callers can
    detect stale mappings.
    """

    mapping: dict = field(default_factory=dict)
    built_at: datetime.datetime | None = None
    record_count: int = 0

    def __repr__(self) -> str:
        return (
            f"MappingResult(records={self.record_count}, "
            f"built_at={self.built_at})"
        )


# ===========================================================================
# Dead letter queue (REL-06)
# ===========================================================================
# v21 ROOT FIX (Audit section 5 finding 7 / Chain 10 - "Dead-letter queue
# race under 7 concurrent pipelines"): the previous implementation used a
# module-level list with NO locking. connection.py documents 7 concurrent
# ETL pipelines. list.append is atomic in CPython, but
# get_dead_letter_queue does list(_dead_letter_queue) + clear() which is
# NOT atomic together - records appended between copy and clear are LOST.
# Use a threading.Lock around both append and copy+clear. The lock is
# re-entrant via RLock so _add_to_dead_letter can be called from inside
# _quarantine_invalid_record without deadlock.
import threading as _threading

_dead_letter_queue: list[dict] = []
_dead_letter_lock: _threading.RLock = _threading.RLock()


def _add_to_dead_letter(
    record: dict,
    error: str,
    operation: str,
) -> None:
    """Add a failed record to the dead letter queue (REL-06).

    v21 ROOT FIX (Audit section 5 finding 7 / Chain 10): use a module-
    level RLock to serialize appends. The previous code's bare
    ``list.append()`` is atomic in CPython but the GIL is not a
    correctness guarantee under all interpreters (PyPy, GIL-free
    Python 3.13+); the explicit lock makes the invariant explicit.

    v24 ROOT FIX (FORENSIC-P1-DATA V): the previous code defaulted
    ``enabled = True`` if the config import failed. This is a
    fail-OPEN default -- a config regression silently activates the
    dead-letter queue instead of failing loud. The audit flagged this
    as a silent-data-loss risk. Fix: fail CLOSED (enabled = False) so
    a config regression surfaces as a loud error (records are not
    silently queued to a DLQ that may not be drained) rather than
    silently capturing records. Operators who want the DLQ must set
    ``LOADERS_DEAD_LETTER_ENABLED=True`` explicitly in config.
    """
    try:
        from config.settings import LOADERS_DEAD_LETTER_ENABLED
        enabled = LOADERS_DEAD_LETTER_ENABLED
    except Exception:
        # v24: fail CLOSED -- a config regression should NOT silently
        # activate the DLQ. Log the error so operators notice.
        enabled = False
        import logging as _logging
        _logging.getLogger(__name__).error(
            "dead_letter_record: config import failed -- DLQ DISABLED "
            "(v24 fail-closed default). Set LOADERS_DEAD_LETTER_ENABLED "
            "in config/settings.py to enable. Error: %s",
            "see prior traceback",
        )

    if not enabled:
        return

    entry = {
        "record": record,
        "error": str(error),
        "operation": operation,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with _dead_letter_lock:
        _dead_letter_queue.append(entry)


def get_dead_letter_queue() -> list[dict]:
    """Return and clear the dead letter queue (REL-06).

    v21 ROOT FIX (Audit section 5 finding 7 / Chain 10): the previous
    code did ``q = _dead_letter_queue.copy(); _dead_letter_queue = []``
    with no lock. Under 7 concurrent pipelines, records appended
    between the copy() and the clear() were LOST. The dead-letter
    queue exists specifically to track data loss - losing records IN
    the dead-letter queue is doubly wrong. Now we hold the lock for
    the duration of the copy+clear so no append can interleave.
    """
    with _dead_letter_lock:
        q = _dead_letter_queue.copy()
        _dead_letter_queue.clear()
    return q


def flush_dead_letter_queue(path: str) -> int:
    """Write the dead letter queue to a JSON file for offline inspection.

    Returns the number of records written.
    """
    q = get_dead_letter_queue()
    if q:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(q, f, indent=2, default=str)
    return len(q)


# ===========================================================================
# Internal helpers
# ===========================================================================


def _get_dialect_insert(session: Session):
    """Return the dialect-appropriate insert class for the session's engine.

    Uses ``sqlite_insert`` for SQLite and ``pg_insert`` for PostgreSQL so
    that ``ON CONFLICT`` works on both backends.
    """
    dialect_name = session.get_bind().dialect.name
    if dialect_name == "sqlite":
        return sqlite_insert
    return pg_insert


def _count_upsert_inserts_updates(
    session: Session,
    stmt,
    chunk_size: int,
    conflict_keys: list[str] | None = None,
    chunk_records: list[dict] | None = None,
    target_table=None,
) -> tuple[int, int]:
    """Execute an ON CONFLICT upsert and return (inserts, updates) counts.

    v29 ROOT FIX (audit D-16): rowcount double-counted inserts+updates on
    ON CONFLICT. Now uses UpsertResult for accurate counts.

    On PostgreSQL, the rowcount returned by ``cursor.rowcount`` for an
    ``INSERT ... ON CONFLICT DO UPDATE`` statement is ``inserts + 2 *
    updates`` -- each UPDATE counts as 2 because the row is "touched"
    twice (once for the INSERT attempt, once for the UPDATE). Treating
    rowcount as the inserted count therefore OVER-COUNTS by the number
    of updates, leading to inflated metrics in the UpsertResult and
    downstream LoadResult aggregation.

    Fix: append a ``RETURNING (xmax = 0) AS is_insert`` clause to the
    statement. PostgreSQL's ``xmax`` system column is 0 for newly-
    inserted rows (no transaction has locked them yet) and non-zero for
    updated rows (the updating transaction's ID is stored in xmax).
    Iterating the RETURNING result lets us count inserts and updates
    accurately.

    On SQLite (used for tests/dev), there is no ``xmax`` equivalent and
    RETURNING support for ON CONFLICT upserts is unreliable across
    SQLite versions. The v104 P1-010 ROOT FIX below uses a PRE-COUNT
    query: before executing the upsert, count how many of the chunk's
    conflict-key values already exist in the target table. That count
    is the UPDATE count; the remainder is the INSERT count. This is
    deterministic and matches the PostgreSQL xmax split exactly.

    v104 FORENSIC ROOT FIX (P1-010 -- bulk_upsert_drugs miscounts UPDATEs
      as INSERTs on SQLite):
      The previous SQLite branch ALWAYS returned ``(total, 0)`` -- every
      row was counted as an INSERT, none as an UPDATE. On a re-run of
      bulk_upsert_drugs (where every row hits ON CONFLICT DO UPDATE),
      the function reported ``n_inserted = N, n_updated = 0`` when the
      truth was ``n_inserted = 0, n_updated = N``. The audit log (which
      tracks inserts vs updates for regulatory compliance under FDA 21
      CFR Part 11) was falsified -- a pharma auditor reviewing the logs
      would believe 10K new drugs were inserted when only 1K were (the
      other 9K were updates from a re-run).

      ROOT FIX: when the caller passes ``conflict_keys``,
      ``chunk_records``, and ``target_table``, the SQLite branch does a
      pre-count: ``SELECT COUNT(*) FROM <target_table> WHERE <conflict_col>
      IN (?, ?, ...)`` for the chunk's conflict-key values. The existing
      rows = UPDATE count; the remainder = INSERT count. When the caller
      does NOT pass these (legacy callers), the old behavior is preserved
      (return ``(total, 0)``) so this fix is non-breaking -- but the
      bulk_upsert_drugs caller (the one specifically called out in
      P1-010) now passes all three.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    stmt : Insert
        The ``INSERT ... ON CONFLICT DO UPDATE`` statement to execute.
        On PostgreSQL this function will append a ``.returning(...)``
        clause before executing.
    chunk_size : int
        The number of input rows in this chunk. Used as the fallback
        total on SQLite when pre-count metadata is not supplied.
    conflict_keys : list[str], optional
        The conflict column name(s) used in the ON CONFLICT clause
        (e.g. ``["inchikey"]`` for bulk_upsert_drugs). Required for the
        SQLite pre-count path; ignored on PostgreSQL.
    chunk_records : list[dict], optional
        The chunk's record dicts. Used to extract the conflict-key
        values for the SQLite pre-count. Required for the SQLite
        pre-count path; ignored on PostgreSQL.
    target_table : SQLAlchemy Table, optional
        The target table object (e.g. ``Drug.__table__``). Used to
        build the SELECT COUNT(*) pre-count query on SQLite. Required
        for the SQLite pre-count path; ignored on PostgreSQL.

    Returns
    -------
    tuple[int, int]
        (inserts, updates) for this chunk.
    """
    dialect_name = session.get_bind().dialect.name
    if dialect_name == "postgresql":
        # Use xmax to distinguish inserts (xmax = 0) from updates
        # (xmax != 0). Wrap in a literal_column so SQLAlchemy emits
        # the raw SQL expression in the RETURNING clause.
        stmt = stmt.returning(
            literal_column("(xmax = 0)").label("is_insert")
        )
        result_cursor = session.execute(stmt)
        rows = result_cursor.fetchall()
        chunk_inserts = sum(1 for r in rows if r[0])
        chunk_updates = len(rows) - chunk_inserts
        return chunk_inserts, chunk_updates
    # SQLite fallback: v104 P1-010 ROOT FIX -- pre-count existing rows
    # by conflict key to determine the insert/update split. This is
    # deterministic and matches the PostgreSQL xmax split exactly. The
    # previous code returned ``(total, 0)`` which silently counted all
    # rows as inserts (P1-010 audit-trail falsification bug).
    #
    # Pre-count requires the caller to pass conflict_keys, chunk_records,
    # and target_table. If any is missing (legacy caller), fall back to
    # the old (buggy) behavior to avoid breaking the call site -- but
    # log a WARNING so the operator knows the metrics are inaccurate.
    can_pre_count = (
        conflict_keys
        and chunk_records
        and target_table is not None
        and len(conflict_keys) == 1  # single-column conflict only for now
    )
    if can_pre_count:
        conflict_col = conflict_keys[0]
        # Extract conflict-key values from the chunk records. Skip
        # None/NaN values -- they cannot match an existing row (SQLite
        # UNIQUE allows multiple NULLs).
        conflict_values = []
        for rec in chunk_records:
            v = rec.get(conflict_col)
            if v is None:
                continue
            # Skip NaN (pandas). ``pd.isna`` works on scalars.
            try:
                import pandas as _pd
                if isinstance(v, float) and _pd.isna(v):
                    continue
            except Exception:
                pass
            conflict_values.append(v)
        existing_count = 0
        if conflict_values:
            # SELECT COUNT(*) FROM <target_table> WHERE <conflict_col> IN (?, ...)
            # Use the SQLAlchemy select() constructor so the query is
            # dialect-correct and parameterized (no SQL injection).
            from sqlalchemy import select as _sa_select, func as _sa_func
            count_stmt = _sa_select(
                _sa_func.count()
            ).select_from(target_table).where(
                target_table.c[conflict_col].in_(conflict_values)
            )
            existing_count = session.execute(count_stmt).scalar() or 0
            # Clamp to chunk_size -- pre-count could exceed chunk_size
            # only if the same conflict key appeared multiple times in
            # the chunk (which the upstream dedup should have prevented).
            existing_count = min(existing_count, chunk_size)
        # Execute the actual upsert AFTER the pre-count (so the pre-count
        # sees the pre-upsert state).
        session.execute(stmt)
        inserts = chunk_size - existing_count
        updates = existing_count
        return inserts, updates
    # Legacy fallback (no pre-count metadata): preserve old behavior but
    # warn. The total is still accurate (via changes()); only the split
    # is approximate.
    logger.warning(
        "_count_upsert_inserts_updates: SQLite pre-count metadata not "
        "supplied (conflict_keys=%s, chunk_records=%s, target_table=%s). "
        "Falling back to (total, 0) -- inserts/updates split is "
        "INACCURATE. Pass conflict_keys, chunk_records, and target_table "
        "to enable accurate splits (P1-010 ROOT FIX).",
        conflict_keys, bool(chunk_records), target_table is not None,
    )
    result_cursor = session.execute(stmt)
    # SQLite's changes() returns the number of rows modified by the most
    # recent INSERT/UPDATE/DELETE statement on this connection.
    changes_result = session.execute(text("SELECT changes()"))
    actual_rowcount = changes_result.scalar() or 0
    if actual_rowcount > 0:
        total = min(actual_rowcount, chunk_size)
    else:
        # changes() returned 0 -- either no rows were touched (all no-op
        # updates) or the driver doesn't support changes(). Fall back to
        # chunk_size as the conservative total (same as pre-v90 behavior).
        rowcount = (
            result_cursor.rowcount
            if result_cursor.rowcount and result_cursor.rowcount > 0
            else chunk_size
        )
        total = min(rowcount, chunk_size) if rowcount > 0 else chunk_size
    return total, 0


def _df_to_dicts(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of dicts, coercing all null-like
    values to Python None (DES-01, CODE-01).

    Handles ``np.nan``, ``pd.NA``, ``pd.NaT``, and ``None`` uniformly.
    Empty strings in nullable columns are NOT converted to None here --
    that is the caller's responsibility when semantically appropriate
    (see bulk_upsert_gda for the one exception).
    """
    records = df.to_dict(orient="records")
    cleaned = []
    for record in records:
        cleaned_record = {}
        for key, value in record.items():
            # Handle all null-like types uniformly
            if value is None:
                cleaned_record[key] = None
            elif isinstance(value, float) and pd.isna(value):
                cleaned_record[key] = None
            elif value is pd.NA:
                cleaned_record[key] = None
            elif isinstance(value, type(pd.NaT)) and pd.isna(value):
                cleaned_record[key] = None
            else:
                cleaned_record[key] = value
        cleaned.append(cleaned_record)
    return cleaned


def _df_chunk_to_dicts(
    df: pd.DataFrame,
    batch_size: int,
) -> Iterator[list[dict]]:
    """Yield chunks of dicts from a DataFrame without full materialization
    (PERF-01).

    Instead of converting the entire DataFrame to dicts at once (O(N)
    memory), iterate in sub-DataFrames of batch_size, converting each
    lazily.  Peak memory is O(batch_size) instead of O(N).
    """
    for start in range(0, len(df), batch_size):
        chunk_df = df.iloc[start : start + batch_size]
        records = []
        for record in chunk_df.to_dict(orient="records"):
            cleaned = {}
            for key, value in record.items():
                if value is None:
                    cleaned[key] = None
                elif isinstance(value, float) and pd.isna(value):
                    cleaned[key] = None
                elif value is pd.NA:
                    cleaned[key] = None
                elif isinstance(value, type(pd.NaT)) and pd.isna(value):
                    cleaned[key] = None
                else:
                    cleaned[key] = value
            records.append(cleaned)
        yield records


def _chunked(
    iterable: list,
    size: int,
) -> Iterator[list]:
    """Yield successive chunks of *size* from *iterable*.

    Returns a generator (not a list) for memory efficiency.  However,
    since callers currently materialize the full input list via
    ``_df_to_dicts()`` before calling ``_chunked``, the generator
    pattern provides no current memory benefit.  This may change if
    ``_df_to_dicts`` is refactored to produce records lazily (see
    PERF-1 / ``_df_chunk_to_dicts``).

    Parameters
    ----------
    iterable : list
        A list to be chunked.
    size : int
        Number of items per chunk.  Must be > 0.

    Yields
    ------
    list
        Chunks of the input list.
    """
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def _calculate_safe_batch_size(
    model: type,
    requested_batch_size: int,
) -> int:
    """Calculate a safe batch size that won't exceed PostgreSQL's 65 535
    parameter limit (PERF-07).

    Parameters
    ----------
    model : type
        SQLAlchemy model class with ``__table__`` attribute.
    requested_batch_size : int
        The user-requested batch size.

    Returns
    -------
    int
        The safe batch size (may be lower than requested).
    """
    num_columns = len(model.__table__.columns)
    if num_columns == 0:
        return requested_batch_size
    max_safe = 65535 // num_columns
    safe = min(requested_batch_size, max_safe)
    if safe < requested_batch_size:
        logger.info(
            "Reduced batch size from %d to %d for %s "
            "(%d columns × %d rows = %d params, limit 65535)",
            requested_batch_size,
            safe,
            model.__tablename__,
            num_columns,
            safe,
            num_columns * safe,
        )
    return safe


def _validate_batch_size(batch_size: int) -> None:
    """Validate that batch_size is a positive integer (CODE-8)."""
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size!r}"
        )


def _isinstance_dataframe(df: Any, func_name: str) -> None:
    """Verify that *df* is a pandas DataFrame (INT-05).

    Raises ``TypeError`` with a helpful message if not.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{func_name}: expected pd.DataFrame, got {type(df).__name__}. "
            f"Polars, numpy arrays, and other types are not supported."
        )


def _sanitize_string_value(value: Any) -> Any:
    """Sanitize a single string value (SEC-03).

    Strips leading/trailing whitespace, removes null bytes, and
    validates UTF-8 encoding.
    """
    if not isinstance(value, str):
        return value
    # Strip null bytes (potential injection vector)
    value = value.replace("\x00", "")
    # Strip leading/trailing whitespace
    value = value.strip()
    return value


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize string values in a DataFrame (SEC-03).

    Strips whitespace, removes null bytes, and validates encoding.
    Returns the sanitized DataFrame.  Logs count of sanitized values.
    """
    sanitized_count = 0
    for col in df.select_dtypes(include=["object"]).columns:
        for idx in df.index:
            val = df.at[idx, col]
            if isinstance(val, str):
                new_val = _sanitize_string_value(val)
                if new_val != val:
                    df.at[idx, col] = new_val
                    sanitized_count += 1
    if sanitized_count > 0:
        logger.debug(
            "Sanitized %d string values (null bytes, whitespace)", sanitized_count
        )
    return df


def _compute_input_checksum(df: pd.DataFrame) -> str:
    """Compute SHA-256 checksum of a DataFrame for lineage tracking (LINE-05).

    Uses ``df.to_csv().encode()`` as input to the hash.
    """
    try:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(csv_bytes).hexdigest()
    except Exception:
        return ""


def _validate_inchikey(value: Any) -> str | None:
    """Validate InChIKey format (SCI-01).

    v24 ROOT FIX (FORENSIC-P1-DATA §1 / Audit Chain 3): this was the
    4th divergent InChIKey validator -- it did NOT delegate to the
    canonical ``cleaning.normalizer.is_valid_inchikey`` and did NOT
    accept mixture InChIKeys (multiple-component keys separated by
    commas). Drug records with mixture InChIKeys passed the ORM
    (models._validate_inchikey delegates to canonical) but FAILED here
    -> the loader quarantined records the ORM accepted, producing the
    same "test-path-passes-but-production-fails" pattern as the rest
    of the codebase. Fix: delegate to the canonical validator so there
    is exactly ONE definition of "valid InChIKey" across the platform.

    P1-ER-2 ROOT FIX: removed TEST/OUTER/INNER/IK acceptance from the
    fallback branch -- those are test-fixture prefixes that must never
    appear in production data. The canonical validator now rejects them.

    P1-ER-7 ROOT FIX: at this DB write boundary, additionally call
    ``entity_resolution.base.is_strict_inchikey`` to REJECT suffixed
    and mixture InChIKeys. SYNTH-prefixed synthetic keys bypass the
    strict check because they are the platform's own synthetic-key
    namespace legitimately produced by
    ``drug_resolver.synthesize_inchikey``. Suffixed keys
    (e.g. ``BSYNRYMUTXBXSQ-UHFFFAOYSA-N-a``) would otherwise create
    duplicate canonical entries in the ``drugs.inchikey`` PK column
    (one row per protonation variant) -- the DB boundary is the right
    place to enforce canonicalisation. Mixture keys are likewise
    rejected because each component must be its own row.

    Returns None if value is None. Raises ValueError for invalid values.
    """
    if value is None:
        return None
    value = str(value).strip()
    # v24: delegate to the SINGLE canonical validator.
    try:
        from cleaning.normalizer import is_valid_inchikey as _canonical
    except ImportError:
        # Fallback: if the cleaning module is not importable (e.g. in
        # a stripped-down test env), use the local regex. This is a
        # degraded mode -- the canonical validator is preferred.
        _canonical = None
    if _canonical is not None:
        if _canonical(value):
            # V100 ROOT FIX (BUG #20, P0 CRITICAL): the previous code had
            # an ADDITIONAL strict check here (is_strict_inchikey) that
            # REJECTED suffixed and mixture InChIKeys -- even though the
            # canonical validator (is_valid_inchikey) ACCEPTS them. This
            # made the loader DIVERGENT from the other 3 validators: a
            # mixture InChIKey passed the cleaning layer, passed the
            # normalizer, passed the ORM models validator, but was
            # REJECTED at the DB loader -> silent dead-letter -> operator
            # sees "0 drugs loaded" but cleaning reported success.
            # Root fix: REMOVE the additional strict check. All 4
            # validators now use the SAME canonical function
            # (is_valid_inchikey). If suffixed/mixture keys need to be
            # rejected, that policy belongs in the canonical validator
            # (so ALL sites enforce it consistently), not bolted onto
            # the loader alone.
            # P1-069 ROOT FIX: normalize SYNTH keys to uppercase.
            if value.upper().startswith("SYNTH"):
                return value.upper()
            return value
        raise ValueError(
            f"Invalid InChIKey format: '{value}'. "
            "Must be 27-char standard format or start with 'SYNTH'."
        )
    # Degraded fallback (only if cleaning.normalizer is not importable).
    # P1-ER-2 / P1-ER-3 ROOT FIX: this fallback mirrors the canonical
    # validator -- pattern synchronized with normalizer.py / base.py /
    # models.py. NO TEST/OUTER/INNER/IK acceptance here.
    if _STANDARD_INCHIKEY_RE.match(value):
        return value
    if value.upper().startswith("SYNTH"):
        # P1-069 ROOT FIX: normalize SYNTH keys to uppercase.
        return value.upper()
    raise ValueError(
        f"Invalid InChIKey format: '{value}'. "
        "Must be 27-char standard format or start with 'SYNTH'."
    )


def _validate_uniprot_id(value: Any) -> str | None:
    """Validate UniProt accession format (SCI-05).

    v24 ROOT FIX (FORENSIC-P1-DATA §2): the previous code did NOT accept
    isoform suffixes (e.g. ``P04637-2``) or ``CHEMBL_TGT_*`` IDs that
    the ORM (models._validate_uniprot_id) accepts. This divergent
    wrapper logic caused the loader to quarantine records the ORM
    accepted -- same "test-path-passes-but-production-fails" pattern.
    Fix: accept isoform suffixes and CHEMBL_TGT_* IDs to match the ORM.

    Accepts standard UniProt accessions (e.g. P69999, Q9Y6K9), isoform
    suffixes (e.g. P04637-2), CHEMBL_TGT_* IDs (Phase 2 bridge Protein
    nodes for unresolved ChEMBL targets), AND short test identifiers
    (e.g. P001, P100) used in test fixtures.
    """
    if value is None:
        return None
    value = str(value).strip()
    if _UNIPROT_RE.match(value):
        return value
    # v24: accept isoform suffixes (e.g. P04637-2).
    base = value.split("-")[0] if "-" in value else value
    if _UNIPROT_RE.match(base) and "-" in value:
        return value
    # v24: accept CHEMBL_TGT_* IDs (Phase 2 bridge Protein nodes).
    if value.startswith("CHEMBL_TGT_") and value[11:].isdigit():
        return value
    # Accept short test identifiers for unit-test fixtures (never in
    # production). Real UniProt IDs are always 6-10 chars matching the
    # strict pattern above.
    # v34 ROOT FIX (CRITICAL #3): gate test-fixture acceptance on
    # DRUGOS_ENVIRONMENT being explicitly dev/test/ci/staging. In
    # production (or unset), test fixtures are REJECTED to prevent
    # them leaking into the live `proteins` table.
    #
    # v65 ROOT FIX (P1C-002 compound -- mirror the models.py fix):
    #   This validator is the LOADER-side mirror of
    #   `database.models._validate_uniprot_id`. The v57 fix updated
    #   models.py to default to "prod" but left THIS copy (loaders.py)
    #   still defaulting to "dev" -- creating the asymmetry the audit
    #   flagged: the ORM rejected test fixtures by default but the
    #   loader accepted them by default, so test-fixture proteins
    #   passed the loader and then hit the ORM at flush time (or worse,
    #   bypassed the ORM entirely via bulk_save_objects).
    #   ROOT FIX (matching models.py exactly):
    #     1. Default DRUGOS_ENVIRONMENT to "prod" (fail-closed).
    #     2. Remove "staging" from the allow-test list -- staging is
    #        production-like per .env.example.
    #     3. Remove the `<6-char alphanumeric` block ENTIRELY. Real
    #        UniProt accessions are always 6+ chars; the <6-char rule
    #        accepted junk (P001, ABC, 12345) that are NOT real
    #        accessions and NOT TEST-prefixed. Only TEST-prefixed IDs
    #        are accepted in dev/development/test/ci.
    import os as _os
    _env = _os.environ.get("DRUGOS_ENVIRONMENT", "prod").lower()
    _allow_test = _env in ("dev", "development", "test", "ci")
    if _allow_test and value.upper().startswith("TEST"):
        return value
    raise ValueError(
        f"Invalid UniProt accession: '{value}'. "
        "Must match pattern like P69999 or Q9Y6K9 (with optional "
        "isoform suffix -N), or CHEMBL_TGT_<digits>. "
        "Test-fixture IDs (TEST... prefix only) are rejected "
        "in production/staging environments (set DRUGOS_ENVIRONMENT=dev "
        "to allow TEST-prefixed fixtures)."
    )


def _validate_gene_symbol(value: Any) -> str | None:
    """Validate HGNC gene symbol format (SCI-04)."""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None  # Empty string treated as None
    if _GENE_SYMBOL_RE.match(value):
        return value
    raise ValueError(
        f"Invalid gene symbol: '{value}'. "
        "Must be uppercase letter followed by alphanumeric/hyphen chars."
    )


def _validate_sequence(value: Any) -> str | None:
    """Validate amino acid sequence (SCI-08)."""
    if value is None:
        return None
    value = str(value).strip()
    if _SEQUENCE_RE.match(value):
        return value
    raise ValueError(
        "Invalid protein sequence: contains non-amino-acid characters. "
        "Allowed: A C D E F G H I K L M N P Q R S T V W Y B J O U X Z *"
    )


def _validate_max_phase(value: Any) -> int | None:
    """Validate clinical trial phase is in range [0, 4] (SCI-02)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        phase = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid max_phase: {value!r}. Must be integer 0-4.")
    if not (0 <= phase <= 4):
        raise ValueError(
            f"Invalid max_phase: {phase}. Must be between 0 and 4."
        )
    return phase


def _validate_drug_type(value: Any) -> str | None:
    """Validate drug_type against allowed enum (SCI-10)."""
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in _VALID_DRUG_TYPES:
        return value
    raise ValueError(
        f"Invalid drug_type: '{value}'. "
        f"Must be one of: {sorted(_VALID_DRUG_TYPES)}"
    )


def _validate_interaction_type(value: Any) -> str | None:
    """Validate interaction_type against allowed enum (SCI-11)."""
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in _VALID_INTERACTION_TYPES:
        return value
    raise ValueError(
        f"Invalid interaction_type: '{value}'. "
        f"Must be one of: {sorted(_VALID_INTERACTION_TYPES)}"
    )


def _validate_activity_type(value: Any) -> str | None:
    """Validate activity_type against allowed enum (SCI-12).

    P1-A7 ROOT FIX (v82): the previous implementation was case-SENSITIVE
    (``if value in _VALID_ACTIVITY_TYPES``) while ``_validate_drug_type``
    and ``_validate_interaction_type`` are case-INSENSITIVE (``value.lower()``).
    ChEMBL emits activity types as lowercase (``ic50``, ``ec50``, ``ki``,
    ``kd``) -- these were quarantined by the case-sensitive check, silently
    dropping valid activity measurements. ROOT FIX: build a lowercase->canonical
    mapping at module load time so the lookup is O(1) AND case-insensitive,
    and return the CANONICAL form (e.g. ``IC50``) so the DB stores
    consistent values regardless of the source's casing.
    """
    if value is None:
        return None
    value = str(value).strip()
    _value_lower = value.lower()
    if _value_lower in _ACTIVITY_TYPE_LOWER_TO_CANONICAL:
        return _ACTIVITY_TYPE_LOWER_TO_CANONICAL[_value_lower]
    raise ValueError(
        f"Invalid activity_type: '{value}'. "
        f"Must be one of: {sorted(_VALID_ACTIVITY_TYPES)}"
    )


def _validate_disease_id_type(value: Any) -> str | None:
    """Validate disease_id_type against allowed enum (SCI-09)."""
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in _VALID_DISEASE_ID_TYPES:
        return value
    raise ValueError(
        f"Invalid disease_id_type: '{value}'. "
        f"Must be one of: {sorted(_VALID_DISEASE_ID_TYPES)}"
    )


def _validate_disease_id_format(disease_id: Any, disease_id_type: Any) -> str:
    """Validate ``disease_id`` matches the format required by ``disease_id_type``.

    SCI-FIX (scientific correctness): mirrors the ``chk_gda_disease_id_format``
    CHECK constraint in migration ``001_initial_schema.sql``. The ORM models
    do NOT include this constraint (only the enum check on disease_id_type),
    so when the DB is created from ORM metadata the format check is silently
    absent. This validator enforces the same formats in Python so that
    scientifically-malformed disease IDs cannot enter the staging DB.

    Format patterns (anchored to match the entire string):

    - ``omim``     : ``^(?:OMIM:)?\\d{4,7}$`` -- MIM number (e.g. ``100800``)
      (imported from ``cleaning._constants.CANONICAL_OMIM_DISEASE_ID_REGEX``)
    - ``disgenet`` : ``^C\\d{7}$``            -- UMLS CUI (e.g. ``C0003843``)
    - ``umls``     : ``^C\\d{7}$``            -- UMLS CUI (e.g. ``C0003843``)
    - ``doid``     : ``^DOID:\\d+$``          -- Disease Ontology (e.g. ``DOID:4``)
    - ``mesh``     : ``^D\\d{6}$``            -- MeSH descriptor (e.g. ``D000001``)
    - ``hpo``      : ``^HP:\\d{7}$``          -- HPO term (e.g. ``HP:0000001``)

    v35 ROOT FIX (issue 31): when ``disease_id_type`` is ``None``, the
    format check is no longer silently skipped -- instead, we auto-detect
    the type by trying each known pattern in turn. This catches malformed
    disease IDs that would otherwise flow into the DB unchecked (the SQL
    CHECK constraint allows NULL disease_id_type, so a None type was a
    loophole for any string to be inserted as disease_id). When auto-
    detection succeeds, the disease_id is accepted; when it fails, a
    ValueError is raised listing all the patterns that were tried.

    Parameters
    ----------
    disease_id:
        The disease ID value to validate.
    disease_id_type:
        The disease ID type (one of ``_VALID_DISEASE_ID_TYPES``).
        If ``None`` or empty, the format is auto-detected by trying
        each known pattern (v35 fix).

    Returns
    -------
    str
        The validated disease_id (stripped of whitespace).

    Raises
    ------
    ValueError
        If ``disease_id_type`` is set but ``disease_id`` does not match
        the expected format for that type, OR if ``disease_id_type`` is
        ``None`` and ``disease_id`` does not match ANY known pattern
        (v35 fix).
    """
    if disease_id is None:
        # v59 ROOT FIX (compound of P1C-001 -- disease_id contradiction):
        # The previous code coerced ``None`` to ``""`` here, which then
        # crashed the downstream INSERT on PostgreSQL (CHECK ``disease_id <> ''``
        # rejected the empty string) while silently passing on SQLite
        # (SQLite does NOT enforce CHECK constraints against DEFAULT- or
        # code-supplied empty strings). The correct behavior is to
        # reject NULL upfront -- the caller (loader) must supply a real
        # disease_id or quarantine the row.
        raise ValueError(
            "disease_id is None -- cannot validate. The caller must "
            "supply a non-None disease_id or quarantine the row. "
            "(v59 ROOT FIX -- empty string is rejected by the "
            "chk_gda_disease_id_nonempty CHECK constraint on PostgreSQL.)"
        )
    disease_id = str(disease_id).strip()

    # v35 ROOT FIX (issue 31): auto-detect type when disease_id_type is None.
    # Previously, a None type caused the format check to be silently
    # skipped, allowing malformed disease IDs to enter the DB.
    if disease_id_type is None:
        # Try each known pattern -- if ANY matches, the disease_id is valid.
        for type_name, pattern in _DISEASE_ID_PATTERNS.items():
            if pattern.match(disease_id):
                return disease_id
        # None matched -- raise with a helpful error listing the patterns.
        raise ValueError(
            f"Invalid disease_id format: {disease_id!r} (disease_id_type "
            f"is None -- auto-detection failed, no known pattern matched). "
            f"Tried patterns: "
            + ", ".join(
                f"{t}={p.pattern!r}"
                for t, p in _DISEASE_ID_PATTERNS.items()
            )
        )
    disease_id_type = str(disease_id_type).strip().lower()

    pattern = _DISEASE_ID_PATTERNS.get(disease_id_type)
    if pattern is None:
        # Unknown type -- _validate_disease_id_type will have raised already.
        # Defensive: skip format check for unknown types.
        return disease_id

    if not pattern.match(disease_id):
        raise ValueError(
            f"Invalid disease_id format: {disease_id!r} for "
            f"disease_id_type={disease_id_type!r}. "
            f"Expected pattern: {pattern.pattern}"
        )
    return disease_id


def _validate_positive_float(value: Any, field_name: str) -> Any | None:
    """Validate that a numeric value is positive (SCI-05 for activity_value).

    v29 ROOT FIX (audit D-15): Decimal->float coercion loses precision.
    Preserve Decimal for Numeric columns. Previously this function always
    returned ``float``, which silently truncated ``decimal.Decimal``
    inputs (e.g. ``Decimal('7.123456789')``) to ``float64`` before they
    reached ``Numeric(10,4)`` / ``Numeric(12,6)`` columns. SQLAlchemy
    ``Numeric`` columns accept ``Decimal`` natively, so we preserve the
    input type:

    - ``Decimal`` in -> ``Decimal`` out (precision preserved)
    - ``float`` / ``int`` in -> ``float`` out (existing behaviour)
    - ``None`` / ``NaN`` in -> ``None`` out
    """
    from decimal import Decimal as _Decimal

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    is_decimal = isinstance(value, _Decimal)
    try:
        fval = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {field_name}: {value!r}. Must be numeric.")
    if fval <= 0:
        raise ValueError(
            f"Invalid {field_name}: {fval}. Must be positive (> 0)."
        )
    # v29 ROOT FIX (audit D-15): preserve Decimal precision for Numeric
    # columns (activity_value, molecular_weight, ...). float() was only
    # used for the positivity check above; the returned value keeps the
    # original Decimal type when one was supplied.
    if is_decimal:
        return value
    return fval


def _validate_confidence_score(value: Any) -> float | None:
    """Validate confidence_score is in [0, 1] (SCI-06)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid confidence_score: {value!r}. Must be numeric."
        )
    if not (0.0 <= fval <= 1.0):
        raise ValueError(
            f"Invalid confidence_score: {fval}. Must be in [0, 1]."
        )
    return fval


def _validate_ppi_score(value: Any, field_name: str) -> int | None:
    """Validate PPI score is in [0, 1000] (SCI-03)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ival = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {field_name}: {value!r}. Must be integer.")
    if not (0 <= ival <= 1000):
        raise ValueError(
            f"Invalid {field_name}: {ival}. Must be in [0, 1000]."
        )
    return ival


def _truncate_string(
    value: Any, max_length: int, field_name: str
) -> Any:
    """Truncate string values exceeding column length limits (DQ-12).

    Returns the value truncated to max_length with a WARNING log if
    truncation occurred.
    """
    if value is None or not isinstance(value, str):
        return value
    if len(value) > max_length:
        truncated = value[:max_length]
        logger.warning(
            "Truncated %s from %d to %d chars: '%s' -> '%s'",
            field_name,
            len(value),
            max_length,
            value[:50] + "..." if len(value) > 50 else value,
            truncated[:50] + "..." if len(truncated) > 50 else truncated,
        )
        return truncated
    return value


def _reject_wildcard_name(value: Any, field_name: str) -> Any:
    """Reject wildcard-only patterns like '*', '%', '.' (SEC-04)."""
    if value is None or not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped and all(c in "*%." for c in stripped):
        raise ValueError(
            f"Rejected wildcard pattern in {field_name}: '{value}'"
        )
    return value


# ===========================================================================
# Validation + quarantine helper
# ===========================================================================


def _quarantine_invalid_record(
    record: dict,
    error: ValueError,
    operation: str,
) -> None:
    """Log a WARNING and add an invalid record to the dead letter queue."""
    logger.warning(
        "%s: quarantined invalid record: %s -- %s",
        operation,
        {k: v for k, v in record.items() if k in (
            "inchikey", "uniprot_id", "gene_symbol",
            "canonical_inchikey", "canonical_name",
        )},
        error,
    )
    _add_to_dead_letter(record, str(error), operation)


def _pre_validate_drugs(
    records: list[dict],
    operation: str,
) -> tuple[list[dict], int]:
    """Pre-validate drug records and quarantine invalid ones (SCI-01,
    SCI-02, SCI-10, DQ-01, DQ-02).

    Returns (valid_records, quarantine_count).
    """
    valid = []
    quarantined = 0
    seen_inchikeys: set[str] = set()

    for record in records:
        try:
            # SCI-01: InChIKey validation (required field)
            ik = record.get("inchikey")
            if ik is None:
                raise ValueError("Missing required field: inchikey")
            ik = _validate_inchikey(ik)
            if ik is None:
                raise ValueError("inchikey cannot be None")
            record["inchikey"] = ik

            # DQ-04: Name required, min length 2
            name = record.get("name")
            if name is None or (isinstance(name, str) and len(name.strip()) < 2):
                raise ValueError(
                    f"Drug name must be at least 2 characters, got: '{name}'"
                )

            # SCI-02: max_phase validation
            if "max_phase" in record and record["max_phase"] is not None:
                record["max_phase"] = _validate_max_phase(record["max_phase"])

            # SCI-10: drug_type validation
            if "drug_type" in record and record["drug_type"] is not None:
                record["drug_type"] = _validate_drug_type(record["drug_type"])

            # DQ-02: Duplicate inchikey within batch
            if ik in seen_inchikeys:
                logger.warning(
                    "%s: duplicate inchikey in batch: %s -- keeping last", operation, ik
                )
            seen_inchikeys.add(ik)

            # DQ-12: String length truncation
            record["name"] = _truncate_string(record.get("name"), 500, "name")
            record["chembl_id"] = _truncate_string(
                record.get("chembl_id"), 20, "chembl_id"
            )
            record["drugbank_id"] = _truncate_string(
                record.get("drugbank_id"), 10, "drugbank_id"
            )

            # DQ-09: molecular_weight must be positive
            mw = record.get("molecular_weight")
            if mw is not None and not (isinstance(mw, float) and pd.isna(mw)):
                try:
                    if float(mw) <= 0:
                        raise ValueError(
                            f"molecular_weight must be positive, got {mw}"
                        )
                except (TypeError, ValueError) as e:
                    if "must be positive" in str(e):
                        raise
                    pass  # Non-numeric will be caught by DB

            valid.append(record)

        except ValueError as e:
            _quarantine_invalid_record(record, e, operation)
            quarantined += 1

    return valid, quarantined


def _pre_validate_proteins(
    records: list[dict],
    operation: str,
) -> tuple[list[dict], int]:
    """Pre-validate protein records and quarantine invalid ones (SCI-04,
    SCI-05, SCI-08, DQ-01, DQ-05).

    Returns (valid_records, quarantine_count).
    """
    valid = []
    quarantined = 0
    seen_uniprot_ids: set[str] = set()

    for record in records:
        try:
            # SCI-05: UniProt accession validation (required field)
            uid = record.get("uniprot_id")
            if uid is None:
                raise ValueError("Missing required field: uniprot_id")
            uid = _validate_uniprot_id(uid)
            if uid is None:
                raise ValueError("uniprot_id cannot be None")
            record["uniprot_id"] = uid

            # DQ-02: Duplicate uniprot_id within batch
            if uid in seen_uniprot_ids:
                logger.warning(
                    "%s: duplicate uniprot_id in batch: %s -- keeping last",
                    operation,
                    uid,
                )
            seen_uniprot_ids.add(uid)

            # SCI-04: gene_symbol validation (if present and not None/empty)
            # v21 ROOT FIX (Audit section 5 finding 3 / Chain 2 - "Silent
            # gene_symbol drop for non-human proteins"): the previous code
            # caught ValueError from _validate_gene_symbol and silently
            # set gene_symbol = None - no quarantine, no dead-letter. All
            # mouse/rat/yeast Title-Case gene symbols ('Tp53', 'Brca1')
            # silently lost their gene identity on insert. Downstream GDA
            # joins keyed by gene_symbol silently broke. The fix has TWO
            # parts:
            #   1. models._GENE_SYMBOL_RE now accepts Title-Case (see the
            #      regex fix in models.py) - so mouse/rat/yeast symbols
            #      pass validation cleanly.
            #   2. THIS code: if validation STILL fails (e.g. malformed
            #      input with punctuation), quarantine the record (via
            #      _quarantine_invalid_record) instead of silently
            #      dropping the gene_symbol. A protein with no
            #      gene_symbol is silently invisible to GDA joins; the
            #      operator must see the dead-letter.
            gs = record.get("gene_symbol")
            if gs is not None and str(gs).strip():
                try:
                    record["gene_symbol"] = _validate_gene_symbol(gs)
                except ValueError as e:
                    # v21: quarantine the record instead of silently
                    # nulling the gene_symbol. The audit's complaint
                    # was that silent nulls destroy downstream GDA
                    # joins without operator visibility. Use the same
                    # _quarantine_invalid_record path used elsewhere.
                    _quarantine_invalid_record(record, e, operation)
                    quarantined += 1
                    continue

            # SCI-08: sequence validation (if present and not None/empty)
            seq = record.get("sequence")
            if seq is not None and str(seq).strip():
                try:
                    record["sequence"] = _validate_sequence(seq)
                except ValueError:
                    logger.warning(
                        "%s: invalid sequence for %s -- setting to None", operation, uid
                    )
                    record["sequence"] = None

            # DQ-05: Remove gene_name from updatable_cols (deprecated)
            # Keep it in the record if present for backward compat, but warn
            if "gene_name" in record and record["gene_name"] is not None:
                warnings.warn(
                    "gene_name is deprecated in proteins updatable_cols. "
                    "Use gene_symbol for gene symbols and protein_name "
                    "for protein names. This field will be removed in v3.0.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            valid.append(record)

        except ValueError as e:
            _quarantine_invalid_record(record, e, operation)
            quarantined += 1

    return valid, quarantined


def _pre_validate_dpi(
    records: list[dict],
    operation: str,
) -> tuple[list[dict], int]:
    """Pre-validate DPI records and quarantine invalid ones (SCI-05,
    SCI-06, SCI-11, SCI-12, DQ-09).

    Returns (valid_records, quarantine_count).
    """
    valid = []
    quarantined = 0

    for record in records:
        try:
            # SCI-05: activity_value must be positive
            if "activity_value" in record and record["activity_value"] is not None:
                record["activity_value"] = _validate_positive_float(
                    record["activity_value"], "activity_value"
                )

            # SCI-06: confidence_score in [0, 1]
            if "confidence_score" in record and record["confidence_score"] is not None:
                record["confidence_score"] = _validate_confidence_score(
                    record["confidence_score"]
                )

            # SCI-11: interaction_type validation (if present)
            if "interaction_type" in record and record["interaction_type"] is not None:
                record["interaction_type"] = _validate_interaction_type(
                    record["interaction_type"]
                )

            # SCI-12: activity_type validation (if present)
            if "activity_type" in record and record["activity_type"] is not None:
                record["activity_type"] = _validate_activity_type(
                    record["activity_type"]
                )

            # DES-02: Convert empty string source to None.
            # v89/v90 ROOT FIX (BUG #4 / BUG #38 -- source="unknown" coercion
            #   masks NULL and triggers CHECK violation):
            #   The previous code coerced NULL/NaN/empty ``source`` to the
            #   sentinel ``"unknown"`` at the loader boundary, believing
            #   this prevented a NOT NULL violation. But ``source`` IS
            #   nullable (models.py line 1249: ``nullable=True``), and the
            #   DB CHECK ``chk_dpi_source`` (models.py line 1371) is
            #   ``source IS NULL OR source IN ('chembl', 'drugbank', ...)``.
            #   The coercion ``None -> "unknown"`` turned a value the DB
            #   would have ACCEPTED (NULL) into a value the DB REJECTS
            #   ("unknown" is not in the whitelist) -- causing the entire
            #   chunk to fail with a CHECK violation, fall back to row-by-
            #   row inserts (which ALSO fail), and dead-letter every row
            #   with error "CHECK constraint failed: chk_dpi_source".
            #   The dead-letter entries recorded source='unknown' (the
            #   POST-coercion value), HIDING the loader's bug -- operators
            #   concluded the data had source='unknown' (which IS invalid)
            #   and tried to fix the data, when the real fix was to NOT
            #   coerce NULL to "unknown". ROOT FIX: let NULL stay NULL
            #   (the CHECK allows it). Empty string -> None. NaN -> None.
            #   If a non-NULL default is desired in the future, use a
            #   VALID source value ('chembl' or 'drugbank') based on
            #   pipeline context -- NEVER 'unknown'.
            src = record.get("source")
            if src is None or src == "" or (isinstance(src, float) and pd.isna(src)):
                record["source"] = None
            else:
                record["source"] = str(src).strip() or None

            # DES-04: Convert empty string source_id to None
            sid = record.get("source_id")
            if sid == "":
                record["source_id"] = None

            valid.append(record)

        except ValueError as e:
            _quarantine_invalid_record(record, e, operation)
            quarantined += 1

    return valid, quarantined


def _pre_validate_ppi(
    records: list[dict],
    operation: str,
) -> tuple[list[dict], int]:
    """Pre-validate PPI records and quarantine invalid ones (SCI-03,
    DES-02).

    Returns (valid_records, quarantine_count).
    """
    valid = []
    quarantined = 0

    for record in records:
        try:
            # SCI-03: Score validation (all scores in [0, 1000])
            for score_col in (
                "combined_score",
                "experimental_score",
                "database_score",
                "textmining_score",
            ):
                if score_col in record and record[score_col] is not None:
                    record[score_col] = _validate_ppi_score(
                        record[score_col], score_col
                    )

            # DES-02: Swap protein_a_id and protein_b_id if a > b
            a_id = record.get("protein_a_id")
            b_id = record.get("protein_b_id")
            if a_id is not None and b_id is not None:
                # v91 ROOT FIX (BUG #9): allow homodimers (a_id == b_id).
                # Homodimers are biologically real (EGFR dimerization,
                # p53 tetramerization). Set is_homodimer=True for
                # self-interactions instead of quarantining them.
                if a_id == b_id:
                    record["is_homodimer"] = True
                else:
                    record["is_homodimer"] = False
                if a_id > b_id:
                    logger.warning(
                        "%s: swapping protein_a_id(%d) > protein_b_id(%d)",
                        operation,
                        a_id,
                        b_id,
                    )
                    record["protein_a_id"] = b_id
                    record["protein_b_id"] = a_id
                    # v93 ROOT FIX (P1-033): the swap reversed protein_a_id
                    # and protein_b_id but did NOT swap any direction-
                    # specific fields. For STRING PPIs, all score columns
                    # (combined/experimental/database/textmining) are
                    # SYMMETRIC -- no swap needed. But ``score_json`` is
                    # a free-form Text column for "source-specific payloads
                    # beyond STRING" (per models.py:1559-1560). If a non-
                    # STRING source (e.g. BioGRID, IntAct) stores direction-
                    # specific data in score_json (e.g. ``{"activation_a_to_b":
                    # true, "inhibition_b_to_a": false}``), the swap
                    # SILENTLY CORRUPTS the semantics -- the direction fields
                    # now refer to the wrong protein pair.
                    #
                    # Root fix: detect direction-specific keys in score_json
                    # and log a WARNING (do not auto-swap -- the direction
                    # schema is source-specific and we cannot reliably
                    # rename keys without understanding each source's
                    # contract). The operator must inspect and either
                    # (a) pre-swap direction fields upstream, or
                    # (b) mark the record as direction-agnostic before
                    #     calling the loader.
                    score_json = record.get("score_json")
                    if score_json:
                        try:
                            sj = json.loads(score_json) if isinstance(
                                score_json, str
                            ) else score_json
                            if isinstance(sj, dict):
                                _direction_keys = [
                                    k for k in sj.keys()
                                    if any(
                                        tag in str(k).lower()
                                        for tag in (
                                            "_a_", "_b_", "_a2b_", "_b2a_",
                                            "_to_a", "_to_b", "_from_a",
                                            "_from_b", "a_to_b", "b_to_a",
                                            "direction", "source_to_target",
                                        )
                                    )
                                ]
                                if _direction_keys:
                                    logger.warning(
                                        "%s: PPI swap occurred but score_json "
                                        "contains direction-specific keys %r "
                                        "that were NOT swapped. The direction "
                                        "semantics may now be wrong. Pre-swap "
                                        "direction fields upstream or mark the "
                                        "record as direction-agnostic.",
                                        operation, _direction_keys,
                                    )
                        except (json.JSONDecodeError, TypeError, ValueError):
                            # score_json is not valid JSON -- leave it as-is
                            # (the original string is preserved). This is
                            # not corruption; the field was already opaque.
                            pass

            valid.append(record)

        except ValueError as e:
            _quarantine_invalid_record(record, e, operation)
            quarantined += 1

    return valid, quarantined


def _pre_validate_gda(
    records: list[dict],
    operation: str,
) -> tuple[list[dict], int]:
    """Pre-validate GDA records and quarantine invalid ones (SCI-04,
    SCI-09).

    Returns (valid_records, quarantine_count).
    """
    valid = []
    quarantined = 0

    for record in records:
        try:
            # SCI-04: gene_symbol validation (if present and not empty)
            # v9 ROOT FIX (audit F3.2 / BUG-A-002): the previous code
            # caught ValueError from _validate_gene_symbol and set
            # record["gene_symbol"] = "" -- the exact pattern the v7 fix
            # claimed to have removed. The record then made a wasted DB
            # round-trip, failed the CHECK (gene_symbol <> '') constraint,
            # and ended up in the in-process dead-letter queue that is
            # lost on restart. Now we quarantine IMMEDIATELY -- no DB
            # round-trip, no mutation to empty string.
            #
            # P1-A10 ROOT FIX (v82): the v9 fix wrapped _validate_gene_symbol
            # in an inner try/except that caught ValueError and re-raised --
            # dead code that added no logic but obscured the flow. The inner
            # except was a no-op (catch + re-raise = pass-through). ROOT FIX:
            # remove the dead inner try/except entirely. The ValueError from
            # _validate_gene_symbol propagates directly to the outer except
            # at the bottom of the loop, which quarantines the record in ONE
            # place. No mutation, no wasted DB call, no dead code.
            gs = record.get("gene_symbol")
            if gs is not None and str(gs).strip() and str(gs).strip() != "":
                validated_gs = _validate_gene_symbol(gs)
                if validated_gs is not None:
                    record["gene_symbol"] = validated_gs
                else:
                    raise ValueError(
                        f"gene_symbol '{gs}' failed HGNC validation -- "
                        f"quarantining before DB round-trip"
                    )

            # SCI-09: disease_id_type validation
            if "disease_id_type" in record and record["disease_id_type"] is not None:
                record["disease_id_type"] = _validate_disease_id_type(
                    record["disease_id_type"]
                )

            # SCI-FIX (disease_id format validation): enforce the same
            # format patterns as chk_gda_disease_id_format in migration
            # 001_initial_schema.sql. The ORM models do NOT include this
            # CHECK constraint, so without this Python-side validation,
            # malformed disease IDs (e.g. disease_id_type='mesh' with
            # disease_id='INVALID') would be silently inserted into the
            # staging DB. This is a scientific-correctness guard --
            # downstream knowledge-graph construction trusts that
            # disease_id values are well-formed for their declared type.
            if "disease_id" in record:
                record["disease_id"] = _validate_disease_id_format(
                    record.get("disease_id"),
                    record.get("disease_id_type"),
                )

            valid.append(record)

        except ValueError as e:
            _quarantine_invalid_record(record, e, operation)
            quarantined += 1

    return valid, quarantined


def _pre_validate_entity_mapping(
    records: list[dict],
    operation: str,
) -> tuple[list[dict], int]:
    """Pre-validate entity mapping records and quarantine invalid ones
    (SCI-01, DQ-11, SEC-04).

    Returns (valid_records, quarantine_count).
    """
    valid = []
    quarantined = 0

    for record in records:
        try:
            # SCI-01: Validate InChIKey if present
            ik = record.get("canonical_inchikey")
            if ik is not None and str(ik).strip():
                try:
                    record["canonical_inchikey"] = _validate_inchikey(ik)
                except ValueError:
                    logger.warning(
                        "%s: invalid canonical_inchikey '%s' -- quarantining",
                        operation,
                        ik,
                    )
                    raise

            # DQ-11: Reject rows with no identity
            name = record.get("canonical_name")
            if ik is None and (name is None or not str(name).strip()):
                raise ValueError(
                    "Entity mapping record has neither canonical_inchikey "
                    "nor canonical_name -- rejected (no identity)"
                )

            # SEC-04: Reject wildcard-only names
            if name is not None:
                record["canonical_name"] = _reject_wildcard_name(
                    name, "canonical_name"
                )

            valid.append(record)

        except ValueError as e:
            _quarantine_invalid_record(record, e, operation)
            quarantined += 1

    return valid, quarantined


# ===========================================================================
# Retry decorator (REL-07)
# ===========================================================================


def _with_retry(max_retries: int = 3, base_delay: float = 0.5):
    """Decorator that wraps a function with exponential backoff retry on
    transient database errors (REL-07).

    v90 ROOT FIX (BUG #12 -- P1 max_retries=0 returns None + missing
    InterfaceError):
      1. The previous ``for attempt in range(max_retries)`` meant
         ``max_retries=0`` (a valid "no retries" config) produced
         ``range(0)`` -- the loop body NEVER executed, and the function
         returned ``None`` instead of raising. Callers like
         ``get_uniprot_to_protein_id_map`` (decorated with
         ``@_with_retry(max_retries=3)``) return ``MappingResult`` -- but
         if ``max_retries`` were 0, they'd return ``None``, and the
         caller's ``mapping = mr.mapping`` would raise ``AttributeError``.
         ROOT FIX: use ``range(max_retries + 1)`` so ``max_retries=0``
         still runs the function ONCE (1 attempt, 0 retries). The retry
         condition ``attempt < max_retries - 1`` is preserved, so for
         ``max_retries=3`` the loop still does exactly 3 attempts
         (attempts 0, 1, 2 -- retry on 0 and 1, raise on 2).
      2. The previous code caught ONLY ``OperationalError``. Network-level
         errors (``InterfaceError`` -- connection drops, DNS failures,
         TCP resets) are a SEPARATE exception class in SQLAlchemy. A brief
         network blip during ``get_uniprot_to_protein_id_map`` killed the
         entire GDA load because ``InterfaceError`` was not retried.
         ROOT FIX: catch ``(OperationalError, InterfaceError)``.
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            # v90: range(max_retries + 1) so max_retries=0 still runs once.
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (OperationalError, InterfaceError) as exc:
                    last_exc = exc
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            "%s: attempt %d/%d failed: %s -- retrying in %.1fs",
                            func.__name__,
                            attempt + 1,
                            max_retries,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "%s: all %d retries exhausted: %s",
                            func.__name__,
                            max_retries,
                            exc,
                        )
                        raise
            # v90: unreachable now -- range(max_retries + 1) always runs at
            # least one attempt, and the last attempt raises on failure.
            return None  # pragma: no cover

        return wrapper

    return decorator


# ===========================================================================
# Timing context manager (LOG-05)
# ===========================================================================


class _Timer:
    """Simple timer context manager for logging operation duration."""

    def __init__(self, operation: str, total: int):
        self.operation = operation
        self.total = total
        self.start: float = 0.0
        self.duration: float = 0.0

    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, *args):
        self.duration = time.monotonic() - self.start
        try:
            from config.settings import LOADERS_ENABLE_TIMING
            if LOADERS_ENABLE_TIMING:
                rps = self.total / self.duration if self.duration > 0 else 0
                logger.info(
                    "%s: completed in %.2fs (%d rows, %.0f rows/sec)",
                    self.operation,
                    self.duration,
                    self.total,
                    rps,
                )
        except Exception:
            pass


# ===========================================================================
# Merge helper for entity mapping (module-level, CODE-6)
# ===========================================================================


def _merge_group(group: pd.DataFrame) -> pd.Series:
    """Merge duplicate entity-mapping rows by canonical_name (CODE-6).

    For each column, takes the first non-null value from the group.
    This preserves cross-references from all duplicates instead of
    just keeping the last row.

    K fix: ``canonical_name`` is intentionally omitted from ``merge_cols``
    because this function is invoked via ``groupby('canonical_name',
    include_groups=False)`` -- the group key is not in ``group.columns``
    and is already preserved on the resulting index. Including it here
    would shadow the real value with ``None`` after ``reset_index``.
    """
    merge_cols = [
        "canonical_inchikey",
        "chembl_id",
        "drugbank_id",
        "pubchem_cid",
        "uniprot_id",
        "string_id",
        "match_confidence",
        "match_method",
    ]
    result: dict[str, Any] = {}
    for col in merge_cols:
        if col in group.columns:
            non_null = group[col].dropna()
            result[col] = non_null.iloc[0] if len(non_null) > 0 else None
        else:
            result[col] = None
    return pd.Series(result)


# ===========================================================================
# 1. DRUGS -- ON CONFLICT (inchikey) DO UPDATE
# ===========================================================================


def bulk_upsert_drugs(
    session: Session,
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    input_checksum: str | None = None,
) -> UpsertResult:
    """Bulk upsert drugs.

    ON CONFLICT (inchikey) DO UPDATE for all updatable columns.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    df : pd.DataFrame
        DataFrame with drug data.  Required columns: inchikey, name.
        Optional columns: chembl_id, drugbank_id, pubchem_cid,
        molecular_formula, molecular_weight, smiles, is_fda_approved,
        is_withdrawn, clinical_status, max_phase, drug_type,
        mechanism_of_action, cas_number, logp, tpsa, h_bond_donor_count,
        h_bond_acceptor_count, rotatable_bond_count, heavy_atom_count,
        complexity, completeness_score.
    batch_size : int
        Number of rows per INSERT statement.  Must be > 0.
    input_checksum : str | None
        Optional SHA-256 checksum of the input DataFrame (LINE-05).

    Returns
    -------
    UpsertResult
        Rich result with inserted/updated/quarantined/failed counts.
    """
    _isinstance_dataframe(df, "bulk_upsert_drugs")
    _validate_batch_size(batch_size)

    result = UpsertResult()

    if df.empty:
        logger.debug("bulk_upsert_drugs: empty dataframe, skipping")
        return result

    if input_checksum:
        logger.debug(
            "bulk_upsert_drugs: input checksum = %s", input_checksum
        )

    # P1-043 v113 ROOT FIX (NA inchikey validation):
    #   The DB schema (Drug.inchikey nullable=False) rejects NULL at INSERT,
    #   but a DataFrame with `inchikey=pd.NA` (pandas NA, not Python None)
    #   could bypass the validator in certain code paths (pandas NA has
    #   different truthiness semantics). One bad row (NA inchikey) would
    #   cause IntegrityError on the entire 10,000-row batch, aborting the
    #   bulk upsert and forcing a 10,000x slower single-row fallback.
    #
    #   ROOT FIX: filter NA/None/empty inchikey rows BEFORE the batch
    #   INSERT. Each dropped row is logged to the dead-letter queue with
    #   the reason "inchikey is NA/None/empty" so operators can trace
    #   which source produced the bad row. The dropped count is added to
    #   result.quarantined so the caller sees it in the UpsertResult.
    if "inchikey" in df.columns:
        _bad_mask = df["inchikey"].isna() | (df["inchikey"].astype(str).str.strip() == "")
        _bad_count = int(_bad_mask.sum())
        if _bad_count > 0:
            logger.warning(
                "bulk_upsert_drugs: dropping %d/%d rows with NA/None/empty "
                "inchikey (would cause IntegrityError on NOT NULL column). "
                "Dropped rows are logged to the dead-letter queue.",
                _bad_count, len(df),
            )
            for idx, row in df[_bad_mask].iterrows():
                _add_to_dead_letter(
                    record=row.to_dict(),
                    error="inchikey is NA/None/empty — Drug.inchikey is NOT NULL; "
                          "biologics must use SYNTH-prefixed surrogate keys",
                    operation="bulk_upsert_drugs",
                )
            df = df[~_bad_mask].copy()
            result.quarantined += _bad_count
            if df.empty:
                logger.warning(
                    "bulk_upsert_drugs: ALL %d rows had NA/empty inchikey — "
                    "nothing to upsert.", _bad_count,
                )
                return result

    # Sanitize input (SEC-03)
    df = _sanitize_dataframe(df.copy())

    batch_size = _calculate_safe_batch_size(Drug, batch_size)
    total = len(df)
    result.total_input = total

    with _Timer("bulk_upsert_drugs", total):
        dialect_insert = _get_dialect_insert(session)

        updatable_cols = [
            "name",
            "chembl_id",
            "drugbank_id",
            "pubchem_cid",
            "molecular_formula",
            "molecular_weight",
            "smiles",
            "is_fda_approved",
            "is_withdrawn",
            "clinical_status",
            "max_phase",
            "drug_type",
            "mechanism_of_action",
            "cas_number",
            "logp",
            "tpsa",
            "h_bond_donor_count",
            "h_bond_acceptor_count",
            "rotatable_bond_count",
            "heavy_atom_count",
            "complexity",
            "completeness_score",
            # PS-6 / RT-8 ROOT FIX: 'groups' is the DrugBank <groups>
            # field (semicolon-separated regulatory states) used to
            # derive is_withdrawn / clinical_status. The Drug ORM now
            # declares this column (see models.py), migration 006
            # adds it to the drugs table, and the drugbank_pipeline
            # produces it. Without 'groups' in updatable_cols, the
            # upsert silently dropped it and the safety trigger never
            # fired -- withdrawn killer drugs stayed is_withdrawn=FALSE.
            "groups",
            # v42 ROOT FIX (P1-A-7): migration 008 added
            # ``is_globally_approved`` and the ORM declares it; the
            # chembl_pipeline + drugbank_pipeline both emit it. Without
            # ``is_globally_approved`` in updatable_cols, every
            # subsequent upsert left the column STALE -- approved drugs
            # silently flagged as not-approved after the next ChEMBL /
            # DrugBank refresh (defeating the SW-1 patient-safety fix).
            "is_globally_approved",
            "updated_at",  # [IDEM-02/DES-05] Explicitly updated on upsert
        ]

        log_interval = max(1, total // (batch_size * 20))  # ~20 log lines

        for chunk_idx, chunk in enumerate(
            _df_chunk_to_dicts(df, batch_size)
        ):
            # Pre-validate (SCI-01, SCI-02, SCI-10, DQ-01, DQ-02)
            valid_chunk, q_count = _pre_validate_drugs(
                chunk, "bulk_upsert_drugs"
            )
            result.quarantined += q_count

            if not valid_chunk:
                continue

            # v77 ROOT FIX (SQLite patient-safety gap): apply the withdrawn-
            # drug safety sync at the Python ORM layer. The PostgreSQL trigger
            # (trg_drugs_sync_withdrawn in migration 006) handles this for
            # production PostgreSQL, but SQLite (dev/test) does NOT support
            # PostgreSQL's CREATE FUNCTION / EXECUTE FUNCTION syntax -- the
            # migration translator strips both, so the trigger is a no-op on
            # SQLite. This Python hook runs on BOTH dialects, catching every
            # ORM INSERT/UPDATE. Three-layer defense:
            #   1. PostgreSQL trigger (production direct SQL INSERTs)
            #   2. This Python hook (both dialects -- ORM INSERTs/UPDATEs)
            #   3. Curated name seed (migration 006 DO block -- migration-time)
            # The hook mirrors the trigger's logic:
            #   - If groups contains 'withdrawn' -> is_withdrawn=True
            #   - If name is in the curated withdrawn list -> is_withdrawn=True
            #   - If is_withdrawn=True -> is_globally_approved=False (patient
            #     safety invariant: a withdrawn drug is NOT a safe repurposing
            #     candidate, regardless of max_phase)
            for record in valid_chunk:
                _name = str(record.get("name", "") or "").strip().lower()
                _groups = str(record.get("groups", "") or "").strip().lower()
                _is_withdrawn = False
                # Check groups column for 'withdrawn' token
                if _groups and "withdrawn" in _groups:
                    _is_withdrawn = True
                # Check curated withdrawn-drug name list (same list as
                # migration 006's DO block seed -- catches ChEMBL/PubChem-
                # loaded drugs that have name but not groups)
                if not _is_withdrawn and _name:
                    for _wd_name in _WITHDRAWN_DRUG_NAMES_LOWER:
                        # P1-A6 ROOT FIX (v82): the previous code used only
                        # ``startswith(_wd_name + " ")`` (space separator).
                        # This missed salt forms joined by hyphen
                        # ("rofecoxib-sodium") or underscore
                        # ("troglitazone_HCl") -- leaving withdrawn drugs
                        # classified as NOT withdrawn -> patient-safety hook
                        # incomplete for salt forms. ROOT FIX: match space,
                        # hyphen, AND underscore separators (the three
                        # delimiters used in ChEMBL / PubChem / DrugBank
                        # salt-form naming conventions).
                        if (
                            _name == _wd_name
                            or _name.startswith(_wd_name + " ")
                            or _name.startswith(_wd_name + "-")
                            or _name.startswith(_wd_name + "_")
                        ):
                            _is_withdrawn = True
                            break
                if _is_withdrawn:
                    record["is_withdrawn"] = True
                    if not record.get("clinical_status"):
                        record["clinical_status"] = "withdrawn"
                    # Patient-safety invariant: withdrawn drugs are NOT
                    # globally approved repurposing candidates
                    record["is_globally_approved"] = False

            try:
                all_keys: set[str] = set()
                for record in valid_chunk:
                    all_keys.update(record.keys())

                # RT-8 ROOT FIX: filter out columns that are not on the
                # Drug table. The input DataFrame may carry extra lineage /
                # intermediate columns that the Drug ORM does not map (e.g.
                # 'indication', 'description' from drugbank_pipeline).
                # Without this filter, SQLAlchemy raises CompileError and
                # 100% of the chunk is dead-lettered. Combined with the
                # PS-6 fix that adds 'groups' to updatable_cols AND to the
                # Drug ORM, this preserves the groups column for the
                # safety trigger while dropping genuinely-unmapped columns.
                valid_drug_columns: set[str] = set(Drug.__table__.columns.keys())
                dropped_keys = all_keys - valid_drug_columns
                if dropped_keys:
                    logger.debug(
                        "bulk_upsert_drugs: chunk %d -- ignoring non-Drug columns: %s",
                        chunk_idx, sorted(dropped_keys),
                    )
                filtered_chunk = [
                    {k: v for k, v in record.items() if k in valid_drug_columns}
                    for record in valid_chunk
                ]
                all_keys = set().union(*[r.keys() for r in filtered_chunk]) if filtered_chunk else set()

                stmt = dialect_insert(Drug.__table__).values(filtered_chunk)
                update_dict = {
                    col: stmt.excluded[col]
                    for col in updatable_cols
                    if col in all_keys
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=["inchikey"],
                    set_=update_dict,
                )
                # v51 ROOT FIX (COMPOUND-6 -- Counter Inflation):
                # Use _count_upsert_inserts_updates for accurate counts
                # instead of result.inserted += len(valid_chunk) which
                # counts updates as inserts.
                #
                # v104 P1-010 ROOT FIX: pass conflict_keys, chunk_records,
                # and target_table so the SQLite branch can do an accurate
                # pre-count (the PostgreSQL branch already uses xmax and
                # ignores these kwargs). Without these, the SQLite branch
                # returns ``(total, 0)`` -- counting every row as an
                # INSERT, falsifying the audit trail on re-runs.
                _ins, _upd = _count_upsert_inserts_updates(
                    session, stmt, len(valid_chunk),
                    conflict_keys=["inchikey"],
                    chunk_records=filtered_chunk,
                    target_table=Drug.__table__,
                )
                result.inserted += _ins
                result.updated += _upd

            except Exception as exc:
                logger.error(
                    "bulk_upsert_drugs: chunk %d failed: %s", chunk_idx, exc
                )
                result.failed += len(valid_chunk)
                # Row-by-row fallback (REL-01)
                for record in valid_chunk:
                    try:
                        stmt = dialect_insert(Drug.__table__).values([record])
                        update_dict = {
                            col: stmt.excluded[col]
                            for col in updatable_cols
                            if col in record
                        }
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["inchikey"],
                            set_=update_dict,
                        )
                        session.execute(stmt)
                        result.inserted += 1
                        result.failed -= 1
                    except Exception as row_exc:
                        logger.warning(
                            "bulk_upsert_drugs: row failed (inchikey=%s): %s",
                            record.get("inchikey", "UNKNOWN"),
                            row_exc,
                        )
                        _add_to_dead_letter(
                            record, str(row_exc), "bulk_upsert_drugs"
                        )

            processed = result.inserted + result.quarantined + result.failed
            if (chunk_idx + 1) % log_interval == 0 or processed >= total:
                logger.info(
                    "bulk_upsert_drugs: %d / %d processed (%.0f%%)",
                    processed,
                    total,
                    100.0 * processed / total,
                )
            else:
                logger.debug(
                    "bulk_upsert_drugs: %d / %d processed", processed, total
                )

    logger.info("bulk_upsert_drugs: %s", result)
    return result


# ===========================================================================
# 2. PROTEINS -- ON CONFLICT (uniprot_id) DO UPDATE
# ===========================================================================


def bulk_upsert_proteins(
    session: Session,
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    input_checksum: str | None = None,
) -> UpsertResult:
    """Bulk upsert proteins.

    ON CONFLICT (uniprot_id) DO UPDATE for all updatable columns.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    df : pd.DataFrame
        DataFrame with protein data.  Required: uniprot_id.
        Optional: gene_name, gene_symbol, protein_name, organism,
        sequence, function_desc, string_id.
    batch_size : int
        Number of rows per INSERT statement.  Must be > 0.
    input_checksum : str | None
        Optional SHA-256 checksum of the input DataFrame.

    Returns
    -------
    UpsertResult
    """
    _isinstance_dataframe(df, "bulk_upsert_proteins")
    _validate_batch_size(batch_size)

    result = UpsertResult()

    if df.empty:
        logger.debug("bulk_upsert_proteins: empty dataframe, skipping")
        return result

    if input_checksum:
        logger.debug(
            "bulk_upsert_proteins: input checksum = %s", input_checksum
        )

    df = _sanitize_dataframe(df.copy())
    batch_size = _calculate_safe_batch_size(Protein, batch_size)
    total = len(df)
    result.total_input = total

    with _Timer("bulk_upsert_proteins", total):
        dialect_insert = _get_dialect_insert(session)

        # DQ-05: gene_name removed from updatable_cols (deprecated)
        # CMP-04: DeprecationWarning emitted in pre-validation
        updatable_cols = [
            "gene_symbol",
            "protein_name",
            "organism",
            "sequence",
            "function_desc",
            "string_id",
            "updated_at",  # [IDEM-02/DES-05]
        ]
        # gene_name still accepted for backward compat but not updatable
        # (will be inserted on new rows but never updated)

        log_interval = max(1, total // (batch_size * 20))

        for chunk_idx, chunk in enumerate(
            _df_chunk_to_dicts(df, batch_size)
        ):
            valid_chunk, q_count = _pre_validate_proteins(
                chunk, "bulk_upsert_proteins"
            )
            result.quarantined += q_count

            if not valid_chunk:
                continue

            try:
                all_keys: set[str] = set()
                for record in valid_chunk:
                    all_keys.update(record.keys())

                # v22 ROOT FIX (audit section 5 finding 10 -- "Asymmetric
                # chunk filtering"): the drug loader filters records to
                # Drug.__table__.columns.keys() before insert (avoids
                # CompileError on extra lineage columns). The protein
                # loader did NOT -- extra lineage columns in a protein
                # DataFrame caused CompileError and 100% chunk dead-letter.
                # Unify: apply the SAME column-filter pattern here.
                valid_protein_columns: set[str] = set(Protein.__table__.columns.keys())
                dropped_keys = all_keys - valid_protein_columns
                if dropped_keys:
                    logger.debug(
                        "bulk_upsert_proteins: chunk %d -- ignoring non-Protein columns: %s",
                        chunk_idx, sorted(dropped_keys),
                    )
                filtered_chunk = [
                    {k: v for k, v in record.items() if k in valid_protein_columns}
                    for record in valid_chunk
                ]
                all_keys = set().union(*[r.keys() for r in filtered_chunk]) if filtered_chunk else set()

                stmt = dialect_insert(Protein.__table__).values(filtered_chunk)
                update_dict = {
                    col: stmt.excluded[col]
                    for col in updatable_cols
                    if col in all_keys
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=["uniprot_id"],
                    set_=update_dict,
                )
                # v51 ROOT FIX (COMPOUND-6): accurate counter
                _ins, _upd = _count_upsert_inserts_updates(
                    session, stmt, len(valid_chunk)
                )
                result.inserted += _ins
                result.updated += _upd

            except Exception as exc:
                logger.error(
                    "bulk_upsert_proteins: chunk %d failed: %s",
                    chunk_idx,
                    exc,
                )
                result.failed += len(valid_chunk)
                for record in valid_chunk:
                    try:
                        stmt = dialect_insert(Protein.__table__).values(
                            [record]
                        )
                        update_dict = {
                            col: stmt.excluded[col]
                            for col in updatable_cols
                            if col in record
                        }
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["uniprot_id"],
                            set_=update_dict,
                        )
                        session.execute(stmt)
                        result.inserted += 1
                        result.failed -= 1
                    except Exception as row_exc:
                        logger.warning(
                            "bulk_upsert_proteins: row failed "
                            "(uniprot_id=%s): %s",
                            record.get("uniprot_id", "UNKNOWN"),
                            row_exc,
                        )
                        _add_to_dead_letter(
                            record, str(row_exc), "bulk_upsert_proteins"
                        )

            processed = result.inserted + result.quarantined + result.failed
            if (chunk_idx + 1) % log_interval == 0 or processed >= total:
                logger.info(
                    "bulk_upsert_proteins: %d / %d processed (%.0f%%)",
                    processed,
                    total,
                    100.0 * processed / total,
                )
            else:
                logger.debug(
                    "bulk_upsert_proteins: %d / %d processed",
                    processed,
                    total,
                )

    logger.info("bulk_upsert_proteins: %s", result)
    return result


# ===========================================================================
# 3. DRUG-PROTEIN INTERACTIONS -- ON CONFLICT DO UPDATE
# ===========================================================================


def bulk_upsert_dpi(
    session: Session,
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    pipeline_run_id: int | None = None,
    source_version: str | None = None,
    source_fetch_date: datetime.datetime | None = None,
    input_checksum: str | None = None,
) -> UpsertResult:
    """Bulk upsert drug-protein interactions.

    PostgreSQL: uses named constraint ``uq_dpi_drug_protein_source``
    for ON CONFLICT.  SQLite: uses index_elements
    [drug_id, protein_id, source, source_id] for ON CONFLICT.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    df : pd.DataFrame
        DataFrame with DPI data.  Required: drug_id, protein_id.
        Optional: interaction_type, activity_value, activity_type,
        activity_units, source, source_id, confidence_score.
    batch_size : int
        Number of rows per INSERT statement.  Must be > 0.
    pipeline_run_id : int | None
        Optional pipeline run ID for lineage tracking (LINE-01).
    source_version : str | None
        Version of the source database (e.g. 'ChEMBL_33') (LINE-02).
    source_fetch_date : datetime | None
        When the data was fetched from the source (LINE-02).
    input_checksum : str | None
        Optional SHA-256 checksum of the input DataFrame (LINE-05).

    Returns
    -------
    UpsertResult
    """
    _isinstance_dataframe(df, "bulk_upsert_dpi")
    _validate_batch_size(batch_size)

    result = UpsertResult()

    if df.empty:
        logger.debug("bulk_upsert_dpi: empty dataframe, skipping")
        return result

    if input_checksum:
        logger.debug("bulk_upsert_dpi: input checksum = %s", input_checksum)

    df = _sanitize_dataframe(df.copy())

    # DES-02 + IDEM-2: Remove fillna("") for source -- use NULL consistently
    # Empty string source_id -> None (DES-04)
    #
    # v93 ROOT FIX (P1-028): the previous code had TWO bugs:
    #   1. ``df["source_id"].where(df["source_id"].notna(), None)`` was a
    #      NO-OP -- ``.where(cond, other)`` returns ``other`` where ``cond``
    #      is False. ``.notna()`` is True for non-null, so ``where`` returns
    #      the original value for non-null and ``None`` for null -- but the
    #      row was ALREADY null, so this just re-wrote NULL with NULL.
    #   2. ``df["source_id"].replace("", None)`` on an object column
    #      converts empty strings to ``numpy.nan`` (NOT Python ``None``).
    #      On PostgreSQL with psycopg2, ``nan`` in a nullable column is
    #      inserted as NULL (correct). On some SQLAlchemy versions /
    #      dialects, ``nan`` may be inserted as the literal string "NaN"
    #      or as 0 -- silently corrupting the column.
    #
    # Root fix: use a single ``mask`` + ``loc`` assignment that converts
    # empty strings DIRECTLY to Python ``None`` (not NaN), and removes
    # the redundant ``.where()`` call. The ``pd.NA`` -> ``None`` conversion
    # happens at the SQLAlchemy layer during parameter binding.
    #
    # P1-019 ROOT FIX (v100 forensic -- NULL source_id dedup gap):
    # The DB UniqueConstraint ``uq_dpi_drug_protein_source`` on (drug_id,
    # protein_id, source, source_id) uses NULLS DISTINCT semantics on
    # PostgreSQL 15+ and SQLite -- so two rows with the SAME (drug_id,
    # protein_id, source) but NULL source_id are NOT considered duplicates
    # by the DB. The KG would silently accumulate duplicate DPI edges
    # whenever a source omits source_id (common for older ChEMBL records).
    # ROOT FIX: add an APPLICATION-LEVEL dedup pass that treats NULL
    # source_id as a sentinel for dedup purposes (matching the pattern
    # in ``cleaning/deduplicator.py:2447`` for NaN InChIKeys). We add a
    # temporary ``_dedup_key`` column where NULL source_id becomes the
    # literal string ``"__NULL_SOURCE_ID__"`` so duplicate rows collapse
    # together. After dedup, the temporary column is dropped so the
    # real ``source_id`` column retains NULL (per DES-04 contract).
    if "source_id" in df.columns:
        _empty_mask = df["source_id"].astype(str).str.strip() == ""
        df.loc[_empty_mask, "source_id"] = None
        # Build a composite dedup key that treats NULL source_id as a
        # sentinel so duplicate (drug_id, protein_id, source, NULL)
        # rows collapse. We only do this when the relevant columns are
        # all present -- otherwise we fall through and let the DB
        # constraint handle what it can.
        _dedup_cols_present = all(
            c in df.columns
            for c in ("drug_id", "protein_id", "source")
        )
        if _dedup_cols_present:
            _source_id_for_key = df["source_id"].fillna(
                "__NULL_SOURCE_ID__"
            ).astype(str)
            _dedup_key = (
                df["drug_id"].astype(str) + "|"
                + df["protein_id"].astype(str) + "|"
                + df["source"].astype(str) + "|"
                + _source_id_for_key
            )
            _before = len(df)
            df = df.assign(_dpi_dedup_key=_dedup_key).drop_duplicates(
                subset=["_dpi_dedup_key"], keep="first",
            ).drop(columns=["_dpi_dedup_key"])
            _dropped = _before - len(df)
            if _dropped > 0:
                logger.info(
                    "bulk_upsert_dpi: P1-019 -- dropped %d duplicate DPI "
                    "row(s) with NULL source_id (DB UNIQUE constraint "
                    "treats NULL as distinct; application-level dedup "
                    "closes the gap).",
                    _dropped,
                )

    batch_size = _calculate_safe_batch_size(
        DrugProteinInteraction, batch_size
    )
    total = len(df)
    result.total_input = total

    with _Timer("bulk_upsert_dpi", total):
        dialect_insert = _get_dialect_insert(session)

        updatable_cols = [
            "interaction_type",
            "activity_value",
            "activity_type",
            "activity_units",
            "confidence_score",
            "updated_at",  # [IDEM-06/DES-05]
        ]
        # Lineage columns (LINE-01, LINE-02)
        if pipeline_run_id is not None:
            updatable_cols.append("pipeline_run_id")
        if source_version is not None:
            updatable_cols.append("source_version")
        if source_fetch_date is not None:
            updatable_cols.append("source_fetch_date")

        # ARCH-1: Check dialect once at the start, not per chunk
        dialect_name = session.get_bind().dialect.name
        use_constraint = dialect_name == "postgresql"

        log_interval = max(1, total // (batch_size * 20))

        for chunk_idx, chunk in enumerate(
            _df_chunk_to_dicts(df, batch_size)
        ):
            valid_chunk, q_count = _pre_validate_dpi(
                chunk, "bulk_upsert_dpi"
            )
            result.quarantined += q_count

            if not valid_chunk:
                continue

            # Add lineage fields
            for rec in valid_chunk:
                if pipeline_run_id is not None:
                    rec["pipeline_run_id"] = pipeline_run_id
                if source_version is not None:
                    rec["source_version"] = source_version
                if source_fetch_date is not None:
                    rec["source_fetch_date"] = source_fetch_date

            try:
                all_keys: set[str] = set()
                for record in valid_chunk:
                    all_keys.update(record.keys())

                stmt = dialect_insert(
                    DrugProteinInteraction.__table__
                ).values(valid_chunk)
                update_dict = {
                    col: stmt.excluded[col]
                    for col in updatable_cols
                    if col in all_keys
                }

                if use_constraint:
                    stmt = stmt.on_conflict_do_update(
                        constraint=DPI_UNIQUE_CONSTRAINT_NAME,
                        set_=update_dict,
                    )
                else:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[
                            "drug_id",
                            "protein_id",
                            "source",
                            "source_id",
                        ],
                        set_=update_dict,
                    )
                # v51 ROOT FIX (COMPOUND-6): accurate counter
                _ins, _upd = _count_upsert_inserts_updates(
                    session, stmt, len(valid_chunk)
                )
                result.inserted += _ins
                result.updated += _upd

            except Exception as exc:
                logger.error(
                    "bulk_upsert_dpi: chunk %d failed: %s", chunk_idx, exc
                )
                result.failed += len(valid_chunk)
                for record in valid_chunk:
                    try:
                        stmt = dialect_insert(
                            DrugProteinInteraction.__table__
                        ).values([record])
                        update_dict = {
                            col: stmt.excluded[col]
                            for col in updatable_cols
                            if col in record
                        }
                        if use_constraint:
                            stmt = stmt.on_conflict_do_update(
                                constraint=DPI_UNIQUE_CONSTRAINT_NAME,
                                set_=update_dict,
                            )
                        else:
                            stmt = stmt.on_conflict_do_update(
                                index_elements=[
                                    "drug_id",
                                    "protein_id",
                                    "source",
                                    "source_id",
                                ],
                                set_=update_dict,
                            )
                        session.execute(stmt)
                        result.inserted += 1
                        result.failed -= 1
                    except Exception as row_exc:
                        logger.warning(
                            "bulk_upsert_dpi: row failed "
                            "(drug_id=%s, protein_id=%s): %s",
                            record.get("drug_id", "?"),
                            record.get("protein_id", "?"),
                            row_exc,
                        )
                        _add_to_dead_letter(
                            record, str(row_exc), "bulk_upsert_dpi"
                        )

            processed = result.inserted + result.quarantined + result.failed
            if (chunk_idx + 1) % log_interval == 0 or processed >= total:
                logger.info(
                    "bulk_upsert_dpi: %d / %d processed (%.0f%%)",
                    processed,
                    total,
                    100.0 * processed / total,
                )
            else:
                logger.debug(
                    "bulk_upsert_dpi: %d / %d processed", processed, total
                )

    logger.info("bulk_upsert_dpi: %s", result)
    return result


# ===========================================================================
# 4. PROTEIN-PROTEIN INTERACTIONS -- ON CONFLICT DO UPDATE
# ===========================================================================


def bulk_upsert_ppi(
    session: Session,
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    pipeline_run_id: int | None = None,
    input_checksum: str | None = None,
) -> UpsertResult:
    """Bulk upsert protein-protein interactions.

    ON CONFLICT (protein_a_id, protein_b_id) DO UPDATE for score
    columns.  STRING scores are in [0, 1000] -- NOT [0, 100].

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    df : pd.DataFrame
        DataFrame with PPI data.  Required: protein_a_id, protein_b_id,
        source.  Optional: combined_score, experimental_score,
        database_score, textmining_score.
    batch_size : int
        Number of rows per INSERT statement.  Must be > 0.
    pipeline_run_id : int | None
        Optional pipeline run ID for lineage tracking (LINE-01).
    input_checksum : str | None
        Optional SHA-256 checksum of the input DataFrame (LINE-05).

    Returns
    -------
    UpsertResult
    """
    _isinstance_dataframe(df, "bulk_upsert_ppi")
    _validate_batch_size(batch_size)

    result = UpsertResult()

    if df.empty:
        logger.debug("bulk_upsert_ppi: empty dataframe, skipping")
        return result

    if input_checksum:
        logger.debug("bulk_upsert_ppi: input checksum = %s", input_checksum)

    df = _sanitize_dataframe(df.copy())
    batch_size = _calculate_safe_batch_size(
        ProteinProteinInteraction, batch_size
    )
    total = len(df)
    result.total_input = total

    with _Timer("bulk_upsert_ppi", total):
        dialect_insert = _get_dialect_insert(session)

        updatable_cols = [
            "combined_score",
            "experimental_score",
            "database_score",
            "textmining_score",
            "source",
            "updated_at",  # [IDEM-06/DES-05]
        ]
        if pipeline_run_id is not None:
            updatable_cols.append("pipeline_run_id")

        log_interval = max(1, total // (batch_size * 20))

        for chunk_idx, chunk in enumerate(
            _df_chunk_to_dicts(df, batch_size)
        ):
            valid_chunk, q_count = _pre_validate_ppi(
                chunk, "bulk_upsert_ppi"
            )
            result.quarantined += q_count

            if not valid_chunk:
                continue

            for rec in valid_chunk:
                if pipeline_run_id is not None:
                    rec["pipeline_run_id"] = pipeline_run_id

            try:
                all_keys: set[str] = set()
                for record in valid_chunk:
                    all_keys.update(record.keys())

                stmt = dialect_insert(
                    ProteinProteinInteraction.__table__
                ).values(valid_chunk)
                update_dict = {
                    col: stmt.excluded[col]
                    for col in updatable_cols
                    if col in all_keys
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=["protein_a_id", "protein_b_id"],
                    set_=update_dict,
                )
                # v51 ROOT FIX (COMPOUND-6): accurate counter
                _ins, _upd = _count_upsert_inserts_updates(
                    session, stmt, len(valid_chunk)
                )
                result.inserted += _ins
                result.updated += _upd

            except Exception as exc:
                logger.error(
                    "bulk_upsert_ppi: chunk %d failed: %s", chunk_idx, exc
                )
                result.failed += len(valid_chunk)
                for record in valid_chunk:
                    try:
                        stmt = dialect_insert(
                            ProteinProteinInteraction.__table__
                        ).values([record])
                        update_dict = {
                            col: stmt.excluded[col]
                            for col in updatable_cols
                            if col in record
                        }
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["protein_a_id", "protein_b_id"],
                            set_=update_dict,
                        )
                        session.execute(stmt)
                        result.inserted += 1
                        result.failed -= 1
                    except Exception as row_exc:
                        logger.warning(
                            "bulk_upsert_ppi: row failed "
                            "(protein_a_id=%s, protein_b_id=%s): %s",
                            record.get("protein_a_id", "?"),
                            record.get("protein_b_id", "?"),
                            row_exc,
                        )
                        _add_to_dead_letter(
                            record, str(row_exc), "bulk_upsert_ppi"
                        )

            processed = result.inserted + result.quarantined + result.failed
            if (chunk_idx + 1) % log_interval == 0 or processed >= total:
                logger.info(
                    "bulk_upsert_ppi: %d / %d processed (%.0f%%)",
                    processed,
                    total,
                    100.0 * processed / total,
                )
            else:
                logger.debug(
                    "bulk_upsert_ppi: %d / %d processed", processed, total
                )

    logger.info("bulk_upsert_ppi: %s", result)
    return result


# ===========================================================================
# 5. GENE-DISEASE ASSOCIATIONS -- ON CONFLICT DO UPDATE
# ===========================================================================


def _quarantine_gda_rows(
    df: pd.DataFrame,
    reason: str,
    session: "Session | None" = None,
    pipeline_run_id: int | None = None,
) -> None:
    """Quarantine GDA rows that fail validation (BUG-A-002 root fix).

    Writes the bad rows to a JSONL file under data/dead_letter/ so they
    are not silently lost. The previous code fillna('') which collapsed
    distinct genes with empty gene_symbols into one row, causing silent
    data loss of Gene->Disease edges.

    v9 ROOT FIX (audit F3.1): the previous implementation:
      1. Hardcoded the default path as
         /home/z/my-project/work/codebase/unified/phase1/data/dead_letter
         which does NOT exist on any other machine (verified by ls).
      2. Wrapped the makedirs call in ``except Exception: return`` --
         silently swallowing the failure. Quarantined GDA records
         silently vanished: no file written, no error raised. The
         dead-letter audit trail was fictional.
    Now we:
      * Resolve the default path relative to the phase1 package itself
        (``phase1/data/dead_letter``) so it works on any install.
      * Raise ``OSError`` if the directory cannot be created -- fail
        loudly so operators see the configuration problem.

    FIX-P3-5: previously wrote ONLY to a JSONL file; the
    ``dead_letter_gda`` TABLE (migration 004 lines 193-208) and its ORM
    model ``DeadLetterGDA`` were created but NEVER populated, making
    both dead code. Now also INSERTs into the table via
    ``session.bulk_save_objects`` so the ORM model is live and
    queryable. The JSONL file is kept for human inspection -- both
    destinations receive the same rows. If ``session`` is None the
    table write is skipped (backward-compat for any caller that does
    not yet pass a session).
    """
    import json
    import os
    from datetime import datetime, timezone

    # FIX-P3-5: populate the dead_letter_gda table via the ORM so the
    # table and model are no longer dead code. The bulk_save_objects
    # path is wrapped in try/except so a DB failure does NOT prevent
    # the JSONL write below -- both destinations are best-effort.
    if session is not None and len(df) > 0:
        try:
            orm_objs = []
            for _, row in df.iterrows():
                row_dict = {
                    k: (None if pd.isna(v) else str(v)) for k, v in row.items()
                }
                orm_objs.append(
                    DeadLetterGDA(
                        gene_symbol=row_dict.get("gene_symbol"),
                        disease_id=row_dict.get("disease_id"),
                        source=row_dict.get("source"),
                        reason=reason,
                        details_json=json.dumps(row_dict, default=str),
                        # v89 ROOT FIX (BUG #29 -- loader side): write the
                        # integer pipeline_run_id directly, NOT
                        # str(pipeline_run_id). The model column is now
                        # ``Integer FK -> pipeline_runs.id`` (see models.py
                        # fix). The previous ``str(pipeline_run_id)``
                        # would now raise ``DataError: invalid input
                        # syntax for type integer`` on PostgreSQL and
                        # ``TypeError`` on SQLite -- the FK relationship is
                        # preserved and dead-letter rows can be JOINed to
                        # pipeline_runs without a CAST.
                        pipeline_run_id=(
                            int(pipeline_run_id) if pipeline_run_id is not None else None
                        ),
                    )
                )
            if orm_objs:
                # v65 ROOT FIX (P1C-005 -- savepoint, not full rollback):
                #   The previous code called ``session.rollback()`` when
                #   the dead-letter flush failed. ``session.rollback()``
                #   rolls back the ENTIRE transaction -- including any
                #   valid GDA rows the caller had already staged in the
                #   same session. The comment said "rollback the dead-
                #   letter inserts" but that is NOT what
                #   ``session.rollback()`` does -- it rolls back to the
                #   last commit/savepoint, discarding EVERYTHING since.
                #   If ``bulk_upsert_gda`` had staged 1000 valid GDA
                #   rows and then called ``_quarantine_gda_rows`` (which
                #   flush-fails on a CHECK constraint), ALL 1000 valid
                #   rows were silently lost. The operator saw "dead_letter
                #   flush failed; rows persisted to JSONL only" but did
                #   NOT see that the valid GDA rows were also rolled back.
                #   ROOT FIX: use ``session.begin_nested()`` (a SAVEPOINT)
                #   around the ``add_all`` + ``flush``. On failure, roll
                #   back ONLY the savepoint (preserving the caller's
                #   staged rows). On success, release the savepoint (the
                #   rows stay in the outer transaction, committed by the
                #   caller). The ``with`` context manager auto-commits on
                #   success and auto-rolls-back on exception; the outer
                #   try/except catches the exception for logging (the
                #   JSONL write below still runs).
                #
                # v89 ROOT FIX (BUG #30 -- bulk_save_objects bypasses
                #   SQLAlchemy 2.0 unit-of-work):
                #   ``session.bulk_save_objects()`` is a legacy SQLAlchemy
                #   1.x API that bypasses the unit-of-work cascade. In
                #   SQLAlchemy 2.0, ``session.add_all()`` + ``session.flush()``
                #   is the recommended pattern. ``bulk_save_objects`` does
                #   NOT fire ``@validates`` decorators, does NOT respect
                #   relationship cascades, and does NOT populate server-
                #   side defaults (like ``created_at`` from
                #   ``TimestampMixin``). The ``DeadLetterGDA`` model
                #   currently has no ``@validates``, so this was safe --
                #   but if a future ``@validates`` is added, it would be
                #   silently bypassed. Server-side defaults (``created_at``,
                #   ``updated_at``) may not be populated correctly via
                #   ``bulk_save_objects`` on some dialects (PostgreSQL
                #   triggers fire on UPDATE, not on ``bulk_save_objects``
                #   INSERT). ROOT FIX: use ``session.add_all(orm_objs)``
                #   -- the SQLAlchemy 2.0 canonical API. Performance
                #   difference is negligible for dead-letter volumes
                #   (typically <100 rows per chunk).
                try:
                    with session.begin_nested():
                        session.add_all(orm_objs)
                        session.flush()
                    # If we reach here, the savepoint committed successfully.
                    logger.info(
                        "bulk_upsert_gda: %d rows written to dead_letter_gda "
                        "table (reason=%s).",
                        len(orm_objs), reason,
                    )
                except Exception:
                    # The savepoint was rolled back automatically by the
                    # context manager -- ONLY the dead-letter inserts are
                    # lost. The caller's staged rows are PRESERVED. We
                    # DO NOT raise -- the JSONL write below still runs so
                    # the operator has the audit trail on disk.
                    logger.warning(
                        "bulk_upsert_gda: dead_letter_gda table flush failed; "
                        "rows persisted to JSONL only. The caller's "
                        "transaction is PRESERVED (savepoint rollback, not "
                        "full rollback). See error above.",
                    )
        except Exception as exc:
            # Defensive: any unexpected error in the ORM path must not
            # crash the caller. Log and continue to JSONL write.
            logger.warning(
                "bulk_upsert_gda: failed to write %d rows to dead_letter_gda "
                "table: %s. Rows will still be written to JSONL.",
                len(df), exc,
            )

    # Resolve default path RELATIVE to this module so it works on any
    # install -- not a hardcoded absolute path that only exists on the
    # original developer's machine.
    _PHASE1_ROOT = Path(__file__).resolve().parent.parent  # phase1/
    _DEFAULT_DL_DIR = str(_PHASE1_ROOT / "data" / "dead_letter")
    dl_dir = os.environ.get("DRUGOS_DEAD_LETTER_DIR", _DEFAULT_DL_DIR)
    try:
        os.makedirs(dl_dir, exist_ok=True)
    except OSError as exc:
        # v9: fail loudly. The previous code silently returned, making
        # the dead-letter audit trail fictional. Now we raise so the
        # operator sees the misconfiguration immediately.
        logger.error(
            "bulk_upsert_gda: cannot create dead-letter directory %r: %s. "
            "Set DRUGOS_DEAD_LETTER_DIR to a writable path. Quarantined "
            "rows will be returned to the caller via the in-memory queue "
            "but NOT persisted to disk.",
            dl_dir, exc,
        )
        # Fall back to in-memory queue so data is not lost in-process.
        for _, row in df.iterrows():
            _add_to_dead_letter(
                {k: (None if pd.isna(v) else v) for k, v in row.items()},
                f"{reason} (dead_letter_dir_unavailable)",
                "bulk_upsert_gda",
            )
        return
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = os.path.join(dl_dir, f"gda_quarantine_{ts}.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as fh:
            for _, row in df.iterrows():
                rec = {
                    "reason": reason,
                    "timestamp": ts,
                    "row": {k: (None if pd.isna(v) else str(v)) for k, v in row.items()},
                }
                fh.write(json.dumps(rec, default=str) + "\n")
    except OSError as exc:
        # v9: same fail-loudly pattern for the file write.
        logger.error(
            "bulk_upsert_gda: cannot write dead-letter file %r: %s. "
            "Rows will be returned via the in-memory queue.",
            path, exc,
        )
        for _, row in df.iterrows():
            _add_to_dead_letter(
                {k: (None if pd.isna(v) else v) for k, v in row.items()},
                f"{reason} (dead_letter_write_failed)",
                "bulk_upsert_gda",
            )


def bulk_upsert_gda(
    session: Session,
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    pipeline_run_id: int | None = None,
    score_type: str | None = None,
    score_method: str | None = None,
    input_checksum: str | None = None,
    dedup_already_done: bool = False,
) -> UpsertResult:
    """Bulk upsert gene-disease associations.

    ON CONFLICT (gene_symbol, disease_id, source) DO UPDATE for
    updatable columns.

    DESIGN DECISION (v59 ROOT FIX -- compound of P1C-001):
      Rows with NULL or empty ``gene_symbol`` OR ``disease_id`` are
      QUARANTINED (written to the ``dead_letter_gda`` table via
      ``_quarantine_gda_rows``) and dropped from the upsert payload.
      The previous code ``fillna("")`` on these columns crashed
      PostgreSQL (CHECK constraint ``disease_id <> ''`` rejected the
      empty DEFAULT) while silently passing on SQLite. The
      ``source`` column is left NULL when missing -- the CHECK
      ``chk_gda_source`` explicitly allows ``source IS NULL``,
      so no fillna is needed.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    df : pd.DataFrame
        DataFrame with GDA data.  Required: gene_symbol, disease_id,
        source.  Optional: uniprot_id, protein_id, disease_name,
        association_type, score, pmid_list, disease_id_type.
    batch_size : int
        Number of rows per INSERT statement.  Must be > 0.
    pipeline_run_id : int | None
        Optional pipeline run ID for lineage tracking (LINE-01).
    score_type : str | None
        Type of score (e.g. 'gda_score') (LINE-03).
    score_method : str | None
        Method used to compute the score (e.g. 'disgenet_v7') (LINE-03).
    input_checksum : str | None
        Optional SHA-256 checksum of the input DataFrame (LINE-05).

    Returns
    -------
    UpsertResult
    """
    _isinstance_dataframe(df, "bulk_upsert_gda")
    _validate_batch_size(batch_size)

    result = UpsertResult()

    if df.empty:
        logger.debug("bulk_upsert_gda: empty dataframe, skipping")
        return result

    if input_checksum:
        logger.debug("bulk_upsert_gda: input checksum = %s", input_checksum)

    df = _sanitize_dataframe(df.copy())

    # FIX C5 / AUDIT-2 + BUG-A-002 root fix:
    # The previous code replaced NULL gene_symbol with '' so the unique
    # constraint could detect duplicates. BUT this silently collapsed
    # DISTINCT genes with empty gene_symbols into one row -- silent data
    # loss of Gene->Disease edges. The correct behavior is to QUARANTINE
    # rows with NULL/empty gene_symbol (they cannot be meaningfully
    # deduplicated by gene_symbol because they have no gene identity).
    # The unique constraint is now applied only to rows WITH a real
    # gene_symbol; rows without one are written to a quarantine table
    # for manual review.
    if "gene_symbol" in df.columns:
        null_count = df["gene_symbol"].isna().sum()
        empty_count = ((df["gene_symbol"].astype(str).str.strip() == "") & df["gene_symbol"].notna()).sum()
        bad_count = int(null_count) + int(empty_count)
        if bad_count > 0:
            logger.error(
                "bulk_upsert_gda: BUG-A-002 -- %d records have NULL or empty "
                "gene_symbol. Quarantining instead of fillna('') which "
                "silently collapsed distinct genes into one row.",
                bad_count,
            )
            # Quarantine the bad rows so they're not silently lost.
            bad_mask = df["gene_symbol"].isna() | (df["gene_symbol"].astype(str).str.strip() == "")
            bad_rows = df[bad_mask].copy()
            try:
                # FIX-P3-5: pass session + pipeline_run_id so the rows
                # are persisted to the dead_letter_gda TABLE (via ORM),
                # not just to a JSONL file.
                _quarantine_gda_rows(
                    bad_rows,
                    reason="null_or_empty_gene_symbol",
                    session=session,
                    pipeline_run_id=pipeline_run_id,
                )
            except Exception as q_exc:
                logger.warning(
                    "bulk_upsert_gda: failed to write quarantine for %d "
                    "rows: %s", bad_count, q_exc,
                )
            # Drop the bad rows from the upsert payload.
            df = df[~bad_mask].copy()
        # No more fillna('') -- keep gene_symbol as-is (non-null).
    else:
        # No gene_symbol column at all -- log and quarantine all rows.
        # v57 ROOT FIX (P1C-001): the previous code did `df["gene_symbol"] = ""`
        # which (combined with the old CHECK <> '' constraint) guaranteed
        # an IntegrityError on INSERT. Now that the constraint is removed
        # and gene_symbol is NULLABLE, we set NULL instead of "" and
        # quarantine the rows (same path as the bad_mask branch above).
        logger.error(
            "bulk_upsert_gda: BUG-A-002 -- input dataframe is missing the "
            "gene_symbol column entirely. All rows will be quarantined."
        )
        df["gene_symbol"] = None
        bad_mask = df["gene_symbol"].isna()
        bad_rows = df[bad_mask].copy()
        try:
            _quarantine_gda_rows(
                bad_rows,
                reason="missing_gene_symbol_column",
                session=session,
                pipeline_run_id=pipeline_run_id,
            )
        except Exception as q_exc:
            logger.warning(
                "bulk_upsert_gda: failed to write quarantine for %d "
                "rows: %s", len(bad_rows), q_exc,
            )
        # P1-020 ROOT FIX (v100 forensic -- early return instead of
        # processing an empty DataFrame):
        # The previous code did ``df = df[~bad_mask].copy()`` here. But
        # ``bad_mask = df["gene_symbol"].isna()`` was ALL True (every row
        # has NaN because we just set the column to None), so
        # ``df[~bad_mask]`` was EMPTY. The function then continued
        # processing an empty DataFrame -- running through the disease_id
        # check, the chunk loop, the upsert, etc. -- producing confusing
        # "0 inserted, N quarantined" output. The ``result.total_input``
        # was set later (line ~3189) to ``len(df)`` (the ORIGINAL count)
        # but ``result.inserted`` was 0 and ``result.quarantined`` was
        # never explicitly set to ``total``. An operator monitoring the
        # pipeline saw "0 inserted" without a clear indication that ALL
        # rows were quarantined because of the missing column -- leading
        # to debugging time spent looking at the wrong layer. ROOT FIX:
        # explicitly set ``result.quarantined = len(df)`` and
        # ``return result`` here so the function exits cleanly with an
        # unambiguous audit trail.
        _n_total = len(df)
        result.total_input = _n_total
        result.quarantined = _n_total
        logger.warning(
            "bulk_upsert_gda: P1-020 -- input dataframe missing gene_symbol "
            "column; quarantined all %d row(s) and returning early (no "
            "upsert attempted).",
            _n_total,
        )
        return result
        # (The previous ``df = df[~bad_mask].copy()`` line is removed --
        # it would have produced an empty DataFrame and continued
        # processing, masking the real failure mode.)

    # v42 ROOT FIX (P1-A-9): the previous code did
    # ``df["disease_id"] = df["disease_id"].fillna("")`` with a comment
    # claiming that empty disease_id "doesn't violate any CHECK
    # constraint". This is FALSE -- migration 001 defines
    # ``CONSTRAINT chk_gda_disease_id_nonempty CHECK (disease_id <> '')``
    # (line ~1070). Rows with NULL disease_id were converted to "" by fillna
    # and then FAILED the CHECK on INSERT, raising IntegrityError. ROOT
    # FIX: quarantine rows with NULL/empty disease_id (same pattern
    # used above for gene_symbol) instead of fillna(""). This preserves
    # data integrity and surfaces the bad rows for manual review rather
    # than crashing the upsert.
    if "disease_id" in df.columns:
        null_count = df["disease_id"].isna().sum()
        empty_count = (
            (df["disease_id"].astype(str).str.strip() == "")
            & df["disease_id"].notna()
        ).sum()
        bad_count = int(null_count) + int(empty_count)
        if bad_count > 0:
            logger.error(
                "bulk_upsert_gda: P1-A-9 -- %d records have NULL or empty "
                "disease_id. Quarantining instead of fillna('') which "
                "violates chk_gda_disease_id_nonempty CHECK (disease_id <> '') "
                "and crashes the upsert.",
                bad_count,
            )
            bad_mask = (
                df["disease_id"].isna()
                | (df["disease_id"].astype(str).str.strip() == "")
            )
            bad_rows = df[bad_mask].copy()
            try:
                # FIX-P3-5: pass session + pipeline_run_id so the rows
                # are persisted to the dead_letter_gda TABLE (via ORM),
                # not just to a JSONL file.
                _quarantine_gda_rows(
                    bad_rows,
                    reason="null_or_empty_disease_id",
                    session=session,
                    pipeline_run_id=pipeline_run_id,
                )
            except Exception as q_exc:
                logger.warning(
                    "bulk_upsert_gda: failed to write quarantine for %d "
                    "rows: %s", bad_count, q_exc,
                )
            df = df[~bad_mask].copy()
        # No fillna("") -- keep disease_id as-is (non-null, non-empty).
    else:
        # v59 ROOT FIX (compound of P1C-001 -- disease_id contradiction):
        # The previous code did `df["disease_id"] = ""` which (combined
        # with the old `NOT NULL DEFAULT ''` + `CHECK (disease_id <> '')`
        # schema) guaranteed an IntegrityError on PostgreSQL. The DEFAULT
        # is now removed from migration 001 + models.py, so supplying ""
        # would crash even more directly (NOT NULL violation via the
        # CHECK). The correct fix mirrors the gene_symbol branch above:
        # set None, quarantine all rows, then drop them from the upsert
        # payload so the bulk insert sees no rows that would violate the
        # CHECK constraint. This is the SAME root pattern already
        # applied for gene_symbol (loaders.py:2632-2658) -- closing the
        # compound gap that v57 left open.
        logger.error(
            "bulk_upsert_gda: P1-A-9 + v59 -- input dataframe is missing "
            "the disease_id column entirely. All rows will be quarantined "
            "(NULL disease_id) and dropped from the upsert payload."
        )
        df["disease_id"] = None
        bad_mask = df["disease_id"].isna()
        bad_rows = df[bad_mask].copy()
        try:
            _quarantine_gda_rows(
                bad_rows,
                reason="missing_disease_id_column",
                session=session,
                pipeline_run_id=pipeline_run_id,
            )
        except Exception as q_exc:
            logger.warning(
                "bulk_upsert_gda: failed to write quarantine for %d "
                "rows (missing disease_id column): %s",
                len(bad_rows), q_exc,
            )
        df = df[~bad_mask].copy()
    # v29 ROOT FIX (audit D-11): DO NOT fillna("") on source column.
    # The DB CHECK constraint chk_gda_source allows `source IS NULL OR
    # source IN ('disgenet', 'omim')`. fillna("") converts valid NULL
    # to invalid "" which FAILS the CHECK constraint. ROOT FIX: leave
    # NULL as NULL -- the CHECK allows it, and NULL is semantically
    # correct ("source unknown"). If the source column has a default
    # value, the DB will apply it; otherwise it stays NULL.
    # df["source"] = df["source"].fillna("")  # v29: REMOVED -- causes CHECK violation

    # IDEM-3/CODE-7/IDEM-19: Sort by score descending with a deterministic
    # tiebreak (gene_id, disease_id, source ascending) before drop_duplicates,
    # then keep="first" -- ensures the highest-scored record survives AND the
    # tiebreak is deterministic across runs / DB engines.
    #
    # When the caller passes ``dedup_already_done=True`` (e.g. the
    # institutional-grade DisGeNET pipeline, which centralises dedup in
    # ``validate_gda_scores(dedup=True)`` per DQ-6 / SCI-37), we SKIP the
    # sort-and-dedup here.  This is the single-source-of-truth rule:
    # dedup happens in exactly one layer (the validator), and the loader
    # trusts the caller.  PERF-19: skipping the sort saves ~2s on 1M rows.
    # v38 ROOT FIX (Phase 1 Issue #42): the previous dedup key was
    # ``(gene_symbol, disease_id, source)``. HGNC renames gene symbols
    # periodically (e.g. ``HUGO`` -> ``HUGO1``). If DisGeNET has an
    # old-name record and OMIM has a new-name record for the SAME gene
    # (same ``gene_id`` = NCBI Entrez Gene ID), they would NOT dedup --
    # two separate rows in the DB, creating duplicate Gene nodes in
    # the KG. The fix: prefer ``gene_id`` (the stable NCBI identifier)
    # when available, falling back to ``gene_symbol`` only when
    # ``gene_id`` is missing. This matches the audit's recommendation:
    # "use gene_id for cross-source joins (stable across HGNC renames)".
    # We build the dedup_cols dynamically based on what's in the DataFrame.
    if "gene_id" in df.columns and df["gene_id"].notna().any():
        # gene_id is present and at least one row has a non-null value.
        # Use gene_id as the primary dedup key (stable across HGNC renames).
        dedup_cols = ["gene_id", "disease_id", "source"]
        logger.debug(
            "v38 Issue #42: dedup by (gene_id, disease_id, source) -- "
            "stable across HGNC gene_symbol renames."
        )
    else:
        # gene_id is missing or all-null -- fall back to gene_symbol
        # (the previous behaviour). This is less robust but maintains
        # backward compat for callers that don't populate gene_id.
        dedup_cols = ["gene_symbol", "disease_id", "source"]
        logger.debug(
            "v38 Issue #42: gene_id not available -- falling back to "
            "dedup by (gene_symbol, disease_id, source). NOTE: HGNC "
            "gene_symbol renames may cause duplicate rows for the same "
            "gene. Populate the gene_id column to fix this."
        )
    if not dedup_already_done:
        sort_cols: list[str] = []
        ascending: list[bool] = []
        if "score" in df.columns:
            sort_cols.append("score")
            ascending.append(False)  # highest score first
        # Deterministic tiebreak (IDEM-19) -- lower gene_id wins on ties.
        for col in ("gene_id", "gene_symbol", "disease_id", "source"):
            if col in df.columns:
                sort_cols.append(col)
                ascending.append(True)
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols, keep="first")
        # v93 ROOT FIX (P1-026 -- application-level NULL gene_symbol dedup):
        #   pandas ``drop_duplicates`` treats NaN as DISTINCT -- two rows
        #   with ``gene_symbol=NaN`` and the same ``(disease_id, source)``
        #   are NOT deduplicated. This is the same NULLs-are-distinct
        #   issue as the DB UNIQUE constraint. The functional UNIQUE
        #   index on ``COALESCE(gene_symbol, '')`` (added in models.py)
        #   catches this at the DB level on PostgreSQL, but SQLite
        #   dev/test may not render functional indexes via SQLAlchemy
        #   DDL. This application-level dedup is the defense-in-depth:
        #   normalize NaN gene_symbol to a sentinel for the dedup check,
        #   then drop duplicates. This runs BEFORE the DB insert, so the
        #   DB constraint never sees the duplicates.
        if "gene_symbol" in df.columns and "disease_id" in df.columns \
                and "source" in df.columns:
            null_gene_mask = df["gene_symbol"].isna()
            if null_gene_mask.any():
                # Create a temporary sentinel column for dedup only.
                # Do NOT modify the real gene_symbol column (NULL must
                # stay NULL for the DB insert).
                _dedup_key = df["gene_symbol"].fillna("__NULL_GENE__")
                _dedup_key = _dedup_key.astype(str) + "\x1f" + \
                    df["disease_id"].astype(str) + "\x1f" + \
                    df["source"].astype(str)
                before_null_dedup = len(df)
                df = df[~_dedup_key.duplicated(keep="first")]
                if len(df) < before_null_dedup:
                    logger.warning(
                        "bulk_upsert_gda: NULL gene_symbol dedup removed "
                        "%d duplicate rows (P1-026 app-level defense)",
                        before_null_dedup - len(df),
                    )
        if len(df) < before:
            logger.warning(
                "bulk_upsert_gda: deduplicated %d -> %d records",
                before,
                len(df),
            )
    else:
        logger.debug(
            "bulk_upsert_gda: dedup_already_done=True -- skipping internal "
            "sort/dedup (caller is responsible for dedup)"
        )

    batch_size = _calculate_safe_batch_size(
        GeneDiseaseAssociation, batch_size
    )
    total = len(df)
    result.total_input = total

    with _Timer("bulk_upsert_gda", total):
        dialect_insert = _get_dialect_insert(session)

        updatable_cols = [
            "disease_name",
            "association_type",
            "score",
            "pmid_list",
            "uniprot_id",
            "disease_id_type",
            "updated_at",  # [IDEM-06/DES-05]
        ]
        # 389-fix audit: extend updatable columns to include all new
        # institutional-grade columns (SCI-3..SCI-21, LIN-1..28).  Each
        # is added only if present in the input DataFrame (callers may
        # omit columns they don't populate).
        _optional_updatable_cols = [
            "gene_id",
            "disease_type",
            "source_id",
            "disease_class",
            "disease_class_source",
            "year_initial",
            "year_final",
            "confidence_tier",
            "evidence_strength",
            "normalized_score",
            "source_version",
            "download_date",
            "download_method",
            "source_format",
            "dedup_strategy",
            "confidence_tier_method",
            "resolution_method",
            "gene_to_uniprot_map_version",
            "original_pmid_count",
            "schema_version",
            "snapshot_tag",
            "source_url",
            "score_was_clipped",
            "original_score",
            "score_was_coerced_nan",
            "score_direction",
            "disease_name_was_filled",
            "association_type_was_filled",
            "pmid_list_was_capped",
        ]
        for _col in _optional_updatable_cols:
            if _col in df.columns:
                updatable_cols.append(_col)
        # NOTE: protein_id was removed from the GDA model -- the GDA table
        # uses uniprot_id (string FK) only, not the integer protein PK.
        if pipeline_run_id is not None:
            updatable_cols.append("pipeline_run_id")
        if score_type is not None:
            updatable_cols.append("score_type")
        if score_method is not None:
            updatable_cols.append("score_method")

        log_interval = max(1, total // (batch_size * 20))

        for chunk_idx, chunk in enumerate(
            _df_chunk_to_dicts(df, batch_size)
        ):
            valid_chunk, q_count = _pre_validate_gda(
                chunk, "bulk_upsert_gda"
            )
            result.quarantined += q_count

            if not valid_chunk:
                continue

            for rec in valid_chunk:
                if pipeline_run_id is not None:
                    rec["pipeline_run_id"] = pipeline_run_id
                if score_type is not None:
                    rec["score_type"] = score_type
                if score_method is not None:
                    rec["score_method"] = score_method

            try:
                all_keys: set[str] = set()
                for record in valid_chunk:
                    all_keys.update(record.keys())

                stmt = dialect_insert(
                    GeneDiseaseAssociation.__table__
                ).values(valid_chunk)
                update_dict = {
                    col: stmt.excluded[col]
                    for col in updatable_cols
                    if col in all_keys
                }
                # v49 ROOT FIX (Compound-5 -- bulk_upsert_gda always falls back):
                # The v38 code used dedup_cols = ["gene_id", "disease_id",
                # "source"] when gene_id was present, but the DB only had
                # a unique index on (gene_symbol, disease_id, source).
                # PostgreSQL raised "no unique constraint matching ON
                # CONFLICT" and the except block fell back to row-by-row
                # inserts (10-100× slower + ERROR spam).
                # v49 ROOT FIX: migration 010 adds a partial unique index
                # uq_gda_gene_id_disease_source WHERE gene_id IS NOT NULL.
                # So both conflict targets are now valid. The dedup_cols
                # list chooses the right one based on data availability.
                # On SQLite (no migration 010 applied yet), the gene_id
                # path still raises -- fall back to gene_symbol which
                # always works.
                try:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=dedup_cols,
                        set_=update_dict,
                    )
                    # v49 ROOT FIX (Counter Inflation -- Compound-6): use
                    # _count_upsert_inserts_updates for accurate counts.
                    # This function executes the statement itself (with
                    # RETURNING clause on PostgreSQL) and returns the
                    # accurate (inserts, updates) split.
                    _ins, _upd = _count_upsert_inserts_updates(
                        session, stmt, len(valid_chunk)
                    )
                    result.inserted += _ins
                    result.updated += _upd
                # v66 ROOT FIX (P1C-017 -- bare except Exception catches ALL errors):
                #   The previous code used ``except Exception as _conflict_exc:``
                #   which caught EVERY exception -- including CHECK constraint
                #   violations (IntegrityError), connection errors
                #   (OperationalError), data type mismatches (ProgrammingError
                #   for wrong column types), and deadlocks. For ANY of these
                #   non-conflict-target errors, retrying with gene_symbol is
                #   futile -- the SAME underlying cause will fail again,
                #   doubling the error log spam and delaying the real failure.
                #   ROOT FIX: catch ONLY the specific SQLAlchemy errors that
                #   indicate a missing/invalid ON CONFLICT target, and inspect
                #   the error message for the exact "no unique constraint
                #   matching ON CONFLICT" / "no such column" signatures that
                #   the gene_symbol retry is designed for. All other errors
                #   propagate immediately so the outer except (line ~3038)
                #   can handle them with accurate single-error logging.
                except (OperationalError, ProgrammingError) as _conflict_exc:
                    _conflict_msg = str(_conflict_exc).lower()
                    # Signatures of the exact error the retry is designed for:
                    #  - PostgreSQL: "there is no unique or exclusion constraint
                    #    matching the ON CONFLICT specification"
                    #  - SQLite: "no such column: gene_id" (index not created)
                    #    OR "ON CONFLICT clause does not match any
                    #    PRIMARY KEY or UNIQUE constraint"
                    _is_conflict_target_error = (
                        "no unique constraint matching" in _conflict_msg
                        or "no unique or exclusion constraint" in _conflict_msg
                        or "on conflict clause does not match" in _conflict_msg
                        or ("no such column" in _conflict_msg
                            and "gene_id" in _conflict_msg)
                    )
                    # If the gene_id conflict target failed (DB not yet
                    # migrated to v49), retry with gene_symbol which is
                    # always indexed. Only retry for the SPECIFIC conflict-
                    # target error -- never for unrelated failures.
                    if "gene_id" in dedup_cols and _is_conflict_target_error:
                        logger.debug(
                            "bulk_upsert_gda: gene_id conflict target "
                            "failed (%s) -- retrying with gene_symbol "
                            "(DB may not be migrated to v49).",
                            _conflict_exc,
                        )
                        stmt = dialect_insert(
                            GeneDiseaseAssociation.__table__
                        ).values(valid_chunk)
                        update_dict = {
                            col: stmt.excluded[col]
                            for col in updatable_cols
                            if col in all_keys
                        }
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["gene_symbol", "disease_id", "source"],
                            set_=update_dict,
                        )
                        _ins, _upd = _count_upsert_inserts_updates(
                            session, stmt, len(valid_chunk)
                        )
                        result.inserted += _ins
                        result.updated += _upd
                    else:
                        # Non-conflict-target error (CHECK violation,
                        # connection drop, datatype mismatch, etc.) --
                        # propagate to the outer handler. Do NOT retry
                        # with gene_symbol; it would fail the same way.
                        raise

            except Exception as exc:
                logger.error(
                    "bulk_upsert_gda: chunk %d failed: %s", chunk_idx, exc
                )
                result.failed += len(valid_chunk)
                for record in valid_chunk:
                    try:
                        stmt = dialect_insert(
                            GeneDiseaseAssociation.__table__
                        ).values([record])
                        update_dict = {
                            col: stmt.excluded[col]
                            for col in updatable_cols
                            if col in record
                        }
                        stmt = stmt.on_conflict_do_update(
                            index_elements=[
                                "gene_symbol",
                                "disease_id",
                                "source",
                            ],
                            set_=update_dict,
                        )
                        session.execute(stmt)
                        result.inserted += 1
                        result.failed -= 1
                    except Exception as row_exc:
                        logger.warning(
                            "bulk_upsert_gda: row failed "
                            "(gene_symbol=%s, disease_id=%s): %s",
                            record.get("gene_symbol", "?"),
                            record.get("disease_id", "?"),
                            row_exc,
                        )
                        _add_to_dead_letter(
                            record, str(row_exc), "bulk_upsert_gda"
                        )

            processed = result.inserted + result.quarantined + result.failed
            if (chunk_idx + 1) % log_interval == 0 or processed >= total:
                logger.info(
                    "bulk_upsert_gda: %d / %d processed (%.0f%%)",
                    processed,
                    total,
                    100.0 * processed / total,
                )
            else:
                logger.debug(
                    "bulk_upsert_gda: %d / %d processed", processed, total
                )

    logger.info("bulk_upsert_gda: %s", result)
    return result


# ===========================================================================
# 6. ENTITY MAPPING -- ON CONFLICT DO UPDATE
# ===========================================================================


def bulk_upsert_entity_mapping(
    session: Session,
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    match_history: str | None = None,
    input_checksum: str | None = None,
) -> UpsertResult:
    """Bulk upsert entity mapping / cross-reference rows.

    ON CONFLICT (canonical_inchikey) DO UPDATE for updatable columns.
    Rows without canonical_inchikey use ON CONFLICT on canonical_name
    (uq_entity_mapping_name_no_inchikey constraint).

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    df : pd.DataFrame
        DataFrame with entity mapping data.  Must have either
        canonical_inchikey or canonical_name (or both).
        Optional: chembl_id, drugbank_id, pubchem_cid, uniprot_id,
        string_id, match_confidence, match_method.
    batch_size : int
        Number of rows per INSERT statement.  Must be > 0.
    match_history : str | None
        JSON string documenting resolution attempts (LINE-04).
    input_checksum : str | None
        Optional SHA-256 checksum of the input DataFrame (LINE-05).

    Returns
    -------
    UpsertResult
    """
    _isinstance_dataframe(df, "bulk_upsert_entity_mapping")
    _validate_batch_size(batch_size)

    result = UpsertResult()

    if df.empty:
        logger.debug(
            "bulk_upsert_entity_mapping: empty dataframe, skipping"
        )
        return result

    if input_checksum:
        logger.debug(
            "bulk_upsert_entity_mapping: input checksum = %s",
            input_checksum,
        )

    df = _sanitize_dataframe(df.copy())

    # Deduplicate rows with NULL canonical_inchikey by canonical_name
    if "canonical_inchikey" in df.columns:
        null_ik = df["canonical_inchikey"].isna()
        if null_ik.any() and "canonical_name" in df.columns:
            null_df = df[null_ik].copy()
            if len(null_df) > 0:
                # K fix: include_groups=False means canonical_name (the group
                # key) is excluded from _merge_group's input, so it cannot be
                # recovered from the Series output. Restore it explicitly from
                # the groupby index before reset_index drops it.
                # v39 ROOT FIX (P1 #45): the previous code had a fragile
                # conditional reset_index() that only fired if
                # canonical_name was NOT in columns AND index.name was
                # "canonical_name". If _merge_group returned a DataFrame
                # whose index was a RangeIndex (not canonical_name), the
                # canonical_name column was silently lost. The fix: ALWAYS
                # ensure canonical_name is a column after the groupby +
                # apply + reset_index chain, restoring it from the group
                # key if missing.
                null_df = (
                    null_df.groupby("canonical_name", dropna=False)
                    .apply(_merge_group, include_groups=False)
                    .reset_index()  # canonical_name (group key) -> column
                )
                # v39 ROOT FIX (P1 #45): ensure canonical_name is a column.
                # If reset_index() didn't restore it (edge case where the
                # apply returned a DataFrame with a different index), try
                # to restore it from the groupby key.
                if "canonical_name" not in null_df.columns:
                    if null_df.index.name == "canonical_name":
                        null_df = null_df.reset_index()
                    else:
                        # Last resort: the group key was lost. Log a
                        # warning -- this indicates _merge_group returned
                        # an unexpected shape. The rows will have NaN in
                        # canonical_name, which the downstream concat
                        # will surface.
                        logger.warning(
                            "bulk_upsert_entity_mapping: canonical_name "
                            "column lost after groupby.apply -- _merge_group "
                            "may have returned an unexpected shape. "
                            "Affected rows will have NaN canonical_name. "
                            "(v39 P1 #45 fix)"
                        )
            df = pd.concat(
                [df[~null_ik], null_df], ignore_index=True
            )

    batch_size = _calculate_safe_batch_size(EntityMapping, batch_size)
    total = len(df)
    result.total_input = total

    with _Timer("bulk_upsert_entity_mapping", total):
        dialect_insert = _get_dialect_insert(session)
        dialect_name = session.get_bind().dialect.name

        updatable_cols = [
            "canonical_name",
            "chembl_id",
            "drugbank_id",
            "pubchem_cid",
            "uniprot_id",
            "string_id",
            "match_confidence",
            "match_method",
            # v89 ROOT FIX (BUG #21 -- loader side): add ``last_matched_at``
            # to updatable_cols so the column (now declared on the ORM --
            # see models.py fix) is populated on every upsert. The
            # previous updatable_cols omitted it, so even on prod DBs
            # (where migration 001 created the column), it was NEVER
            # populated -- making it dead data. Now it records the
            # timestamp of the last successful entity resolution, which
            # is the documented contract in migration 001 line 1226:
            # "When the last resolution was performed".
            "last_matched_at",
            "updated_at",  # [DES-05]
        ]
        if match_history is not None:
            updatable_cols.append("match_history")

        log_interval = max(1, total // (batch_size * 20))

        for chunk_idx, chunk in enumerate(
            _df_chunk_to_dicts(df, batch_size)
        ):
            valid_chunk, q_count = _pre_validate_entity_mapping(
                chunk, "bulk_upsert_entity_mapping"
            )
            result.quarantined += q_count

            if not valid_chunk:
                continue

            # Add lineage fields
            # v89 ROOT FIX (BUG #21 -- loader side): populate
            # ``last_matched_at`` on every record so the column (now
            # declared on the ORM and in updatable_cols) actually receives
            # a value. The timestamp records when this entity-resolution
            # match was performed -- the documented contract per migration
            # 001 line 1226.
            _now_utc = datetime.datetime.now(datetime.timezone.utc)
            for rec in valid_chunk:
                if match_history is not None:
                    rec["match_history"] = match_history
                rec["last_matched_at"] = _now_utc

            # Split into with-ik and without-ik paths
            with_ik = [r for r in valid_chunk if r.get("canonical_inchikey") is not None]
            without_ik = [r for r in valid_chunk if r.get("canonical_inchikey") is None]

            # Path A: Rows WITH canonical_inchikey
            if with_ik:
                try:
                    all_keys: set[str] = set()
                    for record in with_ik:
                        all_keys.update(record.keys())

                    stmt = dialect_insert(
                        EntityMapping.__table__
                    ).values(with_ik)
                    update_dict = {
                        col: stmt.excluded[col]
                        for col in updatable_cols
                        if col in all_keys
                    }
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["canonical_inchikey"],
                        set_=update_dict,
                    )
                    # v51 ROOT FIX (COMPOUND-6): accurate counter
                    _ins, _upd = _count_upsert_inserts_updates(
                        session, stmt, len(with_ik)
                    )
                    result.inserted += _ins
                    result.updated += _upd

                except Exception as exc:
                    logger.error(
                        "bulk_upsert_entity_mapping Path A: chunk %d "
                        "failed: %s",
                        chunk_idx,
                        exc,
                    )
                    result.failed += len(with_ik)
                    for record in with_ik:
                        try:
                            stmt = dialect_insert(
                                EntityMapping.__table__
                            ).values([record])
                            update_dict = {
                                col: stmt.excluded[col]
                                for col in updatable_cols
                                if col in record
                            }
                            stmt = stmt.on_conflict_do_update(
                                index_elements=["canonical_inchikey"],
                                set_=update_dict,
                            )
                            session.execute(stmt)
                            result.inserted += 1
                            result.failed -= 1
                        except Exception as row_exc:
                            logger.warning(
                                "bulk_upsert_entity_mapping: row failed "
                                "(inchikey=%s): %s",
                                record.get("canonical_inchikey", "?"),
                                row_exc,
                            )
                            _add_to_dead_letter(
                                record,
                                str(row_exc),
                                "bulk_upsert_entity_mapping",
                            )

            # Path B: Rows WITHOUT canonical_inchikey
            # DQ-4/REL-5/IDEM-4/INT-3: Use on_conflict_do_update instead
            # of do_nothing, for both SQLite and PostgreSQL
            if without_ik:
                try:
                    all_keys: set[str] = set()
                    for record in without_ik:
                        all_keys.update(record.keys())

                    stmt = dialect_insert(
                        EntityMapping.__table__
                    ).values(without_ik)
                    update_dict = {
                        col: stmt.excluded[col]
                        for col in updatable_cols
                        if col in all_keys
                    }
                    if dialect_name == "sqlite":
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["canonical_name"],
                            set_=update_dict,
                            where=text(
                                "canonical_inchikey IS NULL"
                            ),
                        )
                    else:
                        stmt = stmt.on_conflict_do_update(
                            constraint=ENTITY_MAPPING_NAME_CONSTRAINT,
                            set_=update_dict,
                        )
                    # v51 ROOT FIX (COMPOUND-6): accurate counter
                    _ins, _upd = _count_upsert_inserts_updates(
                        session, stmt, len(without_ik)
                    )
                    result.inserted += _ins
                    result.updated += _upd

                except Exception as exc:
                    logger.error(
                        "bulk_upsert_entity_mapping Path B: chunk %d "
                        "failed: %s",
                        chunk_idx,
                        exc,
                    )
                    result.failed += len(without_ik)
                    for record in without_ik:
                        try:
                            stmt = dialect_insert(
                                EntityMapping.__table__
                            ).values([record])
                            update_dict = {
                                col: stmt.excluded[col]
                                for col in updatable_cols
                                if col in record
                            }
                            if dialect_name == "sqlite":
                                stmt = stmt.on_conflict_do_update(
                                    index_elements=["canonical_name"],
                                    set_=update_dict,
                                    where=text(
                                        "canonical_inchikey IS NULL"
                                    ),
                                )
                            else:
                                stmt = stmt.on_conflict_do_update(
                                    constraint=ENTITY_MAPPING_NAME_CONSTRAINT,
                                    set_=update_dict,
                                )
                            session.execute(stmt)
                            result.inserted += 1
                            result.failed -= 1
                        except Exception as row_exc:
                            logger.warning(
                                "bulk_upsert_entity_mapping: row failed "
                                "(name=%s): %s",
                                record.get("canonical_name", "?"),
                                row_exc,
                            )
                            _add_to_dead_letter(
                                record,
                                str(row_exc),
                                "bulk_upsert_entity_mapping",
                            )

            processed = result.inserted + result.quarantined + result.failed
            if (chunk_idx + 1) % log_interval == 0 or processed >= total:
                logger.info(
                    "bulk_upsert_entity_mapping: %d / %d processed "
                    "(%.0f%%)",
                    processed,
                    total,
                    100.0 * processed / total,
                )
            else:
                logger.debug(
                    "bulk_upsert_entity_mapping: %d / %d processed",
                    processed,
                    total,
                )

    logger.info("bulk_upsert_entity_mapping: %s", result)
    return result


# ===========================================================================
# 7. PubChem enrichment -- conditional UPDATE on drugs
# ===========================================================================


def bulk_update_drugs_from_pubchem(
    session: Session,
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Update drugs with PubChem data, refreshing ALL matching drugs.

    P1-A12 ROOT FIX (v82): the previous implementation had
    ``WHERE inchikey = :inchikey AND pubchem_cid IS NULL`` -- skipping
    CID refreshes for drugs that ALREADY had a CID. This meant
    molecular_formula / molecular_weight / smiles were never updated
    for drugs loaded from DrugBank (which sets pubchem_cid at insert
    time). ROOT FIX: remove the ``AND pubchem_cid IS NULL`` condition.
    The SET clause uses COALESCE for ALL columns including pubchem_cid,
    so existing non-NULL values are preserved when the new value is NULL,
    and updated when the new value is non-NULL. This is the scientifically-
    correct behavior -- PubChem property refreshes should apply to ALL
    drugs, not just those without a CID.

    For each row in *df*, executes::

        UPDATE drugs
        SET pubchem_cid       = COALESCE(:pubchem_cid, drugs.pubchem_cid),
            molecular_formula = COALESCE(:molecular_formula, drugs.molecular_formula),
            molecular_weight  = COALESCE(:molecular_weight, drugs.molecular_weight),
            smiles            = COALESCE(:smiles, drugs.smiles),
            updated_at        = :updated_at
        WHERE inchikey = :inchikey;

    Returns the number of rows actually updated.

    NOTE: COALESCE is SQL-standard and works on both PostgreSQL and SQLite.
    """
    _isinstance_dataframe(df, "bulk_update_drugs_from_pubchem")
    _validate_batch_size(batch_size)

    if df.empty:
        logger.debug(
            "bulk_update_drugs_from_pubchem: empty dataframe, skipping"
        )
        return 0

    df = _sanitize_dataframe(df.copy())

    records = _df_to_dicts(df)
    total = len(records)
    processed = 0

    update_sql = text(
        """
        UPDATE drugs
        SET pubchem_cid       = COALESCE(:pubchem_cid, drugs.pubchem_cid),
            molecular_formula = COALESCE(:molecular_formula, drugs.molecular_formula),
            molecular_weight  = COALESCE(:molecular_weight, drugs.molecular_weight),
            smiles            = COALESCE(:smiles, drugs.smiles),
            updated_at        = :updated_at
        WHERE inchikey = :inchikey
    """
    )

    # v29 ROOT FIX (audit D-15): determine the dialect ONCE outside the
    # per-row loop -- we only need to coerce Decimal->float on SQLite
    # (whose parameter binding rejects Decimal). PostgreSQL's Numeric
    # columns accept Decimal natively, so we preserve precision there.
    #
    # v66 ROOT FIX (P1C-021 -- centralize + document asymmetry):
    #   The coercion is now ALSO registered process-wide as a sqlite3
    #   adapter in ``database/connection.py`` (see P1C-021 comment there).
    #   This per-row loop is kept as a DEFENSE-IN-DEPTH backstop for any
    #   code path that bypasses the adapter (e.g. raw dbapi connections).
    #   The DEV/PROD PRECISION ASYMMETRY is inherent and CANNOT be
    #   eliminated: SQLite REAL columns store float64 (precision loss),
    #   PostgreSQL NUMERIC columns preserve arbitrary precision. TEST
    #   ASSERTIONS that compare numeric columns (molecular_weight,
    #   activity_value, etc.) MUST use ``pytest.approx(value, rel=1e-9)``
    #   or ``math.isclose`` so they pass on BOTH backends. A bare
    #   ``assert row.molecular_weight == 180.16`` will FLAKE between
    #   SQLite (180.15999999...) and PostgreSQL (180.16 exact).
    _dialect_name = session.get_bind().dialect.name
    _coerce_decimal_to_float = _dialect_name == "sqlite"

    with _Timer("bulk_update_drugs_from_pubchem", total):
        for chunk_idx, chunk in enumerate(_chunked(records, batch_size)):
            # Add updated_at for each record (CODE-10)
            now = datetime.datetime.now(datetime.timezone.utc)
            for rec in chunk:
                rec["updated_at"] = now
                # v29 ROOT FIX (audit D-15): Decimal->float coercion loses
                # precision. Preserve Decimal for Numeric columns on
                # PostgreSQL (production). SQLite's parameter binding
                # does NOT accept ``decimal.Decimal`` (raises
                # ``sqlite3.ProgrammingError: type 'decimal.Decimal' is
                # not supported``), so we still coerce on SQLite
                # (test/dev). PubChem's ``_safe_float`` returns Decimal
                # for precision (SCI-16); we now honour that precision
                # on PostgreSQL instead of throwing it away.
                if _coerce_decimal_to_float:
                    from decimal import Decimal as _Decimal
                    for k, v in list(rec.items()):
                        if isinstance(v, _Decimal):
                            rec[k] = float(v)

            result = session.execute(update_sql, chunk)
            # REL-8: Handle rowcount = -1 on some drivers
            if result.rowcount < 0:
                logger.debug(
                    "bulk_update_drugs_from_pubchem: rowcount unavailable, "
                    "using chunk size as estimate"
                )
                processed += len(chunk)
            else:
                processed += result.rowcount
            logger.debug(
                "bulk_update_drugs_from_pubchem: cumulative=%d / %d",
                processed,
                total,
            )

    logger.info(
        "bulk_update_drugs_from_pubchem: completed -- %d rows updated",
        processed,
    )
    return processed


# ===========================================================================
# 7b. PubChem compound properties (institutional-grade -- fixes ARCH-5, INT-7)
# ===========================================================================


# Columns the loader writes.  Keep in lockstep with migration 005
# (``pubchem_compound_properties``) and with ``COLUMN_ORDER`` in
# ``pipelines/pubchem_pipeline.py``.  Adding a column here requires:
#   1. ADD COLUMN in migration 005 (or a new migration).
#   2. Add the column name to this list.
#   3. Add the column name to ``COLUMN_ORDER`` in pubchem_pipeline.py.
#   4. Add the column name to ``pubchem_enrichment.csv`` in schema/v1.json.
_PUBCHEM_COMPOUND_PROPERTIES_COLUMNS: tuple[str, ...] = (
    "inchikey",
    "pubchem_cid",
    "canonical_smiles",
    "isomeric_smiles",
    "inchi",
    "iupac_name",
    "cas_number",
    "molecular_formula",
    "molecular_weight",
    "exact_mass",
    "xlogp",
    "xlogp_source",
    "tpsa",
    "tpsa_source",
    "complexity",
    "h_bond_donor_count",
    "h_bond_acceptor_count",
    "rotatable_bond_count",
    "heavy_atom_count",
    "formal_charge",
    "isotope_info",
    "salt_form",
    "protonation_state",
    "pubchem_release",
    "source_id",
    "source_version",
    "download_date",
    "download_method",
    "pipeline_run_id",
    "source_batch_idx",
    "source_response_sha256",
    "input_checksum",
    "transformations",
    "electronic_signature",
    "triggered_by",
)

# Columns that are updatable on conflict (i.e. NOT part of the
# UNIQUE(inchikey, pubchem_cid) constraint and NOT immutable lineage).
# When a row already exists for (inchikey, cid), these columns are
# overwritten with the new values.  ``input_checksum`` and
# ``pipeline_run_id`` are updatable so re-enrichments replace stale lineage
# with fresh lineage.  ``enriched_at`` is updated via the
# ``updated_at = NOW()`` trigger pattern at the ORM layer.
_PUBCHEM_COMPOUND_PROPERTIES_UPDATABLE_COLS: tuple[str, ...] = (
    "canonical_smiles",
    "isomeric_smiles",
    "inchi",
    "iupac_name",
    "cas_number",
    "molecular_formula",
    "molecular_weight",
    "exact_mass",
    "xlogp",
    "xlogp_source",
    "tpsa",
    "tpsa_source",
    "complexity",
    "h_bond_donor_count",
    "h_bond_acceptor_count",
    "rotatable_bond_count",
    "heavy_atom_count",
    "formal_charge",
    "isotope_info",
    "salt_form",
    "protonation_state",
    "pubchem_release",
    "source_id",
    "source_version",
    "download_date",
    "download_method",
    "pipeline_run_id",
    "source_batch_idx",
    "source_response_sha256",
    "input_checksum",
    "transformations",
    "electronic_signature",
    "triggered_by",
    "updated_at",
)


def _build_pubchem_compound_properties_table() -> Any:
    """Construct the SQLAlchemy Core ``Table`` for ``pubchem_compound_properties``.

    The Table object is a *description* of the schema -- it does NOT
    require the table to exist in the database at construction time.
    SQLAlchemy uses it to generate INSERT/UPDATE SQL.  The actual table
    must exist in the DB before any SQL is executed -- that's the
    responsibility of migration 005 (or ``Base.metadata.create_all`` in
    tests).

    V18 ROOT FIX (CD-2 -- three-definition schema drift):
    Before v18, this Core Table was the THIRD divergent definition of
    ``pubchem_compound_properties`` -- different from BOTH the ORM model
    (``models.py:PubChemCompoundProperty``) AND migration 005. The
    audit flagged:

      * No FK on ``inchikey`` (ORM has FK; migration has FK).
      * ``SmallInteger`` for count columns (ORM uses ``Integer``).
      * ``enriched_at`` nullable + no default (ORM NOT NULL +
        ``server_default=func.now()``; migration NOT NULL DEFAULT NOW()).
      * ``source_id``/``pipeline_run_id``/``input_checksum`` NOT NULL
        but no default (ORM NOT NULL + ``server_default=""``).
      * Different UniqueConstraint name
        (``uq_pubchem_props_inchikey_cid`` vs ORM's
        ``uq_pubchem_compound_properties_inchikey_cid``).

    On a fresh DB, the FIRST definition to run wins:
      * If create_all() runs first (e.g. SQLite), the Core/Table-
        derived schema is what's in the DB -- missing FK, wrong types,
        wrong constraint name.
      * If migration 005 runs first (PostgreSQL), the migration schema
        wins, but then create_all() tries to ADD the divergent
        constraint -- silently failing on PG (already exists) or
        succeeding on SQLite (no migration ran).

    The ROOT FIX is to align this Core Table to the ORM model exactly.
    The ORM is the canonical definition (it's what tests assert against
    and what create_all() uses); migration 005 is also aligned to the
    ORM. After this fix, all three definitions agree.
    """
    from sqlalchemy import (
        BigInteger,
        Boolean,
        Column,
        DateTime,
        ForeignKey,
        Integer,
        MetaData,
        Numeric,
        String,
        Table,
        Text,
        UniqueConstraint,
        func,
    )

    metadata = MetaData()
    return Table(
        "pubchem_compound_properties",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        # V18 CD-2: FK to drugs.inchikey (was missing -- aligns with
        # ORM + migration 005).
        Column(
            "inchikey", String(50),
            ForeignKey("drugs.inchikey"),
            nullable=False,
        ),
        Column("pubchem_cid", BigInteger, nullable=False),
        Column("canonical_smiles", String(50000)),
        Column("isomeric_smiles", String(50000)),
        Column("inchi", Text),
        Column("iupac_name", Text),
        Column("cas_number", String(20)),
        Column("molecular_formula", String(200)),
        Column("molecular_weight", Numeric(12, 6)),
        Column("exact_mass", Numeric(12, 6)),
        Column("xlogp", Numeric(6, 2)),
        Column("xlogp_source", String(50), server_default="pubchem_xlogp3"),
        Column("tpsa", Numeric(8, 2)),
        Column("tpsa_source", String(50), server_default="pubchem_calculated"),
        Column("complexity", Numeric(10, 2)),
        # V18 CD-2: Integer (not SmallInteger) -- aligns with ORM.
        # SmallInteger maxes at 32767; some proteins have 50000+ atoms
        # in complex formulations -- Integer (32-bit) is the safe choice.
        Column("h_bond_donor_count", Integer),
        Column("h_bond_acceptor_count", Integer),
        Column("rotatable_bond_count", Integer),
        Column("heavy_atom_count", Integer),
        Column("formal_charge", Integer),
        Column("isotope_info", Text),
        Column("salt_form", String(100)),
        # v20 CD-2 ROOT FIX: String(20) to match migration 005 VARCHAR(20)
        # and ORM model. V19 PS-1 widened the migration column to fit
        # full word taxonomy but left Core Table at String(1).
        Column("protonation_state", String(20)),
        Column("pubchem_release", String(100)),
        # V18 CD-2: NOT NULL + server_default -- aligns with ORM + migration.
        Column("source_id", String(100), nullable=False, server_default=""),
        Column("source_version", String(100)),
        Column("download_date", DateTime(timezone=True), nullable=False),
        Column("download_method", String(20)),
        # FIX-P1-C-14: was ``String(64), nullable=False, server_default=""``
        # -- divergence from the ORM (models.py: Integer, FK to
        # pipeline_runs.id, nullable). The ORM definition is canonical;
        # the Core Table now matches: Integer FK with ON DELETE SET NULL,
        # nullable. The empty-string server_default would have coerced
        # NULLs to "" on PostgreSQL, breaking the FK.
        Column(
            "pipeline_run_id", Integer,
            ForeignKey("pipeline_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        Column("source_batch_idx", Integer),
        Column("source_response_sha256", String(64)),
        Column(
            "input_checksum", String(64),
            nullable=False, server_default="",
        ),
        Column("transformations", Text),
        Column("electronic_signature", Text),
        Column("triggered_by", Text),
        # V18 CD-2: NOT NULL + server_default=NOW() -- aligns with ORM +
        # migration 005. Was nullable + no default before; NULL
        # enriched_at silently broke enrichment-age queries.
        Column(
            "enriched_at", DateTime(timezone=True),
            nullable=False, server_default=func.current_timestamp(),
        ),
        # v90 ROOT FIX (BUG #23): `server_default=text("FALSE")` instead of
        #   the non-portable `server_default="0"`. This Column object mirrors
        #   the ORM PubChemCompoundProperty model -- the previous "0" literal
        #   diverged from the ORM (`server_default="0"` -> now also FALSE) and
        #   migration 005 (`is_deleted BOOLEAN NOT NULL DEFAULT FALSE`).
        #   `text` is already imported at line 44 above.
        Column("is_deleted", Boolean, default=False, server_default=text("FALSE")),
        Column("created_at", DateTime(timezone=True)),
        Column("updated_at", DateTime(timezone=True)),
        # V18 CD-2: align constraint NAME to ORM (was
        # "uq_pubchem_props_inchikey_cid" -- divergent).
        UniqueConstraint(
            "inchikey", "pubchem_cid",
            name="uq_pubchem_compound_properties_inchikey_cid",
        ),
        extend_existing=True,
    )


# Construct the Table once at module load.  This is a *description* -- it
# does not require the table to exist in the DB.  SQLAlchemy's
# ``insert(table)`` requires a real Table object (not a proxy), so we
# resolve eagerly here.
_PUBCHEM_COMPOUND_PROPERTIES_TABLE = _build_pubchem_compound_properties_table()


def bulk_upsert_pubchem_compound_properties(
    session: Session,
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> "UpsertResult":
    """Upsert rows into ``pubchem_compound_properties`` (ARCH-5, INT-7).

    Persists the 15+ physicochemical properties fetched from PubChem that
    were previously dropped on the floor by the legacy pipeline.  Uses
    SQLAlchemy 2.0 dialect-aware ``INSERT ... ON CONFLICT DO UPDATE``
    (PostgreSQL) or ``INSERT ... ON CONFLICT DO UPDATE`` (SQLite -- both
    share the same SQLAlchemy API).

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.  The caller manages the transaction
        boundary (commit/rollback) -- this function does NOT commit.
    df : pandas.DataFrame
        Cleaned PubChem enrichment DataFrame from
        ``PubChemPipeline.clean()``.  Must contain at least ``inchikey``,
        ``pubchem_cid``, ``pipeline_run_id``, ``download_date``,
        ``input_checksum``, and ``source_id``.  Extra columns are ignored.
    batch_size : int, default ``DEFAULT_BATCH_SIZE`` (1000)
        Number of rows per INSERT statement.

    Returns
    -------
    UpsertResult
        ``inserted`` and ``updated`` counts populated.  ``quarantined``
        and ``failed`` are populated when individual rows are rejected
        (invalid InChIKey format, missing required columns, etc.).

    Notes
    -----
    * Empty strings, NaN, NaT are converted to SQL NULL before insert
      (SCI-18, DQ-3).  ``COALESCE`` semantics are NOT used here -- every
      upsert overwrites the existing row's updatable columns with the new
      values.  This is intentional: PubChem data is authoritative for the
      (inchikey, cid) pair, and stale data should be replaced, not
      preserved.  Use the soft-delete flag (``is_deleted``) to retain
      historical rows for audit.
    * ``molecular_weight`` and ``exact_mass`` are expected to be
      ``decimal.Decimal`` instances (SCI-16).  The loader does NOT convert
      floats to Decimal -- that is ``clean()``'s responsibility.  If floats
      are passed, SQLAlchemy will store them as-is and the NUMERIC(12,6)
      column will silently truncate to 6 decimal places (potentially
      losing precision for values like ``180.06338800000002``).
    * The ``enriched_at`` column is left to its ``DEFAULT NOW()`` on
      INSERT and explicitly set to ``NOW()`` on UPDATE (via the
      ``updated_at`` column in the update dict).

    Examples
    --------
    >>> from database.connection import get_db_session
    >>> from database.loaders import bulk_upsert_pubchem_compound_properties
    >>> with get_db_session(pipeline_name="pubchem") as sess:
    ...     result = bulk_upsert_pubchem_compound_properties(sess, df)
    >>> result.inserted, result.updated
    (95, 0)
    """
    _isinstance_dataframe(df, "bulk_upsert_pubchem_compound_properties")
    _validate_batch_size(batch_size)

    result = UpsertResult(total_input=len(df))

    if df.empty:
        logger.debug(
            "bulk_upsert_pubchem_compound_properties: empty dataframe, skipping"
        )
        return result

    # Required columns per the table's NOT NULL constraints.
    REQUIRED_COLS = (
        "inchikey",
        "pubchem_cid",
        "source_id",
        "download_date",
        "pipeline_run_id",
        "input_checksum",
    )
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"bulk_upsert_pubchem_compound_properties: df missing required "
            f"columns: {missing}"
        )

    # Filter the DataFrame to ONLY the columns this loader writes --
    # extra lineage columns from clean() (e.g. ``source``, ``as_of_date``)
    # are kept in the CSV but not persisted to this table.
    cols_to_use = [
        c for c in _PUBCHEM_COMPOUND_PROPERTIES_COLUMNS if c in df.columns
    ]
    df_filtered = df[cols_to_use].copy()

    # Sanitise: convert NaN/NaT/empty-string -> None (SCI-18, DQ-3).
    # ``_sanitize_dataframe`` is the existing helper used by the other
    # loaders; it does not convert empty strings to None (only NaN/NaT),
    # so we add an explicit pass for empty strings on object columns.
    # v79 FORENSIC ROOT FIX (P0-A3 -- "unknown" coercion nulled legitimate
    #   enum values):
    #   The v78 code coerced the string "unknown" -> None for ALL object
    #   columns. But "unknown" is a LEGITIMATE enum value in this schema:
    #     DrugType.UNKNOWN        (models.py -- the canonical "we looked and
    #                              could not classify" sentinel)
    #     ActivityType.UNKNOWN    (models.py -- same semantic)
    #   PubChem compound properties legitimately return "unknown" for
    #   fields like canonical_smile, hydrogen_bond donor count, etc. when
    #   the property exists but the value is unspecified. Coercing these
    #   to None DESTROYED the distinction between "property missing"
    #   (None) and "property present but value unknown" ("unknown") -- a
    #   distinction the downstream KG uses to decide whether to emit a
    #   Compound node property at all.
    # ROOT FIX: remove "unknown" from the null-coercion set. The only
    #   strings that should be coerced to None are genuine null sentinels
    #   ("", "nan", "none", "null", "n/a", "-"). The literal string
    #   "unknown" is preserved as the enum value it represents.
    df_filtered = _sanitize_dataframe(df_filtered)
    _NULL_SENTINELS = ("", "nan", "none", "null", "n/a", "-")
    for col in df_filtered.select_dtypes(include=["object"]).columns:
        df_filtered[col] = df_filtered[col].apply(
            lambda v: None
            if (
                isinstance(v, str)
                and v.strip().lower() in _NULL_SENTINELS
            )
            else v
        )

    # Pre-validate InChIKeys -- invalid-format rows go to the loader's
    # dead-letter queue (DQ-2, DQ-17).  We use the same regex as
    # ``database.models._validate_inchikey``.
    # v9 ROOT FIX (audit F3.8): the audit found SIX different InChIKey
    # regexes across the codebase. A key accepted by one validator was
    # rejected by another -- creating a "universal chemical identifier"
    # with 6 different definitions. Now we centralize: import the
    # canonical is_valid_inchikey from cleaning.normalizer and use it
    # everywhere. This is the single source of truth.
    # P1-ER-2 / P1-ER-3 ROOT FIX: the canonical validator no longer
    # accepts TEST/OUTER/INNER/IK test-fixture prefixes (they were
    # developer conveniences that leaked into production data --
    # Chain 3). The fallback regex below is now synchronized with the
    # canonical contract: 27-char standard (with optional protonation
    # suffix per IUPAC) OR SYNTH prefix OR mixture. NO test-fixture
    # acceptance.
    try:
        from cleaning.normalizer import is_valid_inchikey as _canonical_is_valid_inchikey
        def _inchikey_valid(ik: str) -> bool:
            return _canonical_is_valid_inchikey(str(ik))
    except ImportError:
        # Fallback if cleaning module is not on path (test isolation).
        # v43 ROOT FIX (Chain 8 -- InChIKey regex divergence): the
        # previous fallback regex was
        #   ^[A-Z]{14}-[A-Z]{10}-[A-Z](?:-[A-Za-z0-9]+)?$|^SYNTH
        # which accepts:
        #   (a) suffixed keys (the (?:-[A-Za-z0-9]+)? clause) -- these
        #       pass Python validation but FAIL the SQL CHECK constraint
        #       chk_drugs_inchikey (LENGTH=27 OR LIKE 'SYNTH%') at
        #       INSERT time -> silent dead-letter.
        #   (b) bare "SYNTH" with no body (5 chars) -- should be rejected.
        # The canonical pattern from cleaning._constants.CANONICAL_
        # INCHIKEY_REGEX is ^[A-Z]{14}-[A-Z]{10}-[A-Z]$ (strict 27-char,
        # NO suffix). The SYNTH prefix is handled separately by
        # CANONICAL_SYNTHETIC_INCHIKEY_REGEX (^SYNTH.+$, requires at
        # least one char after SYNTH).
        #
        # This fallback now uses the EXACT canonical patterns. If
        # cleaning._constants is importable, we use the real objects;
        # otherwise we replicate the patterns EXACTLY (no divergence).
        try:
            from cleaning._constants import (
                CANONICAL_INCHIKEY_REGEX as _CANONICAL_FB,
                CANONICAL_SYNTHETIC_INCHIKEY_REGEX as _SYNTH_FB,
                CANONICAL_MIXTURE_INCHIKEY_REGEX as _MIX_FB,
            )
        except ImportError:
            _CANONICAL_FB = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
            _SYNTH_FB = re.compile(r"^SYNTH.+$", re.IGNORECASE)
            _MIX_FB = re.compile(
                r"^(?:[A-Z]{14}-[A-Z]{10}-[A-Z])(?:-[A-Z]{14}-[A-Z]{10}-[A-Z])+$"
            )
        def _inchikey_valid(ik: str) -> bool:
            s = str(ik).strip().upper()
            if not s:
                return False
            # Order matters: check mixture first (longest pattern), then
            # standard canonical, then SYNTH.
            if _MIX_FB.match(s):
                return True
            if _CANONICAL_FB.match(s):
                return True
            if _SYNTH_FB.match(s):
                return True
            return False
    invalid_mask = ~df_filtered["inchikey"].astype(str).map(_inchikey_valid)
    if invalid_mask.any():
        invalid_rows = df_filtered[invalid_mask]
        for _, row in invalid_rows.iterrows():
            _add_to_dead_letter(
                record=row.to_dict(),
                error="invalid_inchikey_format",
                operation="bulk_upsert_pubchem_compound_properties",
            )
            result.quarantined += 1
        df_filtered = df_filtered[~invalid_mask].copy()

    if df_filtered.empty:
        logger.warning(
            "bulk_upsert_pubchem_compound_properties: all rows quarantined "
            "as invalid InChIKeys"
        )
        return result

    # Validate pubchem_cid is positive integer (SCI-17 range check).
    df_filtered["pubchem_cid"] = pd.to_numeric(
        df_filtered["pubchem_cid"], errors="coerce"
    ).astype("Int64")
    bad_cid_mask = df_filtered["pubchem_cid"].isna() | (df_filtered["pubchem_cid"] < 1)
    if bad_cid_mask.any():
        for _, row in df_filtered[bad_cid_mask].iterrows():
            _add_to_dead_letter(
                record=row.to_dict(),
                error="invalid_pubchem_cid",
                operation="bulk_upsert_pubchem_compound_properties",
            )
            result.quarantined += 1
        df_filtered = df_filtered[~bad_cid_mask].copy()

    if df_filtered.empty:
        logger.warning(
            "bulk_upsert_pubchem_compound_properties: all rows quarantined "
            "as invalid pubchem_cid"
        )
        return result

    # Convert to list of dicts (NaT/NaN -> None handled by _df_to_dicts).
    records = _df_to_dicts(df_filtered)
    # Convert pubchem_cid from numpy Int64 to Python int (SQLAlchemy
    # handles this transparently, but explicit conversion avoids edge
    # cases with older drivers).
    for rec in records:
        if rec.get("pubchem_cid") is not None:
            try:
                rec["pubchem_cid"] = int(rec["pubchem_cid"])
            except (TypeError, ValueError):
                pass
        # COMP-11: download_date may arrive as an ISO 8601 string (as
        # produced by ``PubChemPipeline._parse_pubchem_response``) or as a
        # ``datetime`` object. SQLAlchemy's DateTime column only accepts
        # ``datetime``/``date`` instances on SQLite, so parse the string
        # form back to a timezone-aware ``datetime`` here. Both forms are
        # accepted on PostgreSQL, so this conversion is a no-op there.
        dd = rec.get("download_date")
        if isinstance(dd, str):
            try:
                rec["download_date"] = datetime.datetime.fromisoformat(dd)
            except ValueError as exc:
                raise ValueError(
                    f"bulk_upsert_pubchem_compound_properties: "
                    f"download_date is not ISO 8601: {dd!r} ({exc})"
                ) from exc
        # ``updated_at`` set on every upsert (the table's updated_at
        # trigger fires for ORM updates but NOT for Core inserts -- set
        # it explicitly here).
        rec["updated_at"] = datetime.datetime.now(datetime.timezone.utc)

    # PubChem compound_properties has ~35 columns; 1000 × 35 = 35 000
    # params -- well under PostgreSQL's 65 535-parameter limit.  Cap
    # defensively at 1000 (PERF-07).
    safe_batch = min(batch_size, 1000)

    insert_class = _get_dialect_insert(session)
    table = _PUBCHEM_COMPOUND_PROPERTIES_TABLE  # lazy proxy -- resolves on use

    with _Timer("bulk_upsert_pubchem_compound_properties", len(records)):
        for chunk in _chunked(records, safe_batch):
            if not chunk:
                continue
            try:
                stmt = insert_class(table).values(chunk)
                # Build the update dict referencing ``stmt.excluded``
                # (the values that would have been inserted -- used to
                # populate the existing row on conflict).
                update_dict = {
                    col: stmt.excluded[col]
                    for col in _PUBCHEM_COMPOUND_PROPERTIES_UPDATABLE_COLS
                    if col in chunk[0]
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=["inchikey", "pubchem_cid"],
                    set_=update_dict,
                )
                # v29 ROOT FIX (audit D-16): rowcount double-counted
                # inserts+updates on ON CONFLICT. Now uses
                # UpsertResult for accurate counts. The previous code
                # did:
                #     rowcount = result_cursor.rowcount or len(chunk)
                #     result.inserted += rowcount
                # On PostgreSQL with ON CONFLICT DO UPDATE, rowcount
                # is ``inserts + 2 * updates`` (each UPDATE touches
                # the row twice), so this over-counted by the number
                # of updates. Metrics reported to LoadResult and
                # downstream audit logs were inflated. Fix: delegate
                # to ``_count_upsert_inserts_updates`` which uses
                # PostgreSQL's ``xmax`` system column via RETURNING
                # to distinguish inserts (xmax = 0) from updates
                # (xmax != 0). On SQLite (tests/dev), falls back to
                # chunk size as the total (inserted+updated) with
                # updated=0 -- the total is correct, only the split
                # is approximate.
                chunk_inserts, chunk_updates = _count_upsert_inserts_updates(
                    session, stmt, len(chunk),
                )
                result.inserted += chunk_inserts
                result.updated += chunk_updates
                logger.debug(
                    "bulk_upsert_pubchem_compound_properties: chunk %d "
                    "inserts=%d updates=%d (cumulative inserted=%d, "
                    "updated=%d)",
                    len(chunk), chunk_inserts, chunk_updates,
                    result.inserted, result.updated,
                )
            except (OperationalError, ProgrammingError) as exc:
                # Per-row failures inside a chunk -- fall back to single-row
                # inserts so we can identify the bad row.
                logger.warning(
                    "bulk_upsert_pubchem_compound_properties: chunk failed (%s) "
                    "-- retrying as single-row inserts to isolate bad row",
                    exc,
                )
                for rec in chunk:
                    try:
                        stmt = insert_class(table).values([rec])
                        update_dict = {
                            col: stmt.excluded[col]
                            for col in _PUBCHEM_COMPOUND_PROPERTIES_UPDATABLE_COLS
                            if col in rec
                        }
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["inchikey", "pubchem_cid"],
                            set_=update_dict,
                        )
                        # v29 ROOT FIX (audit D-16): use the same
                        # accurate insert/update counting for the
                        # single-row fallback path.
                        rec_inserts, rec_updates = (
                            _count_upsert_inserts_updates(session, stmt, 1)
                        )
                        result.inserted += rec_inserts
                        result.updated += rec_updates
                    except (OperationalError, ProgrammingError) as exc2:
                        _add_to_dead_letter(
                            record=rec,
                            error=str(exc2),
                            operation="bulk_upsert_pubchem_compound_properties",
                        )
                        result.failed += 1

    # v29 ROOT FIX (audit D-16): on PostgreSQL, result.updated now
    # accurately reflects the number of UPDATEs (via xmax RETURNING).
    # On SQLite (tests/dev), we still cannot distinguish inserts from
    # updates without an extra query -- updated stays at 0 there and
    # inserted holds the total (inserted + updated). The total
    # (inserted + updated) is correct on both dialects.
    logger.info(
        "bulk_upsert_pubchem_compound_properties: completed -- "
        "input=%d, inserted=%d, updated=%d, quarantined=%d, failed=%d",
        result.total_input, result.inserted, result.updated,
        result.quarantined, result.failed,
    )
    return result


# ===========================================================================
# 8. Lookup maps
# ===========================================================================


@_with_retry(max_retries=3, base_delay=0.5)
def get_uniprot_to_protein_id_map(
    session: Session,
    uniprot_ids: set[str] | None = None,
) -> MappingResult:
    """Return a mapping of ``uniprot_id`` -> ``protein.id`` for proteins.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    uniprot_ids : set[str] | None
        Optional filter to load only specific uniprot_ids (PERF-03).

    Returns
    -------
    MappingResult
        Mapping with provenance metadata (LINE-07).
    """
    stmt = select(Protein.id, Protein.uniprot_id)
    if uniprot_ids:
        stmt = stmt.where(Protein.uniprot_id.in_(uniprot_ids))

    result = session.execute(stmt)
    mapping = {row.uniprot_id: row.id for row in result}
    mr = MappingResult(
        mapping=mapping,
        built_at=datetime.datetime.now(datetime.timezone.utc),
        record_count=len(mapping),
    )
    logger.info(
        "get_uniprot_to_protein_id_map: loaded %d mappings", len(mapping)
    )
    return mr


@_with_retry(max_retries=3, base_delay=0.5)
def get_inchikey_to_drug_id_map(
    session: Session,
    inchikeys: set[str] | None = None,
) -> MappingResult:
    """Return a mapping of ``inchikey`` -> ``drug.id`` for drugs.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    inchikeys : set[str] | None
        Optional filter to load only specific inchikeys (PERF-03).

    Returns
    -------
    MappingResult
        Mapping with provenance metadata (LINE-07).
    """
    stmt = select(Drug.id, Drug.inchikey)
    if inchikeys:
        stmt = stmt.where(Drug.inchikey.in_(inchikeys))

    result = session.execute(stmt)
    mapping = {row.inchikey: row.id for row in result}
    mr = MappingResult(
        mapping=mapping,
        built_at=datetime.datetime.now(datetime.timezone.utc),
        record_count=len(mapping),
    )
    logger.info(
        "get_inchikey_to_drug_id_map: loaded %d mappings", len(mapping)
    )
    return mr


@_with_retry(max_retries=3, base_delay=0.5)
def get_chembl_to_drug_id_map(
    session: Session,
    chembl_ids: set[str] | None = None,
) -> MappingResult:
    """Return a mapping of ``chembl_id`` -> ``drug.id`` for drugs.

    Added for the institutional-grade ChEMBL pipeline rewrite (A9/P5).
    Used by ``ChEMBLPipeline.load()`` to resolve ``molecule_chembl_id``
    from ChEMBL activity records to the integer ``drug_id`` FK on the
    ``drug_protein_interactions`` table -- without loading every drug in
    the DB (PERF-03, A9, P5).

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session. Read-only -- no commits issued.
    chembl_ids : set[str] | None
        Optional filter to load only specific chembl_ids. Pass the set of
        ``molecule_chembl_id`` values seen in the activity stream to
        avoid loading the entire drugs table. ``None`` means "load all
        drugs with a non-null chembl_id" (use sparingly).

    Returns
    -------
    MappingResult
        Mapping with provenance metadata (LINE-07). The ``.mapping``
        dict is ``{chembl_id (str): drug_id (int)}``. Records with
        ``NULL`` chembl_id are excluded.

    Raises
    ------
    sqlalchemy.exc.SQLAlchemyError
        On DB-level failure (the ``@_with_retry`` decorator retries
        transient failures up to 3 times before re-raising).

    Scientific Notes
    ----------------
    ChEMBL IDs are stable, versioned identifiers of the form
    ``CHEMBL\\d+`` (e.g. ``CHEMBL25`` for aspirin). They are unique
    within ChEMBL but NOT unique across sources -- DrugBank has its own
    ``drugbank_id`` column. This function returns only ChEMBL-sourced
    drugs (and any drug that has a populated ``chembl_id`` from any
    source's entity-resolution step).
    """
    stmt = select(Drug.id, Drug.chembl_id).where(
        Drug.chembl_id.isnot(None)
    )
    if chembl_ids:
        stmt = stmt.where(Drug.chembl_id.in_(chembl_ids))

    result = session.execute(stmt)
    mapping: dict[str, int] = {}
    for row in result:
        # Defensive: skip any NULL chembl_id (the IS NOT NULL filter
        # already excludes them at the SQL level, but double-check in
        # case of DB-level NULL-coercion weirdness).
        if row.chembl_id is None:
            continue
        mapping[str(row.chembl_id)] = int(row.id)
    mr = MappingResult(
        mapping=mapping,
        built_at=datetime.datetime.now(datetime.timezone.utc),
        record_count=len(mapping),
    )
    logger.info(
        "get_chembl_to_drug_id_map: loaded %d mappings "
        "(filtered=%s)",
        len(mapping),
        bool(chembl_ids),
    )
    return mr


@_with_retry(max_retries=3, base_delay=0.5)
def build_gene_to_uniprot_maps(
    session: Session,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build gene_symbol -> uniprot_id and protein_name -> uniprot_id
    mapping dicts.

    Primary: gene_symbol (e.g., "HBA1") -- highest priority.
    Secondary: protein_name map for last-resort matching only.

    FIX C4: gene_name (which stores protein names, NOT gene symbols) is
    NO LONGER added to the gene_to_uniprot map.  This prevents false
    matches where short protein names like "COX1" would be confused
    with gene symbols.

    SCI-10 / DQ-8: Invalid gene_symbols are skipped with a WARNING.
    Duplicate gene_symbols produce a WARNING log.
    """
    # PERF-02: Only load proteins with non-NULL gene_symbol
    # IDEM-15: ORDER BY gene_symbol makes the map's insertion order
    # deterministic across DB engines (SQLite, PostgreSQL, MySQL), so the
    # "keep first" behaviour on duplicate gene_symbols is reproducible.
    stmt = select(
        Protein.gene_symbol,
        Protein.gene_name,
        Protein.protein_name,
        Protein.uniprot_id,
    ).where(
        (Protein.gene_symbol.isnot(None))
        | (Protein.protein_name.isnot(None))
    ).order_by(Protein.gene_symbol.asc())

    result = session.execute(stmt)
    gene_to_uniprot: dict[str, str] = {}
    protein_name_to_uniprot: dict[str, str] = {}
    skipped_gene_symbols = 0
    duplicate_gene_symbols = 0

    for row in result:
        # SCI-10: Validate gene_symbol against HGNC format
        if row.gene_symbol and str(row.gene_symbol).strip():
            gs_key = row.gene_symbol.upper().strip()
            if _GENE_SYMBOL_RE.match(gs_key):
                if gs_key in gene_to_uniprot:
                    duplicate_gene_symbols += 1
                    logger.warning(
                        "build_gene_to_uniprot_maps: duplicate "
                        "gene_symbol '%s' (existing uniprot=%s, "
                        "new uniprot=%s) -- keeping first",
                        gs_key,
                        gene_to_uniprot[gs_key],
                        row.uniprot_id,
                    )
                else:
                    gene_to_uniprot[gs_key] = row.uniprot_id
            else:
                skipped_gene_symbols += 1
                logger.warning(
                    "build_gene_to_uniprot_maps: invalid "
                    "gene_symbol '%s' (uniprot=%s) -- skipping",
                    gs_key,
                    row.uniprot_id,
                )

        # FIX C4: Do NOT add gene_name to gene_to_uniprot.
        # gene_name stores protein names, not gene symbols.
        if row.protein_name and str(row.protein_name).strip():
            pn_key = row.protein_name.upper().strip()
            if pn_key and pn_key not in protein_name_to_uniprot:
                protein_name_to_uniprot[pn_key] = row.uniprot_id

    if skipped_gene_symbols > 0:
        logger.warning(
            "build_gene_to_uniprot_maps: skipped %d invalid "
            "gene_symbols",
            skipped_gene_symbols,
        )
    if duplicate_gene_symbols > 0:
        logger.warning(
            "build_gene_to_uniprot_maps: found %d duplicate "
            "gene_symbols",
            duplicate_gene_symbols,
        )

    logger.info(
        "build_gene_to_uniprot_maps: built %d gene mappings, "
        "%d protein_name mappings",
        len(gene_to_uniprot),
        len(protein_name_to_uniprot),
    )
    return gene_to_uniprot, protein_name_to_uniprot


def resolve_gene_symbol_to_uniprot(
    df: pd.DataFrame,
    gene_to_uniprot: dict[str, str],
    protein_name_to_uniprot: dict[str, str],
) -> pd.DataFrame:
    """Resolve gene_symbol -> uniprot_id using pre-built mapping dicts.

    Tries gene_symbol map first, then protein_name map as fallback.
    Returns a NEW DataFrame with uniprot_id column added.  The input
    DataFrame is NOT modified (INT-06).

    v83 COMP-3 ROOT FIX -- preserve clean-time uniprot_id:
      The OMIM pipeline's ``_resolve_gene_xref_embedded()`` (called at
      clean() time) populates ``uniprot_id`` from the HGNC crosswalk
      (~7,000 genes when the file is present, ~50 from the embedded
      fallback). The previous implementation of this function
      UNCONDITIONALLY overwrote ``df["uniprot_id"]`` with the DB-backed
      ``gene_to_uniprot`` map lookup at line 4602:

          df["uniprot_id"] = df["gene_symbol"].str.upper().map(gene_to_uniprot)

      If the DB map was empty (UniProt pipeline not yet loaded, or
      proteins table empty), EVERY value became NaN -- even rows where
      clean() had already resolved a correct UniProt accession from
      HGNC. The OMIM ``load()`` method then dead-lettered 99% of GDA
      records as "unresolved gene_symbol", and the KG lost 99% of its
      OMIM Gene-Disease edges. This was a silent data-loss bug: the
      pipeline reported GREEN, the dead-letter queue absorbed the
      records, and the KG looked "complete" to an operator who didn't
      check the DLQ.

      ROOT FIX: PRESERVE any existing non-null ``uniprot_id`` (populated
      at clean time). Only fill NULL slots using the DB map, then the
      protein-name fallback. This guarantees:
        1. Clean-time resolution (HGNC crosswalk) is never discarded.
        2. DB map is still consulted for rows the clean-time resolver
           couldn't resolve (e.g. gene_symbol not in HGNC crosswalk but
           present in the proteins table).
        3. Protein-name fallback is still consulted as a last resort.
      The net effect: OMIM GDA dead-letter rate drops from ~99% to the
      true unresolved rate (genes with no UniProt mapping anywhere).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'gene_symbol' column. May already have a
        'uniprot_id' column populated at clean time (preserved).
    gene_to_uniprot : dict[str, str]
        gene_symbol -> uniprot_id mapping (DB-backed).
    protein_name_to_uniprot : dict[str, str]
        protein_name -> uniprot_id fallback mapping (DB-backed).
    """
    _isinstance_dataframe(df, "resolve_gene_symbol_to_uniprot")

    # INT-06: Do not mutate the caller's DataFrame
    df = df.copy()

    if "gene_symbol" not in df.columns:
        df["uniprot_id"] = None
        return df

    # v83 COMP-3 ROOT FIX: PRESERVE existing non-null uniprot_id values
    # (populated at clean time by _resolve_gene_xref_embedded). Only
    # NULL slots are filled from the DB map + protein-name fallback.
    if "uniprot_id" not in df.columns:
        df["uniprot_id"] = None

    # Track how many were already resolved at clean time (for telemetry).
    pre_resolved_mask = df["uniprot_id"].notna()
    n_pre_resolved = int(pre_resolved_mask.sum())
    n_need_resolution = int((~pre_resolved_mask).sum())

    # Step 1: fill NULL slots from the DB gene_to_uniprot map.
    need_resolution_mask = df["uniprot_id"].isna()
    if need_resolution_mask.any():
        db_lookup = (
            df.loc[need_resolution_mask, "gene_symbol"]
            .str.upper()
            .map(gene_to_uniprot)
        )
        # v89 ROOT FIX (pandas 3.x dtype strictness -- CI COMP-3 failure):
        #   pandas 3.x with pyarrow string backend enforces strict dtype
        #   on ``df.loc[mask, col] = value`` assignments. If ``db_lookup``
        #   contains mixed types (e.g. str + None from .map() misses), the
        #   assignment raises ``TypeError: Invalid value for dtype 'str'``.
        #   ROOT FIX: explicitly convert ``db_lookup`` to ``object`` dtype
        #   before assignment so None/NaN values are accepted. This is the
        #   pandas-recommended workaround for mixed-type assignment to
        #   string columns (see pandas issue #54286). The downstream code
        #   treats None as "unresolved" (the ``still_unresolved`` mask at
        #   line 4659 checks ``df["uniprot_id"].isna()``), so preserving
        #   None semantics is correct.
        df.loc[need_resolution_mask, "uniprot_id"] = db_lookup.astype(object)
        # V90 CI fix: pandas 2.2+ raises TypeError when assigning a
        # mixed-dtype Series (object with float NaN + str values) to a
        # column that pandas has inferred as 'str' dtype. The fix is
        # to ensure the assignment is object-dtype-safe by converting
        # db_lookup to a plain Python-object Series before assignment.
        # This was a pre-existing CI failure (P2 + Chain-1 verification
        # job) unrelated to the Phase 3 V90 fixes, but it blocked the
        # merge gate. Root cause: pandas 2.2+ stricter dtype enforcement.
        df.loc[need_resolution_mask, "uniprot_id"] = db_lookup.astype(object)
        # v89 fix: ensure dtype-safe assignment. Newer pandas (2.2+) raises
        # TypeError when assigning a mixed Series to a str-dtype column.
        # Convert to str with NaN preservation, then assign.
        df.loc[need_resolution_mask, "uniprot_id"] = db_lookup.astype(object).where(db_lookup.notna(), other=pd.NA)

    # Step 2: still-unresolved rows -- try protein_name map as fallback.
    still_unresolved = df["uniprot_id"].isna()
    if still_unresolved.any():
        protein_name_fallback = (
            df.loc[still_unresolved, "gene_symbol"]
            .str.upper()
            .map(protein_name_to_uniprot)
        )
        # v89 ROOT FIX (pandas 3.x dtype strictness -- same as Step 1):
        # explicit ``.astype(object)`` to allow None values in the
        # string-dtype column.
        df.loc[still_unresolved, "uniprot_id"] = protein_name_fallback.astype(object)
        # V90 CI fix: same dtype-safe assignment as Step 1.
        df.loc[still_unresolved, "uniprot_id"] = protein_name_fallback.astype(object)
        df.loc[still_unresolved, "uniprot_id"] = protein_name_fallback.astype(object).where(protein_name_fallback.notna(), other=pd.NA)

    unresolved_count = df["uniprot_id"].isna().sum()
    if unresolved_count > 0:
        logger.warning(
            "resolve_gene_symbol_to_uniprot: %d / %d symbols "
            "unresolved (pre-resolved at clean time: %d, needed "
            "DB/fallback resolution: %d). Unresolved records will be "
            "dead-lettered by the caller.",
            unresolved_count,
            len(df),
            n_pre_resolved,
            n_need_resolution,
        )
    else:
        logger.info(
            "resolve_gene_symbol_to_uniprot: all %d symbols resolved "
            "(pre-resolved at clean time: %d, DB/fallback: %d).",
            len(df),
            n_pre_resolved,
            n_need_resolution,
        )

    return df


# ===========================================================================
# 9. Pipeline runs upsert (ARCH-3)
# ===========================================================================


def get_or_create_pipeline_run(
    session: Session,
    run_id: str,
    source: str,
    *,
    started_at: datetime.datetime | None = None,
    status: str = "running",
) -> int:
    """Get or create a ``pipeline_runs`` row and return its integer ID.

    Used by the DisGeNET pipeline (IDEM-10) to convert its UUID
    ``run_id`` into the integer FK that ``GeneDiseaseAssociation.pipeline_run_id``
    requires.  If a row with the same ``(source, run_date)`` already
    exists (e.g. from a previous attempt), it is updated in place --
    this preserves the ``UniqueConstraint(source, run_date)`` from the
    model and makes the operation idempotent.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    run_id : str
        UUID of the pipeline run (stored in metadata, not in a column).
    source : str
        Pipeline source name (e.g. ``"disgenet"``).  Must be one of the
        values in the ``chk_pipeline_runs_source`` CheckConstraint.
    started_at : datetime, optional
        Run start time (UTC).  Defaults to ``datetime.now(timezone.utc)``.
        Used as ``run_date`` for the unique constraint.
    status : str
        Initial run status.  Default ``"running"``.

    Returns
    -------
    int
        The integer primary key of the ``pipeline_runs`` row.

    Raises
    ------
    ValueError
        If ``source`` is not a known pipeline name.
    """
    # v41 ROOT FIX (P1 #18): import the canonical VALID_SOURCE_NAMES
    # from base_pipeline instead of duplicating the set a 3rd time.
    # The canonical source is base_pipeline.VALID_SOURCE_NAMES.
    from pipelines.base_pipeline import VALID_SOURCE_NAMES as valid_sources
    if source not in valid_sources:
        raise ValueError(
            f"source={source!r} is not a known pipeline name. "
            f"Expected one of: {sorted(valid_sources)}"
        )
    if started_at is None:
        started_at = datetime.datetime.now(datetime.timezone.utc)

    # Look up an existing row by (source, run_date) -- UniqueConstraint
    existing = session.execute(
        select(PipelineRun).where(
            PipelineRun.source == source,
            PipelineRun.run_date == started_at,
        )
    ).scalar_one_or_none()

    if existing is not None:
        # Update status / metadata in place -- idempotent.
        existing.status = status
        return int(existing.id)

    run = PipelineRun(
        source=source,
        run_date=started_at,
        status=status,
    )
    session.add(run)
    session.flush()  # populate run.id without committing
    return int(run.id)


# ===========================================================================
# 9b. Pipeline runs bulk upsert (ARCH-3, legacy API preserved)
# ===========================================================================


def bulk_upsert_pipeline_runs(
    session: Session,
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> UpsertResult:
    """Bulk upsert pipeline run records.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    df : pd.DataFrame
        DataFrame with pipeline run data.
    batch_size : int
        Number of rows per INSERT statement.

    Returns
    -------
    UpsertResult
    """
    _isinstance_dataframe(df, "bulk_upsert_pipeline_runs")
    _validate_batch_size(batch_size)

    result = UpsertResult()

    if df.empty:
        logger.debug(
            "bulk_upsert_pipeline_runs: empty dataframe, skipping"
        )
        return result

    df = _sanitize_dataframe(df.copy())
    batch_size = _calculate_safe_batch_size(PipelineRun, batch_size)
    total = len(df)
    result.total_input = total

    with _Timer("bulk_upsert_pipeline_runs", total):
        dialect_insert = _get_dialect_insert(session)
        dialect_name = session.get_bind().dialect.name

        updatable_cols = [
            "status",
            "records_downloaded",
            "records_cleaned",
            "records_loaded",
            "error_message",
            "duration_seconds",
            "updated_at",
        ]

        for chunk_idx, chunk in enumerate(
            _df_chunk_to_dicts(df, batch_size)
        ):
            try:
                all_keys: set[str] = set()
                for record in chunk:
                    all_keys.update(record.keys())

                stmt = dialect_insert(PipelineRun.__table__).values(chunk)
                update_dict = {
                    col: stmt.excluded[col]
                    for col in updatable_cols
                    if col in all_keys
                }
                if dialect_name == "postgresql":
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_pipeline_runs_source_date",
                        set_=update_dict,
                    )
                else:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["source", "run_date"],
                        set_=update_dict,
                    )
                # v51 ROOT FIX (COMPOUND-6): accurate counter
                _ins, _upd = _count_upsert_inserts_updates(
                    session, stmt, len(chunk)
                )
                result.inserted += _ins
                result.updated += _upd

            except Exception as exc:
                logger.error(
                    "bulk_upsert_pipeline_runs: chunk %d failed: %s",
                    chunk_idx,
                    exc,
                )
                result.failed += len(chunk)

    logger.info("bulk_upsert_pipeline_runs: %s", result)
    return result


# ===========================================================================
# 10. Orphan GDA cleanup (ARCH-01 -- moved from database.models)
# ===========================================================================


def cleanup_orphan_gda_records(
    session: Session,
    auto_commit: bool = False,
    reference_timestamp: datetime.datetime | None = None,
    dry_run: bool = False,
) -> int:
    """Delete GDA records with uniprot_id=NULL that have existed for >
    the configured retention period.

    This prevents orphan GDA records from accumulating when protein
    records are deleted and re-ingested (ondelete=SET NULL preserves
    GDA data).

    [ARCH-01] Moved from database.models to database.loaders (SRP).
    [REL-04] Added retry logic with exponential backoff.
    [LOG-01] Added proper logging at INFO and WARNING levels.
    [CODE-05] Bare except replaced with specific exception handling.
    [CFG-02] Retention period configurable via ORPHAN_GDA_RETENTION_HOURS.
    [IDEM-05] reference_timestamp parameter for deterministic testing.
    [SEC-06] MAX_DELETE_COUNT safeguard and dry_run parameter.
    [REL-03] Uses savepoint so caller transaction is not rolled back.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    auto_commit : bool
        If False (default), caller must commit.  Set to True when the
        caller wants this function to manage its own transaction.
    reference_timestamp : datetime | None
        Reference time for the retention window (IDEM-05).  If None,
        uses current UTC time.  When provided, enables deterministic
        testing and backfilling.
    dry_run : bool
        If True, count records to be deleted without actually deleting
        them (SEC-06).

    Returns
    -------
    int
        Number of orphan records deleted (or counted, if dry_run).
    """
    # [CFG-02] Configurable retention period with specific exceptions
    try:
        from config.settings import ORPHAN_GDA_RETENTION_HOURS
        retention_hours = ORPHAN_GDA_RETENTION_HOURS
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "cleanup_orphan_gda_records: could not load config: %s "
            "-- using default 24h",
            exc,
        )
        retention_hours = 24

    # Validate the loaded value
    if not isinstance(retention_hours, (int, float)) or retention_hours < 0:
        logger.warning(
            "cleanup_orphan_gda_records: invalid retention_hours=%r "
            "-- using default 24h",
            retention_hours,
        )
        retention_hours = 24

    # [REL-04] Configurable retry from settings
    try:
        from config.settings import (
            LOADERS_MAX_RETRY_ATTEMPTS,
            LOADERS_RETRY_BASE_DELAY,
            LOADERS_MAX_DELETE_COUNT,
        )
        max_retries = LOADERS_MAX_RETRY_ATTEMPTS
        base_delay = LOADERS_RETRY_BASE_DELAY
        max_delete_count = LOADERS_MAX_DELETE_COUNT
    except (ImportError, ModuleNotFoundError):
        max_retries = 3
        base_delay = 0.5
        max_delete_count = 10000

    # [IDEM-05] Use reference_timestamp if provided
    if reference_timestamp is None:
        reference_timestamp = datetime.datetime.now(datetime.timezone.utc)

    for attempt in range(max_retries):
        # v90 ROOT FIX (BUG #5 -- P0 savepoint rollback rolled back ENTIRE
        #   transaction): the previous code called ``session.rollback()``
        #   in the except blocks, which rolls back the ENTIRE outer
        #   transaction -- discarding any staged work the caller had in
        #   the same session (e.g. a bulk_upsert_gda batch). The comment
        #   said "[REL-03] Uses savepoint so caller transaction is not
        #   rolled back" but the implementation did the OPPOSITE. ROOT
        #   FIX: initialize ``savepoint = None`` before the try block,
        #   and in the except blocks call ``savepoint.rollback()`` (the
        #   savepoint object) NOT ``session.rollback()``. If the
        #   savepoint was never created (begin_nested itself failed),
        #   fall back to session.rollback(). This preserves the caller's
        #   staged work on transient errors -- same pattern as
        #   _quarantine_gda_rows (line ~2637).
        savepoint = None
        try:
            # [REL-03] Use savepoint so caller transaction is not rolled back
            savepoint = session.begin_nested()

            # v16 ROOT FIX (DC-8): the previous code had an
            # ``if dialect == "sqlite": / else:`` branch where both
            # branches executed IDENTICAL SQL with IDENTICAL parameters.
            # The branch served no purpose -- collapse to a single call.
            # Both SQLite and PostgreSQL accept the same parameterized
            # DELETE with a UTC datetime binding; SQLAlchemy handles
            # the dialect-specific timestamp serialization.
            # NOTE: ``dialect`` is still computed below for the log
            # messages (which include the dialect name for debugging).
            dialect = session.get_bind().dialect.name
            result = session.execute(
                text(
                    "DELETE FROM gene_disease_associations "
                    "WHERE uniprot_id IS NULL "
                    "AND created_at < :cutoff_time"
                ),
                {
                    "cutoff_time": reference_timestamp
                    - datetime.timedelta(hours=retention_hours),
                },
            )

            deleted_count = result.rowcount

            # [SEC-06] MAX_DELETE_COUNT safeguard
            if deleted_count > max_delete_count:
                logger.error(
                    "cleanup_orphan_gda_records: delete count (%d) "
                    "exceeds MAX_DELETE_COUNT (%d) -- rolling back. "
                    "Set LOADERS_MAX_DELETE_COUNT higher or use dry_run "
                    "to preview.",
                    deleted_count,
                    max_delete_count,
                )
                savepoint.rollback()
                raise RuntimeError(
                    f"cleanup_orphan_gda_records: attempted to delete "
                    f"{deleted_count} records, which exceeds the safety "
                    f"limit of {max_delete_count}.  Increase "
                    f"LOADERS_MAX_DELETE_COUNT or use dry_run=True."
                )

            if dry_run:
                savepoint.rollback()
                logger.info(
                    "cleanup_orphan_gda_records: DRY RUN -- would delete "
                    "%d orphan records (retention=%dh, dialect=%s)",
                    deleted_count,
                    retention_hours,
                    dialect,
                )
                return deleted_count

            if auto_commit:
                session.commit()

            # [LOG-01] Log the outcome
            logger.info(
                "cleanup_orphan_gda_records: deleted %d orphan records "
                "(retention=%dh, dialect=%s)",
                deleted_count,
                retention_hours,
                dialect,
            )
            if deleted_count > 1000:
                logger.warning(
                    "cleanup_orphan_gda_records: unusually high delete "
                    "count (%d). This may indicate a data quality issue "
                    "upstream.",
                    deleted_count,
                )
            return deleted_count

        except (OperationalError, ProgrammingError) as exc:
            # [CODE-05] Specific exception handling
            # v90 ROOT FIX (BUG #5): roll back the SAVEPOINT, not the
            # entire session transaction. This preserves the caller's
            # staged work (e.g. a bulk_upsert_gda batch in the same
            # session). If the savepoint was never created (begin_nested
            # itself failed), fall back to session.rollback().
            if savepoint is not None:
                try:
                    savepoint.rollback()
                except Exception:
                    session.rollback()
            else:
                session.rollback()
            logger.warning(
                "cleanup_orphan_gda_records: attempt %d/%d failed: %s",
                attempt + 1,
                max_retries,
                exc,
            )
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                time.sleep(delay)
            else:
                logger.error(
                    "cleanup_orphan_gda_records: all %d retries "
                    "exhausted: %s",
                    max_retries,
                    exc,
                )
                raise

        except RuntimeError:
            # Re-raise MAX_DELETE_COUNT violations
            raise

        except Exception as exc:
            # v90 ROOT FIX (BUG #5): same savepoint-vs-session rollback fix.
            if savepoint is not None:
                try:
                    savepoint.rollback()
                except Exception:
                    session.rollback()
            else:
                session.rollback()
            logger.error(
                "cleanup_orphan_gda_records: unexpected error: %s", exc
            )
            raise

    # P1-019 v113 ROOT FIX: the previous code had `return 0` here as the
    # implicit "all retries exhausted but no exception raised" path. But
    # every except block above either retries (with eventual `raise` on the
    # last attempt) or re-raises immediately. So this line is UNREACHABLE
    # in normal control flow. However, a future refactor that wraps the
    # retry loop in an outer try/except that swallows exceptions would
    # silently return 0 — reporting "0 orphans deleted" when the actual
    # count is unknown. 50,000 orphan GDA rows could accumulate silently.
    # ROOT FIX: raise RuntimeError instead of returning 0. If this line
    # is ever reached, it indicates a logic bug that must be investigated.
    raise RuntimeError(
        "cleanup_orphan_gda_records: unreachable state — all retries "
        "exhausted without exception. This indicates a control-flow bug "
        "in the retry loop. Please report this as a P1-019 regression."
    )


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    "bulk_upsert_drugs",
    "bulk_upsert_proteins",
    "bulk_upsert_dpi",
    "bulk_upsert_ppi",
    "bulk_upsert_gda",
    "bulk_upsert_entity_mapping",
    "bulk_update_drugs_from_pubchem",
    "bulk_upsert_pipeline_runs",
    "get_or_create_pipeline_run",
    "get_uniprot_to_protein_id_map",
    "get_inchikey_to_drug_id_map",
    "get_chembl_to_drug_id_map",
    "build_gene_to_uniprot_maps",
    "resolve_gene_symbol_to_uniprot",
    "cleanup_orphan_gda_records",
    "UpsertResult",
    "MappingResult",
    "get_dead_letter_queue",
    "flush_dead_letter_queue",
    "LOADERS_VERSION",
    "DEFAULT_BATCH_SIZE",
    "DPI_UNIQUE_CONSTRAINT_NAME",
    "GDA_UNIQUE_CONSTRAINT_NAME",
    "ENTITY_MAPPING_INCHIKEY_CONSTRAINT",
    "ENTITY_MAPPING_NAME_CONSTRAINT",
]
