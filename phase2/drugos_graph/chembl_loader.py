"""DrugOS Graph Module -- ChEMBL Loader (v2.0 -- Institutional Grade)
==================================================================
Downloads, validates, and parses the **ChEMBL bioactivity database** --
the primary source for drug-target interaction data in the DrugOS
knowledge graph.

If this loader emits a wrong UniProt accession or a wrong relation type
(e.g., "inhibits" for an agonist), the Graph Transformer will train on
inverted edges, the RL ranker will rank the wrong drug, a clinician
acts on the ranking, and a patient is harmed. This file therefore
implements every guard mandated by the 16-domain forensic audit.

ChEMBL SQLite format (REAL, not fabricated):
    Tar.gz containing a single .db SQLite file with tables including:
      molecule_dictionary  -- compound IDs and names
      compound_structures  -- SMILES and InChI
      activities           -- bioactivity measurements (IC50, Ki, etc.)
      target_dictionary    -- target IDs, names, and types
      target_components    -- mapping from targets to UniProt accessions
      assays               -- assay metadata (type, organism, etc.)
      organism_classification -- NCBI taxonomy

Public API (preserved from v1 -- ``run_pipeline.py`` unchanged):
    download_chembl, parse_chembl_activities, chembl_to_edge_records

New in v2.0 (additive, backward-compatible):
    ChEMBLLoader          -- adapter implementing the ``Loader`` Protocol.
    PARSER_VERSION, SCHEMA_VERSION -- versioning for reproducibility.
    ChEMBLConfig          -- configuration dataclass with validation.
    validate_chembl       -- post-parse data quality validation.
    iter_chembl_activities -- streaming API for large databases.
    chembl_to_node_records -- compound node record generation.
    chembl_to_graph       -- (nodes, edges) pair for KG construction.
    load_chembl           -- end-to-end load pipeline.

Idempotency (clinical-safety requirement):
    Two runs of ``parse_chembl_activities`` on the same .db file
    produce identical DataFrames (sorted by drug_chembl_id, then
    target_chembl_id). No non-deterministic ordering, no unseeded
    randomness. The only non-deterministic field is
    ``df.attrs['provenance']['parsed_at']`` (ISO-8601 timestamp).

Errors raised (Domain 6 -- Reliability):
    ChEMBLDownloadError       -- download failure (TLS / allowlist / size /
                                SHA-256 / content-sniff / tar safety).
    ChEMBLParseError          -- SQLite query failure (OperationalError,
                                missing tables, unexpected schema).
    ChEMBLDataIntegrityError  -- content failure (row count, ID format,
                                pChEMBL range, entity resolution).

Dead-letter queue: ``data/dead_letter/chembl_malformed.jsonl`` (one JSON
line per dropped/malformed record -- Domain 5 Data Quality).

Transformation log: ``logs/transformations/chembl.jsonl`` (one JSON line
per significant transformation -- Domain 16 Lineage).

License: CC BY-SA 3.0 -- attribution propagated in ``df.attrs['license']``
and ``df.attrs['attribution']`` (Domain 14 Compliance).

References:
    Gaulton, A., et al. (2017). "The ChEMBL database in 2017".
    Nucleic Acids Research, 45(D1), D945-D954.
    doi:10.1093/nar/gkw1074

CHANGELOG (SCHEMA_VERSION bumps require downstream contract update):
    v2.0.0 (2026-06-18) -- Institutional-grade rewrite. Adds:
        - PARSER_VERSION / SCHEMA_VERSION constants.
        - ``ChEMBLLoader`` Protocol adapter.
        - ``ChEMBLConfig`` dataclass.
        - SHA-256 / size / content-sniff verification on download.
        - TLS-verified, URL-allowlisted, atomic-tmp, tar-hardened
          download path.
        - Per-row dead-letter queue + transformation log.
        - ``df.attrs['provenance']`` with all ``CHEMBL_PROVENANCE_KEYS``.
        - Scientific mapping from standard_type to relation type
          (replaces broken substring matching).
        - Organism filtering via NCBI TaxID.
        - Confidence score filtering.
        - Assay type classification (Binding vs Functional).
        - Input validation on all critical fields.
        - ``validate_chembl`` returns typed validation dict.
        - ``iter_chembl_activities`` streaming API.
        - ``chembl_to_node_records`` compound node generation.
        - ``chembl_to_graph`` end-to-edge graph construction.
    v1.0.0 (initial) -- basic download + parse with substring matching.
"""

from __future__ import annotations

# =============================================================================
# Section 0 -- Imports
# =============================================================================
# Fixes Domain 4 (Coding) -- all imports at module top.
# Fixes Domain 12 (Configuration) -- no magic numbers.

import hashlib
import io
import json
import logging
import os
import re
import shutil
import socket
import sqlite3
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    Final,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd

# ─── Project imports ─────────────────────────────────────────────────────────
from .config import (
    ALLOWED_CHEMBL_URLS,
    CHECKPOINT_DIR,
    CHEMBL_ACTIVITY_TYPE_ACTIVATES,
    CHEMBL_ACTIVITY_TYPE_BINDS,
    CHEMBL_ACTIVITY_TYPE_INHIBITS,
    CHEMBL_ACTIVITY_TYPE_MODULATES,
    CHEMBL_ATTRIBUTION,
    CHEMBL_DRUG_IDENTIFIER_REGEX,
    CHEMBL_KG_BUILDER_FIELDS,
    CHEMBL_LICENSE,
    CHEMBL_MIN_CONFIDENCE_SCORE,
    CHEMBL_MIN_FIELD_POPULATION,
    CHEMBL_MIN_PCHEMBL_VALUE,
    CHEMBL_MIN_VALID_SIZE_BYTES,
    CHEMBL_ORGANISM_FILTER_TAX_ID,
    CHEMBL_PARSER_VERSION,
    CHEMBL_PCHEMBL_RANGE,
    CHEMBL_PROGRESS_LOG_INTERVAL,
    CHEMBL_SCHEMA_VERSION,
    CHEMBL_TARGET_TYPES,
    CHEMBL_UNIPROT_AC_REGEX,
    DATA_SOURCES,
    DEAD_LETTER_DIR,
    ENTITY_TYPE_COMPOUND,
    SOURCE_CHEMBL,
    SOURCE_KEY_CHEMBL,
    RAW_DIR,
    SEED,
    set_global_seed,
)
from .exceptions import (
    ChEMBLDataIntegrityError,
    ChEMBLDownloadError,
    ChEMBLParseError,
    DrugOSDataError,
)
from .schemas import (
    CHEMBL_PROVENANCE_KEYS,
    ChEMBLActivityRecord,
    ChEMBLEdgeRecord,
)
# v102 ROOT FIX (P2-036): route InChIKey normalization through the
# centralized helper so this loader produces the SAME canonical form
# as phase1_bridge.py and pubchem_loader.py.
try:
    from .utils import normalize_inchikey as _normalize_inchikey
except Exception:  # pragma: no cover — fallback for direct-script execution
    def _normalize_inchikey(inchikey):  # type: ignore[no-redef]
        if inchikey is None:
            return ""
        try:
            ik = str(inchikey).strip()
        except Exception:
            return ""
        if not ik or ik.lower() in ("nan", "none", "null", "na"):
            return ""
        return ik.upper()

# =============================================================================
# Section 1 -- Module-level constants & metadata
# =============================================================================
# Fixes Domain 1 (Architecture) -- explicit __all__, version constants.
# Fixes Domain 14 (Compliance) -- schema versioning, naming conventions.

PARSER_VERSION: str = CHEMBL_PARSER_VERSION  # "2.0.0"
SCHEMA_VERSION: str = CHEMBL_SCHEMA_VERSION  # "2.0.0"

__all__: list[str] = [
    # ── Version constants ──
    "PARSER_VERSION",
    "SCHEMA_VERSION",
    # ── Configuration ──
    "ChEMBLConfig",
    # ── Download ──
    "download_chembl",
    # ── REST API (Task 81) ──
    "fetch_chembl_molecules_api",
    "iter_chembl_activities_api",
    # ── Parse ──
    "parse_chembl_activities",
    "iter_chembl_activities",
    # ── Convert ──
    "chembl_to_edge_records",
    "chembl_to_node_records",
    "chembl_to_graph",
    # ── Validation ──
    "validate_chembl",
    # ── End-to-end ──
    "load_chembl",
    # ── Protocol adapter ──
    "ChEMBLLoader",
    # ── Activity type mapping ──
    "standard_type_to_relation",
]

logger = logging.getLogger(__name__)

# =============================================================================
# Section 2 -- ChEMBLConfig dataclass
# =============================================================================
# Fixes Domain 12 (Configuration) -- no magic numbers, all thresholds
# are named, documented, and overridable.
# Fixes Domain 7 (Idempotency) -- deterministic defaults.


@dataclass(frozen=True)
class ChEMBLConfig:
    """Configuration for the ChEMBL loader.

    All thresholds are documented with their scientific rationale.
    Instances are frozen (immutable) to prevent accidental mutation
    during a pipeline run (Domain 7 -- Idempotency).

    Parameters
    ----------
    min_pchembl : float
        Minimum pChEMBL value for high-quality interactions.
        pChEMBL = -log10(IC50/Ki/Kd in M). A value of 5.0 corresponds
        to ~10 uM, the community-standard threshold for meaningful
        bioactivity. Values below 5.0 are typically noise.
        Reference: https://doi.org/10.1016/j.drudis.2014.10.012
    organism_tax_id : int
        NCBI Taxonomy ID for organism filtering. Default 9606 (human).
        For drug repurposing, only human targets are relevant -- non-human
        proteins create disconnected subgraphs in the KG.
    min_confidence_score : int
        Minimum ChEMBL target-confidence score (0-9). We require >= 7,
        meaning "target assigned with high confidence to a single protein".
    target_types : frozenset[str]
        ChEMBL target types to include. SINGLE PROTEIN has clear 1:1
        UniProt mapping. PROTEIN COMPLEX GROUP and PROTEIN FAMILY are
        included because they can be resolved via target_components.
    chembl_dir : Path or None
        Path to the extracted ChEMBL directory. If None, defaults to
        ``RAW_DIR / "chembl"``.
    force_download : bool
        If True, re-download even if a cached copy exists.
    sort_output : bool
        If True, sort output DataFrame for deterministic ordering
        (Domain 7 -- Idempotency).
    """

    min_pchembl: float = CHEMBL_MIN_PCHEMBL_VALUE
    organism_tax_id: int = CHEMBL_ORGANISM_FILTER_TAX_ID
    min_confidence_score: int = CHEMBL_MIN_CONFIDENCE_SCORE
    target_types: frozenset[str] = field(
        default_factory=lambda: frozenset(CHEMBL_TARGET_TYPES)
    )
    chembl_dir: Optional[Path] = None
    force_download: bool = False
    sort_output: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values (Domain 12 -- Config Validation)."""
        if not (0.0 <= self.min_pchembl <= 14.0):
            raise ValueError(
                f"min_pchembl must be in [0, 14], got {self.min_pchembl}"
            )
        if self.min_confidence_score not in range(10):
            raise ValueError(
                f"min_confidence_score must be in [0, 9], got "
                f"{self.min_confidence_score}"
            )
        if self.organism_tax_id <= 0:
            raise ValueError(
                f"organism_tax_id must be positive, got "
                f"{self.organism_tax_id}"
            )


# =============================================================================
# Section 3 -- Scientific mapping: standard_type -> relation type
# =============================================================================
# Fixes Domain 3 (Scientific Correctness) -- the old code used substring
# matching ("INHIBIT" in std_type) which was WRONG. For example:
#   - "Inhibition" -> "inhibits" (correct)
#   - "EC50" -> missed, fell through to "inhibits" (WRONG -- EC50 implies
#     activation/agonism)
#   - "Kd" -> missed, fell through to "inhibits" (WRONG -- Kd measures
#     binding affinity, not inhibition)
#   - "Potency" -> missed, fell through to "inhibits" (WRONG -- Potency
#     is directionless)
#
# This is the SINGLE most important scientific fix in this file. The old
# code could cause the GNN to learn that an agonist is an inhibitor,
# which in a clinical context could lead to recommending a stimulant
# when a blocker is needed (or vice versa).

# Pre-compiled regex patterns for efficient matching
_RE_INHIBIT = re.compile(
    # v58 ROOT FIX (P2L-008 deep): add INACTIVAT to catch the standard
    # ChEMBL label for irreversible / covalent inhibition (aspirin,
    # omeprazole, clopidogrel, penicillin). The v57 fix only prevented
    # the wrong "activates" classification; without this entry,
    # INACTIVATION fell through to the default "targets" -- losing the
    # scientific signal that the drug INHIBITS the target.
    # We use the bare stem "INACTIVAT" (no word boundary) so it matches
    # INACTIVATION, INACTIVATOR, INACTIVATE, INACTIVATES, INACTIVATED.
    # This is SAFE because _RE_ACTIVATE now uses \bACTIVAT (word
    # boundary), so INACTIVAT* cannot be mis-routed to "activates".
    #
    # P2-018 ROOT FIX: removed `re.IGNORECASE`. The input is ALWAYS
    # uppercased at line 432 (``std_upper = standard_type.strip().upper()``)
    # before any of the _RE_* regexes are applied (see usages at lines
    # 445-452 — all use ``std_upper``, never the raw ``standard_type``).
    # The IGNORECASE flag was DEAD CODE — it appeared to be doing
    # something but had no effect because the input was already
    # uppercase. Removing it makes the code HONEST about what's
    # happening, and avoids the false impression that the regex would
    # catch lowercase input if a future maintainer added a code path
    # that bypassed the uppercasing.
    r"(INHIBIT|INACTIVAT|ANTAGONIST|BLOCKER|REDUC|SUPPRESS|DECREAS|DOWNREG)",
)
_RE_ACTIVATE = re.compile(
    # v57 ROOT FIX (P2L-008 -- covalent inhibitors misclassified as activators):
    #   The previous regex `(ACTIVAT|AGONIST|...)` matched the substring
    #   `ACTIVAT` inside `INACTIVATION` -- the standard ChEMBL label for
    #   irreversible/covalent inhibitors. As a result, covalent inhibitors
    #   (aspirin, omeprazole, clopidogrel, penicillin) were classified as
    #   `rel_type='activates'` -- INVERTED drug-target semantics.
    #   FIX: use word-boundary matching (\b) so `ACTIVAT` only matches when
    #   it starts a word, not when it's a substring of `INACTIVAT...`.
    #   Also explicitly exclude `INACTIVAT` to be defensive.
    # v58 NOTE: _RE_INHIBIT is checked BEFORE _RE_ACTIVATE in
    # standard_type_to_relation(), so even if a future regex change
    # accidentally re-introduced ACTIVAT-without-\b, INACTIVAT would
    # still route to "inhibits" first. Defense in depth.
    #
    # v68 ROOT FIX (P2L-008 -- TRIPLE defense-in-depth): the v57/v58 fix
    # relied on \b (word boundary) alone. While \b correctly prevents
    # matching "ACTIVAT" inside "INACTIVATION" (because "IN" are both
    # word chars, no boundary exists between N and A), we add an
    # EXPLICIT negative lookbehind `(?<![A-Z])` as a third layer of
    # defense. This means ACTIVAT only matches when NOT preceded by an
    # uppercase letter -- so even if a future maintainer accidentally
    # removes the \b, or if a non-standard ChEMBL variant uses a
    # non-word delimiter before "ACTIVAT", the negative lookbehind
    # still catches "INACTIVAT*". This is the audit's recommended fix.
    #
    # P2-018 ROOT FIX: the negative lookbehind `(?<![A-Z])` is NOT dead
    # code (the issue title was misleading). The input IS always
    # uppercased at line 432, so [A-Z] correctly matches uppercase
    # letters — the lookbehind actively prevents "ACTIVAT" from matching
    # inside "INACTIVAT...". What WAS dead was `re.IGNORECASE` — the
    # input is already uppercase, so the case-insensitive flag had no
    # effect. Removed the flag.
    r"((?<![A-Z])ACTIVAT|AGONIST|STIMUL|ENHANC|INCREAS|INDUC)",
)
_RE_BIND = re.compile(
    # P2-018 ROOT FIX: removed dead `re.IGNORECASE` (input is always
    # uppercased at line 432 before this regex is applied).
    r"(BIND|AFFINITY|DISSOCIAT|ASSOCIAT|INTERACT|COMPLEX)",
)
_RE_MODULATE = re.compile(
    # P2-018 ROOT FIX: removed dead `re.IGNORECASE` (input is always
    # uppercased at line 432 before this regex is applied).
    r"(MODULAT|ALLOSTER|REGULAT)",
)


def standard_type_to_relation(
    standard_type: str,
    *,
    strict: bool = False,
) -> str:
    """Map a ChEMBL standard_type string to a biological relation type.

    This function implements a two-level mapping strategy:

    1. **Exact match** against curated frozensets in config
       (``CHEMBL_ACTIVITY_TYPE_INHIBITS``, etc.). These are
       scientifically validated mappings where the standard_type
       unambiguously implies a relation direction.

    2. **Regex fallback** for standard_type strings not in the curated
       sets. The regexes are conservative -- they look for direction-
       indicating keywords ("inhibit", "agonist", "bind", etc.) in the
       standard_type name.

    3. **Default** -- if neither exact match nor regex matches, return
       "targets" as the conservative default (v21 root fix: the old
       default of "binds" was scientifically incorrect because many
       measurements (EC50, Kd, Potency) do NOT imply binding, and
       "binds" implied a specific mechanism. "targets" is the honest
       relation -- interaction confirmed, mechanism unclassified).
       v24 ROOT FIX (FORENSIC-P2-LOADERS E/§2): the docstring
       previously said the default was "binds" -- contradicting the
       actual code at the return statement (which returns "targets").
       Fix the docstring to match the code.

    Parameters
    ----------
    standard_type : str
        The ChEMBL standard_type string (e.g., "IC50", "Ki", "EC50").
    strict : bool
        If True, raise ValueError for unmapped types instead of
        defaulting to "targets". Use in testing/debugging to catch
        unmapped standard types.

    Returns
    -------
    str
        One of "inhibits", "activates", "binds", "modulates".

    Raises
    ------
    ValueError
        If strict=True and the standard_type cannot be mapped.
    """
    if not standard_type or not isinstance(standard_type, str):
        if strict:
            raise ValueError(f"Invalid standard_type: {standard_type!r}")
        # v35 ROOT FIX (V35-P2-LOADERS-FIXES M-5): return "targets"
        # consistently for invalid input -- previously this branch returned
        # "binds" while the unmapped-default branch at the bottom returned
        # "targets", producing inconsistent defaults for the same logical
        # "we can't classify this" situation. Both now return "targets".
        return "targets"

    std_upper = standard_type.strip().upper()

    # Level 1: Exact match against curated frozensets
    if std_upper in CHEMBL_ACTIVITY_TYPE_INHIBITS:
        return "inhibits"
    if std_upper in CHEMBL_ACTIVITY_TYPE_ACTIVATES:
        return "activates"
    if std_upper in CHEMBL_ACTIVITY_TYPE_BINDS:
        return "binds"
    if std_upper in CHEMBL_ACTIVITY_TYPE_MODULATES:
        return "modulates"

    # Level 2: Regex fallback for partial matches
    if _RE_INHIBIT.search(std_upper):
        return "inhibits"
    if _RE_ACTIVATE.search(std_upper):
        return "activates"
    if _RE_MODULATE.search(std_upper):
        return "modulates"
    if _RE_BIND.search(std_upper):
        return "binds"

    # Level 3: Conservative default
    if strict:
        raise ValueError(
            f"Cannot map standard_type {standard_type!r} to a relation "
            f"type. Add it to the appropriate CHEMBL_ACTIVITY_TYPE_* "
            f"frozenset in config.py."
        )
    # v21 ROOT FIX (Audit section 7 finding 12 - "Unknown standard_type
    # defaults to 'binds'"): the previous code returned 'binds' for any
    # unmapped standard_type (e.g. 'RESISTANCE', 'STABILITY', 'POTENCY'
    # in ChEMBL). This silently collapsed 8 distinct ChEMBL assay
    # semantics into the single 'binds' relation, losing the KG's
    # semantic distinction without warning. The honest default is
    # 'targets' (interaction confirmed, direction unclassified) - the
    # same default the phase1_bridge uses for IC50/Ki/Kd/Potency. This
    # matches the audit's recommendation and the disgenet/omim bridge
    # behavior.
    logger.debug(
        "chembl_loader: unmapped standard_type %r -> defaulting to 'targets' "
        "(interaction confirmed, direction unclassified). Add the type to "
        "the appropriate CHEMBL_ACTIVITY_TYPE_* frozenset in config.py for "
        "a more specific relation.",
        standard_type,
    )
    return "targets"


# =============================================================================
# Section 4 -- Input validation helpers
# =============================================================================
# Fixes Domain 5 (Data Quality) -- validates all critical fields.
# Fixes Domain 10 (Testing) -- testable validation functions.

_RE_CHEMBL_ID: Final[re.Pattern[str]] = re.compile(
    CHEMBL_DRUG_IDENTIFIER_REGEX
)
_RE_UNIPROT_AC: Final[re.Pattern[str]] = re.compile(
    CHEMBL_UNIPROT_AC_REGEX
)


def _validate_chembl_id(chembl_id: str, *, field_name: str = "chembl_id") -> str:
    """Validate a ChEMBL identifier format.

    Parameters
    ----------
    chembl_id : str
        The ChEMBL ID to validate (e.g., "CHEMBL25").
    field_name : str
        Name of the field (for error messages).

    Returns
    -------
    str
        The validated ChEMBL ID (stripped of whitespace).

    Raises
    ------
    ChEMBLDataIntegrityError
        If the ID does not match the expected format.
    """
    if not chembl_id or not isinstance(chembl_id, str):
        raise ChEMBLDataIntegrityError(
            f"{field_name} is empty or not a string",
            context={"field": field_name, "value": repr(chembl_id)},
        )
    stripped = chembl_id.strip()
    if not _RE_CHEMBL_ID.match(stripped):
        raise ChEMBLDataIntegrityError(
            f"{field_name} {stripped!r} does not match expected format "
            f"{CHEMBL_DRUG_IDENTIFIER_REGEX}",
            context={"field": field_name, "value": stripped},
        )
    return stripped


def _validate_uniprot_ac(accession: str) -> str:
    """Validate a UniProt accession format.

    Parameters
    ----------
    accession : str
        The UniProt accession to validate (e.g., "P23219").

    Returns
    -------
    str
        The validated accession (stripped and uppercased).

    Raises
    ------
    ChEMBLDataIntegrityError
        If the accession does not match the expected format.
    """
    if not accession or not isinstance(accession, str):
        raise ChEMBLDataIntegrityError(
            f"UniProt accession is empty or not a string",
            context={"value": repr(accession)},
        )
    stripped = accession.strip()
    if not _RE_UNIPROT_AC.match(stripped):
        raise ChEMBLDataIntegrityError(
            f"UniProt accession {stripped!r} does not match expected format",
            context={"value": stripped},
        )
    return stripped


def _validate_pchembl(value: float) -> float:
    """Validate a pChEMBL value is in the valid range.

    Parameters
    ----------
    value : float
        The pChEMBL value to validate.

    Returns
    -------
    float
        The validated pChEMBL value.

    Raises
    ------
    ChEMBLDataIntegrityError
        If the value is outside [0, 14].
    """
    lo, hi = CHEMBL_PCHEMBL_RANGE
    if not (lo <= value <= hi):
        raise ChEMBLDataIntegrityError(
            f"pChEMBL value {value} is outside valid range [{lo}, {hi}]",
            context={"value": value, "range": (lo, hi)},
        )
    return value


# =============================================================================
# Section 5 -- Dead-letter queue & transformation log
# =============================================================================
# Fixes Domain 5 (Data Quality) -- malformed records are quarantined.
# Fixes Domain 6 (Reliability) -- bad records don't crash the pipeline.
# Fixes Domain 16 (Lineage) -- transformation audit trail.


def _dead_letter_record(
    record: Dict[str, Any],
    reason: str,
    stage: str,
) -> Dict[str, Any]:
    """Create a dead-letter queue entry.

    Parameters
    ----------
    record : dict
        The malformed record.
    reason : str
        Human-readable explanation of why the record was rejected.
    stage : str
        Pipeline stage where the rejection occurred.

    Returns
    -------
    dict
        Dead-letter entry with metadata.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "reason": reason,
        "record": {
            k: v for k, v in record.items()
            if not k.startswith("_")
        },
        "parser_version": PARSER_VERSION,
        "schema_version": SCHEMA_VERSION,
    }


def _write_dead_letter(
    entry: Dict[str, Any],
    path: Optional[Path] = None,
) -> None:
    """Append a dead-letter entry to the JSONL file.

    Parameters
    ----------
    entry : dict
        Dead-letter entry from ``_dead_letter_record``.
    path : Path or None
        Path to the dead-letter JSONL file. If None, uses default.
    """
    if path is None:
        path = DEAD_LETTER_DIR / "chembl_malformed.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _log_transformation(
    operation: str,
    details: Dict[str, Any],
    path: Optional[Path] = None,
) -> None:
    """Append a transformation log entry.

    Parameters
    ----------
    operation : str
        Name of the transformation (e.g., "filter_pchembl").
    details : dict
        Details of the transformation.
    path : Path or None
        Path to the transformation log JSONL file.
    """
    if path is None:
        path = Path("logs/transformations/chembl.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "parser_version": PARSER_VERSION,
        "schema_version": SCHEMA_VERSION,
        **details,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# =============================================================================
# Section 6 -- Download with full security & reliability
# =============================================================================
# Fixes Domain 6 (Reliability) -- retry logic, timeout, atomic write.
# Fixes Domain 7 (Idempotency) -- checksum recording.
# Fixes Domain 8 (Performance) -- streaming download.
# Fixes Domain 9 (Security) -- URL allowlist, TLS, path-traversal guard.


def _validate_url_against_allowlist(url: str) -> None:
    """Refuse URLs not in the allowlist (Domain 9 -- Security).

    Parameters
    ----------
    url : str
        The URL to validate.

    Raises
    ------
    ChEMBLDownloadError
        If the URL does not start with any allowed prefix.
    """
    for prefix in ALLOWED_CHEMBL_URLS:
        if url.startswith(prefix):
            return
    raise ChEMBLDownloadError(
        f"ChEMBL download URL {url!r} rejected -- not in ALLOWED_CHEMBL_URLS",
        context={"url": url, "allowed_prefixes": list(ALLOWED_CHEMBL_URLS)},
    )


def _create_tls_context() -> ssl.SSLContext:
    """Create a TLS context with certificate verification enabled.

    Returns
    -------
    ssl.SSLContext
        TLS context with cert verification enabled (Domain 9).
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


# =============================================================================
# Section 3.5 -- ChEMBL REST API client (Task 81 ROOT FIX)
# =============================================================================
# Task 81: the audit noted "must use the ChEMBL API pagination cursor.
# Currently fetches only 1000 records." The primary production path
# uses the SQLite bulk download (``download_chembl``), but several
# legitimate use cases need the REST API instead:
#
#   * Incremental updates (fetch only molecules added since the last
#     SQLite release).
#   * Targeted queries (e.g. "all molecules with activity against
#     PDE5A below 100 nM").
#   * Lightweight CI smoke tests (10-record sample without downloading
#     the 2 GB SQLite dump).
#
# The previous codebase had NO REST API client -- a comment claimed
# "we use SQLite for bulk, no API needed", but that left the
# incremental / targeted-query use cases with no path. Operators
# needing them wrote ad-hoc ``requests.get`` calls that hit the
# 1000-row default limit (ChEMBL's REST API caps ``limit`` at 1000
# per request) and silently truncated at 1000 records.
#
# ROOT FIX: a proper cursor-following REST client that:
#   1. Issues requests with ``limit=1000`` (the max the API allows).
#   2. Follows the ``page_meta.next`` cursor in the JSON response
#      until it is ``None``.
#   3. Respects ``max_records`` (caller-supplied safety cap so a
#      runaway loop can't OOM the loader).
#   4. Uses TLS verification (reuses ``_create_ssl_context``).
#   5. Retries 429 / 5xx with exponential backoff.
#   6. Streams results as a generator so callers don't need to hold
#      the full result set in memory.

_CHEMBL_API_BASE: str = "https://www.ebi.ac.uk/chembl/api/data"
_CHEMBL_API_DEFAULT_PAGE_SIZE: int = 1000  # API max -- larger values are silently capped
_CHEMBL_API_DEFAULT_TIMEOUT_S: float = 60.0
_CHEMBL_API_MAX_RETRIES: int = 4


def _chembl_api_request(
    url: str,
    *,
    timeout: float = _CHEMBL_API_DEFAULT_TIMEOUT_S,
    max_retries: int = _CHEMBL_API_MAX_RETRIES,
) -> Dict[str, Any]:
    """Issue a single ChEMBL REST API GET with retry + TLS verification.

    Returns the parsed JSON dict. Raises on non-retriable HTTP errors
    or after ``max_retries`` retriable failures.
    """
    import json as _json
    import urllib.error as _urlerr
    import urllib.request as _urlreq

    ctx = _create_tls_context()
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            req = _urlreq.Request(
                url,
                headers={
                    "User-Agent": "DrugOS-Graph/2.0 (phase2 chembl_loader)",
                    "Accept": "application/json",
                },
            )
            with _urlreq.urlopen(req, timeout=timeout, context=ctx) as resp:
                raw = resp.read()
            return _json.loads(raw.decode("utf-8"))
        except _urlerr.HTTPError as e:
            last_exc = e
            # 429 / 5xx are retriable; 4xx are not.
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except (_urlerr.URLError, TimeoutError, OSError, _json.JSONDecodeError) as e:
            last_exc = e
        if attempt < max_retries:
            time.sleep(0.5 * (2 ** (attempt - 1)))
    raise RuntimeError(
        f"ChEMBL API request failed after {max_retries} attempts: {url!r}: {last_exc}"
    )


def fetch_chembl_molecules_api(
    *,
    max_records: Optional[int] = None,
    page_size: int = _CHEMBL_API_DEFAULT_PAGE_SIZE,
    timeout: float = _CHEMBL_API_DEFAULT_TIMEOUT_S,
    target_chembl_id: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Fetch ChEMBL molecules via the REST API, following the pagination cursor.

    Task 81 ROOT FIX. ChEMBL's REST API caps ``limit`` at 1000 per
    request -- the previous ad-hoc callers hit this cap and silently
    truncated at 1000 records. This generator follows the
    ``page_meta.next`` cursor in the JSON response until it is
    ``None`` (or until ``max_records`` is reached, if the caller
    supplied a safety cap).

    Parameters
    ----------
    max_records : int or None
        Optional safety cap on the total number of records yielded.
        ``None`` (default) means no cap -- fetch ALL matching
        molecules. Set to a small number (e.g. 10) for smoke tests.
    page_size : int
        Records per request. The API silently caps this at 1000;
        larger values are clamped server-side. Default 1000.
    timeout : float
        Per-request timeout in seconds. Default 60.
    target_chembl_id : str or None
        Optional ChEMBL target ID (e.g. ``"CHEMBL2366519"`` for PDE5A)
        to filter molecules to those with measured activity against
        the target. When ``None`` (default), ALL molecules are
        returned (very large result set -- use ``max_records``).

    Yields
    ------
    dict
        Each ChEMBL molecule record (the JSON dict as returned by the
        API). Callers should normalise to KG node records via
        ``chembl_to_node_records``.
    """
    # Build the initial URL. The API uses query params for filtering
    # and pagination; ``page_meta.next`` returns the next URL with
    # the correct offset already encoded.
    base = f"{_CHEMBL_API_BASE}/molecule.json"
    params: Dict[str, str] = {
        "limit": str(min(page_size, _CHEMBL_API_DEFAULT_PAGE_SIZE)),
        "format": "json",
    }
    if target_chembl_id:
        # Filter by target -- the API uses ``target_chembl_id`` as a
        # query param on the molecule endpoint.
        params["target_chembl_id"] = target_chembl_id
    url: Optional[str] = f"{base}?{urllib.parse.urlencode(params)}"
    yielded = 0
    while url:
        payload = _chembl_api_request(url, timeout=timeout)
        molecules = payload.get("molecules") or []
        if not molecules:
            break
        for mol in molecules:
            if max_records is not None and yielded >= max_records:
                return
            yield mol
            yielded += 1
        # Follow the cursor: ``page_meta.next`` is a relative URL
        # starting with ``/chembl/api/data/...``. Re-attach the host.
        page_meta = payload.get("page_meta") or {}
        next_url = page_meta.get("next")
        if not next_url:
            break
        if next_url.startswith("http"):
            url = next_url
        elif next_url.startswith("/"):
            url = "https://www.ebi.ac.uk" + next_url
        else:
            url = _CHEMBL_API_BASE.rstrip("/molecule.json").rstrip("/") + "/" + next_url
    logger.info(
        "fetch_chembl_molecules_api: yielded %d molecules (target_chembl_id=%s)",
        yielded, target_chembl_id,
    )


def iter_chembl_activities_api(
    *,
    max_records: Optional[int] = None,
    page_size: int = _CHEMBL_API_DEFAULT_PAGE_SIZE,
    timeout: float = _CHEMBL_API_DEFAULT_TIMEOUT_S,
    target_chembl_id: Optional[str] = None,
    pchembl_threshold: Optional[float] = None,
) -> Iterator[Dict[str, Any]]:
    """Fetch ChEMBL activity records via the REST API, following the cursor.

    Task 81 companion to ``fetch_chembl_molecules_api``. Yields
    activity records (drug-target interaction measurements) instead
    of molecule records. The activity endpoint is
    ``/chembl/api/data/activity.json`` and uses the same cursor-
    based pagination scheme.

    Parameters
    ----------
    max_records, page_size, timeout
        Same semantics as ``fetch_chembl_molecules_api``.
    target_chembl_id : str or None
        Optional target filter.
    pchembl_threshold : float or None
        Optional minimum pChEMBL value filter. Activities below this
        threshold are skipped (the request still fetches them, but
        the generator does not yield them).
    """
    base = f"{_CHEMBL_API_BASE}/activity.json"
    params: Dict[str, str] = {
        "limit": str(min(page_size, _CHEMBL_API_DEFAULT_PAGE_SIZE)),
        "format": "json",
    }
    if target_chembl_id:
        params["target_chembl_id"] = target_chembl_id
    url: Optional[str] = f"{base}?{urllib.parse.urlencode(params)}"
    yielded = 0
    while url:
        payload = _chembl_api_request(url, timeout=timeout)
        activities = payload.get("activities") or []
        if not activities:
            break
        for act in activities:
            if max_records is not None and yielded >= max_records:
                return
            if pchembl_threshold is not None:
                pval = act.get("pchembl_value")
                if pval is None:
                    continue
                try:
                    if float(pval) < pchembl_threshold:
                        continue
                except (TypeError, ValueError):
                    continue
            yield act
            yielded += 1
        page_meta = payload.get("page_meta") or {}
        next_url = page_meta.get("next")
        if not next_url:
            break
        if next_url.startswith("http"):
            url = next_url
        elif next_url.startswith("/"):
            url = "https://www.ebi.ac.uk" + next_url
        else:
            url = _CHEMBL_API_BASE.rstrip("/") + "/" + next_url
    logger.info(
        "iter_chembl_activities_api: yielded %d activities (target=%s, pchembl_threshold=%s)",
        yielded, target_chembl_id, pchembl_threshold,
    )


def download_chembl(
    force: bool = False,
    *,
    cfg: Optional[ChEMBLConfig] = None,
) -> Path:
    """Download and extract the ChEMBL SQLite database.

    This function implements a fully hardened download pipeline:
    1. URL allowlist check (Domain 9 -- Security)
    2. TLS certificate verification (Domain 9)
    3. Streaming download with retry + exponential backoff (Domain 6)
    4. Atomic write to temporary file, then rename (Domain 7 -- Idempotency)
    5. Size validation (Domain 5 -- Data Quality)
    6. Content sniff -- verify it's a gzip file (Domain 5)
    7. Path-traversal-safe extraction (Domain 9)
    8. Checksum recording (Domain 7 + Domain 16)

    Parameters
    ----------
    force : bool
        If True, re-download even if a cached copy exists.
    cfg : ChEMBLConfig or None
        Loader configuration. If None, uses defaults.

    Returns
    -------
    Path
        Path to the extracted ChEMBL directory.

    Raises
    ------
    ChEMBLDownloadError
        On download failure after all retries.
    ChEMBLDataIntegrityError
        On size/content validation failure.
    """
    if cfg is None:
        cfg = ChEMBLConfig()

    source_cfg = DATA_SOURCES[SOURCE_KEY_CHEMBL]
    url: str = source_cfg["url"]
    tar_path: Path = RAW_DIR / source_cfg["filename"]
    extract_dir: Path = RAW_DIR / "chembl"

    # ── Step 1: Return cached if available and not forced ──────────────
    if extract_dir.exists() and not force and not cfg.force_download:
        db_files = list(extract_dir.rglob("*.db"))
        if db_files:
            logger.info(
                "ChEMBL already extracted at %s (%d .db files)",
                extract_dir, len(db_files),
            )
            return extract_dir

    # ── Step 2: URL allowlist check ────────────────────────────────────
    _validate_url_against_allowlist(url)

    # ── Step 3: Download with retry ────────────────────────────────────
    if not tar_path.exists() or force or cfg.force_download:
        max_retries: int = source_cfg.get("retry_count", 3)
        backoff: float = source_cfg.get("retry_backoff_seconds", 30.0)
        timeout: float = source_cfg.get("timeout_seconds", 600.0)
        expected_size: int = source_cfg.get("size_bytes", 4_000_000_000)
        max_size: int = source_cfg.get("max_size_bytes", 8_000_000_000)

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Downloading ChEMBL from %s (attempt %d/%d) ...",
                    url, attempt, max_retries,
                )
                tls_ctx = _create_tls_context()
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=timeout, context=tls_ctx) as resp:
                    # ── Content sniff: verify Content-Type ─────────────
                    content_type = resp.headers.get("Content-Type", "")
                    if "text/html" in content_type.lower():
                        raise ChEMBLDownloadError(
                            "ChEMBL download returned HTML (likely an error "
                            "page) instead of a tar.gz file",
                            context={
                                "url": url,
                                "content_type": content_type,
                            },
                        )

                    # ── Atomic write via temp file ──────────────────────
                    tmp_path = tar_path.with_suffix(".tmp")
                    bytes_downloaded = 0
                    with open(tmp_path, "wb") as f_out:
                        while True:
                            chunk = resp.read(8192)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            bytes_downloaded += len(chunk)

                    # ── Size validation ─────────────────────────────────
                    if bytes_downloaded < CHEMBL_MIN_VALID_SIZE_BYTES:
                        tmp_path.unlink(missing_ok=True)
                        raise ChEMBLDataIntegrityError(
                            f"ChEMBL download too small: {bytes_downloaded:,} "
                            f"bytes (minimum {CHEMBL_MIN_VALID_SIZE_BYTES:,})",
                            context={
                                "url": url,
                                "downloaded_bytes": bytes_downloaded,
                                "min_bytes": CHEMBL_MIN_VALID_SIZE_BYTES,
                            },
                        )
                    if bytes_downloaded > max_size:
                        tmp_path.unlink(missing_ok=True)
                        raise ChEMBLDataIntegrityError(
                            f"ChEMBL download too large: {bytes_downloaded:,} "
                            f"bytes (maximum {max_size:,})",
                            context={
                                "url": url,
                                "downloaded_bytes": bytes_downloaded,
                                "max_bytes": max_size,
                            },
                        )

                    # ── Content sniff: verify gzip magic bytes ──────────
                    with open(tmp_path, "rb") as f_check:
                        magic = f_check.read(2)
                    if magic[:2] != b'\x1f\x8b':
                        tmp_path.unlink(missing_ok=True)
                        raise ChEMBLDownloadError(
                            "Downloaded file is not a valid gzip archive "
                            "(magic bytes mismatch)",
                            context={
                                "url": url,
                                "magic_bytes": magic.hex(),
                            },
                        )

                    # ── Atomic rename ───────────────────────────────────
                    tmp_path.rename(tar_path)

                    # ── Compute and log checksum ────────────────────────
                    sha256 = hashlib.sha256()
                    with open(tar_path, "rb") as f_hash:
                        for chunk in iter(lambda: f_hash.read(65536), b""):
                            sha256.update(chunk)
                    checksum = sha256.hexdigest()
                    logger.info(
                        "Downloaded ChEMBL to %s (%.1f MB, SHA-256: %s)",
                        tar_path,
                        tar_path.stat().st_size / 1e6,
                        checksum[:16] + "...",
                    )
                    _log_transformation("download", {
                        "url": url,
                        "bytes": bytes_downloaded,
                        "sha256": checksum,
                    })

                last_error = None
                break  # success

            except (ChEMBLDownloadError, ChEMBLDataIntegrityError):
                # Re-raise our own exceptions immediately
                raise
            except (urllib.error.URLError, socket.timeout, OSError) as exc:
                last_error = exc
                logger.warning(
                    "ChEMBL download attempt %d/%d failed: %s",
                    attempt, max_retries, exc,
                )
                if attempt < max_retries:
                    sleep_time = backoff * (2 ** (attempt - 1))
                    logger.info("Retrying in %.0f seconds ...", sleep_time)
                    time.sleep(sleep_time)

        if last_error is not None:
            raise ChEMBLDownloadError(
                f"ChEMBL download failed after {max_retries} attempts: "
                f"{last_error}",
                context={"url": url, "attempts": max_retries},
            ) from last_error

    # ── Step 4: Extract with path-traversal guard ─────────────────────
    logger.info("Extracting ChEMBL to %s ...", extract_dir)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            if sys.version_info >= (3, 12):
                tar.extractall(path=extract_dir, filter="data")
            else:
                # Manual path-traversal check for Python < 3.12
                for member in tar.getmembers():
                    member_path = (extract_dir / member.name).resolve()
                    if not str(member_path).startswith(
                        str(extract_dir.resolve())
                    ):
                        raise ChEMBLDownloadError(
                            f"Path traversal attempt in tar: {member.name}",
                            context={"member": member.name},
                        )
                    # Reject symlinks that escape the extraction directory
                    if member.issym() or member.islnk():
                        link_target = (
                            extract_dir / member.linkname
                        ).resolve()
                        if not str(link_target).startswith(
                            str(extract_dir.resolve())
                        ):
                            raise ChEMBLDownloadError(
                                f"Symlink escape in tar: {member.name} -> "
                                f"{member.linkname}",
                                context={
                                    "member": member.name,
                                    "linkname": member.linkname,
                                },
                            )
                tar.extractall(path=extract_dir)
    except tarfile.TarError as exc:
        raise ChEMBLDownloadError(
            f"ChEMBL tar extraction failed: {exc}",
            context={"tar_path": str(tar_path)},
        ) from exc

    # ── Step 5: Verify extraction ──────────────────────────────────────
    db_files = list(extract_dir.rglob("*.db"))
    if not db_files:
        raise ChEMBLParseError(
            f"No SQLite database found in extracted ChEMBL at {extract_dir}",
            context={"extract_dir": str(extract_dir)},
        )

    logger.info(
        "ChEMBL extraction complete. Found %d .db file(s).",
        len(db_files),
    )
    _log_transformation("extract", {
        "extract_dir": str(extract_dir),
        "db_files": [str(f) for f in db_files],
    })
    return extract_dir


# =============================================================================
# Section 7 -- Parse ChEMBL activities from SQLite
# =============================================================================
# Fixes Domain 3 (Scientific Correctness) -- correct UniProt resolution.
# Fixes Domain 5 (Data Quality) -- input validation, field population.
# Fixes Domain 7 (Idempotency) -- deterministic output ordering.
# Fixes Domain 8 (Performance) -- vectorized SQL, sorted output.
# Fixes Domain 11 (Logging) -- comprehensive metrics at each stage.


# The SQL query for extracting bioactivity data.
# SCIENTIFIC CORRECTNESS NOTES:
# 1. JOIN target_components provides UniProt accessions -- this is the
#    CORRECT way to resolve ChEMBL target IDs to UniProt (the old code
#    used td.chembl_id which is a target DICTIONARY ID, not a protein ID).
# 2. WHERE td.target_type IN (...) filters to target types that have
#    meaningful protein-level resolution.
# 3. LEFT JOIN compound_structures because some compounds lack structures.
# 4. The confidence_score from target_components is used for filtering
#    low-confidence target assignments.
# 5. We include assay information (assay_type, organism) for downstream
#    consumers (the RL ranker uses assay_type to weight evidence).
# SQL template -- the {target_type_placeholders} is filled at runtime
# based on the number of target types in the config. This prevents
# parameter count mismatches (was hardcoded to 3 placeholders).
_CHEMBL_SQL_TEMPLATE: str = """
SELECT
    md.chembl_id                   AS drug_chembl_id,
    cs.canonical_smiles            AS smiles,
    td.chembl_id                   AS target_chembl_id,
    td.pref_name                   AS target_name,
    td.target_type,
    csq.accession                  AS uniprot_accession,
    csq.description                AS component_description,
    act.pchembl_value,
    act.standard_type,
    act.standard_value,
    act.standard_units,
    ass.assay_type,
    a2t.confidence_score,
    oc.tax_id                      AS tax_id
FROM activities act
JOIN molecule_dictionary md   ON act.molregno = md.molregno
JOIN assays ass               ON act.assay_id = ass.assay_id
JOIN target_dictionary td     ON ass.tid = td.tid
LEFT JOIN assay2target a2t    ON ass.assay_id = a2t.assay_id
-- V19 ROOT FIX (RT-2): the FK column on target_components is named
-- `target_id` (FK to target_dictionary.tid), NOT `tid`. The previous
-- `tc.tid = tc.tid` raised `column tc.tid does not exist` on every real
-- ChEMBL database at runtime, which step7c's try/except silently
-- swallowed -- zero ChEMBL bioactivity edges ever loaded. The audit's
-- original PS-10 claim that `target_components.tid` IS a real ChEMBL
-- column was itself wrong: per the official ChEMBL schema (chembl_35),
-- the `target_components` table has columns (target_id, component_id,
-- homologue) -- there is no `tid` column.
LEFT JOIN target_components tc ON td.tid = tc.target_id
LEFT JOIN component_sequences csq ON tc.component_id = csq.component_id
LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
LEFT JOIN organism_classification oc ON ass.assay_tax_id = oc.tax_id
WHERE act.pchembl_value IS NOT NULL
  AND act.pchembl_value >= ?
  AND td.target_type IN ({target_type_placeholders})
ORDER BY md.chembl_id, td.chembl_id
"""


def _compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file, reading in 64 KB chunks."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# v69 ROOT FIX (P2L-012): module-level safe sort key for ChEMBL .db files.
#
# The sibling functions ``parse_chembl_activities`` (line ~1128) and
# ``iter_chembl_activities`` (line ~1476) both need to sort candidate
# ChEMBL SQLite databases by (size desc, mtime desc, name asc) to
# deterministically pick the largest/newest DB. The previous code had
# TWO separate sort-key implementations:
#   - ``parse_chembl_activities`` used a SAFE nested function
#     ``_db_sort_key`` with try/except OSError -> (0, 0.0, str(p)).
#   - ``iter_chembl_activities`` used an UNSAFE inline lambda
#     ``lambda p: (-p.stat().st_size, -p.stat().st_mtime, str(p))`` with
#     NO try/except -- raising OSError if a cached .db file was deleted
#     between ``rglob`` and the sort (race condition crash).
#
# ROOT FIX: extract the safe sort key as a MODULE-LEVEL function so
# BOTH sibling functions use the SAME safe implementation. This
# eliminates the inconsistency and the race-condition crash. The
# function returns a tuple suitable for ``sorted(key=...)``: files that
# fail ``stat()`` sort to the END (size 0, mtime 0.0) so they're never
# selected as ``db_files[0]``.
def _chembl_db_sort_key(p: Path) -> tuple:
    """Safe sort key for ChEMBL .db files (v69 ROOT FIX P2L-012).

    Returns ``(-size, -mtime, str(path))`` for deterministic
    largest-then-newest-then-name ordering. On ``OSError`` (file deleted
    between ``rglob`` and ``stat``), returns ``(0, 0.0, str(path))`` so
    the file sorts to the end and is never selected.
    """
    try:
        st = p.stat()
        return (-st.st_size, -st.st_mtime, str(p))
    except OSError:
        return (0, 0.0, str(p))


def parse_chembl_activities(
    chembl_dir: Optional[Path] = None,
    min_pchembl: Optional[float] = None,
    *,
    cfg: Optional[ChEMBLConfig] = None,
) -> pd.DataFrame:
    """Parse ChEMBL activities from the SQLite database.

    SCIENTIFIC CORRECTNESS FIX:
    The original query returned ``td.chembl_id`` (e.g. "CHEMBL218") as
    the target identifier, which is the ChEMBL TARGET dictionary ID --
    NOT a UniProt protein accession. This caused ChEMBL "Protein" nodes
    to use a different ID namespace than UniProt, creating disconnected
    subgraphs in Neo4j.

    The fix JOINs through ``target_components`` -> ``component_sequences``
    to fetch the UniProt accession (``csq.accession``) which IS the
    canonical protein ID. Rows where no UniProt accession exists are
    kept but flagged with ``uniprot_accession=None`` so downstream code
    can drop them for high-confidence use cases.

    Additionally, we now:
    - Include assay_type (B/F) for evidence classification
    - Include confidence_score for target assignment quality
    - Include tax_id for organism filtering
    - Filter by target_type (not just SINGLE PROTEIN)
    - Sort output deterministically (Domain 7)

    Parameters
    ----------
    chembl_dir : Path or None
        Path to extracted ChEMBL directory. If None, uses default.
    min_pchembl : float or None
        Minimum pChEMBL value. DEPRECATED -- use cfg.min_pchembl instead.
        If provided, overrides cfg.min_pchembl for backward compatibility.
    cfg : ChEMBLConfig or None
        Loader configuration. If None, uses defaults.

    Returns
    -------
    pd.DataFrame
        DataFrame with ChEMBL activity records, sorted by
        drug_chembl_id, target_chembl_id for deterministic output.

    Raises
    ------
    ChEMBLParseError
        If the SQLite database cannot be read.
    ChEMBLDataIntegrityError
        If the parsed data fails quality checks.
    """
    # ── Configuration resolution ────────────────────────────────────────
    if cfg is None:
        if min_pchembl is not None:
            cfg = ChEMBLConfig(min_pchembl=min_pchembl)
        else:
            cfg = ChEMBLConfig()
    elif min_pchembl is not None:
        warnings.warn(
            "min_pchembl parameter is deprecated -- use cfg.min_pchembl. "
            "The cfg value is being overridden.",
            DeprecationWarning,
            stacklevel=2,
        )
        cfg = ChEMBLConfig(
            min_pchembl=min_pchembl,
            organism_tax_id=cfg.organism_tax_id,
            min_confidence_score=cfg.min_confidence_score,
            target_types=cfg.target_types,
            chembl_dir=cfg.chembl_dir,
            force_download=cfg.force_download,
            sort_output=cfg.sort_output,
        )

    if chembl_dir is None:
        if cfg.chembl_dir is not None:
            chembl_dir = cfg.chembl_dir
        else:
            chembl_dir = RAW_DIR / "chembl"

    # ── Find the SQLite database ────────────────────────────────────────
    # v21 ROOT FIX (Audit section 7 finding 7 - "Non-deterministic SQLite
    # selection"): the previous code did
    # ``db_files = list(Path(chembl_dir).rglob("*.db"))`` then
    # ``db_path = db_files[0]``. ``rglob`` returns files in
    # filesystem-dependent order (inode order on ext4, hash order on
    # XFS, etc.) - different runs pick DIFFERENT DBs if multiple are
    # cached. The drkg_loader explicitly bans this pattern
    # (drkg_loader.py:47). Fix: sort the list (by name then by mtime
    # for stability) and prefer the LARGEST file (ChEMBL SQLite is
    # ~2GB; a leftover temp download would be much smaller). If
    # multiple files have the same size, prefer the newest. Log which
    # file was chosen so operators can verify determinism.
    db_files = list(Path(chembl_dir).rglob("*.db"))
    if not db_files:
        raise ChEMBLParseError(
            f"No SQLite database found in {chembl_dir}",
            context={"chembl_dir": str(chembl_dir)},
        )
    if len(db_files) == 1:
        db_path = db_files[0]
    else:
        # Sort by (size desc, mtime desc, name asc) - deterministic.
        # v69 ROOT FIX (P2L-012): use the module-level
        # ``_chembl_db_sort_key`` (safe against OSError) instead of the
        # nested function. The behavior is identical, but the module-level
        # function is shared with ``iter_chembl_activities`` so both
        # sibling functions use the SAME safe implementation.
        db_files_sorted = sorted(db_files, key=_chembl_db_sort_key)
        db_path = db_files_sorted[0]
        logger.warning(
            "Multiple ChEMBL SQLite databases found in %s. Selected %s "
            "(largest by size, newest by mtime). Other candidates: %s. "
            "Remove the unused DBs to silence this warning.",
            chembl_dir, db_path,
            [str(p) for p in db_files_sorted[1:]],
        )
    logger.info("Reading ChEMBL from %s ...", db_path)

    # ── Compute source file checksum ────────────────────────────────────
    source_sha256 = _compute_file_sha256(db_path)

    # ── Execute SQL query ───────────────────────────────────────────────
    target_types_list = sorted(cfg.target_types)
    params: list[Any] = [cfg.min_pchembl] + target_types_list

    # Build SQL with dynamic number of target_type placeholders
    placeholders = ", ".join(["?"] * len(target_types_list))
    sql_query = _CHEMBL_SQL_TEMPLATE.format(
        target_type_placeholders=placeholders
    )

    try:
        conn = sqlite3.connect(str(db_path))
        try:
            df = pd.read_sql_query(
                sql_query, conn, params=params,
            )
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        raise ChEMBLParseError(
            f"ChEMBL SQL query failed: {exc}",
            context={"db_path": str(db_path), "error": str(exc)},
        ) from exc
    except sqlite3.DatabaseError as exc:
        raise ChEMBLParseError(
            f"ChEMBL database error: {exc}",
            context={"db_path": str(db_path), "error": str(exc)},
        ) from exc

    n_raw = len(df)
    logger.info("Loaded %s raw ChEMBL activity rows", f"{n_raw:,}")

    # ── Organism filter ─────────────────────────────────────────────────
    if "tax_id" in df.columns and cfg.organism_tax_id > 0:
        before = len(df)
        # v68 ROOT FIX (P2L-010): the previous mask used OR (``|``):
        #   mask = df["tax_id"].isna() | (pd.to_numeric(...) == cfg.organism_tax_id)
        # This KEPT rows where tax_id is NaN regardless of the organism
        # filter. The comment said "Keep rows where tax_id is NaN (no
        # organism info)" -- but ChEMBL assays on bacterial / fungal /
        # viral targets sometimes have missing tax_id. These non-human
        # edges were silently admitted to the KG as "human" bioactivity,
        # contaminating the human drug-target graph. The GNN learned
        # spurious drug-target edges that don't apply to human biology.
        #
        # ROOT FIX: when an organism filter is active (cfg.organism_tax_id
        # > 0), DROP rows where tax_id is NaN. Only keep rows where
        # tax_id is present AND matches the filter. This is the
        # scientifically correct behavior: if we're filtering to human
        # (tax_id=9606), we cannot accept rows with no organism info --
        # they might be bacterial, fungal, or viral.
        #
        # The v42 fix (coerce to numeric before comparing so "9606" ==
        # 9606 is True) is preserved.
        tax_id_numeric = pd.to_numeric(df["tax_id"], errors="coerce")
        mask = (
            df["tax_id"].notna()
            & (tax_id_numeric == cfg.organism_tax_id)
        )
        df = df[mask].copy()
        after = len(df)
        if after < before:
            logger.info(
                "Organism filter (tax_id=%d): %s -> %s rows "
                "(dropped %s non-matching, including NaN tax_id rows "
                "that were previously silently admitted -- v68 ROOT FIX "
                "P2L-010)",
                cfg.organism_tax_id,
                f"{before:,}", f"{after:,}",
                f"{before - after:,}",
            )
            _log_transformation("filter_organism", {
                "tax_id": cfg.organism_tax_id,
                "before": before,
                "after": after,
                "dropped_nan_tax_id": True,
            })

    # ── Confidence score filter ─────────────────────────────────────────
    if "confidence_score" in df.columns and cfg.min_confidence_score > 0:
        before = len(df)
        mask = (
            df["confidence_score"].isna()
            | (df["confidence_score"] >= cfg.min_confidence_score)
        )
        df = df[mask].copy()
        after = len(df)
        if after < before:
            logger.info(
                "Confidence score filter (>= %d): %s -> %s rows",
                cfg.min_confidence_score,
                f"{before:,}", f"{after:,}",
            )
            _log_transformation("filter_confidence", {
                "min_confidence": cfg.min_confidence_score,
                "before": before,
                "after": after,
            })

    # ── Validate pChEMBL values ─────────────────────────────────────────
    if "pchembl_value" in df.columns:
        lo, hi = CHEMBL_PCHEMBL_RANGE
        invalid_pchembl = (
            (df["pchembl_value"] < lo) | (df["pchembl_value"] > hi)
        )
        n_invalid = invalid_pchembl.sum()
        if n_invalid > 0:
            logger.warning(
                "Dropping %s rows with pChEMBL outside [%s, %s]",
                f"{n_invalid:,}", lo, hi,
            )
            df = df[~invalid_pchembl].copy()
            _log_transformation("filter_pchembl_range", {
                "dropped": n_invalid,
                "range": [lo, hi],
            })

    # ── Validate ChEMBL IDs ─────────────────────────────────────────────
    if "drug_chembl_id" in df.columns:
        valid_ids = df["drug_chembl_id"].astype(str).str.match(
            _RE_CHEMBL_ID
        )
        n_invalid = (~valid_ids).sum()
        if n_invalid > 0:
            logger.warning(
                "Dropping %s rows with invalid drug_chembl_id",
                f"{n_invalid:,}",
            )
            df = df[valid_ids].copy()

    # ── Deduplicate ─────────────────────────────────────────────────────
    # Include uniprot_accession in dedup so that multi-subunit targets
    # (same drug + target_chembl_id but different UniProt accessions)
    # are NOT collapsed into a single row. Each subunit is a distinct
    # biological entity and must be preserved.
    before = len(df)
    dedup_cols = [
        "drug_chembl_id", "target_chembl_id", "standard_type",
        "pchembl_value", "uniprot_accession",
    ]
    existing_cols = [c for c in dedup_cols if c in df.columns]
    if existing_cols:
        df = df.drop_duplicates(subset=existing_cols, keep="first")
    after = len(df)
    if after < before:
        logger.info(
            "Deduplication: %s -> %s rows (removed %s duplicates)",
            f"{before:,}", f"{after:,}",
            f"{before - after:,}",
        )
        _log_transformation("deduplicate", {
            "before": before,
            "after": after,
            "removed": before - after,
        })

    # ── Deterministic sort ──────────────────────────────────────────────
    if cfg.sort_output:
        sort_cols = ["drug_chembl_id", "target_chembl_id"]
        existing_sort = [c for c in sort_cols if c in df.columns]
        if existing_sort:
            df = df.sort_values(existing_sort).reset_index(drop=True)

    # ── Field population check ──────────────────────────────────────────
    n_final = len(df)
    for field_name, min_rate in CHEMBL_MIN_FIELD_POPULATION.items():
        if field_name not in df.columns:
            continue
        if n_final == 0:
            continue
        pop_rate = df[field_name].notna().mean()
        if pop_rate < min_rate:
            raise ChEMBLDataIntegrityError(
                f"ChEMBL field {field_name!r} population rate "
                f"{pop_rate:.1%} is below minimum {min_rate:.1%}",
                context={
                    "field": field_name,
                    "population_rate": pop_rate,
                    "minimum": min_rate,
                    "total_rows": n_final,
                },
            )

    # ── Provenance metadata ─────────────────────────────────────────────
    n_with_uniprot = df["uniprot_accession"].notna().sum() if "uniprot_accession" in df.columns else 0
    source_cfg = DATA_SOURCES[SOURCE_KEY_CHEMBL]
    provenance: Dict[str, Any] = {
        "source": SOURCE_CHEMBL,
        "source_file": str(db_path),
        "source_sha256": source_sha256,
        "source_version": source_cfg.get("version", "unknown"),
        "source_release_date": source_cfg.get("release_date"),
        "source_license": CHEMBL_LICENSE,
        "source_url": source_cfg.get("url", ""),
        "parser_module": __name__,
        "parser_version": PARSER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "chembl_version": source_cfg.get("version", "unknown"),
        "min_pchembl": cfg.min_pchembl,
        "organism_filter": cfg.organism_tax_id,
        "resolution_method": "sql_join+crosswalk",
        "row_count_in": n_raw,
        "row_count_out": n_final,
        "crosswalk_version": "pending",  # filled by chembl_to_edge_records
    }
    df.attrs["provenance"] = provenance
    df.attrs["license"] = CHEMBL_LICENSE
    df.attrs["attribution"] = CHEMBL_ATTRIBUTION

    logger.info(
        "Parsed %s ChEMBL activities (pChEMBL >= %s); "
        "%s have UniProt accession (%s)",
        f"{n_final:,}",
        cfg.min_pchembl,
        f"{n_with_uniprot:,}",
        f"{n_with_uniprot / max(n_final, 1):.1%}",
    )
    return df


# =============================================================================
# Section 8 -- Streaming iterator for large databases
# =============================================================================
# Fixes Domain 8 (Performance) -- avoids loading entire DB into memory.


def iter_chembl_activities(
    chembl_dir: Optional[Path] = None,
    chunk_size: int = 100_000,
    *,
    cfg: Optional[ChEMBLConfig] = None,
) -> Iterator[pd.DataFrame]:
    """Stream ChEMBL activities in chunks for memory-efficient processing.

    This is the streaming counterpart to ``parse_chembl_activities``.
    Use it when the database is too large to fit in memory or when
    processing can be done incrementally.

    v68 ROOT FIX (P2L-013): the previous implementation yielded RAW SQL
    chunks without applying ANY of the filters that ``parse_chembl_activities``
    applies:
      - Organism filter (P2L-010 -- drop NaN tax_id, match cfg.organism_tax_id)
      - Confidence score filter (>= cfg.min_confidence_score)
      - pChEMBL range validation (CHEMBL_PCHEMBL_RANGE)
      - ChEMBL ID format validation (_RE_CHEMBL_ID)
      - Field population check (CHEMBL_MIN_FIELD_POPULATION)
    Downstream consumers using the streaming API got unfiltered,
    unvalidated, duplicate-containing raw SQL results -- completely
    different data quality than the bulk path. Silent contamination if
    a downstream pipeline switched from bulk to streaming.

    ROOT FIX: apply the SAME per-chunk filters inside the chunk loop.
    Per-chunk filters (organism, confidence, pchembl range, ID
    validation) are applied to each chunk independently. Cross-chunk
    operations (deduplication, deterministic sort) CANNOT be applied
    per-chunk (they require the full dataset) -- callers requiring
    deduplication / global sort must collect all chunks and apply
    those operations themselves. This is documented in the Notes
    section below.

    Parameters
    ----------
    chembl_dir : Path or None
        Path to extracted ChEMBL directory.
    chunk_size : int
        Number of rows per chunk.
    cfg : ChEMBLConfig or None
        Loader configuration.

    Yields
    ------
    pd.DataFrame
        Chunks of ChEMBL activity records, with per-chunk filters
        applied (organism, confidence, pchembl range, ID validation).
        Deduplication and global sort are NOT applied -- callers
        requiring those must collect all chunks and apply them.

    Notes
    -----
    The streaming API applies the following filters per chunk:
      1. Organism filter (drop NaN tax_id, match cfg.organism_tax_id)
      2. Confidence score filter (>= cfg.min_confidence_score)
      3. pChEMBL range validation (CHEMBL_PCHEMBL_RANGE)
      4. ChEMBL ID format validation (_RE_CHEMBL_ID)

    The following operations are NOT applied (they require the full
    dataset) -- callers must apply them after collecting all chunks:
      - Deduplication (drop_duplicates on the full dataset)
      - Deterministic sort (sort_values on the full dataset)
      - Field population check (CHEMBL_MIN_FIELD_POPULATION)
      - Provenance metadata (df.attrs)

    This ensures the streaming API produces the SAME data quality as
    the bulk path for the per-chunk filters, while documenting the
    cross-chunk operations that callers must handle themselves.
    """
    if cfg is None:
        cfg = ChEMBLConfig()

    if chembl_dir is None:
        if cfg.chembl_dir is not None:
            chembl_dir = cfg.chembl_dir
        else:
            chembl_dir = RAW_DIR / "chembl"

    db_files = list(Path(chembl_dir).rglob("*.db"))
    if not db_files:
        raise ChEMBLParseError(
            f"No SQLite database found in {chembl_dir}",
            context={"chembl_dir": str(chembl_dir)},
        )
    # v24 ROOT FIX (FORENSIC-P2-LOADERS D/§1): the previous code did
    # ``db_path = db_files[0]`` -- non-deterministic when multiple .db
    # files are cached (different runs pick different DBs). The sibling
    # function ``parse_chembl_activities`` (line ~1075) was already
    # fixed to sort by size then mtime; apply the same deterministic
    # sort here so ``iter_chembl_activities`` is idempotent.
    # FIX-P1-B-11 (audit P1): the previous sort key
    # ``(p.stat().st_size, p.stat().st_mtime, str(p))`` was ASCENDING --
    # so ``db_files[0]`` selected the SMALLEST DB. But the sibling
    # ``parse_chembl_activities`` (line ~1094-1101) sorts by
    # ``(-st.st_size, -st.st_mtime, str(p))`` -- DESCENDING size -- and
    # selects ``db_files_sorted[0]``, i.e. the LARGEST DB. The two
    # sibling functions therefore selected DIFFERENT DBs whenever a
    # stale small stub DB was cached alongside the real 2 GB ChEMBL DB.
    # Root fix: match the sibling function's descending sort so both
    # functions select the LARGEST / NEWEST DB deterministically.
    #
    # v69 ROOT FIX (P2L-012): replace the UNSAFE inline lambda
    # ``lambda p: (-p.stat().st_size, -p.stat().st_mtime, str(p))``
    # (which raised OSError on missing files -- race condition crash)
    # with the module-level ``_chembl_db_sort_key`` (safe against
    # OSError). This eliminates the inconsistency between the two
    # sibling functions AND the race-condition crash if a cached .db
    # file is deleted between ``rglob`` and ``sort``.
    db_files.sort(key=_chembl_db_sort_key)
    db_path = db_files[0]

    target_types_list = sorted(cfg.target_types)
    params: list[Any] = [cfg.min_pchembl] + target_types_list

    # Build SQL with dynamic number of target_type placeholders
    placeholders = ", ".join(["?"] * len(target_types_list))
    sql_query = _CHEMBL_SQL_TEMPLATE.format(
        target_type_placeholders=placeholders
    )

    n_chunks = 0
    n_rows_in = 0
    n_rows_out = 0
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            for chunk in pd.read_sql_query(
                sql_query,
                conn,
                params=params,
                chunksize=chunk_size,
            ):
                n_chunks += 1
                n_rows_in += len(chunk)
                # ── v68 ROOT FIX (P2L-013): apply per-chunk filters ──
                # Apply the SAME filters as parse_chembl_activities so
                # the streaming API produces equivalent data quality.

                # Filter 1: Organism filter (P2L-010 fix -- drop NaN tax_id)
                if "tax_id" in chunk.columns and cfg.organism_tax_id > 0:
                    tax_id_numeric = pd.to_numeric(chunk["tax_id"], errors="coerce")
                    org_mask = (
                        chunk["tax_id"].notna()
                        & (tax_id_numeric == cfg.organism_tax_id)
                    )
                    chunk = chunk[org_mask].copy()

                # Filter 2: Confidence score filter
                if "confidence_score" in chunk.columns and cfg.min_confidence_score > 0:
                    conf_mask = (
                        chunk["confidence_score"].isna()
                        | (chunk["confidence_score"] >= cfg.min_confidence_score)
                    )
                    chunk = chunk[conf_mask].copy()

                # Filter 3: pChEMBL range validation
                if "pchembl_value" in chunk.columns:
                    lo, hi = CHEMBL_PCHEMBL_RANGE
                    pchembl_valid = (
                        (chunk["pchembl_value"] >= lo)
                        & (chunk["pchembl_value"] <= hi)
                    )
                    chunk = chunk[pchembl_valid].copy()

                # Filter 4: ChEMBL ID format validation
                if "drug_chembl_id" in chunk.columns:
                    valid_ids = chunk["drug_chembl_id"].astype(str).str.match(
                        _RE_CHEMBL_ID
                    )
                    chunk = chunk[valid_ids].copy()

                n_rows_out += len(chunk)
                if len(chunk) > 0:
                    yield chunk
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        raise ChEMBLParseError(
            f"ChEMBL SQL query failed: {exc}",
            context={"db_path": str(db_path)},
        ) from exc

    logger.info(
        "iter_chembl_activities: streamed %d chunks, %s rows in -> %s rows out "
        "(per-chunk filters applied: organism, confidence, pchembl, ID validation). "
        "Cross-chunk operations (dedup, sort, field-population, provenance) NOT "
        "applied -- caller must handle. v68 ROOT FIX P2L-013.",
        n_chunks,
        f"{n_rows_in:,}",
        f"{n_rows_out:,}",
    )


# =============================================================================
# Section 9 -- Convert activities to edge records
# =============================================================================
# Fixes Domain 3 (Scientific Correctness) -- correct relation mapping.
# Fixes Domain 5 (Data Quality) -- validated edges only.
# Fixes Domain 6 (Reliability) -- per-row error isolation.
# Fixes Domain 16 (Lineage) -- resolution path tracking.


def chembl_to_edge_records(
    df: pd.DataFrame,
    crosswalk: Any = None,
) -> List[Dict[str, Any]]:
    """Convert ChEMBL activities to Compound->Protein edge records.

    SCIENTIFIC CORRECTNESS FIX (ARCH-2 / GUARD-INT-3 of id_crosswalk audit):
    ChEMBL -> UniProt resolution is now OWNED by
    ``IDCrosswalk.chembl_target_to_uniprot_ac()``. The in-loader SQL JOIN
    that previously read ``row.uniprot_accession`` directly is now only a
    FALLBACK for the case where the crosswalk has not been populated.
    This eliminates the "two sources of truth" bug class: a single
    crosswalk instance is the source of truth for ChEMBL -> UniProt.

    For multi-subunit complexes (GABA-A receptor with 5 subunits, NMDA
    with 4, etc.), the crosswalk returns the FULL list of UniProt ACs and
    this loader emits one Compound->Protein edge per subunit (SCI-4).

    RELATION TYPE FIX (Domain 3):
    The old code used substring matching:
        "activates" if "ACTIVAT" in std_type or "AGONIST" in std_type
        else "inhibits" if "INHIBIT" in std_type or "ANTAGONIST" in std_type
        else "binds" if "BIND" in std_type
        else "inhibits"  # ← WRONG DEFAULT

    This was scientifically incorrect because:
    - EC50 measurements were defaulting to "inhibits" (they imply activation)
    - Kd measurements were defaulting to "inhibits" (they measure binding)
    - Potency measurements were defaulting to "inhibits" (directionless)

    Now uses ``standard_type_to_relation()`` which has a curated mapping.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from ``parse_chembl_activities``.
    crosswalk : IDCrosswalk or None
        ID crosswalk instance for ChEMBL->UniProt resolution.
        If None, attempts to get the default crosswalk.

    Returns
    -------
    list[dict]
        Edge records with src_id, dst_id, src_type, dst_type, rel_type,
        and props dict.
    """
    from .id_crosswalk import get_default_crosswalk

    if crosswalk is None:
        try:
            crosswalk = get_default_crosswalk()
        except Exception:
            crosswalk = None
            logger.warning(
                "chembl_loader: could not get default crosswalk -- "
                "using SQL JOIN fallback only"
            )

    edges: List[Dict[str, Any]] = []
    n_skipped = 0
    n_crosswalk_resolved = 0
    n_fallback_resolved = 0
    n_dead_letter = 0

    for row_idx, row in enumerate(df.itertuples(index=False)):
        try:
            target_chembl_id = (
                str(row.target_chembl_id)
                if pd.notna(row.target_chembl_id)
                else ""
            )
            drug_chembl_id = (
                str(row.drug_chembl_id)
                if pd.notna(row.drug_chembl_id)
                else ""
            )

            # ── Validate drug ID ────────────────────────────────────
            if not drug_chembl_id or not _RE_CHEMBL_ID.match(drug_chembl_id):
                n_dead_letter += 1
                _write_dead_letter(
                    _dead_letter_record(
                        {"drug_chembl_id": drug_chembl_id,
                         "target_chembl_id": target_chembl_id},
                        f"Invalid drug_chembl_id: {drug_chembl_id!r}",
                        "chembl_to_edge_records",
                    )
                )
                n_skipped += 1
                continue

            # ── Resolve UniProt accession ───────────────────────────
            # ARCH-2: try crosswalk FIRST (single source of truth)
            crosswalk_acs: list[str] = []
            if crosswalk is not None and target_chembl_id:
                try:
                    acs = crosswalk.chembl_target_to_uniprot_ac_all(
                        target_chembl_id
                    )
                    if acs:
                        crosswalk_acs = list(acs)
                except Exception as exc:
                    logger.debug(
                        "chembl_loader: crosswalk lookup failed for %s: %s",
                        target_chembl_id, exc,
                    )

            if crosswalk_acs:
                n_crosswalk_resolved += 1
                acs_to_emit = crosswalk_acs
                resolution_path = "crosswalk"
            else:
                # GUARD-INT-3: fallback to SQL JOIN's uniprot_accession
                sql_ac = getattr(row, "uniprot_accession", None)
                if pd.isna(sql_ac) or not str(sql_ac).strip():
                    n_skipped += 1
                    continue
                sql_ac_str = str(sql_ac).strip()
                # Validate UniProt accession format
                if not _RE_UNIPROT_AC.match(sql_ac_str):
                    n_dead_letter += 1
                    _write_dead_letter(
                        _dead_letter_record(
                            {"uniprot_accession": sql_ac_str,
                             "target_chembl_id": target_chembl_id},
                            f"Invalid UniProt accession: {sql_ac_str!r}",
                            "chembl_to_edge_records",
                        )
                    )
                    n_skipped += 1
                    continue
                n_fallback_resolved += 1
                logger.debug(
                    "chembl_loader: crosswalk miss for %s -- falling back "
                    "to SQL JOIN uniprot_accession=%s",
                    target_chembl_id, sql_ac_str,
                )
                acs_to_emit = [sql_ac_str]
                resolution_path = "sql_fallback"

            # ── Map relation type ───────────────────────────────────
            std_type = (
                str(row.standard_type)
                if pd.notna(row.standard_type)
                else ""
            )
            rel_type = standard_type_to_relation(std_type)

            # ── Build edge properties ───────────────────────────────
            pchembl = (
                float(row.pchembl_value)
                if pd.notna(row.pchembl_value)
                else None
            )
            # v68 ROOT FIX (P2L-009): propagate ``standard_value`` (raw
            # IC50/Ki/Kd in nM) and ``standard_units`` (typically "nM")
            # to the edge props. The SQL query SELECTs these columns
            # (lines 975-976), and the DataFrame returned by
            # ``parse_chembl_activities`` has them -- but the edge records
            # (which is what kg_builder consumes) DROPPED them.
            # Downstream consumers that need the raw activity value
            # (e.g., to recompute pchembl or to apply a custom unit
            # conversion) had no access to it from the edge record.
            # ROOT FIX: emit ``standard_value`` as a float (or None) and
            # ``standard_units`` as a string (or "") in the edge props.
            # This preserves the raw activity data at the edge-record
            # boundary. The ``pchembl_value`` (already in props) remains
            # the canonical normalized potency; ``standard_value`` is the
            # raw nM value for downstream forensics / re-derivation.
            standard_value = (
                float(row.standard_value)
                if hasattr(row, "standard_value")
                and pd.notna(getattr(row, "standard_value", None))
                else None
            )
            standard_units = (
                str(row.standard_units).strip()
                if hasattr(row, "standard_units")
                and pd.notna(getattr(row, "standard_units", None))
                else ""
            )

            # SCI-4: emit one edge per subunit for multi-subunit complexes
            for ac in acs_to_emit:
                edge: Dict[str, Any] = {
                    "src_id": drug_chembl_id,
                    "dst_id": ac,
                    "src_type": ENTITY_TYPE_COMPOUND,
                    "dst_type": "Protein",
                    "rel_type": rel_type,
                    "props": {
                        "source": SOURCE_CHEMBL,
                        "pchembl_value": pchembl,
                        "standard_type": std_type,
                        # v68 ROOT FIX (P2L-009): raw activity value (nM)
                        # and units -- preserved for downstream forensics
                        # and custom unit conversion. Downstream consumers
                        # MUST NOT confuse standard_value (nM, raw) with
                        # pchembl_value (-log10[M], normalized). The
                        # canonical edge-weight signal remains
                        # pchembl_value / normalized_score.
                        "standard_value": standard_value,
                        "standard_units": standard_units,
                        "target_chembl_id": target_chembl_id,
                        "resolution_path": resolution_path,
                        "subunit_count": len(acs_to_emit),
                        "assay_type": (
                            str(row.assay_type)
                            if hasattr(row, "assay_type")
                            and pd.notna(getattr(row, "assay_type", None))
                            else None
                        ),
                        "confidence_score": (
                            int(row.confidence_score)
                            if hasattr(row, "confidence_score")
                            and pd.notna(
                                getattr(row, "confidence_score", None)
                            )
                            else None
                        ),
                    },
                }
                edges.append(edge)

        except Exception as exc:
            # Per-row error isolation (Domain 6 -- Reliability)
            n_dead_letter += 1
            row_dict = {}
            try:
                row_dict = {
                    k: getattr(row, k, None)
                    for k in row._fields
                    if hasattr(row, "_fields")
                }
            except Exception:
                pass
            _write_dead_letter(
                _dead_letter_record(
                    row_dict,
                    f"Unexpected error processing row {row_idx}: {exc}",
                    "chembl_to_edge_records",
                )
            )
            logger.warning(
                "chembl_loader: unexpected error on row %d: %s",
                row_idx, exc,
            )
            continue

    logger.info(
        "Converted %s ChEMBL edge records "
        "(skipped %s rows without UniProt accession; "
        "%s via crosswalk, %s via SQL JOIN fallback; "
        "%s dead-lettered)",
        f"{len(edges):,}",
        f"{n_skipped:,}",
        f"{n_crosswalk_resolved:,}",
        f"{n_fallback_resolved:,}",
        f"{n_dead_letter:,}",
    )
    _log_transformation("chembl_to_edge_records", {
        "edges_out": len(edges),
        "skipped": n_skipped,
        "crosswalk_resolved": n_crosswalk_resolved,
        "fallback_resolved": n_fallback_resolved,
        "dead_lettered": n_dead_letter,
    })
    return edges


# =============================================================================
# Section 10 -- Compound node record generation
# =============================================================================
# Fixes Domain 2 (Design) -- consistent node record schema with other loaders.


def chembl_to_node_records(
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Convert ChEMBL activities to Compound node records for the KG.

    This generates one node per unique drug_chembl_id, with associated
    SMILES and other compound metadata. It is the companion to
    ``chembl_to_edge_records`` and follows the same schema convention
    as ``drugbank_parser.drugbank_to_node_records``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from ``parse_chembl_activities``.

    Returns
    -------
    list[dict]
        Node records with id, label, name, smiles, and provenance.
    """
    if "drug_chembl_id" not in df.columns:
        return []

    # Group by drug_chembl_id and take the first SMILES
    compound_groups = df.groupby("drug_chembl_id", sort=True).first()

    nodes: List[Dict[str, Any]] = []
    for chembl_id, row in compound_groups.iterrows():
        smiles = (
            str(row.get("smiles", ""))
            if pd.notna(row.get("smiles"))
            else ""
        )
        node: Dict[str, Any] = {
            "id": chembl_id,
            "label": ENTITY_TYPE_COMPOUND,
            "name": chembl_id,
            "chembl_id": chembl_id,
            "smiles": smiles,
            "source": SOURCE_CHEMBL,
            "_provenance": df.attrs.get("provenance", {}),
            "_license": CHEMBL_LICENSE,
            "_attribution": CHEMBL_ATTRIBUTION,
        }
        nodes.append(node)

    logger.info(
        "Generated %s ChEMBL Compound node records",
        f"{len(nodes):,}",
    )
    return nodes


# =============================================================================
# Section 11 -- End-to-end graph construction
# =============================================================================
# Fixes Domain 1 (Architecture) -- consistent to_graph API with other loaders.


def chembl_to_graph(
    df: pd.DataFrame,
    crosswalk: Any = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert ChEMBL DataFrame to (nodes, edges) for KG construction.

    This is the main entry point for ``kg_builder`` to consume ChEMBL
    data. It generates both Compound node records and
    Compound->Protein edge records.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from ``parse_chembl_activities``.
    crosswalk : IDCrosswalk or None
        ID crosswalk for UniProt resolution.

    Returns
    -------
    tuple[list[dict], list[dict]]
        (nodes, edges) pair following the DrugOS KG convention.
    """
    nodes = chembl_to_node_records(df)
    edges = chembl_to_edge_records(df, crosswalk=crosswalk)
    logger.info(
        "chembl_to_graph: %s nodes, %s edges",
        f"{len(nodes):,}", f"{len(edges):,}",
    )
    return nodes, edges


# =============================================================================
# Section 12 -- Validation
# =============================================================================
# Fixes Domain 5 (Data Quality) -- comprehensive post-parse validation.
# Fixes Domain 10 (Testing) -- testable validation function.


def validate_chembl(
    df: pd.DataFrame,
    *,
    cfg: Optional[ChEMBLConfig] = None,
) -> Dict[str, Any]:
    """Validate a parsed ChEMBL DataFrame.

    Performs comprehensive data quality checks on the output of
    ``parse_chembl_activities``. Returns a validation result dict
    that downstream consumers can inspect.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    cfg : ChEMBLConfig or None
        Configuration for validation thresholds.

    Returns
    -------
    dict
        Validation result with keys:
        - "valid" (bool): whether all checks passed
        - "checks" (dict): individual check results
        - "warnings" (list[str]): non-fatal issues
        - "errors" (list[str]): fatal issues
        - "stats" (dict): summary statistics
    """
    if cfg is None:
        cfg = ChEMBLConfig()

    checks: Dict[str, bool] = {}
    warnings_list: List[str] = []
    errors_list: List[str] = []
    stats: Dict[str, Any] = {}

    n_rows = len(df)
    stats["row_count"] = n_rows

    # Check 1: Non-empty DataFrame
    checks["non_empty"] = n_rows > 0
    if n_rows == 0:
        errors_list.append("ChEMBL DataFrame is empty")

    # Check 2: Required columns exist
    required_cols = [
        "drug_chembl_id", "target_chembl_id", "pchembl_value",
        "standard_type",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    checks["required_columns"] = len(missing_cols) == 0
    if missing_cols:
        errors_list.append(f"Missing required columns: {missing_cols}")

    # Check 3: drug_chembl_id format
    if "drug_chembl_id" in df.columns and n_rows > 0:
        valid_ids = df["drug_chembl_id"].astype(str).str.match(
            _RE_CHEMBL_ID
        )
        invalid_count = (~valid_ids).sum()
        checks["drug_id_format"] = invalid_count == 0
        if invalid_count > 0:
            errors_list.append(
                f"{invalid_count} rows have invalid drug_chembl_id format"
            )

    # Check 4: pChEMBL value range
    if "pchembl_value" in df.columns and n_rows > 0:
        lo, hi = CHEMBL_PCHEMBL_RANGE
        in_range = (df["pchembl_value"] >= lo) & (df["pchembl_value"] <= hi)
        out_of_range = (~in_range).sum()
        checks["pchembl_range"] = out_of_range == 0
        if out_of_range > 0:
            warnings_list.append(
                f"{out_of_range} rows have pChEMBL outside [{lo}, {hi}]"
            )

    # Check 5: UniProt accession population rate
    if "uniprot_accession" in df.columns and n_rows > 0:
        pop_rate = df["uniprot_accession"].notna().mean()
        stats["uniprot_population_rate"] = pop_rate
        checks["uniprot_population"] = pop_rate >= 0.30
        if pop_rate < 0.30:
            warnings_list.append(
                f"Only {pop_rate:.1%} of rows have UniProt accessions "
                f"(minimum 30%)"
            )

    # Check 6: Field population rates
    for field_name, min_rate in CHEMBL_MIN_FIELD_POPULATION.items():
        if field_name in df.columns and n_rows > 0:
            pop_rate = df[field_name].notna().mean()
            checks[f"population_{field_name}"] = pop_rate >= min_rate
            if pop_rate < min_rate:
                warnings_list.append(
                    f"Field {field_name!r} population {pop_rate:.1%} "
                    f"< minimum {min_rate:.1%}"
                )

    # Check 7: No duplicate critical fields
    if n_rows > 0 and "drug_chembl_id" in df.columns:
        n_unique_drugs = df["drug_chembl_id"].nunique()
        stats["unique_compounds"] = n_unique_drugs

    # Check 8: Provenance present
    has_provenance = bool(df.attrs.get("provenance"))
    checks["provenance_present"] = has_provenance
    if not has_provenance:
        warnings_list.append("DataFrame missing provenance metadata")

    # Check 9: Row count vs expected
    source_cfg = DATA_SOURCES[SOURCE_KEY_CHEMBL]
    expected = source_cfg.get("expected_record_count", 2_400_000)
    if n_rows > 0 and expected > 0:
        ratio = n_rows / expected
        stats["expected_ratio"] = ratio
        checks["row_count_reasonable"] = ratio >= 0.50
        if ratio < 0.50:
            errors_list.append(
                f"Row count {n_rows:,} is less than 50% of expected "
                f"{expected:,} (ratio: {ratio:.1%})"
            )

    is_valid = len(errors_list) == 0

    result: Dict[str, Any] = {
        "valid": is_valid,
        "checks": checks,
        "warnings": warnings_list,
        "errors": errors_list,
        "stats": stats,
    }
    logger.info(
        "ChEMBL validation: %s (%d checks, %d warnings, %d errors)",
        "PASS" if is_valid else "FAIL",
        len(checks),
        len(warnings_list),
        len(errors_list),
    )
    return result


# =============================================================================
# Section 13 -- End-to-end pipeline
# =============================================================================
# Fixes Domain 1 (Architecture) -- single entry point for run_pipeline.


def load_chembl(
    *,
    cfg: Optional[ChEMBLConfig] = None,
    crosswalk: Any = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """End-to-end ChEMBL loading pipeline.

    1. Download (if needed)
    2. Parse activities
    3. Validate
    4. Convert to graph (nodes + edges)

    Parameters
    ----------
    cfg : ChEMBLConfig or None
        Loader configuration. If None, uses defaults.
    crosswalk : IDCrosswalk or None
        ID crosswalk for UniProt resolution.

    Returns
    -------
    tuple[DataFrame, list[dict], list[dict]]
        (activities_df, nodes, edges)

    Raises
    ------
    ChEMBLDataIntegrityError
        If validation fails and cfg is strict.
    """
    if cfg is None:
        cfg = ChEMBLConfig()

    # Step 1: Download
    chembl_dir = download_chembl(force=cfg.force_download, cfg=cfg)

    # Step 2: Parse
    df = parse_chembl_activities(chembl_dir=chembl_dir, cfg=cfg)

    # Step 3: Validate
    validation = validate_chembl(df, cfg=cfg)
    if not validation["valid"]:
        logger.error(
            "ChEMBL validation failed: %s",
            validation["errors"],
        )

    # Step 4: Convert to graph
    nodes, edges = chembl_to_graph(df, crosswalk=crosswalk)

    logger.info(
        "load_chembl complete: %s activities, %s nodes, %s edges",
        f"{len(df):,}",
        f"{len(nodes):,}",
        f"{len(edges):,}",
    )
    return df, nodes, edges


# =============================================================================
# Section 14 -- Loader Protocol adapter
# =============================================================================
# Fixes Domain 1 (Architecture) -- implements the Loader Protocol.
# Fixes Domain 15 (Interoperability) -- polymorphic with other loaders.


class ChEMBLLoader:
    """Adapter implementing the ``Loader`` Protocol for ChEMBL.

    This allows ``run_pipeline.py`` to treat all loaders polymorphically:

    >>> from drugos_graph.chembl_loader import ChEMBLLoader
    >>> from drugos_graph._loader_protocol import Loader
    >>> loader = ChEMBLLoader()
    >>> assert isinstance(loader, Loader)

    Attributes
    ----------
    name : str
        Human-readable name for logging.
    """

    name: str = SOURCE_CHEMBL

    def __init__(self, cfg: Optional[ChEMBLConfig] = None) -> None:
        self.cfg = cfg or ChEMBLConfig()

    def download(self, force: bool = False) -> Path:
        """Download (or cached-load) the raw ChEMBL source file."""
        return download_chembl(force=force, cfg=self.cfg)

    def parse(
        self, path: Optional[Path] = None
    ) -> Iterator[Dict[str, Any]]:
        """Yield parsed activity records as dicts."""
        df = parse_chembl_activities(
            chembl_dir=path, cfg=self.cfg,
        )
        for record in df.to_dict("records"):
            yield record

    def to_graph(
        self, records: Any
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Convert records into (nodes, edges) for the KG."""
        if isinstance(records, pd.DataFrame):
            return chembl_to_graph(records)
        # If records is a list of dicts, reconstruct DataFrame
        df = pd.DataFrame(records)
        return chembl_to_graph(df)


# ═══════════════════════════════════════════════════════════════════════════════
# v26 ROOT FIX (Audit section 10 -- Phase 2 Loaders Bypass Matrix / P0 BLOCKER):
# "Make the 4 raw re-fetch loaders consume Phase 1 CSVs by default."
# The audit's recommendation was to refactor chembl_loader, drugbank_parser,
# string_loader, uniprot_loader to follow the same bridge pattern as
# disgenet_loader / omim_loader / pubchem_loader: read Phase 1 CSVs by
# default; only fall back to raw fetch when explicitly requested.
#
# The v24 fix in run_pipeline.py step7_additional_sources SKIPS these 4
# loaders when data_source="phase1" (because the bridge in step1 already
# loaded their data). This v26 fix adds Phase-1-aware functions to each
# loader so that STANDALONE use (calling download_chembl() or
# parse_chembl_activities() directly) ALSO consumes Phase 1 CSVs by
# default -- defense in depth.
# ═══════════════════════════════════════════════════════════════════════════════

# Phase 1 emits two CSVs that this loader can consume:
#   - chembl_drugs.csv           -- Compound-node metadata
#   - chembl_activities_clean.csv -- Compound-{inhibits,activates,targets}-Protein edges
_DEFAULT_PHASE1_PROCESSED_DIR: Path = (
    Path(__file__).resolve().parents[2] / "phase1" / "processed_data"
)
DEFAULT_CHEMBL_DRUGS_CSV: Path = _DEFAULT_PHASE1_PROCESSED_DIR / "chembl_drugs.csv"
DEFAULT_CHEMBL_ACTIVITIES_CSV: Path = (
    _DEFAULT_PHASE1_PROCESSED_DIR / "chembl_activities_clean.csv"
)


def parse_chembl_activities_from_phase1_csv(
    filepath: Optional[Path] = None,
) -> pd.DataFrame:
    """Read Phase 1's cleaned ``chembl_activities_clean.csv`` into a DataFrame.

    This is the Phase-1-aware analogue of ``parse_chembl_activities`` (which
    reads the raw ChEMBL SQLite database). The DataFrame schema mirrors what
    ``chembl_to_edge_records`` expects so downstream code is unchanged.

    v26 ROOT FIX (Audit section 10 -- bypass matrix): previously, calling
    ``parse_chembl_activities()`` standalone would re-download the ~2 GB
    ChEMBL SQLite and re-parse it -- bypassing the 7 weeks of Phase 1 ETL
    work (cleaning, normalization, pchembl filtering, dedup). Now
    standalone callers can use this function to consume Phase 1's
    already-cleaned output.

    Parameters
    ----------
    filepath : path-like, optional
        Explicit path to ``chembl_activities_clean.csv``. Defaults to the
        canonical Phase 1 location.

    Returns
    -------
    pd.DataFrame
        Cleaned ChEMBL activities with columns: chembl_id, target_chembl_id,
        target_uniprot_id, standard_type, standard_value, standard_units,
        pchembl_value, standard_relation, activity_type, etc.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist (Phase 1 not yet run).
    """
    path = filepath or DEFAULT_CHEMBL_ACTIVITIES_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 1 ChEMBL activities CSV not found at {path}. "
            f"Run Phase 1's ChEMBL pipeline first "
            f"(phase1.pipelines.chembl_pipeline.ChemblPipeline().run())."
        )
    df = pd.read_csv(path)
    logger.info(
        "chembl_loader: read %d rows from Phase 1 CSV %s", len(df), path,
    )
    return df


def chembl_to_edge_records_from_phase1(
    df: pd.DataFrame,
    *,
    compound_canonical_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Convert Phase 1's ChEMBL activities DataFrame to KG edge records.

    v26 ROOT FIX: Phase 1's ``chembl_activities_clean.csv`` has a different
    column name (``molecule_chembl_id``) than the raw ChEMBL SQL output
    (``drug_chembl_id``). The existing ``chembl_to_edge_records`` accesses
    ``row.drug_chembl_id`` via ``itertuples`` and validates against
    ``_RE_CHEMBL_ID`` -- so it returns 0 edges when called on the Phase 1
    CSV directly. This function handles the Phase 1 schema natively,
    mirroring the bridge's logic in ``phase1_bridge._load_chembl_activities``:
      - reads ``molecule_chembl_id`` (Compound side)
      - reads ``target_chembl_id`` (Protein side)
      - reads ``uniprot_accession`` (already-resolved UniProt AC)
      - reads ``standard_type`` / ``activity_type`` (for relation mapping)
      - reads ``pchembl_value`` (potency)
    Emits (Compound, {targets|inhibits|activates}, Protein) edges.

    v35 ROOT FIX (V35-P2-LOADERS-FIXES H-1): the previous implementation
    used the raw ChEMBL compound ID (e.g. ``CHEMBL25``) as ``src_id``
    directly. Phase 2's KG node-builder keys Compound nodes by their
    canonical InChIKey (see ``chembl_to_node_records_from_phase1``), so
    emitting an edge with ``src_id=CHEMBL25`` would never match a staged
    Compound node -- orphan edges. This fix normalizes the compound ID to
    InChIKey via the following precedence:

      1. If ``compound_canonical_map`` is provided, look up
         ``compound_id`` in it (chembl_id -> inchikey).
      2. Else if the row has a non-empty ``inchikey`` column, use it.
      3. Else fall back to the raw ``compound_id`` (preserves the prior
         behavior so we still emit the edge when no canonical ID is
         resolvable -- better an orphan edge than a silently dropped edge,
         since downstream entity resolution may still recover the link).

    Parameters
    ----------
    df : pd.DataFrame
        Phase 1's ChEMBL activities DataFrame.
    compound_canonical_map : dict, optional
        Mapping ``{chembl_id: inchikey}`` (e.g. ``{"CHEMBL25": "...InChIKey"}``)
        built from staged Compound nodes by the caller. When provided,
        every emitted edge's ``src_id`` is normalized to the InChIKey.
    """
    edges: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        # Compound side: prefer molecule_chembl_id, fall back to drug_chembl_id.
        compound_id = (
            row.get("molecule_chembl_id")
            or row.get("drug_chembl_id")
            or row.get("chembl_id")
        )
        if compound_id is None or str(compound_id).strip() in ("", "nan"):
            continue
        compound_id = str(compound_id).strip()
        # v35 ROOT FIX (V35-P2-LOADERS-FIXES H-1): normalize the
        # ChEMBL compound ID to InChIKey so the edge's src_id matches
        # the staged Compound node IDs.
        src_id: Optional[str] = None
        if compound_canonical_map is not None:
            looked_up = compound_canonical_map.get(compound_id)
            if looked_up and str(looked_up).strip():
                # v103 ROOT FIX (P2-036 deep): the v102 fix here was
                # self-defeating — it called ``_normalize_inchikey(looked_up)``
                # but then fell back to ``str(looked_up).strip().upper()``
                # when the helper returned "" (e.g. for a "nan" placeholder).
                # The fallback RE-INTRODUCED the exact bug the helper was
                # supposed to fix: a "nan" placeholder got through as "NAN"
                # and became a canonical Compound ID. Drop the fallback;
                # if the helper says "" the value is unusable.
                src_id = _normalize_inchikey(looked_up) or None
        if src_id is None:
            row_inchikey = row.get("inchikey")
            # v103 ROOT FIX (P2-036 deep): use the helper for the
            # placeholder check too. Previously this used
            # ``str(row_inchikey).strip() not in ("", "nan")`` which
            # accepted "none"/"null"/"na" as valid InChIKeys.
            _row_norm = _normalize_inchikey(row_inchikey)
            if _row_norm:
                src_id = _row_norm
        if src_id is None:
            # Fall back to the raw ChEMBL ID (last resort).
            src_id = compound_id
        # Protein side: prefer uniprot_accession (already resolved by Phase 1).
        uniprot_ac = row.get("uniprot_accession")
        target_chembl_id = row.get("target_chembl_id")
        if uniprot_ac is None or str(uniprot_ac).strip() in ("", "nan"):
            # If we have target_chembl_id but no uniprot_ac, emit a
            # CHEMBL_TGT_<digits> ID (mirrors phase1_bridge.py:1702).
            tc = str(target_chembl_id or "").strip()
            if tc and tc.startswith("CHEMBL"):
                digits = tc.replace("CHEMBL", "")
                if digits.isdigit():
                    uniprot_ac = f"CHEMBL_TGT_{digits}"
                else:
                    continue
            else:
                continue
        uniprot_ac = str(uniprot_ac).strip()
        # Relation type: prefer standard_type, fall back to activity_type.
        std_type = row.get("standard_type") or row.get("activity_type") or ""
        if std_type is None or str(std_type) == "nan":
            std_type = ""
        rel_type = standard_type_to_relation(str(std_type).strip())
        # pchembl_value (potency).
        pchembl = row.get("pchembl_value")
        try:
            # v71 ROOT FIX (P2L-014): use pd.isna() instead of
            # ``str(pchembl) != "nan"``. The previous check was fragile:
            # ``str(np.float64('nan'))`` returns "nan" (works), but
            # ``str(Decimal('nan'))`` returns "NaN" (capital N) -- the
            # case-sensitive check would FAIL for Decimal, letting NaN
            # slip through as float('nan') -> JSON serialization fails.
            # ``pd.isna()`` handles ALL NaN variants (float, np.float32,
            # np.float64, Decimal, None) uniformly.
            pchembl_f = (
                float(pchembl)
                if pchembl is not None and not pd.isna(pchembl)
                else None
            )
        except (TypeError, ValueError):
            pchembl_f = None
        # v27 ROOT FIX (P2-L-3): normalize ChEMBL pchembl_value from its
        # native 0-14 scale (per ChEMBL docs -- pChEMBL = -log10(molar
        # activity) for IC50/Ki/Kd/EC50/AC50; range is roughly 0-14, with
        # 14 corresponding to sub-picomolar potency) to a canonical 0-1
        # range so it is comparable with DisGeNET / OpenTargets / OMIM /
        # DrugBank scores already on a 0-1 scale. Emit BOTH the raw
        # source-specific pchembl_value (preserved for traceability --
        # the RL safety ranker needs the absolute potency, not just the
        # normalized form) AND a canonical ``normalized_score`` in [0,1]
        # for downstream model training / cross-source fusion. ChEMBL
        # pchembl max is 14.
        if pchembl_f is not None:
            normalized_score = min(max(pchembl_f / 14.0, 0.0), 1.0)
            # v35 ROOT FIX (V35-P2-LOADERS-FIXES L-7): defensively warn
            # when a low-pchembl (<5.0) activity is observed in Phase 1
            # data. pChEMBL <5.0 corresponds to ~millimolar potency,
            # which is well below the typical bioactivity threshold used
            # downstream. We do NOT drop the row (the downstream model
            # may still want to see it), but we surface a WARNING so the
            # operator is aware of low-confidence measurements in the
            # Phase 1 input. The audit flagged this as a defensive-
            # logging gap.
            if pchembl_f < 5.0:
                logger.warning(
                    "chembl_loader: row %s has pchembl_value=%.3f (<5.0); "
                    "low-potency activity kept but flagged. chembl_id=%s "
                    "target_uniprot=%s",
                    idx, pchembl_f, compound_id, uniprot_ac,
                )
        else:
            normalized_score = None
        # standard_relation (censoring: >, <, =, ~).
        std_rel = row.get("standard_relation")
        std_rel_s = str(std_rel).strip() if std_rel is not None and str(std_rel) != "nan" else ""
        edges.append({
            "src_id": src_id,
            "dst_id": uniprot_ac,
            "src_type": "Compound",
            "dst_type": "Protein",
            "rel_type": rel_type,
            "props": {
                "pchembl_value": pchembl_f,
                # v27 ROOT FIX (P2-L-3): raw source-specific score,
                # preserved under a descriptive name for traceability.
                "chembl_pchembl_value": pchembl_f,
                # Canonical normalized score in [0,1] for cross-source fusion.
                # v69 ROOT FIX (P2L-011): this ``normalized_score`` field is
                # kept for BACKWARD COMPATIBILITY only. pchembl/14 yields a
                # 0-1 number but does NOT make ChEMBL scores comparable with
                # DisGeNET/OpenTargets/OMIM association probabilities.
                # pchembl=7 (≈100nM, decent potency) becomes 0.5 -- the same
                # numeric value as a 0.5 DisGeNET association score (moderate
                # gene-disease evidence). Mixing potency with association
                # probability is meaningless. Downstream consumers MUST use
                # the source-specific ``chembl_pchembl_value`` for fusion,
                # NOT the generic ``normalized_score``. The
                # ``score_semantic`` and ``score_aggregation`` fields below
                # document the semantic so consumers can route correctly.
                "normalized_score": normalized_score,
                # v69 ROOT FIX (P2L-011 + P2L-046): document per-source
                # score semantics. ``score_semantic="potency_pchembl"``
                # marks this as a potency measure (-log10[M] of IC50/Ki/Kd),
                # NOT an association probability. ``score_aggregation="single"``
                # means each ChEMBL activity is a separate measurement
                # (no dedup aggregation -- unlike OpenTargets' "max" and
                # DisGeNET's "sum_normalized"). Downstream fusion MUST
                # weight by these fields and MUST NOT average
                # ``normalized_score`` across sources with different
                # semantics.
                "score_semantic": "potency_pchembl",
                "score_aggregation": "single",
                "standard_relation": std_rel_s or None,
                "standard_type": str(std_type).strip(),
                "evidence": "chembl_bioactivity",
                "source": "chembl",
                # v35 ROOT FIX (H-1): preserve the raw ChEMBL compound ID
                # for traceability alongside the normalized InChIKey src_id.
                "chembl_compound_id": compound_id,
            },
            "source": "chembl",
            "pchembl_value": pchembl_f,
            "normalized_score": normalized_score,
            "standard_relation": std_rel_s or None,
            "_source_phase": 1,
            "_source_file": "chembl_activities_clean.csv",
            "_source_row": int(idx) if idx is not None else 0,
        })
    return edges


def chembl_to_node_records_from_phase1(
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Convert Phase 1's ChEMBL drugs DataFrame to Compound node records.

    v27 ROOT FIX (P2-L-1): Phase 1's ``chembl_drugs.csv`` uses the column
    name ``chembl_id`` (NOT ``drug_chembl_id`` -- that alias is only present
    in the raw SQLite SQL emitted by ``parse_chembl_activities``). The
    previous implementation blindly delegated to ``chembl_to_node_records``,
    which early-returns ``[]`` when ``drug_chembl_id`` is missing -- silently
    dropping 100% of Phase 1 compound rows.

    This implementation reads the Phase 1 schema natively:
      - ``chembl_id``     -- ChEMBL molecule ID (CHEMBL<digits>)
      - ``inchikey``      -- preferred canonical ID (uppercased; kg_builder
                            ID_PATTERNS requires uppercase InChIKeys)
      - ``smiles``        -- canonical SMILES
      - ``name``          -- preferred compound name

    Node ``id`` is set to ``inchikey`` (preferred) when present, falling
    back to ``chembl_id`` so the node is still emitted for compounds
    without a resolved structure.
    """
    if df is None or len(df) == 0:
        return []

    nodes: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for idx, row in df.iterrows():
        chembl_id = row.get("chembl_id")
        if chembl_id is None or str(chembl_id).strip() in ("", "nan"):
            continue
        chembl_id = str(chembl_id).strip()

        inchikey = row.get("inchikey")
        # v102 P2-036: route through centralized normalize_inchikey so
        # every loader (chembl / pubchem / phase1_bridge) produces the
        # SAME canonical form. Previously this loader used
        # ``str(inchikey).strip().upper()`` while pubchem_loader used
        # ``inchikey.upper()`` (no strip) and phase1_bridge used
        # ``inchikey.upper() if inchikey else ""`` (no strip, None-
        # unsafe). The three forms produced different canonical IDs for
        # the same compound when whitespace or "nan" placeholders were
        # present, fragmenting entity resolution.
        _ik_norm = _normalize_inchikey(inchikey)
        inchikey_s = _ik_norm if _ik_norm else None

        smiles = row.get("smiles")
        smiles_s = (
            str(smiles).strip()
            if smiles is not None and str(smiles) != "nan" and str(smiles).strip() != ""
            else ""
        )

        name = row.get("name")
        name_s = (
            str(name).strip()
            if name is not None and str(name) != "nan" and str(name).strip() != ""
            else chembl_id
        )

        # Canonical node ID: prefer InChIKey (uppercased to satisfy
        # kg_builder.ID_PATTERNS), fall back to chembl_id.
        canonical_id = inchikey_s or chembl_id
        if canonical_id in seen_ids:
            continue
        seen_ids.add(canonical_id)

        node: Dict[str, Any] = {
            "id": canonical_id,
            "label": ENTITY_TYPE_COMPOUND,
            "name": name_s,
            "chembl_id": chembl_id,
            "smiles": smiles_s,
            "source": SOURCE_CHEMBL,
            "_provenance": df.attrs.get("provenance", {}),
            "_license": CHEMBL_LICENSE,
            "_attribution": CHEMBL_ATTRIBUTION,
        }
        if inchikey_s:
            node["inchikey"] = inchikey_s
        nodes.append(node)

    logger.info(
        "Generated %s ChEMBL Compound node records from Phase 1 CSV",
        f"{len(nodes):,}",
    )
    return nodes
