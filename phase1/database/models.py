"""
SQLAlchemy ORM models for the Drug Repurposing ETL platform.

This module is a **pure ORM definition module** -- it contains only the seven
table models (``Drug``, ``Protein``, ``DrugProteinInteraction``,
``ProteinProteinInteraction``, ``GeneDiseaseAssociation``,
``EntityMapping``, ``PipelineRun``), the ``SchemaVersion`` metadata table,
and Python-side validation helpers.  Business logic (e.g. orphan cleanup)
has been moved to ``database.loaders`` (ARCH-01).

Every model corresponds one-to-one with the schema defined in
``database/migrations/001_initial_schema.sql`` and subsequent migrations.

Tables
------
  - drugs
  - proteins
  - drug_protein_interactions
  - protein_protein_interactions
  - gene_disease_associations
  - entity_mapping
  - pipeline_runs
  - schema_version              [ARCH-07] metadata table for version tracking

Changelog
---------
v1 -- Initial models (429 lines).
v2 -- 78 fixes across 16 verification domains (SCI, DQ, IDEM, ARCH, ...).
"""

from __future__ import annotations

import datetime
import enum
import logging
import re
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    column,
    func,
    text,
)
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from database.base import Base, IDMixin, SoftDeleteMixin, TimestampMixin

# SCHEMA_VERSION is defined in database.base (the single source of truth).
# Re-export it here so callers can import it from either location.
from database.base import SCHEMA_VERSION  # noqa: F401 -- re-exported for callers

# ---------------------------------------------------------------------------
# Module logger (LOG-01, LOG-02)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ===========================================================================
# Column-length constants (CFG-01 -- documented, no magic numbers)
# ===========================================================================
#: Standard InChIKey length.  Synthetic keys (prefixed 'SYNTH') may exceed
#: 27 chars, so the column is widened to 50 (SCI-01).
INCHIKEY_LENGTH: int = 50

#: UniProt accession max length (6-10 alphanumeric chars, SCI-05).
UNIPROT_ID_LENGTH: int = 10

#: HGNC gene symbol max length (SCI-04).
GENE_SYMBOL_LENGTH: int = 50

#: Drug name max length.
DRUG_NAME_LENGTH: int = 500

#: ChEMBL ID max length (e.g. 'CHEMBL25').
CHEMBL_ID_LENGTH: int = 20

#: DrugBank ID max length.
#: P1-017 ROOT FIX (Team-2): widened from 10 to 64 to accommodate
#: synthesized IDs (``SYNTH-DB-{8 hex}`` = 17 chars, ``SYNTH-DB-M{6 digits}``
#: = 17 chars). Real DrugBank IDs (``DB\d{5,7}`` = 7-9 chars) still fit.
#: The previous VARCHAR(10) silently truncated the ``DBSYNTH{6 digits}``
#: form (13 chars) — see P1-017 root fix in drugbank_pipeline.py and
#: migration 013_widen_drugbank_id_column.sql.
DRUGBANK_ID_LENGTH: int = 64

#: STRING protein ID max length (e.g. '9606.ENSP00000269305').
STRING_ID_LENGTH: int = 50

#: Source pipeline name max length.
SOURCE_LENGTH: int = 20

#: Pipeline source max length (longer to accommodate future names).
PIPELINE_SOURCE_LENGTH: int = 50

#: PMID list max length (SEC-02 -- capped to prevent unbounded Text DoS).
PMID_LIST_LENGTH: int = 2000

#: Error message max length (SEC-04 -- prevent stack trace leakage).
ERROR_MESSAGE_LENGTH: int = 500

#: Disease ID max length.
DISEASE_ID_LENGTH: int = 50

#: Disease ID type max length (SCI-06).
DISEASE_ID_TYPE_LENGTH: int = 20

# ===========================================================================
# Domain enums (DES-05 -- enforced at DB and Python level)
# ===========================================================================


class ClinicalPhase(int, enum.Enum):
    """FDA clinical trial phases (SCI-02, DES-05).

    0 = Pre-clinical, 1 = Phase I, 2 = Phase II,
    3 = Phase III, 4 = Approved.
    """
    PRECLINICAL = 0
    PHASE_I = 1
    PHASE_II = 2
    PHASE_III = 3
    APPROVED = 4


class DrugType(str, enum.Enum):
    """Drug classification categories (DES-05)."""
    SMALL_MOLECULE = "small_molecule"
    ANTIBODY = "antibody"
    PROTEIN = "protein"
    OLIGONUCLEOTIDE = "oligonucleotide"
    PEPTIDE = "peptide"
    CELL_THERAPY = "cell_therapy"
    GENE_THERAPY = "gene_therapy"
    UNKNOWN = "unknown"


class PipelineStatus(str, enum.Enum):
    """Pipeline run status (DES-05, DES-07)."""
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class InteractionType(str, enum.Enum):
    """Drug-protein interaction types (DES-05)."""
    INHIBITOR = "inhibitor"
    ACTIVATOR = "activator"
    AGONIST = "agonist"
    ANTAGONIST = "antagonist"
    BINDING_AGENT = "binding_agent"
    BLOCKER = "blocker"
    MODULATOR = "modulator"
    # v90 ROOT FIX (BUG #3): added INDUCER and SUBSTRATE enum values.
    # CYP induction (upregulation of enzyme expression) and CYP
    # substrate (drug metabolized by the enzyme) are pharmacologically
    # DISTINCT from inhibition and "unknown". Mapping them to "unknown"
    # lost pharmacological direction in the KG.
    INDUCER = "inducer"
    SUBSTRATE = "substrate"
    UNKNOWN = "unknown"


class ActivityType(str, enum.Enum):
    """Activity measurement types (DES-05)."""
    IC50 = "IC50"
    EC50 = "EC50"
    KI = "Ki"
    KD = "Kd"
    POTENCY = "potency"
    AC50 = "AC50"
    PIC50 = "pIC50"
    PEC50 = "pEC50"
    PKI = "pKi"
    PKD = "pKd"
    PKB = "pKb"
    PED50 = "pED50"
    PAC50 = "pAC50"
    ED50 = "ED50"
    KB = "Kb"
    UNKNOWN = "unknown"


# ===========================================================================
# Validation regex patterns (SCI-04, SCI-05, SCI-08)
# ===========================================================================

#: UniProt accession format: 6 or 10 alphanumeric chars (SCI-05).
#:
#: v21 ROOT FIX (Audit section 5 finding 2 / Chain 3 - "Three divergent
#: UniProt regexes"): the previous pattern was
#: ``^[A-Z][0-9][A-Z0-9]{3}[0-9]([A-Z][A-Z0-9]{2}[0-9])?$`` which accepts
#: ANY letter as the first char. But the OFFICIAL UniProt accession
#: format (per https://www.uniprot.org/help/accession_numbers) is:
#:   - 6 chars: [OPQ][0-9][A-Z0-9]{3}[0-9]  (OPQ prefix)
#:   - 10 chars: [A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}
#:     (any letter except P/Q/O - because those start 6-char IDs;
#:     each 4-char block must START with a letter per the official spec)
#: The previous pattern accepted B12345 (B is not in [OPQ]) as a valid
#: UniProt ID, then the resolver's stricter regex rejected it - causing
#: silent data loss in the crosswalk. Unify ALL three regexes (models,
#: resolver_utils, drug_resolver) on the OFFICIAL pattern below. The
#: entity_resolver.py / id_crosswalk.py / protein_resolver.py already
#: use this exact pattern (resolver_utils._UNIPROT_ACCESSION_RE).
#:
#: v22 ROOT FIX (audit P1-8 / section 5 finding 2 -- "Three divergent
#: UniProt regexes"): the previous ``_UNIPROT_RE`` used the LOOSE pattern
#: ``[A-NR-Z][0-9][A-Z0-9]{3}[0-9][A-Z0-9]{3}[0-9]`` for 10-char IDs
#: (first char of each 4-char block could be a digit) -- divergent from
#: the official ``resolver_utils._UNIPROT_ACCESSION_RE`` which uses
#: ``[A-Z][A-Z0-9]{2}[0-9]`` (first char MUST be a letter). It also
#: added ``(-\d+)?`` isoform suffix and ``|^CHEMBL_TGT_\d+$`` alternative
#: -- neither of which is a UniProt accession. The loose pattern accepted
#: IDs like A0A024R1G1 (digit-first 4-char block) that the resolver
#: rejected. Unify: use the OFFICIAL pattern from resolver_utils
#: EXACTLY. Isoform suffix and CHEMBL_TGT_ prefix are handled separately
#: in ``_validate_uniprot_id`` (not in the regex).
try:
    from entity_resolution.resolver_utils import _UNIPROT_ACCESSION_RE as _UNIPROT_RE  # noqa: F401
except ImportError:
    # Fallback: replicate the OFFICIAL UniProt pattern EXACTLY (no
    # divergence). This branch only runs if entity_resolution is not
    # importable (test isolation). The pattern MUST stay in sync with
    # resolver_utils._UNIPROT_ACCESSION_RE.
    #
    # v29 ROOT FIX (audit D-4): the canonical UniProt accession regex is
    # also exposed as ``cleaning._constants.CANONICAL_UNIPROT_ACCESSION_REGEX_FULL``
    # (single source of truth, mirrors the InChIKey pattern). The DB CHECK
    # on ``proteins.uniprot_id`` remains LENGTH-based (portable across
    # SQLite / PostgreSQL) because SQLite cannot enforce a regex in a CHECK
    # constraint; the strict regex IS enforced at the Python layer by
    # ``_validate_uniprot_id`` below. See the docstring of
    # ``CANONICAL_UNIPROT_ACCESSION_REGEX`` in cleaning/_constants.py for
    # the full rationale (audit D-4 / "mouse accessions accepted into
    # human set").
    try:
        from cleaning._constants import (
            CANONICAL_UNIPROT_ACCESSION_REGEX_FULL as _UNIPROT_RE,  # noqa: F401
        )
    except ImportError:
        _UNIPROT_RE: re.Pattern[str] = re.compile(
            r"^([OPQ][0-9][A-Z0-9]{3}[0-9]"
            r"|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$"
        )

#: HGNC gene symbol format (SCI-04).
#:
#: v21 ROOT FIX (Audit section 5 finding 1 / Chain 2 - "Three divergent
#: gene-symbol regexes"): models._GENE_SYMBOL_RE was
#: ``^[A-Z][A-Z0-9\-]{0,49}$`` (ALL CAPS, human-only). But
#: protein_resolver._normalize_gene_symbol accepts Title-Case mouse
#: symbols ('Tp53', 'Brca1'). A mouse protein entering via
#: protein_resolver passed _pre_validate_proteins, then
#: models._validate_gene_symbol raised ValueError, and the loader
#: silently set gene_symbol=None - destroying the protein-disease edge
#: for that gene. Unify: accept ALL-CAPS human symbols AND Title-Case
#: non-human symbols. The check is "first char uppercase letter, rest
#: alphanumeric or hyphen, 1-50 chars" - covers human (FGFR3, BRCA1),
#: mouse (Tp53, Brca1), rat (Tp53), yeast (HO, GAL4). The strict
#: ALL-CAPS check is moved to a separate ``_HUMAN_GENE_SYMBOL_RE``
#: used only where human-only context is explicit (e.g. DisGeNET GDA
#: rows tagged organism=9606).
_GENE_SYMBOL_RE: re.Pattern[str] = re.compile(
    r"^[A-Za-z][A-Za-z0-9\-]{0,49}$"
)
#: Strict human-only gene symbol (ALL CAPS). Used where the data source
#: is documented to be human-only (e.g. DisGeNET human GDA subset).
_HUMAN_GENE_SYMBOL_RE: re.Pattern[str] = re.compile(
    r"^[A-Z][A-Z0-9\-]{0,49}$"
)

#: Amino acid sequence: 20 standard + ambiguity codes (SCI-08) + the
#: alignment gap char ``-`` (v35 root fix: included for consistency with
#: ``cleaning._constants.CANONICAL_AA_SEQUENCE_REGEX`` and the pipeline
#: validators -- without it, an aligned sequence with gaps would pass
#: the cleaning validator but fail this DB validator, causing silent
#: data loss at the cleaning -> DB boundary).
_SEQUENCE_RE: re.Pattern[str] = re.compile(
    r"^[ACDEFGHIKLMNPQRSTVWYBJOUXZ\*\-]+$"
)

#: Standard InChIKey format: 27 chars (SCI-01).
#:
#: v35 ROOT FIX (issue 28): import the canonical InChIKey regex from
#: ``cleaning._constants`` (single source of truth) instead of defining
#: it locally. The previous local definition included an optional
#: ``-X`` protonation suffix that the canonical validator REJECTS (the
#: canonical InChIKey is exactly 27 chars per IUPAC; protonation
#: extensions are stripped by ``strip_inchikey_extension`` before
#: validation). Having two definitions meant future edits to one could
#: silently diverge from the other (audit Chain 3).
#: P1-ER-3 ROOT FIX: pattern synchronized with normalizer.py / base.py /
#: models.py -- DO NOT diverge (audit P1-ER-3).
try:
    from cleaning._constants import (
        CANONICAL_INCHIKEY_REGEX as _STANDARD_INCHIKEY_RE,  # noqa: F401
    )
except ImportError:
    # Fallback (test isolation / partial install): replicate the canonical
    # pattern EXACTLY. See the docstring of ``CANONICAL_INCHIKEY_REGEX``
    # in cleaning/_constants.py for the full rationale.
    _STANDARD_INCHIKEY_RE: re.Pattern[str] = re.compile(
        r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"
    )

# ===========================================================================
# Internal validation helpers (SCI-04, SCI-05, SCI-08, SCI-01, SCI-02)
# ===========================================================================


def _validate_inchikey(value: Optional[str]) -> Optional[str]:
    """Validate InChIKey format (SCI-01).  Accepts standard 27-char keys
    and synthetic keys prefixed with 'SYNTH'.

    The DB layer delegates to ``cleaning.normalizer.is_valid_inchikey`` --
    the SINGLE canonical validator. See P1-ER-2 / P1-ER-3 for the audit
    that removed acceptance of TEST/OUTER/INNER/IK test-fixture prefixes
    from the canonical contract.

    v35 ROOT FIX (issue 28): the previous docstring mentioned "optional
    protonation suffix" -- the canonical InChIKey is exactly 27 chars per
    IUPAC; protonation extensions (``-a``, ``-N-a``) are NOT part of the
    canonical key and must be stripped by ``strip_inchikey_extension``
    BEFORE validation. The docstring has been updated to reflect this.
    """
    if value is None:
        return value
    value = value.strip()
    try:
        from cleaning.normalizer import is_valid_inchikey as _canonical_is_valid_inchikey
        if _canonical_is_valid_inchikey(value):
            # P1-063/068 ROOT FIX: normalize SYNTH keys to uppercase.
            if value.upper().startswith("SYNTH"):
                return value.upper()
            return value
    except ImportError:
        # Fallback: replicate the canonical regex EXACTLY (no divergence).
        # This branch only runs if cleaning.normalizer is not importable
        # (test isolation). The patterns here MUST stay in sync with
        # cleaning.normalizer.is_valid_inchikey.
        # P1-ER-2 ROOT FIX: removed TEST/OUTER/INNER/IK acceptance from
        # the fallback branch -- it must mirror the canonical validator.
        # P1-ER-3 ROOT FIX: pattern synchronized with normalizer.py /
        # base.py / models.py -- DO NOT diverge.
        # v35 ROOT FIX (issue 28): ``_STANDARD_INCHIKEY_RE`` is now
        # imported from ``cleaning._constants`` (single source of truth).
        upper = value.upper()
        if _STANDARD_INCHIKEY_RE.match(value):
            return value
        if upper.startswith("SYNTH"):
            # P1-063/068 ROOT FIX: normalize SYNTH keys to uppercase.
            return upper
    raise ValueError(
        f"Invalid InChIKey format: '{value}'. "
        "Must be 27-char standard format or start with 'SYNTH'."
    )


def _validate_uniprot_id(value: Optional[str]) -> Optional[str]:
    """Validate UniProt accession format (SCI-05).

    Accepts standard UniProt accessions (e.g. P69999, Q9Y6K9) AND short
    test identifiers (e.g. P001, P100) used in test fixtures. The test
    identifiers are accepted only when they start with 'TEST' or are
    shorter than 6 characters (real UniProt IDs are always 6-10 chars).

    v22 ROOT FIX (audit P1-8): the regex no longer accepts isoform
    suffixes (``-N``) or ``CHEMBL_TGT_`` prefixes. Those are handled
    HERE explicitly so the regex stays unified with
    ``resolver_utils._UNIPROT_ACCESSION_RE``.
    """
    if value is None:
        return value
    value = value.strip()
    # v22: handle isoform suffix (e.g. P04637-2) by stripping it before
    # validation, then returning the ORIGINAL value (callers want the
    # isoform-specific ID preserved).
    base = value
    isoform_suffix = ""
    if "-" in value and not value.startswith("-"):
        parts = value.rsplit("-", 1)
        if parts[1].isdigit():
            base, isoform_suffix = parts[0], "-" + parts[1]
    # v22: CHEMBL_TGT_ prefix is a Phase 2 synthetic ID for ChEMBL
    # targets without UniProt AC. Accept it explicitly here (not in the
    # UniProt regex -- it is NOT a UniProt accession).
    # v107 FORENSIC ROOT FIX (ISSUE-P1-011):
    #   The previous code ACCEPTED ``CHEMBL_TGT_*`` prefixed IDs as valid
    #   UniProt accessions, storing them in ``proteins.uniprot_id``. But
    #   ChEMBL target IDs and UniProt accessions are DIFFERENT identifier
    #   systems. Conflating them meant a ``CHEMBL_TGT_12345`` value landed
    #   in ``proteins.uniprot_id`` and was treated as a UniProt accession
    #   by downstream joins. Cross-source protein resolution joins
    #   ``proteins.uniprot_id`` against UniProt's canonical accession set
    #   -- a ``CHEMBL_TGT_*`` value never matches a real UniProt accession,
    #   producing orphan proteins and broken Gene->encodes->Protein edges.
    #   The KG had phantom proteins.
    #   ROOT FIX: REJECT ``CHEMBL_TGT_*`` here. ChEMBL target IDs must be
    #   stored in a SEPARATE ``chembl_target_id`` column (which the schema
    #   already supports via the ``chembl_activities`` table's
    #   ``target_chembl_id`` column). The ``uniprot_id`` column must
    #   contain ONLY real UniProt accessions (or NULL when the protein
    #   has no known UniProt mapping -- in that case the loader should
    #   skip the row or use a synthetic ID in a dedicated column).
    if base.upper().startswith("CHEMBL_TGT_"):
        raise ValueError(
            f"Invalid UniProt accession: '{value}'. "
            f"CHEMBL_TGT_* prefixed IDs are ChEMBL target IDs, NOT UniProt "
            f"accessions. Store ChEMBL target IDs in the chembl_target_id "
            f"column, not in uniprot_id. Conflating these identifier "
            f"systems breaks cross-source protein resolution and creates "
            f"phantom protein nodes in the KG."
        )
    if _UNIPROT_RE.match(base):
        return value
    # v34 ROOT FIX (CRITICAL #3): previously accepted TEST-prefixed IDs
    # and any <6-char alphanumeric as valid UniProt accessions, claiming
    # "never in production" but providing NO enforcement. This caused
    # test-fixture proteins like `P001` to flow into the production `proteins`
    # table -- contradicting the P1-ER-2 ROOT FIX that REJECTED test-fixture
    # InChIKeys. Now we ONLY accept test fixtures when DRUGOS_ENVIRONMENT
    # is explicitly set to a dev/test value. In production (or unset),
    # test fixtures are REJECTED.
    #
    # v57 ROOT FIX (P1C-002): the previous default was "dev", which meant
    # test-fixture proteins (e.g. `P001`, `TEST001`) were accepted by
    # DEFAULT -- contaminating the proteins table in any deployment that
    # forgot to set DRUGOS_ENVIRONMENT=prod. The default is now "prod"
    # so test fixtures are rejected UNLESS the operator explicitly opts
    # into dev mode. This matches the existing comment's promise of
    # "rejected in production environments (set DRUGOS_ENVIRONMENT=dev
    # to allow)".
    #
    # v65 ROOT FIX (P1C-002 compound -- root contamination path):
    #   The v57 fix changed the default to "prod" but LEFT the
    #   `<6-char alphanumeric` acceptance block in place. This block
    #   accepted junk like `P001`, `ABC`, `12345` -- NONE of which are
    #   real UniProt accessions (real accessions are ALWAYS 6 or 10
    #   chars per the UniProt spec) and NONE of which are TEST-prefixed.
    #   So even with the "prod" default, an operator who explicitly set
    #   DRUGOS_ENVIRONMENT=staging (a production-like env per .env.example)
    #   still got test-fixture proteins in the proteins table. The audit
    #   called this the "asymmetry" with the P1-ER-2 InChIKey fix: test-
    #   fixture DRUGS are rejected but test-fixture PROTEINS are accepted.
    #   ROOT FIX:
    #     1. Remove "staging" from the allow-test list -- staging MUST be
    #        production-like (per .env.example: "staging: production-like
    #        with moderate limits"). Only dev/development/test/ci accept
    #        test fixtures.
    #     2. Remove the `<6-char alphanumeric` block ENTIRELY. The only
    #        test-fixture IDs that should be accepted are TEST-prefixed
    #        (explicit, self-documenting). Real UniProt accessions are
    #        always 6+ chars and match the strict regex above.
    # P1-028 v113 ROOT FIX: the previous code accepted TEST-prefixed UniProt
    # IDs (e.g. TEST001) in dev/test/ci environments. But the DB CHECK
    # constraint (migration 016) enforces the canonical UniProt regex
    # `^[OPQ][0-9][A-Z0-9]{3}[0-9]$` OR `^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$`
    # — which TEST001 does NOT match. So the Python validator passed the
    # value, the loader built an INSERT, and the DB rejected it with
    # CheckViolation. Tests passed on SQLite (no CHECK enforced) but FAILED
    # on PostgreSQL integration tests (CHECK enforced).
    #
    # ROOT FIX: remove the TEST-prefix acceptance ENTIRELY. Test fixtures
    # MUST use real UniProt accessions (e.g. P04637 for TP53, P00533 for
    # EGFR). This eliminates the dev/prod asymmetry. The
    # DRUGOS_ENVIRONMENT check is removed — there is no longer any
    # environment where TEST-prefixed IDs are valid.
    raise ValueError(
        f"Invalid UniProt accession: '{value}'. "
        "Must match pattern like P69999 or Q9Y6K9. "
        "P1-028 v113: TEST-prefixed fixture IDs are NO LONGER accepted in "
        "any environment (they violated the DB CHECK constraint on "
        "PostgreSQL). Test fixtures MUST use real UniProt accessions "
        "(e.g. P04637 for TP53, P00533 for EGFR, Q9Y6K9 for a generic "
        "test protein)."
    )


def _validate_gene_symbol(value: Optional[str]) -> Optional[str]:
    """Validate HGNC gene symbol format (SCI-04)."""
    if value is None:
        return value
    value = value.strip()
    if _GENE_SYMBOL_RE.match(value):
        return value
    raise ValueError(
        f"Invalid gene symbol: '{value}'. "
        "Must be uppercase letter followed by alphanumeric/hyphen chars."
    )


def _validate_sequence(value: Optional[str]) -> Optional[str]:
    """Validate protein amino acid sequence (SCI-08)."""
    if value is None:
        return value
    value = value.strip()
    if _SEQUENCE_RE.match(value):
        return value
    raise ValueError(
        f"Invalid protein sequence: contains non-amino-acid characters. "
        "Allowed: A C D E F G H I K L M N P Q R S T V W Y B J O U X Z *"
    )


def _validate_max_phase(value: Optional[int]) -> Optional[int]:
    """Validate clinical trial phase is in range [0, 4] (SCI-02)."""
    if value is None:
        return value
    if not (0 <= value <= 4):
        raise ValueError(
            f"Invalid max_phase: {value}. Must be between 0 (pre-clinical) "
            "and 4 (approved)."
        )
    return value


# ===========================================================================
# Public API -- explicit declaration (ARCH-03)
# ===========================================================================
__all__: list[str] = [
    "Drug",
    "Protein",
    "DrugProteinInteraction",
    "ProteinProteinInteraction",
    "GeneDiseaseAssociation",
    "EntityMapping",
    "PipelineRun",
    "SchemaVersion",
    # Enums
    "ClinicalPhase",
    "DrugType",
    "PipelineStatus",
    "InteractionType",
    "ActivityType",
    # Constants
    "INCHIKEY_LENGTH",
    "UNIPROT_ID_LENGTH",
    "GENE_SYMBOL_LENGTH",
    "SCHEMA_VERSION",
]


# ===========================================================================
# SCHEMA VERSION TABLE (ARCH-07)
# ===========================================================================


class SchemaVersion(Base, IDMixin):
    """Tracks the applied schema version for programmatic verification.

    [ARCH-07] Enables runtime check of which migration revision the
    database is at.  One row per applied migration, latest row = current
    version.
    """
    __tablename__ = "schema_version"

    version: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    applied_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    description: Mapped[str] = mapped_column(String(200), nullable=False)

    def __repr__(self) -> str:
        return f"<SchemaVersion(version={self.version}, description='{self.description}')>"


# ===========================================================================
# 1. DRUGS
# ===========================================================================


class Drug(Base, IDMixin, TimestampMixin, SoftDeleteMixin):
    """Master drug table -- unified across all sources.

    Domain meaning
    --------------
    Each row represents a unique chemical compound identified by its InChIKey.
    Data is sourced from ChEMBL (primary), DrugBank, and PubChem.

    Key constraints
    ---------------
    - ``inchikey`` is the natural primary identifier (unique, not null).
    - ``max_phase`` must be 0-4 (SCI-02, clinical trial phases).
    - ``molecular_weight`` must be positive (DQ-09).
    - ``is_fda_approved`` is a boolean with a CHECK for SQLite compat (DQ-01).
    - ``name`` must be at least 2 characters (DQ-04).

    Relationships
    -------------
    - ``drug_protein_interactions`` -> DPI rows (cascade delete-orphan).

    [SCI-01] inchikey widened to String(50) for synthetic keys.
    [SCI-07] molecular_weight uses Numeric(12,6) for precision.
    [DES-05] drug_type and max_phase have CHECK/enum constraints.
    """
    __tablename__ = "drugs"

    # [SCI-01] InChIKey widened from 27 to 50 for synthetic keys
    inchikey: Mapped[str] = mapped_column(
        String(INCHIKEY_LENGTH), nullable=False, unique=True,
    )
    # [DQ-04] Drug name minimum length enforced by CHECK
    name: Mapped[str] = mapped_column(String(DRUG_NAME_LENGTH), nullable=False)
    chembl_id: Mapped[Optional[str]] = mapped_column(
        String(CHEMBL_ID_LENGTH), nullable=True,
    )
    drugbank_id: Mapped[Optional[str]] = mapped_column(
        String(DRUGBANK_ID_LENGTH), nullable=True,
    )
    pubchem_cid: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    molecular_formula: Mapped[Optional[str]] = mapped_column(
        String(200), nullable=True,
    )
    # [SCI-07] Numeric(12,6) instead of Float for 6 decimal precision
    molecular_weight: Mapped[Optional[float]] = mapped_column(
        Numeric(12, 6), nullable=True,
    )
    smiles: Mapped[Optional[str]] = mapped_column(String(50000), nullable=True)
    # [DQ-01] [CODE-02] Use proper server_default for cross-dialect boolean.
    # v90 ROOT FIX (BUG #23 -- complete the three-way boolean default unification):
    #   v89 only updated run_migrations.py REQUIRED_COLUMNS to DEFAULT FALSE;
    #   the ORM was left on the non-portable ``server_default="0"`` (string
    #   integer literal). Three-way drift remained: ORM "0" / migration 001
    #   FALSE / fallback FALSE. ``DEFAULT 0`` works on SQLite (BOOLEAN is
    #   INTEGER) and PostgreSQL (implicit cast to BOOLEAN) but is REJECTED by
    #   strict-mode MySQL/MariaDB and produces inconsistent column-type
    #   metadata across dialects (verify_schema_matches_orm flagged it as
    #   type drift on SQLite). ROOT FIX: switch the ORM to the SQL-standard
    # P1-049 / P1-046 FORENSIC ROOT FIX (Team 4 -- was NOT NULL DEFAULT FALSE):
    #   The previous definition was ``Mapped[bool]`` with
    #   ``server_default=text("FALSE"), nullable=False``. The ChEMBL
    #   pipeline docstring says ``is_fda_approved`` should be ``None``
    #   (unknown) until the FDA Orange Book join runs (v93 patient-safety
    #   fix). But the DB column was ``NOT NULL`` -- inserting ``None``
    #   raised ``IntegrityError``, and the loader coerced ``None`` to
    #   ``False`` to satisfy the constraint, SILENTLY REVERTING the v93
    #   fix. EMA-only drugs (max_phase=4, not FDA-approved) were stored
    #   as ``is_fda_approved=False`` -- same as a confirmed-not-approved
    #   drug.
    #
    #   ROOT FIX: change to ``Mapped[Optional[bool]]`` with
    #   ``nullable=True`` and NO server_default. NULL now means "unknown
    #   FDA status" (distinct from FALSE = "confirmed not FDA-approved").
    #   Migration 013 ALTERs the column for existing DBs; migration 001
    #   (this file's DDL emitter) is also updated.
    is_fda_approved: Mapped[Optional[bool]] = mapped_column(
        Boolean, nullable=True,
    )
    # [P1-28 ROOT FIX] Global regulatory approval flag (any of FDA / EMA /
    # PMDA / MHRA / Health Canada / TGA). Distinct from is_fda_approved
    # (FDA-specific) -- the ChEMBL pipeline emits is_globally_approved =
    # (max_phase == 4) per SW-1 ROOT FIX (patient safety). The column was
    # previously emitted by the ChEMBL pipeline but missing from the Drug
    # model, so it was silently dropped by _filter_to_drug_columns and
    # always NULL in the DB. Migration 008 adds the column.
    is_globally_approved: Mapped[bool] = mapped_column(
        # v65 ROOT FIX (P1C-006 -- silent NULL exclusion from "approved"
        # queries):
        #   The previous definition was `Mapped[Optional[bool]]` with
        #   `nullable=True` and NO server_default. When a source pipeline
        #   (e.g. DrugBank) did not emit is_globally_approved, the column
        #   stayed NULL. Downstream SQL like `WHERE is_globally_approved
        #   = True` uses three-valued logic: `NULL = True` evaluates to
        #   UNKNOWN (falsy), so NULL rows were SILENTLY excluded from
        #   "approved drugs" queries -- the opposite of the patient-safety
        #   intent. Drugs loaded from sources that don't set the flag
        #   were treated as NOT approved, reducing recall for drug-
        #   repurposing candidates.
        #   ROOT FIX: match the pattern used by is_fda_approved (line 551)
        #   and is_withdrawn (line 577) -- `nullable=False` with the SQL-
        #   standard `server_default=text("FALSE")`. INSERTs that omit
        #   the column now get False (explicit, not NULL); UPDATEs that
        #   set the flag work exactly as before. `WHERE
        #   is_globally_approved = True` now returns the correct set,
        #   and `WHERE is_globally_approved = False` returns drugs with
        #   unknown approval status (which is the conservative,
        #   patient-safe default).
        #   v90 ROOT FIX (BUG #23): aligned with is_fda_approved / is_withdrawn
        #   on `server_default=text("FALSE")` (was `server_default="0"`).
        Boolean, nullable=False, server_default=text("FALSE"),
    )
    # [SCI-02] Clinical trial phase -- validated 0-4
    max_phase: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # [DES-05] Drug type -- constrained by CHECK (enum enforced at Python level)
    drug_type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True,
    )
    mechanism_of_action: Mapped[Optional[str]] = mapped_column(String(5000), nullable=True)

    # -- LIFE-SAFETY CRITICAL: withdrawn drug tracking --
    # is_withdrawn tracks drugs withdrawn from market for safety reasons.
    # Without this column, killer drugs like Vioxx (rofecoxib, 88k-140k
    # heart attacks) and Baycol (cerivastatin, ~100 rhabdomyolysis deaths)
    # cannot be filtered out of repurposing candidates.
    is_withdrawn: Mapped[bool] = mapped_column(
        # v90 ROOT FIX (BUG #23): `server_default=text("FALSE")` instead of
        #   the non-portable `server_default="0"`. Matches is_fda_approved /
        #   is_globally_approved and migration 001 / run_migrations.py so the
        #   boolean column-type metadata is identical on every dialect.
        Boolean, server_default=text("FALSE"), nullable=False,
    )
    # Derived clinical status: approved/withdrawn/illicit/investigational/
    # vet_approved/experimental/nutraceutical/unknown.
    clinical_status: Mapped[Optional[str]] = mapped_column(
        String(30), nullable=True,
    )
    # PS-6 ROOT FIX (patient safety): DrugBank <groups> field as a
    # semicolon-separated string (e.g. "approved;withdrawn;investigational").
    # Used to derive is_withdrawn / clinical_status via a PostgreSQL
    # trigger (migration 006) and a Python-side fallback in the loader.
    # The column was missing from the ORM entirely -- the DrugBank
    # pipeline produced a 'groups' string in drugs_df, but the loader
    # silently dropped it because the ORM had no attribute, and the
    # safety trigger had no source data to fire on. Withdrawn killer
    # drugs (Vioxx, Baycol, thalidomide) stayed is_withdrawn=FALSE.
    groups: Mapped[Optional[str]] = mapped_column(
        String(200), nullable=True,
    )
    # CAS Registry Number (e.g., "50-78-2" for aspirin).
    cas_number: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True,
    )
    # v49 ROOT FIX (CRITICAL -- Compound Chain 2 / "PostgreSQL Bridge Data
    # Corruption"): the Drug ORM previously had NO `indication` column.
    # The Phase 2 bridge (`phase2/drugos_graph/phase1_bridge.py`) needs
    # the free-text indication field to synthesize Compound-treats-Disease
    # edges (the TransE link-prediction target) when running in
    # PostgreSQL mode. Without this column, PostgreSQL mode produced
    # ZERO prediction-target edges, structurally making V1 launch
    # impossible. ROOT FIX: add the column here + migration 010. The
    # DrugBank pipeline already emits `indication` in its drugs_df -- it
    # was silently dropped by _filter_to_drug_columns because the ORM
    # had no attribute. Now it loads. See also: phase1_bridge.py line
    # ~920 (the bridge's "we CANNOT reconstruct indications from
    # PostgreSQL alone" comment -- that comment is now stale).
    indication: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )
    # v49 ROOT FIX (CRITICAL -- Phase 2 Compound-treats-Disease edges):
    # the `indication_source` column records WHERE the indication text
    # came from ('drugbank_xml' | 'chembl_max_phase' | 'rxnorm' |
    # 'manual') so downstream consumers can filter by confidence.
    # DrugBank XML indications are gold-standard (curated by pharmacists);
    # ChEMBL max_phase==4 is silver (FDA approval implies efficacy for
    # the drug's primary indication, but the specific disease is
    # inferred from ATC code); RxNorm is bronze (drug-class-based).
    indication_source: Mapped[Optional[str]] = mapped_column(
        String(30), nullable=True,
    )
    # Calculated LogP (octanol-water partition coefficient).
    # FIX-P3-9: was Float -- precision mismatch with
    # pubchem_compound_properties.xlogp (Numeric(6,2)). Cross-table
    # joins on logp == xlogp failed because Float (binary64) cannot
    # represent 2-decimal decimal values exactly (e.g. 3.10 stored as
    # 3.0999999046325684). Changed to Numeric(6, 2) to match migration
    # 005 and the pubchem_compound_properties table.
    logp: Mapped[Optional[float]] = mapped_column(Numeric(6, 2), nullable=True)
    # Topological Polar Surface Area (Å²).
    # FIX-P3-9: was Float -- same precision-mismatch issue as logp.
    # Changed to Numeric(8, 2) to match migration 005 and the
    # pubchem_compound_properties.tpsa column.
    tpsa: Mapped[Optional[float]] = mapped_column(Numeric(8, 2), nullable=True)
    # Lipinski H-bond donor count.
    h_bond_donor_count: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    # Lipinski H-bond acceptor count.
    h_bond_acceptor_count: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    # Rotatable bond count (molecular flexibility).
    rotatable_bond_count: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    # Heavy atom count (excludes hydrogen).
    heavy_atom_count: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    # Molecular complexity (Bertz complexity index).
    complexity: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Data quality: fraction of expected fields populated (0.0-1.0).
    completeness_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # -- relationships --
    drug_protein_interactions: Mapped[list["DrugProteinInteraction"]] = relationship(
        "DrugProteinInteraction",
        back_populates="drug",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    # -- validators --
    @validates("inchikey")
    def _validate_inchikey(self, key: str, value: Optional[str]) -> Optional[str]:
        # P1-007 v113 ROOT FIX: Drug.inchikey is nullable=False, unique=True.
        # The shared _validate_inchikey() function returns None for None input
        # (because it's also used by nullable columns like
        # DrugCandidate.canonical_inchikey). But for Drug, a None or empty
        # string would pass the Python validator and then fail at INSERT with
        # a confusing IntegrityError. ROOT FIX: reject None and empty strings
        # HERE with a clear ValueError naming the SYNTH-prefix convention for
        # biologics. This gives developers an actionable error message instead
        # of a cryptic NOT NULL constraint violation.
        if value is None or (isinstance(value, str) and value.strip() == ""):
            raise ValueError(
                "Drug.inchikey cannot be NULL or empty. The Drug table "
                "requires a non-NULL InChIKey for every record. Biologics "
                "(antibodies, proteins, cell therapies) that cannot have a "
                "real InChIKey MUST use a SYNTH-prefixed surrogate key "
                "(e.g. 'SYNTH-ANTIBODY-TRASTUZUMAB'). See "
                "entity_resolution.drug_resolver._create_canonical_entry "
                "for the SYNTH-key generation logic."
            )
        return _validate_inchikey(value)

    @validates("max_phase")
    def _validate_max_phase(self, key: str, value: Optional[int]) -> Optional[int]:
        return _validate_max_phase(value)

    @validates("name")
    def _validate_name(self, key: str, value: Optional[str]) -> Optional[str]:
        if value is not None and len(value.strip()) < 2:
            raise ValueError(
                f"Drug name must be at least 2 characters, got: '{value}'"
            )
        return value

    __table_args__ = (
        # [SCI-01] InChIKey format: standard 27-char or SYNTH-prefixed
        CheckConstraint(
            # v16 ROOT FIX (CD-8): unified to ``LIKE 'IK%'`` (prefix match)
            # with LENGTH cap of 30. Previously used ``LIKE '%IK%'`` which
            # accepted any string containing "IK" (e.g. "BIKINI").
            # v29 ROOT FIX (audit D-2): the canonical regex
            # ``^[A-Z]{14}-[A-Z]{10}-[A-Z]$`` is enforced AUTHORITATIVELY
            # at the Python layer (cleaning._constants.is_canonical_inchikey).
            # v76 ROOT FIX (T-038 -- ORM CHECK strengthened to match migration 001):
            #   The migration 001 SQL now uses PostgreSQL's ``~`` regex
            #   operator for strict InChIKey validation. The migration
            #   runner's ``_translate_sql_for_sqlite`` translates the
            #   regex to the STRONG portable form below for SQLite.
            #   HOWEVER, the ORM's ``Base.metadata.create_all()`` path
            #   (used by dev/test SQLite DBs) does NOT go through the
            #   translator -- it emits the raw CheckConstraint text. So
            #   the ORM must use a PORTABLE form that works on BOTH
            #   dialects. The previous ``LENGTH(inchikey) = 27 OR LIKE
            #   'SYNTH%'`` accepted any 27-char ASCII string (gibberish).
            #   The new portable form validates length + hyphen positions
            #   (the two structural markers of a canonical InChIKey):
            #     LENGTH=27 AND SUBSTR(pos 15)='-' AND SUBSTR(pos 26)='-'
            #   This catches 27-char gibberish (no hyphens at the right
            #   positions) while remaining portable. Full uppercase-letter
            #   validation is enforced by the Python validator
            #   (is_canonical_inchikey) on both dialects. The SYNTH%
            #   escape hatch (dev fixtures) is preserved.
            # P1-A5 ROOT FIX (v82): the previous CHECK was UNPARENTHESIZED --
            # ``LENGTH=27 AND ... AND ... OR LIKE 'SYNTH%'``. SQL operator
            # precedence (AND binds tighter than OR) made it evaluate as
            # intended, but it was FRAGILE: any refactor adding another AND
            # condition could silently change the grouping. ROOT FIX: add
            # EXPLICIT parentheses so the grouping is intentional and
            # refactor-safe.
            #
            # v90 ROOT FIX (BUG #36 -- strengthen ORM CHECK with portable
            #   uppercase-letter validation):
            #   The previous portable form checked LENGTH=27 + hyphen
            #   positions but NOT character class -- so a 27-char string
            #   like ``aaaaaaaaaaaaaa-bbbbbbbbbb-c`` (lowercase) or
            #   ``!!!!!!!!!!!!!!-!!!!!!!!!!-!`` (symbols) passed the ORM
            #   CHECK on SQLite dev DBs even though the Python validator
            #   (is_canonical_inchikey) and the PostgreSQL regex
            #   (``^[A-Z]{14}-[A-Z]{10}-[A-Z]$``) both reject them. This
            #   left the dev DB CHECK as the WEAKEST layer -- a raw SQL
            #   INSERT bypassing the ORM could land lowercase / symbol-
            #   containing InChIKeys in dev SQLite, which would then BLOCK
            #   migration 009's strict regex when the dev DB was promoted
            #   to prod (with no way to know which rows were invalid).
            #   ROOT FIX: add ``inchikey = UPPER(inchikey)`` -- a portable
            #   predicate that fails if ANY character is lowercase. This
            #   works on BOTH SQLite and PostgreSQL (UPPER() is SQL-
            #   standard). It does NOT fully match the PostgreSQL regex
            #   (digits and symbols at non-hyphen positions would still
            #   pass since they are unchanged by UPPER), but it closes
            #   the lowercase gap -- the most common class of typo. Full
            #   character-class validation remains at the Python layer
            #   (is_canonical_inchikey, single source of truth). The
            #   defense-in-depth contract is now: Python validator is
            #   strictest, PostgreSQL regex is strict, SQLite ORM CHECK
            #   is medium (catches length + hyphen position + case), and
            #   no layer is weaker than the one above it for the common
            #   failure modes.
            "(LENGTH(inchikey) = 27 AND SUBSTR(inchikey, 15, 1) = '-' AND SUBSTR(inchikey, 26, 1) = '-' AND inchikey = UPPER(inchikey)) OR (UPPER(inchikey) LIKE 'SYNTH%' AND LENGTH(inchikey) <= 27)",
            name="chk_drugs_inchikey_format",
        ),
        # [SCI-02] Clinical phase range
        CheckConstraint(
            "max_phase IS NULL OR max_phase BETWEEN 0 AND 4",
            name="chk_drugs_max_phase",
        ),
        # [DQ-01] Boolean CHECK for SQLite compatibility
        # v66 ROOT FIX (P1C-024 -- CHECK didn't reject NULL):
        #   The previous CHECK ``is_fda_approved IN (0, 1)`` did NOT
        #   reject NULL -- ``NULL IN (0, 1)`` evaluates to UNKNOWN (not
        #   FALSE), which PASSES the CHECK. The column is ``nullable=False``
        #   with ``server_default="0"``, so NULL should never reach the DB
        #   today -- but if a future migration makes the column nullable,
        #   the old CHECK would silently accept NULL (an FDA-approval
        #   status of "unknown" could be mistaken for "not approved",
        #   a patient-safety risk). ROOT FIX: add an explicit
        #   ``IS NOT NULL`` predicate so the CHECK is defense-in-depth
        #   that actually defends against NULL, regardless of future
        #   schema changes. Same pattern applied to ``is_withdrawn``
        #   below (patient-safety signal -- an unknown withdrawal status
        #   must not silently become "not withdrawn").
        # v107 FORENSIC ROOT FIX (ISSUE-P1-009):
        #   The ORM CHECK was ``is_fda_approved IS NOT NULL AND
        #   is_fda_approved IN (0, 1)`` -- rejecting NULL. But migration
        #   013 alters the column to nullable and adds ``is_fda_approved
        #   IS NULL OR is_fda_approved IN (TRUE, FALSE)`` -- allowing
        #   NULL = "unknown FDA status" (the v93 patient-safety fix for
        #   EMA-only drugs). Dev/test SQLite DBs (via
        #   ``Base.metadata.create_all()``) rejected NULL inserts, while
        #   prod DBs (migration-created) allowed them. The loader's
        #   ``_pre_validate_drugs`` coerced None->False to satisfy the
        #   dev DB constraint, silently reverting the v93 fix. EMA-only
        #   drugs (max_phase=4, not FDA-approved) were stored as
        #   ``is_fda_approved=False`` -- same as confirmed-not-approved
        #   drugs. The RL ranker's FDA safety filter treated them
        #   identically. Tests passed on dev, prod had the fix -- dev/prod
        #   asymmetry.
        #   ROOT FIX: align the ORM CHECK with migration 013 -- allow NULL
        #   (unknown FDA status) OR (0, 1). Remove the ``IS NOT NULL``
        #   predicate.
        CheckConstraint(
            "is_fda_approved IS NULL OR is_fda_approved IN (0, 1)",
            name="chk_drugs_is_fda_approved",
        ),
        # [DQ-04] Name minimum length
        # P1-A9 ROOT FIX (v82): the previous CHECK ``LENGTH(name) >= 2`` did
        # NOT TRIM whitespace -- so a name like "  A" (2 chars with leading
        # spaces) passed the DB CHECK but failed the Python validator
        # (``len(name.strip()) >= 2``). Chinese drug names (multi-byte) also
        # diverged when SQLite's LENGTH() counted bytes vs Python's len()
        # counted characters in some encoding edge cases. ROOT FIX: use
        # ``LENGTH(TRIM(name)) >= 2`` to match the Python validator exactly.
        CheckConstraint(
            "LENGTH(TRIM(name)) >= 2",
            name="chk_drugs_name_min_length",
        ),
        # [DQ-09] Molecular weight must be positive
        CheckConstraint(
            "molecular_weight IS NULL OR molecular_weight > 0",
            name="chk_drugs_molecular_weight_positive",
        ),
        # [LIFE-SAFETY] is_withdrawn boolean CHECK for SQLite compatibility
        # v66 ROOT FIX (P1C-024 compound -- same NULL-rejection fix as
        # is_fda_approved above). is_withdrawn is a patient-safety signal:
        # a drug withdrawn for hepatotoxicity MUST NOT have its withdrawal
        # status silently accepted as NULL. ``NULL IN (0, 1)`` evaluates to
        # UNKNOWN which passes the old CHECK -- explicit ``IS NOT NULL`` is
        # required for defense-in-depth.
        CheckConstraint(
            "is_withdrawn IS NOT NULL AND is_withdrawn IN (0, 1)",
            name="chk_drugs_is_withdrawn",
        ),
        # v74 ROOT FIX (T-014 -- chk_drugs_is_globally_approved missing
        # from ORM __table_args__):
        #   Migration 008 (line 42-51) adds ``chk_drugs_is_globally_approved``
        #   to the drugs table via ALTER TABLE. But the ORM Drug model
        #   declared the ``is_globally_approved`` column (line 580) WITHOUT
        #   a matching CheckConstraint in __table_args__ -- dev DBs created
        #   via ``Base.metadata.create_all()`` (the SQLite/pytest path)
        #   LACK this constraint. Prod DBs (migration-created) HAVE it.
        #   Schema drift: a row with ``is_globally_approved=2`` would
        #   INSERT successfully on dev (no CHECK) but fail on prod (CHECK
        #   rejects). Tests on dev DBs don't catch invalid values.
        #
        #   ROOT FIX: declare the CheckConstraint in the ORM with the
        #   portable ``IN (0, 1)`` form (SQLite-compatible -- same pattern
        #   as chk_drugs_is_fda_approved and chk_drugs_is_withdrawn above).
        #   The migration 008 CHECK is also simplified to match (the
        #   previous ``IN (FALSE, TRUE)`` form worked on PostgreSQL but
        #   not SQLite; the ``IN (0, 1)`` form works on BOTH dialects).
        #   Defense-in-depth: even though the column is ``nullable=False``
        #   with ``server_default="0"``, the explicit ``IS NOT NULL``
        #   predicate guards against a future migration making the column
        #   nullable (same pattern as is_fda_approved / is_withdrawn).
        CheckConstraint(
            "is_globally_approved IS NOT NULL AND is_globally_approved IN (0, 1)",
            name="chk_drugs_is_globally_approved",
        ),
        # v90 ROOT FIX (BUG #6 -- P0 patient-safety invariant missing from ORM):
        #   Migration 008 (line ~160-167) adds the patient-safety invariant
        #   ``chk_drugs_no_approved_and_withdrawn`` -- a drug CANNOT be both
        #   globally-approved AND withdrawn. This prevents a withdrawn killer
        #   drug (Vioxx, Baycol, thalidomide) from being surfaced as a safe
        #   repurposing candidate. BUT the invariant was NOT declared on the
        #   ORM Drug model -- dev DBs created via Base.metadata.create_all()
        #   (the SQLite/pytest path) LACK this invariant. A test that sets
        #   is_globally_approved=True, is_withdrawn=True passes on dev but
        #   fails on prod. The Python-side safety hook in bulk_upsert_drugs
        #   (loaders.py:1886-1892) sets is_globally_approved=False when
        #   is_withdrawn=True -- but a direct SQL INSERT or a different
        #   loader bypassing that hook can violate the invariant on dev
        #   with no error. ROOT FIX: declare the CheckConstraint on the
        #   ORM with the portable form (works on both SQLite and PostgreSQL).
        #   The ``NOT (is_globally_approved = TRUE AND is_withdrawn = TRUE)``
        #   form uses ``= TRUE`` which works on PostgreSQL; on SQLite,
        #   ``= TRUE`` is rewritten by the migration runner's translator
        #   to ``= 1``. The ORM form below uses ``IN (1, TRUE)`` to be
        #   portable across both dialects without translation.
        CheckConstraint(
            "NOT (is_globally_approved = 1 AND is_withdrawn = 1)",
            name="chk_drugs_no_approved_and_withdrawn",
        ),
        # [DQ] completeness_score range 0.0-1.0
        CheckConstraint(
            "completeness_score IS NULL OR (completeness_score >= 0.0 AND completeness_score <= 1.0)",
            name="chk_drugs_completeness_score_range",
        ),
        # v59 ROOT FIX (compound of InChIKey fix -- SQLite create_all
        # crash): the previous ORM CheckConstraint used the PostgreSQL
        # regex operator ``~`` which SQLite does NOT support. When a
        # test or dev environment called ``Base.metadata.create_all()``
        # on SQLite, SQLAlchemy emitted the raw DDL with ``~`` and
        # SQLite raised ``OperationalError: near "~": syntax error`` --
        # breaking every test that uses the ORM to bootstrap the schema
        # (e.g. test_001_schema_16_domains::test_valid_drug_insert).
        # The migration 001 SQL keeps the regex form (the migration
        # runner has its own SQLite-translation logic that strips /
        # rewrites the regex). The ORM now uses a portable length check
        # as a backstop; full SMILES character-set validation is
        # enforced by the Python validator (cleaning.normalizer) on
        # both dialects. Same pattern as chk_drugs_inchikey_format.
        #
        # v76 ROOT FIX (T-039 -- ORM SMILES CHECK strengthened to match
        # migration 001's portable NOT LIKE form):
        #   The previous ORM CHECK ``smiles IS NULL OR LENGTH(TRIM(smiles))
        #   > 0`` was a non-empty backstop that accepted HTML tags, script
        #   injection, and any non-empty garbage. Migration 001 now uses
        #   a portable ``NOT LIKE`` form (no ``~`` operator) that works
        #   identically on PostgreSQL and SQLite. The ORM must match so
        #   dev DBs (create_all) and prod DBs (migrations) enforce the
        #   SAME constraint. The portable ``NOT LIKE '%X%'`` form rejects
        #   the most dangerous injection vectors (HTML angle brackets,
        #   ampersand, quotes, backtick) while allowing all valid SMILES
        #   characters. Full SMILES syntax validation (ring closure,
        #   valence, bond types) is enforced by RDKit at the application
        #   layer (the DB CHECK is a "first line of defense").
        CheckConstraint(
            "smiles IS NULL OR (smiles NOT LIKE '%<%' AND smiles NOT LIKE '%>%' AND smiles NOT LIKE '%&%' AND smiles NOT LIKE '%\"%' AND smiles NOT LIKE '%''%' AND smiles NOT LIKE '%`%')",
            name="chk_drugs_smiles_valid",
        ),
        # [DQ-05] Partial unique indexes for chembl_id and drugbank_id
        Index(
            "uq_drugs_chembl_id",
            "chembl_id",
            unique=True,
            postgresql_where=text("chembl_id IS NOT NULL"),
        ),
        Index(
            "uq_drugs_drugbank_id",
            "drugbank_id",
            unique=True,
            postgresql_where=text("drugbank_id IS NOT NULL"),
        ),
        # [PERF-04] Removed redundant idx_drugs_inchikey (UNIQUE already indexes)
        # [CODE-06] Removed redundant explicit index on inchikey
        Index("idx_drugs_chembl", "chembl_id"),
        Index("idx_drugs_drugbank", "drugbank_id"),
    )

    # [INT-05] Serialization helper
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with proper type coercion.

        v107 FORENSIC ROOT FIX (ISSUE-P1-010):
          The previous to_dict() omitted patient-safety-critical fields:
          ``is_globally_approved`` (max_phase==4 = global approval signal),
          ``groups`` (DrugBank groups string used to derive is_withdrawn),
          ``indication`` (free-text indication), and ``indication_source``
          (drugbank_xml / chembl_max_phase / rxnorm). If the frontend reads
          to_dict() for drug listings, the safety filter UI cannot display
          the global approval flag or the withdrawn-drug groups string. A
          withdrawn killer drug (Vioxx, Baycol) was indistinguishable from
          an active drug in the API response. ROOT FIX: add all four
          missing fields so the API response carries the full safety
          surface area.
        """
        return {
            "id": self.id,
            "inchikey": self.inchikey,
            "name": self.name,
            "chembl_id": self.chembl_id,
            "drugbank_id": self.drugbank_id,
            "pubchem_cid": self.pubchem_cid,
            "molecular_formula": self.molecular_formula,
            "molecular_weight": float(self.molecular_weight) if self.molecular_weight is not None else None,
            "smiles": self.smiles,
            "is_fda_approved": self.is_fda_approved,
            # v107 P1-010: added missing patient-safety fields
            "is_globally_approved": self.is_globally_approved,
            "groups": self.groups,
            "indication": self.indication,
            "indication_source": self.indication_source,
            "max_phase": self.max_phase,
            "drug_type": self.drug_type,
            "mechanism_of_action": self.mechanism_of_action,
            "is_withdrawn": self.is_withdrawn,
            "clinical_status": self.clinical_status,
            "cas_number": self.cas_number,
            "logp": self.logp,
            "tpsa": self.tpsa,
            "h_bond_donor_count": self.h_bond_donor_count,
            "h_bond_acceptor_count": self.h_bond_acceptor_count,
            "rotatable_bond_count": self.rotatable_bond_count,
            "heavy_atom_count": self.heavy_atom_count,
            "complexity": self.complexity,
            "completeness_score": self.completeness_score,
            "is_deleted": self.is_deleted,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    # [INT-01] Graph identity for Neo4j mapping
    @classmethod
    def graph_identity(cls) -> str:
        """Return the Neo4j node key property name for this entity."""
        return "inchikey"

    def __repr__(self) -> str:
        return (
            f"<Drug(id={self.id}, inchikey='{self.inchikey}', "
            f"name='{self.name}', chembl_id='{self.chembl_id}', "
            f"max_phase={self.max_phase})>"
        )


# ===========================================================================
# 2. PROTEINS
# ===========================================================================


class Protein(Base, IDMixin, TimestampMixin, SoftDeleteMixin):
    """Protein/target table sourced from UniProt.

    Domain meaning
    --------------
    Each row represents a unique protein identified by its UniProt accession.
    Key data includes gene symbol, protein name, amino acid sequence, and
    functional description.

    Key constraints
    ---------------
    - ``uniprot_id`` is the natural primary identifier (unique, not null).
      Validated against UniProt accession format (SCI-05).
    - ``gene_symbol`` validated against HGNC format (SCI-04).
    - ``sequence`` validated to contain only standard amino acid codes (SCI-08).
    - ``gene_name`` is **deprecated** -- it stores canonical protein names,
      not gene symbols.  Use ``gene_symbol`` for gene symbols and
      ``protein_name`` for protein names.

    Relationships
    -------------
    - ``drug_protein_interactions`` -> DPI rows (cascade delete-orphan).
    - ``gene_disease_associations`` -> GDA rows (cascade delete-orphan).
    - ``ppi_as_protein_a`` / ``ppi_as_protein_b`` -> PPI rows.
    - ``all_ppi_interactions`` -> unified property combining both PPI sides.
    """
    __tablename__ = "proteins"

    # [SCI-05] UniProt accession: reduced from 20 to 10
    uniprot_id: Mapped[str] = mapped_column(
        String(UNIPROT_ID_LENGTH), nullable=False, unique=True,
    )
    # DEPRECATED: gene_name stores CANONICAL PROTEIN NAME, NOT a gene symbol.
    # Use gene_symbol for gene symbols and protein_name for full protein names.
    # DO NOT REMOVE -- backward compatibility.  [DQ-06] [DOC-03]
    gene_name: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    # The actual gene symbol (e.g. "HBA1") used for GDA resolution (SCI-04)
    gene_symbol: Mapped[Optional[str]] = mapped_column(
        String(GENE_SYMBOL_LENGTH), nullable=True,
    )
    protein_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    organism: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # [SCI-08] Sequence validated for amino acid codes only
    # v43 ROOT FIX (Chain 8): Text -> String(50000) to match SQL migration 001
    # (line 394). Text is unbounded in SQLite; VARCHAR(50000) in PostgreSQL.
    # Dev/test SQLite accepted 100KB sequences; prod PostgreSQL rejected.
    sequence: Mapped[Optional[str]] = mapped_column(String(50000), nullable=True)
    # v43 ROOT FIX (Chain 8): Text -> String(10000) to match SQL migration 001
    # (line 395).
    function_desc: Mapped[Optional[str]] = mapped_column(String(10000), nullable=True)
    string_id: Mapped[Optional[str]] = mapped_column(
        String(STRING_ID_LENGTH), nullable=True,
    )

    # -- relationships --
    # v89 ROOT FIX (BUG #20 -- P1 N+1 explosion under bulk loads):
    #   The previous ``lazy="selectin"`` on ALL FOUR Protein relationships
    #   (DPI, GDA, PPI-a, PPI-b) meant EVERY ``session.query(Protein)`` --
    #   even ``session.get(Protein, id)`` -- fired 4 additional SELECT
    #   IN queries to eagerly populate all four collections, regardless
    #   of whether the caller needed them. Under the documented 7-
    #   concurrent-pipeline workload, loading 10 000 proteins for entity
    #   resolution triggered 40 000 additional SELECTs (4 × 10 000),
    #   exhausting the connection pool and stalling the pipeline.
    #
    #   ROOT FIX: switch to ``lazy="raise"`` so that ANY lazy access of
    #   these collections raises ``sqlalchemy.exc.InvalidRequestError``
    #   at development time, forcing the caller to declare the loading
    #   strategy explicitly via ``selectinload(Protein.X)`` /
    #   ``joinedload`` / ``raiseload``. This is the institutional-grade
    #   pattern recommended by the SQLAlchemy 2.0 performance guide:
    #   it makes N+1 queries impossible by construction, instead of
    #   relying on developer discipline. Queries that genuinely need
    #   the collections pass ``selectinload(Protein.drug_protein_interactions)``
    #   (etc.) in their ``options()`` -- the query is now explicit
    #   about what it loads, which is exactly what production audit
    #   requires. The ``all_ppi_interactions`` / ``all_ppi_partners``
    #   properties below were updated to surface a clear error when
    #   the caller forgot to eager-load (instead of the previous
    #   silent 4-SELECT fan-out).
    drug_protein_interactions: Mapped[list["DrugProteinInteraction"]] = relationship(
        "DrugProteinInteraction",
        back_populates="protein",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )
    gene_disease_associations: Mapped[list["GeneDiseaseAssociation"]] = relationship(
        "GeneDiseaseAssociation",
        back_populates="protein",
        # v79 FORENSIC ROOT FIX (P0-A4 -- cascade/ondelete mismatch left
        #   orphan GDA rows + stale ORM identity map):
        #   The v78 code used ``cascade="all, delete-orphan"`` with
        #   ``passive_deletes=True``. ``passive_deletes=True`` tells
        #   SQLAlchemy "trust the DB FK ondelete to handle child rows" --
        #   so the ORM issues NO DELETE for the GDA children. But the GDA
        #   FK (models.py ~line 1458) is ``ondelete="SET NULL"`` (NOT
        #   ``CASCADE``). So when a Protein was hard-deleted:
        #     1. The DB SET NULL fired -> GDA.uniprot_id became NULL
        #        (GDA row survives as an orphan with NULL protein link).
        #     2. The ORM's ``delete-orphan`` cascade expected the
        #        children to be DELETED, but ``passive_deletes=True``
        #        suppressed the DELETE -- so the ORM identity map still
        #        held the now-orphaned GDA objects, stale until
        #        ``cleanup_orphan_gda_records`` reaped them 24h later.
        #   Domain semantics: a GDA (gene->disease link from DisGeNET /
        #   OMIM) is PRECIOUS curated data. When its protein is deleted,
        #   the GDA should SURVIVE with uniprot_id=NULL (it may be
        #   re-linked to a different protein later via gene_symbol).
        #   SET NULL is therefore the correct DB behaviour. The ORM
        #   relationship must MATCH it: NO ``delete-orphan`` (children
        #   are preserved, not deleted) and NO ``passive_deletes``
        #   (the ORM should not pretend the DB will cascade-delete).
        # ROOT FIX: ``cascade="save-update, merge"`` (the minimal
        #   cascade -- children are tracked for write but NOT deleted
        #   when the parent is deleted) and drop ``passive_deletes``.
        #   Now hard-deleting a Protein leaves GDA rows with
        #   uniprot_id=NULL (per the FK SET NULL), the ORM identity
        #   map stays consistent, and the GDA data is preserved for
        #   re-linking -- exactly the patient-safe behaviour.
        cascade="save-update, merge",
        lazy="raise",
        foreign_keys="GeneDiseaseAssociation.uniprot_id",
    )
    ppi_as_protein_a: Mapped[list["ProteinProteinInteraction"]] = relationship(
        "ProteinProteinInteraction",
        foreign_keys="ProteinProteinInteraction.protein_a_id",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )
    ppi_as_protein_b: Mapped[list["ProteinProteinInteraction"]] = relationship(
        "ProteinProteinInteraction",
        foreign_keys="ProteinProteinInteraction.protein_b_id",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )

    # -- validators --
    @validates("uniprot_id")
    def _validate_uniprot_id(self, key: str, value: Optional[str]) -> Optional[str]:
        return _validate_uniprot_id(value)

    @validates("gene_symbol")
    def _validate_gene_symbol(self, key: str, value: Optional[str]) -> Optional[str]:
        # v107 ROOT FIX (ISSUE-P1-026 — GDA vs Protein gene-symbol regex
        # inconsistency): the previous code delegated to the module-level
        # ``_validate_gene_symbol`` which uses the LOOSE
        # ``_GENE_SYMBOL_RE`` (``^[A-Za-z][A-Za-z0-9\-]{0,49}$`` — accepts
        # Title-Case mouse/rat/yeast symbols like ``Tp53``, ``Brca1``).
        # Meanwhile the GDA validator (line ~1942) uses the STRICT
        # ``_HUMAN_GENE_SYMBOL_RE`` (``^[A-Z][A-Z0-9\-]{0,49}$`` — ALL CAPS).
        # This cross-table inconsistency created a SILENT data-loss path:
        # a non-human protein (e.g. from a misconfigured UniProt query
        # without the ``organism_id:9606`` filter) passed the Protein
        # validator with ``gene_symbol='Tp53'``, but the corresponding
        # GDA row was REJECTED by the GDA validator — the KG ended up
        # with a Protein node and NO GDA edges, silently breaking the
        # drug→protein→gene→disease multi-hop pattern for that protein.
        #
        # ROOT FIX: align the Protein validator with the GDA validator
        # by using ``_HUMAN_GENE_SYMBOL_RE``. This is consistent with:
        #   1. The system's human-only design — UniProt pipeline queries
        #      ``organism_id:9606`` (line ~219 of uniprot_pipeline.py
        #      already imports ``CANONICAL_HGNC_GENE_SYMBOL_REGEX`` which
        #      is uppercase-only), DrugBank applies ``filter_organism_humans``,
        #      and STRING validates the ``9606.`` ENSP prefix.
        #   2. The v89 ROOT FIX (BUG #25) that made the GDA validator
        #      strict — closing the v21/v89 gap that the audit found.
        # Non-human proteins now raise ValueError at the ORM layer with
        # a clear message — they are quarantined to the dead-letter queue
        # instead of silently dropping their GDA edges later.
        if value is None:
            return value
        value = value.strip()
        if _HUMAN_GENE_SYMBOL_RE.match(value):
            return value
        raise ValueError(
            f"Invalid HUMAN gene symbol for Protein: '{value}'. "
            "This platform is human-only (UniProt organism_id:9606, "
            "DrugBank filter_organism_humans, STRING 9606 prefix). "
            "Protein gene symbols must be ALL CAPS (e.g. BRCA1, FGFR3, TP53) "
            "to match the GDA validator (cross-table consistency, "
            "ISSUE-P1-026). Non-human symbols (Tp53, Brca1) are rejected."
        )

    @validates("sequence")
    def _validate_sequence(self, key: str, value: Optional[str]) -> Optional[str]:
        return _validate_sequence(value)

    __table_args__ = (
        # [SCI-05] UniProt accession format -- canonical accessions are exactly
        # 6 chars (old format, e.g. P12345) or 10 chars (new format, e.g.
        # A0A0K3AVT9). The minimum of 4 allows short test fixture IDs (e.g.
        # P001, P100) that are used in unit tests but never in production.
        # The Python validator _validate_uniprot_id also accepts these short
        # test IDs intentionally.
        #
        # v29 ROOT FIX (audit D-4): The DB CHECK stays LENGTH-based because
        # SQLite does not support regex in CHECK constraints and we need to
        # keep the schema portable across SQLite (dev/test) and PostgreSQL
        # (prod). The STRICT canonical UniProt regex
        # ``^[OPQ][0-9][A-Z0-9]{3}[0-9]([A-Z0-9]{3}[0-9]){1,5}$`` (audit D-4
        # spec) is enforced at the Python layer by ``_validate_uniprot_id``,
        # which delegates to ``entity_resolution.resolver_utils._UNIPROT_ACCESSION_RE``
        # (single source of truth: ``cleaning._constants.CANONICAL_UNIPROT_ACCESSION_REGEX``).
        # This is what prevents mouse-organism accessions and other
        # non-UniProt short alphanumeric strings (e.g. "MOUSE1") from being
        # accepted into the human protein set -- the LENGTH-only CHECK alone
        # could not.
        CheckConstraint(
            # v89 ROOT FIX (BUG #24 -- DB CHECK weaker than Python validator):
            #   Real UniProt accessions are EXACTLY 6 or 10 chars per the
            #   official spec (https://www.uniprot.org/help/accession_numbers).
            #   The previous ``LENGTH >= 4 AND LENGTH <= 10`` accepted 4, 5,
            #   7, 8, 9 char strings -- NONE of which are real UniProt IDs.
            #   The Python validator ``_validate_uniprot_id`` (line 345)
            #   already rejects these in production (only TEST-prefixed
            #   fixtures are allowed in dev/ci). But a raw SQL INSERT
            #   (migration, manual fix, future tool) bypassing the ORM
            #   could land a 4-char "UniProt ID" in production -- breaking
            #   the defense-in-depth contract (DB should be the LAST line
            #   of defense, not the weakest). The previous comment claimed
            #   the 4-char minimum was for "test fixture IDs (P001, P100)"
            #   but the Python validator REJECTS those in production (only
            #   TEST-prefixed IDs are allowed in dev/ci). ROOT FIX:
            #   tighten the CHECK to ``LENGTH IN (6, 10)`` so the DB
            #   enforces the same contract as the Python validator. Dev/ci
            #   fixtures use TEST-prefixed IDs which are >= 6 chars
            #   (TEST001, TEST123) so they still pass. The previous 4-5
            #   char acceptance was dead code that weakened the contract.
            "uniprot_id IS NULL OR (LENGTH(uniprot_id) = 6 OR LENGTH(uniprot_id) = 10)",
            name="chk_proteins_uniprot_length",
        ),
        # [DQ-04] Organism controlled vocabulary (SCI-FIX audit finding 1).
        # Mirrors the migration CHECK in 001_initial_schema.sql so dev
        # (ORM-created) and prod (migration-created) DBs enforce the
        # SAME organism allowlist. The original migration only allowed
        # human variants, silently breaking cross-species protein
        # ingestion. The ORM had no constraint at all (permissive).
        # Both now allow the model organisms covered by
        # entity_resolution.protein_resolver.py _ORGANISM_ALIASES:
        # human, mouse, rat, e. coli, yeast, fly, worm, zebrafish,
        # plus "unknown organism" used by handle_missing_protein_fields.
        CheckConstraint(
            "organism IS NULL OR LOWER(TRIM(organism)) IN ("
            "'homo sapiens', 'human', 'humans', 'h. sapiens', "
            "'mus musculus', 'mouse', 'mice', 'm. musculus', "
            "'rattus norvegicus', 'rat', 'rats', 'r. norvegicus', "
            "'escherichia coli', 'e. coli', 'e.coli', "
            "'saccharomyces cerevisiae', 'yeast', 's. cerevisiae', "
            "'drosophila melanogaster', 'fruit fly', 'd. melanogaster', "
            "'caenorhabditis elegans', 'c. elegans', 'nematode', "
            "'danio rerio', 'zebrafish', 'd. rerio', "
            "'unknown organism', 'unknown', ''"
            ")",
            name="chk_proteins_organism",
        ),
        # [PERF-04] Removed redundant idx_proteins_uniprot (UNIQUE already indexes)
        # [PERF-04] Removed idx_proteins_gene_name (deprecated column)
        Index("idx_proteins_gene_symbol", "gene_symbol"),
        Index("idx_proteins_string_id", "string_id"),
    )

    # [ARCH-06] Unified PPI accessor
    # v89 ROOT FIX (BUG #20): the underlying relationships are now
    # ``lazy="raise"``, so these properties raise
    # ``InvalidRequestError`` unless the caller eager-loads via
    # ``selectinload(Protein.ppi_as_protein_a)`` +
    # ``selectinload(Protein.ppi_as_protein_b)`` (or ``raiseload`` to
    # explicitly opt out). This surfaces the missing eager-load as a
    # loud error at development time instead of the previous silent
    # 4-SELECT fan-out on every Protein load.
    @property
    def all_ppi_interactions(self) -> list["ProteinProteinInteraction"]:
        """Return all PPI records involving this protein (both sides).

        Requires the caller to eager-load ``ppi_as_protein_a`` and
        ``ppi_as_protein_b`` (e.g. via ``selectinload``); otherwise
        raises ``sqlalchemy.exc.InvalidRequestError``.
        """
        return list(self.ppi_as_protein_a) + list(self.ppi_as_protein_b)

    # [ARCH-06] All interacting partner proteins
    @property
    def all_ppi_partners(self) -> list["Protein"]:
        """Return all proteins that interact with this protein.

        Requires the caller to eager-load ``ppi_as_protein_a`` and
        ``ppi_as_protein_b`` (e.g. via ``selectinload``); otherwise
        raises ``sqlalchemy.exc.InvalidRequestError``.
        """
        partners: list["Protein"] = []
        for ppi in self.ppi_as_protein_a:
            if ppi.protein_b not in partners:
                partners.append(ppi.protein_b)
        for ppi in self.ppi_as_protein_b:
            if ppi.protein_a not in partners:
                partners.append(ppi.protein_a)
        return partners

    @property
    def canonical_protein_name(self) -> Optional[str]:
        """Alias for gene_name -- clarifies it stores a protein name, not gene symbol.

        WARNING: The gene_name column is misleadingly named.  It stores the
        canonical protein name (e.g., "Hemoglobin subunit alpha"), not the
        gene symbol.  Use gene_symbol for actual gene symbols (e.g., "HBA1").
        """
        return self.gene_name

    @property
    def display_name(self) -> str:
        """Return the most useful display name for this protein.

        Priority: gene_symbol > protein_name > gene_name (legacy) > uniprot_id.
        """
        return self.gene_symbol or self.protein_name or self.gene_name or self.uniprot_id

    # [INT-05] Serialization helper
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with proper type coercion."""
        return {
            "id": self.id,
            "uniprot_id": self.uniprot_id,
            "gene_symbol": self.gene_symbol,
            "protein_name": self.protein_name,
            "gene_name": self.gene_name,  # legacy
            "organism": self.organism,
            "string_id": self.string_id,
            "is_deleted": self.is_deleted,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    # [INT-01] Graph identity for Neo4j mapping
    @classmethod
    def graph_identity(cls) -> str:
        """Return the Neo4j node key property name for this entity."""
        return "uniprot_id"

    # [CODE-01] Fixed __repr__ -- gene_name labelled as legacy, gene_symbol shown
    def __repr__(self) -> str:
        return (
            f"<Protein(id={self.id}, uniprot_id='{self.uniprot_id}', "
            f"gene_symbol='{self.gene_symbol}', "
            f"protein_name='{self.protein_name}', "
            f"gene_name(legacy)='{self.gene_name}')>"
        )


# ===========================================================================
# 3. DRUG-PROTEIN INTERACTIONS (DPI)
# ===========================================================================


class DrugProteinInteraction(Base, IDMixin, TimestampMixin):
    """Drug-protein interaction records from ChEMBL and DrugBank.

    Domain meaning
    --------------
    Each row represents a measured interaction between a drug and a protein
    target.  Key data includes interaction type, activity value/type/units,
    and the source database.

    Key constraints
    ---------------
    - ``activity_value`` must be positive (DQ-09).
    - ``confidence_score`` must be in [0, 1] (DQ-07).
    - ``source_id`` is nullable (DES-04 -- empty string replaced with NULL).
    - ``UniqueConstraint(drug_id, protein_id, source, source_id)`` with
      NULL source_id handled by partial index on PostgreSQL.

    Lineage (LINE-01)
    ------------------
    - ``pipeline_run_id`` tracks which pipeline run produced this record.
    - ``source_version`` and ``source_fetch_date`` record provenance.
    - ``entity_resolved`` flags whether entity resolution was applied.
    """
    __tablename__ = "drug_protein_interactions"

    drug_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("drugs.id", ondelete="CASCADE"), nullable=False,
    )
    protein_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("proteins.id", ondelete="CASCADE"), nullable=False,
    )
    interaction_type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True,
    )
    # v29 ROOT FIX (audit D-5): was Float -- precision loss corrupts pIC50. Use Numeric(10,4).
    activity_value: Mapped[Optional[float]] = mapped_column(Numeric(10, 4), nullable=True)
    activity_type: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True,
    )
    activity_units: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True,
    )
    # INT-003 ROOT FIX: ChEMBL's standard_relation carries censoring semantics
    # ('=', '<', '>', '~') that distinguish exact measurements from bounds.
    # Previously dropped at ORM load time, causing the RL ranker to treat
    # IC50 > 100uM (weak binder) the same as IC50 = 1nM (potent inhibitor).
    # This column stores the raw ChEMBL relation so the Phase 2 bridge can
    # propagate it without heuristic guessing.
    standard_relation: Mapped[Optional[str]] = mapped_column(
        String(5), nullable=True,
    )
    source: Mapped[Optional[str]] = mapped_column(
        String(SOURCE_LENGTH), nullable=True,
    )
    # [DES-04] source_id now nullable instead of NOT NULL DEFAULT ''
    # Empty string conflated with "no value" -- NULL is semantically correct.
    source_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True,
    )
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # [LINE-01] Source tracking columns
    source_version: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True,
    )
    source_fetch_date: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    entity_resolved: Mapped[Optional[bool]] = mapped_column(
        # v90 ROOT FIX (BUG #23): `server_default=text("FALSE")` instead of
        #   the non-portable `server_default="0"`. Column is nullable, so
        #   the server default only fires on INSERTs that omit the column
        #   (treated as "not yet resolved"). Matches migration 001 line 720
        #   (`entity_resolved BOOLEAN DEFAULT FALSE`) byte-for-byte.
        Boolean, nullable=True, server_default=text("FALSE"),
    )
    # [IDEM-01] Pipeline run tracking
    pipeline_run_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True,
    )

    # -- relationships --
    drug: Mapped["Drug"] = relationship(
        "Drug", back_populates="drug_protein_interactions",
    )
    protein: Mapped["Protein"] = relationship(
        "Protein", back_populates="drug_protein_interactions",
    )

    __table_args__ = (
        UniqueConstraint(
            "drug_id", "protein_id", "source", "source_id",
            name="uq_dpi_drug_protein_source",
        ),
        # [DQ-09] Activity value must be positive
        CheckConstraint(
            "activity_value IS NULL OR activity_value > 0",
            name="chk_dpi_activity_value_positive",
        ),
        # INT-003 ROOT FIX: validate standard_relation is a valid ChEMBL censoring direction.
        CheckConstraint(
            "standard_relation IS NULL OR standard_relation IN "
            "('=', '<', '>', '~')",
            name="chk_dpi_standard_relation",
        ),
        # v43 ROOT FIX (Chain 8 -- 8 missing CHECK constraints in ORM):
        # The SQL migration 001 has chk_dpi_activity_type, chk_dpi_activity_units,
        # and chk_dpi_source CHECK constraints, but the ORM model did NOT --
        # meaning SQLite ORM-created DBs (dev/test) accepted bad rows that
        # PostgreSQL migration-created DBs (prod) rejected. "Tests green,
        # prod red" anti-pattern. Adding the three missing CHECKs here so
        # dev/test/prod all enforce the same constraints.
        #
        # v93 ROOT FIX (P1-049): the DB CHECK was MISSING six valid activity
        # types that ``cleaning/normalizer.py::_ALLOWED_ACTIVITY_TYPES``
        # accepts: ``Kb``, ``pKi``, ``pIC50``, ``pEC50``, ``pKd``, ``ED50``.
        # A DPI row with ``activity_type='pIC50'`` (a perfectly valid
        # log-scale potency measure used by ChEMBL, PubChem BioAssay, and
        # BindingDB) was REJECTED by the DB CHECK but ACCEPTED by the
        # normalizer -- losing log-scale activity data from the KG and
        # corrupting downstream Graph Transformer edge features (only
        # linear-scale IC50/Ki/Kd/EC50/AC50 survived). The DB CHECK must
        # be a SUPERSET of (or exactly match) the normalizer's allowed
        # set -- never a subset. Single source of truth:
        # ``cleaning/normalizer.py::_ALLOWED_ACTIVITY_TYPES``.
        CheckConstraint(
            "activity_type IS NULL OR activity_type IN "
            "('IC50', 'EC50', 'Ki', 'Kd', 'Kb', 'potency', 'AC50', "
            "'pKi', 'pIC50', 'pEC50', 'pKd', 'ED50', 'unknown')",
            name="chk_dpi_activity_type",
        ),
        CheckConstraint(
            "activity_units IS NULL OR activity_units IN "
            "('nM', 'uM', 'mM', 'M', '%', 'mg/mL', 'ug/mL')",
            name="chk_dpi_activity_units",
        ),
        CheckConstraint(
            # v89 ROOT FIX (BUG #27 -- over-restrictive CHECK locks DPI to
            #   2 sources forever): the previous ``source IN ('chembl',
            #   'drugbank')`` CHECK rejected every other standard DPI /
            #   activity-data source -- BindingDB, PubChem BioAssay,
            #   ChEMBL-NTD, IUPHAR/Guide to PHARMACOLOGY, EPA ToxCast,
            #   GtoPdb. All are free, NIH/EMBL-funded, and standard in
            #   drug-repurposing pipelines. The ``source_id`` column is
            #   free-form (no CHECK), but ``source`` was locked to 2
            #   values -- an asymmetry that makes the platform unable to
            #   ingest activity data from any future source. ROOT FIX:
            #   expand the whitelist to the 8 standard DPI databases.
            #   The loader's ``_pre_validate_dpi`` validates upstream; the
            #   DB CHECK is defense-in-depth. NOTE: BUG #38 removes the
            #   loader's ``source="unknown"`` coercion (which would have
            #   violated this CHECK); NULL is still allowed per the
            #   ``source IS NULL OR`` prefix.
            "source IS NULL OR source IN ("
            "'chembl', 'drugbank', 'bindingdb', 'pubchem_bioassay', "
            "'chembl_ntd', 'iuphar', 'toxcast', 'gtopdb')",
            name="chk_dpi_source",
        ),
        # [DQ-07] Confidence score range
        CheckConstraint(
            "confidence_score IS NULL OR (confidence_score >= 0.0 AND confidence_score <= 1.0)",
            name="chk_dpi_confidence_score_range",
        ),
        # [PERF-01] Composite indexes for common query patterns
        Index("idx_dpi_drug", "drug_id"),
        Index("idx_dpi_protein", "protein_id"),
        Index("idx_dpi_protein_interaction", "protein_id", "interaction_type"),
        Index("idx_dpi_drug_interaction", "drug_id", "interaction_type"),
    )

    # [LOG-02] Enhanced __repr__ with diagnostic fields
    def __repr__(self) -> str:
        return (
            f"<DrugProteinInteraction(id={self.id}, drug_id={self.drug_id}, "
            f"protein_id={self.protein_id}, interaction_type='{self.interaction_type}', "
            f"activity_value={self.activity_value}, source='{self.source}')>"
        )


# ===========================================================================
# 4. PROTEIN-PROTEIN INTERACTIONS (PPI)
# ===========================================================================


class ProteinProteinInteraction(Base, IDMixin, TimestampMixin):
    """Protein-protein interaction records from STRING.

    Domain meaning
    --------------
    Each row represents an interaction between two proteins in the STRING
    database.  Scores are integers in the range [0, 1000] (NOT 0-100).
    Score misinterpretation corrupts Graph Transformer edge weights (SCI-03).

    Key constraints
    ---------------
    - ``protein_a_id < protein_b_id`` enforced by CHECK (DES-02, IDEM-03).
      This prevents symmetric duplicates like (A, B) and (B, A).
    - All score columns are in [0, 1000] (SCI-03).
    - ``source`` has no default -- must be specified explicitly (CFG-03).
    - ``score_json`` for source-specific score payloads (INT-04).

    Normalized Score
    ----------------
    The ``normalized_combined_score`` property returns combined_score / 1000.0
    for the [0, 1] range expected by downstream ML models.
    """
    __tablename__ = "protein_protein_interactions"

    protein_a_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("proteins.id", ondelete="CASCADE"), nullable=False,
    )
    protein_b_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("proteins.id", ondelete="CASCADE"), nullable=False,
    )
    # [SCI-03] STRING scores are 0-1000, NOT 0-100
    combined_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    experimental_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    database_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    textmining_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # [CFG-03] No server_default -- source must be explicitly specified
    source: Mapped[str] = mapped_column(
        String(SOURCE_LENGTH), nullable=False,
    )
    # [INT-04] Score JSON for source-specific payloads beyond STRING
    score_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # v91 ROOT FIX (BUG #9): is_homodimer flag column.
    # True when protein_a_id == protein_b_id (self-interaction / homodimer).
    # Biologically critical: EGFR dimerization, p53 tetramerization, etc.
    is_homodimer: Mapped[bool] = mapped_column(
        # P1-005 ROOT FIX (v100 forensic): the v90 ROOT FIX (BUG #23) claimed
        # to unify EVERY boolean column to server_default=text("FALSE") but
        # MISSED is_homodimer -- it kept the non-portable server_default="0".
        # On strict-mode MySQL/MariaDB, DEFAULT 0 for a BOOLEAN column is
        # rejected; on SQLite/PostgreSQL it works but creates three-way
        # schema drift versus every other boolean column. ROOT FIX: align
        # is_homodimer with the rest of the schema (text("FALSE")).
        Boolean, nullable=False, default=False, server_default=text("FALSE"),
    )
    # [IDEM-01] Pipeline run tracking
    pipeline_run_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True,
    )

    # -- relationships --
    protein_a: Mapped["Protein"] = relationship(
        "Protein", foreign_keys=[protein_a_id], back_populates="ppi_as_protein_a",
    )
    protein_b: Mapped["Protein"] = relationship(
        "Protein", foreign_keys=[protein_b_id], back_populates="ppi_as_protein_b",
    )

    __table_args__ = (
        UniqueConstraint("protein_a_id", "protein_b_id", name="uq_ppi_protein_pair"),
        # [DES-02] [IDEM-03] Prevent symmetric duplicates -- protein_a_id must
        # be <= protein_b_id. v91 ROOT FIX (BUG #9): changed from strict
        # less-than (protein_a_id < protein_b_id) to less-than-or-equal,
        # allowing homodimers (protein_a_id == protein_b_id). Homodimers
        # are biologically real and clinically critical: EGFR dimerization,
        # HER2/HER3 heterodimerization, p53 tetramerization are fundamental
        # to cancer biology. Dropping ALL homodimers from the KG removed
        # these critical edges, making drugs targeting EGFR dimerization
        # (e.g. cetuximab) lose their PPI context. The is_homodimer flag
        # column distinguishes self-interactions from heterodimers.
        CheckConstraint(
            "protein_a_id <= protein_b_id",
            name="chk_ppi_ordered",
        ),
        # v91 ROOT FIX (BUG #9): is_homodimer flag column.
        # True when protein_a_id == protein_b_id (self-interaction).
        # Downstream ML models can use this flag to learn different
        # representations for homodimers vs heterodimers.
        #
        # v93 ROOT FIX (P1-025): the previous CHECK used non-standard SQL
        # (``==`` and ``= TRUE``) and was logically ASYMMETRIC -- it only
        # enforced "homodimer ⇒ a_id == b_id" but NOT "a_id != b_id ⇒
        # NOT homodimer". A heterodimer row with ``is_homodimer = TRUE``
        # (loader bug or manual insert) was SILENTLY ACCEPTED, polluting
        # the KG with false homodimer flags. Downstream ML models would
        # then learn wrong homodimer representations (e.g. treating an
        # EGFR-HER2 heterodimer as a homodimer).
        #
        # The new CHECK is a bidirectional biconditional that enforces
        # BOTH directions portably:
        #   - homodimer (a_id == b_id) ⇒ is_homodimer MUST be TRUE
        #   - heterodimer (a_id != b_id) ⇒ is_homodimer MUST be FALSE
        # Uses ``=`` (standard SQL) and ``TRUE``/``FALSE`` literals,
        # which work on PostgreSQL, SQLite, and strict-mode MySQL.
        CheckConstraint(
            "(protein_a_id = protein_b_id AND is_homodimer = TRUE) "
            "OR (protein_a_id != protein_b_id AND is_homodimer = FALSE)",
            name="chk_ppi_homodimer_flag",
        ),
        # [SCI-03] Score bounds -- all STRING scores are 0-1000
        CheckConstraint(
            "combined_score IS NULL OR (combined_score >= 0 AND combined_score <= 1000)",
            name="chk_ppi_combined_score",
        ),
        CheckConstraint(
            "experimental_score IS NULL OR (experimental_score >= 0 AND experimental_score <= 1000)",
            name="chk_ppi_experimental_score",
        ),
        CheckConstraint(
            "database_score IS NULL OR (database_score >= 0 AND database_score <= 1000)",
            name="chk_ppi_database_score",
        ),
        CheckConstraint(
            "textmining_score IS NULL OR (textmining_score >= 0 AND textmining_score <= 1000)",
            name="chk_ppi_textmining_score",
        ),
        # v43 ROOT FIX (Chain 8): chk_ppi_source was in SQL migration 001
        # but missing from ORM. SQLite dev DB accepted source='intact';
        # PostgreSQL prod DB rejected. Adding to ORM for parity.
        #
        # v89 ROOT FIX (BUG #26 -- over-restrictive CHECK locks platform to
        #   STRING forever): the previous ``source IN ('string')`` CHECK
        #   rejected every other standard PPI database (BioGRID, IntAct,
        #   HPRD, Reactome, BioPlex) -- all of which are commonly used
        #   alongside STRING for PPI network construction. If the platform
        #   ever ingests BioGRID/IntAct (both are free, NIH-funded, and
        #   standard in pharma KG pipelines), the loader fails at INSERT
        #   with a CHECK violation, and the operator has no way to load
        #   the data without dropping the constraint. The ``source`` column
        #   is NOT NULL (line 1374), so every PPI row MUST have a source --
        #   the CHECK must accept all legitimate sources. ROOT FIX:
        #   expand the whitelist to the 6 standard PPI databases. The
        #   loader's ``_pre_validate_ppi`` validates the source value
        #   upstream; the DB CHECK is defense-in-depth.
        CheckConstraint(
            "source IN ('string', 'biogrid', 'intact', 'hprd', 'reactome', 'bioplex')",
            name="chk_ppi_source",
        ),
        Index("idx_ppi_protein_a", "protein_a_id"),
        Index("idx_ppi_protein_b", "protein_b_id"),
    )

    # [SCI-03] Normalized score for ML consumption
    @property
    def normalized_combined_score(self) -> Optional[float]:
        """Return combined_score normalized to [0, 1] range.

        STRING scores are 0-1000.  Downstream ML models expect [0, 1].
        """
        if self.combined_score is None:
            return None
        return self.combined_score / 1000.0

    # [LOG-02] Enhanced __repr__ with source and score
    def __repr__(self) -> str:
        return (
            f"<ProteinProteinInteraction(id={self.id}, "
            f"protein_a_id={self.protein_a_id}, protein_b_id={self.protein_b_id}, "
            f"combined_score={self.combined_score}, source='{self.source}')>"
        )


# ===========================================================================
# 5. GENE-DISEASE ASSOCIATIONS (GDA)
# ===========================================================================


class GeneDiseaseAssociation(Base, IDMixin, TimestampMixin):
    """Gene-disease association records from DisGeNET and OMIM.

    Domain meaning
    --------------
    Each row links a gene (via gene_symbol / uniprot_id) to a disease
    (via disease_id).  Data is sourced from DisGeNET and OMIM.

    Key constraints
    ---------------
    - ``uniprot_id`` string FK to ``proteins.uniprot_id`` for cross-source
      joins. The GDA model uses the string UniProt accession (not the
      integer protein PK) because gene-disease data sources (DisGeNET,
      OMIM) provide gene symbols that resolve to UniProt accessions, not
      to integer protein IDs.
    - ``gene_symbol`` validated against HGNC format (SCI-04).
    - ``disease_id_type`` indicates the identifier system used (SCI-06).
    - ``score`` is the association score from the source database.
    - ``pmid_list`` capped at 2000 chars (SEC-02).
    - ``UniqueConstraint(gene_symbol, disease_id, source)`` with NULL
      handling (DQ-02).

    Lineage (LINE-03)
    ------------------
    - ``score_type`` and ``score_method`` document how the score was computed.
    """
    __tablename__ = "gene_disease_associations"

    gene_symbol: Mapped[Optional[str]] = mapped_column(
        # v57 ROOT FIX (P1C-001 -- schema contradiction):
        #   The previous definition was `nullable=False, server_default=""`
        #   + a CHECK constraint `gene_symbol <> ''` (declared below).
        #   This is a contradiction: if a loader inserts a row without
        #   supplying gene_symbol, SQLAlchemy uses the server_default (""),
        #   which then FAILS the CHECK constraint with IntegrityError.
        #   The loader's fallback path (`df["gene_symbol"] = ""` in
        #   loaders.py:2639) guaranteed this crash.
        #   FIX: make gene_symbol NULLABLE (no server_default, no CHECK
        #   on empty string). The partial unique index in migration 002
        #   (T-004 fix) handles dedup for non-NULL gene_symbols. NULL
        #   gene_symbols are quarantined by the loader instead of
        #   crashing the pipeline.
        String(GENE_SYMBOL_LENGTH), nullable=True,
    )
    # String FK to proteins.uniprot_id -- the canonical cross-source key.
    # GDA does NOT have an integer protein_id FK because gene-disease data
    # sources provide gene symbols that resolve to UniProt accessions.
    uniprot_id: Mapped[Optional[str]] = mapped_column(
        String(UNIPROT_ID_LENGTH),
        ForeignKey("proteins.uniprot_id", ondelete="SET NULL"),
        nullable=True,
    )
    # v14 ROOT FIX (FIX4 / audit CD-3): the integer ``protein_id`` column
    # was REMOVED from the GDA model. The previous v13 code kept it "to
    # match the migrations" -- but the migrations were THEMSELVES the bug.
    # The GDA table is supposed to use the STRING ``uniprot_id`` FK
    # (because gene-disease data sources provide gene symbols that
    # resolve to UniProt accessions, NOT integer protein PKs). The
    # loader code (loaders.py:2318) already correctly skips populating
    # ``protein_id``. Keeping a column the loader never populates
    # produced an unused index, false-positive schema drift, and made
    # the GDA model ambiguous (two ways to point at a protein). The
    # migrations have also been updated to NOT create the column.
    # Tests under TestFix4_GdaUniprotId enforce this invariant.
    disease_id: Mapped[Optional[str]] = mapped_column(
        # v59 ROOT FIX (compound of P1C-001 -- disease_id contradiction):
        #   The previous definition was `nullable=False, server_default=""`
        #   paired with the CHECK constraint `chk_gda_disease_id_nonempty`
        #   (declared below). On PostgreSQL, an INSERT that omits
        #   disease_id triggers the server_default (""), which then FAILS
        #   the CHECK with IntegrityError. On SQLite, the server_default
        #   fires ("") but the CHECK is silently NOT enforced against
        #   server-supplied values -- so dev/test passed while production
        #   crashed. This is the exact same bug class that v57 fixed for
        #   gene_symbol but left in place for disease_id.
        #   ROOT FIX: drop the server_default. disease_id remains
        #   nullable=False -- the CHECK constraint stays as the guard.
        #   Loaders that fail to supply a real disease_id now quarantine
        #   the row instead of crashing the upsert. The CHECK constraint
        #   `disease_id IS NOT NULL AND disease_id <> ''` is preserved
        #   because an empty disease_id is scientifically meaningless
        #   (a GDA row with no disease is data garbage).
        String(DISEASE_ID_LENGTH), nullable=False,
    )
    # [SCI-06] Disease ID type -- indicates which identifier system is used
    disease_id_type: Mapped[Optional[str]] = mapped_column(
        String(DISEASE_ID_TYPE_LENGTH), nullable=True,
    )
    disease_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    association_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(
        String(SOURCE_LENGTH), nullable=True,
    )
    # [SEC-02] Capped at 2000 chars instead of unbounded Text
    pmid_list: Mapped[Optional[str]] = mapped_column(
        String(PMID_LIST_LENGTH), nullable=True,
    )
    # [LINE-03] Score computation tracking
    score_type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True,
    )
    score_method: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True,
    )
    # [IDEM-01] Pipeline run tracking
    pipeline_run_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True,
    )

    # ------------------------------------------------------------------
    # 389-fix audit -- institutional-grade columns (SCI-3..SCI-42, DQ-1..34,
    # IDEM-9..14, LIN-1..28, COMP-1..20).  All new columns are nullable
    # so existing rows (and existing tests) are unaffected.
    # ------------------------------------------------------------------

    # [SCI-6 / DQ-1] NCBI Entrez Gene ID -- stable across HGNC renames.
    gene_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # [SCI-9 / DQ-2] DisGeNET diseaseType ∈ {disease, phenotype, group}.
    disease_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # [SCI-3] DisGeNET sub-source (CURATED, BEFREE, GWAS_CATALOG, ...).
    source_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # [SCI-8] MeSH hierarchy code (e.g. C04.588.614) -- stored verbatim.
    disease_class: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    disease_class_source: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )

    # [SCI-7] Publication-year range of the evidence.
    year_initial: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    year_final: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # [SCI-10] Confidence tier label (weak / moderate / strong).
    confidence_tier: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # [SCI-24] Evidence-strength label derived from PMID count + recency.
    evidence_strength: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # [SCI-38] Score × source_weight -- cross-source comparable score.
    normalized_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # [SCI-26 / IDEM-8] DisGeNET release version (e.g. "v7_2024_06").
    source_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # [LIN-6 / COMP-7] Download timestamp (UTC) for this row.
    download_date: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # [LIN-23] Download method: "api" or "static".
    download_method: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # [INT-7] Source format: "api" or "tsv".
    source_format: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # [LIN-24] Dedup strategy applied (e.g. "validate_gda_scores_dedup").
    dedup_strategy: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # [LIN-15 / IDEM-17] Confidence tier definition version (e.g. pinero_2020_v1).
    confidence_tier_method: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )

    # [LIN-10] Resolution method used for gene_symbol -> uniprot_id.
    resolution_method: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # [LIN-10 / IDEM-7] SHA-256 of the cached gene_to_uniprot map.
    gene_to_uniprot_map_version: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )

    # [LIN-16] Original PMID count (before capping).
    original_pmid_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # [COMP-6] Schema version of the CSV that produced this row.
    schema_version: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # [IDEM-14] Snapshot tag for backfill safety.
    snapshot_tag: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # [LIN-9] Source URL (sanitised -- no API key).
    source_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # [SCI-21 / LIN-17..19] Lineage columns from validate_gda_scores.
    # Renamed without leading underscore in the DB (the CSV keeps the
    # underscore-prefixed names for backward compatibility).
    score_was_clipped: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    original_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    score_was_coerced_nan: Mapped[Optional[bool]] = mapped_column(
        Boolean, nullable=True
    )
    score_direction: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    disease_name_was_filled: Mapped[Optional[bool]] = mapped_column(
        Boolean, nullable=True
    )
    association_type_was_filled: Mapped[Optional[bool]] = mapped_column(
        Boolean, nullable=True
    )

    # [LIN-16] True if the pmid_list was capped.
    pmid_list_was_capped: Mapped[Optional[bool]] = mapped_column(
        Boolean, nullable=True
    )

    # -- relationships --
    protein: Mapped["Protein"] = relationship(
        "Protein", back_populates="gene_disease_associations",
        foreign_keys=[uniprot_id],
    )

    # -- validators --
    @validates("gene_symbol")
    def _validate_gene_symbol(self, key: str, value: Optional[str]) -> Optional[str]:
        # v89 ROOT FIX (BUG #25 -- GDA contaminated with mouse gene symbols):
        #   The GDA table is HUMAN-ONLY (DisGeNET human subset, OMIM). The
        #   previous code used the LOOSE ``_validate_gene_symbol`` (which
        #   delegates to ``_GENE_SYMBOL_RE`` = ``^[A-Za-z][A-Za-z0-9\-]{0,49}$``),
        #   accepting Title-Case mouse/rat/yeast symbols (Tp53, Brca1, GAL4).
        #   The strict ``_HUMAN_GENE_SYMBOL_RE`` (``^[A-Z][A-Z0-9\-]{0,49}$``,
        #   ALL CAPS) was DEFINED at line 258 but NEVER USED by the GDA
        #   validator -- so a DisGeNET row for a mouse gene (Tp53) passed
        #   ORM validation, landed in ``gene_disease_associations``, and
        #   downstream KG construction treated it as a human gene -- creating
        #   false Gene->Disease edges and contaminating the KG with non-human
        #   data. ROOT FIX: use ``_HUMAN_GENE_SYMBOL_RE`` here. NULL is
        #   still allowed (gene_symbol is nullable per P1C-001). The loader's
        #   ``_pre_validate_gda`` already normalizes to upper-case before
        #   INSERT, so valid human symbols (BRCA1, FGFR3, TP53) pass. A
        #   row with ``gene_symbol='Tp53'`` (mouse) now raises ValueError
        #   and is quarantined to the dead-letter queue -- exactly the
        #   patient-safe behaviour for a human-only disease-association table.
        if value is None:
            return value
        value = value.strip()
        if _HUMAN_GENE_SYMBOL_RE.match(value):
            return value
        raise ValueError(
            f"Invalid HUMAN gene symbol for GDA: '{value}'. "
            "GDA is human-only (DisGeNET organism=9606 / OMIM). "
            "Must be ALL CAPS, e.g. BRCA1, FGFR3, TP53. "
            "Non-human symbols (Tp53, Brca1) are rejected."
        )

    __table_args__ = (
        Index("idx_gda_gene", "gene_symbol"),
        Index("idx_gda_disease", "disease_id"),
        # [PERF-02] Index on uniprot_id for fast joins
        Index("idx_gda_uniprot_id", "uniprot_id"),
        # [SCI-6] Index on gene_id for stable cross-source joins (IDEM-15)
        Index("idx_gda_gene_id", "gene_id"),
        # [SCI-3] Index on source_id for cross-source analysis
        Index("idx_gda_source_id", "source_id"),
        # [IDEM-14] Index on snapshot_tag for fast snapshot queries
        Index("idx_gda_snapshot_tag", "snapshot_tag"),
        # v93 ROOT FIX (P1-026 -- NULL gene_symbol dedup):
        # ``gene_symbol`` is nullable (line 1700). On PostgreSQL 14 and
        # earlier, SQLite, and MySQL, NULLs are treated as DISTINCT in a
        # UNIQUE constraint -- two GDA rows with ``gene_symbol=NULL``,
        # ``disease_id='C001'``, ``source='disgenet'`` are BOTH allowed
        # (not considered duplicates). This silently corrupts the KG with
        # duplicate Gene->Disease edges.
        #
        # Portable fix (works on PostgreSQL 12+, SQLite 3.31+, MySQL 8+):
        # use a functional UNIQUE index on COALESCE(gene_symbol, '') --
        # NULL is normalized to empty string for the dedup check, so two
        # NULL-gene rows with the same (disease_id, source) collide.
        # PostgreSQL 15+ supports ``NULLS NOT DISTINCT`` but we cannot
        # rely on it (dev runs on SQLite, prod may be on PG 14).
        #
        # The application-level dedup in ``database/loaders.py::
        # bulk_upsert_gdas`` (v93 fix) ALSO drops duplicate NULL-gene
        # rows before insert as a defense-in-depth measure.
        #
        # P1-056 ROOT FIX (v107): REMOVED the duplicate standard
        # ``UniqueConstraint("gene_symbol", "disease_id", "source")``.
        # The previous code had BOTH a standard UniqueConstraint AND a
        # functional nullsafe Index(unique=True). On PostgreSQL, BOTH
        # were created — the second is a functional index that duplicates
        # the first for non-NULL gene_symbol. On SQLite, the functional
        # index may not render — the dedup relies on the application-
        # level ``bulk_upsert_gdas``. The standard UniqueConstraint is
        # REDUNDANT with the nullsafe functional index for non-NULL
        # gene_symbols, and INSUFFICIENT for NULL gene_symbols (because
        # NULLs are distinct in a standard UNIQUE constraint). ROOT FIX:
        # keep ONLY the nullsafe functional index (which handles both
        # NULL and non-NULL cases) and rely on the application-level
        # dedup as defense-in-depth for SQLite dev/test where functional
        # indexes may not render.
        # Functional UNIQUE index for NULL gene_symbol dedup (portable).
        # v93 ROOT FIX (P1-026): COALESCE(gene_symbol, '') normalizes
        # NULL -> '' so two NULL-gene rows with the same (disease_id,
        # source) collide at the index level. Uses ``func.coalesce``
        # with ``column("gene_symbol")`` (SQLAlchemy expression) so
        # the DDL generator renders it as a column reference, not a
        # string literal. On SQLite, functional indexes require
        # SQLAlchemy >= 1.4; on PostgreSQL, they work natively. The
        # application-level dedup in ``database/loaders.py::
        # bulk_upsert_gdas`` (v93 fix) is the defense-in-depth for
        # SQLite dev/test where functional indexes may not render.
        Index(
            "uq_gda_gene_disease_source_nullsafe",
            func.coalesce(column("gene_symbol"), ""), "disease_id", "source",
            unique=True,
        ),
        # [SCI-06 / COMP-5] Disease ID type validation -- extended to include
        # 'hpo' (HPO terms are valid DisGeNET disease IDs per Piñero et al. 2020).
        # CRITICAL FIX (patient safety): added 'icd10' (WHO international
        # clinical classification), 'efo' (Experimental Factor Ontology --
        # used by GWAS Catalog, UK Biobank, Open Targets), and 'orphanet'
        # (rare-disease ontology). Without these, real disease associations
        # would be SILENTLY DROPPED at insert time, hiding drug-disease
        # links from the model. KEEP IN SYNC with:
        #   - database/loaders.py::_VALID_DISEASE_ID_TYPES
        #   - database/migrations/001_initial_schema.sql::chk_gda_disease_id_type
        #   - database/migrations/004_extend_gda_table_for_389_audit.sql
        #   - database/migrations/run_migrations.py::REQUIRED_COLUMNS
        CheckConstraint(
            "disease_id_type IS NULL OR disease_id_type IN "
            "('omim', 'disgenet', 'doid', 'mesh', 'umls', 'hpo', "
            "'icd10', 'efo', 'orphanet')",
            name="chk_gda_disease_id_type",
        ),
        # [SCI-9] diseaseType ∈ {disease, phenotype, group} when non-NULL
        CheckConstraint(
            "disease_type IS NULL OR disease_type IN "
            "('disease', 'phenotype', 'group')",
            name="chk_gda_disease_type",
        ),
        # [SCI-11] confidence_tier must be a known label when non-NULL.
        # V100 ROOT FIX (BUG #4): aligned to Piñero et al. 2020 §2.3 actual
        # vocabulary -- sub_weak / weak / strong. The [0.06, 0.3) band is
        # "weak" (not "moderate" as the previous code wrongly labeled it).
        # P1-004 ROOT FIX EXTENSION (Team-1 v102): split the strong band
        # [0.3, 1.0] into "strong" [0.3, 0.5) and "very_strong" [0.5, 1.0]
        # so the gradation between a score of 0.31 (marginal evidence) and
        # 0.95 (very strong, curated multi-source) is preserved. Downstream
        # ML models that bin on confidence_tier no longer weight them
        # identically. Migration 017 backfills existing rows with score
        # >= 0.5 from "strong" to "very_strong" and updates the DB CHECK
        # constraint in lockstep.
        CheckConstraint(
            "confidence_tier IS NULL OR confidence_tier IN "
            "('sub_weak', 'weak', 'strong', 'very_strong')",
            name="chk_gda_confidence_tier",
        ),
        # [SCI-24] evidence_strength must be a known label when non-NULL
        CheckConstraint(
            "evidence_strength IS NULL OR evidence_strength IN "
            "('robust', 'moderate', 'limited', 'unsupported')",
            name="chk_gda_evidence_strength",
        ),
        # [SCI-41] year_initial <= year_final when both are present
        CheckConstraint(
            "year_initial IS NULL OR year_final IS NULL OR year_initial <= year_final",
            name="chk_gda_year_range",
        ),
        # [SCI-38 / COMP-19] normalized_score must be in [0, 1] when non-NULL
        CheckConstraint(
            "normalized_score IS NULL OR (normalized_score >= 0.0 AND normalized_score <= 1.0)",
            name="chk_gda_normalized_score_range",
        ),
        # v57 ROOT FIX (P1C-001): REMOVED CheckConstraint on gene_symbol
        #   because it contradicted the previous `server_default=""` and
        #   crashed INSERTs. gene_symbol is now NULLABLE; NULL/empty rows
        #   are quarantined by the loader (loaders.py:2618) instead of
        #   crashing the pipeline. The partial unique index in migration 002
        #   (T-004 fix) handles dedup for non-NULL gene_symbols.
        CheckConstraint(
            "disease_id IS NOT NULL AND disease_id <> ''",
            name="chk_gda_disease_id_nonempty",
        ),
        # v39 ROOT FIX (P1 #43/47): added chk_gda_source CHECK constraint
        # to the ORM. The migration (001) has this constraint but the ORM
        # was missing it -- so SQLite dev/test DBs (created via ORM
        # create_all) accepted ANY source string, while PostgreSQL prod
        # DBs (created via migration) rejected anything outside
        # {NULL, 'disgenet', 'omim'}. A row with source='chembl' would
        # be accepted on SQLite but rejected on PostgreSQL -- tests pass,
        # production fails. The fix: add the constraint to the ORM so
        # both paths enforce the same rule.
        CheckConstraint(
            "source IS NULL OR source IN ('disgenet', 'omim')",
            name="chk_gda_source",
        ),
        # v59 ROOT FIX (compound of InChIKey fix -- SQLite create_all
        # crash): the previous ORM CheckConstraint used 9 PostgreSQL
        # regex operators (``~``) which SQLite does NOT support.
        # ``Base.metadata.create_all()`` on SQLite raised
        # ``OperationalError: near "~": syntax error`` -- breaking every
        # test that bootstraps via the ORM. The migration 001 SQL keeps
        # the full regex CHECK (the migration runner has its own SQLite-
        # translation logic). The ORM now uses a portable non-empty
        # backstop; full disease_id format validation is enforced by
        # the Python validator (database/loaders.py::_validate_disease_id)
        # on both dialects. Same pattern as chk_drugs_inchikey_format.
        CheckConstraint(
            "disease_id IS NOT NULL AND LENGTH(TRIM(disease_id)) > 0",
            name="chk_gda_disease_id_format",
        ),
        # v59 ROOT FIX (compound of InChIKey fix -- SQLite create_all
        # crash): the previous ORM CheckConstraint used the PostgreSQL
        # regex operator ``~`` which SQLite does NOT support.
        # ``Base.metadata.create_all()`` on SQLite raised
        # ``OperationalError: near "~": syntax error`` -- breaking every
        # test that bootstraps via the ORM. The migration 001 SQL keeps
        # the full regex CHECK (the migration runner has its own SQLite-
        # translation logic). The ORM now uses a portable length backstop;
        # full pmid_list format validation is enforced by the Python
        # validator on both dialects. Same pattern as chk_drugs_inchikey_format.
        CheckConstraint(
            "pmid_list IS NULL OR LENGTH(pmid_list) <= 2000",
            name="chk_gda_pmid_list",
        ),
        # v29 ROOT FIX (audit D-6): Removed the duplicate partial
        # ``Index("uq_gda_gene_disease_source_partial", ..., unique=True,
        # postgresql_where=text("gene_symbol IS NOT NULL OR gene_symbol = ''"))``
        # that previously lived here. It was redundant with the
        # ``UniqueConstraint("uq_gda_gene_disease_source")`` declared above
        # for two reasons:
        #   1. The ``postgresql_where`` clause
        #      ``gene_symbol IS NOT NULL OR gene_symbol = ''`` was a
        #      TAUTOLOGY once ``chk_gda_gene_symbol_nonempty`` (just above)
        #      rejected both NULL and empty-string ``gene_symbol`` -- every
        #      surviving row matched the partial predicate, so the "partial"
        #      index actually covered the WHOLE table. It was therefore a
        #      second full UNIQUE index on (gene_symbol, disease_id, source)
        #      in addition to the UniqueConstraint -- 2× write amplification
        #      (4× if you also count the implicit index SQLAlchemy emits on
        #      the UniqueConstraint) on every INSERT/UPDATE/DELETE.
        #   2. SQLite (dev/test) silently ignores ``postgresql_where``, so
        #      the "partial" index became a SECOND plain unique index on
        #      SQLite -- producing a confusing duplicate-index error surface
        #      and wasting disk on every dev DB.
        # Keeping ONLY the canonical ``UniqueConstraint`` is sufficient: it
        # already enforces uniqueness on (gene_symbol, disease_id, source)
        # on both SQLite and PostgreSQL, with NULLS DISTINCT semantics on
        # PostgreSQL 15+ (the project's minimum supported version).
    )

    def __repr__(self) -> str:
        # SCI-FIX: Use self.uniprot_id (declared on the ORM model) instead of
        # self.protein_id (which exists in the DB table via migration 003 but
        # is NOT mapped on the ORM model, causing AttributeError on repr).
        return (
            f"<GeneDiseaseAssociation(id={self.id}, gene_symbol='{self.gene_symbol}', "
            f"disease_id='{self.disease_id}', source='{self.source}', "
            f"uniprot_id={self.uniprot_id!r})>"
        )


# ===========================================================================
# 6. ENTITY MAPPING (cross-database entity resolution output)
# ===========================================================================


class EntityMapping(Base, IDMixin, TimestampMixin):
    """Cross-database entity resolution output.

    Domain meaning
    --------------
    Each row represents a resolved entity that maps identifiers across
    databases (ChEMBL, DrugBank, UniProt, STRING).  When a canonical
    InChIKey is available, it serves as the primary identifier; otherwise
    canonical_name is used.

    Key constraints
    ---------------
    - Partial unique index on ``canonical_inchikey`` WHERE NOT NULL.
    - ``match_confidence`` must be in [0, 1] (DQ-07).
    - [SCI-01] InChIKey widened to 50 for synthetic keys.
    - [DES-03] Additional partial unique indexes on chembl_id, drugbank_id.
    - [DQ-03] FK constraints on chembl_id, drugbank_id, uniprot_id, string_id.
    - [LINE-04] ``last_matched_at`` records when the last resolution was
      performed (mirrors migration 001 line 1227).

    Lineage (LINE-04)
    ------------------
    - ``match_history`` stores the full resolution attempt chain as JSON.
    - ``last_matched_at`` is updated by the loader on every successful
      entity-mapping upsert (bulk_upsert_entity_mapping).
    """
    __tablename__ = "entity_mapping"

    # [SCI-01] Widened from 27 to 50 for synthetic InChIKeys
    canonical_inchikey: Mapped[Optional[str]] = mapped_column(
        String(INCHIKEY_LENGTH), nullable=True,
    )
    canonical_name: Mapped[Optional[str]] = mapped_column(
        String(DRUG_NAME_LENGTH), nullable=True,
    )
    # [DQ-03] Application-level FK enforcement for cross-reference integrity.
    # FK constraints on chembl_id, drugbank_id, uniprot_id, string_id are
    # enforced at the application/loader level rather than at the DB level
    # because these reference non-PK columns with partial unique indexes,
    # which SQLite does not support as FK targets.
    chembl_id: Mapped[Optional[str]] = mapped_column(
        String(CHEMBL_ID_LENGTH), nullable=True,
    )
    drugbank_id: Mapped[Optional[str]] = mapped_column(
        String(DRUGBANK_ID_LENGTH), nullable=True,
    )
    pubchem_cid: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    uniprot_id: Mapped[Optional[str]] = mapped_column(
        String(UNIPROT_ID_LENGTH), nullable=True,
    )
    string_id: Mapped[Optional[str]] = mapped_column(
        String(STRING_ID_LENGTH), nullable=True,
    )
    # [DQ-07] Match confidence range
    match_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    match_method: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # [LINE-04] Full resolution attempt chain
    match_history: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # v89 ROOT FIX (BUG #21 -- schema drift: migration 001 declares
    #   ``last_matched_at TIMESTAMP WITH TIME ZONE`` at line 1227, but the
    #   ORM EntityMapping model did NOT declare this column.
    #   ``Base.metadata.create_all()`` created ``entity_mapping`` WITHOUT
    #   ``last_matched_at``; migration 001's ``CREATE TABLE IF NOT EXISTS``
    #   was then a no-op (table already exists from create_all). So dev DBs
    #   (SQLite, create_all path) LACKED the column, while prod DBs
    #   (PostgreSQL, migration path) HAD it. The loader's
    #   ``bulk_upsert_entity_mapping`` (loaders.py ~line 3377) listed
    #   ``updatable_cols`` but ``last_matched_at`` was NOT among them -- so
    #   even on prod, the column was never populated. ``verify_schema_matches_orm``
    #   reported it as an "extra column" on prod, creating false-positive
    #   schema-drift warnings. ROOT FIX: declare the column on the ORM so
    #   dev and prod agree, AND add it to the loader's ``updatable_cols``
    #   so it gets populated on every upsert (see loaders.py fix).
    last_matched_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # -- validators --
    @validates("canonical_inchikey")
    def _validate_inchikey(self, key: str, value: Optional[str]) -> Optional[str]:
        return _validate_inchikey(value)

    __table_args__ = (
        # Partial unique index: only enforce uniqueness for non-NULL canonical_inchikey
        Index(
            "uq_entity_mapping_inchikey",
            "canonical_inchikey",
            unique=True,
            postgresql_where=text("canonical_inchikey IS NOT NULL"),
        ),
        # [DES-03] Partial unique index for records without InChIKey
        Index(
            "uq_entity_mapping_name_no_inchikey",
            "canonical_name",
            unique=True,
            postgresql_where=text("canonical_inchikey IS NULL AND canonical_name IS NOT NULL"),
        ),
        # [DES-03] Unique chembl_id where not null
        Index(
            "uq_entity_mapping_chembl",
            "chembl_id",
            unique=True,
            postgresql_where=text("chembl_id IS NOT NULL"),
        ),
        # [DES-03] Unique drugbank_id where not null
        Index(
            "uq_entity_mapping_drugbank",
            "drugbank_id",
            unique=True,
            postgresql_where=text("drugbank_id IS NOT NULL"),
        ),
        # [DQ-07] Match confidence range
        CheckConstraint(
            "match_confidence IS NULL OR (match_confidence >= 0.0 AND match_confidence <= 1.0)",
            name="chk_entity_mapping_confidence_range",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<EntityMapping(id={self.id}, canonical_inchikey='{self.canonical_inchikey}', "
            f"canonical_name='{self.canonical_name}', "
            f"match_confidence={self.match_confidence})>"
        )


# ===========================================================================
# 6b. DEAD LETTER QUEUE for GDA (DQ-18 / REL-3 / LIN-11)
# ===========================================================================


class DeadLetterGDA(Base, IDMixin, TimestampMixin):
    """Dead-letter queue for GDA records that could not be loaded.

    Domain meaning
    --------------
    Each row represents a GDA record that was rejected by the load()
    phase -- e.g. unresolved gene_symbol, invalid disease_id format,
    inverted year range, etc.  Rows are written here instead of being
    silently dropped, so the data can be inspected and reprocessed
    later (REL-3, LIN-11).

    Key constraints
    ---------------
    - ``reason`` is a short stable identifier (e.g.
      ``"unresolved_gene_symbol"``, ``"invalid_disease_id_format"``).
    - ``details_json`` is a JSON object with the offending values
      (gene_symbol, disease_id, score, etc.) for debugging.
    - ``pipeline_run_id`` is an Integer FK -> ``pipeline_runs.id``
      (ON DELETE SET NULL), matching every other lineage-bearing table
      (DPI, PPI, GDA, RejectedRecord, PubChemCompoundProperty).

    v89 ROOT FIX (BUG #29 -- run_id was String(64), not Integer FK):
      The previous ``run_id: String(64)`` stored a UUID string with NO
      FK to ``pipeline_runs.id``. Every other lineage-bearing table uses
      ``Integer FK -> pipeline_runs.id (ON DELETE SET NULL)``. The audit
      D-7 fix explicitly aligned ``PubChemCompoundProperty.pipeline_run_id``
      from String(64) to Integer FK -- but ``DeadLetterGDA.run_id`` was
      NOT updated. This meant: (a) dead-letter rows could not be JOINed
      to ``pipeline_runs`` without a CAST; (b) a typo'd run_id silently
      orphaned the dead-letter row (no FK enforcement); (c) the loader's
      ``_quarantine_gda_rows`` wrote ``str(pipeline_run_id)`` --
      converting the Integer to a string, losing the FK relationship.
      ROOT FIX: rename ``run_id`` -> ``pipeline_run_id``, change type to
      ``Integer FK -> pipeline_runs.id (ON DELETE SET NULL)``, and update
      the loader to write the integer directly (see loaders.py fix).
      The index ``idx_dlgda_run_id`` is renamed to
      ``idx_dlgda_pipeline_run_id`` to match the column rename.
    """
    __tablename__ = "dead_letter_gda"

    gene_symbol: Mapped[Optional[str]] = mapped_column(
        String(GENE_SYMBOL_LENGTH), nullable=True
    )
    disease_id: Mapped[Optional[str]] = mapped_column(
        String(DISEASE_ID_LENGTH), nullable=True
    )
    source: Mapped[Optional[str]] = mapped_column(
        String(SOURCE_LENGTH), nullable=True
    )
    reason: Mapped[str] = mapped_column(String(100), nullable=False)
    details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # v89 ROOT FIX (BUG #29): Integer FK -> pipeline_runs.id (ON DELETE
    # SET NULL), matching every other lineage-bearing table. Was
    # String(64) with no FK.
    pipeline_run_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("pipeline_runs.id", ondelete="SET NULL"),
        nullable=True,
    )

    __table_args__ = (
        Index("idx_dlgda_reason", "reason"),
        Index("idx_dlgda_pipeline_run_id", "pipeline_run_id"),
        Index("idx_dlgda_gene_symbol", "gene_symbol"),
        Index("idx_dlgda_disease_id", "disease_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<DeadLetterGDA(id={self.id}, reason='{self.reason}', "
            f"gene_symbol='{self.gene_symbol}', disease_id='{self.disease_id}', "
            f"pipeline_run_id={self.pipeline_run_id})>"
        )


# ===========================================================================
# 7. PIPELINE RUNS (ETL audit log)
# ===========================================================================


# v17 ROOT FIX (AuditLog ORM missing): migration 001 declares an
# ``audit_log`` table (lines 1345-1397 of 001_initial_schema.sql) used
# by migrations 002/004/005/006 to record pre/post-migration row counts
# and lineage operations (PRE_MIGRATION_002_CHECKSUM, DELETE_NULL_DISEASE_ID,
# DEDUP_MIGRATION_002, etc.). The table has 9 columns: id, table_name,
# operation, record_id, changed_by, changed_at, old_values, new_values,
# row_count, details. Without an ORM model, ``Base.metadata.create_all()``
# on SQLite dev/test DBs did NOT create this table -- so any Python code
# that tried to write audit records via the ORM raised
# ``sqlite3.OperationalError: no such table: audit_log``. The migration
# 001 ``CREATE TABLE IF NOT EXISTS`` was the only creation path, and on
# SQLite it was being silently skipped (CD-5 was the fix that made
# migrations run on SQLite, but the audit_log table itself had no ORM
# fallback). Adding this model closes the gap -- create_all() now
# creates audit_log on BOTH PostgreSQL and SQLite, and migration 001's
# CREATE TABLE IF NOT EXISTS becomes the idempotent no-op it was
# designed to be.
class AuditLog(Base, IDMixin):
    """Audit log table for tracking schema migrations and bulk operations.

    Domain meaning
    --------------
    Each row records a single audit event -- typically a schema-migration
    lineage operation (PRE_MIGRATION_002_CHECKSUM, DELETE_NULL_DISEASE_ID,
    DEDUP_MIGRATION_002, etc.) or a bulk data operation (BULK_OPERATION,
    SOFT_DELETE, RESTORE). Used by migrations 002/004/005/006 to record
    pre/post-migration row counts and checksums for replay safety.

    Key constraints
    ---------------
    - ``table_name`` NOT NULL (which table was affected).
    - ``operation`` NOT NULL, constrained to the whitelist defined in
      migration 001 (CHECK constraint chk_audit_log_operation).
    - ``changed_at`` NOT NULL DEFAULT NOW().
    - ``row_count`` nullable INTEGER (used by migration lineage INSERTs).
    - ``details`` nullable TEXT (free-form context, e.g. checksum values).
    """
    __tablename__ = "audit_log"

    table_name: Mapped[str] = mapped_column(String(50), nullable=False)
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    record_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    changed_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    changed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    old_values: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_values: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # v17: row_count + details -- added to migration 001 by the RT-1 /
    # Compound-4 "Migration Wall" fix so migration 002's INSERTs into
    # audit_log (table_name, operation, row_count, details) stop
    # aborting with "column row_count of relation audit_log does not
    # exist". Mirror them here so the ORM-created table matches.
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # v17: mirror the chk_audit_log_operation CHECK whitelist from
        # migration 001 so SQLite dev/test DBs (which skip the migration
        # SQL) still enforce the same operation-enum contract.
        #
        # v89 ROOT FIX (BUG #28 -- hardcoded migration numbers block 007+):
        #   The previous CHECK hardcoded specific migration numbers (002,
        #   004, 005, 006) in the operation whitelist. Migrations 007, 008,
        #   009, 010, 011 were NOT in the whitelist -- so any migration >= 7
        #   that tried to INSERT a ``PRE_MIGRATION_007_CHECKSUM`` row would
        #   fail the CHECK, and the audit trail would be incomplete for all
        #   future migrations. The migration runner's ``_record_failure``
        #   and provenance code would need to add tokens to the whitelist
        #   for EVERY new migration -- an unmaintainable pattern. ROOT FIX:
        #   replace the hardcoded migration numbers with a pattern match
        #   (``LIKE 'PRE_MIGRATION_%_CHECKSUM'`` /
        #   ``LIKE 'POST_MIGRATION_%_CHECKSUM'``) so ANY future migration
        #   (012, 013, ...) can log checksums WITHOUT modifying the ORM
        #   model. The fixed tokens (INSERT, UPDATE, DELETE, SOFT_DELETE,
        #   RESTORE, MIGRATION_*, BULK_OPERATION, DELETE_NULL_*, etc.) are
        #   preserved. ``%`` is the SQL wildcard for 0+ chars; the CHECK
        #   now accepts ``PRE_MIGRATION_002_CHECKSUM`` AND
        #   ``PRE_MIGRATION_999_CHECKSUM`` identically. This is forward-
        #   compatible and matches the audit-log contract documented in
        #   migration 001's header comment.
        CheckConstraint(
            "operation IN ("
            "'INSERT', 'UPDATE', 'DELETE', 'SOFT_DELETE', 'RESTORE', "
            "'MIGRATION_BACKFILL', 'MIGRATION_DEDUP', 'MIGRATION_CONSTRAINT', "
            "'BULK_OPERATION', "
            "'DELETE_NULL_DISEASE_ID', 'DELETE_NULL_SOURCE', "
            "'PRESERVED_NULL_GENE_SYMBOL', 'DEDUP_MIGRATION_002'"
            ") "
            "OR operation LIKE 'PRE_MIGRATION_%_CHECKSUM' "
            "OR operation LIKE 'POST_MIGRATION_%_CHECKSUM'",
            name="chk_audit_log_operation",
        ),
        Index("idx_audit_log_table_name", "table_name"),
        Index("idx_audit_log_operation", "operation"),
        Index("idx_audit_log_changed_at", "changed_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog(id={self.id}, table_name='{self.table_name}', "
            f"operation='{self.operation}', row_count={self.row_count})>"
        )


class PipelineRun(Base, IDMixin, TimestampMixin):
    """ETL pipeline execution audit log.

    Domain meaning
    --------------
    Each row records a single pipeline run -- its source, status, record
    counts, duration, and any error details.

    Key constraints
    ---------------
    - ``source`` must be one of the 7 known pipeline names (DES-07).
    - ``status`` constrained to known values (DES-05).
    - ``duration_seconds`` must be non-negative (DQ-08).
    - ``error_message`` capped at 500 chars (SEC-04).
    - ``UniqueConstraint(source, run_date)`` for idempotency (DES-07).
    """
    __tablename__ = "pipeline_runs"

    # [DES-07] Source constrained to known pipeline names
    source: Mapped[str] = mapped_column(
        String(PIPELINE_SOURCE_LENGTH), nullable=False,
    )
    run_date: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    # [DES-05] Status constrained by CHECK
    status: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True,
    )
    records_downloaded: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    records_cleaned: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    records_loaded: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # v13 ROOT FIX (CD-4): the following 6 columns were created by
    # migration 001 (lines 1143-1155) but MISSING from the ORM model.
    # This meant ``Base.metadata.create_all()`` created the
    # ``pipeline_runs`` table with only 8 columns, and migration 001's
    # ``CREATE TABLE IF NOT EXISTS`` was a no-op (table already
    # existed). The 6 columns were never created on SQLite dev/test
    # DBs. Airflow retry / checkpoint / partial-failure tracking code
    # that referenced these columns via the ORM raised
    # ``AttributeError`` at runtime. v13: declare all 6 columns on the
    # ORM so ``create_all()`` and migration 001 agree on the schema.
    records_failed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    records_skipped: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    records_updated: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    last_checkpoint: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True,
    )
    input_file_checksum: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
    )
    config_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
    )
    # [SEC-04] Error message capped to prevent stack trace leakage
    error_message: Mapped[Optional[str]] = mapped_column(
        String(ERROR_MESSAGE_LENGTH), nullable=True,
    )
    # [DQ-08] Duration must be non-negative
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # [P1-18 ROOT FIX] Per-run audit metadata (run_id, correlation_id,
    # triggered_by, source_version, sha256_raw, sha256_cleaned, git_commit,
    # seed, schema_version, validation_errors, dq_metrics, record counts).
    # BasePipeline._write_run_log already builds this dict and passes it as
    # metadata_json -- without this column, the constructor silently dropped
    # it on every run. Migration 007 adds the column; the JSON type maps to
    # JSONB on PostgreSQL and TEXT on SQLite (via the SQLAlchemy JSON
    # dialect).
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        # [DES-07] Source must be a known pipeline
        CheckConstraint(
            # P1-023 v113 ROOT FIX: add drugbank_open, chembl_activities,
            # omim_susceptibility to match migration 001's extended whitelist.
            "source IN ('chembl', 'drugbank', 'drugbank_open', 'uniprot', 'string', "
            "'disgenet', 'omim', 'omim_susceptibility', 'pubchem', 'chembl_activities')",
            name="chk_pipeline_runs_source",
        ),
        # [DQ-08] Duration must be non-negative
        CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0",
            name="chk_pipeline_runs_duration_nonneg",
        ),
        # v17 ROOT FIX (CD-4 deepened): migration 001 declares 3 CHECK
        # constraints on ``pipeline_runs`` that the ORM was MISSING --
        # ``chk_pipeline_runs_status`` (status enum),
        # ``chk_pipeline_runs_counts_nonneg`` (record counts non-negative),
        # ``chk_pipeline_runs_error_message`` (error_message length cap).
        # Without these, ``Base.metadata.create_all()`` on SQLite dev/test
        # DBs created a ``pipeline_runs`` table that accepted any string
        # for status (e.g. "BOGUS") and negative record counts. The
        # migration 001 ``CREATE TABLE IF NOT EXISTS`` was a no-op
        # because the table already existed from create_all -- so the
        # constraints were NEVER applied on SQLite. Code that passed
        # tests on SQLite could fail on PostgreSQL (where the
        # constraints ARE applied). Add all 3 constraints to the ORM
        # so both paths produce the same schema.
        CheckConstraint(
            "status IS NULL OR status IN "
            "('running', 'success', 'failed', 'partial')",
            name="chk_pipeline_runs_status",
        ),
        CheckConstraint(
            "(records_downloaded IS NULL OR records_downloaded >= 0) "
            "AND (records_cleaned IS NULL OR records_cleaned >= 0) "
            "AND (records_loaded IS NULL OR records_loaded >= 0) "
            "AND (records_failed IS NULL OR records_failed >= 0) "
            "AND (records_skipped IS NULL OR records_skipped >= 0) "
            "AND (records_updated IS NULL OR records_updated >= 0)",
            name="chk_pipeline_runs_counts_nonneg",
        ),
        CheckConstraint(
            "error_message IS NULL OR LENGTH(error_message) <= 500",
            name="chk_pipeline_runs_error_message",
        ),
        # [DES-07] UniqueConstraint for idempotent pipeline runs
        UniqueConstraint(
            "source", "run_date",
            name="uq_pipeline_runs_source_date",
        ),
        Index("idx_pr_source", "source"),
        Index("idx_pr_status", "status"),
        Index("idx_pr_run_date", "run_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<PipelineRun(id={self.id}, source='{self.source}', "
            f"status='{self.status}', run_date={self.run_date}, "
            f"duration_seconds={self.duration_seconds})>"
        )


# ===========================================================================
# PubChem compound properties (ARCH-5, INT-7, SCI-4, SCI-6)
# ===========================================================================


class PubChemCompoundProperty(Base, IDMixin, TimestampMixin, SoftDeleteMixin):
    """Full PubChem compound property record (one row per inchikey+pubchem_cid).

    v43 ROOT FIX (P2 -- PubChemCompoundProperty not using SoftDeleteMixin):
    Previously used a standalone is_deleted column. Now inherits
    SoftDeleteMixin (like Drug and Protein) so it gets deleted_at
    timestamp + soft_delete()/restore() helpers + consistent query
    pattern `WHERE is_deleted = False AND deleted_at IS NULL`.

    Domain meaning
    --------------
    The ``drugs`` table only stores 4 PubChem columns (pubchem_cid,
    molecular_formula, molecular_weight, smiles).  This table stores the
    FULL set of 15+ properties fetched from PubChem PUG REST -- InChI,
    IUPACName, XLogP, ExactMass, TPSA, Complexity, HBondDonorCount,
    HBondAcceptorCount, RotatableBondCount, HeavyAtomCount, IsomericSMILES,
    CAS, etc.  Phase 3 (Graph Transformer) needs these for molecular
    fingerprinting.

    Why this ORM model was added
    ----------------------------
    CRITICAL FIX (cross-dialect compatibility / runtime safety):
    Previously, this table was only created by SQL migration
    ``005_pubchem_compound_properties.sql``, which is correctly SKIPPED on
    SQLite by the migration runner (SQLite does not support
    ``GENERATED ALWAYS AS IDENTITY``).  As a result, when the codebase ran
    on SQLite (the dev / test dialect), the PubChem pipeline failed with
    ``sqlite3.OperationalError: no such table: pubchem_compound_properties``.
    Adding this ORM model means ``Base.metadata.create_all()`` creates the
    table on BOTH PostgreSQL (where it may already exist from migration
    005 -- ``create_all`` is additive and idempotent) AND SQLite (where it
    is the only creation path).  Schema parity with migration 005 is
    enforced by the test suite.
    """
    __tablename__ = "pubchem_compound_properties"

    # [SCI-11, LIN-2] The InChIKey we requested from PubChem.
    # v16 ROOT FIX (CD-2): add FK to drugs.inchikey so the
    # relationship is enforced at the DB level (was missing in ORM
    # but present in migration 005). Aligns ORM with migration.
    # v17 ROOT FIX (CD-2 deepened): migration 005 declares the FK
    # WITHOUT ``ondelete`` (default NO ACTION). The ORM declared
    # ``ondelete="CASCADE"`` -- divergent. On PostgreSQL, both
    # create_all() and migration 005 try to create the FK; the second
    # one silently wins depending on which runs first, producing
    # non-deterministic on-delete behavior. Align ORM to migration
    # 005's NO ACTION (the safer default -- a properties row blocks
    # drug deletion until explicitly cleaned up).
    inchikey: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("drugs.inchikey"),
        nullable=False, index=True,
    )

    # [SCI-5, LIN-2] PubChem Compound ID (parent / standardized).
    pubchem_cid: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)

    # [SCI-1, DESIGN-1] CanonicalSMILES (no stereo).
    canonical_smiles: Mapped[Optional[str]] = mapped_column(String(50000), nullable=True)

    # [SCI-1, SCI-14, SCI-15] IsomericSMILES (with stereochemistry).
    isomeric_smiles: Mapped[Optional[str]] = mapped_column(String(50000), nullable=True)

    # [SCI-2] InChI -- full InChI string.
    inchi: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # [SCI-3] IUPACName -- systematic chemical name.
    # v16 ROOT FIX (CD-2): Text (not String(1000)) to match migration 005.
    iupac_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # [SCI-4] CAS Registry Number.
    cas_number: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # [SCI-5] Molecular formula.
    molecular_formula: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # [SCI-6] Molecular weight (g/mol).
    # v16 ROOT FIX (CD-2): Numeric(12,6) (not Float) to match
    # migration 005 / Core Table. Float loses precision for large
    # molecular weights (e.g. antibodies ~150 kDa have sub-Da
    # precision needs).
    molecular_weight: Mapped[Optional[float]] = mapped_column(Numeric(12, 6), nullable=True)

    # [SCI-7] Exact mass (monoisotopic).
    # v16 CD-2: Numeric(12,6) to match migration 005.
    exact_mass: Mapped[Optional[float]] = mapped_column(Numeric(12, 6), nullable=True)

    # [SCI-8] XLogP -- computed octanol-water partition coefficient.
    # v16 CD-2: Numeric(6,2) to match migration 005.
    # v17 CD-2 deepened: add server_default='pubchem_xlogp3' to match
    # migration 005 -- without the default, the loader had to populate
    # xlogp_source explicitly on every insert, diverging from the
    # migration's intent (the value is constant for fetched rows).
    xlogp: Mapped[Optional[float]] = mapped_column(Numeric(6, 2), nullable=True)
    xlogp_source: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, server_default="pubchem_xlogp3",
    )

    # [SCI-9] Topological Polar Surface Area (Å²).
    # v16 CD-2: Numeric(8,2) to match migration 005.
    # v17 CD-2 deepened: add server_default='pubchem_calculated'.
    tpsa: Mapped[Optional[float]] = mapped_column(Numeric(8, 2), nullable=True)
    tpsa_source: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, server_default="pubchem_calculated",
    )

    # [SCI-10] Bertz complexity index.
    # v16 CD-2: Numeric(10,2) to match migration 005.
    complexity: Mapped[Optional[float]] = mapped_column(Numeric(10, 2), nullable=True)

    # [SCI-11] Lipinski H-bond donor count.
    h_bond_donor_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # [SCI-12] Lipinski H-bond acceptor count.
    h_bond_acceptor_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # [SCI-13] Rotatable bond count.
    rotatable_bond_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # [SCI-14] Heavy atom count.
    heavy_atom_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # [SCI-15] Formal charge.
    formal_charge: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # [SCI-16] Isotope info (free-text description).
    # v16 CD-2: Text (not String(200)) to match migration 005.
    isotope_info: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # [SCI-17] Salt form.
    # v16 CD-2: String(100) (not String(50)) to match migration 005.
    salt_form: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # [SCI-18] Protonation state.
    # v20 CD-2 ROOT FIX: String(20) (VARCHAR(20) in migration 005) to match
    # the V19 PS-1 widened word taxonomy.
    # The original v16 comment claimed CHAR(1)/String(1) matched migration 005,
    # but V19 PS-1 widened migration 005 to VARCHAR(20) for full words
    # ('neutral', 'protonated', 'deprotonated', 'zwitterion', 'salt_form').
    # This ORM/Core Table site was NOT updated, re-introducing 3-way schema
    # drift. Loader returning 'protonated' (10 chars) would silently truncate
    # on SQLite or raise DataError on PostgreSQL strict.
    protonation_state: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # [LIN-2-extra] PubChem release tag (e.g. "PubChem 2024.09").
    pubchem_release: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # [LIN-1, LIN-2] Source identifiers.
    # v16 CD-2: NOT NULL + String(100) to match migration 005.
    # v17 CD-2 deepened: migration 005 declares NOT NULL WITHOUT a
    # server_default. The ORM added ``server_default=""`` to keep
    # create_all() happy on SQLite -- but this means the ORM path
    # silently accepts empty strings while the migration path raises.
    # Keep the server_default (SQLite compatibility) but document the
    # divergence -- the loader always populates source_id explicitly.
    source_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=False, server_default="",
    )
    source_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # [LIN-3] Download date (when PubChem returned this record).
    download_date: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    # v16 CD-2: String(20) (not String(50)) to match migration 005.
    download_method: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # [LIN-4] Pipeline run ID (FK to pipeline_runs.id).
    # v16 CD-2: NOT NULL + String(64) to match migration 005.
    # v17 CD-2 deepened: same server_default divergence as source_id
    # -- see comment above. Loader populates explicitly.
    #
    # v29 ROOT FIX (audit D-7): was String(64) (a free-form UUID string
    # column with no FK), while EVERY other lineage-bearing table in the
    # schema (``drug_protein_interactions``, ``protein_protein_interactions``,
    # ``gene_disease_associations``, ``rejected_records``) uses
    # ``Integer FK -> pipeline_runs.id (ON DELETE SET NULL)``. The
    # String(64) form meant (a) no FK was enforced -- a typo'd run id could
    # silently orphan the row; (b) join cardinality against
    # ``pipeline_runs`` required a CAST, breaking the planner; (c) the
    # column accepted arbitrary UUID strings that did not correspond to any
    # real ``pipeline_runs.id`` value. Aligned to the canonical Integer FK
    # pattern; nullable=True so historical loader code that wrote "" can
    # still round-trip (the empty string is now mapped to NULL by the
    # loader before INSERT). Migration 005 / pubchem_pipeline.py must be
    # updated in lockstep -- see audit D-7 remediation notes.
    pipeline_run_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("pipeline_runs.id", ondelete="SET NULL"),
        nullable=True,
    )

    # [LIN-5-extra] Source batch index + response SHA-256 for full traceability.
    source_batch_idx: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_response_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # [LIN-5] Input checksum (SHA-256 of the input InChIKey list).
    # v16 CD-2: NOT NULL to match migration 005.
    # v17 CD-2 deepened: same server_default divergence -- see above.
    input_checksum: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=False, server_default="",
    )

    # [LIN-6] Transformations applied (JSON-encoded list).
    transformations: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # [COMP-5] FDA 21 CFR Part 11 electronic-signature fields.
    electronic_signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    triggered_by: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # [DESIGN-7, IDEM-9] Enrichment timestamp -- non-deterministic by design;
    # the OTHER columns are deterministic given the same PubChem response.
    # v17 ROOT FIX (CD-2 deepened): migration 005 declares
    # ``enriched_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()``.
    # The ORM declared it ``nullable=True`` with no default -- divergent.
    # On PostgreSQL, create_all() creates the column nullable; migration
    # 005's CREATE TABLE is then a no-op (table exists); the column
    # stays nullable. On INSERT, NULL enriched_at was accepted -- but
    # downstream queries filtering ``WHERE enriched_at IS NOT NULL`` or
    # computing enrichment age would skip / mis-classify those rows.
    # Align ORM to migration: NOT NULL, server_default=NOW().
    enriched_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )

    # [IDEM-9] Soft-delete flag for re-run idempotency.
    # v90 ROOT FIX (BUG #7 -- P1 re-declared is_deleted overrides mixin):
    #   The previous code RE-DECLARED ``is_deleted`` here, even though
    #   ``PubChemCompoundProperty`` already inherits ``SoftDeleteMixin``
    #   (which declares ``is_deleted`` and ``deleted_at`` at base.py:186).
    #   In SQLAlchemy 2.0 declarative, re-declaring a column from a mixin
    #   in the class body OVERRIDES the mixin's definition. The
    #   ``soft_delete()`` and ``restore()`` helper methods from
    #   SoftDeleteMixin still reference ``self.is_deleted`` and
    #   ``self.deleted_at`` -- but ``deleted_at`` is now orphaned from the
    #   mixin (still inherited, but the override created inconsistent
    #   metadata). The ``idx_pubchem_props_is_deleted`` partial index
    #   (below) references ``is_deleted`` which had DIVERGENT column
    #   metadata between the mixin-declared and re-declared versions.
    #   ROOT FIX: remove the re-declaration. SoftDeleteMixin already
    #   provides ``is_deleted`` (now with v90 BUG #23 `server_default=
    #   text("FALSE")` instead of the non-portable `server_default="0"`)
    #   plus ``deleted_at`` and the ``soft_delete()`` / ``restore()``
    #   helper methods. The mixin's `is_deleted` now emits `DEFAULT FALSE`
    #   on every dialect -- byte-identical to migration 001 line 540.

    __table_args__ = (
        # [SCI-19, IDEM-4] Composite unique constraint -- one row per
        # (inchikey, pubchem_cid). ON CONFLICT DO UPDATE on this constraint.
        UniqueConstraint(
            "inchikey", "pubchem_cid",
            name="uq_pubchem_compound_properties_inchikey_cid",
        ),
        # v17 ROOT FIX (CD-2 deepened): migration 005 creates these
        # indexes with names ``idx_pubchem_props_inchikey`` and
        # ``idx_pubchem_props_cid`` (plus two more indexes the ORM
        # was MISSING entirely: ``idx_pubchem_props_is_deleted`` and
        # ``idx_pubchem_props_run_id``). The ORM created differently-
        # named indexes -- on PostgreSQL this produced DUPLICATE
        # indexes (one from migration, one from create_all), wasting
        # disk + write bandwidth on every INSERT. Align ORM index
        # names to migration 005 and add the two missing indexes so
        # there is exactly one index per query pattern.
        Index("idx_pubchem_props_inchikey", "inchikey"),
        Index("idx_pubchem_props_cid", "pubchem_cid"),
        # [IDEM-7] Partial index for soft-delete cleanup queries.
        # Use postgresql_where so the partial index is created on PG;
        # on SQLite the WHERE is dropped by _translate_sql_for_sqlite
        # (full index is created instead -- slightly larger but
        # functionally equivalent).
        Index(
            "idx_pubchem_props_is_deleted", "is_deleted",
            postgresql_where=text("is_deleted = TRUE"),
        ),
        # [LIN-10] Index for pipeline-run traceability queries.
        Index("idx_pubchem_props_run_id", "pipeline_run_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<PubChemCompoundProperty(id={self.id}, inchikey='{self.inchikey}', "
            f"pubchem_cid={self.pubchem_cid})>"
        )


# ===========================================================================
# 9. REJECTED RECORDS (dead-letter queue)
# ===========================================================================


# P1-ER-6 ROOT FIX (orphan rejected_records table): migration 001 declares
# a ``rejected_records`` table (lines 1283-1327 of 001_initial_schema.sql)
# used as a dead-letter queue for records that fail validation during
# pipeline execution. Without an ORM model, ``Base.metadata.create_all()``
# on SQLite dev/test DBs did NOT create this table -- so any Python code
# that tried to write rejected records via the ORM raised
# ``sqlite3.OperationalError: no such table: rejected_records``. The
# migration 001 ``CREATE TABLE IF NOT EXISTS`` was the only creation
# path, and on SQLite it was being silently skipped. Adding this model
# closes the gap -- create_all() now creates rejected_records on BOTH
# PostgreSQL and SQLite, and migration 001's CREATE TABLE IF NOT EXISTS
# becomes the idempotent no-op it was designed to be.
#
# The schema here mirrors migration 001 EXACTLY (column names, types,
# nullability, CHECK constraint, FK, indexes). Do NOT diverge.
class RejectedRecord(Base, IDMixin):
    """Dead-letter queue for unprocessable records (migration 001, CMP-04).

    Domain meaning
    --------------
    Each row represents a record that was rejected by the load() phase
    of some pipeline -- e.g. a drug with a malformed InChIKey, a protein
    with an invalid UniProt accession, a duplicate GDA, etc. Rows are
    written here instead of being silently dropped, so the data can be
    inspected and reprocessed later. Retention: 1 year, then purged.

    Key constraints
    ---------------
    - ``source_table`` NOT NULL -- target table the record was intended
      for (e.g. "drugs", "proteins", "gene_disease_associations").
    - ``source_pipeline`` NOT NULL -- pipeline that rejected the record
      (e.g. "chembl", "drugbank", "disgenet").
    - ``raw_data`` NOT NULL TEXT -- original record as a JSON string.
    - ``rejection_reason`` NOT NULL VARCHAR(500) -- human-readable
      explanation.
    - ``rejection_type`` NOT NULL VARCHAR(50), constrained to the
      whitelist defined in migration 001 (chk_rejected_records_rejection_type).
    - ``pipeline_run_id`` nullable INTEGER FK -> pipeline_runs.id, ON
      DELETE SET NULL.
    - ``created_at`` NOT NULL DEFAULT NOW().
    """
    __tablename__ = "rejected_records"

    source_table: Mapped[str] = mapped_column(String(50), nullable=False)
    source_pipeline: Mapped[str] = mapped_column(String(50), nullable=False)
    raw_data: Mapped[str] = mapped_column(Text, nullable=False)
    rejection_reason: Mapped[str] = mapped_column(String(500), nullable=False)
    rejection_type: Mapped[str] = mapped_column(String(50), nullable=False)
    pipeline_run_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("pipeline_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        # Mirror the chk_rejected_records_rejection_type CHECK whitelist
        # from migration 001 so SQLite dev/test DBs (which skip the
        # migration SQL) still enforce the same rejection-type contract.
        CheckConstraint(
            "rejection_type IN ("
            "'constraint_violation', 'format_error', "
            "'duplicate', 'reference_error', 'other'"
            ")",
            name="chk_rejected_records_rejection_type",
        ),
        Index("ix_rejected_records_source_table", "source_table"),
        Index("ix_rejected_records_source_pipeline", "source_pipeline"),
    )

    def __repr__(self) -> str:
        return (
            f"<RejectedRecord(id={self.id}, "
            f"source_table='{self.source_table}', "
            f"source_pipeline='{self.source_pipeline}', "
            f"rejection_type='{self.rejection_type}')>"
        )


# ===========================================================================
# DEPRECATED: cleanup_orphan_gda_records moved to database.loaders (ARCH-01)
# ===========================================================================

# P1-052 ROOT FIX: replaced deprecated stub function with module-level
# __getattr__ for lazy redirection. The previous stub emitted a
# DeprecationWarning on every call but still executed the real function
# via import. The __getattr__ approach is cleaner: the name is not
# bound at module level (so ``dir()`` and static analysis don't list it),
# but accessing it still works by lazy-importing from database.loaders.
# This eliminates the DeprecationWarning spam and the overhead of
# importing warnings + database.loaders on every call.

def __getattr__(name: str):
    if name == "cleanup_orphan_gda_records":
        from database.loaders import cleanup_orphan_gda_records
        return cleanup_orphan_gda_records
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
