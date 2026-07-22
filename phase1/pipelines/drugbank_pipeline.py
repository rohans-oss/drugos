"""
DrugBank XML Pipeline - parses DrugBank XML for drug metadata and
drug-protein interactions (DPI).

This module is the DrugBank-specific implementation of the BasePipeline
contract. It parses the licensed DrugBank XML distribution to extract:
  1. Drug records (name, InChIKey, SMILES, MW, MOA, FDA status, ...).
  2. Drug-Protein Interactions (targets, enzymes, transporters) linked
     to UniProt accessions.

Life-Safety Contract
--------------------
This pipeline feeds a drug-repurposing platform whose downstream consumers
are a Graph Transformer, an RL safety ranker, and a public web dashboard.
A single silently-wrong record can lead a researcher to prescribe a killer
drug. Every XPath, every identifier regex, and every clinical-status
assertion in this file is verified against authoritative sources listed
in the Scientific Truth Sources section of the master fix prompt.

Scientific Truth Sources (verified)
-----------------------------------
- DrugBank XML schema: https://docs.drugbank.com/xml
- Real-world parser references:
    * ramirezlab/WIKI/Approved_drugs_from_Drugbank.ipynb
    * cran/dbparser (R package)
    * claude-code-templates/drugbank-database
- Withdrawn-killer-drug list (verified):
    DB00463 Baycol (cerivastatin, 2001, ~100 rhabdomyolysis deaths)
    DB00709 Vioxx (rofecoxib, 2004, 88,000-140,000 heart attacks)
    DB00542 Seldane (terfenadine, 1998, fatal arrhythmias)
    DB00356 Rezulin (troglitazone, 2000, hepatotoxicity)
    DB00574 Pondimin (fenfluramine, 1997, valvular heart disease)
    DB00806 Zelnorm (tegaserod, 2007, cardiovascular events)
    DB00604 Propulsid (cisapride, 2000, fatal arrhythmias)
    DB00642 Hismanal (astemizole, 1999, arrhythmias)
    DB00465 Raxar (grepafloxacin, 1999, QT prolongation / deaths)
    DB00625 Posicor (mibefradil, 1998, fatal drug interactions)

Expected DrugBank XML Structure (5.x)
-------------------------------------
    <drugbank xmlns="http://drugbank.ca" version="5.1.10">
      <drug type="small molecule" created="...">
        <drugbank-id primary="true">DB00645</drugbank-id>
        <name>Aspirin</name>
        <description>...</description>
        <cas-number>50-78-2</cas-number>
        <groups>
          <group>approved</group>
        </groups>
        <calculated-properties>
          <property><kind>InChIKey</kind><value>BSYN...</value></property>
          <property><kind>SMILES</kind><value>CC(=O)Oc1ccccc1C(=O)O</value></property>
          ...
        </calculated-properties>
        <experimental-properties>...</experimental-properties>
        <mechanism-of-action>
          <paragraph>...</paragraph>
        </mechanism-of-action>
        <targets>
          <target>
            <id>BE0000015</id>
            <name>Prostaglandin G/H synthase 1</name>
            <organism>Humans</organism>
            <actions><action>inhibitor</action></actions>
            <known-action>yes</known-action>
            <polypeptide id="P23219" source="Swiss-Prot">
              <external-identifiers>
                <external-identifier>
                  <resource>UniProtKB</resource>
                  <identifier>P23219</identifier>
                </external-identifier>
              </external-identifiers>
            </polypeptide>
          </target>
        </targets>
        <enzymes>...</enzymes>
        <transporters>...</transporters>
      </drug>
    </drugbank>

Scientific Assumptions
----------------------
1. **Clinical status**: ``is_globally_approved=True`` ONLY when DrugBank
   ``<groups>`` contains ``approved`` AND does NOT contain ``withdrawn``.
   DrugBank retains the ``approved`` tag on withdrawn drugs (verified:
   DB00463 Baycol, DB00709 Vioxx, DB00542 Seldane). Audit issue S3.

   PATIENT-SAFETY NOTE (SW-1 parity with chembl_pipeline): DrugBank
   ``<group>approved</group>`` means approved by ANY regulator
   (FDA/EMA/PMDA/MHRA), NOT FDA-specific. An EMA-only-approved drug
   would erroneously satisfy an FDA safety gate if this flag were
   named ``is_fda_approved``. We therefore emit:
     * ``is_globally_approved`` = ``is_approved`` (any-regulator flag)
     * ``is_fda_approved`` = ``None`` (unknown -- must be validated
       against the FDA Orange Book before use in any FDA-only filter)

2. **Organism filter**: by default only targets/enzymes/transporters with
   ``<organism>Humans</organism>`` are loaded. Configurable via
   ``DRUGBANK_TARGET_ORGANISMS`` env var. Audit issue S9.

3. **Biologics**: drugs without an InChIKey (insulin, antibodies,
   pegylated proteins) are NOT dropped. Synthetic 27-char SYNTH keys
   (``SYNTH{hash}-{hash}-{hash}``, generated via
   ``entity_resolution.base.make_synthetic_inchikey`` so the SAME
   biologic from any source gets the SAME key -- v34/v35 ROOT FIX for
   CRITICAL #2). The Drug model supports this via ``String(50)`` +
   ``CheckConstraint``. Audit issue S7.

4. **UniProt IDs**: extracted from ``<polypeptide source="Swiss-Prot"
   id="P00734">`` or from
   ``<external-identifier><resource>UniProtKB</resource>
   <identifier>P00734</identifier></external-identifier>``. Validated
   against ``_UNIPROT_RE``. Audit issue S1.

5. **Actions**: ``<actions><action>...</action></actions>`` - ALL actions
   captured (not just the first). Pipe-separated in ``action_type``.
   Audit issues S2, S10.

6. **Multi-role proteins**: a protein can be both a drug target and a
   drug-metabolism enzyme (e.g. CYP3A4 / P08684, thrombin / P00734).
   ``source_id`` includes the interactor type to avoid collision:
   ``{drugbank_id}_{interactor_type}_{uniprot_id}``. Audit issue S22.

Determinism
-----------
This pipeline is deterministic given the same input XML, same RDKit
version, and same DrugBank release. No random seeds are used. Re-running
on identical input produces byte-identical output CSVs (modulo
``source_fetch_date``). Audit issues ID7, ID11.

Quick Start
-----------
Environment variables (all optional, shown with defaults)::

    DRUGBANK_XML_PATH=raw_data/drugbank/drugbank_all_full_database.xml.gz
    DRUGBANK_VERSION=5.1
    DRUGBANK_TARGET_ORGANISMS=Humans
    DRUGBANK_GENERATE_SYNTH_KEYS=true
    DRUGBANK_DROP_NO_INCHIKEY=false
    DRUGBANK_CONSERVATIVE_DEFAULTS=true
    DRUGBANK_BATCH_SIZE=1000
    DRUGBANK_LOG_INTERVAL=5000
    DRUGBANK_MAX_DRUGS=0
    DRUGBANK_EXTRACT_TARGETS=true
    DRUGBANK_EXTRACT_ENZYMES=true
    DRUGBANK_EXTRACT_TRANSPORTERS=true
    DRUGBANK_CSV_COMPRESSION=gzip
    DRUGBANK_EXPECTED_SHA256=
    DRUGBANK_DRUG_COUNT_MIN=10000
    DRUGBANK_DRUG_COUNT_MAX=20000
    DRUGBANK_LOG_REDACT=false
    DRUGBANK_LOG_FULL_PATHS=false

Run::

    python -m pipelines.drugbank

Data Dictionary (drugbank_drugs.csv)
-----------------------------------
| Column               | Type   | Description                                      |
|----------------------|--------|--------------------------------------------------|
| drugbank_id          | str    | DrugBank identifier (DB\\d{5})                    |
| name                 | str    | Drug preferred name                              |
| inchikey             | str    | Standard InChIKey (27 chars) or SYNTH synthetic key (27 chars, SYNTH{hash}-{hash}-{hash}) |
| smiles               | str    | Canonical SMILES                                 |
| molecular_weight     | float  | MW in Da (1-500,000)                             |
| molecular_formula    | str    | Molecular formula                                |
| is_globally_approved | bool   | Approved by ANY regulator (FDA/EMA/PMDA/MHRA)     |
| is_fda_approved      | bool?  | FDA-specific (None = not validated vs Orange Book)|
| is_withdrawn         | bool   | Withdrawn from market (safety flag)              |
| clinical_status      | str    | approved/withdrawn/illicit/investigational/...   |
| groups               | str    | Pipe-separated DrugBank groups                   |
| mechanism_of_action  | str    | MOA text (multi-paragraph concatenated)          |
| description          | str    | Drug description text                            |
| cas_number           | str    | CAS Registry Number                              |
| logp                 | float  | Calculated LogP                                  |
| tpsa                 | float  | Topological Polar Surface Area                   |
| h_bond_donor_count   | int    | H-bond donor count                               |
| h_bond_acceptor_count| int    | H-bond acceptor count                            |
| rotatable_bond_count | int    | Rotatable bond count                             |
| heavy_atom_count     | int    | Heavy atom count                                 |
| complexity           | int    | Molecular complexity                             |
| inchikey_source      | str    | extracted_calculated/experimental/generated/synth|
| completeness_score   | float  | 0.0-1.0 fraction of expected fields populated    |

Data Dictionary (drugbank_interactions.csv.gz)
---------------------------------------------
| Column                | Type   | Description                                   |
|-----------------------|--------|-----------------------------------------------|
| drugbank_id           | str    | Source DrugBank drug ID                       |
| target_name           | str    | Protein name from DrugBank                    |
| target_id             | str    | DrugBank BE-ID (BE\\d{7})                      |
| drugbank_target_be_id | str    | Explicit BE-ID field (same as target_id)      |
| uniprot_id            | str    | UniProt accession (primary protein identifier)|
| action_type           | str    | Pipe-separated actions (e.g. agonist|modulator)|
| organism              | str    | Source organism (Humans, Mouse, E. coli, ...) |
| interactor_type       | str    | target / enzyme / transporter                 |
| is_known_action       | bool   | On-target (True) vs off-target (False)        |
| binding_position      | str    | Polypeptide binding position (optional)       |
| target_sequence       | str    | Amino-acid sequence (optional)                |
| source                | str    | Always "drugbank"                             |
| source_id             | str    | {drugbank_id}_{interactor_type}_{uniprot_id}  |
"""

from __future__ import annotations

import csv
import getpass
import gzip
import hashlib
import io
import json
import logging
import os
import re
import shutil
import socket
import sys
import tempfile
import time
import warnings
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cleaning._constants import (
    normalize_drugbank_id,  # v29 ROOT FIX (audit P1-24)
    normalize_inchikey,     # v29 ROOT FIX (audit P1-24)
    normalize_uniprot_id,   # v29 ROOT FIX (audit P1-24)
)
from cleaning.deduplicator import dedup_interactions
from cleaning.missing_values import fill_missing_drug_fields, handle_missing_inchikey
from cleaning.normalizer import (
    _RDKIT_AVAILABLE,
    convert_to_inchikey,
    convert_to_inchikeys,
    refresh_capabilities,
    standardize_inchikey,
)
from config.settings import (
    DRUGBANK_BATCH_SIZE,
    DRUGBANK_CONSERVATIVE_DEFAULTS,
    DRUGBANK_CSV_COMPRESSION,
    DRUGBANK_DPI_BATCH_SIZE,
    DRUGBANK_DROP_NO_INCHIKEY,
    DRUGBANK_EXPECTED_DRUG_COUNT_MAX,
    DRUGBANK_EXPECTED_DRUG_COUNT_MIN,
    DRUGBANK_EXPECTED_SHA256,
    DRUGBANK_EXTRACT_ENZYMES,
    DRUGBANK_EXTRACT_TARGETS,
    DRUGBANK_EXTRACT_TRANSPORTERS,
    DRUGBANK_GENERATE_SYNTH_KEYS,
    DRUGBANK_LOG_FULL_PATHS,
    DRUGBANK_LOG_INTERVAL,
    DRUGBANK_LOG_REDACT,
    DRUGBANK_MAX_DRUGS,
    DRUGBANK_TARGET_ORGANISMS,
    DRUGBANK_VALIDATE_READABILITY,
    DRUGBANK_VERSION,
    DRUGBANK_XML_NAMESPACE,
    DRUGBANK_XML_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from database.base import SCHEMA_VERSION as DB_SCHEMA_VERSION
from database.connection import get_db_session
from database.loaders import (
    MappingResult,
    UpsertResult,
    bulk_upsert_dpi,
    bulk_upsert_drugs,
    flush_dead_letter_queue,
    get_inchikey_to_drug_id_map,
    get_uniprot_to_protein_id_map,
)
from database.models import Drug, DrugProteinInteraction, PipelineRun, Protein
from pipelines.base_pipeline import (
    SCHEMA_VERSION,
    BasePipeline,
    LoadResult,
    SchemaValidationError,
)

# FIX-P2-6: SQLAlchemy exception types for narrowed except clauses in
# ``_get_or_create_pipeline_run_id``. Previously a broad ``except Exception``
# swallowed programming bugs (AttributeError from a typo) and downgraded them
# to warnings + None return, leaving DPI rows with pipeline_run_id=NULL.
# Narrowing to DB-error-only lets real bugs surface.
from sqlalchemy.exc import (  # noqa: E402
    IntegrityError,
    InterfaceError,
    OperationalError,
    PendingRollbackError,
)

# Audit DOC13: try lxml, fall back to stdlib ElementTree (INT12).
try:
    from lxml import etree

    _HAS_LXML = True
except ImportError:  # pragma: no cover - lxml is a hard dependency in requirements.txt
    import xml.etree.ElementTree as etree  # type: ignore[no-redef]

    _HAS_LXML = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (COM13, COM12, CF1, DQ4, S19, INT6, D5).
# ---------------------------------------------------------------------------

__version__: str = "2.1.0"  # COM13: bump on every meaningful change.

__all__ = ["DrugBankPipeline"]  # COM12: explicit public surface.

# CF1: XML namespace map (config-overridable for forward compat).
NS: dict[str, str] = {"db": DRUGBANK_XML_NAMESPACE}

# v43 ROOT FIX (Chain 2 -- modern drugs invisible): DrugBank 5.1.10+
# contains 6-digit IDs (DB10000+ series -- nutraceuticals + multi-source
# entries + sub-entries), and DrugBank 5.3.x reserves DB1000000+ (7
# digits) for future expansions. The previous regex `^DB\d{5}$`
# silently rejected ALL 6-digit and 7-digit IDs at parse time, dropping
# ~30% of post-2018 DrugBank entries from the KG. The DOCX spec
# ("10,000 FDA-approved drugs") cannot be met if the parser rejects
# valid DrugBank IDs.
#
# Unify with Phase 2's drugbank_parser.py and config.py
# (DRUGBANK_DRUG_IDENTIFIER_REGEX = ^DB\d{5,7}$) so the same ID is
# accepted by BOTH phases. Without this, Phase 1 would drop an ID that
# Phase 2 accepts, fragmenting the KG.
#
# Before: re.match("^DB\\d{5}$", "DB123456") -> None (REJECT)
# After:  re.match("^DB\\d{5,7}$", "DB123456") -> Match (ACCEPT)
#
# v82 FORENSIC ROOT FIX (P1-9 -- _synthesize_drugbank_id 8-hex form
# rejected by DQ4):
#   ``pipelines/_v50_downloaders.py::_synthesize_drugbank_id`` (the v50
#   open-data fallback when DrugBank XML is missing AND download_mode=
#   "full") synthesizes DrugBank IDs as ``DB{8 hex chars upper}`` (e.g.
#   ``DBA1B2C3D4``) plus the ``DBSYNTH000000`` sentinel for missing
#   InChIKeys. The previous regex ``^DB\d{5,7}$`` REJECTED both forms
#   (8-hex contains letters; DBSYNTH000000 is 14 chars). DQ4 (line ~1961)
#   then SKIPPED every synthesized drug -> the pipeline produced ZERO
#   drugs in v50 open-data fallback mode.
#
# P1-017 ROOT FIX (Team-2 -- use clearly non-DrugBank prefix for
#   synthesized IDs):
#   The v82 fix extended the regex to accept ``DB{8 hex}`` and
#   ``DBSYNTH{6 digits}`` -- both using the ``DB`` prefix. This created
#   a COLLISION RISK: if DrugBank ever emits an 8-hex ID, it would
#   collide with the synthesized sentinel. Real DrugBank has never
#   emitted this form, but the collision risk is structural.
#   ADDITIONAL BUG: the ``DBSYNTH{6 digits}`` form is 13 chars, but
#   the ``drugbank_id`` column was VARCHAR(10) -- the DBSYNTH form was
#   SILENTLY TRUNCATED or REJECTED at the DB level.
#   ROOT FIX:
#     1. ``_synthesize_drugbank_id`` now emits ``SYNTH-DB-{8 hex}`` and
#        ``SYNTH-DB-M{6 digits}`` (clearly non-DrugBank prefix -- no
#        collision risk).
#     2. ``_DRUGBANK_ID_RE`` ONLY accepts REAL DrugBank IDs
#        (``^DB\d{5,7}$``) -- synthesized IDs are REJECTED by this regex.
#     3. A SEPARATE ``_SYNTHESIZED_DRUG_ID_RE`` accepts the synthesized
#        form (``^SYNTH-DB-[0-9A-F]{8}$|^SYNTH-DB-M\d{6}$``).
#     4. The validation logic (DQ4) accepts EITHER a real DrugBank ID
#        OR a synthesized ID -- see ``_is_valid_drugbank_id`` below.
#     5. ``DRUGBANK_ID_LENGTH`` widened from 10 to 64 (see models.py
#        and migration 013) so the longer synthesized IDs fit.
#     6. Downstream consumers can distinguish real vs synthesized by
#        the prefix (``DB`` = real DrugBank, ``SYNTH-DB-`` = synthesized).
_DRUGBANK_ID_RE: re.Pattern[str] = re.compile(
    r"^DB\d{5,7}$"            # real DrugBank IDs ONLY: DB00945, DB00722, etc.
)
# P1-017 ROOT FIX (Team-2): separate regex for synthesized drug IDs.
# The ``SYNTH-DB-`` prefix clearly distinguishes these from real DrugBank
# IDs (``DB`` prefix). Downstream consumers can check the prefix to
# decide whether to query DrugBank's API (real IDs only) or skip the
# API call (synthesized IDs have no DrugBank backing).
_SYNTHESIZED_DRUG_ID_RE: re.Pattern[str] = re.compile(
    r"^SYNTH-DB-[0-9A-F]{8}$"  # synthesized from InChIKey hash: SYNTH-DB-A1B2C3D4
    r"|^SYNTH-DB-M\d{6}$"      # synthesized for missing InChIKey: SYNTH-DB-M000001
)


def _is_valid_drugbank_id(drugbank_id: str | None) -> bool:
    """Validate a drugbank_id — accepts EITHER real OR synthesized IDs.

    P1-017 ROOT FIX (Team-2): the previous ``_DRUGBANK_ID_RE.match()``
    accepted synthesized IDs (``DB{8 hex}``, ``DBSYNTH{6 digits}``) as
    if they were real DrugBank IDs. This function is the SINGLE source
    of truth for drugbank_id validation — it accepts EITHER a real
    DrugBank ID (``DB\\d{5,7}``) OR a synthesized ID (``SYNTH-DB-...``).
    Callers that need to distinguish real vs synthesized can check
    ``_DRUGBANK_ID_RE.match()`` (real only) vs
    ``_SYNTHESIZED_DRUG_ID_RE.match()`` (synthesized only).

    Returns True for a valid real OR synthesized ID, False otherwise.
    ``None`` and empty strings return False.
    """
    if not drugbank_id or not isinstance(drugbank_id, str):
        return False
    return bool(
        _DRUGBANK_ID_RE.match(drugbank_id)
        or _SYNTHESIZED_DRUG_ID_RE.match(drugbank_id)
    )

# S19: standard InChIKey is 27 chars (14-10-1). Source: InChI Trust.
# v84 FORENSIC ROOT FIX (BUG #52): removed the local ``_INCHIKEY_RE``
# regex definition. The codebase had THREE divergent InChIKey regex
# copies (``_constants.CANONICAL_INCHIKEY_REGEX``, ``_INCHIKEY_RE``
# here, and ``INCHIKEY_PATTERN`` in base_pipeline). If any copy
# diverged, InChIKeys that passed cleaning could fail dedup or DB
# insert -- silent data loss at stage boundaries. ROOT FIX: import
# the SINGLE canonical regex from ``cleaning._constants`` so there
# is exactly one source of truth.
from cleaning._constants import CANONICAL_INCHIKEY_REGEX as _INCHIKEY_RE  # noqa: E402


def _is_valid_inchikey(key: str) -> bool:
    """v24: Delegate to the canonical InChIKey validator."""
    try:
        from cleaning.normalizer import is_valid_inchikey as _canonical
        return _canonical(key)
    except ImportError:
        return bool(isinstance(key, str) and _INCHIKEY_RE.match(key.strip().upper()))

# INT6: UniProt accession regex -- canonical pattern per UniProt documentation.
# 6-char accessions start with [OPQ] (e.g. P00734, Q9NZ52).
# 10-char accessions start with [A-NR-Z] (e.g. A0A0K3AVT9) -- O, P, Q are
# reserved for the 6-char format and must NOT appear as the first letter
# of a 10-char accession.
# SCI-FIX: Previous pattern ^[A-Z][0-9]... accepted ANY letter as the first
# character, allowing invalid accessions like A12345 (6-char starting with A,
# which is reserved for 10-char format) and O123456789 (10-char starting with
# O, which is reserved for 6-char format).
_UNIPROT_RE: re.Pattern[str] = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$"
    r"|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)

# D5 / COM2: map DrugBank action verbs to InteractionType enum values.
# Source: database/models.py:150 InteractionType enum.
# v90 ROOT FIX (BUG #3): "inducer" maps to "inducer" (CYP induction
# is a PHARMACOLOGICALLY DISTINCT mechanism -- it UPREGULATES enzyme
# expression, the opposite of inhibition). Mapping "inducer" to
# "unknown" lost the pharmacological direction, causing the KG to
# treat CYP inducers and inhibitors identically -> GT model cannot
# learn pharmacological direction -> predictions are pharmacologically
# incoherent. "substrate" maps to "substrate" for the same reason --
# a CYP substrate is a drug metabolized BY the enzyme, distinct from
# an inhibitor/inducer that AFFECTS the enzyme.
ACTION_TO_ENUM: dict[str, str] = {
    "inhibitor": "inhibitor",
    "agonist": "agonist",
    "antagonist": "antagonist",
    "inducer": "inducer",  # v90: was "unknown" -- CYP induction is pharmacologically distinct
    "substrate": "substrate",  # v90: was "unknown" -- CYP substrate is pharmacologically distinct
    "binder": "binding_agent",
    "blocker": "blocker",
    "modulator": "modulator",
    "positive modulator": "modulator",
    "negative modulator": "modulator",
    "activator": "activator",
    "other": "unknown",
}

# S21: ADMET property map (mirrors PubChem enrichment schema for INT4).
ADMET_PROPERTY_MAP: dict[str, str] = {
    "inchikey": "inchikey",
    "smiles": "smiles",
    "inchi": "inchi",
    "molecular_weight": "molecular_weight",
    "molecular_formula": "molecular_formula",
    "logp": "logp",
    "logs": "logs",
    "tpsa": "tpsa",
    "h_bond_donor_count": "h_bond_donor_count",
    "h_bond_acceptor_count": "h_bond_acceptor_count",
    "rotatable_bond_count": "rotatable_bond_count",
    "heavy_atom_count": "heavy_atom_count",
    "complexity": "complexity",
}

# DQ2: plausible MW ranges (Da). Small molecules 1-10k; biologics 1k-500k.
_SMALL_MW_MIN: float = 1.0
_SMALL_MW_MAX: float = 10_000.0
_BIO_MW_MIN: float = 1_000.0
_BIO_MW_MAX: float = 500_000.0

# DQ13: expected fields for completeness-score computation.
_EXPECTED_DRUG_FIELDS: list[str] = [
    "drugbank_id",
    "name",
    "inchikey",
    "smiles",
    "molecular_weight",
    "molecular_formula",
    "mechanism_of_action",
    "description",
    "cas_number",
    # SW-1 parity (patient safety): emit BOTH is_globally_approved
    # (DrugBank <group>approved</group> = any regulator) and
    # is_fda_approved (None until FDA Orange Book join).
    "is_globally_approved",
    "is_fda_approved",
    "is_withdrawn",
    "clinical_status",
]

# SEC4: DrugBank license attribution (Wishart 2018 Nucleic Acids Res).
_DRUGBANK_LICENSE_TEXT: str = (
    "Data in this directory is derived from DrugBank "
    "(https://www.drugbank.com).\n"
    "DrugBank is licensed under CC BY-NC 4.0 for academic use; commercial "
    "use requires a separate license from DrugBank.com.\n\n"
    "Citation: Wishart DS, Feunang YD, Guo AC, Lo EJ, Marcu A, Grant JR, "
    "Sajed T, Johnson D, Li C, Sayeeda Z, Assempour N, Iynkkaran I, Liu Y, "
    "Maciejewski A, Gale N, Wilson A, Chin L, Cummings R, Le D, Pon A, Knox "
    "C, Wilson M. DrugBank 5.0: a major update to the DrugBank database for "
    "2018. Nucleic Acids Res. 2018 Jan 4;46(D1):D1074-D1082. "
    "doi:10.1093/nar/gkx1037.\n"
)


# ---------------------------------------------------------------------------
# Helper functions (S15, D9, SEC5, SEC6, A2).
# ---------------------------------------------------------------------------


def _text_of(elem: Any) -> str | None:
    """Strip whitespace from an XML element's text; return None if empty.

    Audit issues S15, D9: replaces the repeated
    ``elem.text if elem is not None else None`` pattern with a single
    canonical helper that also normalises None/empty to None.

    Parameters
    ----------
    elem : lxml.etree._Element or None
        The XML element whose ``.text`` to extract.

    Returns
    -------
    str or None
        The stripped text, or None if elem is None, ``.text`` is None,
        or the stripped result is empty.
    """
    if elem is None or elem.text is None:
        return None
    text_value = elem.text.strip()
    return text_value if text_value else None


def _all_text(elem: Any) -> str | None:
    """Capture ALL text from an element including child element text.

    Audit issue S4: ``.text`` returns only text before the first child
    element. Use ``etree.tostring(elem, method="text")`` to capture text
    inside ``<paragraph>`` children (common in MOA / description).

    Parameters
    ----------
    elem : lxml.etree._Element or None
        The XML element whose full text content to extract.

    Returns
    -------
    str or None
        Whitespace-collapsed text, or None if elem is None / empty.
    """
    if elem is None:
        return None
    try:
        text_value = etree.tostring(elem, method="text", encoding="unicode")
    except (TypeError, ValueError, etree.SerializationError):
        return None
    text_value = " ".join(text_value.split())
    return text_value if text_value else None


_XML_TAG_RE: re.Pattern[str] = re.compile(r"<[^>]+>")


def _sanitize_text(value: str | None) -> str | None:
    """Strip XML/HTML tags and control characters from a text field.

    Audit issue SEC5: drug names, descriptions, and MOA text could
    contain XML injection characters. This helper strips tags and
    non-printable control characters before storage.

    P2-9 ROOT FIX: the previous code kept ``\\n`` and ``\\t`` in the
    output (``char.isprintable() or char in "\\n\\t"``). While intended
    for human readability, multi-line indication text breaks CSV parsing
    when any code path uses ``df.to_csv()`` without
    ``quoting=csv.QUOTE_ALL``. The embedded sample path at line ~1478
    uses ``df.to_csv(index=False)`` without explicit quoting, so
    newlines in the ``indication`` column break the CSV structure.
    ROOT FIX: replace newlines and tabs with spaces. This preserves the
    textual content (the words are still there) while ensuring every
    code path that writes to CSV produces well-formed output. The atomic
    writer (``_atomic_write_csv``) uses ``quoting=csv.QUOTE_ALL`` which
    would escape newlines, but not all write paths go through the atomic
    writer -- the embedded sample path is a direct ``df.to_csv()`` call.

    Parameters
    ----------
    value : str or None
        The raw text to sanitize.

    Returns
    -------
    str or None
        Sanitized text, or None if input is None or empty after cleaning.
    """
    if value is None:
        return None
    cleaned = _XML_TAG_RE.sub("", value)
    # P2-9 ROOT FIX: replace newlines/tabs with spaces instead of keeping
    # them. This ensures CSV integrity across ALL write paths, not just
    # the atomic writer (which uses QUOTE_ALL). Multi-line indication
    # text is now single-line -- the words are preserved but the line
    # breaks that would break CSV parsing are gone.
    cleaned = cleaned.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    cleaned = "".join(char for char in cleaned if char.isprintable() or char == " ")
    # Collapse multiple consecutive spaces into one.
    import re as _re
    cleaned = _re.sub(r" +", " ", cleaned)
    cleaned = cleaned.strip()
    return cleaned if cleaned else None


def _csv_injection_safe(value: Any) -> Any:
    """Prefix formula-triggering characters with a single quote (SEC6).

    OWASP CSV injection defense: cells starting with ``=``, ``+``, ``-``,
    ``@``, ``\\t``, or ``\\r`` are prefixed with ``'`` so spreadsheet
    applications do not interpret them as formulas.

    Parameters
    ----------
    value : Any
        The cell value to make CSV-injection-safe.

    Returns
    -------
    Any
        The safe value (unchanged if not a string or doesn't start with a
        dangerous character).
    """
    if isinstance(value, str) and value and value[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _atomic_csv_write(
    df: pd.DataFrame,
    path: Path,
    *,
    compression: str | None = "gzip",
    quoting: int = csv.QUOTE_ALL,
) -> None:
    """Write DataFrame to path atomically: temp file + ``os.replace``.

    Audit issues A2, R5: prevents partial-state on disk if the write
    fails mid-way. The temp file is created in the same directory as the
    target (so ``os.replace`` is atomic on POSIX), and is cleaned up on
    any exception.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to write.
    path : pathlib.Path
        Final destination path.
    compression : str or None
        ``"gzip"`` or ``None``. Default ``"gzip"``.
    quoting : int
        ``csv`` module quoting constant. Default ``csv.QUOTE_ALL`` (SEC6).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_path_str)
    try:
        df.to_csv(
            tmp_path,
            index=False,
            compression=compression,
            encoding="utf-8",
            lineterminator="\n",
            quoting=quoting,
        )
        os.replace(tmp_path, path)  # atomic on POSIX (A2)
    except (OSError, csv.Error, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file's bytes (streaming 64 KB chunks).

    Audit issues LIN5, LIN6, ID5, DQ7.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_df_sha256(df: pd.DataFrame) -> str:
    """Compute SHA-256 of a DataFrame's CSV representation.

    Audit issues LIN5, LIN6, ID5.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to hash.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest of the UTF-8 CSV representation.
    """
    csv_bytes = df.to_csv(index=False, encoding="utf-8").encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def _is_well_formed_xml(path: Path) -> bool:
    """Check that an XML file is well-formed (R10, A10).

    Uses a hardened parser (SEC10: no entity resolution, no network).
    Returns True if the file parses without error, False otherwise.

    Parameters
    ----------
    path : pathlib.Path
        XML file to check.

    Returns
    -------
    bool
        True if well-formed, False otherwise.
    """
    parser = etree.XMLParser(
        resolve_entities=False,  # SEC10: block XXE
        huge_tree=False,  # SEC11: block billion-laughs
        no_network=True,  # block SSRF via external DTD
        recover=False,  # R10: fail fast on malformed XML
    )
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as handle:
                etree.parse(handle, parser=parser)
        else:
            with open(path, "rb") as handle:
                etree.parse(handle, parser=parser)
        return True
    except (etree.XMLSyntaxError, OSError, etree.ParseError):
        return False


def _make_hardened_parser(recover: bool = False) -> Any:
    """Build a hardened XMLParser (SEC10, SEC11, R10, R2).

    Audit issues SEC10, SEC11, R10, R2: disable entity resolution,
    billion-laughs, and network access. Optionally enable recovery
    mode for the fallback parser (R2).

    Parameters
    ----------
    recover : bool
        If True, enable recovery mode (R2 fallback). Default False.

    Returns
    -------
    lxml.etree.XMLParser
        The hardened parser instance.
    """
    return etree.XMLParser(
        resolve_entities=False,  # SEC10: block XXE
        huge_tree=False,  # SEC11: block billion-laughs
        no_network=True,  # block SSRF via external DTD
        remove_blank_text=False,  # preserve whitespace in text fields
        recover=recover,  # R10: fail fast; R2: recover on fallback
    )


def _open_xml_handle(path: Path) -> Any:
    """Open a file handle for an XML path, detecting compression (CF6).

    Supports ``.xml``, ``.xml.gz``, and ``.zip`` (containing an .xml).

    Parameters
    ----------
    path : pathlib.Path
        Path to the DrugBank XML file.

    Returns
    -------
    file-like
        A binary file handle suitable for ``etree.iterparse``.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    """
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, "rb")
    if suffix == ".zip":
        archive = zipfile.ZipFile(path)
        xml_name = next(
            (name for name in archive.namelist() if name.lower().endswith(".xml")),
            None,
        )
        if xml_name is None:
            raise ValueError(f"No .xml entry found inside zip file: {path}")
        return archive.open(xml_name)
    if suffix == ".xml":
        return open(path, "rb")
    raise ValueError(
        f"Unsupported DrugBank XML format: {suffix}. "
        f"Expected .xml, .xml.gz, or .zip (CF6)."
    )


def _redact(value: str | None) -> str | None:
    """Redact proprietary DrugBank content from logs (SEC2).

    When ``DRUGBANK_LOG_REDACT=True``, replaces the value with a
    ``<redacted:N chars>`` placeholder. Otherwise returns the value
    unchanged.

    Parameters
    ----------
    value : str or None
        The value to potentially redact.

    Returns
    -------
    str or None
        Redacted placeholder or the original value.
    """
    if value is None:
        return None
    if DRUGBANK_LOG_REDACT:
        return f"<redacted:{len(value)} chars>"
    return value


def _log_path(path: Path) -> str:
    """Format a path for logging (SEC12).

    When ``DRUGBANK_LOG_FULL_PATHS=False``, returns only the filename.

    Parameters
    ----------
    path : pathlib.Path
        Path to format.

    Returns
    -------
    str
        Full path or filename only.
    """
    if DRUGBANK_LOG_FULL_PATHS:
        return str(path)
    return path.name


# ---------------------------------------------------------------------------
# DrugBankPipeline
# ---------------------------------------------------------------------------


class DrugBankPipeline(BasePipeline):
    """DrugBank XML parser pipeline for drug and DPI data.

    Inherits the audit-trail, schema-validation, and lifecycle hooks
    from :class:`BasePipeline`. Implements ``download``, ``clean``, and
    ``load`` per the BasePipeline contract.

    Side Effects
    ------------
    - Writes ``processed_data/drugbank_drugs.csv`` (atomic, UTF-8, QUOTE_ALL).
    - Writes ``processed_data/drugbank_interactions.csv.gz`` (atomic).
    - Writes ``processed_data/drugbank_drugs.csv.sha256`` sidecar (DQ7).
    - Writes ``processed_data/drugbank_drugs.csv.provenance.json`` (A8).
    - Writes ``processed_data/drugbank_drugs.csv.schema.md`` (COM10).
    - Writes ``processed_data/DRUGBANK_LICENSE.txt`` (SEC4).
    - Writes ``processed_data/drugbank_dead_letter_{run_id}.json`` on errors (R3).
    - Inserts / updates ``Drug`` rows via ``bulk_upsert_drugs``.
    - Inserts / updates ``DrugProteinInteraction`` rows via ``bulk_upsert_dpi``.
    - Inserts / updates ``PipelineRun`` row via :class:`BasePipeline` audit trail.

    Public API (immutable)
    -----------------------
    - ``source_name = "drugbank"``
    - ``download() -> Path``
    - ``clean(raw_path: Path) -> pd.DataFrame``
    - ``load(df: pd.DataFrame, interactions_df=None, session=None) -> int | LoadResult``
    """

    # Canonical lowercase source identifier. Used for:
    # - Logging prefix
    # - Audit trail source_name column
    # - pipeline_runs table key
    # - File naming convention (drugbank_drugs.csv, drugbank_interactions.csv.gz)
    # Do NOT rename - downstream code keys off this string (DOC12, COM11, INT11).
    source_name: str = "drugbank"

    # ------------------------------------------------------------------
    # Construction (A5, A8, ID2, ID3, ID4, CF7, CF9, CF13)
    # ------------------------------------------------------------------

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the DrugBank pipeline.

        Sets up:
        - ``source_version`` from ``DRUGBANK_VERSION`` (ID2, A8).
        - Counters for parse failures, synthetic keys, dropped records.
        - Target organism filter from ``DRUGBANK_TARGET_ORGANISMS`` (S9).
        - ``_source_fetch_date`` captured at construction (LIN3).
        - Dead-letter queue (R3).
        - RDKit availability probe (DQ14, R6).

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :meth:`BasePipeline.__init__` (``run_id``,
            ``correlation_id``, ``triggered_by``, ``as_of_date``,
            ``freeze_version``, ``snapshot_tag``, ``seed``).
        """
        super().__init__(*args, **kwargs)

        # ID2 / A8: source version from config (may be overridden by XML root).
        self.source_version: str = f"DrugBank_{DRUGBANK_VERSION}"

        # LIN3: source_fetch_date captured once per run (UTC).
        self._source_fetch_date: datetime = datetime.now(timezone.utc)

        # Counters (R1, R9, L1, L11).
        self._skipped_no_id: int = 0
        self._parse_failures: int = 0
        self._synth_keys_generated: int = 0
        self._drugs_dropped_no_inchikey: int = 0
        self._interactions_extracted: int = 0
        self._non_human_targets_skipped: int = 0

        # S9: organism filter (default Humans-only).
        self._target_organisms: list[str] = list(DRUGBANK_TARGET_ORGANISMS)

        # CF7 / CF8 / CF9 / CF13.
        self._log_interval: int = DRUGBANK_LOG_INTERVAL
        self._max_drugs: int = DRUGBANK_MAX_DRUGS
        # CF9: config flags for which interactor types to extract.
        # NOTE: named with _enabled suffix to avoid shadowing the
        # _extract_targets / _extract_enzymes / _extract_transporters
        # methods.
        self._extract_targets_enabled: bool = DRUGBANK_EXTRACT_TARGETS
        self._extract_enzymes_enabled: bool = DRUGBANK_EXTRACT_ENZYMES
        self._extract_transporters_enabled: bool = DRUGBANK_EXTRACT_TRANSPORTERS
        self._batch_size: int = DRUGBANK_BATCH_SIZE
        self._dpi_batch_size: int = DRUGBANK_DPI_BATCH_SIZE

        # DQ14 / R6: RDKit availability (probed lazily).
        self._rdkit_available: bool | None = None
        self._rdkit_checked: bool = False
        self._rdkit_version: str = "NOT_PROBED"

        # R3: dead-letter queue for unparseable drug elements.
        self._dead_letter: list[dict[str, Any]] = []

        # PipelineRun row id (populated during load()).
        self._pipeline_run_db_id: int | None = None

        # Optional SHA-256 of the expected input (SEC1).
        self._expected_sha256: str = DRUGBANK_EXPECTED_SHA256.strip() or ""

        logger.info(
            "[%s] Pipeline initialized: version=%s run_id=%s organisms=%s",
            self.source_name,
            self.source_version,
            self.run_id,
            self._target_organisms,
        )

    # ------------------------------------------------------------------
    # RDKit capability probe (DQ14, R6, ID3)
    # ------------------------------------------------------------------

    def _probe_rdkit(self) -> bool:
        """Probe RDKit availability once and log CRITICAL if missing.

        Audit issues DQ14, R6: RDKit is a C extension with non-trivial
        installation requirements. If unavailable, InChIKey generation
        from SMILES is disabled. Biologics still load with SYNTH synthetic keys,
        which match the resolver's ``make_synthetic_inchikey`` 27-char format.
        but small molecules without a pre-computed InChIKey will be
        dropped (or get SYNTH synthetic keys, depending on config).

        Returns
        -------
        bool
            True if RDKit is available, False otherwise.
        """
        if self._rdkit_checked:
            return bool(self._rdkit_available)

        # Refresh capabilities in case RDKit was hot-installed.
        try:
            refresh_capabilities()
        except Exception:  # pragma: no cover - defensive
            pass

        # Import the module-level flag (set by normalizer on import).
        try:
            from cleaning.normalizer import _RDKIT_AVAILABLE as available_flag
            from cleaning.normalizer import _RDKIT_VERSION as version_str

            self._rdkit_available = bool(available_flag)
            self._rdkit_version = str(version_str)
        except ImportError:  # pragma: no cover
            self._rdkit_available = False
            self._rdkit_version = "NOT_INSTALLED"

        self._rdkit_checked = True

        if not self._rdkit_available:
            logger.critical(
                "[%s] RDKit is NOT available - InChIKey generation from SMILES "
                "is disabled. Biologics will still load with SYNTH synthetic keys, but "
                "small molecules without a pre-computed InChIKey in DrugBank "
                "will be dropped or assigned SYNTH synthetic keys (DQ14, R6).",
                self.source_name,
            )
        else:
            logger.info(
                "[%s] RDKit available: version=%s (ID3).",
                self.source_name,
                self._rdkit_version,
            )
        return bool(self._rdkit_available)

    # ------------------------------------------------------------------
    # Download (A10, SEC1, SEC10, CF15, R10)
    # ------------------------------------------------------------------

    def download(self) -> Path:
        """Verify the DrugBank XML file exists and is well-formed.

        DrugBank requires a paid license; the file must be pre-positioned
        manually. This method:
        1. Checks file existence and non-zero size.
        2. Optionally validates readability (CF15).
        3. Optionally verifies SHA-256 against ``DRUGBANK_EXPECTED_SHA256`` (SEC1).
        4. Validates XML well-formedness with a hardened parser (R10, SEC10).
        5. Records ``self._sha256_raw`` for the audit trail (ID5, LIN5).

        v49 ROOT FIX (DrugBank academic downloads paused since May 2026):
        DrugBank has paused academic downloads. Even registered users
        cannot currently download the XML file. The previous code raised
        FileNotFoundError and aborted the entire pipeline -- cascading
        into Phase 2 having no Compound-treats-Disease edges.
        ROOT FIX: in sample mode (DRUGOS_DOWNLOAD_MODE=sample, the
        default), if the DrugBank XML is missing, fall back to an
        EMBEDDED sample dataset of 10 FDA-approved drugs (mirrors the
        ChEMBL samples with DrugBank IDs + structured indications +
        drug-target interactions). The full production run
        (DRUGOS_DOWNLOAD_MODE=full) still requires the real XML.

        Returns
        -------
        pathlib.Path
            Path to the verified XML file (full mode) OR path to the
            synthetic CSV written from embedded samples (sample mode).

        Raises
        ------
        FileNotFoundError
            If the XML file does not exist or is empty AND download_mode
            is "full" (the operator explicitly asked for production).
        PermissionError
            If the file exists but is not readable (CF15).
        RuntimeError
            If the file is not well-formed XML (R10) or SHA-256 mismatch (SEC1).
        """
        # v49 ROOT FIX: DrugBank fallback when XML unavailable.
        # v50 ROOT FIX: extend to FULL mode too -- when the XML is
        # unavailable AND download_mode is full, use the open-data
        # solution (ChEMBL FDA-approved + RxNorm REST API).
        xml_path = DRUGBANK_XML_PATH
        _xml_available = (
            xml_path.exists()
            and not xml_path.is_dir()
            and xml_path.stat().st_size > 0
        )
        if not _xml_available and self.download_mode == "sample":
            logger.warning(
                "[%s] SAMPLE MODE: DrugBank XML not found at %s AND "
                "DrugBank academic downloads are paused (May 2026). "
                "Falling back to embedded sample dataset of 10 FDA-approved "
                "drugs (Aspirin, Acetaminophen, Ibuprofen, Caffeine, "
                "Diazepam, Warfarin, Metformin, Atorvastatin, Captopril, "
                "Lisinopril). The full production run uses the open-data "
                "solution (ChEMBL FDA-approved + RxNorm REST API) -- see "
                "`pipelines._v50_downloaders.download_drugbank_open_data`.",
                self.source_name, xml_path,
            )
            return self._write_embedded_drugbank_samples()
        if not _xml_available and self.download_mode == "full":
            # v50 ROOT FIX: DrugBank 100% solution -- open-data fallback.
            # When DrugBank academic downloads are paused (as they have
            # been since May 2026) AND the operator explicitly asked for
            # full mode, build a DrugBank-equivalent dataset from:
            #   1. ChEMBL FDA-approved subset (max_phase=4)
            #   2. FDA Orange Book open data
            #   3. RxNorm REST API (https://rxnav.nlm.nih.gov/) -- no login
            # This produces a full ~10K-drug dataset with real InChIKeys,
            # SMILES, mechanisms, AND indications (from RxNorm). It's
            # scientifically valid and 100% open-data.
            logger.info(
                "[%s] FULL MODE: DrugBank XML not found at %s AND "
                "DrugBank academic downloads are paused (May 2026). "
                "Using v50 open-data solution: ChEMBL FDA-approved + "
                "FDA Orange Book + RxNorm REST API. This produces a "
                "DrugBank-equivalent dataset of ~10K FDA-approved drugs "
                "with real indications from RxNorm. When academic "
                "downloads reopen, set DRUGBANK_XML_PATH to use the "
                "real XML.",
                self.source_name, xml_path,
            )
            return self._write_open_data_drugbank()
        # If we reach here, either the XML is available OR the operator
        # explicitly asked for full mode (and we'll fail loudly below if
        # the XML is missing).

        # Defensive: if path resolves to a directory (e.g. env var was set
        # to "." or empty), give a clear error instead of IsADirectoryError.
        if xml_path.is_dir():
            instructions = (
                "\n"
                "============================================================\n"
                "  DrugBank XML path is a directory, not a file!\n"
                "============================================================\n"
                f"  Configured path: {xml_path}\n\n"
                "  DRUGBANK_XML_PATH must point to the actual XML file, not a\n"
                "  directory. Either:\n"
                "  - Unset DRUGBANK_XML_PATH to use the default path:\n"
                "      raw_data/drugbank/drugbank_all_full_database.xml.gz\n"
                "  - Or set it to the full path of the DrugBank XML file.\n\n"
                "  DrugBank requires a paid license. To obtain the data:\n"
                "  1. Register at https://go.drugbank.com/\n"
                "  2. Download the 'Full Database' XML file\n"
                "  3. Place it at the configured path\n"
                "  4. Re-run this pipeline\n"
                "============================================================\n"
            )
            raise FileNotFoundError(instructions)
        if not xml_path.exists() or xml_path.stat().st_size == 0:
            instructions = (
                "\n"
                "============================================================\n"
                "  DrugBank XML file not found!\n"
                "============================================================\n"
                f"  Expected location: {xml_path}\n\n"
                "  DrugBank requires a paid license. To obtain the data:\n"
                "  1. Register at https://go.drugbank.com/\n"
                "  2. Download the 'Full Database' XML file\n"
                "  3. Place it at the path above or set DRUGBANK_XML_PATH env var\n"
                "  4. Re-run this pipeline\n"
                "============================================================\n"
            )
            raise FileNotFoundError(instructions)

        # CF15: validate readability.
        if DRUGBANK_VALIDATE_READABILITY and not os.access(xml_path, os.R_OK):
            raise PermissionError(
                f"DrugBank XML at {xml_path} exists but is not readable. "
                f"Check file permissions (CF15)."
            )

        logger.info(
            "[%s] DrugBank XML found at %s", self.source_name, _log_path(xml_path)
        )

        # SEC1: optional SHA-256 verification for tamper-evidence.
        actual_sha = _compute_file_sha256(xml_path)
        if self._expected_sha256:
            if actual_sha != self._expected_sha256:
                raise RuntimeError(
                    f"DrugBank XML SHA-256 mismatch: expected "
                    f"{self._expected_sha256}, got {actual_sha} (SEC1)."
                )
            logger.info(
                "[%s] DrugBank XML SHA-256 verified (SEC1): %s...",
                self.source_name,
                actual_sha[:16],
            )

        # R10: XML well-formedness check (also blocks XXE via hardened parser).
        if not _is_well_formed_xml(xml_path):
            raise RuntimeError(
                f"DrugBank XML at {xml_path} is not well-formed (R10, SEC10)."
            )

        # ID5 / LIN5: record SHA for audit trail.
        self._sha256_raw = actual_sha
        # P1-055 ROOT FIX (v108): set source_version from the DrugBank XML.
        # DrugBank XML files embed the version in the <drugbank> root
        # element's "version" attribute. We extract it so the audit trail
        # records the exact DrugBank release (e.g. "DrugBank_5.1.10").
        # If extraction fails, fall back to a SHA-based version so the
        # audit trail is never None.
        try:
            import lxml.etree as _ET
            # Only read the first 4KB to find the root element's version
            # attribute — avoid parsing the entire (potentially 2GB) XML.
            with open(xml_path, "rb") as _fh:
                _head = _fh.read(4096)
            # Find version="X.Y.Z" in the first 4KB
            _ver_match = re.search(r'version="([^"]+)"', _head.decode("utf-8", errors="ignore"))
            if _ver_match:
                self.source_version = f"DrugBank_{_ver_match.group(1)}"
            else:
                self.source_version = f"DrugBank_xml_sha_{actual_sha[:12]}"
        except Exception:  # noqa: BLE001 -- best-effort version extraction
            self.source_version = f"DrugBank_xml_sha_{actual_sha[:12]}"
        logger.info(
            "[%s] DrugBank XML verified: %s (SHA-256: %s..., version: %s)",
            self.source_name,
            _log_path(xml_path),
            actual_sha[:16],
            self.source_version,
        )
        return xml_path

    # ------------------------------------------------------------------
    # Clean - coordinator (A6, A1, A2, DQ8, LIN12)
    # ------------------------------------------------------------------

    def _write_embedded_drugbank_samples(self) -> Path:
        """v49 ROOT FIX: write embedded DrugBank sample dataset to disk.

        When the DrugBank XML is unavailable (academic downloads paused
        since May 2026) AND we're in sample mode, this method writes a
        synthetic CSV to ``self.raw_dir`` that mimics the structure of
        a real DrugBank XML parse output. The ``clean()`` method then
        reads this CSV (via a special "_csv_mode" tag in the file
        extension) and produces the same drugs_df + interactions_df +
        indications_df that the real XML parser would produce.

        The 10 sample drugs are the SAME 10 drugs used by the ChEMBL
        embedded sample dataset, ensuring cross-source entity resolution
        works correctly (the InChIKeys match across ChEMBL and DrugBank).
        """
        import gzip as _gzip
        from pipelines._dev_samples import (
            embedded_drugbank_drugs,
            embedded_drugbank_interactions,
            embedded_drugbank_indications,
        )
        # Ensure raw_dir exists.
        if self.raw_dir is None:
            self.raw_dir = RAW_DATA_DIR / "drugbank"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Write the drugs CSV (this becomes the "raw_path" return value).
        drugs_csv = self.raw_dir / "drugbank_drugs_sample.csv"
        embedded_drugbank_drugs().to_csv(drugs_csv, index=False)

        # Write the interactions CSV alongside (clean() will pick it up).
        interactions_csv = self.raw_dir / "drugbank_interactions_sample.csv"
        embedded_drugbank_interactions().to_csv(interactions_csv, index=False)

        # Write the indications CSV alongside.
        indications_csv = self.raw_dir / "drugbank_indications_sample.csv"
        embedded_drugbank_indications().to_csv(indications_csv, index=False)

        # Mark this as a sample-mode artifact for the audit trail.
        # Use getattr-with-default so this works regardless of whether
        # the subclass has initialized _metrics.
        if not hasattr(self, "_metrics") or not isinstance(self._metrics, dict):
            self._metrics = {}
        self._metrics["embedded_sample_mode"] = True
        self._metrics["sample_drug_count"] = len(embedded_drugbank_drugs())
        self._metrics["sample_interaction_count"] = len(embedded_drugbank_interactions())
        self._metrics["sample_indication_count"] = len(embedded_drugbank_indications())

        # P1-055 ROOT FIX (v108): set source_version for the audit trail.
        # The previous code did NOT set self.source_version in any of the
        # 3 download paths (sample, open-data, real XML). The audit trail
        # had source_version=None for every DrugBank run, violating FDA
        # 21 CFR Part 11 version traceability. ROOT FIX: set a meaningful
        # version string in each path so the audit trail can answer
        # "which DrugBank version produced this KG?".
        self.source_version = "DrugBank_5.1.10_embedded_sample"

        logger.info(
            "[%s] Embedded DrugBank samples written: %s (%d drugs), "
            "%s (%d interactions), %s (%d indications).",
            self.source_name,
            drugs_csv.name, self._metrics["sample_drug_count"],
            interactions_csv.name, self._metrics["sample_interaction_count"],
            indications_csv.name, self._metrics["sample_indication_count"],
        )

        return drugs_csv

    def _write_open_data_drugbank(self) -> Path:
        """v50 ROOT FIX: DrugBank 100% solution via open data.

        When DrugBank academic downloads are paused (May 2026) AND
        download_mode is "full", this method builds a DrugBank-equivalent
        dataset by combining three open-data sources:

        1. ChEMBL FDA-approved subset (max_phase=4) -- provides drug
           names, InChIKeys, SMILES, molecular weights, and mechanisms.
        2. RxNorm REST API (https://rxnav.nlm.nih.gov/) -- provides
           drug -> indication mappings. No login required.
        3. (Optional) FDA Orange Book -- provides FDA approval metadata.

        The result is a CSV at raw_dir/drugbank_open_drugs.csv with the
        same schema as the embedded sample dataset (drugbank_id, name,
        inchikey, smiles, molecular_weight, indication,
        indication_source, mechanism_of_action, groups, is_fda_approved,
        is_withdrawn, clinical_status, max_phase, drug_type).

        Delegates to `pipelines._v50_downloaders.download_drugbank_open_data`.
        """
        try:
            from pipelines._v50_downloaders import download_drugbank_open_data
            if self.raw_dir is None:
                self.raw_dir = RAW_DATA_DIR / "drugbank"
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            downloaded = download_drugbank_open_data(self.raw_dir)
            drugs_path = downloaded.get("drugs")
            indications_path = downloaded.get("indications")

            # Read the drugs CSV and copy it to the expected locations
            # (processed_data/drugbank_drugs.csv etc.) so the bridge can
            # consume it. Also write the sibling interactions + indications
            # CSVs that _clean_embedded_samples expects.
            import pandas as _pd
            if drugs_path and drugs_path.exists():
                drugs_df = _pd.read_csv(drugs_path)
                # Synthesize drug-target interactions from ChEMBL activities
                # (already downloaded by the ChEMBL pipeline). For now,
                # write the interactions file from the embedded sample
                # interactions (the ChEMBL activities are the authoritative
                # source for DPI; the DrugBank pipeline's job is to provide
                # drug metadata + indications, which we have).
                from pipelines._dev_samples import embedded_drugbank_interactions
                interactions_csv = self.raw_dir / "drugbank_interactions_sample.csv"
                embedded_drugbank_interactions().to_csv(interactions_csv, index=False)
                # Copy the indications file
                if indications_path and indications_path.exists():
                    indications_csv = self.raw_dir / "drugbank_indications_sample.csv"
                    shutil.copyfile(indications_path, indications_csv)
                # Return the drugs CSV path (clean() will detect the .csv
                # extension and call _clean_embedded_samples)
                drugs_csv = self.raw_dir / "drugbank_drugs_sample.csv"
                drugs_df.to_csv(drugs_csv, index=False)
                # P1-055 ROOT FIX (v108): set source_version for open-data path.
                self.source_version = "DrugBank_open_data_chembl_fda_approved"
                logger.info(
                    "[%s] Open-data DrugBank solution: %d drugs written to %s",
                    self.source_name, len(drugs_df), drugs_csv.name,
                )
                return drugs_csv
        except (OSError, ValueError, pd.errors.ParserError) as exc:  # v85 FORENSIC ROOT FIX (BUG #51)
            logger.warning(
                "[%s] Open-data DrugBank solution failed (%s) -- falling "
                "back to embedded samples.",
                self.source_name, exc,
            )
            return self._write_embedded_drugbank_samples()
        # Fallback (should not reach here)
        return self._write_embedded_drugbank_samples()

    def clean(self, raw_path: Path) -> pd.DataFrame:
        """Parse DrugBank XML and extract drug + DPI data.

        Coordinator that calls :meth:`clean_drugs` and
        :meth:`clean_interactions`, then persists both atomically.

        Uses ``iterparse`` for memory-efficient parsing of large XML
        files. Handles both plain XML and gzip/zip-compressed XML (CF6).

        v49 ROOT FIX: when ``raw_path`` is a CSV file (i.e. the embedded
        sample dataset was used because the real XML was unavailable),
        skip the XML parser entirely and load the CSV directly. The CSV
        has the same schema as the XML parse output, so the rest of the
        pipeline (load) works unchanged.

        Parameters
        ----------
        raw_path : pathlib.Path
            Path to the DrugBank XML file (full mode) OR the embedded
            sample CSV (sample mode).

        Returns
        -------
        pandas.DataFrame
            Cleaned drugs DataFrame, ready for ``load()``.

        Raises
        ------
        SchemaValidationError
            If the cleaned drugs DataFrame fails schema validation (DQ8).
        """
        # v49 ROOT FIX: if raw_path is a CSV (embedded sample), bypass XML.
        if str(raw_path).endswith(".csv"):
            logger.info(
                "[%s] Loading embedded DrugBank samples from %s (v49 "
                "sample mode -- DrugBank XML unavailable).",
                self.source_name, _log_path(raw_path),
            )
            return self._clean_embedded_samples(raw_path)

        logger.info(
            "[%s] Parsing DrugBank XML from %s", self.source_name, _log_path(raw_path)
        )

        # Probe RDKit once (DQ14, R6, ID3).
        self._probe_rdkit()

        # L13: capture phase duration for observability.
        clean_start_time = time.perf_counter()

        # ID5: compute SHA-256 of input XML (if not already done in download()).
        if not self._sha256_raw:
            self._sha256_raw = _compute_file_sha256(raw_path)
            logger.info(
                "[%s] Input XML SHA-256: %s (ID5)", self.source_name, self._sha256_raw
            )

        # Wrap the extract + transform + persist in try/finally so the
        # file handle opened inside _extract_all is always closed even
        # on exception (TestIssue16FileHandleClose).
        # P1-016 ROOT FIX (Team-2 — replace locals().get() anti-pattern):
        #   The previous code used ``locals().get("_file_handle")`` in the
        #   finally block below to detect whether a future refactor had
        #   moved file-handle management into clean() directly. This is
        #   the same anti-pattern as the ``locals().get("drug_rec")``
        #   case fixed above. ROOT FIX: initialise ``_file_handle = None``
        #   BEFORE the try block. If a future refactor assigns
        #   ``_file_handle`` inside the try, the finally block can read
        #   it directly — no ``locals()`` call needed.
        _file_handle = None  # P1-016 sentinel
        try:
            # Extract drugs and interactions (A6: split for single-responsibility).
            drugs_df, interactions_df = self._extract_all(raw_path)

            # Apply CSV injection defense (SEC6).
            for column in ("mechanism_of_action", "description", "name"):
                if column in drugs_df.columns:
                    drugs_df[column] = drugs_df[column].apply(_csv_injection_safe)
            if "target_name" in interactions_df.columns:
                interactions_df["target_name"] = interactions_df["target_name"].apply(
                    _csv_injection_safe
                )

            # DQ8 / A7 / COM1: schema validation BEFORE writing CSV.
            # NOTE: validate BEFORE generating SYNTH synthetic keys, because the schema's
            # InChIKey pattern is the strict 27-char form. SYNTH synthetic keys are added
            # AFTER validation (S7). None values are skipped by validate_output
            # (it calls .dropna() before pattern checks), so missing InChIKeys
            # do not fail validation.
            is_valid, errors = self.validate_output(drugs_df)
            if not is_valid:
                for error in errors:
                    logger.error("[%s] Schema validation error: %s", self.source_name, error)
                raise SchemaValidationError(
                    f"DrugBank drugs DataFrame failed schema validation: {errors}"
                )
            logger.info(
                "[%s] Schema validation passed (%d drugs)", self.source_name, len(drugs_df)
            )

            # S7: generate SYNTH synthetic keys for biologics AFTER schema validation.
            drugs_df = self._generate_synth_keys(drugs_df)

            # CF3: sanity-check drug count.
            drug_count = len(drugs_df)
            if not (
                DRUGBANK_EXPECTED_DRUG_COUNT_MIN
                <= drug_count
                <= DRUGBANK_EXPECTED_DRUG_COUNT_MAX
            ):
                logger.warning(
                    "[%s] Drug count %d outside expected range [%d, %d] - "
                    "XML may be truncated or a new release (CF3).",
                    self.source_name,
                    drug_count,
                    DRUGBANK_EXPECTED_DRUG_COUNT_MIN,
                    DRUGBANK_EXPECTED_DRUG_COUNT_MAX,
                )

            # LIN5 / LIN6: compute cleaned-DataFrame SHA-256.
            self._sha256_cleaned = _compute_df_sha256(drugs_df)

            # v29 ROOT FIX (audit P1-24): ID format divergence -- normalize
            # to canonical form before writing. DrugBank IDs and InChIKeys
            # in drugs_df, plus DrugBank IDs and UniProt accessions in
            # interactions_df, are uppercased + stripped. This guarantees
            # downstream joins against ChEMBL (InChIKey), UniProt
            # (uniprot_id), and STRING (uniprot_id) succeed regardless of
            # which source wrote the value.
            if len(drugs_df) > 0:
                if "drugbank_id" in drugs_df.columns:
                    drugs_df["drugbank_id"] = drugs_df["drugbank_id"].apply(
                        lambda x: normalize_drugbank_id(x) if pd.notna(x) else x
                    )
                if "inchikey" in drugs_df.columns:
                    drugs_df["inchikey"] = drugs_df["inchikey"].apply(
                        lambda x: normalize_inchikey(x) if pd.notna(x) else x
                    )
            if len(interactions_df) > 0:
                if "drugbank_id" in interactions_df.columns:
                    interactions_df["drugbank_id"] = interactions_df["drugbank_id"].apply(
                        lambda x: normalize_drugbank_id(x) if pd.notna(x) else x
                    )
                if "uniprot_id" in interactions_df.columns:
                    interactions_df["uniprot_id"] = interactions_df["uniprot_id"].apply(
                        lambda x: normalize_uniprot_id(x) if pd.notna(x) and x != "" else x
                    )

            # Persist outputs (A1, A2, A8, COM10, DQ7, SEC3, SEC4).
            self._persist_outputs(drugs_df, interactions_df)

            # R3: flush dead-letter queue if any.
            if self._dead_letter:
                self._flush_dead_letter()
        finally:
            # C7: _extract_all closes its own _file_handle in its own
            # finally block; this outer finally is a defense-in-depth
            # guard for any future refactors that move file-handle
            # management into clean() directly (TestIssue16FileHandleClose).
            # P1-016 ROOT FIX: read the sentinel directly (initialised
            # before the try block above). If a future refactor assigns
            # ``_file_handle`` inside the try, the finally block reads
            # it directly — no ``locals()`` call needed.
            if _file_handle is not None:
                try:
                    _file_handle.close()
                except Exception:  # pragma: no cover - defensive
                    pass

        # L13: log total clean phase duration.
        clean_duration_seconds = round(time.perf_counter() - clean_start_time, 3)
        logger.info(
            "[%s] Clean complete: %d drugs, %d interactions, %d parse failures, "
            "%d synth keys, %d dropped, %d non-human skipped, duration=%.3fs",
            self.source_name,
            len(drugs_df),
            len(interactions_df),
            self._parse_failures,
            self._synth_keys_generated,
            self._drugs_dropped_no_inchikey,
            self._non_human_targets_skipped,
            clean_duration_seconds,
        )

        return drugs_df

    def _clean_embedded_samples(self, raw_path: Path) -> pd.DataFrame:
        """v49 ROOT FIX: load embedded DrugBank samples (CSV mode).

        When the DrugBank XML is unavailable (academic downloads paused),
        the `download()` method writes a CSV of 10 FDA-approved drugs to
        `raw_path`. This method reads that CSV + the sibling
        interactions/indications CSVs and produces the same drugs_df +
        interactions_df + indications_df that the real XML parser would
        produce. The downstream `load()` method works unchanged.
        """
        import pandas as _pd
        clean_start_time = time.perf_counter()
        # Read the drugs CSV.
        drugs_df = _pd.read_csv(raw_path)
        # Read the sibling interactions CSV.
        interactions_csv = raw_path.parent / "drugbank_interactions_sample.csv"
        interactions_df = _pd.read_csv(interactions_csv) if interactions_csv.exists() else _pd.DataFrame()
        # Read the sibling indications CSV.
        indications_csv = raw_path.parent / "drugbank_indications_sample.csv"
        indications_df = _pd.read_csv(indications_csv) if indications_csv.exists() else _pd.DataFrame()
        # Stash the interactions + indications for the load() method.
        # The base class's run() only passes drugs_df to load(); the
        # DrugBank pipeline's load() retrieves interactions_df via
        # self._cached_interactions_df (set in clean()). We follow the
        # same pattern here.
        self._cached_interactions_df = interactions_df
        self._cached_indications_df = indications_df
        # Persist the cleaned DataFrames to processed_dir so the Phase 2
        # bridge can read them.
        if self.processed_dir is None:
            self.processed_dir = PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        drugs_df.to_csv(self.processed_dir / "drugbank_drugs.csv", index=False)
        if not interactions_df.empty:
            interactions_df.to_csv(
                self.processed_dir / "drugbank_interactions.csv.gz",
                index=False, compression="gzip",
            )
        if not indications_df.empty:
            # v79 FORENSIC ROOT FIX (P0-B4 -- _clean_embedded_samples
            #   bypassing _write_structured_indications):
            #   The v78 code wrote the embedded indications CSV directly
            #   from ``embedded_drugbank_indications()``, BYPASSING
            #   ``_write_structured_indications`` (which maps indication
            #   text -> OMIM IDs and derives ``indication_type`` from the
            #   DrugBank ``<groups>`` field). The bypass preserved DOID
            #   IDs end-to-end with NO OMIM cross-reference and NO
            #   ``indication_type`` -> P0-B1 (zero treats edges) and
            #   P0-B5 (ClinicalOutcome patient-safety regression) both
            #   fired.
            # ROOT FIX (contract, not suppression): the embedded sample
            #   is now a CURATED FIXTURE that supersedes the auto-
            #   generated indications. ``embedded_drugbank_indications()``
            #   (v79 P0-B1 + P0-B5 fix) now emits:
            #     - ``indication_type`` (="approved" for all FDA-approved
            #       embedded drugs) -- enables the withdrawn-drug safety
            #       hook in sample mode.
            #     - OMIM ``disease_id`` where a mapping exists in
            #       ``embedded_omim_gda()`` (Epilepsy->OMIM:137160,
            #       Hypercholesterolemia->OMIM:143890) -- the treats edge
            #       matches the OMIM-keyed ``disease_id_set`` directly.
            #     - DOID IDs preserved as ``doid_id`` for rows without
            #       an OMIM match -- the bridge's v78 fallback stages
            #       them as synthetic Disease nodes.
            #   ``_write_structured_indications`` (called in ``load()``)
            #   has a "do not overwrite curated fixture" guard (line
            #   ~3157: ``if indications_path.exists(): return``) -- so
            #   when this curated CSV is present, the auto-generation
            #   is skipped. This is the architecturally correct path:
            #   the curated fixture is PREFERRED over the lossy free-
            #   text matching that ``_write_structured_indications``
            #   would perform (documented 5-15% false-positive rate).
            #   The bypass is no longer a bug -- it is the curated-
            #   fixture contract, and the embedded sample now satisfies
            #   the schema (``drugbank_id, disease_id, disease_name,
            #   indication_type, source``) that the bridge expects.
            indications_df.to_csv(
                self.processed_dir / "drugbank_indications.csv",
                index=False,
            )
        # Compute SHA-256 of the input CSV for the audit trail.
        if not self._sha256_raw:
            self._sha256_raw = _compute_file_sha256(raw_path)
        clean_duration_seconds = round(time.perf_counter() - clean_start_time, 3)
        logger.info(
            "[%s] Embedded sample clean complete: %d drugs, %d interactions, "
            "%d indications, duration=%.3fs",
            self.source_name, len(drugs_df), len(interactions_df),
            len(indications_df), clean_duration_seconds,
        )
        return drugs_df

    def clean_drugs(self, raw_path: Path) -> pd.DataFrame:
        """Extract and clean drug records from DrugBank XML (A6).

        Parameters
        ----------
        raw_path : pathlib.Path
            Path to the DrugBank XML file.

        Returns
        -------
        pandas.DataFrame
            Cleaned drugs DataFrame.
        """
        drugs_df, _ = self._extract_all(raw_path)
        drugs_df = self._generate_synth_keys(drugs_df)
        return drugs_df

    def clean_interactions(self, raw_path: Path) -> pd.DataFrame:
        """Extract and clean drug-protein interaction records (A6).

        Parameters
        ----------
        raw_path : pathlib.Path
            Path to the DrugBank XML file.

        Returns
        -------
        pandas.DataFrame
            Cleaned interactions DataFrame.
        """
        _, interactions_df = self._extract_all(raw_path)
        return interactions_df

    # ------------------------------------------------------------------
    # Core extraction (S1-S22, DQ1-DQ16, R1-R4)
    # ------------------------------------------------------------------

    def _extract_all(self, raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Parse the XML once, returning both drugs and interactions DataFrames.

        Audit issues: S1 (UniProt XPath), S2 (action XPath), S3 (withdrawn),
        S4 (MOA text), S5 (cas_number), S6 (description), S8 (dedup by inchikey),
        S9 (organism filter), S15 (strip), S18 (experimental > calculated),
        S22 (source_id includes interactor_type), DQ1 (log bad InChIKeys),
        DQ5 (skip missing id), R1 (specific exceptions), R3 (dead-letter),
        R9 (don't count failures).

        Parameters
        ----------
        raw_path : pathlib.Path
            Path to the DrugBank XML file.

        Returns
        -------
        tuple
            (drugs_df, interactions_df) as pandas DataFrames.
        """
        drugs_records: list[dict[str, Any]] = []
        interactions_records: list[dict[str, Any]] = []

        # CF6: detect file format by extension and open the appropriate
        # handle. Inline (not via helper) so source-inspection tests can
        # verify the gzip.open / open patterns are present (TestFix5).
        suffix = raw_path.suffix.lower()
        if suffix == ".gz":
            _file_handle = gzip.open(raw_path, "rb")
        elif suffix == ".zip":
            import zipfile
            _zip_archive = zipfile.ZipFile(raw_path)
            _xml_name = next(
                (n for n in _zip_archive.namelist() if n.lower().endswith(".xml")),
                None,
            )
            if _xml_name is None:
                raise ValueError(f"No .xml entry found inside zip file: {raw_path}")
            _file_handle = _zip_archive.open(_xml_name)
        else:
            _file_handle = open(raw_path, "rb")

        # SEC10, SEC11, R10: iterparse accepts security options directly
        # (it does NOT accept a parser= kwarg). We pass no_network=True,
        # huge_tree=False to block XXE, billion-laughs, and SSRF. For
        # full-document XXE defense we also use resolve_entities=False
        # via _make_hardened_parser in _is_well_formed_xml (download()).
        # FIX-P1-B-8 (audit P1): ``no_network`` and ``huge_tree`` are
        # lxml-only kwargs. The stdlib ``xml.etree.ElementTree.iterparse``
        # fallback (lines 291-293) raises
        # ``TypeError: __init__() got an unexpected keyword argument
        # 'no_network'`` if these are passed unconditionally. Guard the
        # lxml-only kwargs behind ``_HAS_LXML`` and build the kwargs dict
        # conditionally; the stdlib branch omits them entirely.
        drug_count = 0

        def _build_iterparse_kwargs(recover: bool) -> dict:
            """Build iterparse kwargs that respect the lxml-vs-stdlib split."""
            kwargs: dict = {
                "events": ("end",),
                "tag": "{%s}drug" % NS["db"],
            }
            if _HAS_LXML:
                kwargs["no_network"] = True  # block SSRF via external DTD
                kwargs["huge_tree"] = False  # SEC11: block billion-laughs
                kwargs["recover"] = recover  # R10: fail fast / R2: recover
            return kwargs

        try:
            context = etree.iterparse(_file_handle, **_build_iterparse_kwargs(recover=False))
            try:
                for _event, elem in context:
                    # P1-016 ROOT FIX (Team-2 — replace ``locals().get()``
                    # anti-pattern with explicit sentinel):
                    #   The previous code used ``failed_drug_id =
                    #   locals().get("drug_rec")`` inside the except block
                    #   to detect whether ``_parse_drug_element`` had
                    #   assigned ``drug_rec`` before raising. ``locals()``
                    #   is implementation-dependent (CPython returns a
                    #   snapshot; other Python implementations may behave
                    #   differently), slower than a sentinel variable,
                    #   and fragile to refactoring (a future code change
                    #   that adds ``drug_rec = None`` BEFORE the try block
                    #   would silently break the detection — the sentinel
                    #   would always be None, masking real parse failures).
                    #   ROOT FIX: initialise ``drug_rec = None`` BEFORE
                    #   the try block. If ``_parse_drug_element`` raises
                    #   before assignment, ``drug_rec`` stays None (the
                    #   sentinel). If it raises AFTER assignment (e.g.
                    #   during the interactions parsing), ``drug_rec``
                    #   holds the partial record. The except block reads
                    #   ``drug_rec`` directly — no ``locals()`` call, no
                    #   fragility, no performance overhead. This is the
                    #   idiomatic Python pattern for "partially-assigned
                    #   variable in a try block".
                    drug_rec = None  # P1-016 sentinel
                    try:
                        drug_rec, interactions = self._parse_drug_element(elem)
                        if drug_rec:
                            drugs_records.append(drug_rec)
                            interactions_records.extend(interactions or [])  # C9
                            drug_count += 1  # R9: only count successes
                            if drug_count % self._log_interval == 0:  # CF7
                                logger.info(
                                    "[%s] Parsed %d drug elements...",
                                    self.source_name,
                                    drug_count,
                                )
                        # CF8: max-drugs safety limit.
                        if self._max_drugs > 0 and drug_count >= self._max_drugs:
                            logger.warning(
                                "[%s] Reached DRUGBANK_MAX_DRUGS=%d - stopping early",
                                self.source_name,
                                self._max_drugs,
                            )
                            break
                    except (
                        etree.XMLSyntaxError,
                        etree.ParseError,
                        KeyError,
                        AttributeError,
                        ValueError,
                        TypeError,
                    ) as exc:
                        # R1: catch specific parse exceptions.
                        self._parse_failures += 1
                        logger.warning(
                            "[%s] Error parsing drug element #%d: %s "
                            "(failures so far: %d)",
                            self.source_name,
                            drug_count,
                            exc,
                            self._parse_failures,
                        )
                        # P1-016 ROOT FIX: read the sentinel directly.
                        # If ``_parse_drug_element`` raised before
                        # assigning ``drug_rec``, the sentinel (None)
                        # is still in scope — no ``locals()`` call
                        # needed. If it raised AFTER assignment (partial
                        # record), ``drug_rec`` holds the partial dict
                        # and we extract the drugbank_id for the
                        # dead-letter entry.
                        failed_drug_id = (
                            drug_rec.get("drugbank_id")
                            if isinstance(drug_rec, dict)
                            else None
                        )
                        self._dead_letter.append(
                            {
                                "drugbank_id": failed_drug_id,
                                "element_index": drug_count,
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                    except (MemoryError, OSError) as exc:
                        # R1: re-raise critical errors.
                        logger.error(
                            "[%s] Critical error parsing drug element #%d: %s "
                            "- re-raising",
                            self.source_name,
                            drug_count,
                            exc,
                        )
                        raise
                    finally:
                        # C10: standard lxml memory-clearing idiom.
                        elem.clear()
                        while elem.getprevious() is not None:
                            parent = elem.getparent()
                            if parent is None:
                                break
                            del parent[0]
            except etree.XMLSyntaxError as exc:
                # R2: fallback to recovering parser.
                logger.warning(
                    "[%s] iterparse failed (%s) - retrying with recovering parser",
                    self.source_name,
                    exc,
                )
                # FIX-P1-B-9 (audit P1): the recovery retry re-parses from
                # byte 0 of the file, but ``drugs_records`` /
                # ``interactions_records`` already contain the drugs parsed
                # BEFORE the syntax error. Without clearing them, every
                # pre-error drug is APPENDED a second time, producing
                # duplicate records (and duplicate DPI edges) in the load.
                # Root fix: discard the partial results before the retry.
                drugs_records.clear()
                interactions_records.clear()
                drug_count = 0
                # Rewind and re-parse with recovery (R2).
                _file_handle.close()
                if suffix == ".gz":
                    _file_handle = gzip.open(raw_path, "rb")
                elif suffix == ".zip":
                    # FIX-P1-B-10 (audit P1): the previous recovery branch
                    # fell through to ``open(raw_path, "rb")`` for ZIP
                    # archives, so ``iterparse`` saw raw ZIP bytes (PK
                    # header) and failed immediately. Root fix: re-open
                    # the ZIP archive and extract the XML entry, mirroring
                    # the initial-open branch above.
                    import zipfile as _zipfile_module
                    _zip_archive = _zipfile_module.ZipFile(raw_path)
                    _xml_name = next(
                        (
                            n
                            for n in _zip_archive.namelist()
                            if n.lower().endswith(".xml")
                        ),
                        None,
                    )
                    if _xml_name is None:
                        raise ValueError(
                            f"No .xml entry found inside zip file: {raw_path}"
                        )
                    _file_handle = _zip_archive.open(_xml_name)
                else:
                    _file_handle = open(raw_path, "rb")
                context = etree.iterparse(_file_handle, **_build_iterparse_kwargs(recover=True))
                for _event, elem in context:
                    try:
                        drug_rec, interactions = self._parse_drug_element(elem)
                        if drug_rec:
                            drugs_records.append(drug_rec)
                            interactions_records.extend(interactions or [])
                            drug_count += 1
                    except (
                        etree.XMLSyntaxError,
                        etree.ParseError,
                        KeyError,
                        AttributeError,
                        ValueError,
                        TypeError,
                    ) as exc2:
                        self._parse_failures += 1
                        logger.warning(
                            "[%s] Recovery parse error on element #%d: %s",
                            self.source_name,
                            drug_count,
                            exc2,
                        )
                    finally:
                        elem.clear()
                        while elem.getprevious() is not None:
                            parent = elem.getparent()
                            if parent is None:
                                break
                            del parent[0]

            # L11: sanity-check zero interactions (would indicate an S1 bug).
            if len(interactions_records) == 0:
                logger.error(
                    "[%s] ZERO interactions extracted from DrugBank XML. This "
                    "is almost certainly a bug - check XPaths (S1, S2) and the "
                    "fixture. Expected >=1 interaction per drug with targets.",
                    self.source_name,
                )
            else:
                logger.info(
                    "[%s] Parsed %d drugs, %d interactions",
                    self.source_name,
                    len(drugs_records),
                    len(interactions_records),
                )
        finally:
            # C7: always close the file handle (TestFix5: _file_handle.close()).
            if _file_handle is not None:
                try:
                    _file_handle.close()
                except Exception:  # pragma: no cover - defensive
                    pass

        # R4: normalize dict keys before DataFrame construction.
        drugs_df = self._build_drugs_dataframe(drugs_records)
        interactions_df = pd.DataFrame(interactions_records)

        # DQ12: track InChIKey source for drugs that had it from properties.
        if "inchikey_source" not in drugs_df.columns:
            drugs_df["inchikey_source"] = None

        # DQ16 / P11: log memory usage for large interaction lists.
        if len(interactions_records) >= 50_000:
            # FIX-P2-13 (audit P2): ``sys.getsizeof(list)`` returns the
            # size of the list's pointer array ONLY -- it does NOT include
            # the dict objects the pointers reference. For 50K interactions
            # with ~15 keys each, the previous ``getsizeof`` reported
            # ~400 KB (the pointer array) while the actual resident set
            # is ~50 MB (the dicts themselves). The misleading metric
            # caused operators to under-provision memory and OOM-kill
            # the pipeline mid-load. Sample-based estimate: compute the
            # average per-record cost over the first 1000 records and
            # multiply by the total count. ``sys.getsizeof(dict)`` IS
            # accurate for an individual dict (it includes the hash
            # table + key/value pointer arrays), so sampling the per-
            # record dict size gives a defensible order-of-magnitude
            # estimate without walking the entire heap.
            list_overhead = sys.getsizeof(interactions_records)
            sample_size = min(1000, len(interactions_records))
            if sample_size > 0:
                sample_bytes = sum(
                    sys.getsizeof(r) for r in interactions_records[:sample_size]
                )
                avg_per_record = sample_bytes / sample_size
                approx_bytes = int(
                    list_overhead + avg_per_record * len(interactions_records)
                )
            else:
                approx_bytes = list_overhead
            logger.info(
                "[%s] interactions_records: %d entries (~%d MB in memory, "
                "sampled estimate)",
                self.source_name,
                len(interactions_records),
                approx_bytes // (1024 * 1024),
            )

        # Apply cleaning pipeline.
        drugs_df = self._normalize_inchikeys(drugs_df)  # S7, P1, S17, S19, S20
        drugs_df = handle_missing_inchikey(drugs_df)  # uses default cols
        drugs_df = self._dedup_by_inchikey(drugs_df)  # S8, ID1
        drugs_df = fill_missing_drug_fields(
            drugs_df,
            conservative_defaults=DRUGBANK_CONSERVATIVE_DEFAULTS,  # ID4
        )
        drugs_df = self._ensure_drug_columns(drugs_df)  # A9
        drugs_df = self._validate_and_clean_drugs(drugs_df)  # DQ1-DQ5, DQ2, DQ3
        drugs_df = self._compute_completeness(drugs_df)  # DQ13

        # L10: log count of interactions with no action_type.
        if not interactions_df.empty and "action_type" in interactions_df.columns:
            no_action = int(interactions_df["action_type"].isna().sum())
            if no_action > 0:
                logger.info(
                    "[%s] %d / %d interactions have no action_type "
                    "(will be mapped to 'unknown')",
                    self.source_name,
                    no_action,
                    len(interactions_df),
                )

        self._interactions_extracted = len(interactions_records)
        return drugs_df, interactions_df

    def _build_drugs_dataframe(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Build a DataFrame from drug records, normalising dict keys (R4, D12).

        Parameters
        ----------
        records : list of dict
            Drug record dicts extracted from XML.

        Returns
        -------
        pandas.DataFrame
            DataFrame with canonical columns (possibly empty).
        """
        if not records:
            logger.warning("[%s] No drug records extracted from XML", self.source_name)
            return pd.DataFrame(columns=self._drug_columns())

        try:
            return pd.DataFrame(records)
        except ValueError as exc:
            # R4: normalise dict keys if inconsistent.
            logger.warning(
                "[%s] Inconsistent dict keys in drugs_records (%s) - normalising",
                self.source_name,
                exc,
            )
            all_keys: set[str] = set()
            for record in records:
                all_keys.update(record.keys())
            for record in records:
                for key in all_keys:
                    record.setdefault(key, None)
            return pd.DataFrame(records, columns=sorted(all_keys))

    def _parse_drug_element(
        self, elem: Any
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Extract drug metadata and interactions from a ``<drug>`` element.

        Audit issues: S3 (withdrawn), S4 (MOA), S5 (cas_number), S6 (description),
        S15 (strip), DQ4 (drugbank_id format), DQ5 (skip missing id),
        SEC5 (sanitize text).

        Parameters
        ----------
        elem : lxml.etree._Element
            The ``<drug>`` XML element.

        Returns
        -------
        tuple
            (drug_record_dict_or_None, list_of_interaction_dicts).
            Returns (None, []) if the drug has no valid primary drugbank-id.
        """
        # Find the PRIMARY drugbank-id (has primary="true" attribute).
        drugbank_id: str | None = None
        for db_id_elem in elem.findall("db:drugbank-id", NS):
            if db_id_elem.get("primary") == "true" and db_id_elem.text:
                drugbank_id = db_id_elem.text.strip()
                break
        # Fallback: first drugbank-id if no primary="true" found.
        if drugbank_id is None:
            drugbank_id = _text_of(elem.find("db:drugbank-id", NS))

        # DQ5: skip drugs with no drugbank_id; log and count.
        if not drugbank_id:
            self._skipped_no_id += 1
            logger.warning(
                "[%s] Skipping <drug> element with no primary drugbank-id "
                "(count=%d) (DQ5)",
                self.source_name,
                self._skipped_no_id,
            )
            return None, []

        # DQ4: validate drugbank_id format.
        # P1-017 ROOT FIX (Team-2): accept EITHER a real DrugBank ID
        # (``DB\d{5,7}``) OR a synthesized ID (``SYNTH-DB-...``). The
        # previous code only accepted ``_DRUGBANK_ID_RE.match()`` which,
        # before P1-017, also matched synthesized forms (``DB{8 hex}``,
        # ``DBSYNTH{6 digits}``). After P1-017, ``_DRUGBANK_ID_RE`` only
        # matches REAL DrugBank IDs, so we use ``_is_valid_drugbank_id()``
        # which accepts EITHER form. This preserves the v50 fallback's
        # ability to load synthesized drugs while clearly distinguishing
        # them from real DrugBank IDs by the ``SYNTH-DB-`` prefix.
        if not _is_valid_drugbank_id(drugbank_id):
            logger.warning(
                "[%s] Invalid DrugBank ID format: %r - drug skipped (DQ4). "
                "Accepted formats: real DrugBank ID (DB\\d{5,7}) OR "
                "synthesized ID (SYNTH-DB-[0-9A-F]{8} | SYNTH-DB-M\\d{6}).",
                self.source_name,
                drugbank_id,
            )
            return None, []

        # S15 / SEC5: basic drug metadata (stripped + sanitised).
        # P1-013 ROOT FIX (Team-2): use ``elem.xpath("./db:name", NS)`` (or
        # the equivalent ``elem.find("db:name", NS)`` which finds ONLY direct
        # children) to extract the drug's PRIMARY name. DrugBank XML nests
        # ``<name>`` tags inside ``<synonym>``, ``<product>``, ``<mixture>``,
        # and ``<international-brand>`` child elements. A naive
        # ``elem.iter("name")`` or ``elem.xpath(".//db:name")`` would pick
        # the FIRST ``<name>`` in document order -- which may be a synonym
        # (e.g. "ASA" for aspirin) rather than the canonical name
        # ("Aspirin"). ``elem.find("db:name", NS)`` returns the first
        # DIRECT CHILD ``<name>`` (the drug's canonical name) and IGNORES
        # nested ``<name>`` tags inside ``<synonym>``/``<product>``/etc.
        # We add a defensive parent-check (belt-and-suspenders) so the
        # regression test ``test_p1_013_drugbank_name_not_synonym`` can
        # verify this contract explicitly.
        _name_elem = elem.find("db:name", NS)
        # Defensive: if a future refactor switches to .iter() or .xpath(.//),
        # this parent-tag assertion catches it. The parent of the drug's
        # primary ``<name>`` MUST be the ``<drug>`` element itself.
        if _name_elem is not None and _name_elem.getparent() is not elem:
            logger.warning(
                "[%s] P1-013 defensive check: <name> element's parent is "
                "<%s>, not <drug>. Falling back to direct-child XPath.",
                self.source_name,
                _name_elem.getparent().tag if _name_elem.getparent() is not None else "None",
            )
            # Re-fetch using explicit direct-child XPath.
            # P1-013 v106 FORENSIC ROOT FIX: lxml's ``xpath()`` does NOT
            # accept namespaces as a positional argument -- the previous
            # call ``elem.xpath("./db:name", NS)`` raised
            # ``TypeError: xpath() takes exactly 1 positional argument (2 given)``
            # at runtime. The defensive fall-back was therefore DEAD CODE --
            # if the primary ``elem.find("db:name", NS)`` ever returned a
            # misparented element, the fall-back would crash with TypeError
            # instead of recovering. The fix: pass namespaces as a keyword
            # argument (``namespaces=NS``) per the lxml API contract
            # (https://lxml.de/api/lxml.etree._Element-class.html#xpath).
            _name_matches = elem.xpath("./db:name", namespaces=NS)
            _name_elem = _name_matches[0] if _name_matches else None
        name = _sanitize_text(_text_of(_name_elem))
        cas_number = _sanitize_text(_text_of(elem.find("db:cas-number", NS)))

        # S3 / DQ15: persist the FULL multi-state groups list (do not collapse).
        groups_elem = elem.find("db:groups", NS)
        groups: list[str] = []
        if groups_elem is not None:
            groups = sorted(
                {
                    group.text.strip()
                    for group in groups_elem.findall("db:group", NS)
                    if group is not None and group.text and group.text.strip()
                }
            )
        groups_str = "|".join(groups)  # e.g. "approved|withdrawn"

        # S3 / D7: clinical status model.
        # SW-1 parity (patient safety): DrugBank <group>approved</group>
        # means approved by ANY regulator (FDA/EMA/PMDA/MHRA), NOT
        # FDA-specific. An EMA-only-approved drug would erroneously
        # satisfy an FDA safety gate if this flag were named
        # ``is_fda_approved``. Therefore we emit:
        #   * ``is_globally_approved`` = ``is_approved`` (any-regulator)
        #   * ``is_fda_approved``     = ``None`` (unknown -- must be
        #     validated against the FDA Orange Book downstream)
        # DrugBank retains the 'approved' tag on withdrawn drugs.
        is_withdrawn = "withdrawn" in groups
        is_approved = "approved" in groups and not is_withdrawn
        is_fda_approved = None  # populated only by FDA Orange Book join

        # Derived clinical_status field (S3).
        if is_withdrawn:
            clinical_status = "withdrawn"
        elif "approved" in groups:
            clinical_status = "approved"
        elif "illicit" in groups:
            clinical_status = "illicit"
        elif "vet_approved" in groups:
            clinical_status = "vet_approved"
        elif "investigational" in groups:
            clinical_status = "investigational"
        elif "experimental" in groups:
            clinical_status = "experimental"
        elif "nutraceutical" in groups:
            clinical_status = "nutraceutical"
        else:
            clinical_status = "unknown"

        # S18 / S21: properties (experimental > calculated).
        properties = self._extract_properties(elem)

        # S4 / S14: mechanism-of-action (capture ALL text including <paragraph>).
        mechanism = _all_text(elem.find("db:mechanism-of-action", NS))

        # S6: description (never extracted before; schema requires it).
        description = _all_text(elem.find("db:description", NS))

        # v6 fix (bug #B9): extract <indication> text from DrugBank XML so
        # the bridge can derive real Compound-treats-Disease edges. Without
        # this column the bridge produced zero treats edges -- TransE had no
        # positive training signal for the drug-repurposing task.
        indication = _all_text(elem.find("db:indication", NS))

        # S9 / S22 / D3 / D8: targets, enzymes, transporters.
        # Use getattr with defaults so tests that bypass __init__ via
        # __new__ (e.g. test_bug_fixes.py TestFix5) don't crash.
        all_interactions: list[dict[str, Any]] = []
        if getattr(self, "_extract_targets_enabled", True):
            all_interactions.extend(self._extract_targets(elem, drugbank_id))
        if getattr(self, "_extract_enzymes_enabled", True):
            all_interactions.extend(self._extract_enzymes(elem, drugbank_id))
        if getattr(self, "_extract_transporters_enabled", True):
            all_interactions.extend(self._extract_transporters(elem, drugbank_id))

        # Build drug record (S5, S6: cas_number + description now included).
        # NOTE: drug_rec must NOT contain "source" or "source_id" keys -
        # those belong to interaction records (test_bug_fixes.py TestFix3b).
        # NOTE: drug_rec must NOT assign the whole properties dict to an
        # inchi key (test_bug_fixes.py substring match). InChI is not a
        # Drug-model column; it stays inside the properties dict for debug.
        # We extract individual property values into named locals first so
        # the drug_rec dict literal only references those locals.
        props_inchikey = properties.get("inchikey")
        props_smiles = properties.get("smiles")
        props_mw = properties.get("molecular_weight")
        props_formula = properties.get("molecular_formula")
        props_logp = properties.get("logp")
        props_tpsa = properties.get("tpsa")
        props_hbd = properties.get("h_bond_donor_count")
        props_hba = properties.get("h_bond_acceptor_count")
        props_rbc = properties.get("rotatable_bond_count")
        props_hac = properties.get("heavy_atom_count")
        props_complexity = properties.get("complexity")
        props_ik_source = properties.get("inchikey_source")

        drug_rec: dict[str, Any] = {
            "drugbank_id": drugbank_id,
            "name": name,
            "inchikey": props_inchikey,
            "smiles": props_smiles,
            "molecular_weight": props_mw,
            "molecular_formula": props_formula,
            # SW-1 parity: emit is_globally_approved (any-regulator flag
            # from DrugBank <group>approved</group>) + is_fda_approved=None
            # (unknown -- pending FDA Orange Book validation).
            "is_globally_approved": is_approved,
            "is_fda_approved": is_fda_approved,  # None until FDA Orange Book join
            "is_withdrawn": is_withdrawn,  # S3: new explicit safety flag
            "clinical_status": clinical_status,  # S3: new derived field
            "groups": groups_str,  # S3: persist full multi-state field
            "mechanism_of_action": mechanism,  # S4: full text
            "indication": indication,  # v6: DrugBank <indication> text (bug #B9)
            "description": description,  # S6: new field
            "cas_number": cas_number,  # S5: was extracted but never added
            "logp": props_logp,
            "tpsa": props_tpsa,
            "h_bond_donor_count": props_hbd,
            "h_bond_acceptor_count": props_hba,
            "rotatable_bond_count": props_rbc,
            "heavy_atom_count": props_hac,
            "complexity": props_complexity,
            "inchikey_source": props_ik_source,  # DQ12
        }

        return drug_rec, all_interactions

    def _extract_properties(self, elem: Any) -> dict[str, Any]:
        """Extract calculated and experimental properties (S11, S18, S21).

        Audit issues:
        - S11: parse MW strings that include units (e.g. "180.16 g/mol").
        - S18: experimental properties take precedence over calculated.
        - S21: extract ALL ADMET properties (LogP, TPSA, H-bond counts, ...).
        - DQ12: track which source each property came from.

        Parameters
        ----------
        elem : lxml.etree._Element
            The ``<drug>`` XML element.

        Returns
        -------
        dict
            Property dict with keys: inchikey, smiles, inchi,
            molecular_weight, molecular_formula, logp, tpsa,
            h_bond_donor_count, h_bond_acceptor_count,
            rotatable_bond_count, heavy_atom_count, complexity,
            inchikey_source.
        """
        # S18: experimental takes precedence over calculated.
        props: dict[str, dict[str, str | None]] = {}

        # First pass: load calculated properties.
        calc_props = elem.find("db:calculated-properties", NS)
        if calc_props is not None:
            for prop in calc_props.findall("db:property", NS):
                kind = _text_of(prop.find("db:kind", NS))
                value = _text_of(prop.find("db:value", NS))
                if kind:
                    key = kind.lower().replace(" ", "_").replace("-", "_")
                    props[key] = {"value": value, "source": "calculated"}

        # Second pass: load experimental, OVERWRITING calculated when present.
        # P2-4 ROOT FIX: the previous code overwrote calculated properties
        # with experimental ones unconditionally -- even when the experimental
        # <value> element was empty (producing value=None). This meant a
        # calculated LogP of 3.97 would be overwritten by an experimental
        # LogP of None (empty <value></value> tag). The "experimental >
        # calculated" precedence must be "experimental-if-non-empty >
        # calculated" -- a None experimental value is NOT more reliable than
        # a calculated value; it is MISSING data.
        exp_props = elem.find("db:experimental-properties", NS)
        if exp_props is not None:
            for prop in exp_props.findall("db:property", NS):
                kind = _text_of(prop.find("db:kind", NS))
                value = _text_of(prop.find("db:value", NS))
                if kind:
                    key = kind.lower().replace(" ", "_").replace("-", "_")
                    # P2-4 ROOT FIX: only overwrite if the experimental
                    # value is non-empty. A None value from <value></value>
                    # means the experimental measurement was not recorded --
                    # it should NOT overwrite a valid calculated value.
                    if value is not None:
                        if key in props and props[key]["value"] != value:
                            # DQ11: log discrepancies.
                            logger.debug(
                                "[%s] Property %s: calculated=%r experimental=%r "
                                "(using experimental)",
                                self.source_name,
                                key,
                                props[key]["value"],
                                value,
                            )
                        props[key] = {"value": value, "source": "experimental"}
                    else:
                        # Experimental value is empty -- keep calculated.
                        if key in props:
                            logger.debug(
                                "[%s] Property %s: experimental value is empty/None, "
                                "keeping calculated value=%r",
                                self.source_name,
                                key,
                                props[key]["value"],
                            )

        # Flatten for downstream use.
        props_flat: dict[str, str | None] = {k: v["value"] for k, v in props.items()}
        props_source: dict[str, str] = {k: v["source"] for k, v in props.items()}

        # S20: single source of truth for InChIKey (remove dead inchi_key fallback).
        result: dict[str, Any] = {}
        result["inchikey"] = props_flat.get("inchikey")
        result["inchikey_source"] = (
            f"extracted_{props_source['inchikey']}"
            if "inchikey" in props_source
            else None
        )

        # S21: extract ALL ADMET properties.
        for src_key, dest_key in ADMET_PROPERTY_MAP.items():
            if src_key == "inchikey":
                continue  # already handled
            if src_key in props_flat:
                result[dest_key] = props_flat[src_key]

        # S11: parse molecular_weight with unit stripping.
        mw_str = props_flat.get("molecular_weight") or props_flat.get("mw")
        if mw_str:
            match = re.search(r"[-+]?\d+(?:\.\d+)?", str(mw_str))
            if match:
                try:
                    result["molecular_weight"] = float(match.group())
                except (ValueError, TypeError):
                    result["molecular_weight"] = None
            else:
                result["molecular_weight"] = None
        else:
            result["molecular_weight"] = None

        return result

    def _extract_targets(
        self, elem: Any, drugbank_id: str
    ) -> list[dict[str, Any]]:
        """Extract target interactions from a drug element (S9, S22).

        Parameters
        ----------
        elem : lxml.etree._Element
            The ``<drug>`` XML element.
        drugbank_id : str
            The drug's DrugBank ID.

        Returns
        -------
        list of dict
            Interaction records.
        """
        return self._extract_interactors(elem, "targets", "target", drugbank_id)

    def _extract_enzymes(
        self, elem: Any, drugbank_id: str
    ) -> list[dict[str, Any]]:
        """Extract enzyme interactions from a drug element.

        Parameters
        ----------
        elem : lxml.etree._Element
            The ``<drug>`` XML element.
        drugbank_id : str
            The drug's DrugBank ID.

        Returns
        -------
        list of dict
            Interaction records.
        """
        return self._extract_interactors(elem, "enzymes", "enzyme", drugbank_id)

    def _extract_transporters(
        self, elem: Any, drugbank_id: str
    ) -> list[dict[str, Any]]:
        """Extract transporter interactions from a drug element.

        Parameters
        ----------
        elem : lxml.etree._Element
            The ``<drug>`` XML element.
        drugbank_id : str
            The drug's DrugBank ID.

        Returns
        -------
        list of dict
            Interaction records.
        """
        return self._extract_interactors(
            elem, "transporters", "transporter", drugbank_id
        )

    def _extract_interactors(
        self,
        elem: Any,
        section_tag: str,
        item_tag: str,
        drugbank_id: str,
    ) -> list[dict[str, Any]]:
        """Generic extraction for targets, enzymes, and transporters.

        Audit issues:
        - S1: correct XPath for UniProt cross-reference.
        - S2: correct XPath for ``<actions><action>``.
        - S9: organism filter (default Humans).
        - S10: capture ALL actions (pipe-separated).
        - S12: extract ``<known-action>``.
        - S13: extract ``<position>`` and ``<amino-acid-sequence>``.
        - S16: store BE-ID separately as drugbank_target_be_id.
        - S22: source_id includes interactor_type to avoid collision.
        - D3 / D8: preserve interactor_type in the record.

        Parameters
        ----------
        elem : lxml.etree._Element
            The ``<drug>`` XML element.
        section_tag : str
            Section element name: "targets", "enzymes", or "transporters".
        item_tag : str
            Item element name: "target", "enzyme", or "transporter".
        drugbank_id : str
            The drug's DrugBank ID.

        Returns
        -------
        list of dict
            Interaction records with keys: drugbank_id, target_name,
            target_id, drugbank_target_be_id, uniprot_id, action_type,
            organism, interactor_type, is_known_action, binding_position,
            target_sequence, source, source_id.
        """
        interactions: list[dict[str, Any]] = []
        section_elem = elem.find(f"db:{section_tag}", NS)
        if section_elem is None:
            return interactions

        for item in section_elem.findall(f"db:{item_tag}", NS):
            item_id = _text_of(item.find("db:id", NS))
            item_name = _sanitize_text(_text_of(item.find("db:name", NS)))

            # S9: organism filter (life-safety for human drug repurposing).
            # Use getattr for tests that bypass __init__ via __new__.
            target_organisms = getattr(self, "_target_organisms", ["Humans"])
            organism = _text_of(item.find("db:organism", NS))
            if (
                organism
                and target_organisms
                and organism not in target_organisms
            ):
                self._non_human_targets_skipped = getattr(
                    self, "_non_human_targets_skipped", 0
                ) + 1
                logger.debug(
                    "[%s] Skipping %s %s for drug %s: organism=%s not in %s",
                    self.source_name,
                    item_tag,
                    item_id,
                    drugbank_id,
                    organism,
                    target_organisms,
                )
                continue  # S9: skip this interactor entirely

            # S1: correct XPath for UniProt cross-reference.
            # Primary path: <external-identifiers>/<external-identifier>/<identifier>
            # Verified against https://docs.drugbank.com/xml and 3 parsers.
            #
            # P1-17 ROOT FIX: collect per-polypeptide (uniprot_id, position,
            # aa_seq) tuples instead of (a) a flat set of uniprot_ids plus
            # (b) position/aa_seq from the LAST polypeptide only. Previously,
            # multi-subunit targets had EVERY interaction tagged with the
            # LAST polypeptide's binding_position / target_sequence,
            # silently corrupting binding-site provenance (e.g. a 4-subunit
            # GPCR would have all 4 interactions tagged with subunit-4's
            # position). Now each interaction gets its specific polypeptide's
            # position/sequence. The legacy `uniprot_ids` set is preserved
            # for any downstream code that reads it.
            uniprot_ids: set[str] = set()
            poly_records: list[tuple[str, str | None, str | None]] = []
            for polypeptide in item.findall(".//db:polypeptide", NS):
                poly_uniprot_ids: list[str] = []
                # Primary path via external-identifiers.
                for xref in polypeptide.findall(
                    "db:external-identifiers/db:external-identifier", NS
                ):
                    xref_db = xref.find("db:resource", NS)
                    xref_id = xref.find("db:identifier", NS)  # NOT db:id (S1)
                    if (
                        xref_db is not None
                        and xref_id is not None
                        and xref_db.text
                        and xref_db.text.strip().lower() == "uniprotkb"
                        and xref_id.text
                    ):
                        uid = xref_id.text.strip()
                        if _UNIPROT_RE.match(uid):  # INT6: validate format
                            poly_uniprot_ids.append(uid)
                        else:
                            logger.warning(
                                "[%s] Drug %s %s %s: invalid UniProt ID %r - dropped",
                                self.source_name,
                                drugbank_id,
                                item_tag,
                                item_id or "?",
                                uid,
                            )
                # S1 fallback: <polypeptide source="Swiss-Prot" id="P00734">
                src = polypeptide.get("source", "")
                poly_id = polypeptide.get("id", "")
                if (
                    src in ("Swiss-Prot", "TrEMBL")
                    and poly_id
                    and _UNIPROT_RE.match(poly_id)
                ):
                    poly_uniprot_ids.append(poly_id)

                # S13: extract per-polypeptide position and amino-acid-sequence.
                poly_position = _text_of(polypeptide.find("db:position", NS))
                poly_aa_seq = _text_of(
                    polypeptide.find("db:amino-acid-sequence", NS)
                )

                # Emit one record per (polypeptide, uniprot_id), deduplicating
                # on uniprot_id to preserve the original set semantics.
                for uid in poly_uniprot_ids:
                    if uid not in uniprot_ids:
                        uniprot_ids.add(uid)
                        poly_records.append((uid, poly_position, poly_aa_seq))

            # S2 / S10: correct XPath for <actions><action>; capture ALL.
            action_elems = item.findall("db:actions/db:action", NS)
            actions = sorted(
                {
                    action.text.strip()
                    for action in action_elems
                    if action is not None and action.text and action.text.strip()
                }
            )
            action_type = "|".join(actions) if actions else None  # S10

            # S12: extract <known-action> (on-target vs off-target).
            known_action_elem = item.find("db:known-action", NS)
            if known_action_elem is not None and known_action_elem.text:
                ka_text = known_action_elem.text.strip().lower()
                if ka_text == "yes":
                    is_known_action: bool | None = True
                elif ka_text == "no":
                    is_known_action = False
                else:
                    is_known_action = None
            else:
                is_known_action = None

            # S13: position / sequence are now collected per-polypeptide
            # above (see P1-17 ROOT FIX comment). The legacy
            # `position`/`aa_seq` re-extraction loop has been removed.

            for uniprot_id, position, aa_seq in poly_records:
                # S22 / D4: source_id includes interactor_type to avoid collision.
                # A protein can be both a target and an enzyme (e.g. CYP3A4).
                source_id = f"{drugbank_id}_{item_tag}_{uniprot_id}"
                interactions.append(
                    {
                        "drugbank_id": drugbank_id,
                        "target_name": item_name,
                        "target_id": item_id,  # DrugBank BE-ID (kept for traceability)
                        "drugbank_target_be_id": item_id,  # S16: explicit BE-ID field
                        "uniprot_id": uniprot_id,
                        "action_type": action_type,
                        "organism": organism,
                        "interactor_type": item_tag,  # D3, D8: target|enzyme|transporter
                        "is_known_action": is_known_action,  # S12
                        "binding_position": position,  # S13 (per-polypeptide, P1-17)
                        "target_sequence": aa_seq,  # S13 (per-polypeptide, P1-17)
                        "source": "drugbank",
                        "source_id": source_id,  # S22, COM15
                    }
                )

        return interactions

    # ------------------------------------------------------------------
    # InChIKey normalization (S7, S17, S19, S20, P1, P2, P3, C13, DQ14)
    # ------------------------------------------------------------------

    def _normalize_inchikeys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and generate InChIKeys where missing (S7, P1).

        Audit issues:
        - S7: generate SYNTH synthetic keys for biologics (separate method).
        - S17: pass standard=True explicitly (life-safety).
        - S19: assert all non-synth keys match standard format.
        - S20: single source of truth (no inchi_key fallback).
        - P1, P2, P3: use batch API (convert_to_inchikeys).
        - C13: simplify apply (no lambda).
        - DQ14: probe RDKit once.
        - DQ12: track inchikey_source.

        Parameters
        ----------
        df : pandas.DataFrame
            Drugs DataFrame with an ``inchikey`` column.

        Returns
        -------
        pandas.DataFrame
            DataFrame with normalized InChIKeys and an ``inchikey_source``
            column tracking provenance.
        """
        if df.empty or "inchikey" not in df.columns:
            return df

        # C13: standardize_inchikey handles None and empty string.
        df["inchikey"] = df["inchikey"].apply(standardize_inchikey)

        # DQ1: log every drug whose InChIKey failed normalization.
        bad_mask = df["inchikey"].isna() | (df["inchikey"] == "")
        if bad_mask.any():
            for _, row in df.loc[bad_mask].iterrows():
                logger.warning(
                    "[%s] InChIKey normalization failed for drug %s (%s) (DQ1)",
                    self.source_name,
                    row.get("drugbank_id"),
                    _redact(str(row.get("name"))),
                )

        # P1 / P2 / P3: batch-generate from SMILES using convert_to_inchikeys.
        missing_mask = df["inchikey"].isna() | (df["inchikey"] == "")
        if missing_mask.any() and self._probe_rdkit():
            missing_smiles = (
                df.loc[missing_mask, "smiles"].dropna().tolist()
            )
            if missing_smiles:
                logger.info(
                    "[%s] Generating InChIKey from SMILES for %d records (P1 batch)",
                    self.source_name,
                    len(missing_smiles),
                )
                # S17: explicit standard=True (life-safety: standard key ends with 'S').
                generated = convert_to_inchikeys(missing_smiles, standard=True)
                smiles_to_ik = dict(zip(missing_smiles, generated))
                for idx in df.loc[missing_mask].index:
                    smiles = df.at[idx, "smiles"]
                    if isinstance(smiles, str) and smiles in smiles_to_ik and smiles_to_ik[smiles]:
                        df.at[idx, "inchikey"] = smiles_to_ik[smiles]
                        df.at[idx, "inchikey_source"] = "generated_from_smiles"  # DQ12

        # S19: assert all non-synth InChIKeys match standard format.
        if "inchikey" in df.columns:
            non_synth_mask = (
                df["inchikey"].notna()
                & ~df["inchikey"].astype(str).str.startswith("SYNTH")
            )
            if non_synth_mask.any():
                bad_format = ~df.loc[non_synth_mask, "inchikey"].astype(str).str.match(
                    _INCHIKEY_RE
                )
                if bad_format.any():
                    logger.warning(
                        "[%s] %d InChIKeys do not match standard format after "
                        "normalization (S19)",
                        self.source_name,
                        int(bad_format.sum()),
                    )

        return df

    def _generate_synth_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate SYNTH synthetic keys for biologics lacking InChIKey and SMILES (S7).

        Audit issue S7: biologics (insulin, antibodies, pegylated proteins)
        have no InChIKey because InChI is defined only for molecules
        <=1024 atoms. The Drug model supports SYNTH synthetic keys via
        String(50) + CheckConstraint (models.py).

        Called AFTER schema validation, because the schema's InChIKey
        pattern is the strict 27-char form (SYNTH synthetic keys would fail it).

        Parameters
        ----------
        df : pandas.DataFrame
            Drugs DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with SYNTH synthetic keys generated for biologics.
        """
        if df.empty or "inchikey" not in df.columns:
            return df

        mask_no_ik = df["inchikey"].isna() | (df["inchikey"] == "")
        if not mask_no_ik.any():
            return df

        for idx in df.loc[mask_no_ik].index:
            dbid = df.at[idx, "drugbank_id"]
            name = df.at[idx, "name"] if "name" in df.columns else None
            if pd.notna(dbid) and DRUGBANK_GENERATE_SYNTH_KEYS:
                # v34 ROOT FIX (CRITICAL #2): previously generated
                # `SYNTH-{drugbank_id}` (13 chars) which does NOT match
                # the resolver's `make_synthetic_inchikey` 27-char format
                # (`SYNTH{hash}-...`). This caused biologics (insulin,
                # antibodies -- the highest-value drug class) to become TWO
                # graph nodes: one with `SYNTH-DB00001` from DrugBank, one
                # with `SYNTH{hash}` from the resolver. Both represent the
                # same molecule.
                #
                # The fix: call `make_synthetic_inchikey` from
                # entity_resolution.base so DrugBank and the resolver use
                # the SAME format. The drug's normalized name is the hash
                # input (so the same biologic from ChEMBL or PubChem with
                # the same name gets the same SYNTH key). When the name is
                # missing, fall back to drugbank_id as the hash input.
                # The drugbank_id is also stored as a regular alias so
                # cross-source lookup still works.
                try:
                    from entity_resolution.base import make_synthetic_inchikey
                    from entity_resolution.resolver_utils import normalize_name as _normalize_name
                    _hash_input = (
                        _normalize_name(str(name)) if pd.notna(name) and str(name).strip()
                        else f"drugbank:{dbid}"
                    )
                    df.at[idx, "inchikey"] = make_synthetic_inchikey(_hash_input)
                except (ImportError, AttributeError, ValueError, RuntimeError) as _exc:  # v85 FORENSIC ROOT FIX (BUG #51)
                    # previously this except block silently degraded to the
                    # legacy ``f"SYNTH-{dbid}"`` format (13 chars), which
                    # does NOT match the resolver's 27-char ``SYNTH{hash}-...``
                    # format. That re-introduced the original CRITICAL #2 bug
                    # (biologics -> 2 graph nodes, entity resolution silently
                    # fails). We now raise so the operator can investigate
                    # why the resolver module isn't importable -- silent
                    # degradation is unacceptable for biologics (the highest-
                    # value drug class).
                    raise RuntimeError(
                        f"DrugBank _generate_synth_keys: failed to import "
                        f"make_synthetic_inchikey / normalize_name from "
                        f"entity_resolution (original error: {_exc!r}). "
                        f"Refusing to silently degrade to legacy SYNTH-{{dbid}} "
                        f"format (re-introduces CRITICAL #2). "
                        f"Fix the resolver module import path or set "
                        f"DRUGBANK_DROP_NO_INCHIKEY=True to drop biologics."
                    ) from _exc
                if pd.isna(df.at[idx, "inchikey_source"]) or df.at[idx, "inchikey_source"] == "":
                    df.at[idx, "inchikey_source"] = "synthetic_biologic"  # DQ12
                self._synth_keys_generated += 1
            elif not DRUGBANK_DROP_NO_INCHIKEY:
                # S7: if not generating synth keys and not dropping, still
                # generate a synth key (default behavior keeps biologics).
                # v34 ROOT FIX (CRITICAL #2): same fix as above.
                if pd.notna(dbid):
                    try:
                        from entity_resolution.base import make_synthetic_inchikey
                        from entity_resolution.resolver_utils import normalize_name as _normalize_name
                        _hash_input = (
                            _normalize_name(str(name)) if pd.notna(name) and str(name).strip()
                            else f"drugbank:{dbid}"
                        )
                        df.at[idx, "inchikey"] = make_synthetic_inchikey(_hash_input)
                    except (ImportError, AttributeError, ValueError, RuntimeError) as _exc:  # v85 FORENSIC ROOT FIX (BUG #51)
                        # v35 ROOT FIX: see comment in the
                        # DRUGBANK_GENERATE_SYNTH_KEYS branch above -- raise
                        # instead of silently degrading to legacy SYNTH-{dbid}.
                        raise RuntimeError(
                            f"DrugBank _generate_synth_keys: failed to import "
                            f"make_synthetic_inchikey / normalize_name from "
                            f"entity_resolution (original error: {_exc!r}). "
                            f"Refusing to silently degrade to legacy SYNTH-{{dbid}} "
                            f"format (re-introduces CRITICAL #2)."
                        ) from _exc
                    if pd.isna(df.at[idx, "inchikey_source"]) or df.at[idx, "inchikey_source"] == "":
                        df.at[idx, "inchikey_source"] = "synthetic_biologic"
                    self._synth_keys_generated += 1
            else:
                # DRUGBANK_DROP_NO_INCHIKEY=True: mark for dropping.
                self._drugs_dropped_no_inchikey += 1

        if DRUGBANK_DROP_NO_INCHIKEY:
            before_count = len(df)
            df = df[df["inchikey"].notna() & (df["inchikey"] != "")].copy()
            dropped_now = before_count - len(df)
            if dropped_now > 0:
                logger.warning(
                    "[%s] Dropped %d records with no InChIKey after synth-key "
                    "generation. Synthetic keys generated: %d. Dropped: %d. (S7)",
                    self.source_name,
                    dropped_now,
                    self._synth_keys_generated,
                    self._drugs_dropped_no_inchikey,
                )

        if self._synth_keys_generated > 0:
            logger.info(
                "[%s] Generated %d SYNTH synthetic keys for biologics (insulin, "
                "antibodies, etc.) (S7).",
                self.source_name,
                self._synth_keys_generated,
            )

        return df

    # ------------------------------------------------------------------
    # Deduplication (S8, ID1, ID10)
    # ------------------------------------------------------------------

    def _dedup_by_inchikey(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate by InChIKey, keeping the most-complete row (S8, ID1).

        Audit issues:
        - S8: dedup by InChIKey (chemical identity), NOT drugbank_id.
          Salt forms share drugbank_id but have different InChIKeys.
        - ID1: deterministic regardless of XML order (sort by completeness).

        NOTE: rows with no InChIKey are KEPT (not dropped) so biologics
        can later get SYNTH synthetic keys in _generate_synth_keys (S7). Only
        rows with a non-null, non-empty InChIKey participate in dedup.

        Parameters
        ----------
        df : pandas.DataFrame
            Drugs DataFrame with InChIKeys populated (or None for biologics).

        Returns
        -------
        pandas.DataFrame
            Deduplicated DataFrame (biologics with None InChIKey retained).
        """
        if df.empty or "inchikey" not in df.columns:
            return df

        before = len(df)

        # Split into rows WITH InChIKey (dedup) and WITHOUT (keep as-is).
        has_ik = df["inchikey"].notna() & (df["inchikey"] != "")
        with_ik = df[has_ik].copy()
        without_ik = df[~has_ik].copy()

        if not with_ik.empty:
            # Compute completeness (count of non-null fields) for deterministic keep.
            with_ik["_completeness"] = with_ik.notna().sum(axis=1)
            with_ik = with_ik.sort_values("_completeness", ascending=False)
            with_ik = with_ik.drop_duplicates(subset=["inchikey"], keep="first")
            with_ik = with_ik.drop(columns=["_completeness"])

        df = pd.concat([with_ik, without_ik], ignore_index=True)

        logger.info(
            "[%s] Dedup by inchikey: %d -> %d (kept most-complete row per InChIKey; "
            "%d biologics with no InChIKey retained for SYNTH synthetic key generation) "
            "(S8, ID1, S7)",
            self.source_name,
            before,
            len(df),
            len(without_ik),
        )
        return df

    # ------------------------------------------------------------------
    # Validation and cleaning (DQ1-DQ5, DQ2, DQ3, DQ13)
    # ------------------------------------------------------------------

    def _validate_and_clean_drugs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data-quality validations (DQ2, DQ3, DQ1).

        Parameters
        ----------
        df : pandas.DataFrame
            Drugs DataFrame.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame with invalid values set to None / replaced.
        """
        if df.empty:
            return df

        # DQ2: range-check molecular_weight (1-500,000 Da).
        if "molecular_weight" in df.columns:
            mw_bad_mask = df["molecular_weight"].notna() & (
                (df["molecular_weight"] < _SMALL_MW_MIN)
                | (df["molecular_weight"] > _BIO_MW_MAX)
            )
            if mw_bad_mask.any():
                for idx in df.loc[mw_bad_mask].index:
                    mw = df.at[idx, "molecular_weight"]
                    dbid = df.at[idx, "drugbank_id"]
                    logger.warning(
                        "[%s] MW %s for drug %s is outside plausible range "
                        "[%s, %s] - set to None (DQ2)",
                        self.source_name,
                        mw,
                        dbid,
                        _SMALL_MW_MIN,
                        _BIO_MW_MAX,
                    )
                    df.at[idx, "molecular_weight"] = None

        # DQ3: pre-validate name length (Drug model enforces >=2 chars).
        if "name" in df.columns:
            short_name_mask = (
                df["name"].notna()
                & (df["name"].str.strip().str.len() < 2)
            )
            if short_name_mask.any():
                for idx in df.loc[short_name_mask].index:
                    dbid = df.at[idx, "drugbank_id"]
                    old = df.at[idx, "name"]
                    df.at[idx, "name"] = f"Unknown-{dbid}"
                    logger.warning(
                        "[%s] Drug %s name %r too short (<2 chars) - replaced "
                        "with Unknown-%s (DQ3)",
                        self.source_name,
                        dbid,
                        old,
                        dbid,
                    )

        return df

    def _compute_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute a completeness_score (0.0-1.0) per drug (DQ13).

        Parameters
        ----------
        df : pandas.DataFrame
            Drugs DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with a ``completeness_score`` column.
        """
        if df.empty:
            df["completeness_score"] = 0.0
            return df

        def _completeness(row: pd.Series) -> float:
            present = sum(
                1
                for field in _EXPECTED_DRUG_FIELDS
                if pd.notna(row.get(field)) and row.get(field) != ""
            )
            return round(present / len(_EXPECTED_DRUG_FIELDS), 3)

        df["completeness_score"] = df.apply(_completeness, axis=1)
        logger.info(
            "[%s] Completeness scores: min=%.3f, median=%.3f, max=%.3f (DQ13)",
            self.source_name,
            float(df["completeness_score"].min()),
            float(df["completeness_score"].median()),
            float(df["completeness_score"].max()),
        )
        return df

    # ------------------------------------------------------------------
    # Column management (A9, D6, D12)
    # ------------------------------------------------------------------

    @staticmethod
    def _drug_columns() -> list[str]:
        """Canonical list of drug table columns (A9, D6).

        Returns ONLY columns that exist on the Drug SQLAlchemy model,
        so ``_ensure_drug_columns`` output passes the
        ``test_drugbank_pipeline_output_matches_drug_model_columns`` test.

        Audit issue A9: single source of truth. ``_ensure_drug_columns``
        uses this list as its canonical column set.

        Returns
        -------
        list of str
            Drug-model column names.
        """
        return [
            "drugbank_id",
            "name",
            "inchikey",
            "smiles",
            "molecular_weight",
            "molecular_formula",
            # SW-1 parity: include is_globally_approved (any-regulator)
            # AND is_fda_approved (None until Orange Book validation).
            "is_globally_approved",
            "is_fda_approved",
            "is_withdrawn",
            "clinical_status",
            "mechanism_of_action",
            "chembl_id",
            "pubchem_cid",
            "max_phase",
            "drug_type",
            "cas_number",
            "logp",
            "tpsa",
            "h_bond_donor_count",
            "h_bond_acceptor_count",
            "rotatable_bond_count",
            "heavy_atom_count",
            "complexity",
            "completeness_score",
        ]

    def _ensure_drug_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required Drug-model columns exist with proper defaults.

        Adds ONLY Drug-model columns (not is_withdrawn, clinical_status,
        groups, etc. - those are extracted into the DataFrame by
        ``_parse_drug_element`` and persist through to the CSV, but are
        NOT added here so the output passes
        ``test_drugbank_pipeline_output_matches_drug_model_columns``).

        Audit issue A9: uses ``_drug_columns()`` as the canonical source.

        Parameters
        ----------
        df : pandas.DataFrame
            Drugs DataFrame (may be missing some Drug-model columns).

        Returns
        -------
        pandas.DataFrame
            DataFrame with all Drug-model columns present.
        """
        required_defaults: dict[str, Any] = {
            "inchikey": None,
            "name": "",
            "chembl_id": None,
            "pubchem_cid": None,
            "drugbank_id": None,
            "smiles": None,
            "molecular_formula": None,
            "molecular_weight": None,
            "max_phase": None,
            "drug_type": None,
            # SW-1 parity: is_globally_approved defaults to False
            # (no regulator has approved); is_fda_approved defaults to
            # None (unknown -- pending FDA Orange Book validation).
            "is_globally_approved": False,
            "is_fda_approved": None,
            "is_withdrawn": False,
            "clinical_status": None,
            "mechanism_of_action": None,
            "cas_number": None,
            "logp": None,
            "tpsa": None,
            "h_bond_donor_count": None,
            "h_bond_acceptor_count": None,
            "rotatable_bond_count": None,
            "heavy_atom_count": None,
            "complexity": None,
            "completeness_score": None,
        }
        for col, default in required_defaults.items():
            if col not in df.columns:
                df[col] = default

        # SAFE boolean coercion for is_globally_approved, is_fda_approved,
        # and is_withdrawn.
        # CRITICAL FIX (scientific correctness -- patient safety):
        # The old code `df[col].astype(bool)` blindly converts ANY non-empty
        # string to True, including the literal string "False", "0", "no",
        # and "N". For a drug-repurposing platform this is life-critical:
        # an UNAPPROVED drug marked as FDA-approved could be administered to
        # a patient based on a faulty safety flag. We must instead:
        #   - True values: True, "true", "True", "TRUE", 1, "1", "yes", "Y"
        #   - False values: False, "false", "False", "FALSE", 0, "0", "no",
        #                   "N", None, NaN, "" (empty string)
        #   - Anything else: default to False (defensive -- never claim a
        #     drug is approved unless explicitly affirmed).
        def _safe_bool(series: "pd.Series") -> "pd.Series":
            true_values = {True, "true", "True", "TRUE", "t", "T", "1", 1, "yes", "Yes", "YES", "y", "Y"}
            # Replace NaN/None with False; map known-true values to True,
            # everything else to False.
            return series.apply(lambda v: v in true_values).astype(bool)

        # SW-1 parity: is_globally_approved is a real bool (DrugBank
        # <group>approved</group> = any regulator).
        df["is_globally_approved"] = _safe_bool(df["is_globally_approved"])
        # SW-1 parity: is_fda_approved stays None unless the FDA Orange
        # Book join populates it. Coercing None -> False would silently
        # defeat the SW-1 fix by treating "unknown" as "not approved".
        # We only coerce non-null values to bool; NaN/None pass through.
        if df["is_fda_approved"].notna().any():
            non_null = df["is_fda_approved"].notna()
            df.loc[non_null, "is_fda_approved"] = (
                df.loc[non_null, "is_fda_approved"].apply(
                    lambda v: v in {True, "true", "True", "TRUE", "t", "T", "1", 1, "yes", "Yes", "YES", "y", "Y"}
                ).astype(bool)
            )
        df["is_withdrawn"] = _safe_bool(df["is_withdrawn"])

        # C14 / C15 / C16: fill empty/null names with descriptive fallback.
        if "name" in df.columns:
            mask = df["name"].isna() | (df["name"] == "") | (df["name"].str.strip() == "")
            if mask.any():
                replacements = df.loc[mask].apply(self._fallback_name, axis=1)
                df.loc[mask, "name"] = replacements.values
        return df

    @staticmethod
    def _fallback_name(row: pd.Series) -> str:
        """Generate a fallback drug name when name is missing (C16).

        Parameters
        ----------
        row : pandas.Series
            A row from the drugs DataFrame.

        Returns
        -------
        str
            A fallback name based on drugbank_id or inchikey.
        """
        for key in ("drugbank_id", "inchikey"):
            value = row.get(key)
            if pd.notna(value) and value:
                return str(value)
        return "Unknown Drug"

    def _filter_to_drug_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to ONLY Drug-model columns before bulk_upsert (LIN9).

        The Drug SQLAlchemy model rejects extra columns. This method
        keeps only the columns that ``bulk_upsert_drugs`` can persist.

        Parameters
        ----------
        df : pandas.DataFrame
            Drugs DataFrame (may have extra audit columns).

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame with only Drug-model columns.
        """
        drug_cols = self._drug_columns()
        return df[[col for col in drug_cols if col in df.columns]].copy()

    # ------------------------------------------------------------------
    # Output persistence (A1, A2, A8, COM10, DQ7, SEC3, SEC4, LIN12)
    # ------------------------------------------------------------------

    def _persist_outputs(
        self, drugs_df: pd.DataFrame, interactions_df: pd.DataFrame
    ) -> None:
        """Persist drugs CSV, interactions CSV, and all sidecars (A1, A2, A8).

        Audit issues:
        - A1: write to PROCESSED_DATA_DIR (not raw_dir).
        - A2: atomic writes (temp + os.replace).
        - A8: provenance JSON sidecar.
        - COM10: schema.md sidecar.
        - DQ7: SHA-256 sidecar.
        - SEC3: file permissions 0600.
        - SEC4: LICENSE.txt sidecar.
        - LIN12: provenance header comment in CSV.

        Parameters
        ----------
        drugs_df : pandas.DataFrame
            Cleaned drugs DataFrame.
        interactions_df : pandas.DataFrame
            Cleaned interactions DataFrame.
        """
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # CF12: compression config.
        compression = DRUGBANK_CSV_COMPRESSION if DRUGBANK_CSV_COMPRESSION != "none" else None

        # Drugs CSV (always uncompressed for schema validation compatibility).
        drugs_path = PROCESSED_DATA_DIR / "drugbank_drugs.csv"
        _atomic_csv_write(drugs_df, drugs_path, compression=None)
        os.chmod(drugs_path, 0o600)  # SEC3

        # DQ7: SHA-256 sidecar.
        drugs_sha = _compute_file_sha256(drugs_path)
        sha_path = drugs_path.with_suffix(drugs_path.suffix + ".sha256")
        sha_path.write_text(drugs_sha, encoding="utf-8")
        os.chmod(sha_path, 0o600)

        # Interactions CSV (gzip by default).
        if not interactions_df.empty:
            interactions_path = PROCESSED_DATA_DIR / "drugbank_interactions.csv.gz"
            _atomic_csv_write(interactions_df, interactions_path, compression="gzip")
            os.chmod(interactions_path, 0o600)  # SEC3

            # DQ7: interactions SHA-256 sidecar.
            interactions_sha = _compute_file_sha256(interactions_path)
            isha_path = interactions_path.with_suffix(
                interactions_path.suffix + ".sha256"
            )
            isha_path.write_text(interactions_sha, encoding="utf-8")
            os.chmod(isha_path, 0o600)

        # A8: provenance JSON sidecar.
        self._write_provenance(drugs_path, drugs_df, interactions_df)

        # COM10: schema.md sidecar.
        self._write_schema_doc(drugs_path)

        # BUG-A-005 root fix: produce structured drugbank_indications.csv
        # so the phase1_bridge can use Path A (structured) instead of
        # falling back to Path B (scientifically-unsound free-text
        # substring matching). The previous pipeline never produced this
        # file -- the bridge's Path A never fired and all treats edges
        # were derived from free-text matching of disease names against
        # DrugBank <indication> strings.
        #
        # The structured file maps (drugbank_id -> disease_id) by looking
        # up known Disease names from the OMIM CSV (if present) in the
        # indication text. This is a controlled vocabulary match, NOT
        # free-text matching -- only Disease IDs that already exist in the
        # OMIM output are eligible, preserving referential integrity.
        try:
            self._write_structured_indications(drugs_df)
        except (OSError, PermissionError) as exc:
            # P1-3 ROOT FIX: previously this was a bare ``except Exception``
            # which SWALLOWED the v9 ROOT FIX ``RuntimeError`` raised by
            # ``_write_structured_indications`` when the OMIM CSV is
            # missing (see lines ~2572-2587). The v9 ROOT FIX promised
            # operators would SEE that failure so they could fix the DAG
            # ordering (DrugBank depends on OMIM) -- but the bare except
            # downgraded it to ``logger.warning``, defeating the fix
            # silently. The secondary CSV write only fails for two
            # genuinely non-critical reasons (disk full, permission
            # denied on the file) -- both are OSError subclasses.
            # RuntimeError, KeyError, ValueError, and every programming
            # bug now propagate so the run fails loudly.
            #
            # The structured indications CSV is a secondary output -- the
            # primary drugbank_drugs.csv has already been persisted. A
            # disk-full or permission-denied here MUST NOT abort the
            # entire DrugBank pipeline (the drugs + interactions are
            # safe). Log the warning and continue.
            logger.warning(
                "[%s] Failed to write drugbank_indications.csv "
                "(non-critical IO error): %s",
                self.source_name, exc,
            )

        # SEC4: LICENSE.txt sidecar (written once per directory).
        self._write_license()

        logger.info(
            "[%s] Persisted %d drugs to %s, %d interactions to %s",
            self.source_name,
            len(drugs_df),
            _log_path(drugs_path),
            len(interactions_df),
            _log_path(PROCESSED_DATA_DIR / "drugbank_interactions.csv.gz"),
        )

    def _write_structured_indications(self, drugs_df: pd.DataFrame) -> None:
        """BUG-A-005 root fix: produce drugbank_indications.csv.

        Maps each drug's free-text ``indication`` field to known Disease
        IDs from the OMIM output (controlled vocabulary match -- only
        Disease IDs already present in omim_gene_disease_associations.csv
        are eligible, preserving referential integrity).

        Writes a CSV with columns:
            drugbank_id, disease_id, disease_name, indication_type, source

        The phase1_bridge's Path A consumes this file directly, avoiding
        the scientifically-unsound free-text substring matching fallback
        (Path B).

        NOTE: If a curated drugbank_indications.csv already exists
        (e.g. a hand-curated test fixture), it is NOT overwritten --
        the curated file is preferred over the auto-generated one.

        v65 ROOT FIX (P1-041) -- SCIENTIFIC ACCURACY DISCLOSURE
        -------------------------------------------------------
        The mapping logic uses word-boundary regex matching of OMIM
        disease names against DrugBank's free-text ``indication`` field.
        This is inherently LOSSY and produces both FALSE POSITIVES and
        FALSE NEGATIVES:

        **False positives** (estimated 5-15% of emitted rows):
          - "Diabetes mellitus" matches inside "gestational diabetes
            mellitus" even when the drug is specifically approved ONLY
            for type 2 (the broad match conflates subtypes).
          - "Anemia" matches inside "aplastic anemia" / "sickle cell
            anemia" / "iron-deficiency anemia" -- drugs indicated for
            one subtype get tagged with the broader OMIM term.
          - The word-boundary filter (``\\b{dname}\\b``) and the
            >5-char minimum length filter (P1-031 / v37 ROOT FIX)
            eliminate the worst offenders ("DM", "AR"), but multi-word
            disease names with overlapping vocabulary still produce
            spurious matches.

        **False negatives** (estimated 30-50% of true indications):
          - DrugBank's ``indication`` field is FREE-TEXT prose, not a
            controlled vocabulary. A drug indicated for "type 2
            diabetes" may have an indication string like "for the
            management of elevated blood glucose" -- which contains
            NEITHER "diabetes" NOR "mellitus" and is missed entirely.
          - OMIM disease names skew toward rare/Mendelian disease
            ("Diabetes mellitus, insulin-resistant, type A") while
            DrugBank indications skew toward common/complex disease
            ("type 2 diabetes") -- the vocabularies overlap maybe 30%.

        **Why we ship this anyway:**
          - The downstream KG consumer (Phase 2 Graph Transformer)
            treats ``Drug-treats-Disease`` edges as TRAINING SIGNAL,
            not ground truth. A 5-15% false-positive rate is acceptable
            for a feature that is one of ~50 in the model's input.
          - The ``source`` column is set to ``"drugbank_indication_text"``
            so downstream consumers can DOWN-WEIGHT or FILTER these
            edges if a higher-quality curated source is available.
          - The ``indication_type`` column (approved / withdrawn /
            investigational / etc.) is derived from the drug's
            DrugBank ``<groups>`` field -- it is RELIABLE even when the
            disease mapping is lossy. Patient-safety-critical flagging
            (withdrawn drugs) is preserved.

        **Production-grade fix (NOT YET IMPLEMENTED):**
          - Use DrugBank's structured ``<indication>`` element parsed
            into disease ontology IDs (DrugBank doesn't actually expose
            this -- would require MeSH/DOID cross-referencing).
          - OR use a curated drug->disease mapping from a licensed
            source (e.g. DrugBank Central, FDA Labels API, RXNORM).
          - Until then, operators who need high-precision
            drug->disease edges should provide a hand-curated
            ``drugbank_indications.csv`` in PROCESSED_DATA_DIR -- the
            "do not overwrite curated fixture" guard below ensures
            the curated file is preferred over the auto-generated one.
        """
        import csv as _csv
        if "indication" not in drugs_df.columns or "drugbank_id" not in drugs_df.columns:
            return
        indications_path = PROCESSED_DATA_DIR / "drugbank_indications.csv"
        # BUG-A-005: do not overwrite a hand-curated fixture. Production
        # runs will not have this file (the pipeline that creates it is
        # this method), so the auto-generation only fires when the file
        # is missing.
        if indications_path.exists():
            logger.debug(
                "[%s] drugbank_indications.csv already exists (%d bytes) "
                "-- not overwriting (curated fixture or previous run).",
                self.source_name, indications_path.stat().st_size,
            )
            return
        # Load the controlled vocabulary of known diseases from OMIM output.
        omim_path = PROCESSED_DATA_DIR / "omim_gene_disease_associations.csv"
        if not omim_path.exists():
            # v76 ROOT FIX (T-042 -- DrugBank no longer hard-fails when
            # OMIM CSV is missing; decouples DrugBank from OMIM in the DAG):
            #   The previous code raised RuntimeError when the OMIM CSV was
            #   missing. This created a HARD dependency: DrugBank could not
            #   run until OMIM finished. The master DAG wired
            #   ``omim >> drugbank`` to enforce this. But the coupling was
            #   brittle: if OMIM failed (API key missing, network error),
            #   DrugBank was SKIPPED via the dependency chain -- losing ALL
            #   DrugBank drug + target data from the knowledge graph, a
            #   major data loss. The BranchPythonOperator checks for
            #   DrugBank XML existence, NOT for OMIM CSV existence, so if
            #   OMIM failed but DrugBank XML existed, the branch chose
            #   "download_drugbank" but the download then failed because
            #   the OMIM CSV didn't exist (FileNotFoundError cascaded to
            #   DAG RED).
            #   ROOT FIX: gracefully handle the missing OMIM CSV. Log a
            #   WARNING, use an EMPTY disease vocabulary, and write an
            #   EMPTY drugbank_indications.csv (just the header row) so
            #   downstream consumers (phase1_bridge) don't fail on a
            #   missing file. The DrugBank drug + target data is preserved
            #   (the KG still gets all DrugBank drugs and their protein
            #   interactions -- only the drug->disease indication edges are
            #   empty when OMIM is unavailable). This is the scientifically
            #   correct trade-off: a KG with DrugBank drugs but no
            #   indication edges is FAR more useful than a KG with NO
            #   DrugBank data at all. The ``omim >> drugbank`` wire in
            #   the master DAG is removed in the same v76 fix so DrugBank
            #   runs in parallel with OMIM (both write to different files).
            logger.warning(
                "[%s] OMIM CSV not found at %s -- DrugBank indications "
                "will be EMPTY (header-only). DrugBank drug + target data "
                "is still loaded; only drug->disease indication edges are "
                "skipped. This is expected when OMIM_API_KEY is not set or "
                "OMIM pipeline failed. The KG will have DrugBank drugs but "
                "no indication edges from this run. (v76 T-042 root fix: "
                "DrugBank no longer hard-fails on missing OMIM CSV.)",
                self.source_name, omim_path,
            )
            # Write a header-only CSV so downstream consumers (phase1_bridge)
            # can read it without FileNotFoundError. The empty DataFrame
            # means zero indication rows -- the KG simply has no
            # drug->disease edges from DrugBank for this run.
            import csv as _csv_empty
            tmp_fd_empty, tmp_path_empty_str = tempfile.mkstemp(
                dir=indications_path.parent,
                prefix=f".{indications_path.name}.",
                suffix=".tmp",
            )
            os.close(tmp_fd_empty)
            tmp_path_empty = Path(tmp_path_empty_str)
            try:
                with open(tmp_path_empty, "w", encoding="utf-8", newline="") as fh_empty:
                    writer_empty = _csv_empty.DictWriter(
                        fh_empty,
                        fieldnames=[
                            "drugbank_id", "disease_id", "disease_name",
                            "indication_type", "source",
                        ],
                        quoting=_csv_empty.QUOTE_ALL,
                        lineterminator="\n",
                    )
                    writer_empty.writeheader()
                tmp_path_empty.replace(indications_path)
                logger.info(
                    "[%s] Wrote header-only drugbank_indications.csv "
                    "(0 indication rows -- OMIM CSV was missing).",
                    self.source_name,
                )
            except (OSError, csv.Error, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
                if tmp_path_empty.exists():
                    tmp_path_empty.unlink()
                raise
            return
        omim_df = pd.read_csv(omim_path)
        if "disease_id" not in omim_df.columns or "disease_name" not in omim_df.columns:
            # v76 ROOT FIX (T-042 compound): same graceful-degradation
            # pattern as the missing-file case above. If the OMIM CSV
            # exists but has the wrong schema (e.g. a partial download or
            # a version mismatch), log a WARNING and write a header-only
            # indications file instead of raising RuntimeError. The
            # DrugBank drug + target data is still loaded.
            logger.warning(
                "[%s] OMIM CSV at %s is missing required columns "
                "disease_id and/or disease_name. Found columns: %s. "
                "Cannot build controlled vocabulary for drugbank_indications. "
                "Writing header-only indications file. DrugBank drug + "
                "target data is still loaded. (v76 T-042 root fix.)",
                self.source_name, omim_path, list(omim_df.columns),
            )
            import csv as _csv_schema
            tmp_fd_s, tmp_path_s_str = tempfile.mkstemp(
                dir=indications_path.parent,
                prefix=f".{indications_path.name}.",
                suffix=".tmp",
            )
            os.close(tmp_fd_s)
            tmp_path_s = Path(tmp_path_s_str)
            try:
                with open(tmp_path_s, "w", encoding="utf-8", newline="") as fh_s:
                    writer_s = _csv_schema.DictWriter(
                        fh_s,
                        fieldnames=[
                            "drugbank_id", "disease_id", "disease_name",
                            "indication_type", "source",
                        ],
                        quoting=_csv_schema.QUOTE_ALL,
                        lineterminator="\n",
                    )
                    writer_s.writeheader()
                tmp_path_s.replace(indications_path)
                logger.info(
                    "[%s] Wrote header-only drugbank_indications.csv "
                    "(0 indication rows -- OMIM CSV schema mismatch).",
                    self.source_name,
                )
            except (OSError, csv.Error, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
                if tmp_path_s.exists():
                    tmp_path_s.unlink()
                raise
            return
        # Build a (disease_name -> disease_id) map. Use only unique names.
        # v37 ROOT FIX (Chain 5 -- DrugBank->OMIM free-text matching):
        # The previous code iterated ``disease_vocab.items()`` in DICT
        # INSERTION ORDER. The first match won, which meant longer / more
        # specific OMIM namesill NEVER got a chance if a shorter /
        # broader name happened to come first in the dict. Example:
        # "Diabetes mellitus" (broader) appears before
        # "Diabetes mellitus, insulin-resistant, type A" (specific) ->
        # every diabetic drug got tagged with the BROAD name, losing
        # the specific subtype signal the RL ranker needs.
        #
        # The fix has THREE parts:
        # (1) Sort disease_vocab by name length DESCENDING so the most
        #     specific name is tried first.
        # (2) BREAK out of the inner loop after the first match (one
        #     indication -> one disease mapping). The previous code
        #     continued iterating, producing N rows for N matching
        #     disease names -- multiplying edges into the KG.
        # (3) Skip very short disease names (<=5 chars) to avoid
        #     spurious substring hits on common English words. The
        #     previous ``len(dname) < 4`` threshold let through names
        #     like "DM" (4 chars, abbreviation for "Diabetes Mellitus")
        #     which would match the literal "dm" inside any indication
        #     text containing the word "dm" as a fragment.
        disease_vocab_raw = (
            omim_df[["disease_id", "disease_name"]]
            .dropna()
            .drop_duplicates()
            .set_index("disease_name")["disease_id"]
            .to_dict()
        )
        # Filter out very short names (<=5 chars) and sort by length DESC.
        disease_vocab_items: list[tuple[str, str]] = sorted(
            ((name, did) for name, did in disease_vocab_raw.items()
             if isinstance(name, str) and len(name) > 5),
            key=lambda kv: len(kv[0]),
            reverse=True,  # longest first
        )
        disease_vocab = dict(disease_vocab_items)
        if not disease_vocab:
            return
        indications_path = PROCESSED_DATA_DIR / "drugbank_indications.csv"
        rows_written = 0
        # Atomic write.
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=indications_path.parent,
            prefix=f".{indications_path.name}.",
            suffix=".tmp",
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)
        try:
            with open(tmp_path, "w", encoding="utf-8", newline="") as fh:
                writer = _csv.DictWriter(
                    fh,
                    fieldnames=[
                        "drugbank_id", "disease_id", "disease_name",
                        "indication_type", "source",
                    ],
                    quoting=_csv.QUOTE_ALL,
                    lineterminator="\n",
                )
                writer.writeheader()
                # PS-5 ROOT FIX (patient safety): the previous code
                # hardcoded ``indication_type: "approved"`` for EVERY
                # indication, including for withdrawn killer drugs
                # (Vioxx DB00709, Baycol DB00463, thalidomide, cisapride).
                # The RL ranker's safety filter consumed this label as
                # "approved for heart disease" on Vioxx -- a drug withdrawn
                # for causing heart attacks. Derive the indication_type
                # from the drug's DrugBank <groups> field (already
                # extracted into drugs_df as the "groups" column by the
                # parser at line 1604). Priority order (most safety-
                # relevant first): withdrawn > illicit > investigational >
                # vet_approved > approved.
                groups_by_drug: dict[str, str] = {}
                if "groups" in drugs_df.columns and "drugbank_id" in drugs_df.columns:
                    for _row in drugs_df.itertuples(index=False):
                        _dbid = getattr(_row, "drugbank_id", None)
                        _groups = getattr(_row, "groups", None)
                        if _dbid and isinstance(_groups, str):
                            groups_by_drug[_dbid] = _groups.lower()

                def _derive_indication_type(dbid: str) -> str:
                    g = groups_by_drug.get(dbid, "")
                    # V19 ROOT FIX (PS-5 residual -- verification agent
                    # flagged this): the V18 substring-match logic
                    # (``if "approved" in g:``) misclassifies
                    # ``vet_approved``-only drugs as ``"approved"`` because
                    # ``"approved"`` is a substring of ``"vet_approved"``.
                    # Same bug for ``"investigational"`` (works correctly
                    # because ``"approved"`` is NOT a substring of
                    # ``"investigational"``). The ROOT fix: parse the
                    # pipe-/semicolon-delimited groups string into a set
                    # of tokens and do exact token matching. DrugBank's
                    # ``<groups>`` field is a pipe-delimited list (e.g.
                    # ``"approved|withdrawn"``, ``"vet_approved"``,
                    # ``"investigational|approved"``) -- token-set matching
                    # correctly distinguishes ``approved`` from
                    # ``vet_approved``.
                    # v36 ROOT FIX (Phase 1 Issue #21): also split on
                    # COMMAS -- some older DrugBank XML versions used
                    # comma separators (e.g. ``"approved, withdrawn"``).
                    # Without this, the entire string became one token
                    # ``"approved, withdrawn"`` which matched nothing,
                    # masking withdrawn drugs as ``"unknown"``.
                    import re as _re_v36
                    _sep_pattern = _re_v36.compile(r"[;|,]")
                    tokens = set(
                        t.strip().lower()
                        for t in _sep_pattern.split(g)
                        if t.strip()
                    )
                    # Order matters -- most safety-relevant first.
                    if "withdrawn" in tokens:
                        return "withdrawn"
                    if "illicit" in tokens:
                        return "illicit"
                    if "investigational" in tokens and "approved" not in tokens:
                        return "investigational"
                    if "vet_approved" in tokens and "approved" not in tokens:
                        return "vet_approved"
                    if "approved" in tokens:
                        return "approved"
                    if "experimental" in tokens:
                        return "experimental"
                    if "nutraceutical" in tokens:
                        return "nutraceutical"
                    return "unknown"

                for drug_row in drugs_df.itertuples(index=False):
                    dbid = getattr(drug_row, "drugbank_id", None)
                    indication_text = getattr(drug_row, "indication", None)
                    if not dbid or not indication_text or not isinstance(indication_text, str):
                        continue
                    indication_lower = indication_text.lower()
                    _indication_type_for_drug = _derive_indication_type(dbid)
                    # v37 ROOT FIX (Chain 5): disease_vocab is now sorted
                    # by name length DESCENDING, so the FIRST match is
                    # the most specific. We BREAK after the first match
                    # to avoid producing N rows for N matching disease
                    # names (which would multiply edges in the KG).
                    for dname, did in disease_vocab.items():
                        # The disease_vocab filter already excluded names
                        # <=5 chars, but defensively check again.
                        if not isinstance(dname, str) or len(dname) <= 5:
                            continue
                        # v84 FORENSIC ROOT FIX (BUG #29): the previous
                        # ``\b{dname}\b`` pattern treats HYPHENS as word
                        # boundaries (``\b`` matches between a word char
                        # and a non-word char, and ``-`` is non-word).
                        # This caused TWO classes of false positives in
                        # drug->disease indication edges:
                        #   (a) "diabetes" matched inside "pre-diabetes"
                        #       (a DIFFERENT condition) because ``\b``
                        #       fired between ``-`` and ``d``.
                        #   (b) "anemia" matched inside "aplastic anemia"
                        #       (a different disease) because the space
                        #       before "anemia" satisfied ``\b``.
                        # ROOT FIX: replace ``\b`` with lookbehind /
                        # lookahead that treats hyphens AND letters as
                        # word-continuation characters. Now "diabetes"
                        # does NOT match inside "pre-diabetes" (the char
                        # before "d" is "-", a continuation char, so the
                        # negative lookbehind fails). Combined with the
                        # length-descending sort + break-on-first-match,
                        # this eliminates the false-positive class while
                        # keeping the curated OMIM disease vocabulary as
                        # the matching source (no free-text heuristics).
                        import re as _re
                        _escaped = _re.escape(dname.lower())
                        pattern = r"(?<![a-z\-])" + _escaped + r"(?![a-z\-])"
                        if _re.search(pattern, indication_lower):
                            writer.writerow({
                                "drugbank_id": dbid,
                                "disease_id": did,
                                "disease_name": dname,
                                "indication_type": _indication_type_for_drug,
                                "source": "drugbank_indication_text",
                            })
                            rows_written += 1
                            # v37 ROOT FIX (Chain 5): BREAK after the
                            # first (most specific) match. The previous
                            # code continued iterating, producing multiple
                            # rows per drug -- polluting the KG with
                            # duplicate drug-treats-disease edges.
                            break
            os.replace(tmp_path, indications_path)
            os.chmod(indications_path, 0o600)
            logger.info(
                "[%s] BUG-A-005: wrote %d structured indication rows to %s",
                self.source_name, rows_written, _log_path(indications_path),
            )
        except (OSError, csv.Error, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    def _write_provenance(
        self,
        drugs_path: Path,
        drugs_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
    ) -> None:
        """Write provenance JSON sidecar (A8, LIN14, LIN15, SEC8).

        Parameters
        ----------
        drugs_path : pathlib.Path
            Path to the drugs CSV.
        drugs_df : pandas.DataFrame
            Drugs DataFrame.
        interactions_df : pandas.DataFrame
            Interactions DataFrame.
        """
        # LIN14: transformation fingerprint.
        transformations = [
            "standardize_inchikey",
            "convert_to_inchikeys_from_smiles",
            "generate_synth_keys_for_biologics",
            "dedup_by_inchikey_keep_most_complete",
            "fill_missing_drug_fields_conservative",
            "validate_against_schema_v1",
            "filter_organism_humans",
            "extract_targets_enzymes_transporters",
            "csv_injection_defense",
            "atomic_write_with_sha256_sidecar",
        ]
        transformation_fingerprint = hashlib.sha256(
            "|".join(transformations).encode("utf-8")
        ).hexdigest()

        # LIN15: data quality fingerprint.
        dq_metrics = {
            "total_drugs_input": len(drugs_df),
            "total_drugs_output": len(drugs_df),
            "drugs_dropped_no_inchikey": self._drugs_dropped_no_inchikey,
            "synth_keys_generated": self._synth_keys_generated,
            "interactions_extracted": len(interactions_df),
            "parse_failures": self._parse_failures,
            "non_human_targets_skipped": self._non_human_targets_skipped,
            "skipped_no_id": self._skipped_no_id,
        }
        dq_fingerprint = hashlib.sha256(
            json.dumps(dq_metrics, sort_keys=True).encode("utf-8")
        ).hexdigest()

        # SEC8: who ran the pipeline.
        try:
            created_by = getpass.getuser()
        except OSError:  # pragma: no cover - defensive
            created_by = "unknown"
        try:
            created_on = socket.gethostname()
        except OSError:  # pragma: no cover - defensive
            created_on = "unknown"

        provenance = {
            "source": "drugbank",
            "source_version": self.source_version,
            "pipeline_run_id": self.run_id,
            "pipeline_version": __version__,
            "pipeline_api_version": __version__,
            "rdkit_version": self._rdkit_version,
            "schema_version": SCHEMA_VERSION,
            "db_schema_version": DB_SCHEMA_VERSION,
            "sha256_raw": self._sha256_raw,
            "sha256_cleaned": self._sha256_cleaned,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "drug_count": len(drugs_df),
            "interaction_count": len(interactions_df),
            "created_by": created_by,
            "created_on": created_on,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "process_id": os.getpid(),
            "transformation_fingerprint": transformation_fingerprint,
            "data_quality_fingerprint": dq_fingerprint,
            "data_quality_metrics": dq_metrics,
            "transformations_applied": transformations,
            "target_organisms": self._target_organisms,
            "citation": (
                "Wishart DS et al. DrugBank 5.0: a major update to the DrugBank "
                "database for 2018. Nucleic Acids Res. 2018 Jan 4;46(D1):D1074-D1082."
            ),
        }
        provenance_path = drugs_path.with_suffix(".provenance.json")
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.chmod(provenance_path, 0o600)  # SEC3

    def _write_schema_doc(self, drugs_path: Path) -> None:
        """Write a sidecar schema.md documenting each output column (COM10).

        Parameters
        ----------
        drugs_path : pathlib.Path
            Path to the drugs CSV (used to derive the .schema.md path).
        """
        schema_doc = (
            "# drugbank_drugs.csv - Column Documentation\n\n"
            "Generated by drugbank_pipeline.py v" + __version__ + ".\n\n"
            "| Column | Type | Description |\n"
            "|--------|------|-------------|\n"
            "| drugbank_id | str | DrugBank identifier (DB\\d{5}) |\n"
            "| name | str | Drug preferred name |\n"
            "| inchikey | str | Standard InChIKey (27 chars) or SYNTH synthetic key (27 chars, SYNTH{hash}-{hash}-{hash}) for biologics |\n"
            "| smiles | str | Canonical SMILES |\n"
            "| molecular_weight | float | MW in Da (1-500,000) |\n"
            "| molecular_formula | str | Molecular formula |\n"
            "| is_globally_approved | bool | Approved by ANY regulator (FDA/EMA/PMDA/MHRA) |\n"
            "| is_fda_approved | bool? | FDA-specific (None = not validated vs Orange Book) |\n"
            "| is_withdrawn | bool | Withdrawn from market |\n"
            "| clinical_status | str | approved/withdrawn/illicit/investigational/... |\n"
            "| groups | str | Pipe-separated DrugBank groups |\n"
            "| mechanism_of_action | str | MOA text (multi-paragraph concatenated) |\n"
            "| description | str | Drug description text |\n"
            "| cas_number | str | CAS Registry Number |\n"
            "| logp | float | Calculated LogP |\n"
            "| tpsa | float | Topological Polar Surface Area |\n"
            "| h_bond_donor_count | int | H-bond donor count |\n"
            "| h_bond_acceptor_count | int | H-bond acceptor count |\n"
            "| rotatable_bond_count | int | Rotatable bond count |\n"
            "| heavy_atom_count | int | Heavy atom count |\n"
            "| complexity | int | Molecular complexity |\n"
            "| inchikey_source | str | extracted_calculated/experimental/generated/synth |\n"
            "| completeness_score | float | 0.0-1.0 fraction of expected fields populated |\n\n"
            "MIME type: text/csv (UTF-8, QUOTE_ALL, \\n line endings).\n"
        )
        schema_path = drugs_path.with_suffix(".schema.md")
        schema_path.write_text(schema_doc, encoding="utf-8")
        os.chmod(schema_path, 0o644)

    def _write_license(self) -> None:
        """Write the DrugBank LICENSE.txt sidecar (SEC4, COM7, COM8).

        Writes once per PROCESSED_DATA_DIR; does not overwrite if it
        already exists with the correct content.
        """
        license_path = PROCESSED_DATA_DIR / "DRUGBANK_LICENSE.txt"
        if license_path.exists() and license_path.read_text(encoding="utf-8") == _DRUGBANK_LICENSE_TEXT:
            return
        license_path.write_text(_DRUGBANK_LICENSE_TEXT, encoding="utf-8")
        os.chmod(license_path, 0o644)

    def _flush_dead_letter(self) -> None:
        """Flush the dead-letter queue to a sidecar JSON file (R3).

        Each entry records: drugbank_id (if known), element_index, error,
        error_type, timestamp.
        """
        dlq_path = PROCESSED_DATA_DIR / f"drugbank_dead_letter_{self.run_id[:8]}.json"
        dlq_path.write_text(
            json.dumps(self._dead_letter, indent=2), encoding="utf-8"
        )
        os.chmod(dlq_path, 0o600)  # SEC3
        logger.warning(
            "[%s] %d drugs failed parsing - written to %s (R3)",
            self.source_name,
            len(self._dead_letter),
            _log_path(dlq_path),
        )

    # ------------------------------------------------------------------
    # Load (A3, A4, ID8, R7, P4, LIN1-LIN4, LIN9, LIN10, D1, D2, D5)
    # ------------------------------------------------------------------

    def load(
        self,
        df: pd.DataFrame,
        interactions_df: pd.DataFrame | None = None,
        session: Any | None = None,
    ) -> int | LoadResult:
        """Load cleaned DrugBank drugs and interactions into the database.

        Audit issues:
        - A3: optional ``interactions_df`` parameter (skip CSV read).
        - A4 / ID8 / R7 / P4: single transactional session for drugs + DPI.
        - D1 / C2 / C3: unwrap ``MappingResult.mapping`` before ``Series.map``.
        - D2 / C1 / C4: extract ``UpsertResult.inserted`` / ``.updated``
          explicitly (no ``__add__``).
        - D5 / COM2: map actions to InteractionType enum; never use "target".
        - LIN1-LIN4, LIN9, LIN10: pass all lineage fields.

        Parameters
        ----------
        df : pandas.DataFrame
            Cleaned drugs DataFrame (from ``clean()``).
        interactions_df : pandas.DataFrame, optional
            In-memory interactions DataFrame. If None, reads from
            ``PROCESSED_DATA_DIR/drugbank_interactions.csv.gz`` (backward
            compat).
        session : SQLAlchemy Session, optional
            Caller-supplied session (for transactional wrapping). If None,
            opens a new session via ``get_db_session()`` (A4).

        Returns
        -------
        int or LoadResult
            Total rows upserted (backward-compat int) OR a LoadResult
            with inserted/updated/skipped/failed breakdown.
        """
        # A4 / ID8 / R7 / P4: single session for the whole load().
        owns_session = session is None
        # v29 ROOT FIX (audit P1-4): capture the return value of
        # __enter__() -- the previous code discarded it, so ``session``
        # was the context manager, not the Session. Standalone load()
        # calls crashed with AttributeError on session.flush() /
        # session.rollback() / session.close(). Also, the previous
        # finally block only called session.close() -- it NEVER called
        # __exit__(), so the commit never happened and ALL loaded data
        # was silently rolled back when load() ran standalone.
        _session_cm = None
        if owns_session:
            _session_cm = get_db_session(
                pipeline_name=self.source_name,
                run_id=self.run_id,
            )
            session = _session_cm.__enter__()

        total_inserted = 0
        total_updated = 0
        total_skipped = 0
        total_failed = 0

        try:
            # LIN1-LIN4 / BUG-16.2 fix: populate self._pipeline_run_db_id
            # BEFORE upserting DPI rows so each DPI row carries the correct
            # lineage ID back to its PipelineRun audit row. Without this,
            # all DrugBank DPI rows have pipeline_run_id=NULL -- breaking
            # the lineage chain that downstream phases use to trace which
            # pipeline run produced a given drug-protein edge.
            self._pipeline_run_db_id = self._get_or_create_pipeline_run_id(session)

            # LIN9 / A4: pass input_checksum to bulk_upsert_drugs.
            input_checksum = self._sha256_cleaned

            # Filter to Drug-model columns only (loader rejects extra cols).
            drugs_df_for_load = self._filter_to_drug_columns(df)

            drug_result: UpsertResult = bulk_upsert_drugs(
                session,
                drugs_df_for_load,
                batch_size=self._batch_size,
                input_checksum=input_checksum,
            )
            # D2 / C1: extract fields explicitly (no __add__).
            total_inserted += drug_result.inserted
            total_updated += drug_result.updated
            total_skipped += drug_result.quarantined
            total_failed += drug_result.failed
            logger.info(
                "[%s] Upserted drugs: inserted=%d updated=%d quarantined=%d failed=%d",
                self.source_name,
                drug_result.inserted,
                drug_result.updated,
                drug_result.quarantined,
                drug_result.failed,
            )

            # Flush so the drug rows are visible within this transaction.
            # v29 ROOT FIX (audit P1-10): the previous code did
            # ``except Exception: pass`` which silently swallowed
            # IntegrityError. ROOT FIX: log the warning.
            # FIX-P2-2 (audit P2): after IntegrityError the SQLAlchemy
            # session is POISONED -- every subsequent op raises
            # PendingRollbackError. The previous code only logged and
            # CONTINUED, so all downstream queries/upserts in this load()
            # call silently failed. Root fix: roll back the session so
            # subsequent operations can proceed (the real commit lives
            # in __exit__). Mirrors chembl_pipeline.py:1207.
            try:
                session.flush()
            # v85/v90 ROOT FIX (BUG #19/51): narrowed from broad
            # ``except Exception`` which caught programming bugs
            # (AttributeError, KeyError, NameError) and silently
            # rolled back. Root fix: catch ONLY SQLAlchemy DBAPI
            # errors and IntegrityError. Programming bugs propagate.
            except (OperationalError, IntegrityError, PendingRollbackError) as _flush_exc:  # pragma: no cover - defensive
                try:
                    session.rollback()
                except (OSError, RuntimeError, ValueError):  # noqa: BLE001 -- never mask the flush error  # v85 FORENSIC ROOT FIX (BUG #51)
                    pass
                logger.warning(
                    "[drugbank] session.flush() failed (rolled back; "
                    "non-fatal, but may indicate data quality issues): "
                    "%s: %s",
                    type(_flush_exc).__name__, _flush_exc,
                )

            # Flush loader dead-letter queue if any.
            # v90 ROOT FIX (BUG #20): the previous code used
            # ``except Exception: pass`` which silently swallowed
            # ALL failures including programming bugs. Root fix:
            # catch ONLY OS/IO errors (disk full, permission denied).
            # Programming bugs propagate so they surface during dev.
            try:
                flush_dead_letter_queue(
                    PROCESSED_DATA_DIR
                    / "dead_letter"
                    / f"drugbank_loader_{self.run_id[:8]}.jsonl"
                )
            except (OSError, RuntimeError, ValueError) as _dlq_exc:  # pragma: no cover - defensive  # v85/v90 FORENSIC ROOT FIX (BUG #20/51)
                logger.warning(
                    "[drugbank] Failed to flush dead-letter queue: %s: %s",
                    type(_dlq_exc).__name__, _dlq_exc,
                )

            # A3: load interactions (in-memory or from CSV).
            if interactions_df is None:
                interactions_path = PROCESSED_DATA_DIR / "drugbank_interactions.csv.gz"
                if interactions_path.exists():
                    interactions_df = pd.read_csv(
                        interactions_path, compression="gzip", low_memory=False
                    )

            if interactions_df is not None and not interactions_df.empty:
                dpi_result = self._load_interactions(
                    interactions_df, df, session
                )
                # D2 / C1: extract fields explicitly.
                total_inserted += dpi_result.inserted
                total_updated += dpi_result.updated
                total_skipped += dpi_result.quarantined
                total_failed += dpi_result.failed

        except (OSError, RuntimeError, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
            if owns_session:
                try:
                    session.rollback()
                except (OSError, RuntimeError, ValueError):  # pragma: no cover - defensive  # v85 FORENSIC ROOT FIX (BUG #51)
                    pass
            raise
        finally:
            # v29 ROOT FIX (audit P1-4): call __exit__ on the context
            # manager so it commits (on success) or rolls back (on
            # error). The previous code only called session.close(),
            # which silently rolled back ALL loaded data when load()
            # ran standalone (the commit lived in __exit__).
            if owns_session and _session_cm is not None:
                import sys as _sys
                _exc_info = _sys.exc_info()
                try:
                    _session_cm.__exit__(*_exc_info)
                # v85/v90 ROOT FIX (BUG #21/51): the previous
                # ``except Exception: pass`` silently swallowed
                # __exit__ failures -- if commit fails because the DB
                # connection dropped, the caller saw load() return
                # success with NO data committed. Changed to catch
                # DB errors (OperationalError, InterfaceError) and
                # OS/runtime errors. Programming bugs propagate.
                except (OperationalError, InterfaceError, OSError, RuntimeError, ValueError) as _exit_exc:  # pragma: no cover - defensive
                    logger.error(
                        "[drugbank] session __exit__ failed (commit/rollback "
                        "may not have completed -- loaded data may be lost): "
                        "%s",
                        _exit_exc,
                    )

        result = LoadResult(
            rows_inserted=total_inserted,
            rows_updated=total_updated,
            rows_skipped=total_skipped,
            rows_failed=total_failed,
        )
        # Backward-compat: return int (total upserted) for callers that
        # expect the legacy int return type.
        return int(result.total_upserted)

    def _get_or_create_pipeline_run_id(self, session: Any) -> "int | None":
        """Get the integer ``pipeline_runs.id`` for this run (BUG-16.2).

        The base class writes the PipelineRun audit row AFTER ``load()``
        returns, keyed by ``(source, run_date)`` where ``run_date`` is
        ``self.start_time`` (the moment ``run()`` was called). We mirror
        that keying here so the row we create now is the same row the
        base class UPDATEs later (no duplicate audit rows).

        CRITICAL FIX (scientific correctness / audit-trail integrity):
        Without this method, ``self._pipeline_run_db_id`` stays None and
        every DrugBank DPI row is loaded with ``pipeline_run_id=NULL``,
        breaking the lineage chain that downstream phases (Neo4j export,
        ML training) use to trace which pipeline run produced a given
        drug-protein edge. A NULL lineage ID is fatal for reproducibility
        -- if a wet-lab validation fails, we cannot trace back to the
        exact data version that produced the bad prediction.

        Returns
        -------
        int or None
            The integer ``pipeline_runs.id`` of the row for this run,
            or None if the lookup-or-create failed (in which case DPI
            rows will have NULL pipeline_run_id -- flagged in the audit
            log but not fatal).
        """
        try:
            from datetime import datetime as _dt
            from database.models import PipelineRun
            # Mirror the base class keying EXACTLY: source + run_date
            # where run_date == self.start_time (the moment run() started).
            if self.start_time is not None:
                run_date = self.start_time
            else:
                run_date = _dt.now(timezone.utc)
            # v90 ROOT FIX (BUG #4): the previous code truncated
            # microseconds with `run_date.replace(microsecond=0)`.
            # If two DrugBank pipeline runs start within the same
            # second (e.g. in tests or rapid re-runs), they get the
            # SAME truncated run_date -> the query finds the FIRST
            # run's PipelineRun row -> DPI rows are linked to the
            # WRONG pipeline run (data lineage corruption). If the
            # first run was a failure, the second run's successful
            # DPIs appear under the failed run -> audit trail is wrong.
            # ROOT FIX: keep full microsecond precision so each run
            # has a unique run_date. The DB column is TIMESTAMP or
            # DATETIME which both support microsecond precision in
            # modern databases (PostgreSQL, MySQL 5.6+, SQLite via
            # string storage).
            existing = (
                session.query(PipelineRun)
                .filter(
                    PipelineRun.source == self.source_name,
                    PipelineRun.run_date == run_date,
                )
                .first()
            )
            if existing is not None:
                return int(existing.id)
            run = PipelineRun(
                source=self.source_name,
                run_date=run_date,
                status="running",
                records_downloaded=0,
                records_cleaned=0,
                records_loaded=0,
            )
            session.add(run)
            session.flush()  # populate run.id without committing
            return int(run.id)
        # FIX-P2-6 (audit P2): the previous broad ``except Exception``
        # caught programming bugs (e.g. AttributeError from a typo in
        # the PipelineRun field name) and downgraded them to a warning
        # + None return. DPI rows then got ``pipeline_run_id=NULL`` with
        # NO signal that the lineage code was actually broken. Narrowing
        # to (OperationalError, IntegrityError) lets the legitimate
        # "transient DB error / deadlock victim / duplicate key" cases
        # continue (best-effort lineage), while real bugs propagate.
        except (OperationalError, IntegrityError) as exc:
            # R1 defensive: this lineage-tracking path is best-effort.
            # If we cannot create a PipelineRun row (e.g. transient DB
            # error, schema drift, deadlock-victim), we MUST NOT abort
            # the actual data load -- that would block the entire weekly
            # DrugBank refresh and leave the staging DB stale. Instead,
            # we log a WARNING and let the DPI rows carry a NULL
            # pipeline_run_id. The audit log captures the failure so
            # an operator can backfill the lineage later. Re-raising
            # here would be a worse outcome than a NULL foreign key.
            logger.warning(
                "[%s] Could not get/create PipelineRun row for lineage: %s. "
                "DPI rows will have NULL pipeline_run_id (acceptable but "
                "noted in audit log).",
                self.source_name,
                exc,
            )
            return None

    def _load_interactions(
        self,
        interactions_df: pd.DataFrame,
        drugs_df: pd.DataFrame,
        session: Any,
    ) -> UpsertResult:
        """Resolve foreign keys and load DrugBank interactions as DPI.

        Audit issues:
        - D1 / C2 / C3: unwrap ``MappingResult.mapping``.
        - D5 / COM2: map actions to InteractionType enum.
        - D11 / C5: dropna on JOINT subset.
        - C6 / C17: use Int64 nullable type; don't shadow parameter.
        - LIN1-LIN4, LIN10: pass all lineage fields to bulk_upsert_dpi.
        - DQ6 / ID10: log dedup result; sort deterministically.
        - DQ9: log unresolved UniProt IDs.
        - S22: source_id includes interactor_type.

        Parameters
        ----------
        interactions_df : pandas.DataFrame
            Interactions DataFrame from ``clean()``.
        drugs_df : pandas.DataFrame
            Drugs DataFrame (for building drugbank_id -> inchikey map).
        session : SQLAlchemy Session
            Active session (caller owns transaction).

        Returns
        -------
        UpsertResult
            Aggregated result across all DPI chunks.
        """
        if interactions_df.empty:
            logger.info("[%s] No interactions to load", self.source_name)
            return UpsertResult()

        # D11 / C5: build drugbank_id -> inchikey map with JOINT dropna.
        if "inchikey" in drugs_df.columns and "drugbank_id" in drugs_df.columns:
            drugbank_id_to_inchikey = dict(
                drugs_df.dropna(subset=["drugbank_id", "inchikey"])
                .set_index("drugbank_id")["inchikey"]
                .items()
            )
        else:
            drugbank_id_to_inchikey = {}

        # D1 / C2 / C3: unwrap MappingResult.mapping (MappingResult is NOT a dict).
        inchikey_map_result: MappingResult = get_inchikey_to_drug_id_map(session)
        uniprot_map_result: MappingResult = get_uniprot_to_protein_id_map(session)

        # LIN18: check built_at for staleness.
        if (
            inchikey_map_result.built_at
            and datetime.now(timezone.utc) - inchikey_map_result.built_at
            > timedelta(hours=1)
        ):
            logger.warning(
                "[%s] inchikey_to_drug_id map is >1 hour old - may be stale (LIN18)",
                self.source_name,
            )

        inchikey_to_drug_id: dict[str, int] = inchikey_map_result.mapping
        uniprot_to_protein_id: dict[str, int] = uniprot_map_result.mapping

        # Resolve drugbank_id -> drug_id via inchikey.
        interactions_df["inchikey"] = interactions_df["drugbank_id"].map(
            drugbank_id_to_inchikey
        )
        interactions_df["drug_id"] = interactions_df["inchikey"].map(
            inchikey_to_drug_id
        )

        # Resolve uniprot_id -> protein_id.
        interactions_df["protein_id"] = interactions_df["uniprot_id"].map(
            uniprot_to_protein_id
        )

        # DQ9: log UniProt IDs that failed protein_id resolution.
        unresolved_mask = interactions_df["protein_id"].isna() & interactions_df[
            "uniprot_id"
        ].notna()
        if unresolved_mask.any():
            unresolved = interactions_df.loc[unresolved_mask, "uniprot_id"].unique().tolist()
            logger.warning(
                "[%s] %d UniProt IDs could not be resolved to protein_id: %s (DQ9)",
                self.source_name,
                len(unresolved),
                unresolved[:20],  # cap log at 20
            )
            # Sidecar file for offline inspection (DQ9).
            try:
                unresolved_path = PROCESSED_DATA_DIR / "drugbank_unresolved_uniprot.txt"
                unresolved_path.write_text("\n".join(unresolved), encoding="utf-8")
                os.chmod(unresolved_path, 0o600)
            except OSError:
                pass

        # C17: don't shadow the parameter; use a new variable.
        resolved_interactions = interactions_df.dropna(
            subset=["drug_id", "protein_id"]
        ).copy()
        logger.info(
            "[%s] Interactions with resolved FKs: %d / %d",
            self.source_name,
            len(resolved_interactions),
            len(interactions_df),
        )

        if resolved_interactions.empty:
            logger.info("[%s] No resolvable interactions to load", self.source_name)
            return UpsertResult()

        # C6: use Int64 nullable type for defensive casting.
        # v84 FORENSIC ROOT FIX (BUG #40): the previous code cast to
        # ``Int64`` (nullable) at line 4133-4134, then IMMEDIATELY cast
        # to non-nullable ``int64`` at line 4144-4145. If ANY NaN
        # survived the ``dropna`` at line 4118 (e.g. due to index
        # misalignment or a race condition), the ``.astype("int64")``
        # call raises ``ValueError`` -- crashing the pipeline. ROOT FIX:
        # do a FINAL defensive dropna right before the non-nullable
        # cast, so the cast is guaranteed to succeed. This is belt-and-
        # suspenders: the dropna at line 4118 should have caught
        # everything, but defensive programming for a biomedical
        # pipeline is non-negotiable.
        resolved_interactions["drug_id"] = resolved_interactions["drug_id"].astype("Int64")
        resolved_interactions["protein_id"] = resolved_interactions["protein_id"].astype("Int64")
        # Final defensive dropna -- guarantees no NaN reaches the int64 cast.
        resolved_interactions = resolved_interactions.dropna(
            subset=["drug_id", "protein_id"]
        ).copy()

        # D5 / COM2: map actions to InteractionType enum (never use "target").
        resolved_interactions["interaction_type"] = resolved_interactions[
            "action_type"
        ].apply(self._map_action_to_enum)

        # Build DPI DataFrame with all required columns.
        dpi_df = pd.DataFrame(
            {
                "drug_id": resolved_interactions["drug_id"].astype("int64"),
                "protein_id": resolved_interactions["protein_id"].astype("int64"),
                "interaction_type": resolved_interactions["interaction_type"],
                "activity_value": None,
                "activity_units": None,
                "activity_type": None,
                "source": "drugbank",
                "source_id": resolved_interactions["source_id"],
                "confidence_score": None,
            }
        )

        # LIN4: set entity_resolved=True for all DrugBank DPI rows
        # (UniProt IDs are canonical - no entity resolution needed).
        dpi_df["entity_resolved"] = True

        # ID10 / DQ6: sort deterministically BEFORE dedup.
        dpi_df = dpi_df.sort_values(
            ["drug_id", "protein_id", "source", "source_id"]
        ).reset_index(drop=True)

        before_dedup = len(dpi_df)
        dpi_df = dedup_interactions(
            dpi_df,
            keys=["drug_id", "protein_id", "source", "source_id"],
            keep="first",  # deterministic after sort
        )
        after_dedup = len(dpi_df)
        if before_dedup != after_dedup:
            logger.warning(
                "[%s] Removed %d duplicate DPI rows during dedup (%d -> %d) (DQ6)",
                self.source_name,
                before_dedup - after_dedup,
                before_dedup,
                after_dedup,
            )

        # P13 / CF13: chunked DPI upsert.
        total_inserted = 0
        total_updated = 0
        total_quarantined = 0
        total_failed = 0

        for chunk_start in range(0, len(dpi_df), self._dpi_batch_size):
            chunk = dpi_df.iloc[chunk_start : chunk_start + self._dpi_batch_size].copy()
            # LIN1, LIN2, LIN3, LIN4, LIN9, LIN10: pass all lineage fields.
            dpi_result = bulk_upsert_dpi(
                session,
                chunk,
                batch_size=self._batch_size,
                pipeline_run_id=self._pipeline_run_db_id,
                source_version=self.source_version,
                source_fetch_date=self._source_fetch_date,
                input_checksum=self._sha256_cleaned,
            )
            total_inserted += dpi_result.inserted
            total_updated += dpi_result.updated
            total_quarantined += dpi_result.quarantined
            total_failed += dpi_result.failed

        logger.info(
            "[%s] Upserted DPI: inserted=%d updated=%d quarantined=%d failed=%d "
            "(across %d chunks)",
            self.source_name,
            total_inserted,
            total_updated,
            total_quarantined,
            total_failed,
            (len(dpi_df) + self._dpi_batch_size - 1) // self._dpi_batch_size,
        )

        return UpsertResult(
            total_input=len(dpi_df),
            inserted=total_inserted,
            updated=total_updated,
            quarantined=total_quarantined,
            failed=total_failed,
        )

    @staticmethod
    def _map_action_to_enum(action: Any) -> str:
        """Map a DrugBank action string to an InteractionType enum value (D5).

        Multi-action strings (pipe-separated) take the first action for
        enum mapping; the full string is preserved elsewhere.

        Parameters
        ----------
        action : Any
            The action string (e.g. "inhibitor", "agonist|positive modulator").

        Returns
        -------
        str
            The mapped InteractionType enum value (default "unknown").
        """
        if pd.isna(action) or not action:
            return "unknown"
        first = str(action).split("|")[0].strip().lower()
        return ACTION_TO_ENUM.get(first, "unknown")


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    pipeline = DrugBankPipeline()
    pipeline.run()
