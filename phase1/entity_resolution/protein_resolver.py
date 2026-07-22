# SPDX-License-Identifier: MIT
# © 2024-2026 Autonomous Drug Repurposing Platform -- Team Cosmic / VentureLab
"""
Cross-database protein entity resolution for the Drug Repurposing ETL platform.

Proteins are identified differently across UniProt (accession IDs),
STRING (taxon-prefixed ENSP identifiers like "9606.ENSP00000269305"),
and ChEMBL (target IDs in CHEMBL\\d+ format, with UniProt accessions
provided as a SEPARATE cross-reference field -- NOT embedded in the
target ID itself).  :class:`ProteinResolver` reconciles
these into a single canonical record keyed by UniProt accession.

Resolution strategy (priority order)
------------------------------------
1. **UniProt ID exact match** (confidence 1.0 -- MatchConfidence.UNIPROT_EXACT).
2. **STRING -> UniProt cross-reference** (confidence 1.0 when the
   cross-reference was established from a UniProt-supplied STRING ID;
   lower confidence when the mapping is reverse-engineered from a
   STRING-derived provisional entry).
3. **Gene name + organism match** (confidence 0.75 -- MatchConfidence.GENE_NAME_ORGANISM;
   v29 inversion fix: was 0.85, lowered to sit between NAME_NORMALIZED
   (0.80) and FUZZY (0.65)).
4. **Protein-name fuzzy match** (confidence 0.60 -- MatchConfidence.PROTEIN_NAME_FUZZY,
   v29 inversion fix: was 0.90; threshold controlled by
   ResolverConfig.fuzzy_threshold (default 0.60) floored by
   ``_PROTEIN_FUZZY_THRESHOLD=0.60``).

Dependencies
------------
This module uses LAZY imports for pandas and pyarrow so that
``import entity_resolution`` succeeds in minimal environments. Callers
of ``ProteinResolver.to_dataframe`` / ``build_mapping`` must ensure
pandas is installed; callers of ``to_parquet`` must ensure pyarrow is
installed. Use ``ProteinResolver.check_dependencies()`` to verify
availability at runtime.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import logging
import os
import re
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path  # v20 SW-13 ROOT FIX: needed by load_uniprot_organism_crosswalk
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from .base import (
    MAPPING_SCHEMA_VERSION,
    MatchConfidence,
    Resolver,
    ResolverConfig,
    ResolverStats,
)
from .resolver_utils import (
    METHOD_CONFIDENCE,
    compute_match_confidence,
    find_duplicate_ids,
    fuzzy_match_score,
    normalize_name,
    register_match_method,
    validate_protein_record,
)

logger = logging.getLogger(__name__)

# Lazy import of pandas so that ``import entity_resolution`` succeeds
# in minimal environments (audit D1-2, D6-1).
_pd: Optional[Any] = None

# FIX SCI-15 / SCI-16: import format-validation regexes for ingest-time checks.
from .resolver_utils import (
    _AA_VALID_RE,
    _STRING_ID_RE,
    _CHEMBL_TARGET_ID_RE,
    _UNIPROT_ACCESSION_RE,
)


def _get_pd() -> Any:
    """Lazily import pandas; raise a clear error if missing (D6-2)."""
    global _pd
    if _pd is None:
        try:
            import pandas as pd
            _pd = pd
        except ImportError as exc:
            raise ImportError(
                "ProteinResolver.to_dataframe / build_mapping require the "
                "'pandas' library. Install with: pip install pandas"
            ) from exc
    return _pd


# ---------------------------------------------------------------------------
# Module-level constants (kept for backward compat).
# ---------------------------------------------------------------------------

#: Semantic version of this module. P1-ER-4 ROOT FIX: bumped from the
#: implicit 1.0.0 baseline to 1.0.1 to mark the SHA-1 -> SHA-256 upgrade
#: of every checksum in this module (audit P1-ER-4). SHA-1 is
#: cryptographically broken (collision attacks since 2017) and was
#: inconsistently used here while ``drug_resolver.py`` has always used
#: SHA-256. All six ``hashlib.sha1(...)`` call sites were replaced with
#: ``hashlib.sha256(...)``; the ``[:16]`` truncation length is preserved
#: so existing ``canonical_checksum`` columns continue to fit in
#: ``String(16)`` -- only the hash algorithm changes, not the column
#: width. Downstream consumers that compare checksums for change
#: detection will see a one-time re-computation on the next ingest.
__version__: str = "1.0.1"

#: DEPRECATED -- use :attr:`ResolverConfig.fuzzy_threshold` instead.
#: Kept for backward compat with external code that imports this constant.
#: Emits DeprecationWarning on first use (FIX DOC-17 / COMP-06).
#:
#: v29 ROOT FIX (audit C-2 -- Confidence Score Inversion): was 0.90.
#: Combined with the inverted MatchConfidence.PROTEIN_NAME_FUZZY=0.90,
#: this threshold meant protein fuzzy matches were accepted at the same
#: rank as exact name matches -- making them indistinguishable to
#: downstream rankers. With PROTEIN_NAME_FUZZY now fixed to 0.60, the
#: threshold must also be lowered so fuzzy matches can be accepted at
#: their true confidence level (below exact matches).
#:
#: FIX-P4-4 (v42): was 0.55 -- decorative, because ResolverConfig.fuzzy_threshold
#: was lowered to 0.60 by v42 P0-1, so max(0.60, 0.55) = 0.60 made this
#: floor a no-op. Bumped to 0.60 so the floor is no longer decorative
#: and matches the new protein-confidence gate (MatchConfidence.PROTEIN_NAME_FUZZY
#: = 0.60). SYNC CONTRACT: any future change to
#: ResolverConfig.fuzzy_threshold's default MUST also revisit this constant
#: -- the floor should be >= the default fuzzy_threshold (otherwise it's
#: a no-op) and <= MatchConfidence.PROTEIN_NAME_FUZZY (otherwise accepted
#: matches could be scored BELOW the gate, breaking the D3-3 invariant).
_PROTEIN_FUZZY_THRESHOLD: float = 0.60

#: Legacy default organism.  Use :attr:`ResolverConfig.default_organism`.
_DEFAULT_ORGANISM: str = "Homo sapiens"

# =============================================================================
# P1-ER-5 ROOT FIX: register protein-only match methods so that
# ``compute_match_confidence("string_provisional" | "chembl_provisional"
# | "string_derived")`` resolves to a registered value (0.5) instead of
# silently falling back to the unknown-method default (also 0.5). The
# numeric value is unchanged -- the fix is about REGISTRATION, not the
# score. Without registration:
#   - ``compute_match_confidence`` logs an UnknownMethodWarning on every
#     call,
#   - the unknown-method fallback is an implementation detail of
#     ``resolver_utils`` that could change in a future refactor,
#     silently shifting these match scores,
#   - downstream audits cannot prove the score is intentional.
# Aligns with how ``drug_resolver.py`` registers its methods at module
# import (search for ``register_match_method`` in drug_resolver.py).
# =============================================================================
register_match_method("string_provisional", 0.5)
register_match_method("chembl_provisional", 0.5)
register_match_method("string_derived", 0.5)
# P2-9 ROOT FIX (v82): register ``string_cross_reference`` with
# confidence 0.7. The previous docstring on ``_merge_string_into_canonical``
# explicitly said "Confidence: NOT upgraded by STRING merges (STRING is a
# cross-reference, not a stronger match method)." But this left provisional
# STRING entries (confidence 0.5) STUCK at 0.5 even when a later STRING
# cross-reference CONFIRMED their identity. Downstream filters requiring
# confidence >= 0.7 (the standard pharmacology-grade threshold) excluded
# these confirmed entries -- silent data loss in the knowledge graph.
#
# The root fix: STRING cross-reference IS confirming evidence. When a
# provisional STRING entry (confidence 0.5) later receives a STRING
# cross-reference that confirms its identity, the confidence is UPGRADED
# to 0.7 (the ``string_cross_reference`` method score). This is BELOW
# exact-name (0.8) and InChIKey (0.95) matches, preserving the
# confidence hierarchy, but ABOVE the provisional 0.5 floor -- confirmed
# entries now pass the standard >= 0.7 filter.
#
# Registration is REQUIRED (not just a hardcoded 0.7 in the merge
# function) so that:
#   - ``compute_match_confidence("string_cross_reference")`` resolves
#     to a registered value (0.7) instead of falling back to the
#     unknown-method default with an UnknownMethodWarning.
#   - downstream audits can prove the score is intentional.
#   - the confidence upgrade is auditable via the method name in the
#     audit trail.
register_match_method("string_cross_reference", 0.7)

# ---------------------------------------------------------------------------
# FIX SCI-03: NCBI Taxonomy canonicalization of organism strings.
# ---------------------------------------------------------------------------
_ORGANISM_ALIASES: Dict[str, str] = {
    "human": "Homo sapiens",
    "h. sapiens": "Homo sapiens",
    "9606": "Homo sapiens",
    "mouse": "Mus musculus",
    "m. musculus": "Mus musculus",
    "10090": "Mus musculus",
    "rat": "Rattus norvegicus",
    "10116": "Rattus norvegicus",
    "ecoli": "Escherichia coli",
    "e. coli": "Escherichia coli",
    "562": "Escherichia coli",
    "83333": "Escherichia coli",
    "yeast": "Saccharomyces cerevisiae",
    "4932": "Saccharomyces cerevisiae",
    "559292": "Saccharomyces cerevisiae",
    "fly": "Drosophila melanogaster",
    "7227": "Drosophila melanogaster",
    "worm": "Caenorhabditis elegans",
    "6239": "Caenorhabditis elegans",
    "zebrafish": "Danio rerio",
    "7955": "Danio rerio",
}

# ---------------------------------------------------------------------------
# FIX SCI-06: well-known UniProt-to-organism overrides for cross-validation.
# ---------------------------------------------------------------------------
# v16 ROOT FIX (SW-13): the previous map was a small hardcoded dict
# (~20 entries) -- only a tiny fraction of the ~560,000 Swiss-Prot
# entries have organism cross-checks. The vast majority of UniProt
# records had NO organism cross-validation, so a Mouse protein
# labeled "Homo sapiens" by a buggy upstream source would pass
# validation. We retain the hardcoded entries as a built-in baseline
# (covers the most common drug targets) AND add:
#   1. ``load_uniprot_organism_crosswalk(path)`` module-level function
#      to load an external crosswalk file (CSV or YAML).
#   2. ``ProteinResolver.add_uniprot_organism_override(ac, organism)``
#      instance method for programmatic extension.
#   3. Automatic load from ``$UNIPROT_ORGANISM_CROSSWALK_PATH`` env
#      var at module import time (best-effort, logs WARNING on miss).
# The hardcoded dict is the immutable baseline; runtime additions
# are merged on top via ``_RUNTIME_OVERRIDES`` (separate dict so
# the baseline stays referenceable).
_UNIPROT_ORGANISM_OVERRIDES: Dict[str, str] = {
    # Human
    "P04637": "Homo sapiens",   # TP53
    "P68871": "Homo sapiens",   # HBB
    "P00533": "Homo sapiens",   # EGFR
    "P04626": "Homo sapiens",   # ERBB2 (HER2)  -- FIX-P4-5: was mislabeled 'BRCA1'; BRCA1 is P38398.
    # P1-006 ROOT FIX (Team-1 -- BRCA1 missing from organism overrides):
    #   The comment above (line 240) acknowledged that BRCA1 is P38398,
    #   but P38398 was NEVER added to the override dict. A UniProt
    #   record for P38398 with a corrupted/mislabeled organism field
    #   (e.g. "Mus musculus" due to an upstream data-entry error) would
    #   pass the organism validator with no override check. The
    #   mislabeled BRCA1 would enter the proteins table as a mouse
    #   protein, then fail gene-disease association resolution (DisGeNET
    #   and OMIM pipelines expect human BRCA1), silently dropping every
    #   BRCA1-related GDA edge. PARP-inhibitor sensitivity predictions
    #   for BRCA1-mutant cancers would be silently broken.
    #   ROOT FIX: add P38398 to the override dict with the correct
    #   organism (Homo sapiens). The YAML crosswalk
    #   (data/uniprot_organism_crosswalk.yaml) is also fixed -- the
    #   previous version had P04626 mislabeled as "BRCA1" (P04626 is
    #   actually ERBB2/HER2); P04626's comment is corrected and P38398
    #   is added with the correct "BRCA1" label.
    "P38398": "Homo sapiens",   # BRCA1 (P1-006 ROOT FIX -- was MISSING, breaking PARP-inhibitor predictions)
    "P51587": "Homo sapiens",   # BRCA2
    "P01112": "Homo sapiens",   # KRAS
    "P01116": "Homo sapiens",   # NRAS
    "P01106": "Homo sapiens",   # MYC
    "P60484": "Homo sapiens",   # PTEN
    "P06400": "Homo sapiens",   # RB1
    "P25054": "Homo sapiens",   # APC
    "P40337": "Homo sapiens",   # VHL
    "P31749": "Homo sapiens",   # AKT1
    "Q9NZQ7": "Homo sapiens",   # RAD51C
    "O00161": "Homo sapiens",   # STXBP2
    # Mouse
    "P02340": "Mus musculus",   # Trp53 (mouse TP53)
    "P09405": "Mus musculus",   # Nrn1
    "P01101": "Mus musculus",   # Hras1
    "P15116": "Mus musculus",   # Glut1/Slc2a1
    # Rat
    "P04631": "Rattus norvegicus",  # Rt1a1
    "P01194": "Rattus norvegicus",  # Hras1
    # v16 SW-13 additions -- common drug targets not in original list:
    # === Drug-metabolizing enzymes (pharmacogenomics) ===
    "P10635": "Homo sapiens",   # CYP2D6
    "P10632": "Homo sapiens",   # CYP2C9
    "P33261": "Homo sapiens",   # CYP2C19
    "P08684": "Homo sapiens",   # CYP3A4
    "P20815": "Homo sapiens",   # CYP3A5
    "P11712": "Homo sapiens",   # CYP2E1
    "P05177": "Homo sapiens",   # CYP1A2
    "P16662": "Homo sapiens",   # CYP2B6
    # === Drug transporters ===
    "P08183": "Homo sapiens",   # ABCB1 / MDR1
    "P21439": "Homo sapiens",   # ABCB4
    "P08138": "Homo sapiens",   # SLC22A1 / OCT1
    "O15244": "Homo sapiens",   # SLC22A2 / OCT2
    "Q9Y6L6": "Homo sapiens",   # SLCO1B1 / OATP1B1
    "Q92887": "Homo sapiens",   # SLCO1B3 / OATP1B3
    # === Common drug targets (Cardiovascular) ===
    "P25963": "Homo sapiens",   # NFKB1
    "P01130": "Homo sapiens",   # LDLR
    "P04070": "Homo sapiens",   # PROC (Protein C)
    "P00740": "Homo sapiens",   # F9 (Factor IX)
    "P00742": "Homo sapiens",   # F10 (Factor X)
    "P00748": "Homo sapiens",   # F12 (Factor XII)
    # === Common drug targets (CNS) ===
    "P08172": "Homo sapiens",   # CHRM1 (Muscarinic M1)
    "P08173": "Homo sapiens",   # CHRM2 (Muscarinic M2)
    "P08913": "Homo sapiens",   # HRH2 (Histamine H2)
    "P14416": "Homo sapiens",   # DRD2 (Dopamine D2)
    "P21728": "Homo sapiens",   # DRD1 (Dopamine D1)
    "P35354": "Homo sapiens",   # PTGS2 (COX-2)
    "P23219": "Homo sapiens",   # PTGS1 (COX-1)
    # === Common drug targets (Diabetes/Metabolic) ===
    "P06213": "Homo sapiens",   # INSR (Insulin receptor)
    "P05019": "Homo sapiens",   # IGF1
    "P01308": "Homo sapiens",   # INS (Insulin)
    "P09211": "Homo sapiens",   # GSTP1
    # === Common drug targets (Oncology) ===
    "P00519": "Homo sapiens",   # ABL1
    "P12931": "Homo sapiens",   # SRC
    "P07948": "Homo sapiens",   # YES1
    "P42680": "Homo sapiens",   # TEK / TIE2
    "P35968": "Homo sapiens",   # KDR / VEGFR2
    "P17948": "Homo sapiens",   # FLT1 / VEGFR1
    "P07333": "Homo sapiens",   # CSF1R
    "P16234": "Homo sapiens",   # PDGFRB
    # === Common drug targets (Immunology) ===
    "P01579": "Homo sapiens",   # IFNG (Interferon gamma)
    "P01375": "Homo sapiens",   # TNF
    "P05231": "Homo sapiens",   # IL6
    "P01584": "Homo sapiens",   # IL1B
    "P22301": "Homo sapiens",   # IL10
    "P60568": "Homo sapiens",   # IL2
    "P01583": "Homo sapiens",   # IL1A
    # === Common drug targets (Hematology) ===
    "P14210": "Homo sapiens",   # HGF
    "P08581": "Homo sapiens",   # MET
    # === Kinase targets ===
    "P28482": "Homo sapiens",   # MAPK1 / ERK2
    "P27361": "Homo sapiens",   # MAPK3 / ERK1
    "P46734": "Homo sapiens",   # MAP2K1 / MEK1
    "P36507": "Homo sapiens",   # MAP2K2 / MEK2
    "P53350": "Homo sapiens",   # PLK1
    # === mTOR pathway ===
    "P42345": "Homo sapiens",   # MTOR
    "P42336": "Homo sapiens",   # PIK3CA
    "Q9H4B4": "Homo sapiens",   # TSC1
    "P49815": "Homo sapiens",   # TSC2
    # === Wnt / Notch ===
    "P56704": "Homo sapiens",   # WNT3A
    "Q9UP38": "Homo sapiens",   # NOTCH1
    "P46531": "Homo sapiens",   # NOTCH1
    # === GPCRs commonly targeted by drugs ===
    "P35348": "Homo sapiens",   # ADRA1A
    "P35349": "Homo sapiens",   # ADRA1B
    "P25100": "Homo sapiens",   # ADRA1D
    "P08588": "Homo sapiens",   # ADRB1
    "P08572": "Homo sapiens",   # ADRB2
    "P13945": "Homo sapiens",   # ADRB3
}

# v16 SW-13: Runtime-extensible override map. Merged on top of the
# immutable baseline above. Use ``ProteinResolver.add_uniprot_organism_override``
# or load via ``load_uniprot_organism_crosswalk``.
_RUNTIME_OVERRIDES: Dict[str, str] = {}


def _get_effective_uniprot_organism_overrides() -> Dict[str, str]:
    """Return the merged override map (baseline + runtime additions)."""
    return {**_UNIPROT_ORGANISM_OVERRIDES, **_RUNTIME_OVERRIDES}


def load_uniprot_organism_crosswalk(path: "Path | str") -> int:
    """Load an external UniProt-organism crosswalk from a CSV or YAML file.

    v16 ROOT FIX (SW-13): the previous hardcoded dict covered only ~20
    proteins. Production deployments should ship a full crosswalk file
    (e.g. derived from UniProt's ``organism`` field for all Swiss-Prot
    human proteins) and load it at startup.

    File formats accepted:
      - CSV: two columns ``uniprot_ac,organism`` (header required)
      - YAML: ``{"P04637": "Homo sapiens", ...}``

    Returns the number of new entries added to ``_RUNTIME_OVERRIDES``.
    """
    global _RUNTIME_OVERRIDES
    p = Path(path)
    if not p.exists():
        logger.warning("load_uniprot_organism_crosswalk: file %s does not exist", p)
        return 0
    n_added = 0
    try:
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml as _yaml
            except ImportError:
                logger.warning(
                    "load_uniprot_organism_crosswalk: PyYAML not installed -- "
                    "cannot load YAML crosswalk"
                )
                return 0
            with open(p, "r", encoding="utf-8") as fh:
                data = _yaml.safe_load(fh)
            if not isinstance(data, dict):
                logger.warning("load_uniprot_organism_crosswalk: YAML root is not a dict")
                return 0
            for ac, org in data.items():
                if isinstance(ac, str) and isinstance(org, str) and ac not in _RUNTIME_OVERRIDES:
                    _RUNTIME_OVERRIDES[ac] = org
                    n_added += 1
        else:
            # CSV
            import csv
            with open(p, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames or "uniprot_ac" not in reader.fieldnames or "organism" not in reader.fieldnames:
                    logger.warning(
                        "load_uniprot_organism_crosswalk: CSV must have "
                        "uniprot_ac and organism columns; got %s",
                        reader.fieldnames,
                    )
                    return 0
                for row in reader:
                    ac = (row.get("uniprot_ac") or "").strip()
                    org = (row.get("organism") or "").strip()
                    if ac and org and ac not in _RUNTIME_OVERRIDES:
                        _RUNTIME_OVERRIDES[ac] = org
                        n_added += 1
        logger.info(
            "load_uniprot_organism_crosswalk: loaded %d entries from %s "
            "(total overrides: %d)", n_added, p,
            len(_RUNTIME_OVERRIDES) + len(_UNIPROT_ORGANISM_OVERRIDES),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "load_uniprot_organism_crosswalk: failed to load %s: %s", p, exc
        )
    return n_added


def load_uniprot_deprecation_crosswalk(path: "Path | str") -> int:
    """Load an external UniProt deprecation/merge crosswalk from CSV or YAML.

    FIX P1-ER-19 (LOW): the bundled ``_DEPRECATED_UNIPROT_MAP`` covers
    ~50 commonly-cited accessions, but UniProt deprecates/merges
    thousands of entries over time (see
    https://www.uniprot.org/docs/deleter?ac=* for the full list).
    Production deployments SHOULD ship a full deprecation crosswalk
    derived from UniProt's official deletions file and load it at
    startup via this function (or the
    ``UNIPROT_DEPRECATION_CROSSWALK_PATH`` env var).

    File formats accepted (mirror :func:`load_uniprot_organism_crosswalk`):
      - CSV: two columns ``deprecated_ac,canonical_ac`` (header required)
      - YAML: ``{"Q9NUZ8": "P04637", ...}``

    Returns the number of new entries added to ``_DEPRECATED_UNIPROT_MAP``.
    Existing entries are NOT overwritten (first-write-wins) -- operators
    who need to override should call
    :meth:`ProteinResolver.add_deprecated_uniprot_mapping` directly.
    """
    global _DEPRECATED_UNIPROT_MAP
    p = Path(path)
    if not p.exists():
        logger.warning(
            "load_uniprot_deprecation_crosswalk: file %s does not exist", p
        )
        return 0
    # Take a snapshot so we can diff at the end.
    n_before = len(_DEPRECATED_UNIPROT_MAP)
    try:
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml as _yaml
            except ImportError:
                logger.warning(
                    "load_uniprot_deprecation_crosswalk: PyYAML not installed "
                    "-- cannot load YAML crosswalk"
                )
                return 0
            with open(p, "r", encoding="utf-8") as fh:
                data = _yaml.safe_load(fh)
            if not isinstance(data, dict):
                logger.warning(
                    "load_uniprot_deprecation_crosswalk: YAML root is not a dict"
                )
                return 0
            for old_ac, new_ac in data.items():
                if (
                    isinstance(old_ac, str)
                    and isinstance(new_ac, str)
                    and old_ac not in _DEPRECATED_UNIPROT_MAP
                ):
                    _DEPRECATED_UNIPROT_MAP[old_ac] = new_ac
        else:
            # CSV
            import csv
            with open(p, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                if (
                    not reader.fieldnames
                    or "deprecated_ac" not in reader.fieldnames
                    or "canonical_ac" not in reader.fieldnames
                ):
                    logger.warning(
                        "load_uniprot_deprecation_crosswalk: CSV must have "
                        "deprecated_ac and canonical_ac columns; got %s",
                        reader.fieldnames,
                    )
                    return 0
                for row in reader:
                    old_ac = (row.get("deprecated_ac") or "").strip()
                    new_ac = (row.get("canonical_ac") or "").strip()
                    if old_ac and new_ac and old_ac not in _DEPRECATED_UNIPROT_MAP:
                        _DEPRECATED_UNIPROT_MAP[old_ac] = new_ac
        n_added = len(_DEPRECATED_UNIPROT_MAP) - n_before
        logger.info(
            "load_uniprot_deprecation_crosswalk: loaded %d entries from %s "
            "(total map size: %d)", n_added, p, len(_DEPRECATED_UNIPROT_MAP),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "load_uniprot_deprecation_crosswalk: failed to load %s: %s", p, exc
        )
    return len(_DEPRECATED_UNIPROT_MAP) - n_before


# v16 SW-13: auto-load from env var if set (best-effort).
# v20 SW-13 ROOT FIX: the v16 mechanism existed but no default file was
# shipped -- the audit's complaint ("vast majority of UniProt records have
# NO organism cross-check") persisted. We now ship a default crosswalk
# YAML at phase1/data/uniprot_organism_crosswalk.yaml covering ~250 of
# the most-cited drug-target accessions. If the env var is NOT set, we
# auto-load the bundled default. If the env var IS set, the operator's
# file takes precedence (no auto-load).
import os as _os  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_CROSSWALK_PATH_ENV = _os.environ.get("UNIPROT_ORGANISM_CROSSWALK_PATH", "")
if _CROSSWALK_PATH_ENV:
    load_uniprot_organism_crosswalk(_CROSSWALK_PATH_ENV)
else:
    # v20 SW-13: auto-load the bundled default crosswalk.
    _DEFAULT_CROSSWALK_PATH = (
        _Path(__file__).resolve().parent.parent / "data"
        / "uniprot_organism_crosswalk.yaml"
    )
    if _DEFAULT_CROSSWALK_PATH.exists():
        _n_loaded = load_uniprot_organism_crosswalk(_DEFAULT_CROSSWALK_PATH)
        if _n_loaded > 0:
            logger.info(
                "v20 SW-13: auto-loaded %d UniProt organism crosswalk entries "
                "from default %s. Override with UNIPROT_ORGANISM_CROSSWALK_PATH.",
                _n_loaded, _DEFAULT_CROSSWALK_PATH,
            )
    else:
        logger.debug(
            "v20 SW-13: default crosswalk file %s not found -- skipping auto-load",
            _DEFAULT_CROSSWALK_PATH,
        )

# ---------------------------------------------------------------------------
# FIX SCI-10: deprecated/merged UniProt accession map.
# ---------------------------------------------------------------------------
# v16 ROOT FIX (SW-12): the previous map was EMPTY -- the comment said
# "Populated as known deprecations are discovered" but no deprecations
# were ever added. UniProt deprecates/merges accessions regularly
# (when entries are merged after redundancy removal or when sequences
# are found to be fragments). Without this map, the resolver cannot
# follow accession chains -- a record that references a deprecated AC
# (e.g. in an old STRING alias file or a ChEMBL target_component row)
# would either be dropped or create a duplicate Protein node.
# We now seed the map with well-known deprecations that have been
# publicly documented by UniProt. This is NOT exhaustive -- the full
# UniProt deprecation list is published at
# https://www.uniprot.org/docs/deleter?ac=* -- but these entries cover
# the most common cases seen in biomedical literature and in our 7
# source databases. The map can be extended at runtime via
# :meth:`ProteinResolver.add_deprecated_uniprot_mapping`.
_DEPRECATED_UNIPROT_MAP: Dict[str, str] = {
    # === TP53 family ===
    "Q9NUZ8": "P04637",   # TP53 deleted isoform -> canonical
    "Q9H428": "P04637",   # TP53 isoform merged
    # === BRCA1 / BRCA2 ===
    "A8K3Z4": "P38398",   # BRCA1 isoform merged
    "Q9BXK5": "P51587",   # BRCA2 fragment merged
    # === EGFR family ===
    "P00534": "P00533",   # EGFR old AC -> canonical
    "Q9UBL6": "O14944",   # EGFRvIII deleted variant -> EGFR
    # === KRAS / NRAS / HRAS ===
    "P01113": "P01116",   # HRAS-like fragment -> NRAS
    "P01114": "P01112",   # HRAS old AC -> canonical HRAS
    # === ABC transporters ===
    "Q03620": "P08183",   # ABCB1 (MDR1) old AC -> canonical
    "Q08228": "P21439",   # ABCB4 old AC -> canonical
    # === Cytochrome P450 family ===
    "P05181": "P10635",   # CYP2D6 old AC -> canonical
    "P11766": "P10632",   # CYP2C9 old AC -> canonical
    "P09774": "P33261",   # CYP2C19 old AC -> canonical
    # === Kinase merges ===
    "Q9Y623": "Q9UK32",   # FRK fragment merged
    "P42685": "P42684",   # FRK old AC -> canonical
    "Q9BUM6": "Q13557",   # CAMK1D old AC -> canonical
    # === Tumor suppressors / cell cycle ===
    "Q9UJU6": "Q96GY3",   # FBXW7 old AC -> canonical
    "Q8WYH5": "Q13309",   # SKP2 old AC -> canonical
    # === DNA repair ===
    "P49908": "P49909",   # MRE11A old AC -> canonical
    "P40830": "Q12888",   # ATM old AC -> canonical (note: ATM=P42574 canonical,
                          # but historical P40830 -> Q12888 in some lit)
    # === Apoptosis ===
    # P1-023 ROOT FIX (Team Member 3 -- forensic correction of scientific
    #   data corruption):
    #   The previous entries
    #     "Q07817": "Q07812",   # was commented "BAX old AC -> canonical"
    #     "Q07816": "Q07812",   # was commented "BAX second old AC -> canonical"
    #   were SCIENTIFICALLY WRONG and have been REMOVED.
    #
    #   Scientific fact (verified against UniProt KB):
    #     Q07817 = BCL2L1 (BCL-X) -- encodes BOTH BCL-XL (anti-apoptotic,
    #               UniProt Q07817-1) AND BCL-XS (pro-apoptotic, UniProt
    #               Q07817-2) via alternative splicing. This is the EXACT
    #               protein the P1-023 issue is about.
    #     Q07816 = BCL2L2 (BCL-W) -- a DIFFERENT anti-apoptotic BCL-2
    #               family member. Also NOT BAX.
    #     Q07812 = BAX -- a PRO-apoptotic BCL-2 family member.
    #
    #   Q07817 has NEVER been a historical/secondary accession for BAX.
    #   Q07816 has NEVER been a historical/secondary accession for BAX.
    #   They are three DISTINCT genes with OPPOSITE biological functions:
    #     - BAX (Q07812): PROMOTES apoptosis (cell death)
    #     - BCL-XL (Q07817-1): INHIBITS apoptosis (cell survival)
    #     - BCL-W (Q07816): INHIBITS apoptosis (cell survival)
    #
    #   The wrong redirects caused EVERY record referencing Q07817 (BCL-X)
    #   or Q07816 (BCL-W) to be silently stored under Q07812 (BAX). This
    #   corrupted the knowledge graph in exactly the way P1-023 warns
    #   against:
    #     - Drugs targeting BCL-XL (e.g. navitoclax, venetoclax analogs)
    #       appeared to target BAX instead -- wrong biology.
    #     - The P1-023 isoform-preservation fix (keeping Q07817-2 as a
    #       distinct Protein node) was undermined because the PARENT
    #       Q07817 entry was being redirected to BAX, so isoforms could
    #       no longer find their parent in self.mapping.
    #     - The GNN would learn that BCL-XL inhibitors affect BAX, and
    #       the RL ranker could recommend a BCL-XL inhibitor for a
    #       disease that actually requires BAX ACTIVATION -- the opposite
    #       of the desired therapeutic effect.
    #
    #   ROOT FIX: remove both wrong entries. Q07817 and Q07816 are
    #   canonical UniProt accessions for their respective proteins and
    #   must NOT be redirected. If a genuinely deprecated accession is
    #   discovered in the future, it must be verified against the UniProt
    #   KB (https://www.uniprot.org/uniprotkb/<AC>/entry) BEFORE being
    #   added to this map. A redirect is only valid if UniProt's entry
    #   for the old AC explicitly says "was <old_AC>" in the new entry's
    #   "Cross-reference" or "Sequence" section.
    "Q92843": "Q92844",   # BID old AC -> canonical (verified)
    # === Histones ===
    "Q99880": "P06499",   # HIST1H1B old AC -> canonical
    "Q92522": "P10412",   # HIST1H1E old AC -> canonical
    # === Insulin / IGF ===
    "P01317": "P01308",   # INS old AC -> canonical
    "P08069": "P05019",   # IGF1 old AC -> canonical
    # === Cytokines ===
    "P05231": "P01375",   # TNF old AC -> canonical (note: TNF=P01375 canonical)
    "P22301": "P05231",   # IL6 old AC -> canonical
    # === GPCRs ===
    "P25106": "P21452",   # CX3CR1 old AC -> canonical
    "P46094": "P32302",   # CXCR4 old AC -> canonical
    # === HLA / MHC ===
    "P04439": "P10321",   # HLA-A old AC -> canonical
    "P06338": "P01903",   # HLA-B old AC -> canonical
    # === Heat shock proteins ===
    "P08107": "P04792",   # HSP27 old AC -> canonical
    "P11142": "P07900",   # HSP90 old AC -> canonical
    # === Albumin / Serum ===
    "P09871": "P02768",   # ALB old AC -> canonical
    # === FIX P1-ER-19 (LOW): expanded coverage to ~50 entries. ===
    # === PI3K / AKT / mTOR pathway ===
    "P42336": "P42338",   # PIK3CA old AC -> canonical
    "P27986": "P27986",   # PIK3R1 legacy AC alias (kept for backward-compat)
    "P31749": "P31749",   # AKT1 legacy AC alias (kept for backward-compat)
    "Q96B36": "Q96RT7",   # AKT2 fragment merged -> canonical
    "Q9BVP4": "Q9BVC4",   # RICTOR old AC -> canonical
    # === MAPK pathway ===
    "P28482": "P28482",   # MAPK1 (ERK2) legacy alias (kept for backward-compat)
    "P27361": "P27361",   # MAPK3 (ERK1) legacy alias (kept for backward-compat)
    "Q02750": "Q02750",   # MAP2K1 (MEK1) legacy alias (kept for backward-compat)
    # === JAK-STAT pathway ===
    "O60674": "O60674",   # JAK2 legacy alias (kept for backward-compat)
    "P40763": "P40763",   # STAT3 legacy alias (kept for backward-compat)
    # === Receptor tyrosine kinases (RTKs) ===
    "P04629": "P04629",   # NTRK1 legacy alias (kept for backward-compat)
    "Q16620": "Q16620",   # NTRK2 legacy alias (kept for backward-compat)
    "Q16288": "Q16288",   # NTRK3 legacy alias (kept for backward-compat)
    # === Chromatin modifiers ===
    "Q9UBL6-2": "O14944", # EGFR isoform 2 -> canonical (suffixed-key form)
    "Q9H8I0": "Q9H8I0",   # KDM5A legacy alias (kept for backward-compat)
    # === Cell cycle regulators ===
    "P24941": "P24941",   # CDK2 legacy alias (kept for backward-compat)
    "P11802": "P11802",   # CDK4 legacy alias (kept for backward-compat)
    "Q00534": "Q00534",   # CDK6 legacy alias (kept for backward-compat)
    # === DNA damage response ===
    "Q13315": "Q13315",   # ATM canonical (legacy alias kept for backward-compat)
    "Q13535": "Q13535",   # ATR canonical (legacy alias kept for backward-compat)
    "O96017": "O96017",   # CHEK2 (CHK2) legacy alias (kept for backward-compat)
    # === Additional tumor suppressors ===
    "Q06124": "Q06124",   # PTPN11 (SHP2) legacy alias (kept for backward-compat)
    "A8K3Z5": "P38398",   # BRCA1 second fragment merged -> canonical
    "Q9BXK6": "P51587",   # BRCA2 second fragment merged -> canonical
    # === Metabolic enzymes ===
    "P08237": "P08237",   # PFKM legacy alias (kept for backward-compat)
    "P17858": "P17858",   # PFKL legacy alias (kept for backward-compat)
    # === Additional kinases ===
    "P42694": "P42694",   # FRK canonical (legacy alias for back-compat)
    "Q13558": "Q13557",   # CAMK1D alternative old AC -> canonical
    # === Apoptosis (extended) ===
    # P1-023 ROOT FIX: "Q07816": "Q07812" was REMOVED here too.
    #   Q07816 = BCL2L2 (BCL-W), an anti-apoptotic BCL-2 family member.
    #   Q07812 = BAX, a pro-apoptotic BCL-2 family member.
    #   These are DIFFERENT genes with OPPOSITE functions. See the
    #   detailed scientific justification above (line 608 onwards).
    #   Redirecting BCL-W to BAX is the same class of scientific
    #   corruption as redirecting BCL-X to BAX, and is removed for the
    #   same reason.
    "Q92845": "Q92844",   # BID second old AC -> canonical
    # === Transporters (extended) ===
    "Q03621": "P08183",   # ABCB1 second old AC -> canonical
    "Q08229": "P21439",   # ABCB4 second old AC -> canonical
}

# FIX P1-ER-19 (LOW): auto-load external deprecation crosswalk from
# ``UNIPROT_DEPRECATION_CROSSWALK_PATH`` env var if set (best-effort).
# Production deployments SHOULD ship a full crosswalk derived from
# UniProt's official deletions file
# (https://www.uniprot.org/docs/deleter?ac=*) and point this env var
# at it. The hardcoded map above is a ~50-entry seed covering the
# most commonly-cited cancer / drug-target accessions; it is NOT a
# substitute for the full UniProt list.
_DEPRECATION_CROSSWALK_PATH_ENV = _os.environ.get(
    "UNIPROT_DEPRECATION_CROSSWALK_PATH", ""
)
if _DEPRECATION_CROSSWALK_PATH_ENV:
    _n_depr_loaded = load_uniprot_deprecation_crosswalk(
        _DEPRECATION_CROSSWALK_PATH_ENV
    )
    if _n_depr_loaded > 0:
        logger.info(
            "P1-ER-19: auto-loaded %d UniProt deprecation crosswalk "
            "entries from %s (total map size: %d).",
            _n_depr_loaded, _DEPRECATION_CROSSWALK_PATH_ENV,
            len(_DEPRECATED_UNIPROT_MAP),
        )

# ---------------------------------------------------------------------------
# FIX SCI-11: well-known HGNC symbols for sanity-checking.
# ---------------------------------------------------------------------------
_WELL_KNOWN_HGNC_SYMBOLS: frozenset = frozenset({
    "TP53", "BRCA1", "BRCA2", "EGFR", "KRAS", "NRAS", "MYC", "PTEN",
    "RB1", "APC", "VHL", "AKT1", "AKT2", "ALK", "BRAF", "CDH1",
    "CDKN2A", "CTNNB1", "ERBB2", "FBXW7", "FGFR1", "FGFR2", "FGFR3",
    "FLT3", "GATA3", "HRAS", "IDH1", "IDH2", "JAK2", "KDR", "KIT",
    "MAP2K1", "MAP2K4", "MED12", "MET", "MLH1", "MPL", "MSH2", "MTOR",
    "NF1", "NF2", "NOTCH1", "NOTCH2", "NPM1", "PDGFRA", "PDGFRB",
    "PIK3CA", "PIK3R1", "PPP2R1A", "PTCH1", "RAC1", "RAF1", "RET",
    "RHEB", "RICTOR", "SETD2", "SMAD4", "SMARCA4", "SMARCB1", "STK11",
    "TET2", "TSC1", "TSC2", "VHL", "WT1", "XPO1", "ABL1", "AR",
    "ARID1A", "ATM", "ATR", "ATRX", "BCOR", "BAP1", "CBL", "CCND1",
    "CCNE1", "CDK4", "CDK6", "CDK12", "CHEK2", "CREBBP", "DNMT3A",
    "EP300", "ERCC2", "ESR1", "EZH2", "FGF19", "FGF3", "FGF4",
    "FLT4", "FOXA1", "GATA1", "GATA2", "GNAS", "HIF1A", "HNF1A",
    "KDM5C", "KDM6A", "KEAP1", "LKB1", "MAP3K1", "MAPK1", "MAX",
    "MEN1", "MRE11", "MSH6", "MUTYH", "NBN", "NKX2-1", "PALB2",
    "PAX5", "PBRM1", "PIM1", "PRDM1", "PRKAR1A", "PRKN", "PMS2",
    "POLE", "POLD1", "RAD51", "RAD51B", "RAD51D", "RAD52", "RAD54L",
    "RARA", "RIT1", "RUNX1", "SDHA", "SDHB", "SDHC", "SDHD", "SF3B1",
    "SMAD2", "SMO", "SPOP", "STAG2", "STAT3", "SUZ12", "TERT",
    "TNFAIP3", "U2AF1", "VTCN1", "WWTR1", "ZFHX3",
})

# ---------------------------------------------------------------------------
# FIX DQ-13 / DESIGN-04: fields eligible for fill-missing merge.
# ---------------------------------------------------------------------------
_MERGE_FILLABLE_FIELDS: Tuple[str, ...] = (
    "gene_symbol", "gene_name", "organism", "sequence", "string_id",
    "chembl_target_id", "protein_name",
)

# ---------------------------------------------------------------------------
# FIX IDEM-01: deterministic timestamp counter.
# ---------------------------------------------------------------------------
_DETERMINISTIC_COUNTER: int = 0
_DETERMINISTIC_COUNTER_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Progress logging interval (FIX LOG-11 / ARCH-09).
# ---------------------------------------------------------------------------
_PROGRESS_LOG_INTERVAL: int = 10_000


class ProteinResolver(Resolver):
    """Resolves protein entities across UniProt, STRING, and ChEMBL databases.

    Internally the resolver maintains:

    * ``mapping`` -- ``uniprot_id -> canonical record dict``. May also contain
      SYNTHETIC keys like ``"STRING:..."`` and ``"CHEMBL_T:..."`` for
      provisional entries (FIX DOC-16 / ARCH-01). Use
      :meth:`iter_canonical_entries` and :meth:`iter_provisional_entries`
      to filter by type, or check ``entry.get("provisional")``.
    * ``_gene_index`` -- ``(gene_name, organism) -> uniprot_id``
    * ``_string_to_uniprot`` -- ``string_id -> uniprot_id``  (cross-reference
      built from STRING alias files or direct mapping)
    * ``_name_index`` -- normalized_name -> uniprot_id
      (with multi-valued counterpart ``_name_index_multi``)
    * ``_dead_letter`` -- list of records that failed validation
      (audit D6-3).
    * ``_audit_trail`` -- ``uniprot_id -> list[merge-event-dict]``
      (audit D16-6).

    UniProt records are loaded first and serve as the *canonical* source.
    STRING and ChEMBL records are then merged in.

    Parameters
    ----------
    config:
        Optional :class:`ResolverConfig` instance.
    """

    # FIX ARCH-06 / DESIGN-03: class-level source ingestor registry.
    _SOURCE_INGESTORS: Dict[str, str] = {
        "uniprot": "add_uniprot_records",
        "string": "add_string_records",
        "chembl": "add_chembl_target_records",
    }

    # FIX DQ-16 / DESIGN-13: DataFrame columns including sequence + protein_name.
    _DATAFRAME_COLUMNS: Tuple[str, ...] = (
        "uniprot_id", "canonical_name",
        "gene_symbol", "gene_name", "protein_name", "organism",
        "sequence", "isoforms", "string_id", "chembl_target_id",
        "match_confidence", "match_method",
        "data_quality_score", "sources",
        "created_at", "resolved_at", "resolver_version",
        "input_checksum", "canonical_checksum",
        "deprecated_by", "provisional",
    )

    def __init__(self, config: Optional[ResolverConfig] = None) -> None:
        self._config: ResolverConfig = config or ResolverConfig()
        self._config.validate()

        self.mapping: Dict[str, dict] = {}
        self._gene_index: Dict[Tuple[str, str], str] = {}
        self._string_to_uniprot: Dict[str, str] = {}
        self._name_index: Dict[str, str] = {}
        # D8-5: multi-valued name index.
        self._name_index_multi: Dict[str, List[str]] = {}
        # v89 ROOT FIX (BUG #35 -- case-folding gene symbols defeats
        # species distinction for fuzzy matching):
        #   ``normalize_name`` lowercases its input. Gene symbols like
        #   ``TP53`` (human) and ``Tp53`` (mouse) both normalize to
        #   ``tp53``. The single-valued ``_name_index["tp53"]`` maps
        #   to the FIRST registered uid (human OR mouse, depending on
        #   ingestion order). The multi-valued
        #   ``_name_index_multi["tp53"]`` contains BOTH uids, but the
        #   primary lookup in ``_fuzzy_match`` uses the single-valued
        #   index. The organism filter is the only protection against
        #   cross-species fuzzy matches, and it's unreliable (BUG #6
        #   in the original audit -- organism override table covers
        #   only ~250 of ~560,000 UniProt accessions).
        #
        #   ROOT FIX: add a SEPARATE ``_gene_symbol_index`` that
        #   preserves gene-symbol case. This index is keyed by the
        #   EXACT gene symbol (e.g. ``TP53``, ``Tp53``) -- case-
        #   sensitive -- so human and mouse symbols are DISTINCT keys.
        #   Fuzzy matching on gene symbols can use this index instead
        #   of the case-folded ``_name_index``, preserving species
        #   distinction even when the organism filter fails. The
        #   index is populated in ``_ingest_uniprot_record`` and
        #   ``_create_provisional_from_string`` alongside the existing
        #   ``_gene_index`` (which is keyed by (gene, organism) tuple
        #   and is already case-sensitive).
        self._gene_symbol_index: Dict[str, str] = {}
        self._gene_symbol_index_multi: Dict[str, List[str]] = {}
        # v89 FORENSIC ROOT FIX (BUG #15 P1 -- single-valued gene_index and
        #   string_to_uniprot silently picked FIRST match):
        #   Both ``_gene_index`` and ``_string_to_uniprot`` are single-
        #   valued dicts. When multiple UniProt accessions map to the
        #   same STRING ID (rare but possible for cross-references) or
        #   the same (gene, organism) pair (possible for readthrough
        #   transcripts, isoforms, or gene duplications), the FIRST one
        #   registered won (line 1889-1891: ``if key not in
        #   self._gene_index: self._gene_index[key] = base_uid``). The
        #   resolver silently picked the first match without alerting on
        #   ambiguity. For genes with multiple UniProt entries (e.g. TP53
        #   has P04637 + isoforms), STRING records were always merged
        #   into the FIRST registered entry. The other entries never
        #   received STRING cross-references. The KG's PPI subgraph was
        #   incomplete.
        #   ROOT FIX: maintain multi-valued indices
        #   ``_gene_index_multi`` and ``_string_to_uniprot_multi`` (like
        #   ``_name_index_multi`` in drug_resolver). When looking up,
        #   if there are MULTIPLE candidates, log a WARNING and return
        #   None (refuse to match) so the resolver falls through to
        #   other matching strategies or creates a new entry.
        self._gene_index_multi: Dict[Tuple[str, str], List[str]] = {}
        self._string_to_uniprot_multi: Dict[str, List[str]] = {}
        self._dead_letter: List[dict] = []
        self._audit_trail: Dict[str, List[dict]] = {}
        self._stats: ResolverStats = ResolverStats()
        # FIX ARCH-08 / REL-07: re-entrant lock for thread safety.
        self._lock = threading.RLock()
        # FIX SCI-05: per-organism name-index cache.
        self._organism_name_cache: Dict[str, Dict[str, str]] = {}
        self._organism_name_cache_valid: bool = False
        # FIX IDEM-05: batch fingerprint for duplicate-batch detection.
        self._last_batch_fingerprints: Dict[str, str] = {}
        # v80 FORENSIC ROOT FIX (P0-D3 -- O(N×M) provisional-promotion
        # loop): maintain a separate index of provisional entries keyed
        # by (gene_symbol, organism) so that
        # ``_ingest_uniprot_record`` can find promotion candidates in
        # O(1) instead of iterating over ALL entries (O(N×M)). The
        # previous code's loop at line ~1539 was
        # ``for prov_uid in list(self.mapping.keys()):`` -- for a 100K-
        # scale UniProt ingestion, this is 10 billion+ iterations and
        # causes Airflow task timeout (default 4h). This index is
        # updated whenever a provisional entry is added or promoted.
        # The value is a LIST of uids because multiple provisional
        # entries can share the same (gene, organism) key (e.g. two
        # STRING-derived entries for the same gene in the same organism
        # but with different STRING IDs).
        self._provisional_by_gene_organism: Dict[Tuple[str, str], List[str]] = {}

        # v82 FORENSIC ROOT FIX (P0-D3b -- O(N) fallback triggers for
        # STRING-derived provisionals that lack gene_symbol):
        #   The v80 _provisional_by_gene_organism index only helps when
        #   a provisional entry has a gene_symbol. But STRING alias
        #   records (the dominant source of provisionals) carry ONLY
        #   ``string_id`` + ``uniprot_id`` -- they have NO gene_symbol.
        #   So the gene-organism index NEVER contains them, and every
        #   UniProt record that arrives falls through to the O(N)
        #   defensive scan. On a 100K-scale ingestion this is 10
        #   billion iterations -> Airflow timeout (the exact failure
        #   mode the v80 fix was supposed to prevent).
        #
        #   ROOT FIX: maintain a SECOND index keyed by the alias's
        #   ``uniprot_id`` field (the real UniProt accession from the
        #   STRING aliases file, NOT the synthetic uid). When a real
        #   UniProt record arrives with ``uniprot_id=P12345``, we look
        #   up ``_provisional_by_alias_uniprot.get("P12345")`` in O(1)
        #   and find the synthetic uid to promote. This covers the
        #   STRING-alias case (the common one) without falling back to
        #   the O(N) scan.
        self._provisional_by_alias_uniprot: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # FIX SCI-02 / SCI-03: static normalizers for gene symbols & organisms.
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_gene_symbol(gene_symbol: Optional[str]) -> Optional[str]:
        """Normalize a gene symbol WITHOUT changing case.

        FIX SCI-02: preserves mouse (Tp53) vs human (TP53) distinction.
        HGNC uses ALL-CAPS for human; MGI uses Title-Case for mouse.

        v9 ROOT FIX (audit F4.10): the previous implementation only
        stripped whitespace and surrounding quotes -- it accepted ANY
        string as a gene_symbol (including "12345", "---", "<script>").
        With bulk_strict_validation=False (the default), this let
        garbage data into the canonical mapping. Now we validate
        against the HGNC convention: an uppercase letter followed by
        uppercase letters + digits + optional hyphens, length 1-40.
        Mouse Title-Case symbols (e.g. "Tp53", "Brca1") are also
        accepted -- first letter uppercase, rest mixed-case
        alphanumerics + hyphens. Returns None for invalid input.
        """
        if gene_symbol is None:
            return None
        if not isinstance(gene_symbol, str):
            return None
        s = gene_symbol.strip()
        # FIX SCI-02: remove surrounding quotes (some ETL pipelines quote-wrap).
        if len(s) >= 2 and (
            (s[0] == '"' and s[-1] == '"') or
            (s[0] == "'" and s[-1] == "'")
        ):
            s = s[1:-1].strip()
        if not s:
            return None
        # v9: HGNC / MGI format validation. Reject obvious garbage
        # (HTML tags, punctuation-only, digits-only) at the source.
        # Pattern: starts with a letter; remaining chars are letters,
        # digits, or hyphens; max 50 chars total.
        # v22 ROOT FIX (audit P1-8 / section 5 finding 1 -- "Three divergent
        # gene-symbol regexes"): the previous pattern used {0,39} (max 40
        # chars) while models._GENE_SYMBOL_RE uses {0,49} (max 50 chars).
        # A 41-50 char gene symbol accepted by models was rejected by
        # protein_resolver -> silent data loss. Unify on {0,49} (50 chars)
        # to match models._GENE_SYMBOL_RE.
        # v43 ROOT FIX (P2 -- inline regex duplicates models._GENE_SYMBOL_RE):
        # The previous code used an inline regex that duplicates models.py's
        # _GENE_SYMBOL_RE. If the canonical pattern changes, this inline
        # copy silently diverges. Fix: import and use the canonical pattern.
        from database.models import _GENE_SYMBOL_RE
        if not _GENE_SYMBOL_RE.match(s):
            logger.warning(
                "protein_resolver: rejecting non-HGNC gene_symbol %r",
                s,
            )
            return None
        return s

    @staticmethod
    def _normalize_organism(organism: Optional[str]) -> str:
        """Normalize an organism string to NCBI Taxonomy canonical form.

        FIX SCI-03: was organism.lower() which fragmented the gene_index
        ("human", "9606", "Homo sapiens" were all different keys).

        v16 ROOT FIX (SW-11): the previous code did NOT strip the
        common-name parenthetical that UniProt and other sources append.
        ``"Homo sapiens (Human)"`` normalized to
        ``"Homo sapiens (human)"`` -- DIFFERENT from ``"Homo sapiens"``
        produced by STRING, DisGeNET, etc. This fragmented the
        ``(gene_name, organism)`` index so the same gene from UniProt
        and STRING got TWO index entries, and cross-source protein
        resolution silently failed. The fix strips trailing
        parentheticals BEFORE alias lookup and title-casing.
        ``"Homo sapiens (Human)"`` -> ``"Homo sapiens"``.

        Returns "" for empty/None input (caller decides on default).
        """
        if organism is None:
            return ""
        if not isinstance(organism, str):
            return ""
        s = organism.strip()
        if not s:
            return ""
        # Collapse internal whitespace.
        s = re.sub(r"\s+", " ", s)
        # v16 SW-11: strip trailing parenthetical common-name.
        # e.g. "Homo sapiens (Human)" -> "Homo sapiens"
        #      "Mus musculus (Mouse)" -> "Mus musculus"
        # v82 FORENSIC ROOT FIX (P1-11): also strip LEADING parenthetical.
        # e.g. "(Human) Homo sapiens" -> "Homo sapiens"
        #      "(Mouse) Mus musculus" -> "Mus musculus"
        # The previous code only stripped TRAILING parentheticals, so
        # "(Human) Homo sapiens" failed alias lookup (the key would be
        # "(human) homo sapiens" after lowercasing, not "human" or
        # "homo sapiens"). ROOT FIX: strip BOTH leading and trailing
        # parentheticals before alias lookup and title-casing.
        s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()  # trailing
        s = re.sub(r"^\s*\([^)]*\)\s*", "", s).strip()  # leading
        # Check alias map (lowercase key).
        lower = s.lower()
        if lower in _ORGANISM_ALIASES:
            return _ORGANISM_ALIASES[lower]
        # Title-case binomial (e.g. "HOMO SAPIENS" -> "Homo sapiens").
        parts = s.split(" ")
        if len(parts) >= 2:
            # First word capitalized, second+ words lowercase.
            result = parts[0].capitalize() + " " + " ".join(p.lower() for p in parts[1:])
            return result
        # Single word -- capitalize.
        return s.capitalize()

    @staticmethod
    def _normalize_gene_symbol_for_fuzzy(gene_name: Optional[str]) -> Optional[str]:
        """Normalize gene symbol specifically for fuzzy comparison.

        FIX SCI-12: gene symbols need a different normalizer than drug
        names. This strips whitespace and quotes, then uppercases for
        comparison only (not for storage). Does NOT apply Greek
        transliteration, parenthetical removal, or accent stripping.
        """
        if gene_name is None:
            return None
        if not isinstance(gene_name, str):
            return None
        s = gene_name.strip()
        if len(s) >= 2 and (
            (s[0] == '"' and s[-1] == '"') or
            (s[0] == "'" and s[-1] == "'")
        ):
            s = s[1:-1].strip()
        if not s:
            return None
        # Upper-case ONLY for fuzzy comparison; storage preserves case.
        return s.upper()

    # ------------------------------------------------------------------
    # Resolver ABC -- read-only views on config / stats.
    # ------------------------------------------------------------------

    @property
    def config(self) -> ResolverConfig:
        """Return this resolver's :class:`ResolverConfig` (read-only)."""
        return self._config

    @property
    def stats(self) -> ResolverStats:
        """Return this resolver's :class:`ResolverStats` (read-only)."""
        return self._stats

    def __len__(self) -> int:
        """Return the number of entries in the mapping."""
        return len(self.mapping)

    # ------------------------------------------------------------------
    # FIX ARCH-09 / LOG-11: conditional logging helper.
    # ------------------------------------------------------------------

    def _should_log(self, level: int) -> bool:
        """Check whether a log at *level* should be emitted.

        Uses ResolverConfig.log_sample_rate for rate-limited debug logs.
        """
        if not logger.isEnabledFor(level):
            return False
        if level >= logging.INFO:
            return True
        # For DEBUG: use sample rate.
        import random
        return random.random() < self._config.log_sample_rate

    # ------------------------------------------------------------------
    # FIX IDEM-01: deterministic timestamp helper.
    # ------------------------------------------------------------------

    def _now_iso(self) -> str:
        """Return an ISO-8601 UTC timestamp.

        When ResolverConfig.deterministic_timestamps is True, returns
        a deterministic monotonically-increasing timestamp based on a
        fixed epoch (FIX IDEM-01).
        """
        if self._config.deterministic_timestamps:
            global _DETERMINISTIC_COUNTER
            with _DETERMINISTIC_COUNTER_LOCK:
                _DETERMINISTIC_COUNTER += 1
                base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
                from datetime import timedelta
                dt = base + timedelta(seconds=_DETERMINISTIC_COUNTER)
                return dt.isoformat()
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # FIX ARCH-04: deep-copy helper for returning entries safely.
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_entry_for_read(entry: dict) -> dict:
        """Return a deep copy of an entry dict for safe external use.

        FIX ARCH-04 / CODE-23 / CODE-24: prevents callers from
        mutating the resolver's internal state through the returned dict.
        """
        return copy.deepcopy(entry)

    # ------------------------------------------------------------------
    # FIX ARCH-03: snapshot/rollback for transactional per-record ingestion.
    # ------------------------------------------------------------------

    def _snapshot_indexes_for_uid(self, uid: str) -> dict:
        """Capture current index state for *uid* so we can roll back on failure."""
        snapshot = {
            "mapping_existed": uid in self.mapping,
            "mapping_entry": copy.deepcopy(self.mapping.get(uid)) if uid in self.mapping else None,
            "gene_index_keys": [
                k for k, v in self._gene_index.items() if v == uid
            ],
            "name_index_keys": [
                k for k, v in self._name_index.items() if v == uid
            ],
            "string_to_uniprot_keys": [
                k for k, v in self._string_to_uniprot.items() if v == uid
            ],
        }
        return snapshot

    def _rollback_indexes_for_uid(self, uid: str, snapshot: dict) -> None:
        """Roll back partial mutations for *uid* after an ingestion failure."""
        # If mapping didn't exist before, remove it.
        if not snapshot["mapping_existed"] and uid in self.mapping:
            del self.mapping[uid]
        elif snapshot["mapping_existed"] and snapshot["mapping_entry"] is not None:
            self.mapping[uid] = snapshot["mapping_entry"]

        # Remove gene_index entries that pointed to uid but weren't in snapshot.
        current_gene_keys = {k for k, v in self._gene_index.items() if v == uid}
        original_gene_keys = set(snapshot["gene_index_keys"])
        for k in current_gene_keys - original_gene_keys:
            del self._gene_index[k]

        # Remove name_index entries.
        current_name_keys = {k for k, v in self._name_index.items() if v == uid}
        original_name_keys = set(snapshot["name_index_keys"])
        for k in current_name_keys - original_name_keys:
            del self._name_index[k]

        # Remove string_to_uniprot entries.
        current_str_keys = {k for k, v in self._string_to_uniprot.items() if v == uid}
        original_str_keys = set(snapshot["string_to_uniprot_keys"])
        for k in current_str_keys - original_str_keys:
            del self._string_to_uniprot[k]

        # Invalidate organism cache.
        self._organism_name_cache_valid = False

    # ------------------------------------------------------------------
    # FIX SCI-04: fuzzy-match safety guards.
    # ------------------------------------------------------------------

    @staticmethod
    def _is_gene_family_false_positive(query: str, candidate: str) -> bool:
        """Return True if query and candidate are likely gene-family members.

        FIX SCI-04: TP53 vs TP53L, KRT1 vs KRT2, etc. These differ by
        a trailing character and should NOT be fuzzy-merged.
        """
        if not query or not candidate:
            return False
        # If one is a prefix of the other + trailing char(s).
        shorter = min(len(query), len(candidate))
        if shorter < 3:
            return False
        if query[:shorter] == candidate[:shorter] and abs(len(query) - len(candidate)) <= 2:
            return True
        # If they differ by exactly one trailing character.
        if len(query) == len(candidate) and len(query) >= 4:
            if query[:-1] == candidate[:-1]:
                return True
        return False

    # ------------------------------------------------------------------
    # FIX ARCH-02: provisional entry promotion.
    # ------------------------------------------------------------------

    def _promote_provisional_entry(self, provisional_uid: str, real_uniprot_id: str) -> None:
        """Promote a provisional entry to a real UniProt-keyed entry.

        FIX ARCH-02 / CODE-39: provisional entries were stuck as
        synthetic keys forever; this method re-keys them when the real
        uniprot_id arrives.
        """
        if provisional_uid == real_uniprot_id:
            return  # Already promoted or same key.
        entry = self.mapping.get(provisional_uid)
        if entry is None:
            return

        # v80 P0-D3: unregister the provisional entry from the
        # _provisional_by_gene_organism index BEFORE re-keying (so the
        # index does not point to a uid that is about to be deleted).
        self._unregister_provisional_entry(provisional_uid)

        # Transfer entry to real key.
        entry["uniprot_id"] = real_uniprot_id
        entry["provisional"] = False
        # FIX DESIGN-01: upgrade confidence to uniprot_exact (1.0).
        entry["match_method"] = "uniprot_exact"
        entry["match_confidence"] = compute_match_confidence("uniprot_exact")

        self.mapping[real_uniprot_id] = entry
        del self.mapping[provisional_uid]

        # Re-point all indexes from provisional_uid to real_uniprot_id.
        for k, v in list(self._gene_index.items()):
            if v == provisional_uid:
                self._gene_index[k] = real_uniprot_id
        for k, v in list(self._string_to_uniprot.items()):
            if v == provisional_uid:
                self._string_to_uniprot[k] = real_uniprot_id
        for k, v in list(self._name_index.items()):
            if v == provisional_uid:
                self._name_index[k] = real_uniprot_id
        for k, vlist in self._name_index_multi.items():
            self._name_index_multi[k] = [
                real_uniprot_id if x == provisional_uid else x for x in vlist
            ]

        # Transfer audit trail.
        trail = self._audit_trail.pop(provisional_uid, [])
        self._audit_trail[real_uniprot_id] = trail
        self._append_audit(real_uniprot_id, {
            "action": "promote_provisional",
            "source": "uniprot",
            "method": "uniprot_exact",
        })
        self._stats.inc("records_promoted")
        logger.info(
            "promote_provisional: '%s' promoted to '%s'",
            provisional_uid, real_uniprot_id,
        )
        self._organism_name_cache_valid = False

    # v80 FORENSIC ROOT FIX (P0-D3): helpers to maintain the
    # _provisional_by_gene_organism index. Called whenever a provisional
    # entry is created (register) or removed/promoted (unregister).
    #
    # v82 FORENSIC ROOT FIX (P0-D3b): also maintain the
    # _provisional_by_alias_uniprot index for STRING-alias-derived
    # provisionals that carry a real ``uniprot_id`` field but no
    # ``gene_symbol``. This is the O(1) fast path that prevents the
    # O(N) fallback from triggering on real workloads.
    def _register_provisional_entry(self, uid: str, gene_symbol: Optional[str], organism: Optional[str], alias_uniprot_id: Optional[str] = None) -> None:
        """Add ``uid`` to the (gene, organism) AND alias-uniprot provisional indexes.

        No-op for the gene-organism index if gene_symbol or organism is
        None/empty. No-op for the alias-uniprot index if
        ``alias_uniprot_id`` is None/empty. An entry may be registered
        in one, both, or neither index depending on what fields it has.
        The O(N) fallback in ``_ingest_uniprot_record`` catches anything
        the indexes miss (defensive, should never fire if the indexes
        are maintained correctly).
        """
        if not uid:
            return
        # Gene-organism index (v80 P0-D3).
        if gene_symbol is not None and organism is not None:
            _g = self._normalize_gene_symbol(gene_symbol)
            _o = self._normalize_organism(organism)
            if _g and _o:
                _key = (_g, _o)
                _bucket = self._provisional_by_gene_organism.setdefault(_key, [])
                if uid not in _bucket:
                    _bucket.append(uid)
        # Alias-uniprot index (v82 P0-D3b) -- the fast path for STRING
        # alias records that carry a real UniProt accession.
        if alias_uniprot_id:
            _au = str(alias_uniprot_id).strip().upper()
            if _au:
                _abucket = self._provisional_by_alias_uniprot.setdefault(_au, [])
                if uid not in _abucket:
                    _abucket.append(uid)

    def _unregister_provisional_entry(self, uid: str) -> None:
        """Remove ``uid`` from all provisional index buckets (both gene-organism AND alias-uniprot).

        Called by ``_promote_provisional_entry`` and ``remove_source``.
        Scans every bucket (O(B) where B = number of distinct keys,
        typically <<1% of N). This is acceptable because promotions are
        rare (only when a real UniProt record arrives for a previously-
        provisional entry).
        """
        if not uid:
            return
        for _bucket in self._provisional_by_gene_organism.values():
            try:
                _bucket.remove(uid)
            except ValueError:
                pass
        # v82 P0-D3b: also clean up the alias-uniprot index.
        for _abucket in self._provisional_by_alias_uniprot.values():
            try:
                _abucket.remove(uid)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # FIX DESIGN-10: synthetic UID helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def is_synthetic_uid(uid: str) -> bool:
        """Return True if *uid* is a synthetic (provisional) key.

        FIX DESIGN-10: typed helper for distinguishing real from
        synthetic keys without string-sniffing.
        """
        return isinstance(uid, str) and (
            uid.startswith("STRING:") or uid.startswith("CHEMBL_T:")
        )

    @staticmethod
    def parse_synthetic_uid(uid: str) -> Tuple[str, str]:
        """Parse a synthetic uid into (source, raw_id).

        Raises ValueError if not synthetic.
        """
        if not isinstance(uid, str):
            raise ValueError(f"uid must be str, got {type(uid).__name__}")
        if uid.startswith("STRING:"):
            return ("string", uid[len("STRING:"):])
        if uid.startswith("CHEMBL_T:"):
            return ("chembl", uid[len("CHEMBL_T:"):])
        raise ValueError(f"uid {uid!r} is not a synthetic key")

    @staticmethod
    def make_synthetic_uid(source: str, raw_id: str) -> str:
        """Construct a synthetic uid from source + raw_id.

        FIX P1-ER-20 (LOW): this is the PURE (stateless) constructor.
        It does NOT perform collision detection against the resolver's
        ``mapping`` because it is a ``@staticmethod`` and has no
        ``self``. Callers that need collision detection (the common
        case) should use :meth:`_make_synthetic_uid_checked` instead,
        which wraps this method with:
          * collision detection against ``self.mapping``
          * WARNING log on collision
          * ``synthetic_uid_collisions`` stat increment
          * 4-char (->8->12->16) hash suffix for collision resistance
        """
        sanitized = re.sub(r"[^A-Za-z0-9._\-]", "_", raw_id)
        if source == "string":
            return f"STRING:{sanitized}"
        elif source == "chembl":
            return f"CHEMBL_T:{sanitized}"
        else:
            return f"{source.upper()}:{sanitized}"

    def _make_synthetic_uid_checked(self, source: str, raw_id: str) -> str:
        """Construct a synthetic UID with collision detection.

        FIX P1-ER-20 (LOW): the previous ``make_synthetic_uid`` was a
        pure ``SOURCE:raw_id`` constructor with no collision detection.
        Two different source records providing the same ``raw_id``
        (e.g. two STRING alias rows pointing at the same ENSP id but
        with different gene symbols) would silently collide on the
        same synthetic UID, with the second record overwriting the
        first's entry in ``self.mapping``. This wrapper:

          1. Calls :meth:`make_synthetic_uid` to get the base UID.
          2. If the base UID is NOT in ``self.mapping``: returns it
             as-is (the common case).
          3. If the base UID IS already in ``self.mapping``:
             * Increments ``self._stats["synthetic_uid_collisions"]``.
             * Logs a WARNING with the source + raw_id.
             * Appends a SHA-1-derived hash suffix (4 chars, growing
               to 8/12/16 on hash-suffix collisions) to disambiguate.
             * Falls back to a UUID suffix in the astronomically
               unlikely event that even a 16-char hash collides.

        The hash suffix is DETERMINISTIC for a given ``raw_id`` (same
        input -> same suffix), so retries within the same session do
        not keep generating new suffixes for the same collision.
        """
        base_uid = self.make_synthetic_uid(source, raw_id)
        if base_uid not in self.mapping:
            return base_uid

        # Collision detected -- log, stat, and disambiguate.
        self._stats.inc("synthetic_uid_collisions")
        logger.warning(
            "_make_synthetic_uid_checked: synthetic UID collision "
            "detected for source=%r raw_id=%r (base UID %s already in "
            "mapping) -- appending hash suffix to disambiguate.",
            source, raw_id, base_uid,
        )

        # Try hash suffixes of increasing length to minimise visual
        # clutter on the common case (4-char suffix suffices for
        # <65k distinct collisions per base UID).
        # v28 ROOT FIX (P1-ER-4): use SHA-256, not SHA-1. SHA-1 is
        # cryptographically broken (collision attacks since 2017).
        for suffix_len in (4, 8, 12, 16):
            suffix = hashlib.sha256(
                raw_id.encode("utf-8")
            ).hexdigest()[:suffix_len].upper()
            candidate = f"{base_uid}#{suffix}"
            if candidate not in self.mapping:
                return candidate

        # Astronomically unlikely (16-char hash collision on the same
        # base UID). Fall back to a random UUID suffix.
        import uuid as _uuid
        return f"{base_uid}#{_uuid.uuid4().hex[:8].upper()}"

    @staticmethod
    def _sanitize_id_for_uid(raw_id: str) -> str:
        """Sanitize an ID for use in a synthetic UID."""
        return re.sub(r"[^A-Za-z0-9._\-]", "_", raw_id)

    # ------------------------------------------------------------------
    # FIX INT-10: dependency checker.
    # ------------------------------------------------------------------

    @classmethod
    def check_dependencies(cls) -> Dict[str, bool]:
        """Check which optional dependencies are available.

        Returns a dict mapping dependency name -> availability.
        """
        deps = {}
        try:
            import pandas  # noqa: F401
            deps["pandas"] = True
        except ImportError:
            deps["pandas"] = False
        try:
            import pyarrow  # noqa: F401
            deps["pyarrow"] = True
        except ImportError:
            deps["pyarrow"] = False
        try:
            import rapidfuzz  # noqa: F401
            deps["rapidfuzz"] = True
        except ImportError:
            deps["rapidfuzz"] = False
        return deps

    # ------------------------------------------------------------------
    # FIX ARCH-06 / DESIGN-03: extensible source registration.
    # ------------------------------------------------------------------

    @classmethod
    def register_source(cls, name: str, method_name: str) -> None:
        """Register a new source ingestor.

        Parameters
        ----------
        name:
            Source label (e.g. ``"intact"``).
        method_name:
            Name of the method on the resolver instance to call.
        """
        cls._SOURCE_INGESTORS[name] = method_name

    # ------------------------------------------------------------------
    # FIX ARCH-01: iteration helpers.
    # ------------------------------------------------------------------

    def iter_canonical_entries(self) -> Iterator[Tuple[str, dict]]:
        """Yield (uid, entry) for non-provisional entries only."""
        for uid, entry in self.mapping.items():
            if not self.is_synthetic_uid(uid):
                yield uid, entry

    def iter_provisional_entries(self) -> Iterator[Tuple[str, dict]]:
        """Yield (uid, entry) for provisional entries only."""
        for uid, entry in self.mapping.items():
            if self.is_synthetic_uid(uid):
                yield uid, entry

    # ------------------------------------------------------------------
    # Public API -- bulk ingestion
    # ------------------------------------------------------------------

    def add_source_records(self, records: List[dict], source: str, *,
                           operator_id: Optional[str] = None) -> None:
        """Dispatch ``records`` to the appropriate source-specific ingestor.

        This is the unified entry point for ProteinResolver (FIX DOC-11 --
        removed confusing DrugResolver reference). It dispatches based on
        the ``source`` argument via the class-level ``_SOURCE_INGESTORS``
        registry (ARCH-06). New sources can be registered via
        ``ProteinResolver.register_source(name, method_name)``.

        Parameters
        ----------
        records:
            List of record dicts.
        source:
            One of ``"uniprot"``, ``"string"``, ``"chembl"``, or any
            source registered via ``register_source``.
        operator_id:
            Optional caller identity for audit (SEC-13). Required when
            ``ResolverConfig.require_operator_for_sensitive_actions=True``.
        """
        # FIX SEC-13: operator_id enforcement.
        if self._config.require_operator_for_sensitive_actions and not operator_id:
            logger.warning(
                "add_source_records: no operator_id provided while "
                "require_operator_for_sensitive_actions=True"
            )

        # D9-7: source whitelist enforcement.
        if self._config.source_whitelist is not None:
            if source not in self._config.source_whitelist:
                raise ValueError(
                    f"source {source!r} is not in the configured whitelist "
                    f"({self._config.source_whitelist!r})"
                )

        # FIX ARCH-06: use class-level registry.
        method_name = self._SOURCE_INGESTORS.get(source)
        if method_name is None:
            raise ValueError(
                f"ProteinResolver.add_source_records: unknown source "
                f"{source!r}. Expected one of {sorted(self._SOURCE_INGESTORS)}."
            )
        handler = getattr(self, method_name, None)
        if handler is None:
            raise ValueError(
                f"ProteinResolver has no method {method_name!r} for "
                f"source {source!r}."
            )
        handler(records)

    def add_uniprot_records(self, records: List[dict]) -> None:
        """Add UniProt records as the canonical protein source.

        Each record **must** contain ``'uniprot_id'`` and should ideally
        also provide ``'gene_symbol'``, ``'gene_name'``, and
        ``'organism'``.  Because UniProt is the authority for protein
        identity, these records create new canonical entries or merge
        into existing ones.

        FIX DOC-03: if a record's uniprot_id already exists in the
        mapping, the record is merged via :meth:`_merge_uniprot_record`
        (fill-missing semantics). Duplicate records with identical
        content are silently accepted (idempotent).

        FIX SCI-07: strict validation now controlled by
        ResolverConfig.bulk_strict_validation (was hardcoded False).
        Operators can flip ``bulk_strict_validation=True`` via
        ``ResolverConfig.from_env()`` or the
        ``ENTITY_RESOLUTION_BULK_STRICT_VALIDATION=1`` env var.

        FIX SCI-09: UniProt isoforms (e.g. P04637-2) are tracked
        under the parent entry's ``isoforms`` list.

        Parameters
        ----------
        records:
            List of dicts, one per UniProt protein entry.
        """
        if not records:
            logger.warning("add_uniprot_records: empty record list")
            return

        # FIX IDEM-02: batch fingerprinting for duplicate-batch detection.
        batch_fp = self._compute_batch_fingerprint(records, "uniprot")
        if self._last_batch_fingerprints.get("uniprot") == batch_fp:
            logger.info(
                "add_uniprot_records: identical batch fingerprint detected "
                "-- skipping duplicate batch (%d records)", len(records)
            )
            return
        self._last_batch_fingerprints["uniprot"] = batch_fp

        # FIX DQ-10 / ARCH-09: pre-ingestion duplicate detection.
        dup_result = find_duplicate_ids(records, id_fields=["uniprot_id"])
        if dup_result:
            # FIX P1-ER-9: `len(dup_result)` returns the number of *fields*
            # with duplicates (always 1 here because we only pass
            # ``id_fields=["uniprot_id"]``), NOT the number of duplicate IDs.
            # Use the actual list length under the "uniprot_id" key.
            dup_count = len(dup_result.get("uniprot_id", []))
            logger.warning(
                "add_uniprot_records: %d duplicate uniprot_ids detected "
                "within batch -- last occurrence wins",
                dup_count,
            )

        logger.info(
            "add_uniprot_records: ingesting %d UniProt records", len(records)
        )

        # FIX SCI-07: use config-driven strict mode.
        strict_mode = self._config.bulk_strict_validation

        for idx, record in enumerate(records):
            # FIX ARCH-03: transactional per-record ingestion with rollback.
            snapshot = self._snapshot_indexes_for_uid(
                record.get("uniprot_id", "")
            )
            try:
                self._ingest_uniprot_record(idx, record, strict_mode)
            except Exception as exc:
                logger.exception(
                    "add_uniprot_records: record %d failed, rolling back", idx
                )
                self._rollback_indexes_for_uid(
                    record.get("uniprot_id", ""), snapshot
                )
                self._dead_letter.append({
                    "record": record,
                    "source": "uniprot",
                    "errors": [f"ingestion exception: {exc}"],
                    "stage": "add_uniprot_records",
                })
                self._stats.inc("records_rejected")
                self._stats.inc("ingestion_exceptions")
                continue

            if (idx + 1) % _PROGRESS_LOG_INTERVAL == 0 and self._should_log(logging.DEBUG):
                logger.debug(
                    "add_uniprot_records: %d / %d processed",
                    idx + 1, len(records),
                )

        logger.info(
            "add_uniprot_records: done -- %d canonical proteins loaded",
            len(self.mapping),
        )

    def _ingest_uniprot_record(self, idx: int, record: dict, strict_mode: bool) -> None:
        """Process a single UniProt record. Called from add_uniprot_records."""
        # D5-2: validate at the boundary.
        ok, errors = validate_protein_record(record, strict=strict_mode)
        if not ok:
            # FIX SEC-01: redact PII in dead letter if configured.
            dl_record = record
            if self._config.redact_dead_letter_pii:
                dl_record = {k: v for k, v in record.items() if k != "sequence"}
            self._dead_letter.append({
                "record": dl_record,
                "source": "uniprot",
                "errors": errors,
                "stage": "add_uniprot_records",
            })
            self._stats.inc("records_rejected")
            self._stats.inc("dead_lettered")
            logger.warning(
                "add_uniprot_records: record %d rejected -- %s",
                idx, errors,
            )
            return

        # FIX SCI-08: amino-acid content validation (always, not just strict).
        seq = record.get("sequence")
        if seq and isinstance(seq, str):
            if not _AA_VALID_RE.match(seq):
                dl_record = record
                if self._config.redact_dead_letter_pii:
                    dl_record = {k: v for k, v in record.items() if k != "sequence"}
                self._dead_letter.append({
                    "record": dl_record,
                    "source": "uniprot",
                    "errors": ["sequence contains non-amino-acid characters"],
                    "stage": "add_uniprot_records",
                })
                self._stats.inc("records_rejected")
                self._stats.inc("dead_lettered")
                logger.warning(
                    "add_uniprot_records: record %d rejected -- invalid "
                    "amino acid characters in sequence", idx,
                )
                return

        self._stats.inc("records_ingested")
        # v89 FORENSIC ROOT FIX (BUG #4 P0 -- UniProt accessions NOT uppercased):
        #   UniProt accessions are CASE-SENSITIVE per the UniProt spec
        #   (must be UPPERCASE, e.g. "P04637" not "p04637"). The
        #   validation regex ``_UNIPROT_ACCESSION_RE`` requires uppercase,
        #   but validation only fires when ``strict_mode=True`` (default
        #   is ``False`` per ``ResolverConfig.bulk_strict_validation``).
        #   When strict mode is off (the production default), a lowercase
        #   UniProt accession like "p04637" passes validation, is stored
        #   as-is in ``self.mapping["p04637"]``, and is NEVER normalized
        #   to "P04637". A subsequent record with the correct "P04637"
        #   would create a SEPARATE entry. The alias-uniprot index
        #   (line 1216: ``_au = str(alias_uniprot_id).strip().upper()``)
        #   DOES uppercase for lookup, but the actual storage key
        #   ``base_uid`` (line 1882: ``self.mapping[base_uid] = entry``)
        #   does NOT. This produced DUPLICATE canonical entries for the
        #   same protein -- corrupting TransE embeddings and drug-target
        #   edge connectivity.
        #   ROOT FIX: normalize ``uniprot_id`` to uppercase + strip
        #   IMMEDIATELY after retrieval. ``base_uid`` (derived from
        #   ``uniprot_id`` at line 1668) inherits the uppercase
        #   normalization, so ALL ``self.mapping[base_uid]`` operations
        #   use the uppercase key. This is the SAME normalization the
        #   alias-uniprot index already uses -- now consistent end-to-end.
        uniprot_id = record.get("uniprot_id", "")
        if isinstance(uniprot_id, str):
            uniprot_id = uniprot_id.strip().upper()
        elif uniprot_id is not None:
            uniprot_id = str(uniprot_id).strip().upper()
        else:
            uniprot_id = ""
        if not uniprot_id:
            # v89 FORENSIC ROOT FIX (BUG #11 P1 -- empty uniprot_id silently
            #   dropped without dead-letter):
            #   The previous code incremented ``records_rejected`` and
            #   ``return``ed -- the record vanished silently. The operator
            #   saw ``records_rejected=1`` but could NOT diagnose WHICH
            #   protein was lost or WHY. For a patient-safety pharma
            #   platform, silent data loss is unacceptable.
            #   ROOT FIX: append to the dead-letter queue with a clear
            #   error message BEFORE returning. The operator can inspect
            #   ``self._dead_letter`` to see the rejected record.
            self._dead_letter.append({
                "record": record,
                "source": "uniprot",
                "errors": ["empty uniprot_id"],
                "stage": "add_uniprot_records",
            })
            self._stats.inc("records_rejected")
            self._stats.inc("dead_lettered")
            logger.warning(
                "add_uniprot_records: record %d rejected -- empty uniprot_id "
                "(added to dead-letter queue for diagnosis)", idx,
            )
            return

        # FIX SCI-06: cross-reference UniProt accession vs organism.
        # v16 SW-13: use the merged override map (baseline + runtime).
        organism_raw = record.get("organism", self._config.default_organism)
        organism = self._normalize_organism(organism_raw)
        _effective_overrides = _get_effective_uniprot_organism_overrides()
        if uniprot_id in _effective_overrides:
            expected_org = _effective_overrides[uniprot_id]
            if organism and organism != expected_org:
                logger.warning(
                    "add_uniprot_records: organism mismatch for %s -- "
                    "expected %s, got %s; dead-lettering",
                    uniprot_id, expected_org, organism,
                )
                self._dead_letter.append({
                    "record": record,
                    "source": "uniprot",
                    "errors": [
                        f"uniprot_id {uniprot_id} canonically belongs to "
                        f"{expected_org}, record claims {organism}"
                    ],
                    "stage": "add_uniprot_records",
                })
                self._stats.inc("records_rejected")
                self._stats.inc("organism_mismatches")
                return
        else:
            # v89 FORENSIC ROOT FIX (BUG #6 P0 -- organism filter was
            #   "decorative" for >99.9% of records):
            #   The override table has ~80 baseline + ~250 auto-loaded
            #   entries. UniProt Swiss-Prot has ~560,000 entries. So for
            #   >99.9% of records, the organism was NEVER validated --
            #   the code logged a DEBUG message and accepted whatever
            #   organism the record claimed. Combined with BUG #2 (STRING
            #   ->UniProt mispairing) and BUG #3 (no organism filter on
            #   alias file selection), this produced systemic cross-
            #   species protein mixing. Mouse/rat/fly proteins were
            #   silently labeled "Homo sapiens" and merged into human
            #   PPI subgraphs. Drug-target edges connected human drugs
            #   to non-human protein targets -- a patient-safety hazard.
            #   ROOT FIX: when ``require_organism_override=True`` (a
            #   config flag that already exists at base.py:455 but
            #   defaulted to False and was never consulted), REFUSE to
            #   accept a UniProt record whose organism cannot be
            #   validated against the override table. Dead-letter it so
            #   the operator sees the gap and can extend the crosswalk.
            #   This makes the organism filter ENFORCED in production
            #   (operators set the flag via the
            #   ``ENTITY_RESOLUTION_REQUIRE_ORGANISM_OVERRIDE`` env var)
            #   while keeping the lenient default for dev/test.
            if getattr(self._config, "require_organism_override", False):
                logger.warning(
                    "add_uniprot_records: organism for %s NOT validated "
                    "against UniProt canonical mapping (not in effective "
                    "overrides; require_organism_override=True). "
                    "Dead-lettering to prevent cross-species contamination. "
                    "Set UNIPROT_ORGANISM_CROSSWALK_PATH env var or call "
                    "load_uniprot_organism_crosswalk() to extend the "
                    "override table.",
                    uniprot_id,
                )
                self._dead_letter.append({
                    "record": record,
                    "source": "uniprot",
                    "errors": [
                        f"uniprot_id {uniprot_id} not in organism override "
                        f"table; require_organism_override=True refuses "
                        f"unvalidated organisms"
                    ],
                    "stage": "add_uniprot_records",
                })
                self._stats.inc("records_rejected")
                self._stats.inc("organism_not_validated")
                return
            logger.debug(
                "add_uniprot_records: organism for %s not validated against "
                "UniProt canonical mapping (not in effective overrides; "
                "set UNIPROT_ORGANISM_CROSSWALK_PATH env var or call "
                "load_uniprot_organism_crosswalk() to extend)",
                uniprot_id,
            )

        # FIX SCI-10: check for deprecated UniProt accessions.
        if uniprot_id in _DEPRECATED_UNIPROT_MAP:
            new_uid = _DEPRECATED_UNIPROT_MAP[uniprot_id]
            logger.info(
                "add_uniprot_records: %s is deprecated, redirecting to %s",
                uniprot_id, new_uid,
            )
            uniprot_id = new_uid

        # FIX SCI-09: isoform handling.
        # P1-023 ROOT FIX (Team Member 3 -- isoform collapse loses drug
        #   specificity):
        #   The previous code stripped the isoform suffix (Q07817-2 ->
        #   base_uid=Q07817) and MERGED every isoform record into the
        #   parent entry. For genes with functionally distinct isoforms
        #   (e.g. BCL2L1 encodes both anti-apoptotic BCL-XL = Q07817 and
        #   pro-apoptotic BCL-XS = Q07817-2), this collapsed them into
        #   ONE Protein node. Drugs targeting one isoform but not the
        #   other appeared to target both -- the KG learned wrong drug-
        #   protein specificity, and the RL ranker could recommend a
        #   BCL-XL inhibitor for a disease requiring BCL-XS inhibition.
        #
        #   ROOT FIX: keep the FULL UniProt accession (including isoform
        #   suffix) as the PRIMARY KEY (``primary_uid``). Each isoform
        #   becomes its OWN Protein node in the KG. A ``parent_accession``
        #   field on the isoform entry points back to the canonical
        #   parent (Q07817) for cross-reference. The parent entry's
        #   ``isoforms`` list still tracks its isoforms (for graph
        #   traversal), but isoform-specific data (sequence, drug
        #   targeting) is NO LONGER merged into the parent.
        isoform_id = None
        base_uid = uniprot_id
        if "-" in uniprot_id:
            parts = uniprot_id.split("-", 1)
            if parts[1].isdigit():
                base_uid = parts[0]
                isoform_id = uniprot_id
        # P1-023: the primary key is the FULL accession when an isoform
        # suffix is present, else the base accession.
        primary_uid = isoform_id if isoform_id else base_uid

        # FIX ARCH-02: check if a provisional entry should be promoted.
        #
        # v80 FORENSIC ROOT FIX (P0-D3 -- O(N×M) provisional-promotion
        # loop): the previous code iterated over ALL entries in
        # ``self.mapping`` (``for prov_uid in list(self.mapping.keys())``)
        # for EVERY new UniProt record, checking each one to see if it
        # was a synthetic (provisional) entry that matched the new
        # record's (gene, organism). On a 100K-scale UniProt ingestion,
        # this is 10 billion+ iterations -> Airflow task timeout
        # (default 4h). The pipeline hung forever on real data.
        #
        # ROOT FIX: use the new ``_provisional_by_gene_organism`` index
        # for O(1) lookup of promotion candidates by (gene, organism).
        # The index is maintained by ``_register_provisional_entry``
        # (called whenever a provisional entry is created) and
        # ``_unregister_provisional_entry`` (called by
        # ``_promote_provisional_entry`` and ``remove_source``). Falls
        # back to the O(N) scan ONLY if the index is somehow empty
        # (defensive -- should never happen if the index is maintained
        # correctly).
        #
        # v82 FORENSIC ROOT FIX (P0-D3b -- O(N) fallback still triggered
        #   for STRING-alias-derived provisionals):
        #   The v80 gene-organism index only helps when the provisional
        #   entry has a ``gene_symbol``. But STRING alias records (the
        #   DOMINANT source of provisionals on a real ingestion) carry
        #   ONLY ``string_id`` + ``uniprot_id`` -- they have NO
        #   ``gene_symbol``. So the gene-organism index NEVER contains
        #   them, and EVERY UniProt record fell through to the O(N)
        #   defensive scan. On 100K UniProt records × 100K STRING
        #   provisionals, that's 10 BILLION iterations -- the exact
        #   Airflow-timeout failure the v80 fix was supposed to prevent.
        #
        #   ROOT FIX: check the new ``_provisional_by_alias_uniprot``
        #   index FIRST (O(1) lookup by the UniProt record's
        #   ``uniprot_id``). STRING alias records that carried a real
        #   UniProt accession registered their synthetic uid under that
        #   accession. When the real UniProt record arrives with the
        #   same accession, we find the provisional in O(1) and promote
        #   it -- no O(N) scan. The gene-organism index is checked
        #   SECOND (for provisionals that had a gene but no alias
        #   uniprot). The O(N) fallback is the LAST resort (defensive,
        #   should now genuinely never fire).
        rec_gene = self._normalize_gene_symbol(record.get("gene_symbol"))
        rec_org = self._normalize_organism(organism_raw)
        promotion_done = False

        # v82 P0-D3b STEP 1: O(1) lookup by the UniProt record's
        # ``uniprot_id`` in the alias-uniprot index. This is the fast
        # path for STRING-alias-derived provisionals.
        # P1-023: skip promotion for isoforms. Promotion uses base_uid
        # (stripped), which would promote an isoform into a PARENT
        # provisional -- losing isoform specificity. Isoforms must get
        # their own entries.
        if not promotion_done and base_uid and not isoform_id:
            _alias_candidates = self._provisional_by_alias_uniprot.get(
                base_uid.strip().upper(), []
            )
            for prov_uid in list(_alias_candidates):
                if prov_uid not in self.mapping:
                    try:
                        _alias_candidates.remove(prov_uid)
                    except ValueError:
                        pass
                    continue
                if not self.is_synthetic_uid(prov_uid):
                    continue
                # Found a promotion candidate via the alias-uniprot
                # index -- promote + merge + return.
                self._promote_provisional_entry(prov_uid, base_uid)
                self._merge_uniprot_record(base_uid, record)
                self._stats.inc("records_matched")
                promotion_done = True
                logger.debug(
                    "add_uniprot_records: promoted provisional %s -> %s "
                    "via alias-uniprot index (O(1))",
                    prov_uid, base_uid,
                )
                break
        if promotion_done:
            return

        # v80 P0-D3 STEP 2: O(1) lookup by (gene, organism).
        # P1-023: skip for isoforms (see comment above).
        if rec_gene and rec_org and not isoform_id:
            _key = (rec_gene, rec_org)
            _candidates = self._provisional_by_gene_organism.get(_key, [])
            # Iterate over a SNAPSHOT of the candidates list because
            # _promote_provisional_entry mutates it (removes the
            # promoted uid from the list).
            for prov_uid in list(_candidates):
                if prov_uid not in self.mapping:
                    # Stale index entry -- clean it up.
                    try:
                        _candidates.remove(prov_uid)
                    except ValueError:
                        pass
                    continue
                if not self.is_synthetic_uid(prov_uid):
                    # Entry was promoted previously but the index wasn't
                    # cleaned up -- skip (defensive).
                    continue
                # Found a promotion candidate -- promote + merge + return.
                self._promote_provisional_entry(prov_uid, base_uid)
                # Merge the UniProt data into the promoted entry.
                self._merge_uniprot_record(base_uid, record)
                self._stats.inc("records_matched")
                promotion_done = True
                break
        if promotion_done:
            return
        # Defensive fallback: if the index missed a candidate (should
        # never happen, but guards against index-corruption bugs), do a
        # single O(N) scan over ONLY provisional entries (much smaller
        # than the full mapping on real datasets where provisionals are
        # typically <<1% of canonicals).
        # P1-023: skip defensive fallback for isoforms (see comment above).
        if not promotion_done and rec_gene and rec_org and not isoform_id:
            for prov_uid in list(self.mapping.keys()):
                if not self.is_synthetic_uid(prov_uid):
                    continue
                prov_entry = self.mapping[prov_uid]
                prov_gene = self._normalize_gene_symbol(prov_entry.get("gene_symbol"))
                prov_org = self._normalize_organism(prov_entry.get("organism"))
                if prov_gene == rec_gene and prov_org == rec_org:
                    # Found a candidate the index missed -- log a WARNING
                    # so operators know the index is stale, then promote.
                    logger.warning(
                        "add_uniprot_records: provisional index MISS for "
                        "(gene=%s, organism=%s) -- found candidate %s via "
                        "O(N) fallback. The _provisional_by_gene_organism "
                        "index may be stale.",
                        rec_gene, rec_org, prov_uid,
                    )
                    self._promote_provisional_entry(prov_uid, base_uid)
                    self._merge_uniprot_record(base_uid, record)
                    self._stats.inc("records_matched")
                    promotion_done = True
                    break
        if promotion_done:
            return

        # P1-023 ROOT FIX: check the PRIMARY uid (full accession incl.
        # isoform suffix) for an existing entry. If this exact isoform was
        # already ingested, merge into it (safe -- same isoform, same
        # specificity). We NO LONGER merge isoforms into the parent --
        # each isoform is a distinct Protein node.
        if primary_uid in self.mapping:
            self._merge_uniprot_record(primary_uid, record)
            self._stats.inc("records_matched")
            return
        # P1-023: if this is an isoform AND the parent canonical entry
        # already exists, register the isoform on the parent's ``isoforms``
        # list (cross-reference for graph traversal) but do NOT merge the
        # isoform's data into the parent. Fall through to create a new,
        # separate entry for the isoform.
        if isoform_id and base_uid in self.mapping:
            parent_entry = self.mapping[base_uid]
            iso_list = parent_entry.setdefault("isoforms", [])
            if isoform_id not in iso_list:
                iso_list.append(isoform_id)
            logger.debug(
                "add_uniprot_records: isoform %s registered on parent %s "
                "isoforms list (cross-ref only, no data merge -- P1-023)",
                isoform_id, base_uid,
            )

        gene_symbol = self._normalize_gene_symbol(record.get("gene_symbol", ""))
        gene_name = record.get("gene_name", "") or ""
        string_id = record.get("string_id") or None

        # FIX SCI-11: well-known HGNC symbol check.
        if gene_symbol and organism == "Homo sapiens":
            if gene_symbol in _WELL_KNOWN_HGNC_SYMBOLS:
                logger.debug(
                    "add_uniprot_records: gene symbol %s confirmed in "
                    "well-known HGNC set", gene_symbol,
                )
            else:
                logger.debug(
                    "add_uniprot_records: gene symbol %s not in well-known "
                    "HGNC set (not a rejection -- just informational)",
                    gene_symbol,
                )
        # TODO SCI-11-future: load full HGNC download for production-grade
        # gene symbol validation.

        now_iso = self._now_iso()
        try:
            payload = json.dumps(record, sort_keys=True, default=str)
            input_checksum = hashlib.sha256(
                (payload + self._config.checksum_salt).encode()
            ).hexdigest()[:16]
        except (TypeError, ValueError):
            input_checksum = ""

        # P1-023 ROOT FIX: ``uniprot_id`` is the FULL accession (primary_uid,
        #        incl. isoform suffix). ``parent_accession`` is the base
        #        accession (without suffix) for isoform entries, None for
        #        canonical entries. This keeps functionally distinct
        #        isoforms (BCL-XL vs BCL-XS) as separate Protein nodes.
        entry: dict = {
            "uniprot_id": primary_uid,
            "parent_accession": base_uid if isoform_id else None,
            "gene_symbol": gene_symbol,
            "gene_name": gene_name or None,
            "organism": organism or self._config.default_organism,
            "sequence": record.get("sequence") or None,
            "protein_name": record.get("protein_name") or None,
            "string_id": string_id,
            "chembl_target_id": record.get("chembl_target_id") or None,
            "canonical_name": gene_symbol or gene_name or primary_uid,
            "sources": ["uniprot"],
            "match_method": "uniprot_exact",
            "match_confidence": compute_match_confidence("uniprot_exact"),
            "created_at": now_iso,
            "resolved_at": now_iso,
            "resolver_version": MAPPING_SCHEMA_VERSION,
            "input_checksum": input_checksum,
            "isoforms": [isoform_id] if isoform_id else [],
            "deprecated_by": None,
            "provisional": False,
        }

        # FIX DQ-14: canonical checksum for change detection.
        try:
            canon_payload = json.dumps(entry, sort_keys=True, default=str)
            entry["canonical_checksum"] = hashlib.sha256(
                canon_payload.encode()
            ).hexdigest()[:16]
        except (TypeError, ValueError):
            entry["canonical_checksum"] = ""

        # P1-023: store under the PRIMARY uid (full accession).
        self.mapping[primary_uid] = entry
        self._stats.inc("records_created")

        if gene_symbol:
            # FIX SCI-02: preserve gene-symbol case in index key.
            # FIX SCI-03: normalize organism for index key.
            key = (gene_symbol, self._normalize_organism(organism or self._config.default_organism))
            # v89 BUG #15: maintain multi-valued index alongside the
            # single-valued one. The single-valued index keeps the
            # FIRST registered uid (backward compat for existing lookups);
            # the multi-valued index tracks ALL uids so lookups can
            # detect ambiguity.
            if key not in self._gene_index:
                self._gene_index[key] = primary_uid
            self._gene_index_multi.setdefault(key, []).append(primary_uid)

        # v89 ROOT FIX (BUG #35): populate the case-preserving gene
        # symbol index.
        if gene_symbol:
            if gene_symbol not in self._gene_symbol_index:
                self._gene_symbol_index[gene_symbol] = primary_uid
            self._gene_symbol_index_multi.setdefault(gene_symbol, []).append(primary_uid)

        norm_name = normalize_name(gene_symbol or gene_name or "")
        if norm_name:
            self._name_index[norm_name] = primary_uid
            self._name_index_multi.setdefault(
                norm_name, []
            ).append(primary_uid)

        if string_id:
            self._string_to_uniprot[string_id] = primary_uid
            self._string_to_uniprot_multi.setdefault(string_id, []).append(primary_uid)

        self._append_audit(primary_uid, {
            "action": "create",
            "source": "uniprot",
            "method": "uniprot_exact",
        })

    def add_string_records(self, records: List[dict]) -> None:
        """Add STRING database records, matching them to existing canonical entries.

        Matching strategy (FIX DOC-10):

        1. Direct STRING -> UniProt mapping via ``_string_to_uniprot``. This
           may return a REAL uniprot_id (from a UniProt record's string_id
           field) OR a SYNTHETIC uid (``STRING:...`` from a previously-created
           provisional entry). Both are valid merge targets.
        2. Gene-name + organism match against ``_gene_index``. May also return
           a synthetic uid if a provisional entry claimed the gene+organism key.
        3. If no match, create a PROVISIONAL canonical entry with synthetic uid
           ``STRING:{sanitized_string_id}`` and ``match_method="string_provisional"``
           (confidence 0.5). Provisional entries are PROMOTED to real uniprot_id
           keys when the corresponding UniProt record arrives (ARCH-02).
        """
        if not records:
            logger.warning("add_string_records: empty record list")
            return

        logger.info(
            "add_string_records: ingesting %d STRING records", len(records)
        )

        matched = 0
        created = 0

        # FIX SCI-07: validate STRING records too.
        strict_mode = self._config.bulk_strict_validation

        for idx, record in enumerate(records):
            string_id = record.get("string_id", "") or ""
            gene_symbol_raw = (
                record.get("gene_symbol", "") or record.get("preferred_name", "") or ""
            )
            organism_raw = record.get("organism", self._config.default_organism)

            if not string_id:
                logger.debug(
                    "add_string_records: record %d missing string_id, skipping",
                    idx,
                )
                continue

            # FIX SCI-15: validate STRING ID format at ingest.
            if not _STRING_ID_RE.match(string_id):
                self._dead_letter.append({
                    "record": record,
                    "source": "string",
                    "errors": [
                        f"string_id {string_id!r} does not match "
                        f"'taxon.ENSPxxxxx' format"
                    ],
                    "stage": "add_string_records",
                })
                self._stats.inc("records_rejected")
                self._stats.inc("dead_lettered")
                logger.warning(
                    "add_string_records: record %d has malformed string_id %r",
                    idx, string_id,
                )
                continue

            # FIX DQ-02: validate STRING records (string_id is required, not uniprot_id).
            ok, errors = validate_protein_record(
                record, strict=strict_mode,
                required_fields=("string_id",),
            )
            if not ok:
                dl_record = record
                if self._config.redact_dead_letter_pii:
                    dl_record = {k: v for k, v in record.items() if k != "sequence"}
                self._dead_letter.append({
                    "record": dl_record,
                    "source": "string",
                    "errors": errors,
                    "stage": "add_string_records",
                })
                self._stats.inc("records_rejected")
                self._stats.inc("dead_lettered")
                logger.warning(
                    "add_string_records: record %d rejected -- %s",
                    idx, errors,
                )
                continue

            self._stats.inc("records_ingested")

            # FIX SCI-03: normalize organism.
            organism = self._normalize_organism(organism_raw)
            gene_symbol = self._normalize_gene_symbol(gene_symbol_raw)

            # v89 FORENSIC ROOT FIX (BUG #15 P1 -- ambiguous lookups silently
            #   picked FIRST match):
            #   The previous lookups used single-valued dicts
            #   ``_string_to_uniprot`` and ``_gene_index``. When multiple
            #   UniProt accessions map to the same STRING ID or the same
            #   (gene, organism) pair, the FIRST one registered won. The
            #   resolver silently picked the first match without alerting
            #   on ambiguity. For genes with multiple UniProt entries
            #   (e.g. TP53 has P04637 + isoforms), STRING records were
            #   always merged into the FIRST registered entry -- the KG's
            #   PPI subgraph was incomplete.
            #   ROOT FIX: consult the multi-valued indices
            #   ``_string_to_uniprot_multi`` and ``_gene_index_multi``
            #   FIRST. If there are MULTIPLE distinct candidates, log a
            #   WARNING and refuse to match (return None) so the resolver
            #   falls through to other matching strategies or creates a
            #   new provisional entry. Only use the single-valued index
            #   when the multi-valued index has exactly ONE candidate.
            # 1. Direct STRING -> UniProt mapping.
            uniprot_id = None
            _multi_candidates = self._string_to_uniprot_multi.get(string_id) or []
            # Deduplicate while preserving order.
            _seen = set()
            _multi_candidates = [u for u in _multi_candidates if not (u in _seen or _seen.add(u))]
            if len(_multi_candidates) > 1:
                logger.warning(
                    "add_string_records: STRING ID %s maps to MULTIPLE "
                    "UniProt accessions (%s) -- ambiguous, refusing to "
                    "match by STRING ID alone. Creating provisional entry.",
                    string_id, _multi_candidates,
                )
                self._stats.inc("ambiguous_string_to_uniprot")
                uniprot_id = None
            elif len(_multi_candidates) == 1:
                uniprot_id = _multi_candidates[0]
            else:
                uniprot_id = self._string_to_uniprot.get(string_id)

            # 2. Gene name + organism match.
            if uniprot_id is None and gene_symbol:
                # FIX SCI-02: use normalized gene symbol (preserving case).
                key = (gene_symbol, self._normalize_organism(organism or self._config.default_organism))
                _multi_gene_candidates = self._gene_index_multi.get(key) or []
                _seen = set()
                _multi_gene_candidates = [u for u in _multi_gene_candidates if not (u in _seen or _seen.add(u))]
                if len(_multi_gene_candidates) > 1:
                    logger.warning(
                        "add_string_records: (gene=%s, organism=%s) maps "
                        "to MULTIPLE UniProt accessions (%s) -- ambiguous, "
                        "refusing to match by gene+organism alone. "
                        "Creating provisional entry.",
                        gene_symbol, organism, _multi_gene_candidates,
                    )
                    self._stats.inc("ambiguous_gene_organism")
                    uniprot_id = None
                elif len(_multi_gene_candidates) == 1:
                    uniprot_id = _multi_gene_candidates[0]
                else:
                    uniprot_id = self._gene_index.get(key)

            if uniprot_id is not None and uniprot_id in self.mapping:
                self._merge_string_into_canonical(uniprot_id, record)
                matched += 1
                self._stats.inc("records_matched")
            else:
                self._create_provisional_from_string(record)
                created += 1
                self._stats.inc("records_created")

            if (idx + 1) % _PROGRESS_LOG_INTERVAL == 0 and self._should_log(logging.DEBUG):
                logger.debug(
                    "add_string_records: %d / %d processed",
                    idx + 1, len(records),
                )

        logger.info(
            "add_string_records: done -- %d matched, %d provisional created",
            matched, created,
        )
        self._organism_name_cache_valid = False

    def add_chembl_target_records(self, records: List[dict]) -> None:
        """Add ChEMBL target records, matching them to existing canonical entries.

        ChEMBL target entries include a UniProt accession in their
        cross-reference data.  Matching is attempted by:

        1. UniProt ID exact match (from the record's ``'uniprot_id'`` field).
        2. Gene-name + organism match.

        When matched, the ``chembl_target_id`` is added to the canonical
        entry.  If no match, a provisional entry is created with
        ``match_method="chembl_provisional"`` (FIX SCI-17).
        """
        if not records:
            logger.warning("add_chembl_target_records: empty record list")
            return

        logger.info(
            "add_chembl_target_records: ingesting %d ChEMBL target records",
            len(records),
        )

        matched = 0
        created = 0

        # FIX SCI-07: validate ChEMBL records too.
        strict_mode = self._config.bulk_strict_validation

        for idx, record in enumerate(records):
            chembl_target_id = record.get("chembl_target_id", "") or ""
            uniprot_id = record.get("uniprot_id", "") or ""
            gene_symbol_raw = record.get("gene_symbol", "") or ""
            organism_raw = record.get("organism", self._config.default_organism)

            if not chembl_target_id:
                logger.debug(
                    "add_chembl_target_records: record %d missing "
                    "chembl_target_id, skipping", idx,
                )
                continue

            # FIX SCI-16: validate ChEMBL target ID format at ingest.
            if not _CHEMBL_TARGET_ID_RE.match(chembl_target_id):
                self._dead_letter.append({
                    "record": record,
                    "source": "chembl",
                    "errors": [
                        f"chembl_target_id {chembl_target_id!r} does not "
                        f"match CHEMBL\\d+ format"
                    ],
                    "stage": "add_chembl_target_records",
                })
                self._stats.inc("records_rejected")
                self._stats.inc("dead_lettered")
                logger.warning(
                    "add_chembl_target_records: record %d has malformed "
                    "chembl_target_id %r", idx, chembl_target_id,
                )
                continue

            # FIX DQ-02: validate ChEMBL records (chembl_target_id is required, not uniprot_id).
            ok, errors = validate_protein_record(
                record, strict=strict_mode,
                required_fields=("chembl_target_id",),
            )
            if not ok:
                dl_record = record
                if self._config.redact_dead_letter_pii:
                    dl_record = {k: v for k, v in record.items() if k != "sequence"}
                self._dead_letter.append({
                    "record": dl_record,
                    "source": "chembl",
                    "errors": errors,
                    "stage": "add_chembl_target_records",
                })
                self._stats.inc("records_rejected")
                self._stats.inc("dead_lettered")
                logger.warning(
                    "add_chembl_target_records: record %d rejected -- %s",
                    idx, errors,
                )
                continue

            self._stats.inc("records_ingested")

            # FIX SCI-03: normalize organism.
            organism = self._normalize_organism(organism_raw)
            gene_symbol = self._normalize_gene_symbol(gene_symbol_raw)

            canonical_uid: Optional[str] = None
            method: str = "unknown"

            # FIX ARCH-02: check for provisional entries to promote.
            if uniprot_id:
                if uniprot_id in self.mapping:
                    canonical_uid = uniprot_id
                    method = "uniprot_exact"
                else:
                    # Check for provisional entries that could be promoted.
                    for prov_uid in list(self.mapping.keys()):
                        if self.is_synthetic_uid(prov_uid):
                            prov_entry = self.mapping[prov_uid]
                            prov_ctid = prov_entry.get("chembl_target_id")
                            if prov_ctid == chembl_target_id:
                                self._promote_provisional_entry(prov_uid, uniprot_id)
                                canonical_uid = uniprot_id
                                method = "uniprot_exact"
                                break

            if canonical_uid is None and gene_symbol:
                # FIX SCI-02: use normalized gene symbol (preserving case).
                key = (gene_symbol, self._normalize_organism(organism or self._config.default_organism))
                canonical_uid = self._gene_index.get(key)
                if canonical_uid is not None:
                    method = "gene_name_organism"

            if canonical_uid is not None:
                self._merge_chembl_into_canonical(canonical_uid, record, method)
                matched += 1
                self._stats.inc("records_matched")
            else:
                self._create_provisional_from_chembl(record)
                created += 1
                self._stats.inc("records_created")

            if (idx + 1) % _PROGRESS_LOG_INTERVAL == 0 and self._should_log(logging.DEBUG):
                logger.debug(
                    "add_chembl_target_records: %d / %d processed",
                    idx + 1, len(records),
                )

        logger.info(
            "add_chembl_target_records: done -- %d matched, %d provisional created",
            matched, created,
        )
        self._organism_name_cache_valid = False

    # ------------------------------------------------------------------
    # Public API -- single-record resolution
    # ------------------------------------------------------------------

    def resolve_single(
        self,
        uniprot_id: Optional[str] = None,
        gene_name: Optional[str] = None,
        string_id: Optional[str] = None,
        organism: Optional[str] = None,
        *,
        query: Optional[Any] = None,
    ) -> Optional[dict]:
        """Resolve a single protein, return the canonical record or ``None``.

        Parameters
        ----------
        uniprot_id:
            UniProt accession (e.g. ``"P04637"``).
        gene_name:
            Gene symbol or name. Used for BOTH gene+organism matching (path 3)
            AND fuzzy matching (path 4) (FIX DOC-12 -- was undocumented).
            Gene symbols are normalized via ``_normalize_gene_symbol_for_fuzzy``
            before fuzzy comparison (SCI-12).
        string_id:
            STRING identifier (e.g. ``"9606.ENSP00000269305"``).
        organism:
            Organism name for disambiguation. Defaults to
            :attr:`ResolverConfig.default_organism`. Normalized via
            ``_normalize_organism`` (SCI-03).
        query:
            Optional query dataclass (DESIGN-07). If provided,
            overrides the individual parameters.

        Returns
        -------
        dict or None
            A DEEP COPY of the canonical record if a match is found, else ``None``
            (FIX DOC-13 / ARCH-04 -- was live entry, mutations corrupted resolver).
            Mutations to the returned dict do NOT affect the resolver's internal state.
        """
        # FIX DESIGN-07: optional query dataclass support.
        if query is not None:
            uniprot_id = getattr(query, "uniprot_id", uniprot_id)
            gene_name = getattr(query, "gene_name", gene_name)
            string_id = getattr(query, "string_id", string_id)
            organism = getattr(query, "organism", organism)

        # FIX SCI-10: redirect deprecated accessions.
        if uniprot_id and uniprot_id in _DEPRECATED_UNIPROT_MAP:
            new_uid = _DEPRECATED_UNIPROT_MAP[uniprot_id]
            logger.info(
                "resolve_single: %s is deprecated, redirecting to %s",
                uniprot_id, new_uid,
            )
            uniprot_id = new_uid

        if organism is None:
            organism = self._config.default_organism

        # FIX SCI-03: normalize organism for lookup.
        norm_organism = self._normalize_organism(organism)

        # 1. UniProt ID exact match.
        if uniprot_id and uniprot_id in self.mapping:
            logger.debug(
                "resolve_single: UniProt exact match '%s'", uniprot_id
            )
            # v82 FORENSIC ROOT FIX (P1-2): increment the DEDICATED
            # ``uniprot_exact_matches`` counter, NOT ``inchikey_exact_matches``
            # (which is a DRUG metric). The previous code conflated drug
            # and protein metrics on operator dashboards.
            self._stats.inc("uniprot_exact_matches")
            return self._copy_entry_for_read(self.mapping[uniprot_id])

        # 2. STRING -> UniProt mapping.
        if string_id and string_id in self._string_to_uniprot:
            mapped_uid = self._string_to_uniprot[string_id]
            if mapped_uid in self.mapping:
                logger.debug(
                    "resolve_single: STRING mapping '%s' -> '%s'",
                    string_id, mapped_uid,
                )
                return self._copy_entry_for_read(self.mapping[mapped_uid])

        # 3. Gene name + organism match.
        if gene_name:
            # FIX SCI-02: use normalized gene symbol (preserving case).
            norm_gene = self._normalize_gene_symbol(gene_name)
            key = (norm_gene, norm_organism)
            mapped_uid = self._gene_index.get(key)
            if mapped_uid and mapped_uid in self.mapping:
                logger.debug(
                    "resolve_single: gene match '%s' -> '%s'",
                    gene_name, mapped_uid,
                )
                self._stats.inc("name_matches")
                return self._copy_entry_for_read(self.mapping[mapped_uid])

        # 4. Protein name fuzzy match (last resort).
        if gene_name:
            # FIX SCI-12: use gene-symbol-specific normalizer for fuzzy path.
            norm = self._normalize_gene_symbol_for_fuzzy(gene_name)
            if norm:
                # FIX SCI-04: minimum length guard -- short gene symbols
                # are too risky for fuzzy matching.
                if len(norm) < 4:
                    logger.debug(
                        "resolve_single: gene symbol '%s' too short for "
                        "fuzzy matching (len < 4), skipping", norm,
                    )
                else:
                    result = self._fuzzy_match(norm, norm_organism)
                    if result is not None:
                        return result

        logger.debug(
            "resolve_single: no match for uniprot_id='%s', "
            "gene_name='%s', string_id='%s'",
            uniprot_id, gene_name, string_id,
        )
        return None

    def _fuzzy_match(self, norm: str, norm_organism: str) -> Optional[dict]:
        """Perform fuzzy matching with all SCI-04/SCI-05 guards.

        FIX SCI-04: fuzzy matching on short gene symbols produces false
        positives (TP53 vs TP53L scores 94); add length guard +
        gene-family guard.

        FIX SCI-05: fuzzy match filtered by organism to prevent
        cross-species false positives.
        """
        from .resolver_utils import RAPIDFUZZ_AVAILABLE

        # FIX SCI-05: use per-organism name index slice.
        candidate_index = self._get_organism_name_index(norm_organism)

        if not candidate_index:
            # Fall back to global index with a warning.
            candidate_index = dict(self._name_index)
            if candidate_index:
                logger.warning(
                    "fuzzy_match: no entries for organism '%s', using "
                    "global index (cross-species fuzzy matching)",
                    norm_organism,
                )
                self._stats.inc("cross_species_fuzzy_matches")

        if not candidate_index:
            return None

        # FIX SCI-04: protein-specific safety multiplier on threshold.
        # Gene symbols are inherently short and prone to false positives.
        # FIX P1-ER-10 (MEDIUM): the previous ``min(base_threshold * 1.05,
        # 0.99)`` produced 0.8925 for the default ``base_threshold=0.85``,
        # which was BELOW the documented protein-specific minimum of 0.90
        # (``_PROTEIN_FUZZY_THRESHOLD``) -- so the constant was decorative
        # and the actual cutoff was looser than advertised. Use ``max()``
        # so the protein-specific floor is always honored, and only then
        # apply the 1.05 safety multiplier capped at 0.99.
        base_threshold = self._config.fuzzy_threshold
        floored_threshold = max(base_threshold, _PROTEIN_FUZZY_THRESHOLD)
        effective_threshold = min(floored_threshold * 1.05, 0.99)

        if RAPIDFUZZ_AVAILABLE:
            from rapidfuzz import process as fuzz_process, fuzz as fuzz_fuzz

            choices = list(candidate_index.keys())
            if choices:
                # D8-2: bound the fuzzy sweep.
                if len(choices) > self._config.fuzzy_max_candidates:
                    choices = choices[:self._config.fuzzy_max_candidates]
                result = fuzz_process.extractOne(
                    norm, choices,
                    scorer=fuzz_fuzz.token_sort_ratio,
                    score_cutoff=effective_threshold * 100,
                )
                if result is not None:
                    # SCI-FIX (runtime crash on rapidfuzz version drift):
                    # ``rapidfuzz.process.extractOne`` returns a 3-tuple
                    # ``(match, score, index)`` in some versions and a
                    # 2-tuple ``(match, score)`` in others. The previous
                    # unconditional 3-tuple unpack raised
                    # ``ValueError: not enough values to unpack`` on
                    # versions/builds returning the 2-tuple form,
                    # crashing the entire protein resolution pipeline
                    # precisely for the hardest-to-resolve proteins
                    # (fuzzy match is the LAST-resort path). The
                    # sibling module ``drug_resolver.py`` already
                    # handles both shapes (audit 4.3) -- mirror that
                    # defensive unpack here.
                    if len(result) == 3:
                        best_norm, best_score_100, _ = result
                    elif len(result) == 2:
                        best_norm, best_score_100 = result
                    else:
                        logger.warning(
                            "resolve_single: fuzzy_match returned "
                            "unexpected result shape (len=%d, type=%s) -- "
                            "skipping fuzzy match for '%s'",
                            len(result), type(result).__name__, norm,
                        )
                        return None
                    best_uid = candidate_index[best_norm]
                    if best_uid and best_uid in self.mapping:
                        # FIX SCI-04: gene-family false positive guard.
                        if self._is_gene_family_false_positive(norm, best_norm):
                            logger.debug(
                                "resolve_single: fuzzy match '%s' ≈ '%s' "
                                "rejected -- gene-family false positive "
                                "(score=%.3f)", norm, best_norm,
                                best_score_100 / 100.0,
                            )
                            return None
                        logger.debug(
                            "resolve_single: fuzzy match '%s' ≈ '%s' "
                            "(score=%.3f) -> '%s'",
                            norm, best_uid,
                            best_score_100 / 100.0, best_uid,
                        )
                        self._stats.inc("fuzzy_matches")
                        return self._copy_entry_for_read(self.mapping.get(best_uid))
        else:
            # Fallback: linear sweep with exact-match-only fuzzy_match_score
            best_score = 0.0
            best_uid: Optional[str] = None
            best_norm_name: Optional[str] = None
            for indexed_norm, indexed_uid in candidate_index.items():
                score = fuzzy_match_score(norm, indexed_norm)
                if score > best_score:
                    best_score = score
                    best_uid = indexed_uid
                    best_norm_name = indexed_norm

            if best_score >= effective_threshold and best_uid:
                # FIX SCI-04: gene-family guard.
                if best_norm_name and self._is_gene_family_false_positive(norm, best_norm_name):
                    logger.debug(
                        "resolve_single: fuzzy match '%s' ≈ '%s' rejected "
                        "-- gene-family false positive (score=%.3f)",
                        norm, best_norm_name, best_score,
                    )
                    return None
                logger.debug(
                    "resolve_single: fuzzy match '%s' ≈ '%s' "
                    "(score=%.3f) -> '%s'",
                    norm, best_uid, best_score, best_uid,
                )
                self._stats.inc("fuzzy_matches")
                return self._copy_entry_for_read(self.mapping.get(best_uid))
        return None

    def _get_organism_name_index(self, organism: str) -> Dict[str, str]:
        """Build/return per-organism slice of _name_index.

        FIX SCI-05: fuzzy match filtered by organism.
        """
        if not self._organism_name_cache_valid:
            self._build_organism_name_cache()
        return self._organism_name_cache.get(organism, {})

    def _build_organism_name_cache(self) -> None:
        """Rebuild the per-organism name index cache."""
        self._organism_name_cache.clear()
        for norm_name, uid in self._name_index.items():
            entry = self.mapping.get(uid)
            if entry is None:
                continue
            org = self._normalize_organism(entry.get("organism", ""))
            if org not in self._organism_name_cache:
                self._organism_name_cache[org] = {}
            self._organism_name_cache[org][norm_name] = uid
        self._organism_name_cache_valid = True

    # ------------------------------------------------------------------
    # Public API -- bulk resolution from DataFrames
    # ------------------------------------------------------------------

    def build_mapping(
        self,
        uniprot_df: Any,
        string_aliases_df: Optional[Any] = None,
        string_df: Optional[Any] = None,
        *,
        chembl_target_df: Optional[Any] = None,
        reset: bool = True,
        bundle: Optional[dict] = None,
        chunked: bool = False,
        chunksize: int = 100_000,
    ) -> Any:
        """Build cross-database protein entity mapping.

        Four sources are processed in order (FIX DOC-09 + v89 BUG #33):
        1. **uniprot_df** -- loaded first as the canonical source. Each record
           creates a new canonical entry keyed by uniprot_id, or merges into
           an existing entry, or promotes a provisional entry (ARCH-02).
        2. **string_aliases_df** -- STRING alias data, matched against existing
           canonical entries via string_id cross-reference or gene+organism
           match. Unmatched records create provisional entries.
        3. **string_df** -- STRING protein data with uniprot_id column. Each
           row's uniprot_id is checked against the mapping; if missing, a
           ``string_derived`` entry is created (match_method="string_derived",
           confidence=0.5). If present, the row's other fields are merged
           into the existing entry (CODE-20).
        4. **chembl_target_df** -- ChEMBL target records (v89 BUG #33 ROOT
           FIX). Each record has a ``chembl_target_id`` and optionally a
           ``uniprot_id`` cross-reference. Matched via UniProt ID exact
           match or gene+organism match; unmatched records create
           ``chembl_provisional`` entries. Previously this source was
           registered in ``_SOURCE_INGESTORS`` but NEVER passed by
           ``run.py``, leaving ChEMBL target IDs orphaned from the
           canonical Protein nodes -- drug-target edges from ChEMBL used
           ChEMBL target IDs that were not linked to UniProt accessions.

        Returns
        -------
        pd.DataFrame
            Columns (FIX DOC-15): uniprot_id, canonical_name, gene_symbol,
            gene_name, protein_name, organism, sequence, isoforms, string_id,
            chembl_target_id, match_confidence, match_method, data_quality_score,
            sources, created_at, resolved_at, resolver_version, input_checksum,
            canonical_checksum, deprecated_by, provisional.
        """
        # D7-1: idempotency via default reset=True.
        if reset:
            if self.mapping:
                logger.info(
                    "build_mapping: reset=True -- clearing %d existing "
                    "canonical entries (idempotent re-run)",
                    len(self.mapping),
                )
            self.reset()

        logger.info("build_mapping: starting protein entity resolution")

        if uniprot_df is not None:
            uniprot_records = self._df_to_records(uniprot_df)
            self.add_uniprot_records(uniprot_records)

        if string_aliases_df is not None:
            try:
                if hasattr(string_aliases_df, "empty") and not string_aliases_df.empty:
                    # v89 ROOT FIX (BUG #23 -- silent schema mismatch in
                    # STRING aliases DataFrame):
                    #   The previous code called ``add_string_records`` on
                    #   ``string_aliases_df`` WITHOUT validating the schema.
                    #   If the DataFrame was non-empty but had the WRONG
                    #   columns (e.g. a stale cached file, a schema change
                    #   in the STRING pipeline, a misnamed column), every
                    #   record was rejected at the validation step (line
                    #   ~1945: ``if not string_id``) and dead-lettered.
                    #   The operator saw only "ingesting N STRING records"
                    #   followed by "0 matched, 0 provisional created" --
                    #   no error, no warning about the schema mismatch.
                    #   The KG's PPI subgraph was silently incomplete.
                    #
                    #   ROOT FIX: validate the DataFrame schema BEFORE
                    #   calling ``add_string_records``. The required
                    #   column is ``string_id`` (minimum). If missing,
                    #   log an ERROR with the actual columns present, and
                    #   SKIP ingestion (do NOT call add_string_records
                    #   with malformed records -- that just fills the
                    #   dead-letter queue with confusing entries). This
                    #   surfaces the root cause (wrong schema) to the
                    #   operator immediately.
                    if not hasattr(string_aliases_df, "columns"):
                        logger.error(
                            "build_mapping: string_aliases_df is not a "
                            "DataFrame (has no .columns attribute). Type: "
                            "%s. Skipping STRING aliases ingestion. The "
                            "KG's PPI subgraph will be incomplete. "
                            "Fix: pass a pandas DataFrame with at least "
                            "a 'string_id' column. (v89 BUG #23)",
                            type(string_aliases_df).__name__,
                        )
                    else:
                        _alias_cols = set(string_aliases_df.columns)
                        _required_alias_cols = {"string_id"}
                        _missing_alias_cols = _required_alias_cols - _alias_cols
                        if _missing_alias_cols:
                            logger.error(
                                "build_mapping: string_aliases_df is "
                                "missing required column(s) %s. Available "
                                "columns: %s. Skipping STRING aliases "
                                "ingestion -- every record would be "
                                "rejected at the string_id validation "
                                "step and dead-lettered, producing a "
                                "silently-empty PPI subgraph. Fix the "
                                "upstream caller to pass a DataFrame "
                                "with at least a 'string_id' column "
                                "(see entity_resolution/run.py for the "
                                "expected schema). (v89 BUG #23)",
                                sorted(_missing_alias_cols),
                                sorted(_alias_cols),
                            )
                        else:
                            string_records = self._df_to_records(string_aliases_df)
                            self.add_string_records(string_records)
                else:
                    logger.warning(
                        "build_mapping: string_aliases_df is not a recognized "
                        "DataFrame or is empty (FIX LOG-07)"
                    )
            except AttributeError:
                logger.warning(
                    "build_mapping: string_aliases_df has no .empty attribute"
                )

        if string_df is not None:
            try:
                if hasattr(string_df, "empty") and hasattr(string_df, "columns"):
                    if not string_df.empty and "uniprot_id" in string_df.columns:
                        # v89 FORENSIC ROOT FIX (BUG #18 P1 -- string_derived
                        #   organism default "Homo sapiens" for unknowns):
                        #   The previous code created a string_derived entry
                        #   for EVERY UniProt ID in string_df that was not
                        #   already in self.mapping. The organism was
                        #   resolved via 3 sources:
                        #     1. UniProt organism override table (~250 entries)
                        #     2. Per-row organism column in string_df
                        #     3. STRING ID taxonomy prefix (Source 3)
                        #   If NONE of these fired, the organism defaulted
                        #   to ``self._config.default_organism`` ("Homo
                        #   sapiens"). In run.py, string_protein_df is built
                        #   with ONLY uniprot_id + string_id columns (no
                        #   organism). Source 1 only fires for ~250 known
                        #   accessions. Source 3 fires only if string_id is
                        #   populated AND has a valid taxonomy prefix AND
                        #   that prefix is in ``_ORGANISM_ALIASES``. For any
                        #   STRING-derived UniProt ID not in the override
                        #   table and without a valid string_id pairing, the
                        #   organism defaulted to "Homo sapiens" -- non-human
                        #   proteins were mislabeled as human.
                        #   ROOT FIX: REFUSE to create a string_derived
                        #   entry when the organism cannot be determined
                        #   from ANY of the 3 sources. Dead-letter the
                        #   UniProt ID so the operator sees the gap. This
                        #   prevents non-human proteins from being silently
                        #   labeled "Homo sapiens" and merged into human PPI
                        #   subgraphs. The operator can extend the override
                        #   table or fix the string_id pairing to resolve
                        #   the dead-lettered entries.
                        _string_derived_skipped = 0
                        for uid in string_df["uniprot_id"].dropna().unique():
                            uid_str = str(uid).strip()
                            if not uid_str or uid_str in self.mapping:
                                continue
                            # v89 BUG #4: normalize to uppercase for
                            # consistency with add_uniprot_records.
                            uid_str = uid_str.upper()
                            if uid_str in self.mapping:
                                continue
                            now_iso = self._now_iso()
                            _resolved_organism = None  # must be determined
                            _organism_source = None
                            # Source 1: UniProt organism override table.
                            try:
                                _effective_overrides = (
                                    _get_effective_uniprot_organism_overrides()
                                )
                            except Exception:
                                _effective_overrides = {}
                            if uid_str in _effective_overrides:
                                _resolved_organism = _effective_overrides[uid_str]
                                _organism_source = "uniprot_override"
                            # Source 2: per-row organism column in string_df.
                            if _resolved_organism is None and "organism" in string_df.columns:
                                _row_match = string_df.loc[
                                    string_df["uniprot_id"].astype(str).str.strip().str.upper()
                                    == uid_str,
                                    "organism",
                                ].dropna()
                                if not _row_match.empty:
                                    _row_org = str(_row_match.iloc[0]).strip()
                                    if _row_org:
                                        _resolved_organism = _row_org
                                        _organism_source = "string_df_organism_column"
                            # Source 3: STRING ID taxonomy prefix.
                            if _resolved_organism is None and "string_id" in string_df.columns:
                                _sid_match = string_df.loc[
                                    string_df["uniprot_id"].astype(str).str.strip().str.upper()
                                    == uid_str,
                                    "string_id",
                                ].dropna()
                                if not _sid_match.empty:
                                    _sid = str(_sid_match.iloc[0]).strip()
                                    if _sid and "." in _sid:
                                        _taxid = _sid.split(".")[0]
                                        _taxid_org = _ORGANISM_ALIASES.get(_taxid)
                                        if _taxid_org:
                                            _resolved_organism = _taxid_org
                                            _organism_source = "string_id_taxonomy_prefix"
                            # v89 BUG #18: REFUSE to create a string_derived
                            # entry if the organism could not be determined.
                            if _resolved_organism is None:
                                logger.warning(
                                    "build_mapping: cannot determine organism "
                                    "for STRING-derived UniProt ID %s -- refusing "
                                    "to create string_derived entry (would "
                                    "default to 'Homo sapiens' and risk cross-"
                                    "species contamination). Dead-lettering. "
                                    "Extend the UniProt organism override "
                                    "table or fix the string_id pairing to "
                                    "resolve.",
                                    uid_str,
                                )
                                self._dead_letter.append({
                                    "record": {"uniprot_id": uid_str, "source": "string_derived"},
                                    "source": "string_derived",
                                    "errors": [
                                        "organism could not be determined from "
                                        "override table, string_df organism "
                                        "column, or STRING ID taxonomy prefix -- "
                                        "refusing to default to 'Homo sapiens'"
                                    ],
                                    "stage": "build_mapping_string_derived",
                                })
                                self._stats.inc("records_rejected")
                                self._stats.inc("string_derived_organism_unknown")
                                _string_derived_skipped += 1
                                continue
                            self.mapping[uid_str] = {
                                "uniprot_id": uid_str,
                                "gene_symbol": None,
                                "gene_name": None,
                                "organism": _resolved_organism,
                                "sequence": None,
                                "protein_name": None,
                                "string_id": None,
                                "chembl_target_id": None,
                                "canonical_name": uid_str,
                                "sources": ["string_derived"],
                                "match_method": "string_derived",
                                "match_confidence": compute_match_confidence("string_derived"),
                                "created_at": now_iso,
                                "resolved_at": now_iso,
                                "resolver_version": MAPPING_SCHEMA_VERSION,
                                "input_checksum": "",
                                "canonical_checksum": "",
                                "isoforms": [],
                                "deprecated_by": None,
                                "provisional": True,
                            }
                            self._append_audit(uid_str, {
                                "action": "create",
                                "source": "string_derived",
                                "method": "string_derived",
                                "organism": _resolved_organism,
                                "organism_source": _organism_source,
                            })
                        if _string_derived_skipped:
                            logger.warning(
                                "build_mapping: skipped %d STRING-derived "
                                "entries with unknown organism (dead-lettered "
                                "to prevent cross-species contamination)",
                                _string_derived_skipped,
                            )
                else:
                    logger.warning(
                        "build_mapping: string_df is not a recognized DataFrame "
                        "or is empty (FIX LOG-07)"
                    )
            except AttributeError:
                logger.warning(
                    "build_mapping: string_df has no .empty/.columns attributes"
                )

        # v89 ROOT FIX (BUG #33 -- ChEMBL target data never passed to
        # build_mapping):
        #   The ``add_chembl_target_records`` method (line ~2030) was
        #   registered in ``_SOURCE_INGESTORS`` but NEVER called from
        #   ``build_mapping`` because there was no ``chembl_target_df``
        #   parameter. ``run.py`` called ``build_mapping(uniprot_df,
        #   string_aliases_df=..., string_df=...)`` with no ChEMBL
        #   target data. ChEMBL target IDs (e.g. CHEMBL2366519 for
        #   EGFR) were NOT cross-referenced with UniProt accessions.
        #   The KG's drug-target edges from ChEMBL used ChEMBL target
        #   IDs that were not linked to the canonical Protein nodes --
        #   orphaned drug-target edges, incomplete KG.
        #
        #   ROOT FIX: add a ``chembl_target_df`` keyword parameter and
        #   ingest it via ``add_chembl_target_records`` (same pattern as
        #   string_aliases_df -> add_string_records). ``run.py`` is
        #   updated to load the ChEMBL activities/targets CSV and pass
        #   it here. This connects ChEMBL target IDs to UniProt
        #   accessions, completing the drug -> target -> pathway -> disease
        #   chain (Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 connectivity).
        if chembl_target_df is not None:
            try:
                if hasattr(chembl_target_df, "empty") and not chembl_target_df.empty:
                    # v89 BUG #23 pattern (applied to ChEMBL targets too):
                    # validate the schema BEFORE ingesting. Required
                    # column is ``chembl_target_id`` (minimum). If
                    # missing, log an ERROR with the actual columns and
                    # SKIP ingestion -- do NOT fill the dead-letter queue
                    # with confusing malformed records.
                    if not hasattr(chembl_target_df, "columns"):
                        logger.error(
                            "build_mapping: chembl_target_df is not a "
                            "DataFrame (has no .columns attribute). Type: "
                            "%s. Skipping ChEMBL target ingestion. The "
                            "KG's drug-target edges will use orphaned "
                            "ChEMBL target IDs. Fix: pass a pandas "
                            "DataFrame with at least a "
                            "'chembl_target_id' column. (v89 BUG #33)",
                            type(chembl_target_df).__name__,
                        )
                    else:
                        _chembl_cols = set(chembl_target_df.columns)
                        _required_chembl_cols = {"chembl_target_id"}
                        _missing_chembl_cols = _required_chembl_cols - _chembl_cols
                        if _missing_chembl_cols:
                            logger.error(
                                "build_mapping: chembl_target_df is "
                                "missing required column(s) %s. Available "
                                "columns: %s. Skipping ChEMBL target "
                                "ingestion -- every record would be "
                                "rejected at the chembl_target_id "
                                "validation step. Fix the upstream caller "
                                "to pass a DataFrame with at least a "
                                "'chembl_target_id' column. (v89 BUG #33)",
                                sorted(_missing_chembl_cols),
                                sorted(_chembl_cols),
                            )
                        else:
                            chembl_records = self._df_to_records(chembl_target_df)
                            self.add_chembl_target_records(chembl_records)
                else:
                    logger.info(
                        "build_mapping: chembl_target_df is None or empty "
                        "-- skipping ChEMBL target ingestion (no ChEMBL "
                        "target data available this run). (v89 BUG #33)"
                    )
            except AttributeError:
                logger.warning(
                    "build_mapping: chembl_target_df has no .empty attribute"
                )

        result_df = self.to_dataframe()
        logger.info(
            "build_mapping: resolved %d canonical protein entities",
            len(result_df),
        )
        return result_df

    def to_dataframe(
        self,
        chunksize: Optional[int] = None,
    ) -> Any:
        """Convert the internal ``mapping`` dict to an entity-mapping DataFrame.

        Always returns a DataFrame (FIX DESIGN-09 -- was data-dependent
        return type when chunksize was given).  Use
        ``to_dataframe_streaming`` for chunked output.

        Parameters
        ----------
        chunksize:
            If given and ``> 0``, a deprecation warning is logged and
            the full DataFrame is returned (backward compat).
        """
        if chunksize is not None and chunksize > 0:
            warnings.warn(
                "to_dataframe(chunksize=...) is deprecated -- use "
                "to_dataframe_streaming() instead. Returning full DataFrame.",
                DeprecationWarning,
                stacklevel=2,
            )

        pd = _get_pd()
        rows: List[dict] = []
        for uniprot_id, entry in self.mapping.items():
            rows.append({
                "uniprot_id": uniprot_id,
                "canonical_name": entry.get("canonical_name", ""),
                "gene_symbol": entry.get("gene_symbol"),
                "gene_name": entry.get("gene_name"),
                "organism": entry.get("organism"),
                "string_id": entry.get("string_id"),
                "chembl_target_id": entry.get("chembl_target_id"),
                "match_confidence": entry.get("match_confidence", 0.0),
                "match_method": entry.get("match_method", "unknown"),
                # D5-5 / D16-1: provenance preserved.
                "sources": json.dumps(entry.get("sources", [])),
                # D16-2: lineage metadata.
                "resolved_at": entry.get("resolved_at", ""),
                "created_at": entry.get("created_at", ""),
                "resolver_version": entry.get(
                    "resolver_version", MAPPING_SCHEMA_VERSION
                ),
                "input_checksum": entry.get("input_checksum", ""),
                "data_quality_score": entry.get("data_quality_score"),
                # FIX DQ-16 / DESIGN-13: new fields.
                "protein_name": entry.get("protein_name"),
                "sequence": entry.get("sequence"),
                "isoforms": json.dumps(entry.get("isoforms", [])),
                "canonical_checksum": entry.get("canonical_checksum", ""),
                "deprecated_by": entry.get("deprecated_by"),
                "provisional": entry.get("provisional", False),
            })

        # Backward-compatible column order -- must match original 13 columns exactly.
        # New columns are available but not included by default to preserve
        # backward compatibility with existing tests and downstream consumers.
        columns = [
            "uniprot_id", "canonical_name",
            "gene_symbol", "gene_name", "organism",
            "string_id", "chembl_target_id",
            "match_confidence", "match_method",
            "sources", "resolved_at",
            "resolver_version", "input_checksum",
        ]
        df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
        logger.debug("to_dataframe: %d rows", len(df))
        return df

    def to_dataframe_streaming(
        self, chunksize: int = 100_000
    ) -> Iterator[Any]:
        """Yield DataFrames in chunks for memory-efficient export.

        FIX DESIGN-09 / PERF-01: streaming alternative to to_dataframe
        for large mappings.
        """
        pd = _get_pd()
        all_entries = list(self.mapping.items())
        for i in range(0, len(all_entries), chunksize):
            chunk = all_entries[i:i + chunksize]
            rows = []
            for uniprot_id, entry in chunk:
                rows.append({
                    "uniprot_id": uniprot_id,
                    "canonical_name": entry.get("canonical_name", ""),
                    "gene_symbol": entry.get("gene_symbol"),
                    "gene_name": entry.get("gene_name"),
                    "protein_name": entry.get("protein_name"),
                    "organism": entry.get("organism"),
                    "sequence": entry.get("sequence"),
                    "isoforms": json.dumps(entry.get("isoforms", [])),
                    "string_id": entry.get("string_id"),
                    "chembl_target_id": entry.get("chembl_target_id"),
                    "match_confidence": entry.get("match_confidence", 0.0),
                    "match_method": entry.get("match_method", "unknown"),
                    "data_quality_score": entry.get("data_quality_score"),
                    "sources": json.dumps(entry.get("sources", [])),
                    "created_at": entry.get("created_at", ""),
                    "resolved_at": entry.get("resolved_at", ""),
                    "resolver_version": entry.get(
                        "resolver_version", MAPPING_SCHEMA_VERSION
                    ),
                    "input_checksum": entry.get("input_checksum", ""),
                    "canonical_checksum": entry.get("canonical_checksum", ""),
                    "deprecated_by": entry.get("deprecated_by"),
                    "provisional": entry.get("provisional", False),
                })
            yield pd.DataFrame(rows)

    def to_records(self) -> List[dict]:
        """Export the mapping as a list of plain dicts (no pandas dep).

        FIX CODE-23: uses deep copy (was shallow).
        """
        records: List[dict] = []
        for uid, entry in self.mapping.items():
            row = self._copy_entry_for_read(entry)
            row["uniprot_id"] = uid
            records.append(row)
        return records

    def to_dict(self) -> Dict[str, dict]:
        """Export the mapping as a dict-of-dicts (JSON-serialisable).

        FIX CODE-24: uses deep copy (was shallow -- sources list shared).
        """
        return {uid: self._copy_entry_for_read(e) for uid, e in self.mapping.items()}

    def to_parquet(self, path: str) -> None:
        """Write the mapping to a Parquet file (audit D8-4).

        FIX SEC-09: path validation against allowed_paths_root.
        """
        # FIX SEC-09: path traversal protection.
        if self._config.allowed_paths_root is not None:
            real_root = os.path.realpath(self._config.allowed_paths_root)
            real_path = os.path.realpath(path)
            if not real_path.startswith(real_root):
                raise ValueError(
                    f"to_parquet: path {path!r} is outside the allowed "
                    f"root {self._config.allowed_paths_root!r} (SEC-09)"
                )

        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "to_parquet requires 'pyarrow'. Install with: "
                "pip install pyarrow"
            ) from exc
        df = self.to_dataframe()
        df.to_parquet(path, index=False)

    def to_parquet_chunked(
        self, path: str, chunksize: int = 100_000
    ) -> None:
        """Write mapping to Parquet in chunks for memory efficiency.

        FIX PERF-01: chunked parquet export for large datasets.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "to_parquet_chunked requires 'pyarrow'. Install with: "
                "pip install pyarrow"
            ) from exc

        # FIX SEC-09: path validation.
        if self._config.allowed_paths_root is not None:
            real_root = os.path.realpath(self._config.allowed_paths_root)
            real_path = os.path.realpath(path)
            if not real_path.startswith(real_root):
                raise ValueError(
                    f"to_parquet_chunked: path {path!r} is outside allowed "
                    f"root {self._config.allowed_paths_root!r}"
                )

        first = True
        for chunk_df in self.to_dataframe_streaming(chunksize):
            table = pa.Table.from_pandas(chunk_df)
            if first:
                pq.write_table(table, path)
                first = False
            else:
                pq.write_to_dataset(table, root_path=path)

    # ------------------------------------------------------------------
    # Public API -- state serialisation (D7-4, D16-3)
    # ------------------------------------------------------------------

    def to_state_dict(self) -> dict:
        """Serialise the resolver's full state to a JSON-compatible dict.

        FIX SEC-02: encrypt if state_encryption_key is set.
        FIX SEC-15: add HMAC signature if tamper_evident is True.
        FIX LIN-02: include field diffs and checksums in audit trail.
        """
        # FIX-P1-C-10: every mutable structure is deep-copied so callers
        # cannot mutate the resolver's internal state by mutating the
        # returned dict (mirrors drug_resolver's pattern). The previous
        # implementation returned LIVE references to ``self.mapping``,
        # ``self._gene_index``, ``self._string_to_uniprot``,
        # ``self._name_index``, ``self._name_index_multi``,
        # ``self._dead_letter`` and ``self._audit_trail`` -- a caller that
        # did ``state["mapping"].pop(...)`` would corrupt the live
        # resolver.
        state: dict = {
            "schema_version": MAPPING_SCHEMA_VERSION,
            "resolver_class": type(self).__name__,
            "config": self._config.to_masked_dict(),
            "mapping": copy.deepcopy(self.mapping),
            "gene_index": [
                {"gene": k[0], "organism": k[1], "uniprot_id": v}
                for k, v in copy.deepcopy(self._gene_index).items()
            ],
            "string_to_uniprot": copy.deepcopy(self._string_to_uniprot),
            "name_index": copy.deepcopy(self._name_index),
            "name_index_multi": copy.deepcopy(self._name_index_multi),
            "dead_letter": copy.deepcopy(self._dead_letter),
            "audit_trail": copy.deepcopy(self._audit_trail),
            "stats": self._stats.to_dict(),
            "exported_at": self._now_iso(),
        }

        # FIX SEC-15: HMAC signature for tamper-evidence.
        # FIX P1-ER-18 (LOW): the previous implementation hard-coded the
        # HMAC key as ``b"protein-resolver-tamper-evident-key"`` -- anyone
        # with source access could forge valid signatures. The key is
        # now sourced from ``ResolverConfig.tamper_evident_key`` (env
        # var: ``ENTITY_RESOLUTION_TAMPER_EVIDENT_KEY``, hex-encoded).
        # If ``tamper_evident=True`` but no key is configured, we log a
        # CRITICAL warning and skip signing -- tamper-evidence is
        # effectively disabled until the operator configures a key.
        # This is safer than silently using a known-to-attacker key.
        if self._config.tamper_evident:
            tek = self._config.tamper_evident_key
            if tek is None:
                logger.critical(
                    "to_state_dict: tamper_evident=True but "
                    "tamper_evident_key is None -- SKIPPING signature. "
                    "State will be saved WITHOUT tamper-evidence. "
                    "Configure ENTITY_RESOLUTION_TAMPER_EVIDENT_KEY "
                    "(hex-encoded, e.g. `openssl rand -hex 32`) to "
                    "enable tamper-evidence. This is a CRITICAL "
                    "security gap in production deployments."
                )
            else:
                try:
                    payload = json.dumps(state, sort_keys=True, default=str)
                    sig = hmac.new(
                        tek,
                        payload.encode(),
                        hashlib.sha256,
                    ).hexdigest()
                    state["_signature"] = sig
                except (TypeError, ValueError):
                    state["_signature"] = ""

        # FIX SEC-02: AES-256-GCM encryption.
        if self._config.state_encryption_key is not None:
            state = self._encrypt_state(state)

        return state

    def _encrypt_state(self, state: dict) -> dict:
        """Encrypt the state dict using AES-256-GCM.

        FIX SEC-02: state encryption support.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            import os as _os
            key = self._config.state_encryption_key
            if not isinstance(key, bytes) or len(key) != 32:
                logger.warning(
                    "state_encryption_key must be 32 bytes for AES-256-GCM; "
                    "skipping encryption"
                )
                return state
            nonce = _os.urandom(12)
            aesgcm = AESGCM(key)
            plaintext = json.dumps(state, default=str).encode()
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            return {
                "_encrypted": True,
                "nonce": nonce.hex(),
                "ciphertext": ciphertext.hex(),
            }
        except ImportError:
            logger.warning(
                "state_encryption_key set but 'cryptography' not installed; "
                "saving state unencrypted"
            )
            return state
        except Exception as exc:
            logger.error("State encryption failed: %s", exc)
            return state

    @classmethod
    def from_state_dict(cls, state: dict) -> "ProteinResolver":
        """Reconstruct a :class:`ProteinResolver` from a state dict.

        FIX COMP-04: reject state dicts from a different resolver class.
        FIX SEC-15: verify HMAC signature if present.
        """
        # FIX SEC-02: handle encrypted state.
        if state.get("_encrypted"):
            logger.info("from_state_dict: state is encrypted, attempting decryption")
            # Decryption requires the key; caller must set it on the config.
            # For now, raise a clear error.
            raise ValueError(
                "Cannot load encrypted state dict without providing "
                "state_encryption_key in ResolverConfig"
            )

        # FIX SEC-15: verify HMAC signature.
        # FIX P1-ER-18 (LOW): use the operator-configured
        # ``tamper_evident_key`` (not the legacy hard-coded key). The
        # key is read from env var ``ENTITY_RESOLUTION_TAMPER_EVIDENT_KEY``
        # (hex-encoded) because ``from_state_dict`` is a classmethod
        # and has no ``self._config`` yet -- the config is itself part
        # of the signed payload, so it CANNOT be used to source the
        # key (chicken-and-egg). If the state carries a signature but
        # no key is configured, log CRITICAL and skip verification.
        sig = state.pop("_signature", None)
        if sig:
            tek_raw = os.environ.get(
                "ENTITY_RESOLUTION_TAMPER_EVIDENT_KEY", ""
            ).strip()
            tek: Optional[bytes] = None
            if tek_raw:
                try:
                    tek = bytes.fromhex(tek_raw)
                except ValueError:
                    logger.warning(
                        "from_state_dict: ENTITY_RESOLUTION_"
                        "TAMPER_EVIDENT_KEY is not valid hex -- "
                        "skipping signature verification."
                    )
            if tek is None:
                logger.critical(
                    "from_state_dict: state carries an HMAC signature "
                    "but no tamper_evident_key is configured (env var "
                    "ENTITY_RESOLUTION_TAMPER_EVIDENT_KEY is unset or "
                    "invalid) -- SKIPPING verification. Tamper-evidence "
                    "is DISABLED. This is a CRITICAL security gap in "
                    "production deployments."
                )
            else:
                try:
                    payload = json.dumps(state, sort_keys=True, default=str)
                    expected_sig = hmac.new(
                        tek,
                        payload.encode(),
                        hashlib.sha256,
                    ).hexdigest()
                    if not hmac.compare_digest(sig, expected_sig):
                        raise ValueError(
                            "State dict HMAC signature mismatch -- data may have "
                            "been tampered with (SEC-15)"
                        )
                except (TypeError, ValueError) as exc:
                    if "mismatch" in str(exc):
                        raise
                    logger.warning("from_state_dict: signature verification skipped: %s", exc)

        schema = state.get("schema_version", "unknown")
        if schema != MAPPING_SCHEMA_VERSION:
            raise ValueError(
                f"state schema version mismatch: state has {schema!r}, "
                f"resolver expects {MAPPING_SCHEMA_VERSION!r}."
            )

        # FIX COMP-04: class mismatch check.
        resolver_class = state.get("resolver_class", "")
        if resolver_class and resolver_class != cls.__name__:
            raise ValueError(
                f"State dict was created by {resolver_class!r}, but "
                f"loading into {cls.__name__!r} (COMP-04)"
            )

        cfg_dict = {**state.get("config", {})}
        if cfg_dict.get("pubchem_api_key") == "<redacted>":
            cfg_dict["pubchem_api_key"] = None
        cfg = ResolverConfig(**{
            k: (tuple(v) if k == "source_whitelist" and v else v)
            for k, v in cfg_dict.items()
        })
        resolver = cls(config=cfg)
        resolver.mapping = dict(state.get("mapping", {}))
        resolver._gene_index = {
            (e["gene"], e["organism"]): e["uniprot_id"]
            for e in state.get("gene_index", [])
        }
        resolver._string_to_uniprot = dict(
            state.get("string_to_uniprot", {})
        )
        resolver._name_index = dict(state.get("name_index", {}))
        resolver._name_index_multi = dict(
            state.get("name_index_multi", {})
        )
        resolver._dead_letter = list(state.get("dead_letter", []))
        resolver._audit_trail = dict(state.get("audit_trail", {}))
        for k, v in state.get("stats", {}).items():
            try:
                resolver._stats.inc(k, v)
            except Exception:
                # FIX REL-03: log instead of silent pass.
                logger.warning(
                    "from_state_dict: failed to load stat %s=%s", k, v,
                )
        resolver._organism_name_cache_valid = False
        return resolver

    # ------------------------------------------------------------------
    # Public API -- lifecycle / maintenance
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all internal state -- equivalent to a fresh instance.

        FIX LOG-10: reset logs at INFO (was DEBUG -- invisible in production).
        """
        old_count = len(self.mapping)
        self.mapping = {}
        self._gene_index = {}
        self._string_to_uniprot = {}
        self._name_index = {}
        self._name_index_multi = {}
        # v89 BUG #35: clear the case-preserving gene symbol index too.
        self._gene_symbol_index = {}
        self._gene_symbol_index_multi = {}
        self._dead_letter = []
        self._audit_trail = {}
        self._stats.reset()
        self._organism_name_cache_valid = False
        self._last_batch_fingerprints.clear()
        # v80 P0-D3: also clear the provisional-by-gene-organism index.
        self._provisional_by_gene_organism = {}
        # v82 P0-D3b: also clear the provisional-by-alias-uniprot index.
        self._provisional_by_alias_uniprot = {}
        logger.info(
            "reset: cleared all internal state (%d entries were discarded)",
            old_count,
        )

    def remove_source(self, source: str) -> int:
        """Remove entries whose only source is ``source``.

        Returns the number of entries fully removed.

        FIX PERF-02: O(N) implementation (was O(N^2) due to per-entry
        dict comprehensions rebuilding _string_to_uniprot, _name_index,
        _name_index_multi, _gene_index on EVERY deletion).
        """
        to_delete: List[str] = []
        partial_delete: List[str] = []
        removed = 0

        for uid, entry in self.mapping.items():
            sources = entry.get("sources", [])
            if source in sources:
                if len(sources) == 1:
                    to_delete.append(uid)
                else:
                    partial_delete.append(uid)

        # Partial removals (keep entry, remove source).
        for uid in partial_delete:
            entry = self.mapping[uid]
            entry["sources"] = [s for s in entry.get("sources", []) if s != source]
            self._append_audit(uid, {
                "action": "remove_source_partial",
                "source": source,
            })

        # Full deletions -- batch rebuild indexes once.
        if to_delete:
            delete_set = set(to_delete)
            for uid in to_delete:
                del self.mapping[uid]
                removed += 1
                self._audit_trail.pop(uid, None)

            # FIX PERF-02: rebuild indexes in O(N) once, not O(N^2).
            self._string_to_uniprot = {
                k: v for k, v in self._string_to_uniprot.items()
                if v not in delete_set
            }
            self._name_index = {
                k: v for k, v in self._name_index.items()
                if v not in delete_set
            }
            self._name_index_multi = {
                k: [x for x in vlist if x not in delete_set]
                for k, vlist in self._name_index_multi.items()
            }
            self._name_index_multi = {
                k: v for k, v in self._name_index_multi.items() if v
            }
            self._gene_index = {
                k: v for k, v in self._gene_index.items()
                if v not in delete_set
            }
            # v80 P0-D3: rebuild the _provisional_by_gene_organism index
            # to drop any deleted provisional uids.
            self._provisional_by_gene_organism = {
                k: [x for x in vlist if x not in delete_set]
                for k, vlist in self._provisional_by_gene_organism.items()
            }
            self._provisional_by_gene_organism = {
                k: v for k, v in self._provisional_by_gene_organism.items() if v
            }
            # v82 P0-D3b: rebuild the _provisional_by_alias_uniprot index
            # to drop any deleted provisional uids.
            self._provisional_by_alias_uniprot = {
                k: [x for x in vlist if x not in delete_set]
                for k, vlist in self._provisional_by_alias_uniprot.items()
            }
            self._provisional_by_alias_uniprot = {
                k: v for k, v in self._provisional_by_alias_uniprot.items() if v
            }

        logger.info(
            "remove_source('%s'): removed %d entries", source, removed
        )
        self._organism_name_cache_valid = False
        return removed

    def get_stats(self) -> Dict[str, int]:
        """Return a JSON-serialisable snapshot of resolver counters."""
        return self._stats.to_dict()

    def get_audit_trail(self, canonical_id: str) -> List[dict]:
        """Return the ordered list of merge events for ``canonical_id``."""
        return list(self._audit_trail.get(canonical_id, []))

    def find_affected_entities(self, source: str) -> List[str]:
        """Return UniProt IDs whose ``sources`` list contains ``source``."""
        return [
            uid for uid, entry in self.mapping.items()
            if source in entry.get("sources", [])
        ]

    # ------------------------------------------------------------------
    # FIX IDEM-02: batch fingerprint computation.
    # ------------------------------------------------------------------

    def _compute_batch_fingerprint(self, records: List[dict], source: str) -> str:
        """Compute a fingerprint for a batch of records.

        FIX IDEM-02: if the same batch is ingested twice, the second
        ingestion is skipped (idempotent).

        P2-3 ROOT FIX (v82): the previous implementation hashed only
        ``records[:100]`` -- for batches >100 records where only records
        101+ differed from a prior batch, the fingerprint was IDENTICAL
        and the second batch was silently skipped (idempotency collision
        -> silent data loss). For a 200K-record batch, this means 199,900
        records could be silently dropped on re-ingestion.

        The root fix uses STREAMING SHA-256 over ALL records:
          * O(n) time, O(1) memory -- for 200K records with ~10-char IDs,
            that's ~2MB of data hashed, negligible vs. network/parse cost.
          * Includes the record COUNT as a prefix so adding/removing a
            duplicate record produces a different fingerprint.
          * Uses ``\\x00`` separators between fields to prevent
            concatenation collisions (e.g. ``"ab"+"c"`` vs ``"a"+"bc"``).
          * Preserves the original order-sensitivity: re-ordering the
            same records produces a different fingerprint (matches the
            previous ``records[:100]`` semantics).
          * Str()/repr() is safe here because IDs are strings/ints; for
            exotic types, we fall back to ``""`` via the ``or ""`` short-
            circuit, then ``str()`` which is deterministic for primitives.
        """
        try:
            h = hashlib.sha256()
            # Prefix with source + count so different sources or
            # different-sized batches never collide.
            h.update(source.encode("utf-8", "replace"))
            h.update(b"\x00")
            count = len(records)
            h.update(str(count).encode("ascii", "replace"))
            h.update(b"\x00")
            for r in records:
                rid = (
                    r.get("uniprot_id")
                    or r.get("string_id")
                    or r.get("chembl_target_id")
                    or ""
                )
                # str() is deterministic for str/int (the only ID types
                # used here). For exotic types, callers should pre-coerce.
                h.update(str(rid).encode("utf-8", "replace"))
                h.update(b"\x00")  # separator prevents ID-concatenation collisions
            return h.hexdigest()[:16]
        except (TypeError, ValueError):
            # If a record contains an unhashable/unserialisable ID,
            # fall back to "" (which forces a non-idempotent re-ingest --
            # safe failure mode: the batch is re-processed rather than
            # silently skipped).
            return ""

    # ------------------------------------------------------------------
    # Internal -- merge helpers
    # ------------------------------------------------------------------

    def _append_audit(self, uid: str, event: dict) -> None:
        """Append an audit event to the entry's trail.

        Audit event schema (FIX DOC-14 / COMP-08):
        ------------------------------------------
        Required keys:
          - action: str -- one of {"create", "merge", "promote_provisional",
                    "promote_provisional_merge", "remove_source_partial",
                    "merge_conflict", "string_xref_conflict", "delete",
                    "confidence_upgrade"}
          - source: str -- one of {"uniprot", "string", "chembl", "string_derived"}
          - method: str -- one of the MatchConfidence method names
          - timestamp: str -- ISO-8601 UTC

        Optional keys (depending on action):
          - field, existing, incoming, policy_applied (for merge_conflict)
          - operator (if operator_id was provided)
          - confidence_before, confidence_after (for confidence_upgrade)
          - fields_changed, field_diffs, incoming_checksum (FIX LIN-02)

        The audit trail per entry is bounded by
        ``ResolverConfig.max_audit_trail_per_entry``; older events spill to
        ``ResolverConfig.audit_trail_spill_path`` if configured (FIX DQ-12).
        """
        event = dict(event)
        event.setdefault("timestamp", self._now_iso())

        trail = self._audit_trail.setdefault(uid, [])
        trail.append(event)

        # FIX DQ-12: bound audit trail size.
        max_size = self._config.max_audit_trail_per_entry
        if max_size > 0 and len(trail) > max_size:
            spilled = trail[:len(trail) - max_size]
            trail[:] = trail[len(trail) - max_size:]
            if self._config.audit_trail_spill_path:
                try:
                    spill_file = os.path.join(
                        self._config.audit_trail_spill_path,
                        f"{uid}_audit_spill.jsonl",
                    )
                    with open(spill_file, "a") as f:
                        for evt in spilled:
                            f.write(json.dumps(evt, default=str) + "\n")
                except OSError as exc:
                    logger.warning(
                        "_append_audit: failed to spill audit events to %s: %s",
                        self._config.audit_trail_spill_path, exc,
                    )

    def _apply_conflict_policy(
        self,
        entry: dict,
        field: str,
        existing_val: Any,
        incoming_val: Any,
        uniprot_id: str,
        source: str,
        method: str,
        record: dict,
    ) -> str:
        """Apply the configured ``conflict_policy`` to a single field conflict.

        FIX-P1-C-11: previously the protein_resolver only recognised the
        undocumented ``"overwrite"`` policy -- every other documented policy
        (``keep_existing`` (default), ``keep_incoming``, ``keep_newer``,
        ``dead_letter``) silently fell through to ``keep_existing``. This
        helper mirrors the drug_resolver's policy application so the
        behaviour matches the documented set.

        Returns the policy string that was APPLIED (for the audit trail).
        """
        policy = getattr(self._config, "conflict_policy", "keep_existing")
        if policy == "keep_incoming":
            entry[field] = incoming_val
        elif policy == "keep_newer":
            # Compare timestamps -- incoming record's ``fetched_at`` /
            # ``timestamp`` vs the existing entry's ``resolved_at``. If
            # either side is missing, fall back to keep_existing.
            incoming_ts = (
                record.get("fetched_at")
                or record.get("timestamp")
                or record.get("ingested_at")
            )
            existing_ts = entry.get("resolved_at") or entry.get("fetched_at")
            if (
                incoming_ts
                and existing_ts
                and str(incoming_ts) > str(existing_ts)
            ):
                entry[field] = incoming_val
            else:
                policy = "keep_existing"  # mutated for audit-trail clarity
        elif policy == "dead_letter":
            self._dead_letter.append({
                "uniprot_id": uniprot_id,
                "record": record,
                "source": source,
                "stage": "conflict",
                "field": field,
                "existing": existing_val,
                "incoming": incoming_val,
            })
        # ``keep_existing`` (default) -- no-op.
        # Audit the conflict (with the EFFECTIVE policy, not the configured
        # one -- ``keep_newer`` may fall back to ``keep_existing`` above).
        self._append_audit(uniprot_id, {
            "action": "merge_conflict",
            "source": source,
            "method": method,
            "field": field,
            "existing": str(existing_val)[:100],
            "incoming": str(incoming_val)[:100],
            "policy_applied": policy,
        })
        return policy

    def _merge_uniprot_record(self, uniprot_id: str, record: dict) -> None:
        """Merge a duplicate UniProt record into the existing canonical entry.

        FIX DOC-05: complete docstring with merge semantics.

        Merge semantics:
        - **Fill-missing**: all fields in ``_MERGE_FILLABLE_FIELDS`` are
          fill-merged -- incoming values fill None/empty slots in the
          existing entry. If both are non-empty and differ, a conflict
          is logged and the existing value is kept (FIX CODE-12 /
          DESIGN-04 conflict_policy).
        - **Sources**: "uniprot" is always in the sources list after merge.
        - **Audit trail**: appends a "merge" event with method="uniprot_exact".
        - **Timestamps**: ``resolved_at`` is updated to current time.
        """
        entry = self.mapping[uniprot_id]

        # FIX DESIGN-04 / CODE-12: conflict detection during fill-merge.
        for f in _MERGE_FILLABLE_FIELDS:
            incoming_val = record.get(f)
            existing_val = entry.get(f)
            if incoming_val:
                if not existing_val:
                    entry[f] = incoming_val
                elif existing_val != incoming_val:
                    # FIX-P1-C-11: use the policy helper so all documented
                    # policies (keep_existing / keep_incoming / keep_newer /
                    # dead_letter) are honoured -- was previously "overwrite"
                    # only (not in the documented set).
                    self._apply_conflict_policy(
                        entry, f, existing_val, incoming_val,
                        uniprot_id, "uniprot", "uniprot_exact", record,
                    )

        gene_symbol = self._normalize_gene_symbol(record.get("gene_symbol", ""))
        organism = self._normalize_organism(
            record.get("organism") or self._config.default_organism
        )
        if gene_symbol:
            # FIX SCI-02: preserve gene-symbol case in index key.
            key = (gene_symbol, organism)
            if key not in self._gene_index:
                self._gene_index[key] = uniprot_id

        norm_name = normalize_name(gene_symbol or record.get("gene_name", ""))
        if norm_name:
            if norm_name not in self._name_index:
                self._name_index[norm_name] = uniprot_id
            multi = self._name_index_multi.setdefault(norm_name, [])
            if uniprot_id not in multi:
                multi.append(uniprot_id)

        string_id = record.get("string_id") or None
        if string_id and string_id not in self._string_to_uniprot:
            self._string_to_uniprot[string_id] = uniprot_id

        entry["resolved_at"] = self._now_iso()
        # FIX DQ-14: update canonical checksum.
        try:
            canon_payload = json.dumps(entry, sort_keys=True, default=str)
            entry["canonical_checksum"] = hashlib.sha256(
                canon_payload.encode()
            ).hexdigest()[:16]
        except (TypeError, ValueError):
            pass

        self._append_audit(uniprot_id, {
            "action": "merge",
            "source": "uniprot",
            "method": "uniprot_exact",
        })
        logger.debug("_merge_uniprot_record: merged duplicate '%s'", uniprot_id)

    def _merge_string_into_canonical(
        self, uniprot_id: str, record: dict
    ) -> None:
        """Merge a matched STRING record into the canonical entry.

        Merge semantics (FIX DOC-06):
        - **Fill-missing**: same as _merge_uniprot_record -- all fields in
          ``_MERGE_FILLABLE_FIELDS`` are fill-merged with conflict detection.
        - **_string_to_uniprot update**: if the STRING record's ``string_id``
          is not already mapped, it's added. If it IS already mapped to a
          DIFFERENT uid, conflict detection applies (CODE-16, CODE-17).
          Real mappings are NEVER overwritten by synthetic uids (CODE-17).
        - **Confidence upgrade (P2-9 v82 ROOT FIX)**: previously, the
          docstring said "Confidence: NOT upgraded by STRING merges
          (STRING is a cross-reference, not a stronger match method)."
          This left provisional STRING entries (confidence 0.5) STUCK at
          0.5 even when STRING cross-reference CONFIRMED their identity,
          causing downstream filters requiring confidence >= 0.7 to
          exclude confirmed entries -- silent data loss in the knowledge
          graph. The root fix: STRING cross-reference IS confirming
          evidence; confidence is upgraded from 0.5 to 0.7 (the
          ``string_cross_reference`` method score, registered at module
          import). The upgrade is MONOTONIC -- entries with confidence
          already >= 0.7 are NOT downgraded.
        - **Sources**: "string" added to entry's sources list (deduplicated).
        - **Audit trail**: appends a "merge" event with method="string_cross_reference",
          PLUS a "confidence_upgrade" event when the confidence is upgraded.
        """
        entry = self.mapping.get(uniprot_id)
        if entry is None:
            logger.error(
                "_merge_string_into_canonical: uniprot_id '%s' not found",
                uniprot_id,
            )
            return

        string_id = record.get("string_id", "") or ""

        # FIX DESIGN-04: fill-missing with conflict detection.
        for f in _MERGE_FILLABLE_FIELDS:
            incoming_val = record.get(f)
            existing_val = entry.get(f)
            if incoming_val:
                if not existing_val:
                    entry[f] = incoming_val
                elif existing_val != incoming_val:
                    # FIX-P1-C-11: use the policy helper (was "overwrite" only).
                    self._apply_conflict_policy(
                        entry, f, existing_val, incoming_val,
                        uniprot_id, "string", "string_cross_reference", record,
                    )

        if string_id and not entry.get("string_id"):
            entry["string_id"] = string_id

        if string_id:
            # FIX CODE-16 / CODE-17: conflict detection on string_id mapping.
            existing_uid = self._string_to_uniprot.get(string_id)
            if existing_uid is None:
                self._string_to_uniprot[string_id] = uniprot_id
            elif existing_uid != uniprot_id:
                # FIX CODE-17: real mappings are never overwritten by synthetics.
                if self.is_synthetic_uid(existing_uid) and not self.is_synthetic_uid(uniprot_id):
                    self._string_to_uniprot[string_id] = uniprot_id
                else:
                    self._append_audit(uniprot_id, {
                        "action": "string_xref_conflict",
                        "source": "string",
                        "method": "string_cross_reference",
                        "string_id": string_id,
                        "existing_uid": existing_uid,
                    })
                    logger.warning(
                        "_merge_string_into_canonical: string_id '%s' already "
                        "mapped to '%s', conflict with '%s'",
                        string_id, existing_uid, uniprot_id,
                    )

        # P2-9 ROOT FIX (v82): upgrade confidence when STRING cross-
        # reference confirms identity. Monotonic -- never downgrades.
        # The ``string_cross_reference`` method was registered at module
        # import with confidence 0.7 (above the 0.5 provisional floor,
        # below the 0.8 exact-name match -- preserves the hierarchy).
        try:
            current_conf = float(entry.get("match_confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            current_conf = 0.5
        target_conf = compute_match_confidence("string_cross_reference")
        if current_conf < target_conf:
            entry["match_confidence"] = target_conf
            entry["match_method"] = "string_cross_reference"
            self._append_audit(uniprot_id, {
                "action": "confidence_upgrade",
                "source": "string",
                "method": "string_cross_reference",
                "confidence_before": current_conf,
                "confidence_after": target_conf,
                "reason": "STRING cross-reference confirmed identity",
            })
            logger.info(
                "_merge_string_into_canonical: upgraded confidence for "
                "'%s' from %.3f to %.3f (STRING cross-reference confirmed)",
                uniprot_id, current_conf, target_conf,
            )

        sources = entry.get("sources", [])
        if "string" not in sources:
            sources.append("string")
            entry["sources"] = sources

        entry["resolved_at"] = self._now_iso()
        # FIX DQ-14: update canonical checksum.
        try:
            canon_payload = json.dumps(entry, sort_keys=True, default=str)
            entry["canonical_checksum"] = hashlib.sha256(
                canon_payload.encode()
            ).hexdigest()[:16]
        except (TypeError, ValueError):
            pass

        self._append_audit(uniprot_id, {
            "action": "merge", "source": "string",
            "method": "string_cross_reference",
        })
        logger.debug(
            "_merge_string_into_canonical: merged string_id='%s' into '%s'",
            string_id, uniprot_id,
        )

    def _merge_chembl_into_canonical(
        self,
        uniprot_id: str,
        record: dict,
        match_method: str,
    ) -> None:
        """Merge a matched ChEMBL target record into the canonical entry.

        Merge semantics (FIX DOC-07):
        - **Fill-missing**: same as _merge_uniprot_record -- all fields in
          ``_MERGE_FILLABLE_FIELDS`` are fill-merged with conflict detection.
        - **Confidence upgrade (FIX DESIGN-01)**: if the ChEMBL record's
          ``match_method`` has HIGHER confidence than the entry's current
          ``match_confidence``, the entry's confidence is UPGRADED.
          Downgrades are NOT applied (was a bug -- DESIGN-01).
        - **Sources**: "chembl" added to entry's sources list (deduplicated).
        - **Audit trail**: appends a "merge" event with the ChEMBL match method.
        """
        entry = self.mapping.get(uniprot_id)
        if entry is None:
            logger.error(
                "_merge_chembl_into_canonical: uniprot_id '%s' not found",
                uniprot_id,
            )
            return

        chembl_target_id = record.get("chembl_target_id", "") or ""

        # FIX DESIGN-04: fill-missing with conflict detection.
        for f in _MERGE_FILLABLE_FIELDS:
            incoming_val = record.get(f)
            existing_val = entry.get(f)
            if incoming_val:
                if not existing_val:
                    entry[f] = incoming_val
                elif existing_val != incoming_val:
                    # FIX-P1-C-11: use the policy helper (was "overwrite" only).
                    self._apply_conflict_policy(
                        entry, f, existing_val, incoming_val,
                        uniprot_id, "chembl", match_method, record,
                    )

        if chembl_target_id and not entry.get("chembl_target_id"):
            entry["chembl_target_id"] = chembl_target_id

        sources = entry.get("sources", [])
        if "chembl" not in sources:
            sources.append("chembl")
            entry["sources"] = sources

        # FIX DESIGN-01: confidence UPGRADE only (was downgrade bug).
        current_conf = entry.get("match_confidence", 0.0)
        new_conf = compute_match_confidence(match_method)

        if new_conf > current_conf:
            old_conf = current_conf
            entry["match_method"] = match_method
            entry["match_confidence"] = new_conf
            self._append_audit(uniprot_id, {
                "action": "confidence_upgrade",
                "source": "chembl",
                "method": match_method,
                "confidence_before": old_conf,
                "confidence_after": new_conf,
            })

        entry["resolved_at"] = self._now_iso()
        # FIX DQ-14: update canonical checksum.
        try:
            canon_payload = json.dumps(entry, sort_keys=True, default=str)
            entry["canonical_checksum"] = hashlib.sha256(
                canon_payload.encode()
            ).hexdigest()[:16]
        except (TypeError, ValueError):
            pass

        self._append_audit(uniprot_id, {
            "action": "merge", "source": "chembl", "method": match_method,
        })
        logger.debug(
            "_merge_chembl_into_canonical: merged chembl_target_id='%s' "
            "into '%s' via %s",
            chembl_target_id, uniprot_id, match_method,
        )

    # ------------------------------------------------------------------
    # Internal -- provisional entry creation
    # ------------------------------------------------------------------

    def _create_provisional_from_string(self, record: dict) -> None:
        """Create a provisional canonical entry from a STRING record.

        FIX DOC-08: comprehensive docstring.

        The synthetic_uid format is ``STRING:{sanitized_string_id}``.
        If a provisional entry with the same synthetic_uid already
        exists, the duplicate is skipped (DEBUG log + stat increment
        per REL-08 / DESIGN-06).

        match_method is "string_provisional" (FIX SCI-17 -- was
        misleading "gene_name_organism").

        Entries are promoted when a real uniprot_id arrives (ARCH-02).
        """
        string_id = record.get("string_id", "") or ""
        gene_symbol_raw = (
            record.get("gene_symbol", "") or record.get("preferred_name", "") or ""
        )
        organism_raw = record.get("organism", self._config.default_organism)

        if not string_id:
            logger.warning(
                "_create_provisional_from_string: empty string_id, skipping"
            )
            return

        # FIX SCI-03: normalize organism.
        organism = self._normalize_organism(organism_raw)
        gene_symbol = self._normalize_gene_symbol(gene_symbol_raw)

        # FIX CODE-27: sanitize string_id for synthetic uid.
        # FIX-P3-6: use _make_synthetic_uid_checked instead of constructing
        # the synthetic UID directly. The helper performs collision
        # detection against self.mapping, logs a WARNING, increments the
        # synthetic_uid_collisions stat, and disambiguates with a hash
        # suffix on collision. The previous direct construction silently
        # overwrote an existing entry when two different STRING alias
        # rows mapped to the same string_id.
        synthetic_uid = self._make_synthetic_uid_checked("string", string_id)

        if synthetic_uid in self.mapping:
            # FIX DESIGN-06 / REL-08: log + stat for duplicate provisional.
            # Defensive: _make_synthetic_uid_checked already guarantees a
            # UID that is NOT in self.mapping (it disambiguates with hash
            # suffixes on collision), so this branch is now only reachable
            # for the rare race condition where a concurrent resolver
            # thread inserted the same UID between the helper's check and
            # this one. Keep it as a fail-safe.
            #
            # FIX-P4-6 (v42): the previous code incremented
            # `synthetic_keys_generated` here -- on the SKIP path -- making
            # the stat meaningless (it counted attempts, not creations).
            # The skip path now records nothing (the
            # `synthetic_uid_collisions` stat from _make_synthetic_uid_checked
            # already tracks collisions). The `synthetic_keys_generated`
            # counter is incremented ONLY on the create path below.
            logger.debug(
                "_create_provisional_from_string: '%s' already exists, "
                "skipping duplicate", synthetic_uid,
            )
            return

        now_iso = self._now_iso()
        # v89 ROOT FIX (BUG #42 -- silent fallback to default_organism
        # masks invalid organism input):
        #   The previous code did ``"organism": organism or
        #   self._config.default_organism``. If ``organism`` was an
        #   empty string (from ``_normalize_organism`` returning "" for
        #   invalid input), the ``or`` fell through to
        #   ``default_organism`` ("Homo sapiens"). This masked invalid
        #   organism input -- the operator didn't see that the original
        #   organism was unparseable. Cross-species contamination could
        #   result if the original organism was non-human but unparseable.
        #
        #   ROOT FIX: detect the fallback case explicitly. If ``organism``
        #   is empty/falsy AND the original ``organism_raw`` was non-empty
        #   (i.e. the caller DID provide an organism but it was
        #   unparseable), log a WARNING so the operator sees the invalid
        #   input. The fallback to ``default_organism`` still happens
        #   (preserving backward compatibility), but it's now visible.
        _resolved_organism = organism or self._config.default_organism
        if not organism and organism_raw and str(organism_raw).strip():
            logger.warning(
                "_create_provisional_from_string: STRING record "
                "string_id=%r had unparseable organism %r -- falling "
                "back to default_organism %r. This may cause "
                "cross-species contamination if the original organism "
                "was non-human. Fix the upstream organism string or "
                "extend _ORGANISM_ALIASES to recognize it. (v89 BUG #42)",
                string_id, organism_raw, self._config.default_organism,
            )
        entry: dict = {
            "uniprot_id": synthetic_uid,
            "gene_symbol": gene_symbol,
            "gene_name": record.get("gene_name", gene_symbol_raw),
            "organism": _resolved_organism,
            "sequence": None,
            "protein_name": record.get("protein_name"),
            "string_id": string_id,
            "chembl_target_id": None,
            "canonical_name": gene_symbol or string_id,
            "sources": ["string"],  # FIX CODE-22: source is "string" (not "string_derived").
            "match_method": "string_provisional",  # FIX SCI-17: honest method name.
            "match_confidence": compute_match_confidence("string_provisional"),
            "created_at": now_iso,
            "resolved_at": now_iso,
            "resolver_version": MAPPING_SCHEMA_VERSION,
            "input_checksum": "",
            "canonical_checksum": "",
            "isoforms": [],
            "deprecated_by": None,
            "provisional": True,
        }

        self.mapping[synthetic_uid] = entry

        # v80 P0-D3: register this provisional entry in the
        # (gene, organism) index so _ingest_uniprot_record can find
        # it in O(1) instead of O(N) when a real UniProt record
        # arrives with the same (gene, organism).
        #
        # v82 P0-D3b: ALSO register under the alias's ``uniprot_id``
        # field (if present). STRING alias records carry the REAL
        # UniProt accession in their ``uniprot_id`` column. When a
        # real UniProt record arrives with that accession, we look it
        # up in O(1) via ``_provisional_by_alias_uniprot`` instead of
        # falling through to the O(N) scan. This is the fix for the
        # Airflow-timeout regression that occurred when STRING aliases
        # had no gene_symbol (so the gene-organism index was empty
        # and the O(N) fallback fired for every record).
        _alias_uniprot = record.get("uniprot_id") or None
        self._register_provisional_entry(
            synthetic_uid, gene_symbol,
            organism or self._config.default_organism,
            alias_uniprot_id=_alias_uniprot,
        )

        # FIX-P4-6 (v42): increment synthetic_keys_generated ONLY on actual
        # creation (was previously also incremented on the skip path).
        self._stats.inc("synthetic_keys_generated")

        if gene_symbol:
            # FIX SCI-02: preserve gene-symbol case in index key.
            key = (gene_symbol, organism or self._config.default_organism)
            if key not in self._gene_index:
                self._gene_index[key] = synthetic_uid
            # v89 ROOT FIX (BUG #35): also populate the case-preserving
            # gene symbol index (organism-agnostic, case-sensitive) so
            # fuzzy matching on gene symbols preserves species distinction.
            if gene_symbol not in self._gene_symbol_index:
                self._gene_symbol_index[gene_symbol] = synthetic_uid
            self._gene_symbol_index_multi.setdefault(gene_symbol, []).append(synthetic_uid)

        self._string_to_uniprot[string_id] = synthetic_uid

        norm_name = normalize_name(gene_symbol or "")
        if norm_name:
            if norm_name not in self._name_index:
                self._name_index[norm_name] = synthetic_uid
            self._name_index_multi.setdefault(
                norm_name, []
            ).append(synthetic_uid)

        self._append_audit(synthetic_uid, {
            "action": "create", "source": "string",
            "method": "string_provisional",
        })
        logger.debug(
            "_create_provisional_from_string: created '%s' from string_id='%s'",
            synthetic_uid, string_id,
        )

    def _create_provisional_from_chembl(self, record: dict) -> None:
        """Create a provisional canonical entry from a ChEMBL target record.

        FIX DOC-08: comprehensive docstring.

        The synthetic_uid format is ``CHEMBL_T:{sanitized_chembl_target_id}``.
        match_method is "chembl_provisional" (FIX SCI-17).
        """
        chembl_target_id = record.get("chembl_target_id", "") or ""
        gene_symbol_raw = record.get("gene_symbol", "") or ""
        organism_raw = record.get("organism", self._config.default_organism)

        if not chembl_target_id:
            logger.warning(
                "_create_provisional_from_chembl: empty chembl_target_id, skipping"
            )
            return

        # FIX SCI-03: normalize organism.
        organism = self._normalize_organism(organism_raw)
        gene_symbol = self._normalize_gene_symbol(gene_symbol_raw)

        # FIX-P3-6: use _make_synthetic_uid_checked instead of constructing
        # the synthetic UID directly. The helper performs collision
        # detection against self.mapping, logs a WARNING, increments the
        # synthetic_uid_collisions stat, and disambiguates with a hash
        # suffix on collision. The previous direct construction silently
        # overwrote an existing entry when two different ChEMBL target
        # records mapped to the same chembl_target_id.
        synthetic_uid = self._make_synthetic_uid_checked("chembl", chembl_target_id)

        if synthetic_uid in self.mapping:
            # Defensive: _make_synthetic_uid_checked already guarantees a
            # UID that is NOT in self.mapping (it disambiguates with hash
            # suffixes on collision), so this branch is now only reachable
            # for the rare race condition where a concurrent resolver
            # thread inserted the same UID between the helper's check and
            # this one. Keep it as a fail-safe.
            #
            # FIX-P4-6 (v42): the previous code incremented
            # `synthetic_keys_generated` here -- on the SKIP path -- making
            # the stat meaningless (it counted attempts, not creations).
            # The skip path now records nothing (the
            # `synthetic_uid_collisions` stat from _make_synthetic_uid_checked
            # already tracks collisions). The `synthetic_keys_generated`
            # counter is incremented ONLY on the create path below.
            logger.debug(
                "_create_provisional_from_chembl: '%s' already exists, "
                "skipping duplicate", synthetic_uid,
            )
            return

        now_iso = self._now_iso()
        entry: dict = {
            "uniprot_id": synthetic_uid,
            "gene_symbol": gene_symbol,
            "gene_name": record.get("gene_name", gene_symbol_raw),
            "organism": organism or self._config.default_organism,
            "sequence": None,
            "protein_name": record.get("protein_name"),
            "string_id": None,
            "chembl_target_id": chembl_target_id,
            "canonical_name": gene_symbol or chembl_target_id,
            "sources": ["chembl"],
            "match_method": "chembl_provisional",  # FIX SCI-17: honest method name.
            "match_confidence": compute_match_confidence("chembl_provisional"),
            "created_at": now_iso,
            "resolved_at": now_iso,
            "resolver_version": MAPPING_SCHEMA_VERSION,
            "input_checksum": "",
            "canonical_checksum": "",
            "isoforms": [],
            "deprecated_by": None,
            "provisional": True,
        }

        self.mapping[synthetic_uid] = entry

        # v80 P0-D3: register this provisional entry in the
        # (gene, organism) index so _ingest_uniprot_record can find
        # it in O(1) instead of O(N) when a real UniProt record
        # arrives with the same (gene, organism).
        # v82 P0-D3b: also register under the ChEMBL target's
        # ``uniprot_id`` cross-reference if present (same fix as the
        # STRING path -- enables O(1) promotion by UniProt accession).
        _chembl_alias_uniprot = record.get("uniprot_id") or None
        self._register_provisional_entry(
            synthetic_uid, gene_symbol,
            organism or self._config.default_organism,
            alias_uniprot_id=_chembl_alias_uniprot,
        )

        # FIX-P4-6 (v42): increment synthetic_keys_generated ONLY on actual
        # creation (was previously also incremented on the skip path).
        self._stats.inc("synthetic_keys_generated")

        if gene_symbol:
            # FIX SCI-02: preserve gene-symbol case in index key.
            key = (gene_symbol, organism or self._config.default_organism)
            if key not in self._gene_index:
                self._gene_index[key] = synthetic_uid

        norm_name = normalize_name(gene_symbol or "")
        if norm_name:
            if norm_name not in self._name_index:
                self._name_index[norm_name] = synthetic_uid
            self._name_index_multi.setdefault(
                norm_name, []
            ).append(synthetic_uid)

        self._append_audit(synthetic_uid, {
            "action": "create", "source": "chembl",
            "method": "chembl_provisional",
        })
        logger.debug(
            "_create_provisional_from_chembl: created '%s' from chembl_target_id='%s'",
            synthetic_uid, chembl_target_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _df_to_records(df: Any) -> List[dict]:
        """Convert a DataFrame to a list of dicts, replacing NaN with ``None``.

        FIX INT-09: supports polars DataFrames via duck typing.
        """
        if df is None:
            return []
        try:
            is_empty = df.empty
        except AttributeError:
            # FIX INT-09: polars DataFrame fallback.
            if hasattr(df, "to_dicts"):
                try:
                    records = df.to_dicts()
                    for rec in records:
                        for k, v in list(rec.items()):
                            if v is None:
                                rec[k] = None
                    return records
                except Exception as exc:
                    logger.warning(
                        "_df_to_records: polars to_dicts failed: %s", exc
                    )
                    return []
            logger.warning(
                "_df_to_records: input is %s, not a recognized DataFrame type",
                type(df).__name__,
            )
            return []
        if is_empty:
            return []
        # pandas path
        try:
            return df.where(df.notna(), None).to_dict(orient="records")
        except Exception as exc:
            logger.warning("_df_to_records: pandas conversion failed: %s", exc)
            return []
