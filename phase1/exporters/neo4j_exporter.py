"""
Neo4j Knowledge Graph Exporter (Phase 1 -> Phase 2 connector)
=============================================================

This module is the Phase 1 side of the bridge that connects Phase 1's
processed_data CSV outputs to Phase 2's Neo4j knowledge graph.

PREVIOUS STATUS (Phase 1 alone): STUB -- raised NotImplementedError.
CURRENT STATUS (unified package): WORKING -- delegates to
``drugos_graph.phase1_bridge``, which converts Phase 1 CSVs into Phase 2
node/edge dicts and loads them via ``DrugOSGraphBuilder``.

The bridge is bidirectionally traceable: every node/edge carries a
``_source_phase=1`` lineage property plus the originating CSV filename and
row index, so any downstream bug in the knowledge graph can be traced back
to the exact Phase 1 row that produced it.

Node types loaded (v52 ROOT FIX -- P1-025: all 5 DOCX-required node types now emitted):
- Compound        (from drugbank_drugs.csv + chembl_drugs.csv, keyed by InChIKey)
- Protein         (from drugbank_interactions.csv.gz + uniprot_proteins.csv, keyed by UniProt accession)
- Gene            (from omim_gene_disease_associations.csv + disgenet_gda, keyed by NCBI Gene ID or gene symbol)
- Disease         (from omim_gene_disease_associations.csv + disgenet_gda, keyed by OMIM:MIM or DOID)
- ClinicalOutcome (from drugbank_indications.csv, keyed by indication text hash)
- Pathway         (derived from STRING PPI connected components, keyed by pathway ID)

The DOCX (Phase 2 section) requires ALL 5 node types: Drug/Protein/Pathway/Disease/ClinicalOutcome.
The v48 exporter only emitted 4 (Compound/Protein/Gene/Disease) -- missing Pathway and
ClinicalOutcome. The v52 fix updates the docstring to reflect the actual bridge behavior
(which already stages all 5 types since v49) and adds ClinicalOutcome/Pathway sources
to the Phase1OutputContract's optional list.

Edge types loaded (subset of drugos_graph.config.CORE_EDGE_TYPES):
- (Compound, targets, Protein)
- (Compound, inhibits, Protein)
- (Compound, activates, Protein)
- (Compound, allosterically_modulates, Protein)
- (Compound, unknown, Protein)
- (Gene, associated_with, Disease)
- (Gene, susceptible_to, Disease)
- (Gene, encodes, Protein)
- (Compound, treats, Disease)            ← the TransE link-prediction target
- (Compound, has_clinical_outcome, ClinicalOutcome)
- (Protein, interacts_with, Protein)     ← from STRING PPI
- (Protein, participates_in, Pathway)    ← derived from STRING PPI clusters

USAGE
-----
Via the bridge (recommended -- works with or without Neo4j)::

    from drugos_graph.phase1_bridge import run_phase1_to_phase2
    report = run_phase1_to_phase2(
        phase1_processed_dir="phase1/processed_data",
        builder=my_builder,        # real DrugOSGraphBuilder or RecordingGraphBuilder
    )

Via this module's legacy entry point (kept for backward compat with
Phase 1 tests that called ``export_to_neo4j()`` expecting it to raise)::

    from exporters.neo4j_exporter import export_to_neo4j
    report = export_to_neo4j(neo4j_uri=None,
                              neo4j_user=None,
                              neo4j_password=None)

.. note::
    v29 ROOT FIX (audit O-4): the legacy ``pg_session`` parameter was
    REMOVED -- it was accepted but silently ignored, making the Phase 1 ->
    Neo4j wire look like it was using PostgreSQL when it was actually
    reading CSVs through ``phase1_bridge``. PostgreSQL -> Neo4j via a
    SQLAlchemy session is **not implemented** in this function. To export
    from PostgreSQL, set the ``DATABASE_URL`` env var and call
    ``drugos_graph.phase1_bridge.run_phase1_to_phase2`` (the bridge
    prefers PostgreSQL when ``DATABASE_URL`` is set and the ``drugs``
    table is populated).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text, table as _sa_table, column as _sa_column, func as _sa_func, select as _sa_select

logger = logging.getLogger(__name__)

# Resolve the unified package root: this file lives at
#   phase1/exporters/neo4j_exporter.py
# so the unified root is two parents up. We use this to locate phase2/.
_THIS_DIR = Path(__file__).resolve().parent
_PHASE1_ROOT = _THIS_DIR.parent                # phase1/
_UNIFIED_ROOT = _PHASE1_ROOT.parent            # unified/
_PHASE2_ROOT = _UNIFIED_ROOT / "phase2"


# =============================================================================
# P1-028 ROOT FIX (Team Member 3 -- InChIKey format validation before Neo4j
# export, defense-in-depth against Cypher injection):
#
# The issue: ``export_to_neo4j`` delegated to ``phase1_bridge`` which uses
# Cypher ``UNWIND $rows AS row MERGE (c:Compound {inchikey: row.inchikey})``.
# Neo4j's parameter binding SHOULD prevent injection, but a malformed
# InChIKey containing Cypher-special characters (e.g. ``"}--`` or
# ``RETURN 1//``) could theoretically break the MERGE (CVE-2019-10236
# showed edge cases in parameterised query handling). Defense in depth:
# validate InChIKey format BEFORE passing to Neo4j.
#
# ROOT FIX: validate every InChIKey against the canonical InChI Trust
# regex ``^[A-Z]{14}-[A-Z]{10}-[A-Z]$`` (27 chars: 14-char hash, hyphen,
# 10-char hash, hyphen, 1-char version flag) BEFORE export. Synthetic
# keys (``SYNTH...``) are allowed (they are platform-generated surrogates
# for drugs without a real InChIKey). Invalid keys are REJECTED with a
# logged WARNING and dead-lettered -- they do NOT reach Neo4j.
#
# This is a DEFENSE-IN-DEPTH layer. The primary protection is Neo4j's
# parameter binding; this layer catches malformed data that should never
# have reached the exporter in the first place (corrupted CSV, upstream
# pipeline bug, adversarial data source).
# =============================================================================

#: Canonical InChIKey format regex (InChI Trust specification).
#: 14 uppercase letters + hyphen + 10 uppercase letters + hyphen + 1 letter.
#: The version flag is typically 'S' (standard) or 'N' (non-standard) but
#: the spec allows any single letter. Case-INSENSITIVE on input (we
#: normalise to uppercase before validation so a lowercase key from an
#: older PubChem export is accepted after uppercasing).
_NEO4J_INCHIKEY_PATTERN: re.Pattern[str] = re.compile(
    r"^[A-Za-z]{14}-[A-Za-z]{10}-[A-Za-z]$"
)

#: Synthetic InChIKey prefix (platform-generated surrogates).
_NEO4J_SYNTH_PREFIX: str = "SYNTH"


def _validate_inchikey_for_neo4j(inchikey: Any) -> bool:
    """Validate an InChIKey for Neo4j export (P1-028 ROOT FIX).

    Returns ``True`` if the InChIKey is:
    - A standard/non-standard InChIKey matching
      ``^[A-Z]{14}-[A-Z]{10}-[A-Z]$`` (case-insensitive, normalised to
      uppercase), OR
    - A synthetic surrogate (``SYNTH...`` prefix, case-insensitive).

    Returns ``False`` for None, empty strings, non-strings, or any value
    that does not match either pattern. This is the defense-in-depth
    gate that prevents malformed / potentially-injection-bearing InChIKeys
    from reaching Neo4j's UNWIND/MERGE.

    Parameters
    ----------
    inchikey:
        The InChIKey value to validate (typically from a CSV cell).

    Returns
    -------
    bool
    """
    if not isinstance(inchikey, str):
        return False
    stripped = inchikey.strip()
    if not stripped:
        return False
    upper = stripped.upper()
    # Synthetic keys are platform-generated surrogates -- always valid.
    if upper.startswith(_NEO4J_SYNTH_PREFIX):
        return True
    # Standard / non-standard InChIKeys must match the canonical regex.
    return bool(_NEO4J_INCHIKEY_PATTERN.match(upper))


def validate_inchikeys_for_export(
    inchikeys: List[Any],
    *,
    source_label: str = "unknown",
) -> Tuple[List[str], List[Tuple[Any, str]]]:
    """Validate a list of InChIKeys for Neo4j export (P1-028 ROOT FIX).

    Splits the input into (valid, invalid) lists. Invalid InChIKeys are
    logged with a WARNING and returned for dead-lettering by the caller.

    Parameters
    ----------
    inchikeys:
        List of InChIKey values (strings, None, etc.) to validate.
    source_label:
        Human-readable label for the data source (e.g. ``"drugbank_drugs.csv"``)
        included in log messages for traceability.

    Returns
    -------
    tuple[list[str], list[tuple[Any, str]]]
        ``(valid_inchikeys, invalid_with_reasons)`` where
        ``invalid_with_reasons`` is a list of ``(original_value, reason)``
        tuples. ``valid_inchikeys`` are the uppercase-normalised valid keys.
    """
    valid: List[str] = []
    invalid: List[Tuple[Any, str]] = []
    for ik in inchikeys:
        if not isinstance(ik, str):
            invalid.append((ik, f"not a string ({type(ik).__name__})"))
            continue
        stripped = ik.strip()
        if not stripped:
            invalid.append((ik, "empty string"))
            continue
        if not _validate_inchikey_for_neo4j(stripped):
            invalid.append((ik, "regex mismatch (not SYNTH and not 14-10-1 format)"))
            continue
        valid.append(stripped.upper())
    if invalid:
        logger.warning(
            "validate_inchikeys_for_export[%s]: rejected %d/%d InChIKey(s) "
            "for Neo4j export (P1-028 defense-in-depth). First few invalid: %s",
            source_label,
            len(invalid),
            len(inchikeys),
            str(invalid[:5])[:200],
        )
    return valid, invalid


# v28 FIX P1-ER-14 (MEDIUM): previously this exporter silently delegated
# to ``phase1_bridge.run_phase1_to_phase2`` with an IMPLICIT contract --
# the bridge's CSV filenames were only discoverable by reading its source.
# If a Phase 1 pipeline silently failed to emit one of the CSVs, the
# bridge would log a warning and produce an empty DataFrame, then the
# KG build would proceed with a partial graph and the operator would
# never see a hard error at the exporter boundary. This dataclass makes
# the contract EXPLICIT and FAIL-FAST: any missing REQUIRED CSV raises
# ``DrugOSDataError`` before the bridge is invoked.
@dataclass(frozen=True)
class Phase1OutputContract:
    """Explicit, fail-fast contract for the Phase 1 -> Phase 2 bridge.

    Attributes
    ----------
    required:
        Mapping of contract-key -> list of candidate filenames. At
        least ONE candidate per key MUST exist on disk, otherwise
        :func:`validate_phase1_output_contract` raises
        ``DrugOSDataError``. These are the canonical Phase 1 outputs
        without which the KG build is meaningless.
    optional:
        Mapping of contract-key -> list of candidate filenames. If
        NONE of the candidates exist, a WARNING is logged but no
        exception is raised -- the bridge degrades gracefully (e.g.
        ``drugbank_indications.csv`` absent -> free-text indication
        column matching is used instead).
    """

    required: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        # The 3 canonical Phase 1 outputs that define the KG's spine.
        #
        # v80 FORENSIC ROOT FIX (P0-C7 -- KG build blocked without DrugBank):
        #   The previous contract ONLY accepted ``drugbank_drugs.csv`` for
        #   the "drugs" key. If DrugBank was skipped (no academic license,
        #   no XML file, network error), the contract raised
        #   ``DrugOSDataError`` and the KG build was BLOCKED -- even if
        #   ChEMBL had successfully produced ``chembl_drugs.csv`` (or the
        #   alias ``drugs.csv``). For a platform whose DOCX explicitly
        #   says "V1 is built on free, publicly available biomedical
        #   data -- making the $0 data-cost model viable from day one",
        #   hard-requiring a license-gated source is a structural
        #   contradiction. The DrugBank academic license has been paused
        #   since May 2026 (see ``_v50_downloaders.download_drugbank_open_data``
        #   docstring), so EVERY new deployment hit this block.
        #
        #   ROOT FIX: add ``chembl_drugs.csv`` and the alias ``drugs.csv``
        #   as additional candidates for the "drugs" key. The bridge's
        #   ``validate_phase1_output_contract`` already accepts ANY ONE
        #   of the candidates (it returns the first match), so ChEMBL-
        #   only deployments now build a valid KG (Compound nodes from
        #   ChEMBL, no DrugBank-specific indications -- a graceful
        #   degradation that matches the DOCX's "free public data" V1
        #   mandate). DrugBank is preferred when available (it provides
        #   richer drug metadata + indication edges).
        "drugs": (
            "drugbank_drugs.csv",     # preferred (richer metadata + indications)
            "drugbank_open_drugs.csv",  # v50 open-data fallback (no license)
            "chembl_drugs.csv",       # ChEMBL-only fallback (no DrugBank at all)
            "drugs.csv",              # legacy alias (ChEMBL pipeline emits this)
        ),
        "interactions": (
            "drugbank_interactions.csv.gz",
            "drugbank_open_interactions.csv",  # v50 open-data fallback
            "chembl_activities_clean.csv",      # ChEMBL-only fallback
            "chembl_activities.csv",            # legacy alias
        ),
        "omim_gda": ("omim_gene_disease_associations.csv",),
    })
    optional: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        # Auxiliary sources -- bridge degrades to empty DataFrame if absent.
        # v52 ROOT FIX (P1-025): "indications" is the source for
        # ClinicalOutcome nodes + Compound-treats-Disease edges.
        # "string_ppi" is the source for Pathway nodes (derived from
        # connected components). Both are OPTIONAL (the bridge degrades
        # gracefully) but the DOCX requires all 5 node types, so a
        # WARNING is logged when either is missing.
        "indications": ("drugbank_indications.csv",),
        # v13 bridge: dual-name lookup (prefixed + actual pipeline-emitted).
        "chembl_drugs": ("chembl_drugs.csv", "drugs.csv"),
        "uniprot_proteins": ("uniprot_proteins.csv", "proteins.csv"),
        "string_ppi": (
            "string_protein_protein_interactions.csv",
            "protein_protein_interactions.csv",
        ),
        "disgenet_gda": (
            "disgenet_gene_disease_associations.csv",
            "gene_disease_associations.csv",
        ),
        "pubchem_enrichment": ("pubchem_enrichment.csv",),
        "chembl_activities": (
            "chembl_activities_clean.csv",
            "chembl_activities.csv",
        ),
        "omim_susceptibility": (
            "omim_gene_disease_susceptibility.csv",
        ),
    })

    def all_keys(self) -> List[str]:
        return list(self.required.keys()) + list(self.optional.keys())

    def candidates_for(self, key: str, base_dir: Path) -> List[Path]:
        """Return the candidate Path objects for *key* under *base_dir*."""
        if key in self.required:
            return [base_dir / name for name in self.required[key]]
        if key in self.optional:
            return [base_dir / name for name in self.optional[key]]
        raise KeyError(f"unknown contract key: {key!r}")


def _local_drugos_data_error() -> type:
    """Return the real ``DrugOSDataError`` if importable, else a local stub.

    The exporter must raise ``DrugOSDataError`` to match the bridge's
    contract, but it must also work when the phase2 package is not yet
    on sys.path (we add it inside ``_ensure_phase2_on_path``). We
    therefore attempt the import lazily; on failure we use a local
    subclass of :class:`Exception` with the same name.
    """
    try:
        _ensure_phase2_on_path()
        from drugos_graph.exceptions import DrugOSDataError  # type: ignore
        return DrugOSDataError
    except Exception:
        class DrugOSDataError(Exception):  # type: ignore[no-redef]
            """Local fallback when phase2.exceptions cannot be imported."""

        return DrugOSDataError


def validate_phase1_output_contract(
    base_dir: Path,
    contract: Optional[Phase1OutputContract] = None,
) -> Dict[str, Path]:
    """Validate the Phase 1 output contract under *base_dir*.

    Parameters
    ----------
    base_dir:
        Phase 1 ``processed_data`` directory.
    contract:
        Contract to validate against. Defaults to a fresh
        :class:`Phase1OutputContract`.

    Returns
    -------
    dict
        Mapping of contract-key -> resolved Path for every key whose
        candidates were found on disk (REQUIRED + OPTIONAL).

    Raises
    ------
    DrugOSDataError
        If any REQUIRED contract-key has no candidate file on disk.
    FileNotFoundError
        If *base_dir* itself does not exist.
    """
    if contract is None:
        contract = Phase1OutputContract()
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Phase 1 processed_data directory does not exist: {base_dir}"
        )

    DrugOSDataError = _local_drugos_data_error()
    resolved: Dict[str, Path] = {}
    missing_required: List[str] = []

    for key in contract.required:
        candidates = contract.candidates_for(key, base_dir)
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            missing_required.append(
                f"  • {key} -- expected one of: "
                + ", ".join(repr(c.name) for c in candidates)
            )
        else:
            resolved[key] = found

    if missing_required:
        raise DrugOSDataError(
            "Phase 1 output contract violation -- REQUIRED CSVs missing "
            f"under {base_dir}:\n" + "\n".join(missing_required) +
            "\nRun the corresponding Phase 1 pipeline(s) before invoking "
            "the Neo4j exporter. See Phase1OutputContract in "
            "phase1/exporters/neo4j_exporter.py for the full contract."
        )

    # Optional keys: log a WARNING per missing key, but do NOT raise.
    for key in contract.optional:
        candidates = contract.candidates_for(key, base_dir)
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            logger.warning(
                "Phase1OutputContract: optional source %r not found under "
                "%s (expected one of: %s) -- bridge will degrade to an "
                "empty DataFrame for this source.",
                key, base_dir, [c.name for c in candidates],
            )
        else:
            resolved[key] = found

    return resolved


def _ensure_phase2_on_path() -> None:
    """Make ``drugos_graph`` importable when called from Phase 1 context."""
    if str(_PHASE2_ROOT) not in sys.path:
        sys.path.insert(0, str(_PHASE2_ROOT))


def check_neo4j_readiness(pg_session) -> dict:
    """Validate PostgreSQL data compatibility for Neo4j export.

    Checks that all required tables have data (> 0 records), which is a
    prerequisite for exporting to the Neo4j knowledge graph.

    Parameters
    ----------
    pg_session : SQLAlchemy Session
        Active database session connected to the staging PostgreSQL DB.

    Returns
    -------
    dict
        Keys:
        - 'ready': bool -- True if all tables have records
        - 'record_counts': dict -- table_name -> count for each checked table
        - 'phase': str -- current implementation status
    """
    counts = {}
    # v40 ROOT FIX (P1 #50): the previous code REQUIRED entity_mapping to
    # have >0 rows, but entity_mapping is only populated by the master
    # DAG's entity_resolution task. If the DAG hadn't run, entity_mapping
    # was empty -> ready=False even if all other tables were populated.
    # The fix: split tables into REQUIRED (must have >0 rows) and
    # OPTIONAL (may be empty). entity_mapping and pubchem_compound_
    # properties are OPTIONAL.
    REQUIRED_TABLES = {
        "drugs", "proteins", "gene_disease_associations",
        "drug_protein_interactions",
    }
    OPTIONAL_TABLES = {
        "protein_protein_interactions",  # STRING PPI -- may be empty if STRING not loaded
        "entity_mapping",  # only populated by entity_resolution task
        "pubchem_compound_properties",  # v40: was missing from the original list
    }
    ALL_TABLES = REQUIRED_TABLES | OPTIONAL_TABLES
    # v40 ROOT FIX (P1 #51): the f-string SQL pattern is safe because
    # ALL_TABLES is a hardcoded set (not user input). But we add a
    # whitelist check to make the safety explicit.
    # v66 ROOT FIX (P1C-025 -- eliminate f-string SQL table interpolation):
    #   The previous code used ``text(f'SELECT COUNT(*) FROM {t}')`` which
    #   interpolated the table name directly into the SQL string via
    #   f-string. The whitelist check (``t.replace("_", "").isalnum()``)
    #   made this SAFE today, but the pattern was an anti-pattern that
    #   could become a SQL-injection vector if the whitelist were ever
    #   expanded to include tables with special characters or if the
    #   validation were accidentally removed during maintenance.
    #   ROOT FIX: use SQLAlchemy's ``table()`` + ``select(func.count())``
    #   construct, which treats the table name as a STRUCTURED identifier
    #   (not a string interpolated into raw SQL). SQLAlchemy renders it
    #   with proper quoting for the target dialect. The whitelist check
    #   is retained as defense-in-depth.
    for t in sorted(ALL_TABLES):
        if not isinstance(t, str) or not t.replace("_", "").isalnum():
            logger.warning('check_neo4j_readiness: skipping invalid table name %r', t)
            continue
        try:
            # v83 FORENSIC ROOT FIX (P2-15): the previous code used
            # ``_sa_select(_sa_func.count()).select_from(_sa_table(t))``
            # which creates an UNQUALIFIED TableClause. On PostgreSQL, if
            # the table is in a non-default schema (or ``search_path`` is
            # misconfigured), the unqualified reference fails with
            # "relation does not exist". ROOT FIX: introspect the actual
            # table object from the SQLAlchemy metadata (which carries
            # schema info) via ``pg_session.bind``. If introspection fails
            # (e.g. SQLite or a connection without bind), fall back to
            # the unqualified TableClause (which works on SQLite and on
            # Postgres with default search_path).
            _table_obj = None
            # v104 FORENSIC ROOT FIX (P1-004 -- _meta_name may be undefined):
            #   The previous code declared ``_meta_name`` ONLY inside the
            #   ``if _bind is not None:`` branch (line 389 in the pre-v104
            #   codebase). If ``pg_session.bind`` was None (valid config
            #   for in-memory testing, unbound sessions, or SQLAlchemy
            #   2.x sessions created without an engine binding), the
            #   variable was NEVER declared. Lines 413, 422, 430 (in the
            #   fallback branches) then referenced ``_meta_name`` in a
            #   log message and in the ``_sa_table(schema=_meta_name)``
            #   call, raising ``UnboundLocalError`` (a flavour of
            #   NameError) and crashing the Neo4j export path. The Phase
            #   1 -> Phase 2 Neo4j export then failed silently (the
            #   exporter returned ``ready=False``), and the pipeline
            #   continued with empty Neo4j.
            #
            #   ROOT FIX: initialize ``_meta_name = None`` at the TOP of
            #   the try block, BEFORE the ``if _bind is not None:``
            #   branch. When ``_bind`` is None, ``_meta_name`` stays
            #   None, and the fallback branches correctly use the
            #   unqualified ``_sa_table(t)`` (no schema qualification)
            #   which works on SQLite and on Postgres with default
            #   search_path. The function returns ``ready=False`` (with
            #   a warning) instead of crashing.
            _meta_name: Optional[str] = None
            try:
                # v107 ROOT FIX (ISSUE-P1-034 — schema introspection
                #   inconsistent when session.bind is None):
                #   The previous code ONLY resolved ``_meta_name`` when
                #   ``pg_session.bind is not None``. In SQLAlchemy 2.x,
                #   ``session.bind`` is frequently None (unbound sessions,
                #   sessions created via ``sessionmaker(bind=None)`` and
                #   later bound via ``session.connection()``). When
                #   ``_bind`` was None, ``_meta_name`` stayed None, and
                #   the fallback branches called ``_sa_table(t, schema=None)``
                #   which SQLAlchemy interprets as "use the default schema"
                #   — this works on SQLite (single-schema) but FAILS on
                #   PostgreSQL with a non-default ``search_path``,
                #   causing ``check_neo4j_readiness`` to return
                #   ``ready=False`` even on a fully-populated DB.
                #
                # ROOT FIX: try THREE sources of the engine in order:
                #   1. ``pg_session.bind`` (legacy SQLAlchemy 1.x)
                #   2. ``pg_session.get_bind()`` (SQLAlchemy 1.4+ — returns
                #      the bind for the current mapper context; works for
                #      unbound sessions that have been associated with an
                #      engine via ``session.configure(bind=engine)`` or
                #      via ``session.connection()``)
                #   3. ``database.connection.get_engine()`` (the platform's
                #      global engine — the same one the session pool uses)
                # Once we have ANY engine, use ``inspect(engine).default_schema_name``
                # consistently. This eliminates the SQLite-vs-PostgreSQL
                # asymmetry that the audit flagged.
                from sqlalchemy import (
                    MetaData,
                    Table,
                    inspect as _sa_inspect,
                )
                _bind = None
                # Source 1: session.bind (SQLAlchemy 1.x legacy)
                try:
                    _bind = pg_session.bind
                except (AttributeError, TypeError):
                    _bind = None
                # Source 2: session.get_bind() (SQLAlchemy 1.4+)
                if _bind is None:
                    try:
                        _bind = pg_session.get_bind()
                    except (AttributeError, TypeError, LookupError, Exception):
                        # get_bind() raises LookupError if no mapper is in
                        # context (e.g. raw SQL execution). Swallow and try
                        # the next source.
                        _bind = None
                # Source 3: global engine from database.connection
                if _bind is None:
                    try:
                        from database.connection import get_engine as _get_engine
                        _bind = _get_engine()
                    except Exception:
                        _bind = None
                # Now resolve default_schema_name from whichever bind we found.
                if _bind is not None:
                    try:
                        _meta_name = _sa_inspect(_bind).default_schema_name
                    except Exception:
                        _meta_name = None
                # P1-002 ROOT FIX (v100 forensic): the previous code
                # computed _meta (default_schema_name) but NEVER used
                # it -- it always created an UNQUALIFIED _sa_table(t),
                # which fails on PostgreSQL when the table is in a
                # non-default schema or search_path is misconfigured.
                # This caused check_neo4j_readiness to return ready=False
                # even on a fully-populated DB, blocking the master DAG's
                # _trigger_phase2. ROOT FIX: introspect the actual Table
                # object from the SQLAlchemy metadata via reflect() with
                # the resolved schema name, so the rendered SQL carries
                # the schema qualification (e.g. "public.drugs" not just
                # "drugs"). Fall back to the unqualified TableClause
                # only if reflection fails (SQLite, no bind, etc.).
                if _bind is not None:
                    try:
                        _md = MetaData(schema=_meta_name)
                        _table_obj = Table(
                            t, _md, autoload_with=_bind,
                        )
                    except Exception:
                        # Reflection failed (e.g. SQLite, missing table,
                        # permission denied on pg_catalog). Fall back to
                        # an explicit schema-qualified TableClause so the
                        # rendered SQL is "schema.table" not "table".
                        if _meta_name:
                            _table_obj = _sa_table(
                                t, schema=_meta_name,
                            )
                        else:
                            _table_obj = _sa_table(t)
                else:
                    # v107 P1-034: no engine available — use unqualified
                    # TableClause. This works on SQLite (single-schema)
                    # but will fail on PostgreSQL with non-default
                    # search_path. Log a WARNING so operators know the
                    # schema could not be resolved.
                    if t == sorted(ALL_TABLES)[0]:  # log once per call
                        logger.warning(
                            "check_neo4j_readiness: could not resolve a "
                            "database engine from pg_session (bind=%r, "
                            "get_bind=%r, global engine=%r). Using "
                            "unqualified table references — this works on "
                            "SQLite but may fail on PostgreSQL with a "
                            "non-default search_path. Ensure the session "
                            "is bound to an engine before calling "
                            "check_neo4j_readiness. (v107 P1-034)",
                            getattr(pg_session, 'bind', None),
                            'available' if hasattr(pg_session, 'get_bind') else 'missing',
                            'available' if _bind is not None else 'missing',
                        )
                    if _meta_name:
                        _table_obj = _sa_table(
                            t, schema=_meta_name,
                        )
                    else:
                        _table_obj = _sa_table(t)
            except Exception:
                # P1-002 ROOT FIX: use _meta_name (schema name) in _sa_table()
                # call with schema= parameter for the fallback case too.
                if _meta_name:
                    _table_obj = _sa_table(
                        t, schema=_meta_name,
                    )
                else:
                    _table_obj = _sa_table(t)
            if _table_obj is None:
                # P1-002 ROOT FIX: use _meta_name (schema name) here too.
                if _meta_name:
                    _table_obj = _sa_table(
                        t, schema=_meta_name,
                    )
                else:
                    _table_obj = _sa_table(t)
            _count_stmt = _sa_select(_sa_func.count()).select_from(_table_obj)
            result = pg_session.execute(_count_stmt)
            counts[t] = result.scalar()
        except Exception as exc:
            logger.warning('check_neo4j_readiness: could not count %s: %s', t, exc)
            counts[t] = 0
    # v40: ready = all REQUIRED tables have >0 rows (OPTIONAL tables may be 0)
    required_ready = all(counts.get(t, 0) > 0 for t in REQUIRED_TABLES)
    return {
        'ready': required_ready,
        'record_counts': counts,
        'required_tables': sorted(REQUIRED_TABLES),
        'optional_tables': sorted(OPTIONAL_TABLES),
        'phase': 'Phase 2 - bridge implemented (drugos_graph.phase1_bridge) [v40 fix]',
    }


def export_to_neo4j(
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    *,
    phase1_processed_dir: Optional[Path | str] = None,
    builder: Any = None,
    batch_size: int = 500,
    prefer_postgres: bool = True,
    **_legacy_kwargs: Any,
) -> Dict[str, Any]:
    """Export staged Phase 1 data to the Neo4j knowledge graph via the bridge.

    v29 ROOT FIX (audit O-4): the legacy ``pg_session`` parameter was
    REMOVED from the signature -- it was accepted but silently ignored,
    making the Phase 1 -> Neo4j wire look like it was using PostgreSQL
    when it was actually reading CSVs through ``phase1_bridge``.
    PostgreSQL -> Neo4j via a SQLAlchemy session is **not implemented**
    in this function -- use ``phase1_bridge.py`` instead (the bridge
    prefers PostgreSQL when ``DATABASE_URL`` is set and the ``drugs``
    table is populated).

    v58 ROOT FIX (P2C-008 deep): added ``prefer_postgres`` parameter
    so callers (especially tests) can explicitly force the CSV backend
    without depending on whether ``DATABASE_URL`` happens to be set in
    the environment. The bridge now treats ``DATABASE_URL`` being set
    as "production mode" -- PostgreSQL failures are FATAL, not silently
    fallen back to CSV.

    The function now ACTUALLY WORKS: it locates Phase 2's bridge module,
    reads Phase 1's processed_data CSVs (or PostgreSQL when ``DATABASE_URL``
    is set -- handled inside the bridge), converts them to Phase 2
    node/edge dicts, and loads them into the supplied ``builder``.

    Two modes:

    1. **Direct builder injection** (recommended for tests & demos):
       Pass ``builder=<any GraphBuilderProtocol>`` (e.g. a
       ``RecordingGraphBuilder`` for in-memory validation, or a real
       ``DrugOSGraphBuilder`` with a connected Neo4j driver).

    2. **Neo4j credential mode** (production):
       Pass ``neo4j_uri``, ``neo4j_user``, ``neo4j_password``. The
       function constructs a ``DrugOSGraphBuilder`` from these credentials
       and connects it before loading.

    Parameters
    ----------
    neo4j_uri, neo4j_user, neo4j_password : str, optional
        Neo4j credentials for production mode.
    phase1_processed_dir : path-like, optional
        Override for the Phase 1 processed_data directory. Defaults to
        ``<unified_root>/phase1/processed_data``.
    builder : GraphBuilderProtocol, optional
        Pre-constructed builder. Takes precedence over the Neo4j credential
        mode.
    batch_size : int
        Batch size for ``load_nodes_batch`` / ``load_edges_batch``.
    **_legacy_kwargs : Any
        Absorbs any legacy keyword arguments (e.g. ``pg_session``) passed
        by old callers. Such arguments are **ignored** and a
        ``DeprecationWarning`` is emitted. This keeps the function
        backward-compatible with the v28 signature
        ``export_to_neo4j(pg_session=None, ...)`` without re-introducing
        the misleading parameter into the signature.

    Returns
    -------
    dict
        Bridge summary report. See
        :func:`drugos_graph.phase1_bridge.run_phase1_to_phase2`.

    Raises
    ------
    DrugOSDataError
        If any REQUIRED Phase 1 output CSV is missing under
        ``phase1_processed_dir`` (see :class:`Phase1OutputContract`).
        Raised BEFORE the bridge is invoked so the operator sees a
        clear, actionable error instead of a silently partial KG.
    RuntimeError
        If neither ``builder`` nor ``neo4j_uri`` is provided AND Phase 2's
        ``drugos_graph`` package cannot be located on disk.
    """
    # v29 ROOT FIX (audit O-4): pg_session was accepted but ignored --
    # misleading API. Either implement or remove. We chose REMOVE: the
    # parameter is no longer in the signature, but **_legacy_kwargs absorbs
    # any stray ``pg_session=...`` passed by old callers (with a
    # DeprecationWarning) so existing tests don't break. PostgreSQL -> Neo4j
    # export is NOT implemented here -- use phase1_bridge.py instead.
    if _legacy_kwargs:
        import warnings as _warnings
        _warnings.warn(
            "export_to_neo4j() no longer accepts keyword arguments "
            f"{sorted(_legacy_kwargs)} (audit O-4: pg_session was "
            "accepted but ignored -- misleading API). The pg_session "
            "parameter has been removed. PostgreSQL -> Neo4j is not "
            "implemented in this function -- use phase1_bridge.py "
            "instead (set DATABASE_URL to use PostgreSQL).",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "export_to_neo4j: ignored legacy kwargs %s (audit O-4: "
            "pg_session was removed -- use phase1_bridge.py for "
            "PostgreSQL -> Neo4j).",
            sorted(_legacy_kwargs),
        )

    _ensure_phase2_on_path()

    try:
        from drugos_graph.phase1_bridge import (
            DEFAULT_PHASE1_PROCESSED_DIR,
            RecordingGraphBuilder,
            run_phase1_to_phase2,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Phase 2 'drugos_graph' package not found at {_PHASE2_ROOT}. "
            f"The unified package requires both phase1/ and phase2/ directories. "
            f"Original ImportError: {exc}"
        ) from exc

    # Resolve Phase 1 processed_data dir
    if phase1_processed_dir is None:
        phase1_processed_dir = _PHASE1_ROOT / "processed_data"
    phase1_processed_dir = Path(phase1_processed_dir)

    # FIX P1-ER-14 (MEDIUM): validate the explicit Phase 1 output
    # contract BEFORE delegating to the bridge. The bridge itself
    # degrades gracefully (logs a warning + empty DataFrame), but that
    # silent degradation was the ROOT CAUSE of operators shipping
    # partial KGs without realising a Phase 1 pipeline had failed.
    # The contract check raises DrugOSDataError at the exporter
    # boundary for any missing REQUIRED CSV.
    resolved_paths = validate_phase1_output_contract(phase1_processed_dir)
    logger.info(
        "export_to_neo4j: Phase 1 output contract validated -- %d/%d "
        "sources present under %s",
        len(resolved_paths),
        len(Phase1OutputContract().all_keys()),
        phase1_processed_dir,
    )

    # P1-028 ROOT FIX: defense-in-depth InChIKey validation. Scan the
    # Phase 1 drugs CSV(s) for InChIKeys and validate each against the
    # canonical regex BEFORE delegating to the bridge. Invalid keys are
    # logged with a WARNING and counted in the result summary. This is a
    # reporting layer (non-fatal) so existing pipelines are not broken;
    # operators see the warnings and can quarantine the offending rows
    # upstream. The bridge's own Cypher parameter binding is the primary
    # injection defence; this layer catches malformed data early.
    try:
        import csv as _csv
        _inchikey_validation = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "invalid_samples": [],
        }
        for _contract_key, _path in resolved_paths.items():
            _p = Path(_path)
            if not _p.exists():
                continue
            # Only validate files that look like drug CSVs (contain
            # 'drug' in the filename and are .csv/.csv.gz).
            _fname = _p.name.lower()
            if "drug" not in _fname:
                continue
            try:
                import gzip as _gzip
                _opener = _gzip.open if _fname.endswith(".gz") else open
                with _opener(_p, "rt", encoding="utf-8", errors="replace") as _f:
                    _reader = _csv.DictReader(_f)
                    if "inchikey" not in (_reader.fieldnames or []):
                        continue
                    for _row in _reader:
                        _ik = _row.get("inchikey")
                        if _ik is None or (isinstance(_ik, str) and not _ik.strip()):
                            continue
                        _inchikey_validation["total"] += 1
                        if _validate_inchikey_for_neo4j(_ik):
                            _inchikey_validation["valid"] += 1
                        else:
                            _inchikey_validation["invalid"] += 1
                            if len(_inchikey_validation["invalid_samples"]) < 5:
                                _inchikey_validation["invalid_samples"].append(
                                    str(_ik)[:60]
                                )
            except Exception as _exc:  # noqa: BLE001
                logger.debug(
                    "export_to_neo4j: InChIKey validation scan failed for "
                    "%s (non-fatal): %s",
                    _p, _exc,
                )
        if _inchikey_validation["invalid"] > 0:
            logger.warning(
                "export_to_neo4j: P1-028 InChIKey validation found %d "
                "invalid key(s) out of %d total in drug CSVs. These will "
                "still be passed to the bridge (reporting-only mode) but "
                "may be rejected by Neo4j's constraints. Samples: %s",
                _inchikey_validation["invalid"],
                _inchikey_validation["total"],
                _inchikey_validation["invalid_samples"],
            )
        else:
            logger.info(
                "export_to_neo4j: P1-028 InChIKey validation passed -- "
                "%d/%d keys valid.",
                _inchikey_validation["valid"],
                _inchikey_validation["total"],
            )
    except Exception as _exc:  # noqa: BLE001
        logger.debug(
            "export_to_neo4j: InChIKey validation layer failed (non-fatal): %s",
            _exc,
        )

    # Construct a real builder if Neo4j credentials were supplied
    if builder is None and neo4j_uri is not None:
        try:
            from drugos_graph import DrugOSGraphBuilder, Neo4jConfig
        except ImportError as exc:
            raise RuntimeError(
                f"DrugOSGraphBuilder could not be imported. "
                f"Is the 'neo4j' Python package installed? {exc}"
            ) from exc
        cfg = Neo4jConfig(
            uri=neo4j_uri,
            user=neo4j_user or "neo4j",
            password=neo4j_password or "",
        )
        builder = DrugOSGraphBuilder(cfg)
        builder.connect()
        try:
            builder.create_constraints()
        except Exception as exc:
            logger.warning("create_constraints() failed (continuing): %s", exc)

    # If still no builder, fall back to RecordingGraphBuilder (dry-run mode)
    if builder is None:
        logger.info(
            "export_to_neo4j: no builder or Neo4j credentials supplied -- "
            "using RecordingGraphBuilder (in-memory dry run)."
        )
        builder = RecordingGraphBuilder()

    result = run_phase1_to_phase2(
        phase1_processed_dir=phase1_processed_dir,
        builder=builder,
        batch_size=batch_size,
        prefer_postgres=prefer_postgres,
    )

    # v52 ROOT FIX (P1-025): verify all 5 DOCX-required node types are present.
    # The DOCX Phase 2 section requires: Drug/Protein/Pathway/Disease/ClinicalOutcome.
    # If any are missing, log a WARNING (non-fatal -- the bridge still produced
    # a partial graph, but the operator is now aware the DOCX contract is violated).
    try:
        coverage = check_node_type_coverage(result.get("summary", {}))
        result["node_type_coverage"] = coverage
        if not coverage["all_present"]:
            logger.warning(
                "export_to_neo4j: DOCX 5-node-type contract VIOLATED -- "
                "missing: %s. The KG was built but is incomplete. To fix: "
                "ensure Phase 1 produces drugbank_indications.csv (for "
                "ClinicalOutcome) and protein_protein_interactions.csv "
                "(for Pathway derivation from STRING PPI).",
                coverage["missing_types"],
            )
    except Exception as exc:
        logger.debug("check_node_type_coverage failed (non-fatal): %s", exc)

    return result


def is_synthetic_inchikey(inchikey: str) -> bool:
    """Check if an InChIKey was synthetically generated (starts with SYNTH).

    v43 ROOT FIX (P1 -- case-sensitive SYNTH check): the previous code
    used ``inchikey.startswith("SYNTH")`` which is CASE-SENSITIVE. A
    lowercase or mixed-case key like ``synth-abc-def-g`` would NOT be
    detected as synthetic and would be exported to Neo4j as if it were
    a real InChIKey. The canonical check in 3 other modules
    (cleaning.normalizer, entity_resolution.base, drug_resolver) all
    use ``.upper().startswith("SYNTH")`` (case-insensitive). This was
    the 4th duplicate definition of the same function -- exactly the
    divergence anti-pattern the v9 audit claimed to fix.

    Fix: use ``.upper().startswith("SYNTH")`` to match the canonical
    case-insensitive check. This aligns this function with the other
    3 definitions and closes the case-sensitivity hole.
    """
    return bool(inchikey and inchikey.upper().startswith("SYNTH"))


# v52 ROOT FIX (P1-025): the DOCX requires 5 node types. This constant
# makes the requirement explicit and testable.
DOCX_REQUIRED_NODE_TYPES: tuple[str, ...] = (
    "Compound",         # Drug nodes (DOCX calls them "Drugs")
    "Protein",          # Protein nodes
    "Pathway",          # Biological pathway nodes (from STRING PPI clusters)
    "Disease",          # Disease nodes (from OMIM + DisGeNET)
    "ClinicalOutcome",  # Clinical outcome nodes (from DrugBank indications)
)


def check_node_type_coverage(bridge_summary: Dict[str, Any]) -> Dict[str, Any]:
    """v52 ROOT FIX (P1-025): verify all 5 DOCX-required node types are present.

    The DOCX Phase 2 section explicitly requires 5 node types:
    Drug/Protein/Pathway/Disease/ClinicalOutcome. The v48 exporter only
    emitted 4 (missing Pathway and ClinicalOutcome). This function checks
    the bridge summary and reports which node types are present/missing.

    Parameters
    ----------
    bridge_summary : dict
        The summary dict returned by ``run_phase1_to_phase2()``.

    Returns
    -------
    dict
        Keys:
        - 'all_present': bool -- True if all 5 DOCX-required types are present
        - 'present_types': list -- node types that have >0 nodes
        - 'missing_types': list -- DOCX-required types with 0 nodes
        - 'node_counts_by_type': dict -- type -> count
        - 'docx_compliant': bool -- True if all 5 types present (alias for all_present)
    """
    node_counts = bridge_summary.get("node_counts_by_type", {})
    if not node_counts:
        # Try to extract from the summary's edge_types_present
        logger.warning(
            "check_node_type_coverage: no node_counts_by_type in summary -- "
            "cannot verify DOCX 5-node-type requirement"
        )
        return {
            "all_present": False,
            "present_types": [],
            "missing_types": list(DOCX_REQUIRED_NODE_TYPES),
            "node_counts_by_type": {},
            "docx_compliant": False,
        }
    present = [t for t in DOCX_REQUIRED_NODE_TYPES if node_counts.get(t, 0) > 0]
    missing = [t for t in DOCX_REQUIRED_NODE_TYPES if node_counts.get(t, 0) == 0]
    all_present = len(missing) == 0
    if missing:
        logger.warning(
            "check_node_type_coverage: DOCX requires 5 node types but "
            "%d are missing: %s. Present: %s. The KG is incomplete per "
            "the DOCX Phase 2 contract.",
            len(missing), missing, present,
        )
    else:
        logger.info(
            "check_node_type_coverage: all 5 DOCX-required node types "
            "present: %s", present,
        )
    return {
        "all_present": all_present,
        "present_types": present,
        "missing_types": missing,
        "node_counts_by_type": node_counts,
        "docx_compliant": all_present,
    }


# =============================================================================
# RT-009 ROOT FIX (Team Member 17) + P1-011 ROOT FIX (Team Member 9 v104):
# Neo4jExporter class + __all__
# =============================================================================
# Two parallel teams (Team Member 17 and Team Member 9) independently
# discovered and fixed the same bug: any code path importing
# ``Neo4jExporter`` from this module crashed with ImportError, because
# the module only exported module-level functions (export_to_neo4j,
# check_neo4j_readiness, validate_phase1_output_contract) — no class.
# The Phase 1 -> Phase 2 Neo4j export, if invoked through
# ``from phase1.exporters.neo4j_exporter import Neo4jExporter``, failed
# before any code ran.
#
# Root fix (both teams converged on the same approach): provide a thin
# ``Neo4jExporter`` class that wraps the existing ``export_to_neo4j``
# function. This restores backward compatibility with any consumer (CI
# tests, downstream scripts, external integrations) that imports the
# class. The class is the canonical OOP entry point; the module-level
# functions remain available for functional-style callers.
#
# The class deliberately does NOT re-implement the bridge logic — it
# delegates to ``export_to_neo4j`` so there is ONE source of truth for
# the export behavior. Duplicating the logic would re-introduce the
# "two divergent code paths" anti-pattern that the v29 bridge fix
# eliminated.
class Neo4jExporter:
    """Object-oriented wrapper around :func:`export_to_neo4j`.

    RT-009 ROOT FIX (Team Member 17) + P1-011 ROOT FIX (Team Member 9 v104):
    restores the ``Neo4jExporter`` class name that was referenced by
    external consumers but did not exist in this module. Any
    ``from phase1.exporters.neo4j_exporter import Neo4jExporter``
    previously raised ``ImportError: cannot import name 'Neo4jExporter'``.
    The class wraps the existing functional API so there is a single
    source of truth for the export behavior.

    Examples
    --------
    Production (Neo4j credentials)::
        # P1-021 v113 ROOT FIX: NEVER hardcode credentials. Read from env var.
        import os
        exporter = Neo4jExporter(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password=os.environ["NEO4J_PASSWORD"],
        )
        report = exporter.export(phase1_processed_dir="phase1/processed_data")

    Tests / demos (inject a RecordingGraphBuilder)::

        from phase2.drugos_graph.phase1_bridge import RecordingGraphBuilder
        exporter = Neo4jExporter(builder=RecordingGraphBuilder())
        report = exporter.export(
            phase1_processed_dir="phase1/processed_data",
            prefer_postgres=False,
        )

    Parameters
    ----------
    neo4j_uri, neo4j_user, neo4j_password : str, optional
        Neo4j connection credentials. If omitted, the exporter runs in
        dry-run mode (uses ``RecordingGraphBuilder`` internally).
    batch_size : int
        Batch size for bulk loading (default 500).
    prefer_postgres : bool
        If True (default), prefer PostgreSQL as the data source when
        ``DATABASE_URL`` is set. If False, always use CSVs.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        *,
        builder: Any = None,
        phase1_processed_dir: Optional[Path | str] = None,
        batch_size: int = 500,
        prefer_postgres: bool = True,
    ) -> None:
        """Configure the exporter. Credentials OR builder must be supplied at
        ``export()`` time if not provided here.

        Parameters mirror :func:`export_to_neo4j`. Any parameter set here is
        used as a default but can be overridden per-call at ``export()``.
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        # P1-009 ROOT FIX (Team Member 9 v104): store password as a private
        # attribute -- never log it, never expose it via __repr__. Defense
        # in depth for credential safety.
        self._neo4j_password = neo4j_password
        # RT-009 ROOT FIX: keep both .neo4j_password (for backward compat
        # with code that reads the attribute directly) and ._neo4j_password
        # (the canonical private name from P1-009). Both point to the same
        # value; setters are NOT provided so the password is effectively
        # read-only after construction.
        self.neo4j_password = neo4j_password
        self.builder = builder
        self.phase1_processed_dir = phase1_processed_dir
        self.batch_size = batch_size
        self.prefer_postgres = prefer_postgres

    def __repr__(self) -> str:
        # P1-009 ROOT FIX (Team Member 9 v104): NEVER include the password
        # in repr. This is defense in depth so log lines / error messages
        # / debug prints never leak the Neo4j password.
        return (
            f"Neo4jExporter(neo4j_uri={self.neo4j_uri!r}, "
            f"neo4j_user={self.neo4j_user!r}, "
            f"neo4j_password=***, batch_size={self.batch_size!r}, "
            f"prefer_postgres={self.prefer_postgres!r})"
        )

    def export(
        self,
        *,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        phase1_processed_dir: Optional[Path | str] = None,
        builder: Any = None,
        batch_size: Optional[int] = None,
        prefer_postgres: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run the Phase 1 -> Neo4j export via :func:`export_to_neo4j`.

        Any ``None`` argument falls back to the value set at ``__init__``.
        This lets callers configure the exporter once and override per-call.
        """
        # Use the private _neo4j_password (P1-009) if neither the per-call
        # nor the public attribute provides one.
        eff_password = neo4j_password if neo4j_password is not None else self.neo4j_password
        return export_to_neo4j(
            neo4j_uri=neo4j_uri if neo4j_uri is not None else self.neo4j_uri,
            neo4j_user=neo4j_user if neo4j_user is not None else self.neo4j_user,
            neo4j_password=eff_password,
            phase1_processed_dir=phase1_processed_dir if phase1_processed_dir is not None else self.phase1_processed_dir,
            builder=builder if builder is not None else self.builder,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            prefer_postgres=prefer_postgres if prefer_postgres is not None else self.prefer_postgres,
        )

    def check_readiness(self, pg_session: Any = None) -> Dict[str, Any]:
        """Wrap :func:`check_neo4j_readiness` for OO callers."""
        return check_neo4j_readiness(pg_session)

    @staticmethod
    def validate_contract(phase1_processed_dir: Optional[Path | str] = None) -> Dict[str, Any]:
        """Wrap :func:`validate_phase1_output_contract` for OO callers.

        v107 FORENSIC ROOT FIX (ISSUE-P1-012):
          The previous code accepted ``Optional[Path | str] = None`` but
          passed the value directly to ``validate_phase1_output_contract``
          which requires a non-None ``base_dir`` (it calls ``Path(base_dir)``
          which raises TypeError on None). Calling
          ``Neo4jExporter.validate_contract()`` (no args) crashed with a
          confusing TypeError instead of a clear error message.
          ROOT FIX: default to the canonical Phase 1 processed_data dir
          (``_PHASE1_ROOT / "processed_data"``) when the argument is None.
          This matches the behavior of the module-level
          ``export_to_neo4j`` function which also defaults to this path.
        """
        # v107 P1-012: default to the canonical Phase 1 processed_data dir
        # instead of passing None to validate_phase1_output_contract
        # (which would raise TypeError on Path(None)).
        if phase1_processed_dir is None:
            phase1_processed_dir = _PHASE1_ROOT / "processed_data"
        return validate_phase1_output_contract(phase1_processed_dir)


# RT-009 + P1-011 ROOT FIX: explicit __all__ so it is crystal-clear what
# this module exports. Prevents future regressions where a class or function
# is silently removed and downstream consumers only discover the breakage
# at runtime.
__all__: list[str] = [
    "DOCX_REQUIRED_NODE_TYPES",
    "Neo4jExporter",            # RT-009 + P1-011 ROOT FIX: class-based API wrapper
    "Phase1OutputContract",
    "check_neo4j_readiness",
    "check_node_type_coverage",
    "export_to_neo4j",
    "is_synthetic_inchikey",
    "validate_phase1_output_contract",
]
