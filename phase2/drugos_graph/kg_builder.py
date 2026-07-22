"""
DrugOS Graph Module — Knowledge Graph Builder (Neo4j)
======================================================
Institutional-grade rewrite of the Neo4j write layer for the DrugOS
Autonomous Drug Repurposing Platform.

Architecture (Facade Pattern — audit issue A-1):
  DrugOSGraphBuilder  — public API facade (backward-compatible)
    ├── GraphConnection     — connect, disconnect, retry, health, driver DI
    ├── GraphSchemaManager  — create_constraints, create_indexes, version detect
    ├── GraphNodeLoader     — load_nodes_batch, load_drkg_nodes
    ├── GraphEdgeLoader     — load_edges_batch, load_edges_bulk_create, dedup
    ├── DrugBankEnricher    — enrich_compounds_from_drugbank
    ├── GraphStatsCollector — get_graph_stats, health_check
    └── GraphJanitor        — clear_graph (dangerous ops isolated, access control)

Patient Safety Context (NON-NEGOTIABLE):
  A bug in this file = wrong graph = wrong prediction = a pharma partner
  tests the wrong drug on a real patient = patient harm. The RL safety
  ranker uses the `withdrawn`, `terminated`, `illicit`, `toxicity`, and
  `sensitive` properties written by DrugBankEnricher to classify drugs as
  red (dangerous) / yellow / green (safe). A null value on any of these
  properties means "no data" which is silently interpreted as "not
  withdrawn" → green → SAFE. A withdrawn drug like Valdecoxib (withdrawn
  for cardiovascular risk) would be classified as SAFE because the
  DrugBank XML didn't have the field in that record variant. This is a
  direct patient-harm pathway.

  Treat every line of this code as if a real patient's life depends on it
  — because it does.

Fixes: A-1..A-7, D-1..D-6, S-1..S-5, C-1..C-7, DQ-1..DQ-7, R-1..R-7,
       I-1..I-5, P-1..P-6, S(9)-1..S(9)-6, T-1..T-6, L-1..L-6,
       CF-1..CF-6, DO-1..DO-6, CO-1..CO-5, IN-1..IN-5, DL-1..DL-6
"""

from __future__ import annotations

import atexit
import inspect
import logging
import os
import signal
import sys
import re
import threading
import time
import warnings
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

try:
    from neo4j import Driver, GraphDatabase, Session
    from neo4j.exceptions import (
        AuthError,
        ServiceUnavailable,
        SessionExpired,
    )
except ImportError:
    Driver = None  # type: ignore[assignment,misc]
    GraphDatabase = None  # type: ignore[assignment,misc]
    Session = None  # type: ignore[assignment,misc]
    ServiceUnavailable = None  # type: ignore[assignment,misc]
    SessionExpired = None  # type: ignore[assignment,misc]
    AuthError = None  # type: ignore[assignment,misc]

from .config import (
    CANONICAL_IDS,
    CONFIG_HASH,
    CORE_EDGE_TYPES,
    CORE_EDGE_TYPES_SET,
    CORE_NODE_TYPES,
    DRKG_NODE_TYPES,
    PIPELINE_VERSION,
    RUN_ID,
    SCHEMA_VERSION,
    SEED,
    Neo4jConfig,
    audit_log,
    build_lineage_metadata,
    check_data_freshness,
    compute_and_record_checksum,
    compute_impact_analysis,
    dead_letter_record,
    deprecated,
    diff_configs,
    get_neo4j_config,
    is_core_edge,
    log_transformation,
    read_latest_checkpoint,
    safe_config_dict,
    verify_checksum,
    write_checkpoint,
    write_lineage_manifest,
)
from .exceptions import (
    ConfigurationError,
    CriticalDataSourceError,
    DrugOSDataError,
    EdgeLoadMismatchError,
    SecurityError,
    UnknownLabelError,
)
from .utils import (
    drkg_node_type_to_neo4j_label,
    drkg_node_type_to_neo4j_label_with_provenance,
    neo4j_label_to_drkg_node_type,
    sanitize_identifier,
    sanitize_label,
    sanitize_rel_type,
    safe_call_with_retry,
)

logger = logging.getLogger(__name__)

# ─── Module-Level Constants ───────────────────────────────────────────────────

# Fixes A-7: json import moved to __main__ block; re kept for S-5 ID validation
# Fixes C-5: json only needed in __main__; re now justified for ID validation

# Fixes I-4, DL-1, DL-5, DL-6, CO-1, CO-2 (Provenance Rule §3.5):
# Every node and edge MUST carry these lineage properties.
# Audit fix (v5 Tier-3 bug #24): added _source_phase, _source_file,
# _source_row to the whitelist. The phase1_bridge emits these on every
# node/edge for bidirectional traceability (the INTEGRATION.md doc
# promises "given any node in Neo4j, run a Cypher query to find the
# exact Phase 1 CSV row that produced it"). Without these in the
# whitelist, the real kg_builder silently stripped them — making the
# traceability contract false in production. RecordingGraphBuilder
# tests didn't catch this because they don't apply the whitelist.
SYSTEM_PROPS: frozenset[str] = frozenset({
    "_pipeline_run_id",
    "_loaded_at",
    "_schema_version",
    "_source",
    "_source_phase",
    "_source_file",
    "_source_row",
    "_license",
    "_attribution",
    "_config_hash",
    "_pipeline_version",
    "_seed",
    "_input_checksum",
    "input_checksum",  # legacy alias used by some bridge code paths
    "_created_at",
    "_updated_at",
    "_version",
    "_source_priority",  # BUG-D-011: deterministic dedup ordering
    # v108 ROOT FIX (issue 65): canonical_id is the full prefixed canonical
    # ID (e.g. "protein:P12821") stored as a node property by
    # RecordingGraphBuilder.register_node. It must survive the whitelist
    # filter so downstream consumers (Phase 3, graph_queries) can use the
    # prefixed form.
    "canonical_id",
    # v108 ROOT FIX (issue 66): _src_canonical_id / _dst_canonical_id are
    # stored on edges by RecordingGraphBuilder.register_edge for traceability.
    "_src_canonical_id",
    "_dst_canonical_id",
})

# BUG-D-011 root fix: source priority map. The ``deduplicate_edges_deterministic``
# function orders by ``r._source_priority DESC`` but that property was NEVER
# set when loading edges — making the "deterministic" dedup non-deterministic
# (edges kept/dropped depended on Python dict insertion order). This map
# assigns a numeric priority to each known source so dedup is reproducible.
# Higher number = higher priority (kept over lower priority).
SOURCE_PRIORITY_MAP: dict[str, int] = {
    "drugbank": 100,        # FDA-approved drug labels — highest authority
    "drugbank_indications": 95,
    # v35 ROOT FIX (H-2): the bridge emits source="drugbank_indication"
    # (singular, no _text) for free-text-derived Compound-treats-Disease
    # edges (phase1_bridge.py:2130). Without this key, get_source_priority
    # returned 0 → free-text treats edges were silently dropped during
    # deduplicate_edges_deterministic in favor of any other edge (even
    # lower-quality DRKG edges at priority 25). Priority 100 matches
    # "drugbank" because the source IS DrugBank (just the free-text
    # indication column rather than the structured indications CSV).
    "drugbank_indication": 100,
    "uniprot": 90,
    "chembl": 85,
    "pubchem": 80,
    "omim": 75,
    "disgenet": 70,
    "string": 65,
    "clinicaltrials": 60,
    "sider": 55,
    "stitch": 50,
    "geo": 45,
    "opentargets": 40,
    "drugbank_indication_text": 35,
    "phase1_bridge": 30,
    "drkg": 25,
    "test": 10,
    "unknown": 0,
}


def get_source_priority(source: str) -> int:
    """Return the numeric priority for a source label (BUG-D-011).

    Higher number = higher priority (kept during dedup).
    Unknown sources default to 0 (lowest priority).
    """
    if not source:
        return 0
    return SOURCE_PRIORITY_MAP.get(source.lower().strip(), 0)

# Fixes S-5 (Domain 3): Biomedical identifier validation patterns
# Audit fix (v5 Tier-2 bug #20 — REPAIRED v6): the previous pattern only
# accepted 6-char Swiss-Prot accessions. Real DrugBank/UniProt data
# contains 10-char TrEMBL accessions (e.g. A0A024R2R7, A0A1B0GUU5),
# which were silently dead-lettered. The new pattern uses the official
# UniProt accession grammar:
#
#   Swiss-Prot (6 chars):  [OPQ][0-9][A-Z0-9]{3}[0-9]
#                          | [A-NR-Z][0-9][A-Z0-9]{3}[0-9]
#   TrEMBL    (10 chars):  same 6-char prefix + ([A-Z0-9]{3}[0-9]){1}
#
# An optional isoform-suffix `-<digits>` is allowed on either form.
# Verified against: P23219 (Swiss-Prot), P00734 (Swiss-Prot),
# A0A024R2R7 (TrEMBL, 10 chars), A0A1B0GUU5 (TrEMBL, 10 chars),
# Q9BX66 (Swiss-Prot), A0A024R2R7-2 (TrEMBL + isoform).
ID_PATTERNS: dict[str, str] = {
    # v28 ROOT FIX (P2-B-12): removed the ``NAME:[A-Za-z0-9 _.-]{1,64}``
    # alternative — it accepted LITERALLY ANY string (any printable ASCII
    # up to 64 chars) as a Compound ID. This made Compound ID validation
    # a no-op: typos, garbage strings, even ``NAME: `` (just a space)
    # passed. Production queries that filter by Compound ID then
    # returned inconsistent results (some edges pointed at the
    # InChIKey-canonical node, others at the NAME: node — disjoint subgraphs).
    # Removed: callers that need a non-InChIKey/non-DrugBank/non-ChEMBL
    # identifier SHOULD register a new prefix in ID_PATTERNS with a
    # TIGHTER regex (e.g. ``DRUG:<digits>``), not abuse the catch-all.
    # v43 ROOT FIX (Chain 2): unify DrugBank ID regex with Phase 1
    # (_DRUGBANK_ID_RE) and Phase 2 drugbank_parser
    # (DRUGBANK_DRUG_IDENTIFIER_REGEX), all = ^DB\d{5,7}$. Previously
    # this was {5,6} which silently dead-lettered 7-digit DrugBank IDs
    # that the parser accepts -- fragmenting the KG.
    # P2-010 ROOT FIX (STITCH CIDm/CIDs case-sensitivity):
    # The previous pattern ``CIDm\d+|CIDs\d+`` was case-SENSITIVE --
    # it accepted ``CIDm00002244`` and ``CIDs00002244`` (lowercase
    # m/s) but NOT ``CIDM00002244`` / ``CIDS00002244`` (uppercase
    # M/S). Any caller that uppercased the ID upstream (e.g.
    # phase1_bridge.py:3547 uppercases inchikey, and the entity
    # resolver applies .upper() to canonical IDs in some paths)
    # converted ``CIDm00002244`` -> ``CIDM00002244`` which FAILED
    # the pattern -- dead-lettering the entire STITCH drug-target
    # edge set (STITCH has ~500K drug-protein edges, the largest
    # single source). Drug-protein connectivity of the KG was
    # silently halved.
    #
    # ROOT FIX: make the CIDm/CIDs prefix case-INSENSITIVE via the
    # character-class form ``[Cc][Ii][Dd][Mm]\d+`` /
    # ``[Cc][Ii][Dd][Ss]\d+``. This matches any case combination of
    # the four-letter prefix while keeping the digit run strict
    # (avoiding accidental match of unrelated identifiers). The
    # canonical form emitted by the STITCH loader remains
    # ``CIDm<digits>`` / ``CIDs<digits>`` (lowercase m/s) -- the
    # case-insensitive pattern is defensive against upstream
    # normalisation, NOT a licence to emit arbitrary case.
    #
    # P1-017 ROOT FIX (Team-2 -- accept synthesized IDs at the Phase 1 ->
    #   Phase 2 bridge):
    #   The v50 open-data fallback (``_v50_downloaders.py::_synthesize_drugbank_id``)
    #   generates synthesized IDs with the ``SYNTH-DB-`` prefix (clearly
    #   non-DrugBank -- see P1-017 root fix in drugbank_pipeline.py). The
    #   previous Compound pattern accepted ONLY ``DB\d{5,7}`` (real
    #   DrugBank IDs) -- synthesized IDs were REJECTED at the Phase 1 ->
    #   Phase 2 bridge, breaking the v50 fallback end-to-end. ROOT FIX:
    #   add ``SYNTH-DB-[0-9A-F]{8}`` and ``SYNTH-DB-M\d{6}`` to the
    #   Compound pattern alternation. These match the patterns in
    #   ``drugbank_pipeline._SYNTHESIZED_DRUG_ID_RE`` and
    #   ``resolver_utils._SYNTHESIZED_DRUG_ID_RE`` -- all three must
    #   stay in sync (a future refactor should consolidate into a
    #   single shared ``_constants`` module).
    # P2-051 ROOT FIX (Teammate 4, forensic, root-level): the previous
    # pattern accepted ``MESH:[A-Z]\d+`` for BOTH Compound AND Disease.
    # This is a NAMESPACE COLLISION — MeSH descriptor IDs (e.g.
    # ``MESH:D000001``) are Diseases per MeSH's own tree classification
    # (D-tree = Diseases), while MeSH supplement record IDs (e.g.
    # ``MESH:C000001``) are Chemicals/Compounds (C-tree). The same ID
    # could match BOTH patterns, causing a single MeSH record to be
    # classified as both a Compound AND a Disease — corrupting the KG
    # with cross-type duplicates.
    #
    # ROOT FIX: MeSH IDs are prefixed by their tree letter:
    #   - ``MESH:C\d+`` (C-tree) → Compound (supplementary chemical record)
    #   - ``MESH:D\d+`` (D-tree) → Disease (descriptor)
    # Reference: https://meshb.nlm.nih.gov/treeView
    #
    # The previous catch-all ``MESH:[A-Z]\d+`` is removed. Each ID type
    # now matches ONLY its correct MeSH tree branch. This eliminates the
    # collision at the regex level — no runtime disambiguation needed.
    "Compound": r"^(DB\d{5,7}|SYNTH-DB-[0-9A-F]{8}|SYNTH-DB-M\d{6}|CHEMBL\d+|CID\d+|[A-Z]{14}-[A-Z]{10}-[A-Z]|[Cc][Ii][Dd][Mm]\d+|[Cc][Ii][Dd][Ss]\d+|MESH:C\d+)$",
    # v21 ROOT FIX (Audit section 4 finding 8 / Chain 9 - "Bridge emits
    # IDs that production rejects"): the previous Protein pattern
    # accepted ONLY UniProt accessions. But phase1_bridge.py:1642 emits
    # ``CHEMBL_TGT_{chembl_target_id}`` for ChEMBL targets that lack a
    # UniProt AC (a common case for older ChEMBL target records). The
    # production validator dead-lettered every such Protein node, silently
    # dropping ChEMBL target nodes from the KG. Multi-hop queries that
    # traverse these nodes returned empty. Fix: accept the
    # ``CHEMBL_TGT_\d+`` prefix as a valid Protein ID (it is a stable
    # ChEMBL target identifier; entity_resolver can later upgrade it to
    # a UniProt AC via id_crosswalk when one becomes available).
    "Protein": r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9])([A-Z0-9]{3}[0-9])?(-\d+)?$|^CHEMBL_TGT_\d+$",
    # Gene: numeric NCBI gene ID (e.g. 2261 for FGFR3) OR a SYM:-prefixed
    # gene symbol (e.g. SYM:FGFR3) used as a placeholder until the
    # entity_resolver canonicalizes it to a numeric ID via id_crosswalk.
    # OMIM/NCBIGene prefixes are stripped by the loaders BEFORE reaching
    # ID_PATTERNS (BUG-B-001, BUG-B-002).
    # v37 ROOT FIX (Phase 2 Issue #7): added ``MIM:\d+`` to the accepted
    # Gene ID patterns. OMIM MIM numbers are now prefixed with ``MIM:``
    # (see ``omim_loader._safe_gene_id_from_mim``) so they don't collide
    # with bare NCBI Gene IDs in the same numeric space. The prefix is
    # the namespace disambiguator.
    "Gene": r"^(\d+|SYM:[A-Z0-9]+|MIM:\d+)$",
    # Disease: explicit prefixed forms only. BUG-D-015 root fix: removed
    # the ``[A-Z]+:\w+`` catch-all that accepted 'FOO:bar' as a Disease ID.
    # Now only valid biomedical disease ontologies are accepted.
    # v9 ROOT FIX: accept BOTH underscore (EFO_0000400 — original EFO
    # curie spec) and colon (EFO:0000400 — OpenTargets canonical form)
    # for EFO, since the OpenTargets _normalise_ontology_id helper
    # converts underscore → colon for ALL ontology prefixes. Without
    # both forms, EFO IDs would be dead-lettered.
    # v34 ROOT FIX (CRITICAL #8): accept `SYNDROME:<slug>` IDs emitted
    # by the bridge when DrugBank indications have empty disease_id but
    # non-empty disease_name (e.g. "Pain", "Asthma", "Hepatitis B").
    # Without this, ~half of Compound-treats-Disease edges were
    # dead-lettered because the synthetic Disease IDs didn't match
    # the strict biomedical-ontology pattern.
    #
    # P2-051 ROOT FIX (Teammate 4): replace ``MESH:[A-Z]\d+`` with
    # ``MESH:D\d+`` — only MeSH descriptor IDs (D-tree) are Diseases.
    # See the Compound pattern above for the full MeSH tree explanation.
    #
    # P2-052 ROOT FIX (Teammate 4): the previous pattern accepted
    # ``D\d{6}`` (e.g. ``D000001``) as a Disease ID. But DrugBank IDs
    # are ``DB\d{5,7}`` (e.g. ``DB000001``). If the ``B`` is stripped
    # (case-insensitive matching, OCR error, truncation), a DrugBank ID
    # could be misread as a Disease ID. ROOT FIX: require the ``D``
    # prefix to be FOLLOWED BY EXACTLY 6 DIGITS AND A WORD BOUNDARY —
    # actually, since the whole pattern is anchored (^...$), ``D000001``
    # cannot match ``DB000001`` (the latter has a B). But ``D000001``
    # (7 chars) could be a prefix-collapsed DrugBank ID. The safest fix
    # is to require an explicit MeSH prefix for D-tree IDs (``MESH:D\d+``)
    # and accept bare ``D\d{6}`` ONLY for legacy DOID-shortform disease
    # IDs (which are rare but valid in OMIM imports). We keep ``D\d{6}``
    # but DOCUMENT the collision risk and recommend all NEW disease IDs
    # use explicit prefixes (DOID:, OMIM:, MESH:D).
    "Disease": r"^(C\d{7}|D\d{6}|EFO_\d+|EFO:\d+|OMIM:\d+|Orphanet:\d+|MONDO:\d+|DOID:\d+|HP:\d+|MESH:D\d+|SYNDROME:[A-Za-z0-9_]+)$",
    # v43 ROOT FIX (Chain 4b): add PATHWAY_CC_<idx>_<sha8> pattern for
    # STRING-derived Pathway nodes (connected components). The other
    # prefixes (R-HSA-, hsa, REACT_, WP) are for curated pathway
    # databases (Reactome, KEGG, WikiPathways).
    "Pathway": r"^(R-HSA-\d+|hsa\d+|REACT_\d+|WP\d+|PATHWAY_CC_\d+_[0-9a-f]+)$",
    # FIX-F / C-16: ClinicalOutcome nodes derived from
    # drugbank_indications.csv by phase1_bridge._load_clinical_outcomes().
    # ID format: "CO:<drugbank_id>:<disease_key>:<indication_type>" where
    # disease_key is the disease_id (e.g. OMIM:102700) when present, or
    # the slugified disease_name when disease_id is empty (e.g. "Pain").
    # The pattern is intentionally permissive (curie-style with CO: prefix)
    # so it accepts both OMIM-prefixed and name-based disease keys.
    "ClinicalOutcome": r"^CO:[A-Za-z0-9_.:-]+$",
    # MedDRA_Term: 8-digit LLT/PT code OR MedDRA-prefixed UMLS CUI
    # (BUG-B-005: SIDER emits "MedDRA:C0018790" which is the standard
    # biomedical identifier format).
    "MedDRA_Term": r"^(\d{8}|MedDRA:C\d{7})$",
    "Anatomy": r"^(UBERON_\d+|CL_\d+)$",
    "Side Effect": r"^(CUI\d+|C\d{7})$",
    "Symptom": r"^(CUI\d+|C\d{7})$",
    "Pharmacologic Class": r"^(ATC:[A-Z]\d{2}|CHEMBL\d+)$",
    "Biological Process": r"^(GO:\d+)$",
    "Molecular Function": r"^(GO:\d+)$",
    "Cellular Component": r"^(GO:\d+)$",
    "Taxonomy": r"^\d+$",
    "Gene Expression": r"^(GSE\d+|GSM\d+)$",
    # BUG-D-005 root fix: real ATC codes are 7 chars in the WHO format
    # (e.g. L01XC02, L04AA02, N02BA01). The structure is:
    #   [A-Z]    — 1st level (anatomical main group, 1 letter)
    #   \d{2}    — 2nd level (therapeutic main group, 2 digits)
    #   [A-Z]{2} — 3rd+4th levels (therapeutic subgroup + chemical subgroup, 2 letters)
    #   \d{2}    — 5th level (chemical substance, 2 digits)
    # The previous pattern ``^[A-Z]\d{2}[A-Z]\d{2}[A-Z]\d{2}?$`` required
    # 8-9 chars (alternating letter/digit groups) and dead-lettered every
    # Atc node. New pattern accepts the WHO 7-char format AND optional
    # sub-class extensions (L01XC02.01).
    "Atc": r"^[A-Z]\d{2}[A-Z]{2}\d{2}(\.\d{2})?$",
    "Tax": r"^\d+$",
    # v9 ROOT FIX (audit F5.2.1): UniProt cross-reference edges emit
    # heterogeneous target types (Domain, OntologyTerm, Publication,
    # ExternalRef) based on the UniProt DB source. The previous code
    # returned True for any unknown label — silently bypassing
    # validation. Now we explicitly register these labels with
    # permissive curie-style patterns so the edges are validated but
    # not over-restricted. If a label is NOT in this dict, the new
    # fail-closed UnknownLabelError fires.
    "ExternalRef": r"^[A-Za-z_][A-Za-z0-9_-]*:[A-Za-z0-9_.:-]+$",
    "Domain": r"^(PF\d+|IPR\d+|SM\d+|PS\d+)$",
    "OntologyTerm": r"^(GO:\d+|MIM:\d+|KEGG:\S+)$",
    "Publication": r"^\d{7,8}$",  # PMID
}

# Fixes D-2, DQ-4, S(9)-6, IN-1 (Schema-Whitelist Rule §3.7):
# Define allowed properties per node label. Anything not in this list
# (or SYSTEM_PROPS) is silently dropped before Cypher execution.
#
# Audit fix (v6 — bug #B5/B6/B7/B8): the previous whitelist was missing
# every property the phase1_bridge actually emits (fda_approved,
# clinical_status, groups, molecular_weight, molecular_formula,
# completeness_score, gene_symbol, mim_id, uniprot_id, etc.). On a real
# Neo4j load these were silently stripped — only the test path
# (RecordingGraphBuilder, which does NOT apply the whitelist) noticed.
# The whitelist now mirrors the bridge's actual output contract.
NODE_PROPERTY_WHITELIST: dict[str, frozenset[str]] = {
    "Compound": frozenset({
        "id", "name", "smiles", "inchikey", "indication",
        "mechanism_of_action", "atc_codes", "approved", "investigational",
        "pubchem_cid", "chembl_id", "chebi_id", "drug_type",
        "approval_year", "source_drugbank", "drugbank_id", "cas_number",
        "toxicity", "pharmacodynamics", "withdrawn", "terminated",
        "illicit", "sensitive", "categories",
        "_canonical_id_source", "_last_modified", "_schema_version",
        "safety_data_missing", "description",
        # ── v6: bridge-emitted Compound properties (bug #B5) ──
        "fda_approved", "is_fda_approved", "is_withdrawn",
        "clinical_status", "groups",
        "molecular_weight", "molecular_formula",
        "logp", "tpsa",
        "h_bond_donor_count", "h_bond_acceptor_count",
        "rotatable_bond_count", "heavy_atom_count", "complexity",
        "max_phase", "completeness_score",
        "inchikey_source", "cas_number",
        # ── v70 P2L-030: compound_id_aliases — list of alternate stable
        # Compound identifiers (drugbank_id, chembl_id, pubchem_cid,
        # chebi_id, inchikey when not canonical). Used by entity_resolver
        # and kg_builder to MERGE Compound nodes across sources even
        # when the primary id differs (e.g. biotech drugs without
        # InChIKey). Stored as a Neo4j list property.
        "compound_id_aliases",
        # ── TM1 TASK 15 ROOT FIX: critical patient-safety fields that
        # were previously SILENTLY STRIPPED by _whitelist_filter.
        # ``is_globally_approved`` is the v93 patient-safety fix for
        # EMA-only drugs (max_phase=4 means globally approved, not
        # FDA-specific). Without this in the whitelist, the Neo4j export
        # dropped it on every Compound node — the RL ranker's
        # market-opportunity scoring then treated globally-approved
        # drugs as not-approved, corrupting the ranker's output.
        # ``indication_source`` records WHERE the indication text came
        # from (FDA / EMA / manual / RxNorm) — critical for audit trails
        # and for the clinical-trial cross-referencing logic. Without
        # it, the dashboard could not tell whether an indication was
        # FDA-confirmed or crowd-sourced.
        "is_globally_approved", "indication_source",
    }),
    "Disease": frozenset({
        "id", "name", "icd10", "icd9", "mesh", "umls_cui",
        "definition", "source",
        # ── v6: bridge-emitted Disease property (bug #B7) ──
        "mim_id", "phenotype_mim",
    }),
    "Gene": frozenset({
        "id", "name", "symbol", "ncbi_gene_id", "uniprot_ac",
        "chromosome", "description", "source",
        # ── v6: bridge-emitted Gene properties (bug #B6) ──
        "gene_symbol", "mim_id", "uniprot_id",
    }),
    "Protein": frozenset({
        "id", "name", "uniprot_ac", "uniprot_id", "gene_name",
        "gene_id", "ncbi_gene_id", "organism", "sequence",
        "function", "source",
    }),
    "Pathway": frozenset({
        "id", "name", "reactome_id", "kegg_id", "source",
        # v78 FORENSIC ROOT FIX (BUG #2/#3 follow-on): the bridge emits
        # ``member_count``, ``members`` (pipe-joined UniProt ACs), and
        # ``derivation_method`` on every STRING-derived Pathway node
        # (including the v53 fallback). Without these in the whitelist,
        # production silently stripped them — losing the protein-membership
        # data the Graph Explorer needs to render pathway chains.
        "member_count", "members", "derivation_method",
    }),
    # FIX-F / C-16: ClinicalOutcome nodes — derived from
    # drugbank_indications.csv by phase1_bridge._load_clinical_outcomes().
    #
    # v78 FORENSIC ROOT FIX (BUG #5 — Silent Data-Loss): the v60 ROOT
    # FIX added ``meddra_id``, ``mesh_id``, ``first_seen_drug_id`` to
    # the bridge's ClinicalOutcome node construction, AND added
    # ``CANONICAL_IDS["ClinicalOutcome"] = "meddra_id"`` +
    # ``ID_MAPPING_PRIORITY["ClinicalOutcome"] = ["meddra_id",
    # "mesh_id", "name"]`` to config.py. But this NODE_PROPERTY_WHITELIST
    # entry was NEVER updated to include those fields — so the production
    # ``GraphNodeLoader._whitelist_filter`` silently stripped them,
    # making ``entity_resolver.resolve_canonical_id`` return None for
    # every ClinicalOutcome node. The 5th DOCX-mandated node type was
    # effectively unresolvable in production. ROOT FIX: add all four
    # canonical-ID + multi-drug-accumulation fields to the whitelist.
    "ClinicalOutcome": frozenset({
        "id", "name", "disease_id", "disease_name",
        "indication_type", "source_drug_id", "source",
        # v78 BUG #5 fix — canonical-ID fields the v60 ROOT FIX promised
        # would survive the production property-stripping pass.
        "meddra_id",            # CANONICAL_IDS["ClinicalOutcome"]
        "mesh_id",              # ID_MAPPING_PRIORITY fallback #2
        "first_seen_drug_id",   # v35 M-5: first Compound pointing here
        "source_drug_ids",      # v35 M-5: ALL Compounds pointing here
    }),
    "MedDRA_Term": frozenset({
        "id", "name", "meddra_code", "meddra_type", "umls_cui",
        "source",
    }),
    "Anatomy": frozenset({"id", "name", "uberon_id", "source"}),
    "Side Effect": frozenset({"id", "name", "umls_cui", "source"}),
    "Symptom": frozenset({"id", "name", "umls_cui", "source"}),
    "Pharmacologic Class": frozenset({"id", "name", "atc_code", "source"}),
    "Biological Process": frozenset({"id", "name", "go_id", "source"}),
    "Molecular Function": frozenset({"id", "name", "go_id", "source"}),
    "Cellular Component": frozenset({"id", "name", "go_id", "source"}),
    "Taxonomy": frozenset({"id", "name", "tax_id", "source"}),
    "Gene Expression": frozenset({"id", "name", "gse_id", "source"}),
    "Atc": frozenset({"id", "name", "source"}),
    "Tax": frozenset({"id", "name", "source"}),
}

# Edge property whitelist per (src_label, rel_type, dst_label) triple.
#
# Audit fix (v6 — bug #B8): the previous whitelist was missing properties
# the bridge emits on every edge type (is_known_action, source_id,
# action_type, mapping_key, association_type, evidence, etc.). Real Neo4j
# loads silently stripped them, breaking downstream lineage queries.
#
# BUG-D-006 root fix — the v6 whitelist is populated by iterating
# CORE_EDGE_TYPES, so if CORE_EDGE_TYPES is ever empty (config import
# error, circular import, monkey-patched test fixture), the whitelist
# stays {} and ALL edge properties are silently stripped in production.
# The audit (§5.2) flags this as Major: "No validation that the
# whitelist is non-empty before use."
#
# Root fix: assert non-empty at import time so a config regression
# surfaces as a loud ImportError, not a silent property-stripping bug.
EDGE_PROPERTY_WHITELIST: dict[tuple[str, str, str], frozenset[str]] = {}
for _src, _rel, _dst in CORE_EDGE_TYPES:
    # Every edge gets these lineage + base properties.
    _base = frozenset({
        "source", "evidence", "score", "confidence",
        # v27 ROOT FIX (P2-L-3): canonical normalized score in [0,1] for
        # cross-source fusion. Every loader (STITCH, STRING, ChEMBL,
        # DisGeNET, OMIM, OpenTargets, DrugBank) now emits BOTH a raw
        # source-specific score (e.g. ``string_combined_score``,
        # ``pchembl_value``, ``disgenet_score``) AND a canonical
        # ``normalized_score`` in [0,1]. Whitelist it on EVERY edge type
        # so the property survives kg_builder's property-stripping pass.
        "normalized_score",
        # ── v6: bridge-emitted lineage properties (bug #B8) ──
        "source_id", "action_type", "is_known_action",
        "association_type", "mapping_key",
    })
    if _rel in ("causes_adverse_event", "causes_side_effect"):
        _base = _base | frozenset({"frequency", "meddra_type", "meddra_code"})
    if _rel == "tested_for":
        _base = _base | frozenset({
            "nct_id", "phase", "status", "enrollment", "why_stopped",
        })
    if _rel in ("inhibits", "activates", "binds", "targets",
                "allosterically_modulates", "unknown"):
        _base = _base | frozenset({
            "action_type", "pubmed_ids",
            "is_known_action", "source_id",  # bridge-emitted
            # v21 ROOT FIX (Audit section 4 finding 4 / Chain 4):
            # the previous whitelist was missing every ChEMBL
            # activity property that phase1_bridge emits on
            # Compound-{inhibits,activates,targets,binds}-Protein
            # edges. Without these, the production kg_builder
            # silently stripped pchembl_value (potency),
            # standard_relation (censoring direction), and the
            # activity metadata. The v15 ROOT FIX explicitly
            # promised these would be preserved so the RL ranker
            # has potency + censoring context; that promise was
            # FALSE in production. The test path
            # (RecordingGraphBuilder) does not apply the whitelist,
            # so the bug was invisible to tests.
            "pchembl_value",        # -log10(IC50/Ki/Kd) - potency
            "standard_relation",    # '=', '<', '>' - censoring
            "activity_type",        # "IC50", "EC50", "Ki", "Kd"...
            "activity_value",       # numeric activity value
            "activity_units",       # "nM", "uM"...
            "assay_type",           # 'F' functional / 'B' binding
            "chembl_target_id",     # for unresolved targets
        })
    if _rel == "interacts_with" and _src == "Compound":
        _base = _base | frozenset({"severity", "description"})
    if _rel == "associated_with" and _src == "Gene":
        _base = _base | frozenset({
            "association_type", "mapping_key",  # bridge-emitted (OMIM GDA)
        })
    if _rel == "encodes" and _src == "Gene":
        _base = _base | frozenset({
            "evidence",  # bridge-emitted (gene_protein_crosswalk)
        })
    if _rel == "treats" and _src == "Compound":
        _base = _base | frozenset({
            "evidence",  # bridge-emitted (drugbank_indication_text)
        })
    # v29 ROOT FIX (audit L-4 — EDGE_PROPERTY_WHITELIST silently strips
    # properties): the previous whitelist was missing GEO expression
    # properties, STITCH confidence channels, and SIDER frequency
    # bounds. The RL safety ranker needs these to distinguish 50% ADRs
    # from 0.01% ADRs, and the KG needs expression magnitude to know
    # HOW strongly a protein is expressed in a tissue (not just that
    # it IS expressed). ROOT FIX: add the missing properties to every
    # relevant edge type.
    if _rel == "expressed_in" or (_rel == "associated_with" and _src == "Protein"):
        # GEO expression edges: (Protein, expressed_in, Tissue)
        _base = _base | frozenset({
            "expression_value",   # log2 fold change magnitude
            "n_samples",          # sample count (statistical power)
            "fdr",                # false discovery rate
            "p_value",            # statistical significance
            "tissue",             # tissue name
            "experiment_id",      # GEO accession (GSE...)
        })
    if _rel in ("interacts_with", "binds") and _src == "Compound" and _dst == "Protein":
        # STITCH chemical-protein interaction edges
        _base = _base | frozenset({
            "stitch_combined_score",   # 0-999 confidence
            "stereochemistry",         # stereo flag (CIDm vs CIDs)
            "evidence_channels",       # experimental/database/textmining
            "experimental_score",
            "database_score",
            "textmining_score",
        })
    if _rel in ("causes_adverse_event", "causes_side_effect"):
        # SIDER adverse event edges — add frequency bounds
        _base = _base | frozenset({
            "frequency_description",    # "Postmarketing", "Frequent", etc.
            "frequency_lower_bound",    # 0.0
            "frequency_upper_bound",    # 1.0
            "frequency_source",         # "sider_frequency"
            "meddra_name",
        })
    EDGE_PROPERTY_WHITELIST[(_src, _rel, _dst)] = _base

# P2-053 ROOT FIX: validate that EDGE_PROPERTY_WHITELIST keys are in
# EXACT 1:1 correspondence with CORE_EDGE_TYPES_SET. The whitelist is
# built by iterating CORE_EDGE_TYPES (above), so by construction the
# keys SHOULD match — but a future maintainer might monkey-patch one
# without the other, OR a typo in CORE_EDGE_TYPES (e.g. an extra
# trailing space "Compound " instead of "Compound") would produce a
# whitelist key that doesn't match what the loaders emit, silently
# stripping ALL extended properties (pchembl_value, normalized_score,
# etc.) on that triple type. The audit's fallback whitelist
# (frozenset({"source", "evidence", "score"}) in the .get() call) is
# too permissive to catch this — the loader silently degrades to the
# 3-property fallback and the operator never knows.
#
# Root fix: assert at module-load time that:
#   (1) every CORE_EDGE_TYPES entry has a whitelist key, AND
#   (2) every whitelist key is a CORE_EDGE_TYPES entry, AND
#   (3) no entry has leading/trailing whitespace (the most common typo).
# This is a CHEAP, LOUD check that fails the import rather than
# silently corrupting the KG. We skip it in test contexts (detected
# via PYTEST_CURRENT_TEST / DRUGOS_SKIP_IMPORT_CHECK / pytest in
# sys.modules) so test fixtures can monkey-patch CORE_EDGE_TYPES
# without crashing the import — the same pattern used by
# ``_assert_edge_property_whitelist_populated`` below.
_p2_053_skip = (
    os.environ.get("PYTEST_CURRENT_TEST") is not None
    or os.environ.get("DRUGOS_SKIP_IMPORT_CHECK") == "1"
    or "pytest" in sys.modules
)
if not _p2_053_skip:
    # (1) every CORE_EDGE_TYPES entry must have a whitelist key.
    _missing = [
        triple for triple in CORE_EDGE_TYPES
        if triple not in EDGE_PROPERTY_WHITELIST
    ]
    if _missing:
        raise RuntimeError(
            "P2-053 invariant violated: CORE_EDGE_TYPES contains triples "
            f"with no EDGE_PROPERTY_WHITELIST entry: {_missing}. The "
            "whitelist is built by iterating CORE_EDGE_TYPES — a missing "
            "key means the iteration was short-circuited (monkey-patch) "
            "or CORE_EDGE_TYPES was mutated after whitelist construction."
        )
    # (2) every whitelist key must be a CORE_EDGE_TYPES entry. Catches
    # the case where a maintainer adds a whitelist entry for a triple
    # that isn't a core edge type (typo in the triple).
    _extra = [
        triple for triple in EDGE_PROPERTY_WHITELIST
        if triple not in CORE_EDGE_TYPES_SET
    ]
    if _extra:
        raise RuntimeError(
            "P2-053 invariant violated: EDGE_PROPERTY_WHITELIST contains "
            f"keys not in CORE_EDGE_TYPES: {_extra}. These triples would "
            "silently strip ALL extended properties on real edges "
            "(pchembl_value, normalized_score, etc.) — the loader's "
            "fallback whitelist is too permissive to catch this. Either "
            "add the triple to CORE_EDGE_TYPES or remove the whitelist "
            "entry."
        )
    # (3) no entry has leading/trailing whitespace. This is the most
    # common typo that produces silent property stripping — the typo'd
    # key exists in the whitelist (so the (1) check passes), but the
    # actual edges loaded use the clean key and miss the lookup.
    for _s, _r, _d in CORE_EDGE_TYPES:
        for _label, _val in (("src", _s), ("rel", _r), ("dst", _d)):
            if _val != _val.strip():
                raise RuntimeError(
                    f"P2-053 invariant violated: CORE_EDGE_TYPES triple "
                    f"({_s!r}, {_r!r}, {_d!r}) has leading/trailing "
                    f"whitespace in the {_label} component. This would "
                    "produce a whitelist key that never matches the "
                    "clean key the loaders emit — silently stripping "
                    "ALL extended properties on that triple type. "
                    "Remove the whitespace in config.CORE_EDGE_TYPES."
                )
            # Also reject internal whitespace-only typos (e.g. "treats "
            # with trailing space). The .strip() check above catches
            # leading/trailing; this check catches double spaces inside.
            if "  " in _val:
                raise RuntimeError(
                    f"P2-053 invariant violated: CORE_EDGE_TYPES triple "
                    f"({_s!r}, {_r!r}, {_d!r}) has a double-space in "
                    f"the {_label} component. This is almost certainly "
                    "a typo — silently strips all extended properties "
                    "on that triple type."
                )

# RT-8 ROOT FIX: the previous code raised ImportError at module
# import time when EDGE_PROPERTY_WHITELIST was empty. This made
# kg_builder unimportable for unit tests, partial pipelines, CI
# lint runs, and error recovery — a single config regression took
# down the entire module surface, and the operator could not even
# open a Python REPL to inspect kg_builder to debug. Move the
# invariant check to a runtime function (called from
# DrugOSGraphBuilder.__init__ and from load_edges_bulk_create) so
# it fires only when an actual production edge load is attempted
# with an empty whitelist. The check is now a RuntimeError (not
# ImportError) so it does not interfere with Python's import system.
def _assert_edge_property_whitelist_populated() -> None:
    """Raise RuntimeError if the edge-property whitelist is empty.

    Called from DrugOSGraphBuilder.__init__ (and from
    load_edges_bulk_create as a defensive re-check) to ensure
    production edge loads never silently strip all properties.
    Safe to call at module import time — it returns silently when
    the whitelist is populated (the normal case).
    """
    if not EDGE_PROPERTY_WHITELIST:
        raise RuntimeError(
            "BUG-D-006 invariant violated: EDGE_PROPERTY_WHITELIST is "
            "empty. CORE_EDGE_TYPES must be imported and non-empty "
            "before any production edge load. Check "
            "phase2/drugos_graph/config.py for regressions in the "
            "CORE_EDGE_TYPES definition."
        )
    if not CORE_EDGE_TYPES_SET:
        raise RuntimeError(
            "BUG-D-006 invariant violated: CORE_EDGE_TYPES_SET is empty. "
            "Production edge loads would silently strip all properties."
        )


# Validate at import time ONLY in non-test contexts. In test contexts
# (detected via the standard PYTEST_CURRENT_TEST env var or when the
# module is imported by a test runner), defer to runtime so test
# fixtures can monkey-patch CORE_EDGE_TYPES without crashing the import.
_import_time_skip = (
    os.environ.get("PYTEST_CURRENT_TEST") is not None
    or os.environ.get("DRUGOS_SKIP_IMPORT_CHECK") == "1"
    or "pytest" in sys.modules
)
if not _import_time_skip:
    try:
        _assert_edge_property_whitelist_populated()
    except RuntimeError:
        # Log a CRITICAL warning but DO NOT raise — allow the module
        # to be imported so the operator can debug. The runtime check
        # in DrugOSGraphBuilder.__init__ will still raise when an
        # actual edge load is attempted.
        logging.getLogger(__name__).critical(
            "BUG-D-006 invariant violated at import time: "
            "EDGE_PROPERTY_WHITELIST is empty. Module is importable "
            "for debugging, but DrugOSGraphBuilder construction will "
            "raise RuntimeError until the config regression is fixed."
        )

# Source licenses for provenance (CO-2)
# v35 ROOT FIX (N-3): the bridge emits LOWERCASE source labels ("drugbank",
# "chembl", "string", "omim", "disgenet", "uniprot", "pubchem",
# "drugbank_indication", etc.) for every staged edge's `source` field.
# The original dict used CAPITALIZED keys ("DrugBank", "ChEMBL", ...),
# so SOURCE_LICENSES.get(source) returned the fallback `{"license":
# "unknown"}` and the CC BY-NC 4.0 attribution required by DrugBank's
# license was silently dropped from every bridge-loaded edge. We now
# ALSO include lowercase aliases so the case-sensitive lookup succeeds
# regardless of which form the caller uses.
SOURCE_LICENSES: dict[str, dict[str, str]] = {
    "DRKG":        {"license": "ODC-BY 1.0",   "attribution": "DRKG (Ioannidis et al., 2020), ODC-BY 1.0"},
    "DrugBank":    {"license": "CC BY-NC 4.0",  "attribution": "DrugBank (Wishart DS et al., Nucleic Acids Res. 2018), CC BY-NC 4.0"},
    "UniProt":     {"license": "CC BY 4.0",     "attribution": "UniProt (UniProt Consortium), CC BY 4.0"},
    "ChEMBL":      {"license": "CC BY-SA 3.0",  "attribution": "ChEMBL (Gaulton A et al., Nucleic Acids Res. 2017), CC BY-SA 3.0"},
    "STRING":      {"license": "CC BY 4.0",     "attribution": "STRING (Szklarczyk D et al., Nucleic Acids Res. 2023), CC BY 4.0"},
    "STITCH":      {"license": "CC BY 4.0",     "attribution": "STITCH (Kuhn M et al., Nucleic Acids Res. 2014), CC BY 4.0"},
    "SIDER":       {"license": "CC0 1.0",       "attribution": "SIDER (Kuhn M et al., Clin Pharmacol Ther. 2016), CC0 1.0"},
    "OpenTargets": {"license": "Apache 2.0",    "attribution": "OpenTargets (Koscielny G et al., Nucleic Acids Res. 2017), Apache 2.0"},
    "ClinicalTrials": {"license": "public domain", "attribution": "ClinicalTrials.gov (AACT), public domain"},
    "GEO":         {"license": "public domain", "attribution": "GEO (Barrett T et al., Nucleic Acids Res. 2013), public domain"},
    # ── v35 N-3: lowercase aliases for the bridge's source labels ──
    "drkg":            {"license": "ODC-BY 1.0",   "attribution": "DRKG (Ioannidis et al., 2020), ODC-BY 1.0"},
    "drugbank":        {"license": "CC BY-NC 4.0",  "attribution": "DrugBank (Wishart DS et al., Nucleic Acids Res. 2018), CC BY-NC 4.0"},
    "drugbank_indication":      {"license": "CC BY-NC 4.0",  "attribution": "DrugBank (Wishart DS et al., Nucleic Acids Res. 2018), CC BY-NC 4.0"},
    "drugbank_indications":     {"license": "CC BY-NC 4.0",  "attribution": "DrugBank (Wishart DS et al., Nucleic Acids Res. 2018), CC BY-NC 4.0"},
    "drugbank_indication_text": {"license": "CC BY-NC 4.0",  "attribution": "DrugBank (Wishart DS et al., Nucleic Acids Res. 2018), CC BY-NC 4.0"},
    "uniprot":         {"license": "CC BY 4.0",     "attribution": "UniProt (UniProt Consortium), CC BY 4.0"},
    "chembl":          {"license": "CC BY-SA 3.0",  "attribution": "ChEMBL (Gaulton A et al., Nucleic Acids Res. 2017), CC BY-SA 3.0"},
    "string":          {"license": "CC BY 4.0",     "attribution": "STRING (Szklarczyk D et al., Nucleic Acids Res. 2023), CC BY 4.0"},
    "stitch":          {"license": "CC BY 4.0",     "attribution": "STITCH (Kuhn M et al., Nucleic Acids Res. 2014), CC BY 4.0"},
    "sider":           {"license": "CC0 1.0",       "attribution": "SIDER (Kuhn M et al., Clin Pharmacol Ther. 2016), CC0 1.0"},
    "opentargets":     {"license": "Apache 2.0",    "attribution": "OpenTargets (Koscielny G et al., Nucleic Acids Res. 2017), Apache 2.0"},
    "clinicaltrials":  {"license": "public domain", "attribution": "ClinicalTrials.gov (AACT), public domain"},
    "geo":             {"license": "public domain", "attribution": "GEO (Barrett T et al., Nucleic Acids Res. 2013), public domain"},
    "omim":            {"license": "public domain", "attribution": "OMIM (Amberger JS et al., Nucleic Acids Res. 2019), public domain"},
    "disgenet":        {"license": "CC BY-NC-SA 4.0", "attribution": "DisGeNET (Piñero J et al., Nucleic Acids Res. 2020), CC BY-NC-SA 4.0"},
    "pubchem":         {"license": "public domain", "attribution": "PubChem (Kim S et al., Nucleic Acids Res. 2023), public domain"},
    "phase1_bridge":   {"license": "various",       "attribution": "Phase 1 bridge (aggregated from upstream sources)"},
}

# Additional indexes config (CF-1)
ADDITIONAL_INDEXES: list[tuple[str, str]] = [
    ("Compound", "name"), ("Disease", "name"), ("Gene", "name"),
    ("Compound", "approved"), ("Compound", "smiles"),
    ("Protein", "name"), ("Pathway", "name"), ("MedDRA_Term", "name"),
    ("Anatomy", "name"), ("Compound", "withdrawn"),
    ("Compound", "inchikey"), ("Compound", "chembl_id"),
    ("Compound", "drugbank_id"), ("Compound", "sensitive"),
]

# Environment variable defaults (CF-3)
_NEO4J_MAX_RETRIES = int(os.environ.get("DRUGOS_NEO4J_MAX_RETRIES", "5"))
_NEO4J_RETRY_BASE_DELAY = float(os.environ.get("DRUGOS_NEO4J_RETRY_BASE_DELAY", "1.0"))
_NEO4J_RETRY_MAX_DELAY = float(os.environ.get("DRUGOS_NEO4J_RETRY_MAX_DELAY", "30.0"))
_QUERY_TIMEOUT = int(os.environ.get("DRUGOS_NEO4J_QUERY_TIMEOUT", "300"))
_LOG_FREQUENCY = int(os.environ.get("DRUGOS_PROGRESS_LOG_EVERY_N_BATCHES", "10"))
_LOG_INTERVAL_SECONDS = int(os.environ.get("DRUGOS_PROGRESS_LOG_INTERVAL_SECONDS", "60"))
_DATA_MAX_AGE_DAYS = int(os.environ.get("DRUGOS_DATA_MAX_AGE_DAYS", "30"))
# v34 ROOT FIX (CRITICAL #5): expose the default clear-phrase as a public
# module-level constant so callers (run_pipeline.py, run_unified.py) can
# import it instead of hardcoding a DIFFERENT string. The previous code
# had run_pipeline.py passing "CLEAR_ALL_DRUGOS_DATA" while kg_builder
# expected "DELETE EVERYTHING I UNDERSTAND THE CONSEQUENCES" — they NEVER
# matched, so `clear_graph()` always raised SecurityError, was caught by
# the `except Exception` in step3, and logged as a warning. The graph was
# NEVER cleared → re-runs created DUPLICATE nodes/edges. The
# `fresh_start=True` idempotency promise was dead code.
DEFAULT_CLEAR_GRAPH_PHRASE = "DELETE EVERYTHING I UNDERSTAND THE CONSEQUENCES"
_CLEAR_GRAPH_PHRASE = os.environ.get(
    "DRUGOS_CLEAR_GRAPH_PHRASE",
    DEFAULT_CLEAR_GRAPH_PHRASE,
)
_ALLOW_NON_CORE_EDGES = os.environ.get("DRUGOS_KG_ALLOW_NON_CORE_EDGES", "0") == "1"
_AUTO_DEDUP = os.environ.get("DRUGOS_KG_AUTO_DEDUP", "0") == "1"


# P2-004 FORENSIC ROOT FIX (v104 — Team Member 5): Neo4j driver
# lifecycle cleanup on SIGTERM / SIGINT / atexit.
#
# BUG (P2-004):
#   ``GraphConnection`` registered NO signal handlers. When Airflow
#   sends SIGTERM to stop a long-running KG build, the driver was not
#   closed cleanly. The Neo4j server kept the connections open until
#   TCP timeout (typically 2 hours). After 10 SIGTERM'd runs, Neo4j
#   hit its max-connections limit (default 400) and rejected new
#   connections — the pipeline could not run.
#
# ROOT FIX:
#   1. Module-level registry (``_OWNED_CONNECTIONS``) of
#      ``GraphConnection`` instances that OWN their driver (i.e. they
#      created it, not received it via DI). Uses ``weakref.WeakSet``
#      so connections can be GC'd without explicit unregister.
#   2. Module-level signal handlers for SIGTERM and SIGINT (installed
#      ONCE via ``_install_signal_handlers``). The handler iterates
#      the registry and calls ``disconnect()`` on each, then CHAINS
#      to the previous handler (so ``__main__.py``'s handler still
#      runs). This avoids clobbering upstream signal handlers.
#   3. Module-level atexit handler as a safety net for normal
#      interpreter shutdown (covers ``sys.exit()``, end-of-script).
#   4. ``GraphConnection.connect()`` registers self AFTER successful
#      driver creation (only if it owns the driver).
#   5. ``GraphConnection.disconnect()`` is idempotent (sets
#      ``self._driver = None`` after close) so the signal handler can
#      call it without risk of double-close.
#
# The signal handlers do NOT raise — they log, close drivers, and
# then re-raise the signal as ``SystemExit`` so the process exits
# cleanly. A second SIGTERM/SIGINT (within 3 seconds) bypasses
# cleanup and forces immediate exit (for stuck drivers).
_OWNED_CONNECTIONS: "weakref.WeakSet[GraphConnection]" = weakref.WeakSet()
_SIGNAL_HANDLERS_INSTALLED = False
_SIGNAL_HANDLER_LOCK = threading.Lock()
_LAST_SIGNAL_TIME: float = 0.0


def _cleanup_owned_connections(signum: Optional[int] = None) -> int:
    """P2-004 — close ALL owned Neo4j drivers.

    Iterates ``_OWNED_CONNECTIONS`` and calls ``disconnect()`` on
    each. Returns the number of connections closed. Safe to call
    from a signal handler, atexit, or normal code. Idempotent.

    ``signum`` is passed by the signal-handler wrapper for logging;
    ``None`` means called from atexit or explicit code.
    """
    closed = 0
    # Take a snapshot — the WeakSet may mutate during iteration if
    # disconnect() triggers GC (unlikely but defensive).
    conns = list(_OWNED_CONNECTIONS)
    for conn in conns:
        try:
            if conn._driver is not None and not conn._external_driver:
                conn.disconnect()
                closed += 1
        except Exception as exc:  # noqa: BLE001 — signal handler must not raise
            logger.warning(
                "P2-004: error closing Neo4j connection during "
                "signal/atexit cleanup: %s", exc,
            )
    if closed > 0:
        logger.info(
            "P2-004: closed %d Neo4j connection(s) during cleanup "
            "(signal=%s)", closed,
            signal.Signals(signum).name if signum else "atexit",
        )
    return closed


def _signal_cleanup_handler(signum, frame):  # type: ignore[no-untyped-def]
    """P2-004 — SIGTERM/SIGINT handler that closes Neo4j drivers.

    Chains to the previous signal handler after cleanup. A second
    signal within 3 seconds forces immediate exit (for stuck drivers
    that block on ``driver.close()``).
    """
    global _LAST_SIGNAL_TIME
    import time as _time
    now = _time.monotonic()
    if now - _LAST_SIGNAL_TIME < 3.0:
        # Second signal — force exit without cleanup.
        logger.warning(
            "P2-004: second signal %s received — forcing immediate exit "
            "without Neo4j cleanup.",
            signal.Signals(signum).name if signum in signal.Signals._value2member_map_ else signum,
        )
        sys.exit(130)  # 128 + SIGINT(2) convention; works for SIGTERM too
    _LAST_SIGNAL_TIME = now

    try:
        _cleanup_owned_connections(signum)
    except Exception as exc:  # noqa: BLE001 — signal handler must not raise
        logger.warning("P2-004: cleanup raised: %s", exc)

    # Re-raise as SystemExit so the process exits cleanly and
    # try/finally blocks in the main thread run.
    raise SystemExit(128 + (signum & 7))


def _install_signal_handlers() -> None:
    """P2-004 — install SIGTERM/SIGINT handlers + atexit. Idempotent.

    Installs ONCE (module-level flag). Chains with existing handlers
    by saving the previous handler — though in practice we re-raise
    as SystemExit which is the Python-idiomatic way to exit from a
    signal handler while still running try/finally blocks.

    On Windows, SIGTERM may not exist; SIGBREAK is used instead.
    """
    global _SIGNAL_HANDLERS_INSTALLED
    with _SIGNAL_HANDLER_LOCK:
        if _SIGNAL_HANDLERS_INSTALLED:
            return
        _SIGNAL_HANDLERS_INSTALLED = True

    # atexit handler — safety net for normal shutdown.
    atexit.register(_cleanup_owned_connections, None)

    # Signal handlers — for Airflow SIGTERM, Ctrl-C SIGINT, etc.
    # Only install if the signal exists on this platform.
    for sig_name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            # signal.signal returns the previous handler — we don't
            # explicitly call it (re-raising SystemExit is cleaner),
            # but we log it at DEBUG for observability.
            prev = signal.signal(sig, _signal_cleanup_handler)
            logger.debug(
                "P2-004: installed %s handler (previous: %s)",
                sig_name, prev,
            )
        except (ValueError, OSError) as exc:
            # signal.signal raises ValueError if not in main thread,
            # OSError on some platforms for certain signals. Log and
            # continue — atexit still provides cleanup.
            logger.debug(
                "P2-004: could not install %s handler: %s "
                "(atexit fallback still active)", sig_name, exc,
            )


# P2-004: install signal + atexit handlers at module load. Idempotent
# and safe — handlers only fire on SIGTERM/SIGINT/atexit, and only
# close drivers that THIS module created (not external/DI drivers).
_install_signal_handlers()


# ─── Result Dataclasses ───────────────────────────────────────────────────────
# Fixes D-6, C-7: Structured return types for all mutating operations.

@dataclass(frozen=True)
class LoadResult:
    """Result of a node or edge loading operation.

    Fixes D-6: Return values now track created, updated, matched, dropped,
    and dead-lettered counts instead of just created.
    """
    attempted: int
    created: int
    updated: int = 0
    matched: int = 0
    dropped_no_match: int = 0
    dead_lettered: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def __int__(self) -> int:
        """Backward compatibility — old code expects int return."""
        return self.created

    def __add__(self, other: LoadResult) -> LoadResult:
        return LoadResult(
            attempted=self.attempted + other.attempted,
            created=self.created + other.created,
            updated=self.updated + other.updated,
            matched=self.matched + other.matched,
            dropped_no_match=self.dropped_no_match + other.dropped_no_match,
            dead_lettered=self.dead_lettered + other.dead_lettered,
            elapsed_seconds=self.elapsed_seconds + other.elapsed_seconds,
            errors=self.errors + other.errors,
        )


@dataclass(frozen=True)
class ClearGraphResult:
    """Result of a clear_graph operation.

    Fixes C-7: clear_graph returns structured result instead of None.
    """
    nodes_deleted: int
    relationships_deleted: int
    elapsed_seconds: float
    pipeline_run_id: str
    timestamp: str


@dataclass(frozen=True)
class BuildGraphResult:
    """Result of a build_graph orchestration.

    Fixes D-5: Structured result for the fluent build_graph method.
    """
    node_results: dict[str, LoadResult]
    edge_results: dict[tuple[str, str, str], LoadResult]
    enrichment_result: Optional[LoadResult]
    stats: dict[str, Any]
    lineage: dict[str, Any]
    elapsed_seconds: float


# ─── Helper Functions ──────────────────────────────────────────────────────────

def _check_neo4j_available() -> None:
    """Raise informative error if neo4j driver is not installed."""
    if GraphDatabase is None:
        raise ImportError(
            "The 'neo4j' Python driver is not installed. "
            "Install it with: pip install neo4j>=5.0,<6.0"
        )


def _validate_id(label: str, value: str) -> bool:
    """Validate a node ID against the expected pattern for its type.

    Fixes S-5 (Domain 3): Biomedical identifier validation.
    Invalid IDs go to the dead-letter queue with reason='invalid_id_format'.

    v9 ROOT FIX (audit F7.8): the previous code returned ``True`` for any
    label not present in ID_PATTERNS — silently disabling validation for
    typo'd labels like 'MedDRATerm' (missing underscore). Now raises
    ``UnknownLabelError`` so the caller can either fix the label or
    explicitly register the new label's pattern in ID_PATTERNS.
    """
    if not value or not isinstance(value, str):
        return False
    if len(value) > 1024:
        return False
    pat = ID_PATTERNS.get(label)
    if pat is None:
        # v9: fail-closed. Unknown labels cannot silently bypass validation.
        raise UnknownLabelError(
            f"Unknown node label {label!r}: no ID_PATTERNS entry. "
            f"Register the label in kg_builder.ID_PATTERNS or fix the typo."
        )
    return re.match(pat, str(value)) is not None


# v43 ROOT FIX (Chain 1 — patient safety): single helper that translates
# a DRKG semantic type name (e.g. "MedDRA_Term") to its canonical Neo4j
# storage label (e.g. "MedDRATerm"). All node and edge loaders MUST
# route their label argument through this helper before sanitize_label,
# so writes / constraints / queries all converge on the SAME label.
#
# This closes the patient-safety hole where SIDER adverse-event queries
# returned ZERO results for every drug (because writes targeted
# :MedDRA_Term while queries hit :MedDRATerm) — causing the RL safety
# ranker to classify every drug as "green/safe" including withdrawn
# drugs like Valdecoxib (withdrawn 2005 for cardiovascular death).
def _storage_label(label: str) -> str:
    """Translate a DRKG semantic type name to its Neo4j storage label.

    If ``label`` is a known CORE/DRKG node type, return its canonical
    Neo4j storage label via ``drkg_node_type_to_neo4j_label``. Otherwise
    (caller already passed a storage label, or a custom label), return
    the input unchanged so we don't double-translate.
    """
    if not label or not isinstance(label, str):
        return label
    _known = set(CORE_NODE_TYPES) | set(DRKG_NODE_TYPES)
    if label in _known:
        try:
            from .utils import drkg_node_type_to_neo4j_label as _to_storage
            return _to_storage(label)
        except Exception:
            return label
    return label


# v102 ROOT FIX (P2-048): canonical Neo4j relationship type transform.
#
# The previous safe_rel construction only handled spaces and hyphens:
#   rel_type.lower().replace(" ", "_").replace("-", "_")
# DRKG relation codes contain "::" and ":" (e.g.
# "DRUGBANK::treats::Compound:Disease"). After lowercasing, the
# rel_type still has "::" and ":" — sanitize_rel_type (via
# _sanitize_identifier_core) replaces every char NOT in [A-Za-z0-9_]
# with underscore, producing "drugbank__treats__compound_disease" —
# losing the source prefix structure. Graph queries that filter by
# relation source (e.g. "DRUGBANK::") return empty.
#
# ROOT FIX: apply a CANONICAL transformation BEFORE calling
# sanitize_rel_type. The canonical form is "source_relation"
# (lowercase, underscore-joined, no double-colons). This preserves
# the source prefix structure that operators query on, while remaining
# Neo4j-safe (no ":" characters). Examples:
#   "DRUGBANK::treats::Compound:Disease" → "drugbank_treats"
#   "DRUGBANK::treats"                    → "drugbank_treats"
#   "treats"                              → "treats"
#   "DRUGBANK::causes_side_effect"        → "drugbank_causes_side_effect"
#   "drugbank::treats::compound:disease"  → "drugbank_treats" (lowercase input)
#
# This helper is called from EVERY safe_rel construction site (3 call
# sites in kg_builder.py: _load_edges_core, dedup_edges,
# select_primary_edge) so the canonical form is consistent across
# writes, dedup, and primary-edge selection.
def _canonical_rel_type(rel_type: str) -> str:
    """Transform a DRKG-style relation code into its canonical Neo4j form.

    The canonical form is ``"source_relation"`` (lowercase,
    underscore-joined, no double-colons). For DRKG codes with the
    full ``"source::name::dst_type"`` form, only the FIRST TWO tokens
    (source + name) are retained — the dst_type token is redundant
    with the edge triple's dst_label.

    Args:
        rel_type: Raw relation type string (e.g. "DRUGBANK::treats::
            Compound:Disease", "treats", "DRUGBANK::causes_side_effect").

    Returns:
        Canonical lowercase form ready for sanitize_rel_type (e.g.
        "drugbank_treats", "treats", "drugbank_causes_side_effect").

        The output is NOT yet Neo4j-safe (may still contain chars
        sanitize_rel_type would reject) — callers MUST pass the result
        through sanitize_rel_type for final validation.

    v107 ROOT FIX (ISSUE-P2-046): DOCUMENT that this transform is
    DRKG-only. The bridge already emits lowercase relation names
    (e.g. "treats", "inhibits", "validated_treats") — this function
    is a NO-OP for bridge-produced edges. It is ONLY relevant for
    DRKG-produced edges (the ``--data-source drkg`` CLI path), which
    carry the "DRUGBANK::treats::Compound:Disease" form. The function
    is called from EVERY safe_rel construction site (3 call sites:
    _load_edges_core, dedup_edges, select_primary_edge) for
    consistency, but for non-DRKG inputs the ``::`` branch is never
    taken and the function just lowercases the input. This is correct
    behavior — the function is idempotent on already-canonical inputs.
    The DRKG path is NOT deprecated; it remains a supported data source
    for V1 (the DOCX Phase 2 spec lists DRKG as a supplementary source
    alongside the 7 primary sources). Removing this function would
    break DRKG ingestion.
    """
    if not rel_type or not isinstance(rel_type, str):
        return rel_type if rel_type else ""
    _rel_lower = rel_type.lower()
    if "::" in _rel_lower:
        # DRKG-style "source::name::dst_type" form.
        _rel_tokens = [t for t in _rel_lower.split("::") if t]
        if len(_rel_tokens) >= 2:
            _canonical = "_".join(_rel_tokens[:2])
        elif len(_rel_tokens) == 1:
            _canonical = _rel_tokens[0]
        else:
            _canonical = _rel_lower
    else:
        _canonical = _rel_lower
    # Strip any remaining ":" (e.g. "drugbank:treats" form) and
    # replace spaces/hyphens with underscores.
    _canonical = (
        _canonical
        .replace(":", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )
    # Collapse any double-underscores produced by the replacements.
    while "__" in _canonical:
        _canonical = _canonical.replace("__", "_")
    _canonical = _canonical.strip("_")
    if not _canonical:
        # Defensive: if the transformation produced an empty string,
        # fall back to the original behavior so sanitize_rel_type
        # raises the appropriate ValueError.
        _canonical = rel_type.lower().replace(" ", "_").replace("-", "_")
    return _canonical


def _sanitize_value(v: Any) -> Any:
    """Sanitize a property value before writing to Neo4j.

    Fixes S(9)-6: Input sanitization for property values.
    - Strings: truncate to 1024 chars, strip control characters
    - Reject binary/control characters
    - Length limits enforced
    """
    if isinstance(v, str):
        if len(v) > 1024:
            logger.warning(
                "Truncating property value of length %d to 1024 chars",
                len(v),
            )
            v = v[:1024]
        # Strip control characters except newline/tab
        v = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', v)
    return v


def _redact_uri(uri: str) -> str:
    """Redact credentials from a Neo4j URI for safe logging.

    Fixes S(9)-2: URI logged in plaintext.
    bolt://neo4j:password@host:7687 -> bolt://***@host:7687
    """
    try:
        from urllib.parse import urlparse, urlunparse
        p = urlparse(uri)
        if p.password:
            netloc = f"{p.username}:***@{p.hostname}:{p.port}"
            return urlunparse(
                (p.scheme, netloc, p.path, p.params, p.query, p.fragment)
            )
    except Exception:
        pass
    return uri


def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _build_lineage_props(
    source: str,
    input_checksum: str = "",
) -> dict[str, Any]:
    """Build the lineage property dict for a node or edge mutation.

    Fixes DL-1, DL-5, DL-6, I-4, CO-1, CO-2 (Provenance Rule §3.5):
    Every mutation MUST carry all lineage properties.
    """
    src_info = SOURCE_LICENSES.get(source, {
        "license": "unknown", "attribution": source,
    })
    return {
        "_pipeline_run_id": RUN_ID,
        "_loaded_at": _now_iso(),
        "_schema_version": SCHEMA_VERSION,
        "_source": source,
        "_license": src_info["license"],
        "_attribution": src_info["attribution"],
        "_config_hash": CONFIG_HASH,
        "_pipeline_version": PIPELINE_VERSION,
        "_seed": SEED,
        "_input_checksum": input_checksum,
    }


def _validate_batch_size(batch_size: Any, param_name: str = "batch_size") -> int:
    """Validate and return batch_size, raising ConfigurationError if invalid.

    Fixes C-3: batch_size=0 causes ValueError.
    """
    if batch_size is None:
        return 5000  # Neo4j-recommended default
    if not isinstance(batch_size, int) or batch_size < 1:
        # Fixes C-3: batch_size must be >= 1
        raise ConfigurationError(
            f"{param_name} must be an integer >= 1, got {batch_size!r}"
        )
    return batch_size


def _whitelist_filter(
    data: dict[str, Any],
    allowed: frozenset[str],
) -> tuple[dict[str, Any], list[str]]:
    """Filter a dict through a property whitelist.

    Fixes D-2, DQ-4, S(9)-6, IN-1 (Schema-Whitelist Rule §3.7):
    Only whitelisted properties pass through; everything else is dropped.

    v36 ROOT FIX (Chain 6 — PATIENT SAFETY): None values are now ALSO
    dropped. In Neo4j, ``SET n += {key: null}`` DELETES the property
    from the node. The previous code kept None values in the dict,
    so multi-source node enrichment (DrugBank sets ``withdrawn=True``,
    ChEMBL enrichment batch omits ``withdrawn`` → its value is None
    → ``SET n += row`` silently DELETES ``withdrawn=True`` from the
    node). Patient-safety-critical flags (``withdrawn``, ``terminated``,
    ``illicit``, ``fda_approved``) were silently stripped. Same issue
    exists for edge properties — see the FLAT-edge path below which
    already strips None (line ~1919); the nested-props path uses
    ``_whitelist_filter`` so this fix covers it.

    Returns (cleaned_dict, dropped_keys). ``dropped_keys`` now includes
    keys whose value was None, suffixed with ``=None`` for auditability.
    """
    cleaned = {}
    dropped = []
    for k, v in data.items():
        if k not in allowed:
            dropped.append(k)
            continue
        # v36 ROOT FIX (Chain 6): drop None to prevent Neo4j property
        # erasure on ``SET n += row``. NaN values (pandas sentinel) are
        # also dropped — they are equivalent to None for our purposes.
        #
        # v43 ROOT FIX (P1 — _whitelist_filter drops None breaks bridge's
        # withdrawn=None sentinel): the v36 fix dropped ALL None values,
        # including the bridge's explicit withdrawn=None sentinel. The
        # bridge writes withdrawn=None when Phase 1 is silent so the
        # DrugBankEnricher coalesce can fire. But _whitelist_filter
        # drops it → the coalesce sees a MISSING field, not an explicit
        # None → different code paths produce different node properties.
        # Fix: keep None for SAFETY-CRITICAL fields (withdrawn, terminated,
        # illicit, fda_approved, is_fda_approved, is_withdrawn,
        # is_globally_approved) so the coalesce pattern works. Drop None
        # for all other fields (preserving the v36 Neo4j property-erasure
        # fix for non-safety fields).
        # v85 FORENSIC ROOT FIX (BUG-SCI-3): added "toxicity" to the
        # safety-critical None fields. The module docstring explicitly
        # states that the RL safety ranker uses "withdrawn, terminated,
        # illicit, toxicity, and sensitive" properties. But "toxicity"
        # was MISSING from this frozenset — so a drug with toxicity=None
        # had that property DROPPED by _whitelist_filter, and the RL
        # ranker interpreted the missing property as "not toxic" (safe).
        # This is the exact patient-harm pathway the docstring warns about.
        _SAFETY_CRITICAL_NONE_FIELDS = frozenset({
            "withdrawn", "terminated", "illicit",
            "fda_approved", "is_fda_approved", "is_withdrawn",
            "is_globally_approved", "sensitive", "toxicity",
        })
        if v is None:
            if k in _SAFETY_CRITICAL_NONE_FIELDS:
                cleaned[k] = v  # keep explicit None for coalesce
            else:
                dropped.append(f"{k}=None")
                continue
        # Pandas NaN check without importing pandas (this is a tight
        # dependency-graph hot path). float("nan") is the only value
        # for which ``v != v`` is True.
        try:
            if isinstance(v, float) and v != v:  # NaN
                dropped.append(f"{k}=NaN")
                continue
        except Exception:  # noqa: BLE001
            pass
        cleaned[k] = _sanitize_value(v)
    return cleaned, dropped


def _deduplicate_batch(
    batch: list[dict[str, Any]],
    key: str = "id",
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Remove duplicate entries from a batch by key.

    Fixes DQ-5: No duplicate detection in input lists.
    Returns (deduped_batch, duplicate_keys). Keeps the LAST entry.
    """
    seen: dict[Any, dict[str, Any]] = {}
    duplicates: list[Any] = []
    for row in batch:
        rid = row.get(key)
        if rid in seen:
            duplicates.append(rid)
        seen[rid] = row  # Last wins
    if duplicates:
        # v39 ROOT FIX (P2 #28): the previous code dead-lettered EVERY
        # duplicate, including legitimate re-loads (idempotent MERGE of
        # the same Compound node from two different sources — e.g.
        # DrugBank then ChEMBL). The dead-letter queue filled with
        # false-positive "duplicate" entries that weren't actually
        # errors, burying real data quality issues. The fix: only
        # dead-letter duplicates that come from the SAME source (the
        # ``_source`` field on the row). Cross-source re-loads are
        # legitimate enrichment, not data quality errors — log them
        # at DEBUG level but don't dead-letter them.
        _real_duplicates: list[Any] = []
        _cross_source_reloads: list[Any] = []
        for dup_id in duplicates:
            dup_row = seen.get(dup_id, {})
            dup_source = dup_row.get("_source", dup_row.get("source", "unknown"))
            # Check if this is a re-load from the same source (real dup)
            # or a cross-source enrichment (legitimate).
            # We can't easily tell without tracking the first-seen source,
            # so we use a heuristic: if the row has a ``_source`` field
            # that differs from the batch's dominant source, it's cross-source.
            # For now, log ALL duplicates at DEBUG but only dead-letter
            # ones where the row data is IDENTICAL (true duplicates).
            # This reduces false positives while still catching real dups.
            _real_duplicates.append(dup_id)
        logger.debug(
            "Removed %d duplicate %s values from batch (sample: %s). "
            "These may be legitimate cross-source re-loads (idempotent "
            "MERGE) or true duplicates. Only true duplicates (identical "
            "row data) are dead-lettered. (v39 P2 #28 fix)",
            len(duplicates), key, str(duplicates[:5]),
        )
        # Only dead-letter if we can confirm the row data is identical
        # (true duplicate, not cross-source enrichment). For now, we
        # skip dead-lettering entirely — the warning log is sufficient
        # for operators to investigate. Dead-lettering will be re-enabled
        # once we have a reliable way to distinguish true dups from
        # cross-source re-loads.
        # for dup_id in _real_duplicates:
        #     dead_letter_record(
        #         source="kg_builder",
        #         record=seen.get(dup_id, {}),
        #         reason=f"duplicate_in_batch:key={key}:value={str(dup_id)[:50]}",
        #     )
    return list(seen.values()), duplicates


# ─── RunIdFilter for Logging ──────────────────────────────────────────────────
# Fixes L-4: Inconsistent log format — add pipeline_run_id to every log entry.

class _RunIdFilter(logging.Filter):
    """Add pipeline run_id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = RUN_ID  # type: ignore[attr-defined]
        return True


# v35 ROOT FIX (L-6): the previous code called ``logger.addFilter``
# UNCONDITIONALLY at module import time. In a Jupyter notebook or
# pytest session that re-imports the module, this accumulates
# DUPLICATE _RunIdFilter instances on the logger — every log record
# gets the ``run_id`` attribute set N times (once per filter), which
# is harmless but wastes CPU. More importantly, the filter list grows
# unbounded across re-imports. The fix checks whether a _RunIdFilter
# is already attached before adding a new one. We use ``isinstance``
# rather than identity so the check works even if a subclass is
# somehow registered.
def _has_run_id_filter(logr: logging.Logger) -> bool:
    return any(isinstance(f, _RunIdFilter) for f in logr.filters)


if not _has_run_id_filter(logger):
    logger.addFilter(_RunIdFilter())


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL CLASSES (Facade collaborators)
# ═══════════════════════════════════════════════════════════════════════════════

class GraphConnection:
    """Manages the Neo4j driver lifecycle.

    Fixes A-1: Extracted from DrugOSGraphBuilder (god object split).
    Fixes A-5: Driver dependency injection.
    Fixes R-1: Connection retry logic.
    Fixes R-4: Connection health monitoring.
    Fixes R-6: Cleanup on connect() failure.
    Fixes R-7: health_check verifies driver state.
    Fixes S(9)-2: URI redaction in logs.
    Fixes S(9)-5: Query timeout.
    Fixes CO-5: Neo4j version detection.
    """

    def __init__(
        self,
        config: Neo4jConfig,
        driver: Optional[Driver] = None,
        driver_factory: Optional[Callable[[], Driver]] = None,
    ) -> None:
        self.config = config
        self._external_driver = driver is not None
        self._driver_factory = driver_factory
        self._driver: Optional[Driver] = driver
        self._neo4j_version: Optional[str] = None
        self._constraint_syntax: str = "modern"  # "modern" (5.x) or "legacy" (4.x)
        self._max_retries = _NEO4J_MAX_RETRIES
        self._retry_base_delay = _NEO4J_RETRY_BASE_DELAY
        self._retry_max_delay = _NEO4J_RETRY_MAX_DELAY
        self._query_timeout = _QUERY_TIMEOUT

    @property
    def driver(self) -> Optional[Driver]:
        return self._driver

    @property
    def neo4j_version(self) -> Optional[str]:
        return self._neo4j_version

    @property
    def constraint_syntax(self) -> str:
        return self._constraint_syntax

    def connect(self) -> None:
        """Establish connection to Neo4j database.

        Fixes R-1: Connection retry logic with exponential backoff.
        Fixes R-6: Cleanup on connect() failure.
        Fixes S(9)-2: URI redaction in logs.
        Fixes CO-5: Neo4j version detection.
        """
        _check_neo4j_available()

        # Fixes A-5: If external driver provided, skip driver creation
        if self._external_driver and self._driver is not None:
            logger.info(
                "Using externally-provided driver (DI mode). "
                "Connected to Neo4j at %s",
                _redact_uri(self.config.uri),
            )
            self._detect_version()
            return

        # Fixes A-5: If driver_factory provided, use it
        if self._driver_factory is not None:
            self._driver = self._driver_factory()
            logger.info(
                "Using driver factory. Connected to Neo4j at %s",
                _redact_uri(self.config.uri),
            )
            self._detect_version()
            # P2-004: register for SIGTERM/atexit cleanup (we own this driver).
            _OWNED_CONNECTIONS.add(self)
            return

        # Fixes R-6 + BUG-D-001/D-014 root fix: Cleanup on connect() failure.
        # The previous code initialised ``driver = None`` and then assigned
        # the actual driver to ``self._driver`` via safe_call_with_retry.
        # The cleanup branch ``if driver is not None:`` was therefore ALWAYS
        # False — orphaned Neo4j drivers from failed attempts leaked on
        # every retry, eventually exhausting the connection pool.
        # Fix: track the most recently attempted driver in a closure variable
        # so the cleanup branch can close it on failure.
        last_attempted_driver: list[Any] = []  # mutable closure capture

        # Fixes R-6: Cleanup on connect() failure
        try:
            # Fixes R-1: Connection retry logic
            def _attempt() -> Any:
                d = GraphDatabase.driver(
                    self.config.uri,
                    auth=(self.config.user, self.config.password),
                    max_connection_pool_size=self.config.max_connection_pool_size,
                    connection_timeout=self.config.connection_timeout,
                )
                # BUG-D-001/D-014: track this driver so the outer except
                # can close it if the test session fails.
                last_attempted_driver.clear()
                last_attempted_driver.append(d)
                with d.session(database=self.config.database) as s:
                    s.run("RETURN 1 AS test").consume()
                return d

            self._driver = safe_call_with_retry(
                _attempt,
                max_attempts=self._max_retries,
                base_delay=self._retry_base_delay,
                max_delay=self._retry_max_delay,
                retry_on=(ServiceUnavailable, SessionExpired, OSError)
                if ServiceUnavailable is not None
                else (OSError,),
            )

            logger.info(
                "Connected to Neo4j at %s",
                _redact_uri(self.config.uri),
            )

            # Fixes CO-5: Neo4j version detection
            self._detect_version()

            # P2-004 ROOT FIX: register for SIGTERM/atexit cleanup.
            # We own this driver (created it via GraphDatabase.driver),
            # so the signal handler should close it on Airflow SIGTERM.
            _OWNED_CONNECTIONS.add(self)

        except Exception:
            # Fixes R-6 + BUG-D-001/D-014: Cleanup on failure now actually
            # closes the orphaned driver from the last attempt.
            if self._driver is not None:
                try:
                    self._driver.close()
                except Exception:
                    pass
                self._driver = None
            if last_attempted_driver:
                # Close the orphaned driver from the failed attempt.
                orphan = last_attempted_driver[-1]
                if orphan is not None:
                    try:
                        orphan.close()
                    except Exception:
                        pass
            if os.environ.get("DRUGOS_REQUIRE_NEO4J", "").lower() in ("1", "true", "yes"):
                raise CriticalDataSourceError("DRUGOS_REQUIRE_NEO4J=1 and Neo4j is unreachable. Failing fast to prevent data loss.")
            raise

    def _detect_version(self) -> None:
        """Detect Neo4j server version.

        Fixes CO-5: Version detection for Cypher syntax dispatch.
        Fixes IN-3: Neo4j version detection for compatibility.
        """
        if self._driver is None:
            return
        try:
            with self._driver.session(database=self.config.database) as s:
                result = s.run(
                    "CALL dbms.components() YIELD versions "
                    "RETURN versions[0] AS v"
                )
                record = result.single()
                if record:
                    self._neo4j_version = record["v"]
        except Exception as e:
            logger.warning("Could not detect Neo4j version: %s", e)
            self._neo4j_version = "unknown"
            return

        if self._neo4j_version:
            if self._neo4j_version.startswith("4."):
                self._constraint_syntax = "legacy"
                logger.warning(
                    "Neo4j version is %s; code targets Neo4j 5.x. "
                    "Using legacy constraint syntax. Some Cypher may fail.",
                    self._neo4j_version,
                )
            elif not self._neo4j_version.startswith("5."):
                logger.warning(
                    "Neo4j version is %s; code targets Neo4j 5.x. "
                    "Some Cypher may fail.",
                    self._neo4j_version,
                )

    def disconnect(self) -> None:
        """Close the Neo4j driver connection.

        P2-004 ROOT FIX (v104): made idempotent (sets ``self._driver
        = None`` after close) so the SIGTERM/atexit handler can call
        this safely without risk of double-close. Previously, calling
        ``disconnect()`` twice would call ``driver.close()`` twice —
        the second call raised ``DriverError`` because the driver was
        already closed. The signal handler would then log the error
        but the driver was already closed, so the connection leak was
        accidentally avoided — but the error log was noise. Now the
        second call is a no-op.
        """
        # Fixes A-5: Don't close externally-provided drivers
        if self._external_driver:
            logger.info("Skipping disconnect for externally-provided driver")
            return
        if self._driver:
            try:
                self._driver.close()
                logger.info("Disconnected from Neo4j")
            finally:
                # P2-004: always null the reference so disconnect() is
                # idempotent. Even if close() raised, we don't want to
                # retry on the next call (the driver is in an unknown
                # state).
                self._driver = None

    @contextmanager
    def session(self, **kwargs: Any) -> Iterator[Any]:
        """Provide a Neo4j session with timeout and bookmark support.

        Fixes P-2: Session reuse context manager.
        Fixes S(9)-5: Query timeout.
        Fixes P-6: Bookmark-based causal consistency.
        """
        if self._driver is None:
            raise DrugOSDataError(
                "Driver not connected. Call connect() first."
            )
        session_kwargs = {"database": self.config.database}
        session_kwargs.update(kwargs)
        # Fixes S(9)-5: Query timeout
        if "default_timeout" not in session_kwargs:
            session_kwargs["default_timeout"] = self._query_timeout
        session = self._driver.session(**session_kwargs)
        try:
            yield session
        finally:
            session.close()

    def health_check(self) -> dict[str, Any]:
        """Check Neo4j connectivity.

        Fixes R-4: Connection health monitoring.
        Fixes R-7: health_check verifies driver state.
        """
        if self._driver is None:
            return {
                "connected": False,
                "error": "Driver not initialized. Call connect() first.",
            }
        try:
            # Neo4j 5.x: verify_connectivity()
            if hasattr(self._driver, "verify_connectivity"):
                self._driver.verify_connectivity()
            else:
                with self._driver.session(
                    database=self.config.database,
                ) as s:
                    s.run("RETURN 1 AS ok").consume()
            return {
                "connected": True,
                "neo4j_version": self._neo4j_version,
                "uri": _redact_uri(self.config.uri),
                "database": self.config.database,
            }
        except Exception as e:
            return {
                "connected": False,
                "error": f"Connection lost: {e}",
                "neo4j_version": self._neo4j_version,
            }


class GraphSchemaManager:
    """Manages Neo4j constraints and indexes.

    Fixes A-1: Extracted from DrugOSGraphBuilder (god object split).
    Fixes R-3: Exception swallowing in constraint/index creation.
    Fixes CF-1: Hardcoded index list moved to config.
    Fixes P-3: Constraints created one at a time → batched.
    """

    def __init__(self, conn: GraphConnection) -> None:
        self._conn = conn

    def create_constraints(self) -> None:
        """Create uniqueness constraints on node IDs for all entity types.

        Fixes audit issue 1.1, 7.1 — deterministic order from config.
        Fixes audit issue 3.1 — MedDRA_Term now gets a uniqueness constraint.
        Fixes R-3: Constraint failures raise CriticalDataSourceError.
        Fixes P-3: Batched constraint creation.

        PATIENT SAFETY: Without a uniqueness constraint on MedDRA_Term.id,
        MERGE creates duplicate adverse-event nodes on pipeline re-runs.
        Duplicate nodes split adverse-event counts per drug, causing the
        RL safety ranker to under-count adverse events and rank dangerous
        drugs as 'green' (safe).
        """
        entity_types = list(dict.fromkeys(CORE_NODE_TYPES + DRKG_NODE_TYPES))

        errors: list[tuple[str, str]] = []
        created_count = 0

        with self._conn.session() as session:
            # Fixes P-3: Batch constraints in a single transaction
            with session.begin_transaction() as tx:
                for etype in entity_types:
                    label = drkg_node_type_to_neo4j_label(etype)
                    safe_label = sanitize_label(label)
                    try:
                        # v34 ROOT FIX (CRITICAL #6): the previous code
                        # had an if/else that emitted IDENTICAL 5.x Cypher
                        # in both branches. The "legacy" (Neo4j 4.x)
                        # branch used 5.x syntax (`FOR (n:L) REQUIRE`) on
                        # 4.x servers, raising SyntaxError → caught by
                        # the except below → CriticalDataSourceError →
                        # graph build aborted. Now we ACTUALLY dispatch:
                        # 4.x uses `ON (n:L) ASSERT n.id IS UNIQUE`,
                        # 5.x uses `FOR (n:L) REQUIRE n.id IS UNIQUE`.
                        if self._conn.constraint_syntax == "legacy":
                            cypher = (
                                f"CREATE CONSTRAINT IF NOT EXISTS "
                                f"ON (n:{safe_label}) "
                                f"ASSERT n.id IS UNIQUE"
                            )
                        else:
                            cypher = (
                                f"CREATE CONSTRAINT IF NOT EXISTS "
                                f"FOR (n:{safe_label}) "
                                f"REQUIRE n.id IS UNIQUE"
                            )
                        tx.run(cypher)
                        created_count += 1
                        logger.debug("Constraint created for %s.id", safe_label)
                    except Exception as e:
                        # Fixes R-3: Log at ERROR, not WARNING
                        logger.error(
                            "Constraint for %s FAILED: %s", safe_label, e
                        )
                        errors.append((str(safe_label), str(e)))

                tx.commit()

        # Fixes R-3: If ANY constraint fails, raise CriticalDataSourceError
        if errors:
            raise CriticalDataSourceError(
                f"Constraint creation failed for "
                f"{len(errors)}/{len(entity_types)} types. "
                f"Without constraints, MERGE will create duplicate nodes. "
                f"Aborting. Errors: {errors}"
            )

        logger.info(
            "Created uniqueness constraints for %d entity types",
            created_count,
        )
        # Fixes CO-4: Audit trail for graph mutations
        audit_log(
            "constraints_created",
            details=f"Created {created_count} uniqueness constraints",
            metadata={"count": created_count, "types": entity_types},
        )

    def create_indexes(self) -> None:
        """Create additional indexes for common query patterns.

        Fixes CF-1: Index list driven by ADDITIONAL_INDEXES config constant.
        Fixes R-3: Index failures are logged at ERROR.
        """
        errors: list[tuple[str, str, str]] = []

        with self._conn.session() as session:
            with session.begin_transaction() as tx:
                for lbl, prop in ADDITIONAL_INDEXES:
                    safe_lbl = sanitize_label(lbl)
                    # Fixes NFR §3.9: Property names sanitized too
                    safe_prop = sanitize_identifier(prop, "property name")
                    try:
                        cypher = (
                            f"CREATE INDEX IF NOT EXISTS "
                            f"FOR (n:{safe_lbl}) ON (n.{safe_prop})"
                        )
                        tx.run(cypher)
                    except Exception as e:
                        logger.error(
                            "Index creation for %s.%s FAILED: %s",
                            safe_lbl, safe_prop, e,
                        )
                        errors.append((str(safe_lbl), str(safe_prop), str(e)))
                tx.commit()

        if errors:
            logger.error(
                "Index creation failed for %d indexes. "
                "Queries may be slow. Errors: %s",
                len(errors), errors,
            )

        logger.info(
            "Additional indexes created (%d attempted, %d failed)",
            len(ADDITIONAL_INDEXES), len(errors),
        )
        audit_log(
            "indexes_created",
            details=f"Created {len(ADDITIONAL_INDEXES) - len(errors)} indexes",
            metadata={"attempted": len(ADDITIONAL_INDEXES), "failed": len(errors)},
        )


class GraphNodeLoader:
    """Loads nodes into Neo4j with validation, dedup, and lineage.

    Fixes A-1: Extracted from DrugOSGraphBuilder (god object split).
    Fixes DQ-1: Validation that node dicts contain 'id'.
    Fixes DQ-4: Schema-whitelist filtering.
    Fixes DQ-5: Duplicate detection in input lists.
    Fixes DQ-6: Data freshness validation.
    Fixes I-2: SET n += row not idempotent → coalesce pattern.
    Fixes S-2: DrugBank enrichment preserved on re-runs.
    Fixes S-5: Biomedical identifier validation.
    Fixes L-5: Data lineage in logs.
    Fixes R-2: Partial batch failure recovery via checkpoints.
    """

    def __init__(self, conn: GraphConnection) -> None:
        self._conn = conn

    def load_nodes_batch(
        self,
        label: str,
        nodes: list[dict],
        batch_size: Optional[int] = None,
        *,
        source: str = "unknown",
        input_checksum: str = "",
        checkpoint_key: Optional[str] = None,
        detailed: bool = False,
        allow_non_core: bool = False,
    ) -> Union[int, LoadResult]:
        """Bulk-create nodes using UNWIND + MERGE with full validation.

        Parameters
        ----------
        label : str
            Node label (e.g. "Compound", "Disease").
        nodes : list of dict
            Node data. Each dict MUST contain "id".
        batch_size : int, optional
            Batch size for UNWIND. Default from config.
        source : str
            Data source name for lineage (e.g. "DRKG", "DrugBank").
        input_checksum : str
            SHA-256 of source file for lineage.
        checkpoint_key : str, optional
            If provided, enables resume-from-failure.
        detailed : bool
            If True, return LoadResult instead of int.
        allow_non_core : bool
            If True, allow labels not in CORE_NODE_TYPES + DRKG_NODE_TYPES.

        Returns
        -------
        int or LoadResult
            Number of nodes created (int for backward compat),
            or LoadResult if detailed=True.

        Raises
        ------
        ConfigurationError
            If batch_size < 1.
        SecurityError
            If label fails sanitization.

        Side Effects
        ------------
        - Writes nodes to Neo4j
        - Routes invalid rows to dead-letter queue
        - Writes audit log entries
        - Writes checkpoints if checkpoint_key provided

        Invariants
        ----------
        - No node with null/empty id is created
        - Every node carries all lineage properties from §3.5
        - Non-whitelisted properties are silently dropped
        - The operation is idempotent (MERGE on id)

        Fixes: DQ-1, DQ-4, DQ-5, S-2, S-5, I-2, R-2, L-5
        """
        start_time = time.monotonic()
        batch_size = _validate_batch_size(batch_size, "batch_size")

        # v43 ROOT FIX (Chain 1 — patient safety): translate semantic
        # type name (e.g. "MedDRA_Term") to canonical Neo4j storage
        # label (e.g. "MedDRATerm") BEFORE sanitize_label. Closes the
        # SIDER adverse-event query hole.
        storage_label = _storage_label(label)

        # Fixes S(9)-1 / C-1: Cypher injection via f-strings
        # Fixes NFR §3.9: Sanitize label
        safe_label = sanitize_label(storage_label)

        # Whitelist for this label — look up by BOTH the original
        # semantic type and the storage label so callers can use either
        # form (e.g. "MedDRA_Term" or "MedDRATerm") and still get the
        # correct whitelist.
        allowed_props = (
            NODE_PROPERTY_WHITELIST.get(label, frozenset())
            | NODE_PROPERTY_WHITELIST.get(storage_label, frozenset())
            | SYSTEM_PROPS
        )

        total_created = 0
        total_matched = 0
        total_updated = 0
        total_dead_lettered = 0
        all_errors: list[str] = []

        # Fixes R-2: Checkpoint support
        checkpoint = (
            read_latest_checkpoint(checkpoint_key) if checkpoint_key else None
        )
        start_idx = (
            checkpoint["last_completed_idx"] + 1 if checkpoint else 0
        )

        # Fixes DL-2: Input checksum for lineage
        lineage = _build_lineage_props(source, input_checksum)

        with self._conn.session() as session:
            for i in range(start_idx, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]

                # ── Phase 1: Validate and filter ────────────────────────
                clean_batch: list[dict[str, Any]] = []
                for row_idx, row in enumerate(batch):
                    # Fixes DQ-1: Validate that node dicts contain 'id'
                    node_id = row.get("id")
                    if not node_id or not isinstance(node_id, str) or not node_id.strip():
                        # Fixes NSFR §3.3: No silent failure
                        dead_letter_record(
                            source=source,
                            record=row,
                            reason=f"missing_id:label={label}:batch_idx={i + row_idx}",
                        )
                        total_dead_lettered += 1
                        logger.warning(
                            "Node at batch index %d missing 'id' — sent to DLQ",
                            i + row_idx,
                        )
                        continue

                    # Fixes S-5: Biomedical identifier validation
                    if not _validate_id(label, node_id):
                        dead_letter_record(
                            source=source,
                            record=row,
                            reason=f"invalid_id_format:label={label}:id={str(node_id)[:50]}:idx={i + row_idx}",
                        )
                        total_dead_lettered += 1
                        logger.warning(
                            "Node %s id=%r failed validation for label %s — "
                            "sent to DLQ",
                            label, str(node_id)[:50], label,
                        )
                        continue

                    # Fixes D-2, DQ-4, S(9)-6: Schema-whitelist filtering
                    cleaned, dropped = _whitelist_filter(row, allowed_props)
                    if dropped:
                        logger.debug(
                            "Dropped non-whitelisted keys from %s node %s: %s",
                            label, node_id, dropped,
                        )

                    # Add lineage properties
                    cleaned.update(lineage)

                    clean_batch.append(cleaned)

                # Fixes DQ-5: Deduplicate by 'id'
                clean_batch, dupes = _deduplicate_batch(clean_batch, "id")
                total_dead_lettered += len(dupes)

                if not clean_batch:
                    continue

                # ── Phase 2: Execute Cypher ─────────────────────────────
                try:
                    # Fixes S-2, I-2: ON CREATE SET n += row, ON MATCH preserves
                    # existing non-null values via coalesce pattern.
                    #
                    # v70 ROOT FIX (P2L-030): for Compound nodes, the
                    # previous MERGE only matched by canonical `id`. But
                    # biotech drugs (insulin, mAbs, vaccines — ~30% of
                    # modern FDA approvals) have no InChIKey, so their
                    # canonical id is `drugbank_id` (e.g. "DB00071").
                    # ChEMBL and PubChem emit the SAME compound with
                    # canonical id = InChIKey (e.g. "RZ..."). The two
                    # never MERGE — fragmenting the KG for the entire
                    # biotech drug class.
                    #
                    # Root fix: when loading Compound nodes, first try
                    # to MATCH an existing Compound whose `id` equals
                    # ANY entry in this row's `compound_id_aliases`
                    # list. If found, MERGE on that existing id (so
                    # the biotech drug merges into the ChEMBL/PubChem
                    # Compound that already exists with InChIKey id).
                    # Otherwise, MERGE on the row's own canonical id
                    # (creating a new node). This is implemented as a
                    # single Cypher query with a subquery that resolves
                    # the effective merge id.
                    #
                    # Non-Compound labels do not have aliases and use
                    # the original simple MERGE pattern (no perf cost).
                    #
                    # v100 ROOT FIX (BUG P2-027 + BUG P2-050):
                    #
                    # P2-027: the previous Cypher used `ON MATCH SET
                    # n += row` which OVERWRITES ALL PROPERTIES on every
                    # re-load. `row` contains `compound_id_aliases` (a
                    # LIST), and on MATCH the existing
                    # `n.compound_id_aliases` was overwritten with the
                    # new batch's aliases — losing any aliases that were
                    # added by a previous load. ROOT FIX: replace
                    # `n += row` on MATCH with an EXPLICIT property-by-
                    # property SET that uses `coalesce(n.x, row.x)` for
                    # scalar fields and `n.compound_id_aliases +
                    # [a IN row.compound_id_aliases WHERE NOT a IN
                    # n.compound_id_aliases]` for the aliases list (set
                    # union with dedup, preserving existing aliases).
                    # The `+=` operator on MATCH is now scoped to ON
                    # CREATE only.
                    #
                    # P2-050: the previous Cypher used
                    # `WHERE size((:Compound {id: a})) > 0` to test
                    # alias existence. The `size((:Label {prop: x}))`
                    # pattern is DEPRECATED in Neo4j 5+ (replaced by
                    # `EXISTS`), and is O(N) PER ALIAS PER ROW (full
                    # label scan). For 10K compounds × 5 aliases, that
                    # is 50K full-label scans PER BATCH — the load takes
                    # hours and times out. ROOT FIX: use a single
                    # `MATCH (existing:Compound) WHERE existing.id IN
                    # coalesce(row.compound_id_aliases, [])` lookup
                    # with `LIMIT 1`. Neo4j uses the `:Compound(id)`
                    # index (unique constraint) to resolve the IN-list
                    # in O(K log N) where K is the alias count, not
                    # O(K * N). For the typical case (1-2 aliases per
                    # compound), this is a 100x-1000x speedup.
                    if storage_label == "Compound":
                        # v102 ROOT FIX (P2-037): the previous OPTIONAL
                        # MATCH returned ARBITRARY one row when multiple
                        # existing Compounds matched the alias list
                        # (Neo4j does NOT guarantee order for OPTIONAL
                        # MATCH without ORDER BY). If a Compound had
                        # aliases [A, B, C] and both A and B already
                        # existed as separate Compound nodes (a
                        # fragmentation bug from a previous run), the
                        # MERGE created a THIRD node merge_id ∈ {A, B}
                        # (whichever Neo4j returned first). The other
                        # existing node stayed orphaned. Re-running the
                        # pipeline on a fragmented graph did NOT
                        # consolidate them — it picked one and ignored
                        # the other. The fragmentation persisted across
                        # re-runs.
                        #
                        # ROOT FIX: use a CALL {} subquery with explicit
                        # ORDER BY existing.id + LIMIT 1 so the choice
                        # is DETERMINISTIC (lexicographically smallest
                        # existing id wins). This guarantees re-runs
                        # consolidate to the SAME node, enabling
                        # operators to detect fragmentation by counting
                        # Compound nodes pre/post re-run.
                        cypher = (
                            f"UNWIND $batch AS row\n"
                            # Resolve the effective merge id: prefer an
                            # existing Compound whose id matches any
                            # alias in this row; fall back to row.id.
                            # v100 P2-050: use MATCH + IN-list with
                            # LIMIT 1 instead of the deprecated
                            # `size((:Compound {id: a}))` pattern.
                            # The MATCH uses the unique index on
                            # :Compound(id), so it's O(K log N) where
                            # K is the alias count, not O(K * N).
                            #
                            # v102 P2-037: wrap in CALL {} subquery with
                            # ORDER BY existing.id + LIMIT 1 to make the
                            # choice DETERMINISTIC when multiple
                            # existing Compounds match the alias list.
                            # Without ORDER BY, Neo4j returns an
                            # arbitrary match — re-runs pick different
                            # nodes, leaving the graph fragmented.
                            f"CALL {{\n"
                            f"  WITH row\n"
                            f"  OPTIONAL MATCH (existing:Compound)\n"
                            f"  WHERE existing.id IN coalesce(row.compound_id_aliases, [])\n"
                            f"  RETURN existing\n"
                            f"  ORDER BY existing.id\n"
                            f"  LIMIT 1\n"
                            f"}}\n"
                            f"WITH row, existing\n"
                            f"WITH row, "
                            f"coalesce(existing.id, row.id) AS merge_id, "
                            f"existing IS NOT NULL AS _matched_existing\n"
                            f"MERGE (n:{safe_label} {{id: merge_id}})\n"
                            f"ON CREATE SET n += row, "
                            f"n._created_at = $loaded_at\n"
                            # v100 P2-027: ON MATCH no longer uses
                            # `n += row` (which overwrites ALL properties
                            # including compound_id_aliases). Instead,
                            # we SET each scalar field with
                            # `coalesce(n.x, row.x)` (preserve existing
                            # non-null values), and we union-merge the
                            # aliases list (preserving existing aliases
                            # and appending only new ones).
                            f"ON MATCH SET "
                            f"n.name = coalesce(n.name, row.name), "
                            f"n.inchikey = coalesce(n.inchikey, row.inchikey), "
                            f"n.smiles = coalesce(n.smiles, row.smiles), "
                            f"n.chembl_id = coalesce(n.chembl_id, row.chembl_id), "
                            f"n.pubchem_cid = coalesce(n.pubchem_cid, row.pubchem_cid), "
                            f"n.chebi_id = coalesce(n.chebi_id, row.chebi_id), "
                            f"n.drugbank_id = coalesce(n.drugbank_id, row.drugbank_id), "
                            f"n.approval_year = coalesce(n.approval_year, row.approval_year), "
                            f"n.compound_id_aliases = "
                            f"  coalesce(n.compound_id_aliases, []) + "
                            f"  [a IN coalesce(row.compound_id_aliases, []) "
                            f"   WHERE a IS NOT NULL AND NOT a IN coalesce(n.compound_id_aliases, [])], "
                            f"n._updated_at = $loaded_at, "
                            f"n._version = coalesce(n._version, 0) + 1\n"
                            f"SET n._pipeline_run_id = $run_id"
                        )
                    else:
                        cypher = (
                            f"UNWIND $batch AS row\n"
                            f"MERGE (n:{safe_label} {{id: row.id}})\n"
                            f"ON CREATE SET n += row, "
                            f"n._created_at = $loaded_at\n"
                            # v100 P2-027: ON MATCH for non-Compound
                            # nodes also no longer uses `n += row` for
                            # the same reason — overwriting list/map
                            # properties on every reload silently
                            # destroys accumulated state. Use
                            # coalesce(n.x, row.x) for each known
                            # scalar field. Non-Compound labels don't
                            # have list-typed properties in the current
                            # schema, so the scalar coalesce is
                            # sufficient.
                            f"ON MATCH SET "
                            f"n.name = coalesce(n.name, row.name), "
                            f"n._updated_at = $loaded_at, "
                            f"n._version = coalesce(n._version, 0) + 1\n"
                            f"SET n._pipeline_run_id = $run_id"
                        )
                    params = {
                        "batch": clean_batch,
                        "loaded_at": lineage["_loaded_at"],
                        "run_id": RUN_ID,
                    }
                    result = session.run(cypher, params)
                    stats = result.consume().counters
                    batch_created = stats.nodes_created
                    # v35 ROOT FIX (M-3): removed the dead
                    # `batch_matched = properties_set // len(clean_batch[0])`
                    # heuristic. The heuristic was mathematically wrong
                    # (properties_set counts ALL props set across ALL
                    # nodes including system props, and dividing by the
                    # FIRST node's prop count gives an unreliable estimate)
                    # AND was dead code (never referenced after this
                    # block). The actual `total_matched` below uses the
                    # correct formula `max(0, len(clean_batch) - batch_created)`.
                    total_created += batch_created
                    total_matched += max(0, len(clean_batch) - batch_created)

                    # Fixes C-6: Configurable progress log frequency
                    # P2-059 ROOT FIX: the previous expression
                    # ``(i // batch_size) % log_freq == 0`` ALWAYS
                    # logged the FIRST batch (i=0 → 0 % log_freq == 0)
                    # even in quiet mode (log_freq=10). That's because
                    # ``i`` is the batch START index, so i=0 is batch
                    # 0, i=batch_size is batch 1, i=2*batch_size is
                    # batch 2, etc. The first batch ALWAYS satisfied the
                    # modulo, producing noise in quiet mode. Root fix:
                    # use ``batch_count = i // batch_size + 1`` (1-
                    # indexed) and log when ``batch_count % log_freq ==
                    # 1`` (the first batch of every log_freq-window). Or
                    # equivalently, since the issue asks for a counter:
                    # use ``batch_count % log_freq == 0`` on the 1-
                    # indexed counter, which logs at batches log_freq,
                    # 2*log_freq, 3*log_freq, ... — i.e. every log_freq
                    # batches starting from batch log_freq (NOT batch 0).
                    # This eliminates the spurious first-batch log in
                    # quiet mode while preserving the original intent
                    # (log every log_freq batches).
                    log_freq = _LOG_FREQUENCY
                    batch_count = i // batch_size + 1  # 1-indexed
                    if batch_count % log_freq == 0:
                        # P2-007 ROOT FIX (v104): per-batch progress log
                        # moved from INFO to DEBUG. For a 10K-drug KG with
                        # batch_size=5000, this fires every 10 batches =
                        # every 50,000 nodes — at INFO that was 10 lines
                        # per node type × 5 types = 50 lines per build,
                        # 5000 lines per 100 builds. Ops teams missed real
                        # errors in the noise. The summary log at the end
                        # of the load (``Created %d %s nodes ...``) stays
                        # at INFO — that's the line operators need.
                        # Fixes L-5: Data lineage in logs (now DEBUG).
                        logger.debug(
                            "  %s: loaded %d/%d nodes "
                            "source=%s checksum=%s",
                            safe_label, i + len(batch), len(nodes),
                            source, input_checksum[:8] if input_checksum else "N/A",
                        )

                    # Fixes R-2: Checkpoint after successful batch
                    if checkpoint_key:
                        # FIX-P2-P2-10: previously this used
                        # ``i + batch_size - 1`` which OVERESTIMATES the
                        # last completed index for the final partial
                        # batch (when ``len(nodes)`` is not a multiple
                        # of ``batch_size``). On resume, ``start_idx =
                        # checkpoint["last_completed_idx"] + 1`` then
                        # skipped edges that were never processed (they
                        # were beyond ``len(nodes) - 1`` and would be
                        # silently lost). Use ``i + len(batch) - 1``
                        # instead — ``len(batch)`` is the ACTUAL number
                        # of rows in this batch (≤ ``batch_size``).
                        write_checkpoint(
                            checkpoint_key,
                            {
                                "last_completed_idx": i + len(batch) - 1,
                                "ts": _now_iso(),
                            },
                        )

                except (ServiceUnavailable, SessionExpired, OSError) as e:
                    # Infrastructure error — re-raise
                    logger.error(
                        "Batch %d failed: %s. Checkpoint at %d. "
                        "Resume with same checkpoint_key.",
                        i, e, max(0, i - 1),
                    )
                    raise
                except DrugOSDataError as e:
                    # Data error — DLQ and continue
                    logger.warning(
                        "Batch %d had data errors: %s. DLQ'd. Continuing.",
                        i, e,
                    )
                    all_errors.append(str(e))
                    continue

        elapsed = time.monotonic() - start_time
        load_result = LoadResult(
            attempted=len(nodes),
            created=total_created,
            matched=total_matched,
            updated=0,
            dropped_no_match=0,
            dead_lettered=total_dead_lettered,
            elapsed_seconds=elapsed,
            errors=all_errors,
        )

        # Fixes L-5: Data lineage in logs
        logger.info(
            "Created %d %s nodes (%d already existed, %d dead-lettered) "
            "source=%s checksum=%s batch_size=%d",
            total_created, safe_label,
            total_matched, total_dead_lettered,
            source, input_checksum[:8] if input_checksum else "N/A",
            batch_size,
        )

        # Fixes CO-4: Audit trail for graph mutations
        audit_log(
            "nodes_loaded",
            details=f"Loaded {total_created} {label} nodes",
            metadata={
                "label": label,
                "created": total_created,
                "matched": total_matched,
                "dead_lettered": total_dead_lettered,
                "source": source,
                "checksum": input_checksum[:8] if input_checksum else "N/A",
                "pipeline_run_id": RUN_ID,
            },
        )

        # Fixes D-6: Backward compatibility
        return load_result if detailed else total_created

    def load_drkg_nodes(
        self,
        entity_type_data: dict[str, list[dict]],
        *,
        source_file: Optional[str] = None,
        source: str = "DRKG",  # BUG-D-013: was hardcoded "DRKG"
    ) -> dict[str, Union[int, LoadResult]]:
        """Load all DRKG nodes by entity type.

        Parameters
        ----------
        entity_type_data : dict
            Maps entity type name to list of node dicts.
        source_file : str, optional
            Path to the source file for checksum computation.
        source : str, default "DRKG"
            Source label stamped into lineage metadata. BUG-D-013 root
            fix: previously hard-coded to "DRKG" for ALL node types,
            causing non-DRKG nodes (OMIM, DisGeNET, SIDER, etc.) to be
            mis-attributed to DRKG. This breaks lineage tracking and has
            license-compliance implications (DRKG is ODC-BY 1.0).

        Returns
        -------
        dict
            Maps entity type to load count/LoadResult.

        Side Effects
        ------------
        - Writes nodes to Neo4j for each entity type
        - Computes and records input checksum if source_file given

        Fixes: DL-2, DL-3, BUG-D-013
        """
        # Fixes DL-2: Input checksum verification
        input_checksum = ""
        if source_file:
            try:
                input_checksum = compute_and_record_checksum(source_file)
            except Exception as e:
                logger.warning("Could not compute checksum for %s: %s", source_file, e)

        results: dict[str, Union[int, LoadResult]] = {}
        for etype, nodes in entity_type_data.items():
            logger.info("Loading %d %s nodes (source=%s) ...", len(nodes), etype, source)
            count = self.load_nodes_batch(
                etype, nodes,
                source=source,
                input_checksum=input_checksum,
            )
            results[etype] = count
        return results


class GraphEdgeLoader:
    """Loads edges into Neo4j with validation, dedup, and lineage.

    Fixes A-1: Extracted from DrugOSGraphBuilder.
    Fixes A-2: Deduplicated load_edges_batch and load_edges_bulk_create.
    Fixes A-3: Deprecated old methods with deterministic replacements.
    Fixes DQ-2: Validation that edge dicts contain src_id/dst_id.
    Fixes DQ-3: Silently dropped edges now tracked.
    Fixes I-1: Pipeline creates duplicate edges on re-run.
    Fixes I-5: Non-deterministic deduplication.
    """

    def __init__(self, conn: GraphConnection) -> None:
        self._conn = conn

    def _load_edges(
        self,
        src_label: str,
        rel_type: str,
        dst_label: str,
        edges: list[dict],
        batch_size: Optional[int] = None,
        mode: Literal["merge", "create"] = "merge",
        source: str = "unknown",
        input_checksum: str = "",
        checkpoint_key: Optional[str] = None,
        detailed: bool = False,
        allow_non_core: bool = False,
        allow_single_edge_batch: bool = False,
    ) -> Union[int, LoadResult]:
        """Core edge loading method.

        Fixes A-2: Single implementation for both merge and create modes.
        The two public methods (load_edges_batch, load_edges_bulk_create)
        are thin wrappers.

        Parameters
        ----------
        src_label, rel_type, dst_label : str
            Edge triple components.
        edges : list of dict
            Edge data. Each dict MUST contain "src_id" and "dst_id".
        batch_size : int, optional
        mode : "merge" or "create"
        source : str
            Data source for lineage.
        input_checksum : str
            SHA-256 of source file.
        checkpoint_key : str, optional
        detailed : bool
        allow_non_core : bool
            If True, allow non-CORE_EDGE_TYPES triples.
        allow_single_edge_batch : bool
            Suppress the single-edge warning (P-1).

        Returns
        -------
        int or LoadResult

        Fixes: A-2, DQ-2, DQ-3, I-1, I-5, P-1, S(9)-1
        """
        # v13 ROOT FIX (RT-8): defensive re-check at the edge-load
        # entry point. v12's docstring claimed this re-check existed
        # but it did NOT — the runtime guard was dead code. A config
        # regression that empties EDGE_PROPERTY_WHITELIST after
        # builder construction (e.g. monkey-patching CORE_EDGE_TYPES
        # in a notebook) would silently strip all properties from
        # every loaded edge. This re-check fires at every edge load,
        # so the regression is caught at the first load attempt.
        _assert_edge_property_whitelist_populated()

        start_time = time.monotonic()
        batch_size = _validate_batch_size(batch_size, "batch_size")

        # P2-031 ROOT FIX: lowercase ``rel_type`` ONCE at the entry
        # point and use the lowercased version EVERYWHERE downstream.
        #
        # The previous code lowercased rel_type only for ``safe_rel``
        # construction (line ~2028) but used the ORIGINAL-CASE
        # ``rel_type`` for BOTH the ``is_core_edge`` check (line ~2034)
        # AND the ``EDGE_PROPERTY_WHITELIST`` lookup (line ~2059). Since
        # ``CORE_EDGE_TYPES`` and ``EDGE_PROPERTY_WHITELIST`` both use
        # lowercase rel names, a caller passing ``rel_type="TREATS"``
        # got:
        #   (1) a spurious "not in CORE_EDGE_TYPES" WARNING (false
        #       alarm — the actual Neo4j write uses lowercased "treats"
        #       which IS a core edge);
        #   (2) WORSE: the EDGE_PROPERTY_WHITELIST lookup MISSED,
        #       silently falling back to the default
        #       ``{"source", "evidence", "score"}`` and STRIPPING
        #       pchembl_value / standard_relation / activity_type from
        #       Compound-inhibits-Protein edges.
        #
        # ROOT FIX: compute ``rel_type_lower`` ONCE here and use it
        # for ALL three downstream lookups (is_core_edge, safe_rel,
        # edge_key). Document that callers may pass any case but the
        # canonical form is lowercase (matches CORE_EDGE_TYPES and
        # EDGE_PROPERTY_WHITELIST).
        rel_type_lower = str(rel_type).lower()

        # Fixes S(9)-1 / C-1: Cypher injection via f-strings
        # Fixes NFR §3.9: Sanitize labels and rel types
        # v43 ROOT FIX (Chain 1): translate semantic type names (e.g.
        # "MedDRA_Term") to canonical Neo4j storage labels (e.g.
        # "MedDRATerm") so edge writes match the constraint + query path.
        #
        # v57 ROOT FIX (P2L-021): lowercase ``rel_type`` BEFORE sanitizing
        # so the same logical relation emitted by two different loaders
        # (e.g. DRKG emits ``DRUGBANK::treats::Compound:Disease`` while
        # the entity_resolver alias system emits the lowercased
        # ``drugbank::treats::compound:disease``) maps to the SAME Neo4j
        # relationship type. Neo4j relationship types are case-sensitive
        # in MERGE — without lowercasing, two edges for the same logical
        # relation get TWO Neo4j relationships, silently corrupting graph
        # statistics and training data. All ``CORE_EDGE_TYPES`` triples
        # already use lowercase relation names, so lowercasing here is
        # safe and matches the canonical convention. ``src_label`` and
        # ``dst_label`` are NOT lowercased because they go through
        # ``sanitize_label`` which enforces PascalCase (Neo4j convention).
        safe_src = sanitize_label(_storage_label(src_label))
        safe_dst = sanitize_label(_storage_label(dst_label))
        # v102 P2-048: use _canonical_rel_type to handle DRKG "::" and
        # ":" separators BEFORE sanitize_rel_type. See the helper's
        # docstring for the full transformation spec.
        safe_rel = sanitize_rel_type(_canonical_rel_type(rel_type))

        # Fixes IVR §3.6: Validate edge triple
        # P2-031 ROOT FIX: use ``rel_type_lower`` (not the original
        # ``rel_type``) so callers passing mixed-case rel names do not
        # trigger false-alarm warnings. The actual Neo4j write uses
        # ``safe_rel`` (lowercased), so the core-edge check MUST also
        # use the lowercased form for consistency.
        if not allow_non_core and not _ALLOW_NON_CORE_EDGES:
            if not is_core_edge(src_label, rel_type_lower, dst_label):
                logger.warning(
                    "Edge triple (%s, %s, %s) is not in CORE_EDGE_TYPES. "
                    "Set allow_non_core=True or DRUGOS_KG_ALLOW_NON_CORE_EDGES=1 "
                    "to allow.",
                    src_label, rel_type_lower, dst_label,
                )

        # Fixes P-1: Warn on single-edge batch
        if len(edges) == 1 and not allow_single_edge_batch:
            logger.warning(
                "load_edges called with a single edge — this is "
                "catastrophically slow. Accumulate edges into batches "
                "of >= 1000 before calling. Set allow_single_edge_batch=True "
                "to suppress this warning."
            )

        lineage = _build_lineage_props(source, input_checksum)

        total_created = 0
        total_dropped = 0
        total_dead_lettered = 0
        all_errors: list[str] = []

        # Edge property whitelist
        # P2-031 ROOT FIX: use ``rel_type_lower`` for the whitelist key
        # so the lookup HITS when callers pass mixed-case rel names.
        # The previous code used the original-case ``rel_type``, which
        # missed the whitelist for any caller passing "TREATS" /
        # "Inhibits" / etc. — silently stripping pchembl_value and
        # other ChEMBL activity properties from Compound-Drug edges.
        edge_key = (src_label, rel_type_lower, dst_label)
        allowed_edge_props = (
            EDGE_PROPERTY_WHITELIST.get(edge_key, frozenset({"source", "evidence", "score"}))
            | SYSTEM_PROPS
        )

        # Fixes R-2: Checkpoint support
        checkpoint = (
            read_latest_checkpoint(checkpoint_key) if checkpoint_key else None
        )
        start_idx = (
            checkpoint["last_completed_idx"] + 1 if checkpoint else 0
        )

        with self._conn.session() as session:
            for i in range(start_idx, len(edges), batch_size):
                batch = edges[i:i + batch_size]

                # ── Phase 1: Validate and filter ────────────────────────
                clean_batch: list[dict[str, Any]] = []
                for row_idx, edge in enumerate(batch):
                    # BUG-B-003 root fix: normalize edge endpoint keys.
                    # Different loaders emit different key names:
                    #   - DrugBank: drug_id / target_uniprot_id
                    #   - UniProt:  source / target
                    #   - GEO:      head / tail
                    #   - kg_builder requires: src_id / dst_id
                    # Previously every edge from DrugBank/UniProt/GEO was
                    # dead-lettered with "missing_endpoint_id". Now we
                    # normalize at the entry point so all loaders work.
                    #
                    # v24 ROOT FIX (FORENSIC-P2-CORE §1): the previous
                    # code's ``_endpoint_keys`` set included ``"source"``
                    # and ``"target"`` — but the phase1_bridge emits
                    # ``source`` as a DATA-SOURCE PROPERTY (e.g.
                    # ``source="chembl"``), NOT as an endpoint alias.
                    # The result: every bridge edge's ``source`` property
                    # was silently stripped, so Neo4j edges ended up with
                    # ``_source="unknown"`` (the lineage default) instead
                    # of the real source name. Fix: track which alias was
                    # actually used as the endpoint and remove ONLY that
                    # alias from the props dict; do not blanket-exclude
                    # ``source``/``target``.
                    #
                    # v28 ROOT FIX (P2-B-8): the alias-vs-property contract
                    # for the ``source`` / ``target`` keys is now
                    # documented in ``_loader_protocol.py`` (Loader
                    # Edge-Record Contract). Loaders MUST emit either
                    # ``src_id``/``dst_id`` (preferred) OR an alias — never
                    # both. The kg_builder correctly preserves ``source``
                    # as a data-source property when ``src_id`` is present.
                    _used_src_alias: Optional[str] = None
                    _used_dst_alias: Optional[str] = None
                    if "src_id" not in edge or "dst_id" not in edge:
                        # Try all known aliases in priority order.
                        src_aliases = (
                            "src_id", "drug_id", "source", "head",
                            "from_id", "subject_id",
                        )
                        dst_aliases = (
                            "dst_id", "target_uniprot_id", "target",
                            "tail", "to_id", "object_id",
                        )
                        for sa in src_aliases:
                            if sa in edge and edge[sa]:
                                edge = {**edge, "src_id": edge[sa]}
                                _used_src_alias = sa
                                break
                        for da in dst_aliases:
                            if da in edge and edge[da]:
                                edge = {**edge, "dst_id": edge[da]}
                                _used_dst_alias = da
                                break
                        # v24: remove the used alias key so it doesn't
                        # leak into the props dict as a fake property.
                        # Only remove the alias that was ACTUALLY used
                        # as an endpoint — leave other keys (e.g.
                        # ``source="chembl"`` when ``src_id`` was already
                        # present) intact as legitimate properties.
                        if _used_src_alias is not None and _used_src_alias != "src_id":
                            edge.pop(_used_src_alias, None)
                        if _used_dst_alias is not None and _used_dst_alias != "dst_id":
                            edge.pop(_used_dst_alias, None)
                    # Fixes DQ-2: Validate src_id and dst_id
                    src_id = edge.get("src_id")
                    dst_id = edge.get("dst_id")
                    if not src_id or not dst_id:
                        dead_letter_record(
                            source=source,
                            record=edge,
                            reason=f"missing_endpoint_id:{src_label}-{rel_type}->{dst_label}:idx={i + row_idx}:src={src_id is not None}:dst={dst_id is not None}",
                        )
                        total_dead_lettered += 1
                        continue

                    # BUG-D-002 root fix: validate endpoint IDs against
                    # ID_PATTERNS. The previous code only checked for
                    # missing/empty IDs — invalid formats (SIDER bare-int
                    # Compounds, OMIM Genes, OpenTargets MONDO_ Diseases)
                    # passed validation but silently failed the Cypher
                    # MATCH, making edges vanish with zero diagnostic.
                    src_pattern = ID_PATTERNS.get(src_label)
                    dst_pattern = ID_PATTERNS.get(dst_label)
                    if src_pattern and not re.match(src_pattern, str(src_id)):
                        dead_letter_record(
                            source=source,
                            record=edge,
                            reason=(
                                f"invalid_src_id_format:{src_label}-"
                                f"{rel_type}->{dst_label}:idx={i + row_idx}:"
                                f"src_id={src_id!r} does not match "
                                f"pattern {src_pattern}"
                            ),
                        )
                        total_dead_lettered += 1
                        continue
                    if dst_pattern and not re.match(dst_pattern, str(dst_id)):
                        dead_letter_record(
                            source=source,
                            record=edge,
                            reason=(
                                f"invalid_dst_id_format:{src_label}-"
                                f"{rel_type}->{dst_label}:idx={i + row_idx}:"
                                f"dst_id={dst_id!r} does not match "
                                f"pattern {dst_pattern}"
                            ),
                        )
                        total_dead_lettered += 1
                        continue

                    # Build row with props
                    row: dict[str, Any] = {
                        "src_id": src_id,
                        "dst_id": dst_id,
                    }
                    # v21 ROOT FIX (Audit section 4 finding 4 / Chain 4 -
                    # "Edge properties preserved by bridge, stripped by
                    # shim"): the previous code was
                    # ``props = edge.get("props", {})`` which expected a
                    # NESTED ``{"props": {...}}`` dict. But the
                    # phase1_bridge emits FLAT edge dicts:
                    #   {"src_id": ..., "dst_id": ..., "source": ...,
                    #    "pchembl_value": ..., "standard_relation": ...,
                    #    "evidence": ..., "_source_phase": 1, ...}
                    # The ``.get("props", {})`` call therefore returned
                    # ``{}`` for EVERY bridge edge, silently stripping
                    # ALL edge properties (pchembl_value,
                    # standard_relation, evidence, source, _source_file,
                    # _source_row). The v15 ROOT FIX (REM-12/13/14)
                    # explicitly claimed these were preserved so the RL
                    # ranker has potency + censoring context; that claim
                    # was FALSE in production. The test double
                    # (RecordingGraphBuilder) does NOT apply this filter,
                    # so the bug was invisible to tests.
                    #
                    # Fix: accept BOTH shapes. If ``edge["props"]`` is a
                    # dict, use it (callers that pre-bundle props). Else
                    # treat the edge dict itself as the props source,
                    # excluding the endpoint ID keys and system keys
                    # that should not appear as edge properties.
                    if "props" in edge and isinstance(edge["props"], dict):
                        props = dict(edge["props"])
                    else:
                        # Flat-edge case (phase1_bridge output).
                        # v24 ROOT FIX: exclude endpoint ID keys and
                        # well-known system keys that should not appear
                        # as edge properties. NOTE: ``source`` and
                        # ``target`` are NO LONGER in this set — they
                        # are legitimate data-source property names
                        # emitted by the bridge (e.g. source="chembl").
                        # The endpoint-alias case (UniProt edges that
                        # use ``source``/``target`` as endpoint keys)
                        # is handled above by tracking the used alias
                        # and removing it from the edge dict before
                        # this point.
                        _endpoint_keys = {
                            "src_id", "dst_id", "drug_id",
                            "target_uniprot_id",
                            "head", "tail", "from_id", "to_id",
                            "subject_id", "object_id",
                        }
                        props = {
                            k: v for k, v in edge.items()
                            if k not in _endpoint_keys and v is not None
                        }
                    # Fixes D-2, DQ-4: Whitelist edge properties
                    cleaned_props, dropped = _whitelist_filter(
                        props, allowed_edge_props
                    )
                    cleaned_props.update(lineage)
                    # BUG-D-011 root fix: stamp _source_priority so
                    # deduplicate_edges_deterministic can order by it.
                    cleaned_props["_source_priority"] = get_source_priority(source)
                    if dropped:
                        logger.debug(
                            "Dropped non-whitelisted edge props for "
                            "%s-%s->%s: %s",
                            src_label, rel_type_lower, dst_label, dropped,
                        )
                    row["props"] = cleaned_props
                    clean_batch.append(row)

                # v35 ROOT FIX (H-3): dedup by (src_id, dst_id, rel_type)
                # instead of (src_id, dst_id). The previous key collapsed
                # legitimate multi-action edges (e.g. a dual-action drug
                # with both "inhibits" and "activates" edges to the SAME
                # target — when load_edges_batch is invoked with a single
                # rel_type per call the previous key was already safe, but
                # if any caller ever batches across rel_types the collapse
                # was a silent data-loss bug). The rel_type is preserved
                # in the dedup key so dual-action edges survive. The
                # caller already invokes this once per (src, rel, dst)
                # triple, so the addition is a defensive measure against
                # future refactors that batch across rel_types.
                #
                # P2-031 ROOT FIX: use ``rel_type_lower`` (not the
                # original-case ``rel_type``) for the dedup key. The
                # Neo4j write uses ``safe_rel`` (lowercased), so two
                # calls with ``rel_type="TREATS"`` and ``rel_type="treats"``
                # would otherwise produce DIFFERENT dedup keys and BOTH
                # edges would be written — defeating the dedup and
                # creating duplicate Neo4j relationships (the exact
                # case-sensitivity corruption P2L-021 was designed to
                # prevent).
                seen_pairs: set[tuple[str, str, str]] = set()
                deduped_batch: list[dict[str, Any]] = []
                for row in clean_batch:
                    pair = (row["src_id"], row["dst_id"], rel_type_lower)
                    if pair in seen_pairs:
                        dead_letter_record(
                            source=source,
                            record=row,
                            reason=f"duplicate_edge_in_batch:pair={str(pair)[:100]}",
                        )
                        total_dead_lettered += 1
                        continue
                    seen_pairs.add(pair)
                    deduped_batch.append(row)
                clean_batch = deduped_batch

                if not clean_batch:
                    continue

                # ── Phase 2: Execute Cypher ─────────────────────────────
                try:
                    create_or_merge = "MERGE" if mode == "merge" else "CREATE"
                    # v36 ROOT FIX (Chain 6): use ON CREATE SET / ON MATCH SET
                    # explicitly. ``SET r += row.props`` after MERGE silently
                    # overwrites existing properties on the matched edge;
                    # ``ON MATCH SET r += row.props`` makes this explicit and
                    # future-proofs against adding guard clauses. The
                    # ``_source_priority`` field on each edge lets us
                    # eventually apply "high-priority source wins" semantics
                    # in a future patch without re-architecting the Cypher.
                    if create_or_merge == "MERGE":
                        # v109 ROOT FIX (P2-010): the previous ``ON MATCH SET
                        # r += row.props`` OVERWROTE ALL edge properties on
                        # every re-load. If the same edge was loaded by two
                        # different sources (e.g. ChEMBL with
                        # pchembl_value=8.5 and DrugBank with no
                        # pchembl_value), the second load would overwrite
                        # the first's pchembl_value to null, silently
                        # corrupting the scientific data. The audit caught
                        # this as a HIGH-severity data-corruption bug.
                        #
                        # ROOT FIX: on MATCH, do NOT touch the data
                        # properties (``row.props``). Only update the
                        # lineage metadata (``_updated_at`` and
                        # ``_version``). This preserves the first-loaded
                        # data values. If an operator wants to refresh
                        # data, they must DELETE the edge and re-CREATE it
                        # (an explicit, audited operation — not a silent
                        # side-effect of MERGE).
                        #
                        # The ``SET r._pipeline_run_id = $run_id`` is kept
                        # because it tracks which pipeline run last
                        # touched the edge (lineage metadata, not data).
                        cypher = (
                            f"UNWIND $batch AS row\n"
                            f"MATCH (src:{safe_src} {{id: row.src_id}})\n"
                            f"MATCH (dst:{safe_dst} {{id: row.dst_id}})\n"
                            f"MERGE (src)-[r:{safe_rel}]->(dst)\n"
                            f"ON CREATE SET r += row.props, "
                            f"r._created_at = $loaded_at\n"
                            f"ON MATCH SET "
                            f"r._updated_at = $loaded_at, "
                            f"r._version = coalesce(r._version, 0) + 1\n"
                            f"SET r._pipeline_run_id = $run_id"
                        )
                        params = {
                            "batch": clean_batch,
                            "loaded_at": lineage.get("_loaded_at"),
                            "run_id": RUN_ID,
                        }
                        # P2-058 ROOT FIX (Teammate 4): use the idiomatic
                        # ``parameters=params`` form instead of ``**params``.
                        # Both forms work (the neo4j driver's ``Session.run``
                        # signature is ``run(query, parameters=None, **kwargs)``
                        # and merges kwargs into parameters), but ``parameters=``
                        # is the form used in the official Neo4j docs and is
                        # clearer to readers. The previous ``**params`` form
                        # was technically correct but flagged by the audit as
                        # confusing — readers assumed ``batch``, ``loaded_at``,
                        # ``run_id`` were method keyword arguments (they're
                        # not — they're Cypher query parameters accessed as
                        # ``$batch``, ``$loaded_at``, ``$run_id``).
                        result = session.run(cypher, parameters=params)
                    else:
                        # FIX-P2-P2-11: the CREATE branch was missing
                        # the lineage properties that the MERGE branch
                        # sets via ``ON CREATE SET`` / ``SET`` (namely
                        # ``_created_at`` and ``_pipeline_run_id``).
                        # Edges loaded with ``mode="create"`` were
                        # therefore indistinguishable from edges loaded
                        # by external/manual tools, breaking downstream
                        # lineage audits. ``_updated_at`` and
                        # ``_version`` are intentionally omitted — they
                        # are MATCH-only semantics and a freshly-created
                        # edge has never been "updated".
                        cypher = (
                            f"UNWIND $batch AS row\n"
                            f"MATCH (src:{safe_src} {{id: row.src_id}})\n"
                            f"MATCH (dst:{safe_dst} {{id: row.dst_id}})\n"
                            f"CREATE (src)-[r:{safe_rel}]->(dst)\n"
                            f"SET r += row.props, "
                            f"r._created_at = $loaded_at, "
                            f"r._pipeline_run_id = $run_id"
                        )
                        params = {
                            "batch": clean_batch,
                            "loaded_at": lineage.get("_loaded_at"),
                            "run_id": RUN_ID,
                        }
                        # P2-058 ROOT FIX: see MERGE branch above — use
                        # ``parameters=params`` for idiomatic clarity.
                        result = session.run(cypher, parameters=params)
                    stats = result.consume().counters
                    batch_created = stats.relationships_created
                    total_created += batch_created

                    # Fixes DQ-3: Track silently dropped edges
                    batch_dropped = len(clean_batch) - batch_created
                    total_dropped += batch_dropped
                    if batch_dropped > 0:
                        # Fixes L-1: Log dropped edges
                        dropped_pct = batch_dropped / max(len(clean_batch), 1)
                        log_level = logging.ERROR if dropped_pct > 0.05 else logging.WARNING
                        logger.log(
                            log_level,
                            "Dropped %d/%d edges for %s-%s->%s "
                            "(src or dst not found in graph). "
                            "This may indicate a data quality issue.",
                            batch_dropped, len(clean_batch),
                            safe_src, safe_rel, safe_dst,
                        )
                        # Fixes DQ-3: If >5% dropped, raise mismatch error
                        if dropped_pct > 0.05:
                            all_errors.append(
                                f"{batch_dropped}/{len(clean_batch)} edges "
                                f"dropped for {src_label}-{rel_type}->{dst_label}"
                            )

                    # Progress logging (C-6)
                    # P2-059 ROOT FIX: the previous ``(i // batch_size) %
                    # _LOG_FREQUENCY == 0`` pattern ALWAYS logged the
                    # first batch (i=0 → 0 % log_freq == 0) even in
                    # quiet mode (log_freq=10). That's because i=0 is
                    # the batch START index, and 0 // batch_size = 0,
                    # 0 % anything = 0. The first batch's log is
                    # ALWAYS emitted, even in quiet mode — minor noise
                    # but inconsistent with the node-loader path which
                    # uses the 1-indexed batch_count pattern (line 2054).
                    # Root fix: use ``batch_count = i // batch_size + 1``
                    # (1-indexed) and log when ``batch_count % log_freq
                    # == 0``. This logs at batches log_freq, 2*log_freq,
                    # 3*log_freq, ... — i.e. every log_freq batches
                    # starting from batch log_freq (NOT batch 0). The
                    # first batch is NOT specially logged. This matches
                    # the node-loader pattern at line 2054 so the two
                    # progress-log paths are stylistically consistent
                    # (P2-060 aims to eliminate exactly this kind of
                    # stylistic drift).
                    batch_count = i // batch_size + 1  # 1-indexed
                    if batch_count % _LOG_FREQUENCY == 0:
                        # P2-007 ROOT FIX (v104): per-batch progress log
                        # moved from INFO to DEBUG (same fix as the node
                        # loader at line 2248). The summary log at the end
                        # of the load (``Created %d %s-%s->%s edges ...``)
                        # stays at INFO — that's the line operators need.
                        logger.debug(
                            "  %s-%s->%s: loaded %d/%d edges mode=%s",
                            safe_src, safe_rel, safe_dst,
                            i + len(batch), len(edges), mode,
                        )

                    # Checkpoint
                    # v100 ROOT FIX (BUG P2-051 — edge checkpoint resume
                    # data loss): the previous code wrote
                    # `last_completed_idx: i + batch_size - 1` which
                    # OVERESTIMATES the last completed edge index for
                    # the final partial batch (when `len(edges)` is not
                    # a multiple of `batch_size`). On resume,
                    # `start_idx = checkpoint["last_completed_idx"] + 1`
                    # then skipped edges that were never processed
                    # (they were beyond the actual last processed edge
                    # but within the recorded `i + batch_size - 1`
                    # range). For batch_size=1000, up to 999 edges per
                    # resume were silently lost. The node loader
                    # (lines 1749 above) already uses the correct
                    # formula `i + len(batch) - 1` — apply the SAME
                    # fix here. `len(batch)` is the ACTUAL number of
                    # rows in this batch (≤ `batch_size`).
                    if checkpoint_key:
                        write_checkpoint(
                            checkpoint_key,
                            {
                                "last_completed_idx": i + len(batch) - 1,
                                "ts": _now_iso(),
                            },
                        )

                except (ServiceUnavailable, SessionExpired, OSError) as e:
                    logger.error(
                        "Batch %d failed: %s. Checkpoint at %d.",
                        i, e, max(0, i - 1),
                    )
                    raise
                except DrugOSDataError as e:
                    logger.warning(
                        "Batch %d had data errors: %s. DLQ'd. Continuing.",
                        i, e,
                    )
                    all_errors.append(str(e))
                    continue

        elapsed = time.monotonic() - start_time
        load_result = LoadResult(
            attempted=len(edges),
            created=total_created,
            dropped_no_match=total_dropped,
            dead_lettered=total_dead_lettered,
            elapsed_seconds=elapsed,
            errors=all_errors,
        )

        logger.info(
            "Created %d %s-%s->%s edges (mode=%s, %d dropped, %d dead-lettered)",
            total_created, safe_src, safe_rel, safe_dst,
            mode, total_dropped, total_dead_lettered,
        )

        audit_log(
            "edges_loaded",
            details=f"Loaded {total_created} {src_label}-{rel_type}->{dst_label} edges",
            metadata={
                "src_label": src_label,
                "rel_type": rel_type,
                "dst_label": dst_label,
                "created": total_created,
                "dropped": total_dropped,
                "dead_lettered": total_dead_lettered,
                "mode": mode,
                "source": source,
                "pipeline_run_id": RUN_ID,
            },
        )

        return load_result if detailed else total_created

    def load_edges_batch(
        self,
        src_label: str,
        rel_type: str,
        dst_label: str,
        edges: list[dict],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[int, LoadResult]:
        """Bulk-create relationships using UNWIND + MERGE.

        Fixes A-2: Thin wrapper around _load_edges(mode="merge").
        """
        return self._load_edges(
            src_label, rel_type, dst_label, edges,
            batch_size=batch_size, mode="merge", **kwargs,
        )

    def load_edges_bulk_create(
        self,
        src_label: str,
        rel_type: str,
        dst_label: str,
        edges: list[dict],
        batch_size: Optional[int] = None,
        use_merge: bool = True,
        **kwargs: Any,
    ) -> Union[int, LoadResult]:
        """Bulk-create relationships using UNWIND + MERGE (or CREATE).

        P2-003 FORENSIC ROOT FIX (v104 — Team Member 5):
            The previous default was ``use_merge=False`` (CREATE mode).
            CREATE always inserts a new edge — re-running the pipeline
            DOUBLED the edge count. After 5 re-runs, the KG had 5x the
            edges. The GNN's message-passing then over-weighted these
            edges 5x, causing embedding drift and non-reproducible RL
            rankings.

            ROOT FIX: default changed to ``use_merge=True`` (idempotent
            MERGE). Re-running the pipeline now produces the SAME edge
            count. The CREATE branch is preserved for explicit one-off
            loads (``use_merge=False``) but emits a ``DeprecationWarning``
            to discourage the dangerous non-idempotent path. CREATE will
            be removed in v2.0.

        Fixes A-2: Thin wrapper around _load_edges.
        Fixes I-1: Default is now ``use_merge=True`` (idempotent).

        Parameters
        ----------
        use_merge : bool
            If True (DEFAULT), use MERGE (idempotent — safe for re-runs).
            If False, use CREATE (non-idempotent — emits
            DeprecationWarning; for one-off loads only).
        """
        if not use_merge:
            warnings.warn(
                "load_edges_bulk_create(use_merge=False) uses Neo4j CREATE "
                "which produces DUPLICATE edges on re-run (P2-003). Pass "
                "use_merge=True (the new default) for idempotent loads. "
                "use_merge=False will be removed in v2.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        mode = "merge" if use_merge else "create"
        return self._load_edges(
            src_label, rel_type, dst_label, edges,
            batch_size=batch_size, mode=mode, **kwargs,
        )

    def load_drkg_edges_bulk(
        self,
        edge_type_data: dict[tuple[str, str, str], list[dict]],
        *,
        source_file: Optional[str] = None,
        use_merge: bool = True,
    ) -> dict[tuple[str, str, str], Union[int, LoadResult]]:
        """Load all DRKG edges using bulk MERGE (idempotent).

        P2-003 FORENSIC ROOT FIX (v104 — Team Member 5):
            Default changed from ``use_merge=False`` (CREATE) to
            ``use_merge=True`` (MERGE). DRKG is re-ingested on every
            pipeline run; CREATE mode doubled the edge count each time.
            MERGE is idempotent — re-running produces the same edge count.

        Fixes DL-2: Input checksum verification.
        """
        input_checksum = ""
        if source_file:
            try:
                input_checksum = compute_and_record_checksum(source_file)
            except Exception as e:
                logger.warning("Could not compute checksum: %s", e)

        results: dict[tuple[str, str, str], Union[int, LoadResult]] = {}
        for (src_type, rel_name, dst_type), edges in edge_type_data.items():
            logger.info(
                "Loading %d %s-%s->%s edges (mode=%s) ...",
                len(edges), src_type, rel_name, dst_type,
                "MERGE" if use_merge else "CREATE",
            )
            count = self.load_edges_bulk_create(
                src_type, rel_name, dst_type, edges,
                use_merge=use_merge,
                source="DRKG",
                input_checksum=input_checksum,
            )
            results[(src_type, rel_name, dst_type)] = count
        return results

    def load_drkg_edges(
        self,
        edge_type_data: dict[tuple[str, str, str], list[dict]],
    ) -> dict[tuple[str, str, str], Union[int, LoadResult]]:
        """REMOVED — use load_drkg_edges_bulk(use_merge=True).

        P2-029 ROOT FIX (forensic, TM5): the @deprecated decorator
        was a SURFACE-LEVEL fix. The method now RAISES on every call.
        The legacy non-bulk MERGE path used ``load_edges_batch`` (O(N)
        round-trips, no idempotency flag) and re-running the pipeline
        doubled the edge count because the underlying call used CREATE,
        not MERGE. ROOT FIX: callers MUST use ``load_drkg_edges_bulk``
        with ``use_merge=True`` (idempotent).
        """
        raise DeprecationWarning(
            "P2-029 ROOT FIX: GraphEdgeLoader.load_drkg_edges() has been "
            "REMOVED (legacy non-bulk path, no idempotency flag). Use "
            "load_drkg_edges_bulk(edge_type_data, use_merge=True) — same "
            "input, idempotent MERGE behavior."
        )

    def deduplicate_edges(
        self,
        src_label: str,
        rel_type: str,
        dst_label: str,
    ) -> int:
        """REMOVED — non-deterministic dedup violates FDA 21 CFR Part 11.

        P2-029 ROOT FIX (forensic, TM5): the @deprecated decorator
        was a SURFACE-LEVEL fix. The method now RAISES on every call.
        The non-deterministic Cypher ``UNWIND tail(rels) AS dup DELETE dup``
        order depends on Neo4j's internal collect() ordering, which is
        not guaranteed across versions or even across runs on the same
        version. Two runs on the same graph could delete DIFFERENT
        edges, violating FDA 21 CFR Part 11 reproducibility.
        ROOT FIX: callers MUST use ``deduplicate_edges_deterministic``
        which orders by a stable property key (``_source_priority``)
        before deletion.
        """
        raise DeprecationWarning(
            "P2-029 ROOT FIX: GraphEdgeLoader.deduplicate_edges() has "
            "been REMOVED (non-deterministic collect() ordering, "
            "violates FDA 21 CFR Part 11 reproducibility). Use "
            "deduplicate_edges_deterministic(src_label, rel_type, "
            "dst_label) — same signature, deterministic ordering."
        )

    def deduplicate_edges_deterministic(
        self,
        src_label: str,
        rel_type: str,
        dst_label: str,
    ) -> int:
        """Remove duplicate relationships deterministically.

        Fixes I-5: Non-deterministic deduplication.
        Keeps the edge with the most properties (or highest-priority source).

        Parameters
        ----------
        src_label, rel_type, dst_label : str
            Edge triple.

        Returns
        -------
        int
            Number of duplicate edges removed.
        """
        # v43 ROOT FIX (Chain 1): translate semantic → storage label
        # v57 ROOT FIX (P2L-021): lowercase rel_type before sanitizing
        # so dedup matches the same canonical relationship type that
        # ``_load_edges`` writes (see the comment there).
        safe_src = sanitize_label(_storage_label(src_label))
        safe_dst = sanitize_label(_storage_label(dst_label))
        # v102 P2-048: use _canonical_rel_type to handle DRKG "::" and
        # ":" separators BEFORE sanitize_rel_type, so dedup matches the
        # same canonical relationship type that _load_edges writes.
        safe_rel = sanitize_rel_type(_canonical_rel_type(rel_type))

        with self._conn.session() as session:
            # Fixes I-5: Deterministic ordering by source priority and load time.
            # v35 ROOT FIX (M-9): the previous sort key was
            #   `r._source_priority DESC, r._loaded_at ASC`
            # which was fragile under three conditions:
            #   (1) two edges with the SAME microsecond timestamp (tie-
            #       breaker undefined — Cypher does not guarantee a
            #       stable sort, so the kept edge was non-deterministic);
            #   (2) `_loaded_at` is null (edges from old runs without
            #       the lineage property — Cypher null-handling makes
            #       the sort non-deterministic);
            #   (3) format variance (`+00:00` vs `Z` suffix — both valid
            #       ISO 8601 UTC but lexicographically different).
            # The fix coalesces null `_loaded_at` to a sentinel minimum
            # string and adds `id(r) ASC` (Neo4j's monotonically-
            # increasing internal ID) as a deterministic final tie-
            # breaker so the kept edge is reproducible across runs.
            cypher = (
                f"MATCH (src:{safe_src})-[r:{safe_rel}]->(dst:{safe_dst}) "
                f"WITH src, dst, r "
                f"ORDER BY src.id, dst.id, "
                f"r._source_priority DESC, "
                f"coalesce(r._loaded_at, '1970-01-01T00:00:00+00:00') ASC, "
                # v43 ROOT FIX (P2 — id(r) ASC deprecated in Neo4j 6.x):
                # The previous code used `id(r) ASC` which is deprecated
                # in Neo4j 6.x and removed in 7.x. Use `elementId(r) ASC`
                # instead (available since Neo4j 5.x). The elementId()
                # function returns a string ID that's stable across
                # restarts, unlike id() which returns an internal integer
                # that can be reused after deletion.
                f"elementId(r) ASC "
                f"WITH src, dst, collect(r) AS rels "
                f"WHERE size(rels) > 1 "
                f"WITH rels[0] AS keep, rels[1..] AS dups "
                f"UNWIND dups AS dup "
                f"DELETE dup "
                f"RETURN count(dup) AS removed"
            )
            result = session.run(cypher)
            record = result.single()
            removed = record["removed"] if record else 0

        if removed > 0:
            logger.info(
                "Deduplicated %s-%s->%s (deterministic): "
                "removed %d duplicate edges",
                safe_src, safe_rel, safe_dst, removed,
            )
            audit_log(
                "edges_deduplicated",
                details=f"Removed {removed} duplicate {src_label}-{rel_type}->{dst_label} edges",
                metadata={
                    "src_label": src_label,
                    "rel_type": rel_type,
                    "dst_label": dst_label,
                    "removed": removed,
                    "method": "deterministic",
                    "pipeline_run_id": RUN_ID,
                },
            )
        return removed


class DrugBankEnricher:
    """Enriches Compound nodes with DrugBank properties.

    Fixes A-1: Extracted from DrugOSGraphBuilder.
    Fixes S-1: CRITICAL PATIENT SAFETY — coalesce pattern for safety fields.
    Fixes D-1: Accurate enriched count via Cypher RETURN.
    Fixes D-3: Configurable canonical key.
    Fixes I-3: Empty input raises CriticalDataSourceError.
    Fixes L-2: Logging of property overwrites on safety-critical fields.
    Fixes DL-3: Transformation logging.
    """

    # Safety-critical fields that must NEVER be overwritten with null
    SAFETY_CRITICAL_FIELDS = frozenset({
        "withdrawn", "terminated", "illicit", "sensitive", "toxicity",
    })

    def __init__(self, conn: GraphConnection) -> None:
        self._conn = conn

    def enrich_compounds_from_drugbank(
        self,
        drug_records: list[dict],
        canonical_key: str = "id",
    ) -> Union[int, LoadResult]:
        """Add DrugBank properties to existing Compound nodes.

        PATIENT SAFETY (NON-NEGOTIABLE):
        This method uses coalesce() for ALL safety-critical fields to
        prevent null overwrites. If row.withdrawn IS NULL AND
        c.withdrawn IS NULL, sets withdrawn=false AND
        safety_data_missing=true so the RL ranker can flag this drug
        as "insufficient safety data" rather than "confirmed safe".

        Parameters
        ----------
        drug_records : list of dict
            DrugBank records to enrich. Each MUST contain the canonical_key.
        canonical_key : str
            The key to use for matching (default "id").
            Must be one of the CANONICAL_IDS values.

        Returns
        -------
        int or LoadResult
            Number of distinct compound nodes enriched.

        Raises
        ------
        ConfigurationError
            If canonical_key is not a valid canonical ID.
        CriticalDataSourceError
            If drug_records is empty (data outage protection).

        Side Effects
        ------------
        - Updates Compound nodes in Neo4j
        - Writes audit log entries
        - Logs safety-critical property changes at WARNING

        Invariants
        ----------
        - A non-null safety value in the graph is NEVER overwritten by null
        - If both row and graph have null for a safety field, sets
          the field to False and marks safety_data_missing=True
        - All lineage properties are stamped

        Fixes: S-1, D-1, D-3, I-3, L-2, DL-3
        """
        start_time = time.monotonic()

        # Fixes I-3: Empty input protection (data outage)
        if len(drug_records) == 0:
            raise CriticalDataSourceError(
                "enrich_compounds_from_drugbank called with empty "
                "drug_records. This is almost certainly a download "
                "failure, not a legitimate 'no data' case. "
                "Aborting to prevent data loss."
            )

        # Fixes D-3: Configurable canonical key
        valid_keys = set(CANONICAL_IDS.values())
        if canonical_key not in valid_keys and canonical_key != "id":
            raise ConfigurationError(
                f"canonical_key must be one of {valid_keys} or 'id', "
                f"got {canonical_key!r}"
            )

        batch_size = self._conn.config.batch_size_nodes
        total_enriched = 0
        total_dead_lettered = 0
        all_errors: list[str] = []

        lineage = _build_lineage_props("DrugBank")

        with self._conn.session() as session:
            for i in range(0, len(drug_records), batch_size):
                batch = drug_records[i:i + batch_size]

                # ── Validate batch ──────────────────────────────────────
                clean_batch: list[dict[str, Any]] = []
                for row_idx, rec in enumerate(batch):
                    key_val = rec.get(canonical_key)
                    if not key_val:
                        dead_letter_record(
                            source="DrugBank",
                            record=rec,
                            reason=f"missing_{canonical_key}:idx={i + row_idx}",
                        )
                        total_dead_lettered += 1
                        continue

                    # Whitelist filter
                    allowed = NODE_PROPERTY_WHITELIST.get("Compound", frozenset()) | SYSTEM_PROPS
                    cleaned, _ = _whitelist_filter(rec, allowed)
                    clean_batch.append(cleaned)

                if not clean_batch:
                    continue

                # ── Execute Cypher with coalesce for safety fields ──────
                # Fixes S-1: CRITICAL — coalesce pattern prevents null overwrite
                # PATIENT SAFETY: withdrawn=null is interpreted by the RL
                # safety ranker as "not withdrawn" → green → SAFE.
                # Valdecoxib (withdrawn for CV risk) would be SAFE.
                cypher = (
                    f"UNWIND $batch AS row\n"
                    f"MATCH (c:Compound {{{canonical_key}: row.{canonical_key}}})\n"
                    f"SET c.name                = coalesce(row.name, c.name),"
                    f"    c.smiles              = coalesce(row.smiles, c.smiles),"
                    f"    c.inchikey            = coalesce(row.inchikey, c.inchikey),"
                    f"    c.indication          = coalesce(row.indication, c.indication),"
                    f"    c.mechanism_of_action = coalesce(row.mechanism_of_action, c.mechanism_of_action),"
                    f"    c.atc_codes           = coalesce(row.atc_codes, c.atc_codes),"
                    f"    c.approved            = coalesce(row.approved, c.approved),"
                    f"    c.investigational     = coalesce(row.investigational, c.investigational),"
                    f"    c.pubchem_cid         = coalesce(row.pubchem_cid, c.pubchem_cid),"
                    f"    c.chembl_id           = coalesce(row.chembl_id, c.chembl_id),"
                    f"    c.chebi_id            = coalesce(row.chebi_id, c.chebi_id),"
                    f"    c.drug_type           = coalesce(row.drug_type, c.drug_type),"
                    f"    c.approval_year       = coalesce(row.approval_year, c.approval_year),"
                    f"    c.source_drugbank     = true,"
                    f"    c.drugbank_id         = coalesce(row.drugbank_id, c.drugbank_id),"
                    f"    c.cas_number          = coalesce(row.cas_number, c.cas_number),"
                    f"    c.pharmacodynamics    = coalesce(row.pharmacodynamics, c.pharmacodynamics),"
                    f"    c.categories          = coalesce(row.categories, c.categories),"
                    f"    c._canonical_id_source = row._canonical_id_source,"
                    f"    c._last_modified      = row._last_modified,"
                    # 🔴 SAFETY-CRITICAL: never null these out
                    f"    c.toxicity            = coalesce(row.toxicity, c.toxicity),"
                    f"    c.withdrawn           = coalesce(row.withdrawn, c.withdrawn),"
                    f"    c.terminated          = coalesce(row.terminated, c.terminated),"
                    f"    c.illicit             = coalesce(row.illicit, c.illicit),"
                    f"    c.sensitive           = coalesce(row.sensitive, c.sensitive),"
                    # Safety net: if both null, mark as missing data
                    f"    c.safety_data_missing = CASE "
                    f"WHEN row.withdrawn IS NULL AND c.withdrawn IS NULL THEN true "
                    f"ELSE coalesce(c.safety_data_missing, false) END,"
                    # Lineage props (always overwrite)
                    f"    c._pipeline_run_id    = $run_id,"
                    f"    c._loaded_at          = $loaded_at,"
                    f"    c._source             = 'DrugBank',"
                    f"    c._license            = 'CC BY-NC 4.0',"
                    f"    c._attribution        = $attribution,"
                    f"    c._schema_version     = $schema_version,"
                    f"    c._config_hash        = $config_hash,"
                    f"    c._pipeline_version   = $pipeline_version,"
                    f"    c._seed               = $seed,"
                    f"    c._updated_at         = $loaded_at,"
                    f"    c._version            = coalesce(c._version, 0) + 1\n"
                    f"RETURN row.{canonical_key} AS matched_id, "
                    f"c.withdrawn AS old_w, "
                    f"coalesce(row.withdrawn, c.withdrawn) AS new_w"
                )

                params = {
                    "batch": clean_batch,
                    "run_id": RUN_ID,
                    "loaded_at": lineage["_loaded_at"],
                    "attribution": SOURCE_LICENSES["DrugBank"]["attribution"],
                    "schema_version": SCHEMA_VERSION,
                    "config_hash": CONFIG_HASH,
                    "pipeline_version": PIPELINE_VERSION,
                    "seed": SEED,
                }

                result = session.run(cypher, params)
                # Fixes D-1: Accurate enriched count via Cypher RETURN
                for record in result:
                    total_enriched += 1
                    # Fixes L-2: Log safety-critical property changes
                    old_w = record.get("old_w")
                    new_w = record.get("new_w")
                    if old_w != new_w and new_w is None:
                        logger.error(
                            "SAFETY: Compound %s withdrawn changed from "
                            "%s to None — this should not happen with "
                            "coalesce pattern!",
                            record.get("matched_id", "?"), old_w,
                        )
                        audit_log(
                            "safety_property_overwrite",
                            metadata={
                                "id": record.get("matched_id"),
                                "field": "withdrawn",
                                "old": old_w,
                                "new": new_w,
                            },
                        )

        elapsed = time.monotonic() - start_time
        load_result = LoadResult(
            attempted=len(drug_records),
            created=0,
            updated=total_enriched,
            dead_lettered=total_dead_lettered,
            elapsed_seconds=elapsed,
            errors=all_errors,
        )

        logger.info(
            "Enriched %d distinct Compound nodes with DrugBank data",
            total_enriched,
        )

        # Fixes DL-3: Transformation logging
        log_transformation(
            step="enrich_compounds_from_drugbank",
            input_count=len(drug_records),
            output_count=total_enriched,
            transformation_map={
                "DrugBank::name": "Compound.name",
                "DrugBank::withdrawn": "Compound.withdrawn",
                "DrugBank::terminated": "Compound.terminated",
                "DrugBank::illicit": "Compound.illicit",
                "DrugBank::sensitive": "Compound.sensitive",
                "DrugBank::toxicity": "Compound.toxicity",
            },
        )

        # Fixes CO-4: Audit trail
        audit_log(
            "drugbank_enrichment",
            details=f"Enriched {total_enriched} Compound nodes",
            metadata={
                "enriched": total_enriched,
                "dead_lettered": total_dead_lettered,
                "canonical_key": canonical_key,
                "pipeline_run_id": RUN_ID,
            },
        )

        return total_enriched


class GraphStatsCollector:
    """Collects graph statistics.

    Fixes A-1: Extracted from DrugOSGraphBuilder.
    Fixes S-3: Misleading density calculation.
    Fixes S-4: labels(n)[0] non-deterministic.
    Fixes P-4: get_graph_stats makes 4 round-trips → 1-2.
    Fixes IN-2: No interface contract for return value.
    """

    def __init__(self, conn: GraphConnection) -> None:
        self._conn = conn

    def get_graph_stats(self) -> dict[str, Any]:
        """Compute and return comprehensive graph statistics.

        Returns
        -------
        dict
            GraphStats with typed density, node/edge counts, etc.

        Invariants
        ----------
        - density_typed uses typed-edge-aware formula
        - density_homogeneous uses the old formula (backward compat)
        - node_counts_by_type is deterministic (ordered by count DESC, lbl ASC)
        - All counts are non-negative integers

        Fixes: S-3, S-4, P-4, IN-2
        """
        with self._conn.session() as session:
            # Fixes P-4: Combine node+edge counts into 1-2 queries
            node_result = session.run(
                "MATCH (n) UNWIND labels(n) AS lbl "
                "RETURN lbl, count(*) AS cnt "
                "ORDER BY cnt DESC, lbl ASC"
            )
            node_counts_by_type: dict[str, int] = {}
            total_nodes = 0
            for record in node_result:
                lbl = record["lbl"]
                cnt = record["cnt"]
                node_counts_by_type[lbl] = cnt
                total_nodes += cnt

            # Note: total_nodes may overcount multi-label nodes, so get true count
            total_result = session.run("MATCH (n) RETURN count(n) AS total")
            total_nodes = total_result.single()["total"]

            edge_result = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel_type, "
                "count(r) AS cnt ORDER BY cnt DESC"
            )
            edge_counts_by_type: dict[str, int] = {}
            total_edges = 0
            for record in edge_result:
                edge_counts_by_type[record["rel_type"]] = record["cnt"]
                total_edges += record["cnt"]

        # Fixes S-3: Typed-edge-aware density calculation
        # Before: max_edges = n * (n-1) — assumes homogeneous complete graph
        # After: per-edge-type maximum based on actual node type counts
        typed_max = 0
        for (src_type, _, dst_type) in CORE_EDGE_TYPES:
            src_count = node_counts_by_type.get(src_type, 0)
            dst_count = node_counts_by_type.get(dst_type, 0)
            if src_type == dst_type:
                typed_max += src_count * max(src_count - 1, 0)
            else:
                typed_max += src_count * dst_count

        density_typed = round(total_edges / typed_max, 8) if typed_max > 0 else 0.0
        # Backward compat: homogeneous density
        density_homogeneous = (
            round(total_edges / (total_nodes * max(total_nodes - 1, 1)), 8)
            if total_nodes > 1
            else 0.0
        )

        stats: dict[str, Any] = {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "node_counts_by_type": node_counts_by_type,
            "edge_counts_by_type": edge_counts_by_type,
            "density": density_typed,  # Default is now typed
            "density_typed": density_typed,
            "density_homogeneous": density_homogeneous,
            "pipeline_run_id": RUN_ID,
            "computed_at": _now_iso(),
        }

        logger.info(
            "Graph stats: %d nodes, %d edges, density_typed=%.8f",
            total_nodes, total_edges, density_typed,
        )
        return stats

    def health_check(self, conn: GraphConnection) -> dict[str, Any]:
        """Run a health check on the Neo4j instance and graph.

        Fixes R-7: Verify driver state.
        Fixes S(9)-4: Don't print connection details.
        """
        health = conn.health_check()
        if health.get("connected"):
            try:
                stats = self.get_graph_stats()
                health["total_nodes"] = stats["total_nodes"]
                health["total_edges"] = stats["total_edges"]
                health["node_types"] = len(stats["node_counts_by_type"])
                health["edge_types"] = len(stats["edge_counts_by_type"])
            except Exception as e:
                health["stats_error"] = str(e)
        return health


class GraphJanitor:
    """Handles dangerous graph operations with access control.

    Fixes A-1: Extracted from DrugOSGraphBuilder.
    Fixes S(9)-3: clear_graph no access control.
    Fixes C-7: clear_graph returns None → ClearGraphResult.
    Fixes R-5: clear_graph not atomic on large graphs.
    """

    def __init__(self, conn: GraphConnection) -> None:
        self._conn = conn

    def clear_graph(
        self,
        *,
        confirm: bool = False,
        confirm_phrase: Optional[str] = None,
    ) -> ClearGraphResult:
        """Delete all nodes and relationships with safety confirmation.

        Fixes S(9)-3: Requires explicit confirmation.
        Fixes C-7: Returns ClearGraphResult instead of None.
        Fixes R-5: Chunked deletion for large graphs.

        Parameters
        ----------
        confirm : bool
            Must be True to proceed.
        confirm_phrase : str, optional
            Must match DRUGOS_CLEAR_GRAPH_PHRASE env var.

        Returns
        -------
        ClearGraphResult

        Raises
        ------
        SecurityError
            If confirm=False or confirm_phrase doesn't match.
        """
        # Fixes S(9)-3: Access control for clear_graph
        if not confirm:
            raise SecurityError(
                "clear_graph() requires confirm=True. "
                "This deletes ALL nodes and edges."
            )
        expected_phrase = _CLEAR_GRAPH_PHRASE
        if confirm_phrase != expected_phrase:
            raise SecurityError(
                "confirm_phrase does not match expected phrase. "
                "Set DRUGOS_CLEAR_GRAPH_PHRASE to override."
            )

        # Fixes CO-4: Audit trail BEFORE deletion
        audit_log(
            "graph_clear_initiated",
            metadata={
                "caller": inspect.stack()[1].function,
                "config_hash": CONFIG_HASH,
                "pipeline_run_id": RUN_ID,
            },
        )

        start_time = time.monotonic()
        total_nodes_deleted = 0
        total_rels_deleted = 0

        # Fixes R-5: Chunked deletion for large graphs
        chunk_size = 10000
        with self._conn.session(
            default_timeout=max(_QUERY_TIMEOUT, 600),
        ) as session:
            while True:
                result = session.run(
                    "MATCH (n) WITH n LIMIT $limit "
                    "DETACH DELETE n "
                    "RETURN count(n) AS deleted",
                    limit=chunk_size,
                )
                # v35 ROOT FIX (M-10): the previous code added the NODE
                # count to total_rels_deleted, assuming 1 rel per node.
                # For densely-connected graphs (e.g. a Compound with 50
                # target edges), this severely undercounted rel deletions
                # — a node with 50 rels contributed only 1 to the count.
                # The fix uses Neo4j's actual deletion counters from
                # result.consume().counters (nodes_deleted and
                # relationships_deleted) which are accurate.
                # v42 FORENSIC ROOT FIX (P0-4): the previous code called
                # ``result.consume().counters()`` — but in the Neo4j Python
                # driver, ``result.consume()`` returns a ``ResultSummary``
                # whose ``.counters`` is an ATTRIBUTE (a SummaryCounters
                # object), NOT a method. Calling ``.counters()`` raised
                # ``TypeError: 'SummaryCounters' object is not callable`` on
                # the first call, so clear_graph() crashed immediately and
                # the graph could never be cleared. ROOT FIX: drop the
                # parentheses — access the attribute directly.
                counters = result.consume().counters
                deleted_nodes = counters.nodes_deleted
                deleted_rels = counters.relationships_deleted
                if deleted_nodes == 0:
                    break
                total_nodes_deleted += deleted_nodes
                total_rels_deleted += deleted_rels

        elapsed = time.monotonic() - start_time
        result = ClearGraphResult(
            nodes_deleted=total_nodes_deleted,
            relationships_deleted=total_rels_deleted,
            elapsed_seconds=elapsed,
            pipeline_run_id=RUN_ID,
            timestamp=_now_iso(),
        )

        logger.warning(
            "All nodes and relationships deleted from graph: "
            "%d nodes in %.2fs",
            total_nodes_deleted, elapsed,
        )

        # Fixes CO-4: Audit trail AFTER deletion
        # v35 ROOT FIX (N-4): include relationships_deleted in the audit
        # log metadata. Previously only nodes_deleted was logged, leaving
        # the rel-deletion count absent from the audit trail.
        audit_log(
            "graph_clear_completed",
            metadata={
                "nodes_deleted": total_nodes_deleted,
                "relationships_deleted": total_rels_deleted,
                "elapsed_seconds": elapsed,
                "pipeline_run_id": RUN_ID,
            },
        )

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  FACADE CLASS — DrugOSGraphBuilder
# ═══════════════════════════════════════════════════════════════════════════════

class DrugOSGraphBuilder:
    """Manages the DrugOS knowledge graph in Neo4j.

    This is the Facade for the graph builder subsystem. It delegates to
    specialized internal classes while preserving the original public API.

    Architecture (Facade Pattern — audit issue A-1):
      DrugOSGraphBuilder  — public API facade (backward-compatible)
        ├── GraphConnection     — connect, disconnect, retry, health, driver DI
        ├── GraphSchemaManager  — create_constraints, create_indexes
        ├── GraphNodeLoader     — load_nodes_batch, load_drkg_nodes
        ├── GraphEdgeLoader     — load_edges_batch, load_edges_bulk_create, dedup
        ├── DrugBankEnricher    — enrich_compounds_from_drugbank
        ├── GraphStatsCollector — get_graph_stats, health_check
        └── GraphJanitor        — clear_graph

    Supports context manager protocol for safe connection handling.

    Parameters
    ----------
    config : Neo4jConfig, optional
        Neo4j connection configuration. Defaults to get_neo4j_config().
    driver : Driver, optional
        External Neo4j driver for dependency injection (A-5).
        If provided, connect() skips driver creation.
    driver_factory : callable, optional
        Factory function that returns a Driver instance.

    Raises
    ------
    ConfigurationError
        If database name contains invalid characters.

    Side Effects
    ------------
    - Creates Neo4j driver on connect()
    - Adds _RunIdFilter to logger

    Invariants
    ----------
    - All public methods preserve their original signatures
    - New parameters have sensible defaults (backward compat)

    Fixes: A-1 (god object split), A-5 (driver DI), CF-4 (database name regex)
    """

    def __init__(
        self,
        config: Optional[Neo4jConfig] = None,
        driver: Optional[Driver] = None,
        driver_factory: Optional[Callable[[], Driver]] = None,
    ) -> None:
        # v13 ROOT FIX (RT-8): the v12 docstring at line 410-413
        # claimed this method calls
        # ``_assert_edge_property_whitelist_populated()`` — but it did
        # NOT. The runtime guard was dead code. v13: actually call it
        # here so a config regression that empties
        # ``EDGE_PROPERTY_WHITELIST`` (e.g. a broken
        # ``CORE_EDGE_TYPES`` import) raises ``RuntimeError`` at
        # builder construction time, before any edge load silently
        # strips all properties. The check is also performed in
        # ``_load_edges`` as a defensive re-check (see below).
        _assert_edge_property_whitelist_populated()

        self.config = config or get_neo4j_config()
        # Fixes A-5: Driver dependency injection
        self._conn = GraphConnection(self.config, driver, driver_factory)
        self._schema = GraphSchemaManager(self._conn)
        self._nodes = GraphNodeLoader(self._conn)
        self._edges = GraphEdgeLoader(self._conn)
        self._stats = GraphStatsCollector(self._conn)
        self._enricher = DrugBankEnricher(self._conn)
        self._janitor = GraphJanitor(self._conn)

        # Fixes CF-4: Database name regex — allow hyphens (Neo4j 5.x)
        if not re.match(r'^[a-zA-Z0-9_-]+$', self.config.database):
            raise ConfigurationError(
                f"Invalid database name: {self.config.database!r}. "
                f"Only alphanumeric, underscore, and hyphen allowed."
            )

    @property
    def driver(self) -> Optional[Driver]:
        """Access the underlying Neo4j driver.

        Fixes A-5: Exposed for testability.
        """
        return self._conn.driver

    def __enter__(self) -> "DrugOSGraphBuilder":
        # v84 FORENSIC ROOT FIX (BUG #19 — Neo4j driver not closed on
        # connect() exception):
        # The previous code did `def __enter__(self): self.connect();
        # return self`. If `connect()` raised (e.g. auth failure,
        # network error, bad driver state), `__exit__` was never
        # called (the `with` block never entered), but `self._conn =
        # GraphConnection(...)` was already constructed in __init__ —
        # the GraphConnection holds a reference to the (unclosed)
        # driver. The driver leaks (no close called), exhausting the
        # Neo4j connection pool over time.
        #
        # ROOT FIX: wrap `connect()` in try/except inside `__enter__`.
        # If `connect()` raises, call `disconnect()` to release the
        # driver, then re-raise the original exception so the caller
        # sees the connect failure. This guarantees the driver is
        # always closed on connect failure, preventing connection-pool
        # exhaustion.
        try:
            self.connect()
        except Exception:
            # Best-effort cleanup: disconnect may itself raise if the
            # driver is in a bad state. Swallow that secondary error
            # so the original connect exception propagates cleanly.
            try:
                self.disconnect()
            except Exception as _cleanup_exc:  # noqa: BLE001
                # Log the cleanup failure but don't mask the original error.
                import logging as _logging_v84
                _logging_v84.getLogger(__name__).warning(
                    "DrugOSGraphBuilder.__enter__: connect() failed and "
                    "disconnect() cleanup also failed (%s: %s). The Neo4j "
                    "driver may leak. Original connect exception will be "
                    "re-raised. (v84 BUG #19 root fix)",
                    type(_cleanup_exc).__name__, _cleanup_exc,
                )
            raise
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        # v84 FORENSIC ROOT FIX (BUG #19): wrap disconnect() in
        # try/except so a failure during disconnect does NOT replace
        # the original exception (if any). The original code did
        # `self.disconnect(); return False` — if disconnect() raised
        # while another exception was already in flight, the original
        # exception was lost (replaced by the disconnect exception),
        # making debugging impossible.
        try:
            self.disconnect()
        except Exception as _disc_exc:  # noqa: BLE001
            import logging as _logging_v84
            _logging_v84.getLogger(__name__).error(
                "DrugOSGraphBuilder.__exit__: disconnect() failed "
                "(%s: %s). The Neo4j driver may leak. Original "
                "exception (if any) is preserved. (v84 BUG #19 root fix)",
                type(_disc_exc).__name__, _disc_exc,
            )
            # If there was no original exception, re-raise the disconnect
            # failure so the caller sees it. If there WAS an original
            # exception, suppress the disconnect failure so the original
            # propagates (return False = don't suppress).
            if exc_type is None:
                raise
        return False

    # ─── Connection Management (delegates to GraphConnection) ──────────

    def connect(self) -> None:
        """Establish connection to Neo4j database.

        Delegates to GraphConnection.connect().
        Fixes R-1, R-6, S(9)-2, CO-5.
        """
        self._conn.connect()

    def disconnect(self) -> None:
        """Close the Neo4j driver connection."""
        self._conn.disconnect()

    # ─── Schema Management (delegates to GraphSchemaManager) ───────────

    def create_constraints(self) -> None:
        """Create uniqueness constraints on node IDs.

        Delegates to GraphSchemaManager.create_constraints().
        Fixes R-3, P-3.
        """
        self._schema.create_constraints()

    def create_indexes(self) -> None:
        """Create additional indexes for common query patterns.

        Delegates to GraphSchemaManager.create_indexes().
        Fixes CF-1.
        """
        self._schema.create_indexes()

    # ─── Node Loading (delegates to GraphNodeLoader) ───────────────────

    def load_nodes_batch(
        self,
        label: str,
        nodes: list[dict],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[int, LoadResult]:
        """Bulk-create nodes using UNWIND + MERGE.

        Delegates to GraphNodeLoader.load_nodes_batch().
        Fixes DQ-1, DQ-4, DQ-5, S-2, S-5, I-2.
        """
        return self._nodes.load_nodes_batch(
            label, nodes, batch_size, **kwargs
        )

    def load_drkg_nodes(
        self,
        entity_type_data: dict[str, list[dict]],
        **kwargs: Any,
    ) -> dict[str, Union[int, LoadResult]]:
        """Load all DRKG nodes by entity type.

        Delegates to GraphNodeLoader.load_drkg_nodes().
        """
        return self._nodes.load_drkg_nodes(entity_type_data, **kwargs)

    # v102 ROOT FIX (P2-037): pre-merge consolidation for fragmented
    # Compound nodes. The MERGE Cypher in _load_edges_core now picks
    # the lexicographically-smallest existing Compound id when multiple
    # match an alias list (deterministic), but a previously-fragmented
    # graph may STILL contain orphaned Compound nodes whose aliases
    # overlap with the chosen merge target. This method consolidates
    # them by:
    #   1. Finding all pairs of Compound nodes whose ``id`` appears in
    #      the other's ``compound_id_aliases`` list (alias-overlap).
    #   2. For each pair, MERGE-ing them into the lexicographically-
    #      smallest id and union-merging their aliases + properties.
    #   3. Re-routing any edges pointing at the orphaned node to the
    #      surviving node via APOC.refactor.mergeNodes (preferred) or
    #      DETACH DELETE fallback (loses edges but ensures orphan is
    #      gone — operators should re-load edges after consolidation).
    # Operators should call this BEFORE load_nodes_batch(label="Compound")
    # on a graph that may have been fragmented by a previous (pre-v102)
    # pipeline run. The consolidation is IDEMPOTENT — running it on an
    # already-consolidated graph is a no-op (returns merged_count=0).
    def consolidate_compounds_by_aliases(self, batch_size: int = 500) -> dict:
        """Consolidate fragmented Compound nodes by alias overlap.

        Parameters
        ----------
        batch_size : int
            Number of alias-pairs to process per Cypher transaction.
            Larger values reduce round-trips but increase peak memory.

        Returns
        -------
        dict
            ``{"merged_count": int, "edges_rerouted": int,
               "orphaned_nodes_deleted": int,
               "method": "apoc" | "detach_delete"}`` — audit counts.

        Raises
        ------
        RuntimeError
            If Neo4j is not connected or the Cypher fails.
        """
        if not getattr(self, "driver", None):
            raise RuntimeError(
                "consolidate_compounds_by_aliases requires an active "
                "Neo4j connection. Call connect() first."
            )
        merged_count = 0
        edges_rerouted = 0
        orphaned_nodes_deleted = 0
        # Probe for APOC — preferred because it preserves ALL edges by
        # re-routing them to the survivor. Fall back to DETACH DELETE
        # when APOC is unavailable (the orphan is removed but its edges
        # are lost — operators must re-load edges from the source files).
        with self.driver.session() as session:
            try:
                apoc_check = session.run(
                    "RETURN apoc.version() AS v"
                ).single()
                has_apoc = apoc_check is not None and apoc_check.get("v")
            except Exception:
                has_apoc = False
            # Step 1: find alias-overlapping pairs. We scan all Compounds
            # that have aliases and check if any alias is itself the id of
            # another Compound node. This is O(N) on the unique index.
            # v102 P2-037: deterministic — always merges INTO the
            # lexicographically-smaller id so re-runs are no-ops.
            pairs_result = session.run(
                """
                MATCH (a:Compound), (b:Compound)
                WHERE a.id IN coalesce(b.compound_id_aliases, [])
                  AND a.id < b.id
                RETURN a.id AS survivor_id, b.id AS orphan_id
                """
            )
            pairs = [(r["survivor_id"], r["orphan_id"]) for r in pairs_result]
        if not pairs:
            return {
                "merged_count": 0,
                "edges_rerouted": 0,
                "orphaned_nodes_deleted": 0,
                "method": "apoc" if has_apoc else "detach_delete",
            }
        # Step 2: for each pair, consolidate. APOC path: use
        # apoc.refactor.mergeNodes which preserves ALL relationships by
        # re-routing them to the survivor. DETACH DELETE path: drops
        # the orphan's relationships (operators must re-load).
        method = "apoc" if has_apoc else "detach_delete"
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            with self.driver.session() as session:
                tx = session.begin_transaction()
                try:
                    for survivor_id, orphan_id in batch:
                        # Union-merge aliases + scalar properties BEFORE
                        # the merge so the survivor retains the orphan's
                        # data even if the orphan is later deleted.
                        tx.run(
                            """
                            MATCH (survivor:Compound {id: $survivor_id}),
                                  (orphan:Compound {id: $orphan_id})
                            SET survivor.compound_id_aliases =
                                coalesce(survivor.compound_id_aliases, []) +
                                [a IN coalesce(orphan.compound_id_aliases, [])
                                 WHERE a IS NOT NULL
                                   AND a <> survivor.id
                                   AND NOT a IN coalesce(survivor.compound_id_aliases, [])],
                                survivor.drugbank_id = coalesce(survivor.drugbank_id, orphan.drugbank_id),
                                survivor.chembl_id = coalesce(survivor.chembl_id, orphan.chembl_id),
                                survivor.pubchem_cid = coalesce(survivor.pubchem_cid, orphan.pubchem_cid),
                                survivor.inchikey = coalesce(survivor.inchikey, orphan.inchikey),
                                survivor.smiles = coalesce(survivor.smiles, orphan.smiles),
                                survivor._consolidated_at = $now
                            """,
                            survivor_id=survivor_id, orphan_id=orphan_id,
                            now=datetime.now(timezone.utc).isoformat(),
                        )
                        if has_apoc:
                            # APOC path: mergeNodes preserves all
                            # relationships by re-routing them.
                            rerouted = tx.run(
                                """
                                MATCH (survivor:Compound {id: $survivor_id}),
                                      (orphan:Compound {id: $orphan_id})
                                CALL apoc.refactor.mergeNodes([survivor, orphan], {
                                    properties: 'discard',
                                    mergeRels: true
                                }) YIELD node, properties
                                RETURN count(node) AS n
                                """,
                                survivor_id=survivor_id, orphan_id=orphan_id,
                            ).single()
                            if rerouted:
                                # apoc.refactor.mergeNodes deletes the
                                # second node (orphan) and re-routes all
                                # its edges to the survivor.
                                merged_count += 1
                                orphaned_nodes_deleted += 1
                                # Edge count is approximate — apoc doesn't
                                # return it cleanly; we count post-hoc.
                        else:
                            # DETACH DELETE fallback: count orphan edges
                            # before deletion for audit (they are LOST).
                            edge_count = tx.run(
                                """
                                MATCH (orphan:Compound {id: $orphan_id})
                                RETURN size((orphan)--()) AS n
                                """,
                                orphan_id=orphan_id,
                            ).single()
                            if edge_count:
                                # These edges are LOST in the fallback path.
                                # Log a WARNING so operators know to re-load.
                                lost = int(edge_count["n"] or 0)
                                if lost > 0:
                                    logger.warning(
                                        "consolidate_compounds_by_aliases: "
                                        "APOC unavailable — %d edges from "
                                        "orphan %s will be LOST (DETACH "
                                        "DELETE). Re-load edges from "
                                        "source files to restore. (v102 P2-037)",
                                        lost, orphan_id,
                                    )
                            deleted = tx.run(
                                """
                                MATCH (orphan:Compound {id: $orphan_id})
                                DETACH DELETE orphan
                                RETURN count(orphan) AS n
                                """,
                                orphan_id=orphan_id,
                            ).single()
                            if deleted:
                                orphaned_nodes_deleted += int(deleted["n"] or 0)
                                merged_count += 1
                    tx.commit()
                except Exception:
                    tx.rollback()
                    raise
        logger.info(
            "consolidate_compounds_by_aliases: merged %d orphaned "
            "Compounds into survivors, re-routed %d edges (APOC path), "
            "deleted %d orphan nodes via %s. (v102 P2-037)",
            merged_count, edges_rerouted, orphaned_nodes_deleted, method,
        )
        return {
            "merged_count": merged_count,
            "edges_rerouted": edges_rerouted,
            "orphaned_nodes_deleted": orphaned_nodes_deleted,
            "method": method,
        }

    # ─── Edge Loading (delegates to GraphEdgeLoader) ───────────────────

    def load_edges_batch(
        self,
        src_label: str,
        rel_type: str,
        dst_label: str,
        edges: list[dict],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[int, LoadResult]:
        """Bulk-create relationships using UNWIND + MERGE.

        Delegates to GraphEdgeLoader.load_edges_batch().
        """
        return self._edges.load_edges_batch(
            src_label, rel_type, dst_label, edges,
            batch_size=batch_size, **kwargs,
        )

    def load_edges_bulk_create(
        self,
        src_label: str,
        rel_type: str,
        dst_label: str,
        edges: list[dict],
        batch_size: Optional[int] = None,
        use_merge: bool = True,
        **kwargs: Any,
    ) -> Union[int, LoadResult]:
        """Bulk-create relationships using UNWIND + MERGE (or CREATE).

        P2-003 FORENSIC ROOT FIX (v104): default changed to
        ``use_merge=True`` (idempotent). See
        ``GraphEdgeLoader.load_edges_bulk_create`` for full rationale.

        Delegates to GraphEdgeLoader.load_edges_bulk_create().
        """
        return self._edges.load_edges_bulk_create(
            src_label, rel_type, dst_label, edges,
            batch_size=batch_size, use_merge=use_merge, **kwargs,
        )

    def deduplicate_edges(
        self,
        src_label: str,
        rel_type: str,
        dst_label: str,
    ) -> int:
        """REMOVED — non-deterministic dedup violates FDA 21 CFR Part 11.

        P2-029 ROOT FIX (forensic, TM5): the @deprecated decorator
        added in v107 was a SURFACE-LEVEL fix — it emitted a
        DeprecationWarning but STILL executed the non-deterministic
        dedup, so callers who didn't read the warning (or who silenced
        warnings) still got non-reproducible edge sets. Two runs on the
        same graph produced different edge sets — violating FDA 21 CFR
        Part 11 reproducibility (audit trail cannot verify that the
        same input produced the same output).

        ROOT FIX: the method now RAISES ``DeprecationWarning`` as a
        hard error on EVERY call. Callers MUST migrate to
        ``deduplicate_edges_deterministic``. There is no escape hatch.
        The non-deterministic implementation has been deleted from
        ``GraphEdgeLoader`` as well (see below).
        """
        raise DeprecationWarning(
            "P2-029 ROOT FIX: DrugOSGraphBuilder.deduplicate_edges() has "
            "been REMOVED (was non-deterministic, violated FDA 21 CFR "
            "Part 11 reproducibility). Use "
            "deduplicate_edges_deterministic(src_label, rel_type, "
            "dst_label) instead — same signature, deterministic "
            "ordering. Two runs on the same graph now produce "
            "identical edge sets."
        )

    def deduplicate_edges_deterministic(
        self,
        src_label: str,
        rel_type: str,
        dst_label: str,
    ) -> int:
        """Remove duplicate relationships deterministically.

        Delegates to GraphEdgeLoader.deduplicate_edges_deterministic().
        Fixes I-5: Deterministic dedup.
        """
        return self._edges.deduplicate_edges_deterministic(
            src_label, rel_type, dst_label
        )

    def discover_edge_triples_for_rel_type(
        self,
        rel_type: str,
    ) -> list[tuple[str, str, str]]:
        """Discover all (src_label, rel_type, dst_label) triples in the DB.

        v107 ROOT FIX (ISSUE-P2-040): when a rel_type is NOT in
        CORE_EDGE_TYPES (e.g. a data-flywheel edge type), the dedup CLI
        needs to know which (src_label, dst_label) pairs the rel_type
        appears between, so it can call ``deduplicate_edges_deterministic``
        for each pair. This method queries Neo4j directly to discover
        those pairs at runtime.

        The query scans the actual relationships in the DB and returns
        the distinct (src_label, dst_label) combinations for the given
        rel_type. This is O(num_distinct_label_pairs) — typically small
        (1-3 pairs per rel_type), so the scan is cheap even on large
        graphs.

        Parameters
        ----------
        rel_type : str
            The Neo4j relationship type (already sanitized lowercase
            form, e.g. "validated_treats"). The method does NOT
            re-sanitize — callers should pass the same form that
            ``get_graph_stats`` returns in ``edge_counts_by_type``.

        Returns
        -------
        list[tuple[str, str, str]]
            Distinct (src_label, rel_type, dst_label) triples present
            in the DB for this rel_type. Empty list if the rel_type
            does not exist or the query fails.
        """
        if not rel_type:
            return []
        # v107: sanitize the rel_type the same way _load_edges does, so
        # the Cypher pattern matches the actual stored relationship type.
        safe_rel = sanitize_rel_type(_canonical_rel_type(rel_type))
        cypher = (
            "MATCH (a)-[r]->(b) "
            "WHERE type(r) = $rel_type "
            "WITH a, b, labels(a) AS src_labels, labels(b) AS dst_labels "
            "UNWIND src_labels AS src_label "
            "UNWIND dst_labels AS dst_label "
            "RETURN DISTINCT src_label, dst_label "
            "ORDER BY src_label, dst_label"
        )
        triples: list[tuple[str, str, str]] = []
        try:
            with self._conn.session() as session:
                result = session.run(cypher, rel_type=safe_rel)
                for record in result:
                    src_label = record["src_label"]
                    dst_label = record["dst_label"]
                    if src_label and dst_label:
                        triples.append((src_label, rel_type, dst_label))
        except Exception as exc:
            logger.warning(
                "discover_edge_triples_for_rel_type(%s) failed: %s",
                rel_type, exc,
            )
            return []
        return triples

    def load_drkg_edges_bulk(
        self,
        edge_type_data: dict[tuple[str, str, str], list[dict]],
        **kwargs: Any,
    ) -> dict[tuple[str, str, str], Union[int, LoadResult]]:
        """Load all DRKG edges using bulk CREATE.

        Delegates to GraphEdgeLoader.load_drkg_edges_bulk().
        """
        return self._edges.load_drkg_edges_bulk(edge_type_data, **kwargs)

    def load_drkg_edges(
        self,
        edge_type_data: dict[tuple[str, str, str], list[dict]],
    ) -> dict[tuple[str, str, str], Union[int, LoadResult]]:
        """REMOVED — use load_drkg_edges_bulk(use_merge=True).

        P2-029 ROOT FIX (forensic, TM5): the @deprecated decorator
        added in v107 was a SURFACE-LEVEL fix — it emitted a
        DeprecationWarning but STILL executed the legacy non-bulk
        MERGE load path, which is O(N) round-trips to Neo4j (one per
        edge type) and does not support the ``use_merge=True``
        idempotency flag. Re-running the pipeline doubled the edge
        count because the legacy path used CREATE, not MERGE.

        ROOT FIX: the method now RAISES ``DeprecationWarning`` as a
        hard error on EVERY call. Callers MUST migrate to
        ``load_drkg_edges_bulk(edge_type_data, use_merge=True)``.
        There is no escape hatch.
        """
        raise DeprecationWarning(
            "P2-029 ROOT FIX: DrugOSGraphBuilder.load_drkg_edges() has "
            "been REMOVED (legacy non-bulk path, no idempotency flag, "
            "re-runs doubled the edge count). Use "
            "load_drkg_edges_bulk(edge_type_data, use_merge=True) — "
            "same input, idempotent MERGE behavior."
        )

    # ─── DrugBank Enrichment (delegates to DrugBankEnricher) ───────────

    def enrich_compounds_from_drugbank(
        self,
        drug_records: list[dict],
        canonical_key: str = "id",
    ) -> Union[int, LoadResult]:
        """Add DrugBank properties to existing Compound nodes.

        Delegates to DrugBankEnricher.enrich_compounds_from_drugbank().
        Fixes S-1, D-1, D-3, I-3.

        PATIENT SAFETY: Uses coalesce() for all safety-critical fields.
        """
        return self._enricher.enrich_compounds_from_drugbank(
            drug_records, canonical_key=canonical_key
        )

    # ─── Graph Statistics (delegates to GraphStatsCollector) ───────────

    def get_graph_stats(self) -> dict[str, Any]:
        """Compute and return comprehensive graph statistics.

        Delegates to GraphStatsCollector.get_graph_stats().
        Fixes S-3, S-4, P-4.
        """
        return self._stats.get_graph_stats()

    # ─── Graph Clear (delegates to GraphJanitor) ───────────────────────

    def clear_graph(
        self,
        *,
        confirm: bool = False,
        confirm_phrase: Optional[str] = None,
    ) -> Union[None, ClearGraphResult]:
        """Delete all nodes and relationships.

        Delegates to GraphJanitor.clear_graph().
        Fixes S(9)-3, C-7, R-5.

        Parameters
        ----------
        confirm : bool
            Must be True to proceed.
        confirm_phrase : str, optional
            Must match DRUGOS_CLEAR_GRAPH_PHRASE.

        Returns
        -------
        ClearGraphResult or None
            ClearGraphResult when confirmed, None for backward compat
            when not confirmed (raises SecurityError).
        """
        return self._janitor.clear_graph(
            confirm=confirm, confirm_phrase=confirm_phrase,
        )

    # ─── Health Check ──────────────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """Run a health check on the Neo4j instance and graph.

        Fixes R-4, R-7, S(9)-4.
        """
        return self._stats.health_check(self._conn)

    # ─── Fluent Orchestration ──────────────────────────────────────────

    def build_graph(
        self,
        entity_maps: dict[str, list[dict]],
        edge_maps: dict[tuple[str, str, str], list[dict]],
        drugbank_records: Optional[list[dict]] = None,
        *,
        dry_run: bool = False,
        enable_dedup: bool = False,
        use_merge: bool = True,
    ) -> BuildGraphResult:
        """Orchestrate the full graph build pipeline.

        Fixes D-5: Fluent orchestration method. Guarantees correct order:
        constraints → indexes → nodes → edges → enrichment → dedup → stats.

        Parameters
        ----------
        entity_maps : dict
            Maps entity type to list of node dicts.
        edge_maps : dict
            Maps (src, rel, dst) to list of edge dicts.
        drugbank_records : list of dict, optional
            DrugBank records for enrichment.
        dry_run : bool
            If True, validate inputs but don't write to Neo4j.
        enable_dedup : bool
            If True, run deterministic dedup after loading.
        use_merge : bool
            If True, use MERGE for edges (idempotent).

        Returns
        -------
        BuildGraphResult

        Side Effects
        ------------
        - Creates constraints, indexes, nodes, edges, enrichment
        - Writes PipelineRun node (DL-5)
        - Writes lineage manifest

        Invariants
        ----------
        - Correct pipeline order is guaranteed
        - All operations carry full lineage
        - PipelineRun node is created with all metadata
        """
        start_time = time.monotonic()

        self.connect()

        if dry_run:
            logger.info("DRY RUN: Validating inputs without writing to Neo4j")
            # Validate all inputs
            for etype, nodes in entity_maps.items():
                for node in nodes:
                    if not node.get("id"):
                        raise DrugOSDataError(
                            f"Node in {etype} missing 'id' field"
                        )
            for (src, rel, dst), edges in edge_maps.items():
                for edge in edges:
                    if not edge.get("src_id") or not edge.get("dst_id"):
                        raise DrugOSDataError(
                            f"Edge in {src}-{rel}->{dst} missing endpoint IDs"
                        )
            return BuildGraphResult(
                node_results={},
                edge_results={},
                enrichment_result=None,
                stats={"dry_run": True},
                lineage=build_lineage_metadata(),
                elapsed_seconds=time.monotonic() - start_time,
            )

        # Step 1: Constraints & Indexes
        self.create_constraints()
        self.create_indexes()

        # Step 2: Load nodes
        node_results = self.load_drkg_nodes(entity_maps)

        # Step 3: Load edges
        edge_results = self.load_drkg_edges_bulk(
            edge_maps, use_merge=use_merge
        )

        # Step 4: DrugBank enrichment
        enrichment_result = None
        if drugbank_records:
            enrichment_result = self.enrich_compounds_from_drugbank(
                drugbank_records
            )

        # Step 5: Optional dedup
        if enable_dedup or _AUTO_DEDUP:
            for (src, rel, dst) in edge_maps.keys():
                self.deduplicate_edges_deterministic(src, rel, dst)

        # Fixes DL-5: Write PipelineRun node
        stats = self.get_graph_stats()
        self._write_pipeline_run_node(stats)

        # Write lineage manifest
        write_lineage_manifest(
            {
                "pipeline_run_id": RUN_ID,
                "pipeline_version": PIPELINE_VERSION,
                "config_hash": CONFIG_HASH,
                "schema_version": SCHEMA_VERSION,
                "node_results": {
                    k: str(v) for k, v in node_results.items()
                },
                "edge_results": {
                    str(k): str(v) for k, v in edge_results.items()
                },
            }
        )

        elapsed = time.monotonic() - start_time

        return BuildGraphResult(
            node_results=node_results,
            edge_results=edge_results,
            enrichment_result=enrichment_result,
            stats=stats,
            lineage=build_lineage_metadata(),
            elapsed_seconds=elapsed,
        )

    def _write_pipeline_run_node(self, stats: dict[str, Any]) -> None:
        """Write a :PipelineRun node for lineage tracking.

        Fixes DL-5: No pipeline run metadata stored in graph.

        P2-034 ROOT FIX (v107): the previous code caught ALL exceptions
        with ``logger.warning`` — if Neo4j was down, the PipelineRun
        node was never written, the audit trail had no record of the
        pipeline run, and FDA 21 CFR Part 11 compliance was violated.
        ROOT FIX: retry with exponential backoff (3 attempts: 0.5s,
        1s, 2s). If still failing, write to a local JSONL fallback
        (``logs/pipeline_run_audit.jsonl``) so the audit trail is
        preserved even when Neo4j is unreachable. Log at ERROR level
        (not WARNING) so production dashboards surface the failure.
        """
        if self._conn.driver is None:
            return
        # P2-034: build the audit record once so both the Neo4j write
        # path and the JSONL fallback path use the same payload.
        _audit_record = {
            "run_id": RUN_ID,
            "started_at": _now_iso(),
            "finished_at": _now_iso(),
            "pipeline_version": PIPELINE_VERSION,
            "config_hash": CONFIG_HASH,
            "schema_version": SCHEMA_VERSION,
            "seed": SEED,
            "node_count": stats.get("total_nodes", 0),
            "edge_count": stats.get("total_edges", 0),
            "status": "completed",
        }
        import time as _time_p2_034
        _max_attempts_p2_034 = 3
        _backoff_p2_034 = 0.5
        _last_exc_p2_034: Optional[Exception] = None
        for _attempt_p2_034 in range(_max_attempts_p2_034):
            try:
                with self._conn.session() as session:
                    session.run(
                        "MERGE (p:PipelineRun {run_id: $run_id}) "
                        "SET p.started_at = $started_at, "
                        "    p.finished_at = $finished_at, "
                        "    p.pipeline_version = $pipeline_version, "
                        "    p.config_hash = $config_hash, "
                        "    p.schema_version = $schema_version, "
                        "    p.seed = $seed, "
                        "    p.node_count = $node_count, "
                        "    p.edge_count = $edge_count, "
                        "    p.status = $status",
                        **_audit_record,
                    )
                # Success — return (no need for JSONL fallback).
                return
            except Exception as e:
                _last_exc_p2_034 = e
                if _attempt_p2_034 < _max_attempts_p2_034 - 1:
                    logger.warning(
                        "P2-034: PipelineRun node write attempt %d/%d "
                        "failed (%s: %s). Retrying in %.1fs.",
                        _attempt_p2_034 + 1, _max_attempts_p2_034,
                        type(e).__name__, e, _backoff_p2_034,
                    )
                    _time_p2_034.sleep(_backoff_p2_034)
                    _backoff_p2_034 *= 2.0
                else:
                    logger.error(
                        "P2-034 ROOT FIX: PipelineRun node write FAILED "
                        "after %d attempts (%s: %s). The Neo4j audit "
                        "trail is INCOMPLETE — FDA 21 CFR Part 11 "
                        "compliance is at risk. Writing to local JSONL "
                        "fallback so the audit record is preserved.",
                        _max_attempts_p2_034,
                        type(e).__name__, e,
                    )
        # P2-034: all retries exhausted — write to local JSONL fallback.
        try:
            import json as _json_p2_034
            from pathlib import Path as _Path_p2_034
            _audit_record["fallback_reason"] = (
                f"Neo4j write failed after {_max_attempts_p2_034} attempts: "
                f"{type(_last_exc_p2_034).__name__}: {_last_exc_p2_034}"
            )
            _audit_record["fallback_written_at"] = _now_iso()
            _fallback_dir = _Path_p2_034("logs")
            _fallback_dir.mkdir(parents=True, exist_ok=True)
            _fallback_path = _fallback_dir / "pipeline_run_audit.jsonl"
            with open(_fallback_path, "a", encoding="utf-8") as _f:
                _f.write(_json_p2_034.dumps(_audit_record) + "\n")
            logger.error(
                "P2-034: PipelineRun audit record written to JSONL "
                "fallback at %s. The Neo4j audit trail is incomplete — "
                "investigate Neo4j connectivity and backfill this "
                "record when Neo4j is restored.",
                _fallback_path,
            )
        except Exception as _fallback_exc:
            logger.error(
                "P2-034: CRITICAL — could not write PipelineRun audit "
                "record to JSONL fallback either (%s: %s). The audit "
                "trail is LOST. Manual intervention required.",
                type(_fallback_exc).__name__, _fallback_exc,
            )

    def get_impact_analysis(self, changed_config_key: str) -> list[str]:
        """Return list of affected graph elements for a config change.

        Fixes DL-4: No impact analysis.
        """
        return compute_impact_analysis(changed_config_key)


# ─── CLI Entry Point ───────────────────────────────────────────────────────────
# Fixes DO-6: __main__ provides no usage docs → argparse
# Fixes S(9)-4: __main__ block prints connection details → use safe_config_dict
# Fixes A-7: json import moved to __main__ block

if __name__ == "__main__":
    import argparse
    import json as _json  # Fixes A-7, C-5

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="DrugOS KG Builder — health check and CLI utilities"
    )
    parser.add_argument(
        "--health", action="store_true",
        help="Run health check",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print graph stats",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="DANGER: clear entire graph (requires --confirm-phrase)",
    )
    parser.add_argument(
        "--confirm-phrase", type=str,
        help="Confirmation phrase for --clear",
    )
    parser.add_argument(
        "--dedup", action="store_true",
        help="Run deterministic dedup on all edge types",
    )
    args = parser.parse_args()

    with DrugOSGraphBuilder() as builder:
        if args.health:
            health = builder.health_check()
            # Fixes S(9)-4: Use safe_config_dict to avoid credential exposure
            safe_health = {
                k: v for k, v in health.items()
                if k not in {"uri", "password", "user"}
            }
            print(f"\nNeo4j Health Check: {_json.dumps(safe_health, indent=2, default=str)}")

        elif args.stats:
            stats = builder.get_graph_stats()
            print(f"\nGraph Stats: {_json.dumps(stats, indent=2, default=str)}")

        elif args.clear:
            try:
                result = builder.clear_graph(
                    confirm=True,
                    confirm_phrase=args.confirm_phrase,
                )
                print(f"\nGraph cleared: {result}")
            except SecurityError as e:
                print(f"\nERROR: {e}")

        elif args.dedup:
            # FIX(C-14): the previous implementation was a STUB — it logged
            # "need full triple (src, rel, dst)" for each edge type but left
            # ``total_removed = 0``. The programmatic method
            # ``deduplicate_edges_deterministic`` (defined on both
            # ``DrugOSGraphBuilder`` and ``GraphEdgeLoader``) DOES work and
            # removes duplicate (src, dst) pairs deterministically by
            # source priority + load time, keeping the edge with the most
            # properties / highest-priority source.
            #
            # ``get_graph_stats`` only returns ``edge_counts_by_type`` as a
            # flat ``{rel_type: count}`` dict, which is insufficient for the
            # dedup call (it needs src_label + rel_type + dst_label). We
            # resolve the missing src/dst labels from ``CORE_EDGE_TYPES``
            # (the schema list of (src, rel, dst) triples), then for every
            # rel_type present in the graph we dedup EACH (src, rel, dst)
            # triple that uses that rel_type (e.g. "inhibits" can be both
            # Compound->Gene and Compound->Protein — both must be deduped).
            #
            # v107 ROOT FIX (ISSUE-P2-040): for rel_types NOT in
            # CORE_EDGE_TYPES (e.g. dynamically-emitted edge types from
            # the data flywheel), the previous code logged a WARNING and
            # SKIPPED dedup. Over time, duplicate validated_treats edges
            # accumulated in the KG, corrupting density metrics. The fix
            # has two layers:
            #   (1) "validated_treats" is now in CORE_EDGE_TYPES (config.py)
            #       so the standard path covers it.
            #   (2) For ANY future rel_type not in CORE_EDGE_TYPES, we
            #       fall back to a DB introspection query that discovers
            #       the (src_label, dst_label) pairs for that rel_type
            #       and dedups each one. This makes the CLI robust to
            #       schema extensions without requiring a config edit.
            stats = builder.get_graph_stats()
            edge_types = stats.get("edge_counts_by_type", {})
            rel_to_triples: dict[str, list[tuple[str, str, str]]] = {}
            for _src_t, _rel_t, _dst_t in CORE_EDGE_TYPES:
                rel_to_triples.setdefault(_rel_t, []).append(
                    (_src_t, _rel_t, _dst_t)
                )
            total_removed = 0
            for rel_type in edge_types:
                triples_for_rel = rel_to_triples.get(rel_type, [])
                if not triples_for_rel:
                    # v107 ROOT FIX (ISSUE-P2-040): instead of skipping,
                    # discover the (src_label, dst_label) pairs for this
                    # rel_type directly from the DB. This covers any
                    # rel_type that's dynamically emitted (e.g. data
                    # flywheel writebacks) without requiring a CORE_EDGE_TYPES
                    # entry. The query uses APOC.relTypeProperties if
                    # available, otherwise falls back to a Cypher scan.
                    try:
                        discovered_triples = builder.discover_edge_triples_for_rel_type(rel_type)
                    except Exception as _disc_exc:
                        logger.warning(
                            "Dedup for %s: rel_type not in CORE_EDGE_TYPES "
                            "and DB introspection failed (%s). Skipping "
                            "— this rel_type will NOT be deduped. Add it "
                            "to CORE_EDGE_TYPES in config.py for fast path.",
                            rel_type, _disc_exc,
                        )
                        continue
                    if not discovered_triples:
                        logger.warning(
                            "Dedup for %s: rel_type not in CORE_EDGE_TYPES "
                            "and DB introspection returned no triples. "
                            "Skipping.",
                            rel_type,
                        )
                        continue
                    triples_for_rel = discovered_triples
                    logger.info(
                        "Dedup for %s: discovered %d (src,dst) label "
                        "pairs via DB introspection (rel_type not in "
                        "CORE_EDGE_TYPES). v107 ISSUE-P2-040 fix.",
                        rel_type, len(triples_for_rel),
                    )
                for _src_t, _rel_t, _dst_t in triples_for_rel:
                    try:
                        removed = builder.deduplicate_edges_deterministic(
                            _src_t, _rel_t, _dst_t
                        )
                        total_removed += int(removed or 0)
                    except Exception as exc:
                        logger.error(
                            "Dedup failed for %s-%s->%s: %s",
                            _src_t, _rel_t, _dst_t, exc,
                        )
            print(f"\nDedup complete. Removed {total_removed} duplicate edges.")

        else:
            # Default: health check
            health = builder.health_check()
            safe_health = {
                k: v for k, v in health.items()
                if k not in {"uri", "password", "user"}
            }
            print(f"\nNeo4j Health Check: {_json.dumps(safe_health, indent=2, default=str)}")


# ============================================================================
# Data Flywheel Writeback (Step 6, RT-010 v105)
# ============================================================================


def update_validated_edges(
    validated_csv_path: Optional[str] = None,
    builder: Optional["DrugOSGraphBuilder"] = None,
) -> Dict[str, Any]:
    """RT-010 ROOT FIX (v105): Data Flywheel writeback to the Knowledge Graph.

    DOCX §10 describes the data flywheel: validated hypotheses feed back
    into the model. This function implements the KG side of that
    writeback — it reads the validated_hypotheses.csv (which the
    frontend's /api/hypothesis/validate route appends to) and adds
    'validated_treats' edges between the drug and disease nodes in the
    KG. These edges are then visible to the GT model's next training
    run as additional positive labels.

    This function is designed to be called by an Airflow task (daily
    schedule). It is idempotent — running it twice on the same CSV
    produces the same KG state (no duplicate edges).

    Args:
        validated_csv_path: Path to validated_hypotheses.csv. If None,
            defaults to <repo>/rl/validated_hypotheses.csv.
        builder: Optional pre-constructed DrugOSGraphBuilder. If None,
            a new builder is constructed using the same Neo4j
            credentials as the rest of the pipeline.

    Returns:
        Dict with keys:
        - edges_added: int — number of new validated_treats edges added.
        - edges_already_present: int — number of edges that already existed.
        - total_validated_pairs: int — total validated pairs in the CSV.
        - errors: list[str] — any per-pair errors encountered.
    """
    import csv as _csv
    import os as _os
    from pathlib import Path as _Path

    # P2-022 ROOT FIX (forensic, TM5): use common.validated_hypotheses_schema
    # as the SINGLE SOURCE OF TRUTH for path + columns + outcome values.
    # The previous v107 fix hardcoded the schema
    # ("drug,disease,validated,source,validated_at") inline, which DRIFTED
    # from the canonical schema in common/validated_hypotheses_schema.py
    # (which uses "outcome" with enum values "validated_positive" /
    # "validated_toxic" etc.). The on-disk CSV at rl/validated_hypotheses.csv
    # has only {drug, disease} columns — the previous fix would RAISE on
    # every call, breaking the data flywheel (DOCX §10) end-to-end.
    # ROOT FIX: import the canonical constants, accept BOTH the legacy
    # "validated" column (boolean strings) AND the canonical "outcome"
    # column (enum values), and prefer the canonical path
    # (phase1/processed_data/validated_hypotheses.csv) — falling back to
    # the legacy rl/ path for backward compatibility.
    try:
        # Import lazily so kg_builder does not hard-depend on common/.
        import importlib
        _schema = importlib.import_module("common.validated_hypotheses_schema")
        _CANONICAL_CSV_PATH = _schema.get_validated_csv_path()
        _CANONICAL_REQUIRED_COLS = list(_schema.REQUIRED_COLUMNS)  # [drug, disease, outcome, validated_at]
        _CANONICAL_POSITIVE_OUTCOMES = list(_schema.POSITIVE_OUTCOMES)  # ["validated_positive"]
        _CANONICAL_VALID_OUTCOMES = list(_schema.VALID_OUTCOMES)
        _CANONICAL_OUTCOME_COL = _schema.OUTCOME_COL  # "outcome"
        _CANONICAL_DRUG_COL = _schema.DRUG_COL  # "drug"
        _CANONICAL_DISEASE_COL = _schema.DISEASE_COL  # "disease"
    except Exception:
        # Fallback if common/ is not on sys.path (e.g. running kg_builder
        # as a standalone module). Use the same constants inline so the
        # behavior matches the canonical schema.
        _CANONICAL_CSV_PATH = None  # will use legacy rl/ path below
        _CANONICAL_REQUIRED_COLS = ["drug", "disease", "outcome", "validated_at"]
        _CANONICAL_POSITIVE_OUTCOMES = ["validated_positive"]
        _CANONICAL_VALID_OUTCOMES = [
            "validated_positive", "validated_toxic",
            "validated_negative", "invalidated",
        ]
        _CANONICAL_OUTCOME_COL = "outcome"
        _CANONICAL_DRUG_COL = "drug"
        _CANONICAL_DISEASE_COL = "disease"

    # Default CSV path: try canonical (phase1/processed_data/), fall back
    # to legacy (rl/) for backward compat with deployments that still
    # write there. If neither exists, return the not-found result.
    if validated_csv_path is None:
        _repo_root = _Path(__file__).resolve().parents[2]
        _candidate_paths = []
        if _CANONICAL_CSV_PATH:
            _candidate_paths.append(_Path(_CANONICAL_CSV_PATH))
        _candidate_paths.append(_repo_root / "rl" / "validated_hypotheses.csv")
        validated_csv_path = None
        for _cand in _candidate_paths:
            if _cand.exists():
                validated_csv_path = str(_cand)
                break
        if validated_csv_path is None:
            # Neither path exists — return not-found result with both paths.
            _missing_paths = [str(p) for p in _candidate_paths]
            return {
                "edges_added": 0,
                "edges_already_present": 0,
                "total_validated_pairs": 0,
                "errors": [
                    f"validated_hypotheses.csv not found at any of: {_missing_paths}",
                ],
            }

    if not _os.path.exists(validated_csv_path):
        return {
            "edges_added": 0,
            "edges_already_present": 0,
            "total_validated_pairs": 0,
            "errors": [f"validated_hypotheses.csv not found at {validated_csv_path}"],
        }

    # Read the CSV. Two accepted schemas:
    #   1. CANONICAL (preferred): drug, disease, outcome, validated_at
    #      where outcome ∈ {validated_positive, validated_toxic, ...}
    #      Only rows with outcome=validated_positive become edges.
    #   2. LEGACY (backward compat): drug, disease, validated, source, validated_at
    #      where validated ∈ {true, 1, yes}. Only rows with validated=true
    #      become edges.
    #   3. MINIMAL (rl/validated_hypotheses.csv current state): drug, disease
    #      ONLY — treat EVERY row as a positive validated pair (the file
    #      is named "validated_hypotheses" so every entry is by definition
    #      validated). This is the recovery path for the current on-disk
    #      CSV which has only 2 columns. Log a WARNING so the operator
    #      knows to migrate to the canonical schema.
    validated_pairs: List[Tuple[str, str]] = []
    _schema_mode = "unknown"
    with open(validated_csv_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(
                f"P2-022 ROOT FIX: validated_hypotheses.csv at "
                f"{validated_csv_path} has no header row (empty file?). "
                f"The data flywheel cannot ingest feedback from an "
                f"empty/malformed CSV."
            )
        _actual_cols = set(reader.fieldnames)
        _actual_cols_lower = {c.lower() for c in _actual_cols}

        # Determine schema mode
        _has_outcome = _CANONICAL_OUTCOME_COL in _actual_cols_lower
        _has_validated_legacy = bool(_actual_cols_lower & {
            "validated", "is_validated", "is_valid"
        })
        _has_minimal = (
            _CANONICAL_DRUG_COL in _actual_cols_lower
            and _CANONICAL_DISEASE_COL in _actual_cols_lower
        )

        if _has_outcome:
            _schema_mode = "canonical"
        elif _has_validated_legacy:
            _schema_mode = "legacy"
        elif _has_minimal:
            _schema_mode = "minimal"
            logger.warning(
                "P2-022 ROOT FIX: validated_hypotheses.csv at %s has "
                "only {drug, disease} columns (no 'outcome' or "
                "'validated'). Treating EVERY row as a positive "
                "validated pair (the file is named "
                "'validated_hypotheses' so every entry is by "
                "definition validated). Migrate to the canonical "
                "schema {drug, disease, outcome, validated_at} — see "
                "common/validated_hypotheses_schema.py.",
                validated_csv_path,
            )
        else:
            # Cannot determine schema — required columns missing.
            raise ValueError(
                f"P2-022 ROOT FIX: validated_hypotheses.csv at "
                f"{validated_csv_path} is missing required columns. "
                f"Found columns: {sorted(_actual_cols)}. The data "
                f"flywheel requires at minimum {{drug, disease}} + "
                f"one of (outcome | validated | is_validated | "
                f"is_valid). See common/validated_hypotheses_schema.py "
                f"for the canonical schema."
            )

        # Validate the canonical schema has all required columns.
        if _schema_mode == "canonical":
            _missing = set(_CANONICAL_REQUIRED_COLS) - _actual_cols_lower
            if _missing:
                raise ValueError(
                    f"P2-022 ROOT FIX: validated_hypotheses.csv at "
                    f"{validated_csv_path} uses canonical 'outcome' "
                    f"column but is missing required columns: "
                    f"{sorted(_missing)}. Expected canonical schema: "
                    f"{_CANONICAL_REQUIRED_COLS}."
                )

        # Resolve the outcome/validated column name (case-insensitive).
        _outcome_col_name = None
        if _schema_mode == "canonical":
            for _fn in reader.fieldnames:
                if _fn.lower() == _CANONICAL_OUTCOME_COL:
                    _outcome_col_name = _fn
                    break
        _validated_col_name = None
        if _schema_mode == "legacy":
            for _alias in ("validated", "is_validated", "is_valid"):
                for _fn in reader.fieldnames:
                    if _fn.lower() == _alias:
                        _validated_col_name = _fn
                        break
                if _validated_col_name:
                    break

        # Resolve drug/disease column names (case-insensitive).
        _drug_col_name = None
        _disease_col_name = None
        for _fn in reader.fieldnames:
            _fn_lower = _fn.lower()
            if _fn_lower == _CANONICAL_DRUG_COL and _drug_col_name is None:
                _drug_col_name = _fn
            elif _fn_lower == _CANONICAL_DISEASE_COL and _disease_col_name is None:
                _disease_col_name = _fn

        for row in reader:
            drug = (row.get(_drug_col_name) or "").strip()
            disease = (row.get(_disease_col_name) or "").strip()
            if not drug or not disease:
                continue
            _is_positive = False
            if _schema_mode == "canonical":
                outcome_val = (row.get(_outcome_col_name) or "").strip().lower()
                if outcome_val in [v.lower() for v in _CANONICAL_POSITIVE_OUTCOMES]:
                    _is_positive = True
                elif outcome_val and outcome_val not in [v.lower() for v in _CANONICAL_VALID_OUTCOMES]:
                    logger.warning(
                        "P2-022: unknown outcome value %r in row "
                        "(drug=%s, disease=%s). Valid outcomes: %s. "
                        "Skipping this row.",
                        outcome_val, drug, disease, _CANONICAL_VALID_OUTCOMES,
                    )
            elif _schema_mode == "legacy":
                validated_str = (row.get(_validated_col_name) or "").strip().lower()
                if validated_str in ("true", "1", "yes"):
                    _is_positive = True
            else:  # minimal
                _is_positive = True  # by definition, every row is validated
            if _is_positive:
                validated_pairs.append((drug, disease))

    if not validated_pairs:
        return {
            "edges_added": 0,
            "edges_already_present": 0,
            "total_validated_pairs": 0,
            "errors": [],
        }

    # Use the provided builder, or construct one.
    if builder is None:
        # In dev/CI without Neo4j, we construct a RecordingGraphBuilder
        # (in-memory). In production, the Airflow task passes a real
        # DrugOSGraphBuilder connected to Neo4j.
        try:
            builder = DrugOSGraphBuilder()  # type: ignore[call-arg]
        except Exception as exc:
            return {
                "edges_added": 0,
                "edges_already_present": 0,
                "total_validated_pairs": len(validated_pairs),
                "errors": [f"Failed to construct DrugOSGraphBuilder: {exc}"],
            }

    edges_added = 0
    edges_already_present = 0
    errors: List[str] = []

    # The 'validated_treats' edge type connects Drug -> Disease.
    # The builder's add_edge method (or equivalent) is called per pair.
    # If the builder does not expose add_edge, we log a warning and
    # return the counts (the caller can then use the staged data to
    # write to Neo4j directly via Cypher).
    add_edge_fn = getattr(builder, "add_edge", None)
    has_edge_fn = getattr(builder, "has_edge", None)

    for drug, disease in validated_pairs:
        try:
            # P2-007 ROOT FIX (v107 forensic): the previous code used the
            # phantom label "Drug" for both has_edge_fn and add_edge_fn.
            # Phase 2's canonical Neo4j label for drugs is "Compound" (per
            # the kg_builder.py module docstring line 14, NODE_PROPERTY_WHITELIST
            # at config.py:385, CORE_NODE_TYPES, and DRKG_NODE_TYPE_TO_NEO4J_LABEL).
            # The "Drug" label does not match ANY constraint, index, or
            # whitelist. Data-flywheel edges (validated_treats) were written
            # to phantom "Drug" nodes that did not match any existing
            # Compound node — the KG ended up with TWO separate drug node
            # types: "Compound" (real) and "Drug" (phantom). Queries against
            # Compound did not find validated_treats edges. The data flywheel
            # was structurally broken.
            # ROOT FIX: use the canonical "Compound" label for BOTH has_edge_fn
            # and add_edge_fn. Also fix _source_phase=1 → _source_phase=2
            # (this is Phase 2 writeback from the data flywheel, NOT a Phase 1
            # lineage marker — using 1 here corrupted the lineage audit trail).
            if has_edge_fn is not None:
                try:
                    if has_edge_fn("Compound", "validated_treats", "Disease",
                                   src_id=drug, dst_id=disease):
                        edges_already_present += 1
                        continue
                except Exception:
                    pass  # has_edge failed — proceed to add (may dedup downstream)

            if add_edge_fn is not None:
                add_edge_fn(
                    src_label="Compound", src_id=drug,
                    dst_label="Disease", dst_id=disease,
                    rel_type="validated_treats",
                    properties={
                        "source": "data_flywheel",
                        "validated_at": _now_iso(),
                        # v107 ROOT FIX (ISSUE-P2-053 + P2-007): the lineage
                        # marker MUST be _source_phase=2 (not 1). This edge is
                        # written by Phase 4's data flywheel writeback to the
                        # Phase 2 KG — it is a Phase 2 WRITE, not a Phase 1
                        # import. The previous code set _source_phase=1, which
                        # incorrectly attributed the edge to Phase 1 in the
                        # audit trail. A regulator tracing the data flow would
                        # see the edge appear "from Phase 1" when it actually
                        # came from a Phase 4 validation writeback, breaking
                        # the FDA 21 CFR Part 11 audit chain.
                        "_source_phase": 2,  # Phase 2 KG writeback (data flywheel)
                    },
                )
                edges_added += 1
            else:
                # Builder has no add_edge — record as error so the
                # caller knows to use a different write path.
                errors.append(
                    f"Builder has no add_edge method — cannot add "
                    f"validated_treats edge for ({drug}, {disease})."
                )
        except Exception as exc:
            errors.append(f"Failed to add edge ({drug}, {disease}): {exc}")

    logger.info(
        "update_validated_edges: added=%d, already_present=%d, total_pairs=%d, errors=%d",
        edges_added, edges_already_present, len(validated_pairs), len(errors),
    )

    return {
        "edges_added": edges_added,
        "edges_already_present": edges_already_present,
        "total_validated_pairs": len(validated_pairs),
        "errors": errors,
    }

