"""DrugOS Graph — Schema Configuration (extracted from config.py).

v108 ROOT FIX (ISSUE-P2-056): this module is the FIRST real extraction
from the 8400-line ``config.py`` monolith. The previous v107 "fix" only
DOCUMENTED the intended split and explicitly deferred execution to
"v2.1.0" — a surface-level fix. This module DOES the split: the schema
constants are MOVED here (not copied), and ``config.py`` imports them
back so ``from .config import CORE_NODE_TYPES`` continues to work.

The schema constants are leaf definitions (lists of strings / tuples)
with no dependencies on other config sections, so they can be extracted
cleanly without circular imports.

Consumers should prefer importing from this module directly:
    from .config_schema import CORE_NODE_TYPES, CORE_EDGE_TYPES
The legacy ``from .config import CORE_NODE_TYPES`` continues to work via
re-export in ``config.py``.
"""

from __future__ import annotations

from typing import Tuple

# ─── Core Node Types ─────────────────────────────────────────────────────────
#
# The 7 canonical node types in the DrugOS knowledge graph.
#
# v107 ROOT FIX (ISSUE-P2-040 / P2-053 invariant): this list is the
# AUTHORITATIVE schema. ``CORE_EDGE_TYPES`` (below) MUST only reference
# node types from this list (or from ``DRKG_NODE_TYPES`` for legacy
# DRKG edges). The ``kg_builder._validate_core_edge_types`` invariant
# check enforces this at import time.

CORE_NODE_TYPES = ["Compound", "Disease", "Gene", "Protein", "Pathway",
                   # FIX-F / C-16: DOCX Phase 2 spec mandates 5 node types
                   # (Drugs, Proteins, Pathways, Diseases, Clinical Outcomes).
                   # The bridge previously emitted only 4 (Compound, Protein,
                   # Gene, Disease) — ClinicalOutcome was missing entirely.
                   # phase1_bridge._load_clinical_outcomes() now derives
                   # ClinicalOutcome nodes from drugbank_indications.csv.
                   "ClinicalOutcome",
                   # Phase 0.3 (master_prompt D2.9 / D14.12) — SIDER uses
                   # MedDRA vocabulary for adverse events.
                   # v38 ROOT FIX (Phase 2 Issue #12): the previous code
                   # shipped BOTH "MedDRA_Term" (canonical, underscore) AND
                   # "Side Effect" (legacy, space) as "migration-period
                   # dual-write". Neo4j labels with spaces require backtick
                   # quoting (``:``Side Effect````) which is fragile and
                   # error-prone. The fix: standardize on "MedDRA_Term".
                   "MedDRA_Term",  # "Side Effect" deprecated — see v38 fix
                   # v109 ROOT FIX (P2-004): "Drug" was referenced by
                   # CORE_EDGE_TYPES entry ("Drug", "validated_treats",
                   # "Disease") at line 136 but was NOT in CORE_NODE_TYPES
                   # — invariant violation. The data flywheel creates
                   # "Drug" nodes from literature-validated treatment
                   # records (separate from "Compound" nodes which are
                   # sourced from ChEMBL/DrugBank structural records).
                   # ROOT FIX: add "Drug" as a CORE_NODE_TYPE alias. The
                   # ``NODE_LABEL_LOWERCASE`` map (below) already maps
                   # both "Compound" and "Drug" to "drug" — so they are
                   # the SAME Phase 3 node type, but stored as distinct
                   # Neo4j labels in Phase 2 (allowing the data flywheel
                   # to distinguish literature-validated from
                   # structurally-derived drug records).
                   "Drug"]

# Extended node types from DRKG (13 entity types)
# NOTE: 'Protein' is NOT in DRKG; DRKG uses 'Gene' for both gene and protein
# product. Protein nodes are added only when UniProt data is loaded.
# Fixes audit issue 3.2 — Added "Atc" and "Tax" DRKG node types
DRKG_NODE_TYPES = [
    "Compound", "Disease", "Gene", "Anatomy",
    "Pharmacologic Class", "Side Effect", "Symptom",
    "Pathway", "Biological Process", "Molecular Function",
    "Cellular Component", "Taxonomy", "Gene Expression",
    "Atc", "Tax",
    # Fixes audit issue 3.10 — MedDRA_Term for SIDER adverse events
    "MedDRA_Term",
]

# ─── Core Edge Types ─────────────────────────────────────────────────────────
#
# The core edge types in DrugOS (spec + Gene-encodes-Protein bridge)
#
# SCIENTIFIC CORRECTNESS FIXES (highest priority):
#   - Added ("Gene", "encodes", "Protein") — the biological bridge
#     between gene and protein product. Without this edge, the graph
#     is disconnected between Gene-side and Protein-side data.
#   - Added ("Compound", "targets", "Protein") — drug-target edges
#     from UniProt/ChEMBL/STITCH/STRING all use the Protein endpoint.
#   - Added ("Protein", "participates_in", "Pathway") — protein (not
#     gene) participates in pathways per Reactome / KEGG convention.
#   - Added ("Compound", "inhibits", "Protein") — issue 3.1: many
#     ChEMBL/DrugBank inhibition targets are proteins (not genes).
#   - Added ("Compound", "activates", "Protein") — issue 3.1: many
#     activation targets are proteins.
#   - Added ("Compound", "tested_for", "Disease") — issue 3.8:
#     clinical trial records use "tested for" rather than "treats"
#     for unapproved indications.
#   - Added ("Protein", "associated_with", "Disease") — issue 3.3:
#     GWAS and PheWAS associate PROTEINS (gene products) with diseases,
#     not genes directly.
#   - Added ("Compound", "causes_adverse_event", "MedDRA_Term") —
#     issue 3.10: SIDER uses MedDRA terms, not "Side Effect" generic.
#   - Added ("Protein", "expressed_in", "Anatomy") — issue 3.11:
#     protein expression (from HPA) is distinct from gene expression.
#   - Added ("Pathway", "associated_with", "Disease") — issue 3.9:
#     pathway-disease associations (e.g., from KEGG Disease).

CORE_EDGE_TYPES: list[Tuple[str, str, str]] = [
    # ── Original edges (backward compat) ──
    ("Compound", "treats", "Disease"),
    ("Compound", "inhibits", "Gene"),          # DRKG drug-gene inhibition
    ("Compound", "activates", "Gene"),          # DRKG drug-gene activation
    ("Compound", "targets", "Protein"),         # cross-database drug-protein
    ("Compound", "binds", "Protein"),           # physical binding (ChEMBL/STITCH)
    ("Gene", "encodes", "Protein"),             # gene -> protein product bridge
    ("Gene", "associated_with", "Disease"),     # DRKG gene-disease
    ("Gene", "interacts_with", "Gene"),         # PPI (DRKG uses Gene for both ends)
    ("Protein", "interacts_with", "Protein"),   # STRING PPI (UniProt accession IDs)
    # v113 FORENSIC ROOT FIX (P2-049, MEDIUM — KG-Semantics):
    #   ``RecordingGraphBuilder.load_edges_batch`` checks
    #   ``edge_key not in CORE_EDGE_TYPES`` and dead-letters any edge
    #   NOT in the whitelist. The previous whitelist included BOTH:
    #     ("Compound", "causes_side_effect", "Side Effect")   # SIDER legacy
    #     ("Compound", "causes_adverse_event", "MedDRA_Term") # SIDER canonical
    #   The legacy edge bypassed the canonical schema -- a SIDER edge
    #   emitted with the legacy "Side Effect" label (with a SPACE in
    #   the label, requiring backtick quoting in Cypher) was accepted
    #   into the KG, splitting adverse-event counts per drug between
    #   two label namespaces. The RL safety ranker queries
    #   ``(:Compound)-[:causes_adverse_event]->(:MedDRA_Term)`` and
    #   would MISS adverse events recorded under the legacy label,
    #   under-counting adverse events and ranking dangerous drugs as
    #   'green' (safe).
    #
    #   ROOT FIX: the legacy edge type is REMOVED from the whitelist
    #   (the tuple is commented out below -- DO NOT re-enable it). Any
    #   new edge emission MUST use the canonical form
    #   ``("Compound", "causes_adverse_event", "MedDRA_Term")``. The
    #   ``RecordingGraphBuilder`` will dead-letter any edge emitted
    #   with the legacy tuple, forcing callers to migrate. Existing
    #   KGs with legacy ``:Side Effect`` nodes should run the one-time
    #   Cypher migration in
    #   ``scripts/migrate_sidetoeffect_to_meddraterm.py`` (already
    #   exists in the repo) to convert legacy edges to canonical form.
    # ("Compound", "causes_side_effect", "Side Effect"),  # SIDER legacy -- REMOVED v113 P2-049
    ("Gene", "expressed_in", "Anatomy"),
    ("Gene", "participates_in", "Pathway"),
    ("Protein", "participates_in", "Pathway"),  # Reactome uses protein participants
    ("Pathway", "disrupted_in", "Disease"),     # spec edge from Phase 2 doc
    # ── New edges from scientific correctness audit ──
    ("Compound", "inhibits", "Protein"),        # ChEMBL/DrugBank protein targets
    ("Compound", "activates", "Protein"),       # ChEMBL/DrugBank protein targets
    ("Compound", "tested_for", "Disease"),
    # v102 ROOT FIX (P2-042): "failed_for" edge for clinical trials that
    # completed but FAILED their primary endpoint.
    ("Compound", "failed_for", "Disease"),
    ("Protein", "associated_with", "Disease"),
    ("Compound", "causes_adverse_event", "MedDRA_Term"),  # SIDER canonical (see v113 P2-049 above)
    ("Protein", "expressed_in", "Anatomy"),
    ("Pathway", "associated_with", "Disease"),
    # ── DrugBank v2.0 audit-fix edge types ──
    ("Compound", "metabolized_by", "Protein"),
    ("Compound", "carried_by", "Protein"),
    ("Compound", "transported_by", "Protein"),
    ("Compound", "induces", "Protein"),
    ("Compound", "allosterically_modulates", "Protein"),
    ("Compound", "unknown", "Protein"),
    ("Compound", "interacts_with", "Compound"),
    # ── v15 ROOT FIX (REM-12): OMIM susceptibility vs causative GDA ──
    ("Gene", "susceptible_to", "Disease"),
    # ── FIX-F / C-16: ClinicalOutcome node + edge ──
    ("Compound", "has_clinical_outcome", "ClinicalOutcome"),
    # v107 ROOT FIX (ISSUE-P2-040): "validated_treats" was missing from
    # CORE_EDGE_TYPES — the data flywheel creates these edges.
    ("Drug", "validated_treats", "Disease"),
]

# Fixes audit issue 2.1 — CORE_EDGE_TYPES_SET for O(1) lookup
CORE_EDGE_TYPES_SET: frozenset[Tuple[str, str, str]] = frozenset(CORE_EDGE_TYPES)


# ─── v108 ROOT FIX (issues 70 & 71): Lowercase canonical node labels & canonical edge-type strings ──
#
# Phase 3 (graph_transformer) expects LOWERCASE node labels and canonical
# `src_verb_dst` edge-type strings. The PascalCase labels above remain the
# internal storage format (for backward compat with the Neo4j schema and
# all existing loaders); these new constants provide the canonical export
# format consumed by Phase 3 and by the KG-builder's `register_node` /
# `register_edge` methods (issue 65 / 66).
#
# Mapping rationale (per the project docx + Phase 3 contract):
#   * "Compound"  → "drug"        (the docx calls them "Drugs (10,000 FDA-approved compounds)")
#   * "Protein"   → "protein"
#   * "Gene"      → "gene"
#   * "Disease"   → "disease"
#   * "Pathway"   → "pathway"
#   * "ClinicalOutcome" → "clinical_outcome"
#   * "MedDRA_Term"     → "side_effect"   (per Phase 3 audit spec)
#   * "Side Effect"     → "side_effect"   (legacy alias normalised)
#   * "Anatomy"        → "anatomy"
#   * "Pharmacologic Class" → "pharmacologic_class"
#   * "Symptom"        → "symptom"
#   * "Biological Process" → "biological_process"
#   * "Molecular Function" → "molecular_function"
#   * "Cellular Component" → "cellular_component"
#   * "Taxonomy"       → "taxonomy"
#   * "Gene Expression" → "gene_expression"
#   * "Atc" / "ATC"    → "atc"
#   * "Tax" / "TAX"    → "tax"
#   * "Drug"           → "drug"           (literature-validated drug alias)
NODE_LABEL_LOWERCASE: dict[str, str] = {
    "Compound": "drug",
    "Drug": "drug",
    "Protein": "protein",
    "Gene": "gene",
    "Disease": "disease",
    "Pathway": "pathway",
    "ClinicalOutcome": "clinical_outcome",
    "MedDRA_Term": "side_effect",
    "Side Effect": "side_effect",
    "Anatomy": "anatomy",
    "Pharmacologic Class": "pharmacologic_class",
    "Symptom": "symptom",
    "Biological Process": "biological_process",
    "Molecular Function": "molecular_function",
    "Cellular Component": "cellular_component",
    "Taxonomy": "taxonomy",
    "Gene Expression": "gene_expression",
    "Atc": "atc",
    "ATC": "atc",
    "Tax": "tax",
    "TAX": "tax",
}

# Reverse mapping: lowercase canonical → primary PascalCase label.
# For aliases ("Drug"→"drug", "Side Effect"→"side_effect", "ATC"→"atc",
# "TAX"→"tax"), we map back to the PRIMARY form (the first one listed above).
NODE_LABEL_PASCALCASE: dict[str, str] = {
    "drug": "Compound",
    "protein": "Protein",
    "gene": "Gene",
    "disease": "Disease",
    "pathway": "Pathway",
    "clinical_outcome": "ClinicalOutcome",
    "side_effect": "MedDRA_Term",
    "anatomy": "Anatomy",
    "pharmacologic_class": "Pharmacologic Class",
    "symptom": "Symptom",
    "biological_process": "Biological Process",
    "molecular_function": "Molecular Function",
    "cellular_component": "Cellular Component",
    "taxonomy": "Taxonomy",
    "gene_expression": "Gene Expression",
    "atc": "Atc",
    "tax": "Tax",
}


def canonical_node_label(pascal_or_lower: str) -> str:
    """Return the lowercase canonical form of a node label.

    Accepts either PascalCase ("Compound") or already-lowercase ("drug").
    Unknown labels are returned lowercased with non-alphanumerics replaced
    by underscores (so callers can pass novel labels without crashing).
    """
    if pascal_or_lower is None:
        return ""
    s = str(pascal_or_lower)
    if s in NODE_LABEL_LOWERCASE:
        return NODE_LABEL_LOWERCASE[s]
    if s in NODE_LABEL_PASCALCASE:
        return s
    # Unknown: normalise to lowercase + underscore
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def pascal_node_label(lower: str) -> str:
    """Return the PascalCase form of a lowercase canonical node label."""
    if lower is None:
        return ""
    s = str(lower)
    return NODE_LABEL_PASCALCASE.get(s, s)


# ─── Canonical edge-type strings (issue 71) ─────────────────────────────────
#
# Each CORE_EDGE_TYPES tuple (src_pascal, verb, dst_pascal) maps to a single
# canonical `src_verb_dst` string using the LOWERCASE node labels above.
# Examples:
#   ("Compound", "treats", "Disease")     → "drug_treats_disease"
#   ("Compound", "inhibits", "Protein")   → "drug_inhibits_protein"
#   ("Protein", "interacts_with", "Protein") → "protein_interacts_with_protein"
#   ("Gene", "encodes", "Protein")        → "gene_encodes_protein"
#   ("Compound", "causes_adverse_event", "MedDRA_Term") → "drug_causes_adverse_event_side_effect"
#   ("Compound", "causes_side_effect", "Side Effect")   → "drug_causes_side_effect_side_effect"
#
# The verb is kept LITERAL (not collapsed) so each CORE_EDGE_TYPES tuple
# produces a UNIQUE canonical name. The lowercase dst label is appended
# even when the verb already contains a related noun (e.g. "causes_adverse_event"
# + "side_effect" → "drug_causes_adverse_event_side_effect") — this preserves
# the 1:1 mapping between CORE_EDGE_TYPES tuples and canonical strings.
EDGE_TYPE_CANONICAL: dict[Tuple[str, str, str], str] = {}
_EDGE_TYPE_CANONICAL_NAMES: set[str] = set()
for _src, _verb, _dst in CORE_EDGE_TYPES:
    _src_l = NODE_LABEL_LOWERCASE.get(_src, _src.lower())
    _dst_l = NODE_LABEL_LOWERCASE.get(_dst, _dst.lower())
    _name = f"{_src_l}_{_verb}_{_dst_l}"
    EDGE_TYPE_CANONICAL[(_src, _verb, _dst)] = _name
    _EDGE_TYPE_CANONICAL_NAMES.add(_name)
del _src, _verb, _dst, _src_l, _dst_l, _name

# Reverse map: canonical edge-type string → (src_pascal, verb, dst_pascal).
# For ambiguous names (e.g. "drug_causes_side_effect" maps to both legacy
# and canonical forms), we keep the CANONICAL form (MedDRA_Term) — the
# legacy "Side Effect" form is mapped but not preferred.
EDGE_TYPE_FROM_CANONICAL: dict[str, Tuple[str, str, str]] = {}
for _tup, _name in EDGE_TYPE_CANONICAL.items():
    _existing_tup = EDGE_TYPE_FROM_CANONICAL.get(_name)
    if _existing_tup is None:
        EDGE_TYPE_FROM_CANONICAL[_name] = _tup
    else:
        # Prefer the non-legacy form. Legacy edges use "Side Effect" as dst.
        if "Side Effect" in (_existing_tup[0], _existing_tup[2]) and "Side Effect" not in (_tup[0], _tup[2]):
            EDGE_TYPE_FROM_CANONICAL[_name] = _tup


def canonical_edge_type(src: str, verb: str, dst: str) -> str:
    """Return the canonical `src_verb_dst` string for an edge type.

    Accepts PascalCase OR lowercase src/dst. The verb must already be in
    snake_case form (e.g. "interacts_with", "treats", "causes_adverse_event").

    Examples:
        >>> canonical_edge_type("Compound", "treats", "Disease")
        'drug_treats_disease'
        >>> canonical_edge_type("drug", "inhibits", "protein")
        'drug_inhibits_protein'
    """
    src_l = canonical_node_label(src)
    dst_l = canonical_node_label(dst)
    return f"{src_l}_{verb}_{dst_l}"


def parse_canonical_edge_type(name: str) -> Tuple[str, str, str]:
    """Inverse of :func:`canonical_edge_type`.

    Returns the (PascalCase_src, verb, PascalCase_dst) tuple for a canonical
    edge-type string. Raises ``KeyError`` if the name is not registered.
    """
    if name not in EDGE_TYPE_FROM_CANONICAL:
        raise KeyError(
            f"Unknown canonical edge type: {name!r}. "
            f"Known types: {sorted(_EDGE_TYPE_CANONICAL_NAMES)}"
        )
    return EDGE_TYPE_FROM_CANONICAL[name]


__all__ = [
    "CORE_NODE_TYPES",
    "DRKG_NODE_TYPES",
    "CORE_EDGE_TYPES",
    "CORE_EDGE_TYPES_SET",
    # v108 (issues 70 & 71) — lowercase canonical labels & edge-type strings
    "NODE_LABEL_LOWERCASE",
    "NODE_LABEL_PASCALCASE",
    "EDGE_TYPE_CANONICAL",
    "EDGE_TYPE_FROM_CANONICAL",
    "canonical_node_label",
    "pascal_node_label",
    "canonical_edge_type",
    "parse_canonical_edge_type",
]
