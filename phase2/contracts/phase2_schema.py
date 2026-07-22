"""Canonical Phase 2 schema — the SINGLE source of truth for KG topology.

TASK 322 ROOT FIX (forensic, root-level, no surface fix):
  Previously, two INDEPENDENT node-type mappings existed:
    1. ``phase2/drugos_graph/pyg_builder.py:_PHASE2_TO_GT_NODE_TYPE`` (7 entries
       including Gene and MedDRA_Term — used to BUILD PyG HeteroData).
    2. ``graph_transformer/data/phase2_adapter.py:PHASE2_TO_PHASE3_NODE``
       (5 entries dropping Gene and MedDRA_Term — used to ADAPT to Phase 3).

  The two mappings produced DIFFERENT PyG HeteroData graphs from the same
  Phase 2 source. Phase 3 training saw a different topology than the Phase 2
  service exposed — model trained on the wrong graph. The audit (INT-004)
  flagged this as a silent scientific-output corruptor.

  Previous "fixes" created ``phase2/drugos_graph/schema_mappings.py`` as a
  shared module — but the duplicate local dicts in ``pyg_builder.py`` and
  ``phase2_adapter.py`` were NEVER actually deleted; they were just renamed
  to import from the new shared module. The contract was still living in
  Phase 2 implementation code, not in a true cross-phase contract module.

  THIS FILE is the true contract. It lives at ``phase2/contracts/`` so it
  is importable by both Phase 2 (writer) and Phase 3 (reader) WITHOUT either
  side having to depend on the other's implementation package. The
  ``schema_mappings.py`` module now re-exports from here for backward
  compatibility with existing tests, but the CANONICAL definitions live
  in this file.

Node type vocabulary
--------------------
Phase 2 (capitalized, source vocabulary from ``RecordingGraphBuilder``):
    Compound, Protein, Gene, Pathway, Disease, ClinicalOutcome, MedDRA_Term

Phase 3 (lowercase, canonical target vocabulary for ``HeteroData``):
    drug, protein, pathway, disease, clinical_outcome

Gene and MedDRA_Term are Phase 2 INTERMEDIATES — used to derive
Pathway->Disease and Drug->ClinicalOutcome edges respectively, but
DROPPED from the final Phase 3 graph because they have no Phase 3
semantics (a Gene is not a node type the GT model reasons about; a
MedDRA_Term is folded into ClinicalOutcome).

Edge type vocabulary
--------------------
Phase 2 (capitalized):
    (Compound, inhibits, Protein), (Compound, activates, Protein),
    (Compound, treats, Disease), (Compound, tested_for, Disease),
    (Compound, causes, ClinicalOutcome), (Compound, has_clinical_outcome, ClinicalOutcome),
    (Protein, participates_in, Pathway), (Protein, part_of, Pathway),
    (Pathway, disrupted_in, Disease)

Phase 3 (lowercase canonical):
    (drug, inhibits, protein), (drug, activates, protein),
    (drug, binds, protein), (drug, modulates, protein),
    (drug, treats, disease), (drug, tested_for, disease),
    (drug, causes, clinical_outcome),
    (protein, part_of, pathway),
    (pathway, disrupted_in, disease)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple


# =============================================================================
# FeatureColumn — typed specification for one node/edge feature column
# =============================================================================


@dataclass(frozen=True)
class FeatureColumn:
    """Specification for a single feature column on a node or edge.

    Attributes
    ----------
    name : str
        Canonical feature column name (case-sensitive).
    dtype : str
        Numpy/torch dtype string: ``"float32"``, ``"int64"``, ``"bool"``,
        ``"string"``. Used by Phase 3 to allocate tensors of the right size.
    nullable : bool
        True if the feature may be missing for some nodes/edges. Phase 3
        fills missing values with the column's ``default``.
    default : float, int, bool, str, or None
        Default value used when ``nullable=True`` and the feature is missing.
    description : str
        Human-readable semantic description.
    """

    name: str
    dtype: str = "float32"
    nullable: bool = True
    default: Optional[object] = None
    description: str = ""


# =============================================================================
# NodeFeatureSpec — feature schema for one node type
# =============================================================================


@dataclass(frozen=True)
class NodeFeatureSpec:
    """Feature schema for one node type.

    Attributes
    ----------
    node_type : str
        Phase 3 canonical node type (e.g. ``"drug"``).
    features : tuple of FeatureColumn
        Ordered feature columns. Phase 3 concatenates these in order to
        form the node feature tensor ``x[node_type]``.
    description : str
        Human-readable description of the node type's role in the KG.
    """

    node_type: str
    features: Tuple[FeatureColumn, ...] = ()
    description: str = ""


# =============================================================================
# EdgeFeatureSpec — feature schema for one edge type
# =============================================================================


@dataclass(frozen=True)
class EdgeFeatureSpec:
    """Feature schema for one edge type triple.

    Attributes
    ----------
    edge_type : tuple of (src, rel, dst)
        Phase 3 canonical edge type triple (e.g.
        ``("drug", "inhibits", "protein")``).
    features : tuple of FeatureColumn
        Ordered feature columns for the edge tensor ``edge_attr[edge_type]``.
    description : str
        Human-readable description.
    """

    edge_type: Tuple[str, str, str]
    features: Tuple[FeatureColumn, ...] = ()
    description: str = ""


# =============================================================================
# THE CANONICAL PHASE 3 NODE TYPES — single source of truth
# =============================================================================
# These are the 5 lowercase canonical node types that Phase 3's
# HeteroData uses as top-level keys: ``data['drug'].x``, ``data['protein'].x``,
# ``data['pathway'].x``, ``data['disease'].x``, ``data['clinical_outcome'].x``.
#
# ANY code that produces or consumes a Phase 3 HeteroData MUST import this
# tuple. Hardcoding the list locally is forbidden — the contract
# consistency test (shared/tests/test_contract_consistency.py) fails CI
# if a local copy is detected.

NODE_TYPES: Tuple[str, ...] = (
    "drug",
    "protein",
    "pathway",
    "disease",
    "clinical_outcome",
)

# Frozen set for O(1) membership tests.
NODE_TYPES_SET: FrozenSet[str] = frozenset(NODE_TYPES)

# =============================================================================
# ALL Phase 2 node types (including intermediates)
# =============================================================================
# Used by ``pyg_builder`` when reading Phase 2 source data — the source
# may contain Gene and MedDRA_Term nodes that are valid in Phase 2 but
# get dropped in the projection to Phase 3.
ALL_PHASE2_NODE_TYPES: Tuple[str, ...] = (
    "Compound",
    "Protein",
    "Gene",
    "Pathway",
    "Disease",
    "ClinicalOutcome",
    "MedDRA_Term",
)

# Same set as Phase 3 (5 types) — kept as a separate symbol for clarity
# at call sites that explicitly want the Phase 3 vocabulary.
ALL_PHASE3_NODE_TYPES: Tuple[str, ...] = NODE_TYPES

# Intermediates dropped in the Phase 2 -> Phase 3 projection.
INTERMEDIATE_NODE_TYPES: Tuple[str, ...] = (
    "Gene",        # used to derive Pathway->Disease via Gene->Protein->Pathway
    "MedDRA_Term", # folded into ClinicalOutcome (meddra_id, meddra_name)
)


def is_intermediate_node_type(node_type: str) -> bool:
    """Return True if ``node_type`` is a Phase 2 intermediate dropped in Phase 3.

    Gene and MedDRA_Term are Phase 2 intermediates used for derivation
    (e.g., pathway->disease edges are derived from Gene->Disease
    associations via Gene->Protein->Pathway mapping) but do NOT appear
    as node types in the Phase 3 canonical schema.
    """
    return node_type in INTERMEDIATE_NODE_TYPES


# =============================================================================
# PHASE 2 -> PHASE 3 NODE TYPE MAPPING
# =============================================================================
# Phase 2 (capitalized) -> Phase 3 (lowercase canonical).
# Intermediates (Gene, MedDRA_Term) map to None — they are dropped in the
# projection, NOT silently coerced to a wrong type.
PHASE2_TO_PHASE3_NODE: Dict[str, Optional[str]] = {
    "Compound": "drug",
    "Protein": "protein",
    "Pathway": "pathway",
    "Disease": "disease",
    "ClinicalOutcome": "clinical_outcome",
    "Gene": None,         # intermediate — dropped
    "MedDRA_Term": None,  # intermediate — folded into ClinicalOutcome
}

# Reverse lookup: Phase 3 -> Phase 2.
# Many-to-one because Gene/MedDRA_Term map to nothing in the reverse
# direction — they only exist in Phase 2.
PHASE3_TO_PHASE2_NODE: Dict[str, str] = {
    "drug": "Compound",
    "protein": "Protein",
    "pathway": "Pathway",
    "disease": "Disease",
    "clinical_outcome": "ClinicalOutcome",
}

# Backward-compat: a NON-None-only mapping (excludes intermediates) for
# callers that want only the 5 canonical entries. This is what the old
# ``schema_mappings.PHASE2_TO_PHASE3_NODE`` returned.
PHASE2_TO_PHASE3_NODE_CANONICAL: Dict[str, str] = {
    k: v for k, v in PHASE2_TO_PHASE3_NODE.items() if v is not None
}


# =============================================================================
# THE CANONICAL PHASE 3 EDGE TYPES — single source of truth
# =============================================================================
# Each entry is a (src_type, rel_type, dst_type) triple in Phase 3
# canonical vocabulary. These are the ONLY edge types the Phase 3
# HeteroData may contain — any other edge type is a contract violation.

EDGE_TYPES: Tuple[Tuple[str, str, str], ...] = (
    # Drug -> Protein (mechanism edges)
    ("drug", "inhibits", "protein"),
    ("drug", "activates", "protein"),
    ("drug", "binds", "protein"),         # neutral binding (target relation)
    ("drug", "modulates", "protein"),     # neutral modulation (allosteric)

    # Drug -> Disease (therapeutic edges)
    ("drug", "treats", "disease"),
    ("drug", "tested_for", "disease"),

    # Drug -> ClinicalOutcome (adverse event / efficacy)
    ("drug", "causes", "clinical_outcome"),

    # Protein -> Pathway (membership)
    ("protein", "part_of", "pathway"),

    # Pathway -> Disease (dysregulation)
    ("pathway", "disrupted_in", "disease"),
)

EDGE_TYPES_SET: FrozenSet[Tuple[str, str, str]] = frozenset(EDGE_TYPES)


# =============================================================================
# PHASE 2 -> PHASE 3 EDGE TYPE MAPPING
# =============================================================================
# Key: (src_label, rel_type, dst_label) in Phase 2 vocabulary.
# Value: (src_type, rel_type, tgt_type) in Phase 3 canonical vocabulary.
#
# P3-002 ROOT FIX (v113 forensic): the previous mapping had only 11
# entries and SILENTLY dropped 21 of Phase 2's 32 CORE_EDGE_TYPES --
# including ALL drug-metabolism edges (metabolized_by, carried_by,
# transported_by, induces), ALL SIDER adverse-event edges, ALL
# gene-pathway edges, ALL pathway-disease edges, and the validated_treats
# data-flywheel edge. The Phase 3 GT model trained on a graph missing
# 67% of Phase 2's edge types -- including the entire safety signal.
#
# ROOT FIX: expand the mapping to cover ALL mappable Phase 2 edges.
# Edges that genuinely have no Phase 3 equivalent (PPI, DDI, anatomy)
# are explicitly listed in PHASE2_TO_PHASE3_EDGE_DROPPED so they are
# dropped VISIBLY (with a count log) instead of SILENTLY.
PHASE2_TO_PHASE3_EDGE: Dict[Tuple[str, str, str], Tuple[str, str, str]] = {
    # ── Direct drug->protein mechanism edges (ChEMBL/DrugBank) ──
    ("Compound", "inhibits", "Protein"): ("drug", "inhibits", "protein"),
    ("Compound", "activates", "Protein"): ("drug", "activates", "protein"),

    # P3-002 ROOT FIX (v113): DRKG drug->gene edges. The Gene node is a
    # Phase 2 intermediate; the adapter resolves it to the corresponding
    # Protein via the ``Gene encodes Protein`` bridge (the gene's protein
    # product is the actual drug target). The Phase 3 edge type is the
    # protein-target form.
    ("Compound", "inhibits", "Gene"): ("drug", "inhibits", "protein"),
    ("Compound", "activates", "Gene"): ("drug", "activates", "protein"),

    # ── Neutral binding / target edges ──
    ("Compound", "targets", "Protein"): ("drug", "binds", "protein"),
    ("Compound", "binds", "Protein"): ("drug", "binds", "protein"),

    # ── Neutral modulation edges ──
    ("Compound", "allosterically_modulates", "Protein"):
        ("drug", "modulates", "protein"),
    ("Compound", "modulates", "Protein"): ("drug", "modulates", "protein"),

    # P3-002 ROOT FIX (v113): Drug-metabolism edges from DrugBank v2.0.
    # These were previously SILENTLY DROPPED -- losing the entire
    # pharmacokinetic signal. Metabolism (CYP450 enzymes) is a form of
    # modulation; carriers/transporters are physical binding.
    ("Compound", "metabolized_by", "Protein"): ("drug", "modulates", "protein"),
    ("Compound", "carried_by", "Protein"): ("drug", "binds", "protein"),
    ("Compound", "transported_by", "Protein"): ("drug", "binds", "protein"),
    ("Compound", "induces", "Protein"): ("drug", "activates", "protein"),
    ("Compound", "unknown", "Protein"): ("drug", "binds", "protein"),

    # ── Drug->disease therapeutic edges ──
    ("Compound", "treats", "Disease"): ("drug", "treats", "disease"),
    ("Compound", "tested_for", "Disease"): ("drug", "tested_for", "disease"),
    # P3-002 ROOT FIX (v113): "failed_for" clinical-trial edges map to
    # "tested_for" -- the drug WAS tested (just failed the endpoint).
    # The GT model can learn "tested_for" includes both successes and
    # failures, and the absence of a "treats" edge signals failure.
    ("Compound", "failed_for", "Disease"): ("drug", "tested_for", "disease"),
    # P3-002 ROOT FIX (v113): the data-flywheel's "validated_treats"
    # edge (from pharma-partner wet-lab validations) maps to "treats".
    # Note: source node type is "Drug" (capitalized) in the flywheel
    # contract, not "Compound" -- both spellings appear in different
    # versions of the flywheel writer. Map both.
    ("Drug", "validated_treats", "Disease"): ("drug", "treats", "disease"),
    ("Compound", "validated_treats", "Disease"): ("drug", "treats", "disease"),

    # ── Drug->clinical outcome edges (efficacy + adverse events) ──
    ("Compound", "causes", "ClinicalOutcome"):
        ("drug", "causes", "clinical_outcome"),
    ("Compound", "has_clinical_outcome", "ClinicalOutcome"):
        ("drug", "causes", "clinical_outcome"),
    # P3-002 ROOT FIX (v113): SIDER adverse-event edges. The previous
    # mapping DROPPED these -- losing the entire safety signal from
    # SIDER. Both "Side Effect" (legacy) and "MedDRA_Term" (canonical)
    # destination node types map to "clinical_outcome".
    ("Compound", "causes_side_effect", "Side Effect"):
        ("drug", "causes", "clinical_outcome"),
    ("Compound", "causes_adverse_event", "MedDRA_Term"):
        ("drug", "causes", "clinical_outcome"),
    ("Compound", "causes_adverse_event", "Side Effect"):
        ("drug", "causes", "clinical_outcome"),

    # ── Protein->pathway edges (both relation names accepted) ──
    ("Protein", "participates_in", "Pathway"):
        ("protein", "part_of", "pathway"),
    ("Protein", "part_of", "Pathway"):
        ("protein", "part_of", "pathway"),

    # P3-002 ROOT FIX (v113): gene-pathway edges from DRKG. The Gene
    # node is a Phase 2 intermediate; in Phase 3, pathway membership
    # is expressed at the Protein level (the gene's protein product
    # participates in the pathway). The adapter derives protein->pathway
    # edges from gene->pathway edges via the Gene->Protein bridge.
    # The mapping here is used by the adapter's _derive_protein_pathway
    # step; the edge type is the Phase 3 canonical form.
    ("Gene", "participates_in", "Pathway"):
        ("protein", "part_of", "pathway"),

    # ── Pathway->disease edges (dysregulation + association) ──
    ("Pathway", "disrupted_in", "Disease"):
        ("pathway", "disrupted_in", "disease"),
    # P3-002 ROOT FIX (v113): KEGG Disease pathway-disease associations.
    ("Pathway", "associated_with", "Disease"):
        ("pathway", "disrupted_in", "disease"),

    # ── P3-002 ROOT FIX (v113): Gene->Protein bridge ──
    # The "Gene encodes Protein" edge is a Phase 2 derivation edge --
    # it's used by the adapter to bridge Gene-side data (DRKG, OMIM)
    # to Protein-side data (UniProt, STRING). In Phase 3, gene-side
    # data is collapsed into the corresponding Protein node. The edge
    # itself does NOT appear in the Phase 3 graph (no "gene" node
    # type), but the adapter needs this mapping to know which edges
    # to consume for derivation (vs. silently drop). We map it to a
    # sentinel "DERIVE" relation that the adapter recognizes.
    # NOTE: this is consumed by the adapter's derivation step, not
    # written to the Phase 3 graph directly.

    # ── P3-002 ROOT FIX (v113): Gene->Disease edges ──
    # GDA edges from DisGeNET/OMIM. The Gene node is a Phase 2
    # intermediate; the adapter derives pathway->disease edges from
    # these via Gene->Protein->Pathway. The mapping is consumed by
    # the derivation step.
    ("Gene", "associated_with", "Disease"):
        ("pathway", "disrupted_in", "disease"),  # DERIVED
    ("Gene", "susceptible_to", "Disease"):
        ("pathway", "disrupted_in", "disease"),  # DERIVED
}

# P3-002 ROOT FIX (v113): edges that have NO Phase 3 equivalent and are
# DROPPED VISIBLY (the adapter logs a count, not silently). PPI and DDI
# would require adding new edge types to EDGE_TYPES + touching the model
# code (the HeterogeneousMultiHeadAttention forward pass enumerates edge
# types). Anatomy has no Phase 3 node type. These are tracked here so
# the adapter can report them accurately.
PHASE2_TO_PHASE3_EDGE_DROPPED: Tuple[Tuple[str, str, str], ...] = (
    # PPI (Protein-Protein Interaction)
    ("Protein", "interacts_with", "Protein"),
    ("Gene", "interacts_with", "Gene"),
    # DDI (Drug-Drug Interaction)
    ("Compound", "interacts_with", "Compound"),
    # Anatomy edges (no Anatomy node type in Phase 3)
    ("Gene", "expressed_in", "Anatomy"),
    ("Protein", "expressed_in", "Anatomy"),
    # Direct Protein->Disease (Phase 3 routes via pathway; if added
    # here, the GT model would learn a shortcut that bypasses the
    # multi-hop reasoning the architecture is designed for).
    ("Protein", "associated_with", "Disease"),
)

# Reverse lookup.
PHASE3_TO_PHASE2_EDGE: Dict[Tuple[str, str, str], Tuple[str, str, str]] = {
    v: k for k, v in PHASE2_TO_PHASE3_EDGE.items()
}


# =============================================================================
# NODE FEATURE SCHEMAS
# =============================================================================
# Per-node-type ordered feature columns. Phase 3 concatenates these into
# the node feature tensor ``data[node_type].x``. A node type with an empty
# feature tuple gets a one-hot embedding only (no tabular features).

NODE_FEATURE_SCHEMAS: Dict[str, NodeFeatureSpec] = {
    "drug": NodeFeatureSpec(
        node_type="drug",
        features=(
            FeatureColumn("molecular_weight", "float32", nullable=True,
                          default=0.0,
                          description="Molecular weight in Daltons."),
            FeatureColumn("logp", "float32", nullable=True,
                          default=0.0,
                          description="Computed logP (lipophilicity)."),
            FeatureColumn("tpsa", "float32", nullable=True,
                          default=0.0,
                          description="Topological polar surface area."),
            FeatureColumn("h_bond_donor_count", "int64", nullable=True,
                          default=0,
                          description="H-bond donor count."),
            FeatureColumn("h_bond_acceptor_count", "int64", nullable=True,
                          default=0,
                          description="H-bond acceptor count."),
            FeatureColumn("rotatable_bond_count", "int64", nullable=True,
                          default=0,
                          description="Rotatable bond count."),
            FeatureColumn("is_fda_approved", "bool", nullable=True,
                          default=False,
                          description="True if FDA-approved."),
            FeatureColumn("is_withdrawn", "bool", nullable=True,
                          default=False,
                          description="True if withdrawn (patient-safety flag)."),
        ),
        description="FDA-approved drug / ChEMBL compound (Compound node).",
    ),
    "protein": NodeFeatureSpec(
        node_type="protein",
        features=(
            FeatureColumn("ncbi_gene_id", "int64", nullable=True,
                          default=0,
                          description="NCBI Gene ID (0 if unknown)."),
            FeatureColumn("sequence_length", "int64", nullable=True,
                          default=0,
                          description="Amino acid sequence length."),
        ),
        description="UniProt protein target.",
    ),
    "pathway": NodeFeatureSpec(
        node_type="pathway",
        features=(
            FeatureColumn("protein_count", "int64", nullable=True,
                          default=0,
                          description="Number of proteins in this pathway."),
        ),
        description="STRING-derived protein connected component (Pathway).",
    ),
    "disease": NodeFeatureSpec(
        node_type="disease",
        features=(
            FeatureColumn("is_rare", "bool", nullable=True,
                          default=False,
                          description="True if classified as rare/orphan disease."),
            FeatureColumn("is_omim", "bool", nullable=True,
                          default=False,
                          description="True if sourced from OMIM (Mendelian)."),
        ),
        description="Disease (DisGeNET/OMIM/DrugBank indication).",
    ),
    "clinical_outcome": NodeFeatureSpec(
        node_type="clinical_outcome",
        features=(
            FeatureColumn("outcome_kind", "string", nullable=True,
                          default="unknown",
                          description="efficacy | adverse_event | mortality | unknown."),
            FeatureColumn("meddra_id", "string", nullable=True,
                          default="",
                          description="MedDRA concept ID (if available)."),
        ),
        description="Clinical outcome (efficacy score, adverse event, etc.).",
    ),
}


# =============================================================================
# EDGE FEATURE SCHEMAS
# =============================================================================
# Per-edge-type feature columns. Phase 3 builds ``data[edge_type].edge_attr``
# from these. Most edge types have only a single ``weight`` feature.

EDGE_FEATURE_SCHEMAS: Dict[Tuple[str, str, str], EdgeFeatureSpec] = {
    ("drug", "inhibits", "protein"): EdgeFeatureSpec(
        edge_type=("drug", "inhibits", "protein"),
        features=(
            FeatureColumn("pchembl_value", "float32", nullable=True,
                          default=0.0,
                          description="-log10(IC50/Ki in M). Higher = more potent."),
            FeatureColumn("confidence", "float32", nullable=True,
                          default=0.5,
                          description="Confidence in the inhibition assertion [0,1]."),
        ),
        description="Drug inhibits protein target.",
    ),
    ("drug", "activates", "protein"): EdgeFeatureSpec(
        edge_type=("drug", "activates", "protein"),
        features=(
            FeatureColumn("pchembl_value", "float32", nullable=True,
                          default=0.0,
                          description="-log10(EC50 in M). Higher = more potent."),
            FeatureColumn("confidence", "float32", nullable=True,
                          default=0.5,
                          description="Confidence in the activation assertion [0,1]."),
        ),
        description="Drug activates protein target.",
    ),
    ("drug", "binds", "protein"): EdgeFeatureSpec(
        edge_type=("drug", "binds", "protein"),
        features=(
            FeatureColumn("confidence", "float32", nullable=True,
                          default=0.5,
                          description="Binding confidence [0,1]."),
        ),
        description="Drug binds protein (neutral, no direction).",
    ),
    ("drug", "modulates", "protein"): EdgeFeatureSpec(
        edge_type=("drug", "modulates", "protein"),
        features=(
            FeatureColumn("confidence", "float32", nullable=True,
                          default=0.5,
                          description="Modulation confidence [0,1]."),
        ),
        description="Drug allosterically modulates protein.",
    ),
    ("drug", "treats", "disease"): EdgeFeatureSpec(
        edge_type=("drug", "treats", "disease"),
        features=(
            FeatureColumn("indication_type", "string", nullable=True,
                          default="approved",
                          description="approved | off_label | investigational."),
            FeatureColumn("confidence", "float32", nullable=True,
                          default=1.0,
                          description="Treatment confidence [0,1]."),
        ),
        description="Drug is approved/used to treat disease.",
    ),
    ("drug", "tested_for", "disease"): EdgeFeatureSpec(
        edge_type=("drug", "tested_for", "disease"),
        features=(
            FeatureColumn("trial_phase", "int64", nullable=True,
                          default=0,
                          description="Clinical trial phase (0-4)."),
            FeatureColumn("confidence", "float32", nullable=True,
                          default=0.3,
                          description="Testing confidence [0,1]."),
        ),
        description="Drug is being tested for disease (clinical trial).",
    ),
    ("drug", "causes", "clinical_outcome"): EdgeFeatureSpec(
        edge_type=("drug", "causes", "clinical_outcome"),
        features=(
            FeatureColumn("frequency", "float32", nullable=True,
                          default=0.0,
                          description="Adverse-event frequency [0,1]."),
            FeatureColumn("severity", "float32", nullable=True,
                          default=0.5,
                          description="Severity score [0,1] (1=death, 0=mild)."),
        ),
        description="Drug causes clinical outcome (adverse event or efficacy).",
    ),
    ("protein", "part_of", "pathway"): EdgeFeatureSpec(
        edge_type=("protein", "part_of", "pathway"),
        features=(
            FeatureColumn("combined_score", "float32", nullable=True,
                          default=0.0,
                          description="STRING combined score [0,1]."),
        ),
        description="Protein is part of biological pathway.",
    ),
    ("pathway", "disrupted_in", "disease"): EdgeFeatureSpec(
        edge_type=("pathway", "disrupted_in", "disease"),
        features=(
            FeatureColumn("confidence", "float32", nullable=True,
                          default=0.5,
                          description="Disruption confidence [0,1]."),
        ),
        description="Pathway is disrupted in disease.",
    ),
}


# =============================================================================
# Convenience accessors
# =============================================================================


def get_node_feature_names(node_type: str) -> List[str]:
    """Return the ordered list of feature column names for ``node_type``."""
    spec = NODE_FEATURE_SCHEMAS[node_type]
    return [f.name for f in spec.features]


def get_edge_feature_names(edge_type: Tuple[str, str, str]) -> List[str]:
    """Return the ordered list of feature column names for ``edge_type``."""
    spec = EDGE_FEATURE_SCHEMAS[edge_type]
    return [f.name for f in spec.features]


def map_phase2_node_to_phase3(phase2_label: str) -> Optional[str]:
    """Map a Phase 2 node label to its Phase 3 canonical type.

    Returns None for intermediate types (Gene, MedDRA_Term) that are
    dropped in the projection. Raises ``KeyError`` for truly unknown
    labels (typo detection — fail-closed).
    """
    if phase2_label not in PHASE2_TO_PHASE3_NODE:
        raise KeyError(
            f"Unknown Phase 2 node label {phase2_label!r}. "
            f"Known labels: {list(ALL_PHASE2_NODE_TYPES)}. "
            f"Either fix the typo or register the new label in "
            f"phase2/contracts/phase2_schema.py:ALL_PHASE2_NODE_TYPES."
        )
    return PHASE2_TO_PHASE3_NODE[phase2_label]


def map_phase2_edge_to_phase3(
    src: str, rel: str, dst: str
) -> Tuple[str, str, str]:
    """Map a Phase 2 edge triple to its Phase 3 canonical triple.

    Raises ``KeyError`` for unknown edge triples (fail-closed — silent
    fallthrough would let a typo'd edge type corrupt the graph).
    """
    key = (src, rel, dst)
    if key not in PHASE2_TO_PHASE3_EDGE:
        raise KeyError(
            f"Unknown Phase 2 edge type {key!r}. "
            f"Known types: {list(PHASE2_TO_PHASE3_EDGE.keys())}. "
            f"Either fix the typo or register the new edge type in "
            f"phase2/contracts/phase2_schema.py:PHASE2_TO_PHASE3_EDGE."
        )
    return PHASE2_TO_PHASE3_EDGE[key]
