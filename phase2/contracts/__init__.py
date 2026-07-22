"""Phase 2 contracts package — the SINGLE source of truth for Phase 2 schema.

Tasks 322, 323, 331 ROOT FIX (forensic, root-level):
  This package defines the canonical Phase 2 knowledge-graph schema:
    - Node types (Compound, Protein, Pathway, Disease, ClinicalOutcome, + intermediates)
    - Edge types (treats, inhibits, activates, etc.)
    - Node feature schemas
    - Edge feature schemas
    - RecordingGraphBuilder serialization format

  Both Phase 2 (writer: ``drugos_graph.schema_mappings``,
  ``drugos_graph.pyg_builder``, ``drugos_graph.kg_builder``) and
  Phase 3 (reader: ``graph_transformer.data.phase2_adapter``) import
  from this package — eliminating the dual-mapping drift that
  previously caused pyg_builder's 7-entry mapping to silently disagree
  with phase2_adapter's 5-entry mapping on the same source data.

Public API
----------
- :data:`NODE_TYPES` — canonical Phase 3 node types (5 types: drug, protein,
  pathway, disease, clinical_outcome).
- :data:`ALL_PHASE2_NODE_TYPES` — full Phase 2 vocabulary (7 types, includes
  Gene and MedDRA_Term intermediates used for derivation only).
- :data:`PHASE2_TO_PHASE3_NODE` — mapping Phase 2 (capitalized) -> Phase 3
  (lowercase canonical). Intermediate types map to None (dropped).
- :data:`EDGE_TYPES` — canonical Phase 3 edge type triples.
- :data:`PHASE2_TO_PHASE3_EDGE` — mapping Phase 2 edge triples ->
  Phase 3 edge triples.
- :data:`NODE_FEATURE_SCHEMAS` — per-node-type feature column specs.
- :data:`EDGE_FEATURE_SCHEMAS` — per-edge-type feature column specs.
"""
from __future__ import annotations

from phase2.contracts.phase2_schema import (
    NODE_TYPES,
    ALL_PHASE2_NODE_TYPES,
    ALL_PHASE3_NODE_TYPES,
    PHASE2_TO_PHASE3_NODE,
    PHASE3_TO_PHASE2_NODE,
    EDGE_TYPES,
    PHASE2_TO_PHASE3_EDGE,
    PHASE3_TO_PHASE2_EDGE,
    NODE_FEATURE_SCHEMAS,
    EDGE_FEATURE_SCHEMAS,
    INTERMEDIATE_NODE_TYPES,
    is_intermediate_node_type,
    NodeFeatureSpec,
    EdgeFeatureSpec,
    FeatureColumn,
)
from phase2.contracts.kg_builder_contract import (
    RECORDING_GRAPH_BUILDER_FORMAT_VERSION,
    RECORDING_GRAPH_BUILDER_SUPPORTED_FORMATS,
    RECORDING_GRAPH_BUILDER_SNAPSHOT_KEYS,
    validate_recording_graph_builder_snapshot,
)

__all__ = [
    # Node types
    "NODE_TYPES",
    "ALL_PHASE2_NODE_TYPES",
    "ALL_PHASE3_NODE_TYPES",
    "PHASE2_TO_PHASE3_NODE",
    "PHASE3_TO_PHASE2_NODE",
    "INTERMEDIATE_NODE_TYPES",
    "is_intermediate_node_type",
    # Edge types
    "EDGE_TYPES",
    "PHASE2_TO_PHASE3_EDGE",
    "PHASE3_TO_PHASE2_EDGE",
    # Feature schemas
    "NODE_FEATURE_SCHEMAS",
    "EDGE_FEATURE_SCHEMAS",
    "NodeFeatureSpec",
    "EdgeFeatureSpec",
    "FeatureColumn",
    # KG builder contract
    "RECORDING_GRAPH_BUILDER_FORMAT_VERSION",
    "RECORDING_GRAPH_BUILDER_SUPPORTED_FORMATS",
    "RECORDING_GRAPH_BUILDER_SNAPSHOT_KEYS",
    "validate_recording_graph_builder_snapshot",
]
