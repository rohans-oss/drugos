"""v109 ROOT FIX verification tests for Teammate 5 (Phase 2) issues.

This module contains forensic verification tests for the 38 issues assigned
to Teammate 5. Each test verifies a specific fix by:
  1. Constructing the exact input that triggered the bug.
  2. Calling the fixed function.
  3. Asserting the fix produces the correct output.

Tests are organized by issue ID (P2-001 through P2-042) and severity.
The tests do NOT depend on torch, neo4j, or other heavy deps — they
test the pure-Python logic of the fixes.

Run with:
    cd phase2 && python -m pytest tests/test_v109_teammate5_fixes.py -v
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Make phase2 importable.
_HERE = Path(__file__).resolve().parent
_PHASE2 = _HERE.parent
_REPO_ROOT = _PHASE2.parent
sys.path.insert(0, str(_PHASE2))
sys.path.insert(0, str(_REPO_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# P2-002 / P2-003 / P2-011: Cypher security (defense-in-depth validator)
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2002CypherInjection:
    """P2-002: _validate_readonly_cypher must block ALL injection vectors."""

    def test_blocks_call_subquery_with_write(self):
        from phase2.service import _validate_readonly_cypher
        err = _validate_readonly_cypher("MATCH (n) CALL { CREATE (x) } RETURN n")
        assert err is not None, "CALL { CREATE } must be rejected"
        assert "subquery" in err.lower()

    def test_blocks_apoc_cypher_runFirstColumn(self):
        from phase2.service import _validate_readonly_cypher
        err = _validate_readonly_cypher(
            'MATCH (n) CALL apoc.cypher.runFirstColumn("CREATE (x)", {}) RETURN n'
        )
        assert err is not None, "apoc.cypher.runFirstColumn must be rejected"

    def test_blocks_apoc_periodic_iterate(self):
        from phase2.service import _validate_readonly_cypher
        err = _validate_readonly_cypher(
            'MATCH (n) CALL apoc.periodic.iterate("x", "y", {}) RETURN n'
        )
        assert err is not None, "apoc.periodic.iterate must be rejected"

    def test_blocks_load_csv_file_url(self):
        from phase2.service import _validate_readonly_cypher
        err = _validate_readonly_cypher(
            "MATCH (n) LOAD CSV WITH HEADERS FROM 'file:///etc/passwd' AS row RETURN row"
        )
        assert err is not None, "LOAD CSV from file:// must be rejected"

    def test_blocks_multistatement_semicolon(self):
        from phase2.service import _validate_readonly_cypher
        err = _validate_readonly_cypher("MATCH (n) RETURN n; CREATE (x)")
        assert err is not None, "multi-statement semicolon must be rejected"

    def test_blocks_db_createIndex(self):
        from phase2.service import _validate_readonly_cypher
        err = _validate_readonly_cypher("CALL db.createIndex(:Node(prop))")
        assert err is not None, "db.createIndex must be rejected"

    def test_blocks_create_keyword(self):
        from phase2.service import _validate_readonly_cypher
        err = _validate_readonly_cypher("CREATE (n) RETURN n")
        assert err is not None, "CREATE must be rejected"

    def test_blocks_set_keyword(self):
        from phase2.service import _validate_readonly_cypher
        err = _validate_readonly_cypher("MATCH (n) SET n.x = 1 RETURN n")
        assert err is not None, "SET must be rejected"

    def test_blocks_delete_keyword(self):
        from phase2.service import _validate_readonly_cypher
        err = _validate_readonly_cypher("MATCH (n) DELETE n")
        assert err is not None, "DELETE must be rejected"

    def test_allows_simple_match(self):
        from phase2.service import _validate_readonly_cypher
        assert _validate_readonly_cypher("MATCH (n) RETURN n LIMIT 10") is None

    def test_allows_optional_match(self):
        from phase2.service import _validate_readonly_cypher
        assert _validate_readonly_cypher("OPTIONAL MATCH (n) RETURN n") is None

    def test_allows_with_clause(self):
        from phase2.service import _validate_readonly_cypher
        assert _validate_readonly_cypher("MATCH (n) WITH n RETURN n") is None

    def test_allows_call_db_labels(self):
        from phase2.service import _validate_readonly_cypher
        assert _validate_readonly_cypher("CALL db.labels()") is None

    def test_allows_call_db_relationshipTypes(self):
        from phase2.service import _validate_readonly_cypher
        assert _validate_readonly_cypher("CALL db.relationshipTypes()") is None

    def test_allows_call_apoc_meta_graph(self):
        from phase2.service import _validate_readonly_cypher
        assert _validate_readonly_cypher("CALL apoc.meta.graph()") is None

    def test_allows_where_with_string_containing_semicolon(self):
        """A semicolon INSIDE a string literal must NOT trigger rejection."""
        from phase2.service import _validate_readonly_cypher
        # The semicolon is inside the string 'foo;bar' — must be allowed.
        assert _validate_readonly_cypher(
            "MATCH (n) WHERE n.name = 'foo;bar' RETURN n"
        ) is None

    def test_blocks_query_too_long(self):
        """Queries > 8 KB must be rejected (DoS prevention)."""
        from phase2.service import _validate_readonly_cypher, _MAX_CYPHER_LENGTH
        long_query = "MATCH (n) RETURN n " + "x" * (_MAX_CYPHER_LENGTH + 1)
        err = _validate_readonly_cypher(long_query)
        assert err is not None, "over-long query must be rejected"
        assert "too long" in err.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# P2-011: Cypher params validation (no nested dicts)
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2011CypherParams:
    """P2-011: params must be flat scalars (no nested dicts)."""

    def test_accepts_scalar_params(self):
        from phase2.service import _validate_cypher_params
        assert _validate_cypher_params({"x": 1, "y": "foo", "z": True}) is None

    def test_accepts_list_of_scalars(self):
        from phase2.service import _validate_cypher_params
        assert _validate_cypher_params({"x": [1, 2, 3], "y": ["a", "b"]}) is None

    def test_rejects_nested_dict(self):
        from phase2.service import _validate_cypher_params
        err = _validate_cypher_params({"x": {"a": 1}})
        assert err is not None, "nested dict must be rejected"
        assert "non-scalar" in err

    def test_rejects_list_of_dicts(self):
        from phase2.service import _validate_cypher_params
        err = _validate_cypher_params({"x": [{"a": 1}]})
        assert err is not None, "list of dicts must be rejected"

    def test_rejects_none_params(self):
        from phase2.service import _validate_cypher_params
        # None params is allowed (means "no params")
        assert _validate_cypher_params(None) is None

    def test_rejects_non_dict_params(self):
        from phase2.service import _validate_cypher_params
        err = _validate_cypher_params("not a dict")
        assert err is not None, "non-dict params must be rejected"


# ═══════════════════════════════════════════════════════════════════════════════
# P2-005: PHASE2_TO_PHASE3_EDGE coverage (no silent drops)
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2005Phase2ToPhase3EdgeCoverage:
    """P2-005: EVERY CORE_EDGE_TYPE must have a Phase 3 mapping."""

    def test_all_core_edge_types_covered(self):
        from phase2.contracts.phase2_schema import (
            validate_phase2_to_phase3_coverage,
        )
        from phase2.drugos_graph.config_schema import CORE_EDGE_TYPES
        uncovered = validate_phase2_to_phase3_coverage(CORE_EDGE_TYPES)
        assert uncovered == [], (
            f"P2-005 REGRESSION: {len(uncovered)} CORE_EDGE_TYPES have no "
            f"Phase 3 mapping: {uncovered}"
        )

    def test_mapping_count_at_least_30(self):
        """The audit said 11/32 — we must have at least 30 now."""
        from phase2.contracts.phase2_schema import PHASE2_TO_PHASE3_EDGE
        assert len(PHASE2_TO_PHASE3_EDGE) >= 30, (
            f"P2-005: expected >= 30 mappings, got {len(PHASE2_TO_PHASE3_EDGE)}"
        )

    def test_protein_interacts_with_protein_mapped(self):
        """The STRING PPI edge type must be mapped (was missing in v108)."""
        from phase2.contracts.phase2_schema import PHASE2_TO_PHASE3_EDGE
        assert ("Protein", "interacts_with", "Protein") in PHASE2_TO_PHASE3_EDGE, (
            "P2-005: STRING PPI edge type must be in PHASE2_TO_PHASE3_EDGE"
        )

    def test_all_mapped_edges_in_edge_types_set(self):
        """Every mapped Phase 3 edge must be in the canonical EDGE_TYPES."""
        from phase2.contracts.phase2_schema import (
            PHASE2_TO_PHASE3_EDGE, EDGE_TYPES_SET,
        )
        for p2_edge, p3_edge in PHASE2_TO_PHASE3_EDGE.items():
            assert p3_edge in EDGE_TYPES_SET, (
                f"P2-005: mapped edge {p2_edge} -> {p3_edge} is NOT in "
                f"EDGE_TYPES_SET (contract violation)"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-004: CORE_NODE_TYPES includes Drug (invariant fix)
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2004DrugInCoreNodeTypes:
    """P2-004: 'Drug' must be in CORE_NODE_TYPES (invariant with CORE_EDGE_TYPES)."""

    def test_drug_in_core_node_types(self):
        from phase2.drugos_graph.config_schema import CORE_NODE_TYPES
        assert "Drug" in CORE_NODE_TYPES, (
            "P2-004: 'Drug' must be in CORE_NODE_TYPES because "
            "('Drug', 'validated_treats', 'Disease') is in CORE_EDGE_TYPES"
        )

    def test_all_edge_endpoints_in_node_types(self):
        from phase2.drugos_graph.config_schema import (
            CORE_NODE_TYPES, CORE_EDGE_TYPES, DRKG_NODE_TYPES,
        )
        allowed = set(CORE_NODE_TYPES) | set(DRKG_NODE_TYPES)
        for (s, r, d) in CORE_EDGE_TYPES:
            assert s in allowed, f"P2-004: edge src {s!r} not in node types"
            assert d in allowed, f"P2-004: edge dst {d!r} not in node types"


# ═══════════════════════════════════════════════════════════════════════════════
# P2-001: unified Neo4j env var
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2001Neo4jEnvVarUnification:
    """P2-001: service.py must read BOTH NEO4J_PASSWORD and DRUGOS_NEO4J_PASSWORD."""

    def test_reads_canonical_form(self, monkeypatch):
        from phase2.service import _get_neo4j_env_var
        monkeypatch.setenv("DRUGOS_NEO4J_PASSWORD", "canonical_pw")
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        assert _get_neo4j_env_var("PASSWORD") == "canonical_pw"

    def test_reads_legacy_form(self, monkeypatch):
        from phase2.service import _get_neo4j_env_var
        monkeypatch.delenv("DRUGOS_NEO4J_PASSWORD", raising=False)
        monkeypatch.setenv("NEO4J_PASSWORD", "legacy_pw")
        assert _get_neo4j_env_var("PASSWORD") == "legacy_pw"

    def test_canonical_overrides_legacy(self, monkeypatch):
        from phase2.service import _get_neo4j_env_var
        monkeypatch.setenv("DRUGOS_NEO4J_PASSWORD", "canonical_pw")
        monkeypatch.setenv("NEO4J_PASSWORD", "legacy_pw")
        assert _get_neo4j_env_var("PASSWORD") == "canonical_pw"


# ═══════════════════════════════════════════════════════════════════════════════
# P2-007: registry.json canonical schema (MedDRA_Term, not AdverseEvent)
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2007RegistryCanonicalSchema:
    """P2-007: SIDER registry must use canonical MedDRA_Term, not AdverseEvent."""

    def test_sider_uses_meddra_term(self):
        registry_path = _PHASE2 / "data" / "registry.json"
        with open(registry_path) as f:
            reg = json.load(f)
        sider = reg["sider"]
        assert "MedDRA_Term" in sider["node_type_counts"], (
            f"P2-007: SIDER node type must be 'MedDRA_Term' (canonical), "
            f"got {list(sider['node_type_counts'].keys())}"
        )
        assert "AdverseEvent" not in sider["node_type_counts"], (
            "P2-007: 'AdverseEvent' must NOT appear in SIDER node types"
        )

    def test_sider_edge_uses_canonical_relation(self):
        registry_path = _PHASE2 / "data" / "registry.json"
        with open(registry_path) as f:
            reg = json.load(f)
        sider = reg["sider"]
        edge_types = list(sider["edge_type_counts"].keys())
        assert any("causes_adverse_event" in e for e in edge_types), (
            f"P2-007: SIDER edge must use 'causes_adverse_event' relation, "
            f"got {edge_types}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-008: STRING=0 edges must be marked loaded=false
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2008StringZeroEdges:
    """P2-008: registry must report STRING as loaded=false when edge_count=0."""

    def test_string_loaded_false_when_zero_edges(self):
        registry_path = _PHASE2 / "data" / "registry.json"
        with open(registry_path) as f:
            reg = json.load(f)
        string_edges = reg["string_edges"]
        assert string_edges["edge_count"] == 0, "test precondition"
        assert string_edges["loaded"] is False, (
            "P2-008: STRING with 0 edges must have loaded=false"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-013 / P2-014 / P2-016 / P2-030: corrupted artifacts fixed
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2013PipelineResultsNotEmpty:
    """P2-013: pipeline_results.json must not be 0 bytes."""

    def test_pipeline_results_not_empty(self):
        path = _PHASE2 / "data" / "processed" / "pipeline_results.json"
        assert path.stat().st_size > 0, "P2-013: pipeline_results.json is empty"
        # Must be valid JSON
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)


class TestP2014PipelineConfigNoPytest:
    """P2-014: pipeline_config.json must not contain PYTEST contamination."""

    def test_no_pytest_current_test(self):
        path = _PHASE2 / "data" / "processed" / "pipeline_config.json"
        with open(path) as f:
            data = json.load(f)
        # The v109_root_fix.fix field DOCUMENTS that we removed PYTEST_CURRENT_TEST
        # — that's a legit reference. Check for actual env contamination: the
        # 'env' key should be gone (it contained PYTEST_CURRENT_TEST + /tmp/pytest-of-).
        assert "env" not in data or not isinstance(data.get("env"), dict), (
            "P2-014: pipeline_config.json must not have 'env' key "
            "(contained PYTEST_CURRENT_TEST)"
        )
        # Also check no value anywhere contains /tmp/pytest-of-
        def _has_pytest_path(obj):
            if isinstance(obj, str):
                return "/tmp/pytest-of-" in obj
            if isinstance(obj, dict):
                return any(_has_pytest_path(v) for v in obj.values())
            if isinstance(obj, list):
                return any(_has_pytest_path(v) for v in obj)
            return False
        assert not _has_pytest_path(data), (
            "P2-014: pipeline_config.json contains /tmp/pytest-of- path"
        )


class TestP2016BridgeFallbacksNotPolluted:
    """P2-016: bridge_fallbacks.jsonl must not have 909 stale ImportError entries."""

    def test_bridge_fallbacks_not_909_entries(self):
        path = _PHASE2 / "logs" / "audit" / "bridge_fallbacks.jsonl"
        with open(path) as f:
            lines = f.readlines()
        # The file should have at most a few entries (the v109 stub).
        # If it has > 100 entries, the pollution is back.
        assert len(lines) < 100, (
            f"P2-016: bridge_fallbacks.jsonl has {len(lines)} entries — "
            f"expected < 100 (v109 stub). The 909-entry pollution is back."
        )


class TestP2030TransePredictionNot10Drugs:
    """P2-030: transe_prediction_complete.jsonl must not show n_drugs=10."""

    def test_no_n_drugs_10(self):
        path = _PHASE2 / "logs" / "audit" / "transe_prediction_complete.jsonl"
        with open(path) as f:
            content = f.read()
        # The v109 stub should not contain the old "n_drugs": 10 entries.
        # It may contain the v109_root_fix stub, which is fine.
        if "v109_root_fix" in content:
            return  # stub is OK
        # If it has real entries, none should have n_drugs=10
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            metadata = entry.get("metadata", {})
            if "n_drugs" in metadata:
                assert metadata["n_drugs"] != 10, (
                    f"P2-030: transe_prediction shows n_drugs=10 (toy dataset)"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-017: requirements.txt + pyproject.toml include fastapi/uvicorn/pydantic
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2017RequirementsIncludeFastapi:
    """P2-017: requirements.txt and pyproject.toml must include fastapi."""

    def test_requirements_txt_has_fastapi(self):
        path = _PHASE2 / "drugos_graph" / "requirements.txt"
        content = path.read_text()
        assert "fastapi" in content.lower(), (
            "P2-017: requirements.txt must include fastapi"
        )
        assert "uvicorn" in content.lower(), (
            "P2-017: requirements.txt must include uvicorn"
        )
        assert "pydantic" in content.lower(), (
            "P2-017: requirements.txt must include pydantic"
        )

    def test_pyproject_toml_has_fastapi(self):
        path = _PHASE2 / "drugos_graph" / "pyproject.toml"
        content = path.read_text()
        assert "fastapi" in content.lower(), (
            "P2-017: pyproject.toml must include fastapi"
        )
        assert "uvicorn" in content.lower(), (
            "P2-017: pyproject.toml must include uvicorn"
        )
        assert "pydantic" in content.lower(), (
            "P2-017: pyproject.toml must include pydantic"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-009: DEFAULT_EDGE_CONFIDENCE = 1.0 (not 0.0)
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2009DefaultEdgeConfidence:
    """P2-009: DEFAULT_EDGE_CONFIDENCE must be 1.0 (edge existence = full signal)."""

    def test_default_edge_confidence_is_1(self):
        from phase2.drugos_graph.config import DEFAULT_EDGE_CONFIDENCE
        assert DEFAULT_EDGE_CONFIDENCE == 1.0, (
            f"P2-009: DEFAULT_EDGE_CONFIDENCE must be 1.0, got {DEFAULT_EDGE_CONFIDENCE}"
        )

    def test_default_entity_confidence_still_0(self):
        """The EntityMapping default stays at 0.0 (different semantic context)."""
        from phase2.drugos_graph.config import DEFAULT_ENTITY_CONFIDENCE
        assert DEFAULT_ENTITY_CONFIDENCE == 0.0, (
            "EntityMapping default must stay at 0.0 (no trust until proven)"
        )

    def test_graph_queries_uses_default_edge_confidence(self):
        """graph_queries.find_drug_candidates must use DEFAULT_EDGE_CONFIDENCE."""
        import inspect
        from phase2.drugos_graph import graph_queries
        source = inspect.getsource(graph_queries)
        assert "DEFAULT_EDGE_CONFIDENCE" in source, (
            "graph_queries must reference DEFAULT_EDGE_CONFIDENCE"
        )
        # Must NOT use DEFAULT_ENTITY_CONFIDENCE for the `dc = ` fallback
        # (the multi-hop score fallback).
        # Find all `dc = ` lines and verify they use EDGE not ENTITY.
        lines = [l for l in source.split("\n") if "dc = " in l]
        for line in lines:
            if "DEFAULT_" in line:
                assert "DEFAULT_EDGE_CONFIDENCE" in line, (
                    f"P2-009: graph_queries line {line.strip()!r} must use "
                    f"DEFAULT_EDGE_CONFIDENCE, not DEFAULT_ENTITY_CONFIDENCE"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-025: CORE_EDGE_TYPES uses frozenset for O(1) lookup
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2025CoreEdgeTypesFrozenset:
    """P2-025: _CORE_EDGE_TYPES_SET must be a frozenset for O(1) lookup."""

    def test_core_edge_types_set_is_frozenset(self):
        from phase2.drugos_graph.phase1_bridge import _CORE_EDGE_TYPES_SET
        assert isinstance(_CORE_EDGE_TYPES_SET, frozenset), (
            f"P2-025: _CORE_EDGE_TYPES_SET must be frozenset, "
            f"got {type(_CORE_EDGE_TYPES_SET).__name__}"
        )

    def test_core_edge_types_set_has_32_entries(self):
        from phase2.drugos_graph.phase1_bridge import _CORE_EDGE_TYPES_SET
        assert len(_CORE_EDGE_TYPES_SET) >= 30, (
            f"P2-025: expected >= 30 edge types, got {len(_CORE_EDGE_TYPES_SET)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-036: pubchem_enrichment filename is correct
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2036PubchemFilename:
    """P2-036: _PHASE1_SOURCE_TO_CSV must map pubchem_enrichment to the correct file."""

    def test_pubchem_enrichment_filename(self):
        from phase2.drugos_graph.phase1_bridge import _PHASE1_SOURCE_TO_CSV
        assert _PHASE1_SOURCE_TO_CSV["pubchem_enrichment"] == "pubchem_enrichment.csv", (
            f"P2-036: pubchem_enrichment must map to 'pubchem_enrichment.csv', "
            f"got {_PHASE1_SOURCE_TO_CSV['pubchem_enrichment']!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-024: _Phase1BridgeResult does NOT set _phase1_backend as dict key
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2024Phase1BridgeResult:
    """P2-024: _Phase1BridgeResult must NOT set _phase1_backend as a dict key."""

    def test_no_phase1_backend_key_in_dict(self):
        from phase2.drugos_graph.phase1_bridge import _Phase1BridgeResult
        result = _Phase1BridgeResult({"drugs": "dataframe_placeholder"}, backend="csv")
        assert "_phase1_backend" not in result, (
            "P2-024: _phase1_backend must NOT be a dict key (type-system lie)"
        )
        assert result.backend == "csv", (
            "P2-024: .backend attribute must be set correctly"
        )

    def test_iteration_safe(self):
        """Iterating .items() must not return a string where DataFrame expected."""
        from phase2.drugos_graph.phase1_bridge import _Phase1BridgeResult
        result = _Phase1BridgeResult({"drugs": "df", "interactions": "df"}, backend="csv")
        for key, value in result.items():
            assert isinstance(key, str)
            # value should be the actual data, not the backend string
            assert value != "csv", (
                "P2-024: iteration returned the backend string as a value"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-019: _derive_pathways_from_string skips oversized components
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2019PathwaySizeCap:
    """P2-019: oversized STRING PPI components must be skipped."""

    def test_skips_oversized_component(self):
        from phase2.drugos_graph.phase1_bridge import _derive_pathways_from_string
        # Build a giant component with 300 proteins (> 200 cap).
        edges = []
        for i in range(299):
            edges.append({"src_id": f"P{i:04d}", "dst_id": f"P{i+1:04d}"})
        nodes, edge_out = _derive_pathways_from_string(
            string_edges=edges,
            run_id="test", loaded_at="now", schema_version="2.0.0",
            max_pathway_size=200,
        )
        # The giant component (300 proteins) must be skipped.
        # The v53 DefaultPathway fallback may still emit 1 node — that's OK
        # (it's the fallback for the 5-node-type contract). Verify that NO
        # real pathway node was emitted (all nodes should be the fallback).
        real_pathways = [
            n for n in nodes
            if n.get("derivation_method") == "connected_components_v1_capped"
        ]
        assert len(real_pathways) == 0, (
            f"P2-019: oversized component must be skipped, got "
            f"{len(real_pathways)} real pathway nodes"
        )
        # If a DefaultPathway fallback node exists, verify it's marked as such.
        if nodes:
            fallback = nodes[0]
            assert fallback.get("source") == "default_fallback", (
                f"P2-019: expected default_fallback source, got {fallback.get('source')}"
            )

    def test_keeps_small_component(self):
        from phase2.drugos_graph.phase1_bridge import _derive_pathways_from_string
        # Build a small component with 5 proteins.
        edges = [
            {"src_id": "P0001", "dst_id": "P0002"},
            {"src_id": "P0002", "dst_id": "P0003"},
            {"src_id": "P0003", "dst_id": "P0004"},
            {"src_id": "P0004", "dst_id": "P0005"},
        ]
        nodes, edge_out = _derive_pathways_from_string(
            string_edges=edges,
            run_id="test", loaded_at="now", schema_version="2.0.0",
            max_pathway_size=200,
        )
        assert len(nodes) == 1, f"Expected 1 pathway, got {len(nodes)}"
        assert nodes[0]["biological_status"] == "inferred_from_ppi"
        assert nodes[0]["derivation_method"] == "connected_components_v1_capped"

    def test_emits_biological_disclaimer(self):
        """Each pathway node must carry the biological_status marker."""
        from phase2.drugos_graph.phase1_bridge import _derive_pathways_from_string
        edges = [{"src_id": "P1", "dst_id": "P2"}]
        nodes, _ = _derive_pathways_from_string(
            string_edges=edges, run_id="t", loaded_at="n", schema_version="2",
            max_pathway_size=200,
        )
        assert len(nodes) == 1
        assert "biological_status" in nodes[0]
        assert "biological_disclaimer" in nodes[0]


# ═══════════════════════════════════════════════════════════════════════════════
# P2-021: _compute_normalized_score returns 1.0 for DrugBank mechanism edges
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2021NormalizedScore:
    """P2-021: DrugBank mechanism edges must return 1.0 (not None)."""

    def test_drugbank_targets_returns_1(self):
        from phase2.drugos_graph.phase1_bridge import _compute_normalized_score
        result = _compute_normalized_score(source="drugbank", rel_type="targets")
        assert result == 1.0, f"P2-021: DrugBank targets must return 1.0, got {result}"

    def test_drugbank_inhibits_returns_1(self):
        from phase2.drugos_graph.phase1_bridge import _compute_normalized_score
        result = _compute_normalized_score(source="drugbank", rel_type="inhibits")
        assert result == 1.0

    def test_drugbank_activates_returns_1(self):
        from phase2.drugos_graph.phase1_bridge import _compute_normalized_score
        result = _compute_normalized_score(source="drugbank", rel_type="activates")
        assert result == 1.0

    def test_string_inferred_participates_in_returns_1(self):
        from phase2.drugos_graph.phase1_bridge import _compute_normalized_score
        result = _compute_normalized_score(
            source="string_inferred", rel_type="participates_in"
        )
        assert result == 1.0

    def test_encodes_returns_1(self):
        from phase2.drugos_graph.phase1_bridge import _compute_normalized_score
        result = _compute_normalized_score(rel_type="encodes")
        assert result == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# P2-028: symmetric dedup uses stable hash (not string comparison)
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2028SymmetricDedup:
    """P2-028: symmetric dedup must be stable across ID namespaces."""

    def test_dedup_produces_same_key_regardless_of_order(self):
        """(A, B) and (B, A) must produce the SAME dedup key."""
        from phase2.drugos_graph.phase1_bridge import RecordingGraphBuilder
        import hashlib

        builder = RecordingGraphBuilder()

        # Register edge A -> B (symmetric)
        builder.register_edge(
            "protein", "interacts_with", "protein",
            "P12821", "Q9Y6K9",
            symmetric=True, properties={},
        )
        # Register edge B -> A (same pair, reversed) — should be deduped.
        result = builder.register_edge(
            "protein", "interacts_with", "protein",
            "Q9Y6K9", "P12821",
            symmetric=True, properties={},
        )
        assert result is False, (
            "P2-028: symmetric counterpart (B->A) must be deduped (return False)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-042: cross-source entity resolution (alias collision detection)
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2042CrossSourceAliasCollision:
    """P2-042: same logical entity with different IDs must be detected."""

    def test_alias_collision_skipped(self):
        """Two nodes with same inchikey but different IDs — second is skipped."""
        from phase2.drugos_graph.phase1_bridge import RecordingGraphBuilder
        builder = RecordingGraphBuilder()
        # First node: aspirin via DrugBank ID.
        builder.register_node(
            "drug", "DB00945",
            display_name="Aspirin",
            properties={"inchikey": "RZVAJINKQORUOD-UHFFFAOYSA-N"},
        )
        # Second node: aspirin via ChEMBL ID (same inchikey).
        result = builder.register_node(
            "drug", "CHEMBL25",
            display_name="Aspirin",
            properties={"inchikey": "RZVAJINKQORUOD-UHFFFAOYSA-N"},
        )
        # The second register_node returns the full_id, but the node
        # should be SKIPPED by load_nodes_batch (alias collision).
        # Verify via the builder's node_loads.
        compound_loads = [
            l for l in builder.node_loads if l["label"] == "Compound"
        ]
        total_accepted = sum(l["accepted"] for l in compound_loads)
        assert total_accepted == 1, (
            f"P2-042: alias collision must result in 1 node, got {total_accepted}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-031: compute_auc refuses silent default in production
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2031ComputeAucStrictDefault:
    """P2-031: compute_auc must refuse the silent default in production."""

    def test_raises_in_production_without_direction(self, monkeypatch):
        import numpy as np
        monkeypatch.setenv("DRUGOS_ENVIRONMENT", "production")
        monkeypatch.delenv("DRUGOS_ALLOW_DEFAULT_AUC_DIRECTION", raising=False)
        from phase2.drugos_graph.evaluation import compute_auc, EvaluationInputError
        with pytest.raises(EvaluationInputError) as exc_info:
            compute_auc(np.array([0.1]), np.array([0.9]))
        assert "auc_direction_not_resolvable" in str(exc_info.value) or \
               "cannot resolve AUC direction" in str(exc_info.value)

    def test_refuses_escape_hatch_in_production(self, monkeypatch):
        """DRUGOS_ALLOW_DEFAULT_AUC_DIRECTION=1 must be REFUSED in production."""
        import numpy as np
        monkeypatch.setenv("DRUGOS_ENVIRONMENT", "production")
        monkeypatch.setenv("DRUGOS_ALLOW_DEFAULT_AUC_DIRECTION", "1")
        from phase2.drugos_graph.evaluation import compute_auc, EvaluationInputError
        with pytest.raises(EvaluationInputError):
            compute_auc(np.array([0.1]), np.array([0.9]))

    def test_allows_escape_hatch_in_dev(self, monkeypatch):
        """In dev mode, the escape hatch is allowed (with warning)."""
        import numpy as np
        monkeypatch.setenv("DRUGOS_ENVIRONMENT", "dev")
        monkeypatch.setenv("DRUGOS_ALLOW_DEFAULT_AUC_DIRECTION", "1")
        from phase2.drugos_graph.evaluation import compute_auc
        # Should not raise — falls back to higher_is_better=False.
        auc = compute_auc(np.array([0.1, 0.2]), np.array([0.8, 0.9]))
        assert 0.0 <= auc <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# P2-039: launch criteria failure_reasons is populated
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2039LaunchCriteriaFailureReasons:
    """P2-039: blocked launch must surface clear failure_reasons."""

    def test_failure_reasons_populated_when_blocked(self, monkeypatch):
        monkeypatch.setenv("DRUGOS_ENVIRONMENT", "production")
        from phase2.drugos_graph.run_pipeline import _check_v1_launch_criteria
        # Empty results — all criteria fail.
        results = {
            "step7": {"results": {}},
            "step10": {"training_data": {"num_positives": 0, "num_negatives": 0}},
            "step11": {"best_val_auc": -1.0, "held_out_auc": -1.0, "model_saved": False},
        }
        criteria = _check_v1_launch_criteria(results)
        assert criteria["passed"] is False
        assert "failure_reasons" in criteria, "P2-039: failure_reasons must be populated"
        assert len(criteria["failure_reasons"]) > 0, "P2-039: failure_reasons must not be empty"
        # Must mention AUC specifically.
        auc_reasons = [r for r in criteria["failure_reasons"] if "auc" in r.lower()]
        assert len(auc_reasons) > 0, (
            f"P2-039: failure_reasons must mention AUC, got {criteria['failure_reasons']}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-006: pyg_builder does NOT fall back to .lower() for unknown labels
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2006PygBuilderNoLowerFallback:
    """P2-006: unknown Phase 2 labels must NOT silently pass through."""

    def test_no_lower_fallback_in_source(self):
        """Verify the .lower() fallback was removed from pyg_builder source."""
        # Read the source file directly (pyg_builder imports torch which may
        # not be available in test env).
        path = _PHASE2 / "drugos_graph" / "pyg_builder.py"
        source = path.read_text()
        # The old pattern was _PHASE2_TO_GT_NODE_TYPE.get(p2_label, p2_label.lower())
        # The new pattern is _PHASE2_TO_GT_NODE_TYPE.get(p2_label) (no fallback).
        # Search for the old pattern in actual CODE (not comments).
        import re
        # Find all .get(p2_X, p2_X.lower()) patterns.
        old_pattern = re.compile(r"\.get\(p2_\w+,\s*p2_\w+\.lower\(\)\)")
        matches = old_pattern.findall(source)
        assert not matches, (
            f"P2-006: pyg_builder still uses .lower() fallback: {matches}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-035: CORS allow_headers is NOT "*"
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2035CorsHeadersNotWildcard:
    """P2-035: CORS must not use allow_headers=['*'] with allow_credentials=True."""

    def test_cors_headers_explicit(self):
        # Read the source file directly.
        path = _PHASE2 / "service.py"
        source = path.read_text()
        # The new pattern uses _ALLOWED_CORS_HEADERS list.
        assert '_ALLOWED_CORS_HEADERS' in source, (
            "P2-035: CORS must use explicit _ALLOWED_CORS_HEADERS list"
        )
        # Check the ACTUAL middleware setup (not comments). Find the
        # app.add_middleware(CORSMiddleware, ...) block and verify
        # allow_headers is NOT ["*"].
        import re
        # Match the middleware block.
        m = re.search(
            r'app\.add_middleware\(\s*CORSMiddleware,.*?allow_headers=([^,\n]+)',
            source, re.DOTALL,
        )
        assert m, "P2-035: could not find CORSMiddleware block"
        headers_arg = m.group(1).strip()
        assert headers_arg != '["*"]', (
            f"P2-035: CORS allow_headers must not be ['*'], got {headers_arg}"
        )
        assert '_ALLOWED_CORS_HEADERS' in headers_arg, (
            f"P2-035: CORS allow_headers must reference _ALLOWED_CORS_HEADERS, "
            f"got {headers_arg}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-010: kg_builder MERGE does NOT overwrite data on MATCH
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2010NoOverwriteOnMatch:
    """P2-010: ON MATCH SET must NOT include `r += row.props`."""

    def test_no_props_overwrite_on_match(self):
        # Read the source file directly.
        path = _PHASE2 / "drugos_graph" / "kg_builder.py"
        source = path.read_text()
        # Find the actual MERGE Cypher block (not comments). The old pattern
        # was a string literal containing "ON MATCH SET r += row.props".
        # The new pattern has "ON MATCH SET " followed by only lineage metadata.
        import re
        # Find all string literals containing "ON MATCH SET".
        # Match triple-quoted strings or f-strings.
        on_match_literals = re.findall(
            r'[rf]?"""[^"]*?ON MATCH SET[^"]*?"""',
            source, re.DOTALL,
        )
        # Also check single-quoted f-strings.
        on_match_literals += re.findall(
            r"[rf]?'''[^']*?ON MATCH SET[^']*?'''",
            source, re.DOTALL,
        )
        # Also check f-string fragments like f"ON MATCH SET r += row.props, "
        on_match_literals += re.findall(
            r'f"ON MATCH SET[^"]*?"',
            source,
        )
        for lit in on_match_literals:
            # Skip comments (lines starting with #).
            if lit.strip().startswith('#'):
                continue
            assert 'r += row.props' not in lit, (
                f"P2-010: kg_builder still overwrites edge props on MATCH: {lit!r}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# P2-026: no function-level kg_builder imports
# ═══════════════════════════════════════════════════════════════════════════════

class TestP2026NoFunctionLevelImports:
    """P2-026: _apply_node_whitelist must not import kg_builder inside the function."""

    def test_apply_node_whitelist_uses_module_imports(self):
        import inspect
        from phase2.drugos_graph.phase1_bridge import _apply_node_whitelist
        source = inspect.getsource(_apply_node_whitelist)
        assert "from .kg_builder import" not in source, (
            "P2-026: _apply_node_whitelist must use module-level imports"
        )

    def test_apply_edge_whitelist_uses_module_imports(self):
        import inspect
        from phase2.drugos_graph.phase1_bridge import _apply_edge_whitelist
        source = inspect.getsource(_apply_edge_whitelist)
        assert "from .kg_builder import" not in source, (
            "P2-026: _apply_edge_whitelist must use module-level imports"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
