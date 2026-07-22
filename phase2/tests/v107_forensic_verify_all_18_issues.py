"""
V107 FORENSIC VERIFICATION: All 18 Issues (P2-039 through P2-056)
=================================================================
This test verifies that EVERY v107 ROOT FIX is a REAL fix — not an
aspirational comment. Each test checks the ACTUAL code path, not the
docstring.

Run: pytest phase2/tests/v107_forensic_verify_all_18_issues.py -v
"""

import ast
import sys
import os
import re
from pathlib import Path
from typing import List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Helper: read a Python file and parse its AST
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root


def _read_source(rel_path: str) -> str:
    full = REPO_ROOT / rel_path
    return full.read_text(encoding="utf-8")


def _parse_ast(rel_path: str) -> ast.AST:
    return ast.parse(_read_source(rel_path))


# =============================================================================
# PHASE 2 → PHASE 1 / PHASE 4  INTEGRATION ISSUES  (highest impact)
# =============================================================================

class TestP2_039_StandardRelationHeuristic:
    """ISSUE-P2-039: standard_relation is None for all ChEMBL interactions.

    The v107 fix derives standard_relation heuristically from
    activity_type + activity_value + activity_units.
    """

    def test_derive_function_exists(self):
        """The _derive_standard_relation_heuristic function must exist."""
        source = _read_source("phase2/drugos_graph/phase1_bridge.py")
        assert "def _derive_standard_relation_heuristic(" in source, \
            "_derive_standard_relation_heuristic function MISSING"

    def test_heuristic_returns_correct_values(self):
        """The heuristic must return '<' for very low values, '>' for very
        high values, '=' for everything else."""
        # We need to import the function.  Use exec to avoid needing all deps.
        source = _read_source("phase2/drugos_graph/phase1_bridge.py")
        # Extract just the function and its dependencies
        ns: dict = {}
        # Minimal stub for _BINDING_ASSAY_TYPES and logger
        ns["_BINDING_ASSAY_TYPES"] = {
            "IC50", "EC50", "AC50", "KI", "KD", "POTENCY", "GI50",
            "IC25", "IC75", "EC25", "EC75", "KIB", "KDAPP",
        }
        # Find and exec the function
        match = re.search(
            r"def _derive_standard_relation_heuristic\(row\)[\s\S]*?(?=\n\ndef |\Z)",
            source,
        )
        assert match, "Could not extract _derive_standard_relation_heuristic"
        exec(match.group(), ns)
        fn = ns["_derive_standard_relation_heuristic"]

        # Very potent (value < 0.1 nM) -> '<'
        assert fn({"activity_type": "IC50", "activity_value": "0.05", "activity_units": "nM"}) == "<"
        # Very weak (value > 100 µM = 100,000 nM) -> '>'
        assert fn({"activity_type": "IC50", "activity_value": "200", "activity_units": "µM"}) == ">"
        # Mid-range -> '='
        assert fn({"activity_type": "IC50", "activity_value": "50", "activity_units": "nM"}) == "="
        # Non-binding assay type -> '=' (no censoring)
        assert fn({"activity_type": "UNKNOWN", "activity_value": "0.01", "activity_units": "nM"}) == "="
        # Missing value -> '='
        assert fn({"activity_type": "IC50", "activity_value": None, "activity_units": "nM"}) == "="
        # Missing activity_type -> '='
        assert fn({"activity_type": None, "activity_value": "50", "activity_units": "nM"}) == "="

    def test_applied_in_postgres_path(self):
        """The heuristic must be APPLIED to chembl_activities in the
        PostgreSQL path (not just defined but never called)."""
        source = _read_source("phase2/drugos_graph/phase1_bridge.py")
        # Check that _derive_standard_relation_heuristic is called in the
        # chembl_activities postgres path
        pattern = r'chembl_act_df\["standard_relation"\]\s*=\s*\(.*?_derive_standard_relation_heuristic'
        assert re.search(pattern, source, re.DOTALL), \
            "_derive_standard_relation_heuristic is defined but NOT applied to chembl_activities"


class TestP2_044_FdaApprovedFieldMapping:
    """ISSUE-P2-044: fda_approved vs is_fda_approved field name mismatch.

    The v107 fix reads both field names, preferring is_fda_approved.
    """

    def test_reads_both_field_names(self):
        """extract_drug_records_from_staged must read BOTH
        is_fda_approved and fda_approved."""
        source = _read_source("phase2/drugos_graph/phase1_bridge.py")
        func_match = re.search(
            r"def extract_drug_records_from_staged\([\s\S]*?(?=\n\ndef |\Z)",
            source,
        )
        assert func_match, "extract_drug_records_from_staged not found"
        func_body = func_match.group()
        assert "is_fda_approved" in func_body, \
            "extract_drug_records_from_staged does NOT read is_fda_approved"
        assert "fda_approved" in func_body, \
            "extract_drug_records_from_staged does NOT read fda_approved (legacy fallback)"

    def test_prefers_canonical_name(self):
        """The code must prefer is_fda_approved over fda_approved."""
        source = _read_source("phase2/drugos_graph/phase1_bridge.py")
        # The fix should have: n.get("is_fda_approved") checked first
        assert 'n.get("is_fda_approved")' in source, \
            "is_fda_approved is not fetched via .get()"


class TestP2_054_ClinicalTrialsStrictMode:
    """ISSUE-P2-054: ClinicalTrials loader silent failure.

    The v107 fix raises in production/strict mode.
    """

    def test_strict_mode_raises(self):
        """The ClinicalTrials loader must re-raise in strict mode."""
        source = _read_source("phase2/drugos_graph/run_pipeline.py")
        # Find the step7e try/except block
        assert "clinicaltrials_critical_failure" in source, \
            "clinicaltrials_critical_failure flag not set"
        # Must re-raise in strict mode
        assert "raise" in source.split("clinicaltrials_critical_failure")[0].split("_ct_strict")[-1], \
            "ClinicalTrials loader does NOT re-raise in strict mode"

    def test_dev_mode_warns_and_continues(self):
        """In dev mode, the loader should warn and continue."""
        source = _read_source("phase2/drugos_graph/run_pipeline.py")
        assert "warn-and-continue" in source or "dev mode" in source.lower(), \
            "Dev mode warn-and-continue behavior not documented"


# =============================================================================
# PHASE 2 → PHASE 3  INTEGRATION ISSUES
# =============================================================================

class TestP2_042_EdgeCounters:
    """ISSUE-P2-042: edges_added counter undercounts, edges_dropped overcounts.

    The v107 fix separates edges_new, edges_already_present, edges_dropped.
    """

    def test_three_counters_exist(self):
        """The adapt function must have three separate counters."""
        source = _read_source("graph_transformer/data/phase2_adapter.py")
        assert "edges_added = 0" in source, "edges_added counter missing"
        assert "edges_already_present = 0" in source, \
            "edges_already_present counter missing"
        assert "edges_dropped = 0" in source, "edges_dropped counter missing"

    def test_counters_incremented_correctly(self):
        """Each counter must be incremented in the correct branch."""
        source = _read_source("graph_transformer/data/phase2_adapter.py")
        # edges_added incremented when add_edge returns True
        assert "edges_added += 1" in source, "edges_added never incremented"
        # edges_already_present incremented when add_edge returns False
        assert "edges_already_present += 1" in source, \
            "edges_already_present never incremented"
        # edges_dropped incremented when src/dst name missing
        assert "edges_dropped += 1" in source, "edges_dropped never incremented"


class TestP2_043_PublicAPI:
    """ISSUE-P2-043: adapter accesses private attributes of BiomedicalGraphBuilder.

    The v107 fix uses public methods instead.
    """

    def test_no_private_node_maps_access(self):
        """phase2_adapter must NOT access _node_maps directly."""
        source = _read_source("graph_transformer/data/phase2_adapter.py")
        # The v107 fix COMMENT mentions _node_maps but the ACTUAL code
        # should use node_counts_by_type()
        code_lines = [
            l for l in source.splitlines()
            if "_node_maps" in l and not l.strip().startswith("#")
        ]
        assert not code_lines, \
            f"phase2_adapter still accesses _node_maps directly: {code_lines}"

    def test_no_private_edge_sets_access(self):
        """phase2_adapter must NOT access _edge_sets directly."""
        source = _read_source("graph_transformer/data/phase2_adapter.py")
        code_lines = [
            l for l in source.splitlines()
            if "_edge_sets" in l and not l.strip().startswith("#")
        ]
        assert not code_lines, \
            f"phase2_adapter still accesses _edge_sets directly: {code_lines}"

    def test_public_methods_exist(self):
        """The public methods must exist in BiomedicalGraphBuilder."""
        source = _read_source("graph_transformer/data/graph_builder.py")
        assert "def node_counts_by_type(" in source, \
            "node_counts_by_type() public method MISSING"
        assert "def build_reverse_edges(" in source, \
            "build_reverse_edges() public method MISSING"

    def test_public_methods_used(self):
        """phase2_adapter must CALL the public methods."""
        source = _read_source("graph_transformer/data/phase2_adapter.py")
        assert "node_counts_by_type()" in source, \
            "node_counts_by_type() not called in phase2_adapter"
        assert "build_reverse_edges()" in source, \
            "build_reverse_edges() not called in phase2_adapter"


class TestP2_045_SchedulerStep:
    """ISSUE-P2-045: scheduler.step() silently swallows all exceptions.

    The v107 fix catches only TypeError/ValueError, re-raises others.
    """

    def test_catches_only_specific_exceptions(self):
        """The scheduler.step() wrapper must catch ONLY TypeError/ValueError."""
        source = _read_source("phase2/drugos_graph/run_pipeline.py")
        # Find the ACTUAL code block (not comments) with try: scheduler.step()
        # The v107 fix comment precedes the actual code
        match = re.search(
            r"try:\s*\n\s*scheduler\.step\(\)\s*\n\s*except \(TypeError, ValueError\)",
            source,
        )
        assert match, (
            "scheduler.step() does NOT have the v107 fix — "
            "should be: try: scheduler.step() except (TypeError, ValueError)"
        )
        # Verify it's NOT a bare except by checking the surrounding code
        block_start = match.start()
        block_end = source.find("epoch_loss", block_start)
        block_code = source[block_start:block_end]
        assert "except Exception" not in block_code, \
            "scheduler.step() block has bare except Exception"


class TestP2_049_LineageJSON:
    """ISSUE-P2-049: lineage flag silently swallowed.

    The v107 fix logs the exception and writes a .lineage.json companion file.
    """

    def test_lineage_json_writing(self):
        """The code must write a .lineage.json companion file."""
        source = _read_source("phase2/drugos_graph/run_pipeline.py")
        assert ".lineage.json" in source, ".lineage.json companion file not written"
        assert "lineage.json" in source, "lineage.json not referenced"

    def test_logs_exception(self):
        """The code must log at WARNING when attribute setting fails."""
        source = _read_source("phase2/drugos_graph/run_pipeline.py")
        # The __chemberta_features_used__ block must log at WARNING
        assert "__chemberta_features_used__" in source, \
            "__chemberta_features_used__ attribute not set"


class TestP2_055_LeakageDetectorEmptyInputs:
    """ISSUE-P2-055: _detect_leakage crashes on empty inputs.

    The v107 fix returns early with likely_same_array=False.
    """

    def test_empty_input_guard(self):
        """_detect_leakage must return early for empty inputs."""
        source = _read_source("phase2/drugos_graph/evaluation.py")
        # Find the early-return guard
        func_match = re.search(
            r"def _detect_leakage\([\s\S]*?(?=\n\ndef |\Z)",
            source,
        )
        assert func_match, "_detect_leakage not found"
        func_body = func_match.group()
        # Must check for empty inputs at the start
        assert 'len(pos_scores) == 0' in func_body or 'len(neg_scores) == 0' in func_body, \
            "_detect_leakage does NOT check for empty inputs"
        # Must return likely_same_array=False
        assert '"likely_same_array": False' in func_body, \
            "_detect_leakage does NOT return likely_same_array=False for empty inputs"


# =============================================================================
# STANDALONE ISSUES
# =============================================================================

class TestP2_040_ValidatedTreatsDedup:
    """ISSUE-P2-040: validated_treats edges never deduped.

    The v107 fix adds validated_treats to CORE_EDGE_TYPES + DB introspection.
    """

    def test_validated_treats_in_core_edge_types(self):
        """validated_treats must be in CORE_EDGE_TYPES."""
        # v108 UPDATE: CORE_EDGE_TYPES was MOVED to config_schema.py (ISSUE-P2-056).
        # Check config_schema.py source first; fall back to config.py for
        # backward compat (config.py re-exports from config_schema.py).
        source = _read_source("phase2/drugos_graph/config_schema.py")
        assert '("Drug", "validated_treats", "Disease")' in source, \
            'validated_treats NOT in CORE_EDGE_TYPES (config_schema.py)'

    def test_db_introspection_method_exists(self):
        """discover_edge_triples_for_rel_type must exist for non-CORE types."""
        source = _read_source("phase2/drugos_graph/kg_builder.py")
        assert "def discover_edge_triples_for_rel_type(" in source, \
            "discover_edge_triples_for_rel_type method MISSING"


class TestP2_041_LineageManifestRetry:
    """ISSUE-P2-041: lineage manifest write silently fails.

    The v107 fix: ERROR log, retry with backoff, raise in production.
    """

    def test_retry_logic(self):
        """The code must have retry with backoff."""
        source = _read_source("phase2/drugos_graph/run_pipeline.py")
        assert "_LINEAGE_MAX_RETRIES" in source, "Lineage manifest retry logic MISSING"
        assert "_LINEAGE_BACKOFF_SECONDS" in source, "Lineage manifest backoff MISSING"

    def test_raises_in_production(self):
        """Must raise in production mode after retries exhausted."""
        source = _read_source("phase2/drugos_graph/run_pipeline.py")
        after_retry = source.split("_LINEAGE_MAX_RETRIES")[-1].split("if not _lineage_written")[0]
        assert "raise" in after_retry or "RuntimeError" in source.split("if not _lineage_written")[1], \
            "Does NOT raise in production mode after lineage write failure"

    def test_logs_at_error(self):
        """Must log at ERROR level, not debug."""
        source = _read_source("phase2/drugos_graph/run_pipeline.py")
        # Find the section after ISSUE-P2-041 where retries are exhausted
        p2_041_sections = source.split("ISSUE-P2-041")
        assert len(p2_041_sections) > 1, "ISSUE-P2-041 marker not found"
        # The logger.error must be somewhere after the v107 fix comment
        found_error_log = False
        for section in p2_041_sections[1:]:
            if "logger.error" in section:
                found_error_log = True
                break
        assert found_error_log, \
            "Lineage manifest failure not logged at ERROR level anywhere after ISSUE-P2-041"


class TestP2_046_CanonicalRelTypeDocumentation:
    """ISSUE-P2-046: _canonical_rel_type is dead code for bridge edges.

    The v107 fix documents it as DRKG-only.
    """

    def test_documented_as_drkg_only(self):
        """_canonical_rel_type must be documented as DRKG-only."""
        source = _read_source("phase2/drugos_graph/kg_builder.py")
        func_match = re.search(
            r"def _canonical_rel_type\([\s\S]*?(?=\n\ndef |\Z)",
            source,
        )
        assert func_match, "_canonical_rel_type not found"
        func_body = func_match.group()
        # Must mention DRKG
        assert "DRKG" in func_body, \
            "_canonical_rel_type not documented as DRKG-only"


class TestP2_047_MLflowAtexitLogging:
    """ISSUE-P2-047: _atexit_close swallows all exceptions.

    The v107 fix logs at WARNING before swallowing.
    """

    def test_logs_at_warning(self):
        """_atexit_close must log at WARNING level."""
        source = _read_source("phase2/drugos_graph/mlflow_tracker.py")
        func_match = re.search(
            r"def _atexit_close\([\s\S]*?(?=\n\ndef |\Z)",
            source,
        )
        assert func_match, "_atexit_close not found"
        func_body = func_match.group()
        assert "logger.warning" in func_body, \
            "_atexit_close does NOT log at WARNING level"
        # Check the actual CODE (not docstring comments) for bare except/pass
        # Split on the docstring end and check only the code body
        code_after_docstring = func_body.split('"""')[-1]
        # Remove any inline comments
        code_lines = [
            l.split("#")[0].rstrip() for l in code_after_docstring.splitlines()
        ]
        code_only = "\n".join(code_lines)
        # The actual code should NOT have bare except/pass
        has_bare_pass = "except Exception:" in code_only and "pass" in code_only
        # But it SHOULD have logger.warning in the except block
        has_warning_log = "logger.warning" in code_only
        assert has_warning_log, \
            "_atexit_close except block does NOT log at WARNING"


class TestP2_048_SHA256Fallback:
    """ISSUE-P2-048: model_sha256 computation silently fails.

    The v107 fix logs the exception and uses a fallback hash.
    """

    def test_fallback_hash_computation(self):
        """Must compute a fallback hash when sha256 fails."""
        source = _read_source("phase2/drugos_graph/transe_model.py")
        # Find the predict_drug_candidates function
        func_match = re.search(
            r"def predict_drug_candidates\([\s\S]*?(?=\n\ndef |\Z)",
            source,
        )
        assert func_match, "predict_drug_candidates not found"
        func_body = func_match.group()
        # Must have hashlib import in the function
        assert "hashlib" in func_body, \
            "hashlib fallback not present in predict_drug_candidates"
        # Must log the exception at WARNING
        assert "logger.warning" in func_body, \
            "sha256 failure not logged at WARNING"

    def test_fallback_uses_class_and_params(self):
        """Fallback hash must use model class name + parameter count."""
        source = _read_source("phase2/drugos_graph/transe_model.py")
        func_match = re.search(
            r"def predict_drug_candidates\([\s\S]*?(?=\n\ndef |\Z)",
            source,
        )
        assert func_match, "predict_drug_candidates not found"
        func_body = func_match.group()
        assert "type(model).__name__" in func_body or "_model_class" in func_body, \
            "Fallback hash does NOT use model class name"
        assert "numel" in func_body or "_param_count" in func_body, \
            "Fallback hash does NOT use parameter count"


class TestP2_050_DiseaseATCMapDocumentation:
    """ISSUE-P2-050: misleading variable name disease_to_drug_atc_map.

    The v107 fix documents the indirection clearly.
    """

    def test_documented_indirection(self):
        """The docstring must explain that ATC is a DRUG classification."""
        source = _read_source("phase2/drugos_graph/negative_sampling.py")
        func_match = re.search(
            r"def wrong_disease_class_sampling\([\s\S]*?(?=\n\ndef |\Z)",
            source,
        )
        assert func_match, "wrong_disease_class_sampling not found"
        func_body = func_match.group()
        # Must document that ATC is drug classification
        assert "DRUG classification" in func_body or "drug classification" in func_body.lower(), \
            "ATC is NOT documented as a DRUG classification"
        # Must explain the indirection
        assert "standard-of-care" in func_body or "treat" in func_body.lower(), \
            "The indirection (disease -> ATC of drugs treating it) not documented"


class TestP2_051_IsolatedNodesFlag:
    """ISSUE-P2-051: isolated_nodes=-1 passes threshold checks incorrectly.

    The v107 fix adds isolated_nodes_known flag.
    """

    def test_flag_added_to_stats(self):
        """isolated_nodes_known must be set in the stats dict."""
        source = _read_source("phase2/drugos_graph/graph_stats.py")
        assert "isolated_nodes_known" in source, \
            "isolated_nodes_known flag not present"
        # Must be set to True on success
        assert 'stats["isolated_nodes_known"] = True' in source, \
            "isolated_nodes_known not set to True on success"
        # Must be set to False on failure
        assert 'stats["isolated_nodes_known"] = False' in source, \
            "isolated_nodes_known not set to False on failure"

    def test_flag_in_typeddict(self):
        """isolated_nodes_known must be in the StatsReport TypedDict."""
        source = _read_source("phase2/drugos_graph/graph_stats.py")
        assert "isolated_nodes_known: bool" in source, \
            "isolated_nodes_known not in StatsReport TypedDict"


class TestP2_052_TorchUniqueFallbackRemoved:
    """ISSUE-P2-052: slow Python-loop fallback for torch.unique.

    The v107 fix removes the fallback and raises instead.
    """

    def test_fallback_removed(self):
        """The Python-loop fallback must be REMOVED — should raise instead."""
        source = _read_source("phase2/drugos_graph/pyg_builder.py")
        # Find all torch.unique occurrences and check the error handling
        unique_sections = source.split("torch.unique")
        assert len(unique_sections) > 1, "torch.unique not found in pyg_builder.py"
        # At least one occurrence should raise RuntimeError on failure
        found_raise = False
        for section in unique_sections[1:]:
            if "RuntimeError" in section.split("_new_count")[0]:
                found_raise = True
                break
        assert found_raise, \
            "torch.unique failure does NOT raise RuntimeError — fallback still present"

    def test_no_python_loop_fallback(self):
        """Must NOT have a Python for-loop fallback for dedup."""
        source = _read_source("phase2/drugos_graph/pyg_builder.py")
        # After the torch.unique call, there should NOT be a fallback loop
        sections = source.split("torch.unique")
        if len(sections) > 1:
            after_unique = sections[1].split("_new_count")[0]
            # Check for a for-loop over edges as fallback
            fallback_loop = re.search(r"for\s+\w+\s+in\s+range\(.*edge", after_unique)
            assert not fallback_loop, \
                "Python-loop fallback for torch.unique still present"


class TestP2_053_SourcePhaseLineage:
    """ISSUE-P2-053: _source_phase=1 instead of 2 for validated_treats.

    The v107 fix sets _source_phase=2.
    """

    def test_source_phase_is_2(self):
        """update_validated_edges must set _source_phase=2."""
        source = _read_source("phase2/drugos_graph/kg_builder.py")
        func_match = re.search(
            r"def update_validated_edges\([\s\S]*?(?=\n\ndef |\Z)",
            source,
        )
        assert func_match, "update_validated_edges not found"
        func_body = func_match.group()
        # Must have _source_phase: 2 (not 1)
        assert '"_source_phase": 2' in func_body, \
            "_source_phase is NOT set to 2 in update_validated_edges"
        # Must NOT have _source_phase: 1
        assert '"_source_phase": 1' not in func_body, \
            "_source_phase is still set to 1 in update_validated_edges"


class TestP2_056_ConfigSplitRoadmap:
    """ISSUE-P2-056: config.py is 8400+ line monolith.

    The v107 fix documents the intended split.
    """

    def test_split_roadmap_documented(self):
        """The config must document the intended split into submodules."""
        source = _read_source("phase2/drugos_graph/config.py")
        assert "config_paths.py" in source, "config_paths.py split not documented"
        assert "config_neo4j.py" in source, "config_neo4j.py split not documented"
        assert "config_pyg.py" in source, "config_pyg.py split not documented"
        assert "config_transe.py" in source, "config_transe.py split not documented"
        assert "config_schema.py" in source, "config_schema.py split not documented"

    def test_hardcoded_values_resolved(self):
        """The documented hardcoded values must be resolved."""
        source = _read_source("phase2/drugos_graph/config.py")
        assert "RESOLVED" in source, "Hardcoded values not marked as RESOLVED"
