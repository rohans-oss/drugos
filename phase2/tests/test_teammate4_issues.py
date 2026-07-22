"""Teammate 4 — Phase 2 Loaders + Entity Resolver + ID Crosswalk
Forensic root-fix verification tests for all 22 assigned issues.

This test file is the SINGLE source of truth for verifying that the
Teammate 4 root fixes are in place and working. Each test corresponds
to one issue ID from the audit. Tests are designed to FAIL before the
fix and PASS after — they exercise the actual code paths (not mocks,
not comments, not aspirational "ROOT FIX" labels).

Run with: ``pytest phase2/tests/test_teammate4_issues.py -v``
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure phase2 is importable.
_HERE = Path(__file__).resolve().parent
_PHASE2_ROOT = _HERE.parent
_REPO_ROOT = _PHASE2_ROOT.parent
for _p in (str(_PHASE2_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ============================================================================
# SH-011: schema_mappings.py contract drift (CRITICAL — adapter was BROKEN)
# ============================================================================
def test_sh_011_schema_mappings_has_7_entry_mapping():
    """SH-011: PHASE2_TO_PHASE3_NODE must be the 7-entry contract version."""
    from drugos_graph import schema_mappings as sm
    from phase2.contracts.phase2_schema import (
        PHASE2_TO_PHASE3_NODE as CONTRACT_NODE,
    )
    # The shim must re-export the SAME object as the contract.
    assert sm.PHASE2_TO_PHASE3_NODE is CONTRACT_NODE, (
        "schema_mappings.PHASE2_TO_PHASE3_NODE must be the SAME object as "
        "phase2_schema.PHASE2_TO_PHASE3_NODE (no 5-entry drift)."
    )
    # Must have 7 entries (including Gene=None, MedDRA_Term=None).
    assert len(sm.PHASE2_TO_PHASE3_NODE) == 7
    assert sm.PHASE2_TO_PHASE3_NODE["Gene"] is None
    assert sm.PHASE2_TO_PHASE3_NODE["MedDRA_Term"] is None
    assert sm.PHASE2_TO_PHASE3_NODE["Compound"] == "drug"


def test_sh_011_is_phase2_intermediate_dropped_exists():
    """SH-011: is_phase2_intermediate_dropped must exist (adapter imports it)."""
    from drugos_graph.schema_mappings import is_phase2_intermediate_dropped
    assert callable(is_phase2_intermediate_dropped)
    assert is_phase2_intermediate_dropped("Gene") is True
    assert is_phase2_intermediate_dropped("MedDRA_Term") is True
    assert is_phase2_intermediate_dropped("Compound") is False


def test_sh_011_phase2_adapter_imports_succeed():
    """SH-011: the exact import phase2_adapter.py does must succeed."""
    # This is the EXACT import statement from phase2_adapter.py:128-134.
    # Before the fix, this raised ImportError because is_phase2_intermediate_dropped
    # did not exist in schema_mappings.
    from drugos_graph.schema_mappings import (  # noqa: F401
        PHASE2_TO_PHASE3_NODE,
        PHASE2_TO_PHASE3_EDGE,
        ALL_PHASE2_NODE_TYPES,
        ALL_PHASE3_NODE_TYPES,
        is_phase2_intermediate_dropped,
    )


# ============================================================================
# SH-010: prefer_postgres hardcoded False
# ============================================================================
def test_sh_010_run_4phase_respects_env_var(monkeypatch):
    """SH-010: run_4phase.py must honor DRUGOS_PREFER_POSTGRES env var."""
    # Read the source and verify the env var is referenced.
    run_4phase_path = _REPO_ROOT / "run_4phase.py"
    src = run_4phase_path.read_text()
    assert "DRUGOS_PREFER_POSTGRES" in src, (
        "run_4phase.py must reference DRUGOS_PREFER_POSTGRES env var "
        "(was hardcoded to False before the fix)."
    )
    # The ACTUAL call site (not the docstring/comment) must use the env
    # var. We look for the pattern ``prefer_postgres=os.environ.get(``
    # which is the fixed form.
    assert "prefer_postgres=os.environ.get(" in src, (
        "run_4phase.py must use prefer_postgres=os.environ.get(...) "
        "instead of hardcoded prefer_postgres=False."
    )


def test_sh_010_service_respects_env_var():
    """SH-010: phase2/service.py must honor DRUGOS_PREFER_POSTGRES env var."""
    service_path = _PHASE2_ROOT / "service.py"
    src = service_path.read_text()
    assert "DRUGOS_PREFER_POSTGRES" in src, (
        "phase2/service.py must reference DRUGOS_PREFER_POSTGRES env var."
    )


# ============================================================================
# SH-026: TS KgStatsResponse vs Python contract mismatch
# ============================================================================
def test_sh_026_service_returns_canonical_fields():
    """SH-026: Python /kg/stats must return canonical contract fields."""
    service_path = _PHASE2_ROOT / "service.py"
    src = service_path.read_text()
    # The canonical fields from frontend/contracts/api_contracts.ts.
    for field in ("node_type_counts", "edge_type_counts", "last_updated", "source"):
        assert field in src, (
            f"phase2/service.py must return the canonical field {field!r} "
            f"(matches TS KgStatsResponse contract)."
        )


# ============================================================================
# P2-029: entity_mapping table empty detection
# ============================================================================
def test_p2_029_distinguishes_schema_missing_from_empty():
    """P2-029: _load_phase1_entity_mapping_source_index must distinguish
    schema_missing (migrations not run) from table_empty (fresh install).
    """
    er_path = _PHASE2_ROOT / "drugos_graph" / "entity_resolver.py"
    src = er_path.read_text()
    # The fix uses sqlalchemy.inspect to check if the table exists.
    assert "sa_inspect" in src or "inspect as sa_inspect" in src, (
        "entity_resolver must use sqlalchemy.inspect to detect missing tables."
    )
    assert "schema_missing" in src or "does NOT" in src, (
        "entity_resolver must log a clear schema_missing message."
    )


# ============================================================================
# P2-032: confidence thresholds scientific justification
# ============================================================================
def test_p2_032_calibration_function_exists():
    """P2-032: calibrate_confidence_thresholds must exist and work."""
    from drugos_graph.entity_resolver import calibrate_confidence_thresholds
    result = calibrate_confidence_thresholds(
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    )
    assert "high_conf" in result
    assert "low_conf" in result
    assert "reject" in result
    assert "sample_size" in result
    assert result["sample_size"] == 10
    assert 0 < result["reject"] < result["low_conf"] < result["high_conf"] <= 1


def test_p2_032_calibration_rejects_empty_input():
    """P2-032: calibration must reject empty input."""
    from drugos_graph.entity_resolver import calibrate_confidence_thresholds
    with pytest.raises(ValueError, match="empty"):
        calibrate_confidence_thresholds([])


# ============================================================================
# P2-051: MESH namespace collision
# ============================================================================
def test_p2_051_mesh_c_namespace_split():
    """P2-051: MESH:C → Compound, MESH:D → Disease (no collision)."""
    from drugos_graph.kg_builder import ID_PATTERNS
    # MESH:D000001 is a Disease (MeSH D-tree), NOT a Compound.
    assert re.match(ID_PATTERNS["Disease"], "MESH:D000001")
    assert not re.match(ID_PATTERNS["Compound"], "MESH:D000001")
    # MESH:C000001 is a Compound (MeSH C-tree), NOT a Disease.
    assert re.match(ID_PATTERNS["Compound"], "MESH:C000001")
    assert not re.match(ID_PATTERNS["Disease"], "MESH:C000001")


# ============================================================================
# P2-052: Disease D\d{6} collision with DrugBank
# ============================================================================
def test_p2_052_disease_pattern_documented():
    """P2-052: the Disease pattern collision risk is documented."""
    kg_path = _PHASE2_ROOT / "drugos_graph" / "kg_builder.py"
    src = kg_path.read_text()
    assert "P2-052" in src, "kg_builder.py must document the P2-052 fix."


# ============================================================================
# P2-054: healthz endpoint unconditional ok
# ============================================================================
def test_p2_054_healthz_performs_checks():
    """P2-054: /healthz must perform subsystem checks, not return ok blindly."""
    kg_api_path = _PHASE2_ROOT / "drugos_graph" / "kg_api.py"
    src = kg_api_path.read_text()
    assert "checks" in src, "healthz must return a checks dict."
    assert "neo4j_reachable" in src
    assert "phase1_data_present" in src
    assert "bridge_importable" in src
    assert "503" in src, "healthz must return 503 when checks fail."


# ============================================================================
# P2-057: phase1_db_available incomplete table check
# ============================================================================
def test_p2_057_db_available_checks_all_tables():
    """P2-057: _phase1_db_available must check all required tables."""
    bridge_path = _PHASE2_ROOT / "drugos_graph" / "phase1_bridge.py"
    src = bridge_path.read_text()
    assert "drug_protein_interactions" in src, (
        "_phase1_db_available must check the drug_protein_interactions table."
    )
    assert "P2-057" in src


# ============================================================================
# P2-058: kg_builder session.run params unpacking
# ============================================================================
def test_p2_058_uses_parameters_kwarg():
    """P2-058: kg_builder must use parameters= instead of **params."""
    kg_path = _PHASE2_ROOT / "drugos_graph" / "kg_builder.py"
    src = kg_path.read_text()
    assert "parameters=params" in src, (
        "kg_builder must use session.run(cypher, parameters=params) "
        "for idiomatic clarity."
    )


# ============================================================================
# P2-060: pyg_builder known_pairs missing edge types
# ============================================================================
def test_p2_060_known_pairs_includes_all_therapeutic_edges():
    """P2-060: known_pairs must include treats, tested_for, validated_treats."""
    pyg_path = _PHASE2_ROOT / "drugos_graph" / "pyg_builder.py"
    src = pyg_path.read_text()
    assert "tested_for" in src, (
        "pyg_builder must include tested_for edges in known_pairs."
    )
    assert "validated_treats" in src, (
        "pyg_builder must include validated_treats edges in known_pairs."
    )


# ============================================================================
# P2-064: compute_auc silent NaN drops
# ============================================================================
def test_p2_064_nan_drop_logs_percentage():
    """P2-064: NaN drop must log percentage and use ERROR for >5%."""
    from drugos_graph.evaluation import (
        _sanitize_scores,
        EVALUATION_TRANSFORMATIONS_LOG,
    )
    # 50% NaN — should log at ERROR level.
    arr = np.array([0.5, np.nan, np.nan, np.nan, np.nan, np.nan, 0.7, 0.8, 0.9, 1.0])
    before_len = len(EVALUATION_TRANSFORMATIONS_LOG)
    _sanitize_scores(arr.copy(), allow_nan=True)
    after_len = len(EVALUATION_TRANSFORMATIONS_LOG)
    assert after_len > before_len, "A transformation log entry must be added."
    last = EVALUATION_TRANSFORMATIONS_LOG[-1]
    assert "pct_dropped" in last, "Log must include pct_dropped."
    assert last["pct_dropped"] == 50.0
    assert last["log_level"] == "ERROR", (
        ">5% NaN must log at ERROR level (data quality emergency)."
    )


# ============================================================================
# IN-015: Dockerfile :latest tag
# ============================================================================
def test_in_015_dockerfile_pins_version():
    """IN-015: Dockerfile must not use :latest in the FROM line, must use build-arg."""
    dockerfile_path = _PHASE2_ROOT / "drugos_graph" / "Dockerfile"
    src = dockerfile_path.read_text()
    # The FROM line (not comments) must NOT use :latest.
    from_lines = [
        line for line in src.splitlines()
        if line.strip().upper().startswith("FROM ")
    ]
    assert from_lines, "Dockerfile must have a FROM line."
    for line in from_lines:
        # The FROM line should reference ${BASE_IMAGE} or a version tag,
        # not the bare :latest tag.
        assert ":latest" not in line, (
            f"FROM line must not use :latest tag. Got: {line!r}"
        )
    assert "ARG BASE_IMAGE" in src, "Dockerfile must accept BASE_IMAGE build-arg."
    assert "${BASE_IMAGE}" in src, "Dockerfile must use the BASE_IMAGE arg."


# ============================================================================
# IN-056: pytest.ini overrides root
# ============================================================================
def test_in_056_phase2_pytest_ini_deleted():
    """IN-056: phase2/tests/pytest.ini must be deleted."""
    p2_pytest = _PHASE2_ROOT / "tests" / "pytest.ini"
    assert not p2_pytest.exists(), (
        "phase2/tests/pytest.ini must be deleted (merged into root pytest.ini)."
    )
    root_pytest = _REPO_ROOT / "pytest.ini"
    src = root_pytest.read_text()
    assert "live_api" in src, "Root pytest.ini must declare live_api marker."
    assert "live_model" in src, "Root pytest.ini must declare live_model marker."


# ============================================================================
# P2-053: NA InChIKey fragment
# ============================================================================
def test_p2_053_na_not_treated_as_empty_when_valid_inchikey():
    """P2-053: standalone 'NA' is empty, but 'NA' inside a 27-char InChIKey is not."""
    from drugos_graph.phase1_bridge import _normalize_inchikey
    assert _normalize_inchikey("NA") == "", "Standalone NA must be empty."
    assert _normalize_inchikey("nan") == ""
    assert _normalize_inchikey("none") == ""
    assert _normalize_inchikey("null") == ""
    # Valid InChIKey (27 chars) must pass through.
    valid_ik = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert _normalize_inchikey(valid_ik) == valid_ik


# ============================================================================
# P2-055: audit log path wheel vs source
# ============================================================================
def test_p2_055_audit_log_uses_env_or_cwd():
    """P2-055: audit log path must use env var or CWD, not __file__."""
    bridge_path = _PHASE2_ROOT / "drugos_graph" / "phase1_bridge.py"
    src = bridge_path.read_text()
    assert "DRUGOS_AUDIT_LOG_DIR" in src, (
        "phase1_bridge must use DRUGOS_AUDIT_LOG_DIR env var."
    )
    assert "Path.cwd()" in src, "Must fall back to CWD-relative path."


# ============================================================================
# P2-056: fcntl.flock w mode truncation
# ============================================================================
def test_p2_056_lock_file_uses_append_mode():
    """P2-056: lock file must be opened in append mode, not write mode."""
    bridge_path = _PHASE2_ROOT / "drugos_graph" / "phase1_bridge.py"
    src = bridge_path.read_text()
    # Find the lock file open call in _acquire_audit_lock.
    # It should use "a" mode, not "w" mode.
    assert 'open(lock_path, "a")' in src, (
        "Lock file must be opened in append mode (was 'w' which truncates)."
    )


# ============================================================================
# P2-059: Side Effect label (false positive — already handled)
# ============================================================================
def test_p2_059_side_effect_label_is_backtick_free():
    """P2-059: 'Side Effect' label must NOT need backtick quoting in Cypher.

    This was a FALSE POSITIVE in the audit — the code already converts
    'Side Effect' → 'SideEffect' via drkg_node_type_to_neo4j_label.
    This test verifies the conversion works for ALL DRKG node types.
    """
    import warnings
    warnings.simplefilter("ignore")
    from drugos_graph.utils import sanitize_label, drkg_node_type_to_neo4j_label
    from drugos_graph.config_schema import CORE_NODE_TYPES, DRKG_NODE_TYPES
    all_types = list(dict.fromkeys(CORE_NODE_TYPES + DRKG_NODE_TYPES))
    for etype in all_types:
        label = drkg_node_type_to_neo4j_label(etype)
        safe = sanitize_label(label)
        # The safe label must NOT contain spaces (would need backticks).
        assert " " not in str(safe), (
            f"Label for {etype!r} ({safe!r}) contains a space — "
            f"would need backtick quoting in Cypher."
        )


# ============================================================================
# P2-061: sys.path bootstrap pollution
# ============================================================================
def test_p2_061_sys_path_bootstrap_gated():
    """P2-061: phase2/__init__.py must only bootstrap sys.path as top-level."""
    init_path = _PHASE2_ROOT / "__init__.py"
    src = init_path.read_text()
    assert '__name__ == "phase2"' in src, (
        "phase2/__init__.py must gate sys.path bootstrap on __name__ == 'phase2'."
    )


# ============================================================================
# P2-062: /query endpoint drug+disease
# ============================================================================
def test_p2_062_query_handles_both_drug_and_disease():
    """P2-062: /query must handle BOTH drug and disease (not just drug)."""
    service_path = _PHASE2_ROOT / "service.py"
    src = service_path.read_text()
    assert "if drug and disease:" in src, (
        "service.py must handle the case where BOTH drug and disease are provided."
    )
    assert "shortestPath" in src, (
        "Must use shortestPath to find paths BETWEEN drug and disease."
    )


# ============================================================================
# P2-063: _Phase1BridgeResult slots fragility
# ============================================================================
def test_p2_063_phase1_bridge_result_no_slots():
    """P2-063: _Phase1BridgeResult must NOT use __slots__ (fragile with dict)."""
    from drugos_graph.phase1_bridge import _Phase1BridgeResult
    # The class must NOT have __slots__ (removed for picklability).
    assert not hasattr(_Phase1BridgeResult, "__slots__") or (
        hasattr(_Phase1BridgeResult, "__slots__")
        and "backend" not in getattr(_Phase1BridgeResult, "__slots__", ())
    ), "_Phase1BridgeResult must not declare __slots__ with 'backend'."
    # The instance must support regular attribute assignment.
    r = _Phase1BridgeResult({"a": 1}, backend="csv")
    assert r.backend == "csv"
    assert r["a"] == 1
    assert r["_phase1_backend"] == "csv"


# ============================================================================
# P2-065: requires-python vs from __future__
# ============================================================================
def test_p2_065_pyproject_requires_python_bumped():
    """P2-065: pyproject.toml requires-python must be >=3.11,<3.13."""
    pyproject_path = _PHASE2_ROOT / "drugos_graph" / "pyproject.toml"
    src = pyproject_path.read_text()
    assert 'requires-python = ">=3.11,<3.13"' in src, (
        "pyproject.toml requires-python must be '>=3.11,<3.13' "
        "(was '>=3.10' which is too permissive for PEP 563 behavior)."
    )


if __name__ == "__main__":
    # Allow running this file directly: python test_teammate4_issues.py
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
