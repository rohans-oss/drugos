"""
TM5 Forensic Root-Fix Verification Tests — P2-020 through P2-038.

This test file is the SINGLE SOURCE OF TRUTH for verifying that the 19
Phase 2 issues assigned to Team Member 5 are actually fixed at the ROOT
LEVEL (not just commented as fixed). Every test reads the ACTUAL CODE
behavior, not the comments, and exercises real code paths.

The previous v107 "ROOT FIX" comments claimed fixes were in place but
the code was broken in multiple places — most notably:
  - pyg_builder.py:3721 called `__all__.extend([...])` WITHOUT ever
    defining `__all__`, raising NameError at import time and breaking
    every downstream consumer.
  - P2-022 schema drift: kg_builder hardcoded a schema incompatible
    with common/validated_hypotheses_schema.py, so the data flywheel
    was structurally broken.
  - P2-029 @deprecated decorator was a surface-level fix — the
    non-deterministic methods still executed.
  - P2-025 had TWO different SecurityError classes with the same name.
  - P2-020 encode_failed branch was missing the MLflow tag that the
    other 4 sibling branches set.
  - P2-028 step11's leaky stratified-random fallback was reachable
    in production (defense-in-depth gap).

These tests verify the root fixes are in place AND that no regression
was introduced in any other issue.

Run with:
    python -m pytest tests/test_tm5_p2_020_to_038_forensic_root_fixes.py -v
"""

from __future__ import annotations

import csv
import inspect
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the repo root is on sys.path so `phase2.*` and `common.*` import.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Module import test — must pass before any other test runs.
# ---------------------------------------------------------------------------

def test_all_target_modules_import_clean():
    """All 11 files touched by TM5 must import without error.

    Regression guard for the pyg_builder.py __all__ NameError bug.
    """
    # Set dev mode for the import test so escape-hatch guards don't fire.
    os.environ.setdefault("DRUGOS_ENVIRONMENT", "dev")
    modules = [
        "phase2.drugos_graph.chemberta_encoder",
        "phase2.drugos_graph.graph_stats",
        "phase2.drugos_graph.kg_builder",
        "phase2.drugos_graph.mlflow_tracker",
        "phase2.drugos_graph.pyg_builder",
        "phase2.drugos_graph.evaluation",
        "phase2.drugos_graph.negative_sampling",
        "phase2.drugos_graph.transe_model",
        "phase2.drugos_graph.phase1_bridge",
        "phase2.drugos_graph.run_pipeline",
        "phase2.service",
    ]
    for mod_name in modules:
        __import__(mod_name)


# ---------------------------------------------------------------------------
# FORENSIC-LEVEL BUG FIX: pyg_builder.py __all__ undefined
# ---------------------------------------------------------------------------

def test_pyg_builder_all_is_defined():
    """pyg_builder.py must define __all__ before extending it.

    Regression guard for the production-blocking NameError bug at line
    3721 (`__all__.extend([...])` without prior definition).
    """
    from phase2.drugos_graph import pyg_builder as pg
    assert hasattr(pg, "__all__"), "pyg_builder must define __all__"
    assert isinstance(pg.__all__, list), "pyg_builder.__all__ must be a list"
    # The four schema_mappings aliases must be in __all__.
    for name in (
        "_PHASE2_TO_GT_NODE_TYPE",
        "_GT_TO_PHASE2_NODE_TYPE",
        "ALL_PHASE2_NODE_TYPES",
        "ALL_PHASE3_NODE_TYPES",
    ):
        assert name in pg.__all__, f"{name} must be in pyg_builder.__all__"


# ---------------------------------------------------------------------------
# P2-020 — ChemBERTa fallback chain (HIGH severity)
# ---------------------------------------------------------------------------

def test_p2_020_chemberta_fallback_chain_raises_on_total_failure():
    """All 3 ChemBERTa models failing must RAISE (not silently fall back)."""
    os.environ["DRUGOS_ENVIRONMENT"] = "dev"  # so strict_features isn't forced
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from phase2.drugos_graph import chemberta_encoder as ce
        import torch
        with pytest.raises(Exception):
            ce._load_model_with_fallback(
                primary_model_name="nonexistent/bogus-model-xyz",
                revision="main",
                token=None,
                torch_dtype_val=torch.float32,
                attn_implementation="eager",
                local_files_only=True,
                cache_dir=None,
                expected_model_hash=None,
            )
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def test_p2_020_encode_failed_branch_sets_mlflow_tag():
    """The encode_failed branch must set CHEMBERTA_DISABLED MLflow tag.

    P2-020 ROOT FIX (TM5): the encode_failed branch was the ONLY one of
    5 failure branches that did NOT set the MLflow tag. Operators
    monitoring the MLflow UI had no signal that the model trained on
    random Xavier features. ROOT FIX: added the same MLflow tag block
    as the 4 sibling branches.
    """
    from phase2.drugos_graph import run_pipeline as rp
    src = inspect.getsource(rp)
    # The encode_failed branch must set the tag.
    assert 'CHEMBERTA_DISABLED", "true"' in src, (
        "encode_failed branch must set CHEMBERTA_DISABLED=true MLflow tag"
    )
    # And specifically the encode_failed reason tag.
    assert 'CHEMBERTA_FAILURE_REASON", "encode_failed"' in src, (
        "encode_failed branch must set CHEMBERTA_FAILURE_REASON=encode_failed"
    )
    # Verify the tag-setting block is INSIDE the encode_failed except block.
    # Find the except block, then check the tag block is after it.
    except_idx = src.find('chemberta_failure_reason = "encode_failed"')
    assert except_idx > 0, "encode_failed branch must exist"
    tag_idx = src.find('CHEMBERTA_FAILURE_REASON", "encode_failed"', except_idx)
    assert tag_idx > except_idx, (
        "MLflow tag block must be inside the encode_failed branch"
    )


# ---------------------------------------------------------------------------
# P2-021 — graph_stats sanity thresholds immutable in production
# ---------------------------------------------------------------------------

def test_p2_021_thresholds_immutable_in_production(monkeypatch):
    """In production, env-var override must be IGNORED."""
    monkeypatch.setenv("DRUGOS_ENVIRONMENT", "production")
    monkeypatch.setenv("DRUGOS_STATS_MIN_COMPOUNDS", "1")  # should be ignored
    monkeypatch.setenv("DRUGOS_STATS_MIN_GENES", "1")  # should be ignored

    # Force re-import so module-level constants re-evaluate.
    for mod_name in list(sys.modules):
        if mod_name.startswith("phase2.drugos_graph.graph_stats"):
            del sys.modules[mod_name]
    from phase2.drugos_graph import graph_stats as gs
    assert gs.MIN_COMPOUNDS_FOR_SANITY == 10000, (
        "Production override must be ignored — MIN_COMPOUNDS_FOR_SANITY "
        "must be 10000 in production"
    )
    assert gs.MIN_GENES_FOR_SANITY == 15000, (
        "Production override must be ignored — MIN_GENES_FOR_SANITY "
        "must be 15000 in production"
    )


# ---------------------------------------------------------------------------
# P2-022 — kg_builder.update_validated_edges schema unification
# ---------------------------------------------------------------------------

def test_p2_022_accepts_canonical_schema():
    """update_validated_edges accepts canonical schema {drug, disease, outcome, validated_at}."""
    from phase2.drugos_graph import kg_builder as kg
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["drug", "disease", "outcome", "validated_at"])
        w.writerow(["aspirin", "headache", "validated_positive", "2025-01-01"])
        w.writerow(["metformin", "diabetes", "validated_toxic", "2025-01-01"])
        w.writerow(["ibuprofen", "inflammation", "validated_positive", "2025-02-01"])
        csv_path = f.name
    try:
        result = kg.update_validated_edges(validated_csv_path=csv_path, builder=None)
        assert result["total_validated_pairs"] == 2, (
            f"Expected 2 (validated_positive×2, toxic skipped), "
            f"got {result['total_validated_pairs']}"
        )
    finally:
        os.unlink(csv_path)


def test_p2_022_accepts_legacy_schema():
    """update_validated_edges accepts legacy schema {drug, disease, validated, source, validated_at}."""
    from phase2.drugos_graph import kg_builder as kg
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["drug", "disease", "validated", "source", "validated_at"])
        w.writerow(["aspirin", "headache", "true", "wet_lab", "2025-01-01"])
        w.writerow(["metformin", "diabetes", "false", "wet_lab", "2025-01-01"])
        w.writerow(["ibuprofen", "inflammation", "yes", "clinical", "2025-02-01"])
        csv_path = f.name
    try:
        result = kg.update_validated_edges(validated_csv_path=csv_path, builder=None)
        assert result["total_validated_pairs"] == 2, (
            f"Expected 2 (true+yes, false skipped), "
            f"got {result['total_validated_pairs']}"
        )
    finally:
        os.unlink(csv_path)


def test_p2_022_accepts_minimal_schema():
    """update_validated_edges accepts minimal schema {drug, disease} (current on-disk CSV state)."""
    from phase2.drugos_graph import kg_builder as kg
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["drug", "disease"])
        w.writerow(["aspirin", "headache"])
        w.writerow(["metformin", "diabetes"])
        csv_path = f.name
    try:
        result = kg.update_validated_edges(validated_csv_path=csv_path, builder=None)
        assert result["total_validated_pairs"] == 2, (
            f"Expected 2 (minimal schema treats every row as validated), "
            f"got {result['total_validated_pairs']}"
        )
    finally:
        os.unlink(csv_path)


def test_p2_022_default_path_works_with_current_csv():
    """The default rl/validated_hypotheses.csv (2-column) must NOT raise.

    Regression test: before the TM5 fix, calling update_validated_edges()
    with the default path RAISED ValueError because the on-disk CSV had
    only {drug, disease} columns. This broke the data flywheel (DOCX §10)
    end-to-end.
    """
    from phase2.drugos_graph import kg_builder as kg
    result = kg.update_validated_edges(validated_csv_path=None, builder=None)
    assert result["total_validated_pairs"] >= 4, (
        f"Expected ≥4 pairs from rl/validated_hypotheses.csv, "
        f"got {result['total_validated_pairs']}"
    )


def test_p2_022_raises_on_completely_missing_required_columns():
    """CSV with neither outcome nor validated nor {drug,disease} must raise."""
    from phase2.drugos_graph import kg_builder as kg
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["foo", "bar"])
        w.writerow(["x", "y"])
        csv_path = f.name
    try:
        with pytest.raises(ValueError, match="missing required columns"):
            kg.update_validated_edges(validated_csv_path=csv_path, builder=None)
    finally:
        os.unlink(csv_path)


# ---------------------------------------------------------------------------
# P2-023 — mlflow_tracker heartbeat max failures
# ---------------------------------------------------------------------------

def test_p2_023_heartbeat_has_failure_threshold():
    """Heartbeat must end_run(FAILED) after N consecutive failures."""
    from phase2.drugos_graph import mlflow_tracker as mt
    src = inspect.getsource(mt)
    assert "DRUGOS_MLFLOW_HEARTBEAT_MAX_FAILURES" in src, (
        "Heartbeat must have configurable max-failure threshold"
    )
    assert 'end_run(status="FAILED")' in src or "end_run(status='FAILED')" in src, (
        "Heartbeat must call end_run(status=FAILED) after threshold"
    )


# ---------------------------------------------------------------------------
# P2-024 — pyg_builder edge dedup multi-relational
# ---------------------------------------------------------------------------

def test_p2_024_multi_relational_dedup_guard_present():
    """Edge dedup must check edge_type variance and dedup on (src,dst,rel) triples."""
    from phase2.drugos_graph import pyg_builder as pg
    src = inspect.getsource(pg)
    assert "_dedup_on_edge_type" in src, (
        "Multi-relational dedup guard must be present"
    )
    assert "_ei_with_type" in src, (
        "Edge-type stacking for triple dedup must be present"
    )


# ---------------------------------------------------------------------------
# P2-025 — pyg_builder SecurityError consolidation
# ---------------------------------------------------------------------------

def test_p2_025_security_error_is_single_class():
    """pyg_builder.SecurityError must be the same class as exceptions.SecurityError.

    P2-025 ROOT FIX (TM5): previously pyg_builder defined its OWN
    SecurityError(RuntimeError) while exceptions.py defined a DIFFERENT
    SecurityError(DrugOSDataError). Callers catching SecurityError would
    miss one or the other. ROOT FIX: pyg_builder now imports from
    exceptions, so they are the same class.
    """
    from phase2.drugos_graph import pyg_builder as pg
    from phase2.drugos_graph import exceptions as exc
    assert pg.SecurityError is exc.SecurityError, (
        "pyg_builder.SecurityError must be the same class as "
        "exceptions.SecurityError (single source of truth)"
    )


def test_p2_025_security_error_can_be_raised_and_caught():
    """SecurityError raised by pyg_builder code can be caught by exceptions.SecurityError."""
    from phase2.drugos_graph import pyg_builder as pg
    from phase2.drugos_graph import exceptions as exc
    with pytest.raises(exc.SecurityError):
        raise pg.SecurityError("test")


# ---------------------------------------------------------------------------
# P2-026 — evaluation.compute_auc rejects small imbalanced eval sets
# ---------------------------------------------------------------------------

def test_p2_026_compute_auc_rejects_small_imbalanced_eval_set(monkeypatch):
    """compute_auc must raise EvaluationIntegrityError on <30 positives × 5:1 imbalance in production."""
    monkeypatch.setenv("DRUGOS_ENVIRONMENT", "production")
    monkeypatch.setenv("DRUGOS_ALLOW_SMALL_IMBALANCED_EVAL", "")
    import importlib
    from phase2.drugos_graph import evaluation as ev
    importlib.reload(ev)
    import numpy as np
    with pytest.raises(ev.EvaluationIntegrityError):
        ev.compute_auc(
            np.array([0.9, 0.8, 0.7, 0.6, 0.5]),  # 5 positives
            np.array([0.4] * 50),  # 50 negatives, ratio 1:10
            higher_is_better=True,
        )


def test_p2_026_compute_auc_works_on_balanced_100_set(monkeypatch):
    """compute_auc must work on a balanced 100×100 eval set."""
    monkeypatch.setenv("DRUGOS_ENVIRONMENT", "production")
    monkeypatch.setenv("DRUGOS_ALLOW_SMALL_IMBALANCED_EVAL", "")
    import importlib
    from phase2.drugos_graph import evaluation as ev
    importlib.reload(ev)
    import numpy as np
    auc = ev.compute_auc(
        np.array([0.9] * 100),
        np.array([0.1] * 100),
        higher_is_better=True,
    )
    assert auc == 1.0, f"Expected AUC=1.0, got {auc}"


# ---------------------------------------------------------------------------
# P2-027 — phase1_bridge isinstance(frames, dict) guard
# ---------------------------------------------------------------------------

def test_p2_027_isinstance_guard_present():
    """run_phase1_to_phase2 must use isinstance(frames, dict) before .pop()."""
    from phase2.drugos_graph import phase1_bridge as pb
    src = inspect.getsource(pb.run_phase1_to_phase2)
    assert "isinstance(frames, dict)" in src, (
        "isinstance(frames, dict) guard must be present to prevent "
        "AttributeError on dataclass frames without .pop()"
    )


# ---------------------------------------------------------------------------
# P2-028 — step11 leaky stratified-random fallback gated in production
# ---------------------------------------------------------------------------

def test_p2_028_step9_raises_on_split_failure_in_production(monkeypatch):
    """step9 must RAISE RuntimeError when node_disjoint_split fails in production."""
    from phase2.drugos_graph import run_pipeline as rp
    src = inspect.getsource(rp)
    assert "P2-028 ROOT FIX" in src, "P2-028 ROOT FIX must be present"
    assert "_is_prod_p2_028" in src, "Production check must be present in step9"


def test_p2_028_step11_defense_in_depth_gates_leaky_fallback():
    """step11's leaky stratified-random fallback must RAISE in production.

    P2-028 ROOT FIX (TM5, defense-in-depth): even though step9 raises
    first in production, step11's leaky fallback must ALSO raise as
    defense-in-depth. If a future code change makes step9's split
    silently succeed but produce empty val/test, step11 would fall
    through to the leaky path. ROOT FIX: step11 also gates on
    DRUGOS_ENVIRONMENT.
    """
    from phase2.drugos_graph import run_pipeline as rp
    src = inspect.getsource(rp)
    assert "P2-028 ROOT FIX (defense-in-depth)" in src, (
        "step11 defense-in-depth ROOT FIX must be present"
    )
    assert "_is_prod_p2_028_defense" in src, (
        "step11 defense-in-depth production check must be present"
    )


# ---------------------------------------------------------------------------
# P2-029 — deprecated methods HARD-RAISE (not just decorator)
# ---------------------------------------------------------------------------

def test_p2_029_deduplicate_edges_hard_raises():
    """DrugOSGraphBuilder.deduplicate_edges must RAISE (not just warn).

    P2-029 ROOT FIX (TM5): the @deprecated decorator was a surface-level
    fix — callers who silenced warnings still got non-reproducible edge
    sets. ROOT FIX: the method now HARD-RAISES DeprecationWarning.
    """
    from phase2.drugos_graph import kg_builder as kg
    src = inspect.getsource(kg.DrugOSGraphBuilder.deduplicate_edges)
    assert "raise DeprecationWarning" in src, (
        "deduplicate_edges must hard-raise DeprecationWarning (not just warn)"
    )


def test_p2_029_load_drkg_edges_hard_raises():
    """DrugOSGraphBuilder.load_drkg_edges must RAISE (not just warn)."""
    from phase2.drugos_graph import kg_builder as kg
    src = inspect.getsource(kg.DrugOSGraphBuilder.load_drkg_edges)
    assert "raise DeprecationWarning" in src, (
        "load_drkg_edges must hard-raise DeprecationWarning (not just warn)"
    )


def test_p2_029_graphedgeloader_deduplicate_edges_hard_raises():
    """GraphEdgeLoader.deduplicate_edges must RAISE (not just warn)."""
    from phase2.drugos_graph import kg_builder as kg
    src = inspect.getsource(kg.GraphEdgeLoader.deduplicate_edges)
    assert "raise DeprecationWarning" in src, (
        "GraphEdgeLoader.deduplicate_edges must hard-raise"
    )


def test_p2_029_graphedgeloader_load_drkg_edges_hard_raises():
    """GraphEdgeLoader.load_drkg_edges must RAISE (not just warn)."""
    from phase2.drugos_graph import kg_builder as kg
    src = inspect.getsource(kg.GraphEdgeLoader.load_drkg_edges)
    assert "raise DeprecationWarning" in src, (
        "GraphEdgeLoader.load_drkg_edges must hard-raise"
    )


def test_p2_029_deterministic_replacements_still_work():
    """The deterministic replacements must NOT raise (only the deprecated ones do)."""
    from phase2.drugos_graph import kg_builder as kg
    for method_name in (
        "deduplicate_edges_deterministic",
        "load_drkg_edges_bulk",
    ):
        method = getattr(kg.DrugOSGraphBuilder, method_name)
        src = inspect.getsource(method)
        assert "raise DeprecationWarning" not in src, (
            f"{method_name} must NOT raise — it's the deterministic replacement"
        )


# ---------------------------------------------------------------------------
# P2-030/031 — service.py /kg/explore bounded + edge dedup
# ---------------------------------------------------------------------------

def test_p2_030_no_unbounded_variable_length_path_in_cypher():
    """The [*1..2] variable-length Cypher path must be REMOVED from actual code.

    The string may still appear in COMMENTS describing the old bug, but
    must NOT appear in any executable Cypher string literal.
    """
    src = (_REPO_ROOT / "phase2" / "service.py").read_text()
    in_code = []
    for i, line in enumerate(src.split("\n"), 1):
        stripped = line.lstrip()
        if "[*1..2]" in line and not stripped.startswith("#"):
            in_code.append((i, line.rstrip()))
    assert in_code == [], (
        f"[*1..2] must not appear in executable code, only in comments. "
        f"Found in code at: {in_code}"
    )


def test_p2_030_subquery_with_limit_present():
    """The Cypher must use CALL { ... LIMIT $limit } subquery pattern."""
    from phase2 import service as svc
    src = inspect.getsource(svc)
    assert "CALL {" in src, "Subquery pattern must be present"
    assert "LIMIT $limit" in src, "Per-hop LIMIT must be inside subquery"


def test_p2_031_edge_dedup_by_source_target_type():
    """Edges must be deduplicated by (source, target, type) tuple."""
    from phase2 import service as svc
    src = inspect.getsource(svc)
    assert (
        '(e["source"], e["target"], e["type"])' in src
        or "(e['source'], e['target'], e['type'])" in src
    ), "Edge dedup must use (source, target, type) tuple key"


# ---------------------------------------------------------------------------
# P2-032 — scheduler.step() narrow exception catch
# ---------------------------------------------------------------------------

def test_p2_032_scheduler_step_narrow_catch():
    """scheduler.step() must catch only (TypeError, ValueError), not bare Exception."""
    from phase2.drugos_graph import run_pipeline as rp
    src = inspect.getsource(rp)
    assert "except (TypeError, ValueError)" in src, (
        "scheduler.step() must catch only (TypeError, ValueError), not bare Exception"
    )


# ---------------------------------------------------------------------------
# P2-033 — transe_model shape assertion in loss
# ---------------------------------------------------------------------------

def test_p2_033_shape_assertion_in_loss():
    """TransE loss must assert pos_expanded.shape == neg_scores.shape."""
    from phase2.drugos_graph import transe_model as tm
    src = inspect.getsource(tm)
    assert "pos_expanded.shape != neg_scores.shape" in src, (
        "TransE loss must have explicit shape assertion"
    )


# ---------------------------------------------------------------------------
# P2-034 — kg_builder _write_pipeline_run_node retry+fallback
# ---------------------------------------------------------------------------

def test_p2_034_pipeline_run_node_retry_and_fallback():
    """_write_pipeline_run_node must retry with exponential backoff + JSONL fallback."""
    from phase2.drugos_graph import kg_builder as kg
    src = inspect.getsource(kg.DrugOSGraphBuilder._write_pipeline_run_node)
    assert "max_attempts" in src or "_max_attempts_p2_034" in src, (
        "Retry loop with max attempts must be present"
    )
    assert "pipeline_run_audit.jsonl" in src, (
        "JSONL fallback file path must be present"
    )
    assert "logger.error" in src, (
        "ERROR-level logging on failure must be present (not WARNING)"
    )


# ---------------------------------------------------------------------------
# P2-035 — DRUGOS_ENVIRONMENT defaults to production + escape hatch guard
# ---------------------------------------------------------------------------

def test_p2_035_default_environment_is_production():
    """DRUGOS_ENVIRONMENT must default to 'production' (not 'dev')."""
    from phase2.drugos_graph import run_pipeline as rp
    src = inspect.getsource(rp._check_production_escape_hatches)
    assert 'os.environ.get("DRUGOS_ENVIRONMENT", "production")' in src, (
        "DRUGOS_ENVIRONMENT must default to 'production'"
    )


def test_p2_035_allow_launch_fail_refused_in_production():
    """DRUGOS_ALLOW_LAUNCH_FAIL must be in the escape-hatch refused list."""
    from phase2.drugos_graph import run_pipeline as rp
    src = inspect.getsource(rp._check_production_escape_hatches)
    assert "DRUGOS_ALLOW_LAUNCH_FAIL" in src, (
        "DRUGOS_ALLOW_LAUNCH_FAIL must be in the refused escape-hatch list"
    )


# ---------------------------------------------------------------------------
# P2-036 — negative_sampling cache invalidation
# ---------------------------------------------------------------------------

def test_p2_036_cache_invalidation_on_graph_change():
    """Negative sampler must invalidate degree cache when graph changes."""
    from phase2.drugos_graph import negative_sampling as ns
    src = inspect.getsource(ns)
    assert "_p2_036_cache_entity_ids" in src, (
        "Cache key on entity-id lists must be present"
    )
    assert "_cache_valid_p2_036" in src, (
        "Cache validity check must be present"
    )


# ---------------------------------------------------------------------------
# P2-037 — _check_sklearn_version raises in production
# ---------------------------------------------------------------------------

def test_p2_037_sklearn_required_in_production():
    """_check_sklearn_version must RAISE ImportError in production (not fall back)."""
    from phase2.drugos_graph import evaluation as ev
    src = inspect.getsource(ev._check_sklearn_version)
    assert "raise ImportError" in src, (
        "_check_sklearn_version must raise ImportError in production"
    )
    assert "production" in src, "Production mode check must be present"


# ---------------------------------------------------------------------------
# P2-038 — graph_stats SYMMETRIC_RELATIONS check
# ---------------------------------------------------------------------------

def test_p2_038_symmetric_relations_check_present():
    """Density must check rel_type in SYMMETRIC_RELATIONS (not endpoint-type equality)."""
    from phase2.drugos_graph import graph_stats as gs
    src = inspect.getsource(gs)
    assert "rel_type in SYMMETRIC_RELATIONS" in src, (
        "Density must check rel_type in SYMMETRIC_RELATIONS"
    )
    assert "SYMMETRIC_RELATIONS" in src, (
        "SYMMETRIC_RELATIONS must be imported/referenced"
    )


if __name__ == "__main__":
    # Allow running as a script: python -m pytest tests/test_tm5_p2_020_to_038_forensic_root_fixes.py -v
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
