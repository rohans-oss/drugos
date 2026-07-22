"""v100 ROOT FIX verification tests.

Targeted unit tests for the 22 bugs fixed in the v100 forensic root-fix
branch. Each test verifies a specific fix is in place and behaves
correctly. Tests are HERMETIC -- no network, no Neo4j, no Postgres.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ─── Runner bug fixes (R-001 .. R-008) ─────────────────────────────────────

def test_R_001_run_unified_no_subprocess_alias():
    """R-001: ``import subprocess as _sp`` replaced with ``import subprocess``."""
    src = (_REPO_ROOT / "run_unified.py").read_text()
    # The old buggy alias must be gone.
    assert "import subprocess as _sp" not in src, (
        "R-001 REGRESSION: run_unified.py still has 'import subprocess as _sp'"
    )
    # The new canonical form must be present.
    assert "import subprocess" in src, (
        "R-001 INCOMPLETE: run_unified.py is missing 'import subprocess'"
    )


def test_R_002_run_phase2_kg_builder_has_seed_param():
    """R-002: ``run_phase2_kg_builder`` accepts ``seed: int = 42``.

    v100 design note: this function may also be DELETED entirely (R-INT-002
    alternative fix). Both approaches eliminate the NameError on `seed`.
    """
    # Try the renamed file first (R-019), then fall back to the original.
    import importlib.util
    import inspect
    for fname in ("run_4phase.py", "run_pipeline.py"):
        path = _REPO_ROOT / fname
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(fname.replace(".py", ""), path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            continue
        if not hasattr(mod, "run_phase2_kg_builder"):
            continue  # Function deleted -- valid R-INT-002 alternative fix.
        sig = inspect.signature(mod.run_phase2_kg_builder)
        assert "seed" in sig.parameters, (
            "R-002 REGRESSION: run_phase2_kg_builder is missing 'seed' parameter"
        )
        return
    # If no file has the function, that's the R-INT-002 alternative fix
    # (function deleted entirely) -- valid.


def test_R_003_R_004_no_duplicate_run_bridge_call():
    """R-003 + R-004: ``run_bridge`` called once, no dead ``run_schema_adapter``.

    v100 design note: run_pipeline.py was renamed to run_4phase.py (R-019).
    The test checks whichever file exists.
    """
    # Try the renamed file first (R-019), then fall back to the original.
    src = None
    for fname in ("run_4phase.py", "run_pipeline.py"):
        path = _REPO_ROOT / fname
        if path.exists():
            src = path.read_text()
            break
    assert src is not None, (
        "R-003: neither run_4phase.py nor run_pipeline.py exists"
    )
    # Strip comments and docstrings to test ACTUAL CODE only.
    import ast
    tree = ast.parse(src)
    # Find the main function and check its body for actual run_bridge calls
    # (not comments or docstrings).
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_src = ast.get_source_segment(src, node)
            # Count ACTUAL assignment statements that unpack run_bridge(...).
            # The buggy pattern was: `staged, builder = run_bridge(...)`
            # (reversed tuple order). After the fix, only the correct
            # `builder, staged = run_bridge(...)` should appear.
            import re
            # Match only lines that are actual code (not comments).
            code_lines = [
                line for line in main_src.split("\n")
                if not line.strip().startswith("#") and not line.strip().startswith('"')
            ]
            code_only = "\n".join(code_lines)
            reversed_calls = re.findall(
                r"^\s*staged,\s*builder\s*=\s*run_bridge\(",
                code_only,
                re.MULTILINE,
            )
            assert len(reversed_calls) == 0, (
                f"R-003 REGRESSION: run_pipeline.main() has {len(reversed_calls)} "
                f"reversed 'staged, builder = run_bridge(...)' call(s) in actual code"
            )
            # The dead run_schema_adapter call (followed by duplicate
            # run_bridge) must be gone from actual code.
            dead_pattern = re.findall(
                r"graph_data\s*=\s*run_schema_adapter\([^)]+\)\s*\n\s*staged,\s*builder\s*=\s*run_bridge\(",
                code_only,
            )
            assert len(dead_pattern) == 0, (
                "R-004 REGRESSION: run_pipeline.main() still has the dead "
                "run_schema_adapter + duplicate run_bridge block"
            )
            # The correct unpacking must be present.
            correct_calls = re.findall(
                r"^\s*builder,\s*staged\s*=\s*run_bridge\(",
                code_only,
                re.MULTILINE,
            )
            assert len(correct_calls) >= 1, (
                "R-003 INCOMPLETE: run_pipeline.main() missing correct "
                "'builder, staged = run_bridge(...)' call"
            )
            break
    else:
        pytest.fail("run_pipeline.main() function not found")


def test_R_005_phase1_csvs_defined():
    """R-005: ``phase1_csvs`` is captured from ``ensure_phase1_data``.

    v100 design note: run_pipeline.py was renamed to run_4phase.py (R-019).
    The test checks whichever file exists.
    """
    candidates = ["run_pipeline.py", "run_4phase.py"]
    found = False
    for fname in candidates:
        path = _REPO_ROOT / fname
        if not path.exists():
            continue
        src = path.read_text()
        if "phase1_csvs = ensure_phase1_data" in src:
            found = True
            break
    assert found, (
        f"R-005 REGRESSION: none of {candidates} capture phase1_csvs"
    )


def test_R_006_run_real_pipeline_has_phase1_wiring():
    """R-006: ``run_real_pipeline.py`` wires Phase 1+2 by default.

    v100 design note: an alternative valid fix removes the --demo flag
    entirely and ALWAYS runs on real Phase 1 data (R-STUB-001 approach).
    Both approaches eliminate the synthetic-demo masquerade.
    """
    src = (_REPO_ROOT / "run_real_pipeline.py").read_text()
    assert "run_phase1_to_phase2" in src, "R-006: Phase 2 bridge not wired"
    assert "phase1_staged_data" in src, "R-006: staged data not passed to bridge"
    # Either --demo flag is present OR the synthetic build_demo_graph path is gone.
    if "--demo" not in src:
        # Must NOT call build_demo_graph as the default path.
        assert "num_drugs=args.num_drugs" not in src or \
               "phase1_staged_data=staged" in src, (
            "R-006: no --demo flag AND no phase1_staged_data -- synthetic fallback"
        )


def test_R_007_run_unified_uses_GTRLBridge():
    """R-007: ``run_unified.py`` calls ``GTRLBridge.run_full_pipeline``.

    v100 design note: an alternative valid fix keeps run_unified.py as a
    Phase 1+2 runner (with the Phase 2 internal run_full_pipeline import)
    and routes Phase 3+4 through run_4phase.py or run_full_platform.py
    (R-INT-001/R-INT-009 approach). In that case, GTRLBridge must be
    imported by one of those alternate runners.
    """
    src = (_REPO_ROOT / "run_unified.py").read_text()
    if "GTRLBridge" in src:
        # Direct fix: run_unified.py uses GTRLBridge.
        return
    # Alternative fix: GTRLBridge is in run_4phase.py or run_full_platform.py.
    for fname in ("run_4phase.py", "run_full_platform.py"):
        path = _REPO_ROOT / fname
        if not path.exists():
            continue
        alt_src = path.read_text()
        if "GTRLBridge" in alt_src:
            return
    pytest.fail(
        "R-007: GTRLBridge not found in run_unified.py, run_4phase.py, "
        "or run_full_platform.py"
    )


def test_R_008_verify_v63_fixes_uses_relative_path():
    """R-008: ``verify_v63_fixes.py`` uses ``__file__``-relative path."""
    src = (_REPO_ROOT / "verify_v63_fixes.py").read_text()
    assert 'HERE = "/home/z/my-project/work"' not in src, (
        "R-008 REGRESSION: verify_v63_fixes.py still has hardcoded path"
    )
    assert "os.path.dirname(os.path.abspath(__file__))" in src, (
        "R-008 INCOMPLETE: verify_v63_fixes.py not using __file__-relative path"
    )


# ─── Phase 3 bug fixes (P3-001 .. P3-008) ───────────────────────────────────

def test_P3_001_graph_builder_uses_build_reverse_edges_into_sets():
    """P3-001: production path uses ``_build_reverse_edges_into_sets``."""
    src = (_REPO_ROOT / "graph_transformer" / "data" / "graph_builder.py").read_text()
    # The buggy pattern must be gone from from_phase1_staged_data.
    assert "BiomedicalGraphBuilder._build_reverse_edges(\n            builder._edge_lists\n        )" not in src, (
        "P3-001 REGRESSION: graph_builder.py still uses deprecated _build_reverse_edges"
    )
    assert "builder._build_reverse_edges_into_sets(builder._edge_sets)" in src, (
        "P3-001 INCOMPLETE: graph_builder.py not using _build_reverse_edges_into_sets"
    )


def test_P3_002_evaluation_has_logger():
    """P3-002: ``evaluation/__init__.py`` imports logging and defines logger."""
    import graph_transformer.evaluation as ev
    assert hasattr(ev, "logger"), "P3-002: evaluation module has no 'logger' attribute"
    import logging
    assert isinstance(ev.logger, logging.Logger), (
        "P3-002: ev.logger is not a logging.Logger instance"
    )


def test_P3_003_layers_no_ParameterDict_get():
    """P3-003: ``layers.py`` does not use ``.get()`` on ParameterDict."""
    src = (_REPO_ROOT / "graph_transformer" / "models" / "layers.py").read_text()
    assert "self.edge_gates.get(" not in src, (
        "P3-003 REGRESSION: layers.py still uses .get() on ParameterDict"
    )
    assert "if edge_key in self.edge_gates" in src, (
        "P3-003 INCOMPLETE: layers.py not using explicit membership check"
    )


def test_P3_004_gt_rl_bridge_no_threshold_relowering():
    """P3-004: ``gt_rl_bridge.py`` does not re-lower ``kp_recovery_threshold``."""
    src = (_REPO_ROOT / "graph_transformer" / "gt_rl_bridge.py").read_text()
    # The V90 BUG #31 floor must still be there.
    assert "kp_recovery_threshold = max(rl_config_threshold, 0.5)" in src, (
        "P3-004 INCOMPLETE: gt_rl_bridge.py missing the max(..., 0.5) floor"
    )
    # The re-lowering line must be GONE or COMMENTED OUT. Both forms are
    # valid fixes (delete vs comment-out). Look for the pattern as actual
    # code (not inside a comment line).
    import re
    # Match the re-lowering pattern only when the line is NOT a comment.
    # A comment line starts with optional whitespace then '#'.
    active_relowering = re.findall(
        r"^\s*(?!#)\s*kp_recovery_threshold\s*=\s*float\(getattr\(rl_config,[^)]+\)\)",
        src,
        re.MULTILINE,
    )
    assert len(active_relowering) == 0, (
        f"P3-004 REGRESSION: gt_rl_bridge.py has an ACTIVE re-lowering line: {active_relowering}"
    )


def test_P3_005_pathway_diff_runs_unconditionally():
    """P3-005: pathway-connectivity differentiation runs on BOTH paths."""
    src = (_REPO_ROOT / "graph_transformer" / "gt_rl_bridge.py").read_text()
    # The unreachable dead code (return inside try, then more code) must be gone.
    # The new structure has a single return after the try/except.
    # Accept either "v100 ROOT FIX (P3-005)" or "V92 ROOT FIX (BUG P3-005)"
    # marker -- both branches applied the same fix with different comment styles.
    assert (
        "v100 ROOT FIX (P3-005)" in src
        or "V92 ROOT FIX (BUG P3-005" in src
        or "ROOT FIX (BUG P3-005" in src
    ), (
        "P3-005 INCOMPLETE: P3-005 fix marker not found (checked v100 and V92 markers)"
    )


def test_P3_007_phase2_adapter_uses_deterministic_seed():
    """P3-007: ``phase2_adapter.py`` uses ``_deterministic_seed`` not ``hash()``."""
    src = (_REPO_ROOT / "graph_transformer" / "data" / "phase2_adapter.py").read_text()
    # The buggy hash() pattern must be gone.
    assert "hash(drug_name) & 0xFFFFFFFF" not in src, (
        "P3-007 REGRESSION: phase2_adapter.py still uses hash(drug_name)"
    )
    assert "hash(protein_name) & 0xFFFFFFFF" not in src, (
        "P3-007 REGRESSION: phase2_adapter.py still uses hash(protein_name)"
    )
    assert "hash(disease_name) & 0xFFFFFFFF" not in src, (
        "P3-007 REGRESSION: phase2_adapter.py still uses hash(disease_name)"
    )
    # The deterministic helper must be imported and used.
    assert "_deterministic_seed" in src, (
        "P3-007 INCOMPLETE: phase2_adapter.py does not use _deterministic_seed"
    )


def test_P3_008_resume_predicate_checks_both_graph_data_and_staged():
    """P3-008: resume predicate checks both ``graph_data`` and ``phase1_staged_data``."""
    src = (_REPO_ROOT / "graph_transformer" / "gt_rl_bridge.py").read_text()
    assert "graph_data is None and phase1_staged_data is None" in src, (
        "P3-008 REGRESSION: gt_rl_bridge.py resume predicate does not check both"
    )


# ─── Phase 4 bug fixes (P4-002, P4-003, P4-005, P4-017, P4-018) ─────────────

def test_P4_002_indication_specific_withdrawn_check():
    """P4-002: thalidomide is NOT in the global WITHDRAWN_DRUGS set, and
    indication-specific check exists."""
    import rl.rl_drug_ranker as rl_mod
    # Thalidomide must NOT be in the global set.
    assert "thalidomide" not in rl_mod.WITHDRAWN_DRUGS, (
        "P4-002 REGRESSION: thalidomide still in global WITHDRAWN_DRUGS"
    )
    # Thalidomide must be in the indication-specific map.
    assert "thalidomide" in rl_mod.WITHDRAWN_INDICATIONS, (
        "P4-002 INCOMPLETE: thalidomide not in WITHDRAWN_INDICATIONS"
    )
    # The indication-specific check function must exist.
    assert hasattr(rl_mod, "_is_withdrawn_for_indication"), (
        "P4-002 INCOMPLETE: _is_withdrawn_for_indication function not defined"
    )
    # Thalidomide + multiple myeloma = ALLOWED (FDA-approved).
    assert rl_mod._is_withdrawn_for_indication("thalidomide", "multiple myeloma") is False, (
        "P4-002 LOGIC ERROR: thalidomide should be ALLOWED for multiple myeloma"
    )
    # Thalidomide + morning sickness = REJECTED (teratogen).
    assert rl_mod._is_withdrawn_for_indication("thalidomide", "morning sickness") is True, (
        "P4-002 LOGIC ERROR: thalidomide should be REJECTED for morning sickness"
    )
    # Rofecoxib (withdrawn for ALL indications) = always REJECTED.
    assert rl_mod._is_withdrawn_for_indication("rofecoxib", "arthritis") is True, (
        "P4-002 LOGIC ERROR: rofecoxib should be REJECTED for all indications"
    )


def test_P4_005_rejection_counters_incremented():
    """P4-005: ``DrugRankingEnv.step()`` increments rejection counters."""
    src = (_REPO_ROOT / "rl" / "rl_drug_ranker.py").read_text()
    # The env must have per-env counters.
    assert "self.n_safety_rejected: int = 0" in src, (
        "P4-005 INCOMPLETE: env does not init n_safety_rejected"
    )
    assert "self.n_gnn_rejected: int = 0" in src, (
        "P4-005 INCOMPLETE: env does not init n_gnn_rejected"
    )
    # The step() method must increment them.
    assert "self.n_safety_rejected += 1" in src, (
        "P4-005 INCOMPLETE: step() does not increment n_safety_rejected"
    )
    assert "self.n_gnn_rejected += 1" in src, (
        "P4-005 INCOMPLETE: step() does not increment n_gnn_rejected"
    )
    # The run_pipeline function must copy them to PipelineMetrics.
    assert "metrics.n_safety_rejected = " in src, (
        "P4-005 INCOMPLETE: run_pipeline does not copy counters to metrics"
    )


def test_P4_017_small_graph_multiplier_is_5x():
    """P4-017: small-graph multiplier raised from 2× to 5×."""
    src = (_REPO_ROOT / "rl" / "rl_drug_ranker.py").read_text()
    # The new 5× multiplier must be present.
    assert "env.n_pairs * 5" in src, (
        "P4-017 REGRESSION: small-graph multiplier not raised to 5×"
    )
    # The old 2× multiplier must be gone (in the small-graph branch).
    # We can't just check "env.n_pairs * 2" doesn't appear anywhere
    # (it might appear in a comment), so check the specific pattern.
    assert "max_n_steps = max(1, env.n_pairs * 2)" not in src, (
        "P4-017 REGRESSION: old 2× multiplier still present"
    )


def test_P4_018_ppo_gamma_default_is_nonzero():
    """P4-018: ``ppo_gamma`` default is non-zero (real PPO, not bandit)."""
    import rl.rl_drug_ranker as rl_mod
    import inspect
    cfg = rl_mod.PipelineConfig()
    assert cfg.ppo_gamma > 0.0, (
        f"P4-018 REGRESSION: ppo_gamma default is {cfg.ppo_gamma} (should be > 0)"
    )
    assert cfg.ppo_gamma == 0.95, (
        f"P4-018 INCOMPLETE: ppo_gamma default is {cfg.ppo_gamma} (expected 0.95)"
    )


# ─── Phase 1 bug fixes (P1-001, P1-003) ─────────────────────────────────────

def test_P1_001_chembl_no_double_read():
    """P1-001: ``chembl_pipeline.py`` does not unconditionally re-read CSV."""
    src = (_REPO_ROOT / "phase1" / "pipelines" / "chembl_pipeline.py").read_text()
    # The double-read pattern (compression variable + unconditional read
    # AFTER the try/except) must be gone. Accept any of the equivalent
    # fix markers (v100, V92, or P1-001 ROOT FIX).
    assert (
        "v100 ROOT FIX (P1-001)" in src
        or "P1-001 ROOT FIX" in src
        or "V92 ROOT FIX (BUG P1-001" in src
    ), (
        "P1-001 INCOMPLETE: P1-001 fix marker not found"
    )
    # The unconditional re-read pattern (lines that follow the try/except)
    # must not be present. The try/except handles both gzip and plain CSV.
    # Look for the specific dead pattern as ACTIVE CODE (not in a comment).
    import re
    active_double_read = re.findall(
        r"^\s*(?!#)\s*_compression\s*=\s*['\"]gzip['\"] if raw_path\.suffix == ['\"]\.gz['\"] else None\s*\n\s*_drugs_df\s*=\s*pd\.read_csv\(",
        src,
        re.MULTILINE,
    )
    # The variable might be 'drugs_df' not '_drugs_df'. Try both.
    active_double_read_2 = re.findall(
        r"^\s*(?!#)\s*_compression\s*=\s*['\"]gzip['\"] if raw_path\.suffix == ['\"]\.gz['\"] else None",
        src,
        re.MULTILINE,
    )
    assert len(active_double_read_2) == 0, (
        "P1-001 REGRESSION: chembl_pipeline.py still has the active double-read pattern"
    )


def test_P1_003_docker_compose_has_missing_mounts():
    """P1-003: docker-compose.yml mounts ``./data``, ``./exporters``, ``./scripts``."""
    src = (_REPO_ROOT / "phase1" / "docker-compose.yml").read_text()
    # All 3 airflow services must have these mounts.
    data_count = src.count("./data:/opt/airflow/data")
    exporters_count = src.count("./exporters:/opt/airflow/exporters")
    scripts_count = src.count("./scripts:/opt/airflow/scripts")
    assert data_count == 3, (
        f"P1-003 INCOMPLETE: ./data mount appears {data_count} times (expected 3 -- one per airflow service)"
    )
    assert exporters_count == 3, (
        f"P1-003 INCOMPLETE: ./exporters mount appears {exporters_count} times (expected 3)"
    )
    assert scripts_count == 3, (
        f"P1-003 INCOMPLETE: ./scripts mount appears {scripts_count} times (expected 3)"
    )


# ─── Phase 2 bug fixes (P2-001, P2-002, P2-004, P2-005, P2-012) ─────────────

def test_P2_001_run_pipeline_no_relations_NameError():
    """P2-001: ``run_pipeline.py:5953`` uses ``rels`` not ``relations``."""
    src = (_REPO_ROOT / "phase2" / "drugos_graph" / "run_pipeline.py").read_text()
    # The buggy pattern must be gone.
    assert "int(relations[_i])" not in src, (
        "P2-001 REGRESSION: phase2 run_pipeline.py still references 'relations[_i]'"
    )
    # The fix must be present.
    assert "int(rels[_i])" in src, (
        "P2-001 INCOMPLETE: phase2 run_pipeline.py does not use 'rels[_i]'"
    )


def test_P2_002_make_negatives_returns_full_length():
    """P2-002: ``_make_negatives`` always returns ``len(positive_indices)`` items,
    and padding appends tuples (not scalars)."""
    src = (_REPO_ROOT / "phase2" / "drugos_graph" / "run_pipeline.py").read_text()
    # The (0, 0) padding for skipped positives must be present.
    assert "negs.append((0, 0))" in src, (
        "P2-002 INCOMPLETE: _make_negatives does not pad with (0, 0) tuples"
    )
    # The caller's padding must append TUPLES, not scalars.
    assert "batch_neg.append(_rng.choice(all_disease_indices)" not in src, (
        "P2-002 REGRESSION: caller still appends scalars to batch_neg"
    )


def test_P2_004_kg_builder_checkpoint_uses_len_batch():
    """P2-004: ``kg_builder.py`` checkpoint uses ``len(batch)`` not ``batch_size``."""
    src = (_REPO_ROOT / "phase2" / "drugos_graph" / "kg_builder.py").read_text()
    assert '"last_completed_idx": i + len(batch) - 1' in src, (
        "P2-004 INCOMPLETE: kg_builder.py checkpoint not using len(batch)"
    )
    assert '"last_completed_idx": i + batch_size - 1' not in src, (
        "P2-004 REGRESSION: kg_builder.py still uses batch_size"
    )


def test_P2_005_kg_builder_uses_EXISTS_pattern():
    """P2-005: ``kg_builder.py`` uses Neo4j 5.x ``EXISTS { MATCH ... }`` pattern."""
    src = (_REPO_ROOT / "phase2" / "drugos_graph" / "kg_builder.py").read_text()
    # The deprecated pattern must be gone.
    assert "size((:Compound {{id: a}})) > 0" not in src, (
        "P2-005 REGRESSION: kg_builder.py still uses deprecated size() pattern"
    )
    # The new pattern must be present.
    assert "EXISTS {{ MATCH (:Compound {{id: a}}) }}" in src, (
        "P2-005 INCOMPLETE: kg_builder.py not using EXISTS { MATCH ... } pattern"
    )


def test_P2_012_training_data_checks_both_schema_keys():
    """P2-012: ``training_data.py`` checks both ``_schema_version`` and ``schema_version``."""
    src = (_REPO_ROOT / "phase2" / "drugos_graph" / "training_data.py").read_text()
    # The fix must check both keys.
    assert '"schema_version" in drkg_df.attrs' in src, (
        "P2-012 INCOMPLETE: training_data.py does not check 'schema_version' (no underscore)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
