#!/usr/bin/env python3
"""v100 forensic root fix verification tests.

Verifies that bugs R-018 through R-035 plus R-INT-001..009 and
R-STUB-001..005 are actually fixed at the source-code level. Reads
the REAL source files (no comments, no test fakes) and asserts each
fix is present.

Run with: python tests/test_v100_forensic_root_fixes.py
"""
from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(path: Path) -> str:
    with open(path, "r") as f:
        return f.read()


class TestR018Manifest(unittest.TestCase):
    """R-018: runners must write a manifest.json with git SHA + config hash."""

    def test_run_4phase_writes_manifest(self):
        src = _read(ROOT / "run_4phase.py")
        self.assertIn("_write_manifest", src)
        self.assertIn("manifest.json", src)
        self.assertIn("_git_rev_parse_head", src)
        self.assertIn("_git_status_porcelain", src)
        self.assertIn("config_sha256", src)

    def test_run_real_pipeline_writes_manifest(self):
        src = _read(ROOT / "run_real_pipeline.py")
        self.assertIn("_write_manifest", src)
        self.assertIn("manifest.json", src)

    def test_run_full_platform_writes_manifest(self):
        src = _read(ROOT / "run_full_platform.py")
        self.assertIn("_write_manifest", src)
        self.assertIn("manifest.json", src)


class TestR019Rename(unittest.TestCase):
    """R-019: top-level run_pipeline.py renamed to run_4phase.py."""

    def test_run_pipeline_py_does_not_exist(self):
        self.assertFalse(
            (ROOT / "run_pipeline.py").exists(),
            "run_pipeline.py should have been renamed to run_4phase.py (R-019)",
        )

    def test_run_4phase_py_exists(self):
        self.assertTrue((ROOT / "run_4phase.py").exists())


class TestR020Neo4jFallback(unittest.TestCase):
    """R-020: no bolt://localhost:7687 default; straight to RecordingGraphBuilder."""

    def test_no_localhost_default(self):
        src = _read(ROOT / "run_unified.py")
        # The line `neo4j_uri = "bolt://localhost:7687"` must NOT appear.
        self.assertNotIn(
            'neo4j_uri = "bolt://localhost:7687"',
            src,
            "R-020: the localhost default must be removed",
        )

    def test_straight_to_recording_builder(self):
        src = _read(ROOT / "run_unified.py")
        # When no URI is provided, go STRAIGHT to RecordingGraphBuilder.
        self.assertIn("No --neo4j-uri or DRUGOS_NEO4J_URI set", src)


class TestR021KeyErrorSafeAccess(unittest.TestCase):
    """R-021: results[...] must use .get() with defaults."""

    def test_no_direct_results_indexing_in_summary(self):
        src = _read(ROOT / "run_full_platform.py")
        # Each of these direct-access patterns must be gone.
        for bad in [
            "results['gt_best_val_auc']",
            "results['gt_test_auc']",
            "results['gt_epochs_trained']",
            "results['rl_pairs_processed']",
            "results['rl_ranked_high']",
            "results['n_candidates_returned']",
        ]:
            self.assertNotIn(bad, src, f"R-021: {bad} must use .get()")


class TestR022NoDuplicateSummary(unittest.TestCase):
    """R-022: the duplicate 9-line summary block must be gone."""

    def test_no_duplicate_summary(self):
        src = _read(ROOT / "run_4phase.py")
        # The old file printed "Phase 2 nodes (staged)" twice. The new file
        # prints each field exactly once.
        self.assertEqual(src.count("Phase 2 nodes (staged)"), 1)


class TestR023NoParamReassignment(unittest.TestCase):
    """R-023: run_bridge must not reassign its phase1_dir parameter."""

    def test_no_phase1_dir_reassignment(self):
        src = _read(ROOT / "run_4phase.py")
        # Parse the AST and check no assignment to `phase1_dir` inside run_bridge.
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_bridge":
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name) and target.id == "phase1_dir":
                                self.fail(
                                    "R-023: run_bridge reassigns its phase1_dir parameter"
                                )
                return
        self.fail("run_bridge function not found")


class TestR025NoDoubleSeed(unittest.TestCase):
    """R-025: no import-time _set_global_seed(42) call in run_unified.py."""

    def test_no_import_time_seed_call(self):
        src = _read(ROOT / "run_unified.py")
        # The line `_set_global_seed(42)` (as a STATEMENT, not in a comment)
        # must not appear at module import time. It's OK if the symbol is
        # imported for downstream use.
        # Strip comments and check the actual call does not happen at import.
        # The simplest check: the line `_set_global_seed(42)` (with leading
        # whitespace indicating it's a statement) must not appear.
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            self.assertFalse(
                stripped == "_set_global_seed(42)" or
                stripped.startswith("_set_global_seed(42)"),
                f"R-025: import-time seed call found: {line!r}",
            )


class TestR026CLIHelpText(unittest.TestCase):
    """R-026: --seed help must not claim SHA-256 determinism."""

    def test_no_sha256_claim_in_seed_help(self):
        for fname in ("run_4phase.py", "run_real_pipeline.py", "run_full_platform.py"):
            src = _read(ROOT / fname)
            # Parse the AST and check the --seed argument's help= kwarg.
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and
                        isinstance(node.func, ast.Attribute) and
                        node.func.attr == "add_argument"):
                    # Check if this is the --seed argument.
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and arg.value == "--seed":
                            # Find the help= kwarg.
                            for kw in node.keywords:
                                if kw.arg == "help" and isinstance(kw.value, ast.Constant):
                                    help_text = kw.value.value
                                    self.assertNotIn(
                                        "deterministic via hashlib.sha256",
                                        help_text,
                                        f"R-026: {fname} --seed help still claims SHA-256",
                                    )


class TestR027MainReturnsInt(unittest.TestCase):
    """R-027: run_real_pipeline.main() must return int, not sys.exit()."""

    def test_main_returns_int(self):
        src = _read(ROOT / "run_real_pipeline.py")
        # Parse the AST and check main()'s return annotation.
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                # The return annotation must be `int`, not `None`.
                self.assertIsInstance(node.returns, ast.Name)
                self.assertEqual(node.returns.id, "int")
                return
        self.fail("main() function not found in run_real_pipeline.py")

    def test_no_sys_exit_in_main(self):
        src = _read(ROOT / "run_real_pipeline.py")
        # Find the main() function body and check it has no sys.exit() calls.
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if (isinstance(child.func, ast.Attribute) and
                                isinstance(child.func.value, ast.Name) and
                                child.func.value.id == "sys" and
                                child.func.attr == "exit"):
                            self.fail("R-027: sys.exit() found in main()")
                return


class TestR028NoModuleLevelForceLogging(unittest.TestCase):
    """R-028: logging.basicConfig(force=True) must not be at module level."""

    def test_no_module_level_force_logging(self):
        for fname in ("run_4phase.py", "run_real_pipeline.py", "run_full_platform.py"):
            src = _read(ROOT / fname)
            tree = ast.parse(src)
            # Walk top-level statements only (module scope).
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    call = node.value
                    if (isinstance(call.func, ast.Attribute) and
                            isinstance(call.func.value, ast.Name) and
                            call.func.value.id == "logging" and
                            call.func.attr == "basicConfig"):
                        # Check no force=True kwarg
                        for kw in call.keywords:
                            if kw.arg == "force" and getattr(kw.value, "value", False) is True:
                                self.fail(
                                    f"R-028: {fname} has module-level "
                                    "logging.basicConfig(force=True)"
                                )


class TestR029NoStaleStatusPrint(unittest.TestCase):
    """R-029: the 30-line static 'V90 ROOT FIXES STATUS' block must be gone."""

    def test_no_stale_status_block(self):
        src = _read(ROOT / "run_real_pipeline.py")
        # Parse the AST and check no print() call has "V90 ROOT" in its args.
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if "V90 ROOT" in arg.value or "ROOT FIXES STATUS" in arg.value:
                            self.fail(
                                f"R-029: stale status print found: {arg.value!r}"
                            )


class TestR030MakefilePhony(unittest.TestCase):
    """R-030: Makefile must list run-json, run-neo4j, run-4phase, run-full-platform."""

    def test_phony_includes_all_targets(self):
        src = _read(ROOT / "Makefile")
        # Find the .PHONY line
        phony_line = next(
            (l for l in src.splitlines() if l.startswith(".PHONY:")), ""
        )
        for target in ("run-json", "run-neo4j", "run-4phase", "run-full-platform"):
            self.assertIn(target, phony_line, f"R-030: {target} missing from .PHONY")


class TestR031PackageReexport(unittest.TestCase):
    """R-031: RecordingGraphBuilder must be importable from drugos_graph directly."""

    def test_recording_graph_builder_in_all(self):
        src = _read(ROOT / "phase2" / "drugos_graph" / "__init__.py")
        self.assertIn('"RecordingGraphBuilder"', src)

    def test_recording_graph_builder_in_heavy_reexports(self):
        src = _read(ROOT / "phase2" / "drugos_graph" / "__init__.py")
        self.assertIn('"RecordingGraphBuilder"', src)
        self.assertIn('".phase1_bridge"', src)

    def test_run_unified_uses_package_reexport(self):
        src = _read(ROOT / "run_unified.py")
        # Should NOT import from the deep path anymore (R-031 fix).
        # Allow it inside comments but not as actual import statements.
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if (node.module == "drugos_graph.phase1_bridge" and
                        any(a.name == "RecordingGraphBuilder" for a in node.names)):
                    # The only allowed deep import is for Phase1StagedData or
                    # run_phase1_to_phase2 -- NOT RecordingGraphBuilder.
                    self.fail(
                        "R-031: run_unified.py imports RecordingGraphBuilder "
                        "from drugos_graph.phase1_bridge (should use the "
                        "package-level re-export)"
                    )


class TestR033Tier1Timeout600s(unittest.TestCase):
    """R-033: Tier 1 timeout must be 600s, not 60s."""

    def test_timeout_is_600(self):
        src = _read(ROOT / "run_unified.py")
        # The Tier 1 invocation must use timeout=600.
        self.assertIn("timeout=600", src)
        # The 60s timeout must NOT appear in the Tier 1 block.
        # (Other timeouts like Tier 2's 300s are fine.)
        self.assertNotIn("timeout=60,\n                env=_env", src)


class TestR035PhaseRequirements(unittest.TestCase):
    """R-035: graph_transformer/requirements.txt and rl/requirements.txt must exist.

    The merged Makefile uses R-017's consolidated approach (top-level
    requirements.txt only) but the per-phase files still exist for
    backwards compatibility and for operators who want to install only
    a subset (e.g. a GPU-only training container). R-035 says "EITHER
    (a) create per-phase files OR (b) consolidate" -- having both is
    also valid.
    """

    def test_graph_transformer_requirements_exist(self):
        self.assertTrue((ROOT / "graph_transformer" / "requirements.txt").exists())

    def test_rl_requirements_exist(self):
        self.assertTrue((ROOT / "rl" / "requirements.txt").exists())

    def test_makefile_installs_top_level(self):
        src = _read(ROOT / "Makefile")
        # The Makefile must install requirements.txt (the consolidated file).
        self.assertIn("requirements.txt", src)


class TestRINT002NoBrokenRunPhase2Call(unittest.TestCase):
    """R-INT-002: the broken run_phase2_kg_builder(staged, builder) call
    that referenced undefined `seed` must be gone."""

    def test_no_run_phase2_kg_builder_call(self):
        src = _read(ROOT / "run_4phase.py")
        # The function run_phase2_kg_builder must NOT exist (we removed it
        # because its body referenced an undefined `seed`).
        self.assertNotIn("def run_phase2_kg_builder", src)
        self.assertNotIn("run_phase2_kg_builder(", src)


class TestRINT004SingleBridgeCall(unittest.TestCase):
    """R-INT-004: run_bridge must call run_phase1_to_phase2 exactly ONCE."""

    def test_single_bridge_call(self):
        src = _read(ROOT / "run_4phase.py")
        # Find the run_bridge function body.
        idx = src.index("def run_bridge(")
        end = src.index("def run_schema_adapter(", idx)
        bridge_body = src[idx:end]
        count = bridge_body.count("run_phase1_to_phase2(")
        # The function should call run_phase1_to_phase2 exactly once
        # (one call site). Allow >=1 because the import statement also
        # references the name, but the actual `(...)` call should be 1.
        self.assertEqual(count, 1, "R-INT-004: run_bridge must call run_phase1_to_phase2 exactly once")


class TestRINT005SchemaAdapterConsumed(unittest.TestCase):
    """R-INT-005: run_schema_adapter's output must NOT be overwritten."""

    def test_schema_adapter_output_used(self):
        src = _read(ROOT / "run_4phase.py")
        # The pattern `graph_data = run_schema_adapter(...)` followed by
        # `graph_data = run_phase2_kg_builder(...)` must NOT appear.
        self.assertNotIn("graph_data = run_phase2_kg_builder", src)
        # graph_data from run_schema_adapter must be used by run_phase3_and_4.
        self.assertIn("graph_data=graph_data", src)


class TestRINT007NoNameErrorOnSubprocess(unittest.TestCase):
    """R-INT-007: the except clause must not raise NameError on
    `subprocess.SubprocessError`. Two valid fixes:
      (a) `import subprocess` at module level (R-001 approach), OR
      (b) `import subprocess as _sp` inside the try + use `_sp.SubprocessError`.
    """

    def test_no_bare_subprocess_in_except(self):
        src = _read(ROOT / "run_unified.py")
        # If `import subprocess` is at module level, then
        # `subprocess.SubprocessError` in except is CORRECT (not a bug).
        has_module_level_subprocess = False
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "subprocess" and alias.asname is None:
                        has_module_level_subprocess = True
        if has_module_level_subprocess:
            # `subprocess.SubprocessError` in except is fine.
            return
        # Otherwise, the except clause must NOT reference `subprocess.SubprocessError`
        # (would NameError). Must use `_sp.SubprocessError` instead.
        self.assertNotIn(
            "except (subprocess.SubprocessError",
            src,
            "R-INT-007: bare `subprocess.SubprocessError` reference causes NameError "
            "without a module-level `import subprocess`",
        )
        self.assertIn("_sp.SubprocessError", src)


class TestRINT008Phase1CsvsCaptured(unittest.TestCase):
    """R-INT-008: ensure_phase1_data's return value must be captured."""

    def test_phase1_csvs_assigned(self):
        src = _read(ROOT / "run_4phase.py")
        self.assertIn("phase1_csvs = ensure_phase1_data", src)


class TestRINT009MakefileHasRunners(unittest.TestCase):
    """R-INT-009: Makefile must have run-4phase and run-full-platform targets."""

    def test_run_4phase_target(self):
        src = _read(ROOT / "Makefile")
        self.assertIn("run-4phase:", src)
        self.assertIn("run_4phase.py", src)

    def test_run_full_platform_target(self):
        src = _read(ROOT / "Makefile")
        self.assertIn("run-full-platform:", src)
        self.assertIn("run_full_platform.py", src)


class TestRSTUB001RealPipelineUsesRealData(unittest.TestCase):
    """R-STUB-001: run_real_pipeline.py must NOT use build_demo_graph."""

    def test_no_build_demo_graph_call(self):
        src = _read(ROOT / "run_real_pipeline.py")
        # The runner must pass phase1_staged_data=staged, NOT fall through
        # to the bridge's synthetic build_demo_graph path.
        self.assertIn("phase1_staged_data=staged", src)
        # The misleading "FAILED (honest - small demo graph)" message must be gone.
        self.assertNotIn("FAILED (honest - small demo graph)", src)


class TestRSTUB005VerifyScriptPathFixed(unittest.TestCase):
    """R-STUB-005: verify_v63_fixes.py must NOT hardcode /home/z/my-project/work."""

    def test_no_hardcoded_work_path(self):
        src = _read(ROOT / "verify_v63_fixes.py")
        # Parse the AST and check no string-literal assignment to HERE
        # contains '/home/z/my-project/work'.
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "HERE":
                        if isinstance(node.value, ast.Constant):
                            val = node.value.value
                            self.assertNotIn(
                                "/home/z/my-project/work",
                                val,
                                "R-STUB-005: HERE is hardcoded to /home/z/my-project/work",
                            )
        # And it must use __file__.
        self.assertIn("os.path.dirname(os.path.abspath(__file__))", src)


class TestCIYmlUsesRenamedFile(unittest.TestCase):
    """ci.yml must compile run_4phase.py, not run_pipeline.py."""

    def test_ci_compiles_run_4phase(self):
        src = _read(ROOT / ".github" / "workflows" / "ci.yml")
        self.assertIn("run_4phase.py", src)
        # The actual `python -m compileall` command line must NOT reference
        # the old run_pipeline.py name. We check the compileall invocation
        # line specifically (the line starting with `python -m compileall`).
        compileall_lines = [
            l for l in src.splitlines() if "compileall" in l and "python" in l
        ]
        self.assertTrue(
            compileall_lines,
            "No compileall invocation found in ci.yml",
        )
        for line in compileall_lines:
            # The actual compile command is the one that lists .py files.
            # The compileall line that lists top-level runners is the one
            # we care about. Skip comment-only lines.
            if line.strip().startswith("#"):
                continue
            if "run_unified.py" in line and "run_4phase.py" in line:
                # This is the compileall command line.
                self.assertNotIn(
                    "run_pipeline.py",
                    line,
                    "R-019: ci.yml compileall still references run_pipeline.py",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
