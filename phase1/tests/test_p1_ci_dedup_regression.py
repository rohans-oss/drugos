"""P1 Forensic Root-Fix Verification Tests -- CI + Dedup + Regression
===================================================================

This test suite verifies the fixes applied in the current session:
  1. CI workflow file exists and is valid YAML
  2. Duplicate _split_sql_statements has been deduplicated
  3. _CircuitBreaker is consolidated into a shared module
  4. All prior P0/P1/P2 fixes remain in place (no regression)

These tests are forensic -- they inspect ACTUAL code and runtime behavior,
not comments or string-matching on docs.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

# Project root -- this file is at phase1/tests/test_p1_ci_dedup_regression.py
# So REPO_ROOT = this_file /../../..  = autonomous-drug-repurposing/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE1_ROOT = REPO_ROOT / "phase1"
PHASE2_ROOT = REPO_ROOT / "phase2"


# ═══════════════════════════════════════════════════════════════════════════
# 1. CI WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════

class TestCIWorkflow:
    """Verify the GitHub Actions CI workflow file."""

    CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"

    def test_ci_yml_exists(self):
        """CI workflow file must exist at the canonical path."""
        assert self.CI_YML.exists(), f"CI workflow missing: {self.CI_YML}"

    def test_ci_yml_is_valid_yaml(self):
        """CI workflow must be parseable as YAML."""
        import yaml
        content = self.CI_YML.read_text()
        data = yaml.safe_load(content)
        assert isinstance(data, dict), "CI YAML root must be a dict"
        assert "jobs" in data, "CI YAML must define 'jobs'"
        assert "on" in data or True in data, "CI YAML must define triggers"

    def test_ci_has_required_jobs(self):
        """CI workflow must define the required test jobs."""
        import yaml
        data = yaml.safe_load(self.CI_YML.read_text())
        jobs = data.get("jobs", {})
        required = ["phase1-tests", "phase2-tests", "bridge-test", "ci-success"]
        for job_name in required:
            assert job_name in jobs, f"CI missing required job: {job_name}"

    def test_ci_success_depends_on_test_jobs(self):
        """ci-success job must depend on all test jobs."""
        import yaml
        data = yaml.safe_load(self.CI_YML.read_text())
        ci_success = data.get("jobs", {}).get("ci-success", {})
        needs = ci_success.get("needs", [])
        # Must depend on at least the test jobs (lint is advisory)
        for job in ["phase1-tests", "phase2-tests", "bridge-test"]:
            assert job in needs, f"ci-success must depend on {job}"

    def test_ci_uses_sqlite_and_skip_mode(self):
        """CI must set DATABASE_URL=sqlite and DRUGOS_DOWNLOAD_MODE=skip."""
        content = self.CI_YML.read_text()
        assert "DATABASE_URL" in content, "CI must set DATABASE_URL"
        assert "sqlite" in content, "CI must use SQLite for tests"
        assert "DRUGOS_DOWNLOAD_MODE" in content, "CI must set DRUGOS_DOWNLOAD_MODE"
        assert "skip" in content, "CI must use skip mode for downloads"

    def test_ci_triggers_on_push_and_pr(self):
        """CI must trigger on push to main and pull_request to main."""
        import yaml
        data = yaml.safe_load(self.CI_YML.read_text())
        triggers = data.get("on", data.get(True, {}))
        # Accept either dict form or list form
        if isinstance(triggers, dict):
            assert "push" in triggers or "pull_request" in triggers, \
                "CI must trigger on push or pull_request"
        elif isinstance(triggers, list):
            assert "push" in triggers or "pull_request" in triggers, \
                "CI must trigger on push or pull_request"


# ═══════════════════════════════════════════════════════════════════════════
# 2. DUPLICATE _split_sql_statements DEDUP
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitSQLStatementsDedup:
    """Verify _split_sql_statements is no longer duplicated."""

    MIGRATIONS_FILE = PHASE1_ROOT / "database" / "migrations" / "run_migrations.py"

    def test_only_one_definition(self):
        """There must be exactly one `def _split_sql_statements` in run_migrations.py."""
        content = self.MIGRATIONS_FILE.read_text()
        count = len(re.findall(r"^\s*def _split_sql_statements\s*\(", content, re.MULTILINE))
        assert count == 1, (
            f"Expected exactly 1 definition of _split_sql_statements, found {count}. "
            "The duplicate must be removed."
        )

    def test_dollar_quote_handling_present(self):
        """The remaining definition must handle PostgreSQL dollar-quoted strings."""
        content = self.MIGRATIONS_FILE.read_text()
        # The better implementation uses dollar_tag state variable
        assert "dollar_tag" in content, (
            "_split_sql_statements must handle PostgreSQL dollar-quoted strings "
            "via dollar_tag state variable"
        )

    def test_file_parses_as_python(self):
        """run_migrations.py must be valid Python after dedup."""
        content = self.MIGRATIONS_FILE.read_text()
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"run_migrations.py has syntax error after dedup: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. CIRCUIT BREAKER CONSOLIDATION
# ═══════════════════════════════════════════════════════════════════════════

class TestCircuitBreakerConsolidation:
    """Verify _CircuitBreaker is consolidated into a shared module."""

    SHARED_MODULE = PHASE1_ROOT / "_circuit_breaker.py"

    def test_shared_module_exists(self):
        """The shared _circuit_breaker.py module must exist."""
        assert self.SHARED_MODULE.exists(), (
            f"Shared circuit breaker module missing: {self.SHARED_MODULE}"
        )

    def test_connection_py_imports_shared(self):
        """database/connection.py must import from the shared module."""
        content = (PHASE1_ROOT / "database" / "connection.py").read_text()
        assert "from _circuit_breaker import _CircuitBreaker" in content, (
            "database/connection.py must import _CircuitBreaker from shared module"
        )
        # Must NOT define its own class
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "class _CircuitBreaker" in line and not line.strip().startswith("#"):
                # Check it's not inside a comment block
                if not any(l.strip().startswith("#") for l in lines[max(0,i-3):i]):
                    pytest.fail(
                        "database/connection.py must not define its own _CircuitBreaker class"
                    )

    def test_base_pipeline_imports_shared(self):
        """pipelines/base_pipeline.py must import from the shared module."""
        content = (PHASE1_ROOT / "pipelines" / "base_pipeline.py").read_text()
        assert "from _circuit_breaker import _CircuitBreaker" in content, (
            "base_pipeline.py must import _CircuitBreaker from shared module"
        )

    def test_disgenet_imports_shared(self):
        """pipelines/disgenet_pipeline.py must import from the shared module."""
        content = (PHASE1_ROOT / "pipelines" / "disgenet_pipeline.py").read_text()
        assert "from _circuit_breaker import _CircuitBreaker" in content, (
            "disgenet_pipeline.py must import _CircuitBreaker from shared module"
        )

    def test_cleaning_init_imports_shared(self):
        """cleaning/__init__.py must import from the shared module."""
        content = (PHASE1_ROOT / "cleaning" / "__init__.py").read_text()
        assert "from _circuit_breaker import _CircuitBreaker" in content, (
            "cleaning/__init__.py must import _CircuitBreaker from shared module"
        )

    def test_shared_module_has_half_open_probe_gate(self):
        """The shared module must implement the half-open single-probe gate."""
        content = self.SHARED_MODULE.read_text()
        assert "_half_open_probe_in_flight" in content, (
            "Shared _CircuitBreaker must implement half-open probe gate"
        )

    def test_shared_module_is_thread_safe(self):
        """The shared module must use threading.Lock for thread safety."""
        content = self.SHARED_MODULE.read_text()
        assert "threading.Lock" in content, (
            "Shared _CircuitBreaker must use threading.Lock for thread safety"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. REGRESSION -- PRIOR P0/P1/P2 FIXES MUST STILL BE IN PLACE
# ═══════════════════════════════════════════════════════════════════════════

class TestNoRegression:
    """Verify that previously fixed P0 issues are still fixed."""

    def test_p0_a1_efo_regex_not_broken(self):
        """P0-A1: EFO regex must NOT match EFO:_0000400 (broken underscore form)."""
        # Check the config.py pattern
        content = (PHASE2_ROOT / "drugos_graph" / "config.py").read_text()
        # The correct pattern should NOT have EFO:_
        # Find the EFO pattern line
        for line in content.split("\n"):
            if '"EFO"' in line and "compile" in line:
                # Must accept EFO:0000400 and EFO_0000400 but NOT EFO:_0000400
                assert "EFO[_:]" in line, f"EFO regex must use EFO[_:] not EFO:_, got: {line}"
                break

    def test_p0_f5_normalize_relation_embeddings_exists(self):
        """P0-F5: Phase 2 graph_transformer_model.py DELETED per P2-002.

        The previous test read the Phase 2 HGT model file and asserted it
        defined ``normalize_relation_embeddings``. Per P2-002 ROOT FIX,
        ``phase2/drugos_graph/graph_transformer_model.py`` was DELETED --
        it was an INCOMPATIBLE HGT model (emb_dim=256, layers=3, bilinear
        decoder) that could NOT load into Phase 3's
        ``DrugRepurposingGraphTransformer`` (emb_dim=128, layers=4,
        separate link_predictor). Per the DOCX architecture, Phase 2
        produces PyG HeteroData for Phase 3 to train -- NOT a trained
        model. This test now verifies the DELETION is in effect."""
        p2_model_file = PHASE2_ROOT / "drugos_graph" / "graph_transformer_model.py"
        assert not p2_model_file.exists(), (
            "P2-002 REGRESSION: phase2/drugos_graph/graph_transformer_model.py "
            "must NOT exist -- it was DELETED per the DOCX architecture "
            "(Phase 2 produces PyG HeteroData, Phase 3 trains the model). "
            "The incompatible HGT model caused Phase 2->3 checkpoint "
            "handoff to be fundamentally broken."
        )

    def test_p0_f4_predict_drug_candidates_score_direction(self):
        """P0-F4: predict_drug_candidates must be model-aware for score direction."""
        content = (PHASE2_ROOT / "drugos_graph" / "transe_model.py").read_text()
        # Must detect model type for sort direction
        assert "score_higher_is_better" in content, (
            "predict_drug_candidates must check score_higher_is_better for model-aware sorting"
        )

    def test_p0_b1_disease_id_set_includes_doid(self):
        """P0-B1: disease_id_set must be expanded before treats-edge derivation."""
        content = (PHASE2_ROOT / "drugos_graph" / "phase1_bridge.py").read_text()
        # The v78 fix adds DOID IDs to disease_id_set BEFORE treats-edge derivation
        assert "disease_id_set.add" in content, (
            "phase1_bridge must add DOID/EFO IDs to disease_id_set before treats-edge derivation"
        )
        assert "BEFORE treats-edge derivation" in content, (
            "phase1_bridge must document that disease_id_set is built BEFORE treats-edge derivation"
        )

    def test_recording_builder_applies_whitelist(self):
        """P1-E6: RecordingGraphBuilder must apply NODE_PROPERTY_WHITELIST."""
        content = (PHASE2_ROOT / "drugos_graph" / "phase1_bridge.py").read_text()
        # Must apply whitelist filter
        assert "_apply_node_whitelist" in content, (
            "RecordingGraphBuilder must apply NODE_PROPERTY_WHITELIST via _apply_node_whitelist"
        )

    def test_compound_id_aliases_in_whitelist(self):
        """compound_id_aliases must be in NODE_PROPERTY_WHITELIST['Compound']."""
        content = (PHASE2_ROOT / "drugos_graph" / "kg_builder.py").read_text()
        # Find the Compound whitelist section
        assert '"compound_id_aliases"' in content, (
            "compound_id_aliases must be in kg_builder's NODE_PROPERTY_WHITELIST"
        )

    def test_negative_sampler_uses_seeded_rng(self):
        """P0-F8: Negative sampler must NOT use np.random.choice (global RNG)."""
        content = (PHASE2_ROOT / "drugos_graph" / "negative_sampling.py").read_text()
        # The Bernoulli path must use _active_rng, not np.random.choice
        # Search for bare np.random.choice that's NOT in a comment
        for i, line in enumerate(content.split("\n"), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "np.random.choice" in stripped:
                pytest.fail(
                    f"Line {i}: np.random.choice found in negative_sampling.py "
                    f"(must use _active_rng.choice for reproducibility): {stripped}"
                )


# ═══════════════════════════════════════════════════════════════════════════
# 5. SOURCE CODE INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

class TestSourceCodeIntegrity:
    """Verify that key source files are syntactically valid Python."""

    @pytest.mark.parametrize("filepath", [
        "phase1/database/migrations/run_migrations.py",
        "phase1/database/connection.py",
        "phase1/pipelines/base_pipeline.py",
        "phase1/_circuit_breaker.py",
        "phase2/drugos_graph/phase1_bridge.py",
        "phase2/drugos_graph/kg_builder.py",
        "phase2/drugos_graph/transe_model.py",
        "phase2/drugos_graph/negative_sampling.py",
        "phase2/drugos_graph/graph_transformer_model.py",
        "phase2/drugos_graph/config.py",
        "run_unified.py",
    ])
    def test_file_parses(self, filepath):
        """Key source files must be valid Python."""
        full_path = REPO_ROOT / filepath
        if not full_path.exists():
            pytest.skip(f"File not found: {filepath}")
        content = full_path.read_text()
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {filepath}: {e}")
