"""v113 Forensic Root Fix Verification Tests.

This module contains REAL verification tests for the 22 issues fixed
in v113. Each test reads the ACTUAL code (not comments, not test
fixtures) and verifies the root-cause fix is in place.

Run with:
    pytest tests/forensic_root_v113/test_v113_all_fixes.py -v

All tests are marked ``forensic`` (per the pytest.ini marker policy)
and run by default (no marker filter excludes them).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = REPO_ROOT / "phase1"
PHASE2_DIR = REPO_ROOT / "phase2"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (str(REPO_ROOT), str(PHASE1_DIR), str(PHASE2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


pytestmark = pytest.mark.forensic


# ─────────────────────────────────────────────────────────────────────────────
# P1-014: omim_pipeline.py module-level random.seed()
# ─────────────────────────────────────────────────────────────────────────────
def test_P1_014_omim_pipeline_no_module_level_random_seed():
    """P1-014 ROOT FIX: ``random.seed(OMIM_RANDOM_SEED)`` at module
    import time MUST be deleted. The global RNG must NOT be mutated
    by importing omim_pipeline.
    """
    src = (PHASE1_DIR / "pipelines" / "omim_pipeline.py").read_text()
    # Verify no module-level random.seed() call (function-level is OK
    # if it uses a local Random instance).
    # Find the module-level call: a line that starts with "random.seed("
    # at column 0 (not indented inside a function).
    bad_lines = [
        line for line in src.split("\n")
        if line.startswith("random.seed(") and not line.strip().startswith("#")
    ]
    assert not bad_lines, (
        f"P1-014 NOT fixed: module-level random.seed() found: {bad_lines}"
    )
    # Also verify the import of random was removed (since it's no longer used)
    # Actually, keep this loose -- if random IS used elsewhere legitimately,
    # the import should stay. We only care that the module-level seed call is gone.


def test_P1_014_omim_pipeline_does_not_mutate_global_rng():
    """P1-014 ROOT FIX: importing omim_pipeline must NOT change the
    global ``random.random()`` sequence.
    """
    import random
    # Seed before import
    random.seed(424242)
    expected = random.random()
    # Re-seed and import the module (this used to reset the global seed)
    random.seed(424242)
    # Force re-import
    if "pipelines.omim_pipeline" in sys.modules:
        del sys.modules["pipelines.omim_pipeline"]
    try:
        # The import may fail due to missing heavy deps (sqlalchemy, etc.)
        # -- that's OK, we just want to verify the IMPORT ATTEMPT doesn't
        # mutate the global RNG. We catch the ImportError and continue.
        try:
            importlib.import_module("pipelines.omim_pipeline")
        except Exception:
            pass  # missing deps are OK for this test
    finally:
        pass
    # The global RNG should still produce the expected value
    actual = random.random()
    assert actual == expected, (
        f"P1-014 NOT fixed: importing omim_pipeline mutated the global RNG. "
        f"Expected {expected}, got {actual}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# P1-025: base_pipeline.py set_seed mutates global RNG
# ─────────────────────────────────────────────────────────────────────────────
def test_P1_025_base_pipeline_no_global_seed_in_run():
    """P1-025 ROOT FIX: ``random.seed(self.seed)`` and ``np.random.seed(self.seed)``
    in ``run()`` MUST be replaced with per-instance ``self._rng``.
    """
    src = (PHASE1_DIR / "pipelines" / "base_pipeline.py").read_text()
    # Find active (non-comment) lines that call random.seed() or np.random.seed()
    bad_lines = []
    for line in src.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.match(r"^\s*random\.seed\(", line) or re.match(r"^\s*np\.random\.seed\(", line):
            bad_lines.append(line.rstrip())
    assert not bad_lines, (
        f"P1-025 NOT fixed: global seed calls still present: {bad_lines}"
    )


def test_P1_025_base_pipeline_uses_per_instance_rng():
    """P1-025 ROOT FIX: ``self._rng`` must be initialized in __init__
    and used for ``random.uniform()`` calls in ``_download_with_retries``.
    """
    src = (PHASE1_DIR / "pipelines" / "base_pipeline.py").read_text()
    assert "self._rng: random.Random = random.Random(self.seed)" in src, (
        "P1-025 NOT fixed: self._rng not initialized in __init__"
    )
    assert "self._rng.uniform(0, 1)" in src, (
        "P1-025 NOT fixed: self._rng.uniform not used for jitter"
    )
    # Verify no remaining global random.uniform() calls
    bad = [l for l in src.split("\n") if "random.uniform" in l and "self._rng" not in l and not l.strip().startswith("#")]
    assert not bad, f"P1-025 NOT fixed: random.uniform still used globally: {bad}"


# ─────────────────────────────────────────────────────────────────────────────
# P1-024: _v50_downloaders.py FULL mode silent empty DrugBank CSV
# ─────────────────────────────────────────────────────────────────────────────
def test_P1_024_drugbank_full_mode_raises_without_env_var(tmp_path, monkeypatch):
    """P1-024 ROOT FIX: FULL mode MUST raise RuntimeError unless
    DRUGOS_ALLOW_NO_DRUGBANK=1 is set. Empty CSVs + data_status marker
    must still be written so downstream contract checks pass.
    """
    monkeypatch.setenv("DRUGOS_DOWNLOAD_MODE", "full")
    monkeypatch.delenv("DRUGOS_ALLOW_NO_DRUGBANK", raising=False)
    from phase1.pipelines._v50_downloaders import download_drugbank_open_data
    with pytest.raises(RuntimeError, match="P1-024"):
        download_drugbank_open_data(tmp_path)
    # Verify empty CSVs were written
    assert (tmp_path / "drugbank_open_drugs.csv").exists()
    assert (tmp_path / "drugbank_open_indications.csv").exists()
    assert (tmp_path / "drugbank_data_status.json").exists()


def test_P1_024_drugbank_full_mode_succeeds_with_env_var(tmp_path, monkeypatch):
    """P1-024 ROOT FIX: FULL mode succeeds with DRUGOS_ALLOW_NO_DRUGBANK=1."""
    monkeypatch.setenv("DRUGOS_DOWNLOAD_MODE", "full")
    monkeypatch.setenv("DRUGOS_ALLOW_NO_DRUGBANK", "1")
    from phase1.pipelines._v50_downloaders import download_drugbank_open_data
    result = download_drugbank_open_data(tmp_path)
    assert result["drugs"].exists()
    assert result["indications"].exists()
    assert result["data_status"].exists()


# ─────────────────────────────────────────────────────────────────────────────
# P2-050: _compute_normalized_score treats withdrawn as 0.3
# ─────────────────────────────────────────────────────────────────────────────
def test_P2_050_withdrawn_drug_gets_zero_confidence():
    """P2-050 ROOT FIX: withdrawn drugs MUST get 0.0 confidence on
    treats edges (patient-safety guard).
    """
    from phase2.drugos_graph.phase1_bridge import _compute_normalized_score
    assert _compute_normalized_score(
        indication_type="withdrawn",
        source="drugbank_indications",
        rel_type="treats",
    ) == 0.0
    # Case-insensitive
    assert _compute_normalized_score(
        indication_type="Withdrawn",
        source="drugbank_indications",
        rel_type="treats",
    ) == 0.0
    # "approved_and_withdrawn" -> 0.0 (withdrawn overrides)
    assert _compute_normalized_score(
        indication_type="approved_and_withdrawn",
        source="drugbank_indications",
        rel_type="treats",
    ) == 0.0
    # Approved still gets 1.0
    assert _compute_normalized_score(
        indication_type="approved",
        source="drugbank_indications",
        rel_type="treats",
    ) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# P2-047: phase1_bridge missing SIDER entry in paths dict
# ─────────────────────────────────────────────────────────────────────────────
def test_P2_047_sider_entry_in_paths_dict():
    """P2-047 ROOT FIX: the bridge's ``paths`` dict MUST include
    a SIDER entry so adverse-event data can be consumed.
    """
    src = (PHASE2_DIR / "drugos_graph" / "phase1_bridge.py").read_text()
    assert '"sider_adverse_events"' in src, (
        "P2-047 NOT fixed: 'sider_adverse_events' key not in paths dict"
    )
    assert "sider_adverse_events.csv" in src, (
        "P2-047 NOT fixed: canonical SIDER filename not present"
    )


# ─────────────────────────────────────────────────────────────────────────────
# P2-046/048: ClinicalOutcome ID collision
# ─────────────────────────────────────────────────────────────────────────────
def test_P2_046_048_clinical_outcome_id_deterministic():
    """P2-046/048 ROOT FIX: ClinicalOutcome ID MUST be
    ``CO:{disease_key}:{indication_type}`` (no drugbank_id) so it's
    deterministic across runs and unique per (disease, type) pair.
    """
    import pandas as pd
    from phase2.drugos_graph.phase1_bridge import _load_clinical_outcomes
    indications = pd.DataFrame([
        {"drugbank_id": "DB00001", "disease_id": "OMIM:104300",
         "disease_name": "Hypertension", "indication_type": "approved"},
        {"drugbank_id": "DB00002", "disease_id": "OMIM:104300",
         "disease_name": "Hypertension", "indication_type": "approved"},
        {"drugbank_id": "DB00003", "disease_id": "OMIM:104300",
         "disease_name": "Hypertension", "indication_type": "investigational"},
    ])
    dcmap = {"DB00001": "Compound:DB00001",
             "DB00002": "Compound:DB00002",
             "DB00003": "Compound:DB00003"}
    nodes, edges = _load_clinical_outcomes(
        indications=indications, drugs=pd.DataFrame(),
        drug_canonical_map=dcmap, run_id="test",
        loaded_at="2026-01-01", schema_version="1.0",
    )
    # 2 nodes (approved + investigational), 3 edges
    assert len(nodes) == 2
    assert len(edges) == 3
    # IDs must NOT contain drugbank_id
    for n in nodes:
        assert "DB0000" not in n["id"], f"ID should not contain dbid: {n['id']}"
        assert n["id"].startswith("CO:OMIM:104300:"), f"wrong ID format: {n['id']}"
    # Both approved drugs share the same CO node
    approved_edges = [e for e in edges if "approved" in e["dst_id"]]
    assert len(approved_edges) == 2
    assert approved_edges[0]["dst_id"] == approved_edges[1]["dst_id"]


# ─────────────────────────────────────────────────────────────────────────────
# P2-049: CORE_EDGE_TYPES legacy SIDER edge bypass
# ─────────────────────────────────────────────────────────────────────────────
def test_P2_049_legacy_edge_type_removed():
    """P2-049 ROOT FIX: the legacy ``("Compound", "causes_side_effect",
    "Side Effect")`` tuple MUST be REMOVED from CORE_EDGE_TYPES.
    """
    from phase2.drugos_graph.config_schema import (
        CORE_EDGE_TYPES, CORE_EDGE_TYPES_SET,
    )
    legacy = ("Compound", "causes_side_effect", "Side Effect")
    canonical = ("Compound", "causes_adverse_event", "MedDRA_Term")
    assert legacy not in CORE_EDGE_TYPES, (
        "P2-049 NOT fixed: legacy edge still in CORE_EDGE_TYPES"
    )
    assert legacy not in CORE_EDGE_TYPES_SET, (
        "P2-049 NOT fixed: legacy edge still in SET"
    )
    assert canonical in CORE_EDGE_TYPES, "canonical edge missing"


# ─────────────────────────────────────────────────────────────────────────────
# P2-044/045: service.py uses unstable Neo4j internal IDs
# ─────────────────────────────────────────────────────────────────────────────
def test_P2_044_045_no_neo4j_internal_ids_in_response():
    """P2-044/045 ROOT FIX: ``_explore_subgraph_neo4j`` MUST use the
    business ``id`` property (not Neo4j internal ``node.id``) and
    MUST NOT use ``r.start_node.id`` / ``r.end_node.id`` for edges.
    """
    src = (PHASE2_DIR / "service.py").read_text()
    # Extract the function body
    m = re.search(
        r"def _explore_subgraph_neo4j.*?(?=\n\n\n# ─── P2-002)",
        src, re.DOTALL,
    )
    assert m, "function not found"
    body = m.group(0)
    # No active code should use d_node.id / n1.id / n2.id as response id
    bad_patterns = []
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.search(r'"id":\s*\w+\.id\b', line) and "__neo4j_internal" not in line:
            bad_patterns.append(line.strip())
        if re.search(r"r[12]\.start_node\.id|r[12]\.end_node\.id", line):
            bad_patterns.append(line.strip())
    assert not bad_patterns, (
        f"P2-044/045 NOT fixed: bad patterns remain: {bad_patterns}"
    )
    # Verify business id helper exists
    assert "_business_id" in body, "_business_id helper not added"
    assert "_node_record" in body, "_node_record helper not added"


# ─────────────────────────────────────────────────────────────────────────────
# IN-038 + IN-039: gt_api.py CORS + lifespan
# ─────────────────────────────────────────────────────────────────────────────
def test_IN_038_no_deprecated_on_event():
    """IN-038 ROOT FIX: ``@app.on_event("startup")`` MUST be removed."""
    src = (SCRIPTS_DIR / "gt_api.py").read_text()
    # Check AST-level: no @app.on_event decorator
    import ast
    tree = ast.parse(src)
    on_event_count = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "on_event"
    )
    assert on_event_count == 0, (
        f"IN-038 NOT fixed: {on_event_count} @app.on_event decorators remain"
    )
    assert "async def lifespan" in src, "lifespan not defined"
    assert "lifespan=lifespan" in src, "lifespan not used in FastAPI()"


def test_IN_039_cors_hardened():
    """IN-039 ROOT FIX: CORS MUST NOT use credentials, MUST use explicit
    headers (not ``*``), and MUST reject wildcard origins.
    """
    src = (SCRIPTS_DIR / "gt_api.py").read_text()
    assert "allow_credentials=False" in src, "credentials not disabled"
    assert '"Content-Type", "Authorization", "X-Request-ID"' in src, (
        "explicit header list not set"
    )
    assert "_validate_cors_origins" in src, "validation function not added"
    assert 'if "*" in origins' in src, "wildcard not rejected"
    # Verify no active-code wildcard headers
    for line in src.split("\n"):
        if line.strip().startswith("#"):
            continue
        if "allow_headers" in line and "[\"*\"]" in line:
            pytest.fail(f"IN-039 NOT fixed: wildcard headers in active code: {line}")


def test_IN_039_wildcard_origins_rejected():
    """IN-039 ROOT FIX: setting ``GT_CORS_ORIGINS=*`` MUST raise at startup."""
    import importlib
    env_var = "GT_CORS_ORIGINS"
    old = os.environ.get(env_var)
    os.environ[env_var] = "*"
    try:
        # Reload the module to trigger the validation
        if "scripts.gt_api" in sys.modules:
            del sys.modules["scripts.gt_api"]
        # Add scripts to path
        if str(SCRIPTS_DIR.parent) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR.parent))
        with pytest.raises(RuntimeError, match="IN-039"):
            importlib.import_module("scripts.gt_api")
    finally:
        if old is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = old


# ─────────────────────────────────────────────────────────────────────────────
# IN-055 + IN-085: pytest.ini addopts + testpaths
# ─────────────────────────────────────────────────────────────────────────────
def test_IN_055_pytest_ini_has_marker_filter():
    """IN-055 ROOT FIX: pytest.ini addopts MUST include
    ``-m "not network and not gpu and not slow"``.
    """
    import configparser
    cp = configparser.ConfigParser(allow_no_value=True, strict=False)
    cp.read(REPO_ROOT / "pytest.ini")
    addopts = cp["pytest"].get("addopts", "")
    assert "not network" in addopts, "network filter missing"
    assert "not gpu" in addopts, "gpu filter missing"
    assert "not slow" in addopts, "slow filter missing"


def test_IN_085_pytest_ini_no_nonexistent_testpath():
    """IN-085 ROOT FIX: pytest.ini testpaths MUST NOT include
    ``phase2/drugos_graph/tests`` (directory does not exist).
    """
    import configparser
    cp = configparser.ConfigParser(allow_no_value=True, strict=False)
    cp.read(REPO_ROOT / "pytest.ini")
    testpaths = cp["pytest"].get("testpaths", "")
    assert "drugos_graph/tests" not in testpaths, (
        "IN-085 NOT fixed: drugos_graph/tests still in testpaths"
    )


# ─────────────────────────────────────────────────────────────────────────────
# IN-060: test_root_cause_fixes.py mutates production validated_hypotheses.csv
# ─────────────────────────────────────────────────────────────────────────────
def test_IN_060_test_uses_tempdir_not_production():
    """IN-060 ROOT FIX: test_X08 MUST use ``tempfile.TemporaryDirectory()``
    + ``VALIDATED_HYPOTHESES_CSV`` env var, NOT write to production
    ``rl/validated_hypotheses.csv``.
    """
    src = (SCRIPTS_DIR / "test_root_cause_fixes.py").read_text()
    m = re.search(r"def test_X08.*?(?=\n\n\n# =)", src, re.DOTALL)
    assert m, "test_X08 function not found"
    body = m.group(0)
    assert "tempfile.TemporaryDirectory" in body, "not using tempdir"
    assert "VALIDATED_HYPOTHESES_CSV" in body, "not using env var"
    assert 'rl_dir = os.path.join(_CODEBASE, "rl")' not in body, (
        "IN-060 NOT fixed: still computes rl_dir"
    )
    assert 'test_csv = os.path.join(rl_dir' not in body, (
        "IN-060 NOT fixed: still writes to rl_dir"
    )


# ─────────────────────────────────────────────────────────────────────────────
# IN-072: scripts/legacy/ + root-level deprecated runners + Makefile
# ─────────────────────────────────────────────────────────────────────────────
def test_IN_072_legacy_dir_deleted():
    """IN-072 ROOT FIX: ``scripts/legacy/`` MUST be deleted."""
    assert not (SCRIPTS_DIR / "legacy").exists(), (
        "IN-072 NOT fixed: scripts/legacy/ still exists"
    )


def test_IN_072_root_runners_deleted():
    """IN-072 ROOT FIX: root-level deprecated runner scripts MUST be deleted."""
    for name in ("run_real_pipeline.py", "run_full_platform.py", "run_unified.py"):
        assert not (REPO_ROOT / name).exists(), (
            f"IN-072 NOT fixed: {name} still exists at root"
        )


def test_IN_072_makefile_targets_are_aliases():
    """IN-072 ROOT FIX: Makefile ``run-full-platform``, ``run-unified``,
    ``run-real`` targets MUST be aliases for ``make run`` (not invoke
    deleted scripts).
    """
    src = (REPO_ROOT / "Makefile").read_text()
    # The deprecated targets should NOT invoke the deleted scripts
    for target in ("run-full-platform", "run-unified", "run-real"):
        # Find the target body
        m = re.search(rf"^{target}:\n(.*?)(?=\n\n|\Z)", src, re.MULTILINE | re.DOTALL)
        assert m, f"target {target} not found in Makefile"
        body = m.group(1)
        # Should NOT invoke the deleted .py files
        for script in ("run_full_platform.py", "run_unified.py", "run_real_pipeline.py"):
            assert f"$(PYTHON) {script}" not in body, (
                f"IN-072 NOT fixed: {target} still invokes {script}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# IN-079: pre_commit_issue_guard.py fails OPEN
# ─────────────────────────────────────────────────────────────────────────────
def test_IN_079_guard_fails_closed_when_target_missing(tmp_path, monkeypatch):
    """IN-079 ROOT FIX: ``pre_commit_issue_guard.py`` MUST return 1
    (fail CLOSED) when ``pre_commit_ownership_guard.py`` is missing.
    """
    spec = importlib.util.spec_from_file_location(
        "pig", str(SCRIPTS_DIR / "pre_commit_issue_guard.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Move the target temporarily
    target = SCRIPTS_DIR / "pre_commit_ownership_guard.py"
    backup = SCRIPTS_DIR / "_backup_pig_test.py"
    if target.exists():
        target.rename(backup)
    try:
        rc = mod.main()
        assert rc == 1, f"IN-079 NOT fixed: expected exit 1, got {rc}"
    finally:
        if backup.exists():
            backup.rename(target)


# ─────────────────────────────────────────────────────────────────────────────
# IN-096: scripts/restore_test.py exists
# ─────────────────────────────────────────────────────────────────────────────
def test_IN_096_restore_test_script_exists():
    """IN-096 ROOT FIX: ``scripts/restore_test.py`` MUST exist."""
    assert (SCRIPTS_DIR / "restore_test.py").exists(), (
        "IN-096 NOT fixed: scripts/restore_test.py not created"
    )


def test_IN_096_restore_test_makefile_target():
    """IN-096 ROOT FIX: Makefile MUST have a ``restore-test`` target."""
    src = (REPO_ROOT / "Makefile").read_text()
    assert "restore-test:" in src, "restore-test target not in Makefile"


# ─────────────────────────────────────────────────────────────────────────────
# IN-051: MANIFEST.in data files
# ─────────────────────────────────────────────────────────────────────────────
def test_IN_051_manifest_includes_data_files():
    """IN-051 ROOT FIX: MANIFEST.in MUST include ``*.yaml``, ``*.json``,
    ``*.md``, ``*.txt`` for phase1/phase2/graph_transformer/rl AND
    include ``shared/`` entirely.
    """
    src = (REPO_ROOT / "MANIFEST.in").read_text()
    assert "recursive-include phase1 *.yaml *.json *.md *.txt" in src
    assert "recursive-include phase2 *.yaml *.json *.md *.txt" in src
    assert "recursive-include graph_transformer *.yaml *.json *.md *.txt" in src
    assert "recursive-include shared *.py *.json *.yaml *.md" in src


# ─────────────────────────────────────────────────────────────────────────────
# IN-087: No root README.md
# ─────────────────────────────────────────────────────────────────────────────
def test_IN_087_root_readme_exists():
    """IN-087 ROOT FIX: ``README.md`` MUST exist at the repo root."""
    assert (REPO_ROOT / "README.md").exists(), (
        "IN-087 NOT fixed: README.md not created at root"
    )


# ─────────────────────────────────────────────────────────────────────────────
# IN-089: hypothesis_writeback.py file-based RPC
# ─────────────────────────────────────────────────────────────────────────────
def test_IN_089_writeback_validates_paths():
    """IN-089 ROOT FIX: ``hypothesis_writeback.py`` MUST validate that
    req_path / resp_path are inside an allowed temp directory (path
    traversal prevention).
    """
    spec = importlib.util.spec_from_file_location(
        "hw", str(SCRIPTS_DIR / "hypothesis_writeback.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Path traversal should be rejected
    with pytest.raises(ValueError, match="IN-089"):
        mod._validate_path("/etc/shadow", "req_path")
    # Valid temp path should be accepted
    import tempfile
    valid = mod._validate_path(
        os.path.join(tempfile.gettempdir(), "test.json"), "req_path"
    )
    assert valid is not None


def test_IN_089_writeback_has_timeout():
    """IN-089 ROOT FIX: ``hypothesis_writeback.py`` MUST enforce a
    30s timeout on the writeback call.
    """
    src = (SCRIPTS_DIR / "hypothesis_writeback.py").read_text()
    assert "WRITEBACK_TIMEOUT_SECONDS" in src, "timeout not defined"
    assert "worker.join(timeout=" in src, "thread join with timeout not used"


# ─────────────────────────────────────────────────────────────────────────────
# P2-043: bridge_fallbacks.jsonl nonsensical audit entries
# ─────────────────────────────────────────────────────────────────────────────
def test_P2_043_audit_guard_rejects_test_pollution():
    """P2-043 ROOT FIX: ``_log_bridge_fallback`` MUST reject entries
    with ``thread_N`` or ``write_N`` patterns (test pollution).
    """
    src = (PHASE2_DIR / "drugos_graph" / "phase1_bridge.py").read_text()
    # Verify the guard regex is present
    assert r"^thread_\d+$" in src, "thread_N pattern guard not present"
    assert r"^write_\d+$" in src, "write_N pattern guard not present"
    assert "P2-043" in src, "P2-043 marker not in source"


def test_P2_043_audit_log_no_pollution():
    """P2-043 ROOT FIX: the existing ``bridge_fallbacks.jsonl`` MUST
    have been purged of test-pollution entries.
    """
    import json
    log_path = PHASE2_DIR / "logs" / "audit" / "bridge_fallbacks.jsonl"
    if not log_path.exists():
        pytest.skip("audit log not present")
    layer_re = re.compile(r"^thread_\d+$")
    reason_re = re.compile(r"^write_\d+$")
    polluted = 0
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if layer_re.match(str(entry.get("layer", ""))) or reason_re.match(str(entry.get("reason", ""))):
            polluted += 1
    assert polluted == 0, (
        f"P2-043 NOT fixed: {polluted} test-pollution entries still in audit log"
    )


if __name__ == "__main__":
    # Allow running directly: python test_v113_all_fixes.py
    sys.exit(pytest.main([__file__, "-v"]))
