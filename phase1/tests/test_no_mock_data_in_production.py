"""TM1 Task 16: regression guard for C-09 — no mock data in production.

This test imports Phase 1 in PRODUCTION mode (DRUGOS_ENVIRONMENT=production)
and asserts that:

1. ``pipelines._dev_samples`` CANNOT be imported in production — the
   import-time guard raises ``ImportError`` immediately. This is the
   PRIMARY defense: the mock-data module is unreachable from production
   code.

2. ``pipelines._dev_samples.write_all_samples`` (if somehow obtained,
   e.g. via importlib trickery) raises ``RuntimeError`` when called in
   production. This is the SECONDARY defense.

3. ``python -m phase1.pipelines all`` does NOT write embedded samples
   even when all pipelines fail. The `all` command's fallback path
   was REMOVED in TM1 Task 2 — this test verifies the removal is
   permanent (regression guard).

4. ``python -m phase1.pipelines samples`` refuses to run in production
   with a clear error and exit code 1.

The test is run in CI on every push. If any of these checks fails, the
platform has regressed to a state where mock data could leak into
production — the test blocks the merge.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure phase1/ is on sys.path (matches the conftest.py setup).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE1_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT, PHASE1_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---------------------------------------------------------------------------
# Helper: force a clean import state for _dev_samples
# ---------------------------------------------------------------------------


def _purge_dev_samples_from_sys_modules() -> None:
    """Remove any cached ``pipelines._dev_samples`` from sys.modules."""
    for key in list(sys.modules.keys()):
        if "_dev_samples" in key or "_embedded_samples" in key:
            del sys.modules[key]


# ---------------------------------------------------------------------------
# Test 1: import-time guard blocks production imports
# ---------------------------------------------------------------------------


def test_dev_samples_module_cannot_be_imported_in_production(monkeypatch):
    """TM1 Task 16 / C-09 regression: ``pipelines._dev_samples`` raises
    ``ImportError`` at import time when ``DRUGOS_ENVIRONMENT=production``.
    """
    monkeypatch.setenv("DRUGOS_ENVIRONMENT", "production")
    monkeypatch.setenv("ENVIRONMENT", "production")  # legacy fallback
    _purge_dev_samples_from_sys_modules()

    with pytest.raises(ImportError, match="DEVELOPMENT-ONLY module"):
        import importlib
        importlib.import_module("pipelines._dev_samples")


def test_dev_samples_module_cannot_be_imported_with_unset_env(monkeypatch):
    """TM1 Task 16: unset ``DRUGOS_ENVIRONMENT`` defaults to production
    (defensive — matches settings.py). Import MUST fail.
    """
    monkeypatch.delenv("DRUGOS_ENVIRONMENT", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    _purge_dev_samples_from_sys_modules()

    with pytest.raises(ImportError, match="DEVELOPMENT-ONLY module"):
        import importlib
        importlib.import_module("pipelines._dev_samples")


def test_dev_samples_module_imports_in_development(monkeypatch):
    """TM1 Task 16: in development, the import succeeds AND the
    runtime guard on ``write_all_samples`` still blocks production calls.
    """
    monkeypatch.setenv("DRUGOS_ENVIRONMENT", "development")
    _purge_dev_samples_from_sys_modules()

    import importlib
    mod = importlib.import_module("pipelines._dev_samples")
    assert hasattr(mod, "write_all_samples"), "write_all_samples must exist"

    # Now switch to production and verify the runtime guard fires.
    monkeypatch.setenv("DRUGOS_ENVIRONMENT", "production")
    with pytest.raises(RuntimeError, match="PRODUCTION"):
        mod.write_all_samples("/tmp/tm1_test_samples_should_not_exist")

    # Cleanup: ensure no CSVs were written.
    tmp_path = Path("/tmp/tm1_test_samples_should_not_exist")
    assert not tmp_path.exists(), "write_all_samples must not write files in production"


# ---------------------------------------------------------------------------
# Test 2: the `all` command never writes embedded samples
# ---------------------------------------------------------------------------


def test_all_command_does_not_call_write_all_samples():
    """TM1 Task 2 regression: the source of ``phase1/pipelines/__init__.py``
    must NOT contain any ACTIVE call to ``write_all_samples`` in the
    ``all`` command path. The fallback block was REMOVED in TM1 Task 2.

    We inspect the source code (not run it) because running the `all`
    command would make real API calls. The source-level check is the
    precise regression guard.
    """
    init_path = PHASE1_ROOT / "pipelines" / "__init__.py"
    src = init_path.read_text()

    # The `all` command's fallback block was REMOVED. The ONLY remaining
    # references to write_all_samples should be in the `samples` command
    # (which is itself dev-only — gated by the import-time guard).
    # Find the `all` command block.
    assert "elif cmd == \"all\"" in src or "cmd == \"all\"" in src, \
        "Could not locate the `all` command in pipelines/__init__.py"

    # Find the start of the `all` block and the start of the next command.
    # The `all` command must NOT call write_all_samples in its body.
    all_idx = src.find('cmd == "all"')
    # Find the next `elif cmd ==` after the `all` block.
    next_elif = src.find("elif cmd ==", all_idx + 1)
    if next_elif == -1:
        # `all` is the last command; check the rest of the file.
        all_block = src[all_idx:]
    else:
        all_block = src[all_idx:next_elif]

    # The `all` command must NOT contain an ACTIVE call to write_all_samples.
    # Comments mentioning write_all_samples are allowed (they document
    # the removal). An ACTIVE call is `write_all_samples(...)` not
    # preceded by `#`.
    import re
    # Find all `write_all_samples(` occurrences in the all_block.
    # Each must be on a comment line (preceded by `#`).
    for match in re.finditer(r"write_all_samples\s*\(", all_block):
        # Find the start of the line containing this match.
        line_start = all_block.rfind("\n", 0, match.start()) + 1
        line = all_block[line_start:match.start()]
        # Strip leading whitespace and check if the line starts with #.
        stripped = line.lstrip()
        assert stripped.startswith("#"), (
            f"TM1 Task 2 regression: found an ACTIVE call to "
            f"write_all_samples in the `all` command block at line: "
            f"{all_block[line_start:match.end()+30]!r}. The `all` "
            f"command must NEVER write mock samples."
        )


# ---------------------------------------------------------------------------
# Test 3: the `samples` command refuses to run in production
# ---------------------------------------------------------------------------


def test_samples_command_refuses_in_production(tmp_path):
    """TM1 Task 2: ``python -m phase1.pipelines samples`` exits 1 in
    production with a clear error message.
    """
    env = os.environ.copy()
    env["DRUGOS_ENVIRONMENT"] = "production"
    env["ENVIRONMENT"] = "production"
    env["PYTHONPATH"] = f"{PHASE1_ROOT}:{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(
        [sys.executable, "-m", "phase1.pipelines", "samples", str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, (
        f"`samples` command must exit 1 in production. Got exit "
        f"{result.returncode}. stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # The error message must mention "DEVELOPMENT-ONLY" or similar.
    combined = result.stdout + result.stderr
    assert "DEVELOPMENT-ONLY" in combined or "dev" in combined.lower(), (
        f"`samples` command must explain that it is dev-only. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # No CSVs must have been written.
    csvs = list(tmp_path.glob("*.csv*"))
    assert not csvs, (
        f"`samples` command must not write any CSVs in production. "
        f"Found: {csvs}"
    )


# ---------------------------------------------------------------------------
# Test 4: run_4phase.py no longer defines _ensure_phase1_samples
# ---------------------------------------------------------------------------


def test_run_4phase_no_longer_defines_ensure_phase1_samples():
    """TM1 Task 3 regression: ``_ensure_phase1_samples`` was DELETED
    from run_4phase.py. The function definition must not exist.
    """
    run_4phase_path = PROJECT_ROOT / "run_4phase.py"
    src = run_4phase_path.read_text()
    # The function definition must NOT exist.
    assert "def _ensure_phase1_samples(" not in src, (
        "TM1 Task 3 regression: _ensure_phase1_samples() is still defined "
        "in run_4phase.py. It was supposed to be DELETED and replaced "
        "with an inline hard check that exits 1 on empty processed_data."
    )
    # The hard check must be present.
    assert "TM1 TASK 3 ROOT FIX" in src or "TM1 Task 3" in src, (
        "TM1 Task 3: the hard check replacing _ensure_phase1_samples "
        "must be present and reference TM1 Task 3 for traceability."
    )


if __name__ == "__main__":
    # Allow running this test file directly: python -m pytest phase1/tests/test_no_mock_data_in_production.py -v
    sys.exit(pytest.main([__file__, "-v"]))
