"""Test for P4-003 ROOT FIX (HIGH).

P4-003: validated_hypotheses.csv path is RELATIVE to CWD — breaks in
        Docker/Kubernetes/systemd. The module_dir path is correct (ships
        with the package), but if the package is pip-installed without
        MANIFEST.in / package_data, the file may not be included in the
        wheel. In production, the validated bonus is silently 0 for ALL
        pairs, breaking the data flywheel (DOCX §10).

        Fix: (1) Add MANIFEST.in to include validated_hypotheses.csv in
        the package. (2) Add a runtime CRITICAL log if the file is not
        found in ANY of the 3 candidate paths. (3) Allow override via
        the RL_VALIDATED_HYPOTHESES_PATH env var.

This test verifies:
  1. MANIFEST.in exists and includes rl/*.csv.
  2. The runtime check logs CRITICAL when the file is missing.
  3. The RL_VALIDATED_HYPOTHESES_PATH env var override works.
  4. The canonical file (rl/validated_hypotheses.csv) exists in the repo.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "rl"))


def test_p4_003_manifest_in_exists_and_includes_csv():
    """MANIFEST.in must exist and include rl/*.csv.

    Without MANIFEST.in, `pip install` does not include
    validated_hypotheses.csv in the wheel, breaking the data flywheel
    in production.
    """
    manifest_path = _REPO_ROOT / "MANIFEST.in"
    assert manifest_path.exists(), (
        "P4-003: MANIFEST.in must exist at the repo root to ensure "
        "validated_hypotheses.csv ships with the package."
    )
    manifest = manifest_path.read_text()
    assert "rl *.csv" in manifest or "rl/*.csv" in manifest or "validated_hypotheses.csv" in manifest, (
        "P4-003: MANIFEST.in must include rl/*.csv (or the specific "
        "validated_hypotheses.csv file) so it ships with the package."
    )


def test_p4_003_canonical_validated_hypotheses_csv_exists():
    """The canonical rl/validated_hypotheses.csv must exist in the repo."""
    csv_path = _REPO_ROOT / "rl" / "validated_hypotheses.csv"
    assert csv_path.exists(), (
        "P4-003: the canonical rl/validated_hypotheses.csv must exist "
        "in the repo. This is the file that ships with the package."
    )
    # The file must have at least 1 validated pair (header + 1 row).
    content = csv_path.read_text().strip()
    lines = content.splitlines()
    assert len(lines) >= 2, (
        f"P4-003: rl/validated_hypotheses.csv has {len(lines)} lines, "
        f"expected at least 2 (header + 1 pair)."
    )
    assert "drug" in lines[0].lower() and "disease" in lines[0].lower(), (
        f"P4-003: rl/validated_hypotheses.csv header must contain "
        f"'drug' and 'disease'. Got: {lines[0]}"
    )


def test_p4_003_runtime_check_logs_critical_when_file_missing(monkeypatch):
    """The runtime check must log CRITICAL when no file is found.

    We force the file to be missing by:
      1. Setting RL_VALIDATED_HYPOTHESES_PATH to a non-existent path.
      2. cd-ing to a temp dir (so CWD-relative and CWD-absolute paths
         don't find the file).
      3. Monkey-patching the module's __file__ so module_dir points to
         a temp dir (so the module-local path doesn't find the canonical
         rl/validated_hypotheses.csv).

    Then we re-invoke _load_validated_hypotheses() and check that a
    CRITICAL log was emitted.
    """
    import rl.rl_drug_ranker as mod

    # Save original state.
    orig_cwd = os.getcwd()
    orig_env = os.environ.pop("RL_VALIDATED_HYPOTHESES_PATH", None)
    orig_file = mod.__file__

    # Attach a manual handler to capture CRITICAL logs.
    captured_records: list[logging.LogRecord] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            captured_records.append(record)

    capture_handler = _CaptureHandler(level=logging.CRITICAL)
    rl_logger = logging.getLogger("rl.rl_drug_ranker")
    rl_logger.addHandler(capture_handler)
    rl_logger.setLevel(logging.CRITICAL)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            os.environ["RL_VALIDATED_HYPOTHESES_PATH"] = "/nonexistent/path/to/validated_hypotheses.csv"
            # Monkey-patch __file__ so module_dir resolves to tmpdir
            # (no validated_hypotheses.csv there).
            mod.__file__ = str(Path(tmpdir) / "rl_drug_ranker.py")
            # Re-invoke the loader directly (no reload needed — the
            # function reads __file__ at call time).
            result = mod._load_validated_hypotheses()
            # The result must be empty (no file found anywhere).
            assert result == [], (
                f"P4-003: expected empty result when no "
                f"validated_hypotheses.csv is found, got {result}"
            )
            # Check that a CRITICAL log was emitted about the missing file.
            critical_msgs = [
                r.getMessage() for r in captured_records
                if r.levelno >= logging.CRITICAL
                and "validated_hypotheses.csv" in r.getMessage()
                and "NOT FOUND" in r.getMessage()
            ]
            assert critical_msgs, (
                "P4-003: when validated_hypotheses.csv is not found in "
                "ANY candidate path, a CRITICAL log must be emitted "
                "(so operators can fix the deployment). No CRITICAL log "
                "was emitted. Captured records:\n"
                + "\n".join(r.getMessage() for r in captured_records)
            )
    finally:
        rl_logger.removeHandler(capture_handler)
        mod.__file__ = orig_file
        os.chdir(orig_cwd)
        if orig_env is not None:
            os.environ["RL_VALIDATED_HYPOTHESES_PATH"] = orig_env
        else:
            os.environ.pop("RL_VALIDATED_HYPOTHESES_PATH", None)


def test_p4_003_env_var_override_works():
    """RL_VALIDATED_HYPOTHESES_PATH env var must override the default paths.

    In production (Docker, Kubernetes, systemd), a deployment may want
    to point at a specific file (e.g., a ConfigMap mount in Kubernetes).
    The env var takes PRIORITY over the 3 default paths.
    """
    # Write a temp CSV with a known pair.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="vh_test_"
    ) as f:
        f.write("drug,disease\n")
        f.write("testdrug,testdisease\n")
        temp_csv_path = f.name
    try:
        os.environ["RL_VALIDATED_HYPOTHESES_PATH"] = temp_csv_path
        # Re-import to pick up the env var.
        import importlib
        import rl.rl_drug_ranker as mod
        importlib.reload(mod)
        # The env var path should have been loaded (testdrug, testdisease
        # should be in the validated hypotheses set, possibly merged
        # with the canonical file's pairs).
        validated = mod.VALIDATED_HYPOTHESES
        assert ("testdrug", "testdisease") in validated, (
            f"P4-003: RL_VALIDATED_HYPOTHESES_PATH override did not work. "
            f"Expected ('testdrug', 'testdisease') in VALIDATED_HYPOTHESES. "
            f"Got: {validated}"
        )
    finally:
        os.environ.pop("RL_VALIDATED_HYPOTHESES_PATH", None)
        os.unlink(temp_csv_path)
        # Reload to restore the original state.
        import importlib
        import rl.rl_drug_ranker as mod
        importlib.reload(mod)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
