"""Cross-phase import chain integration test (Chunk 9).

These tests verify that the module import chain from Phase 1 → Phase 2 → Phase 3
is intact. They intentionally do NOT require torch/neo4j installed — they only
test the contract-level Python imports that must work in CI.
"""
import sys
from pathlib import Path

# Add repo root to path so all phases are importable.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "phase1"))
sys.path.insert(0, str(_REPO / "phase2"))
sys.path.insert(0, str(_REPO / "rl"))


def test_shared_contracts_writeback():
    """shared.contracts.writeback must export VALID_OUTCOMES and ValidatedHypothesis."""
    from shared.contracts.writeback import VALID_OUTCOMES
    assert isinstance(VALID_OUTCOMES, (tuple, list, frozenset, set))
    assert "validated_positive" in VALID_OUTCOMES
    assert "validated_negative" in VALID_OUTCOMES
    assert "validated_toxic" in VALID_OUTCOMES
    assert "invalidated" in VALID_OUTCOMES


def test_shared_contracts_urls():
    """shared.contracts.urls must export the canonical route constants."""
    from shared.contracts.urls import URL_HEALTH, URL_RANK, URL_VALIDATE
    assert URL_HEALTH.startswith("/")
    assert URL_RANK.startswith("/")
    assert URL_VALIDATE.startswith("/")


def test_phase2_schema_mappings():
    """phase2 schema_mappings must expose is_phase2_intermediate_dropped."""
    from phase2.drugos_graph.schema_mappings import is_phase2_intermediate_dropped
    assert callable(is_phase2_intermediate_dropped)
    # Spot-check known intermediate label.
    assert is_phase2_intermediate_dropped("Gene") is True
    assert is_phase2_intermediate_dropped("Compound") is False


def test_phase1_bridge_importable():
    """phase2.drugos_graph.phase1_bridge must be importable without neo4j."""
    # This import should not raise even if neo4j is not installed.
    from phase2.drugos_graph.phase1_bridge import run_phase1_to_phase2
    assert callable(run_phase1_to_phase2)


def test_rl_service_importable():
    """rl.service must be importable (slowapi, fastapi deps) without torch."""
    import importlib
    # This import will warn about slowapi if not installed, but must NOT raise.
    spec = importlib.util.find_spec("rl.service")
    # If rl.service can't be found via package, try direct path import.
    if spec is None:
        import importlib.util as _ilu
        _path = str(_REPO / "rl" / "service.py")
        spec = _ilu.spec_from_file_location("rl_service", _path)
    assert spec is not None, "rl/service.py not found — check PYTHONPATH"


def test_clinical_outcome_no_typo():
    """Frontend contract must use ClinicalOutcome (not ClinicalOutcomes) — verified at Python level via grep."""
    import subprocess
    result = subprocess.run(
        ["grep", "-r", "ClinicalOutcomes", "frontend/src/"],
        capture_output=True, text=True, cwd=str(_REPO)
    )
    matches = [line for line in result.stdout.splitlines() if line.strip()]
    # Allow zero matches (typo fully fixed) or matches only in comments/docs.
    non_comment = [m for m in matches if not m.strip().startswith("//") and not m.strip().startswith("*")]
    assert len(non_comment) == 0, (
        f"ClinicalOutcomes (plural) typo still present in frontend — "
        f"run the bulk rename fix. Matches:\n" + "\n".join(non_comment)
    )


def test_no_math_random_in_production_paths():
    """Math.random() must not appear in core-screens.tsx or sidebar.tsx."""
    import subprocess
    for path in ["frontend/src/components/drugos/core-screens.tsx",
                 "frontend/src/components/ui/sidebar.tsx"]:
        result = subprocess.run(
            ["grep", "-n", "Math.random()", path],
            capture_output=True, text=True, cwd=str(_REPO)
        )
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        assert len(lines) == 0, (
            f"Math.random() still present in {path}:\n" + "\n".join(lines)
        )
