"""
End-to-end data flywheel test — issue #349.

ACCEPTANCE CRITERIA (per audit task #349):
    1. Validate a hypothesis via the writeback API.
    2. Verify it appears in writeback CSV with correct `outcome`.
    3. Trigger retrain → trainer reads the CSV, runs N epochs,
       saves a new checkpoint.
    4. RL ranker loads the new bonus/penalty.
    5. toxic-pair validation results in a NEGATIVE reward.

This test exercises the FULL flywheel:
    Phase 4 writeback -> Phase 1 CSV -> Phase 3 trainer fine-tune
                                       -> Phase 4 RL ranker loads new bonus

It does NOT mock the writeback, the CSV read/write, or the trainer's
fine-tune loop. It DOES mock the Neo4j driver (no live Neo4j in CI)
and uses a tiny synthetic graph for the GT model (so the test runs in
<30 seconds on CPU).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

# Ensure repo root is on sys.path so we can import phase4, graph_transformer,
# rl, shared.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_flywheel_env(tmp_path, monkeypatch):
    """Isolate the flywheel to a tmp directory.

    Sets VALIDATED_HYPOTHESES_CSV and PHASE3_RETRAIN_TRIGGER to paths
    inside tmp_path so the test does not pollute the repo's checked-in
    CSVs. Also unsets DRUGOS_NEO4J_URI so writeback_to_phase2 is a no-op.
    """
    validated_csv = tmp_path / "validated_hypotheses.csv"
    retrain_trigger = tmp_path / "retrain_triggered.json"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("VALIDATED_HYPOTHESES_CSV", str(validated_csv))
    monkeypatch.setenv("PHASE1_VALIDATED_CSV", str(validated_csv))
    monkeypatch.setenv("PHASE3_RETRAIN_TRIGGER", str(retrain_trigger))
    monkeypatch.delenv("DRUGOS_NEO4J_URI", raising=False)

    return {
        "tmp_path": tmp_path,
        "validated_csv": validated_csv,
        "retrain_trigger": retrain_trigger,
        "checkpoint_dir": checkpoint_dir,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_hypothesis(
    drug: str,
    disease: str,
    outcome: str,
    validated_by: str = "test_partner",
    study_id: str = "NCT_TEST_001",
) -> Dict[str, Any]:
    """Call phase4.writeback.write_validated_hypothesis and return result."""
    from phase4.writeback import write_validated_hypothesis
    return write_validated_hypothesis(
        drug=drug,
        disease=disease,
        outcome=outcome,
        validated_by=validated_by,
        validation_study_id=study_id,
    )


def _read_validated_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Read validated_hypotheses.csv as a list of dicts."""
    if not csv_path.exists():
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_tiny_checkpoint(checkpoint_dir: Path) -> Tuple[str, str]:
    """Build a tiny GT checkpoint + graph_state.pt for fine-tuning.

    Returns (checkpoint_path, graph_state_path).
    """
    import torch
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )

    # Tiny graph: 4 drugs, 4 diseases, 3 known treats edges.
    num_drugs, num_diseases = 4, 4
    drug_names = ["metformin", "aspirin", "warfarin", "thalidomide"]
    disease_names = ["diabetes", "pain", "epilepsy", "multiple myeloma"]

    node_features = {
        "drug": torch.randn(num_drugs, 8),
        "disease": torch.randn(num_diseases, 8),
    }
    edge_indices = {
        ("drug", "treats", "disease"): torch.tensor(
            [[0, 1, 0], [0, 1, 3]], dtype=torch.long
        ),
    }
    node_maps = {
        "drug": {n: i for i, n in enumerate(drug_names)},
        "disease": {n: i for i, n in enumerate(disease_names)},
    }
    known_pairs = [
        ("metformin", "diabetes"),
        ("aspirin", "pain"),
        ("metformin", "multiple myeloma"),
    ]

    model_config = {
        "embedding_dim": 16,
        "num_layers": 2,
        "num_heads": 2,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "link_predictor_hidden_dims": [16, 8],
    }
    node_features_dims = {"drug": 8, "disease": 8}

    model = DrugRepurposingGraphTransformer(
        feature_dims=node_features_dims,
        embedding_dim=model_config["embedding_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        attention_dropout=model_config["attention_dropout"],
        link_predictor_hidden_dims=model_config["link_predictor_hidden_dims"],
        edge_types=list(edge_indices.keys()),
        node_types=list(node_features.keys()),
        min_edge_types=1,  # tiny test graph — ablation mode
    )

    checkpoint_path = checkpoint_dir / "gt_checkpoint.pt"
    graph_state_path = checkpoint_dir / "graph_state.pt"

    bundle = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "node_maps": node_maps,
        "drug_names": drug_names,
        "disease_names": disease_names,
        "known_pairs": known_pairs,
        "best_val_auc": 0.0,
        "best_val_loss": float("inf"),
        "best_epoch": 0,
        "best_state_dict": model.state_dict(),
    }
    torch.save(bundle, str(checkpoint_path))

    graph_state = {
        "node_features": node_features,
        "edge_indices": edge_indices,
        "node_maps": node_maps,
        "model_config": model_config,
        "node_features_dims": node_features_dims,
        "feature_dims": node_features_dims,
    }
    torch.save(graph_state, str(graph_state_path))

    return str(checkpoint_path), str(graph_state_path)


# ---------------------------------------------------------------------------
# Test 1: validate → CSV → trainer → new checkpoint
# ---------------------------------------------------------------------------

def test_flywheel_validate_then_retrain(isolated_flywheel_env):
    """Issue #349 acceptance criteria 1-3.

    1. Validate a hypothesis via write_validated_hypothesis.
    2. Verify it appears in the CSV with outcome='validated_positive'.
    3. Trigger retrain_on_validated → trainer reads CSV, runs N epochs,
       saves new checkpoint.
    """
    env = isolated_flywheel_env

    # Step 1: validate a hypothesis (positive outcome).
    result = _validate_hypothesis(
        drug="aspirin",
        disease="diabetes",
        outcome="validated_positive",
        validated_by="test_pharma",
        study_id="NCT_FLYWHEEL_001",
    )

    # Step 2: verify it appears in the CSV with the correct outcome.
    rows = _read_validated_csv(env["validated_csv"])
    assert len(rows) == 1, f"expected 1 row, got {len(rows)}"
    assert rows[0]["drug"] == "aspirin"
    assert rows[0]["disease"] == "diabetes"
    assert rows[0]["outcome"] == "validated_positive"
    assert rows[0]["validated_by"] == "test_pharma"
    assert rows[0]["validation_study_id"] == "NCT_FLYWHEEL_001"

    # Step 3: trigger retrain_on_validated.
    checkpoint_path, graph_state_path = _build_tiny_checkpoint(
        env["checkpoint_dir"]
    )

    from graph_transformer.training.trainer import retrain_on_validated

    new_checkpoint = str(env["checkpoint_dir"] / "gt_checkpoint_finetuned.pt")
    retrain_result = retrain_on_validated(
        checkpoint_path=checkpoint_path,
        validated_csv_path=str(env["validated_csv"]),
        output_checkpoint_path=new_checkpoint,
        fine_tune_epochs=3,
        learning_rate=1e-4,
    )

    # Verify the trainer ran N epochs (not 0).
    assert retrain_result["fine_tune_epochs"] == 3, (
        f"expected fine_tune_epochs=3, got {retrain_result['fine_tune_epochs']}. "
        f"Full result: {retrain_result}"
    )
    # Verify a new checkpoint was saved.
    assert os.path.exists(new_checkpoint), (
        f"new checkpoint not saved at {new_checkpoint}"
    )
    # Verify the validated pair was added to known_pairs in the new checkpoint.
    import torch
    new_bundle = torch.load(new_checkpoint, map_location="cpu", weights_only=False)
    new_known_pairs = new_bundle.get("known_pairs", [])
    assert ("aspirin", "diabetes") in new_known_pairs, (
        f"validated pair not in new known_pairs: {new_known_pairs}"
    )


# ---------------------------------------------------------------------------
# Test 2: retrain trigger JSON → load_validated_for_retraining → fine-tune
# ---------------------------------------------------------------------------

def test_flywheel_retrain_trigger_to_finetune(isolated_flywheel_env):
    """Issue #349 — verify the load_validated_for_retraining path.

    The full flywheel uses writeback_to_phase3 to write a JSON trigger,
    then load_validated_for_retraining reads the trigger and calls
    retrain_on_validated. This test verifies that chain.

    ISSUE #337 ROOT FIX verification: the temp CSV written by
    load_validated_for_retraining MUST use the `outcome` column with
    canonical enum values, NOT the `validated` column with "true"/"false".
    """
    env = isolated_flywheel_env

    # Step 1: validate TWO hypotheses (1 positive, 1 toxic).
    _validate_hypothesis("metformin", "pain", "validated_positive",
                         validated_by="partner_a")
    _validate_hypothesis("warfarin", "epilepsy", "validated_toxic",
                         validated_by="partner_b")

    # Verify both are in the CSV.
    rows = _read_validated_csv(env["validated_csv"])
    assert len(rows) == 2

    # Verify the retrain trigger JSON was written (writeback_to_phase3
    # is called by write_validated_hypothesis).
    assert env["retrain_trigger"].exists(), (
        f"retrain trigger not written at {env['retrain_trigger']}"
    )
    with open(env["retrain_trigger"]) as f:
        trigger_entries = json.load(f)
    assert len(trigger_entries) == 2

    # Step 2: build a checkpoint with graph_state.
    checkpoint_path, _ = _build_tiny_checkpoint(env["checkpoint_dir"])

    # Step 3: call load_validated_for_retraining (which reads the JSON
    # trigger, writes a temp CSV with the canonical schema, then calls
    # retrain_on_validated).
    from graph_transformer.training.trainer import load_validated_for_retraining

    new_checkpoint = str(env["checkpoint_dir"] / "gt_checkpoint_v2.pt")
    result = load_validated_for_retraining(
        checkpoint_path=checkpoint_path,
        retrain_trigger_path=str(env["retrain_trigger"]),
        output_checkpoint_path=new_checkpoint,
        fine_tune_epochs=2,
    )

    # ISSUE #337 ROOT FIX verification: the temp CSV must have used the
    # `outcome` column with `validated_positive` value, so the positive
    # pair was actually added. If the bug were still present, the temp
    # CSV would have `validated=true` and retrain_on_validated would
    # silently skip ALL rows → validated_pairs_added=0.
    assert result["positive_pairs"] == 1, (
        f"expected 1 positive pair, got {result['positive_pairs']}. "
        f"This indicates the #337 fix is NOT working — the temp CSV "
        f"is still using the `validated` column instead of `outcome`."
    )
    assert result["negative_pairs"] == 1, (
        f"expected 1 negative (toxic) pair, got {result['negative_pairs']}"
    )
    # The positive pair should have been added to known_pairs.
    assert result["validated_pairs_added"] >= 1, (
        f"expected >=1 validated pair added, got {result['validated_pairs_added']}. "
        f"If 0, the #337 fix is broken."
    )


# ---------------------------------------------------------------------------
# Test 3: RL ranker loads new bonus after validation
# ---------------------------------------------------------------------------

def test_rl_ranker_loads_validated_bonus(isolated_flywheel_env):
    """Issue #349 acceptance criterion 4: RL ranker loads new bonus.

    After a validated_positive hypothesis is written back, the RL
    ranker's _load_validated_hypotheses() should pick it up.
    """
    env = isolated_flywheel_env

    # Write a validated hypothesis.
    _validate_hypothesis("aspirin", "pain", "validated_positive",
                         validated_by="partner_x")

    # Import the RL ranker's loader.
    from rl.rl_drug_ranker import _load_validated_hypotheses

    # The loader should return the pair we just wrote.
    validated = _load_validated_hypotheses()

    # Convert to lowercase for comparison (the loader lowercases).
    found = any(
        (d.lower() == "aspirin" and dis.lower() == "pain")
        for d, dis in validated
    )
    assert found, (
        f"validated pair (aspirin, pain) not loaded by RL ranker. "
        f"Loaded pairs: {validated}"
    )
