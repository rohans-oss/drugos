"""
Atomic checkpoint save test — issue #351.

ACCEPTANCE CRITERIA (per audit task #351):
    Verify checkpoint save is ATOMIC — a crash during save does not
    corrupt the file.

SCIENTIFIC INVARIANT:
    The GT model checkpoint is a multi-megabyte pickle file containing
    model weights, optimizer state, and graph metadata. If the save
    process is interrupted (OOM, signal, disk full, power loss), the
    file must NOT be left in a corrupt state. The next training run
    must load EITHER the old checkpoint (intact) OR the new checkpoint
    (intact) — NEVER a half-written file that fails to unpickle.

The atomic-save pattern (POSIX):
    1. Write to a temp file in the SAME directory as the target
       (so os.rename is a single inode operation — atomic).
    2. fsync the temp file (so the bytes are durable on disk before
       the rename, surviving power loss).
    3. os.replace(temp, target) — atomic rename. Either the old file
       or the new file is visible, never a partial write.

This test verifies the pattern by:
    1. Saving a baseline checkpoint.
    2. Patching torch.save to raise mid-write (simulating a crash).
    3. Verifying the baseline checkpoint is still loadable.
    4. Verifying the temp file is cleaned up.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tiny_checkpoint(checkpoint_dir: Path) -> str:
    """Build a tiny GT checkpoint + graph_state.pt for atomic-save test."""
    import torch
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
    )

    num_drugs, num_diseases = 3, 3
    drug_names = ["metformin", "aspirin", "warfarin"]
    disease_names = ["diabetes", "pain", "epilepsy"]

    node_features = {
        "drug": torch.randn(num_drugs, 8),
        "disease": torch.randn(num_diseases, 8),
    }
    edge_indices = {
        ("drug", "treats", "disease"): torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
    }
    node_maps = {
        "drug": {n: i for i, n in enumerate(drug_names)},
        "disease": {n: i for i, n in enumerate(disease_names)},
    }
    known_pairs = [("metformin", "diabetes"), ("aspirin", "pain")]

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
        "best_val_auc": 0.75,
        "best_val_loss": 0.5,
        "best_epoch": 10,
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

    return str(checkpoint_path)


def _write_validated_csv(csv_path: Path):
    """Write a tiny validated_hypotheses.csv with 1 positive pair."""
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["drug", "disease", "outcome", "validated_by",
                  "validation_study_id", "validated_at", "notes",
                  "original_gt_score", "original_rl_rank", "writeback_version"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "drug": "aspirin",
            "disease": "diabetes",
            "outcome": "validated_positive",
            "validated_by": "atomic_test_partner",
            "validation_study_id": "NCT_ATOMIC_001",
            "validated_at": "2026-01-01T00:00:00+00:00",
            "notes": "atomic save test",
            "original_gt_score": "0.85",
            "original_rl_rank": "1",
            "writeback_version": "2.0.0-shared-contract",
        })


# ---------------------------------------------------------------------------
# Test 1: successful save leaves a valid checkpoint
# ---------------------------------------------------------------------------

def test_successful_save_produces_valid_checkpoint(tmp_path, monkeypatch):
    """Baseline: a successful retrain_on_validated produces a loadable checkpoint."""
    validated_csv = tmp_path / "validated_hypotheses.csv"
    monkeypatch.setenv("VALIDATED_HYPOTHESES_CSV", str(validated_csv))
    _write_validated_csv(validated_csv)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _build_tiny_checkpoint(checkpoint_dir)

    from graph_transformer.training.trainer import retrain_on_validated
    new_checkpoint = str(checkpoint_dir / "gt_checkpoint_finetuned.pt")

    result = retrain_on_validated(
        checkpoint_path=checkpoint_path,
        validated_csv_path=str(validated_csv),
        output_checkpoint_path=new_checkpoint,
        fine_tune_epochs=1,
        learning_rate=1e-4,
    )

    assert os.path.exists(new_checkpoint), "new checkpoint not saved"

    import torch
    bundle = torch.load(new_checkpoint, map_location="cpu", weights_only=False)
    assert "model_state_dict" in bundle
    assert "known_pairs" in bundle
    assert ("aspirin", "diabetes") in bundle["known_pairs"]


# ---------------------------------------------------------------------------
# Test 2: crash during save leaves OLD checkpoint intact
# ---------------------------------------------------------------------------

def test_crash_during_save_leaves_old_checkpoint_intact(tmp_path, monkeypatch):
    """Issue #351: a crash during the atomic save MUST NOT corrupt the file.

    We patch torch.save to raise mid-write. The atomic-save pattern
    (temp file + os.replace) must ensure the OLD checkpoint is still
    loadable and no temp file is left behind.
    """
    validated_csv = tmp_path / "validated_hypotheses.csv"
    monkeypatch.setenv("VALIDATED_HYPOTHESES_CSV", str(validated_csv))
    _write_validated_csv(validated_csv)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _build_tiny_checkpoint(checkpoint_dir)

    # Snapshot the original checkpoint bytes for comparison.
    original_bytes = Path(checkpoint_path).read_bytes()
    original_size = len(original_bytes)

    from graph_transformer.training import trainer as trainer_mod

    # Patch torch.save inside the trainer module to raise mid-write.
    # This simulates a crash (OOM, signal, disk full) during the save.
    real_torch_save = trainer_mod._torch.save if hasattr(trainer_mod, '_torch') else None

    call_count = [0]
    def crashing_save(obj, path, **kwargs):
        call_count[0] += 1
        # Write partial bytes to the temp file, then raise.
        # This simulates a partial write before the crash.
        try:
            with open(path, "wb") as f:
                f.write(b"\x80\x04PARTIAL_WRITE_CRASH_SIMULATED")
                f.flush()
        except Exception:
            pass
        raise RuntimeError("SIMULATED CRASH DURING SAVE")

    # Apply the patch. The trainer imports torch inside the function
    # (``import torch as _torch``), so we need to patch at the module
    # level where it's actually looked up.
    import torch
    with patch.object(torch, "save", side_effect=crashing_save):
        from graph_transformer.training.trainer import retrain_on_validated

        with pytest.raises(RuntimeError, match="SIMULATED CRASH"):
            retrain_on_validated(
                checkpoint_path=checkpoint_path,
                validated_csv_path=str(validated_csv),
                output_checkpoint_path=checkpoint_path,  # overwrite same path
                fine_tune_epochs=1,
                learning_rate=1e-4,
            )

    # CRITICAL ASSERTION 1: the OLD checkpoint must still exist.
    assert os.path.exists(checkpoint_path), (
        "old checkpoint DISAPPEARED after crash — atomic save failed"
    )

    # CRITICAL ASSERTION 2: the OLD checkpoint must still be loadable.
    # If it were corrupted, torch.load would raise.
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in bundle, (
        "old checkpoint is corrupted — model_state_dict missing"
    )
    assert ("aspirin", "diabetes") not in bundle.get("known_pairs", []), (
        "old checkpoint was OVERWRITTEN with new data despite the crash — "
        "atomic save failed"
    )

    # CRITICAL ASSERTION 3: no temp file left behind.
    tmp_files = list(checkpoint_dir.glob(".gt_checkpoint_tmp_*"))
    assert len(tmp_files) == 0, (
        f"temp file left behind after crash: {tmp_files}. "
        f"The atomic-save cleanup is broken."
    )


# ---------------------------------------------------------------------------
# Test 3: temp file is in the SAME directory as the target
# ---------------------------------------------------------------------------

def test_temp_file_in_same_directory_as_target(tmp_path, monkeypatch):
    """Issue #351: the temp file MUST be in the same directory as the target.

    os.rename / os.replace is only atomic when the source and destination
    are on the SAME filesystem (same inode table). If the temp file is in
    /tmp and the target is in /data, os.replace falls back to a non-atomic
    copy+delete, leaving a window where the file is partially written.

    The trainer uses tempfile.mkstemp(dir=str(_out_dir)) to ensure the
    temp file is in the same directory. This test verifies that contract.
    """
    validated_csv = tmp_path / "validated_hypotheses.csv"
    monkeypatch.setenv("VALIDATED_HYPOTHESES_CSV", str(validated_csv))
    _write_validated_csv(validated_csv)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _build_tiny_checkpoint(checkpoint_dir)

    # Intercept tempfile.mkstemp to record the dir argument.
    import tempfile
    real_mkstemp = tempfile.mkstemp
    recorded_dirs = []

    def recording_mkstemp(*args, **kwargs):
        dir_arg = kwargs.get("dir", args[0] if args else None)
        recorded_dirs.append(dir_arg)
        return real_mkstemp(*args, **kwargs)

    with patch.object(tempfile, "mkstemp", side_effect=recording_mkstemp):
        from graph_transformer.training.trainer import retrain_on_validated
        new_checkpoint = str(checkpoint_dir / "gt_finetuned.pt")

        retrain_on_validated(
            checkpoint_path=checkpoint_path,
            validated_csv_path=str(validated_csv),
            output_checkpoint_path=new_checkpoint,
            fine_tune_epochs=1,
            learning_rate=1e-4,
        )

    # The mkstemp call must have been called with a dir argument pointing
    # to the SAME directory as the target checkpoint.
    assert len(recorded_dirs) > 0, "tempfile.mkstemp was never called"
    target_dir = str(Path(new_checkpoint).parent)
    for recorded_dir in recorded_dirs:
        assert recorded_dir == target_dir, (
            f"temp file dir mismatch: mkstemp was called with dir={recorded_dir}, "
            f"but target dir is {target_dir}. The temp file is NOT in the same "
            f"directory as the target — os.replace is NOT atomic."
        )
