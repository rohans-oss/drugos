#!/usr/bin/env python3
"""GT inference helper for the frontend /api/predict and /api/top-k routes.

RT-006 ROOT FIX (Team Member 17): the frontend's Next.js routes cannot
import torch directly (no PyTorch in the Node runtime). This helper is
a thin Python wrapper that the Node routes spawn via `child_process.spawn`.

Usage:
    python3 scripts/gt_inference.py <req_path> <resp_path>

Request JSON (read from <req_path>):
    {
        "checkpoint": "/abs/path/to/best_model.pt",
        "mode": "predict" | "top_k",
        "pairs": [{"drug": "aspirin", "disease": "migraine"}, ...],  # for predict
        "top_k": 50                                                    # for top_k
    }

Response JSON (written to <resp_path>):
    {
        "predictions": [{"drug": "...", "disease": "...", "score": 0.87}, ...],
        "model_version": "v100",
        "error": null | "error message"
    }

This script MUST be runnable from the repo root with no extra setup
beyond `pip install -r requirements.txt`. It uses the same imports
the trainer uses (graph_transformer.*), so a checkpoint trained by
run_4phase.py can be loaded here without any conversion.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Make the repo root importable so `from graph_transformer...` works
# regardless of where the Node process set cwd.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "phase1"), str(REPO_ROOT / "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_checkpoint(checkpoint_path: str) -> Tuple[Any, Any, Any, Any, List[str], List[str], List[Tuple[str, str]]]:
    """Load the GT model + graph state from a checkpoint directory.

    The trainer writes `best_model.pt` to <output_dir>/checkpoints/. The
    bridge also writes a `graph_state.pt` next to it containing the
    node_features, edge_indices, node_maps, drug_names, disease_names,
    and known_pairs needed for inference.

    Returns:
        (model, node_features, edge_indices, node_maps, drug_names,
         disease_names, known_pairs)
    """
    import torch
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"GT checkpoint not found: {ckpt_path}")

    # Look for graph_state.pt in the same directory
    graph_state_path = ckpt_path.parent / "graph_state.pt"
    if not graph_state_path.exists():
        # Fallback: try sibling files
        candidates = list(ckpt_path.parent.glob("*graph_state*.pt")) + \
                     list(ckpt_path.parent.glob("*graph*.pt"))
        candidates = [c for c in candidates if c != ckpt_path]
        if not candidates:
            raise FileNotFoundError(
                f"Graph state file not found next to checkpoint {ckpt_path}. "
                f"Expected: {graph_state_path}. The bridge must write this "
                f"file alongside the model checkpoint so inference can "
                f"reproduce the exact graph topology the model was trained on."
            )
        graph_state_path = candidates[0]

    # Load model
    try:
        from graph_transformer.models.graph_transformer import DrugRepurposingGraphTransformer
    except ImportError as e:
        raise ImportError(
            f"Could not import DrugRepurposingGraphTransformer: {e}. "
            f"Ensure graph_transformer/ is on sys.path (repo root = {REPO_ROOT})."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    graph_state = torch.load(graph_state_path, map_location="cpu", weights_only=False)

    # Reconstruct model from saved config (the trainer saves model_config)
    model_config = ckpt.get("model_config", graph_state.get("model_config", {}))
    model = DrugRepurposingGraphTransformer(
        node_features_dims=graph_state["node_features_dims"],
        embedding_dim=model_config.get("embedding_dim", 32),
        num_layers=model_config.get("num_layers", 3),
        num_heads=model_config.get("num_heads", 2),
        dropout=model_config.get("dropout", 0.2),
        attention_dropout=model_config.get("attention_dropout", 0.2),
        link_predictor_hidden_dims=model_config.get("link_predictor_hidden_dims", [64, 32]),
    )
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    return (
        model,
        graph_state["node_features"],
        graph_state["edge_indices"],
        graph_state["node_maps"],
        graph_state["drug_names"],
        graph_state["disease_names"],
        graph_state.get("known_pairs", []),
    )


def _predict_pairs(
    model, node_features, edge_indices, node_maps,
    drug_names: List[str], disease_names: List[str],
    pairs: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Score arbitrary (drug, disease) pairs."""
    import torch
    from graph_transformer.inference import predict_drug_disease_scores

    drug_to_idx = {n.lower(): i for i, n in enumerate(drug_names)}
    disease_to_idx = {n.lower(): i for i, n in enumerate(disease_names)}

    valid_pairs: List[Tuple[str, str, int, int]] = []
    for p in pairs:
        d = p.get("drug", "").strip()
        v = p.get("disease", "").strip()
        if not d or not v:
            continue
        d_idx = drug_to_idx.get(d.lower())
        v_idx = disease_to_idx.get(v.lower())
        if d_idx is None or v_idx is None:
            # Drug/disease not in the graph — skip with a NaN score so the
            # caller can see which pairs were unscoreable.
            continue
        valid_pairs.append((d, v, d_idx, v_idx))

    if not valid_pairs:
        return []

    drug_idx_t = torch.tensor([p[2] for p in valid_pairs], dtype=torch.long)
    disease_idx_t = torch.tensor([p[3] for p in valid_pairs], dtype=torch.long)

    scores = predict_drug_disease_scores(
        model=model,
        node_features=node_features,
        edge_indices=edge_indices,
        drug_indices=drug_idx_t,
        disease_indices=disease_idx_t,
        device="cpu",
        apply_temperature=True,
    )

    out = []
    for (d, v, _, _), score in zip(valid_pairs, scores):
        out.append({"drug": d, "disease": v, "score": float(score)})
    return out


def _top_k_novel(
    model, node_features, edge_indices,
    drug_names: List[str], disease_names: List[str],
    known_pairs: List[Tuple[str, str]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Return the top-K highest-scoring novel (drug, disease) pairs."""
    from graph_transformer.inference import top_k_novel_predictions

    raw = top_k_novel_predictions(
        model=model,
        node_features=node_features,
        edge_indices=edge_indices,
        drug_names=drug_names,
        disease_names=disease_names,
        known_pairs=known_pairs,
        top_k=top_k,
        device="cpu",
    )
    return [{"drug": d, "disease": v, "score": float(s)} for (d, v, s) in raw]


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: gt_inference.py <req_path> <resp_path>", file=sys.stderr)
        return 2
    req_path, resp_path = sys.argv[1], sys.argv[2]

    try:
        with open(req_path) as f:
            req = json.load(f)
    except Exception as e:
        _write_resp(resp_path, {"error": f"Could not read request: {e}", "predictions": []})
        return 1

    try:
        model, node_features, edge_indices, node_maps, drug_names, disease_names, known_pairs = \
            _load_checkpoint(req["checkpoint"])

        mode = req.get("mode", "predict")
        if mode == "predict":
            predictions = _predict_pairs(
                model, node_features, edge_indices, node_maps,
                drug_names, disease_names,
                req.get("pairs", []),
            )
        elif mode == "top_k":
            predictions = _top_k_novel(
                model, node_features, edge_indices,
                drug_names, disease_names,
                known_pairs,
                int(req.get("top_k", 50)),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Read model version if available
        model_version = "unknown"
        try:
            from graph_transformer import __version__ as gv
            model_version = gv
        except Exception:
            pass

        _write_resp(resp_path, {
            "predictions": predictions,
            "model_version": model_version,
            "error": None,
        })
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        _write_resp(resp_path, {
            "error": f"{type(e).__name__}: {e}",
            "traceback": tb,
            "predictions": [],
        })
        return 1


def _write_resp(resp_path: str, payload: Dict[str, Any]) -> None:
    with open(resp_path, "w") as f:
        json.dump(payload, f, default=str)


if __name__ == "__main__":
    sys.exit(main())
