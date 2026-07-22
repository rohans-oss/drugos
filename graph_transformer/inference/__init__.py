"""Inference utilities for the Graph Transformer."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..data import LABEL_LEAKING_EDGES


@torch.no_grad()
def predict_drug_disease_scores(
    model: Any,
    node_features: Dict[str, torch.Tensor],
    edge_indices: Dict,
    drug_indices: torch.Tensor,
    disease_indices: torch.Tensor,
    batch_size: int = 1024,
    exclude_edges: Any = None,
    device: str = "cpu",
    apply_temperature: bool = True,
) -> np.ndarray:
    """Predict probability scores for a list of (drug, disease) pairs.

    V90 ROOT FIX (BUG #46): encode the graph ONCE, then extract per-batch
    embeddings. The previous code called ``model(...)`` (i.e.,
    ``model.forward(...)``) per batch, which internally calls
    ``model.encode(...)`` (the expensive Graph Transformer forward pass)
    for EVERY batch. For N batches, this ran N full graph encodings
    instead of 1. On a 10K-node graph with 100 batches, this was 100x
    slower than necessary.

    The fix mirrors ``trainer.evaluate()`` and ``evaluate_link_prediction()``:
    encode the graph ONCE at the start, then for each batch extract the
    drug/disease embeddings via indexing and call
    ``link_predictor.forward()`` directly (no encode call). This reduces
    encoder calls from N_batches to 1, cutting inference compute by
    ~N_batches×.

    Args:
        model: Trained DrugRepurposingGraphTransformer.
        node_features: Dict of node feature tensors.
        edge_indices: Dict of edge index tensors.
        drug_indices: (N,) drug node indices.
        disease_indices: (N,) disease node indices.
        batch_size: Batch size.
        exclude_edges: Edge types to exclude (defaults to LABEL_LEAKING_EDGES).
        device: Device.
        apply_temperature: If True, apply the link predictor's learned
            temperature (calibrated probabilities).

    Returns:
        (N,) numpy array of probabilities in [0, 1].
    """
    if exclude_edges is None:
        exclude_edges = set(LABEL_LEAKING_EDGES)

    # V90 ROOT FIX (BUG #19, P1): save prior training state and restore
    # in finally. The previous code called ``model.eval()`` and NEVER
    # restored training mode. If predict_drug_disease_scores was called
    # mid-training (by a background thread, an API server, or an
    # interactive notebook), it silently disabled dropout and BatchNorm
    # updates for the rest of the process.
    #
    # V91 ROOT FIX (dead code removal): the previous edit prepended a
    # BUG #19 try/finally block but LEFT the old per-batch-encode body
    # as the executing path (calling model(...) per batch, which re-
    # encodes the graph every batch -- BUG #46 NOT actually fixed), and
    # appended the encode-once optimization as DEAD CODE after the
    # finally (unreachable because the try block returns). This combined
    # both defects: BUG #46 was never in effect, AND 35 lines of dead
    # code misled reviewers. The fix below merges both: the encode-once
    # optimization runs INSIDE the try/finally, so BUG #19 (save/restore
    # training mode) AND BUG #46 (encode once) are both live.
    prior_training = model.training
    model.eval()
    try:
        model.to(device)
        nf = {k: v.to(device) for k, v in node_features.items()}
        ei = {k: v.to(device) for k, v in edge_indices.items()}

        # V90 BUG #46: encode the graph ONCE for ALL pairs (not per batch).
        # The encoder processes the entire graph through the Graph Transformer
        # layers, producing node embeddings. This is the expensive operation.
        # Calling model(...) per batch re-encodes every batch, wasting
        # N_batches × compute. Encode once, then index per batch.
        embeddings = model.encode(
            nf, ei,
            exclude_edges_override=set(exclude_edges),
        )
        drug_emb_all = embeddings["drug"]
        disease_emb_all = embeddings["disease"]

        all_probs: List[torch.Tensor] = []
        n = len(drug_indices)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            d_idx = drug_indices[start:end].to(device)
            ds_idx = disease_indices[start:end].to(device)
            # V90 BUG #46: extract per-batch embeddings via indexing (NO
            # redundant encode call). Then call link_predictor.forward
            # directly with apply_temperature (V4 B-F5 fix preserved).
            drug_emb_batch = drug_emb_all[d_idx]
            disease_emb_batch = disease_emb_all[ds_idx]
            probs = model.link_predictor.forward(
                drug_emb_batch, disease_emb_batch,
                apply_temperature=apply_temperature,
            ).squeeze(-1)
            all_probs.append(probs.cpu())

        return torch.cat(all_probs).numpy()
    finally:
        # V90 ROOT FIX (BUG #19): restore the prior training state so
        # callers that invoke this mid-training do not silently lose
        # dropout / BatchNorm updates for the rest of the process.
        model.train(prior_training)


@torch.no_grad()
def top_k_novel_predictions(
    model: Any,
    node_features: Dict[str, torch.Tensor],
    edge_indices: Dict,
    drug_names: List[str],
    disease_names: List[str],
    known_pairs: List[Tuple[str, str]],
    top_k: int = 50,
    exclude_edges: Any = None,
    device: str = "cpu",
) -> List[Tuple[str, str, float]]:
    """Return the top-K highest-scoring NOVEL (drug, disease) pairs.

    "Novel" = (drug, disease) not in ``known_pairs``. This is what the
    V1 launch contract requires for the PubMed literature cross-check
    (Phase 6 DOCX: "We take the model's top 50 novel predictions").

    Args:
        model: Trained model.
        node_features: Node features dict.
        edge_indices: Edge indices dict.
        drug_names: List of all drug names (index = node index).
        disease_names: List of all disease names (index = node index).
        known_pairs: List of (drug_name, disease_name) tuples that are
            already known and should be excluded from the "novel" set.
        top_k: Number of top novel predictions to return.
        exclude_edges: Edge types to exclude (defaults to LABEL_LEAKING_EDGES).
        device: Device.

    Returns:
        List of (drug_name, disease_name, score) tuples, sorted by score desc.
    """
    num_drugs = len(drug_names)
    num_diseases = len(disease_names)

    # P3-040 + P3-004 ROOT FIX (v113 forensic): use the new
    # ``predict_all_pairs_dual`` method to compute BOTH raw and
    # calibrated scores in a SINGLE encode pass. The previous code
    # called ``predict_all_pairs`` once with apply_temperature=False
    # (raw sigmoid) -- this was already efficient (single encode),
    # but it wrote the RAW sigmoid to ``gnn_score``, which the RL
    # reward function reads. Temperature calibration was dead for
    # Phase 6.
    #
    # The fix: use ``predict_all_pairs_dual`` (single encode pass)
    # and use the CALIBRATED matrix as the source of ``gnn_score``.
    # This aligns Phase 6 with the RL training distribution (which
    # now also uses calibrated gnn_score per P3-004 fix in
    # ``generate_rl_input``). Both paths now use the SAME calibrated
    # value -- no more distribution mismatch between training and
    # Phase 6 inference.
    raw_matrix, calibrated_matrix = model.predict_all_pairs_dual(
        node_features, edge_indices,
        num_drugs=num_drugs, num_diseases=num_diseases,
        exclude_edges=exclude_edges,
    )  # SINGLE encode pass; both matrices differ only in sigmoid transform

    # P3-004: use calibrated score as gnn_score (matches bridge fix).
    score_matrix = calibrated_matrix

    # Flatten and find top-K novel
    known_set = set((d.lower(), v.lower()) for d, v in known_pairs)
    flat_scores = score_matrix.cpu().numpy().flatten()
    flat_indices = np.argsort(-flat_scores)  # descending

    results: List[Tuple[str, str, float]] = []
    for flat_idx in flat_indices:
        d_idx = int(flat_idx // num_diseases)
        ds_idx = int(flat_idx % num_diseases)
        d_name = drug_names[d_idx]
        ds_name = disease_names[ds_idx]
        if (d_name.lower(), ds_name.lower()) in known_set:
            continue  # skip known positives
        results.append((d_name, ds_name, float(flat_scores[flat_idx])))
        if len(results) >= top_k:
            break

    return results
