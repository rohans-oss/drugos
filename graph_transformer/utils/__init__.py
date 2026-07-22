"""Utility helpers for the Graph Transformer."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for FULL reproducibility.

    P3-008 ROOT FIX (v113 forensic): the previous ``set_seed`` seeded
    Python's ``random``, NumPy, and PyTorch's CPU+CUDA RNGs, but did
    NOT set:
      - ``torch.backends.cudnn.deterministic = True``
      - ``torch.backends.cudnn.benchmark = False``
      - ``torch.use_deterministic_algorithms(True)``
      - ``os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"``
    On GPU, the same seed produced DIFFERENT model weights across runs
    (cuDNN's convolution algorithms are non-deterministic by default;
    cuDNN auto-tunes per input shape; some PyTorch ops like
    ``scatter_add_`` have non-deterministic CUDA implementations).
    The reproducibility claim ("Same seed => same split, every run")
    was only true on CPU. The V1 launch contract requires
    reproducibility for FDA 21 CFR Part 11 compliance -- partial
    reproducibility (CPU only) is insufficient for pharma production.

    ROOT FIX: set all four determinism flags. The
    ``CUBLAS_WORKSPACE_CONFIG`` env var MUST be set BEFORE
    ``import torch`` to take effect; we set it here as a best-effort
    (it may not take effect for the current process if torch is
    already imported, but it WILL take effect for child processes).
    The ``torch.use_deterministic_algorithms(True)`` call is wrapped
    in try/except because some ops (e.g., ``scatter_reduce_`` on
    older PyTorch) do not have deterministic implementations and
    will raise ``NotImplementedError`` -- we log a warning and
    continue without strict determinism for those ops.

    Fixes the B5 bug pattern: the original codebase mixed
    ``np.random.default_rng(seed)`` (local RNG) with
    ``torch.randperm(...)`` (global RNG) for the same split, so the
    "seeded" split was actually non-reproducible across runs. Calling
    ``set_seed`` before any splitting ensures torch's global RNG is
    also seeded.
    """
    import random
    # P3-008: CUBLAS_WORKSPACE_CONFIG must be set BEFORE torch is
    # imported for the first time. We set it here as a best-effort
    # (works for child processes; for the current process, torch is
    # already imported by the time set_seed is called, so the env
    # var may not take effect — but the cudnn flags below will).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # P3-008 ROOT FIX: enable full cuDNN + cuBLAS determinism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        # Some ops (e.g., scatter_reduce_ on older PyTorch) do not
        # have deterministic implementations and will raise
        # NotImplementedError. We use ``warn_only=True`` so those ops
        # fall back to non-deterministic with a warning, instead of
        # raising and breaking the training loop.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (TypeError, AttributeError):
        # Older PyTorch (<1.11) does not support ``warn_only``. Fall
        # back to strict mode (which may raise on some ops).
        try:
            torch.use_deterministic_algorithms(True)
        except (TypeError, AttributeError):
            # Even older PyTorch (<1.8) does not have this function.
            logger.warning(
                "P3-008: torch.use_deterministic_algorithms not "
                "available (PyTorch <1.8). Full reproducibility "
                "requires PyTorch >=1.11."
            )
    except RuntimeError as exc:
        # Some environments (CPU-only) may not support full
        # determinism. Log and continue.
        logger.warning(
            "P3-008: torch.use_deterministic_algorithms failed: %s. "
            "Reproducibility may be partial.", exc
        )


def drug_aware_split(
    drug_indices: torch.Tensor,
    disease_indices: torch.Tensor,
    labels: torch.Tensor,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    stratify_positives: bool = True,
    held_out_drugs: Optional[set] = None,
) -> Dict[str, torch.Tensor]:
    """Drug-aware train/val/test split (C4 fix).

    Splits by DRUG, not by pair. A drug that appears in train never
    appears in val or test. This prevents the model from memorizing
    drug-specific embedding features and trivially acing val AUC.

    ROOT FIX (v2): ``stratify_positives=True`` (default) distributes
    drugs that have at least one positive label across all three splits,
    so val and test each contain a proportional share of positives.

    V4 ROOT FIX (B-F6): ``held_out_drugs`` parameter. The Phase 4 RL
    ranker forces ALL ``KNOWN_POSITIVES`` pairs into its TEST set (so
    the recovery test can measure generalization to real therapeutic
    relationships). But the Phase 3 GT ``drug_aware_split`` (with
    ``stratify_positives=True``) distributes positive drugs across
    train+val+test -- so the GT model TRAINS on drugs like aspirin,
    then produces gnn_scores for aspirin->cardiovascular that the RL
    agent sees at test time. This is drug-level train/test leakage at
    the GT->RL boundary: the gnn_score for a KNOWN_POSITIVES pair is
    artificially high because the GT model trained on that drug's
    features (even with a different disease).

    The fix: the bridge passes ``held_out_drugs={drug names from
    KNOWN_POSITIVES}`` to ``drug_aware_split``. Those drugs are forced
    into val+test ONLY (never train), so the GT model never trains on
    their features. The gnn_score the RL agent sees for a
    KNOWN_POSITIVES pair is then a TRUE generalization measure, not
    an artifact of drug-level memorization. This unifies the GT and RL
    split strategies at the GT->RL boundary.

    V4 ROOT FIX (S-F5): the fallback (when the drug-aware split
    produces an empty split on tiny graphs) no longer falls back to a
    sequential pair-index split (which silently drops drug-awareness).
    Instead, it falls back to a drug-aware sequential split (sort drugs
    by index, take the first 70% for train, next 15% for val, rest for
    test). Drug-awareness is preserved even in the fallback path.

    Args:
        drug_indices: (N,) tensor of drug node indices.
        disease_indices: (N,) tensor of disease node indices.
        labels: (N,) tensor of binary labels.
        train_frac: Fraction of drugs in train.
        val_frac: Fraction of drugs in val.
        seed: Random seed (uses a torch.Generator for reproducibility).
        stratify_positives: If True (default), distribute drugs that have
            at least one positive label across train/val/test so each
            split has positives.
        held_out_drugs: Optional set of drug indices that MUST be in
            val or test (never train). Used by the bridge to exclude
            KNOWN_POSITIVES drugs from GT training so the gnn_score
            the RL agent sees is a true generalization measure.

    Returns:
        Dict with keys: train_drug_idx, train_disease_idx, train_labels,
        val_drug_idx, val_disease_idx, val_labels, test_drug_idx,
        test_disease_idx, test_labels.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac must be < 1.0, got {train_frac + val_frac}"
        )

    # Use a torch.Generator for reproducibility (B5 fix).
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    unique_drugs = torch.unique(drug_indices)
    n_drugs = len(unique_drugs)

    # V4 B-F6 fix: peel off held-out drugs first. These go to val/test
    # only, NEVER train. This is the GT->RL split-compatibility fix.
    held_out_set = set(int(d) for d in held_out_drugs) if held_out_drugs else set()
    if held_out_set:
        held_out_mask = torch.tensor(
            [int(d) in held_out_set for d in unique_drugs.tolist()],
            dtype=torch.bool,
        )
        held_out_drugs_tensor = unique_drugs[held_out_mask]
        remaining_drugs = unique_drugs[~held_out_mask]
        logger.info(
            f"V4 B-F6 fix: holding out {len(held_out_drugs_tensor)} drugs "
            f"(KNOWN_POSITIVES drugs) -> val/test only. "
            f"{len(remaining_drugs)} drugs available for train."
        )
    else:
        held_out_drugs_tensor = torch.tensor([], dtype=unique_drugs.dtype)
        remaining_drugs = unique_drugs

    # ROOT FIX (v2): stratify positives.
    # Find drugs that have at least one positive label.
    if stratify_positives:
        # Build a map drug_idx -> has_positive (only for remaining drugs)
        drug_has_positive: Dict[int, bool] = {}
        remaining_set = set(int(d) for d in remaining_drugs.tolist())
        for i in range(len(labels)):
            d = int(drug_indices[i].item())
            if d not in remaining_set:
                continue
            if labels[i].item() > 0.5:
                drug_has_positive[d] = True
            elif d not in drug_has_positive:
                drug_has_positive[d] = False

        positive_drug_set = {d for d, has_pos in drug_has_positive.items() if has_pos}
        negative_drug_set = {d for d, has_pos in drug_has_positive.items() if not has_pos}
        positive_drugs = torch.tensor(sorted(positive_drug_set), dtype=unique_drugs.dtype)
        negative_drugs = torch.tensor(sorted(negative_drug_set), dtype=unique_drugs.dtype)

        # Shuffle each independently with the same generator.
        if len(positive_drugs) > 0:
            pos_perm = positive_drugs[torch.randperm(len(positive_drugs), generator=gen)]
        else:
            pos_perm = positive_drugs
        if len(negative_drugs) > 0:
            neg_perm = negative_drugs[torch.randperm(len(negative_drugs), generator=gen)]
        else:
            neg_perm = negative_drugs

        # Distribute each across train/val/test according to fractions.
        n_pos = len(pos_perm)
        n_pos_train = int(train_frac * n_pos)
        n_pos_val = int(val_frac * n_pos)
        # Guarantee at least 1 positive drug in val and test when possible
        # (so AUC is computable on both).
        if n_pos >= 3:
            n_pos_val = max(n_pos_val, 1)
            n_pos_test = max(n_pos - n_pos_train - n_pos_val, 1)
            # Re-clamp train if val+test ate too much
            n_pos_train = max(0, n_pos - n_pos_val - n_pos_test)
        else:
            n_pos_test = max(0, n_pos - n_pos_train - n_pos_val)

        pos_train = pos_perm[:n_pos_train]
        pos_val = pos_perm[n_pos_train: n_pos_train + n_pos_val]
        pos_test = pos_perm[n_pos_train + n_pos_val: n_pos_train + n_pos_val + n_pos_test]

        n_neg = len(neg_perm)
        n_neg_train = int(train_frac * n_neg)
        n_neg_val = int(val_frac * n_neg)
        neg_train = neg_perm[:n_neg_train]
        neg_val = neg_perm[n_neg_train: n_neg_train + n_neg_val]
        neg_test = neg_perm[n_neg_train + n_neg_val:]

        train_drugs = torch.cat([pos_train, neg_train])
        # V4 B-F6 fix: held-out drugs go to val and test (split 50/50).
        if len(held_out_drugs_tensor) > 0:
            n_held = len(held_out_drugs_tensor)
            n_held_val = max(1, n_held // 2)
            held_val = held_out_drugs_tensor[:n_held_val]
            held_test = held_out_drugs_tensor[n_held_val:]
            val_drugs = torch.cat([pos_val, neg_val, held_val])
            test_drugs = torch.cat([pos_test, neg_test, held_test])
        else:
            val_drugs = torch.cat([pos_val, neg_val])
            test_drugs = torch.cat([pos_test, neg_test])
    else:
        perm = torch.randperm(n_drugs, generator=gen)
        n_train = int(train_frac * n_drugs)
        n_val = int(val_frac * n_drugs)
        train_drugs = unique_drugs[perm[:n_train]]
        val_drugs = unique_drugs[perm[n_train: n_train + n_val]]
        test_drugs = unique_drugs[perm[n_train + n_val:]]  # C5 fix: actually create test set
        # V4 B-F6 fix: ensure held-out drugs are not in train.
        # ROOT FIX (FORENSIC-AUDIT-I07): guard against empty train_drugs
        # after held-out filtering. If ALL train drugs were held-out, the
        # previous code produced an empty train_drugs tensor, which made
        # torch.isin return all-False, which triggered the fallback, which
        # ALSO filtered held-out drugs -- potentially infinite confusion.
        # The fix: after filtering, if train_drugs is empty, pull drugs
        # from val_drugs (which has non-held-out drugs) to populate train.
        #
        # V90 ROOT FIX (BUG #14, P1): the previous fallback moved drugs
        # from val_drugs to train_drugs WITHOUT checking whether those
        # val drugs were themselves held-out. On a tiny demo graph
        # where the fallback triggers AND val_drugs happens to contain
        # held-out drugs (e.g., after the stratified split put KPs in
        # val), KP drugs ended up in train. This violated the C-3
        # fix's guarantee that KPs never appear in training. The fix:
        # filter held-out drugs from val_drugs BEFORE moving them to
        # train_drugs. If val_drugs becomes empty after filtering, we
        # raise (the graph is too small to satisfy the split contract).
        if held_out_set:
            train_drugs = train_drugs[~torch.tensor(
                [int(d) in held_out_set for d in train_drugs.tolist()], dtype=torch.bool
            )]
            # V90 BUG #14: filter held-out drugs from val_drugs BEFORE
            # moving them to train_drugs. The previous code moved
            # val_drugs[:n_move] to train WITHOUT checking if those
            # drugs were held-out (KP drugs), leaking KPs into train.
            val_drugs = val_drugs[~torch.tensor(
                [int(d) in held_out_set for d in val_drugs.tolist()], dtype=torch.bool
            )]
            # ROOT FIX (FORENSIC-AUDIT-I07): if train_drugs is now empty,
            # move drugs from val_drugs (now filtered of held-out) to
            # train_drugs.
            if len(train_drugs) == 0 and len(val_drugs) > 0:
                n_move = max(1, len(val_drugs) // 2)
                train_drugs = val_drugs[:n_move]
                val_drugs = val_drugs[n_move:]
                logger.warning(
                    f"FORENSIC-AUDIT-I07: train_drugs was empty after "
                    f"held-out filtering. Moved {n_move} drugs from val "
                    f"to train to prevent degenerate split. "
                    f"(V90 BUG #14: val_drugs was filtered of held-out "
                    f"drugs BEFORE moving, so no KP leakage.)"
                )
            elif len(train_drugs) == 0 and len(val_drugs) == 0:
                raise RuntimeError(
                    f"V90 ROOT FIX (BUG #14): after filtering held-out "
                    f"drugs, BOTH train_drugs and val_drugs are empty. "
                    f"The graph is too small ({n_drugs} drugs, "
                    f"{len(held_out_set)} held-out) to satisfy the "
                    f"drug-aware split. Either increase the graph size "
                    f"or reduce the held-out set."
                )
            # Re-add held-out drugs to val/test if not already there
            existing_val = set(int(d) for d in val_drugs.tolist())
            existing_test = set(int(d) for d in test_drugs.tolist())
            missing = [d for d in held_out_set if d not in existing_val and d not in existing_test]
            if missing:
                missing_tensor = torch.tensor(sorted(missing), dtype=unique_drugs.dtype)
                half = max(1, len(missing) // 2)
                val_drugs = torch.cat([val_drugs, missing_tensor[:half]])
                test_drugs = torch.cat([test_drugs, missing_tensor[half:]])

    # Bucket pairs by their drug's split assignment
    train_mask = torch.isin(drug_indices, train_drugs)
    val_mask = torch.isin(drug_indices, val_drugs)
    test_mask = torch.isin(drug_indices, test_drugs)

    # V4 S-F5 fix: fallback preserves drug-awareness. The original
    # code fell back to a sequential pair-index split, which silently
    # dropped the drug-aware guarantee exactly when it's hardest to
    # satisfy (tiny graphs). The new fallback sorts drugs by index and
    # takes the first 70% for train, next 15% for val, rest for test --
    # still drug-aware, just deterministic instead of random.
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        logger.warning(
            f"Drug-aware split produced empty split "
            f"(train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}); "
            f"falling back to DETERMINISTIC drug-aware sequential split "
            f"(V4 S-F5 fix: no longer drops drug-awareness)."
        )
        # Sort drugs by index, take slices. Still drug-aware.
        sorted_drugs = torch.sort(unique_drugs).values
        n_total_drugs = len(sorted_drugs)
        n_train_d = max(1, int(0.7 * n_total_drugs))
        n_val_d = max(1, int(0.15 * n_total_drugs))
        train_drugs = sorted_drugs[:n_train_d]
        val_drugs = sorted_drugs[n_train_d: n_train_d + n_val_d]
        test_drugs = sorted_drugs[n_train_d + n_val_d:]
        # V4 B-F6 fix: ensure held-out drugs are not in train.
        # ROOT FIX (FORENSIC-AUDIT-I07): same empty-train guard as above.
        # V90 ROOT FIX (BUG #14, P1): same val_drugs filtering fix as
        # above -- filter held-out drugs from val_drugs BEFORE moving
        # them to train_drugs to prevent KP leakage into train.
        if held_out_set:
            train_drugs = train_drugs[~torch.tensor(
                [int(d) in held_out_set for d in train_drugs.tolist()], dtype=torch.bool
            )]
            # V90 BUG #14: filter held-out drugs from val_drugs BEFORE
            # moving them to train_drugs (fallback path).
            val_drugs = val_drugs[~torch.tensor(
                [int(d) in held_out_set for d in val_drugs.tolist()], dtype=torch.bool
            )]
            # ROOT FIX (FORENSIC-AUDIT-I07): if train_drugs is empty after
            # filtering, move drugs from val to train.
            if len(train_drugs) == 0 and len(val_drugs) > 0:
                n_move = max(1, len(val_drugs) // 2)
                train_drugs = val_drugs[:n_move]
                val_drugs = val_drugs[n_move:]
                logger.warning(
                    f"FORENSIC-AUDIT-I07 (fallback): train_drugs was empty "
                    f"after held-out filtering. Moved {n_move} drugs from "
                    f"val to train. (V90 BUG #14: val_drugs was filtered "
                    f"of held-out drugs BEFORE moving, so no KP leakage.)"
                )
            elif len(train_drugs) == 0 and len(val_drugs) == 0:
                raise RuntimeError(
                    f"V90 ROOT FIX (BUG #14, fallback): after filtering "
                    f"held-out drugs, BOTH train_drugs and val_drugs are "
                    f"empty. The graph is too small ({n_drugs} drugs, "
                    f"{len(held_out_set)} held-out) to satisfy the "
                    f"drug-aware split."
                )
            existing_val = set(int(d) for d in val_drugs.tolist())
            existing_test = set(int(d) for d in test_drugs.tolist())
            missing = [d for d in held_out_set if d not in existing_val and d not in existing_test]
            if missing:
                missing_tensor = torch.tensor(sorted(missing), dtype=unique_drugs.dtype)
                half = max(1, len(missing) // 2)
                val_drugs = torch.cat([val_drugs, missing_tensor[:half]])
                test_drugs = torch.cat([test_drugs, missing_tensor[half:]])
        train_mask = torch.isin(drug_indices, train_drugs)
        val_mask = torch.isin(drug_indices, val_drugs)
        test_mask = torch.isin(drug_indices, test_drugs)

    return {
        "train_drug_idx": drug_indices[train_mask],
        "train_disease_idx": disease_indices[train_mask],
        "train_labels": labels[train_mask],
        "val_drug_idx": drug_indices[val_mask],
        "val_disease_idx": disease_indices[val_mask],
        "val_labels": labels[val_mask],
        "test_drug_idx": drug_indices[test_mask],
        "test_disease_idx": disease_indices[test_mask],
        "test_labels": labels[test_mask],
    }


def compute_graph_degrees(
    edge_indices: Dict[Tuple[str, str, str], torch.Tensor],
    node_type: str,
    direction: str = "out",
) -> Dict[int, int]:
    """Compute per-node degree for a given node type.

    P3-007 ROOT FIX (v113 forensic): VECTORIZED dict construction.
    The previous code converted the bincount tensor to a Python dict
    via a ``for idx in range(len(counts))`` loop with ``.item()`` calls.
    For a graph with 10M nodes, this was 10M Python iterations + 10M
    Python-to-C transitions (each ``.item()`` is a transition). At
    ~1µs per iteration, that's ~10 seconds of pure Python overhead
    PER CALL. The bridge calls this function 4 times per
    ``generate_rl_input`` invocation, so ~40 seconds of overhead.

    ROOT FIX: use ``np.where`` + ``counts[counts > 0]`` to build the
    dict via vectorized numpy operations. The Python loop is replaced
    by a single numpy operation that runs in C. The dict is then
    constructed via ``dict(zip(...))`` which is a single C-level
    iteration over a (num_nonzero,) array (typically << num_nodes).

    ROOT FIX (B1): this function is now ACTUALLY CALLED by the bridge's
    ``save_rl_input_streaming`` and ``_compute_supplementary_features``
    methods to compute REAL supplementary features (safety, market,
    unmet_need) from graph topology. Previously it was dead code --
    defined but never invoked. The V4 "dead code fix #2, #7" removed
    the CALL but left the DEFINITION, creating a false claim of cleanup.
    The V8 root fix RE-WIRES the function into 4 active call sites:
      1. AE count per drug (safety score)
      2. Pathway count per disease (market score) -- streaming path
      3. Treat count per disease (unmet_need score) -- streaming path
      4. AE count, pathway count, treat count -- non-streaming path

    ROOT FIX (FORENSIC-AUDIT-I08): vectorized with ``torch.bincount``
    instead of a Python-level loop. The previous code iterated over
    every edge index with ``indices.tolist()`` and ``degrees.get(idx, 0) + 1``
    -- a Python loop that is ~100× slower than ``torch.bincount`` for
    large edge counts. The bridge calls this function 4 times per
    ``generate_rl_input`` invocation, so the speedup compounds.

    V90 ROOT FIX (BUG #50): the function still returns a Dict[int, int]
    for backward compatibility with existing callers that use
    ``degrees.get(d_idx, 0)``. However, a NEW function
    ``compute_graph_degrees_array`` is added that returns the counts as
    a numpy array for vectorized use. Callers that need vectorized
    access (e.g., the bridge's _compute_supplementary_features) should
    migrate to the new function. The dict-returning version is kept for
    backward compat.

    Args:
        edge_indices: Dict mapping (src, rel, tgt) to (2, E) tensor.
        node_type: Node type to compute degrees for.
        direction: 'out' (outgoing edges), 'in' (incoming edges), or
            'both' (sum).

    Returns:
        Dict mapping node index -> degree.
    """
    # Collect all relevant indices first, then use torch.bincount for
    # vectorized counting.
    all_indices: List[torch.Tensor] = []

    for (src, rel, tgt), ei in edge_indices.items():
        if ei.numel() == 0:
            continue
        if direction in ("out", "both") and src == node_type:
            all_indices.append(ei[0])
        if direction in ("in", "both") and tgt == node_type:
            all_indices.append(ei[1])

    if not all_indices:
        return {}

    # Concatenate all indices and use bincount for vectorized counting.
    # bincount is ~100× faster than a Python loop for large tensors.
    concatenated = torch.cat(all_indices)
    if concatenated.numel() == 0:
        return {}
    # bincount returns a tensor of length (max_index + 1) with counts.
    counts = torch.bincount(concatenated)
    # P3-007 ROOT FIX: VECTORIZED dict construction. The previous code
    # used a Python for-loop over ``range(len(counts))`` with ``.item()``
    # calls (~10s for 10M nodes). The fix uses ``np.where`` to find
    # non-zero indices in a single C-level operation, then constructs
    # the dict via ``dict(zip(...))`` (also C-level). For 10M nodes
    # with ~100K non-zero entries, this is ~100K iterations vs 10M --
    # a 100x speedup.
    counts_np = counts.cpu().numpy()
    nonzero_mask = counts_np > 0
    nonzero_indices = np.where(nonzero_mask)[0]
    nonzero_values = counts_np[nonzero_indices]
    degrees = dict(zip(nonzero_indices.tolist(), nonzero_values.tolist()))
    return degrees


def compute_graph_degrees_array(
    edge_indices: Dict[Tuple[str, str, str], torch.Tensor],
    node_type: str,
    direction: str = "out",
    num_nodes: Optional[int] = None,
) -> np.ndarray:
    """Compute per-node degree as a numpy array (vectorized).

    V90 ROOT FIX (BUG #50): the bridge's ``_compute_supplementary_features``
    uses ``compute_graph_degrees`` (which returns a Dict[int, int]) in a
    Python-level loop: ``ae_count_per_drug.get(d_idx, 0)``. For 10K drugs,
    this is 10K dict lookups. The audit's BUG #50 finding: "Could be
    vectorized by returning the counts tensor directly."

    This new function returns the counts as a numpy array of length
    ``num_nodes`` (or ``max_index + 1`` if num_nodes is None). Callers
    can then use vectorized numpy indexing (``ae_counts[drug_indices]``)
    instead of Python-level dict lookups.

    Args:
        edge_indices: Dict mapping (src, rel, tgt) to (2, E) tensor.
        node_type: Node type to compute degrees for.
        direction: 'out' (outgoing edges), 'in' (incoming edges), or
            'both' (sum).
        num_nodes: Total number of nodes of this type. If None, uses
            ``max_index + 1``. Pass the actual count for a correctly-
            sized array (avoids IndexError on lookups for high-index
            nodes that have zero edges).

    Returns:
        numpy array of length num_nodes (or max_index + 1) with per-node
        degree counts. Nodes with no edges have count 0.
    """
    all_indices: List[torch.Tensor] = []

    for (src, rel, tgt), ei in edge_indices.items():
        if ei.numel() == 0:
            continue
        if direction in ("out", "both") and src == node_type:
            all_indices.append(ei[0])
        if direction in ("in", "both") and tgt == node_type:
            all_indices.append(ei[1])

    if not all_indices:
        size = num_nodes if num_nodes is not None else 0
        return np.zeros(size, dtype=np.int64)

    concatenated = torch.cat(all_indices)
    if concatenated.numel() == 0:
        size = num_nodes if num_nodes is not None else 0
        return np.zeros(size, dtype=np.int64)

    counts = torch.bincount(concatenated)
    counts_np = counts.numpy().astype(np.int64)

    if num_nodes is not None and len(counts_np) < num_nodes:
        # Pad with zeros to reach num_nodes length.
        padded = np.zeros(num_nodes, dtype=np.int64)
        padded[:len(counts_np)] = counts_np
        return padded
    return counts_np


def save_dict_to_json(data: Dict[str, Any], path: str) -> None:
    """Save a dict to a JSON file, creating parent directories."""
    import json

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON to {path}")
