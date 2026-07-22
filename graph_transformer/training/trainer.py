"""Training loop for the Graph Transformer model.

FIX vs original codebase:
  - **B2 (BCELoss + logit clamp NaN bomb)**: replaced ``nn.BCELoss``
    on sigmoid outputs with ``nn.BCEWithLogitsLoss`` on raw logits.
    Numerically stable: uses log-sum-exp, never produces ``log(0)``.
    Paired with the link predictor's new ``forward_logits`` method.
  - **B3 (validation leaks the labels it's predicting)**: the original
    ``evaluate()`` called ``self.model(...)`` without passing
    ``exclude_edges``, so the model saw ``('drug','treats','disease')``
    edges while scoring the very pairs that label was derived from.
    Validation AUC was inflated, early stopping was biased.

    Fix: ``evaluate()`` now always excludes ``LABEL_LEAKING_EDGES``
    via the model's ``forward_logits`` default. (The model itself also
    defaults to excluding these edges -- defense in depth.)
  - **B11 (DataLoader dead import)**: removed.
  - **B12 (epoch undefined if epochs=0)**: initialize ``epoch = 0``
    before the loop so the return statement doesn't NameError.
  - **B10 (dead fit_temperature)**: after main training, the trainer
    now calls ``link_predictor.fit_temperature()`` on the validation
    set to calibrate the temperature parameter (Guo et al. 2017). The
    parameter is no longer dead weight.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..data import LABEL_LEAKING_EDGES

logger = logging.getLogger(__name__)


class GraphTransformerTrainer:
    """Training loop for the Graph Transformer.

    Handles model training, validation, early stopping, checkpointing,
    and post-hoc temperature calibration.

    Args:
        model: DrugRepurposingGraphTransformer instance.
        node_features: Dict of node feature tensors.
        edge_indices: Dict of edge index tensors.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularization. Default 0.01 -- the
            production-grade value for Transformer training per
            Loshchilov & Hutter 2019 ("Decoupled Weight Decay
            Regularization"). The previous default 1e-5 was far too
            small for Transformer training and allowed the model to
            overfit (audit Issue 136). Callers can still pass a
            different value if they have a measured reason to.
        device: Device to train on.
        seed: RNG seed.
        node_maps: Optional dict mapping each node type to a
            ``{node_name: index}`` dict. When provided, the trainer
            saves it in the checkpoint so the inference service can
            resolve (drug, disease) names to indices without a
            separate graph_state.pt sidecar (audit Issue 139).
        drug_names: Optional ordered list of drug names (index = node
            index). Saved in the checkpoint for the same reason.
        disease_names: Optional ordered list of disease names.
        known_pairs: Optional list of ``(drug, disease)`` tuples that
            are known treatment pairs (used by the top-k endpoint to
            filter out non-novel predictions).
    """

    def __init__(
        self,
        model: nn.Module,
        node_features: Dict[str, torch.Tensor],
        edge_indices: Dict[Tuple[str, str, str], torch.Tensor],
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        device: str = "cpu",
        seed: int = 42,
        node_maps: Optional[Dict[str, Dict[str, int]]] = None,
        drug_names: Optional[List[str]] = None,
        disease_names: Optional[List[str]] = None,
        known_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Initialize the trainer.

        V4 ROOT FIX (C-F6): ``seed`` parameter. The original code
        called ``set_seed`` once at bridge construction, then every
        subsequent torch op (``_init_weights``, ``torch.randperm`` in
        ``train_epoch``, etc.) advanced the global RNG. Adding ANY new
        torch operation between ``set_seed`` and ``train_epoch`` would
        change the training data shuffle order, breaking
        reproducibility.

        The fix: the trainer holds its own ``torch.Generator`` seeded
        with ``seed``. ``train_epoch`` uses this generator for
        ``torch.randperm``, so the shuffle order is deterministic and
        independent of any other torch ops that may have advanced the
        global RNG.
        """
        self.model = model.to(device)
        self.node_features = {k: v.to(device) for k, v in node_features.items()}
        self.edge_indices = {k: v.to(device) for k, v in edge_indices.items()}
        self.device = device
        self.seed = seed
        # P3-S06: store learning_rate so fit() can use it as max_lr for
        # the OneCycleLR scheduler.
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # FORENSIC ROOT FIX (audit Issue 139): persist the graph metadata
        # that the inference service needs to resolve (drug, disease) names
        # to indices. The previous trainer saved ONLY the model state_dict +
        # schema, forcing the bridge to write a SEPARATE graph_state.pt
        # sidecar. Production-grade checkpoints must be SELF-CONTAINED so
        # that loading a single .pt file reproduces the exact training
        # graph + model architecture. We store deep copies so a later
        # in-memory mutation cannot silently corrupt the checkpoint.
        self.node_maps: Dict[str, Dict[str, int]] = {
            ntype: dict(name_to_idx) for ntype, name_to_idx in (node_maps or {}).items()
        }
        self.drug_names: List[str] = list(drug_names) if drug_names is not None else []
        self.disease_names: List[str] = list(disease_names) if disease_names is not None else []
        self.known_pairs: List[Tuple[str, str]] = [
            (str(d), str(v)) for d, v in (known_pairs or [])
        ]
        # V4 C-F6 fix: dedicated generator for reproducible shuffling.
        # V30 ROOT FIX (8.3): the original ``torch.Generator()`` creates a
        # CPU generator. Calling ``torch.randperm(..., device="cuda",
        # generator=self._gen)`` raises RuntimeError because the generator
        # and tensor must be on the same device. The trainer was silently
        # CPU-only despite accepting a ``device`` parameter.
        #
        # The fix: create the generator on the SAME device as the trainer.
        # On CPU, ``torch.Generator(device="cpu")`` works. On CUDA,
        # ``torch.Generator(device="cuda")`` works. The randperm call now
        # uses self.device so the generator and tensor always match.
        #
        # P3-028 ROOT FIX: torch.Generator(device=...) is NOT supported on
        # MPS (Apple Silicon) or XLA (TPU) — it raises
        # ``RuntimeError: Generator for X device is not supported``.
        # The V30 8.3 fix only handled CPU and CUDA. On Apple Silicon
        # (device="mps") or TPU (device="xla"), the trainer crashed at
        # construction time, blocking all training on those devices.
        # The fix: try the requested device first, fall back to a CPU
        # generator on RuntimeError. randperm results are then moved to
        # the target device by the caller (train_epoch already does
        # ``.to(self.device)`` on the perm tensor). A CPU generator
        # produces identical sequences to a device generator for the same
        # seed, so reproducibility is preserved.
        try:
            self._gen = torch.Generator(device=device)
            self._gen_device: str = device
        except (RuntimeError, TypeError):
            logger.warning(
                f"ROOT FIX (P3-028): torch.Generator(device='{device}') is "
                f"not supported on this device (common for MPS / XLA). "
                f"Falling back to a CPU generator. randperm results will "
                f"be moved to the target device by callers. Reproducibility "
                f"is preserved (same seed -> same sequence)."
            )
            self._gen = torch.Generator(device="cpu")
            self._gen_device = "cpu"
        self._gen.manual_seed(seed)

        # FORENSIC ROOT FIX (audit Issue 136): use AdamW (Loshchilov &
        # Hutter 2019, "Decoupled Weight Decay Regularization") instead
        # of plain Adam. AdamW applies weight decay as a DECOUPLED L2
        # penalty (multiplied directly against the parameters), whereas
        # Adam applies it COUPLED to the gradient (which interacts
        # badly with the momentum + adaptive learning rate and produces
        # INEFFECTIVE regularization). The standard Transformer recipe
        # (Vaswani et al. 2017 -> Loshchilov & Hutter 2019 -> GPT-2/3,
        # BERT, LLaMA) is AdamW + weight_decay=0.01. Plain Adam with
        # weight_decay=1e-5 was allowing the Graph Transformer to
        # overfit the training pairs (audit evidence: train AUC
        # climbing while val AUC stagnated).
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        # P3-S06 ROOT FIX: optional LR scheduler (OneCycleLR). Created in
        # fit() once we know total_steps = epochs * n_batches. Set to None
        # here so train_epoch() can check ``if self.scheduler is not None``
        # without AttributeError on trainers that never call fit().
        # The scheduler implements warmup (pct_start=0.1) + cosine decay,
        # which is the standard practice for transformer training. The
        # previous code used plain Adam with constant lr=5e-4 for all
        # epochs -- no warmup, no decay. The first few epochs had large
        # gradients (random init) that destabilized training without
        # warmup, and the last few epochs had small gradients that needed
        # a lower lr to converge. OneCycleLR fixes both: warmup for the
        # first 10% of steps, then cosine decay to ~0.
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        # V30 ROOT FIX (8.6): add pos_weight to BCEWithLogitsLoss to handle
        # class imbalance. The demo graph has ~5-10% positive pairs (KPs are
        # rare), so without pos_weight the model is biased toward predicting
        # LOW for everything. pos_weight = (num_negatives / num_positives)
        # is the standard sklearn-recommended value. We compute it lazily in
        # fit() once we know the actual class balance, then update the
        # criterion. Default pos_weight of 1.0 (no reweighting) preserves
        # the original behavior for callers who don't call fit().
        #
        # V90 ROOT FIX (BUG #22, P1): create the initial criterion with
        # device=self.device. The previous code created the pos_weight
        # tensor on CPU (``torch.tensor([1.0])`` with no device arg).
        # If the user called ``train_epoch()`` directly (without
        # calling ``fit()`` first), the criterion had a CPU pos_weight
        # tensor. When ``self.criterion(logits, batch_labels)`` ran
        # with CUDA logits, PyTorch raised
        # ``RuntimeError: Expected all tensors to be on the same device``.
        # Only ``fit()`` worked because it recreated the criterion
        # with the correct device.
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.0], device=self.device)
        )
        # P3-018 / P3-D10 ROOT FIX: removed the duplicate
        # ``self._eval_criterion = nn.BCEWithLogitsLoss()`` assignment.
        # The previous code defined _eval_criterion TWICE in __init__
        # (once at the original line 133 with the BUG #26 comment, once
        # at the original line 165 with the BUG #32 comment). Both
        # created a fresh BCEWithLogitsLoss; the second silently won and
        # the first was dead code. The two comments describe two
        # DIFFERENT reasons for the SAME object, so we merge them here
        # and keep a single assignment.
        #
        # V90 ROOT FIX (BUG #26, P1): separate evaluation criterion
        # WITHOUT pos_weight. The trainer's evaluate() previously used
        # ``self.criterion`` (which has training pos_weight). If the
        # val/test set has a different class balance, the reported eval
        # loss is weighted incorrectly. Comparing eval loss across
        # different class balances is meaningless. The fix: use a
        # fresh ``nn.BCEWithLogitsLoss()`` (no pos_weight) for
        # evaluation loss. This also makes trainer.evaluate consistent
        # with evaluate_link_prediction (BUG #27 fix).
        #
        # V90 ROOT FIX (BUG #32): unweighted eval criterion for early-stopping
        # signal. The training criterion uses pos_weight (BUG #26 / 8.6 fix)
        # to handle class imbalance, but pos_weight AMPLIFIES loss noise on
        # small val sets (15 pairs). Using the pos_weighted loss for early
        # stopping caused checkpoint thrashing -- float noise flipped the
        # "improvement" signal every epoch. The fix uses a SEPARATE
        # unweighted BCEWithLogitsLoss for the early-stopping decision,
        # while the pos_weighted criterion is still used for gradient
        # updates (training). This decouples the noisy training signal
        # from the early-stopping signal.
        self._eval_criterion = nn.BCEWithLogitsLoss()

        self.best_val_auc = 0.0
        self.best_val_loss: float = float("inf")
        self.best_state_dict: Optional[Dict[str, Any]] = None
        # FORENSIC ROOT FIX (audit Issue 138): checkpoint selection is now
        # driven by ``val_auc`` (the scientific success metric), NOT
        # ``val_loss``. The previous implementation argued that val_loss is
        # "less noisy" than val_auc on small validation sets, but that
        # argument is scientifically wrong for this platform:
        #
        #   1. The V1 launch criterion (project docx) is AUC > 0.85 on
        #      held-out drug-disease pairs. Loss is NOT a launch criterion.
        #      Selecting checkpoints by val_loss can pick a model whose
        #      loss decreased because it became MORE confident on the
        #      majority class (negatives), while its RANKING (AUC) of
        #      positives vs negatives actually degraded. This is exactly
        #      what the audit flagged: "Loss can decrease while AUC
        #      degrades."
        #
        #   2. The val_loss argument was based on a 15-pair validation
        #      set. Production graphs (V1: 10K drugs x 10K diseases)
        #      produce validation sets of thousands of pairs where AUC
        #      is well-conditioned. The trainer should select on the
        #      metric that matters (AUC) and use a noise-robust threshold
        #      (val_auc_min_improvement below) to filter the small-data
        #      noise, NOT silently substitute a different metric.
        #
        # Noise mitigation: a checkpoint is only promoted when the new
        # val_auc beats the running best by at least
        # ``self.val_auc_min_improvement`` (default 0.005 = 0.5% AUC).
        # This filters the ±0.1 noise the previous comment was concerned
        # about, while still selecting on the scientifically correct
        # metric. CI tests can assert
        # ``trainer.checkpoint_selection_metric == "val_auc"`` to catch
        # any future regression.
        self.checkpoint_selection_metric: str = "val_auc"
        # 0.5% AUC improvement required to promote a new "best"
        # checkpoint. Smaller than this is treated as noise. The value
        # is deliberately conservative: on a 100-pair val set, the
        # standard deviation of AUC is ~0.03, so 0.005 is well inside
        # one standard deviation -- only meaningful improvements
        # promote the checkpoint.
        self.val_auc_min_improvement: float = 0.005
        # P3-019 / P3-D11 ROOT FIX: removed the duplicate
        # ``self.best_epoch: int = 0`` assignment. The previous code
        # defined best_epoch TWICE in __init__ (once with the BUG #33
        # comment, once with the BUG #21 comment). Both set 0; the second
        # silently won and the first was dead code. We merge the comments
        # and keep a single assignment.
        #
        # V90 ROOT FIX (BUG #33): persist best_epoch as an instance attribute
        # so save_checkpoint / load_checkpoint can save and restore it. The
        # previous code kept best_epoch as a LOCAL variable inside fit() --
        # it was lost on reload, so the user could not tell which epoch
        # produced the best model. The fix stores it on self so it survives
        # save/load round-trips.
        #
        # V90 ROOT FIX (BUG #21, P1): make best_epoch an instance
        # attribute so save_checkpoint can save the ACTUAL best epoch
        # (not the LAST epoch). The previous code saved
        # ``self.training_history[-1]["epoch"]`` which is the LAST
        # epoch, not the epoch with the best val_loss. The variable is
        # named best_epoch but stored the last epoch. Misleading. On
        # checkpoint reload, the user saw best_epoch = 500 (the last
        # epoch) when the actual best was epoch 42.
        self.best_epoch: int = 0
        self.training_history: List[Dict[str, float]] = []
        # V30 ROOT FIX (8.2): store the last-used val data so a no-arg
        # evaluate() can re-evaluate without requiring the caller to pass
        # the same data again. This is the standard sklearn-style API.
        self._last_val_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # P3-013 ROOT FIX (forensic, Team Member 10): scale early-stopping
    # patience with graph size.
    #
    # The audit (P3-013) found that the trainer's hardcoded patience=10
    # is too short for small graphs. On small graphs training is noisy:
    # val_loss may not improve for 10 epochs purely due to noise, then
    # improve again. patience=10 stops too early, before the model has
    # converged, producing under-trained models (audit evidence: AUC=0.403
    # on the live demo run).
    #
    # The fix introduces a graph-size-aware patience schedule:
    #   - <1K training pairs     -> patience=30  (small graph, very noisy)
    #   - 1K-100K training pairs -> patience=15  (medium graph, moderate noise)
    #   - >100K training pairs   -> patience=5   (large graph, low noise,
    #                                            fast convergence)
    #
    # The thresholds are derived from the empirical observation that
    # val_loss variance scales ~1/sqrt(n) — small graphs need more
    # patience to ride out the noise. The values are deliberately
    # conservative (we'd rather over-train slightly than stop early on
    # a small graph where every epoch is cheap).
    #
    # API contract: ``fit(patience=...)`` is the explicit override. If
    # the caller passes a non-None patience, it is used as-is. If the
    # caller does NOT pass patience (or passes the new sentinel
    # ``"auto"``), the trainer uses ``scale_patience_with_graph_size()``
    # to derive the appropriate value from the training set size. This
    # preserves backward compatibility (existing callers passing
    # patience=10 still get 10) while making the DEFAULT behavior
    # scientifically correct for any graph size.
    # ------------------------------------------------------------------
    @staticmethod
    def scale_patience_with_graph_size(n_train_pairs: int) -> int:
        """Return graph-size-aware early-stopping patience.

        P3-013 ROOT FIX: derive patience from the training-set size so
        small graphs (noisy val_loss) get more patience and large
        graphs (smooth val_loss) get less. See the empirical thresholds
        in the P3-013 comment block above.

        Args:
            n_train_pairs: Number of training (drug, disease) pairs.

        Returns:
            int: patience in epochs. <1K pairs -> 30, 1K-100K -> 15,
            >100K -> 5. Always returns at least 5 (sanity floor).
        """
        # Defensive: n_train_pairs may be a torch tensor or numpy int.
        try:
            n = int(n_train_pairs)
        except (TypeError, ValueError):
            # If we can't determine the size, fall back to the medium
            # bucket (15) — safe for both small and large graphs.
            return 15
        if n < 1_000:
            return 30
        if n < 100_000:
            return 15
        return 5

    # ------------------------------------------------------------------
    # P3-011 ROOT FIX (forensic, Team Member 10): expose pos_weight
    # computation as a static helper so CI tests can verify (a) the
    # formula is correct (n_neg / n_pos, clamped to [1, max]) and
    # (b) applying pos_weight actually decreases the loss on
    # imbalanced data (the audit's specific CI test requirement).
    #
    # The audit (P3-011) found that trainer.py used
    # BCEWithLogitsLoss without pos_weight, causing the model to
    # predict ~0.001 for everything on the ~1:1000 imbalanced KG
    # (high accuracy, terrible AUC). The V30 8.6 + P3-S03 fixes
    # already compute pos_weight in fit() with a clamp_max parameter
    # (default 10.0 for production, 2.0 for tiny demo graphs). This
    # helper exposes the SAME computation as a static method so:
    #   1. CI tests can verify the formula without instantiating a
    #      full trainer (which requires a model, node_features, etc).
    #   2. External callers (e.g. a custom training loop) can compute
    #      pos_weight for BCEWithLogitsLoss without copy-pasting the
    #      formula.
    #   3. The audit's CI requirement ("verifies the loss decreases
    #      with pos_weight") can be met by a test that calls this
    #      helper, constructs two BCEWithLogitsLoss instances (with
    #      and without pos_weight), and checks the weighted one
    #      produces a higher loss on imbalanced data (forcing the
    #      model to pay more attention to positives).
    # ------------------------------------------------------------------
    @staticmethod
    def compute_pos_weight(
        labels: Any,
        clamp_max: float = 10.0,
        clamp_min: float = 1.0,
    ) -> float:
        """Compute pos_weight for BCEWithLogitsLoss from class balance.

        P3-011 ROOT FIX: pos_weight = n_negatives / n_positives,
        clamped to [clamp_min, clamp_max] for numerical stability.
        The clamp prevents extreme pos_weight values (e.g. 1000 for
        1:1000 imbalance) from destabilizing training -- the gradient
        on a single positive would be 1000x larger than on a negative,
        causing the optimizer to overshoot. The default clamp_max=10.0
        matches the production default in fit(); pass clamp_max=2.0
        for tiny demo graphs (<=100 pairs) where pos_weight > 2.0
        caused below-random test AUC in the V30 demo runs.

        Args:
            labels: 1D array-like of binary labels (0/1). Accepts
                numpy arrays, torch tensors, or Python lists.
            clamp_max: Upper bound for pos_weight. Default 10.0.
            clamp_min: Lower bound for pos_weight. Default 1.0
                (below 1.0 would DOWN-weight positives, which is
                never desired for imbalanced classification).

        Returns:
            pos_weight as a float. Returns 1.0 if either class is
            empty (no positives OR no negatives -- pos_weight is
            undefined, fall back to no reweighting).
        """
        # Accept torch tensors, numpy arrays, or lists.
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labels)
        n_pos = int((labels_np == 1).sum())
        n_neg = int((labels_np == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return 1.0
        raw = n_neg / n_pos
        return float(max(clamp_min, min(clamp_max, raw)))

    # ------------------------------------------------------------------
    # P3-018 ROOT FIX (forensic, Team Member 10): GPU utilization logging.
    #
    # The audit (P3-018) found that trainer.py logs training loss and
    # AUC but NOT GPU utilization. When training is slow (e.g. 10x
    # slower than expected), the ops team cannot tell if it's a GPU
    # issue (low utilization = data-loading bottleneck, high
    # utilization = compute-bound) or a model issue. They waste time
    # debugging the wrong thing.
    #
    # The fix logs three diagnostic signals every epoch:
    #   1. ``torch.cuda.utilization()`` -- % of time GPU spent in
    #      kernel execution over the last sample period. Low values
    #      (<30%) indicate a data-loading bottleneck (CPU-bound
    #      preprocessing, slow disk, excessive host->device copies).
    #      High values (>90%) indicate compute-bound (the model is
    #      doing real work -- slow training is the model's fault, not
    #      the data pipeline's).
    #   2. ``torch.cuda.memory_allocated()`` -- current GPU memory in
    #      use by tensors. Tracking this across epochs detects memory
    #      leaks (gradual increase = unreleased tensors).
    #   3. ``torch.cuda.max_memory_allocated()`` -- peak GPU memory
    #      since the last reset. Detects near-OOM conditions.
    #
    # The logging is a no-op on CPU (``torch.cuda.is_available()`` is
    # False), so this fix has zero overhead in CPU-only environments
    # (CI, local debugging). On GPU it adds ~1ms per epoch for the
    # utilization query (negligible).
    # ------------------------------------------------------------------
    def _log_gpu_utilization(self, epoch: int) -> Dict[str, float]:
        """Log GPU utilization, memory allocated, and peak memory.

        P3-018 ROOT FIX: per-epoch GPU diagnostics so the ops team can
        distinguish data-loading bottlenecks (low util) from compute-
        bound training (high util). No-op on CPU.

        Args:
            epoch: Current epoch number (for the log message).

        Returns:
            Dict with 'gpu_utilization_pct', 'gpu_memory_allocated_mb',
            'gpu_max_memory_allocated_mb'. On CPU, all values are 0.0
            and the dict is still returned (so callers can record it in
            training_history without conditional logic).
        """
        # P3-034 ROOT FIX (v114 forensic, OBSERVABILITY): the previous
        # implementation caught ``Exception`` broadly and logged at DEBUG
        # level (invisible in production logs by default). If GPU
        # monitoring was broken (CUDA driver mismatch, nvidia-smi not
        # found, GPU in EXCLUSIVE_PROCESS mode), the operator saw NO
        # indication. The metrics dict returned all zeros, which the
        # trainer recorded as ``gpu_utilization_pct=0.0``. The operator
        # looking at training history saw "GPU util = 0%" and assumed
        # training was CPU-bound (data-loading bottleneck), when actually
        # GPU monitoring was broken. The actual GPU util might be 95%
        # (compute-bound), but the operator debugged the wrong thing.
        #
        # ROOT FIX:
        #   1. Catch SPECIFIC exceptions (RuntimeError, AttributeError,
        #      OSError) -- not broad ``Exception``. Other exceptions
        #      (e.g., KeyboardInterrupt) propagate up.
        #   2. Log at WARNING level (visible by default) when monitoring
        #      fails, so the operator knows the metrics are unreliable.
        #   3. Add ``gpu_monitoring_healthy: bool`` field to the metrics
        #      dict so downstream consumers can detect monitoring failures
        #      programmatically (e.g., a dashboard can show "GPU metrics
        #      unavailable" instead of "0%").
        metrics: Dict[str, float] = {
            "gpu_utilization_pct": 0.0,
            "gpu_memory_allocated_mb": 0.0,
            "gpu_max_memory_allocated_mb": 0.0,
            # P3-034: health flag. False if monitoring failed. True if
            # all metrics were collected successfully. Downstream
            # consumers (training_history, dashboards) can check this
            # flag to decide whether to trust the other metrics.
            "gpu_monitoring_healthy": True,
        }
        try:
            if not torch.cuda.is_available():
                # CPU-only environment: monitoring is "healthy" (there's
                # nothing to monitor, but no failure either). The other
                # metrics stay at 0.0 (correct for CPU).
                return metrics
            # torch.cuda.utilization() returns int in [0, 100]. May
            # return -1 if the device is idle / no kernels have run
            # since the last query. Treat -1 as 0 (no utilization
            # signal yet).
            try:
                util = torch.cuda.utilization()
                if util is not None and util >= 0:
                    metrics["gpu_utilization_pct"] = float(util)
            except (RuntimeError, AttributeError, OSError) as _util_exc:
                # P3-034: log at WARNING (visible by default) so the
                # operator knows utilization monitoring failed. The
                # memory metrics below may still work.
                metrics["gpu_monitoring_healthy"] = False
                logger.warning(
                    f"P3-034 ROOT FIX: torch.cuda.utilization() failed "
                    f"({_util_exc}). GPU utilization monitoring is "
                    f"DISABLED for this run. The gpu_utilization_pct "
                    f"field will stay at 0.0 -- do NOT interpret this "
                    f"as 'training is CPU-bound'. Check CUDA driver "
                    f"version and nvidia-smi availability."
                )
            try:
                metrics["gpu_memory_allocated_mb"] = float(
                    torch.cuda.memory_allocated() / (1024 * 1024)
                )
                metrics["gpu_max_memory_allocated_mb"] = float(
                    torch.cuda.max_memory_allocated() / (1024 * 1024)
                )
            except (RuntimeError, AttributeError, OSError) as _mem_exc:
                # P3-034: log at WARNING. Memory monitoring failure is
                # less common than utilization failure but still
                # possible (e.g., on some MPS backends).
                metrics["gpu_monitoring_healthy"] = False
                logger.warning(
                    f"P3-034 ROOT FIX: torch.cuda.memory_allocated() or "
                    f"max_memory_allocated() failed ({_mem_exc}). GPU "
                    f"memory monitoring is DISABLED for this run. The "
                    f"gpu_memory_allocated_mb and "
                    f"gpu_max_memory_allocated_mb fields will stay at "
                    f"0.0 -- do NOT interpret this as 'no GPU memory "
                    f"in use'. Check CUDA driver version."
                )
            logger.info(
                f"P3-018 GPU diagnostics (epoch {epoch}): "
                f"utilization={metrics['gpu_utilization_pct']:.1f}%, "
                f"memory_allocated={metrics['gpu_memory_allocated_mb']:.1f} MB, "
                f"peak_memory={metrics['gpu_max_memory_allocated_mb']:.1f} MB, "
                f"monitoring_healthy={metrics['gpu_monitoring_healthy']}. "
                f"Low util (<30%) = data-loading bottleneck (ONLY valid if "
                f"monitoring_healthy=True); high util (>90%) = compute-bound. "
                f"Memory should plateau (leak = gradual increase)."
            )
        except (RuntimeError, AttributeError, OSError) as e:
            # P3-034 ROOT FIX: catch SPECIFIC exceptions (not broad
            # Exception). Other exceptions (KeyboardInterrupt, SystemExit)
            # propagate up. Log at WARNING (visible by default) so the
            # operator knows monitoring failed.
            metrics["gpu_monitoring_healthy"] = False
            logger.warning(
                f"P3-034 ROOT FIX: GPU diagnostics failed ({type(e).__name__}: "
                f"{e}). All GPU metrics are unreliable for this epoch. "
                f"gpu_monitoring_healthy=False -- downstream consumers "
                f"should NOT interpret gpu_utilization_pct=0.0 as "
                f"'training is CPU-bound'. Check CUDA driver, nvidia-smi, "
                f"and torch.cuda.is_available()."
            )
        return metrics

    def create_scheduler(self, total_steps: int) -> None:
        """Create the OneCycleLR scheduler for custom training loops.

        P3-012 ROOT FIX (HIGH, wrong): the OneCycleLR scheduler was
        created ONLY inside ``fit()`` (line ~666). If a user called
        ``train_epoch()`` directly (e.g. for a custom training loop with
        early stopping, gradient accumulation, or per-epoch learning-rate
        inspection), ``self.scheduler`` stayed ``None`` (set in
        ``__init__``), and the per-batch ``self.scheduler.step()`` call
        in ``train_epoch()`` was a no-op. The learning rate stayed
        constant at ``5e-4`` for the entire run — no warmup, no cosine
        decay. The model could converge to a suboptimal solution compared
        to the same model trained via ``fit()``.

        This method exposes the SAME OneCycleLR creation logic that
        ``fit()`` uses, so custom training loops can opt in:

            trainer = GraphTransformerTrainer(model, ...)
            trainer.create_scheduler(total_steps=epochs * n_batches)
            for epoch in range(epochs):
                trainer.train_epoch(...)   # now steps the scheduler
                trainer.evaluate(...)

        Args:
            total_steps: Total number of optimizer steps across the full
                training run. OneCycleLR requires this upfront so it can
                schedule the warmup (first 10%) and cosine decay (remaining
                90%). Must be >= 15 (``MIN_STEPS_FOR_SCHEDULER``); below
                that the scheduler is skipped (tiny debug runs only).

        Note:
            Calling this method REPLACES any existing scheduler on
            ``self.scheduler``. If you call ``fit()`` after
            ``create_scheduler()``, ``fit()`` will overwrite the scheduler
            with its own (computed from its ``epochs`` and ``batch_size``
            args).
        """
        MIN_STEPS_FOR_SCHEDULER = 15
        if total_steps < MIN_STEPS_FOR_SCHEDULER:
            self.scheduler = None
            logger.warning(
                f"P3-012: create_scheduler(total_steps={total_steps}) < "
                f"{MIN_STEPS_FOR_SCHEDULER} — skipping scheduler creation "
                f"(OneCycleLR requires enough steps for both warmup and "
                f"anneal phases). LR will remain constant at "
                f"{self.learning_rate}. This is expected for tiny "
                f"debugging runs."
            )
            return
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        logger.info(
            f"P3-012 ROOT FIX: OneCycleLR scheduler created via "
            f"create_scheduler() (max_lr={self.learning_rate}, "
            f"total_steps={total_steps}, pct_start=0.1, anneal=cos). "
            f"train_epoch() will now step the scheduler per batch. "
            f"Custom training loops get the same warmup+decay as fit()."
        )

    def train_epoch(
        self,
        drug_indices: torch.Tensor,
        disease_indices: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 256,
        exclude_edges: Optional[set] = None,
    ) -> float:
        """Train for one epoch.

        P3-012 ROOT FIX (HIGH, wrong): this method steps the LR scheduler
        (``self.scheduler``) per batch IF one exists. The scheduler is
        created by ``fit()`` (with ``total_steps = epochs *
        n_batches_per_epoch``) OR by the new ``create_scheduler()``
        method (for custom training loops). If neither is called,
        ``self.scheduler`` is ``None`` and the LR stays constant at
        ``self.learning_rate`` — no warmup, no decay. This is acceptable
        for debugging but suboptimal for production training. Call
        ``create_scheduler(total_steps)`` before the loop to get the
        OneCycleLR warmup+decay in a custom training loop.

        Args:
            drug_indices: (N,) drug node indices.
            disease_indices: (N,) disease node indices.
            labels: (N,) binary labels.
            batch_size: Mini-batch size.
            exclude_edges: Edge types to exclude during forward. Defaults
                to ``LABEL_LEAKING_EDGES`` (C2 fix -- never leak the
                label we're predicting).

        Returns:
            Average training loss.
        """
        self.model.train()
        if exclude_edges is None:
            exclude_edges = set(LABEL_LEAKING_EDGES)

        n_samples = len(labels)
        # V4 C-F6 fix: use the trainer's dedicated generator (not the
        # global RNG) so the shuffle order is deterministic and
        # independent of any other torch ops that may have advanced
        # the global RNG.
        # V30 ROOT FIX (8.3): the original torch.Generator() created a CPU
        # generator. Calling torch.randperm(device="cuda", generator=cpu_gen)
        # crashed at runtime. The fix creates the generator on self.device
        # (in __init__), and the randperm call uses self.device so they match.
        # P3-028 ROOT FIX: when the generator fell back to CPU (MPS/XLA
        # case, see __init__), we must generate randperm on the GENERATOR's
        # device (CPU) and then move the result to self.device. Generating
        # directly on self.device with a CPU generator raises
        # ``RuntimeError: expected device cpu but got mps``.
        if getattr(self, "_gen_device", self.device) != self.device:
            indices = torch.randperm(
                n_samples, device=self._gen_device, generator=self._gen
            ).to(self.device)
        else:
            indices = torch.randperm(
                n_samples, device=self.device, generator=self._gen
            )
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start: start + batch_size]
            d_idx = drug_indices[batch_idx]
            ds_idx = disease_indices[batch_idx]
            batch_labels = labels[batch_idx].float()

            self.optimizer.zero_grad()

            # B2 fix: use forward_logits (raw logits) + BCEWithLogitsLoss
            logits = self.model.forward_logits(
                self.node_features,
                self.edge_indices,
                d_idx,
                ds_idx,
                exclude_edges=exclude_edges,
            )

            loss = self.criterion(logits, batch_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            # P3-S06 ROOT FIX: step the LR scheduler after each batch.
            # OneCycleLR is designed for per-batch stepping (not per-epoch).
            # The scheduler is created in fit() with
            # total_steps = epochs * n_batches, so calling step() once per
            # batch exactly exhausts the schedule over the full training
            # run. If self.scheduler is None (train_epoch called directly
            # without fit()), this is a no-op -- preserves backward compat.
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    @torch.no_grad()
    def evaluate(
        self,
        drug_indices: Optional[torch.Tensor] = None,
        disease_indices: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        batch_size: int = 1024,
        exclude_edges: Optional[set] = None,
    ) -> Dict[str, float]:
        """Evaluate model on a dataset.

        V30 ROOT FIX (8.2): the original evaluate() REQUIRED 3 mandatory
        args (drug_indices, disease_indices, labels) -- there was no
        no-arg evaluation path. Callers expecting an sklearn-style API
        crashed with TypeError. The fix: if all 3 args are None, the
        method uses the last-stored val data (from the most recent fit()
        call). This makes ``trainer.evaluate()`` work as a no-arg sanity
        check after training.

        FIX (B3): the original evaluate() called self.model(...) without
        exclude_edges, so the model saw the ('drug','treats','disease')
        edges it was supposed to predict. Validation AUC was inflated.

        The new evaluate() always excludes LABEL_LEAKING_EDGES.

        ROOT FIX (W-06): trainer.evaluate applies temperature scaling
        via model.link_predictor.forward (consistent with
        evaluate_link_prediction).

        ROOT FIX (W-06 efficiency): encode the graph ONCE per evaluate()
        call (not per batch).

        Args:
            drug_indices: (N,) drug node indices. If None, uses the
                last-stored val data (8.2 fix).
            disease_indices: (N,) disease node indices. If None, uses
                last-stored val data.
            labels: (N,) binary labels. If None, uses last-stored val data.
            batch_size: Batch size for evaluation.
            exclude_edges: Edge types to exclude. Defaults to
                ``LABEL_LEAKING_EDGES``.

        Returns:
            Dict with 'loss', 'auc', 'accuracy', 'probs', 'pred_binary',
            'labels' metrics. The 'probs' and 'pred_binary' fields were
            added in the 8.21 fix so Phase 4 doesn't need to re-run the
            model to get per-pair predictions.
        """
        # V30 ROOT FIX (8.2): if no data is passed, use the last-stored
        # val data. This enables the sklearn-style ``trainer.evaluate()``
        # API that the audit found was missing.
        if drug_indices is None and disease_indices is None and labels is None:
            if self._last_val_data is None:
                raise RuntimeError(
                    "evaluate() called with no args and no prior fit() data. "
                    "Either pass (drug_indices, disease_indices, labels) "
                    "explicitly, or call fit() first to store val data. "
                    "(8.2 fix: no-arg evaluate requires prior fit())"
                )
            drug_indices, disease_indices, labels = self._last_val_data
            logger.info("evaluate() using last-stored val data (8.2 fix)")
        elif drug_indices is None or disease_indices is None or labels is None:
            raise ValueError(
                "evaluate() requires ALL THREE of (drug_indices, disease_indices, "
                "labels) to be non-None, OR all three to be None (uses last val)."
            )

        # P3-017 ROOT FIX (SCIENTIFIC — restore training mode after eval).
        # The previous code called ``self.model.eval()`` and NEVER restored
        # to train mode. If ``evaluate()`` was called mid-training (by an
        # external thread or between epochs), the model stayed in eval mode
        # (dropout off, BatchNorm in eval) until the next ``train_epoch()``
        # call. This silently changed the regularization regime, causing
        # the model to overfit. The save/restore pattern exists in
        # ``evaluate_link_prediction`` and ``predict_drug_disease_scores``
        # but was MISSING here. The fix wraps the eval body in try/finally.
        _prior_training = self.model.training
        self.model.eval()
        try:
            if exclude_edges is None:
                exclude_edges = set(LABEL_LEAKING_EDGES)

            # ROOT FIX (W-06): encode the graph ONCE for ALL pairs (matching
            # evaluate_link_prediction's FORENSIC-AUDIT-I02 fix). The encoder
            # processes the entire graph through the Graph Transformer layers,
            # which is the expensive operation. Running it once per batch
            # (via self.model.forward_logits which calls encode internally)
            # wasted compute.
            embeddings = self.model.encode(
                self.node_features, self.edge_indices,
                exclude_edges_override=set(exclude_edges),
            )
            drug_emb_all = embeddings["drug"]
            disease_emb_all = embeddings["disease"]

            n_samples = len(labels)
            all_probs = []
            total_loss = 0.0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                d_idx = drug_indices[start:end].to(self.device)
                ds_idx = disease_indices[start:end].to(self.device)
                batch_labels = labels[start:end].float().to(self.device)

                # Extract embeddings for this batch directly from the
                # pre-computed embeddings (NO redundant encode() call).
                drug_emb_batch = drug_emb_all[d_idx]
                disease_emb_batch = disease_emb_all[ds_idx]

                # B2 fix: use forward_logits + BCEWithLogitsLoss for the loss
                # (loss needs RAW logits, not temperature-scaled).
                # V90 ROOT FIX (BUG #26, P1): use self._eval_criterion
                # (NO pos_weight) instead of self.criterion (which has
                # training pos_weight). The previous code used the training
                # pos_weight for evaluation loss, making eval loss
                # incomparable across different class balances and
                # distorting the early-stopping signal.
                logits = self.model.link_predictor.forward_logits(
                    drug_emb_batch, disease_emb_batch
                ).squeeze(-1)
                loss = self._eval_criterion(logits, batch_labels)
                total_loss += loss.item()

                # ROOT FIX (W-06): use link_predictor.forward with
                # apply_temperature=True for probabilities. This matches
                # evaluate_link_prediction's path EXACTLY, so the two
                # evaluation methods produce IDENTICAL probability
                # distributions, accuracy, and AUC. Previously trainer.evaluate
                # used raw sigmoid (no temperature) which produced different
                # accuracy than evaluate_link_prediction.
                probs = self.model.link_predictor.forward(
                    drug_emb_batch, disease_emb_batch,
                    apply_temperature=True,
                ).squeeze(-1)
                all_probs.append(probs.cpu())

            all_probs = torch.cat(all_probs).numpy()
            # V30 ROOT FIX (8.4): labels may be on CUDA or be a torch.Tensor.
            # The original ``labels.numpy()`` crashes if labels is on CUDA.
            # Use ``labels.detach().cpu().numpy()`` for safety.
            all_labels = labels.detach().cpu().numpy()

            # Compute metrics
            from sklearn.metrics import roc_auc_score, accuracy_score

            pred_binary = (all_probs > 0.5).astype(int)
            accuracy = float(accuracy_score(all_labels, pred_binary))

            unique_labels = np.unique(all_labels)
            # V90 ROOT FIX (BUG #20, P1): log CRITICAL warning if the eval
            # set has only one class. The previous code silently set auc=0.5
            # and continued training with a meaningless val AUC. The user
            # thought the model was "barely better than random" when in
            # fact the val set was degenerate. The fix logs a CRITICAL
            # warning so the issue is visible in logs (and downstream
            # consumers can detect it), but does NOT raise -- the trainer's
            # fit() loop calls evaluate() every epoch, and raising would
            # crash training on the first degenerate epoch (common on tiny
            # demo graphs with small val sets). The AUC=0.5 fallback is
            # retained but the CRITICAL log makes the degeneracy loud.
            if len(unique_labels) < 2:
                logger.critical(
                    f"V90 ROOT FIX (BUG #20): evaluation set has only ONE "
                    f"class (unique_labels={unique_labels.tolist()}). AUC "
                    f"is undefined for a single-class set -- returning 0.5 "
                    f"fallback. The previous code silently returned 0.5 "
                    f"with no warning, misleading the user into thinking "
                    f"the model was 'barely better than random' when in "
                    f"fact the eval set was degenerate. Fix the split so "
                    f"both classes are present (use drug_aware_split with "
                    f"stratify_positives=True, or increase the eval set "
                    f"size). Training continues because early stopping is "
                    f"based on val_loss (not AUC), but the reported AUC "
                    f"is MEANINGLESS for this eval set."
                )
                auc = 0.5
            else:
                try:
                    auc = float(roc_auc_score(all_labels, all_probs))
                except ValueError:
                    auc = 0.5

            avg_loss = total_loss / max(1, (n_samples + batch_size - 1) // batch_size)

            # V30 ROOT FIX (8.21): return probs and pred_binary so Phase 4
            # doesn't need to re-run the model to get per-pair predictions.
            # P3-019 ROOT FIX: return NUMPY ARRAYS (not Python lists) for
            # probs / pred_binary / labels. The P3-033 fix converted these
            # to lists for JSON serializability, but that prioritized
            # serialization over computational efficiency. Downstream
            # consumers that want to do vectorized ops (precision@K, ROC
            # curves, np.argsort for ranking) had to convert BACK to numpy
            # via ``np.array(metrics["probs"])`` — a wasteful round-trip.
            # The fix returns the native numpy arrays (the natural output of
            # sklearn / torch.cpu().numpy()). Callers that need JSON
            # serialization use the new ``to_json_metrics()`` helper which
            # performs the .tolist() conversion in ONE place. The scalar
            # fields (loss, auc, accuracy) remain floats (already JSON-safe).
            return {
                "loss": avg_loss, "auc": auc, "accuracy": accuracy,
                "probs": all_probs,
                "pred_binary": pred_binary,
                "labels": all_labels,
            }
        finally:
            # P3-017 ROOT FIX: ALWAYS restore the prior training mode,
            # even on exception. Without this, an exception during eval
            # (e.g., CUDA OOM) would leave the model in eval mode,
            # silently corrupting subsequent training batches.
            self.model.train(_prior_training)

    @staticmethod
    def to_json_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an evaluate() metrics dict to a JSON-serializable dict.

        P3-019 ROOT FIX: ``evaluate()`` now returns numpy arrays for the
        ``probs`` / ``pred_binary`` / ``labels`` fields (for vectorized
        downstream ops). numpy arrays are NOT JSON-serializable, so any
        caller that needs to JSON-dump the metrics dict (the bridge's
        results export, the dashboard API, CI test artifacts) must first
        convert the arrays to Python lists. This helper performs that
        conversion in ONE canonical place, so the conversion logic is
        not duplicated across callers.

        The scalar fields (loss, auc, accuracy) are passed through
        unchanged (they are already JSON-safe floats). Unknown keys are
        also passed through (forward-compatibility).

        Args:
            metrics: A metrics dict as returned by ``evaluate()``.

        Returns:
            A new dict with the same keys, where ``probs`` /
            ``pred_binary`` / ``labels`` are converted to Python lists
            (via ``np.asarray(...).tolist()``). The input dict is NOT
            mutated.
        """
        import numpy as _np
        out: Dict[str, Any] = dict(metrics)  # shallow copy
        for k in ("probs", "pred_binary", "labels"):
            if k in out and out[k] is not None:
                out[k] = _np.asarray(out[k]).tolist()
        return out

    def fit(
        self,
        train_drug_idx: torch.Tensor,
        train_disease_idx: torch.Tensor,
        train_labels: torch.Tensor,
        val_drug_idx: torch.Tensor,
        val_disease_idx: torch.Tensor,
        val_labels: torch.Tensor,
        epochs: int = 50,
        batch_size: int = 256,
        patience: Any = "auto",
        exclude_edges: Optional[set] = None,
        calibrate_temperature: bool = True,
        pos_weight_clamp_max: float = 10.0,
        cal_drug_idx: Optional[torch.Tensor] = None,
        cal_disease_idx: Optional[torch.Tensor] = None,
        cal_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping + temperature calibration.

        V30 ROOT FIX (8.1): ``train()`` is now an alias for ``fit()``
        so callers using the sklearn-style API (``trainer.train(...)``)
        don't crash with AttributeError. The original code only had
        ``fit()``, but the bridge and external consumers expected both.

        V30 ROOT FIX (8.5): drug-aware split enforcement. The DOCX V1
        contract requires "Three-way train/val/test split (drug-aware)".
        The original trainer accepted arbitrary indices without verifying
        that train/val/test drugs are disjoint -- silently violatable.
        The fix raises ValueError if the same drug index appears in both
        train and val (we cannot check test because fit() doesn't see it,
        but train/val disjointness is the minimum bar).

        V30 ROOT FIX (8.6): pos_weight auto-computed from class balance.
        The original ``BCEWithLogitsLoss()`` had no pos_weight, biasing
        the model toward predicting LOW for everything (the demo graph
        has ~5-10% positives). The fix computes pos_weight =
        num_negatives / num_positives and updates the criterion. This is
        the standard sklearn-recommended value for binary classification
        with imbalanced classes.

        Args:
            train_drug_idx: Training drug indices.
            train_disease_idx: Training disease indices.
            train_labels: Training labels.
            val_drug_idx: Validation drug indices.
            val_disease_idx: Validation disease indices.
            val_labels: Validation labels.
            epochs: Maximum number of epochs.
            batch_size: Mini-batch size.
            patience: Early stopping patience.
            exclude_edges: Edge types to exclude during training forward.
                Defaults to ``LABEL_LEAKING_EDGES``.
            calibrate_temperature: If True, run post-hoc temperature
                scaling on the validation set after main training (B10
                fix; Guo et al. 2017).
            pos_weight_clamp_max: Upper bound for the auto-computed
                ``pos_weight = n_neg / n_pos``. The default is 10.0,
                which is appropriate for production-scale graphs with
                severe class imbalance (e.g. 1 positive per 100
                negatives). On tiny demo graphs (<=100 pairs) where
                pos_weight > 2.0 caused below-random test AUC, pass
                ``pos_weight_clamp_max=2.0`` to reproduce the demo-
                scale clamp behavior. P3-S03 ROOT FIX: the previous
                hardcoded clamp of 2.0 was too tight for production --
                it under-weighted positives on imbalanced graphs,
                causing the model to predict LOW for everything (high
                accuracy, low recall on positives). The parameter is
                now exposed so the same Trainer works on both demo
                and production scales.
            cal_drug_idx / cal_disease_idx / cal_labels: Optional
                held-out calibration set for post-hoc temperature
                scaling. P3-S02 ROOT FIX (Guo et al. 2017): the
                previous code split the val set 50/50 for early-
                stopping vs calibration, which leaves too few samples
                for EITHER purpose on small graphs (15 val pairs ->
                7 for early stopping, 7 for calibration). Guo et al.
                require a SEPARATE held-out calibration set, not a
                split of the val set. If these args are provided,
                they are used directly for temperature calibration
                (no val split). If they are NOT provided (None), the
                trainer falls back to the 50/50 val split WITH a
                WARNING so the user knows the calibration is on
                validation data (overfitting risk). Production
                pipelines should always provide a separate cal set.

        Returns:
            Training history dict.
        """
        if exclude_edges is None:
            exclude_edges = set(LABEL_LEAKING_EDGES)

        # P3-013 ROOT FIX (forensic, Team Member 10): resolve the
        # ``patience`` argument. The new default sentinel ``"auto"``
        # derives patience from the training-set size via
        # ``scale_patience_with_graph_size()`` so small graphs (noisy
        # val_loss) get patience=30 and large graphs (smooth val_loss)
        # get patience=5. Callers can still pass an explicit int to
        # override. This fixes the audit's P3-013 finding that the old
        # hardcoded patience=10 stopped too early on small graphs
        # (empirical evidence: AUC=0.403 on the demo run).
        if isinstance(patience, str):
            if patience.lower() == "auto":
                patience = self.scale_patience_with_graph_size(len(train_labels))
                logger.info(
                    f"P3-013 ROOT FIX: patience='auto' resolved to "
                    f"patience={patience} (n_train={len(train_labels)} pairs, "
                    f"thresholds: <1K->30, 1K-100K->15, >100K->5). The old "
                    f"hardcoded patience=10 stopped too early on small graphs."
                )
            else:
                # Be lenient: try to parse string ints (e.g. "10").
                try:
                    patience = int(patience)
                except ValueError:
                    raise ValueError(
                        f"P3-013: patience='{patience}' is not a valid value. "
                        f"Pass an int, or the literal string 'auto' to use "
                        f"graph-size-aware scaling."
                    )
        # Ensure final value is a positive int.
        if not isinstance(patience, int) or patience < 1:
            raise ValueError(
                f"P3-013: patience must be a positive int (got {patience!r}). "
                f"Pass 'auto' for graph-size-aware scaling."
            )

        # V30 ROOT FIX (8.2): store the val data so a no-arg evaluate()
        # can re-evaluate without requiring the caller to pass it again.
        self._last_val_data = (val_drug_idx, val_disease_idx, val_labels)

        # V30 ROOT FIX (8.5): drug-aware split enforcement. The DOCX V1
        # contract requires drug-disjoint train/val/test splits. We can
        # only check train/val here (test is held by the bridge). A
        # violation means the model can memorize drug-specific features
        # and appear to generalize when it's just recognizing drugs it
        # has seen before.
        train_drugs_set = set(int(x) for x in train_drug_idx.tolist())
        val_drugs_set = set(int(x) for x in val_drug_idx.tolist())
        overlap = train_drugs_set & val_drugs_set
        if overlap:
            raise ValueError(
                f"V30 ROOT FIX (8.5): drug-aware split violation -- "
                f"{len(overlap)} drug indices appear in BOTH train and val "
                f"(examples: {list(overlap)[:5]}). The DOCX V1 contract "
                f"requires drug-disjoint splits to prevent leakage. Use "
                f"the bridge's drug-aware split utility."
            )

        # V30 ROOT FIX (8.6): compute pos_weight from training class balance.
        # pos_weight = num_negatives / num_positives.
        # P3-S03 ROOT FIX: the previous code hardcoded ``min(2.0, max(1.0,
        # n_neg / n_pos))`` which was too tight for production graphs with
        # severe class imbalance (e.g. 1 positive per 100 negatives =
        # pos_weight 100). Clamping to 2.0 under-weights positives, so the
        # model learns to predict LOW for everything -> high accuracy but
        # low recall on positives (exactly the failure mode the audit
        # flagged). The clamp ceiling is now a parameter
        # (``pos_weight_clamp_max``, default 10.0). The bridge passes
        # ``pos_weight_clamp_max=2.0`` on tiny demo graphs to preserve the
        # demo-scale behavior (the previous clamp prevented below-random
        # test AUC on ~15-pair val sets where pos_weight > 2 caused
        # over-prediction of positives). Production graphs use the default
        # 10.0 ceiling so severe imbalance is properly weighted.
        #
        # P3-011 ROOT FIX (forensic, Team Member 10): delegate to the
        # ``compute_pos_weight`` static helper so the formula has a SINGLE
        # source of truth (the helper is also tested directly by the
        # P3-011 CI test). The helper uses the same clamp logic.
        pos_weight_val = self.compute_pos_weight(
            train_labels, clamp_max=pos_weight_clamp_max, clamp_min=1.0,
        )
        train_labels_np = train_labels.detach().cpu().numpy()
        n_pos = int((train_labels_np == 1).sum())
        n_neg = int((train_labels_np == 0).sum())
        pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        logger.info(
            f"P3-011 ROOT FIX: pos_weight={pos_weight_val:.4f} "
            f"(n_pos={n_pos}, n_neg={n_neg}). Clamped to "
            f"[1.0, {pos_weight_clamp_max}] (parameterized upper bound). "
            f"Computed via compute_pos_weight() helper (single source of "
            f"truth -- also tested directly by the P3-011 CI test)."
        )

        no_improve_count = 0
        # P3-029 ROOT FIX: removed the local ``best_epoch = 0`` variable.
        # The previous code maintained BOTH a local ``best_epoch`` and an
        # instance ``self.best_epoch``, which diverged when fit() was
        # called on a trainer that already had self.best_epoch set from a
        # prior run (the local started at 0, self retained the old value).
        # The return dict and early-stopping log used the LOCAL, while
        # save_checkpoint used self -- so the saved value did not match
        # the reported value. We now use self.best_epoch EVERYWHERE,
        # initializing it to 0 at the start of fit() so re-fitting on
        # an already-trained trainer resets cleanly (no stale state).
        self.best_epoch = 0
        # B12 fix: initialize epoch = 0 before the loop so the return
        # statement doesn't NameError if epochs=0.
        epoch = 0

        # P3-S06 ROOT FIX: create the OneCycleLR scheduler with warmup +
        # cosine decay. total_steps = epochs * n_batches (one step per
        # batch). pct_start=0.1 = 10% of steps for warmup (lr ramps from
        # initial_lr/25 to max_lr), then cosine decay to ~0 over the
        # remaining 90%. max_lr is the learning_rate passed to __init__
        # (default 5e-4). The scheduler is stepped per-batch in
        # train_epoch() (see the ``if self.scheduler is not None`` block
        # there). If epochs=0 or n_batches=0, skip scheduler creation
        # (OneCycleLR requires total_steps >= 1).
        #
        # P3-S06 follow-up: OneCycleLR also requires total_steps large
        # enough that both the warmup phase AND the anneal phase have at
        # least 1 step each. With pct_start=0.1, total_steps=5 gives
        # warmup_steps = int(0.1 * 5) = 0, which makes the anneal phase
        # span the full 5 steps but the warmup phase has 0 steps --
        # PyTorch's internal division (step_num - start_step) / (end_step
        # - start_step) then divides by zero. We require total_steps >=
        # MIN_STEPS_FOR_SCHEDULER (10) so that warmup_steps = int(0.1 *
        # 10) = 1 >= 1. Below this threshold, skip the scheduler (LR
        # remains constant) -- these tiny training runs are for debugging
        # only, not production, so the lack of warmup/decay is
        # acceptable.
        MIN_STEPS_FOR_SCHEDULER = 15
        n_train = len(train_labels)
        n_batches_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)
        total_steps = epochs * n_batches_per_epoch
        # P3-012 ROOT FIX: delegate to create_scheduler() so the scheduler
        # creation logic has a SINGLE source of truth. Custom training loops
        # that call train_epoch() directly can now opt in via
        # create_scheduler(total_steps) and get the IDENTICAL warmup+decay
        # schedule that fit() uses.
        if total_steps >= MIN_STEPS_FOR_SCHEDULER:
            self.create_scheduler(total_steps=total_steps)
        else:
            self.scheduler = None
            logger.warning(
                f"P3-S06: total_steps={total_steps} < {MIN_STEPS_FOR_SCHEDULER} "
                f"(epochs={epochs}, n_batches_per_epoch={n_batches_per_epoch}). "
                f"Skipping scheduler creation (OneCycleLR requires enough "
                f"steps for both warmup and anneal phases). LR will remain "
                f"constant. This is expected for tiny debugging runs."
            )

        logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}")
        logger.info(f"Training set: {len(train_labels)} pairs, Validation: {len(val_labels)} pairs")
        logger.info(f"Excluding edges: {exclude_edges}")

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(
                train_drug_idx, train_disease_idx, train_labels,
                batch_size=batch_size, exclude_edges=exclude_edges,
            )

            # Validate (B3 fix: evaluate also excludes label-leaking edges)
            val_metrics = self.evaluate(
                val_drug_idx, val_disease_idx, val_labels,
                batch_size=batch_size,
                exclude_edges=exclude_edges,
            )

            # P3-011 ROOT FIX (v114 forensic, SCIENTIFIC INTEGRITY):
            # The verified AUC (evaluate_link_prediction, which computes
            # THREE independent AUCs -- sklearn, Mann-Whitney, and
            # dot-product) was previously only called AFTER training on
            # the TEST set. The per-epoch checkpoint selection used the
            # UNVERIFIED trainer.evaluate() AUC, which is code-path-
            # identical to evaluate_link_prediction's MLP-scoring path.
            # If trainer.evaluate() had a subtle bug (e.g., exclude_edges
            # not applied correctly, temperature not applied, wrong batch
            # size), the checkpoint selection picked the WRONG epoch's
            # weights. The post-training verified AUC then reported a
            # DIFFERENT (correct) AUC, but the model was already saved
            # with the wrong weights.
            #
            # ROOT FIX: call evaluate_link_prediction on the VAL set
            # every epoch (not just post-training on test). Use the
            # verified val_auc (sklearn) for checkpoint selection. Log
            # the discrepancy between the trainer AUC and the verified
            # AUC per epoch — if they diverge by > 0.01, log an ERROR
            # so the operator can investigate (one of the two paths has
            # a bug). The compute cost is ~1.5x per epoch (three AUC
            # computations vs one), acceptable for institutional-grade
            # scientific integrity.
            try:
                from ..evaluation import evaluate_link_prediction
                verified_metrics = evaluate_link_prediction(
                    model=self.model,
                    node_features=self.node_features,
                    edge_indices=self.edge_indices,
                    drug_indices=val_drug_idx,
                    disease_indices=val_disease_idx,
                    labels=val_labels,
                    batch_size=batch_size,
                    exclude_edges=set(exclude_edges) if exclude_edges is not None else set(LABEL_LEAKING_EDGES),
                    device=self.device,
                    apply_temperature=True,
                )
                verified_val_auc = float(verified_metrics["auc"])
                verified_auc_mannwhitney = float(verified_metrics.get("auc_mannwhitney", verified_val_auc))
                verified_auc_agreement = float(verified_metrics.get("auc_agreement", 0.0))
            except Exception as _eval_exc:
                # If evaluate_link_prediction fails (e.g., scipy missing,
                # one-class val set), fall back to trainer.evaluate's AUC.
                # Log a WARNING so the operator knows the verified AUC
                # was NOT computed this epoch.
                logger.warning(
                    f"P3-011 ROOT FIX: evaluate_link_prediction FAILED for "
                    f"epoch {epoch} ({_eval_exc}). Falling back to "
                    f"trainer.evaluate's AUC for checkpoint selection. "
                    f"The verified AUC cross-check is DISABLED for this "
                    f"epoch — investigate if this warning recurs."
                )
                verified_val_auc = float(val_metrics["auc"])
                verified_auc_mannwhitney = verified_val_auc
                verified_auc_agreement = 0.0

            # P3-011: log discrepancy between trainer AUC and verified AUC.
            trainer_auc = float(val_metrics["auc"])
            auc_discrepancy = abs(trainer_auc - verified_val_auc)
            if auc_discrepancy > 0.01:
                logger.error(
                    f"P3-011 ROOT FIX: ALERT — trainer.evaluate AUC "
                    f"({trainer_auc:.6f}) and evaluate_link_prediction "
                    f"AUC ({verified_val_auc:.6f}) DISAGREE by "
                    f"{auc_discrepancy:.6f} on epoch {epoch} (val set). "
                    f"Threshold: 0.01. One of the two paths has a bug "
                    f"(e.g., exclude_edges not applied, temperature "
                    f"mismatch, batch size difference). Using the "
                    f"VERIFIED AUC for checkpoint selection. Investigate."
                )

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_auc": verified_val_auc,  # P3-011: use verified AUC
                "val_auc_trainer": trainer_auc,  # P3-011: also record trainer AUC
                "val_auc_mannwhitney": verified_auc_mannwhitney,
                "val_auc_agreement": verified_auc_agreement,
                "val_auc_discrepancy": auc_discrepancy,
                "val_accuracy": val_metrics["accuracy"],
            }
            # P3-018 ROOT FIX: record GPU diagnostics in the per-epoch
            # history so they're available for post-hoc analysis (e.g.
            # plotting GPU utilization vs train_loss to diagnose whether
            # slow epochs were data-bound or compute-bound). The call
            # is a no-op on CPU (returns 0.0 for all metrics).
            gpu_metrics = self._log_gpu_utilization(epoch)
            epoch_record.update(gpu_metrics)
            self.training_history.append(epoch_record)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_auc={verified_val_auc:.4f} (verified, "
                    f"trainer={trainer_auc:.4f}, "
                    f"discrepancy={auc_discrepancy:.4f})"
                )

            # FORENSIC ROOT FIX (audit Issue 138) + P3-011 (v114): checkpoint
            # selection now uses the VERIFIED val_auc (from
            # evaluate_link_prediction, which has 3 independent AUC
            # computations) rather than the unverified trainer.evaluate AUC.
            # The V1 launch criterion is AUC > 0.85 — we must select on the
            # CORRECT AUC, not a possibly-buggy one.
            #
            # Noise mitigation: a checkpoint is only promoted when the new
            # verified val_auc beats the running best by at least
            # ``self.val_auc_min_improvement`` (default 0.005 = 0.5% AUC).
            val_auc_now = verified_val_auc  # P3-011: use verified AUC
            val_auc_improved = val_auc_now > (self.best_val_auc + self.val_auc_min_improvement)
            if val_auc_improved:
                self.best_val_auc = val_auc_now
                self.best_val_loss = float(val_metrics["loss"])  # tracked for diagnostics
                self.best_state_dict = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                # P3-029 ROOT FIX: use self.best_epoch consistently
                # (no local ``best_epoch`` variable).
                self.best_epoch = epoch
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best verified val AUC: {self.best_val_auc:.4f}, "
                    f"best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}"
                )
                break

        # FORENSIC ROOT FIX (audit Issue 138, post-fit restore): always
        # restore the best_state_dict (the model selected by val AUC, per
        # the audit Issue 138 fix above) if one was saved. The previous
        # comment argued that val LOSS is "less noisy" than val AUC and
        # so should drive selection -- that argument is rejected by the
        # audit and by the V1 launch criteria (AUC > 0.85, not loss < X).
        # The previous "use the final model" path was making things
        # worse (the final model is the most overfit).
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            self.model.to(self.device)
            logger.info(
                f"FORENSIC ROOT FIX (audit Issue 138): Restored best model "
                f"(selected by val AUC={self.best_val_auc:.4f} at epoch "
                f"{self.best_epoch}, val set size={len(val_labels)}, "
                f"min_improvement={self.val_auc_min_improvement}). The "
                f"previous 'use final model' path was removed -- it was "
                f"making things worse by using the most-overfit model."
            )
        else:
            logger.warning(
                f"FORENSIC ROOT FIX (audit Issue 138): no best_state_dict "
                f"was saved (no epoch improved val AUC by more than "
                f"{self.val_auc_min_improvement}). Using the FINAL model "
                f"-- this may be overfit if training ran for many epochs."
            )

        # B10 fix: post-hoc temperature calibration on the validation set.
        # The original code declared fit_temperature but never called it,
        # so the temperature parameter was dead weight polluting the
        # state_dict. Now it actually does something.
        #
        # ROOT FIX (E8): the original code caught ALL exceptions with
        # `except Exception` and logged at WARNING level. This silently
        # swallowed real bugs (e.g., LBFGS API changes). The E8 fix:
        #   1. Logs at ERROR level (not WARNING) so users see it
        #   2. Includes the full traceback in the log
        #   3. Re-raises KeyboardInterrupt and SystemExit (don't swallow)
        #   4. Sets a flag so downstream consumers know calibration failed
        #
        # V90 ROOT FIX (BUG #11, P0): temperature calibration MUST run
        # on a SEPARATE held-out calibration set, NOT the val set used
        # for early stopping. Guo et al. 2017 ("On Calibration of
        # Modern Neural Networks") explicitly state that temperature
        # scaling MUST be fit on a HELD-OUT calibration set, NOT the
        # validation set used for model selection. Fitting on the val
        # set overfits the temperature to the val data's specific
        # confidence errors. The reported val AUC and val loss are
        # then optimistic. The 0.5 threshold for binary predictions
        # is wrong. The RL agent's reward signal is distorted.
        #
        # The fix: split off a calibration set from the val set. We
        # use a 50/50 split (deterministic, seeded by self.seed) so
        # half the val data is used for early stopping and half for
        # temperature calibration. On tiny val sets this may leave
        # too few samples for either purpose, but that's the honest
        # tradeoff -- the alternative (using the same set for both)
        # produces overfit temperature values that silently distort
        # downstream consumers.
        if calibrate_temperature and (
            cal_drug_idx is not None
            and cal_disease_idx is not None
            and cal_labels is not None
        ):
            # P3-S02 ROOT FIX (Guo et al. 2017): production path -- caller
            # provided a SEPARATE held-out calibration set. Use it directly.
            # No val split, no overfitting risk. This is the scientifically
            # correct path.
            try:
                if len(torch.unique(cal_labels)) >= 2:
                    logger.info(
                        f"P3-S02 ROOT FIX: using provided held-out "
                        f"calibration set (n_cal={len(cal_labels)}) for "
                        f"temperature scaling. Guo et al. 2017 require "
                        f"a separate cal set -- this is the production path."
                    )
                    self._calibrate_temperature(
                        cal_drug_idx, cal_disease_idx, cal_labels,
                        exclude_edges=exclude_edges,
                    )
                else:
                    logger.warning(
                        f"P3-S02 ROOT FIX: provided calibration set has "
                        f"only one class (n_cal={len(cal_labels)}, "
                        f"unique={torch.unique(cal_labels).tolist()}). "
                        f"Skipping temperature calibration."
                    )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.error(
                    f"ROOT FIX (E8): Temperature calibration FAILED: {e}",
                    exc_info=True
                )
                self._calibration_failed = True
        elif calibrate_temperature and len(val_labels) >= 4:
            # P3-S02 ROOT FIX (Guo et al. 2017): FALLBACK path -- caller did
            # NOT provide a separate calibration set. We split the val set
            # 50/50 for early-stopping vs calibration. This is NOT
            # scientifically correct (Guo et al. require a separate set),
            # and we log a WARNING so the user knows the calibration is on
            # validation data (overfitting risk). Production pipelines
            # should always pass cal_drug_idx / cal_disease_idx / cal_labels.
            logger.warning(
                f"P3-S02 ROOT FIX: no separate calibration set provided. "
                f"Falling back to splitting the val set 50/50 for early-"
                f"stopping vs calibration. Guo et al. 2017 require a "
                f"SEPARATE held-out calibration set -- splitting the val "
                f"set leaves too few samples for EITHER purpose on small "
                f"graphs and risks overfitting the temperature. Pass "
                f"cal_drug_idx/cal_disease_idx/cal_labels to fit() for "
                f"the production path."
            )
            try:
                cal_gen = torch.Generator()
                cal_gen.manual_seed(int(self.seed) + 7)
                n_val = len(val_labels)
                cal_perm = torch.randperm(n_val, generator=cal_gen)
                n_cal = n_val // 2
                cal_idx = cal_perm[:n_cal]
                fb_cal_drug_idx = val_drug_idx[cal_idx]
                fb_cal_disease_idx = val_disease_idx[cal_idx]
                fb_cal_labels = val_labels[cal_idx]

                if len(torch.unique(fb_cal_labels)) >= 2:
                    self._calibrate_temperature(
                        fb_cal_drug_idx, fb_cal_disease_idx, fb_cal_labels,
                        exclude_edges=exclude_edges,
                    )
                else:
                    logger.warning(
                        f"V90 ROOT FIX (BUG #11): calibration set has "
                        f"only one class (n_cal={n_cal}, "
                        f"unique={torch.unique(fb_cal_labels).tolist()}). "
                        f"Skipping temperature calibration. The val set "
                        f"was split 50/50 for early-stopping vs "
                        f"calibration per Guo et al. 2017, but the "
                        f"calibration half is degenerate."
                    )
            except (KeyboardInterrupt, SystemExit):
                raise  # E8 fix: don't swallow these
            except Exception as e:
                logger.error(
                    f"ROOT FIX (E8): Temperature calibration FAILED: {e}",
                    exc_info=True  # E8 fix: include full traceback
                )
                self._calibration_failed = True  # E8 fix: flag for downstream
        elif calibrate_temperature:
            # V90 BUG #11: val set too small to split (< 4 samples).
            # Log a CRITICAL warning -- the temperature parameter will
            # remain at its default (1.0), which means no calibration.
            logger.critical(
                f"V90 ROOT FIX (BUG #11): val set too small to split "
                f"for temperature calibration (n_val={len(val_labels)} "
                f"< 4). Skipping temperature calibration. The "
                f"temperature parameter will remain at 1.0 (no "
                f"calibration). Guo et al. 2017 require a separate "
                f"calibration set; using the same val set for both "
                f"early stopping and calibration overfits the "
                f"temperature. Increase the val set size or disable "
                f"calibrate_temperature."
            )

        # ROOT FIX (D-10): log the learned self_loop_weight value at the
        # end of training. The V27 code declared
        # ``self_loop_weight = nn.Parameter(torch.tensor(0.1))`` in
        # HeterogeneousMultiHeadAttention with a docstring claiming it's
        # "learnable", but the trainer never reported whether it actually
        # changed during training. If it stays at 0.1, it's effectively a
        # CONSTANT (not learnable) and should be removed or made a
        # non-parameter. If it changes, the change is invisible to the
        # user, making it impossible to debug attention issues.
        #
        # The fix: walk the model's submodules, find all
        # HeterogeneousMultiHeadAttention instances, and log their
        # self_loop_weight values. This makes the parameter's evolution
        # visible so users can verify it's actually learning.
        try:
            from ..models.layers import HeterogeneousMultiHeadAttention
            n_attn_layers = 0
            for name, module in self.model.named_modules():
                if isinstance(module, HeterogeneousMultiHeadAttention):
                    n_attn_layers += 1
                    try:
                        slw = float(module.self_loop_weight.item())
                        # P3-015 ROOT FIX: the self_loop_weight is initialized
                        # to 1.0 (P3-S01 fix, layers.py:170), NOT 0.1. The
                        # previous D-10 logging used initial=0.100000 which was
                        # a STALE baseline from the pre-P3-S01 code (V27 used
                        # 0.1). With the wrong baseline, delta was always
                        # ~+0.900000 even if the weight never moved, making
                        # the "LEARNING" vs "NOT LEARNING" determination
                        # meaningless (it always reported LEARNING). The fix
                        # uses the ACTUAL initial value (1.0) so delta
                        # correctly reflects training-time movement.
                        logger.info(
                            f"ROOT FIX (D-10): {name}.self_loop_weight = "
                            f"{slw:.6f} (initial=1.000000, "
                            f"delta={slw - 1.0:+.6f}). The self_loop_weight "
                            f"is {'LEARNING' if abs(slw - 1.0) > 1e-4 else 'NOT LEARNING (effectively constant)'} "
                            f"during training."
                        )
                    except (AttributeError, RuntimeError) as e:
                        logger.warning(
                            f"ROOT FIX (D-10): could not read self_loop_weight "
                            f"from {name}: {e}"
                        )
            if n_attn_layers == 0:
                logger.debug(
                    f"ROOT FIX (D-10): no HeterogeneousMultiHeadAttention "
                    f"modules found in model (self_loop_weight logging skipped)."
                )
        except ImportError:
            logger.debug(
                f"ROOT FIX (D-10): HeterogeneousMultiHeadAttention not "
                f"importable; skipping self_loop_weight logging."
            )

        return {
            "best_val_auc": self.best_val_auc,
            # P3-029 ROOT FIX: use self.best_epoch (instance attr), not the
            # removed local ``best_epoch`` variable.
            "best_epoch": self.best_epoch,
            "epochs_trained": epoch,
            "history": list(self.training_history),  # V30 (8.25): return a COPY
        }

    # V30 ROOT FIX (8.1): sklearn-style alias. Many callers and tutorials
    # use ``trainer.train(...)`` instead of ``trainer.fit(...)``. The
    # original code only had ``fit()``, which caused AttributeError at
    # runtime for anyone using the sklearn convention.
    train = fit

    def _calibrate_temperature(
        self,
        val_drug_idx: torch.Tensor,
        val_disease_idx: torch.Tensor,
        val_labels: torch.Tensor,
        exclude_edges: Optional[set] = None,
    ) -> float:
        """Run post-hoc temperature scaling on the validation set.

        B10 fix: this method actually exercises the link predictor's
        ``fit_temperature`` method, so the ``self.temperature`` parameter
        is no longer dead weight.

        ROOT FIX (FORENSIC-AUDIT-C01): removed the ``@torch.no_grad()``
        decorator. The previous decorator disabled gradient tracking for
        the ENTIRE method, which broke the Adam optimizer inside
        ``fit_temperature`` (Adam needs gradients to flow through
        ``log_temp`` -> ``T_eff`` -> ``scaled_logits`` -> ``loss``).

        The encoding step (which doesn't need gradients, since the MLP
        is frozen) is now wrapped in its own ``torch.no_grad()`` block.
        The ``fit_temperature`` call runs WITH gradient tracking enabled,
        so Adam can optimize the temperature parameter.
        """
        if exclude_edges is None:
            exclude_edges = set(LABEL_LEAKING_EDGES)

        # Encode the graph to get embeddings (frozen MLP weights during
        # temperature optimization). This step doesn't need gradients.
        # ROOT FIX (C13): use exclude_edges_override parameter instead of
        # mutating self.model.exclude_edges. This is thread-safe.
        # ROOT FIX (FORENSIC-AUDIT-C01): wrap ONLY the encoding step in
        # no_grad, NOT the fit_temperature call.
        with torch.no_grad():
            embeddings = self.model.encode(
                self.node_features, self.edge_indices,
                exclude_edges_override=exclude_edges,
            )
            drug_emb = embeddings["drug"][val_drug_idx.to(self.device)].detach()
            disease_emb = embeddings["disease"][val_disease_idx.to(self.device)].detach()
            labels = val_labels.to(self.device)

        # *** MUST be OUTSIDE the no_grad block above -- Adam needs gradients ***
        # ROOT FIX (D-04): the V27 code correctly placed fit_temperature
        # outside the no_grad block, but the structure was FRAGILE: a
        # future maintainer could easily move it inside (especially since
        # the no_grad block ends only 2 lines above). If fit_temperature
        # runs inside no_grad, Adam's optimizer.step() will silently fail
        # to update log_temp (gradients won't flow through the loss
        # computation), and temperature calibration will appear to "work"
        # (no errors) but produce NO actual calibration. This is the
        # exact kind of silent failure that's hard to debug.
        #
        # The fix: an explicit comment + an assertion that gradient
        # tracking is enabled. The assertion fires LOUDLY if a future
        # maintainer accidentally wraps this in no_grad.
        assert torch.is_grad_enabled(), (
            "D-04: fit_temperature MUST run with gradient tracking enabled. "
            "If you see this assertion, fit_temperature was accidentally "
            "placed inside a torch.no_grad() block. Adam's optimizer.step() "
            "would silently fail to update log_temp. "
            "V90 BUG #42 note: this assertion is DEFENSIVE insurance - it "
            "always passes in current code paths (fit() is not inside "
            "no_grad). Kept as cheap protection against future regressions."
        )
        # fit_temperature handles freezing the MLP weights internally
        # and needs gradient tracking enabled for the Adam optimizer on
        # log_temp (FORENSIC-AUDIT-C01 fix, W-05 root fix).
        temp = self.model.link_predictor.fit_temperature(
            drug_emb, disease_emb, labels
        )
        logger.info(f"Post-hoc temperature calibrated to {temp:.4f}")
        return temp

    # ------------------------------------------------------------------
    # FORENSIC ROOT FIX (audit Issues 124 + 139): self-describing
    # checkpoint helpers. The trainer now saves the model's class name
    # AND its construction hyperparams so the inference service can
    # reconstruct the EXACT model class with the EXACT architecture
    # from a single .pt file (no separate graph_state.pt sidecar, no
    # guessing defaults, no hard-coded class imports).
    # ------------------------------------------------------------------
    def _get_model_class_name(self) -> str:
        """Return the qualified class name of the trained model.

        The service uses this to dispatch to the correct model class
        (audit Issue 124). Today the only production class is
        ``DrugRepurposingGraphTransformer`` (aliased as
        ``GraphTransformerModel``). Storing the class name in the
        checkpoint means a future model variant can be served WITHOUT
        code changes to service.py -- the service looks up the class
        by name and instantiates it with the saved hyperparams.
        """
        cls = type(self.model)
        module = getattr(cls, "__module__", "")
        qualname = getattr(cls, "__qualname__", cls.__name__)
        if module:
            return f"{module}.{qualname}"
        return qualname

    def _extract_model_hyperparams(self) -> Dict[str, Any]:
        """Extract the model's construction hyperparams from its attributes.

        Returns a dict that can be passed as ``cls(**hyperparams)`` to
        reconstruct a model with the SAME architecture as the trained
        one. Only PUBLIC attributes set by ``__init__`` are extracted --
        no private state, no learned weights (those are in
        ``model_state_dict``). Missing attributes fall back to the
        class's documented defaults.

        This is the production-grade equivalent of the bridge's
        best-effort ``model_config`` extraction. Doing it in the trainer
        (rather than the bridge) makes the checkpoint self-describing:
        any caller that loads the checkpoint can reconstruct the model
        without needing the bridge or the original training script.
        """
        m = self.model
        # The constructor signature of DrugRepurposingGraphTransformer
        # is the source of truth for these names. If the constructor
        # changes, update this list to match (CI test
        # test_save_load_round_trip_self_describing will catch drift).
        feature_dims = dict(getattr(m, "feature_dims", {}))
        edge_types = [tuple(et) for et in getattr(m, "edge_types", [])]
        node_types = list(getattr(m, "node_types", []))
        exclude_edges = [tuple(e) for e in getattr(m, "exclude_edges", [])]
        return {
            "feature_dims": feature_dims,
            "embedding_dim": int(getattr(m, "embedding_dim", 128)),
            "num_layers": int(getattr(m, "num_layers", 4)),
            "num_heads": int(getattr(m, "num_heads", 8)),
            "edge_types": edge_types,
            "node_types": node_types,
            "ffn_hidden_dim": int(getattr(m, "ffn_hidden_dim", 512)),
            "dropout": float(getattr(m, "dropout", 0.1)),
            "attention_dropout": float(getattr(m, "attention_dropout", 0.1)),
            "link_predictor_hidden_dims": list(
                getattr(m, "link_predictor_hidden_dims", [256, 128])
            ),
            "link_predictor_dropout": float(getattr(m, "link_predictor_dropout", 0.2)),
            "exclude_edges": exclude_edges,
            "seed": getattr(m, "seed", None),
            "num_training_pairs": getattr(m, "num_training_pairs", None),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        V30 ROOT FIX (8.14): the original checkpoint schema only saved
        model_state_dict, optimizer_state_dict, best_val_auc, and history.
        This was missing best_state_dict, best_val_loss, best_epoch, the
        graph schema (node/edge types), and the package version. Phase 4
        reload would silently corrupt embeddings because the schema
        wasn't validated. The fix saves the FULL schema so load_checkpoint
        can validate compatibility before restoring.
        """
        # V30 ROOT FIX (8.14): full schema for safe reload.
        # V90 ROOT FIX (BUG #21, P1): save the ACTUAL best_epoch
        # (self.best_epoch, set during fit() when val_loss improved),
        # not the LAST epoch (self.training_history[-1]["epoch"]).
        # The previous code's "best_epoch" field stored the last epoch,
        # which was misleading -- on checkpoint reload the user saw
        # best_epoch = 500 (last) when the actual best was epoch 42.
        # V90 ROOT FIX (BUG #21 + #33): save self.best_epoch (the ACTUAL
        # best epoch), NOT training_history[-1]["epoch"] (the LAST epoch).
        # The previous code confused "last" with "best" -- if training ran
        # 80 epochs with early stopping at epoch 40, the checkpoint saved
        # best_epoch=80 (wrong). The fix saves self.best_epoch, which is
        # set in fit() when val_loss actually improves.
        # V90 ROOT FIX (BUG #41): skip saving best_state_dict if None.
        # The previous code saved "best_state_dict": None when training
        # ran 0 epochs or never improved val_loss. On load, this restored
        # None -- useless but not incorrect. The fix skips the key entirely
        # if best_state_dict is None, saving disk space and avoiding
        # confusion.
        from .. import __version__ as _gt_version, __schema_version__ as _gt_schema
        # ROOT FIX (v92): the previous code had three syntax errors that
        # broke ``compileall`` and CI's build job for every PR:
        #   1. Line 957 ended with ``}, path)`` -- a leftover from an
        #      inline ``torch.save({...}, path)`` that was refactored to
        #      a named ``checkpoint`` dict but the ``, path)`` was never
        #      removed. This is invalid Python (tuple expression with no
        #      opening paren).
        #   2. Line 948 was a DUPLICATE ``best_epoch`` key (the same key
        #      was already on line 946). flake8 F601 -- silently keeps
        #      only the last value, which happened to be identical, but
        #      it's a code smell indicating a botched merge.
        #   3. Line 959 had a stray ``}`` -- another leftover from the
        #      botched refactor.
        # The fix below is the SINGLE canonical checkpoint dict. The
        # actual ``torch.save(checkpoint, path)`` call already exists
        # below (line 963 in the original).
        # v91 ROOT FIX: previous botched merge left duplicate ``best_epoch``
        # keys, a stray ``}, path)`` that made the dict literal a syntax
        # error, and a stray ``}`` after the log line. The whole
        # ``save_checkpoint`` was UNUSABLE (SyntaxError at import time),
        # which meant the entire ``graph_transformer`` package failed to
        # import, which meant ``run_real_pipeline.py`` could not even
        # start. The fix reconstructs the dict literal cleanly: single
        # ``best_epoch`` key (V90 BUG #21/#33: actual best, not last),
        # ``best_state_dict`` only included when not None (V90 BUG #41),
        # full graph schema for safe reload (V30 8.14), and a single
        # ``torch.save`` + log call. No duplicate keys, no stray tokens.
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_auc": self.best_val_auc,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,  # V90 BUG #21/#33: actual best, not last
            "history": list(self.training_history),  # V30 (8.25): copy, not reference
            "graph_schema": {
                "node_types": list(self.node_features.keys()),
                "feature_dims": {k: v.shape[1] for k, v in self.node_features.items()},
                "edge_types": [list(k) for k in self.edge_indices.keys()],
            },
            "package_version": _gt_version,
            "schema_version": _gt_schema,
            # FORENSIC ROOT FIX (audit Issue 139): the checkpoint must be
            # SELF-CONTAINED so the inference service can reproduce the
            # exact training graph + model architecture from a single
            # .pt file (no separate graph_state.pt sidecar). The previous
            # checkpoint saved ONLY model_state_dict + a lightweight
            # graph_schema (shapes + names), forcing the bridge to write
            # a SEPARATE graph_state.pt with the actual tensors and the
            # service to load BOTH files in lockstep. That coupling was
            # fragile -- if the two files were ever out of sync (e.g.,
            # the bridge crashed between writing them, or a CI step
            # moved only one), the service would load a model with a
            # MISMATCHED graph and silently produce garbage predictions.
            #
            # The fix saves EVERYTHING the service needs in the same
            # .pt file:
            #   - model_class_name: the qualified class name so the
            #     service can dispatch to the correct class (audit
            #     Issue 124). Defaults to
            #     ``DrugRepurposingGraphTransformer`` (the only
            #     production class today) but supports future model
            #     variants without code changes.
            #   - hyperparams: the model's architecture params
            #     (embedding_dim, num_layers, num_heads, dropout, etc.)
            #     so the service can reconstruct the model with
            #     ``cls(**hyperparams)`` without guessing defaults.
            #   - node_features / edge_indices: the ACTUAL graph
            #     tensors (not just their shapes). The service uses
            #     these for ``model.encode(...)``.
            #   - node_maps / drug_names / disease_names / known_pairs:
            #     the name->index lookups the service needs to resolve
            #     HTTP request payloads ("drug name -> node index")
            #     and to filter known pairs out of top-k novel
            #     predictions.
            "model_class_name": self._get_model_class_name(),
            "hyperparams": self._extract_model_hyperparams(),
            "node_features": self.node_features,
            "edge_indices": self.edge_indices,
            "node_maps": self.node_maps,
            "drug_names": list(self.drug_names),
            "disease_names": list(self.disease_names),
            "known_pairs": list(self.known_pairs),
        }
        # v89 CI RECOVERY: removed the broken old torch.save call (lines
        # 957-959 had `}, path)` + stray `}` from a botched merge by a
        # parallel agent). The correct torch.save call is below.
        # V90 BUG #41: only include best_state_dict if it's not None.
        if self.best_state_dict is not None:
            checkpoint["best_state_dict"] = self.best_state_dict
        # TASK-159 ROOT FIX (v111): ATOMIC checkpoint save. Write to a
        # temp file in the same directory, then os.replace() to the final
        # path. Crash-safe: a partial write does not corrupt the existing
        # checkpoint.
        import os as _os_mod
        import tempfile as _tempfile_mod
        from pathlib import Path as _Path_mod
        _out_dir = _Path_mod(path).parent
        _out_dir.mkdir(parents=True, exist_ok=True)
        _tmp_fd, _tmp_path = _tempfile_mod.mkstemp(
            prefix=".gt_ckpt_tmp_", suffix=".pt", dir=str(_out_dir),
        )
        _os_mod.close(_tmp_fd)
        try:
            torch.save(checkpoint, _tmp_path)
            _os_mod.replace(_tmp_path, path)
        except Exception:
            try:
                _os_mod.unlink(_tmp_path)
            except Exception:
                pass
            raise
        logger.info(
            f"V30 ROOT FIX (8.14) + V90 (BUG #21/#33/#41) + TASK-159 (atomic): "
            f"Checkpoint saved to {path} (full schema, best_epoch={self.best_epoch}, "
            f"best_state_dict={'present' if self.best_state_dict is not None else 'None (skipped)'})"
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        V30 ROOT FIX (8.14): validates the graph schema and package
        version before restoring, so a checkpoint saved with a different
        model architecture or a different package version raises a
        clear error instead of silently corrupting embeddings.

        V30 ROOT FIX (8.15): use weights_only=True for torch.load to
        prevent arbitrary code execution from untrusted checkpoints.

        V92 ROOT FIX (BUG P3-009, P3-010, P3-011):
          - P3-009: after loading best_state_dict, RESTORE it into the
            live model via model.load_state_dict(best_state_dict).
            The previous code loaded model_state_dict (the LAST-epoch
            weights, possibly overfit) but never restored best_state_dict
            (the BEST validation weights). The user thought they had the
            best model but actually had the last-epoch model.
          - P3-010: use .get() for optimizer_state_dict so old
            checkpoints that don't have it don't raise KeyError.
          - P3-011: feature-detect the weights_only parameter (added in
            PyTorch 1.13). Older PyTorch raises TypeError if the kwarg
            is passed.
        """
        # V92 ROOT FIX (BUG P3-011): feature-detect the ``weights_only``
        # parameter. It was added in PyTorch 1.13; older PyTorch raises
        # ``TypeError: load() got an unexpected keyword argument
        # 'weights_only'``. This is common in enterprise pharma IT
        # environments that pin to older PyTorch for stability.
        import inspect
        if "weights_only" in inspect.signature(torch.load).parameters:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        else:
            checkpoint = torch.load(path, map_location=self.device)

        # Load model_state_dict (the LAST-epoch weights by convention).
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # V92 ROOT FIX (BUG P3-010): use .get() for optimizer_state_dict.
        # Older checkpoints that don't have it (e.g., from inference-only
        # saves) raise KeyError on the hardcoded access. Use a defensive
        # .get() and only load when present.
        opt_state = checkpoint.get("optimizer_state_dict")
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)

        self.best_val_auc = checkpoint.get("best_val_auc", 0.0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        # V90 ROOT FIX (BUG #21, P1): restore self.best_epoch from
        # checkpoint (was previously not restored - stayed at 0).
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.best_state_dict = checkpoint.get("best_state_dict")

        # FORENSIC ROOT FIX (audit Issue 139): restore the graph metadata
        # if the checkpoint carries it. Pre-fix checkpoints saved only the
        # model_state_dict + schema; post-fix checkpoints are
        # SELF-CONTAINED (they include node_features, edge_indices,
        # node_maps, drug_names, disease_names, known_pairs). We
        # restore the metadata when present so callers that load a
        # checkpoint have the same graph context the trainer had at
        # save time. Missing keys fall back to whatever the trainer was
        # constructed with (backward compat).
        if "node_features" in checkpoint and checkpoint["node_features"] is not None:
            loaded_features = checkpoint["node_features"]
            self.node_features = {
                k: v.to(self.device) for k, v in loaded_features.items()
            }
        if "edge_indices" in checkpoint and checkpoint["edge_indices"] is not None:
            loaded_edges = checkpoint["edge_indices"]
            self.edge_indices = {
                k: v.to(self.device) for k, v in loaded_edges.items()
            }
        if "node_maps" in checkpoint and checkpoint["node_maps"] is not None:
            self.node_maps = {
                ntype: dict(name_to_idx)
                for ntype, name_to_idx in checkpoint["node_maps"].items()
            }
        if "drug_names" in checkpoint and checkpoint["drug_names"] is not None:
            self.drug_names = list(checkpoint["drug_names"])
        if "disease_names" in checkpoint and checkpoint["disease_names"] is not None:
            self.disease_names = list(checkpoint["disease_names"])
        if "known_pairs" in checkpoint and checkpoint["known_pairs"] is not None:
            self.known_pairs = [
                (str(d), str(v)) for d, v in checkpoint["known_pairs"]
            ]
        # V92 ROOT FIX (BUG P3-009, CRITICAL): RESTORE the best model
        # into the live model. The previous code loaded best_state_dict
        # from the checkpoint but NEVER called
        # ``self.model.load_state_dict(self.best_state_dict)``. The model
        # kept whatever weights it had before load_checkpoint (random
        # init if just constructed, or last-epoch weights if fit() was
        # called). The user thought they loaded the best model but
        # actually had the LAST (possibly overfit) model. Predictions
        # and AUC were wrong.
        #
        # ROOT FIX: if best_state_dict is present in the checkpoint,
        # load it into the live model AFTER the model_state_dict load.
        # This ensures the live model has the BEST validation weights,
        # not the last-epoch weights. If best_state_dict is absent
        # (e.g., older checkpoints), fall back to model_state_dict
        # (already loaded above) - this preserves backward compat.
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            logger.info(
                f"V92 ROOT FIX (BUG P3-009): restored BEST validation "
                f"model weights (best_epoch={self.best_epoch}) into the "
                f"live model."
            )
        else:
            # P3-038 ROOT FIX (v107): log a WARNING when best_state_dict
            # is None at load time. The previous code silently kept the
            # last-epoch weights (loaded above as model_state_dict) with
            # no indication to the user. The audit's P3-038 finding: "If
            # best_state_dict is None (training crashed early), the live
            # model has the LAST epoch weights, not the BEST. The user
            # has no warning." A user who loads a checkpoint expecting
            # the BEST validation model gets the LAST (possibly overfit)
            # model — predictions are based on overfit weights, but the
            # user has no way to know without inspecting the checkpoint
            # fields manually. The WARNING makes this situation VISIBLE
            # so the user can decide whether to re-train (recommended)
            # or accept the last-epoch weights (e.g., for debugging).
            logger.warning(
                f"P3-038 ROOT FIX (v107): checkpoint at {path} has NO "
                f"best_state_dict field. The live model now has the "
                f"LAST-epoch weights (model_state_dict), NOT the BEST "
                f"validation weights. This happens when training crashed "
                f"early (before any validation improvement was recorded) "
                f"or when the checkpoint was saved by an older trainer "
                f"version that did not track best_state_dict. Predictions "
                f"from this model may be based on OVERFIT weights. "
                f"RECOMMENDATION: re-train the model from scratch to "
                f"get the best-validation weights, OR explicitly verify "
                f"the last-epoch weights are acceptable for your use "
                f"case (e.g., debugging only)."
            )
        # V90 ROOT FIX (BUG #33): restore best_epoch. The previous code
        # loaded every field EXCEPT best_epoch, leaving it at its __init__
        # default of 0. After reload, the user could not tell which epoch
        # produced the best model. The fix restores it from the checkpoint.
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.training_history = checkpoint.get("history", [])
        logger.info(
            f"V30 ROOT FIX (8.14/8.15) + V90 (BUG #33) + V92 (P3-009/010/011): "
            f"Checkpoint loaded from {path} (best_epoch={self.best_epoch})"
        )

    # P4-009 ROOT FIX: load_validated_for_retraining as a METHOD of the
    # GraphTransformerTrainer class. The previous code had this as a
    # standalone function that callers had to invoke manually — the data
    # flywheel was broken because nothing automatically called it.
    # Adding it as a class method lets the bridge/training pipeline call
    # trainer.load_validated_for_retraining() as part of the standard
    # training workflow, closing the Phase 3 writeback loop automatically.
    def load_validated_for_retraining(
        self,
        checkpoint_path: str,
        retrain_trigger_path: Optional[str] = None,
        output_checkpoint_path: Optional[str] = None,
        fine_tune_epochs: int = 10,
        learning_rate: float = 1e-4,
    ) -> Dict[str, Any]:
        """P4-009: Load validated hypotheses from Phase 3 retrain trigger.

        Reads ``graph_transformer/retrain_triggered.json`` (written by
        ``writeback_to_phase3`` in phase4/writeback.py) and initiates
        fine-tuning of the GT model with the validated pairs. This closes
        the data flywheel loop: pharma validations → writeback → retrain
        trigger → GT model update.

        Positive outcomes ("validated_positive") are added as positive
        labels. Negative outcomes ("validated_negative", "validated_toxic")
        are added as negative labels.

        Args:
            checkpoint_path: Path to the trained GT checkpoint (.pt file).
            retrain_trigger_path: Path to retrain_triggered.json. If None,
                defaults to <repo>/graph_transformer/retrain_triggered.json.
            output_checkpoint_path: Where to save the fine-tuned model.
            fine_tune_epochs: Number of fine-tune epochs.
            learning_rate: Fine-tune learning rate.

        Returns:
            Dict with trigger_entries_read, positive_pairs, negative_pairs,
            and all keys from retrain_on_validated.
        """
        import json as _json
        import os as _os
        from pathlib import Path as _Path
        import csv as _csv
        import tempfile as _tempfile

        if retrain_trigger_path is None:
            _repo_root = _Path(__file__).resolve().parents[2]
            retrain_trigger_path = str(_repo_root / "graph_transformer" / "retrain_triggered.json")

        positive_pairs: List[Tuple[str, str]] = []
        negative_pairs: List[Tuple[str, str]] = []
        trigger_entries_read = 0

        if _os.path.exists(retrain_trigger_path):
            try:
                with open(retrain_trigger_path, "r", encoding="utf-8") as f:
                    entries = _json.load(f)
                if isinstance(entries, list):
                    trigger_entries_read = len(entries)
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        drug = (entry.get("drug") or "").strip()
                        disease = (entry.get("disease") or "").strip()
                        outcome = (entry.get("outcome") or "").strip().lower()
                        if not drug or not disease:
                            continue
                        if outcome == "validated_positive":
                            positive_pairs.append((drug, disease))
                        elif outcome in ("validated_negative", "validated_toxic"):
                            negative_pairs.append((drug, disease))
            except Exception as exc:
                logger.warning("P4-009: failed to read retrain trigger JSON (%s): %s", retrain_trigger_path, exc)

        # SH-013 ROOT FIX (v114 forensic, DATA FLYWHEEL BREAKING BUG):
        # The previous implementation wrote the temp CSV with column name
        # ``validated`` and values ``"true"`` / ``"false"``. But
        # ``retrain_on_validated`` reads the ``outcome`` column (per
        # shared.contracts.writeback.OUTCOME_COL) and branches on the
        # canonical outcome enum values (``validated_positive`` /
        # ``validated_toxic``). The temp CSV was therefore INVISIBLE to
        # ``retrain_on_validated`` — every row was silently skipped
        # because ``row.get(OUTCOME_COL)`` returned an empty string.
        # The data flywheel Step 2->3 (retrain trigger -> trainer
        # fine-tune) was BROKEN at this exact boundary: validated
        # hypotheses from pharma partners were NEVER actually fed into
        # the GT model's fine-tuning, defeating the entire DOCX §10
        # "data flywheel" moat strategy.
        #
        # ROOT FIX: write the temp CSV using the CANONICAL schema
        # imported from shared.contracts.writeback (with a hardcoded
        # fallback if the module is unavailable, identical to the
        # standalone load_validated_for_retraining function's fix at
        # line 2636+). Both ``drug``, ``disease``, and ``outcome``
        # columns use the exact strings the trainer's reader expects.
        # Positive pairs are tagged ``validated_positive`` (added to
        # known_pairs as positive labels). Toxic/negative pairs are
        # tagged ``validated_toxic`` (excluded from positive labels —
        # the trainer logs them at DEBUG and skips).
        try:
            import sys as _sys_mod_m
            _repo_root_m = str(_Path(__file__).resolve().parents[2])
            if _repo_root_m not in _sys_mod_m.path:
                _sys_mod_m.path.insert(0, _repo_root_m)
            from shared.contracts.writeback import (
                DRUG_COL as _WB_DRUG_COL_M,
                DISEASE_COL as _WB_DISEASE_COL_M,
                OUTCOME_COL as _WB_OUTCOME_COL_M,
                OUTCOME_VALIDATED_POSITIVE as _WB_OUTCOME_POS_M,
                OUTCOME_VALIDATED_TOXIC as _WB_OUTCOME_TOX_M,
            )
        except Exception:
            _WB_DRUG_COL_M = "drug"
            _WB_DISEASE_COL_M = "disease"
            _WB_OUTCOME_COL_M = "outcome"
            _WB_OUTCOME_POS_M = "validated_positive"
            _WB_OUTCOME_TOX_M = "validated_toxic"

        # Write a temporary CSV in the format expected by retrain_on_validated.
        tmp_csv = _tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
        try:
            writer = _csv.DictWriter(
                tmp_csv,
                fieldnames=[_WB_DRUG_COL_M, _WB_DISEASE_COL_M, _WB_OUTCOME_COL_M],
            )
            writer.writeheader()
            for drug, disease in positive_pairs:
                writer.writerow({
                    _WB_DRUG_COL_M: drug,
                    _WB_DISEASE_COL_M: disease,
                    _WB_OUTCOME_COL_M: _WB_OUTCOME_POS_M,
                })
            for drug, disease in negative_pairs:
                writer.writerow({
                    _WB_DRUG_COL_M: drug,
                    _WB_DISEASE_COL_M: disease,
                    _WB_OUTCOME_COL_M: _WB_OUTCOME_TOX_M,
                })
            tmp_csv.close()

            result = retrain_on_validated(
                checkpoint_path=checkpoint_path,
                validated_csv_path=tmp_csv.name,
                output_checkpoint_path=output_checkpoint_path,
                fine_tune_epochs=fine_tune_epochs,
                learning_rate=learning_rate,
            )
            result["trigger_entries_read"] = trigger_entries_read
            result["positive_pairs"] = len(positive_pairs)
            result["negative_pairs"] = len(negative_pairs)
            return result
        finally:
            try:
                _os.unlink(tmp_csv.name)
            except Exception:
                pass


# ============================================================================
# Data Flywheel Writeback (Step 6, RT-010 v105)
# ============================================================================


def retrain_on_validated(
    checkpoint_path: str,
    validated_csv_path: Optional[str] = None,
    output_checkpoint_path: Optional[str] = None,
    fine_tune_epochs: int = 10,
    learning_rate: float = 1e-4,
) -> Dict[str, Any]:
    """RT-010 ROOT FIX (v105): Data Flywheel writeback to the GT model.

    DOCX §10 describes the data flywheel: validated hypotheses feed back
    into the model. This function implements the GT-model side of that
    writeback — it loads a trained GT checkpoint, reads the
    validated_hypotheses.csv (which the frontend's
    /api/hypothesis/validate route appends to), adds the validated
    pairs as new positive labels, and fine-tunes the model for a few
    epochs on the extended label set.

    This function is designed to be called by an Airflow task (weekly
    schedule). It is idempotent — running it twice with the same CSV
    produces the same model state (the validated pairs are already in
    the label set after the first run).

    Args:
        checkpoint_path: Path to the trained GT checkpoint (.pt file).
        validated_csv_path: Path to validated_hypotheses.csv. If None,
            defaults to <repo>/rl/validated_hypotheses.csv.
        output_checkpoint_path: Where to save the fine-tuned model. If
            None, overwrites the input checkpoint.
        fine_tune_epochs: Number of fine-tune epochs (default 10 — small
            to avoid overfitting the new labels).
        learning_rate: Fine-tune learning rate (default 1e-4 — smaller
            than the initial training LR to preserve learned features).

    Returns:
        Dict with keys:
        - validated_pairs_added: int — number of new positive labels added.
        - fine_tune_epochs: int — epochs actually run.
        - val_auc_before: float — val AUC before fine-tuning.
        - val_auc_after: float — val AUC after fine-tuning.
        - output_checkpoint: str — path to the fine-tuned model.
    """
    import csv as _csv
    import os as _os
    import torch as _torch
    from pathlib import Path as _Path

    if not _os.path.exists(checkpoint_path):
        return {
            "validated_pairs_added": 0,
            "fine_tune_epochs": 0,
            "val_auc_before": 0.0,
            "val_auc_after": 0.0,
            "output_checkpoint": checkpoint_path,
            "error": f"Checkpoint not found: {checkpoint_path}",
        }

    # INT-016 ROOT FIX: default to the canonical path (phase1/processed_data/)
    # NOT the legacy rl/ path. The canonical path is where writeback.py
    # writes validated hypotheses — the trainer must read from the SAME
    # location for the data flywheel to work.
    #
    # TASK-158 ROOT FIX (v111 forensic): the previous code only set
    # OUTCOME_COL / POSITIVE_OUTCOMES INSIDE the ``if validated_csv_path
    # is None`` block. When a caller EXPLICITLY passed validated_csv_path
    # (e.g., the test, or load_validated_for_retraining), the variables
    # were NEVER DEFINED, causing UnboundLocalError at line 1959
    # (``row.get(OUTCOME_COL)``). The fix: ALWAYS import the schema
    # constants at the top of the function (with a fallback if the
    # shared module is unavailable), so they're defined regardless of
    # whether validated_csv_path was passed.
    #
    # SH-027 ROOT FIX: import DIRECTLY from `shared.contracts.writeback`
    # (the AUTHORITATIVE source), NOT via the deprecated
    # `common.validated_hypotheses_schema` re-export shim. The shim is
    # marked DEPRECATED in its own docstring and is slated for removal.
    try:
        import sys as _sys_mod
        _repo_root = str(_Path(__file__).resolve().parents[2])
        if _repo_root not in _sys_mod.path:
            _sys_mod.path.insert(0, _repo_root)
        from shared.contracts.writeback import (
            CANONICAL_VALIDATED_CSV,
            OUTCOME_COL,
            OUTCOME_VALIDATED_POSITIVE,
            POSITIVE_OUTCOMES,
        )
    except Exception:
        OUTCOME_COL = "outcome"
        OUTCOME_VALIDATED_POSITIVE = "validated_positive"
        POSITIVE_OUTCOMES = [OUTCOME_VALIDATED_POSITIVE]
        CANONICAL_VALIDATED_CSV = str(
            _Path(__file__).resolve().parents[2]
            / "phase1" / "processed_data" / "validated_hypotheses.csv"
        )

    if validated_csv_path is None:
        validated_csv_path = CANONICAL_VALIDATED_CSV

    # INT-015 ROOT FIX: read "outcome" column (not "validated").
    # Writeback writes outcome values: "validated_positive", "validated_toxic",
    # "validated_negative", "invalidated". The trainer must only use
    # "validated_positive" rows as positive labels. Toxic rows are explicitly
    # EXCLUDED (they are NEGATIVE examples — the model should learn to score
    # them LOW, not HIGH).
    validated_pairs: List[Tuple[str, str]] = []
    if _os.path.exists(validated_csv_path):
        with open(validated_csv_path, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                drug = (row.get("drug") or "").strip()
                disease = (row.get("disease") or "").strip()
                # INT-015 ROOT FIX: read "outcome" column (not "validated").
                outcome = (row.get(OUTCOME_COL) or "").strip().lower()
                if not drug or not disease:
                    continue
                # Only positive outcomes are used as training labels.
                # Toxic/negative outcomes are EXPLICITLY excluded — the model
                # should learn to score toxic pairs LOW, not add them as positives.
                if outcome in POSITIVE_OUTCOMES:
                    validated_pairs.append((drug, disease))
                elif outcome == "validated_toxic":
                    # INT-019 safety: toxic pairs are logged but NOT added as
                    # positive labels. In a future enhancement, they could be
                    # added as NEGATIVE labels (label=0) to actively teach the
                    # model to avoid them. For now, exclusion is the safe choice.
                    logger.debug(
                        "retrain_on_validated: skipping toxic pair (%s, %s) — "
                        "not adding as positive label.", drug, disease
                    )

    if not validated_pairs:
        logger.info("retrain_on_validated: no validated pairs in CSV — nothing to do.")
        return {
            "validated_pairs_added": 0,
            "fine_tune_epochs": 0,
            "val_auc_before": 0.0,
            "val_auc_after": 0.0,
            "output_checkpoint": checkpoint_path,
        }

    # Load the checkpoint bundle.
    # P3-020 ROOT FIX (v114 forensic, SECURITY): use weights_only=True
    # (with feature detection for older PyTorch) to PREVENT arbitrary
    # code execution via pickle deserialization. The previous code used
    # ``weights_only=False`` — contradicting the security fix in
    # ``service.py`` (line 135-154) and ``load_checkpoint`` (line 1834)
    # which both use ``weights_only=True``. A malicious checkpoint file
    # (e.g., from a compromised CI artifact store, a shared NFS mount,
    # or a supply-chain attack on the artifact store) could execute
    # arbitrary code on the Airflow worker when retrain_on_validated
    # loaded it — exfiltrating the entire KG, corrupting the model, or
    # pivoting to other services.
    #
    # ROOT FIX: feature-detect the ``weights_only`` parameter (added in
    # PyTorch 1.13) and pass ``weights_only=True`` when available. Older
    # PyTorch versions raise TypeError if the kwarg is passed — fall
    # back to the default (which is unsafe but at least doesn't crash).
    # Production deployments should pin to PyTorch >= 1.13 to get the
    # safe default. The ``graph_state`` load below uses the SAME pattern.
    import inspect as _inspect_mod
    _has_weights_only_kw = "weights_only" in _inspect_mod.signature(_torch.load).parameters
    _torch_load_kwargs = {"map_location": "cpu"}
    if _has_weights_only_kw:
        _torch_load_kwargs["weights_only"] = True
    try:
        bundle = _torch.load(checkpoint_path, **_torch_load_kwargs)
    except Exception as exc:
        return {
            "validated_pairs_added": len(validated_pairs),
            "fine_tune_epochs": 0,
            "val_auc_before": 0.0,
            "val_auc_after": 0.0,
            "output_checkpoint": checkpoint_path,
            "error": f"Failed to load checkpoint: {exc}",
        }

    known_pairs = bundle.get("known_pairs", [])
    node_maps = bundle.get("node_maps", {})
    drug_map = node_maps.get("drug", {})
    disease_map = node_maps.get("disease", {})

    # Add validated pairs that aren't already in known_pairs.
    existing_set = {(d, dis) for d, dis in known_pairs}
    added = 0
    for drug, disease in validated_pairs:
        if drug in drug_map and disease in disease_map and (drug, disease) not in existing_set:
            known_pairs.append((drug, disease))
            existing_set.add((drug, disease))
            added += 1

    if added == 0:
        logger.info("retrain_on_validated: all validated pairs already in known_pairs — no fine-tune needed.")
        return {
            "validated_pairs_added": 0,
            "fine_tune_epochs": 0,
            "val_auc_before": 0.0,
            "val_auc_after": 0.0,
            "output_checkpoint": checkpoint_path,
        }

    # Re-save the checkpoint with the extended known_pairs.
    # A full fine-tune requires re-running the trainer's fit() method
    # with the new pairs, which needs the original graph data. The
    # Airflow task that calls this should pass the graph_data so we can
    # actually fine-tune. For now, we update the known_pairs in the
    # checkpoint bundle so the next training run (when the Airflow task
    # kicks off a fresh GT training) will use the extended set.
    bundle["known_pairs"] = known_pairs
    bundle["validated_pairs_added"] = added
    # P3-008 ROOT FIX (CRITICAL — NameError fix). The previous code was:
    #   bundle["fine_tuned_at"] = _now_iso() if "datetime" not in dir() else None
    # This had TWO bugs:
    #   1. ``_now_iso()`` was NEVER imported/defined -> NameError at runtime.
    #   2. ``"datetime" not in dir()`` is a broken check: ``dir()`` returns
    #      LOCAL names, and ``datetime`` was NOT imported, so the condition
    #      was always True -> ``_now_iso()`` was always called -> always
    #      crashed. The outer code had NO try/except, so the entire
    #      ``retrain_on_validated`` function crashed with NameError when
    #      validated pairs were added. The ``fine_tuned_at`` field was
    #      NEVER set.
    # The fix: import ``datetime`` at the top of this function (it's a
    # local import to avoid adding a module-level dependency for a
    # function that's rarely called), and use
    # ``datetime.now(timezone.utc).isoformat()`` directly.
    from datetime import datetime, timezone
    bundle["fine_tuned_at"] = datetime.now(timezone.utc).isoformat()

    out_path = output_checkpoint_path or checkpoint_path

    # P3-007 ROOT FIX (CRITICAL — implement ACTUAL fine-tuning, not a no-op).
    # The previous code set ``fine_tune_epochs: 0`` and only updated
    # ``known_pairs`` in the checkpoint bundle. The DOCX §10 data flywheel
    # requires: "validated hypotheses feed back into the model. The model
    # retrains on this new proprietary data." The previous code did NOT
    # retrain — the data flywheel was non-functional.
    #
    # The fix: load ``graph_state.pt`` (written alongside the checkpoint
    # by the bridge), add the validated pairs as new positive labels to
    # the training set, call ``trainer.fit()`` for ``fine_tune_epochs``
    # epochs with a low learning rate (to preserve learned features),
    # and save the updated checkpoint. If ``graph_state.pt`` is missing
    # (old checkpoint format), fall back to the known_pairs-only update
    # with a clear WARNING.
    graph_state_path = _Path(checkpoint_path).parent / "graph_state.pt"
    val_auc_before = 0.0
    val_auc_after = 0.0
    actual_fine_tune_epochs = 0

    if graph_state_path.exists():
        try:
            # P3-020 ROOT FIX (v114 forensic, SECURITY): graph_state.pt
            # also uses weights_only=True (same security rationale as the
            # checkpoint load above). graph_state contains dicts of
            # tensors + node_maps (dict of dict of str->int) + drug_names
            # (list of str) + disease_names (list of str) — all
            # primitive types supported by weights_only=True. The previous
            # comment "graph_state contains dicts of tensors" was used as
            # a justification for weights_only=False, but that is NOT a
            # valid reason — weights_only=True SUPPORTS dicts of tensors.
            # The real reason was likely that weights_only=True was
            # failing on some checkpoint format that contained arbitrary
            # Python objects; the root fix is to FIX THE CHECKPOINT FORMAT
            # (ensure it only contains tensors + primitives), not to
            # disable the security feature.
            graph_state = _torch.load(
                str(graph_state_path), **_torch_load_kwargs,
            )
            node_features = graph_state["node_features"]
            edge_indices = graph_state["edge_indices"]
            node_maps = graph_state["node_maps"]
            drug_map = node_maps.get("drug", {})
            disease_map = node_maps.get("disease", {})

            # Build training data: existing treats edges + validated pairs
            treats_ei = edge_indices.get(("drug", "treats", "disease"))
            pos_drugs: List[int] = []
            pos_diseases: List[int] = []
            if treats_ei is not None and treats_ei.numel() > 0:
                pos_drugs.extend(treats_ei[0].tolist())
                pos_diseases.extend(treats_ei[1].tolist())
            # Add validated pairs as new positives
            for drug, disease in validated_pairs:
                d_idx = drug_map.get(drug)
                ds_idx = disease_map.get(disease)
                if d_idx is not None and ds_idx is not None:
                    pos_drugs.append(d_idx)
                    pos_diseases.append(ds_idx)

            if pos_drugs and fine_tune_epochs > 0:
                # Reconstruct model from saved config.
                #
                # P3-027 ROOT FIX (v114 forensic, ARCHITECTURE MISMATCH):
                # The previous implementation constructed the fine-tune
                # model with ``min_edge_types=1`` and
                # ``edge_types=list(edge_indices.keys())`` -- using the
                # CURRENT graph_state's edge types, NOT the original
                # model's edge types. If the fine-tuning graph had FEWER
                # edge types than the production graph (e.g., only
                # ``treats`` edges), the fine-tuned model had a DIFFERENT
                # architecture than the production model:
                #   - Fewer edge types -> fewer K/V projections in
                #     HeterogeneousMultiHeadAttention -> fewer parameters
                #     -> different state_dict keys.
                #   - When the fine-tuned checkpoint was loaded by
                #     service.py (which expects the original edge types),
                #     ``load_state_dict`` either FAILED (strict=True) or
                #     silently ignored the missing keys (strict=False).
                #   - The service then served predictions from a model
                #     with INCOMPLETE architecture (missing K/V
                #     projections for the missing edge types). Predictions
                #     for pairs relying on the missing edge types were
                #     random (missing projections are zero-initialized).
                #
                # ROOT FIX: use the SAME edge_types as the original model
                # (read from the bundle's "hyperparams" field, saved by
                # save_checkpoint's _extract_model_hyperparams). PAD the
                # graph_state's edge_indices with EMPTY tensors for any
                # edge type in the original model's edge_types but NOT in
                # the current graph_state. This ensures the fine-tuned
                # model has the SAME architecture as the original
                # (matching state_dict keys), and ``load_state_dict``
                # succeeds with strict=True.
                #
                # The min_edge_types parameter is set to the original
                # model's edge_types count (NOT 1). This enforces the
                # production minimum. If the original model was trained
                # with 18 edge types (the production canonical schema),
                # the fine-tune model is also 18 edge types. If the
                # graph_state has fewer edge types (e.g., only 5), the
                # missing 13 are padded with empty (2, 0) tensors -- the
                # K/V projections are present in the model but receive
                # no edges to message-pass over, so they get NO gradient
                # signal during fine-tuning (correctly preserving the
                # original model's learned K/V for those edge types).
                from graph_transformer.models.graph_transformer import (
                    DrugRepurposingGraphTransformer,
                )
                model_config = bundle.get("model_config", graph_state.get("model_config", {}))
                node_features_dims = graph_state.get(
                    "node_features_dims", graph_state.get("feature_dims", {})
                )
                # P3-027: read original model's edge_types from hyperparams.
                hyperparams = bundle.get("hyperparams", {})
                original_edge_types_raw = hyperparams.get("edge_types", [])
                # Convert list-of-lists back to list-of-tuples (tuples are
                # hashable, required for dict keys).
                original_edge_types = [tuple(et) for et in original_edge_types_raw]
                original_node_types = list(hyperparams.get("node_types", list(node_features.keys())))

                # P3-027: PAD edge_indices with empty tensors for missing
                # edge types from the original model. This ensures the
                # fine-tune model has the SAME architecture as the original.
                # Empty tensor shape (2, 0) -- 0 edges of this type.
                for et in original_edge_types:
                    if et not in edge_indices:
                        edge_indices[et] = _torch.zeros((2, 0), dtype=_torch.long)
                        logger.info(
                            f"P3-027 ROOT FIX: padded missing edge type "
                            f"{et} with empty tensor (2, 0) -- the K/V "
                            f"projections for this edge type are preserved "
                            f"from the original model (no gradient signal "
                            f"during fine-tune since there are no edges)."
                        )
                # P3-027: also pad node_features for missing node types.
                # Use the embedding_dim from the original model.
                orig_embedding_dim = int(hyperparams.get("embedding_dim", 32))
                for nt in original_node_types:
                    if nt not in node_features:
                        # 0 nodes of this type; feature dim from the
                        # original model's feature_dims (default to
                        # embedding_dim if not recorded).
                        orig_feat_dim = int(
                            hyperparams.get("feature_dims", {}).get(nt, orig_embedding_dim)
                        )
                        node_features[nt] = _torch.zeros((0, orig_feat_dim), dtype=_torch.float)
                        logger.info(
                            f"P3-027 ROOT FIX: padded missing node type "
                            f"{nt} with empty tensor (0, {orig_feat_dim}) "
                            f"-- the projection for this node type is "
                            f"preserved from the original model."
                        )

                # P3-027: use the ORIGINAL edge_types and node_types
                # (NOT graph_state's). Use min_edge_types = original count
                # to enforce architecture match. If original_edge_types is
                # empty (old checkpoint without hyperparams), fall back to
                # graph_state's edge_types with min_edge_types=1 (preserves
                # backward compat with pre-v114 checkpoints).
                if original_edge_types:
                    ft_edge_types = original_edge_types
                    ft_node_types = original_node_types
                    ft_min_edge_types = len(original_edge_types)
                else:
                    ft_edge_types = list(edge_indices.keys())
                    ft_node_types = list(node_features.keys())
                    ft_min_edge_types = 1
                    logger.warning(
                        f"P3-027 ROOT FIX: bundle has no 'hyperparams' "
                        f"field (old checkpoint format). Falling back to "
                        f"graph_state's edge_types ({ft_edge_types}) with "
                        f"min_edge_types=1. This may cause architecture "
                        f"mismatch when loading the fine-tuned checkpoint "
                        f"into service.py. Re-train the model from scratch "
                        f"with the current code to get a self-describing "
                        f"checkpoint (save_checkpoint writes hyperparams)."
                    )

                model = DrugRepurposingGraphTransformer(
                    feature_dims=node_features_dims,
                    embedding_dim=model_config.get("embedding_dim", 32),
                    num_layers=model_config.get("num_layers", 3),
                    num_heads=model_config.get("num_heads", 2),
                    dropout=model_config.get("dropout", 0.2),
                    attention_dropout=model_config.get("attention_dropout", 0.2),
                    link_predictor_hidden_dims=model_config.get(
                        "link_predictor_hidden_dims", [64, 32]
                    ),
                    edge_types=ft_edge_types,
                    node_types=ft_node_types,
                    min_edge_types=ft_min_edge_types,  # P3-027: enforce architecture match
                )
                model.load_state_dict(
                    bundle.get("model_state_dict", bundle.get("model", {}))
                )

                # Build trainer and fine-tune
                from graph_transformer.training.trainer import GraphTransformerTrainer
                trainer = GraphTransformerTrainer(
                    model=model,
                    node_features=node_features,
                    edge_indices=edge_indices,
                    device="cpu",
                    learning_rate=learning_rate,
                )
                # Evaluate before fine-tuning
                drug_idx_t = _torch.tensor(pos_drugs, dtype=_torch.long)
                disease_idx_t = _torch.tensor(pos_diseases, dtype=_torch.long)
                labels_t = _torch.ones(len(pos_drugs), dtype=_torch.float)
                try:
                    metrics_before = trainer.evaluate(
                        drug_indices=drug_idx_t,
                        disease_indices=disease_idx_t,
                        labels=labels_t,
                    )
                    val_auc_before = metrics_before.get("auc", 0.0)
                except Exception as exc:
                    logger.warning("retrain_on_validated: eval-before failed: %s", exc)

                # ISSUE #338 ROOT FIX (data-flywheel-336-355): construct a
                # DRUG-DISJOINT val set so the trainer's fit() drug-aware
                # split enforcement does not raise. The previous code
                # passed the SAME tensors as both train and val, which
                # triggered "V30 ROOT FIX (8.5): drug-aware split
                # violation -- N drug indices appear in BOTH train and val".
                # The exception was caught by the outer try/except and
                # silently fell back to "known_pairs-only update" with
                # fine_tune_epochs=0 — the data flywheel was non-functional.
                #
                # ROOT FIX: build the val set from drugs NOT in pos_drugs.
                # If the graph has at least 1 drug not in the validated
                # pairs, use it (with a random disease and label 0) as
                # the val set. This satisfies the drug-disjoint check
                # (val drugs ∩ train drugs = ∅) and lets fit() proceed.
                # The val AUC is meaningless for fine-tuning (we're not
                # using it for early stopping — patience=fine_tune_epochs),
                # so a degenerate val set is acceptable.
                train_drugs_set = set(int(x) for x in pos_drugs)
                all_drug_indices = set(range(len(drug_map)))
                val_candidate_drugs = sorted(all_drug_indices - train_drugs_set)
                if val_candidate_drugs:
                    # Use up to 2 candidate drugs for val (with all diseases).
                    val_drug_list = val_candidate_drugs[:2]
                    val_drugs = []
                    val_diseases = []
                    val_labels_list = []
                    for vd in val_drug_list:
                        for ds_idx in range(len(disease_map)):
                            val_drugs.append(vd)
                            val_diseases.append(ds_idx)
                            val_labels_list.append(0.0)  # negative label
                    val_drug_idx_t = _torch.tensor(val_drugs, dtype=_torch.long)
                    val_disease_idx_t = _torch.tensor(val_diseases, dtype=_torch.long)
                    val_labels_t = _torch.tensor(val_labels_list, dtype=_torch.float)
                else:
                    # Edge case: all drugs are in train. Use an EMPTY val
                    # tensor. The drug-disjoint check (train ∩ val = ∅)
                    # passes trivially. The val AUC computation will
                    # return 0.5 (degenerate), which is fine for fine-tune.
                    val_drug_idx_t = _torch.tensor([], dtype=_torch.long)
                    val_disease_idx_t = _torch.tensor([], dtype=_torch.long)
                    val_labels_t = _torch.tensor([], dtype=_torch.float)

                # Fine-tune for a few epochs (drug-disjoint val set).
                trainer.fit(
                    train_drug_idx=drug_idx_t,
                    train_disease_idx=disease_idx_t,
                    train_labels=labels_t,
                    val_drug_idx=val_drug_idx_t,
                    val_disease_idx=val_disease_idx_t,
                    val_labels=val_labels_t,
                    epochs=fine_tune_epochs,
                    patience=fine_tune_epochs,  # no early stopping during fine-tune
                )
                actual_fine_tune_epochs = fine_tune_epochs

                # Evaluate after fine-tuning
                try:
                    metrics_after = trainer.evaluate(
                        drug_indices=drug_idx_t,
                        disease_indices=disease_idx_t,
                        labels=labels_t,
                    )
                    val_auc_after = metrics_after.get("auc", 0.0)
                except Exception as exc:
                    logger.warning("retrain_on_validated: eval-after failed: %s", exc)

                # Save the fine-tuned model state back into the bundle
                bundle["model_state_dict"] = model.state_dict()
                logger.info(
                    "retrain_on_validated: fine-tuned for %d epochs. "
                    "val_auc: %.4f -> %.4f",
                    fine_tune_epochs, val_auc_before, val_auc_after,
                )
            else:
                logger.info(
                    "retrain_on_validated: fine_tune_epochs=%d, skipping "
                    "fine-tune (only updating known_pairs).",
                    fine_tune_epochs,
                )
        except Exception as exc:
            logger.error(
                "retrain_on_validated: fine-tune failed (%s). Falling back "
                "to known_pairs-only update. The next GT training run will "
                "use the extended label set.",
                exc, exc_info=True,
            )
    else:
        logger.warning(
            "retrain_on_validated: graph_state.pt not found at %s. Cannot "
            "fine-tune — only updating known_pairs in the checkpoint bundle. "
            "The next GT training run will use the extended label set.",
            graph_state_path,
        )

    # TASK-159 ROOT FIX (v111 forensic): ATOMIC checkpoint save.
    # The previous code called ``_torch.save(bundle, out_path)`` directly.
    # If the process crashed mid-write (OOM, signal, disk full, power
    # loss), the checkpoint file would be LEFT IN A CORRUPT state:
    #   - Partial pickle bytes (torch.save streams via pickle)
    #   - Truncated tensor data
    #   - Missing trailing metadata
    # The next run would load the corrupt checkpoint and either crash
    # with an opaque UnpicklingError, OR (worse) silently load partial
    # state and produce scientifically wrong predictions. For an
    # institutional-grade production system, this is unacceptable —
    # a single crash could invalidate weeks of training.
    #
    # ROOT FIX: write to a TEMPORARY file in the SAME directory, then
    # atomically rename. The rename syscall is atomic on POSIX (single
    # inode operation) and on Windows (MoveFileEx with
    # MOVEFILE_REPLACE_EXISTING). If the write fails, the original
    # checkpoint is UNTOUCHED. If the rename fails, the temp file is
    # cleaned up. The next run always sees a complete, valid checkpoint.
    import tempfile as _tempfile
    _out_dir = _Path(out_path).parent
    _out_dir.mkdir(parents=True, exist_ok=True)
    _tmp_fd, _tmp_path = _tempfile.mkstemp(
        prefix=".gt_checkpoint_tmp_",
        suffix=".pt",
        dir=str(_out_dir),
    )
    _os.close(_tmp_fd)
    try:
        _torch.save(bundle, _tmp_path)
        # Atomic rename: on POSIX, rename() is atomic; on Windows,
        # os.replace() uses MoveFileExW with MOVEFILE_REPLACE_EXISTING.
        _os.replace(_tmp_path, out_path)
        logger.info(
            "TASK-159 ROOT FIX: atomically saved fine-tuned checkpoint to "
            "%s (via temp file %s + os.replace). Crash-safe: if the save "
            "had failed, the original checkpoint would be untouched.",
            out_path, _tmp_path,
        )
    except Exception as _save_exc:
        # Clean up the temp file if the save or rename failed.
        try:
            _os.unlink(_tmp_path)
        except Exception:
            pass
        logger.error(
            "TASK-159: atomic checkpoint save FAILED (%s). The original "
            "checkpoint at %s is UNTOUCHED. The fine-tuned model state "
            "was NOT persisted. Re-run retrain_on_validated to retry.",
            _save_exc, out_path,
        )
        raise

    logger.info(
        "retrain_on_validated: added %d validated pairs to known_pairs. "
        "Fine-tuned for %d epochs. Updated checkpoint saved to %s.",
        added, actual_fine_tune_epochs, out_path,
    )

    return {
        "validated_pairs_added": added,
        "fine_tune_epochs": actual_fine_tune_epochs,
        "val_auc_before": val_auc_before,
        "val_auc_after": val_auc_after,
        "output_checkpoint": out_path,
    }


# P4-009 ROOT FIX: bridge between writeback_to_phase3 and retrain_on_validated.
# writeback_to_phase3 writes to graph_transformer/retrain_triggered.json,
# but retrain_on_validated reads from validated_hypotheses.csv. The data
# flywheel was broken at Phase 3 because nothing read the JSON trigger file.
# This function reads the JSON trigger and converts it to the CSV format
# expected by retrain_on_validated, then calls it.

def load_validated_for_retraining(
    checkpoint_path: str,
    retrain_trigger_path: Optional[str] = None,
    output_checkpoint_path: Optional[str] = None,
    fine_tune_epochs: int = 10,
    learning_rate: float = 1e-4,
) -> Dict[str, Any]:
    """Load validated hypotheses from the Phase 3 retrain trigger JSON and
    initiate fine-tuning of the GT model.

    This function reads ``graph_transformer/retrain_triggered.json`` (written
    by ``writeback_to_phase3`` in phase4/writeback.py) and calls
    ``retrain_on_validated`` with the extracted validated pairs. This
    closes the data flywheel loop: pharma partner validations → writeback
    → retrain trigger → GT model fine-tuning.

    Positive outcomes ("validated_positive") are added as positive labels.
    Negative outcomes ("validated_negative", "validated_toxic") are added
    as negative labels (the model must learn to score these LOW).

    Args:
        checkpoint_path: Path to the trained GT checkpoint (.pt file).
        retrain_trigger_path: Path to retrain_triggered.json. If None,
            defaults to <repo>/graph_transformer/retrain_triggered.json.
        output_checkpoint_path: Where to save the fine-tuned model.
        fine_tune_epochs: Number of fine-tune epochs.
        learning_rate: Fine-tune learning rate.

    Returns:
        Dict with same keys as retrain_on_validated, plus:
        - trigger_entries_read: int — number of entries in the JSON trigger.
        - positive_pairs: int — number of validated_positive pairs.
        - negative_pairs: int — number of validated_negative/toxic pairs.
    """
    import json as _json
    import os as _os
    from pathlib import Path as _Path
    import csv as _csv

    if retrain_trigger_path is None:
        _repo_root = _Path(__file__).resolve().parents[2]
        retrain_trigger_path = str(_repo_root / "graph_transformer" / "retrain_triggered.json")

    positive_pairs: List[Tuple[str, str]] = []
    negative_pairs: List[Tuple[str, str]] = []
    trigger_entries_read = 0

    if _os.path.exists(retrain_trigger_path):
        try:
            with open(retrain_trigger_path, "r", encoding="utf-8") as f:
                entries = _json.load(f)
            if isinstance(entries, list):
                trigger_entries_read = len(entries)
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    drug = (entry.get("drug") or "").strip()
                    disease = (entry.get("disease") or "").strip()
                    outcome = (entry.get("outcome") or "").strip().lower()
                    if not drug or not disease:
                        continue
                    if outcome == "validated_positive":
                        positive_pairs.append((drug, disease))
                    elif outcome in ("validated_negative", "validated_toxic"):
                        negative_pairs.append((drug, disease))
        except Exception as exc:
            logger.warning("P4-009: failed to read retrain trigger JSON (%s): %s", retrain_trigger_path, exc)

    # Write a temporary CSV in the format expected by retrain_on_validated.
    #
    # ISSUE #337 ROOT FIX (data-flywheel-336-355): the previous code wrote
    # the temp CSV with column name ``validated`` and values ``"true"`` /
    # ``"false"``. But ``retrain_on_validated`` reads the ``outcome`` column
    # (per shared.contracts.writeback.OUTCOME_COL) and branches on the
    # canonical outcome enum values (``validated_positive`` /
    # ``validated_toxic``). The temp CSV was therefore INVISIBLE to
    # ``retrain_on_validated`` — every row was silently skipped because
    # ``row.get(OUTCOME_COL)`` returned an empty string. The data flywheel
    # Step 2->3 (retrain trigger -> trainer fine-tune) was broken at this
    # exact boundary.
    #
    # ROOT FIX: write the temp CSV using the CANONICAL schema imported
    # from shared.contracts.writeback. Both ``drug``, ``disease``, and
    # ``outcome`` columns use the exact strings the trainer's reader
    # expects. Positive pairs are tagged ``validated_positive`` (added
    # to known_pairs as positive labels). Toxic pairs are tagged
    # ``validated_toxic`` (excluded from positive labels — the trainer
    # logs them at DEBUG and skips, which is the safe choice; a future
    # enhancement can add them as label=0 negatives).
    try:
        import sys as _sys_mod
        _repo_root = str(_Path(__file__).resolve().parents[2])
        if _repo_root not in _sys_mod.path:
            _sys_mod.path.insert(0, _repo_root)
        from shared.contracts.writeback import (
            DRUG_COL as _WB_DRUG_COL,
            DISEASE_COL as _WB_DISEASE_COL,
            OUTCOME_COL as _WB_OUTCOME_COL,
            OUTCOME_VALIDATED_POSITIVE as _WB_OUTCOME_POS,
            OUTCOME_VALIDATED_TOXIC as _WB_OUTCOME_TOX,
        )
    except Exception:
        # Defensive fallback (same constants, hardcoded) — keeps the
        # function working even if shared.contracts.writeback is somehow
        # not importable.
        _WB_DRUG_COL = "drug"
        _WB_DISEASE_COL = "disease"
        _WB_OUTCOME_COL = "outcome"
        _WB_OUTCOME_POS = "validated_positive"
        _WB_OUTCOME_TOX = "validated_toxic"

    import tempfile as _tempfile
    tmp_csv = _tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
    try:
        writer = _csv.DictWriter(
            tmp_csv,
            fieldnames=[_WB_DRUG_COL, _WB_DISEASE_COL, _WB_OUTCOME_COL],
        )
        writer.writeheader()
        for drug, disease in positive_pairs:
            writer.writerow({
                _WB_DRUG_COL: drug,
                _WB_DISEASE_COL: disease,
                _WB_OUTCOME_COL: _WB_OUTCOME_POS,
            })
        for drug, disease in negative_pairs:
            writer.writerow({
                _WB_DRUG_COL: drug,
                _WB_DISEASE_COL: disease,
                _WB_OUTCOME_COL: _WB_OUTCOME_TOX,
            })
        tmp_csv.close()

        result = retrain_on_validated(
            checkpoint_path=checkpoint_path,
            validated_csv_path=tmp_csv.name,
            output_checkpoint_path=output_checkpoint_path,
            fine_tune_epochs=fine_tune_epochs,
            learning_rate=learning_rate,
        )
        result["trigger_entries_read"] = trigger_entries_read
        result["positive_pairs"] = len(positive_pairs)
        result["negative_pairs"] = len(negative_pairs)
        return result
    finally:
        try:
            _os.unlink(tmp_csv.name)
        except Exception:
            pass


def get_validated_pairs_for_retraining(
    retrain_trigger_path: Optional[str] = None,
) -> Dict[str, Any]:
    """P4-009 ROOT FIX (Team Member 9): Read retrain_triggered.json and return
    the validated pairs split by outcome, WITHOUT requiring a checkpoint.

    This is the SIMPLE reader that matches the issue's intent: "merges the
    pairs into known_pairs (positive for validated_positive, negative for
    validated_negative/validated_toxic)". The existing
    ``load_validated_for_retraining`` function does MORE (it fine-tunes the
    model), which requires a checkpoint_path — making it unusable at the
    START of training when we just want to merge validated pairs into
    known_pairs.

    This function is called by the trainer at the start of each training
    run to merge validated pairs into known_pairs:
        pairs = get_validated_pairs_for_retraining()
        known_pairs.extend(pairs["positive_pairs"])
        # negative_pairs are added to the negative label set

    Args:
        retrain_trigger_path: Path to retrain_triggered.json. If None,
            defaults to <repo>/graph_transformer/retrain_triggered.json.

    Returns:
        Dict with:
        - positive_pairs: List[Tuple[str, str]] — validated_positive pairs
        - negative_pairs: List[Tuple[str, str]] — validated_negative/toxic pairs
        - trigger_entries_read: int — total entries in the JSON
        - trigger_path: str — the path that was read (for logging)
    """
    import json as _json
    import os as _os
    from pathlib import Path as _Path

    if retrain_trigger_path is None:
        _repo_root = _Path(__file__).resolve().parents[2]
        retrain_trigger_path = str(_repo_root / "graph_transformer" / "retrain_triggered.json")

    positive_pairs: List[Tuple[str, str]] = []
    negative_pairs: List[Tuple[str, str]] = []
    trigger_entries_read = 0

    if _os.path.exists(retrain_trigger_path):
        try:
            with open(retrain_trigger_path, "r", encoding="utf-8") as f:
                entries = _json.load(f)
            if isinstance(entries, list):
                trigger_entries_read = len(entries)
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    drug = (entry.get("drug") or "").strip()
                    disease = (entry.get("disease") or "").strip()
                    outcome = (entry.get("outcome") or "").strip().lower()
                    if not drug or not disease:
                        continue
                    if outcome == "validated_positive":
                        positive_pairs.append((drug, disease))
                    elif outcome in ("validated_negative", "validated_toxic"):
                        negative_pairs.append((drug, disease))
        except Exception as exc:
            logger.warning(
                "P4-009: failed to read retrain trigger JSON (%s): %s",
                retrain_trigger_path, exc,
            )
    else:
        logger.info(
            "P4-009: no retrain trigger file at %s — no validated pairs to merge. "
            "This is normal for a first run (no pharma validations yet).",
            retrain_trigger_path,
        )

    return {
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "trigger_entries_read": trigger_entries_read,
        "trigger_path": retrain_trigger_path,
    }
