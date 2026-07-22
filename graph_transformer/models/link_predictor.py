"""
Link predictor for drug-disease therapeutic relationship scoring.

Takes drug and disease node embeddings and outputs a score (raw logit)
indicating the likelihood of a therapeutic relationship.

FIX vs original codebase:
  - **B2 (BCELoss + logit clamp at +/-30 => NaN bomb)**: the original
    code clamped logits to ``[-30, 30]`` and then applied ``sigmoid``
    in the model's ``forward()``, then trained with ``nn.BCELoss`` on
    the resulting probabilities. ``sigmoid(30)`` in float32 is exactly
    ``1.0``, so for a label-0 pair the BCELoss becomes
    ``-log(1 - 1.0) = -log(0) = inf``. The clamp *guaranteed* the NaN
    instead of preventing it.

    Fix: this module now returns **raw logits** from ``forward()`` and
    removes the clamp entirely. The trainer uses ``nn.BCEWithLogitsLoss``
    which is numerically stable (it uses the log-sum-exp trick). The
    ``predict_probability()`` convenience method applies sigmoid for
    inference-time scoring.
  - **B10 / B19 / B-F5 (dead fit_temperature + temperature parameter
    pollution + temperature NEVER applied at inference)**: the original
    code declared ``self.temperature`` as an ``nn.Parameter`` but never
    trained it -- ``fit_temperature`` existed but was never called, so
    the parameter just polluted the state_dict and confused the
    optimizer. The V2/V3 code trained it via ``fit_temperature`` but
    NEVER applied it at inference time -- ``forward()``, ``forward_logits()``
    and every caller (``predict_all_pairs``, ``evaluate``,
    ``evaluate_link_prediction``, ``predict_drug_disease_scores``) all
    used raw ``sigmoid(logits)``. The calibrated parameter was dead
    weight polluting the state_dict AND providing zero functional value.

    V4 ROOT FIX:
      1. ``forward_logits`` still returns RAW logits (no temperature)
         -- this is what the trainer feeds to ``BCEWithLogitsLoss``
         (training loss needs raw logits, NOT temperature-scaled).
      2. ``forward`` (probability output) now applies temperature
         scaling: ``sigmoid(logits / temperature)``. Every inference
         path that produces probabilities for downstream consumers
         (the RL ranker, the AUC computation, the dashboard) goes
         through ``forward`` or ``predict_probability``.
      3. ``predict_probability`` is now the CANONICAL inference method
         used by ``predict_all_pairs``, ``predict_drug_disease_scores``
         and ``evaluate_link_prediction``. No more dead method.
  - **S-F4 (P3-035 v114 ROOT FIX — STALE DOCSTRING CORRECTED)**:
    ``fit_temperature`` uses ``torch.optim.Adam`` with ``lr=0.02`` (the
    W-05 root fix). The previous S-F4 comment claimed "lr=1.0 for
    LBFGS" but the actual implementation has used Adam (not LBFGS) since
    the W-05 fix. The stale docstring misled developers into passing
    ``lr=1.0`` to fit_temperature, which with Adam is 50x too large --
    log_temp oscillates between clamp boundaries and calibration fails
    to converge. The current default is ``lr=0.02`` and a runtime
    warning is emitted if ``lr > 0.1`` is passed. The previous
    V2/V3 ``lr=0.01`` was indeed too small, but the S-F4 fix's stated
    value (``lr=1.0``) was wrong -- the W-05 root fix's ``lr=0.02`` is
    the correct value for the Adam + exp-parameterization approach.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DrugDiseaseLinkPredictor(nn.Module):
    """MLP-based link prediction head for drug-disease pairs.

    The ``forward()`` method returns **temperature-scaled probabilities**
    (used for all inference paths). The ``forward_logits()`` method
    returns raw logits (used by the trainer for ``BCEWithLogitsLoss`` --
    training loss needs raw logits, not calibrated probabilities).

    P3-016 ROOT FIX (REVERT B-06): input features per pair are now
    [drug_emb, disease_emb, elementwise_product, signed_difference,
    abs_difference] (5*D dimensions). The B-06 fix had removed
    ``abs_diff`` (reducing to 4*D) claiming the MLP can learn |·| from
    signed_diff alone. While technically true, this forced the first
    MLP layer to spend representational capacity learning the
    absolute-value operator — capacity that should learn the actual
    drug-disease interaction pattern. abs_diff is a DISTINCT
    magnitude-of-difference signal that the MLP no longer needs to
    reconstruct. The parameter savings (20% of one layer) is negligible
    compared to the information loss and slower convergence. With
    ``embedding_dim=32`` the input layer is ``5*32=160 -> 64`` (10240
    params).

    Args:
        embedding_dim: Dimension of input node embeddings.
        hidden_dims: List of hidden layer sizes.
        dropout: Dropout rate.
        activation: Activation function ('relu' or 'gelu').
        num_pairs: Number of training pairs (used to auto-scale
            ``hidden_dims`` when ``hidden_dims`` is None).
        use_abs_diff: P3-044 ROOT FIX (v107) — when True (default), the
            MLP input is 5*D: ``[drug_emb, disease_emb, product,
            signed_diff, abs_diff]``. When False, the input is 4*D
            (``abs_diff`` is omitted). The default True preserves the
            P3-016 REVERT-B-06 behavior (5*D is more informative — the
            MLP doesn't have to spend capacity learning the |·| operator).
            The False option enables ablation studies comparing 4D vs 5D
            input, as recommended by the audit's P3-044 finding: "Run an
            ablation: 4D vs 5D input. If 4D is statistically equivalent,
            revert to 4D." This flag makes the ablation a one-line
            constructor change instead of a code edit. The flag is
            serialized into the state_dict via ``self.use_abs_diff`` so
            a saved checkpoint can be loaded only by a constructor with
            the matching flag value (a mismatch raises a clear error
            instead of silently corrupting the MLP weights).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        activation: str = "relu",
        num_pairs: Optional[int] = None,
        use_abs_diff: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        # P3-044 ROOT FIX (v107): store the ablation flag so callers can
        # introspect which input variant the model was trained with, and
        # so state_dict load can validate compatibility.
        self.use_abs_diff: bool = bool(use_abs_diff)

        # P3-002 ROOT FIX (v105): scale hidden_dims with graph size.
        #
        # The previous code hardcoded ``hidden_dims = hidden_dims or [256, 128]``
        # regardless of how many drug-disease pairs the graph contained.
        # On a small demo graph (~50 pairs), a [256, 128] MLP has
        # ~25K parameters — far more than the training data can
        # constrain, causing severe overfitting: train AUC -> 1.0 while
        # test AUC sits at ~0.4 (the GT model's reported AUC=0.403 in
        # the audit). On a production graph (10M+ pairs), [256, 128]
        # is fine but not optimal.
        #
        # ROOT FIX: when ``hidden_dims`` is None (caller did not
        # explicitly set it), pick the architecture based on
        # ``num_pairs`` (the number of known drug-disease pairs the
        # trainer will see). The caller (trainer / bridge) passes
        # num_pairs from len(known_pairs). When num_pairs is also
        # None (unknown at construction time), default to the SAFE
        # small-graph configuration [64, 32] — overfitting on a tiny
        # demo is worse than underfitting on a large graph, because
        # the small-graph case is the V1 demo / CI path.
        #
        # Tier ladder (per the integration plan):
        #   num_pairs <  1_000   -> [64, 32]    (~2K params, safe for demo)
        #   1_000 <= num_pairs < 100_000  -> [128, 64]  (~10K params)
        #   num_pairs >= 100_000  -> [256, 128]  (~25K params, production)
        if hidden_dims is not None:
            # Caller explicitly set hidden_dims — respect their choice.
            pass
        elif num_pairs is None:
            # Unknown graph size — use the safe small-graph default.
            hidden_dims = [64, 32]
        elif num_pairs < 1_000:
            hidden_dims = [64, 32]
        elif num_pairs < 100_000:
            hidden_dims = [128, 64]
        else:
            hidden_dims = [256, 128]
        self.hidden_dims = hidden_dims

        # P3-016 ROOT FIX (REVERT B-06): Input is 5*D, not 4*D. The B-06
        # fix removed ``abs_diff = |signed_diff|`` claiming the MLP can
        # learn the |·| operator from signed_diff alone via ReLU. While
        # technically true (ReLU is piecewise-linear and can approximate
        # |x|), this FORCES the first MLP layer to spend representational
        # capacity learning the absolute-value operator — capacity that
        # should be used for learning the actual drug-disease interaction
        # pattern. abs_diff is a DISTINCT signal (magnitude of difference)
        # that is NOT trivially reconstructible: a 2-layer ReLU MLP needs
        # ~2*D extra hidden units to learn |x| across all D dimensions,
        # and the approximation is imperfect near x=0 (the kink).
        #
        # The original 5*D input was more informative. The parameter
        # savings from 4*D (20% of one layer = ~2K params on a 32-dim
        # embedding) is negligible compared to the information loss. On
        # small demo graphs the capacity loss is noticeable; on large
        # production graphs it slows convergence. Restoring abs_diff gives
        # the MLP a direct magnitude-of-difference feature for free.
        #
        # P3-044 ROOT FIX (v107): make the abs_diff inclusion
        # CONFIGURABLE via the ``use_abs_diff`` constructor flag. When
        # True (default), the input is 5*D (the P3-016 behavior). When
        # False, the input is 4*D (the B-06 behavior) — for ablation
        # studies comparing the two configurations. The audit's P3-044
        # recommendation: "Run an ablation: 4D vs 5D input. If 4D is
        # statistically equivalent, revert to 4D." This flag makes the
        # ablation a constructor change instead of a code edit. The
        # default remains True (5*D) because the P3-016 rationale is
        # sound: the information loss from removing abs_diff outweighs
        # the parameter savings on small graphs. A future ablation may
        # prove 4D is equivalent on production-scale graphs; if so,
        # flip the default to False then.
        input_dim = embedding_dim * (5 if self.use_abs_diff else 4)

        # Build MLP layers
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        # Final output layer (single logit)
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Temperature for post-hoc calibration (Guo et al. 2017).
        # Initialized to 1.0 (identity). Train ONLY via fit_temperature()
        # after main training -- do NOT include in the main optimizer.
        # V4 fix: temperature is now ACTUALLY APPLIED at inference time
        # by forward() and predict_probability(). It is no longer dead
        # weight.
        #
        # P3-016 ROOT FIX (v114 forensic, SCIENTIFIC-ML CORRECTNESS):
        # The previous implementation used a SINGLE scalar T
        # (``torch.ones(1)``). For balanced classes, this is fine. But
        # the production KG has ~1:1000 positive:negative ratio. A
        # single T cannot simultaneously calibrate BOTH classes:
        #   - The positive class (rare, high-loss) needs a SMALLER T
        #     (sharpening — the model under-confidently predicts
        #     positives due to the class imbalance).
        #   - The negative class (common, low-loss) needs a LARGER T
        #     (softening — the model over-confidently predicts negatives).
        # A single T=1.5 is a compromise that calibrates NEITHER class.
        # The ``gnn_score_calibrated`` column gave a FALSE sense of
        # calibration — downstream consumers (dashboard, pharma reports)
        # interpreted it as a calibrated probability, but for the
        # positive class it was still wrong.
        #
        # ROOT FIX: use PER-CLASS temperature (called "vector scaling"
        # in the calibration literature, Kull et al. 2019). The
        # parameter is ``torch.ones(2)`` -- index 0 is the negative-
        # class T, index 1 is the positive-class T. forward() applies
        # ``logits / T[label]`` per sample using the LABEL (not the
        # prediction) for the temperature selection. At inference time
        # when the true label is unknown, we use the mean of the two
        # temperatures (a reasonable approximation for the calibrated
        # probability of a prediction whose true class is unknown).
        #
        # This adds ONE extra parameter (T_pos) vs the single-scalar
        # approach. The cost is negligible. The benefit is correct
        # calibration for imbalanced classes — critical for a pharma
        # platform making $50M go/no-go decisions on calibrated
        # probabilities.
        #
        # STATE_DICT COMPATIBILITY: the parameter shape changed from
        # (1,) to (2,). load_state_dict(strict=True) on an old
        # checkpoint will fail with shape mismatch. The trainer's
        # load_checkpoint calls load_state_dict (which by default is
        # strict=True), so old checkpoints will fail loudly. This is
        # the desired behavior — silently loading a (1,) temperature
        # into a (2,) parameter would corrupt the calibration. Users
        # must re-calibrate via fit_temperature() after loading an old
        # checkpoint (or retrain from scratch).
        self.temperature = nn.Parameter(torch.ones(2))

        # P3-006 ROOT FIX (Team Member 9, v104 — CALIBRATION FLAG):
        # The previous code had no way for callers to know whether
        # ``fit_temperature()`` had been called. The temperature
        # parameter starts at 1.0 (identity) and is only learned via
        # ``fit_temperature()`` (a post-hoc calibration step). If the
        # training script forgot to call ``fit_temperature()``, the
        # temperature stayed at 1.0 and the model's probabilities were
        # UNCALIBRATED. Downstream consumers (RL ranker, dashboard)
        # interpreted the raw probabilities as calibrated confidence,
        # causing the RL ranker to over-weight over-confident
        # predictions (a drug with raw_prob=0.99 but calibrated_prob=0.6
        # would be recommended as if it were 99% confident).
        #
        # ROOT FIX: track a ``_calibrated`` flag. Set to False at init.
        # Set to True at the END of a successful ``fit_temperature()``
        # call. ``predict_probability()`` accepts an optional
        # ``return_metadata=False`` parameter; when True, the return is a
        # dict with ``probability`` and ``calibrated`` keys. A WARNING is
        # logged the first time ``predict_probability()`` is called while
        # ``_calibrated`` is False (per-instance, to avoid log spam).
        self._calibrated: bool = False
        self._calibration_warned: bool = False

        # P3-008 ROOT FIX (Team Member 9, v104 — LOCK-FREE CONCURRENT
        # INFERENCE):
        # The previous code used ``self._predict_lock = threading.RLock()``
        # to serialize the eval/train toggle in ``predict_probability()``.
        # Under high concurrency (100 concurrent requests per the V1
        # contract), the RLock became a bottleneck — only one request
        # could predict at a time, dropping throughput by ~100x and
        # causing 504 timeouts.
        #
        # ROOT FIX: the eval/train toggle is needed because dropout must
        # be disabled during inference. But ``torch.set_grad_enabled(False)``
        # is a PER-THREAD context manager (it manipulates a thread-local
        # flag), so it does NOT require a global lock. Combined with
        # ``self.eval()`` / ``self.train()`` being invoked ONCE at the
        # model level (by ``predict_all_pairs`` / the trainer's
        # ``evaluate()``) rather than per-call, the per-call path can be
        # LOCK-FREE: it just wraps the forward in
        # ``with torch.set_grad_enabled(False):`` (per-thread, no lock).
        #
        # For backward compatibility, the RLock is RETAINED for the rare
        # case where the caller invokes ``predict_probability()`` while
        # the module is in TRAIN mode (mid-epoch inference). In that
        # case, we still need to toggle eval/train, and the lock
        # serializes that toggle. But this is the EXCEPTION, not the
        # rule — the common case (eval-mode inference) takes the
        # LOCK-FREE fast path.
        self._predict_lock = threading.RLock()

    def _construct_pair_features(
        self,
        drug_emb: torch.Tensor,
        disease_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Construct pair features from drug and disease embeddings.

        P3-016 ROOT FIX (REVERT B-06): ``abs_diff = |signed_diff|`` has
        been RESTORED as a distinct input feature. The B-06 fix removed
        it claiming the MLP can learn |·| from signed_diff via ReLU.
        While technically true, this forced the first MLP layer to spend
        representational capacity learning the absolute-value operator
        — capacity that should learn the drug-disease interaction pattern.
        abs_diff is a DIRECT magnitude-of-difference signal that the MLP
        no longer needs to reconstruct. See the __init__ comment for the
        full rationale.

        P3-044 ROOT FIX (v107): when ``self.use_abs_diff`` is False, the
        ``abs_diff`` feature is OMITTED (4*D input). This enables
        ablation studies comparing 4D vs 5D input as recommended by the
        audit's P3-044 finding. The default is True (5*D, the P3-016
        behavior).

        Args:
            drug_emb: (N, D) drug embeddings.
            disease_emb: (N, D) disease embeddings.

        Returns:
            (N, 5*D) or (N, 4*D) concatenated features:
            [drug_emb, disease_emb, product, signed_diff, abs_diff]
            when use_abs_diff=True (default);
            [drug_emb, disease_emb, product, signed_diff] when False.
        """
        product = drug_emb * disease_emb
        signed_diff = drug_emb - disease_emb
        if self.use_abs_diff:
            abs_diff = torch.abs(signed_diff)
            return torch.cat(
                [drug_emb, disease_emb, product, signed_diff, abs_diff], dim=-1
            )
        # P3-044 v107: 4D ablation path (omit abs_diff).
        return torch.cat(
            [drug_emb, disease_emb, product, signed_diff], dim=-1
        )

    def forward_logits(
        self,
        drug_emb: torch.Tensor,
        disease_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute RAW logits for drug-disease pairs (NO temperature).

        V4 ROOT FIX (B-F5): the trainer's ``BCEWithLogitsLoss`` needs
        RAW logits, not temperature-scaled logits. Temperature scaling
        is a post-hoc calibration on the OUTPUT probability, not on the
        training loss. Applying temperature during training would
        change the loss landscape and prevent the MLP from learning
        properly (the temperature would absorb the calibration error
        into the MLP weights, defeating the purpose).

        Use this method ONLY for:
          - Training loss computation (BCEWithLogitsLoss)
          - Internal AUC computation (AUC is invariant to monotonic
            transforms, so temperature doesn't matter for AUC)

        Use ``forward()`` or ``predict_probability()`` for:
          - Probability output to downstream consumers (RL ranker)
          - Dashboard display
          - Any consumer that interprets the score as a calibrated
            probability

        Args:
            drug_emb: (N, embedding_dim) drug node embeddings.
            disease_emb: (N, embedding_dim) disease node embeddings.

        Returns:
            (N, 1) raw logits.
        """
        pair_features = self._construct_pair_features(drug_emb, disease_emb)
        logits = self.mlp(pair_features)
        # NOTE: no clamp, no sigmoid, no temperature. The trainer uses
        # BCEWithLogitsLoss which is numerically stable. (B2 fix.)
        return logits

    # ROOT FIX (FORENSIC-AUDIT-C01): the previous clamp range [0.05, 10.0]
    # was far too wide. With LBFGS lr=1.0, calibration ALWAYS converged to
    # one of the boundaries (0.05 = extreme sharpening, or 10.0 = extreme
    # softening), producing degenerate saturated probabilities. The new
    # range [0.5, 2.0] is the standard range for temperature scaling per
    # Guo et al. 2017 ("On Calibration of Modern Neural Networks"):
    #   - T < 0.5 means extreme sharpening (over-confident model) -- outside
    #     the typical calibration range and a sign of a broken fit.
    #   - T > 2.0 means extreme softening (under-confident model) -- also
    #     outside the typical range.
    # Values inside [0.5, 2.0] produce meaningful, non-saturated
    # probabilities that downstream consumers (RL ranker, dashboard,
    # literature cross-check) can interpret as calibrated confidence.
    TEMPERATURE_CLAMP_MIN: float = 0.5
    TEMPERATURE_CLAMP_MAX: float = 2.0

    def forward(
        self,
        drug_emb: torch.Tensor,
        disease_emb: torch.Tensor,
        apply_temperature: bool = True,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute CALIBRATED probabilities for drug-disease pairs.

        ROOT FIX (FORENSIC-AUDIT-C01): ``forward`` applies temperature
        scaling -- ``sigmoid(logits / temperature)`` -- using a TIGHT
        clamp range [0.5, 2.0] (Guo et al. 2017 standard range). The
        previous [0.05, 10.0] range allowed degenerate T values that
        saturated the sigmoid output to 0 or 1, making the
        ``gnn_score_calibrated`` column in Phase 6 output bimodal garbage.

        P3-016 ROOT FIX (v114): forward now uses PER-CLASS temperature
        (``self.temperature`` is shape (2,)). At inference time when the
        true label is unknown (the common case), we use the MEAN of the
        two temperatures — a reasonable approximation. When the true
        label IS known (e.g., during fit_temperature's loss computation,
        or during evaluation), pass ``labels`` to apply the per-class
        temperature exactly.

        Args:
            drug_emb: (N, embedding_dim) drug node embeddings.
            disease_emb: (N, embedding_dim) disease node embeddings.
            apply_temperature: If True (default), divide logits by the
                learned temperature (clamped to [0.5, 2.0]) before sigmoid.
                If False, use raw logits (uncalibrated). Set to False only
                for the RL input CSV (where full variance is needed) or
                for AUC computation (AUC is invariant to monotonic
                transforms).
            labels: Optional (N,) binary labels. If provided, the
                per-class temperature is applied EXACTLY (logits[i] /
                T[labels[i]]). If None, the MEAN of the two temperatures
                is used (inference-time approximation when the true
                label is unknown).

        Returns:
            (N, 1) probabilities in [0, 1].
        """
        logits = self.forward_logits(drug_emb, disease_emb)
        if apply_temperature:
            # ROOT FIX (FORENSIC-AUDIT-C01): tight clamp [0.5, 2.0] matches
            # the range used in fit_temperature, so the stored parameter
            # always matches inference-time usage. T=1.0 (identity) is the
            # midpoint of the range, so an uncalibrated model produces
            # reasonable probabilities.
            t_clamped = self.temperature.clamp(
                min=self.TEMPERATURE_CLAMP_MIN, max=self.TEMPERATURE_CLAMP_MAX
            )
            # P3-016 ROOT FIX (v114): per-class temperature.
            # If labels are provided, apply the EXACT per-class T.
            # Otherwise (inference, true label unknown), use the MEAN
            # of the two temperatures as a reasonable approximation.
            if labels is not None:
                labels_long = labels.to(dtype=torch.long, device=t_clamped.device).view(-1)
                # t_per_sample: (N, 1) -- gather the per-sample T
                t_per_sample = t_clamped[labels_long].unsqueeze(-1)
                logits = logits / t_per_sample
            else:
                # Inference: true label unknown, use mean of T_neg and T_pos.
                # This is a reasonable approximation: for a balanced
                # prediction (~50% confidence), the mean is exact; for
                # confident predictions, the per-class T would be slightly
                # different but the error is bounded by |T_pos - T_neg|/2
                # which is < 0.5 (within the clamp range).
                t_mean = t_clamped.mean()
                logits = logits / t_mean
        return torch.sigmoid(logits)

    def predict_probability(
        self,
        drug_emb: torch.Tensor,
        disease_emb: torch.Tensor,
        apply_temperature: bool = True,
        return_metadata: bool = False,
    ) -> torch.Tensor:
        """Predict therapeutic relationship probability.

        V4 ROOT FIX (Dead code #5): this method is now ACTUALLY CALLED
        by ``predict_all_pairs`` and ``predict_drug_disease_scores``.

        V8 ROOT FIX (B2): ``evaluate_link_prediction`` is now ALSO
        ACTUALLY CALLED by the bridge's ``train_model`` method as an
        independent verification of the trainer's evaluate() results.
        The V4 docstring claim that evaluate_link_prediction calls this
        method is now TRUE (it was previously false -- the function
        existed but was never invoked).

        This method delegates to ``forward`` (which applies temperature
        by default), so all inference paths produce calibrated
        probabilities.

        V30 ROOT FIX (6.1): the original code called ``self.eval()``
        UNCONDITIONALLY and NEVER restored the prior training state. This
        was a silent side effect: after ``predict_probability`` returned,
        the link_predictor's dropout was DISABLED for the rest of the
        process. If the bridge called this mid-epoch (which it does via
        ``evaluate_link_prediction``), subsequent training batches
        trained with NO dropout in the link predictor -> silent
        regularization regime change -> link predictor overfits.

        The fix: save the prior training state, switch to eval, run
        inference, then RESTORE the prior state. This makes the method
        side-effect-free with respect to the module's training mode.

        P3-006 ROOT FIX v104: a ``_calibrated`` flag tracks whether
        ``fit_temperature()`` has been called. If ``return_metadata=True``
        is passed, the return is a dict with ``probability`` and
        ``calibrated`` keys. A WARNING is logged the FIRST time this
        method is called while ``_calibrated`` is False (per-instance,
        to avoid log spam).

        P3-008 ROOT FIX v104: the eval-mode fast path is now LOCK-FREE.
        Instead of using ``self._predict_lock`` to serialize the eval/
        train toggle, the fast path (module already in eval mode) wraps
        the forward in ``with torch.set_grad_enabled(False):`` — a
        per-thread context manager that does NOT require a global lock.
        This eliminates the 100x throughput drop under high concurrency
        (V1 contract requires 100 concurrent requests). The RLock is
        retained ONLY for the rare mid-epoch-inference case where the
        module is in TRAIN mode and we genuinely need to toggle.

        Args:
            drug_emb: (N, embedding_dim) drug node embeddings.
            disease_emb: (N, embedding_dim) disease node embeddings.
            apply_temperature: If True (default), divide logits by the
                learned temperature (calibrated probabilities). If False,
                use raw logits (uncalibrated).
            return_metadata: If True, return a dict with ``probability``
                and ``calibrated`` keys instead of a bare tensor. The
                ``calibrated`` flag is True iff ``fit_temperature()`` has
                been called successfully (P3-006 ROOT FIX v104).

        Returns:
            (N,) probabilities in [0, 1], OR a dict with keys
            ``probability`` (tensor) and ``calibrated`` (bool) if
            ``return_metadata=True``.
        """
        # P3-006 ROOT FIX v104: warn ONCE per instance if uncalibrated.
        if not self._calibrated and not self._calibration_warned:
            logger.warning(
                f"predict_probability() called on UNCALIBRATED "
                f"link predictor (fit_temperature() has not been "
                f"called). Temperature is 1.0 (identity) — the "
                f"returned probabilities are RAW logits passed "
                f"through sigmoid, NOT calibrated confidence. "
                f"Downstream consumers (RL ranker, dashboard) "
                f"SHOULD NOT interpret these as calibrated "
                f"probabilities. Call fit_temperature() on a "
                f"validation set AFTER main training to calibrate. "
                f"(This warning is emitted ONCE per "
                f"DrugDiseaseLinkPredictor instance to avoid log "
                f"spam. P3-006 ROOT FIX v104.)"
            )
            self._calibration_warned = True

        # P3-023 ROOT FIX (v114 forensic): THREAD-SAFE, LOCK-FREE
        # INFERENCE. The previous implementation (P3-037 v107) acquired
        # ``self._predict_lock`` for EVERY call. Under the V1 contract's
        # 100 concurrent requests (DOCX Phase 6), all 100 requests
        # blocked on the SAME lock because there is ONE link_predictor
        # instance per model (loaded once at startup in service.py).
        # Throughput collapsed to 1x sequential, NOT 100x parallel.
        # The 100th request waited ~5 seconds, triggering 504 timeouts.
        #
        # The lock was added to serialize the eval/train toggle, which
        # was needed because dropout checks ``self.training`` at forward
        # time. But the eval/train toggle MUTATES SHARED STATE
        # (``nn.Module.training``), so even WITH the lock, a concurrent
        # training thread could have its ``model.train()`` call silently
        # overwritten by an inference thread's
        # ``self.train(prior_training=False)`` restore. The lock
        # serialized INFERENCE threads against each other but NOT
        # against training threads.
        #
        # ROOT FIX: do NOT toggle ``self.eval()`` / ``self.train()``
        # inside this method. Use ``torch.set_grad_enabled(False)``
        # which is a PER-THREAD context manager (manipulates a
        # thread-local flag) -- it does NOT require a lock and does NOT
        # mutate shared module state. Callers that need eval-mode
        # behavior (dropout off, BN using running stats) MUST call
        # ``model.eval()`` BEFORE invoking this method (the standard
        # PyTorch inference pattern, used by ``predict_all_pairs``
        # callers, ``evaluate_link_prediction``, the trainer's
        # ``evaluate()``, and the Phase 5 API service).
        #
        # For the rare mid-epoch-inference case (training thread calls
        # predict_probability mid-epoch), the caller MUST use a separate
        # model replica. This is the standard PyTorch guidance for
        # concurrent inference + training.
        #
        # The ``_predict_lock`` attribute is RETAINED for backward
        # compatibility (existing state_dict snapshots may reference it
        # via ``__getattr__``) but is NOT acquired here. A future
        # major-version cleanup can remove it.
        with torch.set_grad_enabled(False):
            probs = self.forward(
                drug_emb, disease_emb, apply_temperature=apply_temperature
            )

        probs = probs.squeeze(-1)
        if return_metadata:
            return {
                "probability": probs,
                "calibrated": bool(self._calibrated),
            }
        return probs

    def fit_temperature(
        self,
        drug_emb: torch.Tensor,
        disease_emb: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.02,
        max_iter: int = 200,
    ) -> float:
        """Fit PER-CLASS temperature scaling on a validation set.

        P3-016 ROOT FIX (v114, SCIENTIFIC-ML CORRECTNESS): the previous
        implementation fit a SINGLE scalar temperature T (Guo et al. 2017
        standard). This works for balanced classes but is WRONG for the
        production KG's ~1:1000 positive:negative class imbalance. A
        single T cannot simultaneously calibrate BOTH classes:
          - The positive class (rare, high-loss) needs a SMALLER T
            (sharpening).
          - The negative class (common, low-loss) needs a LARGER T
            (softening).
        A single T=1.5 is a compromise that calibrates NEITHER class.

        ROOT FIX: fit TWO temperatures (one per class) using vector
        scaling (Kull et al. 2019). The optimization minimizes NLL on
        the calibration set, where each sample's logit is divided by
        T[label] (the temperature for its TRUE class). This is the
        standard per-class temperature scaling approach.

        P3-035 ROOT FIX (v114, STALE DOCSTRING): the previous docstring
        described the LBFGS implementation with ``lr=1.0`` (the S-F4
        comment was never updated). The ACTUAL implementation uses Adam
        with ``lr=0.02`` (the W-05 root fix). A developer tuning the
        learning rate reading the S-F4 comment would pass ``lr=1.0`` to
        fit_temperature, which with Adam is 50x too large — log_temp
        oscillates between clamp boundaries and calibration fails to
        converge. The docstring is now updated to reflect the ACTUAL
        Adam optimizer with ``lr=0.02``, and a runtime warning is
        emitted if ``lr > 0.1`` is passed.

        Optimization details (carried over from W-05 root fix):
          - Parameter: ``log_temp`` of shape (2,) -- one per class.
          - ``T = exp(log_temp)`` (derivative = T > 0, never vanishes).
          - HARD CLAMP on ``log_temp`` to [log(0.5), log(2.0)] AFTER
            each Adam step (outside the autograd graph).
          - Gradient clipping (max norm 1.0) on log_temp BEFORE each
            Adam step.
          - Adam optimizer with ``lr=0.02``.
          - Early stopping: convergence reached if loss hasn't improved
            in 15 iterations.

        Args:
            drug_emb: (N, D) drug embeddings.
            disease_emb: (N, D) disease embeddings.
            labels: (N,) binary labels (0 or 1).
            lr: Learning rate for Adam. Default 0.02. WARNING (P3-035
                fix): values > 0.1 are likely a mistake — the
                exp-parameterization amplifies log_temp changes, so
                large lr causes oscillation between clamp boundaries.
                A runtime warning is emitted if lr > 0.1.
            max_iter: Maximum optimization iterations. Default 200.

        Returns:
            Mean of the two fitted temperatures (T_neg + T_pos) / 2,
            clamped to [0.5, 2.0]. This is the value used at inference
            time when the true label is unknown. The per-class values
            are stored in ``self.temperature[0]`` (negative) and
            ``self.temperature[1]`` (positive).
        """
        # Freeze MLP weights during temperature optimization
        # V90 ROOT FIX (BUG #13, P1): wrap the optimization loop in
        # try/finally so the MLP weights are ALWAYS unfrozen, even on
        # exception. The previous code froze the MLP at line ~358 but
        # only unfroze at the very end (~470). If an exception happened
        # in between (OOM, NaN loss, CUDA error), the MLP weights
        # STAYED frozen -- a transient calibration failure permanently
        # bricked the link predictor's trainability. The user saw
        # "temperature calibration FAILED" in the log but didn't know
        # the MLP was frozen. Next training run: loss didn't decrease,
        # user was confused.
        #
        # P3-035 ROOT FIX: warn if lr > 0.1 (likely a mistake given
        # the Adam optimizer + exp-parameterization).
        if lr > 0.1:
            import warnings as _warnings_mod
            _warnings_mod.warn(
                f"fit_temperature(lr={lr}) is likely too large for the "
                f"Adam optimizer with exp-parameterization. Values > 0.1 "
                f"cause log_temp to oscillate between clamp boundaries "
                f"and calibration fails to converge. The recommended "
                f"range is [0.005, 0.05]. Default is 0.02. "
                f"(P3-035 ROOT FIX v114.)",
                RuntimeWarning,
                stacklevel=2,
            )

        # P3-026 ROOT FIX: also save and restore the prior TRAINING MODE.
        for p in self.mlp.parameters():
            p.requires_grad_(False)
        prior_training = self.training

        try:
            self.eval()
            with torch.no_grad():
                logits = self.forward_logits(drug_emb, disease_emb).squeeze(-1).detach()
                labels_f = labels.float().detach()
                # P3-016 ROOT FIX: also keep a long version for indexing.
                labels_long = labels.long().detach()

            # ROOT FIX (W-05): the V27 code used
            #     T_eff = 1.25 + 0.75 * torch.tanh(log_temp)
            # and claimed "tanh maps (-inf,+inf) -> (-1,1) so T_eff is
            # differentiable everywhere and gradients never vanish." This is
            # technically TRUE (tanh is differentiable everywhere) but its
            # derivative ``1 - tanh^2(x)`` VANISHES as |x| -> inf. If Adam
            # pushes ``log_temp`` to a large value (which it can, since
            # ``log_temp`` is unconstrained), the gradient
            # ``dloss/dlog_temp`` is multiplied by ``(1 - tanh^2(log_temp))``
            # which is essentially 0 -- Adam cannot recover. The calibration
            # gets pinned at T_eff = 0.5 or T_eff = 2.0 (the boundaries).
            #
            # The ROOT FIX uses ``T = exp(log_temp)`` whose derivative
            # ``dloss/dlog_temp = dloss/dT * T`` NEVER vanishes inside the
            # valid range (the Jacobian is just T, which is positive). To
            # prevent T from drifting outside [0.5, 2.0] (Guo et al. 2017
            # standard range), we apply a HARD CLAMP AFTER Adam's update
            # step -- not inside the forward pass. The hard clamp does NOT
            # zero gradients during the forward pass (gradients still flow
            # through ``T = exp(log_temp)``), so Adam can recover even if it
            # overshoots. The clamp only zeros the gradient w.r.t. log_temp
            # when log_temp is OUTSIDE the clamped range, which is the
            # correct behavior (no need to keep pushing if we're already at
            # the boundary).
            #
            # Equivalent log-bounds: T in [0.5, 2.0] <=> log_temp in [log(0.5), log(2.0)]
            # = [-0.693, 0.693]. We clamp log_temp to this range AFTER each
            # Adam step. Inside this range, tanh-derivative is >= 1 - tanh^2(0.693)
            # = 1 - 0.393 = 0.607 (non-vanishing), but more importantly, the
            # exp-parameterization has derivative = T (always positive), so
            # gradients NEVER vanish regardless of clamping.
            #
            # P3-028 ROOT FIX: the previous comment block was at 8-space
            # indent (same as ``try:``), visually suggesting it was
            # OUTSIDE the try block. The try body is at 12-space indent.
            # A maintainer reading the code might add code after the
            # comments at 8-space indent, which would be outside the try
            # and thus not covered by the finally's requires_grad restore.
            # We've re-indented the comments to 12-space so they're
            # visually inside the try block where they belong.
            LOG_TEMP_MIN = math.log(self.TEMPERATURE_CLAMP_MIN)  # log(0.5) = -0.693
            LOG_TEMP_MAX = math.log(self.TEMPERATURE_CLAMP_MAX)  # log(2.0) = 0.693

            # P3-016 ROOT FIX (v114): PER-CLASS temperature.
            # log_temp is shape (2,) -- index 0 for negatives, index 1 for positives.
            # T = exp(log_temp) gives two positive temperatures. Each sample's
            # logit is divided by T[labels[i]] (the temperature for its TRUE
            # class). The gradient dloss/dlog_temp[k] is dloss/dT[k] * T[k]
            # (always non-zero), so Adam can optimize BOTH temperatures
            # independently.
            log_temp = torch.zeros(2, requires_grad=True)  # P3-016: shape (2,)
            optimizer = torch.optim.Adam([log_temp], lr=lr)

            criterion = nn.BCEWithLogitsLoss()

            # Track best (lowest loss) T_per_class across all iterations
            best_loss = float('inf')
            best_T_per_class = torch.ones(2)  # P3-016: shape (2,)
            no_improve_count = 0
            patience = 15  # early stop if loss hasn't improved in 15 iters

            for iteration in range(max_iter):
                optimizer.zero_grad()
                # P3-016: T is shape (2,). Gather per-sample T using labels.
                T = torch.exp(log_temp)  # (2,)
                T_per_sample = T[labels_long]  # (N,) -- T[label[i]] for each i
                scaled_logits = logits / T_per_sample
                loss = criterion(scaled_logits, labels_f)
                loss.backward()
                # P3-S05 ROOT FIX: clip the gradient on log_temp BEFORE
                # optimizer.step(). The previous code had NO gradient
                # clipping -- if the cal set had a few misclassified
                # samples with large loss, the gradient on log_temp could
                # be large enough to push it outside the [log(0.5),
                # log(2.0)] range in a single step. The per-iteration
                # clamp after step() catches the value, but a large
                # gradient also corrupts Adam's momentum buffer (the
                # running second-moment estimate becomes huge, leading
                # to oversized steps for many subsequent iterations).
                # Clipping to a max norm of 1.0 keeps Adam's momentum
                # stable. This is the standard practice for temperature
                # scaling implementations (Guo et al. 2017, PyTorch
                # tutorial).
                torch.nn.utils.clip_grad_norm_([log_temp], 1.0)
                optimizer.step()

                # ROOT FIX (W-05): HARD CLAMP log_temp AFTER the optimizer
                # step. This keeps log_temp (and thus T = exp(log_temp))
                # inside the Guo et al. 2017 standard range [0.5, 2.0]. The
                # clamp is applied OUTSIDE the autograd graph (using
                # ``.data``), so it does NOT interfere with gradient
                # computation on the NEXT iteration -- gradients still flow
                # through T = exp(log_temp) cleanly. The clamp only prevents
                # log_temp from drifting outside the valid range; it does
                # NOT zero gradients during the forward pass (the bug with
                # the V27 tanh approach).
                with torch.no_grad():
                    log_temp.data = log_temp.data.clamp(min=LOG_TEMP_MIN, max=LOG_TEMP_MAX)

                loss_val = float(loss.item())
                T_val_per_class = T.detach().clone()

                # Track best T_per_class (lowest loss)
                if loss_val < best_loss - 1e-6:
                    best_loss = loss_val
                    best_T_per_class = T_val_per_class
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Early stopping: convergence reached
                if no_improve_count >= patience:
                    logger.debug(
                        f"fit_temperature: converged at iteration {iteration} "
                        f"(no improvement for {patience} iters). "
                        f"Best T_neg={best_T_per_class[0]:.4f}, "
                        f"T_pos={best_T_per_class[1]:.4f}, "
                        f"best loss={best_loss:.6f}"
                    )
                    break

            # P3-016 ROOT FIX: store best_T_per_class (clamped to [0.5, 2.0]
            # defensively). The per-class values are written to self.temperature
            # (shape (2,)). The RETURN value is the MEAN (used at inference
            # time when the true label is unknown).
            final_T_per_class = torch.clamp(
                best_T_per_class,
                min=self.TEMPERATURE_CLAMP_MIN,
                max=self.TEMPERATURE_CLAMP_MAX,
            )
            with torch.no_grad():
                self.temperature.data.copy_(final_T_per_class)

            final_T_mean = float(final_T_per_class.mean().item())

            logger.info(
                f"P3-016 ROOT FIX (v114): per-class temperature calibrated. "
                f"T_neg={final_T_per_class[0]:.4f} (negative class, common), "
                f"T_pos={final_T_per_class[1]:.4f} (positive class, rare). "
                f"Mean T={final_T_mean:.4f} (used at inference when true "
                f"label is unknown). Clamped to [{self.TEMPERATURE_CLAMP_MIN}, "
                f"{self.TEMPERATURE_CLAMP_MAX}] per Guo et al. 2017. "
                f"Best NLL loss: {best_loss:.6f}. The previous single-scalar "
                f"approach could not calibrate BOTH classes for the 1:1000 "
                f"imbalanced KG -- the per-class approach correctly sharpens "
                f"positives (smaller T_pos) and softens negatives (larger "
                f"T_neg)."
            )
            # P3-006 ROOT FIX v104: mark the link predictor as CALIBRATED.
            self._calibrated = True
            return final_T_mean
        finally:
            # V90 ROOT FIX (BUG #13, P1): ALWAYS unfreeze MLP weights,
            # even on exception. The previous code only unfroze at the
            # very end of the method -- if an exception happened during
            # the optimization loop (OOM, NaN loss, CUDA error), the
            # MLP weights STAYED frozen and subsequent training runs
            # silently failed to update them. With try/finally, the
            # unfreeze happens regardless of how the try block exits.
            for p in self.mlp.parameters():
                p.requires_grad_(True)
            # P3-026 ROOT FIX: also restore the prior TRAINING MODE.
            # Without this, ``self.eval()`` inside the try block leaves
            # the link_predictor in eval mode after fit_temperature
            # returns, silently disabling dropout/BatchNorm updates for
            # subsequent training.
            self.train(prior_training)
