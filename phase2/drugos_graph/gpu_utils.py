"""DrugOS Graph Module -- GPU Utilities
========================================
GPU memory validation and batch size testing for PyG data loading.
"""

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def check_gpu_available() -> Dict[str, Any]:
    """Check GPU availability and memory.

    P2-004 FORENSIC ROOT FIX (Team 4 -- multi-GPU device_index missing):
    The previous version did NOT report ``device_index``.
    ``transe_model._get_device`` reads ``info.get("device_index")`` to
    respect the operator's chosen GPU on multi-GPU hosts (the v35 L-31
    root fix). But this function never set ``device_index`` -- so
    ``info.get("device_index")`` always returned None, and the L-31 fix
    silently fell through to ``return torch.device("cuda")`` which is
    always ``cuda:0``. On multi-GPU hosts (e.g. an 8x A100 box), the
    operator's ``CUDA_VISIBLE_DEVICES=2`` was silently ignored --
    training always ran on GPU 0.

    ROOT FIX: report ``device_index = torch.cuda.current_device()``.
    ``torch.cuda.current_device()`` honors ``CUDA_VISIBLE_DEVICES`` --
    when the operator sets ``CUDA_VISIBLE_DEVICES=2``, PyTorch remaps
    device 2 to ``cuda:0`` (the only visible device), and
    ``current_device()`` returns 0 (the local index). This is the
    correct semantic: ``device_index`` is the LOCAL index within the
    visible devices, which is what ``torch.device("cuda", idx)`` expects.
    """
    info = {"cuda_available": torch.cuda.is_available()}

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
        # P2-004 ROOT FIX: report device_index so transe_model._get_device
        # can respect the operator's CUDA_VISIBLE_DEVICES choice on
        # multi-GPU hosts. torch.cuda.current_device() returns the LOCAL
        # index within the visible devices (honors CUDA_VISIBLE_DEVICES).
        info["device_index"] = torch.cuda.current_device()
        info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_mem / 1e9
        info["allocated_memory_gb"] = torch.cuda.memory_allocated(0) / 1e9
        info["free_memory_gb"] = info["total_memory_gb"] - info["allocated_memory_gb"]

    logger.info(f"GPU check: {info}")
    return info


def test_batch_memory(
    num_nodes: int = 100000,
    num_edges: int = 6000000,
    feat_dim: int = 256,
    batch_size: int = 512,
    # P2-051 ROOT FIX: the previous signature hard-coded
    # ``device="cuda"`` inside the function body, which on a multi-GPU
    # host always allocates the test batch on cuda:0. If cuda:0 was
    # occupied (e.g. by another process) the function reported
    # "FAIL — OOM" even when cuda:2 had 70 GB free — a FALSE failure
    # that misled operators into thinking the GPU was unusable. Root
    # fix: accept an explicit ``device`` parameter (string or
    # ``torch.device``) so the caller can target a specific GPU.
    # Default to "cuda" (== cuda:0) for backward compat, but operators
    # on multi-GPU hosts can now pass device="cuda:2" to validate the
    # specific GPU they intend to train on. We also surface the chosen
    # device in the result dict so the audit log records WHICH GPU was
    # tested — without this, the result was ambiguous on multi-GPU
    # hosts.
    device: str = "cuda",
) -> Dict[str, Any]:
    """Test if GPU memory can fit a mini-batch.

    Args:
        num_nodes: Approximate total node count.
        num_edges: Approximate total edge count.
        feat_dim: Node feature dimension.
        batch_size: Mini-batch size to test.
        device: Target CUDA device specifier ("cuda", "cuda:0",
            "cuda:2", ...). Defaults to "cuda" (cuda:0). On multi-GPU
            hosts, pass the specific device you intend to train on so
            the test measures the correct GPU's free memory (P2-051).

    Returns:
        Dict with memory estimates and pass/fail.
    """
    # Estimate memory per node feature
    node_feat_bytes = num_nodes * feat_dim * 4  # float32
    # Estimate memory per edge (2 int64 indices)
    edge_index_bytes = num_edges * 2 * 8  # int64

    total_estimated_gb = (node_feat_bytes + edge_index_bytes) / 1e9

    result = {
        "estimated_total_gb": round(total_estimated_gb, 2),
        "node_feat_gb": round(node_feat_bytes / 1e9, 2),
        "edge_index_gb": round(edge_index_bytes / 1e9, 2),
        "batch_size": batch_size,
        # P2-051: record the device under test so the audit log is
        # unambiguous on multi-GPU hosts.
        "device_requested": str(device),
    }

    if torch.cuda.is_available():
        # P2-051 ROOT FIX: resolve the requested device explicitly so
        # we measure the FREE memory of the GPU the caller intends to
        # use, not always cuda:0. ``torch.device(device).index`` is
        # None for "cuda" (defaults to 0) — coerce to int so the
        # ``torch.cuda`` APIs accept it.
        _dev = torch.device(device)
        _dev_idx = _dev.index if _dev.index is not None else 0
        # Validate the index is in range — a typo like "cuda:9" on a
        # 4-GPU host should produce a clear error, not a cryptic
        # CUDA error inside ``torch.randn``.
        if _dev_idx >= torch.cuda.device_count():
            result["fits_gpu"] = False
            result["batch_test"] = (
                f"FAIL — invalid device {device!r} "
                f"(host has {torch.cuda.device_count()} GPU(s))"
            )
            logger.error(
                "test_batch_memory: requested device %s but host has "
                "only %d GPU(s). (P2-051)",
                device, torch.cuda.device_count(),
            )
            return result
        free_gb = (
            torch.cuda.get_device_properties(_dev_idx).total_mem
            - torch.cuda.memory_allocated(_dev_idx)
        ) / 1e9
        result["gpu_free_gb"] = round(free_gb, 2)
        result["device_tested"] = f"cuda:{_dev_idx}"
        result["fits_gpu"] = total_estimated_gb < free_gb

        # Test actual mini-batch allocation ON THE REQUESTED DEVICE.
        # Task 96 ROOT FIX: the previous except clause referenced
        # ``torch.cuda.OutOfMemoryError`` directly. That attribute does
        # NOT exist on PyTorch < 1.13 -- on those versions the import
        # itself succeeds but the attribute access raises
        # AttributeError when the except clause is evaluated, turning
        # every OOM into a CRASH. Even on PyTorch >= 1.13, OOM from
        # inside ``torch.randn`` is raised as ``RuntimeError`` in some
        # code paths (notably when the CUDA driver refuses the
        # allocation). The fix: (a) resolve the exception class via
        # ``getattr`` with a ``RuntimeError`` fallback; (b) catch BOTH
        # the dedicated OOM exception and ``RuntimeError``; (c)
        # narrow RuntimeError matches by inspecting the message so
        # non-OOM runtime errors (e.g. device-side assert) re-raise;
        # (d) set ``fits_gpu=False`` in the OOM branch so callers that
        # branch on ``fits_gpu`` do not proceed to allocate a real
        # training batch and crash. Previously ``fits_gpu`` retained
        # the pre-OOM estimate (which could be True) -- a "false PASS"
        # that misled operators into thinking the GPU was usable.
        _OOMExc = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
        try:
            test_batch = torch.randn(batch_size, feat_dim, device=_dev)
            result["batch_test"] = "PASS"
            del test_batch
            # P2-051: empty the cache on the tested device, not the
            # default device.
            torch.cuda.empty_cache()
        except _OOMExc:
            # Task 96: dedicated OOM exception path.
            result["batch_test"] = f"FAIL — OOM on cuda:{_dev_idx}"
            result["fits_gpu"] = False
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 -- best-effort cache clear
                pass
        except RuntimeError as exc:
            # Task 96: older PyTorch versions raise plain RuntimeError
            # on OOM. Narrow by message so genuine device errors
            # (e.g. device-side assert, misaligned access) still
            # propagate and surface as real bugs rather than being
            # silently swallowed as "OOM".
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda memory" in msg or "oom" in msg:
                result["batch_test"] = f"FAIL — OOM on cuda:{_dev_idx}"
                result["fits_gpu"] = False
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            else:
                raise
    else:
        result["fits_gpu"] = False
        result["batch_test"] = "SKIP -- no GPU"

    logger.info(f"GPU memory test: {result}")
    return result


def recommend_batch_size(
    total_memory_gb: float,
    feat_dim: int = 256,
    safety_factor: float = 0.7,
    # v34 ROOT FIX (HIGH #9): the previous default was `num_negatives=1`
    # "for backward compat with callers that don't pass it." But
    # TransEConfig.num_negatives defaults to 10. Callers that didn't
    # pass `num_negatives` got a batch size recommendation 11× too large
    # -> OOM on GPUs the function claimed were safe. The fix: default to
    # 10 (matching TransEConfig) so callers get the CORRECT memory
    # estimate out of the box. Callers that want the old behavior can
    # explicitly pass `num_negatives=1`.
    num_negatives: int = 10,
) -> int:
    """Recommend maximum batch size based on available GPU memory.

    v28 ROOT FIX (audit ML-11): the previous formula
    ``bytes_per_sample = feat_dim * 4 * 2`` assumed exactly 2 nodes
    per sample (src + dst of a positive edge). But the TransE trainer
    (and any link-prediction trainer using negative sampling) actually
    loads ``1 + num_negatives`` nodes per sample -- the positive's src
    and dst, PLUS one tail embedding per negative sample. With the
    default ``num_negatives=10``, the true memory cost per positive
    sample is ``feat_dim * 4 * 2 * (1 + 10) = 22 * feat_dim`` bytes --
    11× what the old formula assumed. The recommended batch size was
    therefore 11× too large, causing OOM crashes on GPUs that the
    function claimed were safe.

    The fix adds an explicit ``num_negatives`` parameter and corrects
    the formula to ``bytes_per_sample = feat_dim * 4 * 2 * (1 +
    num_negatives)``.

    v34 ROOT FIX (HIGH #9): default changed from 1 to 10 to match
    TransEConfig.num_negatives. Callers that don't pass `num_negatives`
    now get the CORRECT memory estimate (11× smaller batch) instead of
    an OOM-inducing 11× over-estimate.

    Parameters
    ----------
    total_memory_gb : float
        Total GPU memory in GB.
    feat_dim : int
        Node feature / embedding dimension. Default 256 (matches
        TransEConfig.embedding_dim default).
    safety_factor : float
        Fraction of total memory to use. Default 0.7 (leaves 30% for
        gradients, activations, and framework overhead).
    num_negatives : int
        Number of negative samples per positive sample. Default 10
        (matches TransEConfig.num_negatives).
    """
    available_bytes = total_memory_gb * 1e9 * safety_factor
    # v72 ROOT FIX (P2C-020): warn when num_negatives=1 is passed. The v34
    # fix changed the default from 1 to 10 to match TransEConfig, but the
    # function signature still accepts any int. A caller that explicitly
    # passes num_negatives=1 (backward compat) gets a batch size
    # recommendation 11× too large -> OOM on GPUs the function claims are
    # safe. The fix logs a WARNING so the operator can see the
    # discrepancy. Callers that genuinely want num_negatives=1 (e.g. a
    # custom trainer with 1:1 neg ratio) can silence the warning by
    # setting DRUGOS_ALLOW_NUM_NEGATIVES_1=1.
    if num_negatives == 1:
        import os as _os_p2c020
        _allow_n1 = _os_p2c020.environ.get(
            "DRUGOS_ALLOW_NUM_NEGATIVES_1", "0"
        ) == "1"
        if not _allow_n1:
            logger.warning(
                "recommend_batch_size called with num_negatives=1 -- the "
                "recommended batch size will be 11× too large for a "
                "trainer using the default 1:10 pos:neg ratio "
                "(TransEConfig.num_negatives=10). This causes OOM on "
                "GPUs the function claims are safe. Pass "
                "num_negatives=10 (the default) to match "
                "TransEConfig, OR set DRUGOS_ALLOW_NUM_NEGATIVES_1=1 "
                "to silence this warning if you genuinely use a 1:1 "
                "neg ratio. (P2C-020 root fix)"
            )
    # v28 ML-11 + v41 ROOT FIX (P2 #3): the previous formula only counted
    # embedding lookups. It did NOT count:
    #   - gradients (2x parameters for autograd)
    #   - Adam optimizer state (2x parameters: m + v)
    #   - intermediate activations
    # The fix adds a 5x multiplier to account for these. The formula is
    # now: bytes_per_sample = feat_dim * 4 * 2 * (1 + num_negatives) * 5
    # where 5 = 1 (forward) + 2 (gradients) + 2 (Adam m+v).
    # * 4  : float32 bytes per scalar.
    # * 2  : src + dst of the positive triple.
    # * (1 + num_negatives): the positive itself + one tail embedding
    #   per negative sample.
    # * 5  : v41 fix -- forward + gradients + Adam state.
    bytes_per_sample = feat_dim * 4 * 2 * (1 + num_negatives) * 5  # v41: added *5
    max_batch = int(available_bytes / bytes_per_sample)

    # Cap at reasonable values
    # v43 ROOT FIX (P2 -- recommend_batch_size cap arbitrary): the
    # previous cap was 8192 with no documented basis. The cap should be
    # configurable via env var DRUGOS_MAX_BATCH_SIZE (default 8192 for
    # backward compat). This lets operators with large GPUs (A100 80GB)
    # raise the cap and operators with small GPUs (T4 16GB) lower it.
    import os as _os
    _max_cap = int(_os.environ.get("DRUGOS_MAX_BATCH_SIZE", "8192"))
    recommended = min(max_batch, _max_cap)
    logger.info(
        f"Recommended batch size: {recommended} "
        f"(GPU: {total_memory_gb:.1f}GB, feat_dim={feat_dim}, "
        f"num_negatives={num_negatives}, "
        f"bytes_per_sample={bytes_per_sample})"
    )
    return recommended
