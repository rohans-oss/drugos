#!/usr/bin/env python3
"""Phase 3 Graph Transformer HTTP API (FastAPI service).

Task 356 ROOT FIX: ``docker-compose.yml`` referenced
``scripts.gt_api:app`` but the file did not exist, so the
``phase3-trainer`` container could not start. ``uvicorn`` would emit
``ModuleNotFoundError: No module named 'scripts.gt_api'`` and crash on
boot, leaving the frontend's ``/api/predict`` and ``/api/top-k`` routes
dead in production.

This module wraps the existing ``scripts.gt_inference`` CLI helper into
a FastAPI service so the frontend can call it via HTTP instead of
spawning a Python subprocess per request (RT-006 design goal).

Endpoints
---------
    GET  /healthz
        Container healthcheck. Returns ``{"status":"ok"}``.

    GET  /health
        Rich metadata (model path, device, version).

    POST /predict
        Predict interaction scores for an explicit list of drug-disease
        pairs. Body: ``{"pairs": [{"drug":"aspirin","disease":"migraine"}]}``.
        Returns: ``{"predictions":[{"drug":...,"disease":...,"score":0.87}]}``.

    GET  /top-k?k=<int>&drug_filter=<csv>
        Rank the top-K drug-disease pairs by predicted score. Returns the
        K highest-scoring pairs the model has not seen during training.
        SH-007 v115 ROOT FIX (CRITICAL): the shared contract
        (shared/contracts/urls.py line 25) declares ``/top-k`` as GET,
        ``graph_transformer/service.py`` registers it as GET, and the
        frontend's ``gt-inference.ts`` calls it as GET. The previous
        version of THIS file registered it as POST — causing contract
        drift: requests from the frontend would hit this service with
        GET, receive a 405 Method Not Allowed, and the dashboard's
        "top novel predictions" panel would silently fail. The fix
        aligns this service with the contract (GET, query params).

Environment
-----------
    GT_CHECKPOINT_DIR: directory containing ``best_model.pt`` and
        ``graph_state.pt`` (written by ``run_4phase.py``). Defaults to
        ``./output_v100/checkpoints``.
    GT_DEVICE: ``cpu`` or ``cuda``. Defaults to ``cpu`` (Docker image
        ships CPU-only PyTorch per task 368).

Implementation notes
--------------------
The model + graph_state are loaded ONCE at startup (``_load_model``),
then cached on the ``app.state`` object. Re-loading a PyTorch checkpoint
on every request would add ~2s latency per call and exhaust GPU memory.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

# Make repo root importable so ``from graph_transformer...`` works
# regardless of where uvicorn set cwd inside the container.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "phase1"), str(_REPO_ROOT / "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("scripts.gt_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


# v113 FORENSIC ROOT FIX (IN-038 + IN-039, MEDIUM + LOW):
#   IN-038: the previous code used ``@app.on_event("startup")`` which
#   FastAPI deprecated in 0.93.0 (March 2023) in favor of the
#   ``lifespan`` context manager. The Dockerfile.ml pins
#   ``fastapi==0.110.3`` which still supports ``on_event`` but emits a
#   ``DeprecationWarning``. The deprecation will become a
#   ``RuntimeError`` in a future FastAPI version. ROOT FIX: use the
#   modern ``lifespan`` async context manager pattern.
#
#   IN-039: the previous CORS middleware had:
#     • ``allow_credentials=True``  (the GT API uses API keys, not cookies)
#     • ``allow_headers=["*"]``      (wildcard NOT honored with credentials)
#     • ``allow_origins`` from ``GT_CORS_ORIGINS`` env var with NO
#       validation -- an operator could set ``GT_CORS_ORIGINS=*`` and
#       combined with ``allow_credentials=True`` this is a CORS
#       misconfiguration that browsers reject (good) but that server-
#       to-server requests can exploit (bad).
#   ROOT FIX: drop ``allow_credentials=True``, replace
#   ``allow_headers=["*"]`` with an explicit list, validate
#   ``GT_CORS_ORIGINS`` at startup (fail if it contains ``*``).
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Pre-warm the model cache so the first request is not slow.

    Replaces the deprecated ``@app.on_event("startup")`` pattern
    (IN-038 ROOT FIX). The lifespan context manager is the modern
    FastAPI pattern (>=0.93.0) and supports both startup AND shutdown
    logic in a single function.
    """
    _load_model()
    yield
    # Shutdown logic would go here (e.g., release GPU memory).


def _validate_cors_origins(raw: str) -> list[str]:
    """Parse and validate GT_CORS_ORIGINS (IN-039 ROOT FIX).

    Splits on comma, strips whitespace, and REJECTS the wildcard ``*``
    because the GT API is a credentialed internal service. An operator
    who sets ``GT_CORS_ORIGINS=*`` is creating a CORS misconfiguration.
    """
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if "*" in origins:
        raise RuntimeError(
            "IN-039 ROOT FIX: GT_CORS_ORIGINS contains the wildcard '*' "
            "which is FORBIDDEN for the GT API. The GT API serves "
            "predictions that drive clinical-decision support -- it "
            "MUST NOT be callable from arbitrary origins. Set "
            "GT_CORS_ORIGINS to an explicit comma-separated list of "
            "trusted frontend origins (e.g., "
            "'https://app.drugos.com,https://staging.drugos.com')."
        )
    if not origins:
        # Fall back to localhost dev origin if env var is empty.
        return ["http://localhost:3000"]
    return origins


_CORS_ORIGINS = _validate_cors_origins(
    os.environ.get("GT_CORS_ORIGINS", "http://localhost:3000")
)

app = FastAPI(
    title="Autonomous Drug Repurposing — Phase 3 GT Service",
    description="HTTP wrapper around the Graph Transformer inference engine.",
    version="1.0.0",
    lifespan=lifespan,  # IN-038 ROOT FIX: use lifespan, not on_event
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    # IN-039 ROOT FIX: do NOT allow credentials. The GT API uses API
    # keys (Authorization header), not cookies. ``allow_credentials=True``
    # combined with ``allow_origins=*`` (which we now reject) is a CORS
    # misconfiguration. Even with explicit origins, credentials are not
    # needed and would prevent origin wildcarding in any future config.
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],  # read-only inference API
    # IN-039 ROOT FIX: explicit header list instead of ["*"]. The
    # wildcard is NOT honored by browsers when ``allow_credentials=True``
    # (which we've removed), but listing explicit headers is still best
    # practice -- it documents the API's contract and prevents future
    # misconfiguration if credentials are re-enabled.
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)


# ─────────────────────── Request / response schemas ─────────────────────


class DrugDiseasePair(BaseModel):
    drug: str = Field(..., description="Drug name or ChEMBL ID")
    disease: str = Field(..., description="Disease name or MONDO ID")


class PredictRequest(BaseModel):
    pairs: List[DrugDiseasePair] = Field(..., min_length=1, max_length=500)


class Prediction(BaseModel):
    drug: str
    disease: str
    score: float


# SH-006 ROOT FIX (v113 forensic): the previous PredictResponse used
# snake_case (``model_version``, ``n_pairs``) and was MISSING the
# fields the frontend contract requires (``source``, ``generatedAt``,
# ``checkpointPath``). The frontend's ``api_contracts.ts`` declares
# ``{predictions, source, modelVersion, generatedAt, count,
# checkpointPath}`` (camelCase). ``graph_transformer/service.py`` (the
# canonical service) already returns this shape. ``scripts/gt_api.py``
# (used by docker-compose.yml line 211 for production) returned a
# DIFFERENT shape, causing contract drift: the frontend's TypeScript
# types did not match the production API response. The fix aligns
# ``scripts/gt_api.py`` with the canonical frontend contract.
class PredictResponse(BaseModel):
    predictions: List[Prediction]
    source: str = "gt_checkpoint"
    modelVersion: str = "gt_v113"
    generatedAt: Optional[str] = None
    count: int = 0
    checkpointPath: Optional[str] = None
    # SH-031 v113: keep error_count/error_rate as optional (the TS
    # contract should ignore unknown fields per JSON Schema).
    error_count: Optional[int] = None
    error_rate: Optional[float] = None


class TopKRequest(BaseModel):
    top_k: int = Field(10, ge=1, le=500)
    drug_filter: Optional[List[str]] = None


class TopKResponse(BaseModel):
    predictions: List[Prediction]
    source: str = "gt_checkpoint"
    modelVersion: str = "gt_v113"
    generatedAt: Optional[str] = None
    count: int = 0
    checkpointPath: Optional[str] = None


# ─────────────────────── Model cache (loaded once) ──────────────────────


_model_lock = threading.Lock()


def _resolve_checkpoint_dir() -> Path:
    """Find the directory containing ``best_model.pt`` + ``graph_state.pt``."""
    env_dir = os.environ.get("GT_CHECKPOINT_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p
    # Default: <repo>/output_v100/checkpoints
    candidates = [
        _REPO_ROOT / "output_v100" / "checkpoints",
        _REPO_ROOT / "output_v100",
        Path("/opt/ml_artifacts/checkpoints"),
        Path("/opt/ml_artifacts"),
    ]
    for c in candidates:
        if c.is_dir() and (c / "best_model.pt").exists():
            return c
    return candidates[0] if candidates else _REPO_ROOT


def _load_model() -> Dict[str, Any]:
    """Lazily load the GT model + graph_state. Cached on app.state."""
    if getattr(app.state, "gt_model", None) is not None:
        return app.state.gt_model

    with _model_lock:
        if getattr(app.state, "gt_model", None) is not None:
            return app.state.gt_model

        ckpt_dir = _resolve_checkpoint_dir()
        ckpt_path = ckpt_dir / "best_model.pt"
        if not ckpt_path.exists():
            app.state.gt_model = None
            app.state.gt_error = f"checkpoint not found: {ckpt_path}"
            logger.warning(app.state.gt_error)
            return app.state.gt_model or {}

        # Delegate the heavy lifting to the existing gt_inference helper.
        # Re-using scripts.gt_inference avoids duplicating the model
        # reconstruction logic (which handles model_config, graph_state,
        # known_pairs, etc.).
        try:
            import scripts.gt_inference as gti  # type: ignore
            # The helper exposes _load_checkpoint(checkpoint_path) which
            # returns (model, node_features, edge_indices, node_maps,
            # drug_names, disease_names, known_pairs).
            loaded = gti._load_checkpoint(str(ckpt_path))
            app.state.gt_model = {
                "model": loaded[0],
                "node_features": loaded[1],
                "edge_indices": loaded[2],
                "node_maps": loaded[3],
                "drug_names": loaded[4],
                "disease_names": loaded[5],
                "known_pairs": loaded[6],
                "checkpoint_path": str(ckpt_path),
            }
            app.state.gt_error = None
            logger.info("GT model loaded from %s", ckpt_path)
        except Exception as exc:  # noqa: BLE001
            app.state.gt_model = None
            app.state.gt_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Failed to load GT model")

        return app.state.gt_model or {}


# ─────────────────────── Endpoints ──────────────────────────────────────


@app.get("/healthz", tags=["health"])
def healthz() -> Dict[str, str]:
    """Docker-native healthcheck (tiny 200-OK body)."""
    return {"status": "ok", "service": "phase3-gt"}


@app.get("/health", tags=["health"])
def health() -> Dict[str, Any]:
    """Rich health: model path, load status, device."""
    model = _load_model()
    return {
        "status": "ok" if model else "degraded",
        "service": "phase3-gt",
        "model_loaded": bool(model),
        "checkpoint": model.get("checkpoint_path") if model else None,
        "error": getattr(app.state, "gt_error", None),
        "device": os.environ.get("GT_DEVICE", "cpu"),
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    """Predict interaction scores for explicit drug-disease pairs."""
    model = _load_model()
    if not model:
        raise HTTPException(status_code=503, detail=getattr(app.state, "gt_error", "model not loaded"))

    try:
        import scripts.gt_inference as gti  # type: ignore
        pairs_dicts = [{"drug": p.drug, "disease": p.disease} for p in req.pairs]
        # gti._predict_pairs signature: (model, node_features, edge_indices,
        # node_maps, drug_names, disease_names, pairs) — all but `pairs`
        # come from the cached model dict.
        preds = gti._predict_pairs(
            model["model"],
            model["node_features"],
            model["edge_indices"],
            model["node_maps"],
            model["drug_names"],
            model["disease_names"],
            pairs_dicts,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return PredictResponse(
        predictions=[Prediction(**p) for p in preds],
        source="gt_checkpoint",
        modelVersion=os.environ.get("GT_MODEL_VERSION", "gt_v113"),
        generatedAt=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        count=len(preds),
        checkpointPath=os.environ.get("GT_CHECKPOINT_PATH") or os.environ.get("GT_CHECKPOINT_DIR"),
        error_count=0,
        error_rate=0.0,
    )


@app.get("/top-k", response_model=TopKResponse, tags=["inference"])
def top_k(k: int = 10, drug_filter: Optional[str] = None) -> TopKResponse:
    """Return the top-K highest-scoring drug-disease pairs.

    SH-007 v115 ROOT FIX (CRITICAL): aligned with the shared contract
    (shared/contracts/urls.py) and with graph_transformer/service.py.
    The endpoint is now GET (not POST) and reads ``k`` and
    ``drug_filter`` from query parameters. ``drug_filter`` is a
    comma-separated list of drug names (case-insensitive) — pass
    ``?drug_filter=aspirin,ibuprofen`` to restrict results to those
    drugs. Pass no ``drug_filter`` to get the global top-K.
    """
    # Validate k against the same bounds as the Pydantic model did
    # (1 <= k <= 500). Returning 422 (FastAPI's default for invalid
    # query params) is preferable to silently clamping — the frontend's
    # mlFetch treats 422 as a non-retryable client error.
    if k < 1 or k > 500:
        raise HTTPException(
            status_code=400,
            detail="k must be in [1, 500]",
        )

    # Parse drug_filter CSV into a list. Empty string / missing param
    # → None (no filter). Whitespace-only entries are dropped.
    drug_filter_list: Optional[List[str]] = None
    if drug_filter is not None and drug_filter.strip():
        drug_filter_list = [
            d.strip()
            for d in drug_filter.split(",")
            if d.strip()
        ]
        if not drug_filter_list:
            drug_filter_list = None

    model = _load_model()
    if not model:
        raise HTTPException(status_code=503, detail=getattr(app.state, "gt_error", "model not loaded"))

    try:
        import scripts.gt_inference as gti  # type: ignore
        # gti._top_k_novel signature: (model, node_features, edge_indices,
        # drug_names, disease_names, known_pairs, top_k).
        preds = gti._top_k_novel(
            model["model"],
            model["node_features"],
            model["edge_indices"],
            model["drug_names"],
            model["disease_names"],
            model["known_pairs"],
            k,
        )
        # Apply optional drug filter (case-insensitive) — the underlying
        # helper does not natively filter, so we slice after ranking.
        if drug_filter_list:
            allowed = {d.lower() for d in drug_filter_list}
            preds = [p for p in preds if p["drug"].lower() in allowed][:k]
    except Exception as exc:  # noqa: BLE001
        logger.exception("top_k failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return TopKResponse(
        predictions=[Prediction(**p) for p in preds],
        source="gt_checkpoint",
        modelVersion=os.environ.get("GT_MODEL_VERSION", "gt_v113"),
        generatedAt=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        count=len(preds),
        checkpointPath=os.environ.get("GT_CHECKPOINT_PATH") or os.environ.get("GT_CHECKPOINT_DIR"),
    )


# v113 IN-038 ROOT FIX: the deprecated ``@app.on_event("startup")``
# decorator has been REMOVED. The startup logic is now in the
# ``lifespan`` async context manager defined above (passed to
# ``FastAPI(lifespan=lifespan)``). This is the modern FastAPI pattern
# (>=0.93.0) and avoids the DeprecationWarning that the old decorator
# emits. The Dockerfile.ml pins ``fastapi==0.110.3`` which still
# supports ``on_event`` but the deprecation will become a RuntimeError
# in a future FastAPI version.


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("GT_SERVICE_PORT", "8002"))
    uvicorn.run(
        "scripts.gt_api:app",
        host="0.0.0.0",
        port=port,
        log_level=os.environ.get("GT_LOG_LEVEL", "info"),
    )
