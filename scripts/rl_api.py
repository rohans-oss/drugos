#!/usr/bin/env python3
"""Phase 4 RL Hypothesis Ranker HTTP API (canonical FastAPI service).

Task 356 ROOT FIX: ``docker-compose.yml`` referenced
``scripts.rl_api:app`` but the file did not exist, so the
``phase4-rl`` container could not start. ``uvicorn`` would emit
``ModuleNotFoundError: No module named 'scripts.rl_api'`` and crash on
boot, leaving the frontend's ``/api/rl`` route dead in production.

This module re-exports the FastAPI ``app`` from ``rl.service`` (the
canonical Phase 4 service — see ``rl/service.py`` for full endpoint
documentation) and adds a Docker-native ``/healthz`` endpoint that the
docker-compose healthcheck directive can call without parsing JSON.

Run
---
    uvicorn scripts.rl_api:app --host 0.0.0.0 --port 8003

The service is stateless: it reads either the trained PPO checkpoint
(``RL_CHECKPOINT_PATH``) or the latest ``top_candidates_*.csv``
(``RL_OUTPUT_DIR``) on every request, so the docker-compose volume
mount (``ml-artifacts:/opt/ml_artifacts:ro``) is sufficient.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make repo root + rl/ importable regardless of where uvicorn set cwd
# inside the container. The docker-compose service mounts ``./`` at
# ``/opt/repo``; this file lives at ``/opt/repo/scripts/rl_api.py``.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_RL_DIR = _REPO_ROOT / "rl"
for _p in (str(_REPO_ROOT), str(_RL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Re-export the canonical app. rl.service implements /health, /rank,
# /rank/{drug}, and POST /rank with checkpoint + CSV fallback logic.
# Re-exporting avoids code duplication (the same bug class that created
# the original Task 356 issue — see docker-compose.yml line 222).
from rl.service import app, logger  # noqa: E402  (after sys.path setup)


@app.get("/healthz", tags=["health"])
def healthz() -> dict:
    """Docker-native healthcheck (tiny 200-OK body).

    Used by docker-compose's healthcheck directive so the orchestrator
    can mark this service healthy without parsing the richer ``/health``
    JSON response.
    """
    return {"status": "ok", "service": "phase4-rl"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("RL_SERVICE_PORT", "8003"))
    uvicorn.run(
        "scripts.rl_api:app",
        host="0.0.0.0",
        port=port,
        log_level=os.environ.get("RL_LOG_LEVEL", "info"),
    )
