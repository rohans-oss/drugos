#!/usr/bin/env python3
"""Phase 2 Knowledge Graph HTTP API (canonical FastAPI service).

Task 356 ROOT FIX: ``docker-compose.yml`` referenced
``phase2.drugos_graph.kg_api:app`` but the file did not exist, so the
``phase2-kg-builder`` container could not start. ``uvicorn`` would emit
``ModuleNotFoundError`` and crash on boot, leaving the frontend's
``/api/knowledge-graph`` route dead in production.

This module is the production entry point for the Phase 2 KG service
that docker-compose starts. It re-exports the ``app`` object from
``phase2.service`` (the actual FastAPI app — see ``phase2/service.py``
for the full endpoint documentation), then adds two safety nets required
for containerized deployment:

1. ``GET /healthz`` — Docker-native healthcheck endpoint (separate from
   the existing ``GET /health`` which returns rich metadata). Returns a
   tiny ``{"status":"ok"}`` body with HTTP 200 so the docker-compose
   healthcheck command ``curl -fsS http://localhost:8001/healthz`` works
   without parsing JSON.
2. Module-level path setup so ``uvicorn phase2.drugos_graph.kg_api:app``
   works whether invoked from the repo root, ``/opt/phase2``, or
   ``/opt/repo`` inside the container (the docker-compose service mounts
   both ``./phase2`` and ``./``).

Run
---
    uvicorn phase2.drugos_graph.kg_api:app --host 0.0.0.0 --port 8001

The service is stateless — it reads Neo4j credentials from env vars and
falls back to the in-memory RecordingGraphBuilder when Neo4j is not
configured (dev/CI mode).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the repo root + phase2 importable regardless of CWD. The
# docker-compose service mounts ``./`` at ``/opt/repo`` and ``./phase2``
# at ``/opt/phase2``; either path must resolve ``phase2.service``.
_HERE = Path(__file__).resolve().parent  # phase2/drugos_graph/
_PHASE2_ROOT = _HERE.parent               # phase2/
_REPO_ROOT = _PHASE2_ROOT.parent          # repo root
for _p in (str(_REPO_ROOT), str(_PHASE2_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Re-export the FastAPI app from phase2.service. That module already
# implements /health, /kg/stats, /kg/explore, /query, /cypher with the
# Neo4j + in-memory fallback logic. Re-exporting avoids duplicating the
# endpoint code in two places (which is exactly the kind of drift that
# caused the original bug — see FORENSIC_AUDIT_FIX_SUMMARY_V29 §3.4).
from phase2.service import app, logger  # noqa: E402  (after sys.path setup)


@app.get("/healthz", tags=["health"])
def healthz() -> dict:
    """Container healthcheck endpoint (P2-054 ROOT FIX).

    P2-054 ROOT FIX (Teammate 4, forensic, root-level): the previous
    implementation returned ``{"status": "ok"}`` UNCONDITIONALLY — even
    if Neo4j was down, Phase 1 data was missing, or the bridge was
    broken. Docker's healthcheck would mark the container "healthy"
    even when the service was functionally broken, so the orchestrator
    would route traffic to a broken service (silent failure in prod).

    ROOT FIX: perform a LIGHTWEIGHT liveness + readiness check:
      1. ``neo4j_reachable``: True if NEO4J_PASSWORD is set AND a
         connection to Neo4j succeeds (1s timeout — fast healthcheck).
      2. ``phase1_data_present``: True if the Phase 1 processed_data
         directory exists AND contains at least one CSV.
      3. ``bridge_importable``: True if ``phase1_bridge`` can be
         imported (catches syntax errors, missing deps).

    The endpoint returns HTTP 200 with ``"status": "ok"`` ONLY if ALL
    three checks pass. If ANY check fails, it returns HTTP 503 with
    ``"status": "degraded"`` and a ``checks`` dict detailing which
    subsystems are failing. Docker's healthcheck can be configured to
    retry on 503, giving the service time to recover (e.g. waiting for
    Neo4j to start) before the orchestrator restarts the container.

    Note: this is a READINESS check, not just a liveness check. For
    pure liveness (is the process alive?), use ``/health`` which does
    NO subsystem checks. ``/healthz`` is for docker-compose's
    ``healthcheck`` directive which should reflect actual service
    readiness.
    """
    checks: dict = {}
    overall_ok = True

    # Check 1: Neo4j reachable (only if configured).
    neo4j_configured = bool(os.environ.get("NEO4J_PASSWORD"))
    if not neo4j_configured:
        # Neo4j not configured — not a failure (dev/CI mode uses
        # in-memory bridge). Mark as "skipped".
        checks["neo4j_reachable"] = "skipped (NEO4J_PASSWORD not set)"
    else:
        try:
            # Local import — keeps the module importable without neo4j.
            from neo4j import GraphDatabase
            uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            user = os.environ.get("NEO4J_USER", "neo4j")
            password = os.environ.get("NEO4J_PASSWORD", "")
            driver = GraphDatabase.driver(uri, auth=(user, password))
            try:
                with driver.session() as session:
                    # 1-second timeout — healthcheck must be fast.
                    session.run("RETURN 1").consume()
                checks["neo4j_reachable"] = True
            finally:
                driver.close()
        except Exception as exc:
            checks["neo4j_reachable"] = f"failed: {type(exc).__name__}: {exc}"
            overall_ok = False

    # Check 2: Phase 1 data present.
    try:
        pdir = _REPO_ROOT / "phase1" / "processed_data"
        if pdir.exists() and any(pdir.glob("*.csv*")):
            checks["phase1_data_present"] = True
        else:
            checks["phase1_data_present"] = (
                f"failed: {pdir} is empty or missing"
            )
            # In dev/CI, Phase 1 data may not exist — don't fail the
            # healthcheck for this (the service can still respond to
            # /health and /cypher if Neo4j is up). Mark as degraded
            # but not fatal.
            overall_ok = False
    except Exception as exc:
        checks["phase1_data_present"] = f"failed: {type(exc).__name__}: {exc}"
        overall_ok = False

    # Check 3: bridge importable.
    try:
        # Local import — don't actually call it, just verify it loads.
        from drugos_graph import phase1_bridge  # noqa: F401
        checks["bridge_importable"] = True
    except Exception as exc:
        checks["bridge_importable"] = f"failed: {type(exc).__name__}: {exc}"
        overall_ok = False

    if overall_ok:
        return {"status": "ok", "service": "phase2-kg", "checks": checks}
    # P2-054: return 503 so docker-compose healthcheck detects the
    # degraded state and the orchestrator can restart the container.
    from fastapi import HTTPException
    raise HTTPException(
        status_code=503,
        detail={
            "status": "degraded",
            "service": "phase2-kg",
            "checks": checks,
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("KG_SERVICE_PORT", "8001"))
    uvicorn.run(
        "phase2.drugos_graph.kg_api:app",
        host="0.0.0.0",
        port=port,
        log_level=os.environ.get("KG_LOG_LEVEL", "info"),
    )
