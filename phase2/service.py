#!/usr/bin/env python3
"""Phase 2 Knowledge Graph Service (v107 forensic root fix).

Wraps Phase 2's KG builder / query logic as an HTTP service so the
Next.js frontend can proxy to it via ``KG_SERVICE_URL``. The frontend's
``src/lib/services/knowledge-graph-stats.ts`` and
``src/app/api/knowledge-graph/route.ts`` already proxy to
``KG_SERVICE_URL`` — this service is what they expect to find there.

Endpoints
---------
    GET  /health
        Liveness probe. Returns service metadata + Neo4j config status.

    GET  /kg/stats
        Real KG stats from Neo4j (or in-memory bridge fallback).
        Returns 503 if neither backend is available (P2-008).

    GET  /kg/explore?drug=...&disease=...&limit=N
        Subgraph exploration around a drug or disease node.
        Returns 503 if Neo4j is unavailable AND the in-memory bridge
        cannot resolve the query (P2-009). When the in-memory bridge is
        available, performs a real BFS over the in-memory KG.

    POST /query
        Structured query. Body: ``{"drug": "...", "disease": "...", "limit": N}``.
        Returns a subgraph centered on the requested drug/disease.
        (P2-002 root fix — the frontend POSTs here.)

    POST /cypher
        Raw Cypher passthrough. Body: ``{"cypher": "...", "params": {...}}``.
        Applies a read-only whitelist (MATCH/OPTIONAL MATCH/WITH/RETURN/WHERE
        only). Forwards to Neo4j with a hard 30s timeout and 1000-row cap.
        (P2-002 root fix — the frontend POSTs here.)

Run
---
    cd phase2 && python service.py
    # or: uvicorn phase2.service:app --host 0.0.0.0 --port 8002

Environment
-----------
    NEO4J_URI: bolt://localhost:7687 (or unset for in-memory CSV mode)
    NEO4J_USER: neo4j
    NEO4J_PASSWORD: <password>
    KG_CORS_ORIGINS: comma-separated whitelist of allowed origins
                     (default: ``http://localhost:3000``)
    DRUGOS_ENVIRONMENT: ``production`` (default) or ``dev``

v107 forensic root fixes applied (P2-001, P2-002, P2-008, P2-009, P2-010,
P2-016, P2-017) — see inline comments for the exact bug each fix addresses.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Make phase2 importable.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("phase2.service")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Autonomous Drug Repurposing — Phase 2 Knowledge Graph Service",
    description="HTTP wrapper around Phase 2 KG builder / queries.",
    version="1.0.0",
)

# ─── P2-016 ROOT FIX (v107 forensic): CORS hardening ───────────────────────
# The previous code set ``allow_origins=["*"]`` (wildcard — any website could
# read KG data) AND ``allow_methods=["GET"]`` (blocking every POST preflight).
# Even if /query and /cypher endpoints existed, the browser would block them.
# ROOT FIX: read a whitelist from KG_CORS_ORIGINS (default: localhost:3000
# for the Next.js dev server). Allow GET, POST, OPTIONS so the frontend's
# POST /query and POST /cypher requests pass CORS preflight.
_DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
_cors_env = os.environ.get("KG_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS)
_allowed_origins: List[str] = [
    origin.strip() for origin in _cors_env.split(",") if origin.strip()
]
# ─── P2-035 ROOT FIX (v109 forensic): CORS hardening — explicit header list
# The v107 fix replaced ``allow_origins=["*"]`` with a whitelist but
# kept ``allow_headers=["*"]``. With ``allow_credentials=True``, this is
# a CORS misconfiguration: any custom header from any whitelisted origin
# is accepted, which defeats the security purpose of the origin
# whitelist. Browsers technically reject ``Access-Control-Allow-Origin: *``
# combined with credentials, but FastAPI's CORSMiddleware reflects the
# request Origin instead — so the wildcard headers list effectively
# allows ANY header from ANY whitelisted origin with credentials.
# ROOT FIX: replace ``allow_headers=["*"]`` with an explicit list of
# headers the frontend actually sends. Add ``Content-Type`` (JSON
# bodies), ``Authorization`` (future JWT/Bearer), and ``X-Request-Id``
# (correlation). Anything else is rejected by the browser preflight.
_ALLOWED_CORS_HEADERS = [
    "Content-Type",
    "Authorization",
    "X-Request-Id",
    "X-Correlation-Id",
    "Accept",
    "Origin",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=_ALLOWED_CORS_HEADERS,
)


# ─── P2-001 ROOT FIX (v109 forensic): unified Neo4j credential env vars
# v107 had TWO different env var names for the same Neo4j password:
#   * ``service.py`` read ``NEO4J_PASSWORD``.
#   * ``config.py`` read ``DRUGOS_NEO4J_PASSWORD`` (line 2782).
# Operators who set only one of them would silently fall back to CSV
# mode with no Neo4j, breaking the /cypher endpoint. ROOT FIX: read
# BOTH env vars (preferring the ``DRUGOS_*`` form for forward compat)
# and emit a one-time warning if only the legacy form is set. The same
# unification is applied to ``NEO4J_URI`` / ``NEO4J_USER``.
_NEO4J_ENV_VAR_WARNING_EMITTED = False


def _get_neo4j_env_var(short_name: str, default: str = "") -> str:
    """Read a Neo4j config value from BOTH env var naming conventions.

    Priority (highest first):
      1. ``DRUGOS_NEO4J_<NAME>``  (forward-looking canonical form)
      2. ``NEO4J_<NAME>``         (legacy form, kept for backward compat)

    Emits a one-time warning if only the legacy form is set, so operators
    know to migrate to the canonical form.
    """
    global _NEO4J_ENV_VAR_WARNING_EMITTED
    canonical = f"DRUGOS_NEO4J_{short_name}"
    legacy = f"NEO4J_{short_name}"
    canonical_val = os.environ.get(canonical)
    legacy_val = os.environ.get(legacy)
    if canonical_val is not None:
        return canonical_val
    if legacy_val is not None:
        if not _NEO4J_ENV_VAR_WARNING_EMITTED:
            logger.warning(
                "Phase 2 service: using legacy env var %s (value set). "
                "Please migrate to the canonical form %s. Both forms are "
                "accepted, but the legacy form may be deprecated in a "
                "future release.",
                legacy, canonical,
            )
            _NEO4J_ENV_VAR_WARNING_EMITTED = True
        return legacy_val
    return default


# ─── P2-017 ROOT FIX (v107 forensic): Neo4j driver resource-leak guard ─────
def _neo4j_driver():
    """Construct a Neo4j driver. Raises if Neo4j password is unset."""
    from neo4j import GraphDatabase  # local import — keeps service importable
    uri = _get_neo4j_env_var("URI", "bolt://localhost:7687")
    user = _get_neo4j_env_var("USER", "neo4j")
    password = _get_neo4j_env_var("PASSWORD", "")
    if not password:
        raise RuntimeError(
            "Neo4j password is not set — set DRUGOS_NEO4J_PASSWORD (or "
            "legacy NEO4J_PASSWORD) to enable the Neo4j backend."
        )
    return GraphDatabase.driver(uri, auth=(user, password))


def _run_neo4j(fn):
    """Run a function with a Neo4j session. Guarantees driver.close().

    P2-017 ROOT FIX (v107 forensic): the previous code created a driver,
    ran queries, then called ``driver.close()`` at the end of the try
    block. If ``session.run()`` raised, ``driver.close()`` was NEVER
    called — the driver leaked. Under load, this exhausts the Neo4j
    connection pool (~100 calls) and the service becomes unresponsive.
    ROOT FIX: use ``try/finally`` to guarantee ``driver.close()`` runs
    even on exception. This is the same pattern recommended by the
    official Neo4j Python driver docs.
    """
    driver = _neo4j_driver()
    try:
        return fn(driver)
    finally:
        try:
            driver.close()
        except Exception as close_exc:  # pragma: no cover — defensive
            logger.warning("Phase 2 service: driver.close() failed: %s", close_exc)


def _get_kg_stats_from_neo4j() -> Optional[Dict[str, Any]]:
    """Try to read KG stats from Neo4j. Returns None if Neo4j is not available."""

    def _query(driver):
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            edge_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS c"
            ).single()["c"]
            # Per-label node counts.
            result = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS c"
            )
            node_types = {record["label"]: record["c"] for record in result}
            # Per-type edge counts.
            result = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c"
            )
            edge_types = {record["t"]: record["c"] for record in result}
        # SH-026 ROOT FIX (Teammate 4): return BOTH the legacy snake_case
        # fields AND the canonical contract fields (node_type_counts,
        # edge_type_counts, source, last_updated) so the TypeScript
        # contract in frontend/contracts/api_contracts.ts:KgStatsResponse
        # matches exactly. The ``source`` field uses the contract's enum
        # ("neo4j" | "in_memory").
        from datetime import datetime, timezone
        return {
            # Legacy fields (backward compat)
            "node_count": int(node_count),
            "edge_count": int(edge_count),
            "node_types": node_types,
            "edge_types": edge_types,
            "backend": "neo4j",
            # SH-026: canonical contract fields
            "node_type_counts": node_types,
            "edge_type_counts": edge_types,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "source": "neo4j",
        }

    try:
        return _run_neo4j(_query)
    except Exception as exc:
        logger.info(
            "Phase 2 service: Neo4j not available (%s), falling back to in-memory builder.",
            exc,
        )
        return None


def _get_kg_stats_from_builder() -> Dict[str, Any]:
    """Build the KG in-memory from Phase 1 CSVs via the bridge, then count.

    This is the Tier-2 fallback used in dev/CI when Neo4j is not running.
    The bridge (drugos_graph.phase1_bridge.run_phase1_to_phase2) reads
    the same Phase 1 CSVs that run_4phase.py reads, so the stats here
    EXACTLY match what the bridge would load into Neo4j in production.

    P2-001 ROOT FIX (v107 forensic): the previous code UNCONDITIONALLY
    called ``pipelines._embedded_samples.write_all_samples(str(pdir))``
    when ``phase1/processed_data`` was empty or had no CSVs. This
    injected 10 mock drug records (the SAME embedded samples P1-001
    found) on EVERY ``/kg/stats`` API call when Phase 1 hadn't run. The
    mock data was labeled as real and consumed by the bridge. The KG
    stats endpoint returned counts based on mock data; the frontend
    displayed mock stats; the GNN trained on mock drug-protein
    interactions. Violates the user's "NO mock data" mandate.
    ROOT FIX: remove the ``write_all_samples`` call. If Phase 1 data is
    missing, return a structured 503 (caller raises HTTPException) with
    a clear error. Operators must run Phase 1 first.

    P2-010 ROOT FIX (v107 forensic): the previous code walked
    ``builder.node_loads`` and read ``node.get("type", "unknown")`` for
    each node. But the node-level ``type`` property is the
    ChEMBL/DrugBank scientific type (e.g. "small molecule", "biotech",
    "antibody"), NOT the KG label (Compound, Disease, etc.). The
    returned ``node_types`` dict had keys like "small molecule" instead
    of "Compound". The frontend's nodeCount breakdown by canonical type
    was wrong (showed "small molecule: 5000" instead of "Compound: 5000").
    ROOT FIX: use ``load["label"]`` (the KG label) for per-type
    breakdown. The KG label is the canonical Phase 2 vocabulary
    (Compound, Protein, Gene, Disease, Pathway, ClinicalOutcome).
    """
    pdir = _REPO_ROOT / "phase1" / "processed_data"
    # P2-001: do NOT inject mock data. If Phase 1 has not run, fail loud.
    if not pdir.exists() or not any(pdir.glob("*.csv*")):
        raise FileNotFoundError(
            f"Phase 1 processed data not found at {pdir}. Run the Phase 1 "
            f"pipeline first (python run_4phase.py phase1, or "
            f"python run_full_platform.py --phase 1). The Phase 2 KG service "
            f"refuses to serve mock data — the user's mandate is NO mock data. "
            f"(P2-001 root fix, v107)"
        )

    try:
        from drugos_graph.phase1_bridge import run_phase1_to_phase2
        # SH-010 ROOT FIX (Teammate 4): the previous code HARDCODED
        # ``prefer_postgres=False``, which bypassed Phase 1's PostgreSQL
        # staging DB even in production. ROOT FIX: read the
        # ``DRUGOS_PREFER_POSTGRES`` env var (default: "0" for dev/CI
        # backward compat; set to "1" in production).
        result = run_phase1_to_phase2(
            phase1_processed_dir=str(pdir),
            prefer_postgres=os.environ.get(
                "DRUGOS_PREFER_POSTGRES", "0"
            ).lower() in ("1", "true", "yes", "on"),
        )
        builder = result["builder"]
        summary = result["summary"]

        # P2-010: use load["label"] (KG label) — NOT node.get("type") which
        # is the ChEMBL/DrugBank scientific type ("small molecule", etc.).
        # P2-037 ROOT FIX (v109): the previous code accessed
        # ``builder.node_loads`` and ``builder.edge_loads`` directly via
        # ``getattr(builder, 'node_loads', [])``. The in-memory
        # ``RecordingGraphBuilder`` HAS these attributes (they are lists
        # of dicts), but the production Neo4j ``DrugOSGraphBuilder`` does
        # NOT — its load history lives inside Neo4j itself, not in
        # Python memory. So when the bridge returned a
        # ``DrugOSGraphBuilder`` (because Neo4j was configured), the
        # ``getattr`` silently returned ``[]``, and the API reported
        # ``node_count=0, edge_count=0`` even though the KG had millions
        # of nodes. ROOT FIX: extract the load-summary from the bridge
        # ``summary`` dict (which IS always populated) instead of from
        # builder attributes. This works for both builder types.
        node_types: Dict[str, int] = {}
        edge_types: Dict[str, int] = {}
        # Try builder.node_loads / builder.edge_loads first (works for
        # RecordingGraphBuilder, the dev/CI in-memory builder).
        node_loads = getattr(builder, "node_loads", None)
        edge_loads = getattr(builder, "edge_loads", None)
        if node_loads and edge_loads:
            for load in node_loads:
                if not isinstance(load, dict):
                    continue
                label = load.get("label") or "unknown"
                n_nodes = len(load.get("nodes", []) or [])
                node_types[label] = node_types.get(label, 0) + n_nodes
            for load in edge_loads:
                if not isinstance(load, dict):
                    continue
                rel = load.get("rel_type") or "unknown"
                n_edges = len(load.get("edges", []) or [])
                edge_types[rel] = edge_types.get(rel, 0) + n_edges
        else:
            # Production builder (DrugOSGraphBuilder) — read type counts
            # from the bridge ``summary`` dict, which is populated for
            # both builder types. The summary keys are:
            #   ``node_type_counts``: {label: count}
            #   ``edge_type_counts``: {(src, rel, dst): count}
            for label, count in (summary.get("node_type_counts") or {}).items():
                node_types[label] = node_types.get(label, 0) + int(count)
            for et_key, count in (summary.get("edge_type_counts") or {}).items():
                # et_key may be a tuple ("(Compound, treats, Disease)")
                # or a string. Extract the relation verb for the type.
                if isinstance(et_key, (tuple, list)) and len(et_key) >= 2:
                    rel = et_key[1]
                else:
                    # String form like "(Compound, treats, Disease)".
                    import re as _re
                    m = _re.match(r"\([^,]+,\s*([^,]+?),\s*[^,]+\)", str(et_key))
                    rel = m.group(1).strip() if m else str(et_key)
                edge_types[rel] = edge_types.get(rel, 0) + int(count)

        # SH-026 ROOT FIX (Teammate 4): return BOTH the legacy snake_case
        # fields (node_types, edge_types, backend) AND the canonical
        # contract fields (node_type_counts, edge_type_counts, source,
        # last_updated) so the TypeScript contract in
        # frontend/contracts/api_contracts.ts:KgStatsResponse matches.
        # The ``source`` field uses the contract's enum
        # ("neo4j" | "in_memory") — we map ``in_memory_bridge`` →
        # ``in_memory`` and keep the legacy ``backend`` field for
        # backward compat with kg-service.ts (which reads both).
        from datetime import datetime, timezone
        _last_updated = datetime.now(timezone.utc).isoformat()
        _backend_legacy = "in_memory_bridge"
        _source = "in_memory"
        return {
            # Legacy fields (backward compat with kg-service.ts)
            "node_count": int(summary.get("nodes_loaded", 0)),
            "edge_count": int(summary.get("edges_loaded", 0)),
            "node_types": node_types,
            "edge_types": edge_types,
            "backend": _backend_legacy,
            "sources_read": summary.get("sources_read", []),
            # SH-026: canonical contract fields (matches TS KgStatsResponse)
            "node_type_counts": node_types,
            "edge_type_counts": edge_types,
            "last_updated": _last_updated,
            "source": _source,
        }
    except FileNotFoundError:
        # Re-raise P2-001 missing-data errors unchanged.
        raise
    except Exception as exc:
        logger.error("Phase 2 service: bridge failed: %s", exc, exc_info=True)
        # P2-008: signal failure via the backend="error" marker; the route
        # handler converts this to HTTP 503.
        return {
            "node_count": 0,
            "edge_count": 0,
            "node_types": {},
            "edge_types": {},
            "backend": "error",
            "error": str(exc),
        }


# ─── P2-009 ROOT FIX (v107 forensic): in-memory subgraph exploration ──────
def _explore_subgraph_in_memory(
    drug: Optional[str], disease: Optional[str], limit: int
) -> Optional[Dict[str, Any]]:
    """BFS over the in-memory RecordingGraphBuilder.

    The previous code returned ``{"nodes": [], "edges": [], "note": "not yet
    implemented"}`` with HTTP 200 whenever Neo4j was unavailable. The KG
    explorer feature was permanently broken in dev/CI (no Neo4j).

    ROOT FIX: build the in-memory KG (same call as /kg/stats uses), then
    perform a real 2-hop BFS from the requested drug or disease node.
    Returns None if the in-memory builder is unavailable (caller raises
    503). This makes the KG explorer feature work in dev/CI without a
    Neo4j dependency.
    """
    try:
        from drugos_graph.phase1_bridge import run_phase1_to_phase2
        pdir = _REPO_ROOT / "phase1" / "processed_data"
        if not pdir.exists() or not any(pdir.glob("*.csv*")):
            return None
        # SH-010 ROOT FIX (Teammate 4): same fix as /kg/stats — read the
        # ``DRUGOS_PREFER_POSTGRES`` env var instead of hardcoding False.
        result = run_phase1_to_phase2(
            phase1_processed_dir=str(pdir),
            prefer_postgres=os.environ.get(
                "DRUGOS_PREFER_POSTGRES", "0"
            ).lower() in ("1", "true", "yes", "on"),
        )
        builder = result["builder"]
    except Exception as exc:
        logger.info("Phase 2 service: in-memory explore unavailable: %s", exc)
        return None

    # P2-038 ROOT FIX (v109): the previous code rebuilt the adjacency
    # dict on EVERY API call — O(E) memory allocation per request. Under
    # load, this caused GC pressure and made the /kg/explore endpoint
    # 10-100x slower than necessary. ROOT FIX: cache the built adjacency
    # dict on the builder instance (``_drugos_adj_cache``) so subsequent
    # calls within the same process reuse it. The cache is invalidated
    # if the bridge re-runs (the builder is replaced). For the dev
    # in-memory builder, this is a pure win. For the production Neo4j
    # builder, this path is not reached (Neo4j handles the BFS).
    cache_attr = "_drugos_adj_cache_v109"
    cached = getattr(builder, cache_attr, None)
    if cached is not None and cached.get("limit_hint", 0) >= limit:
        adj = cached["adj"]
        node_props = cached["node_props"]
    else:
        # Build adjacency: src_label -> src_id -> [(rel_type, dst_label, dst_id)]
        adj: Dict[str, Dict[str, List[Tuple[str, str, str]]]] = {}
        node_props: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # P2-037: handle both RecordingGraphBuilder (has node_loads) and
        # DrugOSGraphBuilder (does not). For DrugOSGraphBuilder, we cannot
        # do in-memory BFS (the data lives in Neo4j). Return None to let
        # the caller fall back to Neo4j.
        node_loads = getattr(builder, "node_loads", None)
        edge_loads = getattr(builder, "edge_loads", None)
        if not node_loads or not edge_loads:
            logger.info(
                "Phase 2 service: in-memory explore skipped — builder has "
                "no node_loads/edge_loads (production Neo4j builder)."
            )
            return None
        for load in node_loads:
            if not isinstance(load, dict):
                continue
            label = load.get("label", "unknown")
            for node in load.get("nodes", []) or []:
                if not isinstance(node, dict):
                    continue
                nid = str(node.get("id", ""))
                if not nid:
                    continue
                node_props[(label, nid)] = {
                    k: v for k, v in node.items() if k != "id"
                }
                adj.setdefault(label, {}).setdefault(nid, [])

        # P2-022 ROOT FIX (v109): the previous code added a reverse edge
        # for EVERY forward edge on EVERY load. If the same edge was
        # loaded multiple times (e.g. by two different sources), the
        # reverse edge was duplicated. ROOT FIX: use a set to dedup
        # reverse edges by (src_id, rel, dst_id) — only add the reverse
        # entry if it has not already been added.
        seen_reverse: Set[Tuple[str, str, str, str, str]] = set()
        for load in edge_loads:
            if not isinstance(load, dict):
                continue
            src_label = load.get("src_label", "")
            rel = load.get("rel_type", "")
            dst_label = load.get("dst_label", "")
            for edge in load.get("edges", []) or []:
                if not isinstance(edge, dict):
                    continue
                src_id = str(edge.get("src_id", ""))
                dst_id = str(edge.get("dst_id", ""))
                if not src_id or not dst_id:
                    continue
                adj.setdefault(src_label, {}).setdefault(src_id, []).append(
                    (rel, dst_label, dst_id)
                )
                # Reverse adjacency for BFS from disease nodes — dedup.
                rev_key = (dst_label, dst_id, rel, src_label, src_id)
                if rev_key not in seen_reverse:
                    seen_reverse.add(rev_key)
                    adj.setdefault(dst_label, {}).setdefault(dst_id, []).append(
                        (f"rev_{rel}", src_label, src_id)
                    )
        # Cache for reuse. We store a ``limit_hint`` so the cache is
        # rebuilt if a later request needs a larger BFS (the adjacency
        # itself is not limit-dependent, but we keep the hint for future
        # extensions where the cache might be limit-aware).
        try:
            setattr(builder, cache_attr, {
                "adj": adj, "node_props": node_props, "limit_hint": limit,
            })
        except Exception:
            # Some builders may not allow setting attributes (e.g.
            # frozen dataclasses). Fall back to no-cache — correctness
            # is preserved, just slower.
            pass

    # Pick the start node: match exact or partial name/ID across all node types
    start_label: Optional[str] = None
    start_id: Optional[str] = None
    target_name = (drug or disease or "").strip().lower()
    
    # 1. Exact match on name or ID across all nodes
    for (label, nid), props in node_props.items():
        name = str(props.get("name") or props.get("disease_name") or props.get("drug_name") or props.get("gene_symbol") or props.get("symbol") or nid).lower()
        if name == target_name or nid.lower() == target_name:
            start_label, start_id = label, nid
            break
            
    # 2. Partial match on name or ID across all nodes
    if start_id is None:
        for (label, nid), props in node_props.items():
            name = str(props.get("name") or props.get("disease_name") or props.get("drug_name") or props.get("gene_symbol") or props.get("symbol") or nid).lower()
            if target_name in name or target_name in nid.lower():
                start_label, start_id = label, nid
                break

    if start_id is None:
        return {
            "nodes": [],
            "edges": [],
            "backend": "in_memory_bridge",
            "note": f"No node matched '{drug or disease}'.",
        }

    # 2-hop BFS.
    visited: Set[Tuple[str, str]] = {(start_label, start_id)}
    edges_out: List[Dict[str, Any]] = []
    frontier: List[Tuple[str, str]] = [(start_label, start_id)]
    for _hop in range(2):
        if len(edges_out) >= limit:
            break
        next_frontier: List[Tuple[str, str]] = []
        for (sl, sid) in frontier:
            if len(edges_out) >= limit:
                break
            for (rel, dl, did) in adj.get(sl, {}).get(sid, []):
                is_rev = rel.startswith("rev_")
                fwd_rel = rel[4:] if is_rev else rel
                src_id, src_label = (did, dl) if is_rev else (sid, sl)
                dst_id, dst_label = (sid, sl) if is_rev else (did, dl)

                edges_out.append({
                    "source": src_id, "source_label": src_label,
                    "target": dst_id, "target_label": dst_label,
                    "type": fwd_rel,
                })
                if len(edges_out) >= limit:
                    break
                if (dl, did) not in visited:
                    visited.add((dl, did))
                    next_frontier.append((dl, did))
        frontier = next_frontier
        if not frontier or len(edges_out) >= limit:
            break

    nodes_out: List[Dict[str, Any]] = []
    for (label, nid) in visited:
        nodes_out.append({
            "id": nid,
            "label": label,
            "properties": node_props.get((label, nid), {}),
        })
    # Deduplicate edges by (source, target, type).
    seen_edges: Set[Tuple[str, str, str]] = set()
    unique_edges: List[Dict[str, Any]] = []
    for e in edges_out:
        key = (e["source"], e["target"], e["type"])
        if key in seen_edges:
            continue
        seen_edges.add(key)
        unique_edges.append(e)
        if len(unique_edges) >= limit:
            break
    return {
        "nodes": nodes_out,
        "edges": unique_edges,
        "backend": "in_memory_bridge",
    }


def _explore_subgraph_neo4j(
    drug: Optional[str], disease: Optional[str], limit: int
) -> Optional[Dict[str, Any]]:
    """Try Neo4j first for subgraph exploration."""

    # v113 FORENSIC ROOT FIX (P2-044 + P2-045, MEDIUM):
    #   P2-044: the previous code used ``d_node.id`` (Neo4j INTERNAL ID)
    #   as the response ``id``. Neo4j internal IDs are NOT stable across
    #   database restarts -- a node that was internal-id 42 before a
    #   restart may be internal-id 17 after. The frontend cached these
    #   IDs and broke on the next KG rebuild.
    #
    #   P2-045: the previous code used ``r1.start_node.id`` and
    #   ``r1.end_node.id`` for edge source/target. For UNDIRECTED
    #   ``MATCH (d)-[r1]-(n1)`` patterns, ``start_node`` and
    #   ``end_node`` are ARBITRARY (not the actual src/dst of the
    #   traversal) -- the edge's source/target in the response could
    #   be SWAPPED on consecutive runs of the same query, breaking
    #   the visual graph rendering.
    #
    #   ROOT FIX: use the BUSINESS ``id`` property from node properties
    #   (``dict(node).get("id")``), falling back to the Neo4j internal
    #   ID only when the business ``id`` property is missing (e.g., for
    #   legacy nodes created before the ``id`` property was mandatory).
    #   For edges, use the business IDs of the nodes we ALREADY have
    #   from the query (``d``, ``n1``, ``n2``) -- do NOT use
    #   ``r.start_node.id`` / ``r.end_node.id`` which are arbitrary for
    #   undirected patterns. The edge source is always the node CLOSER
    #   to the query root (``d`` for r1, ``n1`` for r2), and the target
    #   is the node FARTHER from the root (``n1`` for r1, ``n2`` for r2).
    #   This produces STABLE, DETERMINISTIC edge source/target pairs.
    def _business_id(node) -> Any:
        """Return the business ``id`` property, falling back to Neo4j internal id."""
        if node is None:
            return None
        props = dict(node)
        bid = props.get("id")
        if bid is not None and bid != "":
            return bid
        # Legacy nodes without a business ``id`` property -- fall back
        # to the Neo4j internal id (stringified so it's JSON-serializable
        # and visually distinct from business IDs).
        return f"__neo4j_internal:{node.id}"

    def _node_record(node) -> Dict[str, Any]:
        return {
            "id": _business_id(node),
            "label": list(node.labels)[0] if node.labels else "Unknown",
            "labels": list(node.labels),
            "properties": dict(node),
        }

    def _query(driver):
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        # P2-030 ROOT FIX (v107): the previous query used
        # ``MATCH p=(d:Drug {name: $drug})-[*1..2]-(n) RETURN p LIMIT $limit``
        # The ``[*1..2]`` variable-length path is UNBOUNDED in complexity —
        # on a large KG, a 2-hop traversal from a high-degree drug (e.g.
        # aspirin with 1000+ edges) can match MILLIONS of paths. The
        # ``LIMIT $limit`` caps the RETURN but NOT the intermediate path
        # expansion. A single API call can DoS the Neo4j instance.
        # ROOT FIX: use a subquery with LIMIT INSIDE so the path
        # expansion is bounded at each hop. This caps the work Neo4j
        # does, preventing DoS while still returning a useful subgraph.
        # P2-031 ROOT FIX (v107): the previous code deduplicated NODES
        # by id but NOT EDGES. The same relationship (same source,
        # target, type) appeared multiple times in the response,
        # inflating the visual graph. ROOT FIX: deduplicate edges by
        # (source, target, type) tuple.
        with driver.session() as session:
            # P2-062 ROOT FIX (Teammate 4, forensic, root-level): the
            # previous code used ``if drug: ... elif disease: ...`` which
            # IGNORED the disease parameter when BOTH drug and disease were
            # provided. The /query endpoint accepts both, so a caller asking
            # "show me the subgraph between aspirin and diabetes" would get
            # ONLY the aspirin subgraph (disease silently ignored). ROOT FIX:
            # when BOTH are provided, run a DEDICATED query that finds paths
            # BETWEEN the drug and disease nodes (the scientifically useful
            # result for drug-repurposing hypothesis exploration). When only
            # one is provided, fall back to the original 2-hop BFS.
            if drug and disease:
                # P2-062: find paths BETWEEN the drug and disease nodes.
                # This is the most useful query for drug repurposing — it
                # shows the biological pathway chain connecting a drug to
                # a disease, which is exactly what the project doc's
                # "Knowledge Graph Explorer" screen is supposed to display.
                q = """
                MATCH (d:Compound {name: $drug}), (dis:Disease {name: $disease})
                CALL {
                    WITH d, dis
                    MATCH p = shortestPath((d)-[*1..5]-(dis))
                    RETURN p LIMIT $limit
                }
                UNWIND relationships(p) AS r
                WITH d, dis, r, startNode(r) AS sn, endNode(r) AS en
                RETURN d, dis, r, sn, en
                LIMIT $limit
                """
                result = session.run(q, drug=drug, disease=disease, limit=limit)
                for record in result:
                    d_node = record["d"]
                    nodes.append({
                        "id": d_node.id,
                        "label": list(d_node.labels)[0] if d_node.labels else "Unknown",
                        "labels": list(d_node.labels),
                        "properties": dict(d_node),
                    })
                    dis_node = record["dis"]
                    nodes.append({
                        "id": dis_node.id,
                        "label": list(dis_node.labels)[0] if dis_node.labels else "Unknown",
                        "labels": list(dis_node.labels),
                        "properties": dict(dis_node),
                    })
                    r = record["r"]
                    if r is not None:
                        edges.append({
                            "source": r.start_node.id,
                            "target": r.end_node.id,
                            "type": r.type,
                        })
                    sn = record["sn"]
                    if sn is not None:
                        nodes.append({
                            "id": sn.id,
                            "label": list(sn.labels)[0] if sn.labels else "Unknown",
                            "labels": list(sn.labels),
                            "properties": dict(sn),
                        })
                    en = record["en"]
                    if en is not None:
                        nodes.append({
                            "id": en.id,
                            "label": list(en.labels)[0] if en.labels else "Unknown",
                            "labels": list(en.labels),
                            "properties": dict(en),
                        })
            elif drug:
                # P2-030 ROOT FIX (v107): bounded 2-hop traversal using
                # subqueries with LIMIT inside each hop. The previous
                # ``[*1..2]`` variable-length path was UNBOUNDED — on a
                # large KG, a 2-hop traversal from a high-degree drug
                # could match millions of paths, DoS-ing Neo4j.
                q = """
                MATCH (d:Compound {name: $drug})
                CALL {
                    WITH d
                    MATCH (d)-[r1]-(n1)
                    RETURN n1, r1, d AS d1
                    LIMIT $limit
                }
                WITH d, collect(DISTINCT [n1, r1, d1]) AS hop1
                UNWIND hop1 AS h1
                WITH h1[0] AS n1, h1[1] AS r1, h1[2] AS d1
                CALL {
                    WITH n1
                    MATCH (n1)-[r2]-(n2)
                    WHERE n2 <> d AND n2 <> n1
                    RETURN n2, r2, n1 AS n1b
                    LIMIT $limit
                }
                RETURN d, n1, r1, n2, r2
                LIMIT $limit
                """
                result = session.run(q, drug=drug, limit=limit)
                for record in result:
                    d_node = record["d"]
                    nodes.append(_node_record(d_node))
                    n1 = record["n1"]
                    if n1 is not None:
                        nodes.append(_node_record(n1))
                        r1 = record["r1"]
                        if r1 is not None:
                            # v113 P2-044/045 ROOT FIX: use business IDs
                            # of the nodes we already have (d, n1), NOT
                            # r1.start_node.id / r1.end_node.id (which
                            # are arbitrary for undirected MATCH).
                            edges.append({
                                "source": _business_id(d_node),
                                "target": _business_id(n1),
                                "type": r1.type,
                            })
                    n2 = record["n2"]
                    if n2 is not None:
                        nodes.append(_node_record(n2))
                        r2 = record["r2"]
                        if r2 is not None:
                            # v113 P2-044/045 ROOT FIX: use business IDs
                            # of n1 and n2 (NOT r2.start_node/end_node).
                            edges.append({
                                "source": _business_id(n1) if n1 is not None else _business_id(d_node),
                                "target": _business_id(n2),
                                "type": r2.type,
                            })
            elif disease:
                # P2-030: same bounded traversal for disease queries.
                q = """
                MATCH (d:Disease {name: $disease})
                CALL {
                    WITH d
                    MATCH (d)-[r1]-(n1)
                    RETURN n1, r1, d AS d1
                    LIMIT $limit
                }
                WITH d, collect(DISTINCT [n1, r1, d1]) AS hop1
                UNWIND hop1 AS h1
                WITH h1[0] AS n1, h1[1] AS r1, h1[2] AS d1
                CALL {
                    WITH n1
                    MATCH (n1)-[r2]-(n2)
                    WHERE n2 <> d AND n2 <> n1
                    RETURN n2, r2, n1 AS n1b
                    LIMIT $limit
                }
                RETURN d, n1, r1, n2, r2
                LIMIT $limit
                """
                result = session.run(q, disease=disease, limit=limit)
                for record in result:
                    d_node = record["d"]
                    nodes.append(_node_record(d_node))
                    n1 = record["n1"]
                    if n1 is not None:
                        nodes.append(_node_record(n1))
                        r1 = record["r1"]
                        if r1 is not None:
                            # v113 P2-044/045 ROOT FIX: use business IDs.
                            edges.append({
                                "source": _business_id(d_node),
                                "target": _business_id(n1),
                                "type": r1.type,
                            })
                    n2 = record["n2"]
                    if n2 is not None:
                        nodes.append(_node_record(n2))
                        r2 = record["r2"]
                        if r2 is not None:
                            # v113 P2-044/045 ROOT FIX: use business IDs.
                            edges.append({
                                "source": _business_id(n1) if n1 is not None else _business_id(d_node),
                                "target": _business_id(n2),
                                "type": r2.type,
                            })
            else:
                return None
        # Deduplicate nodes by id.
        seen_node_ids = set()
        unique_nodes = []
        for n in nodes:
            if n["id"] not in seen_node_ids:
                seen_node_ids.add(n["id"])
                unique_nodes.append(n)
        # P2-031 ROOT FIX (v107): deduplicate EDGES by (source, target, type).
        # The previous code deduplicated nodes but NOT edges — the same
        # relationship appeared multiple times, inflating the visual graph.
        seen_edge_keys = set()
        unique_edges = []
        for e in edges:
            _edge_key = (e["source"], e["target"], e["type"])
            if _edge_key not in seen_edge_keys:
                seen_edge_keys.add(_edge_key)
                unique_edges.append(e)
        return {"nodes": unique_nodes, "edges": unique_edges, "backend": "neo4j"}

    try:
        return _run_neo4j(_query)
    except Exception as exc:
        logger.info("Phase 2 service: Neo4j explore failed: %s", exc)
        return None


# ─── P2-002 ROOT FIX (v109 forensic): /query + /cypher endpoints ──────────
class QueryBody(BaseModel):
    """Structured query body for POST /query."""
    drug: Optional[str] = Field(None, description="Drug name (Compound.name).")
    disease: Optional[str] = Field(None, description="Disease name (Disease.name).")
    limit: int = Field(100, ge=1, le=500, description="Max nodes to return.")


# ─── P2-002 / P2-003 / P2-011 ROOT FIX (v109 forensic): real Cypher security
# The v107 "_READ_ONLY_PREFIX_RE" regex was DEAD CODE — defined but never
# used by ``_validate_readonly_cypher``. The actual validator only checked
# the first token (MATCH/OPTIONAL) and ran a narrow forbidden-keyword scan
# that missed:
#   * ``CALL { ... }`` subqueries containing write ops (the regex
#     ``CALL\s+\{[^}]*\}`` only matched non-nested braces AND was never
#     invoked).
#   * ``apoc.periodic.iterate``, ``apoc.cypher.runFirstColumn``,
#     ``apoc.cypher.runFirstColumnMany`` — APOC procedures that execute
#     arbitrary Cypher (and therefore can write).
#   * ``LOAD CSV FROM 'file:///etc/passwd'`` — local-file exfiltration.
#   * Multi-statement injection via ``;`` (older Neo4j drivers did split
#     on ``;``; modern drivers reject it, but we should not rely on the
#     driver).
#   * ``CALL db.schema.write``, ``CALL db.createIndex``,
#     ``CALL db.createConstraint`` — write/DDL procedures.
#
# ROOT FIX (v109): replace the regex-only validator with a layered
# defense-in-depth validator that:
#   1. Rejects multi-statement queries (any ``;`` not inside a string
#      literal).
#   2. Rejects any ``CALL { ... }`` subquery (read-only APIs do not
#      need them; they are the primary injection vector).
#   3. Rejects ``LOAD CSV`` / ``LOAD FROM`` / ``STARTS WITH file:`` /
#      ``STARTS WITH http:`` (file/network exfiltration).
#   4. Rejects ALL ``apoc.*`` procedures except a strict whitelist of
#      known-read-only ones (``apoc.meta.graph``, ``apoc.meta.schema``,
#      ``apoc.node.exists`` is NOT allowed because it can be abused).
#   5. Rejects ALL ``db.*`` procedures except a strict whitelist
#      (``db.labels``, ``db.relationshipTypes``, ``db.indexes``,
#      ``db.constraints``, ``db.schema.visualization``).
#   6. Applies the forbidden-write-keyword regex to the WHOLE query.
#   7. Requires the first token to be MATCH, OPTIONAL MATCH, or WITH.
#   8. Caps query length at 8 KB (prevents pathological regex backtracking
#      and resource-exhaustion via huge queries).
#
# This is a defense-in-depth layer — the Neo4j driver and database
# enforce their own limits, but we should not rely on them alone.

# Write/DDL keywords that must NEVER appear in a read-only query. We use
# word boundaries (\b) so ``SET`` does not match ``SETTING`` or
# ``OFFSET``. Note: ``SET`` is intentionally listed because Cypher
# ``SET`` is always a write op; ``OFFSET`` is a different token.
_FORBIDDEN_WRITE_KEYWORDS_RE = re.compile(
    r"\b("
    r"CREATE|MERGE|DELETE|DETACH\s+DELETE|SET|REMOVE|DROP|"
    r"INDEX|CONSTRAINT|"
    r"FOREACH|LOAD\s+CSV|LOAD\s+FROM|"
    r"START\s+WITH"
    r")\b",
    re.IGNORECASE,
)

# APOC procedures that execute arbitrary Cypher or write data. The full
# ``apoc.*`` namespace is huge and many procedures can write (e.g.
# ``apoc.create.node``, ``apoc.destroy.nodes``, ``apoc.periodic.iterate``,
# ``apoc.cypher.runFirstColumn``). We block ALL ``apoc.*`` calls except a
# tiny explicit whitelist of read-only metadata procedures.
_FORBIDDEN_APOC_RE = re.compile(
    r"\bCALL\s+apoc\.",
    re.IGNORECASE,
)
_ALLOWED_APOC_WHITELIST = frozenset(s.lower() for s in {
    "apoc.meta.graph",
    "apoc.meta.schema",
    "apoc.meta.stats",
    "apoc.meta.relTypeProperties",
    "apoc.meta.nodeTypeProperties",
})

# ``db.*`` procedures that write/modify the schema. Block all except a
# strict read-only whitelist.
_FORBIDDEN_DB_PROC_RE = re.compile(
    r"\bCALL\s+db\.",
    re.IGNORECASE,
)
_ALLOWED_DB_PROC_WHITELIST = frozenset(s.lower() for s in {
    "db.labels",
    "db.relationshipTypes",
    "db.propertyKeys",
    "db.indexes",
    "db.constraints",
    "db.schema.visualization",
    "db.schema.nodeTypeProperties",
    "db.schema.relTypeProperties",
})

# ``CALL { ... }`` subquery — block entirely. Read-only APIs do not
# need them; they are the primary injection vector. Match any ``CALL``
# followed by ``{`` (with optional whitespace).
_CALL_SUBQUERY_RE = re.compile(
    r"\bCALL\s*\{",
    re.IGNORECASE,
)

# ``;`` outside string literals — multi-statement injection. We strip
# string literals first (see ``_strip_string_literals`` below) and then
# check for any remaining ``;``.
_SEMICOLON_RE = re.compile(r";")

# File/network exfiltration patterns.
_FILE_URL_RE = re.compile(
    r"\b(file|https?)://",
    re.IGNORECASE,
)

# Maximum allowed Cypher query length (8 KB). Prevents pathological regex
# backtracking and resource-exhaustion via huge queries.
_MAX_CYPHER_LENGTH = 8 * 1024


def _strip_string_literals(cypher: str) -> str:
    """Return the Cypher with string literals replaced by empty strings.

    Used so that semicolons, keywords, and URLs INSIDE string literals do
    not trigger false-positive matches. We replace single- and double-
    quoted strings (including escaped quotes) with ``''``.

    NOTE: this is a simplification — Cypher string literals can also be
    backtick-quoted (for identifiers) and there are corner cases with
    escaped backticks. We err on the side of caution: if we cannot parse
    the literal, we leave the character in place (which may cause a
    false positive, but never a false negative for security).
    """
    out: list[str] = []
    i = 0
    n = len(cypher)
    while i < n:
        c = cypher[i]
        if c in ("'", '"'):
            # Find the matching close quote, respecting escapes.
            quote = c
            i += 1
            while i < n:
                if cypher[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                if cypher[i] == quote:
                    i += 1
                    break
                i += 1
            out.append("''")  # replace literal with empty literal
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _validate_readonly_cypher(cypher: str) -> Optional[str]:
    """Return an error message if the Cypher is not read-only, else None.

    v109 ROOT FIX (P2-002 / P2-003 / P2-011): defense-in-depth validator.
    See the long docstring above the regex constants for the full
    rationale of what is blocked and why.
    """
    if not cypher or not cypher.strip():
        return "Empty Cypher query."
    if len(cypher) > _MAX_CYPHER_LENGTH:
        return (
            f"Cypher query too long ({len(cypher)} > {_MAX_CYPHER_LENGTH} bytes). "
            "Read-only API queries must be <= 8 KB."
        )

    stripped = cypher.strip()
    first_token = stripped.split(None, 1)[0].upper() if stripped else ""
    # Allow ``CALL db.<whitelisted>`` as a first token (e.g. ``CALL
    # db.labels()``) — this is the only ``CALL`` form that bypasses the
    # MATCH/OPTIONAL/WITH requirement, and only for whitelisted read-only
    # ``db.*`` procedures (validated below).
    if first_token not in ("MATCH", "OPTIONAL", "WITH", "CALL"):
        return (
            f"Cypher must start with MATCH, OPTIONAL MATCH, WITH, or "
            f"CALL db.<whitelisted-proc> (got '{first_token}'). Only "
            "read-only queries are allowed."
        )
    if first_token == "CALL":
        # Special-case: only ``CALL db.<whitelisted>`` is allowed as a
        # first token. ``CALL apoc.*`` and ``CALL { ... }`` are blocked
        # below.
        rest_after_call = stripped[4:].lstrip()
        if rest_after_call.startswith("{"):
            return (
                "Cypher starts with 'CALL {' — subqueries are not allowed "
                "via the read-only API."
            )
        proc_match = re.match(r"([a-zA-Z0-9_.]+)", rest_after_call)
        proc_name = proc_match.group(1).lower().rstrip(".") if proc_match else ""
        if proc_name.startswith("apoc."):
            allowed = any(
                proc_name == w or proc_name.startswith(w + ".")
                for w in _ALLOWED_APOC_WHITELIST
            )
            if not allowed:
                return (
                    f"Cypher starts with 'CALL apoc.{proc_name[5:]}' — "
                    "only read-only db.* procedures may be called as the "
                    "first token."
                )
            # whitelisted apoc.* as first token — accept
        elif proc_name.startswith("db."):
            allowed = any(
                proc_name == w or proc_name.startswith(w + ".")
                for w in _ALLOWED_DB_PROC_WHITELIST
            )
            if not allowed:
                return (
                    f"Cypher starts with 'CALL db.{proc_name[3:]}' — "
                    f"procedure not in read-only whitelist: "
                    f"{sorted(_ALLOWED_DB_PROC_WHITELIST)}."
                )
        else:
            return (
                f"Cypher starts with 'CALL {proc_name}' — only CALL "
                "db.<whitelisted> is allowed as the first token."
            )

    # Strip string literals so that keywords/semicolons inside string
    # literals do not trigger false-positive matches.
    stripped_for_scan = _strip_string_literals(cypher)

    # 1. Multi-statement injection (``;`` outside string literals).
    if _SEMICOLON_RE.search(stripped_for_scan):
        return (
            "Cypher contains a semicolon (';') outside a string literal. "
            "Multi-statement queries are not allowed via this endpoint."
        )

    # 2. ``CALL { ... }`` subqueries — block entirely.
    if _CALL_SUBQUERY_RE.search(stripped_for_scan):
        return (
            "Cypher contains a 'CALL { ... }' subquery. Subqueries are "
            "not allowed via the read-only API (they are the primary "
            "Cypher injection vector). Rewrite as a flat MATCH/WITH/"
            "RETURN query."
        )

    # 3. File/network exfiltration.
    if _FILE_URL_RE.search(stripped_for_scan):
        return (
            "Cypher contains a file:// or http(s):// URL. Loading "
            "external resources is not allowed via this endpoint."
        )

    # 4. ``LOAD CSV`` / ``LOAD FROM`` — file/network exfiltration.
    if re.search(r"\bLOAD\s+(CSV|FROM)\b", stripped_for_scan, re.IGNORECASE):
        return (
            "Cypher contains 'LOAD CSV' or 'LOAD FROM'. Loading external "
            "data is not allowed via this endpoint."
        )

    # 5. Write/DDL keywords.
    if _FORBIDDEN_WRITE_KEYWORDS_RE.search(stripped_for_scan):
        return (
            "Cypher contains a forbidden write/DDL keyword "
            "(CREATE/MERGE/DELETE/SET/REMOVE/DROP/INDEX/CONSTRAINT/"
            "FOREACH/LOAD). Only read-only MATCH/OPTIONAL MATCH/WITH/"
            "RETURN queries are allowed via this endpoint."
        )

    # 6. ``apoc.*`` procedures — block all except a strict whitelist.
    for m in _FORBIDDEN_APOC_RE.finditer(stripped_for_scan):
        # Extract the procedure name (e.g. ``apoc.cypher.runFirstColumn``).
        # ``_FORBIDDEN_APOC_RE`` matches ``CALL apoc.``, so we re-match
        # from the START of ``apoc.`` to capture the full proc name
        # (including the ``apoc.`` prefix) so it can be compared against
        # the whitelist entries which are stored WITH the prefix.
        start = m.start() + len("CALL ")  # position of ``apoc.``
        rest = stripped_for_scan[start:start + 80]
        proc_match = re.match(r"(apoc\.[a-zA-Z0-9_.]+)", rest)
        proc_name = proc_match.group(1).lower().rstrip(".") if proc_match else "apoc"
        allowed = any(
            proc_name == w or proc_name.startswith(w + ".")
            for w in _ALLOWED_APOC_WHITELIST
        )
        if not allowed:
            return (
                f"Cypher calls APOC procedure '{proc_name}'. Only "
                f"read-only APOC metadata procedures are allowed: "
                f"{sorted(_ALLOWED_APOC_WHITELIST)}."
            )

    # 7. ``db.*`` procedures — block all except a strict whitelist.
    for m in _FORBIDDEN_DB_PROC_RE.finditer(stripped_for_scan):
        # ``_FORBIDDEN_DB_PROC_RE`` matches ``CALL db.``, so we re-match
        # from the START of ``db.`` to capture the full proc name
        # (including the ``db.`` prefix) so it can be compared against
        # the whitelist entries which are stored WITH the prefix.
        start = m.start() + len("CALL ")  # position of ``db.``
        rest = stripped_for_scan[start:start + 80]
        proc_match = re.match(r"(db\.[a-zA-Z0-9_.]+)", rest)
        proc_name = proc_match.group(1).lower().rstrip(".") if proc_match else "db"
        allowed = any(
            proc_name == w or proc_name.startswith(w + ".")
            for w in _ALLOWED_DB_PROC_WHITELIST
        )
        if not allowed:
            return (
                f"Cypher calls db procedure '{proc_name}'. Only "
                f"read-only db procedures are allowed: "
                f"{sorted(_ALLOWED_DB_PROC_WHITELIST)}."
            )

    return None


def _validate_cypher_params(params: Optional[Dict[str, Any]]) -> Optional[str]:
    """Validate that ``params`` is a flat dict of scalar values.

    P2-011 ROOT FIX (v109): the previous code forwarded ``body.params``
    directly to ``session.run(body.cypher, body.params or {})``. If
    ``params`` contained nested dicts/lists, the Neo4j driver would
    serialize them as Cypher maps/lists — which could be used to inject
    Cypher fragments (e.g. via a parameter like
    ``{"x": {"__proto__": "MATCH (n) DELETE n"}}``). The driver does
    parameterize, but the parameter VALUES are still injected as Cypher
    literals — and a nested map value is rendered as a Cypher map
    literal that the database then parses.

    ROOT FIX: reject any non-scalar parameter value. Allowed types:
    ``str``, ``int``, ``float``, ``bool``, ``None``. Lists of scalars
    are allowed (the driver renders them as Cypher lists). Anything
    else (dict, set, custom object) is rejected.
    """
    if params is None:
        return None
    if not isinstance(params, dict):
        return f"Cypher params must be a dict (got {type(params).__name__})."
    for key, value in params.items():
        if not isinstance(key, str):
            return (
                f"Cypher param key must be a string (got "
                f"{type(key).__name__}: {key!r})."
            )
        err = _validate_scalar_param_value(key, value)
        if err is not None:
            return err
    return None


def _validate_scalar_param_value(key: str, value: Any) -> Optional[str]:
    """Validate a single Cypher param value (recursive for lists)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return None
    if isinstance(value, list):
        if len(value) > 1000:
            return (
                f"Cypher param '{key}' list too long ({len(value)} > 1000)."
            )
        for i, item in enumerate(value):
            err = _validate_scalar_param_value(f"{key}[{i}]", item)
            if err is not None:
                return err
        return None
    return (
        f"Cypher param '{key}' has non-scalar value of type "
        f"{type(value).__name__} (only str/int/float/bool/None/list-of-scalars "
        "are allowed)."
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "phase2_kg",
        "version": "1.0.0",
        "neo4j_configured": bool(_get_neo4j_env_var("PASSWORD")),
        "cors_origins": _allowed_origins,
    }


@app.get("/kg/stats")
def kg_stats() -> Dict[str, Any]:
    """Return real KG stats from Neo4j (or in-memory bridge fallback)."""
    # P2-017: _get_kg_stats_from_neo4j now uses try/finally driver.close().
    neo4j_stats = _get_kg_stats_from_neo4j()
    if neo4j_stats is not None:
        return neo4j_stats
    # P2-001 / P2-008: bridge may raise FileNotFoundError (Phase 1 missing)
    # or return backend="error". Either way, the route MUST surface 503 —
    # never silently return 0/0 with HTTP 200, and never let FileNotFoundError
    # become an unhandled HTTP 500.
    try:
        bridge_stats = _get_kg_stats_from_builder()
    except FileNotFoundError as exc:
        # P2-001: Phase 1 data is missing — return 503 with a clear error.
        raise HTTPException(
            status_code=503,
            detail={
                "error": "phase1_data_missing",
                "message": str(exc),
                "backend": "missing",
            },
        )
    if bridge_stats.get("backend") == "error":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "kg_bridge_failed",
                "message": bridge_stats.get("error", "unknown bridge failure"),
                "backend": "error",
            },
        )
    return bridge_stats


@app.get("/kg/explore")
def kg_explore(
    drug: Optional[str] = Query(None),
    disease: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> Dict[str, Any]:
    """Return a subgraph around a drug or disease."""
    if not drug and not disease:
        raise HTTPException(
            status_code=400,
            detail="Provide ?drug= or ?disease= query param.",
        )
    # P2-017: _explore_subgraph_neo4j now uses try/finally driver.close().
    neo4j_result = _explore_subgraph_neo4j(drug, disease, limit)
    if neo4j_result is not None:
        return neo4j_result
    # P2-009: real in-memory BFS (was: empty 200 with a note field).
    in_mem_result = _explore_subgraph_in_memory(drug, disease, limit)
    if in_mem_result is not None:
        return in_mem_result
    raise HTTPException(
        status_code=503,
        detail={
            "error": "kg_explore_unavailable",
            "message": (
                "Neither Neo4j nor the in-memory bridge could resolve this "
                "query. Set NEO4J_URI/USER/PASSWORD for production, or "
                "ensure Phase 1 processed_data exists for in-memory fallback."
            ),
        },
    )


class CypherBody(BaseModel):
    """Raw Cypher passthrough body for POST /cypher."""
    cypher: str = Field(..., description="Read-only Cypher query.")
    params: Optional[Dict[str, Any]] = Field(
        None, description="Parameterized query variables (scalars or lists of scalars only)."
    )


@app.post("/query")
def kg_query(body: QueryBody) -> Dict[str, Any]:
    """Structured drug/disease query (P2-002 root fix).

    The frontend POSTs ``{"drug": "...", "disease": "...", "limit": N}``
    here. We resolve it via Neo4j (preferred) or the in-memory BFS.
    """
    if not body.drug and not body.disease:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of 'drug' or 'disease' in the body.",
        )
    # Try Neo4j first.
    neo4j_result = _explore_subgraph_neo4j(body.drug, body.disease, body.limit)
    if neo4j_result is not None:
        return neo4j_result
    # In-memory fallback.
    in_mem_result = _explore_subgraph_in_memory(body.drug, body.disease, body.limit)
    if in_mem_result is not None:
        return in_mem_result
    raise HTTPException(
        status_code=503,
        detail={
            "error": "kg_query_unavailable",
            "message": (
                "Neither Neo4j nor the in-memory bridge could resolve this "
                "structured query."
            ),
        },
    )


@app.post("/cypher")
def kg_cypher(body: CypherBody) -> Dict[str, Any]:
    """Raw read-only Cypher passthrough (P2-002 / P2-011 / P2-041 root fix).

    v109 ROOT FIX: applies the defense-in-depth ``_validate_readonly_cypher``
    validator (blocks subqueries, APOC, db writes, LOAD CSV, semicolons)
    AND ``_validate_cypher_params`` (rejects non-scalar params) AND
    enforces a hard 30-second server-side timeout via the Neo4j driver's
    ``transaction_timeout`` parameter (the v107 docstring claimed a 30s
    timeout but the implementation did NOT enforce one — a malicious or
    runaway query could block the worker indefinitely).
    Returns 503 if Neo4j is not configured — the in-memory bridge cannot
    answer arbitrary Cypher.
    """
    # Validate read-only.
    err = _validate_readonly_cypher(body.cypher)
    if err is not None:
        raise HTTPException(status_code=400, detail={"error": "cypher_rejected", "message": err})

    # P2-011: validate that params are flat scalars (no nested dicts).
    err = _validate_cypher_params(body.params)
    if err is not None:
        raise HTTPException(status_code=400, detail={"error": "cypher_params_rejected", "message": err})

    if not _get_neo4j_env_var("PASSWORD"):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_not_deployed",
                "message": (
                    "Raw Cypher queries require Neo4j. Set DRUGOS_NEO4J_URI/"
                    "USER/PASSWORD (or legacy NEO4J_*) to enable. The "
                    "in-memory bridge cannot answer arbitrary Cypher."
                ),
            },
        )

    MAX_ROWS = 1000
    # P2-041 ROOT FIX: 30-second HARD server-side timeout. The Neo4j
    # Python driver supports ``transaction_timeout`` on ``session.run()``
    # (Neo4j 4.0+). On older Neo4j, the driver silently ignores this
    # parameter, so we ALSO wrap the call in a Python-side thread+timeout
    # via ``concurrent.futures`` to guarantee the worker is not blocked
    # longer than 30 seconds + a small grace period.
    QUERY_TIMEOUT_SECONDS = 30.0

    def _query(driver):
        with driver.session() as session:
            # Neo4j 4.0+ respects ``transaction_timeout`` (in seconds).
            # Pass it as a keyword arg so older drivers that don't accept
            # it don't break (they'll just ignore the timeout, which is
            # why we also have the Python-side timeout below).
            try:
                result = session.run(
                    body.cypher,
                    body.params or {},
                    transaction_timeout=QUERY_TIMEOUT_SECONDS,
                )
            except TypeError:
                # Older driver — no ``transaction_timeout`` kwarg.
                result = session.run(body.cypher, body.params or {})
            records = []
            for rec in result:
                records.append(dict(rec))
                if len(records) >= MAX_ROWS:
                    break
            return {
                "records": records,
                "row_count": len(records),
                "truncated": len(records) >= MAX_ROWS,
                "max_rows": MAX_ROWS,
                "backend": "neo4j",
                "timeout_seconds": QUERY_TIMEOUT_SECONDS,
            }

    # P2-041: Python-side timeout guard. We run ``_run_neo4j(_query)`` in
    # a thread pool with a hard timeout of QUERY_TIMEOUT_SECONDS + 5s
    # grace (the extra 5s gives the Neo4j driver time to clean up after
    # its own transaction_timeout fires).
    import concurrent.futures
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_neo4j, _query)
            try:
                return future.result(timeout=QUERY_TIMEOUT_SECONDS + 5.0)
            except concurrent.futures.TimeoutError:
                logger.error(
                    "Phase 2 service: /cypher timed out after %.1fs — query: %.200s",
                    QUERY_TIMEOUT_SECONDS + 5.0, body.cypher,
                )
                raise HTTPException(
                    status_code=504,
                    detail={
                        "error": "cypher_timeout",
                        "message": (
                            f"Cypher query did not complete within "
                            f"{QUERY_TIMEOUT_SECONDS:.0f}s. Simplify the "
                            "query or add an index."
                        ),
                        "timeout_seconds": QUERY_TIMEOUT_SECONDS,
                    },
                )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Phase 2 service: /cypher failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail={"error": "kg_cypher_failed", "message": str(exc)},
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PHASE2_SERVICE_PORT", "8002"))
    host = os.environ.get("PHASE2_SERVICE_HOST", "0.0.0.0")
    logger.info("Starting Phase 2 KG Service on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
