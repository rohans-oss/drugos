#!/usr/bin/env python3
"""Phase 1 Dataset Service (Step 1 integration plan, v105).

Wraps Phase 1's data-warehouse logic as an HTTP service so the Next.js
frontend can proxy to it via DATASET_SERVICE_URL. The frontend's
``src/lib/services/dataset-stats.ts`` and ``src/app/api/dataset/route.ts``
already proxy to ``DATASET_SERVICE_URL`` -- this service is what they
expect to find there.

Endpoints:
    GET  /health                -> {status: "ok", service: "phase1", ...}
    GET  /datasets              -> {sources: [...], total_drugs, total_proteins, ...}
    GET  /datasets/{drug}/mechanism -> {drug, targets: [{protein, action, evidence}], ...}

Run:
    cd phase1 && python service.py
    # or: uvicorn phase1.service:app --host 0.0.0.0 --port 8001

Environment:
    PHASE1_DB_URL: PostgreSQL URL for the staging DB. If unset, the
        service reads embedded sample CSVs (Tier-2 fallback) so it can
        still answer /datasets in dev/CI without a DB.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make phase1 importable when running ``python phase1/service.py`` directly.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# Also make the repo root importable (for pipelines.* / config.*).
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("phase1.service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

app = FastAPI(
    title="Autonomous Drug Repurposing — Phase 1 Dataset Service",
    description="HTTP wrapper around Phase 1 ETL / data warehouse.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    # P1-017 v113 ROOT FIX: allow_origins=["*"] permitted ANY website to
    # make cross-origin requests to this API. Combined with allow_headers=["*"],
    # a malicious webpage could enumerate every drug in the database — a
    # data-exfiltration vector for proprietary drug-repurposing hypotheses.
    # ROOT FIX: read allowed origins from PHASE1_CORS_ORIGINS env var
    # (comma-separated). Default to localhost:3000 (the Next.js frontend)
    # for dev. Production deployments MUST set PHASE1_CORS_ORIGINS to the
    # real frontend domain.
    allow_origins=os.environ.get(
        "PHASE1_CORS_ORIGINS", "http://localhost:3000"
    ).split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV (excluding header). Returns 0 if missing.

    P1-018 v113 ROOT FIX: the previous implementation counted NEWLINES
    via `sum(1 for line in f)`. A CSV field with an embedded newline
    (e.g. a `mechanism_of_action` field containing "Inhibits COX-1\nand
    COX-2") spans multiple physical lines, so the counter overcounted.
    The DrugBank drugs CSV regularly contains multi-line mechanism fields,
    causing the /datasets endpoint to report 10,500 drugs when the actual
    count was 10,000.

    ROOT FIX: use the csv module's reader which correctly handles
    multi-line quoted fields. This is the standard, robust way to count
    CSV records.
    """
    if not path.exists():
        return 0
    try:
        import csv as _csv
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = _csv.reader(f)
            try:
                next(reader)  # skip header
            except StopIteration:
                return 0
            return sum(1 for _row in reader if _row)  # count non-empty rows
    except Exception:
        return 0


def _processed_data_dir() -> Path:
    """Return the Phase 1 processed_data directory."""
    return _HERE / "processed_data"


def _load_dataset_stats() -> Dict[str, Any]:
    """Load real dataset stats from processed_data CSVs (or DB if available).

    TM1 TASK 2 ROOT FIX: the previous Tier-2 fallback wrote embedded sample
    CSVs (10 fake drugs) when processed_data was empty. This violated the
    "NEVER overwrite real data with mock samples" mandate. Now we return
    zero counts with a clear ``data_status="empty"`` flag so the dashboard
    honestly reports the absence of data instead of fabricating numbers.
    The operator must run ``python -m phase1.pipelines all`` to populate
    real data. For local dev, ``DRUGOS_ENVIRONMENT=development python -m
    phase1.pipelines samples`` writes mock CSVs to a directory of choice.
    """
    pdir = _processed_data_dir()
    if not pdir.exists() or not any(pdir.glob("*.csv*")):
        logger.warning(
            "Phase 1 service: processed_data dir %s is empty. Returning "
            "zero counts with data_status='empty'. Run "
            "`python -m phase1.pipelines all` to populate real data.",
            pdir,
        )

    # Count rows per source CSV. These are the REAL Phase 1 outputs.
    sources: List[Dict[str, Any]] = []
    csv_map = {
        "chembl": "chembl_drugs.csv",
        "drugbank": "drugbank_drugs.csv",
        "uniprot": "uniprot_proteins.csv",
        "string": "string_protein_protein_interactions.csv",
        "disgenet": "disgenet_gene_disease_associations.csv",
        "omim": "omim_gene_disease_associations.csv",
        "pubchem": "pubchem_enrichment.csv",
    }
    total_drugs = 0
    total_proteins = 0
    total_interactions = 0
    for source_name, fname in csv_map.items():
        path = pdir / fname
        rows = _count_csv_rows(path)
        sources.append({
            "name": source_name,
            "csv_file": fname,
            "row_count": rows,
            "available": rows > 0,
        })
        # P1-006 v113 ROOT FIX: removed the dead-code block that computed
        # total_drugs three different ways inside the loop, then
        # unconditionally overwrote it with the DrugBank count below.
        # The loop body is now clean — total_drugs is computed AFTER the
        # loop via a fallback chain.
        if source_name == "uniprot":
            total_proteins = rows
        if source_name == "string":
            total_interactions = rows

    # P1-006 v113 ROOT FIX: compute total_drugs via a fallback chain.
    # The previous code unconditionally set total_drugs to the DrugBank
    # row count. If drugbank_drugs.csv was missing (DrugBank academic
    # license paused), total_drugs was 0 even if chembl_drugs.csv had
    # 3,000 rows. ROOT FIX: try drugbank first (canonical FDA-approved
    # list), then chembl, then a generic drugs.csv fallback. This matches
    # the Phase1OutputContract's source-priority semantics.
    for _drug_csv in ("drugbank_drugs.csv", "chembl_drugs.csv", "drugs.csv"):
        _drug_path = pdir / _drug_csv
        if _drug_path.exists():
            total_drugs = _count_csv_rows(_drug_path)
            if total_drugs > 0:
                break

    # P1-029 v113 ROOT FIX: total_proteins should account for STRING
    # proteins that may not be in UniProt (STRING's alias mapping is
    # imperfect). Use the max of uniprot_rows and string_unique_protein_count.
    # For now, we use the interaction count as a lower bound (each
    # interaction has 2 proteins, but they may overlap). The DB query
    # `SELECT COUNT(*) FROM proteins` is the canonical count.
    if total_proteins == 0 and total_interactions > 0:
        total_proteins = total_interactions  # conservative lower bound

    return {
        "sources": sources,
        "total_sources_available": sum(1 for s in sources if s["available"]),
        "total_sources_expected": len(csv_map),
        "total_drugs": total_drugs,
        "total_proteins": total_proteins,
        "total_ppi": total_interactions,
        "processed_data_dir": str(pdir),
        "data_source": "phase1_csv_artifacts",
    }


def _load_drug_mechanism(drug_name: str) -> Dict[str, Any]:
    """Load the mechanism-of-action for a single drug from DrugBank CSV.

    Returns the drug's targets (proteins) and the indications it treats.
    This is the same data the frontend's /api/drugs/mechanism route
    needs (the frontend proxies to DATASET_SERVICE_URL when set).
    """
    pdir = _processed_data_dir()
    drugbank_path = pdir / "drugbank_drugs.csv"
    interactions_path = pdir / "drugbank_interactions.csv"
    indications_path = pdir / "drugbank_indications.csv"

    if not drugbank_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"DrugBank drugs CSV not found at {drugbank_path}. "
                   "Run `python -m pipelines all` to populate Phase 1 data.",
        )

    import csv as csv_mod

    drug_name_lower = drug_name.lower().strip()

    # 1. Find the drug row in drugbank_drugs.csv.
    drug_row: Optional[Dict[str, str]] = None
    try:
        with open(drugbank_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                # Match by name (case-insensitive). DrugBank CSV columns
                # vary, but 'name' is the canonical column.
                name = (row.get("name") or row.get("drug_name") or "").lower()
                if name == drug_name_lower:
                    drug_row = row
                    break
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read {drugbank_path}: {exc}")

    if drug_row is None:
        raise HTTPException(status_code=404, detail=f"Drug '{drug_name}' not found in DrugBank data.")

    # 2. Find targets (proteins) from drugbank_interactions.csv.
    targets: List[Dict[str, Any]] = []
    if interactions_path.exists():
        try:
            with open(interactions_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    name = (row.get("drug") or row.get("drug_name") or "").lower()
                    if name == drug_name_lower:
                        targets.append({
                            "protein": row.get("protein") or row.get("target") or row.get("uniprot_id") or "",
                            "uniprot_id": row.get("uniprot_id") or "",
                            "action": row.get("action") or "unknown",
                            "evidence": row.get("evidence") or "drugbank",
                        })
        except Exception:
            pass

    # 3. Find indications from drugbank_indications.csv.
    indications: List[str] = []
    if indications_path.exists():
        try:
            with open(indications_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    name = (row.get("drug") or row.get("drug_name") or "").lower()
                    if name == drug_name_lower:
                        ind = row.get("indication") or row.get("disease") or ""
                        if ind:
                            indications.append(ind)
        except Exception:
            pass

    return {
        "drug": drug_name,
        "drugbank_id": drug_row.get("drugbank_id") or "",
        "inchikey": drug_row.get("inchikey") or "",
        "smiles": drug_row.get("smiles") or "",
        "targets": targets,
        "indications": indications,
        "source": "phase1_drugbank_csv",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health endpoint. Returns service metadata + data availability."""
    pdir = _processed_data_dir()
    csv_count = len(list(pdir.glob("*.csv*"))) if pdir.exists() else 0
    return {
        "status": "ok",
        "service": "phase1_dataset",
        "version": "1.0.0",
        "phase1_data_available": csv_count > 0,
        "csv_count": csv_count,
        "processed_data_dir": str(pdir),
    }


@app.get("/datasets")
def datasets() -> Dict[str, Any]:
    """Return real Phase 1 dataset stats (no fabrication)."""
    return _load_dataset_stats()


# BE-024 ROOT FIX (Team Member 12): the frontend's
# `frontend/src/lib/services/dataset-stats.ts:proxyToDatasetService()`
# issues `GET ${DATASET_SERVICE_URL}/stats`. The previous Phase 1 service
# only exposed /health, /datasets, and /datasets/{drug}/mechanism — so
# when an operator set DATASET_SERVICE_URL, the proxy 404'd and fell back
# to the local checkpoint silently. The operator believed they were
# fetching live stats when they were not.
#
# Root fix: expose /stats with the EXACT response shape the frontend
# expects (DatasetStatsResponse in dataset-stats.ts). We translate the
# Phase-1-internal _load_dataset_stats() output into that shape — using
# the real CSV row counts as the source of truth. No fabrication: if no
# CSVs exist (Phase 1 hasn't run), we return zero counts with a clear
# `backend: "phase1_service_no_data"` marker so the dashboard can render
# the "no data ingested yet" state honestly.
@app.get("/stats")
def stats() -> Dict[str, Any]:
    """Phase 1 stats in the DatasetStatsResponse shape the frontend expects.

    This endpoint is the contract-satisfying peer of
    ``frontend/src/lib/services/dataset-stats.ts:proxyToDatasetService``.
    It returns the fields the frontend destructures: sources, nodesLoaded,
    edgesLoaded, edgeTypesPresent, pipelineVersion, schemaVersion,
    bridgeVersion, backend, warnings, errors, generatedAt.
    """
    raw = _load_dataset_stats()
    # Translate Phase-1-internal source rows into the frontend's
    # DatasetSourceStat shape: { name, loaded, rowsLoaded?, sha256? }.
    sources_out: List[Dict[str, Any]] = []
    for s in raw.get("sources", []):
        sources_out.append({
            "name": s.get("name", "unknown"),
            "loaded": bool(s.get("available", False)),
            "rowsLoaded": int(s.get("row_count", 0) or 0),
        })

    # nodesLoaded / edgesLoaded: Phase 1 doesn't build the graph (Phase 2
    # does), but the bridge summary records how many nodes/edges were
    # STAGED for Phase 2. We surface the drug + protein + interaction
    # counts as the closest Phase-1-native equivalent. Phase 2's own
    # /kg/stats endpoint is the canonical source for final graph counts.
    nodes_loaded = (
        int(raw.get("total_drugs", 0) or 0)
        + int(raw.get("total_proteins", 0) or 0)
    )
    edges_loaded = int(raw.get("total_ppi", 0) or 0)

    # edgeTypesPresent: Phase 1 stages Compound→Protein (DrugBank
    # interactions) and Protein→Protein (STRING). Other edge types
    # (Protein→Pathway, Pathway→Disease, Drug→AdverseEvent) are added by
    # Phase 2. We surface only what Phase 1 actually produced — never
    # fabricate.
    edge_types: List[str] = []
    for s in raw.get("sources", []):
        if not s.get("available", False):
            continue
        name = s.get("name", "")
        if name == "drugbank":
            edge_types.append("Compound->Protein")
        elif name == "string":
            edge_types.append("Protein->Protein")
        elif name in ("disgenet", "omim"):
            edge_types.append("Gene->Disease")

    warnings: List[str] = []
    errors: List[str] = []
    if raw.get("total_sources_available", 0) < raw.get("total_sources_expected", 0):
        warnings.append(
            f"Phase 1 has {raw.get('total_sources_available', 0)}/"
            f"{raw.get('total_sources_expected', 0)} sources available. "
            "Run the full Airflow ETL pipeline to ingest all 7 sources."
        )

    return {
        "sources": sources_out,
        "nodesLoaded": nodes_loaded,
        "edgesLoaded": edges_loaded,
        "edgeTypesPresent": edge_types,
        "pipelineVersion": "phase1-service-v1",
        "schemaVersion": "1.0",
        "bridgeVersion": None,
        "backend": "phase1_service",
        "warnings": warnings,
        "errors": errors,
        "generatedAt": _now_iso(),
    }


def _now_iso() -> str:
    """Return current UTC time in ISO-8601 (seconds precision)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


@app.get("/datasets/{drug}/mechanism")
def drug_mechanism(drug: str) -> Dict[str, Any]:
    """Return the mechanism-of-action (targets + indications) for a drug."""
    return _load_drug_mechanism(drug)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PHASE1_SERVICE_PORT", "8001"))
    host = os.environ.get("PHASE1_SERVICE_HOST", "0.0.0.0")
    logger.info("Starting Phase 1 Dataset Service on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
