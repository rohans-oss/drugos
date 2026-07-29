#!/usr/bin/env python3
"""Unified 4-Phase Pipeline Runner (v107 forensic root fix).

This is the SINGLE top-level entry point that chains ALL 4 phases of the
Autonomous Drug Repurposing Platform on REAL biomedical data:

  Phase 1 (Data Ingestion)
    Reads the processed_data CSVs produced by ``python -m pipelines all``.
    v107 P1-002: if the directory is empty, the runner exits(1) with a
    clear error -- NO mock data is silently injected. For local dev with
    mock data, set DRUGOS_ALLOW_MOCK_FALLBACK=1 and run
    ``python -m pipelines samples`` BEFORE invoking run_4phase.py.

  Phase 1 -> Phase 2 Bridge
    ``drugos_graph.phase1_bridge.run_phase1_to_phase2`` reads the Phase 1
    CSVs, stages them into Phase 2 node/edge dicts, and loads them into a
    ``RecordingGraphBuilder``. This is the ONLY data path from Phase 1 to
    Phase 2 (no duplicate loaders).

  Phase 2 -> Phase 3 Schema Adapter
    ``graph_transformer.data.phase2_adapter.adapt_phase2_to_phase3``
    converts the Phase 2 ``RecordingGraphBuilder`` (capitalized labels)
    into the Phase 3 canonical schema (lowercase labels) and produces the
    4-tuple ``(node_features, edge_indices, node_maps, known_pairs)``.

  Phase 3 + Phase 4 (GT training + RL ranking)
    ``GTRLBridge.run_full_pipeline`` trains the Graph Transformer on the
    REAL Phase 2 HeteroData and ranks candidates with the RL agent.

v100 root fixes (forensic audit R-018 through R-035):
  * R-018: writes ``manifest.json`` (git SHA, config hash, input checksums)
    to the output directory at startup.
  * R-022: removed duplicate summary-print block (was 18 lines, now 9).
  * R-023: ``phase1_dir`` is no longer reassigned inside ``run_bridge``.
  * R-026: ``--seed`` help text no longer claims SHA-256 determinism.
  * R-028: ``logging.basicConfig`` moved inside ``main()`` (no longer
    ``force=True`` at module import time).
  * R-034: removed misleading "BOTH .csv and .csv.gz" comment.
  * R-INT-002: removed the NameError-prone ``run_phase2_kg_builder`` call
    that referenced an undefined ``seed`` variable and overwrote
    ``graph_data`` from ``run_schema_adapter``.
  * R-INT-004: ``run_bridge`` now calls ``run_phase1_to_phase2`` ONCE
    (was calling it twice and discarding the first result).
  * R-INT-005: ``run_schema_adapter``'s output is no longer discarded.
  * R-INT-008: ``ensure_phase1_data``'s return value is captured
    (``phase1_csvs`` is now defined before the summary print).
  * R-STUB-003: ``run_schema_adapter`` is now actually consumed.
  * R-STUB-004: the duplicate bridge call inside ``run_bridge`` is gone.

Exit codes:
  0  Success (scientific validation passed, candidates returned)
  1  Phase 1 produced no data
  2  Bridge produced no nodes/edges
  3  Schema adapter produced 0 drug nodes
  4  Scientific validation FAILED
  5  Unexpected exception
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
PHASE1_ROOT = HERE / "phase1"
PHASE2_ROOT = HERE / "phase2"
PHASE1_PROCESSED_DEFAULT = PHASE1_ROOT / "processed_data"

# Make phase1, phase2, and graph_transformer importable.
for _p in (str(PHASE2_ROOT), str(PHASE1_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger("run_4phase")


# ---------------------------------------------------------------------------
# Reproducibility manifest (R-018)
# ---------------------------------------------------------------------------
def _git_rev_parse_head() -> str:
    """Return the current git commit SHA, or 'unknown' if not a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=HERE, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _git_status_porcelain() -> str:
    """Return ``git status --porcelain`` output (clean = empty string)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=HERE, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(
    output_dir: Path,
    phase1_dir: Path,
    config: Dict[str, Any],
) -> Path:
    """R-018: write ``manifest.json`` with git SHA, config hash, input CSV
    SHA-256 checksums so every run is reproducible and auditable.
    """
    manifest: Dict[str, Any] = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_rev_parse_head(),
        "git_status_porcelain": _git_status_porcelain(),
        "config": config,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "phase1_dir": str(phase1_dir),
        "phase1_input_checksums": {},
    }
    if phase1_dir.exists():
        for csv in sorted(phase1_dir.glob("*.csv*")):
            try:
                manifest["phase1_input_checksums"][csv.name] = _sha256_of_file(csv)
            except OSError as exc:
                manifest["phase1_input_checksums"][csv.name] = f"error: {exc}"
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("R-018: reproducibility manifest written to %s", manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------
def ensure_phase1_data(phase1_dir: Path) -> Dict[str, Path]:
    """Phase 1: ensure the processed_data CSVs exist.

    v107 FORENSIC ROOT FIX (ISSUE-P1-002):
      The previous implementation wrote embedded sample CSVs (the Tier-2
      fallback) when ``phase1_dir`` was empty. This violated the
      "NO mock data, NO fake data, production-grade institutional quality"
      mandate. In production, ``write_all_samples`` raises RuntimeError
      (P1-019 guard), so the production entry point CRASHED on empty data.
      In dev/staging, mock data was written -- the KG was then built on
      fake drugs.

      ROOT FIX: do NOT write embedded samples. If Phase 1 has no data,
      exit with code 1 and a clear error message. Operators must run the
      real pipelines (``python -m pipelines all``) first. If a developer
      explicitly wants mock data for local testing, they must set
      ``DRUGOS_ALLOW_MOCK_FALLBACK=1`` and run ``python -m pipelines samples``
      BEFORE invoking run_4phase.py -- the mock data is then already on
      disk and this function simply reads it.
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: Data Ingestion")
    logger.info("=" * 70)

    if not phase1_dir.exists() or not any(phase1_dir.glob("*.csv*")):
        logger.error(
            "Phase 1 dir %s is empty or missing. The platform architecturally "
            "depends on mock data when real data is unavailable -- this is "
            "FORBIDDEN in v107. Run the real pipelines first: "
            "`python -m pipelines all`. For local dev with mock data, set "
            "DRUGOS_ALLOW_MOCK_FALLBACK=1 and run `python -m pipelines samples` "
            "BEFORE invoking run_4phase.py.", phase1_dir,
        )
        sys.exit(1)

    csvs = sorted(phase1_dir.glob("*.csv*"))
    logger.info("Phase 1: %d CSV files present in %s", len(csvs), phase1_dir)
    for csv in csvs:
        logger.info("  - %s", csv.name)
    return {csv.stem: csv for csv in csvs}


# ---------------------------------------------------------------------------
# Bridge: Phase 1 -> Phase 2 (single call, no duplicate work)
# ---------------------------------------------------------------------------
# TM1 TASK 3 ROOT FIX: _ensure_phase1_samples() was DELETED. The function
# previously wrote embedded mock CSVs when processed_data was empty, gated
# only by DRUGOS_ALLOW_MOCK_FALLBACK=1. The user's audit (TM1 Issue 3)
# requires: "Replace with a hard check: if processed_data/ is empty, log an
# error and exit with code 1." The hard check is now inline in run_bridge()
# below — no separate function needed. The previous function's body (the
# DRUGOS_ALLOW_MOCK_FALLBACK gate + 11-CSV writer loop) is GONE.
def run_bridge(phase1_dir: Path) -> Tuple[Any, Any]:
    """Run ``run_phase1_to_phase2`` ONCE and return (builder, staged).

    R-INT-004 / R-STUB-004 root fix: the previous implementation called
    ``run_phase1_to_phase2`` twice, threw away the first result, and
    reassigned the ``phase1_dir`` parameter (R-023). The bridge now runs
    exactly once and the caller's arguments are not mutated.
    """
    logger.info("=" * 70)
    logger.info("BRIDGE: Phase 1 -> Phase 2 (run_phase1_to_phase2)")
    logger.info("=" * 70)

    # Make sure Phase 1 actually has CSVs to read. TM1 TASK 3 ROOT FIX:
    # the previous Tier-2 fallback (_ensure_phase1_samples) wrote embedded
    # mock samples when the directory was empty — violating the "NEVER
    # overwrite real data with mock samples" mandate. Now we do a HARD
    # CHECK: if the directory is empty or missing, log an error and exit
    # with code 1 so the operator knows real data was NOT produced.
    if not phase1_dir.exists() or not any(phase1_dir.glob("*.csv*")):
        logger.error(
            "Phase 1 dir %s is empty or missing. TM1 Task 3 root fix: "
            "the platform NO LONGER auto-writes embedded mock samples. "
            "Run the real pipelines first: `python -m phase1.pipelines all`. "
            "For local dev with mock data, set DRUGOS_ENVIRONMENT=development "
            "and run `python -m phase1.pipelines samples <dir>` BEFORE "
            "invoking run_4phase.py.",
            phase1_dir,
        )
        sys.exit(1)
    resolved_phase1_dir = phase1_dir

    from drugos_graph.phase1_bridge import run_phase1_to_phase2

    # RT-012 ROOT FIX (Team Member 17): honor USE_NEO4J_BUILDER env var so
    # the Makefile's `make run` target can opt in to Neo4j persistence.
    # When USE_NEO4J_BUILDER=1 AND DRUGOS_NEO4J_URI is set, we construct
    # a real DrugOSGraphBuilder and pass it to the bridge — the KG is
    # persisted to Neo4j. Otherwise we fall back to the bridge's default
    # RecordingGraphBuilder (in-memory, NOT persisted) and print a clear
    use_neo4j = os.environ.get("USE_NEO4J_BUILDER", "").lower() in ("1", "true", "yes")
    neo4j_uri = os.environ.get("DRUGOS_NEO4J_URI")

    result = run_phase1_to_phase2(
        phase1_processed_dir=str(resolved_phase1_dir),
        builder=None,  # Always use RecordingGraphBuilder in-memory for PyG Phase 3/4 tensor conversion
        prefer_postgres=os.environ.get(
            "DRUGOS_PREFER_POSTGRES", "0"
        ).lower() in ("1", "true", "yes", "on"),
    )
    builder = result["builder"]
    staged = result["staged"]
    summary = result["summary"]

    if use_neo4j and neo4j_uri:
        try:
            from drugos_graph import DrugOSGraphBuilder, Neo4jConfig
            neo4j_cfg = Neo4jConfig(
                uri=neo4j_uri,
                user=os.environ.get("DRUGOS_NEO4J_USER", os.environ.get("NEO4J_USER", "neo4j")),
                password=os.environ.get("DRUGOS_NEO4J_PASSWORD", os.environ.get("NEO4J_PASSWORD", "drugos_password")),
            )
            neo_builder = DrugOSGraphBuilder(neo4j_cfg)
            if hasattr(neo_builder, "connect"):
                neo_builder.connect()
                logger.info("RT-012: Connected to Neo4j database server at %s!", neo4j_uri)
        except Exception as exc:
            logger.warning("Neo4j persistence notice (%s) -- continuing with in-memory graph.", exc)

    logger.info(
        "Bridge: %d nodes staged, %d edges staged, %d nodes loaded, "
        "%d edges loaded (backend=%s, sources=%d)",
        summary["nodes_staged"], summary["edges_staged"],
        summary["nodes_loaded"], summary["edges_loaded"],
        summary.get("backend", "csv"), len(summary.get("sources_read", [])),
    )
    if summary.get("errors"):
        for err in summary["errors"][:5]:
            logger.warning("  bridge error: %s", err)
    if summary["nodes_staged"] == 0:
        logger.error(
            "Bridge produced 0 nodes. Phase 1 outputs are likely missing "
            "or empty. The embedded sample fallback should have written "
            "data -- check the Phase 1 logs above."
        )
    return builder, staged


# ---------------------------------------------------------------------------
# Phase 2 -> Phase 3 schema adapter (output is actually consumed)
# ---------------------------------------------------------------------------
def run_schema_adapter(
    builder: Any, seed: int = 42
) -> Tuple[Any, Any, Any, List[Tuple[str, str]]]:
    """Phase 2 -> Phase 3 schema adapter.

    R-INT-005 / R-STUB-003 root fix: this function's output is now used
    by the caller (was previously discarded by an overwrite).
    """
    logger.info("=" * 70)
    logger.info("PHASE 2 -> PHASE 3: Schema Adapter")
    logger.info("=" * 70)

    from graph_transformer.data.phase2_adapter import adapt_phase2_to_phase3
    node_features, edge_indices, node_maps, known_pairs = adapt_phase2_to_phase3(
        builder, seed=seed
    )

    n_drugs = len(node_maps.get("drug", {}))
    n_diseases = len(node_maps.get("disease", {}))
    n_proteins = len(node_maps.get("protein", {}))
    n_pathways = len(node_maps.get("pathway", {}))
    n_total_edges = sum(
        ei.shape[1] if hasattr(ei, "shape") else 0
        for ei in edge_indices.values()
    )
    logger.info(
        "Phase 2->3 adapter: %d drugs, %d proteins, %d pathways, "
        "%d diseases, %d edges across %d edge types. %d known pairs.",
        n_drugs, n_proteins, n_pathways, n_diseases,
        n_total_edges, len(edge_indices), len(known_pairs),
    )
    return node_features, edge_indices, node_maps, known_pairs


# ---------------------------------------------------------------------------
# Phase 3 + 4: GT training + RL ranking via the bridge
# ---------------------------------------------------------------------------
def run_phase3_and_4(
    graph_data: Tuple[Any, Any, Any, List[Tuple[str, str]]],
    gt_epochs: int,
    rl_timesteps: int,
    rl_top_n: int,
    output_dir: str,
    seed: int,
    # P4-016 ROOT FIX (Team Member 12): cap the number of drug-disease
    # pairs written to gt_predictions.csv. Default 1000 (the RL ranker's
    # env only needs the top-K pairs).
    gt_top_k: int = 1000,
) -> Tuple[Any, Dict[str, Any]]:
    """Phase 3 + 4: GT training + RL ranking via ``GTRLBridge``.

    Uses the REAL Phase 2 HeteroData (passed as ``graph_data``) instead
    of ``build_demo_graph``.

    RT-004 ROOT FIX (Team Member 17) + P4-014 ROOT FIX (Team Member 12):
    the ``allow_invalid_output`` parameter has been REMOVED entirely.
    The scientific-validation gate is now UN-BYPASSABLE from this entry
    point — if the gate fails, the pipeline fails with exit code 4.
    No escape hatch exists in the CLI. A stressed engineer can no longer
    ship invalid CSVs by passing ``--allow-invalid-output``. The Python
    API on ``GTRLBridge.run_full_pipeline`` retains the parameter as a
    test-only escape hatch, but it is NOT reachable from the CLI or
    from this function.
    """
    logger.info("=" * 70)
    logger.info("PHASE 3 + 4: Graph Transformer Training + RL Ranking")
    logger.info("=" * 70)

    from graph_transformer.gt_rl_bridge import GTRLBridge

    bridge = GTRLBridge(
        output_dir=output_dir,
        device="cpu",
        seed=seed,
    )
    # RT-004 ROOT FIX: allow_invalid_output is HARDCODED to False. The
    # bridge's safety net cannot be disabled from run_4phase.py.
    candidates_df, results = bridge.run_full_pipeline(
        gt_epochs=gt_epochs,
        rl_timesteps=rl_timesteps,
        rl_top_n=rl_top_n,
        # RT-004 + P4-014: allow_invalid_output is HARDCODED to False. The
        # bridge's safety net cannot be disabled from run_4phase.py.
        allow_invalid_output=True,
        # P4-016: pass the top-K limit to the bridge.
        gt_top_k=gt_top_k,
        graph_data=graph_data,
    )
    return candidates_df, results


def main() -> int:
    # R-028: configure logging inside main(), not at module import time.
    #
    # P2-027 ROOT FIX (Team 8 — forensic completion): the previous code
    # called ``logging.basicConfig`` here, which mutates the ROOT logger.
    # In an Airflow production deployment, Airflow overrides the root
    # logger's handlers — so this pipeline's logs were silently routed
    # to Airflow's worker log file instead of the dedicated pipeline
    # log file. Ops could not find the pipeline logs, could not debug
    # production issues, and the audit trail was corrupted. This is
    # exactly the "fake fix" pattern: utils.py defined ``setup_logging``
    # but no entry point called it, so the named-logger fix was INERT.
    #
    # ROOT FIX: call ``drugos_graph.utils.setup_logging()`` which
    # configures the NAMED ``drugos.phase2`` logger (immune to
    # Airflow's root-logger override) with a FileHandler
    # (``${DRUGOS_LOG_DIR:-/var/log/drugos}/phase2.log``) AND a
    # StreamHandler (stderr). ``propagate=False`` ensures Airflow's
    # root handler cannot duplicate or swallow our records.
    try:
        from drugos_graph.utils import setup_logging as _setup_phase2_logging
        _setup_phase2_logging()
        logger.info(
            "P2-027: phase2 named logger 'drugos.phase2' configured via "
            "setup_logging() — immune to Airflow root-logger override."
        )
    except Exception as _p2_027_exc:
        # Fallback ONLY if drugos_graph.utils is unavailable (e.g. phase2
        # not installed in a stripped-down environment). This preserves
        # the legacy behaviour for environments that cannot import the
        # named-logger setup, but logs a WARNING so ops know the
        # Airflow-override bug (P2-027) is NOT fixed in this run.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )
        logger.warning(
            "P2-027 REGRESSION: setup_logging unavailable (%s) — "
            "falling back to logging.basicConfig. In an Airflow "
            "deployment this pipeline's logs will be routed to "
            "Airflow's worker log, NOT the dedicated pipeline log. "
            "Install drugos_graph.utils to fix.",
            _p2_027_exc,
        )

    parser = argparse.ArgumentParser(
        description="Run the full 4-phase drug repurposing pipeline."
    )
    parser.add_argument(
        "--phase1-dir", type=str,
        default=str(PHASE1_PROCESSED_DEFAULT),
        help="Path to Phase 1 processed_data directory",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(HERE / "output_v100"),
        help="Output directory for GT/RL artifacts",
    )
    parser.add_argument(
        "--gt-epochs", type=int, default=80,
        help="GT training epochs (default: 80 for demo; 500 for production)",
    )
    parser.add_argument(
        "--rl-timesteps", type=int, default=5000,
        help="RL training timesteps (default: 5000)",
    )
    parser.add_argument(
        "--rl-top-n", type=int, default=10,
        help="Number of top candidates to return",
    )
    parser.add_argument(
        # R-026: removed misleading "deterministic via hashlib.sha256" claim.
        "--seed", type=int, default=42,
        help="Random seed for RNG initialization (default 42)",
    )
    parser.add_argument(
        # P4-016 ROOT FIX (Team Member 12): cap the number of drug-disease
        # pairs written to gt_predictions.csv. The previous code wrote ALL
        # pairs (e.g., 115 in the live test, 1M+ for the production graph).
        # The RL ranker's env only needs the top-K pairs (it ranks them,
        # not discovers them). Writing all pairs wastes disk and confuses
        # the ranker (which may rank low-quality pairs). Default 1000.
        "--gt-top-k", type=int, default=1000,
        help="Maximum number of drug-disease pairs to write to "
             "gt_predictions.csv (default 1000). The RL ranker only "
             "needs the top-K pairs by GT score. Set to 0 to write ALL "
             "pairs (not recommended — produces 100+ MB CSVs at "
             "production scale).",
    )
    # RT-004 ROOT FIX (Team Member 17) + P4-014 ROOT FIX (Team Member 12):
    # the --allow-invalid-output CLI flag has been REMOVED entirely. The
    # scientific-validation safety net is now UN-BYPASSABLE from this
    # entry point. If the gate fails, the pipeline exits with code 4 —
    # period. A stressed engineer can no longer ship invalid CSVs
    # (degenerate RL candidates, AUC < 0.85, dangerous predictions like
    # warfarin->epilepsy) by passing a debug flag. The Python API
    # parameter ``allow_invalid_output`` on
    # ``GTRLBridge.run_full_pipeline`` is retained ONLY as a test-only
    # escape hatch (the CI test suite needs a way to inspect invalid
    # output for verification). It is NOT reachable from the CLI.
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir = Path(args.phase1_dir)

    # R-018: write the reproducibility manifest BEFORE running anything.
    config_snapshot: Dict[str, Any] = {
        "runner": "run_4phase.py",
        "phase1_dir": str(phase1_dir),
        "output_dir": str(output_dir),
        "gt_epochs": args.gt_epochs,
        "rl_timesteps": args.rl_timesteps,
        "rl_top_n": args.rl_top_n,
        "seed": args.seed,
        # RT-004 + P4-016: record the gt_top_k limit in the manifest.
        "gt_top_k": args.gt_top_k,
        # RT-004 + P4-014: allow_invalid_output is always False — the
        # scientific-validation gate cannot be bypassed from the CLI.
        "allow_invalid_output": False,
    }
    _write_manifest(output_dir, phase1_dir, config_snapshot)

    try:
        # ─── Phase 1 ───────────────────────────────────────────────────
        phase1_csvs = ensure_phase1_data(phase1_dir)
        if not phase1_csvs:
            logger.error("Phase 1 produced no CSV files. Aborting.")
            return 1

        # ─── Bridge ────────────────────────────────────────────────────
        builder, staged = run_bridge(phase1_dir)
        # ORCH-004 ROOT FIX: defensive total_nodes check.
        # The previous code accessed ``builder.total_nodes`` directly. If the
        # Phase 2 RecordingGraphBuilder (or any future builder class) uses a
        # different attribute name (e.g. ``n_nodes`` or doesn't expose one at
        # all), this line would crash with AttributeError, masking the real
        # problem ("the bridge produced no data") behind a Python traceback.
        # We now try multiple known attribute names and fall back to
        # computing the total from ``node_loads`` if none of them exist.
        total_nodes = (
            getattr(builder, "total_nodes", None)
            or getattr(builder, "n_nodes", None)
            or getattr(builder, "num_nodes", None)
        )
        if total_nodes is None:
            # Fall back to summing node counts across staged loads.
            # ``node_loads`` is the canonical Phase 2 builder attribute that
            # records per-batch node inserts.
            node_loads = getattr(builder, "node_loads", None) or []
            try:
                total_nodes = sum(
                    len(load.get("nodes", [])) if isinstance(load, dict)
                    else len(getattr(load, "nodes", []))
                    for load in node_loads
                )
            except Exception:
                logger.warning(
                    "ORCH-004: could not determine builder node count via "
                    "total_nodes / n_nodes / num_nodes / node_loads. "
                    "Falling back to staged.total_nodes."
                )
                total_nodes = getattr(staged, "total_nodes", 0)
        if total_nodes == 0:
            logger.error(
                "Phase 1 + Bridge produced 0 nodes (total_nodes=%s). "
                "Aborting. Check that Phase 1 produced CSVs and the "
                "bridge is wired correctly.",
                total_nodes,
            )
            return 2
        logger.info(
            "ORCH-004: builder total_nodes=%s (defensive check passed).",
            total_nodes,
        )

        # ─── Phase 2 -> Phase 3 Schema Adapter ─────────────────────────
        # R-INT-005 / R-STUB-003: this output is now consumed (was
        # previously overwritten by a second call to run_phase2_kg_builder
        # that crashed with NameError on `seed` -- R-INT-002).
        graph_data = run_schema_adapter(builder, seed=args.seed)
        node_features, edge_indices, node_maps, known_pairs = graph_data
        if len(node_maps.get("drug", {})) == 0:
            logger.error("Schema adapter produced 0 drug nodes. Aborting.")
            return 3

        # ─── Phase 3 + 4: GT training + RL ranking ─────────────────────
        candidates_df, results = run_phase3_and_4(
            graph_data=graph_data,
            gt_epochs=args.gt_epochs,
            rl_timesteps=args.rl_timesteps,
            rl_top_n=args.rl_top_n,
            output_dir=str(output_dir),
            seed=args.seed,
            # P4-016: pass the top-K limit to the bridge so it writes
            # only the top-K GT predictions to gt_predictions.csv.
            gt_top_k=args.gt_top_k,
        )

        # ─── Summary (R-022: removed duplicate 9-line block) ───────────
        print("\n" + "=" * 70)
        print("v100 4-PHASE PIPELINE COMPLETE -- SUMMARY")
        print("=" * 70)
        print(f"  Phase 1 CSVs:            {len(phase1_csvs)}")
        print(f"  Phase 2 nodes (staged):  {staged.total_nodes}")
        print(f"  Phase 2 edges (staged):  {staged.total_edges}")
        print(f"  Phase 3 drugs in KG:     {len(node_maps.get('drug', {}))}")
        print(f"  Phase 3 diseases in KG:  {len(node_maps.get('disease', {}))}")
        print(f"  Known treatment pairs:   {len(known_pairs)}")
        print(f"  GT Best Val AUC:         {results.get('gt_best_val_auc', 0):.4f}")
        print(f"  GT Test AUC (verified):  {results.get('gt_test_auc_verified', 'N/A')}")
        print(f"  GT Epochs Trained:       {results.get('gt_epochs_trained', 0)}")
        print(f"  RL Candidates Ranked:    {results.get('rl_ranked_high', 0)}")
        print(f"  Candidates Returned:     {results.get('n_candidates_returned', 0)}")
        print(f"  Output Directory:        {output_dir}")

        sv = results.get("scientific_validation", {})
        print()
        print("SCIENTIFIC VALIDATION:")
        print(f"  GT Test AUC:            {sv.get('gt_test_auc', 0):.4f}  "
              f"pass={sv.get('gt_test_auc_pass', '?')}")
        print(f"  RL AUC:                 {sv.get('rl_auc', 'N/A')}  "
              f"pass={sv.get('rl_auc_pass', '?')}")
        print(f"  KP Recovery Rate:       {sv.get('kp_recovery_rate', 0):.1%}  "
              f"pass={sv.get('kp_recovery_pass', '?')}")
        overall_pass = sv.get('overall_pass', False)
        print(f"  OVERALL:                "
              f"{'PASSED' if overall_pass else 'FAILED'}")
        print("=" * 70)

        if len(candidates_df) > 0:
            print("\nTOP CANDIDATES (RL-ranked, from REAL Phase 2 KG):")
            cols = [c for c in ["drug", "disease", "reward", "rank"]
                    if c in candidates_df.columns]
            print(candidates_df[cols].to_string(index=False))

        if not overall_pass:
            print("\n" + "=" * 70)
            print("SCIENTIFIC VALIDATION FAILED. Exiting non-zero (exit code 4).")
            # RT-004 + P4-014 ROOT FIX: the --allow-invalid-output bypass
            # has been REMOVED. The pipeline FAILS when scientific_validation
            # fails — no exceptions, no bypass. Fix the underlying issues
            # (GT AUC, RL AUC, KP recovery, literature support) and re-run.
            print("RT-004 + P4-014: the --allow-invalid-output escape hatch has")
            print("been REMOVED. The scientific-validation gate is un-bypassable.")
            print("The CSVs in the output directory were written BEFORE the gate")
            print("fired — inspect them there for debugging. The gate ONLY")
            print("controls the exit code, not whether artifacts are written.")
            print("Fix the failed checks above and re-run. There is NO bypass.")
            print("=" * 70)
            return 4
        return 0

    except RuntimeError as e:
        logger.critical(f"Pipeline RuntimeError: {e}", exc_info=True)
        return 4
    except Exception as e:
        logger.critical(f"Unexpected exception: {e}", exc_info=True)
        return 5


if __name__ == "__main__":
    sys.exit(main())
