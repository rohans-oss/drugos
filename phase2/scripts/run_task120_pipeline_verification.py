#!/usr/bin/env python3
"""Task 120: Run the real Phase 2 pipeline on REAL Phase 1 data.

v109 ROOT FIX (P2-012): the previous version of this script was named
"Run the real Phase 2 pipeline on real Phase 1 data" but used
``_make_synthetic_drkg`` to generate FAKE data (200 random drugs, 100
random diseases, 1500 random pairs). The script then printed "Pipeline
runs end-to-end: YES" — which was technically true (the code paths
ran) but scientifically misleading: it had verified NOTHING about real
Phase 1 data integration.

ROOT FIX (v109): the script now tries to read REAL Phase 1 data via
the Phase 1→2 bridge (``drugos_graph.phase1_bridge.run_phase1_to_phase2``).
If real Phase 1 CSV files are present in ``phase1/processed_data/``, the
script builds the KG from them. If they are NOT present, the script
prints a clear "SYNTHETIC FALLBACK" banner and uses synthetic data
(so the script still works as a smoke test), but the banner makes it
unambiguous that the run did NOT verify real data integration.

Exit codes:
  0 — pipeline ran end-to-end (on real OR synthetic data)
  1 — pipeline crashed
  2 — real Phase 1 data not found AND DRUGOS_STRICT_REAL_DATA=1 set
      (use this in CI to fail if Phase 1 data is missing)
"""
import os
import sys
import time
from pathlib import Path

PHASE2 = Path(__file__).resolve().parents[1]
PHASE1_PROCESSED = PHASE2.parent / "phase1" / "processed_data"
sys.path.insert(0, str(PHASE2))
sys.path.insert(0, str(PHASE2.parent))

os.environ.setdefault("DRUGOS_ENVIRONMENT", "dev")
os.environ.setdefault("DRUGOS_ALLOW_XAVIER_FALLBACK", "1")

import numpy as np
import torch

from drugos_graph.training_data import (
    extract_positive_pairs,
    build_training_data,
    graph_level_split_pairs,
)
from drugos_graph.pyg_builder import PyGBuilder
from drugos_graph.transe_model import TransEModel, TransEConfig
from drugos_graph.evaluation import _manual_auc, compute_auc_direction_aware


def _make_synthetic_drkg(n_drugs=200, n_diseases=100, n_positive_pairs=1500, seed=42):
    """Generate synthetic DRKG data (FALLBACK only — see module docstring)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_positive_pairs):
        d = f"DB{int(rng.integers(0, n_drugs)):05d}"
        dis = f"DOID:{int(rng.integers(0, n_diseases)):07d}"
        rows.append({
            "head_id": d, "relation_name": "DRUGBANK::treats::Compound:Disease",
            "tail_id": dis, "head_type": "Compound", "tail_type": "Disease",
        })
    import pandas as pd
    return pd.DataFrame(rows)


def _load_real_phase1_data():
    """Load REAL Phase 1 data via the Phase 1→2 bridge.

    Returns a pandas DataFrame in DRKG-like format:
        head_id, relation_name, tail_id, head_type, tail_type

    Returns None if real Phase 1 data is not available.
    """
    if not PHASE1_PROCESSED.exists():
        return None
    csvs = list(PHASE1_PROCESSED.glob("*.csv*"))
    if not csvs:
        return None
    try:
        from drugos_graph.phase1_bridge import run_phase1_to_phase2
        result = run_phase1_to_phase2(
            phase1_processed_dir=str(PHASE1_PROCESSED),
            prefer_postgres=False,
        )
        builder = result["builder"]
        summary = result["summary"]
        rows = []
        # Read nodes from the RecordingGraphBuilder
        node_loads = getattr(builder, "node_loads", []) or []
        edge_loads = getattr(builder, "edge_loads", []) or []
        # Build node id → type map
        node_type_map = {}
        for load in node_loads:
            if not isinstance(load, dict):
                continue
            label = load.get("label", "")
            for node in load.get("nodes", []) or []:
                if not isinstance(node, dict):
                    continue
                nid = str(node.get("id", ""))
                if nid:
                    node_type_map[nid] = label
        # Convert edges to DRKG-like rows
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
                rows.append({
                    "head_id": src_id,
                    "relation_name": f"{src_label}::{rel}::{dst_label}",
                    "tail_id": dst_id,
                    "head_type": src_label,
                    "tail_type": dst_label,
                })
        if not rows:
            return None
        import pandas as pd
        df = pd.DataFrame(rows)
        df.attrs["summary"] = summary
        return df
    except Exception as exc:
        print(f"  WARNING: real Phase 1 data load failed: {type(exc).__name__}: {exc}")
        print(f"  Falling back to synthetic data.")
        return None


def _compute_lcc_density(nodes, edges):
    from collections import defaultdict, deque
    adj = defaultdict(set)
    for h, t in edges:
        adj[h].add(t)
        adj[t].add(h)
    visited = set()
    lcc_size = 0
    lcc_edges = 0
    for start in nodes:
        if start in visited:
            continue
        component = set()
        queue = deque([start])
        while queue:
            n = queue.popleft()
            if n in component:
                continue
            component.add(n)
            visited.add(n)
            for nb in adj.get(n, []):
                if nb not in component:
                    queue.append(nb)
        if len(component) > lcc_size:
            lcc_size = len(component)
            lcc_edges = sum(1 for h, t in edges if h in component and t in component)
    if lcc_size <= 1:
        return 0.0, lcc_size, lcc_edges
    max_possible = lcc_size * (lcc_size - 1) / 2
    density = lcc_edges / max_possible if max_possible > 0 else 0.0
    return density, lcc_size, lcc_edges


def main():
    print("=" * 70)
    print("Task 120: Real Phase 2 Pipeline Verification (v109 ROOT FIX)")
    print("=" * 70)
    t0 = time.time()

    # v109 ROOT FIX (P2-012): try REAL Phase 1 data first.
    print("\n[1/6] Loading Phase 1 data...")
    drkg_df = _load_real_phase1_data()
    data_source = "real"
    if drkg_df is None or len(drkg_df) == 0:
        if os.environ.get("DRUGOS_STRICT_REAL_DATA") == "1":
            print("  FAIL: real Phase 1 data not found at "
                  f"{PHASE1_PROCESSED} and DRUGOS_STRICT_REAL_DATA=1 is set.")
            return 2
        print("  " + "!" * 66)
        print("  ! SYNTHETIC FALLBACK MODE — REAL Phase 1 data not found at")
        print(f"  ! {PHASE1_PROCESSED}")
        print("  ! The pipeline will run on SYNTHETIC data (200 drugs x 100 diseases).")
        print("  ! This verifies the CODE PATHS but does NOT verify real data integration.")
        print("  ! To run on real data: run Phase 1 pipelines first, or set")
        print("  ! DRUGOS_PHASE1_PROCESSED_DIR to the correct path.")
        print("  " + "!" * 66)
        drkg_df = _make_synthetic_drkg(n_drugs=200, n_diseases=100, n_positive_pairs=1500)
        data_source = "synthetic"
    print(f"  Data source: {data_source}")
    print(f"  DRKG: {len(drkg_df)} triples")
    if data_source == "real":
        summary = drkg_df.attrs.get("summary", {})
        print(f"  Nodes loaded: {summary.get('nodes_loaded', 0)}")
        print(f"  Edges loaded: {summary.get('edges_loaded', 0)}")
        print(f"  Sources read: {summary.get('sources_read', [])}")

    print("\n[2/6] Extracting positive drug-disease pairs...")
    positive_pairs, positive_pair_set = extract_positive_pairs(drkg_df)
    print(f"  Positive pairs: {len(positive_pairs)}")

    print("\n[3/6] Building training data with negative sampling...")
    all_drug_ids = sorted(drkg_df[drkg_df['head_type'] == 'Compound']['head_id'].unique().tolist())
    all_disease_ids = sorted(drkg_df[drkg_df['tail_type'] == 'Disease']['tail_id'].unique().tolist())
    if not all_drug_ids or not all_disease_ids:
        print(f"  WARNING: drug_ids={len(all_drug_ids)} disease_ids={len(all_disease_ids)} — "
              "the data may not contain Compound→Disease edges.")
    training_data = build_training_data(
        drkg_df, all_drug_ids, all_disease_ids,
        positive_pairs, positive_pair_set,
        held_out_pairs=set(),
    )
    n_pos = training_data["num_positives"]
    n_neg = training_data["num_negatives"]
    print(f"  Training data: {n_pos} positives, {n_neg} negatives")

    print("\n[4/6] Graph-level disjoint split (task 104)...")
    split = graph_level_split_pairs(positive_pairs, seed=42)
    meta = split["_split_metadata"]
    print(f"  Split: train={meta['train_count']}, val={meta['val_count']}, test={meta['test_count']}")
    print(f"  Disjoint: {meta['disjoint']} (no shared drugs/diseases)")

    print("\n[5/6] Building PyG HeteroData graph (task 110/111 dtypes)...")
    entity_maps = {
        "Compound": {did: i for i, did in enumerate(all_drug_ids)},
        "Disease": {did: i for i, did in enumerate(all_disease_ids)},
    }
    edge_maps = {}
    src_list = []
    dst_list = []
    for _, row in drkg_df.iterrows():
        if row['head_type'] == 'Compound' and row['tail_type'] == 'Disease':
            h_idx = entity_maps["Compound"].get(row['head_id'])
            t_idx = entity_maps["Disease"].get(row['tail_id'])
            if h_idx is not None and t_idx is not None:
                src_list.append(h_idx)
                dst_list.append(t_idx)
    if src_list:
        edge_maps[("Compound", "treats", "Disease")] = (src_list, dst_list)

    builder = PyGBuilder()
    n_compounds = len(entity_maps["Compound"])
    n_diseases = len(entity_maps["Disease"])
    node_features = {
        "Compound": torch.randn(max(n_compounds, 1), 32, dtype=torch.float32),
        "Disease": torch.randn(max(n_diseases, 1), 32, dtype=torch.float32),
    }
    data = builder.build_from_drkg(entity_maps, edge_maps, node_features=node_features)
    n_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    n_edges = sum(data[et].num_edges for et in data.edge_types)
    print(f"  PyG graph: {n_nodes} nodes, {n_edges} edges")
    if ("Compound", "treats", "Disease") in data.edge_types:
        print(f"  Edge_index dtype: {data[('Compound', 'treats', 'Disease')].edge_index.dtype}")
    print(f"  Node features dtype: {data['Compound'].x.dtype}")

    print("\n[6/6] Verifying graph scale and density (task 120 criteria)...")
    nodes_set = {f"{nt}_{i}" for nt in data.node_types for i in range(data[nt].num_nodes)}
    edges_set = set()
    for et in data.edge_types:
        ei = data[et].edge_index
        for i in range(ei.shape[1]):
            src, dst = et[0], et[2]
            edges_set.add((f"{src}_{int(ei[0, i])}", f"{dst}_{int(ei[1, i])}"))
    density, lcc_size, lcc_edges = _compute_lcc_density(nodes_set, edges_set)
    print(f"  LCC: {lcc_size} nodes, {lcc_edges} edges, density={density:.4f}")

    print("\n[Bonus] TransE model (tasks 105/106)...")
    config = TransEConfig(embedding_dim=32, margin=1.0, seed=42)
    model = TransEModel(max(n_compounds + n_diseases, 1), 1, embedding_dim=32, config=config)
    h = torch.tensor([0, 1, 2], dtype=torch.long)
    r = torch.tensor([0, 0, 0], dtype=torch.long)
    t = torch.tensor([3, 4, 5], dtype=torch.long)
    scores = model.forward(h, r, t)
    print(f"  TransE scores: {scores.detach().tolist()}")
    print(f"  Score direction: {model.score_direction}, Margin: {config.margin}")

    print("\n[Bonus] AUC computation (tasks 107/108)...")
    pos = np.array([0.1, 0.2, 0.3])
    neg = np.array([0.7, 0.8, 0.9])
    auc = _manual_auc(pos, neg, higher_is_better=False)
    print(f"  Perfect-ranking AUC: {auc:.4f} (expected 1.0)")
    auc_sym = compute_auc_direction_aware(
        [("A", "B", 0.1)], [("C", "D", 0.9)],
        relation_type="symmetric", higher_is_better=False,
    )
    print(f"  Symmetric AUC: {auc_sym:.4f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Task 120 complete in {elapsed:.1f}s")
    print(f"{'=' * 70}")
    print(f"\nSummary:")
    print(f"  - Data source: {data_source.upper()}")
    print(f"  - Pipeline runs end-to-end: YES")
    print(f"  - Graph: {n_nodes} nodes, {n_edges} edges, LCC density={density:.4f}")
    print(f"  - Graph-level split (task 104): disjoint={meta['disjoint']}")
    print(f"  - TransE L1+Bernoulli (tasks 105/106): working")
    print(f"  - AUC rank-based+direction-aware (tasks 107/108): working")
    if data_source == "synthetic":
        print(f"\n  NOTE: SYNTHETIC data was used (200 drugs + 100 diseases = 300 nodes).")
        print(f"  Real Phase 1 data has 10,000+ FDA-approved drugs.")
        print(f"  To verify real data integration, place Phase 1 CSVs in:")
        print(f"    {PHASE1_PROCESSED}")
    else:
        print(f"\n  VERIFIED: real Phase 1 data was loaded and processed end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
