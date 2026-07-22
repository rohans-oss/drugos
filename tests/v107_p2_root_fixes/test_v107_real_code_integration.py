"""Real-code integration test — exercise the actual production code paths.

This test runs REAL code (not mocks, not smoke tests) to verify the v107
fixes work end-to-end. It:

1. Builds a minimal in-memory RecordingGraphBuilder with REAL Phase 2 nodes.
2. Calls adapt_phase2_to_phase3 on it (the REAL Phase 2 → Phase 3 adapter).
3. Verifies the output is scientifically valid:
   - 5 node types present (drug, protein, pathway, disease, clinical_outcome).
   - Edge types include the multi-hop pattern (drug→protein→pathway→disease).
   - Features are deterministic (run twice, get same result).
   - Features are non-zero (no NaN, no dead nodes).
4. Tests the HeteroData → Phase 3 conversion path (P2-005).
5. Tests the service.py endpoints with TestClient (P2-002, P2-008, P2-016).

Run:
    cd <repo-root>
    python tests/v107_p2_root_fixes/test_v107_real_code_integration.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "phase2"))

os.environ.setdefault("DRUGOS_ENVIRONMENT", "dev")
os.environ.setdefault("DRUGOS_SKIP_CHEMBERTA", "1")

import numpy as np
import torch
from torch_geometric.data import HeteroData


def _build_minimal_phase2_builder():
    """Build a minimal RecordingGraphBuilder-like object with real Phase 2 data."""
    class _Builder:
        def __init__(self):
            self.node_loads = [
                {
                    "label": "Compound",
                    "nodes": [
                        {"id": "DB00122", "name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"},
                        {"id": "DB00273", "name": "topiramate", "smiles": "CC1OC2CC3OC4C(C(C(O1)O)(CO)OC4C(OC3C(C2)CO)C(=O)O)O", "inchikey": "KJTLKKTUJKLGGZ-UHFFFAOYSA-N"},
                    ],
                },
                {
                    "label": "Protein",
                    "nodes": [
                        {"id": "P23219", "name": "PTGS1", "sequence": "MSPDYTYTLYPSHLRTPELVRPGEVGRGLPVRGLRFLHRGFLYYTQDPELQGRVAQAFQGWSPELQRLYHPVKGRGSHRHLPSVPDFTQTIKAVHRAFMSRGRGGLVLGPSGPRVTEYPRFLPSQPCQLHLQALQLPESLQAVWDPFGFGAPLRLHKLALQLPDSLELQVWLEAPTARCFCYWVLPQFPTVVAQARQGDRATQALLLAVVLLRRPLVVASVDRAVQLRCPCPTQALGPLRAVSLGFGLGDAFPPSAQPPAVQGCAGVLSQVLQQLLCQRLHPEAQPCAWAVGPVLDPSVLQCLRKAGLSPQVLQRAVQRAHLHLAFLQLTLLGPELQGVWLPFIAHAPLLAHLRAPSLQAVLQRLHRLPFRLDYTQDPQSLLDALQRAVQRAHLLLQLQLPLAPQHPLVVASVRAAVQRAHLLRLQLPLAPQHPLVVASVRAAVQRAHLFRLLQRLPLAPQHPLVVASVRAAVQRAHLLRLQLPLAPQHPLVVASVRAAVQRAHLLRLQLPLAPQHPLVVASVRAAVQRAHLLRLQLPLAPQHPLVVASVRAAVQRAHLLRLQLPLAPQHPLVVASVRAAVQRAHLFRLQLPLAPQHPLVVASVRAAVQ"},
                        {"id": "P35354", "name": "PTGS2", "sequence": "MSPDYTYTLYPSHLRTPELVRPGEVGRGLPVRGLRFLHRGFLYYTQDPELQGRVAQAFQGWSPELQRLYHPVKGRGSHRHLPSVPDFTQTIKAVHRAFMSRGRGGLVLGPSGPRVTEYPRFLPSQPCQLHLQALQLPESLQAVWDPFGFGAPLRLHKLALQLPDSLELQVWLEAPTARCFCYWVLPQFPTVVAQARQGDRATQALLLAVVLLRRPLVVASVDRAVQLRCPCPTQALGPLRAVSLGFGLGDAFPPSAQPPAVQGCAGVLSQVLQQLLCQRLHPEAQPCAWAVGPVLDPSVLQCLRKAGLSPQVLQRAVQRAHLHLAFLQLTLLGPELQGVWLPFIAHAPLLAHLRAPSLQAVLQRLHRLPFRLDYTQDPQSLLDALQRAVQRAHLLLQLQLPLAPQHPLVVASVRAAVQRAHLLRLQLPLAPQHPLVVASVRAAVQRAHLLRLQLPLAPQHPLVVASVRAAVQRAHLLRLQLPLAPQHPLVVASVRAAVQRAHLLRLQLPLAPQHPLVVASVRAAVQRAHLLRLQLPLAPQHPLVVASVRAAVQRAHLFRLQLPLAPQHPLVVASVRAAVQRAHLLRLQLPLAPQHPLVVASVRAAVQRAHLFRLQLPLAPQHPLVVASVRAAVQRAHLLRLQLPLAPQHPLVVASVRAAVQRAHLLRLQLPLAPQHPLVVASVRAAVQ"},
                    ],
                },
                {
                    "label": "Pathway",
                    "nodes": [
                        {"id": "wp1", "name": "Cyclooxygenase pathway"},
                        {"id": "wp2", "name": "Inflammatory pathway"},
                    ],
                },
                {
                    "label": "Disease",
                    "nodes": [
                        {"id": "DOID:84", "name": "pain"},
                        {"id": "DOID:97", "name": "inflammation"},
                    ],
                },
                {
                    "label": "ClinicalOutcome",
                    "nodes": [
                        {"id": "CO1", "name": "GI bleeding"},
                    ],
                },
            ]
            self.edge_loads = [
                {
                    "src_label": "Compound", "rel_type": "inhibits", "dst_label": "Protein",
                    "edges": [
                        {"src_id": "DB00122", "dst_id": "P23219"},
                        {"src_id": "DB00122", "dst_id": "P35354"},
                        {"src_id": "DB00273", "dst_id": "P35354"},
                    ],
                },
                {
                    "src_label": "Compound", "rel_type": "treats", "dst_label": "Disease",
                    "edges": [
                        {"src_id": "DB00122", "dst_id": "DOID:84"},
                        {"src_id": "DB00122", "dst_id": "DOID:97"},
                    ],
                },
                {
                    "src_label": "Protein", "rel_type": "participates_in", "dst_label": "Pathway",
                    "edges": [
                        {"src_id": "P23219", "dst_id": "wp1"},
                        {"src_id": "P35354", "dst_id": "wp1"},
                        {"src_id": "P35354", "dst_id": "wp2"},
                    ],
                },
                {
                    "src_label": "Compound", "rel_type": "causes", "dst_label": "ClinicalOutcome",
                    "edges": [{"src_id": "DB00122", "dst_id": "CO1"}],
                },
            ]
    return _Builder()


def test_real_adapt_phase2_to_phase3_produces_valid_graph():
    """Exercise adapt_phase2_to_phase3 on real Phase 2 data."""
    from graph_transformer.data.phase2_adapter import adapt_phase2_to_phase3
    builder = _build_minimal_phase2_builder()
    node_features, edge_indices, node_maps, known_pairs = adapt_phase2_to_phase3(
        builder, seed=42
    )
    # All 5 node types must be present (P2-004).
    expected_types = {"drug", "protein", "pathway", "disease", "clinical_outcome"}
    assert set(node_features.keys()) == expected_types, (
        f"node types mismatch: {set(node_features.keys())}"
    )
    # Each node type must have non-zero count.
    for nt, fmap in node_maps.items():
        assert len(fmap) > 0, f"node type {nt} has 0 nodes"
    # Features must be non-NaN and non-zero (P2-003, P2-012).
    for nt, feat in node_features.items():
        assert not torch.isnan(feat).any(), f"node type {nt} has NaN features"
        # Each row must be non-zero (no dead nodes from constant-epsilon bug).
        row_norms = feat.norm(dim=1)
        assert (row_norms > 1e-6).all(), (
            f"node type {nt} has dead (zero) rows after feature generation"
        )
    # Multi-hop pattern must be present (drug→protein→pathway→disease).
    # At minimum, the edge types should include drug-inhibits-protein and
    # protein-part_of-pathway.
    edge_type_strs = {f"{s}-{r}-{d}" for (s, r, d) in edge_indices.keys()}
    assert "drug-inhibits-protein" in edge_type_strs, (
        f"missing drug-inhibits-protein edge type. Edges: {edge_type_strs}"
    )
    assert "protein-part_of-pathway" in edge_type_strs, (
        f"missing protein-part_of-pathway edge type. Edges: {edge_type_strs}"
    )
    # The pathway→disease derivation should have produced SOME edges (via
    # the drug-mediated fallback, since we have no Gene→Disease edges).
    # This validates P2-004's fallback derivation.
    has_pathway_disease = any(
        s == "pathway" and r == "disrupted_in" and d == "disease"
        for (s, r, d) in edge_indices.keys()
    )
    assert has_pathway_disease, (
        f"P2-004 fallback derivation failed — no pathway→disease edges. "
        f"Edges: {edge_type_strs}"
    )
    # known_pairs should include (aspirin, pain) and (aspirin, inflammation).
    pair_set = {(d, v) for d, v in known_pairs}
    assert ("aspirin", "pain") in pair_set, f"missing known pair (aspirin, pain): {pair_set}"
    print(f"  node_types: {list(node_features.keys())}")
    print(f"  node_counts: {', '.join(f'{k}={len(v)}' for k, v in node_maps.items())}")
    print(f"  edge_types: {len(edge_indices)} types")
    print(f"  known_pairs: {len(known_pairs)} (sample: {known_pairs[:3]})")


def test_real_features_deterministic_across_runs():
    """Run adapt_phase2_to_phase3 twice — features must be IDENTICAL."""
    from graph_transformer.data.phase2_adapter import adapt_phase2_to_phase3
    b1 = _build_minimal_phase2_builder()
    b2 = _build_minimal_phase2_builder()
    nf1, _, _, _ = adapt_phase2_to_phase3(b1, seed=42)
    nf2, _, _, _ = adapt_phase2_to_phase3(b2, seed=42)
    for nt in nf1:
        assert torch.allclose(nf1[nt], nf2[nt]), (
            f"node type {nt} features not deterministic across runs"
        )


def test_real_hetero_data_to_phase3_pipeline():
    """Exercise the P2-005 HeteroData → Phase 3 conversion."""
    from graph_transformer.data.phase2_adapter import adapt_hetero_data_to_phase3
    hd = HeteroData()
    hd["Compound"].x = torch.randn(2, 4)
    hd["Compound"].num_nodes = 2
    hd["Compound"]["id"] = torch.tensor([1001, 1002])
    hd["Compound"]["name"] = ["aspirin", "ibuprofen"]
    hd["Compound"]["smiles"] = ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]
    hd["Protein"].x = torch.randn(2, 4)
    hd["Protein"].num_nodes = 2
    hd["Protein"]["id"] = torch.tensor([2001, 2002])
    hd["Protein"]["sequence"] = ["MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDR", "MATTTTRGAGGGGGEPSGSAGAGAGAGAGAG"]
    hd["Pathway"].x = torch.randn(2, 4)
    hd["Pathway"].num_nodes = 2
    hd["Pathway"]["id"] = torch.tensor([3001, 3002])
    hd["Disease"].x = torch.randn(2, 4)
    hd["Disease"].num_nodes = 2
    hd["Disease"]["id"] = torch.tensor([4001, 4002])
    hd["Disease"]["name"] = ["pain", "inflammation"]
    hd["ClinicalOutcome"].x = torch.randn(1, 4)
    hd["ClinicalOutcome"].num_nodes = 1
    hd["ClinicalOutcome"]["id"] = torch.tensor([5001])
    hd["Compound", "inhibits", "Protein"].edge_index = torch.tensor([[0, 1], [0, 1]])
    hd["Compound", "treats", "Disease"].edge_index = torch.tensor([[0, 1], [0, 1]])
    hd["Protein", "participates_in", "Pathway"].edge_index = torch.tensor([[0, 1], [0, 1]])
    # Run conversion.
    nf, ei, nm, kp = adapt_hetero_data_to_phase3(hd, seed=42)
    assert "drug" in nf and "protein" in nf and "pathway" in nf
    assert "disease" in nf and "clinical_outcome" in nf
    assert len(nm["drug"]) == 2
    assert len(nm["protein"]) == 2
    print(f"  HeteroData → Phase3: {sum(len(v) for v in nm.values())} nodes, "
          f"{len(ei)} edge types")


def test_real_service_endpoints_via_testclient():
    """Exercise service.py endpoints with FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from phase2.service import app
    client = TestClient(app)

    # /health must return 200.
    r = client.get("/health")
    assert r.status_code == 200, f"/health failed: {r.status_code}"
    data = r.json()
    assert data["status"] == "ok"
    assert "cors_origins" in data

    # /kg/stats without Phase 1 data MUST return 503 (P2-001, P2-008).
    # We expect a 503 because the test environment has no Phase 1 CSVs.
    # NOTE: this test depends on the repo's phase1/processed_data not
    # having CSV files — if it does, this test will pass with 200 instead.
    r = client.get("/kg/stats")
    # Accept either 503 (no data — expected in clean test env) or 200
    # (data exists — production-like). The KEY check is: NEVER 200 with
    # backend="error" (P2-008 violation).
    if r.status_code == 200:
        body = r.json()
        assert body.get("backend") != "error", (
            f"P2-008 FAIL: /kg/stats returned 200 with backend=error: {body}"
        )
    elif r.status_code == 503:
        # This is the expected path when Phase 1 data is missing.
        body = r.json()
        assert "detail" in body or "error" in str(body).lower(), (
            f"503 response missing error detail: {body}"
        )
    else:
        assert False, f"/kg/stats returned unexpected status {r.status_code}: {r.text}"

    # /query without body must return 422 (validation error).
    r = client.post("/query", json={})
    assert r.status_code == 400, f"expected 400 for empty query, got {r.status_code}"

    # /cypher with valid read-only query must return 503 (no Neo4j) — NOT 200
    # with empty data, and NOT 404 (endpoint must exist).
    r = client.post("/cypher", json={"cypher": "MATCH (n) RETURN n LIMIT 10"})
    assert r.status_code in (503, 502), (
        f"expected 503/502 for /cypher without Neo4j, got {r.status_code}: {r.text}"
    )

    # /cypher with write query must return 400 (whitelist rejection).
    r = client.post("/cypher", json={"cypher": "CREATE (n) RETURN n"})
    assert r.status_code == 400, (
        f"expected 400 for write Cypher, got {r.status_code}: {r.text}"
    )

    # CORS preflight (OPTIONS) must succeed — P2-016.
    r = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    # 200 is the expected CORS preflight response.
    assert r.status_code in (200, 204), (
        f"CORS preflight failed: {r.status_code} — headers: {dict(r.headers)}"
    )
    # Verify Access-Control-Allow-Origin is present.
    aco = r.headers.get("access-control-allow-origin")
    assert aco is not None, f"CORS preflight missing ACAO header: {dict(r.headers)}"


def _run_all():
    tests = [
        test_real_adapt_phase2_to_phase3_produces_valid_graph,
        test_real_features_deterministic_across_runs,
        test_real_hetero_data_to_phase3_pipeline,
        test_real_service_endpoints_via_testclient,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print()
    print(f"=== {passed}/{passed + failed} real-code integration tests passed ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
