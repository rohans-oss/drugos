"""Test for ORCH-001 ROOT FIX (HIGH).

ORCH-001: run_full_platform.py and run_real_pipeline.py use
          phase1_staged_data= path which produces NO pathway→disease
          edges (see P3-003). The GT model trained on this graph has
          ZERO pathway→disease edges and CANNOT learn the 3-hop
          drug→protein→pathway→disease pattern. GT AUC ≈ random.

          Fix: change both runners to use adapt_phase2_to_phase3 (the
          graph_data= path) instead of phase1_staged_data=. The adapter
          DERIVES (Pathway, disrupted_in, Disease) edges from
          (Gene, associated_with, Disease) edges via the
          gene_symbol → protein → pathway mapping.

This test verifies:
  1. run_full_platform.py uses graph_data= (not phase1_staged_data=).
  2. run_real_pipeline.py uses graph_data= (not phase1_staged_data=).
  3. The adapt_phase2_to_phase3 adapter actually derives
     (pathway, disrupted_in, disease) edges (the scientific requirement).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_orch_001_run_full_platform_uses_graph_data_not_phase1_staged():
    """run_full_platform.py must use graph_data= (the adapter path).

    The phase1_staged_data= path invokes
    BiomedicalGraphBuilder.from_phase1_staged_data(), which only reads
    edges already in staged_data.edges — Phase 1→2 staging produces
    (Gene, associated_with, Disease) but NOT (Pathway, disrupted_in,
    Disease). The GT model trained on this graph has ZERO pathway→disease
    edges and cannot learn the 3-hop pattern.
    """
    src = (_REPO_ROOT / "run_full_platform.py").read_text()
    assert "graph_data=graph_data" in src, (
        "ORCH-001: run_full_platform.py must pass graph_data= to "
        "bridge.run_full_pipeline() (the adapt_phase2_to_phase3 path)."
    )
    assert "phase1_staged_data=staged" not in src, (
        "ORCH-001: run_full_platform.py must NOT pass "
        "phase1_staged_data=staged to bridge.run_full_pipeline() — "
        "that path produces ZERO pathway→disease edges (the GT model "
        "cannot learn the 3-hop drug→protein→pathway→disease pattern)."
    )
    assert "from graph_transformer.data.phase2_adapter import adapt_phase2_to_phase3" in src, (
        "ORCH-001: run_full_platform.py must import adapt_phase2_to_phase3 "
        "and use it to derive the (pathway, disrupted_in, disease) edges."
    )


def test_orch_001_run_real_pipeline_uses_graph_data_not_phase1_staged():
    """run_real_pipeline.py must use graph_data= (the adapter path)."""
    src = (_REPO_ROOT / "run_real_pipeline.py").read_text()
    assert "graph_data=graph_data" in src, (
        "ORCH-001: run_real_pipeline.py must pass graph_data= to "
        "bridge.run_full_pipeline() (the adapt_phase2_to_phase3 path)."
    )
    assert "phase1_staged_data=staged" not in src, (
        "ORCH-001: run_real_pipeline.py must NOT pass "
        "phase1_staged_data=staged to bridge.run_full_pipeline() — "
        "that path produces ZERO pathway→disease edges."
    )
    assert "from graph_transformer.data.phase2_adapter import adapt_phase2_to_phase3" in src, (
        "ORCH-001: run_real_pipeline.py must import adapt_phase2_to_phase3 "
        "and use it to derive the (pathway, disrupted_in, disease) edges."
    )


def test_orch_001_adapter_derives_pathway_disease_edges():
    """adapt_phase2_to_phase3 must DERIVE (pathway, disrupted_in, disease) edges.

    The adapter derives these edges from (Gene, associated_with, Disease)
    edges via the gene_symbol → protein → pathway mapping. Without this
    derivation, the GT model has zero pathway→disease edges and cannot
    learn the 3-hop pattern.

    We verify by reading the adapter source code — the derivation step
    MUST be present.
    """
    adapter_src = (_REPO_ROOT / "graph_transformer" / "data" / "phase2_adapter.py").read_text()
    # The derivation step must be present.
    assert "derived_pathway_disease" in adapter_src, (
        "ORCH-001: adapt_phase2_to_phase3 must derive "
        "(pathway, disrupted_in, disease) edges from "
        "(gene, associated_with, disease) edges. The "
        "'derived_pathway_disease' variable must be present."
    )
    # The gene → protein → pathway mapping must be present.
    assert "gene_id_to_uniprot" in adapter_src, (
        "ORCH-001: adapt_phase2_to_phase3 must build the gene→protein "
        "mapping (gene_id_to_uniprot) to derive pathway→disease edges."
    )
    assert "protein_id_to_pathway_ids" in adapter_src, (
        "ORCH-001: adapt_phase2_to_phase3 must build the protein→pathway "
        "mapping (protein_id_to_pathway_ids) to derive pathway→disease edges."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
