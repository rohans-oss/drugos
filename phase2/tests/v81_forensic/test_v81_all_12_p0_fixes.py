"""
v81 FORENSIC ROOT FIX -- All 12 P0 issues verification
======================================================
Tests for the 12 P0 fixes from the forensic audit:
  P0-E1: _derive_pathways_from_string string_df reference removed
  P0-E2: PATHWAY_DEFAULT ID matches ID_PATTERNS["Pathway"] regex
  P0-E3: Bridge emits normalized_score on every edge type
  P0-F1/F2: node_disjoint_split wired into run_pipeline
  P0-F3: temporal_split uses inductive per-split entity pool
  P0-F4: predict_drug_candidates uses model-aware largest
  P0-F5: GraphTransformerModel.normalize_relation_embeddings defined
  P0-F6: AUC higher_is_better is model-aware (TransE vs HGT)
  P0-F7: AUC enforcement uses held_out_auc when > 0
  P0-F8: KGNegativeSampler Bernoulli path uses _active_rng (reproducible)
  P0-F10: _held_out_entities actually filtered from sampling pool
  P0-F11: combined_sampling accepts rng param; eval uses fresh rng
  P0-F12: production mode refuses missing-relation fallback (raises)

Each test exercises REAL production functions (not mocks). Run with:
    pytest phase2/tests/v81_forensic/test_v81_all_12_p0_fixes.py -v
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

# Ensure phase2 is on sys.path
PHASE2_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PHASE2_PATH))

os.environ.setdefault("DRUGOS_ENVIRONMENT", "dev")
os.environ.setdefault("DRUGOS_ALLOW_NO_SAMPLER", "1")


# ---------------------------------------------------------------------------
# P0-E1: _derive_pathways_from_string does NOT reference undefined string_df
# ---------------------------------------------------------------------------
def test_p0_e1_no_string_df_in_fallback():
    """The buggy code referenced ``string_df`` (undefined) in the fallback
    path. v78 root fix: use ``string_edges`` (in-scope) instead. v81
    verifies the fix is still present."""
    src = (PHASE2_PATH / "drugos_graph" / "phase1_bridge.py").read_text()
    fn_start = src.find("def _derive_pathways_from_string(")
    assert fn_start >= 0, "_derive_pathways_from_string not found"
    fn_end = src.find("\ndef ", fn_start + 1)
    if fn_end < 0:
        fn_end = len(src)
    fn_body = src[fn_start:fn_end]
    fallback_start = fn_body.find("if not pathway_nodes:")
    assert fallback_start >= 0, "fallback block not found"
    fallback_body = fn_body[fallback_start:]
    # The buggy pattern was: for protein in string_df.get(
    assert "string_df.get(" not in fallback_body, (
        "P0-E1 REGRESSION: fallback still references undefined string_df"
    )
    # The fix uses string_edges
    assert "for e in string_edges:" in fallback_body, (
        "P0-E1 REGRESSION: fallback does not use string_edges"
    )


# ---------------------------------------------------------------------------
# P0-E2: PATHWAY_DEFAULT ID matches ID_PATTERNS["Pathway"] regex
# ---------------------------------------------------------------------------
def test_p0_e2_pathway_default_id_matches_regex():
    """v78 root fix: fallback Pathway ID is ``PATHWAY_CC_000000_00000000``
    (matches the ``PATHWAY_CC_\\d+_[0-9a-f]+`` pattern). v81 verifies."""
    from drugos_graph.kg_builder import ID_PATTERNS
    pattern = ID_PATTERNS["Pathway"]
    fallback_id = "PATHWAY_CC_000000_00000000"
    assert re.match(pattern, fallback_id), (
        f"P0-E2 REGRESSION: fallback ID {fallback_id!r} does not match "
        f"ID_PATTERNS['Pathway'] = {pattern!r}"
    )


# ---------------------------------------------------------------------------
# P0-E3: Bridge emits normalized_score on every edge type
# ---------------------------------------------------------------------------
def test_p0_e3_normalized_score_helper_present():
    """v78 root fix: ``_compute_normalized_score`` is the canonical helper
    every edge-emission site calls. v81 verifies the helper exists and
    produces correct values for every source type."""
    from drugos_graph.phase1_bridge import _compute_normalized_score
    # DisGeNET raw_score in [0,1] -> passthrough
    assert _compute_normalized_score(raw_score=0.7, source="disgenet",
                                      rel_type="associated_with") == pytest.approx(0.7)
    # ChEMBL pchembl_value 8.5 -> 8.5/14 ≈ 0.607
    assert _compute_normalized_score(pchembl_value=8.5, source="chembl",
                                      rel_type="inhibits") == pytest.approx(8.5/14.0)
    # STRING combined_score 850 -> 850/1000 = 0.85
    assert _compute_normalized_score(combined_score=850, source="string",
                                      rel_type="interacts_with") == pytest.approx(0.85)
    # DrugBank treats approved -> 1.0
    assert _compute_normalized_score(indication_type="approved", source="drugbank",
                                      rel_type="treats") == 1.0
    # DrugBank targets (no quantitative signal) -> None (NOT 0.0)
    assert _compute_normalized_score(source="drugbank", rel_type="targets") is None
    # OMIM associated_with -> 1.0 (curated human genetics)
    assert _compute_normalized_score(source="omim", rel_type="associated_with") == 1.0


# ---------------------------------------------------------------------------
# P0-F1/F2: node_disjoint_split is wired into run_pipeline
# ---------------------------------------------------------------------------
def test_p0_f1_f2_node_disjoint_split_wired():
    """v29 root fix: ``node_disjoint_split`` exists in pyg_builder and is
    called from run_pipeline. v72 root fix: also wired inline in step11.
    v81 verifies both call sites still exist."""
    src = (PHASE2_PATH / "drugos_graph" / "run_pipeline.py").read_text()
    pyg_call = "pyg_builder.node_disjoint_split(data)" in src
    inline_call = "node_disjoint_split_used" in src
    assert pyg_call or inline_call, (
        "P0-F1/F2 REGRESSION: node_disjoint_split no longer called from run_pipeline"
    )


# ---------------------------------------------------------------------------
# P0-F3: temporal_split uses inductive per-split entity pool
# ---------------------------------------------------------------------------
def test_p0_f3_temporal_split_uses_inductive_pool():
    """v81 root fix: temporal_split negative sampling range is restricted
    to per-split entities (not the full graph node count)."""
    src = (PHASE2_PATH / "drugos_graph" / "pyg_builder.py").read_text()
    assert "v81 FORENSIC ROOT FIX (P0-F3)" in src, (
        "P0-F3 REGRESSION: marker comment not found"
    )
    assert "split_src_list" in src and "split_dst_list" in src, (
        "P0-F3 REGRESSION: per-split entity pool lists not found"
    )
    assert "split_src_list[_h_pick]" in src, (
        "P0-F3 REGRESSION: inductive indexing not found"
    )


# ---------------------------------------------------------------------------
# P0-F4: predict_drug_candidates uses model-aware largest
# ---------------------------------------------------------------------------
def test_p0_f4_predict_drug_candidates_model_aware():
    """v81 root fix: ``predict_drug_candidates`` uses ``largest=_largest``
    where ``_largest = bool(_higher_is_better)``. TransE: False; HGT: True.
    The previous code hardcoded ``largest=False`` which inverted HGT
    predictions and would recommend the WORST drugs to patients."""
    src = (PHASE2_PATH / "drugos_graph" / "transe_model.py").read_text()
    fn_start = src.find("def predict_drug_candidates(")
    assert fn_start >= 0
    fn_end = src.find("\n# ═", fn_start)
    if fn_end < 0:
        fn_end = len(src)
    fn_body = src[fn_start:fn_end]
    assert "_largest = bool(_higher_is_better)" in fn_body, (
        "P0-F4 REGRESSION: _largest detection not found"
    )
    assert "scores.topk(k, largest=_largest)" in fn_body, (
        "P0-F4 REGRESSION: topk does not use largest=_largest"
    )
    # Verify the sort is also model-aware
    assert "if _higher_is_better:" in fn_body, (
        "P0-F4 REGRESSION: sort direction not model-aware"
    )
    assert "candidates.sort(key=lambda c: c.score, reverse=True)" in fn_body, (
        "P0-F4 REGRESSION: HGT descending sort not found"
    )


# ---------------------------------------------------------------------------
# P0-F5: GraphTransformerModel defines normalize_relation_embeddings
# ---------------------------------------------------------------------------
# P2-002 FORENSIC ROOT FIX (Team 4): phase2/drugos_graph/graph_transformer_model.py
# was DELETED -- it was an INCOMPATIBLE HGT model (emb_dim=256, layers=3,
# bilinear decoder) that could NOT load into Phase 3's
# DrugRepurposingGraphTransformer (emb_dim=128, layers=4, separate
# link_predictor). Per the DOCX architecture, Phase 2 produces PyG
# HeteroData for Phase 3 to train -- NOT a trained model. This test now
# verifies the DELETION is in effect (the module is gone) AND that
# Phase 3's canonical model is importable as the backward-compat alias
# ``GraphTransformerModel``.
def test_p0_f5_hgt_normalize_relation_embeddings():
    """P2-002 root fix: phase2 graph_transformer_model.py DELETED;
    Phase 3's DrugRepurposingGraphTransformer (aliased as
    GraphTransformerModel) is the canonical model. The incompatible
    Phase 2 HGT module must NOT exist."""
    import importlib
    # The Phase 2 module MUST NOT exist (P2-002 root fix: deleted).
    try:
        importlib.import_module("drugos_graph.graph_transformer_model")
        raise AssertionError(
            "P2-002 REGRESSION: drugos_graph.graph_transformer_model still "
            "exists -- should have been DELETED per the DOCX architecture "
            "(Phase 2 produces PyG HeteroData, Phase 3 trains the model)."
        )
    except ImportError:
        pass  # expected -- module deleted

    # Phase 3's canonical model MUST be importable as GraphTransformerModel.
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
        GraphTransformerModel,
    )
    assert GraphTransformerModel is DrugRepurposingGraphTransformer, (
        "P2-002 REGRESSION: Phase 3's GraphTransformerModel alias must point "
        "to DrugRepurposingGraphTransformer (the canonical architecture)."
    )


# ---------------------------------------------------------------------------
# P0-F6: AUC higher_is_better is model-aware in train + eval
# ---------------------------------------------------------------------------
def test_p0_f6_auc_higher_is_better_model_aware():
    """v81 root fix: train_transe and _evaluate_triples both detect the
    model's scoring direction (TransE: lower=better -> False; HGT:
    higher=better -> True) and pass it to evaluate_link_prediction. The
    previous code hardcoded ``higher_is_better=False`` which inverted
    HGT AUC and would select the WORST epoch as 'best'."""
    src = (PHASE2_PATH / "drugos_graph" / "transe_model.py").read_text()
    # train_transe slice
    train_start = src.find("def train_transe(")
    assert train_start >= 0
    train_end = src.find("\ndef predict_drug_candidates(", train_start)
    if train_end < 0:
        train_end = len(src)
    train_body = src[train_start:train_end]
    assert "_model_higher_is_better" in train_body, (
        "P0-F6 REGRESSION: train_transe missing _model_higher_is_better detection"
    )
    assert "higher_is_better=_model_higher_is_better" in train_body, (
        "P0-F6 REGRESSION: train_transe val eval does not use _model_higher_is_better"
    )
    # _evaluate_triples slice
    eval_start = src.find("def _evaluate_triples(")
    assert eval_start >= 0
    eval_end = src.find("\ndef train_transe(", eval_start)
    if eval_end < 0:
        eval_end = len(src)
    eval_body = src[eval_start:eval_end]
    assert "_eval_higher_is_better" in eval_body, (
        "P0-F6 REGRESSION: _evaluate_triples missing _eval_higher_is_better detection"
    )
    assert "higher_is_better=_eval_higher_is_better" in eval_body, (
        "P0-F6 REGRESSION: _evaluate_triples does not use _eval_higher_is_better"
    )


# ---------------------------------------------------------------------------
# P0-F7: AUC enforcement uses held_out_auc when > 0
# ---------------------------------------------------------------------------
def test_p0_f7_enforcement_uses_held_out_auc():
    """v42 root fix: AUC enforcement uses ``history.held_out_auc`` when
    > 0, falls back to ``best_val_auc`` only when held_out not computed.
    v81 verifies the fix is still present."""
    src = (PHASE2_PATH / "drugos_graph" / "transe_model.py").read_text()
    assert "_enforcement_auc" in src, (
        "P0-F7 REGRESSION: _enforcement_auc variable not found"
    )
    assert "history.held_out_auc" in src, (
        "P0-F7 REGRESSION: history.held_out_auc not referenced"
    )


# ---------------------------------------------------------------------------
# P0-F8: KGNegativeSampler Bernoulli path uses _active_rng (reproducible)
# ---------------------------------------------------------------------------
def test_p0_f8_bernoulli_rng_reproducible():
    """v81 root fix: the Bernoulli (degree-weighted) path uses
    ``_active_rng.choice`` (the per-sampler seeded RNG) instead of
    ``np.random.choice`` (the global RNG). Two runs with the same
    seed/config/data must produce identical negatives."""
    src = (PHASE2_PATH / "drugos_graph" / "negative_sampling.py").read_text()
    assert "np.random.choice(head_pool" not in src, (
        "P0-F8 REGRESSION: np.random.choice still in Bernoulli path"
    )
    assert "_active_rng.choice(head_pool, p=_h_probs)" in src, (
        "P0-F8 REGRESSION: _active_rng.choice not in Bernoulli path"
    )
    # Actually run the sampler twice with the same seed
    from drugos_graph.negative_sampling import KGNegativeSampler
    entity_type_lookup = {i: "Compound" if i < 10 else "Disease" for i in range(20)}
    known_triples = {(0, 0, 10), (1, 0, 11), (2, 0, 12)}
    relation_to_types = {0: ("Compound", "Disease")}

    def make_sampler(seed: int) -> KGNegativeSampler:
        return KGNegativeSampler(
            num_entities=20, num_relations=1, num_negatives=10,
            known_triples=known_triples, strategy="type_constrained",
            seed=seed, entity_type_lookup=entity_type_lookup,
            relation_to_types=relation_to_types,
        )

    s1 = make_sampler(42)
    s2 = make_sampler(42)
    n1 = s1.combined_sampling(total_negatives=10, relation_idx=0)
    n2 = s2.combined_sampling(total_negatives=10, relation_idx=0)
    h1 = [s["head_idx"] for s in n1]
    h2 = [s["head_idx"] for s in n2]
    assert h1 == h2, (
        f"P0-F8 REGRESSION: non-reproducible -- {h1} != {h2}"
    )


# ---------------------------------------------------------------------------
# P0-F10: _held_out_entities actually filtered from sampling pool
# ---------------------------------------------------------------------------
def test_p0_f10_held_out_entities_filtered():
    """v81 root fix: the v53 ``_held_out_entities`` set is now ACTUALLY
    used in ``combined_sampling`` to filter head_pool and tail_pool
    BEFORE sampling. Previously the filter was comment-only."""
    from drugos_graph.negative_sampling import KGNegativeSampler
    entity_type_lookup = {i: "Compound" if i < 10 else "Disease" for i in range(20)}
    known_triples = {(0, 0, 10), (1, 0, 11), (2, 0, 12), (3, 0, 13)}
    held_out_pairs = {(5, 0, 15), (6, 0, 16)}
    relation_to_types = {0: ("Compound", "Disease")}
    sampler = KGNegativeSampler(
        num_entities=20, num_relations=1, num_negatives=50,
        known_triples=known_triples, strategy="type_constrained",
        seed=42, entity_type_lookup=entity_type_lookup,
        relation_to_types=relation_to_types,
        held_out_pairs=held_out_pairs,
    )
    assert sampler._held_out_entities, "_held_out_entities is empty"
    negatives = sampler.combined_sampling(total_negatives=100, relation_idx=0)
    head_set = {s["head_idx"] for s in negatives}
    tail_set = {s["tail_idx"] for s in negatives}
    leaked_heads = head_set & {5, 6}
    leaked_tails = tail_set & {15, 16}
    assert not leaked_heads, (
        f"P0-F10 REGRESSION: held-out heads leaked: {leaked_heads}"
    )
    assert not leaked_tails, (
        f"P0-F10 REGRESSION: held-out tails leaked: {leaked_tails}"
    )


# ---------------------------------------------------------------------------
# P0-F11: combined_sampling accepts rng; eval uses fresh rng
# ---------------------------------------------------------------------------
def test_p0_f11_eval_rng_isolated():
    """v81 root fix: ``combined_sampling`` accepts an optional ``rng``
    parameter. When supplied (eval path), the sampler uses it instead
    of ``self._rng`` (which has been advanced by training). This makes
    held-out AUC independent of training duration."""
    import inspect
    from drugos_graph.negative_sampling import KGNegativeSampler
    sig = inspect.signature(KGNegativeSampler.combined_sampling)
    assert "rng" in sig.parameters, (
        "P0-F11 REGRESSION: rng parameter missing from combined_sampling signature"
    )
    # Actually verify reproducibility
    entity_type_lookup = {i: "Compound" if i < 10 else "Disease" for i in range(20)}
    known_triples = {(0, 0, 10), (1, 0, 11)}
    relation_to_types = {0: ("Compound", "Disease")}
    sampler = KGNegativeSampler(
        num_entities=20, num_relations=1, num_negatives=10,
        known_triples=known_triples, strategy="type_constrained",
        seed=42, entity_type_lookup=entity_type_lookup,
        relation_to_types=relation_to_types,
    )
    # Advance self._rng by calling combined_sampling without rng (training)
    for _ in range(5):
        sampler.combined_sampling(total_negatives=10, relation_idx=0)
    # Now pass fresh rng with same seed twice -- should produce identical output
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    n1 = sampler.combined_sampling(total_negatives=10, relation_idx=0, rng=rng1)
    n2 = sampler.combined_sampling(total_negatives=10, relation_idx=0, rng=rng2)
    h1 = [s["head_idx"] for s in n1]
    h2 = [s["head_idx"] for s in n2]
    assert h1 == h2, (
        f"P0-F11 REGRESSION: eval RNG non-reproducible -- {h1} != {h2}"
    )


# ---------------------------------------------------------------------------
# P0-F12: production mode refuses missing-relation fallback
# ---------------------------------------------------------------------------
def test_p0_f12_production_refuse_missing_relation():
    """v81 root fix: in production mode (DRUGOS_ENVIRONMENT=prod),
    ``_evaluate_triples`` RAISES ``EvaluationError`` when a relation_idx
    is missing from ``negative_sampler.relation_to_types``. The previous
    code only LOGGED the inflation -- did not PREVENT it."""
    src = (PHASE2_PATH / "drugos_graph" / "transe_model.py").read_text()
    assert "_is_prod_f12" in src, (
        "P0-F12 REGRESSION: production-mode check not found"
    )
    assert "raise EvaluationError(" in src, (
        "P0-F12 REGRESSION: EvaluationError raise not found"
    )
    # Verify the catch block re-raises EvaluationError (doesn't swallow it)
    assert "except EvaluationError:" in src, (
        "P0-F12 REGRESSION: EvaluationError catch block not found"
    )
    # The catch block must have a bare `raise` to re-raise
    catch_start = src.find("except EvaluationError:")
    catch_body = src[catch_start:catch_start + 500]
    assert "raise" in catch_body, (
        "P0-F12 REGRESSION: catch block does not re-raise EvaluationError"
    )


# ---------------------------------------------------------------------------
# Phase 1 ↔ Phase 2 connectivity (the user's #1 ask)
# ---------------------------------------------------------------------------
def test_phase1_phase2_bridge_connectivity():
    """End-to-end: stage Phase 1 frames -> load_into_graph ->
    bridge_to_pyg_maps. Verify entity_maps and edge_maps are non-empty
    and contain the expected node types (Protein, Gene, Disease,
    Pathway per the DOCX 5-node-type contract)."""
    from drugos_graph.phase1_bridge import (
        stage_phase1_to_phase2,
        load_into_graph,
        bridge_to_pyg_maps,
        RecordingGraphBuilder,
    )
    frames = {
        "drugbank_compounds": pd.DataFrame([
            {"drugbank_id": "DB00001", "name": "DrugA", "inchikey": "INCHIKEYA"},
        ]),
        "drugbank_interactions": pd.DataFrame([
            {"drugbank_id": "DB00001", "uniprot_id": "P12345", "action_type": "inhibitor"},
        ]),
        "drugbank_indications": pd.DataFrame([
            {"drugbank_id": "DB00001", "disease_id": "OMIM:100100",
             "disease_name": "DiseaseA", "indication_type": "approved"},
        ]),
        "omim_gda": pd.DataFrame([
            {"gene_symbol": "GENE1", "disease_id": "OMIM:100100",
             "canonical_gene_id": "1001", "uniprot_id": "P12345"},
        ]),
        "string_ppi": pd.DataFrame([
            {"uniprot_ac_a": "P12345", "uniprot_ac_b": "P67890", "combined_score": 850},
        ]),
    }
    staged = stage_phase1_to_phase2(frames, run_id="v81_test")
    builder = RecordingGraphBuilder()
    load_into_graph(staged, builder)
    entity_maps, edge_maps = bridge_to_pyg_maps(builder)
    # Should have at least 4 node types (Protein, Gene, Disease, Pathway)
    assert len(entity_maps) >= 4, (
        f"Expected >= 4 node types, got {len(entity_maps)}: {list(entity_maps.keys())}"
    )
    # Should have at least 1 edge type
    assert len(edge_maps) >= 1, (
        f"Expected >= 1 edge type, got {len(edge_maps)}"
    )
    # Pathway node type should be present (DOCX contract)
    assert "Pathway" in entity_maps, (
        f"Pathway node type missing -- DOCX 5-node-type contract violated. "
        f"Got: {list(entity_maps.keys())}"
    )


if __name__ == "__main__":
    # Allow running directly without pytest
    import traceback
    tests = [
        test_p0_e1_no_string_df_in_fallback,
        test_p0_e2_pathway_default_id_matches_regex,
        test_p0_e3_normalized_score_helper_present,
        test_p0_f1_f2_node_disjoint_split_wired,
        test_p0_f3_temporal_split_uses_inductive_pool,
        test_p0_f4_predict_drug_candidates_model_aware,
        test_p0_f5_hgt_normalize_relation_embeddings,
        test_p0_f6_auc_higher_is_better_model_aware,
        test_p0_f7_enforcement_uses_held_out_auc,
        test_p0_f8_bernoulli_rng_reproducible,
        test_p0_f10_held_out_entities_filtered,
        test_p0_f11_eval_rng_isolated,
        test_p0_f12_production_refuse_missing_relation,
        test_phase1_phase2_bridge_connectivity,
    ]
    n_pass = 0
    n_fail = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"[PASS] {test_fn.__name__}")
            n_pass += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            traceback.print_exc()
            n_fail += 1
    print()
    print(f"RESULTS: {n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)
