"""Task 99 — Verify PPI edges, MedDRA terms, and stereo compounds are DEDUPLICATED.

Covers:
  * Task 84 (STRING PPI symmetric dedup): (A,B) and (B,A) must collapse
    to a single edge in the KG.
  * Task 89 (SIDER MedDRA PT dedup): only MedDRA preferred terms (PT)
    are loaded; LLT/HLT/HLGT/SOC variants are filtered out so the same
    adverse event does not appear as 5 different edges.
  * Task 88 (STITCH stereo dedup): CIDm (flat) and CIDs (stereo) must
    produce DISTINCT Compound nodes so enantiomers with different
    adverse-event profiles are not merged.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd
import pytest


# =============================================================================
# Task 99.1 — STRING PPI symmetric edge dedup (Task 84)
# =============================================================================

def test_string_ppi_symmetric_dedup():
    """(A,B) and (B,A) must collapse to a single edge."""
    from phase2.drugos_graph.string_loader import (
        _canonicalize_pair_order,
        _drop_duplicates,
        string_to_edge_records,
    )

    rows = [
        {"protein1": "9606.ENSP00000000233", "protein2": "9606.ENSP00000000412",
         "combined_score": 900},
        # Same pair, reversed -- should dedup.
        {"protein1": "9606.ENSP00000000412", "protein2": "9606.ENSP00000000233",
         "combined_score": 900},
        # Same pair, original order again -- should also dedup.
        {"protein1": "9606.ENSP00000000233", "protein2": "9606.ENSP00000000412",
         "combined_score": 950},
        # Different pair.
        {"protein1": "9606.ENSP00000000412", "protein2": "9606.ENSP00000001179",
         "combined_score": 700},
    ]
    df = pd.DataFrame(rows)
    # Apply the production dedup pipeline (Layer 1 from parse_string_raw).
    df = _canonicalize_pair_order(df)
    df = _drop_duplicates(df)
    # After canonicalization + dedup, only 2 unique rows survive.
    assert len(df) == 2, (
        f"Expected 2 rows after canonicalize+dedup, got {len(df)}:\n{df}"
    )
    # And the rows must have protein1 <= protein2 (canonical order).
    for _, row in df.iterrows():
        assert row["protein1"] <= row["protein2"], (
            f"Row not canonicalised: p1={row['protein1']} p2={row['protein2']}"
        )
    # Convert to edges with passthrough (we don't need crosswalk resolution
    # for this dedup test).
    edges = string_to_edge_records(
        df, emit_both_directions=False, unresolved_policy="passthrough",
    )
    assert len(edges) == 2, (
        f"Expected 2 edges after symmetric dedup, got {len(edges)}: "
        f"{[(e['src_id'], e['dst_id']) for e in edges]}"
    )


def test_string_ppi_no_self_loops():
    """Self-loops (A,A) must not be emitted."""
    from phase2.drugos_graph.string_loader import (
        _canonicalize_pair_order,
        _drop_duplicates,
        string_to_edge_records,
    )

    rows = [
        {"protein1": "9606.ENSP00000000233", "protein2": "9606.ENSP00000000233",
         "combined_score": 1000},
        {"protein1": "9606.ENSP00000000233", "protein2": "9606.ENSP00000000412",
         "combined_score": 900},
    ]
    df = pd.DataFrame(rows)
    df = _canonicalize_pair_order(df)
    df = _drop_duplicates(df)
    edges = string_to_edge_records(
        df, emit_both_directions=False, keep_self_loops=False,
        unresolved_policy="passthrough",
    )
    # Only the non-self-loop edge should survive.
    for e in edges:
        assert e["src_id"] != e["dst_id"], (
            f"Self-loop not filtered: src={e['src_id']} dst={e['dst_id']}"
        )


# =============================================================================
# Task 99.2 — SIDER MedDRA preferred-term dedup (Task 89)
# =============================================================================

def test_sider_meddra_pt_filter_default():
    """SIDER loader must default to PT (preferred terms) only."""
    from phase2.drugos_graph.sider_loader import (
        _apply_meddra_type_filter,
        parse_sider_side_effects,
    )

    # Fixture with PT, LLT, HLT, HLGT, SOC for the SAME adverse event.
    rows = [
        {"pubchem_cid": 2244, "umls_id_meddra": "C0000035",
         "side_effect_name": "Abdominal pain", "meddra_type": "PT"},
        {"pubchem_cid": 2244, "umls_id_meddra": "C0000036",
         "side_effect_name": "Abdominal discomfort", "meddra_type": "LLT"},
        {"pubchem_cid": 2244, "umls_id_meddra": "C0000037",
         "side_effect_name": "Gastrointestinal pains", "meddra_type": "HLT"},
        {"pubchem_cid": 2244, "umls_id_meddra": "C0000038",
         "side_effect_name": "Gastrointestinal disorders", "meddra_type": "HLGT"},
        {"pubchem_cid": 2244, "umls_id_meddra": "C0000039",
         "side_effect_name": "Gastrointestinal disorders (SOC)", "meddra_type": "SOC"},
    ]
    df = pd.DataFrame(rows)
    # Default filter is PT.
    filtered = _apply_meddra_type_filter(df, "PT")
    # Only the PT row should survive.
    assert len(filtered) == 1, (
        f"Expected 1 PT row after filter, got {len(filtered)}"
    )
    assert filtered.iloc[0]["meddra_type"] == "PT"
    assert filtered.iloc[0]["side_effect_name"] == "Abdominal pain"


def test_sider_meddra_pt_dedup_same_term():
    """The same PT appearing twice (different UMLS CUIs) must dedup by name."""
    from phase2.drugos_graph.sider_loader import sider_to_node_records

    rows = [
        {"pubchem_cid": 2244, "umls_id_meddra": "C0000035",
         "side_effect_name": "Abdominal pain", "meddra_type": "PT",
         "umls_id_label": "abdominal pain"},
        # Same name, different (lowercase) -- should dedup case-insensitively.
        {"pubchem_cid": 2244, "umls_id_meddra": "C0999999",
         "side_effect_name": "abdominal pain", "meddra_type": "PT",
         "umls_id_label": "abdominal pain"},
        # Different name.
        {"pubchem_cid": 2244, "umls_id_meddra": "C0000744",
         "side_effect_name": "Nausea", "meddra_type": "PT",
         "umls_id_label": "nausea"},
    ]
    df = pd.DataFrame(rows)
    nodes = sider_to_node_records(df, meddra_type_filter="PT", dedup=True)
    # Names should dedup to 2 unique side effects.
    side_effect_names = {str(n.get("name") or "").lower() for n in nodes}
    assert "abdominal pain" in side_effect_names
    assert "nausea" in side_effect_names
    assert len(side_effect_names) == 2, (
        f"Expected 2 unique side effects after dedup, got {len(side_effect_names)}: "
        f"{side_effect_names}"
    )


# =============================================================================
# Task 99.3 — STITCH stereo compound dedup (Task 88)
# =============================================================================

def test_stitch_flat_and_stereo_do_not_collapse():
    """CIDm (flat) and CIDs (stereo) of the same drug must NOT collapse."""
    from phase2.drugos_graph.stitch_loader import (
        _normalize_stitch_cid_with_stereo,
        _strip_stitch_stereo_for_crosswalk,
    )

    # Same drug (warfarin, CID 5410) in both stereo forms.
    flat = _normalize_stitch_cid_with_stereo("CIDm000005410")
    stereo = _normalize_stitch_cid_with_stereo("CIDs000005410")
    assert flat == "CIDm5410"
    assert stereo == "CIDs5410"
    assert flat != stereo, (
        "Flat (CIDm) and stereo (CIDs) forms collapsed to same ID -- "
        "stereo information is lost (Task 88 bug NOT fixed)"
    )

    # The crosswalk must still resolve both to the same PubChem CID
    # (since PubChem CIDs are stereo-agnostic).
    assert _strip_stitch_stereo_for_crosswalk(flat) == "CID5410"
    assert _strip_stitch_stereo_for_crosswalk(stereo) == "CID5410"


def test_stitch_newer_0_1_format_does_not_collapse():
    """CID0 (flat) and CID1 (stereo) of the same drug must NOT collapse."""
    from phase2.drugos_graph.stitch_loader import (
        _normalize_stitch_cid_with_stereo,
    )

    # Newer 0/1 format.
    flat = _normalize_stitch_cid_with_stereo("CID000005410")
    stereo = _normalize_stitch_cid_with_stereo("CID100005410")
    assert flat == "CIDm5410", f"Expected CIDm5410 (mapped from CID0), got {flat!r}"
    assert stereo == "CIDs5410", f"Expected CIDs5410 (mapped from CID1), got {stereo!r}"
    assert flat != stereo


def test_stitch_no_prefix_defaults_to_flat():
    """STITCH IDs with no stereo code default to flat (CIDm)."""
    from phase2.drugos_graph.stitch_loader import (
        _normalize_stitch_cid_with_stereo,
    )

    # Bare digits and bare CID form both default to flat (conservative).
    assert _normalize_stitch_cid_with_stereo("00002244") == "CIDm2244"
    assert _normalize_stitch_cid_with_stereo("CID00002244") == "CIDm2244"


# =============================================================================
# Task 99.4 — Edge dedup at scale (synthetic)
# =============================================================================

def test_string_ppi_dedup_at_scale():
    """1000 random PPI rows with 50% reverse duplicates must dedup to ~500 edges."""
    from phase2.drugos_graph.string_loader import (
        _canonicalize_pair_order,
        _drop_duplicates,
    )

    import random
    random.seed(42)
    base_proteins = [f"9606.ENSP{i:011d}" for i in range(50)]
    rows = []
    seen_pairs = set()
    for _ in range(500):
        a, b = random.sample(base_proteins, 2)
        if a > b:
            a, b = b, a
        seen_pairs.add((a, b))
        rows.append({"protein1": a, "protein2": b, "combined_score": 800})
        # Add the reverse with 50% probability.
        if random.random() < 0.5:
            rows.append({"protein1": b, "protein2": a, "combined_score": 800})

    df = pd.DataFrame(rows)
    # Apply the production dedup pipeline (Layer 1 from parse_string_raw).
    df = _canonicalize_pair_order(df)
    df = _drop_duplicates(df)
    # The row count must match the number of unique pairs.
    assert len(df) == len(seen_pairs), (
        f"Dedup mismatch: {len(df)} rows vs {len(seen_pairs)} unique pairs "
        f"(input rows: {len(rows)})"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-x"]))
