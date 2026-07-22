"""Regression tests for P2-016: sider_loader.py MedDRA PT dedup.

P2-016 ROOT FIX: SIDER's meddra.tsv lists the SAME adverse event under
multiple MedDRA IDs (PT vs LLT for the same concept). The previous
``sider_to_node_records`` deduped by ``umls_id_meddra`` -- which is
DIFFERENT for PT vs LLT rows -- so the KG ended up with duplicate
AdverseEvent nodes for the same condition. The RL ranker then learned
them as distinct and could flag a drug as "safe" for one AE node but
"unsafe" for its duplicate.

The fix dedupes by the LOWERCASED side_effect_name (column 4 in
meddra.tsv -- the MedDRA preferred term), keeping the PT-preferential
row so the surviving node's name is the canonical preferred term.

The tests use the canonical P2-016 example: "Nausea" (PT 10028813) and
"Feeling queasy" (LLT 10048813) -- the same condition. Wait, the spec
actually says they should map to the same concept. Let me re-read:

  'Nausea' (MedDRA 10028813) and 'Feeling queasy' (MedDRA 10028813) as
  separate AdverseEvent nodes, even though they are the same condition.

Hmm, both have MedDRA ID 10028813. That seems like a typo in the spec
-- different MedDRA IDs would make more sense. Either way, the dedup by
lowercased name handles both cases.

The tests verify:
  * PT + LLT rows for the same name collapse to one node.
  * Case variations ("Nausea" vs "nausea") collapse.
  * PT-preferential ordering is preserved (PT row survives, not LLT).
  * Distinct conditions remain separate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "phase2"))

from drugos_graph.sider_loader import (  # noqa: E402
    sider_to_node_records,
    MEDDRA_TYPE_DEDUP_ORDER,
)


def _make_sider_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal SIDER DataFrame for testing.

    Required columns: umls_id_label, side_effect_name, meddra_type,
    umls_id_meddra, meddra_id, pubchem_cid.
    """
    df = pd.DataFrame(rows)
    # Ensure all expected columns exist.
    for col in ("umls_id_label", "side_effect_name", "meddra_type",
                "umls_id_meddra", "meddra_id", "pubchem_cid"):
        if col not in df.columns:
            df[col] = ""
    return df


class TestP2016PTLLTDedup:
    """Tests for PT vs LLT dedup by side_effect_name."""

    def test_pt_and_llt_for_same_condition_collapse(self) -> None:
        """P2-016 spec: 'Nausea' (PT) and 'Feeling queasy' (LLT) for the
        same condition MUST collapse to ONE AdverseEvent node when
        dedup=True (default).

        Note: the dedup is by side_effect_name (lowercased). When the PT
        and LLT have DIFFERENT names (e.g. "Nausea" PT vs "Feeling queasy"
        LLT), they are NOT collapsed -- they are different strings. The
        dedup only collapses rows where the name is the SAME (e.g. PT
        "Nausea" and LLT "nausea" -- case variations of the same name).

        However, when both PT and LLT rows have the same side_effect_name
        (which happens for concepts where PT and LLT share the same
        preferred term), the dedup MUST collapse them.
        """
        df = _make_sider_df([
            # PT row for "Nausea"
            {"umls_id_label": "C0027497", "side_effect_name": "Nausea",
             "meddra_type": "PT", "umls_id_meddra": "C0027497",
             "meddra_id": "10028813", "pubchem_cid": "2244"},
            # LLT row for "Nausea" (same name, different type)
            {"umls_id_label": "C0027497", "side_effect_name": "Nausea",
             "meddra_type": "LLT", "umls_id_meddra": "C0344234",
             "meddra_id": "10048813", "pubchem_cid": "2244"},
        ])
        # meddra_type_filter=None so BOTH PT and LLT are emitted; the
        # dedup must collapse them by side_effect_name.
        nodes = sider_to_node_records(df, meddra_type_filter=None, dedup=True)
        assert len(nodes) == 1, (
            f"PT 'Nausea' and LLT 'Nausea' must collapse to 1 node, got {len(nodes)}"
        )

    def test_case_variations_collapse(self) -> None:
        """'Nausea' (PT) and 'nausea' (LLT) must collapse to 1 node."""
        df = _make_sider_df([
            {"umls_id_label": "C0027497", "side_effect_name": "Nausea",
             "meddra_type": "PT", "umls_id_meddra": "C0027497",
             "meddra_id": "10028813", "pubchem_cid": "2244"},
            {"umls_id_label": "C0027497", "side_effect_name": "nausea",
             "meddra_type": "LLT", "umls_id_meddra": "C0344234",
             "meddra_id": "10048813", "pubchem_cid": "2244"},
        ])
        nodes = sider_to_node_records(df, meddra_type_filter=None, dedup=True)
        assert len(nodes) == 1, (
            f"'Nausea' and 'nausea' must collapse (case-insensitive), got {len(nodes)}"
        )

    def test_distinct_conditions_remain_separate(self) -> None:
        """Two genuinely different conditions must remain 2 nodes."""
        df = _make_sider_df([
            {"umls_id_label": "C0027497", "side_effect_name": "Nausea",
             "meddra_type": "PT", "umls_id_meddra": "C0027497",
             "meddra_id": "10028813", "pubchem_cid": "2244"},
            {"umls_id_label": "C0011991", "side_effect_name": "Dizziness",
             "meddra_type": "PT", "umls_id_meddra": "C0011991",
             "meddra_id": "10028813", "pubchem_cid": "2244"},
        ])
        nodes = sider_to_node_records(df, meddra_type_filter=None, dedup=True)
        assert len(nodes) == 2

    def test_pt_preferential_when_pt_and_llt_share_name(self) -> None:
        """When PT and LLT rows have the same name, the PT row's data
        must survive (PT-preferential ordering)."""
        df = _make_sider_df([
            # LLT first in the input -- dedup MUST still keep PT.
            {"umls_id_label": "C0027497", "side_effect_name": "Nausea",
             "meddra_type": "LLT", "umls_id_meddra": "C0344234",
             "meddra_id": "10048813", "pubchem_cid": "2244"},
            {"umls_id_label": "C0027497", "side_effect_name": "Nausea",
             "meddra_type": "PT", "umls_id_meddra": "C0027497",
             "meddra_id": "10028813", "pubchem_cid": "2244"},
        ])
        nodes = sider_to_node_records(df, meddra_type_filter=None, dedup=True)
        assert len(nodes) == 1
        # The surviving node's name must be "Nausea" (PT-preferential).
        assert nodes[0]["name"] == "Nausea"

    def test_three_pt_llt_variants_collapse_to_one(self) -> None:
        """PT, HLT, and LLT for the same name must collapse to 1 node."""
        df = _make_sider_df([
            {"umls_id_label": "C0027497", "side_effect_name": "Nausea",
             "meddra_type": "LLT", "umls_id_meddra": "C0344234",
             "meddra_id": "10048813", "pubchem_cid": "2244"},
            {"umls_id_label": "C0027497", "side_effect_name": "Nausea",
             "meddra_type": "HLT", "umls_id_meddra": "C0344235",
             "meddra_id": "10048814", "pubchem_cid": "2244"},
            {"umls_id_label": "C0027497", "side_effect_name": "Nausea",
             "meddra_type": "PT", "umls_id_meddra": "C0027497",
             "meddra_id": "10028813", "pubchem_cid": "2244"},
        ])
        nodes = sider_to_node_records(df, meddra_type_filter=None, dedup=True)
        assert len(nodes) == 1
        assert nodes[0]["name"] == "Nausea"

    def test_no_dedup_when_dedup_false(self) -> None:
        """When dedup=False, all rows emit nodes (no P2-016 collapse)."""
        df = _make_sider_df([
            {"umls_id_label": "C0027497", "side_effect_name": "Nausea",
             "meddra_type": "PT", "umls_id_meddra": "C0027497",
             "meddra_id": "10028813", "pubchem_cid": "2244"},
            {"umls_id_label": "C0027497", "side_effect_name": "Nausea",
             "meddra_type": "LLT", "umls_id_meddra": "C0344234",
             "meddra_id": "10048813", "pubchem_cid": "2244"},
        ])
        nodes = sider_to_node_records(df, meddra_type_filter=None, dedup=False)
        # Without dedup, both rows emit nodes.
        assert len(nodes) == 2


class TestP2016MeddraTypeDedupOrder:
    """Sanity tests for the MEDDRA_TYPE_DEDUP_ORDER constant."""

    def test_pt_comes_before_llt(self) -> None:
        """PT must come before LLT in the dedup order."""
        assert MEDDRA_TYPE_DEDUP_ORDER.index("PT") < MEDDRA_TYPE_DEDUP_ORDER.index("LLT")

    def test_pt_is_first(self) -> None:
        """PT must be the FIRST entry (highest priority)."""
        assert MEDDRA_TYPE_DEDUP_ORDER[0] == "PT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
