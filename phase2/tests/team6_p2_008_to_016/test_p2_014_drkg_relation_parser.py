"""Regression tests for P2-014: drkg_loader.py relation-prefix format.

P2-014 ROOT FIX: DRKG relations come in two formats:
  1. Hetionet/DRUGBANK: "Hetionet::CtD::Compound:Disease"
  2. GNBR short-code:   "GNBR::A::Gene:Compound"

The previous codebook mapped "A+" and "A-" but NOT plain "A" (the
general "affects" relation), so GNBR::A:: rows had
``relation_human_name="unknown"``. The fix:

  * Adds "A" -> "Compound-affects-Gene" to DRKG_RELATION_ABBREV_TO_NAME.
  * Adds ("A", "Compound", "Gene") and ("A", "Compound", "Disease") to
    DRKG_VALID_TRIPLE_SCHEMAS.
  * Adds ``parse_drkg_relation_head_tail`` for colon-safe splitting.
  * Adds ``canonical_drkg_relation_name`` for case-insensitive lookup.

The tests verify both relation formats parse correctly AND the GNBR "A"
code resolves to a meaningful name (not "unknown").
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "phase2"))

from drugos_graph.config import (  # noqa: E402
    DRKG_RELATION_ABBREV_TO_NAME,
    DRKG_VALID_TRIPLE_SCHEMAS,
    canonical_drkg_relation_name,
    join_drkg_relation,
    parse_drkg_relation_head_tail,
    split_drkg_relation,
)


class TestP2014RelationCodebook:
    """Tests for the DRKG relation codebook (DRKG_RELATION_ABBREV_TO_NAME)."""

    def test_gnbr_a_code_is_mapped(self) -> None:
        """P2-014 spec: GNBR 'A' = 'affects' must be in the codebook."""
        assert "A" in DRKG_RELATION_ABBREV_TO_NAME, (
            "P2-014: GNBR 'A' code must be in DRKG_RELATION_ABBREV_TO_NAME"
        )
        assert DRKG_RELATION_ABBREV_TO_NAME["A"] == "Compound-affects-Gene"

    def test_gnbr_a_plus_and_a_minus_still_mapped(self) -> None:
        """A+ (activate) and A- (inhibit) must still be mapped."""
        assert "A+" in DRKG_RELATION_ABBREV_TO_NAME
        assert "A-" in DRKG_RELATION_ABBREV_TO_NAME

    def test_all_known_gnbr_codes_mapped(self) -> None:
        """Sanity: all GNBR codes referenced in the codebook are present."""
        expected_codes = {"A", "A+", "A-", "B", "E", "E+", "E-", "N", "K", "O", "Z", "J", "L", "Te", "U", "Y", "Md", "X"}
        for code in expected_codes:
            assert code in DRKG_RELATION_ABBREV_TO_NAME, (
                f"GNBR code {code!r} missing from DRKG_RELATION_ABBREV_TO_NAME"
            )


class TestP2014ValidTripleSchemas:
    """Tests for DRKG_VALID_TRIPLE_SCHEMAS."""

    def test_gnbr_a_compound_gene_triple_valid(self) -> None:
        """P2-014: GNBR::A::Compound:Gene rows must pass the valid-triple check."""
        assert ("A", "Compound", "Gene") in DRKG_VALID_TRIPLE_SCHEMAS

    def test_gnbr_a_compound_disease_triple_valid(self) -> None:
        assert ("A", "Compound", "Disease") in DRKG_VALID_TRIPLE_SCHEMAS


class TestP2014RelationSplit:
    """Tests for ``split_drkg_relation`` -- handles both DRKG formats."""

    def test_split_hetonet_format(self) -> None:
        """Hetionet::CtD::Compound:Disease -> ("Hetionet", "CtD", "Compound:Disease")."""
        src, name, head_tail = split_drkg_relation("Hetionet::CtD::Compound:Disease")
        assert src == "Hetionet"
        assert name == "CtD"
        assert head_tail == "Compound:Disease"

    def test_split_gnbr_short_code_format(self) -> None:
        """GNBR::A::Gene:Compound -> ("GNBR", "A", "Gene:Compound")."""
        src, name, head_tail = split_drkg_relation("GNBR::A::Gene:Compound")
        assert src == "GNBR"
        assert name == "A"
        assert head_tail == "Gene:Compound"

    def test_split_drugbank_format(self) -> None:
        src, name, head_tail = split_drkg_relation("DRUGBANK::target::Compound:Gene")
        assert src == "DRUGBANK"
        assert name == "target"
        assert head_tail == "Compound:Gene"

    def test_invalid_relation_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            split_drkg_relation("invalid-no-separator")
        with pytest.raises(ValueError):
            split_drkg_relation("")
        with pytest.raises(ValueError):
            split_drkg_relation(None)  # type: ignore[arg-type]


class TestP2014HeadTailParser:
    """Tests for ``parse_drkg_relation_head_tail`` -- colon-safe split."""

    def test_simple_head_tail(self) -> None:
        head, tail = parse_drkg_relation_head_tail("Compound:Disease")
        assert head == "Compound"
        assert tail == "Disease"

    def test_head_tail_with_colon_in_tail(self) -> None:
        """Entity IDs may contain colons (e.g. DOID:1438) -- the split
        must be on the FIRST colon only."""
        head, tail = parse_drkg_relation_head_tail("Disease:DOID:1438")
        assert head == "Disease"
        assert tail == "DOID:1438"

    def test_invalid_head_tail_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_drkg_relation_head_tail("no-colon")
        with pytest.raises(ValueError):
            parse_drkg_relation_head_tail("")


class TestP2014CanonicalNameLookup:
    """Tests for ``canonical_drkg_relation_name`` -- case-insensitive lookup."""

    def test_exact_match_a(self) -> None:
        assert canonical_drkg_relation_name("A") == "Compound-affects-Gene"

    def test_exact_match_ctd(self) -> None:
        assert canonical_drkg_relation_name("CtD") == "Compound-treats-Disease"

    def test_case_insensitive_match(self) -> None:
        """Lowercase 'ctd' must still resolve to 'Compound-treats-Disease'."""
        assert canonical_drkg_relation_name("ctd") == "Compound-treats-Disease"
        assert canonical_drkg_relation_name("CTD") == "Compound-treats-Disease"

    def test_case_insensitive_match_target(self) -> None:
        assert canonical_drkg_relation_name("TARGET") == "Compound-targets-Gene"
        assert canonical_drkg_relation_name("Target") == "Compound-targets-Gene"

    def test_unknown_code_returns_input(self) -> None:
        """Unknown codes return the input string (caller can detect via
        DRKG_RELATION_ABBREV_TO_NAME.get(code) is None)."""
        assert canonical_drkg_relation_name("XYZ") == "XYZ"

    def test_empty_string_returns_empty(self) -> None:
        assert canonical_drkg_relation_name("") == ""

    def test_case_sensitive_codes_preserved(self) -> None:
        """E+ and E- are case-sensitive -- must not be lowercased."""
        assert canonical_drkg_relation_name("E+") == "Compound-upregulates-expression-of-Gene"
        assert canonical_drkg_relation_name("E-") == "Compound-downregulates-expression-of-Gene"


class TestP2014BothFormatsEndToEnd:
    """End-to-end test: both DRKG relation formats parse and resolve."""

    def test_hetonet_format_resolves_to_canonical_name(self) -> None:
        """Hetionet::CtD::Compound:Disease -> 'CtD' -> 'Compound-treats-Disease'."""
        _, name, _ = split_drkg_relation("Hetionet::CtD::Compound:Disease")
        canonical = canonical_drkg_relation_name(name)
        assert canonical == "Compound-treats-Disease"

    def test_gnbr_a_format_resolves_to_canonical_name(self) -> None:
        """GNBR::A::Gene:Compound -> 'A' -> 'Compound-affects-Gene'."""
        _, name, _ = split_drkg_relation("GNBR::A::Gene:Compound")
        canonical = canonical_drkg_relation_name(name)
        assert canonical == "Compound-affects-Gene", (
            f"GNBR 'A' must resolve to 'Compound-affects-Gene', got {canonical!r}"
        )

    def test_gnbr_b_format_resolves_to_canonical_name(self) -> None:
        _, name, _ = split_drkg_relation("GNBR::B::Gene:Compound")
        canonical = canonical_drkg_relation_name(name)
        assert canonical == "Compound-binds-Gene"

    def test_join_relation_roundtrip(self) -> None:
        """join_drkg_relation must round-trip with split_drkg_relation."""
        original = "GNBR::A::Gene:Compound"
        src, name, head_tail = split_drkg_relation(original)
        rejoined = join_drkg_relation(src, name, head_tail)
        assert rejoined == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
