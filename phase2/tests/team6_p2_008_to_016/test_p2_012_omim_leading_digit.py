"""Regression tests for P2-012: omim_loader.py MIM leading digit validation.

P2-012 ROOT FIX: OMIM MIM numbers have a 6-digit range [100000, 999999]
AND the leading digit has semantic meaning per OMIM's numbering:
  1 = autosomal dominant
  2 = autosomal recessive
  3 = X-linked
  4 = Y-linked
  5 = mitochondrial
  6 = autosomal (newly assigned)

A leading 0 (e.g. 099999) is NOT a valid OMIM ID -- it's a string-padded
5-digit number. A leading 7/8/9 is in the 6-digit range but has no
semantic meaning in OMIM's scheme. The fix rejects these.

The tests verify:
  * Valid MIMs with leading digits 1-6 are accepted (return MIM:<id>).
  * Invalid MIMs with leading 0 (e.g. 099999) fall back to SYM:<symbol>.
  * Invalid MIMs with leading 7/8/9 fall back to SYM:<symbol>.
  * Out-of-range MIMs (5-digit, 7-digit) still fall back as before.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "phase2"))

from drugos_graph.omim_loader import _safe_gene_id_from_mim  # noqa: E402


class TestP2012LeadingDigitValidation:
    """P2-012: leading digit must be in [1-6]."""

    @pytest.mark.parametrize(
        "mim,expected_prefix",
        [
            # Valid: leading digits 1-6.
            ("104300", "MIM:"),  # 1 = autosomal dominant (BRCA1 locus)
            ("217400", "MIM:"),  # 2 = autosomal recessive
            ("300376", "MIM:"),  # 3 = X-linked
            ("400005", "MIM:"),  # 4 = Y-linked
            ("516060", "MIM:"),  # 5 = mitochondrial
            ("603903", "MIM:"),  # 6 = autosomal (newly assigned)
            ("134934", "MIM:"),  # FGFR3 -- canonical example from code
            ("176805", "MIM:"),  # POU3F4
        ],
    )
    def test_valid_mims_with_leading_1_to_6(
        self, mim: str, expected_prefix: str,
    ) -> None:
        result = _safe_gene_id_from_mim(mim, "TEST_GENE")
        assert result is not None
        assert result.startswith(expected_prefix), (
            f"MIM {mim} (leading digit {mim[0]}) must be valid, got {result!r}"
        )

    @pytest.mark.parametrize(
        "mim",
        [
            # NOTE: "099999" cannot be tested as a leading-0 case because
            # int(float("099999")) = 99999 (5 digits), which fails the
            # 6-digit range check BEFORE reaching the leading-digit check.
            # The leading-0 case is conceptually unreachable via int(float())
            # -- any 6-character string starting with "0" becomes a 5-digit
            # int after the leading 0 is stripped. The 6-digit range check
            # catches it first. So we only test leading 7/8/9 here.
            "700000",  # leading 7 -- invalid (no semantic meaning)
            "800000",  # leading 8 -- invalid
            "900000",  # leading 9 -- invalid
        ],
    )
    def test_invalid_mims_fall_back_to_sym(
        self, mim: str, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            result = _safe_gene_id_from_mim(mim, "TEST_GENE")
        # Must fall back to SYM:<symbol>, NOT MIM:<id>.
        assert result is not None
        assert result.startswith("SYM:"), (
            f"Invalid MIM {mim} (leading digit {mim[0]}) must fall back to "
            f"SYM:TEST_GENE, got {result!r}"
        )
        assert result == "SYM:TEST_GENE"
        # A warning must be logged mentioning P2-012 or "invalid leading
        # digit". The warning is emitted via the standard logging module
        # (omim_loader uses logger.warning), so caplog.text captures it.
        assert (
            "P2-012" in caplog.text
            or "invalid leading digit" in caplog.text
        ), (
            f"Warning for invalid MIM {mim} must mention P2-012 or "
            f"'invalid leading digit'. Captured log: {caplog.text!r}"
        )

    def test_leading_zero_caught_by_6_digit_range_check(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """P2-012 edge case: a string like '099999' is parsed by
        int(float()) as 99999 (5 digits), which fails the 6-digit range
        check BEFORE the leading-digit check. The warning mentions the
        6-digit range failure, not the leading-digit failure. This test
        documents the behaviour so future contributors understand why
        '099999' doesn't trigger the P2-012 leading-digit warning."""
        with caplog.at_level(logging.WARNING):
            result = _safe_gene_id_from_mim("099999", "TEST_GENE")
        assert result == "SYM:TEST_GENE"
        # The 6-digit range warning must fire (NOT the leading-digit warning).
        assert "not a 6-digit MIM number" in caplog.text, (
            "099999 should fail the 6-digit range check (int('099999')=99999 "
            "is 5 digits), not the leading-digit check."
        )


class TestP2012OutOfRangeMims:
    """MIMs outside the 6-digit range still fall back (existing behaviour)."""

    @pytest.mark.parametrize(
        "mim",
        [
            "9999",     # 4-digit -- too short
            "99999",    # 5-digit -- too short
            "1000000",  # 7-digit -- too long
            "9999999",  # 7-digit -- too long
        ],
    )
    def test_out_of_range_mims_fall_back(self, mim: str) -> None:
        result = _safe_gene_id_from_mim(mim, "TEST_GENE")
        assert result is not None
        assert result.startswith("SYM:"), (
            f"Out-of-range MIM {mim} must fall back to SYM:TEST_GENE"
        )


class TestP2012NonNumericMims:
    """Non-numeric MIM values fall back to SYM:<symbol>."""

    @pytest.mark.parametrize(
        "mim",
        ["", "nan", "None", "null", "?", "-", "abc", None],
    )
    def test_non_numeric_mims_fall_back(self, mim: object) -> None:
        result = _safe_gene_id_from_mim(mim, "TEST_GENE")
        assert result is not None
        assert result.startswith("SYM:")


class TestP2012EdgeCases:
    """Edge cases for the leading-digit validator."""

    def test_float_input_with_valid_mim(self) -> None:
        # int(float("104300")) == 104300 -- leading digit 1, valid.
        result = _safe_gene_id_from_mim(104300.0, "TEST_GENE")
        assert result == "MIM:104300"

    def test_float_input_with_invalid_leading_digit(self) -> None:
        # int(float("700000")) == 700000 -- leading 7, invalid.
        result = _safe_gene_id_from_mim(700000.0, "TEST_GENE")
        assert result == "SYM:TEST_GENE"

    def test_int_input_with_valid_mim(self) -> None:
        result = _safe_gene_id_from_mim(104300, "TEST_GENE")
        assert result == "MIM:104300"

    def test_int_input_with_invalid_leading_digit(self) -> None:
        result = _safe_gene_id_from_mim(700000, "TEST_GENE")
        assert result == "SYM:TEST_GENE"

    def test_no_symbol_returns_none_for_invalid(self) -> None:
        """When the MIM is invalid AND no gene_symbol is provided, the
        function returns None (not a SYM: string)."""
        result = _safe_gene_id_from_mim("700000", "")
        assert result is None

    def test_allele_variant_truncation_still_validates_leading_digit(self) -> None:
        """Allele variants like '134934.001' are truncated to 134934 by
        int(float()), then the leading-digit check applies."""
        result = _safe_gene_id_from_mim("134934.001", "TEST_GENE")
        assert result == "MIM:134934"

    def test_allele_variant_with_invalid_leading_digit(self) -> None:
        result = _safe_gene_id_from_mim("700000.001", "TEST_GENE")
        assert result == "SYM:TEST_GENE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
