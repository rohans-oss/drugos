"""Regression tests for P2-009: stitch_loader.py stereo-code documentation.

P2-009 ROOT FIX: the 6+1 STITCH stereo codes (sm, s, f, m, 0, 1, "") are
now documented in ``_stitch_stereo_label`` with a citation to the STITCH
paper (Kuhn et al., 2008, Nucleic Acids Res. 36:D684-D688,
doi:10.1093/nar/gkm858). The tests verify:

  * All 6+1 codes map to the correct semantic labels.
  * The canonical CID is identical across all 6+1 prefix variants
    (i.e., the collapse is correct).
  * The docstring cites the STITCH paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "phase2"))

from drugos_graph.stitch_loader import (  # noqa: E402
    _normalize_stitch_cid,
    _stitch_stereo_code,
    _stitch_stereo_label,
    _stitch_stereo_label as _ssl,  # alias for shorter test names
)


class TestP2009StereoCodeLabels:
    """All 6+1 stereo codes must map to their documented labels."""

    @pytest.mark.parametrize(
        "code,expected_label",
        [
            ("sm", "stereo_specific_merged"),
            ("s", "stereo_specific"),
            ("f", "different_connectivity"),
            ("m", "non_stereo_merged"),
            ("0", "non_stereo_merged"),
            ("1", "stereo_specific"),
            ("", "unknown"),
        ],
    )
    def test_stereo_code_maps_to_correct_label(
        self, code: str, expected_label: str,
    ) -> None:
        assert _ssl(code) == expected_label

    def test_unknown_code_returns_unknown(self) -> None:
        assert _ssl("xyz") == "unknown"

    def test_none_returns_unknown(self) -> None:
        # _stitch_stereo_label takes a str; passing None falls through
        # to the .get() default.
        assert _ssl(None) == "unknown"  # type: ignore[arg-type]


class TestP2009StereoCodeExtraction:
    """The ``_stitch_stereo_code`` helper must extract the correct code."""

    @pytest.mark.parametrize(
        "cid,expected_code",
        [
            ("CIDsm00002244", "sm"),
            ("CIDs00002244", "s"),
            ("CIDf00002244", "f"),
            ("CIDm00002244", "m"),
            # NOTE: "CID00002244" matches the regex as CID + "0" (stereo
            # code) + "00002244" (digits). So the stereo code is "0",
            # NOT "". The regex treats the 4th char as the stereo code
            # when it is "0" or "1" (newer STITCH format).
            ("CID00002244", "0"),  # 4th char is '0' -> stereo code "0"
            ("CID000002244", "0"),  # same -- "0" + "00002244"
            ("CID100002244", "1"),  # 4th char is '1' -> stereo code "1"
            # Bare digits "00002244" matches as no-CID + "0" (stereo
            # code) + "0002244" (digits). The regex is greedy and
            # treats the leading "0" as the stereo code.
            ("00002244", "0"),
            # Bare CID with no stereo code and shorter digits (no leading
            # 0/1 in the digit position).
            ("CID2244", ""),
        ],
    )
    def test_stereo_code_extraction(
        self, cid: str, expected_code: str,
    ) -> None:
        assert _stitch_stereo_code(cid) == expected_code


class TestP2009CanonicalCollapse:
    """All 6+1 prefix variants must collapse to the SAME canonical CID."""

    @pytest.mark.parametrize(
        "cid",
        [
            "CIDsm00002244",
            "CIDs00002244",
            "CIDf00002244",
            "CIDm00002244",
            "CID00002244",
            "CID000002244",  # CID0 format
            "CID100002244",  # CID1 format
            "00002244",
        ],
    )
    def test_all_variants_collapse_to_same_canonical(
        self, cid: str,
    ) -> None:
        canonical = _normalize_stitch_cid(cid)
        assert canonical == "2244", (
            f"CID {cid!r} must collapse to canonical '2244', got {canonical!r}"
        )

    def test_invalid_cid_returns_empty(self) -> None:
        assert _normalize_stitch_cid("not-a-cid") == ""
        assert _normalize_stitch_cid("") == ""
        assert _normalize_stitch_cid(None) == ""  # type: ignore[arg-type]


class TestP2009DocumentationCompleteness:
    """The docstring must cite the STITCH paper and document all codes."""

    def test_docstring_cites_stitch_paper(self) -> None:
        docstring = _stitch_stereo_label.__doc__ or ""
        # Normalise whitespace so newlines in the docstring don't break
        # substring checks (e.g. "Nucleic Acids\n      Res." should match
        # "Nucleic Acids Res").
        normalised = " ".join(docstring.split())
        # Must mention the STITCH paper citation.
        assert "Kuhn" in normalised, "Docstring must cite Kuhn et al. (STITCH paper)"
        assert "10.1093/nar/gkm858" in normalised, (
            "Docstring must cite the STITCH paper DOI"
        )
        assert "Nucleic Acids Res" in normalised, (
            "Docstring must cite the STITCH paper journal"
        )

    def test_docstring_documents_all_codes(self) -> None:
        docstring = _stitch_stereo_label.__doc__ or ""
        # All 6+1 codes must be documented.
        for code in ("sm", "s", "f", "m", "0", "1"):
            assert f'``"{code}"``' in docstring, (
                f"Docstring must document stereo code {code!r}"
            )

    def test_docstring_documents_compound_effect(self) -> None:
        docstring = _stitch_stereo_label.__doc__ or ""
        assert "Compound effect" in docstring, (
            "Docstring must document the P2-009 compound effect"
        )
        assert "P2-009" in docstring, "Docstring must reference P2-009"

    def test_docstring_documents_m_means_merged_not_racemic(self) -> None:
        """Critical scientific note: CIDm = merged/flat, NOT racemic mixture."""
        docstring = _stitch_stereo_label.__doc__ or ""
        assert "merged" in docstring.lower()
        assert "racemic" in docstring.lower(), (
            "Docstring must clarify that CIDm means 'merged' not 'racemic mixture'"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
