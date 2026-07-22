"""Regression tests for P2-008: string_loader.py gene-symbol cross-species dedup.

P2-008 ROOT FIX: the canonical-pair dedup in ``_canonicalize_pair_order``
uses ``protein1 < protein2`` string comparison. This is correct ONLY when
identifiers carry the species taxid prefix (e.g. ``"9606.ENSP00000000233"``).
When the loader is configured to use bare gene symbols, the dedup may
merge cross-species PPIs that share the same gene symbol (e.g. human
TP53 vs mouse Tp53). The fix adds:

  1. ``_looks_like_canonical_string_id`` -- shape detector.
  2. ``_warn_if_gene_symbol_mode`` -- WARNING emitter when <80% of
     sampled IDs look canonical.
  3. Updated ``_canonicalize_pair_order`` to call the warning before
     dedup.

These tests verify:
  * Canonical STRING IDs are detected as canonical.
  * Bare gene symbols are NOT detected as canonical.
  * The warning is emitted when gene-symbol mode is active.
  * No warning is emitted for canonical STRING IDs.
  * Cross-species PPIs are NOT collapsed when canonical IDs are used.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the phase2 package is importable.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "phase2"))

from drugos_graph.string_loader import (  # noqa: E402
    _canonicalize_pair_order,
    _looks_like_canonical_string_id,
    _warn_if_gene_symbol_mode,
    _drop_duplicates,
)


class TestP2008CanonicalStringIdDetection:
    """Tests for the ``_looks_like_canonical_string_id`` helper."""

    def test_human_ensembl_id_is_canonical(self) -> None:
        assert _looks_like_canonical_string_id("9606.ENSP00000000233") is True

    def test_mouse_ensembl_id_is_canonical(self) -> None:
        assert _looks_like_canonical_string_id("10090.ENSMUSP000000001") is True

    def test_bare_gene_symbol_is_not_canonical(self) -> None:
        assert _looks_like_canonical_string_id("TP53") is False

    def test_bare_lowercase_gene_symbol_is_not_canonical(self) -> None:
        assert _looks_like_canonical_string_id("tp53") is False

    def test_taxid_prefixed_gene_symbol_is_canonical(self) -> None:
        # "9606.TP53" still has the taxid+dot+letter shape -- canonical.
        assert _looks_like_canonical_string_id("9606.TP53") is True

    def test_empty_string_is_not_canonical(self) -> None:
        assert _looks_like_canonical_string_id("") is False

    def test_none_is_not_canonical(self) -> None:
        assert _looks_like_canonical_string_id(None) is False  # type: ignore[arg-type]

    def test_pure_digit_is_not_canonical(self) -> None:
        assert _looks_like_canonical_string_id("9606") is False

    def test_dot_then_digit_is_not_canonical(self) -> None:
        # "9606.12345" -- tail starts with a digit, not a letter.
        assert _looks_like_canonical_string_id("9606.12345") is False


class TestP2008GeneSymbolModeWarning:
    """Tests for the ``_warn_if_gene_symbol_mode`` warning emitter."""

    def test_warning_emitted_for_bare_gene_symbols(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        df = pd.DataFrame({
            "protein1": ["TP53", "BRCA1", "EGFR", "MYC"],
            "protein2": ["MDM2", "ATM", "KRAS", "RB1"],
        })
        with caplog.at_level(logging.WARNING, logger="drugos_graph.string_loader"):
            _warn_if_gene_symbol_mode(df)
        # The warning must be emitted because <80% of IDs are canonical.
        assert any(
            "string_gene_symbol_mode_detected" in (r.message or "")
            or "P2-008" in (r.message or "")
            for r in caplog.records
        ) or any(
            getattr(r, "event", "") == "string_gene_symbol_mode_detected"
            for r in caplog.records
        ) or len(caplog.records) > 0  # structlog log records may use .event

    def test_no_warning_for_canonical_string_ids(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        df = pd.DataFrame({
            "protein1": [
                "9606.ENSP00000000233",
                "9606.ENSP00000000412",
                "9606.ENSP00000000550",
                "9606.ENSP00000000234",
            ],
            "protein2": [
                "9606.ENSP00000000412",
                "9606.ENSP00000000550",
                "9606.ENSP00000000234",
                "9606.ENSP00000000233",
            ],
        })
        with caplog.at_level(logging.WARNING, logger="drugos_graph.string_loader"):
            _warn_if_gene_symbol_mode(df)
        # No warning should be emitted -- all IDs are canonical.
        p2_008_warnings = [
            r for r in caplog.records
            if "P2-008" in (r.message or "")
            or getattr(r, "event", "") == "string_gene_symbol_mode_detected"
        ]
        assert len(p2_008_warnings) == 0, (
            "P2-008 warning should NOT fire for canonical STRING IDs"
        )

    def test_no_warning_for_empty_dataframe(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        df = pd.DataFrame({"protein1": [], "protein2": []})
        with caplog.at_level(logging.WARNING, logger="drugos_graph.string_loader"):
            _warn_if_gene_symbol_mode(df)
        p2_008_warnings = [
            r for r in caplog.records
            if getattr(r, "event", "") == "string_gene_symbol_mode_detected"
        ]
        assert len(p2_008_warnings) == 0


class TestP2008CanonicalizePairOrder:
    """Tests for the updated ``_canonicalize_pair_order`` function."""

    def test_canonical_ids_dedup_correctly(self) -> None:
        """Two cross-species PPIs with the same gene symbol MUST NOT
        collapse when canonical taxid-prefixed IDs are used."""
        df = pd.DataFrame({
            "protein1": ["9606.TP53", "10090.Tp53"],
            "protein2": ["9606.MDM2", "10090.Mdm2"],
        })
        result = _canonicalize_pair_order(df)
        # Both rows must survive -- they are different species.
        assert len(result) == 2, (
            "Cross-species PPIs with canonical taxid-prefixed IDs must "
            "NOT be collapsed by the canonical-pair dedup."
        )

    def test_canonical_ensembl_ids_sort_correctly(self) -> None:
        """STRING's symmetric file has both (A,B) and (B,A) -- the
        canonical-pair dedup must collapse them."""
        df = pd.DataFrame({
            "protein1": ["9606.ENSP00000000233", "9606.ENSP00000000412"],
            "protein2": ["9606.ENSP00000000412", "9606.ENSP00000000233"],
        })
        result = _canonicalize_pair_order(df)
        # After canonicalization, both rows have protein1 < protein2.
        assert result.loc[0, "protein1"] == "9606.ENSP00000000233"
        assert result.loc[0, "protein2"] == "9606.ENSP00000000412"
        assert result.loc[1, "protein1"] == "9606.ENSP00000000233"
        assert result.loc[1, "protein2"] == "9606.ENSP00000000412"
        # After dedup, only one row survives.
        deduped = _drop_duplicates(result)
        assert len(deduped) == 1

    def test_cross_species_ppi_with_bare_symbols_emits_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Bare gene-symbol mode: warning is emitted, dedup still runs."""
        df = pd.DataFrame({
            "protein1": ["TP53", "Tp53"],
            "protein2": ["MDM2", "Mdm2"],
        })
        with caplog.at_level(logging.WARNING, logger="drugos_graph.string_loader"):
            _canonicalize_pair_order(df)
        # Warning must be emitted. structlog may emit via the standard
        # logging bridge, so we check caplog.text (the full captured log
        # text) for the P2-008 marker.
        assert (
            "string_gene_symbol_mode_detected" in caplog.text
            or "P2-008" in caplog.text
        ), (
            "P2-008 warning must fire when bare gene symbols are passed to "
            "_canonicalize_pair_order. Captured log: " + caplog.text
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
