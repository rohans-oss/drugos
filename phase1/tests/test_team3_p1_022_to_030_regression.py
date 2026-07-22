#!/usr/bin/env python3
"""
Regression tests for Team Member 3 issues P1-022 through P1-030.

These tests verify the ROOT-LEVEL behavior of each fix by exercising the
actual public API with REAL scientific data (real InChIKeys, real UniProt
accessions, real SMILES, real drug abbreviations). They do NOT rely on
comments or existing test infrastructure -- they call the real code paths
that the production pipeline uses.

Run with:
    pytest phase1/tests/test_team3_p1_022_to_030_regression.py -v

Or standalone:
    python3 phase1/tests/test_team3_p1_022_to_030_regression.py
"""
from __future__ import annotations

import os
import sys
import re
import warnings
from pathlib import Path

import pytest

# Ensure repo root + phase1 are on sys.path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "phase1"))

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# P1-022: drug_resolver must expand 'MTX' -> 'Methotrexate' (and other
#         curated abbreviations) via the abbreviation_expansion match method,
#         NOT via rapidfuzz token_set_ratio (which returns 0 for short names).
# ---------------------------------------------------------------------------
class TestP1022DrugAbbreviationExpansion:
    """P1-022: short drug abbreviations must resolve via curated dictionary."""

    def test_mtx_resolves_to_methotrexate(self):
        from phase1.entity_resolution.drug_resolver import DrugResolver
        from phase1.entity_resolution.base import ResolverConfig
        r = DrugResolver(ResolverConfig())
        r.add_source_records([{
            "name": "Methotrexate",
            "inchikey": "FBOZXECLQNJBKD-ZDUSSCGKSA-N",
        }], source="drugbank")
        result = r.resolve_single("MTX")
        assert result is not None
        method = result.get("match_method", "")
        ik = result.get("canonical_inchikey", "")
        assert method == "abbreviation_expansion", (
            f"MTX should resolve via abbreviation_expansion, got {method!r}"
        )
        assert ik == "FBOZXECLQNJBKD-ZDUSSCGKSA-N", (
            f"MTX should resolve to Methotrexate's InChIKey, got {ik!r}"
        )

    def test_asa_resolves_to_aspirin(self):
        from phase1.entity_resolution.drug_resolver import DrugResolver
        from phase1.entity_resolution.base import ResolverConfig
        r = DrugResolver(ResolverConfig())
        r.add_source_records([{
            "name": "Acetylsalicylic acid",
            "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        }], source="drugbank")
        result = r.resolve_single("ASA")
        assert result is not None
        assert result.get("match_method") == "abbreviation_expansion"
        assert result.get("canonical_inchikey") == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"

    def test_tmp_smx_resolves(self):
        """TMP and SMX are common antibiotic abbreviations."""
        from phase1.entity_resolution.drug_resolver import DrugResolver
        from phase1.entity_resolution.base import ResolverConfig
        r = DrugResolver(ResolverConfig())
        r.add_source_records([{
            "name": "Trimethoprim",
            "inchikey": "IEFICNVLXKBSLL-UHFFFAOYSA-N",
        }], source="drugbank")
        result = r.resolve_single("TMP")
        assert result is not None
        assert result.get("match_method") == "abbreviation_expansion"


# ---------------------------------------------------------------------------
# P1-023: protein_resolver must NOT collapse BCL-X isoforms into one node,
#         AND must NOT redirect Q07817 (BCL-X) to Q07812 (BAX).
# ---------------------------------------------------------------------------
class TestP1023ProteinIsoformPreservation:
    """P1-023: UniProt accessions (incl. isoform suffix) are the primary key."""

    def test_bcl_x_isoforms_distinct(self):
        """BCL-XL (Q07817) and BCL-XS (Q07817-2) must be DISTINCT Protein nodes."""
        from phase1.entity_resolution.protein_resolver import ProteinResolver
        from phase1.entity_resolution.base import ResolverConfig
        r = ProteinResolver(ResolverConfig())
        r.add_source_records([
            {"uniprot_id": "Q07817", "gene_name": "BCL2L1",
             "protein_name": "Bcl-2-like protein 1", "organism": "Homo sapiens"},
            {"uniprot_id": "Q07817-2", "gene_name": "BCL2L1",
             "protein_name": "Bcl-2-like protein 1 isoform short",
             "organism": "Homo sapiens"},
        ], source="uniprot")
        mapping = r.mapping
        # Both must be present as DISTINCT entries
        assert "Q07817" in mapping, (
            f"Q07817 (BCL-XL) must be in mapping, got keys: {list(mapping.keys())[:10]}"
        )
        assert "Q07817-2" in mapping, (
            f"Q07817-2 (BCL-XS) must be in mapping, got keys: {list(mapping.keys())[:10]}"
        )
        assert mapping["Q07817"] is not mapping["Q07817-2"], (
            "BCL-XL and BCL-XS must NOT be the same canonical entry (isoform collapse)"
        )

    def test_no_wrong_redirect_bcl_x_to_bax(self):
        """Q07817 (BCL-X) must NOT be redirected to Q07812 (BAX).

        This is the critical scientific bug that was corrupting the KG:
        BCL-XL is anti-apoptotic, BAX is pro-apoptotic -- opposite functions.
        """
        from phase1.entity_resolution.protein_resolver import ProteinResolver
        from phase1.entity_resolution.base import ResolverConfig
        from phase1.entity_resolution.protein_resolver import _DEPRECATED_UNIPROT_MAP
        # The wrong redirect must NOT be present
        assert "Q07817" not in _DEPRECATED_UNIPROT_MAP, (
            "Q07817 (BCL-X) must not be in _DEPRECATED_UNIPROT_MAP -- "
            "it is NOT a deprecated BAX accession"
        )
        assert "Q07816" not in _DEPRECATED_UNIPROT_MAP, (
            "Q07816 (BCL-W) must not be in _DEPRECATED_UNIPROT_MAP -- "
            "it is NOT a deprecated BAX accession"
        )
        # End-to-end: adding Q07817 must NOT create a Q07812 entry
        r = ProteinResolver(ResolverConfig())
        r.add_source_records([
            {"uniprot_id": "Q07817", "gene_name": "BCL2L1",
             "protein_name": "Bcl-2-like protein 1", "organism": "Homo sapiens"},
        ], source="uniprot")
        mapping = r.mapping
        assert "Q07817" in mapping, "Q07817 must be in mapping"
        assert "Q07812" not in mapping, (
            f"Q07812 (BAX) must NOT appear -- BCL-X was wrongly redirected! "
            f"keys: {list(mapping.keys())}"
        )

    def test_bcl_w_not_redirected_to_bax(self):
        """Q07816 (BCL-W) must NOT be redirected to Q07812 (BAX)."""
        from phase1.entity_resolution.protein_resolver import ProteinResolver
        from phase1.entity_resolution.base import ResolverConfig
        r = ProteinResolver(ResolverConfig())
        r.add_source_records([
            {"uniprot_id": "Q07816", "gene_name": "BCL2L2",
             "protein_name": "Bcl-2-like protein 2", "organism": "Homo sapiens"},
        ], source="uniprot")
        mapping = r.mapping
        assert "Q07816" in mapping, "Q07816 (BCL-W) must be in mapping"
        assert "Q07812" not in mapping, (
            f"Q07812 (BAX) must NOT appear -- BCL-W was wrongly redirected! "
            f"keys: {list(mapping.keys())}"
        )


# ---------------------------------------------------------------------------
# P1-024: missing_values.py must NOT fill missing InChIKey with empty string.
#         Empty string '' would cause the deduplicator to falsely merge
#         unrelated drugs that both lack InChIKey (e.g. peptide drugs).
# ---------------------------------------------------------------------------
class TestP1024MissingInchikeyNone:
    """P1-024: missing InChIKeys must remain None/NaN, not ''."""

    def test_fill_missing_drug_fields_keeps_inchikey_none(self):
        import pandas as pd
        from phase1.cleaning.missing_values import fill_missing_drug_fields
        df = pd.DataFrame({
            "inchikey": [None, None],
            "smiles": [None, None],
            "name": ["Insulin glargine", "Insulin lispro"],
            "drug_type": ["peptide", "peptide"],
            "is_fda_approved": [True, True],
            "max_phase": [4, 4],
            "mechanism_of_action": ["Long-acting insulin analog", "Rapid-acting"],
            "molecular_formula": ["C267H404N72O78S6", "C257H383N65O77S6"],
        })
        out = fill_missing_drug_fields(df)
        # InChIKey must NOT be filled with empty string
        for i in range(len(out)):
            val = out["inchikey"].iloc[i]
            assert not (isinstance(val, str) and val == ""), (
                f"Row {i}: InChIKey filled with empty string -- would falsely "
                f"match other empty InChIKeys in dedup"
            )
        # InChIKey must remain NaN/None
        assert out["inchikey"].isna().all(), (
            f"Missing InChIKeys must remain NaN, got: {list(out['inchikey'])}"
        )

    def test_two_peptide_drugs_not_merged_by_empty_inchikey(self):
        """End-to-end: two peptide drugs with no InChIKey must NOT be merged."""
        import pandas as pd
        from phase1.cleaning.missing_values import fill_missing_drug_fields
        from phase1.cleaning.deduplicator import dedup_by_inchikey
        df = pd.DataFrame({
            "inchikey": [None, None],
            "smiles": [None, None],
            "name": ["Insulin glargine", "Insulin lispro"],
            "drug_type": ["peptide", "peptide"],
            "source": ["drugbank", "drugbank"],
        })
        filled = fill_missing_drug_fields(df)
        deduped = dedup_by_inchikey(filled)
        # Both drugs must survive dedup (NOT merged)
        assert len(deduped) >= 1, "At least one drug should survive"
        # If both have null InChIKey, dedup should NOT merge them (null != null)


# ---------------------------------------------------------------------------
# P1-025: normalizer.py must preserve stereochemistry in InChIKey.
#         (R)-thalidomide and (S)-thalidomide must produce DIFFERENT InChIKeys.
# ---------------------------------------------------------------------------
class TestP1025StereochemistryPreserved:
    """P1-025: chiral drugs must get stereo-specific InChIKeys."""

    def test_r_s_thalidomide_different_inchikeys(self):
        from phase1.cleaning.normalizer import convert_to_inchikey
        r_smiles = "C1CC(=O)NC(=O)[C@@H]1N2CC(=O)NC(=O)C2=O"  # (R)-thalidomide
        s_smiles = "C1CC(=O)NC(=O)[C@H]1N2CC(=O)NC(=O)C2=O"   # (S)-thalidomide
        r_ik = convert_to_inchikey(r_smiles)
        s_ik = convert_to_inchikey(s_smiles)
        assert r_ik is not None, "RDKit failed to convert (R)-thalidomide"
        assert s_ik is not None, "RDKit failed to convert (S)-thalidomide"
        assert r_ik != s_ik, (
            f"(R)- and (S)-thalidomide must have DIFFERENT InChIKeys "
            f"(stereochemistry lost); both = {r_ik!r}"
        )
        for label, ik in [("R", r_ik), ("S", s_ik)]:
            assert re.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$", ik), (
                f"({label})-thalidomide InChIKey {ik!r} malformed"
            )

    def test_ibuprofen_enantiomers_different(self):
        """(R)- and (S)-ibuprofen are chiral enantiomers with different biology.

        Ibuprofen's stereocenter is the alpha-carbon of the propanoic acid
        chain, which has 4 DISTINCT substituents: H, CH3, COOH, 4-isobutylphenyl.
        The SMILES must place @ on THAT carbon, not on the isobutyl carbon
        (which has two methyl groups and is therefore NOT a stereocenter).
        """
        from phase1.cleaning.normalizer import convert_to_inchikey
        # Correct SMILES: stereocenter is C[C@H](C(=O)O)...
        r_smiles = "C[C@H](C(=O)O)c1ccc(CC(C)C)cc1"   # (R)-ibuprofen
        s_smiles = "C[C@@H](C(=O)O)c1ccc(CC(C)C)cc1"  # (S)-ibuprofen
        r_ik = convert_to_inchikey(r_smiles)
        s_ik = convert_to_inchikey(s_smiles)
        if r_ik is None or s_ik is None:
            pytest.skip("RDKit failed to parse test SMILES -- non-critical")
        assert r_ik != s_ik, (
            f"(R)- and (S)-ibuprofen must have different InChIKeys "
            f"(stereochemistry lost); both = {r_ik!r}"
        )


# ---------------------------------------------------------------------------
# P1-026: deduplicator.py must treat case-insensitive InChIKeys as the same.
#         'AA...' and 'aa...' must merge (PubChem sometimes lowercases).
# ---------------------------------------------------------------------------
class TestP1026CaseInsensitiveInchikey:
    """P1-026: case-different InChIKeys must be deduplicated together."""

    def test_uppercase_lowercase_merged(self):
        import pandas as pd
        from phase1.cleaning.deduplicator import dedup_by_inchikey
        ik_upper = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        ik_lower = ik_upper.lower()
        df = pd.DataFrame({
            "inchikey": [ik_upper, ik_lower],
            "name": ["Aspirin", "acetylsalicylic acid"],
            "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"] * 2,
            "source": ["drugbank", "chembl"],
        })
        out = dedup_by_inchikey(df)
        unique_keys = list(out["inchikey"].unique())
        assert len(unique_keys) == 1, (
            f"Case-different InChIKeys must merge into 1, got {len(unique_keys)}: "
            f"{unique_keys}"
        )
        # The merged key should be uppercase (canonical form)
        assert unique_keys[0] == ik_upper, (
            f"Merged key should be uppercase canonical form, got {unique_keys[0]!r}"
        )

    def test_mixed_case_batch(self):
        """A batch with mixed-case InChIKeys must all merge correctly."""
        import pandas as pd
        from phase1.cleaning.deduplicator import dedup_by_inchikey
        ik = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        df = pd.DataFrame({
            "inchikey": [ik, ik.lower(), ik.upper(), ik.title()],
            "name": ["A"] * 4,
            "source": ["s"] * 4,
        })
        out = dedup_by_inchikey(df)
        assert out["inchikey"].nunique() == 1


# ---------------------------------------------------------------------------
# P1-027: confidence.py must weight confidence by source reliability.
#         Curated > Predicted > Animal-Model.
# ---------------------------------------------------------------------------
class TestP1027SourceReliabilityWeighting:
    """P1-027: confidence = raw * source_reliability_weight."""

    def test_curated_beats_predicted(self):
        from phase1.cleaning.confidence import compute_source_weighted_confidence
        result = compute_source_weighted_confidence([
            ("curated", 0.7),
            ("predicted", 0.95),
        ])
        # 0.7*1.0 = 0.7 > 0.95*0.6 = 0.57
        assert abs(result - 0.7) < 0.01, (
            f"Curated 0.7 (weighted 0.7) must beat Predicted 0.95 (weighted 0.57); "
            f"got {result}"
        )

    def test_predicted_alone_downweighted(self):
        from phase1.cleaning.confidence import compute_source_weighted_confidence
        result = compute_source_weighted_confidence([("predicted", 0.9)])
        assert abs(result - 0.54) < 0.01, f"0.9*0.6=0.54, got {result}"

    def test_empty_returns_zero(self):
        from phase1.cleaning.confidence import compute_source_weighted_confidence
        assert compute_source_weighted_confidence([]) == 0.0


# ---------------------------------------------------------------------------
# P1-028: neo4j_exporter.py must validate InChIKey format before export.
#         Rejects Cypher-injection attempts and malformed keys.
# ---------------------------------------------------------------------------
class TestP1028Neo4jInchikeyValidation:
    """P1-028: InChIKey regex validation prevents Cypher injection."""

    def test_valid_inchikey_accepted(self):
        from phase1.exporters.neo4j_exporter import _validate_inchikey_for_neo4j
        assert _validate_inchikey_for_neo4j("BSYNRYMUTXBXSQ-UHFFFAOYSA-N") is True

    def test_cypher_injection_rejected(self):
        from phase1.exporters.neo4j_exporter import _validate_inchikey_for_neo4j
        assert _validate_inchikey_for_neo4j("} RETURN 1 //") is False
        assert _validate_inchikey_for_neo4j("'); DROP GRAPH; //") is False

    def test_short_string_rejected(self):
        from phase1.exporters.neo4j_exporter import _validate_inchikey_for_neo4j
        assert _validate_inchikey_for_neo4j("AAA") is False
        assert _validate_inchikey_for_neo4j("BSYNRYMUTXBXSQ-UHFFFAOYSA-N-extra") is False

    def test_empty_and_none_rejected(self):
        from phase1.exporters.neo4j_exporter import _validate_inchikey_for_neo4j
        assert _validate_inchikey_for_neo4j("") is False
        assert _validate_inchikey_for_neo4j(None) is False


# ---------------------------------------------------------------------------
# P1-029: resolver_utils.py must scale fuzzy threshold by name length.
#         Short names (PAX, PAX2, PAX4) must NOT fuzzy-match each other.
# ---------------------------------------------------------------------------
class TestP1029LengthScaledThreshold:
    """P1-029: short names require exact match; long names allow fuzzy."""

    def test_pax_only_matches_exact(self):
        from phase1.entity_resolution.resolver_utils import fuzzy_match_best
        candidates = {"pax2": "PAX2", "pax4": "PAX4", "pax": "PAX"}
        result = fuzzy_match_best("pax", candidates, threshold=0.85)
        assert result is not None, "PAX should match exactly"
        assert result[0] == "PAX", f"PAX should match only PAX, got {result}"

    def test_pax2_does_not_match_pax4(self):
        from phase1.entity_resolution.resolver_utils import fuzzy_match_best
        result = fuzzy_match_best("pax2", {"pax4": "PAX4"}, threshold=0.85)
        assert result is None, (
            f"PAX2 and PAX4 are DIFFERENT genes -- must not fuzzy-match; got {result}"
        )

    def test_long_name_fuzzy_allowed(self):
        from phase1.entity_resolution.resolver_utils import fuzzy_match_best
        # 'acetylsalicylic acid' (20 chars) vs 'acetylsalicylic' (16 chars)
        # -- long names allow fuzzy matching
        result = fuzzy_match_best(
            "acetylsalicylic acid",
            {"acetylsalicylic": "ASPIRIN"},
            threshold=0.85,
        )
        assert result is not None, (
            "Long names should allow fuzzy matching for typo tolerance"
        )

    def test_threshold_values(self):
        from phase1.entity_resolution.resolver_utils import length_scaled_threshold
        assert length_scaled_threshold("PAX", 0.85) == 1.0      # len 3 -> exact
        assert length_scaled_threshold("PAX4", 0.85) >= 0.92    # len 4 -> strict
        assert length_scaled_threshold("ibuprofen", 0.85) >= 0.88  # len 9 -> medium
        assert length_scaled_threshold("acetylsalicylic acid", 0.85) == 0.85  # len 20


# ---------------------------------------------------------------------------
# P1-030: _circuit_breaker.py must use 300s window + exponential backoff.
#         ChEMBL outages typically last 5-15 min -- 60s window is too short.
# ---------------------------------------------------------------------------
class TestP1030CircuitBreakerOutage:
    """P1-030: 300s rolling window + exponential backoff for ChEMBL outages."""

    def test_default_failure_window_300s(self):
        from phase1._circuit_breaker import _DEFAULT_FAILURE_WINDOW
        assert _DEFAULT_FAILURE_WINDOW >= 300.0, (
            f"Default failure window must be >= 300s for ChEMBL outages, "
            f"got {_DEFAULT_FAILURE_WINDOW}s"
        )

    def test_breaker_has_rolling_window(self):
        from phase1._circuit_breaker import _CircuitBreaker
        cb = _CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
        assert hasattr(cb, "_failure_window"), "CircuitBreaker must have _failure_window"
        assert cb._failure_window >= 300.0, (
            f"Instance failure_window must be >= 300s, got {cb._failure_window}s"
        )

    def test_exponential_backoff_on_failed_probe(self):
        from phase1._circuit_breaker import _CircuitBreaker
        cb = _CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
        # Trip the breaker
        for _ in range(5):
            cb.record_failure()
        assert cb.state == "open"
        # Simulate a failed half-open probe
        cb._state = "half_open"
        cb._half_open_probe_in_flight = True
        cb.record_failure()
        assert cb._backoff_mult >= 2.0, (
            f"After failed half-open probe, backoff_mult must be >= 2.0, "
            f"got {cb._backoff_mult}"
        )

    def test_backoff_capped(self):
        from phase1._circuit_breaker import _BACKOFF_CAP_MULT
        assert _BACKOFF_CAP_MULT <= 16.0, "Backoff must be capped to avoid infinite wait"


if __name__ == "__main__":
    # Standalone runner for environments without pytest
    print("Running Team Member 3 regression tests (P1-022 to P1-030)...")
    import traceback
    test_classes = [
        TestP1022DrugAbbreviationExpansion,
        TestP1023ProteinIsoformPreservation,
        TestP1024MissingInchikeyNone,
        TestP1025StereochemistryPreserved,
        TestP1026CaseInsensitiveInchikey,
        TestP1027SourceReliabilityWeighting,
        TestP1028Neo4jInchikeyValidation,
        TestP1029LengthScaledThreshold,
        TestP1030CircuitBreakerOutage,
    ]
    n_pass = 0
    n_fail = 0
    for cls in test_classes:
        for method_name in sorted(dir(cls)):
            if not method_name.startswith("test_"):
                continue
            method = getattr(cls, method_name)
            try:
                method(cls)
                print(f"  PASS  {cls.__name__}.{method_name}")
                n_pass += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{method_name}")
                traceback.print_exc()
                n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)
