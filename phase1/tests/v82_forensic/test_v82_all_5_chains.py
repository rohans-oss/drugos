"""v82 Forensic Root-Fix Tests -- 5 compound/cross-file chains.

Each test reproduces the EXACT failure scenario from the issue audit
and verifies the root-level fix holds. Tests exercise REAL production
code paths (no mocks of the functions under test).

Chain-1: drug_resolver._normalize_inchikey strips protonation suffix
Chain-2: STRING aliases populate _string_to_uniprot via run.py path
Chain-3: O(1) alias-uniprot index prevents O(N*M) promotion scan
Chain-4: pipeline propagates activity_censored; deduplicator respects it
Chain-5: classify_confidence accepts negative scores by default
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

# Ensure phase1 is importable
PHASE1 = Path(__file__).resolve().parents[2]
if str(PHASE1) not in sys.path:
    sys.path.insert(0, str(PHASE1))


# ============================================================================
# CHAIN-1: InChIKey protonation-suffix stripping
# ============================================================================

class TestChain1InchikeySuffix:
    """Chain-1: suffixed + canonical InChIKeys collapse to 1 Compound node."""

    def test_normalize_inchikey_strips_protonation_suffix(self):
        """_normalize_inchikey must produce the same key for suffixed and canonical forms."""
        from entity_resolution.drug_resolver import _normalize_inchikey
        suffixed = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N-a"  # 29 chars, with -a suffix
        canonical = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"   # 27 chars, canonical
        assert _normalize_inchikey(suffixed) == _normalize_inchikey(canonical), (
            "Suffixed and canonical InChIKeys must normalize to the same key"
        )

    def test_build_mapping_collapses_suffixed_and_canonical(self):
        """build_mapping must produce 1 canonical entry for suffixed+canonical pair."""
        from entity_resolution.drug_resolver import DrugResolver
        suffixed = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N-a"
        canonical = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        r = DrugResolver()
        df_a = pd.DataFrame([{"inchikey": suffixed, "name": "DrugA"}])
        df_b = pd.DataFrame([{"inchikey": canonical, "name": "DrugB"}])
        mapping = r.build_mapping(chembl_df=df_b, drugbank_df=df_a, pubchem_df=pd.DataFrame())
        assert len(mapping) == 1, (
            f"Expected 1 canonical Compound node, got {len(mapping)} -- "
            "duplicate nodes would split Compound->Protein edges in the KG"
        )

    def test_synthetic_keys_not_stripped(self):
        """SYNTH-prefixed keys must NOT be stripped (they have their own format)."""
        from entity_resolution.drug_resolver import _normalize_inchikey
        synth = "SYNTH-ABCDEF0123-ABCDEF0123-A"
        result = _normalize_inchikey(synth)
        assert result == synth, f"SYNTH key must be preserved, got {result!r}"


# ============================================================================
# CHAIN-2: STRING aliases -> _string_to_uniprot population
# ============================================================================

class TestChain2StringAliases:
    """Chain-2: STRING aliases populate _string_to_uniprot; resolve_single works."""

    def test_string_aliases_populate_index(self):
        """build_mapping with string_aliases_df must populate _string_to_uniprot."""
        from entity_resolution.protein_resolver import ProteinResolver
        aliases_df = pd.DataFrame([
            {"string_id": "9606.ENSP00000000233", "uniprot_id": "Q5JTK92",
             "source": "UniProt_AC", "source_database": "UniProt"},
            {"string_id": "9606.ENSP00000000442", "uniprot_id": "P04637",
             "source": "UniProt_AC", "source_database": "UniProt"},
        ])
        r = ProteinResolver()
        r.build_mapping(pd.DataFrame(), string_aliases_df=aliases_df, string_df=pd.DataFrame())
        assert len(r._string_to_uniprot) == 2, (
            f"_string_to_uniprot must have 2 entries, got {len(r._string_to_uniprot)}"
        )

    def test_resolve_single_by_string_id_works(self):
        """resolve_single(string_id=...) must return a non-None entry."""
        from entity_resolution.protein_resolver import ProteinResolver
        aliases_df = pd.DataFrame([
            {"string_id": "9606.ENSP00000000233", "uniprot_id": "Q5JTK92",
             "source": "UniProt_AC", "source_database": "UniProt"},
        ])
        r = ProteinResolver()
        r.build_mapping(pd.DataFrame(), string_aliases_df=aliases_df, string_df=pd.DataFrame())
        result = r.resolve_single(string_id="9606.ENSP00000000233")
        assert result is not None, (
            "resolve_single(string_id=...) must return the provisional entry -- "
            "if this returns None, STRING-only proteins are unresolved in the KG"
        )


# ============================================================================
# CHAIN-3: O(1) alias-uniprot index prevents O(N*M) promotion scan
# ============================================================================

class TestChain3PromotionPerformance:
    """Chain-3: STRING-alias provisionals promote in O(1), not O(N*M)."""

    def test_alias_uniprot_index_populated(self):
        """_provisional_by_alias_uniprot must be populated after STRING alias ingestion."""
        from entity_resolution.protein_resolver import ProteinResolver
        aliases = [
            {"string_id": f"9606.ENSP{i:011d}", "uniprot_id": f"P{i:05d}",
             "source": "UniProt_AC", "source_database": "UniProt"}
            for i in range(100)
        ]
        r = ProteinResolver()
        r.build_mapping(pd.DataFrame(), string_aliases_df=pd.DataFrame(aliases), string_df=pd.DataFrame())
        assert len(r._provisional_by_alias_uniprot) == 100, (
            f"Expected 100 entries in _provisional_by_alias_uniprot, "
            f"got {len(r._provisional_by_alias_uniprot)}"
        )

    def test_promotion_via_alias_uniprot_index(self):
        """1000 UniProt records + 1000 STRING-alias provisionals must complete < 2s."""
        from entity_resolution.protein_resolver import ProteinResolver
        r = ProteinResolver()
        aliases = [
            {"string_id": f"9606.ENSP{i:011d}", "uniprot_id": f"P{i:05d}",
             "source": "UniProt_AC", "source_database": "UniProt"}
            for i in range(1000)
        ]
        r.build_mapping(pd.DataFrame(), string_aliases_df=pd.DataFrame(aliases), string_df=pd.DataFrame())
        uniprot_records = [
            {"uniprot_id": f"P{i:05d}", "gene_symbol": f"GENE{i}", "organism": "Homo sapiens"}
            for i in range(1000)
        ]
        t0 = time.perf_counter()
        r.add_uniprot_records(uniprot_records)
        elapsed = time.perf_counter() - t0
        stats = r.get_stats()
        # O(N*M) would take 10+ seconds; O(1) index takes < 1 second.
        assert elapsed < 2.0, (
            f"Promotion took {elapsed:.3f}s -- O(N*M) fallback likely still firing"
        )
        assert stats["records_matched"] == 1000, (
            f"Expected 1000 promotions via alias-uniprot index, got {stats['records_matched']}"
        )

    def test_promotion_unregisters_from_alias_index(self):
        """After promotion, the synthetic uid must be removed from the alias index."""
        from entity_resolution.protein_resolver import ProteinResolver
        aliases = [
            {"string_id": "9606.ENSP00000000233", "uniprot_id": "P12345",
             "source": "UniProt_AC", "source_database": "UniProt"},
        ]
        r = ProteinResolver()
        r.build_mapping(pd.DataFrame(), string_aliases_df=pd.DataFrame(aliases), string_df=pd.DataFrame())
        assert len(r._provisional_by_alias_uniprot) == 1
        # Ingest the matching UniProt record -- should promote.
        r.add_uniprot_records([{"uniprot_id": "P12345", "gene_symbol": "TEST", "organism": "Homo sapiens"}])
        # The promoted synthetic uid should be removed from the index bucket.
        bucket = r._provisional_by_alias_uniprot.get("P12345", [])
        synthetic_uids = [uid for uid in bucket if r.is_synthetic_uid(uid)]
        assert len(synthetic_uids) == 0, (
            f"Promoted uid must be unregistered from alias index, found {synthetic_uids}"
        )


# ============================================================================
# CHAIN-4: p-scale censor preservation through the pipeline
# ============================================================================

class TestChain4CensorPreservation:
    """Chain-4: pipeline propagates censored flag; deduplicator respects it."""

    def test_normalizer_preserves_censored_on_pscale(self):
        """normalize_activity_value must preserve censored=True for '>6' pIC50."""
        from cleaning.normalizer import normalize_activity_value
        r = normalize_activity_value(">6", "", activity_type="pIC50")
        assert r.value == 1000.0, f"Expected 1000.0 nM, got {r.value}"
        assert r.censored is True, "Censored flag must be preserved through p-scale conversion"
        assert r.censor_direction == ">", f"Expected '>', got {r.censor_direction!r}"

    def test_dedup_prefers_precise_over_censored(self):
        """dedup_interactions must keep precise 500nM over censored 1000nM."""
        from cleaning.normalizer import normalize_activity_value
        from cleaning.deduplicator import dedup_interactions
        r1 = normalize_activity_value(">6", "", activity_type="pIC50")  # censored 1000nM
        r2 = normalize_activity_value(500.0, "nM", activity_type="IC50")  # precise 500nM
        df = pd.DataFrame([
            {"drug_id": "D1", "target_id": "T1", "activity_value": r1.value,
             "activity_units": r1.unit, "activity_type": "IC50",
             "activity_censored": r1.censored, "activity_censor_direction": r1.censor_direction},
            {"drug_id": "D1", "target_id": "T1", "activity_value": r2.value,
             "activity_units": r2.unit, "activity_type": "IC50",
             "activity_censored": r2.censored, "activity_censor_direction": r2.censor_direction},
        ])
        deduped = dedup_interactions(df, keys=["drug_id", "target_id"], keep="best", handle_censored=True)
        assert len(deduped) == 1
        assert float(deduped["activity_value"].iloc[0]) == 500.0, (
            "Precise 500nM must win over censored 1000nM -- censored values are deprioritized"
        )

    def test_dedup_legacy_fallback_no_crash(self):
        """dedup_interactions must not crash when activity_censored column is absent (legacy)."""
        from cleaning.deduplicator import dedup_interactions
        df = pd.DataFrame([
            {"drug_id": "D1", "target_id": "T1", "activity_value": 1000.0, "activity_type": "IC50"},
            {"drug_id": "D1", "target_id": "T1", "activity_value": 500.0, "activity_type": "IC50"},
        ])
        deduped = dedup_interactions(df, keys=["drug_id", "target_id"], keep="best", handle_censored=True)
        assert len(deduped) == 1


# ============================================================================
# CHAIN-5: negative GDA scores + classify_confidence
# ============================================================================

class TestChain5NegativeScores:
    """Chain-5: classify_confidence accepts negatives by default."""

    def test_classify_confidence_accepts_negative_default(self):
        """classify_confidence(-0.3) must return 'weak' by default (no crash)."""
        from cleaning.confidence import classify_confidence
        tier = classify_confidence(-0.3)
        assert tier == "weak", f"Expected 'weak' for -0.3, got {tier!r}"

    def test_validate_gda_then_classify_no_crash(self):
        """Full path: validate_gda_scores(preserve_direction=True) + classify_confidence."""
        from cleaning.missing_values import validate_gda_scores
        from cleaning.confidence import classify_confidence
        df = pd.DataFrame({
            "gene_id": ["G1"], "disease_id": ["D1"],
            "score": [-0.3], "source": ["disgenet"],
        })
        df_v = validate_gda_scores(df, score_range=(-1.0, 1.0), preserve_direction=True, source="disgenet")
        tier = classify_confidence(df_v["score"].iloc[0])
        assert tier == "weak"
        assert df_v["score"].iloc[0] == -0.3, "Score must be preserved (not clipped) in protective mode"

    def test_classify_confidence_rejects_below_minus_one(self):
        """Scores below -1.0 must still raise (outside protective-association range)."""
        from cleaning.confidence import classify_confidence
        with pytest.raises(ValueError, match="score=.*< -1"):
            classify_confidence(-1.5)

    def test_classify_confidence_rejects_above_one(self):
        """Scores above 1.0 must still raise."""
        from cleaning.confidence import classify_confidence
        with pytest.raises(ValueError, match="score=.*> 1"):
            classify_confidence(1.5)

    def test_allow_negative_param_removed(self):
        """v84 BUG #49: the allow_negative parameter is removed entirely.

        Negative scores in [-1, 0) are ALWAYS classified as 'weak' (the
        lowest tier). The _score_direction lineage column preserves the
        sign for downstream ranking. Passing allow_negative=... now raises
        TypeError since the parameter no longer exists.
        """
        from cleaning.confidence import classify_confidence
        # Negative score classifies as "weak" (no crash, no warning).
        assert classify_confidence(-0.5) == "weak"
        # Passing the removed parameter raises TypeError.
        with pytest.raises(TypeError, match="allow_negative"):
            classify_confidence(0.5, allow_negative=False)
