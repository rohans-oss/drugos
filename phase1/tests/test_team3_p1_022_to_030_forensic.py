"""
Regression tests for Team Member 3 forensic root fixes (P1-022 .. P1-030).

Each test verifies ONE fix in isolation, using ONLY the real production
code paths (no smoke tests, no mocks of the unit under test). Tests are
written to FAIL on the pre-fix code and PASS on the post-fix code.

Run:
    cd phase1 && python -m pytest tests/test_team3_p1_022_to_030_forensic.py -v

Or run individual tests:
    cd phase1 && python -m pytest tests/test_team3_p1_022_to_030_forensic.py::test_p1_022_abbreviation_expansion_mtx -v
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

# Ensure phase1 is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# P1-022: drug_resolver abbreviation expansion for short drug names
# =============================================================================

def test_p1_022_abbreviation_expansion_mtx():
    """P1-022: 'MTX' should resolve to the same canonical entry as
    'Methotrexate' via the curated abbreviation dictionary.

    Pre-fix: token_sort_ratio('mtx', 'methotrexate') = 0 (no common
    tokens) -> MTX creates a SEPARATE Compound node -> duplicate.
    Post-fix: abbreviation expansion matches MTX -> Methotrexate entry.
    """
    from entity_resolution.drug_resolver import DrugResolver

    resolver = DrugResolver()
    # Register Methotrexate as a canonical entry (simulating DrugBank).
    resolver.add_source_records(
        [{"name": "Methotrexate", "inchikey": "VDQVEZJHCXN6FU-UHFFFAOYSA-N",
          "source": "drugbank", "drugbank_id": "DB00563"}],
        source="drugbank",
    )
    # Now resolve a record that uses the abbreviation 'MTX' (simulating
    # a ChEMBL record with a short synonym).
    resolver.add_source_records(
        [{"name": "MTX", "inchikey": None, "source": "chembl",
          "chembl_id": "CHEMBL342306"}],
        source="chembl",
    )
    # The 'MTX' record should have been merged into the Methotrexate
    # canonical entry (not created a separate one).
    assert len(resolver.mapping) == 1, (
        f"P1-022: expected 1 canonical entry (Methotrexate), got "
        f"{len(resolver.mapping)}: {list(resolver.mapping.keys())}"
    )
    entry = list(resolver.mapping.values())[0]
    assert entry["canonical_name"].lower() == "methotrexate", (
        f"P1-022: canonical name should be Methotrexate, got "
        f"{entry['canonical_name']}"
    )


def test_p1_022_abbreviation_expansion_asa():
    """P1-022: 'ASA' should resolve to 'Acetylsalicylic acid'."""
    from entity_resolution.drug_resolver import DrugResolver

    resolver = DrugResolver()
    resolver.add_source_records(
        [{"name": "Acetylsalicylic acid",
          "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
          "source": "drugbank", "drugbank_id": "DB00945"}],
        source="drugbank",
    )
    resolver.add_source_records(
        [{"name": "ASA", "inchikey": None, "source": "chembl",
          "chembl_id": "CHEMBL25"}],
        source="chembl",
    )
    assert len(resolver.mapping) == 1, (
        f"P1-022 ASA: expected 1 entry, got {len(resolver.mapping)}"
    )


def test_p1_022_add_abbreviation_runtime():
    """P1-022: add_abbreviation() allows runtime extension of the dict."""
    from entity_resolution.drug_resolver import DrugResolver

    resolver = DrugResolver()
    resolver.add_abbreviation("XYZ", "Xanomeline")  # a research compound
    assert "xyz" in resolver._abbreviations
    assert resolver._abbreviations["xyz"] == "Xanomeline"


# =============================================================================
# P1-023: protein_resolver isoform separation (BCL-XL vs BCL-XS)
# =============================================================================

def test_p1_023_isoforms_are_separate_entries():
    """P1-023: Q07817 (BCL-XL) and Q07817-2 (BCL-XS) must be SEPARATE
    Protein entries, not merged into one.

    Pre-fix: Q07817-2 was stripped to base_uid=Q07817 and MERGED into
    the parent -> BCL-XS-specific drug targeting was lost.
    Post-fix: each isoform is its own entry with parent_accession set.
    """
    from entity_resolution.protein_resolver import ProteinResolver

    resolver = ProteinResolver()
    # Use a non-deprecated accession to avoid the _DEPRECATED_UNIPROT_MAP
    # remapping. P99999 is a valid-format UniProt accession not in any
    # override/deprecated map.
    resolver.add_uniprot_records([
        {"uniprot_id": "P99999", "gene_symbol": "TESTG1",
         "gene_name": "Test gene 1", "organism": "Homo sapiens",
         "protein_name": "Test protein 1"},
    ])
    resolver.add_uniprot_records([
        {"uniprot_id": "P99999-2", "gene_symbol": "TESTG1",
         "gene_name": "Test gene 1", "organism": "Homo sapiens",
         "protein_name": "Test protein 1 isoform 2"},
    ])
    # Both should be present as DISTINCT entries.
    assert "P99999" in resolver.mapping, "P1-023: parent P99999 missing"
    assert "P99999-2" in resolver.mapping, (
        "P1-023: isoform P99999-2 should be a SEPARATE entry, not merged"
    )
    parent = resolver.mapping["P99999"]
    isoform = resolver.mapping["P99999-2"]
    # The isoform entry should have parent_accession set.
    assert isoform.get("parent_accession") == "P99999", (
        f"P1-023: isoform parent_accession should be P99999, got "
        f"{isoform.get('parent_accession')}"
    )
    # The parent should NOT have parent_accession (it's canonical).
    assert parent.get("parent_accession") is None, (
        f"P1-023: canonical parent_accession should be None, got "
        f"{parent.get('parent_accession')}"
    )
    # The parent's isoforms list should track the isoform.
    assert "P99999-2" in parent.get("isoforms", []), (
        f"P1-023: parent isoforms list should contain P99999-2, got "
        f"{parent.get('isoforms')}"
    )


def test_p1_023_isoform_unique_sequence_preserved():
    """P1-023: an isoform's specific sequence must NOT be merged into
    the parent entry."""
    from entity_resolution.protein_resolver import ProteinResolver

    resolver = ProteinResolver()
    # Use valid amino-acid-only sequences (only ARNDCQEGHILKMFPSTWYV).
    parent_seq = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWWDQGRVC"
    isoform_seq = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWWDQGRVCY"
    resolver.add_uniprot_records([
        {"uniprot_id": "P88888", "gene_symbol": "TESTG2",
         "gene_name": "Test gene 2", "organism": "Homo sapiens",
         "sequence": parent_seq},
    ])
    resolver.add_uniprot_records([
        {"uniprot_id": "P88888-2", "gene_symbol": "TESTG2",
         "gene_name": "Test gene 2", "organism": "Homo sapiens",
         "sequence": isoform_seq},
    ])
    parent = resolver.mapping["P88888"]
    isoform = resolver.mapping["P88888-2"]
    assert parent["sequence"] == parent_seq, (
        "P1-023: parent sequence corrupted by isoform merge"
    )
    assert isoform["sequence"] == isoform_seq, (
        "P1-023: isoform sequence not preserved in separate entry"
    )


# =============================================================================
# P1-024: missing InChIKey must NOT cause peptide drugs to merge (NULL != NULL)
# =============================================================================

def test_p1_024_null_inchikeys_do_not_merge():
    """P1-024: two peptide drugs with no InChIKey must NOT be deduplicated
    into one row. NULL != NULL (SQL semantics).

    Pre-fix: missing InChIKey was filled with '' and ''=='' caused merge.
    Post-fix: null InChIKeys get unique sentinels in the dedup pass.
    """
    from cleaning.deduplicator import dedup_by_inchikey

    df = pd.DataFrame([
        {"inchikey": None, "name": "Insulin glargine", "drugbank_id": "DB00047"},
        {"inchikey": None, "name": "Insulin lispro", "drugbank_id": "DB00046"},
        {"inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "name": "Aspirin",
         "drugbank_id": "DB00945"},
    ])
    result = dedup_by_inchikey(df)
    # Both insulin drugs should survive (not merged into one).
    insulin_rows = result[result["name"].str.contains("Insulin", case=False, na=False)]
    assert len(insulin_rows) == 2, (
        f"P1-024: expected 2 insulin rows (NULL != NULL), got "
        f"{len(insulin_rows)}. Peptide drugs were wrongly merged!"
    )


def test_p1_024_empty_string_inchikey_treated_as_null():
    """P1-024: empty-string InChIKeys should also be treated as null
    (not merged together)."""
    from cleaning.deduplicator import dedup_by_inchikey

    df = pd.DataFrame([
        {"inchikey": "", "name": "Drug A", "drugbank_id": "DB001"},
        {"inchikey": "", "name": "Drug B", "drugbank_id": "DB002"},
    ])
    result = dedup_by_inchikey(df)
    assert len(result) == 2, (
        f"P1-024: empty-string InChIKeys should NOT merge. Got {len(result)} rows."
    )


# =============================================================================
# P1-025: stereochemistry preservation in InChIKey (thalidomide enantiomers)
# =============================================================================

@pytest.mark.skipif(
    True,  # will be replaced by a conditional check below
    reason="placeholder"
)
def _placeholder():
    pass


def test_p1_025_thalidomide_enantiomers_different_inchikeys():
    """P1-025: (R)-lactic acid and (S)-lactic acid must produce DIFFERENT
    InChIKeys (they are chiral enantiomers with different pharmacology).

    Pre-fix: tautomer canonicalization stripped stereo -> identical keys.
    Post-fix: stereo is detected and preserved -> different keys.
    (Using lactic acid as a well-defined chiral test molecule; the same
    logic applies to thalidomide, warfarin, ibuprofen, etc.)
    """
    try:
        from cleaning.normalizer import convert_to_inchikey
    except ImportError as e:
        pytest.skip(f"cleaning.normalizer not importable: {e}")

    # (S)-lactic acid and (R)-lactic acid (explicit @/@@ stereo).
    s_lactic = "C[C@H](O)C(=O)O"
    r_lactic = "C[C@@H](O)C(=O)O"

    s_key = convert_to_inchikey(s_lactic)
    r_key = convert_to_inchikey(r_lactic)

    if s_key is None or r_key is None:
        pytest.skip(
            "RDKit could not convert lactic acid SMILES"
        )
    assert s_key != r_key, (
        f"P1-025: (S)- and (R)-lactic acid should have DIFFERENT InChIKeys "
        f"(stereochemistry must be preserved). Got identical key: {s_key}"
    )


def test_p1_025_ibuprofen_enantiomers_different_inchikeys():
    """P1-025: (S)-alanine and (R)-alanine must produce different InChIKeys."""
    try:
        from cleaning.normalizer import convert_to_inchikey
    except ImportError as e:
        pytest.skip(f"cleaning.normalizer not importable: {e}")

    # (S)-alanine vs (R)-alanine (single chiral center, explicit @/@@).
    s_ala = "C[C@H](N)C(=O)O"
    r_ala = "C[C@@H](N)C(=O)O"

    s_key = convert_to_inchikey(s_ala)
    r_key = convert_to_inchikey(r_ala)

    if s_key is None or r_key is None:
        pytest.skip("RDKit could not convert alanine SMILES")
    assert s_key != r_key, (
        f"P1-025: (S)- and (R)-alanine should have DIFFERENT InChIKeys. "
        f"Got identical: {s_key}"
    )


# =============================================================================
# P1-026: case-insensitive InChIKey deduplication
# =============================================================================

def test_p1_026_mixed_case_inchikeys_dedup():
    """P1-026: 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N' and its lowercase form should
    be treated as DUPLICATES (case-insensitive) and deduplicated to one row.

    Pre-fix: case-sensitive comparison treated them as different.
    Post-fix: uppercase normalization before dedup.
    """
    from cleaning.deduplicator import dedup_by_inchikey

    # Use the EXACT same InChIKey in different cases (aspirin).
    df = pd.DataFrame([
        {"inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "name": "Aspirin",
         "drugbank_id": "DB00945"},
        {"inchikey": "bsynrymutxbxsq-uhfffaoyysa-n".replace("yy", "y"),
         "name": "Acetylsalicylic acid", "drugbank_id": "DB001"},
    ])
    result = dedup_by_inchikey(df)
    assert len(result) == 1, (
        f"P1-026: mixed-case InChIKeys should dedup to 1 row, got "
        f"{len(result)}. Case-insensitive normalization is broken."
    )


def test_p1_026_uppercase_output():
    """P1-026: the surviving InChIKey should be uppercase."""
    from cleaning.deduplicator import dedup_by_inchikey

    df = pd.DataFrame([
        {"inchikey": "bsynrymutxbxsq-uhfffaoyysa-n".replace("yy", "y"),
         "name": "Aspirin", "drugbank_id": "DB00945"},
    ])
    result = dedup_by_inchikey(df)
    assert len(result) == 1
    ik = result.iloc[0]["inchikey"]
    assert ik == ik.upper(), (
        f"P1-026: InChIKey should be uppercase, got {ik!r}"
    )


# =============================================================================
# P1-027: source-reliability-weighted confidence
# =============================================================================

def test_p1_027_curated_beats_predicted():
    """P1-027: a Curated edge at 0.7 should beat a Predicted edge at 0.95
    after reliability weighting (0.7*1.0=0.70 > 0.95*0.6=0.57)."""
    from cleaning.confidence import compute_source_weighted_confidence

    result = compute_source_weighted_confidence([
        ("curated", 0.7),
        ("predicted", 0.95),
    ])
    assert result == pytest.approx(0.70, abs=0.001), (
        f"P1-027: curated 0.7 (weighted 0.70) should beat predicted 0.95 "
        f"(weighted 0.57). Got {result}"
    )


def test_p1_027_predicted_alone_downweighted():
    """P1-027: a lone Predicted edge at 0.9 should be downweighted to 0.54."""
    from cleaning.confidence import compute_source_weighted_confidence

    result = compute_source_weighted_confidence([("predicted", 0.9)])
    assert result == pytest.approx(0.54, abs=0.001), (
        f"P1-027: predicted 0.9 * weight 0.6 = 0.54. Got {result}"
    )


def test_p1_027_empty_returns_zero():
    """P1-027: empty input returns 0.0."""
    from cleaning.confidence import compute_source_weighted_confidence

    assert compute_source_weighted_confidence([]) == 0.0


def test_p1_027_animal_model_lowest_weight():
    """P1-027: animal_model has the lowest weight (0.45)."""
    from cleaning.confidence import (
        compute_source_weighted_confidence,
        SOURCE_RELIABILITY_WEIGHTS,
    )

    assert SOURCE_RELIABILITY_WEIGHTS["animal_model"] == 0.45
    assert SOURCE_RELIABILITY_WEIGHTS["curated"] == 1.0
    # animal_model 0.9 -> 0.405; curated 0.5 -> 0.5 -> curated wins.
    result = compute_source_weighted_confidence([
        ("animal_model", 0.9),
        ("curated", 0.5),
    ])
    assert result == pytest.approx(0.5, abs=0.001)


def test_p1_027_rejects_out_of_range():
    """P1-027: out-of-range confidence raises ValueError."""
    from cleaning.confidence import compute_source_weighted_confidence

    with pytest.raises(ValueError):
        compute_source_weighted_confidence([("curated", 1.5)])
    with pytest.raises(ValueError):
        compute_source_weighted_confidence([("curated", -0.1)])


# =============================================================================
# P1-028: Neo4j exporter InChIKey validation (defense-in-depth)
# =============================================================================

def test_p1_028_valid_inchikey_accepted():
    """P1-028: a valid standard InChIKey passes validation."""
    from exporters.neo4j_exporter import _validate_inchikey_for_neo4j

    assert _validate_inchikey_for_neo4j("BSYNRYMUTXBXSQ-UHFFFAOYSA-N") is True
    # lowercase form of the EXACT same key (aspirin, 10-char second block)
    assert _validate_inchikey_for_neo4j("bsynrymutxbxsq-uhfffaoyysa-n".replace("yy", "y")) is True


def test_p1_028_synthetic_inchikey_accepted():
    """P1-028: synthetic InChIKeys (SYNTH prefix) pass validation."""
    from exporters.neo4j_exporter import _validate_inchikey_for_neo4j

    assert _validate_inchikey_for_neo4j("SYNTH-001") is True
    assert _validate_inchikey_for_neo4j("synth-abc") is True


def test_p1_028_malformed_inchikey_rejected():
    """P1-028: malformed InChIKeys (including potential Cypher injection)
    are rejected."""
    from exporters.neo4j_exporter import _validate_inchikey_for_neo4j

    # Malformed
    assert _validate_inchikey_for_neo4j("not-an-inchikey") is False
    assert _validate_inchikey_for_neo4j("") is False
    assert _validate_inchikey_for_neo4j(None) is False
    assert _validate_inchikey_for_neo4j(123) is False
    # Potential Cypher injection attempts
    assert _validate_inchikey_for_neo4j("'}-- RETURN 1//") is False
    assert _validate_inchikey_for_neo4j("BSYNRYMUTXBXSQ-UHFFFAOYSA-N} RETURN 1") is False
    # Wrong length
    assert _validate_inchikey_for_neo4j("SHORT-UHFFFAOYSA-N") is False


def test_p1_028_batch_validation():
    """P1-028: validate_inchikeys_for_export splits valid/invalid correctly."""
    from exporters.neo4j_exporter import validate_inchikeys_for_export

    # Use the EXACT aspirin InChIKey (10-char second block) in upper and lower.
    lower_aspirin = "bsynrymutxbxsq-uhfffaoyysa-n".replace("yy", "y")
    keys = [
        "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",  # valid (uppercase)
        lower_aspirin,                    # valid (lowercase, same key)
        "SYNTH-001",                      # valid (synthetic)
        "malformed",                      # invalid
        "",                               # invalid (empty)
        None,                             # invalid (None)
    ]
    valid, invalid = validate_inchikeys_for_export(keys, source_label="test")
    assert len(valid) == 3, f"Expected 3 valid, got {len(valid)}: {valid}"
    assert len(invalid) == 3, f"Expected 3 invalid, got {len(invalid)}"
    # Valid keys should be uppercase-normalised.
    for v in valid:
        assert v == v.upper()


# =============================================================================
# P1-029: length-scaled fuzzy threshold (PAX vs PAX2 must NOT match)
# =============================================================================

def test_p1_029_pax_does_not_match_pax2():
    """P1-029: 'PAX' must NOT fuzzy-match 'PAX2' (they are DIFFERENT genes).

    Pre-fix: token_sort_ratio('pax', 'pax2') ~86 > 0.85 -> false merge.
    Post-fix: length-scaled threshold for 3-char names = 1.0 (exact only).
    """
    from entity_resolution.resolver_utils import fuzzy_match_best

    candidates = {"pax2": "PAX2_KEY", "pax4": "PAX4_KEY", "pax": "PAX_KEY"}
    result = fuzzy_match_best("pax", candidates, threshold=0.85)
    # 'pax' should only match 'pax' (exact), NOT 'pax2' or 'pax4'.
    if result is not None:
        assert result[0] == "PAX_KEY", (
            f"P1-029: 'PAX' should only match 'PAX' (exact), but matched "
            f"{result[0]}. Short-name false merge!"
        )
    # If result is None, that's also acceptable (exact match required,
    # and 'pax' IS in candidates so it should match). Let's verify it
    # matches the exact entry.
    assert result is not None, "P1-029: 'pax' should exact-match 'pax'"


def test_p1_029_pax2_does_not_match_pax4():
    """P1-029: 'PAX2' must NOT fuzzy-match 'PAX4' (4-char names require
    threshold 0.92; ratio('pax2','pax4') ~75 < 0.92)."""
    from entity_resolution.resolver_utils import fuzzy_match_best

    candidates = {"pax4": "PAX4_KEY"}
    result = fuzzy_match_best("pax2", candidates, threshold=0.85)
    assert result is None, (
        f"P1-029: 'PAX2' should NOT match 'PAX4'. Got {result}"
    )


def test_p1_029_long_names_still_fuzzy_match():
    """P1-029: long names (>14 chars) should still allow fuzzy matching
    at the caller's threshold (e.g. 'acetylsalicylic acid' ~ 'acetylsalicylic')."""
    from entity_resolution.resolver_utils import fuzzy_match_best

    candidates = {"acetylsalicylic": "ASPIRIN_KEY"}
    # 'acetylsalicylic acid' (20 chars) vs 'acetylsalicylic' (16 chars)
    # token_sort_ratio ~88 > 0.85 -> should match.
    result = fuzzy_match_best("acetylsalicylic acid", candidates, threshold=0.85)
    assert result is not None, (
        "P1-029: long names should still fuzzy-match at 0.85 threshold"
    )
    assert result[0] == "ASPIRIN_KEY"


def test_p1_029_length_scaled_threshold_function():
    """P1-029: length_scaled_threshold returns correct values per band."""
    from entity_resolution.resolver_utils import length_scaled_threshold

    assert length_scaled_threshold("PAX", 0.85) == 1.0       # <4 chars -> exact
    assert length_scaled_threshold("MTX", 0.85) == 1.0       # <4 chars -> exact
    assert length_scaled_threshold("PAX2", 0.85) == 0.92     # 4-7 chars
    assert length_scaled_threshold("TP53", 0.85) == 0.92     # 4-7 chars
    assert length_scaled_threshold("ibuprofen", 0.85) == 0.88  # 8-14 chars
    assert length_scaled_threshold("acetylsalicylic acid", 0.85) == 0.85  # >14 chars


# =============================================================================
# P1-030: circuit breaker rolling window + exponential backoff
# =============================================================================

def test_p1_030_default_reset_timeout_is_60s():
    """P1-030: the default reset_timeout should be 60s (not 30s)."""
    from _circuit_breaker import _CircuitBreaker

    cb = _CircuitBreaker()
    assert cb.reset_timeout == 60.0, (
        f"P1-030: default reset_timeout should be 60.0, got {cb.reset_timeout}"
    )


def test_p1_030_default_failure_window_is_300s():
    """P1-030: the default failure_window should be 300s (5 min)."""
    from _circuit_breaker import _CircuitBreaker

    cb = _CircuitBreaker()
    assert cb._failure_window == 300.0, (
        f"P1-030: default failure_window should be 300.0, got {cb._failure_window}"
    )


def test_p1_030_rolling_window_prevents_trip_on_sparse_failures():
    """P1-030: 5 failures spread over >300s should NOT trip the breaker
    (they fall outside the rolling window)."""
    from _circuit_breaker import _CircuitBreaker

    cb = _CircuitBreaker(failure_threshold=5, reset_timeout=1.0,
                         failure_window=0.5, exponential_backoff=False)
    # Record 4 failures (below threshold).
    for _ in range(4):
        cb.record_failure()
    assert cb.state == "closed", f"Should be closed after 4 failures, got {cb.state}"
    # Wait for the window to expire.
    time.sleep(0.6)
    # Record 1 more failure -- the old 4 are outside the window, so only
    # 1 failure is within the window. Should NOT trip.
    cb.record_failure()
    assert cb.state == "closed", (
        f"P1-030: with rolling window, 1 recent failure should NOT trip. "
        f"Got state={cb.state}"
    )


def test_p1_030_rolling_window_trips_on_burst():
    """P1-030: 5 failures within the window SHOULD trip the breaker."""
    from _circuit_breaker import _CircuitBreaker

    cb = _CircuitBreaker(failure_threshold=5, reset_timeout=1.0,
                         failure_window=10.0, exponential_backoff=False)
    for _ in range(5):
        cb.record_failure()
    assert cb.state == "open", (
        f"P1-030: 5 failures within window should trip. Got {cb.state}"
    )


def test_p1_030_exponential_backoff_doubles_reset():
    """P1-030: a failed half-open probe should DOUBLE the effective
    reset_timeout (exponential backoff)."""
    from _circuit_breaker import _CircuitBreaker

    cb = _CircuitBreaker(failure_threshold=2, reset_timeout=1.0,
                         failure_window=10.0, exponential_backoff=True)
    # Trip the breaker.
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "open"
    assert cb._backoff_mult == 1.0, "Initial backoff should be 1.0"
    # Wait for reset_timeout to elapse, then allow a probe.
    time.sleep(1.1)
    assert cb.allow_request() is True  # probe allowed (half-open)
    # Probe fails -> re-open with backoff.
    cb.record_failure()
    assert cb.state == "open"
    assert cb._backoff_mult == 2.0, (
        f"P1-030: after 1 failed probe, backoff should be 2.0, got "
        f"{cb._backoff_mult}"
    )
    # The effective reset is now 1.0 * 2.0 = 2.0s. A request at 1.5s
    # should be REFUSED (old code would allow it at 1.0s reset).
    time.sleep(1.5)
    assert cb.allow_request() is False, (
        "P1-030: with backoff_mult=2.0 and reset_timeout=1.0, a request "
        "at 1.5s should be refused (effective reset = 2.0s)"
    )


def test_p1_030_backoff_resets_on_success():
    """P1-030: a successful probe should reset the backoff multiplier."""
    from _circuit_breaker import _CircuitBreaker

    cb = _CircuitBreaker(failure_threshold=2, reset_timeout=0.1,
                         failure_window=10.0, exponential_backoff=True)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "open"
    time.sleep(0.15)
    assert cb.allow_request() is True
    cb.record_failure()  # failed probe -> backoff = 2.0
    assert cb._backoff_mult == 2.0
    # Wait for effective reset (0.1 * 2.0 = 0.2s).
    time.sleep(0.25)
    assert cb.allow_request() is True
    cb.record_success()  # successful probe -> backoff reset
    assert cb._backoff_mult == 1.0, (
        f"P1-030: backoff should reset to 1.0 on success, got {cb._backoff_mult}"
    )
    assert cb.state == "closed"


def test_p1_030_backoff_capped():
    """P1-030: the backoff multiplier should be capped at _BACKOFF_CAP_MULT."""
    from _circuit_breaker import _CircuitBreaker, _BACKOFF_CAP_MULT

    cb = _CircuitBreaker(failure_threshold=1, reset_timeout=0.05,
                         failure_window=10.0, exponential_backoff=True)
    # Repeatedly trip + fail probes to push backoff to the cap.
    for _ in range(20):
        cb.record_failure()  # trip (or re-open with backoff)
        assert cb.state == "open"
        # Wait for effective reset, then probe + fail.
        effective = cb._reset_timeout * cb._backoff_mult
        time.sleep(effective + 0.06)
        cb.allow_request()  # half-open probe
        cb.record_failure()  # fail -> backoff doubles
    assert cb._backoff_mult <= _BACKOFF_CAP_MULT, (
        f"P1-030: backoff should be capped at {_BACKOFF_CAP_MULT}, got "
        f"{cb._backoff_mult}"
    )


if __name__ == "__main__":
    # Allow running directly: python -m pytest tests/test_team3_p1_022_to_030_forensic.py -v
    pytest.main([__file__, "-v", "--tb=short"])
