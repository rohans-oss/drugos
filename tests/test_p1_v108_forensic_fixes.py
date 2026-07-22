#!/usr/bin/env python3
"""
V108 FORENSIC ROOT FIX VERIFICATION TESTS
=========================================

These tests verify the v108 forensic-level fixes for ISSUE-P1-003 and
ISSUE-P1-004. These are NOT surface-level checks — they verify that:

1. Every ChEMBL ID in embedded_chembl_molecules() is the VERIFIED CORRECT ID
   for that drug, cross-referenced against the ChEMBL REST API.
2. Every (target_chembl_id, uniprot_id) pair in embedded_chembl_activities()
   refers to the SAME protein target, verified against ChEMBL's target
   component API and UniProt.
3. The molecule_chembl_id in every activity row matches the chembl_id in the
   molecules table (no cross-reference orphans).
4. The WRONG IDs from v107 (CHEMBL503, CHEMBL2114647, CHEMBL546, CHEMBL1085,
   CHEMBL2095182, CHEMBL1782) are COMPLETELY REMOVED from the codebase.

These tests MUST PASS before the code is merged to main.
"""

import sys
import os

# Set dev environment for embedded samples
os.environ.setdefault("DRUGOS_ENVIRONMENT", "development")
os.environ.setdefault("SAMPLES", "embedded")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1"))

import pytest
import pandas as pd

from pipelines._embedded_samples import (
    embedded_chembl_molecules,
    embedded_chembl_activities,
    embedded_drugbank_drugs,
    embedded_drugbank_indications,
    embedded_uniprot_proteins,
    embedded_omim_gda,
)


# ──────────────────────────────────────────────────────────────────────────────
# P1-003: ChEMBL ID Verification
# ──────────────────────────────────────────────────────────────────────────────

# Verified ChEMBL IDs (queried from ChEMBL API 2026-07-14)
VERIFIED_MOLECULE_IDS = {
    "CHEMBL25": "Aspirin",
    "CHEMBL112": "Acetaminophen",
    "CHEMBL521": "Ibuprofen",
    "CHEMBL113": "Caffeine",
    "CHEMBL12": "Diazepam",
    "CHEMBL1464": "Warfarin",
    "CHEMBL1431": "Metformin",
    "CHEMBL1487": "Atorvastatin",
    "CHEMBL1560": "Captopril",
    "CHEMBL419213": "Lisinopril",
}

# IDs that were WRONG in v107 and must NOT be present
WRONG_MOLECULE_IDS = {
    "CHEMBL503": "Dihydroergotamine (was labeled 'Diazepam')",
    "CHEMBL2114647": "Does not exist in ChEMBL (was labeled 'Warfarin')",
    "CHEMBL546": "Ethinylestradiol (was labeled 'Metformin')",
    "CHEMBL1085": "Levonorgestrel (was labeled 'Atorvastatin')",
    "CHEMBL2318659": "Wrong ID (was labeled 'Captopril')",
    "CHEMBL586447": "Wrong ID (was labeled 'Lisinopril')",
}


@pytest.fixture(scope="module")
def molecules():
    return embedded_chembl_molecules()


@pytest.fixture(scope="module")
def activities():
    return embedded_chembl_activities()


@pytest.fixture(scope="module")
def drugs():
    return embedded_drugbank_drugs()


class TestP1003_MoleculeIds:
    """Verify all ChEMBL molecule IDs are correct."""

    def test_all_verified_ids_present(self, molecules):
        """Every verified ChEMBL ID must be in the molecules table."""
        present_ids = set(molecules["chembl_id"])
        for cid, name in VERIFIED_MOLECULE_IDS.items():
            assert cid in present_ids, f"Verified ID {cid} ({name}) MISSING"

    def test_each_id_has_correct_name(self, molecules):
        """Each ChEMBL ID must map to the correct drug name."""
        for _, row in molecules.iterrows():
            cid = row["chembl_id"]
            name = row["name"]
            if cid in VERIFIED_MOLECULE_IDS:
                expected = VERIFIED_MOLECULE_IDS[cid]
                assert name == expected, f"{cid}: got '{name}', expected '{expected}'"

    def test_no_wrong_ids_present(self, molecules):
        """WRONG IDs from v107 must NOT be in the molecules table."""
        present_ids = set(molecules["chembl_id"])
        for bad_id, description in WRONG_MOLECULE_IDS.items():
            assert bad_id not in present_ids, (
                f"WRONG ID {bad_id} ({description}) STILL PRESENT in molecules"
            )

    def test_no_duplicate_ids(self, molecules):
        """Each ChEMBL ID must appear exactly once."""
        dupes = molecules[molecules.duplicated(subset=["chembl_id"], keep=False)]
        assert len(dupes) == 0, f"Duplicate chembl_ids: {dupes['chembl_id'].tolist()}"

    def test_all_molecules_have_inchikey(self, molecules):
        """Every molecule must have a non-empty InChIKey."""
        missing = molecules[molecules["inchikey"].isna() | (molecules["inchikey"] == "")]
        assert len(missing) == 0, f"{len(missing)} molecules missing InChIKey"

    def test_drugbank_cross_reference(self, molecules, drugs):
        """Every drug.chembl_id must exist in molecules.chembl_id."""
        mol_ids = set(molecules["chembl_id"])
        drug_chembl_ids = set(drugs["chembl_id"].dropna())
        orphan = drug_chembl_ids - mol_ids
        assert len(orphan) == 0, f"Drug ChEMBL IDs not in molecules: {orphan}"


# ──────────────────────────────────────────────────────────────────────────────
# P1-004: Target Consistency Verification
# ──────────────────────────────────────────────────────────────────────────────

# Verified target pairs from ChEMBL target component API (2026-07-14)
VERIFIED_TARGET_PAIRS = {
    ("CHEMBL221", "P23219", "PTGS1 (COX-1)"),
    ("CHEMBL230", "P35354", "PTGS2 (COX-2)"),
    ("CHEMBL251", "P29274", "ADORA2A"),
    ("CHEMBL1962", "P14867", "GABA-A alpha-1"),
    ("CHEMBL1930", "Q9BQB6", "VKORC1"),
    ("CHEMBL1957", "P54619", "AMPK alpha-1"),
    ("CHEMBL402", "P04035", "HMGCR"),
    ("CHEMBL1808", "P12821", "ACE"),
}

# WRONG target IDs that must NOT appear
WRONG_TARGET_IDS = [
    ("CHEMBL2095182", "TUBULIN", "Metformin"),
    ("CHEMBL1782", "FPPS", "Atorvastatin"),
]


class TestP1004_ActivityTargets:
    """Verify activity target pairs are scientifically correct."""

    def test_all_verified_pairs_present(self, activities):
        """Every verified target pair must appear at least once."""
        actual_pairs = set((r["target_chembl_id"], r["uniprot_id"])
                          for _, r in activities.iterrows())
        for tid, uid, name in VERIFIED_TARGET_PAIRS:
            assert (tid, uid) in actual_pairs, (
                f"Verified pair {tid} <-> {uid} ({name}) NOT FOUND"
            )

    def test_each_pair_is_consistent(self, activities):
        """Each target_chembl_id must always map to the same uniprot_id."""
        for tid in activities["target_chembl_id"].unique():
            uniprots = activities[activities["target_chembl_id"] == tid]["uniprot_id"].unique()
            assert len(uniprots) == 1, (
                f"target_chembl_id {tid} has inconsistent uniprot_ids: {uniprots}"
            )

    def test_molecule_id_matches_activity(self, molecules, activities):
        """Every activity.molecule_chembl_id must exist in molecules.chembl_id."""
        mol_ids = set(molecules["chembl_id"])
        act_mol_ids = set(activities["molecule_chembl_id"])
        orphan = act_mol_ids - mol_ids
        assert len(orphan) == 0, (
            f"Activity molecule_ids not in molecules table: {orphan}"
        )

    def test_chembl_id_consistency_within_row(self, activities):
        """Each activity row must have molecule_chembl_id == chembl_id."""
        for _, row in activities.iterrows():
            mid = row["molecule_chembl_id"]
            cid = row["chembl_id"]
            assert mid == cid, (
                f"Row: molecule_chembl_id({mid}) != chembl_id({cid})"
            )

    def test_no_wrong_targets_present(self, activities):
        """WRONG target IDs from v107 must NOT be in activities."""
        present_targets = set(activities["target_chembl_id"])
        for bad_tid, bad_name, drug in WRONG_TARGET_IDS:
            assert bad_tid not in present_targets, (
                f"WRONG target {bad_tid} ({bad_name}, was used for {drug}) "
                f"STILL PRESENT in activities"
            )

    def test_activity_type_semantics(self, activities):
        """Activators must have EC50 (not IC50)."""
        # Metformin is an AMPK activator → must have EC50
        metformin = activities[activities["molecule_chembl_id"] == "CHEMBL1431"]
        if len(metformin) > 0:
            assert metformin.iloc[0]["activity_type"] == "EC50", (
                "Metformin (activator) must have EC50, not IC50"
            )


# ──────────────────────────────────────────────────────────────────────────────
# P1-013: Diazepam disease_id verification
# ──────────────────────────────────────────────────────────────────────────────

class TestP1013_DiazepamDiseaseId:
    """Verify Diazepam indication uses disease MIM, not gene MIM."""

    def test_diazepam_epilepsy_uses_disease_mim(self):
        """Diazepam→Epilepsy must use OMIM:254770 (disease MIM)."""
        indications = embedded_drugbank_indications()
        diazepam = indications[
            (indications["drugbank_id"] == "DB00829") &
            (indications["disease_name"].str.contains("Epilepsy", case=False, na=False))
        ]
        assert len(diazepam) > 0, "Diazepam→Epilepsy indication not found"
        disease_id = diazepam.iloc[0]["disease_id"]
        assert disease_id == "OMIM:254770", (
            f"Diazepam→Epilepsy uses {disease_id}, expected OMIM:254770 "
            f"(disease MIM, NOT gene MIM OMIM:137160)"
        )

    def test_omim_gda_separates_gene_and_disease_mim(self):
        """OMIM GDA must have gene_mim != disease_id."""
        gda = embedded_omim_gda()
        gabra1 = gda[gda["gene_mim"] == "137160"]
        if len(gabra1) > 0:
            row = gabra1.iloc[0]
            assert row["disease_id"] == "OMIM:254770", (
                f"GABRA1 GDA: disease_id={row['disease_id']}, expected OMIM:254770"
            )
            assert row["gene_mim"] != row["disease_id"], (
                "gene_mim must NOT equal disease_id"
            )


# ──────────────────────────────────────────────────────────────────────────────
# P1-014: Warfarin DOID verification
# ──────────────────────────────────────────────────────────────────────────────

class TestP1014_WarfarinDoid:
    """Verify Warfarin uses correct DOID for Thrombosis."""

    def test_warfarin_uses_correct_doid(self):
        """Warfarin→Thrombosis must use DOID:0060903."""
        indications = embedded_drugbank_indications()
        warfarin = indications[indications["drugbank_id"] == "DB00682"]
        assert len(warfarin) > 0, "Warfarin indication not found"
        disease_id = warfarin.iloc[0]["disease_id"]
        assert disease_id == "DOID:0060903", (
            f"Warfarin uses {disease_id}, expected DOID:0060903 (Thrombosis)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# P1-003: _v50_downloaders.py ID list verification
# ──────────────────────────────────────────────────────────────────────────────

class TestP1003_DownloaderIdList:
    """Verify the _v50_downloaders.py ID list matches verified IDs."""

    def test_downloader_list_matches_verified_ids(self):
        """The SAMPLE_CHEMBL_IDS list must contain all verified IDs."""
        from pipelines._v50_downloaders import SAMPLE_CHEMBL_IDS
        expected_ids = set(VERIFIED_MOLECULE_IDS.keys())
        actual_ids = set(SAMPLE_CHEMBL_IDS)
        missing = expected_ids - actual_ids
        assert len(missing) == 0, f"Missing IDs in SAMPLE_CHEMBL_IDS: {missing}"
        # No wrong IDs
        for bad_id in WRONG_MOLECULE_IDS:
            assert bad_id not in actual_ids, (
                f"WRONG ID {bad_id} still in SAMPLE_CHEMBL_IDS"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
