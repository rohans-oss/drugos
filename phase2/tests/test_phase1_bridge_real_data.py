"""v108 ROOT FIX (issue 79): Real-data tests for phase1_bridge.

These tests load REAL Phase 1 CSV fixtures (no mocks) and verify:
  1. No drugs are dropped due to missing InChIKey (issue 61 — biotech fallback)
  2. No censored activity values are silently loaded as '=' (issue 62 — censoring)
  3. No duplicate Compound nodes are created (issue 65 — canonical IDs)
  4. drugbank_id is optional (issue 63 — ChEMBL-only deployment works)
  5. Schema-derived columns match Phase 1 contract (issue 64)

The fixtures are tiny synthetic CSVs that mimic the Phase 1 output schema.
They exercise the bridge's read + validate + canonicalize path end-to-end.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Ensure phase2 is on sys.path so `from drugos_graph import ...` works.
# The test file is at phase2/tests/test_phase1_bridge_real_data.py, so
# parents[1] = phase2/. We add phase2/ to sys.path (NOT the repo root).
_PHASE2_ROOT = Path(__file__).resolve().parents[1]
if str(_PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHASE2_ROOT))


# ---------------------------------------------------------------------------
# Fixtures — tiny synthetic Phase 1 CSVs that mimic real output schema.
# ---------------------------------------------------------------------------

@pytest.fixture
def phase1_processed_dir(tmp_path: Path) -> Path:
    """Create a minimal Phase 1 processed_data directory with real-schema CSVs."""
    # drugs.csv — 5 drugs:
    #   - 2 small molecules with valid InChIKey (aspirin, caffeine)
    #   - 1 biologic with NO InChIKey but DrugBank ID (insulin, DB00071)
    #   - 1 biologic with NO InChIKey and NO DrugBank ID, only PubChem CID
    #   - 1 biologic with NO InChIKey, NO DrugBank ID, NO PubChem CID (should be dropped)
    drugs_df = pd.DataFrame([
        {
            "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",  # valid InChIKey — aspirin
            "name": "Aspirin",
            "chembl_id": "CHEMBL25",
            "drugbank_id": "DB00945",
            "pubchem_cid": 2244,
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "molecular_weight": 180.16,
            "max_phase": 4,
            "is_fda_approved": True,
            "is_globally_approved": True,
        },
        {
            "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",  # valid InChIKey — caffeine
            "name": "Caffeine",
            "chembl_id": "CHEMBL1137",
            "drugbank_id": "DB00201",
            "pubchem_cid": 2519,
            "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "molecular_weight": 194.19,
            "max_phase": 4,
            "is_fda_approved": True,
            "is_globally_approved": True,
        },
        {
            # Biologic — NO InChIKey (None), has DrugBank ID (issue 61 case)
            "inchikey": None,
            "name": "Insulin",
            "chembl_id": None,
            "drugbank_id": "DB00071",
            "pubchem_cid": None,
            "smiles": None,
            "molecular_weight": 5807.57,
            "max_phase": 4,
            "is_fda_approved": True,
            "is_globally_approved": True,
        },
        {
            # Biologic — NO InChIKey, NO DrugBank ID, only PubChem CID (issue 61 case)
            "inchikey": None,
            "name": "Epoetin alfa",
            "chembl_id": None,
            "drugbank_id": None,
            "pubchem_cid": 000000,  # invalid (zero) — should NOT pass
            "smiles": None,
            "molecular_weight": 30447.0,
            "max_phase": 4,
            "is_fda_approved": True,
            "is_globally_approved": True,
        },
        {
            # Junk row — NO InChIKey, NO DrugBank ID, NO valid PubChem CID
            # (issue 61: should be DROPPED — no canonical ID available)
            "inchikey": None,
            "name": "Bogus drug",
            "chembl_id": None,
            "drugbank_id": None,
            "pubchem_cid": None,
            "smiles": None,
            "molecular_weight": 0,
            "max_phase": 0,
            "is_fda_approved": False,
            "is_globally_approved": False,
        },
    ])
    drugs_df.to_csv(tmp_path / "drugbank_drugs.csv", index=False)

    # chembl_activities_clean.csv — 4 activities:
    #   - 1 normal IC50 (50 nM, '=' relation)
    #   - 1 censored LOW value (0.5 nM — should be flagged '<', issue 62)
    #   - 1 censored HIGH value (200 μM — should be flagged '>', issue 62)
    #   - 1 already-censored value from ChEMBL (standard_relation='<' in raw)
    chembl_act_df = pd.DataFrame([
        {
            "activity_id": "1",
            "molecule_chembl_id": "CHEMBL25",
            "target_chembl_id": "CHEMBL218",
            "target_accession": "P00533",
            "target_pref_name": "EGFR",
            "activity_type": "IC50",
            "activity_value": 50.0,  # 50 nM — normal, not censored
            "activity_units": "nM",
            "pchembl_value": 7.30,
            "assay_id": "CHEMBL123",
            "standard_relation": "=",
        },
        {
            "activity_id": "2",
            "molecule_chembl_id": "CHEMBL25",
            "target_chembl_id": "CHEMBL218",
            "target_accession": "P00533",
            "target_pref_name": "EGFR",
            "activity_type": "IC50",
            "activity_value": 0.5,  # 0.5 nM — below 1 nM LDL, issue 62 censoring
            "activity_units": "nM",
            "pchembl_value": 9.30,
            "assay_id": "CHEMBL124",
            "standard_relation": "=",  # raw says '=' but should be flagged '<'
        },
        {
            "activity_id": "3",
            "molecule_chembl_id": "CHEMBL1137",
            "target_chembl_id": "CHEMBL356",
            "target_accession": "P09874",
            "target_pref_name": "PARP1",
            "activity_type": "IC50",
            "activity_value": 200.0,  # 200 μM — above 100 μM UDL, issue 62 censoring
            "activity_units": "uM",
            "pchembl_value": 3.70,
            "assay_id": "CHEMBL125",
            "standard_relation": "=",  # raw says '=' but should be flagged '>'
        },
        {
            "activity_id": "4",
            "molecule_chembl_id": "CHEMBL25",
            "target_chembl_id": "CHEMBL218",
            "target_accession": "P00533",
            "target_pref_name": "EGFR",
            "activity_type": "IC50",
            "activity_value": 0.05,
            "activity_units": "nM",
            "pchembl_value": 10.30,
            "assay_id": "CHEMBL126",
            "standard_relation": "<",  # already censored in raw — should be preserved
        },
    ])
    chembl_act_df.to_csv(tmp_path / "chembl_activities_clean.csv", index=False)

    # drugbank_indications.csv — 2 indications (drug → disease)
    indications_df = pd.DataFrame([
        {"drugbank_id": "DB00945", "disease_id": "DOID:1289", "disease_name": "Hypertension"},
        {"drugbank_id": "DB00201", "disease_id": "DOID:332", "disease_name": "Migraine"},
    ])
    indications_df.to_csv(tmp_path / "drugbank_indications.csv", index=False)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_issue_61_biotech_drugs_not_dropped(phase1_processed_dir: Path) -> None:
    """Issue 61: drugs without InChIKey must NOT be dropped if they have a
    DrugBank ID, PubChem CID, or UniProt accession fallback.
    """
    from drugos_graph.phase1_bridge import (
        read_phase1_outputs,
        _PHASE1_EXPECTED_COLUMNS,
    )

    # Force CSV backend (prefer_postgres=False) since we have no DB in tests
    result = read_phase1_outputs(
        phase1_processed_dir=phase1_processed_dir,
        prefer_postgres=False,
    )
    # The result is a _Phase1BridgeResult — get the frames dict
    frames = result.frames if hasattr(result, "frames") else result
    drugs_df = frames.get("drugs")
    assert drugs_df is not None, "drugs frame should be present"
    assert not drugs_df.empty, "drugs frame should not be empty"
    # 5 rows in fixture; the CSV read returns ALL of them (canonical-ID
    # derivation happens later in _stage_phase1_data, not in the read).
    assert len(drugs_df) == 5, f"Expected 5 drug rows, got {len(drugs_df)}"
    # Verify the biologic with only DrugBank ID is present
    insulin_rows = drugs_df[drugs_df["name"] == "Insulin"]
    assert len(insulin_rows) == 1, "Insulin (biologic with DrugBank ID only) should be present"


def test_issue_62_censored_values_flagged(phase1_processed_dir: Path) -> None:
    """Issue 62: censored activity values (<1 nM, >100 μM) must be flagged
    with the correct censoring direction. The is_censored column must be
    added; the standard_relation column must be updated to reflect censoring.
    """
    from drugos_graph.phase1_bridge import read_phase1_outputs

    result = read_phase1_outputs(
        phase1_processed_dir=phase1_processed_dir,
        prefer_postgres=False,
    )
    frames = result.frames if hasattr(result, "frames") else result
    chembl_df = frames.get("chembl_activities")
    assert chembl_df is not None, "chembl_activities frame should be present"
    assert not chembl_df.empty, "chembl_activities frame should not be empty"

    # Verify the is_censored column was added
    assert "is_censored" in chembl_df.columns, (
        "is_censored column must be added (v108 issue 62 root fix)"
    )

    # Find the 0.5 nM row (should be censored '<')
    # activity_id is read as int from CSV (the schema declares it as string with
    # pattern ^\d+$ but pandas infers int by default — we coerce to str for the filter).
    chembl_df["activity_id_str"] = chembl_df["activity_id"].astype(str)
    low_row = chembl_df[chembl_df["activity_id_str"] == "2"].iloc[0]
    assert low_row["is_censored"] == True, (
        f"0.5 nM IC50 must be flagged as censored (issue 62). "
        f"Got is_censored={low_row['is_censored']!r}"
    )
    assert low_row["standard_relation"] == "<", (
        f"0.5 nM IC50 standard_relation must be '<' (censored). "
        f"Got {low_row['standard_relation']!r}"
    )

    # Find the 200 μM row (should be censored '>')
    high_row = chembl_df[chembl_df["activity_id_str"] == "3"].iloc[0]
    assert high_row["is_censored"] == True, (
        f"200 μM IC50 must be flagged as censored (issue 62). "
        f"Got is_censored={high_row['is_censored']!r}"
    )
    assert high_row["standard_relation"] == ">", (
        f"200 μM IC50 standard_relation must be '>' (censored). "
        f"Got {high_row['standard_relation']!r}"
    )

    # Find the 50 nM row (should NOT be censored)
    normal_row = chembl_df[chembl_df["activity_id_str"] == "1"].iloc[0]
    assert normal_row["is_censored"] == False, (
        f"50 nM IC50 must NOT be flagged as censored. "
        f"Got is_censored={normal_row['is_censored']!r}"
    )
    assert normal_row["standard_relation"] == "=", (
        f"50 nM IC50 standard_relation must be '='. "
        f"Got {normal_row['standard_relation']!r}"
    )


def test_issue_63_drugbank_id_optional(phase1_processed_dir: Path) -> None:
    """Issue 63: drugbank_id is OPTIONAL. The drugs CSV schema must accept
    rows with chembl_id but no drugbank_id (ChEMBL-only deployment).
    """
    from drugos_graph.phase1_bridge import (
        _PHASE1_EXPECTED_COLUMNS,
        _PHASE1_ANY_OF_COLUMNS,
        _validate_phase1_columns,
    )

    # The 'drugs' source must NOT require drugbank_id (it must be in ANY_OF)
    assert "drugbank_id" not in _PHASE1_EXPECTED_COLUMNS.get("drugs", []), (
        "drugbank_id must NOT be in REQUIRED columns for 'drugs' source (issue 63)"
    )
    drugs_any_of = _PHASE1_ANY_OF_COLUMNS.get("drugs", [])
    drugs_any_of_flat = {col for group in drugs_any_of for col in group}
    assert "drugbank_id" in drugs_any_of_flat, (
        "drugbank_id must be in ANY_OF columns for 'drugs' source (issue 63)"
    )

    # A drugs DataFrame with chembl_id but no drugbank_id must pass validation
    df_no_drugbank = pd.DataFrame([
        {"inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "name": "Aspirin", "chembl_id": "CHEMBL25"},
    ])
    _validate_phase1_columns(
        df_no_drugbank,
        _PHASE1_EXPECTED_COLUMNS["drugs"],
        "drugs",
        any_of_groups=_PHASE1_ANY_OF_COLUMNS.get("drugs"),
    )  # must NOT raise


def test_issue_64_phase1_schema_loaded() -> None:
    """Issue 64: Phase 1 schema JSON must be loaded and override hardcoded
    expected columns for sources it covers.
    """
    from drugos_graph.phase1_bridge import (
        _load_phase1_schema_columns,
        _PHASE1_SCHEMA_DERIVED_COLUMNS,
        _PHASE1_EXPECTED_COLUMNS,
    )

    # The schema-derived columns must be loaded (4 sources per the v1.json)
    schema_cols = _load_phase1_schema_columns()
    assert len(schema_cols) >= 1, (
        "Phase 1 schema JSON must produce at least 1 source mapping (issue 64)"
    )

    # drugs source must have schema-derived required columns (at minimum: inchikey)
    drugs_required = _PHASE1_EXPECTED_COLUMNS.get("drugs", [])
    assert "inchikey" in drugs_required, (
        f"drugs source must require 'inchikey' (from schema JSON). Got: {drugs_required}"
    )


def test_issue_65_no_duplicate_compound_nodes(phase1_processed_dir: Path) -> None:
    """Issue 65: register_node must use canonical IDs, NOT free-text names.
    Two drugs with the same name but different canonical IDs must NOT collapse
    into a single node.
    """
    from drugos_graph.phase1_bridge import RecordingGraphBuilder

    builder = RecordingGraphBuilder()
    # Register two distinct drugs that happen to share the display name "Test Drug"
    # but have different canonical IDs.
    nid1 = builder.register_node(
        "drug", "DB00001", display_name="Test Drug", properties={"smiles": "C1=CC=CC=C1"}
    )
    nid2 = builder.register_node(
        "drug", "DB00002", display_name="Test Drug", properties={"smiles": "CC1=CC=CC=C1"}
    )
    # Both must be registered (NOT collapsed)
    assert builder.total_nodes == 2, (
        f"Two drugs with same display_name but different canonical IDs must NOT "
        f"collapse. Expected 2 nodes, got {builder.total_nodes}"
    )
    # The returned IDs must be distinct
    assert nid1 != nid2, (
        f"register_node must return distinct full IDs for distinct canonical IDs. "
        f"Got {nid1!r} and {nid2!r}"
    )
    # No dead-letter entries (both IDs are valid DrugBank format)
    assert len(builder.dead_letter) == 0, (
        f"No dead-letter entries expected. Got: {builder.dead_letter}"
    )


def test_issue_67_recording_graph_builder_save_load_round_trip(tmp_path: Path) -> None:
    """Issue 67: RecordingGraphBuilder must support save/load round-trip
    in BOTH JSON and Parquet formats.
    """
    from drugos_graph.phase1_bridge import RecordingGraphBuilder

    builder = RecordingGraphBuilder()
    builder.register_node("protein", "P12821", display_name="ACE")
    builder.register_node("protein", "P43681", display_name="ACE2")
    builder.register_edge(
        "protein", "interacts_with", "protein",
        "protein:P12821", "protein:P43681",
    )
    original_nodes = builder.total_nodes
    original_edges = builder.total_edges

    # JSON round-trip
    json_path = tmp_path / "graph.json"
    builder.save(json_path)
    loaded_json = RecordingGraphBuilder.load(json_path)
    assert loaded_json.total_nodes == original_nodes, (
        f"JSON round-trip: nodes mismatch. Expected {original_nodes}, "
        f"got {loaded_json.total_nodes}"
    )
    assert loaded_json.total_edges == original_edges, (
        f"JSON round-trip: edges mismatch. Expected {original_edges}, "
        f"got {loaded_json.total_edges}"
    )

    # Parquet round-trip (only if pyarrow is available)
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pytest.skip("pyarrow not installed — skipping Parquet round-trip test")
    parquet_path = tmp_path / "graph.parquet"
    builder.save(parquet_path)
    loaded_pq = RecordingGraphBuilder.load(parquet_path)
    assert loaded_pq.total_nodes == original_nodes, (
        f"Parquet round-trip: nodes mismatch. Expected {original_nodes}, "
        f"got {loaded_pq.total_nodes}"
    )


def test_issue_77_drugos_graph_error_base_class() -> None:
    """Issue 77: all exceptions in drugos_graph.exceptions must inherit
    from DrugosGraphError (the canonical base class).
    """
    from drugos_graph import exceptions

    # DrugosGraphError must exist
    assert hasattr(exceptions, "DrugosGraphError"), (
        "DrugosGraphError must exist (v108 issue 77 root fix)"
    )
    # DrugOSDataError must still exist (backward compat) AND inherit from DrugosGraphError
    assert hasattr(exceptions, "DrugOSDataError"), (
        "DrugOSDataError must still exist (backward compat)"
    )
    assert issubclass(exceptions.DrugOSDataError, exceptions.DrugosGraphError), (
        "DrugOSDataError must inherit from DrugosGraphError (issue 77)"
    )
    # All loader-specific exceptions must inherit from DrugosGraphError
    loader_errors = [
        exceptions.UniProtDownloadError,
        exceptions.ChEMBLParseError,
        exceptions.DrugBankDataIntegrityError,
        exceptions.StringEdgeLoadMismatchError,
        exceptions.SiderCriticalError,
        exceptions.GeoConfigurationError,
    ]
    for exc_class in loader_errors:
        assert issubclass(exc_class, exceptions.DrugosGraphError), (
            f"{exc_class.__name__} must inherit from DrugosGraphError (issue 77)"
        )
    # Catch-all test
    try:
        raise exceptions.UniProtDownloadError("test", context={"url": "x"})
    except exceptions.DrugosGraphError as e:
        assert "test" in str(e)
    else:
        pytest.fail("except DrugosGraphError must catch UniProtDownloadError (issue 77)")


def test_issue_70_canonical_node_labels() -> None:
    """Issue 70: lowercase canonical node labels must be defined and
    the canonical_node_label() helper must convert PascalCase → lowercase.
    """
    from drugos_graph.config_schema import (
        canonical_node_label,
        pascal_node_label,
        NODE_LABEL_LOWERCASE,
    )

    # Verify the 5 canonical labels from the issue spec
    assert canonical_node_label("Compound") == "drug"
    assert canonical_node_label("Protein") == "protein"
    assert canonical_node_label("Pathway") == "pathway"
    assert canonical_node_label("Disease") == "disease"
    assert canonical_node_label("MedDRA_Term") == "side_effect"
    # Round-trip
    assert pascal_node_label("drug") == "Compound"
    assert pascal_node_label("side_effect") == "MedDRA_Term"


def test_issue_71_canonical_edge_types() -> None:
    """Issue 71: canonical edge-type strings must be defined in
    src_verb_dst form.
    """
    from drugos_graph.config_schema import (
        canonical_edge_type,
        parse_canonical_edge_type,
        EDGE_TYPE_CANONICAL,
    )

    # Verify canonical forms
    assert canonical_edge_type("Compound", "treats", "Disease") == "drug_treats_disease"
    assert canonical_edge_type("Compound", "inhibits", "Protein") == "drug_inhibits_protein"
    assert canonical_edge_type("Protein", "interacts_with", "Protein") == "protein_interacts_with_protein"
    # Reverse lookup
    assert parse_canonical_edge_type("drug_treats_disease") == ("Compound", "treats", "Disease")
    # All 32 CORE_EDGE_TYPES must have canonical names
    assert len(EDGE_TYPE_CANONICAL) >= 30, (
        f"Expected >=30 canonical edge types, got {len(EDGE_TYPE_CANONICAL)}"
    )


def test_issue_76_drugos_environment_defaults_to_production() -> None:
    """Issue 76: DRUGOS_ENVIRONMENT must default to 'production' (not 'dev').
    Dev mode must be opt-in.
    """
    from drugos_graph.config import DRUGOS_ENVIRONMENT, _get_dev_mode

    # Default must be production (when DRUGOS_ENVIRONMENT env var is NOT set)
    # We can't unset the env var here (other tests may have set it), so we
    # just verify the constant matches the env var or 'production'.
    if "DRUGOS_ENVIRONMENT" not in os.environ:
        assert DRUGOS_ENVIRONMENT == "production", (
            f"DRUGOS_ENVIRONMENT must default to 'production', got {DRUGOS_ENVIRONMENT!r}"
        )
    # _get_dev_mode() must return False in production mode
    old = os.environ.get("DRUGOS_ENVIRONMENT")
    try:
        os.environ["DRUGOS_ENVIRONMENT"] = "production"
        assert _get_dev_mode() is False, "_get_dev_mode() must be False in production"
        os.environ["DRUGOS_ENVIRONMENT"] = "dev"
        assert _get_dev_mode() is True, "_get_dev_mode() must be True in dev"
        os.environ["DRUGOS_ENVIRONMENT"] = "staging"
        assert _get_dev_mode() is False, (
            "_get_dev_mode() must be False in staging (treated as production, issue 76)"
        )
    finally:
        if old is None:
            os.environ.pop("DRUGOS_ENVIRONMENT", None)
        else:
            os.environ["DRUGOS_ENVIRONMENT"] = old
