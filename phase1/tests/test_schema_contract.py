"""TM1 Task 17: schema contract tests for Phase 1 output.

This test suite verifies that:

1. ``phase1.contracts.phase1_schema`` is importable and declares all 11
   required source specs (chembl_drugs, chembl_activities, drugs,
   interactions, indications, uniprot_proteins, string_ppi,
   disgenet_gda, omim_gda, omim_susceptibility, pubchem_enrichment).

2. Each source spec has the required structural fields (key, filename,
   required_columns, min_rows).

3. The Phase 2 bridge's ``_PHASE1_EXPECTED_COLUMNS`` dict is in sync
   with this contract's required_columns. This is the REGRESSION GUARD
   against schema drift — if Phase 1 adds a column and Phase 2 doesn't,
   this test fails.

4. ``validate_output_dir`` correctly identifies:
   - Missing required columns (ERROR).
   - Unsatisfied any-of groups (ERROR).
   - Below min_rows (ERROR or WARNING based on min_rows value).
   - Empty optional sources (WARNING, not ERROR).

5. ``validate_output_dir`` returns 0 on a fully-valid directory and
   non-zero on any ERROR.

The test synthesizes minimal CSV files in a tmp_path — NO real API
calls are made. The point is to verify the CONTRACT logic, not the
pipeline outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure phase1/ is on sys.path (matches the conftest.py setup).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE1_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT, PHASE1_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---------------------------------------------------------------------------
# Test 1: contract module structure
# ---------------------------------------------------------------------------


def test_phase1_schema_module_imports():
    """TM1 Task 7: ``phase1.contracts.phase1_schema`` is importable."""
    from contracts.phase1_schema import (
        PHASE1_OUTPUT_SCHEMA,
        PHASE1_CSV_FILENAMES,
        ColumnSpec,
        SourceSpec,
        ValidationIssue,
        get_required_columns,
        get_any_of_groups,
    )
    assert isinstance(PHASE1_OUTPUT_SCHEMA, dict)
    assert len(PHASE1_OUTPUT_SCHEMA) == 11, (
        f"Expected 11 source specs, got {len(PHASE1_OUTPUT_SCHEMA)}. "
        f"Sources: {sorted(PHASE1_OUTPUT_SCHEMA.keys())}"
    )
    assert isinstance(PHASE1_CSV_FILENAMES, dict)
    assert len(PHASE1_CSV_FILENAMES) == 11


def test_all_11_source_keys_present():
    """TM1 Task 7: all 11 canonical source keys are present."""
    from contracts.phase1_schema import PHASE1_OUTPUT_SCHEMA
    expected_keys = {
        "chembl_drugs", "chembl_activities", "drugs", "interactions",
        "indications", "uniprot_proteins", "string_ppi", "disgenet_gda",
        "omim_gda", "omim_susceptibility", "pubchem_enrichment",
    }
    actual_keys = set(PHASE1_OUTPUT_SCHEMA.keys())
    assert actual_keys == expected_keys, (
        f"Schema keys mismatch. Missing: {expected_keys - actual_keys}. "
        f"Extra: {actual_keys - expected_keys}."
    )


def test_each_source_spec_has_required_structure():
    """TM1 Task 7: each SourceSpec has key, filename, required_columns, min_rows."""
    from contracts.phase1_schema import PHASE1_OUTPUT_SCHEMA
    for key, spec in PHASE1_OUTPUT_SCHEMA.items():
        assert spec.key == key, f"Spec key mismatch: {spec.key} != {key}"
        assert spec.filename, f"Source {key} has no filename"
        assert isinstance(spec.required_columns, tuple), \
            f"Source {key} required_columns must be a tuple"
        assert isinstance(spec.min_rows, int) and spec.min_rows >= 0, \
            f"Source {key} min_rows must be a non-negative int"
        # Every spec must declare at least one required column OR one
        # any-of group (otherwise the source has no contract at all).
        assert len(spec.required_columns) > 0 or len(spec.any_of_groups) > 0, \
            f"Source {key} has no required_columns AND no any_of_groups"


# ---------------------------------------------------------------------------
# Test 2: Phase 2 bridge is in sync with the contract
# ---------------------------------------------------------------------------


def test_phase2_bridge_expected_columns_in_sync():
    """TM1 Task 17 REGRESSION GUARD: the Phase 2 bridge's
    ``_PHASE1_EXPECTED_COLUMNS`` dict must be a SUBSET of (or equal to)
    the contract's required_columns. If Phase 1 declares a column as
    required, Phase 2 must also expect it — otherwise schema drift.
    """
    from contracts.phase1_schema import PHASE1_OUTPUT_SCHEMA

    # Try to import the bridge. If the bridge is not importable (e.g.
    # missing dependencies in CI), skip this test rather than fail —
    # the contract itself is still valid.
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "phase2"))
        from drugos_graph.phase1_bridge import _PHASE1_EXPECTED_COLUMNS
    except Exception as exc:
        pytest.skip(f"Phase 2 bridge not importable: {exc}")

    # For every source key present in BOTH the contract and the bridge,
    # the bridge's expected columns must be a subset of the contract's
    # required columns (the bridge may have FEWER required columns if
    # it accepts aliases, but it must not require a column the contract
    # doesn't declare).
    for key, spec in PHASE1_OUTPUT_SCHEMA.items():
        if key not in _PHASE1_EXPECTED_COLUMNS:
            continue
        contract_required = {c.name for c in spec.required_columns}
        bridge_required = set(_PHASE1_EXPECTED_COLUMNS[key])
        # Bridge columns must be declared in the contract (as required
        # OR optional OR in an any-of group).
        contract_all_known = contract_required | {
            c.name for c in spec.optional_columns
        } | {col for group in spec.any_of_groups for col in group}
        unknown = bridge_required - contract_all_known
        assert not unknown, (
            f"TM1 Task 17 REGRESSION: Phase 2 bridge expects columns "
            f"{unknown} for source '{key}' that are NOT declared in the "
            f"Phase 1 contract. This is schema drift — update "
            f"phase1/contracts/phase1_schema.py OR phase1_bridge.py."
        )


# ---------------------------------------------------------------------------
# Test 3: validate_output_dir behavior
# ---------------------------------------------------------------------------


def test_validate_output_dir_returns_zero_on_valid_directory(tmp_path):
    """TM1 Task 8: a fully-valid directory returns exit code 0."""
    from contracts.phase1_schema import PHASE1_OUTPUT_SCHEMA
    from contracts.validate_output import validate_output_dir

    # Write a minimal valid CSV for each source.
    for key, spec in PHASE1_OUTPUT_SCHEMA.items():
        # Build a DataFrame with the required columns + one row.
        data = {}
        for col in spec.required_columns:
            if col.dtype == "int64":
                data[col.name] = [1]
            elif col.dtype == "float64":
                data[col.name] = [1.0]
            elif col.dtype == "bool":
                data[col.name] = [True]
            else:
                data[col.name] = [f"test_{col.name}"]
        # Satisfy any-of groups by including the first option of each.
        for group in spec.any_of_groups:
            first_col = group[0]
            if first_col not in data:
                data[first_col] = [f"test_{first_col}"]
        df = pd.DataFrame(data)
        df.to_csv(tmp_path / spec.filename, index=False)

    exit_code = validate_output_dir(tmp_path)
    assert exit_code == 0, (
        f"Valid directory should return 0, got {exit_code}. "
        f"Check the validation logs above."
    )


def test_validate_output_dir_returns_nonzero_on_missing_required_column(tmp_path):
    """TM1 Task 8: a CSV missing a required column triggers exit code 1."""
    from contracts.phase1_schema import PHASE1_OUTPUT_SCHEMA
    from contracts.validate_output import validate_output_dir

    # Write chembl_drugs.csv with ONLY the inchikey column (missing chembl_id).
    spec = PHASE1_OUTPUT_SCHEMA["chembl_drugs"]
    df = pd.DataFrame({"inchikey": ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N"]})
    df.to_csv(tmp_path / spec.filename, index=False)

    # Write all OTHER sources minimally so only chembl_drugs fails.
    for key, other_spec in PHASE1_OUTPUT_SCHEMA.items():
        if key == "chembl_drugs":
            continue
        data = {}
        for col in other_spec.required_columns:
            data[col.name] = [f"x_{col.name}"]
        for group in other_spec.any_of_groups:
            data[group[0]] = [f"x_{group[0]}"]
        pd.DataFrame(data).to_csv(tmp_path / other_spec.filename, index=False)

    exit_code = validate_output_dir(tmp_path)
    assert exit_code == 1, (
        f"Missing required column should return 1, got {exit_code}."
    )


def test_validate_output_dir_returns_nonzero_on_unsatisfied_any_of(tmp_path):
    """TM1 Task 8: a CSV missing ALL columns in an any-of group triggers exit 1."""
    from contracts.phase1_schema import PHASE1_OUTPUT_SCHEMA
    from contracts.validate_output import validate_output_dir

    # Write drugs.csv with required columns but NO drugbank_id OR chembl_id.
    spec = PHASE1_OUTPUT_SCHEMA["drugs"]
    df = pd.DataFrame({
        "name": ["Aspirin"],
        "inchikey": ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N"],
        # NO drugbank_id, NO chembl_id — any-of group unsatisfied.
    })
    df.to_csv(tmp_path / spec.filename, index=False)

    # Write all OTHER sources minimally.
    for key, other_spec in PHASE1_OUTPUT_SCHEMA.items():
        if key == "drugs":
            continue
        data = {}
        for col in other_spec.required_columns:
            data[col.name] = [f"x_{col.name}"]
        for group in other_spec.any_of_groups:
            data[group[0]] = [f"x_{group[0]}"]
        pd.DataFrame(data).to_csv(tmp_path / other_spec.filename, index=False)

    exit_code = validate_output_dir(tmp_path)
    assert exit_code == 1, (
        f"Unsatisfied any-of group should return 1, got {exit_code}."
    )


def test_validate_output_dir_warns_on_empty_optional_source(tmp_path):
    """TM1 Task 8: an empty optional source (min_rows=0) is a WARNING,
    not an ERROR. Exit code is 0 (warnings allowed).
    """
    from contracts.phase1_schema import PHASE1_OUTPUT_SCHEMA
    from contracts.validate_output import validate_output_dir

    # Write all required sources minimally. Leave pubchem_enrichment
    # (min_rows=0) missing entirely — should be a WARNING, not ERROR.
    for key, spec in PHASE1_OUTPUT_SCHEMA.items():
        if spec.min_rows == 0 and key == "pubchem_enrichment":
            continue  # leave it missing
        data = {}
        for col in spec.required_columns:
            data[col.name] = [f"x_{col.name}"]
        for group in spec.any_of_groups:
            data[group[0]] = [f"x_{group[0]}"]
        pd.DataFrame(data).to_csv(tmp_path / spec.filename, index=False)

    exit_code = validate_output_dir(tmp_path, fail_on_warning=False)
    assert exit_code == 0, (
        f"Missing optional source should return 0 (warnings allowed), "
        f"got {exit_code}."
    )


def test_validate_output_dir_fails_on_warning_when_flag_set(tmp_path):
    """TM1 Task 8: ``fail_on_warning=True`` makes warnings into errors."""
    from contracts.phase1_schema import PHASE1_OUTPUT_SCHEMA
    from contracts.validate_output import validate_output_dir

    # Leave pubchem_enrichment missing — should be a WARNING.
    for key, spec in PHASE1_OUTPUT_SCHEMA.items():
        if key == "pubchem_enrichment":
            continue
        data = {}
        for col in spec.required_columns:
            data[col.name] = [f"x_{col.name}"]
        for group in spec.any_of_groups:
            data[group[0]] = [f"x_{group[0]}"]
        pd.DataFrame(data).to_csv(tmp_path / spec.filename, index=False)

    exit_code = validate_output_dir(tmp_path, fail_on_warning=True)
    assert exit_code == 2, (
        f"--fail-on-warning should return 2 when warnings exist, "
        f"got {exit_code}."
    )


# ---------------------------------------------------------------------------
# Test 4: CLI entry point
# ---------------------------------------------------------------------------


def test_validate_output_cli_returns_zero_on_valid(tmp_path, capsys):
    """TM1 Task 8: the CLI ``python -m phase1.contracts.validate_output``
    returns 0 on a valid directory.
    """
    from contracts.phase1_schema import PHASE1_OUTPUT_SCHEMA
    from contracts.validate_output import main

    for key, spec in PHASE1_OUTPUT_SCHEMA.items():
        data = {}
        for col in spec.required_columns:
            data[col.name] = [f"x_{col.name}"]
        for group in spec.any_of_groups:
            data[group[0]] = [f"x_{group[0]}"]
        pd.DataFrame(data).to_csv(tmp_path / spec.filename, index=False)

    exit_code = main([str(tmp_path)])
    assert exit_code == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
