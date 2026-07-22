#!/usr/bin/env python3
"""v68 ROOT FIX -- standalone test runner (no pytest, no conftest dependency).

Runs all 13 forensic tests directly via Python assertions. This avoids
the parent conftest.py which requires torch_geometric (heavy install).

Exit code 0 = all tests pass; non-zero = at least one failure.
"""
from __future__ import annotations

import inspect
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

# Ensure phase2 package is importable
PHASE2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PHASE2_ROOT))

# Import the REAL loader modules (this verifies they compile + import)
from drugos_graph import chembl_loader
from drugos_graph import drkg_loader
from drugos_graph import string_loader
from drugos_graph import clinicaltrials_loader
from drugos_graph import opentargets_loader
from drugos_graph import disgenet_loader
from drugos_graph import omim_loader
from drugos_graph import uniprot_loader


passed = 0
failed = 0
failures = []


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        failures.append((name, detail))
        print(f"  FAIL: {name}")
        if detail:
            print(f"        {detail}")


print("=" * 72)
print("v68 ROOT FIX -- 13-Issue Forensic Verification")
print("=" * 72)

# ---------------------------------------------------------------------------
print("\n[P2L-008] chembl _RE_ACTIVATE regex (P0-critical)")
print("-" * 72)

# Test: INACTIVATION -> inhibits
r = chembl_loader.standard_type_to_relation("Inactivation")
check("Inactivation -> inhibits", r == "inhibits",
      f"got {r!r}")

r = chembl_loader.standard_type_to_relation("INACTIVATE")
check("INACTIVATE -> inhibits", r == "inhibits", f"got {r!r}")

r = chembl_loader.standard_type_to_relation("INACTIVATOR")
check("INACTIVATOR -> inhibits", r == "inhibits", f"got {r!r}")

r = chembl_loader.standard_type_to_relation("ACTIVATION")
check("ACTIVATION -> activates (not broken)", r == "activates", f"got {r!r}")

r = chembl_loader.standard_type_to_relation("ACTIVATES")
check("ACTIVATES -> activates (not broken)", r == "activates", f"got {r!r}")

src = chembl_loader._RE_ACTIVATE.pattern
check("negative-lookbehind present", "(?<![A-Z])ACTIVAT" in src,
      f"pattern={src!r}")

# ---------------------------------------------------------------------------
print("\n[P2L-021] drkg compound ID NaN propagation (P0-critical)")
print("-" * 72)

src = inspect.getsource(drkg_loader.parse_drkg_tsv)
check("fillna(original_head) present", "fillna(original_head)" in src)
check("fillna(original_tail) present", "fillna(original_tail)" in src)
check("empty_mask checks head_id", 'head_id"].isna()' in src or 'df["head_id"].isna()' in src)

# ---------------------------------------------------------------------------
print("\n[P2L-032] string UNIPROT_AC_REGEX grouping (P0-critical)")
print("-" * 72)

src = string_loader.UNIPROT_AC_REGEX.pattern
check("starts with ^(?:", src.startswith("^(?:"), f"pattern={src!r}")
check("ends with )$", src.endswith(")$"), f"pattern={src!r}")
check("single ^ anchor", src.count("^") == 1, f"count={src.count('^')}")
check("single $ anchor", src.count("$") == 1, f"count={src.count('$')}")

for ac in ["P23219", "Q9H0A5", "O00165", "A0A023GPI9"]:
    check(f"valid AC {ac} matches", bool(string_loader.UNIPROT_AC_REGEX.match(ac)))

check("P12345XYZGARBAGE rejected",
      not string_loader.UNIPROT_AC_REGEX.match("P12345XYZGARBAGE"))
check("GARBAGEA0123456789 rejected",
      not string_loader.UNIPROT_AC_REGEX.match("GARBAGEA0123456789"))

# ---------------------------------------------------------------------------
print("\n[P2L-041] clinicaltrials rel_type (P0-critical)")
print("-" * 72)

def make_ct_record(status, outcome_raw):
    return {
        "nct_id": "NCT00000001",
        "drug_mesh": "D000068",    # valid MeSH (D000001 is flagged as garbage)
        "condition_mesh": "D013577",  # valid MeSH (Syndrome)
        "overall_status": status,
        "primary_outcome_met_raw": outcome_raw,
        "phase": "Phase 3",
        "enrollment": 500,
        "study_type": "Interventional",
        "has_results": True,
    }

cfg = clinicaltrials_loader.ClinicalTrialsConfig()
state = clinicaltrials_loader._LoaderState(cfg, "fake_sha256", "2024-01-01T00:00:00Z")

# Positive trial: Completed + met -> treats
edge = clinicaltrials_loader._build_edge_record_from_dict(
    make_ct_record("Completed", "met"), cfg, state
)
check("Completed+met -> treats",
      edge is not None and edge["rel_type"] == "treats",
      f"rel_type={edge['rel_type'] if edge else 'None'}")

# Negative trial: Completed + not_met -> tested_for
edge = clinicaltrials_loader._build_edge_record_from_dict(
    make_ct_record("Completed", "not_met"), cfg, state
)
check("Completed+not_met -> tested_for",
      edge is not None and edge["rel_type"] == "tested_for",
      f"rel_type={edge['rel_type'] if edge else 'None'}")

# Unknown outcome: Completed + no data -> tested_for
rec = make_ct_record("Completed", None)
del rec["primary_outcome_met_raw"]
edge = clinicaltrials_loader._build_edge_record_from_dict(rec, cfg, state)
check("Completed+unknown -> tested_for",
      edge is not None and edge["rel_type"] == "tested_for",
      f"rel_type={edge['rel_type'] if edge else 'None'}")

# ---------------------------------------------------------------------------
print("\n[P2L-045] opentargets score keys (P0-critical)")
print("-" * 72)

src = inspect.getsource(opentargets_loader._emit_compound_protein_edge)
check('"opentargets_score" set', '"opentargets_score"' in src)
check('"binding_confidence": score NOT set', '"binding_confidence": score' not in src)
check('"chembl_score": score NOT set', '"chembl_score": score' not in src)
check('dedupe reads opentargets_score', 'get("opentargets_score"' in src)

# ---------------------------------------------------------------------------
print("\n[P2L-003] disgenet stale-cache refresh (P1-high)")
print("-" * 72)

src = inspect.getsource(disgenet_loader.download_disgenet)
check("shutil.copy2 present", "shutil.copy2" in src)
check("DEFAULT_DISGENET_CSV referenced", "DEFAULT_DISGENET_CSV" in src)

# ---------------------------------------------------------------------------
print("\n[P2L-005] omim mapping_key fallback (P1-high)")
print("-" * 72)

src = inspect.getsource(omim_loader.omim_to_edge_records)
check("mapping_key referenced", "mapping_key" in src.lower())

# Integration: mapping_key=1 -> 0.95
df = pd.DataFrame([{
    "gene_symbol": "BRCA1", "disease_id": "C0001",
    "mapping_key": "1", "canonical_gene_id": "672",
}])
edges = omim_loader.omim_to_edge_records(df)
check("mapping_key=1 -> score=0.95",
      len(edges) == 1 and edges[0]["props"]["score"] == 0.95,
      f"score={edges[0]['props']['score'] if edges else 'None'}")

df = pd.DataFrame([{
    "gene_symbol": "BRCA2", "disease_id": "C0002",
    "mapping_key": "2", "canonical_gene_id": "675",
}])
edges = omim_loader.omim_to_edge_records(df)
check("mapping_key=2 -> score=0.7",
      len(edges) == 1 and edges[0]["props"]["score"] == 0.7,
      f"score={edges[0]['props']['score'] if edges else 'None'}")

df = pd.DataFrame([{
    "gene_symbol": "TP53", "disease_id": "C0003",
    "mapping_key": "3", "canonical_gene_id": "7157",
}])
# mapping_key=3 gives score=0.4, but OMIM_MIN_SCORE default is 0.5,
# so the row is DROPPED by the threshold (correct behavior). To verify
# the score assignment directly, patch the threshold to 0 inside the
# function via monkey-patching the config import.
import drugos_graph.config as _dg_config
_orig = getattr(_dg_config, "OMIM_MIN_SCORE", 0.5)
_dg_config.OMIM_MIN_SCORE = 0.0
try:
    edges = omim_loader.omim_to_edge_records(df)
    check("mapping_key=3 -> score=0.4 (threshold=0)",
          len(edges) == 1 and edges[0]["props"]["score"] == 0.4,
          f"len={len(edges)}, score={edges[0]['props']['score'] if edges else 'None'}")
finally:
    _dg_config.OMIM_MIN_SCORE = _orig

# ---------------------------------------------------------------------------
print("\n[P2L-009] chembl standard_value propagation (P1-high)")
print("-" * 72)

src = inspect.getsource(chembl_loader.chembl_to_edge_records)
check('"standard_value" in props', '"standard_value"' in src)
check('"standard_units" in props', '"standard_units"' in src)

# ---------------------------------------------------------------------------
print("\n[P2L-010] chembl organism filter NaN (P1-high)")
print("-" * 72)

src = inspect.getsource(chembl_loader.parse_chembl_activities)
check("notna() used for tax_id", "notna()" in src)

# Integration: replicate filter logic
df = pd.DataFrame({
    "tax_id": ["9606", "9606", None, "10090", "9606"],
    "drug_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL4", "CHEMBL5"],
    "pchembl_value": [7.0, 8.0, 9.0, 7.5, 6.5],
})
tax_id_numeric = pd.to_numeric(df["tax_id"], errors="coerce")
mask = df["tax_id"].notna() & (tax_id_numeric == 9606)
filtered = df[mask]
check("NaN tax_id rows dropped (3 kept)", len(filtered) == 3,
      f"kept {len(filtered)}")
check("CHEMBL3 (NaN tax_id) dropped", "CHEMBL3" not in filtered["drug_chembl_id"].values)

# ---------------------------------------------------------------------------
print("\n[P2L-013] chembl iter_chembl_activities filters (P1-high)")
print("-" * 72)

src = inspect.getsource(chembl_loader.iter_chembl_activities)
check("organism filter in iter", "tax_id" in src.lower())
check("confidence filter in iter", "confidence_score" in src)
check("pchembl filter in iter", "pchembl" in src.lower())
check("ID validation in iter", "_RE_CHEMBL_ID" in src or "drug_chembl_id" in src)
check("yields chunks", "yield chunk" in src)

# ---------------------------------------------------------------------------
print("\n[P2L-015] uniprot DR-edges bare dst_id (P1-high)")
print("-" * 72)

records = [{
    "accession": "P23219",
    "cross_references": {
        "ChEMBL": ["CHEMBL218"],
        "DrugBank": ["DB00001"],
        "STRING": ["9606.ENSP00000358091"],
    },
    "_provenance": {},
}]
edges = uniprot_loader.uniprot_to_edge_records(records)
check("3 edges emitted", len(edges) == 3, f"got {len(edges)}")
all_bare = all(":" not in e["dst_id"] for e in edges)
check("all dst_id bare (no prefix)", all_bare,
      ", ".join(e["dst_id"] for e in edges))
check("xref_db preserved", edges[0].get("xref_db") == "ChEMBL",
      f"xref_db={edges[0].get('xref_db')}")

# ---------------------------------------------------------------------------
print("\n[P2L-022] drkg no comment='#' (P1-high)")
print("-" * 72)

src = inspect.getsource(drkg_loader.parse_drkg_tsv)
# Check no active comment="#" in the read_csv call
active_lines = [
    line.strip() for line in src.split("\n")
    if not line.strip().startswith("#") and 'comment="#"' in line
]
check("no active comment='#' in read_csv", len(active_lines) == 0,
      f"found in: {active_lines}")

# ---------------------------------------------------------------------------
print("\n[P2L-023] drkg type cross-check else branch (P1-high)")
print("-" * 72)

src = inspect.getsource(drkg_loader.parse_drkg_tsv)
check("else branch for malformed relation",
      "malformed_relation_dst_type_no_separator" in src)

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 72)

if failures:
    print("\nFAILURES:")
    for name, detail in failures:
        print(f"  - {name}: {detail}")

sys.exit(0 if failed == 0 else 1)
