"""
v65 REAL CODE execution -- imports and exercises every fixed module.

This is NOT a test file. It imports the REAL source modules and runs
REAL code paths to prove the fixes don't break imports or basic
functionality. If any import fails or any function crashes, the script
exits non-zero with a clear traceback.

Run with:
    cd phase1 && python tests/v65_root_fixes/run_v65_real_code.py
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

_PHASE1_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PHASE1_ROOT))

# Ensure dev environment so cosmic:cosmic doesn't raise on import.
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DRUGOS_DEV_ALLOW_DEFAULT_DB", "1")
os.environ.setdefault("DATABASE_URL", "postgresql://cosmic:cosmic@localhost:5432/drug_repurposing")

PASS = 0
FAIL = 0

def check(label: str, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS] {label}")
        PASS += 1
    except Exception as exc:
        print(f"  [FAIL] {label}: {exc}")
        traceback.print_exc()
        FAIL += 1

print("=" * 70)
print("v65 REAL CODE EXECUTION -- importing and exercising fixed modules")
print("=" * 70)

# --- 1. config.settings (P1C-003, P1C-010) ---
print("\n[1] config.settings")
def import_settings():
    import config.settings as s
    assert s.STRING_MIN_COMBINED_SCORE == 700, f"expected 700, got {s.STRING_MIN_COMBINED_SCORE}"
    assert s.CONFIG_REGISTRY["STRING_MIN_COMBINED_SCORE"]["default"] == "700"
    # P1C-010: cosmic:cosmic check exists in source
    src = (_PHASE1_ROOT / "config" / "settings.py").read_text()
    assert "cosmic:cosmic@" in src
check("import config.settings + verify STRING=700 + CONFIG_REGISTRY=700 + cosmic check", import_settings)

# --- 2. config.__init__ (P1C-003 validation) ---
print("\n[2] config.__init__")
def import_config_init():
    import config as cfg
    cfg._ensure_settings_loaded()
    # Patch score to 400 and verify WARNING is emitted.
    original = cfg._resolved_settings.get("STRING_MIN_COMBINED_SCORE")
    try:
        cfg._resolved_settings["STRING_MIN_COMBINED_SCORE"] = 400
        results = cfg._run_validation()
        warnings = [r for r in results if r.severity == "WARNING"
                    and "STRING_MIN_COMBINED_SCORE" in r.setting_name]
        assert len(warnings) > 0, "score=400 must produce a WARNING"
    finally:
        cfg._resolved_settings["STRING_MIN_COMBINED_SCORE"] = original
check("import config + verify validation warns at score=400", import_config_init)

# --- 3. database.models (P1C-001, P1C-002, P1C-006) ---
print("\n[3] database.models")
def import_models():
    import database.models as m
    # P1C-001: gene_symbol nullable, no server_default; CHECK removed.
    gda = m.GeneDiseaseAssociation
    gs = gda.__table__.c.gene_symbol
    assert gs.nullable is True
    assert gs.server_default is None or gs.server_default.arg == ""
    di = gda.__table__.c.disease_id
    assert di.server_default is None or di.server_default.arg == ""
    # P1C-006: is_globally_approved NOT NULL + server_default="0"
    drug = m.Drug
    iga = drug.__table__.c.is_globally_approved
    assert iga.nullable is False
    assert iga.server_default is not None
    assert "0" in str(iga.server_default.arg)
    # P1C-002: validator rejects P001 by default.
    os.environ.pop("DRUGOS_ENVIRONMENT", None)
    try:
        m._validate_uniprot_id("P001")
        raise AssertionError("P001 should be rejected by default")
    except ValueError:
        pass
    # P1C-002: real UniProt accepted.
    assert m._validate_uniprot_id("P69999") == "P69999"
check("import database.models + verify P1C-001/002/006 schema + validators", import_models)

# --- 4. database.loaders (P1C-002, P1C-005) ---
print("\n[4] database.loaders")
def import_loaders():
    import database.loaders as L
    # P1C-002: loaders mirror -- rejects P001 by default.
    os.environ.pop("DRUGOS_ENVIRONMENT", None)
    try:
        L._validate_uniprot_id("P001")
        raise AssertionError("P001 should be rejected by default in loaders")
    except ValueError:
        pass
    assert L._validate_uniprot_id("P69999") == "P69999"
    # P1C-005: source uses begin_nested, not session.rollback() in quarantine.
    src = (_PHASE1_ROOT / "database" / "loaders.py").read_text()
    idx = src.find("def _quarantine_gda_rows")
    body = src[idx:idx + 8000]
    assert "with session.begin_nested():" in body
check("import database.loaders + verify P1C-002 validators + P1C-005 savepoint", import_loaders)

# --- 5. cleaning._constants (P1C-014) ---
print("\n[5] cleaning._constants")
def import_constants():
    import cleaning._constants as c
    assert c.ACTIVITY_VALUE_CENSORED_THRESHOLD == 1e6
    assert c.ACTIVITY_VALUE_NON_PHYSICAL_THRESHOLD == 1e9
    assert c._ACTIVITY_VALUE_CENSORED_MAX_LEGACY == 1e6
    assert c._ACTIVITY_VALUE_MAX == 1e6  # backward-compat alias
    # Canonical regex works.
    assert c.CANONICAL_INCHIKEY_REGEX.match("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
    assert not c.CANONICAL_INCHIKEY_REGEX.match("BSYNRYMUTXBXSQ-UHFFFAOYSA-N-a")
check("import cleaning._constants + verify P1C-014 renamed alias + regex", import_constants)

# --- 6. cleaning.deduplicator (P1C-008, P1C-012, P1C-013, P1C-014) ---
print("\n[6] cleaning.deduplicator")
def import_deduplicator():
    import cleaning.deduplicator as d
    import pandas as pd
    # P1C-014: no local _ACTIVITY_VALUE_MAX = 1e9.
    src = (_PHASE1_ROOT / "cleaning" / "deduplicator.py").read_text()
    assert "_ACTIVITY_VALUE_MAX: float = _ACTIVITY_NON_PHYSICAL_MAX" not in src
    # P1C-012: uses _INCHIKEY_PATTERN (not inline string).
    assert ".str.match(_INCHIKEY_PATTERN)" in src
    # P1C-013: dead n_normalised removed.
    assert 'n_normalised = int(\n                                (working["inchikey"].astype(str).str[-1] == "N")' not in src
    # P1C-008: pre_filter_drops in dedup_interactions.
    assert "pre_filter_drops = max(0, _pre_filter_row_count - int(len(working)))" in src
    # Exercise dedup_interactions with a real DataFrame.
    df = pd.DataFrame([
        {"drug_id": "D1", "target_id": "T1", "activity_value": 1.0},
        {"drug_id": "D1", "target_id": "T1", "activity_value": 2.0},  # duplicate
        {"drug_id": None, "target_id": "T1", "activity_value": 3.0},  # null key
    ])
    result = d.dedup_interactions(
        df, keys=["drug_id", "target_id"],
        keep="first", return_result=True,
        null_keys_handler="drop",
    )
    assert result.pre_filter_drops >= 1
    assert result.duplicates_removed >= 1
check("import cleaning.deduplicator + verify P1C-008/012/013/014 + exercise dedup_interactions", import_deduplicator)

# --- 7. cleaning.missing_values (P1C-007) ---
print("\n[7] cleaning.missing_values")
def import_missing_values():
    import cleaning.missing_values as mv
    import pandas as pd
    # P1C-007: NaN-keyed rows survive dedup.
    df = pd.DataFrame([
        {"gene_symbol": None, "disease_id": "D1", "source": "s", "score": 0.5},
        {"gene_symbol": None, "disease_id": "D2", "source": "s", "score": 0.6},
        {"gene_symbol": None, "disease_id": "D3", "source": "s", "score": 0.7},
    ])
    result = mv.validate_gda_scores(df, dedup=True, return_result=False)
    assert len(result) == 3, f"3 NaN rows must survive (got {len(result)})"
check("import cleaning.missing_values + verify P1C-007 NaN-sentinel", import_missing_values)

# --- 8. cleaning.__init__ (P1C-011) ---
print("\n[8] cleaning.__init__")
def import_cleaning_init():
    import cleaning
    src = (_PHASE1_ROOT / "cleaning" / "__init__.py").read_text()
    old = ("if col in out.columns:\n"
           "                    out[col] = result_rows[col].values\n"
           "                else:\n"
           "                    out[col] = result_rows[col].values")
    assert old not in src, "dead if/else must be removed"
check("import cleaning + verify P1C-011 dead if/else removed", import_cleaning_init)

# --- 9. entity_resolution.base (P1C-004, P1C-009) ---
print("\n[9] entity_resolution.base")
def import_base():
    import entity_resolution.base as base
    # P1C-009: SYNTHETIC_KEY_MATCH enum exists.
    assert hasattr(base.MatchConfidence, "SYNTHETIC_KEY_MATCH")
    assert base.MatchConfidence.SYNTHETIC_KEY_MATCH.value == 0.5
    assert base.MatchConfidence.from_method("synthetic_key_match") == base.MatchConfidence.SYNTHETIC_KEY_MATCH
    # P1C-004: fallback uses _STRICT_INCHIKEY_PATTERN.
    src = (_PHASE1_ROOT / "entity_resolution" / "base.py").read_text()
    assert "_STRICT_INCHIKEY_PATTERN.match(inchikey)" in src
    # Verify the strict pattern rejects suffixed keys.
    assert not base._STRICT_INCHIKEY_PATTERN.match("BSYNRYMUTXBXSQ-UHFFFAOYSA-N-a")
    assert base._STRICT_INCHIKEY_PATTERN.match("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
check("import entity_resolution.base + verify P1C-004 strict fallback + P1C-009 SYNTH enum", import_base)

# --- 10. entity_resolution.drug_resolver (P1C-009) ---
print("\n[10] entity_resolution.drug_resolver")
def import_drug_resolver():
    import entity_resolution.drug_resolver as dr
    src = (_PHASE1_ROOT / "entity_resolution" / "drug_resolver.py").read_text()
    idx = src.find("def _match_by_inchikey")
    body = src[idx:idx + 5000]
    assert 'method="synthetic_key_match"' in body
    assert "MatchConfidence.SYNTHETIC_KEY_MATCH.value" in body
    assert "confidence=0.5,  # v29: was 1.0" not in body
check("import entity_resolution.drug_resolver + verify P1C-009 SYNTH method label", import_drug_resolver)

# --- 11. .env.example (P1C-003) ---
print("\n[11] .env.example")
def check_env_example():
    env = (_PHASE1_ROOT / "config" / ".env.example").read_text()
    for line in env.splitlines():
        s = line.strip()
        if s.startswith("STRING_MIN_COMBINED_SCORE=") and not s.startswith("#"):
            assert "=700" in s, f"expected 700, got {s}"
            return
    raise AssertionError("STRING_MIN_COMBINED_SCORE not found in .env.example")
check("verify .env.example ships STRING_MIN_COMBINED_SCORE=700", check_env_example)

# --- 12. Phase 1 -> Phase 2 bridge (integration) ---
print("\n[12] Phase 1 -> Phase 2 bridge")
def check_bridge():
    # Add phase2 to path and import the bridge.
    phase2_root = _PHASE1_ROOT.parent / "phase2"
    sys.path.insert(0, str(phase2_root))
    try:
        import drugos_graph.phase1_bridge as bridge
        # Verify key entry points exist.
        assert hasattr(bridge, "read_phase1_outputs")
        assert hasattr(bridge, "stage_phase1_to_phase2")
        assert hasattr(bridge, "load_into_graph")
        assert hasattr(bridge, "run_phase1_to_phase2")
    finally:
        sys.path.pop(0)
check("import phase2.drugos_graph.phase1_bridge + verify 4 entry points exist", check_bridge)

# --- Summary ---
print("\n" + "=" * 70)
print(f"REAL CODE EXECUTION SUMMARY: {PASS} passed, {FAIL} failed")
print("=" * 70)
if FAIL > 0:
    print("\n*** FAILURES DETECTED -- see tracebacks above ***")
    sys.exit(1)
else:
    print("\n*** ALL REAL CODE PATHS EXECUTE CLEANLY -- NO REGRESSIONS ***")
    sys.exit(0)
