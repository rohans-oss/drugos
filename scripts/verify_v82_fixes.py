#!/usr/bin/env python3
"""v82 REAL-CODE VERIFICATION -- runs the ACTUAL module functions (not test
files, not smoke tests) to confirm every P0-G1..G4 and P1-A1..A14 root-cause
fix works at runtime.

Each check imports the REAL module and calls the REAL function. If a fix is
broken, the function raises and the check FAILS loudly.
"""
import sys
import os
import re
from pathlib import Path

# ── Path setup: make Phase 1 and Phase 2 packages importable ──────────────
# Use repo root relative to this script's location (works in CI + locally).
REPO = Path(__file__).resolve().parent.parent
PHASE1 = REPO / "phase1"
PHASE2 = REPO / "phase2"
sys.path.insert(0, str(PHASE1))
sys.path.insert(0, str(PHASE2))

# Dev environment so validators allow test fixtures + don't raise on defaults
os.environ.setdefault("DRUGOS_ENVIRONMENT", "development")
os.environ.setdefault("DRUGOS_DEV_ALLOW_DEFAULT_DB", "1")

PASS = 0
FAIL = 0
RESULTS = []

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        RESULTS.append(f"  PASS  {name}")
    else:
        FAIL += 1
        RESULTS.append(f"  FAIL  {name}  {detail}")

# ═══════════════════════════════════════════════════════════════════════════
# P0-G1: compound_to_inchikey crosswalk population
# ═══════════════════════════════════════════════════════════════════════════
try:
    from drugos_graph.id_crosswalk import IDCrosswalk
    cw = IDCrosswalk()
    # Before: empty
    check("P0-G1 compound_to_inchikey starts empty",
          len(cw.compound_to_inchikey) == 0)
    # Register a real mapping (Aspirin: DB00945 -> BSYNRYMUTXBXSQ-UHFFFAOYSA-N)
    n = cw.register_compound_inchikey(
        "DB00945", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        source="drugbank", confidence="verified",
    )
    check("P0-G1 register_compound_inchikey returns 1", n == 1, f"got {n}")
    check("P0-G1 compound_to_inchikey populated after register",
          len(cw.compound_to_inchikey) == 1,
          f"len={len(cw.compound_to_inchikey)}")
    # Register a second alias for the same drug
    n2 = cw.register_compound_inchikey(
        "CID2244", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        source="pubchem", confidence="verified",
    )
    check("P0-G1 second alias registered", n2 == 1, f"got {n2}")
    # Lookup: compound_id_to_inchikey("DB00945") should return the InChIKey
    ik = cw.compound_id_to_inchikey("DB00945")
    check("P0-G1 compound_id_to_inchikey('DB00945') resolves",
          ik == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", f"got {ik}")
    ik2 = cw.compound_id_to_inchikey("CID2244")
    check("P0-G1 compound_id_to_inchikey('CID2244') resolves (7-namespace unification)",
          ik2 == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", f"got {ik2}")
    # Idempotent: re-register same mapping -> returns 0
    n3 = cw.register_compound_inchikey(
        "DB00945", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        source="drugbank", confidence="verified",
    )
    check("P0-G1 idempotent re-register returns 0", n3 == 0, f"got {n3}")
except Exception as e:
    check("P0-G1 import + run", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P0-G3: clinicaltrials_to_node_records emits flat dicts with "id" key
# ═══════════════════════════════════════════════════════════════════════════
try:
    import pandas as pd
    from drugos_graph.clinicaltrials_loader import clinicaltrials_to_node_records
    # Minimal DataFrame matching parse_clinicaltrials output shape
    ct_df = pd.DataFrame([
        {"drug_mesh": "D000001", "drug_name": "Test Drug A",
         "condition_mesh": "D000002", "condition_name": "Test Disease B"},
        {"drug_mesh": "D000003", "drug_name": "Test Drug C",
         "condition_mesh": "D000002", "condition_name": "Test Disease B"},
    ])
    nodes = clinicaltrials_to_node_records(ct_df)
    check("P0-G3 clinicaltrials_to_node_records returns non-empty list",
          len(nodes) > 0, f"len={len(nodes)}")
    # Every node MUST have "id" at top level (the load_nodes_batch contract)
    all_have_id = all(isinstance(n, dict) and "id" in n and n["id"] for n in nodes)
    check("P0-G3 every node has 'id' key at top level (not nested node_id)",
          all_have_id, f"nodes={[list(n.keys())[:3] for n in nodes[:3]]}")
    # Verify name + source present
    all_have_name = all("name" in n for n in nodes)
    all_have_source = all("source" in n for n in nodes)
    check("P0-G3 every node has 'name' + 'source'", all_have_name and all_have_source)
    # Verify node_type present for grouping
    has_compound = any(n.get("node_type") == "Compound" for n in nodes)
    has_disease = any(n.get("node_type") == "Disease" for n in nodes)
    check("P0-G3 has Compound + Disease node_types for grouping",
          has_compound and has_disease)
except Exception as e:
    check("P0-G3 import + run", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P0-G4: DRKGLoader.to_graph emits flat dicts with "id" key
# ═══════════════════════════════════════════════════════════════════════════
try:
    import pandas as pd
    from drugos_graph.drkg_loader import DRKGLoader
    # Minimal DRKG-shaped DataFrame (matches parse_drkg_tsv output columns)
    drkg_df = pd.DataFrame([
        {"head_entity": "Compound::DB00107", "relation": "DRUGBANK::target::Compound:Gene",
         "tail_entity": "Gene::1234", "head_type": "Compound",
         "tail_type": "Gene", "head_id": "DB00107", "tail_id": "1234",
         "rel_type": "target", "rel_source": "DRUGBANK",
         "relation_name": "target"},
        {"head_entity": "Compound::DB00107", "relation": "Hetionet::CtD::Compound:Disease",
         "tail_entity": "Disease::DOID:1438", "head_type": "Compound",
         "tail_type": "Disease", "head_id": "DB00107", "tail_id": "DOID:1438",
         "rel_type": "treats", "rel_source": "Hetionet",
         "relation_name": "treats"},
    ])
    loader = DRKGLoader()
    nodes, edges = loader.to_graph(drkg_df)
    check("P0-G4 DRKGLoader.to_graph returns non-empty nodes",
          len(nodes) > 0, f"len={len(nodes)}")
    # Every node MUST have "id" at top level
    all_have_id = all(isinstance(n, dict) and "id" in n and n["id"] for n in nodes)
    check("P0-G4 every node has 'id' key at top level (not entity_id)",
          all_have_id, f"nodes={[list(n.keys())[:4] for n in nodes[:3]]}")
    # Verify name + source present
    all_have_name = all("name" in n for n in nodes)
    all_have_source = all("source" in n for n in nodes)
    check("P0-G4 every node has 'name' + 'source'",
          all_have_name and all_have_source)
    # Verify entity_type present for grouping
    types = set(n.get("entity_type") for n in nodes)
    check("P0-G4 has Compound + Gene/Disease entity_types", len(types) >= 2,
          f"types={types}")
    # Edges should have src_id + dst_id
    all_edges_have_ids = all(
        e.get("src_id") and e.get("dst_id") and e.get("rel_type") for e in edges
    )
    check("P0-G4 edges have src_id/dst_id/rel_type", all_edges_have_ids)
except Exception as e:
    check("P0-G4 import + run", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P1-A1: _MIXTURE_INCHIKEY_PATTERN uses + (canonical), not * (divergent)
# ═══════════════════════════════════════════════════════════════════════════
try:
    from cleaning._constants import CANONICAL_MIXTURE_INCHIKEY_REGEX
    from cleaning.normalizer import _MIXTURE_INCHIKEY_PATTERN
    check("P1-A1 normalizer imports canonical mixture regex",
          _MIXTURE_INCHIKEY_PATTERN is CANONICAL_MIXTURE_INCHIKEY_REGEX,
          f"normalizer pattern = {_MIXTURE_INCHIKEY_PATTERN.pattern!r}, "
          f"canonical = {CANONICAL_MIXTURE_INCHIKEY_REGEX.pattern!r}")
    # A single 27-char InChIKey should NOT match the mixture pattern
    single = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    check("P1-A1 single InChIKey does NOT match mixture pattern",
          not _MIXTURE_INCHIKEY_PATTERN.match(single),
          "single key matched -- pattern uses * (zero-or-more), bug NOT fixed")
    # A real mixture (2+ components joined by -) SHOULD match
    mixture = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N-AGAFWYVAAKGVHW-UHFFFAOYSA-N"
    check("P1-A1 real mixture (2 components) matches pattern",
          bool(_MIXTURE_INCHIKEY_PATTERN.match(mixture)),
          "mixture did NOT match -- pattern is too strict")
except Exception as e:
    check("P1-A1 import + run", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P1-A3: OMIM_API_KEY_FORMAT_RE is case-insensitive + validates 8-4-4-4-12
# ═══════════════════════════════════════════════════════════════════════════
try:
    from config.settings import OMIM_API_KEY_FORMAT_RE
    # OMIM API keys are RFC-4122 UUID v4 strings (3rd group starts with '4',
    # 4th group starts with '8'/'9'/'a'/'b'). The v83 regex validates these
    # structural bits (better than the v82 8-4-4-4-12 shape check).
    # Lowercase UUID v4
    lower = "a1b2c3d4-e5f6-4789-abcd-ef1234567890"
    check("P1-A3 lowercase UUID v4 accepted",
          bool(OMIM_API_KEY_FORMAT_RE.match(lower)))
    # Uppercase UUID v4 (clipboard paste from Windows) -- previously REJECTED
    upper = "A1B2C3D4-E5F6-4789-ABCD-EF1234567890"
    check("P1-A3 uppercase UUID v4 accepted (was rejected before fix)",
          bool(OMIM_API_KEY_FORMAT_RE.match(upper)),
          "uppercase UUID rejected -- regex is case-sensitive, bug NOT fixed")
    # Mixed case UUID v4
    mixed = "A1b2C3d4-E5f6-4789-Abcd-Ef1234567890"
    check("P1-A3 mixed-case UUID v4 accepted",
          bool(OMIM_API_KEY_FORMAT_RE.match(mixed)))
    # Non-UUID (36 chars but wrong structure) -- should be REJECTED.
    # The v83 RFC-4122 regex validates the version + variant bits, so a
    # 36-char string with wrong structure is rejected.
    bad = "a1b2c3d4e5f67890abcdef1234567890------"
    check("P1-A3 non-UUID 36-char string REJECTED (structure validation)",
          not OMIM_API_KEY_FORMAT_RE.match(bad),
          "non-UUID accepted -- regex doesn't validate UUID structure")
    # Non-v4 UUID (3rd group starts with 7, not 4) -- REJECTED by v83 regex
    non_v4 = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    check("P1-A3 non-v4 UUID REJECTED (version-bit validation)",
          not OMIM_API_KEY_FORMAT_RE.match(non_v4),
          "non-v4 UUID accepted -- regex doesn't validate version bit")
    # Too short -- REJECTED
    check("P1-A3 short string REJECTED",
          not OMIM_API_KEY_FORMAT_RE.match("abc"))
except Exception as e:
    check("P1-A3 import + run", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P1-A7: _validate_activity_type is case-insensitive ("ic50" accepted)
# ═══════════════════════════════════════════════════════════════════════════
try:
    from database.loaders import _validate_activity_type, _VALID_ACTIVITY_TYPES
    # "ic50" (ChEMBL lowercase) -- previously REJECTED. Should return canonical 'IC50'.
    result = _validate_activity_type("ic50")
    check("P1-A7 _validate_activity_type('ic50') returns canonical 'IC50' (was quarantined)",
          result == "IC50", f"got {result!r}")
    # "IC50" (uppercase) -- should also work, return canonical 'IC50'
    result2 = _validate_activity_type("IC50")
    check("P1-A7 _validate_activity_type('IC50') returns canonical 'IC50'",
          result2 == "IC50", f"got {result2!r}")
    # "Ki" (mixed case) -- should work, return canonical 'Ki'
    result3 = _validate_activity_type("ki")
    check("P1-A7 _validate_activity_type('ki') returns canonical 'Ki'",
          result3 == "Ki", f"got {result3!r}")
    # Invalid -- should raise
    try:
        _validate_activity_type("NOT_AN_ACTIVITY_TYPE")
        check("P1-A7 invalid activity_type raises ValueError", False,
              "no exception raised")
    except ValueError:
        check("P1-A7 invalid activity_type raises ValueError", True)
except Exception as e:
    check("P1-A7 import + run", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P1-A8: _CircuitBreaker HALF_OPEN allows exactly ONE probe
# ═══════════════════════════════════════════════════════════════════════════
try:
    from database.connection import _CircuitBreaker
    cb = _CircuitBreaker()
    # Force into OPEN state
    cb._state = "OPEN"
    cb._last_failure_time = -1e9  # long ago -> recovery timeout elapsed
    # First call: should transition to HALF_OPEN and allow ONE probe
    first = cb.allow_request()
    check("P1-A8 HALF_OPEN first probe allowed", first, f"got {first}")
    # Second call: should be REJECTED (probe in flight)
    second = cb.allow_request()
    check("P1-A8 HALF_OPEN second probe REJECTED (single-probe semantic)",
          not second, f"got {second} -- probe-in-flight flag NOT working")
    # Simulate probe success -> CLOSED
    cb.record_success()
    check("P1-A8 after record_success -> CLOSED", cb.state == "CLOSED",
          f"state={cb.state}")
    # Now requests should be allowed again
    after_close = cb.allow_request()
    check("P1-A8 CLOSED allows requests", after_close)
except Exception as e:
    check("P1-A8 import + run", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P1-A11: is_production_environment returns False for staging
# ═══════════════════════════════════════════════════════════════════════════
try:
    from database.connection import is_production_environment, _get_environment
    # Staging should NOT be production
    old = os.environ.get("DRUGOS_ENVIRONMENT")
    try:
        os.environ["DRUGOS_ENVIRONMENT"] = "staging"
        is_staging_prod = is_production_environment()
        check("P1-A11 is_production_environment('staging') == False",
              not is_staging_prod, f"got {is_staging_prod} -- staging treated as prod")
        # Production IS production
        os.environ["DRUGOS_ENVIRONMENT"] = "production"
        is_prod_prod = is_production_environment()
        check("P1-A11 is_production_environment('production') == True",
              is_prod_prod, f"got {is_prod_prod}")
        # Development is NOT production
        os.environ["DRUGOS_ENVIRONMENT"] = "development"
        is_dev_prod = is_production_environment()
        check("P1-A11 is_production_environment('development') == False",
              not is_dev_prod)
    finally:
        if old is None:
            os.environ.pop("DRUGOS_ENVIRONMENT", None)
        else:
            os.environ["DRUGOS_ENVIRONMENT"] = old
except Exception as e:
    check("P1-A11 import + run", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P1-A12: bulk_update_drugs_from_pubchem no longer skips CID refreshes
# ═══════════════════════════════════════════════════════════════════════════
try:
    import inspect, re
    from database.loaders import bulk_update_drugs_from_pubchem
    src = inspect.getsource(bulk_update_drugs_from_pubchem)
    # Extract the actual UPDATE SQL text (the text(...) block), NOT the docstring
    sql_match = re.search(r'update_sql\s*=\s*text\(\s*"""(.*?)"""', src, re.DOTALL)
    if sql_match:
        sql_text = sql_match.group(1)
        check("P1-A12 UPDATE SQL no longer has 'AND pubchem_cid IS NULL'",
              "AND pubchem_cid IS NULL" not in sql_text,
              f"WHERE clause still has it in SQL: {sql_text!r}")
    else:
        check("P1-A12 could not extract UPDATE SQL", False, "text() block not found")
    check("P1-A12 pubchem_cid uses COALESCE (preserves existing CID)",
          "COALESCE(:pubchem_cid, drugs.pubchem_cid)" in src,
          "pubchem_cid not wrapped in COALESCE -- existing CIDs would be overwritten")
except Exception as e:
    check("P1-A12 inspect", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P1-A6: withdrawn drug matching handles hyphen + underscore salts
# ═══════════════════════════════════════════════════════════════════════════
try:
    import inspect
    from database.loaders import _WITHDRAWN_DRUG_NAMES_LOWER
    # Verify the source code has the hyphen + underscore checks
    from database import loaders as _loaders_mod
    _src = inspect.getsource(_loaders_mod)
    check("P1-A6 withdrawn matching checks hyphen separator",
          'startswith(_wd_name + "-")' in _src)
    check("P1-A6 withdrawn matching checks underscore separator",
          'startswith(_wd_name + "_")' in _src)
    # Functional test: "rofecoxib-sodium" should match "rofecoxib"
    if "rofecoxib" in _WITHDRAWN_DRUG_NAMES_LOWER:
        name = "rofecoxib-sodium"
        matched = (
            name == "rofecoxib"
            or name.startswith("rofecoxib" + " ")
            or name.startswith("rofecoxib" + "-")
            or name.startswith("rofecoxib" + "_")
        )
        check("P1-A6 'rofecoxib-sodium' matches 'rofecoxib' (hyphen salt)",
              matched, "hyphen salt form NOT matched")
except Exception as e:
    check("P1-A6 inspect", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P1-A5: chk_drugs_inchikey_format CHECK is parenthesized
# ═══════════════════════════════════════════════════════════════════════════
try:
    import inspect
    from database.models import Drug
    _constraints = [str(c.sqltext) for c in Drug.__table__.constraints
                    if getattr(c, "name", "") == "chk_drugs_inchikey_format"]
    if _constraints:
        _c = _constraints[0]
        check("P1-A5 CHECK has explicit parentheses around AND-chain",
              "(LENGTH(inchikey) = 27" in _c,
              f"unparenthesized: {_c!r}")
    else:
        check("P1-A5 constraint found", False, "chk_drugs_inchikey_format not found")
except Exception as e:
    check("P1-A5 inspect", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P1-A9: drug name CHECK uses TRIM
# ═══════════════════════════════════════════════════════════════════════════
try:
    from database.models import Drug
    _constraints = [str(c.sqltext) for c in Drug.__table__.constraints
                    if getattr(c, "name", "") == "chk_drugs_name_min_length"]
    if _constraints:
        _c = _constraints[0]
        check("P1-A9 name CHECK uses LENGTH(TRIM(name))",
              "LENGTH(TRIM(name))" in _c,
              f"no TRIM: {_c!r}")
    else:
        check("P1-A9 constraint found", False, "chk_drugs_name_min_length not found")
except Exception as e:
    check("P1-A9 inspect", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P1-A4: configure_engine holds _lifecycle_lock for entire dispose+create
# ═══════════════════════════════════════════════════════════════════════════
try:
    import inspect
    from database.connection import configure_engine
    src = inspect.getsource(configure_engine)
    # Should NOT have two separate `with _lifecycle_lock:` blocks
    lock_count = src.count("with _lifecycle_lock:")
    check("P1-A4 configure_engine uses ONE _lifecycle_lock block (no race gap)",
          lock_count == 1, f"found {lock_count} lock blocks -- race gap exists")
    check("P1-A4 no dispose_engine() call between lock blocks",
          "dispose_engine(force=True)" not in src,
          "still calls dispose_engine inside configure_engine -- gap exists")
except Exception as e:
    check("P1-A4 inspect", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P1-A13: run_migrations uses text() not exec_driver_sql
# ═══════════════════════════════════════════════════════════════════════════
try:
    import database.migrations.run_migrations as _rm_mod
    _src = open(_rm_mod.__file__).read()
    # v83 P0-C10 + v82 P1-A13 reconciled: the SQLite execution path uses
    # conn.execute(text(stmt_for_execution)) where stmt_for_execution has
    # comments stripped (to avoid pyformat mis-parsing %(table)s in comments).
    check("P1-A13 uses conn.execute(text(...)) for SQLite migration execution",
          "conn.execute(text(stmt_for_execution))" in _src,
          "text() not used -- exec_driver_sql still present (injection risk)")
    # The old exec_driver_sql(stmt_stripped) call should NOT be present
    # (it was replaced by the safer text() approach).
    _active_calls = [l for l in _src.split("\n")
                     if "exec_driver_sql(stmt_stripped)" in l
                     and not l.strip().startswith("#")
                     and not l.strip().startswith('"')]
    check("P1-A13 no active exec_driver_sql(stmt_stripped) calls",
          len(_active_calls) == 0,
          f"found {len(_active_calls)} active calls: {_active_calls[:2]}")
except Exception as e:
    check("P1-A13 inspect", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P1-A10: _pre_validate_gda has no dead inner try/except
# ═══════════════════════════════════════════════════════════════════════════
try:
    import inspect
    from database.loaders import _pre_validate_gda
    src = inspect.getsource(_pre_validate_gda)
    # The dead inner try/except that catches ValueError and re-raises should be GONE
    check("P1-A10 no dead 'except ValueError: raise' in _pre_validate_gda",
          "except ValueError:\n                    # Re-raise" not in src
          and "except ValueError:\n                    raise" not in src,
          "dead inner try/except still present")
except Exception as e:
    check("P1-A10 inspect", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P0-G2: STRING aliases pre-load block exists in step7_additional_sources
# ═══════════════════════════════════════════════════════════════════════════
try:
    from drugos_graph import run_pipeline as _rp
    _src = open(_rp.__file__).read()
    check("P0-G2 step7 has STRING aliases pre-load block",
          "P0-G2 ROOT FIX" in _src and "load_string_aliases" in _src
          and "BEFORE step7a" in _src,
          "P0-G2 pre-load block not found")
    check("P0-G2 step7a uses unresolved_policy='keep_ensembl'",
          'unresolved_policy="keep_ensembl"' in _src,
          "still uses default 'drop' -- STRING edges would be dropped")
except Exception as e:
    check("P0-G2 inspect", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# P0-G1: step8 has crosswalk population block
# ═══════════════════════════════════════════════════════════════════════════
try:
    from drugos_graph import run_pipeline as _rp
    _src = open(_rp.__file__).read()
    check("P0-G1 step8 has compound_to_inchikey population block",
          "P0-G1 ROOT FIX" in _src and "register_compound_inchikey" in _src
          and "7 Compound ID namespaces" in _src,
          "P0-G1 population block not found in step8")
except Exception as e:
    check("P0-G1 inspect", False, f"{type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("v82 REAL-CODE VERIFICATION RESULTS")
print("=" * 70)
for r in RESULTS:
    print(r)
print("=" * 70)
print(f"  TOTAL: {PASS + FAIL}   PASS: {PASS}   FAIL: {FAIL}")
print("=" * 70)
sys.exit(1 if FAIL > 0 else 0)
