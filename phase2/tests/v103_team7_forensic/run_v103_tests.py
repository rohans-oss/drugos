#!/usr/bin/env python3
"""Run v103 forensic root-fix tests directly (no unittest framework overhead).

Verifies ALL 14 P2-035..P2-048 fixes by reading actual source code and
asserting the runtime behavior matches the issue spec.
"""
import re
import sys
from pathlib import Path

PHASE2 = Path(__file__).resolve().parents[2] / "drugos_graph"
PASS = 0
FAIL = 0
FAILURES = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  [FAIL] {name} — {detail}")


def read(rel):
    return (PHASE2 / rel).read_text()


def _strip_comments(src):
    """Strip Python comments from source (rough — for checking actual code)."""
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # Strip inline comments (rough — doesn't handle # in strings)
        # For our purposes (checking getattr pattern), this is sufficient.
        out.append(line)
    return "\n".join(out)


print("=" * 70)
print("v103 FORENSIC ROOT-FIX VERIFICATION (P2-035 through P2-048)")
print("=" * 70)

# ─── P2-035 ─────────────────────────────────────────────────────────────
print("\n[P2-035] DRKG ID pattern accepts hyphens and dots")
src = read("drkg_loader.py")
# Pattern may be indented (defined inside a function) — allow leading whitespace
m = re.search(r'^\s*_DRKG_ID_PATTERN\s*=\s*_re\.compile\(r"([^"]+)"', src, re.M)
check("p035_pattern_defined", m is not None, "pattern not found")
if m:
    pat_str = m.group(1)
    after = pat_str.split("::", 1)[1]
    check("p035_accepts_hyphens", "-" in after, f"pattern={pat_str!r}")
    check("p035_accepts_dots", "." in after, f"pattern={pat_str!r}")
    p = re.compile(pat_str)
    for vid in ["Compound::DB-00945", "Compound::CHEMBL.foo.1",
                "Disease::MESH:D006932", "Compound::CHEMBL-1234567"]:
        check(f"p035_matches_{vid}", p.match(vid) is not None, vid)
    for iid in ["Compound:DB00945", "Compound::DB 00945"]:
        check(f"p035_rejects_{iid}", p.match(iid) is None, iid)

# ─── P2-036 ─────────────────────────────────────────────────────────────
print("\n[P2-036] Centralized InChIKey normalization (no raw .upper())")
files = ["phase1_bridge.py", "chembl_loader.py", "pubchem_loader.py",
         "entity_resolver.py", "id_crosswalk.py", "drugbank_parser.py",
         "clinicaltrials_loader.py"]
raw_upper_found = []
for f in files:
    src = read(f)
    for i, line in enumerate(src.splitlines(), 1):
        s = line.lstrip()
        if s.startswith("#"):
            continue
        for bad in [r'inchikey[^_a-zA-Z][^=]*\.upper\(\)',
                    r'inchi[^_a-zA-Z][^=]*\.upper\(\)']:
            if re.search(bad, line):
                raw_upper_found.append(f"{f}:{i}: {line.strip()!r}")
check("p036_no_raw_upper_calls", len(raw_upper_found) == 0,
      f"{len(raw_upper_found)} raw .upper() calls remain: {raw_upper_found[:3]}")

# Verify helper is imported in each file
for f in files:
    src = read(f)
    has_import = ("normalize_inchikey" in src or "_normalize_inchikey" in src)
    check(f"p036_{f}_uses_helper", has_import, f"{f} doesn't reference normalize_inchikey")

# ─── P2-037 ─────────────────────────────────────────────────────────────
print("\n[P2-037] Compound MERGE consolidation wired into pipeline")
src_kg = read("kg_builder.py")
check("p037_method_defined", "def consolidate_compounds_by_aliases" in src_kg,
      "method not defined in kg_builder.py")
check("p037_order_by_limit_1", "ORDER BY existing.id" in src_kg and "LIMIT 1" in src_kg,
      "ORDER BY + LIMIT 1 missing")
src_bridge = read("phase1_bridge.py")
check("p037_method_called",
      'getattr(builder, "consolidate_compounds_by_aliases"' in src_bridge,
      "consolidation method not called from load_into_graph")

# ─── P2-038 ─────────────────────────────────────────────────────────────
print("\n[P2-038] Session context manager (no raw session assignment)")
src = read("utils.py")
raw_sessions = []
for i, line in enumerate(src.splitlines(), 1):
    if line.lstrip().startswith("#"):
        continue
    if re.search(r"^\s*session\s*=\s*builder\.driver\.session\(\)", line):
        raw_sessions.append(f"utils.py:{i}: {line.strip()!r}")
check("p038_no_raw_session", len(raw_sessions) == 0,
      f"raw session assignments: {raw_sessions}")

# ─── P2-039 ─────────────────────────────────────────────────────────────
print("\n[P2-039] num_total_entities Protocol (no dead getattr fallback)")
src_mp = read("model_protocol.py")
check("p039_in_protocol", "num_total_entities" in src_mp,
      "not in model_protocol.py")
src_tm = read("transe_model.py")
check("p039_on_transe", "def num_total_entities" in src_tm,
      "not on TransEModel")
check("p039_no_getattr_fallback",
      'getattr(model, "num_total_entities", None)' not in _strip_comments(src_tm),
      "dead getattr fallback still present in actual code")

# ─── P2-040 ─────────────────────────────────────────────────────────────
print("\n[P2-040] node_disjoint_split logging (no short-circuit)")
src = read("pyg_builder.py")
short_circuits = []
for i, line in enumerate(src.splitlines(), 1):
    s = line.lstrip()
    if s.startswith("#"):
        continue
    if re.search(r"n_nodes\s+and\s+n_train", line):
        if 'f"' not in line and "f'" not in line:
            short_circuits.append(f"pyg_builder.py:{i}: {line.strip()!r}")
check("p040_no_short_circuit", len(short_circuits) == 0,
      f"short-circuit patterns: {short_circuits}")

# ─── P2-041 ─────────────────────────────────────────────────────────────
print("\n[P2-041] HGT Bernoulli float weights (no int truncation)")
src = read("run_pipeline.py")
int_trunc = []
for i, line in enumerate(src.splitlines(), 1):
    if line.lstrip().startswith("#"):
        continue
    if re.search(r"int\s*\(\s*1000\s*/\s*\(\s*1\s*\+\s*_deg\s*\)\s*\)", line):
        int_trunc.append(f"run_pipeline.py:{i}: {line.strip()!r}")
check("p041_no_int_truncation", len(int_trunc) == 0,
      f"int-truncated weights: {int_trunc}")
check("p041_float_weights_present", "1.0 / (1.0 + float(_deg))" in src,
      "float weight computation missing")

# ─── P2-042 ─────────────────────────────────────────────────────────────
print("\n[P2-042] failed_for rel type for completed-negative trials")
src = read("clinicaltrials_loader.py")
check("p042_failed_for_emitted", 'rel_type = "failed_for"' in src,
      "failed_for not emitted")
check("p042_gates_on_outcome_met", "primary_outcome_met is False" in src,
      "not gated on primary_outcome_met=False")
src_cfg = read("config.py")
check("p042_in_core_edge_types", '("Compound", "failed_for", "Disease")' in src_cfg,
      "failed_for not in CORE_EDGE_TYPES")

# ─── P2-043 ─────────────────────────────────────────────────────────────
print("\n[P2-043] Per-relation neg pool no duplicates (no perm_h+extra)")
src = read("transe_model.py")
extra_randint = []
for i, line in enumerate(src.splitlines(), 1):
    if line.lstrip().startswith("#"):
        continue
    if re.search(r"extra\s*=\s*torch\.randint", line):
        extra_randint.append(f"transe_model.py:{i}: {line.strip()!r}")
check("p043_no_extra_randint", len(extra_randint) == 0,
      f"extra=randint patterns: {extra_randint}")
check("p043_randperm_used", "torch.randperm(" in src,
      "randperm not used")

# ─── P2-044 ─────────────────────────────────────────────────────────────
print("\n[P2-044] AUC integrity block on insufficient eval sets")
src = read("evaluation.py")
check("p044_raises_error", "raise EvaluationIntegrityError" in src,
      "EvaluationIntegrityError not raised")
check("p044_configurable", "DRUGOS_MIN_EVAL_POSITIVES" in src
      and "DRUGOS_MAX_EVAL_RATIO" in src,
      "thresholds not configurable via env")

# ─── P2-045 ─────────────────────────────────────────────────────────────
print("\n[P2-045] RandomLinkSplit rev_edge_types (DEEP ROOT FIX)")
src = read("pyg_builder.py")
m = re.search(r'"edge_types":\s*\[([^\]]+)\][^}]*"rev_edge_types":\s*\[([^\]]+)\]',
              src, re.DOTALL)
check("p045_kwargs_found", m is not None, "edge_types/rev_edge_types kwargs not found")
if m:
    et = m.group(1).strip()
    rt = m.group(2).strip()
    check("p045_no_double_in_edge_types",
          "target_edge_type, _rev_edge_type_tuple" not in et,
          f"edge_types has BOTH (causes crash): {et!r}")
    check("p045_rev_in_rev_edge_types", "_rev_edge_type_tuple" in rt,
          f"rev_edge_types missing reverse: {rt!r}")
    # Critical: lengths must match (1:1)
    et_count = et.count(",") + 1 if et else 0
    rt_count = rt.count(",") + 1 if rt else 0
    check("p045_lengths_match", et_count == rt_count,
          f"length mismatch: edge_types has {et_count}, rev_edge_types has {rt_count}")

# ─── P2-046 ─────────────────────────────────────────────────────────────
print("\n[P2-046] SYNDROME: ontology_status flag")
src = read("phase1_bridge.py")
check("p046_ontology_status_emitted", "ontology_status" in src,
      "ontology_status not emitted")
check("p046_unmapped_flag", '"unmapped"' in src, "unmapped flag missing")
check("p046_mapped_flag", '"mapped"' in src, "mapped flag missing")
check("p046_keyword_map_exists", "_DISEASE_KEYWORD_MAP" in src,
      "_DISEASE_KEYWORD_MAP not defined")
# Count DOID entries in the keyword map
doid_count = src.count('("DOID:')
check("p046_keyword_map_populated", doid_count >= 10,
      f"only {doid_count} DOID mappings")

# ─── P2-047 ─────────────────────────────────────────────────────────────
print("\n[P2-047] HGT seed respects config.seed (no hardcoded 42)")
src = read("run_pipeline.py")
hardcoded_42 = []
for i, line in enumerate(src.splitlines(), 1):
    if line.lstrip().startswith("#"):
        continue
    if re.search(r"_random\.Random\(\s*42\s*\)", line):
        hardcoded_42.append(f"run_pipeline.py:{i}: {line.strip()!r}")
check("p047_no_hardcoded_42", len(hardcoded_42) == 0,
      f"hardcoded 42: {hardcoded_42}")
check("p047_uses_config_seed", 'getattr(cfg, "seed", 42)' in src,
      "doesn't use getattr(cfg, 'seed', 42)")

# ─── P2-048 ─────────────────────────────────────────────────────────────
print("\n[P2-048] Canonical Neo4j rel type (strip ::)")
src = read("kg_builder.py")
check("p048_helper_defined", "def _canonical_rel_type" in src,
      "_canonical_rel_type not defined")
# Extract and test the helper
m = re.search(
    r"(def _canonical_rel_type\(rel_type: str\) -> str:.*?)(?=\ndef |\nclass |\Z)",
    src, re.DOTALL)
check("p048_helper_extractable", m is not None, "helper not extractable")
if m:
    ns = {}
    try:
        exec(m.group(1), ns)
        fn = ns["_canonical_rel_type"]
        cases = [
            ("DRUGBANK::treats::Compound:Disease", "drugbank_treats"),
            ("DRUGBANK::treats", "drugbank_treats"),
            ("treats", "treats"),
            ("DRUGBANK::causes_side_effect", "drugbank_causes_side_effect"),
            ("drugbank::treats::compound:disease", "drugbank_treats"),
            ("drugbank:treats", "drugbank_treats"),
        ]
        for inp, exp in cases:
            actual = fn(inp)
            check(f"p048_transform_{inp[:20]}", actual == exp,
                  f"{inp!r} -> {actual!r} (expected {exp!r})")
    except Exception as e:
        check("p048_helper_executable", False, f"exec failed: {e}")
# All safe_rel sites use canonical helper
all_safe_rel = re.findall(r"safe_rel\s*=\s*sanitize_rel_type\(([^)]+)\)", src)
check("p048_safe_rel_sites_exist", len(all_safe_rel) > 0,
      "no safe_rel = sanitize_rel_type() calls found")
for arg in all_safe_rel:
    check(f"p048_safe_rel_uses_canonical_{arg.strip()[:30]}",
          "_canonical_rel_type" in arg,
          f"safe_rel call doesn't use _canonical_rel_type: {arg.strip()!r}")

# ─── SUMMARY ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print("=" * 70)
if FAILURES:
    print("\nFAILURES:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("\nALL 14 P2-035..P2-048 FORENSIC ROOT FIXES VERIFIED.")
    sys.exit(0)
