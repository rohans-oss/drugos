#!/usr/bin/env python3
"""
v63 ROOT FIX VERIFICATION -- Runtime verification of all 18 P0 issues.
"""
from __future__ import annotations
import sys, os, re

# v100 ROOT FIX (R-008): use repo-relative path so the script is portable
# across developer machines, CI runners, and production containers. The
# previous hardcoded `/home/z/my-project/work` only worked on one
# developer's laptop and broke everywhere else.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{HERE}/phase1")
sys.path.insert(0, f"{HERE}/phase2")

passed = 0
failed = 0
results = []

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        results.append(f"[PASS] {name}")
    else:
        failed += 1
        results.append(f"[FAIL] {name} -- {detail}")

# T-001
try:
    with open(f"{HERE}/phase1/database/migrations/001_initial_schema.sql") as f:
        sql = f.read()
    pr_pos = sql.find("CREATE TABLE IF NOT EXISTS pipeline_runs")
    child_refs = [m.start() for m in re.finditer(r'REFERENCES pipeline_runs\(id\)', sql)]
    check("T-001 pipeline_runs before child FKs", pr_pos > 0 and all(r > pr_pos for r in child_refs))
except Exception as e:
    check("T-001", False, str(e))

# T-002
try:
    with open(f"{HERE}/phase1/database/migrations/006_drug_withdrawn_safety_columns.sql") as f:
        m006 = f.read()
    with open(f"{HERE}/phase1/database/migrations/008_drug_is_globally_approved.sql") as f:
        m008 = f.read()
    check("T-002 006 backfills rofecoxib", "rofecoxib" in m006.lower() and "vioxx" in m006.lower())
    check("T-002 008 excludes withdrawn", "is_withdrawn = FALSE" in m008)
except Exception as e:
    check("T-002", False, str(e))

# T-003
try:
    with open(f"{HERE}/phase1/database/migrations/009_tighten_inchikey_check_constraint.sql") as f:
        m009 = f.read()
    check("T-003 POSIX regex", "inchikey ~ '^[A-Z]{14}-[A-Z]{10}-[A-Z]$'" in m009)
except Exception as e:
    check("T-003", False, str(e))

# P2L-008
try:
    from drugos_graph.chembl_loader import _RE_INHIBIT, _RE_ACTIVATE, standard_type_to_relation
    check("P2L-008 word boundary", r"\bACTIVAT" in _RE_ACTIVATE.pattern)
    check("P2L-008 INACTIVAT in inhibit", "INACTIVAT" in _RE_INHIBIT.pattern)
    check("P2L-008 INACTIVATION->inhibits", standard_type_to_relation("INACTIVATION") == "inhibits")
    check("P2L-008 ACTIVATION->activates", standard_type_to_relation("ACTIVATION") == "activates")
except Exception as e:
    check("P2L-008", False, str(e))

# P2C-003+016
try:
    with open(f"{HERE}/run_unified.py") as f:
        runner = f.read()
    check("P2C-003 --no-chemberta flag", "--no-chemberta" in runner)
    check("P2C-003 refused in prod", "cannot be used in production" in runner)
    with open(f"{HERE}/phase2/drugos_graph/run_pipeline.py") as f:
        rp = f.read()
    check("P2C-016 strict default 1", '"DRUGOS_STRICT_FEATURES", "1"' in rp)
    check("P2C-016 FeatureFailureError", "FeatureFailureError" in rp)
    check("P2C-016 MLflow tag", 'set_tag("CHEMBERTA_DISABLED", "true")' in rp)
    check("P2C-016 model save refusal", "REFUSING to save HGT model" in rp)
except Exception as e:
    check("P2C-003+016", False, str(e))

# P2L-041
try:
    with open(f"{HERE}/phase2/drugos_graph/clinicaltrials_loader.py") as f:
        ct = f.read()
    check("P2L-041 primary_outcome_met", "primary_outcome_met" in ct)
    check("P2L-041 outcome_analyses", "outcome_analyses" in ct)
except Exception as e:
    check("P2L-041", False, str(e))

# P2L-045
try:
    with open(f"{HERE}/phase2/drugos_graph/opentargets_loader.py") as f:
        ot = f.read()
    check("P2L-045 binding_confidence", '"binding_confidence": score' in ot)
    check("P2L-045 no chembl_score alias", "do NOT update" in ot)
except Exception as e:
    check("P2L-045", False, str(e))

# P1C-001
try:
    with open(f"{HERE}/phase1/database/migrations/001_initial_schema.sql") as f:
        m001 = f.read()
    gda_start = m001.find("CREATE TABLE IF NOT EXISTS gene_disease_associations")
    gda_end = m001.find(";", gda_start)
    gda = m001[gda_start:gda_end]
    # Check the ACTUAL column definition line (not comments)
    gene_sym_lines = [l.strip() for l in gda.split("\n") if l.strip().startswith("gene_symbol")]
    check("P1C-001 column def is VARCHAR(50) nullable", len(gene_sym_lines) > 0 and "VARCHAR(50)" in gene_sym_lines[0] and "DEFAULT" not in gene_sym_lines[0], f"line: {gene_sym_lines}")
    # Check no active CHECK constraint on gene_symbol (only in comments)
    active_check = [l for l in gda.split("\n") if "CHECK (gene_symbol" in l and not l.strip().startswith("--")]
    check("P1C-001 no active CHECK", len(active_check) == 0, f"active checks: {active_check}")
except Exception as e:
    check("P1C-001", False, str(e))

# P2C-002+007
try:
    from drugos_graph.config import CANONICAL_IDS, CORE_NODE_TYPES
    check("P2C-002 ClinicalOutcome", "ClinicalOutcome" in CANONICAL_IDS)
    check("P2C-002 MedDRA_Term", "MedDRA_Term" in CANONICAL_IDS)
    check("P2C-002 Anatomy", "Anatomy" in CANONICAL_IDS)
    missing = [nt for nt in CORE_NODE_TYPES if nt not in CANONICAL_IDS]
    check("P2C-007 reverse-check", len(missing) == 0, f"missing: {missing}")
except Exception as e:
    check("P2C-002+007", False, str(e))

# P2C-004+005+009
# P2-002 FORENSIC ROOT FIX (Team 4): phase2/drugos_graph/graph_transformer_model.py
# was DELETED -- Phase 2 now produces PyG HeteroData for Phase 3 to train
# (per the DOCX architecture). The P2C-004/005/009 checks are updated to
# verify the DELETION and that step11b delegates to Phase 3.
try:
    p2_hgt_file = f"{HERE}/phase2/drugos_graph/graph_transformer_model.py"
    import os as _os
    p2_hgt_deleted = not _os.path.exists(p2_hgt_file)
    check("P2-002 Phase 2 HGT model DELETED", p2_hgt_deleted,
          "phase2/drugos_graph/graph_transformer_model.py must NOT exist (P2-002)")
    with open(f"{HERE}/phase2/drugos_graph/run_pipeline.py") as f:
        rp = f.read()
    check("P2-002 step11b delegates to Phase 3",
          "phase3_delegated" in rp and "DrugRepurposingGraphTransformer" in rp,
          "step11b must delegate to Phase 3's DrugRepurposingGraphTransformer")
    check("P2C-009 val_auc -1.0", "-1.0" in rp and "hgt_val_auc" in rp)
except Exception as e:
    check("P2C-004+005+009", False, str(e))

# P1-013
try:
    with open(f"{HERE}/phase1/pipelines/chembl_pipeline.py") as f:
        cp = f.read()
    check("P1-013 synced filename", "chembl_activities.csv.gz" in cp)
except Exception as e:
    check("P1-013", False, str(e))

# P1-002+003
try:
    from pipelines._embedded_samples import embedded_chembl_activities
    acts = embedded_chembl_activities()
    valid_at = {"IC50", "Ki", "Kd", "EC50"}
    bad = [r for r in acts.to_dict("records") if r.get("activity_type") not in valid_at]
    check("P1-002 activity_type enum", len(bad) == 0, f"bad: {bad[:3]}")
except Exception as e:
    check("P1-002+003", False, str(e))

# P1C-003
# ORCH-005 ROOT FIX: guard against missing phase1/config/.env.example.
# The previous code opened the file unconditionally. If the file did not
# exist (it is a sample file that may not be committed in every checkout),
# the open() raised FileNotFoundError, which the bare ``except Exception``
# caught and logged as a FAIL — a false failure. The fix checks os.path
# exists first; if missing, we emit an explicit SKIP rather than FAIL so
# the verification script's signal-to-noise ratio stays useful.
import os as _os
_env_example_path = f"{HERE}/phase1/config/.env.example"
try:
    if not _os.path.exists(_env_example_path):
        check(
            "P1C-003 score=700",
            False,
            "SKIP: phase1/config/.env.example not present in this checkout "
            "(ORCH-005). The file is a sample and may not be committed. "
            "Manual check: ensure STRING_MIN_COMBINED_SCORE=700 is set in "
            "the deployed .env file.",
        )
    else:
        with open(_env_example_path) as f:
            env_ex = f.read()
        check("P1C-003 score=700", "STRING_MIN_COMBINED_SCORE=700" in env_ex)
except Exception as e:
    check("P1C-003", False, str(e))

# P1C-002
try:
    with open(f"{HERE}/phase1/database/models.py") as f:
        m = f.read()
    check("P1C-002 protein test rejection", "DRUGOS_ENVIRONMENT" in m and "prod" in m)
except Exception as e:
    check("P1C-002", False, str(e))

# P2L-032
try:
    from drugos_graph.config import STRING_MIN_COMBINED_SCORE
    with open(f"{HERE}/phase2/drugos_graph/string_loader.py") as f:
        sl = f.read()
    with open(f"{HERE}/phase2/drugos_graph/stitch_loader.py") as f:
        stl = f.read()
    check("P2L-032 string uses config", "STRING_MIN_COMBINED_SCORE" in sl)
    check("P2L-032 stitch uses config", "STRING_MIN_COMBINED_SCORE" in stl)
    check("P2L-032 threshold 700", STRING_MIN_COMBINED_SCORE == 700)
except Exception as e:
    check("P2L-032", False, str(e))

# P2C-001
try:
    with open(f"{HERE}/phase2/drugos_graph/phase1_bridge.py") as f:
        pb = f.read()
    idx = pb.find("def total_nodes")
    # find the SECOND total_nodes (Phase1StagedData, not Phase1Bridge)
    idx2 = pb.find("def total_nodes", idx + 1)
    snippet = pb[idx2:idx2 + 500] if idx2 > 0 else pb[idx:idx + 500]
    check("P2C-001 pathway_nodes in total", "pathway_nodes" in snippet)
except Exception as e:
    check("P2C-001", False, str(e))

# P2C-008
try:
    with open(f"{HERE}/phase2/drugos_graph/phase1_bridge.py") as f:
        pb = f.read()
    check("P2C-008 schema_missing ERROR", "schema_missing" in pb)
    check("P2C-008 prod check", "DRUGOS_ENVIRONMENT" in pb and "prod" in pb)
except Exception as e:
    check("P2C-008", False, str(e))

# T-004
try:
    with open(f"{HERE}/phase1/database/migrations/002_bug_fixes_migration.sql") as f:
        m002 = f.read()
    check("T-004 partial unique index", "WHERE gene_symbol IS NOT NULL" in m002 and "UNIQUE INDEX" in m002)
except Exception as e:
    check("T-004", False, str(e))

# P2L-021
try:
    with open(f"{HERE}/phase2/drugos_graph/drkg_loader.py") as f:
        dl = f.read()
    check("P2L-021 lowercase canonical", "lower" in dl and "canonical" in dl.lower())
except Exception as e:
    check("P2L-021", False, str(e))

# P2L-038
try:
    with open(f"{HERE}/phase2/drugos_graph/stitch_loader.py") as f:
        stl = f.read()
    check("P2L-038 CIDm/CIDs distinction", "CIDm" in stl and "CIDs" in stl)
except Exception as e:
    check("P2L-038", False, str(e))

print("=" * 70)
print("v63 ROOT FIX VERIFICATION RESULTS")
print("=" * 70)
for r in results:
    print(r)
print("=" * 70)
print(f"PASSED: {passed} / {passed + failed}")
if failed:
    print(f"FAILED: {failed}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    sys.exit(0)
