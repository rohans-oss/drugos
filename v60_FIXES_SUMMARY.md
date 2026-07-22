# v60 ROOT FIX RELEASE — Forensic Deep-Level Fix Summary

## Executive Summary

All 10 critical issues from the audit have been addressed. **6 issues were already correctly fixed in v57/v58** (verified by running the actual code). **4 issues required deeper root-cause fixes** that are now applied in v60. All 11 verification tests PASS. The HGT model trains successfully end-to-end with BCEWithLogitsLoss on raw logits (no `log(0) → -inf`).

## Test Results

```
v60 ROOT FIX TEST RESULTS: 11 passed, 0 failed (of 11)
ALL TESTS PASSED — all 10 critical issues + integration verified.
```

## Issue-by-Issue Status

### Issue #1: ChEMBL _RE_ACTIVATE regex matches INACTIVATION → covalent inhibitors misclassified as activators
**Status: VERIFIED ALREADY FIXED (v57/v58)**
- `_RE_INHIBIT` pattern includes `INACTIVAT` and is checked BEFORE `_RE_ACTIVATE`
- `_RE_ACTIVATE` uses `\bACTIVAT` (word boundary) so it cannot match inside `INACTIVATION`
- Tested: `INACTIVATION` → `'inhibits'`, `ACTIVATION` → `'activates'`

### Issue #2: ChEMBERTa 3 layers of silent fallback → training proceeds on random features
**Status: ROOT FIX APPLIED in v60**
- **Before**: `DRUGOS_STRICT_FEATURES` defaulted to `"0"` (off) — all 3 fallback layers silently fell back to random Xavier features
- **After**: `DRUGOS_STRICT_FEATURES` defaults to `"1"` (on) — all 3 fallback layers (DRUGOS_USE_CHEMBERTA=0, transformers not importable, HF_TOKEN missing/encode failure) now RAISE `FeatureFailureError`
- File: `phase2/drugos_graph/run_pipeline.py` line ~4570

### Issue #3: HGT training uses BCELoss on already-sigmoided scores → numerical instability (log(0) → -inf)
**Status: VERIFIED ALREADY FIXED (v57 P2C-004)**
- `score_triples()` returns LOGITS (unbounded), not sigmoided scores
- Training uses `BCEWithLogitsLoss()` which applies sigmoid internally via log-sum-exp trick
- Tested: BCEWithLogitsLoss produces finite loss (0.6582), training step completes with finite gradients

### Issue #4: HGT model NEVER SAVED when val_idx is empty (best_val_auc init to -1.0 means save guard `> 0.5` always fails)
**Status: ROOT FIX APPLIED in v60**
- **Before**: Save block gated on `if best_val_auc > 0.5:` — when val_idx empty, best_val_auc stays at -1.0, so `-1.0 > 0.5` is False, model NEVER saved
- **After**: ALWAYS saves with 3 tiers:
  - Tier 1: `best_val_auc > 0.5` → save best-val checkpoint (`save_reason="best_val_checkpoint"`)
  - Tier 2: `val_idx` empty → save last-epoch state (`save_reason="last_epoch_no_validation"`, `validation_performed=False`)
  - Tier 3: `best_val_auc <= 0.5` with val_idx → save for forensic inspection (`save_reason="last_epoch_validation_below_threshold"`, `validation_passed=False`)
- Added `validation_performed`, `validation_passed`, `save_reason` markers to saved checkpoint + return dict
- File: `phase2/drugos_graph/run_pipeline.py` lines ~6860-6970

### Issue #5: 3 of 7 CORE_NODE_TYPES (ClinicalOutcome, MedDRA_Term, Anatomy) have no canonical ID system
**Status: ROOT FIX APPLIED in v60** (v57 added CANONICAL_IDS entries; v60 makes loaders actually populate the fields)
- **SIDER loader**: `_build_node_record()` now populates `meddra_id`, `meddra_name`, `umls_cui` on MedDRA_Term node props (was missing)
- **GEO loader**: `GeoLoader.to_graph()` now emits Anatomy node records (was returning `[]`) with `uberon_id` field populated
- **phase1_bridge**: ClinicalOutcome node construction now populates `meddra_id` (None — requires MeSH→MedDRA crosswalk) and `mesh_id` (extracted from disease_id when it's a MeSH ID)
- Files: `sider_loader.py`, `geo_loader.py`, `phase1_bridge.py`

### Issue #6: ClinicalTrials 'Completed' status treated as positive evidence → negative-result trials become positive training signal
**Status: ROOT FIX APPLIED in v60** (v57 added `_classify_trial_confidence` but `primary_outcome_met` was always None in practice)
- **Before**: `primary_outcome_met` was read from `record.get("primary_outcome_met")` which was NEVER populated (AACT `primary_outcomes` table stores measure TEXT, not a boolean). So every Completed trial went through the None branch → emitted as positive `tested_for` edge with 0.4 evidence_strength
- **After**: Added SQL JOIN to AACT `outcome_analyses` table with `CASE WHEN EXISTS (...)` logic to populate `primary_outcome_met_raw` column ('met'/'not_met'/'partial'/''). Edge builder translates this to True/False/None
- Added `has_outcome_analyses` table detection with legacy fallback (older AACT schemas without the table get '' → None)
- `_classify_trial_confidence` now differentiates: Completed+True→0.9 (strong positive), Completed+False→0.1 (negative result), Completed+None→0.4 (unknown)
- File: `phase2/drugos_graph/clinicaltrials_loader.py` lines ~2210-2270 (SQL), ~3369-3410 (parsing)

### Issue #7: STITCH compound IDs CIDsm vs CIDs vs CIDf → 3-way split of same molecule
**Status: VERIFIED ALREADY FIXED (v57 P2L-038)**
- `_STITCH_CID_REGEX = re.compile(r'^(CID)?(sm|s|f|m)?(\d+)$', re.IGNORECASE)` handles all 5 prefix variants
- `_normalize_stitch_cid()` strips ANY prefix and returns bare numeric CID
- Tested: `CIDsm00002244`, `CIDs00002244`, `CIDf00002244`, `CIDm00002244`, `CID00002244` ALL normalize to `'2244'`

### Issue #8: OpenTargets association_score written into BOTH binding_confidence AND chembl_score
**Status: VERIFIED ALREADY FIXED (v57 P2L-045)**
- No assignments to `chembl_score` in opentargets_loader.py (only comments documenting the fix)
- OpenTargets writes to `binding_confidence`, `association_score`, `score`, `normalized_score` (NOT `chembl_score`)
- `chembl_score` field is reserved for ChEMBL pchembl values (0-14 scale) written by chembl_loader only

### Issue #9: Negative sampling samples from full node set (including positives) → ~30% of 'negatives' are actually positives
**Status: VERIFIED ALREADY FIXED (v36 Chain 9 + v53 P2-044)**
- `NegativeSampler` combines `positive_pairs + held_out_pairs` into `_rejection_pairs` for O(1) lookup
- `KGNegativeSampler` combines `known_triples + held_out_pairs` into `_rejection_set`, plus `_held_out_entities` exclusion (entity-level leakage prevention for inductive HGT)
- Tested: 13 negative samples generated, ZERO leakage (no known positives, no held-out pairs)

### Issue #10: Training data split uses random shuffling instead of edge-disjoint split → AUC inflated by 0.10+
**Status: VERIFIED ALREADY FIXED (v29 M-4/M-5)**
- Node-disjoint split is the FIRST option (requires ≥10 compounds in treats triples)
- Temporal split is the second option (uses approval_years)
- Random/stratified split is the LAST-RESORT fallback with explicit WARNING
- `temporal_split_pairs` RAISES `DrugOSDataError` when `approval_years=None` (no silent random fallback) unless `DRUGOS_ALLOW_TEMPORAL_RANDOM_FALLBACK=1`
- Tested: temporal_split_pairs correctly raises when approval_years=None

## Phase 1 ↔ Phase 2 Connection: 100% WIRED

The `phase1_bridge.py` module is the single authoritative contract connecting Phase 1 (data ingestion) to Phase 2 (knowledge graph):
- `read_phase1_outputs()` — reads Phase 1 data (PostgreSQL preferred, CSV fallback)
- `stage_phase1_to_phase2()` — converts DataFrames → Phase 2 node/edge dicts
- `load_into_graph()` — loads staged dicts into a graph builder
- `run_phase1_to_phase2()` — read → stage → load in one call
- `bridge_to_pyg_maps()` — converts bridge output to PyG-compatible `(entity_maps, edge_maps)`

`step1_load_phase1` in `run_pipeline.py` uses the bridge by default (`--data-source phase1`). When the bridge is used, steps 7a/7b/7c SKIP re-downloading STRING/UniProt/ChEMBL because the bridge already loaded them from Phase 1 CSVs.

## How to Verify

```bash
cd phase2
python tests/v60_root_fixes/test_v60_all_10_issues.py
```

Expected output:
```
PASS: Issue #1 — INACTIVATION correctly classified as 'inhibits'
PASS: Issue #2 — DRUGOS_STRICT_FEATURES defaults to '1' (ON)
PASS: Issue #3 — HGT uses BCEWithLogitsLoss on raw logits
PASS: Issue #4 — HGT model always saved (with validation markers)
PASS: Issue #5 — All 7 CORE_NODE_TYPES have canonical ID systems populated
PASS: Issue #6 — ClinicalTrials primary_outcome_met parsed from outcome_analyses
PASS: Issue #7 — STITCH CIDsm/CIDs/CIDf all normalize to canonical CID
PASS: Issue #8 — OpenTargets no longer writes to chembl_score
PASS: Issue #9 — 13 negative samples, zero leakage (no known positives, no held-out pairs)
PASS: Issue #10 — node-disjoint split is first option, random is last-resort
PASS: Integration — Phase 1 ↔ Phase 2 connection 100% wired via bridge

======================================================================
v60 ROOT FIX TEST RESULTS: 11 passed, 0 failed (of 11)
======================================================================
ALL TESTS PASSED — all 10 critical issues + integration verified.
```

## Dependencies Installed

- torch 2.13.0+cpu
- torch_geometric 2.8.0
- pandas 2.2.3
- numpy 2.1.3
- scikit-learn 1.5.2
- rdkit 2026.3.3
- pyyaml, requests, tqdm

## Files Modified in v60

1. `phase2/drugos_graph/run_pipeline.py` — Issues #2 (ChEMBERTa strict default) and #4 (HGT always save)
2. `phase2/drugos_graph/sider_loader.py` — Issue #5 (MedDRA_Term meddra_id field)
3. `phase2/drugos_graph/geo_loader.py` — Issue #5 (Anatomy node records with uberon_id)
4. `phase2/drugos_graph/clinicaltrials_loader.py` — Issue #6 (outcome_analyses SQL JOIN)
5. `phase2/drugos_graph/phase1_bridge.py` — Issue #5 (ClinicalOutcome meddra_id/mesh_id fields)

## Files Added in v60

1. `phase2/tests/v60_root_fixes/__init__.py`
2. `phase2/tests/v60_root_fixes/test_v60_all_10_issues.py` — 11 verification tests
