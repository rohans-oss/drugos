# v63 ROOT FIX SUMMARY

## Status: ALL 18 P0 ISSUES VERIFIED FIXED AT RUNTIME (40/40 checks pass)

## What was done in v63

### Verified Already-Fixed in v62 (16 issues)
All 16 issues below were already fixed in v62 source code. v63 verifies each at RUNTIME (not just reading comments) via `verify_v63_fixes.py`:

1. **T-001** — Migration 001 FK ordering: pipeline_runs created at line 134, before all child tables
2. **T-002** — Vioxx withdrawn: migration 006 backfills from curated FDA-withdrawn name list; migration 008 excludes withdrawn from globally_approved
3. **T-003** — Migration 009 InChIKey: uses POSIX regex `~ '^[A-Z]{14}-[A-Z]{10}-[A-Z]$'`
4. **P2L-008** — ChEMBL regex: `\bACTIVAT` word boundary; INACTIVAT in _RE_INHIBIT checked FIRST
5. **P2L-041** — ClinicalTrials: primary_outcome_met filter via outcome_analyses table
6. **P2L-045** — OpenTargets: association_score NOT aliased to chembl_score
7. **P1C-001** — GDA gene_symbol: VARCHAR(50) nullable, no DEFAULT '', no CHECK
8. **P2C-002+007** — CANONICAL_IDS for ClinicalOutcome/MedDRA_Term/Anatomy + reverse-check
9. **P2C-004+005+009** — HGT: BCEWithLogitsLoss, skip unknown decoder keys, val_auc init -1.0
10. **P1-013** — ChEMBL v50 filename synced: chembl_activities.csv.gz
11. **P1-002+003** — Embedded samples: IC50 (not Potency), causal (not causative)
12. **P1C-003** — .env.example: STRING_MIN_COMBINED_SCORE=700
13. **P1C-002** — Test protein rejection: DRUGOS_ENVIRONMENT=prod default
14. **P2L-032** — STRING threshold centralized in config (700)
15. **P2C-001** — total_nodes includes pathway_nodes
16. **P2C-008** — Phase1_bridge: ERROR log on schema_missing, prod fatality
17. **T-004** — Migration 002: partial unique index WHERE gene_symbol IS NOT NULL
18. **P2L-021** — DRKG: lowercase canonical case for relation codes
19. **P2L-038** — STITCH: CIDm vs CIDs distinction preserved

### Genuinely Fixed in v63 (P2C-003+016 — ChEMBERTa silent-disable cascade)
The v62 fix added DRUGOS_STRICT_FEATURES=1 default and FeatureFailureError, but was missing 3 pieces the audit required:

1. **--no-chemberta CLI flag** (run_unified.py): dev opt-out that sets DRUGOS_USE_CHEMBERTA=0 and DRUGOS_STRICT_FEATURES=0. REFUSED in production (DRUGOS_ENVIRONMENT=prod) with exit code 1.

2. **MLflow CHEMBERTA_DISABLED=true tagging** (mlflow_tracker.py + run_pipeline.py): added `set_tag()` method to MLflowTracker. All 4 step9 failure paths (disabled_by_env, transformers_not_importable, hf_token_missing, no_drug_records) now log CHEMBERTA_DISABLED=true, CHEMBERTA_FAILURE_REASON, FEATURE_FALLBACK, MOLECULAR_STRUCTURE_LEARNED tags to MLflow.

3. **Model-save refusal in prod** (run_pipeline.py step11b): added `chemberta_disabled` parameter. When ChEMBERTa was disabled AND DRUGOS_ENVIRONMENT=prod, step11b REFUSES to save the HGT model (returns model_saved=False with model_save_refused_reason="chemberta_disabled_in_production"). This prevents audit theater where a model trained on random Xavier features is reported as "saved".

### Runtime Verification
- `verify_v63_fixes.py`: 40/40 checks PASS
- Dev mode with --no-chemberta: pipeline completes, MLflow tagged
- Prod mode with --no-chemberta: REFUSED (exit 1)
- Prod mode without HF_TOKEN: FeatureFailureError at step9, pipeline ABORTS (exit 5)
- Phase 1 ORM creates 12 tables with correct columns
- Phase 1↔Phase 2 bridge reads Phase 1 CSVs, stages nodes/edges for KG
