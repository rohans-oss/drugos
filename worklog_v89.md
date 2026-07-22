# v89 P0 Forensic Root Fixes — Work Log

---
Task ID: v89-p0-forensic-root-fixes
Agent: main (Sonnet)
Task: Pull the autonomous-drug-repurposing repo, read each actual source
file line-by-line, fix the P0 bugs and compound bug chains listed in the
user's audit, install deps, run real code, push to a feature branch,
verify CI/build/tests pass, then merge to main.

Work Log:
- Cloned repo (MANOFHATERS/autonomous-drug-repurposing) to /home/z/my-project/adr
- Read project docx (Team_Cosmic_Build_Process_Updated.docx) — 4-phase
  autonomous drug repurposing platform (Phase 1 data ingestion, Phase 2
  KG build, Phase 3 Graph Transformer, Phase 4 RL ranker)
- Read actual source files at the specific bug locations mentioned in
  the user's audit (NOT tests, NOT comments — actual code):
  - graph_transformer/data/graph_builder.py (1027 lines)
  - rl/rl_drug_ranker.py (4981+ lines)
  - phase2/drugos_graph/phase1_bridge.py (5553 lines)
  - phase1/pipelines/omim_pipeline.py (3361 lines)
  - phase1/cleaning/missing_values.py (3574 lines)
  - graph_transformer/gt_rl_bridge.py (3022 lines)
  - phase1/config/settings.py (3964 lines)
  - phase2/drugos_graph/config.py (7954 lines)
- Created feature branch: fix/v89-p0-forensic-root-fixes

Confirmed P0 bugs (the user's audit is CORRECT):
1. graph_builder.py 3-hop path injection (L869-878, L956-970) — for
   every KP AND training positive, a guaranteed drug→protein→pathway→
   disease path was injected. LABEL_LEAKING_EDGES only strips the
   direct treats edge, not the injected path → GT model learns
   "3-hop path exists → positive" trivially → val AUC = 1.0 (fraud).
2. hash() non-determinism (L870, L959) — PYTHONHASHSEED makes
   hash("aspirin") differ per process → non-reproducible graphs.
3. rl_drug_ranker.py VecNormalize bypass (L2767, L2886, L3338) — raw
   obs passed to model.policy.obs_to_tensor without normalization.
4. rl_drug_ranker.py compute_auc (L3341) — defensive off-by-one fix
   (capture row_idx explicitly before extract_policy_prob_high).
5. phase1_bridge.py covalent-inhibitor misclassification (L2599) —
   "activ" substring matches "Inactivation" → classified as "activates".
6. phase1_bridge.py organism filter missing (L1830) — no
   Protein.ncbi_taxid == 9606 filter on ChEMBL query.
7. omim_pipeline.py vs missing_values.py score inversion —
   pipeline: mk=3→0.9, mk=4→0.8 (CORRECT); validator: mk=3→0.8,
   mk=4→0.9 (INVERTED).
8. gt_rl_bridge.py gate fires AFTER CSV write — fixed by (a) making
   RL pipeline gate use 0.85 threshold (matches bridge V1 contract)
   and (b) deleting candidate CSVs on bridge gate failure.
9. gnn_score reward — weight capped at 0.04 (was 0.20) + multiplicative
   gnn_factor gate REMOVED (RL agent was circular distillation of GT).
10. DRUGOS_ENVIRONMENT default = "dev" → changed to "production".
11. validated_hypotheses.csv — contained (aspirin, cardiovascular
    disease) and (metformin, type 2 diabetes) which are in
    KNOWN_POSITIVES → disjointness check rejected them → +0.1 bonus
    never applied → flywheel fiction. Replaced with REAL pharma-
    validated pairs NOT in KNOWN_POSITIVES (thalidomide/MM,
    sildenafil/PAH, mifepristone/Cushing, topiramate/migraine).
12. _is_rare_disease — used hardcoded frozenset that marked Parkinson's
    (~1M US), MS (~400K), Alzheimer's (~6.7M), migraine (~39M), etc.
    as "rare". Rewrote to use REAL US prevalence data (GARD/NIH/
    Orphanet) with FDA Orphan Drug Act threshold (<200K = rare).

Fixes applied (manually, NOT via scripts):
- graph_builder.py: removed BOTH 3-hop path injection blocks (KP +
  training positive); added _deterministic_seed helper using
  hashlib.sha256 to replace hash() for reproducibility.
- rl_drug_ranker.py: extract_policy_prob_high now accepts vec_normalize
  param and normalizes obs before policy network; train_agent returns
  3-tuple (model, checkpoint_path, vec_normalize); evaluate_agent +
  compute_auc pass vec_normalize through; compute_auc captures
  current_row_idx explicitly before extract_policy_prob_high
  (defensive off-by-one fix); gnn_score weight capped at 0.04 + gnn_factor
  gate REMOVED; _is_rare_disease rewritten with US_PREVALENCE table;
  PipelineConfig gained gt_test_auc_threshold (default 0.85) +
  rl_auc_threshold (default 0.5); scientific_validation gate uses
  configurable thresholds.
- gt_rl_bridge.py: bridge loads VecNormalize stats from
  .vecnormalize.pkl alongside PPO checkpoint; on scientific_validation
  failure, DELETES candidate CSVs + intermediate gt_predictions.csv;
  run_full_pipeline gained graph_data parameter for REAL Phase 2
  HeteroData integration (skips build_demo_graph when provided).
- phase1_bridge.py: covalent-inhibitor classification uses word-
  boundary regex (\b(inactiv|deactiv|inhibit|antagon) → inhibits;
  \b(activ|agon) → activates); ChEMBL activities query filtered to
  Protein.ncbi_taxid == 9606 (human only).
- missing_values.py: OMIM score map aligned with pipeline
  ({1: 0.5, 2: 0.6, 3: 0.9, 4: 0.8} — was inverted for mk=3/mk=4).
- settings.py + config.py: DRUGOS_ENVIRONMENT default changed from
  "dev" to "production" (5 occurrences in config.py + 1 in settings.py).
- validated_hypotheses.csv: replaced with 4 REAL pharma-validated
  repurposing pairs NOT in KNOWN_POSITIVES.
- run_pipeline.py: NEW top-level entry point that chains Phase 1 →
  Bridge → Phase 2 kg_builder → Phase 3 GT trainer (REAL HeteroData)
  → Phase 4 RL ranker.

Stage Summary:
- 8 files modified, 1 file created (run_pipeline.py)
- 770 insertions, 226 deletions in first commit
- All P0 bugs from the user's audit addressed with root-cause fixes
  (NOT surface-level patches)
- Phase 1-4 integration now possible via run_pipeline.py + bridge's
  new graph_data parameter
- Next: install deps, run real code end-to-end, push branch, verify
  CI/build/tests, merge to main
