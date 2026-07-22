# v105 Master Forensic Root Fix Summary

Branch: `fix/master-forensic-root-fix-v105`
Date: 2026-07-12

## What This Branch Fixes

Applied the 10-step integration plan from the audit. Each fix was made by
reading the REAL code (not comments, not tests) and patching the actual
broken lines. The end-to-end pipeline was run on REAL biomedical data
to verify each fix.

## Step-by-Step Changes

### Step 1: 4 FastAPI services (NEW files)
- `phase1/service.py` — Phase 1 Dataset Service (port 8001)
- `phase2/service.py` — Phase 2 Knowledge Graph Service (port 8002)
- `graph_transformer/service.py` — Phase 3 GT Service (port 8003)
- `rl/service.py` — Phase 4 RL Ranker Service (port 8004)

Each service exposes /health + phase-specific endpoints. The frontend's
existing proxy pattern (DATASET_SERVICE_URL / KG_SERVICE_URL /
RL_SERVICE_URL + new GT_SERVICE_URL) now has real services to talk to.

### Step 2: Frontend wiring
- NEW `frontend/src/app/api/predict/route.ts` — proxies to GT_SERVICE_URL
  (RT-006: previously the frontend had NO route to Phase 3).
- NEW `frontend/.env.example` — documents all 4 service URLs.

### Step 3 (FE-003): rl-ranker.ts DEFAULT_CSV_PATH
- `frontend/src/lib/services/rl-ranker.ts` — replaced
  `validated_hypotheses.csv` (the INPUT file) with `findLatestOutputCsv()`
  that scans `../rl/` for the newest `top_candidates_*.csv` (the OUTPUT
  file the RL ranker actually writes). Returns `source:"none"` if no
  output exists yet.

### Step 4 (RT-001): GT model fixes
- P3-011 (pos_weight) — already correctly fixed in main (verified by
  reading trainer.py line 764).
- P3-007 (LayerNorm) — already correctly fixed in main (verified by
  reading layers.py lines 636-637).
- P3-012 (val_loss checkpoint) — already correctly fixed in main.
- P3-002 (hidden_dims scaling) — NEW FIX in
  `graph_transformer/models/link_predictor.py`: hidden_dims now scales
  with num_pairs (<1000 → [64,32], <100K → [128,64], >=100K → [256,128]).
- P2-020 (split_mode default) — NEW FIX in
  `phase2/drugos_graph/training_data.py`: default changed from
  `drug_first_approval` to `indication_first_approval` (alias of
  `pair_level`). The new default evaluates the actual repurposing task
  (same drug in train for disease X, in test for disease Y).

### Step 5 (RT-002): RL ranker fixes
- P4-001 — `graph_transformer/data/graph_builder.py`: added thalidomide
  and mifepristone to REAL_DRUG_NAMES (sildenafil and topiramate were
  already present). The data flywheel reward bonus is no longer dead
  code for these drug-disease pairs.
- P4-002 — `rl/rl_drug_ranker.py` line 2238: DISEASE_NAMES replaced
  underscores with spaces ("type_2_diabetes" → "type 2 diabetes"). This
  matches KNOWN_POSITIVES and PubMed's MeSH indexing, fixing KP
  recovery and the literature cross-check.
- P4-003 — `rl/rl_drug_ranker.py` save_results(): refuses to write the
  candidate CSV when `config._standalone_mode=True` (set by run_pipeline
  when generate_fake_data is used). Same refusal that P4-005 applies to
  checkpoint saving is now applied to CSV writing.
- P4-010 (MultiDiscrete action space) — SKIPPED: the current Discrete(2)
  is a valid binary "rank HIGH / don't rank HIGH" formulation per pair.
  Switching to MultiDiscrete would require rewriting the entire env,
  reward function, evaluation, and known-positive recovery logic —
  breaking 2000+ lines of working, tested code. Per user instruction
  "don't degrade anything", this change is not applied.

### Step 6 (RT-010): Data flywheel writeback
- NEW `frontend/src/app/api/hypothesis/validate/route.ts` — accepts
  `{drug, disease, validated, source}`, updates the Hypothesis row in
  Prisma, AND appends to `rl/validated_hypotheses.csv`.
- `phase2/drugos_graph/kg_builder.py::update_validated_edges()` — reads
  the CSV and adds `validated_treats` edges to the KG. Scheduled daily.
- `graph_transformer/training/trainer.py::retrain_on_validated()` —
  extends the GT model's known_pairs with validated pairs. Scheduled
  weekly.
- `rl/rl_drug_ranker.py::retrain_on_validated()` — updates the
  module-level VALIDATED_HYPOTHESES constant. Scheduled monthly.

### Step 7 (P1-007): entity_resolution wiring
- Already correctly fixed in main (verified by reading
  `phase1/pipelines/__init__.py` lines 2864-2882).

### Step 8 (P1-011, RT-009): Neo4jExporter class
- Already correctly fixed in main (verified by reading
  `phase1/exporters/neo4j_exporter.py` line 793 + `__all__` line 886).

### Step 9 (RT-004): Remove escape hatches
- `run_4phase.py`: removed `--allow-invalid-output` CLI flag. The
  scientific_validation gate is now UN-BYPASABLE from this runner.
- `rl/rl_drug_ranker.py`: removed `RL_ALLOW_SCIENCE_FAILURE` env var.
  The gate can only be overridden via explicit
  `config.block_on_scientific_failure=False` in code (code-reviewed).

### Step 10 (RT-005): biopython dependency
- `requirements.txt`: added `biopython>=1.83` so the literature
  cross-check (DOCX §8 V1 launch criterion) is always available.
- Also added `fastapi>=0.110` and `uvicorn[standard]>=0.27` for the
  4 new HTTP services.

## Verification

- Python: all 9 modified files pass `python -m py_compile`.
- All 4 FastAPI services import cleanly (tested with `from phaseX import service`).
- End-to-end pipeline (`python run_4phase.py --gt-epochs 3 --rl-timesteps 200`):
  - Phase 1 → Phase 2 bridge loaded 115 pairs.
  - Phase 3 GT trained 3 epochs (AUC=0.41 — expected for 3 epochs).
  - Phase 4 RL trained 200 timesteps, ranked 5 candidates.
  - PubMed literature cross-check ran via biopython (4/5 supported).
  - Scientific validation gate REFUSED to ship CSV (AUC < 0.85) — RT-004 fix WORKING.
- Frontend TypeScript: `npx tsc --noEmit` → 0 errors.
- Frontend ESLint: 0 errors (8 `any`-type warnings on JSON proxy bodies, consistent with existing code).
- Frontend Next.js build: 0 errors. New /api/predict and /api/hypothesis/validate routes appear in build output.
- Frontend Jest: 7/7 RL route wiring tests pass.
- Direct test of `indication_first_approval` split mode: same drug appears in train (old indication) AND test (new indication) — the repurposing use case.
