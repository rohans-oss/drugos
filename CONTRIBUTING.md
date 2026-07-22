# Contributing to Autonomous Drug Repurposing Platform

Thank you for contributing! This guide explains how to make safe, reviewable changes.

---

## Table of Contents
1. [Development Setup](#development-setup)
2. [How to Update Shared Contracts](#how-to-update-shared-contracts)
3. [How to Add a New Phase 1 Data Source](#how-to-add-a-new-phase-1-data-source)
4. [How to Update the RL Reward Function](#how-to-update-the-rl-reward-function)
5. [PR Checklist](#pr-checklist)
6. [Commit Message Format](#commit-message-format)

---

## Development Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd autonomous-drug-repurposing-main

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Set up environment
cp .env.example .env
# Edit .env — fill in POSTGRES_PASSWORD, NEO4J_PASSWORD, MLFLOW_ADMIN_PASSWORD,
# AIRFLOW__CORE__FERNET_KEY, JWT_SECRET

# 5. Install frontend deps
cd frontend && npm install && cd ..
```

---

## How to Update Shared Contracts

All cross-phase data schemas live in `shared/contracts/`. **Never duplicate them.**

### Python side

1. Edit the relevant file in `shared/contracts/` (e.g., `writeback.py`, `urls.py`)
2. Run the contract consistency tests:
   ```bash
   pytest tests/test_cross_phase_imports.py -v
   ```
3. Update the TypeScript mirror in `frontend/src/lib/ml-contracts.ts` to match
4. Verify no `ClinicalOutcomes` (plural) typo was introduced:
   ```bash
   grep -r "ClinicalOutcomes" frontend/src/   # must return 0
   ```

### TypeScript side

1. Edit `frontend/src/lib/ml-contracts.ts`
2. Ensure the Zod schema field names exactly match the Python Pydantic models in `shared/contracts/`
3. Run frontend build: `cd frontend && npm run build`

> **Rule:** If a field exists in Python `ValidatedHypothesis`, it MUST exist in the TypeScript `ValidateRequest` Zod schema with the same name and type.

---

## How to Add a New Phase 1 Data Source

1. Create `phase1/pipelines/<source>_pipeline.py` following the `BasePipeline` interface
2. Add the source to `phase1/pipelines/__init__.py`
3. Add the output CSV filename to `phase1_bridge.py`'s `paths` dict in `phase2/drugos_graph/phase1_bridge.py`
4. Add the source to `csv_map` in `phase1/service.py:_load_dataset_stats()`
5. Add an Airflow DAG in `phase1/dags/`
6. Add the CSV name to the docker-compose volume mount if needed
7. Write tests in `phase1/tests/`

> **Important:** Never write mock/sample data to production paths. Use `DRUGOS_ENVIRONMENT=development` for dev-only sample data.

---

## How to Update the RL Reward Function

The RL reward function is in `rl/rl_drug_ranker.py :: RewardConfig` and `RewardFunction`.

Before changing reward weights:
1. Read the **EV analysis** in `FIX_LOG.md` (ROOT v2 FIX 1) — changing weights can collapse the agent to "always LOW"
2. Run the full pipeline on the demo graph and verify:
   - `RL AUC > 0.55` (agent is learning)
   - Top-10 contains `> 3` distinct drugs (no collapse to a single drug)
   - KP recovery rate `> 0` on the test split
3. Update the `DATA_DICTIONARY` docstring in `rl_drug_ranker.py` if semantics change
4. Update the `.meta.json` sidecar format if new fields are added to the reward config

```bash
# Quick smoke test after reward changes:
python run_4phase.py --gt-epochs 5 --rl-timesteps 500 --seed 42
```

---

## PR Checklist

Before opening a PR, verify:

- [ ] `pytest tests/test_cross_phase_imports.py -v` — all 7 cross-phase tests pass
- [ ] `cd frontend && npm run build` — frontend builds without errors
- [ ] `grep -r "Math.random()" frontend/src/` — returns 0 results
- [ ] `grep -r "ClinicalOutcomes" frontend/src/` — returns 0 results
- [ ] No hardcoded passwords/secrets in any file (`grep -r "drugos_dev_password" .`)
- [ ] No new `Math.random()` in scientific computation paths
- [ ] `.env` is NOT committed (check `git status`)
- [ ] All new modules have unit tests
- [ ] Docstrings updated for any changed function signatures
- [ ] `MANIFEST.in` updated if new data file types were added

---

## Commit Message Format

```
<type>(<scope>): <short description>

<body — explain WHY, not WHAT>

Fixes: <issue-id>
```

**Types:** `fix`, `feat`, `refactor`, `test`, `docs`, `ci`, `chore`

**Scopes:** `phase1`, `phase2`, `graph_transformer`, `rl`, `frontend`, `shared`, `docker`, `ci`

**Examples:**
```
fix(rl): enforce vecnormalize sidecar at inference

Without the sidecar the policy receives raw (un-normalized) observations
causing a silent train/inference distribution shift. Now raises RuntimeError
when the .vecnormalize.pkl file is missing (P4-004 ROOT FIX).

Fixes: P4-004
```

```
feat(phase1): add SIDER adverse-event pipeline

Wires SIDER adverse-event CSV through the phase1_bridge so the KG's
causes_adverse_event edges are populated from Phase 1 data directly.

Fixes: P2-047
```
