# shared/contracts — Contract-First Architecture

This directory contains the **canonical contracts** that define the
interfaces between the four phases of the Autonomous Drug Repurposing
Platform. Every cross-phase data exchange (CSV file, serialized graph,
model checkpoint, HTTP response) has a contract module here that both
the WRITER and the READER import.

## Why contracts exist

Before contracts, each phase defined its own schema INLINE:

- Phase 1 pipelines wrote CSVs with column names hardcoded in
  `phase1/pipelines/*.py`.
- Phase 2 bridge read those CSVs with column names hardcoded in
  `phase2/drugos_graph/phase1_bridge.py:_PHASE1_EXPECTED_COLUMNS`.
- Phase 3 trainer saved checkpoints with keys hardcoded in
  `graph_transformer/training/trainer.py:save_checkpoint()`.
- Phase 3 bridge wrote RL features with column names hardcoded in
  `graph_transformer/gt_rl_bridge.py`.
- Phase 4 env read RL features with column names hardcoded in
  `rl/rl_drug_ranker.py`.

When one phase changed a column name or a dict key, the other phase
silently broke. The bug only surfaced as wrong scientific output:
- A missing `gnn_score` column → Phase 4 reads 0.0 for every drug →
  the RL agent ranks by safety/market only → pharma partner gets
  scientifically wrong recommendations.
- A renamed checkpoint key `model_class_name` → Phase 4 can't
  reconstruct the model → silent fallback to a default architecture
  with wrong embedding dim → garbage predictions.

**A bug in a contract = wrong graph = wrong prediction = a pharma
partner tests the wrong drug on a real patient = patient harm.**

The contracts in this directory make schema changes a **compile-time
error on both sides** instead of a silent runtime corruption.

## Contract modules

### Cross-phase contracts (this directory)

| Module | Writer | Reader | What it defines |
|--------|--------|--------|-----------------|
| `urls.py` | Python services (FastAPI routes) | Frontend (Next.js API proxies) | Canonical URL paths for all 6 service endpoints |
| `feature_names.py` | Phase 3 bridge (`gt_rl_bridge.py`) | Phase 4 env (`rl/env.py`) | The 6 canonical RL feature names: `gnn_score`, `safety_score`, `market_score`, `efficacy_score`, `patent_score`, `adme_score` |
| `writeback.py` | Phase 4 (`rl/validate.py`) | Phase 3 trainer (`graph_transformer/training/trainer.py`) | Writeback CSV schema + writer/reader paths for the data flywheel |

### Per-phase contracts (in each phase's `contracts/` directory)

| Module | Writer | Reader | What it defines |
|--------|--------|--------|-----------------|
| `phase1/contracts/phase1_schema.py` | Phase 1 pipelines (CSV writers) | Phase 2 bridge (`phase1_bridge.py`) | Canonical CSV column names + dtypes for all 11 Phase 1 output files |
| `phase2/contracts/phase2_schema.py` | Phase 2 (`schema_mappings.py`) | Phase 3 (`phase2_adapter.py`) | Canonical node types, edge types, node/edge feature schemas |
| `phase2/contracts/kg_builder_contract.py` | Phase 2 (`RecordingGraphBuilder.save()`) | Phase 3 (`RecordingGraphBuilder.load()`) | Serialization format (JSON/Parquet) for the in-memory KG |
| `graph_transformer/contracts/phase3_schema.py` | Phase 3 trainer (`save_checkpoint()`) | Phase 3 service + Phase 4 env | Model checkpoint .pt file format (keys, value types) |
| `rl/contracts/phase4_schema.py` | Phase 4 (`rl/validate.py`) | Phase 3 trainer (retraining) | `validated_hypotheses.csv` format (columns, outcome enum) |
| `frontend/contracts/api_contracts.ts` | (mirrors Python services) | Frontend (all `src/lib/services/*.ts`) | TypeScript types for all HTTP response shapes |

## Contract-first workflow

When you need to change a cross-phase interface:

1. **Update the contract module first.** Edit the canonical definition
   in `shared/contracts/` (or the appropriate phase's `contracts/` dir).
2. **Run the contract consistency test.**
   ```bash
   python -c "from shared.tests.test_contract_consistency import test_all; test_all()"
   ```
   This will FAIL, listing every writer and reader that doesn't yet
   match the new contract.
3. **Update each writer and reader** to match the new contract. The
   test failure messages tell you exactly which files to edit.
4. **Re-run the test** until it passes.
5. **Commit the contract change + writer/reader updates in the same
   commit.** Never merge a contract change without the corresponding
   writer/reader updates — that would break CI for every other developer.

## Contract consistency test (Task 330)

Location: `shared/tests/test_contract_consistency.py`

This test verifies:

1. **Phase 1 schema is imported by Phase 2 bridge.** The bridge's
   `_PHASE1_EXPECTED_COLUMNS` dict must be DERIVED from
   `phase1.contracts.phase1_schema.PHASE1_OUTPUT_SCHEMA` (not
   independently hardcoded).
2. **Phase 2 schema is the single source for node types.** The
   `pyg_builder._PHASE2_TO_GT_NODE_TYPE` and
   `phase2_adapter.PHASE2_TO_PHASE3_NODE` dicts must both derive from
   `phase2.contracts.phase2_schema.NODE_TYPES` (no independent copies).
3. **Phase 3 checkpoint keys match the contract.** The trainer's
   `save_checkpoint()` must produce a dict whose keys exactly match
   `graph_transformer.contracts.phase3_schema.CHECKPOINT_REQUIRED_KEYS`.
4. **Phase 4 CSV schema matches the contract.** The writer's
   `OUTPUT_SCHEMA` must match
   `rl.contracts.phase4_schema.VALIDATED_HYPOTHESES_COLUMNS`.
5. **RL feature names match the contract.** Phase 3 bridge writes
   exactly the 6 features in
   `shared.contracts.feature_names.CANONICAL_RL_FEATURE_NAMES`, and
   Phase 4 env reads the same 6.
6. **Service URLs match the contract.** Each Python service registers
   exactly the URLs declared in `shared.contracts.urls.ALL_SERVICE_URLS`.
7. **Frontend TypeScript contracts match Python.** The
   `frontend/contracts/api_contracts.ts` URL constants match
   `shared.contracts.urls` (a static-string check, since we can't run
   TypeScript from Python CI).

## CI integration (Task 332)

Add this step to `.github/workflows/ci.yml`:

```yaml
- name: Contract consistency check
  run: |
    python -c "from shared.tests.test_contract_consistency import test_all; test_all()"
```

This MUST pass before any PR can merge to `main`.

## Anti-patterns to avoid

1. **Don't define schema locally.** If you find yourself writing
   `EXPECTED_COLUMNS = [...]` or `_NODE_TYPE_MAP = {...}` in a non-
   contract module, stop. Move it to a contract module and import it.
2. **Don't bypass the contract with `.get()`.** If a reader uses
   `checkpoint.get("model_class_name", "DefaultModel")`, the contract
   is meaningless — a writer removing the key won't trigger an error.
   Use direct indexing: `checkpoint["model_class_name"]`.
3. **Don't add optional keys without updating the contract.** If you
   add a new optional key to a checkpoint, add it to
   `CHECKPOINT_OPTIONAL_KEYS` in the contract. The contract test
   verifies ALL keys (required + optional) are documented.
4. **Don't change a contract without updating all readers.** The
   contract test will catch this, but only if you RUN it before merging.
   Always run `python -c "from shared.tests.test_contract_consistency import test_all; test_all()"`
   locally before pushing.

## Workshop (Task 334)

A workshop covering all 19 Team Members (TMs) should be run to explain:
- Where each contract module lives.
- How to import from a contract module (vs. defining a local copy).
- How to run the contract consistency test locally.
- What to do when the contract test fails (update the contract first,
  then update the writers/readers).

Workshop slides should be recorded and added to the project wiki.
