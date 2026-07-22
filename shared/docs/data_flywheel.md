# The Data Flywheel

> **DOCX §10**: "The most critical aspect of the build strategy is not
> just building V1 — it is building the self-reinforcing data flywheel
> that makes the platform defensible over time."

This document is the canonical reference for the data flywheel: the
self-reinforcing loop that turns pharma-partner validations into a
proprietary training signal that competitors cannot replicate.

## 1. Sequence Diagram

The flywheel has 4 steps. Each step is implemented by a specific module
and governed by a specific contract.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA FLYWHEEL — 4 STEPS                          │
└─────────────────────────────────────────────────────────────────────────┘

  Pharma Partner        Phase 4           Phase 1            Phase 3         Phase 4
  (wet lab /        writeback.py       validated_         trainer.py        rl_drug_ranker.py
   clinical study)                     hypotheses.csv
       │                   │                │                   │                │
       │  1. POST           │                │                   │                │
       │  /api/hypothesis/  │                │                   │                │
       │  validate          │                │                   │                │
       │ ──────────────────>│                │                   │                │
       │                   │                │                   │                │
       │                   │ 2. write_       │                   │                │
       │                   │ validated_      │                   │                │
       │                   │ hypothesis()    │                   │                │
       │                   │ ──────────────>│                   │                │
       │                   │                │ (CSV append,      │                │
       │                   │                │  atomic,          │                │
       │                   │                │  idempotent)      │                │
       │                   │                │                   │                │
       │                   │ 3. writeback_  │                   │                │
       │                   │ to_phase3()    │                   │                │
       │                   │ ──────────────────────────────────>│                │
       │                   │                │                   │ (JSON trigger  │
       │                   │                │                   │  append,       │
       │                   │                │                   │  atomic)       │
       │                   │                │                   │                │
       │                   │                │                   │ 4. Airflow     │
       │                   │                │                   │ schedule:      │
       │                   │                │                   │ load_validated_│
       │                   │                │                   │ for_retraining │
       │                   │                │                   │ (reads JSON    │
       │                   │                │                   │  trigger,      │
       │                   │                │                   │  writes temp   │
       │                   │                │                   │  CSV with      │
       │                   │                │                   │  outcome col,  │
       │                   │                │                   │  calls         │
       │                   │                │                   │  retrain_on_   │
       │                   │                │                   │  validated)    │
       │                   │                │                   │                │
       │                   │                │                   │ 5. trainer.    │
       │                   │                │                   │ fit(N epochs)  │
       │                   │                │                   │ on new pairs   │
       │                   │                │                   │ (atomic ckpt   │
       │                   │                │                   │  save)         │
       │                   │                │                   │                │
       │                   │                │                   │ 6. New         │
       │                   │                │                   │ checkpoint     │
       │                   │                │                   │ saved with     │
       │                   │                │                   │ extended       │
       │                   │                │                   │ known_pairs    │
       │                   │                │                   │ and fine-tuned │
       │                   │                │                   │ model weights  │
       │                   │                │                   │                │
       │                   │                │                   │ 7. RL ranker   │
       │                   │                │                   │ reads          │
       │                   │                │                   │ validated_     │
       │                   │                │                   │ hypotheses.csv │
       │                   │                │ <─────────────────│                │
       │                   │                │                   │ ──────────────>│
       │                   │                │                   │                │
       │                   │                │                   │                │ 8. _load_
       │                   │                │                   │                │ validated_
       │                   │                │                   │                │ hypotheses()
       │                   │                │                   │                │ loads +0.1
       │                   │                │                   │                │ bonus set
       │                   │                │                   │                │
       │                   │                │                   │                │ 9. _load_
       │                   │                │                   │                │ validated_
       │                   │                │                   │                │ toxic_
       │                   │                │                   │                │ hypotheses()
       │                   │                │                   │                │ loads -0.5
       │                   │                │                   │                │ penalty set
       │                   │                │                   │                │
       │                   │                │                   │                │ 10. step()
       │                   │                │                   │                │ applies
       │                   │                │                   │                │ bonus/
       │                   │                │                   │                │ penalty
       │                   │                │                   │                │ AFTER
       │                   │                │                   │                │ high_action_
       │                   │                │                   │                │ bonus
       │                   │                │                   │                │ multiplier
       │                   │                │                   │                │
       │                   │                │                   │                │ 11. RL agent
       │                   │                │                   │                │ learns to
       │                   │                │                   │                │ rank
       │                   │                │                   │                │ validated
       │                   │                │                   │                │ positives
       │                   │                │                   │                │ HIGH and
       │                   │                │                   │                │ validated
       │                   │                │                   │                │ toxics LOW
       │                   │                │                   │                │
       │  12. Better        │                │                   │                │
       │  predictions       │                │                   │                │
       │ <──────────────────│                │                   │                │
       │  attract more      │                │                   │                │
       │  partners, who     │                │                   │                │
       │  produce more      │                │                   │                │
       │  validations,      │                │                   │                │
       │  repeat.           │                │                   │                │
       │                   │                │                   │                │
```

## 2. Contract Modules (single source of truth)

All 4 phases import from the SAME canonical contracts. No module may
hardcode its own path, column name, outcome value, edge label, or
feature name.

| Contract | Location | Purpose |
|----------|----------|---------|
| `shared.contracts.writeback` | `shared/contracts/writeback.py` | Canonical path, CSV columns, outcome enum, edge labels, Neo4j node labels, atomic-write profile. |
| `shared.contracts.feature_names` | `shared/contracts/feature_names.py` | Canonical 17-column RL feature schema (bridge → RL env). |
| `common.validated_hypotheses_schema` | `common/validated_hypotheses_schema.py` | DEPRECATED — re-exports from `shared.contracts.writeback` for backward compat. |

### 2.1 Canonical path (issue #336)

```
phase1/processed_data/validated_hypotheses.csv
```

Both `phase4/writeback.py` AND `graph_transformer/training/trainer.py`
AND `rl/rl_drug_ranker.py` read/write this EXACT path. The legacy
`rl/validated_hypotheses.csv` default has been REMOVED from the trainer.

### 2.2 Canonical column (issue #337)

```
outcome   (column name)
```

Both the writeback (writer) and the trainer (reader) use the `outcome`
column. The previous `validated` column with `"true"`/`"false"` values
has been REMOVED — it was the bug that broke flywheel Step 2→3.

### 2.3 Canonical outcome enum (issue #340)

| Outcome string | Meaning | Trainer behavior | RL ranker behavior |
|----------------|---------|------------------|--------------------|
| `validated_positive` | Wet lab confirmed efficacy | Add to known_pairs as positive label | +0.1 bonus when ranked HIGH |
| `validated_toxic` | Caused adverse events | EXCLUDE from positive labels (safe) | -0.5 penalty (FLAT OVERRIDE) when ranked HIGH |
| `validated_negative` | Confirmed NO efficacy | EXCLUDE from positive labels | (no bonus, no penalty) |
| `invalidated` | Partner could not reproduce | EXCLUDE from positive labels | (no bonus, no penalty) |

### 2.4 Canonical edge labels (issue #342)

| Outcome | Neo4j edge label |
|---------|------------------|
| `validated_positive` | `VALIDATED_TREATS` |
| `validated_toxic` | `VALIDATED_TOXIC_FOR` |
| `validated_negative` | `VALIDATED_NEGATIVE_FOR` |
| `invalidated` | `VALIDATED_NEGATIVE_FOR` |

The `FOR` suffix on `VALIDATED_TOXIC_FOR` makes the semantics explicit:
the drug is toxic FOR this disease (a drug→disease relationship), not
just "toxic" in general (which would be a drug property, not an edge).

### 2.5 Canonical Neo4j node labels (issue #341)

The TM 17 contract specifies `:Drug` with canonical `drug_id`. The
current Phase 2 kg_builder uses `:Compound` with `name`. The writeback
defensively MERGEs against BOTH labels so it works regardless of which
schema the KG is in. This prevents node fragmentation (the bug where
MERGE on `:Drug` would create a duplicate of an existing `:Compound`
node).

Preferred order:
1. `:Drug` (TM 17 contract) — try first
2. `:Compound` (current KG) — fall back if `:Drug` finds 0 nodes

### 2.6 Canonical 17-column RL feature schema (issue #344)

The bridge produces 17 columns; the RL env consumes them by name.
See `shared/contracts/feature_names.py` for the full list.

| Category | Columns |
|----------|---------|
| Identity (2) | `drug`, `disease` |
| GT output (4) | `gnn_score`, `gnn_score_calibrated`, `confidence`, `gnn_score_timestamp` |
| Drug-level (4) | `safety_score`, `patent_score`, `adme_score`, `efficacy_score` |
| Disease-level (3) | `disease_pair_count`, `disease_avg_gnn`, `disease_avg_safety` |
| Supplementary (4) | `market_score`, `pathway_score`, `rare_disease_flag`, `unmet_need_score` |

### 2.7 Atomic-write profile (issue #351)

Both the CSV writeback AND the checkpoint save MUST use the atomic-write
pattern:

1. Write to a temp file in the SAME directory as the target (so
   `os.rename` is a single inode operation — atomic on POSIX).
2. `fsync` the temp file (so the bytes are durable on disk before the
   rename, surviving power loss).
3. `os.replace(temp, target)` — atomic rename.

If the write fails, the temp file is cleaned up and the original file
is UNTOUCHED. The next run always sees a complete, valid file.

## 3. Reward Math (issues #348, #340, #350)

The RL ranker's reward function applies bonuses/penalties AFTER the
`high_action_bonus` multiplier, NOT before. This prevents the bug where
`validated_bonus` (0.1) was amplified 5x by `high_action_bonus` (5.0)
into an effective 0.5 bonus.

```
final_reward = float(reward) * cfg.high_action_bonus    # base * multiplier
if action == 1 and row._is_validated:                   # +bonus AFTER mult
    final_reward += cfg.validated_bonus                 # +0.1 (exact)
if action == 1 and row._is_validated_toxic:             # -penalty AFTER mult
    final_reward = -abs(cfg.validated_toxic_penalty)    # -0.5 (FLAT OVERRIDE)
```

The toxic penalty is a FLAT OVERRIDE (not a subtraction) to GUARANTEE
the reward is negative even if the base reward is high (e.g.,
`gnn_score=0.95 → base * 5.0 = 4.75 → -0.5 = 4.25 (still positive!)`).
The flat override ensures the agent NEVER learns to rank toxic pairs
HIGH, regardless of the base reward.

## 4. Idempotency (issue #354)

The flywheel is IDEMPOTENT: re-validating the same hypothesis does NOT
double-count.

| Step | Idempotency mechanism |
|------|----------------------|
| Phase 1 CSV | `writeback_to_phase1` checks for duplicate (drug, disease, validated_by) and UPDATES the existing row instead of appending. |
| Phase 2 Neo4j | MERGE with `ON MATCH SET r.revalidation_count = coalesce(r.revalidation_count, 0) + 1`. |
| Phase 3 JSON trigger | Append-only (each trigger entry is timestamped). The trainer's `retrain_on_validated` dedupes via `existing_set = {(d, dis) for d, dis in known_pairs}`. |
| Phase 3 checkpoint | The trainer only adds pairs NOT already in `known_pairs`. Running retrain twice with the same CSV produces the same `known_pairs`. |
| Phase 4 RL ranker | `_load_validated_hypotheses` uses a `seen` set to dedupe across multiple CSV paths. |

## 5. Test Plan

| Test | File | Issue |
|------|------|-------|
| End-to-end flywheel (validate → CSV → trainer → ckpt → RL bonus) | `shared/tests/test_data_flywheel_e2e.py` | #349 |
| Toxic-pair penalty (HIGH action → negative reward) | `shared/tests/test_flywheel_toxic_penalty.py` | #350 |
| Atomic checkpoint save (crash mid-save → old ckpt intact) | `shared/tests/test_flywheel_checkpoint_atomic.py` | #351 |
| Idempotency (re-validate → no double-count) | `shared/tests/test_data_flywheel_e2e.py::test_flywheel_idempotent` | #354 |

Run:
```bash
cd <repo>
pytest shared/tests/test_data_flywheel_e2e.py shared/tests/test_flywheel_toxic_penalty.py shared/tests/test_flywheel_checkpoint_atomic.py -v
```

## 6. Monitoring (issue #355)

The flywheel has 4 steps. Each step can fail. The monitoring module
`shared/monitoring/flywheel_monitor.py` provides:

- `check_writeback_health()` — verifies the CSV exists, is non-empty,
  and the last entry is recent (within `FLYWHEEL_STALENESS_HOURS`).
- `check_retrain_trigger_health()` — verifies the JSON trigger exists
  and is parseable.
- `check_checkpoint_health()` — verifies the latest checkpoint is
  loadable and not corrupted.
- `check_rl_ranker_health()` — verifies the RL ranker loaded the
  expected number of validated bonus pairs.

Each check returns a `FlywheelStepStatus` with `ok: bool`, `message: str`,
and `last_run: datetime`. The orchestrator (Airflow / cron) calls all 4
checks and emits an alert if any returns `ok=False`.

## 7. Acceptance Criteria (audit task #355)

1. ✅ Validate a hypothesis via the API → it appears in writeback CSV
   with correct `outcome`.
2. ✅ Trigger retrain → trainer reads the CSV, runs N epochs, saves a
   new checkpoint.
3. ✅ RL ranker loads the new bonus/penalty.
4. ✅ Toxic-pair validation results in a NEGATIVE reward.
5. ✅ `pytest shared/tests/test_data_flywheel_e2e.py` passes.
