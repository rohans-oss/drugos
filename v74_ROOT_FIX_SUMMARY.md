# v74 ROOT-LEVEL FIX SUMMARY — T-013 through T-023

**Date:** 2026-07-10
**Base:** v73_ROOT_FIXED_codebase
**Status:** ALL 11 ISSUES ROOT-LEVEL FIXED AND VERIFIED

## Verification Results

```
V74 ROOT FIX VERIFICATION: 27 passed, 0 failed (of 27 tests)
ALL ROOT-LEVEL FIXES VERIFIED SUCCESSFULLY
```

Run `python tests/test_v74_root_fixes.py` from `phase1/` to re-verify.

---

## Issue-by-Issue Root-Level Fixes

### T-013 [P1-high] — activity_value FLOAT vs Numeric(10,4) schema drift
**Root cause:** Migration 001 declared `drug_protein_interactions.activity_value FLOAT`
(IEEE 754 binary64). ORM `models.py` declared `Numeric(10, 4)` (decimal). Dev DBs
(ORM-created) had NUMERIC; prod DBs (migration-created) had FLOAT. IC50 values like
0.00123 nM differed by 1 ULP between dev and prod, breaking pIC50 reproducibility.

**Root fix (two-part):**
1. Amended migration 001 line 624: `activity_value FLOAT,` → `activity_value NUMERIC(10, 4),`
2. Created migration `011_align_activity_value_to_orm.sql` that ALTERs the column type
   for already-deployed databases: `ALTER TABLE drug_protein_interactions ALTER COLUMN
   activity_value TYPE NUMERIC(10, 4) USING activity_value::numeric(10, 4);`
3. Migration 011 is SQLite-translatable (no `RETURN;` statements — uses IF/ELSIF/ELSE).
4. Created rollback `011_align_activity_value_to_orm_rollback.sql`.

**Verification:** ORM type is `Numeric(10, 4)` (precision=10, scale=4); migration 001
declares `NUMERIC(10, 4)`; migration 011 applies cleanly on SQLite; no type mismatch
in `verify_schema_matches_orm`.

---

### T-014 [P1-high] — chk_drugs_is_globally_approved missing from ORM
**Root cause:** Migration 008 added `chk_drugs_is_globally_approved` CHECK via ALTER
TABLE, but the ORM `Drug` model declared the column WITHOUT a matching CheckConstraint
in `__table_args__`. Dev DBs (SQLite/ORM-created) lacked the constraint; prod DBs
(migration-created) had it. A row with `is_globally_approved=2` would INSERT on dev
but fail on prod.

**Root fix:**
1. Added `CheckConstraint("is_globally_approved IS NOT NULL AND is_globally_approved
   IN (0, 1)", name="chk_drugs_is_globally_approved")` to `Drug.__table_args__` in
   `models.py` (lines 793-818).
2. Used the portable `IN (0, 1)` form (SQLite-compatible) instead of `IN (FALSE, TRUE)`
   (PostgreSQL-specific). Matches the pattern of `chk_drugs_is_fda_approved` and
   `chk_drugs_is_withdrawn`.
3. Added `IS NOT NULL` predicate as defense-in-depth (column is NOT NULL today, but
   the explicit predicate guards against a future migration making it nullable).
4. Simplified migration 008's CHECK to match (replaces pre-v74 `IN (FALSE, TRUE)`
   with `IN (0, 1)` if the old form was already applied).

**Verification:** ORM has the constraint; SQLite dev DB rejects `is_globally_approved=2`
with IntegrityError; migration 008 active SQL uses `IN (0, 1)`.

---

### T-015 [P1-high] — DAG docstrings contradict actual schedules
**Root cause:** All 4 standalone DAGs (disgenet, drugbank, pubchem, chembl) had
docstrings stating one schedule while the code implemented a different one. The v49/v43
ROOT FIX comments moved the schedules to avoid the "Sunday Morning Pile-Up" with the
master DAG, but the docstrings were never updated. chembl's docstring was internally
contradictory ("every Wednesday at 04:00 UTC" then "runs weekly on Sunday").

**Root fix:**
- `disgenet_dag.py`: docstring now says "Tuesday at 06:00 UTC" (matches `0 6 * * 2`)
- `drugbank_dag.py`: docstring now says "Monday at 03:00 UTC" (matches `0 3 * * 1`)
- `pubchem_dag.py`: docstring now says "Wednesday at 08:00 UTC" (matches `0 8 * * 3`)
- `chembl_dag.py`: docstring now says "Wednesday at 04:00 UTC" (matches `0 4 * * 3`),
  removed the contradictory "runs weekly on Sunday" sentence

**Verification:** Each DAG's docstring mentions the correct day; the actual `schedule=`
cron expression matches; no DAG claims Sunday (except the master DAG which IS on Sunday).

---

### T-016 [P2-mid] — Contradictory BEGIN/COMMIT comments in migration 002
**Root cause:** Migration 002's header (lines 8-21) said BEGIN/COMMIT was ADDED as the
v21 ROOT FIX. A later comment (lines 67-71) said "Do NOT wrap this file in BEGIN/COMMIT".
The file IS wrapped in BEGIN/COMMIT. The two comments directly contradicted each other.

**Root fix:** Replaced the contradictory "Do NOT wrap" comment with a clarifying
comment explaining:
- The file IS wrapped in BEGIN/COMMIT (intentional — atomicity under `psql -f`)
- Under the Python migration runner, the inner BEGIN becomes a SAVEPOINT (benign)
- Both invocation modes are correct; the wrapper is defense-in-depth

**Verification:** "Do NOT wrap this file in BEGIN/COMMIT" no longer appears; v74 ROOT
FIX T-016 clarifying comment is present.

---

### T-017 [P2-mid] — Pointless DROP INDEX + CREATE INDEX cycle in migration 002
**Root cause:** Migration 002 DROPPED `uq_entity_mapping_inchikey` (line 944) then
RE-CREATED it with an identical definition (lines 996-998). Migration 001 already
creates this exact index. The DROP+CREATE cycle acquired ACCESS EXCLUSIVE lock twice
on `entity_mapping`, blocking all concurrent reads/writes for zero schema benefit.

**Root fix:** Removed both the DROP INDEX statement (line 944) and the CREATE UNIQUE
INDEX statement (lines 996-998) from migration 002. The index from migration 001 is
authoritative and is no longer touched by migration 002. Updated the rollback file
comment to note the cycle was removed.

**Verification:** Migration 002 active SQL has 0 DROP INDEX and 0 CREATE UNIQUE INDEX
statements for `uq_entity_mapping_inchikey`. The GDA COALESCE index (which IS new in
002) is preserved.

---

### T-018 [P2-mid] — Dead `uniprot_id IS NULL OR` branch in migration 003
**Root cause:** Migration 003's `chk_proteins_uniprot_length` CHECK included
`uniprot_id IS NULL OR` — but migration 001 declares `uniprot_id VARCHAR(10) NOT NULL`,
so the `IS NULL` branch is dead code. Migration 001's version correctly omits the dead
branch. The divergence meant the constraint's strictness depended on migration
application order.

**Root fix:** Changed migration 003 line 95 from
`CHECK (uniprot_id IS NULL OR (LENGTH(uniprot_id) >= 4 AND LENGTH(uniprot_id) <= 10))`
to `CHECK (LENGTH(uniprot_id) >= 4 AND LENGTH(uniprot_id) <= 10)` (matches migration 001).
Also added a replacement branch: if the constraint already exists with the dead branch
(from a pre-v74 run), it's DROPped and re-ADDed with the tightened form.

**Verification:** Migration 003 no longer contains `uniprot_id IS NULL OR`; the
tightened `CHECK (LENGTH(uniprot_id) >= 4 AND LENGTH(uniprot_id) <= 10)` is present.

---

### T-019 [P2-mid] — Makefile error-swallowing patterns
**Root cause:** `setup` and `install-deps` targets used `|| echo "WARN: ..."` and
`|| true` to swallow errors. Docker-compose failures, venv creation failures, and
init_db failures all returned exit code 0. CI saw green even when the environment
was broken.

**Root fix:** Rewrote `setup` and `install-deps` targets to:
- Check for docker-compose/docker availability BEFORE attempting to use it
- Use explicit `|| { echo "ERROR: ..."; exit 1; }` patterns
- Provide actionable error messages (e.g., "set DATABASE_URL=sqlite:///phase1.db")
- Default `DATABASE_URL` to `sqlite:///phase1.db` for dev convenience

**Verification:** Makefile no longer contains `|| echo` or `|| true` for error
swallowing; uses `exit 1` on failures; recipe lines use TABs (not spaces).

---

### T-020 [P2-mid] — Deprecated `airflow db init` in Makefile
**Root cause:** `airflow db migrate || airflow db init` — `airflow db init` was
deprecated in Airflow 2.7 and removed in Airflow 3.0. With the unbounded pin
(`apache-airflow>=2.8.0`), pip could install Airflow 3.x, breaking the Makefile.

**Root fix:** Changed to `airflow db migrate || airflow db upgrade` (`db upgrade` is
the 2.x-compatible alias that's NOT deprecated).

**Verification:** Makefile no longer contains `airflow db init`; uses
`airflow db migrate || airflow db upgrade`.

---

### T-021 [P2-mid] — Unbounded apache-airflow pin
**Root cause:** `apache-airflow>=2.8.0` (and `>=2.10.0` for Python 3.12+) had NO
upper bound. pip could resolve to Airflow 3.x (released 2025) which has breaking
changes (`airflow db init` removed, DAG API changes, decorator semantics changed).
The Docker image is pinned to `apache/airflow:2.9.3-python3.11`, but requirements.txt
is pip-installed INSIDE that image — an unbounded pin would UPGRADE Airflow to 3.x
inside the 2.9.3 image.

**Root fix:** Added `<3.0.0` upper bound to both pins:
- `apache-airflow>=2.10.0,<3.0.0; python_version>="3.12"`
- `apache-airflow>=2.8.0,<3.0.0; python_version<"3.12"`

**Verification:** Both pins have `<3.0.0`; the unbounded versions no longer appear.

---

### T-022 [P2-mid] — rdkit silent degradation on ARM64
**Root cause:** The `platform_machine=="x86_64"` marker was already removed in v38/v54
(v74 verified — requirements.txt line 29 installs rdkit on ALL Python 3.8+ platforms).
BUT: `resolver_utils.py` did `except ImportError: pass` when rdkit was unavailable,
silently skipping the InChIKey↔InChI cross-field consistency check. On ARM64 dev
machines where rdkit may not be installed, entity resolution quality dropped with NO
visible signal.

**Root fix:**
1. Added module-level `_RDKIT_UNAVAILABLE_WARNED: bool = False` flag in `resolver_utils.py`.
2. Changed the `except ImportError: pass` to emit a one-time WARNING log explaining
   that entity resolution is degraded (name-only matching instead of fingerprint
   similarity). The warning fires at most once per process to avoid log spam.
3. Added an import-time rdkit availability probe in `drug_resolver.py` that emits a
   WARNING if rdkit is not installed. This is a second line of defense for code paths
   that import `drug_resolver` WITHOUT importing `cleaning.normalizer` first.

**Verification:** `resolver_utils.py` has the `_RDKIT_UNAVAILABLE_WARNED` flag and the
v74 T-022 warning message; `drug_resolver.py` has the import-time probe and
`_RDKIT_AVAILABLE` flag; neither file silently passes on ImportError.

---

### T-023 [P2-mid] — retries=2 on 4xx HTTP errors waste 60 min
**Root cause:** All 7 standalone DAGs had `retries=2, retry_delay=30min` on every
task. For 4xx errors (401 Unauthorized — bad API key, 403 Forbidden — quota exceeded,
404 Not Found — wrong endpoint), retrying after 30 minutes wastes 60 minutes (2
retries × 30min) and still fails — the API key won't un-expire, the endpoint won't
un-disappear.

**Root fix:**
1. Created `dags/_retry_policy.py` — a shared helper module exporting:
   - `DEFAULT_RETRY_ARGS`: dict with `retries=2`, `retry_delay=5min` (was 30min),
     `retry_exponential_backoff=True`, `max_retry_delay=20min`
   - `is_http_4xx_error(exc)`: classifies exceptions by HTTP status code. 429 (Too
     Many Requests) is EXCLUDED — it's a rate-limit signal and retries are the
     correct response. 400/401/402/403/404/405/410/451 are non-retryable.
   - `fail_fast_on_http_4xx(func)`: decorator that wraps a task function. If the
     raised exception is a non-retryable 4xx, it's re-raised as
     `AirflowFailException` (which Airflow treats as terminal — no retry). All
     other exceptions (5xx, network timeout, 429) are re-raised unchanged so
     Airflow's normal retry logic applies.
2. Updated all 7 standalone DAGs to:
   - Import `DEFAULT_RETRY_ARGS` and `fail_fast_on_http_4xx` from `dags._retry_policy`
   - Use `**DEFAULT_RETRY_ARGS` in their `DEFAULT_ARGS` dict
   - Apply `@fail_fast_on_http_4xx` decorator to the `run_XXX` task function
   - Use `retry_exponential_backoff=True` and `retry_delay=timedelta(minutes=5)`
     in the `@task` decorator

**Verification:** All 7 DAGs import and use `fail_fast_on_http_4xx`; all use
exponential backoff; none use the old 30-min retry delay; `is_http_4xx_error`
correctly classifies 400/401/403/404 as non-retryable and 429/500/502/503 as retryable.

---

## Phase 1 ↔ Phase 2 Bridge Verification

The Phase 1 → Phase 2 connection is **100% wired and verified**:
- `phase2/drugos_graph/phase1_bridge.py` provides 4 callable entry points:
  `read_phase1_outputs`, `stage_phase1_to_phase2`, `load_into_graph`,
  `run_phase1_to_phase2`
- `phase2/drugos_graph/run_pipeline.py::step1_load_phase1` consumes the bridge
  output and converts it to the DRKG-style df shim for downstream PyG/TransE steps
- `run_unified.py` (project root) chains Phase 1 → Bridge → Phase 2 in one command
- Verified end-to-end: 55 nodes + 45 edges flow from Phase 1 CSVs → Phase 2 graph
  builder across all 5 node types (Compound, Protein, Gene, Disease, Pathway) and
  6 edge types (targets, inhibits, interacts_with, participates_in, associated_with,
  susceptible_to)

---

## How to Verify

```bash
cd phase1
DATABASE_URL=sqlite:////tmp/v74_test.db \
DRUGOS_DEV_ALLOW_DEFAULT_DB=1 \
DISGENET_USE_API=false \
PYTHONPATH=. \
python tests/test_v74_root_fixes.py
```

Expected output:
```
V74 ROOT FIX VERIFICATION: 27 passed, 0 failed (of 27 tests)
ALL ROOT-LEVEL FIXES VERIFIED SUCCESSFULLY
```

For the full unified pipeline (Phase 1 → Phase 2):
```bash
cd /path/to/v74_codebase
python run_unified.py --no-full-pipeline
```

Expected output:
```
UNIFIED RUN COMPLETE — 55 nodes, 45 edges loaded
```
