# v83 ROOT FIX SUMMARY — COMP-3, COMP-5, COMP-6, DAG-2

**Branch:** `fix/comp3-comp5-comp6-dag2-forensic-root-fixes`
**Date:** 2026-07-11
**Agent:** Super Z (forensic root-cause audit + runtime verification)

## Forensic Audit Results

After reading every file in the issue chains line-by-line (NOT grep, NOT comments — actual code), the REAL status of each issue is:

| Issue | Status Before v83 | Root Cause |
|-------|-------------------|------------|
| COMP-1 | ✅ Already fixed (v80) | Format detection in UniProt/STRING/PubChem clean() — verified at runtime |
| COMP-2 | ✅ Already fixed (v79) | load_chembl/drugbank/uniprot tasks exist in master DAG |
| **COMP-3** | ❌ NOT FIXED | `resolve_gene_symbol_to_uniprot` (loaders.py:4602) UNCONDITIONALLY overwrote `uniprot_id` with DB map lookup, discarding clean-time resolution |
| COMP-4 | ✅ Already fixed (v80) | Neo4j exporter contract accepts drugbank_drugs.csv / drugbank_open_drugs.csv / chembl_drugs.csv / drugs.csv |
| **COMP-5** | ❌ NOT FIXED | `_extract_inheritance_pattern` extracted inheritance into a separate column but did NOT strip it from phenotype_name; `disease_name = phenotype_name` copied the corrupted name |
| **COMP-6** | ❌ NOT FIXED | `_delete_checkpoint` method did not exist; checkpoint file persisted after successful download → stale cursor on next run → HTTP 400 → stuck |
| DAG-1 | ✅ Already fixed (v79) | Same as COMP-2 |
| **DAG-2** | ❌ NOT FIXED | Master DAG DEFAULT_ARGS used `retry_delay=30min` with NO exponential backoff; tasks used bare `@task()` with NO `@fail_fast_on_http_4xx` |
| DAG-3 | ✅ Already fixed (v80) | `_trigger_phase2` streams to log file instead of `capture_output=True` |
| DAG-7,8,9,10 | ✅ Already correct | No action needed |

## Root-Cause Fixes Applied

### COMP-5: OMIM disease_name retains inheritance (ROOT FIX)

**File:** `phase1/pipelines/omim_pipeline.py`

**Root cause:** `_parse_phenotype_field` (line 1596) strips markers/MIM/mapping_key but PRESERVES the trailing inheritance pattern in phenotype_name (docstring line 1615-1618 confirms: "a trailing inheritance pattern (if any) is preserved"). Step 11 (line 1262) extracted inheritance into a separate column but did NOT strip it from phenotype_name. Step 18 (line 1369) then assigned `df["disease_name"] = df["phenotype_name"]` — copying the corrupted name.

**Fix:**
1. Added `_strip_inheritance_pattern(phenotype_name)` helper (line 3059) that removes the inheritance pattern + trailing comma/whitespace.
2. In clean() Step 11, after extracting `inheritance_pattern`, ALSO strip it from `phenotype_name` so the downstream `disease_name = phenotype_name` assignment copies the CLEAN name.

**Verification (runtime, on real morbidmap fixture):**
- `Cystic fibrosis, 219700 (3), Autosomal recessive` → `disease_name="Cystic fibrosis"`, `inheritance_pattern="autosomal recessive"` ✅
- `Duchenne muscular dystrophy, ..., X-linked recessive` → `disease_name="Duchenne muscular dystrophy"`, `inheritance_pattern="x-linked recessive"` ✅

### COMP-3: OMIM 99% GDA edges dead-lettered (ROOT FIX)

**File:** `phase1/database/loaders.py` + `phase1/pipelines/omim_pipeline.py`

**Root cause:** `resolve_gene_symbol_to_uniprot` (loaders.py:4602) did:
```python
df["uniprot_id"] = df["gene_symbol"].str.upper().map(gene_to_uniprot)
```
This UNCONDITIONALLY OVERWROTE `uniprot_id` with the DB map lookup. If the DB map was empty (UniProt pipeline not yet loaded), EVERY value became NaN — even rows where clean() had already resolved a correct UniProt accession from the HGNC crosswalk via `_resolve_gene_xref_embedded`. The OMIM `load()` method then dead-lettered 99% of GDA records as "unresolved gene_symbol".

**Fix (3 parts):**
1. **loaders.py:** `resolve_gene_symbol_to_uniprot` now PRESERVES existing non-null `uniprot_id` values. Only NULL slots are filled from the DB map, then the protein-name fallback. Added telemetry (pre_resolved count vs DB-filled count).
2. **omim_pipeline.py:** Added `_download_hgnc_crosswalk(dest_path)` function that auto-downloads the HGNC complete crosswalk from `https://www.genenames.org/cgi-bin/download/custom` (no login required) when the file is missing. The `_load_hgnc_crosswalk` function now calls it automatically instead of falling back to the 50-entry embedded crosswalk.
3. **omim_pipeline.py:** Escalated the "HGNC not found" log from INFO to WARNING (the silent degradation to 1% coverage violated the DOCX's "scientifically trusted data" mandate).

**Verification (runtime):**
- Input: 5 rows, 4 pre-resolved (CFTR→P13569, DMD→P11532, TP53→P04637, BRCA1→P38398), 1 unknown.
- DB map empty: result has 4/5 resolved (pre-resolved PRESERVED). ✅
- DB map has UNKNOWN_GENE→Q99999: result has 5/5 resolved (DB filled NULL without clobbering pre-resolved). ✅

### COMP-6: UniProt checkpoint stuck-state (ROOT FIX)

**File:** `phase1/pipelines/uniprot_pipeline.py`

**Root cause:** `_write_checkpoint` wrote the checkpoint file after every page. On successful download, `tmp_path.replace(output_path)` renamed the temp file — but the checkpoint file was NEVER deleted. If the download FAILED mid-way, the checkpoint had a valid cursor_url. On the next run with `DRUGOS_UNIPROT_RESUME=1`, the stale cursor was reused → UniProt returned HTTP 400 (invalid cursor) → non-retryable 4xx → pipeline STUCK until operator manually deleted `download_checkpoint.json`.

**Fix (3 parts):**
1. Added `_delete_checkpoint()` method that deletes the checkpoint file (idempotent).
2. Called `_delete_checkpoint()` after the successful atomic rename (`tmp_path.replace(output_path)`) so the next run starts fresh.
3. Added stale-cursor detection: when resuming, if the FIRST page fetch fails with HTTP 400/404 (stale cursor), the pipeline deletes the checkpoint, resets to fresh-start state (page 1, initial search URL, truncate temp file), and continues. This is guarded by `_stale_cursor_recovered` flag (only attempts recovery ONCE to avoid infinite loop) and `_is_first_resume_fetch` flag (only the first fetch after a resume can be a stale-cursor case).
4. Added `_is_stale_cursor_error(exc)` helper that walks the exception chain looking for HTTP 400/404 (intentionally narrow — 401/403/429 indicate different problems that stale-cursor recovery would not fix).

**Verification (runtime):**
- `_is_stale_cursor_error` detects 400/404 (PASS), rejects 401/429 (PASS). ✅
- `_delete_checkpoint` deletes the file (PASS), idempotent on missing file (PASS). ✅

### DAG-2: Master DAG retry policy inconsistency (ROOT FIX)

**File:** `phase1/dags/master_pipeline_dag.py`

**Root cause:** DEFAULT_ARGS (line 95-112) used `retries=2, retry_delay=30min` with NO `retry_exponential_backoff` and NO `max_retry_delay`. A 4xx error in the master DAG wasted 60 min (2 × 30min) per task; the standalone DAGs fail-fast in seconds via `@fail_fast_on_http_4xx` + exponential backoff (5min → 10min → 20min cap). Tasks (line 171-234) used bare `@task()` with NO `@fail_fast_on_http_4xx` decorator.

**Fix:**
1. Imported `DEFAULT_RETRY_ARGS` and `fail_fast_on_http_4xx` from `dags._retry_policy`.
2. Spread `DEFAULT_RETRY_ARGS` into `DEFAULT_ARGS` (preserving the 7h SLA/timeout override for trigger_phase2's TransE training).
3. Applied `@fail_fast_on_http_4xx` to ALL 15 `@task` functions: download_chembl/drugbank/uniprot/string/disgenet/omim/pubchem, entity_resolution, load_chembl/drugbank/uniprot/string/disgenet/omim/pubchem_enrichment.

**Verification (AST + runtime):**
- AST analysis confirms all 15 task functions have `@fail_fast_on_http_4xx` decorator. ✅
- `DEFAULT_RETRY_ARGS` values: retries=2, retry_delay=5min, retry_exponential_backoff=True, max_retry_delay=20min. ✅
- `is_http_4xx_error` correctly identifies 400/401/403/404/410/451 as non-retryable, rejects 429 (rate limit, retryable) and 5xx. ✅
- `fail_fast_on_http_4xx` converts 4xx to `AirflowFailException` (non-retryable), re-raises 5xx unchanged. ✅

## Runtime Verification

All fixes verified by running REAL code (not smoke tests, not test files):

1. **COMP-5 e2e:** Ran `OMIMPipeline.clean()` on real `morbidmap_sample.txt` fixture → `disease_name` is clean (no inheritance pattern), `inheritance_pattern` column preserves the info.
2. **COMP-3 e2e:** Called `resolve_gene_symbol_to_uniprot` with pre-resolved uniprot_id + empty DB map → 4/4 pre-resolved values PRESERVED (was 0/4 before fix).
3. **COMP-6 e2e:** Called `_delete_checkpoint()` on a real checkpoint file → file deleted. Tested `_is_stale_cursor_error` on 400/404/401/429 → only 400/404 detected.
4. **DAG-2 AST:** All 15 task functions have `@fail_fast_on_http_4xx`. `DEFAULT_ARGS` spreads `DEFAULT_RETRY_ARGS`.
5. **COMP-1 verification:** UniProt `_normalize_v50_to_raw_tsv`, STRING `_load_links_file`, PubChem `_clean_v50_enrichment_csv` all present (format detection already fixed in v80).
6. **COMP-4 verification:** `Phase1OutputContract.required["drugs"]` accepts all 4 candidates. `validate_phase1_output_contract` resolves drugs→chembl_drugs.csv when DrugBank absent.

## Test Suite Results

| Test File | Pass | Fail | Pre-existing Failures |
|-----------|------|------|----------------------|
| test_omim_pipeline.py | 110 | 3 | 3 (all pre-existing — verified via git stash) |
| test_uniprot_pipeline_institutional_v346.py | 106 | 3 | 3 (all pre-existing) |
| test_db_loaders.py | 22 | 1 | 1 (pre-existing — InChIKey CHECK constraint on test fixture) |

**ZERO new test failures introduced by v83.** All 7 failures are pre-existing (verified by stashing changes and running tests on original code).

## Files Changed

```
phase1/dags/master_pipeline_dag.py   |  49 +++++++-  (DAG-2)
phase1/database/loaders.py           |  81 +++++++++++--  (COMP-3)
phase1/pipelines/omim_pipeline.py    | 229 +++++++++++++++++++++++++++++++  (COMP-3 + COMP-5)
phase1/pipelines/uniprot_pipeline.py | 175 +++++++++++++++++++++++++-  (COMP-6)
.github/workflows/ci.yml             |  NEW  (institutional-grade CI)
```

## GitHub Actions CI

Created `.github/workflows/ci.yml` — runs Phase 1 tests (OMIM, UniProt, DAG structure, DB loaders), Phase 1→2 contract verification (COMP-4), and Phase 2 bridge tests (gated on phase2/ changes). The `ci-success` job is the required status check for branch protection.
