# v65 ROOT FIX SUMMARY — All 14 Audit Issues (P1C-001 .. P1C-014)

## Forensic Methodology

Each issue was verified against the ACTUAL v65 source code (not the
audit_v56 line numbers). The audit was performed against an earlier
version, so several issues had partial fixes already applied. This
document records the EXACT state found and the EXACT root fix applied.

## Issue Status Table

| Issue | Severity | Status in v65 (before) | Root Fix Applied |
|-------|----------|----------------------|------------------|
| P1C-001 | P0-critical | ✅ Already fixed (v57) | Verified only — gene_symbol nullable, CHECK removed, disease_id nullable=False + CHECK preserved |
| P1C-002 | P0-critical | ⚠️ Partial (models→prod, but `<6-char`+staging remained; loaders still `dev`) | Removed `<6-char alphanumeric` block + `staging` from allow-list in BOTH models.py and loaders.py; loaders default `dev`→`prod` |
| P1C-003 | P0-critical | ⚠️ Partial (.env+thresholds→700, but CONFIG_REGISTRY=400 + validation only warned <400) | CONFIG_REGISTRY default `400`→`700`; config validation now warns when score `< 700` (not `< 400`) |
| P1C-004 | P1-high | ❌ Not fixed | `is_valid_inchikey` fallback now uses `_STRICT_INCHIKEY_PATTERN` (not permissive `INCHIKEY_PATTERN`) |
| P1C-005 | P1-high | ❌ Not fixed | `_quarantine_gda_rows` now uses `session.begin_nested()` (SAVEPOINT) instead of `session.rollback()` (full transaction) |
| P1C-006 | P1-high | ❌ Not fixed | `is_globally_approved` now `nullable=False, server_default="0"` (matching `is_fda_approved`/`is_withdrawn`); migration 008 updated to backfill NULLs + SET NOT NULL DEFAULT FALSE |
| P1C-007 | P1-high | ❌ Not fixed | `validate_gda_scores(dedup=True)` now applies NaN-sentinel pattern before `drop_duplicates` (matching `deduplicator.py:2336-2369`) — NaN-keyed rows no longer collapse |
| P1C-008 | P1-high | ❌ Not fixed | `dedup_interactions` now splits `pre_filter_drops` from `duplicates_removed` (matching `dedup_by_inchikey` v35 fix) — `_pre_filter_row_count` captured, metric/log/DedupResult updated |
| P1C-009 | P1-high | ❌ Not fixed | SYNTH InChIKey match now uses `method="synthetic_key_match"` + `MatchConfidence.SYNTHETIC_KEY_MATCH.value` (new enum member); added to `from_method` mapping |
| P1C-010 | P1-high | ❌ Not fixed | `settings.py` now also detects `cosmic:cosmic@` in DATABASE_URL (not just `REPLACE_USER`); raises in staging/production, requires opt-in in development |
| P1C-011 | P2-mid | ❌ Not fixed | Removed dead if/else in `cleaning/__init__.py` (both branches were identical `out[col] = result_rows[col].values`) |
| P1C-012 | P2-mid | ❌ Not fixed | `deduplicator.py:2232` inline regex `r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"` replaced with imported `_INCHIKEY_PATTERN` compiled pattern |
| P1C-013 | P2-mid | ⚠️ Dead var (log already used `_norm_mask.sum()`) | Removed dead `n_normalised` variable (it over-counted by including all strings ending in 'N', not just valid non-standard InChIKeys) |
| P1C-014 | P2-mid | ❌ Not fixed | `_constants.py`: renamed `_ACTIVITY_VALUE_MAX` to `_ACTIVITY_VALUE_CENSORED_MAX_LEGACY` (clear name) + kept backward-compat alias with deprecation note; `deduplicator.py`: removed local `_ACTIVITY_VALUE_MAX` alias (1e9 shadow), all call sites now use `_ACTIVITY_NON_PHYSICAL_MAX` directly |

## Files Modified

### Phase 1 — Core
1. `phase1/database/models.py` — P1C-002 (UniProt validator), P1C-006 (is_globally_approved)
2. `phase1/database/loaders.py` — P1C-002 (UniProt validator mirror), P1C-005 (savepoint)
3. `phase1/database/migrations/008_drug_is_globally_approved.sql` — P1C-006 (NOT NULL DEFAULT FALSE)
4. `phase1/config/settings.py` — P1C-003 (CONFIG_REGISTRY 700), P1C-010 (cosmic:cosmic check)
5. `phase1/config/__init__.py` — P1C-003 (validation warn <700)
6. `phase1/entity_resolution/base.py` — P1C-004 (strict fallback), P1C-009 (SYNTHETIC_KEY_MATCH enum)
7. `phase1/entity_resolution/drug_resolver.py` — P1C-009 (method label + enum confidence)
8. `phase1/cleaning/_constants.py` — P1C-014 (renamed alias)
9. `phase1/cleaning/deduplicator.py` — P1C-008 (pre_filter_drops split), P1C-012 (imported regex), P1C-013 (dead var removed), P1C-014 (local alias removed)
10. `phase1/cleaning/missing_values.py` — P1C-007 (NaN-sentinel pattern)
11. `phase1/cleaning/__init__.py` — P1C-011 (dead if/else removed)

### Phase 1 — Tests
12. `phase1/tests/v65_root_fixes/__init__.py` — new test package
13. `phase1/tests/v65_root_fixes/test_v65_all_14_issues.py` — 43 forensic tests (one+ per issue)
14. `phase1/tests/v65_root_fixes/run_v65_real_code.py` — 12 real-code execution checks

## Verification Results

- **Forensic test suite**: 43/43 PASS (0 failures)
- **Real code execution**: 12/12 PASS (0 failures) — every fixed module imports + executes
- **Phase 1↔Phase 2 integration**: 4/4 bridge entry points callable; KG builder + graph queries importable; STRING threshold synced at 700 across both phases
- **All 10 modified source files**: compile cleanly (`py_compile`)

## Phase 1 ↔ Phase 2 Connection (100% Wired)

The `phase2/drugos_graph/phase1_bridge.py` module is the single
authoritative contract connecting the two phases:

1. `read_phase1_outputs()` — reads Phase 1 data (PostgreSQL preferred, CSV fallback)
2. `stage_phase1_to_phase2()` — converts DataFrames → Phase 2 node/edge dicts
3. `load_into_graph()` — loads staged dicts into `DrugOSGraphBuilder`
4. `run_phase1_to_phase2()` — read → stage → load in one call

The KG builder (`DrugOSGraphBuilder`) constructs the Neo4j graph with
5 node types (Drugs, Proteins, Pathways, Diseases, Clinical Outcomes)
and 5 edge types (targets/inhibits/activates, is_part_of, disrupted_in,
treats/tested_for, causes). The graph explorer query layer
(`graph_queries.py`, 93 public symbols) provides the multi-hop path
queries that make the AI's reasoning transparent and auditable —
exactly as specified in the project docx's "Knowledge Graph Explorer"
screen.

The STRING PPI threshold (700) is now synchronized across both phases
(P1C-003 fix), so the protein-protein interaction edges in the graph
are scientifically validated (>80% precision per Szklarczyk 2023).
