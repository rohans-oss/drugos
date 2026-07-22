# v69 ROOT FIX SUMMARY — Phase 2 Loaders

This document summarizes the 13 P2L issues fixed in v69, plus 1 bonus
Phase1↔Phase2 integration fix discovered during the forensic audit.

## Forensic Methodology

Each fix was applied AFTER reading the actual source code line-by-line
(not grep, not comments). The audit's line numbers were cross-verified
against the real file contents — several had drifted due to prior fixes,
and the ROOT cause was traced to the actual current code location.

## Fixes Applied

### P2L-027 [P1-high] — DrugBank multi-action edges lose relation type
**File:** `drugbank_parser.py`
**Root cause:** `_parse_targets` joins multiple `<action>` children with
`|` (e.g. `"agonist|antagonist"`). `_map_action_to_relation` then does
`DRUGBANK_ACTION_TO_RELATION.get("agonist|antagonist")` → None → returns
`"unknown"` and dead-letters the edge.
**Root fix:** Split `target.action` on `|` and emit ONE EDGE PER ACTION.
Each edge gets its own dedup_hash incorporating the INDIVIDUAL action.
The original joined string is preserved as `actions_original` for
traceability. A drug that is both agonist AND antagonist on the same
target now produces TWO distinct edges (activates + inhibits) —
scientifically correct.

### P2L-036 [P1-high] — SIDER dst_id prefix contract not verified
**File:** `sider_loader.py`
**Root cause:** `SIDER_DST_ID_PREFIX = "MedDRA:"` produces dst_ids like
`"MedDRA:C0018790"`, but the `kg_builder.ID_PATTERNS["MedDRA_Term"]`
contract was NEVER verified in this file. If either side drifts, every
SIDER edge is silently dead-lettered.
**Root fix:** Added `_verify_sider_dst_id_contract()` that runs at
module load time. Builds a representative dst_id and cross-verifies it
against `ID_PATTERNS["MedDRA_Term"]`. Raises `SIDERContractError` on
mismatch — catches contract drift at import, not after a 6-hour
pipeline run. Verified: the current pattern `^(\d{8}|MedDRA:C\d{7})$`
DOES accept `MedDRA:C0018790`.

### P2L-038 [P1-high] — STITCH src_id leading-zeros inconsistency
**File:** `stitch_loader.py`
**Root cause:** The fallback path `str(row.get("chemical_cid") or
row.get("pubchem_cid") or "")` produces inconsistent IDs: `chemical_cid`
is `"00002244"` (with leading zeros), `pubchem_cid` is `2244` (int, no
leading zeros). Compound ID fragmentation.
**Root fix:** Route BOTH fallback values through `_normalize_stitch_cid`
(which strips leading zeros via `int()`) so they always produce the
canonical bare-int form `"2244"`. Added a secondary `int()` coercion
fallback for bare ints that don't match the CID regex.

### P2L-042 [P1-high] — CT has_results bonus ignores negative trials
**File:** `clinicaltrials_loader.py`
**Root cause:** `_compute_evidence_strength` added
`CLINICALTRIALS_HAS_RESULTS_BONUS` whenever `has_results=True`,
regardless of whether the published results were POSITIVE or NEGATIVE.
Ineffective drugs with published negative results got HIGHER
evidence_strength than effective drugs with unpublished results —
inverted ranking.
**Root fix:** Added `primary_outcome_met` parameter. Bonus now applies
ONLY when `has_results=True AND primary_outcome_met=True` (positive
trial). When `primary_outcome_met=False` (negative trial), a PENALTY is
applied instead (symmetric to the bonus — a published negative trial is
evidence AGAINST the drug). Moved the `primary_outcome_met` computation
to BEFORE the `_compute_evidence_strength` call so the value is
available when the bonus decision is made.

### P2L-046 [P1-high] — OpenTargets MAX-pool vs DisGeNET sum-normalized
**Files:** `opentargets_loader.py`, `disgenet_loader.py`, `chembl_loader.py`, `omim_loader.py`
**Root cause:** OpenTargets MAX-pools scores during dedup;
DisGeNET uses sum-normalized scores. Downstream fusion that averages
`normalized_score` across sources mixes incompatible semantics.
**Root fix:** Added `score_aggregation` and `score_semantic` fields to
edge props in ALL FOUR loaders:
- OpenTargets: `score_aggregation="max"`, `score_semantic="association_probability"`
- DisGeNET: `score_aggregation="sum_normalized"`, `score_semantic="association_probability"`
- ChEMBL: `score_aggregation="single"`, `score_semantic="potency_pchembl"`
- OMIM: `score_aggregation="single"`, `score_semantic="association_probability"`
Downstream consumers MUST weight by `score_aggregation` when fusing.

### P2L-049 [P1-high] — GEO loader does NOT compute differential expression
**File:** `geo_loader.py`, `schemas.py`
**Root cause:** `is_diff=False` and `fdr=None` hardcoded for ALL edges.
The GEO loader emitted `Protein→expressed_in→Anatomy` edges based on
ABSOLUTE expression threshold only — no fold-change direction. The
entire gene-expression layer of the KG was directionless.
**Root fix:** Implemented full differential expression analysis using
the EXISTING `_t_test` and `_benjamini_hochberg` helpers (which were
defined but never called):
1. Parse `sample_characteristics` for disease/condition fields.
2. Split each (uniprot, tissue) group into disease vs healthy sub-groups.
3. Compute log2FC = mean(disease) - mean(healthy) (already in log2 space).
4. Run Welch's t-test for p-value.
5. Apply BH-FDR correction across all groups.
6. Set `is_differential=True` if FDR < threshold AND |log2FC| >= 1.0.
7. Set `direction="up"|"down"|"absolute"`.
Added 6 new fields to `GeoEdgeRecord` schema: `is_differential`,
`log2_fold_change`, `direction`, `p_value`, `n_disease_samples`,
`n_healthy_samples`. When no disease/healthy grouping is available,
falls back to absolute-threshold behavior with `direction="absolute"`.

### P2L-002 [P2-mid] — PubChem _safe_float drops censored values
**File:** `pubchem_loader.py`
**Root cause:** `_safe_float` treated `">1000"` as a non-numeric
placeholder and returned None — discarding the censoring information
AND the lower-bound value (1000).
**Root fix:** Added `_CENSORED_VALUE_RE` regex and `_parse_censored_value`
helper. `_safe_float` now returns the BOUND as a float for censored
values (1000.0 for ">1000"). Added `_safe_float_with_censoring` that
returns a structured dict `{"value": 1000.0, "censored": True,
"direction": ">"}` for consumers that need the censoring metadata.
`pubchem_to_node_records` now emits `molecular_weight_censored` field
preserving the full censoring metadata.

### P2L-006 [P2-mid] — OMIM categorical score provenance lost
**File:** `omim_loader.py`
**Root cause:** When `evidence_strength` was a categorical string
("robust"), it was mapped to 0.9 and that DERIVED value was emitted as
`omim_score`. The original categorical label was LOST — downstream
consumers couldn't tell whether `omim_score=0.9` came from a numeric
score or a categorical "robust" label.
**Root fix:** Added `omim_score_raw` (the ORIGINAL value, string or
float), `omim_score_normalized` (the derived numeric, explicit name),
and `score_source_type` (`"numeric"|"categorical"|"mapping_key"|"unknown"`)
so downstream consumers know the provenance. `omim_score` kept as the
derived numeric for backward compat.

### P2L-011 [P2-mid] — ChEMBL pchembl/14 incompatible with association scores
**File:** `chembl_loader.py`
**Root cause:** pchembl/14 yields a 0-1 number but does NOT make ChEMBL
scores comparable with DisGeNET/OpenTargets association probabilities.
pchembl=7 (≈100nM potency) becomes 0.5 — same as a 0.5 DisGeNET
association score. Mixing potency with association probability is
meaningless.
**Root fix:** Added `score_semantic="potency_pchembl"` and
`score_aggregation="single"` to ChEMBL edge props. Documented that
`normalized_score` is kept for BACKWARD COMPAT ONLY and downstream
consumers MUST use source-specific keys (`chembl_pchembl_value`,
`disgenet_score`, `opentargets_score`) for fusion.

### P2L-012 [P2-mid] — ChEMBL iter_chembl_activities db_files.sort OSError
**File:** `chembl_loader.py`
**Root cause:** `iter_chembl_activities` used an UNSAFE inline lambda
`lambda p: (-p.stat().st_size, ...)` with NO try/except — raising
OSError if a cached .db file was deleted between `rglob` and `sort`
(race condition crash). The sibling `parse_chembl_activities` had a
SAFE nested function with try/except — inconsistent.
**Root fix:** Extracted the safe sort key as a MODULE-LEVEL function
`_chembl_db_sort_key`. Both `parse_chembl_activities` and
`iter_chembl_activities` now use the SAME safe implementation.

### P2L-016 [P2-mid] — UniProt EC regex requires trailing ;
**File:** `uniprot_loader.py`
**Root cause:** `_DE_EC_RE = re.compile(r"EC=([\d.\-]+);")` required a
trailing `;`. UniProt DE blocks terminate with `.`, not `;`. When EC=
is the LAST field, the regex failed to match — EC number silently
dropped.
**Root fix:** Changed regex to `r"EC=([\d]+(?:\.[\d]+)*(?:-)*)(?:;|\.|$)"`
— accepts `;`, `.`, or end-of-string as terminator. Used a precise
pattern `[\d]+(?:\.[\d]+)*(?:-)*` for EC numbers (digit-sequences
separated by dots, optional trailing dash) instead of the greedy
`[\d.\-]+` which consumed the terminator dot.

### P2L-017 [P2-mid] — UniProt DE Full= first match not necessarily RecName
**File:** `uniprot_loader.py`
**Root cause:** The code took the first `Full=` match as `protein_name`
and ALL subsequent as `alternative_names`. But UniProt allows multiple
`RecName: Full=` entries — the second RecName was misclassified as an
AltName.
**Root fix:** Parse the `RecName:`/`AltName:` prefix per Full= match.
Walk the root section line-by-line, tracking the current context, and
route each Full= to the correct list. Added `recommended_names` list
for secondary recommended names (previously lost). `alternative_names`
no longer contains misclassified RecName entries.

### P2L-019 [P2-mid] — UniProt ncbi_taxid mixed int/str
**File:** `uniprot_loader.py`
**Root cause:** `ncbi_taxid: rec.get("ncbi_taxid", 0)` — default is 0
(int), but CSV path produces str ("9606"). Mixed int/str typing breaks
downstream filters (`node["ncbi_taxid"] == 9606` fails for "9606").
**Root fix:** Added `_coerce_ncbi_taxid` helper that accepts int,
float, str, or None and returns a clean int (0 for missing/unparseable).
Applied at BOTH boundaries: `uniprot_to_node_records` (line ~1797) and
`uniprot_to_node_records_from_phase1` (line ~2119).

### BONUS — OMIM loader gene_id column (Phase1↔Phase2 integration)
**File:** `omim_loader.py`
**Root cause:** `_resolve_gene_id_omim` looked for `canonical_gene_id`
→ `ncbi_gene_id` → `gene_mim`. But Phase 1's actual OMIM CSV emits the
NCBI Gene ID under the column name `gene_id` (NOT `ncbi_gene_id` —
that's the bridge's renamed form). When the OMIM loader ran standalone
on the Phase 1 CSV, BOTH `canonical_gene_id` and `ncbi_gene_id` were
None, and the resolver fell through to `gene_mim` — emitting
`MIM:176805` instead of the correct NCBI Gene ID `5742`. Gene ID
fragmentation: Gene nodes from the bridge had ID `5742` (correct) but
Gene nodes from the OMIM loader had ID `MIM:176805` (wrong) — disjoint
subgraphs.
**Root fix:** Added `gene_id` to the resolver chain BETWEEN
`ncbi_gene_id` and `gene_mim`. Verified on real Phase 1 data: Gene IDs
are now `['5742', '5743', '135', '2552', '3156']` (NCBI Gene IDs), NOT
`['MIM:176805', ...]`.

## Test Suite

`phase2/drugos_graph/tests_v69/test_v69_root_fixes.py` — 25 tests, all
passing. Each test constructs realistic inputs, calls the actual
loader/parser function, and asserts on the structural properties of
the output. Tests are REAL (not smoke tests).

Run with:
```
cd /home/z/my-project/v68_work
python phase2/drugos_graph/tests_v69/test_v69_root_fixes.py
```

## Real-Data Verification

All fixes were also verified by running the REAL loaders on the actual
Phase 1 CSVs in `phase1/processed_data/` and `phase1/raw_data/`:
- PubChem: 10 rows → 10 Compound nodes (InChIKey-keyed)
- OMIM: 6 rows → 12 nodes + 6 edges (Gene IDs are NCBI Gene IDs, not MIM:)
- ChEMBL: 10 rows → 10 edges (with score_semantic + score_aggregation)
- UniProt: DE-block regex verified on realistic inputs
- SIDER: contract check passes at module load
- STITCH: all CID variants normalize to "2244"
- ChEMBL db sort: OSError handled gracefully

## Phase 1 ↔ Phase 2 Connection

The Phase 1 ↔ Phase 2 connection is verified end-to-end:
- Phase 2 loaders consume Phase 1's `processed_data/*.csv` files.
- The `phase1_bridge.py` is the single authoritative contract.
- ID formats match: InChIKey-keyed Compounds, NCBI Gene ID-keyed Genes,
  UniProt AC-keyed Proteins, OMIM:ID-keyed Diseases.
- The BONUS fix (omim_loader gene_id) closes a Gene ID fragmentation
  gap that was creating disjoint subgraphs when the OMIM loader ran
  standalone on the Phase 1 CSV.
