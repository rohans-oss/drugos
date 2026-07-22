# Phase 1 Output Contract

**Single source of truth for the 11 Phase 1 output CSV files.**

This document is the canonical contract between Phase 1 (data ingestion) and Phase 2 (knowledge graph build). **Phase 2 owners MUST read this before consuming Phase 1 outputs.**

---

## Why this contract exists

Previous versions of the platform suffered from **schema drift**: Phase 1 renamed a column, Phase 2 didn't notice, and the KG silently lost edges or had phantom nodes. The audit found multiple cases where a "ROOT FIX" in Phase 1 was undone by a stale Phase 2 expectation — the bridge read a column that no longer existed, got `None` for every row, and built a KG with zero edges of that type.

This contract eliminates that drift by declaring **one** canonical schema, in code, in `phase1_schema.py`. Both Phase 1 (producer) and Phase 2 (consumer) import the same module. The `validate_output.py` script (Task 8) runs as the **final task** in the Airflow DAG and fails the DAG if any CSV deviates from the contract.

---

## The 11 Phase 1 output files

| # | Source key | Canonical filename | Aliases | Min rows | Required by Phase 2 |
|---|-----------|--------------------|---------|---------|--------------------|
| 1 | `chembl_drugs` | `chembl_drugs.csv` | `drugs.csv` | 1 | Compound nodes (ChEMBL-only fallback) |
| 2 | `chembl_activities` | `chembl_activities_clean.csv` | `chembl_activities.csv` | 1 | Compound→Protein edges |
| 3 | `drugs` | `drugbank_drugs.csv` | `drugbank_open_drugs.csv`, `chembl_drugs.csv`, `drugs.csv` | 0 | Compound nodes (preferred) |
| 4 | `interactions` | `drugbank_interactions.csv` | `.gz`, `drugbank_open_interactions.csv`, `chembl_activities_clean.csv`, `chembl_activities.csv` | 0 | Compound→Protein edges (DrugBank form) |
| 5 | `indications` | `drugbank_indications.csv` | — | 0 | ClinicalOutcome nodes + treats edges |
| 6 | `uniprot_proteins` | `uniprot_proteins.csv` | `proteins.csv` | 1 | Protein nodes |
| 7 | `string_ppi` | `string_protein_protein_interactions.csv` | `protein_protein_interactions.csv` | 1 | Pathway nodes + Protein→Protein edges |
| 8 | `disgenet_gda` | `disgenet_gene_disease_associations.csv` | `gene_disease_associations.csv` | 1 | Gene→Disease edges |
| 9 | `omim_gda` | `omim_gene_disease_associations.csv` | — | 1 | Gene→Disease edges (Mendelian) |
| 10 | `omim_susceptibility` | `omim_gene_disease_susceptibility.csv` | — | 0 | Gene→Disease edges (complex) |
| 11 | `pubchem_enrichment` | `pubchem_enrichment.csv` | — | 0 | Compound property enrichment |

### Why some `min_rows` are 0

- `drugs` (DrugBank) — DrugBank academic downloads have been paused since May 2026. When unavailable, the bridge degrades to `chembl_drugs.csv`. The `drugbank_drugs.csv` file is emitted as header-only (0 rows) so the file-existence check passes.
- `interactions` — Same DrugBank degradation; the bridge falls back to `chembl_activities_clean.csv`.
- `indications` — DrugBank-only source. When DrugBank is unavailable, no ClinicalOutcome nodes are produced (the bridge logs a WARNING).
- `omim_susceptibility` — OMIM susceptibility associations are a SUBSET of the GDA file. If no susceptibility associations exist for the current dataset, the file is header-only.
- `pubchem_enrichment` — PubChem enrichment is best-effort. If the PubChem API is rate-limited or unreachable, the file is header-only.

### Why some columns are "any-of"

For sources where the bridge accepts multiple alternative column names, the contract declares an **any-of group**: at least ONE column in the group must be present. Examples:

- `drugs.any_of_groups = (("drugbank_id", "chembl_id"),)` — accept either DrugBank ID or ChEMBL ID (the bridge uses InChIKey as the canonical Compound key).
- `uniprot_proteins.any_of_groups = (("uniprot_ac", "accession", "uniprot_id"),)` — the UniProt pipeline emits `uniprot_id` (canonical); legacy code may emit `uniprot_ac` or `accession`.
- `disgenet_gda.any_of_groups = (("gene_id", "ncbi_gene_id"),)` — DisGeNET's API returns `gene_id`; some cached datasets use `ncbi_gene_id`.
- `string_ppi.any_of_groups = (("uniprot_ac_a", "protein_a", "uniprot_id_a", "string_id_a"), ...)` — STRING emits ENSP IDs (`string_id_a`); the pipeline crosswalks to UniProt (`uniprot_id_a`) when possible.

---

## Column schemas (canonical)

The full column-level schema (name, dtype, nullable, description) is defined in code at `phase1/contracts/phase1_schema.py`. That file is the **single source of truth** — this README summarizes it but the code wins in any conflict.

### `chembl_drugs` (ChEMBL FDA-approved drugs)

| Column | Dtype | Nullable | Required | Description |
|--------|-------|----------|----------|-------------|
| `chembl_id` | string | No | Yes | ChEMBL molecule ID (`CHEMBL\d+`) |
| `inchikey` | string | No | Yes | 27-char InChIKey (canonical compound key) |
| `name` | string | No | Optional | Preferred drug name (e.g. "Aspirin") |
| `smiles` | string | Yes | Optional | Canonical SMILES |
| `molecular_weight` | float64 | Yes | Optional | Molecular weight (Daltons) |
| `max_phase` | int64 | Yes | Optional | ChEMBL max phase (0-4). 4 = globally approved |
| `is_fda_approved` | bool | Yes | Optional | True=FDA, False=not, None=unknown |
| `is_globally_approved` | bool | Yes | Optional | True if max_phase==4 (any regulator) |
| `indication` | string | Yes | Optional | Free-text indication |
| `indication_source` | string | Yes | Optional | Source of indication field |
| `mechanism_of_action` | string | Yes | Optional | Mechanism of action text |

### `chembl_activities` (ChEMBL bioactivity)

| Column | Dtype | Nullable | Required | Description |
|--------|-------|----------|----------|-------------|
| `molecule_chembl_id` | string | No | Yes | Foreign key to `chembl_drugs.chembl_id` |
| `target_chembl_id` | string | No | Yes | ChEMBL target ID |
| `pchembl_value` | float64 | Yes | Yes | -log10(activity in M); higher = more potent |
| `standard_relation` | string | Yes | Yes | `=`, `<`, or `>` |
| `uniprot_id` | string | Yes | Optional | UniProt accession for target |
| `target_name` | string | Yes | Optional | Target protein name |
| `activity_type` | string | Yes | Optional | IC50, Ki, EC50, etc. |
| `activity_value` | float64 | Yes | Optional | Activity value (nM) |
| `activity_units` | string | Yes | Optional | Units (typically "nM") |

### `drugs` (DrugBank drugs — preferred Compound source)

**Required**: `name`, `inchikey`
**Any-of**: at least one of `drugbank_id` OR `chembl_id`
**Optional**: `pubchem_cid`, `smiles`, `molecular_weight`, `molecular_formula`, `indication`, `indication_source`, `mechanism_of_action`, `groups`, `is_fda_approved`, `is_globally_approved`, `is_withdrawn`, `clinical_status`, `max_phase`, `drug_type`, `cas_number`, `logp`, `tpsa`

### `interactions` (DrugBank drug-protein)

**Required**: `drugbank_id`, `uniprot_id`, `action_type`
**Optional**: `target_name`, `target_chembl_id`

### `indications` (DrugBank drug-disease)

**Required**: `drugbank_id`, `disease_id`
**Optional**: `drug_inchikey`, `drug_name`, `disease_name`, `doid_id`, `omim_disease_id`, `indication`, `indication_type`, `source`

### `uniprot_proteins` (UniProt proteins)

**Required**: `gene_symbol`
**Any-of**: at least one of `uniprot_ac`, `accession`, `uniprot_id`
**Optional**: `protein_name`, `ncbi_gene_id`, `chromosome`, `sequence`, `function`, `organism`

### `string_ppi` (STRING PPI)

**Required**: `combined_score`
**Any-of**:
- At least one of `uniprot_ac_a`, `protein_a`, `uniprot_id_a`, `string_id_a`
- At least one of `uniprot_ac_b`, `protein_b`, `uniprot_id_b`, `string_id_b`
- At least one of `score`, `combined_score`

### `disgenet_gda` (DisGeNET GDA)

**Required**: `gene_symbol`, `disease_id`, `score`
**Any-of**: at least one of `gene_id` OR `ncbi_gene_id`
**Optional**: `disease_name`, `source`, `year`

### `omim_gda` (OMIM Mendelian)

**Required**: `gene_mim`, `gene_symbol`, `disease_id`, `disease_name`
**Optional**: `is_susceptibility` (always False for this file)

### `omim_susceptibility` (OMIM complex-disease)

**Required**: `gene_mim`, `gene_symbol`, `disease_id`, `disease_name`
**Optional**: `is_susceptibility` (always True for this file)

### `pubchem_enrichment` (PubChem properties)

**Required**: `inchikey`, `canonical_smiles`
**Optional**: `pubchem_cid`, `iupac_name`, `molecular_weight`, `molecular_formula`, `logp`, `tpsa`, `h_bond_donor_count`, `h_bond_acceptor_count`, `rotatable_bond_count`, `heavy_atom_count`, `complexity`

---

## Validation

The `validate_output.py` script runs 5 checks per CSV:

1. **File existence** — at least one candidate (filename or alias) must exist.
2. **Required columns** — every `required_columns` entry must be present.
3. **Any-of groups** — for each group, at least one column must be present.
4. **Non-nullable check** — non-nullable required columns must have zero NULLs.
5. **Row count** — `len(df) >= min_rows`.
6. **Dtype compatibility** — soft warning if a column's dtype is grossly mismatched.

### CLI usage

```bash
python -m phase1.contracts.validate_output phase1/processed_data
python -m phase1.contracts.validate_output phase1/processed_data --fail-on-warning
python -m phase1.contracts.validate_output phase1/processed_data -v
```

Exit codes:
- `0` — all sources valid (warnings allowed).
- `1` — at least one ERROR issue.
- `2` — `--fail-on-warning` set and at least one WARNING issue.

### Python API usage (Airflow DAG final task)

```python
from pathlib import Path
from phase1.contracts.validate_output import validate_output_dir

def airflow_final_task(**context):
    exit_code = validate_output_dir(Path("phase1/processed_data"))
    if exit_code != 0:
        raise SystemExit(exit_code)
```

---

## Phase 2 integration

Phase 2's bridge (`phase2/drugos_graph/phase1_bridge.py`) imports this contract:

```python
from phase1.contracts.phase1_schema import PHASE1_OUTPUT_SCHEMA, get_required_columns

required = get_required_columns("drugs")  # ["name", "inchikey"]
```

The bridge's legacy `_PHASE1_EXPECTED_COLUMNS` dict is a DERIVED copy that MUST stay in sync with this contract. The `test_schema_contract.py` test (Task 17) verifies they match at CI time.

---

## Change management

**If you need to add/rename/remove a column:**

1. Update `phase1/contracts/phase1_schema.py` (this is the canonical change).
2. Update the producer (the Phase 1 pipeline that emits the CSV).
3. Update the consumer (the Phase 2 bridge code that reads the CSV).
4. Run `python -m phase1.contracts.validate_output phase1/processed_data` to verify.
5. Run `pytest phase1/tests/test_schema_contract.py` to verify the bridge stays in sync.
6. Update this README if the change is user-visible.

**Never** update one without the other. The contract exists precisely to prevent the silent-schema-drift class of bugs that previously corrupted the KG.
