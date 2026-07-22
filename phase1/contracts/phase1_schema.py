"""Canonical Phase 1 output schema — the SINGLE source of truth.

TM1 TASK 7 ROOT FIX (Team Member 1, Phase 1 Real Data Pipeline):
  This module defines the canonical CSV column names, dtypes, and
  constraints for ALL 11 Phase 1 output files. Phase 2 (the bridge)
  imports this module instead of maintaining a divergent
  ``_PHASE1_EXPECTED_COLUMNS`` dict — eliminating the schema drift
  that previously caused silent data loss when Phase 1 changed a
  column name and Phase 2 didn't.

The 11 Phase 1 output files
---------------------------
  1.  chembl_drugs.csv                       (chembl_drugs)
  2.  chembl_activities_clean.csv            (chembl_activities)
  3.  drugbank_drugs.csv                     (drugs)
  4.  drugbank_interactions.csv[.gz]         (interactions)
  5.  drugbank_indications.csv               (indications)
  6.  uniprot_proteins.csv                   (uniprot_proteins)
  7.  string_protein_protein_interactions.csv (string_ppi)
  8.  disgenet_gene_disease_associations.csv  (disgenet_gda)
  9.  omim_gene_disease_associations.csv     (omim_gda)
  10. omim_gene_disease_susceptibility.csv   (omim_susceptibility)
  11. pubchem_enrichment.csv                 (pubchem_enrichment)

Each source declares:
  - ``required_columns``: columns that MUST be present (non-empty).
  - ``any_of_groups``: lists of column names where AT LEAST ONE must
    be present (e.g. ``["gene_id", "ncbi_gene_id"]`` — accept either).
  - ``optional_columns``: columns that MAY be present (enrichment).
  - ``dtypes``: pandas dtype hints for column coercion.
  - ``min_rows``: minimum row count for the source to be considered
    "populated" (1 = at least the header + 1 row; 0 = header-only OK).

This module is IMPORTED BY PHASE 2. The bridge's
``_PHASE1_EXPECTED_COLUMNS`` dict in ``phase1_bridge.py`` is a
DERIVED copy that must stay in sync with this module. The
``validate_output.py`` script (Task 8) enforces this at CI time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# ColumnSpec — typed specification for one Phase 1 output column
# =============================================================================


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single Phase 1 output column.

    Attributes
    ----------
    name : str
        Canonical column name (case-sensitive, exact match required).
    dtype : str
        Pandas dtype hint: ``"string"``, ``"float64"``, ``"int64"``,
        ``"bool"``, ``"object"``. Used by ``validate_output.py`` to
        coerce and verify column types.
    nullable : bool
        True if the column may contain NULL/NaN values. False means
        every row must have a non-null value (validated at read time).
    description : str
        Human-readable description of the column's semantic meaning.
        Surfaced in error messages and in ``contracts/README.md``.
    """

    name: str
    dtype: str = "string"
    nullable: bool = True
    description: str = ""


# =============================================================================
# SourceSpec — typed specification for one Phase 1 source CSV
# =============================================================================


@dataclass(frozen=True)
class SourceSpec:
    """Specification for one Phase 1 source CSV file.

    Attributes
    ----------
    key : str
        Canonical source key (e.g. ``"drugs"``, ``"chembl_drugs"``).
        Matches the keys used by the Phase 2 bridge.
    filename : str
        Canonical CSV filename (e.g. ``"drugbank_drugs.csv"``).
    aliases : tuple of str
        Alternate filenames the bridge accepts (e.g. ``"drugs.csv"``
        is an alias for ``"chembl_drugs.csv"`` when DrugBank is absent).
    required_columns : tuple of ColumnSpec
        Columns that MUST be present in the CSV. A missing required
        column is a HARD validation failure.
    any_of_groups : tuple of tuple of str
        Groups of column names where AT LEAST ONE per group must be
        present. E.g. ``(("gene_id", "ncbi_gene_id"),)`` accepts the
        CSV if either ``gene_id`` OR ``ncbi_gene_id`` is present.
    optional_columns : tuple of ColumnSpec
        Columns that MAY be present. Their absence is not an error,
        but if present they must match the declared dtype.
    min_rows : int
        Minimum row count (excluding header) for the source to be
        considered "populated". 0 means header-only is acceptable
        (e.g. DrugBank when academic license is paused). 1+ means
        real data rows are required.
    description : str
        Human-readable description of the source.
    """

    key: str
    filename: str
    aliases: tuple = ()
    required_columns: tuple = ()
    any_of_groups: tuple = ()
    optional_columns: tuple = ()
    min_rows: int = 0
    description: str = ""


# =============================================================================
# ValidationIssue — typed result of a single column-level check
# =============================================================================


@dataclass
class ValidationIssue:
    """One validation issue found in a Phase 1 output CSV.

    Attributes
    ----------
    source : str
        Source key (e.g. ``"drugs"``).
    severity : str
        ``"error"`` (hard fail) or ``"warning"`` (soft fail).
    code : str
        Machine-readable issue code (e.g. ``"missing_required_column"``,
        ``"any_of_group_unsatisfied"``, ``"dtype_mismatch"``,
        ``"below_min_rows"``).
    message : str
        Human-readable error message.
    column : str, optional
        Column name involved (if applicable).
    """

    source: str
    severity: str
    code: str
    message: str
    column: Optional[str] = None

    def __str__(self) -> str:
        col_part = f" column={self.column!r}" if self.column else ""
        return (
            f"[{self.severity.upper()}] source={self.source!r} "
            f"code={self.code!r}{col_part} :: {self.message}"
        )


# =============================================================================
# THE CANONICAL PHASE 1 OUTPUT SCHEMA
# =============================================================================
# This is the SINGLE source of truth. Phase 2's bridge imports this dict
# and MUST NOT maintain a divergent copy. The ``validate_output.py``
# script (Task 8) verifies the bridge stays in sync.

PHASE1_OUTPUT_SCHEMA: Dict[str, SourceSpec] = {

    # -------------------------------------------------------------------------
    # 1. ChEMBL drugs (chembl_drugs.csv)
    #    Backbone source for Compound nodes when DrugBank is unavailable.
    #    Produced by: phase1/pipelines/chembl_pipeline.py
    # -------------------------------------------------------------------------
    "chembl_drugs": SourceSpec(
        key="chembl_drugs",
        filename="chembl_drugs.csv",
        aliases=("drugs.csv",),
        required_columns=(
            ColumnSpec("chembl_id", "string", nullable=False,
                       description="ChEMBL molecule ID (CHEMBL\\d+)."),
            ColumnSpec("inchikey", "string", nullable=False,
                       description="27-char InChIKey (canonical compound key)."),
        ),
        any_of_groups=(),
        optional_columns=(
            ColumnSpec("name", "string", nullable=False,
                       description="Preferred drug name (e.g. 'Aspirin')."),
            ColumnSpec("smiles", "string", nullable=True,
                       description="Canonical SMILES string."),
            ColumnSpec("molecular_weight", "float64", nullable=True,
                       description="Molecular weight in Daltons."),
            ColumnSpec("max_phase", "int64", nullable=True,
                       description="ChEMBL max phase (0-4). 4 = globally approved."),
            ColumnSpec("is_fda_approved", "bool", nullable=True,
                       description="True=FDA-approved, False=not, None=unknown."),
            ColumnSpec("is_globally_approved", "bool", nullable=True,
                       description="True if max_phase==4 (any regulator)."),
            ColumnSpec("indication", "string", nullable=True,
                       description="Free-text indication."),
            ColumnSpec("indication_source", "string", nullable=True,
                       description="Source of indication field."),
            ColumnSpec("mechanism_of_action", "string", nullable=True,
                       description="Mechanism of action text."),
        ),
        min_rows=1,
        description="ChEMBL FDA-approved drugs (Compound source).",
    ),

    # -------------------------------------------------------------------------
    # 2. ChEMBL activities (chembl_activities_clean.csv)
    #    Source for Compound->inhibits/activates->Protein edges.
    #    Produced by: phase1/pipelines/chembl_pipeline.py
    # -------------------------------------------------------------------------
    "chembl_activities": SourceSpec(
        key="chembl_activities",
        filename="chembl_activities_clean.csv",
        aliases=("chembl_activities.csv",),
        required_columns=(
            ColumnSpec("molecule_chembl_id", "string", nullable=False,
                       description="ChEMBL molecule ID (foreign key to chembl_drugs)."),
            ColumnSpec("target_chembl_id", "string", nullable=False,
                       description="ChEMBL target ID."),
            ColumnSpec("pchembl_value", "float64", nullable=True,
                       description="-log10(activity_value in M). Higher = more potent."),
            ColumnSpec("standard_relation", "string", nullable=True,
                       description="'=', '<', '>' — relation between activity and value."),
        ),
        any_of_groups=(),
        optional_columns=(
            ColumnSpec("uniprot_id", "string", nullable=True,
                       description="UniProt accession for the target protein."),
            ColumnSpec("target_name", "string", nullable=True,
                       description="Human-readable target name."),
            ColumnSpec("activity_type", "string", nullable=True,
                       description="IC50, Ki, EC50, etc."),
            ColumnSpec("activity_value", "float64", nullable=True,
                       description="Activity value in nM."),
            ColumnSpec("activity_units", "string", nullable=True,
                       description="Units of activity_value (typically 'nM')."),
            ColumnSpec("chembl_id", "string", nullable=True,
                       description="Alias for molecule_chembl_id (legacy)."),
        ),
        min_rows=1,
        description="ChEMBL bioactivity measurements (Compound-Protein edges).",
    ),

    # -------------------------------------------------------------------------
    # 3. DrugBank drugs (drugbank_drugs.csv)
    #    Preferred Compound source when DrugBank academic license is active.
    #    Produced by: phase1/pipelines/drugbank_pipeline.py
    #    NOTE: when DrugBank is unavailable, this CSV may be EMPTY (header-only)
    #    — the bridge degrades to chembl_drugs.csv. min_rows=0 reflects this.
    # -------------------------------------------------------------------------
    "drugs": SourceSpec(
        key="drugs",
        filename="drugbank_drugs.csv",
        aliases=("drugbank_open_drugs.csv", "chembl_drugs.csv", "drugs.csv"),
        required_columns=(
            ColumnSpec("name", "string", nullable=False,
                       description="Drug name (preferred form)."),
            ColumnSpec("inchikey", "string", nullable=False,
                       description="27-char InChIKey (canonical compound key)."),
        ),
        any_of_groups=(
            # Accept either drugbank_id (when DrugBank is the source) OR
            # chembl_id (when ChEMBL is the source). The bridge uses
            # inchikey as the canonical Compound key; drugbank_id/chembl_id
            # are source-specific aliases.
            ("drugbank_id", "chembl_id"),
        ),
        optional_columns=(
            ColumnSpec("drugbank_id", "string", nullable=True,
                       description="DrugBank ID (DB\\d+)."),
            ColumnSpec("chembl_id", "string", nullable=True,
                       description="ChEMBL ID (CHEMBL\\d+)."),
            ColumnSpec("pubchem_cid", "int64", nullable=True,
                       description="PubChem Compound ID."),
            ColumnSpec("smiles", "string", nullable=True,
                       description="Canonical SMILES."),
            ColumnSpec("molecular_weight", "float64", nullable=True,
                       description="Molecular weight (Daltons)."),
            ColumnSpec("molecular_formula", "string", nullable=True,
                       description="Molecular formula (e.g. C9H8O4)."),
            ColumnSpec("indication", "string", nullable=True,
                       description="FDA-approved indication text."),
            ColumnSpec("indication_source", "string", nullable=True,
                       description="Source of indication (FDA/EMA/manual)."),
            ColumnSpec("mechanism_of_action", "string", nullable=True,
                       description="Mechanism of action."),
            ColumnSpec("groups", "string", nullable=True,
                       description="DrugBank groups (approved/illicit/withdrawn/nutracet)."),
            ColumnSpec("is_fda_approved", "bool", nullable=True,
                       description="True=FDA-approved, False=not, None=unknown."),
            ColumnSpec("is_globally_approved", "bool", nullable=True,
                       description="True if approved by any regulator globally."),
            ColumnSpec("is_withdrawn", "bool", nullable=True,
                       description="True if withdrawn from market (patient-safety)."),
            ColumnSpec("clinical_status", "string", nullable=True,
                       description="Clinical trial status."),
            ColumnSpec("max_phase", "int64", nullable=True,
                       description="Max clinical phase (0-4)."),
            ColumnSpec("drug_type", "string", nullable=True,
                       description="small_molecule / biotech / etc."),
            ColumnSpec("cas_number", "string", nullable=True,
                       description="CAS registry number."),
            ColumnSpec("logp", "float64", nullable=True,
                       description="Computed logP."),
            ColumnSpec("tpsa", "float64", nullable=True,
                       description="Topological polar surface area."),
        ),
        min_rows=0,
        description="DrugBank drugs (preferred Compound source; empty when license paused).",
    ),

    # -------------------------------------------------------------------------
    # 4. DrugBank interactions (drugbank_interactions.csv[.gz])
    #    Source for DrugBank-derived Drug->Drug interaction edges.
    #    Produced by: phase1/pipelines/drugbank_pipeline.py
    # -------------------------------------------------------------------------
    "interactions": SourceSpec(
        key="interactions",
        filename="drugbank_interactions.csv",
        aliases=("drugbank_interactions.csv.gz", "drugbank_open_interactions.csv",
                 "chembl_activities_clean.csv", "chembl_activities.csv"),
        required_columns=(
            ColumnSpec("drugbank_id", "string", nullable=False,
                       description="Source DrugBank ID."),
            ColumnSpec("uniprot_id", "string", nullable=False,
                       description="Target UniProt accession."),
            ColumnSpec("action_type", "string", nullable=True,
                       description="inhibitor/activator/antagonist/etc."),
        ),
        any_of_groups=(),
        optional_columns=(
            ColumnSpec("target_name", "string", nullable=True,
                       description="Target protein name."),
            ColumnSpec("target_chembl_id", "string", nullable=True,
                       description="ChEMBL target ID alias."),
        ),
        min_rows=0,
        description="DrugBank drug-protein interactions (may degrade to ChEMBL activities).",
    ),

    # -------------------------------------------------------------------------
    # 5. DrugBank indications (drugbank_indications.csv)
    #    Source for ClinicalOutcome nodes + Compound-treats-Disease edges.
    #    Produced by: phase1/pipelines/drugbank_pipeline.py
    # -------------------------------------------------------------------------
    "indications": SourceSpec(
        key="indications",
        filename="drugbank_indications.csv",
        aliases=(),
        required_columns=(
            ColumnSpec("drugbank_id", "string", nullable=False,
                       description="DrugBank ID of the treating drug."),
            ColumnSpec("disease_id", "string", nullable=False,
                       description="Disease identifier (DOID/MESH/ICD10)."),
        ),
        any_of_groups=(),
        optional_columns=(
            ColumnSpec("drug_inchikey", "string", nullable=True,
                       description="InChIKey of the treating drug."),
            ColumnSpec("drug_name", "string", nullable=True,
                       description="Name of the treating drug."),
            ColumnSpec("disease_name", "string", nullable=True,
                       description="Human-readable disease name."),
            ColumnSpec("doid_id", "string", nullable=True,
                       description="Disease Ontology ID."),
            ColumnSpec("omim_disease_id", "string", nullable=True,
                       description="OMIM disease ID."),
            ColumnSpec("indication", "string", nullable=True,
                       description="Free-text indication."),
            ColumnSpec("indication_type", "string", nullable=True,
                       description="approved/off_label/etc."),
            ColumnSpec("source", "string", nullable=True,
                       description="Source of the indication mapping."),
        ),
        min_rows=0,
        description="DrugBank drug-indication mappings (ClinicalOutcome source).",
    ),

    # -------------------------------------------------------------------------
    # 6. UniProt proteins (uniprot_proteins.csv)
    #    Source for Protein nodes.
    #    Produced by: phase1/pipelines/uniprot_pipeline.py
    # -------------------------------------------------------------------------
    "uniprot_proteins": SourceSpec(
        key="uniprot_proteins",
        filename="uniprot_proteins.csv",
        aliases=("proteins.csv",),
        required_columns=(
            ColumnSpec("gene_symbol", "string", nullable=True,
                       description="HGNC gene symbol (e.g. 'PTGS2'). May be null for non-human."),
        ),
        any_of_groups=(
            # Bridge accepts uniprot_ac OR accession OR uniprot_id.
            ("uniprot_ac", "accession", "uniprot_id"),
        ),
        optional_columns=(
            ColumnSpec("uniprot_id", "string", nullable=True,
                       description="Canonical UniProt accession."),
            ColumnSpec("uniprot_ac", "string", nullable=True,
                       description="Alias for uniprot_id (legacy)."),
            ColumnSpec("accession", "string", nullable=True,
                       description="Alias for uniprot_id (legacy)."),
            ColumnSpec("protein_name", "string", nullable=True,
                       description="Full protein name."),
            ColumnSpec("ncbi_gene_id", "int64", nullable=True,
                       description="NCBI Gene ID."),
            ColumnSpec("chromosome", "string", nullable=True,
                       description="Chromosomal location."),
            ColumnSpec("sequence", "string", nullable=True,
                       description="Amino acid sequence."),
            ColumnSpec("function", "string", nullable=True,
                       description="Functional description."),
            ColumnSpec("organism", "string", nullable=True,
                       description="Source organism (e.g. 'Homo sapiens')."),
        ),
        min_rows=1,
        description="UniProt proteins (Protein node source).",
    ),

    # -------------------------------------------------------------------------
    # 7. STRING PPI (string_protein_protein_interactions.csv)
    #    Source for Pathway nodes (connected components) + Protein->Protein edges.
    #    Produced by: phase1/pipelines/string_pipeline.py
    # -------------------------------------------------------------------------
    "string_ppi": SourceSpec(
        key="string_ppi",
        filename="string_protein_protein_interactions.csv",
        aliases=("protein_protein_interactions.csv",),
        required_columns=(
            ColumnSpec("combined_score", "int64", nullable=True,
                       description="STRING combined confidence score (0-1000)."),
        ),
        any_of_groups=(
            # Bridge accepts uniprot_ac_a OR protein_a OR uniprot_id_a OR string_id_a.
            ("uniprot_ac_a", "protein_a", "uniprot_id_a", "string_id_a"),
            ("uniprot_ac_b", "protein_b", "uniprot_id_b", "string_id_b"),
            # Bridge accepts score OR combined_score.
            ("score", "combined_score"),
        ),
        optional_columns=(
            ColumnSpec("string_id_a", "string", nullable=True,
                       description="STRING ENSP ID for protein A."),
            ColumnSpec("string_id_b", "string", nullable=True,
                       description="STRING ENSP ID for protein B."),
            ColumnSpec("uniprot_id_a", "string", nullable=True,
                       description="UniProt accession for protein A (after crosswalk)."),
            ColumnSpec("uniprot_id_b", "string", nullable=True,
                       description="UniProt accession for protein B (after crosswalk)."),
            ColumnSpec("protein_a", "string", nullable=True,
                       description="Alias for string_id_a / uniprot_id_a."),
            ColumnSpec("protein_b", "string", nullable=True,
                       description="Alias for string_id_b / uniprot_id_b."),
            ColumnSpec("score", "int64", nullable=True,
                       description="Alias for combined_score."),
        ),
        min_rows=1,
        description="STRING protein-protein interactions (Pathway source).",
    ),

    # -------------------------------------------------------------------------
    # 8. DisGeNET GDA (disgenet_gene_disease_associations.csv)
    #    Source for Gene->associated_with->Disease edges.
    #    Produced by: phase1/pipelines/disgenet_pipeline.py
    # -------------------------------------------------------------------------
    "disgenet_gda": SourceSpec(
        key="disgenet_gda",
        filename="disgenet_gene_disease_associations.csv",
        aliases=("gene_disease_associations.csv",),
        required_columns=(
            ColumnSpec("gene_symbol", "string", nullable=False,
                       description="HGNC gene symbol."),
            ColumnSpec("disease_id", "string", nullable=False,
                       description="Disease identifier (CUI/DOID/MESH)."),
            ColumnSpec("score", "float64", nullable=True,
                       description="DisGeNET confidence score [0,1]."),
        ),
        any_of_groups=(
            # Bridge accepts gene_id OR ncbi_gene_id.
            ("gene_id", "ncbi_gene_id"),
        ),
        optional_columns=(
            ColumnSpec("gene_id", "int64", nullable=True,
                       description="NCBI Gene ID."),
            ColumnSpec("ncbi_gene_id", "int64", nullable=True,
                       description="Alias for gene_id."),
            ColumnSpec("disease_name", "string", nullable=True,
                       description="Human-readable disease name."),
            ColumnSpec("source", "string", nullable=True,
                       description="DisGeNET source (CURATED/BEFREE/etc.)."),
            ColumnSpec("year", "int64", nullable=True,
                       description="Year of first association."),
        ),
        min_rows=1,
        description="DisGeNET gene-disease associations (Gene-Disease edge source).",
    ),

    # -------------------------------------------------------------------------
    # 9. OMIM GDA (omim_gene_disease_associations.csv)
    #    Source for Gene->causes->Disease edges (Mendelian).
    #    Produced by: phase1/pipelines/omim_pipeline.py
    # -------------------------------------------------------------------------
    "omim_gda": SourceSpec(
        key="omim_gda",
        filename="omim_gene_disease_associations.csv",
        aliases=(),
        required_columns=(
            ColumnSpec("gene_mim", "string", nullable=False,
                       description="OMIM gene MIM number."),
            ColumnSpec("gene_symbol", "string", nullable=False,
                       description="HGNC gene symbol."),
            ColumnSpec("disease_id", "string", nullable=False,
                       description="Disease MIM number."),
            ColumnSpec("disease_name", "string", nullable=False,
                       description="Human-readable disease name."),
        ),
        any_of_groups=(),
        optional_columns=(
            ColumnSpec("is_susceptibility", "bool", nullable=True,
                       description="True if susceptibility (not causal)."),
        ),
        min_rows=1,
        description="OMIM gene-disease associations (Mendelian).",
    ),

    # -------------------------------------------------------------------------
    # 10. OMIM susceptibility (omim_gene_disease_susceptibility.csv)
    #     Source for Gene->predisposes->Disease edges (complex disease).
    #     Produced by: phase1/pipelines/omim_pipeline.py
    # -------------------------------------------------------------------------
    "omim_susceptibility": SourceSpec(
        key="omim_susceptibility",
        filename="omim_gene_disease_susceptibility.csv",
        aliases=(),
        required_columns=(
            ColumnSpec("gene_mim", "string", nullable=False,
                       description="OMIM gene MIM number."),
            ColumnSpec("gene_symbol", "string", nullable=False,
                       description="HGNC gene symbol."),
            ColumnSpec("disease_id", "string", nullable=False,
                       description="Disease MIM number."),
            ColumnSpec("disease_name", "string", nullable=False,
                       description="Human-readable disease name."),
        ),
        any_of_groups=(),
        optional_columns=(
            ColumnSpec("is_susceptibility", "bool", nullable=True,
                       description="Always True for this file."),
        ),
        min_rows=0,
        description="OMIM susceptibility (complex disease) — may be empty if no suscept associations.",
    ),

    # -------------------------------------------------------------------------
    # 11. PubChem enrichment (pubchem_enrichment.csv)
    #     Source for additional Compound properties (logP, TPSA, etc.).
    #     Produced by: phase1/pipelines/pubchem_pipeline.py
    # -------------------------------------------------------------------------
    "pubchem_enrichment": SourceSpec(
        key="pubchem_enrichment",
        filename="pubchem_enrichment.csv",
        aliases=(),
        required_columns=(
            ColumnSpec("inchikey", "string", nullable=False,
                       description="27-char InChIKey (foreign key to drugs)."),
            ColumnSpec("canonical_smiles", "string", nullable=True,
                       description="PubChem canonical SMILES."),
        ),
        any_of_groups=(),
        optional_columns=(
            ColumnSpec("pubchem_cid", "int64", nullable=True,
                       description="PubChem Compound ID."),
            ColumnSpec("iupac_name", "string", nullable=True,
                       description="IUPAC systematic name."),
            ColumnSpec("molecular_weight", "float64", nullable=True,
                       description="Molecular weight (Daltons)."),
            ColumnSpec("molecular_formula", "string", nullable=True,
                       description="Molecular formula."),
            ColumnSpec("logp", "float64", nullable=True,
                       description="Computed XLogP."),
            ColumnSpec("tpsa", "float64", nullable=True,
                       description="Topological polar surface area."),
            ColumnSpec("h_bond_donor_count", "int64", nullable=True,
                       description="H-bond donor count."),
            ColumnSpec("h_bond_acceptor_count", "int64", nullable=True,
                       description="H-bond acceptor count."),
            ColumnSpec("rotatable_bond_count", "int64", nullable=True,
                       description="Rotatable bond count."),
            ColumnSpec("heavy_atom_count", "int64", nullable=True,
                       description="Heavy atom count."),
            ColumnSpec("complexity", "float64", nullable=True,
                       description="PubChem complexity score."),
        ),
        min_rows=0,
        description="PubChem enrichment data (additional Compound properties).",
    ),
}


# =============================================================================
# Convenience: flat dict of canonical CSV filenames
# =============================================================================

PHASE1_CSV_FILENAMES: Dict[str, str] = {
    spec.key: spec.filename for spec in PHASE1_OUTPUT_SCHEMA.values()
}


# =============================================================================
# Convenience accessors
# =============================================================================


def get_required_columns(source_key: str) -> List[str]:
    """Return the list of required column names for ``source_key``."""
    spec = PHASE1_OUTPUT_SCHEMA[source_key]
    return [c.name for c in spec.required_columns]


def get_any_of_groups(source_key: str) -> List[List[str]]:
    """Return the list of any-of column groups for ``source_key``."""
    spec = PHASE1_OUTPUT_SCHEMA[source_key]
    return [list(group) for group in spec.any_of_groups]


def get_optional_columns(source_key: str) -> List[str]:
    """Return the list of optional column names for ``source_key``."""
    spec = PHASE1_OUTPUT_SCHEMA[source_key]
    return [c.name for c in spec.optional_columns]


def get_all_aliases(source_key: str) -> List[str]:
    """Return [filename] + list of aliases for ``source_key``."""
    spec = PHASE1_OUTPUT_SCHEMA[source_key]
    return [spec.filename] + list(spec.aliases)
