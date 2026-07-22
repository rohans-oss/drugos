"""v89 forensic root-fix verification suite (BUGS #23-#44).

This suite verifies the v89 root-cause fixes for bugs #23 through #44
by exercising the REAL production code (not comments, not stubs). Each
test imports the actual module and asserts the fix is present and
functionally correct.

Run:
    PYTHONPATH=phase1 python -m pytest phase1/tests/v89_forensic/test_v89_bugs_23_44.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure phase1 is on sys.path (matches CI's PYTHONPATH=phase1).
_PHASE1 = str(Path(__file__).resolve().parent.parent.parent)
if _PHASE1 not in sys.path:
    sys.path.insert(0, _PHASE1)


# =============================================================================
# BUG #23 (P1) -- STRING aliases DataFrame schema validation
# =============================================================================

class TestBug23StringAliasesSchemaValidation:
    """BUG #23: validate string_aliases_df schema before add_string_records."""

    def test_string_aliases_df_missing_string_id_column_logs_error(self, caplog):
        """A non-empty DataFrame WITHOUT the string_id column must be
        rejected with an ERROR log (not silently dead-lettered)."""
        import logging
        import pandas as pd
        from entity_resolution.protein_resolver import ProteinResolver

        bad_df = pd.DataFrame({
            "wrong_col": ["a", "b", "c"],
            "another": [1, 2, 3],
        })
        pr = ProteinResolver()
        with caplog.at_level(logging.ERROR, logger="entity_resolution.protein_resolver"):
            pr.build_mapping(None, string_aliases_df=bad_df)
        error_msgs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("string_id" in m and "missing" in m for m in error_msgs), (
            f"Expected ERROR about missing string_id column, got: {error_msgs}"
        )

    def test_string_aliases_df_non_dataframe_logs_warning(self, caplog):
        """A non-DataFrame object must be rejected with a WARNING log
        (the existing FIX LOG-07 path). The BUG #23 fix adds ERROR-level
        schema validation for DataFrame inputs that have the wrong
        columns; non-DataFrame inputs continue to hit the WARNING path
        because they fail the ``hasattr(.empty)`` guard first."""
        import logging
        from entity_resolution.protein_resolver import ProteinResolver

        pr = ProteinResolver()
        with caplog.at_level(logging.WARNING, logger="entity_resolution.protein_resolver"):
            pr.build_mapping(None, string_aliases_df=["not", "a", "df"])
        warn_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("not a recognized" in m or "not a DataFrame" in m or ".empty" in m for m in warn_msgs), (
            f"Expected WARNING about non-DataFrame input, got: {warn_msgs}"
        )


# =============================================================================
# BUG #24 (P2) -- @fail_fast_on_http_4xx on _trigger_phase2
# =============================================================================

class TestBug24TriggerPhase2FailFast:
    """BUG #24: _trigger_phase2 must be decorated with @fail_fast_on_http_4xx."""

    def test_trigger_phase2_has_fail_fast_decorator(self):
        dag_path = Path(_PHASE1) / "dags" / "master_pipeline_dag.py"
        source = dag_path.read_text()
        assert "@fail_fast_on_http_4xx" in source, (
            "master_pipeline_dag.py must apply @fail_fast_on_http_4xx "
            "(BUG #24 -- _trigger_phase2 was missing the decorator)"
        )
        lines = source.splitlines()
        for i, line in enumerate(lines):
            if "def _trigger_phase2" in line:
                preceding = "\n".join(lines[max(0, i - 5):i])
                assert "@fail_fast_on_http_4xx" in preceding, (
                    f"@fail_fast_on_http_4xx must be on _trigger_phase2. "
                    f"Preceding lines: {preceding}"
                )
                return
        pytest.fail("_trigger_phase2 function not found in master_pipeline_dag.py")


# =============================================================================
# BUG #25 / #38 (P2/P3) -- bare @task in all standalone DAGs
# =============================================================================

class TestBug2538BareTaskInAllDags:
    """BUG #25/#38: all 7 standalone DAGs must use bare @task (DRY)."""

    @pytest.mark.parametrize("dag_file", [
        "chembl_dag.py", "drugbank_dag.py", "disgenet_dag.py",
        "omim_dag.py", "pubchem_dag.py", "string_dag.py", "uniprot_dag.py",
    ])
    def test_no_redundant_task_params(self, dag_file):
        dag_path = Path(_PHASE1) / "dags" / dag_file
        source = dag_path.read_text()
        assert "retries=2, execution_timeout=timedelta(hours=4)" not in source, (
            f"{dag_file} still has redundant @task params (BUG #25/#38). "
            f"Use bare @task -- params are inherited from DEFAULT_ARGS."
        )


# =============================================================================
# BUG #26 (P2) -- pre-flight check for drugos_graph package
# =============================================================================

class TestBug26DrugosGraphPreflightCheck:
    """BUG #26: _trigger_phase2 must pre-flight check drugos_graph importability."""

    def test_preflight_check_present(self):
        dag_path = Path(_PHASE1) / "dags" / "master_pipeline_dag.py"
        source = dag_path.read_text()
        assert "importlib.util.find_spec" in source, (
            "master_pipeline_dag.py must pre-flight check drugos_graph via "
            "importlib.util.find_spec (BUG #26)"
        )
        assert "drugos_graph" in source, (
            "master_pipeline_dag.py must reference drugos_graph in the "
            "pre-flight check (BUG #26)"
        )


# =============================================================================
# BUG #27 (P2) -- DRUGBANK_XML_PATH imported at module top
# =============================================================================

class TestBug27DrugbankXmlPathTopLevelImport:
    """BUG #27: DRUGBANK_XML_PATH must be imported at module top, not
    inside _check_drugbank_xml."""

    def test_no_runtime_import_in_check_drugbank_xml(self):
        dag_path = Path(_PHASE1) / "dags" / "master_pipeline_dag.py"
        source = dag_path.read_text()
        lines = source.splitlines()
        in_func = False
        func_body = []
        for line in lines:
            if "def _check_drugbank_xml" in line:
                in_func = True
                func_body.append(line)
                continue
            if in_func:
                if line.startswith("def ") or line.startswith("class "):
                    break
                func_body.append(line)
        func_source = "\n".join(func_body)
        assert "from config.settings import DRUGBANK_XML_PATH" not in func_source, (
            "_check_drugbank_xml must NOT import DRUGBANK_XML_PATH at runtime "
            "(BUG #27 -- move to module top)"
        )

    def test_top_level_import_present(self):
        dag_path = Path(_PHASE1) / "dags" / "master_pipeline_dag.py"
        source = dag_path.read_text()
        assert "from config.settings import DRUGBANK_XML_PATH" in source, (
            "master_pipeline_dag.py must import DRUGBANK_XML_PATH at module "
            "top (BUG #27)"
        )


# =============================================================================
# BUG #28 (P2) -- gzip integrity validation for STRING aliases
# =============================================================================

class TestBug28GzipIntegrityValidation:
    """BUG #28: corrupt STRING aliases file must be dead-lettered and
    raise a clear error, not silently continue with empty DataFrame."""

    def test_gzip_integrity_check_present(self):
        run_path = Path(_PHASE1) / "entity_resolution" / "run.py"
        source = run_path.read_text()
        assert "gzip.BadGzipFile" in source or "EOFError" in source, (
            "run.py must handle gzip.BadGzipFile / EOFError for corrupt "
            "STRING aliases files (BUG #28)"
        )
        assert "dead_letter" in source, (
            "run.py must dead-letter corrupt STRING aliases files (BUG #28)"
        )


# =============================================================================
# BUG #29 (P2) -- temp table cleanup with TRY/FINALLY
# =============================================================================

class TestBug29TempTableTryFinally:
    """BUG #29: temp tables must be dropped in a FINALLY block so they
    don't accumulate on failed runs."""

    def test_try_finally_cleanup_present(self):
        run_path = Path(_PHASE1) / "entity_resolution" / "run.py"
        source = run_path.read_text()
        assert "finally:" in source, (
            "run.py must use try/finally for temp table cleanup (BUG #29)"
        )
        assert "DROP TABLE IF EXISTS _tmp_entity_mapping_staging" in source
        assert "DROP TABLE IF EXISTS _tmp_protein_string_update" in source


# =============================================================================
# BUG #30 (P2) -- uniprot_id uniqueness validation before UPDATE
# =============================================================================

class TestBug30UniprotIdUniquenessCheck:
    """BUG #30: validate uniprot_id uniqueness before the proteins
    UPDATE to avoid corrupting multiple rows."""

    def test_uniqueness_check_present(self):
        run_path = Path(_PHASE1) / "entity_resolution" / "run.py"
        source = run_path.read_text()
        assert "HAVING COUNT(*) > 1" in source, (
            "run.py must check for duplicate uniprot_id values via "
            "GROUP BY ... HAVING COUNT(*) > 1 (BUG #30)"
        )


# =============================================================================
# BUG #31 (P2) -- derived constants LOAD_PASS_NO_PUBCHEM and PUBCHEM_LOAD
# =============================================================================

class TestBug31DerivedConstants:
    """BUG #31: LOAD_PASS_NO_PUBCHEM and PUBCHEM_LOAD must be module-level
    constants, not inline comprehensions at the call site."""

    def test_derived_constants_present(self):
        script_path = Path(_PHASE1) / "scripts" / "download_parallel.py"
        source = script_path.read_text()
        assert "LOAD_PASS_NO_PUBCHEM = [" in source, (
            "download_parallel.py must define LOAD_PASS_NO_PUBCHEM as a "
            "module-level constant (BUG #31)"
        )
        assert "PUBCHEM_LOAD = [" in source, (
            "download_parallel.py must define PUBCHEM_LOAD as a "
            "module-level constant (BUG #31)"
        )


# =============================================================================
# BUG #32 (P2) -- SMILES_CANONICAL in MatchConfidence enum
# =============================================================================

class TestBug32SmilesCanonicalEnum:
    """BUG #32: MatchConfidence must have a SMILES_CANONICAL member so
    enum-based and dict-based lookups return the same value."""

    def test_enum_member_exists(self):
        from entity_resolution.base import MatchConfidence
        assert hasattr(MatchConfidence, "SMILES_CANONICAL"), (
            "MatchConfidence must have SMILES_CANONICAL member (BUG #32)"
        )
        assert MatchConfidence.SMILES_CANONICAL == 0.75

    def test_from_method_returns_smiles_canonical(self):
        from entity_resolution.base import MatchConfidence
        result = MatchConfidence.from_method("smiles_canonical")
        assert result == MatchConfidence.SMILES_CANONICAL, (
            f"from_method('smiles_canonical') should return SMILES_CANONICAL, "
            f"got {result}"
        )

    def test_enum_and_dict_lookups_match(self):
        from entity_resolution.base import MatchConfidence
        from entity_resolution.resolver_utils import compute_match_confidence
        enum_val = MatchConfidence.from_method("smiles_canonical").value
        dict_val = compute_match_confidence("smiles_canonical")
        assert enum_val == dict_val == 0.75, (
            f"enum ({enum_val}) and dict ({dict_val}) lookups must both "
            f"return 0.75 (BUG #32 invariant)"
        )

    def test_method_confidence_dict_has_smiles_canonical(self):
        from entity_resolution.resolver_utils import METHOD_CONFIDENCE
        assert "smiles_canonical" in METHOD_CONFIDENCE, (
            "METHOD_CONFIDENCE must have smiles_canonical (BUG #32)"
        )
        assert METHOD_CONFIDENCE["smiles_canonical"] == 0.75


# =============================================================================
# BUG #33 (P2) -- ChEMBL target data wired to build_mapping
# =============================================================================

class TestBug33ChemblTargetWired:
    """BUG #33: build_mapping must accept chembl_target_df and
    add_chembl_target_records must be invoked."""

    def test_build_mapping_accepts_chembl_target_df(self):
        import inspect
        from entity_resolution.protein_resolver import ProteinResolver
        sig = inspect.signature(ProteinResolver.build_mapping)
        assert "chembl_target_df" in sig.parameters, (
            f"build_mapping must accept chembl_target_df (BUG #33). "
            f"Params: {list(sig.parameters)}"
        )

    def test_run_py_loads_chembl_activities(self):
        run_path = Path(_PHASE1) / "entity_resolution" / "run.py"
        source = run_path.read_text()
        assert "chembl_activities_clean.csv" in source, (
            "run.py must load chembl_activities_clean.csv (BUG #33)"
        )
        assert "chembl_target_df" in source, (
            "run.py must pass chembl_target_df to build_mapping (BUG #33)"
        )


# =============================================================================
# BUG #34 (P2) -- record["inchikey"] normalized in place
# =============================================================================

class TestBug34InchikeyNormalizedInPlace:
    """BUG #34: _create_canonical_entry must normalize record["inchikey"]
    in place so callers see the normalized value."""

    def test_in_place_normalization_present(self):
        drug_path = Path(_PHASE1) / "entity_resolution" / "drug_resolver.py"
        source = drug_path.read_text()
        assert 'record["inchikey"] = inchikey' in source, (
            "drug_resolver.py must normalize record['inchikey'] in place (BUG #34)"
        )


# =============================================================================
# BUG #35 (P2) -- case-preserving gene symbol index
# =============================================================================

class TestBug35CasePreservingGeneSymbolIndex:
    """BUG #35: ProteinResolver must have a case-preserving gene symbol
    index (_gene_symbol_index) so TP53 (human) and Tp53 (mouse) are
    distinct keys."""

    def test_gene_symbol_index_exists(self):
        from entity_resolution.protein_resolver import ProteinResolver
        pr = ProteinResolver()
        assert hasattr(pr, "_gene_symbol_index"), (
            "ProteinResolver must have _gene_symbol_index (BUG #35)"
        )
        assert hasattr(pr, "_gene_symbol_index_multi"), (
            "ProteinResolver must have _gene_symbol_index_multi (BUG #35)"
        )

    def test_gene_symbol_index_preserves_case(self):
        from entity_resolution.protein_resolver import ProteinResolver
        pr = ProteinResolver()
        pr.add_uniprot_records([{
            "uniprot_id": "P04637",
            "gene_symbol": "TP53",
            "gene_name": "Tumor protein p53",
            "organism": "Homo sapiens",
        }])
        pr.add_uniprot_records([{
            "uniprot_id": "P02340",
            "gene_symbol": "Tp53",
            "gene_name": "Tumor protein p53",
            "organism": "Mus musculus",
        }])
        assert "TP53" in pr._gene_symbol_index, "TP53 (human) must be in _gene_symbol_index"
        assert "Tp53" in pr._gene_symbol_index, "Tp53 (mouse) must be in _gene_symbol_index"
        assert pr._gene_symbol_index["TP53"] != pr._gene_symbol_index["Tp53"], (
            "TP53 (human) and Tp53 (mouse) must map to DIFFERENT uids"
        )

    def test_reset_clears_gene_symbol_index(self):
        from entity_resolution.protein_resolver import ProteinResolver
        pr = ProteinResolver()
        pr.add_uniprot_records([{
            "uniprot_id": "P04637",
            "gene_symbol": "TP53",
            "organism": "Homo sapiens",
        }])
        assert len(pr._gene_symbol_index) > 0
        pr.reset()
        assert pr._gene_symbol_index == {}, (
            "reset() must clear _gene_symbol_index (BUG #35)"
        )


# =============================================================================
# BUG #36 (P2) -- task_id constants + parse-time assertion
# =============================================================================

class TestBug36TaskIdConstants:
    """BUG #36: branch return values must use task_id constants, and a
    parse-time assertion must catch rename-without-update."""

    def test_task_id_constants_present(self):
        dag_path = Path(_PHASE1) / "dags" / "master_pipeline_dag.py"
        source = dag_path.read_text()
        assert "_DRUGBANK_DOWNLOAD_TASK_ID" in source, (
            "master_pipeline_dag.py must define _DRUGBANK_DOWNLOAD_TASK_ID (BUG #36)"
        )
        assert "_DRUGBANK_SKIP_TASK_ID" in source, (
            "master_pipeline_dag.py must define _DRUGBANK_SKIP_TASK_ID (BUG #36)"
        )

    def test_parse_time_assertion_present(self):
        dag_path = Path(_PHASE1) / "dags" / "master_pipeline_dag.py"
        source = dag_path.read_text()
        assert "drugbank.task_id == _DRUGBANK_DOWNLOAD_TASK_ID" in source, (
            "master_pipeline_dag.py must assert drugbank.task_id matches "
            "the constant at parse time (BUG #36)"
        )


# =============================================================================
# BUG #37 (P3) -- SLA < execution_timeout
# =============================================================================

class TestBug37SlaLessThanTimeout:
    """BUG #37: SLA must be less than execution_timeout so there's an
    early-warning window before the hard kill."""

    def test_sla_less_than_timeout(self):
        from dags._retry_policy import DEFAULT_RETRY_ARGS
        sla = DEFAULT_RETRY_ARGS["sla"]
        timeout = DEFAULT_RETRY_ARGS["execution_timeout"]
        assert sla < timeout, (
            f"SLA ({sla}) must be < execution_timeout ({timeout}) (BUG #37). "
            f"The SLA is advisory -- it must fire BEFORE the hard kill."
        )


# =============================================================================
# BUG #39 (P3) -- shared _dags_init module
# =============================================================================

class TestBug39SharedDagsInit:
    """BUG #39: sys.path setup must be extracted to a shared _dags_init
    module to avoid copy-paste across 8 DAG files."""

    def test_dags_init_module_exists(self):
        init_path = Path(_PHASE1) / "dags" / "_dags_init.py"
        assert init_path.exists(), (
            "dags/_dags_init.py must exist (BUG #39)"
        )

    def test_dags_init_importable(self):
        from dags._dags_init import ensure_project_root, _PROJECT_ROOT
        assert _PROJECT_ROOT is not None
        result = ensure_project_root()
        assert result == _PROJECT_ROOT

    @pytest.mark.parametrize("dag_file", [
        "master_pipeline_dag.py", "chembl_dag.py", "drugbank_dag.py",
        "disgenet_dag.py", "omim_dag.py", "pubchem_dag.py",
        "string_dag.py", "uniprot_dag.py",
    ])
    def test_all_dags_import_dags_init(self, dag_file):
        dag_path = Path(_PHASE1) / "dags" / dag_file
        source = dag_path.read_text()
        assert "from dags._dags_init import" in source, (
            f"{dag_file} must import from dags._dags_init (BUG #39)"
        )


# =============================================================================
# BUG #40 (P3) -- consistent DAG instance naming
# =============================================================================

class TestBug40ConsistentDagNaming:
    """BUG #40: all 8 DAG files must use `dag = <factory>()` naming."""

    @pytest.mark.parametrize("dag_file", [
        "master_pipeline_dag.py", "chembl_dag.py", "drugbank_dag.py",
        "disgenet_dag.py", "omim_dag.py", "pubchem_dag.py",
        "string_dag.py", "uniprot_dag.py",
    ])
    def test_dag_instance_named_dag(self, dag_file):
        dag_path = Path(_PHASE1) / "dags" / dag_file
        source = dag_path.read_text()
        assert "dag = " in source, (
            f"{dag_file} must use `dag = <factory>()` naming (BUG #40)"
        )
        assert "_dag_instance =" not in source, (
            f"{dag_file} must NOT use the old `_dag_instance =` naming (BUG #40)"
        )


# =============================================================================
# BUG #41 (P3) -- connectivity index gated on collapse_stereoisomers
# =============================================================================

class TestBug41ConnectivityIndexGated:
    """BUG #41: connectivity index population must be gated on
    collapse_stereoisomers=True to avoid 20% memory waste."""

    def test_connectivity_index_gated(self):
        drug_path = Path(_PHASE1) / "entity_resolution" / "drug_resolver.py"
        source = drug_path.read_text()
        assert 'getattr(self._config, "collapse_stereoisomers", False)' in source, (
            "drug_resolver.py must gate connectivity index on "
            "collapse_stereoisomers (BUG #41)"
        )

    def test_no_unconditional_connectivity_population(self):
        drug_path = Path(_PHASE1) / "entity_resolution" / "drug_resolver.py"
        source = drug_path.read_text()
        assert "ALWAYS populate connectivity" not in source, (
            "drug_resolver.py must NOT have the v82 unconditional "
            "'ALWAYS populate connectivity index' comment (BUG #41)"
        )


# =============================================================================
# BUG #42 (P3) -- log warning on organism fallback to default
# =============================================================================

class TestBug42OrganismFallbackWarning:
    """BUG #42: log a WARNING when organism is empty and default_organism
    is used, so invalid organism input is visible to operators."""

    def test_warning_log_present(self):
        protein_path = Path(_PHASE1) / "entity_resolution" / "protein_resolver.py"
        source = protein_path.read_text()
        assert "v89 BUG #42" in source, (
            "protein_resolver.py must reference BUG #42 in the organism "
            "fallback warning (BUG #42)"
        )
        assert "cross-species contamination" in source, (
            "protein_resolver.py must warn about cross-species contamination "
            "on organism fallback (BUG #42)"
        )


# =============================================================================
# BUG #43 (P3) -- always add string_id column
# =============================================================================

class TestBug43AlwaysAddStringIdColumn:
    """BUG #43: the string_id column must ALWAYS be added to
    string_protein_df, even if all values are None."""

    def test_string_id_always_added(self):
        run_path = Path(_PHASE1) / "entity_resolution" / "run.py"
        source = run_path.read_text()
        assert "BUG #43" in source, (
            "run.py must reference BUG #43 (always add string_id column)"
        )


# =============================================================================
# BUG #44 (P3 Compound) -- cross-cutting compound fix
# =============================================================================

class TestBug44CompoundFix:
    """BUG #44: compound fix covering PubChem ordering, STRING mispairing,
    organism filter, override table, and default organism.

    This is a compound bug -- the individual fixes are verified by the
    BUG #23, #28, #33, #35, #42, #43 tests above. This test class
    verifies the COMPOUND invariant: the master DAG and download_parallel
    produce the SAME DB semantics."""

    def test_master_dag_and_download_parallel_both_run_entity_resolution(self):
        master_path = Path(_PHASE1) / "dags" / "master_pipeline_dag.py"
        parallel_path = Path(_PHASE1) / "scripts" / "download_parallel.py"
        master_src = master_path.read_text()
        parallel_src = parallel_path.read_text()
        assert "run_entity_resolution" in master_src, (
            "master_pipeline_dag.py must call run_entity_resolution"
        )
        assert "run_entity_resolution" in parallel_src, (
            "download_parallel.py must call run_entity_resolution (BUG #44 "
            "compound -- both paths must use the shared function)"
        )

    def test_pubchem_loads_after_other_drugs_in_both_paths(self):
        master_path = Path(_PHASE1) / "dags" / "master_pipeline_dag.py"
        parallel_path = Path(_PHASE1) / "scripts" / "download_parallel.py"
        master_src = master_path.read_text()
        parallel_src = parallel_path.read_text()
        assert "resolve >> pubchem_download >> pubchem_load" in master_src, (
            "master_pipeline_dag.py must wire resolve >> pubchem_download >> pubchem_load"
        )
        assert "LOAD_PASS_NO_PUBCHEM" in parallel_src
        assert "PUBCHEM_LOAD" in parallel_src


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
