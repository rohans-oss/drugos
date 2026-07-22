"""Team 4 Forensic Root Fix Tests — P2-001 to P1-050.

Each test verifies ONE assigned issue's root fix. Tests are written to
catch the EXACT bug described in the issue (regression tests).

Run with: pytest tests/test_team4_p2_root_fixes.py -v
"""
import os
import sys
from pathlib import Path

import pytest

# Ensure the repo root is on sys.path so we can import phase1/phase2/graph_transformer
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Issue 1 (P2-001): ClinicalOutcome and MedDRA_Term share the same canonical ID
# =============================================================================
def test_p2_001_clinical_outcome_medra_namespace_distinct():
    """P2-001: ClinicalOutcome and MedDRA_Term MUST have distinct canonical IDs."""
    sys.path.insert(0, str(REPO_ROOT / "phase2"))
    from drugos_graph.config import (
        CANONICAL_IDS,
        CANONICAL_IDS_METADATA,
        ID_MAPPING_PRIORITY,
    )

    co_field = CANONICAL_IDS["ClinicalOutcome"]
    medra_field = CANONICAL_IDS["MedDRA_Term"]
    assert co_field != medra_field, (
        f"P2-001 REGRESSION: ClinicalOutcome ({co_field!r}) and MedDRA_Term "
        f"({medra_field!r}) MUST have distinct canonical ID fields — sharing "
        f"the same field causes namespace collision."
    )
    assert co_field == "clinical_outcome_id", (
        f"P2-001 REGRESSION: ClinicalOutcome canonical ID must be "
        f"'clinical_outcome_id' (the CO:<drug>:<disease>:<indication> format), "
        f"got {co_field!r}."
    )
    assert medra_field == "meddra_id", (
        f"P2-001 REGRESSION: MedDRA_Term canonical ID must be 'meddra_id', "
        f"got {medra_field!r}."
    )

    # Verify metadata consistency
    assert CANONICAL_IDS_METADATA["ClinicalOutcome"]["field"] == "clinical_outcome_id"
    assert CANONICAL_IDS_METADATA["MedDRA_Term"]["field"] == "meddra_id"

    # Verify ID_MAPPING_PRIORITY puts clinical_outcome_id first for ClinicalOutcome
    assert ID_MAPPING_PRIORITY["ClinicalOutcome"][0] == "clinical_outcome_id"


def test_p2_001_clinical_outcome_id_validator():
    """P2-001: is_clinical_outcome_id validator exists and works."""
    sys.path.insert(0, str(REPO_ROOT / "phase2"))
    from drugos_graph.utils import is_clinical_outcome_id

    assert is_clinical_outcome_id("CO:DB00001:DOID:0050133:approved") is True
    assert is_clinical_outcome_id("CO:DB00071:OMIM:102700:investigational") is True
    # Must NOT accept a bare meddra_id (that's MedDRA_Term's namespace)
    assert is_clinical_outcome_id("10002083") is False
    assert is_clinical_outcome_id("") is False
    assert is_clinical_outcome_id(None) is False


# =============================================================================
# Issue 2 (P2-002): Phase 2 graph_transformer_model.py DELETED
# =============================================================================
def test_p2_002_phase2_hgt_model_deleted():
    """P2-002: phase2/drugos_graph/graph_transformer_model.py MUST NOT exist."""
    p2_model = REPO_ROOT / "phase2" / "drugos_graph" / "graph_transformer_model.py"
    assert not p2_model.exists(), (
        "P2-002 REGRESSION: phase2/drugos_graph/graph_transformer_model.py must "
        "NOT exist — it was an INCOMPATIBLE HGT model that could not load into "
        "Phase 3's DrugRepurposingGraphTransformer."
    )


def test_p2_002_phase3_model_is_canonical():
    """P2-002: Phase 3's DrugRepurposingGraphTransformer is the canonical model."""
    sys.path.insert(0, str(REPO_ROOT / "graph_transformer"))
    # Skip if torch not available
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not available")
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
        GraphTransformerModel,
    )
    assert GraphTransformerModel is DrugRepurposingGraphTransformer, (
        "P2-002 REGRESSION: Phase 3's GraphTransformerModel alias must point "
        "to DrugRepurposingGraphTransformer."
    )


def test_p2_002_step11b_delegates_to_phase3():
    """P2-002: step11b_train_graph_transformer delegates to Phase 3 (no training)."""
    rp_src = (REPO_ROOT / "phase2" / "drugos_graph" / "run_pipeline.py").read_text()
    assert "phase3_delegated" in rp_src, (
        "P2-002 REGRESSION: step11b must return model_type='phase3_delegated'."
    )
    assert "DrugRepurposingGraphTransformer" in rp_src, (
        "P2-002 REGRESSION: step11b must reference Phase 3's "
        "DrugRepurposingGraphTransformer."
    )


# =============================================================================
# Issue 3 (P2-003): step7a/b/c fail loudly when Phase 1 CSV missing in prod
# =============================================================================
def test_p2_003_step7_raises_in_production():
    """P2-003: step7a/b/c must raise RuntimeError when Phase 1 CSV missing in prod."""
    rp_src = (REPO_ROOT / "phase2" / "drugos_graph" / "run_pipeline.py").read_text()
    # Must have production-mode guards for STRING, UniProt, ChEMBL
    assert "DRUGOS_ENVIRONMENT" in rp_src, (
        "P2-003 REGRESSION: run_pipeline must check DRUGOS_ENVIRONMENT."
    )
    # Must raise RuntimeError (not just log) in production
    p2_003_count = rp_src.count("P2-003 ROOT FIX")
    assert p2_003_count >= 3, (
        f"P2-003 REGRESSION: expected >=3 P2-003 guards (STRING/UniProt/ChEMBL), "
        f"got {p2_003_count}."
    )
    # Must have raise RuntimeError in production mode
    assert rp_src.count("raise RuntimeError(_err_msg_p2_003") >= 1, (
        "P2-003 REGRESSION: step7a must raise RuntimeError in production."
    )
    assert rp_src.count("raise RuntimeError(_err_msg_p2_003b") >= 1, (
        "P2-003 REGRESSION: step7b must raise RuntimeError in production."
    )
    assert rp_src.count("raise RuntimeError(_err_msg_p2_003c") >= 1, (
        "P2-003 REGRESSION: step7c must raise RuntimeError in production."
    )


# =============================================================================
# Issue 4 (P2-004): gpu_utils.check_gpu_available reports device_index
# =============================================================================
def test_p2_004_gpu_utils_reports_device_index():
    """P2-004: check_gpu_available must set 'device_index' when CUDA available."""
    sys.path.insert(0, str(REPO_ROOT / "phase2"))
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not available")
    from drugos_graph.gpu_utils import check_gpu_available
    info = check_gpu_available()
    if info["cuda_available"]:
        assert "device_index" in info, (
            "P2-004 REGRESSION: check_gpu_available must set 'device_index' "
            "when CUDA is available."
        )
        assert isinstance(info["device_index"], int), (
            f"P2-004 REGRESSION: device_index must be int, got "
            f"{type(info['device_index'])}."
        )


def test_p2_004_gpu_utils_source_has_device_index():
    """P2-004: gpu_utils.py source must contain device_index assignment."""
    src = (REPO_ROOT / "phase2" / "drugos_graph" / "gpu_utils.py").read_text()
    assert 'info["device_index"]' in src, (
        "P2-004 REGRESSION: gpu_utils.py must assign info['device_index']."
    )
    assert "torch.cuda.current_device()" in src, (
        "P2-004 REGRESSION: gpu_utils.py must use torch.cuda.current_device()."
    )


# =============================================================================
# Issue 5 (P2-005): bridge_to_pyg_maps consolidates Compound via compound_id_aliases
# =============================================================================
def test_p2_005_bridge_consolidates_compound_aliases():
    """P2-005: bridge_to_pyg_maps merges Compound nodes sharing an alias."""
    sys.path.insert(0, str(REPO_ROOT / "phase2"))
    from drugos_graph.phase1_bridge import (
        RecordingGraphBuilder,
        bridge_to_pyg_maps,
    )

    builder = RecordingGraphBuilder()
    # Simulate a biologic drug (DB00071, no InChIKey) and its ChEMBL
    # equivalent (with InChIKey). Both share chembl_id=CHEMBL123456 in
    # compound_id_aliases. The fix MUST merge them into ONE Compound node.
    builder.node_loads = [
        {
            "label": "Compound",
            "nodes": [
                {
                    "id": "DB00071",
                    "compound_id_aliases": ["CHEMBL123456", "DB00071"],
                },
                {
                    "id": "RZVAJINKQORUOD-UHFFFAOYSA-N",  # inchikey
                    "compound_id_aliases": ["CHEMBL123456"],
                },
            ],
        }
    ]
    builder.edge_loads = []
    entity_maps, edge_maps = bridge_to_pyg_maps(builder)
    compound_map = entity_maps["Compound"]
    # Both nodes must map to the SAME index (merged)
    assert compound_map["DB00071"] == compound_map["RZVAJINKQORUOD-UHFFFAOYSA-N"], (
        f"P2-005 REGRESSION: biologic DB00071 and its ChEMBL equivalent "
        f"must merge into ONE Compound node (same index), got "
        f"DB00071={compound_map['DB00071']} vs "
        f"inchikey={compound_map['RZVAJINKQORUOD-UHFFFAOYSA-N']}."
    )
    # Only ONE unique index (not two)
    assert len(set(compound_map.values())) == 1, (
        f"P2-005 REGRESSION: expected 1 unique Compound index (merged), "
        f"got {len(set(compound_map.values()))}."
    )


# =============================================================================
# Issue 6 (P2-006): split_for_link_prediction default node_disjoint=True
# =============================================================================
def test_p2_006_default_node_disjoint_true():
    """P2-006: split_for_link_prediction default must be node_disjoint=True."""
    import inspect
    sys.path.insert(0, str(REPO_ROOT / "phase2"))
    # We can't import PyGBuilder without torch/pyg, so inspect the source
    src = (REPO_ROOT / "phase2" / "drugos_graph" / "pyg_builder.py").read_text()
    # Find the function signature
    import re
    m = re.search(
        r"def split_for_link_prediction\([^)]*node_disjoint:\s*bool\s*=\s*(\w+)",
        src,
    )
    assert m is not None, (
        "P2-006 REGRESSION: could not find split_for_link_prediction signature."
    )
    default_val = m.group(1)
    assert default_val == "True", (
        f"P2-006 REGRESSION: split_for_link_prediction default must be "
        f"node_disjoint=True (GNN-safe), got node_disjoint={default_val}."
    )


def test_p2_006_warns_on_edge_disjoint():
    """P2-006: a WARNING is logged when node_disjoint=False is explicitly passed."""
    src = (REPO_ROOT / "phase2" / "drugos_graph" / "pyg_builder.py").read_text()
    assert "DRUGOS_ALLOW_EDGE_DISJOINT_SPLIT" in src, (
        "P2-006 REGRESSION: must check DRUGOS_ALLOW_EDGE_DISJOINT_SPLIT env var."
    )
    assert "P2-006 ROOT FIX" in src, (
        "P2-006 REGRESSION: must have P2-006 ROOT FIX warning log."
    )


# =============================================================================
# Issue 7 (P1-049): is_fda_approved nullable + migration 013
# =============================================================================
def test_p1_049_migration_001_is_fda_approved_nullable():
    """P1-049: migration 001 must declare is_fda_approved as nullable (no DEFAULT)."""
    src = (REPO_ROOT / "phase1" / "database" / "migrations" / "001_initial_schema.sql").read_text()
    # Must NOT have the old NOT NULL DEFAULT FALSE
    assert "is_fda_approved     BOOLEAN NOT NULL DEFAULT FALSE" not in src, (
        "P1-049 REGRESSION: is_fda_approved must NOT be NOT NULL DEFAULT FALSE."
    )
    # Must have the new nullable definition
    assert "is_fda_approved     BOOLEAN," in src, (
        "P1-049 REGRESSION: is_fda_approved must be nullable (BOOLEAN, with no NOT NULL)."
    )
    # Must have the CHECK constraint
    assert "chk_drugs_is_fda_approved" in src, (
        "P1-049 REGRESSION: must have chk_drugs_is_fda_approved CHECK constraint."
    )


def test_p1_049_migration_013_exists():
    """P1-049: migration 013 (is_fda_approved nullable) must exist."""
    mig_013 = REPO_ROOT / "phase1" / "database" / "migrations" / "013_is_fda_approved_nullable.sql"
    assert mig_013.exists(), (
        "P1-049 REGRESSION: migration 013_is_fda_approved_nullable.sql must exist."
    )
    src = mig_013.read_text()
    assert "DROP NOT NULL" in src, (
        "P1-049 REGRESSION: migration 013 must drop NOT NULL."
    )
    assert "DROP DEFAULT" in src, (
        "P1-049 REGRESSION: migration 013 must drop DEFAULT FALSE."
    )
    assert "P1-049 VERIFICATION" in src, (
        "P1-049 REGRESSION: migration 013 must have post-migration verification."
    )


def test_p1_049_orm_model_nullable():
    """P1-049: ORM model must declare is_fda_approved as nullable Optional[bool]."""
    src = (REPO_ROOT / "phase1" / "database" / "models.py").read_text()
    assert "is_fda_approved: Mapped[Optional[bool]]" in src, (
        "P1-049 REGRESSION: ORM model must use Mapped[Optional[bool]] for is_fda_approved."
    )
    assert "nullable=True" in src, (
        "P1-049 REGRESSION: ORM model must have nullable=True for is_fda_approved."
    )


# =============================================================================
# Issue 8 (P1-043): migration 012 post-migration verification
# =============================================================================
def test_p1_043_migration_012_has_verification():
    """P1-043: migration 012 must have post-migration verification."""
    src = (REPO_ROOT / "phase1" / "database" / "migrations" / "012_confidence_tier_pinero_alignment.sql").read_text()
    assert "P1-043 VERIFICATION" in src, (
        "P1-043 REGRESSION: migration 012 must have post-migration verification."
    )
    # Must check the constraint exists
    assert "chk_gda_confidence_tier" in src, (
        "P1-043 REGRESSION: verification must check chk_gda_confidence_tier exists."
    )
    # Must check no residual 'moderate' rows
    assert "moderate" in src, (
        "P1-043 REGRESSION: verification must check for residual 'moderate' rows."
    )


# =============================================================================
# Issue 9 (P1-045): disgenet_pipeline safe_classify
# =============================================================================
def test_p1_045_safe_classify_never_raises():
    """P1-045: _safe_classify_confidence must never raise on malformed input."""
    sys.path.insert(0, str(REPO_ROOT / "phase1"))
    from pipelines.disgenet_pipeline import _safe_classify_confidence
    # Must NOT raise on any of these malformed inputs
    assert _safe_classify_confidence("weak") is None  # string, not numeric
    assert _safe_classify_confidence("not_a_number") is None
    assert _safe_classify_confidence(None) is None
    assert _safe_classify_confidence([]) is None
    assert _safe_classify_confidence({}) is None
    # Valid inputs must still work
    result = _safe_classify_confidence(0.5)
    assert result in ("sub_weak", "weak", "strong"), (
        f"P1-045 REGRESSION: valid score 0.5 must classify, got {result!r}."
    )


def test_p1_045_caller_uses_safe_classify():
    """P1-045: the caller must use _safe_classify_confidence + pd.to_numeric."""
    src = (REPO_ROOT / "phase1" / "pipelines" / "disgenet_pipeline.py").read_text()
    assert "_safe_classify_confidence" in src, (
        "P1-045 REGRESSION: caller must use _safe_classify_confidence."
    )
    assert 'pd.to_numeric(df["score"], errors="coerce")' in src, (
        "P1-045 REGRESSION: caller must coerce score to numeric first."
    )


# =============================================================================
# Issue 10 (P1-047): stagger DAG schedules to avoid overlap
# =============================================================================
def test_p1_047_dag_schedules_staggered():
    """P1-047: standalone DAGs must NOT overlap with master's Sunday window."""
    dag_dir = REPO_ROOT / "phase1" / "dags"
    schedules = {}
    for dag_file in dag_dir.glob("*_dag.py"):
        src = dag_file.read_text()
        import re
        m = re.search(r'schedule="([^"]+)"', src)
        if m:
            schedules[dag_file.stem] = m.group(1)
    # Master must remain Sunday 02:00 UTC
    assert schedules.get("master_pipeline_dag") == "0 2 * * 0", (
        f"P1-047: master must be '0 2 * * 0' (Sunday 02:00 UTC), got "
        f"{schedules.get('master_pipeline_dag')!r}."
    )
    # OMIM/UniProt/STRING must NOT use day-of-month 15 (the collision risk)
    assert "15" not in schedules.get("omim_dag", ""), (
        f"P1-047: omim_dag must not use day-15 schedule, got {schedules['omim_dag']!r}."
    )
    assert "15" not in schedules.get("uniprot_dag", ""), (
        f"P1-047: uniprot_dag must not use day-15 schedule, got {schedules['uniprot_dag']!r}."
    )
    assert "15" not in schedules.get("string_dag", ""), (
        f"P1-047: string_dag must not use day-15 schedule, got {schedules['string_dag']!r}."
    )
    # PubChem must NOT be on Wednesday (ChEMBL overlap)
    pubchem_sched = schedules.get("pubchem_dag", "")
    assert "* * 3" not in pubchem_sched, (
        f"P1-047: pubchem_dag must not be on Wednesday (ChEMBL overlap), got {pubchem_sched!r}."
    )


# =============================================================================
# Issue 11 (P1-003 / superseded P1-048): SCHEMA_VERSION_FALLBACK semantics
# =============================================================================
# v104 FORENSIC ROOT FIX (P1-003): the previous test asserted
# ``SCHEMA_VERSION_FALLBACK == max_mig``. That assertion was the BUG FARM --
# it forced every contributor to bump the fallback when adding a migration,
# which is exactly the chore that was forgotten for migrations 016 and 017.
# The new semantics (P1-003): ``SCHEMA_VERSION_FALLBACK = 0`` means "if the
# migrations directory is missing, treat as fresh install -- apply ALL
# migrations." The real correctness check is that ``SCHEMA_VERSION`` (the
# derived value, NOT the fallback) equals ``max_mig`` when the migrations
# directory is present.
def test_p1_003_schema_version_fallback_is_zero():
    """P1-003: SCHEMA_VERSION_FALLBACK must be 0 (fresh-install semantics)."""
    sys.path.insert(0, str(REPO_ROOT / "phase1"))
    if "database.base" in sys.modules:
        del sys.modules["database.base"]
    from database.base import SCHEMA_VERSION_FALLBACK, SCHEMA_VERSION
    assert SCHEMA_VERSION_FALLBACK == 0, (
        f"P1-003 REGRESSION: SCHEMA_VERSION_FALLBACK must be 0 "
        f"(fresh-install semantics -- the old 'bump on new migration' "
        f"contract was the bug farm). Got {SCHEMA_VERSION_FALLBACK}."
    )
    # The REAL correctness check: when the migrations directory is present
    # (which it is in this repo), SCHEMA_VERSION (derived) must equal the
    # max migration version. This is the invariant that matters.
    mig_dir = REPO_ROOT / "phase1" / "database" / "migrations"
    mig_versions = []
    for f in mig_dir.glob("0*.sql"):
        if "_rollback" in f.name:
            continue
        try:
            v = int(f.name[:3])
            mig_versions.append(v)
        except ValueError:
            continue
    max_mig = max(mig_versions) if mig_versions else 0
    assert SCHEMA_VERSION == max_mig, (
        f"P1-003 REGRESSION: SCHEMA_VERSION ({SCHEMA_VERSION}) must equal "
        f"max migration version ({max_mig}) when migrations dir is present."
    )


# Backward-compat alias: older CI runners may still call the old name.
# Delegate to the new test so old test-collection does not break.
def test_p1_048_schema_version_fallback_bumped():
    """P1-048 (superseded by P1-003): delegate to the new semantics test."""
    test_p1_003_schema_version_fallback_is_zero()


# =============================================================================
# Issue 12 (P1-044): STRING dedup uses kind=mergesort
# =============================================================================
def test_p1_044_string_dedup_mergesort():
    """P1-044: STRING dedup must use kind='mergesort' for determinism."""
    src = (REPO_ROOT / "phase1" / "pipelines" / "string_pipeline.py").read_text()
    assert 'kind="mergesort"' in src, (
        "P1-044 REGRESSION: string_pipeline.py must use kind='mergesort' for "
        "deterministic dedup."
    )


# =============================================================================
# Issue 13 (P1-046): is_fda_approved nullable (duplicate of #7, same fix)
# =============================================================================
def test_p1_046_is_fda_approved_nullable():
    """P1-046: is_fda_approved must be nullable (same fix as P1-049)."""
    # This is the same fix as P1-049 — verify it's in place
    src = (REPO_ROOT / "phase1" / "database" / "migrations" / "001_initial_schema.sql").read_text()
    assert "is_fda_approved     BOOLEAN NOT NULL DEFAULT FALSE" not in src, (
        "P1-046 REGRESSION: is_fda_approved must NOT be NOT NULL DEFAULT FALSE."
    )


# =============================================================================
# Issue 14 (P1-050): remove module-level ensure_project_root() call
# =============================================================================
def test_p1_050_no_module_level_side_effect():
    """P1-050: _dags_init.py must NOT call ensure_project_root() at module level."""
    src = (REPO_ROOT / "phase1" / "dags" / "_dags_init.py").read_text()
    # The module-level call (the LAST line of the file, no indent) must be removed
    lines = src.split("\n")
    # Find any module-level (no indent) call to ensure_project_root()
    for i, line in enumerate(lines, start=1):
        stripped = line.rstrip()
        if stripped == "ensure_project_root()" or stripped.startswith("ensure_project_root()  #"):
            # Check if it's at module level (no leading whitespace)
            if not line.startswith(" ") and not line.startswith("\t"):
                pytest.fail(
                    f"P1-050 REGRESSION: module-level ensure_project_root() call "
                    f"found at line {i}: {line!r}. Must be removed."
                )
    # The function definition must still exist
    assert "def ensure_project_root()" in src, (
        "P1-050 REGRESSION: ensure_project_root() function must still exist."
    )


def test_p1_050_dags_call_explicitly():
    """P1-050: all 8 DAG files must explicitly call ensure_project_root()."""
    dag_dir = REPO_ROOT / "phase1" / "dags"
    dag_files = [
        "master_pipeline_dag.py",
        "chembl_dag.py",
        "disgenet_dag.py",
        "drugbank_dag.py",
        "omim_dag.py",
        "pubchem_dag.py",
        "string_dag.py",
        "uniprot_dag.py",
    ]
    for dag_file in dag_files:
        src = (dag_dir / dag_file).read_text()
        # Must import ensure_project_root
        assert "from dags._dags_init import ensure_project_root" in src, (
            f"P1-050 REGRESSION: {dag_file} must import ensure_project_root."
        )
        # Must explicitly call it (not rely on module-level side effect)
        # Look for a line that's just "ensure_project_root()" (possibly with comment)
        import re
        # Match "ensure_project_root()" as a statement (not inside a string)
        assert re.search(r"^ensure_project_root\(\)", src, re.MULTILINE), (
            f"P1-050 REGRESSION: {dag_file} must explicitly call "
            f"ensure_project_root() at module top."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
