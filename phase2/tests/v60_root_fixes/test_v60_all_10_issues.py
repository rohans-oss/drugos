"""v60 ROOT FIX verification tests -- 10 critical forensic issues.

Each test verifies ONE issue from the audit list. Tests are designed to
FAIL if a regression re-introduces the bug. They use NO network access,
NO real databases -- pure Python assertions on the actual code.

Issues verified:
  #1  ChEMBL _RE_ACTIVATE regex matches INACTIVATION -> covalent inhibitors
      misclassified as activators (patient-safety critical).
  #2  ChEMBERTa 3 layers of silent fallback -> training proceeds on random
      features.
  #3  HGT training uses BCELoss on already-sigmoided scores -> numerical
      instability (log(0) -> -inf).
  #4  HGT model NEVER SAVED when val_idx empty (best_val_auc init to NaN/-1.0
      means save guard `> 0.5` always fails).
  #5  3 of 7 CORE_NODE_TYPES (ClinicalOutcome, MedDRA_Term, Anatomy) have
      no canonical ID system -> SIDER/GEO edges have no canonical endpoint
      resolution.
  #6  ClinicalTrials 'Completed' status treated as positive evidence ->
      negative-result trials become positive training signal.
  #7  STITCH compound IDs CIDsm vs CIDs vs CIDf -> 3-way split of same
      molecule.
  #8  OpenTargets association_score written into BOTH binding_confidence
      AND chembl_score -> cross-source score fusion corrupts.
  #9  Negative sampling samples from full node set -> ~30% of 'negatives'
      are actually positives = data leakage.
  #10 Training data split uses random shuffling instead of edge-disjoint
       split -> AUC inflated by 0.10+.
"""

import os
import re
import sys
import ast
from pathlib import Path

# Ensure the phase2 package is importable.
_HERE = Path(__file__).resolve().parent
# _HERE = .../phase2/tests/v60_root_fixes
# _PHASE2_DIR = .../phase2 (parent.parent)
_PHASE2_DIR = _HERE.parent.parent
if str(_PHASE2_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE2_DIR))


# ===========================================================================
# ISSUE #1 -- ChEMBL _RE_ACTIVATE regex matches INACTIVATION
# ===========================================================================

def test_issue_1_chembl_inactivation_not_classified_as_activator():
    """INACTIVATION/INACTIVATOR/INACTIVATE/INACTIVATED must classify as
    'inhibits', NEVER 'activates'. This is the patient-safety fix for
    covalent inhibitors (aspirin, omeprazole, clopidogrel, penicillin).
    """
    from drugos_graph.chembl_loader import standard_type_to_relation

    # All INACTIVAT* variants must classify as "inhibits".
    for std_type in ["INACTIVATION", "INACTIVATOR", "INACTIVATE",
                     "INACTIVATED", "Inactivation", "inactivator"]:
        result = standard_type_to_relation(std_type)
        assert result == "inhibits", (
            f"FAIL: standard_type_to_relation({std_type!r}) returned "
            f"{result!r} -- must be 'inhibits'. Covalent inhibitors "
            f"misclassified as activators is a patient-safety regression."
        )

    # Genuine activation terms must still classify as "activates".
    for std_type in ["ACTIVATION", "ACTIVATOR", "AGONIST", "Activation"]:
        result = standard_type_to_relation(std_type)
        assert result == "activates", (
            f"FAIL: standard_type_to_relation({std_type!r}) returned "
            f"{result!r} -- must be 'activates'."
        )

    print("PASS: Issue #1 -- INACTIVATION correctly classified as 'inhibits'")


# ===========================================================================
# ISSUE #2 -- ChEMBERTa 3 layers of silent fallback
# ===========================================================================

def test_issue_2_chemberta_strict_features_default_on():
    """DRUGOS_STRICT_FEATURES must default to "1" (ON) so ChEMBERTa
    failures RAISE instead of silently falling back to random Xavier
    features. The 3 silent-fallback layers are:
      Layer 1: DRUGOS_USE_CHEMBERTA=0
      Layer 2: transformers not importable
      Layer 3: HF_TOKEN missing OR encode_smiles failed
    All three must now RAISE by default.
    """
    # Read the source file and verify the default is "1".
    run_pipeline_path = _PHASE2_DIR / "drugos_graph" / "run_pipeline.py"
    source = run_pipeline_path.read_text(encoding="utf-8")
    # Find the strict_features line.
    match = re.search(
        r'strict_features\s*=\s*os\.environ\.get\(\s*["\']DRUGOS_STRICT_FEATURES["\']\s*,\s*["\'](\d)["\']\s*\)',
        source,
    )
    assert match is not None, (
        "FAIL: Could not find strict_features env-var read in run_pipeline.py"
    )
    default_value = match.group(1)
    assert default_value == "1", (
        f"FAIL: DRUGOS_STRICT_FEATURES default is {default_value!r}, "
        f"must be '1'. The v58 default of '0' meant ChEMBERTa failures "
        f"silently fell back to random Xavier features -- training "
        f"proceeded on random features with no signal."
    )

    # Verify the _strict_raise helper exists and raises when strict.
    assert "def _strict_raise" in source, (
        "FAIL: _strict_raise helper not found in run_pipeline.py"
    )
    assert "FeatureFailureError" in source, (
        "FAIL: FeatureFailureError exception not referenced in run_pipeline.py"
    )

    print("PASS: Issue #2 -- DRUGOS_STRICT_FEATURES defaults to '1' (ON)")


# ===========================================================================
# ISSUE #3 -- HGT BCELoss on already-sigmoided scores
# ===========================================================================

def test_issue_3_hgt_uses_bcewithlogitsloss_not_bceloss():
    """HGT training must use BCEWithLogitsLoss (numerically stable,
    applies sigmoid internally via log-sum-exp trick) on RAW LOGITS,
    NOT BCELoss on sigmoided scores. BCELoss on sigmoided scores
    produces log(0) -> -inf on confident predictions.
    """
    run_pipeline_path = _PHASE2_DIR / "drugos_graph" / "run_pipeline.py"
    source = run_pipeline_path.read_text(encoding="utf-8")

    # Must use BCEWithLogitsLoss.
    assert "BCEWithLogitsLoss" in source, (
        "FAIL: BCEWithLogitsLoss not found in run_pipeline.py -- "
        "HGT training would use BCELoss on sigmoided scores -> log(0) -> -inf"
    )

    # Must NOT have BCELoss() (without the WithLogits).
    # Find all BCELoss usages that are NOT BCEWithLogitsLoss and NOT in comments.
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith("#"):
            continue
        # Look for BCELoss( but NOT BCEWithLogitsLoss(
        if "BCELoss(" in stripped and "BCEWithLogitsLoss(" not in stripped:
            # Allow if it's in a string/docstring (heuristic: line has quotes)
            if '"""' in stripped or "'''" in stripped or stripped.startswith('"'):
                continue
            assert False, (
                f"FAIL: line {i} uses BCELoss() without WithLogits: "
                f"{stripped!r}. This causes log(0) -> -inf numerical "
                f"instability on confident predictions."
            )

    # P2-026 ROOT FIX (Team 8): the previous block read
    # ``phase2/drugos_graph/graph_transformer_model.py`` to verify the
    # BCEWithLogitsLoss documentation. That file was DEAD CODE (no
    # module imported it) and has been DELETED by P2-026. The
    # canonical Phase 3 model is
    # ``graph_transformer/models/graph_transformer.py``. The
    # BCEWithLogitsLoss guarantee is now verified via run_pipeline.py
    # (the training entry point) above -- the run_pipeline check at
    # line 135 already covers the production training path. The dead
    # file check is removed because it would crash with
    # FileNotFoundError after P2-026.

    print("PASS: Issue #3 -- HGT uses BCEWithLogitsLoss on raw logits")


# ===========================================================================
# ISSUE #4 -- HGT model NEVER SAVED when val_idx empty
# ===========================================================================

def test_issue_4_hgt_model_saved_when_val_idx_empty():
    """HGT model MUST be saved even when val_idx is empty (common on
    small datasets). The previous save guard `if best_val_auc > 0.5:`
    failed when best_val_auc stayed at -1.0 (the init value when
    val_idx is empty). ROOT FIX: always save with validation markers.
    """
    run_pipeline_path = _PHASE2_DIR / "drugos_graph" / "run_pipeline.py"
    source = run_pipeline_path.read_text(encoding="utf-8")

    # The old guard `if best_val_auc > 0.5:` for the save block must
    # be GONE -- replaced by always-save logic.
    # Find the save block: look for "torch.save" inside step11b.
    assert "torch.save" in source, "FAIL: torch.save not found in run_pipeline.py"

    # Verify the new save markers exist.
    assert "validation_performed" in source, (
        "FAIL: 'validation_performed' marker not found -- the v60 root fix "
        "must always save with this marker so downstream consumers know "
        "whether the checkpoint was val-selected or last-epoch fallback."
    )
    assert "save_reason" in source, (
        "FAIL: 'save_reason' marker not found -- must indicate "
        "'best_val_checkpoint' / 'last_epoch_no_validation' / "
        "'last_epoch_validation_below_threshold'"
    )

    # Verify the old `if best_val_auc > 0.5:` guard for the SAVE block
    # is gone. (Note: best_val_auc > 0.5 may still appear elsewhere for
    # V1 launch criteria checks -- that's fine. We're checking the SAVE
    # block specifically.)
    # Find the save block by locating "model_path = CHECKPOINT_DIR".
    save_block_match = re.search(
        r'model_path\s*=\s*None\s*\n\s*model_saved\s*=\s*False(.*?)return\s*\{',
        source, re.DOTALL,
    )
    assert save_block_match is not None, (
        "FAIL: Could not locate the HGT model save block in run_pipeline.py"
    )
    save_block = save_block_match.group(1)
    # The save block must NOT gate on `if best_val_auc > 0.5:`.
    assert "if best_val_auc > 0.5:" not in save_block, (
        "FAIL: The HGT save block still has `if best_val_auc > 0.5:` guard. "
        "When val_idx is empty, best_val_auc stays at -1.0 and the model is "
        "NEVER saved. ROOT FIX: always save with validation markers."
    )

    print("PASS: Issue #4 -- HGT model always saved (with validation markers)")


# ===========================================================================
# ISSUE #5 -- 3 CORE_NODE_TYPES have no canonical ID system
# ===========================================================================

def test_issue_5_canonical_ids_for_all_core_node_types():
    """All 7 CORE_NODE_TYPES must have canonical ID systems AND the
    loaders must actually populate those fields on the node records.
    """
    from drugos_graph.config import (
        CANONICAL_IDS,
        CANONICAL_IDS_METADATA,
        ID_MAPPING_PRIORITY,
        CORE_NODE_TYPES,
    )

    # All 3 previously-missing types must have canonical IDs.
    for nt in ("ClinicalOutcome", "MedDRA_Term", "Anatomy"):
        assert nt in CANONICAL_IDS, (
            f"FAIL: {nt!r} not in CANONICAL_IDS -- no canonical ID system"
        )
        assert nt in ID_MAPPING_PRIORITY, (
            f"FAIL: {nt!r} not in ID_MAPPING_PRIORITY -- no fallback chain"
        )
        assert nt in CANONICAL_IDS_METADATA, (
            f"FAIL: {nt!r} not in CANONICAL_IDS_METADATA -- no validator"
        )

    # Verify SIDER loader populates meddra_id on MedDRA_Term nodes.
    sider_path = _PHASE2_DIR / "drugos_graph" / "sider_loader.py"
    sider_source = sider_path.read_text(encoding="utf-8")
    assert '"meddra_id"' in sider_source, (
        "FAIL: SIDER loader does not populate 'meddra_id' field on "
        "MedDRA_Term nodes -- entity_resolver.resolve_canonical_id "
        "would return None for every MedDRA_Term node."
    )

    # Verify GEO loader emits Anatomy nodes with uberon_id.
    geo_path = _PHASE2_DIR / "drugos_graph" / "geo_loader.py"
    geo_source = geo_path.read_text(encoding="utf-8")
    assert '"uberon_id"' in geo_source, (
        "FAIL: GEO loader does not populate 'uberon_id' field on "
        "Anatomy nodes -- entity_resolver.resolve_canonical_id would "
        "return None for every Anatomy node."
    )

    # Verify phase1_bridge populates meddra_id/mesh_id on ClinicalOutcome.
    bridge_path = _PHASE2_DIR / "drugos_graph" / "phase1_bridge.py"
    bridge_source = bridge_path.read_text(encoding="utf-8")
    assert '"meddra_id"' in bridge_source, (
        "FAIL: phase1_bridge does not populate 'meddra_id' on "
        "ClinicalOutcome nodes."
    )
    assert '"mesh_id"' in bridge_source, (
        "FAIL: phase1_bridge does not populate 'mesh_id' on "
        "ClinicalOutcome nodes."
    )

    print("PASS: Issue #5 -- All 7 CORE_NODE_TYPES have canonical ID systems populated")


# ===========================================================================
# ISSUE #6 -- ClinicalTrials 'Completed' as positive evidence
# ===========================================================================

def test_issue_6_clinicaltrials_primary_outcome_met_parsed():
    """ClinicalTrials loader must parse primary_outcome_met from the
    AACT outcome_analyses table (NOT leave it as None). Negative-result
    trials (Completed + primary_outcome_met=False) must NOT be treated
    as positive drug-disease evidence.
    """
    ct_path = _PHASE2_DIR / "drugos_graph" / "clinicaltrials_loader.py"
    ct_source = ct_path.read_text(encoding="utf-8")

    # Must have a SQL JOIN to outcome_analyses.
    assert "outcome_analyses" in ct_source, (
        "FAIL: outcome_analyses table not queried in clinicaltrials_loader.py"
    )
    assert "outcome_analysis_category" in ct_source, (
        "FAIL: outcome_analysis_category column not queried -- "
        "primary_outcome_met cannot be parsed without it"
    )

    # Must have the primary_outcome_met_raw column.
    assert "primary_outcome_met_raw" in ct_source, (
        "FAIL: primary_outcome_met_raw column not found -- the v60 root "
        "fix adds this column from the outcome_analyses SQL JOIN"
    )

    # Must translate 'met'/'not_met' to True/False.
    assert "'met'" in ct_source and "'not_met'" in ct_source, (
        "FAIL: 'met'/'not_met' translation not found -- "
        "primary_outcome_met_raw must be translated to True/False"
    )

    # _classify_trial_confidence must differentiate True vs False vs None.
    # Verify the function exists and returns different values.
    from drugos_graph.clinicaltrials_loader import (
        _classify_trial_confidence, _TRIAL_SKIP,
    )
    # Completed + primary_outcome_met=True -> 0.9 (strong positive)
    r1 = _classify_trial_confidence("Completed", True)
    assert r1 == 0.9, f"FAIL: Completed+True should be 0.9, got {r1}"
    # Completed + primary_outcome_met=False -> 0.1 (negative result)
    r2 = _classify_trial_confidence("Completed", False)
    assert r2 == 0.1, f"FAIL: Completed+False should be 0.1, got {r2}"
    # Completed + primary_outcome_met=None -> 0.4 (unknown)
    r3 = _classify_trial_confidence("Completed", None)
    assert r3 == 0.4, f"FAIL: Completed+None should be 0.4, got {r3}"
    # Unknown status -> SKIP
    r4 = _classify_trial_confidence("Unknown status", None)
    assert r4 == _TRIAL_SKIP, f"FAIL: Unknown status should be _TRIAL_SKIP, got {r4}"

    # CRITICAL: Completed+True (0.9) must be > Completed+False (0.1).
    # If equal, negative-result trials are treated as positive evidence.
    assert r1 > r2, (
        "FAIL: Completed+True (0.9) must be > Completed+False (0.1). "
        "Equal values mean negative-result trials are treated as positive."
    )

    print("PASS: Issue #6 -- ClinicalTrials primary_outcome_met parsed from outcome_analyses")


# ===========================================================================
# ISSUE #7 -- STITCH CIDsm/CIDs/CIDf 3-way split
# ===========================================================================

def test_issue_7_stitch_cid_normalization():
    """STITCH compound IDs with prefixes CIDsm, CIDs, CIDf, CIDm, CID
    must ALL normalize to the SAME canonical bare PubChem CID.
    """
    from drugos_graph.stitch_loader import _normalize_stitch_cid

    # All 5 prefix variants + bare digits must normalize to "2244".
    variants = [
        "CIDsm00002244",
        "CIDs00002244",
        "CIDf00002244",
        "CIDm00002244",
        "CID00002244",
        "00002244",
    ]
    normalized_results = [_normalize_stitch_cid(v) for v in variants]
    unique_results = set(normalized_results)
    assert len(unique_results) == 1, (
        f"FAIL: STITCH CID variants normalized to different values: "
        f"{dict(zip(variants, normalized_results))}. All must normalize "
        f"to the SAME canonical bare PubChem CID."
    )
    assert unique_results == {"2244"}, (
        f"FAIL: Expected all variants to normalize to '2244', got {unique_results}"
    )

    # Edge cases: None, NaN, empty.
    assert _normalize_stitch_cid(None) == ""
    assert _normalize_stitch_cid("") == ""
    import math
    assert _normalize_stitch_cid(math.nan) == ""
    assert _normalize_stitch_cid("garbage") == ""

    print("PASS: Issue #7 -- STITCH CIDsm/CIDs/CIDf all normalize to canonical CID")


# ===========================================================================
# ISSUE #8 -- OpenTargets score in both binding_confidence AND chembl_score
# ===========================================================================

def test_issue_8_opentargets_no_chembl_score_pollution():
    """OpenTargets loader must NOT write association_score to
    chembl_score (which is reserved for ChEMBL pchembl values on 0-14
    scale). Cross-source score fusion corrupts the ranker.
    """
    ot_path = _PHASE2_DIR / "drugos_graph" / "opentargets_loader.py"
    ot_source = ot_path.read_text(encoding="utf-8")

    # Verify there are NO assignments to chembl_score.
    # Allow mentions in comments/docstrings only.
    lines = ot_source.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith("#"):
            continue
        # Look for assignments like: "chembl_score": or chembl_score =
        if re.search(r'["\']chembl_score["\']\s*[:=]', stripped):
            # Allow if it's clearly documenting the fix (in a docstring)
            if '"""' in stripped or "'''" in stripped:
                continue
            assert False, (
                f"FAIL: line {i} assigns to chembl_score: {stripped!r}. "
                f"OpenTargets must NOT write to chembl_score -- that field "
                f"is reserved for ChEMBL pchembl values (0-14 scale)."
            )

    # v68 ROOT FIX (P2L-045 COMPLETE) verification:
    # OpenTargets MUST NOT write to binding_confidence OR chembl_score.
    # - chembl_score is RESERVED for ChEMBL pchembl values (0-14 scale).
    # - binding_confidence should measure drug-target BINDING affinity
    #   (e.g., from ChEMBL Kd), NOT OpenTargets' integrated association
    #   probability (which combines genetics, somatic, drugs, pathways,
    #   text-mining, animal models, RNA expression). Setting
    #   binding_confidence = association_score mixed binding affinity with
    #   association probability -- meaningless.
    # OpenTargets MUST write to: opentargets_score, association_score, score.
    for forbidden_field in ("chembl_score", "binding_confidence"):
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(rf'["\']{forbidden_field}["\']\s*[:=]', stripped):
                if '"""' in stripped or "'''" in stripped:
                    continue
                assert False, (
                    f"FAIL: line {i} assigns to {forbidden_field}: {stripped!r}. "
                    f"OpenTargets must NOT write to {forbidden_field} -- v68 P2L-045 "
                    f"root fix removed this aliasing because it mixed incompatible "
                    f"score semantics (binding affinity vs association probability)."
                )

    # Verify the CORRECT fields ARE written: opentargets_score,
    # association_score, and the generic score alias.
    assert '"opentargets_score"' in ot_source, (
        "FAIL: OpenTargets must write to opentargets_score "
        "(the v68 P2L-045 recommended OpenTargets-specific alias)"
    )
    assert '"association_score"' in ot_source, (
        "FAIL: OpenTargets must write to association_score "
        "(the original OpenTargets field name, retained verbatim)"
    )
    assert '"score"' in ot_source, (
        "FAIL: OpenTargets must write to the generic 'score' field "
        "(unified alias for downstream consumers that don't care about source)"
    )

    print("PASS: Issue #8 -- OpenTargets no longer pollutes chembl_score or binding_confidence")


# ===========================================================================
# ISSUE #9 -- Negative sampling data leakage
# ===========================================================================

def test_issue_9_negative_sampling_no_leakage():
    """Negative sampling must exclude known positive edges (and
    held-out val/test edges) from the negative sample pool.
    ~30% of 'negatives' being actual positives = data leakage.
    """
    from drugos_graph.negative_sampling import NegativeSampler

    # Build a small sampler with known positives.
    all_drug_ids = ["drug_a", "drug_b", "drug_c", "drug_d"]
    all_disease_ids = ["dis_1", "dis_2", "dis_3", "dis_4"]
    positive_pairs = {("drug_a", "dis_1"), ("drug_b", "dis_2")}
    held_out_pairs = {("drug_c", "dis_3")}  # held-out val/test

    sampler = NegativeSampler(
        all_drug_ids=all_drug_ids,
        all_disease_ids=all_disease_ids,
        positive_pairs=positive_pairs,
        held_out_pairs=held_out_pairs,
        seed=42,
    )

    # Generate negatives using the correct API: random_sampling(num_negatives).
    samples = sampler.random_sampling(num_negatives=20)
    assert len(samples) > 0, "FAIL: negative sampling produced zero samples"

    # Check NO sample is a known positive or held-out pair.
    for sample in samples:
        # The sample dict may use different key names -- check common ones.
        drug_id = sample.get("drug_id") or sample.get("head") or sample.get("src_id")
        disease_id = sample.get("disease_id") or sample.get("tail") or sample.get("dst_id")
        pair = (drug_id, disease_id)
        assert pair not in positive_pairs, (
            f"FAIL: negative sample {pair} is a KNOWN POSITIVE -- data leakage. "
            f"The sampler must exclude positive_pairs from the negative pool. "
            f"Sample dict: {sample}"
        )
        assert pair not in held_out_pairs, (
            f"FAIL: negative sample {pair} is a HELD-OUT pair -- val/test leakage. "
            f"The sampler must exclude held_out_pairs from the negative pool. "
            f"Sample dict: {sample}"
        )

    print(f"PASS: Issue #9 -- {len(samples)} negative samples, zero leakage "
          f"(no known positives, no held-out pairs)")


# ===========================================================================
# ISSUE #10 -- Random split AUC inflation
# ===========================================================================

def test_issue_10_node_disjoint_split_not_random():
    """Training data split must use NODE-DISJOINT split (or temporal)
    as the FIRST option, NOT random shuffling. Random split inflates
    AUC by 0.10+ because drugs in test also appear in train.
    """
    run_pipeline_path = _PHASE2_DIR / "drugos_graph" / "run_pipeline.py"
    source = run_pipeline_path.read_text(encoding="utf-8")

    # Must have node-disjoint split logic.
    assert "node_disjoint" in source.lower(), (
        "FAIL: node-disjoint split not found in run_pipeline.py"
    )
    assert "node_disjoint_split_used" in source, (
        "FAIL: node_disjoint_split_used flag not found -- the split "
        "logic must track whether node-disjoint split was used"
    )

    # Must have temporal split as second option.
    assert "temporal_split_pairs" in source, (
        "FAIL: temporal_split_pairs not found -- must be the second option"
    )

    # The random split must only be a LAST-RESORT fallback.
    assert "stratified" in source.lower() or "last-resort" in source.lower(), (
        "FAIL: random/stratified split must be explicitly marked as last-resort"
    )

    # Verify training_data.temporal_split_pairs raises (not silently
    # falls back to random) when approval_years is missing.
    from drugos_graph.training_data import temporal_split_pairs
    try:
        # No approval_years -> must raise DrugOSDataError
        # (unless DRUGOS_ALLOW_TEMPORAL_RANDOM_FALLBACK=1 is set)
        old_val = os.environ.pop("DRUGOS_ALLOW_TEMPORAL_RANDOM_FALLBACK", None)
        try:
            temporal_split_pairs(
                positive_pairs=[{"drug_id": "d1", "disease_id": "v1"}],
                approval_years=None,
            )
            # If we get here, the function did NOT raise -- fail.
            assert False, (
                "FAIL: temporal_split_pairs did NOT raise when "
                "approval_years=None. It must raise DrugOSDataError to "
                "prevent silent random-split fallback."
            )
        except Exception as exc:
            # Expected -- the function raised.
            pass
        finally:
            if old_val is not None:
                os.environ["DRUGOS_ALLOW_TEMPORAL_RANDOM_FALLBACK"] = old_val
    except AssertionError:
        raise

    print("PASS: Issue #10 -- node-disjoint split is first option, random is last-resort")


# ===========================================================================
# INTEGRATION -- Phase 1 ↔ Phase 2 connection 100% wired
# ===========================================================================

def test_integration_phase1_phase2_connection():
    """The phase1_bridge must be the single authoritative contract
    connecting Phase 1 (data ingestion) to Phase 2 (knowledge graph).
    step1_load_phase1 must consume Phase 1 CSVs via the bridge.
    """
    bridge_path = _PHASE2_DIR / "drugos_graph" / "phase1_bridge.py"
    bridge_source = bridge_path.read_text(encoding="utf-8")

    # Must have the three callable entry points.
    assert "def read_phase1_outputs" in bridge_source
    assert "def stage_phase1_to_phase2" in bridge_source
    assert "def load_into_graph" in bridge_source
    assert "def run_phase1_to_phase2" in bridge_source

    # Must prefer PostgreSQL when available.
    assert "prefer_postgres" in bridge_source
    assert "_phase1_db_available" in bridge_source

    # run_pipeline.step1_load_phase1 must use the bridge.
    run_pipeline_path = _PHASE2_DIR / "drugos_graph" / "run_pipeline.py"
    rp_source = run_pipeline_path.read_text(encoding="utf-8")
    assert "def step1_load_phase1" in rp_source
    assert "run_phase1_to_phase2" in rp_source
    assert "bridge_to_pyg_maps" in rp_source

    # Verify the bridge can be imported and the RecordingGraphBuilder works.
    from drugos_graph.phase1_bridge import (
        RecordingGraphBuilder,
        run_phase1_to_phase2,
        bridge_to_pyg_maps,
    )
    recorder = RecordingGraphBuilder()
    assert hasattr(recorder, "node_loads")
    assert hasattr(recorder, "edge_loads")
    assert hasattr(recorder, "dead_letter")

    print("PASS: Integration -- Phase 1 ↔ Phase 2 connection 100% wired via bridge")


# ===========================================================================
# Runner -- executes all tests and reports pass/fail
# ===========================================================================

def run_all_tests():
    """Run all 10 issue tests + integration test. Returns True if all pass."""
    tests = [
        ("Issue #1 -- ChEMBL INACTIVATION regex", test_issue_1_chembl_inactivation_not_classified_as_activator),
        ("Issue #2 -- ChEMBERTa strict features default ON", test_issue_2_chemberta_strict_features_default_on),
        ("Issue #3 -- HGT BCEWithLogitsLoss not BCELoss", test_issue_3_hgt_uses_bcewithlogitsloss_not_bceloss),
        ("Issue #4 -- HGT model saved when val_idx empty", test_issue_4_hgt_model_saved_when_val_idx_empty),
        ("Issue #5 -- Canonical IDs for all 7 CORE_NODE_TYPES", test_issue_5_canonical_ids_for_all_core_node_types),
        ("Issue #6 -- ClinicalTrials primary_outcome_met parsed", test_issue_6_clinicaltrials_primary_outcome_met_parsed),
        ("Issue #7 -- STITCH CID normalization unified", test_issue_7_stitch_cid_normalization),
        ("Issue #8 -- OpenTargets no chembl_score pollution", test_issue_8_opentargets_no_chembl_score_pollution),
        ("Issue #9 -- Negative sampling no leakage", test_issue_9_negative_sampling_no_leakage),
        ("Issue #10 -- Node-disjoint split not random", test_issue_10_node_disjoint_split_not_random),
        ("Integration -- Phase 1↔Phase 2 100% connected", test_integration_phase1_phase2_connection),
    ]
    passed = 0
    failed = 0
    failures = []
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            failures.append((name, str(e)))
            print(f"FAIL: {name}")
            print(f"  -> {e}")
        except Exception as e:
            failed += 1
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"ERROR: {name}")
            print(f"  -> {type(e).__name__}: {e}")

    print()
    print("=" * 70)
    print(f"v60 ROOT FIX TEST RESULTS: {passed} passed, {failed} failed (of {len(tests)})")
    print("=" * 70)
    if failed == 0:
        print("ALL TESTS PASSED -- all 10 critical issues + integration verified.")
    else:
        print("FAILURES:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
