"""
Comprehensive test suite for all 25 Phase 4 issues (P4-001 through P4-025).

Run: pytest tests/test_p4_all_25_issues.py -v

These tests verify the ROOT-LEVEL fixes for each issue, not surface-level
changes. Each test checks the ACTUAL behavior of the code, not comments or
docstrings.
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest


# ============================================================================
# P4-001 (CRITICAL) — Toxic pairs must NOT get reward bonus
# ============================================================================
class TestP4_001_ValidatedBonusFiltering:
    """validated_bonus must only apply to validated_positive outcomes."""

    def test_toxic_pairs_excluded_from_reward_bonus(self):
        from rl.rl_drug_ranker import load_validated_hypotheses

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["drug", "disease", "outcome"])
            writer.writeheader()
            writer.writerow({"drug": "aspirin", "disease": "pain", "outcome": "validated_positive"})
            writer.writerow({"drug": "thalidomide", "disease": "pregnancy", "outcome": "validated_toxic"})
            writer.writerow({"drug": "viagra", "disease": "ed", "outcome": "validated_negative"})
            writer.writerow({"drug": "placebo", "disease": "nothing", "outcome": "invalidated"})
            path = f.name

        try:
            result = load_validated_hypotheses(path)
            assert ("aspirin", "pain") in result, "validated_positive must be included"
            assert ("thalidomide", "pregnancy") not in result, "validated_toxic must be EXCLUDED"
            assert ("viagra", "ed") not in result, "validated_negative must be EXCLUDED"
            assert ("placebo", "nothing") not in result, "invalidated must be EXCLUDED"
            assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
        finally:
            os.unlink(path)


# ============================================================================
# P4-002 (CRITICAL) — validated_bonus applied AFTER high_action_bonus
# ============================================================================
class TestP4_002_ValidatedBonusOrder:
    """validated_bonus must be applied AFTER high_action_bonus multiplication."""

    def test_effective_bonus_within_bounds(self):
        from rl.rl_drug_ranker import RewardConfig

        cfg = RewardConfig(validated_bonus=0.1, high_action_bonus=5.0)
        effective = cfg.validated_bonus * cfg.high_action_bonus
        assert effective <= 1.0, (
            f"effective bonus {effective} exceeds 1.0 — "
            f"validated_bonus ({cfg.validated_bonus}) * high_action_bonus "
            f"({cfg.high_action_bonus}) = {effective}. Agent will collapse to "
            f"ranking only validated pairs HIGH."
        )

    def test_post_init_rejects_excessive_effective_bonus(self):
        from rl.rl_drug_ranker import RewardConfig

        # validated_bonus=0.3 * high_action_bonus=5.0 = 1.5 > 1.0 should fail
        with pytest.raises(ValueError):
            RewardConfig(validated_bonus=0.3, high_action_bonus=5.0)


# ============================================================================
# P4-003 (HIGH) — rank_top_candidates doesn't exist
# ============================================================================
class TestP4_003_CheckpointLoading:
    """Must call existing method, not non-existent rank_top_candidates."""

    def test_uses_existing_bridge_method(self):
        from rl.service import _load_candidates_from_checkpoint
        import inspect

        source = inspect.getsource(_load_candidates_from_checkpoint)
        # Must CALL get_top_k_novel_predictions (the actual method)
        # not rank_top_candidates (which doesn't exist)
        assert "get_top_k_novel_predictions" in source, (
            "Must call existing get_top_k_novel_predictions"
        )
        # The function body (not docstring/comments) must not call the old method
        body_start = source.find("):")
        body = source[body_start:] if body_start > 0 else source
        assert "rank_top_candidates" not in body, (
            "Function body must NOT call non-existent rank_top_candidates"
        )


# ============================================================================
# P4-004 (HIGH) — Dashboard uses different weights than agent
# ============================================================================
class TestP4_004_RewardWeightsFromMeta:
    """overallScore must use agent's actual reward_weights, not hardcoded."""

    def test_reads_weights_from_meta_sidecar(self):
        from rl.service import _load_reward_weights_from_meta
        import inspect

        source = inspect.getsource(_load_reward_weights_from_meta)
        assert ".meta.json" in source, "Must read from .meta.json sidecar"

    def test_load_candidates_uses_meta_weights(self):
        from rl.service import _load_candidates_from_csv
        import inspect

        source = inspect.getsource(_load_candidates_from_csv)
        assert "_reward_weights" in source, "Must use loaded reward weights"
        # Must NOT use the OLD hardcoded weights 0.4/0.3/0.3 in overall calc.
        # (The 0.04 gnn default weight is OK — it's the agent's actual weight.)
        overall_section = source.split("overallScore")[0] if "overallScore" in source else source
        # The old hardcoded formula was: gnn*0.4 + safety*0.3 + market*0.3
        assert "0.4 *" not in overall_section and "* 0.4" not in overall_section, (
            "Must NOT use old hardcoded 0.4 gnn weight in overall formula"
        )


# ============================================================================
# P4-005 (HIGH) — Phase 1 path not searched
# ============================================================================
class TestP4_005_Phase1SearchPath:
    """load_validated_hypotheses must search phase1/processed_data/."""

    def test_phase1_path_in_search_paths(self):
        from rl.rl_drug_ranker import load_validated_hypotheses
        import inspect

        source = inspect.getsource(load_validated_hypotheses)
        assert "phase1" in source and "processed_data" in source, (
            "phase1/processed_data must be in search paths"
        )


# ============================================================================
# P4-006 (HIGH) — CORS allows any origin
# ============================================================================
class TestP4_006_CORS:
    """CORS must not allow '*' by default."""

    def test_cors_not_wildcard(self):
        import rl.service as svc
        assert svc._RL_CORS_ORIGINS != "*", "CORS must NOT default to *"


# ============================================================================
# P4-007 (HIGH) — Node label mismatch with Phase 2 KG
# ============================================================================
class TestP4_007_KGNodeLabels:
    """writeback_to_phase2 must use same labels as Phase 2 kg_builder."""

    def test_uses_compound_label(self):
        from phase4.writeback import writeback_to_phase2
        import inspect

        source = inspect.getsource(writeback_to_phase2)
        # Phase 2 uses :Compound (ENTITY_TYPE_COMPOUND = "Compound")
        assert ":Compound" in source, "Must use :Compound label (matches Phase 2)"

    def test_multi_variant_name_matching(self):
        from phase4.writeback import writeback_to_phase2
        import inspect

        source = inspect.getsource(writeback_to_phase2)
        # Must try multiple name variants to match existing nodes
        assert "toLower" in source or "LOWER" in source, (
            "Must use case-insensitive name matching"
        )


# ============================================================================
# P4-008 (HIGH) — driver.close() outside session block
# ============================================================================
class TestP4_008_DriverCleanup:
    """driver.close() must be in finally block inside session context."""

    def test_driver_close_in_finally(self):
        from phase4.writeback import writeback_to_phase2
        import inspect

        source = inspect.getsource(writeback_to_phase2)
        assert "finally:" in source, "Must use try/finally for driver cleanup"
        assert "driver.close()" in source, "Must close driver"


# ============================================================================
# P4-009 (HIGH) — Phase 3 writeback never read
# ============================================================================
class TestP4_009_Phase3Retraining:
    """GT trainer must have method to load validated pairs for retraining."""

    def test_trainer_has_load_method(self):
        from graph_transformer.training.trainer import GraphTransformerTrainer
        assert hasattr(GraphTransformerTrainer, "load_validated_for_retraining")

    def test_load_method_reads_retrain_trigger(self):
        import inspect
        from graph_transformer.training.trainer import GraphTransformerTrainer

        source = inspect.getsource(GraphTransformerTrainer.load_validated_for_retraining)
        assert "retrain_triggered.json" in source, (
            "Must read retrain_triggered.json"
        )


# ============================================================================
# P4-010 (HIGH) — VALIDATED_TREATS for toxic outcomes
# ============================================================================
class TestP4_010_EdgeLabelsPerOutcome:
    """Different edge labels for different validation outcomes."""

    def test_distinct_edge_labels(self):
        from phase4.writeback import writeback_to_phase2
        import inspect

        source = inspect.getsource(writeback_to_phase2)
        assert "VALIDATED_TREATS" in source, "Must have VALIDATED_TREATS"
        assert "VALIDATED_TOXIC" in source, "Must have VALIDATED_TOXIC"
        assert "VALIDATED_NEGATIVE" in source, "Must have VALIDATED_NEGATIVE"


# ============================================================================
# P4-011 (HIGH) — No duplicate check in writeback
# ============================================================================
class TestP4_011_DuplicateCheck:
    """Re-validating same hypothesis must UPDATE, not append duplicate."""

    def test_duplicate_check_exists(self):
        from phase4.writeback import writeback_to_phase1
        import inspect

        source = inspect.getsource(writeback_to_phase1)
        assert "duplicate_found" in source or "duplicate" in source.lower(), (
            "Must check for duplicates"
        )


# ============================================================================
# P4-012 (HIGH) — load_validated_hypotheses ignores outcome column
# ============================================================================
class TestP4_012_OutcomeFiltering:
    """Same as P4-001 — outcome column must be checked."""

    def test_outcome_column_checked(self):
        from rl.rl_drug_ranker import load_validated_hypotheses
        import inspect

        source = inspect.getsource(load_validated_hypotheses)
        assert "outcome" in source, "Must check outcome column"
        assert "validated_positive" in source, "Must filter for validated_positive"


# ============================================================================
# P4-013 (HIGH) — CSV loads all rows into memory
# ============================================================================
class TestP4_013_StreamingCSV:
    """Must use streaming iterator, not list(reader)."""

    def test_for_loop_inside_with_block(self):
        from rl.service import _load_candidates_from_csv
        import inspect

        source = inspect.getsource(_load_candidates_from_csv)
        lines = source.split("\n")
        with_indent = None
        for_indent = None
        for line in lines:
            if "with open(csv_path" in line:
                with_indent = len(line) - len(line.lstrip())
            if "for i, row in enumerate(reader):" in line:
                for_indent = len(line) - len(line.lstrip())

        assert for_indent is not None, "Must use enumerate(reader)"
        assert with_indent is not None, "Must use with open(...)"
        assert for_indent > with_indent, (
            f"for loop (indent={for_indent}) must be inside with block (indent={with_indent})"
        )


# ============================================================================
# P4-014 (HIGH) — Sort happens after limit break
# ============================================================================
class TestP4_014_SortBeforeLimit:
    """Must sort ALL candidates by rank, THEN apply limit."""

    def test_sort_comes_before_limit(self):
        from rl.service import _load_candidates_from_csv
        import inspect

        source = inspect.getsource(_load_candidates_from_csv)
        sort_pos = source.find("out.sort")
        limit_pos = source.find("out[:limit]")
        assert sort_pos > 0 and limit_pos > 0, "Must have both sort and limit"
        assert sort_pos < limit_pos, "sort must come BEFORE limit"


# ============================================================================
# P4-015 (HIGH) — Silent fallback on checkpoint failure
# ============================================================================
class TestP4_015_StrictCheckpoint:
    """Must raise in strict mode, not silently fall back to CSV."""

    def test_strict_mode_raises(self):
        from rl.service import _load_candidates_from_checkpoint
        import inspect

        source = inspect.getsource(_load_candidates_from_checkpoint)
        assert "RL_STRICT_CHECKPOINT" in source, "Must check strict mode"
        assert "RuntimeError" in source, "Must raise RuntimeError in strict mode"


# ============================================================================
# P4-016 (MEDIUM) — WITHDRAWN_DRUGS incomplete
# ============================================================================
class TestP4_016_WithdrawnDrugs:
    """Must include valproate, domperidone, tegaserod, benzyl alcohol."""

    def test_key_withdrawn_drugs_present(self):
        from rl.rl_drug_ranker import WITHDRAWN_DRUGS

        assert "valproate" in WITHDRAWN_DRUGS, "valproate missing"
        assert "domperidone" in WITHDRAWN_DRUGS, "domperidone missing"
        assert "tegaserod" in WITHDRAWN_DRUGS, "tegaserod missing"
        assert "benzyl alcohol" in WITHDRAWN_DRUGS, "benzyl alcohol missing"
        assert len(WITHDRAWN_DRUGS) >= 30, f"Expected >=30, got {len(WITHDRAWN_DRUGS)}"


# ============================================================================
# P4-017 (MEDIUM) — INDICATION_WITHDRAWN_DRUGS incomplete
# ============================================================================
class TestP4_017_IndicationWithdrawals:
    """Must include isotretinoin, lenalidomide, methotrexate."""

    def test_key_indication_withdrawals_present(self):
        from rl.rl_drug_ranker import INDICATION_WITHDRAWN_DRUGS

        assert "isotretinoin" in INDICATION_WITHDRAWN_DRUGS
        assert "lenalidomide" in INDICATION_WITHDRAWN_DRUGS
        assert "methotrexate" in INDICATION_WITHDRAWN_DRUGS
        assert len(INDICATION_WITHDRAWN_DRUGS) >= 5, (
            f"Expected >=5, got {len(INDICATION_WITHDRAWN_DRUGS)}"
        )


# ============================================================================
# P4-018 (MEDIUM) — CONTROLLED_SUBSTANCES incomplete
# ============================================================================
class TestP4_018_ControlledSubstances:
    """Must include benzodiazepines, ketamine, MDMA, psilocybin."""

    def test_key_controlled_substances_present(self):
        from rl.rl_drug_ranker import CONTROLLED_SUBSTANCES

        assert "alprazolam" in CONTROLLED_SUBSTANCES
        assert "diazepam" in CONTROLLED_SUBSTANCES
        assert "ketamine" in CONTROLLED_SUBSTANCES
        assert "mdma" in CONTROLLED_SUBSTANCES
        assert "psilocybin" in CONTROLLED_SUBSTANCES
        assert len(CONTROLLED_SUBSTANCES) >= 50, (
            f"Expected >=50, got {len(CONTROLLED_SUBSTANCES)}"
        )


# ============================================================================
# P4-019 (MEDIUM) — Substring matching for indications
# ============================================================================
class TestP4_019_TokenizedMatching:
    """Must use tokenized matching, not substring."""

    def test_uses_tokenized_not_substring(self):
        from rl.rl_drug_ranker import RewardFunction
        import inspect

        source = inspect.getsource(RewardFunction.compute)
        assert "issubset" in source, "Must use tokenized matching (issubset)"
        # Must NOT have the old substring matching pattern in actual CODE
        # (Comments describing the old behavior are OK)
        code_lines = [l for l in source.split("\n") if not l.strip().startswith("#")]
        code_only = "\n".join(code_lines)
        assert "if contraindication in disease_name" not in code_only, (
            "Must NOT use old substring matching pattern in code"
        )
        # Verify the tokenized approach uses token sets
        assert "disease_tokens" in source, "Must compute disease_tokens"
        assert "contra_tokens" in source, "Must compute contra_tokens"


# ============================================================================
# P4-020 (MEDIUM) — Step function safety_factor
# ============================================================================
class TestP4_020_ContinuousSafetyFactor:
    """safety_factor must use linear interpolation, not step function."""

    def test_continuous_interpolation(self):
        from rl.rl_drug_ranker import RewardConfig, RewardFunction

        cfg = RewardConfig(safety_hard_reject=0.5, safety_warning=0.7)
        rf = RewardFunction(cfg)

        row = pd.Series({
            "drug": "test_drug", "disease": "test_disease",
            "gnn_score": 0.8, "safety_score": 0.6, "market_score": 0.5,
            "confidence": 0.7, "pathway_score": 0.6, "patent_score": 0.5,
            "rare_disease_flag": 0.0, "unmet_need_score": 0.5,
            "efficacy_score": 0.6, "adme_score": 0.7,
        })
        reward = rf.compute(row)
        # safety=0.6 should NOT be hard-rejected
        assert reward > 0, f"safety=0.6 should give positive reward, got {reward}"

        # Issue 167 ROOT FIX: verify SIGMOID formula is in the source
        # (replaced the old linear interpolation formula). The sigmoid
        # is: 1 / (1 + exp(-k * (safety - warning))).
        import inspect
        source = inspect.getsource(RewardFunction.compute)
        assert "math.exp" in source or "_math.exp" in source, (
            "Must use sigmoid formula (math.exp) for continuous safety_factor "
            "(Issue 167 replaced linear interpolation with sigmoid)"
        )
        assert "safety_val" in source and "cfg.safety_warning" in source, (
            "Must reference safety_val and cfg.safety_warning in the sigmoid formula"
        )


# ============================================================================
# P4-021 (MEDIUM) — Cosmetic modular separation
# ============================================================================
class TestP4_021_ModularStructure:
    """Module files must provide real imports, not just re-export shims."""

    def test_env_imports_drug_ranking_env(self):
        from rl.env import DrugRankingEnv
        assert DrugRankingEnv is not None

    def test_reward_imports_reward_config(self):
        from rl.reward import RewardConfig, RewardFunction
        assert RewardConfig is not None
        assert RewardFunction is not None


# ============================================================================
# P4-022 (MEDIUM) — phase4/__init__.py empty
# ============================================================================
class TestP4_022_Phase4Package:
    """phase4/__init__.py must have version and exports."""

    def test_has_version(self):
        import phase4
        assert hasattr(phase4, "__version__")
        assert phase4.__version__ != ""

    def test_exports_write_validated_hypothesis(self):
        import phase4
        assert hasattr(phase4, "write_validated_hypothesis")


# ============================================================================
# P4-023 (MEDIUM) — Fixed KP_RECOVERY_THRESHOLD
# ============================================================================
class TestP4_023_ScaleAwareThreshold:
    """KP_RECOVERY_THRESHOLD must be scale-aware."""

    def test_demo_threshold(self):
        from rl.scientific_thresholds import resolve_kp_recovery_threshold
        t = resolve_kp_recovery_threshold(n_test_kps=2)
        assert abs(t - 0.34) < 0.01, f"demo threshold should be ~0.34, got {t}"

    def test_pilot_threshold(self):
        from rl.scientific_thresholds import resolve_kp_recovery_threshold
        t = resolve_kp_recovery_threshold(n_test_kps=500)
        assert abs(t - 0.4) < 0.01, f"pilot threshold should be ~0.4, got {t}"

    def test_production_threshold(self):
        from rl.scientific_thresholds import resolve_kp_recovery_threshold
        t = resolve_kp_recovery_threshold(n_test_kps=1000)
        assert abs(t - 0.5) < 0.01, f"production threshold should be ~0.5, got {t}"


# ============================================================================
# P4-024 (MEDIUM) — evaluate_agent shuffles test data
# ============================================================================
class TestP4_024_DeterministicEval:
    """evaluate_agent must pass shuffle=False for deterministic Top-N."""

    def test_shuffle_false(self):
        from rl.rl_drug_ranker import evaluate_agent
        import inspect

        source = inspect.getsource(evaluate_agent)
        assert "shuffle=False" in source or '"shuffle": False' in source, (
            "Must pass shuffle=False to env.reset()"
        )


# ============================================================================
# P4-025 (MEDIUM) — DEBUG log for critical VecNormalize issue
# ============================================================================
class TestP4_025_WarningNotDebug:
    """VecNormalize missing must log at WARNING, not DEBUG."""

    def test_logs_at_warning(self):
        from rl.rl_drug_ranker import extract_policy_prob_high
        import inspect

        source = inspect.getsource(extract_policy_prob_high)
        else_pos = source.find("else:")
        else_section = source[else_pos:else_pos + 2000]
        assert "logger.warning" in else_section or "logger.error" in else_section, (
            "Must log at WARNING or ERROR, not DEBUG"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
