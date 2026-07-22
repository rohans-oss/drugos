"""REAL CODE RUNNER for Team Member 12 P4-012 to P4-018 fixes.

This script runs the ACTUAL rl_drug_ranker pipeline (not a smoke test,
not a mock) to verify the fixes don't break runtime behavior. The pipeline
is run with:
  - block_on_scientific_failure=False (TEST-ONLY Python API escape hatch,
    since the CLI bypass was removed in P4-014)
  - RL_SKIP_LITERATURE=1 (TEST-ONLY env var, since biopython PubMed calls
    would fail in CI without network access)
  - minimal timesteps (256) and top-n (5) for speed

The pipeline should COMPLETE and produce output. If it CRASHES (not
raises ScientificFailureError — that's controlled), the fixes broke
something.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Repo root on sys.path
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# TEST-ONLY: skip literature cross-check (no PubMed network calls in CI)
os.environ["RL_SKIP_LITERATURE"] = "1"


def main() -> int:
    print("=" * 70)
    print("REAL CODE RUN: rl_drug_ranker pipeline (P4-012..P4-018 verification)")
    print("=" * 70)

    from rl.rl_drug_ranker import PipelineConfig, run_pipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build a config with TEST-ONLY escapes (the CLI bypass was
        # removed in P4-014; this Python API escape is for tests only).
        config = PipelineConfig(
            timesteps=256,        # minimal for speed
            top_n=5,
            seed=42,
            output_dir=os.path.join(tmpdir, "output"),
            checkpoint_dir=os.path.join(tmpdir, "checkpoints"),
            # TEST-ONLY: disable the scientific_validation gate so the
            # pipeline completes and produces output for inspection.
            # The CLI bypass (--allow-invalid-output) was removed in
            # P4-014; this Python API escape is the ONLY way to disable
            # the gate, and it is NOT reachable from the CLI.
            block_on_scientific_failure=False,
        )
        print(f"\nConfig:")
        print(f"  timesteps = {config.timesteps}")
        print(f"  top_n = {config.top_n}")
        print(f"  seed = {config.seed}")
        print(f"  min_kp_recovery_rate = {config.min_kp_recovery_rate}  (P4-013: should be 0.5)")
        print(f"  block_on_scientific_failure = {config.block_on_scientific_failure}")
        print(f"  output_dir = {config.output_dir}")

        print(f"\nRunning run_pipeline(config, seed=42)...")
        print(f"  (P4-015: seed is passed EXPLICITLY to run_pipeline)")

        try:
            candidates, metrics = run_pipeline(config, seed=42)
            print(f"\nRESULT: pipeline COMPLETED")
            print(f"  n_candidates = {len(candidates)}")
            print(f"  n_pairs_processed = {metrics.n_pairs_processed}")
            print(f"  n_ranked_high = {metrics.n_ranked_high}")
            print(f"  run_id = {metrics.run_id}")
            if candidates:
                print(f"\n  Top candidate: {candidates[0].drug} -> {candidates[0].disease}")
                print(f"    reward = {candidates[0].reward:.4f}")
                print(f"    literature_support = {candidates[0].literature_support}")

            # Check the output CSV exists
            import glob
            csvs = glob.glob(os.path.join(config.output_dir, "*.csv"))
            print(f"\n  Output CSVs: {len(csvs)}")
            for csv in csvs:
                print(f"    - {csv}")

            # Check the metadata JSON for scientific_validation
            import json
            jsons = glob.glob(os.path.join(config.output_dir, "*.json"))
            for jf in jsons:
                if "metadata" in jf or "manifest" in jf:
                    with open(jf) as f:
                        meta = json.load(f)
                    sv = meta.get("scientific_validation", {})
                    if sv:
                        print(f"\n  scientific_validation:")
                        print(f"    gt_test_auc = {sv.get('gt_test_auc')}")
                        print(f"    rl_auc = {sv.get('rl_auc')}")
                        print(f"    kp_recovery_rate = {sv.get('kp_recovery_rate')}")
                        print(f"    kp_recovery_pass = {sv.get('kp_recovery_pass')}")
                        print(f"    n_literature_supported = {sv.get('n_literature_supported')}")
                        print(f"    literature_check_skipped = {sv.get('literature_check_skipped')}")
                        print(f"    literature_check_failed_missing_biopython = {sv.get('literature_check_failed_missing_biopython')}")
                        print(f"    literature_pass = {sv.get('literature_pass')}")
                        print(f"    overall_pass = {sv.get('overall_pass')}")
                        print(f"    checks_passed = {sv.get('checks_passed')}")
                        print(f"    checks_failed = {sv.get('checks_failed')}")
                        print(f"    seed (P4-015) = {meta.get('seed')}")

            print("\n" + "=" * 70)
            print("REAL CODE RUN: SUCCESS (pipeline ran to completion)")
            print("=" * 70)
            return 0

        except Exception as e:
            print(f"\nRESULT: pipeline FAILED with {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("\n" + "=" * 70)
            print("REAL CODE RUN: FAILED (pipeline crashed — fixes may have broken something)")
            print("=" * 70)
            return 1


if __name__ == "__main__":
    sys.exit(main())
