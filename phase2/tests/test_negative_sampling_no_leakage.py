"""Task 113: Verify no val/test positive appears as a negative sample.

Root test for the task 101 fix: NegativeSampler now REQUIRES held_out_pairs
explicitly (no None default). Val/test positives MUST be excluded from
negative sampling — if any val/test positive appears in the generated
negatives, the test fails.
"""
import os
import sys
import unittest
from pathlib import Path

# Add phase2 to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugos_graph.negative_sampling import NegativeSampler


class TestNegativeSamplingNoLeakage(unittest.TestCase):
    """Task 113: verify no val/test positive appears as a negative."""

    def setUp(self):
        os.environ.pop("DRUGOS_ENVIRONMENT", None)
        os.environ.pop("DRUGOS_REGULATORY_MODE", None)
        os.environ.pop("DRUGOS_DETERMINISTIC_MODE", None)

    def test_held_out_pairs_required_no_default(self):
        """Task 101: NegativeSampler must NOT have a default for held_out_pairs."""
        import inspect
        sig = inspect.signature(NegativeSampler.__init__)
        held_out_param = sig.parameters.get("held_out_pairs")
        self.assertIsNotNone(held_out_param, "held_out_pairs must be a parameter")
        self.assertEqual(
            held_out_param.default, inspect.Parameter.empty,
            "held_out_pairs must NOT have a default value (task 101). "
            f"Got default={held_out_param.default!r}.",
        )

    def test_held_out_pairs_none_raises_typeerror(self):
        """Task 101: passing None explicitly must raise TypeError."""
        all_drug_ids = ["D1", "D2", "D3", "D4", "D5"]
        all_disease_ids = ["DIS1", "DIS2", "DIS3", "DIS4", "DIS5"]
        positive_pairs = {("D1", "DIS1"), ("D2", "DIS2")}
        with self.assertRaises(TypeError) as ctx:
            NegativeSampler(
                all_drug_ids, all_disease_ids, positive_pairs,
                held_out_pairs=None,  # explicit None must raise
            )
        self.assertIn("task 101", str(ctx.exception).lower() + str(ctx.exception))

    def test_val_test_positives_never_sampled_as_negatives(self):
        """Task 101/113: held-out positives must NEVER appear in negatives."""
        all_drug_ids = ["D1", "D2", "D3", "D4", "D5"]
        all_disease_ids = ["DIS1", "DIS2", "DIS3", "DIS4", "DIS5"]
        # Train positives
        positive_pairs = {("D1", "DIS1"), ("D2", "DIS2"), ("D3", "DIS3")}
        # Held-out (val + test) positives — these MUST NOT appear in negatives
        held_out_pairs = {("D4", "DIS4"), ("D5", "DIS5")}

        sampler = NegativeSampler(
            all_drug_ids, all_disease_ids, positive_pairs,
            seed=42, held_out_pairs=held_out_pairs,
        )
        negatives = sampler.random_sampling(num_negatives=50)

        # Extract (drug, disease) from generated negatives
        neg_pairs = {(n["drug_id"], n["disease_id"]) for n in negatives}

        # Verify NO held-out positive appears in negatives
        leaked = neg_pairs & held_out_pairs
        self.assertEqual(
            len(leaked), 0,
            f"LEAKAGE: {len(leaked)} held-out positives were sampled as "
            f"negatives: {leaked}. This is the exact bug task 101/113 "
            f"prevents. (task 113 root test)",
        )

    def test_train_positives_also_excluded(self):
        """Train positives must also not appear in negatives."""
        all_drug_ids = ["D1", "D2", "D3"]
        all_disease_ids = ["DIS1", "DIS2", "DIS3"]
        positive_pairs = {("D1", "DIS1"), ("D2", "DIS2")}
        held_out_pairs = {("D3", "DIS3")}

        sampler = NegativeSampler(
            all_drug_ids, all_disease_ids, positive_pairs,
            seed=42, held_out_pairs=held_out_pairs,
        )
        negatives = sampler.random_sampling(num_negatives=30)
        neg_pairs = {(n["drug_id"], n["disease_id"]) for n in negatives}

        leaked_train = neg_pairs & positive_pairs
        self.assertEqual(
            len(leaked_train), 0,
            f"Train positives leaked into negatives: {leaked_train}",
        )

    def test_empty_held_out_allowed(self):
        """Empty set is allowed (for pure-train samplers with no val/test)."""
        all_drug_ids = ["D1", "D2", "D3"]
        all_disease_ids = ["DIS1", "DIS2", "DIS3"]
        positive_pairs = {("D1", "DIS1"), ("D2", "DIS2")}
        # Empty set — explicitly acknowledges no held-out split
        sampler = NegativeSampler(
            all_drug_ids, all_disease_ids, positive_pairs,
            seed=42, held_out_pairs=set(),
        )
        self.assertEqual(sampler.held_out_pairs, set())
        self.assertEqual(sampler._rejection_pairs, positive_pairs)


if __name__ == "__main__":
    unittest.main()
