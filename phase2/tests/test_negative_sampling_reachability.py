"""Task 114: Verify multi-hop reachable pairs are not sampled as negatives.

Root test for the task 102 fix: NegativeSampler now accepts a
reachability_pairs parameter. Pairs reachable via drug→protein→pathway→disease
in the KG must NOT be sampled as negatives — they are biologically plausible
(unstudied but connected), not true negatives.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugos_graph.negative_sampling import NegativeSampler


class TestNegativeSamplingReachability(unittest.TestCase):
    """Task 114: verify multi-hop reachable pairs are not sampled."""

    def setUp(self):
        os.environ.pop("DRUGOS_ENVIRONMENT", None)
        os.environ.pop("DRUGOS_REGULATORY_MODE", None)

    def test_reachability_pairs_in_rejection_set(self):
        """Reachable pairs must be added to the rejection set."""
        all_drug_ids = ["D1", "D2", "D3", "D4", "D5", "D6"]
        all_disease_ids = ["DIS1", "DIS2", "DIS3", "DIS4", "DIS5", "DIS6"]
        positive_pairs = {("D1", "DIS1"), ("D2", "DIS2")}
        held_out_pairs = {("D3", "DIS3")}
        # Pairs reachable via drug→protein→pathway→disease (multi-hop)
        reachability_pairs = {("D4", "DIS4"), ("D5", "DIS5")}

        sampler = NegativeSampler(
            all_drug_ids, all_disease_ids, positive_pairs,
            seed=42,
            held_out_pairs=held_out_pairs,
            reachability_pairs=reachability_pairs,
        )

        # Verify reachability_pairs are in the rejection set
        self.assertIn(("D4", "DIS4"), sampler._rejection_pairs)
        self.assertIn(("D5", "DIS5"), sampler._rejection_pairs)
        # Also verify train + held_out are still in rejection set
        self.assertIn(("D1", "DIS1"), sampler._rejection_pairs)
        self.assertIn(("D3", "DIS3"), sampler._rejection_pairs)

    def test_reachable_pairs_never_sampled_as_negatives(self):
        """Multi-hop reachable pairs must NEVER appear in negatives."""
        all_drug_ids = ["D1", "D2", "D3", "D4", "D5", "D6"]
        all_disease_ids = ["DIS1", "DIS2", "DIS3", "DIS4", "DIS5", "DIS6"]
        positive_pairs = {("D1", "DIS1"), ("D2", "DIS2")}
        held_out_pairs = set()
        # Simulate multi-hop reachable pairs:
        # D3→P1→PW1→DIS3, D4→P2→PW2→DIS4, etc.
        reachability_pairs = {
            ("D3", "DIS3"), ("D4", "DIS4"), ("D5", "DIS5"),
            ("D6", "DIS6"),
        }

        sampler = NegativeSampler(
            all_drug_ids, all_disease_ids, positive_pairs,
            seed=42,
            held_out_pairs=held_out_pairs,
            reachability_pairs=reachability_pairs,
        )
        # Generate many negatives
        negatives = sampler.random_sampling(num_negatives=100)
        neg_pairs = {(n["drug_id"], n["disease_id"]) for n in negatives}

        # Verify NO reachable pair appears in negatives
        leaked = neg_pairs & reachability_pairs
        self.assertEqual(
            len(leaked), 0,
            f"LEAKAGE: {len(leaked)} multi-hop reachable pairs were sampled "
            f"as negatives: {leaked}. These are biologically plausible "
            f"(unstudied but connected via drug→protein→pathway→disease), "
            f"NOT true negatives. Sampling them as negatives teaches the "
            f"model that biologically connected pairs are negative, "
            f"inverting the learned ranking. (task 114 root test)",
        )

    def test_no_reachability_pairs_still_works(self):
        """When reachability_pairs=None, only train+held_out are rejected."""
        all_drug_ids = ["D1", "D2", "D3"]
        all_disease_ids = ["DIS1", "DIS2", "DIS3"]
        positive_pairs = {("D1", "DIS1")}
        held_out_pairs = {("D2", "DIS2")}

        sampler = NegativeSampler(
            all_drug_ids, all_disease_ids, positive_pairs,
            seed=42,
            held_out_pairs=held_out_pairs,
            reachability_pairs=None,
        )
        self.assertEqual(sampler.reachability_pairs, set())
        # Rejection set = positive + held_out only (no reachability)
        self.assertEqual(
            sampler._rejection_pairs,
            positive_pairs | held_out_pairs,
        )

    def test_reachability_simulates_drug_protein_pathway_disease(self):
        """End-to-end: simulate the drug→protein→pathway→disease chain.

        Build a small KG:
          D1 → targets → P1 → part_of → PW1 → disrupted_in → DIS1
          D2 → targets → P2 → part_of → PW2 → disrupted_in → DIS2

        (D1, DIS1) and (D2, DIS2) are reachable via 3-hop paths.
        They should NOT be sampled as negatives even though they are
        not direct positive "treats" edges.
        """
        all_drug_ids = ["D1", "D2", "D3"]
        all_disease_ids = ["DIS1", "DIS2", "DIS3"]
        # Only (D3, DIS3) is a known positive "treats" edge
        positive_pairs = {("D3", "DIS3")}
        held_out_pairs = set()
        # (D1, DIS1) and (D2, DIS2) are reachable via multi-hop
        # (drug→protein→pathway→disease) — computed upstream
        reachability_pairs = {("D1", "DIS1"), ("D2", "DIS2")}

        sampler = NegativeSampler(
            all_drug_ids, all_disease_ids, positive_pairs,
            seed=42,
            held_out_pairs=held_out_pairs,
            reachability_pairs=reachability_pairs,
        )
        negatives = sampler.random_sampling(num_negatives=50)
        neg_pairs = {(n["drug_id"], n["disease_id"]) for n in negatives}

        # (D1, DIS1) and (D2, DIS2) must NOT be in negatives
        self.assertNotIn(("D1", "DIS1"), neg_pairs,
                         "Multi-hop reachable (D1, DIS1) was sampled as negative")
        self.assertNotIn(("D2", "DIS2"), neg_pairs,
                         "Multi-hop reachable (D2, DIS2) was sampled as negative")


if __name__ == "__main__":
    unittest.main()
