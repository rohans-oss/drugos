"""Task 117: Verify graph-level disjoint split — no node feature leakage.

Root test for the task 104 fix: graph_level_split_pairs produces splits
where NO drug or disease appears in more than one split. This prevents
node feature leakage (the model learning a drug's embedding from train
edges, then using it for test edges).
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugos_graph.training_data import (
    graph_level_split_pairs,
    temporal_split_pairs,
)
from drugos_graph.exceptions import DrugOSDataError


class TestTrainingDataSplit(unittest.TestCase):
    """Task 117: verify graph-level disjoint split."""

    def setUp(self):
        os.environ.pop("DRUGOS_ENVIRONMENT", None)
        os.environ.pop("DRUGOS_ALLOW_TEMPORAL_RANDOM_FALLBACK", None)

    def _make_pairs(self, n_drugs=20, n_diseases=15, seed=42):
        """Generate synthetic positive pairs."""
        import random
        rng = random.Random(seed)
        pairs = []
        for d in range(n_drugs):
            # Each drug treats 1-3 diseases
            n_treats = rng.randint(1, 3)
            diseases = rng.sample(range(n_diseases), n_treats)
            for dis in diseases:
                pairs.append({"drug_id": f"D{d}", "disease_id": f"DIS{dis}"})
        return pairs

    def test_graph_level_split_is_disjoint(self):
        """Task 104: no drug or disease appears in more than one split."""
        pairs = self._make_pairs(n_drugs=30, n_diseases=20)
        result = graph_level_split_pairs(pairs, seed=42)

        train = result["train"]
        val = result["val"]
        test = result["test"]

        train_drugs = {p["drug_id"] for p in train}
        val_drugs = {p["drug_id"] for p in val}
        test_drugs = {p["drug_id"] for p in test}
        train_diseases = {p["disease_id"] for p in train}
        val_diseases = {p["disease_id"] for p in val}
        test_diseases = {p["disease_id"] for p in test}

        # Verify NO overlap between any pair of splits
        self.assertEqual(len(train_drugs & val_drugs), 0,
                         f"train-val drug overlap: {train_drugs & val_drugs}")
        self.assertEqual(len(train_drugs & test_drugs), 0,
                         f"train-test drug overlap: {train_drugs & test_drugs}")
        self.assertEqual(len(val_drugs & test_drugs), 0,
                         f"val-test drug overlap: {val_drugs & test_drugs}")
        self.assertEqual(len(train_diseases & val_diseases), 0,
                         f"train-val disease overlap")
        self.assertEqual(len(train_diseases & test_diseases), 0,
                         f"train-test disease overlap")
        self.assertEqual(len(val_diseases & test_diseases), 0,
                         f"val-test disease overlap")

    def test_graph_level_split_metadata(self):
        """Task 104: split metadata reports disjoint=True and component count."""
        pairs = self._make_pairs(n_drugs=20, n_diseases=15)
        result = graph_level_split_pairs(pairs, seed=42)
        meta = result["_split_metadata"]

        self.assertEqual(meta["method"], "graph_level_disjoint")
        self.assertTrue(meta["disjoint"], "disjoint must be True")
        self.assertGreater(meta["n_components"], 0, "must find ≥1 component")
        self.assertEqual(meta["train_count"], len(result["train"]))
        self.assertEqual(meta["val_count"], len(result["val"]))
        self.assertEqual(meta["test_count"], len(result["test"]))

    def test_graph_level_split_deterministic(self):
        """Task 103/104: same seed → same split (deterministic)."""
        pairs = self._make_pairs(n_drugs=20, n_diseases=15)
        r1 = graph_level_split_pairs(pairs, seed=42)
        r2 = graph_level_split_pairs(pairs, seed=42)

        # Same splits
        self.assertEqual(r1["train"], r2["train"])
        self.assertEqual(r1["val"], r2["val"])
        self.assertEqual(r1["test"], r2["test"])

    def test_graph_level_split_all_pairs_assigned(self):
        """All input pairs must be assigned to a split (no drops)."""
        pairs = self._make_pairs(n_drugs=20, n_diseases=15)
        n_input = len(pairs)
        result = graph_level_split_pairs(pairs, seed=42)
        n_output = (
            len(result["train"]) + len(result["val"]) + len(result["test"])
        )
        self.assertEqual(n_input, n_output,
                         f"Input {n_input} pairs but output {n_output} — "
                         f"pairs were dropped")

    def test_temporal_split_no_random_fallback(self):
        """Task 103: temporal_split_pairs must NOT fall back to random.

        When approval_years is missing, it must RAISE (not silently
        produce a random split).
        """
        pairs = self._make_pairs(n_drugs=10, n_diseases=5)
        with self.assertRaises(DrugOSDataError) as ctx:
            temporal_split_pairs(pairs, approval_years=None)
        self.assertIn("task 103", str(ctx.exception).lower() + str(ctx.exception))

    def test_temporal_split_with_approval_years_works(self):
        """Task 103: temporal split works when approval_years is provided."""
        pairs = self._make_pairs(n_drugs=20, n_diseases=10)
        # Assign approval years
        approval_years = {}
        for i, p in enumerate(pairs):
            if i < len(pairs) * 0.7:
                approval_years[(p["drug_id"], p["disease_id"])] = 2015
            elif i < len(pairs) * 0.85:
                approval_years[(p["drug_id"], p["disease_id"])] = 2019
            else:
                approval_years[(p["drug_id"], p["disease_id"])] = 2022

        result = temporal_split_pairs(pairs, approval_years=approval_years,
                                       cutoff_year=2020)
        meta = result["_split_metadata"]
        self.assertEqual(meta["method"], "temporal")
        self.assertFalse(meta["fell_back_to_random"],
                         "Must not fall back to random")

    def test_empty_pairs_raises(self):
        """Empty positive_pairs must raise."""
        with self.assertRaises(DrugOSDataError):
            graph_level_split_pairs([])


if __name__ == "__main__":
    unittest.main()
