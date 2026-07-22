"""Task 115: Verify AUC against known inputs.

Tests that the AUC computation produces correct values for:
  - Perfect ranking (AUC = 1.0)
  - Random ranking (AUC ≈ 0.5)
  - Inverted ranking (AUC = 0.0)
  - Ties (Wilcoxon half-credit)

Also verifies the AUC is rank-based (O(n log n)), not O(n²).
"""
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugos_graph.evaluation import _manual_auc, compute_auc_direction_aware


class TestAUCComputation(unittest.TestCase):
    """Task 115: verify AUC against known inputs."""

    def test_perfect_ranking_auc_1(self):
        """Perfect separation: all pos scores < all neg scores → AUC = 1.0."""
        # TransE: lower score = more plausible
        pos = np.array([0.1, 0.2, 0.3])
        neg = np.array([0.8, 0.9, 1.0])
        auc = _manual_auc(pos, neg, higher_is_better=False)
        self.assertAlmostEqual(auc, 1.0, places=6,
                               msg="Perfect ranking should give AUC=1.0")

    def test_inverted_ranking_auc_0(self):
        """Inverted: all pos scores > all neg scores → AUC = 0.0."""
        pos = np.array([0.8, 0.9, 1.0])
        neg = np.array([0.1, 0.2, 0.3])
        auc = _manual_auc(pos, neg, higher_is_better=False)
        self.assertAlmostEqual(auc, 0.0, places=6,
                               msg="Inverted ranking should give AUC=0.0")

    def test_random_ranking_auc_05(self):
        """Random: identical distributions → AUC ≈ 0.5."""
        rng = np.random.default_rng(42)
        pos = rng.normal(0.5, 0.1, 1000)
        neg = rng.normal(0.5, 0.1, 1000)
        auc = _manual_auc(pos, neg, higher_is_better=False)
        self.assertAlmostEqual(auc, 0.5, delta=0.05,
                               msg="Random ranking should give AUC≈0.5")

    def test_higher_is_better_direction(self):
        """For higher_is_better=True, perfect ranking gives AUC=1.0
        when pos > neg."""
        pos = np.array([0.8, 0.9, 1.0])
        neg = np.array([0.1, 0.2, 0.3])
        auc = _manual_auc(pos, neg, higher_is_better=True)
        self.assertAlmostEqual(auc, 1.0, places=6)

    def test_ties_wilcoxon_half_credit(self):
        """Tied scores get Wilcoxon half-credit (0.5)."""
        # All scores tied → AUC = 0.5
        pos = np.array([0.5, 0.5, 0.5])
        neg = np.array([0.5, 0.5, 0.5])
        auc = _manual_auc(pos, neg, higher_is_better=False)
        self.assertAlmostEqual(auc, 0.5, places=6,
                               msg="All ties should give AUC=0.5 (half-credit)")

    def test_partial_ties(self):
        """Partial ties: verify correct Wilcoxon handling."""
        # pos = [1, 2, 3], neg = [2, 3, 4]
        # Ties at 2 and 3. Expected AUC with half-credit:
        # Pairs: (1,2)pos<, (1,3)pos<, (1,4)pos<, (2,2)tie, (2,3)pos<, (2,4)pos<,
        #         (3,2)pos>, (3,3)tie, (3,4)pos<
        # Wins (pos<neg): (1,2),(1,3),(1,4),(2,3),(2,4),(3,4) = 6
        # Ties: (2,2),(3,3) = 2 → half-credit = 1.0
        # Losses: (3,2) = 1
        # AUC = (6 + 1.0) / 9 = 7/9 ≈ 0.778
        pos = np.array([1.0, 2.0, 3.0])
        neg = np.array([2.0, 3.0, 4.0])
        auc = _manual_auc(pos, neg, higher_is_better=False)
        expected = 7.0 / 9.0
        self.assertAlmostEqual(auc, expected, places=6,
                               msg=f"Partial ties AUC should be {expected}")

    def test_auc_is_rank_based_on_log_n(self):
        """Task 107: AUC must be O(n log n), not O(n²).

        Verify by timing: doubling n should roughly double time (O(n log n)),
        not quadruple it (O(n²)).
        """
        rng = np.random.default_rng(42)
        # Small n
        pos_small = rng.normal(0.3, 0.15, 1000)
        neg_small = rng.normal(0.7, 0.15, 1000)
        t0 = time.time()
        for _ in range(10):
            _manual_auc(pos_small, neg_small, higher_is_better=False)
        t_small = time.time() - t0

        # Large n (10x)
        pos_large = rng.normal(0.3, 0.15, 10000)
        neg_large = rng.normal(0.7, 0.15, 10000)
        t0 = time.time()
        for _ in range(10):
            _manual_auc(pos_large, neg_large, higher_is_better=False)
        t_large = time.time() - t0

        # O(n log n): ratio should be ~10-15x (not 100x for O(n²))
        ratio = t_large / max(t_small, 1e-6)
        self.assertLess(
            ratio, 50,
            f"AUC scaling ratio {ratio:.1f}x for 10x input suggests O(n²) "
            f"complexity (expected <50x for O(n log n)). "
            f"t_small={t_small:.4f}s, t_large={t_large:.4f}s. "
            f"(task 107 root test — rank-based AUC must be O(n log n))",
        )

    def test_direction_aware_symmetric_filters_false_negatives(self):
        """Task 108: symmetric relations filter reversed positives from negatives."""
        # Positives: (A,B), (C,D)
        # Negatives: (B,A) [reversed positive — should be filtered], (E,F)
        pos = [("A", "B", 0.1), ("C", "D", 0.2)]
        neg = [("B", "A", 0.9), ("E", "F", 0.8)]  # (B,A) is reversed (A,B)

        auc = compute_auc_direction_aware(
            pos, neg, relation_type="symmetric", higher_is_better=False
        )
        # After filtering (B,A), only (E,F) remains as negative.
        # pos=[0.1, 0.2], neg=[0.8]. Perfect separation → AUC=1.0
        self.assertAlmostEqual(auc, 1.0, places=6,
                               msg="Symmetric: reversed positive filtered, AUC=1.0")

    def test_direction_aware_asymmetric_no_filtering(self):
        """Task 108: asymmetric relations do NOT filter reversed pairs."""
        pos = [("A", "B", 0.1), ("C", "D", 0.2)]
        neg = [("B", "A", 0.9), ("E", "F", 0.8)]

        auc = compute_auc_direction_aware(
            pos, neg, relation_type="asymmetric", higher_is_better=False
        )
        # No filtering: neg=[0.9, 0.8]. pos=[0.1, 0.2]. Perfect → AUC=1.0
        self.assertAlmostEqual(auc, 1.0, places=6)

    def test_direction_aware_invalid_relation_type_raises(self):
        """Task 108: invalid relation_type must raise."""
        pos = [("A", "B", 0.1)]
        neg = [("C", "D", 0.9)]
        from drugos_graph.exceptions import EvaluationInputError
        with self.assertRaises(EvaluationInputError):
            compute_auc_direction_aware(
                pos, neg, relation_type="invalid", higher_is_better=False
            )


if __name__ == "__main__":
    unittest.main()
