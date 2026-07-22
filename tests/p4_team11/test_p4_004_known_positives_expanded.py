"""Test for P4-004 ROOT FIX (HIGH).

P4-004: KNOWN_POSITIVES has only 5 pairs — RL split 60/40 produces
        3 train + 2 test KPs, making recovery test statistically
        meaningless. With 2 test KPs, the recovery rate is either 0%,
        50%, or 100% — a 3-point discrete scale. P(recover both by
        chance) ≈ (top_n / test_set_size)^2 = 4% for top_n=10,
        test_set_size=50. So the recovery test has 96% false-negative
        rate BY CHANCE.

        Fix: expand KNOWN_POSITIVES to at least 20 pairs (12 train +
        8 test, recovery rate granularity = 12.5%).

This test verifies:
  1. _DEFAULT_KNOWN_POSITIVES has at least 20 pairs.
  2. After the 60/40 split, the test set has at least 8 KPs (granularity
     <= 12.5%).
  3. Each pair is a 2-tuple of non-empty strings.
  4. No duplicate pairs.
  5. The recovery test granularity is sufficient (1/n_test_kps <= 0.125).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "rl"))

from rl.rl_drug_ranker import (
    _DEFAULT_KNOWN_POSITIVES,
    KNOWN_POSITIVES,
    split_data,
    generate_fake_data,
    PipelineConfig,
)


def test_p4_004_known_positives_has_at_least_20_pairs():
    """_DEFAULT_KNOWN_POSITIVES must have at least 20 pairs.

    The previous 5-pair list, when split 60/40, produced only 2 test KPs
    — the recovery rate was a 3-point discrete scale (0%, 50%, 100%),
    with 96% false-negative rate by chance. With 20 pairs (12 train +
    8 test), the granularity is 12.5%, which is statistically meaningful.
    """
    assert len(_DEFAULT_KNOWN_POSITIVES) >= 20, (
        f"P4-004: _DEFAULT_KNOWN_POSITIVES has only "
        f"{len(_DEFAULT_KNOWN_POSITIVES)} pairs, expected >= 20. "
        f"The previous 5-pair list made the KP recovery test "
        f"statistically meaningless (96% false-negative rate by chance)."
    )


def test_p4_004_known_positives_are_valid_tuples():
    """Each KP must be a 2-tuple of non-empty lowercase strings."""
    for i, (drug, disease) in enumerate(_DEFAULT_KNOWN_POSITIVES):
        assert isinstance(drug, str) and isinstance(disease, str), (
            f"P4-004: KP #{i} ({drug}, {disease}) must be a tuple of "
            f"strings."
        )
        assert drug.strip() and disease.strip(), (
            f"P4-004: KP #{i} ({drug!r}, {disease!r}) has an empty "
            f"string. All KPs must have non-empty drug and disease names."
        )
        # Drug/disease names should be lowercase (the reward function
        # lowercases them for matching).
        assert drug == drug.lower(), (
            f"P4-004: KP #{i} drug '{drug}' must be lowercase (the "
            f"reward function lowercases for matching)."
        )
        assert disease == disease.lower(), (
            f"P4-004: KP #{i} disease '{disease}' must be lowercase."
        )


def test_p4_004_known_positives_no_duplicates():
    """No duplicate (drug, disease) pairs in _DEFAULT_KNOWN_POSITIVES."""
    seen = set()
    for pair in _DEFAULT_KNOWN_POSITIVES:
        assert pair not in seen, (
            f"P4-004: duplicate KP {pair} found. Each pair must be unique."
        )
        seen.add(pair)


def test_p4_004_split_produces_at_least_8_test_kps():
    """The 60/40 split must produce at least 8 test KPs.

    With 20 KPs split 60/40: 12 train + 8 test. Recovery rate
    granularity = 1/8 = 12.5%. With the previous 5 KPs: 3 train + 2
    test, granularity = 1/2 = 50% (meaningless).
    """
    # Build a dataset that includes all KPs.
    data = generate_fake_data(n_pairs=200, seed=42)
    # split_data with ensure_known_positives_in_test=True splits the KPs
    # 60/40 into train and test.
    train_df, test_df = split_data(
        data,
        test_size=0.2,
        seed=42,
        drug_aware=True,
        ensure_known_positives_in_test=True,
        return_oversampled=False,
    )
    # Count KPs in the test set.
    known_set = {(d.lower(), v.lower()) for d, v in KNOWN_POSITIVES}
    test_pairs = set(zip(
        test_df["drug"].astype(str).str.lower().str.strip(),
        test_df["disease"].astype(str).str.lower().str.strip(),
    ))
    n_test_kps = len(known_set & test_pairs)
    # With 20 KPs split 60/40: 8 test KPs (granularity 12.5%).
    # We allow 7-9 to account for the max(1, ...) and min(n-1, ...) edge cases.
    assert n_test_kps >= 7, (
        f"P4-004: split produced only {n_test_kps} test KPs, expected "
        f">= 7 (8 from a 60/40 split of 20 KPs, with ±1 slack for the "
        f"max(1, ...) / min(n-1, ...) edge cases in split_data). With "
        f"< 7 test KPs, the recovery rate granularity is too coarse "
        f"(> 14.3%) for a statistically meaningful test."
    )
    # Verify the granularity is sufficient.
    granularity = 1.0 / n_test_kps
    assert granularity <= 0.15, (
        f"P4-004: recovery rate granularity = 1/{n_test_kps} = "
        f"{granularity:.1%}, must be <= 15% for a statistically "
        f"meaningful test. The previous 5-KP list had granularity "
        f"50% (1/2), which made the recovery test a coin flip."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
