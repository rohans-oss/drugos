"""P1-045 ROOT FIX (v108): CI test for withdrawn-drug list freshness.

The code comment in ``phase1/database/loaders.py`` claimed a CI test
``tests/test_p1_045_withdrawn_drug_list_freshness.py`` existed, but it
DID NOT. This is the kind of "aspirational fix" the audit warned about:
the comment claimed a CI check exists, but no test file was ever created.

This test file provides the REAL CI check:

1. **Freshness check**: the ``WITHDRAWN_DRUG_LIST_LAST_VERIFIED`` date
   must be within ``WITHDRAWN_DRUG_LIST_MAX_AGE_DAYS`` (90 days) of
   today. If the list is stale, the test FAILS the build — prompting
   operators to re-diff against the FDA database.

2. **Completeness check**: the list must contain a set of KNOWN
   FDA-withdrawn drugs (the "canary" set). If any canary drug is
   missing, the test FAILS — the list has been corrupted or trimmed.

3. **Structural check**: the list must be a non-empty frozenset of
   lowercase strings (no None, no duplicates, no mixed-case).

4. **Source URL check**: the ``WITHDRAWN_DRUG_LIST_SOURCE_URL`` must
   point to the FDA CDER database (the canonical source).

This test runs in CI on every push and on a weekly schedule. If it
fails, the build is blocked until an operator re-verifies the list
against the FDA database and updates ``WITHDRAWN_DRUG_LIST_LAST_VERIFIED``.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure phase1 is importable
_PHASE1_ROOT = Path(__file__).resolve().parent.parent / "phase1"
if str(_PHASE1_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHASE1_ROOT))


# ============================================================================
# Canary set: KNOWN FDA-withdrawn drugs that MUST be in the list.
# These are high-profile withdrawals that any complete list must include.
# Sourced from FDA CDER withdrawn-drug database (accessdata.fda.gov).
# ============================================================================
CANARY_WITHDRAWN_DRUGS = frozenset({
    # Cox-2 inhibitors (cardiovascular toxicity)
    "rofecoxib",      # Vioxx - withdrawn 2004
    "valdecoxib",     # Bextra - withdrawn 2005
    # Statins (rhabdomyolysis)
    "cerivastatin",   # Baycol - withdrawn 2001
    # Thiazolidinediones (hepatotoxicity)
    "troglitazone",   # Rezulin - withdrawn 2000
    # Appetite suppressants (cardiac valve damage / PAH)
    "fenfluramine",   # Pondimin - withdrawn 1997
    "dexfenfluramine", # Redux - withdrawn 1997
    "sibutramine",    # Meridia - withdrawn 2010
    # GI pro-kinetics (fatal arrhythmia)
    "cisapride",      # Propulsid - withdrawn 2000
    # Analgesics (hepatotoxicity)
    "bromfenac",      # Duract - withdrawn 1998
    # Antihistamines (QT prolongation)
    "terfenadine",    # Seldane - withdrawn 1998
    "astemizole",     # Hismanal - withdrawn 1999
    # Fluoroquinolones (QT prolongation / hepatotoxicity)
    "grepafloxacin",  # Raxar - withdrawn 1999
    "trovafloxacin",  # Trovan - withdrawn 1999
    "temafloxacin",   # Omniflox - withdrawn 1992
})


class TestP1_045_WithdrawnDrugListFreshness:
    """P1-045: verify the withdrawn-drug list is fresh and complete."""

    def test_list_is_non_empty_frozenset(self):
        """The list must be a non-empty frozenset of lowercase strings."""
        from database.loaders import _WITHDRAWN_DRUG_NAMES_LOWER
        assert isinstance(_WITHDRAWN_DRUG_NAMES_LOWER, frozenset), (
            f"_WITHDRAWN_DRUG_NAMES_LOWER must be a frozenset, "
            f"got {type(_WITHDRAWN_DRUG_NAMES_LOWER).__name__}"
        )
        assert len(_WITHDRAWN_DRUG_NAMES_LOWER) > 0, (
            "Withdrawn drug list is empty — patient safety is at risk"
        )
        # All entries must be lowercase strings
        for name in _WITHDRAWN_DRUG_NAMES_LOWER:
            assert isinstance(name, str), (
                f"Withdrawn drug entry {name!r} is not a string"
            )
            assert name == name.lower(), (
                f"Withdrawn drug entry {name!r} is not lowercase"
            )
            assert len(name) > 0, "Empty string in withdrawn drug list"

    def test_last_verified_date_is_valid_iso8601(self):
        """The last-verified date must be a valid ISO 8601 date."""
        from database.loaders import WITHDRAWN_DRUG_LIST_LAST_VERIFIED
        assert WITHDRAWN_DRUG_LIST_LAST_VERIFIED is not None
        # Must parse as YYYY-MM-DD
        datetime.strptime(WITHDRAWN_DRUG_LIST_LAST_VERIFIED, "%Y-%m-%d")

    def test_list_is_not_stale(self):
        """The list must be re-verified within 90 days (freshness check)."""
        from database.loaders import (
            WITHDRAWN_DRUG_LIST_LAST_VERIFIED,
            WITHDRAWN_DRUG_LIST_MAX_AGE_DAYS,
        )
        verified = datetime.strptime(
            WITHDRAWN_DRUG_LIST_LAST_VERIFIED, "%Y-%m-%d"
        ).replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - verified).days
        assert age_days <= WITHDRAWN_DRUG_LIST_MAX_AGE_DAYS, (
            f"Withdrawn drug list is {age_days} days old "
            f"(max {WITHDRAWN_DRUG_LIST_MAX_AGE_DAYS}). "
            f"Re-diff against FDA database at "
            f"https://accessdata.fda.gov/scripts/cder/daf/ and update "
            f"WITHDRAWN_DRUG_LIST_LAST_VERIFIED in database/loaders.py."
        )

    def test_canary_drugs_are_present(self):
        """Known FDA-withdrawn drugs MUST be in the list (completeness check)."""
        from database.loaders import _WITHDRAWN_DRUG_NAMES_LOWER
        missing = CANARY_WITHDRAWN_DRUGS - _WITHDRAWN_DRUG_NAMES_LOWER
        assert not missing, (
            f"Withdrawn drug list is MISSING {len(missing)} known "
            f"FDA-withdrawn drugs: {sorted(missing)}. These drugs MUST "
            f"be in the list — a missing entry means the drug is loaded "
            f"with is_withdrawn=False, and the RL ranker could recommend "
            f"it as a repurposing candidate (patient safety risk). "
            f"Re-diff against the FDA CDER withdrawn-drug database."
        )

    def test_source_url_points_to_fda(self):
        """The source URL must point to the FDA CDER database."""
        from database.loaders import WITHDRAWN_DRUG_LIST_SOURCE_URL
        assert "accessdata.fda.gov" in WITHDRAWN_DRUG_LIST_SOURCE_URL, (
            f"WITHDRAWN_DRUG_LIST_SOURCE_URL must point to the FDA CDER "
            f"database, got {WITHDRAWN_DRUG_LIST_SOURCE_URL}"
        )

    def test_max_age_is_90_days(self):
        """The max age must be 90 days (quarterly re-verification)."""
        from database.loaders import WITHDRAWN_DRUG_LIST_MAX_AGE_DAYS
        assert WITHDRAWN_DRUG_LIST_MAX_AGE_DAYS == 90, (
            f"WITHDRAWN_DRUG_LIST_MAX_AGE_DAYS must be 90, "
            f"got {WITHDRAWN_DRUG_LIST_MAX_AGE_DAYS}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
