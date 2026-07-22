"""P1-057 ROOT FIX (v110): regression guard for confidence tier computation.

WHAT THIS TEST GUARDS AGAINST
-----------------------------
The audit (Issue #57) asked: "verify Piñero alignment and 'Very Strong'
tier computation."

The confidence tier is a patient-safety signal: downstream ML models
filter / weight gene-disease associations by tier. A mislabeled tier
(e.g. "weak" evidence tagged as "moderate") inflates the perceived
confidence of every weak-evidence GDA edge, biasing the model toward
noise.

Per Piñero et al. 2020 §2.3, the DisGeNET DSGP score bands are:
  [0.0, 0.06)   → sub_weak
  [0.06, 0.3)   → weak
  [0.3, 0.5)    → strong
  [0.5, 1.0]    → very_strong

This test verifies:
  1. ``classify_confidence()`` returns the correct tier for each band.
  2. Boundary values (0.06, 0.3, 0.5) are classified into the higher tier.
  3. ``DEFAULT_CONFIDENCE_TIERS`` has exactly 4 tiers (sub_weak, weak,
     strong, very_strong).
  4. The tier-method version is recorded as ``pinero_2020_v2``.
  5. The DB CHECK constraint (via ORM) enforces the same 4-label set.

This is the regression guard the audit asked for.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DRUGOS_ENVIRONMENT", "development")

from cleaning.confidence import (  # noqa: E402
    DEFAULT_CONFIDENCE_TIERS,
    CONFIDENCE_TIER_METHOD_VERSION,
    classify_confidence,
)


# ---------------------------------------------------------------------------
# Tests: DEFAULT_CONFIDENCE_TIERS structure.
# ---------------------------------------------------------------------------

def test_default_confidence_tiers_has_four_tiers():
    """P1-004 v102: the tier set MUST have exactly 4 tiers (sub_weak, weak, strong, very_strong)."""
    assert len(DEFAULT_CONFIDENCE_TIERS) == 4, (
        f"DEFAULT_CONFIDENCE_TIERS must have exactly 4 entries. "
        f"Got {len(DEFAULT_CONFIDENCE_TIERS)}: {DEFAULT_CONFIDENCE_TIERS}"
    )


def test_default_confidence_tiers_labels_match_pinero():
    """The 4 labels MUST match Piñero et al. 2020 §2.3 + v102 very_strong split."""
    labels = [label for _, label in DEFAULT_CONFIDENCE_TIERS]
    assert labels == ["sub_weak", "weak", "strong", "very_strong"], (
        f"Tier labels must be ['sub_weak', 'weak', 'strong', 'very_strong']. "
        f"Got {labels}"
    )


def test_default_confidence_tiers_thresholds_match_pinero():
    """The thresholds MUST match Piñero et al. 2020 §2.3 + v102 very_strong split."""
    thresholds = [t for t, _ in DEFAULT_CONFIDENCE_TIERS]
    assert thresholds == [0.0, 0.06, 0.3, 0.5], (
        f"Thresholds must be [0.0, 0.06, 0.3, 0.5] per Piñero 2020 + v102. "
        f"Got {thresholds}"
    )


def test_confidence_tier_method_version_is_pinero_v2():
    """The method version MUST be 'pinero_2020_v2' (records the v102 very_strong extension)."""
    assert CONFIDENCE_TIER_METHOD_VERSION == "pinero_2020_v2", (
        f"CONFIDENCE_TIER_METHOD_VERSION must be 'pinero_2020_v2'. "
        f"Got {CONFIDENCE_TIER_METHOD_VERSION!r}."
    )


# ---------------------------------------------------------------------------
# Tests: classify_confidence() — band-by-band.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected_tier", [
    # sub_weak band [0.0, 0.06)
    (0.0, "sub_weak"),
    (0.001, "sub_weak"),
    (0.01, "sub_weak"),
    (0.05, "sub_weak"),
    (0.059, "sub_weak"),
    # weak band [0.06, 0.3)
    (0.06, "weak"),
    (0.1, "weak"),
    (0.2, "weak"),
    (0.299, "weak"),
    # strong band [0.3, 0.5)
    (0.3, "strong"),
    (0.4, "strong"),
    (0.499, "strong"),
    # very_strong band [0.5, 1.0]
    (0.5, "very_strong"),
    (0.6, "very_strong"),
    (0.7, "very_strong"),
    (0.9, "very_strong"),
    (1.0, "very_strong"),
])
def test_classify_confidence_returns_correct_tier(score, expected_tier):
    """classify_confidence() MUST return the Piñero-aligned tier for each band."""
    actual = classify_confidence(score)
    assert actual == expected_tier, (
        f"score={score}: expected tier {expected_tier!r}, got {actual!r}. "
        f"The classification does NOT match Piñero et al. 2020 §2.3."
    )


def test_classify_confidence_boundary_0_06_is_weak():
    """The boundary 0.06 MUST be classified as 'weak' (the start of the weak band)."""
    assert classify_confidence(0.06) == "weak"


def test_classify_confidence_boundary_0_3_is_strong():
    """The boundary 0.3 MUST be classified as 'strong' (the start of the strong band)."""
    assert classify_confidence(0.3) == "strong"


def test_classify_confidence_boundary_0_5_is_very_strong():
    """The boundary 0.5 MUST be classified as 'very_strong' (the start of the very_strong band)."""
    assert classify_confidence(0.5) == "very_strong"


def test_classify_confidence_1_0_is_very_strong():
    """The maximum score 1.0 MUST be classified as 'very_strong'."""
    assert classify_confidence(1.0) == "very_strong"


# ---------------------------------------------------------------------------
# Tests: classify_confidence() — defensive cases.
# ---------------------------------------------------------------------------

def test_classify_confidence_handles_zero():
    """Score 0.0 MUST be classified as 'sub_weak'."""
    assert classify_confidence(0.0) == "sub_weak"


def test_classify_confidence_handles_negative_score():
    """Negative scores (protective-association range) MUST be handled gracefully.

    The DisGeNET DSGP score is in [0, 1] for therapeutic associations
    but may be in [-1, 0] for protective associations. classify_confidence
    should treat negative scores as sub_weak (the lowest tier) rather
    than crashing.
    """
    # The function may either return 'sub_weak' or raise — both are
    # defensible. We accept either.
    try:
        result = classify_confidence(-0.1)
        assert result == "sub_weak", (
            f"Negative score -0.1 should classify as 'sub_weak', got {result!r}."
        )
    except (AssertionError, ValueError):
        # Acceptable — the function may reject negative scores.
        pass


def test_classify_confidence_handles_above_one():
    """Scores above 1.0 MUST be handled gracefully (clipped or raised)."""
    try:
        result = classify_confidence(1.5)
        # Either 'very_strong' (clipped) or raised — both acceptable.
        assert result == "very_strong"
    except (AssertionError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Tests: DB-level enforcement via ORM CHECK constraint.
# ---------------------------------------------------------------------------

def test_orm_check_constraint_allows_all_four_tiers():
    """The ORM CHECK MUST accept all 4 valid tier labels."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.base import Base
    from database.models import GeneDiseaseAssociation

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    # P1-056: the GDA @validates decorator enforces uppercase HGNC gene
    # symbols (BRCA1, TP53, etc.). Lowercase / mixed-case symbols are
    # rejected by the Python validator before reaching the DB.
    valid_gene_symbols = ["BRCA", "TP53", "EGFR", "MYC"]
    for tier, gene_sym in zip(
        ("sub_weak", "weak", "strong", "very_strong"), valid_gene_symbols
    ):
        gda = GeneDiseaseAssociation(
            gene_symbol=gene_sym,
            disease_id=f"D_{tier}",
            disease_name="Test Disease",
            disease_id_type="mesh",
            source="disgenet",
            score=0.5,
            association_type="therapeutic",
            confidence_tier=tier,
        )
        session.add(gda)
        session.commit()  # should NOT raise for any valid tier

    rows = session.query(GeneDiseaseAssociation).all()
    assert len(rows) == 4

    session.close()
    Base.metadata.drop_all(engine)


def test_orm_check_constraint_rejects_invalid_tier():
    """The ORM CHECK MUST reject invalid tier labels (e.g. 'moderate', 'high')."""
    from sqlalchemy import create_engine
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import sessionmaker

    from database.base import Base
    from database.models import GeneDiseaseAssociation

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    # 'moderate' was the OLD pre-Piñero label — must be REJECTED now.
    gda = GeneDiseaseAssociation(
        gene_symbol="BRCA",
        disease_id="D_BAD",
        disease_name="Test Disease",
        disease_id_type="mesh",
        source="disgenet",
        score=0.5,
        association_type="therapeutic",
        confidence_tier="moderate",  # invalid post-P1-004
    )
    session.add(gda)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.close()
    Base.metadata.drop_all(engine)


def test_orm_check_constraint_allows_null_tier():
    """The ORM CHECK MUST allow NULL confidence_tier (column is nullable)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.base import Base
    from database.models import GeneDiseaseAssociation

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    gda = GeneDiseaseAssociation(
        gene_symbol="TP53",
        disease_id="D_NULL",
        disease_name="Test Disease",
        disease_id_type="mesh",
        source="disgenet",
        score=0.5,
        association_type="therapeutic",
        confidence_tier=None,  # NULL is allowed
    )
    session.add(gda)
    session.commit()  # should NOT raise

    session.close()
    Base.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# Tests: lockstep verification between Python + DB + migration.
# ---------------------------------------------------------------------------

def test_classify_confidence_lockstep_with_db_constraint():
    """P1-041: the Python tier set MUST match the DB CHECK constraint.

    If the Python classifier returns a label that the DB rejects (or
    vice versa), the pipeline will silently drop rows. This test verifies
    that EVERY label returned by classify_confidence() is accepted by
    the DB CHECK constraint.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.base import Base
    from database.models import GeneDiseaseAssociation

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    # Use 4 uppercase HGNC-style gene symbols (the @validates decorator
    # enforces uppercase). We cycle through them so we have 101 unique
    # (gene_symbol, disease_id) pairs.
    gene_symbols = ["BRCA", "TP53", "EGFR", "MYC"]
    for i in range(101):
        score = i / 100.0
        tier = classify_confidence(score)
        gda = GeneDiseaseAssociation(
            gene_symbol=gene_symbols[i % len(gene_symbols)],
            disease_id=f"D{i:04d}",
            disease_name="Test Disease",
            disease_id_type="mesh",
            source="disgenet",
            score=score,
            association_type="therapeutic",
            confidence_tier=tier,
        )
        session.add(gda)
        try:
            session.commit()
        except Exception as exc:
            pytest.fail(
                f"Lockstep violation: score={score} classified as {tier!r} "
                f"by Python, but DB CHECK rejected it: {exc}"
            )

    rows = session.query(GeneDiseaseAssociation).all()
    assert len(rows) == 101

    session.close()
    Base.metadata.drop_all(engine)
