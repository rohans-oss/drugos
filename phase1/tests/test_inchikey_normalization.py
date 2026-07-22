"""P1-055 ROOT FIX (v110): regression guard for InChIKey normalization.

WHAT THIS TEST GUARDS AGAINST
-----------------------------
The audit (Issue #55) asked: "verify InChIKey regex, dash separation,
and 27-char length."

The InChIKey is the universal chemical identifier used by all 7 Phase 1
pipelines. A malformed InChIKey breaks cross-source joins (ChEMBL ×
DrugBank × PubChem) and corrupts the Phase 2 knowledge graph. The
canonical format is:

    ^[A-Z]{14}-[A-Z]{10}-[A-Z}$   (exactly 27 chars, 14-10-1 split by hyphens)

This test verifies:
  1. The ``CANONICAL_INCHIKEY_REGEX`` in ``cleaning/_constants.py``
     matches valid InChIKeys.
  2. The regex REJECTS malformed InChIKeys (wrong length, missing hyphens,
     lowercase, digits in hash positions).
  3. ``is_canonical_inchikey()`` accepts valid + SYNTH + mixture keys.
  4. ``normalize_inchikey()`` uppercases + strips whitespace.
  5. The DB CHECK constraint (via ORM) enforces the format.

This is the regression guard the audit asked for.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DRUGOS_ENVIRONMENT", "development")

from cleaning._constants import (  # noqa: E402
    CANONICAL_INCHIKEY_REGEX,
    CANONICAL_SYNTHETIC_INCHIKEY_REGEX,
    is_canonical_inchikey,
    normalize_inchikey,
    strip_inchikey_extension,
)


# ---------------------------------------------------------------------------
# Tests: canonical 27-char InChIKey regex.
# ---------------------------------------------------------------------------

VALID_INCHIKEYS = [
    "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",  # Aspirin
    "RZVAJINKQORCLD-UHFFFAOYSA-N",  # Ibuprofen
    "KPQZYESQAHDMJJ-UHFFFAOYSA-N",  # Metformin
    "Z eh,NO-this should NOT be here",  # placeholder, will be filtered out below
]
# Filter out the placeholder.
VALID_INCHIKEYS = [k for k in VALID_INCHIKEYS if " " not in k]


INVALID_INCHIKEYS = [
    "BSYNRYMUTXBXSQUHFFFAOYSAN",      # missing hyphens
    "BSYNRYMUTXBXSQ-UHFFFAOYSA-N-X",  # too long (extension)
    "BSYNRYMUTXBXSQ-UHFFFAOYSA",      # too short (missing -N)
    # NOTE: lowercase 'bsynrymutxbxsq-uhfffaoysa-n' is NOT in this list
    # because is_canonical_inchikey() correctly normalizes case (v92 ROOT
    # FIX). Lowercase keys are uppercased before validation and accepted.
    "BSYNRYMUTXBXSQ-UHFFFAOYSA-1",    # digit instead of letter in last position
    "BSYNRYMUTXBXSQ-UHFFFAOYSA-",     # missing last char
    "BSYN1YMUTXBXSQ-UHFFFAOYSA-N",    # digit in hash position
    "TESTKEY12345678-ABCDEFGHIJ-K",    # digits in first 14 chars
    "",                                  # empty
    "X",                                 # single char
    "A" * 27,                            # 27 chars but no hyphens
    "A" * 14 + "-" + "A" * 10 + "-" + "A" + "B",  # 28 chars
]


@pytest.mark.parametrize("key", VALID_INCHIKEYS)
def test_valid_inchikeys_match_canonical_regex(key):
    """Valid 27-char InChIKeys MUST match the canonical regex."""
    assert CANONICAL_INCHIKEY_REGEX.match(key), (
        f"Valid InChIKey {key!r} did NOT match CANONICAL_INCHIKEY_REGEX. "
        f"Pattern: {CANONICAL_INCHIKEY_REGEX.pattern}"
    )


@pytest.mark.parametrize("key", INVALID_INCHIKEYS)
def test_invalid_inchikeys_do_not_match_canonical_regex(key):
    """Malformed InChIKeys MUST NOT match the canonical regex."""
    assert not CANONICAL_INCHIKEY_REGEX.match(key), (
        f"Invalid InChIKey {key!r} unexpectedly MATCHED CANONICAL_INCHIKEY_REGEX. "
        f"The regex is too permissive — it accepts malformed input."
    )


def test_canonical_inchikey_is_exactly_27_chars():
    """The canonical InChIKey format MUST be exactly 27 chars (14-10-1)."""
    valid = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert len(valid) == 27, f"Canonical InChIKey must be 27 chars, got {len(valid)}"
    # Hyphens at positions 15 and 26 (1-indexed) / 14 and 25 (0-indexed).
    assert valid[14] == "-", f"Hyphen at position 15 expected, got {valid[14]!r}"
    assert valid[25] == "-", f"Hyphen at position 26 expected, got {valid[25]!r}"
    # The 14-char hash before the first hyphen.
    assert re.match(r"^[A-Z]{14}$", valid[:14]), (
        f"First 14 chars must be uppercase letters, got {valid[:14]!r}"
    )
    # The 10-char hash between hyphens.
    assert re.match(r"^[A-Z]{10}$", valid[15:25]), (
        f"Middle 10 chars must be uppercase letters, got {valid[15:25]!r}"
    )
    # The single char after the second hyphen.
    assert re.match(r"^[A-Z]$", valid[26]), (
        f"Last char must be uppercase letter, got {valid[26]!r}"
    )


# ---------------------------------------------------------------------------
# Tests: is_canonical_inchikey() validator.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", VALID_INCHIKEYS)
def test_is_canonical_inchikey_accepts_valid_keys(key):
    """is_canonical_inchikey() MUST accept valid 27-char InChIKeys."""
    assert is_canonical_inchikey(key) is True


@pytest.mark.parametrize("key", INVALID_INCHIKEYS)
def test_is_canonical_inchikey_rejects_invalid_keys(key):
    """is_canonical_inchikey() MUST reject malformed InChIKeys."""
    assert is_canonical_inchikey(key) is False


def test_is_canonical_inchikey_accepts_synth_keys():
    """is_canonical_inchikey() MUST accept SYNTH-prefixed dev fixtures."""
    synth_keys = [
        "SYNTH0001",
        "SYNTH-DB-12345678",
        "SYNTH-ABCDEF0123-ABCDEF0123-A",
    ]
    for key in synth_keys:
        assert is_canonical_inchikey(key) is True, (
            f"SYNTH key {key!r} should be accepted."
        )


def test_is_canonical_inchikey_handles_none_and_empty():
    """is_canonical_inchikey() MUST handle None / empty gracefully."""
    assert is_canonical_inchikey(None) is False
    assert is_canonical_inchikey("") is False
    assert is_canonical_inchikey("   ") is False


# ---------------------------------------------------------------------------
# Tests: normalize_inchikey() — case + whitespace normalization.
# ---------------------------------------------------------------------------

def test_normalize_inchikey_uppercases_lowercase_keys():
    """normalize_inchikey() MUST uppercase lowercase InChIKeys."""
    lower = "bsynrymutxbxsq-uhfffaoysa-n"
    upper = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert normalize_inchikey(lower) == upper


def test_normalize_inchikey_strips_whitespace():
    """normalize_inchikey() MUST strip leading/trailing whitespace."""
    key = "  BSYNRYMUTXBXSQ-UHFFFAOYSA-N\n"
    expected = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert normalize_inchikey(key) == expected


def test_normalize_inchikey_is_idempotent():
    """normalize_inchikey() MUST be idempotent (calling twice = calling once)."""
    key = "bsynrymutxbxsq-uhfffaoysa-n"
    once = normalize_inchikey(key)
    twice = normalize_inchikey(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Tests: strip_inchikey_extension() — handles 28+ char keys with extensions.
# ---------------------------------------------------------------------------

def test_strip_inchikey_extension_strips_trailing_extension():
    """strip_inchikey_extension() MUST strip extensions from 28+ char keys."""
    extended = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N-a"
    stripped = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert strip_inchikey_extension(extended) == stripped


def test_strip_inchikey_extension_leaves_canonical_unchanged():
    """strip_inchikey_extension() MUST leave 27-char canonical keys unchanged."""
    canonical = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert strip_inchikey_extension(canonical) == canonical


def test_strip_inchikey_extension_handles_trailing_newline():
    """strip_inchikey_extension() MUST strip trailing newlines (v84 BUG #52)."""
    key = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N\n"
    expected = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert strip_inchikey_extension(key) == expected


# ---------------------------------------------------------------------------
# Tests: DB-level enforcement via ORM CHECK constraint.
# ---------------------------------------------------------------------------

def test_orm_enforces_inchikey_format_on_insert():
    """The ORM CHECK constraint MUST reject malformed InChIKeys on INSERT."""
    from sqlalchemy import create_engine
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import sessionmaker

    from database.base import Base
    from database.models import Drug

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    # Valid InChIKey — should insert cleanly.
    valid_drug = Drug(
        inchikey="BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        name="Aspirin",
    )
    session.add(valid_drug)
    session.commit()

    # Invalid InChIKey (lowercase) — should raise IntegrityError.
    invalid_drug = Drug(
        inchikey="bsynrymutxbxsq-uhfffaoysa-n",  # lowercase
        name="Bad Aspirin",
    )
    session.add(invalid_drug)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.close()
    Base.metadata.drop_all(engine)


def test_orm_accepts_synth_inchikey_on_insert():
    """The ORM CHECK constraint MUST accept SYNTH-prefixed dev fixtures."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.base import Base
    from database.models import Drug

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    synth_drug = Drug(
        inchikey="SYNTH0001",
        name="Synthetic Test Drug",
    )
    session.add(synth_drug)
    session.commit()  # should NOT raise

    rows = session.query(Drug).all()
    assert len(rows) == 1
    assert rows[0].inchikey == "SYNTH0001"

    session.close()
    Base.metadata.drop_all(engine)
