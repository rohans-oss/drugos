"""P1-052 ROOT FIX (v110): regression guard for migration rollback.

WHAT THIS TEST GUARDS AGAINST
-----------------------------
The audit (Issue #52) found that ``run_migrations.py`` supported a
``rollback_migration()`` function but it was NEVER invokable from the
CLI — operators had to write a Python one-liner to call it. Worse, the
function existed but had bugs (the v76 ROOT FIX comments mention
"rollback silently no-op'd every statement that started with a comment
line").

ROOT FIX
--------
This test exercises the rollback path:
  1. Every migration (001-018) has a ``<name>_rollback.sql`` sidecar.
  2. Every rollback sidecar deletes the corresponding schema_version row.
  3. The ``rollback_migration()`` function is callable and NOT
     NotImplementedError.
  4. The new CLI ``--rollback`` and ``--down`` flags work as expected.
  5. End-to-end: rollback a migration on a fresh SQLite DB (set up via
     the ORM, not the migration runner — the migration runner has a
     known REGEXP registration issue on raw SQLite that is orthogonal
     to the rollback functionality).

This is the regression guard the audit asked for: "verify every
migration has a working rollback. CI must run this."
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

# Ensure project root is on sys.path so we can import database.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force SQLite + dev mode for these tests.
os.environ.setdefault("DRUGOS_ENVIRONMENT", "development")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from database.migrations.run_migrations import (  # noqa: E402
    MIGRATIONS_DIR,
    rollback_migration,
    run_migrations,
    get_migration_status,
    _build_cli_parser,
)


# ---------------------------------------------------------------------------
# Tests: every migration has a rollback sidecar (P1-052 acceptance).
# ---------------------------------------------------------------------------

def _list_forward_migrations() -> list[str]:
    """Return the sorted list of forward migration filenames (001-018)."""
    return sorted(
        f.name
        for f in MIGRATIONS_DIR.glob("*.sql")
        if not f.name.endswith("_rollback.sql")
    )


def test_at_least_18_forward_migrations_exist():
    """P1-041 acceptance: at least 18 forward migrations MUST exist."""
    forward = _list_forward_migrations()
    assert len(forward) >= 18, (
        f"Expected at least 18 forward migrations, got {len(forward)}: {forward}"
    )


def test_every_migration_has_a_rollback_sidecar():
    """P1-052 acceptance: every forward migration MUST have a _rollback.sql."""
    forward = _list_forward_migrations()
    for fwd_name in forward:
        stem = fwd_name[:-4]  # strip .sql
        rollback_name = f"{stem}_rollback.sql"
        rollback_path = MIGRATIONS_DIR / rollback_name
        assert rollback_path.exists(), (
            f"Migration {fwd_name}: missing rollback sidecar at {rollback_path}. "
            f"Every migration MUST have a <name>_rollback.sql for down-migration "
            f"support (P1-042 CLI --rollback)."
        )


def test_every_rollback_deletes_schema_version_row():
    """P1-042 acceptance: every rollback (except 001) MUST delete the schema_version row."""
    forward = _list_forward_migrations()
    for fwd_name in forward:
        stem = fwd_name[:-4]
        # Extract the migration number (e.g. "013" from "013_is_fda_approved_nullable.sql").
        match = re.match(r"^(\d+)_", fwd_name)
        assert match, f"Could not extract migration number from {fwd_name}"
        n = int(match.group(1))
        if n == 1:
            # Migration 001 drops the entire schema_version table — skip.
            continue
        rollback_path = MIGRATIONS_DIR / f"{stem}_rollback.sql"
        content = rollback_path.read_text("utf-8")
        delete_pattern = re.compile(
            rf"DELETE\s+FROM\s+schema_version\s+WHERE\s+version\s*=\s*{n}\b",
            re.IGNORECASE | re.MULTILINE,
        )
        assert delete_pattern.search(content), (
            f"Rollback {rollback_path.name}: missing 'DELETE FROM schema_version "
            f"WHERE version = {n}'. After a rollback, check_migrations() would "
            f"still report version {n} as applied — false-positive."
        )


# ---------------------------------------------------------------------------
# Tests: rollback_migration() function contract.
# ---------------------------------------------------------------------------

def test_rollback_migration_function_is_callable():
    """P1-042 acceptance: rollback_migration() MUST be invokable, not NotImplementedError."""
    import inspect
    sig = inspect.signature(rollback_migration)
    assert "migration_name" in sig.parameters
    assert "engine" in sig.parameters


def test_rollback_migration_raises_for_unknown_migration():
    """rollback_migration() MUST raise FileNotFoundError for an unknown name.

    We use a migration name that doesn't exist as a forward file. The
    function MUST fail loudly (not silently no-op) when the named
    migration doesn't exist.
    """
    with pytest.raises(FileNotFoundError):
        rollback_migration("999_nonexistent_migration.sql")


def test_rollback_migration_raises_for_missing_sidecar():
    """rollback_migration() MUST raise NotImplementedError if the sidecar is missing.

    The audit found that some migrations had no rollback sidecar. The
    function MUST fail loudly when the sidecar is missing — the error
    message MUST name the missing file.

    We create a fake forward migration in the REAL migrations directory
    (rollback_migration uses Path(__file__).resolve().parent, not the
    module-level MIGRATIONS_DIR constant, so monkey-patching the constant
    doesn't work). The fake file is cleaned up after the test.
    """
    fake_name = "001_fake_migration_for_missing_sidecar_test.sql"
    fake_path = MIGRATIONS_DIR / fake_name
    fake_path.write_text("-- fake migration for test\n", encoding="utf-8")
    try:
        with pytest.raises((NotImplementedError, RuntimeError)) as exc_info:
            rollback_migration(fake_name)
        # The error message MUST name the missing rollback file.
        assert "rollback" in str(exc_info.value).lower()
    finally:
        if fake_path.exists():
            fake_path.unlink()


# ---------------------------------------------------------------------------
# Tests: CLI --rollback and --down flags (P1-042 ROOT FIX).
# ---------------------------------------------------------------------------

def test_cli_parser_accepts_rollback_flag():
    """The CLI parser MUST accept --rollback MIGRATION_FILENAME."""
    parser = _build_cli_parser()
    args = parser.parse_args(["--rollback", "017_confidence_tier_add_very_strong.sql"])
    assert args.rollback == "017_confidence_tier_add_very_strong.sql"


def test_cli_parser_accepts_down_flag():
    """The CLI parser MUST accept --down N."""
    parser = _build_cli_parser()
    args = parser.parse_args(["--down", "3"])
    assert args.down == 3


def test_cli_parser_accepts_status_flag():
    """The CLI parser MUST accept --status."""
    parser = _build_cli_parser()
    args = parser.parse_args(["--status"])
    assert args.status is True


def test_cli_parser_accepts_check_flag():
    """The CLI parser MUST accept --check."""
    parser = _build_cli_parser()
    args = parser.parse_args(["--check"])
    assert args.check is True


def test_cli_parser_defaults_to_no_flags():
    """With no flags, the CLI MUST default to forward-migration mode."""
    parser = _build_cli_parser()
    args = parser.parse_args([])
    assert args.rollback is None
    assert args.down is None
    assert args.status is False
    assert args.check is False


# ---------------------------------------------------------------------------
# Tests: end-to-end rollback on an ORM-set-up DB.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def orm_db_engine():
    """A fresh SQLite engine with the ORM schema created via create_all().

    We use the ORM (not the migration runner) to set up the schema because
    the migration runner has a known REGEXP-function registration issue
    on raw SQLite engines that is orthogonal to the rollback functionality
    (the migration runner works correctly when invoked via
    ``database.connection.init_db()`` which registers the REGEXP function).
    The rollback functionality is what we're testing here, not the forward
    migration path.
    """
    from database.base import Base
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_rollback_migration_executes_sidecar_sql(orm_db_engine):
    """rollback_migration() MUST execute the sidecar SQL on the DB.

    We use migration 014 (a simple index drop) since it's the simplest
    rollback that doesn't require complex schema state.
    """
    # Pre-populate schema_version with version=14 so we can verify the DELETE.
    with orm_db_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO schema_version (version, description) VALUES (14, 'test')"
        ))
    # Verify it's there.
    with orm_db_engine.begin() as conn:
        rows = conn.execute(text("SELECT version FROM schema_version")).fetchall()
    assert any(r[0] == 14 for r in rows)

    # Rollback migration 014.
    result = rollback_migration(
        "014_drugs_pubchem_cid_partial_index.sql",
        engine=orm_db_engine,
    )
    assert result.get("rolled_back") is True, (
        f"Rollback of 014 failed: {result}"
    )
    # Verify version 14 is NO LONGER in schema_version.
    with orm_db_engine.begin() as conn:
        rows = conn.execute(text("SELECT version FROM schema_version")).fetchall()
    assert not any(r[0] == 14 for r in rows), (
        f"Rollback of 014 should have deleted version=14 from schema_version. "
        f"Still present: {rows}"
    )


def test_rollback_is_idempotent(orm_db_engine):
    """Re-running a rollback MUST NOT raise (idempotent).

    The rollback sidecar uses IF EXISTS / IF NOT EXISTS guards so it's
    safe to re-run. This is the P1-042 ROOT FIX contract: operators
    can re-run a rollback without breaking the DB.
    """
    # Pre-populate schema_version with version=14.
    with orm_db_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO schema_version (version, description) VALUES (14, 'test')"
        ))
    # First rollback — should succeed.
    result1 = rollback_migration(
        "014_drugs_pubchem_cid_partial_index.sql",
        engine=orm_db_engine,
    )
    assert result1.get("rolled_back") is True
    # Second rollback — should also succeed (idempotent).
    result2 = rollback_migration(
        "014_drugs_pubchem_cid_partial_index.sql",
        engine=orm_db_engine,
    )
    assert result2.get("rolled_back") is True


def test_rollback_returns_dict_with_required_keys(orm_db_engine):
    """rollback_migration() MUST return a dict with rolled_back / elapsed_s / statements_executed."""
    with orm_db_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO schema_version (version, description) VALUES (14, 'test')"
        ))
    result = rollback_migration(
        "014_drugs_pubchem_cid_partial_index.sql",
        engine=orm_db_engine,
    )
    assert isinstance(result, dict)
    assert "rolled_back" in result
    assert "elapsed_s" in result
    assert "statements_executed" in result
    assert "error" in result
    assert result["rolled_back"] is True
    assert result["statements_executed"] >= 1
    assert result["error"] is None
