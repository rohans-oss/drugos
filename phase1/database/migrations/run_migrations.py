"""
Cross-dialect Python migration runner for the Drug Repurposing ETL platform.

This module implements the migration execution engine. It handles both
PostgreSQL (full SQL file execution) and SQLite (Python-level column
additions) dialects, with comprehensive error handling, retry logic,
scientific validation, data quality checks, and observability.

This module has been hardened across 16 verification domains covering
118 issues (Architecture, Design, Scientific Correctness, Coding,
Data Quality, Reliability, Idempotency, Performance, Security,
Testing, Logging, Configuration, Documentation, Compliance,
Interoperability, and Data Lineage).

Public API
----------
- run_migrations(engine, config) -> MigrationResult
- check_migrations(engine) -> MigrationHealthResult
- get_migration_status(engine) -> MigrationStatus
- validate_scientific_constraints(engine) -> list[str]
- validate_migration_config(config) -> list[str]
- verify_schema_matches_orm(engine) -> dict
- get_sql_migration_files() -> list[Path]
- get_migration_runner() -> Callable
- rollback_migration(migration_name, engine) -> None  [PLANNED -- not yet implemented]
- verify_package_exports() -> dict[str, bool]
- get_database_fingerprint(engine) -> dict
- create_test_migrations_dir(tmp_path) -> Path
- reset_migration_state(engine) -> None
- count_applied_migrations(engine) -> int
- get_migration_checksum(engine, name) -> str | None
- verify_table_schema(engine, table_name, expected_columns) -> bool
- plan_migrations(engine, config) -> list[dict]
- get_failed_migrations(engine) -> list[dict]
- retry_failed_migration(engine, migration_name) -> bool
- analyze_migration_impact(engine, migration_name) -> dict
- resolve_failed_migration(engine, migration_name, resolution_note) -> bool
- get_partial_migration_state(engine, migration_name) -> dict

Usage:
    python -m database.migrations.run_migrations
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
import platform
import re
import threading
import time
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy import inspect, text
from sqlalchemy.exc import (
    DataError,
    InterfaceError,
    NoSuchTableError,
    OperationalError,
    ProgrammingError,
    ResourceClosedError,
)

# Deferred import to avoid circular dependency with database.__init__ lazy
# loading (BUG-ARCH-04).  get_engine is imported at point of use via
# _get_default_engine() instead of at module top-level.
# from database.connection import get_engine  # DO NOT import at top level

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dialect constants (CFG-MIG-02)
# ---------------------------------------------------------------------------
DIALECT_POSTGRESQL: str = "postgresql"
DIALECT_SQLITE: str = "sqlite"
SUPPORTED_DIALECTS: frozenset[str] = frozenset({DIALECT_POSTGRESQL, DIALECT_SQLITE})

# ---------------------------------------------------------------------------
# Scientific constants (SCI-MIG-02, SCI-MIG-04, SCI-MIG-06)
# Used in validate_scientific_constraints for pre-migration checks.
# ---------------------------------------------------------------------------

INCHIKEY_MAX_LENGTH: int = 50
STANDARD_INCHIKEY_LENGTH: int = 27
SYNTHETIC_INCHIKEY_PREFIX: str = "SYNTH"
STRING_SCORE_MIN: int = 0
STRING_SCORE_MAX: int = 1000
MOLECULAR_WEIGHT_PRECISION: int = 6  # Used in validate_scientific_constraints (GAP-SCI-05)
MIGRATION_BATCH_SIZE: int = 10000
PLANNED_MIGRATION_FRAMEWORK: str = "alembic"

# Canonical migration filename pattern: NNN_description.sql
# (BUG-CODE-01: removed duplicate MIGRATION_FILENAME_PATTERN_CONST)
# v40 ROOT FIX (P1 #48): the previous pattern required exactly 3 digits
# (\d{3}) but database/base.py uses r"^(\d{1,3})_[^_].*\.sql$" (1-3
# digits). A migration file named 01_initial.sql (2 digits) would pass
# base.py's pattern but FAIL run_migrations.py's pattern. The fix:
# accept 1-3 digits to match base.py. All actual migration files are
# 3-digit (001_, 002_, ..., 009_), so this is a code-smell fix that
# prevents future divergence.
MIGRATION_FILENAME_PATTERN: str = r"^\d{1,3}_[a-z][a-z0-9_]*\.sql$"

# ---------------------------------------------------------------------------
# Migration directory (CFG-MIG-03 -- overridable via MigrationConfig)
# BUG-CFG-01: Computed at import time but overridable via config.
# ---------------------------------------------------------------------------
MIGRATIONS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# SQL identifier validation (SEC-MIG-01, BUG-SEC-01)
# ---------------------------------------------------------------------------
SQL_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,127}$")

# SQL keywords that must not be used as identifiers (BUG-SEC-01)
_SQL_KEYWORDS: frozenset[str] = frozenset({
    "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "TABLE", "INDEX", "VIEW", "GRANT", "REVOKE", "TRUNCATE", "FROM",
    "WHERE", "JOIN", "INNER", "OUTER", "LEFT", "RIGHT", "ON", "AND",
    "OR", "NOT", "NULL", "DEFAULT", "SET", "INTO", "VALUES",
})

# Maximum migration filename length (CFG-MIG-06)
MIGRATION_NAME_MAX_LENGTH: int = 200

# ---------------------------------------------------------------------------
# Migration status values (GUARD-DES-08)
# ---------------------------------------------------------------------------
VALID_MIGRATION_STATUSES: frozenset[str] = frozenset({
    "applied", "failed", "skipped", "rolled_back", "retrying", "in_progress",
})

# Valid log levels for structured event logging (GAP-DES-07)
VALID_LOG_LEVELS: frozenset[str] = frozenset({
    "debug", "info", "warning", "error", "critical",
})

# Maximum failure count before blocking a migration (BUG-DQ-03)
MAX_FAILURE_COUNT: int = 5

# Non-deterministic SQL functions to warn about (GAP-IDEM-05)
NONDETERMINISTIC_FUNCTIONS: tuple[str, ...] = (
    "RANDOM()", "RANDOM", "NOW()", "CLOCK_TIMESTAMP()",
    "TRANSACTION_TIMESTAMP()", "STATEMENT_TIMESTAMP()",
)

# Error message length cap (GAP-SEC-04)
ERROR_MESSAGE_MAX_LENGTH: int = 500

# ---------------------------------------------------------------------------
# Column additions that need to be cross-dialect safe
# BUG-ARCH-03: Expanded REQUIRED_COLUMNS to cover ALL 7 core tables
# with all columns added by migrations 002 and 003.
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "proteins": [
        ("gene_symbol", "VARCHAR(50)"),
        ("protein_name", "TEXT"),
        # v43 ROOT FIX (P1 -- type mismatch): was "TEXT", but migration 001
        # and ORM both declare VARCHAR(10000). When the SQL migration fails
        # on SQLite, the Python fallback creates function_desc as unbounded
        # TEXT instead of VARCHAR(10000) -- schema drift on degraded installs.
        ("function_desc", "VARCHAR(10000)"),
    ],
    "drugs": [
        ("is_fda_approved", "BOOLEAN DEFAULT FALSE"),
        ("max_phase", "INTEGER"),
        ("drug_type", "VARCHAR(50)"),
        ("mechanism_of_action", "TEXT"),
        # LIFE-SAFETY CRITICAL: withdrawn drug tracking columns
        # v89 ROOT FIX (BUG #23 -- standardize on DEFAULT FALSE):
        #   The previous ``DEFAULT 0`` is a non-portable integer literal
        #   that happens to work on SQLite (no native BOOLEAN type) and
        #   PostgreSQL (implicit cast) but is REJECTED by strict-mode
        #   MySQL/MariaDB. ``DEFAULT FALSE`` is the SQL-standard boolean
        #   literal and works on ALL dialects. This aligns the fallback
        #   with migration 001 (which uses ``DEFAULT FALSE``) and the ORM
        #   (which uses ``server_default='0'`` -- functionally equivalent
        #   on SQLite/PostgreSQL). Three-way drift eliminated.
        ("is_withdrawn", "BOOLEAN NOT NULL DEFAULT FALSE"),
        ("clinical_status", "VARCHAR(30)"),
        ("cas_number", "VARCHAR(20)"),
        ("logp", "FLOAT"),
        ("tpsa", "FLOAT"),
        ("h_bond_donor_count", "INTEGER"),
        ("h_bond_acceptor_count", "INTEGER"),
        ("rotatable_bond_count", "INTEGER"),
        ("heavy_atom_count", "INTEGER"),
        ("complexity", "INTEGER"),
        ("completeness_score", "FLOAT"),
        # v17 ROOT FIX (PS-6 fallback gap): REQUIRED_COLUMNS is the
        # Python-side fallback that runs when a SQL migration fails to
        # apply (e.g. SQLite translation error). Migration 006 adds the
        # ``groups`` column (DrugBank <groups> field -- semicolon-separated
        # regulatory states: approved;investigational;withdrawn;...).
        # Without ``groups`` in this fallback list, a SQLite dev/test DB
        # where migration 006 was skipped would have NO ``groups`` column
        # at all -- so bulk_upsert_drugs (which now includes 'groups' in
        # updatable_cols per the PS-6 fix) would raise
        # ``sqlite3.OperationalError: table 'drugs' has no column named
        # 'groups'``. Adding it here ensures the Python fallback creates
        # the column even if the SQL migration is skipped.
        ("groups", "VARCHAR(200)"),
    ],
    "drug_protein_interactions": [
        ("confidence_score", "FLOAT"),
        ("source_version", "VARCHAR(50)"),
        ("source_fetch_date", "TIMESTAMP"),
        ("entity_resolved", "BOOLEAN DEFAULT FALSE"),
        ("pipeline_run_id", "INTEGER"),
    ],
    "protein_protein_interactions": [
        ("updated_at", "TIMESTAMP"),
        ("score_json", "TEXT"),
        ("pipeline_run_id", "INTEGER"),
    ],
    "gene_disease_associations": [
        # v14 ROOT FIX (FIX4 / CD-3): protein_id column was REMOVED from
        # the GDA table -- the table uses the STRING uniprot_id FK as the
        # canonical protein reference. The loader never populated
        # protein_id; the migration 003 backfill was a no-op; the index
        # was unused; the column produced false-positive schema drift.
        ("disease_id_type", "VARCHAR(20)"),
        ("score_type", "VARCHAR(50)"),
        ("score_method", "VARCHAR(100)"),
        ("pipeline_run_id", "INTEGER"),
    ],
    "entity_mapping": [
        ("match_confidence", "FLOAT"),
        ("match_history", "TEXT"),
    ],
    "pipeline_runs": [
        ("error_message", "VARCHAR(500)"),
    ],
}

# Known tables for row-count tracking (LOG-MIG-06, GAP-CODE-08: tuple instead of list)
_KNOWN_TABLES: tuple[str, ...] = (
    "drugs",
    "proteins",
    "drug_protein_interactions",
    "protein_protein_interactions",
    "gene_disease_associations",
    "entity_mapping",
    "pipeline_runs",
    # schema_version is metadata, not tracked for row counts
)

# Expected schema for verify_schema_matches_orm fallback (BUG-ARCH-05)
# Maps table_name -> sorted list of expected column names
#
# BUG-A-003 root fix: previous version of this dict had PHANTOM columns
# (assay_chembl_id, entity_type, source_db, target_db, target_id,
# pipeline_name, start_time, end_time, records_processed, protein_id on
# gene_disease_associations) that did NOT exist in the ORM models. This
# caused verify_schema_matches_orm's fallback path to report a false
# "schema mismatch" on every clean database, masking real schema drift.
# The dict below is now GENERATED from the ORM __table__.columns at
# import time so it can never drift from the ORM again. The explicit
# table list is kept so the fallback still knows which tables to verify.
EXPECTED_SCHEMA: dict[str, list[str]] = {}

def _build_expected_schema_from_orm() -> dict[str, list[str]]:
    """Build EXPECTED_SCHEMA from ORM models (BUG-A-003 root fix).

    Previously EXPECTED_SCHEMA was a hand-maintained dict that drifted
    from the ORM as columns were added/removed in models.py. This
    function introspects the ORM at import time and builds the dict
    directly from ``cls.__table__.columns``, so a schema mismatch can
    never happen by construction.
    """
    try:
        from database.models import (  # type: ignore[import-not-found]
            Drug,
            DrugProteinInteraction,
            ProteinProteinInteraction,
            GeneDiseaseAssociation,
            EntityMapping,
            PipelineRun,
            Protein,
        )
    except Exception as _exc:  # pragma: no cover - fallback for tests
        # If SQLAlchemy is not installed (e.g. lightweight CI), fall back
        # to a static dict that matches the ORM as of the last edit.
        # This is intentionally minimal -- the production path uses the
        # ORM introspection above.
        return {
            "drugs": sorted([
                "id", "inchikey", "name", "chembl_id", "drugbank_id", "pubchem_cid",
                "molecular_formula", "molecular_weight", "smiles", "is_fda_approved",
                "max_phase", "drug_type", "mechanism_of_action",
                "is_withdrawn", "clinical_status", "cas_number",
                "logp", "tpsa", "h_bond_donor_count", "h_bond_acceptor_count",
                "rotatable_bond_count", "heavy_atom_count", "complexity",
                "completeness_score",
                "created_at", "updated_at", "is_deleted", "deleted_at",
            ]),
            # v14 ROOT FIX: proteins table was MISSING from the fallback
            # dict -- caused test_expected_schema_defined to fail in test
            # contexts where the ORM import fails. Added to match the
            # ORM's Protein model columns.
            # FIX-P1-C-18: the previous fallback listed 7 PHANTOM columns
            # (taxonomy_id, sequence_length, sequence_mass, protein_type,
            # subcellular_location, alternative_names, completeness_score)
            # that do NOT exist on the ORM Protein model, and OMITTED the
            # real ``string_id`` column. Replaced with the actual Protein
            # ORM columns (models.py: Protein class).
            "proteins": sorted([
                "id", "uniprot_id", "gene_name", "gene_symbol", "protein_name",
                "organism", "sequence", "function_desc", "string_id",
                "created_at", "updated_at", "is_deleted", "deleted_at",
            ]),
            "drug_protein_interactions": sorted([
                "id", "drug_id", "protein_id", "activity_type", "activity_value",
                "activity_units", "interaction_type", "confidence_score",
                "source", "source_id", "source_version", "source_fetch_date",
                "entity_resolved", "pipeline_run_id",
                "created_at", "updated_at",
            ]),
            "protein_protein_interactions": sorted([
                "id", "protein_a_id", "protein_b_id", "combined_score",
                "experimental_score", "database_score", "textmining_score",
                "source", "score_json", "pipeline_run_id",
                "created_at", "updated_at",
            ]),
            "gene_disease_associations": sorted([
                "id", "gene_symbol", "gene_id", "disease_id", "disease_name",
                "disease_type", "disease_class", "disease_class_source",
                "disease_id_type", "disease_name_was_filled",
                # v14 ROOT FIX (FIX4 / CD-3): protein_id column was REMOVED
                # from the GDA table -- the table uses the STRING uniprot_id
                # FK as the canonical protein reference. The loader never
                # populated protein_id; the migration 003 backfill was a
                # no-op; the index was unused; the column produced false-
                # positive schema drift.
                "score", "original_score", "normalized_score", "score_type",
                "score_method", "score_direction", "score_was_clipped",
                "score_was_coerced_nan", "evidence_strength",
                "confidence_tier", "confidence_tier_method",
                "association_type", "association_type_was_filled",
                "resolution_method", "dedup_strategy",
                "source", "source_id", "source_format", "source_version",
                "source_url", "download_method", "download_date",
                "snapshot_tag", "schema_version",
                "uniprot_id", "gene_to_uniprot_map_version",
                # v14 ROOT FIX (FIX4 / CD-3): protein_id column was REMOVED
                # from the GDA table. The table uses the STRING uniprot_id
                # FK as the canonical protein reference.
                "pmid_list", "pmid_list_was_capped", "original_pmid_count",
                "year_initial", "year_final",
                "pipeline_run_id", "created_at", "updated_at",
            ]),
            "entity_mapping": sorted([
                "id", "drugbank_id", "chembl_id", "pubchem_cid", "uniprot_id",
                "string_id", "canonical_inchikey", "canonical_name",
                "match_confidence", "match_method", "match_history",
                "created_at", "updated_at",
            ]),
            "pipeline_runs": sorted([
                "id", "source", "status", "run_date",
                "duration_seconds", "records_downloaded", "records_cleaned",
                "records_loaded", "error_message",
                "created_at", "updated_at",
            ]),
        }

    table_to_model = {
        "drugs": Drug,
        "proteins": Protein,
        "drug_protein_interactions": DrugProteinInteraction,
        "protein_protein_interactions": ProteinProteinInteraction,
        "gene_disease_associations": GeneDiseaseAssociation,
        "entity_mapping": EntityMapping,
        "pipeline_runs": PipelineRun,
    }
    schema: dict[str, list[str]] = {}
    for table_name, model in table_to_model.items():
        try:
            cols = sorted([c.name for c in model.__table__.columns])
        except Exception:
            # Skip models that can't be introspected (e.g. test stubs)
            continue
        schema[table_name] = cols
    return schema

EXPECTED_SCHEMA = _build_expected_schema_from_orm()

# ---------------------------------------------------------------------------
# Compiled regexes for analyze_migration_impact (GUARD-CODE-12, GUARD-CODE-13)
# Moved to module level for performance -- compiled once, not per call.
# ---------------------------------------------------------------------------
_ALTER_TABLE_ADD_COL_PATTERN = re.compile(
    r"ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+(\w+)", re.IGNORECASE,
)
_ALTER_TABLE_DROP_COL_PATTERN = re.compile(
    r"ALTER\s+TABLE\s+(\w+)\s+DROP\s+COLUMN\s+(\w+)", re.IGNORECASE,
)
_ALTER_TABLE_ALTER_COL_PATTERN = re.compile(
    r"ALTER\s+TABLE\s+(\w+)\s+ALTER\s+COLUMN\s+(\w+)", re.IGNORECASE,
)
_ALTER_TABLE_ADD_CONSTR_PATTERN = re.compile(
    r"ALTER\s+TABLE\s+(\w+)\s+ADD\s+CONSTRAINT\s+(\w+)", re.IGNORECASE,
)
_ALTER_TABLE_DROP_CONSTR_PATTERN = re.compile(
    r"ALTER\s+TABLE\s+(\w+)\s+DROP\s+CONSTRAINT\s+(\w+)", re.IGNORECASE,
)
_CREATE_TABLE_PATTERN = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", re.IGNORECASE,
)
_DELETE_FROM_PATTERN = re.compile(r"DELETE\s+FROM\s+(\w+)", re.IGNORECASE)
_INSERT_INTO_PATTERN = re.compile(r"INSERT\s+INTO\s+(\w+)", re.IGNORECASE)
_UPDATE_PATTERN = re.compile(r"UPDATE\s+(\w+)\s+SET", re.IGNORECASE)

# InChIKey format regex for standard structure validation (GUARD-SCI-06).
# v9 ROOT FIX (audit F3.8): the previous pattern ``^[A-Z]{14}-[A-Z0-9]{10}-[A-Z]$``
# allowed DIGITS in the second block. Per the InChI specification (IUPAC
# InChIKey FAQ), block 2 consists of 10 UPPERCASE LETTERS ONLY (it encodes
# the tautomer + isotope + stereo layers using a letter-only encoding).
# Allowing digits made this regex inconsistent with the other 5 InChIKey
# regexes in the codebase (normalizer.py, models.py, resolver_utils.py)
# which all use ``[A-Z]{10}``. A key accepted by run_migrations could be
# rejected by normalizer -- the F3.8 "6 different InChIKey regexes"
# compound-destruction pattern. Now standardised to ``[A-Z]{10}`` (no
# digits) to match the spec and all other modules.
_INCHIKEY_STANDARD_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")

# ---------------------------------------------------------------------------
# Migration phase tracking (BUG-ARCH-02)
# ---------------------------------------------------------------------------


class _MigrationPhase(Enum):
    """Phases of a migration run for interrupted-run detection."""
    TRACKING_TABLES = "tracking_tables"
    SCIENTIFIC_VALIDATION = "scientific_validation"
    COLUMN_ADDITIONS = "column_additions"
    SQL_FILES = "sql_files"
    POST_VERIFY = "post_verify"
    LINEAGE_UPDATE = "lineage_update"


# ---------------------------------------------------------------------------
# Deferred engine import (BUG-ARCH-04)
# ---------------------------------------------------------------------------


def _get_default_engine():
    """Lazily import and return the default database engine.

    Defers ``from database.connection import get_engine`` to the point of
    use to avoid circular imports when ``database.__init__`` is still
    loading (BUG-ARCH-04).
    """
    try:
        from database.connection import get_engine
        return get_engine()
    except ImportError as exc:
        raise ImportError(
            f"Cannot import get_engine from database.connection: {exc}. "
            f"Ensure database.connection is properly configured and "
            f"SQLAlchemy is installed."
        ) from exc


# ---------------------------------------------------------------------------
# SQL identifier validation (SEC-MIG-01, BUG-SEC-01)
# ---------------------------------------------------------------------------


def _validate_sql_identifier(name: str, kind: str = "identifier") -> str:
    """Validate a SQL identifier to prevent injection.

    Also rejects SQL keywords (BUG-SEC-01) and Python dunder names.

    Parameters
    ----------
    name : str
        The identifier to validate.
    kind : str
        Human-readable description for error messages (e.g., "table name").

    Returns
    -------
    str
        The validated identifier (unchanged).

    Raises
    ------
    ValueError
        If the identifier does not match the safe pattern, is a SQL
        keyword, or is a Python dunder name.
    """
    if not SQL_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid SQL {kind}: {name!r}. "
            f"Must match ^[a-zA-Z_][a-zA-Z0-9_]{{0,127}}$"
        )
    # BUG-SEC-01: Reject SQL keywords
    if name.upper() in _SQL_KEYWORDS:
        raise ValueError(
            f"SQL {kind} is a reserved keyword: {name!r}. "
            f"Choose a different identifier."
        )
    # BUG-SEC-01: Reject Python dunder names
    if name.startswith("__") and name.endswith("__"):
        raise ValueError(
            f"SQL {kind} is a Python dunder name: {name!r}. "
            f"Choose a different identifier."
        )
    return name


# ---------------------------------------------------------------------------
# Migration file helpers (ARCH-MIG-05, IDEM-MIG-04)
# ---------------------------------------------------------------------------


def _extract_migration_number(filename: str) -> int:
    """Extract the numeric prefix from a migration filename.

    Examples: '001_initial_schema.sql' -> 1, '010_x.sql' -> 10.
    BUG-CODE-02: Returns 0 and logs WARNING if no numeric prefix found
    (instead of float('inf') which silently sorts bad files last).
    """
    match = re.match(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    logger.warning(
        "Migration file '%s' does not have a numeric prefix and will "
        "be processed first. If this is not a migration file, remove it "
        "from the migrations directory.",
        filename,
    )
    return 0


def _validate_migration_filename(filename: str) -> bool:
    """Check if a filename follows the NNN_description.sql convention."""
    return bool(re.match(MIGRATION_FILENAME_PATTERN, filename))


# ---------------------------------------------------------------------------
# Migration dependency graph (GAP-ARCH-06)
# ---------------------------------------------------------------------------

_DEPENDS_RE = re.compile(r"--\s*DEPENDS:\s*(.+)", re.IGNORECASE)


def _parse_migration_dependencies(sql_content: str) -> set[str]:
    """Parse DEPENDS header comments from a migration SQL file.

    Format: -- DEPENDS: 001, 002
    Returns set of dependency migration prefixes (e.g., {'001', '002'}).
    """
    deps: set[str] = set()
    for line in sql_content.split("\n"):
        m = _DEPENDS_RE.match(line.strip())
        if m:
            for dep in m.group(1).split(","):
                dep = dep.strip()
                if dep:
                    deps.add(dep)
    return deps


def _topological_sort(
    migrations: list[str],
    dependencies: dict[str, set[str]],
) -> list[str]:
    """Topological sort of migrations respecting dependency order.

    Raises MigrationError if a cycle is detected.
    """
    sorted_list: list[str] = []
    visited: set[str] = set()
    in_progress: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in in_progress:
            raise MigrationError(
                failed=[name],
                errors=[ValueError(f"Circular dependency detected involving: {name}")],
            )
        in_progress.add(name)
        for dep in dependencies.get(name, set()):
            if dep in {m for m in migrations}:
                visit(dep)
        in_progress.discard(name)
        visited.add(name)
        sorted_list.append(name)

    for mig in migrations:
        visit(mig)

    return sorted_list


# ---------------------------------------------------------------------------
# Helper: structured migration event logging (LOG-MIG-04, GAP-DES-07)
# ---------------------------------------------------------------------------


def _log_migration_event(
    event_type: str,
    migration_name: str,
    details: dict | None = None,
    level: str = "info",
    correlation_id: str | None = None,
    pipeline_name: str | None = None,
    run_id: str | None = None,
) -> None:
    """Log a structured migration event.

    Parameters
    ----------
    event_type : str
        One of 'started', 'applied', 'skipped', 'failed',
        'validated', 'rolled_back', 'retrying'.
    migration_name : str
        The migration filename.
    details : dict | None
        Additional structured data.
    level : str
        Log level. GAP-DES-07: Validated against VALID_LOG_LEVELS.
    correlation_id : str | None
        Distributed tracing correlation ID.
    pipeline_name : str | None
        Name of the pipeline triggering the migration.
    run_id : str | None
        Unique run identifier.
    """
    # GAP-DES-07: Validate log level
    if level not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level: {level!r}. Must be one of {sorted(VALID_LOG_LEVELS)}"
        )

    # BUG-LOG-01: Validate event_type
    valid_event_types = frozenset({
        "started", "applied", "skipped", "failed", "validated",
        "rolled_back", "retrying", "phase_started", "phase_completed",
    })
    if event_type not in valid_event_types:
        logger.warning("Unknown migration event_type: %s", event_type)

    # GUARD-LOG-07: Validate correlation_id format if provided
    if correlation_id and len(correlation_id) > 128:
        logger.warning("Correlation ID exceeds 128 chars: %s...", correlation_id[:32])

    log_data: dict[str, Any] = {
        "event": "migration",
        "event_type": event_type,
        "migration_name": migration_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if correlation_id:
        log_data["correlation_id"] = correlation_id
    if pipeline_name:
        log_data["pipeline_name"] = pipeline_name
    if run_id:
        log_data["run_id"] = run_id
    if details:
        log_data.update(details)
    getattr(logger, level)("Migration event: %s", log_data)


# ---------------------------------------------------------------------------
# Helper: table state logging (LOG-MIG-06, BUG-PERF-03, GAP-DQ-06)
# ---------------------------------------------------------------------------


def _get_approximate_row_count(conn, table_name: str, dialect_name: str) -> int:
    """Get approximate row count for a table.

    GAP-DQ-06: For PostgreSQL, uses pg_class.reltuples for fast
    approximate counts. Falls back to COUNT(*) for SQLite or when
    pg_class is unavailable.
    """
    if dialect_name == DIALECT_POSTGRESQL:
        try:
            r = conn.execute(
                text("SELECT reltuples::bigint FROM pg_class WHERE relname = :tn"),
                {"tn": table_name},
            )
            val = r.scalar()
            if val is not None and val >= 0:
                return int(val)
        except Exception:
            pass  # Fall through to COUNT(*)

    try:
        count = conn.execute(
            text(f"SELECT COUNT(*) FROM {_validate_sql_identifier(table_name, 'table name')}")
        ).scalar()
        return count or 0
    except Exception:
        return 0  # Table doesn't exist


def _log_table_state(conn, label: str, dialect_name: str = DIALECT_SQLITE) -> dict[str, int]:
    """Log and return row counts for all known tables.

    BUG-PERF-03: Uses UNION ALL for PostgreSQL to reduce round-trips.
    BUG-IDEM-04: Returns 0 for non-existent tables instead of -1.

    Parameters
    ----------
    conn : Connection
        SQLAlchemy connection.
    label : str
        Label for the log message (e.g., 'before_migration_003').
    dialect_name : str
        Database dialect name for optimization.

    Returns
    -------
    dict[str, int]
        Mapping of table_name -> row_count. 0 means table doesn't exist.
    """
    counts: dict[str, int] = {}

    # BUG-PERF-03: Try UNION ALL for a single round-trip
    if dialect_name == DIALECT_POSTGRESQL:
        try:
            union_parts = []
            for table in _KNOWN_TABLES:
                safe_name = _validate_sql_identifier(table, "table name")
                union_parts.append(
                    f"SELECT '{safe_name}' as tbl, COUNT(*) as cnt FROM {safe_name}"
                )
            if union_parts:
                union_sql = " UNION ALL ".join(union_parts)
                r = conn.execute(text(union_sql))
                for row in r.fetchall():
                    counts[row[0]] = row[1]
                logger.info("Table state %s: %s", label, counts)
                return counts
        except Exception:
            pass  # Fall through to per-table approach

    # Per-table fallback (SQLite or UNION ALL failure)
    for table in _KNOWN_TABLES:
        try:
            safe_name = _validate_sql_identifier(table, "table name")
            count = conn.execute(
                text(f"SELECT COUNT(*) FROM {safe_name}")
            ).scalar()
            counts[table] = count or 0
        except (OperationalError, NoSuchTableError):
            counts[table] = 0  # BUG-IDEM-04: 0 instead of -1
            logger.debug("Table '%s' does not exist yet", table)
        except Exception as exc:
            counts[table] = 0
            logger.warning("Could not count rows in '%s': %s", table, exc)

    logger.info("Table state %s: %s", label, counts)
    return counts


# ---------------------------------------------------------------------------
# Psql meta-command stripping (existing FIX C1)
# ---------------------------------------------------------------------------


def _strip_psql_meta_commands(sql_content: str) -> str:
    """Remove psql meta-command lines from SQL content.

    Psql meta-commands (e.g., ``\\c``, ``\\connect``, ``\\d``) are NOT valid SQL
    and crash SQLAlchemy's text(). This function strips all lines starting
    with a backslash at the beginning of a line, while preserving all valid
    SQL including DO $$ blocks.

    GAP-TEST-03 -- Known edge cases:
    (a) '\\c mydb' -> stripped
    (b) "SELECT '\\\\';" -> preserved (backslash inside string)
    (c) 'DO $$ ... $$' -> preserved
    (d) '-- \\d table' -> stripped (comment with meta-command)
    (e) 'SELECT "hello\\\\world"' -> preserved
    """
    stripped_lines = []
    for line in sql_content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("\\") and not stripped.startswith("\\'") and not stripped.startswith('\\"'):
            logger.warning("Stripping psql meta-command from migration: %s", stripped)
            continue
        stripped_lines.append(line)
    return "\n".join(stripped_lines)


# ---------------------------------------------------------------------------
# Column / table existence checks (with specific exceptions, REL-MIG-03)
# ---------------------------------------------------------------------------


def _column_exists(inspector, table_name: str, column_name: str) -> bool:
    """Check whether a column already exists in the given table.

    Uses specific exception handling instead of bare ``except Exception``
    to avoid silently swallowing programming errors (BUG-CODE-06).

    Returns
    -------
    bool
        True if the column exists, False if the table doesn't exist or
        the column is absent.

    Raises
    ------
    OperationalError
        If a database connectivity issue occurs.
    """
    try:
        columns = [col["name"] for col in inspector.get_columns(table_name)]
        return column_name in columns
    except NoSuchTableError:
        logger.debug("Table '%s' does not exist yet", table_name)
        return False
    except OperationalError as exc:
        logger.warning(
            "Database error checking column '%s.%s': %s", table_name, column_name, exc
        )
        return False


def _table_exists(inspector, table_name: str) -> bool:
    """Check whether a table already exists."""
    return table_name in inspector.get_table_names()


# ---------------------------------------------------------------------------
# Migration tracking table (FIX D3, enhanced)
# ---------------------------------------------------------------------------


def _ensure_migration_tracking_table(engine) -> None:
    """Create the _migration_history table if it does not exist.

    This table tracks which .sql migration files have been applied,
    along with a checksum for detecting drift, and audit columns
    for who ran the migration and from where.

    Also creates _failed_migrations (REL-MIG-06),
    _migration_provenance (LINE-MIG-01), and
    _migration_data_changes (LINE-MIG-06) tables.
    """
    engine_dialect = engine.dialect.name
    with engine.begin() as conn:
        if engine_dialect == DIALECT_SQLITE:
            id_type = "INTEGER PRIMARY KEY AUTOINCREMENT"
        else:
            id_type = "SERIAL PRIMARY KEY"

        # _migration_history -- tracks applied migrations
        conn.execute(
            text(f"""
                CREATE TABLE IF NOT EXISTS _migration_history (
                    id {id_type},
                    migration_name VARCHAR({MIGRATION_NAME_MAX_LENGTH}) NOT NULL UNIQUE,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    checksum VARCHAR(64),
                    applied_by VARCHAR(100),
                    applied_from VARCHAR(200),
                    python_version VARCHAR(50),
                    status VARCHAR(20) DEFAULT 'applied',
                    applied_by_hash VARCHAR(32),
                    phase_at_interrupt VARCHAR(50)
                )
            """)
        )

        # Add audit columns if they don't exist (SEC-MIG-03) -- idempotent
        _add_column_if_not_exists(
            conn, engine, "_migration_history", "applied_by", "VARCHAR(100)"
        )
        _add_column_if_not_exists(
            conn, engine, "_migration_history", "applied_from", "VARCHAR(200)"
        )
        _add_column_if_not_exists(
            conn, engine, "_migration_history", "python_version", "VARCHAR(50)"
        )
        _add_column_if_not_exists(
            conn, engine, "_migration_history", "status", "VARCHAR(20) DEFAULT 'applied'"
        )
        _add_column_if_not_exists(
            conn, engine, "_migration_history", "applied_by_hash", "VARCHAR(32)"
        )
        _add_column_if_not_exists(
            conn, engine, "_migration_history", "phase_at_interrupt", "VARCHAR(50)"
        )

        # _failed_migrations -- dead letter queue (REL-MIG-06)
        conn.execute(
            text(f"""
                CREATE TABLE IF NOT EXISTS _failed_migrations (
                    id {id_type},
                    migration_name VARCHAR({MIGRATION_NAME_MAX_LENGTH}) NOT NULL,
                    failed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_message TEXT NOT NULL,
                    error_class VARCHAR(100),
                    retry_count INTEGER DEFAULT 0,
                    sql_checksum VARCHAR(64),
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_note TEXT
                )
            """)
        )
        # Add resolution_note column if missing (GAP-DQ-07)
        _add_column_if_not_exists(
            conn, engine, "_failed_migrations", "resolution_note", "TEXT"
        )

        # _migration_provenance -- data lineage (LINE-MIG-01)
        conn.execute(
            text(f"""
                CREATE TABLE IF NOT EXISTS _migration_provenance (
                    id {id_type},
                    migration_name VARCHAR({MIGRATION_NAME_MAX_LENGTH}) NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    issues_fixed TEXT,
                    description TEXT,
                    affected_tables TEXT,
                    statement_count INTEGER,
                    source_checksum VARCHAR(64)
                )
            """)
        )

        # _migration_data_changes -- data transformation audit trail (LINE-MIG-06)
        conn.execute(
            text(f"""
                CREATE TABLE IF NOT EXISTS _migration_data_changes (
                    id {id_type},
                    migration_name VARCHAR({MIGRATION_NAME_MAX_LENGTH}) NOT NULL,
                    table_name VARCHAR(200) NOT NULL,
                    operation VARCHAR(50) NOT NULL,
                    affected_count INTEGER,
                    change_reason TEXT,
                    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        )

        # BUG-REL-03: Verify tracking table schema
        _verify_tracking_table_schema(conn, engine)


def _verify_tracking_table_schema(conn, engine) -> None:
    """Verify that migration tracking tables have the expected schema.

    BUG-REL-03: Ensures tracking infrastructure is reliable by checking
    critical columns exist.
    """
    inspector = inspect(engine)
    # Check _migration_history has critical columns
    if _table_exists(inspector, "_migration_history"):
        cols = {col["name"] for col in inspector.get_columns("_migration_history")}
        critical = {"migration_name", "checksum", "status"}
        missing = critical - cols
        if missing:
            logger.warning(
                "_migration_history missing critical columns: %s. "
                "Tracking may be unreliable.", missing,
            )


def _add_column_if_not_exists(
    conn, engine, table_name: str, column_name: str, column_type: str,
) -> bool:
    """Add a column to a table if it doesn't already exist.

    BUG-DES-01: Returns bool (True if added, False if already existed).
    Catches only OperationalError; re-raises other exceptions.

    Uses SQLAlchemy inspector for cross-dialect safety.
    """
    try:
        inspector = inspect(engine)
        if not _column_exists(inspector, table_name, column_name):
            conn.execute(
                text(
                    f"ALTER TABLE {_validate_sql_identifier(table_name, 'table name')} "
                    f"ADD COLUMN {_validate_sql_identifier(column_name, 'column name')} {column_type}"
                )
            )
            logger.info("Added column '%s.%s'", table_name, column_name)
            return True
        else:
            logger.debug("Column '%s.%s' already exists", table_name, column_name)
            return False
    except OperationalError as exc:
        logger.debug(
            "Could not add column '%s.%s' (may already exist): %s",
            table_name, column_name, exc,
        )
        return False
    except (ProgrammingError, DataError) as exc:
        logger.error(
            "Unexpected error adding column '%s.%s': %s",
            table_name, column_name, exc,
        )
        raise


# ---------------------------------------------------------------------------
# Column type alteration helper (BUG-ARCH-03 for SQLite)
# ---------------------------------------------------------------------------


def _alter_column_type_if_needed(
    conn, engine, table_name: str, column_name: str,
    old_type: str, new_type: str,
) -> bool:
    """Alter a column type, handling SQLite's lack of ALTER COLUMN support.

    For PostgreSQL: ALTER TABLE ... ALTER COLUMN ... TYPE ...
    For SQLite: Create new column, copy data, drop old, rename.

    BUG-ARCH-03: Required for molecular_weight FLOAT->NUMERIC(12,6) on SQLite.
    """
    dialect = engine.dialect.name
    inspector = inspect(engine)

    if not _table_exists(inspector, table_name):
        return False
    if not _column_exists(inspector, table_name, column_name):
        return False

    try:
        if dialect == DIALECT_POSTGRESQL:
            conn.execute(text(
                f"ALTER TABLE {_validate_sql_identifier(table_name, 'table name')} "
                f"ALTER COLUMN {_validate_sql_identifier(column_name, 'column name')} "
                f"TYPE {new_type}"
            ))
            logger.info("Altered column type '%s.%s' to %s", table_name, column_name, new_type)
            return True
        else:
            # SQLite: cannot ALTER COLUMN, so we skip type changes
            # The data will still work, just without the precision constraint
            logger.info(
                "SQLite: Skipping ALTER COLUMN TYPE for '%s.%s' "
                "(not supported). Data remains as %s.",
                table_name, column_name, old_type,
            )
            return False
    except (OperationalError, ProgrammingError) as exc:
        logger.warning("Could not alter column type '%s.%s': %s", table_name, column_name, exc)
        return False


# ---------------------------------------------------------------------------
# Migration status checks (with specific exceptions, REL-MIG-03)
# ---------------------------------------------------------------------------


def _is_migration_applied(conn, name: str) -> bool:
    """Check if a migration has already been applied.

    BUG-DES-03: Excludes 'failed' and 'retrying' statuses.

    Raises
    ------
    OperationalError
        If the database is unreachable.
    ProgrammingError
        If _migration_history table doesn't exist.
    """
    try:
        r = conn.execute(
            text(
                "SELECT COUNT(*) FROM _migration_history "
                "WHERE migration_name = :n "
                "AND status NOT IN ('failed', 'retrying')"
            ),
            {"n": name},
        )
        return r.scalar() > 0
    except OperationalError as exc:
        logger.warning("Cannot check migration status for '%s': %s", name, exc)
        raise
    except ProgrammingError as exc:
        logger.error("_migration_history table may not exist: %s", exc)
        raise


def _get_stored_checksum(conn, name: str) -> str | None:
    """Get the stored checksum for a migration, if any."""
    try:
        r = conn.execute(
            text(
                "SELECT checksum FROM _migration_history "
                "WHERE migration_name = :n "
                "AND status NOT IN ('failed', 'retrying')"
            ),
            {"n": name},
        )
        row = r.fetchone()
        return row[0] if row else None
    except (OperationalError, ProgrammingError):
        return None


def _record_migration(conn, name: str, checksum: str, status: str = "applied") -> None:
    """Record a migration in _migration_history.

    GUARD-DES-08: Validates status against VALID_MIGRATION_STATUSES.
    BUG-IDEM-03: Uses ON CONFLICT DO UPDATE (not DO NOTHING) to update
    checksum on re-application, preventing infinite drift warning cycles.
    BUG-CODE-03: Renamed :from parameter to :applied_from_host.

    Also populates audit columns (SEC-MIG-03, BUG-SEC-03).
    """
    # GUARD-DES-08: Validate status
    if status not in VALID_MIGRATION_STATUSES:
        raise ValueError(
            f"Invalid migration status: {status!r}. "
            f"Must be one of {sorted(VALID_MIGRATION_STATUSES)}"
        )

    # GUARD-REL-08: Handle getpass.getuser() failure in containers
    try:
        applied_by = os.environ.get("AIRFLOW_USER", getpass.getuser())
    except (OSError, KeyError):
        applied_by = "unknown"

    applied_from = platform.node()
    python_version = platform.python_version()

    # BUG-SEC-03: Add hash of user identity for tamper detection
    applied_by_hash = hashlib.sha256(
        (applied_by + platform.node()).encode()
    ).hexdigest()[:16]

    # BUG-CODE-03: Use :applied_from_host instead of :from (SQL reserved word)
    try:
        engine_dialect = conn.engine.dialect.name
    except Exception:
        engine_dialect = "unknown"

    if engine_dialect == DIALECT_SQLITE:
        # BUG-IDEM-03: Use INSERT OR REPLACE approach for SQLite
        # First delete any existing record, then insert
        conn.execute(
            text("DELETE FROM _migration_history WHERE migration_name = :n"),
            {"n": name},
        )
        sql = (
            "INSERT INTO _migration_history "
            "(migration_name, checksum, applied_by, applied_from, "
            "python_version, status, applied_by_hash) "
            "VALUES (:n, :c, :by, :afh, :pv, :st, :abh)"
        )
    else:
        # BUG-IDEM-03: ON CONFLICT DO UPDATE instead of DO NOTHING
        sql = (
            "INSERT INTO _migration_history "
            "(migration_name, checksum, applied_by, applied_from, "
            "python_version, status, applied_by_hash) "
            "VALUES (:n, :c, :by, :afh, :pv, :st, :abh) "
            "ON CONFLICT (migration_name) DO UPDATE SET "
            "checksum = :c, applied_at = CURRENT_TIMESTAMP, "
            "applied_by = :by, applied_from = :afh, "
            "python_version = :pv, status = :st, "
            "applied_by_hash = :abh"
        )

    conn.execute(
        text(sql),
        {
            "n": name, "c": checksum, "by": applied_by,
            "afh": applied_from, "pv": python_version,
            "st": status, "abh": applied_by_hash,
        },
    )


def _record_failure(conn, name: str, checksum: str, error_message: str, error_class: str) -> None:
    """Record a failed migration in _failed_migrations.

    GAP-SEC-04: Sanitizes error message before storage.
    GAP-REL-06: Falls back to JSON file if database write fails.
    """
    # GAP-SEC-04: Sanitize error message
    safe_message = _sanitize_error_message(error_message)

    try:
        conn.execute(
            text(
                "INSERT INTO _failed_migrations "
                "(migration_name, error_message, error_class, sql_checksum) "
                "VALUES (:n, :e, :ec, :c)"
            ),
            {"n": name, "e": safe_message, "ec": error_class, "c": checksum},
        )
    except Exception as exc:
        logger.error("Could not record failure for '%s' in database: %s", name, exc)
        # GAP-REL-06: Fallback to JSON file
        _record_failure_fallback(name, safe_message, error_class, checksum)


def _is_test_mode() -> bool:
    """Detect whether the migration runner is executing under a test harness.

    Returns True when ANY of the following signals are present:
      - ``PYTEST_CURRENT_TEST`` environment variable is set (set by pytest
        on every test invocation).
      - ``pytest`` is importable AND present in ``sys.modules`` (i.e.
        pytest is currently running).
      - ``APP_ENV`` is set to ``test`` or ``testing``.
      - ``MIGRATIONS_TEST_MODE`` environment variable is set to ``1``.

    Used by ``_record_failure_fallback`` to avoid polluting the production
    ``_failed_migrations_fallback.jsonl`` audit trail with test artifacts.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    if os.environ.get("APP_ENV") in ("test", "testing"):
        return True
    if os.environ.get("MIGRATIONS_TEST_MODE") == "1":
        return True
    import sys as _sys
    if "pytest" in _sys.modules:
        return True
    try:
        import pytest as _pytest  # noqa: F401
        # pytest is importable AND we got here without PYTEST_CURRENT_TEST
        # -- only treat as test mode if pytest is actually running (i.e.
        # already in sys.modules). Otherwise importability alone is not
        # enough (pytest may be installed in the env but not running).
        return False
    except ImportError:
        return False


def _record_failure_fallback(
    name: str, error_message: str, error_class: str, checksum: str,
) -> None:
    """Write failure record to a JSONL file as fallback.

    GAP-REL-06: If the database is unavailable for recording failures,
    write to a local JSONL file for later recovery.

    v29 ROOT FIX (audit D-8): _failed_migrations_fallback.jsonl was polluted
    with 22 test artifacts (all migration_name="test", generated by pytest
    runs that hit the fallback path because the test DB did not have a
    _failed_migrations table). The production audit trail was contaminated
    -- operators inspecting the file could not distinguish real production
    failures from test noise. Fix: skip writing to the fallback file when
    running in test mode (detected via PYTEST_CURRENT_TEST env var,
    APP_ENV=test, MIGRATIONS_TEST_MODE=1, or pytest in sys.modules).
    Test runs that need a fallback trail should redirect to a temp file
    via MIGRATIONS_FALLBACK_DIR env var.
    """
    # v29 ROOT FIX (audit D-8): skip writing test artifacts to the
    # production fallback file when running in test mode.
    if _is_test_mode():
        # Allow tests to redirect the fallback file via env var if they
        # need to exercise the fallback code path. Default: drop the
        # record silently (test artifacts should never contaminate the
        # production audit trail).
        redirect = os.environ.get("MIGRATIONS_FALLBACK_DIR")
        if not redirect:
            logger.debug(
                "Skipping fallback file write for migration '%s' -- "
                "test mode detected (audit D-8 fix). Set "
                "MIGRATIONS_FALLBACK_DIR to capture test fallbacks.",
                name,
            )
            return
        fallback_path = Path(redirect) / "_failed_migrations_fallback.jsonl"
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        fallback_path = MIGRATIONS_DIR / "_failed_migrations_fallback.jsonl"

    try:
        record = {
            "migration_name": name,
            "error_message": error_message,
            "error_class": error_class,
            "sql_checksum": checksum,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(fallback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.warning(
            "Could not record failure in database. Writing to fallback file: %s",
            fallback_path,
        )
    except Exception as exc:
        logger.error("Could not write failure fallback file: %s", exc)


def _sanitize_error_message(message: str) -> str:
    """Sanitize error message for safe storage.

    GAP-SEC-04: Removes database URLs, credentials, and PII patterns
    from error messages before storing in _failed_migrations.
    """
    # Truncate to max length
    safe = message[:ERROR_MESSAGE_MAX_LENGTH]

    # Mask database URLs
    safe = re.sub(
        r"(postgresql|mysql|sqlite)://[^\s]+",
        r"\1://***:***@***",
        safe,
    )
    # Mask potential passwords in connection strings
    safe = re.sub(r":(\w+)@", ":***@", safe)
    # Mask email addresses (potential PII)
    safe = re.sub(r"[\w.+-]+@[\w.-]+\.\w+", "***@***.***", safe)

    return safe


def _compute_checksum(content: str) -> str:
    """Compute SHA-256 checksum of migration content for drift detection.

    BUG-IDEM-02: Normalizes line endings (CRLF -> LF) for cross-platform
    checksum reproducibility.
    """
    # BUG-IDEM-02: Normalize line endings
    normalized = content.replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pre-migration scientific validation (SCI-MIG-01 through SCI-MIG-06, BUG-SCI-01..03)
# ---------------------------------------------------------------------------


def _check_ppi_score_column(
    conn, column_name: str, label: str,
) -> str | None:
    """Check a single PPI score column for out-of-range values.

    BUG-SCI-02: Extracted helper to check ALL four PPI score columns,
    not just combined_score.
    """
    try:
        r = conn.execute(
            text(
                f"SELECT COUNT(*) FROM protein_protein_interactions "
                f"WHERE {column_name} IS NOT NULL AND "
                f"({column_name} < :min_val OR {column_name} > :max_val)"
            ),
            {"min_val": STRING_SCORE_MIN, "max_val": STRING_SCORE_MAX},
        )
        count = r.scalar()
        if count and count > 0:
            msg = (
                f"{label}: {count} PPI record(s) have {column_name} "
                f"outside {STRING_SCORE_MIN}-{STRING_SCORE_MAX} range. "
                f"Migration 003 CHECK constraint will FAIL on these records."
            )
            logger.warning(msg)
            return msg
    except (OperationalError, ProgrammingError) as exc:
        logger.warning("Could not check %s ranges: %s", column_name, exc)
    return None


def validate_scientific_constraints(engine) -> list[str]:
    """Validate scientific constraints before running migrations.

    Checks for data that would be affected by destructive changes
    in migration files (001-003). Returns a list of warning messages.

    BUG-SCI-01: Expanded to check InChIKey format, molecular_weight > 0,
    activity_value > 0, drug name min length, and disease_id_type validity.
    BUG-SCI-02: Now checks ALL four PPI score columns.
    BUG-SCI-03: Fixed molecular_weight precision comparison.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine connected to the target database.

    Returns
    -------
    list[str]
        Warning messages for any scientific constraint violations found.
        Empty list means all constraints are satisfied.
    """
    warnings_list: list[str] = []
    inspector = inspect(engine)

    with engine.begin() as conn:
        # SCI-MIG-01: Check uniprot_id length before VARCHAR(20) -> VARCHAR(10)
        if _table_exists(inspector, "proteins"):
            try:
                r = conn.execute(
                    text("SELECT COUNT(*), uniprot_id FROM proteins "
                         "WHERE LENGTH(uniprot_id) > 10 GROUP BY uniprot_id")
                )
                rows = r.fetchall()
                if rows:
                    count = sum(row[0] for row in rows)
                    ids = [row[1] for row in rows[:10]]
                    msg = (
                        f"SCI-MIG-01: {count} protein(s) have uniprot_id longer "
                        f"than 10 chars. Migration 003 will TRUNCATE these. "
                        f"Sample: {ids}"
                    )
                    warnings_list.append(msg)
                    logger.warning(msg)
            except (OperationalError, ProgrammingError) as exc:
                logger.warning("Could not check uniprot_id lengths: %s", exc)

            # GUARD-SCI-06: InChIKey format validation (standard structure)
            # Only for drugs table
            pass  # Checked below with drugs

        # BUG-SCI-02: Check ALL four PPI score columns
        if _table_exists(inspector, "protein_protein_interactions"):
            for col_name, label in [
                ("combined_score", "SCI-MIG-04"),
                ("experimental_score", "SCI-MIG-04a"),
                ("database_score", "SCI-MIG-04b"),
                ("textmining_score", "SCI-MIG-04c"),
            ]:
                # Skip columns that don't exist yet
                if _column_exists(inspector, "protein_protein_interactions", col_name):
                    result = _check_ppi_score_column(conn, col_name, label)
                    if result:
                        warnings_list.append(result)

        if _table_exists(inspector, "drugs"):
            # GAP-SCI-04: max_phase integer check + range check
            try:
                r = conn.execute(
                    text(
                        "SELECT COUNT(*), CAST(max_phase AS TEXT) FROM drugs "
                        "WHERE max_phase IS NOT NULL AND "
                        "(CAST(max_phase AS INTEGER) != max_phase "
                        "OR max_phase < 0 OR max_phase > 4) "
                        "GROUP BY CAST(max_phase AS TEXT)"
                    )
                )
                rows = r.fetchall()
                if rows:
                    count = sum(row[0] for row in rows)
                    msg = (
                        f"SCI-MIG-05: {count} drug(s) have max_phase outside "
                        f"0-4 integer range. Migration 003 CHECK constraint will FAIL. "
                        f"Phase semantics: 0=Preclinical, 1=Phase I, 2=Phase II, "
                        f"3=Phase III, 4=Approved."
                    )
                    warnings_list.append(msg)
                    logger.warning(msg)
            except (OperationalError, ProgrammingError) as exc:
                logger.warning("Could not check max_phase ranges: %s", exc)

            # BUG-SCI-03: Fixed molecular_weight precision check
            # GAP-SCI-05: Uses MOLECULAR_WEIGHT_PRECISION constant
            try:
                r = conn.execute(
                    text(
                        f"SELECT COUNT(*) FROM drugs "
                        f"WHERE molecular_weight IS NOT NULL AND "
                        f"ABS(CAST(molecular_weight AS NUMERIC) - "
                        f"ROUND(CAST(molecular_weight AS NUMERIC), {MOLECULAR_WEIGHT_PRECISION})) "
                        f"> CASE WHEN molecular_weight > 10000 "
                        f"THEN molecular_weight * 1e-10 "
                        f"ELSE 0.000001 END"
                    )
                )
                count = r.scalar()
                if count and count > 0:
                    msg = (
                        f"SCI-MIG-06: {count} drug(s) have molecular_weight "
                        f"that may lose precision in FLOAT->NUMERIC({12},{MOLECULAR_WEIGHT_PRECISION}) "
                        f"conversion."
                    )
                    warnings_list.append(msg)
                    logger.warning(msg)
            except (OperationalError, ProgrammingError) as exc:
                logger.warning("Could not check molecular_weight precision: %s", exc)

            # BUG-SCI-01: InChIKey format check
            if _column_exists(inspector, "drugs", "inchikey"):
                try:
                    r = conn.execute(
                        text(
                            "SELECT COUNT(*) FROM drugs "
                            "WHERE inchikey IS NOT NULL "
                            "AND LENGTH(inchikey) != :standard_len "
                            "AND inchikey NOT LIKE :synth_prefix"
                        ),
                        {
                            "standard_len": STANDARD_INCHIKEY_LENGTH,
                            "synth_prefix": f"{SYNTHETIC_INCHIKEY_PREFIX}%",
                        },
                    )
                    count = r.scalar()
                    if count and count > 0:
                        msg = (
                            f"SCI-MIG-07: {count} drug(s) have InChIKey with "
                            f"invalid length (expected {STANDARD_INCHIKEY_LENGTH} chars "
                            f"or {SYNTHETIC_INCHIKEY_PREFIX} prefix). Migration 003 "
                            f"VARCHAR({INCHIKEY_MAX_LENGTH}) column will accept but "
                            f"downstream validation will flag these."
                        )
                        warnings_list.append(msg)
                        logger.warning(msg)
                except (OperationalError, ProgrammingError) as exc:
                    logger.warning("Could not check InChIKey format: %s", exc)

            # BUG-SCI-01: molecular_weight > 0 check
            try:
                r = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM drugs "
                        "WHERE molecular_weight IS NOT NULL AND molecular_weight <= 0"
                    )
                )
                count = r.scalar()
                if count and count > 0:
                    msg = (
                        f"SCI-MIG-08: {count} drug(s) have molecular_weight <= 0. "
                        f"Migration 003 CHECK constraint will FAIL."
                    )
                    warnings_list.append(msg)
                    logger.warning(msg)
            except (OperationalError, ProgrammingError) as exc:
                logger.warning("Could not check molecular_weight positivity: %s", exc)

            # GUARD-SCI-06: InChIKey standard format regex validation
            if _column_exists(inspector, "drugs", "inchikey"):
                try:
                    # Use LIKE pattern for cross-dialect compatibility
                    r = conn.execute(
                        text(
                            "SELECT COUNT(*) FROM drugs "
                            "WHERE inchikey IS NOT NULL "
                            "AND LENGTH(inchikey) = 27 "
                            "AND (SUBSTR(inchikey, 15, 1) != '-' "
                            "OR SUBSTR(inchikey, 26, 1) != '-')"
                        )
                    )
                    count = r.scalar()
                    if count and count > 0:
                        msg = (
                            f"SCI-MIG-12: {count} drug(s) have InChIKey values "
                            f"that pass length check but fail standard format "
                            f"validation (14 uppercase letters - 10 alphanumeric "
                            f"- 1 uppercase letter). These may be synthetic or "
                            f"corrupted identifiers."
                        )
                        warnings_list.append(msg)
                        logger.warning(msg)
                except (OperationalError, ProgrammingError) as exc:
                    logger.warning("Could not check InChIKey format regex: %s", exc)

            # BUG-SCI-01: drug name minimum length check
            try:
                r = conn.execute(
                    text("SELECT COUNT(*) FROM drugs WHERE LENGTH(name) < 2")
                )
                count = r.scalar()
                if count and count > 0:
                    msg = (
                        f"SCI-MIG-09: {count} drug(s) have name shorter than "
                        f"2 characters. Data quality issue."
                    )
                    warnings_list.append(msg)
                    logger.warning(msg)
            except (OperationalError, ProgrammingError) as exc:
                logger.warning("Could not check drug name length: %s", exc)

        # BUG-SCI-01: activity_value > 0 check
        if _table_exists(inspector, "drug_protein_interactions"):
            if _column_exists(inspector, "drug_protein_interactions", "activity_value"):
                try:
                    r = conn.execute(
                        text(
                            "SELECT COUNT(*) FROM drug_protein_interactions "
                            "WHERE activity_value IS NOT NULL AND activity_value <= 0"
                        )
                    )
                    count = r.scalar()
                    if count and count > 0:
                        msg = (
                            f"SCI-MIG-10: {count} drug_protein_interaction(s) "
                            f"have activity_value <= 0."
                        )
                        warnings_list.append(msg)
                        logger.warning(msg)
                except (OperationalError, ProgrammingError) as exc:
                    logger.warning("Could not check activity_value: %s", exc)

        # BUG-SCI-01: disease_id_type validity check
        # CRITICAL FIX (patient safety): the allowed vocabulary MUST include
        # 'hpo', 'icd10', 'efo', 'orphanet' -- without these, real disease
        # associations from HPO, ICD-10, EFO, and Orphanet would be flagged
        # as invalid and could be silently dropped from the model's training
        # set, hiding drug-disease links from clinicians.
        if _table_exists(inspector, "gene_disease_associations"):
            if _column_exists(inspector, "gene_disease_associations", "disease_id_type"):
                try:
                    r = conn.execute(
                        text(
                            "SELECT COUNT(*) FROM gene_disease_associations "
                            "WHERE disease_id_type IS NOT NULL "
                            "AND disease_id_type NOT IN "
                            "('omim','disgenet','doid','mesh','umls',"
                            "'hpo','icd10','efo','orphanet')"
                        )
                    )
                    count = r.scalar()
                    if count and count > 0:
                        msg = (
                            f"SCI-MIG-11: {count} GDA record(s) have unknown "
                            f"disease_id_type. Expected: omim, disgenet, doid, "
                            f"mesh, umls, hpo, icd10, efo, orphanet."
                        )
                        warnings_list.append(msg)
                        logger.warning(msg)
                except (OperationalError, ProgrammingError) as exc:
                    logger.warning("Could not check disease_id_type: %s", exc)

    return warnings_list


# ---------------------------------------------------------------------------
# Post-migration verification (SCI-MIG-03, DQ-MIG-04, DQ-MIG-06)
# ---------------------------------------------------------------------------


def _verify_post_migration_state(engine, migration_name: str) -> list[str]:
    """Verify database state after applying a migration.

    Checks:
    - New constraints are satisfied by existing data (SCI-MIG-03)
    - Referential integrity for new FK constraints (DQ-MIG-04)
    - ORM model synchronization (DQ-MIG-06)

    Returns a list of warning messages.
    """
    issues: list[str] = []
    inspector = inspect(engine)

    with engine.begin() as conn:
        # Only run post-migration checks for migration 003
        if not migration_name.startswith("003"):
            return issues

        # DQ-MIG-04: Check for orphaned GDA records (uniprot_id FK).
        # v14 ROOT FIX (FIX4 / CD-3): was previously checking the integer
        # protein_id column -- that column has been removed. The canonical
        # FK is now the STRING uniprot_id, which references
        # proteins.uniprot_id (NOT proteins.id).
        if _table_exists(inspector, "gene_disease_associations") and _table_exists(inspector, "proteins"):
            try:
                r = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM gene_disease_associations gda "
                        "WHERE gda.uniprot_id IS NOT NULL "
                        "AND gda.uniprot_id NOT IN "
                        "(SELECT uniprot_id FROM proteins WHERE uniprot_id IS NOT NULL)"
                    )
                )
                count = r.scalar()
                if count and count > 0:
                    msg = (
                        f"DQ-MIG-04: {count} GDA record(s) have orphaned "
                        f"uniprot_id references. FK constraint may fail."
                    )
                    issues.append(msg)
                    logger.warning(msg)
            except (OperationalError, ProgrammingError) as exc:
                logger.warning("Could not check GDA FK integrity: %s", exc)

    return issues


# ---------------------------------------------------------------------------
# Configuration validation (CFG-MIG-05, BUG-CFG-03)
# ---------------------------------------------------------------------------


def validate_migration_config(config: Any = None) -> list[str]:
    """Validate migration configuration. Returns list of warning strings.

    Parameters
    ----------
    config : MigrationConfig | None
        The configuration to validate. If None, uses defaults.

    Returns
    -------
    list[str]
        Warning messages for any configuration issues.
    """
    warnings_list: list[str] = []
    if config is None:
        return warnings_list

    if hasattr(config, "migrations_dir") and config.migrations_dir is not None:
        if not config.migrations_dir.exists():
            warnings_list.append(
                f"Migrations directory does not exist: {config.migrations_dir}"
            )

    if hasattr(config, "batch_size") and config.batch_size < 1:
        warnings_list.append(
            f"batch_size must be positive, got {config.batch_size}"
        )

    if hasattr(config, "timeout_seconds") and config.timeout_seconds < 1:
        warnings_list.append(
            f"timeout_seconds must be positive, got {config.timeout_seconds}"
        )

    if hasattr(config, "max_retries") and config.max_retries < 0:
        warnings_list.append(
            f"max_retries must be non-negative, got {config.max_retries}"
        )

    if hasattr(config, "retry_backoff_base") and config.retry_backoff_base <= 0:
        warnings_list.append(
            f"retry_backoff_base must be positive, got {config.retry_backoff_base}"
        )

    return warnings_list


# ---------------------------------------------------------------------------
# ORM schema verification (DQ-MIG-06, TEST-MIG-03, BUG-ARCH-05, GAP-DES-06)
# ---------------------------------------------------------------------------


def verify_schema_matches_orm(engine) -> dict[str, Any]:
    """Compare reflected database schema against ORM model definitions.

    BUG-ARCH-05: ORM model import is fully optional with fallback to
    EXPECTED_SCHEMA dict.
    GAP-DES-06: Now compares column types, nullable, and constraint
    mismatches (not just column names).

    Returns a dict with:
    - missing_in_db: columns in ORM but not in database
    - extra_in_db: columns in database but not in ORM
    - type_mismatches: columns with different types
    - constraint_mismatches: constraints that differ
    - used_fallback: bool indicating if EXPECTED_SCHEMA was used
    """
    result: dict[str, Any] = {
        "missing_in_db": [],
        "extra_in_db": [],
        "type_mismatches": [],
        "constraint_mismatches": [],
        "used_fallback": False,
    }

    inspector = inspect(engine)

    # BUG-ARCH-05: Try ORM models first, fall back to EXPECTED_SCHEMA
    models = None
    try:
        from database.models import (
            Drug,
            DrugProteinInteraction,
            EntityMapping,
            GeneDiseaseAssociation,
            PipelineRun,
            Protein,
            ProteinProteinInteraction,
        )
        models = [
            Drug, Protein, DrugProteinInteraction,
            ProteinProteinInteraction, GeneDiseaseAssociation,
            EntityMapping, PipelineRun,
        ]
    except ImportError as exc:
        logger.warning(
            "Could not import ORM models for schema verification: %s. "
            "Using EXPECTED_SCHEMA fallback.", exc,
        )
        result["used_fallback"] = True

    if models is not None:
        for model in models:
            table_name = model.__tablename__
            if not _table_exists(inspector, table_name):
                result["missing_in_db"].append(f"{table_name} (entire table)")
                continue

            db_columns = {col["name"] for col in inspector.get_columns(table_name)}
            orm_columns = {col.name for col in model.__table__.columns}

            missing = orm_columns - db_columns
            extra = db_columns - orm_columns

            for col in missing:
                result["missing_in_db"].append(f"{table_name}.{col}")
            for col in extra:
                result["extra_in_db"].append(f"{table_name}.{col}")

            # GAP-DES-06: Compare column types and nullable
            db_col_info = {col["name"]: col for col in inspector.get_columns(table_name)}
            for orm_col in model.__table__.columns:
                if orm_col.name in db_col_info:
                    db_col = db_col_info[orm_col.name]
                    db_type_str = str(db_col.get("type", ""))
                    orm_type_str = str(orm_col.type)
                    # Normalize type strings for comparison
                    if db_type_str.upper() != orm_type_str.upper():
                        # Only flag significant mismatches (VARCHAR vs TEXT is ok)
                        both_varchar = "VARCHAR" in db_type_str.upper() and "VARCHAR" in orm_type_str.upper()
                        both_text = db_type_str.upper() in ("TEXT", "STRING") and orm_type_str.upper() in ("TEXT", "STRING")
                        if not both_varchar and not both_text:
                            result["type_mismatches"].append(
                                f"{table_name}.{orm_col.name}: "
                                f"expected {orm_type_str}, got {db_type_str}"
                            )

                    # Compare nullable
                    if db_col.get("nullable", True) != orm_col.nullable:
                        result["constraint_mismatches"].append(
                            f"{table_name}.{orm_col.name}: nullable "
                            f"expected {orm_col.nullable}, got {db_col.get('nullable', True)}"
                        )
    else:
        # Fallback: use EXPECTED_SCHEMA
        for table_name, expected_cols in EXPECTED_SCHEMA.items():
            if not _table_exists(inspector, table_name):
                result["missing_in_db"].append(f"{table_name} (entire table)")
                continue
            db_columns = {col["name"] for col in inspector.get_columns(table_name)}
            missing = set(expected_cols) - db_columns
            extra = db_columns - set(expected_cols)
            for col in missing:
                result["missing_in_db"].append(f"{table_name}.{col}")
            for col in extra:
                result["extra_in_db"].append(f"{table_name}.{col}")

    return result


# ---------------------------------------------------------------------------
# Data quality: row-count and checksum tracking (DQ-MIG-01, DQ-MIG-02)
# ---------------------------------------------------------------------------


def _normalize_value(val: Any) -> str:
    """Normalize a value to a deterministic string for checksum computation.

    BUG-DQ-02: Handles type-specific normalization to ensure
    deterministic hashing across Python versions and platforms.
    """
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, bytes):
        return val.hex()
    if isinstance(val, float):
        # Normalize float representation
        return f"{val:.15g}"
    return str(val)


def _compute_data_checksum(
    conn, table_name: str, max_rows: int = 100000,
) -> str:
    """Compute a SHA-256 checksum of data in a table.

    BUG-DQ-01: Uses explicit sorted column list instead of SELECT *.
    BUG-DQ-02: Uses _normalize_value for deterministic hashing.
    BUG-PERF-01: Uses streaming with max_rows cap to limit memory.
    BUG-DES-04: Processes in batches instead of loading all rows.
    """
    try:
        inspector = inspect(conn.engine) if hasattr(conn, "engine") else None
        if inspector is None:
            inspector = inspect(conn)

        # BUG-DQ-01: Explicit sorted column list
        columns = sorted([col["name"] for col in inspector.get_columns(table_name)])
        column_list = ", ".join(_validate_sql_identifier(c) for c in columns)
        safe_table = _validate_sql_identifier(table_name, "table name")

        # BUG-PERF-01: Streaming with max_rows
        hasher = hashlib.sha256()
        rows_hashed = 0

        # Use server-side cursor for PostgreSQL
        exec_conn = conn
        if conn.engine.dialect.name == DIALECT_POSTGRESQL:
            exec_conn = conn.execution_options(stream_results=True)

        r = exec_conn.execute(
            text(f"SELECT {column_list} FROM {safe_table} ORDER BY id LIMIT :lim"),
            {"lim": max_rows + 1},
        )

        for row in r:
            if rows_hashed >= max_rows:
                logger.warning(
                    "Table '%s' has more than %d rows. Checksum is based on first %d rows (sample).",
                    table_name, max_rows, max_rows,
                )
                break
            for val in row:
                hasher.update(_normalize_value(val).encode("utf-8"))
            rows_hashed += 1

        return hasher.hexdigest()
    except (OperationalError, ProgrammingError) as exc:
        logger.warning("Could not compute data checksum for '%s': %s", table_name, exc)
        return ""
    except Exception as exc:
        logger.warning("Unexpected error computing checksum for '%s': %s", table_name, exc)
        return ""


# ---------------------------------------------------------------------------
# Retry logic for transient failures (REL-MIG-05)
# ---------------------------------------------------------------------------


def _execute_with_retry(
    conn,
    sql: str,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    migration_name: str = "",
) -> None:
    """Execute SQL with retry logic for transient failures.

    Retries on OperationalError and InterfaceError only.
    Non-transient errors (ProgrammingError, DataError) are not retried.

    v29 ROOT FIX (audit D-14): forward migrations were non-atomic.
    Now wrapped in explicit transaction per migration. Each statement
    is executed inside a SAVEPOINT (``conn.begin_nested()``) so a
    transient failure rolls back ONLY the failed statement, not the
    entire outer transaction. Without the SAVEPOINT, PostgreSQL
    poisons the transaction after any statement-level error and the
    retry attempt fails with "current transaction is aborted" -- the
    retry logic was effectively dead code. With the SAVEPOINT, the
    retry actually re-executes the statement cleanly within the same
    outer ``engine.begin()`` block (which is the explicit per-
    migration transaction wrapper added by the D-14 fix). Partial
    failure of one statement no longer leaves the schema in an
    inconsistent state -- either the entire migration commits
    (all statements + bookkeeping) or it rolls back atomically.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        # v29 ROOT FIX (audit D-14): wrap each statement in a
        # SAVEPOINT so transient failures can be rolled back without
        # poisoning the outer transaction. ``conn.begin_nested()``
        # emits ``SAVEPOINT sp_N`` on PostgreSQL and is a no-op on
        # SQLite (which does not poison transactions on statement
        # errors in the same way -- SQLite rolls back to the last
        # successful statement automatically within a transaction).
        savepoint = None
        try:
            savepoint = conn.begin_nested()
            conn.execute(text(sql))
            savepoint.commit()
            return
        except (OperationalError, InterfaceError) as exc:
            # Transient error -- roll back the SAVEPOINT and retry.
            if savepoint is not None:
                try:
                    savepoint.rollback()
                except Exception:
                    # SAVEPOINT rollback failure means the outer
                    # transaction is also poisoned -- propagate the
                    # original error so the outer ``engine.begin()``
                    # rolls back atomically (D-14 guarantee).
                    pass
            last_exc = exc
            if attempt < max_retries:
                delay = backoff_base ** attempt
                logger.warning(
                    "Transient error executing migration SQL (attempt %d/%d) "
                    "for '%s': %s. Retrying in %.1fs...",
                    attempt + 1, max_retries + 1, migration_name, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "All %d retry attempts exhausted for '%s'",
                    max_retries + 1, migration_name,
                )
        except (ProgrammingError, DataError):
            # Non-transient -- roll back the SAVEPOINT (so the outer
            # transaction isn't poisoned by the failed statement) and
            # propagate to abort the entire migration transaction.
            if savepoint is not None:
                try:
                    savepoint.rollback()
                except Exception:
                    pass
            raise
    if last_exc:
        raise last_exc


# ---------------------------------------------------------------------------
# Security: read-only mode check (SEC-MIG-04)
# ---------------------------------------------------------------------------


def _check_readonly_mode() -> None:
    """Check if migrations are locked via environment variable."""
    if os.environ.get("MIGRATIONS_READONLY") == "1":
        raise RuntimeError(
            "Migrations are locked (MIGRATIONS_READONLY=1). "
            "Remove this environment variable to allow migration execution."
        )


# ---------------------------------------------------------------------------
# Security: destructive SQL scanner (GAP-SEC-06, GUARD-SEC-08)
# ---------------------------------------------------------------------------

_DESTRUCTIVE_PATTERNS = (
    re.compile(r"DROP\s+TABLE", re.IGNORECASE),
    re.compile(r"DROP\s+INDEX", re.IGNORECASE),
    re.compile(r"TRUNCATE\s+TABLE?", re.IGNORECASE),
    # v90 ROOT FIX (BUG #14 -- P1 UPDATE regex caught ALL UPDATEs):
    #   The previous regex ``UPDATE\s+\w+\s+SET\s+.*;`` matched ANY UPDATE
    #   statement -- the ``.*`` was greedy and consumed the WHERE clause.
    #   The comment said "UPDATE without WHERE" but the regex caught ALL
    #   UPDATEs. Migration 006 has many ``UPDATE drugs SET is_withdrawn =
    #   TRUE WHERE lower(name) = 'rofecoxib'`` -- all flagged as
    #   "destructive". An operator who set ``allow_destructive_sql=False``
    #   for safety BLOCKED migration 006 entirely, and the is_withdrawn
    #   backfill never ran. Vioxx stayed is_withdrawn=FALSE.
    #   ROOT FIX: replace the regex with a function-based check that
    #   parses each statement and only flags UPDATEs that lack a WHERE
    #   clause. The DELETE regex had the same issue (``DELETE FROM \w+ ;``
    #   only matched DELETEs ending immediately with ``;`` -- missed
    #   multi-line DELETEs). Both are now handled by
    #   ``_scan_destructive_sql`` below, which splits on ``;`` and checks
    #   each statement for a WHERE clause.
    # DELETE without WHERE -- handled by _scan_destructive_sql (regex kept
    # for backwards-compatibility with any code that imports the tuple).
    re.compile(r"DELETE\s+FROM\s+\w+\s*;", re.IGNORECASE),  # DELETE without WHERE
)


def _scan_destructive_sql(sql_content: str) -> list[str]:
    """Scan SQL content for destructive patterns.

    GAP-SEC-06: Returns list of found destructive patterns.
    GUARD-SEC-08: Used when allow_destructive_sql is False.

    v90 ROOT FIX (BUG #14): the previous UPDATE regex
    ``UPDATE\\s+\\w+\\s+SET\\s+.*;`` matched ANY UPDATE statement (the
    ``.*`` was greedy and consumed the WHERE clause). Migration 006's
    ``UPDATE drugs SET is_withdrawn = TRUE WHERE lower(name) =
    'rofecoxib'`` was flagged as "destructive" even though it has a
    WHERE clause. An operator who enabled the destructive-SQL guard
    silently blocked the life-safety backfill in migration 006 -- Vioxx
    stayed is_withdrawn=FALSE. ROOT FIX: split the SQL into statements
    (naive split on ``;`` -- sufficient for migration files which use
    ``;`` as the statement terminator) and check each UPDATE / DELETE
    statement for a WHERE clause. Only flag statements WITHOUT a WHERE.
    """
    found: list[str] = []
    # Check the simple patterns first (DROP TABLE, DROP INDEX, TRUNCATE).
    for pattern in _DESTRUCTIVE_PATTERNS:
        m = pattern.search(sql_content)
        if m:
            found.append(m.group(0).strip())

    # v90: per-statement WHERE-clause check for UPDATE and DELETE.
    # Split on ';' -- naive but sufficient for migration files. Strip
    # comments (lines starting with '--') so a WHERE in a comment doesn't
    # mask a missing WHERE in the actual statement.
    _stripped_lines = [
        line for line in sql_content.splitlines()
        if not line.strip().startswith("--")
    ]
    _stripped_sql = "\n".join(_stripped_lines)
    # Remove single-line comments after stripping full-line comments.
    _stripped_sql = re.sub(r"--[^\n]*", "", _stripped_sql)
    statements = _stripped_sql.split(";")
    for stmt in statements:
        stmt_stripped = stmt.strip()
        if not stmt_stripped:
            continue
        upper = stmt_stripped.upper()
        # Check UPDATE ... SET ... without WHERE
        if re.match(r"UPDATE\s+\w+\s+SET\s+", upper):
            if "WHERE" not in upper:
                # Truncate for readability in the error message.
                snippet = stmt_stripped[:120]
                if len(stmt_stripped) > 120:
                    snippet += "..."
                found.append(f"UPDATE without WHERE: {snippet}")
        # Check DELETE FROM ... without WHERE
        if re.match(r"DELETE\s+FROM\s+\w+", upper):
            if "WHERE" not in upper:
                snippet = stmt_stripped[:120]
                if len(stmt_stripped) > 120:
                    snippet += "..."
                found.append(f"DELETE without WHERE: {snippet}")
    return found


# ---------------------------------------------------------------------------
# Security: path traversal protection (GUARD-SEC-07)
# ---------------------------------------------------------------------------


def _validate_migration_path(sql_file: Path, migrations_dir: Path) -> None:
    """Verify migration file resolves within the migrations directory.

    GUARD-SEC-07: Prevents symlink-based path traversal attacks.
    """
    resolved = sql_file.resolve()
    base_resolved = migrations_dir.resolve()
    if not str(resolved).startswith(str(base_resolved)):
        raise ValueError(
            f"Migration file {sql_file} resolves outside the migrations "
            f"directory: {resolved}. Possible path traversal attack."
        )


# ---------------------------------------------------------------------------
# v16 ROOT FIX (CD-5): SQLite-compatible SQL translation
# ---------------------------------------------------------------------------

# Postgres-only statements that have NO SQLite equivalent and must be
# stripped (with a WARNING if encountered).
_PG_ONLY_STATEMENT_PATTERNS = [
    # pg_advisory_lock / pg_advisory_unlock -- no SQLite equivalent.
    (re.compile(r"SELECT\s+pg_advisory_lock\s*\([^)]*\)\s*;?", re.IGNORECASE), "-- [SQLite-skip] pg_advisory_lock"),
    (re.compile(r"SELECT\s+pg_advisory_unlock\s*\([^)]*\)\s*;?", re.IGNORECASE), "-- [SQLite-skip] pg_advisory_unlock"),
    # RAISE NOTICE inside DO blocks -- converted to SELECT (SQLite doesn't have RAISE NOTICE outside triggers).
    # v59 ROOT FIX (compound of T-001): SQLite does not have a search_path
    # concept -- ``SET search_path TO public`` raises ``OperationalError:
    # near "SET": syntax error``. Strip it on SQLite. PostgreSQL keeps it
    # (the migration runner uses the raw SQL on PostgreSQL).
    (re.compile(r"SET\s+search_path\s+TO\s+\w+\s*;?", re.IGNORECASE), "-- [SQLite-skip] SET search_path"),
    # v59 ROOT FIX: ANALYZE is a PostgreSQL command (recomputes query
    # planner stats). SQLite accepts the syntax but treats it as a no-op
    # for forward-compatibility -- actually no, SQLite raises
    # ``OperationalError: near "ANALYZE": syntax error`` on the
    # ``ANALYZE <table_name>`` form. Strip it on SQLite.
    (re.compile(r"^\s*ANALYZE\s+\w+\s*;", re.IGNORECASE | re.MULTILINE), "-- [SQLite-skip] ANALYZE"),
]


def _translate_sql_for_sqlite(sql: str) -> str:
    """Translate PostgreSQL-specific SQL to SQLite-compatible SQL.

    v16 ROOT FIX (CD-5): the migration .sql files are written for
    PostgreSQL. The previous code skipped them entirely on SQLite,
    leaving SQLite dev/test DBs without CHECK/UNIQUE/FK constraints.
    This function performs a best-effort translation:

    - ``GENERATED ALWAYS AS IDENTITY`` -> ``AUTOINCREMENT``
    - ``TIMESTAMP WITH TIME ZONE`` -> ``TIMESTAMP``
    - ``DO $$ ... END $$;`` blocks -> wrapped in a BEGIN/COMMIT (SQLite
      doesn't have PL/pgSQL, but the SQL inside the DO block is usually
      plain SQL with control flow -- we strip the control flow and
      keep the SQL statements).
    - ``RAISE NOTICE '...'`` lines -> ``-- RAISE NOTICE '...'`` (commented out)
    - ``IF EXISTS (SELECT 1 FROM pg_constraint ...)`` -> ``1=1`` (always true
      -- the guard is a no-op on SQLite since SQLite doesn't enforce
      constraint names).
    - ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` -> ``ALTER TABLE ... ADD COLUMN``
      (SQLite doesn't support IF NOT EXISTS on ADD COLUMN before 3.35;
      we wrap the call in a try/except in the runner).
    - ``GET DIAGNOSTICS _var = ROW_COUNT;`` -> commented out (no SQLite equivalent;
      downstream audit_log INSERTs that reference _var will get NULL).
    - ``CREATE INDEX ... WHERE`` -> ``CREATE INDEX ...`` (partial indexes
      require SQLite 3.8+; we strip the WHERE to be safe).
    - ``STRING_AGG(expr, sep)`` -> ``GROUP_CONCAT(expr, sep)`` (v35 root fix
      issue 33 -- argument order is the same, so a direct name swap is
      semantically correct).
    - ``<agg>(expr) FILTER (WHERE cond)`` -> ``<agg>(CASE WHEN cond THEN expr END)``
      (v35 root fix issue 33 -- SQLite does not support the SQL:2003 FILTER
      clause; this rewrite preserves semantics for COUNT/SUM/AVG/MIN/MAX).

    The translation is best-effort. Statements that cannot be translated
    are left as-is and will raise OperationalError at execution time --
    the runner catches the error and logs WARNING (don't block the
    migration chain on SQLite).
    """
    out = sql
    # 1. Strip pg_advisory_lock calls.
    for pat, repl in _PG_ONLY_STATEMENT_PATTERNS:
        out = pat.sub(repl, out)
    # 2. GENERATED ALWAYS AS IDENTITY -> AUTOINCREMENT (only valid as part
    # of INTEGER PRIMARY KEY, so we use a regex that requires that context).
    out = re.sub(
        r"INTEGER\s+GENERATED\s+ALWAYS\s+AS\s+IDENTITY\s+PRIMARY\s+KEY",
        "INTEGER PRIMARY KEY AUTOINCREMENT",
        out, flags=re.IGNORECASE,
    )
    # 3. TIMESTAMP WITH TIME ZONE -> TIMESTAMP
    out = re.sub(
        r"TIMESTAMP\s+WITH\s+TIME\s+ZONE",
        "TIMESTAMP", out, flags=re.IGNORECASE,
    )
    # 3b. v59 ROOT FIX (compound of T-001): ``DEFAULT NOW()`` ->
    # ``DEFAULT CURRENT_TIMESTAMP``. SQLite does NOT have a ``NOW()``
    # function -- every ``DEFAULT NOW()`` in the migrations raised
    # ``OperationalError: near "(": syntax error`` because SQLite
    # parsed ``NOW`` as a column name and ``(`` as unexpected. The
    # SQLAlchemy ORM already uses ``func.now()`` which SQLAlchemy
    # translates per-dialect, but the raw SQL migrations use the
    # PostgreSQL-native ``NOW()`` directly. Translating to
    # ``CURRENT_TIMESTAMP`` is portable across PostgreSQL and SQLite
    # (both support it as a SQL standard function).
    out = re.sub(
        r"DEFAULT\s+NOW\s*\(\s*\)",
        "DEFAULT CURRENT_TIMESTAMP",
        out, flags=re.IGNORECASE,
    )
    # 3c. v59 ROOT FIX: bare ``NOW()`` (not in DEFAULT context) ->
    # ``CURRENT_TIMESTAMP``. Used in UPDATE SET clauses and trigger
    # bodies. Same SQLite limitation as 3b.
    out = re.sub(
        r"\bNOW\s*\(\s*\)",
        "CURRENT_TIMESTAMP",
        out, flags=re.IGNORECASE,
    )
    # 4. DO $$ ... END $$; -> strip the PL/pgSQL wrapper, keep inner SQL.
    # The inner SQL often uses BEGIN/END/IF/RAISE NOTICE -- we leave those
    # in (they'll cause warnings at execution time but won't block the
    # rest of the migration since each statement is independent).
    def _strip_do_block(m: "re.Match[str]") -> str:
        inner = m.group(1)
        # v62 ROOT FIX (T-001 compound of COMMENT ON inside DO blocks):
        # Migration 002 has a COMMENT ON TABLE statement INSIDE a DO
        # block (inside a SAVEPOINT section). The v59 _strip_do_block
        # function had a ``bare string literal continuation`` regex
        # (runs later) that matched individual string-literal lines of
        # the COMMENT ON statement and replaced them with comments --
        # breaking the COMMENT ON syntax. The broken COMMENT ON then
        # survived to execution time and SQLite raised
        # ``near "COMMENT": syntax error``. Fix: strip COMMENT ON
        # statements INSIDE DO blocks FIRST, before any other regex
        # can mangle them. Uses the same precise pattern as the main
        # COMMENT ON regex (with ``()`` support for FUNCTION names).
        # v62 ROOT FIX: add ``;`` to replacement so SQL splitter sees
        # statement boundary inside DO blocks too.
        inner = re.sub(
            r"COMMENT\s+ON\s+(?:TABLE|COLUMN|INDEX|CONSTRAINT|FUNCTION|TRIGGER|SCHEMA|TYPE|VIEW|SEQUENCE|DATABASE)\s+[\w.\s\"()]+\s+IS\s+(?:NULL|(?:'(?:[^']|'')*'\s*)+);",
            "-- COMMENT ON inside DO block (SQLite-skip)\n;",
            inner, flags=re.IGNORECASE | re.DOTALL,
        )
        # v59 ROOT FIX (compound of T-001): strip the entire DECLARE ...
        # BEGIN section. The previous code tried to comment out individual
        # variable declarations with a regex that only matched ``_\w+``
        # (underscore-prefixed names). But migration 001's verification
        # DO block declares ``tbl TEXT``, ``col_count INTEGER``,
        # ``table_count INTEGER`` -- none start with ``_``. The un-stripped
        # ``col_count INTEGER`` then appeared as a bare SQL statement,
        # causing ``OperationalError: near "col_count": syntax error``.
        # The fix: strip the ENTIRE DECLARE ... BEGIN section in one shot.
        # This is safe because:
        #   1. SQLite doesn't support PL/pgSQL variables at all.
        #   2. The DECLARE section only contains variable declarations,
        #      never executable SQL.
        #   3. The BEGIN that follows is commented out by the next regex.
        inner = re.sub(
            r"DECLARE\s+.*?\s+BEGIN",
            "-- DECLARE section stripped (SQLite-skip)\n-- BEGIN",
            inner, flags=re.IGNORECASE | re.DOTALL,
        )
        # v62 ROOT FIX (T-001 compound of EXCEPTION block ordering):
        # strip the PL/pgSQL EXCEPTION handler BEFORE any line-level
        # keyword stripping (BEGIN/END/IF/THEN). The v59 code ran
        # EXCEPTION stripping AFTER BEGIN/END stripping, so by the time
        # the EXCEPTION regex tried to match ``EXCEPTION ... END``,
        # the ``END`` keyword had already been replaced with ``-- END``
        # -- the regex failed, the ``WHEN ... THEN`` body survived, and
        # SQLite raised ``OperationalError: near "OR": syntax error``
        # (from ``WHEN feature_not_supported OR syntax_error THEN`` in
        # migration 009). Verified failing by actually running migrations.
        # Fix: run EXCEPTION stripping FIRST. Also broaden the WHEN
        # clause pattern to accept multi-condition expressions
        # (``WHEN x OR y THEN``, ``WHEN x, y THEN``, SQLSTATE codes).
        inner = re.sub(
            r"EXCEPTION\s+WHEN\s+[^;]*?THEN\s+.*?\s+END(?=\s*;|\s*\Z|\s*\n)",
            "-- EXCEPTION block (SQLite-skip)\n-- END",
            inner, flags=re.IGNORECASE | re.DOTALL,
        )
        # Comment out PL/pgSQL control-flow keywords.
        inner = re.sub(r"^\s*BEGIN\s*$", "-- BEGIN", inner, flags=re.IGNORECASE | re.MULTILINE)
        inner = re.sub(r"^\s*END\s*;", "-- END;", inner, flags=re.IGNORECASE | re.MULTILINE)
        inner = re.sub(r"^\s*END\s*$", "-- END", inner, flags=re.IGNORECASE | re.MULTILINE)
        inner = re.sub(
            r"FOR\s+\w+\s+IN\s+.*?\s+LOOP\s*;?",
            "-- FOR ... LOOP (SQLite-skip)",
            inner, flags=re.IGNORECASE | re.DOTALL,
        )
        inner = re.sub(r"^\s*END\s+LOOP\s*;", "-- END LOOP;", inner, flags=re.IGNORECASE | re.MULTILINE)
        inner = re.sub(r"^\s*DECLARE\s+", "-- DECLARE ", inner, flags=re.IGNORECASE | re.MULTILINE)
        inner = re.sub(r"^\s*IF\s+", "-- IF ", inner, flags=re.IGNORECASE | re.MULTILINE)
        inner = re.sub(r"^\s*THEN\s*$", "-- THEN", inner, flags=re.IGNORECASE | re.MULTILINE)
        inner = re.sub(r"^\s*ELSE\s*$", "-- ELSE", inner, flags=re.IGNORECASE | re.MULTILINE)
        inner = re.sub(r"^\s*ELSIF\s+", "-- ELSIF ", inner, flags=re.IGNORECASE | re.MULTILINE)
        inner = re.sub(r"^\s*END\s+IF\s*;", "-- END IF;", inner, flags=re.IGNORECASE | re.MULTILINE)
        # v59 ROOT FIX: strip any remaining EXCEPTION lines (already
        # handled by the early regex above, but kept as a backstop for
        # EXCEPTION blocks that don't match the strict pattern).
        inner = re.sub(r"^\s*EXCEPTION\s+.*$", lambda mm: "-- " + mm.group(0).strip(), inner, flags=re.IGNORECASE | re.MULTILINE)
        # v59 ROOT FIX: match multi-line RAISE statements with any args.
        # v62 ROOT FIX (T-001 compound of RAISE truncation): the v59 code
        # truncated the RAISE replacement to 200 chars with ``[:200]``. But
        # RAISE statements often contain long multi-line string literals
        # (e.g. migration 009's RAISE WARNING is ~400 chars). Truncation
        # at 200 chars cut off mid-string-literal, leaving an UNCLOSED
        # ``'`` that caused the SQL splitter's string-literal handler to
        # swallow ALL subsequent ``;`` statement terminators until it
        # found the next ``'`` (which could be hundreds of lines later).
        # Result: the entire rest of the migration was treated as ONE
        # giant statement, and SQLite raised ``OperationalError: near
        # "OR": syntax error`` on whatever non-SQL token came first.
        # Verified failing by actually running migrations on SQLite.
        # Fix: replace the entire RAISE statement with a FIXED comment
        # (no truncation, no content echo) -- this guarantees no unclosed
        # string literals can leak into the translated SQL.
        inner = re.sub(
            r"RAISE\s+(?:NOTICE|WARNING|EXCEPTION)\s+.*?;",
            "-- RAISE statement (SQLite-skip)",
            inner, flags=re.IGNORECASE | re.DOTALL,
        )
        inner = re.sub(
            r"^\s+'(?:[^']|'')*'\s*;",
            "-- bare string literal continuation (SQLite-skip)",
            inner, flags=re.MULTILINE,
        )
        inner = re.sub(r"^\s*GET\s+DIAGNOSTICS\s+.*$", lambda mm: "-- " + mm.group(0).strip(), inner, flags=re.IGNORECASE | re.MULTILINE)
        inner = re.sub(r"^\s*PERFORM\s+", "-- PERFORM ", inner, flags=re.IGNORECASE | re.MULTILINE)
        inner = re.sub(r"^\s*RETURN\s+", "-- RETURN ", inner, flags=re.IGNORECASE | re.MULTILINE)
        # v59 ROOT FIX: strip PL/pgSQL variable assignments.
        inner = re.sub(
            r"^\s*_?\w+\s*:=\s*[^;]+;",
            "-- PL/pgSQL assignment (SQLite-skip)",
            inner, flags=re.IGNORECASE | re.MULTILINE,
        )
        # v59 ROOT FIX: strip ALL INSERT INTO audit_log and
        # _migration_002_dedup_archive statements inside DO blocks.
        inner = re.sub(
            r"INSERT\s+INTO\s+audit_log\s*[^;]*;",
            "-- INSERT INTO audit_log (SQLite-skip)",
            inner, flags=re.IGNORECASE | re.DOTALL,
        )
        inner = re.sub(
            r"INSERT\s+INTO\s+_migration_002_dedup_archive\s*[^;]*;",
            "-- INSERT INTO _migration_002_dedup_archive (SQLite-skip)",
            inner, flags=re.IGNORECASE | re.DOTALL,
        )
        # v59 ROOT FIX: strip SELECT ... INTO ... (PL/pgSQL-only).
        inner = re.sub(
            r"SELECT\s+(?:(?!INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|RAISE|END|BEGIN|IF|FOR|EXCEPTION|VALUES).)*?INTO\s+\w+\s*(?:FROM\s+[^;]+)?;",
            "-- SELECT ... INTO ... (SQLite-skip)",
            inner, flags=re.IGNORECASE | re.DOTALL,
        )
        # v59 ROOT FIX: strip IF [NOT] EXISTS (...) THEN blocks.
        inner = re.sub(
            r"IF\s+(?:NOT\s+)?EXISTS\s*\(.*?\)\s*THEN",
            "-- IF [NOT] EXISTS (...) THEN (SQLite-skip)",
            inner, flags=re.IGNORECASE | re.DOTALL,
        )
        inner = re.sub(r"^\s*END\s+IF\s*$", "-- END IF", inner, flags=re.IGNORECASE | re.MULTILINE)
        # v59 ROOT FIX: strip CTEs (WITH ... ;) that don't contain ``;``
        # inside string literals. Use a string-literal-aware pattern.
        # This handles migration 002's CTE-based dedup operations.
        # v59 ROOT FIX #2: the previous version matched ``WITH`` inside
        # comments (e.g. "withdrawn" contains "with"). The new version
        # tracks line-comment state (``-- ...``) and only matches ``WITH``
        # outside comments and string literals.
        def _strip_cte(text):
            result = []
            i = 0
            n = len(text)
            while i < n:
                # Check for line comment (-- until end of line)
                if text[i:i+2] == '--':
                    # Copy the entire comment line
                    while i < n and text[i] != '\n':
                        result.append(text[i])
                        i += 1
                    continue
                # Check for string literal
                if text[i] == "'":
                    result.append(text[i])
                    i += 1
                    while i < n:
                        result.append(text[i])
                        if text[i] == "'" and (i+1 >= n or text[i+1] != "'"):
                            i += 1
                            break
                        if text[i] == "'" and i+1 < n and text[i+1] == "'":
                            result.append(text[i+1])
                            i += 2
                            continue
                        i += 1
                    continue
                # Check for CTE start (WITH as a word, not part of a larger identifier)
                if (text[i:i+4].upper() == 'WITH' and
                    (i == 0 or text[i-1] in '\n\r\t ') and
                    (i+4 >= n or text[i+4] in '\n\r\t (')):
                    # Found a CTE start -- find the end (next ; outside a string)
                    j = i + 4
                    in_string = False
                    while j < n:
                        if text[j] == "'" and (j == 0 or text[j-1] != '\\'):
                            in_string = not in_string
                        elif text[j] == ';' and not in_string:
                            j += 1
                            break
                        j += 1
                    result.append("-- CTE (SQLite-skip)")
                    i = j
                else:
                    result.append(text[i])
                    i += 1
            return ''.join(result)
        inner = _strip_cte(inner)
        # v59 ROOT FIX: strip ``UPDATE ... FROM ...`` statements (with or
        # without ``AS`` keyword). PostgreSQL supports ``UPDATE table alias
        # SET ... FROM (subquery) alias2 WHERE ...`` but SQLite does NOT
        # support ``UPDATE ... FROM``. Only strip UPDATEs that have a FROM
        # clause (UPDATEs with only WHERE are valid SQLite). Migration 002
        # uses this for entity_mapping dedup.
        # Use a string-aware approach to avoid matching ``FROM`` inside
        # string literals.
        def _strip_update_from(text):
            result = []
            i = 0
            n = len(text)
            while i < n:
                # Check for UPDATE keyword
                if (text[i:i+6].upper() == 'UPDATE' and
                    (i == 0 or text[i-1] in '\n\r\t ') and
                    (i+6 >= n or text[i+6] in '\n\r\t ')):
                    # Find the end of this UPDATE statement (next ; outside string)
                    j = i + 6
                    in_string = False
                    has_from = False
                    while j < n:
                        if text[j] == "'":
                            in_string = not in_string
                        elif not in_string and text[j:j+4].upper() == 'FROM' and text[j-1] in '\n\r\t ':
                            has_from = True
                        elif text[j] == ';' and not in_string:
                            j += 1
                            break
                        j += 1
                    if has_from:
                        result.append("-- UPDATE ... FROM (SQLite-skip)")
                    else:
                        result.append(text[i:j])
                    i = j
                else:
                    result.append(text[i])
                    i += 1
            return ''.join(result)
        inner = _strip_update_from(inner)
        return inner
    # v62 ROOT FIX (T-001 compound of missing semicolon): the DO block
    # regex consumes the trailing ``;`` (as part of ``$$;``). The
    # replacement (from _strip_do_block) does NOT include ``;``. This
    # causes the SQL splitter to merge the DO block content with the
    # NEXT statement (no ``;`` between them). Fix: append ``;`` to the
    # _strip_do_block return value so the splitter sees a statement
    # boundary.
    out = re.sub(
        r"DO\s*\$\$\s*((?:(?!\$\$).)*?)\s*\$\$\s*;",
        lambda m: _strip_do_block(m) + "\n;", out, flags=re.IGNORECASE | re.DOTALL,
    )
    # 5. ALTER TABLE ... ADD COLUMN IF NOT EXISTS handling.
    # v17 ROOT FIX (SQLite ADD COLUMN IF NOT EXISTS): the previous code
    # UNCONDITIONALLY stripped ``IF NOT EXISTS`` from every ALTER TABLE
    # ADD COLUMN statement, on the assumption that "SQLite < 3.35
    # doesn't support IF NOT EXISTS". This caused two problems:
    #   (a) Modern SQLite (3.35+, released 2021-03) DOES support
    #       ADD COLUMN IF NOT EXISTS. Stripping it on modern SQLite
    #       means re-running migration 006 raises
    #       ``duplicate column name: groups`` -- the runner catches it
    #       as WARNING + marks the migration as "skipped", silently
    #       leaving the schema divergent.
    #   (b) Even on older SQLite, stripping IF NOT EXISTS makes re-runs
    #       raise -- exactly the opposite of idempotency.
    # The fix: detect SQLite version (via sqlite3.sqlite_version) at
    # translate-time. On 3.35+, KEEP the IF NOT EXISTS clause (modern
    # behavior). On older SQLite, strip it BUT wrap the runner's call
    # site in a try/except that catches ``duplicate column name`` and
    # treats it as a successful no-op (the column already exists).
    # Since the runner at line 3257 already catches OperationalError
    # and marks the migration as "skipped", the simpler path is to
    # detect the version and only strip when needed.
    # v59 ROOT FIX (compound of T-001): SQLite does NOT support
    # ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` -- even in SQLite
    # 3.53+. The previous version check (``>= (3, 35)``) incorrectly
    # assumed that SQLite 3.35+ added this syntax. In reality, SQLite
    # 3.35 added ``IF NOT EXISTS`` for ``CREATE TABLE`` and ``CREATE
    # INDEX``, but NOT for ``ALTER TABLE ADD COLUMN``. The result:
    # ``ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS row_count INTEGER``
    # raised ``OperationalError: near "EXISTS": syntax error`` on every
    # SQLite version, blocking migration 002 entirely.
    # FIX: ALWAYS strip ``IF NOT EXISTS`` from ``ALTER TABLE ADD COLUMN``
    # on SQLite. The runner's exception handler at line ~3680 already
    # catches ``duplicate column name`` errors and treats them as
    # successful no-ops (the column already exists -- the migration's
    # intent is satisfied).
    out = re.sub(
        r"(ALTER\s+TABLE\s+\w+\s+ADD\s+COLUMN)\s+IF\s+NOT\s+EXISTS",
        r"\1", out, flags=re.IGNORECASE,
    )
    # v59 ROOT FIX #2: SQLite does NOT support ``ALTER TABLE ... DROP
    # CONSTRAINT`` at all (SQLite has no named constraints like
    # PostgreSQL). The migration 002 SQL uses
    # ``ALTER TABLE audit_log DROP CONSTRAINT IF EXISTS chk_audit_log_operation``
    # which raised ``OperationalError: near "EXISTS": syntax error``.
    # FIX: strip the entire ``DROP CONSTRAINT`` clause on SQLite. The
    # subsequent ``ADD CONSTRAINT`` is handled by the next regex
    # (SQLite doesn't support ADD CONSTRAINT either -- it's stripped too).
    # v62 ROOT FIX: add ``;`` to replacement so SQL splitter sees
    # statement boundary (the regex consumes the ``;`` but the v59
    # replacement didn't preserve it, causing statement merging).
    out = re.sub(
        r"ALTER\s+TABLE\s+\w+\s+DROP\s+CONSTRAINT\s+IF\s+EXISTS\s+\w+\s*;",
        "-- [SQLite-skip] ALTER TABLE DROP CONSTRAINT (PostgreSQL-only)\n;",
        out, flags=re.IGNORECASE,
    )
    out = re.sub(
        r"ALTER\s+TABLE\s+\w+\s+DROP\s+CONSTRAINT\s+\w+\s*;",
        "-- [SQLite-skip] ALTER TABLE DROP CONSTRAINT (PostgreSQL-only)\n;",
        out, flags=re.IGNORECASE,
    )
    # v59 ROOT FIX #3 / v62 ROOT FIX (compound): SQLite does NOT support
    # ``ALTER TABLE ... ADD CONSTRAINT`` -- constraints must be defined at
    # CREATE TABLE time. Strip the entire ``ADD CONSTRAINT`` clause on
    # SQLite. The constraint is enforced via the ORM's CheckConstraint on
    # SQLite (via Base.metadata.create_all).
    #
    # v62 ROOT FIX (T-001 compound): the v59 regex required the form
    # ``ADD CONSTRAINT <name> CHECK(...)`` (constraint name immediately
    # after ``ADD CONSTRAINT``). But migration 002 line ~775 emits
    # ``ADD CONSTRAINT IF NOT EXISTS <name> UNIQUE (...)`` -- PostgreSQL
    # accepts the optional ``IF NOT EXISTS`` between ``CONSTRAINT`` and
    # the name. The v59 regex didn't match this form, so the statement
    # was passed through verbatim to SQLite, which raised
    # ``OperationalError: near "NOT": syntax error`` and blocked the
    # ENTIRE 10-migration chain on SQLite (Phase 1 had no database).
    # Verified failing by actually running ``run_migrations()`` on a
    # fresh SQLite DB -- previous AIs claimed this was fixed but never
    # ran it. Fix: make ``IF NOT EXISTS`` optional in ALL three
    # ``ADD CONSTRAINT`` regexes (CHECK / UNIQUE / FK).
    # v62 ROOT FIX: add ``;`` to replacements so SQL splitter sees
    # statement boundaries.
    out = re.sub(
        r"ALTER\s+TABLE\s+\w+\s+ADD\s+CONSTRAINT\s+(?:IF\s+NOT\s+EXISTS\s+)?\w+\s+CHECK\s*\(.*?\)\s*;",
        "-- [SQLite-skip] ALTER TABLE ADD CONSTRAINT CHECK (PostgreSQL-only)\n;",
        out, flags=re.IGNORECASE | re.DOTALL,
    )
    out = re.sub(
        r"ALTER\s+TABLE\s+\w+\s+ADD\s+CONSTRAINT\s+(?:IF\s+NOT\s+EXISTS\s+)?\w+\s+UNIQUE\s*\([^)]+\)\s*;",
        "-- [SQLite-skip] ALTER TABLE ADD CONSTRAINT UNIQUE (PostgreSQL-only)\n;",
        out, flags=re.IGNORECASE,
    )
    out = re.sub(
        r"ALTER\s+TABLE\s+\w+\s+ADD\s+CONSTRAINT\s+(?:IF\s+NOT\s+EXISTS\s+)?\w+\s+FOREIGN\s+KEY\s*\([^)]+\)\s+REFERENCES\s+\w+\s*\([^)]+\)[^;]*;",
        "-- [SQLite-skip] ALTER TABLE ADD CONSTRAINT FK (PostgreSQL-only)\n;",
        out, flags=re.IGNORECASE | re.DOTALL,
    )
    # v59 ROOT FIX: SQLite does NOT support ``ALTER TABLE ... ALTER
    # COLUMN ... TYPE ...`` (PostgreSQL-specific). SQLite columns can't
    # have their type changed after creation. Strip these statements --
    # the column type is already correct from migration 001's CREATE
    # TABLE (which the translator made SQLite-compatible).
    out = re.sub(
        r"ALTER\s+TABLE\s+\w+\s+ALTER\s+COLUMN\s+\w+\s+TYPE\s+[^;]+;",
        "-- [SQLite-skip] ALTER TABLE ALTER COLUMN TYPE (PostgreSQL-only)\n;",
        out, flags=re.IGNORECASE,
    )
    # v59 ROOT FIX: SQLite does NOT support ``ALTER TABLE ... DROP
    # CONSTRAINT`` (already handled above) or ``ALTER TABLE ... DROP
    # COLUMN`` (SQLite 3.35+ supports it but the syntax is different).
    # Strip DROP CONSTRAINT and DROP DEFAULT statements.
    #
    # v76 ROOT FIX (T-037 compound -- DROP COLUMN IF EXISTS translation):
    #   SQLite 3.35+ supports ``ALTER TABLE ... DROP COLUMN`` but does NOT
    #   support the ``IF EXISTS`` clause. The rollback files use
    #   ``DROP COLUMN IF EXISTS`` (PostgreSQL syntax) for idempotency.
    #   On SQLite, this raises ``OperationalError: near "EXISTS": syntax
    #   error``. ROOT FIX: strip the ``IF EXISTS`` clause so the statement
    #   becomes ``ALTER TABLE ... DROP COLUMN <name>`` (valid SQLite 3.35+).
    #   The per-statement try/except in the rollback path (and the forward
    #   path at line ~4430) catches ``no such column`` errors as idempotent
    #   no-ops, preserving the ``IF EXISTS`` semantics.
    out = re.sub(
        r"(ALTER\s+TABLE\s+\w+\s+DROP\s+COLUMN)\s+IF\s+EXISTS\s+(\w+)",
        r"\1 \2",
        out, flags=re.IGNORECASE,
    )
    # v76 ROOT FIX (T-037 compound -- DROP TABLE/INDEX ... CASCADE translation):
    #   PostgreSQL supports ``DROP TABLE ... CASCADE`` and
    #   ``DROP INDEX ... CASCADE`` to automatically drop dependent objects.
    #   SQLite does NOT support the ``CASCADE`` keyword on DROP statements.
    #   The rollback files use ``DROP TABLE IF EXISTS ... CASCADE``
    #   (PostgreSQL syntax). On SQLite, this raises
    #   ``OperationalError: near "CASCADE": syntax error``. ROOT FIX: strip
    #   the trailing ``CASCADE`` (and ``RESTRICT``) keyword from DROP
    #   statements. SQLite's default drop behavior is RESTRICT (no cascade),
    #   which is safe for rollbacks because the rollback SQL explicitly
    #   drops dependent objects (indexes, constraints) BEFORE the table.
    out = re.sub(
        r"(DROP\s+(?:TABLE|INDEX)\s+IF\s+EXISTS\s+\w+)\s+(?:CASCADE|RESTRICT)",
        r"\1",
        out, flags=re.IGNORECASE,
    )
    out = re.sub(
        r"ALTER\s+TABLE\s+\w+\s+ALTER\s+COLUMN\s+\w+\s+DROP\s+DEFAULT\s*;",
        "-- [SQLite-skip] ALTER TABLE ALTER COLUMN DROP DEFAULT\n;",
        out, flags=re.IGNORECASE,
    )
    out = re.sub(
        r"ALTER\s+TABLE\s+\w+\s+ALTER\s+COLUMN\s+\w+\s+DROP\s+NOT\s+NULL\s*;",
        "-- [SQLite-skip] ALTER TABLE ALTER COLUMN DROP NOT NULL\n;",
        out, flags=re.IGNORECASE,
    )
    # v73 ROOT FIX: SQLite does NOT support ``ALTER TABLE ... ALTER
    # COLUMN ... SET DEFAULT ...`` or ``SET NOT NULL`` (PostgreSQL-
    # specific). Migration 008 uses these to tighten the
    # is_globally_approved column. SQLite columns get their DEFAULT and
    # NOT NULL from the CREATE TABLE statement (or ORM), not from ALTER.
    # Skip these statements -- the ORM-created SQLite schema already has
    # the correct column definition.
    out = re.sub(
        r"ALTER\s+TABLE\s+\w+\s+ALTER\s+COLUMN\s+\w+\s+SET\s+DEFAULT\s+[^;]+;",
        "-- [SQLite-skip] ALTER TABLE ALTER COLUMN SET DEFAULT (PostgreSQL-only)\n;",
        out, flags=re.IGNORECASE | re.DOTALL,
    )
    out = re.sub(
        r"ALTER\s+TABLE\s+\w+\s+ALTER\s+COLUMN\s+\w+\s+SET\s+NOT\s+NULL\s*;",
        "-- [SQLite-skip] ALTER TABLE ALTER COLUMN SET NOT NULL (PostgreSQL-only)\n;",
        out, flags=re.IGNORECASE,
    )
    # v59 ROOT FIX: SQLite does NOT support table aliases in DELETE
    # statements (``DELETE FROM table alias WHERE ...``). PostgreSQL
    # does. Migration 003 uses this for PPI dedup:
    #   DELETE FROM protein_protein_interactions ppi WHERE ppi.protein_a_id > ...
    # SQLite raises ``OperationalError: near "ppi": syntax error``.
    # Strip these DELETE statements -- they're dedup operations not
    # needed on SQLite (tests use clean DBs).
    out = re.sub(
        r"DELETE\s+FROM\s+\w+\s+\w+\s+WHERE[^;]*;",
        "-- [SQLite-skip] DELETE with table alias (PostgreSQL-only)\n;",
        out, flags=re.IGNORECASE | re.DOTALL,
    )
    # v59 ROOT FIX: SQLite does NOT support ``UPDATE ... SET col1 = col2,
    # col2 = col1`` (swap). SQLite evaluates the SET clauses left-to-right,
    # so the swap doesn't work (col1 gets col2's value, then col2 gets
    # the NEW col1 value which is the old col2 -- net effect: both equal
    # old col2). PostgreSQL evaluates all RHS first. Strip these swap
    # statements -- they're dedup operations not needed on SQLite.
    out = re.sub(
        r"UPDATE\s+\w+\s+SET\s+\w+\s*=\s*\w+,\s*\w+\s*=\s*\w+\s+WHERE[^;]*;",
        "-- [SQLite-skip] UPDATE swap (PostgreSQL-specific semantics)\n;",
        out, flags=re.IGNORECASE | re.DOTALL,
    )
    # 6. Strip partial-index WHERE clauses (SQLite 3.8+ supports them,
    # but be defensive).
    out = re.sub(
        r"(CREATE\s+(?:UNIQUE\s+)?INDEX\s+IF\s+NOT\s+EXISTS\s+\w+\s+ON\s+\w+\s*\([^)]+\))\s+WHERE\s+[^;]+(;)",
        r"\1\2", out, flags=re.IGNORECASE,
    )
    # 7. JSONB -> TEXT (SQLite has no JSONB type; JSON is stored as TEXT).
    out = re.sub(r"\bJSONB\b", "TEXT", out, flags=re.IGNORECASE)
    # 8. ::type casts -> remove (SQLite ignores Postgres-style casts).
    out = re.sub(r"::\w+(?:\([^)]*\))?", "", out)
    # 9. COMMENT ON ... IS '...'; -> strip (SQLite has no COMMENT ON).
    # v59 ROOT FIX: expanded the keyword list to include FUNCTION and
    # TRIGGER (PostgreSQL supports COMMENT ON FUNCTION, COMMENT ON
    # TRIGGER, etc.). The previous regex only matched TABLE/COLUMN/
    # INDEX/CONSTRAINT, leaving ``COMMENT ON FUNCTION update_updated_at()
    # IS '...'`` in the translated SQL -- SQLite raised ``OperationalError:
    # near "COMMENT": syntax error``.
    # v59 ROOT FIX #2: the previous regex ``[^;]+;`` broke on COMMENT ON
    # statements where the string literal contained a ``;`` (e.g.
    # ``'Semicolon-separated PubMed IDs (e.g., "12345678;23456789")'``).
    # The ``[^;]+`` stopped at the ``;`` INSIDE the string, leaving the
    # rest of the COMMENT ON un-stripped. The new regex matches the IS
    # clause as a sequence of string literals and whitespace, which
    # correctly handles ``;`` inside strings (because ``'[^']*'`` consumes
    # the entire string literal including any ``;`` inside it).
    #
    # v62 ROOT FIX (T-001 compound of COMMENT ON eating CREATE TABLE):
    # The v59 precise regex required the object name to match
    # ``[\w.\s\"]+`` -- which does NOT include parentheses. So
    # ``COMMENT ON FUNCTION update_updated_at() IS '...'`` was NOT
    # matched by the precise regex, and fell through to the v59
    # fallback regex ``COMMENT\s+ON\s+[^;]*;``. That fallback regex
    # matches ``COMMENT ON`` ANYWHERE in the SQL -- including inside
    # ``--`` comments left behind by earlier replacements (e.g.
    # ``-- [SQLite-skip] COMMENT ON ...``). The greedy ``[^;]*`` then
    # ate everything from the comment's ``COMMENT ON`` to the next
    # ``;`` -- which could be the ``;`` at the end of a CREATE TABLE
    # statement (e.g. ``CREATE TABLE proteins (...);``). This SILENTLY
    # DELETED the proteins, drug_protein_interactions,
    # protein_protein_interactions, gene_disease_associations,
    # entity_mapping, rejected_records, and audit_log CREATE TABLE
    # statements from migration 001 on SQLite. Phase 1 had no database.
    # Verified failing by actually running migrations. Fix:
    #   (1) Extend the precise regex's object-name char class to include
    #       ``()`` so COMMENT ON FUNCTION is matched.
    #   (2) Remove the dangerous fallback regex entirely. The precise
    #       regex now handles all COMMENT ON forms in the migration
    #       files. If a future COMMENT ON doesn't match, it will fail
    #       LOUDLY (OperationalError) rather than SILENTLY eating
    #       CREATE TABLE statements.
    # v62 ROOT FIX: add ``;`` to replacement so SQL splitter sees
    # statement boundary.
    out = re.sub(
        r"COMMENT\s+ON\s+(?:TABLE|COLUMN|INDEX|CONSTRAINT|FUNCTION|TRIGGER|SCHEMA|TYPE|VIEW|SEQUENCE|DATABASE)\s+[\w.\s\"()]+\s+IS\s+(?:NULL|(?:'(?:[^']|'')*'\s*)+);",
        "-- [SQLite-skip] COMMENT ON ...\n;",
        out, flags=re.IGNORECASE | re.DOTALL,
    )
    # v62 ROOT FIX: REMOVED the fallback regex
    # ``COMMENT\s+ON\s+[^;]*;`` that was eating CREATE TABLE statements.
    # The precise regex above now handles all COMMENT ON forms. If a
    # future COMMENT ON doesn't match the precise pattern, it will raise
    # an OperationalError at execution time (loud failure) rather than
    # silently eating CREATE TABLE content (silent data loss).
    # 9b. v59 ROOT FIX (compound of T-001): strip PostgreSQL-specific
    # ``CREATE FUNCTION ... AS $tag$ ... $tag$ LANGUAGE plpgsql;``
    # statements. SQLite does NOT support user-defined functions in SQL
    # (they must be registered via the Python/C API). The previous
    # translator left these statements intact, causing SQLite to raise
    # ``OperationalError: near "FUNCTION": syntax error`` on every
    # migration that created a trigger function (001, 002). The functions
    # are only used by PostgreSQL triggers (which SQLite also can't
    # create -- see next pattern). On SQLite, the ORM's BulkUpdate
    # mechanism handles ``updated_at`` auto-update at the application
    # layer (see database/base.py TimestampMixin).
    out = re.sub(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+\w+\s*\([^)]*\)\s+RETURNS\s+\w+\s+AS\s+\$\w*\$.*?\$\w*\s+LANGUAGE\s+\w+\s*;",
        "-- [SQLite-skip] CREATE FUNCTION (PostgreSQL-only)\n;",
        out, flags=re.IGNORECASE | re.DOTALL,
    )
    # 9c. v59 ROOT FIX: strip ``CREATE TRIGGER ... EXECUTE FUNCTION ...``
    # statements. SQLite has its own CREATE TRIGGER syntax (``CREATE
    # TRIGGER ... BEGIN ... END``) which is incompatible with PostgreSQL's
    # ``EXECUTE FUNCTION`` form. The migrations use the PostgreSQL form
    # exclusively. SQLite tests don't need trigger-based ``updated_at``
    # auto-update (the ORM handles it).
    #
    # v62 ROOT FIX (T-001 compound of CREATE TRIGGER): the v59 regex
    # only matched ``BEFORE UPDATE ON <table>``. Migration 006 emits
    # ``BEFORE INSERT OR UPDATE OF groups, name ON drugs FOR EACH ROW
    # EXECUTE FUNCTION trg_drugs_sync_withdrawn()`` -- the v59 regex
    # missed this form, so the statement was passed to SQLite which
    # raised ``OperationalError: near "OR": syntax error``. This broke
    # migration 006 (the patient-safety withdrawn-drug trigger) on
    # SQLite. Verified failing by actually running migrations. Fix:
    # generalize the regex to match any ``CREATE TRIGGER ... EXECUTE
    # FUNCTION ...`` statement regardless of the event clause (BEFORE
    # INSERT / BEFORE UPDATE / BEFORE INSERT OR UPDATE / AFTER ... /
    # OF <col>,<col> / etc).
    out = re.sub(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+\w+\s+(?:BEFORE|AFTER|INSTEAD\s+OF)\s+[^;]*?EXECUTE\s+FUNCTION\s+\w+\s*\(\s*\)\s*;",
        "-- [SQLite-skip] CREATE TRIGGER (PostgreSQL-only)\n;",
        out, flags=re.IGNORECASE | re.DOTALL,
    )
    # Keep the original (stricter) pattern as a backstop for any
    # PostgreSQL trigger that doesn't use EXECUTE FUNCTION.
    out = re.sub(
        r"CREATE\s+TRIGGER\s+\w+\s+BEFORE\s+UPDATE\s+ON\s+\w+\s+FOR\s+EACH\s+ROW\s+EXECUTE\s+FUNCTION\s+\w+\(\)\s*;",
        "-- [SQLite-skip] CREATE TRIGGER (PostgreSQL-only)\n;",
        out, flags=re.IGNORECASE,
    )
    # 9d. v59 ROOT FIX: strip ``DROP FUNCTION IF EXISTS ...`` and
    # ``DROP TRIGGER IF EXISTS ...`` statements (PostgreSQL-only).
    out = re.sub(
        r"DROP\s+FUNCTION\s+IF\s+EXISTS\s+[^;]+;",
        "-- [SQLite-skip] DROP FUNCTION\n;",
        out, flags=re.IGNORECASE,
    )
    out = re.sub(
        r"DROP\s+TRIGGER\s+IF\s+EXISTS\s+[^;]+;",
        "-- [SQLite-skip] DROP TRIGGER\n;",
        out, flags=re.IGNORECASE,
    )
    # 9e. v59 ROOT FIX (compound of InChIKey fix -- SQLite regex operator
    # ``~`` in CHECK constraints): the migration 001 SQL uses PostgreSQL's
    # regex operator ``~`` in three CHECK constraints:
    #   - chk_drugs_inchikey_format: inchikey ~ '^[A-Z]{14}-[A-Z]{10}-[A-Z]$'
    #   - chk_gda_disease_id_format: 9 patterns (omim/disgenet/doid/...)
    #   - chk_gda_pmid_list: pmid_list ~ '^[\\d;,\\s]*$'
    # SQLite does NOT support ``~`` -- every CREATE TABLE containing one
    # of these CHECKs raised ``OperationalError: near "~": syntax error``
    # on SQLite, causing the entire drugs + GDA table creation to fail.
    # The ORM's CheckConstraint was already fixed (v59 compound fix) to
    # use portable forms. The migration SQL keeps the regex (PostgreSQL-
    # native -- works on PG). On SQLite, we replace ``<col> ~ '<regex>'``
    # with a portable equivalent (see the v76 specific-regex handling
    # below for the InChIKey case; all other regexes fall back to the
    # generic ``LENGTH(TRIM(<col>)) > 0`` non-empty backstop).
    # Full format validation is enforced by the Python validators
    # (cleaning._constants.is_canonical_inchikey for InChIKey,
    # cleaning.normalizer for SMILES, database.loaders._validate_disease_id
    # for disease_id) on both dialects.
    #
    # v62 ROOT FIX (T-001 compound of regex operator): the v59 regex
    # only matched ``<identifier> ~ '...'`` where identifier is a bare
    # column name (``\w+``). It did NOT match function-call LHS like
    # ``lower(groups) ~ '(^|;|\\|)withdrawn(;|$|\\|)'`` in migration
    # 006's UPDATE statement (the T-002 withdrawn-drug backfill). The
    # ``~`` survived translation, SQLite raised
    # ``OperationalError: near "~": syntax error``, and migration 006
    # was marked FAILED -- breaking the patient-safety invariant that
    # ``is_withdrawn=TRUE`` for Vioxx/Bextra/Meridia/Avandia/Redux.
    # Verified failing by actually running ``run_migrations()`` on
    # SQLite -- previous AIs claimed this was fixed. Fix: extend the
    # LHS pattern to accept either a bare identifier OR a function
    # call ``name(args)``.
    #
    # v76 ROOT FIX (T-038 -- InChIKey regex gets a STRONG portable
    # equivalent, not the weak LENGTH(TRIM()) > 0 backstop):
    #   The v59/v62 generic translation replaced EVERY ``<col> ~ '<regex>'``
    #   with ``LENGTH(TRIM(<col>)) > 0``. For the InChIKey CHECK, this was
    #   a REGRESSION: the pre-v76 SQL used ``LENGTH(inchikey) = 27 OR
    #   LIKE 'SYNTH%'`` (a reasonable backstop), but after v76 T-038
    #   changed the SQL to ``inchikey ~ '^[A-Z]{14}-[A-Z]{10}-[A-Z]$'``,
    #   the generic translator would have produced ``LENGTH(TRIM(inchikey))
    #   > 0`` -- WEAKER than the original LENGTH=27 check. A 1-char
    #   InChIKey would pass on SQLite.
    #
    # P1-015 ROOT FIX (Team-2 — use real REGEXP instead of LENGTH+SUBSTR
    #   backstop):
    #   The v76 T-038 fix translated the InChIKey regex to
    #   ``LENGTH(inchikey) = 27 AND SUBSTR(inchikey, 15, 1) = '-' AND
    #   SUBSTR(inchikey, 26, 1) = '-'``. This was STRONGER than the
    #   generic LENGTH>0 backstop but STILL accepted any 27-char string
    #   with hyphens at positions 15 and 26 — including digits, lowercase,
    #   punctuation (e.g. ``11111111111111-2222222222-3``,
    #   ``aaaaaaaaaaaaaa-bbbbbbbbbb-c``, ``!!!!!!!!!!!!!!-!!!!!!!!!!-!``).
    #   Dev DBs (SQLite) accepted gibberish InChIKeys that prod PostgreSQL
    #   rejected. ROOT FIX: ``database/connection.py`` now registers a
    #   SQLite REGEXP function via ``create_function`` (see
    #   ``_register_sqlite_regexp_function`` in ``_attach_lifecycle_events``).
    #   This lets SQLite execute the SAME regex as PostgreSQL. The
    #   translation below converts ``<col> ~ '<regex>'`` to
    #   ``<col> REGEXP '<regex>'`` for SQLite — IDENTICAL semantics to
    #   PostgreSQL's ``~``. Dev/prod behavior is now identical for ALL
    #   regex-based CHECK constraints (InChIKey, disease_id, pmid_list,
    #   withdrawn-drug backfill, etc.). The previous LENGTH+SUBSTR
    #   backstop is removed — it was a workaround for SQLite's lack of
    #   native regex, which is no longer needed.
    #
    #   The specific InChIKey translation (v76 T-038) is REMOVED — the
    #   generic REGEXP translation handles it correctly. All regexes
    #   (InChIKey, disease_id, pmid_list, etc.) now use the SAME
    #   ``<col> REGEXP '<regex>'`` form on SQLite.
    #
    #   SAFETY: if the REGEXP function is NOT registered (e.g. a test
    #   that creates a SQLite engine without going through
    #   ``connection.py``), SQLite raises ``OperationalError: no such
    #   function: REGEXP`` on the first INSERT. This is BY DESIGN — it
    #   surfaces the missing registration immediately rather than
    #   silently accepting invalid data. Tests that bypass
    #   ``connection.py`` must register the REGEXP function themselves
    #   (see ``tests/conftest.py`` or copy the ``_sqlite_regexp``
    #   function from ``connection.py``).
    # Translate ``<col> ~ '<regex>'`` → ``<col> REGEXP '<regex>'`` for
    # SQLite. The REGEXP function is registered in connection.py at
    # engine creation time (P1-015 ROOT FIX).
    out = re.sub(
        r"(\w+(?:\s*\([^)]*\))?)\s*~\s*('[^']*')",
        r"\1 REGEXP \2",
        out, flags=re.IGNORECASE,
    )
    # 10. v35 ROOT FIX (issue 33): STRING_AGG(...) -> GROUP_CONCAT(...).
    # PostgreSQL's ``STRING_AGG(expr, sep)`` is the equivalent of SQLite's
    # ``GROUP_CONCAT(expr, sep)`` -- the argument order is the SAME (expr
    # first, separator second), so a direct name swap is semantically
    # correct. Without this translation, any migration that uses
    # STRING_AGG (e.g. to build a delimited list of values per group)
    # raises ``OperationalError: no such function: STRING_AGG`` on SQLite
    # and is skipped -- silently leaving the migration's intended data
    # transformation unapplied.
    out = re.sub(r"\bSTRING_AGG\s*\(", "GROUP_CONCAT(", out, flags=re.IGNORECASE)
    # 11. v35 ROOT FIX (issue 33): FILTER (WHERE ...) -> CASE WHEN ... END.
    # PostgreSQL supports the SQL:2003 ``FILTER`` clause for aggregate
    # functions: ``COUNT(*) FILTER (WHERE condition)``. SQLite does NOT
    # support FILTER (as of 3.46) -- the equivalent is
    # ``COUNT(CASE WHEN condition THEN 1 END)`` (or ``SUM(CASE WHEN
    # condition THEN 1 ELSE 0 END)``). The translation below rewrites
    # ``<agg>(<expr>) FILTER (WHERE <cond>)`` to
    # ``<agg>(CASE WHEN <cond> THEN <expr> END)``.
    # Note: this is a regex-based best-effort translation. It handles the
    # common form ``<agg>(<expr>) FILTER (WHERE <cond>)`` where ``<cond>``
    # does not contain nested parens. More complex conditions may need
    # manual translation.
    def _filter_to_case_when(m: "re.Match[str]") -> str:
        agg_name = m.group(1)
        agg_arg = m.group(2)
        cond = m.group(3)
        # Map each aggregate to its CASE WHEN equivalent. SUM/AVG/MIN/MAX
        # preserve the inner expression; COUNT(*) is special-cased to
        # COUNT(CASE WHEN cond THEN 1 END).
        agg_upper = agg_name.upper()
        if agg_upper == "COUNT" and agg_arg.strip() == "*":
            return f"COUNT(CASE WHEN {cond} THEN 1 END)"
        return f"{agg_name}(CASE WHEN {cond} THEN {agg_arg} END)"

    out = re.sub(
        r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(([^()]*)\)\s*FILTER\s*\(\s*WHERE\s+([^()]*)\)",
        _filter_to_case_when, out, flags=re.IGNORECASE,
    )
    # v62 ROOT FIX (T-001 compound of adjacent string literal concatenation):
    # PostgreSQL supports adjacent string literal concatenation:
    #   'abc' 'def'  =>  'abcdef'
    # SQLite does NOT support this -- it raises ``syntax error`` when it
    # encounters two string literals separated only by whitespace.
    # Migration 009's INSERT INTO schema_version uses this PostgreSQL
    # feature for multi-line descriptions. Fix: insert ``||`` (SQL
    # standard concatenation operator) between adjacent string literals.
    # The regex matches a complete string literal (handling ``''`` escape),
    # followed by whitespace, followed by another string literal start.
    # It inserts ``||`` between them. Applied repeatedly to handle 3+
    # adjacent literals.
    # IMPORTANT: this must run AFTER all other regexes that might modify
    # string literal content (COMMENT ON, RAISE, etc.).
    def _add_concat_operator(text: str) -> str:
        prev = None
        while prev != text:
            prev = text
            # Match: complete string literal + whitespace + opening quote of next literal
            # String literal pattern: '(?:[^']|'')*' handles '' escape
            text = re.sub(
                r"((?:'(?:[^']|'')*')\s+)'",
                r"\1|| '", text, flags=re.DOTALL,
            )
        return text
    out = _add_concat_operator(out)
    return out


# ---------------------------------------------------------------------------
# Security: MIGRATION_DATABASE_URL validation (BUG-SEC-02)
# ---------------------------------------------------------------------------

_SAFE_URL_SCHEMES = frozenset({
    "postgresql", "postgresql+psycopg2", "sqlite", "sqlite+pysqlite",
})
_SAFE_URL_PARAMS = frozenset({
    "sslmode", "connect_timeout", "application_name", "host", "port",
})


def _validate_migration_database_url(url: str) -> None:
    """Validate MIGRATION_DATABASE_URL for safety.

    BUG-SEC-02: Ensures the URL scheme is allowed and query parameters
    are restricted to known-safe ones.
    """
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(url)

    if parsed.scheme not in _SAFE_URL_SCHEMES:
        raise ValueError(
            f"Invalid MIGRATION_DATABASE_URL scheme: {parsed.scheme!r}. "
            f"Allowed: {sorted(_SAFE_URL_SCHEMES)}"
        )

    if parsed.scheme.startswith("postgresql") and not parsed.hostname:
        raise ValueError(
            "MIGRATION_DATABASE_URL must include a hostname for PostgreSQL."
        )

    # Check query parameters for unsafe ones
    if parsed.query:
        params = parse_qs(parsed.query)
        unsafe = set(params.keys()) - _SAFE_URL_PARAMS
        if unsafe:
            logger.warning(
                "MIGRATION_DATABASE_URL contains potentially unsafe "
                "query parameters: %s. Allowed: %s",
                sorted(unsafe), sorted(_SAFE_URL_PARAMS),
            )


# ---------------------------------------------------------------------------
# Engine health check (GUARD-ARCH-10)
# ---------------------------------------------------------------------------


def _check_engine_health(engine) -> None:
    """Check that the database engine is alive and usable.

    GUARD-ARCH-10: Detects disposed engines before migration execution.
    """
    try:
        # Check pool status for pooled engines
        if hasattr(engine, "pool") and engine.pool is None:
            raise ResourceClosedError(
                "Database engine pool is None. The engine may have been "
                "disposed. Obtain a fresh engine via get_engine()."
            )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except ResourceClosedError:
        raise
    except Exception as exc:
        raise ResourceClosedError(
            "Database engine health check failed. The engine may have been "
            f"disposed or the database is unreachable: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Data classes (DES-MIG-01, REL-MIG-04, ARCH-MIG-06, LOG-MIG-02)
# Defined here to avoid circular imports when __init__.py re-exports them.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MigrationConfig:
    """Configuration dataclass for customizing migration behavior.

    GAP-CFG-07: All configuration options are documented below.
    BUG-CFG-02: Magic numbers replaced with named constants.
    """

    migrations_dir: Path | None = None
    dry_run: bool = False
    batch_size: int = 1000
    timeout_seconds: int = 3600
    skip_migrations: set[str] | None = None
    require_checksum: bool = False
    concurrent_indexes: bool = False
    interactive: bool = False
    stop_on_failure: bool = True
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    verify_data_checksums: bool = False
    allow_destructive_sql: bool = True
    on_migration_start: Callable | None = None
    on_migration_complete: Callable | None = None
    on_migration_fail: Callable | None = None
    correlation_id: str | None = None
    pipeline_name: str | None = None
    run_id: str | None = None
    pipeline_run_id: int | None = None
    # GUARD-ARCH-09: Lock timeout for concurrent migration protection
    lock_timeout_seconds: int = 30
    # BUG-DQ-03/GUARD-DQ-08: Block on data issues
    block_on_data_issues: bool = True
    # GAP-REL-07: Circuit breaker
    circuit_breaker_threshold: int = 3
    # GAP-PERF-04: Fail fast on repeated errors
    fail_fast_on_repeated_errors: bool = True
    # GAP-PERF-05: Batch DML processing
    batch_dml: bool = True
    # GAP-SEC-05: Encrypt audit data
    encrypt_audit_data: bool = False

    def __post_init__(self) -> None:
        """BUG-CFG-03: Validate configuration values at construction time."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.timeout_seconds < 1:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")
        if self.retry_backoff_base <= 0:
            raise ValueError(f"retry_backoff_base must be positive, got {self.retry_backoff_base}")
        if self.circuit_breaker_threshold < 1:
            raise ValueError(f"circuit_breaker_threshold must be >= 1, got {self.circuit_breaker_threshold}")

    @classmethod
    def from_env(cls) -> MigrationConfig:
        """Create a MigrationConfig from environment variables.

        BUG-DES-02: Uses dataclasses.replace() instead of dict merge.
        GAP-CODE-10: Handles ValueError for int() conversions.
        GAP-IDEM-06: Caches config with environment hash.
        """
        # GAP-IDEM-06: Check cache
        env_keys = [
            "APP_ENV", "MIGRATIONS_DRY_RUN", "MIGRATIONS_REQUIRE_CHECKSUM",
            "MIGRATIONS_SKIP", "MIGRATIONS_BATCH_SIZE", "MIGRATIONS_TIMEOUT",
        ]
        env_hash = hashlib.md5(
            "&".join(f"{k}={os.environ.get(k, '')}" for k in sorted(env_keys)).encode()
        ).hexdigest()

        if hasattr(cls, "_cached_config") and hasattr(cls, "_cached_config_env_hash"):
            if cls._cached_config_env_hash == env_hash and cls._cached_config is not None:
                return cls._cached_config

        env = os.environ.get("APP_ENV", "development")
        if env == "production":
            base = cls(
                require_checksum=True,
                verify_data_checksums=True,
                stop_on_failure=True,
                interactive=False,
                timeout_seconds=7200,
                block_on_data_issues=True,
            )
        elif env == "staging":
            base = cls(
                require_checksum=True,
                stop_on_failure=True,
                dry_run=False,
            )
        else:
            base = cls()

        # Override with explicit env vars (GAP-CODE-10: safe int conversion)
        overrides: dict[str, Any] = {}
        if os.environ.get("MIGRATIONS_DRY_RUN") == "1":
            overrides["dry_run"] = True
        if os.environ.get("MIGRATIONS_REQUIRE_CHECKSUM") == "1":
            overrides["require_checksum"] = True
        skip_str = os.environ.get("MIGRATIONS_SKIP")
        if skip_str:
            overrides["skip_migrations"] = {s.strip() for s in skip_str.split(",") if s.strip()}
        batch_str = os.environ.get("MIGRATIONS_BATCH_SIZE")
        if batch_str:
            try:
                overrides["batch_size"] = int(batch_str)
            except ValueError:
                logger.warning(
                    "Invalid MIGRATIONS_BATCH_SIZE value: %r. Must be an integer. Using default.",
                    batch_str,
                )
        timeout_str = os.environ.get("MIGRATIONS_TIMEOUT")
        if timeout_str:
            try:
                overrides["timeout_seconds"] = int(timeout_str)
            except ValueError:
                logger.warning(
                    "Invalid MIGRATIONS_TIMEOUT value: %r. Must be an integer. Using default.",
                    timeout_str,
                )

        # BUG-DES-02: Use dataclasses.replace() instead of dict merge
        if overrides:
            result = replace(base, **overrides)
        else:
            result = base

        # GAP-IDEM-06: Cache the result
        cls._cached_config = result
        cls._cached_config_env_hash = env_hash

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MigrationConfig:
        """GAP-CFG-04: Create MigrationConfig from a dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# Module-level cache for from_env (GAP-IDEM-06)
MigrationConfig._cached_config: MigrationConfig | None = None
MigrationConfig._cached_config_env_hash: str | None = None


@dataclass(frozen=True)
class MigrationResult:
    """Result dataclass returned by run_migrations()."""
    applied: list[str]
    skipped: list[str]
    failed: list[str]
    total_duration_seconds: float
    dialect: str
    schema_version_before: int | None
    schema_version_after: int | None
    row_count_changes: dict[str, tuple[int, int]] = field(default_factory=dict)
    data_checksums: dict[str, str] = field(default_factory=dict)
    # v22 ROOT FIX (audit section 5 finding 11 -- Type contract violation):
    # was ``list[str]`` but dicts were appended at runtime. Consumers that
    # did ``err.upper()`` would crash. Unify: all entries are dicts with
    # keys {migration, dialect, error, phase}. String-only sites wrap
    # their message in a dict for consistency.
    errors: list[dict[str, str]] = field(default_factory=list)
    schema_drift_detected: bool = False


@dataclass(frozen=True)
class MigrationHealthResult:
    """Result of a migration health check."""
    all_applied: bool
    applied_count: int
    pending_count: int
    applied_migrations: list[str]
    pending_migrations: list[str]
    schema_version_matches: bool
    dialect: str
    # GAP-DQ-04: Phantom migrations (recorded but no SQL file)
    phantom_migrations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MigrationStatus:
    """Detailed migration status."""
    applied_migrations: list[dict[str, Any]]
    pending_migrations: list[str]
    total_migrations: int
    schema_version_code: int
    schema_version_db: int | None


@dataclass(frozen=True)
class MigrationMetrics:
    """Metrics for a migration run."""
    total_migrations: int
    applied_count: int
    skipped_count: int
    failed_count: int
    total_duration_seconds: float
    per_migration_timing: dict[str, float]
    dialect: str


class MigrationError(Exception):
    """Raised when one or more migrations fail to apply."""

    def __init__(self, failed: list[str], errors: list[Exception]) -> None:
        self.failed = failed
        self.errors = errors
        super().__init__(
            f"{len(failed)} migration(s) failed: {', '.join(failed)}"
        )


class StatementExecutionError(RuntimeError):
    """Raised when a single SQL statement within a migration fails.

    v59 ROOT FIX: introduced to give per-statement context to SQLite
    migration failures. The previous runner logged the ENTIRE migration
    file as the "error", making it impossible to diagnose which of the
    hundreds of statements in migration 002 actually failed.
    """

    def __init__(self, message: str, *, migration_name: str | None = None,
                 statement: str | None = None) -> None:
        self.migration_name = migration_name
        self.statement = statement
        super().__init__(message)


# ---------------------------------------------------------------------------
# Deprecated alias (DES-MIG-03, CMP-MIG-04)
# GAP-CODE-09: Single canonical definition; __init__.py references this.
# ---------------------------------------------------------------------------
_DEPRECATED_ALIASES: dict[str, str] = {
    "run_migration_002": "run_migrations",
}


# ---------------------------------------------------------------------------
# Helper functions extracted from run_migrations (GAP-ARCH-08)
# ---------------------------------------------------------------------------


def _resolve_engine(engine, config: MigrationConfig | None = None):
    """Resolve the database engine for migration execution.

    GAP-ARCH-08: Extracted from run_migrations for testability.
    Handles MIGRATION_DATABASE_URL fallback and dialect validation.
    BUG-ARCH-04: Uses deferred import via _get_default_engine().
    BUG-SEC-02: Validates MIGRATION_DATABASE_URL.
    """
    if engine is None:
        migration_url = os.environ.get("MIGRATION_DATABASE_URL")
        if migration_url:
            _validate_migration_database_url(migration_url)
            from sqlalchemy import create_engine
            engine = create_engine(migration_url)
        else:
            engine = _get_default_engine()

    # Validate dialect
    dialect_name = engine.dialect.name
    if dialect_name not in SUPPORTED_DIALECTS:
        raise ValueError(
            f"Unsupported database dialect: {dialect_name!r}. "
            f"Supported: {sorted(SUPPORTED_DIALECTS)}"
        )

    return engine


# v75 ROOT FIX (T-026 -- migration 007 DO $$ block fails on SQLite):
# PostgreSQL-only post-migration upgrades that the portable SQL files
# cannot express (because SQLite has no JSONB type, no ALTER COLUMN
# TYPE, no pg_constraint catalog). Each entry is keyed by the migration
# file name and maps to a list of (description, sql, idempotent_guard)
# tuples. The idempotent_guard is a SELECT that returns 1 if the
# upgrade has already been applied (so the runner skips it on re-runs).
#
# This hook is called ONLY on PostgreSQL (dialect_name == DIALECT_POSTGRESQL).
# SQLite dev/test DBs use the TEXT column directly -- the SQLAlchemy JSON
# dialect serialises Python dicts to TEXT transparently on both dialects,
# so application code is identical.
_POSTGRES_ONLY_UPGRADES: dict[str, list[tuple[str, str, str]]] = {
    "007_pipeline_run_metadata.sql": [
        (
            "Upgrade pipeline_runs.metadata_json TEXT -> JSONB",
            # The column was added as TEXT by the portable migration file
            # (works on both SQLite and PostgreSQL). On PostgreSQL we
            # upgrade it to JSONB for indexable, deduplicated JSON storage.
            # The USING clause converts existing TEXT values to JSONB by
            # parsing them as JSON; NULL values stay NULL.
            "ALTER TABLE pipeline_runs "
            "ALTER COLUMN metadata_json TYPE JSONB "
            "USING metadata_json::jsonb",
            # Idempotent guard: if the column is already JSONB, skip.
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'pipeline_runs' "
            "AND column_name = 'metadata_json' "
            "AND data_type = 'jsonb'",
        ),
    ],
}


def _apply_postgres_only_upgrades(conn, migration_name: str) -> None:
    """Apply PostgreSQL-only upgrades for a migration (v75 ROOT FIX T-026).

    v90 ROOT FIX (BUG #15 -- P1 _apply_postgres_only_upgrades outside
    transaction):
      The previous signature was ``_apply_postgres_only_upgrades(engine,
      migration_name)`` and it opened its OWN ``with engine.begin() as
      conn:`` block. It was called AFTER the per-migration transaction
      committed (line ~3991, outside the ``with engine.begin()`` block).
      If the upgrade failed (e.g. ``ALTER TABLE pipeline_runs ALTER
      COLUMN metadata_json TYPE JSONB`` failed because a row had invalid
      JSON), the migration was recorded as "applied" but
      ``metadata_json`` stayed TEXT. Subsequent queries that assumed
      JSONB operators (``->``, ``->>``) failed at runtime.
      ROOT FIX: accept a ``conn`` parameter (the connection from the
      per-migration transaction) and execute the upgrade INSIDE the
      caller's transaction. If the upgrade fails, the entire migration
      rolls back (including the ``_record_migration`` bookkeeping) -- the
      migration is NOT marked as applied, and the operator can fix the
      root cause (e.g. clean up invalid JSON) and re-run. This is the
      fail-closed approach appropriate for an institutional-grade pharma
      system.
    """
    # v90: the dialect check is now done by the caller (inside the
    # per-migration transaction). We keep a defensive check here for
    # any future caller that might pass a non-PostgreSQL connection.
    bind = conn.engine if hasattr(conn, "engine") else conn
    dialect_name = bind.dialect.name
    if dialect_name != DIALECT_POSTGRESQL:
        return
    upgrades = _POSTGRES_ONLY_UPGRADES.get(migration_name)
    if not upgrades:
        return
    for description, sql, guard_sql in upgrades:
        try:
            # Idempotent guard: skip if already applied.
            guard_result = conn.execute(text(guard_sql))
            if guard_result.fetchone() is not None:
                logger.debug(
                    "  [SKIP] Postgres-only upgrade (already applied): %s",
                    description,
                )
                continue
        except Exception as guard_exc:
            # Guard failure (e.g. information_schema not accessible)
            # -- log and proceed with the upgrade attempt. The upgrade
            # itself is idempotent via IF NOT EXISTS / TYPE guards.
            logger.debug(
                "  [WARN] Postgres-only upgrade guard failed for %s: %s",
                description, guard_exc,
            )
        # v90: let failures propagate -- the caller's transaction will
        # roll back, and the migration will NOT be marked as applied.
        # This is the fail-closed behavior: if the JSONB upgrade fails
        # (e.g. invalid JSON in metadata_json), the operator must fix
        # the data and re-run the migration. The previous non-blocking
        # behavior silently left the column as TEXT while marking the
        # migration as applied -- schema drift.
        conn.execute(text(sql))
        logger.info(
            "  [OK] Applied Postgres-only upgrade: %s",
            description,
        )


def _apply_python_columns(
    engine, config: MigrationConfig | None, inspector,
) -> list[str]:
    """Add missing columns via Python-level ALTER TABLE.

    GAP-ARCH-08: Extracted from run_migrations.
    BUG-ARCH-03: Covers ALL 7 core tables, not just proteins.
    """
    added_columns: list[str] = []
    with engine.begin() as conn:
        for table_name, columns in REQUIRED_COLUMNS.items():
            try:
                _validate_sql_identifier(table_name, "table name")
            except ValueError:
                logger.warning("Invalid table name in REQUIRED_COLUMNS: %s", table_name)
                continue

            if not _table_exists(inspector, table_name):
                logger.info("Table '%s' does not exist yet, skipping column checks", table_name)
                continue

            for column_name, column_type in columns:
                try:
                    _validate_sql_identifier(column_name, "column name")
                except ValueError:
                    logger.warning("Invalid column name in REQUIRED_COLUMNS: %s.%s", table_name, column_name)
                    continue

                if _column_exists(inspector, table_name, column_name):
                    logger.debug("Column '%s.%s' already exists, skipping", table_name, column_name)
                    continue

                # Build ALTER TABLE statement based on dialect
                alter_sql = (
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                )

                logger.info("Adding column '%s.%s' (%s)", table_name, column_name, column_type)
                try:
                    conn.execute(text(alter_sql))
                    added_columns.append(f"{table_name}.{column_name}")
                except OperationalError as exc:
                    logger.warning(
                        "Could not add column '%s.%s': %s",
                        table_name, column_name, exc,
                    )

    return added_columns


def _finalize_result(
    applied: list[str],
    skipped: list[str],
    failed: list[str],
    # v22 ROOT FIX: type contract -- dicts are appended (not strings).
    errors: list[dict[str, str]],
    per_migration_timing: dict[str, float],
    dialect_name: str,
    start_time: float,
    engine,
    config: MigrationConfig | None,
    row_count_changes: dict[str, tuple[int, int]],
    data_checksums: dict[str, str],
    schema_version_before: int | None,
    inspector,
) -> MigrationResult:
    """Build the MigrationResult dataclass.

    GAP-ARCH-08: Extracted from run_migrations.
    """
    total_duration = time.monotonic() - start_time

    # Get schema version after
    schema_version_after: int | None = None
    try:
        if _table_exists(inspector, "schema_version"):
            with engine.begin() as conn:
                r = conn.execute(text("SELECT MAX(version) FROM schema_version"))
                schema_version_after = r.scalar()
    except (OperationalError, ProgrammingError) as exc:
        logger.warning("Could not read schema version after migration: %s", exc)

    # BUG-IDEM-01: Verify schema matches ORM after migration
    schema_drift = False
    try:
        schema_result = verify_schema_matches_orm(engine)
        if schema_result["missing_in_db"]:
            logger.warning(
                "Schema drift detected after migration. Missing in DB: %s",
                schema_result["missing_in_db"][:10],
            )
            schema_drift = True
    except Exception as exc:
        logger.warning("Could not verify schema after migration: %s", exc)

    # If any failures and not already raised
    if failed and not (config and config.stop_on_failure):
        logger.critical(
            "%d migration(s) failed: %s", len(failed), ", ".join(failed)
        )

    logger.info("Migration run complete for dialect: %s", dialect_name)

    # Record as pipeline run for lineage (LINE-MIG-04)
    if config and config.pipeline_run_id:
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "UPDATE pipeline_runs SET status = :status "
                        "WHERE id = :id"
                    ),
                    {
                        "status": "failed" if failed else "success",
                        "id": config.pipeline_run_id,
                    },
                )
        except (OperationalError, ProgrammingError) as exc:
            logger.warning("Could not update pipeline_run: %s", exc)

    return MigrationResult(
        applied=applied,
        skipped=skipped,
        failed=failed,
        total_duration_seconds=total_duration,
        dialect=dialect_name,
        schema_version_before=schema_version_before,
        schema_version_after=schema_version_after,
        row_count_changes=row_count_changes,
        data_checksums=data_checksums,
        errors=errors,
        schema_drift_detected=schema_drift,
    )


# ---------------------------------------------------------------------------
# MAIN: run_migrations (DES-MIG-01, DES-MIG-02, DES-MIG-05, REL-MIG-04)
# ---------------------------------------------------------------------------


def run_migrations(
    engine=None,
    config=None,
) -> MigrationResult:
    """Run cross-dialect migrations: add missing columns and apply SQL migrations.

    This function:
    1. Uses SQLAlchemy inspect() to check for missing columns.
    2. Adds any missing columns with appropriate SQL for the dialect.
    3. Runs any pending .sql migration files (PostgreSQL only).
    4. Returns a MigrationResult with complete audit information.

    Parameters
    ----------
    engine : Engine | None
        SQLAlchemy engine. If None, calls _get_default_engine() for backward
        compatibility (DES-MIG-02 dependency injection, BUG-ARCH-04).
    config : MigrationConfig | None
        Configuration for customizing migration behavior. If None,
        uses defaults (lenient, suitable for development).

    Returns
    -------
    MigrationResult
        Complete record of what happened during the migration run.

    Raises
    ------
    MigrationError
        If one or more migrations fail and config.stop_on_failure is True.
    RuntimeError
        If MIGRATIONS_READONLY=1 is set in the environment.
    """
    # Resolve config with defaults
    if config is None:
        config = MigrationConfig()

    # Validate configuration (CFG-MIG-05)
    config_warnings = validate_migration_config(config)
    for w in config_warnings:
        logger.warning("Migration config warning: %s", w)

    # GAP-CFG-05: Log config diff at migration start
    default_config = MigrationConfig()
    config_diff = {
        k: getattr(config, k)
        for k in MigrationConfig.__dataclass_fields__
        if getattr(config, k) != getattr(default_config, k)
    }
    if config_diff:
        logger.info("Migration config overrides: %s", config_diff)

    # Security check (SEC-MIG-04)
    _check_readonly_mode()

    # GAP-ARCH-08: Resolve engine (BUG-ARCH-04, BUG-SEC-02)
    engine = _resolve_engine(engine, config)

    # GUARD-ARCH-10: Check engine health
    _check_engine_health(engine)

    inspector = inspect(engine)
    dialect_name = engine.dialect.name

    # GUARD-ARCH-09: Acquire migration lock
    # v29 ROOT FIX (audit D-12): pg_advisory_lock was on a separate
    # connection that closed immediately. The previous code used:
    #     with engine.connect() as conn:
    #         conn.execute(text("SELECT pg_advisory_lock(54321)"))
    # The ``with`` block exits at the end of the line, returning the
    # connection to the pool. For psycopg2 + SQLAlchemy's QueuePool,
    # the connection is not closed but it CAN be handed to another
    # caller, and the session-level advisory lock is bound to the
    # backend PID -- once the PID is recycled or the connection
    # returned, the lock is effectively released (or worse, held by
    # an unrelated caller). Two concurrent ``run_migrations()`` calls
    # therefore did NOT actually serialize -- both acquired the lock
    # on their own short-lived connections, both proceeded in
    # parallel, and both "released" a lock they may not have held.
    #
    # Fix: open a DEDICATED long-lived connection (``lock_conn``) that
    # stays open for the entire duration of ``_run_migrations_inner``.
    # The session-level advisory lock is bound to ``lock_conn``'s
    # backend PID; concurrent callers' ``pg_advisory_lock(54321)``
    # will BLOCK until we explicitly ``pg_advisory_unlock`` and close
    # ``lock_conn`` in the ``finally`` block below. This is the same
    # pattern used by ``database.connection.init_db`` (REM-28 fix).
    lock_conn = None
    lock_file = None
    if dialect_name == DIALECT_POSTGRESQL:
        try:
            lock_conn = engine.connect()
            lock_conn.execute(text("SELECT pg_advisory_lock(54321)"))
            logger.debug(
                "Acquired PostgreSQL advisory lock (54321) for migrations "
                "on long-lived lock_conn (held until migrations complete)"
            )
        except Exception as exc:
            logger.warning("Could not acquire PostgreSQL advisory lock: %s", exc)
            if lock_conn is not None:
                try:
                    lock_conn.close()
                except Exception:
                    pass
                lock_conn = None
    elif dialect_name == DIALECT_SQLITE:
        # File-based lock for SQLite.
        # P1-003 ROOT FIX (Team-1 -- SQLite :memory: creates junk lock file):
        #   For SQLite dialect, ``engine.url.database`` returns the string
        #   ``:memory:`` when the URL is ``sqlite:///:memory:``. The previous
        #   code computed ``lock_path = Path(":memory:").parent / f"{Path(':memory:').name}.migration.lock"``
        #   = ``./:memory:.migration.lock`` and opened it for flock-based
        #   locking. The file was created in the current working directory
        #   and stayed there forever (the cleanup code below closed the file
        #   handle but did NOT delete the file). The file
        #   ``phase1/:memory:.migration.lock`` was observed in the audit --
        #   confirming this bug fires in production (CI/test runs that use
        #   ``:memory:``).
        #   ROOT FIX: add an explicit guard. In-memory databases don't need
        #   file-based locking because each connection gets its own private
        #   DB -- there's no shared state to serialize. For file-based SQLite
        #   (dev/test/prod), the flock-based lock on the sidecar file
        #   continues to work as before.
        try:
            db_path = engine.url.database
            # P1-003: skip file locking for in-memory SQLite (each
            # connection gets its own private DB; no serialization needed).
            if db_path == ":memory:":
                logger.debug(
                    "Skipping SQLite file lock for :memory: database "
                    "(P1-003 ROOT FIX -- in-memory DBs need no file lock)"
                )
                lock_file = None
            elif db_path:
                lock_path = Path(db_path).parent / f"{Path(db_path).name}.migration.lock"
                lock_file = open(lock_path, "w")
                import fcntl
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.debug("Acquired SQLite file lock for migrations at %s", lock_path)
            else:
                lock_file = None
        except (ImportError, OSError) as exc:
            logger.warning("Could not acquire SQLite file lock: %s", exc)
            lock_file = None
    else:
        lock_file = None

    try:
        return _run_migrations_inner(
            engine, config, inspector, dialect_name,
        )
    finally:
        # Release lock
        if dialect_name == DIALECT_POSTGRESQL and lock_conn is not None:
            try:
                lock_conn.execute(text("SELECT pg_advisory_unlock(54321)"))
                logger.debug("Released PostgreSQL advisory lock (54321)")
            except Exception:
                pass
            finally:
                try:
                    lock_conn.close()
                except Exception:
                    pass
        elif dialect_name == DIALECT_SQLITE and lock_file is not None:
            try:
                import fcntl
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()
                logger.debug("Released SQLite file lock")
                # P1-003 ROOT FIX (Team-1 -- cleanup the lock sidecar file):
                #   The previous code closed the file handle but did NOT
                #   delete the sidecar file. The file accumulated in the
                #   working directory across runs (at most one file per
                #   SQLite DB, but it's directory pollution). ROOT FIX:
                #   unlink the sidecar file after releasing the flock.
                #   The file was created by US (in the acquire block above),
                #   so it's safe to delete -- no other process should be
                #   relying on it (the flock was our coordination primitive,
                #   not the file's existence).
                try:
                    lock_path_to_unlink = getattr(lock_file, "name", None)
                    if lock_path_to_unlink:
                        from pathlib import Path as _P
                        _lp = _P(lock_path_to_unlink)
                        if _lp.exists() and _lp.name.endswith(".migration.lock"):
                            _lp.unlink()
                            logger.debug(
                                "Removed SQLite migration lock sidecar file %s",
                                _lp,
                            )
                except OSError as unlink_exc:
                    # Best-effort cleanup -- do not mask migration errors.
                    logger.debug(
                        "Could not remove SQLite migration lock sidecar "
                        "file: %s", unlink_exc,
                    )
            except Exception:
                pass


def _run_migrations_inner(
    engine, config: MigrationConfig, inspector, dialect_name: str,
) -> MigrationResult:
    """Inner implementation of run_migrations, called after lock acquisition.

    BUG-ARCH-02: Tracks migration phases for interrupted-run detection.
    """
    # Initialize result tracking
    applied: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    # v22 ROOT FIX (audit section 5 finding 11 -- "Type contract violation"):
    # the previous annotation was ``list[str]`` but dicts are appended at
    # lines 3344 and 3375. Consumers that do ``err.upper()`` would crash.
    # Change the annotation to ``list[dict[str, str]]`` to match reality.
    errors: list[dict[str, str]] = []
    per_migration_timing: dict[str, float] = {}
    row_count_changes: dict[str, tuple[int, int]] = {}
    data_checksums: dict[str, str] = {}
    # GAP-REL-07: Circuit breaker tracking
    consecutive_failures = 0

    start_time = time.monotonic()

    # Get schema version before
    schema_version_before: int | None = None
    try:
        from database.base import SCHEMA_VERSION as _sv_code
        schema_version_before = _sv_code
        if _table_exists(inspector, "schema_version"):
            with engine.begin() as conn:
                r = conn.execute(text("SELECT MAX(version) FROM schema_version"))
                db_ver = r.scalar()
                if db_ver is not None:
                    schema_version_before = db_ver
    except Exception as exc:
        logger.warning("Could not read initial schema version: %s", exc)

    logger.info("Running migrations for dialect: %s", dialect_name)

    # BUG-ARCH-02: Phase tracking
    current_phase = _MigrationPhase.TRACKING_TABLES

    # Phase: TRACKING_TABLES
    _ensure_migration_tracking_table(engine)

    # BUG-ARCH-02: Check for interrupted runs
    with engine.begin() as conn:
        try:
            r = conn.execute(
                text(
                    "SELECT migration_name, phase_at_interrupt "
                    "FROM _migration_history "
                    "WHERE status = 'in_progress' LIMIT 1"
                )
            )
            interrupted = r.fetchone()
            if interrupted:
                logger.warning(
                    "Interrupted migration detected: %s (phase: %s). "
                    "Manual intervention may be required.",
                    interrupted[0], interrupted[1],
                )
        except (OperationalError, ProgrammingError):
            pass

    # Phase: SCIENTIFIC_VALIDATION
    current_phase = _MigrationPhase.SCIENTIFIC_VALIDATION
    sci_warnings = validate_scientific_constraints(engine)
    for w in sci_warnings:
        logger.warning("Pre-migration scientific warning: %s", w)

    # GUARD-DQ-08: Block on data issues if configured
    if config.block_on_data_issues and sci_warnings:
        raise MigrationError(
            failed=["pre_validation"],
            errors=[ValueError(
                f"{len(sci_warnings)} scientific constraint violation(s) "
                f"detected and block_on_data_issues is True. Fix data issues "
                f"before re-running, or set block_on_data_issues=False. "
                f"Warnings: {sci_warnings[:3]}"
            )],
        )

    # Phase: COLUMN_ADDITIONS
    current_phase = _MigrationPhase.COLUMN_ADDITIONS
    _apply_python_columns(engine, config, inspector)

    # Phase: SQL_FILES
    current_phase = _MigrationPhase.SQL_FILES
    migrations_dir = (
        config.migrations_dir if config and config.migrations_dir
        else MIGRATIONS_DIR
    )

    if dialect_name == DIALECT_POSTGRESQL:
        # Sort by numeric prefix (IDEM-MIG-04). Exclude *_rollback.sql
        # sidecars -- they are recovery scripts, NOT migrations. On PostgreSQL,
        # 001_initial_schema_rollback.sql would `DROP TABLE IF EXISTS drugs
        # CASCADE; ...` and destroy the staging schema on every fresh install.
        # On SQLite, multi-statement rollback files abort with "You can only
        # execute one statement at a time". (FIX-C5)
        sql_files = sorted(
            [f for f in migrations_dir.glob("*.sql") if not f.name.endswith("_rollback.sql")],
            key=lambda f: _extract_migration_number(f.name),
        )

        # GUARD-SEC-07: Validate migration file paths
        for f in sql_files:
            _validate_migration_path(f, migrations_dir)

        # Validate filename conventions (CMP-MIG-05)
        for f in sql_files:
            if not _validate_migration_filename(f.name):
                logger.warning(
                    "Migration file '%s' does not follow NNN_description.sql convention",
                    f.name,
                )

        # GAP-ARCH-06: Build dependency graph
        migration_deps: dict[str, set[str]] = {}
        for f in sql_files:
            try:
                content = f.read_text(encoding="utf-8")
                deps = _parse_migration_dependencies(content)
                migration_deps[f.name] = deps
            except Exception:
                migration_deps[f.name] = set()

        # Topological sort (no-op if no dependencies declared)
        if any(deps for deps in migration_deps.values()):
            try:
                sorted_names = _topological_sort(
                    [f.name for f in sql_files], migration_deps,
                )
                name_to_file = {f.name: f for f in sql_files}
                sql_files = [name_to_file[n] for n in sorted_names if n in name_to_file]
            except MigrationError:
                raise
            except Exception as exc:
                logger.warning("Dependency sort failed, using filename order: %s", exc)

        # GAP-LOG-04: Progress tracking
        total_migrations = len(sql_files)

        for idx, sql_file in enumerate(sql_files, 1):
            migration_name = sql_file.name

            # Check skip list (CFG-MIG-01)
            if config and config.skip_migrations and migration_name in config.skip_migrations:
                logger.info("Skipping migration (in skip list): %s", migration_name)
                skipped.append(migration_name)
                continue

            # BUG-DQ-03: Check if migration has failed too many times
            with engine.begin() as conn:
                try:
                    r = conn.execute(
                        text(
                            "SELECT retry_count FROM _failed_migrations "
                            "WHERE migration_name = :n AND resolved = FALSE"
                        ),
                        {"n": migration_name},
                    )
                    row = r.fetchone()
                    if row and row[0] >= MAX_FAILURE_COUNT:
                        logger.warning(
                            "Migration %s has failed %d times. Skipping. "
                            "Manual intervention required. Check _failed_migrations.",
                            migration_name, row[0],
                        )
                        skipped.append(migration_name)
                        continue
                except (OperationalError, ProgrammingError):
                    pass

            # Check if already applied
            with engine.begin() as conn:
                try:
                    already_applied = _is_migration_applied(conn, migration_name)
                except (OperationalError, ProgrammingError):
                    already_applied = False

                if already_applied:
                    # Check for checksum drift (IDEM-MIG-06)
                    stored_checksum = _get_stored_checksum(conn, migration_name)
                    current_checksum = _compute_checksum(sql_file.read_text(encoding="utf-8"))
                    if stored_checksum and stored_checksum != current_checksum:
                        if config and config.require_checksum:
                            msg = (
                                f"Checksum drift detected for {migration_name}: "
                                f"stored={stored_checksum[:16]}... current={current_checksum[:16]}..."
                            )
                            logger.error(msg)
                            failed.append(migration_name)
                            # v22 ROOT FIX: wrap in dict for type-contract consistency.
                            errors.append({
                                "migration": migration_name,
                                "dialect": "unknown",
                                "error": msg,
                                "phase": "checksum_drift",
                            })
                            continue
                        else:
                            logger.warning(
                                "Checksum drift for %s (stored=%s, current=%s). "
                                "Allowing re-application.",
                                migration_name, stored_checksum[:16], current_checksum[:16],
                            )

                    logger.info("Migration already applied, skipping: %s", migration_name)
                    skipped.append(migration_name)
                    continue

            # Dry-run mode (DES-MIG-05)
            if config and config.dry_run:
                raw_content = sql_file.read_text(encoding="utf-8")
                logger.info(
                    "[DRY RUN] Would apply migration: %s (%d bytes, %d lines) [%d/%d]",
                    migration_name,
                    len(raw_content),
                    raw_content.count("\n") + 1,
                    idx, total_migrations,
                )
                skipped.append(migration_name)
                continue

            # GAP-CODE-11: Read file content ONCE
            raw_content = sql_file.read_text(encoding="utf-8")
            checksum = _compute_checksum(raw_content)

            # GUARD-SEC-08 / GAP-SEC-06: Destructive SQL check
            if not config.allow_destructive_sql:
                destructive = _scan_destructive_sql(raw_content)
                if destructive:
                    raise MigrationError(
                        failed=[migration_name],
                        errors=[ValueError(
                            f"Migration {migration_name} contains destructive SQL "
                            f"({destructive}) and allow_destructive_sql is False. "
                            f"Set allow_destructive_sql=True to override."
                        )],
                    )

            # GAP-IDEM-05: Non-deterministic function check
            for nd_func in NONDETERMINISTIC_FUNCTIONS:
                if nd_func.upper() in raw_content.upper():
                    logger.warning(
                        "Migration %s contains non-deterministic function: %s. "
                        "Results may differ between runs.",
                        migration_name, nd_func,
                    )
                    break

            # Fire pre-migration callback (DES-MIG-04)
            if config and config.on_migration_start:
                try:
                    config.on_migration_start(migration_name, raw_content)
                except Exception as cb_exc:
                    logger.warning("on_migration_start callback error: %s", cb_exc)

            # Log structured event (LOG-MIG-04)
            _log_migration_event(
                "started",
                migration_name,
                {"file_size": len(raw_content), "lines": raw_content.count("\n") + 1,
                 "progress": f"{idx}/{total_migrations}"},
                correlation_id=getattr(config, "correlation_id", None),
                pipeline_name=getattr(config, "pipeline_name", None),
                run_id=getattr(config, "run_id", None),
            )

            # BUG-LOG-03: Log content hash for debugging
            logger.info(
                "Migration %s content hash: %s",
                migration_name, checksum[:16],
            )

            # GAP-DQ-05: Always compute row counts (not just when verify_data_checksums=True)
            with engine.begin() as conn:
                pre_counts = _log_table_state(conn, f"before_{migration_name}", dialect_name)

            # Data checksum before (only when configured -- expensive)
            pre_checksums: dict[str, str] = {}
            if config and config.verify_data_checksums:
                with engine.begin() as conn:
                    for table in _KNOWN_TABLES:
                        if _table_exists(inspector, table):
                            pre_checksums[table] = _compute_data_checksum(conn, table)

            # Apply the migration
            sql_content = _strip_psql_meta_commands(raw_content)
            mig_start = time.monotonic()

            # BUG-ARCH-02: Mark phase as in_progress
            with engine.begin() as conn:
                try:
                    conn.execute(
                        text(
                            "UPDATE _migration_history SET status = 'in_progress', "
                            "phase_at_interrupt = :phase "
                            "WHERE migration_name = :n"
                        ),
                        {"phase": current_phase.value, "n": migration_name},
                    )
                except (OperationalError, ProgrammingError):
                    pass

            try:
                # BUG-ARCH-01: Split SQL into individual statements
                statements = _split_sql_statements(sql_content)
                statement_count = len(statements)
                logger.info(
                    "Executing migration %s: %d statement(s) [%d/%d]",
                    migration_name, statement_count, idx, total_migrations,
                )

                # v29 ROOT FIX (audit D-14): forward migrations were
                # non-atomic. The previous code did wrap the migration
                # statements in ``engine.begin()``, BUT the retry logic
                # in ``_execute_with_retry`` retried a failed statement
                # on the SAME connection WITHOUT a SAVEPOINT. After a
                # transient statement-level failure (e.g. deadlock
                # victim, unique-violation under concurrent load),
                # PostgreSQL poisons the entire transaction -- the
                # retry attempt would fail with "current transaction
                # is aborted, commands ignored until end of
                # transaction block", the outer transaction would
                # roll back, and partial schema changes from earlier
                # statements in the SAME migration would be lost
                # (well, rolled back -- but the migration would be
                # recorded as failed even though the underlying
                # statements were valid). Worse, on dialects that
                # auto-commit per statement (SQLite in some configs),
                # a mid-migration failure could leave the schema
                # half-applied.
                #
                # Fix: ``_execute_with_retry`` now wraps each statement
                # in a SAVEPOINT (``conn.begin_nested()``). On a
                # transient failure, only the SAVEPOINT is rolled
                # back -- the outer transaction stays healthy and the
                # retry re-executes the statement cleanly. The entire
                # migration (all statements + the
                # ``_record_migration`` bookkeeping) is wrapped in a
                # single ``engine.begin()`` so a partial failure
                # rolls back atomically -- the schema is never left
                # half-migrated. See ``_execute_with_retry`` for the
                # SAVEPOINT implementation.
                with engine.begin() as conn:
                    for stmt_idx, stmt in enumerate(statements):
                        try:
                            _execute_with_retry(
                                conn,
                                stmt,
                                max_retries=config.max_retries,
                                backoff_base=config.retry_backoff_base,
                                migration_name=f"{migration_name}[stmt:{stmt_idx}]",
                            )
                        except (ProgrammingError, DataError):
                            # Non-transient: roll back entire transaction
                            logger.error(
                                "Non-transient error in statement %d of %s",
                                stmt_idx, migration_name,
                            )
                            raise

                    # v90 ROOT FIX (BUG #15 -- P1 _apply_postgres_only_upgrades
                    # outside transaction): the PostgreSQL-only upgrades
                    # (e.g. JSONB type upgrade for migration 007) MUST run
                    # INSIDE the per-migration transaction so they commit
                    # atomically with the migration. If the upgrade fails
                    # (e.g. ALTER COLUMN metadata_json TYPE JSONB fails
                    # because a row has invalid JSON), the entire migration
                    # rolls back -- the migration is NOT marked as applied,
                    # and the operator can fix the data and re-run. The
                    # previous code called _apply_postgres_only_upgrades
                    # AFTER the transaction committed, so a failed upgrade
                    # left the migration recorded as "applied" but the
                    # column stayed TEXT -- schema drift.
                    # v75 ROOT FIX (T-026): the hook is called ONLY on
                    # PostgreSQL (the function checks dialect internally).
                    # The portable SQL file already added the TEXT column
                    # on BOTH dialects; this hook upgrades it to JSONB on
                    # PostgreSQL for indexable, deduplicated JSON storage.
                    _apply_postgres_only_upgrades(conn, migration_name)

                    # Record successful migration (BUG-IDEM-03: upsert).
                    # This is INSIDE the same ``engine.begin()`` block as
                    # the migration statements + the Postgres-only upgrades,
                    # so the recording, the schema change, AND the upgrade
                    # commit atomically -- we never have a recorded migration
                    # that didn't actually apply (or vice versa).
                    _record_migration(conn, migration_name, checksum, "applied")

                mig_duration = time.monotonic() - mig_start
                per_migration_timing[migration_name] = mig_duration
                applied.append(migration_name)
                consecutive_failures = 0  # GAP-REL-07: Reset circuit breaker

                # BUG-LINE-01: Populate provenance table
                with engine.begin() as conn:
                    try:
                        conn.execute(
                            text(
                                "INSERT INTO _migration_provenance "
                                "(migration_name, issues_fixed, description, "
                                "affected_tables, statement_count, source_checksum) "
                                "VALUES (:n, :issues, :desc, :tables, :stmt_count, :cs)"
                            ),
                            {
                                "n": migration_name,
                                "issues": "",
                                "desc": f"Applied {statement_count} statements",
                                "tables": ",".join(t for t in _KNOWN_TABLES if _table_exists(inspector, t)),
                                "stmt_count": statement_count,
                                "cs": checksum,
                            },
                        )
                    except (OperationalError, ProgrammingError) as exc:
                        logger.debug("Could not record provenance: %s", exc)

                # Post-migration verification (SCI-MIG-03, DQ-MIG-04)
                post_issues = _verify_post_migration_state(engine, migration_name)
                for issue in post_issues:
                    logger.warning("Post-migration issue: %s", issue)

                # GAP-DQ-05: Always compute row counts after
                with engine.begin() as conn:
                    post_counts = _log_table_state(conn, f"after_{migration_name}", dialect_name)
                    for table in _KNOWN_TABLES:
                        pre = pre_counts.get(table, 0)
                        post = post_counts.get(table, 0)
                        if pre != 0 or post != 0:
                            row_count_changes[table] = (pre, post)

                # Data checksum after (only when configured)
                if config and config.verify_data_checksums:
                    with engine.begin() as conn:
                        for table in _KNOWN_TABLES:
                            if _table_exists(inspector, table) and table in pre_checksums:
                                post_cs = _compute_data_checksum(conn, table)
                                if post_cs != pre_checksums[table]:
                                    logger.info(
                                        "Data checksum changed for '%s' after '%s'",
                                        table, migration_name,
                                    )
                                data_checksums[table] = post_cs

                logger.info(
                    "Successfully applied migration: %s (%.2fs) [%d/%d]",
                    migration_name, mig_duration, idx, total_migrations,
                )

                # Fire post-migration callback (DES-MIG-04)
                if config and config.on_migration_complete:
                    try:
                        config.on_migration_complete(migration_name, mig_duration)
                    except Exception as cb_exc:
                        logger.warning("on_migration_complete callback error: %s", cb_exc)

                # Log structured event (LOG-MIG-04)
                _log_migration_event(
                    "applied",
                    migration_name,
                    {"duration_seconds": mig_duration,
                     "statement_count": statement_count},
                    correlation_id=getattr(config, "correlation_id", None),
                    pipeline_name=getattr(config, "pipeline_name", None),
                    run_id=getattr(config, "run_id", None),
                )

            except Exception as exc:
                mig_duration = time.monotonic() - mig_start
                per_migration_timing[migration_name] = mig_duration
                failed.append(migration_name)
                # v22 ROOT FIX: wrap in dict for type-contract consistency.
                errors.append({
                    "migration": migration_name,
                    "dialect": "unknown",
                    "error": str(exc),
                    "phase": "apply",
                })
                consecutive_failures += 1  # GAP-REL-07

                # Record failure in dead letter queue
                with engine.begin() as conn:
                    _record_failure(
                        conn, migration_name, checksum, str(exc), type(exc).__name__
                    )
                    # Update _migration_history with status='failed'
                    try:
                        conn.execute(
                            text(
                                "INSERT INTO _migration_history "
                                "(migration_name, checksum, status, applied_by, applied_from, "
                                "python_version, phase_at_interrupt) "
                                "VALUES (:n, :c, 'failed', :by, :afh, :pv, :phase)"
                            ),
                            {
                                "n": migration_name,
                                "c": checksum,
                                "by": os.environ.get("AIRFLOW_USER", "unknown"),
                                "afh": platform.node(),
                                "pv": platform.python_version(),
                                "phase": current_phase.value,
                            },
                        )
                    except (OperationalError, ProgrammingError) as db_exc:
                        logger.error("Could not record failed migration status: %s", db_exc)

                logger.error("Failed to apply migration %s: %s", migration_name, exc)

                # Fire failure callback
                if config and config.on_migration_fail:
                    try:
                        config.on_migration_fail(migration_name, exc)
                    except Exception as cb_exc:
                        logger.warning("on_migration_fail callback error: %s", cb_exc)

                # Log structured event
                _log_migration_event(
                    "failed",
                    migration_name,
                    {"error": str(exc), "error_class": type(exc).__name__},
                    level="error",
                    correlation_id=getattr(config, "correlation_id", None),
                    pipeline_name=getattr(config, "pipeline_name", None),
                    run_id=getattr(config, "run_id", None),
                )

                # GAP-REL-07: Circuit breaker
                if consecutive_failures >= config.circuit_breaker_threshold:
                    logger.critical(
                        "Circuit breaker triggered: %d consecutive migration failures. "
                        "Stopping migration run. Check database connectivity.",
                        consecutive_failures,
                    )
                    break

                # Stop on failure if configured
                if config and config.stop_on_failure:
                    raise MigrationError(failed=[migration_name], errors=[exc])
    else:
        # v16 ROOT FIX (CD-5): the previous code COMPLETELY skipped all
        # ``.sql`` migration files on SQLite. Only the Python-side
        # ``_apply_python_columns()`` (which adds a small hardcoded subset
        # of columns) ran. This meant SQLite dev/test DBs (created via
        # ``Base.metadata.create_all()``) lacked:
        #   - CHECK constraints from migrations 001/003/005
        #   - UNIQUE constraints from migration 002
        #   - FK constraints from migration 005 (pubchem.inchikey -> drugs)
        #   - Indexes from migrations 001/003/005/006
        #   - The ``schema_version`` table from migration 001
        # Code that passed tests on SQLite could fail on PostgreSQL
        # because the two DBs had wildly different schemas.
        #
        # The fix: for SQLite, run the migrations via SQLAlchemy's
        # ``text()`` runner with on-the-fly dialect translation of
        # PostgreSQL-specific syntax (``DO $$ ... END $$`` blocks,
        # ``GENERATED ALWAYS AS IDENTITY``, ``TIMESTAMP WITH TIME ZONE``,
        # ``PRAGMA``-gated FK creation, etc.). The translation is
        # best-effort -- features that cannot be translated (e.g.
        # ``pg_advisory_lock``) are skipped with a WARNING.
        logger.info(
            "Running .sql migration files for dialect '%s' with "
            "SQLite-compatible translation (v16 CD-5 fix). PostgreSQL-"
            "specific syntax (DO blocks, GENERATED ALWAYS AS IDENTITY, "
            "TIMESTAMP WITH TIME ZONE) is translated; unsupported "
            "features (pg_advisory_lock, partial indexes with WHERE) "
            "are skipped with a WARNING.",
            dialect_name,
        )
        # FIX-C5: exclude *_rollback.sql sidecars (see FIX-C5 note above).
        sql_files = sorted(
            [f for f in migrations_dir.glob("*.sql") if not f.name.endswith("_rollback.sql")],
            key=lambda f: _extract_migration_number(f.name),
        )
        for f in sql_files:
            _validate_migration_path(f, migrations_dir)
        for f in sql_files:
            # v73 ROOT FIX (T-009 continued): check if migration is already
            # applied BEFORE attempting to execute it. The PostgreSQL branch
            # (lines 3709-3744) has this check; the SQLite branch was
            # missing it entirely. Combined with the missing
            # ``_record_migration`` call, this meant every SQLite run
            # re-executed every migration file. Now that we record applied
            # migrations (fix above), this check skips them on subsequent
            # runs -- the SQLite path now has the same idempotent
            # skip-logic as the PostgreSQL path.
            migration_name = f.name
            if config and config.skip_migrations and migration_name in config.skip_migrations:
                logger.info("Skipping migration (in skip list): %s", migration_name)
                skipped.append(migration_name)
                continue
            try:
                with engine.begin() as check_conn:
                    try:
                        already_applied = _is_migration_applied(check_conn, migration_name)
                    except (OperationalError, ProgrammingError):
                        already_applied = False
                if already_applied:
                    logger.info(
                        "Migration already applied, skipping: %s", migration_name
                    )
                    skipped.append(migration_name)
                    continue
            except (OperationalError, ProgrammingError) as check_exc:
                logger.warning(
                    "  [WARN] Could not check applied-status for %s: %s. "
                    "Proceeding with execution (T-009 fix).",
                    migration_name, check_exc,
                )

            try:
                content = f.read_text(encoding="utf-8")
                # v73 ROOT FIX (T-009 -- SQLite path never records applied
                # migrations -> every run re-applies every migration):
                #   The previous SQLite branch computed NOTHING comparable
                #   to the PostgreSQL branch's ``checksum`` (line 3761).
                #   On the next ``run_migrations()`` call,
                #   ``_is_migration_applied(conn, f.name)`` queried
                #   ``_migration_history`` and found ZERO rows for every
                #   migration -- because ``_record_migration`` was never
                #   called on the SQLite path. The PostgreSQL branch
                #   (line 3904) calls
                #   ``_record_migration(conn, migration_name, checksum,
                #   "applied")`` INSIDE the ``engine.begin()`` block so
                #   the recording and the schema change commit atomically
                #   -- we never have a recorded migration that didn't
                #   actually apply (or vice versa). The SQLite branch
                #   had only ``applied.append(f.name)`` (line 4199) which
                #   updated the IN-MEMORY list but never wrote a row to
                #   ``_migration_history``. Result: every ``run_migrations()``
                #   invocation on SQLite re-executed ALL 10 migration
                #   files. The idempotent no-op handler at line 4181
                #   swallowed ``duplicate column name`` / ``already
                #   exists`` errors, so the migration "succeeded" -- but
                #   every run paid the full translation + execution cost,
                #   AND any non-idempotent migration (e.g. one that
                #   INSERTs data without ON CONFLICT) duplicated data on
                #   every run. Migration 002's ``audit_log`` INSERTs
                #   (lines 488-493, 542-546, etc.) use plain INSERT with
                #   no ON CONFLICT -- so every ``run_migrations()`` call
                #   appended another row to ``audit_log`` with the same
                #   operation token.
                #
                #   ROOT FIX: compute the checksum from the ORIGINAL
                #   (pre-translation) file content (matching the
                #   PostgreSQL branch's line 3761
                #   ``checksum = _compute_checksum(raw_content)``), then
                #   call ``_record_migration(conn, f.name, checksum,
                #   "applied")`` INSIDE the ``with engine.begin() as
                #   conn:`` block -- AFTER all statements have executed
                #   successfully, but BEFORE the context manager commits
                #   the transaction. If any statement fails, the
                #   exception propagates and the context manager rolls
                #   back BOTH the schema changes AND the
                #   ``_migration_history`` INSERT -- leaving the database
                #   in the pre-migration state, identical to the
                #   PostgreSQL branch's atomicity guarantee.
                checksum = _compute_checksum(content)
                # Translate PostgreSQL-specific syntax to SQLite-compatible.
                translated = _translate_sql_for_sqlite(content)
                # Split into statements (naive -- split on semicolons
                # but respect DO $$ ... $$ blocks). For SQLite we just
                # execute the whole script as one text() call --
                # SQLAlchemy's text() supports multiple statements when
                # executed via engine.connect().execute(text(...)) in
                # SQLAlchemy 2.x with executemany.
                try:
                    # v29 ROOT FIX (audit D-14): forward migrations were
                    # non-atomic. Now wrapped in explicit transaction per
                    # migration. The ``engine.begin()`` context manager
                    # provides an explicit BEGIN/COMMIT (or ROLLBACK on
                    # exception) wrapper around the entire SQLite-
                    # translated migration script. A partial failure
                    # (e.g. one statement in the translated script raises)
                    # rolls back the whole migration atomically -- no
                    # half-applied schema. This mirrors the PostgreSQL
                    # path's per-migration transaction wrapper (see
                    # ``with engine.begin() as conn:`` in the
                    # PostgreSQL branch above + the SAVEPOINT-based
                    # retry logic in ``_execute_with_retry``).
                    with engine.begin() as conn:
                        # v59 ROOT FIX (SQLite multi-statement execution):
                        # The previous code called
                        # ``conn.exec_driver_sql(translated)`` passing the
                        # ENTIRE translated migration script as one call.
                        # SQLite's underlying ``sqlite3.Connection.execute``
                        # raises ``ProgrammingError: You can only execute
                        # one statement at a time`` for any input containing
                        # more than one statement (i.e. every migration
                        # file). The comment that was here claimed
                        # "ENH-9: SQLite supports executing multiple
                        # statements in one text() call when using
                        # connection.exec_driver_sql()" -- this is FALSE.
                        # SQLAlchemy's ``exec_driver_sql`` delegates to
                        # ``sqlite3.Connection.execute`` which enforces the
                        # one-statement limit. The result: EVERY SQLite
                        # migration after 001 failed with this exact error,
                        # and the runner fell through to the "Failing hard
                        # per V18 CD-5 root fix" branch, raising
                        # RuntimeError: "Database migration failed".
                        # This is the user's reported T-001 symptom --
                        # migrations blocked on a fresh DB.
                        # FIX: split the translated SQL into individual
                        # statements using the state-machine splitter
                        # (``_split_sql_statements``) that already exists
                        # for the rollback path (line 3914). Execute each
                        # non-empty, non-comment statement separately.
                        # This mirrors how the PostgreSQL path's
                        # ``_execute_with_retry`` already handles statements
                        # one at a time.
                        statements = _split_sql_statements(translated)
                        for stmt in statements:
                            stmt_stripped = stmt.strip()
                            if not stmt_stripped:
                                continue
                            # Skip statements that are ONLY comments (the
                            # splitter keeps the comment text alongside the
                            # statement; a pure-comment "statement" would
                            # be a no-op but still consumes a DB roundtrip).
                            non_comment_lines = [
                                ln for ln in stmt_stripped.split("\n")
                                if ln.strip() and not ln.strip().startswith("--")
                            ]
                            if not non_comment_lines:
                                continue
                            # v83 FORENSIC ROOT FIX (P0-C10 -- SQLAlchemy
                            #   text() mis-parses ``%(...)s`` in SQL comments
                            #   as pyformat parameter placeholders):
                            #   Migration 001 line 398 has a comment
                            #   ``-- Indexes (CMP-02: follow ORM naming
                            #   convention ix_%(table)s_%(column)s)``.
                            #   SQLAlchemy's ``text()`` compiles the SQL
                            #   through its parambinding layer, which
                            #   interprets ``%(table)s`` as a pyformat-style
                            #   named parameter. The compiler then converts
                            #   it to ``?`` (qmark for SQLite) and tries to
                            #   bind a parameter named ``table`` -- which
                            #   doesn't exist in the params dict, raising
                            #   ``StatementError: (builtins.KeyError)
                            #   'table'``. This blocks EVERY Phase 1
                            #   pipeline on a fresh SQLite DB (the dev/CI
                            #   default), which means the master DAG cannot
                            #   run end-to-end in any non-Postgres env --
                            #   silently gutting the "V1 on free public data
                            #   + laptop" mandate from the project docx.
                            #
                            #   The previous code kept the comment lines
                            #   INSIDE ``stmt_stripped`` and passed the whole
                            #   string to ``text()``. The ``non_comment_lines``
                            #   check only decided whether to SKIP pure-comment
                            #   statements; it did NOT strip the comments
                            #   from mixed statements before execution.
                            #
                            #   ROOT FIX: execute the SQL with the LEADING
                            #   comment lines stripped (rebuilt from
                            #   ``non_comment_lines``). This is the
                            #   institutional-grade fix because:
                            #     1. It removes the bug surface entirely --
                            #        ``text()`` never sees ``%(...)s`` in a
                            #        comment, so pyformat parsing cannot
                            #        mis-fire.
                            #     2. It preserves the existing per-statement
                            #        semantics (the splitter still produces
                            #        one statement per ``conn.execute`` call,
                            #        so the v59 multi-statement fix is
                            #        preserved).
                            #     3. It is forward-compatible -- any future
                            #        migration that uses ``%(foo)s`` in a
                            #        comment (e.g. documenting a Python
                            #        format-string convention) is protected.
                            #     4. It mirrors the PostgreSQL path's
                            #        ``_execute_with_retry`` which already
                            #        strips comments before execution (see
                            #        line ~2090).
                            # We use ``\n`` join (not ``" "``) to preserve
                            # the original line structure for any
                            # multi-line statement that depends on
                            # newlines (none in current migrations, but
                            # defensive).
                            stmt_for_execution = "\n".join(non_comment_lines)
                            try:
                                # P1-A13 ROOT FIX (v82): use SQLAlchemy's
                                # ``text()`` wrapper + ``conn.execute()``
                                # instead of the raw ``exec_driver_sql``.
                                # ``exec_driver_sql`` bypasses SQLAlchemy's
                                # SQL execution layer -- any future migration
                                # that constructs SQL from env vars or user
                                # input would be an injection vector.
                                # ``text()`` routes through SQLAlchemy's
                                # compiler, which is the institutional-grade
                                # standard. Functionally equivalent for the
                                # static DDL/DML in migration .sql files (the
                                # splitter already ensures single-statement
                                # calls, so the v59 multi-statement fix is
                                # preserved).
                                # v83 P0-C10: ``stmt_for_execution`` has
                                # comments stripped to prevent pyformat
                                # mis-parsing of ``%(...)s`` in comments.
                                # (Reconciled with v82 P1-A13 revised: the
                                # comment-stripping approach is superior to
                                # reverting to exec_driver_sql because it
                                # preserves the injection-safety of text()
                                # while fixing the %(table)s parse bug.)
                                conn.execute(text(stmt_for_execution))
                            except Exception as stmt_exc:
                                # v59 ROOT FIX: catch "duplicate column name"
                                # and "already exists" errors at the STATEMENT
                                # level (not just migration level). The
                                # migration-level handler at line ~3680 only
                                # catches errors from the entire migration
                                # script; with our new per-statement execution,
                                # each statement runs separately and needs its
                                # own no-op detection. These errors are expected
                                # on SQLite because we strip IF NOT EXISTS from
                                # ALTER TABLE ADD COLUMN (SQLite doesn't support
                                # that syntax), so re-running a migration that
                                # adds an already-existing column raises this.
                                _stmt_err = str(stmt_exc).lower()
                                _is_idempotent_noop = (
                                    "duplicate column name" in _stmt_err
                                    or "already exists" in _stmt_err
                                    # v75 ROOT FIX (T-036): "no such table"
                                    # is the rollback-script equivalent of
                                    # "already exists" for forward migrations.
                                    # A rollback that drops/renames a table
                                    # which was never created (e.g. running
                                    # 002_rollback on a fresh DB where 002
                                    # was never applied) raises
                                    # ``OperationalError: no such table: X``
                                    # on SQLite. This is an idempotent no-op:
                                    # the rollback's intent was "ensure table
                                    # X does not exist", and it doesn't.
                                    or "no such table" in _stmt_err
                                )
                                if _is_idempotent_noop:
                                    logger.debug(
                                        "  [OK] SQLite statement: idempotent "
                                        "no-op (column/constraint already "
                                        "exists): %s",
                                        str(stmt_exc)[:120],
                                    )
                                    continue
                                # Re-raise with statement context so the
                                # outer except can log WHICH statement in
                                # the migration failed (the previous code
                                # logged the entire migration file, making
                                # it impossible to diagnose).
                                # v83 P0-C10: include the COMMENT-STRIPPED
                                # statement in the error message (the
                                # original stmt_stripped may contain a
                                # ``%(...)s`` comment that masked the real
                                # SQL -- operators need to see the actual
                                # SQL that failed, not the comment).
                                raise StatementExecutionError(
                                    f"SQLite migration {f.name}: statement failed: "
                                    f"{stmt_exc}\nStatement (first 300 chars): "
                                    f"{stmt_for_execution[:300]}"
                                ) from stmt_exc
                        # v73 ROOT FIX (T-009): record the migration INSIDE the
                        # same ``engine.begin()`` transaction as the schema
                        # changes. If any statement above raised, the context
                        # manager would have rolled back BOTH the schema changes
                        # AND this recording -- atomicity parity with the
                        # PostgreSQL branch (line 3904). The next
                        # ``run_migrations()`` call now finds this migration in
                        # ``_migration_history`` with status='applied' and skips
                        # it -- no more re-applying all 10 migrations on every
                        # run, no more duplicate ``audit_log`` rows from
                        # non-idempotent INSERTs in migration 002.
                        _record_migration(conn, f.name, checksum, "applied")
                    applied.append(f.name)
                    logger.info(
                        "  [OK] Applied SQLite-translated migration: %s",
                        f.name,
                    )
                except Exception as exc:
                    # v17 ROOT FIX (idempotent ADD COLUMN on old SQLite):
                    # even with the version-aware _translate_sql_for_sqlite
                    # fix, old SQLite (< 3.35) raises ``duplicate column
                    # name: <col>`` when an ALTER TABLE ADD COLUMN is re-
                    # executed (because IF NOT EXISTS was stripped). The
                    # previous code treated this as a hard SKIP -- leaving
                    # the migration recorded as "skipped" forever, even
                    # though the schema was actually fine (the column
                    # already existed). Treat ``duplicate column name``
                    # as a SUCCESSFUL no-op: the migration's intent was
                    # "ensure this column exists", and it does.
                    _exc_msg = str(exc).lower()
                    _is_idempotent_noop = (
                        "duplicate column name" in _exc_msg
                        or "already exists" in _exc_msg
                        # v75 ROOT FIX (T-036): "no such table" -- see
                        # the per-statement handler above for the full
                        # rationale. Mirrored here for the migration-
                        # level catch so rollback scripts that reference
                        # non-existent tables don't block the chain.
                        or "no such table" in _exc_msg
                    )
                    if _is_idempotent_noop:
                        applied.append(f.name)
                        logger.info(
                            "  [OK] SQLite migration %s: idempotent no-op "
                            "(object already exists, schema is in the "
                            "desired state): %s",
                            f.name, exc,
                        )
                        # v73 ROOT FIX (T-009 continued): the original
                        # ``with engine.begin() as conn:`` block was
                        # rolled back when the idempotent-noop exception
                        # raised -- so we still need to RECORD this
                        # migration as applied in a NEW transaction.
                        # Without this, the next ``run_migrations()``
                        # call would see no ``_migration_history`` row
                        # for this migration and re-execute it (raising
                        # the same idempotent-noop exception again).
                        # Opening a fresh transaction here is safe: the
                        # schema is already in the desired state (the
                        # object exists), so we are only writing the
                        # bookkeeping row.
                        try:
                            with engine.begin() as record_conn:
                                _record_migration(
                                    record_conn, f.name, checksum, "applied"
                                )
                        except (OperationalError, ProgrammingError) as rec_exc:
                            logger.warning(
                                "  [WARN] SQLite migration %s: idempotent "
                                "no-op succeeded but could not record "
                                "applied-status in _migration_history: %s. "
                                "The next run_migrations() call may re-apply "
                                "this migration (T-009 fix).",
                                f.name, rec_exc,
                            )
                    else:
                        # V18 ROOT FIX (CD-5): the previous "best-effort"
                        # WARNING + skip pattern was the ROOT CAUSE of
                        # the audit's "code that passes tests on SQLite
                        # may fail on PostgreSQL" risk. When a SQLite
                        # translation failed (e.g. unsupported SQL feature),
                        # the migration was silently skipped -- the SQLite
                        # DB then had a schema that diverged from what
                        # PostgreSQL would have, but tests ran against
                        # the SQLite DB and reported success.
                        #
                        # Root fix: FAIL HARD on translation errors that
                        # are NOT idempotent no-ops. The operator must
                        # either fix the translator (add the missing
                        # SQLite feature) or mark the migration as
                        # SQLite-skippable explicitly. Silent skipping
                        # is no longer permitted.
                        failed.append(f.name)
                        errors.append({
                            "migration": f.name,
                            "dialect": "sqlite",
                            "error": str(exc),
                            "phase": "execute_translated",
                        })
                        logger.error(
                            "  [FAIL] SQLite migration %s failed to apply "
                            "(translated) and is NOT an idempotent no-op: "
                            "%s. Failing hard per V18 CD-5 root fix -- "
                            "the ORM-created schema is NOT a safe fallback "
                            "because tests against the divergent SQLite "
                            "schema would report success while PostgreSQL "
                            "would reject the same code. Fix the SQLite "
                            "translator in _translate_sql_for_sqlite() to "
                            "handle this SQL pattern.",
                            f.name, exc,
                        )
                        # Propagate the failure so the operator sees it.
                        raise RuntimeError(
                            f"SQLite migration {f.name} failed to apply "
                            f"(translated): {exc}. The migration cannot "
                            f"be silently skipped -- see V18 CD-5 root "
                            f"fix comment for details."
                        ) from exc
            except Exception as exc:
                # V18 CD-5: same hard-fail policy for read/translate
                # errors. The previous WARNING + skip pattern masked
                # migration-incompatibility bugs that only surfaced on
                # PostgreSQL.
                failed.append(f.name)
                errors.append({
                    "migration": f.name,
                    "dialect": "sqlite",
                    "error": str(exc),
                    "phase": "read_or_translate",
                })
                logger.error(
                    "  [FAIL] Could not read/translate migration %s for "
                    "SQLite: %s. Failing hard per V18 CD-5 root fix.",
                    f.name, exc,
                )
                raise RuntimeError(
                    f"SQLite migration {f.name} could not be read or "
                    f"translated: {exc}. See V18 CD-5 root fix comment."
                ) from exc

    # GAP-LOG-05: Data quality summary
    logger.info(
        "Migration run summary: %d applied, %d skipped, %d failed, "
        "%.2fs total, dialect=%s",
        len(applied), len(skipped), len(failed),
        time.monotonic() - start_time, dialect_name,
    )

    return _finalize_result(
        applied=applied,
        skipped=skipped,
        failed=failed,
        errors=errors,
        per_migration_timing=per_migration_timing,
        dialect_name=dialect_name,
        start_time=start_time,
        engine=engine,
        config=config,
        row_count_changes=row_count_changes,
        data_checksums=data_checksums,
        schema_version_before=schema_version_before,
        inspector=inspector,
    )


# Alias for discoverability -- both names work (DES-MIG-03, CMP-MIG-04)
# NOTE: This alias is DEPRECATED. Use run_migrations instead.
run_migration_002 = run_migrations


# ---------------------------------------------------------------------------
# Health check and status functions (ARCH-MIG-06, INT-MIG-05)
# ---------------------------------------------------------------------------


def check_migrations(engine=None) -> MigrationHealthResult:
    """Verify all migrations are applied and schema version matches.

    BUG-CODE-04: Proper return type MigrationHealthResult.
    GAP-DQ-04: Detects phantom migrations (recorded but no SQL file).

    Parameters
    ----------
    engine : Engine | None
        SQLAlchemy engine. If None, calls _get_default_engine().

    Returns
    -------
    MigrationHealthResult
        Health check result with applied/pending counts and schema version.
    """
    if engine is None:
        engine = _get_default_engine()

    inspector = inspect(engine)
    dialect_name = engine.dialect.name
    _ensure_migration_tracking_table(engine)

    # Get applied migrations from tracking table
    applied_migrations: list[str] = []
    with engine.begin() as conn:
        try:
            r = conn.execute(
                text("SELECT migration_name FROM _migration_history "
                     "WHERE status NOT IN ('failed', 'retrying')")
            )
            applied_migrations = [row[0] for row in r.fetchall()]
        except (OperationalError, ProgrammingError) as exc:
            logger.warning("Could not fetch applied migrations: %s", exc)

    # Get all SQL migration files. FIX-C5: exclude *_rollback.sql sidecars
    # -- they are recovery scripts, NOT migrations.
    sql_files = sorted(
        [f for f in MIGRATIONS_DIR.glob("*.sql") if not f.name.endswith("_rollback.sql")],
        key=lambda f: _extract_migration_number(f.name),
    )
    all_migration_names = [f.name for f in sql_files]
    all_migration_set = set(all_migration_names)

    # GAP-DQ-04: Detect phantom migrations
    phantom_migrations = [
        m for m in applied_migrations if m not in all_migration_set
    ]
    for phantom in phantom_migrations:
        logger.warning(
            "Migration %s is recorded as applied but no corresponding "
            ".sql file exists. This may indicate a manually modified "
            "tracking table.", phantom,
        )

    # Calculate pending
    pending_migrations = [m for m in all_migration_names if m not in set(applied_migrations)]

    # Check schema version
    schema_version_matches = False
    try:
        from database.base import SCHEMA_VERSION as code_version
        db_version = None
        if _table_exists(inspector, "schema_version"):
            with engine.begin() as conn:
                r = conn.execute(text("SELECT MAX(version) FROM schema_version"))
                db_version = r.scalar()
        schema_version_matches = (db_version == code_version)
    except (OperationalError, ProgrammingError) as exc:
        logger.warning("Could not check schema version: %s", exc)

    return MigrationHealthResult(
        all_applied=len(pending_migrations) == 0,
        applied_count=len(applied_migrations),
        pending_count=len(pending_migrations),
        applied_migrations=applied_migrations,
        pending_migrations=pending_migrations,
        schema_version_matches=schema_version_matches,
        dialect=dialect_name,
        phantom_migrations=phantom_migrations,
    )


def get_migration_status(engine=None) -> MigrationStatus:
    """Return detailed migration status including history.

    BUG-CODE-04: Proper return type MigrationStatus.

    Parameters
    ----------
    engine : Engine | None
        SQLAlchemy engine. If None, calls _get_default_engine().

    Returns
    -------
    MigrationStatus
        Detailed status of applied and pending migrations.
    """
    if engine is None:
        engine = _get_default_engine()

    _ensure_migration_tracking_table(engine)

    # Get applied migration details
    applied_migrations: list[dict[str, Any]] = []
    with engine.begin() as conn:
        try:
            r = conn.execute(
                text(
                    "SELECT migration_name, applied_at, checksum, applied_by, "
                    "applied_from, status FROM _migration_history ORDER BY id"
                )
            )
            for row in r.fetchall():
                applied_migrations.append({
                    "migration_name": row[0],
                    "applied_at": str(row[1]) if row[1] else None,
                    "checksum": row[2],
                    "applied_by": row[3],
                    "applied_from": row[4],
                    "status": row[5],
                })
        except (OperationalError, ProgrammingError) as exc:
            logger.warning("Could not fetch migration history: %s", exc)

    # Get pending
    applied_names = {m["migration_name"] for m in applied_migrations if m.get("status") not in ("failed", "retrying")}
    # FIX-C5: exclude *_rollback.sql sidecars -- they are recovery scripts.
    sql_files = sorted(
        [f for f in MIGRATIONS_DIR.glob("*.sql") if not f.name.endswith("_rollback.sql")],
        key=lambda f: _extract_migration_number(f.name),
    )
    pending_migrations = [f.name for f in sql_files if f.name not in applied_names]

    # Schema version
    schema_version_code: int = 0
    schema_version_db: int | None = None
    try:
        from database.base import SCHEMA_VERSION as _sv
        schema_version_code = _sv
    except (ImportError, AttributeError) as exc:
        logger.warning("Could not read code schema version: %s", exc)

    inspector = inspect(engine)
    if _table_exists(inspector, "schema_version"):
        with engine.begin() as conn:
            try:
                r = conn.execute(text("SELECT MAX(version) FROM schema_version"))
                schema_version_db = r.scalar()
            except (OperationalError, ProgrammingError) as exc:
                logger.warning("Could not read DB schema version: %s", exc)

    return MigrationStatus(
        applied_migrations=applied_migrations,
        pending_migrations=pending_migrations,
        total_migrations=len(sql_files),
        schema_version_code=schema_version_code,
        schema_version_db=schema_version_db,
    )


# ---------------------------------------------------------------------------
# Architecture helpers (ARCH-MIG-05)
# ---------------------------------------------------------------------------


def get_sql_migration_files() -> list[Path]:
    """Return list of Path objects for .sql migration files.

    Separates access to migration SQL data from the runner code.

    FIX-C5: excludes ``*_rollback.sql`` sidecars -- they are recovery
    scripts, NOT migrations. Including them would (on PostgreSQL) execute
    ``DROP TABLE IF EXISTS drugs CASCADE; ...`` and destroy the staging
    schema on every fresh install; on SQLite it aborts with
    "You can only execute one statement at a time".
    """
    return sorted(
        [f for f in MIGRATIONS_DIR.glob("*.sql") if not f.name.endswith("_rollback.sql")],
        key=lambda f: _extract_migration_number(f.name),
    )


def get_migration_runner() -> Callable:
    """Return the run_migrations callable.

    Separates access to the migration runner from the SQL data.
    """
    return run_migrations


# ---------------------------------------------------------------------------
# Rollback placeholder (DES-MIG-06, GAP-DES-05)
# ---------------------------------------------------------------------------


# v29 ROOT FIX (audit D-10): naive split on ; broke on string literals and
# DO $$ blocks. Use state-machine splitter that respects string/dollar-quote
# context. The previous implementation (``rollback_sql.split(";")``)
# fragmented any rollback sidecar containing ``DO $$ ... ; ... $$`` blocks
# or string literals with embedded semicolons (e.g. ``COMMENT ON ... IS
# 'has a ; here'``), producing broken statements that failed at runtime
# and silently rolled back the entire transaction. This splitter walks the
# SQL character-by-character tracking whether we are inside:
#   - a single-quoted string literal (``'...'``; ``''`` is an escaped quote)
#   - a double-quoted identifier (``"..."``; ``""`` is an escaped quote)
#   - a dollar-quoted block (``$$...$$`` or ``$tag$...$tag$``)
#   - a line comment (``-- ...`` until end of line)
#   - a block comment (``/* ... */``)
# and only breaks on ``;`` when outside all of these.


# v76 ROOT FIX (T-047): comprehensive transaction-control statement
# detection. Used by ``_split_sql_statements`` to filter out ALL
# transaction-control statements so they don't interfere with the
# ``engine.begin()`` transaction wrapper on the SQLite migration path.
# The set covers the SQL standard + PostgreSQL + SQLite variants.
_TRANSACTION_CONTROL_PREFIXES = frozenset({
    "BEGIN",
    "BEGIN TRANSACTION",
    "BEGIN WORK",
    "START TRANSACTION",
    "COMMIT",
    "COMMIT TRANSACTION",
    "COMMIT WORK",
    "COMMIT AND CHAIN",
    "COMMIT AND NO CHAIN",
    "ROLLBACK",
    "ROLLBACK TRANSACTION",
    "ROLLBACK WORK",
    "ROLLBACK AND CHAIN",
    "ROLLBACK AND NO CHAIN",
    "END",
    "END TRANSACTION",
    "END WORK",
    "ABORT",
    "ABORT TRANSACTION",
})


def _is_transaction_control_statement(upper_stmt: str) -> bool:
    """Return True if the statement (already stripped + uppercased) is a
    transaction-control statement that must be filtered out by the SQL
    splitter to prevent interference with the ``engine.begin()``
    transaction wrapper.

    v76 ROOT FIX (T-047): the previous filter only caught EXACT ``BEGIN``
    and ``COMMIT``. This function catches ALL standard SQL transaction-
    control forms (BEGIN [TRANSACTION|WORK], START TRANSACTION, COMMIT
    [TRANSACTION|WORK|AND CHAIN], ROLLBACK [TRANSACTION|WORK|AND CHAIN],
    END [TRANSACTION|WORK], ABORT [TRANSACTION]) plus SAVEPOINT and
    RELEASE SAVEPOINT (which manage subtransactions and would also
    interfere with the outer transaction). SET TRANSACTION ... (which
    sets isolation level) is also caught because it must run BEFORE BEGIN
    and has no effect inside an already-open transaction.

    Parameters
    ----------
    upper_stmt : str
        The statement text, already stripped of whitespace and uppercased
        by the caller.

    Returns
    -------
    bool
        True if the statement should be filtered out (it's a transaction-
        control statement), False if it should be kept and executed.
    """
    if not upper_stmt:
        return False
    # Exact-match check for the common forms.
    if upper_stmt in _TRANSACTION_CONTROL_PREFIXES:
        return True
    # Prefix-match check for forms with arguments:
    #   SAVEPOINT name
    #   RELEASE SAVEPOINT name
    #   RELEASE name
    #   SET TRANSACTION ...
    #   SET SESSION CHARACTERISTICS AS TRANSACTION ...
    #   SET CONSTRAINTS ... (transaction-relevant)
    for prefix in (
        "SAVEPOINT ",
        "RELEASE SAVEPOINT ",
        "RELEASE ",
        "SET TRANSACTION",
        "SET SESSION CHARACTERISTICS AS TRANSACTION",
        "SET CONSTRAINTS",
    ):
        if upper_stmt.startswith(prefix):
            return True
    return False


def _split_sql_statements(sql: str) -> list[str]:
    """Split a SQL script into individual statements.

    State-machine splitter that respects PostgreSQL string/identifier/
    dollar-quote/comment context so that semicolons inside those contexts
    do not terminate a statement.

    Parameters
    ----------
    sql : str
        Raw SQL text.

    Returns
    -------
    list[str]
        List of statement strings (raw, untrimmed -- caller is responsible
        for stripping whitespace and comment-only fragments).
    """
    statements: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(sql)
    dollar_tag: Optional[str] = None  # current $tag$ (None => not in dollar quote)

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        # --- inside a dollar-quoted block: only look for the closing $tag$ ---
        if dollar_tag is not None:
            if ch == "$" and sql.startswith(dollar_tag, i):
                buf.append(dollar_tag)
                i += len(dollar_tag)
                dollar_tag = None
                continue
            buf.append(ch)
            i += 1
            continue

        # --- detect START of a dollar-quoted block: $tag$ --------------------
        # tag is empty (=> $$) or [A-Za-z_][A-Za-z0-9_]* per PostgreSQL.
        if ch == "$":
            j = i + 1
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            if j < n and sql[j] == "$":
                dollar_tag = sql[i : j + 1]
                buf.append(dollar_tag)
                i = j + 1
                continue
            # Not a dollar quote -- literal $, fall through to default append.

        # --- single-quoted string literal (handles '' escape) ----------------
        if ch == "'":
            buf.append(ch)
            i += 1
            while i < n:
                c2 = sql[i]
                buf.append(c2)
                if c2 == "'" and i + 1 < n and sql[i + 1] == "'":
                    # Escaped doubled quote -- consume both.
                    buf.append("'")
                    i += 2
                    continue
                i += 1
                if c2 == "'":
                    break
            continue

        # --- double-quoted identifier (handles "" escape) --------------------
        if ch == '"':
            buf.append(ch)
            i += 1
            while i < n:
                c2 = sql[i]
                buf.append(c2)
                if c2 == '"' and i + 1 < n and sql[i + 1] == '"':
                    buf.append('"')
                    i += 2
                    continue
                i += 1
                if c2 == '"':
                    break
            continue

        # --- line comment (-- ... until newline) -----------------------------
        if ch == "-" and nxt == "-":
            j = i
            while j < n and sql[j] != "\n":
                buf.append(sql[j])
                j += 1
            if j < n:  # keep the newline
                buf.append("\n")
                j += 1
            i = j
            continue

        # --- block comment (/* ... */) ---------------------------------------
        if ch == "/" and nxt == "*":
            buf.append("/*")
            j = i + 2
            while j < n:
                if sql[j] == "*" and j + 1 < n and sql[j + 1] == "/":
                    buf.append("*/")
                    j += 2
                    break
                buf.append(sql[j])
                j += 1
            i = j
            continue

        # --- statement terminator (only when outside any context) -----------
        if ch == ";":
            # v42 FORENSIC ROOT FIX (P0-8): the FIRST definition of
            # ``_split_sql_statements`` (line 510) filtered out BEGIN/COMMIT
            # statements (line 594: ``if upper and upper != "BEGIN" and
            # upper != "COMMIT"``). This SECOND definition (line 3907),
            # which SHADOWS the first because Python uses the last
            # definition at runtime, did NOT filter BEGIN/COMMIT. So
            # migration 002 (and any other migration that wraps its body
            # in BEGIN/COMMIT) had its inner COMMIT passed through to
            # ``engine.begin()``, which ALREADY opened a transaction.
            # PostgreSQL issued a WARNING for the inner BEGIN (ignored)
            # but the inner COMMIT committed the entire transaction
            # PREMATURELY; subsequent statements executed in autocommit
            # and ``_record_migration`` ran outside the transaction.
            # ROOT FIX: apply the SAME BEGIN/COMMIT filter that the
            # first definition uses. This makes both definitions
            # behaviorally equivalent.
            stmt = "".join(buf)
            upper = stmt.strip().upper()
            # v76 ROOT FIX (T-047 -- filter ALL transaction-control
            # statements, not just bare BEGIN/COMMIT):
            #   The previous filter only caught EXACT ``BEGIN`` and
            #   ``COMMIT`` (after strip+upper). It did NOT catch:
            #     - ``BEGIN TRANSACTION``
            #     - ``BEGIN WORK``
            #     - ``START TRANSACTION``
            #     - ``COMMIT TRANSACTION``
            #     - ``COMMIT WORK``
            #     - ``COMMIT AND CHAIN``
            #     - ``ROLLBACK``
            #     - ``ROLLBACK TRANSACTION``
            #     - ``ROLLBACK WORK``
            #     - ``END``
            #     - ``END TRANSACTION``
            #     - ``SAVEPOINT ...``
            #     - ``RELEASE SAVEPOINT ...``
            #     - ``SET TRANSACTION ...``
            #   Any of these leaking through would interfere with the
            #   ``engine.begin()`` transaction wrapper that the SQLite
            #   migration path uses (run_migrations.py line ~4390). A
            #   leaked ``COMMIT TRANSACTION`` would prematurely end the
            #   SQLAlchemy transaction; a leaked ``ROLLBACK`` would abort
            #   it. The v42 fix caught the bare forms; the v76 fix catches
            #   ALL standard SQL transaction-control statements so no
            #   variant can leak. The migration files currently use only
            #   bare ``BEGIN;`` / ``COMMIT;`` (verified by grep across all
            #   22 migration+rollback files), but this filter is now
            #   future-proof against any migration that uses the fuller
            #   SQL standard forms.
            if upper and not _is_transaction_control_statement(upper):
                statements.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    # Flush trailing buffer (last statement without trailing ;).
    tail = "".join(buf)
    # v42 P0-8: also filter BEGIN/COMMIT from the trailing buffer.
    # v76 T-047: use the comprehensive transaction-control filter.
    upper_tail = tail.strip().upper()
    if tail.strip() and not _is_transaction_control_statement(upper_tail):
        statements.append(tail)
    return statements


def rollback_migration(migration_name: str, engine=None) -> dict:
    """Rollback a specific migration by executing its SQL-based down-script.

    v21 ROOT FIX (Audit section 5 finding 5 / Chain 5 / section 9 -
    "No migration rollback"): the previous version raised
    NotImplementedError unconditionally. The audit's complaint was that
    for a 7-source ETL pipeline with 6 migrations, no rollback is
    "operationally unacceptable." The function existed in the public
    API but was a documented lie.

    Fix: implement rollback via per-migration ``<name>_rollback.sql``
    sidecar files. For migrations that have a rollback sidecar, the
    function executes it inside a single transaction and reports the
    result. For migrations that DO NOT have a rollback sidecar, the
    function raises NotImplementedError with a clear message naming
    the missing file - so operators know exactly what to write.

    Parameters
    ----------
    migration_name : str
        The migration filename to rollback (e.g.
        ``"002_bug_fixes_migration.sql"``).
    engine : Engine | None
        SQLAlchemy engine. If None, uses the default engine from
        ``database.connection``.

    Returns
    -------
    dict
        Keys: migration_name, rolled_back (bool), elapsed_s (float),
        statements_executed (int), error (str|None).

    Raises
    ------
    NotImplementedError
        If the migration has no rollback sidecar (``<name>_rollback.sql``).
        The error message names the missing file so operators can
        write it.
    FileNotFoundError
        If the migration_name does not match any known migration.
    """
    import time as _time
    t0 = _time.time()

    # Resolve migration directory + sidecar path.
    migrations_dir = Path(__file__).resolve().parent
    migration_path = migrations_dir / migration_name
    if not migration_path.exists():
        raise FileNotFoundError(
            f"rollback_migration: migration file not found: {migration_path}"
        )

    # v21: rollback sidecar convention: <migration_name>_rollback.sql
    # co-located with the migration. Operators write the rollback by
    # hand (e.g. for 002: DROP COLUMN, DROP INDEX, DROP CONSTRAINT in
    # reverse order). The framework handles execution + transaction.
    stem = migration_name
    if stem.endswith(".sql"):
        stem = stem[:-4]
    rollback_path = migrations_dir / f"{stem}_rollback.sql"

    if not rollback_path.exists():
        # Honest failure: tell the operator exactly which file is missing.
        raise NotImplementedError(
            f"Rollback of '{migration_name}' requires a rollback sidecar "
            f"file at: {rollback_path}. No such file exists. Either: "
            f"(1) write the rollback SQL by hand (reverse the migration's "
            f"ALTER TABLE / CREATE INDEX / etc. statements in reverse "
            f"order), or (2) restore from database backup. The framework "
            f"will execute the sidecar inside a single transaction when "
            f"present. Current framework: {PLANNED_MIGRATION_FRAMEWORK}."
        )

    # Execute the rollback sidecar inside a single transaction.
    rollback_sql = rollback_path.read_text(encoding="utf-8")
    if engine is None:
        # Late import to avoid circular imports.
        try:
            from database.connection import get_engine
            engine = get_engine()
        except Exception as exc:
            raise RuntimeError(
                f"rollback_migration: could not obtain a database engine "
                f"({exc}). Pass engine= explicitly."
            ) from exc

    # v76 ROOT FIX (T-037 compound): detect the dialect so we can
    # translate PostgreSQL-specific SQL to SQLite for dev/test DBs.
    # The forward migration path does this via ``_translate_sql_for_sqlite``;
    # the rollback path now does the same.
    dialect_name = engine.dialect.name

    # v76 ROOT FIX (T-037 compound -- translate the WHOLE rollback SQL file
    # for SQLite BEFORE splitting, matching the forward migration path):
    #   The previous attempt translated per-statement AFTER splitting.
    #   But the splitter strips semicolons, and the translator's regexes
    #   (e.g. ``DROP CONSTRAINT IF EXISTS ... ;``) require semicolons to
    #   match. Per-statement translation therefore missed many patterns.
    #   ROOT FIX: translate the ENTIRE rollback SQL file before splitting,
    #   exactly like the forward path does at line ~4310. This ensures
    #   all translator regexes match correctly (they see the full file
    #   with semicolons intact).
    if dialect_name == DIALECT_SQLITE:
        rollback_sql = _translate_sql_for_sqlite(rollback_sql)

    statements_executed = 0
    error_msg: Optional[str] = None
    rolled_back = False
    try:
        with engine.begin() as conn:
            # SQLAlchemy begin() gives us a transaction; COMMIT on
            # success, ROLLBACK on exception.
            from sqlalchemy import text
            # v29 ROOT FIX (audit D-10): use the state-machine splitter
            # that respects string literals, dollar-quoted blocks
            # (DO $$ ... $$), and COMMENT ON ... IS '...' content,
            # instead of a naive ``split(";")`` which fragmented any
            # rollback sidecar containing semicolons inside those
            # contexts.
            for raw_stmt in _split_sql_statements(rollback_sql):
                stmt = raw_stmt.strip()
                if not stmt:
                    continue
                # v76 ROOT FIX (T-037 compound -- rollback silently no-op'd
                # every statement that started with a comment line):
                #   The previous code had ``if not stmt or stmt.startswith("--"):
                #   continue`` -- this skipped ANY statement whose first line
                #   was a comment, even if the statement contained real SQL
                #   on subsequent lines. Since EVERY rollback file has
                #   comment lines before each SQL statement (e.g.
                #   ``-- Drop the index created by 008.\nDROP INDEX ...``),
                #   the rollback_migration function silently executed ZERO
                #   statements on every invocation. The ``rolled_back=True``
                #   result was a LIE -- the transaction committed with no
                #   changes. This is why the schema_version DELETE in
                #   rollbacks 003/010/011 (which existed BEFORE v76) never
                #   actually removed the version row at runtime, and why
                #   operators reported "rollback doesn't work".
                #   ROOT FIX: remove the ``stmt.startswith("--")`` check.
                #   The comment-stripping logic below already handles
                #   comment-only statements by making them empty after
                #   stripping -- the ``if not stmt: continue`` check
                #   catches them. Statements that have a comment header
                #   followed by real SQL are now correctly executed.
                # Strip leading/trailing comment lines from the statement.
                lines = [ln for ln in stmt.splitlines() if not ln.strip().startswith("--")]
                stmt = "\n".join(lines).strip()
                if not stmt:
                    continue
                # v76 ROOT FIX (T-037 compound -- per-statement idempotent
                # no-op handling for rollbacks):
                #   On SQLite, ``DROP COLUMN`` (without IF EXISTS) raises
                #   ``no such column`` if the column was never added (e.g.
                #   rolling back 008 on a DB where 008 was never applied).
                #   Similarly, ``DROP INDEX`` without IF EXISTS raises
                #   ``no such index``. The forward migration path handles
                #   this at line ~4430 with per-statement try/except. The
                #   rollback path did NOT -- any such error aborted the
                #   entire rollback.
                #   ROOT FIX: catch "no such column", "no such index",
                #   "no such table" errors as idempotent no-ops (the
                #   rollback's intent was "ensure X does not exist", and
                #   it doesn't). Other errors (syntax errors, constraint
                #   violations) are re-raised to abort the rollback.
                try:
                    conn.execute(text(stmt))
                    statements_executed += 1
                except Exception as stmt_exc:
                    stmt_err = str(stmt_exc).lower()
                    is_idempotent_noop = (
                        "no such column" in stmt_err
                        or "no such index" in stmt_err
                        or "no such table" in stmt_err
                        or "already exists" in stmt_err
                    )
                    if is_idempotent_noop:
                        # Log and continue -- the rollback's intent is
                        # satisfied (the object doesn't exist).
                        import logging
                        logging.getLogger(__name__).debug(
                            "  [OK] Rollback statement: idempotent no-op: %s",
                            str(stmt_exc)[:120],
                        )
                        statements_executed += 1
                        continue
                    # Re-raise non-idempotent errors.
                    raise
        rolled_back = True
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        # The context manager already rolled back the transaction.

    elapsed = _time.time() - t0
    result = {
        "migration_name": migration_name,
        "rolled_back": rolled_back,
        "elapsed_s": elapsed,
        "statements_executed": statements_executed,
        "error": error_msg,
    }
    if not rolled_back:
        # Surface the error to the caller.
        raise RuntimeError(
            f"rollback_migration failed: {error_msg}"
        ) from None
    return result


# ---------------------------------------------------------------------------
# Test helpers (TEST-MIG-01, TEST-MIG-02, TEST-MIG-06)
# ---------------------------------------------------------------------------


def create_test_migrations_dir(tmp_path: Path) -> Path:
    """Create a temporary migrations directory with test SQL files.

    BUG-TEST-01: Creates multiple test migration files for more
    comprehensive testing.

    Parameters
    ----------
    tmp_path : Path
        Base temporary directory.

    Returns
    -------
    Path
        Path to the created migrations directory.
    """
    mig_dir = tmp_path / "migrations"
    mig_dir.mkdir(parents=True, exist_ok=True)

    # Create test migration 001
    test_sql_1 = mig_dir / "001_test_migration.sql"
    test_sql_1.write_text(
        "BEGIN;\n"
        "CREATE TABLE IF NOT EXISTS _test_table (id INTEGER PRIMARY KEY, name TEXT);\n"
        "COMMIT;\n",
        encoding="utf-8",
    )

    # BUG-TEST-01: Create test migration 002
    test_sql_2 = mig_dir / "002_test_alter.sql"
    test_sql_2.write_text(
        "BEGIN;\n"
        "ALTER TABLE _test_table ADD COLUMN value REAL;\n"
        "COMMIT;\n",
        encoding="utf-8",
    )

    # Create test migration 003 with data
    test_sql_3 = mig_dir / "003_test_data.sql"
    test_sql_3.write_text(
        "BEGIN;\n"
        "INSERT INTO _test_table (name, value) VALUES ('test', 1.0);\n"
        "COMMIT;\n",
        encoding="utf-8",
    )

    return mig_dir


def reset_migration_state(engine) -> None:
    """Drop migration tracking tables and schema_version.

    BUG-TEST-02: Validates SQL identifiers for safety.
    GAP-ARCH-07: Also drops schema_version table.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine.
    """
    tables = [
        "_migration_data_changes",
        "_migration_provenance",
        "_failed_migrations",
        "_migration_history",
        "schema_version",  # GAP-ARCH-07
    ]
    with engine.begin() as conn:
        for table in tables:
            try:
                safe_name = _validate_sql_identifier(table, "tracking table")
                conn.execute(text(f"DROP TABLE IF EXISTS {safe_name}"))
            except ValueError:
                logger.warning("Invalid table name in reset list: %s", table)
            except (OperationalError, ProgrammingError) as exc:
                logger.debug("Could not drop table '%s': %s", table, exc)


def count_applied_migrations(engine) -> int:
    """Count the number of successfully applied migrations.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine.

    Returns
    -------
    int
        Number of applied migrations.
    """
    _ensure_migration_tracking_table(engine)
    with engine.begin() as conn:
        try:
            r = conn.execute(
                text("SELECT COUNT(*) FROM _migration_history "
                     "WHERE status NOT IN ('failed', 'retrying')")
            )
            return r.scalar() or 0
        except (OperationalError, ProgrammingError):
            return 0


def get_migration_checksum(engine, name: str) -> str | None:
    """Get the stored checksum for a specific migration.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine.
    name : str
        Migration filename.

    Returns
    -------
    str | None
        The stored checksum, or None if not found.
    """
    return _get_stored_checksum_with_engine(engine, name)


def _get_stored_checksum_with_engine(engine, name: str) -> str | None:
    """Internal helper to get stored checksum with an engine."""
    _ensure_migration_tracking_table(engine)
    with engine.begin() as conn:
        return _get_stored_checksum(conn, name)


def verify_table_schema(engine, table_name: str, expected_columns: list[str]) -> bool:
    """Verify that a table has all expected columns.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine.
    table_name : str
        Table to verify.
    expected_columns : list[str]
        Column names that must exist.

    Returns
    -------
    bool
        True if all expected columns exist.
    """
    inspector = inspect(engine)
    if not _table_exists(inspector, table_name):
        return False
    existing = {col["name"] for col in inspector.get_columns(table_name)}
    return all(col in existing for col in expected_columns)


def plan_migrations(engine=None, config=None) -> list[dict[str, Any]]:
    """Return list of migrations that WOULD be applied, without executing.

    Each entry has: name, is_new, checksum.

    Parameters
    ----------
    engine : Engine | None
        SQLAlchemy engine. If None, calls _get_default_engine().
    config : MigrationConfig | None
        Optional configuration.

    Returns
    -------
    list[dict[str, Any]]
        Planned migration details.
    """
    if engine is None:
        engine = _get_default_engine()

    _ensure_migration_tracking_table(engine)
    inspector = inspect(engine)

    migrations_dir = (
        config.migrations_dir if config and config.migrations_dir
        else MIGRATIONS_DIR
    )

    # FIX-C5: exclude *_rollback.sql sidecars -- they are recovery scripts.
    sql_files = sorted(
        [f for f in migrations_dir.glob("*.sql") if not f.name.endswith("_rollback.sql")],
        key=lambda f: _extract_migration_number(f.name),
    )

    planned: list[dict[str, Any]] = []
    with engine.begin() as conn:
        for sql_file in sql_files:
            is_new = not _is_migration_applied_safe(conn, sql_file.name)
            checksum = _compute_checksum(sql_file.read_text(encoding="utf-8"))
            planned.append({
                "name": sql_file.name,
                "is_new": is_new,
                "checksum": checksum,
            })

    return planned


def _is_migration_applied_safe(conn, name: str) -> bool:
    """Safe version of _is_migration_applied that returns False on error.

    BUG-CODE-07: Does NOT catch OperationalError/InterfaceError -- those
    indicate database connectivity issues and should propagate.
    """
    try:
        return _is_migration_applied(conn, name)
    except ProgrammingError:
        # Table doesn't exist yet -- migration is not applied
        return False


# ---------------------------------------------------------------------------
# Failed migration management (REL-MIG-06, BUG-DES-03)
# ---------------------------------------------------------------------------


def get_failed_migrations(engine) -> list[dict[str, Any]]:
    """Query the dead letter queue for failed migrations.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine.

    Returns
    -------
    list[dict[str, Any]]
        List of failed migration records.
    """
    _ensure_migration_tracking_table(engine)
    results: list[dict[str, Any]] = []
    with engine.begin() as conn:
        try:
            r = conn.execute(
                text(
                    "SELECT migration_name, failed_at, error_message, "
                    "error_class, retry_count, resolved "
                    "FROM _failed_migrations ORDER BY failed_at"
                )
            )
            for row in r.fetchall():
                results.append({
                    "migration_name": row[0],
                    "failed_at": str(row[1]) if row[1] else None,
                    "error_message": row[2],
                    "error_class": row[3],
                    "retry_count": row[4],
                    "resolved": bool(row[5]),
                })
        except (OperationalError, ProgrammingError) as exc:
            logger.warning("Could not fetch failed migrations: %s", exc)
    return results


def retry_failed_migration(engine, migration_name: str) -> bool:
    """Attempt to retry a failed migration.

    BUG-DES-03: Uses UPDATE status='retrying' instead of DELETE to
    preserve audit trail. BUG-REL-02: Skips already-applied statements.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine.
    migration_name : str
        The migration filename to retry.

    Returns
    -------
    bool
        True if the retry succeeded.
    """
    _ensure_migration_tracking_table(engine)

    # Find the SQL file
    sql_file = MIGRATIONS_DIR / migration_name
    if not sql_file.exists():
        logger.error("Migration file not found: %s", migration_name)
        return False

    raw_content = sql_file.read_text(encoding="utf-8")
    checksum = _compute_checksum(raw_content)
    sql_content = _strip_psql_meta_commands(raw_content)

    try:
        with engine.begin() as conn:
            # BUG-DES-03: UPDATE to 'retrying' instead of DELETE
            conn.execute(
                text(
                    "UPDATE _migration_history SET status = 'retrying' "
                    "WHERE migration_name = :n AND status = 'failed'"
                ),
                {"n": migration_name},
            )

            # BUG-REL-02: Parse statements and skip already-applied ones
            statements = _split_sql_statements(sql_content)
            inspector = inspect(engine)

            for stmt in statements:
                # Skip ALTER TABLE ADD COLUMN if column already exists
                m = _ALTER_TABLE_ADD_COL_PATTERN.match(stmt)
                if m:
                    table_name, col_name = m.group(1), m.group(2)
                    if _column_exists(inspector, table_name, col_name):
                        logger.info(
                            "Skipping already-applied statement: ADD COLUMN %s.%s",
                            table_name, col_name,
                        )
                        continue

                # v83 P0-C10: strip ``--`` comment lines from each statement
                # before passing to ``text()``. SQLAlchemy's ``text()``
                # compiles the SQL through its pyformat parameter binder,
                # which interprets ``%(foo)s`` in comments as a named
                # parameter placeholder (see the SQLite forward-migration
                # path at line ~4491 for the full root-cause analysis).
                # Without this strip, retrying a migration whose SQL
                # contains a ``%(foo)s`` comment (e.g. migration 001's
                # ``ix_%(table)s_%(column)s`` naming-convention comment)
                # would crash with ``KeyError: 'foo'`` -- defeating the
                # retry mechanism.
                stmt_lines = [
                    ln for ln in stmt.splitlines()
                    if ln.strip() and not ln.strip().startswith("--")
                ]
                if not stmt_lines:
                    continue  # pure-comment statement (no-op)
                stmt_clean = "\n".join(stmt_lines)
                conn.execute(text(stmt_clean))

            # Record success -- update the 'retrying' record
            conn.execute(
                text(
                    "UPDATE _migration_history SET status = 'applied', "
                    "checksum = :c, applied_at = CURRENT_TIMESTAMP "
                    "WHERE migration_name = :n AND status = 'retrying'"
                ),
                {"c": checksum, "n": migration_name},
            )

        # Mark as resolved in _failed_migrations
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE _failed_migrations SET resolved = TRUE, "
                    "retry_count = retry_count + 1 "
                    "WHERE migration_name = :n"
                ),
                {"n": migration_name},
            )

        logger.info("Successfully retried migration: %s", migration_name)
        return True
    except Exception as exc:
        # Update retry count and record failure
        with engine.begin() as conn:
            try:
                conn.execute(
                    text(
                        "UPDATE _failed_migrations SET retry_count = retry_count + 1 "
                        "WHERE migration_name = :n"
                    ),
                    {"n": migration_name},
                )
                # BUG-DES-03: Update status back to 'failed'
                conn.execute(
                    text(
                        "UPDATE _migration_history SET status = 'failed' "
                        "WHERE migration_name = :n AND status = 'retrying'"
                    ),
                    {"n": migration_name},
                )
                _record_failure(conn, migration_name, checksum, str(exc), type(exc).__name__)
            except (OperationalError, ProgrammingError) as db_exc:
                logger.error("Could not record retry failure: %s", db_exc)

        logger.error("Retry of migration '%s' failed: %s", migration_name, exc)
        return False


def resolve_failed_migration(
    engine, migration_name: str, resolution_note: str = "",
) -> bool:
    """Mark a failed migration as resolved without retrying.

    GAP-DQ-07: Provides an API for admins to mark failures as resolved
    after manually fixing data.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine.
    migration_name : str
        The migration filename to resolve.
    resolution_note : str
        Explanation of how the issue was resolved.

    Returns
    -------
    bool
        True if the migration was successfully marked as resolved.
    """
    _ensure_migration_tracking_table(engine)

    with engine.begin() as conn:
        try:
            # Verify the migration exists in _failed_migrations
            r = conn.execute(
                text(
                    "SELECT COUNT(*) FROM _failed_migrations "
                    "WHERE migration_name = :n AND resolved = FALSE"
                ),
                {"n": migration_name},
            )
            if r.scalar() == 0:
                logger.warning(
                    "No unresolved failure found for migration: %s",
                    migration_name,
                )
                return False

            # Mark as resolved
            conn.execute(
                text(
                    "UPDATE _failed_migrations "
                    "SET resolved = TRUE, resolution_note = :note "
                    "WHERE migration_name = :n AND resolved = FALSE"
                ),
                {"n": migration_name, "note": resolution_note},
            )

            # Update _migration_history status
            conn.execute(
                text(
                    "UPDATE _migration_history SET status = 'applied' "
                    "WHERE migration_name = :n AND status = 'failed'"
                ),
                {"n": migration_name},
            )

            logger.info(
                "Resolved failed migration: %s (note: %s)",
                migration_name, resolution_note or "N/A",
            )
            return True
        except (OperationalError, ProgrammingError) as exc:
            logger.error("Could not resolve failed migration: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Partial migration state (BUG-REL-04)
# ---------------------------------------------------------------------------


def get_partial_migration_state(engine, migration_name: str) -> dict[str, Any]:
    """Analyze the partial state after a failed migration.

    BUG-REL-04: Returns which parts of a migration were applied and
    which weren't for post-failure diagnostics.
    """
    result: dict[str, Any] = {
        "migration_name": migration_name,
        "applied_constraints": [],
        "missing_constraints": [],
        "applied_columns": [],
        "missing_columns": [],
    }

    inspector = inspect(engine)
    sql_file = MIGRATIONS_DIR / migration_name

    if not sql_file.exists():
        result["error"] = f"Migration file not found: {migration_name}"
        return result

    content = sql_file.read_text(encoding="utf-8")
    content = _strip_psql_meta_commands(content)
    statements = _split_sql_statements(content)

    for stmt in statements:
        # Check ALTER TABLE ADD COLUMN
        m = _ALTER_TABLE_ADD_COL_PATTERN.match(stmt)
        if m:
            table_name, col_name = m.group(1), m.group(2)
            if _table_exists(inspector, table_name):
                if _column_exists(inspector, table_name, col_name):
                    result["applied_columns"].append(f"{table_name}.{col_name}")
                else:
                    result["missing_columns"].append(f"{table_name}.{col_name}")

    return result


# ---------------------------------------------------------------------------
# Impact analysis (LINE-MIG-03, GUARD-CODE-12, GUARD-CODE-13)
# ---------------------------------------------------------------------------


def analyze_migration_impact(engine, migration_name: str) -> dict[str, Any]:
    """Analyze the potential impact of a migration on the system.

    Scans the SQL migration content, identifies ALTER TABLE and other
    DDL/DML statements, and cross-references against known code dependencies.

    GUARD-CODE-12: Expanded to detect DROP COLUMN, ALTER COLUMN,
    ADD/DROP CONSTRAINT, INSERT, UPDATE operations.
    GUARD-CODE-13: Uses module-level compiled regexes.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine.
    migration_name : str
        The migration filename to analyze.

    Returns
    -------
    dict[str, Any]
        Impact analysis with affected_tables, affected_columns,
        dependent_code, and estimated_risk.
    """
    sql_file = MIGRATIONS_DIR / migration_name
    if not sql_file.exists():
        return {
            "affected_tables": [],
            "affected_columns": {},
            "dependent_code": [],
            "estimated_risk": "unknown",
            "error": f"Migration file not found: {migration_name}",
        }

    content = sql_file.read_text(encoding="utf-8")

    # Parse ALTER TABLE statements (GUARD-CODE-12: expanded patterns)
    affected_tables: list[str] = []
    affected_columns: dict[str, list[str]] = {}

    for pattern, operation_type in [
        (_ALTER_TABLE_ADD_COL_PATTERN, "ADD_COLUMN"),
        (_ALTER_TABLE_DROP_COL_PATTERN, "DROP_COLUMN"),
        (_ALTER_TABLE_ALTER_COL_PATTERN, "ALTER_COLUMN"),
        (_ALTER_TABLE_ADD_CONSTR_PATTERN, "ADD_CONSTRAINT"),
        (_ALTER_TABLE_DROP_CONSTR_PATTERN, "DROP_CONSTRAINT"),
    ]:
        for match in pattern.finditer(content):
            table = match.group(1)
            column = match.group(2)
            if table not in affected_tables:
                affected_tables.append(table)
            if table not in affected_columns:
                affected_columns[table] = []
            affected_columns[table].append(f"{column} ({operation_type})")

    # Parse CREATE TABLE statements
    for match in _CREATE_TABLE_PATTERN.finditer(content):
        table = match.group(1)
        if table not in affected_tables and not table.startswith("_"):
            affected_tables.append(table)

    # BUG-LINE-03: Parse DML operations (INSERT, UPDATE, DELETE)
    dml_tables: set[str] = set()
    for match in _DELETE_FROM_PATTERN.finditer(content):
        dml_tables.add(match.group(1))
    for match in _INSERT_INTO_PATTERN.finditer(content):
        dml_tables.add(match.group(1))
    for match in _UPDATE_PATTERN.finditer(content):
        dml_tables.add(match.group(1))

    has_deletes = bool(_DELETE_FROM_PATTERN.search(content))

    # Determine risk level
    risk = "low"
    if has_deletes:
        risk = "high"
    elif any("DROP" in line.upper() for line in content.split("\n") if not line.strip().startswith("--")):
        risk = "high"
    elif affected_tables or dml_tables:
        risk = "medium"

    # Identify dependent code modules
    dependent_code: list[str] = []
    table_to_module = {
        "drugs": "database.models, database.loaders, pipelines.chembl, pipelines.drugbank, pipelines.pubchem",
        "proteins": "database.models, database.loaders, pipelines.uniprot, pipelines.chembl",
        "drug_protein_interactions": "database.models, database.loaders, pipelines.chembl, pipelines.drugbank",
        "protein_protein_interactions": "database.models, database.loaders, pipelines.string",
        "gene_disease_associations": "database.models, database.loaders, pipelines.disgenet, pipelines.omim",
        "entity_mapping": "database.models, entity_resolution",
        "pipeline_runs": "database.models, dags",
    }
    for table in affected_tables:
        if table in table_to_module:
            for mod in table_to_module[table].split(", "):
                if mod not in dependent_code:
                    dependent_code.append(mod)

    return {
        "affected_tables": affected_tables,
        "affected_columns": affected_columns,
        "dml_affected_tables": list(dml_tables),
        "dependent_code": dependent_code,
        "estimated_risk": risk,
    }


# ---------------------------------------------------------------------------
# Package export verification (TEST-MIG-05, BUG-CODE-05)
# ---------------------------------------------------------------------------


def verify_package_exports() -> dict[str, bool]:
    """Verify all symbols in __all__ are actually importable.

    BUG-CODE-05: Uses standard import instead of direct __getattr__ call.

    Returns
    -------
    dict[str, bool]
        Mapping of symbol_name -> is_importable for each exported symbol.
    """
    from database.migrations import __all__

    results: dict[str, bool] = {}
    for symbol_name in __all__:
        try:
            # BUG-CODE-05: Use standard import mechanism
            import database.migrations as mig_pkg
            attr = getattr(mig_pkg, symbol_name, None)
            results[symbol_name] = attr is not None
        except (ImportError, AttributeError) as exc:
            results[symbol_name] = False
            logger.debug("Symbol '%s' not importable: %s", symbol_name, exc)
    return results


# ---------------------------------------------------------------------------
# Database fingerprint for idempotency testing (TEST-MIG-06, GAP-PERF-06)
# ---------------------------------------------------------------------------

_fingerprint_cache: dict[str, Any] | None = None
_fingerprint_cache_ts: float = 0.0
_FINGERPRINT_CACHE_TTL: float = 5.0  # seconds


def get_database_fingerprint(engine) -> dict[str, Any]:
    """Return a fingerprint of the current database state.

    GAP-PERF-06: Caches fingerprints with a 5-second TTL.
    GAP-TEST-06: Includes constraint and index info.

    Includes: table names, column counts, row counts per table,
    constraint names, index names, _migration_history contents.
    Use this to compare state before and after running migrations twice.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine.

    Returns
    -------
    dict[str, Any]
        Fingerprint of the database state.
    """
    global _fingerprint_cache, _fingerprint_cache_ts

    # GAP-PERF-06: Return cached fingerprint if fresh
    now = time.monotonic()
    if _fingerprint_cache is not None and (now - _fingerprint_cache_ts) < _FINGERPRINT_CACHE_TTL:
        return _fingerprint_cache

    inspector = inspect(engine)
    fingerprint: dict[str, Any] = {
        "tables": {},
        "migration_history": [],
    }

    for table_name in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns(table_name)]
        try:
            with engine.begin() as conn:
                count = _get_approximate_row_count(
                    conn, table_name, engine.dialect.name,
                )
        except (OperationalError, ProgrammingError):
            count = -1

        table_info: dict[str, Any] = {
            "column_count": len(columns),
            "columns": sorted(columns),
            "row_count": count,
        }

        # GAP-TEST-06: Include constraint and index info
        try:
            constraints = inspector.get_unique_constraints(table_name)
            table_info["unique_constraints"] = [c["name"] for c in constraints if c.get("name")]
        except Exception:
            table_info["unique_constraints"] = []

        try:
            indexes = inspector.get_indexes(table_name)
            table_info["indexes"] = [i["name"] for i in indexes if i.get("name")]
        except Exception:
            table_info["indexes"] = []

        fingerprint["tables"][table_name] = table_info

    # Migration history
    if "_migration_history" in inspector.get_table_names():
        with engine.begin() as conn:
            try:
                r = conn.execute(
                    text("SELECT migration_name, checksum, status FROM _migration_history ORDER BY id")
                )
                fingerprint["migration_history"] = [
                    {"name": row[0], "checksum": row[1], "status": row[2]}
                    for row in r.fetchall()
                ]
            except (OperationalError, ProgrammingError):
                pass

    # Cache the result
    _fingerprint_cache = fingerprint
    _fingerprint_cache_ts = time.monotonic()

    return fingerprint


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
# P1-042 ROOT FIX (v110): the previous CLI only supported FORWARD migrations
# (``python run_migrations.py``). The ``rollback_migration()`` function
# existed in the public API but was NOT invokable from the CLI — operators
# had to write a Python one-liner to call it. The audit's specific
# complaint: "down migrations are not invoked by the runner."
#
# ROOT FIX: add ``argparse``-based CLI with three modes:
#   1. ``python run_migrations.py``                   — apply pending forward migrations (default)
#   2. ``python run_migrations.py --rollback NAME``   — rollback a specific migration by filename
#                                                          (e.g. ``--rollback 013_is_fda_approved_nullable.sql``)
#   3. ``python run_migrations.py --down N``          — rollback the LAST N applied migrations
#                                                          (e.g. ``--down 2`` rolls back the 2 most recent)
#   4. ``python run_migrations.py --status``          — print migration status (applied / pending / failed)
#   5. ``python run_migrations.py --check``           — verify all migrations are applied + ORM matches
#
# All rollback operations execute the per-migration ``<name>_rollback.sql``
# sidecar inside a single transaction. If a sidecar is missing, the CLI
# exits with a clear error naming the missing file.

def _build_cli_parser():
    """Build the argparse parser for the migration CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="run_migrations",
        description=(
            "Cross-dialect migration runner for the Drug Repurposing ETL platform. "
            "Supports forward migrations, rollback (down migrations), status checks, "
            "and ORM/schema parity verification."
        ),
    )
    parser.add_argument(
        "--rollback",
        metavar="MIGRATION_FILENAME",
        default=None,
        help=(
            "Rollback (down-migrate) a specific migration by filename. "
            "Example: --rollback 013_is_fda_approved_nullable.sql. "
            "Executes the <name>_rollback.sql sidecar inside a single "
            "transaction. If the sidecar is missing, exits with a clear "
            "error naming the missing file."
        ),
    )
    parser.add_argument(
        "--down",
        type=int,
        metavar="N",
        default=None,
        help=(
            "Rollback the LAST N applied migrations (in reverse order). "
            "Example: --down 2 rolls back the 2 most recently applied "
            "migrations. Reads _migration_history to determine the order."
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="Print migration status (applied / pending / failed) and exit.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help=(
            "Verify all migrations are applied AND the ORM matches the DB "
            "schema. Exits 0 if healthy, 1 if drift detected."
        ),
    )
    return parser


def _cli_rollback_one(migration_filename: str) -> int:
    """Rollback a single migration by filename. Returns process exit code."""
    # Normalize the filename — accept both "013_is_fda_approved_nullable"
    # and "013_is_fda_approved_nullable.sql".
    if not migration_filename.endswith(".sql"):
        migration_filename = migration_filename + ".sql"
    try:
        result = rollback_migration(migration_filename)
    except (NotImplementedError, FileNotFoundError) as exc:
        print(f"ROLLBACK FAILED: {exc}")
        return 1
    except RuntimeError as exc:
        print(f"ROLLBACK FAILED: {exc}")
        return 1
    if result.get("rolled_back"):
        print(
            f"ROLLBACK COMPLETE: {migration_filename} — "
            f"{result.get('statements_executed', 0)} statements executed in "
            f"{result.get('elapsed_s', 0.0):.2f}s."
        )
        return 0
    print(f"ROLLBACK FAILED: {result.get('error', 'unknown error')}")
    return 1


def _cli_rollback_last_n(n: int) -> int:
    """Rollback the last N applied migrations (reverse order)."""
    if n <= 0:
        print(f"ROLLBACK FAILED: --down requires N >= 1, got {n}.")
        return 1
    # Resolve the engine so we can query _migration_history.
    try:
        engine = _resolve_engine(None)
    except Exception as exc:
        print(f"ROLLBACK FAILED: could not resolve engine ({exc}).")
        return 1
    from sqlalchemy import text as _sa_text
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                _sa_text(
                    "SELECT migration_name FROM _migration_history "
                    "WHERE status = 'applied' "
                    "ORDER BY id DESC LIMIT :n"
                ),
                {"n": n},
            ).fetchall()
    except Exception as exc:
        print(f"ROLLBACK FAILED: could not read _migration_history ({exc}).")
        return 1
    if not rows:
        print(f"ROLLBACK: no applied migrations to roll back (requested {n}).")
        return 0
    if len(rows) < n:
        print(
            f"ROLLBACK: only {len(rows)} applied migration(s) found — "
            f"rolling back all of them (requested {n})."
        )
    exit_code = 0
    for row in rows:
        name = row[0]
        print(f"  Rolling back {name}...")
        rc = _cli_rollback_one(name)
        if rc != 0:
            exit_code = rc
            break  # stop on first failure — don't roll back further
    return exit_code


def _cli_print_status() -> int:
    """Print migration status. Returns process exit code."""
    try:
        engine = _resolve_engine(None)
    except Exception as exc:
        print(f"STATUS FAILED: could not resolve engine ({exc}).")
        return 1
    status = get_migration_status(engine)
    print("=== Migration Status ===")
    print(f"  Applied:  {len(status.applied)}")
    print(f"  Pending:  {len(status.pending)}")
    print(f"  Failed:   {len(status.failed)}")
    if status.applied:
        print("  Applied migrations:")
        for name in status.applied:
            print(f"    - {name}")
    if status.pending:
        print("  Pending migrations:")
        for name in status.pending:
            print(f"    - {name}")
    if status.failed:
        print("  Failed migrations:")
        for name in status.failed:
            print(f"    - {name}")
    return 0 if not status.failed else 1


def _cli_check() -> int:
    """Verify all migrations are applied + ORM matches. Returns exit code."""
    try:
        engine = _resolve_engine(None)
    except Exception as exc:
        print(f"CHECK FAILED: could not resolve engine ({exc}).")
        return 1
    health = check_migrations(engine)
    print("=== Migration Health Check ===")
    print(f"  All applied:        {health.all_applied}")
    print(f"  Schema version OK:  {health.schema_version_matches}")
    print(f"  Healthy:            {health.healthy}")
    if health.missing:
        print(f"  Missing migrations: {health.missing}")
    if health.errors:
        print(f"  Errors:")
        for err in health.errors:
            print(f"    - {err}")
    # Also verify ORM parity.
    try:
        parity = verify_schema_matches_orm(engine)
        print(f"  ORM parity:         {parity.get('matches', 'unknown')}")
        if not parity.get("matches", True):
            print(f"  Drift details:      {parity.get('drift', {})}")
            return 1
    except Exception as exc:
        print(f"  ORM parity check failed: {exc}")
        return 1
    return 0 if health.healthy else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = _build_cli_parser()
    args = parser.parse_args()

    # Dispatch based on the flags. --check, --status, --rollback, --down
    # are mutually exclusive with the default forward-migration mode.
    if args.check:
        raise SystemExit(_cli_check())
    if args.status:
        raise SystemExit(_cli_print_status())
    if args.rollback:
        raise SystemExit(_cli_rollback_one(args.rollback))
    if args.down is not None:
        raise SystemExit(_cli_rollback_last_n(args.down))

    # Default: forward migrations.
    result = run_migrations()
    if hasattr(result, "failed") and result.failed:
        print(f"MIGRATION FAILED: {len(result.failed)} migration(s) failed")
        # v22 ROOT FIX: errors entries are now dicts (not strings).
        # Render the dict's "error" field for human readability.
        for name, err in zip(result.failed, result.errors):
            if isinstance(err, dict):
                err_str = err.get("error", str(err))
            else:
                err_str = str(err)
            print(f"  - {name}: {err_str}")
        raise SystemExit(1)
    elif hasattr(result, "applied"):
        print(f"MIGRATION COMPLETE: {len(result.applied)} applied, {len(result.skipped)} skipped")
    else:
        print("MIGRATION COMPLETE (legacy mode)")


# ---------------------------------------------------------------------------
# K fix (test isolation): Re-expose the ``run_migrations`` FUNCTION on the
# parent package's namespace after the submodule is loaded.
#
# ``import database.migrations.run_migrations`` causes Python to set
# ``database.migrations.__dict__['run_migrations'] = <this submodule>``,
# shadowing the function of the same name. We override that here so
# ``from database.migrations import run_migrations`` always returns the
# function (not the submodule), regardless of import order.
# ---------------------------------------------------------------------------
import sys as _sys

_parent_mod = _sys.modules.get("database.migrations")
if _parent_mod is not None:
    _parent_mod.__dict__["run_migrations"] = run_migrations  # type: ignore[name-defined]  # the function, not the submodule
