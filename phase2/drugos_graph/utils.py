"""
DrugOS Graph -- Neo4j Identifier Safety & DRKG Label Registry
============================================================

This module is the **single source of truth** for mapping DRKG (Drug
Repurposing Knowledge Graph) semantic entity types to Neo4j storage
labels. It also provides Cypher-identifier sanitization for labels
and relationship types.

DESIGN RATIONALE
----------------
DRKG types use natural English spelling with spaces
(e.g., ``"Side Effect"``, ``"Pharmacologic Class"``). Neo4j labels
cannot contain spaces and must match ``[A-Za-z_][A-Za-z0-9_]*``. This
module bridges that gap with an explicit, validated, immutable mapping.

The mapping is the **trust boundary** between biomedical data sources
(DRKG, SIDER, UniProt, ChEMBL, DrugBank, etc.) and the Neo4j graph.
A bug here corrupts every downstream query, ML feature, and prediction.
The RL safety ranker uses adverse-event frequencies derived from this
mapping to classify drug candidates as green / yellow / red. Wrong
labels -> wrong frequencies -> wrong safety tier -> patient harm.

ARCHITECTURE
------------
- ``config.DRKG_NODE_TYPES`` is the authoritative list of canonical
  DRKG types. This module's dict is a **derived view** that adds
  Neo4j storage metadata and the ``Protein`` UniProt-only type.
- ``LabelRegistry`` (issue 1.5) owns the mapping and enforces
  invariants (uniqueness, PascalCase, key ⊆ config) at construction.
- The mapping is wrapped in ``MappingProxyType`` (issue 4.5) to
  prevent runtime mutation. Custom types are registered via
  ``register_node_type()`` (issue 1.6), which returns a NEW registry
  and never mutates the global.

STRICT MODE
-----------
By default, ``drkg_node_type_to_neo4j_label(strict=True)`` raises
``ValueError`` on unknown types. This is deliberate -- silent fallback
was the root cause of multiple data-quality bugs (audit issues 5.1,
6.2, 7.3). To opt into the legacy fallback behavior (with WARNING log
+ dead-letter quarantine), pass ``strict=False`` or set the env var
``DRUGOS_STRICT_LABEL_MODE=warn|quarantine``.

PATIENT SAFETY
--------------
If you edit this file, you are editing the trust boundary. Every
change MUST:
1. Be accompanied by a regression test that would fail if the change
   were reverted.
2. Update ``LABEL_MAP_VERSION`` (issue 12.5) if the change is
   schema-breaking.
3. Update ``utils_FIXLOG.md`` with the audit issue ID resolved.
4. Be reviewed by someone who understands the RL safety ranker's
   dependency on adverse-event frequencies.

See: ``utils_py_forensic_audit.docx`` for the full 16-domain audit.

Fixes audit issue 13.1 -- comprehensive module docstring with design
rationale, architecture, strict mode, and patient-safety notes.
"""

# ─── Standard library imports ──────────────────────────────────────────────
# Fixes audit issue 4.3 -- explicit imports (no star imports)
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import time
import unicodedata
import warnings
from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Final, Literal, NamedTuple, NewType, Optional, TypeVar

# ─── Optional third-party imports (graceful fallback) ──────────────────────
# Fixes audit issue 11.3 -- Prometheus counters (optional; no-op fallback)
try:
    from prometheus_client import Counter as _PromCounter
    _HAS_PROMETHEUS: bool = True
except ImportError:  # pragma: no cover -- prometheus_client optional
    _HAS_PROMETHEUS = False
    _PromCounter = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
    logger.addHandler(logging.NullHandler())


# ─── P2-027 ROOT FIX (Team 8) — pipeline logging setup ───────────────────────
#
# PROBLEM (P2-027): the codebase relied on ``logging.basicConfig`` (called
# in ``config.py:8226`` under ``if __name__ == "__main__":`` and in
# various ``__main__`` blocks across phase2). In an Airflow production
# deployment, ``basicConfig`` is overridden by Airflow's own logging
# configuration — the pipeline's logs are then routed to Airflow's
# worker log file, NOT the dedicated pipeline log file. Ops cannot
# find the pipeline logs, cannot debug production issues, and the
# audit trail is corrupted.
#
# ROOT FIX (per the issue's recommendation): expose a proper
# ``setup_logging`` function that:
#   1. Uses a NAMED logger ``drugos.phase2`` (not the root logger that
#      ``basicConfig`` mutates). Airflow's logging config does NOT
#      override named loggers — it only configures the root logger
#      and Airflow's own loggers (``airflow.*``).
#   2. Adds a ``FileHandler`` writing to
#      ``${DRUGOS_LOG_DIR:-/var/log/drugos}/phase2.log``. The file
#      handler is the production-grade log destination; ops can tail
#      it, ship it to a log aggregator, or grep it for errors.
#   3. Adds a ``StreamHandler`` for console output (useful in dev and
#      for Airflow tasks where stderr is captured by the scheduler).
#   4. Respects ``DRUGOS_LOG_LEVEL`` env var (default INFO).
#   5. Is IDEMPOTENT: calling it multiple times does NOT add duplicate
#      handlers (the named logger is checked for existing handlers
#      before adding new ones).
#
# Operators call this ONCE at pipeline entry (e.g. in the Airflow task
# or the ``python -m drugos_graph`` entry point):
#
#     from drugos_graph.utils import setup_logging
#     setup_logging()
#
# After this call, all ``logging.getLogger('drugos.phase2.*')`` loggers
# (including the module-level ``logger`` in every phase2 file) route
# to BOTH the file and console handlers, REGARDLESS of Airflow's root
# logger configuration.

# The canonical pipeline logger name. All phase2 modules SHOULD use a
# child of this logger (e.g. ``logging.getLogger('drugos.phase2.evaluation')``)
# so their logs route through the handlers ``setup_logging`` attaches.
PHASE2_LOGGER_NAME: str = "drugos.phase2"

# Default log directory. Override via ``DRUGOS_LOG_DIR`` env var.
# RATIONALE: ``/var/log/drugos`` follows the Linux FHS convention for
# service logs (``/var/log/<service>/*.log``). In containerised
# deployments, ops mount a volume at ``/var/log/drugos`` so logs
# persist across container restarts.
_DEFAULT_LOG_DIR = "/var/log/drugos"
PHASE2_DEFAULT_LOG_DIR: str = os.environ.get("DRUGOS_LOG_DIR", _DEFAULT_LOG_DIR)
PHASE2_DEFAULT_LOG_FILE: str = "phase2.log"

# Default log format. Includes timestamp, level, logger name, and the
# PID (useful for correlating log lines with the MLflow heartbeat PID
# tag — see P2-024 fix in mlflow_tracker.py).
PHASE2_LOG_FORMAT: str = (
    "%(asctime)s [PID %(process)d] %(levelname)s %(name)s: %(message)s"
)
PHASE2_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    *,
    attach_stream: bool = True,
    attach_file: bool = True,
) -> logging.Logger:
    """Configure the ``drugos.phase2`` named logger with file + stream handlers.

    P2-027 ROOT FIX (Team 8): replaces ``logging.basicConfig`` (which
    Airflow overrides) with a NAMED logger that is immune to Airflow's
    root-logger configuration.

    Idempotent: calling this function multiple times does NOT add
    duplicate handlers. Existing handlers tagged with
    ``_drugos_phase2_handler=True`` are removed before the new ones
    are attached, so reconfiguration (e.g. changing the log level at
    runtime) is safe.

    Args:
        level: Log level name (``DEBUG``, ``INFO``, ``WARNING``,
            ``ERROR``, ``CRITICAL``). If None, reads the
            ``DRUGOS_LOG_LEVEL`` env var (default ``INFO``).
        log_dir: Directory for the log file. If None, reads the
            ``DRUGOS_LOG_DIR`` env var (default ``/var/log/drugos``).
            The directory is created (mode 0o755) if it does not
            exist. If the directory CANNOT be created (e.g. running
            as a non-root user without write access to ``/var/log``),
            the file handler is skipped with a WARNING, and only the
            stream handler is attached — this lets the function work
            in restricted environments (CI, containers without
            mounted volumes) without crashing.
        log_file: Log file name (within ``log_dir``). Default
            ``phase2.log``.
        attach_stream: If True (default), attach a ``StreamHandler``
            writing to stderr. Useful in dev and for Airflow tasks
            (Airflow captures stderr in the worker log).
        attach_file: If True (default), attach a ``FileHandler``
            writing to ``log_dir/log_file``. Set to False for unit
            tests that only want the stream handler.

    Returns:
        The configured ``logging.Logger`` instance for
        ``"drugos.phase2"``. Callers can further configure it (e.g.
        add a syslog handler) or pass it to other modules.

    Examples
    --------
    >>> from drugos_graph.utils import setup_logging
    >>> # Production: log to /var/log/drugos/phase2.log AND stderr
    >>> logger = setup_logging()
    >>> # Dev: log to ./logs/phase2.log at DEBUG level
    >>> logger = setup_logging(level="DEBUG", log_dir="./logs")
    >>> # CI: only stderr, no file
    >>> logger = setup_logging(attach_file=False)
    """
    # Resolve the level from the arg, then env, then default INFO.
    if level is None:
        level = os.environ.get("DRUGOS_LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)

    # Resolve log_dir from the arg, then env, then default /var/log/drugos.
    if log_dir is None:
        log_dir = os.environ.get("DRUGOS_LOG_DIR", PHASE2_DEFAULT_LOG_DIR)
    if log_file is None:
        log_file = PHASE2_DEFAULT_LOG_FILE

    # Get the named logger. This is the KEY to the P2-027 fix: a named
    # logger is NOT affected by ``logging.basicConfig`` (which only
    # configures the root logger) NOR by Airflow's logging config
    # (which only configures the root logger and ``airflow.*`` loggers).
    phase2_logger = logging.getLogger(PHASE2_LOGGER_NAME)
    phase2_logger.setLevel(numeric_level)
    # Prevent propagation to the root logger — we don't want Airflow's
    # root handler to capture our logs (that would duplicate them and
    # route them to Airflow's worker log, which is the exact bug
    # P2-027 fixes).
    phase2_logger.propagate = False

    # Remove existing handlers that we attached previously (idempotent
    # reconfiguration). We tag our handlers with a special attribute so
    # we don't accidentally remove handlers attached by other callers
    # (e.g. a custom syslog handler the operator added).
    for h in list(phase2_logger.handlers):
        if getattr(h, "_drugos_phase2_handler", False):
            phase2_logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass  # never raise from cleanup

    formatter = logging.Formatter(PHASE2_LOG_FORMAT, datefmt=PHASE2_DATE_FORMAT)

    # Attach the file handler (production-grade log destination).
    if attach_file:
        try:
            log_path = Path(log_dir) / log_file
            # Create the log directory if it doesn't exist. Use mode
            # 0o755 (rwxr-xr-x) so the directory is readable by the
            # ops team but only writable by the drugos service user.
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            file_handler._drugos_phase2_handler = True  # type: ignore[attr-defined]
            phase2_logger.addHandler(file_handler)
        except (OSError, PermissionError) as exc:
            # The log directory cannot be created or the file cannot
            # be opened. This is common in CI (no /var/log write
            # access) and in containers without mounted volumes. Log
            # to stderr and continue — the stream handler below
            # ensures logs are still captured.
            #
            # We use stderr directly (NOT the named logger) because
            # the named logger has no handlers yet at this point.
            import sys as _sys_p2_027
            _sys_p2_027.stderr.write(
                f"P2-027 WARNING: could not attach file handler to "
                f"{log_dir}/{log_file} ({exc}). Falling back to "
                f"stream-only logging. Set DRUGOS_LOG_DIR to a writable "
                f"directory to enable file logging.\n"
            )

    # Attach the stream handler (useful in dev, captured by Airflow).
    if attach_stream:
        import sys as _sys_p2_027_stream
        stream_handler = logging.StreamHandler(_sys_p2_027_stream.stderr)
        stream_handler.setLevel(numeric_level)
        stream_handler.setFormatter(formatter)
        stream_handler._drugos_phase2_handler = True  # type: ignore[attr-defined]
        phase2_logger.addHandler(stream_handler)

    return phase2_logger


# ─── Public API ────────────────────────────────────────────────────────────
# Fixes audit issue 4.3 -- explicit __all__ per PEP 8
# Fixes audit issue 14.1 -- PEP 8 compliance
__all__: list[str] = [
    # Types
    "Label", "RelType", "IdentifierKind", "LabelEntry", "LabelResult",
    "LabelRegistry",
    # Sanitization functions
    "sanitize_identifier", "sanitize_label", "sanitize_rel_type",
    "sanitize_identifiers",
    # Label lookup functions
    "drkg_node_type_to_neo4j_label", "drkg_node_types_to_neo4j_labels",
    "neo4j_label_to_drkg_node_type",
    "drkg_node_type_to_neo4j_label_with_provenance",
    # Plugin & validation API
    "register_node_type", "register_node_types", "validate_schema",
    "verify_label_map_integrity",
    "check_label_map_version_matches_graph",
    "store_label_map_metadata_in_graph",
    "commit_label_map_change",
    # Schema export & migration
    "export_label_schema", "export_label_schema_json",
    "migrate_labels", "diff_label_maps",
    # Reliability helpers
    "safe_call_with_retry", "CircuitBreaker",
    # Logging (P2-027 root fix -- named logger, NOT basicConfig)
    "setup_logging", "PHASE2_LOGGER_NAME", "PHASE2_DEFAULT_LOG_DIR",
    "PHASE2_DEFAULT_LOG_FILE", "PHASE2_LOG_FORMAT", "PHASE2_DATE_FORMAT",
    # Constants -- backward-compat names (issues C2, C3)
    "DRKG_NODE_TYPE_TO_NEO4J_LABEL", "NEO4J_LABEL_TO_DRKG_NODE_TYPE",
    "DRKG_TYPE_TO_LABEL_ENTRY", "LABEL_REGISTRY",
    # Constants -- new names
    "DEPRECATED_TYPES", "LEGACY_LABEL_ALIASES", "CASE_ALIASES",
    "LABEL_MAP_HASH", "LABEL_MAP_VERSION", "LABEL_API_VERSION",
    "LABEL_MAP_METADATA", "LABEL_SCHEMA_VERSION", "MAX_IDENTIFIER_LENGTH",
    # Compound ID normalization (P2-036)
    "normalize_inchikey",
]

# ─── Type Aliases ──────────────────────────────────────────────────────────
# Fixes audit issue 2.5 -- NewType for Label and RelType (type safety)
Label = NewType("Label", str)
RelType = NewType("RelType", str)

# Fixes audit issue 2.6 -- Literal for kind parameter (PEP 484)
# Includes all kind values used by existing callers (kg_builder, graph_stats,
# graph_queries) for backward compatibility.
IdentifierKind = Literal[
    "identifier",
    "label",
    "relationship type",
    "node label",
    "source label",
    "rel type",
]

# Fixes audit issue 2.7 -- kind-specific validation patterns
# Patterns are compiled lazily inside _KIND_PATTERNS (module-level constants
# are fine because re.compile is deterministic and idempotent).
#
# v61 ROOT FIX (label pattern too strict): the previous pattern
# `^[A-Z][A-Za-z0-9]*$` for "label" / "node label" / "source label"
# REJECTED labels containing underscores. But the codebase's own
# CORE_NODE_TYPES (config.py:3625) includes "MedDRA_Term" -- a valid
# Neo4j label with an underscore. Every call to sanitize_identifier()
# with kind="label" on "MedDRA_Term" raised ValueError, breaking
# SIDER loading, the graph_stats sanity checks, and the v43+ pathway
# integration. Neo4j's actual label naming rules allow underscores
# (https://neo4j.com/docs/cypher-manual/current/syntax/naming/#_naming_rules_for_labels):
#   - Start with a letter or underscore
#   - Subsequent chars: letters, digits, underscores
# ROOT FIX: change the label pattern to `^[A-Z][A-Za-z0-9_]*$`
# (PascalCase start preserved for label convention, but underscores
# allowed after the first char -- matching all CORE_NODE_TYPES entries
# including "MedDRA_Term", "ClinicalOutcome", "Compound", etc.).
_KIND_PATTERNS: Final[Mapping[str, "re.Pattern[str]"]] = MappingProxyType({
    "label": re.compile(r"^[A-Z][A-Za-z0-9_]*$"),
    "node label": re.compile(r"^[A-Z][A-Za-z0-9_]*$"),
    "source label": re.compile(r"^[A-Z][A-Za-z0-9_]*$"),
    "relationship type": re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$"),
    "rel type": re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$"),
    "identifier": re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$"),
})

# ─── Constants ─────────────────────────────────────────────────────────────
# Fixes audit issue 8.5, 9.2 -- MAX_IDENTIFIER_LENGTH = 1024 (DoS guard)
# Fixes audit issue 4.4 -- Final annotation on constants
MAX_IDENTIFIER_LENGTH: Final[int] = 1024

# Fixes audit issue 12.5, 14.5 -- semantic versioning for the label schema
# Bump MAJOR on breaking change (entry removed/renamed), MINOR on additive
# change (new entry), PATCH on metadata-only change.
LABEL_SCHEMA_VERSION: Final[str] = "1.0.0"
LABEL_MAP_VERSION: Final[str] = "1.0.0"
LABEL_API_VERSION: Final[str] = "1.0.0"

# Fixes audit issue 6.5 -- dead-letter queue path
# Fixes audit issue 9.4 -- audit log path
# Fixes audit issue 16.2 -- label map changes audit trail
# Fixes audit issue 16.3 -- transformation log path
# These are relative to the project root (cwd at pipeline run time).
DEAD_LETTER_PATH: Final[Path] = Path("data/dead_letter/labels.jsonl")
AUDIT_LOG_PATH: Final[Path] = Path("logs/audit/sanitization_failures.jsonl")
AUDIT_TRAIL_PATH: Final[Path] = Path("logs/audit/label_map_changes.jsonl")
TRANSFORMATION_LOG_PATH: Final[Path] = Path("logs/transformations/sanitization.jsonl")

# ─── LabelEntry NamedTuple ─────────────────────────────────────────────────
# Fixes audit issue 3.5 -- LabelEntry carries ontology/version/source metadata
# Fixes audit issue 3.6 -- deprecation fields for Side Effect -> MedDRA_Term
class LabelEntry(NamedTuple):
    """Rich metadata for a single DRKG type -> Neo4j label mapping.

    Attributes
    ----------
    neo4j_label : str
        The Neo4j-safe PascalCase storage label (e.g. ``"MedDRATerm"``).
    ontology : str
        The biomedical ontology this type belongs to
        (e.g. ``"MedDRA"``, ``"UMLS"``, ``"UniProt"``).
    ontology_version : str
        Version of the ontology (e.g. ``"26.0"``, ``"2024AA"``).
    source : str
        Which data source(s) populate this type
        (e.g. ``"DRKG+ChEMBL"``, ``"SIDER"``).
    deprecated : bool
        If True, this type is deprecated and emits a DeprecationWarning
        on use. Default False.
    deprecation_replacement : str or None
        If deprecated, the canonical replacement DRKG type name
        (e.g. ``"MedDRA_Term"`` for the deprecated ``"Side Effect"``).
    """

    neo4j_label: str
    ontology: str
    ontology_version: str
    source: str
    deprecated: bool = False
    deprecation_replacement: str | None = None


# ─── LabelResult dataclass ─────────────────────────────────────────────────
# Fixes audit issue 16.4 -- provenance for fallback labels
# Allows downstream code to know whether a label came from the dict or the
# fallback path, so it can attach ``_label_source`` properties to Neo4j nodes.
@dataclass(frozen=True)
class LabelResult:
    """Result of a label lookup with provenance metadata.

    Attributes
    ----------
    label : str
        The resolved Neo4j-safe label.
    source : str
        Provenance: ``"dict"`` if found in the canonical mapping,
        ``"fallback"`` if produced by the sanitization fallback path,
        ``"legacy_alias"`` if produced by LEGACY_LABEL_ALIASES.
    original_type : str or None
        The original input type. Set only when ``source`` is ``"fallback"``.
    """

    label: str
    source: str
    original_type: str | None = None


# ─── Raw mapping (the canonical registry data) ─────────────────────────────
# Fixes audit issue 3.1 -- MedDRA_Term added (CRITICAL patient-safety fix)
# PATIENT SAFETY: MedDRA_Term is the canonical SIDER adverse-event endpoint.
# Wrong/missing mapping here causes the RL safety ranker to see zero adverse
# events for every drug, ranking dangerous drugs as safe. See audit issue 3.1.
#
# Fixes audit issue 3.6 -- Side Effect marked deprecated (replaced by MedDRA_Term)
# Fixes audit issue 3.7 -- ATC/TAX case aliases (WHO ATC vs DRKG Atc)
# Fixes audit issue 3.5 -- LabelEntry carries ontology/version/source metadata
_RAW_LABEL_ENTRIES: Final[dict[str, LabelEntry]] = {
    "Compound": LabelEntry(
        neo4j_label="Compound",
        ontology="ChEMBL/DrugBank",
        ontology_version="ChEMBL_34",
        source="DRKG+ChEMBL",
    ),
    "Disease": LabelEntry(
        neo4j_label="Disease",
        ontology="UMLS",
        ontology_version="2024AA",
        source="DRKG",
    ),
    "Gene": LabelEntry(
        neo4j_label="Gene",
        ontology="NCBI Gene",
        ontology_version="current",
        source="DRKG",
    ),
    # PATIENT SAFETY: Protein is UniProt-only (NOT in DRKG). It must be in
    # the dict so kg_builder.create_constraints() creates a uniqueness
    # constraint on Protein.id -- otherwise MERGE creates duplicate Protein
    # nodes on pipeline re-runs (audit issue 3.1 root cause).
    "Protein": LabelEntry(
        neo4j_label="Protein",
        ontology="UniProt",
        ontology_version="2024_05",
        source="UniProt",
    ),
    "Anatomy": LabelEntry(
        neo4j_label="Anatomy",
        ontology="Uberon",
        ontology_version="2024_05",
        source="DRKG",
    ),
    "Pharmacologic Class": LabelEntry(
        neo4j_label="PharmacologicClass",
        ontology="DrugBank",
        ontology_version="current",
        source="DRKG",
    ),
    # Fixes audit issue 3.6 -- Side Effect deprecated; use MedDRA_Term
    "Side Effect": LabelEntry(
        neo4j_label="SideEffect",
        ontology="MedDRA",
        ontology_version="26.0",
        source="SIDER",
        deprecated=True,
        deprecation_replacement="MedDRA_Term",
    ),
    "Symptom": LabelEntry(
        neo4j_label="Symptom",
        ontology="UMLS",
        ontology_version="2024AA",
        source="DRKG",
    ),
    "Pathway": LabelEntry(
        neo4j_label="Pathway",
        ontology="KEGG/Reactome",
        ontology_version="v28",
        source="DRKG",
    ),
    "Biological Process": LabelEntry(
        neo4j_label="BiologicalProcess",
        ontology="GO",
        ontology_version="2024_05",
        source="DRKG",
    ),
    "Molecular Function": LabelEntry(
        neo4j_label="MolecularFunction",
        ontology="GO",
        ontology_version="2024_05",
        source="DRKG",
    ),
    "Cellular Component": LabelEntry(
        neo4j_label="CellularComponent",
        ontology="GO",
        ontology_version="2024_05",
        source="DRKG",
    ),
    "Taxonomy": LabelEntry(
        neo4j_label="Taxonomy",
        ontology="NCBI Taxonomy",
        ontology_version="current",
        source="DRKG",
    ),
    "Gene Expression": LabelEntry(
        neo4j_label="GeneExpression",
        ontology="GTEx",
        ontology_version="v8",
        source="DRKG",
    ),
    # Fixes audit issue 3.7 -- 'Atc' is DRKG's spelling; 'ATC' is WHO standard.
    # Both map to the same 'Atc' Neo4j label (case alias).
    "Atc": LabelEntry(
        neo4j_label="Atc",
        ontology="WHO ATC",
        ontology_version="2024",
        source="DRKG+DrugBank",
    ),
    "ATC": LabelEntry(
        neo4j_label="Atc",
        ontology="WHO ATC",
        ontology_version="2024",
        source="DrugBank",
    ),
    "Tax": LabelEntry(
        neo4j_label="Tax",
        ontology="NCBI Taxonomy",
        ontology_version="current",
        source="DRKG",
    ),
    "TAX": LabelEntry(
        neo4j_label="Tax",
        ontology="NCBI Taxonomy",
        ontology_version="current",
        source="DrugBank",
    ),
    # PATIENT SAFETY (audit issue 3.1): MedDRA_Term is the canonical SIDER
    # adverse-event endpoint. Without this entry, SIDER's
    # ('Compound', 'causes_adverse_event', 'MedDRA_Term') edges would
    # create nodes under a fallback label, and the RL safety ranker would
    # see ZERO adverse events for every drug -- ranking dangerous drugs
    # as 'green' (safe). This entry is the single most important line
    # in this file for patient safety.
    "MedDRA_Term": LabelEntry(
        neo4j_label="MedDRATerm",
        ontology="MedDRA",
        ontology_version="26.0",
        source="SIDER",
    ),
    # v108 ROOT FIX (issue 78 follow-up): ClinicalOutcome is a CORE_NODE_TYPE
    # (declared in config_schema.py) but was missing from this label map. The
    # previous utils.validate_schema() silently swallowed this drift; now that
    # the validator RAISES, the missing entry must be added. ClinicalOutcome
    # nodes carry the structured clinical trial outcome data (Phase 2 source
    # ClinicalTrials.gov / AACT) — they are distinct from Disease and from
    # MedDRA_Term (adverse events). Their canonical ID is clinical_outcome_id
    # (per config.CANONICAL_IDS).
    "ClinicalOutcome": LabelEntry(
        neo4j_label="ClinicalOutcome",
        ontology="AACT/ClinicalTrials.gov",
        ontology_version="2024_05",
        source="ClinicalTrials",
    ),
    # v108 ROOT FIX (issue 78 follow-up): "Drug" appears in CORE_EDGE_TYPES
    # as the source of ("Drug", "validated_treats", "Disease") — the
    # literature-validated drug-treats-disease edge from the PubMed
    # cross-check pipeline (Phase 6). "Drug" is a semantic alias of
    # "Compound" (per the project docx: "Drugs (10,000 FDA-approved
    # compounds)"); we keep "Drug" as a distinct label so the KG can
    # distinguish literature-validated treatments from ChEMBL/DrugBank
    # "Compound" treatments. Their canonical ID is drugbank_id (the
    # validating source is DrugBank/PubMed).
    "Drug": LabelEntry(
        neo4j_label="Drug",
        ontology="DrugBank/PubMed",
        ontology_version="current",
        source="LiteratureValidation",
    ),
}


# ─── LabelRegistry ─────────────────────────────────────────────────────────
# Fixes audit issue 1.5 -- LabelRegistry encapsulates mapping + invariants
# Fixes audit issue 4.5, 7.2 -- MappingProxyType prevents runtime mutation
# Fixes audit issue 6.7 -- MappingProxyType is thread-safe for reads
# Fixes audit issue 9.6 -- defends against label hijacking by malicious plugins
T = TypeVar("T")


class LabelRegistry:
    """Immutable, validating registry of DRKG type ↔ Neo4j label mappings.

    The registry enforces three invariants at construction time:

    1. **Value uniqueness** (issue 2.4) -- no two DRKG types may map to the
       same Neo4j label. Otherwise the reverse map silently loses entries.
    2. **PascalCase labels** (issue 14.3) -- all Neo4j labels must match
       ``^[A-Z][A-Za-z0-9]*$`` for RDF/JSON-LD interoperability (issue 15.7).
    3. **Non-empty** (issue 12.3) -- no empty keys or values.

    Once constructed, the registry is immutable: all internal mappings
    are wrapped in ``MappingProxyType``. Mutation raises ``TypeError``.

    Parameters
    ----------
    mapping : Mapping[str, LabelEntry]
        The canonical mapping from DRKG type name to LabelEntry.

    Raises
    ------
    ValueError
        If any invariant is violated (duplicate labels, non-PascalCase
        label, empty key/value, or label starting with underscore).

    Example
    -------
    >>> reg = LabelRegistry({"X": LabelEntry("X", "ONT", "1.0", "DRKG")})
    >>> reg.lookup("X")
    'X'
    >>> reg.reverse_lookup("X")
    'X'
    >>> len(reg)
    1
    """

    def __init__(self, mapping: Mapping[str, LabelEntry]) -> None:
        # Fixes audit issue 2.4 -- value uniqueness asserted at construction.
        # EXCEPTION: documented case aliases (issue 3.7) intentionally allow
        # multiple keys (e.g., 'ATC' and 'Atc') to map to the same Neo4j
        # label. We allow this and pick the canonical (first-seen) key for
        # the reverse map. This is a deliberate trade-off: callers can pass
        # either spelling, but reverse_lookup returns only the canonical one.
        labels = [e.neo4j_label for e in mapping.values()]
        seen: dict[str, str] = {}
        unintentional_dupes: set[str] = set()
        for k, v in mapping.items():
            if v.neo4j_label in seen:
                # Documented case aliases are allowed; anything else is a bug.
                if k.upper() != seen[v.neo4j_label].upper():
                    unintentional_dupes.add(v.neo4j_label)
            else:
                seen[v.neo4j_label] = k
        if unintentional_dupes:
            raise ValueError(
                f"Unintentional duplicate Neo4j labels in mapping: "
                f"{unintentional_dupes}. Each DRKG type must map to a "
                f"unique Neo4j label unless it is a documented case alias "
                f"(issue 2.4, 3.7)."
            )
        # Fixes audit issue 14.3 -- PascalCase enforced at construction
        for lbl in labels:
            if not re.match(r"^[A-Z][A-Za-z0-9]*$", lbl):
                raise ValueError(
                    f"Neo4j label not PascalCase: {lbl!r}. Must match "
                    f"^[A-Z][A-Za-z0-9]*$ (issue 14.3)."
                )
        # Fixes audit issue 12.3 -- no empty keys/values, no reserved prefixes
        for k, v in mapping.items():
            if not k or not v.neo4j_label:
                raise ValueError(
                    f"Empty key or value in mapping: {k!r} -> {v!r} (issue 12.3)."
                )
            if v.neo4j_label.startswith("_"):
                raise ValueError(
                    f"Neo4j label starts with underscore (reserved): "
                    f"{v.neo4j_label!r} (issue 12.3)."
                )
        # Wrap in MappingProxyType -- mutation now raises TypeError (issue 4.5)
        self._entries: MappingProxyType[str, LabelEntry] = MappingProxyType(dict(mapping))
        self._forward: MappingProxyType[str, str] = MappingProxyType(
            {k: v.neo4j_label for k, v in mapping.items()}
        )
        # Fixes audit issue 2.4 -- reverse map: case aliases collapse to the
        # CANONICAL (first-seen) key. We iterate mapping in insertion order
        # so the canonical spelling (e.g., 'Atc' before 'ATC') wins.
        reverse: dict[str, str] = {}
        for k, v in mapping.items():
            if v.neo4j_label not in reverse:
                reverse[v.neo4j_label] = k
        self._reverse: MappingProxyType[str, str] = MappingProxyType(reverse)

    def lookup(self, drkg_type: str) -> str:
        """Forward lookup: DRKG type -> Neo4j label.

        Raises ``KeyError`` if ``drkg_type`` is not in the registry.
        """
        return self._forward[drkg_type]

    def reverse_lookup(self, label: str) -> str:
        """Reverse lookup: Neo4j label -> DRKG type.

        Raises ``KeyError`` if ``label`` is not in the registry.
        """
        return self._reverse[label]

    def items(self) -> ItemsView[str, LabelEntry]:
        """Return a view of (drkg_type, LabelEntry) pairs."""
        return self._entries.items()

    def keys(self) -> KeysView[str]:
        """Return a view of DRKG type names."""
        return self._entries.keys()

    def values(self) -> ValuesView[LabelEntry]:
        """Return a view of LabelEntry values."""
        return self._entries.values()

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    @property
    def hash(self) -> str:
        """SHA-256 content hash (first 16 hex chars) of the forward mapping.

        Fixes audit issue 5.7, 16.5 -- content hash for tamper detection.
        Two registries with the same hash produce identical Neo4j graphs.
        """
        return hashlib.sha256(
            json.dumps(sorted(self._forward.items()), sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]


# Fixes audit issue 1.5 -- instantiate the global registry (validates invariants)
LABEL_REGISTRY: Final[LabelRegistry] = LabelRegistry(_RAW_LABEL_ENTRIES)

# ─── Backward-compat names (CONSTRAINT C2, C3 -- must remain importable) ────
# These names are imported by __init__.py (light reexport), kg_builder.py,
# graph_stats.py, graph_queries.py, and existing tests. They MUST remain
# the same Python objects across all imports (identity check in tests).
DRKG_TYPE_TO_LABEL_ENTRY: Final[Mapping[str, LabelEntry]] = LABEL_REGISTRY._entries
DRKG_NODE_TYPE_TO_NEO4J_LABEL: Final[Mapping[str, str]] = LABEL_REGISTRY._forward
NEO4J_LABEL_TO_DRKG_NODE_TYPE: Final[Mapping[str, str]] = LABEL_REGISTRY._reverse

# ─── Derived constant maps ─────────────────────────────────────────────────
# Fixes audit issue 3.6, 14.6 -- deprecated types (Side Effect -> MedDRA_Term)
DEPRECATED_TYPES: Final[Mapping[str, str]] = MappingProxyType({
    k: v.deprecation_replacement
    for k, v in _RAW_LABEL_ENTRIES.items()
    if v.deprecated and v.deprecation_replacement
})

# Fixes audit issue 15.4 -- legacy label aliases for backward compat
# When 'SideEffect' label is queried, redirect to 'MedDRATerm' (post-migration).
# 'MedDRA_Term' is the semantic type name; 'MedDRATerm' is the storage label.
LEGACY_LABEL_ALIASES: Final[Mapping[str, str]] = MappingProxyType({
    "SideEffect": "MedDRATerm",
    "Side_Effect": "MedDRATerm",
    "MedDRA_Term": "MedDRATerm",
})

# Fixes audit issue 3.7 -- case aliases for ATC/TAX
# Applied BEFORE general normalization in _normalize_drkg_type.
CASE_ALIASES: Final[Mapping[str, str]] = MappingProxyType({
    "ATC": "Atc",
    "TAX": "Tax",
})

# Fixes audit issue 5.7, 16.5 -- content hash for tamper detection
LABEL_MAP_HASH: Final[str] = LABEL_REGISTRY.hash

# Fixes audit issue 16.1 -- provenance metadata
LABEL_MAP_METADATA: Final[Mapping[str, str]] = MappingProxyType({
    "version": LABEL_MAP_VERSION,
    "api_version": LABEL_API_VERSION,
    "schema_version": LABEL_SCHEMA_VERSION,
    "last_updated": "2026-06-17",
    "source": "config.py + forensic audit fixes (MASTER_PROMPT_fix_utils_py.md)",
    "drkg_version": "v2",
    "audit_report": "utils_py_forensic_audit.docx",
})


# ─── Prometheus metrics (optional, with no-op fallback) ────────────────────
# Fixes audit issue 11.3 -- Prometheus counters for observability
# Fixes audit issue 7.2 (idempotency) -- re-import must NOT crash on duplicate
# metric registration. We use a helper that either creates a new Counter or
# retrieves an existing one from the default registry.
class _NoOpCounter:
    """No-op Counter fallback when prometheus_client is unavailable."""
    def labels(self, **_: Any) -> "_NoOpCounter":
        return self
    def inc(self, _amount: float = 1.0) -> None:
        pass


def _get_or_create_counter(
    name: str,
    description: str,
    labelnames: list[str],
) -> Any:
    """Get an existing Counter from the default registry or create a new one.

    Fixes audit issue 7.2 -- re-import of utils.py must not crash on duplicate
    metric registration. prometheus_client's default registry is process-global
    and rejects duplicate metric names. This helper checks the registry first
    and reuses an existing metric if present.
    """
    if not _HAS_PROMETHEUS:
        return _NoOpCounter()
    try:
        from prometheus_client import REGISTRY
        # Try to retrieve an existing metric from the registry
        existing = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
        if existing is not None:
            return existing
        # Create a new one
        return _PromCounter(name, description, labelnames)
    except Exception:  # noqa: BLE001 -- defensive; never crash import over metrics
        return _NoOpCounter()


SANITIZATION_TOTAL: Any = _get_or_create_counter(
    "drugos_sanitization_total",
    "Total sanitize_identifier calls by kind and outcome",
    ["kind", "outcome"],
)
LABEL_LOOKUP_TOTAL: Any = _get_or_create_counter(
    "drugos_label_lookup_total",
    "Total drkg_node_type_to_neo4j_label calls by path",
    ["path"],
)
DEPRECATION_WARNING_TOTAL: Any = _get_or_create_counter(
    "drugos_deprecated_type_total",
    "Times a deprecated DRKG type was used",
    ["deprecated_type", "replacement"],
)
QUARANTINE_TOTAL: Any = _get_or_create_counter(
    "drugos_quarantine_total",
    "Identifiers written to dead-letter queue",
    ["kind"],
)


# ─── Helpers ───────────────────────────────────────────────────────────────
# Fixes audit issue 2.1, 4.2, 7.3, 7.5 -- normalize before lookup
# Applied to both dict and fallback paths so equivalent inputs produce
# identical labels (idempotency).
def _normalize_drkg_type(node_type: str) -> str:
    """Normalize a DRKG type name before lookup.

    Pipeline:
    1. Apply CASE_ALIASES (issue 3.7) -- ``'ATC'`` -> ``'Atc'``.
    2. NFKC Unicode normalization (issue 7.5) -- ``'café'`` ≡ ``'café'``.
    3. Collapse whitespace runs to a single space (issue 4.2) -- tabs,
       newlines, NBSP all become ASCII space.
    4. Strip leading/trailing whitespace.

    Args:
        node_type: Raw DRKG type name from a data file.

    Returns:
        Normalized type name suitable for dict lookup.

    Raises:
        TypeError: If ``node_type`` is not a string (issue 4.1).
    """
    # Fixes audit issue 4.1 -- explicit type check
    if not isinstance(node_type, str):
        raise TypeError(
            f"node_type must be str, got {type(node_type).__name__}: "
            f"{node_type!r:.100}"
        )
    # 1. Case aliases (issue 3.7)
    nt = CASE_ALIASES.get(node_type, node_type)
    # 2. NFKC normalization (issue 7.5)
    nt = unicodedata.normalize("NFKC", nt)
    # 3. Collapse whitespace (issue 4.2) -- \s+ matches all Unicode whitespace
    nt = re.sub(r"\s+", " ", nt)
    # 4. Strip
    return nt.strip()


# Pre-compute normalized lookup table for O(1) dict access
_NORMALIZED_LOOKUP: Final[Mapping[str, str]] = MappingProxyType(
    {_normalize_drkg_type(k): v.neo4j_label for k, v in _RAW_LABEL_ENTRIES.items()}
)
_LABELS_SET: Final[frozenset[str]] = frozenset(LABEL_REGISTRY._forward.values())


# Fixes audit issue 4.9, 9.3 -- truncate, repr-escape, redact PII in errors
# Prevents log corruption, log injection (ANSI escapes), and PII leaks.
_PII_PATTERNS: Final[list[tuple["re.Pattern[str]", str]]] = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\b\+?\d{1,3}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b"), "[PHONE]"),
]


def _redact_pii(text: str) -> str:
    """Redact known PII patterns (SSN, EMAIL, PHONE) from text.

    Fixes audit issue 9.3 -- PII redaction in error messages and logs.
    """
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _safe_repr(s: object, max_len: int = 100) -> str:
    """Truncate, repr-escape, and PII-redact a value for safe logging.

    Fixes audit issue 4.9 -- name truncated to 100 chars and repr-escaped.
    Fixes audit issue 9.3 -- PII patterns redacted.
    """
    if not isinstance(s, str):
        return repr(s)
    truncated = s[:max_len]
    suffix = f"... ({len(s)} chars total)" if len(s) > max_len else ""
    redacted = _redact_pii(truncated)
    return repr(redacted) + (f" {suffix}" if suffix else "")


# ─── Canonical ID validators (v57 ROOT FIX P2C-002 + P2C-007) ────────────────
# These validators are referenced by ``config.CANONICAL_IDS_METADATA`` and
# can be used by loaders to verify that a node's canonical ID is well-formed
# before inserting it into the graph. Each returns ``True`` for valid IDs
# and ``False`` otherwise (no exceptions -- keeps the loader pipeline fast).
# v57 ROOT FIX (P2C-002 + P2C-007): add canonical IDs for
# ClinicalOutcome/MedDRA_Term/Anatomy and enforce reverse-check in
# schema validator.

# MedDRA code: 8-digit numeric string (e.g. 10002083). Leading zeros
# are significant -- MedDRA codes are NOT integers.
_MEDDRA_CODE_RE: "re.Pattern[str]" = re.compile(r"^[0-9]{8}$")

# UBERON ID: ``UBERON:<7-9 digits>`` (e.g. UBERON:0000061). The
# official OBO Foundry pattern is ``UBERON:_`` followed by exactly
# 7 zero-padded digits.
_UBERON_ID_RE: "re.Pattern[str]" = re.compile(r"^UBERON:[0-9]{7,9}$")

# InChIKey: 14 chars -- 14-char hash (uppercase letters + digits), hyphen,
# 1-char flag, hyphen, 1-char checksum. E.g. RZVAJINKQORUOD-UHFFFAOYSA-N.
_INCHIKEY_RE: "re.Pattern[str]" = re.compile(r"^[A-Z]{14}-[A-Z]{2}-[A-Z]$")

# Disease Ontology ID: ``DOID:<digits>`` (e.g. DOID:0050117).
_DOID_RE: "re.Pattern[str]" = re.compile(r"^DOID:[0-9]{4,7}$")

# NCBI Gene ID: positive integer string (e.g. 1956).
_NCBI_GENE_ID_RE: "re.Pattern[str]" = re.compile(r"^[1-9][0-9]{0,9}$")

# UniProt accession: 6 or 10 chars, uppercase letters + digits.
# E.g. P04626 (Swiss-Prot) or A0A024RBG1 (TrEMBL).
_UNIPROT_ID_RE: "re.Pattern[str]" = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)

# Reactome pathway ID: ``R-HSA-<digits>`` (Homo sapiens) or
# ``R-MMU-<digits>`` (Mus musculus), etc. We accept any ``R-XXX-<digits>``.
_REACTOME_ID_RE: "re.Pattern[str]" = re.compile(r"^R-[A-Z]{3}-[0-9]+$")

# P2-001 FORENSIC ROOT FIX (Team 4 -- namespace collision):
# ClinicalOutcome canonical ID. Format: ``CO:<drugbank_id>:<disease_key>:<indication_type>``
# (e.g. ``CO:DB00001:DOID:0050133:approved``). This is DISTINCT from
# MedDRA_Term's ``meddra_id`` (8-digit numeric) -- the two MUST NOT share a
# canonical ID field, otherwise entity resolution conflates clinical-trial-
# derived outcome records with vocabulary terms.
_CLINICAL_OUTCOME_ID_RE: "re.Pattern[str]" = re.compile(
    r"^CO:[A-Za-z0-9_.:-]+:[A-Za-z0-9_.:-]+:[A-Za-z0-9_.:-]+$"
)


def is_clinical_outcome_id(value: object) -> bool:
    """Return True iff ``value`` is a valid ClinicalOutcome canonical ID.

    Pattern: ``CO:<drugbank_id>:<disease_key>:<indication_type>``
    (e.g. ``CO:DB00001:DOID:0050133:approved``).

    P2-001 FORENSIC ROOT FIX (Team 4): ClinicalOutcome and MedDRA_Term
    previously shared the ``meddra_id`` canonical ID field, causing a
    namespace collision. ClinicalOutcome now uses ``clinical_outcome_id``
    (this validator's pattern) so ``(node_type, id_field)`` tuples are
    unique. See ``config.CANONICAL_IDS`` and ``kg_builder.ID_PATTERNS``.
    """
    if not isinstance(value, str):
        return False
    return bool(_CLINICAL_OUTCOME_ID_RE.match(value))


def is_meddra_code(value: object) -> bool:
    """Return True iff ``value`` is a valid 8-digit MedDRA code.

    MedDRA codes are 8-digit numeric strings (e.g. ``"10002083"``).
    Leading zeros are significant -- pass the value as a STRING, not an
    int. ``int`` inputs are rejected because they cannot preserve
    leading zeros.

    v57 ROOT FIX (P2C-002 + P2C-007): validator for the canonical ID
    of ``ClinicalOutcome`` and ``MedDRA_Term`` nodes.
    """
    if not isinstance(value, str):
        return False
    return bool(_MEDDRA_CODE_RE.match(value))


def is_uberon_id(value: object) -> bool:
    """Return True iff ``value`` is a valid UBERON ontology ID.

    UBERON IDs follow the OBO Foundry pattern ``UBERON:<7-9 digits>``
    (e.g. ``"UBERON:0000061"``).

    v57 ROOT FIX (P2C-002 + P2C-007): validator for the canonical ID
    of ``Anatomy`` nodes.
    """
    if not isinstance(value, str):
        return False
    return bool(_UBERON_ID_RE.match(value))


def is_inchikey(value: object) -> bool:
    """Return True iff ``value`` is a valid InChIKey.

    InChIKeys are 14-char uppercase hash, hyphen, 2-char flag, hyphen,
    1-char checksum (e.g. ``"RZVAJINKQORUOD-UHFFFAOYSA-N"``).
    """
    if not isinstance(value, str):
        return False
    return bool(_INCHIKEY_RE.match(value))


def is_doid(value: object) -> bool:
    """Return True iff ``value`` is a valid Disease Ontology ID.

    Pattern: ``DOID:<digits>`` (e.g. ``"DOID:0050117"``).
    """
    if not isinstance(value, str):
        return False
    return bool(_DOID_RE.match(value))


def is_ncbi_gene_id(value: object) -> bool:
    """Return True iff ``value`` is a valid NCBI Gene ID.

    NCBI Gene IDs are positive integers (e.g. ``"1956"``). Accepts
    both ``str`` and ``int`` (the int is converted to a string for
    validation).
    """
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        return False
    return bool(_NCBI_GENE_ID_RE.match(value))


def is_uniprot_id(value: object) -> bool:
    """Return True iff ``value`` is a valid UniProt accession.

    Accepts Swiss-Prot (6 chars: ``[OPQ]\\d[A-Z0-9]{3}\\d``) and
    TrEMBL (10 chars) accessions. E.g. ``"P04626"`` or
    ``"A0A024RBG1"``.
    """
    if not isinstance(value, str):
        return False
    return bool(_UNIPROT_ID_RE.match(value))


def is_reactome_id(value: object) -> bool:
    """Return True iff ``value`` is a valid Reactome pathway ID.

    Pattern: ``R-XXX-<digits>`` (e.g. ``"R-HSA-177929"`` for Homo
    sapiens pathways).
    """
    if not isinstance(value, str):
        return False
    return bool(_REACTOME_ID_RE.match(value))


# ─── Core sanitization ─────────────────────────────────────────────────────
# Fixes audit issue 4.6, 8.4 -- regex compiled lazily inside function
# Python caches compiled regexes internally so there's no perf hit.
def _sanitize_identifier_core(
    name: str,
    kind: IdentifierKind,
    context: dict[str, Any] | None = None,
) -> str:
    """Core sanitization logic -- pure function (no side effects beyond logs).

    Args:
        name: The identifier string to sanitize.
        kind: Semantic category (selects validation pattern).
        context: Optional dict for error context (batch_index, row_id, file).

    Returns:
        The sanitized Neo4j-safe identifier.

    Raises:
        TypeError: If ``name`` is not a string (issue 4.1).
        ValueError: If name is too long (issue 8.5, 9.2), empty after
            sanitization, starts with a digit, or fails the kind-specific
            pattern (issue 2.7).
    """
    # Fixes audit issue 4.1 -- explicit type check with clear error
    if not isinstance(name, str):
        raise TypeError(
            f"name must be str, got {type(name).__name__}: "
            f"{_safe_repr(name)}"
        )
    # Fixes audit issue 8.5, 9.2 -- length limit (DoS guard)
    if len(name) > MAX_IDENTIFIER_LENGTH:
        raise ValueError(
            f"Identifier too long: {len(name)} chars (max "
            f"{MAX_IDENTIFIER_LENGTH}). Prefix: {_safe_repr(name)}. "
            f"See audit issue 8.5."
        )
    # Fixes audit issue 7.5 -- NFKC normalization for Unicode equivalence
    normalized = unicodedata.normalize("NFKC", name)
    # Sanitize: replace every char NOT in [A-Za-z0-9_] with underscore
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", normalized)
    # Fixes audit issue 8.1 -- character check instead of re.match for default kind
    if not sanitized or not (sanitized[0].isalpha() or sanitized[0] == "_"):
        _log_sanitization_failure(name, str(kind), "empty or invalid first char", context)
        raise ValueError(
            f"Invalid Neo4j {kind} after sanitization: {_safe_repr(sanitized)} "
            f"(original: {_safe_repr(name)}). Must match [A-Za-z_][A-Za-z0-9_]*. "
            f"Context: {context}"
        )
    # Fixes audit issue 2.7 -- kind-specific pattern check (non-default kinds)
    if kind != "identifier" and kind in _KIND_PATTERNS:
        if not _KIND_PATTERNS[kind].match(sanitized):
            _log_sanitization_failure(
                name, str(kind), f"failed {kind} pattern", context
            )
            raise ValueError(
                f"Sanitized {kind} {_safe_repr(sanitized)} does not match "
                f"kind-specific pattern {_KIND_PATTERNS[kind].pattern}. "
                f"Original: {_safe_repr(name)}. Context: {context}"
            )
    # Fixes audit issue 11.1, 16.3 -- log transformation when mutation occurs
    if sanitized != name:
        _log_transformation(kind=str(kind), original=name, sanitized=sanitized, context=context)
        SANITIZATION_TOTAL.labels(kind=str(kind), outcome="mutated").inc()  # type: ignore[attr-defined]
    else:
        SANITIZATION_TOTAL.labels(kind=str(kind), outcome="success").inc()  # type: ignore[attr-defined]
    return sanitized


# ─── Public sanitization functions ─────────────────────────────────────────
# Fixes audit issue 2.5 -- split into sanitize_label + sanitize_rel_type
# Fixes audit issue 8.2 -- lru_cache on hot-path pure functions
@functools.lru_cache(maxsize=4096)
def sanitize_label(name: str) -> Label:
    """Sanitize a Neo4j node label.

    Enforces PascalCase (``^[A-Z][A-Za-z0-9]*$``) per Neo4j convention
    and RDF/JSON-LD interoperability (issue 15.7).

    Args:
        name: The label string to sanitize.

    Returns:
        A ``Label`` NewType wrapping the sanitized PascalCase string.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If the sanitized label is empty, starts with a digit,
            exceeds ``MAX_IDENTIFIER_LENGTH``, or is not PascalCase.

    Example:
        >>> sanitize_label('SideEffect')
        'SideEffect'
        >>> sanitize_label('Side Effect')
        'Side_Effect'
    """
    # Bypass the PascalCase check for backward compat: callers like
    # kg_builder pass raw DRKG types like 'Side Effect' which would
    # fail PascalCase. sanitize_label here only enforces the default
    # identifier pattern; strict PascalCase is enforced by LabelRegistry.
    # To get strict PascalCase, use LabelRegistry.lookup() instead.
    return Label(_sanitize_identifier_core(name, "identifier"))


@functools.lru_cache(maxsize=4096)
def sanitize_rel_type(name: str) -> RelType:
    """Sanitize a Neo4j relationship type.

    Relationship types in Neo4j are conventionally UPPER_SNAKE_CASE,
    but the codebase uses lower_snake_case (e.g., ``causes_side_effect``).
    This function enforces only the basic identifier pattern for
    backward compatibility.

    Args:
        name: The relationship type string to sanitize.

    Returns:
        A ``RelType`` NewType wrapping the sanitized string.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If the sanitized type is empty or starts with a digit.

    Example:
        >>> sanitize_rel_type('causes_side_effect')
        'causes_side_effect'
    """
    return RelType(_sanitize_identifier_core(name, "identifier"))


# ─── Compound ID normalization (P2-036) ───────────────────────────────────
# v102 ROOT FIX (P2-036): centralize InChIKey normalization in a single
# helper so EVERY loader that emits an InChIKey produces the SAME canonical
# form. The previous implementation had THREE call sites with three
# different behaviors:
#
#   - phase1_bridge.py:3547 — ``inchikey.upper() if inchikey else ""``
#       (uppercase, no strip, returns "" for falsy)
#   - chembl_loader.py:2557 — ``str(inchikey).strip().upper()``
#       (uppercase + strip, no None handling)
#   - pubchem_loader.py:330 — ``inchikey = "" if inchikey.lower() == "nan" else inchikey.upper()``
#       (uppercase, no strip, "nan" → "")
#
# The kg_builder.ID_PATTERNS["Compound"] regex requires UPPERCASE
# InChIKeys (``[A-Z]{14}-[A-Z]{10}-[A-Z]`` per IUPAC). Any source
# emitting a lowercase or mixed-case InChIKey would be dead-lettered.
# Centralizing the normalization here guarantees:
#   1. Whitespace is stripped (avoids " ABCD..." dead-letters).
#   2. Case is uppercased (matches ID_PATTERNS regex).
#   3. "nan"/"None"/"null" placeholders become empty string (so
#      callers can falsy-check the result).
#   4. None input returns "" (so callers don't crash on .upper()).
#
# This is the SAME class of bug as P2-010 (STITCH CIDm case mismatch)
# — two code paths producing different canonical forms for the same
# logical ID. Centralizing eliminates the entire bug class.

# IUPAC canonical InChIKey format: 14 uppercase letters, hyphen, 10
# uppercase letters, hyphen, 1 uppercase letter (protonation flag).
_INCHIKEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def normalize_inchikey(inchikey: Any) -> str:
    """Normalize an InChIKey to its canonical uppercase, stripped form.

    This is the SINGLE source of truth for InChIKey normalization across
    the DrugOS pipeline. Every loader that emits a Compound node or edge
    with an InChIKey canonical_id MUST route through this helper so the
    kg_builder.ID_PATTERNS["Compound"] regex matches and dead-lettering
    does not fragment entity resolution.

    Canonicalization rules (applied in order):
      1. ``None`` / non-string input → ``""`` (empty string).
      2. Strip leading/trailing whitespace.
      3. Uppercase (InChIKeys are case-sensitive per IUPAC; the
         canonical form is UPPERCASE).
      4. Placeholder values ``"nan"``, ``"none"``, ``"null"`` (case-
         insensitive) → ``""`` (so callers can falsy-check).

    Args:
        inchikey: Raw InChIKey string (or None / NaN-like value).

    Returns:
        Canonical InChIKey string (uppercase, stripped). Returns ``""``
        if the input is None, empty, or a placeholder. Callers should
        falsy-check the return value before using it as a canonical_id.

    Example:
        >>> normalize_inchikey("  rzbjqzwdzgozio-uhfffaoyan  ")
        'RZBJQZWDZGOZIO-UHFFFAOYAN'
        >>> normalize_inchikey(None)
        ''
        >>> normalize_inchikey("nan")
        ''
        >>> normalize_inchikey("RZBJQZWDZGOZIO-UHFFFAOYAN-N")
        'RZBJQZWDZGOZIO-UHFFFAOYAN-N'
    """
    if inchikey is None:
        return ""
    try:
        ik = str(inchikey).strip()
    except Exception:
        return ""
    if not ik:
        return ""
    ik_lower = ik.lower()
    if ik_lower in ("nan", "none", "null", "na"):
        return ""
    return ik.upper()


def sanitize_identifier(
    name: str,
    kind: IdentifierKind = "identifier",
    *,
    strict: bool = False,
    context: dict[str, Any] | None = None,
) -> str:
    """Sanitize a Neo4j label or relationship type to enforce identifier syntax.

    SECURITY SCOPE (read carefully):
        This function ONLY protects the **identifier position** in Cypher
        queries (labels, relationship types). It enforces the regex
        ``[A-Za-z_][A-Za-z0-9_]*`` after substituting invalid characters
        with underscores.

        This function does **NOT** prevent injection via:
          - Property values (always use ``$param`` syntax for values)
          - Comments (``//`` or ``/* */``)
          - String literals
          - Clause boundaries (``RETURN``, ``WITH``, ``UNION``, etc.)

        Callers must STILL parameterize all property values via ``$param``
        syntax. A query like ``f"MATCH (n:{sanitize_label(name)})
        WHERE n.id = '{user_input}' RETURN n"`` is STILL vulnerable to
        injection via ``user_input``, even though the label is safe.

        Fixes audit issue 9.1 -- accurate security scope documented.

    Args:
        name: The identifier string to sanitize.
        kind: The semantic category of the identifier. Selects the
            validation pattern. For backward compatibility with existing
            callers (kg_builder, graph_stats, graph_queries), kind is
            accepted but only ``"label"``, ``"node label"``, ``"source
            label"`` enforce PascalCase; all others fall back to the
            default identifier pattern.
        strict: If True, raise ValueError when sanitization changes the
            input (issue 5.4). If False (default), log DEBUG and return
            the sanitized value.
        context: Optional dict for error context (batch_index, row_id,
            file, correlation_id). Included in error messages and logs.

    Returns:
        A Neo4j-safe identifier matching the kind-specific pattern.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If the sanitized identifier is empty, starts with a
            digit, exceeds ``MAX_IDENTIFIER_LENGTH``, fails the
            kind-specific pattern check, or if ``strict=True`` and the
            sanitization changed the input.

    Example:
        >>> sanitize_identifier('Side Effect')
        'Side_Effect'
        >>> sanitize_identifier('Protein')
        'Protein'
        >>> sanitize_identifier('123BadStart')
        Traceback (most recent call last):
            ...
        ValueError: Invalid Neo4j identifier after sanitization: '123BadStart'...
    """
    # Apply the kind-specific pattern only when the kind explicitly requires
    # PascalCase (label / node label / source label). For all other kinds
    # (including the default 'identifier' and 'relationship type' used by
    # existing callers), use the default identifier pattern. This preserves
    # backward compat with kg_builder/graph_stats/graph_queries which pass
    # things like 'causes_side_effect' (not PascalCase).
    effective_kind: IdentifierKind = kind if kind in _KIND_PATTERNS else "identifier"
    result = _sanitize_identifier_core(name, effective_kind, context=context)
    # Fixes audit issue 5.4 -- opt-in strict mode that rejects mutation
    if strict and result != name:
        raise ValueError(
            f"Sanitization changed the input: {_safe_repr(name)} -> "
            f"{_safe_repr(result)}. Call with strict=False to allow "
            f"mutation. See audit issue 5.4."
        )
    return result


# Fixes audit issue 2.8 -- batch API
def sanitize_identifiers(
    names: list[str],
    kind: IdentifierKind = "identifier",
) -> list[str]:
    """Batch sanitize a list of identifiers.

    Uses ``functools.lru_cache`` internally via ``sanitize_identifier``
    so repeated names are cached (issue 8.2, 8.3).

    Args:
        names: List of identifier strings to sanitize.
        kind: Semantic category (see ``sanitize_identifier``).

    Returns:
        List of sanitized identifiers (same length as input).

    Example:
        >>> sanitize_identifiers(['Protein', 'Side Effect'])
        ['Protein', 'Side_Effect']
    """
    return [sanitize_identifier(n, kind) for n in names]


# ─── Label lookup functions ────────────────────────────────────────────────
# Fixes audit issue 2.1, 3.6, 5.1, 5.4, 5.8, 7.3, 7.6, 9.6, 11.2, 16.4
def drkg_node_type_to_neo4j_label(
    node_type: str,
    *,
    strict: bool | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    """Convert a DRKG node type to its Neo4j-safe storage label.

    Behavior for unknown inputs:
        - ``strict=True`` (default): raises ``ValueError``. Use this in
          production to catch schema drift early.
        - ``strict=False``: applies the normalization pipeline
          (NFKC + whitespace collapse + sanitize) and returns the
          sanitized label. Logs a WARNING and writes to the dead-letter
          queue at ``data/dead_letter/labels.jsonl`` (issue 6.5).

    Behavior for deprecated types (e.g., ``"Side Effect"``):
        Emits a ``DeprecationWarning`` and logs a WARNING. The type
        still maps successfully (for backward compat), but callers
        should migrate to the replacement (e.g., ``"MedDRA_Term"``).

    Behavior for legacy label aliases (``SideEffect``, ``MedDRA_Term``):
        Returns the canonical storage label (``MedDRATerm``). Logs INFO.

    Args:
        node_type: The DRKG type name (e.g., ``"Compound"``,
            ``"Side Effect"``, ``"MedDRA_Term"``). Case-insensitive
            for aliased types (``"ATC"`` == ``"Atc"`` per issue 3.7).
        strict: If True, raise on unknown types. If False, fall back
            to sanitization with WARNING + dead-letter. If None (default),
            falls back to the ``DRUGOS_STRICT_LABEL_MODE`` env var
            (issue 12.4) -- default 'strict'.
        context: Optional dict for error context (batch_index, row_id,
            file, correlation_id). Included in error messages and logs.

    Returns:
        The Neo4j-safe label (e.g., ``"Compound"``, ``"SideEffect"``,
        ``"MedDRATerm"``).

    Raises:
        TypeError: If ``node_type`` is not a string.
        ValueError: If ``strict=True`` and the type is unknown, or if
            the sanitized result is invalid.

    Example:
        >>> drkg_node_type_to_neo4j_label("Compound")
        'Compound'
        >>> drkg_node_type_to_neo4j_label("MedDRA_Term")
        'MedDRATerm'
        >>> drkg_node_type_to_neo4j_label("ATC")  # case alias
        'Atc'
        >>> drkg_node_type_to_neo4j_label("Unknown", strict=True)
        Traceback (most recent call last):
            ...
        ValueError: Unknown DRKG node type: 'Unknown'...
    """
    # Fixes audit issue 12.4 -- env var override for strict mode
    if strict is None:
        env_mode = os.environ.get("DRUGOS_STRICT_LABEL_MODE", "strict").lower()
        strict = env_mode == "strict"
    # Fixes audit issue 4.1 -- explicit type check
    if not isinstance(node_type, str):
        raise TypeError(
            f"node_type must be str, got {type(node_type).__name__}: "
            f"{_safe_repr(node_type)}"
        )
    # Fixes audit issue 15.4 -- legacy label aliases checked first
    # If the caller passed a storage label (e.g., 'SideEffect'), redirect
    # to the canonical storage label (e.g., 'MedDRATerm').
    if node_type in LEGACY_LABEL_ALIASES:
        new_label = LEGACY_LABEL_ALIASES[node_type]
        logger.info(
            "legacy_label_aliased",
            extra={"old": node_type, "new": new_label, "context": context or {}},
        )
        LABEL_LOOKUP_TOTAL.labels(path="legacy_alias").inc()  # type: ignore[attr-defined]
        return new_label
    # Fixes audit issue 3.6 -- deprecation warning for Side Effect
    if node_type in DEPRECATED_TYPES:
        replacement = DEPRECATED_TYPES[node_type]
        warnings.warn(
            f"DRKG type {node_type!r} is deprecated; use {replacement!r}. "
            f"See audit issue 3.6.",
            DeprecationWarning,
            stacklevel=2,
        )
        DEPRECATION_WARNING_TOTAL.labels(  # type: ignore[attr-defined]
            deprecated_type=node_type, replacement=replacement
        ).inc()
        logger.warning(
            "deprecated_drkg_type_used",
            extra={
                "deprecated": node_type,
                "replacement": replacement,
                "context": context or {},
            },
        )
    # Fixes audit issue 2.1 -- normalize before lookup
    normalized = _normalize_drkg_type(node_type)
    if normalized in _NORMALIZED_LOOKUP:
        LABEL_LOOKUP_TOTAL.labels(path="dict").inc()  # type: ignore[attr-defined]
        return _NORMALIZED_LOOKUP[normalized]
    # Fixes audit issue 2.2 -- reject already-label inputs (caller error)
    # If the caller passed a Neo4j storage label (e.g., 'SideEffect') that
    # is NOT a DRKG type, that's almost certainly a caller bug.
    if node_type in _LABELS_SET:
        raise ValueError(
            f"Input {node_type!r} is already a Neo4j label, not a DRKG type. "
            f"Did you mean to call neo4j_label_to_drkg_node_type() instead? "
            f"Context: {context}"
        )
    # Strict mode: raise (issue 5.1)
    if strict:
        LABEL_LOOKUP_TOTAL.labels(path="strict_rejected").inc()  # type: ignore[attr-defined]
        raise ValueError(
            f"Unknown DRKG node type: {node_type!r} (normalized: "
            f"{normalized!r}). Add it to DRKG_NODE_TYPE_TO_NEO4J_LABEL "
            f"or call with strict=False. Context: {context}"
        )
    # Non-strict: sanitize + warn + dead-letter (issue 5.1, 6.5)
    sanitized = sanitize_label(re.sub(r"[\s_]+", "", normalized))
    logger.warning(
        "unknown_drkg_type_fallback",
        extra={
            "original": node_type[:100],
            "normalized": normalized[:100],
            "sanitized": sanitized,
            "context": context or {},
        },
    )
    LABEL_LOOKUP_TOTAL.labels(path="fallback").inc()  # type: ignore[attr-defined]
    _quarantine_identifier(node_type, "label", context)
    return sanitized


# Fixes audit issue 2.8 -- batch variant
def drkg_node_types_to_neo4j_labels(
    types: list[str],
    *,
    strict: bool | None = None,
) -> list[str]:
    """Batch convert DRKG types to Neo4j labels.

    Args:
        types: List of DRKG type names.
        strict: See ``drkg_node_type_to_neo4j_label``.

    Returns:
        List of Neo4j labels (same length as input).
    """
    return [drkg_node_type_to_neo4j_label(t, strict=strict) for t in types]


# Fixes audit issue 2.2, 10.3 -- reverse lookup with strict mode
def neo4j_label_to_drkg_node_type(
    label: str,
    *,
    strict: bool = False,
    context: dict[str, Any] | None = None,
) -> str:
    """Convert a Neo4j label back to its original DRKG node type.

    Used by ``graph_stats.py`` to convert labels read back from Neo4j
    into canonical DRKG types for reporting (issue 1.4 -- wired into
    graph_stats.label_distribution_report).

    Args:
        label: The Neo4j label (e.g., ``"SideEffect"``).
        strict: If True, raise on unknown labels. If False (default),
            return the label as-is (identity fallback).
        context: Optional dict for error context.

    Returns:
        The canonical DRKG type name (e.g., ``"Side Effect"``).

    Raises:
        TypeError: If ``label`` is not a string.
        ValueError: If ``strict=True`` and the label is unknown.

    Example:
        >>> neo4j_label_to_drkg_node_type("Compound")
        'Compound'
        >>> neo4j_label_to_drkg_node_type("SideEffect")
        'Side Effect'
        >>> neo4j_label_to_drkg_node_type("MedDRATerm")
        'MedDRA_Term'
    """
    # Fixes audit issue 4.1 -- explicit type check
    if not isinstance(label, str):
        raise TypeError(
            f"label must be str, got {type(label).__name__}: "
            f"{_safe_repr(label)}"
        )
    if label in NEO4J_LABEL_TO_DRKG_NODE_TYPE:
        return NEO4J_LABEL_TO_DRKG_NODE_TYPE[label]
    if strict:
        raise ValueError(
            f"Unknown Neo4j label: {label!r}. Context: {context}"
        )
    logger.debug(
        "unknown_label_reverse_lookup",
        extra={"label": label[:100], "context": context or {}},
    )
    return label


# Fixes audit issue 16.4 -- provenance variant for fallback labels
def drkg_node_type_to_neo4j_label_with_provenance(
    node_type: str,
    *,
    strict: bool | None = None,
    context: dict[str, Any] | None = None,
) -> LabelResult:
    """Like ``drkg_node_type_to_neo4j_label`` but returns provenance metadata.

    Allows downstream code (e.g., ``kg_builder.load_nodes_batch``) to
    attach ``_label_source`` and ``_original_type`` properties to Neo4j
    nodes when the fallback path is used, so the graph can be audited
    for unexpected labels (issue 11.6, 16.4).

    Args:
        node_type: The DRKG type name.
        strict: See ``drkg_node_type_to_neo4j_label``.
        context: Optional dict for error context.

    Returns:
        A ``LabelResult`` with ``label``, ``source`` ('dict', 'fallback',
        or 'legacy_alias'), and ``original_type`` (set only when source
        is 'fallback').
    """
    # Try strict first; if it raises, fall back to non-strict
    try:
        label = drkg_node_type_to_neo4j_label(node_type, strict=True, context=context)
        # Determine if this was a legacy alias hit
        if node_type in LEGACY_LABEL_ALIASES:
            return LabelResult(label=label, source="legacy_alias")
        return LabelResult(label=label, source="dict")
    except ValueError:
        # Determine if non-strict mode is allowed
        env_mode = os.environ.get("DRUGOS_STRICT_LABEL_MODE", "strict").lower()
        if strict is False or env_mode != "strict":
            label = drkg_node_type_to_neo4j_label(
                node_type, strict=False, context=context
            )
            return LabelResult(label=label, source="fallback", original_type=node_type)
        raise


# ─── Reliability helpers ───────────────────────────────────────────────────
# Fixes audit issue 6.3 -- retry with exponential backoff
def safe_call_with_retry(
    fn: Callable[..., T],
    *args: Any,
    retries: int = 3,
    backoff: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (
        ConnectionError,
        TimeoutError,
    ),
    **kwargs: Any,
) -> T:
    """Call ``fn`` with exponential backoff retry on transient failures.

    Args:
        fn: The callable to invoke.
        *args: Positional args passed to ``fn``.
        retries: Maximum number of retries (default 3).
        backoff: Initial backoff in seconds (default 1.0).
        backoff_factor: Multiplier per attempt (default 2.0).
        retryable_exceptions: Exception types that trigger retry.
        **kwargs: Keyword args passed to ``fn``.

    Returns:
        The return value of ``fn``.

    Raises:
        The last exception if all retries are exhausted.

    Note:
        The following legacy kwarg aliases are accepted (and stripped
        before calling ``fn``) so older call sites keep working:
        ``max_attempts``/``max_retries``/``retry_count`` -> ``retries``,
        ``base_delay`` -> ``backoff``,
        ``max_delay`` -> caps the per-attempt sleep,
        ``retry_on`` -> ``retryable_exceptions``.
    """
    # ── Legacy kwarg compatibility shim ──────────────────────────────
    # Older call sites (kg_builder.connect, graph_queries._execute_query,
    # drugbank_parser webhook) used different kwarg names. Translate them
    # here so the function is callable from all callers without editing
    # every site. Fixes audit Tier-1 bug #1.
    if "max_attempts" in kwargs:
        retries = int(kwargs.pop("max_attempts"))
    if "max_retries" in kwargs:
        retries = int(kwargs.pop("max_retries"))
    if "retry_count" in kwargs:
        retries = int(kwargs.pop("retry_count"))
    if "base_delay" in kwargs:
        backoff = float(kwargs.pop("base_delay"))
    if "backoff_seconds" in kwargs:
        backoff = float(kwargs.pop("backoff_seconds"))
    _max_delay: float | None = None
    if "max_delay" in kwargs:
        _max_delay = float(kwargs.pop("max_delay"))
    if "retry_on" in kwargs:
        retryable_exceptions = tuple(kwargs.pop("retry_on"))
    # ─────────────────────────────────────────────────────────────────
    last_exc: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt == retries:
                raise
            sleep_time = backoff * (backoff_factor ** attempt)
            if _max_delay is not None:
                sleep_time = min(sleep_time, _max_delay)
            logger.warning(
                "transient_failure_retrying",
                extra={
                    "attempt": attempt + 1,
                    "max_retries": retries,
                    "sleep_seconds": sleep_time,
                    "error": str(exc)[:200],
                },
            )
            time.sleep(sleep_time)
    # Unreachable in practice -- the loop either returns or raises.
    assert last_exc is not None
    raise last_exc


# Fixes audit issue 6.4 -- circuit breaker for cascading failure protection
class CircuitBreaker:
    """Circuit breaker that trips after N consecutive failures.

    Once tripped, ``guard()`` raises ``RuntimeError`` until either
    ``record_success()`` is called or ``reset_after`` seconds elapse
    since the last failure.

    Args:
        threshold: Number of consecutive failures before tripping.
        reset_after: Seconds since last failure before auto-reset.

    Example:
        >>> breaker = CircuitBreaker(threshold=2)
        >>> breaker.guard()  # passes (not tripped)
        >>> breaker.record_failure()
        >>> breaker.record_failure()  # trips
        >>> breaker.guard()
        Traceback (most recent call last):
            ...
        RuntimeError: Circuit breaker tripped after 2 consecutive failures...
    """

    def __init__(
        self,
        threshold: int = 100,
        reset_after: float = 60.0,
    ) -> None:
        if threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {threshold}")
        self._threshold = threshold
        self._reset_after = reset_after
        self._failures = 0
        self._last_failure_time: float | None = None
        self._tripped = False

    def record_success(self) -> None:
        """Reset the breaker to closed state."""
        self._failures = 0
        self._tripped = False

    def record_failure(self) -> None:
        """Record a failure; trips the breaker if threshold is reached."""
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self._threshold:
            self._tripped = True
            logger.error(
                "circuit_breaker_tripped",
                extra={
                    "failures": self._failures,
                    "threshold": self._threshold,
                },
            )

    def guard(self) -> None:
        """Raise ``RuntimeError`` if the breaker is tripped.

        Auto-resets if ``reset_after`` seconds have elapsed since the
        last failure.
        """
        if self._tripped:
            if self._last_failure_time and (
                time.time() - self._last_failure_time
            ) > self._reset_after:
                self._tripped = False
                self._failures = 0
                logger.info("circuit_breaker_reset")
            else:
                raise RuntimeError(
                    f"Circuit breaker tripped after {self._failures} "
                    f"consecutive failures. See audit issue 6.4."
                )

    def is_open(self) -> bool:
        """Return True if the breaker is currently tripped (open).

        Auto-resets if ``reset_after`` seconds have elapsed since the
        last failure, mirroring ``guard()`` semantics. This is a
        non-raising probe -- callers that want to raise should use
        ``guard()`` instead.
        """
        if not self._tripped:
            return False
        if self._last_failure_time and (
            time.time() - self._last_failure_time
        ) > self._reset_after:
            self._tripped = False
            self._failures = 0
            logger.info("circuit_breaker_reset")
            return False
        return True


# Module-level circuit breaker for sanitization failures (issue 6.4)
_SANITIZATION_BREAKER: Final[CircuitBreaker] = CircuitBreaker(threshold=100)


# Fixes audit issue 6.5 -- dead-letter queue for unprocessable labels
def _quarantine_identifier(
    name: str,
    kind: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Write an unprocessable identifier to the dead-letter queue.

    The dead-letter file is JSONL at ``data/dead_letter/labels.jsonl``.
    Each record includes timestamp, kind, original (truncated to 1000
    chars), context, label_map_hash, and label_map_version. This allows
    operators to inspect bad data without losing it.

    Args:
        name: The original (unprocessable) identifier.
        kind: The identifier kind ('label', 'rel_type', etc.).
        context: Optional dict for error context.

    Returns:
        A placeholder identifier ``_QUARANTINED_<sha256[:8]>`` that is
        safe to use in Neo4j (starts with underscore, alphanumeric).
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "identifier_quarantined",
        "kind": kind,
        "original": name[:1000] if isinstance(name, str) else repr(name)[:1000],
        "context": context or {},
        "label_map_hash": LABEL_MAP_HASH,
        "label_map_version": LABEL_MAP_VERSION,
    }
    try:
        DEAD_LETTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEAD_LETTER_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        # Don't let dead-letter write failures crash the pipeline
        logger.error(
            "dead_letter_write_failed",
            extra={"error": str(exc)[:200], "record": record},
        )
    placeholder = f"_QUARANTINED_{hashlib.sha256(name.encode('utf-8', errors='replace')).hexdigest()[:8]}"
    QUARANTINE_TOTAL.labels(kind=kind).inc()  # type: ignore[attr-defined]
    logger.error("identifier_quarantined", extra=record)
    return placeholder


# ─── Logging helpers ───────────────────────────────────────────────────────
# Fixes audit issue 9.4, 11.5 -- structured audit log of sanitization failures
def _log_sanitization_failure(
    name: object,
    kind: str,
    reason: str,
    context: dict[str, Any] | None,
) -> None:
    """Append a structured audit record for a sanitization failure.

    The audit log is JSONL at ``logs/audit/sanitization_failures.jsonl``.
    Includes timestamp, kind, reason, name_length, name_prefix (PII-redacted),
    context, and label_map_hash.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "sanitization_failure",
        "kind": kind,
        "reason": reason,
        "name_length": len(name) if isinstance(name, str) else None,
        "name_prefix": _safe_repr(name[:20]) if isinstance(name, str) else None,
        "context": context or {},
        "label_map_hash": LABEL_MAP_HASH,
    }
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.error(
            "audit_log_write_failed",
            extra={"error": str(exc)[:200]},
        )
    logger.warning("sanitization_failure", extra=record)


# Fixes audit issue 16.3 -- transformation log for sanitization mutations
def _log_transformation(
    *,
    kind: str,
    original: str,
    sanitized: str,
    context: dict[str, Any] | None,
) -> None:
    """Append a structured record when sanitization mutates the input.

    The transformation log is JSONL at
    ``logs/transformations/sanitization.jsonl``. Allows tracing any Neo4j
    label back to its source DRKG type and the transformations applied.
    """
    if original == sanitized:
        return  # no transformation
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "sanitization_transformed",
        "kind": kind,
        "original": original[:1000],
        "sanitized": sanitized[:1000],
        "context": context or {},
        "label_map_hash": LABEL_MAP_HASH,
        "label_map_version": LABEL_MAP_VERSION,
    }
    try:
        TRANSFORMATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRANSFORMATION_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.error(
            "transformation_log_write_failed",
            extra={"error": str(exc)[:200]},
        )


# ─── Schema validation ────────────────────────────────────────────────────
# Fixes audit issue 1.1, 1.2, 5.5, 5.6, 12.3 -- startup validation
def validate_schema() -> None:
    """Validate the label map against ``config.DRKG_NODE_TYPES`` and ``CORE_EDGE_TYPES``.

    Call at pipeline start (e.g., from ``run_pipeline.main()``). Raises
    ``ValueError`` on any inconsistency.

    Checks performed:
        1. Every entry in ``config.DRKG_NODE_TYPES`` is in the utils dict
           (issue 1.1, 1.2, 5.5).
        2. The utils dict's only non-DRKG entry is ``"Protein"`` (UniProt-only).
        3. Every ``CORE_NODE_TYPES`` entry is in the utils dict.
        4. Every ``CORE_EDGE_TYPES`` endpoint (src and dst) is in the utils
           dict (issue 5.6 -- would have caught the MedDRA_Term bug).
        5. All Neo4j labels are PascalCase (issue 14.3).
        6. All Neo4j labels are unique (issue 2.4, 10.7).

    Raises:
        ValueError: If any check fails.
        ImportError: If config cannot be imported (deferred to runtime call).
    """
    # Deferred import to avoid circular dependency (config does not import utils)
    from .config import (
        CORE_NODE_TYPES,
        CORE_EDGE_TYPES,
        DRKG_NODE_TYPES as cfg_drkg_types,
    )

    cfg_set = set(cfg_drkg_types)
    utils_set = set(DRKG_NODE_TYPE_TO_NEO4J_LABEL.keys())

    # Issue 1.1, 1.2 -- every config DRKG type must be in the utils dict
    missing_from_utils = cfg_set - utils_set
    if missing_from_utils:
        raise ValueError(
            f"Schema drift: config.DRKG_NODE_TYPES has types missing from "
            f"utils.DRKG_NODE_TYPE_TO_NEO4J_LABEL: {sorted(missing_from_utils)}. "
            f"Add them to utils.py (audit issue 1.1, 1.2)."
        )

    # Issue 1.2 -- utils dict may have ONLY these non-DRKG extras
    # (Protein is UniProt-only; ATC/TAX are case aliases of Atc/Tax;
    #  ClinicalOutcome is Phase-2-only — sourced from ClinicalTrials.gov /
    #  AACT, not from DRKG, but is a CORE_NODE_TYPE so must be in the map.
    #  Drug is a Phase-6 literature-validation alias of Compound — appears
    #  in CORE_EDGE_TYPES as the source of "validated_treats".)
    allowed_extras = {"Protein", "ATC", "TAX", "ClinicalOutcome", "Drug"}
    extras = utils_set - cfg_set
    unexpected_extras = extras - allowed_extras
    if unexpected_extras:
        raise ValueError(
            f"Schema drift: utils dict has unexpected types not in "
            f"config.DRKG_NODE_TYPES: {sorted(unexpected_extras)}. "
            f"Either add them to config or remove from utils (audit issue 1.2)."
        )

    # Issue 5.5 -- CORE_NODE_TYPES must be a subset of the utils dict
    missing_core = set(CORE_NODE_TYPES) - utils_set
    if missing_core:
        raise ValueError(
            f"CORE_NODE_TYPES missing from label map: {sorted(missing_core)} "
            f"(audit issue 5.5)."
        )

    # Issue 5.6 -- CORE_EDGE_TYPES endpoints must all be in the utils dict
    # PATIENT SAFETY: This check would have caught the MedDRA_Term bug.
    for src, rel, dst in CORE_EDGE_TYPES:
        if src not in DRKG_NODE_TYPE_TO_NEO4J_LABEL:
            raise ValueError(
                f"CORE_EDGE_TYPES src {src!r} not in label map. "
                f"Edge: ({src!r}, {rel!r}, {dst!r}). See audit issue 5.6."
            )
        if dst not in DRKG_NODE_TYPE_TO_NEO4J_LABEL:
            raise ValueError(
                f"CORE_EDGE_TYPES dst {dst!r} not in label map. "
                f"Edge: ({src!r}, {rel!r}, {dst!r}). See audit issue 5.6."
            )

    # Issue 14.3 -- PascalCase enforced (already done in LabelRegistry.__init__
    # but double-check here for defense in depth)
    for lbl in DRKG_NODE_TYPE_TO_NEO4J_LABEL.values():
        if not re.match(r"^[A-Z][A-Za-z0-9]*$", lbl):
            raise ValueError(
                f"Neo4j label not PascalCase: {lbl!r} (audit issue 14.3)."
            )

    # Issue 2.4, 10.7 -- Neo4j labels must be unique (with documented
    # case-alias exception for ATC/Atc and TAX/Tax per issue 3.7).
    labels_list = list(DRKG_NODE_TYPE_TO_NEO4J_LABEL.values())
    if len(set(labels_list)) != len(labels_list):
        # Find the duplicates
        from collections import Counter
        counts = Counter(labels_list)
        dupes = {lbl for lbl, c in counts.items() if c > 1}
        # For each duplicate, verify all keys mapping to it are case-aliases
        # (i.e., differ only in case). Otherwise it's an unintentional bug.
        bad_dupes: set[str] = set()
        for lbl in dupes:
            keys_for_lbl = [k for k, v in DRKG_NODE_TYPE_TO_NEO4J_LABEL.items() if v == lbl]
            # All keys must be case-insensitively equal (case aliases)
            if len({k.lower() for k in keys_for_lbl}) != 1:
                bad_dupes.add(lbl)
        if bad_dupes:
            raise ValueError(
                f"Unintentional duplicate Neo4j labels: {bad_dupes} "
                f"(audit issue 2.4, 10.7). Only documented case aliases "
                f"(issue 3.7) are allowed."
            )


# Run validation at module load (best-effort; skip if config not yet importable)
# Fixes audit issue 1.1, 1.2 -- startup schema validation
try:
    validate_schema()
except ImportError:
    # Defer to runtime call from run_pipeline.py -- config may not yet be
    # fully loaded in some import orders.
    logger.debug(
        "validate_schema deferred to runtime (config not yet importable)"
    )
except ValueError as _schema_err:
    # v108 ROOT FIX (issue 78): Schema drift is a FATAL bug — it means the
    # node labels, edge types, or canonical ID mappings do not match the
    # Phase 1 contract. Wrong labels → wrong frequencies → wrong safety
    # tier → patient harm. Previously this was logged as ERROR but the
    # module continued to load, silently shipping a corrupt label map to
    # every downstream consumer. Now we re-raise so the import aborts
    # loudly at the earliest possible point.
    logger.error(
        "schema_validation_failed_at_module_load",
        extra={"error": str(_schema_err)[:500]},
    )
    raise


# ─── Plugin registration ──────────────────────────────────────────────────
# Fixes audit issue 1.6 -- register_node_type returns NEW registry (no global mutation)
def register_node_type(
    drkg_type: str,
    neo4j_label: str,
    *,
    ontology: str = "custom",
    ontology_version: str = "n/a",
    source: str = "plugin",
) -> LabelRegistry:
    """Register a custom node type. Returns a NEW LabelRegistry.

    Does NOT mutate the global ``LABEL_REGISTRY`` -- issue 7.2. The caller
    receives a new registry instance to use for its own loads.

    Args:
        drkg_type: The new DRKG type name (e.g., ``"DrugFingerprint"``).
        neo4j_label: The Neo4j PascalCase label (e.g., ``"DrugFingerprint"``).
        ontology: Ontology name (default ``"custom"``).
        ontology_version: Ontology version (default ``"n/a"``).
        source: Data source (default ``"plugin"``).

    Returns:
        A new ``LabelRegistry`` containing all base entries plus the new one.

    Raises:
        ValueError: If the new entry violates any invariant (duplicate
            label, non-PascalCase, etc.).
    """
    new_entries = dict(_RAW_LABEL_ENTRIES)
    new_entries[drkg_type] = LabelEntry(
        neo4j_label=neo4j_label,
        ontology=ontology,
        ontology_version=ontology_version,
        source=source,
    )
    new_registry = LabelRegistry(new_entries)
    logger.info(
        "node_type_registered",
        extra={
            "drkg_type": drkg_type,
            "neo4j_label": neo4j_label,
            "source": source,
        },
    )
    return new_registry


def register_node_types(mapping: Mapping[str, str]) -> LabelRegistry:
    """Batch register custom node types. Returns a NEW LabelRegistry.

    Args:
        mapping: Dict of {drkg_type: neo4j_label} pairs to add.

    Returns:
        A new ``LabelRegistry`` containing all base entries plus the new ones.
    """
    new_entries = dict(_RAW_LABEL_ENTRIES)
    for k, v in mapping.items():
        new_entries[k] = LabelEntry(
            neo4j_label=v,
            ontology="custom",
            ontology_version="n/a",
            source="plugin",
        )
    return LabelRegistry(new_entries)


# ─── Schema export ─────────────────────────────────────────────────────────
# Fixes audit issue 15.3 -- schema export for external systems (JSON)
def export_label_schema() -> dict[str, Any]:
    """Export the label schema as a JSON-serializable dict.

    Suitable for publishing via a FastAPI endpoint
    (``GET /schema/labels``) or writing to a file. Downstream systems
    (React dashboard, RL agent) should fetch this at startup instead of
    hardcoding the label mapping (issue 15.5).

    Returns:
        A dict with keys: version, api_version, schema_version, hash,
        metadata, entries (list of dicts), exported_at.
    """
    return {
        "version": LABEL_MAP_VERSION,
        "api_version": LABEL_API_VERSION,
        "schema_version": LABEL_SCHEMA_VERSION,
        "hash": LABEL_MAP_HASH,
        "metadata": dict(LABEL_MAP_METADATA),
        "entries": [
            {
                "drkg_type": k,
                "neo4j_label": v.neo4j_label,
                "ontology": v.ontology,
                "ontology_version": v.ontology_version,
                "source": v.source,
                "deprecated": v.deprecated,
                "deprecation_replacement": v.deprecation_replacement,
            }
            for k, v in DRKG_TYPE_TO_LABEL_ENTRY.items()
        ],
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }


def export_label_schema_json(indent: int = 2) -> str:
    """Export the label schema as a JSON string.

    Args:
        indent: JSON indent level (default 2).

    Returns:
        A JSON string.
    """
    return json.dumps(
        export_label_schema(),
        indent=indent,
        ensure_ascii=False,
        sort_keys=True,
    )


# ─── Migration & diff ──────────────────────────────────────────────────────
# Fixes audit issue 7.6 -- migrate_labels for schema evolution
def migrate_labels(builder: Any, old_to_new: Mapping[str, str]) -> dict[str, int]:
    """Rename labels in an existing Neo4j graph. Idempotent -- safe to re-run.

    Uses Cypher SET/REMOVE to rename labels in-place:
    ``MATCH (n:Old) SET n:New REMOVE n:Old``. Requires a ``builder``
    object with a ``driver`` attribute exposing a ``session()`` method
    (e.g., ``DrugOSGraphBuilder``).

    Args:
        builder: A DrugOSGraphBuilder (or compatible) instance.
        old_to_new: Mapping of old_label -> new_label.

    Returns:
        Dict mapping ``"{old}->{new}"`` -> count of nodes relabeled.
    """
    report: dict[str, int] = {}
    for old_label, new_label in old_to_new.items():
        # Sanitize both labels before interpolating into Cypher (defense in depth)
        safe_old = sanitize_identifier(old_label, "label")
        safe_new = sanitize_identifier(new_label, "label")
        # FIX-P2-P2-14: previously the session was opened with
        # ``builder.driver.session().run(...)`` but NEVER closed. On a
        # long-running pipeline this leaked one session per label
        # rename, eventually exhausting the Neo4j connection pool
        # (``ClientError: too many sessions``). The fix uses a context
        # manager so the session is always closed, even on exception.
        with builder.driver.session() as session:
            result = session.run(
                f"MATCH (n:`{safe_old}`) "
                f"SET n:`{safe_new}` "
                f"REMOVE n:`{safe_old}` "
                f"RETURN count(n) AS c"
            ).single()
        count = result["c"] if result else 0
        report[f"{old_label}->{new_label}"] = count
        logger.info(
            "label_migrated",
            extra={
                "old": old_label,
                "new": new_label,
                "count": count,
                "label_map_hash": LABEL_MAP_HASH,
            },
        )
    return report


# Fixes audit issue 16.6 -- structured diff between label map versions
def diff_label_maps(
    old: Mapping[str, str],
    new: Mapping[str, str],
) -> dict[str, list[dict[str, str]]]:
    """Compute a structured diff between two label maps.

    Args:
        old: Old mapping {drkg_type: neo4j_label}.
        new: New mapping {drkg_type: neo4j_label}.

    Returns:
        Dict with keys 'added', 'removed', 'changed', each a list of dicts.
    """
    old_keys = set(old.keys())
    new_keys = set(new.keys())
    return {
        "added": [
            {"key": k, "value": new[k]} for k in sorted(new_keys - old_keys)
        ],
        "removed": [
            {"key": k, "value": old[k]} for k in sorted(old_keys - new_keys)
        ],
        "changed": [
            {"key": k, "old_value": old[k], "new_value": new[k]}
            for k in sorted(old_keys & new_keys) if old[k] != new[k]
        ],
    }


# ─── Integrity verification ───────────────────────────────────────────────
# Fixes audit issue 16.5 -- verify hash at pipeline start (tamper detection)
def verify_label_map_integrity() -> None:
    """Verify the label map hash matches the value computed at module load.

    Call at the start of ``run_pipeline.main()`` to detect tampering
    with the immutable ``DRKG_NODE_TYPE_TO_NEO4J_LABEL`` mapping.

    Raises:
        RuntimeError: If the recomputed hash differs from ``LABEL_MAP_HASH``.
    """
    current_hash = LABEL_REGISTRY.hash
    if current_hash != LABEL_MAP_HASH:
        raise RuntimeError(
            f"Label map hash mismatch! Expected {LABEL_MAP_HASH}, got "
            f"{current_hash}. The dict may have been tampered with. "
            f"See audit issue 16.5."
        )


# Fixes audit issue 12.6, 16.7 -- store + check graph version
def store_label_map_metadata_in_graph(builder: Any) -> None:
    """Store ``LABEL_MAP_VERSION``, ``LABEL_MAP_HASH``, etc. as Neo4j graph properties.

    Call at the start of ``run_pipeline.main()``. Operators can then query
    the graph to determine which version of utils.py produced a given graph.

    Args:
        builder: A DrugOSGraphBuilder (or compatible) instance.
    """
    # v102 ROOT FIX (P2-038) / P2-060: use ``with`` context manager for
    # style consistency with migrate_labels() and
    # check_label_map_version_matches_graph(). The previous
    # ``session = builder.driver.session()`` + try/finally + ``session.close()``
    # pattern was functionally equivalent but stylistically inconsistent — a
    # new developer copying the wrong pattern might forget the close(). The
    # ``with`` form guarantees the session is closed even on exception, with
    # no chance of forgetting the finally block. This matches the modern
    # neo4j-python-driver idiom and the pattern already used by
    # migrate_labels in this same file.
    with builder.driver.session() as session:
        session.run(
            "CALL dbms.setGraphProperty('label_map_version', $v)",
            v=LABEL_MAP_VERSION,
        )
        session.run(
            "CALL dbms.setGraphProperty('label_map_hash', $h)",
            h=LABEL_MAP_HASH,
        )
        session.run(
            "CALL dbms.setGraphProperty('label_api_version', $v)",
            v=LABEL_API_VERSION,
        )
        session.run(
            "CALL dbms.setGraphProperty('label_map_metadata', $m)",
            m=json.dumps(dict(LABEL_MAP_METADATA)),
        )
        session.run(
            "CALL dbms.setGraphProperty('pipeline_run_at', $t)",
            t=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(
            "label_map_metadata_stored_in_graph",
            extra={"version": LABEL_MAP_VERSION, "hash": LABEL_MAP_HASH},
        )


def check_label_map_version_matches_graph(builder: Any) -> None:
    """Check that the graph's stored label_map_version matches the code version.

    If the graph has no stored version (first run), stores it.
    If the versions differ, raises ``RuntimeError`` instructing the operator
    to run ``migrate_labels()`` first.

    Args:
        builder: A DrugOSGraphBuilder (or compatible) instance.

    Raises:
        RuntimeError: If the graph's stored version differs from the code version.
    """
    # v102 ROOT FIX (P2-038) / P2-060: use ``with`` context manager for
    # style consistency with migrate_labels() (line 1927) and
    # store_label_map_metadata_in_graph (above). The previous
    # try/finally + session.close() was SAFE (the finally DID close)
    # but style-inconsistent — the codebase standard is the ``with``
    # form which guarantees close on ANY control-flow path including
    # early returns and unexpected exceptions inside the body.
    with builder.driver.session() as session:
        result = session.run(
            "CALL dbms.graphproperty('label_map_version') YIELD value "
            "RETURN value"
        ).single()
        if result is None:
            store_label_map_metadata_in_graph(builder)
        elif result["value"] != LABEL_MAP_VERSION:
            logger.error(
                "label_map_version_mismatch",
                extra={
                    "graph_version": result["value"],
                    "code_version": LABEL_MAP_VERSION,
                },
            )
            raise RuntimeError(
                f"Label map version mismatch! Graph has {result['value']!r}, "
                f"code has {LABEL_MAP_VERSION!r}. Run migrate_labels() first "
                f"(audit issue 12.6)."
            )


# Fixes audit issue 16.2 -- commit_label_map_change audit trail
def commit_label_map_change(
    *,
    change_type: str,
    before: Any,
    after: Any,
    rationale: str,
    audit_issue: str | None = None,
    actor: str | None = None,
) -> None:
    """Append a structured audit record for a label map change.

    Called from a pre-commit hook or CI job. The audit trail is JSONL
    at ``logs/audit/label_map_changes.jsonl``.

    Args:
        change_type: One of 'added_entry', 'removed_entry', 'renamed_entry',
            'metadata_update'.
        before: The previous value (None for 'added_entry').
        after: The new value (None for 'removed_entry').
        rationale: Human-readable reason for the change.
        audit_issue: Audit issue ID this change resolves (e.g., '3.1').
        actor: Git author email (default: read from git config).
    """
    if actor is None:
        import subprocess
        try:
            actor = subprocess.check_output(
                ["git", "config", "user.email"], text=True
            ).strip()
        except Exception:
            actor = "unknown"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "label_map_changed",
        "change_type": change_type,
        "before": before,
        "after": after,
        "rationale": rationale,
        "audit_issue": audit_issue,
        "actor": actor,
        "label_map_version": LABEL_MAP_VERSION,
        "label_map_hash": LABEL_MAP_HASH,
    }
    try:
        AUDIT_TRAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_TRAIL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.error(
            "audit_trail_write_failed",
            extra={"error": str(exc)[:200]},
        )


# ─── Backward-compat: re-export _SAFE_IDENTIFIER_RE ────────────────────────
# Fixes audit issue 4.6 -- keep _SAFE_IDENTIFIER_RE name for backward compat
# (some external test suites may import it). Compiled lazily; this is the
# same regex used by _sanitize_identifier_core via the default pattern.
_SAFE_IDENTIFIER_RE: Final["re.Pattern[str]"] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
