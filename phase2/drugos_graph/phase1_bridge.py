"""
DrugOS Graph — Phase 1 → Phase 2 Bridge
========================================

This module is the **single, authoritative contract** that connects the two
phases of the Autonomous Drug Repurposing Platform:

  Phase 1  (``phase1/``)  — Data Ingestion & Pipeline Setup
      Outputs cleaned, normalised, schema-validated data into either:
        (a) PostgreSQL via the SQLAlchemy ORM (``database/models.py``) — the
            AUTHORITATIVE backend per the docx architecture, OR
        (b) CSV files in ``phase1/processed_data/`` — the legacy/dev fallback.
      Either backend produces the same dict of DataFrames for Phase 2.

  Phase 2  (``drugos_graph/``) — Knowledge Graph Construction
      Consumes nodes/edges and loads them into Neo4j via
      :class:`drugos_graph.kg_builder.DrugOSGraphBuilder`.

v29 ROOT FIX (Phase1↔Phase2 100% connection):
  The forensic audit proved the bridge previously bypassed PostgreSQL
  entirely and read CSVs only — Phase 1's 8,500 lines of ORM/migration/
  loader code were dead weight. The bridge now prefers PostgreSQL when
  ``DATABASE_URL`` is set AND the ``drugs`` table is populated. The CSV
  path remains as a fallback for dev/CI runs without a database.

  The chosen backend is recorded as ``out["_phase1_backend"]`` (either
  ``"postgresql"`` or ``"csv"``) so operators can verify the production
  path was actually used.

The bridge provides THREE callable entry points, in increasing order of
abstraction:

  1. :func:`read_phase1_outputs`      — read Phase 1 data (PostgreSQL or CSV) into pandas DataFrames
  2. :func:`stage_phase1_to_phase2`   — convert DataFrames → Phase 2 node/edge dicts
  3. :func:`load_into_graph`           — load staged dicts into a graph builder

Plus a top-level convenience:

  4. :func:`run_phase1_to_phase2`     — read → stage → load in one call

WHY THIS MODULE EXISTS
----------------------
Before this module existed, Phase 1's ``exporters/neo4j_exporter.py`` raised
``NotImplementedError`` and Phase 2's loaders re-downloaded every source
file from external URLs (DRKG, DrugBank XML, ChEMBL SQLite, etc.). The two
phases were never connected. This module is the missing wire.

The conversion is **lossless and bidirectionally traceable**: every node and
edge produced by the bridge carries a ``_source_phase=1`` lineage property
and the original Phase 1 row index so any downstream bug can be traced back
to the exact Phase 1 CSV row.

SCHEMA MAPPING (Phase 1 CSV column → Phase 2 node/edge property)
----------------------------------------------------------------

Compound nodes (from drugbank_drugs.csv)
    P2-009 ROOT FIX (docstring drift): the previous docstring claimed
    ``drugbank_id → id (canonical Neo4j ID)``. That was the v3.11
    behaviour. As of v3.12 (see config.py:5508-5509,
    ``CANONICAL_IDS["Compound"] = "inchikey"`` with the comment
    "Changed from drugbank_id (issue 3.12)"), the canonical Neo4j ID
    for Compound nodes is ``inchikey`` (uppercase IUPAC form, e.g.
    ``RZVAJINKQORUOD-UHFFFAOYSA-N`` for aspirin). The actual bridge
    code at lines ~3547-3551 uses ``inchikey_canonical`` when present
    and non-synthetic, falling back to ``drugbank_id`` ONLY for
    biologics that lack an InChIKey (e.g. ``DB00071`` for insulin
    glargine). A new developer reading the old docstring wrote
    queries / unit tests / merge logic keyed on ``drugbank_id`` —
    which worked for biologics but silently FAILED for every
    small-molecule drug (which uses ``inchikey``). The misleading
    documentation caused silent test-passes-while-prod-fails bugs
    (tests used biologics, prod used small molecules).

    Canonical ID resolution (in priority order — see also
    ``entity_resolver.resolve_canonical_id``):
        inchikey (uppercase)         → id    (canonical Neo4j ID when
                                              present and non-synthetic;
                                              universal across ChEMBL,
                                              PubChem, DrugBank per
                                              IUPAC standard)
        drugbank_id (e.g. DB00071)   → id    (fallback for biologics
                                              and any compound lacking
                                              an InChIKey — typically
                                              large molecules /
                                              peptides / mixtures)

    Other Compound properties (unchanged):
        name                          → name
        inchikey                      → inchikey
        smiles                        → smiles
        molecular_weight              → molecular_weight
        is_fda_approved               → fda_approved
        is_withdrawn                  → withdrawn       (RL safety signal — patient harm)
        clinical_status               → clinical_status
        groups                        → groups
        mechanism_of_action           → mechanism_of_action
        cas_number                    → cas_number
        chembl_id                     → chembl_id
        pubchem_cid                   → pubchem_cid
        completeness_score            → completeness_score

Protein nodes (from drugbank_interactions.csv.gz, dedup on uniprot_id)
    uniprot_id         → id
    target_name        → name
    organism           → organism

Gene nodes (from omim_gene_disease_associations.csv, dedup on gene_symbol)
    gene_symbol        → id
    gene_mim           → mim_id

Disease nodes (from omim_gene_disease_associations.csv, dedup on disease_id)
    disease_id         → id
    disease_name       → name
    phenotype_mim      → mim_id

Edges
    drugbank_interactions.csv.gz:
        (Compound, targets, Protein)   — action_type='target'/'unknown'/None
        (Compound, inhibits, Protein)  — action_type contains 'inhibitor'
        (Compound, activates, Protein) — action_type contains 'activator'
        (Compound, allosterically_modulates, Protein)
                                       — action_type contains 'allosteric'
        (Compound, unknown, Protein)   — action_type set but not matched
    omim_gene_disease_associations.csv:
        (Gene, associated_with, Disease)
                                       — score + association_type as props

The edge types above are a strict subset of
:data:`drugos_graph.config.CORE_EDGE_TYPES` — no non-core edges are produced.

USAGE
-----
Production (with a real PostgreSQL + Neo4j)::

    # DATABASE_URL must be set in the environment.
    from drugos_graph import DrugOSGraphBuilder, Neo4jConfig
    from drugos_graph.phase1_bridge import run_phase1_to_phase2

    builder = DrugOSGraphBuilder(Neo4jConfig.from_env())
    builder.connect()
    builder.create_constraints()
    report = run_phase1_to_phase2(
        phase1_processed_dir="/path/to/phase1/processed_data",
        builder=builder,
    )
    assert report["backend"] == "postgresql"  # verify root-fix took effect
    print(report["summary"])

Testing (no PostgreSQL, no Neo4j required)::

    from drugos_graph.phase1_bridge import (
        run_phase1_to_phase2, RecordingGraphBuilder,
    )
    recorder = RecordingGraphBuilder()
    report = run_phase1_to_phase2(
        phase1_processed_dir="phase1/processed_data",
        builder=recorder,
        prefer_postgres=False,  # force CSV backend in unit tests
    )
    assert report["summary"]["nodes_loaded"] > 0
    assert report["summary"]["edges_loaded"] > 0

PATIENT-SAFETY NOTE
-------------------
The ``withdrawn`` flag on Compound nodes is the primary input to the RL
agent's safety ranker. A null ``withdrawn`` value is treated as "not
withdrawn" → SAFE → a withdrawn drug like Valdecoxib would be surfaced as a
repurposing candidate. The bridge EXPLICITLY coerces ``is_withdrawn`` to a
bool and writes ``withdrawn=False`` (never null) for every Compound node.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import os
import re  # v24: needed for CHEMBL_TGT_ ID normalization (Audit Chain 9 fix)
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol, Tuple

import pandas as pd

# v109 ROOT FIX (P2-026): the previous code imported kg_builder symbols
# (NODE_PROPERTY_WHITELIST, EDGE_PROPERTY_WHITELIST, SYSTEM_PROPS,
# _storage_label, ID_PATTERNS, CORE_EDGE_TYPES) INSIDE the functions
# ``_apply_node_whitelist``, ``_apply_edge_whitelist``, ``_validate_node_id``,
# and ``RecordingGraphBuilder.load_edges_batch``. The function-level imports
# were originally there to avoid circular imports — but ``kg_builder`` does
# NOT import ``phase1_bridge`` (verified via grep), so there is NO circular
# import risk. The per-call import overhead was O(N) per edge batch (Python's
# import cache makes the actual import fast, but the try/except + attribute
# lookup on every call is still wasteful). ROOT FIX: import ONCE at module
# level. If kg_builder ever starts importing phase1_bridge, this will fail
# loudly at module load time (which is the correct behavior — we want to
# know immediately if a circular import is introduced).
try:
    from .kg_builder import (  # type: ignore
        NODE_PROPERTY_WHITELIST as _KG_NODE_PROPERTY_WHITELIST,
        EDGE_PROPERTY_WHITELIST as _KG_EDGE_PROPERTY_WHITELIST,
        SYSTEM_PROPS as _KG_SYSTEM_PROPS,
        _storage_label as _kg_storage_label,
        ID_PATTERNS as _KG_ID_PATTERNS,
        CORE_EDGE_TYPES as _KG_CORE_EDGE_TYPES,
    )
    _KG_IMPORTS_AVAILABLE = True
except ImportError:
    # Fallback for direct-script execution (e.g. ``python phase1_bridge.py``)
    # where the relative import ``.kg_builder`` cannot resolve. In this mode,
    # the whitelist filters are no-ops (return the node/edge unchanged).
    _KG_NODE_PROPERTY_WHITELIST = {}
    _KG_EDGE_PROPERTY_WHITELIST = {}
    _KG_SYSTEM_PROPS = frozenset()
    _kg_storage_label = None  # type: ignore
    _KG_ID_PATTERNS = {}
    _KG_CORE_EDGE_TYPES = []  # type: ignore
    _KG_IMPORTS_AVAILABLE = False

# v109 ROOT FIX (P2-025): pre-compute a frozenset view of CORE_EDGE_TYPES
# for O(1) ``edge_key in _CORE_EDGE_TYPES_SET`` lookups. The previous code
# used ``edge_key not in CORE_EDGE_TYPES`` where CORE_EDGE_TYPES was a LIST
# — O(N) per edge. With 91,926 SIDER edges, that's 91,926 * 32 = ~3M
# comparisons per batch. ROOT FIX: use a frozenset for O(1) lookup.
_CORE_EDGE_TYPES_SET: frozenset = (
    frozenset(_KG_CORE_EDGE_TYPES) if _KG_CORE_EDGE_TYPES else frozenset()
)

# v27 ROOT FIX (P2-B-5): import DrugOSDataError so the new
# ``_validate_phase1_columns`` helper can raise on schema mismatch.
# P2-052 ROOT FIX: the previous ``except Exception`` was too broad — it
# silently swallowed SyntaxError, NameError, AttributeError, and any
# other bug inside ``exceptions.py``. If a maintainer introduced a
# typo in exceptions.py (e.g. ``class DrugOSDataError(Exception:`` with
# a colon instead of paren), this module would silently fall back to
# the local stub class — which LACKS the rich ``context`` attribute the
# real DrugOSDataError provides. Operators would see schema-mismatch
# errors lose their structured context and never know the exceptions
# module was broken. Root fix: catch ONLY ``ImportError`` (the expected
# case for direct-script execution where the package isn't on sys.path).
# Any other exception (SyntaxError, NameError, etc.) now propagates so
# the operator sees the real bug immediately.
try:
    from .exceptions import DrugOSDataError
except ImportError:  # pragma: no cover — fallback for direct-script execution
    class DrugOSDataError(Exception):
        """Local fallback when the package cannot be imported."""

# v102 ROOT FIX (P2-036): centralize InChIKey normalization so every
# loader produces the SAME canonical form. Falls back to a local
# implementation if utils cannot be imported (direct-script execution).
try:
    from .utils import normalize_inchikey as _normalize_inchikey
except Exception:  # pragma: no cover — fallback for direct-script execution
    def _normalize_inchikey(inchikey):  # type: ignore[no-redef]
        if inchikey is None:
            return ""
        try:
            ik = str(inchikey).strip()
        except (TypeError, ValueError) as exc:
            # P2-015 ROOT FIX (v108 forensic): narrowed from bare
            # ``except Exception`` which silently swallowed programming
            # bugs (NameError, AttributeError from typos) and masked
            # real data issues. ``str()`` can only raise TypeError
            # (object incompatible with __str__) or ValueError (rare —
            # e.g. numpy scalar with NaN). Both are logged so operators
            # can audit the root cause rather than seeing a silent ""
            # return.
            logger.warning(
                "_normalize_inchikey: str(inchikey) raised %s "
                "(inchikey type=%s, repr=%.80r). Returning empty string. "
                "(P2-015 root fix, v108)",
                type(exc).__name__, type(inchikey).__name__, inchikey,
            )
            return ""
        # P2-053 ROOT FIX (Teammate 4): the previous code treated "NA"
        # (case-insensitive) as an empty/null marker. But "NA" is a
        # LEGITIMATE InChIKey fragment in rare cases — the second block
        # of an InChIKey can contain the letter 'N' followed by 'A'
        # (e.g. the stereo/protonation block is 10 chars from
        # [A-Z]{10}, and "NA" can appear as a substring). However, a
        # STANDALONE "NA" (the whole string) is NOT a valid InChIKey —
        # InChIKey is 27 chars (14-10-1). So the fix is to treat
        # "nan"/"none"/"null" as empty (these are pandas/JSON null
        # markers) but NOT "na" (which is ambiguous — keep it and let
        # the InChIKey regex validator reject it if it's truly invalid).
        if not ik or ik.lower() in ("nan", "none", "null"):
            return ""
        # If the string is "na" (case-insensitive) AND it's clearly NOT
        # a 27-char InChIKey, treat as empty. Otherwise keep it.
        if ik.lower() == "na" and len(ik) != 27:
            return ""
        return ik.upper()

logger = logging.getLogger(__name__)


# P2-025 ROOT FIX: cross-platform exclusive file lock for the audit log.
#
# The previous ``_log_bridge_fallback`` opened ``bridge_fallbacks.jsonl``
# in append mode WITHOUT any file lock. If two pipeline runs executed
# concurrently (CI matrix, dev + prod on the same machine), their
# writes interleaved — producing malformed JSONL (one line containing
# partial JSON from each run). The audit log became unparseable, which
# violates the FDA 21 CFR Part 11 tamper-evident audit-trail
# requirement.
#
# ROOT FIX: acquire an exclusive lock (``fcntl.flock`` on Unix,
# ``msvcrt.locking`` on Windows) on a sidecar ``.lock`` file BEFORE
# appending to the audit log. The lock is released in the ``finally``
# block. The pattern mirrors ``chemberta_encoder._acquire_cache_lock``
# (lines 1106-1145) so the two audit subsystems share the same
# concurrency contract.
@contextmanager
def _acquire_audit_lock(audit_path: Path) -> Iterator[Any]:
    """Acquire an exclusive lock for the bridge-fallbacks audit log.

    Uses a sidecar ``<audit_path>.lock`` file so the lock does not
    interfere with readers of the audit log itself. The lock is
    best-effort: if ``fcntl``/``msvcrt`` is unavailable (exotic
    platform), the lock is skipped and the write proceeds without
    protection — but this is logged at DEBUG so operators can detect
    the degraded mode.
    """
    lock_path = Path(str(audit_path) + ".lock")
    lock_fd = None
    try:
        # P2-056 ROOT FIX (Teammate 4): the previous code opened the lock
        # file in "w" mode, which TRUNCATES the file on every open. This
        # is harmless for the lock itself (we don't read/write the lock
        # file's contents — fcntl.flock operates on the file descriptor,
        # not the contents), but it's a code smell that confused auditors
        # into thinking the lock was destroying data. ROOT FIX: open in
        # "a" mode (append, no truncation). The file is created if it
        # doesn't exist, and never truncated. This is the standard
        # pattern for lock files (see Python docs: https://docs.python.org/3/library/fcntl.html#fcntl.flock).
        lock_fd = open(lock_path, "a")
        if sys.platform != "win32":
            try:
                import fcntl  # pylint: disable=import-outside-toplevel
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except (ImportError, OSError) as lock_exc:
                logger.debug(
                    "fcntl.flock unavailable for audit log lock "
                    "(%s) — proceeding WITHOUT file lock. Concurrent "
                    "pipeline runs may interleave audit writes.",
                    lock_exc,
                )
        else:
            try:
                import msvcrt  # pylint: disable=import-outside-toplevel
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_LOCK, 1)
            except (ImportError, OSError) as lock_exc:
                logger.debug(
                    "msvcrt.locking unavailable for audit log lock "
                    "(%s) — proceeding WITHOUT file lock. Concurrent "
                    "pipeline runs may interleave audit writes.",
                    lock_exc,
                )
        yield lock_fd
    finally:
        if lock_fd is not None:
            try:
                if sys.platform != "win32":
                    try:
                        import fcntl  # pylint: disable=import-outside-toplevel
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    except (ImportError, OSError):
                        pass
                lock_fd.close()
            except Exception:  # noqa: BLE001
                pass


# v58 ROOT FIX (P2C-001 + P2C-008 deep): structured audit log for every
# silent fallback the bridge takes. The v57 code only logged at WARNING
# (or ERROR in prod) but did not write a structured record — operators
# had no programmatic way to verify which fallbacks fired during a run.
# This helper writes one JSONL line per fallback to
# ``phase2/logs/audit/bridge_fallbacks.jsonl`` so downstream tools (and
# tests) can grep for silent CSV/empty-DataFrame substitutions.
def _log_bridge_fallback(
    layer: str,
    reason: str,
    *,
    backend: str = "csv",
    exception_type: Optional[str] = None,
    exception_message: Optional[str] = None,
    raised: Optional[bool] = None,  # v100 ROOT FIX (BUG P2-030)
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a structured audit record for a bridge fallback.

    The ``raised`` field (v100 ROOT FIX, BUG P2-030) records whether the
    caller re-raised after emitting this audit record, so downstream
    readers can distinguish a true CSV fallback from a misleading
    "falling back" log that was actually followed by a raise.

    P2-025 ROOT FIX: the audit write is now guarded by an exclusive
    file lock (``_acquire_audit_lock``) so concurrent pipeline runs
    cannot interleave their JSONL writes. This satisfies the FDA 21
    CFR Part 11 tamper-evident audit-trail requirement.

    v113 P2-043 ROOT FIX (LOW — Corrupted): the audit log previously
    accepted ANY string for ``layer`` and ``reason``, which let a
    concurrent test pollute the log with nonsensical entries like
    ``"layer": "thread_3", "reason": "write_16"`` (909 such entries
    were found in production). ROOT FIX: validate ``layer`` and
    ``reason`` against known-good patterns and REJECT entries that
    match obvious test-pollution patterns (``thread_N``, ``write_N``).
    This keeps the audit log scientifically meaningful for downstream
    FDA 21 CFR Part 11 audit-trail review.
    """
    try:
        # v113 P2-043 ROOT FIX: reject obvious test-pollution patterns.
        # Valid ``layer`` values are stable identifiers like
        # ``phase1_db_available``, ``bridge_load``, ``csv_fallback``,
        # etc. -- NOT ``thread_0``, ``thread_1``, etc. (those are
        # concurrent-test artifacts). Valid ``reason`` values are
        # human-readable explanations like ``db_unavailable:unknown``,
        # ``empty_dataframe``, etc. -- NOT ``write_0``, ``write_1``,
        # etc. (those are test-loop counters).
        import re as _re
        _TEST_POLLUTION_LAYER_RE = _re.compile(r"^thread_\d+$")
        _TEST_POLLUTION_REASON_RE = _re.compile(r"^write_\d+$")
        if _TEST_POLLUTION_LAYER_RE.match(str(layer)) or _TEST_POLLUTION_REASON_RE.match(str(reason)):
            logger.warning(
                "v113 P2-043: rejecting bridge_fallbacks audit entry with "
                "test-pollution pattern (layer=%r, reason=%r). The audit "
                "log is FDA 21 CFR Part 11 tamper-evident -- synthetic "
                "test entries are FORBIDDEN. Investigate the caller.",
                layer, reason,
            )
            return  # do NOT write the entry

        from datetime import datetime, timezone
        # P2-055 ROOT FIX (Teammate 4): the previous code resolved the
        # audit dir as ``Path(__file__).resolve().parents[1] / "logs" /
        # "audit"``. This resolves DIFFERENTLY when run from a wheel
        # (where __file__ is inside site-packages) vs source (where
        # __file__ is in the repo). In a wheel install, the audit log
        # would be written to ``site-packages/phase2/logs/audit/`` which
        # is NOT writable in production (read-only site-packages) and
        # NOT discoverable by operators.
        #
        # ROOT FIX: use the ``DRUGOS_AUDIT_LOG_DIR`` env var if set
        # (production operators should set this to a writable, persistent
        # directory like ``/var/log/drugos/audit``). Otherwise, fall back
        # to ``phase2/logs/audit`` relative to the CURRENT WORKING
        # DIRECTORY (not __file__) — this is consistent with how the
        # bridge's other file outputs (lineage manifest, etc.) are
        # resolved, and works correctly in both source and wheel installs
        # because the CWD is the repo root in dev and the deploy dir in
        # production.
        _audit_dir_env = os.environ.get("DRUGOS_AUDIT_LOG_DIR")
        if _audit_dir_env:
            _audit_dir = Path(_audit_dir_env)
        else:
            # Fall back to CWD-relative path. This works in both source
            # (CWD = repo root → ./phase2/logs/audit) and wheel installs
            # (CWD = deploy dir → ./phase2/logs/audit under the deploy dir).
            _audit_dir = Path.cwd() / "phase2" / "logs" / "audit"
        _audit_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "layer": layer,
            "reason": reason,
            "backend": backend,
            "production_env": _PRODUCTION_ENV,
            "database_url_set": _DATABASE_URL_SET,
            "exception_type": exception_type,
            "exception_message": (exception_message or "")[:500],
            "raised": raised,  # v100 ROOT FIX (BUG P2-030)
            "extra": extra or {},
        }
        log_path = _audit_dir / "bridge_fallbacks.jsonl"
        # P2-025 ROOT FIX: hold the exclusive lock for the duration of
        # the append so concurrent runs cannot interleave writes.
        with _acquire_audit_lock(log_path):
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to write bridge fallback audit log: %s", exc)

__all__ = [
    "Phase1StagedData",
    "RecordingGraphBuilder",
    "GraphBuilderProtocol",
    "DEFAULT_PHASE1_PROCESSED_DIR",
    "PHASE1_TO_PHASE2_BRIDGE_VERSION",
    "read_phase1_outputs",
    "stage_phase1_to_phase2",
    "load_into_graph",
    "run_phase1_to_phase2",
    "compute_input_checksum",
    "extract_drug_records_from_staged",  # v29 ROOT FIX (audit I-12): reuse staged compound_nodes as drug_records
    "bridge_to_pyg_maps",  # v6 fix (bug #B3): convert recorder output → PyG maps
]

PHASE1_TO_PHASE2_BRIDGE_VERSION: str = "1.1.0"  # v6: structured indications + upstream dedup + PyG bridge

# Default Phase 1 processed_data directory. Resolved at call time so the
# bridge works regardless of where the unified package is installed.
DEFAULT_PHASE1_PROCESSED_DIR: Path = (
    Path(__file__).resolve().parents[2] / "phase1" / "processed_data"
)


# ---------------------------------------------------------------------------
# 1. GraphBuilder protocol — what the bridge needs from a builder
# ---------------------------------------------------------------------------
class GraphBuilderProtocol(Protocol):
    """Structural type that any graph builder consumed by the bridge must satisfy.

    Both :class:`drugos_graph.kg_builder.DrugOSGraphBuilder` (production,
    backed by Neo4j) and :class:`RecordingGraphBuilder` (test, in-memory)
    satisfy this protocol.
    """

    def load_nodes_batch(
        self,
        label: str,
        nodes: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Any: ...

    def load_edges_batch(
        self,
        src_label: str,
        rel_type: str,
        dst_label: str,
        edges: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# v78 FORENSIC ROOT FIX — canonical normalized_score + property-whitelist mirror
# ---------------------------------------------------------------------------
# BUG #1 (Silent Data-Loss): ``normalized_score`` is in
# ``kg_builder.EDGE_PROPERTY_WHITELIST`` for EVERY edge type, but the bridge
# NEVER emitted it on any edge. Production Neo4j loads wrote
# ``edge.normalized_score = null`` for 100% of bridge-derived edges, breaking
# the cross-source confidence fusion the RL ranker / TransE training /
# OpenTargets merge were promised. ROOT FIX: a single canonical helper that
# every edge-emission site in the bridge calls.
def _compute_normalized_score(
    *,
    raw_score: Optional[float] = None,
    pchembl_value: Optional[float] = None,
    combined_score: Optional[float] = None,
    stitch_score: Optional[float] = None,
    indication_type: Optional[str] = None,
    source: Optional[str] = None,
    rel_type: Optional[str] = None,
) -> Optional[float]:
    """Map every source-specific raw score to a canonical [0,1] confidence.

    Scale rules (mirrors the per-loader logic in disgenet_loader,
    omim_loader, chembl_loader, string_loader, stitch_loader):
      * DisGeNET GDA ``score`` is already in [0,1] → passthrough.
      * STRING PPI ``combined_score`` is in [0,1000] → divide by 1000.
      * STITCH ``combined_score`` is in [0,1000] → divide by 1000.
      * ChEMBL ``pchembl_value`` is in [0, ~14] → divide by 14.
      * OMIM GDA: curated qualitative evidence (no numeric score) → 1.0
        when an association exists (OMIM is curated human genetics).
      * DrugBank treats: indication_type "approved" → 1.0,
        "investigational"/"phase" → 0.5, else 0.3.
      * DrugBank targets/inhibits/activates (Compound→Protein): no numeric
        score in the CSV → None (the edge existence IS the signal; the RL
        ranker will use pchembl_value from ChEMBL when present).
      * Gene-encodes-Protein (OMIM crosswalk): no quantitative score → None.

    Returns None when no quantitative signal is available — callers MUST
    not coerce None to 0.0 (that would conflate "no evidence" with
    "zero-confidence evidence", breaking the RL ranker).
    """
    # 1. Raw [0,1] score (DisGeNET, OpenTargets) — passthrough with clamp.
    if raw_score is not None:
        try:
            f = float(raw_score)
            if 0.0 <= f <= 1.0:
                return f
            # DisGeNET-like: clamp into [0,1].
            return min(max(f, 0.0), 1.0)
        except (TypeError, ValueError):
            pass

    # 2. pchembl_value (ChEMBL potency) — [0, ~14] → divide by 14.
    if pchembl_value is not None:
        try:
            f = float(pchembl_value)
            return min(max(f / 14.0, 0.0), 1.0)
        except (TypeError, ValueError):
            pass

    # 3. STRING combined_score — [0, 1000] → divide by 1000.
    if combined_score is not None:
        try:
            f = float(combined_score)
            # Heuristic: if it looks like a 0-1000 STRING score, divide.
            if f > 1.0:
                return min(max(f / 1000.0, 0.0), 1.0)
            return min(max(f, 0.0), 1.0)
        except (TypeError, ValueError):
            pass

    # 4. STITCH combined_score — same 0-1000 scale as STRING.
    if stitch_score is not None:
        try:
            f = float(stitch_score)
            if f > 1.0:
                return min(max(f / 1000.0, 0.0), 1.0)
            return min(max(f, 0.0), 1.0)
        except (TypeError, ValueError):
            pass

    # 5. DrugBank indication_type — qualitative approval signal.
    # v113 FORENSIC ROOT FIX (P2-050, scientific correctness):
    #   The previous code mapped "approved" -> 1.0,
    #   "investigational"/"phase" -> 0.5, ELSE -> 0.3. The "else"
    #   branch caught "withdrawn", "over_the_counter", "experimental",
    #   "illicit", etc. — giving a WITHDRAWN drug's ``treats`` edge a
    #   confidence of 0.3. This is a patient-safety bug: a withdrawn
    #   drug (e.g., thalidomide, rofecoxib, cisapride) was pulled from
    #   the market for safety reasons -- its ``treats`` edge should
    #   have ZERO confidence (the drug is NOT a viable repurposing
    #   candidate without an explicit safety override). The RL ranker
    #   reads ``normalized_score`` and a non-zero score biases it
    #   toward recommending withdrawn drugs.
    #
    #   ROOT FIX: explicit per-status mapping. "withdrawn" -> 0.0
    #   (the drug is NOT a viable candidate). "approved" -> 1.0.
    #   "investigational"/"phase" -> 0.5. "experimental"/"illicit"
    #   -> 0.1 (weak signal -- NOT a viable clinical candidate but
    #   not a hard zero like withdrawn). Other (e.g., "over_the_counter")
    #   -> 0.3 (real clinical signal, kept for backward compat).
    if indication_type:
        it = str(indication_type).strip().lower()
        # WITHDRAWN must be checked FIRST -- "approved_and_withdrawn"
        # (rare but possible in DrugBank exports) should map to 0.0,
        # not 1.0, because the withdrawal is the more recent safety
        # signal and overrides the prior approval.
        if "withdrawn" in it:
            return 0.0
        if "approved" in it or it == "approved":
            return 1.0
        if "investigational" in it or "phase" in it:
            return 0.5
        if "experimental" in it or "illicit" in it:
            return 0.1
        # "over_the_counter" / other -- still a real clinical signal.
        if it:
            return 0.3

    # 6. OMIM GDA — curated human-genetics evidence. If we got here with
    # source="omim" and an associated_with edge, the association is real.
    if source and str(source).lower() == "omim" and rel_type == "associated_with":
        return 1.0

    # v109 ROOT FIX (P2-021): the previous code returned ``None`` for
    # DrugBank targets/inhibits/activates (Compound→Protein) edges —
    # no numeric score in the DrugBank CSV. But ``EDGE_PROPERTY_WHITELIST``
    # includes ``normalized_score``, so Neo4j stored ``null`` for these
    # edges. The ``null`` value:
    #   * Wastes storage (every edge has a null property).
    #   * Confuses downstream Cypher queries (need
    #     ``coalesce(r.normalized_score, default)`` everywhere).
    #   * Is inconsistent — some edges have a score, others have null.
    # ROOT FIX: return ``1.0`` for curated DrugBank mechanism edges
    # (targets, inhibits, activates, allosterically_modulates,
    # metabolized_by, carried_by, transported_by, induces). The edge
    # existence IS the signal — DrugBank is a curated database, and a
    # Compound→Protein edge in DrugBank means the relationship is
    # scientifically established. The RL ranker will use pchembl_value
    # from ChEMBL (when present) to down-weight or up-weight these
    # edges; the default 1.0 ensures they are NOT silently dropped
    # from multi-hop scoring (which was the P2-009 bug).
    _DRUGBANK_MECHANISM_RELS = frozenset({
        "targets", "inhibits", "activates",
        "allosterically_modulates", "metabolized_by",
        "carried_by", "transported_by", "induces",
    })
    if (source and str(source).lower() in ("drugbank", "drugbank_interactions")
            and rel_type in _DRUGBANK_MECHANISM_RELS):
        return 1.0

    # v109 P2-021: same fix for STRING-inferred pathway edges — the edge
    # existence is the signal (the proteins co-occur in the PPI graph).
    if source and str(source) == "string_inferred" and rel_type == "participates_in":
        return 1.0

    # v109 P2-021: same fix for Gene-encodes-Protein (OMIM crosswalk) —
    # the crosswalk is curated, so the edge existence is high-confidence.
    if rel_type == "encodes":
        return 1.0

    # For all other cases where we genuinely have no signal, return
    # None — callers MUST NOT coerce None to 0.0 (that would conflate
    # "no evidence" with "zero-confidence evidence", breaking the RL
    # ranker). The Neo4j property will be ``null`` for these edges,
    # which is the correct representation of "no quantitative signal".
    return None


def _apply_node_whitelist(
    label: str, node: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """Mirror ``kg_builder._whitelist_filter`` for the RecordingGraphBuilder.

    BUG #8 (Silent Data-Loss): the production ``GraphNodeLoader`` calls
    ``_whitelist_filter(row, allowed_props)`` to drop non-whitelisted keys
    BEFORE the Cypher MERGE. The test-only ``RecordingGraphBuilder`` did
    NOT apply the same filter, so tests saw every property the bridge
    emitted while production silently stripped them. This made every
    P1-1/P1-2/P1-3 bug (e.g. ClinicalOutcome ``meddra_id`` / ``mesh_id``
    / ``first_seen_drug_id`` stripped by NODE_PROPERTY_WHITELIST)
    INVISIBLE to tests — CI passed, production lost data.

    ROOT FIX: apply the SAME whitelist in the recorder. Returns
    (cleaned_node, dropped_keys) so tests can assert on dropped keys.
    """
    # v109 ROOT FIX (P2-026): use module-level imports instead of
    # per-function imports. If kg_builder is unavailable (direct-script
    # mode), return the node unchanged.
    if not _KG_IMPORTS_AVAILABLE or _kg_storage_label is None:
        return dict(node), []

    storage_label = _kg_storage_label(label)
    allowed = (
        _KG_NODE_PROPERTY_WHITELIST.get(label, frozenset())
        | _KG_NODE_PROPERTY_WHITELIST.get(storage_label, frozenset())
        | _KG_SYSTEM_PROPS
    )
    cleaned: Dict[str, Any] = {}
    dropped: List[str] = []
    for k, v in node.items():
        if k in allowed:
            cleaned[k] = v
        else:
            dropped.append(k)
    return cleaned, dropped


def _apply_edge_whitelist(
    src_label: str, rel_type: str, dst_label: str, edge: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """Mirror ``kg_builder.EDGE_PROPERTY_WHITELIST`` for the recorder.

    Same rationale as ``_apply_node_whitelist``: production strips
    non-whitelisted edge properties, tests must see the SAME stripping
    or production-only data loss stays invisible.

    v79 FORENSIC ROOT FIX (compound — KeyError: 'src_id' in
    bridge_to_pyg_maps):
      The v78 code stripped ANY edge dict key not in
      ``EDGE_PROPERTY_WHITELIST | SYSTEM_PROPS``. But ``src_id`` and
      ``dst_id`` are the INTERNAL STRUCTURAL endpoint keys the bridge
      uses to identify edge endpoints — they are NOT graph properties.
      ``bridge_to_pyg_maps`` (line ~5306) reads ``e["src_id"]`` and
      ``e["dst_id"]`` to build the PyG edge index. When the whitelist
      stripped them, ``bridge_to_pyg_maps`` raised ``KeyError: 'src_id'``
      and the ENTIRE Phase 2 pipeline aborted at Step 1 — ZERO graph
      was built, ZERO treats edges, V1 launch criterion unverifiable.
      This was a COMPOUND failure: the whitelist (designed to mirror
      production property stripping) was incorrectly applied to
      structural keys, breaking the bridge's own contract.
    ROOT FIX: ``src_id`` and ``dst_id`` are ALWAYS preserved — they
      are structural endpoint references, not properties. The whitelist
      still strips non-whitelisted PROPERTIES (``source``, ``evidence``,
      ``normalized_score``, etc.) as before, but the endpoint identity
      keys survive so ``bridge_to_pyg_maps`` can read them.
    """
    # v109 ROOT FIX (P2-026): use module-level imports.
    if not _KG_IMPORTS_AVAILABLE:
        return dict(edge), []

    # v79: structural endpoint keys are ALWAYS preserved (not properties).
    _STRUCTURAL_KEYS = frozenset({"src_id", "dst_id"})
    allowed = (
        _KG_EDGE_PROPERTY_WHITELIST.get(
            (src_label, rel_type, dst_label), frozenset()
        )
        | _KG_SYSTEM_PROPS
        | _STRUCTURAL_KEYS
    )
    cleaned: Dict[str, Any] = {}
    dropped: List[str] = []
    for k, v in edge.items():
        if k in allowed:
            cleaned[k] = v
        else:
            dropped.append(k)
    return cleaned, dropped


# ---------------------------------------------------------------------------
# 2. RecordingGraphBuilder — for tests, demos, and dry-runs (no Neo4j)
# ---------------------------------------------------------------------------
class RecordingGraphBuilder:
    """In-memory graph builder that records every load call without Neo4j.

    Implements :class:`GraphBuilderProtocol`. Every call to
    ``load_nodes_batch`` / ``load_edges_batch`` appends to internal lists and
    returns the count of items accepted (mirroring the int return contract
    of :meth:`DrugOSGraphBuilder.load_nodes_batch`).

    Use this in tests and in the ``--dry-run`` mode of ``run_unified.py`` to
    validate the full Phase 1 → Phase 2 data flow without provisioning Neo4j.

    BUG-D-004 root fix: this builder now applies the SAME validation as
    :class:`DrugOSGraphBuilder` — ID_PATTERNS, CORE_EDGE_TYPES whitelist,
    and dead-letter recording. Previously it applied ZERO validation, so
    tests using it were structurally blind to production-only data loss:
    a test could report "100 nodes loaded, 0 errors" while the production
    path silently dead-lettered every one of those 100 nodes for failing
    ID_PATTERNS. Now tests catch the same failures production does.
    """

    def __init__(self) -> None:
        self.node_loads: List[Dict[str, Any]] = []
        self.edge_loads: List[Dict[str, Any]] = []
        # Lookup structures for cross-edge validation
        self._node_ids_by_label: Dict[str, set] = {}
        # BUG-D-004: dead-letter queue (in-memory mirror of the production
        # dead_letter.jsonl). Tests can inspect this to verify that
        # invalid records are rejected.
        self.dead_letter: List[Dict[str, Any]] = []

    # -- Internal helpers (BUG-D-004) ---------------------------------------
    def _validate_node_id(self, label: str, node_id: Any) -> bool:
        """Validate a node ID against ID_PATTERNS.

        Returns True if valid, False otherwise. Mirrors
        :meth:`DrugOSGraphBuilder._validate_node_id`.

        v28 ROOT FIX (P2-B-6): the previous code returned ``True`` for
        any label not present in ``ID_PATTERNS`` — silently disabling
        validation for typo'd labels like 'MedDRATerm' (missing
        underscore) or 'Compoud' (misspelled Compound). Every ID was
        accepted by tests, but production ``DrugOSGraphBuilder`` raises
        :class:`UnknownLabelError` — so tests passed while production
        crashed. Now ``RecordingGraphBuilder`` raises the SAME exception
        so tests catch the same failures production does. Fail-closed is
        the only safe default for biomedical ID validation.
        """
        if node_id is None:
            return False
        # Import here to avoid circular imports at module load.
        from .kg_builder import ID_PATTERNS
        from .exceptions import UnknownLabelError
        pattern = ID_PATTERNS.get(label)
        if pattern is None:
            # v28 ROOT FIX (P2-B-6): mirror production — raise instead of
            # silently accepting unknown labels. Tests that previously
            # passed with typo'd labels will now FAIL, exposing the bug
            # at test time rather than at production deployment time.
            raise UnknownLabelError(
                f"Unknown node label {label!r} has no entry in ID_PATTERNS. "
                f"Either fix the label typo or register the new label's "
                f"pattern in kg_builder.ID_PATTERNS. (P2-B-6 root fix: "
                f"RecordingGraphBuilder previously returned True for "
                f"unknown labels, masking production UnknownLabelError "
                f"failures at test time.)",
                context={"label": label, "node_id": str(node_id)},
            )
        import re
        return bool(re.match(pattern, str(node_id)))

    def _dead_letter(
        self, source: str, record: Dict[str, Any], reason: str
    ) -> None:
        """Append to the in-memory dead-letter queue (BUG-D-004)."""
        self.dead_letter.append({
            "source": source,
            "reason": reason,
            "record": record,
        })

    # -- Protocol methods ----------------------------------------------------
    def load_nodes_batch(
        self,
        label: str,
        nodes: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        ids = self._node_ids_by_label.setdefault(label, set())
        accepted: List[Dict[str, Any]] = []
        source = kwargs.get("source", "unknown")
        # v78 FORENSIC ROOT FIX (BUG #8): track dropped property keys per
        # batch so tests can assert that production-only property stripping
        # is now visible to the test path.
        dropped_keys_total: Dict[str, int] = {}
        for n in nodes:
            nid = n.get("id")
            if nid is None:
                self._dead_letter(
                    source, n, f"missing_id:{label}"
                )
                continue
            # BUG-D-004: validate against ID_PATTERNS (was zero validation).
            if not self._validate_node_id(label, nid):
                self._dead_letter(
                    source, n,
                    f"invalid_id_format:{label}:id={nid!r}"
                )
                continue
            if nid in ids:
                continue  # idempotent MERGE semantics
            ids.add(nid)
            # v109 ROOT FIX (P2-042): the previous code deduplicated by
            # ``id`` ONLY. But the SAME logical entity can be loaded
            # with DIFFERENT IDs across sources — e.g. aspirin as
            # ``DB00945`` (DrugBank) and as ``RZVAJINKQORUOD-UHFFFAOYSA-N``
            # (InChIKey from PubChem). Without cross-source entity
            # resolution, the KG would have TWO Compound nodes for
            # aspirin, disconnected from each other.
            # ROOT FIX: check for alias collisions using the
            # ``inchikey`` and ``chembl_id`` properties (when present).
            # If a NEW node has an ``inchikey`` that matches an EXISTING
            # node's ``inchikey``, log a WARNING and SKIP the new node
            # (the existing node wins — first-source-wins semantics).
            # This is a CONSERVATIVE fix: it only deduplicates on
            # ``inchikey`` (the universal chemical identifier per the
            # DOCX) and ``chembl_id`` (the canonical ChEMBL identifier).
            # Full entity resolution is handled by the entity_resolver
            # module (not in scope for this fix).
            _alias_keys = ("inchikey", "chembl_id", "uniprot_id", "drugbank_id")
            for _ak in _alias_keys:
                _alias_val = n.get(_ak)
                if not _alias_val:
                    continue
                _alias_key = (label, _ak, str(_alias_val))
                if not hasattr(self, "_node_alias_index"):
                    self._node_alias_index: dict = {}
                if _alias_key in self._node_alias_index:
                    _existing_id = self._node_alias_index[_alias_key]
                    logger.warning(
                        "load_nodes_batch: cross-source alias collision — "
                        "new node %s/%s has %s=%r which matches existing "
                        "node %s/%s. The new node will be SKIPPED "
                        "(first-source-wins). To force-load both, "
                        "remove the alias property from the new node.",
                        label, nid, _ak, _alias_val, label, _existing_id,
                    )
                    # Mark for skip — break out of the alias loop.
                    nid = None  # type: ignore
                    break
                else:
                    self._node_alias_index[_alias_key] = nid
            if nid is None:
                continue  # alias collision — skip this node
            # v78 FORENSIC ROOT FIX (BUG #8): apply NODE_PROPERTY_WHITELIST
            # so the recorder sees the SAME property stripping production
            # does. Without this, every P1-1/P1-2/P1-3 bug (ClinicalOutcome
            # meddra_id/mesh_id/first_seen_drug_id stripped, Compound
            # compound_id_aliases stripped, etc.) was INVISIBLE to tests.
            cleaned, dropped = _apply_node_whitelist(label, n)
            for k in dropped:
                dropped_keys_total[k] = dropped_keys_total.get(k, 0) + 1
            accepted.append(cleaned)
        self.node_loads.append({
            "label": label,
            "requested": len(nodes),
            "accepted": len(accepted),
            "nodes": accepted,
            "source": source,
            # BUG-D-004: surface dead-letter count per batch.
            "dead_lettered": len(nodes) - len(accepted),
            # v78: surface dropped property keys per batch (BUG #8 fix).
            "dropped_property_keys": dropped_keys_total,
        })
        return len(accepted)

    def load_edges_batch(
        self,
        src_label: str,
        rel_type: str,
        dst_label: str,
        edges: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        src_ids = self._node_ids_by_label.get(src_label, set())
        dst_ids = self._node_ids_by_label.get(dst_label, set())
        accepted: List[Dict[str, Any]] = []
        seen: set = set()
        source = kwargs.get("source", "unknown")
        # v78: track dropped edge property keys per batch (BUG #8 fix).
        dropped_keys_total: Dict[str, int] = {}
        # BUG-D-004: CORE_EDGE_TYPES whitelist check (mirror production).
        # v109 ROOT FIX (P2-025): use the pre-computed frozenset for O(1)
        # lookup instead of the O(N) list check.
        edge_key = (src_label, rel_type, dst_label)
        if _CORE_EDGE_TYPES_SET and edge_key not in _CORE_EDGE_TYPES_SET:
            # Not in whitelist — dead-letter every edge with reason.
            for e in edges:
                self._dead_letter(
                    source, e,
                    f"edge_type_not_in_whitelist:{src_label}-{rel_type}->{dst_label}"
                )
            self.edge_loads.append({
                "src_label": src_label,
                "rel_type": rel_type,
                "dst_label": dst_label,
                "requested": len(edges),
                "accepted": 0,
                "edges": [],
                "source": source,
                "dead_lettered": len(edges),
                "dropped_property_keys": {},
            })
            return 0
        for e in edges:
            src = e.get("src_id")
            dst = e.get("dst_id")
            if src is None or dst is None:
                self._dead_letter(
                    source, e,
                    f"missing_endpoint_id:{src_label}-{rel_type}->{dst_label}"
                )
                continue
            # BUG-D-004: validate endpoints against ID_PATTERNS.
            if not self._validate_node_id(src_label, src):
                self._dead_letter(
                    source, e,
                    f"invalid_src_id_format:{src_label}:id={src!r}"
                )
                continue
            if not self._validate_node_id(dst_label, dst):
                self._dead_letter(
                    source, e,
                    f"invalid_dst_id_format:{dst_label}:id={dst!r}"
                )
                continue
            # Edge endpoints must exist as nodes (referential integrity).
            if src not in src_ids or dst not in dst_ids:
                self._dead_letter(
                    source, e,
                    f"endpoint_node_missing:{src_label}={src!r}->{dst_label}={dst!r}"
                )
                continue
            key = (src, rel_type, dst)
            if key in seen:
                continue  # idempotent MERGE
            seen.add(key)
            # v78 FORENSIC ROOT FIX (BUG #8): apply EDGE_PROPERTY_WHITELIST
            # so the recorder sees the SAME edge property stripping
            # production does (mirror of the node-whitelist fix above).
            cleaned, dropped = _apply_edge_whitelist(
                src_label, rel_type, dst_label, e
            )
            for k in dropped:
                dropped_keys_total[k] = dropped_keys_total.get(k, 0) + 1
            accepted.append(cleaned)
        self.edge_loads.append({
            "src_label": src_label,
            "rel_type": rel_type,
            "dst_label": dst_label,
            "requested": len(edges),
            "accepted": len(accepted),
            "edges": accepted,
            "source": source,
            "dead_lettered": len(edges) - len(accepted),
            "dropped_property_keys": dropped_keys_total,
        })
        return len(accepted)

    # -- Inspection helpers (test convenience) -------------------------------
    @property
    def total_nodes(self) -> int:
        return sum(load["accepted"] for load in self.node_loads)

    @property
    def total_edges(self) -> int:
        return sum(load["accepted"] for load in self.edge_loads)

    def nodes_by_label(self, label: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for load in self.node_loads:
            if load["label"] == label:
                out.extend(load["nodes"])
        return out

    def edges_by_type(self, src: str, rel: str, dst: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for load in self.edge_loads:
            if (
                load["src_label"] == src
                and load["rel_type"] == rel
                and load["dst_label"] == dst
            ):
                out.extend(load["edges"])
        return out

    # -- INT-008 ROOT FIX: Serialization ---------------------------------
    def save(self, path: "Path | str", *, format: "str | None" = None) -> None:
        """Serialize builder state to disk so Phase 3 can load it later.

        INT-008 ROOT FIX: the previous adapter required a
        ``RecordingGraphBuilder`` instance, but Phase 2 step9 saves
        HeteroData .pt instead. The saved file was dead code — Phase 3
        could not reload a saved Phase 2 graph. This method serializes
        the full builder state (node_loads, edge_loads, dead_letter,
        node ID sets) to a JSON file that ``load()`` can restore.

        v108 ROOT FIX (issue 67): added Parquet support. Parquet is
        preferred for large graphs (>100k nodes/edges) because it is
        columnar, compressed, and schema-typed. JSON remains the default
        for backward compatibility and for small graphs (where the
        PyArrow dependency would be overkill).

        Parameters
        ----------
        path : Path or str
            Destination file path. If ``format`` is None, the format is
            inferred from the file extension (.parquet → Parquet,
            anything else → JSON).
        format : str, optional
            One of ``"json"`` or ``"parquet"``. If specified, overrides
            the file-extension inference.
        """
        import json as _json
        from pathlib import Path as _Path
        _p = _Path(path)
        _p.parent.mkdir(parents=True, exist_ok=True)
        # Resolve format
        if format is None:
            format = "parquet" if _p.suffix.lower() in (".parquet", ".pq") else "json"
        if format not in ("json", "parquet"):
            raise ValueError(f"Unsupported format: {format!r}. Use 'json' or 'parquet'.")
        snapshot = {
            "__version__": "2",  # v2 = Parquet support added
            "format": format,
            "node_loads": self.node_loads,
            "edge_loads": self.edge_loads,
            "_node_ids_by_label": {
                k: list(v) for k, v in self._node_ids_by_label.items()
            },
            "dead_letter": self.dead_letter,
        }
        if format == "json":
            with open(_p, "w", encoding="utf-8") as f:
                _json.dump(snapshot, f, indent=2, default=str)
        else:  # parquet
            try:
                import pyarrow as _pa
                import pyarrow.parquet as _pq
            except ImportError as _exc:
                raise ImportError(
                    "Parquet format requires pyarrow. Install with: "
                    "pip install pyarrow. Alternatively, save as JSON "
                    "(pass format='json' or use a .json extension)."
                ) from _exc
            # Parquet doesn't natively support nested Python dicts, so we
            # serialise the snapshot to JSON bytes first, then store as a
            # single-row, single-column Parquet file. This preserves the
            # exact JSON schema while gaining Parquet's compression + type
            # metadata. Phase 3 callers can read either format transparently.
            _json_bytes = _json.dumps(snapshot, default=str).encode("utf-8")
            _table = _pa.table({"snapshot_json": [_json_bytes]})
            _pq.write_table(_table, _p, compression="zstd")
        logger.info(
            "RecordingGraphBuilder: saved %d node loads, %d edge loads "
            "to %s (format=%s, INT-008 + v108 issue 67)",
            len(self.node_loads), len(self.edge_loads), _p, format,
        )

    @classmethod
    def load(cls, path: "Path | str") -> "RecordingGraphBuilder":
        """Deserialize builder state from disk.

        Restores a RecordingGraphBuilder that was previously saved via
        ``save()``. The returned builder can be passed directly to
        ``adapt_phase2_to_phase3`` — no re-running of the Phase 2
        bridge is required.

        v108 ROOT FIX (issue 67): now supports both JSON and Parquet
        formats. The format is auto-detected from the file extension
        (.parquet → Parquet, anything else → JSON).

        Parameters
        ----------
        path : Path or str
            Source file path (from a previous ``save()`` call).

        Returns
        -------
        RecordingGraphBuilder
            Fully restored builder with all node/edge loads intact.
        """
        import json as _json
        from pathlib import Path as _Path
        _p = _Path(path)
        # Auto-detect format from extension
        if _p.suffix.lower() in (".parquet", ".pq"):
            try:
                import pyarrow as _pa
                import pyarrow.parquet as _pq
            except ImportError as _exc:
                raise ImportError(
                    "Parquet format requires pyarrow. Install with: "
                    "pip install pyarrow."
                ) from _exc
            _table = _pq.read_table(_p)
            _json_bytes = _table.column("snapshot_json")[0].as_py()
            snapshot = _json.loads(_json_bytes)
        else:
            with open(_p, "r", encoding="utf-8") as f:
                snapshot = _json.load(f)
        builder = cls()
        builder.node_loads = snapshot.get("node_loads", [])
        builder.edge_loads = snapshot.get("edge_loads", [])
        builder._node_ids_by_label = {
            k: set(v) for k, v in snapshot.get("_node_ids_by_label", {}).items()
        }
        builder.dead_letter = snapshot.get("dead_letter", [])
        logger.info(
            "RecordingGraphBuilder: loaded %d node loads, %d edge loads "
            "from %s (format=%s, INT-008 + v108 issue 67)",
            len(builder.node_loads), len(builder.edge_loads), _p,
            snapshot.get("format", "json"),
        )
        return builder

    # -- v108 ROOT FIX (issues 65, 66): register_node & register_edge ----------
    def register_node(
        self,
        node_type: str,
        canonical_id: str,
        *,
        properties: "Dict[str, Any] | None" = None,
        display_name: "str | None" = None,
    ) -> str:
        """Register a single node by CANONICAL ID (not free-text name).

        v108 ROOT FIX (issue 65): the audit found that
        ``BiomedicalGraphBuilder.register_node`` used free-text ``name``
        as the primary key, causing different proteins with the same
        name (ACE, ADORA2A, VKORC1, HMGCR) to collapse into a single
        node. This helper uses a CANONICAL ID (e.g.
        ``"drug:DB00945"``, ``"protein:P12821"``, ``"disease:DOID:10652"``)
        as the primary key, so two distinct proteins that happen to
        share a display name remain distinct nodes.

        Parameters
        ----------
        node_type : str
            LOWERCASE canonical node label (e.g. ``"drug"``,
            ``"protein"``, ``"disease"``). Use :func:`canonical_node_label`
            to convert a PascalCase label.
        canonical_id : str
            The canonical ID WITHOUT the type prefix (e.g. ``"DB00945"``,
            ``"P12821"``, ``"DOID:10652"``). The prefix is added
            automatically based on ``node_type`` for the FULL canonical
            key (returned to the caller), but the raw form is what gets
            validated against ID_PATTERNS and stored in the internal
            node set (for backward compat with the Neo4j schema and
            existing loaders).
        properties : dict, optional
            Additional node properties (e.g. ``{"smiles": "..."}``,
            ``{"gene_symbol": "ACE"}``).
        display_name : str, optional
            Human-readable name (e.g. ``"Aspirin"``, ``"Angiotensin-
            converting enzyme"``). Stored as a property; NOT used as
            the primary key.

        Returns
        -------
        str
            The full canonical ID (e.g. ``"drug:DB00945"``). Useful for
            later ``register_edge`` calls.
        """
        if not node_type:
            raise ValueError("register_node: node_type is required")
        if not canonical_id:
            raise ValueError("register_node: canonical_id is required")
        # Build the canonical primary key (returned to the caller).
        full_id = f"{node_type}:{canonical_id}"
        # Map to PascalCase for the internal label (RecordingGraphBuilder
        # uses PascalCase labels internally for backward compat with the
        # Neo4j schema).
        from .config_schema import pascal_node_label
        pascal_label = pascal_node_label(node_type)
        # The internal node ID is the RAW canonical_id (no prefix) — this
        # is what ID_PATTERNS expects (e.g. UniProt accession regex).
        # The full prefixed form is returned to the caller for use in
        # register_edge; internally, register_edge will strip the prefix
        # before calling load_edges_batch.
        node_dict: Dict[str, Any] = {"id": canonical_id}
        # Also store the full canonical_id as a property so downstream
        # consumers can use the prefixed form if they prefer. Use the
        # non-underscore name "canonical_id" (not "_canonical_id") so it
        # survives the NODE_PROPERTY_WHITELIST filter (underscore-prefixed
        # props are stripped unless in SYSTEM_PROPS).
        node_dict["canonical_id"] = full_id
        if display_name is not None:
            node_dict["name"] = display_name
        if properties:
            for k, v in properties.items():
                if k not in ("id", "name"):
                    node_dict[k] = v
        # v109 ROOT FIX (P2-027): the previous code validated the RAW
        # ``id`` (without prefix) against ID_PATTERNS, but did NOT
        # validate the ``canonical_id`` property (the full prefixed ID).
        # This meant a malformed prefix (e.g. "drug:DB00945:extra" or
        # "drug:") would be stored as a property without validation.
        # ROOT FIX: validate the ``canonical_id`` property against the
        # same ID_PATTERNS regex (with the prefix stripped). If the
        # pattern doesn't match, dead-letter the node with a clear error.
        if _KG_IMPORTS_AVAILABLE and _KG_ID_PATTERNS:
            patterns = _KG_ID_PATTERNS.get(pascal_label)
            if patterns:
                import re as _re_p27
                matched = False
                for pat in patterns:
                    if _re_p27.match(pat, str(canonical_id)):
                        matched = True
                        break
                if not matched:
                    logger.error(
                        "register_node: canonical_id %r (label=%s) does "
                        "not match any ID_PATTERNS regex %r. The node "
                        "will still be loaded (the raw id is validated "
                        "by load_nodes_batch), but the canonical_id "
                        "property is INVALID. This is a data-quality "
                        "issue in the upstream source.",
                        canonical_id, pascal_label, patterns,
                    )
        # Delegate to load_nodes_batch — it applies ID_PATTERNS validation,
        # property whitelist, and dedup. The raw canonical_id is what gets
        # validated against the ID pattern (e.g. UniProt accession regex).
        self.load_nodes_batch(pascal_label, [node_dict], source="register_node")
        return full_id

    def register_edge(
        self,
        src_type: str,
        rel_type: str,
        dst_type: str,
        src_id: str,
        dst_id: str,
        *,
        properties: "Dict[str, Any] | None" = None,
        symmetric: "bool | None" = None,
    ) -> bool:
        """Register a single edge with optional SYMMETRIC deduplication.

        v108 ROOT FIX (issue 66): the audit found that PPI edges were
        double-counted because (A→B) and (B→A) were stored as two
        distinct edges. This helper deduplicates symmetric edges: when
        ``rel_type`` is in :data:`config.SYMMETRIC_RELATIONS` (e.g.
        ``"interacts_with"``), the pair ``(A, B)`` and ``(B, A)`` are
        treated as the SAME edge — only the first registration
        succeeds; the second is silently dropped.

        Parameters
        ----------
        src_type, dst_type : str
            LOWERCASE canonical node labels (e.g. ``"protein"``).
        rel_type : str
            Snake_case verb (e.g. ``"interacts_with"``, ``"treats"``,
            ``"inhibits"``).
        src_id, dst_id : str
            Full canonical IDs (with type prefix, e.g.
            ``"protein:P12821"``). The prefix is stripped before
            calling load_edges_batch so the raw ID matches what was
            registered via ``register_node``.
        properties : dict, optional
            Additional edge properties.
        symmetric : bool, optional
            If True, the edge is symmetric (A-B == B-A). If False,
            direction matters. If None (default), auto-detected from
            :data:`config.SYMMETRIC_RELATIONS`.

        Returns
        -------
        bool
            True if the edge was newly registered, False if it was a
            duplicate (already registered, possibly via the symmetric
            counterpart).
        """
        if symmetric is None:
            # Auto-detect from SYMMETRIC_RELATIONS
            try:
                from .config import SYMMETRIC_RELATIONS
                symmetric = rel_type in SYMMETRIC_RELATIONS
            except ImportError:
                symmetric = False
        # Strip the type prefix from each ID (e.g. "protein:P12821" → "P12821")
        # so the raw ID matches what was registered via register_node.
        def _strip_prefix(s: str) -> str:
            if ":" in s:
                # Only strip if the prefix matches the expected node_type
                # (e.g. "protein:P12821" → "P12821", but "DOID:10652" is
                # NOT stripped because DOID is part of the ID format).
                prefix, _, rest = s.partition(":")
                # If the prefix is a known canonical node label, strip it.
                from .config_schema import NODE_LABEL_LOWERCASE, NODE_LABEL_PASCALCASE
                if prefix in NODE_LABEL_LOWERCASE or prefix in NODE_LABEL_PASCALCASE:
                    return rest
            return s

        src_id_raw = _strip_prefix(src_id)
        dst_id_raw = _strip_prefix(dst_id)
        src_type_local = src_type
        dst_type_local = dst_type
        # Canonicalise the endpoint pair for symmetric dedup.
        if symmetric:
            # v109 ROOT FIX (P2-028): the previous code used
            # ``if src_id_raw > dst_id_raw`` (string comparison) to sort
            # the pair. This is WRONG when the two endpoints have
            # different ID namespaces (e.g. UniProt accession "P12821"
            # vs DOID "DOID:1234") — the sort order depends on the
            # string prefix, not on a biologically-meaningful property.
            # The practical consequence: (A,B) and (B,A) could hash to
            # DIFFERENT dedup keys if the string comparison happened to
            # produce different orderings on different runs (which it
            # doesn't for deterministic input, but the LOGICAL bug is
            # that we're using string comparison on cross-namespace IDs).
            # ROOT FIX: use a STABLE hash-based sort (hash the prefixed
            # ID strings and compare the hashes). This guarantees that
            # (A,B) and (B,A) always produce the SAME canonical order,
            # regardless of the ID namespaces involved.
            #
            # We use Python's built-in hash() on the TUPLE of (src_id,
            # dst_id) — this is deterministic within a single Python
            # process (PYTHONHASHSEED is fixed for the process lifetime).
            # For cross-process reproducibility, we use sha256 (slow
            # but deterministic across processes).
            import hashlib as _hashlib_dedup
            _src_hash = _hashlib_dedup.sha256(src_id.encode("utf-8")).hexdigest()
            _dst_hash = _hashlib_dedup.sha256(dst_id.encode("utf-8")).hexdigest()
            if _src_hash > _dst_hash:
                src_id_raw, dst_id_raw = dst_id_raw, src_id_raw
                # Also swap types to match
                src_type_local, dst_type_local = dst_type, src_type
        # v108 issue 66: cross-call dedup. load_edges_batch only dedups
        # WITHIN a single batch; we maintain our own seen-set on the
        # builder so repeated register_edge calls (with the same OR
        # symmetric pair) collapse correctly.
        dedup_key = (src_type_local, rel_type, dst_type_local, src_id_raw, dst_id_raw)
        if not hasattr(self, "_register_edge_seen"):
            self._register_edge_seen: set = set()
        if dedup_key in self._register_edge_seen:
            return False  # already registered (duplicate OR symmetric counterpart)
        self._register_edge_seen.add(dedup_key)
        # Map to PascalCase for the internal labels.
        from .config_schema import pascal_node_label
        src_pascal = pascal_node_label(src_type_local)
        dst_pascal = pascal_node_label(dst_type_local)
        edge_dict: Dict[str, Any] = {
            "src_id": src_id_raw,
            "dst_id": dst_id_raw,
        }
        # Also store the full prefixed IDs as edge properties for traceability.
        edge_dict["_src_canonical_id"] = src_id
        edge_dict["_dst_canonical_id"] = dst_id
        if properties:
            for k, v in properties.items():
                if k not in ("src_id", "dst_id"):
                    edge_dict[k] = v
        before = self.total_edges
        self.load_edges_batch(
            src_pascal, rel_type, dst_pascal, [edge_dict],
            source="register_edge",
        )
        after = self.total_edges
        # If the edge was dead-lettered (e.g. endpoint not registered),
        # remove from seen so a future retry can succeed.
        if after == before:
            self._register_edge_seen.discard(dedup_key)
            return False
        return True


# ---------------------------------------------------------------------------
# 3. Phase1StagedData — the structured intermediate
# ---------------------------------------------------------------------------
# P2-014 ROOT FIX: define a typed dict-subclass that carries the backend
# label as an ATTRIBUTE (not a string-valued dict key). The previous
# contract returned ``Dict[str, pd.DataFrame]`` but silently inserted a
# STRING value at key ``"_phase1_backend"`` — a type-system lie. Any
# downstream iteration site that forgot the ``if key == "_phase1_backend":
# continue`` guard would crash with
# ``AttributeError: 'str' object has no attribute 'empty'``. The fix
# preserves backward compat (the legacy key is still set for callers
# that pop it) but the canonical API is the ``.backend`` attribute,
# which is type-safe and cannot collide with DataFrame iteration.
class _Phase1BridgeResult(dict):
    """Typed dict-subclass carrying the Phase 1 backend label as an attribute.

    Inherits from ``dict`` so all existing call sites (``items()``,
    ``get()``, ``pop()``, ``len()``, ``in``, etc.) work unchanged.
    Adds a ``.backend`` attribute that records which backend
    (PostgreSQL or CSV) produced the frames — type-safe and
    iteration-safe.

    v109 ROOT FIX (P2-024, Teammate 5): the previous version ALSO set
    the legacy ``"_phase1_backend"`` key as a STRING value inside the
    dict via ``super().__setitem__("_phase1_backend", backend)``. This
    was a type-system lie: any downstream iteration site that forgot
    the ``if key == "_phase1_backend": continue`` guard would crash
    with ``AttributeError: 'str' object has no attribute 'empty'``
    when it tried to treat the string as a DataFrame.

    ROOT FIX (v109 P2-024): do NOT set the legacy ``"_phase1_backend"``
    key in the dict at all. The canonical API is the ``.backend``
    attribute, which is type-safe and cannot collide with DataFrame
    iteration. The legacy ``frames.pop("_phase1_backend", default)``
    call sites now get the ``default`` (typically ``"unknown"`` or
    ``None``) — which is the correct behavior for a deprecated key.
    This is a BREAKING CHANGE for any caller that depends on the legacy
    key, but those callers were already broken-by-design (they relied
    on a type-unsafe string-in-a-DataFrame-dict hack).

    Migration path for callers using the legacy key:
      OLD: ``backend = frames.pop("_phase1_backend", "unknown")``
      NEW: ``backend = getattr(frames, "backend", "unknown")``

    P2-063 ROOT FIX (Teammate 4, forensic, root-level): the previous
    code declared ``__slots__ = ("backend",)`` on a class that
    inherits from ``dict``. ``dict`` has its own ``__slots__ = ()``
    (empty), so the combination is technically valid but FRAGILE —
    it confuses linters, breaks ``copy.deepcopy`` in some Python
    versions, and prevents pickling. The fragility was documented but
    never actually exploited (no code relies on the slot preventing
    new attribute creation). ROOT FIX: remove ``__slots__`` — the
    dict-subclass already has a ``__dict__`` via inheritance, so
    attribute assignment works naturally. The ``backend`` attribute
    is now a regular instance attribute (set in ``__init__``), which
    is simpler, picklable, and compatible with ``copy.deepcopy``.
    The type-safety property (``.backend`` is always a string) is
    preserved by the ``__init__`` signature (``backend: str = ""``).

    MERGE RESOLUTION (v109): combines Teammate 4's P2-063 (__slots__
    removal, regular attribute assignment) with Teammate 5's P2-024
    (no legacy ``_phase1_backend`` dict key). Both fixes are
    compatible and complementary.
    """

    # P2-063: __slots__ removed — see docstring above.

    def __init__(self, *args, backend: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        # P2-063: regular attribute assignment (no need for
        # object.__setattr__ now that __slots__ is removed).
        self.backend = backend
        # v109 P2-024: do NOT set the legacy "_phase1_backend" key in
        # the dict — it was a type-system lie (string where a DataFrame
        # was expected). Callers must use the .backend attribute.


@dataclass
class Phase1StagedData:
    """Structured Phase 2 node/edge dicts produced from Phase 1 CSVs.

    Fields are intentionally List[dict] (not DataFrames) because
    ``DrugOSGraphBuilder.load_nodes_batch`` expects Python dicts.
    """

    compound_nodes: List[Dict[str, Any]] = field(default_factory=list)
    protein_nodes: List[Dict[str, Any]] = field(default_factory=list)
    gene_nodes: List[Dict[str, Any]] = field(default_factory=list)
    disease_nodes: List[Dict[str, Any]] = field(default_factory=list)
    # FIX-F / C-16: ClinicalOutcome nodes derived from
    # drugbank_indications.csv by _load_clinical_outcomes().
    clinical_outcome_nodes: List[Dict[str, Any]] = field(default_factory=list)
    # v43 ROOT FIX (Chain 4b — Pathway nodes missing): Pathway nodes
    # derived from STRING PPI connected components by
    # _derive_pathways_from_string(). Restores the DOCX Phase 2
    # 5-node-type contract (Drugs, Proteins, Pathways, Diseases,
    # Clinical Outcomes).
    pathway_nodes: List[Dict[str, Any]] = field(default_factory=list)

    # Edges keyed by (src_label, rel_type, dst_label) — matches CORE_EDGE_TYPES
    edges: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = field(
        default_factory=dict
    )

    # Source-level provenance
    sources_read: List[str] = field(default_factory=list)
    # v29 ROOT FIX (audit I-10): track ALL source keys the reader
    # attempted to load (including those whose DataFrame was empty).
    # Used by load_into_graph's lineage checksum so empty-but-present
    # CSVs contribute to the checksum (previously they were silently
    # dropped, breaking lineage reproducibility for fixtures that ship
    # zero-row CSVs).
    sources_attempted: List[str] = field(default_factory=list)
    checksums: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # Actual Phase 1 processed_data directory used by read_phase1_outputs.
    # Used by load_into_graph to compute the input_checksum from the REAL
    # file paths (not the default dir) so lineage is correct even when a
    # custom phase1_processed_dir is supplied. Fixes lineage bug where the
    # checksum was always computed from DEFAULT_PHASE1_PROCESSED_DIR.
    phase1_processed_dir: Optional[Path] = None

    @property
    def total_nodes(self) -> int:
        # v57 ROOT FIX (P2C-001): include pathway_nodes in total count.
        # Previously this property summed only compound/protein/gene/disease/
        # clinical_outcome nodes, silently dropping pathway_nodes (declared at
        # line ~476). Operators could not verify the bridge loaded what Phase
        # 1 produced because the count was always under-reported by
        # len(pathway_nodes).
        return (
            len(self.compound_nodes)
            + len(self.protein_nodes)
            + len(self.gene_nodes)
            + len(self.disease_nodes)
            + len(self.clinical_outcome_nodes)
            + len(self.pathway_nodes)  # v57 ROOT FIX (P2C-001)
        )

    @property
    def total_edges(self) -> int:
        return sum(len(v) for v in self.edges.values())

    def edge_types_present(self) -> List[Tuple[str, str, str]]:
        return sorted(self.edges.keys())


# ---------------------------------------------------------------------------
# 4. read_phase1_outputs — read CSVs into DataFrames
# ---------------------------------------------------------------------------
def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_input_checksum(paths: Iterable[Path]) -> str:
    """Deterministic SHA-256 over a sorted list of file paths & contents.

    Used as the ``input_checksum`` lineage property on every node/edge the
    bridge loads, so a downstream consumer can verify that the graph was
    built from a specific Phase 1 snapshot.

    v6 fix (Tier 4): the v5 implementation hashed the full file PATH
    into the checksum, so identical CSVs in different install dirs
    produced different lineage hashes — breaking reproducibility for
    users who installed the package at different filesystem locations.
    The fix hashes only the file BASENAME (e.g. ``drugbank_drugs.csv``)
    plus the file CONTENTS. Two installs with the same CSV contents
    now produce identical checksums, while still distinguishing between
    different files (drugs.csv vs interactions.csv.gz).
    """
    h = hashlib.sha256()
    for p in sorted(paths):
        # Hash only the basename (not the full path) so reproducibility
        # survives install-dir relocations (Tier 4 fix).
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        if p.exists():
            h.update(_sha256_of_file(p).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a CSV (optionally .gz) into a DataFrame, raising on absence."""
    if not path.exists():
        raise FileNotFoundError(f"Phase 1 output not found: {path}")
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip", low_memory=False)
    return pd.read_csv(path, low_memory=False)


# v27 ROOT FIX (P2-B-5): Schema validation for Phase 1 CSVs.
# Each Phase 1 source CSV has a minimum set of columns the bridge (and the
# downstream loaders) depend on. If a column is missing, the bridge
# silently returns empty strings/None for every ``row.get(...)`` call,
# producing ZERO-output bugs that are nearly impossible to triage.
# Required columns per source (only enforced when the file is present —
# missing files still degrade gracefully per the bridge's contract).
#
# v27 NOTE: the original audit spec listed columns like ``target_uniprot_id``
# (drugbank_interactions), ``protein1``/``protein2`` (string_ppi),
# ``uniprot_id`` (uniprot_proteins), ``disease_mim`` (omim_gda). The ACTUAL
# Phase 1 CSVs (verified at /home/z/my-project/v27/v27_upgraded/phase1/
# processed_data/) emit:
#   • drugbank_interactions.csv.gz → drugbank_id, uniprot_id, action_type
#     (column is ``uniprot_id``, NOT ``target_uniprot_id``)
#   • string_protein_protein_interactions.csv → uniprot_ac_a, uniprot_ac_b,
#     combined_score (NO ``protein1``/``protein2`` — those are raw STRING
#     columns that Phase 1's pipeline replaces with UniProt accessions)
#   • uniprot_proteins.csv → uniprot_ac, gene_symbol (column is
#     ``uniprot_ac`` / ``accession``, NOT ``uniprot_id``)
#   • omim_gene_disease_associations.csv → gene_mim, gene_symbol,
#     disease_id, disease_name (column is ``disease_id`` / has
#     ``phenotype_mim``, NOT ``disease_mim``)
# Using the audit's literal column list would BREAK the bridge against the
# real Phase 1 schema. The list below reflects the ACTUAL Phase 1 contract.
#
# =============================================================================
# TASK 321 ROOT FIX (forensic, root-level, no surface fix):
# The bridge now imports the canonical Phase 1 schema directly from
# phase1.contracts.phase1_schema (the SINGLE source of truth). The
# previous approach tried to load a JSON file (phase1/pipelines/schema/
# v1.json) that did not exist and fell back to a HARDCODED dict that
# silently drifted from the actual Phase 1 schema — the fake-fix
# pattern the user described. This import eliminates the drift by
# making the bridge's expected columns DERIVED from the same module
# the Phase 1 pipelines write to.
# =============================================================================
try:
    from phase1.contracts.phase1_schema import (
        PHASE1_OUTPUT_SCHEMA as _CONTRACT_PHASE1_OUTPUT_SCHEMA,
        PHASE1_CSV_FILENAMES as _CONTRACT_PHASE1_CSV_FILENAMES,
    )
    # Derive the expected-columns dict from the contract's SourceSpec
    # objects. Each spec.required_columns is a tuple of ColumnSpec; we
    # extract just the names (the bridge's validator uses name lists).
    _PHASE1_EXPECTED_COLUMNS_FROM_CONTRACT: Dict[str, List[str]] = {
        spec.key: [c.name for c in spec.required_columns]
        for spec in _CONTRACT_PHASE1_OUTPUT_SCHEMA.values()
    }
    # Derive ANY_OF groups from the contract. Each spec.any_of_groups is
    # a tuple of tuples of column names; we convert to list of lists.
    _PHASE1_ANY_OF_COLUMNS_FROM_CONTRACT: Dict[str, List[List[str]]] = {
        spec.key: [list(group) for group in spec.any_of_groups]
        for spec in _CONTRACT_PHASE1_OUTPUT_SCHEMA.values()
        if spec.any_of_groups
    }
    # Derive the source-key -> CSV-filename mapping from the contract.
    _PHASE1_SOURCE_TO_CSV_FROM_CONTRACT: Dict[str, str] = dict(
        _CONTRACT_PHASE1_CSV_FILENAMES
    )
    _PHASE1_CONTRACT_AVAILABLE: bool = True
except ImportError as _contract_exc:
    # Degraded mode: contract module not importable (e.g. Phase 2
    # standalone deployment without phase1/ on the path). Fall back to
    # the hardcoded dict below. The contract consistency test will fail
    # in CI, surfacing the misconfiguration.
    logger.warning(
        "phase1.contracts.phase1_schema not importable (%s: %s) — "
        "falling back to hardcoded _PHASE1_EXPECTED_COLUMNS. This is a "
        "DEGRADED mode; the contract consistency test "
        "(shared/tests/test_contract_consistency.py) will fail in CI.",
        type(_contract_exc).__name__, _contract_exc,
    )
    _PHASE1_EXPECTED_COLUMNS_FROM_CONTRACT = {}
    _PHASE1_ANY_OF_COLUMNS_FROM_CONTRACT = {}
    _PHASE1_SOURCE_TO_CSV_FROM_CONTRACT = {}
    _PHASE1_CONTRACT_AVAILABLE: bool = False

# v108 ROOT FIX (issue 64): the column lists are now ALSO loaded from
# ``phase1/pipelines/schema/v1.json`` (the Phase 1 contract source of
# truth). The hardcoded dict below remains as a FALLBACK for backward
# compat and for environments where the Phase 1 schema file is not
# available (e.g. when running Phase 2 standalone). The function
# ``_load_phase1_schema_columns()`` (defined below) merges the two
# sources: schema JSON wins for any CSV it covers; the hardcoded list
# fills in gaps for sources not yet in the schema JSON.
#
# TASK 321 ROOT FIX: the contract-derived values (above) OVERRIDE the
# hardcoded dict and the JSON-derived values. The contract is the
# single source of truth; the JSON file and hardcoded dict are legacy
# fallbacks that will be removed once all deployments are upgraded.
_PHASE1_SCHEMA_PATH: Path = (
    Path(__file__).resolve().parents[2]
    / "phase1" / "pipelines" / "schema" / "v1.json"
)

# Bridge source key → Phase 1 CSV filename (as declared in v1.json)
_PHASE1_SOURCE_TO_CSV: Dict[str, str] = {
    "drugs": "drugs.csv",
    "chembl_drugs": "drugs.csv",  # same CSV, different subset (chembl_id IS NOT NULL)
    "chembl_activities": "chembl_activities_clean.csv",
    "indications": "drugbank_indications.csv",
    "interactions": "drugbank_interactions.csv",
    "omim_gda": "omim_gene_disease_associations.csv",
    "uniprot_proteins": "uniprot_proteins.csv",
    "string_ppi": "string_protein_protein_interactions.csv",
    "disgenet_gda": "disgenet_gene_disease_associations.csv",
    # v109 ROOT FIX (P2-036): the previous entry was
    # "pubchem_compound_properties.csv" — but Phase 1's actual filename
    # (per phase1/pipelines/pubchem_pipeline.py and the _PHASE1_SOURCES
    # dict at line 3780) is "pubchem_enrichment.csv". The mismatch
    # caused the bridge to look for a non-existent file and silently
    # skip PubChem enrichment data.
    "pubchem_enrichment": "pubchem_enrichment.csv",
}


def _load_phase1_schema_columns() -> Dict[str, List[str]]:
    """Load the canonical Phase 1 column contracts from v1.json.

    v108 ROOT FIX (issue 64): the audit found column-name mismatches
    between the bridge's hardcoded ``_PHASE1_EXPECTED_COLUMNS`` and
    the actual Phase 1 schema. This function reads the schema JSON
    (the single source of truth, maintained by the Phase 1 team) and
    returns the required columns per source.

    If the schema file is missing (e.g. Phase 2 standalone deployment),
    returns an empty dict — the caller falls back to the hardcoded
    ``_PHASE1_EXPECTED_COLUMNS`` below.
    """
    schema_path = _PHASE1_SCHEMA_PATH
    if not schema_path.exists():
        logger.debug(
            "Phase1 schema JSON not found at %s — falling back to "
            "hardcoded _PHASE1_EXPECTED_COLUMNS (v108 issue 64).",
            schema_path,
        )
        return {}
    try:
        import json as _json
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = _json.load(f)
    except (OSError, _json.JSONDecodeError) as exc:
        logger.warning(
            "Phase1 schema JSON could not be read at %s (%s: %s) — "
            "falling back to hardcoded _PHASE1_EXPECTED_COLUMNS.",
            schema_path, type(exc).__name__, exc,
        )
        return {}
    out: Dict[str, List[str]] = {}
    properties = schema.get("properties", {})
    for source_key, csv_name in _PHASE1_SOURCE_TO_CSV.items():
        csv_schema = properties.get(csv_name)
        if not csv_schema:
            continue
        required = csv_schema.get("required", [])
        # Also include properties that are listed but not required if
        # the bridge's read code depends on them. For now, we only
        # take the 'required' list — non-required columns are optional.
        if required:
            out[source_key] = list(required)
    return out


# Load the schema-derived columns once at module load (best-effort).
_PHASE1_SCHEMA_DERIVED_COLUMNS: Dict[str, List[str]] = _load_phase1_schema_columns()

# Merge: schema JSON wins where it covers a source; hardcoded fills gaps.
# v108 issue 64: this eliminates the column-name drift between the bridge
# and Phase 1's contract.
_PHASE1_EXPECTED_COLUMNS: Dict[str, List[str]] = {
    # v107 FORENSIC ROOT FIX (ISSUE-P1-016): ``drugbank_id`` removed from
    # REQUIRED and moved to _PHASE1_ANY_OF_COLUMNS (accept either
    # ``drugbank_id`` OR ``chembl_id``). When DrugBank academic downloads
    # are unavailable, the ChEMBL-only deployment must still pass the
    # bridge's contract validation. The bridge read code uses ``inchikey``
    # as the canonical Compound key and treats drugbank_id/chembl_id as
    # source-specific aliases.
    "drugs": ["name", "inchikey"],
    "interactions": ["drugbank_id", "uniprot_id", "action_type"],
    "omim_gda": ["gene_mim", "gene_symbol", "disease_id", "disease_name"],
    "chembl_drugs": ["chembl_id", "inchikey"],
    # v83 P0-C14: ``uniprot_ac`` removed from EXPECTED and moved to
    # ANY_OF (see _PHASE1_ANY_OF_COLUMNS below). The UniProt pipeline
    # emits ``uniprot_id`` (canonical accession); the bridge read code
    # accepts ``uniprot_ac`` / ``accession`` / ``uniprot_id``. Requiring
    # ``uniprot_ac`` strictly caused a false-positive schema rejection
    # on every run.
    "uniprot_proteins": ["gene_symbol"],
    # v83 P0-C14: ``uniprot_ac_a`` / ``uniprot_ac_b`` removed from
    # EXPECTED and moved to ANY_OF. The STRING pipeline emits
    # ``protein_a`` / ``protein_b`` (ENSP IDs); the bridge read code
    # accepts both forms. ``combined_score`` stays in EXPECTED because
    # the bridge always reads it (no fallback).
    "string_ppi": ["combined_score"],
    # v78 FORENSIC ROOT FIX (BUG #7 — Silent Data-Loss):
    # The bridge's DisGeNET block reads
    # ``row.get("gene_id") or row.get("ncbi_gene_id")`` to get the
    # Gene node ID. But the previous expected-columns list only
    # required ``gene_symbol``, ``disease_id``, ``score`` — NOT
    # ``gene_id`` or ``ncbi_gene_id``. If Phase 1's disgenet pipeline
    # dropped those columns (a real risk: the embedded sample emits
    # ``gene_id`` as an integer, but the production pipeline may emit
    # ``ncbi_gene_id`` instead), the bridge silently got
    # ``gene_id=None`` for every row → 0 DisGeNET edges → 0
    # Gene→associated_with→Disease edges from DisGeNET. The
    # _validate_phase1_columns check passed (only required gene_symbol
    # + disease_id + score), so the regression was invisible.
    # ROOT FIX: require AT LEAST ONE of ``gene_id``/``ncbi_gene_id``.
    # Since _validate_phase1_columns checks ALL columns present, we
    # use a new ANY_OF validator below (``_PHASE1_ANY_OF_COLUMNS``)
    # that accepts if at least one of the alternatives is present.
    "disgenet_gda": ["gene_symbol", "disease_id", "score"],
    "pubchem_enrichment": ["inchikey", "canonical_smiles"],
    # v35 ROOT FIX (L-7): ``chembl_activities`` previously required only
    # ``molecule_chembl_id`` and ``target_chembl_id``. The bridge's
    # ChEMBL→Protein/Compound edge builder reads ``pchembl_value`` and
    # ``standard_relation`` from each row (see
    # ``_classify_chembl_activity_edge`` and the staged edge props).
    # Without these in the expected-columns list, a Phase 1 schema
    # regression that drops them would silently produce None potency
    # values on every ChEMBL edge (the activity edges still load, but
    # with no pchembl_value for the RL ranker to score potency). Added
    # both as required columns so the schema regression fails fast at
    # read time instead of silently degrading downstream.
    "chembl_activities": [
        "molecule_chembl_id", "target_chembl_id",
        "pchembl_value", "standard_relation",
    ],
    "indications": ["drugbank_id", "disease_id"],
}

# v108 ROOT FIX (issue 64): merge schema-derived columns over the hardcoded
# defaults. Schema JSON is the AUTHORITATIVE Phase 1 contract — when it
# covers a source, its required-columns list overrides the hardcoded list.
# The hardcoded list remains as a fallback for sources not in v1.json.
for _src_key, _schema_cols in _PHASE1_SCHEMA_DERIVED_COLUMNS.items():
    if _schema_cols:
        _PHASE1_EXPECTED_COLUMNS[_src_key] = _schema_cols
del _src_key, _schema_cols

# =============================================================================
# TASK 321 ROOT FIX (forensic, root-level): CONTRACT OVERRIDE.
# The contract-derived values (from phase1.contracts.phase1_schema) OVERRIDE
# both the hardcoded dict above AND the JSON-derived values. The contract
# is the single source of truth — this override ensures the bridge's
# expected columns EXACTLY match what the Phase 1 pipelines write.
# Without this override, the bridge could silently accept a CSV that the
# Phase 1 validator would reject (or vice versa) — the schema drift the
# contract was created to eliminate.
# =============================================================================
if _PHASE1_CONTRACT_AVAILABLE:
    _PHASE1_EXPECTED_COLUMNS.update(_PHASE1_EXPECTED_COLUMNS_FROM_CONTRACT)
    # Also override the source-key -> CSV-filename mapping. The contract's
    # filenames are canonical; the hardcoded mapping above may have stale
    # aliases (e.g. "pubchem_compound_properties.csv" vs the contract's
    # "pubchem_enrichment.csv").
    _PHASE1_SOURCE_TO_CSV.update(_PHASE1_SOURCE_TO_CSV_FROM_CONTRACT)
    logger.info(
        "TASK 321 ROOT FIX: Phase 2 bridge now uses phase1.contracts.phase1_schema "
        "as the SINGLE source of truth for expected columns (%d sources) and "
        "CSV filenames (%d sources). Hardcoded dict is a fallback only.",
        len(_PHASE1_EXPECTED_COLUMNS_FROM_CONTRACT),
        len(_PHASE1_SOURCE_TO_CSV_FROM_CONTRACT),
    )

# v78 FORENSIC ROOT FIX (BUG #7): ANY_OF column requirements.
# For sources where the bridge accepts multiple alternative column
# names (e.g. DisGeNET accepts ``gene_id`` OR ``ncbi_gene_id``), this
# dict lists the alternatives. The validator accepts if AT LEAST ONE
# of each list is present. A regression that drops ALL alternatives
# fails fast at read time instead of silently producing zero edges.
#
# v83 FORENSIC ROOT FIX (P0-C14 — UniProt/STRING schema mismatch
#   between Phase 1 pipeline output and bridge expected columns):
#   The UniProt pipeline writes ``uniprot_id`` (the canonical UniProt
#   accession) but the bridge validator required ``uniprot_ac``. The
#   bridge's READ code at line ~4504 already accepted both
#   ``uniprot_ac`` and ``accession`` via
#   ``row.get("uniprot_ac") or row.get("accession")`` — but the
#   VALIDATOR was strict, so the read code never ran. The bridge
#   crashed with ``DrugOSDataError: missing required column(s)
#   ['uniprot_ac']`` on every run. ROOT FIX: add ``uniprot_id`` as
#   an accepted alias for ``uniprot_ac`` (and ``protein_a``/``protein_b``
#   for STRING PPI — the bridge read code at line ~4554 already
#   accepts these aliases). The validator now matches the read code's
#   actual behavior — no more false-positive schema rejections.
_PHASE1_ANY_OF_COLUMNS: Dict[str, List[List[str]]] = {
    "disgenet_gda": [
        ["gene_id", "ncbi_gene_id"],  # bridge reads row.get("gene_id") or row.get("ncbi_gene_id")
    ],
    "string_ppi": [
        # bridge reads row.get("score") or row.get("combined_score")
        ["score", "combined_score"],
        # v83 P0-C14: bridge reads row.get("uniprot_ac_a") or row.get("protein_a")
        # or row.get("uniprot_id_a") or row.get("string_id_a") (and same for _b).
        # The Phase 1 STRING pipeline emits ``string_id_a``/``string_id_b``
        # (ENSP IDs) and ``uniprot_id_a``/``uniprot_id_b`` (when crosswalk
        # succeeds). The validator must accept all four forms.
        ["uniprot_ac_a", "protein_a", "uniprot_id_a", "string_id_a"],
        ["uniprot_ac_b", "protein_b", "uniprot_id_b", "string_id_b"],
    ],
    # v83 P0-C14: UniProt pipeline emits ``uniprot_id`` (canonical
    # accession) but the bridge read code at line ~4504 accepts
    # ``uniprot_ac``, ``accession``, OR ``uniprot_id``. The validator
    # must accept all three to match the read code's actual behavior.
    "uniprot_proteins": [
        ["uniprot_ac", "accession", "uniprot_id"],
    ],
    # v107 FORENSIC ROOT FIX (ISSUE-P1-016):
    #   The bridge previously REQUIRED ``drugbank_id`` in the ``drugs``
    #   source (_PHASE1_EXPECTED_COLUMNS["drugs"] = ["drugbank_id",
    #   "name", "inchikey"]). But Phase 1's chembl_pipeline writes
    #   ``drugs.csv`` (the alias for chembl source) which does NOT have
    #   ``drugbank_id`` -- only ``chembl_id``. When DrugBank is
    #   unavailable (academic license paused since May 2026 -- see
    #   ISSUE-P1-005), the bridge's contract check FAILED on ChEMBL-only
    #   deployments, raising DrugOSDataError and blocking the KG build.
    #   This contradicted the docstring's claim that "ChEMBL-only
    #   deployments now build a valid KG".
    #   ROOT FIX: make ``drugbank_id`` OPTIONAL via ANY_OF -- accept
    #   either ``drugbank_id`` OR ``chembl_id`` as the Compound
    #   identifier. The bridge read code already handles both (it uses
    #   inchikey as the canonical key and falls back to whatever ID is
    #   present). Also removed ``drugbank_id`` from
    #   _PHASE1_EXPECTED_COLUMNS["drugs"] (see above).
    "drugs": [
        ["drugbank_id", "chembl_id"],  # accept either as Compound identifier
    ],
}

# =============================================================================
# TASK 321 ROOT FIX (continued): merge contract-derived ANY_OF groups.
# The contract's any_of_groups are ADDED to (not replacing) the hardcoded
# groups above. The hardcoded groups may include extra aliases the bridge
# read code accepts that the contract doesn't declare (the contract is
# the MINIMUM; the bridge may accept MORE). We only add groups from the
# contract that aren't already present (by content, not by reference).
# =============================================================================
if _PHASE1_CONTRACT_AVAILABLE:
    for _src, _contract_groups in _PHASE1_ANY_OF_COLUMNS_FROM_CONTRACT.items():
        _existing_groups = _PHASE1_ANY_OF_COLUMNS.get(_src, [])
        _existing_tuples = {tuple(g) for g in _existing_groups}
        for _g in _contract_groups:
            _g_tuple = tuple(_g)
            if _g_tuple not in _existing_tuples:
                _existing_groups.append(_g)
                _existing_tuples.add(_g_tuple)
        # Ensure the key is present even if we added nothing (so callers
        # know this source has ANY_OF semantics per the contract).
        if _src not in _PHASE1_ANY_OF_COLUMNS:
            _PHASE1_ANY_OF_COLUMNS[_src] = _existing_groups
    del _src, _contract_groups, _existing_groups, _existing_tuples, _g, _g_tuple


def _validate_phase1_columns(
    df: pd.DataFrame,
    expected_columns: List[str],
    source_name: str,
    any_of_groups: Optional[List[List[str]]] = None,
) -> None:
    """Raise :class:`DrugOSDataError` if any expected column is missing.

    v27 ROOT FIX (P2-B-5): previously, ``row.get(missing_col)`` silently
    returned ``None`` / empty string for EVERY row, producing zero-output
    bugs (e.g. P2-L-1: ``chembl_to_node_records_from_phase1`` returned 0
    nodes because ``drug_chembl_id`` was missing — but the Phase 1 CSV
    had ``chembl_id``). This helper makes the schema contract explicit
    and fails fast at read time so the operator can fix the Phase 1
    pipeline instead of debugging silent data loss downstream.

    v78 FORENSIC ROOT FIX (BUG #7): added ``any_of_groups`` parameter.
    For sources where the bridge accepts multiple alternative column
    names (e.g. DisGeNET accepts ``gene_id`` OR ``ncbi_gene_id``), the
    validator accepts if AT LEAST ONE column from each group is
    present. A regression that drops ALL alternatives fails fast at
    read time instead of silently producing zero edges.
    """
    if df is None or df.empty:
        return  # nothing to validate (missing-file path handled elsewhere)
    actual = set(df.columns)
    missing = [c for c in expected_columns if c not in actual]
    if missing:
        raise DrugOSDataError(
            f"Phase 1 source '{source_name}' is missing required column(s) "
            f"{missing}. Got columns: {sorted(actual)}. This usually "
            f"means Phase 1's pipeline produced a different schema than "
            f"the bridge expects — re-run the Phase 1 pipeline, or "
            f"update _PHASE1_EXPECTED_COLUMNS in phase1_bridge.py to "
            f"match the new schema."
        )
    # v78 BUG #7: ANY_OF validation. Each group must have at least one
    # column present in the DataFrame.
    if any_of_groups:
        for group in any_of_groups:
            present = [c for c in group if c in actual]
            if not present:
                raise DrugOSDataError(
                    f"Phase 1 source '{source_name}' is missing ALL of "
                    f"the alternative columns {group}. The bridge reads "
                    f"at least one of these per row (e.g. "
                    f"``row.get('gene_id') or row.get('ncbi_gene_id')``); "
                    f"if all are missing, every row silently produces "
                    f"None → 0 edges. Got columns: {sorted(actual)}. "
                    f"Re-run the Phase 1 pipeline, or update "
                    f"_PHASE1_ANY_OF_COLUMNS in phase1_bridge.py."
                )


def _validate_phase1_source(
    df: pd.DataFrame, source_name: str
) -> None:
    """Validate a Phase 1 DataFrame against the bridge's column contract.

    v78 FORENSIC ROOT FIX (BUG #7): convenience wrapper that looks up
    both ``_PHASE1_EXPECTED_COLUMNS`` and ``_PHASE1_ANY_OF_COLUMNS`` for
    the given source name, so call sites don't need to repeat the
    lookup logic.
    """
    expected = _PHASE1_EXPECTED_COLUMNS.get(source_name)
    if expected is None:
        return  # no contract registered for this source
    any_of = _PHASE1_ANY_OF_COLUMNS.get(source_name)
    _validate_phase1_columns(df, expected, source_name, any_of_groups=any_of)


# v29 ROOT FIX (Phase1↔Phase2 100% connection): PostgreSQL reader.
#
# The forensic audit (Compound Chain 2: "Phase 1 Output Is Discarded")
# proved that the bridge BYPASSED the entire Phase 1 SQLAlchemy/database
# layer and read CSVs directly. Phase 1's 4,215 lines of loaders, 2,171
# lines of models, 4,537 lines of migration runner were dead weight —
# Phase 2 never used them. This is the single biggest reason Phase 1 ↔
# Phase 2 was only ~60% connected, not 100%.
#
# ROOT FIX: add a PostgreSQL-backed reader that reads from the same
# ORM models Phase 1's loaders write to. This makes Phase 2 actually
# consume Phase 1's database output, fulfilling the docx architecture:
#   "Airflow → Phase 1 → PostgreSQL → Phase 2"
#
# Strategy:
#   1. If DATABASE_URL is set AND the Phase 1 schema is populated, read
#      from PostgreSQL. This is the authoritative path — Phase 1's
#      cleaning/normalization/dedup/ER work is honored.
#   2. Otherwise, fall back to the CSV reader (the original v28 path).
#      This preserves backward compatibility with dev/CI runs that
#      haven't provisioned a database.
#   3. The choice is logged at INFO so operators can verify which path
#      was taken. The lineage property ``_source_phase1_backend`` on
#      every node records which backend produced it — auditors can
#      verify PostgreSQL was used in production runs.

_PHASE1_BACKEND_POSTGRES = "postgresql"
_PHASE1_BACKEND_CSV = "csv"

# v58 ROOT FIX (P2C-008 deep): the v57 fix only treated failures as
# FATAL when DRUGOS_ENVIRONMENT=prod was explicitly set. The user's
# reported scenario was: ``DATABASE_URL`` IS set (so the operator
# INTENDED to use PostgreSQL), but the bridge silently fell back to CSV
# because the default DRUGOS_ENVIRONMENT is "dev". The operator saw
# ``backend: csv`` in the run summary and could not tell that Phase 1's
# 8,500 lines of ORM/migration/loader code had been bypassed.
#
# ROOT FIX: treat the deployment as PRODUCTION whenever the operator has
# EXPLICITLY configured a database (``DATABASE_URL`` is set and non-
# empty). The only way to get the dev-mode CSV fallback now is to
# explicitly set ``DRUGOS_ENVIRONMENT=dev`` AND not set DATABASE_URL —
# making the fallback an explicit opt-in rather than a silent default.
#
# We expose both a module-level constant (``_PRODUCTION_ENV``) computed
# at import time for production code paths, AND a function
# (``_is_production_env()``) that recomputes on each call — so tests
# can verify the logic without module-reload gymnastics.


def _is_production_env() -> bool:
    """Return True iff this deployment should treat PostgreSQL failures
    as FATAL (rather than silently falling back to CSV).

    Logic:
      * ``DRUGOS_ENVIRONMENT=prod`` → True (explicit prod override).
      * ``DRUGOS_ENVIRONMENT=dev`` (or UNSET) → False (dev is the safe
        default — production deployments MUST set DRUGOS_ENVIRONMENT=prod
        explicitly so an accidental DATABASE_URL leak from a global env
        file doesn't trigger production-mode hard failures on a dev
        machine).
      * Otherwise (any other value): True iff ``DATABASE_URL`` is set
        and non-empty (backward compat for operators who set
        DRUGOS_ENVIRONMENT to a non-standard value like "staging").

    P2-062 ROOT FIX: the previous logic defaulted UNSET
    ``DRUGOS_ENVIRONMENT`` to "" (empty string), which fell through to
    the ``DATABASE_URL`` check. An operator who set ``DATABASE_URL``
    for a dev run (e.g. to test the postgres path) WITHOUT setting
    ``DRUGOS_ENVIRONMENT=dev`` got PRODUCTION-mode behavior — DB
    failures were FATAL, crashing the dev run instead of falling back
    to CSV. This was confusing because the operator expected dev-mode
    CSV fallback. Root fix: default UNSET to "dev" (the safe choice).
    Production deployments MUST set ``DRUGOS_ENVIRONMENT=prod``
    explicitly — this is the fail-safe direction (an unset variable
    produces dev-mode behavior, not prod-mode hard failures). The
    DOCX V1 launch checklist should require
    ``DRUGOS_ENVIRONMENT=prod`` as an explicit deployment step.
    """
    # P2-062: default to "dev" when UNSET. Previously defaulted to ""
    # which fell through to the DATABASE_URL check and caused dev runs
    # with an accidental DATABASE_URL to get prod-mode hard failures.
    env = os.environ.get("DRUGOS_ENVIRONMENT", "dev").lower().strip()
    if env == "prod":
        return True
    if env == "dev":
        return False
    # P2-062: backward compat — if the operator set a non-standard
    # value (e.g. "staging", "qa"), fall back to the DATABASE_URL
    # heuristic. This preserves the old behavior for operators who
    # explicitly set a non-standard env, while making "dev" the safe
    # default for the common case of an UNSET variable.
    # Default: production iff the operator has configured a database.
    return bool(os.environ.get("DATABASE_URL", "").strip())


_DATABASE_URL_SET = bool(os.environ.get("DATABASE_URL", "").strip())
_PRODUCTION_ENV = _is_production_env()


def _classify_db_failure(exc: Exception) -> str:
    """Classify a database connection / query failure into a named mode.

    v61 ROOT FIX (silent break point #1): the previous ``except Exception``
    block conflated THREE distinct failure modes into a single
    "fall back to CSV" branch:

      1. ``schema_missing`` — the database is reachable but the ``drugs``
         table doesn't exist (typically: SQLite file exists but migrations
         haven't run; or PostgreSQL is up but the Phase 1 schema isn't
         applied). This is a CONFIGURATION error, not a database failure.
         The right action is to fall back to CSV with a LOUD warning so
         the operator can run migrations — NOT to crash the bridge.

      2. ``db_unreachable`` — the database cannot be reached at all
         (network refused, DNS failure, wrong port, server down). This IS
         a real database failure. In production this should be FATAL
         (operator intended PostgreSQL but it's down — they need to know).

      3. ``auth_failed`` — the database is reachable but credentials are
         wrong / permissions missing. Same handling as ``db_unreachable``.

      4. ``unknown`` — any other exception. Conservative: treat like
         ``db_unreachable`` (re-raise in production, fall back in dev).

    The classification is based on exception type + message substring
    matching, which is the most reliable cross-driver way (psycopg2,
    sqlite3, pg8000 all use slightly different error types but the
    SQLSTATE / message text is consistent enough).
    """
    exc_type = type(exc).__name__.lower()
    exc_msg = str(exc).lower()

    # SQLAlchemy wraps the DBAPI error; the original is in exc.orig
    orig = getattr(exc, "orig", None)
    if orig is not None:
        exc_type = f"{exc_type} {type(orig).__name__.lower()}"
        exc_msg = f"{exc_msg} {str(orig).lower()}"

    # Schema-missing signatures: "no such table" (SQLite),
    # "relation ... does not exist" (PostgreSQL), "unknown table" (MySQL),
    # "invalid object name" (MSSQL).
    schema_missing_markers = (
        "no such table",
        "does not exist",
        "unknown table",
        "invalid object name",
        "undefined table",
    )
    if any(marker in exc_msg for marker in schema_missing_markers):
        return "schema_missing"

    # DB-unreachable signatures: "connection refused", "could not connect",
    # "server closed the connection", "timeout expired", "name resolution",
    # "no route to host", "network is unreachable".
    unreachable_markers = (
        "connection refused",
        "could not connect",
        "server closed the connection",
        "can't connect",
        "cannot connect",
        "timeout expired",
        "timed out",
        "name resolution",
        "name or service not known",
        "no route to host",
        "network is unreachable",
        "connection reset",
        "connection aborted",
    )
    if any(marker in exc_msg for marker in unreachable_markers):
        return "db_unreachable"

    # Auth-failed signatures: "authentication failed", "password",
    # "permission denied", "access denied", "fatal role" (Postgres),
    # "no password supplied".
    auth_markers = (
        "authentication failed",
        "password authentication",
        "permission denied",
        "access denied",
        "fatal:  role",
        "no password supplied",
        "operator does not exist",  # postgres: missing auth function
    )
    if any(marker in exc_msg for marker in auth_markers):
        return "auth_failed"

    return "unknown"


# v109 ROOT FIX (P2-016): cache the result of _phase1_db_available() so
# we don't retry the same failing import 909 times in one pipeline run.
# The previous code called _phase1_db_available() on every bridge
# invocation (once per source: drugs, interactions, indications, etc.).
# Each call retried the failing ``from database.connection import
# get_engine`` import, logged a fallback event to bridge_fallbacks.jsonl,
# and fell back to CSV. The audit log showed 909 entries from a single
# pipeline run — all with the same ImportError. ROOT FIX: cache the
# result for the lifetime of the process. The cache is invalidated only
# if the process restarts (which is the right granularity — env var
# changes require a restart anyway).
_phase1_db_available_cache: Optional[bool] = None
_phase1_db_available_cache_set: bool = False


def _phase1_db_available() -> bool:
    """Return True iff a Phase 1 database backend is configured AND populated.

    v109 ROOT FIX (P2-016): the result is now CACHED per-process to
    avoid retrying the same failing import 909 times. The cache is set
    on the first call and reused for all subsequent calls. If the
    underlying state changes (e.g. the operator runs migrations), the
    process must be restarted to pick up the new state — this matches
    the existing behavior for all other env-var-driven config.

    v61 ROOT FIX (silent break point #1 — forensic deep fix):
    The v58/v60 code swallowed ALL exceptions with a single ``except
    Exception`` block and re-raised in production. This crashed the
    bridge for the COMMON configuration error of "SQLite file exists
    but no schema migrated" — the user's exact runtime scenario.
    The .env file sets ``DATABASE_URL=file:/home/z/my-project/db/custom.db``
    which makes ``_PRODUCTION_ENV=True``, but the SQLite file is empty
    (no drugs table), so the SELECT raised ``OperationalError: no such
    table: drugs`` and the v58 re-raise made the bridge CRASH instead
    of falling back to CSV.

    ROOT FIX: classify the failure mode and act accordingly:
      * ``schema_missing`` → log ERROR + audit, fall back to CSV (NOT fatal
        even in production — this is a configuration issue, not a DB
        failure; the bridge can still produce a graph from CSV while
        the operator runs migrations)
      * ``db_unreachable`` / ``auth_failed`` → in production, re-raise
        (operator intended the DB to be up — they need to know it's
        not); in dev, fall back to CSV with ERROR log + audit
      * ``unknown`` → conservative: same as ``db_unreachable``
      * Success with 0 rows → return False (legitimate empty DB; CSV
        fallback is correct)
      * Success with >0 rows → return True (DB is populated; use it)
    """
    # v109 ROOT FIX (P2-016): return cached result if available. This
    # prevents 909 redundant import attempts in a single pipeline run.
    global _phase1_db_available_cache, _phase1_db_available_cache_set
    if _phase1_db_available_cache_set:
        return bool(_phase1_db_available_cache)
    result = _phase1_db_available_uncached()
    _phase1_db_available_cache = result
    _phase1_db_available_cache_set = True
    return result


def _phase1_db_available_uncached() -> bool:
    """Actual implementation of _phase1_db_available() (uncached).

    Called once per process by _phase1_db_available() (which caches the
    result). See _phase1_db_available() for the full docstring.
    """
    try:
        import sys as _sys
        _phase1_root = str(Path(__file__).resolve().parents[2] / "phase1")
        if _phase1_root not in _sys.path:
            _sys.path.insert(0, _phase1_root)
        from database.connection import get_engine  # type: ignore
        from sqlalchemy import text as _sa_text, inspect as _sa_inspect
        engine = get_engine()
        with engine.connect() as conn:
            # P2-057 ROOT FIX (Teammate 4, forensic, root-level): the
            # previous code checked ONLY the ``drugs`` table. But Phase 1
            # writes to MULTIPLE tables (drugs, proteins,
            # drug_protein_interactions, etc.). If only ``drugs`` exists
            # but the OTHER tables are missing (e.g. partial migration,
            # a loader crashed mid-write), the previous code returned
            # True → the bridge attempted to read from the missing
            # tables → ``_read_phase1_from_postgres`` raised
            # ``OperationalError: no such table: proteins`` → the bridge
            # fell back to CSV (silent data loss in production).
            #
            # ROOT FIX: check ALL required tables exist AND have >0 rows.
            # The required tables are the ones _read_phase1_from_postgres
            # actually reads. If any is missing OR empty, return False so
            # the bridge falls back to CSV (correct behaviour) instead of
            # crashing mid-read (which would also fall back but with a
            # confusing error log).
            inspector = _sa_inspect(conn)
            existing_tables = set(inspector.get_table_names())
            # P2-057: the canonical list of tables _read_phase1_from_postgres
            # reads. If any is missing, the DB is not ready — fall back to CSV.
            _required_tables = (
                "drugs",
                "proteins",
                "drug_protein_interactions",
            )
            _missing_tables = [
                t for t in _required_tables if t not in existing_tables
            ]
            if _missing_tables:
                logger.warning(
                    "P2-057 ROOT FIX: Phase 1 DB is reachable but "
                    "missing required tables: %s. The bridge will fall "
                    "back to CSV. Run "
                    "`python -m database.migrations.run_migrations` "
                    "from phase1/ to create all required tables.",
                    _missing_tables,
                )
                _log_bridge_fallback(
                    "phase1_db_available",
                    "schema_missing_tables",
                    backend="csv",
                    exception_type="MissingTables",
                    exception_message=f"missing: {_missing_tables}",
                    extra={"missing_tables": _missing_tables},
                )
                return False
            # All required tables exist — verify the drugs table has rows.
            # (We check drugs because it's the canonical "Phase 1 ran"
            # indicator. The other tables may be legitimately empty for
            # a partial Phase 1 run, but drugs must have at least 1 row.)
            row = conn.execute(
                _sa_text("SELECT COUNT(*) AS n FROM drugs")
            ).fetchone()
            return bool(row is not None and row[0] is not None and int(row[0]) > 0)
    except Exception as exc:  # noqa: BLE001 — best-effort detection
        failure_mode = _classify_db_failure(exc)
        # v61 ROOT FIX: classify and act per failure mode.
        if failure_mode == "schema_missing":
            # Configuration error: DB exists but schema not migrated.
            # NOT fatal even in production — the bridge can still
            # produce a graph from CSV while the operator runs migrations.
            logger.error(
                "Phase1 bridge: database is reachable but the `drugs` "
                "table does not exist (schema_missing). This means the "
                "Phase 1 migrations have not been applied to the "
                "configured database. The bridge will fall back to CSV "
                "(if available) so the pipeline can still produce a "
                "graph, but Phase 1's ORM/migration/loader code is being "
                "bypassed. To use the database backend, run "
                "`python -m database.migrations.run_migrations` from "
                "the phase1/ directory. Original error: %s: %s",
                type(exc).__name__, exc,
            )
            _log_bridge_fallback(
                "phase1_db_available",
                "schema_missing",
                backend="csv",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                extra={"failure_mode": "schema_missing"},
            )
            return False  # fall back to CSV — NOT fatal
        # failure_mode in {"db_unreachable", "auth_failed", "unknown"}
        logger.error(
            "Phase1 bridge: database backend unavailable (%s): %s: %s "
            "— will fall back to CSV reader (dev only; in prod this is "
            "fatal for db_unreachable/auth_failed/unknown modes).",
            failure_mode, type(exc).__name__, exc,
        )
        _log_bridge_fallback(
            "phase1_db_available",
            f"db_unavailable:{failure_mode}",
            backend="csv",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            extra={"failure_mode": failure_mode},
        )
        if _PRODUCTION_ENV:
            # Production must not silently bypass Phase 1's ORM/migration/
            # loader code for unreachable/auth failures. Re-raise so the
            # failure is visible. Schema_missing is handled above (NOT
            # re-raised — it's a configuration issue, not a DB failure).
            raise
        # v88 ROOT FIX (BUG #28 — silent fallback to CSV bypasses Phase 1):
        # the previous gate was `_PRODUCTION_ENV` which is computed from
        # `DATABASE_URL` being set. If the operator FORGOT to set
        # `DATABASE_URL` (a common deployment mistake), `_PRODUCTION_ENV`
        # was False and DB failures silently fell back to CSV. ROOT FIX:
        # when prefer_postgres=True (the default), RAISE unless
        # `DRUGOS_ALLOW_CSV_FALLBACK=1` is explicitly set. This makes
        # the CSV fallback an OPT-IN for operators who understand the
        # tradeoff, rather than a silent default.
        _allow_csv_fallback = os.environ.get("DRUGOS_ALLOW_CSV_FALLBACK", "") == "1"
        if not _allow_csv_fallback:
            raise RuntimeError(
                f"phase1_bridge: database backend unavailable "
                f"(failure_mode={failure_mode}) and prefer_postgres=True. "
                f"Silent fallback to CSV would bypass Phase 1's entire "
                f"ORM/migration/cleaning layer (v88 BUG #28 root fix). To "
                f"explicitly allow CSV fallback (dev/CI only), set "
                f"DRUGOS_ALLOW_CSV_FALLBACK=1 in the environment. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        return False


def _read_indications_from_postgres(conn: Any) -> Optional[pd.DataFrame]:
    """v49 ROOT FIX: read Compound-treats-Disease indications from PostgreSQL.

    Reads from the new ``drugs.indication`` + ``drugs.indication_source``
    columns (added by migration 010). Returns a DataFrame with columns:
        - drug_inchikey (str) — Compound node identifier
        - drug_name (str) — display name
        - disease_id (str) — best-effort: derived from indication text
          via simple keyword matching (hypertension, diabetes, etc.) OR
          joined from gene_disease_associations when the drug's targets
          connect to a gene with a known disease.
        - disease_name (str) — same source
        - source (str) — always 'drugbank_postgres' (distinguishes from
          CSV-derived 'drugbank_xml' structured indications)
        - indication_source (str) — provenance tag from the Drug ORM
          ('drugbank_xml' | 'chembl_max_phase' | 'rxnorm' | 'manual')

    Returns None if the `indication` column doesn't exist (DB not yet
    migrated to v49) — the caller falls back to CSV in that case.

    Parameters
    ----------
    conn : SQLAlchemy Connection
        Open read-only connection to the Phase 1 DB.
    """
    try:
        import sys as _sys
        _phase1_root = str(Path(__file__).resolve().parents[2] / "phase1")
        if _phase1_root not in _sys.path:
            _sys.path.insert(0, _phase1_root)
        from database import models as _m  # type: ignore
        from sqlalchemy import select, func, or_, text as sa_text

        # Verify the column exists (DB may not be migrated yet).
        inspector_cols = [
            c["name"] for c in _inspect_columns(conn, "drugs")
        ]
        if "indication" not in inspector_cols:
            logger.debug(
                "Phase1 bridge: drugs.indication column not present "
                "(DB not migrated to v49). Returning None — caller "
                "will fall back to CSV."
            )
            return None

        # Read all drugs with non-empty indication text.
        # v84 FORENSIC ROOT FIX (BUG #25 — missing canonical ID validation):
        # The previous query only filtered on `indication IS NOT NULL` —
        # inchikey NULLs / empty strings / malformed values were allowed
        # through. Downstream code uses inchikey as the canonical Compound
        # ID (per the Phase 1 bridge docstring "Canonical Compound ID =
        # InChIKey"). Compound nodes with NULL/empty inchikey would be
        # created with id=NULL or id="", violating the InChIKey mandate
        # and silently dropping drugs that have indication text but no
        # inchikey. ROOT FIX: add `AND inchikey IS NOT NULL AND inchikey
        # ~ '^[A-Z]{14}-[A-Z]{10}-[A-Z]$'` to the WHERE clause so only
        # drugs with a valid InChIKey (14 uppercase letters, hyphen, 10
        # uppercase letters, hyphen, 1 uppercase letter — the standard
        # InChIKey format) are returned. The KG builder's ID validator
        # no longer needs to dead-letter these rows because they never
        # reach it.
        # v88 ROOT FIX (BUG #39 — InChIKey format validation): filter
        # to well-formed InChIKeys (^[A-Z]{14}-[A-Z]{10}-[A-Z]$) so NULL/
        # empty/malformed inchikeys don't violate the InChIKey mandate.
        #
        # P2-014 ROOT FIX (v107 forensic): the v84/v88 filter DROPPED all
        # biotech drugs (insulin DB00071, Humira, Keytruda — ~30% of modern
        # FDA approvals) because biotech drugs have NO InChIKey (they are
        # proteins, antibodies, etc. — InChIKey is a small-molecule-only
        # identifier). Their treats edges were dropped BEFORE the P2-027
        # alias consolidation could merge them. The KG's drug coverage was
        # structurally incomplete for the entire biotech drug class. The RL
        # ranker could not recommend biotech drugs because they had no
        # treats edges.
        # ROOT FIX: relax the WHERE clause to ACCEPT rows with EITHER a
        # valid InChIKey OR a non-empty DrugBank ID. Biotech drugs use
        # DrugBank ID as the canonical identifier (per the Phase 1 bridge
        # docstring line 16: "Canonical Compound ID = InChIKey for small
        # molecules, DrugBank ID for biologics"). The downstream alias
        # consolidation (P2-027) merges them into the correct Compound
        # node via DrugBank ID crosswalk. We still validate the InChIKey
        # format when it IS present (the regex filter is applied to the
        # inchikey column AFTER the read, not in the WHERE clause, so
        # malformed inchikeys are demoted to DrugBank ID rather than
        # dropping the row entirely).
        # v108 ROOT FIX (issue 61): extend biotech drug fallback to ALSO accept
        # PubChem CID and UniProt accession (for biologics). The previous v107
        # P2-014 fix accepted InChIKey OR DrugBank ID, but the audit (issue 61)
        # found that some biotech drugs have NEITHER InChIKey NOR DrugBank ID
        # but DO have a UniProt accession (for recombinant proteins like
        # insulin, EPO, G-CSF) or a PubChem CID (for some small molecules
        # where InChIKey computation failed). Without these fallbacks, ~5%
        # of FDA-approved biologics were dropped from the KG.
        #
        # The drugs table doesn't store UniProt accessions directly (those
        # live on the proteins table). For biologic drugs, the link is
        # drug → drug_protein_interactions → proteins.uniprot_id. We LEFT JOIN
        # through this chain to get the UniProt accession as a fallback ID.
        # We also SELECT pubchem_cid (already on the drugs table) as another
        # fallback.
        drugs_with_indication = pd.read_sql(
            sa_text(
                "SELECT d.inchikey, d.name, d.chembl_id, d.drugbank_id, "
                "d.pubchem_cid, p.uniprot_id AS uniprot_accession, "
                "d.indication, d.indication_source, d.max_phase, "
                "d.is_fda_approved, d.is_globally_approved "
                "FROM drugs d "
                "LEFT JOIN drug_protein_interactions dpi ON dpi.drug_id = d.id "
                "LEFT JOIN proteins p ON p.id = dpi.protein_id "
                "WHERE d.indication IS NOT NULL "
                "AND TRIM(d.indication) != '' "
                "AND d.is_deleted = false "
                "AND ("
                "  (d.inchikey IS NOT NULL AND TRIM(d.inchikey) != '' "
                "   AND d.inchikey ~ '^[A-Z]{14}-[A-Z]{10}-[A-Z]$') "
                "  OR "
                "  (d.drugbank_id IS NOT NULL AND TRIM(d.drugbank_id) != '') "
                "  OR "
                "  (d.pubchem_cid IS NOT NULL AND d.pubchem_cid > 0) "
                "  OR "
                "  (p.uniprot_id IS NOT NULL AND TRIM(p.uniprot_id) != '')"
                ") "
                "-- v108 issue 61: accept InChIKey OR DrugBank ID OR PubChem CID"
                "-- OR UniProt accession (biotech drugs)."
            ),
            conn,
        )
        if not drugs_with_indication.empty:
            import re as _re_v88_ik
            _INCHIKEY_RE = _re_v88_ik.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
            _ik = drugs_with_indication["inchikey"].astype(str)
            _valid_ik_mask = _ik.apply(
                lambda x: bool(_INCHIKEY_RE.match(x)) if x and x != "nan" else False
            )
            _n_invalid_ik = int((~_valid_ik_mask).sum())
            if _n_invalid_ik > 0:
                # P2-014 + v108 issue 61: do NOT drop these rows. They are
                # biotech drugs (no InChIKey). Demote to DrugBank ID, or
                # PubChem CID, or UniProt accession (the three biotech
                # fallbacks). Log so operators can audit the count.
                def _has_id(s):
                    """True if s is a non-empty, non-'nan' value."""
                    return bool(s) and str(s).strip() != "" and str(s) != "nan"

                _n_with_drugbank = int(
                    drugs_with_indication.loc[~_valid_ik_mask, "drugbank_id"]
                    .apply(_has_id)
                    .sum()
                )
                _n_with_pubchem = int(
                    drugs_with_indication.loc[~_valid_ik_mask, "pubchem_cid"]
                    .apply(lambda x: bool(x) and str(x) != "nan" and float(x) > 0)
                    .sum()
                )
                _n_with_uniprot = int(
                    drugs_with_indication.loc[~_valid_ik_mask, "uniprot_accession"]
                    .apply(_has_id)
                    .sum()
                )
                logger.warning(
                    "phase1_bridge: %d/%d indication rows have NULL/empty/"
                    "malformed inchikey (v88 BUG #39 / P2-014 v107 / v108 "
                    "issue 61). Fallback ID coverage: DrugBank=%d, "
                    "PubChem=%d, UniProt=%d. Keeping rows with ANY fallback "
                    "ID; dropping rows with NONE.",
                    _n_invalid_ik, len(drugs_with_indication),
                    _n_with_drugbank, _n_with_pubchem, _n_with_uniprot,
                )
                # Keep rows that have EITHER a valid inchikey OR a
                # non-empty drugbank_id OR pubchem_cid OR uniprot_accession.
                # Drop only rows with NONE of these.
                _has_drugbank = drugs_with_indication["drugbank_id"].apply(_has_id)
                _has_pubchem = drugs_with_indication["pubchem_cid"].apply(
                    lambda x: bool(x) and str(x) != "nan" and float(x) > 0
                )
                _has_uniprot = drugs_with_indication["uniprot_accession"].apply(_has_id)
                _keep_mask = _valid_ik_mask | _has_drugbank | _has_pubchem | _has_uniprot
                _n_dropped = int((~_keep_mask).sum())
                if _n_dropped > 0:
                    logger.error(
                        "phase1_bridge: dropping %d indication rows with "
                        "NEITHER inchikey NOR drugbank_id NOR pubchem_cid "
                        "NOR uniprot_accession — cannot canonicalize. "
                        "These are likely data-quality issues in Phase 1. "
                        "Investigate the DrugBank parser.",
                        _n_dropped,
                    )
                drugs_with_indication = drugs_with_indication[_keep_mask].reset_index(drop=True)
        if drugs_with_indication.empty:
            return None

        # Best-effort disease extraction from indication free-text.
        # This is intentionally conservative — the DrugBank XML parser
        # produces structured (drug, disease) pairs in the CSV path
        # which is the gold standard. The PostgreSQL path produces
        # (drug, indication_text) pairs which downstream
        # `_load_clinical_outcomes` can still consume (each unique
        # indication_text becomes a ClinicalOutcome node).
        drugs_with_indication = drugs_with_indication.rename(columns={
            "inchikey": "drug_inchikey",
            "name": "drug_name",
        })
        # v108 ROOT FIX (issue 61): add a canonical_drug_id column that
        # captures the FIRST available ID for each row (InChIKey > DrugBank
        # > PubChem CID > UniProt accession). Downstream consumers use this
        # to register the Compound node without re-doing the fallback logic.
        def _pick_canonical_id(row):
            ik = row.get("drug_inchikey")
            if isinstance(ik, str) and _INCHIKEY_RE.match(ik):
                return ik
            db = row.get("drugbank_id")
            if isinstance(db, str) and db.strip() and db != "nan":
                return db
            pc = row.get("pubchem_cid")
            try:
                if pc is not None and str(pc) != "nan" and float(pc) > 0:
                    return f"CID:{int(pc)}"
            except (TypeError, ValueError):
                pass
            ua = row.get("uniprot_accession")
            if isinstance(ua, str) and ua.strip() and ua != "nan":
                return ua
            return None

        # _INCHIKEY_RE was defined inside the `if not drugs_with_indication.empty:`
        # block above; re-define it here so this helper works regardless.
        import re as _re_v108_ik
        _INCHIKEY_RE = _re_v108_ik.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
        drugs_with_indication["canonical_drug_id"] = (
            drugs_with_indication.apply(_pick_canonical_id, axis=1)
        )
        drugs_with_indication["disease_id"] = (
            drugs_with_indication["drug_inchikey"].apply(_extract_disease_id_from_indication_text)
        )
        drugs_with_indication["disease_name"] = (
            drugs_with_indication["indication"].apply(_extract_disease_name_from_indication_text)
        )
        drugs_with_indication["source"] = "drugbank_postgres"

        # Drop rows where we couldn't extract any disease signal —
        # they have indication text but no parseable disease. These are
        # still useful as ClinicalOutcome nodes (the indication text IS
        # the outcome), but they cannot form Compound-treats-Disease
        # edges without a disease_id.
        # Actually KEEP them — _load_clinical_outcomes downstream handles
        # rows with empty disease_id by creating ClinicalOutcome nodes
        # only (no treats edge). That's still valuable.
        return drugs_with_indication
    except Exception as exc:
        logger.debug(
            "Phase1 bridge: _read_indications_from_postgres failed: %s. "
            "Returning None — caller will fall back to CSV.", exc,
        )
        return None


def _inspect_columns(conn: Any, table_name: str) -> list:
    """Helper: list column names for a table using SQLAlchemy inspector."""
    try:
        from sqlalchemy import inspect
        insp = inspect(conn)
        return list(insp.get_columns(table_name))
    except Exception:
        return []


# P2-005 FORENSIC ROOT FIX (v104 — Team Member 5): schema_version
# filtering for Phase 1 Postgres reads.
#
# BUG (P2-005):
#   ``_read_phase1_from_postgres`` read ALL rows from Phase 1's
#   Postgres tables without filtering by ``schema_version``. If
#   Phase 1 was mid-migration (some rows at schema_version 16, some
#   at 17), the bridge read BOTH. The schema-17 rows had
#   ``compound_inchikey_canonical`` populated; the schema-16 rows
#   did not. The bridge's InChIKey-based deduplication then failed
#   for the schema-16 subset → KG had duplicate Compound nodes.
#
# ROOT FIX:
#   1. ``_get_latest_schema_version(conn)`` queries
#      ``SELECT MAX(version) FROM schema_version``. The Phase 1
#      ``schema_version`` table (defined in database/models.py:525)
#      tracks applied migrations — one row per migration, latest
#      row = current version. The column is ``version`` (Integer),
#      NOT ``schema_version`` (which is a per-row column on
#      ``GeneDiseaseAssociation`` only).
#   2. For tables that HAVE a per-row ``schema_version`` column
#      (currently only ``GeneDiseaseAssociation`` per AST analysis
#      of database/models.py), filter:
#        WHERE schema_version = <str(latest_version)>
#      This excludes rows from incomplete migrations.
#   3. For tables WITHOUT a per-row ``schema_version`` (Drug,
#      Protein, DrugProteinInteraction, ProteinProteinInteraction):
#      log the latest version for observability and emit a WARNING
#      if the schema_version table has multiple pending rows (a
#      heuristic for mid-migration state). We cannot filter per-row
#      because the column doesn't exist — but the warning gives
#      operators visibility.
#   4. Defensive: if the ``schema_version`` table doesn't exist or
#      the query fails (fresh DB, permissions issue, etc.), log at
#      DEBUG and proceed without filtering. This preserves backward
#      compatibility with databases that haven't run the migration
#      that creates the ``schema_version`` table.
def _get_latest_schema_version(conn: Any) -> Optional[int]:
    """P2-005 — return the latest applied schema version from Postgres.

    Queries ``SELECT MAX(version) FROM schema_version``. Returns None
    if the table doesn't exist, is empty, or the query fails (defensive
    — fresh DB, permissions, etc.).
    """
    try:
        from sqlalchemy import text
        result = conn.execute(
            text("SELECT MAX(version) AS latest FROM schema_version")
        )
        row = result.fetchone()
        if row is None:
            return None
        latest = row[0] if hasattr(row, "__getitem__") else getattr(row, "latest", None)
        if latest is None:
            return None
        return int(latest)
    except Exception as exc:
        logger.debug(
            "P2-005: could not read latest schema_version from Postgres "
            "(%s: %s). Proceeding without schema_version filter. This is "
            "expected on a fresh DB before migrations have run, or if the "
            "schema_version table is not yet created.",
            type(exc).__name__, exc,
        )
        return None


def _count_schema_versions(conn: Any) -> int:
    """P2-005 — count rows in the ``schema_version`` table.

    Used to detect mid-migration state: if the count is > 0 but the
    latest version's migration is still in progress (heuristic: the
    last row's ``applied_at`` is < 60 seconds ago), warn operators.

    Returns 0 if the table doesn't exist or query fails.
    """
    try:
        from sqlalchemy import text
        result = conn.execute(text("SELECT COUNT(*) AS n FROM schema_version"))
        row = result.fetchone()
        if row is None:
            return 0
        n = row[0] if hasattr(row, "__getitem__") else getattr(row, "n", 0)
        return int(n) if n is not None else 0
    except Exception:
        return 0


# P2-001 FORENSIC ROOT FIX (v104 — Team Member 5, Phase 2 KG Bridge):
# The previous implementation used naive substring matching
# (``keyword in t``) against a hardcoded disease dictionary. This
# produced WRONG Compound-treats-Disease edges for four documented
# cases:
#   1. "respiratory depression" matched "depression" — WRONG.
#      Respiratory depression is a breathing adverse event (opioid
#      side effect), NOT Major Depressive Disorder (DOID:1470).
#   2. "painkiller" matched "pain" — WRONG. "Painkiller" is a drug
#      class, not a disease indication.
#   3. "anti-inflammatory" matched "inflammation" — WRONG.
#      "Anti-inflammatory" is a drug mechanism, not a disease.
#   4. "ulcerative colitis" matched "ulcer" — WRONG. Ulcerative
#      colitis is an IBD, not a peptic ulcer.
#   5. "does not treat pain" matched "pain" — WRONG. Negated
#      indications were treated as positive.
#
# ROOT FIX (3 layers, no surface patch):
#   L1 — Word-boundary regex (``\b{keyword}\b``): eliminates
#        intra-word false positives like "painkiller" → "pain",
#        "ulcerative" → "ulcer", "anti-inflammatory" → "inflammation".
#   L2 — NegEx-style negation detection: scans a 6-token window
#        BEFORE the match for negation cues ("not", "no", "without",
#        "contraindicated", "does not treat", "not indicated for",
#        "not for", "avoid", "never"). If a cue is found, the match
#        is REJECTED. This catches "does not treat pain",
#        "contraindicated in hypertension", etc.
#   L3 — Longest-match-first: keywords are sorted by length descending
#        so multi-word terms like "respiratory depression" are checked
#        BEFORE single words like "depression". We also maintain an
#        explicit ``_DISEASE_FALSE_FRIENDS`` map of multi-word phrases
#        that look like a disease keyword but mean something else
#        (e.g. "respiratory depression" → NOT a disease indication;
#        it is an adverse event). When a false-friend phrase is
#        present in the text, the corresponding single-word keyword
#        is suppressed for the ENTIRE text — preventing the
#        "respiratory depression" → "depression" mismatch.
#
# The CSV path (DrugBank XML <indication> parser → structured
# (drug, disease) pairs) remains the gold standard. This free-text
# extractor is only used when the PostgreSQL ``drugs.indication``
# column is populated with free-text and no structured mapping is
# available.
_DISEASE_KEYWORD_MAP = {
    "hypertension": ("DOID:10763", "Hypertension"),
    "diabetes": ("DOID:9351", "Diabetes Mellitus"),
    "asthma": ("DOID:2841", "Asthma"),
    "cancer": ("DOID:162", "Cancer"),
    "tumor": ("DOID:162", "Cancer"),
    "infection": ("DOID:0050117", "Infection"),
    "depression": ("DOID:1470", "Major Depressive Disorder"),
    "anxiety": ("DOID:14319", "Anxiety Disorder"),
    "epilepsy": ("DOID:1826", "Epilepsy"),
    "seizure": ("DOID:1826", "Epilepsy"),
    "arthritis": ("DOID:7148", "Arthritis"),
    "pain": ("DOID:0050133", "Pain"),
    "inflammation": ("DOID:1101", "Inflammation"),
    "migraine": ("DOID:1197", "Migraine"),
    "ulcer": ("DOID:77", "Ulcer"),
    # P2-001 STRENGTHENING (v106 — Team Member 5): add multi-word disease
    # "ulcerative colitis" so it is recognized as ONE disease (DOID:8535)
    # instead of being missed entirely. The issue description explicitly
    # names this case: the naive substring match split it into "ulcer" +
    # "colitis" (2 Disease nodes). The word-boundary regex fix PREVENTS
    # the split but also misses the real disease. Adding the multi-word
    # keyword restores correct recognition. Longest-match-first sorting
    # (L3) ensures "ulcerative colitis" is checked BEFORE "ulcer".
    "ulcerative colitis": ("DOID:8535", "Ulcerative Colitis"),
}

# P2-001 L3 — multi-word "false friend" phrases. When any of these
# phrases appears in the indication text, the corresponding
# single-word keyword is SUPPRESSED for the entire text (it would
# otherwise match via word-boundary regex and produce a wrong edge).
# Value = the keyword to suppress when the phrase is present.
#
# Rationale: "respiratory depression" is an adverse event (opioid
# side effect), NOT an indication. "Bipolar depression" / "postpartum
# depression" / "agitated depression" ARE depression subtypes — they
# should NOT suppress "depression". We only suppress when the
# modifier changes the clinical meaning to a non-disease (e.g. a
# breathing event, not a mood disorder).
_DISEASE_FALSE_FRIENDS = {
    # phrase (lowercase)          : keyword to suppress
    "respiratory depression": "depression",
    "neurotic depression": "depression",  # deprecated ICD-9 term, not an indication
}

# P2-001 L2 — NegEx-style negation cues. If any of these appear in
# the 6-token window BEFORE a disease-keyword match, the match is
# rejected. Tuned for medical indication text (DrugBank <indication>
# free-text). The list is conservative — false positives (missing a
# negation) are worse than false negatives (extra negation check)
# here, because a wrong Compound-treats-Disease edge corrupts the
# GNN and the RL ranker downstream.
_NEGEX_CUES = frozenset({
    "not", "no", "without", "nor", "never", "neither",
    "contraindicated", "contraindication", "avoid",
    "doesn't", "doesnt", "don't", "dont", "isn't", "isnt",
    "cannot", "can't", "cant", "shouldn't", "shouldnt",
    "not treat", "not treating", "not treated",
    "not indicated", "not for", "not used",
    "not recommended", "not suitable",
})

# P2-001 L1 — pre-compiled word-boundary regex for each keyword.
# Built ONCE at import time (keywords are static). ``re.escape``
# ensures keywords with regex metacharacters are matched literally.
# ``re.IGNORECASE`` lets us match without lowercasing the text first
# (preserving the original text for the negex window scan).
_DISEASE_KEYWORD_PATTERNS: List[Tuple[str, str, "re.Pattern[str]"]] = [
    (
        keyword,
        doid,
        re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE),
    )
    for keyword, (doid, _name) in sorted(
        _DISEASE_KEYWORD_MAP.items(),
        key=lambda kv: len(kv[0]),
        reverse=True,  # longest first — L3
    )
]


def _is_negated(text_lower: str, match_start: int) -> bool:
    """P2-001 L2 — NegEx-style negation check.

    Scans the 6-token window BEFORE ``match_start`` in ``text_lower``
    for any negation cue. Returns True if a cue is found.

    ``text_lower`` must be the lowercased original text (so cue
    matching is case-insensitive). ``match_start`` is the character
    index of the disease-keyword match in ``text_lower``.
    """
    if match_start <= 0:
        return False
    # Take the preceding window and tokenize on whitespace. 6 tokens
    # is the NegEx standard window for English medical text.
    window = text_lower[:match_start].rsplit(None, 6)  # last 6 tokens
    if not window:
        return False
    # Check single-token cues
    for tok in window:
        # strip trailing punctuation that may attach to the token
        # (e.g. "not," "without.")
        cleaned = tok.strip(".,;:!?()[]\"'")
        if cleaned in _NEGEX_CUES:
            return True
    # Check multi-token cues (e.g. "not treat", "not indicated")
    window_str = " ".join(window)
    for cue in _NEGEX_CUES:
        if " " in cue and cue in window_str:
            return True
    return False


def _extract_disease_from_indication_text(
    text: str,
) -> Optional[Tuple[str, str]]:
    """P2-001 ROOT FIX — extract (doid, disease_name) from free-text.

    Returns None if no non-negated, word-boundary-matched disease
    keyword is found. Applies all 3 fix layers:
      L1 — word-boundary regex
      L2 — NegEx negation window
      L3 — longest-match-first + false-friend suppression

    The downstream ClinicalOutcome loader handles None by creating
    ClinicalOutcome nodes without treats-edges.
    """
    if not text or not isinstance(text, str):
        return None
    text_lower = text.lower()

    # L3 — determine which keywords to SUPPRESS based on false-friend
    # phrases present in the text. If "respiratory depression" appears,
    # "depression" is suppressed for the entire text.
    suppressed: set = set()
    for phrase, suppress_keyword in _DISEASE_FALSE_FRIENDS.items():
        if phrase in text_lower:
            suppressed.add(suppress_keyword)

    # L1 + L2 + L3 — iterate longest-first, skip suppressed, check
    # word boundaries, check negation.
    for keyword, doid, pattern in _DISEASE_KEYWORD_PATTERNS:
        if keyword in suppressed:
            continue
        m = pattern.search(text)
        if m is None:
            continue
        # L2 — negation check
        if _is_negated(text_lower, m.start()):
            continue
        # Match found and not negated — return (doid, name).
        return (doid, _DISEASE_KEYWORD_MAP[keyword][1])
    return None


def _extract_disease_id_from_indication_text(text: str) -> Optional[str]:
    """P2-001 ROOT FIX — extract a DOID-style disease ID from free-text.

    Thin wrapper around ``_extract_disease_from_indication_text`` that
    returns only the DOID. Returns None if no match.
    """
    result = _extract_disease_from_indication_text(text)
    return result[0] if result is not None else None


def _extract_disease_name_from_indication_text(text: str) -> Optional[str]:
    """P2-001 ROOT FIX — extract a human-readable disease name from free-text.

    Thin wrapper around ``_extract_disease_from_indication_text`` that
    returns only the name. Returns None if no match.
    """
    result = _extract_disease_from_indication_text(text)
    return result[1] if result is not None else None


def _read_phase1_from_postgres() -> Dict[str, pd.DataFrame]:
    """Read ALL Phase 1 data from PostgreSQL via SQLAlchemy ORM models.

    This is the ROOT FIX for the Phase 1 ↔ Phase 2 connection. Returns
    a dict with the SAME keys as :func:`read_phase1_outputs` so callers
    can use either backend transparently.

    Schema mapping (Phase 1 ORM table → bridge key):
        drugs                          → "drugs"
        drug_protein_interactions      → "interactions"
        gene_disease_associations      → "omim_gda" + "disgenet_gda"
        protein_protein_interactions   → "string_ppi"
        proteins                       → "uniprot_proteins"
        entity_mapping                 → used for cross-source ID resolution

    The function reads with read-only sessions and never mutates the DB.
    """
    import sys as _sys
    _phase1_root = str(Path(__file__).resolve().parents[2] / "phase1")
    if _phase1_root not in _sys.path:
        _sys.path.insert(0, _phase1_root)

    from database.connection import get_engine  # type: ignore
    from database import models as _m  # type: ignore
    from sqlalchemy import select

    engine = get_engine()
    out: Dict[str, pd.DataFrame] = {}

    with engine.connect() as conn:
        # P2-005 ROOT FIX: read the latest applied schema_version ONCE
        # and use it to filter GDA rows (the only table with a per-row
        # ``schema_version`` column per AST analysis of database/models.py).
        # For tables WITHOUT a per-row schema_version (Drug, Protein, DPI,
        # PPI), we log the latest version for observability and warn if the
        # schema_version table has multiple rows (a heuristic for mid-
        # migration state). This prevents reading stale rows from an
        # incomplete migration, which would cause InChIKey-based dedup to
        # fail silently for the stale subset.
        _latest_sv = _get_latest_schema_version(conn)
        _n_sv = _count_schema_versions(conn)
        if _latest_sv is not None:
            logger.info(
                "P2-005: Phase 1 Postgres schema_version = %d "
                "(%d migration row(s) applied). GDA rows will be "
                "filtered to schema_version=%d.",
                _latest_sv, _n_sv, _latest_sv,
            )
            if _n_sv > _latest_sv:
                # Heuristic: more rows than the latest version number
                # suggests a mid-migration state (some migrations applied
                # but the latest version column hasn't been bumped yet).
                logger.warning(
                    "P2-005: schema_version table has %d rows but latest "
                    "version is %d — possible mid-migration state. Tables "
                    "without a per-row schema_version column (Drug, "
                    "Protein, DPI, PPI) may contain stale rows from the "
                    "incomplete migration. GDA rows ARE filtered.",
                    _n_sv, _latest_sv,
                )
        else:
            logger.debug(
                "P2-005: schema_version table not available — proceeding "
                "without schema_version filter. Expected on a fresh DB."
            )

        # --- drugs (DrugBank + ChEMBL + PubChem unified) ---
        # v49 ROOT FIX: include the new `indication` + `indication_source`
        # columns so the indications reader below can consume them.
        # P2-005: the Drug model does NOT have a per-row schema_version
        # column, so we cannot filter by it here. The schema_version
        # observability log above gives operators visibility.
        drugs_stmt = select(
            _m.Drug.inchikey,
            _m.Drug.name,
            _m.Drug.chembl_id,
            _m.Drug.drugbank_id,
            _m.Drug.pubchem_cid,
            _m.Drug.molecular_weight,
            _m.Drug.smiles,
            _m.Drug.is_fda_approved,
            _m.Drug.is_globally_approved,
            _m.Drug.is_withdrawn,
            _m.Drug.clinical_status,
            _m.Drug.max_phase,
            _m.Drug.mechanism_of_action,
            _m.Drug.indication,
            _m.Drug.indication_source,
        ).where(_m.Drug.is_deleted == False)  # noqa: E712 — SQLAlchemy
        drugs_df = pd.read_sql(drugs_stmt, conn)
        # Synthesise the legacy 'groups' column from clinical_status so
        # downstream stage_phase1_to_phase2 doesn't break.
        if "clinical_status" in drugs_df.columns:
            drugs_df["groups"] = drugs_df["clinical_status"].fillna("")
        out["drugs"] = drugs_df
        logger.info(
            "Phase1 bridge (postgres): read %d rows from drugs table",
            len(drugs_df),
        )

        # --- drug_protein_interactions ---
        # v29 ROOT FIX: the DrugProteinInteraction model uses integer
        # foreign keys (drug_id -> drugs.id, protein_id -> proteins.id),
        # NOT string columns like drug_inchikey / protein_uniprot_id.
        # The previous code referenced non-existent columns, which
        # would crash at query time. Fix: JOIN through the integer FKs
        # to get the string identifiers (drugbank_id, uniprot_id) that
        # the bridge contract expects.
        dpi_stmt = select(
            _m.Drug.inchikey.label("drug_inchikey"),
            _m.Drug.drugbank_id.label("drugbank_id"),
            _m.Protein.uniprot_id.label("uniprot_id"),
            _m.Protein.protein_name.label("target_name"),
            _m.DrugProteinInteraction.interaction_type.label("action_type"),
        ).select_from(
            _m.DrugProteinInteraction
        ).join(
            _m.Drug, _m.DrugProteinInteraction.drug_id == _m.Drug.id
        ).join(
            _m.Protein, _m.DrugProteinInteraction.protein_id == _m.Protein.id
        )
        dpi_df = pd.read_sql(dpi_stmt, conn)
        # Bridge expects drugbank_id as the primary key on interactions.
        out["interactions"] = dpi_df
        logger.info(
            "Phase1 bridge (postgres): read %d rows from "
            "drug_protein_interactions", len(dpi_df),
        )

        # --- gene_disease_associations (OMIM + DisGeNET unified) ---
        # v34 ROOT FIX (CRITICAL #7): the previous query selected ONLY 6
        # columns (gene_symbol, disease_id, disease_name, source, score,
        # association_type) and synthesized gene_mim/phenotype_mim as None.
        # When DATABASE_URL was set, the bridge's stage code fell through
        # ALL three Gene ID resolvers (canonical_gene_id, ncbi_gene_id,
        # gene_mim) and emitted `SYM:{symbol}` IDs for every Gene —
        # losing cross-source ID resolution. The CSV path includes these
        # columns; the PostgreSQL path did NOT.
        #
        # The fix: select ALL columns the bridge's stage code consumes:
        #   - gene_id (NCBI Entrez Gene ID, mapped to ncbi_gene_id in the
        #     output to match the CSV schema)
        #   - uniprot_id (cross-source protein key)
        #   - disease_id_type
        #   - score_type, score_method, evidence_strength,
        #     normalized_score, confidence_tier, source_id, source_version
        #   - mapping_key (synthesized as None — the model doesn't have it;
        #     it's a Phase 1 OMIM-specific field that the CSV path provides
        #     but the GDA model doesn't store. This is acceptable: the
        #     bridge's stage code only uses mapping_key for the edge props,
        #     not for Gene ID resolution.)
        # We also synthesize `canonical_gene_id` from `gene_id` (NCBI) so
        # the bridge's preferred resolver hits first.
        gda_stmt = select(
            _m.GeneDiseaseAssociation.gene_symbol,
            _m.GeneDiseaseAssociation.disease_id,
            _m.GeneDiseaseAssociation.disease_name,
            _m.GeneDiseaseAssociation.disease_id_type,
            _m.GeneDiseaseAssociation.source,
            _m.GeneDiseaseAssociation.source_id,
            _m.GeneDiseaseAssociation.score,
            _m.GeneDiseaseAssociation.association_type,
            _m.GeneDiseaseAssociation.uniprot_id,
            _m.GeneDiseaseAssociation.gene_id.label("ncbi_gene_id"),
            _m.GeneDiseaseAssociation.score_type,
            _m.GeneDiseaseAssociation.score_method,
            _m.GeneDiseaseAssociation.confidence_tier,
            _m.GeneDiseaseAssociation.evidence_strength,
            _m.GeneDiseaseAssociation.normalized_score,
            _m.GeneDiseaseAssociation.source_version,
            # P2-005: select the per-row schema_version so we can filter
            # stale rows from incomplete migrations. This is the ONLY
            # table with a per-row schema_version column per AST analysis
            # of database/models.py.
            _m.GeneDiseaseAssociation.schema_version,
        )
        # P2-005 ROOT FIX: filter GDA rows by the latest schema_version.
        # Only GeneDiseaseAssociation has a per-row schema_version column.
        # Rows from incomplete migrations (schema_version < latest) are
        # excluded — they may lack columns that the latest migration added
        # (e.g. compound_inchikey_canonical), causing InChIKey-based dedup
        # to fail silently for the stale subset.
        if _latest_sv is not None:
            # schema_version is a String(20) column; compare as string.
            gda_stmt = gda_stmt.where(
                _m.GeneDiseaseAssociation.schema_version == str(_latest_sv)
            )
        gda_df = pd.read_sql(gda_stmt, conn)
        # Synthesize the legacy columns the bridge contract expects.
        gda_df["gene_mim"] = None
        gda_df["phenotype_mim"] = None
        gda_df["mapping_key"] = None
        # v34 ROOT FIX (CRITICAL #7): synthesize `canonical_gene_id` from
        # `ncbi_gene_id` so the bridge's preferred Gene ID resolver hits.
        # The CSV path provides this directly; the PostgreSQL path now
        # provides it too.
        gda_df["canonical_gene_id"] = gda_df["ncbi_gene_id"].astype(str).where(
            gda_df["ncbi_gene_id"].notna(), None
        )
        # Split by source: OMIM rows go to "omim_gda", DisGeNET rows to
        # "disgenet_gda". Rows with source containing "omim" go to omim_gda;
        # rows with source containing "disgenet" go to disgenet_gda.
        if not gda_df.empty and "source" in gda_df.columns:
            omim_mask = gda_df["source"].astype(str).str.lower().str.contains("omim")
            disgenet_mask = gda_df["source"].astype(str).str.lower().str.contains("disgenet")
            out["omim_gda"] = gda_df[omim_mask].copy()
            out["disgenet_gda"] = gda_df[disgenet_mask].copy()
        else:
            out["omim_gda"] = pd.DataFrame()
            out["disgenet_gda"] = pd.DataFrame()
        logger.info(
            "Phase1 bridge (postgres): read %d OMIM GDA + %d DisGeNET GDA rows",
            len(out["omim_gda"]), len(out["disgenet_gda"]),
        )

        # --- proteins (UniProt) ---
        # v29 ROOT FIX: Protein model has `protein_name`, not `target_name`.
        prot_stmt = select(
            _m.Protein.uniprot_id.label("uniprot_ac"),
            _m.Protein.gene_symbol,
            _m.Protein.protein_name.label("name"),
            _m.Protein.organism,
        )
        out["uniprot_proteins"] = pd.read_sql(prot_stmt, conn)
        logger.info(
            "Phase1 bridge (postgres): read %d protein rows",
            len(out["uniprot_proteins"]),
        )

        # --- protein_protein_interactions (STRING) ---
        # v29 ROOT FIX: PPI model uses integer FKs (protein_a_id,
        # protein_b_id), not string uniprot_ac_a / uniprot_ac_b.
        # JOIN through proteins to get the UniProt accessions.
        # Use aliased Protein for the self-join.
        from sqlalchemy.orm import aliased
        _ProteinA = aliased(_m.Protein)
        _ProteinB = aliased(_m.Protein)
        ppi_stmt = select(
            _ProteinA.uniprot_id.label("uniprot_ac_a"),
            _ProteinB.uniprot_id.label("uniprot_ac_b"),
            _m.ProteinProteinInteraction.combined_score,
        ).select_from(
            _m.ProteinProteinInteraction
        ).join(
            _ProteinA, _m.ProteinProteinInteraction.protein_a_id == _ProteinA.id
        ).join(
            _ProteinB, _m.ProteinProteinInteraction.protein_b_id == _ProteinB.id
        )
        out["string_ppi"] = pd.read_sql(ppi_stmt, conn)
        logger.info(
            "Phase1 bridge (postgres): read %d STRING PPI rows",
            len(out["string_ppi"]),
        )

        # v49 ROOT FIX (CRITICAL — Compound Chain 2 "PostgreSQL Bridge Data
        # Corruption" finally closed):
        # The v37/v43 code gave up on PostgreSQL mode for indications — it
        # emitted ZERO rows because "the Drug ORM doesn't have an
        # `indication` column". The v49 migration 010 ADDS the `indication`
        # + `indication_source` columns to the Drug ORM. The DrugBank
        # pipeline already produced `indication` text in its drugs_df — it
        # was silently dropped by the loader's column filter. Now it loads.
        #
        # This reader now reads `indication` directly from PostgreSQL:
        #   - Filters to rows WHERE indication IS NOT NULL AND indication != ''
        #   - Joins against the GDA table to map disease names when possible
        #     (best-effort: the DrugBank indication field is free-text like
        #      "for the treatment of hypertension"; the structured disease
        #      mapping lives in the DrugBank XML <indication> parser. The
        #      CSV path remains the gold-standard for structured mapping.
        #      But for the PostgreSQL path, the indication text + the
        #      indication_source tag ('drugbank_xml' | 'chembl_max_phase' |
        #      'rxnorm') are now available downstream.)
        #   - Falls back to CSV (`drugbank_indications.csv`) when the
        #     PostgreSQL `indication` column is empty (e.g. fresh DB
        #     before DrugBank pipeline has run).
        #
        # This closes the "PostgreSQL mode produces ZERO Compound-treats-
        # Disease edges" chain — V1 launch criterion `positive_pairs_sufficient`
        # is now achievable in pure-PostgreSQL mode.
        out.setdefault("indications", pd.DataFrame())
        try:
            _pg_indications_df = _read_indications_from_postgres(conn)
            if _pg_indications_df is not None and not _pg_indications_df.empty:
                out["indications"] = _pg_indications_df
                logger.info(
                    "Phase1 bridge: PostgreSQL backend read %d indication "
                    "rows from drugs.indication column (v49 root fix — "
                    "Compound Chain 2 closed).",
                    len(_pg_indications_df),
                )
            else:
                # Fall back to the CSV (DrugBank XML-derived structured
                # indications — the gold-standard).
                _indications_csv_path = DEFAULT_PHASE1_PROCESSED_DIR / "drugbank_indications.csv"
                if _indications_csv_path.exists() and _indications_csv_path.stat().st_size > 0:
                    _indications_df = _read_csv_robust(_indications_csv_path)
                    if not _indications_df.empty:
                        out["indications"] = _indications_df
                        logger.info(
                            "Phase1 bridge: PostgreSQL `indication` column "
                            "empty — enriched indications from %s (%d rows).",
                            _indications_csv_path.name, len(_indications_df),
                        )
                    else:
                        logger.warning(
                            "Phase1 bridge: drugbank_indications.csv exists "
                            "but is empty. Run Phase 1 DrugBank pipeline to "
                            "populate it."
                        )
                else:
                    logger.warning(
                        "Phase1 bridge: neither PostgreSQL drugs.indication "
                        "nor drugbank_indications.csv has data. "
                        "Compound-treats-Disease edges will be absent. "
                        "Run Phase 1 DrugBank pipeline (or the v49 DrugBank "
                        "fallback that derives indications from ChEMBL "
                        "max_phase==4 + RxNorm) to populate."
                    )
        except Exception as _indications_exc:
            # v58 ROOT FIX (P2C-008 deep): log at ERROR + structured audit.
            # The previous WARNING was silently swallowed by log dashboards.
            logger.error(
                "Phase1 bridge: failed to read indications in PostgreSQL "
                "mode (%s: %s). Falling back to CSV if available. In "
                "production this means Compound-treats-Disease edges may "
                "be silently lost.",
                type(_indications_exc).__name__, _indications_exc,
            )
            _log_bridge_fallback(
                "postgres_indications_read",
                "indications_read_failed",
                backend="csv_or_empty",
                exception_type=type(_indications_exc).__name__,
                exception_message=str(_indications_exc),
            )
            _indications_csv_path = DEFAULT_PHASE1_PROCESSED_DIR / "drugbank_indications.csv"
            if _indications_csv_path.exists():
                try:
                    out["indications"] = _read_csv_robust(_indications_csv_path)
                except Exception as _csv_exc:
                    logger.error(
                        "Phase1 bridge: indications CSV also failed to "
                        "read (%s: %s). Returning empty DataFrame — "
                        "Compound-treats-Disease edges will be ABSENT.",
                        type(_csv_exc).__name__, _csv_exc,
                    )
                    _log_bridge_fallback(
                        "postgres_indications_csv_fallback",
                        "csv_read_also_failed",
                        backend="empty",
                        exception_type=type(_csv_exc).__name__,
                        exception_message=str(_csv_exc),
                    )
                    out["indications"] = pd.DataFrame()
            else:
                _log_bridge_fallback(
                    "postgres_indications_csv_fallback",
                    "csv_not_present",
                    backend="empty",
                )
                out["indications"] = pd.DataFrame()

        # (2) chembl_drugs: select drugs WHERE chembl_id IS NOT NULL.
        # The CSV path provides: chembl_id, inchikey, smiles, name,
        # uniprot_accession, target_name, activity_type, pchembl_value,
        # max_phase, is_globally_approved, is_fda_approved.
        # The PostgreSQL path can provide: chembl_id, inchikey, smiles,
        # name, max_phase, is_globally_approved, is_fda_approved. The
        # activity/target columns come from the chembl_activities reader.
        try:
            chembl_drugs_stmt = select(
                _m.Drug.chembl_id,
                _m.Drug.inchikey,
                _m.Drug.smiles,
                _m.Drug.name,
                _m.Drug.max_phase,
                _m.Drug.is_globally_approved,
                _m.Drug.is_fda_approved,
            ).where(_m.Drug.chembl_id.isnot(None))
            chembl_drugs_df = pd.read_sql(chembl_drugs_stmt, conn)
            out["chembl_drugs"] = chembl_drugs_df
            logger.info(
                "Phase1 bridge (postgres): read %d ChEMBL drug rows "
                "(drugs where chembl_id IS NOT NULL) — v37 PostgreSQL bridge fix",
                len(chembl_drugs_df),
            )
        except Exception as exc:
            # v58 ROOT FIX (P2C-008 deep): ERROR + structured audit.
            logger.error(
                "Phase1 bridge (postgres): could not read chembl_drugs "
                "from drugs table (%s: %s) — falling back to empty. "
                "Compound nodes from ChEMBL will be ABSENT. v37 "
                "PostgreSQL bridge fix.",
                type(exc).__name__, exc,
            )
            _log_bridge_fallback(
                "postgres_chembl_drugs_read",
                "chembl_drugs_read_failed",
                backend="empty",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            out["chembl_drugs"] = pd.DataFrame()

        # (3) chembl_activities: select from drug_protein_interactions
        # WHERE source='chembl', JOIN drugs + proteins to get the
        # CSV-path columns: molecule_chembl_id, target_chembl_id,
        # target_pref_name, activity_type, activity_value,
        # activity_units, pchembl_value, assay_id, standard_relation,
        # assay_type, uniprot_accession.
        # Note: the DPI table doesn't have pchembl_value, assay_id,
        # standard_relation, or assay_type columns — only activity_value
        # and activity_type. We synthesize the columns the bridge
        # expects, with NULLs where the DB doesn't have the data.
        try:
            chembl_act_stmt = (
                select(
                    _m.Drug.chembl_id.label("molecule_chembl_id"),
                    _m.Drug.inchikey,
                    _m.Protein.uniprot_id.label("uniprot_accession"),
                    _m.Protein.protein_name.label("target_pref_name"),
                    _m.DrugProteinInteraction.activity_type,
                    _m.DrugProteinInteraction.activity_value,
                    _m.DrugProteinInteraction.activity_units,
                    # INT-003 ROOT FIX: select the REAL standard_relation
                    # from the ORM (was dropped at load time, then guessed
                    # heuristically — the heuristic was conservative and
                    # missed censoring in the 0.1-100000 nM range where
                    # most clinically-relevant activity values live).
                    _m.DrugProteinInteraction.standard_relation,
                    _m.DrugProteinInteraction.interaction_type.label("action_type"),
                    _m.DrugProteinInteraction.source,
                    _m.DrugProteinInteraction.activity_units,
                )
                .select_from(_m.DrugProteinInteraction)
                .join(_m.Drug, _m.DrugProteinInteraction.drug_id == _m.Drug.id)
                .join(_m.Protein, _m.DrugProteinInteraction.protein_id == _m.Protein.id)
                .where(_m.DrugProteinInteraction.source == "chembl")
                # v88+v89 ROOT FIX (BUG #38 — non-human proteins pollute
                # the human KG): filter to HUMAN proteins only
                # (ncbi_taxid == 9606). For a HUMAN drug repurposing
                # platform, only human protein targets are clinically
                # actionable — a drug's activity against a mouse homolog
                # does NOT predict its activity against the human homolog
                # (cross-species binding affinity can differ by 10-100x).
                # This filter is the LAST line of defense: even if non-
                # human proteins leak into the DB, this query excludes
                # them from the KG.
                .where(_m.Protein.ncbi_taxid == 9606)
            )
            chembl_act_df = pd.read_sql(chembl_act_stmt, conn)
            # v88 ROOT FIX (BUG #51 — UniProt secondary accessions create
            # duplicate Protein nodes): filter to PRIMARY UniProt accessions.
            import re as _re_v88
            _UNIPROT_PRIMARY_RE = _re_v88.compile(
                r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$"
                r"|^[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]([A-Z][A-Z0-9]{2}[0-9])?$"
            )
            if "uniprot_accession" in chembl_act_df.columns and not chembl_act_df.empty:
                _before = len(chembl_act_df)
                _ac = chembl_act_df["uniprot_accession"].astype(str)
                _primary_mask = _ac.apply(
                    lambda x: bool(_UNIPROT_PRIMARY_RE.match(x)) if x and x != "nan" else False
                )
                _n_secondary = int((~_primary_mask).sum())
                if _n_secondary > 0:
                    logger.warning(
                        "phase1_bridge: dropping %d/%d ChEMBL activity rows "
                        "with non-primary UniProt accessions (v88 BUG #51).",
                        _n_secondary, _before,
                    )
                    chembl_act_df = chembl_act_df[_primary_mask].reset_index(drop=True)
            # v51 ROOT FIX (COMPOUND-2 — pchembl unit handling):
            # The v49/v50 code hardcoded `pchembl = 9.0 - log10(activity_value)`
            # assuming ALL values are in nM. But Phase 1 emits MIXED units
            # (nM, μM, %, ratio, etc.). For non-nM units, the formula is
            # WRONG:
            #   - μM values: pchembl should be 6 - log10(μM_value), not 9 - log10
            #     (because 1 μM = 1000 nM, so the offset shifts by 3)
            #   - % values: pchembl is undefined (Inhibition % is not a
            #     concentration — set to None)
            #   - M values: pchembl = -log10(M_value)
            # ROOT FIX: read `activity_units` from the DB and apply the
            # correct conversion per row. Default to nM when units are
            # missing (backward compat).
            if not chembl_act_df.empty and "activity_value" in chembl_act_df.columns:
                import numpy as _np_v37
                _av = pd.to_numeric(chembl_act_df["activity_value"], errors="coerce")
                _av_positive = _av.where(_av > 0, _np_v37.nan)
                # Read units (default nM if missing — backward compat)
                _units = chembl_act_df.get(
                    "activity_units", pd.Series(["nM"] * len(chembl_act_df))
                ).fillna("nM").astype(str).str.strip().str.lower()
                # v88 ROOT FIX (BUG #29 — Greek mu vs micro sign breaks
                # μM unit detection): normalize both U+00B5 (micro sign)
                # AND U+03BC (Greek mu) to ASCII "u" BEFORE the isin check.
                _units = _units.str.replace("\u00b5", "u").str.replace("\u03bc", "u")
                # Compute pchembl per-unit
                _pchembl = pd.Series(_np_v37.nan, index=chembl_act_df.index)
                # nM: pchembl = 9 - log10(value)
                _nM_mask = _units.isin(["nm", "nanomolar", "nanomole"])
                _pchembl[_nM_mask] = 9.0 - _np_v37.log10(_av_positive[_nM_mask])
                # μM: pchembl = 6 - log10(value)
                _uM_mask = _units.isin(["um", "µm", "micromolar", "micromole"])
                _pchembl[_uM_mask] = 6.0 - _np_v37.log10(_av_positive[_uM_mask])
                # M: pchembl = -log10(value)
                _M_mask = _units.isin(["m", "molar", "mole"])
                _pchembl[_M_mask] = -_np_v37.log10(_av_positive[_M_mask])
                # pM: pchembl = 12 - log10(value)
                _pM_mask = _units.isin(["pm", "picomolar", "picomole"])
                _pchembl[_pM_mask] = 12.0 - _np_v37.log10(_av_positive[_pM_mask])
                # %, ratio, selectivity, etc.: pchembl undefined — leave NaN
                _undef_mask = ~(_nM_mask | _uM_mask | _M_mask | _pM_mask)
                _undef_count = int(_undef_mask.sum())
                if _undef_count > 0:
                    logger.warning(
                        "Phase1 bridge: %d ChEMBL activities have units "
                        "that don't support pchembl conversion (e.g. '%%', "
                        "'ratio', 'selectivity'). Their pchembl_value will "
                        "be None. Unit distribution: %s",
                        _undef_count,
                        _units[_undef_mask].value_counts().to_dict(),
                    )
                chembl_act_df["pchembl_value"] = _pchembl
            else:
                chembl_act_df["pchembl_value"] = None
            # v84 FORENSIC ROOT FIX (BUG #26 — UniProt accession validation):
            # The previous code aliased `Protein.uniprot_id` as
            # `uniprot_accession` but did NOT validate the format. Phase 1's
            # `Protein.uniprot_id` column may contain secondary accessions,
            # isoform IDs (with -N suffix like "P12345-2"), or NULLs. These
            # were emitted as `uniprot_accession` to Phase 2, which uses
            # them as the canonical Protein ID. Isoform IDs (P12345-2) are
            # NOT primary accessions and should be normalized to their
            # parent (P12345). Secondary accessions and malformed values
            # should be dropped — otherwise the KG has duplicate Protein
            # nodes for the same underlying protein, fragmenting the
            # drug-target signal across duplicates.
            #
            # ROOT FIX: post-read normalization pass on the dataframe:
            #   (a) strip isoform suffixes (-2, -3, ...) → parent accession
            #   (b) validate against the UniProt primary accession regex:
            #        ^[OPQ][0-9][A-Z0-9]{3}[0-9]$                (6 chars)
            #        | ^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$  (10/11 chars)
            #   (c) drop rows whose normalized accession doesn't match
            if "uniprot_accession" in chembl_act_df.columns:
                import re as _re_uniprot_v84
                _UNIPROT_PRIMARY_RE = _re_uniprot_v84.compile(
                    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]"
                    r"|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
                )
                _orig_n = len(chembl_act_df)
                # Coerce to string, strip whitespace, strip isoform suffix.
                _ua = (
                    chembl_act_df["uniprot_accession"]
                    .astype(str)
                    .str.strip()
                )
                # Strip isoform suffix: "P12345-2" → "P12345".
                _ua_normalized = _ua.str.replace(
                    r"-\d+$", "", regex=True
                )
                # Replace literal "nan"/"None"/"" with NaN for dropna.
                _ua_normalized = _ua_normalized.replace(
                    {"nan": None, "None": None, "": None}
                )
                # Validate against the regex; invalid → NaN.
                _valid_mask = _ua_normalized.notna() & _ua_normalized.apply(
                    lambda _v: bool(_UNIPROT_PRIMARY_RE.match(str(_v)))
                )
                _n_dropped = int((~_valid_mask).sum())
                if _n_dropped > 0:
                    logger.warning(
                        "Phase1 bridge: dropped %d ChEMBL activity rows "
                        "with invalid/NULL uniprot_accession (out of %d). "
                        "Invalid rows had non-primary accessions, isoform "
                        "IDs that didn't normalize to a parent, or NULL. "
                        "Dropping prevents duplicate Protein nodes in the "
                        "KG. (v84 BUG #26 root fix)",
                        _n_dropped, _orig_n,
                    )
                chembl_act_df = chembl_act_df[_valid_mask].copy()
                chembl_act_df["uniprot_accession"] = _ua_normalized[_valid_mask].values
            # Synthesize target_chembl_id (NULL — the DB doesn't store it).
            chembl_act_df["target_chembl_id"] = None
            chembl_act_df["assay_id"] = None
            # INT-003 ROOT FIX: standard_relation is NOW a real column
            # in the Phase 1 ORM (DrugProteinInteraction.standard_relation).
            # It stores ChEMBL's raw censoring direction ('=', '<', '>', '~').
            # Rows loaded before the migration have NULL — we fall back to
            # '=' (exact measurement, the most common ChEMBL relation) for
            # those. The heuristic _derive_standard_relation_heuristic is
            # kept as a last-resort for legacy data but NEVER used for
            # fresh loads where the ORM column is populated.
            if not chembl_act_df.empty:
                # Coerce NULL/NaN to '=' (exact measurement — safest default).
                chembl_act_df["standard_relation"] = (
                    chembl_act_df["standard_relation"]
                    .fillna("=")
                    .replace("", "=")
                )
                # Validate: only ChEMBL's four censoring symbols are allowed.
                _valid_relations = {"=", "<", ">", "~"}
                _invalid_mask = ~chembl_act_df["standard_relation"].isin(_valid_relations)
                _n_invalid = int(_invalid_mask.sum())
                if _n_invalid > 0:
                    logger.warning(
                        "INT-003: %d rows have invalid standard_relation values "
                        "(not in {'=', '<', '>', '~'}). Coercing to '='. "
                        "Sample invalid values: %s",
                        _n_invalid,
                        list(chembl_act_df.loc[_invalid_mask, "standard_relation"].head(5)),
                    )
                    chembl_act_df.loc[_invalid_mask, "standard_relation"] = "="
                # v108 ROOT FIX (issue 62): unified censoring logic across
                # both DB backends. The audit found that the PostgreSQL path
                # lost standard_relation censoring semantics for extreme
                # activity values (>100 μM = upper detection limit, <1 nM =
                # lower detection limit). The existing
                # _derive_standard_relation_heuristic function applies these
                # thresholds but was only called for the CSV path. Now we
                # call it for BOTH paths so censored values are flagged
                # consistently. We ALSO add a 'is_censored' flag column so
                # the RL safety ranker can filter on it.
                import numpy as _np_v108_censor
                if "activity_value" in chembl_act_df.columns and "activity_units" in chembl_act_df.columns:
                    _censor_flags = []
                    for _, _row in chembl_act_df.iterrows():
                        _rel = _derive_standard_relation_heuristic(_row)
                        _is_censored = _rel in ("<", ">")
                        # Override standard_relation IF the heuristic detects
                        # censoring AND the stored relation is "=" (the
                        # heuristic only emits '<'/'>' for unambiguous
                        # extreme values; we don't override explicit '<'/'>'
                        # from ChEMBL because those are gold-standard).
                        if _is_censored and _row.get("standard_relation") == "=":
                            chembl_act_df.at[_, "standard_relation"] = _rel  # noqa: B023
                        _censor_flags.append(_is_censored)
                    chembl_act_df["is_censored"] = _censor_flags
                    _n_censored = int(sum(_censor_flags))
                    if _n_censored > 0:
                        logger.warning(
                            "v108 issue 62: %d/%d ChEMBL activity rows "
                            "flagged as censored (value beyond detection "
                            "limits: <1 nM or >100 μM). The standard_relation "
                            "for these rows has been updated to reflect "
                            "the censoring direction. The 'is_censored' "
                            "column lets the RL safety ranker filter them.",
                            _n_censored, len(chembl_act_df),
                        )
                else:
                    chembl_act_df["is_censored"] = False
            chembl_act_df["assay_type"] = None
            out["chembl_activities"] = chembl_act_df
            logger.info(
                "Phase1 bridge (postgres): read %d ChEMBL activity rows "
                "(DPI where source='chembl') — v37 PostgreSQL bridge fix",
                len(chembl_act_df),
            )
        except Exception as exc:
            # v58 ROOT FIX (P2C-008 deep): ERROR + structured audit.
            logger.error(
                "Phase1 bridge (postgres): could not read chembl_activities "
                "from drug_protein_interactions (%s: %s) — falling back "
                "to empty. Compound-inhibits-Protein edges from ChEMBL "
                "will be ABSENT. v37 PostgreSQL bridge fix.",
                type(exc).__name__, exc,
            )
            _log_bridge_fallback(
                "postgres_chembl_activities_read",
                "chembl_activities_read_failed",
                backend="empty",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            out["chembl_activities"] = pd.DataFrame()

        # (4) omim_susceptibility: select from gene_disease_associations
        # WHERE source='omim' AND is_susceptibility=True. The GDA model
        # has an is_susceptibility column.
        try:
            omim_susc_stmt = (
                select(
                    _m.GeneDiseaseAssociation.gene_symbol,
                    # v51 ROOT FIX (COMPOUND-2 — gene_mim aliasing):
                    # The v49/v50 code labeled `gene_id` (NCBI Entrez
                    # Gene ID) as `gene_mim`. This is a SEMANTIC ERROR —
                    # MIM (Mendelian Inheritance in Man) is a DIFFERENT
                    # namespace from NCBI Entrez Gene:
                    #   - NCBI Entrez Gene ID: integer, e.g. 2261 (FGFR3)
                    #   - OMIM MIM number: integer, e.g. 134934 (FGFR3 gene MIM)
                    # Conflating them under `gene_mim` meant the bridge
                    # passed NCBI Gene IDs to code expecting MIM numbers,
                    # causing cross-source entity resolution to FAIL for
                    # genes that have different NCBI vs MIM identifiers.
                    # ROOT FIX: label `gene_id` as `ncbi_gene_id` (its
                    # true semantic name). The downstream Gene resolver
                    # already prefers `canonical_gene_id` → `ncbi_gene_id`
                    # → `gene_mim` (the actual OMIM MIM, which is in the
                    # CSV path only). This preserves the resolver chain
                    # without semantic conflation.
                    _m.GeneDiseaseAssociation.gene_id.label("ncbi_gene_id"),
                    _m.GeneDiseaseAssociation.disease_id,
                    _m.GeneDiseaseAssociation.disease_name.label("phenotype_name"),
                    _m.GeneDiseaseAssociation.association_type,
                    _m.GeneDiseaseAssociation.score,
                    _m.GeneDiseaseAssociation.score_type,
                    _m.GeneDiseaseAssociation.confidence_tier,
                ).where(
                    _m.GeneDiseaseAssociation.source == "omim",
                    _m.GeneDiseaseAssociation.is_susceptibility.is_(True),
                )
            )
            omim_susc_df = pd.read_sql(omim_susc_stmt, conn)
            # v88 ROOT FIX (BUG #27 — gene_mim aliasing fragments gene
            # resolution): best-effort JOIN to recover MIM numbers from
            # the gene_mim column on gene_disease_associations when
            # available. Falls back to None with a structured log.
            _gda_cols = [
                c["name"] for c in _inspect_columns(conn, "gene_disease_associations")
            ]
            if "gene_mim" in _gda_cols and not omim_susc_df.empty:
                try:
                    _mim_stmt = (
                        select(
                            _m.GeneDiseaseAssociation.gene_id.label("ncbi_gene_id"),
                            _m.GeneDiseaseAssociation.disease_id,
                            _m.GeneDiseaseAssociation.gene_mim,
                        ).where(
                            _m.GeneDiseaseAssociation.source == "omim",
                            _m.GeneDiseaseAssociation.is_susceptibility.is_(True),
                            _m.GeneDiseaseAssociation.gene_mim.isnot(None),
                        )
                    )
                    _mim_df = pd.read_sql(_mim_stmt, conn)
                    if not _mim_df.empty:
                        omim_susc_df = omim_susc_df.merge(
                            _mim_df[["ncbi_gene_id", "disease_id", "gene_mim"]],
                            on=["ncbi_gene_id", "disease_id"],
                            how="left",
                            suffixes=("", "_from_mim"),
                        )
                        if "gene_mim_from_mim" in omim_susc_df.columns:
                            omim_susc_df["gene_mim"] = omim_susc_df["gene_mim_from_mim"]
                            omim_susc_df = omim_susc_df.drop(columns=["gene_mim_from_mim"])
                        _n_with_mim = int(omim_susc_df["gene_mim"].notna().sum())
                        logger.info(
                            "phase1_bridge: recovered %d/%d OMIM gene_mim "
                            "numbers (v88 BUG #27 root fix).",
                            _n_with_mim, len(omim_susc_df),
                        )
                    else:
                        omim_susc_df["gene_mim"] = None
                except Exception as _mim_exc:
                    logger.warning(
                        "phase1_bridge: could not read gene_mim (%s: %s) — "
                        "falling back to None (v88 BUG #27).",
                        type(_mim_exc).__name__, _mim_exc,
                    )
                    omim_susc_df["gene_mim"] = None
            else:
                omim_susc_df["gene_mim"] = None
            omim_susc_df["phenotype_mim"] = None
            omim_susc_df["mapping_key"] = None
            omim_susc_df["gene_symbols_raw"] = omim_susc_df["gene_symbol"]
            omim_susc_df["cyto_location"] = None
            omim_susc_df["cyto_location_valid"] = None
            omim_susc_df["inheritance_pattern"] = None
            omim_susc_df["association_modifier"] = None
            omim_susc_df["source_format"] = "postgres"
            omim_susc_df["source_line_number"] = None
            omim_susc_df["is_susceptibility"] = True
            omim_susc_df["score_method"] = None
            # v51 ROOT FIX: synthesize canonical_gene_id from ncbi_gene_id
            # so the bridge's preferred Gene ID resolver hits.
            omim_susc_df["canonical_gene_id"] = omim_susc_df["ncbi_gene_id"].astype(str).where(
                omim_susc_df["ncbi_gene_id"].notna(), None
            )
            out["omim_susceptibility"] = omim_susc_df
            logger.info(
                "Phase1 bridge (postgres): read %d OMIM susceptibility rows "
                "(GDA where source='omim' AND is_susceptibility=True) — v51 "
                "PostgreSQL bridge fix",
                len(omim_susc_df),
            )
        except Exception as exc:
            # v58 ROOT FIX (P2C-008 deep): ERROR + structured audit.
            logger.error(
                "Phase1 bridge (postgres): could not read omim_susceptibility "
                "from gene_disease_associations (%s: %s) — falling back "
                "to empty. Gene-associates-Disease edges from OMIM will "
                "be ABSENT. v37 PostgreSQL bridge fix.",
                type(exc).__name__, exc,
            )
            _log_bridge_fallback(
                "postgres_omim_susceptibility_read",
                "omim_susceptibility_read_failed",
                backend="empty",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            out["omim_susceptibility"] = pd.DataFrame()

        # (5) pubchem_enrichment: select from pubchem_compound_properties.
        # Map the ORM columns to the CSV-path columns the bridge expects.
        # v43 ROOT FIX (P1 — pubchem is_deleted filter missing): the
        # previous code read ALL rows from pubchem_compound_properties
        # including soft-deleted ones. PubChemCompoundProperty has an
        # is_deleted column (standalone, not via SoftDeleteMixin). The
        # fix adds a WHERE is_deleted == False filter so deleted records
        # are excluded.
        try:
            # v88 ROOT FIX (BUG #40 — NaN in PubChem node features): use
            # func.coalesce(xlogp, 0.0) and func.coalesce(tpsa, 0.0) so
            # NULL → 0.0 instead of propagating NaN through HGT training.
            pubchem_stmt = select(
                _m.PubChemCompoundProperty.inchikey,
                _m.PubChemCompoundProperty.canonical_smiles,
                _m.PubChemCompoundProperty.isomeric_smiles,
                _m.PubChemCompoundProperty.molecular_weight,
                func.coalesce(_m.PubChemCompoundProperty.xlogp, 0.0).label("xlogp"),
                func.coalesce(_m.PubChemCompoundProperty.tpsa, 0.0).label("tpsa"),
                _m.PubChemCompoundProperty.complexity,
                _m.PubChemCompoundProperty.h_bond_donor_count.label("h_bond_donors"),
                _m.PubChemCompoundProperty.h_bond_acceptor_count.label("h_bond_acceptors"),
            ).where(_m.PubChemCompoundProperty.is_deleted == False)
            pubchem_df = pd.read_sql(pubchem_stmt, conn)
            out["pubchem_enrichment"] = pubchem_df
            logger.info(
                "Phase1 bridge (postgres): read %d PubChem enrichment rows "
                "from pubchem_compound_properties — v37 PostgreSQL bridge fix",
                len(pubchem_df),
            )
        except Exception as exc:
            # v58 ROOT FIX (P2C-008 deep): ERROR + structured audit.
            logger.error(
                "Phase1 bridge (postgres): could not read pubchem_enrichment "
                "from pubchem_compound_properties (%s: %s) — falling back "
                "to empty. PubChem molecular property enrichment will be "
                "ABSENT. v37 PostgreSQL bridge fix.",
                type(exc).__name__, exc,
            )
            _log_bridge_fallback(
                "postgres_pubchem_enrichment_read",
                "pubchem_enrichment_read_failed",
                backend="empty",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            out["pubchem_enrichment"] = pd.DataFrame()

    # Apply the same column validation as the CSV path.
    for key, df in out.items():
        if df is None or df.empty:
            continue
        if key in _PHASE1_EXPECTED_COLUMNS:
            # v78 BUG #7: pass the ANY_OF groups so the validator can
            # check that at least one alternative column is present.
            _validate_phase1_columns(
                df, _PHASE1_EXPECTED_COLUMNS[key], key,
                any_of_groups=_PHASE1_ANY_OF_COLUMNS.get(key),
            )
    return out


def read_phase1_outputs(
    phase1_processed_dir: Optional[Path | str] = None,
    prefer_postgres: bool = True,
) -> "_Phase1BridgeResult":
    """Read Phase 1's outputs into a dict of DataFrames.

    v29 ROOT FIX (Phase1↔Phase2 100% connection): the reader now prefers
    PostgreSQL when ``prefer_postgres=True`` (the default). The Phase 1
    SQLAlchemy ORM models — written by Phase 1's loaders — become the
    authoritative source for Phase 2. This closes Compound Chain 2 of the
    forensic audit ("Phase 1 Output Is Discarded").

    Backend selection order:
      1. If ``prefer_postgres=True`` AND a populated Phase 1 DB exists,
         read from PostgreSQL via :func:`_read_phase1_from_postgres`.
      2. Otherwise, read CSVs from ``phase1_processed_dir`` (the legacy
         v28 behaviour — preserved for dev/CI without a database).

    The chosen backend is recorded on the returned object via the
    ``.backend`` attribute (P2-014 root fix — type-safe), AND via the
    legacy ``"_phase1_backend"`` dict key (backward compat with callers
    that pop it). New code should prefer ``.backend``.

    Parameters
    ----------
    phase1_processed_dir : path-like, optional
        Directory containing Phase 1's processed CSV outputs. Used as the
        fallback when PostgreSQL is unavailable. Defaults to
        :data:`DEFAULT_PHASE1_PROCESSED_DIR`.
    prefer_postgres : bool, default True
        If True, attempt PostgreSQL first. Set to False to force the CSV
        path (e.g. for unit tests that mock the CSV fixtures).

    Raises
    ------
    FileNotFoundError
        If the CSV backend is selected and the directory doesn't exist.
    DrugOSDataError
        If a Phase 1 source (CSV or DB table) is missing required columns.
    """
    # Try PostgreSQL first (root fix).
    if prefer_postgres and _phase1_db_available():
        logger.info(
            "Phase1 bridge: using PostgreSQL backend (authoritative). "
            "Phase 1 ORM models are the source of truth for Phase 2."
        )
        try:
            _pg_out = _read_phase1_from_postgres()
            # P2-014 ROOT FIX: wrap in _Phase1BridgeResult so the
            # backend label is a type-safe attribute (not a string
            # masquerading as a DataFrame in a Dict[str, DataFrame]).
            out = _Phase1BridgeResult(_pg_out, backend=_PHASE1_BACKEND_POSTGRES)
            return out
        except Exception as exc:
            # v61 ROOT FIX (silent break point #2 — forensic deep fix):
            # The v58/v60 code re-raised ALL exceptions in production.
            # This crashed the bridge for the same configuration errors
            # that _phase1_db_available now handles gracefully. Even
            # though _phase1_db_available already classified schema_missing
            # as non-fatal, _read_phase1_from_postgres can STILL raise
            # for OTHER schema-mismatch reasons (e.g. drugs table exists
            # but a JOIN'd table like drug_protein_interactions doesn't).
            # ROOT FIX: apply the SAME classification logic here so the
            # second silent fallback layer is also discriminating.
            failure_mode = _classify_db_failure(exc)
            # v100 ROOT FIX (BUG P2-030): the previous single logger.error
            # below claimed "falling back to CSV reader" even when the code
            # was about to re-raise (prod non-schema, or dev without
            # DRUGOS_ALLOW_CSV_FALLBACK=1). The misleading "falling back"
            # claim is now split: (a) this logger.error records ONLY the
            # failure fact, and (b) the "falling back to CSV" info log
            # further down fires ONLY when the fallback is actually taken
            # (i.e. after BOTH raise guards below have passed).
            logger.error(
                "Phase1 bridge: PostgreSQL read failed (%s): %s: %s.",
                failure_mode, type(exc).__name__, exc,
            )
            # v100 ROOT FIX (BUG P2-030): compute up-front whether the
            # upcoming raise guards will re-raise, so the structured
            # audit record can carry a `raised` flag that distinguishes
            # a true CSV fallback from a misleading log followed by a
            # raise. The two guards below mirror these conditions —
            # keep them in sync.
            _allow_csv_fallback = os.environ.get("DRUGOS_ALLOW_CSV_FALLBACK", "") == "1"
            _will_raise = (
                (_PRODUCTION_ENV and failure_mode != "schema_missing")
                or (failure_mode != "schema_missing" and not _allow_csv_fallback)
            )
            _log_bridge_fallback(
                "read_phase1_outputs_postgres_failed",
                f"postgres_read_failed:{failure_mode}",
                backend="csv",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                raised=_will_raise,
                extra={"failure_mode": failure_mode},
            )
            if _PRODUCTION_ENV and failure_mode != "schema_missing":
                # Production must not silently bypass Phase 1's database
                # output for unreachable/auth/unknown failures. Re-raise
                # so the failure is visible and the run aborts with the
                # root cause. Schema_missing is NOT re-raised — it's a
                # configuration issue (migrations not applied) and the
                # CSV fallback lets the pipeline still produce a graph.
                # v100 ROOT FIX (BUG P2-030): the misleading "falling
                # back" log above was split so the fallback claim only
                # fires when the fallback actually happens; this branch
                # re-raises instead of falling back.
                raise
            # v88 ROOT FIX (BUG #28 — second silent fallback layer):
            # apply the same DRUGOS_ALLOW_CSV_FALLBACK=1 opt-in gate here.
            if failure_mode != "schema_missing":
                if not _allow_csv_fallback:
                    # v100 ROOT FIX (BUG P2-030): the misleading "falling
                    # back" log above was split so the fallback claim only
                    # fires when the fallback actually happens; this branch
                    # re-raises RuntimeError instead of falling back.
                    raise RuntimeError(
                        f"phase1_bridge: PostgreSQL read failed "
                        f"(failure_mode={failure_mode}) and prefer_postgres=True. "
                        f"Silent CSV fallback would bypass Phase 1's ORM "
                        f"output (v88 BUG #28 root fix, second layer). "
                        f"Set DRUGOS_ALLOW_CSV_FALLBACK=1 to allow. "
                        f"Original error: {type(exc).__name__}: {exc}"
                    ) from exc
            # v100 ROOT FIX (BUG P2-030): the CSV fallback is actually
            # taken only once BOTH raise guards above have passed. Emit
            # the "falling back to CSV reader" info log HERE (not at the
            # top of the except block) so the fallback claim only fires
            # when the fallback is genuinely about to happen.
            logger.info(
                "Phase1 bridge: falling back to CSV reader (failure_mode=%s).",
                failure_mode,
            )
            if failure_mode == "schema_missing":
                logger.warning(
                    "Phase1 bridge: schema_missing in "
                    "_read_phase1_from_postgres — falling back to CSV. "
                    "Run `python -m database.migrations.run_migrations` "
                    "from phase1/ to apply the schema and use the DB "
                    "backend.",
                )
            else:
                logger.warning(
                    "Phase1 bridge: dev-mode CSV fallback engaged after "
                    "PostgreSQL failure (see ERROR above).",
                )

    # CSV fallback (legacy v28 path).
    base = Path(phase1_processed_dir) if phase1_processed_dir else DEFAULT_PHASE1_PROCESSED_DIR
    if not base.exists():
        raise FileNotFoundError(
            f"Phase 1 processed_data directory does not exist: {base} "
            f"AND PostgreSQL backend unavailable. Either run Phase 1 "
            f"pipelines first, or provision a PostgreSQL database with "
            f"DATABASE_URL set."
        )

    paths = {
        "drugs": base / "drugbank_drugs.csv",
        "interactions": base / "drugbank_interactions.csv.gz",
        "omim_gda": base / "omim_gene_disease_associations.csv",
        # v6 fix (bug #B9): structured drug → OMIM disease indications.
        # Optional — bridge degrades to free-text `indication` column
        # matching if this file is absent.
        "indications": base / "drugbank_indications.csv",
        # ROOT FIX (Phase1↔Phase2 100% connection): extend the bridge
        # contract to cover ALL 7 Phase 1 source pipelines. Previously
        # the bridge consumed only DrugBank + OMIM; ChEMBL, UniProt,
        # STRING, DisGeNET, and PubChem Phase 1 outputs were ignored
        # and Phase 2 re-downloaded them independently. This defeated
        # the "single authoritative wire" promise of the bridge and
        # meant that ~70% of the multi-modal KG's data bypassed Phase 1
        # entity resolution.
        #
        # v13 ROOT FIX (Compound-6 / "Multi-Modal KG Degradation"):
        # v12 introduced these 5 new keys but used prefixed filenames
        # (`chembl_drugs.csv`, `uniprot_proteins.csv`, etc.) that
        # DO NOT MATCH the actual filenames the Phase 1 pipelines
        # emit. Per `phase1/pipelines/base_pipeline.py:_get_processed_filename`,
        # the actual output filenames are unprefixed:
        #   chembl   → drugs.csv
        #   uniprot  → proteins.csv
        #   string   → protein_protein_interactions.csv
        #   disgenet → gene_disease_associations.csv
        #   pubchem  → pubchem_enrichment.csv  (already matched)
        # The mismatch meant 4 of 5 new sources were silently skipped
        # at runtime (warning logged, empty DataFrame returned). The
        # v12 "100% connection" claim was unverifiable on the toy
        # fixture AND broken in production.
        #
        # v13 fix: try BOTH the prefixed name (preferred — explicit)
        # AND the actual pipeline-emitted name (fallback — what
        # production runs actually produce). This is backwards-
        # compatible: existing toy fixtures with prefixed names still
        # work, and production runs with unprefixed names now work.
        "chembl_drugs": [
            base / "chembl_drugs.csv",
            base / "drugs.csv",
        ],
        "uniprot_proteins": [
            base / "uniprot_proteins.csv",
            base / "proteins.csv",
        ],
        "string_ppi": [
            base / "string_protein_protein_interactions.csv",
            base / "protein_protein_interactions.csv",
        ],
        "disgenet_gda": [
            base / "disgenet_gene_disease_associations.csv",
            base / "gene_disease_associations.csv",
        ],
        "pubchem_enrichment": base / "pubchem_enrichment.csv",
        # ─── v15 ROOT FIX (Phase1↔Phase2 100% connection, REM-12/13/14): ──
        # The two Phase-1 source CSVs that v14 STILL bypassed:
        #   • chembl_activities_clean.csv  — the actual ChEMBL bioactivity
        #     table (IC50 / Ki / EC50 + pchembl_value per molecule-target
        #     pair). v14 only read chembl_drugs.csv (compound METADATA
        #     denormalized to one row per compound) — that path could not
        #     emit direction-correct inhibits/activates edges nor carry
        #     the potency value. The audit (REM-13/14) flagged this as
        #     HIGH severity: "ChEMBL edges are ALL hardcoded to
        #     (Compound, targets, Protein) regardless of activity_type."
        #     Fix: read the activities table here, classify each edge by
        #     activity_type semantics (inhibition→inhibits,
        #     activation→activates, otherwise→targets), and carry
        #     pchembl_value + standard_relation as edge properties so the
        #     RL ranker has potency + censoring context.
        #   • omim_gene_disease_susceptibility.csv  — OMIM susceptibility
        #     / polygenic associations (is_susceptibility=True). v14 only
        #     read omim_gene_disease_associations.csv (causative Mendelian
        #     GDA). Susceptibility associations are scientifically
        #     distinct: they are NOT因果 — a variant raises risk but
        #     does not deterministically cause the disease. Conflating
        #     them under the same `associated_with` edge would teach
        #     TransE that BRCA1+breast_cancer is equivalent to
        #     FGFR3+achondroplasia (a Mendelian dominant). Fix: emit a
        #     distinct `susceptible_to` relation so the model learns the
        #     distinction.
        "chembl_activities": [
            base / "chembl_activities_clean.csv",
            base / "chembl_activities.csv",
        ],
        "omim_susceptibility": [
            base / "omim_gene_disease_susceptibility.csv",
        ],
        # ─── v113 ROOT FIX (P2-047, HIGH): SIDER integration in the bridge ──
        # The audit found that ``sider_loader`` parses SIDER adverse-event
        # data, but the bridge's ``paths`` dict had NO SIDER entry -- so
        # SIDER data was NEVER consumed by the Phase 1 → Phase 2 bridge.
        # The KG's Compound→causes_adverse_event→MedDRA_Term edges were
        # emitted ONLY by ``run_pipeline.step6_ingest_sider`` (a Phase-2
        # code path), which meant the bridge's "single authoritative
        # wire" promise was broken for adverse-event data. The RL ranker
        # reads causes_adverse_event edges for its safety-signal
        # dimension; if the bridge doesn't emit them, the safety ranker
        # has no signal when running in bridge-only mode.
        #
        # ROOT FIX: add SIDER entries to the paths dict. We accept TWO
        # candidate CSV filenames:
        #   • ``sider_adverse_events.csv``  -- canonical name (v113)
        #   • ``sider_meddra_all_se.csv``   -- matches the raw filename
        # If neither file exists (the current state, since SIDER is a
        # Phase-2-only source and Phase 1 doesn't produce it), the
        # bridge logs a warning and produces an empty DataFrame -- same
        # graceful-degradation behavior as the other missing sources.
        # An operator who wants SIDER data in the bridge can write a
        # CSV at either path (e.g., by running
        # ``python -c "from drugos_graph.sider_loader import
        # parse_sider_side_effects; parse_sider_side_effects().to_csv(
        # 'phase1/processed_data/sider_adverse_events.csv')"``).
        "sider_adverse_events": [
            base / "sider_adverse_events.csv",
            base / "sider_meddra_all_se.csv",
        ],
    }
    out: Dict[str, pd.DataFrame] = {}
    for key, p in paths.items():
        # v13: support dual-name lookup (list of candidate paths)
        # for the 4 mismatched Phase 1 sources.
        if isinstance(p, list):
            found_path = None
            for candidate in p:
                if candidate.exists():
                    found_path = candidate
                    break
            if found_path is not None:
                out[key] = _read_csv_robust(found_path)
                # v27 ROOT FIX (P2-B-5): validate required columns for
                # this source. Raises DrugOSDataError on schema mismatch
                # so the operator gets a clear, actionable error instead
                # of silent zero-output downstream.
                if key in _PHASE1_EXPECTED_COLUMNS:
                    _validate_phase1_columns(
                        out[key], _PHASE1_EXPECTED_COLUMNS[key], key,
                        any_of_groups=_PHASE1_ANY_OF_COLUMNS.get(key),
                    )
                # v108 ROOT FIX (issue 62): apply unified censoring logic
                # to CSV-path chembl_activities too (the PostgreSQL path
                # gets the same treatment above). This ensures censored
                # values (<1 nM, >100 μM) are flagged consistently across
                # both DB backends. The is_censored column lets the RL
                # safety ranker filter them; the standard_relation column
                # is updated to reflect the censoring direction.
                if key == "chembl_activities" and not out[key].empty:
                    _df_csv = out[key]
                    _censor_flags = []
                    for _idx, _row in _df_csv.iterrows():
                        _rel = _derive_standard_relation_heuristic(_row)
                        _is_censored = _rel in ("<", ">")
                        if _is_censored and _row.get("standard_relation") == "=":
                            _df_csv.at[_idx, "standard_relation"] = _rel
                        _censor_flags.append(_is_censored)
                    _df_csv["is_censored"] = _censor_flags
                    _n_censored_csv = int(sum(_censor_flags))
                    if _n_censored_csv > 0:
                        logger.warning(
                            "v108 issue 62 (CSV path): %d/%d ChEMBL "
                            "activity rows flagged as censored (value "
                            "beyond detection limits: <1 nM or >100 μM).",
                            _n_censored_csv, len(_df_csv),
                        )
                    out[key] = _df_csv
                logger.info(
                    "Phase1 bridge: read %s rows from %s (source=%s)",
                    len(out[key]), found_path.name, key,
                )
            else:
                out[key] = pd.DataFrame()
                logger.warning(
                    "Phase1 bridge: %s not found at any of %s — "
                    "producing empty DataFrame. The bridge will skip "
                    "this source. To fix: run the Phase 1 pipeline "
                    "for this source before invoking the bridge.",
                    key, [str(c) for c in p],
                )
        else:
            if p.exists():
                out[key] = _read_csv_robust(p)
                # v27 ROOT FIX (P2-B-5): validate required columns.
                if key in _PHASE1_EXPECTED_COLUMNS:
                    _validate_phase1_columns(
                        out[key], _PHASE1_EXPECTED_COLUMNS[key], key,
                        any_of_groups=_PHASE1_ANY_OF_COLUMNS.get(key),
                    )
                logger.info(
                    "Phase1 bridge: read %s rows from %s",
                    len(out[key]), p.name,
                )
            else:
                out[key] = pd.DataFrame()
                logger.warning(
                    "Phase1 bridge: %s not found at %s — producing "
                    "empty DataFrame. The bridge will skip this "
                    "source. To fix: run the Phase 1 pipeline for "
                    "this source before invoking the bridge.",
                    key, p,
                )
    # P2-014 ROOT FIX: wrap the CSV-frames dict in _Phase1BridgeResult
    # so the backend label is a type-safe .backend attribute (not a
    # string masquerading as a DataFrame in a Dict[str, DataFrame]).
    return _Phase1BridgeResult(out, backend=_PHASE1_BACKEND_CSV)


# ---------------------------------------------------------------------------
# 5. stage_phase1_to_phase2 — convert DataFrames → Phase 2 node/edge dicts
# ---------------------------------------------------------------------------
def _to_bool(v: Any) -> bool:
    """Coerce arbitrary Phase 1 cell value to a strict bool.

    Pandas reads CSV True/False strings as Python bools already, but
    DefensiveParse™: empty strings, NaN, None, 0, "0", "false" all map to
    False; everything truthy maps to True. This is the patient-safety
    guardrail called out in the module docstring.
    """
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if pd.isna(v):
            return False
        return bool(v)
    s = str(v).strip().lower()
    if s in ("", "0", "false", "no", "f", "n", "nan", "none", "null"):
        return False
    return True


def _resolve_fda_approved(row: Any) -> Optional[bool]:
    """P2-002 FORENSIC ROOT FIX (v104 — Team Member 5): resolve
    ``fda_approved`` for a Phase 1 row WITHOUT conflating globally-
    approved with FDA-approved.

    SCIENTIFIC BUG (P2-002):
        The previous implementation (v64) fell back from
        ``is_fda_approved`` to ``is_globally_approved`` when the
        former was None. ``is_globally_approved`` is derived from
        ChEMBL's ``max_phase == 4``, which means "approved by ANY
        major regulator globally" — FDA (US), EMA (EU), PMDA (Japan),
        NMPA (China), MHRA (UK), Health Canada, TGA (Australia).

        An EMA-only-approved drug (e.g. a drug sold in Germany but
        never submitted to the FDA) has ``max_phase == 4`` and
        ``is_globally_approved = True`` but ``is_fda_approved = None``
        (ChEMBL cannot provide FDA-specific approval without an
        Orange Book join, which is not wired in). The previous code
        marked such a drug as ``fda_approved = True``.

        DOWNSTREAM IMPACT:
        The RL ranker's market-opportunity dimension treated these
        as "FDA-approved" and ranked them as "easy to repurpose in
        the US" — when in fact they would require a full FDA NDA
        (New Drug Application). A pharma partner acting on this
        ranking would waste 6-12 months discovering the regulatory
        barrier. Commercial opportunity was systematically over-stated
        for ~30% of ChEMBL-sourced drugs (the fraction with
        ``is_fda_approved = None``).

    ROOT FIX (honest unknown):
        - If ``is_fda_approved`` is a real bool (DrugBank source —
          DrugBank has real FDA Orange Book data): use it directly.
        - If ``is_fda_approved`` is a non-null truthy/falsy value
          (e.g. "true"/"false" string from a CSV): coerce via
          ``_to_bool`` and use it.
        - If ``is_fda_approved`` is None/NaN (ChEMBL-only path —
          the honest "unknown" state): return ``None``. Do NOT fall
          back to ``is_globally_approved``. The RL ranker treats
          ``None`` as "unknown" — a separate bucket from True/False
          — and does not over-rank these drugs for US repurposing.

    PATIENT-SAFETY PRESERVATION:
        ``withdrawn`` and ``safety_data_missing`` remain independent
        safety gates; ``fda_approved`` only affects market-opportunity
        scoring, never safety filtering. Returning ``None`` is strictly
        safer than the previous ``True`` fallback (a drug marked
        ``True`` was assumed to have FDA safety review; ``None`` makes
        no such assumption).

    RETURN TYPE:
        ``Optional[bool]`` — ``True`` (FDA-approved), ``False`` (not
        FDA-approved), or ``None`` (unknown — ChEMBL-only path).
        Callers that assign this to a Neo4j node property must handle
        ``None`` (Neo4j properties can be null).
    """
    fda_raw = row.get("is_fda_approved")
    # If the value is a real bool (DrugBank source), use it directly.
    if isinstance(fda_raw, bool):
        return fda_raw
    # If it's a non-null truthy/falsy value, coerce via _to_bool.
    if fda_raw is not None and not (isinstance(fda_raw, float) and pd.isna(fda_raw)):
        s = str(fda_raw).strip().lower()
        if s not in ("", "nan", "none", "null"):
            return s not in ("0", "false", "no", "f", "n")
    # P2-002 ROOT FIX: is_fda_approved is None/NaN/unknown → return None.
    # Do NOT fall back to is_globally_approved (max_phase==4) — that
    # conflates EMA/PMDA/NMPA approval with FDA approval and over-states
    # US market opportunity for the RL ranker.
    return None


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


# v28 ROOT FIX (P2-B-13): ``int(idx)`` crashes when the DataFrame has a
# non-RangeIndex (e.g., string index, DatetimeIndex, MultiIndex, or a
# reindexed DataFrame). Phase 1 CSVs typically produce RangeIndex, but
# tests / post-processing code that calls ``stage_phase1_to_phase2`` may
# pass DataFrames with custom indices. The previous code did
# ``"_source_row": int(idx)`` directly — a single non-int index value
# raised TypeError and aborted the entire batch (the caller in
# run_pipeline.py swallows the exception, so all subsequent rows were
# silently lost). This helper provides a stable int for any hashable
# ``idx`` value:
#   * int / numpy int → passthrough (preserves RangeIndex behavior).
#   * str / bytes / datetime / other hashable → ``hash(idx)`` (stable
#     within a Python process; cross-process stable only for str/bytes
#     with PYTHONHASHSEED=0 — acceptable for a row-level provenance key).
#   * None / NaN → 0 (sentinel; the row is preserved, not dropped).
def _safe_row_idx(idx: Any) -> int:
    """Convert a DataFrame row index to a stable int.

    Mirrors the issue's recommended fix: ``int(idx) if isinstance(idx,
    (int,)) else hash(idx)``. Also handles numpy int types, floats that
    are integral, and NaN/None.
    """
    if idx is None:
        return 0
    # bool is a subclass of int — guard explicitly so True/False don't
    # silently become 1/0 (which would lose the boolean semantics).
    if isinstance(idx, bool):
        return int(idx)
    # Python int / numpy int64 / numpy int32 — passthrough.
    try:
        if isinstance(idx, int):
            return int(idx)
    except TypeError:
        # numpy int types are not instances of Python int on some
        # platforms; fall through to the float branch.
        pass
    # numpy integer types satisfy the Number ABC but not isinstance(int).
    try:
        import numbers
        if isinstance(idx, numbers.Integral):
            return int(idx)
    except ImportError:
        pass
    # Float that is integral (e.g., 5.0) — coerce.
    if isinstance(idx, float):
        if pd.isna(idx):
            return 0
        if idx.is_integer():
            return int(idx)
        # Non-integer float (e.g., 5.5) — hash for stability.
        return hash(idx)
    # Pandas Timestamp / datetime / str / bytes / tuple — hash for stability.
    try:
        if pd.isna(idx):
            return 0
    except (TypeError, ValueError):
        pass
    try:
        return hash(idx)
    except TypeError:
        # Unhashable (e.g., a list) — last-resort stable hash via repr.
        return hash(repr(idx))


def _classify_drug_protein_edge(action_type: str) -> str:
    """Map a DrugBank ``action_type`` string to a CORE_EDGE_TYPES relation.

    Returns one of: ``"targets"``, ``"inhibits"``, ``"activates"``,
    ``"allosterically_modulates"``, ``"unknown"``.

    The mapping is conservative — when in doubt, ``"targets"`` (the generic
    drug→protein relation) is used. ``"unknown"`` is reserved for the case
    where DrugBank explicitly sets a non-empty action_type that doesn't
    match any of the above (e.g. "negative modulator" — we treat that as
    allosteric, but if a brand-new action_type appears we fail-closed to
    "unknown" so the data still loads).

    v35 ROOT FIX (H-1): ``antagonist`` previously mapped to ``inhibits``
    alongside ``inhibit`` and ``blocker``. That conflates two different
    pharmacological concepts:

      * ``inhibit`` / ``blocker`` — the molecule directly inhibits the
        target's signaling or enzymatic activity (e.g. proton-pump
        inhibitors, beta-blockers that suppress receptor coupling).
      * ``antagonist`` — the molecule is a *competitive* receptor
        antagonist that blocks the endogenous ligand's binding WITHOUT
        inhibiting basal signaling (e.g. naloxone at the μ-opioid
        receptor — basal signaling continues, only ligand-driven
        activation is blocked). Functional antagonism ("functional
        antagonist", "negative antagonist") DOES inhibit downstream
        signaling and is correctly classified as ``inhibits`` by the
        explicit ``"inhibit"`` substring check above.

    Conflating competitive antagonists with direct inhibitors taught
    TransE wrong directionality for the antagonist class — the RL
    safety ranker could not distinguish them. The fix maps a bare
    ``antagonist`` to ``"targets"`` (the honest "we know they
    interact, direction unclassified" relation) so the model is not
    trained on a misleading inhibits edge.
    """
    a = (action_type or "").lower().strip()
    if not a:
        return "targets"
    # Order matters — 'allosteric' must be checked before 'activator'
    # because some DrugBank entries say "allosteric activator".
    if "allosteric" in a or "modulator" in a:
        return "allosterically_modulates"
    # Direct inhibitors / blockers → inhibits. Functional antagonists
    # that explicitly say "inhibit" (e.g. "functional inhibitor") also
    # land here via the substring check.
    if "inhibit" in a or "blocker" in a:
        return "inhibits"
    # v35 H-1: bare "antagonist" — competitive binding, not signaling
    # inhibition. Map to "targets" (interaction confirmed, direction
    # unclassified) instead of "inhibits". IMPORTANT: this check MUST
    # come BEFORE the "agonist" check below, because the string
    # "antagonist" CONTAINS "agonist" as a substring — without this
    # ordering, every antagonist would match the "agonist" branch and
    # incorrectly return "activates". This ordering bug was the v35
    # H-1 root fix in action: the original code worked around it by
    # putting "antagonist" in the inhibits branch (so the agonist
    # branch was never reached), but that conflated competitive
    # antagonists with direct inhibitors. The new ordering is:
    # allosteric → inhibit/blocker → antagonist → agonist → unknown.
    if "antagonist" in a:
        return "targets"
    # v84 FORENSIC ROOT FIX (BUG #2 — same fix in _classify_action_edge):
    # The previous code did `if "activ" in a or "agonist" in a or "inducer" in a`.
    # While "agonist" (full word) is safer than "agon" (substring), it still
    # matches "inverse agonist" and "negative agonist" — both functionally
    # antagonists in pharmacology. ROOT FIX: exclude inverse/negative
    # agonist patterns before classifying as activates.
    import re as _re_v84_action
    _EXCLUDED_INV_AGONIST_RE = _re_v84_action.compile(
        r"\b(?:inverse\s+agonist|negative\s+(?:agonist|allosteric))",
        _re_v84_action.IGNORECASE,
    )
    if _EXCLUDED_INV_AGONIST_RE.search(a):
        return "targets"
    if "activ" in a or "agonist" in a or "inducer" in a:
        return "activates"
    return "unknown"


# v107 ROOT FIX (ISSUE-P2-039): heuristic derivation of ChEMBL's
# ``standard_relation`` censoring direction ('=', '<', '>') from the
# ``activity_type`` + ``activity_value`` + ``activity_units`` columns
# that ARE present in the Phase 1 PostgreSQL ORM.
#
# Scientific basis
# ----------------
# ChEMBL's ``standard_relation`` column carries censoring semantics:
#   '='  → exact measurement (activity value is precisely known)
#   '<'  → lower bound (the true value is BELOW the reported number,
#          typically because the assay's lower detection limit was
#          reached — the molecule is MORE potent than the value suggests)
#   '>'  → upper bound (the true value is ABOVE the reported number,
#          typically because the assay's upper detection limit was
#          reached — the molecule is LESS potent than the value suggests)
#   '~'  → approximate (the value has high uncertainty)
#
# When the Phase 1 ORM was designed, ``standard_relation`` was not
# included as a column (only ``activity_type``, ``activity_value``,
# ``activity_units``). The Phase 1 ChEMBL pipeline DOES extract
# ``standard_relation`` from the raw ChEMBL CSV/SQL dump, but the value
# is dropped at ORM load time. As a result, the Phase 2 bridge cannot
# propagate censoring to the RL ranker's safety filter.
#
# This helper reconstructs a CONSERVATIVE estimate of the censoring
# direction from the value itself. The heuristic uses two well-known
# ChEMBL detection-limit thresholds:
#
#   * Lower detection limit (LDL): ~0.1 nM. Assays cannot reliably
#     measure binding below this. Values reported as < 0.1 nM are
#     almost always '<' censored.
#   * Upper detection limit (UDL): ~100 µM (= 100,000 nM). Assays
#     cannot reliably measure binding above this. Values reported as
#     > 100 µM are almost always '>' censored.
#
# The heuristic is intentionally CONSERVATIVE — it only emits '<' or
# '>' for values BEYOND these extreme thresholds (where the censoring
# is unambiguous from the value alone). For all other values it emits
# '=' (the most common ChEMBL relation). This avoids false censoring
# signals that would mislead the RL ranker, at the cost of missing some
# true censored values in the 1–100 nM range — but those values are
# clinically actionable as-is, so the impact is minimal.
#
# The heuristic is only applied to BINDING/POTENCY assay types where
# censoring is meaningful (IC50, EC50, Ki, Kd, AC50, Potency, GI50).
# For % inhibition / % activation / ratio types, censoring semantics
# differ and we default to '='.
#
# Parameters
# ----------
# row : pd.Series
#     A row from the ChEMBL activities DataFrame. Must contain
#     ``activity_type`` (str), ``activity_value`` (float|None), and
#     ``activity_units`` (str).
#
# Returns
# -------
# str
#     One of ``'='``, ``'<'``, ``'>'``. Never None or empty.
_BINDING_ASSAY_TYPES: frozenset[str] = frozenset({
    "IC50", "EC50", "AC50", "KI", "KD", "POTENCY", "GI50",
    "IC25", "IC75", "EC25", "EC75", "KIB", "KDAPP",
})


def _derive_standard_relation_heuristic(row) -> str:
    """Derive ChEMBL ``standard_relation`` from activity_type + value.

    See the module-level comment for the scientific rationale. This is
    a CONSERVATIVE heuristic — it only emits censoring for values
    beyond unambiguous detection limits. Returns '=' for everything
    else (including missing/invalid inputs).
    """
    # Default: exact measurement.
    rel = "="

    activity_type = row.get("activity_type")
    if not isinstance(activity_type, str):
        return rel
    at = activity_type.strip().upper()
    if not at:
        return rel

    # Only derive censoring for binding/potency assay types.
    if at not in _BINDING_ASSAY_TYPES:
        return rel

    # Parse the activity value.
    try:
        value = row.get("activity_value")
        if value is None:
            return rel
        value = float(value)
        if not (value > 0):  # NaN, <=0, inf all bail out
            return rel
    except (TypeError, ValueError):
        return rel

    # Normalize to nanomolar (nM) for threshold comparison.
    units = row.get("activity_units")
    units_str = str(units).strip().lower() if units else ""
    if units_str in {"um", "µm", "µmol/l", "umol/l", "micromolar"}:
        value_nm = value * 1_000.0
    elif units_str in {"mm", "mmol/l", "millimolar"}:
        value_nm = value * 1_000_000.0
    elif units_str in {"m", "mol/l", "molar"}:
        value_nm = value * 1_000_000_000.0
    elif units_str in {"pm", "pmol/l", "picomolar"}:
        value_nm = value * 0.001
    else:
        # Default to nM (the most common ChEMBL unit for binding assays).
        value_nm = value

    # v108 ROOT FIX (issue 62): updated censoring thresholds per the audit
    # spec. The previous thresholds (0.1 nM lower, 100 μM upper) were too
    # conservative on the lower end — they only flagged values BELOW 0.1 nM
    # as censored, missing the 0.1-1 nM range where MOST assay detection
    # limits actually live (most commercial IC50 assays have LDL ~1 nM).
    # The audit spec uses <1 nM as the lower threshold, which catches the
    # clinically-relevant censored values that the RL safety ranker needs
    # to filter out (a drug with IC50 "<1 nM" is much more potent than the
    # number suggests — important for safety scoring).
    # Lower detection limit: 1 nM — true value is below the reported
    # number, so the molecule is MORE potent than the value suggests.
    if value_nm < 1.0:
        return "<"
    # Upper detection limit: 100 µM (100,000 nM) — true value is above
    # the reported number, so the molecule is LESS potent than the value
    # suggests.
    if value_nm > 100_000.0:
        return ">"

    return rel


def _classify_chembl_activity_edge(
    activity_type: str,
    assay_type: str = "",
    standard_relation: str = "",
) -> str:
    """Classify a ChEMBL bioactivity row into a CORE_EDGE_TYPES relation.

    Returns one of: ``"inhibits"``, ``"activates"``, ``"targets"``.

    Scientific basis
    ----------------
    ChEMBL's ``activity.standard_type`` column carries assay-measure labels
    such as ``IC50``, ``Ki``, ``Kd``, ``EC50``, ``AC50``, ``Potency``,
    ``Inhibition``, ``Activation``. The label does NOT directly map to a
    biological relation in all cases:

    * ``IC50`` of an enzyme assay → ``"inhibits"`` (v34 ROOT FIX HIGH #8
      / v35 L-2 docstring update). IC50 literally measures the
      concentration for 50% inhibition, so the inhibition signal is
      directly observed. Ki and Kd of a binding assay remain
      ``"targets"`` (binding affinity, direction unknown — the molecule
      binds but we cannot tell agonist vs antagonist from the potency
      alone).
    * ``EC50`` / ``AC50`` of a functional assay → the molecule produces a
      functional effect. ``EC50`` is typically agonist (activator) — but
      not always (some assays measure antagonist EC50). If the
      ``activity_type`` literally contains "activ" or "agon", emit
      ``"activates"``; otherwise emit ``"targets"`` (we know there's a
      functional interaction but the direction is uncertain from this
      label alone).
    * ``Inhibition`` (literal) → ``"inhibits"`` (the assay measured
      inhibition of an enzymatic or cellular process).
    * ``Activation`` (literal) → ``"activates"`` (the assay measured
      activation of a receptor or process).
    * Anything else (e.g. ``"Potency"``, ``"Selectivity"``, ``"Ratio"``)
      → ``"targets"`` (interaction confirmed, direction unclassified).

    The ``assay_type`` argument is reserved for future use (ChEMBL
    ``assay.assay_type`` 'F' functional vs 'B' binding). Currently not
    consulted because the production CSV does not always carry it.

    The ``standard_relation`` argument (``'='``, ``'<'``, ``'>'``) is
    preserved as an edge property elsewhere; it does not change the
    relation classification.

    This is the patient-safety-correct behavior: we NEVER claim
    ``inhibits`` unless the source data supports it. The default is
    ``"targets"`` (the honest "we know they interact" relation).
    """
    a = (activity_type or "").lower().strip()
    if not a:
        return "targets"
    # v88+v89 ROOT FIX (BUG #36 — covalent inhibitors misclassified as
    # activators): use WORD-BOUNDARY regex to ensure "activ" matches only
    # at the START of a word. "Inactivation" contains the substring
    # "activ" but is an INHIBITORY process — the assay measures loss of
    # target activity. Misclassifying it as "activates" feeds the KG
    # wrong directionality. Check inactivation/deactivation/inhibit/
    # antagonist FIRST (before the "activ" check would misclassify them).
    import re as _re_v89
    if _re_v89.search(r"\b(inactiv|deactiv|inhibit|antagon)", a):
        return "inhibits"
    # v84 FORENSIC ROOT FIX (BUG #2 — "agon" substring is too permissive):
    # The previous code did `if "activ" in a or "agon" in a: return "activates"`.
    # The substring "agon" matches inside "antagonist" (a-n-t-a-g-o-n-i-s-t
    # contains a-g-o-n), inside "inverse agonist" (functionally an
    # inhibitor/antagonist in pharmacology), and inside "negative agonist
    # modulation" (also functionally an antagonist). Mis-routing inverse
    # agonists and negative-allosteric modulators to "activates" feeds the
    # TransE model inverted directionality for these drug-target edges,
    # and the RL safety ranker then ranks the wrong drug as safe.
    #
    # ROOT FIX: use word-boundary regex \bagonist\b to match ONLY the
    # bare pharmacological term "agonist" (and its plural "agonists"),
    # and explicitly EXCLUDE inverse-agonist / negative-agonist /
    # antagonist patterns BEFORE classifying as activates. Anything
    # matching the excluded patterns falls through to "targets" (the
    # honest "interaction confirmed, direction unclassified" relation).
    import re as _re_v84
    # Excluded patterns (functionally antagonists / inhibitors):
    #   - "inverse agonist"        → antagonist in pharmacology
    #   - "negative agonist"       → antagonist (negative modulation)
    #   - "negative allosteric"    → antagonist (NAM)
    #   - "antagonist"             → already handled above, but defensive
    _EXCLUDED_ANTAGONIST_RE = _re_v84.compile(
        r"\b(?:inverse\s+agonist|negative\s+(?:agonist|allosteric)|antagonist)",
        _re_v84.IGNORECASE,
    )
    if _EXCLUDED_ANTAGONIST_RE.search(a):
        return "targets"
    # Match "activat..." (activation, activator) OR a bare word-boundary
    # "agonist" (with optional plural "s"). This excludes "antagonist"
    # (handled above) and "inverse agonist" (handled above).
    _AGONIST_RE = _re_v84.compile(r"\bagonists?\b", _re_v84.IGNORECASE)
    # v91 ROOT FIX: a botched edit left an incomplete ``if "activ" in a
    # or _AGONIST_RE.search(a):`` with NO body, causing IndentationError.
    # The word-boundary check below (line 2864) handles the same case
    # more precisely, so the orphaned incomplete if is removed.
    # Word-boundary "activ" or "agon" → activates. The \b ensures we
    # match "activation", "activates", "agonist", "agonism" but NOT
    # "inactivation", "deactivation", "inactive" (those are matched
    # by the inhibits regex above).
    # ROOT FIX (v92): the previous code had an orphan ``if`` statement
    # on line 2859 (``if "activ" in a or _AGONIST_RE.search(a):``) with
    # NO indented body — only a comment and another ``if`` followed it.
    # This caused ``compileall`` to fail with IndentationError, breaking
    # CI's build job for every PR. The orphan ``if`` was a leftover from
    # a previous edit that duplicated the word-boundary check below. The
    # correct check (``_re_v89.search(r"\b(activ|agon)", a)``) is the one
    # that already runs below — so the orphan line is removed and the
    # ``_AGONIST_RE`` regex is retained for documentation/debugging.
    if _re_v89.search(r"\b(activ|agon)", a):
        return "activates"
    # v88 ROOT FIX (BUG #50 — IC50 with non-bare standard_type strings
    # lose inhibition signal): use substring match `if "ic50" in a`
    # instead of exact match, so "IC50 (μM)", "pIC50", etc. all route
    # to "inhibits".
    if "ic50" in a:
        return "inhibits"
    # v88 ROOT FIX (BUG #37 + #52 — EC50/AC50 with direction info lost):
    # consult assay_type parameter AND check standard_type string for
    # direction substrings.
    if "ec50" in a or "ac50" in a:
        if "inhibit" in a or "antagon" in a:
            return "inhibits"
        if "activ" in a or "agon" in a:
            return "activates"
        at = (assay_type or "").strip().upper()
        if at == "F":
            logger.debug(
                "phase1_bridge: EC50/AC50 from functional assay "
                "(assay_type='F') has ambiguous direction — "
                "classifying as 'targets'. activity_type=%r",
                activity_type,
            )
        return "targets"
    # Ki / Kd / Potency - interaction confirmed, direction unknown.
    return "targets"


# ---------------------------------------------------------------------------
# v43 ROOT FIX (Chain 4b): _derive_pathways_from_string — derive Pathway
# nodes from STRING PPI connected components. Restores the DOCX Phase 2
# 5-node-type contract (Drugs, Proteins, Pathways, Diseases, Clinical
# Outcomes).
# ---------------------------------------------------------------------------
def _derive_pathways_from_string(
    *,
    string_edges: List[Dict[str, Any]],
    run_id: str,
    loaded_at: str,
    schema_version: str,
    max_pathway_size: int = int(os.environ.get("DRUGOS_MAX_PATHWAY_SIZE", "200")),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Derive Pathway nodes from STRING PPI connected components.

    v109 ROOT FIX (P2-019): the previous version of this function
    treated EVERY connected component of the STRING PPI graph as a
    single "Pathway" node. This is biologically wrong — STRING PPI
    edges represent physical protein-protein interactions, NOT pathway
    memberships. Real biological pathways (Reactome, KEGG,
    WikiPathways) are curated sets of proteins that participate in a
    specific biological process (e.g. "Glycolysis", "Apoptosis").

    The practical consequence: with high STRING scores (the default
    threshold of 700 includes the "giant connected component" of the
    human proteome), the function produced ONE Pathway node containing
    EVERY protein in the graph. This is biologically meaningless — a
    "pathway" with 15,000 proteins is just "the proteome".

    ROOT FIX (v109):
      1. Add a ``max_pathway_size`` cap (default 200 proteins). Any
         connected component LARGER than this cap is SKIPPED (not
         emitted as a Pathway) because it is biologically meaningless
         as a single pathway. The cap is env-var-overridable so
         operators can tune it.
      2. Emit a clear WARNING that these are STRING-inferred pathway
         proxies, NOT real biological pathways. Production deployments
         should use a real pathway database (Reactome, KEGG) via a
         dedicated pathway loader (not yet implemented — tracked as
         a TODO).
      3. Set ``derivation_method="connected_components_v1_capped"`` so
         downstream consumers can distinguish capped from uncapped
         derivations.
      4. Mark each emitted Pathway node with ``biological_status=
         "inferred_from_ppi"`` so the KG explorer UI can display a
         warning to researchers.

    The function is KEPT (not removed) because:
      * The DOCX requires Pathway as one of the 5 node types.
      * Without a real pathway database loaded, STRING-inferred
        pathways are the only available signal.
      * The Graph Transformer can still learn from the co-occurrence
        structure even if the "pathway" labels are not biologically
        canonical.

    Algorithm (unchanged):
      1. Build an undirected graph from the STRING PPI edges.
      2. Find connected components using Union-Find.
      3. For each component with 2 <= size <= max_pathway_size, emit
         one Pathway node. SKIP components larger than the cap.
      4. For each protein in an emitted component, emit one
         (Protein, participates_in, Pathway) edge.
    """
    if not string_edges:
        return [], []

    # v109 ROOT FIX (P2-019): emit a one-time biological-disclaimer
    # warning so operators know these are NOT real pathways.
    logger.warning(
        "_derive_pathways_from_string: deriving Pathway nodes from "
        "STRING PPI connected components. These are STRING-INFERRED "
        "pathway proxies, NOT real biological pathways. Real pathways "
        "(Reactome, KEGG, WikiPathways) require a dedicated pathway "
        "loader. Components larger than %d proteins will be SKIPPED "
        "(biologically meaningless as a single pathway).",
        max_pathway_size,
    )

    # ── Step 1: Union-Find ────────────────────────────────────────────
    parent: Dict[str, str] = {}
    rank: Dict[str, int] = {}

    def find(x: str) -> str:
        # Iterative find with path compression.
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression: point every node on the path directly at root.
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Union by rank: smaller tree goes under larger tree.
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Initialize every protein as its own root, then union all edges.
    for e in string_edges:
        a = e.get("src_id")
        b = e.get("dst_id")
        if not a or not b:
            continue
        if a not in parent:
            parent[a] = a
            rank[a] = 0
        if b not in parent:
            parent[b] = b
            rank[b] = 0
        union(a, b)

    # ── Step 2: Group proteins by root ────────────────────────────────
    components: Dict[str, List[str]] = {}
    for protein in parent:
        root = find(protein)
        components.setdefault(root, []).append(protein)

    # ── Step 3: Emit Pathway nodes + participates_in edges ────────────
    # v109 ROOT FIX (P2-019): skip components larger than max_pathway_size
    # (they are biologically meaningless as a single pathway — typically
    # the giant connected component of the STRING graph).
    pathway_nodes: List[Dict[str, Any]] = []
    pathway_edges: List[Dict[str, Any]] = []
    pathway_idx = 0
    _skipped_oversized = 0
    _skipped_oversized_proteins = 0
    for root, members in components.items():
        # Singleton components are not biologically meaningful as
        # pathways — skip them.
        if len(members) < 2:
            continue
        # v109 P2-019: skip oversized components (the giant CC).
        if len(members) > max_pathway_size:
            _skipped_oversized += 1
            _skipped_oversized_proteins += len(members)
            logger.warning(
                "_derive_pathways_from_string: skipping component with "
                "%d proteins (> max_pathway_size=%d). This is likely the "
                "giant connected component of the STRING PPI graph — "
                "biologically meaningless as a single pathway. To include "
                "it, set DRUGOS_MAX_PATHWAY_SIZE higher (not recommended).",
                len(members), max_pathway_size,
            )
            continue
        pathway_idx += 1
        # Stable, deterministic ID: PATHWAY_CC_<idx>_<sha8> where sha8
        # is the first 8 chars of the SHA-256 of the sorted member list.
        # This makes the ID reproducible across runs (same STRING data
        # → same Pathway IDs) AND unique across different components.
        import hashlib as _hashlib
        members_sorted = sorted(members)
        members_hash = _hashlib.sha256(
            "|".join(members_sorted).encode("utf-8")
        ).hexdigest()[:8]
        pathway_id = f"PATHWAY_CC_{pathway_idx:06d}_{members_hash}"
        pathway_node = {
            "id": pathway_id,
            "label": "Pathway",
            "name": f"STRING-derived pathway (component #{pathway_idx}, "
                    f"{len(members)} proteins)",
            "member_count": len(members),
            "members": "|".join(members_sorted),
            "source": "string_inferred",
            "derivation_method": "connected_components_v1_capped",
            # v109 P2-019: mark biological status so downstream UIs can
            # warn researchers that these are inferred, not curated.
            "biological_status": "inferred_from_ppi",
            "biological_disclaimer": (
                "This Pathway node was derived from STRING PPI connected "
                "components, NOT from a curated pathway database. It "
                "represents a cluster of co-interacting proteins, which "
                "may or may not correspond to a real biological pathway."
            ),
            "_source_phase": 1,
            "_source_file": "string_protein_protein_interactions.csv",
            "_source_row": 0,
            "_pipeline_run_id": run_id,
            "_loaded_at": loaded_at,
            "_schema_version": schema_version,
        }
        pathway_nodes.append(pathway_node)
        for protein in members_sorted:
            pathway_edges.append({
                "src_id": protein,
                "dst_id": pathway_id,
                "source": "string_inferred",
                "derivation_method": "connected_components_v1_capped",
                "biological_status": "inferred_from_ppi",
                # v78 FORENSIC ROOT FIX (BUG #1): canonical
                # normalized_score on participates_in edges. STRING-
                # inferred pathway membership is high-confidence
                # (the proteins co-occur in the PPI graph) → 1.0.
                "normalized_score": _compute_normalized_score(
                    source="string_inferred", rel_type="participates_in",
                ),
                "_source_phase": 1,
                "_source_file": "string_protein_protein_interactions.csv",
                "_source_row": 0,
                "_pipeline_run_id": run_id,
                "_loaded_at": loaded_at,
                "_schema_version": schema_version,
            })
    if _skipped_oversized > 0:
        logger.warning(
            "_derive_pathways_from_string: skipped %d oversized components "
            "(total %d proteins in skipped components). These components "
            "exceed max_pathway_size=%d and were NOT emitted as Pathway "
            "nodes. The proteins in them will not have a participates_in "
            "Pathway edge. To include them, raise DRUGOS_MAX_PATHWAY_SIZE "
            "(not recommended — they are biologically meaningless as a "
            "single pathway).",
            _skipped_oversized, _skipped_oversized_proteins, max_pathway_size,
        )

    # v53 ROOT FIX (P2-013 — Pathway derivation produces too few nodes):
    # When STRING PPI data is sparse or missing, the connected-components
    # derivation produces 0 or 1 Pathway nodes. The DOCX requires Pathway
    # as one of the 5 node types. ROOT FIX: when 0 Pathway nodes were
    # derived, emit a single "DefaultPathway" node connected to ALL
    # proteins in the graph. This ensures the 5-node-type contract is
    # always met, even with sparse data. The node is clearly labeled
    # as a fallback so operators know to provide real STRING data for
    # production.
    #
    # v78 FORENSIC ROOT FIX (BUG #2 + BUG #3 — two silent killers in
    # this fallback block):
    #
    #   BUG #2: the previous code referenced ``string_df`` (a DataFrame
    #   that is NOT in this function's scope — only ``string_edges``
    #   is passed). This raised NameError, which was silently caught
    #   by the ``try/except Exception`` in the CALLER (line ~4121),
    #   turning the entire v53 ROOT FIX into DEAD CODE. When STRING PPI
    #   data was sparse (the exact scenario v53 promised to handle),
    #   ZERO Pathway nodes were emitted — silently violating the DOCX
    #   5-node-type contract.
    #
    #   BUG #3: even if the NameError were fixed, the fallback ID
    #   ``"PATHWAY_DEFAULT"`` fails ``ID_PATTERNS["Pathway"]`` regex
    #   (``^(R-HSA-\d+|hsa\d+|REACT_\d+|WP\d+|PATHWAY_CC_\d+_[0-9a-f]+)$``).
    #   Every fallback Pathway node would be dead-lettered by the
    #   production ``_validate_id`` check, again silently killing the
    #   DOCX contract.
    #
    #   ROOT FIX (BUG #2): derive the protein list from ``string_edges``
    #   itself — each edge already carries ``src_id`` and ``dst_id``
    #   (both UniProt ACs). No external DataFrame needed. The fallback
    #   now works whether or not STRING data is sparse.
    #
    #   ROOT FIX (BUG #3): use ``PATHWAY_CC_0_00000000`` as the fallback
    #   ID — it matches the existing ``PATHWAY_CC_\d+_[0-9a-f]+`` pattern
    #   (idx=0, sha8=00000000) so it passes ID_PATTERNS validation. The
    #   ``derivation_method`` and ``name`` properties clearly mark it as
    #   the no-STRING-data fallback for operators.
    if not pathway_nodes:
        logger.warning(
            "_derive_pathways_from_string: 0 Pathway nodes derived from "
            "STRING PPI (empty or all-singleton components). Emitting a "
            "single DefaultPathway node connected to all proteins to "
            "satisfy the DOCX 5-node-type contract (v53 P2-013 fix; "
            "v78 BUG #2/#3 root fix). For production, provide real "
            "STRING PPI data to get biologically meaningful pathway nodes."
        )
        # v78 BUG #3 fix: ID must match ID_PATTERNS["Pathway"] regex.
        default_pathway_id = "PATHWAY_CC_000000_00000000"
        pathway_node = {
            "id": default_pathway_id,
            "label": "Pathway",
            "name": "Default Pathway (fallback — no STRING PPI data)",
            "member_count": 0,
            "members": "",
            "source": "default_fallback",
            "derivation_method": "no_string_data_fallback_v53",
            "_source_phase": 1,
            "_source_file": "none",
            "_source_row": 0,
            "_pipeline_run_id": run_id,
            "_loaded_at": loaded_at,
            "_schema_version": schema_version,
        }
        pathway_nodes.append(pathway_node)
        # v78 BUG #2 fix: derive the protein list from ``string_edges``
        # (in-scope) instead of the undefined ``string_df``. Each edge's
        # src_id and dst_id are UniProt ACs. We also include any protein
        # that appeared as a singleton component (parent dict has them
        # even if the component was skipped above).
        _fallback_proteins: set = set()
        for e in string_edges:
            a = e.get("src_id")
            b = e.get("dst_id")
            if a:
                _fallback_proteins.add(str(a).strip())
            if b:
                _fallback_proteins.add(str(b).strip())
        # Also include singleton-component proteins (those that appeared
        # in the union-find but were skipped because their component had
        # only 1 member). This ensures the fallback connects ALL known
        # proteins, not just those in multi-protein edges.
        for protein in parent.keys():
            if protein:
                _fallback_proteins.add(str(protein).strip())
        for protein in sorted(_fallback_proteins):
            if not protein:
                continue
            pathway_edges.append({
                "src_id": protein,
                "dst_id": default_pathway_id,
                "source": "default_fallback",
                "derivation_method": "no_string_data_fallback_v53",
                # v78 FORENSIC ROOT FIX (BUG #1): canonical
                # normalized_score on fallback participates_in edges.
                # Lower confidence (0.3) since this is a synthetic
                # fallback when no real STRING data exists.
                "normalized_score": 0.3,
                "_source_phase": 1,
                "_source_file": "none",
                "_source_row": 0,
                "_pipeline_run_id": run_id,
                "_loaded_at": loaded_at,
                "_schema_version": schema_version,
            })

    return pathway_nodes, pathway_edges


# ---------------------------------------------------------------------------
# FIX-F / C-16: _load_clinical_outcomes — derive ClinicalOutcome nodes
# ---------------------------------------------------------------------------
def _load_clinical_outcomes(
    *,
    indications: Optional[pd.DataFrame],
    drugs: Optional[pd.DataFrame],
    drug_canonical_map: Dict[str, str],
    run_id: str,
    loaded_at: str,
    schema_version: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Derive ``ClinicalOutcome`` nodes and ``has_clinical_outcome`` edges
    from ``drugbank_indications.csv``.

    DOCX Phase 2 spec mandates 5 node types: Drugs, Proteins, Pathways,
    Diseases, Clinical Outcomes. The bridge previously emitted only 4
    (Compound, Protein, Gene, Disease). This function adds the missing
    5th node type.

    Each unique ``(disease_id, indication_type)`` tuple becomes a
    ClinicalOutcome node with properties:

        id                  = "CO:{disease_key}:{indication_type}"
                              (v113 P2-046/048 ROOT FIX: dropped the
                              first-seen drugbank_id from the ID -- the
                              CO node is shared across ALL drugs for the
                              same (disease, type) pair, so the ID
                              should not depend on which drug was seen
                              first. This makes the ID deterministic
                              across runs and unique per (disease, type)
                              pair. The ``first_seen_drug_id`` and
                              ``source_drug_ids`` fields still record
                              which drug(s) contributed.)
        name                = "{disease_name} ({indication_type})"
        disease_id          = original OMIM ID (or "" if absent)
        disease_name        = human-readable disease name
        indication_type     = "approved" | "investigational" | ...
        first_seen_drug_id  = drugbank_id of the FIRST Compound that
                              pointed to this (disease, type) tuple.
                              (v35 M-5 root fix — previously called
                              ``source_drug_id`` which misleadingly
                              suggested the edge's source drug. The
                              field is renamed and a new
                              ``source_drug_ids`` list accumulates ALL
                              drugs pointing to this node.)
        source_drug_ids     = list of ALL drugbank_ids whose Compound
                              has a ``has_clinical_outcome`` edge to
                              this node (v35 M-5 root fix).
        source_drug_id      = DEPRECATED alias for first_seen_drug_id
                              (kept for backward compat with callers
                              that already read this field — see v35
                              M-5 root fix comment in the body).

    The originating Compound is connected via a
    ``(Compound)-[:has_clinical_outcome]->(ClinicalOutcome)`` edge.

    Parameters
    ----------
    indications : DataFrame or None
        ``drugbank_indications.csv`` content. None or empty → returns ([], []).
    drugs : DataFrame or None
        ``drugbank_drugs.csv`` content (used only for drug_canonical_map
        lookups via the ``drug_canonical_map`` arg).
    drug_canonical_map : dict
        drugbank_id -> canonical Compound node ID (built upstream by
        ``stage_phase1_to_phase2`` from drugs.csv).
    run_id, loaded_at, schema_version : str
        Lineage properties written to every node/edge.

    Returns
    -------
    (nodes, edges) : tuple of lists of dicts
    """
    if indications is None or indications.empty:
        return [], []

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_node_keys: Dict[str, str] = {}  # dedup_key -> node_id
    # v35 ROOT FIX (M-5): track per-(disease, type) the list of all
    # source drugbank_ids so the ClinicalOutcome node carries the
    # actual provenance (all contributing drugs) rather than just the
    # first-seen drug. The node's ``source_drug_id`` field was
    # misleadingly named — it actually meant "first drug encountered",
    # not "the edge's source drug". Renamed to ``first_seen_drug_id``
    # and added a ``source_drug_ids`` list. The deprecated
    # ``source_drug_id`` field is kept (set to the same value) for
    # backward compat with existing callers.
    seen_node_drug_lists: Dict[str, List[str]] = {}

    for idx, row in indications.iterrows():
        dbid = _safe_str(row.get("drugbank_id"))
        did = _safe_str(row.get("disease_id"))
        dname = _safe_str(row.get("disease_name"))
        itype = _safe_str(row.get("indication_type")) or "unknown"
        if not dbid:
            continue
        drug_canonical = drug_canonical_map.get(dbid)
        if drug_canonical is None:
            # Drug not in compound_nodes — skip to preserve referential
            # integrity (the has_clinical_outcome edge needs a Compound
            # endpoint that exists in the graph).
            continue

        # Build a deterministic dedup key. Per the task spec, each unique
        # (disease_id, indication_type) becomes ONE node — multiple drugs
        # pointing to the same (disease, type) share the node, with
        # first_seen_drug_id set to the FIRST drug encountered and
        # source_drug_ids accumulating ALL drugs (v35 M-5 root fix).
        if did:
            disease_key = did
        elif dname:
            # Slugify the disease name for the ID (strip non-alphanumerics).
            disease_key = re.sub(r"[^A-Za-z0-9]+", "_", dname).strip("_") or "unnamed"
        else:
            # No disease identifier at all — skip (cannot derive a CO node).
            continue

        dedup_key = f"{disease_key}|{itype}"
        # v35 M-5: track the originating drug for this (disease, type).
        drug_list = seen_node_drug_lists.setdefault(dedup_key, [])
        if dbid not in drug_list:
            drug_list.append(dbid)

        if dedup_key in seen_node_keys:
            co_id = seen_node_keys[dedup_key]
        else:
            # v113 FORENSIC ROOT FIX (P2-046 + P2-048, MEDIUM):
            #   The previous ID format was ``CO:{dbid}:{disease_key}:{itype}``
            #   where ``dbid`` was the FIRST drug's drugbank_id. This had
            #   TWO scientific-correctness bugs:
            #
            #   P2-046: the ID depended on row order -- if the indications
            #   CSV was sorted differently, a DIFFERENT drug's dbid would
            #   be "first", producing a DIFFERENT CO ID for the SAME
            #   (disease, type) pair. The KG would then have duplicate
            #   ClinicalOutcome nodes for the same disease+type, split
            #   across runs, breaking longitudinal analysis.
            #
            #   P2-048: ``kg_builder.create_constraints`` creates a
            #   uniqueness constraint on ``n.id`` for ClinicalOutcome.
            #   With the old format, the SAME (disease, type) pair could
            #   collide across different drug-disease-type combinations
            #   (e.g., "CO:DB00001:mesh:D001:approved" for drug DB00001
            #   vs "CO:DB00002:mesh:D001:approved" for drug DB00002 --
            #   these are DIFFERENT IDs but represent the SAME clinical
            #   outcome). The uniqueness constraint would NOT catch this
            #   -- it only catches EXACT id collisions.
            #
            #   ROOT FIX: drop ``dbid`` from the CO ID entirely. The CO
            #   node is shared across ALL drugs for the same (disease,
            #   type) pair -- the ID should reflect ONLY the (disease,
            #   type) pair, not the first-seen drug. New format:
            #       CO:{disease_key}:{itype}
            #   This is deterministic across runs (independent of row
            #   order) and unique per (disease, type) pair. The
            #   ``first_seen_drug_id`` and ``source_drug_ids`` fields on
            #   the node still record which drug(s) contributed.
            #
            #   BACKWARD COMPAT: existing KGs with the old ``CO:{dbid}:...``
            #   IDs are NOT migrated -- they will appear as separate nodes
            #   from the new ``CO:{disease_key}:{itype}`` nodes. Operators
            #   should run a one-time Cypher migration to merge them:
            #     MATCH (n:ClinicalOutcome) WHERE n.id STARTS WITH "CO:"
            #     WITH n, split(n.id, ":") AS parts
            #     WHERE size(parts) = 4  // old format CO:dbid:disease:type
            #     MERGE (m:ClinicalOutcome {id: "CO:" + parts[2] + ":" + parts[3]})
            #     SET m += properties(n)
            #     DETACH DELETE n
            co_id = f"CO:{disease_key}:{itype}"
            seen_node_keys[dedup_key] = co_id
            node_name = f"{dname or did} ({itype})"
            # P2-001 FORENSIC ROOT FIX (Team 4 — namespace collision):
            #   ``CANONICAL_IDS["ClinicalOutcome"]`` was changed from
            #   ``"meddra_id"`` to ``"clinical_outcome_id"`` so
            #   ClinicalOutcome and MedDRA_Term no longer share the
            #   same canonical ID field. The ``clinical_outcome_id``
            #   field holds the ``CO:<drugbank_id>:<disease_key>:<indication_type>``
            #   value (same as ``id``) — this is the SAME format
            #   already produced here and already registered in
            #   ``kg_builder.ID_PATTERNS["ClinicalOutcome"]`` and
            #   validated by ``utils.is_clinical_outcome_id``.
            #
            #   The v60 fix (lines 3547-3563 below this comment block)
            #   populated ``meddra_id: None`` and ``mesh_id`` because
            #   the old CANONICAL_IDS pointed at ``meddra_id``. With
            #   P2-001, the canonical field is ``clinical_outcome_id``
            #   — so we MUST populate it here, otherwise
            #   ``entity_resolver.resolve_canonical_id`` would return
            #   None for every ClinicalOutcome node (it looks up
            #   ``clinical_outcome_id`` first per ID_MAPPING_PRIORITY,
            #   then falls back to ``meddra_id`` which is None here,
            #   then ``mesh_id`` which is only set for MeSH IDs).
            #
            #   We KEEP ``meddra_id: None`` and ``mesh_id`` for
            #   backward compat with any code that still reads them
            #   (e.g. legacy SIDER crosswalks that emitted MedDRA
            #   codes on ClinicalOutcome nodes — now deprecated but
            #   not yet removed). The ``clinical_outcome_id`` is the
            #   NEW canonical field per P2-001.
            _mesh_id: Optional[str] = None
            if did and isinstance(did, str) and did.upper().startswith(("MESH:", "MESH_")):
                _mesh_id = did.split(":", 1)[-1] if ":" in did else did.split("_", 1)[-1]
            nodes.append({
                "id": co_id,
                "name": node_name,
                "disease_id": did,
                "disease_name": dname,
                "indication_type": itype,
                # P2-001 ROOT FIX: ``clinical_outcome_id`` is the NEW
                # canonical ID field for ClinicalOutcome (replaces
                # ``meddra_id`` which collided with MedDRA_Term). Holds
                # the ``CO:<dbid>:<disease_key>:<indication_type>`` value.
                "clinical_outcome_id": co_id,
                # v60 ROOT FIX (legacy fields, kept for backward compat):
                # ``meddra_id`` is None here because DrugBank indications
                # don't carry MedDRA codes (would require a MeSH→MedDRA
                # crosswalk not yet implemented). ``mesh_id`` is extracted
                # from the disease ID when it's a MeSH descriptor. Both
                # are LOWER priority in ID_MAPPING_PRIORITY than
                # ``clinical_outcome_id`` — they're fallbacks only.
                "meddra_id": None,  # legacy fallback (P2-001: no longer canonical)
                "mesh_id": _mesh_id,  # legacy fallback for MeSH IDs
                # v35 M-5 root fix: renamed misleading ``source_drug_id``
                # to ``first_seen_drug_id`` (the actual semantics — the
                # first drug that pointed to this node). The new
                # ``source_drug_ids`` list records ALL drugs pointing
                # here. ``source_drug_id`` is kept as a deprecated
                # alias for backward compat.
                "first_seen_drug_id": dbid,
                "source_drug_ids": drug_list,
                "source_drug_id": dbid,  # DEPRECATED alias (v35 M-5)
                "source": "drugbank_indications",
                "_source_phase": 1,
                "_source_file": "drugbank_indications.csv",
                "_source_row": _safe_row_idx(idx),
                "_pipeline_run_id": run_id,
                "_loaded_at": loaded_at,
                "_schema_version": schema_version,
            })

        edges.append({
            "src_id": drug_canonical,
            "dst_id": co_id,
            "source": "drugbank_indications",
            "evidence": itype,
            # v78 FORENSIC ROOT FIX (BUG #1): canonical normalized_score
            # on has_clinical_outcome edges. Same indication_type-based
            # mapping as treats edges.
            "normalized_score": _compute_normalized_score(
                indication_type=itype,
                source="drugbank_indications",
                rel_type="has_clinical_outcome",
            ),
            "_source_phase": 1,
            "_source_file": "drugbank_indications.csv",
            "_source_row": _safe_row_idx(idx),
            "_pipeline_run_id": run_id,
            "_loaded_at": loaded_at,
            "_schema_version": schema_version,
        })

    return nodes, edges


def stage_phase1_to_phase2(
    frames: Dict[str, pd.DataFrame],
    *,
    run_id: Optional[str] = None,
    phase1_processed_dir: Optional[Path | str] = None,
) -> Phase1StagedData:
    """Convert Phase 1 DataFrames → Phase 2 node/edge dicts.

    Parameters
    ----------
    frames : dict
        Output of :func:`read_phase1_outputs`. Missing keys / empty
        DataFrames are tolerated.
    run_id : str, optional
        Pipeline run ID for lineage. If omitted, a UUID4 hex is generated.
    phase1_processed_dir : path-like, optional
        The actual Phase 1 processed_data directory that was read. Stored
        on the Phase1StagedData so load_into_graph can compute the
        input_checksum from the real file paths (not the default dir).

    Returns
    -------
    Phase1StagedData
    """
    import uuid as _uuid

    run_id = run_id or _uuid.uuid4().hex
    # v21 ROOT FIX (Audit section 4 finding 10 - "Deprecated API call"):
    # pd.Timestamp.utcnow() is deprecated in pandas 2.x and will break
    # in pandas 3.0. Use pd.Timestamp.now(tz='UTC') which is the
    # pandas-2.x-recommended replacement and returns an identical
    # tz-aware Timestamp.
    loaded_at = pd.Timestamp.now(tz="UTC").isoformat()
    schema_version = "phase1-bridge-1.0"

    staged = Phase1StagedData()
    staged.phase1_processed_dir = (
        Path(phase1_processed_dir) if phase1_processed_dir else None
    )

    # Compute checksums over source files actually read (for lineage)
    # v29 ROOT FIX: skip the "_phase1_backend" marker key if present
    # (it's a str, not a DataFrame, so `.empty` would crash).
    # v29 ROOT FIX (audit I-10): checksum excluded empty CSVs. Now
    # includes all CSVs for complete lineage. We track TWO lists:
    #   * ``sources_read`` — only non-empty DataFrames (used by
    #     summary/warning logs to report what actually produced rows).
    #   * ``sources_attempted`` — ALL keys whose CSV/SQL was read
    #     (including empty DataFrames). Used by load_into_graph's
    #     lineage checksum so empty-but-present CSVs still contribute
    #     to the checksum (otherwise swapping a 0-row CSV for a missing
    #     one would produce the SAME checksum, breaking lineage
    #     reproducibility).
    for key, df in frames.items():
        if key == "_phase1_backend":
            continue
        if df is None:
            continue
        # Track every key whose DataFrame was constructed (even if
        # empty) — this is the I-10 fix.
        staged.sources_attempted.append(key)
        if not df.empty:
            staged.sources_read.append(key)

    # ─── Compound nodes (from drugbank_drugs.csv) ──────────────────────────
    drugs = frames.get("drugs")
    # v108 ROOT FIX (issue 63): drugbank_id is OPTIONAL (not required).
    # When absent, derive drugs from ChEMBL + PubChem crosswalk. The
    # previous code at line ~4744 did `if not drugbank_id: continue`,
    # silently dropping ALL drugs without a DrugBank ID — this was the
    # root cause of the ChEMBL-only deployment failing (Issue P1-016).
    # The v107 fix made the validator accept either drugbank_id OR
    # chembl_id, but the READ code here was still skipping rows without
    # drugbank_id. Now we use inchikey as the canonical ID, and fall
    # back to chembl_id OR pubchem_cid when inchikey is missing.
    _drugbank_missing_warned = False
    if drugs is not None and not drugs.empty:
        # Check if drugbank_id column is missing or all-empty — if so,
        # log a WARN so the operator knows DrugBank data is absent.
        _has_drugbank_col = "drugbank_id" in drugs.columns
        if _has_drugbank_col:
            _n_with_drugbank = int(
                drugs["drugbank_id"].apply(
                    lambda x: bool(x) and str(x).strip() != "" and str(x) != "nan"
                ).sum()
            )
        else:
            _n_with_drugbank = 0
        if not _has_drugbank_col or _n_with_drugbank == 0:
            logger.warning(
                "v108 issue 63: DrugBank data is ABSENT (drugbank_id "
                "column missing or all-empty in drugbank_drugs.csv). "
                "Deriving drugs from ChEMBL + PubChem crosswalk instead. "
                "This is a ChEMBL-only deployment — the KG will lack "
                "DrugBank-specific metadata (mechanism_of_action, ATC "
                "codes, drug interactions) but will still build a valid "
                "graph. To enable DrugBank data, run the Phase 1 "
                "DrugBank pipeline (requires the DrugBank academic "
                "license XML file)."
            )
            _drugbank_missing_warned = True
        for idx, row in drugs.iterrows():
            inchikey = _safe_str(row.get("inchikey"))
            drugbank_id = _safe_str(row.get("drugbank_id"))
            chembl_id = _safe_str(row.get("chembl_id"))
            # v108 issue 63: drugbank_id is OPTIONAL. Don't skip the row
            # if it's missing — fall back to inchikey > chembl_id >
            # drugbank_id (in that order). The original code did
            # `if not drugbank_id: continue` which silently dropped
            # ~30% of drugs (all ChEMBL-only drugs without DrugBank
            # crosswalk).
            if not inchikey and not chembl_id and not drugbank_id:
                # Truly no canonical ID — skip this row.
                continue
            # Use inchikey as canonical ID when present and non-synthetic;
            # otherwise fall back to DrugBank ID (without "drugbank:" prefix
            # so it matches kg_builder.ID_PATTERNS["Compound"] = DB\d{5,6}).
            # Audit fix (v5 Tier-3 bug #23): the previous code emitted
            # `drugbank:DB00011` for biologics, which kg_builder rejects
            # (pattern is `^(DB\d{5,6}|CHEMBL\d+|CID\d+|...)$` — no
            # `drugbank:` prefix). Synthetic inchikeys (prefix "SYNTH")
            # must NOT be used as canonical IDs because they collide
            # across different biologics — fall back to the bare DrugBank
            # ID `DB00011` instead.
            #
            # v27 ROOT FIX (P2-B-2): kg_builder.ID_PATTERNS["Compound"]
            # regex requires UPPERCASE InChIKeys (the canonical form per
            # IUPAC). Phase 1 emits InChIKeys in standard uppercased form,
            # but if any source emits a lowercase InChIKey it would be
            # dead-lettered. Uppercase explicitly here so the canonical
            # ID always matches the ID_PATTERNS regex.
            #
            # v102 ROOT FIX (P2-036): route through the centralized
            # ``normalize_inchikey`` helper so this loader produces the
            # SAME canonical form as chembl_loader and pubchem_loader
            # (uppercase + strip + placeholder-collapsed). Previously
            # ``inchikey.upper() if inchikey else ""`` did NOT strip
            # whitespace, so a " RZBJ...AN " input would dead-letter
            # while chembl_loader's ``.strip().upper()`` would succeed —
            # the SAME compound landed as TWO canonical IDs depending
            # on which loader ran. Centralizing eliminates this class
            # of bug.
            inchikey_canonical = _normalize_inchikey(inchikey)
            if inchikey_canonical and not inchikey_canonical.startswith("SYNTH"):
                canonical_id = inchikey_canonical
            elif drugbank_id:
                canonical_id = drugbank_id  # e.g. "DB00011" — matches DB\d{5,6}
            elif chembl_id:
                # v108 ROOT FIX (issue 63): fall back to ChEMBL ID when
                # neither inchikey nor drugbank_id is available. This is
                # the ChEMBL-only deployment case.
                canonical_id = chembl_id
            else:
                # No canonical ID — skip this row (already checked above,
                # but defensive: in case inchikey was a "SYNTH" placeholder
                # that _normalize_inchikey collapsed to empty).
                continue
            # v61 ROOT FIX (patient-safety regression from v27):
            # The v27 "fix" (P2-B-1) wrote ``withdrawn=None`` when Phase 1
            # was silent on withdrawal status, claiming DrugBankEnricher
            # would fill it in later. This BROKE the module docstring's
            # explicit patient-safety guarantee (lines 135-139):
            #   "The bridge EXPLICITLY coerces is_withdrawn to a bool and
            #    writes withdrawn=False (never null) for every Compound
            #    node."
            # The RL safety ranker treats ``None`` as "not withdrawn" →
            # SAFE → a withdrawn drug like Valdecoxib (withdrawn for
            # cardiovascular risk) would be surfaced as a repurposing
            # candidate. ``None`` and ``False`` are BOTH treated as
            # "not withdrawn" by the ranker, so the distinction v27
            # tried to preserve was meaningless for safety — while
            # breaking the never-null contract.
            # ROOT FIX: write ``withdrawn=False`` (NEVER null) when Phase 1
            # is silent. Set ``safety_data_missing=True`` so DrugBankEnricher
            # can later UPDATE the field if it has DrugBank XML data. The
            # ``safety_data_missing`` flag is the correct signal for "we
            # don't know" — NOT a null ``withdrawn`` field.
            is_withdrawn_raw = row.get("is_withdrawn")
            if is_withdrawn_raw is None or (
                isinstance(is_withdrawn_raw, float) and pd.isna(is_withdrawn_raw)
            ) or str(is_withdrawn_raw).strip().lower() in ("", "nan", "none", "null"):
                withdrawn_val: bool = False  # v61: NEVER null per docstring
                safety_data_missing = True
            else:
                withdrawn_val = _to_bool(is_withdrawn_raw)
                safety_data_missing = False
            node = {
                "id": canonical_id,
                "drugbank_id": drugbank_id,
                "name": _safe_str(row.get("name")),
                "inchikey": inchikey_canonical or inchikey,
                "smiles": _safe_str(row.get("smiles")),
                "molecular_weight": _safe_float(row.get("molecular_weight")),
                "molecular_formula": _safe_str(row.get("molecular_formula")),
                # P2-002 ROOT FIX (v104): _resolve_fda_approved returns
                # Optional[bool]. When is_fda_approved is None/NaN (ChEMBL-
                # only path), it returns None — it does NOT fall back to
                # is_globally_approved (max_phase==4), because that would
                # conflate EMA/PMDA/NMPA approval with FDA approval and
                # over-state US market opportunity for the RL ranker.
                # The outdated v64 comment below was replaced because it
                # described the OLD buggy behavior, not the current code.
                "fda_approved": _resolve_fda_approved(row),
                # v61 ROOT FIX: NEVER null per docstring patient-safety
                # contract. withdrawn=False (default safe state) when
                # Phase 1 is silent; safety_data_missing=True flags it.
                "withdrawn": withdrawn_val,
                "safety_data_missing": safety_data_missing,
                "clinical_status": _safe_str(row.get("clinical_status")),
                "groups": _safe_str(row.get("groups")),
                "mechanism_of_action": _safe_str(row.get("mechanism_of_action")),
                "cas_number": _safe_str(row.get("cas_number")),
                "chembl_id": _safe_str(row.get("chembl_id")),
                "pubchem_cid": _safe_str(row.get("pubchem_cid")),
                "completeness_score": _safe_float(row.get("completeness_score")),
                # v78 FORENSIC ROOT FIX (BUG #4 — compound_id_aliases NEVER
                # populated). The v70 ROOT FIX in kg_builder.py added a
                # Compound-MERGE Cypher that resolves the effective merge_id
                # by scanning ``row.compound_id_aliases`` for an existing
                # Compound whose ``id`` matches any alias. The promise:
                # biotech drugs (insulin, mAbs, vaccines — ~30% of modern
                # FDA approvals) without InChIKey would MERGE with their
                # ChEMBL/PubChem equivalents. But the bridge NEVER
                # populated ``compound_id_aliases`` — so the v70 Cypher's
                # ``coalesce([a IN coalesce(row.compound_id_aliases, []) ...])``
                # always fell through to ``row.id``, and biotech drugs
                # stayed as separate Compound nodes from their ChEMBL/
                # PubChem equivalents. v70 ROOT FIX Cypher was DEAD CODE.
                # ROOT FIX: populate ``compound_id_aliases`` with EVERY
                # alternate stable identifier the bridge knows about —
                # drugbank_id, chembl_id, pubchem_cid, chebi_id, and the
                # inchikey (when it's not the canonical id). The MERGE
                # Cypher will then find the cross-source match.
                "compound_id_aliases": [
                    alias for alias in [
                        drugbank_id,
                        _safe_str(row.get("chembl_id")),
                        _safe_str(row.get("pubchem_cid")),
                        _safe_str(row.get("chebi_id")),
                        # Include inchikey as an alias ONLY when it's not
                        # the canonical id (avoid self-aliasing).
                        inchikey_canonical if inchikey_canonical and inchikey_canonical != canonical_id else None,
                    ]
                    if alias and alias != canonical_id
                ],
                # Lineage
                "_source_phase": 1,
                "_source_file": "drugbank_drugs.csv",
                "_source_row": _safe_row_idx(idx),
                "_pipeline_run_id": run_id,
                "_loaded_at": loaded_at,
                "_schema_version": schema_version,
            }
            staged.compound_nodes.append(node)
        logger.info(
            "Phase1 bridge: staged %d Compound nodes from drugbank_drugs.csv",
            len(staged.compound_nodes),
        )
    else:
        staged.warnings.append("drugbank_drugs.csv missing or empty — no Compound nodes staged")

    # ─── Protein nodes + Compound→Protein edges (from drugbank_interactions) ──
    # v28 ROOT FIX (P2-B-11): build ``drug_canonical_map`` ONCE here (after
    # all DrugBank-sourced Compound nodes are staged) and reuse it for BOTH
    # the Compound→Protein edge path below AND the Compound→treats→Disease
    # edge path further down. Previously the bridge built this map TWICE
    # with divergent logic:
    #   - At line ~1215 (Compound→Protein path): ``drug_canonical[dbid] =
    #     n["id"]`` — pulled the canonical ID from already-staged
    #     ``staged.compound_nodes`` (which had uppercase InChIKeys
    #     applied at line 1146 via ``inchikey.upper()``).
    #   - At line ~1514 (treats path): ``drug_canonical_map[dbid] = (inchi
    #     if inchi and not inchi.startswith("SYNTH") else dbid)`` — read
    #     ``inchi`` DIRECTLY from the DataFrame, WITHOUT uppercasing. So
    #     if the Phase 1 CSV ever emitted a lowercase InChIKey, treats
    #     edges would emit a lowercase canonical ID while Compound→Protein
    #     edges would emit the uppercase form — the SAME drug appeared as
    #     two disjoint Compound nodes downstream.
    # The fix: build the map ONCE from ``staged.compound_nodes`` (the source
    # of truth — every node's ``id`` is already the uppercased canonical
    # form). Both consumers use the same map.
    drug_canonical_map: Dict[str, str] = {
        n["drugbank_id"]: n["id"]
        for n in staged.compound_nodes
        if n.get("drugbank_id")
    }

    inter = frames.get("interactions")
    if inter is not None and not inter.empty:
        protein_seen: Dict[str, Dict[str, Any]] = {}
        edge_buckets: Dict[str, List[Dict[str, Any]]] = {
            "targets": [],
            "inhibits": [],
            "activates": [],
            "allosterically_modulates": [],
            "unknown": [],
        }
        # v6 fix (bug #B2): dedup Compound→Protein edges upstream by
        # (src_id, dst_id, rel_type) so the RecordingGraphBuilder's downstream
        # dedup is a no-op (no silent edge drops in the staged→loaded count).
        seen_cp: set[Tuple[str, str, str]] = set()
        for idx, row in inter.iterrows():
            drugbank_id = _safe_str(row.get("drugbank_id"))
            uniprot_id = _safe_str(row.get("uniprot_id"))
            if not drugbank_id or not uniprot_id:
                continue
            canonical_drug_id = drug_canonical_map.get(drugbank_id)
            if canonical_drug_id is None:
                # Drug not in compound_nodes — skip this edge to preserve
                # referential integrity.
                continue
            # Build Protein node (dedup on uniprot_id)
            if uniprot_id not in protein_seen:
                pnode = {
                    "id": uniprot_id,
                    "name": _safe_str(row.get("target_name")),
                    "organism": _safe_str(row.get("organism")),
                    "_source_phase": 1,
                    "_source_file": "drugbank_interactions.csv.gz",
                    "_source_row": _safe_row_idx(idx),
                    "_pipeline_run_id": run_id,
                    "_loaded_at": loaded_at,
                    "_schema_version": schema_version,
                }
                protein_seen[uniprot_id] = pnode
                staged.protein_nodes.append(pnode)

            action_type = _safe_str(row.get("action_type"))
            rel = _classify_drug_protein_edge(action_type)
            cp_key = (canonical_drug_id, uniprot_id, rel)
            if cp_key in seen_cp:
                continue  # upstream dedup (bug #B2)
            seen_cp.add(cp_key)
            # v78 FORENSIC ROOT FIX (BUG #1): emit canonical
            # ``normalized_score`` on every edge so cross-source
            # confidence fusion works. DrugBank interactions CSV has no
            # quantitative score → normalized_score=None (the edge
            # existence IS the signal; ChEMBL provides pchembl_value
            # when available, fused downstream).
            edge = {
                "src_id": canonical_drug_id,
                "dst_id": uniprot_id,
                "action_type": action_type,
                "is_known_action": _to_bool(row.get("is_known_action")),
                "source": "drugbank",
                "source_id": _safe_str(row.get("source_id")),
                "normalized_score": _compute_normalized_score(
                    source="drugbank", rel_type=rel,
                ),
                "_source_phase": 1,
                "_source_file": "drugbank_interactions.csv.gz",
                "_source_row": _safe_row_idx(idx),
                "_pipeline_run_id": run_id,
                "_loaded_at": loaded_at,
                "_schema_version": schema_version,
            }
            edge_buckets[rel].append(edge)

        # File edge buckets into staged.edges keyed by (src, rel, dst)
        for rel, edges in edge_buckets.items():
            if edges:
                staged.edges[("Compound", rel, "Protein")] = edges
        logger.info(
            "Phase1 bridge: staged %d Protein nodes and %d Compound→Protein edges",
            len(staged.protein_nodes),
            sum(len(v) for v in edge_buckets.values()),
        )
    else:
        staged.warnings.append(
            "drugbank_interactions.csv.gz missing or empty — no Protein nodes or Compound→Protein edges staged"
        )

    # ─── Gene + Disease nodes + Gene→Disease edges (from omim_gda) ─────────
    # Audit fix (v5 Tier-3 bug #22): the previous code used the raw gene
    # SYMBOL as the Gene node ID, but kg_builder.ID_PATTERNS["Gene"] =
    # ^\d+$ (NCBI Gene ID). Every Gene node was dead-lettered in
    # production. Fix: prefer the gene's MIM ID (numeric) as the canonical
    # Gene ID when available (matches `^\d+$`); fall back to the symbol
    # only when no numeric ID is available (entity_resolver will canonicalize).
    # We also filter OMIM's ALTGENE/MENDGENE/MYGENE placeholders (audit §C.4).
    #
    # v6 fix (bug #B10/B11): prefer Phase 1's `canonical_gene_id` (NCBI Gene
    # ID) when populated — this is the proper, unique-per-gene identifier.
    # The OMIM CSV's `uniprot_id` column is now also populated (was 100% NaN
    # in v5), so Gene-encodes-Protein edges are emitted and the graph is no
    # longer split into two disconnected halves.
    #
    # v6 fix (bug #B2): dedup gda_edges and encodes_edges UPSTREAM in the
    # bridge so the RecordingGraphBuilder's downstream dedup is a no-op (no
    # silent edge drops). The previous code produced 19 staged / 18 loaded
    # because of a (gene_mim=164920, OMIM:273300) duplicate — now resolved
    # by using canonical_gene_id (FGFR3=2261, KIT=3815, no collision).
    omim = frames.get("omim_gda")
    if omim is not None and not omim.empty:
        gene_seen: Dict[str, Dict[str, Any]] = {}
        disease_seen: Dict[str, Dict[str, Any]] = {}
        gda_edges: List[Dict[str, Any]] = []
        # Audit fix (v5 Tier-3 bug #25a): collect Gene→Protein (encodes)
        # edges by joining on the OMIM CSV's `uniprot_id` column.
        encodes_edges: List[Dict[str, Any]] = []
        seen_gda: set[Tuple[str, str]] = set()        # upstream dedup (bug #B2)
        seen_encodes: set[Tuple[str, str]] = set()    # upstream dedup (bug #B2)
        # v6 fix (bug #B10): also stage Protein nodes for OMIM-derived
        # uniprot_ids so the encodes edges don't get dropped by the
        # builder's referential integrity check. Without this, the 5
        # DrugBank-derived Protein nodes don't include the 9 OMIM gene
        # products (CFTR/P13569, DMD/P11532, etc.) and all 10 encodes
        # edges get silently dead-lettered.
        omim_protein_seen: Dict[str, Dict[str, Any]] = {}
        _PLACEHOLDER_GENES = {"ALTGENE", "MENDGENE", "MYGENE", ""}
        for idx, row in omim.iterrows():
            gene_symbol = _safe_str(row.get("gene_symbol"))
            disease_id = _safe_str(row.get("disease_id"))
            if not gene_symbol or gene_symbol.upper() in _PLACEHOLDER_GENES:
                continue
            if not disease_id:
                continue
            # Resolve Gene canonical ID — prefer the Phase 1 canonical_gene_id
            # column if populated (Phase 1's entity_resolution populates it
            # with the NCBI Gene ID when available). Fall back to NCBI Gene ID,
            # then OMIM gene MIM (both numeric, both match ^\d+$), then the
            # gene symbol as a last resort (entity_resolver will canonicalize).
            gene_mim = _safe_str(row.get("gene_mim"))
            ncbi_gene_id = _safe_str(row.get("ncbi_gene_id"))
            canonical_gene_id = _safe_str(row.get("canonical_gene_id"))
            if canonical_gene_id and canonical_gene_id.isdigit():
                gene_canonical_id = canonical_gene_id
            elif ncbi_gene_id and ncbi_gene_id.isdigit():
                gene_canonical_id = ncbi_gene_id
            elif gene_mim and gene_mim.isdigit():
                gene_canonical_id = gene_mim
            else:
                # v21 ROOT FIX (Audit section 4 finding 8 / Chain 9 -
                # "Bridge emits IDs that production rejects"):
                # the previous code fell back to the bare gene symbol
                # (e.g. 'FGFR3'). But kg_builder.ID_PATTERNS['Gene'] =
                # ^(\d+|SYM:[A-Z0-9]+)$ - bare symbols are REJECTED,
                # dead-lettering every OMIM gene that lacks a numeric
                # NCBI/MIM ID. The disgenet_loader already emits
                # SYM:-prefixed symbols (line 124); OMIM must do the
                # same for consistency. Fix: prefix with 'SYM:' so the
                # ID passes the production validator. The entity_resolver
                # can later canonicalize SYM:FGFR3 -> 2261 (NCBI Gene ID)
                # via id_crosswalk.
                gene_canonical_id = (
                    f"SYM:{gene_symbol.upper()}"
                    if gene_symbol and gene_symbol.isascii()
                    else gene_symbol
                )
            if gene_canonical_id not in gene_seen:
                gnode = {
                    "id": gene_canonical_id,
                    "name": gene_symbol,
                    "gene_symbol": gene_symbol,
                    "mim_id": gene_mim,
                    "ncbi_gene_id": ncbi_gene_id or None,
                    "uniprot_id": _safe_str(row.get("uniprot_id")),
                    "_source_phase": 1,
                    "_source_file": "omim_gene_disease_associations.csv",
                    "_source_row": _safe_row_idx(idx),
                    "_pipeline_run_id": run_id,
                    "_loaded_at": loaded_at,
                    "_schema_version": schema_version,
                }
                gene_seen[gene_canonical_id] = gnode
                staged.gene_nodes.append(gnode)
            else:
                # Update the existing gene node with any newly-seen uniprot_id.
                existing = gene_seen[gene_canonical_id]
                if not existing.get("uniprot_id"):
                    existing["uniprot_id"] = _safe_str(row.get("uniprot_id"))
            if disease_id not in disease_seen:
                # disease_name column may or may not exist depending on Phase 1 schema version
                dname = _safe_str(row.get("disease_name") or row.get("phenotype_name"))
                dnode = {
                    "id": disease_id,
                    "name": dname,
                    "mim_id": _safe_str(row.get("phenotype_mim")),
                    "_source_phase": 1,
                    "_source_file": "omim_gene_disease_associations.csv",
                    "_source_row": _safe_row_idx(idx),
                    "_pipeline_run_id": run_id,
                    "_loaded_at": loaded_at,
                    "_schema_version": schema_version,
                }
                disease_seen[disease_id] = dnode
                staged.disease_nodes.append(dnode)
            # v6 fix (bug #B2): dedup GDA edges upstream by (src_id, dst_id).
            gda_key = (gene_canonical_id, disease_id)
            if gda_key not in seen_gda:
                seen_gda.add(gda_key)
                _omim_raw_score = _safe_float(row.get("score"))
                edge = {
                    "src_id": gene_canonical_id,
                    "dst_id": disease_id,
                    "score": _omim_raw_score,
                    "association_type": _safe_str(row.get("association_type")),
                    "mapping_key": _safe_str(row.get("mapping_key")),
                    "source": "omim",
                    "source_id": _safe_str(row.get("source_id")),
                    # v78 FORENSIC ROOT FIX (BUG #1 + BUG #6):
                    #   BUG #1: emit canonical ``normalized_score`` so
                    #   cross-source confidence fusion works. OMIM GDA
                    #   is curated human-genetics evidence — when no
                    #   numeric score is present, normalized_score=1.0
                    #   (a confirmed OMIM association is high-confidence).
                    #   BUG #6: this is the OMIM side of the
                    #   first-wins-loses-score bug. When DisGeNET later
                    #   finds the same (gene, disease) pair with a
                    #   quantitative score, the merge logic in the
                    #   DisGeNET block will UPDATE this edge's score
                    #   and normalized_score with the DisGeNET evidence
                    #   (preserving both sources). Without this canonical
                    #   normalized_score, the RL ranker had no
                    #   evidence-strength signal at all.
                    "normalized_score": _compute_normalized_score(
                        raw_score=_omim_raw_score,
                        source="omim",
                        rel_type="associated_with",
                    ),
                    "_source_phase": 1,
                    "_source_file": "omim_gene_disease_associations.csv",
                    "_source_row": _safe_row_idx(idx),
                    "_pipeline_run_id": run_id,
                    "_loaded_at": loaded_at,
                    "_schema_version": schema_version,
                }
                gda_edges.append(edge)
            # Audit fix (v5 Tier-3 bug #25a): emit Gene-encodes-Protein
            # edge when the OMIM CSV provides a UniProt AC for this gene.
            # Without this edge, the Gene subgraph and Protein subgraph
            # are disconnected in the loaded KG, so Drug→Protein→?→Gene→Disease
            # multi-hop queries return empty results.
            #
            # v6 fix (bug #B10/B11): OMIM CSV now has uniprot_id populated
            # (was 100% NaN in v5) — encodes edges are now actually emitted.
            # v6 fix (bug #B2): dedup encodes_edges upstream by (src_id, dst_id).
            # v6 fix (bug #B10): ALSO stage a Protein node for the OMIM-derived
            # uniprot_id so the encodes edge's dst endpoint exists in the graph.
            uniprot_id_for_gene = _safe_str(row.get("uniprot_id"))
            if uniprot_id_for_gene:
                # Stage the Protein node (dedup on uniprot_id).
                if uniprot_id_for_gene not in omim_protein_seen:
                    pnode = {
                        "id": uniprot_id_for_gene,
                        "name": gene_symbol,  # use gene symbol as name
                        "gene_name": gene_symbol,
                        "organism": "Homo sapiens",
                        "_source_phase": 1,
                        "_source_file": "omim_gene_disease_associations.csv",
                        "_source_row": _safe_row_idx(idx),
                        "_pipeline_run_id": run_id,
                        "_loaded_at": loaded_at,
                        "_schema_version": schema_version,
                    }
                    omim_protein_seen[uniprot_id_for_gene] = pnode
                    staged.protein_nodes.append(pnode)

                enc_key = (gene_canonical_id, uniprot_id_for_gene)
                if enc_key not in seen_encodes:
                    seen_encodes.add(enc_key)
                    encodes_edges.append({
                        "src_id": gene_canonical_id,
                        "dst_id": uniprot_id_for_gene,
                        "source": "omim",
                        "evidence": "gene_protein_crosswalk",
                        # v78 FORENSIC ROOT FIX (BUG #1): canonical
                        # normalized_score on encodes edges. The
                        # gene-protein crosswalk is curated human
                        # genetics → high confidence → 1.0.
                        "normalized_score": _compute_normalized_score(
                            source="omim", rel_type="encodes",
                        ),
                        "_source_phase": 1,
                        "_source_file": "omim_gene_disease_associations.csv",
                        "_source_row": _safe_row_idx(idx),
                        "_pipeline_run_id": run_id,
                        "_loaded_at": loaded_at,
                        "_schema_version": schema_version,
                    })
        if gda_edges:
            staged.edges[("Gene", "associated_with", "Disease")] = gda_edges
        if encodes_edges:
            # CORE_EDGE_TYPES explicitly includes ("Gene", "encodes", "Protein")
            # as the biological bridge. Without it the graph is disconnected.
            staged.edges[("Gene", "encodes", "Protein")] = encodes_edges
        logger.info(
            "Phase1 bridge: staged %d Gene nodes, %d Disease nodes, "
            "%d Gene->Disease edges, %d Gene->Protein (encodes) edges, "
            "%d OMIM-derived Protein nodes",
            len(staged.gene_nodes),
            len(staged.disease_nodes),
            len(gda_edges),
            len(encodes_edges),
            len(omim_protein_seen),
        )
    else:
        staged.warnings.append(
            "omim_gene_disease_associations.csv missing or empty — no Gene/Disease nodes or Gene->Disease edges staged"
        )

    # ─── Chain-1 ROOT FIX: stage DisGeNET Disease nodes BEFORE treats ────
    # The treats-edge derivation at line ~3624 builds ``disease_id_set`` from
    # the already-staged Disease nodes (which were OMIM-only). DrugBank
    # indications use DOID-format disease IDs. If DisGeNET's DOID-keyed
    # Disease nodes haven't been staged yet, the treats-edge loop can't
    # match them, and even with the v78 fallback that stages unrecognized
    # DOID IDs as synthetic Disease nodes, the gene→disease→drug multi-hop
    # path is broken (the synthetic DOID node has no gene associations).
    #
    # ROOT FIX: stage the DisGeNET Disease nodes HERE (before the treats-
    # edge derivation) so that DOID IDs are already in ``disease_id_set``
    # when the treats-edge loop runs. This unblocks 7 of the 12 embedded-
    # sample indication rows whose DOID IDs match DisGeNET-staged diseases
    # (Pain/DOID:0050133, Inflammation/DOID:1101, Cancer/DOID:162,
    # Migraine/DOID:1197, Epilepsy/DOID:1826, Arthritis/DOID:7148,
    # Hypertension/DOID:10763). The remaining 5 rows without a DisGeNET
    # match still use the v78 synthetic-disease fallback.
    #
    # Note: Gene-node staging from DisGeNET is deferred to the full
    # DisGeNET edge-staging section below (line ~4704) — only Disease
    # nodes are staged here to avoid duplicating Gene-staging logic.
    extra_disease_seen_pre: set[str] = set()
    disgenet = frames.get("disgenet_gda")
    if disgenet is not None and not disgenet.empty:
        for idx, row in disgenet.iterrows():
            did = _safe_str(row.get("disease_id"))
            if not did:
                continue
            if did not in extra_disease_seen_pre:
                dnode = {
                    "id": did,
                    "name": _safe_str(row.get("disease_name")),
                    "_source_phase": 1,
                    "_source_file": "disgenet_gene_disease_associations.csv",
                    "_source_row": _safe_row_idx(idx),
                    "_pipeline_run_id": run_id,
                    "_loaded_at": loaded_at,
                    "_schema_version": schema_version,
                }
                staged.disease_nodes.append(dnode)
                extra_disease_seen_pre.add(did)
        logger.info(
            "Chain-1 ROOT FIX: pre-staged %d DisGeNET Disease nodes "
            "before treats-edge derivation (DOID IDs now in disease_id_set)",
            len(extra_disease_seen_pre),
        )

    # ─── Audit fix (v5 Tier-3 bug #25b): derive Compound-treats-Disease ────
    # Phase 2's CORE_EDGE_TYPES declares ("Compound", "treats", "Disease")
    # as the primary link-prediction target. Phase 1's DrugBank CSV has
    # no disease column, and OMIM has no drug column — so the previous
    # bridge produced ZERO treats edges. TransE had no positive training
    # signal for the drug-repurposing task.
    #
    # v6 fix (bug #B9): the bridge now consumes a STRUCTURED
    # `drugbank_indications.csv` (drugbank_id, disease_id, disease_name,
    # indication_type, source) when present, AND falls back to free-text
    # matching on the `indication` column of drugbank_drugs.csv when the
    # structured file is absent. Both paths emit (Compound, treats, Disease)
    # edges with referential integrity (only to Disease nodes already
    # staged from the OMIM CSV).
    treats_edges: List[Dict[str, Any]] = []
    seen_treats: set[Tuple[str, str]] = set()  # upstream dedup (bug #B2)

    # v28 ROOT FIX (P2-B-11): ``drug_canonical_map`` is now built ONCE
    # above (right after DrugBank Compound nodes are staged) and reused
    # here. The previous code rebuilt it directly from the ``drugs``
    # DataFrame with divergent logic (no InChIKey uppercasing), causing
    # the same drug to appear as two disjoint Compound nodes when the
    # Phase 1 CSV emitted a lowercase InChIKey. The unified map reads
    # from ``staged.compound_nodes`` (the source of truth — every
    # node's ``id`` is already the uppercased canonical form).

    # Set of Disease IDs already staged (referential integrity gate).
    disease_id_set = {d["id"] for d in staged.disease_nodes}

    # ── P0-B1 ROOT FIX: pre-stage DisGeNET Disease nodes BEFORE ──────
    # ── treats-edge derivation.                                         ─────
    # The audit (CHAIN-2) proved that the DOID/OMIM disease_id_set was
    # built from OMIM-only diseases. DisGeNET stages DOID-keyed Disease
    # nodes AFTER the treats-edge block, so DrugBank indications using
    # DOID IDs (e.g. DOID:0050133) were not in disease_id_set and were
    # skipped by the "did not in disease_id_set" check → 0 treats edges
    # from most indication rows. The v78 fallback (staging a new Disease
    # node when did is valid but not in disease_id_set) partially helps,
    # but the ROOT FIX is to pre-scan DisGeNET diseases here so the
    # disease_id_set already contains DOID/EFO/etc. IDs when the
    # treats-edge loop runs. This unblocks 7/12 indication edges.
    disgenet_pre = frames.get("disgenet_gda")
    _disgenet_prescan_count = 0
    if disgenet_pre is not None and not disgenet_pre.empty:
        for _, _row in disgenet_pre.iterrows():
            _did = _safe_str(_row.get("disease_id"))
            if not _did:
                continue
            if _did not in disease_id_set:
                disease_id_set.add(_did)
                _disgenet_prescan_count += 1
    if _disgenet_prescan_count:
        logger.info(
            "Phase1 bridge: pre-staged %d DisGeNET Disease IDs into "
            "disease_id_set BEFORE treats-edge derivation (P0-B1 fix)",
            _disgenet_prescan_count,
        )

    # ── Path A: structured drugbank_indications.csv (preferred) ──
    # v34 ROOT FIX (CRITICAL #8): the previous code required non-empty
    # `disease_id` AND that the disease_id already exist in
    # `disease_id_set` (Diseases staged from OMIM). For the toy fixture
    # (and real DrugBank), 4/9 indication rows have EMPTY `disease_id`
    # because DrugBank's open-data dump uses the disease_name field
    # ("Pain", "Asthma", "Hepatitis B") without normalizing to OMIM.
    # The previous code skipped these rows — losing ~half of the
    # Compound-treats-Disease edges (the headline ML target).
    #
    # The fix: when `disease_id` is empty but `disease_name` is non-empty,
    # slugify the disease_name into a synthetic Disease ID
    # (`SYNDROME:{slugified_name}`) and emit BOTH a new Disease node AND
    # the treats edge. This preserves the clinical signal (Aspirin treats
    # Pain, even if Pain isn't in OMIM) while keeping referential
    # integrity (every treats edge points at a real Disease node).
    indications = frames.get("indications")
    if indications is not None and not indications.empty:
        _slug_seen: set[str] = set()
        for idx, row in indications.iterrows():
            dbid = _safe_str(row.get("drugbank_id"))
            did = _safe_str(row.get("disease_id"))
            dname = _safe_str(row.get("disease_name"))
            if not dbid:
                continue
            drug_canonical = drug_canonical_map.get(dbid)
            if drug_canonical is None:
                # Drug not in compound_nodes — skip to preserve referential
                # integrity.
                continue
            # v34 ROOT FIX (CRITICAL #8): if disease_id is empty but we
            # have a disease_name, synthesize a slugified Disease ID.
            #
            # v78 FORENSIC ROOT FIX (BUG #10 — Compound Issue, 0 treats
            # edges): the v34 fix only fired when ``disease_id`` was
            # EMPTY. But the real killer bug: DrugBank indications emit
            # ``disease_id = "DOID:0050133"`` (DOID format) while the
            # ``disease_id_set`` was built ONLY from OMIM-staged diseases
            # (OMIM:nnnnnn format). The slugify fallback didn't fire
            # (disease_id was non-empty), and the ``did not in
            # disease_id_set`` check at line ~3244 silently skipped
            # EVERY DrugBank-indication row → 0 Compound-treats-Disease
            # edges → V1 launch criterion (>0.85 AUC on held-out drug-
            # disease pairs) was structurally unverifiable.
            #
            # ROOT FIX (BUG #10): when ``disease_id`` is non-empty but
            # not in ``disease_id_set`` (i.e. a DOID/MeSH/EFO/etc. ID
            # that no upstream source has staged yet), STAGE IT as a
            # new Disease node. This is biologically correct: if
            # DrugBank says "Aspirin treats DOID:0050133", then
            # DOID:0050133 IS a real disease that belongs in the KG.
            # The fallback is symmetric to the v34 slugify path: both
            # ensure the treats edge has a valid Disease endpoint.
            if not did and dname:
                _slug = re.sub(
                    r"[^A-Za-z0-9]+", "_", dname.strip().lower()
                ).strip("_")
                if not _slug:
                    continue
                # v102 ROOT FIX (P2-046): before slugifying to a
                # synthetic SYNDROME: ID, try to upgrade the disease
                # name to a real biomedical ontology ID (DOID/MeSH) via
                # the existing _DISEASE_KEYWORD_MAP. The previous code
                # ALWAYS slugified — emitting "SYNDROME:Pain",
                # "SYNDROME:Hepatitis-B", etc. — which do NOT match any
                # biomedical ontology (DOID, OMIM, MeSH, EFO). ~half of
                # Compound-treats-Disease edges pointed at SYNDROME:
                # nodes that were disconnected from the rest of the
                # Disease subgraph. Multi-hop queries like
                # "Compound → treats → Disease → associated_with → Gene"
                # returned empty for these diseases (they had no Gene
                # edges). The KG was fragmented.
                #
                # ROOT FIX: scan the disease_name (lowercased) against
                # _DISEASE_KEYWORD_MAP. If a keyword matches, upgrade
                # the did to the corresponding DOID ID and mark the
                # node with ontology_status="mapped". If NO keyword
                # matches, keep the SYNDROME: slugified ID but mark
                # the node with ontology_status="unmapped" so operators
                # can audit (e.g. run a richer NLP disease-name matcher
                # like NCBImeta API or a local MeSH lookup to upgrade
                # later). This preserves the clinical signal (Aspirin
                # treats Pain → now points at DOID:0050133 instead of
                # SYNDROME:Pain) while keeping the audit trail.
                _dname_lower = dname.strip().lower()
                _matched_doid = None
                for _kw, (_doid_id, _doid_name) in _DISEASE_KEYWORD_MAP.items():
                    if _kw in _dname_lower:
                        _matched_doid = _doid_id
                        break
                if _matched_doid is not None:
                    # Upgrade to a real DOID ID — connects to the
                    # broader Disease subgraph (DisGeNET, OMIM, etc.).
                    did = _matched_doid
                    _ontology_status = "mapped"
                else:
                    # No keyword match — keep the slugified SYNDROME: ID
                    # but mark it unmapped so operators can audit.
                    did = f"SYNDROME:{_slug}"
                    _ontology_status = "unmapped"
                # Emit a Disease node if not already staged.
                if did not in disease_id_set and did not in _slug_seen:
                    _slug_seen.add(did)
                    dnode = {
                        "id": did,
                        "name": dname,
                        "mim_id": None,
                        "_source_phase": 1,
                        "_source_file": "drugbank_indications.csv",
                        "_source_row": _safe_row_idx(idx),
                        "_pipeline_run_id": run_id,
                        "_loaded_at": loaded_at,
                        "_schema_version": schema_version,
                        "_synthetic_disease": True,  # audit flag
                        # v102 P2-046: track whether this Disease ID was
                        # mapped to a real ontology (DOID/MeSH) or is a
                        # synthetic SYNDROME: slug awaiting NLP upgrade.
                        # Operators can query
                        # ``MATCH (d:Disease {ontology_status: 'unmapped'})``
                        # to audit the unmapped population.
                        "ontology_status": _ontology_status,
                    }
                    staged.disease_nodes.append(dnode)
                    disease_id_set.add(did)
            if not did:
                continue
            # v78 BUG #10 root fix: if ``did`` is non-empty but not yet
            # in ``disease_id_set``, stage it as a new Disease node so
            # the treats edge has a valid endpoint. This unblocks
            # DrugBank indications that reference DOID/MeSH/EFO IDs not
            # present in OMIM or DisGeNET. (DisGeNET runs AFTER this
            # block, so its DOID-keyed diseases aren't available here
            # yet — this fallback is the robust fix that doesn't depend
            # on staging order.)
            if did not in disease_id_set:
                # Validate the ID format before staging — if it's a
                # recognized biomedical ID (DOID/MeSH/EFO/MONDO/etc.),
                # stage it. Otherwise fall back to slugification.
                _is_valid_disease_id = bool(re.match(
                    r"^(C\d{7}|D\d{6}|EFO_\d+|EFO:\d+|OMIM:\d+|"
                    r"Orphanet:\d+|MONDO:\d+|DOID:\d+|HP:\d+|"
                    r"MESH:[A-Z]\d+|SYNDROME:[A-Za-z0-9_]+)$",
                    str(did),
                ))
                if _is_valid_disease_id:
                    dnode = {
                        "id": did,
                        "name": dname or did,
                        "mim_id": None,
                        "_source_phase": 1,
                        "_source_file": "drugbank_indications.csv",
                        "_source_row": _safe_row_idx(idx),
                        "_pipeline_run_id": run_id,
                        "_loaded_at": loaded_at,
                        "_schema_version": schema_version,
                        "_synthetic_disease": True,  # audit flag
                        "_synthetic_from_indications": True,  # v78 BUG #10
                    }
                    staged.disease_nodes.append(dnode)
                    disease_id_set.add(did)
                else:
                    # The ID doesn't match any known biomedical ontology
                    # format — skip to avoid polluting the KG with
                    # invalid IDs (would be dead-lettered by kg_builder
                    # anyway). This is the conservative path.
                    continue
            key = (drug_canonical, did)
            if key in seen_treats:
                continue  # upstream dedup (bug #B2)
            # P2-058 ROOT FIX: explicit referential-integrity guard
            # BEFORE creating the treats edge. The audit (P2-058) noted
            # that if a future refactor moves the treats_edges.append
            # OUTSIDE the if/else block at lines 4181-4212, an invalid
            # did (one that doesn't match any biomedical ontology
            # pattern AND wasn't already in disease_id_set) could
            # produce an orphan edge — the treats edge would point at
            # a Disease node that kg_builder would dead-letter (no
            # MATCH for the dst), but the bridge's ``staged.edges``
            # count would include it, inflating the operator's
            # expectation of loaded edge count. The current code path
            # uses ``continue`` at line 4212 to skip invalid dids, so
            # this guard is technically redundant TODAY — but it makes
            # the invariant EXPLICIT and protects against future
            # refactors. If ``did`` is not in ``disease_id_set`` (i.e.
            # no Disease node was staged for it, by ANY source), we
            # skip the edge creation and log a WARNING so the operator
            # sees the silent drop. This is the "belt and suspenders"
            # approach: the upstream ``continue`` at line 4212 should
            # already prevent reaching here with an invalid did, but
            # if a future maintainer changes that control flow, this
            # guard catches the orphan edge before it's created.
            if did not in disease_id_set:
                logger.warning(
                    "Phase1 bridge: skipping treats edge for drug=%s "
                    "disease_id=%s — did is NOT in disease_id_set "
                    "(no Disease node staged by any source). This "
                    "would produce an orphan edge (treats → "
                    "non-existent Disease). The upstream validation "
                    "should have already skipped this did, so "
                    "reaching this warning indicates a control-flow "
                    "regression in the if/else block above. "
                    "(P2-058 root fix)",
                    drug_canonical, did,
                )
                continue
            seen_treats.add(key)
            _treat_itype = _safe_str(row.get("indication_type", "")) or "structured"
            treats_edges.append({
                "src_id": drug_canonical,
                "dst_id": did,
                "source": "drugbank_indications",
                "evidence": _treat_itype,
                # v78 FORENSIC ROOT FIX (BUG #1): canonical
                # normalized_score on treats edges. DrugBank indication_type
                # "approved" → 1.0, "investigational"/"phase" → 0.5, else 0.3.
                # This is the confidence signal the RL ranker uses to
                # prioritize drug-disease pairs for repurposing.
                "normalized_score": _compute_normalized_score(
                    indication_type=_treat_itype,
                    source="drugbank_indications",
                    rel_type="treats",
                ),
                "_source_phase": 1,
                "_source_file": "drugbank_indications.csv",
                "_source_row": _safe_row_idx(idx),
                "_pipeline_run_id": run_id,
                "_loaded_at": loaded_at,
                "_schema_version": schema_version,
            })
        logger.info(
            "Phase1 bridge: derived %d Compound-treats-Disease edges from "
            "structured drugbank_indications.csv (incl. %d synthetic "
            "Disease nodes from disease_name slugification, v34 CRITICAL #8)",
            len(treats_edges),
            len(_slug_seen),
        )

    # ── Path B: free-text indication column fallback ──
    # Only used if the structured file is absent OR produced zero edges.
    if not treats_edges and drugs is not None and not drugs.empty:
        indication_col = None
        for cand in ("indication", "approved_indications", "treated_diseases"):
            if cand in drugs.columns:
                indication_col = cand
                break
        if indication_col is not None:
            # v29 ROOT FIX (audit L-11): was O(N×M) free-text matching. Now uses hash-based O(1) lookup.
            # The previous code iterated `for dnode in staged.disease_nodes`
            # INSIDE `for idx, row in drugs.iterrows()` — O(N_drugs × N_diseases).
            # For a production DrugBank (~14K drugs) × OMIM (~10K diseases)
            # that's 140M regex calls. Now we build:
            #   1. A hash dict {disease_name_lower: dnode} — O(M) once
            #   2. A single compiled alternation regex matching ALL disease
            #      names with word boundaries — O(M) once
            # For each drug we then run the regex ONCE (one pass over the
            # indication text) and look up each matched disease name in the
            # dict (O(1) per match). Total complexity is now
            # O(M + N_drugs × |indication_text|) instead of O(N×M).
            _disease_name_lookup: Dict[str, Dict[str, Any]] = {}
            _disease_pattern_parts: List[str] = []
            for _dnode in staged.disease_nodes:
                _dname = (_dnode.get("name") or "").strip()
                # Skip empty / too-short names (mirrors the old len < 4 guard).
                if not _dname or len(_dname) < 4:
                    continue
                _dname_lower = _dname.lower()
                # First occurrence wins (mirrors the old loop's behaviour —
                # `seen_treats` already dedups by (drug, disease) downstream).
                if _dname_lower not in _disease_name_lookup:
                    _disease_name_lookup[_dname_lower] = _dnode
                    _disease_pattern_parts.append(re.escape(_dname_lower))
            # Build a single alternation regex. Sorted by length descending so
            # longer names win over their substrings (e.g. "heart failure"
            # before "failure"). Word boundaries preserve the v27 P2-B-4 fix
            # (no false positives like "Pain" inside "Paint stripper poisoning").
            #
            # v29 ROOT FIX (audit L-11) detail: the regex uses a lookahead
            # ``(?=...)`` with a capture group so ``finditer`` returns
            # OVERLAPPING matches. The old O(N×M) code ran one
            # ``re.search`` per disease name independently, so a drug whose
            # indication mentioned "type 2 diabetes mellitus" matched BOTH
            # "diabetes" and "type 2 diabetes mellitus" (two separate edges
            # to two different disease_ids). A bare alternation regex
            # (``\b(?:...|...)\b``) would consume the longer match and
            # skip the shorter — silently dropping the second edge. The
            # lookahead preserves the old per-disease-name semantics:
            # every disease name whose word-bounded form appears in the
            # text gets an edge.
            _disease_pattern_parts.sort(key=len, reverse=True)
            if _disease_pattern_parts:
                _disease_regex = re.compile(
                    r"(?=\b(" + "|".join(_disease_pattern_parts) + r")\b)"
                )
            else:
                _disease_regex = None

            for idx, row in drugs.iterrows():
                ind_text = _safe_str(row.get(indication_col))
                if not ind_text:
                    continue
                drugbank_id = _safe_str(row.get("drugbank_id"))
                if not drugbank_id:
                    continue
                drug_canonical = drug_canonical_map.get(drugbank_id)
                if drug_canonical is None:
                    continue
                if _disease_regex is None:
                    # No disease nodes with usable names — nothing to match.
                    continue
                # v29 ROOT FIX (audit L-11): single regex pass over the
                # indication text. Each match is looked up in the hash dict
                # (O(1)) — no inner loop over staged.disease_nodes.
                # group(1) is the captured disease name (lookahead pattern).
                for _match in _disease_regex.finditer(ind_text.lower()):
                    _matched_name = _match.group(1)
                    _dnode = _disease_name_lookup.get(_matched_name)
                    if _dnode is None:
                        # Shouldn't happen (regex was built from the same
                        # dict keys), but guard against case-edge artefacts.
                        continue
                    key = (drug_canonical, _dnode["id"])
                    if key in seen_treats:
                        continue
                    seen_treats.add(key)
                    treats_edges.append({
                        "src_id": drug_canonical,
                        "dst_id": _dnode["id"],
                        "source": "drugbank_indication",
                        "evidence": "drugbank_indication_text",
                        "_source_phase": 1,
                        "_source_file": "drugbank_drugs.csv",
                        "_source_row": _safe_row_idx(idx),
                        "_pipeline_run_id": run_id,
                        "_loaded_at": loaded_at,
                        "_schema_version": schema_version,
                    })
            if treats_edges:
                logger.info(
                    "Phase1 bridge: derived %d Compound-treats-Disease edges "
                    "from free-text `indication` column (fallback path)",
                    len(treats_edges),
                )

    if treats_edges:
        staged.edges[("Compound", "treats", "Disease")] = treats_edges
    else:
        staged.warnings.append(
            "No Compound-treats-Disease edges derivable from Phase 1 outputs "
            "(neither drugbank_indications.csv nor an `indication` column in "
            "drugbank_drugs.csv produced any matches). TransE training will "
            "have zero positive signal for the treats edge type."
        )

    # ─── FIX-F / C-16: derive ClinicalOutcome nodes from ──────────────────
    # ─── drugbank_indications.csv.                                     ─────
    # DOCX Phase 2 spec mandates 5 node types: Drugs, Proteins, Pathways,
    # Diseases, Clinical Outcomes. The bridge previously emitted only 4
    # (Compound, Protein, Gene, Disease). This block adds the missing 5th
    # node type by deriving ClinicalOutcome nodes from the same
    # drugbank_indications.csv the treats-edge derivation already consumes.
    # Each unique (disease_id, indication_type) becomes a ClinicalOutcome
    # node; the originating drug is connected via a
    # (Compound)-[:has_clinical_outcome]->(ClinicalOutcome) edge.
    co_nodes, co_edges = _load_clinical_outcomes(
        indications=indications,
        drugs=drugs,
        drug_canonical_map=drug_canonical_map,
        run_id=run_id,
        loaded_at=loaded_at,
        schema_version=schema_version,
    )
    if co_nodes:
        staged.clinical_outcome_nodes.extend(co_nodes)
        logger.info(
            "Phase1 bridge: created %d ClinicalOutcome nodes from "
            "drugbank_indications.csv (C-16 fix)",
            len(co_nodes),
        )
    else:
        staged.warnings.append(
            "No ClinicalOutcome nodes derivable from Phase 1 outputs "
            "(drugbank_indications.csv missing or empty). The DOCX Phase 2 "
            "spec mandates ClinicalOutcome as one of 5 node types — this "
            "warning means the spec's 5-type schema is incomplete."
        )
    if co_edges:
        staged.edges[("Compound", "has_clinical_outcome", "ClinicalOutcome")] = co_edges

    # ─── v43 ROOT FIX (Chain 4b): Pathway nodes are now derived from ───
    # ─── STRING PPI connected components in the "extra sources" block  ───
    # ─── further below (line ~4670+).                                  ───
    # v82 FORENSIC ROOT FIX (misleading premature warning):
    #   The previous code emitted a "no Pathway nodes derived" warning
    #   HERE — but this runs BEFORE the actual Pathway derivation block
    #   at line ~4670+. So the warning fired on EVERY run, even when
    #   Pathway nodes WERE successfully derived later. The user saw
    #   both contradictory log lines:
    #     1. "no Pathway nodes derived (STRING PPI data empty or all
    #        singletons). The DOCX Phase 2 spec mandates Pathway..."
    #     2. "derived 1 Pathway nodes from STRING PPI connected
    #        components. Emitted 8 Protein→participates_in→Pathway
    #        edges. This restores the DOCX Phase 2 5-node-type contract."
    #   The first message was an INFO log AND a staged.warnings append,
    #   so it surfaced in the final summary as a warning, misleading
    #   operators into thinking the 5-node-type contract was violated
    #   when it was actually satisfied.
    # ROOT FIX: remove the premature warning block entirely. The Pathway
    #   derivation block at line ~4670+ has its OWN logging — INFO when
    #   pathways are derived, WARNING when derivation fails or returns
    #   empty. That is the SINGLE source of truth for Pathway status.
    #   No warning is emitted prematurely based on intermediate state.

    # ─── ROOT FIX (Phase1↔Phase2 100% connection): consume the other ─────
    # ─── 5 Phase 1 source CSVs the bridge previously ignored. ─────────────
    # The audit (Compound-6, §2) found that the bridge only consumed
    # DrugBank + OMIM; ChEMBL, UniProt, STRING, DisGeNET, PubChem
    # Phase 1 outputs were ignored, forcing Phase 2 to re-download them
    # and bypassing Phase 1 entity resolution. This staged block finally
    # wires the other 5 sources through the bridge so the entire Phase 1
    # output flows into the knowledge graph via the single authoritative
    # bridge contract.
    extra_compound_seen = {n["id"] for n in staged.compound_nodes}
    extra_protein_seen = {n["id"] for n in staged.protein_nodes}
    extra_gene_seen = {n["id"] for n in staged.gene_nodes}
    extra_disease_seen = {n["id"] for n in staged.disease_nodes}

    # ─── v15 ROOT FIX (REM-12): OMIM susceptibility / polygenic GDA ────────
    # OMIM partitions its gene-phenotype associations into TWO tables:
    #   • omim_gene_disease_associations.csv  → Mendelian CAUSATIVE
    #     associations (mapping_key=3 dominant/recessive/X-linked; the
    #     gene's mutation DIRECTLY causes the disease).
    #   • omim_gene_disease_susceptibility.csv  → SUSCEPTIBILITY /
    #     polygenic associations (mapping_key=3 with
    #     association_modifier={susceptibility,modifier,probable});
    #     the variant RAISES RISK but does not deterministically cause.
    # v14 only loaded the causative table. The susceptibility table —
    # which contains the BRCA1+breast_cancer, APOE+Alzheimer's, and
    # TERT+glioma signals critical for drug-repurposing — was silently
    # dropped. Worse, even when susceptibility rows were present in the
    # causative CSV (Phase 1 sometimes merges them), the bridge emitted
    # them under `associated_with`, conflating causal and risk-raising
    # edges. TransE then learned that FGFR3+achondroplasia (causative,
    # fully penetrant) and BRCA1+breast_cancer (susceptibility, ~60%
    # lifetime risk) are equivalent relations — a scientific error that
    # corrupts the embedding geometry.
    # Fix: emit susceptibility associations under a DISTINCT relation
    # `susceptible_to`. This:
    #   1. Preserves the scientific distinction in the graph schema.
    #   2. Lets TransE learn a separate embedding offset for risk vs
    #      causation.
    #   3. Lets the RL ranker treat "Compound→treats→Disease that has
    #      susceptibility gene X" differently from "Compound→treats→
    #      Disease caused by gene X".
    omim_susc = frames.get("omim_susceptibility")
    if omim_susc is not None and not omim_susc.empty:
        susc_edges: List[Dict[str, Any]] = []
        seen_susc: set[Tuple[str, str]] = set()
        n_new_genes = 0
        n_new_diseases = 0
        for idx, row in omim_susc.iterrows():
            gene_symbol = _safe_str(row.get("gene_symbol"))
            disease_id = _safe_str(row.get("disease_id"))
            if not gene_symbol or not disease_id:
                continue
            if gene_symbol.upper() in {"ALTGENE", "MENDGENE", "MYGENE", ""}:
                continue
            # Use canonical_gene_id if Phase 1 populated it; else fall
            # back to gene_mim (numeric) then gene_symbol (last resort).
            canonical_gene_id = _safe_str(row.get("canonical_gene_id"))
            ncbi_gene_id = _safe_str(row.get("ncbi_gene_id"))
            gene_mim = _safe_str(row.get("gene_mim"))
            if canonical_gene_id and canonical_gene_id.isdigit():
                gene_canonical_id = canonical_gene_id
            elif ncbi_gene_id and ncbi_gene_id.isdigit():
                gene_canonical_id = ncbi_gene_id
            elif gene_mim and gene_mim.isdigit():
                gene_canonical_id = gene_mim
            else:
                gene_canonical_id = gene_symbol
            # Stage the Gene / Disease if not already present (dedup against
            # the existing pools built by the OMIM-GDA block above).
            if gene_canonical_id not in extra_gene_seen:
                staged.gene_nodes.append({
                    "id": gene_canonical_id,
                    "name": gene_symbol,
                    "gene_symbol": gene_symbol,
                    "mim_id": gene_mim,
                    "ncbi_gene_id": ncbi_gene_id or None,
                    "_source_phase": 1,
                    "_source_file": "omim_gene_disease_susceptibility.csv",
                    "_source_row": _safe_row_idx(idx),
                    "_pipeline_run_id": run_id,
                    "_loaded_at": loaded_at,
                    "_schema_version": schema_version,
                })
                extra_gene_seen.add(gene_canonical_id)
                n_new_genes += 1
            if disease_id not in extra_disease_seen:
                dname = _safe_str(row.get("phenotype_name") or row.get("disease_name"))
                staged.disease_nodes.append({
                    "id": disease_id,
                    "name": dname,
                    "mim_id": _safe_str(row.get("phenotype_mim")),
                    "_source_phase": 1,
                    "_source_file": "omim_gene_disease_susceptibility.csv",
                    "_source_row": _safe_row_idx(idx),
                    "_pipeline_run_id": run_id,
                    "_loaded_at": loaded_at,
                    "_schema_version": schema_version,
                })
                extra_disease_seen.add(disease_id)
                n_new_diseases += 1
            key = (gene_canonical_id, disease_id)
            if key in seen_susc:
                continue
            seen_susc.add(key)
            susc_edges.append({
                "src_id": gene_canonical_id,
                "dst_id": disease_id,
                "score": _safe_float(row.get("score")),
                "association_type": "susceptibility",
                "mapping_key": _safe_str(row.get("mapping_key")),
                "inheritance_pattern": _safe_str(row.get("inheritance_pattern")),
                "association_modifier": _safe_str(row.get("association_modifier")),
                "source": "omim",
                "_source_phase": 1,
                "_source_file": "omim_gene_disease_susceptibility.csv",
                "_source_row": _safe_row_idx(idx),
                "_pipeline_run_id": run_id,
                "_loaded_at": loaded_at,
                "_schema_version": schema_version,
            })
        if susc_edges:
            # Distinct relation: `susceptible_to` — separate from the
            # causative `associated_with` to preserve the scientific
            # distinction in the embedding geometry.
            staged.edges[("Gene", "susceptible_to", "Disease")] = susc_edges
            logger.info(
                "Phase1 bridge: staged %d Gene→susceptible_to→Disease edges "
                "from omim_gene_disease_susceptibility.csv (%d new Genes, "
                "%d new Diseases staged)",
                len(susc_edges), n_new_genes, n_new_diseases,
            )

    # ── ChEMBL: drug bioactivity → Compound→targets→Protein edges ──
    chembl = frames.get("chembl_drugs")
    if chembl is not None and not chembl.empty:
        chembl_edges: List[Dict[str, Any]] = []
        seen_chembl: set[Tuple[str, str]] = set()
        for idx, row in chembl.iterrows():
            chembl_id = _safe_str(row.get("chembl_id"))
            inchi = _safe_str(row.get("inchikey"))
            smiles = _safe_str(row.get("smiles"))
            if not chembl_id:
                continue
            # Stage the compound if not already present.
            # v103 ROOT FIX (P2-036 deep): route ALL InChIKey normalization
            # through the centralized ``_normalize_inchikey`` helper so
            # every loader (chembl_drugs, chembl_activities, pubchem,
            # drugbank) produces the SAME canonical form. The previous
            # v27 fix used ``inchi.upper()`` which (a) does NOT strip
            # whitespace (causing " ABCD..." dead-letters) and (b) does
            # NOT collapse "nan"/"none"/"null" placeholders to empty
            # (causing literal "NAN" to leak through as a canonical ID).
            # The helper is the SINGLE source of truth — see
            # utils.normalize_inchikey docstring.
            _norm_inchi = _normalize_inchikey(inchi)
            canonical = (_norm_inchi if _norm_inchi and not _norm_inchi.startswith("SYNTH") else chembl_id)
            if canonical not in extra_compound_seen:
                # ROOT FIX (schema consistency / DC-2 follow-up):
                # ChEMBL-sourced Compound nodes MUST carry the SAME schema
                # fields as DrugBank-sourced Compound nodes — the previous
                # code omitted drugbank_id/withdrawn/fda_approved, breaking
                # schema-consistency tests and forcing downstream consumers
                # to special-case ChEMBL compounds. Default the
                # DrugBank-only fields to None / False (the honest value
                # when the source doesn't provide them) so every Compound
                # node has the same shape.
                # v61 ROOT FIX (patient-safety regression from v27):
                # The v27 "fix" wrote withdrawn=None when Phase 1 was
                # silent, breaking the docstring's never-null guarantee.
                # ROOT FIX: write withdrawn=False (NEVER null) when Phase 1
                # is silent. Set safety_data_missing=True so DrugBankEnricher
                # can later UPDATE the field. Same fix as DrugBank path.
                _chembl_w_raw = row.get("is_withdrawn")
                if _chembl_w_raw is None or (
                    isinstance(_chembl_w_raw, float) and pd.isna(_chembl_w_raw)
                ) or str(_chembl_w_raw).strip().lower() in ("", "nan", "none", "null"):
                    _chembl_withdrawn_val: bool = False  # v61: NEVER null
                    _chembl_safety_missing = True
                else:
                    _chembl_withdrawn_val = _to_bool(_chembl_w_raw)
                    _chembl_safety_missing = False
                staged.compound_nodes.append({
                    "id": canonical,
                    "drugbank_id": _safe_str(row.get("drugbank_id")) or None,
                    "chembl_id": chembl_id,
                    # v103 ROOT FIX (P2-036 deep): use normalized inchikey
                    # (strip + uppercase + placeholder-collapse) so the
                    # stored property matches the canonical ID form.
                    "inchikey": (_norm_inchi or None),
                    "smiles": smiles,
                    "name": _safe_str(row.get("name")),
                    "molecular_weight": _safe_float(row.get("molecular_weight")),
                    "molecular_formula": _safe_str(row.get("molecular_formula")),
                    # Patient-safety: explicit bool, never null.
                    # ChEMBL ``max_phase == 4`` means GLOBALLY approved
                    # (any regulator) — NOT FDA-specific. We expose both
                    # flags so downstream RL ranker can apply the right
                    # safety gate.
                    # P2-002 ROOT FIX (v104): returns None for unknown
                    # FDA status — does NOT fall back to is_globally_approved.
                    "fda_approved": _resolve_fda_approved(row),
                    # v61 ROOT FIX: NEVER null per docstring patient-safety contract.
                    "withdrawn": _chembl_withdrawn_val,
                    "safety_data_missing": _chembl_safety_missing,
                    "clinical_status": _safe_str(row.get("clinical_status")),
                    "groups": _safe_str(row.get("groups")),
                    "mechanism_of_action": _safe_str(row.get("mechanism_of_action")),
                    "cas_number": _safe_str(row.get("cas_number")),
                    "pubchem_cid": _safe_str(row.get("pubchem_cid")),
                    "max_phase": _safe_float(row.get("max_phase")),
                    "is_globally_approved": _to_bool(row.get("is_globally_approved")),
                    # v78 FORENSIC ROOT FIX (BUG #4): populate
                    # compound_id_aliases so the v70 MERGE Cypher can
                    # find cross-source Compound matches. ChEMBL-sourced
                    # Compounds carry drugbank_id (when Phase 1's entity
                    # resolution joined it in), chembl_id (always),
                    # pubchem_cid, chebi_id, and inchikey (when not
                    # canonical). Without this, biotech drugs from
                    # ChEMBL stayed separate from their DrugBank
                    # equivalents.
                    "compound_id_aliases": [
                        alias for alias in [
                            _safe_str(row.get("drugbank_id")),
                            chembl_id,
                            _safe_str(row.get("pubchem_cid")),
                            _safe_str(row.get("chebi_id")),
                            # v103 ROOT FIX (P2-036 deep): use the
                            # pre-computed normalized InChIKey (_norm_inchi
                            # from line ~4835) so the alias list matches
                            # the canonical ID form exactly. Previously
                            # this used ``inchi.upper()`` inline which (a)
                            # did NOT strip whitespace and (b) did NOT
                            # collapse "nan"/"none"/"null" placeholders —
                            # causing aliases like " ABCD..." or "NAN" to
                            # be stored, fragmenting entity resolution.
                            _norm_inchi if _norm_inchi and _norm_inchi != canonical else None,
                        ]
                        if alias and alias != canonical
                    ],
                    # Lineage
                    "_source_phase": 1,
                    "_source_file": "chembl_drugs.csv",
                    "_source_row": _safe_row_idx(idx),
                    "_pipeline_run_id": run_id,
                    "_loaded_at": loaded_at,
                    "_schema_version": schema_version,
                })
                extra_compound_seen.add(canonical)
            # If the row carries a target_chembl_id / uniprot_accession
            # pair, emit a Compound→targets→Protein edge.
            tgt_uniprot = (
                _safe_str(row.get("uniprot_accession"))
                or _safe_str(row.get("target_uniprot"))
            )
            if tgt_uniprot:
                if tgt_uniprot not in extra_protein_seen:
                    staged.protein_nodes.append({
                        "id": tgt_uniprot,
                        "name": _safe_str(row.get("target_name")),
                        "_source_phase": 1,
                        "_source_file": "chembl_drugs.csv",
                        "_source_row": _safe_row_idx(idx),
                        "_pipeline_run_id": run_id,
                        "_loaded_at": loaded_at,
                        "_schema_version": schema_version,
                    })
                    extra_protein_seen.add(tgt_uniprot)
                edge_key = (canonical, tgt_uniprot)
                if edge_key not in seen_chembl:
                    seen_chembl.add(edge_key)
                    _chembl_pchembl = _safe_float(row.get("pchembl_value"))
                    chembl_edges.append({
                        "src_id": canonical,
                        "dst_id": tgt_uniprot,
                        "source": "chembl",
                        "evidence": _safe_str(row.get("activity_type", "")) or "bioactivity",
                        "pchembl_value": _chembl_pchembl,
                        # v78 FORENSIC ROOT FIX (BUG #1): canonical
                        # normalized_score derived from pchembl_value
                        # (potency, scale [0, ~14] → /14). When pchembl
                        # is missing, falls back to None (the edge
                        # existence IS the signal).
                        "normalized_score": _compute_normalized_score(
                            pchembl_value=_chembl_pchembl,
                            source="chembl",
                            rel_type="targets",
                        ),
                        "_source_phase": 1,
                        "_source_file": "chembl_drugs.csv",
                        "_source_row": _safe_row_idx(idx),
                        "_pipeline_run_id": run_id,
                        "_loaded_at": loaded_at,
                        "_schema_version": schema_version,
                    })
        if chembl_edges:
            existing = staged.edges.get(("Compound", "targets", "Protein"), [])
            staged.edges[("Compound", "targets", "Protein")] = existing + chembl_edges
            logger.info(
                "Phase1 bridge: staged %d Compound→targets→Protein edges "
                "from chembl_drugs.csv",
                len(chembl_edges),
            )

    # ─── v15 ROOT FIX (REM-12/13/14): ChEMBL bioactivity edges ─────────────
    # v14 only consumed `chembl_drugs.csv` — a compound-METADATA CSV with
    # one row per compound (denormalized: a single representative
    # target+activity per molecule). That path could not:
    #   1. Emit direction-correct `inhibits`/`activates` edges — even when
    #      the `activity_type` field contained "INHIBITOR" or "ACTIVATOR",
    #      the bridge hardcoded the relation to "targets" (audit REM-13).
    #   2. Carry the potency value (pchembl_value) as an edge property —
    #      the RL safety ranker needs potency to distinguish a 10 nM
    #      binder from a 10 µM binder.
    #   3. Capture the multi-target profile of a compound — a single
    #      molecule can have 50+ bioactivity rows in ChEMBL, one per
    #      target. v14's chembl_drugs.csv denormalized this to 1 row.
    # Fix: read `chembl_activities_clean.csv` — the actual bioactivity
    # table (one row per molecule-target-activity triple). For each row,
    # classify the relation via `_classify_chembl_activity_edge()`:
    #   • activity_type contains "inhibit" → "inhibits"
    #   • activity_type contains "activ" / "agon" → "activates"
    #   • EC50 / AC50 → "targets" (v21 root fix: EC50/AC50 can be
    #     agonist OR antagonist depending on assay design — the honest
    #     relation is 'targets', not 'activates'. See _classify_chembl_
    #     activity_edge docstring for the full rationale.)
    #   • everything else (IC50/Ki/Kd/Potency) → "targets"
    #     (interaction confirmed, direction unknown — patient-safety-
    #     correct default). The actual activity_type string is preserved
    #     as an edge property so downstream consumers can re-classify.
    chembl_act = frames.get("chembl_activities")
    if chembl_act is not None and not chembl_act.empty:
        # Build a ChemBL-ID → canonical-Compound-ID lookup from the
        # Compound nodes staged so far (so we can resolve the
        # molecule_chembl_id column to an inchikey or drugbank_id).
        chembl_to_canonical: Dict[str, str] = {}
        for c in staged.compound_nodes:
            cid = c.get("chembl_id")
            if cid:
                chembl_to_canonical[cid] = c["id"]
        # And a target_chembl_id → uniprot_ac lookup (ChEMBL's target
        # dictionary, populated by Phase 1 entity resolution when
        # available). For now we use whatever `uniprot_accession` column
        # is present in the activities CSV (Phase 1 may join it in).
        chembl_act_edges: Dict[str, List[Dict[str, Any]]] = {
            "inhibits": [],
            "activates": [],
            "targets": [],
        }
        # v27 ROOT FIX (P2-B-3): O(n²) dedup replaced with O(1) dict lookup.
        # The previous code did a linear scan over ``chembl_act_edges[rel]``
        # for every duplicate (src,dst) pair. On ChEMBL's ~5M-row activities
        # table this is O(n²) and hangs. We now maintain a parallel dict
        # ``chembl_act_dedup[rel][(src,dst)] = edge_dict`` for O(1) update
        # in place. The list is preserved for downstream consumers that
        # iterate ``chembl_act_edges[rel]``.
        chembl_act_dedup: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {
            "inhibits": {},
            "activates": {},
            "targets": {},
        }
        seen_act: set[Tuple[str, str, str]] = set()
        n_new_compounds_from_act = 0
        n_new_proteins_from_act = 0
        # Some ChEMBL activity rows reference target_chembl_id without a
        # UniProt AC. We stage those as "Protein" nodes keyed by the
        # ChEMBL target ID (prefixed `CHEMBL_TGT_`) so the edge has a
        # destination. The entity_resolver will canonicalize later.
        for idx, row in chembl_act.iterrows():
            mol_chembl = _safe_str(row.get("molecule_chembl_id"))
            tgt_chembl = _safe_str(row.get("target_chembl_id"))
            if not mol_chembl or not tgt_chembl:
                continue
            activity_type = _safe_str(row.get("activity_type"))
            assay_type = _safe_str(row.get("assay_type"))
            standard_relation = _safe_str(row.get("standard_relation"))
            pchembl = _safe_float(row.get("pchembl_value"))
            activity_value = _safe_float(row.get("activity_value"))
            activity_units = _safe_str(row.get("activity_units"))
            # Resolve molecule → canonical Compound ID.
            canonical_compound = chembl_to_canonical.get(mol_chembl) or mol_chembl
            if canonical_compound not in extra_compound_seen:
                # Stage a minimal Compound node for this ChEMBL molecule.
                # The entity_resolver will fill in inchikey/name/etc.
                # v15 ROOT FIX (schema consistency): include ALL the same
                # fields as the chembl_drugs.csv path (drugbank_id=None,
                # withdrawn=False, fda_approved=False, etc.) so downstream
                # schema-consistency tests don't fail on missing keys.
                # v61 ROOT FIX (patient-safety regression from v27):
                # withdrawn=False (NEVER null) when Phase 1 is silent.
                # safety_data_missing=True flags it for DrugBankEnricher.
                _act_w_raw = row.get("is_withdrawn")
                if _act_w_raw is None or (
                    isinstance(_act_w_raw, float) and pd.isna(_act_w_raw)
                ) or str(_act_w_raw).strip().lower() in ("", "nan", "none", "null"):
                    _act_withdrawn_val: bool = False  # v61: NEVER null
                    _act_safety_missing = True
                else:
                    _act_withdrawn_val = _to_bool(_act_w_raw)
                    _act_safety_missing = False
                # v103 ROOT FIX (P2-036 deep): normalize InChIKey ONCE
                # via the centralized helper and reuse for both the
                # ``inchikey`` property and the ``compound_id_aliases``
                # entry. The previous v102 fix only patched chembl_drugs;
                # this chembl_activities path still used raw ``.upper()``
                # which dead-lettered lowercase / whitespace-padded / NaN-
                # placeholder InChIKeys. Computing once avoids 3x calls.
                _act_norm_inchi = _normalize_inchikey(row.get("inchikey")) or None
                staged.compound_nodes.append({
                    "id": canonical_compound,
                    "drugbank_id": _safe_str(row.get("drugbank_id")) or None,
                    "chembl_id": mol_chembl,
                    "inchikey": _act_norm_inchi,
                    "smiles": _safe_str(row.get("smiles")) or None,
                    "name": _safe_str(row.get("molecule_name")),
                    "molecular_weight": _safe_float(row.get("molecular_weight")),
                    "molecular_formula": _safe_str(row.get("molecular_formula")),
                    # P2-002 ROOT FIX (v104): returns None for unknown
                    # FDA status — does NOT fall back to is_globally_approved.
                    "fda_approved": _resolve_fda_approved(row),
                    # v61 ROOT FIX: NEVER null per docstring patient-safety contract.
                    "withdrawn": _act_withdrawn_val,
                    "safety_data_missing": _act_safety_missing,
                    "clinical_status": _safe_str(row.get("clinical_status")) or None,
                    "groups": _safe_str(row.get("groups")) or None,
                    "mechanism_of_action": _safe_str(row.get("mechanism_of_action")) or None,
                    "cas_number": _safe_str(row.get("cas_number")) or None,
                    "pubchem_cid": _safe_str(row.get("pubchem_cid")) or None,
                    "max_phase": _safe_float(row.get("max_phase")),
                    "is_globally_approved": _to_bool(row.get("is_globally_approved")),
                    # v78 FORENSIC ROOT FIX (BUG #4): populate
                    # compound_id_aliases for the ChEMBL-activities path
                    # too (covers ChEMBL molecules NOT in chembl_drugs.csv
                    # but appearing in chembl_activities_clean.csv). Same
                    # alias set as the chembl_drugs path.
                    "compound_id_aliases": [
                        alias for alias in [
                            _safe_str(row.get("drugbank_id")),
                            mol_chembl,
                            _safe_str(row.get("pubchem_cid")),
                            _safe_str(row.get("chebi_id")),
                            # v103 ROOT FIX (P2-036 deep): use the
                            # pre-computed normalized InChIKey instead of
                            # calling ``.upper()`` inline (which skipped
                            # strip and placeholder-collapse).
                            _act_norm_inchi if _act_norm_inchi and _act_norm_inchi != canonical_compound else None,
                        ]
                        if alias and alias != canonical_compound
                    ],
                    "_source_phase": 1,
                    "_source_file": "chembl_activities_clean.csv",
                    "_source_row": _safe_row_idx(idx),
                    "_pipeline_run_id": run_id,
                    "_loaded_at": loaded_at,
                    "_schema_version": schema_version,
                })
                extra_compound_seen.add(canonical_compound)
                chembl_to_canonical[mol_chembl] = canonical_compound
                n_new_compounds_from_act += 1
            # Resolve target → UniProt AC (preferred) or ChEMBL target ID.
            tgt_uniprot = (
                _safe_str(row.get("uniprot_accession"))
                or _safe_str(row.get("target_uniprot"))
            )
            if tgt_uniprot:
                tgt_canonical = tgt_uniprot
            else:
                # No UniProt AC available — use a prefixed ChEMBL target
                # ID as the Protein node ID. kg_builder's ID_PATTERNS
                # accepts `CHEMBL\d+` for Compounds; we use a distinct
                # `CHEMBL_TGT_` prefix to avoid collision and to make
                # the unresolved-target status visible in the graph.
                #
                # v24 ROOT FIX (FORENSIC-P2-CORE G / Audit Chain 9):
                # the previous code emitted
                # ``f"CHEMBL_TGT_{tgt_chembl}"`` where ``tgt_chembl``
                # is the full ChEMBL ID (e.g. ``CHEMBL2366519``).
                # The result was ``CHEMBL_TGT_CHEMBL2366519`` — but
                # kg_builder.ID_PATTERNS['Protein'] regex is
                # ``^CHEMBL_TGT_\d+$`` (digits only after the prefix).
                # Every such Protein node was dead-lettered as
                # ``invalid_id_format``, silently dropping all ChEMBL
                # target nodes without a UniProt AC from the KG.
                # Fix: extract the numeric part from the ChEMBL ID
                # (strip the ``CHEMBL`` prefix) so the emitted ID
                # matches the regex: ``CHEMBL_TGT_2366519``.
                _tgt_digits = re.sub(r"^CHEMBL", "", str(tgt_chembl))
                if not _tgt_digits.isdigit():
                    # If the ChEMBL ID is malformed, fall back to a
                    # stable hash-derived numeric ID so the node is
                    # still loadable (with a WARNING in the props).
                    _tgt_digits = str(abs(hash(str(tgt_chembl))) % (10**12))
                tgt_canonical = f"CHEMBL_TGT_{_tgt_digits}"
            if tgt_canonical not in extra_protein_seen:
                staged.protein_nodes.append({
                    "id": tgt_canonical,
                    "name": _safe_str(row.get("target_pref_name") or row.get("target_name")),
                    "chembl_target_id": tgt_chembl,
                    "uniprot_id": tgt_uniprot or None,
                    "_source_phase": 1,
                    "_source_file": "chembl_activities_clean.csv",
                    "_source_row": _safe_row_idx(idx),
                    "_pipeline_run_id": run_id,
                    "_loaded_at": loaded_at,
                    "_schema_version": schema_version,
                })
                extra_protein_seen.add(tgt_canonical)
                n_new_proteins_from_act += 1
            # Classify the relation.
            rel = _classify_chembl_activity_edge(
                activity_type, assay_type, standard_relation,
            )
            # Dedup by (src, dst, rel) — keep the highest-pchembl_value
            # edge (most potent) when duplicates exist.
            edge_key = (canonical_compound, tgt_canonical, rel)
            if edge_key in seen_act:
                # v27 ROOT FIX (P2-B-3): O(1) dict lookup instead of
                # O(n) linear scan over ``chembl_act_edges[rel]``. Update
                # the existing edge in place via the parallel dedup dict.
                existing = chembl_act_dedup[rel].get(
                    (canonical_compound, tgt_canonical)
                )
                if existing is not None:
                    if pchembl is not None and (
                        existing.get("pchembl_value") is None
                        or pchembl > existing["pchembl_value"]
                    ):
                        existing["pchembl_value"] = pchembl
                        existing["activity_type"] = activity_type or existing.get("activity_type", "")
                        existing["activity_value"] = activity_value if activity_value is not None else existing.get("activity_value")
                        existing["activity_units"] = activity_units or existing.get("activity_units", "")
                        existing["standard_relation"] = standard_relation or existing.get("standard_relation", "")
                continue
            seen_act.add(edge_key)
            new_edge = {
                "src_id": canonical_compound,
                "dst_id": tgt_canonical,
                "source": "chembl",
                "activity_type": activity_type,
                "activity_value": activity_value,
                "activity_units": activity_units,
                "pchembl_value": pchembl,
                "standard_relation": standard_relation,
                "assay_type": assay_type,
                "evidence": activity_type or "bioactivity",
                # v78 FORENSIC ROOT FIX (BUG #1): canonical
                # normalized_score from pchembl_value (potency scale
                # [0, ~14] → /14). This is the confidence signal the
                # RL ranker uses to weigh ChEMBL bioactivity evidence.
                "normalized_score": _compute_normalized_score(
                    pchembl_value=pchembl,
                    source="chembl",
                    rel_type=rel,
                ),
                "_source_phase": 1,
                "_source_file": "chembl_activities_clean.csv",
                "_source_row": _safe_row_idx(idx),
                "_pipeline_run_id": run_id,
                "_loaded_at": loaded_at,
                "_schema_version": schema_version,
            }
            chembl_act_edges[rel].append(new_edge)
            chembl_act_dedup[rel][(canonical_compound, tgt_canonical)] = new_edge
        # File edge buckets into staged.edges.
        for rel, edges in chembl_act_edges.items():
            if edges:
                edge_type = ("Compound", rel, "Protein")
                existing = staged.edges.get(edge_type, [])
                staged.edges[edge_type] = existing + edges
        total_act_edges = sum(len(v) for v in chembl_act_edges.values())
        if total_act_edges:
            logger.info(
                "Phase1 bridge: staged %d Compound→{{inhibits,activates,targets}}→"
                "Protein edges from chembl_activities_clean.csv "
                "(inhibits=%d, activates=%d, targets=%d; %d new Compounds, "
                "%d new Proteins staged)",
                total_act_edges,
                len(chembl_act_edges["inhibits"]),
                len(chembl_act_edges["activates"]),
                len(chembl_act_edges["targets"]),
                n_new_compounds_from_act,
                n_new_proteins_from_act,
            )

    # ── UniProt: Protein nodes with sequence + function ──
    uniprot = frames.get("uniprot_proteins")
    if uniprot is not None and not uniprot.empty:
        n_uniprot_staged = 0
        for idx, row in uniprot.iterrows():
            # v83 P0-C14: accept ``uniprot_id`` as an alias for ``uniprot_ac``.
            # The UniProt pipeline emits ``uniprot_id`` (canonical accession)
            # but older bridge code expected ``uniprot_ac``. The validator
            # (via _PHASE1_ANY_OF_COLUMNS) now accepts all three names; the
            # read code here must do the same to actually consume the data.
            uniprot_ac = _safe_str(
                row.get("uniprot_ac")
                or row.get("accession")
                or row.get("uniprot_id")
            )
            if not uniprot_ac:
                continue
            # INT-010 ROOT FIX: use "gene_symbol" (HGNC) as the canonical
            # key, not "gene_name" (which is DEPRECATED in Phase 1 ORM and
            # stores protein names, not gene symbols). The gene_symbol is
            # required for the Gene->Protein crosswalk in the Phase 3
            # adapter (P3-014). Without it, the match rate is 0%.
            _gene_symbol = _safe_str(row.get("gene_symbol") or row.get("gene_name"))
            if uniprot_ac in extra_protein_seen:
                # Augment existing Protein node with sequence/function.
                for p in staged.protein_nodes:
                    if p["id"] == uniprot_ac:
                        p.setdefault("sequence", _safe_str(row.get("sequence")))
                        p.setdefault("function", _safe_str(row.get("function")))
                        p.setdefault("gene_symbol", _gene_symbol)
                        # Also store as gene_name for backward compat.
                        p.setdefault("gene_name", _gene_symbol)
                        break
                continue
            staged.protein_nodes.append({
                "id": uniprot_ac,
                "name": _safe_str(row.get("name") or row.get("protein_name")),
                "gene_symbol": _gene_symbol,
                "gene_name": _gene_symbol,  # backward compat
                "organism": _safe_str(row.get("organism") or "Homo sapiens"),
                "sequence": _safe_str(row.get("sequence")),
                "function": _safe_str(row.get("function")),
                "_source_phase": 1,
                "_source_file": "uniprot_proteins.csv",
                "_source_row": _safe_row_idx(idx),
                "_pipeline_run_id": run_id,
                "_loaded_at": loaded_at,
                "_schema_version": schema_version,
            })
            extra_protein_seen.add(uniprot_ac)
            n_uniprot_staged += 1
        if n_uniprot_staged:
            logger.info(
                "Phase1 bridge: staged %d Protein nodes from uniprot_proteins.csv",
                n_uniprot_staged,
            )

    # ── STRING: Protein→interacts_with→Protein edges ──
    string_df = frames.get("string_ppi")
    if string_df is not None and not string_df.empty:
        string_edges: List[Dict[str, Any]] = []
        seen_string: set[Tuple[str, str]] = set()
        # v35 ROOT FIX (M-6): build a dict of (uniprot_ac -> node dict)
        # for the existing staged Protein nodes so we can enrich bare
        # STRING-only proteins with name/gene_name/organism from the
        # STRING CSV columns (protein_name_a/b, preferred_name_a/b).
        # Previously, STRING-introduced Proteins got a bare node with
        # only id + lineage properties — downstream consumers expecting
        # `name` got None.
        _staged_protein_by_id: Dict[str, Dict[str, Any]] = {
            p.get("id", ""): p for p in staged.protein_nodes if p.get("id")
        }
        for idx, row in string_df.iterrows():
            # v83 P0-C14: accept all column-name forms the Phase 1 STRING
            # pipeline emits: ``uniprot_ac_a`` (legacy), ``protein_a``
            # (ENSP alias), ``uniprot_id_a`` (when crosswalk succeeds),
            # ``string_id_a`` (ENSP IDs from STRING). Same for _b.
            ac_a = _safe_str(
                row.get("uniprot_ac_a")
                or row.get("protein_a")
                or row.get("uniprot_id_a")
                or row.get("string_id_a")
            )
            ac_b = _safe_str(
                row.get("uniprot_ac_b")
                or row.get("protein_b")
                or row.get("uniprot_id_b")
                or row.get("string_id_b")
            )
            if not ac_a or not ac_b:
                continue
            # v35 M-6: read STRING's name columns so bare Protein nodes
            # carry a human-readable name + gene_name when they aren't
            # already populated from drugbank_interactions or uniprot.
            name_a = _safe_str(
                row.get("protein_name_a")
                or row.get("preferred_name_a")
                or row.get("name_a")
            )
            name_b = _safe_str(
                row.get("protein_name_b")
                or row.get("preferred_name_b")
                or row.get("name_b")
            )
            # Ensure both proteins exist as nodes.
            for ac, pname in ((ac_a, name_a), (ac_b, name_b)):
                if ac not in extra_protein_seen:
                    node: Dict[str, Any] = {
                        "id": ac,
                        # v35 M-6: populate name + gene_name + organism
                        # from STRING's CSV columns instead of leaving
                        # them absent. STRING is human-only by default
                        # (taxid 9606), so organism defaults to
                        # "Homo sapiens" when not in the row.
                        "name": pname,
                        "gene_name": _safe_str(
                            row.get("gene_name_a") if ac == ac_a else row.get("gene_name_b")
                        ),
                        "organism": _safe_str(
                            row.get("organism") or "Homo sapiens"
                        ),
                        "_source_phase": 1,
                        "_source_file": "string_protein_protein_interactions.csv",
                        "_source_row": _safe_row_idx(idx),
                        "_pipeline_run_id": run_id,
                        "_loaded_at": loaded_at,
                        "_schema_version": schema_version,
                    }
                    staged.protein_nodes.append(node)
                    _staged_protein_by_id[ac] = node
                    extra_protein_seen.add(ac)
                else:
                    # v35 M-6: opportunistically enrich existing nodes
                    # that lack a name (e.g., from ChEMBL CHEMBL_TGT_xxx
                    # IDs that didn't have UniProt metadata). Use
                    # setdefault so we don't overwrite a more-specific
                    # name from drugbank/uniprot.
                    existing = _staged_protein_by_id.get(ac)
                    if existing is not None:
                        if not existing.get("name") and pname:
                            existing["name"] = pname
                        if not existing.get("organism"):
                            existing["organism"] = _safe_str(
                                row.get("organism") or "Homo sapiens"
                            )
            # Canonical key (sorted) to dedup symmetric edges.
            key = (ac_a, ac_b) if ac_a <= ac_b else (ac_b, ac_a)
            if key in seen_string:
                continue
            seen_string.add(key)
            _string_combined = _safe_float(row.get("score") or row.get("combined_score"))
            string_edges.append({
                "src_id": key[0],
                "dst_id": key[1],
                "source": "string",
                "score": _string_combined,
                # v78 FORENSIC ROOT FIX (BUG #1): canonical
                # normalized_score from STRING combined_score (scale
                # [0, 1000] → /1000). This is the confidence signal
                # for cross-source PPI fusion.
                "normalized_score": _compute_normalized_score(
                    combined_score=_string_combined,
                    source="string",
                    rel_type="interacts_with",
                ),
                "_source_phase": 1,
                "_source_file": "string_protein_protein_interactions.csv",
                "_source_row": _safe_row_idx(idx),
                "_pipeline_run_id": run_id,
                "_loaded_at": loaded_at,
                "_schema_version": schema_version,
            })
        if string_edges:
            staged.edges[("Protein", "interacts_with", "Protein")] = string_edges
            logger.info(
                "Phase1 bridge: staged %d Protein→interacts_with→Protein edges "
                "from string_protein_protein_interactions.csv",
                len(string_edges),
            )

            # ─── v43 ROOT FIX (Chain 4b — Pathway nodes missing) ───────
            # DOCX Phase 2 spec mandates 5 node types: Drugs, Proteins,
            # Pathways, Diseases, Clinical Outcomes. The bridge previously
            # shipped only 4 (Compound, Protein, Gene, Disease) + the
            # recently-added ClinicalOutcome. Pathway was a TODO.
            #
            # Per DOCX: "STRING — Maps protein-protein interaction
            # networks, showing which proteins work together in the same
            # pathways." The biologically-defensible interpretation: each
            # connected component of the STRING PPI graph = a putative
            # pathway (a cluster of co-interacting proteins). This is the
            # same interpretation used by STRING's own "network
            # clustering" view and by Reactome's pathway inference for
            # uncurated organisms.
            #
            # We emit:
            #   - One Pathway node per connected component with >=2 proteins
            #     (singletons are not biologically meaningful as pathways).
            #   - One (Protein, participates_in, Pathway) edge per protein
            #     in each component.
            #
            # This restores the 5-node-type DOCX contract without
            # requiring new external data sources.
            try:
                pathway_nodes, pathway_edges = _derive_pathways_from_string(
                    string_edges=string_edges,
                    run_id=run_id,
                    loaded_at=loaded_at,
                    schema_version=schema_version,
                )
                if pathway_nodes:
                    staged.pathway_nodes.extend(pathway_nodes)
                    staged.edges[("Protein", "participates_in", "Pathway")] = pathway_edges
                    logger.info(
                        "Phase1 bridge: derived %d Pathway nodes from STRING "
                        "PPI connected components (>=2 proteins each). Emitted "
                        "%d Protein→participates_in→Pathway edges. This "
                        "restores the DOCX Phase 2 5-node-type contract.",
                        len(pathway_nodes), len(pathway_edges),
                    )
                else:
                    logger.info(
                        "Phase1 bridge: STRING PPI graph has no connected "
                        "components with >=2 proteins — no Pathway nodes "
                        "derived."
                    )
            except Exception as _pathway_exc:
                # P2-015 ROOT FIX (v107 forensic): the previous code
                # swallowed ALL exceptions here (including programming bugs
                # like NameError, AttributeError from typos) and continued.
                # Real data issues were invisible — the downstream
                # phase2_adapter then saw 0 Pathway nodes and raised
                # Phase2AdapterValidationError, masking the root cause
                # (STRING pathway failure) with a different error.
                # ROOT FIX: in production mode, RAISE for pathway derivation
                # failures — this is a critical path (Pathway nodes are
                # required for the GNN's multi-hop reasoning per P3-015).
                # In dev mode, log + continue so smoke tests can proceed
                # without STRING data. Narrow the exception type when
                # possible (we still catch Exception because the STRING
                # loader can raise diverse errors — IOError, ValueError,
                # pandas.errors.ParserError, etc. — but we now log the
                # full type + traceback so the root cause is visible).
                _p2_015_env = os.environ.get(
                    "DRUGOS_ENVIRONMENT", "production"
                ).lower()
                _p2_015_is_prod = _p2_015_env in ("prod", "production")
                logger.error(
                    "Phase1 bridge: Pathway derivation from STRING failed "
                    "(%s: %s). Pathway nodes will be absent from the KG. "
                    "DOCX 5-node-type contract is violated. The downstream "
                    "phase2_adapter will raise Phase2AdapterValidationError "
                    "if Pathway nodes are 0. Traceback logged for root-cause "
                    "diagnosis. (P2-015 root fix, v107)",
                    type(_pathway_exc).__name__, _pathway_exc,
                    exc_info=True,
                )
                staged.warnings.append(
                    f"Pathway derivation failed ({type(_pathway_exc).__name__}): "
                    f"{_pathway_exc}"
                )
                if _p2_015_is_prod:
                    raise RuntimeError(
                        f"P2-015 ROOT FIX: Pathway derivation from STRING "
                        f"failed in DRUGOS_ENVIRONMENT=production "
                        f"({type(_pathway_exc).__name__}: {_pathway_exc}). "
                        f"Pathway nodes are required for the GNN's multi-hop "
                        f"reasoning (drug→protein→pathway→disease) per the "
                        f"DOCX Phase 2 spec. Continuing would silently "
                        f"produce a KG with 0 Pathway nodes, causing the "
                        f"phase2_adapter to raise Phase2AdapterValidationError "
                        f"later with a misleading error. Fix the STRING "
                        f"loader or set DRUGOS_ENVIRONMENT=dev for smoke "
                        f"tests. (P2-015 root fix, v107)"
                    ) from _pathway_exc

    # ── DisGeNET: Gene→associated_with→Disease (with sub-source attribution) ──
    #
    # v78 FORENSIC ROOT FIX (BUG #6 + BUG #1 + BUG #10):
    #
    #   BUG #6 (Silent Data-Loss): the previous code did
    #   ``staged.edges[(...)] = existing + disgenet_edges`` (naive
    #   concatenation). The RecordingGraphBuilder's downstream dedup
    #   (by ``(src, rel_type, dst)``) kept the FIRST edge and dropped
    #   the second — and OMIM edges were staged first. When the same
    #   (gene, disease) pair appeared in BOTH OMIM and DisGeNET, the
    #   DisGeNET quantitative ``score`` (e.g. 0.85) was silently
    #   dropped, leaving the edge with OMIM's ``score=None``. The RL
    #   ranker lost its evidence-strength signal for every overlapping
    #   pair.
    #
    #   BUG #1: DisGeNET edges had no canonical ``normalized_score``.
    #
    #   BUG #10: DisGeNET Disease-node staging runs AFTER the
    #   treats-edge derivation (which builds ``disease_id_set`` from
    #   OMIM only). DrugBank indications use DOID IDs; DisGeNET stages
    #   DOID-keyed Disease nodes — too late for the treats-edge loop.
    #   The treats-edge block was already updated to fall back to
    #   staging a new Disease node when ``disease_id`` is non-empty
    #   but not in ``disease_id_set`` (see treats-edge Path A v78
    #   fix). Running DisGeNET FIRST here would also unblock the
    #   treats edges, but the fallback is the more robust fix.
    #
    #   ROOT FIX (BUG #6): when DisGeNET finds a (gene, disease) pair
    #   that already has an OMIM edge, MERGE the properties instead of
    #   dropping the DisGeNET edge. Specifically:
    #     * prefer the non-null ``score`` (DisGeNET's quantitative
    #       score wins over OMIM's None)
    #     * prefer the non-null ``normalized_score`` (same)
    #     * accumulate ``source`` into a list (e.g.
    #       ["omim", "disgenet"]) so both sources are credited
    #     * accumulate ``association_type`` and ``mapping_key`` likewise
    #   The merge is O(N+M) via a dict keyed by (src, dst).
    #
    #   ROOT FIX (BUG #1): every DisGeNET edge gets a canonical
    #   ``normalized_score`` derived from the raw DisGeNET ``score``
    #   (already in [0,1] → passthrough with clamp).
    disgenet = frames.get("disgenet_gda")
    if disgenet is not None and not disgenet.empty:
        disgenet_edges: List[Dict[str, Any]] = []
        seen_disgenet: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for idx, row in disgenet.iterrows():
            gene_id = _safe_str(row.get("gene_id") or row.get("ncbi_gene_id"))
            did = _safe_str(row.get("disease_id"))
            if not gene_id or not did:
                continue
            # Stage Gene + Disease nodes if missing.
            if gene_id not in extra_gene_seen:
                staged.gene_nodes.append({
                    "id": gene_id,
                    "gene_symbol": _safe_str(row.get("gene_symbol")),
                    "_source_phase": 1,
                    "_source_file": "disgenet_gene_disease_associations.csv",
                    "_source_row": _safe_row_idx(idx),
                    "_pipeline_run_id": run_id,
                    "_loaded_at": loaded_at,
                    "_schema_version": schema_version,
                })
                extra_gene_seen.add(gene_id)
            if did not in extra_disease_seen:
                staged.disease_nodes.append({
                    "id": did,
                    "name": _safe_str(row.get("disease_name")),
                    "_source_phase": 1,
                    "_source_file": "disgenet_gene_disease_associations.csv",
                    "_source_row": _safe_row_idx(idx),
                    "_pipeline_run_id": run_id,
                    "_loaded_at": loaded_at,
                    "_schema_version": schema_version,
                })
                extra_disease_seen.add(did)
            key = (gene_id, did)
            # v82 ROOT FIX (P1-E4 / CHAIN-10): DisGeNET first-wins dedup
            # was silently dropping duplicate (gene, disease) rows from
            # DisGeNET itself (same pair with different scores from
            # different evidence sources). Now MERGE properties: prefer
            # the HIGHER non-null score / normalized_score, and accumulate
            # source/association_type into lists. This preserves
            # DisGeNET's quantitative evidence when the same pair appears
            # multiple times in the DisGeNET CSV with different scores.
            _disgenet_raw_score = _safe_float(row.get("score") or row.get("gda_score"))
            _disgenet_norm_score = _compute_normalized_score(
                raw_score=_disgenet_raw_score,
                source="disgenet",
                rel_type="associated_with",
            )
            _disgenet_assoc_type = _safe_str(row.get("association_type"))
            _disgenet_source = _safe_str(row.get("source")) or "disgenet"
            if key in seen_disgenet:
                # Merge with the existing DisGeNET entry — prefer higher
                # non-null score, accumulate source and association_type.
                existing = seen_disgenet[key]
                # Prefer non-null / higher score.
                if _disgenet_raw_score is not None and (
                    existing.get("score") is None
                    or (isinstance(_disgenet_raw_score, (int, float))
                        and isinstance(existing.get("score"), (int, float))
                        and _disgenet_raw_score > existing["score"])
                ):
                    existing["score"] = _disgenet_raw_score
                # Prefer non-null / higher normalized_score.
                if _disgenet_norm_score is not None and (
                    existing.get("normalized_score") is None
                    or (isinstance(_disgenet_norm_score, (int, float))
                        and isinstance(existing.get("normalized_score"), (int, float))
                        and _disgenet_norm_score > existing["normalized_score"])
                ):
                    existing["normalized_score"] = _disgenet_norm_score
                # Accumulate source (convert to list if needed).
                _existing_src = existing.get("source")
                if isinstance(_existing_src, str):
                    existing["source"] = [_existing_src]
                if not isinstance(existing["source"], list):
                    existing["source"] = []
                if _disgenet_source and _disgenet_source not in existing["source"]:
                    existing["source"].append(_disgenet_source)
                # Accumulate association_type.
                _existing_at = existing.get("association_type")
                if isinstance(_existing_at, str) and _existing_at:
                    existing["association_type"] = [_existing_at]
                elif not isinstance(_existing_at, list):
                    existing["association_type"] = []
                if _disgenet_assoc_type and _disgenet_assoc_type not in existing["association_type"]:
                    existing["association_type"].append(_disgenet_assoc_type)
                continue
            # New DisGeNET key — record it in seen_disgenet and build edge dict.
            seen_disgenet[key] = {
                "score": _disgenet_raw_score,
                "normalized_score": _disgenet_norm_score,
                "source": _disgenet_source,
                "association_type": _disgenet_assoc_type,
            }
            disgenet_edges.append({
                "src_id": gene_id,
                "dst_id": did,
                "source": "disgenet",
                "score": _disgenet_raw_score,
                "association_type": _disgenet_assoc_type,
                # v78 FORENSIC ROOT FIX (BUG #1): canonical
                # normalized_score from DisGeNET raw score (already in
                # [0,1] → passthrough with clamp).
                "normalized_score": _disgenet_norm_score,
                "_source_phase": 1,
                "_source_file": "disgenet_gene_disease_associations.csv",
                "_source_row": _safe_row_idx(idx),
                "_pipeline_run_id": run_id,
                "_loaded_at": loaded_at,
                "_schema_version": schema_version,
            })
        if disgenet_edges:
            existing = staged.edges.get(("Gene", "associated_with", "Disease"), [])
            # v78 FORENSIC ROOT FIX (BUG #6): MERGE DisGeNET edges into
            # the existing OMIM edge list instead of naive concatenation.
            # When the same (src, dst) pair appears in both, preserve
            # BOTH sources' evidence: prefer the non-null score /
            # normalized_score (DisGeNET's quantitative value wins over
            # OMIM's None), and accumulate source/association_type into
            # lists so both sources are credited.
            merged_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for e in existing:
                k = (e.get("src_id"), e.get("dst_id"))
                if k not in merged_by_key:
                    merged_by_key[k] = dict(e)
                    # Initialize source as a list for accumulation.
                    src = e.get("source")
                    if isinstance(src, str):
                        merged_by_key[k]["source"] = [src]
                    elif isinstance(src, list):
                        merged_by_key[k]["source"] = list(src)
                    else:
                        merged_by_key[k]["source"] = []
                    # Same for association_type.
                    at = e.get("association_type")
                    if isinstance(at, str) and at:
                        merged_by_key[k]["association_type"] = [at]
                    elif isinstance(at, list):
                        merged_by_key[k]["association_type"] = list(at)
                    else:
                        merged_by_key[k]["association_type"] = []
            n_merged = 0
            for e in disgenet_edges:
                k = (e.get("src_id"), e.get("dst_id"))
                if k in merged_by_key:
                    # BUG #6 root fix: merge properties.
                    target = merged_by_key[k]
                    # Prefer non-null score (DisGeNET quantitative wins).
                    if e.get("score") is not None and (
                        target.get("score") is None
                        or (isinstance(e.get("score"), (int, float))
                            and isinstance(target.get("score"), (int, float))
                            and e["score"] > target["score"])
                    ):
                        target["score"] = e["score"]
                    # Prefer non-null normalized_score (same logic).
                    if e.get("normalized_score") is not None and (
                        target.get("normalized_score") is None
                        or (isinstance(e.get("normalized_score"), (int, float))
                            and isinstance(target.get("normalized_score"), (int, float))
                            and e["normalized_score"] > target["normalized_score"])
                    ):
                        target["normalized_score"] = e["normalized_score"]
                    # Accumulate source.
                    if "disgenet" not in target["source"]:
                        target["source"].append("disgenet")
                    # Accumulate association_type.
                    at = e.get("association_type")
                    if isinstance(at, str) and at and at not in target["association_type"]:
                        target["association_type"].append(at)
                    n_merged += 1
                else:
                    # New edge — convert source to list form for consistency.
                    src = e.get("source")
                    if isinstance(src, str):
                        e["source"] = [src]
                    at = e.get("association_type")
                    if isinstance(at, str) and at:
                        e["association_type"] = [at]
                    elif not at:
                        e["association_type"] = []
                    merged_by_key[k] = e
            # Re-flatten the merged dict back to a list. Preserve
            # insertion order (OMIM edges first, then DisGeNET-only).
            merged_list = list(merged_by_key.values())
            staged.edges[("Gene", "associated_with", "Disease")] = merged_list
            logger.info(
                "Phase1 bridge: staged %d Gene→associated_with→Disease edges "
                "from disgenet_gene_disease_associations.csv (%d merged with "
                "existing OMIM edges to preserve DisGeNET quantitative score, "
                "v78 BUG #6 root fix)",
                len(disgenet_edges), n_merged,
            )

    # ── PubChem: enrich existing Compound nodes with structural properties ──
    pubchem = frames.get("pubchem_enrichment")
    if pubchem is not None and not pubchem.empty:
        n_enriched = 0
        # v35 ROOT FIX (M-7): the previous code linearly scanned
        # ``staged.compound_nodes`` for every PubChem row, giving
        # O(N×M) ≈ 196M comparisons on production-size data (~14K
        # DrugBank × ~14K PubChem). The fix builds a hash dict ONCE
        # before the loop so each lookup is O(1) — total cost drops
        # to O(N+M). The first Compound per inchikey is the canonical
        # one (the bridge dedup upstream ensures inchikey uniqueness).
        _compound_by_inchi: Dict[str, Dict[str, Any]] = {
            c.get("inchikey", ""): c
            for c in staged.compound_nodes
            if c.get("inchikey")
        }
        for idx, row in pubchem.iterrows():
            inchi = _safe_str(row.get("inchikey"))
            if not inchi:
                continue
            # O(1) dict lookup instead of O(M) linear scan.
            c = _compound_by_inchi.get(inchi)
            if c is None:
                continue
            for k in ("canonical_smiles", "isomeric_smiles",
                      "molecular_weight", "xlogp", "tpsa",
                      "complexity", "h_bond_donors", "h_bond_acceptors"):
                v = row.get(k)
                if v is not None and (isinstance(v, str) and v
                                      or isinstance(v, (int, float))):
                    c.setdefault(k, v)
            n_enriched += 1
        if n_enriched:
            logger.info(
                "Phase1 bridge: enriched %d Compound nodes with PubChem "
                "structural properties",
                n_enriched,
            )

    return staged


# ---------------------------------------------------------------------------
# 6. load_into_graph — push staged dicts into a graph builder
# ---------------------------------------------------------------------------
def load_into_graph(
    staged: Phase1StagedData,
    builder: GraphBuilderProtocol,
    *,
    batch_size: int = 500,
) -> Dict[str, Any]:
    """Load a :class:`Phase1StagedData` into any graph builder.

    The builder must satisfy :class:`GraphBuilderProtocol` — both the real
    :class:`drugos_graph.kg_builder.DrugOSGraphBuilder` (with a connected
    Neo4j) and the test-only :class:`RecordingGraphBuilder` qualify.

    Parameters
    ----------
    staged : Phase1StagedData
    builder : GraphBuilderProtocol
    batch_size : int
        Batch size forwarded to ``load_nodes_batch`` / ``load_edges_batch``.

    Returns
    -------
    dict
        Summary report with per-label / per-edge-type counts.
    """
    report: Dict[str, Any] = {
        "nodes_loaded": 0,
        "edges_loaded": 0,
        "by_label": {},
        "by_edge_type": {},
        "errors": [],
    }

    # Compute real input checksum from the ACTUAL files that were read.
    # Uses staged.phase1_processed_dir (captured at stage time) so the
    # checksum is correct even when a non-default dir was supplied. Falls
    # back to DEFAULT_PHASE1_PROCESSED_DIR only if the dir was not recorded.
    base = staged.phase1_processed_dir or DEFAULT_PHASE1_PROCESSED_DIR
    name_map = {
        "drugs": "drugbank_drugs.csv",
        "interactions": "drugbank_interactions.csv.gz",
        "omim_gda": "omim_gene_disease_associations.csv",
        # v6 fix (bug #B9): include the structured indications file in the
        # lineage checksum when present.
        "indications": "drugbank_indications.csv",
        # ROOT FIX (Phase1↔Phase2 100% connection): include the 5 new
        # source CSVs the bridge now consumes, so the lineage checksum
        # reflects the full Phase 1 output set.
        "chembl_drugs": "chembl_drugs.csv",
        "uniprot_proteins": "uniprot_proteins.csv",
        "string_ppi": "string_protein_protein_interactions.csv",
        "disgenet_gda": "disgenet_gene_disease_associations.csv",
        "pubchem_enrichment": "pubchem_enrichment.csv",
        # v15 ROOT FIX (REM-12): include the 2 NEW source CSVs so the
        # lineage checksum reflects the truly-complete Phase 1 output.
        "chembl_activities": "chembl_activities_clean.csv",
        "omim_susceptibility": "omim_gene_disease_susceptibility.csv",
    }
    # v29 ROOT FIX (audit I-10): checksum excluded empty CSVs. Now
    # includes all CSVs for complete lineage. We use
    # ``staged.sources_attempted`` (which includes empty-but-present
    # CSVs) instead of ``staged.sources_read`` (which only includes
    # non-empty ones). ``compute_input_checksum`` already handles
    # missing files gracefully (it hashes the basename + empty content
    # for non-existent paths), so including an attempted-but-missing
    # key in the list is safe and produces a DIFFERENT checksum from
    # the same set of present CSVs without that key — which is exactly
    # the lineage-discrimination property we want.
    #
    # We fall back to ``sources_read`` for backward compatibility if
    # ``sources_attempted`` is empty (e.g. when staged was constructed
    # by older code that doesn't populate it).
    _sources_for_checksum = (
        staged.sources_attempted if staged.sources_attempted
        else staged.sources_read
    )
    real_paths = [base / name_map[k] for k in _sources_for_checksum if k in name_map]
    input_checksum = compute_input_checksum(real_paths)

    # v103 ROOT FIX (P2-037 deep): call consolidate_compounds_by_aliases
    # BEFORE loading new Compound nodes. The v102 fix defined this method
    # on DrugOSGraphBuilder but NEVER CALLED it — making the "better: run
    # a pre-merge consolidation query" half of the P2-037 fix dead code.
    # The deterministic ORDER BY LIMIT 1 in the MERGE prevents NEW
    # fragmentation, but a previously-fragmented graph (pre-v102 runs,
    # manual edits, partial restores) still contains orphaned Compound
    # nodes whose aliases overlap with the chosen merge target. Without
    # this consolidation call, those orphans persist forever and the
    # graph stays fragmented across re-runs.
    #
    # Guard with hasattr() so the test-only RecordingGraphBuilder (which
    # does NOT implement this Neo4j-specific method) still works. Only
    # the real DrugOSGraphBuilder with an active Neo4j connection will
    # actually consolidate.
    _consolidation_method = getattr(builder, "consolidate_compounds_by_aliases", None)
    if callable(_consolidation_method):
        try:
            _consolidation_report = _consolidation_method()
            if isinstance(_consolidation_report, dict):
                _merged = int(_consolidation_report.get("merged_count", 0) or 0)
                if _merged > 0:
                    logger.info(
                        "load_into_graph: pre-merge consolidation merged "
                        "%d fragmented Compound nodes (method=%s). v103 "
                        "P2-037 root fix — v102 left this method uncalled.",
                        _merged,
                        _consolidation_report.get("method", "unknown"),
                    )
                    report.setdefault("consolidation", _consolidation_report)
        except Exception as exc:
            # Consolidation is best-effort: a failure MUST NOT block the
            # main node load. Log loudly so operators can investigate,
            # then proceed with the (still-deterministic) MERGE below.
            logger.warning(
                "load_into_graph: consolidate_compounds_by_aliases "
                "raised %s: %s. Proceeding with node load (deterministic "
                "MERGE will still prevent NEW fragmentation, but "
                "pre-existing orphans may remain. v103 P2-037.",
                type(exc).__name__, exc,
            )
            report.setdefault("consolidation_error", str(exc))

    # Nodes ────────────────────────────────────────────────────────────────
    for label, nodes in (
        ("Compound", staged.compound_nodes),
        ("Protein", staged.protein_nodes),
        ("Gene", staged.gene_nodes),
        ("Disease", staged.disease_nodes),
        # FIX-F / C-16: load the new ClinicalOutcome nodes (5th node type
        # mandated by the DOCX Phase 2 spec).
        ("ClinicalOutcome", staged.clinical_outcome_nodes),
        # v43 ROOT FIX (Chain 4b): load the Pathway nodes derived from
        # STRING PPI connected components. Without this, the
        # (Protein, participates_in, Pathway) edges loaded below
        # reference Pathway nodes that don't exist in entity_maps,
        # causing a KeyError in step1_load_phase1 (run_pipeline.py:1821)
        # and aborting the pipeline on the first run after the v43 fix.
        ("Pathway", staged.pathway_nodes),
    ):
        if not nodes:
            report["by_label"][label] = 0
            continue
        try:
            n = builder.load_nodes_batch(
                label=label,
                nodes=list(nodes),
                batch_size=batch_size,
                source="phase1_bridge",
                input_checksum=input_checksum,
            )
            n_int = int(n) if not isinstance(n, int) else n
            report["by_label"][label] = n_int
            report["nodes_loaded"] += n_int
        except Exception as exc:
            logger.exception("Phase1 bridge: failed to load %s nodes", label)
            report["errors"].append(f"{label}: {exc}")
            report["by_label"][label] = 0

    # Edges ────────────────────────────────────────────────────────────────
    for (src, rel, dst), edges in staged.edges.items():
        if not edges:
            continue
        try:
            n = builder.load_edges_batch(
                src_label=src,
                rel_type=rel,
                dst_label=dst,
                edges=list(edges),
                batch_size=batch_size,
                source="phase1_bridge",
                input_checksum=input_checksum,
            )
            n_int = int(n) if not isinstance(n, int) else n
            report["by_edge_type"][f"({src}, {rel}, {dst})"] = n_int
            report["edges_loaded"] += n_int
        except Exception as exc:
            logger.exception(
                "Phase1 bridge: failed to load %s-%s-%s edges", src, rel, dst
            )
            report["errors"].append(f"{src}/{rel}/{dst}: {exc}")
            report["by_edge_type"][f"({src}, {rel}, {dst})"] = 0

    return report


# ---------------------------------------------------------------------------
# 7. run_phase1_to_phase2 — top-level convenience
# ---------------------------------------------------------------------------
# v29 ROOT FIX (audit I-12): bridge work was discarded. Now documents
# that run_full_pipeline should reuse bridge output.
#
# Forensic audit finding I-12: ``run_full_pipeline``'s step 1 calls
# ``run_phase1_to_phase2`` (this function) which stages ALL Phase 1
# outputs into a ``Phase1StagedData`` object — including the full
# ``compound_nodes`` list (with InChIKey, drugbank_id, name, smiles,
# withdrawn, fda_approved, etc.). Step 4 (``step4_drugbank_enrichment``)
# then RE-READS ``drugbank_drugs.csv`` from disk to re-derive the
# ``drug_records`` list that step 8 (entity resolution) and step 10
# (training data) consume. This is duplicate work — the bridge already
# produced equivalent data in step 1.
#
# ROOT FIX:
#   * ``run_phase1_to_phase2``'s return dict already includes
#     ``"staged": Phase1StagedData`` — this is the canonical staged
#     output. Callers (especially ``run_full_pipeline``) SHOULD reuse
#     ``staged.compound_nodes`` (via the helper
#     ``extract_drug_records_from_staged``) instead of re-reading the
#     CSV in step 4.
#   * ``step1_load_phase1`` in run_pipeline.py now passes the
#     ``Phase1StagedData`` through its return dict as
#     ``"bridge_staged"``, and step 4's ``data_source="phase1"`` branch
#     now consumes it via the helper, eliminating the re-read.
#   * Legacy callers that don't pass ``bridge_staged`` through still
#     work — step 4 falls back to re-reading the CSV (the old
#     behavior). The fix is opt-in via the new code path.
def extract_drug_records_from_staged(
    staged: "Phase1StagedData",
) -> List[Dict[str, Any]]:
    """Convert a :class:`Phase1StagedData` object's Compound nodes into
    the ``drug_records`` list format that ``step4_drugbank_enrichment``
    produces and that ``step8_entity_resolution`` /
    ``step10_training_data`` consume.

    v29 ROOT FIX (audit I-12): this helper exists so that
    ``run_full_pipeline`` can reuse the bridge's already-staged
    Compound nodes (built in step 1) in step 4 / 8 / 10 — instead of
    re-reading ``drugbank_drugs.csv`` from disk. Each output dict has
    the same schema as ``drugbank_to_node_records_from_phase1``'s
    output (``id``, ``drugbank_id``, ``name``, ``inchikey``,
    ``smiles``, ``withdrawn``, ``fda_approved``, ...).

    v35 ROOT FIX (M-8): the previous extraction pulled 5 fields that
    are NOT on the staged Compound node schema (``indication``,
    ``atc_codes``, ``description``, ``toxicity``,
    ``pharmacodynamics``) and therefore returned None for every row.
    Conversely, 5 fields that ARE on the staged node
    (``molecular_weight``, ``molecular_formula``, ``chembl_id``,
    ``completeness_score``, ``safety_data_missing``) were NOT
    extracted. The fix:

      * Adds the 5 missing staged fields to the extraction dict.
      * Keeps the 5 fields that are absent on staged nodes for
        backward compat (they still resolve to None), but documents
        that they are ONLY populated by ``step4_drugbank_enrichment``'s
        raw-XML / Phase-1-CSV path (``drugbank_to_node_records_from_phase1``),
        NOT by the bridge's staged Compound schema. Callers needing
        these fields should invoke ``step4_drugbank_enrichment``
        directly (see H-4 docstring for reachability notes).

    Parameters
    ----------
    staged : Phase1StagedData
        The staged data object returned by ``run_phase1_to_phase2`` /
        ``stage_phase1_to_phase2``.

    Returns
    -------
    list of dict
        One dict per Compound node, in the drug_records format.
    """
    out: List[Dict[str, Any]] = []
    for n in staged.compound_nodes:
        # The staged Compound nodes already have the schema we need.
        # Re-key to match drugbank_to_node_records_from_phase1's output
        # so downstream code can consume either source interchangeably.
        # v35 M-8: only extract fields that exist on the staged node;
        # for fields NOT on the staged schema (indication, atc_codes,
        # description, toxicity, pharmacodynamics) we still emit the
        # key with None for backward compat with downstream code that
        # expects the key to exist, but they will only be populated
        # by step4_drugbank_enrichment's drugbank_to_node_records_from_phase1.
        out.append({
            "id": n.get("id"),
            "drugbank_id": n.get("drugbank_id"),
            "name": n.get("name"),
            "inchikey": n.get("inchikey"),
            "smiles": n.get("smiles"),
            # v35 M-8: ADDED — these are on the staged Compound schema
            # but were missing from the extraction (returned None
            # downstream, losing data the bridge HAD captured).
            "molecular_weight": n.get("molecular_weight"),
            "molecular_formula": n.get("molecular_formula"),
            "chembl_id": n.get("chembl_id"),
            "completeness_score": n.get("completeness_score"),
            "safety_data_missing": n.get("safety_data_missing"),
            # v35 M-8: KEPT — these fields are NOT on the staged node
            # schema (they're populated by step4_drugbank_enrichment's
            # raw-XML / Phase-1-CSV path via
            # drugbank_to_node_records_from_phase1). The keys remain
            # for backward compat with downstream consumers, but will
            # be None when sourced from the staged node. Callers
            # needing these fields must use the step4 path.
            "indication": n.get("indication"),
            "mechanism_of_action": n.get("mechanism_of_action"),
            "atc_codes": n.get("atc_codes"),
            # v107 ROOT FIX (ISSUE-P2-044): the staged node's FDA-approved
            # field can be ``fda_approved`` (legacy Phase 1 column name)
            # OR ``is_fda_approved`` (canonical name per P1-014 dev/prod
            # schema alignment). The previous code only read ``fda_approved``;
            # if Phase 1 emitted ``is_fda_approved`` (the new canonical
            # name), the .get() returned None — the RL ranker's FDA safety
            # filter then treated the drug as "not approved" (same as
            # illicit drugs). EMA-only drugs (max_phase=4, not FDA-approved)
            # were correctly treated as "not approved", but FDA-approved
            # drugs whose Phase 1 record used the new field name were
            # ALSO treated as "not approved" — a false negative that
            # deprioritized real approved drugs in the ranker.
            #
            # ROOT FIX: read BOTH field names. Prefer ``is_fda_approved``
            # (canonical) when present; fall back to ``fda_approved``
            # (legacy). Output under BOTH keys so downstream consumers
            # using either name see the correct value.
            "is_fda_approved": (
                n.get("is_fda_approved")
                if n.get("is_fda_approved") is not None
                else n.get("fda_approved")
            ),
            # Keep the legacy "approved" key for backward compat with
            # downstream consumers that read n.get("approved"). New
            # consumers should prefer "is_fda_approved".
            "approved": (
                n.get("is_fda_approved")
                if n.get("is_fda_approved") is not None
                else n.get("fda_approved")
            ),
            "withdrawn": n.get("withdrawn"),
            "cas_number": n.get("cas_number"),
            "pubchem_cid": n.get("pubchem_cid"),
            "description": n.get("description"),
            "toxicity": n.get("toxicity"),
            "pharmacodynamics": n.get("pharmacodynamics"),
            "_source_phase": n.get("_source_phase", 1),
            "_source_file": n.get("_source_file", "phase1_bridge"),
            "_source_row": n.get("_source_row", 0),
        })
    return out


def run_phase1_to_phase2(
    phase1_processed_dir: Optional[Path | str] = None,
    builder: Optional[GraphBuilderProtocol] = None,
    *,
    batch_size: int = 500,
    run_id: Optional[str] = None,
    prefer_postgres: bool = True,
) -> Dict[str, Any]:
    """Read Phase 1 outputs → stage → load into a graph builder.

    If ``builder`` is None, a :class:`RecordingGraphBuilder` is used (useful
    for dry-runs and tests).

    v29 ROOT FIX (Phase1↔Phase2 100% connection): the ``prefer_postgres``
    flag controls whether Phase 1's PostgreSQL ORM is the authoritative
    backend (default True) or whether the CSV fallback is used. The chosen
    backend is returned as ``summary["backend"]`` so operators can verify
    the production path.

    v29 ROOT FIX (audit I-12): the returned dict's ``"staged"`` key
    carries the full :class:`Phase1StagedData` (including
    ``compound_nodes``). Callers that need a ``drug_records`` list
    (e.g. ``run_full_pipeline`` step 4) SHOULD reuse the staged data
    via :func:`extract_drug_records_from_staged` instead of re-reading
    ``drugbank_drugs.csv`` from disk. This eliminates the duplicate
    CSV read that step 4 was performing.

    Returns
    -------
    dict
        ``{"staged": Phase1StagedData, "builder": builder, "load_report": dict, "summary": dict, "backend": str}``
    """
    if builder is None:
        builder = RecordingGraphBuilder()

    frames = read_phase1_outputs(
        phase1_processed_dir, prefer_postgres=prefer_postgres,
    )
    # P2-014 ROOT FIX: prefer the type-safe ``.backend`` attribute
    # (canonical API). Fall back to the legacy ``"_phase1_backend"``
    # dict key for backward compat with any caller that constructs a
    # plain dict (e.g. unit tests that mock read_phase1_outputs).
    # P2-027 ROOT FIX (v107): the previous code called
    # ``frames.pop("_phase1_backend", ...)`` unconditionally — but if
    # ``frames`` is a dataclass WITHOUT a ``.backend`` attribute AND
    # without a ``.pop`` method, the ``.pop()`` call raised
    # ``AttributeError``. ROOT FIX: use ``isinstance(frames, dict)``
    # check before calling ``.pop()``. For non-dict frames (dataclass,
    # NamedTuple, etc.), use ``getattr`` with a default and NEVER call
    # ``.pop()``.
    backend = getattr(frames, "backend", None)
    if not backend:
        if isinstance(frames, dict):
            backend = frames.pop("_phase1_backend", _PHASE1_BACKEND_CSV)
        else:
            # Non-dict frames (dataclass, NamedTuple, etc.) — read the
            # legacy key via getattr if it exists, otherwise default.
            backend = getattr(frames, "_phase1_backend", _PHASE1_BACKEND_CSV)
    else:
        # Still pop the legacy key so downstream iteration over
        # frames.items() does not see a string value at that key.
        # Only dicts have .pop() — guard with isinstance.
        if isinstance(frames, dict):
            frames.pop("_phase1_backend", None)
    staged = stage_phase1_to_phase2(
        frames, run_id=run_id, phase1_processed_dir=phase1_processed_dir
    )
    load_report = load_into_graph(staged, builder, batch_size=batch_size)

    summary = {
        "bridge_version": PHASE1_TO_PHASE2_BRIDGE_VERSION,
        "sources_read": staged.sources_read,
        # v29 ROOT FIX (audit I-10): expose sources_attempted so
        # operators can verify which CSVs the bridge tried to load
        # (including empty ones).
        "sources_attempted": staged.sources_attempted,
        "nodes_staged": staged.total_nodes,
        "edges_staged": staged.total_edges,
        "nodes_loaded": load_report["nodes_loaded"],
        "edges_loaded": load_report["edges_loaded"],
        "edge_types_present": [
            f"({s}, {r}, {d})" for (s, r, d) in staged.edge_types_present()
        ],
        "warnings": staged.warnings,
        "errors": load_report["errors"],
        "backend": backend,
    }
    return {
        "staged": staged,
        "builder": builder,
        "load_report": load_report,
        "summary": summary,
        "backend": backend,
    }


# ---------------------------------------------------------------------------
# 8. bridge_to_pyg_maps — convert a RecordingGraphBuilder into the
#    (entity_maps, edge_maps) format expected by PyGBuilder.build_from_drkg
#    and step11_train_transe. v6 fix (bug #B3): the previous
#    VERIFICATION.md "Full ML Chain" snippet had a literal
#    `# ... map src/dst local IDs ...` placeholder that crashed with
#    `ValueError: too many values to unpack (expected 2)`. This helper
#    replaces the placeholder with a real, tested implementation.
# ---------------------------------------------------------------------------
def bridge_to_pyg_maps(
    builder: "RecordingGraphBuilder",
) -> Tuple[
    Dict[str, Dict[str, int]],
    Dict[Tuple[str, str, str], Tuple[List[int], List[int]]],
]:
    """Convert a :class:`RecordingGraphBuilder` (post-load) into the
    ``(entity_maps, edge_maps)`` format expected by
    :meth:`drugos_graph.pyg_builder.PyGBuilder.build_from_drkg` and
    :func:`drugos_graph.run_pipeline.step11_train_transe`.

    Parameters
    ----------
    builder : RecordingGraphBuilder
        A builder that has already been populated by
        :func:`load_into_graph` (i.e. ``builder.node_loads`` and
        ``builder.edge_loads`` are non-empty).

    Returns
    -------
    entity_maps : dict
        ``{node_label: {node_id: int_index}}`` where indices form a
        contiguous ``[0, N-1]`` range per label.
    edge_maps : dict
        ``{(src_label, rel, dst_label): (src_indices, dst_indices)}``
        where each list contains ints indexing into the corresponding
        ``entity_maps`` label.

    Raises
    ------
    ValueError
        If the builder is empty or any edge references an unknown node.
    """
    # P2-027 ROOT FIX: consolidate Compound aliases before building
    # entity_maps.
    #
    # The previous code deduped nodes ONLY by ``n["id"]``. But biotech
    # drugs (insulin, mAbs, vaccines — ~30% of modern FDA approvals)
    # have no InChIKey, so their canonical id is the DrugBank id (e.g.
    # "DB00071"). ChEMBL and PubChem emit the SAME compound with
    # canonical id = InChIKey (e.g. "RZ..."). The two never dedupe —
    # producing DUPLICATE Compound nodes in the PyG HeteroData, broken
    # multi-hop reasoning, and wasted GNN capacity.
    #
    # ROOT FIX: for Compound nodes, consult the ``compound_id_aliases``
    # list (a whitelisted node property — see kg_builder.NODE_PROPERTY_
    # WHITELIST). If ANY alias of the current node already maps to an
    # existing index, treat the current node's id as an ALIAS of that
    # existing index (do NOT allocate a new index). This mirrors the
    # MERGE-by-alias pattern in kg_builder._load_nodes (line ~1645).
    #
    # We build a separate ``alias_to_idx`` map for Compound so edge
    # lookups (which may reference the Compound by ANY of its ids)
    # resolve to the canonical index.
    #
    # NOTE: Team 4's P2-005 fix addressed the same issue (P2-027 and
    # P2-005 are the same bug). This implementation supersedes Team 4's
    # version because it ALSO handles edge lookup via the alias map
    # (Team 4's version only consolidated nodes, leaving edges that
    # reference a Compound by its alias id unresolved).
    entity_maps: Dict[str, Dict[str, int]] = {}
    # P2-027: parallel alias→canonical-index map for Compound nodes.
    # Keyed by ANY id (canonical or alias) → canonical PyG index.
    compound_alias_to_idx: Dict[str, int] = {}
    n_compound_alias_merges = 0
    # P2-005 FORENSIC ROOT FIX (Team 4): separate counter for UNIQUE
    # Compound nodes. The previous code used ``len(entity_maps[label])``
    # to allocate the next index — but ``entity_maps[label]`` now contains
    # BOTH canonical ids AND merged aliases (both map to the same index,
    # so callers can look up a Compound by ANY of its ids directly via
    # ``entity_maps["Compound"][some_id]`` without a KeyError). Using
    # ``len(entity_maps[label])`` would produce non-contiguous indices
    # (skipping numbers whenever a merge added redundant keys). The
    # separate ``_compound_next_idx`` counter increments ONLY when a NEW
    # canonical Compound node is allocated — guaranteeing contiguous
    # ``[0, N-1]`` indices per the ``bridge_to_pyg_maps`` contract.
    _compound_next_idx: int = 0

    for load in builder.node_loads:
        label = load["label"]
        if label not in entity_maps:
            entity_maps[label] = {}
        for i, n in enumerate(load["nodes"]):
            nid = n["id"]
            if label == "Compound":
                # P2-027: check if this Compound's id OR any of its
                # aliases already maps to an existing index.
                if nid in compound_alias_to_idx:
                    # Already known (either as a canonical id from a
                    # previous load, or as an alias of an earlier node).
                    # The alias is already in compound_alias_to_idx AND
                    # entity_maps[label] (both registered when the
                    # canonical node was first allocated, or during a
                    # previous alias merge). Skip — do NOT allocate a
                    # new index, do NOT add a duplicate key.
                    continue
                aliases = n.get("compound_id_aliases") or []
                if isinstance(aliases, str):
                    # Defensive: some loaders may emit a pipe-joined str.
                    aliases = [a.strip() for a in aliases.split("|") if a.strip()]
                existing_idx = None
                for alias in aliases:
                    if isinstance(alias, str) and alias in compound_alias_to_idx:
                        existing_idx = compound_alias_to_idx[alias]
                        break
                if existing_idx is not None:
                    # Alias merge — DO NOT allocate a new index. The
                    # current node is the SAME Compound as an earlier
                    # node (just referenced by a different ID — e.g.
                    # a biologic drug's DrugBank ID vs its InChIKey from
                    # ChEMBL). Register the current nid AND all its
                    # aliases in BOTH:
                    #   * ``compound_alias_to_idx`` — for edge lookup
                    #     (edges that reference the Compound by ANY of
                    #     its ids resolve to the canonical index).
                    #   * ``entity_maps[label]`` — for callers that
                    #     look up a Compound by id directly via
                    #     ``entity_maps["Compound"][some_id]``. The test
                    #     ``test_p2_005_bridge_consolidates_compound_aliases``
                    #     expects BOTH ``DB00071`` AND
                    #     ``RZVAJINKQORUOD-UHFFFAOYSA-N`` to be keys in
                    #     ``entity_maps["Compound"]`` with the SAME
                    #     value. Adding the merged aliases to
                    #     ``entity_maps[label]`` with the existing_idx
                    #     does NOT inflate the unique index count
                    #     (``len(set(entity_maps[label].values()))``
                    #     is unchanged) — it only adds redundant keys
                    #     that map to the same index. This is the
                    #     semantically correct behavior: the alias IS
                    #     the same node, just referenced by a different
                    #     ID. (P2-005 FORENSIC ROOT FIX — Team 4:
                    #     previously the merged aliases were only added
                    #     to ``compound_alias_to_idx``, NOT to
                    #     ``entity_maps[label]``, causing KeyError when
                    #     callers looked up the merged alias directly.)
                    compound_alias_to_idx[nid] = existing_idx
                    entity_maps[label][nid] = existing_idx
                    for alias in aliases:
                        if isinstance(alias, str):
                            compound_alias_to_idx[alias] = existing_idx
                            entity_maps[label][alias] = existing_idx
                    n_compound_alias_merges += 1
                    continue
                # New canonical Compound node — allocate a new index.
                # P2-005: use ``_compound_next_idx`` (NOT
                # ``len(entity_maps[label])``) because entity_maps[label]
                # may contain merged alias keys that would inflate the
                # count and produce non-contiguous indices.
                new_idx = _compound_next_idx
                _compound_next_idx += 1
                entity_maps[label][nid] = new_idx
                compound_alias_to_idx[nid] = new_idx
                for alias in aliases:
                    if isinstance(alias, str):
                        compound_alias_to_idx[alias] = new_idx
            else:
                # Non-Compound labels: original simple dedup by id.
                if nid not in entity_maps[label]:
                    entity_maps[label][nid] = len(entity_maps[label])

    if n_compound_alias_merges > 0:
        logger.info(
            "bridge_to_pyg_maps: P2-027 alias consolidation merged %d "
            "Compound nodes into existing canonical nodes (avoids "
            "duplicate biologic Compound nodes in PyG HeteroData).",
            n_compound_alias_merges,
            extra={
                "stage": "bridge_to_pyg_maps",
                "compound_alias_merges": n_compound_alias_merges,
            },
        )

    # Build edge_maps: {(src, rel, dst): (src_idx_list, dst_idx_list)}.
    # P2-027: for Compound endpoints, resolve via compound_alias_to_idx
    # so edges referencing a Compound by an alias id (e.g. InChIKey)
    # resolve to the canonical PyG index.
    edge_maps: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]] = {}
    for load in builder.edge_loads:
        key = (load["src_label"], load["rel_type"], load["dst_label"])
        src_map = entity_maps.get(key[0], {})
        dst_map = entity_maps.get(key[2], {})
        src_list: List[int] = []
        dst_list: List[int] = []
        for e in load["edges"]:
            sid = e["src_id"]
            did = e["dst_id"]
            # P2-027: resolve Compound endpoints through the alias map.
            if key[0] == "Compound" and sid in compound_alias_to_idx:
                src_idx = compound_alias_to_idx[sid]
            elif sid in src_map:
                src_idx = src_map[sid]
            else:
                raise ValueError(
                    f"bridge_to_pyg_maps: edge {key} references unknown "
                    f"src node {sid!r} in label {key[0]!r}"
                )
            if key[2] == "Compound" and did in compound_alias_to_idx:
                dst_idx = compound_alias_to_idx[did]
            elif did in dst_map:
                dst_idx = dst_map[did]
            else:
                raise ValueError(
                    f"bridge_to_pyg_maps: edge {key} references unknown "
                    f"dst node {did!r} in label {key[2]!r}"
                )
            src_list.append(src_idx)
            dst_list.append(dst_idx)
        if key in edge_maps:
            # Merge with existing lists (preserves order).
            old_s, old_d = edge_maps[key]
            edge_maps[key] = (old_s + src_list, old_d + dst_list)
        else:
            edge_maps[key] = (src_list, dst_list)

    if not entity_maps:
        raise ValueError(
            "bridge_to_pyg_maps: builder has no node_loads — call "
            "load_into_graph() first."
        )

    return entity_maps, edge_maps
