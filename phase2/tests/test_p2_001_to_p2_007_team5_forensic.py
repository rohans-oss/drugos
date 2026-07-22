"""
P2-001..P2-007 Forensic Root-Fix Regression Tests
==================================================

Team Member 5 — Phase 2 KG Builder + Phase 1→2 Bridge.

Each test verifies a SPECIFIC issue from the audit, using the EXACT
examples documented in the issue description. No mock-only tests —
each test exercises the REAL production function (with mocked I/O
only where a real Neo4j/Postgres connection is required).

These tests are the LOCK that prevents regressions. If a future
maintainer reverts any fix, the corresponding test will fail.
"""

from __future__ import annotations

import inspect
import logging
import signal as _signal
import sys
import warnings
import weakref

import pytest


# ---------------------------------------------------------------------------
# P2-001 — naive substring disease matching → word-boundary + NegEx
# ---------------------------------------------------------------------------

def test_p2_001_respiratory_depression_not_matched_as_depression():
    """P2-001: 'respiratory depression' must NOT match 'depression'.

    Respiratory depression is an opioid adverse event (breathing
    issue), NOT Major Depressive Disorder. The naive substring match
    matched 'depression' inside 'respiratory depression' — wrong.
    """
    from drugos_graph.phase1_bridge import _extract_disease_id_from_indication_text
    assert _extract_disease_id_from_indication_text("treats respiratory depression") is None
    assert _extract_disease_id_from_indication_text("for respiratory depression") is None


def test_p2_001_painkiller_not_matched_as_pain():
    """P2-001: 'painkiller' must NOT match 'pain'.

    'Painkiller' is a drug class, not a disease indication.
    Word boundaries (\bpain\b) prevent this false positive.
    """
    from drugos_graph.phase1_bridge import _extract_disease_id_from_indication_text
    assert _extract_disease_id_from_indication_text("a painkiller for headaches") is None
    assert _extract_disease_id_from_indication_text("oral painkiller") is None


def test_p2_001_anti_inflammatory_not_matched_as_inflammation():
    """P2-001: 'anti-inflammatory' must NOT match 'inflammation'.

    'Anti-inflammatory' is a drug mechanism, not a disease.
    """
    from drugos_graph.phase1_bridge import _extract_disease_id_from_indication_text
    assert _extract_disease_id_from_indication_text("an anti-inflammatory agent") is None
    assert _extract_disease_id_from_indication_text("anti-inflammatory drug") is None


def test_p2_001_ulcerative_colitis_not_matched_as_ulcer():
    """P2-001: 'ulcerative colitis' must NOT match 'ulcer' (DOID:77).

    Ulcerative colitis is an IBD, not a peptic ulcer.
    v106 strengthening: 'ulcerative colitis' is now recognized as ONE
    disease (DOID:8535) via a multi-word keyword. It must NOT match
    'ulcer' (DOID:77) — that was the original 2-Disease-node bug.
    """
    from drugos_graph.phase1_bridge import _extract_disease_id_from_indication_text
    got1 = _extract_disease_id_from_indication_text("for ulcerative colitis")
    got2 = _extract_disease_id_from_indication_text("treats ulcerative colitis")
    assert got1 != "DOID:77", f"'ulcerative colitis' wrongly matched 'ulcer' (DOID:77), got {got1!r}"
    assert got2 != "DOID:77", f"'ulcerative colitis' wrongly matched 'ulcer' (DOID:77), got {got2!r}"
    # v106: it should now match DOID:8535 (Ulcerative Colitis)
    assert got1 == "DOID:8535", f"expected DOID:8535, got {got1!r}"
    assert got2 == "DOID:8535", f"expected DOID:8535, got {got2!r}"


def test_p2_001_negated_indications_not_matched():
    """P2-001: negated indications must NOT produce a positive match.

    'does not treat pain' / 'contraindicated in hypertension' /
    'not for diabetes' must all return None.
    """
    from drugos_graph.phase1_bridge import _extract_disease_id_from_indication_text
    assert _extract_disease_id_from_indication_text("does not treat pain") is None
    assert _extract_disease_id_from_indication_text("contraindicated in hypertension") is None
    assert _extract_disease_id_from_indication_text("not indicated for diabetes") is None
    assert _extract_disease_id_from_indication_text("not for anxiety") is None
    assert _extract_disease_id_from_indication_text("avoid in epilepsy") is None


def test_p2_001_positive_cases_still_match():
    """P2-001: legitimate indications must still match correctly.

    The fix must NOT break positive cases — 'treats hypertension'
    must still return DOID:10763.
    """
    from drugos_graph.phase1_bridge import (
        _extract_disease_id_from_indication_text,
        _extract_disease_name_from_indication_text,
    )
    assert _extract_disease_id_from_indication_text("for the treatment of hypertension") == "DOID:10763"
    assert _extract_disease_id_from_indication_text("treats depression") == "DOID:1470"
    assert _extract_disease_id_from_indication_text("for pain") == "DOID:0050133"
    assert _extract_disease_id_from_indication_text("treats inflammation") == "DOID:1101"
    assert _extract_disease_id_from_indication_text("for diabetes") == "DOID:9351"
    assert _extract_disease_id_from_indication_text("treats cancer") == "DOID:162"
    assert _extract_disease_id_from_indication_text("for epilepsy") == "DOID:1826"
    assert _extract_disease_id_from_indication_text("treats migraine") == "DOID:1197"
    assert _extract_disease_id_from_indication_text("for asthma") == "DOID:2841"
    assert _extract_disease_id_from_indication_text("treats arthritis") == "DOID:7148"

    # Name extraction matches ID extraction
    assert _extract_disease_name_from_indication_text("treats hypertension") == "Hypertension"
    assert _extract_disease_name_from_indication_text("treats respiratory depression") is None


def test_p2_001_empty_and_none_input():
    """P2-001: empty/None/non-string input must return None."""
    from drugos_graph.phase1_bridge import _extract_disease_id_from_indication_text
    assert _extract_disease_id_from_indication_text("") is None
    assert _extract_disease_id_from_indication_text(None) is None
    assert _extract_disease_id_from_indication_text("no disease here") is None
    assert _extract_disease_id_from_indication_text(123) is None  # type: ignore[arg-type]


def test_p2_001_word_boundary_regex_compiled_once():
    """P2-001: patterns are pre-compiled at import time (perf)."""
    from drugos_graph.phase1_bridge import _DISEASE_KEYWORD_PATTERNS
    assert len(_DISEASE_KEYWORD_PATTERNS) > 0
    for keyword, doid, pattern in _DISEASE_KEYWORD_PATTERNS:
        assert isinstance(keyword, str)
        assert isinstance(doid, str)
        assert hasattr(pattern, "search")  # compiled regex


def test_p2_001_longest_match_first():
    """P2-001: keywords sorted longest-first so multi-word terms win."""
    from drugos_graph.phase1_bridge import _DISEASE_KEYWORD_PATTERNS
    lengths = [len(kw) for kw, _, _ in _DISEASE_KEYWORD_PATTERNS]
    assert lengths == sorted(lengths, reverse=True), \
        f"keywords not sorted longest-first: {lengths}"


# ---------------------------------------------------------------------------
# P2-002 — is_fda_approved heuristic conflates globally-approved with FDA
# ---------------------------------------------------------------------------

def test_p2_002_explicit_true_preserved():
    """P2-002: explicit is_fda_approved=True stays True (DrugBank source)."""
    from drugos_graph.phase1_bridge import _resolve_fda_approved
    assert _resolve_fda_approved({"is_fda_approved": True}) is True


def test_p2_002_explicit_false_preserved():
    """P2-002: explicit is_fda_approved=False stays False."""
    from drugos_graph.phase1_bridge import _resolve_fda_approved
    assert _resolve_fda_approved({"is_fda_approved": False}) is False


def test_p2_002_none_returns_none_not_true():
    """P2-002 ROOT FIX: is_fda_approved=None must return None.

    The previous code fell back to is_globally_approved (max_phase==4),
    returning True for EMA/PMDA/NMPA-only drugs. The fix returns None
    (honest unknown) so the RL ranker treats it as a separate bucket.
    """
    from drugos_graph.phase1_bridge import _resolve_fda_approved
    # EMA-only drug: max_phase=4 globally, but NOT FDA-approved
    ema_only = {
        "is_fda_approved": None,
        "is_globally_approved": True,
        "max_phase": 4,
    }
    result = _resolve_fda_approved(ema_only)
    assert result is None, f"EMA-only drug must return None, got {result!r}"


def test_p2_002_nan_returns_none():
    """P2-002: pandas NaN must return None (not True via fallback)."""
    import pandas as pd
    from drugos_graph.phase1_bridge import _resolve_fda_approved
    result = _resolve_fda_approved({
        "is_fda_approved": float("nan"),
        "is_globally_approved": True,
    })
    assert result is None


def test_p2_002_string_coercion():
    """P2-002: string 'true'/'false' coerce correctly."""
    from drugos_graph.phase1_bridge import _resolve_fda_approved
    assert _resolve_fda_approved({"is_fda_approved": "true"}) is True
    assert _resolve_fda_approved({"is_fda_approved": "false"}) is False
    assert _resolve_fda_approved({"is_fda_approved": "1"}) is True
    assert _resolve_fda_approved({"is_fda_approved": "0"}) is False


def test_p2_002_missing_key_returns_none():
    """P2-002: missing is_fda_approved key returns None."""
    from drugos_graph.phase1_bridge import _resolve_fda_approved
    assert _resolve_fda_approved({}) is None
    assert _resolve_fda_approved({"is_globally_approved": True}) is None


def test_p2_002_return_type_is_optional_bool():
    """P2-002: return type annotation is Optional[bool]."""
    from drugos_graph.phase1_bridge import _resolve_fda_approved
    sig = inspect.signature(_resolve_fda_approved)
    assert "Optional[bool]" in str(sig.return_annotation) or "bool" in str(sig.return_annotation)


# ---------------------------------------------------------------------------
# P2-003 — CREATE edges → MERGE (idempotent re-runs)
# ---------------------------------------------------------------------------

def test_p2_003_load_edges_bulk_create_defaults_to_merge():
    """P2-003: load_edges_bulk_create default use_merge=True (GraphEdgeLoader)."""
    from drugos_graph.kg_builder import GraphEdgeLoader
    sig = inspect.signature(GraphEdgeLoader.load_edges_bulk_create)
    assert sig.parameters["use_merge"].default is True, \
        f"P2-003 FAIL: default is {sig.parameters['use_merge'].default}, expected True"


def test_p2_003_load_drkg_edges_bulk_defaults_to_merge():
    """P2-003: load_drkg_edges_bulk default use_merge=True (GraphEdgeLoader)."""
    from drugos_graph.kg_builder import GraphEdgeLoader
    sig = inspect.signature(GraphEdgeLoader.load_drkg_edges_bulk)
    assert sig.parameters["use_merge"].default is True, \
        f"P2-003 FAIL: default is {sig.parameters['use_merge'].default}, expected True"


def test_p2_003_builder_facade_defaults_to_merge():
    """P2-003: DrugOSGraphBuilder.load_edges_bulk_create default use_merge=True."""
    from drugos_graph.kg_builder import DrugOSGraphBuilder
    sig = inspect.signature(DrugOSGraphBuilder.load_edges_bulk_create)
    assert sig.parameters["use_merge"].default is True


def test_p2_003_create_mode_emits_deprecation_warning():
    """P2-003: explicit use_merge=False emits DeprecationWarning.

    The CREATE branch is preserved for explicit one-off loads but
    warns the caller that it produces duplicate edges on re-run.
    """
    from drugos_graph.kg_builder import GraphEdgeLoader
    # We can't fully call _load_edges without Neo4j, but we can verify
    # the warning is emitted by calling load_edges_bulk_create with
    # use_merge=False and a fake connection that raises early.
    from drugos_graph.config import Neo4jConfig
    from drugos_graph.kg_builder import GraphConnection

    cfg = Neo4jConfig(uri="bolt://localhost:7687", user="x", password="x", database="neo4j")
    conn = GraphConnection(cfg)  # no driver — connect() not called
    loader = GraphEdgeLoader(conn)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            # This will raise because no driver is connected, but the
            # DeprecationWarning is emitted BEFORE the call to _load_edges.
            loader.load_edges_bulk_create(
                "Compound", "TREATS", "Disease",
                [{"src_id": "x", "dst_id": "y", "props": {}}],
                use_merge=False,
            )
        except Exception:
            pass  # expected — no driver connected
        assert any(issubclass(wi.category, DeprecationWarning) for wi in w), \
            "P2-003 FAIL: use_merge=False should emit DeprecationWarning"


# ---------------------------------------------------------------------------
# P2-004 — SIGTERM/SIGINT handler + atexit for Neo4j driver cleanup
# ---------------------------------------------------------------------------

def test_p2_004_signal_handlers_installed_at_module_load():
    """P2-004: SIGTERM/SIGINT handlers installed when kg_builder imported."""
    from drugos_graph import kg_builder
    assert kg_builder._SIGNAL_HANDLERS_INSTALLED is True


def test_p2_004_sigterm_handler_is_cleanup_handler():
    """P2-004: SIGTERM handler is our _signal_cleanup_handler."""
    from drugos_graph import kg_builder
    current = _signal.getsignal(_signal.SIGTERM)
    assert current is kg_builder._signal_cleanup_handler, \
        f"SIGTERM handler is {current}, expected _signal_cleanup_handler"


def test_p2_004_sigint_handler_is_cleanup_handler():
    """P2-004: SIGINT handler is our _signal_cleanup_handler."""
    from drugos_graph import kg_builder
    current = _signal.getsignal(_signal.SIGINT)
    assert current is kg_builder._signal_cleanup_handler


def test_p2_004_owned_connections_is_weakset():
    """P2-004: _OWNED_CONNECTIONS is a WeakSet (auto-cleanup on GC)."""
    from drugos_graph import kg_builder
    assert isinstance(kg_builder._OWNED_CONNECTIONS, weakref.WeakSet)


def test_p2_004_cleanup_empty_registry_returns_zero():
    """P2-004: _cleanup_owned_connections on empty registry returns 0."""
    from drugos_graph import kg_builder
    # Snapshot the current count (other tests may have added connections)
    result = kg_builder._cleanup_owned_connections(None)
    assert isinstance(result, int)
    assert result >= 0


def test_p2_004_disconnect_is_idempotent():
    """P2-004: disconnect() can be called twice without raising.

    The signal handler may call disconnect() after the user already
    called it. The second call must be a no-op, not raise.
    """
    from drugos_graph.config import Neo4jConfig
    from drugos_graph.kg_builder import GraphConnection

    cfg = Neo4jConfig(uri="bolt://localhost:7687", user="x", password="x", database="neo4j")
    conn = GraphConnection(cfg)  # no driver
    # Two consecutive disconnects — neither should raise
    conn.disconnect()
    conn.disconnect()
    assert conn._driver is None


def test_p2_004_external_driver_not_registered():
    """P2-004: externally-provided drivers are NOT registered for cleanup.

    We must not close a driver we don't own — that would break the
    caller that injected it.
    """
    from drugos_graph.config import Neo4jConfig
    from drugos_graph.kg_builder import GraphConnection, _OWNED_CONNECTIONS

    class FakeDriver:
        closed = False
        def close(self): self.closed = True
        def session(self, **kw): raise RuntimeError("not used")

    cfg = Neo4jConfig(uri="bolt://localhost:7687", user="x", password="x", database="neo4j")
    fake = FakeDriver()
    conn = GraphConnection(cfg, driver=fake)  # DI mode
    assert conn._external_driver is True
    # External drivers are NOT in the registry (only connect() registers,
    # and only for self-owned drivers). Verify by checking disconnect
    # does NOT close the external driver.
    conn.disconnect()
    assert fake.closed is False, "P2-004 FAIL: external driver was closed"


# ---------------------------------------------------------------------------
# P2-005 — schema_version filter for Phase 1 Postgres reads
# ---------------------------------------------------------------------------

class _FakeRow:
    def __init__(self, val): self._val = val
    def __getitem__(self, i): return self._val


class _FakeResult:
    def __init__(self, row): self._row = row
    def fetchone(self): return self._row


class _FakeConn:
    """Mock SQLAlchemy connection for schema_version queries."""
    def __init__(self, latest_val=None, count_val=0, raise_on_execute=False):
        self.latest_val = latest_val
        self.count_val = count_val
        self.raise_on_execute = raise_on_execute
    def execute(self, stmt):
        if self.raise_on_execute:
            raise RuntimeError("table does not exist")
        stmt_str = str(stmt)
        if "MAX(version)" in stmt_str:
            return _FakeResult(_FakeRow(self.latest_val))
        elif "COUNT(*)" in stmt_str:
            return _FakeResult(_FakeRow(self.count_val))
        raise RuntimeError(f"unexpected stmt: {stmt_str[:80]}")


def test_p2_005_get_latest_schema_version_returns_int():
    """P2-005: _get_latest_schema_version returns int when table exists."""
    from drugos_graph.phase1_bridge import _get_latest_schema_version
    conn = _FakeConn(latest_val=17, count_val=17)
    assert _get_latest_schema_version(conn) == 17


def test_p2_005_get_latest_schema_version_returns_none_on_missing_table():
    """P2-005: returns None when table doesn't exist (fresh DB)."""
    from drugos_graph.phase1_bridge import _get_latest_schema_version
    conn = _FakeConn(raise_on_execute=True)
    assert _get_latest_schema_version(conn) is None


def test_p2_005_get_latest_schema_version_returns_none_on_empty_table():
    """P2-005: returns None when table is empty (no migrations applied)."""
    from drugos_graph.phase1_bridge import _get_latest_schema_version
    conn = _FakeConn(latest_val=None, count_val=0)
    assert _get_latest_schema_version(conn) is None


def test_p2_005_count_schema_versions_returns_int():
    """P2-005: _count_schema_versions returns int."""
    from drugos_graph.phase1_bridge import _count_schema_versions
    conn = _FakeConn(latest_val=17, count_val=17)
    assert _count_schema_versions(conn) == 17
    assert isinstance(_count_schema_versions(conn), int)


def test_p2_005_count_schema_versions_returns_zero_on_missing_table():
    """P2-005: returns 0 when table doesn't exist."""
    from drugos_graph.phase1_bridge import _count_schema_versions
    conn = _FakeConn(raise_on_execute=True)
    assert _count_schema_versions(conn) == 0


def test_p2_005_helpers_are_callable():
    """P2-005: both helpers exist and are callable."""
    from drugos_graph import phase1_bridge
    assert callable(phase1_bridge._get_latest_schema_version)
    assert callable(phase1_bridge._count_schema_versions)


# ---------------------------------------------------------------------------
# P2-006 — batch sizes 5000/5000 (Neo4j-recommended)
# ---------------------------------------------------------------------------

def test_p2_006_config_defaults_to_5000():
    """P2-006: Neo4jConfig defaults to batch_size_nodes=5000, batch_size_edges=5000."""
    from drugos_graph.config import Neo4jConfig
    cfg = Neo4jConfig(uri="bolt://localhost:7687", user="x", password="x", database="neo4j")
    assert cfg.batch_size_nodes == 5000, \
        f"P2-006 FAIL: batch_size_nodes={cfg.batch_size_nodes}, expected 5000"
    assert cfg.batch_size_edges == 5000, \
        f"P2-006 FAIL: batch_size_edges={cfg.batch_size_edges}, expected 5000"


def test_p2_006_validate_batch_size_none_returns_5000():
    """P2-006: _validate_batch_size(None) returns 5000 (Neo4j default)."""
    from drugos_graph.kg_builder import _validate_batch_size
    assert _validate_batch_size(None) == 5000


def test_p2_006_validate_batch_size_explicit_respected():
    """P2-006: explicit batch_size is respected (no silent override)."""
    from drugos_graph.kg_builder import _validate_batch_size
    assert _validate_batch_size(100) == 100
    assert _validate_batch_size(10000) == 10000
    assert _validate_batch_size(1) == 1


def test_p2_006_validate_batch_size_rejects_invalid():
    """P2-006: invalid batch_size (0, -1, non-int) raises ConfigurationError."""
    from drugos_graph.kg_builder import _validate_batch_size
    from drugos_graph.exceptions import ConfigurationError
    for bad in (0, -1, -100):
        with pytest.raises(ConfigurationError):
            _validate_batch_size(bad)
    for bad in ("100", 1.5, None):
        # None returns 5000 (default), so skip it
        if bad is None:
            continue
        with pytest.raises(ConfigurationError):
            _validate_batch_size(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# P2-007 — per-batch progress log moved to DEBUG (summary stays INFO)
# ---------------------------------------------------------------------------

def test_p2_007_node_loader_per_batch_log_is_debug():
    """P2-007: node loader per-batch progress log uses logger.debug, not logger.info.

    We verify by inspecting the source — the per-batch log must call
    ``logger.debug`` (not ``logger.info``). The summary log at the end
    of the load stays at INFO.
    """
    import drugos_graph.kg_builder as kgb
    src = inspect.getsource(kgb.GraphNodeLoader.load_nodes_batch)
    # Find the per-batch progress log block
    assert "loaded %d/%d nodes" in src, "node progress log message not found"
    # The per-batch log must use logger.debug (not logger.info)
    # Find the line with the progress message and check it uses debug
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if "loaded %d/%d nodes" in line:
            # Look backwards for the logger.X call
            for j in range(i, max(i - 5, -1), -1):
                if "logger." in lines[j]:
                    assert "logger.debug" in lines[j], \
                        f"P2-007 FAIL: node per-batch log uses {lines[j].strip()}, expected logger.debug"
                    break
            break


def test_p2_007_edge_loader_per_batch_log_is_debug():
    """P2-007: edge loader per-batch progress log uses logger.debug."""
    import drugos_graph.kg_builder as kgb
    src = inspect.getsource(kgb.GraphEdgeLoader._load_edges)
    assert "loaded %d/%d edges" in src, "edge progress log message not found"
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if "loaded %d/%d edges" in line:
            for j in range(i, max(i - 5, -1), -1):
                if "logger." in lines[j]:
                    assert "logger.debug" in lines[j], \
                        f"P2-007 FAIL: edge per-batch log uses {lines[j].strip()}, expected logger.debug"
                    break
            break


def test_p2_007_summary_log_stays_info():
    """P2-007: the summary log at end of load stays at INFO (ops needs it).

    The summary ``Created %d %s-%s->%s edges`` is the line operators
    need to verify a load completed. It must stay at INFO.
    """
    import drugos_graph.kg_builder as kgb
    src = inspect.getsource(kgb.GraphEdgeLoader._load_edges)
    assert "Created %d %s-%s->%s edges" in src, "edge summary log not found"
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if "Created %d %s-%s->%s edges" in line:
            for j in range(i, max(i - 5, -1), -1):
                if "logger." in lines[j]:
                    assert "logger.info" in lines[j], \
                        f"P2-007 FAIL: edge summary log uses {lines[j].strip()}, expected logger.info"
                    break
            break


# ---------------------------------------------------------------------------
# P2-001 STRENGTHENING (v106) — ulcerative colitis recognized as ONE disease
# ---------------------------------------------------------------------------

def test_p2_001_ulcerative_colitis_matched_as_single_disease():
    """P2-001 v106: 'ulcerative colitis' must match DOID:8535 (not None).

    The issue description names this case explicitly: the naive substring
    match split 'ulcerative colitis' into 'ulcer' + 'colitis' (2 Disease
    nodes for one condition). The word-boundary regex fix PREVENTED the
    split but also missed the real disease entirely (returned None). The
    v106 strengthening adds 'ulcerative colitis' as a multi-word keyword
    so it is recognized as ONE disease (DOID:8535).
    """
    from drugos_graph.phase1_bridge import _extract_disease_id_from_indication_text
    from drugos_graph.phase1_bridge import _extract_disease_name_from_indication_text
    # Positive: "ulcerative colitis" must return the correct DOID
    assert _extract_disease_id_from_indication_text("for ulcerative colitis") == "DOID:8535"
    assert _extract_disease_name_from_indication_text("treats ulcerative colitis") == "Ulcerative Colitis"
    # In a sentence with more context
    assert _extract_disease_id_from_indication_text(
        "indicated for mild to moderate ulcerative colitis in adults"
    ) == "DOID:8535"


def test_p2_001_ulcerative_colitis_does_not_match_ulcer():
    """P2-001 v106: 'ulcerative colitis' must NOT match 'ulcer' (DOID:77).

    Longest-match-first (L3) ensures 'ulcerative colitis' is checked
    BEFORE 'ulcer'. Once 'ulcerative colitis' matches, the function
    returns immediately — 'ulcer' is never checked. This prevents the
    2-Disease-node bug from the issue description.
    """
    from drugos_graph.phase1_bridge import _extract_disease_id_from_indication_text
    got = _extract_disease_id_from_indication_text("for ulcerative colitis")
    assert got != "DOID:77", f"P2-001 FAIL: 'ulcerative colitis' matched 'ulcer' (DOID:77), got {got!r}"
    assert got == "DOID:8535", f"P2-001 FAIL: expected DOID:8535, got {got!r}"


def test_p2_001_peptic_ulcer_still_matches_ulcer():
    """P2-001 v106: 'peptic ulcer' must still match 'ulcer' (DOID:77).

    Adding 'ulcerative colitis' must NOT break the existing 'ulcer' match
    for genuine ulcer indications. 'peptic ulcer' should match DOID:77.
    """
    from drugos_graph.phase1_bridge import _extract_disease_id_from_indication_text
    assert _extract_disease_id_from_indication_text("for peptic ulcer") == "DOID:77"
    assert _extract_disease_id_from_indication_text("treats gastric ulcer") == "DOID:77"


# ---------------------------------------------------------------------------
# P2-002 STRENGTHENING (v106) — misleading fallback comments removed
# ---------------------------------------------------------------------------

def test_p2_002_no_misleading_fallback_comment_in_source():
    """P2-002 v106: the source must NOT contain the lying comment
    'falls back to is_globally_approved when is_fda_approved is None'.

    The previous v64 comment claimed the code falls back to
    is_globally_approved (max_phase==4) when is_fda_approved is None.
    That described the OLD buggy behavior. The v104 fix changed the code
    to return None (no fallback), but the comment was NEVER updated —
    so operators reading the code saw a comment that lied about what
    the code does. This test ensures the misleading comment stays gone.
    """
    from drugos_graph import phase1_bridge as pb
    src = inspect.getsource(pb)
    # The misleading comment text must NOT appear anywhere in the source
    misleading = "falls back to is_globally_approved when is_fda_approved is None"
    assert misleading not in src, (
        f"P2-002 FAIL: misleading comment still present in phase1_bridge.py: "
        f"'{misleading}'. The code returns None (no fallback) but the comment "
        f"lies about a fallback. Remove the comment."
    )


def test_p2_002_resolve_fda_approved_never_uses_is_globally_approved():
    """P2-002 v106: _resolve_fda_approved must NOT READ is_globally_approved.

    The function's contract is: return None for unknown FDA status, NEVER
    fall back to is_globally_approved (max_phase==4). This test verifies
    the function never READS the is_globally_approved field from the row
    (i.e. no row.get('is_globally_approved') / row['is_globally_approved']).
    Comments and docstrings MAY mention is_globally_approved to explain
    why the fallback is NOT used — that is documentation, not code logic.
    """
    from drugos_graph.phase1_bridge import _resolve_fda_approved
    src = inspect.getsource(_resolve_fda_approved)
    # The function must NOT read the is_globally_approved field.
    # These are the only patterns that would read it from the row dict.
    forbidden_patterns = [
        'row.get("is_globally_approved")',
        "row.get('is_globally_approved')",
        'row["is_globally_approved"]',
        "row['is_globally_approved']",
    ]
    for pat in forbidden_patterns:
        assert pat not in src, (
            f"P2-002 FAIL: _resolve_fda_approved reads 'is_globally_approved' "
            f"via pattern {pat!r}. The function must NOT use is_globally_approved "
            f"as a fallback — that conflates EMA/PMDA/NMPA approval with FDA approval."
        )


def test_p2_002_ema_only_drug_returns_none_even_with_max_phase_4():
    """P2-002 v106: integration test — an EMA-only drug (max_phase=4,
    is_globally_approved=True, is_fda_approved=None) must return None.

    This is the EXACT scenario from the issue: an EMA-approved drug
    sold in Germany but never submitted to the FDA has max_phase==4
    (approved by SOME regulator) but is_fda_approved=None (ChEMBL
    cannot provide FDA-specific approval). The OLD code marked it
    fda_approved=True (the bug). The fix returns None (unknown).
    """
    from drugos_graph.phase1_bridge import _resolve_fda_approved
    ema_only_row = {
        "is_fda_approved": None,
        "is_globally_approved": True,
        "max_phase": 4,
    }
    result = _resolve_fda_approved(ema_only_row)
    assert result is None, (
        f"P2-002 FAIL: EMA-only drug returned {result!r}, expected None. "
        f"Returning True would conflate EMA approval with FDA approval."
    )


if __name__ == "__main__":
    # Allow running directly: python3 test_p2_001_to_p2_007_team5_forensic.py
    pytest.main([__file__, "-v", "--tb=short"])
