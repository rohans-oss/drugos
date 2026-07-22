"""Regression tests for P2-013: opentargets_loader.py GraphQL pagination.

P2-013 ROOT FIX: the OpenTargets GraphQL API returns at most 10,000
associations per request. For well-studied diseases (breast cancer has
50,000+), a single-request client silently truncates the result. The
fix adds ``fetch_opentargets_associations`` which follows the
``cursor`` field in the response until all rows are fetched.

The tests use mocked urlopen (no network calls) to verify:
  * A 3-page response returns all rows.
  * The cursor is followed correctly.
  * Truncation is logged when max_pages is reached before completion.
  * Empty diseases short-circuit (no infinite loop).
  * Invalid disease IDs raise RuntimeError.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "phase2"))

from drugos_graph.opentargets_loader import (  # noqa: E402
    _build_opentargets_associations_query,
    fetch_opentargets_associations,
    OPENTARGETS_GRAPHQL_ENDPOINT,
)


def _make_graphql_response(
    rows: list[dict],
    cursor: str | None,
    count: int,
) -> dict:
    """Build a mock OpenTargets GraphQL response payload."""
    return {
        "data": {
            "disease": {
                "id": "EFO_0000311",
                "name": "breast cancer",
                "associatedTargets": {
                    "count": count,
                    "cursor": cursor,
                    "rows": rows,
                },
            },
        },
    }


def _make_target_row(target_id: str, symbol: str, score: float) -> dict:
    return {
        "target": {"id": target_id, "approvedSymbol": symbol},
        "score": score,
        "datatypeScores": [
            {"id": "genetic_association", "score": score * 0.5},
            {"id": "known_drug", "score": score * 0.3},
        ],
    }


class TestP2013QueryBuilder:
    """Tests for ``_build_opentargets_associations_query``."""

    def test_query_without_cursor(self) -> None:
        q = _build_opentargets_associations_query("EFO_0000311", 5000)
        assert "EFO_0000311" in q
        assert "BSize: 5000" in q
        assert "after:" not in q  # no cursor on first page

    def test_query_with_cursor(self) -> None:
        q = _build_opentargets_associations_query(
            "EFO_0000311", 5000, cursor="abc123",
        )
        assert 'after: "abc123"' in q
        assert "BSize: 5000" in q


class TestP2013FetchWithMockedPagination:
    """Tests for ``fetch_opentargets_associations`` with mocked urlopen."""

    def test_3_page_pagination_test(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P2-013 spec: CI test with a 3-page response."""
        page_responses = [
            _make_graphql_response(
                rows=[_make_target_row(f"ENSG00000{i}", f"GENE{i}", 0.9 - i * 0.1)],
                cursor=f"page{i+1}_cursor",
                count=3,  # 3 total rows across 3 pages
            )
            for i in range(2)  # pages 1 and 2 have cursors
        ] + [
            _make_graphql_response(
                rows=[_make_target_row("ENSG000002", "GENE2", 0.7)],
                cursor=None,  # last page -- no cursor
                count=3,
            ),
        ]

        call_count = [0]

        class _MockResponse:
            def __init__(self, payload: dict) -> None:
                self._payload = payload

            def getcode(self) -> int:
                return 200

            def read(self) -> bytes:
                return json.dumps(self._payload).encode("utf-8")

            def __enter__(self) -> "_MockResponse":
                return self

            def __exit__(self, *args: object) -> None:
                pass

        def mock_urlopen(req, timeout=None, context=None):  # noqa: ANN001
            idx = min(call_count[0], len(page_responses) - 1)
            call_count[0] += 1
            return _MockResponse(page_responses[idx])

        import drugos_graph.opentargets_loader as otl
        monkeypatch.setattr(otl.urllib.request, "urlopen", mock_urlopen)

        results = fetch_opentargets_associations(
            "EFO_0000311", max_pages=10, page_size=1,
        )
        # 3 pages * 1 row per page = 3 rows.
        assert len(results) == 3, (
            f"Expected 3 associations across 3 pages, got {len(results)}"
        )
        # Each row must have the page_fetched field populated.
        assert results[0]["page_fetched"] == 1
        assert results[1]["page_fetched"] == 2
        assert results[2]["page_fetched"] == 3
        # The mock must have been called 3 times.
        assert call_count[0] == 3

    def test_stops_when_cursor_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pagination must stop when cursor is null."""
        single_page = _make_graphql_response(
            rows=[_make_target_row("ENSG1", "GENE1", 0.9)],
            cursor=None,
            count=1,
        )
        call_count = [0]

        class _MockResponse:
            def __init__(self, payload: dict) -> None:
                self._payload = payload

            def getcode(self) -> int:
                return 200

            def read(self) -> bytes:
                return json.dumps(self._payload).encode("utf-8")

            def __enter__(self) -> "_MockResponse":
                return self

            def __exit__(self, *args: object) -> None:
                pass

        def mock_urlopen(req, timeout=None, context=None):  # noqa: ANN001
            call_count[0] += 1
            return _MockResponse(single_page)

        import drugos_graph.opentargets_loader as otl
        monkeypatch.setattr(otl.urllib.request, "urlopen", mock_urlopen)

        results = fetch_opentargets_associations("EFO_0000311", max_pages=10)
        assert len(results) == 1
        assert call_count[0] == 1  # only 1 request

    def test_truncation_warning_when_max_pages_exceeded(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When max_pages is reached before all rows are fetched, a
        truncation WARNING must be logged."""
        # Every page returns 1 row + a cursor, claiming 100 total.
        page = _make_graphql_response(
            rows=[_make_target_row("ENSG1", "GENE1", 0.9)],
            cursor="always_more",
            count=100,  # 100 total, but max_pages=2 -> 2 fetched
        )
        call_count = [0]

        class _MockResponse:
            def __init__(self, payload: dict) -> None:
                self._payload = payload

            def getcode(self) -> int:
                return 200

            def read(self) -> bytes:
                return json.dumps(self._payload).encode("utf-8")

            def __enter__(self) -> "_MockResponse":
                return self

            def __exit__(self, *args: object) -> None:
                pass

        def mock_urlopen(req, timeout=None, context=None):  # noqa: ANN001
            call_count[0] += 1
            return _MockResponse(page)

        import drugos_graph.opentargets_loader as otl
        monkeypatch.setattr(otl.urllib.request, "urlopen", mock_urlopen)

        with caplog.at_level(logging.WARNING, logger="drugos_graph.opentargets_loader"):
            results = fetch_opentargets_associations(
                "EFO_0000311", max_pages=2, page_size=1,
            )
        # Only 2 rows fetched (max_pages=2).
        assert len(results) == 2
        # Truncation warning must be logged. structlog emits via the
        # standard logging bridge, so we check caplog.text for the
        # event name.
        assert (
            "opentargets_graphql_truncated" in caplog.text
        ), (
            "Truncation WARNING must fire when max_pages is reached before "
            "all rows are fetched. Captured log: " + caplog.text
        )


class TestP2013ErrorHandling:
    """Tests for error handling in ``fetch_opentargets_associations``."""

    def test_empty_disease_id_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="disease_id"):
            fetch_opentargets_associations("")

    def test_disease_not_found_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload = {"data": {"disease": None}}

        class _MockResponse:
            def getcode(self) -> int:
                return 200

            def read(self) -> bytes:
                return json.dumps(payload).encode("utf-8")

            def __enter__(self) -> "_MockResponse":
                return self

            def __exit__(self, *args: object) -> None:
                pass

        def mock_urlopen(req, timeout=None, context=None):  # noqa: ANN001
            return _MockResponse()

        import drugos_graph.opentargets_loader as otl
        monkeypatch.setattr(otl.urllib.request, "urlopen", mock_urlopen)

        with pytest.raises(RuntimeError, match="not found"):
            fetch_opentargets_associations("EFO_INVALID_9999", max_pages=2)

    def test_malformed_response_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload = {"data": None, "errors": [{"message": "GraphQL error"}]}

        class _MockResponse:
            def getcode(self) -> int:
                return 200

            def read(self) -> bytes:
                return json.dumps(payload).encode("utf-8")

            def __enter__(self) -> "_MockResponse":
                return self

            def __exit__(self, *args: object) -> None:
                pass

        def mock_urlopen(req, timeout=None, context=None):  # noqa: ANN001
            return _MockResponse()

        import drugos_graph.opentargets_loader as otl
        monkeypatch.setattr(otl.urllib.request, "urlopen", mock_urlopen)

        with pytest.raises(RuntimeError, match="no data"):
            fetch_opentargets_associations("EFO_0000311", max_pages=2)

    def test_invalid_max_pages_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="max_pages"):
            fetch_opentargets_associations("EFO_0000311", max_pages=0)
        with pytest.raises(ValueError, match="max_pages"):
            fetch_opentargets_associations("EFO_0000311", max_pages=1001)

    def test_invalid_page_size_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="page_size"):
            fetch_opentargets_associations("EFO_0000311", page_size=0)
        with pytest.raises(ValueError, match="page_size"):
            fetch_opentargets_associations("EFO_0000311", page_size=10001)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__, "-v"])
