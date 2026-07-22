"""Regression tests for P2-011: clinicaltrials_loader.py schema v2 support.

P2-011 ROOT FIX: ClinicalTrials.gov migrated from v1 to v2 schema in
2024. A loader that parses only v1 crashes with ``KeyError`` on the
first v2 trial. The fix adds:

  * ``_detect_ctgov_schema_version`` -- inspects the response shape.
  * ``_parse_ctgov_v1_study`` -- parses legacy CamelCase schema.
  * ``_parse_ctgov_v2_study`` -- parses current camelCase schema.
  * ``parse_ctgov_study`` -- auto-detects and dispatches.
  * ``fetch_ctgov_studies`` -- paginated GraphQL-style client.

The tests use both v1 and v2 fixture responses (no network calls).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "phase2"))

from drugos_graph.clinicaltrials_loader import (  # noqa: E402
    _detect_ctgov_schema_version,
    _parse_ctgov_v1_study,
    _parse_ctgov_v2_study,
    parse_ctgov_study,
    fetch_ctgov_studies,
)


# =============================================================================
# Fixture: v1 schema response (legacy, CamelCase).
# =============================================================================
V1_RESPONSE = {
    "studies": [
        {
            "StudyFieldsSection": {
                "StudyFields": {
                    "NCTId": ["NCT00000001"],
                    "BriefTitle": ["Test Trial v1"],
                    "OverallStatus": ["RECRUITING"],
                    "Phase": ["Phase 3"],
                    "EnrollmentCount": ["500"],
                    "Condition": ["Breast Cancer"],
                    "InterventionName": ["Drug A"],
                },
            },
        },
        {
            "StudyFieldsSection": {
                "StudyFields": {
                    "NCTId": ["NCT00000002"],
                    "BriefTitle": ["Test Trial v1 #2"],
                    "OverallStatus": ["COMPLETED"],
                    "Phase": ["Phase 2"],
                    "EnrollmentCount": ["200"],
                    "Condition": ["Hypertension", "Diabetes"],
                    "InterventionName": ["Drug B", "Placebo"],
                },
            },
        },
    ],
}


# =============================================================================
# Fixture: v2 schema response (current, camelCase).
# =============================================================================
V2_RESPONSE = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000003",
                    "briefTitle": "Test Trial v2",
                },
                "statusModule": {
                    "overallStatus": "RECRUITING",
                    "startDateStruct": {"date": "2024-01-15"},
                    "completionDateStruct": {"date": "2026-01-15"},
                },
                "designModule": {
                    "phases": ["PHASE3"],
                    "enrollmentInfo": {"count": 750, "type": "ACTUAL"},
                },
                "conditionsModule": {
                    "conditions": ["Breast Cancer", "Lung Cancer"],
                },
                "armsInterventionsModule": {
                    "interventions": [
                        {"name": "Drug A", "type": "DRUG"},
                        {"name": "Drug B", "type": "DRUG"},
                    ],
                },
            },
        },
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000004",
                    "briefTitle": "Test Trial v2 #2",
                },
                "statusModule": {
                    "overallStatus": "COMPLETED",
                },
                "designModule": {
                    "phases": ["PHASE2"],
                    "enrollmentInfo": {"count": 100},
                },
                "conditionsModule": {
                    "conditions": ["Diabetes"],
                },
                "armsInterventionsModule": {
                    "interventions": [],
                },
            },
        },
    ],
    "nextPageToken": None,
}


class TestP2011SchemaVersionDetection:
    """Tests for ``_detect_ctgov_schema_version``."""

    def test_detects_v2_response(self) -> None:
        assert _detect_ctgov_schema_version(V2_RESPONSE) == "v2"

    def test_detects_v1_response(self) -> None:
        assert _detect_ctgov_schema_version(V1_RESPONSE) == "v1"

    def test_returns_unknown_for_empty_response(self) -> None:
        assert _detect_ctgov_schema_version({"studies": []}) == "unknown"

    def test_returns_unknown_for_non_dict(self) -> None:
        assert _detect_ctgov_schema_version(None) == "unknown"  # type: ignore[arg-type]
        assert _detect_ctgov_schema_version("not a dict") == "unknown"  # type: ignore[arg-type]

    def test_returns_unknown_for_missing_studies_key(self) -> None:
        assert _detect_ctgov_schema_version({}) == "unknown"


class TestP2011V1StudyParser:
    """Tests for ``_parse_ctgov_v1_study``."""

    def test_parses_v1_study_fields(self) -> None:
        study = V1_RESPONSE["studies"][0]
        result = _parse_ctgov_v1_study(study)
        assert result["schema_version"] == "v1"
        assert result["nct_id"] == "NCT00000001"
        assert result["brief_title"] == "Test Trial v1"
        assert result["overall_status"] == "RECRUITING"
        assert result["phase"] == "Phase 3"
        assert result["enrollment"] == "500"
        assert "Breast Cancer" in result["conditions"]
        assert "Drug A" in result["interventions"]

    def test_parses_v1_study_with_multiple_conditions(self) -> None:
        study = V1_RESPONSE["studies"][1]
        result = _parse_ctgov_v1_study(study)
        assert result["nct_id"] == "NCT00000002"
        assert "Hypertension" in result["conditions"]
        assert "Diabetes" in result["conditions"]
        assert "Drug B" in result["interventions"]
        assert "Placebo" in result["interventions"]


class TestP2011V2StudyParser:
    """Tests for ``_parse_ctgov_v2_study``."""

    def test_parses_v2_study_modules(self) -> None:
        study = V2_RESPONSE["studies"][0]
        result = _parse_ctgov_v2_study(study)
        assert result["schema_version"] == "v2"
        assert result["nct_id"] == "NCT00000003"
        assert result["brief_title"] == "Test Trial v2"
        assert result["overall_status"] == "RECRUITING"
        assert "PHASE3" in str(result["phase"])
        assert result["enrollment"] == 750
        assert result["start_date"] == "2024-01-15"
        assert result["completion_date"] == "2026-01-15"
        assert "Breast Cancer" in result["conditions"]
        assert "Lung Cancer" in result["conditions"]
        assert "Drug A" in result["interventions"]
        assert "Drug B" in result["interventions"]

    def test_parses_v2_study_with_empty_interventions(self) -> None:
        study = V2_RESPONSE["studies"][1]
        result = _parse_ctgov_v2_study(study)
        assert result["nct_id"] == "NCT00000004"
        assert result["interventions"] == []


class TestP2011AutoDispatchParser:
    """Tests for ``parse_ctgov_study`` (auto-detects schema)."""

    def test_auto_detects_v1_study(self) -> None:
        study = V1_RESPONSE["studies"][0]
        result = parse_ctgov_study(study)
        assert result["schema_version"] == "v1"
        assert result["nct_id"] == "NCT00000001"

    def test_auto_detects_v2_study(self) -> None:
        study = V2_RESPONSE["studies"][0]
        result = parse_ctgov_study(study)
        assert result["schema_version"] == "v2"
        assert result["nct_id"] == "NCT00000003"

    def test_returns_unknown_for_unrecognized_shape(self) -> None:
        result = parse_ctgov_study({"weird": "shape"})
        assert result["schema_version"] == "unknown"
        assert result["nct_id"] == ""

    def test_explicit_schema_override(self) -> None:
        # When schema_version is explicitly provided, skip auto-detect.
        result = parse_ctgov_study({}, schema_version="v2")
        assert result["schema_version"] == "v2"
        # All fields empty because the input has no v2 modules.
        assert result["nct_id"] == ""


class TestP2011FetchWithMockedPagination:
    """Tests for ``fetch_ctgov_studies`` with mocked urlopen (no network)."""

    def test_3_page_pagination_test(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P2-011 spec: CI test with a 3-page response."""
        # Build 3 mock responses where each has a nextPageToken pointing
        # to the next, and the last has no nextPageToken.
        page_responses = [
            {
                "studies": [
                    {"protocolSection": {"identificationModule": {"nctId": f"NCT00{i}0001"}}},
                    {"protocolSection": {"identificationModule": {"nctId": f"NCT00{i}0002"}}},
                ],
                "nextPageToken": f"page{i+1}_cursor",
            }
            for i in range(2)  # pages 1 and 2 have cursors
        ] + [
            # Page 3 has NO cursor -- end of pagination.
            {
                "studies": [
                    {"protocolSection": {"identificationModule": {"nctId": "NCT0020001"}}},
                    {"protocolSection": {"identificationModule": {"nctId": "NCT0020002"}}},
                ],
                "nextPageToken": None,
            },
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

        # Patch urlopen in the clinicaltrials_loader namespace.
        import drugos_graph.clinicaltrials_loader as ctl
        monkeypatch.setattr(ctl.urllib.request, "urlopen", mock_urlopen)

        results = fetch_ctgov_studies("breast cancer", max_pages=10)
        # 3 pages * 2 studies per page = 6 studies.
        assert len(results) == 6, (
            f"Expected 6 studies across 3 pages, got {len(results)}"
        )
        # All studies must have schema_version="v2" (our mock uses v2 shape).
        assert all(r["schema_version"] == "v2" for r in results)
        # The mock must have been called 3 times (once per page).
        assert call_count[0] == 3, (
            f"Expected 3 paginated requests, got {call_count[0]}"
        )

    def test_stops_when_no_next_page_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pagination must stop when nextPageToken is null/missing."""
        single_page = {
            "studies": [
                {"protocolSection": {"identificationModule": {"nctId": "NCT0001"}}},
            ],
            "nextPageToken": None,
        }
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

        import drugos_graph.clinicaltrials_loader as ctl
        monkeypatch.setattr(ctl.urllib.request, "urlopen", mock_urlopen)

        results = fetch_ctgov_studies("test", max_pages=10)
        assert len(results) == 1
        # Even though max_pages=10, pagination must stop after 1 page.
        assert call_count[0] == 1


class TestP2011MixedSchemaResponse:
    """A response with mixed v1 + v2 studies must not crash."""

    def test_v1_and_v2_studies_both_parse(self) -> None:
        v1_study = V1_RESPONSE["studies"][0]
        v2_study = V2_RESPONSE["studies"][0]
        v1_parsed = parse_ctgov_study(v1_study)
        v2_parsed = parse_ctgov_study(v2_study)
        # Both must parse without KeyError.
        assert v1_parsed["schema_version"] == "v1"
        assert v2_parsed["schema_version"] == "v2"
        assert v1_parsed["nct_id"] == "NCT00000001"
        assert v2_parsed["nct_id"] == "NCT00000003"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
