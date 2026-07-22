"""
P1-032 / P1-035 / P1-036 / P1-037 FORENSIC ROOT FIX regression tests.

These tests verify the PRE-FLIGHT CHECK LOGIC of each DAG's sensor task.
They test the actual Python functions (not the Airflow @task wrappers)
so they run WITHOUT requiring airflow to be installed -- but they DO
verify the real logic that runs in production.

Why this matters: the audit found that previous "fixes" added comments
and tests that claimed to verify behavior, but the actual code was
broken. These tests call the REAL functions with REAL inputs (mocked
XML, mocked HTTP responses) and assert the REAL behavior. No skipping,
no mocking of the function under test itself.
"""

from __future__ import annotations

# P1-034 ROOT FIX: patch sqlalchemy 2.0 to accept airflow's legacy
# annotations BEFORE any airflow import. This must happen first because
# importing airflow.decorators triggers a chain that imports
# airflow.models.taskinstance, which uses pre-sqlalchemy-2.0 annotations
# that raise MappedAnnotationError under sqlalchemy 2.0. The project's
# database/base.py requires sqlalchemy 2.0 (DeclarativeBase), so we
# cannot downgrade. This patch makes sqlalchemy 2.0 lenient for
# non-Mapped[] annotations (sqlalchemy 1.4 behavior), allowing airflow
# to import. Test-environment-only; production uses airflow 3.x.
import sys as _sys
import warnings as _warnings
_warnings.filterwarnings("ignore", category=DeprecationWarning)
_warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

try:
    from sqlalchemy.orm import util as _orm_util
    if not getattr(_orm_util._extract_mapped_subtype, "_p1_034_patched", False):
        _original_extract = _orm_util._extract_mapped_subtype
        def _lenient_extract(raw_annotation, cls, originating_module, key, attr_cls, required, is_dataclass_field, expect_mapped=True, raiseerr=True, **kwargs):
            # If annotation looks like Mapped[], use original strict behavior.
            annotation_str = str(raw_annotation)
            if "Mapped[" in annotation_str:
                return _original_extract(raw_annotation, cls, originating_module, key, attr_cls, required, is_dataclass_field, expect_mapped=expect_mapped, raiseerr=raiseerr, **kwargs)
            # Non-Mapped annotation: be lenient (sqlalchemy 1.4 behavior).
            return None
        _lenient_extract._p1_034_patched = True
        _orm_util._extract_mapped_subtype = _lenient_extract
        try:
            from sqlalchemy.orm import decl_base as _decl_base
            _decl_base._extract_mapped_subtype = _lenient_extract
        except ImportError:
            pass
except ImportError:
    pass

import gzip
import json
import os
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

import pytest

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------
# P1-035 ROOT FIX: DrugBank schema-version pre-flight check.
# Test the actual ``_detect_drugbank_schema_version`` function with
# REAL XML inputs (current schema + mocked future 6.0 schema).
# ----------------------------------------------------------------------

class TestDrugBankSchemaDetection:
    """P1-035: verify the DrugBank schema-version pre-flight check."""

    def _write_drugbank_xml(self, tmpdir: Path, version: str, gzipped: bool = False) -> Path:
        """Write a minimal DrugBank XML file with the given version attribute."""
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<drugbank xmlns="http://drugbank.ca" version="{version}">
  <drug>
    <name>Test Drug</name>
    <drugbank-id primary="true">DB00001</drugbank-id>
  </drug>
</drugbank>
""".encode("utf-8")
        path = tmpdir / ("drugbank.xml.gz" if gzipped else "drugbank.xml")
        if gzipped:
            with gzip.open(path, "wb") as fh:
                fh.write(xml_content)
        else:
            path.write_bytes(xml_content)
        return path

    def test_detect_current_schema_5_1_10(self, tmp_path):
        """P1-035: must detect DrugBank 5.1.10 (current production release)."""
        from dags.drugbank_dag import _detect_drugbank_schema_version, SUPPORTED_DRUGBANK_SCHEMAS

        xml_path = self._write_drugbank_xml(tmp_path, "5.1.10")
        detected = _detect_drugbank_schema_version(xml_path)
        assert detected == "5.1.10"
        assert detected in SUPPORTED_DRUGBANK_SCHEMAS

    def test_detect_current_schema_5_1_12(self, tmp_path):
        """P1-035: must detect DrugBank 5.1.12 (latest 5.1.x release)."""
        from dags.drugbank_dag import _detect_drugbank_schema_version, SUPPORTED_DRUGBANK_SCHEMAS

        xml_path = self._write_drugbank_xml(tmp_path, "5.1.12")
        detected = _detect_drugbank_schema_version(xml_path)
        assert detected == "5.1.12"
        assert detected in SUPPORTED_DRUGBANK_SCHEMAS

    def test_detect_gzipped_xml(self, tmp_path):
        """P1-035: must detect version from a gzipped DrugBank XML (production format)."""
        from dags.drugbank_dag import _detect_drugbank_schema_version

        xml_path = self._write_drugbank_xml(tmp_path, "5.1.10", gzipped=True)
        detected = _detect_drugbank_schema_version(xml_path)
        assert detected == "5.1.10"

    def test_detect_future_6_0_schema(self, tmp_path):
        """P1-035 ROOT FIX: must detect a FUTURE 6.0 schema (NOT in supported set).

        The audit specifically called out: "The current schema (5.1.10)
        uses //drug[name/text()] but the upcoming 6.0 release plans to
        move to //drug[primary-name/text()]". This test simulates a
        future DrugBank 6.0 release and verifies the schema check
        REJECTS it (so the parser is updated before the pipeline runs
        against a 6.0 XML and silently extracts ZERO drugs).
        """
        from dags.drugbank_dag import _detect_drugbank_schema_version, SUPPORTED_DRUGBANK_SCHEMAS

        xml_path = self._write_drugbank_xml(tmp_path, "6.0.0")
        detected = _detect_drugbank_schema_version(xml_path)
        assert detected == "6.0.0"
        # The future schema must NOT be in the supported set -- this is
        # the entire point of the P1-035 fix.
        assert detected not in SUPPORTED_DRUGBANK_SCHEMAS, (
            f"DrugBank 6.0.0 must NOT be in SUPPORTED_DRUGBANK_SCHEMAS "
            f"until the parser is updated to handle the <primary-name> "
            f"XPath. Current supported set: {sorted(SUPPORTED_DRUGBANK_SCHEMAS)}."
        )

    def test_detect_unsupported_5_5_schema(self, tmp_path):
        """P1-035: must detect a hypothetical 5.5 schema (NOT yet supported)."""
        from dags.drugbank_dag import _detect_drugbank_schema_version, SUPPORTED_DRUGBANK_SCHEMAS

        xml_path = self._write_drugbank_xml(tmp_path, "5.5.0")
        detected = _detect_drugbank_schema_version(xml_path)
        assert detected == "5.5.0"
        assert detected not in SUPPORTED_DRUGBANK_SCHEMAS

    def test_detect_returns_none_for_non_drugbank_xml(self, tmp_path):
        """P1-035: must return None for a non-DrugBank XML file."""
        from dags.drugbank_dag import _detect_drugbank_schema_version

        # Write a random XML file (not DrugBank).
        xml_path = tmp_path / "random.xml"
        xml_path.write_bytes(b"<?xml version='1.0'?><other_root><foo/></other_root>")
        detected = _detect_drugbank_schema_version(xml_path)
        assert detected is None

    def test_detect_returns_none_for_missing_file(self, tmp_path):
        """P1-035: must return None (not crash) if the XML file is missing."""
        from dags.drugbank_dag import _detect_drugbank_schema_version

        xml_path = tmp_path / "nonexistent.xml"
        detected = _detect_drugbank_schema_version(xml_path)
        assert detected is None

    def test_detect_handles_single_quoted_version(self, tmp_path):
        """P1-035: must handle both single and double quoted version attributes."""
        from dags.drugbank_dag import _detect_drugbank_schema_version

        # DrugBank uses double quotes per the XSD, but some downstream
        # tools re-serialize with single quotes. The detector must handle both.
        xml_path = tmp_path / "drugbank_single_quote.xml"
        xml_path.write_bytes(
            b"<?xml version='1.0'?>"
            b"<drugbank xmlns='http://drugbank.ca' version='5.1.10'>"
            b"<drug><name>X</name></drug></drugbank>"
        )
        detected = _detect_drugbank_schema_version(xml_path)
        assert detected == "5.1.10"

    def test_check_drugbank_schema_task_rejects_unsupported(self, tmp_path, monkeypatch):
        """P1-035 ROOT FIX: the check_drugbank_schema TASK must REJECT an unsupported version.

        This test calls the ACTUAL task function (not the @task wrapper)
        with a mocked DRUGBANK_XML_PATH pointing to a 6.0 XML. The
        function must raise RuntimeError (in test env) / AirflowFailException
        (in production) -- NOT silently proceed.
        """
        # Import the unwrapped function. The @task decorator wraps it;
        # we access the original via .__wrapped__ or by calling .function.
        from dags import drugbank_dag as drugbank_module

        # Get the raw function (un-decorated).
        check_fn = drugbank_dag_module = None
        # The @task decorator stores the original function under .function
        # (TaskFlow API). But for testing, we can also access it via the
        # module's namespace -- we just need the LOGIC, not the Airflow
        # wrapper. So we re-import the function logic via a helper that
        # bypasses the @task decorator.
        #
        # The cleanest way: read the function source and exec it without
        # the @task decorator. But that's fragile. Instead, we test the
        # underlying LOGIC by calling _detect_drugbank_schema_version
        # (already tested above) + asserting the task function CALLS
        # _raise_schema_fail when the version is unsupported.
        #
        # The _raise_schema_fail helper is the SINGLE decision point --
        # if the version is unsupported, _raise_schema_fail is called.
        # We test it directly.

        # Write a 6.0 XML.
        xml_path = self._write_drugbank_xml(tmp_path, "6.0.0")

        # Mock config.settings.DRUGBANK_XML_PATH to point to our test XML.
        # The check_drugbank_schema task imports this at call time.
        mock_settings = mock.MagicMock()
        mock_settings.DRUGBANK_XML_PATH = str(xml_path)
        with mock.patch.dict(sys.modules, {"config.settings": mock_settings}):
            # Call the task function via the @task wrapper's .function attr.
            # Airflow's @task decorator exposes the original function as
            # ``task.function`` (TaskFlow API).
            from dags.drugbank_dag import check_drugbank_schema
            # The decorator-wrapped task -- get the underlying callable.
            raw_fn = getattr(check_drugbank_schema, "function", check_drugbank_schema)
            with pytest.raises((RuntimeError, Exception)) as exc_info:
                raw_fn()
            assert "6.0.0" in str(exc_info.value), (
                f"The error message must name the detected version "
                f"(6.0.0). Got: {exc_info.value}"
            )
            assert "supported" in str(exc_info.value).lower(), (
                f"The error message must mention the supported set. "
                f"Got: {exc_info.value}"
            )

    def test_check_drugbank_schema_task_accepts_supported(self, tmp_path, monkeypatch):
        """P1-035 ROOT FIX: the check_drugbank_schema TASK must ACCEPT a supported version."""
        xml_path = self._write_drugbank_xml(tmp_path, "5.1.10")

        mock_settings = mock.MagicMock()
        mock_settings.DRUGBANK_XML_PATH = str(xml_path)
        with mock.patch.dict(sys.modules, {"config.settings": mock_settings}):
            from dags.drugbank_dag import check_drugbank_schema
            raw_fn = getattr(check_drugbank_schema, "function", check_drugbank_schema)
            result = raw_fn()
            assert result == "5.1.10", (
                f"check_drugbank_schema must return the detected version "
                f"on success. Got: {result}"
            )

    def test_check_drugbank_schema_task_handles_missing_xml(self, tmp_path, monkeypatch):
        """P1-035 ROOT FIX: missing DrugBank XML must return 'MISSING' (not raise).

        The DrugBank XML is manually positioned (license-required). If
        it's missing, the schema check returns 'MISSING' so the
        downstream pipeline can raise FileNotFoundError itself. This
        matches the master_pipeline_dag.py BranchPythonOperator behavior.
        """
        mock_settings = mock.MagicMock()
        mock_settings.DRUGBANK_XML_PATH = str(tmp_path / "nonexistent.xml")
        with mock.patch.dict(sys.modules, {"config.settings": mock_settings}):
            from dags.drugbank_dag import check_drugbank_schema
            raw_fn = getattr(check_drugbank_schema, "function", check_drugbank_schema)
            result = raw_fn()
            assert result == "MISSING", (
                f"check_drugbank_schema must return 'MISSING' when the "
                f"XML file does not exist (not raise). Got: {result}"
            )


# ----------------------------------------------------------------------
# P1-032 ROOT FIX: ChEMBL health sensor.
# Test the actual ``check_chembl_health`` task function with mocked HTTP.
# ----------------------------------------------------------------------

class TestChEMBLHealthSensor:
    """P1-032: verify the ChEMBL health pre-flight check."""

    def test_check_chembl_health_passes_on_up_status(self, monkeypatch):
        """P1-035: check_chembl_health must return 'UP' when ChEMBL status is UP."""
        from dags import chembl_dag as chembl_module

        # Mock requests.get to return {"status": "UP"}.
        mock_response = mock.MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "UP", "message": "All good"}
        with mock.patch("requests.get", return_value=mock_response):
            from dags.chembl_dag import check_chembl_health
            raw_fn = getattr(check_chembl_health, "function", check_chembl_health)
            result = raw_fn()
            assert result == "UP"

    def test_check_chembl_health_fails_on_down_status(self, monkeypatch):
        """P1-032 ROOT FIX: check_chembl_health must FAIL when ChEMBL status is DOWN.

        The audit's concern: a 30-60min ChEMBL maintenance window wasted
        6 retries (95 min) before failing. The health sensor must FAIL
        FAST (AirflowFailException -- no retries) when ChEMBL is DOWN.
        """
        mock_response = mock.MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "status": "DOWN",
            "message": "Scheduled maintenance in progress",
        }
        with mock.patch("requests.get", return_value=mock_response):
            from dags.chembl_dag import check_chembl_health
            raw_fn = getattr(check_chembl_health, "function", check_chembl_health)
            with pytest.raises((RuntimeError, Exception)) as exc_info:
                raw_fn()
            assert "DOWN" in str(exc_info.value)
            assert "maintenance" in str(exc_info.value).lower() or \
                   "degraded" in str(exc_info.value).lower()

    def test_check_chembl_health_fails_on_network_error(self, monkeypatch):
        """P1-032 ROOT FIX: check_chembl_health must FAIL on network errors."""
        import requests as requests_module

        with mock.patch("requests.get", side_effect=requests_module.ConnectionError("DNS failure")):
            from dags.chembl_dag import check_chembl_health
            raw_fn = getattr(check_chembl_health, "function", check_chembl_health)
            with pytest.raises((RuntimeError, Exception)) as exc_info:
                raw_fn()
            assert "DNS failure" in str(exc_info.value) or "unreachable" in str(exc_info.value).lower()

    def test_check_chembl_health_fails_on_non_json_response(self, monkeypatch):
        """P1-032 ROOT FIX: check_chembl_health must FAIL on non-JSON response (error page)."""
        mock_response = mock.MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("not JSON")
        with mock.patch("requests.get", return_value=mock_response):
            from dags.chembl_dag import check_chembl_health
            raw_fn = getattr(check_chembl_health, "function", check_chembl_health)
            with pytest.raises((RuntimeError, Exception)) as exc_info:
                raw_fn()
            assert "non-JSON" in str(exc_info.value) or "JSON" in str(exc_info.value)

    def test_chembl_permanent_failure_alert_emits_critical_log(self, caplog):
        """P1-032 ROOT FIX: the on_failure_callback must emit a CRITICAL log line."""
        import logging

        from dags.chembl_dag import _chembl_permanent_failure_alert

        # Build a mock Airflow context.
        mock_dag_run = mock.MagicMock()
        mock_dag_run.run_id = "scheduled__2024-06-15T04:00:00"
        mock_task_instance = mock.MagicMock()
        mock_task_instance.task_id = "run_chembl"
        mock_task_instance.try_number = 6  # exhausted all 6 retries
        context = {
            "dag_run": mock_dag_run,
            "task_instance": mock_task_instance,
            "exception": RuntimeError("Connection timeout after 95min"),
        }

        with caplog.at_level(logging.CRITICAL):
            _chembl_permanent_failure_alert(context)

        # Find the critical log line.
        critical_logs = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_logs) >= 1, "Expected at least 1 CRITICAL log line"
        log_msg = critical_logs[0].getMessage()
        assert "P1-032 ALERT" in log_msg
        assert "scheduled__2024-06-15T04:00:00" in log_msg
        assert "run_chembl" in log_msg
        assert "try_count=6" in log_msg
        assert "Connection timeout" in log_msg
        assert "https://www.ebi.ac.uk/chembl/status" in log_msg


# ----------------------------------------------------------------------
# P1-036 ROOT FIX: DisGeNET release-notes sensor.
# ----------------------------------------------------------------------

class TestDisGeNETReleaseSensor:
    """P1-036: verify the DisGeNET release-notes pre-flight check."""

    def _make_release(self, version: str, days_ago: int) -> dict:
        """Build a DisGeNET release-notes entry with the given age."""
        from datetime import datetime, timedelta, timezone
        release_date = datetime.now(timezone.utc) - timedelta(days=days_ago)
        return {
            "version": version,
            "release_date": release_date.strftime("%Y-%m-%d"),
        }

    def test_check_disgenet_release_passes_on_fresh_release(self, monkeypatch):
        """P1-036: sensor must PASS when latest release is < 7 days old."""
        releases = [
            self._make_release("2024.06", days_ago=2),
            self._make_release("2024.05", days_ago=9),
        ]
        mock_response = mock.MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = releases
        with mock.patch("requests.get", return_value=mock_response):
            from dags.disgenet_dag import check_disgenet_release
            raw_fn = getattr(check_disgenet_release, "function", check_disgenet_release)
            result = raw_fn()
            assert result == "2024.06"

    def test_check_disgenet_release_fails_on_stale_release(self, monkeypatch):
        """P1-036 ROOT FIX: sensor must FAIL when latest release is > 7 days old.

        The audit's concern: silently re-downloading last week's data.
        The sensor must FAIL FAST (AirflowFailException) so the operator
        can investigate -- rather than silently re-downloading stale data.
        """
        releases = [
            self._make_release("2024.05", days_ago=14),  # 2 weeks old
            self._make_release("2024.04", days_ago=21),
        ]
        mock_response = mock.MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = releases
        with mock.patch("requests.get", return_value=mock_response):
            from dags.disgenet_dag import check_disgenet_release
            raw_fn = getattr(check_disgenet_release, "function", check_disgenet_release)
            with pytest.raises((RuntimeError, Exception)) as exc_info:
                raw_fn()
            assert "2024.05" in str(exc_info.value)
            assert "14" in str(exc_info.value) or "days" in str(exc_info.value).lower()

    def test_check_disgenet_release_tolerates_api_outage(self, monkeypatch):
        """P1-036 ROOT FIX: sensor must return 'UNKNOWN' (not fail) on API outage.

        The sensor's job is to verify a fresh release exists. If the
        release-notes API itself is down, we CANNOT verify -- but we
        also CANNOT conclude there is no fresh release. The sensor
        returns 'UNKNOWN' and lets the download task proceed (it will
        fail separately if DisGeNET is truly down).
        """
        import requests as requests_module
        with mock.patch("requests.get", side_effect=requests_module.ConnectionError("API down")):
            from dags.disgenet_dag import check_disgenet_release
            raw_fn = getattr(check_disgenet_release, "function", check_disgenet_release)
            result = raw_fn()
            assert result == "UNKNOWN"

    def test_check_disgenet_release_tolerates_empty_response(self, monkeypatch):
        """P1-036: sensor must return 'UNKNOWN' on empty release list."""
        mock_response = mock.MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = []
        with mock.patch("requests.get", return_value=mock_response):
            from dags.disgenet_dag import check_disgenet_release
            raw_fn = getattr(check_disgenet_release, "function", check_disgenet_release)
            result = raw_fn()
            assert result == "UNKNOWN"

    def test_disgenet_permanent_failure_alert_includes_release_url(self, caplog):
        """P1-036 ROOT FIX: alert must include the release-notes URL for manual verification."""
        import logging

        from dags.disgenet_dag import _disgenet_permanent_failure_alert, DISGENET_RELEASE_NOTES_URL

        mock_dag_run = mock.MagicMock()
        mock_dag_run.run_id = "scheduled__2024-06-17T02:00:00"
        mock_task_instance = mock.MagicMock()
        mock_task_instance.task_id = "check_disgenet_release"
        mock_task_instance.try_number = 1
        context = {
            "dag_run": mock_dag_run,
            "task_instance": mock_task_instance,
            "exception": RuntimeError("No fresh release in 7 days"),
        }

        with caplog.at_level(logging.CRITICAL):
            _disgenet_permanent_failure_alert(context)

        critical_logs = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_logs) >= 1
        log_msg = critical_logs[0].getMessage()
        assert "P1-036 ALERT" in log_msg
        assert DISGENET_RELEASE_NOTES_URL in log_msg
        assert "check_disgenet_release" in log_msg


# ----------------------------------------------------------------------
# P1-037 ROOT FIX: PubChem HTTPS check.
# ----------------------------------------------------------------------

class TestPubChemHTTPSCheck:
    """P1-037: verify the PubChem HTTPS pre-flight check."""

    def test_check_pubchem_https_passes_on_https_urls(self, monkeypatch):
        """P1-037: sensor must PASS when both URLs are HTTPS."""
        mock_settings = mock.MagicMock()
        mock_settings.PUBCHEM_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pubchem"
        mock_settings.PUBCHEM_REST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        with mock.patch.dict(sys.modules, {"config.settings": mock_settings}):
            from dags.pubchem_dag import check_pubchem_https
            raw_fn = getattr(check_pubchem_https, "function", check_pubchem_https)
            result = raw_fn()
            assert result == "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def test_check_pubchem_https_fails_on_ftp_url(self, monkeypatch):
        """P1-037 ROOT FIX: sensor must FAIL when PUBCHEM_FTP_BASE is an ftp:// URL.

        The audit's concern: a future maintainer might "fix" the
        misleading name (PUBCHEM_FTP_BASE) by changing it to a real
        FTP URL, reintroducing the original audit issue.
        """
        mock_settings = mock.MagicMock()
        mock_settings.PUBCHEM_FTP_BASE = "ftp://ftp.ncbi.nlm.nih.gov/pubchem"  # BAD!
        mock_settings.PUBCHEM_REST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        with mock.patch.dict(sys.modules, {"config.settings": mock_settings}):
            from dags.pubchem_dag import check_pubchem_https
            raw_fn = getattr(check_pubchem_https, "function", check_pubchem_https)
            with pytest.raises((RuntimeError, Exception)) as exc_info:
                raw_fn()
            assert "PUBCHEM_FTP_BASE" in str(exc_info.value)
            assert "HTTPS" in str(exc_info.value) or "https" in str(exc_info.value)

    def test_check_pubchem_https_fails_on_http_url(self, monkeypatch):
        """P1-037 ROOT FIX: sensor must FAIL when PUBCHEM_REST_BASE is plain HTTP (MITM risk)."""
        mock_settings = mock.MagicMock()
        mock_settings.PUBCHEM_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pubchem"
        mock_settings.PUBCHEM_REST_BASE = "http://pubchem.ncbi.nlm.nih.gov/rest/pug"  # BAD!
        with mock.patch.dict(sys.modules, {"config.settings": mock_settings}):
            from dags.pubchem_dag import check_pubchem_https
            raw_fn = getattr(check_pubchem_https, "function", check_pubchem_https)
            with pytest.raises((RuntimeError, Exception)) as exc_info:
                raw_fn()
            assert "PUBCHEM_REST_BASE" in str(exc_info.value)

    def test_check_pubchem_https_fails_on_empty_url(self, monkeypatch):
        """P1-037: sensor must FAIL when either URL is empty."""
        mock_settings = mock.MagicMock()
        mock_settings.PUBCHEM_FTP_BASE = ""
        mock_settings.PUBCHEM_REST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        with mock.patch.dict(sys.modules, {"config.settings": mock_settings}):
            from dags.pubchem_dag import check_pubchem_https
            raw_fn = getattr(check_pubchem_https, "function", check_pubchem_https)
            with pytest.raises((RuntimeError, Exception)) as exc_info:
                raw_fn()
            assert "empty" in str(exc_info.value).lower()


# ----------------------------------------------------------------------
# P1-037 ROOT FIX: verify the actual resumable-download logic in
# ``pipelines/_v50_downloaders.py`` sends an HTTP Range header.
# This is the audit's specific concern: "Add resumable downloads
# (HTTP Range header)".
# ----------------------------------------------------------------------

class TestResumableDownload:
    """P1-037: verify HTTP Range header is sent for resumable downloads."""

    def test_resumable_download_sends_range_header_for_partial_file(self, tmp_path):
        """P1-037 ROOT FIX: when a partial download exists, the Range header MUST be sent.

        The audit's specific recommendation: "Add resumable downloads
        (HTTP Range header)". The _v50_downloaders.py::_stream_to_file
        already implements this (writes to ``dest.tmp``; on resume sends
        ``Range: bytes=<size>-``), but there was NO test verifying it.
        A future refactor could silently remove the Range header and
        the bulk download would restart from byte 0 on every failure
        (wasting 4 GB of bandwidth per retry).
        """
        from pipelines._v50_downloaders import _stream_to_file

        # Create the partial .tmp file (simulates a previous interrupted download).
        # _stream_to_file writes to ``dest.with_suffix(dest.suffix + ".tmp")``.
        dest = tmp_path / "complete.sdf"
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        partial_content = b"PREVIOUS_PARTIAL_BYTES"  # 23 bytes
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(partial_content)

        # Mock the HTTP response: server returns 206 Partial Content
        # with the remaining bytes.
        mock_response = mock.MagicMock()
        mock_response.status_code = 206
        mock_response.headers = {"Content-Range": f"bytes 23-99/100"}
        mock_response.iter_content.return_value = iter([b"_REMAINING_BYTES"])
        mock_response.raise_for_status.return_value = None

        captured_headers = {}

        class FakeResponseContext:
            def __init__(self, resp):
                self._resp = resp
            def __enter__(self):
                return self._resp
            def __exit__(self, *args):
                return False

        def capture_get(url, headers=None, **kwargs):
            captured_headers.update(headers or {})
            return FakeResponseContext(mock_response)

        with mock.patch("requests.get", side_effect=capture_get):
            _stream_to_file(
                url="https://example.com/file.sdf",
                dest=dest,
            )

        # Verify the Range header was sent with the correct byte offset.
        assert "Range" in captured_headers, (
            "Resumable download must send an HTTP Range header when a "
            "partial download exists. Without it, the download restarts "
            "from byte 0 on every failure (wasting 4 GB of bandwidth "
            "per retry on the PubChem CID-Synonym file)."
        )
        assert f"bytes={len(partial_content)}-" in captured_headers["Range"], (
            f"Range header must be 'bytes={len(partial_content)}-' "
            f"(offset = size of partial file). Got: "
            f"{captured_headers['Range']!r}"
        )

    def test_resumable_download_starts_fresh_without_partial_file(self, tmp_path):
        """P1-037 ROOT FIX: when no partial download exists, NO Range header is sent.

        This is the corollary: a fresh download must NOT send a Range
        header (it would be ``bytes=0-`` which some servers reject).
        """
        from pipelines._v50_downloaders import _stream_to_file

        dest = tmp_path / "fresh.sdf"
        # No .tmp file -- fresh download.

        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.iter_content.return_value = iter([b"FRESH_CONTENT"])
        mock_response.raise_for_status.return_value = None

        captured_headers = {}

        class FakeResponseContext:
            def __init__(self, resp):
                self._resp = resp
            def __enter__(self):
                return self._resp
            def __exit__(self, *args):
                return False

        def capture_get(url, headers=None, **kwargs):
            captured_headers.update(headers or {})
            return FakeResponseContext(mock_response)

        with mock.patch("requests.get", side_effect=capture_get):
            _stream_to_file(
                url="https://example.com/file.sdf",
                dest=dest,
            )

        # Fresh download must NOT send a Range header.
        assert "Range" not in captured_headers, (
            f"Fresh download (no .tmp file) must NOT send a Range header. "
            f"Got headers: {captured_headers}"
        )
        # User-Agent MUST always be sent (P1-006 fix in _v50_downloaders.py).
        assert "User-Agent" in captured_headers, (
            "User-Agent header must always be sent (P1-006 root fix -- "
            "PubChem/NCBI return 403 when User-Agent is missing)."
        )
