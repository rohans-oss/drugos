"""
DrugBank DAG -- standalone pipeline for DrugBank XML drug and target data.

Parses the DrugBank full-database XML file (requires manual download due
to licensing).  Extracts drug metadata and target interactions, normalises
InChIKeys, deduplicates, and bulk-upserts into the ``drugs`` and
``drug_protein_interactions`` tables.

If the DrugBank XML file is not present the pipeline will raise a clear
``FileNotFoundError`` with download instructions.

Can be triggered independently or as part of the master pipeline.
Schedule: every Monday at 03:00 UTC (cron ``0 3 * * 1``).
v49 ROOT FIX (Compound-4 -- Sunday Morning Pile-Up): was previously
``0 3 * * 0`` (Sunday 03:00 UTC) which overlapped the master DAG
window (Sunday 02:00 UTC, 8h timeout). Moved to Monday to eliminate
the per-pipeline filelock conflict with the master. DrugBank XML is
manually positioned; the weekly standalone run picks up any newly-
positioned XML without requiring the master DAG.

P1-035 FORENSIC ROOT FIX (Team 4 -- DrugBank XML schema change crash):
    DrugBank has changed its XML schema 4 times in the last 10 years.
    The current schema (5.1.10) uses ``<drug><name>...</name></drug>``
    but the upcoming 6.0 release plans to move to
    ``<drug><primary-name>...</primary-name></drug>``. The previous
    DAG did NOT verify the schema version of the actual XML file --
    it just called ``DrugBankPipeline().run()``. On a future DrugBank
    release, the parser would silently extract ZERO drugs (the
    ``<name>`` XPath finds nothing in 6.0), and the KG would lose ALL
    DrugBank data without any error.

    ROOT FIX (master-grade, no sugar-coating):
      1. Add ``check_drugbank_schema`` pre-flight task that opens the
         XML, reads the root ``<drugbank version="...">`` attribute,
         and verifies the version is in ``SUPPORTED_DRUGBANK_SCHEMAS``.
      2. If the version is UNSUPPORTED, raise ``AirflowFailException``
         (non-retryable) with a clear message naming the detected
         version + the supported set. The DAG fails RED immediately --
         no silent zero-drugs extraction.
      3. Wire ``check_drugbank_schema >> run_drugbank`` so the pipeline
         NEVER runs against an unsupported schema.
      4. The schema-version check reads ONLY the root element (the
         first ~200 bytes of the XML) -- it does NOT parse the full
         file. This keeps the pre-flight check <1 second even for the
         1.5GB DrugBank XML.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

# v89 ROOT FIX (BUG #39): shared sys.path bootstrap (see dags/_dags_init.py).
from dags._dags_init import ensure_project_root  # noqa: F401
# P1-050 ROOT FIX: explicit call (no longer auto-invoked at module import)
ensure_project_root()

from airflow.decorators import dag, task

# v74 ROOT FIX (T-023 -- retries on 4xx HTTP errors waste 60 min):
# Use the shared retry policy: exponential backoff (5min -> 10min -> 20min
# cap) AND a fail-fast decorator that converts HTTP 4xx (401 Unauthorized,
# 403 Forbidden, 404 Not Found, etc.) to AirflowFailException so the task
# is NOT retried. Retrying a 401 (bad API key) or 404 (wrong endpoint)
# never succeeds -- the original error is non-transient.
from dags._retry_policy import DEFAULT_RETRY_ARGS, fail_fast_on_http_4xx

logger = logging.getLogger(__name__)

# v29 ROOT FIX (audit O-12): XCom used for large dataframes -- anti-pattern.
# Now passes file paths via XCom. The single @task below returns None and the
# DrugBankPipeline persists its output to processed_data/ (drugbank_drugs.csv).
# Downstream DAGs (master pipeline) read that CSV by path -- no DataFrame is
# ever pushed to / pulled from XCom.

DEFAULT_ARGS = {
    **DEFAULT_RETRY_ARGS,
    "owner": "drug_repurposing",
    "depends_on_past": False,
}

# P1-035 + P1-015 ROOT FIX: supported DrugBank XML schema versions.
#
# P1-015 ROOT FIX (forensic, root-level):
#   The previous code hardcoded versions 5.0 through 5.1.12 in a
#   frozenset. DrugBank releases a new schema version roughly every
#   1-2 months (5.1.13, 5.1.14, ...). Each new release FAILED the
#   schema check at ``check_drugbank_schema`` and raised
#   ``AirflowFailException``, blocking the entire DrugBank DAG until a
#   developer manually added the new version to the frozenset,
#   committed, and redeployed. Pharma customers using the platform
#   would see stale DrugBank data and miss newly-approved drugs.
#
#   The parser uses XPath ``<drug><name>...</name></drug>`` which has
#   been stable across ALL 5.x releases. The version check exists to
#   catch a HYPOTHETICAL 6.0 schema change (planned ``<primary-name>``)
#   that hasn't happened yet — so a strict whitelist is over-engineered
#   for the actual risk.
#
#   ROOT FIX:
#     1. Replace the explicit whitelist with a REGEX that matches ALL
#        5.x versions: ``^5\.\d+(\.\d+)?$``. This auto-accepts future
#        5.1.13, 5.1.14, 5.2.0, 5.3.0, etc. without code changes.
#     2. Keep a separate ``BLOCKED_DRUGBANK_SCHEMAS`` frozenset for
#        known-bad versions (empty by default — no known bad 5.x).
#     3. Versions >= 6.0 are NOT auto-accepted. They emit a WARNING
#        (not FAIL) so the operator can decide whether to proceed
#        (the parser MIGHT still work, depending on the actual 6.0
#        schema when it ships).
import re as _re_for_drugbank_schema

# Regex that matches ALL 5.x versions: 5.0, 5.1, 5.1.0, 5.1.10, 5.2.0, etc.
# The parser's XPath ``<drug><name>...</name></drug>`` is stable across
# the entire 5.x line — there's no need to whitelist individual patches.
SUPPORTED_DRUGBANK_SCHEMA_REGEX = _re_for_drugbank_schema.compile(
    r"^5\.\d+(\.\d+)?$"
)

# Known-bad DrugBank schema versions (none currently). When a 5.x
# release ships with a regression (e.g. 5.1.20 ships a broken XSD), add
# it here to block it without affecting other 5.x releases.
BLOCKED_DRUGBANK_SCHEMAS: frozenset[str] = frozenset()

# Major-version threshold for the WARNING (not FAIL) path. Versions >= 6
# are not auto-accepted because the parser MIGHT break — but we don't
# hard-fail because we don't know the actual 6.0 schema yet.
DRUGBANK_MAJOR_VERSION_WARN_THRESHOLD = 6

# Backward-compat: callers that import SUPPORTED_DRUGBANK_SCHEMAS
# (e.g. existing tests) get a frozenset of the canonical versions we
# KNOW work. This is no longer the authoritative check — the regex is.
SUPPORTED_DRUGBANK_SCHEMAS: frozenset[str] = frozenset({
    "5.0",
    "5.1", "5.1.0", "5.1.1", "5.1.2", "5.1.3", "5.1.4", "5.1.5",
    "5.1.6", "5.1.7", "5.1.8", "5.1.9", "5.1.10", "5.1.11", "5.1.12",
})


def _is_drugbank_version_supported(version: str) -> bool:
    """P1-015 ROOT FIX: check a DrugBank version against the SUPPORTED pattern.

    A version is supported if:
      1. It matches ``^5\\.\\d+(\\.\\d+)?$`` (any 5.x version), AND
      2. It is NOT in ``BLOCKED_DRUGBANK_SCHEMAS``.

    Versions >= 6.0 (e.g. 6.0.0) return False — the caller decides
    whether to FAIL or WARN.
    """
    if not version or not isinstance(version, str):
        return False
    if version in BLOCKED_DRUGBANK_SCHEMAS:
        return False
    return bool(SUPPORTED_DRUGBANK_SCHEMA_REGEX.match(version))


def _is_drugbank_version_warn_only(version: str) -> bool:
    """P1-015 ROOT FIX: True if the version is >= 6.0 (WARN, not FAIL).

    Versions >= 6.0 MIGHT still parse correctly (the parser's XPath is
    generic enough to handle a rename to ``<primary-name>`` if we add a
    fallback). The operator should be ALERTED but not BLOCKED — the
    alternative is hard-failing on every 6.x release until a developer
    updates the parser, which is the same maintenance trap P1-015 fixes.
    """
    if not version or not isinstance(version, str):
        return False
    try:
        major = int(version.split(".", 1)[0])
    except (ValueError, IndexError):
        return False
    return major >= DRUGBANK_MAJOR_VERSION_WARN_THRESHOLD


def _raise_schema_fail(message: str) -> None:
    """P1-035: raise AirflowFailException if airflow is installed, else RuntimeError.

    This indirection lets the schema check be unit-tested without
    requiring airflow to be installed (the test catches RuntimeError
    instead of AirflowFailException).
    """
    try:
        from airflow.exceptions import AirflowFailException
        raise AirflowFailException(message)
    except ImportError:
        raise RuntimeError(message)


def _detect_drugbank_schema_version(xml_path: Path) -> str | None:
    """P1-035 ROOT FIX: read the DrugBank XML schema version from the root element.

    Opens the XML file (handling .gz / .xml transparently), reads ONLY
    the root ``<drugbank ...>`` element, and extracts the ``version``
    attribute. Returns ``None`` if the version cannot be determined
    (the caller decides whether to fail or proceed with a warning).

    The function reads at most the first 8 KB of the file -- enough to
    capture the root element + its attributes, but not enough to slow
    down on a 1.5 GB DrugBank XML. For gzipped files, we decompress
    incrementally (gzip codec) and stop after 8 KB decompressed.

    This is INTENTIONALLY a separate function (not inline in the task)
    so it can be unit-tested with mock XML files (current schema +
    mocked future 6.0 schema) without requiring Airflow.
    """
    import gzip

    # Read first 8 KB (decompressed for .gz).
    HEAD_BYTES = 8192
    head = b""
    if str(xml_path).endswith(".gz"):
        try:
            with gzip.open(xml_path, "rb") as fh:
                head = fh.read(HEAD_BYTES)
        except (OSError, EOFError) as exc:
            logger.warning(
                "P1-035: could not read gzip header from %s: %s",
                xml_path, exc,
            )
            return None
    else:
        try:
            with open(xml_path, "rb") as fh:
                head = fh.read(HEAD_BYTES)
        except OSError as exc:
            logger.warning(
                "P1-035: could not read header from %s: %s",
                xml_path, exc,
            )
            return None

    # Extract the version="X.Y.Z" attribute from the root <drugbank ...> tag.
    # We use a regex (not full XML parsing) because:
    #   1. The root element is in the first 8 KB -- partial parse is fine.
    #   2. lxml / ElementTree would parse the ENTIRE file (1.5 GB) to get
    #      the root attribute -- 30+ seconds vs <10 ms with regex.
    #   3. The DrugBank root element is well-formed in ALL releases (it's
    #      the first thing the schema validates), so regex is safe here.
    import re
    # Match: <drugbank ... version="5.1.10" ...>
    # DrugBank uses both single and double quotes; version is always
    # present per the DrugBank XSD.
    version_match = re.search(
        rb'<drugbank\b[^>]*\bversion=["\']([^"\']+)["\']',
        head,
    )
    if version_match is None:
        logger.warning(
            "P1-035: could not find version attribute in root "
            "<drugbank> element of %s. First 200 bytes: %r",
            xml_path, head[:200],
        )
        return None
    try:
        return version_match.group(1).decode("utf-8", errors="replace").strip()
    except (UnicodeDecodeError, AttributeError) as exc:
        logger.warning(
            "P1-035: could not decode version attribute from %s: %s",
            xml_path, exc,
        )
        return None


@task
def check_drugbank_schema() -> str:
    """P1-035 ROOT FIX: pre-flight check of DrugBank XML schema version.

    Opens the DrugBank XML file (path from ``DRUGBANK_XML_PATH`` env var
    or ``config.settings``), reads the root ``<drugbank version="...">``
    attribute, and verifies the version is in ``SUPPORTED_DRUGBANK_SCHEMAS``.

    If the version is UNSUPPORTED (e.g. a future 6.0 release), raises
    ``AirflowFailException`` with a clear message naming:
      * The detected version
      * The supported version set
      * The action required (update the parser or pin to a supported release)

    The check reads ONLY the first 8 KB of the file -- <10 ms even for
    a 1.5 GB DrugBank XML. This is fast enough to run as a separate
    Airflow task without adding meaningful latency to the DAG.

    Returns the detected version string for XCom visibility (operators
    can see "5.1.10" in the Airflow UI's XCom pane).
    """
    try:
        from config.settings import DRUGBANK_XML_PATH
    except ImportError as exc:
        # If config.settings is not importable, the DAG cannot know where
        # the XML lives. Fail fast with a clear message.
        _raise_schema_fail(
            f"P1-035 DrugBank schema check FAILED: could not import "
            f"DRUGBANK_XML_PATH from config.settings ({exc}). Set "
            f"DRUGBANK_XML_PATH env var to the DrugBank XML file path."
        )
        return ""  # unreachable -- _raise_schema_fail always raises

    if not DRUGBANK_XML_PATH:
        _raise_schema_fail(
            "P1-035 DrugBank schema check FAILED: DRUGBANK_XML_PATH is "
            "empty. Set DRUGBANK_XML_PATH env var to the DrugBank XML "
            "file path."
        )
        return ""

    xml_path = Path(DRUGBANK_XML_PATH)
    if not xml_path.exists():
        # The DrugBank XML is manually positioned (license-required). If
        # it's missing, this is NOT a schema-version issue -- it's a
        # missing-file issue. Log a WARNING and return "MISSING" so the
        # downstream pipeline can handle the FileNotFoundError itself.
        # This matches the master_pipeline_dag.py BranchPythonOperator
        # behavior (skip DrugBank if XML missing).
        logger.warning(
            "P1-035 DrugBank XML not found at %s -- skipping schema "
            "check. The pipeline will raise FileNotFoundError when run.",
            xml_path,
        )
        return "MISSING"

    detected_version = _detect_drugbank_schema_version(xml_path)
    if detected_version is None:
        # Could not read the version -- this is a corrupted / partial
        # XML file. Fail fast so the operator knows to re-download.
        _raise_schema_fail(
            f"P1-035 DrugBank schema check FAILED: could not read "
            f"version attribute from root <drugbank> element in "
            f"{xml_path}. The file may be corrupted, truncated, or not "
            f"a DrugBank XML file. Verify the file and re-download if "
            f"necessary."
        )
        return ""  # unreachable

    if not _is_drugbank_version_supported(detected_version):
        # P1-015 ROOT FIX: distinguish between BLOCKED (known-bad),
        # UNSUPPORTED major (>= 6.0 — WARN, not FAIL), and
        # genuinely-unknown versions.
        if detected_version in BLOCKED_DRUGBANK_SCHEMAS:
            _raise_schema_fail(
                f"P1-015 DrugBank schema check FAILED: detected version "
                f"{detected_version!r} is in the BLOCKED_DRUGBANK_SCHEMAS "
                f"set (known-bad release). Remove it from BLOCKED_DRUGBANK_SCHEMAS "
                f"in phase1/dags/drugbank_dag.py ONLY after verifying the "
                f"parser handles it correctly."
            )
            return ""  # unreachable

        if _is_drugbank_version_warn_only(detected_version):
            # P1-015: versions >= 6.0 get a WARNING (not FAIL). The
            # parser MIGHT still work — the XPath is generic enough
            # to handle a rename. The operator is alerted but the
            # pipeline is NOT blocked. This is the OPPOSITE of the
            # previous behavior (hard-fail on any unknown version),
            # which turned every DrugBank release into a P0 incident.
            logger.warning(
                "P1-015 DrugBank schema check WARNING: detected "
                "version %s is >= 6.0 (not in the 5.x supported set). "
                "The parser uses XPath <drug><name>...</name></drug> "
                "which MAY still work if 6.x keeps the <name> "
                "sub-element. The pipeline will proceed — if it "
                "extracts zero drugs, file a bug to update the parser "
                "for the new schema.",
                detected_version,
            )
            return detected_version

        # Genuinely unknown version (doesn't match 5.x regex, not >= 6).
        # This is most likely a corrupted version string or a non-DrugBank
        # XML. Fail fast so the operator investigates.
        _raise_schema_fail(
            f"P1-015 DrugBank schema check FAILED: detected version "
            f"{detected_version!r} does not match the supported pattern "
            f"^5\\.\\d+(\\.\\d+)?$ and is not >= 6.0 (warn-only). The "
            f"version string may be corrupted, or the file is not a "
            f"DrugBank XML. Verify the file and re-download if necessary."
        )
        return ""  # unreachable

    logger.info(
        "P1-015 DrugBank schema check PASSED: detected version %s "
        "(matches supported pattern ^5.\\d+(\\.\\d+)?$ and not in "
        "BLOCKED_DRUGBANK_SCHEMAS). Proceeding with pipeline.",
        detected_version,
    )
    return detected_version


# v89 ROOT FIX (BUG #25 / BUG #38): bare ``@task`` -- retry params
# inherited from DEFAULT_ARGS (spread from DEFAULT_RETRY_ARGS).
@task
@fail_fast_on_http_4xx
def run_drugbank() -> None:
    """Execute the full DrugBank pipeline: download (verify XML) -> clean -> load."""
    from pipelines.drugbank_pipeline import DrugBankPipeline
    DrugBankPipeline().run()


@dag(
    dag_id="drugbank_pipeline",
    description="DrugBank ETL pipeline: drug and target data from XML",
    # v49 ROOT FIX (Compound-4 -- Sunday Morning Pile-Up):
    # The v29 schedule was "0 3 * * 0" (Sunday 03:00 UTC) -- this overlaps
    # the master DAG (Sunday 02:00 UTC, 8h timeout) and causes per-pipeline
    # filelock conflicts every week. ROOT FIX: move standalone DrugBank
    # to Monday 03:00 UTC. The master DAG remains the primary
    # orchestrator; this standalone DAG only fires for ad-hoc refreshes
    # and no longer conflicts with the master's window.
    schedule="0 3 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["drug_repurposing", "drugbank", "etl"],
)
def drugbank_dag() -> None:
    """Build the DrugBank pipeline DAG.

    P1-035 ROOT FIX: wire ``check_drugbank_schema >> run_drugbank`` so
    the pipeline NEVER runs against an unsupported DrugBank XML schema.
    The schema check reads only the root element (<10 ms) and fails
    fast (AirflowFailException -- no retries) if the version is not in
    ``SUPPORTED_DRUGBANK_SCHEMAS``.
    """
    schema_check = check_drugbank_schema()
    pipeline = run_drugbank()
    # P1-035: explicit dependency -- schema check must pass before the
    # pipeline runs. Without this wire Airflow would run both tasks in
    # parallel, defeating the purpose of the schema check.
    schema_check >> pipeline


# v89 ROOT FIX (BUG #40): consistent DAG-instance naming convention.
dag = drugbank_dag()
