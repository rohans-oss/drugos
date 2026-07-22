"""v50 ROOT FIX: Real auto-downloaders for the 4 no-login biomedical sources.

This module implements FULL automatic downloads from:
  1. ChEMBL -- EBI REST API (https://www.ebi.ac.uk/chembl/api/data/molecule.json)
     No login, no API key. Paginates through all max_phase=4 molecules.
  2. UniProt -- FTP dump (https://ftp.uniprot.org/pub/databases/uniprot/
     current_release/knowledgebase/complete/uniprot_sprot.dat.gz)
     No login. Downloads the full Swiss-Prot curated set (~550K proteins).
  3. STRING -- Direct download (https://stringdb-downloads.org/download/
     protein.links.full.v12.0/9606.protein.links.full.v12.0.txt.gz)
     No login. Downloads the full human PPI network.
  4. PubChem -- PUG-REST for targeted queries + FTP for bulk.
     No login. Uses PUG-REST for property enrichment.

Each function:
  - Detects DRUGOS_DOWNLOAD_MODE env var (sample | full | skip)
  - In "sample" mode: fetches a small subset (50-200 records)
  - In "full" mode: fetches the COMPLETE dataset (may take hours / GB)
  - In "skip" mode: uses existing files only
  - Resumes from partial downloads (HTTP Range headers)
  - Streams to disk (does NOT hold the full dataset in memory)
  - Verifies SHA-256 checksums when published
  - Falls back to embedded samples if the live API is unreachable

The DrugBank solution is in drugbank_pipeline.py (open-data fallback when
academic downloads are paused).
"""
from __future__ import annotations

import gzip
import hashlib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urljoin

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# P1-052 ROOT FIX (v107): helper to sanitize URLs for logging.
# Strips query parameters that may contain API keys (e.g. api_key=...).
def _sanitize_url_for_log(url: str) -> str:
    """Sanitize a URL for logging — redact query params that may contain secrets."""
    if not isinstance(url, str):
        return repr(url)
    # Truncate very long URLs (next_uri can be long)
    if len(url) > 200:
        url = url[:200] + "...[truncated]"
    # Redact api_key, key, token query params
    import re
    url = re.sub(r"([?&](?:api_key|key|token|apikey)=)[^&]+", r"\1[REDACTED]", url, flags=re.IGNORECASE)
    return url


# ─── Constants ──────────────────────────────────────────────────────────

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_PAGE_SIZE = 1000  # Max per ChEMBL API contract
CHEMBL_MAX_RETRIES = 5
CHEMBL_BACKOFF_BASE = 2.0

UNIPROT_SPROT_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete/uniprot_sprot.dat.gz"
)
UNIPROT_REVIEWED_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query="
    "(reviewed:true)&compressed=true"
)

STRING_BASE_URL = "https://stringdb-downloads.org/download"
STRING_DEFAULT_VERSION = "v12.0"
STRING_DEFAULT_ORGANISM = "9606"  # human

PUBCHEM_PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
# v64 ROOT FIX (P1-005): renamed PUBCHEV_FTP_BASE -> PUBCHEM_FTP_BASE
# (typo: 'V' instead of 'M'). The constant was dead code (never referenced).
# Kept and corrected so that future PubChem SDF bulk-download support can use it
# as the canonical base URL. The current pipeline uses PUG REST only, but the
# constant is now a valid, correctly-named reference for that future work.
PUBCHEM_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF"
# v84 FORENSIC ROOT FIX (BUG #46): removed the ``PUBCHEV_FTP_BASE``
# backward-compat alias (typo: 'V' instead of 'M'). No caller
# references it (verified via grep across the entire repo). The typo
# was misleading and the alias was dead code.

# v64 ROOT FIX (P1-006): canonical User-Agent for all HTTP downloads.
# PubChem/NCBI/UniProt/STRING all require a User-Agent header and return
# HTTP 403 when it is missing. This constant is applied in _stream_to_file
# and in every direct requests.get/requests.post call site below.
HTTP_USER_AGENT = "DrugRepurposingPipeline/1.0 (contact=team-cosmic@venturelab.example)"

# 10 well-known FDA-approved drugs for sample mode (InChIKeys + ChEMBL IDs)
SAMPLE_CHEMBL_IDS = [
    # v108 FORENSIC ROOT FIX (ISSUE-P1-003): every ChEMBL ID below was WRONG.
    # The IDs were a jumbled mess: CHEMBL112 is Acetaminophen (not Aspirin),
    # CHEMBL521 is Ibuprofen (not Caffeine), CHEMBL503 is Dihydroergotamine
    # (not Diazepam), CHEMBL2114647 does not exist, CHEMBL546 is Ethinylestradiol
    # (not Metformin), CHEMBL1085 is Levonorgestrel (not Atorvastatin).
    # Verified against ChEMBL API 2026-07-14.
    "CHEMBL25",     # Aspirin
    "CHEMBL112",    # Acetaminophen
    "CHEMBL521",    # Ibuprofen
    "CHEMBL113",    # Caffeine
    "CHEMBL12",     # Diazepam
    "CHEMBL1464",   # Warfarin
    "CHEMBL1431",   # Metformin
    "CHEMBL1487",   # Atorvastatin
    "CHEMBL1560",   # Captopril
    "CHEMBL419213",   # Lisinopril
]

# 8 well-known human proteins for sample mode (UniProt accessions)
SAMPLE_UNIPROT_IDS = [
    "P23219",   # PTGS1 (COX-1)
    "P35354",   # PTGS2 (COX-2)
    "P29274",   # ADORA2A
    "P14867",   # GABRA1
    "Q9BQV0",   # VKORC1
    "P54619",   # PRKAA1 (AMPK)
    "P04035",   # HMGCR
    "P12821",   # ACE
]


def _download_mode() -> str:
    """Read DRUGOS_DOWNLOAD_MODE env var (default 'sample')."""
    return os.environ.get("DRUGOS_DOWNLOAD_MODE", "sample").lower().strip()


def _parse_retry_after(raw: str, *, default: int = 5, max_seconds: int = 300) -> int:
    """Parse an HTTP ``Retry-After`` header value (P1-014 root fix).

    The HTTP spec (RFC 7231 §7.1.3) allows ``Retry-After`` to be EITHER:
      * an integer number of seconds (e.g. ``"120"``), OR
      * an HTTP-date string (e.g. ``"Wed, 21 Oct 2025 07:28:00 GMT"``).

    The previous code did ``int(raw)`` which raised ``ValueError`` on the
    HTTP-date form, silently falling into the generic retry path with the
    default backoff -- ignoring the server's requested wait time and risking
    additional 429s if the server's wait was longer.

    Root fix: try integer seconds first; if that fails, parse as HTTP-date
    via ``email.utils.parsedate_to_datetime`` and compute the remaining
    seconds from now. Clamp to ``max_seconds`` to avoid pathological waits.
    Returns ``default`` on any parse failure so the caller always gets a
    usable positive integer.
    """
    if not raw or not isinstance(raw, str):
        return default
    raw = raw.strip()
    # Case 1: integer seconds.
    try:
        seconds = int(raw)
        return max(0, min(seconds, max_seconds))
    except ValueError:
        pass
    # Case 2: HTTP-date form.
    try:
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            seconds = int((dt - now).total_seconds())
            return max(0, min(seconds, max_seconds))
    except (TypeError, ValueError, OverflowError):
        pass
    return default


def _stream_to_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: Optional[str] = None,
    timeout: tuple[float, float] = (30.0, 600.0),
    chunk_size: int = 256 * 1024,
) -> Path:
    """Stream a URL to disk atomically (R5, A7).

    Downloads to `dest.tmp` first, then atomically renames to `dest`.
    Supports HTTP Range for resume (records bytes downloaded so far).
    Verifies SHA-256 if `expected_sha256` is provided.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    # Resume support: check if tmp exists, send Range header.
    # v107 ROOT FIX (ISSUE-P1-030 — TOCTOU race): the previous code used
    # ``tmp.stat().st_size if tmp.exists() else 0`` which is a classic
    # time-of-check/time-of-use race. Between ``tmp.exists()`` returning
    # True and ``tmp.stat()`` being called, another process (e.g. a
    # parallel Airflow worker, a cleanup cron, or a disk-pressure eviction)
    # could delete the partial file. ``tmp.stat()`` would then raise
    # ``FileNotFoundError``, crashing the download with a confusing error
    # instead of starting fresh. ROOT FIX: use EAFP (``try / except
    # FileNotFoundError``) — a single atomic syscall, no race window.
    try:
        existing_bytes = tmp.stat().st_size
    except FileNotFoundError:
        existing_bytes = 0
    # v64 ROOT FIX (P1-006): always send a User-Agent header. PubChem/NCBI
    # FTP-mirror and many biomedical APIs return HTTP 403 when the
    # User-Agent is missing. The previous code only set the Range header,
    # so all downloads from UniProt FTP, STRING, and PubChem could fail
    # with 403 in production (masked by embedded-sample fallback).
    headers = {"User-Agent": HTTP_USER_AGENT}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    logger.info("Downloading %s (resume from byte %d) -> %s", url, existing_bytes, dest.name)

    with requests.get(url, headers=headers, stream=True, timeout=timeout, verify=True) as resp:
        if resp.status_code == 416:  # Range Not Satisfiable -- file already complete
            logger.info("Already complete: %s", dest.name)
            # v84 FORENSIC ROOT FIX (BUG #34): os.replace is atomic on
            # both POSIX and Windows; Path.rename fails on Windows if
            # dest already exists.
            os.replace(tmp, dest)
            return dest
        resp.raise_for_status()
        # P2-8 ROOT FIX: the previous code set mode = "ab" whenever
        # existing_bytes > 0 and the server returned 206. But if the
        # server IGNORES the Range header and returns 200 (full content
        # from byte 0), appending to the existing file produces a
        # corrupted file (bytes 0-N prepended twice). Even with 206, if
        # the server returns the WRONG range (e.g. bytes 0-1000 when we
        # asked for bytes 500-1000), the append writes wrong bytes at
        # offset 500. ROOT FIX: for resumed downloads (206), validate
        # the Content-Range header. If the range doesn't start at the
        # expected offset, discard the existing tmp file and start fresh.
        if existing_bytes > 0 and resp.status_code == 206:
            content_range = resp.headers.get("Content-Range", "")
            # Content-Range format: "bytes START-END/TOTAL" or "bytes */TOTAL"
            range_start = None
            if content_range.startswith("bytes "):
                try:
                    range_spec = content_range.split(" ", 1)[1].split("/")[0]
                    range_start = int(range_spec.split("-")[0])
                except (ValueError, IndexError):
                    range_start = None
            if range_start is not None and range_start == existing_bytes:
                # Server returned the correct range -- safe to append.
                mode = "ab"
            else:
                # P2-8 ROOT FIX: Content-Range is missing, malformed, or
                # doesn't match the expected resume offset. The safest
                # action is to discard the partial file and start fresh.
                logger.warning(
                    "Resume mismatch for %s: expected range start %d, "
                    "got Content-Range=%r. Discarding partial download "
                    "and starting fresh.",
                    dest.name, existing_bytes, content_range,
                )
                mode = "wb"
                existing_bytes = 0  # reset so SHA-256 covers full file
        else:
            mode = "wb"
        sha = hashlib.sha256()
        with open(tmp, mode) as f:
            bytes_written = existing_bytes if mode == "ab" else 0
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    sha.update(chunk)
                    bytes_written += len(chunk)
                    if bytes_written % (100 * 1024 * 1024) < chunk_size:
                        logger.info(
                            "  %s: %d MB downloaded", dest.name, bytes_written // (1024 * 1024)
                        )
        actual_sha = sha.hexdigest() if mode == "wb" else None

    # v64 ROOT FIX (P1-008): for resumed downloads (mode == "ab"), the
    # incremental sha above only covers the NEW bytes -- not the whole
    # file. The previous code set actual_sha = None for resumed downloads
    # and silently skipped checksum verification, so a corrupted+resumed
    # file (e.g. a proxy injecting HTML into the resumed portion) would
    # pass undetected. Root fix: re-hash the FULL file after resume so
    # the checksum covers every byte on disk.
    if mode == "ab" and expected_sha256:
        full_sha = hashlib.sha256()
        with open(tmp, "rb") as f_full:
            for chunk in iter(lambda: f_full.read(chunk_size), b""):
                full_sha.update(chunk)
        actual_sha = full_sha.hexdigest()

    # Verify checksum if provided.
    # v64: actual_sha is now set for BOTH fresh and resumed downloads
    # (see the P1-008 fix above), so this check fires correctly in both cases.
    if expected_sha256 and actual_sha and actual_sha != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for {dest.name}: expected {expected_sha256}, got {actual_sha}"
        )

    # v84 FORENSIC ROOT FIX (BUG #34): os.replace is atomic on both
    # POSIX and Windows. Path.rename (os.rename) fails on Windows if
    # the destination already exists, breaking re-downloads.
    os.replace(tmp, dest)
    logger.info("Download complete: %s (%d bytes)", dest.name, dest.stat().st_size)
    return dest


# ─── 1. ChEMBL ─────────────────────────────────────────────────────────

def download_chembl_full(raw_dir: Path) -> dict[str, Path]:
    """Download the FULL ChEMBL FDA-approved molecule set + bioactivities.

    Uses the ChEMBL REST API (https://www.ebi.ac.uk/chembl/api/data).
    No login, no API key required.

    Downloads:
      - molecules.jsonl   -- all max_phase=4 molecules (~10,000 compounds)
      - activities.jsonl  -- all bioactivities for those molecules (~2M rows)
      - targets.jsonl     -- all ChEMBL targets referenced by activities

    In sample mode: fetches only the 10 sample molecules (via ChEMBL API).
    Falls back to embedded samples if the API is unreachable.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    mode = _download_mode()
    result: dict[str, Path] = {}

    if mode == "skip":
        logger.info("ChEMBL: skip mode -- using existing files")
        return {
            "molecules": raw_dir / "chembl_molecules.jsonl",
            "activities": raw_dir / "chembl_activities.jsonl",
        }

    if mode == "sample":
        logger.info("ChEMBL: SAMPLE mode -- fetching %d molecules via API", len(SAMPLE_CHEMBL_IDS))
        molecules_path = raw_dir / "chembl_molecules.jsonl"
        activities_path = raw_dir / "chembl_activities.jsonl"
        import json

        with open(molecules_path, "w") as f_mol, open(activities_path, "w") as f_act:
            for chembl_id in SAMPLE_CHEMBL_IDS:
                try:
                    url = f"{CHEMBL_API_BASE}/molecule/{chembl_id}.json"
                    # v64 ROOT FIX (P1-006): send User-Agent header.
                    resp = requests.get(url, headers={"User-Agent": HTTP_USER_AGENT}, timeout=30.0)
                    if resp.status_code == 200:
                        f_mol.write(json.dumps(resp.json()) + "\n")
                    # Fetch activities for this molecule
                    act_url = f"{CHEMBL_API_BASE}/activity.json"
                    act_params = {"molecule_chembl_id": chembl_id, "limit": 100}
                    # v64 ROOT FIX (P1-006): send User-Agent header.
                    act_resp = requests.get(act_url, params=act_params, headers={"User-Agent": HTTP_USER_AGENT}, timeout=30.0)
                    if act_resp.status_code == 200:
                        for act in act_resp.json().get("activities", []):
                            f_act.write(json.dumps(act) + "\n")
                    time.sleep(0.5)  # rate-limit courtesy
                except (requests.RequestException, OSError, ValueError, json.JSONDecodeError) as exc:
                    # v90 ROOT FIX (BUG #22): narrowed from broad
                    # ``except Exception`` which caught programming bugs
                    # (AttributeError, KeyError, NameError) and silently
                    # skipped them. Root fix: catch ONLY network/data
                    # errors. Programming bugs propagate.
                    logger.warning("ChEMBL sample fetch failed for %s: %s", chembl_id, exc)

        # If we got 0 molecules, fall back to embedded samples
        if molecules_path.stat().st_size == 0:
            logger.warning("ChEMBL: API unreachable -- falling back to embedded samples")
            from pipelines._dev_samples import embedded_chembl_molecules, embedded_chembl_activities
            # v107 ROOT FIX (ISSUE-P1-028 — silent CSV/JSONL contract switch):
            # The previous v85/v90 fix changed the fallback to write .csv
            # and unlink the .jsonl — but the docstring above AND the
            # downstream chembl_pipeline.clean() reader BOTH expect JSONL.
            # Switching extensions silently broke the format/extension
            # contract: downstream pd.read_json(lines=True) would crash
            # (FileNotFoundError on the .jsonl) or, worse, a stale .jsonl
            # from a previous run would be silently read as if it were
            # the fallback output. ROOT FIX: write JSONL to the SAME
            # .jsonl path the contract specifies. Each DataFrame row is
            # serialized with json.dumps() + "\n", matching the format
            # the API path produces (line 332: f_mol.write(json.dumps(...)+"\n")).
            # No extension switch, no unlink, no contract drift.
            import json as _json
            emb_mol_df = embedded_chembl_molecules()
            emb_act_df = embedded_chembl_activities()
            with open(molecules_path, "w") as f_mol:
                for record in emb_mol_df.to_dict(orient="records"):
                    f_mol.write(_json.dumps(record, default=str) + "\n")
            with open(activities_path, "w") as f_act:
                for record in emb_act_df.to_dict(orient="records"):
                    f_act.write(_json.dumps(record, default=str) + "\n")
            result["molecules"] = molecules_path
            result["activities"] = activities_path
        else:
            result["molecules"] = molecules_path
            result["activities"] = activities_path
        return result

    # FULL mode: paginate through ALL max_phase=4 molecules
    # v64 ROOT FIX (P1-009): the previous code used pure offset-based paging
    # (`offset += CHEMBL_PAGE_SIZE`). ChEMBL's REST API has a hard limit on
    # the `offset` parameter (typically 10,000) -- beyond this the API
    # returns 400 Bad Request or silently returns empty results. For the
    # ~10K max_phase=4 molecule corpus this is borderline, but for the
    # activities endpoint (millions of rows) it is a hard blocker.
    # Root fix: use cursor-based pagination via `page_meta.next_uri` (which
    # ChEMBL provides for exactly this purpose). The first request uses
    # offset=0; subsequent requests follow `next_uri` until it is absent
    # or empty. We still cap at a safety maximum to avoid infinite loops.
    logger.info("ChEMBL: FULL mode -- paginating through all max_phase=4 molecules (cursor-based)")
    molecules_path = raw_dir / "chembl_molecules.jsonl"
    activities_path = raw_dir / "chembl_activities.jsonl"
    import json

    total_molecules = 0
    total_activities = 0
    # v64 safety cap: 50 pages * 1000 = 50K molecules (well above the ~10K
    # max_phase=4 corpus). Prevents infinite loops if next_uri is malformed.
    CHEMBL_MAX_PAGES = 50
    next_uri: Optional[str] = None
    with open(molecules_path, "w") as f_mol:
        page_num = 0
        while page_num < CHEMBL_MAX_PAGES:
            if next_uri:
                # Cursor-based: follow the server-provided next_uri verbatim.
                url = next_uri
                params = None  # next_uri already contains all query params
            else:
                # First page: use offset=0 with the standard filter.
                url = f"{CHEMBL_API_BASE}/molecule.json"
                params = {
                    "max_phase": 4,
                    "format": "json",
                    "limit": CHEMBL_PAGE_SIZE,
                    "offset": 0,
                }
            # v64 ROOT FIX (P1-006): send User-Agent on every ChEMBL request
            # (NCBI/EBI may 403 without it).
            req_headers = {"User-Agent": HTTP_USER_AGENT}
            for attempt in range(CHEMBL_MAX_RETRIES):
                try:
                    resp = requests.get(url, params=params, headers=req_headers, timeout=60.0)
                    if resp.status_code == 429:
                        # v64 ROOT FIX (P1-014): Retry-After can be either
                        # an integer (seconds) OR an HTTP-date string
                        # (e.g. "Wed, 21 Oct 2025 07:28:00 GMT"). The
                        # previous `int(...)` raised ValueError on the
                        # HTTP-date form, silently falling into the generic
                        # retry path with the default backoff -- ignoring
                        # the server's requested wait time.
                        retry_after_raw = resp.headers.get("Retry-After", "5")
                        retry_after = _parse_retry_after(retry_after_raw)
                        logger.info("ChEMBL 429 -- sleeping %ds", retry_after)
                        time.sleep(retry_after)
                        continue
                    resp.raise_for_status()
                    break
                except (requests.exceptions.RequestException, ValueError) as exc:
                    # P1-052 ROOT FIX (v107): narrowed from broad
                    # ``except Exception``. The previous code caught ALL
                    # exceptions including programming bugs (AttributeError,
                    # NameError, TypeError) — masking real bugs as transient
                    # network errors. A malformed ``next_uri`` (e.g. contains
                    # a null byte) raised ``requests.exceptions.InvalidURL``
                    # which was caught, retried 5 times, then re-raised —
                    # the pagination was stuck on the malformed URI and the
                    # operator saw a generic exception instead of "malformed
                    # next_uri". ROOT FIX: catch ONLY RequestException
                    # (network/HTTP errors) and ValueError (JSON decode
                    # errors, InvalidURL). Programming bugs propagate
                    # immediately. Log the URL (sanitized) so the operator
                    # can report malformed URIs to ChEMBL.
                    if attempt == CHEMBL_MAX_RETRIES - 1:
                        logger.error(
                            "ChEMBL pagination failed after %d attempts. "
                            "URL was: %s. If this is a next_uri, it may be "
                            "malformed — report to ChEMBL. Error: %s",
                            CHEMBL_MAX_RETRIES,
                            _sanitize_url_for_log(url),
                            exc,
                        )
                        raise
                    wait = CHEMBL_BACKOFF_BASE ** attempt
                    logger.warning(
                        "ChEMBL attempt %d failed: %s -- retry in %.1fs. "
                        "URL: %s",
                        attempt + 1, exc, wait, _sanitize_url_for_log(url),
                    )
                    time.sleep(wait)
            data = resp.json()
            molecules = data.get("molecules", [])
            if not molecules:
                break
            for mol in molecules:
                f_mol.write(json.dumps(mol) + "\n")
            total_molecules += len(molecules)
            logger.info("ChEMBL: %d molecules fetched (page=%d)", total_molecules, page_num + 1)
            # Also fetch activities for each molecule (batched)
            for mol in molecules:
                mol_id = mol.get("molecule_chembl_id")
                if not mol_id:
                    continue
                try:
                    act_url = f"{CHEMBL_API_BASE}/activity.json"
                    act_params = {"molecule_chembl_id": mol_id, "limit": 1000}
                    # v64 ROOT FIX (P1-006): send User-Agent on activity fetches.
                    act_resp = requests.get(act_url, params=act_params, headers={"User-Agent": HTTP_USER_AGENT}, timeout=60.0)
                    if act_resp.status_code == 200:
                        with open(activities_path, "a") as f_act:
                            for act in act_resp.json().get("activities", []):
                                f_act.write(json.dumps(act) + "\n")
                                total_activities += 1
                    time.sleep(0.1)  # rate-limit courtesy (10 req/sec)
                except Exception as exc:
                    logger.debug("Activity fetch failed for %s: %s", mol_id, exc)
            # v64 ROOT FIX (P1-009): prefer cursor-based next_uri over offset.
            page_meta = data.get("page_meta", {}) or {}
            next_uri = page_meta.get("next_uri")
            page_num += 1
            if not next_uri:
                # No next_uri -> end of results (or server doesn't support
                # cursor paging for this endpoint). Stop cleanly.
                break

    logger.info("ChEMBL FULL: %d molecules, %d activities", total_molecules, total_activities)
    result["molecules"] = molecules_path
    result["activities"] = activities_path
    return result


# ─── 2. UniProt ────────────────────────────────────────────────────────

def download_uniprot_full(raw_dir: Path) -> dict[str, Path]:
    """Download the UniProt Swiss-Prot reviewed protein set.

    Uses the UniProt REST API (https://rest.uniprot.org).
    No login required.

    In sample mode: fetches only the 8 sample proteins via REST.
    In full mode: streams the full uniprot_sprot.dat.gz (~500MB compressed).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    mode = _download_mode()
    result: dict[str, Path] = {}

    if mode == "skip":
        logger.info("UniProt: skip mode -- using existing files")
        return {"proteins": raw_dir / "uniprot_sprot.dat.gz"}

    if mode == "sample":
        logger.info("UniProt: SAMPLE mode -- fetching %d proteins via REST", len(SAMPLE_UNIPROT_IDS))
        proteins_path = raw_dir / "uniprot_proteins.jsonl"
        import json

        with open(proteins_path, "w") as f:
            for acc in SAMPLE_UNIPROT_IDS:
                try:
                    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
                    # v64 ROOT FIX (P1-006): send User-Agent header.
                    resp = requests.get(url, headers={"User-Agent": HTTP_USER_AGENT}, timeout=30.0)
                    if resp.status_code == 200:
                        f.write(json.dumps(resp.json()) + "\n")
                    time.sleep(0.5)  # rate-limit courtesy
                except Exception as exc:
                    logger.warning("UniProt sample fetch failed for %s: %s", acc, exc)

        if proteins_path.stat().st_size == 0:
            logger.warning("UniProt: API unreachable -- falling back to embedded samples")
            from pipelines._dev_samples import embedded_uniprot_proteins
            embedded_uniprot_proteins().to_csv(proteins_path.with_suffix(".csv"), index=False)
            proteins_path.unlink(missing_ok=True)
            result["proteins"] = proteins_path.with_suffix(".csv")
        else:
            result["proteins"] = proteins_path
        return result

    # FULL mode: stream the full Swiss-Prot dat.gz
    logger.info("UniProt: FULL mode -- streaming uniprot_sprot.dat.gz (~500MB)")
    dest = raw_dir / "uniprot_sprot.dat.gz"
    _stream_to_file(UNIPROT_SPROT_URL, dest, timeout=(30.0, 7200.0))
    result["proteins"] = dest
    return result


# ─── 3. STRING ─────────────────────────────────────────────────────────

def download_string_full(raw_dir: Path, organism: str = STRING_DEFAULT_ORGANISM) -> dict[str, Path]:
    """Download the STRING PPI network for the given organism.

    Uses the STRING download server (https://stringdb-downloads.org).
    No login required.

    In sample mode: fetches a small subset via the STRING API.
    In full mode: downloads the full protein.links.full.vXX.txt.gz
    (human: ~400MB compressed, ~4M PPI edges).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    mode = _download_mode()
    version = os.environ.get("STRING_VERSION", STRING_DEFAULT_VERSION)
    result: dict[str, Path] = {}

    if mode == "skip":
        logger.info("STRING: skip mode -- using existing files")
        return {"ppi": raw_dir / f"{organism}.protein.links.full.{version}.txt.gz"}

    if mode == "sample":
        logger.info("STRING: SAMPLE mode -- fetching PPIs for %d proteins via API", len(SAMPLE_UNIPROT_IDS))
        ppi_path = raw_dir / "string_ppi_sample.tsv"
        import json

        # Use STRING's API to fetch PPIs between sample proteins
        string_api = "https://string-db.org/api"
        with open(ppi_path, "w") as f:
            f.write("protein1\tprotein2\tcombined_score\texperimental_score\tdatabase_score\ttextmining_score\n")
            # Get STRING IDs for our sample UniProt accessions
            string_ids = {}
            for acc in SAMPLE_UNIPROT_IDS:
                try:
                    url = f"{string_api}/tsv/get_string_ids"
                    params = {
                        "identifiers": acc,
                        "species": organism,
                        "limit": 1,
                    }
                    # v64 ROOT FIX (P1-006): send User-Agent header.
                    resp = requests.post(url, data=params, headers={"User-Agent": HTTP_USER_AGENT}, timeout=30.0)
                    if resp.status_code == 200 and resp.text.strip():
                        for line in resp.text.strip().split("\n")[1:]:
                            parts = line.split("\t")
                            if len(parts) >= 3:
                                string_ids[acc] = parts[2]
                                break
                    time.sleep(0.5)
                except Exception as exc:
                    logger.debug("STRING ID lookup failed for %s: %s", acc, exc)

            # Fetch PPIs between pairs of sample proteins
            string_id_list = list(string_ids.values())
            for i, p1 in enumerate(string_id_list):
                for p2 in string_id_list[i:]:
                    try:
                        url = f"{string_api}/json/interaction_partners"
                        params = {
                            # v64 ROOT FIX (P1-007): STRING's interaction_partners
                            # API expects multiple identifiers separated by `%0a`
                            # (URL-encoded LF / newline) or a bare `\n`. The
                            # previous code used `%0d` (CR), which STRING may
                            # treat as part of a single identifier -- returning
                            # zero interactions and masking the bug via the
                            # embedded-sample fallback.
                            "identifiers": f"{p1}%0a{p2}",
                            "species": organism,
                            "limit": 1,
                        }
                        # v64 ROOT FIX (P1-006): send User-Agent header.
                        resp = requests.post(url, data=params, headers={"User-Agent": HTTP_USER_AGENT}, timeout=30.0)
                        if resp.status_code == 200:
                            for interaction in resp.json():
                                f.write(
                                    f"{interaction.get('stringId_A')}\t"
                                    f"{interaction.get('stringId_B')}\t"
                                    f"{interaction.get('score', 0)}\t"
                                    f"{interaction.get('experimental', 0)}\t"
                                    f"{interaction.get('database', 0)}\t"
                                    f"{interaction.get('textmining', 0)}\n"
                                )
                        time.sleep(0.5)
                    except Exception as exc:
                        logger.debug("STRING PPI fetch failed for %s/%s: %s", p1, p2, exc)

        if ppi_path.stat().st_size < 100:  # essentially empty
            logger.warning("STRING: API unreachable -- falling back to embedded samples")
            from pipelines._dev_samples import embedded_string_ppi
            embedded_string_ppi().to_csv(ppi_path.with_suffix(".csv"), index=False)
            ppi_path.unlink(missing_ok=True)
            result["ppi"] = ppi_path.with_suffix(".csv")
        else:
            result["ppi"] = ppi_path
        return result

    # FULL mode: download the full protein.links.full file
    logger.info("STRING: FULL mode -- downloading %s.protein.links.full.%s.txt.gz", organism, version)
    url = f"{STRING_BASE_URL}/protein.links.full.{version}/{organism}.protein.links.full.{version}.txt.gz"
    dest = raw_dir / f"{organism}.protein.links.full.{version}.txt.gz"
    _stream_to_file(url, dest, timeout=(30.0, 3600.0))
    result["ppi"] = dest
    return result


# ─── 4. PubChem ────────────────────────────────────────────────────────

def download_pubchem_full(raw_dir: Path, inchikeys: list[str] | None = None) -> dict[str, Path]:
    """Download PubChem physicochemical properties for compounds.

    Uses PubChem PUG-REST (https://pubchem.ncbi.nlm.nih.gov/rest/pug).
    No login required.

    PubChem has 110M+ compounds -- we cannot download all of them.
    Instead, we enrich the compounds already in our KG (identified by
    InChIKey). This is the standard pattern: PubChem enrichment is
    always compound-targeted, not bulk.

    In sample mode: enriches the 10 sample InChIKeys.
    In full mode: enriches ALL InChIKeys from Phase 1's drugs table.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    mode = _download_mode()
    result: dict[str, Path] = {}

    # Determine which InChIKeys to enrich
    # v83 FORENSIC ROOT FIX (P1-8): the previous code silently switched
    # from DrugBank InChIKeys to ChEMBL InChIKeys when drugbank_drugs.csv
    # was missing. DrugBank and ChEMBL may produce different InChIKeys
    # for the same drug (salt-form differences, stereochemistry handling).
    # The PubChem enrichment output did NOT record which drug source was
    # used -- making provenance ambiguous. ROOT FIX: track the drug source
    # explicitly (``drug_source``) and write it into the enrichment CSV
    # as a new column so downstream consumers (KG builder, audits) know
    # which drug source each PubChem-enriched row came from.
    drug_source = "sample"
    if inchikeys is None:
        if mode == "sample":
            from pipelines._dev_samples import embedded_chembl_molecules
            inchikeys = list(embedded_chembl_molecules()["inchikey"])
            drug_source = "embedded_sample"
        else:
            # FULL mode: read InChIKeys from Phase 1's processed_data/drugbank_drugs.csv
            # OR from the PostgreSQL drugs table.
            try:
                import pandas as pd
                drugs_csv = raw_dir.parent / "processed_data" / "drugbank_drugs.csv"
                if drugs_csv.exists():
                    df = pd.read_csv(drugs_csv)
                    inchikeys = list(df["inchikey"].dropna().unique())
                    drug_source = "drugbank"
                    logger.info("PubChem: enriching %d InChIKeys from DrugBank (%s)", len(inchikeys), drugs_csv)
                else:
                    # Fall back to ChEMBL drugs CSV -- but RECORD the source switch.
                    chembl_csv = raw_dir.parent / "processed_data" / "chembl_drugs.csv"
                    if chembl_csv.exists():
                        df = pd.read_csv(chembl_csv)
                        inchikeys = list(df["inchikey"].dropna().unique())
                        drug_source = "chembl"
                        logger.warning(
                            "PubChem: DrugBank drugs CSV not found at %s -- "
                            "FALLING BACK to ChEMBL drugs CSV (%s). Note: "
                            "DrugBank and ChEMBL may produce different "
                            "InChIKeys for the same drug (salt-form / stereo "
                            "differences). The drug_source column in the "
                            "enrichment output records which source was used.",
                            drugs_csv, chembl_csv,
                        )
                    else:
                        logger.warning("PubChem: no drugs CSV found -- using sample InChIKeys")
                        from pipelines._dev_samples import embedded_chembl_molecules
                        inchikeys = list(embedded_chembl_molecules()["inchikey"])
                        drug_source = "embedded_sample"
            except Exception as exc:
                logger.warning("PubChem: failed to read InChIKeys -- using samples: %s", exc)
                from pipelines._dev_samples import embedded_chembl_molecules
                inchikeys = list(embedded_chembl_molecules()["inchikey"])
                drug_source = "embedded_sample"

    if mode == "skip":
        logger.info("PubChem: skip mode -- using existing files")
        return {"enrichment": raw_dir / "pubchem_enrichment.csv"}

    logger.info(
        "PubChem: %s mode -- enriching %d InChIKeys via PUG-REST",
        mode, len(inchikeys),
    )
    enrichment_path = raw_dir / "pubchem_enrichment.csv"
    import csv

    # PubChem PUG-REST rate limit: 5 requests per second (200ms between requests)
    with open(enrichment_path, "w", newline="") as f:
        writer = csv.writer(f)
        # v83 P1-8: added ``drug_source`` column so downstream consumers
        # know which drug source (DrugBank / ChEMBL / sample) each
        # PubChem-enriched row came from.
        writer.writerow([
            "inchikey", "pubchem_cid", "canonical_smiles",
            "xlogp", "tpsa",
            "h_bond_donor_count", "h_bond_acceptor_count",
            "rotatable_bond_count", "drug_source",
        ])
        for i, inchikey in enumerate(inchikeys):
            try:
                # Resolve InChIKey -> CID
                url = f"{PUBCHEM_PUG_REST}/compound/inchikey/{inchikey}/property/"
                properties = "CanonicalSMILES,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount"
                # v64 ROOT FIX (P1-015): percent-encode the comma-separated
                # property list. PubChem PUG-REST accepts bare commas in
                # practice, but strict proxies/CDNs may reject them with 400
                # Bad Request (which the pipeline's retry logic treats as
                # permanent, dead-lettering the whole batch). Using
                # urllib.parse.quote(properties, safe="") encodes commas as
                # %2C, which PubChem accepts and strict proxies pass through.
                from urllib.parse import quote as _url_quote
                full_url = url + _url_quote(properties, safe="") + "/JSON"
                # v64 ROOT FIX (P1-006): send User-Agent header.
                resp = requests.get(full_url, headers={"User-Agent": HTTP_USER_AGENT}, timeout=30.0)
                if resp.status_code == 200:
                    data = resp.json()
                    props_list = data.get("PropertyTable", {}).get("Properties", [])
                    if props_list:
                        p = props_list[0]
                        writer.writerow([
                            inchikey,
                            p.get("CID", ""),
                            p.get("CanonicalSMILES", ""),
                            p.get("XLogP", ""),
                            p.get("TPSA", ""),
                            p.get("HBondDonorCount", ""),
                            p.get("HBondAcceptorCount", ""),
                            p.get("RotatableBondCount", ""),
                            drug_source,
                        ])
                elif resp.status_code == 404:
                    logger.debug("PubChem: InChIKey %s not found", inchikey)
                else:
                    logger.debug("PubChem: %s -> HTTP %d", inchikey, resp.status_code)
                # 5 req/sec rate limit
                time.sleep(0.2)
                if (i + 1) % 50 == 0:
                    logger.info("PubChem: %d/%d enriched", i + 1, len(inchikeys))
            except Exception as exc:
                logger.debug("PubChem fetch failed for %s: %s", inchikey, exc)

    # If we got 0 rows (header only), fall back to embedded samples
    if enrichment_path.stat().st_size < 200:
        logger.warning("PubChem: API unreachable -- falling back to embedded samples")
        from pipelines._dev_samples import embedded_pubchem_enrichment
        embedded_pubchem_enrichment().to_csv(enrichment_path, index=False)

    result["enrichment"] = enrichment_path
    return result


# ─── 5. DrugBank open-data fallback ────────────────────────────────────

def download_drugbank_open_data(raw_dir: Path) -> dict[str, Path]:
    """v50 ROOT FIX: DrugBank 100% solution -- open-data fallback.

    DrugBank has paused academic downloads since May 2026. Even registered
    users cannot download the XML file. This function provides a 100%
    solution by combining THREE open-data sources:

    1. ChEMBL FDA-approved subset (max_phase=4) -- provides drug names,
       InChIKeys, SMILES, molecular weights, and mechanisms of action.
       ChEMBL is CC-BY-SA licensed and freely downloadable.

    2. FDA Orange Book open data (https://www.fda.gov/drugs/drug-approvals-
       and-databases/orange-book-data-files) -- provides FDA approval status,
       active ingredients, and reference listed drugs. Public domain.

    3. RxNorm (https://www.nlm.nih.gov/research/umls/rxnorm/) -- provides
       drug -> indication mappings via the RXNREL table. UMLS license
       required but free for research. We use the open RxNorm REST API
       (https://rxnav.nlm.nih.gov/) which requires no login.

    The result is a DrugBank-equivalent dataset with:
      - drugbank_id (synthesized as DB<checksum> for compounds without one)
      - name, inchikey, smiles, molecular_weight
      - indication (from RxNorm / FDA Orange Book)
      - mechanism_of_action (from ChEMBL)
      - is_fda_approved = True (since we filter to FDA-approved)
      - groups = "approved"
      - drug_interactions (from RxNorm)

    When DrugBank academic downloads reopen, set DRUGBANK_XML_PATH to
    use the real XML -- the embedded parser will take precedence.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    mode = _download_mode()
    result: dict[str, Path] = {}

    if mode == "skip":
        return {
            "drugs": raw_dir / "drugbank_open_drugs.csv",
            "indications": raw_dir / "drugbank_open_indications.csv",
        }

    # In sample mode: use embedded DrugBank samples (10 FDA-approved drugs).
    # In full mode: build from ChEMBL FDA-approved + RxNorm.
    if mode == "sample":
        logger.info("DrugBank: SAMPLE mode -- using embedded 10 FDA-approved drugs")
        from pipelines._dev_samples import (
            embedded_drugbank_drugs,
            embedded_drugbank_indications,
        )
        drugs_path = raw_dir / "drugbank_open_drugs.csv"
        indications_path = raw_dir / "drugbank_open_indications.csv"
        embedded_drugbank_drugs().to_csv(drugs_path, index=False)
        embedded_drugbank_indications().to_csv(indications_path, index=False)
        result["drugs"] = drugs_path
        result["indications"] = indications_path
        return result

    # FULL mode: v107 FORENSIC ROOT FIX (ISSUE-P1-005 + ISSUE-P1-015):
    #   P1-005: The previous code SYNTHESIZED fake DrugBank data from
    #   ChEMBL+RxNorm and wrote it to ``drugbank_open_drugs.csv``. The
    #   ``drugbank_id`` was fabricated as ``SYNTH-DB-{8 hex chars from
    #   SHA-256 of inchikey}``. This is NOT real DrugBank data, but it
    #   was labeled as DrugBank and consumed by Phase 2 as if it were.
    #   The KG had phantom DrugBank IDs that 404 against the real DrugBank
    #   API. Any clinical-trial cross-referencing against DrugBank failed
    #   silently. The ``uq_drugs_drugbank_id`` unique index treated
    #   synthesized IDs as unique, preventing future merges with real
    #   DrugBank records when the academic license reopens.
    #
    #   ROOT FIX: do NOT synthesize DrugBank data. If DrugBank academic
    #   downloads are unavailable (paused since May 2026), emit ZERO
    #   DrugBank rows and let the Phase 2 bridge degrade to ChEMBL-only
    #   mode (the bridge's _PHASE1_EXPECTED_COLUMNS now treats drugbank_id
    #   as optional -- see ISSUE-P1-016). The ChEMBL pipeline already
    #   produces chembl_drugs.csv with real FDA-approved drugs; the KG
    #   is built on that real data, not on fabricated DrugBank IDs.
    #
    #   P1-015: the previous ``_synthesize_drugbank_id`` used a
    #   function-attribute counter (``_synthesize_drugbank_id._counter``)
    #   which is NOT thread-safe and NOT process-safe. Two parallel
    #   workers would both start the counter at 0 and generate the same
    #   ``SYNTH-DB-M000001`` ID for different missing-InChIKey drugs --
    #   silent data corruption via ID collision. This entire function is
    #   now DEAD CODE (we no longer synthesize DrugBank IDs), but we keep
    #   the definition for backward-compat imports and make it raise
    #   RuntimeError so any stale caller fails loudly instead of
    #   silently producing fake IDs.
    logger.warning(
        "DrugBank: FULL mode -- academic downloads paused since May 2026. "
        "v107 P1-005: NOT synthesizing fake DrugBank data (would create "
        "phantom drugbank_ids that 404 against the real DrugBank API). "
        "v113 P1-024 ROOT FIX: this used to silently emit ZERO DrugBank "
        "rows and return success -- the operator saw a green DAG run with "
        "no DrugBank data and no clear error. The KG lost the withdrawn-"
        "drug safety signal entirely. Now we RAISE RuntimeError unless "
        "DRUGOS_ALLOW_NO_DRUGBANK=1 is set, so the operator must "
        "explicitly opt into ChEMBL-only degraded mode. The empty CSVs "
        "are still written (with a data_status marker file) so downstream "
        "contract checks pass."
    )

    def _synthesize_drugbank_id(inchikey: str) -> str:  # noqa: ARG001
        """DEAD in v107 -- kept for backward-compat imports.

        v107 P1-005/P1-015: this function previously fabricated DrugBank
        IDs from InChIKey hashes. It is no longer called. If invoked, it
        raises RuntimeError so any stale caller fails loudly instead of
        silently producing fake DrugBank IDs.
        """
        raise RuntimeError(
            "v107 P1-005: _synthesize_drugbank_id() is DISABLED. "
            "DrugBank data synthesis is forbidden -- it creates phantom "
            "drugbank_ids that 404 against the real DrugBank API. Use "
            "real DrugBank academic downloads, or degrade to ChEMBL-only."
        )

    # Emit empty DrugBank CSVs with the correct schema (headers only).
    # This ensures the Phase 2 bridge's contract check passes (files
    # exist) but yields zero rows -- the KG is built from ChEMBL data.
    # Do NOT use the ``drugbank_`` filename prefix for non-DrugBank data.
    drugs_path = raw_dir / "drugbank_open_drugs.csv"
    indications_path = raw_dir / "drugbank_open_indications.csv"
    drugs_df = pd.DataFrame(columns=[
        "drugbank_id", "name", "inchikey", "smiles", "molecular_weight",
        "indication", "indication_source", "mechanism_of_action", "groups",
        "is_fda_approved", "is_withdrawn", "clinical_status", "max_phase",
        "drug_type", "chembl_id", "pubchem_cid",
    ])
    indications_df = pd.DataFrame(columns=[
        "drugbank_id", "drug_inchikey", "drug_name", "disease_id",
        "disease_name", "doid_id", "omim_disease_id", "indication",
        "indication_type", "source",
    ])
    drugs_df.to_csv(drugs_path, index=False)
    indications_df.to_csv(indications_path, index=False)
    # v113 P1-024 ROOT FIX: write a data_status marker file so the
    # dashboard / ops team can surface "DrugBank missing" as a
    # first-class data-quality signal. The marker is read by the
    # bridge's manifest emission (see ``phase1_bridge._emit_manifest``)
    # and surfaced in the run summary.
    data_status_path = raw_dir / "drugbank_data_status.json"
    import json as _json
    data_status_path.write_text(_json.dumps({
        "source": "drugbank",
        "status": "drugbank_missing",
        "mode": "full",
        "reason": (
            "DrugBank academic downloads paused since May 2026. "
            "v113 P1-024: FULL mode emits empty CSVs and raises "
            "RuntimeError unless DRUGOS_ALLOW_NO_DRUGBANK=1 is set. "
            "The KG will be built from ChEMBL FDA-approved drugs; "
            "withdrawn-drug safety signal is ABSENT until real "
            "DrugBank data is provided."
        ),
        "rows_drugs": 0,
        "rows_indications": 0,
        "allow_no_drugbank_env": os.environ.get("DRUGOS_ALLOW_NO_DRUGBANK", "0"),
    }, indent=2))
    logger.info(
        "DrugBank: wrote EMPTY drugbank_open_drugs.csv and "
        "drugbank_open_indications.csv (0 rows each) + data_status "
        "marker. KG will use ChEMBL FDA-approved drugs as the "
        "Compound source."
    )

    # v107 P1-005: the RxNorm enrichment loop below is now DEAD CODE
    # because drugs_df is empty (zero rows). We skip it entirely.
    # When DrugBank academic access resumes, this block should be
    # re-enabled with REAL DrugBank XML data (not synthesized IDs).

    result["drugs"] = drugs_path
    result["indications"] = indications_path
    result["data_status"] = data_status_path

    # v113 P1-024 ROOT FIX: RAISE RuntimeError unless the operator has
    # explicitly opted into ChEMBL-only degraded mode. The previous
    # code silently returned success with zero DrugBank rows -- the
    # operator saw a green DAG run with no DrugBank data and no clear
    # error, and the RL ranker's withdrawn-drug safety filter saw NULL
    # for every drug (a withdrawn drug like thalidomide could be
    # recommended as a repurposing candidate). This is a patient-safety
    # bug. Now the operator MUST set DRUGOS_ALLOW_NO_DRUGBANK=1 to
    # acknowledge the degraded mode; otherwise the pipeline fails
    # loudly and the operator can either provide real DrugBank XML or
    # explicitly accept the ChEMBL-only degradation.
    _allow_no_drugbank = os.environ.get("DRUGOS_ALLOW_NO_DRUGBANK", "0")
    if _allow_no_drugbank not in ("1", "true", "True", "TRUE", "yes", "YES"):
        raise RuntimeError(
            "v113 P1-024 ROOT FIX: DrugBank FULL mode produced ZERO rows "
            "(academic downloads paused since May 2026). The previous "
            "code silently returned success -- this hid a patient-safety "
            "bug where the RL ranker's withdrawn-drug safety filter saw "
            "NULL for every drug (withdrawn drugs like thalidomide could "
            "be recommended as repurposing candidates). To acknowledge "
            "ChEMBL-only degraded mode and proceed, set the environment "
            "variable DRUGOS_ALLOW_NO_DRUGBANK=1. To fix properly, "
            "provide real DrugBank XML via DRUGBANK_XML_PATH. The empty "
            "DrugBank CSVs and a data_status marker have been written "
            f"to {raw_dir} so downstream contract checks still pass."
        )
    logger.warning(
        "v113 P1-024: DRUGOS_ALLOW_NO_DRUGBANK=%r is set -- proceeding "
        "in ChEMBL-only degraded mode. The KG will NOT have DrugBank "
        "withdrawn-drug safety signals. This is acceptable for dev/demo "
        "ONLY -- production deployments MUST provide real DrugBank XML.",
        _allow_no_drugbank,
    )
    return result


# ─── CLI entry point ───────────────────────────────────────────────────
# v65 ROOT FIX (P1-039): the previous ``main()`` function here was a
# standalone CLI that DUPLICATED the download logic already provided by
# the package-level CLI (``python -m pipelines run <source>`` ->
# ``pipelines/__init__.py:_main()``). Having two CLIs is a maintenance
# burden -- changes to download logic had to be applied in two places,
# and they had already diverged (this one had no --dry-run, no logging
# config, no manifest emission, no DB integration). The audit's fix is
# "Either remove ``main()`` from ``_v50_downloaders.py`` OR document
# that it's a lower-level utility for direct invocation without
# pipeline orchestration." We choose to REMOVE it -- the package-level
# CLI is the single authoritative entry point. Direct callers who need
# the lower-level ``download_*_full`` functions can import them
# explicitly:
#
#   from pipelines._v50_downloaders import download_chembl_full
#   result = download_chembl_full(Path("raw_data/v50/chembl"))
#
# This is also safer because the lower-level functions don't go through
# BasePipeline's lifecycle (download -> clean -> load), so callers
# explicitly opt into "raw download only" semantics.
#
# (Function body removed; if __name__ == "__main__" guard removed.)
# v84 FORENSIC ROOT FIX (BUG #47): removed the dead
# ``if __name__ == "__main__"`` guard that only raised ``SystemExit``.
# The guard provided no useful functionality -- it was a placeholder
# from the v65 refactor that removed ``main()``. Dead code that adds
# no value has no place in a production codebase.
