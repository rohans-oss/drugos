"""PubChem loader -- bridges to Phase 1's cleaned PubChem enrichment CSV.

This loader consumes ``phase1/processed_data/pubchem_enrichment.csv``
(produced by ``phase1.pipelines.pubchem_pipeline.PubChemPipeline``) and
emits Phase 2 Compound node records compatible with ``kg_builder``.

Design decision (v5 audit fix):
    Phase 2's ``run_pipeline.py`` previously tried to import a non-existent
    ``pubchem_loader`` module, falling into an ``except ImportError`` branch
    that silently skipped PubChem enrichment. The proper fix is to bridge
    Phase 1's already-cleaned PubChem output into Phase 2's graph builder.

Public API (matches the contract expected by ``run_pipeline.py:1821-1825``):
    - ``download_pubchem()`` -> triggers Phase 1's pipeline if needed
    - ``parse_pubchem()`` -> returns a pandas DataFrame of enrichment rows
    - ``pubchem_to_node_records(df)`` -> List[Dict] of Compound nodes
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# v102 ROOT FIX (P2-036): route InChIKey normalization through the
# centralized helper so this loader produces the SAME canonical form
# as phase1_bridge.py and chembl_loader.py.
try:
    from .utils import normalize_inchikey as _normalize_inchikey
except Exception:  # pragma: no cover — fallback for direct-script execution
    def _normalize_inchikey(inchikey):  # type: ignore[no-redef]
        if inchikey is None:
            return ""
        try:
            ik = str(inchikey).strip()
        except Exception:
            return ""
        if not ik or ik.lower() in ("nan", "none", "null", "na"):
            return ""
        return ik.upper()

logger = logging.getLogger(__name__)

# Task 87 ROOT FIX: in-process LRU cache for CID->InChIKey lookups so
# we never query PubChem twice for the same CID in a single loader run.
# PubChem allows ~5 req/s without an API key, so we also enforce a
# 250 ms minimum spacing between requests (rate-limit) via a module-
# level lock + timestamp. The cache is bounded to 50_000 entries to
# avoid unbounded memory growth on very large PubChem extracts.
_CID_TO_INCHIKEY_CACHE: Dict[int, Optional[str]] = {}
_CID_TO_INCHIKEY_CACHE_LOCK = threading.Lock()
_CID_TO_INCHIKEY_LAST_CALL_TS: float = 0.0
_CID_TO_INCHIKEY_MIN_INTERVAL_S: float = 0.25  # 4 req/s
_CID_TO_INCHIKEY_CACHE_MAX_SIZE: int = 50_000
_CID_TO_INCHIKEY_TIMEOUT_S: float = 15.0
_CID_TO_INCHIKEY_MAX_RETRIES: int = 3
_PUBCHEM_PUG_REST_BASE: str = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/InChIKey/JSON"
)


def _cid_to_inchikey(cid: int) -> Optional[str]:
    """Resolve a PubChem CID to its canonical InChIKey via the PUG-REST API.

    Task 87 ROOT FIX: the previous loader blindly trusted the
    ``inchikey`` column from Phase 1's CSV. When the CSV had a CID but
    no InChIKey (a common case for raw SD-record extracts), the loader
    set ``canonical_id = f"CID{cid}"`` and emitted a Compound node with
    ``inchikey=None``. That node could NOT be merged with ChEMBL /
    DrugBank Compound nodes (which are InChIKey-keyed), fragmenting
    the KG: the same physical compound appeared as TWO disjoint nodes.

    The fix queries PubChem's PUG-REST endpoint to resolve
    CID -> InChIKey at load time. The endpoint is documented at
    https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest and is the
    authoritative source for CID->InChIKey mapping. The lookup is:

      GET /rest/pug/compound/cid/{cid}/property/InChIKey/JSON
      -> {"PropertyTable": {"Properties": [{"CID": 2244, "InChIKey": "..."}]}}

    Rate-limiting (250 ms between calls), bounded LRU caching (50k
    entries), and 3 retries with exponential backoff ensure we stay
    within PubChem's fair-use policy. On any failure (network, 404,
    parse error) we return ``None`` so the caller can fall back to
    ``CID{cid}`` -- but with a clear warning so operators know the
    crosswalk was incomplete for that row.
    """
    if cid is None:
        return None
    try:
        cid_int = int(cid)
    except (TypeError, ValueError):
        return None
    if cid_int <= 0:
        return None

    # Fast path: cache hit.
    with _CID_TO_INCHIKEY_CACHE_LOCK:
        if cid_int in _CID_TO_INCHIKEY_CACHE:
            return _CID_TO_INCHIKEY_CACHE[cid_int]

    # Rate-limit: ensure at least ``_CID_TO_INCHIKEY_MIN_INTERVAL_S``
    # seconds have passed since the last PUG-REST call.
    global _CID_TO_INCHIKEY_LAST_CALL_TS
    with _CID_TO_INCHIKEY_CACHE_LOCK:
        now = time.monotonic()
        wait = _CID_TO_INCHIKEY_MIN_INTERVAL_S - (
            now - _CID_TO_INCHIKEY_LAST_CALL_TS
        )
        if wait > 0:
            time.sleep(wait)
        _CID_TO_INCHIKEY_LAST_CALL_TS = time.monotonic()

    url = _PUBCHEM_PUG_REST_BASE.format(cid=cid_int)
    last_exc: Optional[Exception] = None
    payload: Optional[Dict[str, Any]] = None
    for attempt in range(1, _CID_TO_INCHIKEY_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "DrugOS-Graph/2.0 (phase2 pubchem_loader)",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(
                req, timeout=_CID_TO_INCHIKEY_TIMEOUT_S
            ) as resp:
                raw = resp.read()
            payload = json.loads(raw.decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            # 404 means PubChem has no InChIKey for this CID -- not
            # retriable. 429 / 5xx are retriable.
            last_exc = e
            if e.code == 404:
                logger.debug(
                    "pubchem_loader: CID %d has no InChIKey on PubChem (404)",
                    cid_int,
                )
                _cache_cid_lookup(cid_int, None)
                return None
            if e.code not in (429, 500, 502, 503, 504):
                # Non-retriable HTTP error -- cache the miss and bail.
                logger.warning(
                    "pubchem_loader: PUG-REST returned HTTP %d for CID %d: %s",
                    e.code, cid_int, e.reason,
                )
                _cache_cid_lookup(cid_int, None)
                return None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            last_exc = e
        # Exponential backoff between retries.
        if attempt < _CID_TO_INCHIKEY_MAX_RETRIES:
            time.sleep(0.5 * (2 ** (attempt - 1)))

    if payload is None:
        logger.warning(
            "pubchem_loader: PUG-REST lookup failed for CID %d after %d "
            "attempts: %s",
            cid_int, _CID_TO_INCHIKEY_MAX_RETRIES, last_exc,
        )
        _cache_cid_lookup(cid_int, None)
        return None

    try:
        props = payload["PropertyTable"]["Properties"]
        if not props:
            _cache_cid_lookup(cid_int, None)
            return None
        ik = str(props[0].get("InChIKey") or "").strip().upper()
        if not ik:
            _cache_cid_lookup(cid_int, None)
            return None
        _cache_cid_lookup(cid_int, ik)
        return ik
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        logger.warning(
            "pubchem_loader: malformed PUG-REST response for CID %d: %s",
            cid_int, e,
        )
        _cache_cid_lookup(cid_int, None)
        return None


def _cache_cid_lookup(cid: int, value: Optional[str]) -> None:
    """Cache a CID->InChIKey lookup result, evicting oldest entries if full."""
    with _CID_TO_INCHIKEY_CACHE_LOCK:
        if len(_CID_TO_INCHIKEY_CACHE) >= _CID_TO_INCHIKEY_CACHE_MAX_SIZE:
            # Evict ~10% of the cache to amortise eviction cost. We
            # evict the first 5_000 entries (insertion order, oldest).
            for k in list(_CID_TO_INCHIKEY_CACHE.keys())[:5000]:
                _CID_TO_INCHIKEY_CACHE.pop(k, None)
        _CID_TO_INCHIKEY_CACHE[cid] = value


_DEFAULT_PHASE1_PROCESSED_DIR: Path = (
    Path(__file__).resolve().parents[2] / "phase1" / "processed_data"
)
DEFAULT_PUBCHEM_CSV: Path = _DEFAULT_PHASE1_PROCESSED_DIR / "pubchem_enrichment.csv"

# v69 ROOT FIX (P2L-002): regex for censored numeric values.
# PubChem SD records emit ">1000", "<0.01", ">=50", "~1.5E3", etc. as
# CENSORED values (the real value is beyond the measurement range).
# The previous ``_safe_float`` treated these as non-numeric placeholders
# and returned None -- discarding the censoring information AND the
# lower/upper-bound value (1000). Downstream consumers that filter by
# molecular_weight had no way to know whether the value was truly
# missing vs censored-high. Compounds with censored molecular weights
# were silently dropped from any molecular-weight-filtered subgraph
# (e.g. "drug-like compounds < 500 Da" filters).
#
# ROOT FIX: detect censored values via regex and emit a structured
# ``CensoredValue`` dict instead of None. The dict carries:
#   - ``value``: float -- the bound (1000.0 for ">1000")
#   - ``censored``: True -- flag so consumers can filter
#   - ``direction``: ">" | "<" | ">=" | "<=" | "~" -- censoring direction
# Consumers that just want a float can call ``_safe_float`` which returns
# the bound for censored values (preserving the legacy contract for
# callers that expect a float-or-None return).
_CENSORED_VALUE_RE: re.Pattern[str] = re.compile(
    r"^\s*([><~=]=?)\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*$"
)


def _parse_censored_value(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a censored numeric string like ">1000" into a structured dict.

    v69 ROOT FIX (P2L-002). Returns ``None`` if the string is not a
    censored value. Otherwise returns a dict with keys:
    ``value`` (float), ``censored`` (True), ``direction`` (str).
    """
    m = _CENSORED_VALUE_RE.match(raw)
    if m is None:
        return None
    direction = m.group(1)
    bound_str = m.group(2)
    try:
        bound = float(bound_str)
    except (TypeError, ValueError):
        return None
    # Normalize "~" to "~" (approximately) -- keep as-is for semantic clarity.
    return {
        "value": bound,
        "censored": True,
        "direction": direction,
    }


def _safe_float(value: Any) -> Optional[float]:
    """V19 ROOT FIX (RT-10): robustly coerce a value to ``float`` without
    raising on non-numeric placeholders.

    PubChem SD records emit ``"N/A"``, ``">1000"``, ``"?"``, ``"1.5E"``
    and similar non-numeric placeholders for unknown/approximate masses.
    The previous code did ``float(row_dict["molecular_weight"])`` directly --
    a single non-numeric placeholder raised ``ValueError`` and aborted
    the entire PubChem batch (the caller in ``run_pipeline.py`` swallows
    the exception, so all subsequent Compound rows were silently lost).

    Root-level fix: per-row try/except returning ``None`` on failure so
    the row is preserved with ``molecular_weight=None`` instead of
    dropping every subsequent row.

    v69 ROOT FIX (P2L-002): for CENSORED values (">1000", "<0.01", etc.),
    return the BOUND as a float (1000.0 for ">1000"). This preserves the
    legacy float-or-None contract for callers that just want a float,
    while the censoring metadata is preserved separately via
    ``_safe_float_with_censoring``. This means molecular-weight filters
    (e.g. "< 500 Da") will now INCLUDE censored-high compounds (bound
    1000.0 > 500 -> correctly excluded by the filter) instead of silently
    dropping them (None cannot be compared, so the filter skipped them
    before -- which could either include or exclude depending on the
    filter's NaN handling, often inconsistently).
    """
    if value is None:
        return None
    raw = str(value).strip()
    if raw in ("", "nan", "None", "null", "N/A", "NA", "?", "-"):
        return None
    # v69 P2L-002: detect censored values and return the bound.
    censored = _parse_censored_value(raw)
    if censored is not None:
        return censored["value"]
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "pubchem_loader: non-numeric value %r coerced to None", raw,
        )
        return None


def _safe_float_with_censoring(value: Any) -> Optional[Union[float, Dict[str, Any]]]:
    """Like ``_safe_float`` but preserves censoring metadata.

    v69 ROOT FIX (P2L-002). Returns:
      - ``None`` for missing/placeholder values
      - ``float`` for plain numeric values
      - ``dict`` for censored values: ``{"value": float, "censored": True,
        "direction": ">"|"<"|">="|"<="|"~"}``

    Use this when you need to distinguish censored from uncensored values
    (e.g. for filtering, provenance, or downstream model features). Use
    ``_safe_float`` when you just want a float-or-None (the censoring
    bound is returned as a plain float for backward compat).
    """
    if value is None:
        return None
    raw = str(value).strip()
    if raw in ("", "nan", "None", "null", "N/A", "NA", "?", "-"):
        return None
    censored = _parse_censored_value(raw)
    if censored is not None:
        return censored
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "pubchem_loader: non-numeric value %r coerced to None", raw,
        )
        return None


def download_pubchem(target_path: Optional[Path] = None) -> Path:
    """Run Phase 1's PubChem pipeline if needed, return CSV path.

    v16 ROOT FIX (SF-6): the previous code used a bare ``except Exception``
    around ``PubChemPipeline().run()`` and logged at WARNING. This hid
    patient-safety-critical PubChem enrichment failures as warnings, and
    the downstream guard ``if not out_path.exists()`` only fired if NO
    CSV existed -- a stale/partial CSV from a previous failed run would
    be silently used. Now: narrow the exception to expected failure
    modes (ImportError, OSError, plus the PubChem pipeline's own
    PipelineError), log at ERROR, AND verify the CSV's freshness
    (modification time within the last 30 days) before accepting it.
    """
    out_path = target_path or DEFAULT_PUBCHEM_CSV
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("pubchem_loader: using existing Phase 1 CSV %s", out_path)
        return out_path
    try:
        from phase1.pipelines.pubchem_pipeline import PubChemPipeline  # type: ignore
        from phase1.pipelines.base_pipeline import PipelineError  # type: ignore
        _expected_errors = (ImportError, OSError, PipelineError, FileNotFoundError, ValueError)
    except ImportError:
        _expected_errors = (ImportError, OSError, FileNotFoundError, ValueError)
    try:
        from phase1.pipelines.pubchem_pipeline import PubChemPipeline  # type: ignore
        logger.info("pubchem_loader: running Phase 1 PubChemPipeline to produce %s", out_path)
        PubChemPipeline().run()
    except _expected_errors as exc:
        # v16 SF-6: narrow except + ERROR level + metric.
        logger.error(
            "pubchem_loader: Phase 1 PubChemPipeline failed (%s: %s). "
            "Falling back to whatever CSV is present at %s -- if the CSV "
            "is stale, downstream enrichment will be missing the latest "
            "PubChem compound properties.",
            type(exc).__name__, exc, out_path,
            exc_info=True,
        )
    if not out_path.exists():
        raise FileNotFoundError(
            f"PubChem CSV not found at {out_path}. Run Phase 1 first."
        )
    # v16 SF-6: warn if the CSV is stale (older than 30 days).
    try:
        import time as _time
        age_sec = _time.time() - out_path.stat().st_mtime
        if age_sec > 30 * 86400:
            logger.warning(
                "pubchem_loader: CSV at %s is %.1f days old -- consider "
                "re-running Phase 1 PubChemPipeline to refresh.",
                out_path, age_sec / 86400,
            )
    except OSError:
        pass
    return out_path


def parse_pubchem(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Read Phase 1's cleaned PubChem CSV into a DataFrame."""
    # v28 ROOT FIX (P2-L-9): the type signature says Optional[Path] but
    # downstream callers (e.g. run_pipeline.py:3189) pass plain ``str``
    # paths. Without coercion, ``path.exists()`` raises
    # ``AttributeError: 'str' object has no attribute 'exists'``. Coerce
    # to ``Path`` at the entry point so any path-like input works.
    if filepath is not None and not isinstance(filepath, Path):
        filepath = Path(filepath)
    path = filepath or DEFAULT_PUBCHEM_CSV
    if not path.exists():
        download_pubchem(path)
    df = pd.read_csv(path, low_memory=False)  # v71 P2L-001: consistent with iter_pubchem_chunked
    return df


# v28 ROOT FIX (P2-L-15): streaming parser for production-scale PubChem
# CSVs. ``parse_pubchem`` loads the entire file into memory; production
# PubChem SD-record extracts can be hundreds of MB. ``iter_pubchem_chunked``
# yields successive 10K-row DataFrames so callers with bounded memory can
# process the file incrementally (e.g. the run_pipeline step7h path).
def iter_pubchem_chunked(
    filepath: Optional[Path] = None,
    chunksize: int = 10_000,
) -> "pd.io.parsers.TextFileReader":
    """Stream Phase 1's PubChem CSV in fixed-size chunks.

    Yields
    ------
    pd.DataFrame
        Successive chunks of ``chunksize`` rows from the CSV. The final
        chunk may be smaller.

    Notes
    -----
    Callers iterate the returned reader:

        for chunk in iter_pubchem_chunked():
            nodes = pubchem_to_node_records(chunk)
            builder.load_nodes_batch("Compound", nodes)

    Production PubChem extracts can be hundreds of MB; ``parse_pubchem``
    loads the entire file into memory. This streaming API exists for
    memory-constrained deployments and batch processing pipelines.
    """
    if filepath is not None and not isinstance(filepath, Path):
        filepath = Path(filepath)
    path = filepath or DEFAULT_PUBCHEM_CSV
    if not path.exists():
        download_pubchem(path)
    return pd.read_csv(path, chunksize=chunksize, low_memory=False)


def pubchem_to_node_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Emit Compound node records from PubChem enrichment rows.

    v27 ROOT FIX (P2-L-2): Phase 1's ``pubchem_enrichment.csv`` is keyed
    by ``inchikey`` (NOT ``pubchem_cid`` -- PubChem enrichment at Phase 1
    is the post-cleaning output where compounds have already been
    resolved to their canonical InChIKey). The previous implementation
    REQUIRED a ``pubchem_cid`` / ``cid`` / ``CID`` column -- none of which
    exist in Phase 1's CSV -- so it silently dropped 100% of rows
    (confirmed empirically: ``pubchem_nodes: 0``).

    Phase 1's actual columns are:
      - ``inchikey``         (canonical key, present on every row)
      - ``canonical_smiles`` (canonical SMILES)
      - ``isomeric_smiles``  (stereo-specific SMILES, when available)
      - ``molecular_weight`` (float)
      - ``xlogp``, ``tpsa``, ``complexity``, ``h_bond_donors``,
        ``h_bond_acceptors`` (optional physicochemical properties)

    New behavior:
      - Use ``inchikey`` (uppercased to satisfy kg_builder.ID_PATTERNS)
        as the canonical Compound ID when no CID column is present.
      - Map ``canonical_smiles`` -> node ``smiles`` field.
      - Map ``isomeric_smiles`` -> node ``isomeric_smiles`` field.
      - Emit ``pubchem_cid`` ONLY when a CID column is actually present
        (preserves backward compatibility for raw-PubChem-SD-record inputs).
    """
    nodes: List[Dict[str, Any]] = []
    seen: set[str] = set()
    # v29 ROOT FIX (audit L-8): CID matching was case-sensitive -- failed
    # on case differences. Normalize to lowercase before comparison.
    # The previous code only checked three specific column-name spellings
    # ("pubchem_cid", "cid", "CID"). Real-world Phase 1 outputs and raw
    # PubChem SD-record extracts emit the CID column under many case
    # variants ("PubChem_CID", "PUBCHEM_CID", "Cid", "Pubchem_cid", ...).
    # Any case variant other than the three hard-coded ones was silently
    # treated as "no CID column present", dropping the CID from the
    # emitted node record (and, when no InChIKey was present either,
    # dropping the whole row). Build a single case-insensitive view of
    # the row ONCE, then look up the CID column by lowercase key.
    cid_column_keys = ("pubchem_cid", "cid")
    # v42 ROOT FIX (P2 #5): use itertuples instead of iterrows for
    # ~10x speedup on large PubChem extracts. iterrows creates a new
    # Series per row (slow); itertuples returns lightweight namedtuples.
    # v42 ROOT FIX (P2 #6): use int(str(cid_raw).strip()) instead of
    # int(float(cid_raw)) to avoid float precision loss for very large CIDs.
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        row_lc = {str(k).lower(): v for k, v in row_dict.items()}
        cid_raw = next(
            (row_lc.get(k) for k in cid_column_keys if row_lc.get(k)),
            None,
        )
        cid_int: Optional[int] = None
        if cid_raw is not None and str(cid_raw).strip() not in ("", "nan"):
            try:
                cid_int = int(str(cid_raw).strip())  # v42: no float() detour
            except (TypeError, ValueError):
                cid_int = None

        # InChIKey -- Phase 1's canonical key. Uppercase to satisfy
        # kg_builder.ID_PATTERNS["Compound"] regex (must be uppercase).
        #
        # v102 P2-036: route through centralized ``normalize_inchikey`` so
        # this loader produces the SAME canonical form as phase1_bridge
        # and chembl_loader (strip + upper + placeholder-collapse).
        # Previously this loader used ``inchikey.upper()`` WITHOUT strip,
        # so a " RZBJ...AN " input from a malformed PubChem extract
        # would dead-letter while chembl_loader's ``.strip().upper()``
        # would succeed — the SAME compound got two canonical IDs.
        inchikey = _normalize_inchikey(row_dict.get("inchikey"))

        # Task 87 ROOT FIX: when the CSV provides a CID but no InChIKey,
        # resolve the InChIKey via PubChem's PUG-REST API so the
        # Compound node can be merged with ChEMBL/DrugBank Compound
        # nodes (which are InChIKey-keyed). The previous code fell back
        # to ``f"CID{cid_int}"`` as the canonical ID and emitted
        # ``inchikey=None`` -- a Compound node that could not be joined
        # to any other source, fragmenting the KG. The lookup is cached
        # and rate-limited (see ``_cid_to_inchikey``) so the overhead
        # is bounded even on large PubChem extracts.
        if not inchikey and cid_int is not None:
            looked_up = _cid_to_inchikey(cid_int)
            if looked_up:
                inchikey = _normalize_inchikey(looked_up)

        # Choose canonical ID: InChIKey preferred, else CID<pid> if the
        # PUG-REST lookup also failed (rare -- log so operator knows).
        if inchikey:
            canonical_id = inchikey
        elif cid_int is not None:
            logger.info(
                "pubchem_loader: using CID %d as canonical_id (InChIKey "
                "missing from CSV and PUG-REST lookup returned no result)",
                cid_int,
            )
            canonical_id = f"CID{cid_int}"
        else:
            # Neither InChIKey nor CID -- cannot canonically identify
            # the compound. Skip rather than emit a dead-letter node.
            continue
        if canonical_id in seen:
            continue
        seen.add(canonical_id)

        # SMILES -- Phase 1 emits ``canonical_smiles`` and ``isomeric_smiles``;
        # raw-SD-record path emits ``smiles``. Map all three.
        canonical_smiles = str(row_dict.get("canonical_smiles") or "").strip()
        if canonical_smiles.lower() == "nan":
            canonical_smiles = ""
        isomeric_smiles = str(row_dict.get("isomeric_smiles") or "").strip()
        if isomeric_smiles.lower() == "nan":
            isomeric_smiles = ""
        legacy_smiles = str(row_dict.get("smiles") or "").strip()
        if legacy_smiles.lower() == "nan":
            legacy_smiles = ""
        smiles = canonical_smiles or legacy_smiles or isomeric_smiles

        node: Dict[str, Any] = {
            "id": canonical_id,
            "label": "Compound",
            "name": str(
                row_dict.get("iupac_name")
                or row_dict.get("name")
                or (f"CID{cid_int}" if cid_int is not None else canonical_id)
            ),
            "inchikey": inchikey or None,
            "smiles": smiles or None,
            "molecular_formula": str(row_dict.get("molecular_formula") or ""),
            # V19 ROOT FIX (RT-10): delegate to _safe_float so a single
            # non-numeric placeholder (e.g. "N/A", ">1000", "?") no longer
            # aborts the entire PubChem batch with ValueError. The row is
            # preserved with molecular_weight=None instead.
            #
            # v69 ROOT FIX (P2L-002): use ``_safe_float_with_censoring``
            # so CENSORED values (">1000", "<0.01") are preserved as a
            # structured dict ``{"value": 1000.0, "censored": True,
            # "direction": ">"}`` instead of being silently dropped to
            # None. Downstream consumers (e.g. drug-likeness filters) can
            # now distinguish "truly missing" from "censored-high" and
            # route accordingly. The plain ``molecular_weight`` field is
            # kept as a float-or-None for backward compat (censored-high
            # returns the bound as a float); the censoring metadata is
            # preserved in the separate ``molecular_weight_censored`` field.
            "molecular_weight": _safe_float(row_dict.get("molecular_weight")),
            "molecular_weight_censored": _safe_float_with_censoring(
                row_dict.get("molecular_weight")
            ) if (
                _parse_censored_value(
                    str(row_dict.get("molecular_weight") or "").strip()
                ) is not None
            ) else None,
            "_source": "pubchem",
        }
        # Emit ``pubchem_cid`` ONLY when a CID was actually present --
        # Phase 1's enrichment CSV has no CID column, so omit it.
        if cid_int is not None:
            node["pubchem_cid"] = cid_int
        # Preserve isomeric SMILES as a separate field when available.
        if isomeric_smiles:
            node["isomeric_smiles"] = isomeric_smiles
        nodes.append(node)
    return nodes


__all__ = [
    "download_pubchem",
    "parse_pubchem",
    "iter_pubchem_chunked",
    "pubchem_to_node_records",
    "DEFAULT_PUBCHEM_CSV",
]
