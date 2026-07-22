"""OMIM loader -- bridges to Phase 1's cleaned OMIM CSV output.

This loader consumes ``phase1/processed_data/omim_gene_disease_associations.csv``
(produced by ``phase1.pipelines.omim_pipeline.OMIMPipeline``) and emits
Phase 2 node/edge records compatible with ``kg_builder``.

Design decision (v5 audit fix):
    Phase 2's ``run_pipeline.py`` previously tried to import a non-existent
    ``omim_loader`` module, falling into an ``except ImportError`` branch
    that silently skipped OMIM ingestion. The proper fix is to bridge
    Phase 1's already-cleaned OMIM output into Phase 2's graph builder.

Public API (matches the contract expected by ``run_pipeline.py:1784-1789``):
    - ``download_omim()`` -> triggers Phase 1's pipeline if needed
    - ``parse_omim()`` -> returns a pandas DataFrame of OMIM GDA rows
    - ``omim_to_node_records(df)`` -> List[Dict] of Disease/Gene nodes
    - ``omim_to_edge_records(df)`` -> List[Dict] of (Gene, associated_with, Disease) edges
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_PHASE1_PROCESSED_DIR: Path = (
    Path(__file__).resolve().parents[2] / "phase1" / "processed_data"
)
DEFAULT_OMIM_CSV: Path = _DEFAULT_PHASE1_PROCESSED_DIR / "omim_gene_disease_associations.csv"


def _normalise_mim_id(raw_id: Any) -> str:
    """Normalise a MIM-style identifier to its canonical ``MIM:<digits>`` form.

    Task 86 ROOT FIX: OMIM disease/gene IDs may appear in either of two
    forms across Phase 1 outputs:

      * ``"100650"``       (bare numeric)
      * ``"MIM:100650"``   (already namespaced)

    Without normalisation, the same disease would be loaded as two
    DIFFERENT Disease nodes (``MIM:100650`` and ``100650``), splitting
    the KG and breaking multi-hop queries. This helper strips a
    case-insensitive ``"MIM:"`` prefix, validates the remainder is a
    6-digit integer in OMIM's valid range, and re-emits the canonical
    ``MIM:<int>`` form. Non-MIM strings are returned unchanged so
    external vocabularies (e.g. ``"DOID:1438"``, ``"ORPHA:15"``)
    pass through.
    """
    if raw_id is None:
        return ""
    s = str(raw_id).strip()
    if not s:
        return ""
    if s.upper().startswith("MIM:"):
        s = s[4:].strip()
    # If what remains is a clean integer in OMIM's range, re-emit the
    # canonical prefixed form so downstream consumers can rely on the
    # ``MIM:`` namespace. Otherwise return the original string verbatim
    # so non-MIM vocabularies survive untouched.
    try:
        n = int(float(s))
    except (TypeError, ValueError):
        return str(raw_id).strip()
    if 100000 <= n <= 999999:
        return f"MIM:{n}"
    return str(raw_id).strip()


def _safe_gene_id_from_mim(gene_mim: Any, gene_symbol: str) -> Optional[str]:
    """V19 ROOT FIX (RT-9): robustly convert an OMIM ``gene_mim`` value to a
    Gene ID string, falling back to ``SYM:<symbol>`` when the value is
    non-numeric.

    v37 ROOT FIX (Phase 2 Issue #7 -- MIM-as-GeneID shadow nodes): the
    previous code returned the bare MIM number (e.g. ``"12345"``) as the
    Gene ID. OMIM MIM numbers and NCBI Gene IDs share the same numeric
    space occasionally -- a gene with NCBI Gene ID 12345 and a different
    gene with OMIM MIM number 12345 would COLLIDE on the same
    ``:Gene {id: "12345"}`` node, producing a Frankenstein node with
    properties from both sources. Multi-hop Drug->targets->Protein->
    interacts_with->Protein->associated_with->Disease queries that
    traverse Gene edges silently returned empty results for affected genes.

    The fix: prefix MIM numbers with ``MIM:`` so they're namespace-
    disambiguated from bare NCBI Gene IDs. The ``kg_builder.ID_PATTERNS``
    regex for Gene already accepts ``SYM:<symbol>`` -- we extend it to
    also accept ``MIM:<digits>`` (see the ID_PATTERNS fix in
    ``kg_builder.py``). When the bridge's higher-priority resolvers
    (canonical_gene_id, ncbi_gene_id) hit, they emit bare numeric IDs;
    only the lowest-priority MIM fallback emits ``MIM:``-prefixed IDs.

    OMIM's ``morbidmap.txt`` emits non-numeric placeholders (``"?"``,
    ``"FGFR3"``, ``"-"``, ``"1A2B"``) for entries where the gene has no
    MIM number assigned. The previous code did ``str(int(float(gene_mim)))``
    without a try/except -- a single non-numeric placeholder raised
    ``ValueError`` and aborted the entire OMIM batch.

    Root-level fix: per-row try/except with a deterministic fallback to
    ``SYM:<gene_symbol>``. If neither MIM nor symbol is available,
    returns ``None`` so the caller can skip the row entirely.
    """
    if gene_mim is None:
        return f"SYM:{gene_symbol}" if gene_symbol else None
    raw = str(gene_mim).strip()
    if raw in ("", "nan", "None", "null", "?", "-"):
        return f"SYM:{gene_symbol}" if gene_symbol else None
    # Task 86 ROOT FIX: strip a leading "MIM:" prefix (case-insensitive)
    # before numeric parsing. The previous code called ``int(float(raw))``
    # directly, which raises ``ValueError`` when ``raw`` is e.g.
    # ``"MIM:100650"`` (a value Phase 1 already prefixed with the OMIM
    # namespace). The except branch then silently dropped the value to
    # ``SYM:<symbol>``, splitting one gene into two disjoint KG nodes:
    # ``MIM:100650`` (from unprefixed inputs) and ``SYM:FGFR3`` (from
    # prefixed inputs). The same bug applied to disease IDs that
    # appeared as ``"MIM:100650"`` in some rows and ``"100650"`` in
    # others. The fix: strip the prefix BEFORE parsing, so both forms
    # resolve to the same canonical ``MIM:<int>`` ID.
    if raw.upper().startswith("MIM:"):
        raw = raw[4:].strip()
    if raw in ("", "nan", "None", "null", "?", "-"):
        return f"SYM:{gene_symbol}" if gene_symbol else None
    try:
        # v37 ROOT FIX (Issue #7): prefix with MIM: to namespace-disambiguate
        # from bare NCBI Gene IDs.
        mim_int = int(float(raw))
        # BUG #64 ROOT FIX: validate MIM number is exactly 6 digits.
        # OMIM MIM numbers are 6-digit integers (e.g., 134934 for FGFR3).
        # Without validation, allele variants like "134934.001" are
        # truncated to 134934 by int(float()) -- but non-standard inputs
        # (5-digit or 7-digit numbers) would create malformed Gene node
        # IDs, fragmenting gene resolution. Validate strictly: the 6-digit
        # range is [100000, 999999]. Fall back to SYM: prefix for
        # non-6-digit values so malformed MIMs never become canonical IDs.
        if not (100000 <= mim_int <= 999999):
            logger.warning(
                "omim_loader: gene_mim=%r is not a 6-digit MIM number "
                "(parsed as %d); falling back to SYM:%s",
                raw, mim_int, gene_symbol,
            )
            return f"SYM:{gene_symbol}" if gene_symbol else None
        # P2-012 ROOT FIX: validate the MIM leading digit is in [1-6].
        # OMIM's leading digit has semantic meaning per OMIM's FAQs:
        #   1 = autosomal dominant (e.g. 100650 — Marfan syndrome)
        #   2 = autosomal recessive (e.g. 215400 — cystic fibrosis)
        #   3 = X-linked           (e.g. 300376 — Duchenne muscular dystrophy)
        #   4 = Y-linked           (e.g. 400005 — Y-linked deafness)
        #   5 = mitochondrial       (e.g. 516060 — MELAS)
        #   6 = autosomal (newly assigned post-1994) (e.g. 603903)
        # A leading 0 (e.g. 099999) is NOT a valid OMIM ID -- it is
        # either a string-padded 5-digit number or a malformed input.
        # A leading 7/8/9 (e.g. 700000, 800000, 900000) is in the 6-digit
        # range but has NO semantic meaning in OMIM's numbering scheme --
        # such MIMs do not exist in the OMIM database. Without this
        # check, malformed MIMs would be loaded into the KG and may
        # match DisGeNET diseases incorrectly during entity resolution.
        leading_digit = mim_int // 100000
        if leading_digit not in (1, 2, 3, 4, 5, 6):
            logger.warning(
                "omim_loader: gene_mim=%r has invalid leading digit %d "
                "(must be 1-6 per OMIM numbering: 1=autosomal dominant, "
                "2=autosomal recessive, 3=X-linked, 4=Y-linked, "
                "5=mitochondrial, 6=autosomal); falling back to SYM:%s "
                "(P2-012)",
                raw, leading_digit, gene_symbol,
            )
            return f"SYM:{gene_symbol}" if gene_symbol else None
        return f"MIM:{mim_int}"
    except (TypeError, ValueError):
        logger.warning(
            "omim_loader: non-numeric gene_mim=%r; falling back to SYM:%s",
            raw, gene_symbol,
        )
        return f"SYM:{gene_symbol}" if gene_symbol else None


# v27 ROOT FIX (P2-L-6): mirror phase1_bridge's Gene ID resolution priority.
# The bridge resolves gene IDs in this order:
#   1. ``canonical_gene_id``  (Phase 1's normalized ID, when available)
#   2. ``ncbi_gene_id``       (NCBI Gene Database numeric ID)
#   3. ``gene_mim``           (OMIM's MIM number -- last resort because
#                              MIM numbers are NOT NCBI Gene IDs, they
#                              are OMIM's own phenotype/gene numbering)
#   4. ``SYM:<gene_symbol>``  (symbolic fallback for unresolved genes)
# The previous omim_loader used ONLY ``gene_mim`` -- causing Gene ID
# fragmentation: the same gene appeared as two disjoint nodes (one keyed
# by NCBI Gene ID via the bridge, another keyed by MIM number via the
# OMIM loader). This function mirrors the bridge's priority so both
# paths emit the same Gene ID for the same gene.
def _resolve_gene_id_omim(row: pd.Series) -> Optional[str]:
    """Resolve a Gene ID from an OMIM Phase 1 row using bridge priority.

    Priority: canonical_gene_id -> ncbi_gene_id -> gene_id -> gene_mim -> SYM:<symbol>.

    v69 ROOT FIX (Phase1↔Phase2 integration): the previous priority chain
    looked for ``canonical_gene_id`` -> ``ncbi_gene_id`` -> ``gene_mim``.
    But Phase 1's actual OMIM CSV emits the NCBI Gene ID under the column
    name ``gene_id`` (NOT ``ncbi_gene_id`` -- that's the bridge's renamed
    form). So when the OMIM loader ran standalone on the Phase 1 CSV
    (without going through the bridge), BOTH ``canonical_gene_id`` and
    ``ncbi_gene_id`` were None, and the resolver fell through to
    ``gene_mim`` -- emitting ``MIM:176805`` instead of the correct NCBI
    Gene ID ``5742``. This caused Gene ID fragmentation: Gene nodes from
    the bridge had ID ``5742`` (correct) but Gene nodes from the OMIM
    loader had ID ``MIM:176805`` (wrong) -- disjoint subgraphs.

    ROOT FIX: add ``gene_id`` to the resolver chain BETWEEN
    ``ncbi_gene_id`` and ``gene_mim``. This matches Phase 1's actual CSV
    schema (verified: ``gene_symbol,gene_id,gene_mim,...``). The bridge
    still renames ``gene_id`` -> ``ncbi_gene_id`` for the PostgreSQL path,
    so this fix only affects the standalone-CSV path -- but that's the
    path the OMIM loader uses when called directly.
    """
    gene_symbol = str(row.get("gene_symbol") or "").strip()
    # 1. canonical_gene_id (Phase 1's normalized gene ID).
    cgid = row.get("canonical_gene_id")
    if cgid is not None and str(cgid).strip() not in ("", "nan", "None", "null"):
        raw = str(cgid).strip()
        # Strip any NCBIGene: prefix that may already be present.
        if raw.startswith("NCBIGene:"):
            raw = raw[len("NCBIGene:"):]
        try:
            return str(int(float(raw)))
        except (TypeError, ValueError):
            pass  # fall through to next priority
    # 2. ncbi_gene_id (bridge-renamed form for the PostgreSQL path).
    ncbi = row.get("ncbi_gene_id")
    if ncbi is not None and str(ncbi).strip() not in ("", "nan", "None", "null"):
        raw = str(ncbi).strip()
        if raw.startswith("NCBIGene:"):
            raw = raw[len("NCBIGene:"):]
        try:
            return str(int(float(raw)))
        except (TypeError, ValueError):
            pass  # fall through
    # v69 Phase1↔Phase2: 3. gene_id (Phase 1 CSV's actual column name for
    # the NCBI Entrez Gene ID). This is the fix that connects the OMIM
    # loader to Phase 1's actual output schema.
    gene_id_csv = row.get("gene_id")
    if gene_id_csv is not None and str(gene_id_csv).strip() not in ("", "nan", "None", "null"):
        raw = str(gene_id_csv).strip()
        if raw.startswith("NCBIGene:"):
            raw = raw[len("NCBIGene:"):]
        try:
            return str(int(float(raw)))
        except (TypeError, ValueError):
            pass  # fall through
    # 4. gene_mim (OMIM's MIM number -- last-resort numeric).
    gene_mim = row.get("gene_mim")
    mim_id = _safe_gene_id_from_mim(gene_mim, gene_symbol)
    if mim_id is not None:
        return mim_id
    # 5. SYM:<symbol>.
    return f"SYM:{gene_symbol}" if gene_symbol else None


# v27 ROOT FIX (P2-L-13): map OMIM ``association_type`` to distinct
# ``rel_type`` (was: collapse ALL to ``associated_with``).
_OMIM_ASSOC_TYPE_TO_REL: Dict[str, str] = {
    "causal": "associated_with",
    "susceptibility": "susceptible_to",
    "therapeutic": "treats",
    "biomarker": "biomarker_for",
    "gene_locus": "mapped_to",  # v43: gene_locus = physical mapping, not an association
}


def download_omim(target_path: Optional[Path] = None) -> Path:
    """Run Phase 1's OMIM pipeline if needed, return CSV path.

    v22 ROOT FIX (audit section 7 finding 11 -- "Silent stale-CSV fallback"):
    the previous code returned ANY non-empty CSV with only an INFO log,
    regardless of age. A years-stale CSV would be silently used in
    production. Add a freshness check: if the CSV is older than
    ``DRUGOS_OMIM_MAX_AGE_DAYS`` (default 30), warn loudly and re-run
    the pipeline (unless DRUGOS_ALLOW_STALE_CSV=1 is set).
    """
    import time as _time
    import os as _os
    out_path = target_path or DEFAULT_OMIM_CSV
    if out_path.exists() and out_path.stat().st_size > 0:
        max_age_days = float(_os.environ.get("DRUGOS_OMIM_MAX_AGE_DAYS", "30"))
        allow_stale = _os.environ.get("DRUGOS_ALLOW_STALE_CSV", "") == "1"
        try:
            age_days = (_time.time() - out_path.stat().st_mtime) / 86400.0
        except OSError:
            age_days = 0.0
        if age_days > max_age_days and not allow_stale:
            logger.warning(
                "omim_loader: Phase 1 CSV %s is %.1f days old (max=%g). "
                "Re-running OMIMPipeline to refresh. Set "
                "DRUGOS_ALLOW_STALE_CSV=1 to suppress.",
                out_path, age_days, max_age_days,
            )
            # Fall through to the pipeline invocation below.
        else:
            if age_days > max_age_days:
                logger.warning(
                    "omim_loader: using STALE Phase 1 CSV %s "
                    "(%.1f days old, max=%g) -- DRUGOS_ALLOW_STALE_CSV=1.",
                    out_path, age_days, max_age_days,
                )
            else:
                logger.info(
                    "omim_loader: using existing Phase 1 CSV %s "
                    "(age=%.1f days, max=%g)",
                    out_path, age_days, max_age_days,
                )
            return out_path
    try:
        from phase1.pipelines.omim_pipeline import OMIMPipeline  # type: ignore
        logger.info("omim_loader: running Phase 1 OMIMPipeline to produce %s", out_path)
        OMIMPipeline().run()
    except Exception as exc:
        logger.warning(
            "omim_loader: Phase 1 OMIMPipeline could not be invoked (%s). "
            "Falling back to whatever CSV is present at %s.", exc, out_path,
        )
    if not out_path.exists():
        raise FileNotFoundError(f"OMIM CSV not found at {out_path}. Run Phase 1 first.")
    return out_path


def parse_omim(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Read Phase 1's cleaned OMIM CSV into a DataFrame."""
    # v28 ROOT FIX (P2-L-9): the type signature says Optional[Path] but
    # downstream callers (e.g. run_pipeline.py:3122) pass plain ``str``
    # paths. Without coercion, ``path.exists()`` raises
    # ``AttributeError: 'str' object has no attribute 'exists'``. Coerce
    # to ``Path`` at the entry point so any path-like input works.
    if filepath is not None and not isinstance(filepath, Path):
        filepath = Path(filepath)
    path = filepath or DEFAULT_OMIM_CSV
    if not path.exists():
        download_omim(path)
    df = pd.read_csv(path)
    required = {"gene_symbol", "disease_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"OMIM CSV {path} missing required columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )
    return df


# v28 ROOT FIX (P2-L-15): streaming parser for production-scale OMIM
# CSVs. Mirrors ``disgenet_loader.iter_disgenet_chunked``. OMIM's
# morbidmap is typically small, but this API exists for symmetry and
# memory-constrained deployments.
def iter_omim_chunked(
    filepath: Optional[Path] = None,
    chunksize: int = 10_000,
) -> "pd.io.parsers.TextFileReader":
    """Stream Phase 1's OMIM CSV in fixed-size chunks.

    Yields
    ------
    pd.DataFrame
        Successive chunks of ``chunksize`` rows from the CSV.

    Notes
    -----
    Callers iterate the returned reader:

        for chunk in iter_omim_chunked():
            nodes = omim_to_node_records(chunk)
            edges = omim_to_edge_records(chunk)
            ...
    """
    if filepath is not None and not isinstance(filepath, Path):
        filepath = Path(filepath)
    path = filepath or DEFAULT_OMIM_CSV
    if not path.exists():
        download_omim(path)
    return pd.read_csv(path, chunksize=chunksize, low_memory=False)


def omim_to_node_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Emit Disease and Gene node records from OMIM GDA rows.

    v27 ROOT FIX (P2-L-6): use ``_resolve_gene_id_omim`` to resolve the
    Gene ID with the SAME priority as ``phase1_bridge`` (canonical_gene_id
    -> ncbi_gene_id -> gene_mim -> SYM:<symbol>). The previous code used
    ONLY ``gene_mim``, causing the same gene to appear as two disjoint
    nodes (one keyed by NCBI Gene ID via the bridge, another keyed by
    MIM number via the OMIM loader).
    """
    nodes: List[Dict[str, Any]] = []
    seen_disease: set[str] = set()
    seen_gene: set[str] = set()
    for _, row in df.iterrows():
        # Task 86 ROOT FIX: normalise ``MIM:`` prefix so disease IDs
        # that arrive as either ``"100650"`` or ``"MIM:100650"`` both
        # resolve to the same canonical Disease node ID.
        disease_id = _normalise_mim_id(row.get("disease_id") or "")
        if disease_id and disease_id not in seen_disease:
            seen_disease.add(disease_id)
            nodes.append({
                "id": disease_id,
                "label": "Disease",
                "name": str(row.get("disease_name") or disease_id),
                "mim_id": str(row.get("phenotype_mim") or ""),
                "_source": "omim",
            })
        gene_symbol = str(row.get("gene_symbol") or "").strip()
        # Filter OMIM's ALTGENE/MENDGENE/MYGENE placeholders (audit §C.4).
        if gene_symbol.upper() in {"ALTGENE", "MENDGENE", "MYGENE", ""}:
            continue
        # v27 ROOT FIX (P2-L-6): use bridge-compatible priority.
        gene_id = _resolve_gene_id_omim(row)
        if gene_id is None:
            continue
        if gene_id not in seen_gene:
            seen_gene.add(gene_id)
            nodes.append({
                "id": gene_id,
                "label": "Gene",
                "name": gene_symbol or gene_id,
                "mim_id": str(row.get("gene_mim") or ""),
                "uniprot_id": str(row.get("uniprot_id") or ""),
                "gene_symbol": gene_symbol,  # BUG-D-009: preserve for canonicalization
                "_source": "omim",
            })
    return nodes


def omim_to_edge_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Emit (Gene, <rel_type>, Disease) edge records.

    v27 ROOT FIXES (P2-L-3 + P2-L-6 + P2-L-13):
      - **P2-L-3 (score scale)**: OMIM ``score`` is already on a 0-1 scale
        (per Phase 1's score_method=omim_mapping_key). Emit BOTH the raw
        source-specific score (``omim_score`` -- preserved for traceability)
        AND a canonical ``normalized_score`` in [0,1] for downstream
        cross-source fusion.
      - **P2-L-6 (gene ID resolution)**: use ``_resolve_gene_id_omim``
        (bridge-compatible priority: canonical_gene_id -> ncbi_gene_id ->
        gene_mim -> SYM:<symbol>) instead of ``_safe_gene_id_from_mim``
        alone.
      - **P2-L-13 (association_type collapse)**: map ``association_type``
        to distinct ``rel_type`` (was: collapse ALL to
        ``associated_with``). The raw ``association_type`` is preserved
        in ``props``.

    v28 ROOT FIX (P2-L-16): the previous code applied NO score threshold.
    OMIM mapping_key scores (1=confirmed, 2=likely, 3=provisional) map to
    evidence_strength values; a 0.05-score mapping (provisional evidence)
    carried the SAME edge weight as a 1.0-score confirmed mapping. Now
    drop edges with ``score < config.OMIM_MIN_SCORE`` (default 0.5). The
    dropped-row count is logged at WARNING. Rows with missing/unparseable
    scores are KEPT (they may carry high-quality curated evidence whose
    score was lost during ETL).
    """
    # v28 ROOT FIX (P2-L-16): import OMIM min score threshold.
    try:
        from .config import OMIM_MIN_SCORE as _OMIM_MIN_SCORE
    except ImportError:
        _OMIM_MIN_SCORE = 0.5
    _dropped_below_threshold = 0
    _total_seen = 0

    edges: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        _total_seen += 1
        # Task 86 ROOT FIX: normalise MIM prefix so ``"100650"`` and
        # ``"MIM:100650"`` map to the same Disease node ID (matches
        # the node-builder path in ``omim_to_node_records``).
        disease_id = _normalise_mim_id(row.get("disease_id") or "")
        gene_symbol = str(row.get("gene_symbol") or "").strip()
        if gene_symbol.upper() in {"ALTGENE", "MENDGENE", "MYGENE", ""}:
            continue
        # v27 ROOT FIX (P2-L-6): bridge-compatible Gene ID resolution.
        gene_id = _resolve_gene_id_omim(row) or ""
        if not disease_id or not gene_id:
            continue
        # OMIM score: prefer Phase 1's ``evidence_strength`` (normalized),
        # fall back to ``score``, fall back to ``normalized_score``.
        # v68 ROOT FIX (P2L-005): the previous fallback chain consulted
        # ``evidence_strength`` -> ``normalized_score`` -> ``score`` -- NEVER
        # ``mapping_key``. OMIM's ``morbidmap.txt`` keys evidence by
        # ``mapping_key`` (1=confirmed, 2=likely, 3=provisional). If
        # Phase 1 emits a row with mapping_key=1 (confirmed) but no
        # evidence_strength/score/normalized_score, ``score_f`` was None,
        # the threshold check was skipped, and the row was KEPT with
        # ``normalized_score=None`` -- a confirmed OMIM association ended
        # up with NULL score in the KG, defeating the score-based ranking
        # in the GNN.
        # ROOT FIX: add ``mapping_key`` to the END of the fallback chain
        # (after evidence_strength / normalized_score / score), mapping
        # "1"->0.95 (confirmed), "2"->0.7 (likely), "3"->0.4 (provisional).
        # This ensures confirmed OMIM associations ALWAYS get a numeric
        # score even when Phase 1's evidence_strength/score fields are
        # missing. The mapping follows OMIM's official mapping_key
        # semantics (1=highest confidence, 3=lowest).
        score = row.get("evidence_strength")
        if score is None or str(score) == "nan":
            score = row.get("normalized_score")
        if score is None or str(score) == "nan":
            score = row.get("score")
        # v68 ROOT FIX (P2L-005): mapping_key fallback.
        _OMIM_MAPPING_KEY_TO_SCORE: Dict[str, float] = {
            "1": 0.95,  # confirmed -- highest confidence
            "2": 0.7,   # likely -- medium confidence
            "3": 0.4,   # provisional -- lowest confidence
        }
        if (score is None or str(score) == "nan"):
            mapping_key_raw = row.get("mapping_key")
            if mapping_key_raw is not None:
                mk_str = str(mapping_key_raw).strip()
                if mk_str in _OMIM_MAPPING_KEY_TO_SCORE:
                    score = _OMIM_MAPPING_KEY_TO_SCORE[mk_str]
                    logger.debug(
                        "omim_loader: mapping_key=%s -> score=%.2f "
                        "(v68 ROOT FIX P2L-005 fallback).",
                        mk_str, score,
                    )
        # v43 ROOT FIX (P1 -- OMIM_MIN_SCORE bypassed for evidence_strength):
        # Phase 1 emits evidence_strength as a CATEGORICAL STRING
        # ("robust"/"moderate"/"limited"/"unsupported"), not a float.
        # The previous code did float(score) which raises ValueError on
        # "robust" -> score_f=None -> threshold check skipped -> row KEPT
        # regardless of evidence quality. This loaded "unsupported"
        # evidence with the same weight as "robust", polluting the
        # embedding geometry.
        # Fix: if score is a categorical string, map it to a numeric
        # value FIRST, then apply the threshold. The mapping follows
        # the OMIM evidence_strength convention:
        #   "robust"     -> 0.9  (strongest, multiple lines of evidence)
        #   "moderate"   -> 0.7
        #   "limited"    -> 0.4
        #   "unsupported"-> 0.05 (weakest, below default threshold 0.5)
        #
        # v69 ROOT FIX (P2L-006): preserve the ORIGINAL raw score (string
        # "robust" or float 0.95) as ``omim_score_raw`` so downstream
        # consumers can distinguish curated numeric scores from
        # categorical-string-derived scores. The previous code emitted
        # ``omim_score = score_f`` (the DERIVED numeric), which erased
        # the distinction: ``omim_score=0.9`` could have come from a
        # numeric Phase 1 score OR from a categorical "robust" label.
        # ROOT FIX:
        #   - ``omim_score_raw``: the ORIGINAL value (string or float)
        #   - ``omim_score``: the derived numeric (for backward compat)
        #   - ``omim_score_normalized``: same as ``omim_score`` but
        #     explicitly named as derived (for clarity in new code)
        #   - ``score_source_type``: "numeric" | "categorical" | "mapping_key"
        #     so downstream consumers know the provenance
        _EVIDENCE_STRENGTH_MAP = {
            "robust": 0.9,
            "moderate": 0.7,
            "limited": 0.4,
            "unsupported": 0.05,
        }
        # v69 P2L-006: track the original raw value BEFORE any mapping.
        omim_score_raw: Any = score
        score_source_type: str = "numeric"  # default assumption
        if isinstance(score, str) and score.lower().strip() in _EVIDENCE_STRENGTH_MAP:
            score_f = _EVIDENCE_STRENGTH_MAP[score.lower().strip()]
            score_source_type = "categorical"
        elif (
            score is not None
            and str(score) != "nan"
            and str(score).strip() in _OMIM_MAPPING_KEY_TO_SCORE
        ):
            # v69 P2L-006: also flag mapping_key-derived scores.
            score_f = _OMIM_MAPPING_KEY_TO_SCORE[str(score).strip()]
            score_source_type = "mapping_key"
        else:
            try:
                score_f = float(score) if score is not None and str(score) != "nan" else None
            except (TypeError, ValueError):
                score_f = None
            if score_f is not None:
                score_source_type = "numeric"
            else:
                score_source_type = "unknown"
        # v28 ROOT FIX (P2-L-16): apply min-score threshold. Rows with
        # missing scores are KEPT.
        if score_f is not None and score_f < _OMIM_MIN_SCORE:
            _dropped_below_threshold += 1
            continue
        # v27 ROOT FIX (P2-L-3): OMIM scores already 0-1; passthrough.
        if score_f is not None:
            normalized_score = min(max(score_f, 0.0), 1.0)
        else:
            normalized_score = None
        # v27 ROOT FIX (P2-L-13): distinct rel_type per association_type.
        raw_assoc_type = str(row.get("association_type") or "").strip().lower()
        if raw_assoc_type == "nan":
            raw_assoc_type = ""
        rel_type = _OMIM_ASSOC_TYPE_TO_REL.get(raw_assoc_type, "associated_with")
        edges.append({
            "src_id": gene_id,
            "dst_id": disease_id,
            "src_type": "Gene",
            "dst_type": "Disease",
            "rel_type": rel_type,
            "props": {
                "score": score_f,
                # v27 ROOT FIX (P2-L-3): raw source-specific score.
                # v69 ROOT FIX (P2L-006): ``omim_score`` is kept as the
                # DERIVED numeric for backward compat. The ORIGINAL raw
                # value (string "robust" or float 0.95) is preserved in
                # ``omim_score_raw`` so downstream consumers can
                # distinguish curated numeric scores from categorical-
                # string-derived scores. ``omim_score_normalized`` is
                # the same derived numeric, explicitly named as derived.
                "omim_score": score_f,
                "omim_score_raw": omim_score_raw,
                "omim_score_normalized": normalized_score,
                # v69 P2L-006: provenance flag -- how was the score derived?
                #   "numeric"     -- score was already a float in Phase 1
                #   "categorical" -- score was "robust"/"moderate"/etc.
                #   "mapping_key" -- score was derived from OMIM mapping_key
                #   "unknown"     -- score was missing/unparseable
                "score_source_type": score_source_type,
                # Canonical normalized score in [0,1] for cross-source fusion.
                "normalized_score": normalized_score,
                # v69 P2L-046: document the aggregation method and score
                # semantic so downstream consumers can fuse scores correctly
                # across sources. OMIM scores are single-source curated
                # values (no dedup aggregation). The semantic is
                # "association_probability" -- the score reflects the
                # strength of the gene-disease association evidence.
                "score_aggregation": "single",
                "score_semantic": "association_probability",
                "source": "omim",
                "evidence": raw_assoc_type or "genetic_association",
                # v27 ROOT FIX (P2-L-13): ALWAYS preserve raw association_type.
                "association_type": raw_assoc_type or None,
                "mapping_key": str(row.get("mapping_key") or ""),
            },
            "_source": "omim",
        })
    # v28 ROOT FIX (P2-L-16): log dropped rows so operators can audit.
    if _dropped_below_threshold > 0:
        logger.warning(
            "omim_to_edge_records: dropped %d of %d rows with score < %.3f "
            "(config.OMIM_MIN_SCORE). Set DRUGOS_OMIM_MIN_SCORE=0 to "
            "disable the threshold (not recommended in production).",
            _dropped_below_threshold, _total_seen, _OMIM_MIN_SCORE,
        )
    return edges


__all__ = [
    "download_omim",
    "parse_omim",
    "iter_omim_chunked",
    "omim_to_node_records",
    "omim_to_edge_records",
    "DEFAULT_OMIM_CSV",
]
