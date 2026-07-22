"""DisGeNET loader -- bridges to Phase 1's cleaned DisGeNET CSV output.

This loader consumes ``phase1/processed_data/disgenet_gene_disease_associations.csv``
(produced by ``phase1.pipelines.disgenet_pipeline.DisgenetPipeline``) and
emits Phase 2 node/edge records compatible with ``kg_builder``.

v21 ROOT FIX (Audit section 5 finding / bypass matrix - "DEFAULT filename
is wrong: gene_disease_associations.csv vs Phase 1's actual
disgenet_gene_disease_associations.csv"): the previous default filename
was ``gene_disease_associations.csv`` (without the ``disgenet_`` prefix).
Phase 1's actual output is ``disgenet_gene_disease_associations.csv``.
This caused FileNotFoundError on standalone use and was unreachable from
step7 due to the NameError on phase1_processed_dir (now also fixed).
Fix: use the correct prefixed filename as the default; the parser still
accepts an explicit filepath override for backward compat.

v70 ROOT FIX (P2L-047): OpenTargets DOID uses underscore ("DOID_1438")
but DisGeNET DOID uses colon ("DOID:1438") -- no join was possible
between OpenTargets-emitted and DisGeNET-emitted Disease nodes for the
same disease. Phase 1's disgenet_pipeline ALREADY normalizes to colon
form, so the disgenet_loader receives canonical colon-form IDs. As a
DEFENSE-IN-DEPTH measure (in case a future Phase 1 regression or a
custom-pinned stale CSV re-introduces underscore-form IDs), we now
apply the SAME ``_normalise_ontology_id`` normalization that
opentargets_loader uses, ensuring EVERY disease_id emitted by this
loader is in canonical colon form. This guarantees cross-source MERGE
works regardless of which form the source file used.

Public API (matches the contract expected by ``run_pipeline.py:1746-1751``):
    - ``download_disgenet()`` -> triggers Phase 1's pipeline if needed
    - ``parse_disgenet()`` -> returns a pandas DataFrame of GDA rows
    - ``disgenet_to_node_records(df)`` -> List[Dict] of Disease/Gene nodes
    - ``disgenet_to_edge_records(df)`` -> List[Dict] of (Gene, associated_with, Disease) edges
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# v42 ROOT FIX (P2 #9): module-level constant (was rebuilt inside function).
_DISGENET_ASSOC_TYPE_TO_REL: Dict[str, str] = {
    "causal": "associated_with",
    "susceptibility": "susceptible_to",
    "therapeutic": "treats",
    "biomarker": "biomarker_for",
}

# v70 ROOT FIX (P2L-047): canonical OBO Foundry case for each ontology
# prefix that DisGeNET (or any other source) might emit in underscore
# or colon form. Used by ``_normalise_disease_id_to_colon`` below to
# ensure cross-source MERGE works between OpenTargets-emitted and
# DisGeNET-emitted Disease nodes. The mapping mirrors the one in
# opentargets_loader._normalise_ontology_id -- kept here as a defensive
# copy so the two loaders do not hard-depend on each other for this
# critical normalization.
_CANONICAL_ONTOLOGY_CASE: Dict[str, str] = {
    "orphanet": "Orphanet",
    "mondo":    "MONDO",
    "efo":      "EFO",
    "doid":     "DOID",
    "hp":       "HP",
    "mp":       "MP",
    "snomedct": "SNOMEDCT",
    "otar":     "OTAR",
    # DisGeNET also emits these (not OpenTargets):
    "mesh":     "MESH",
    "orpha":    "ORPHA",   # ORPHA is an alias for Orphanet used by some sources
    "icd10":    "ICD10",
    "omim":     "OMIM",
}


def _normalise_disease_id_to_colon(disease_id: Any) -> str:
    """Normalise a disease ID to canonical OBO Foundry colon form (v70 P2L-047).

    Accepts BOTH underscore ("DOID_1438") and colon ("DOID:1438") forms
    for any known ontology prefix and returns the canonical colon form
    with the canonical case (e.g. "Orphanet:558", "DOID:1438",
    "MONDO:0004975"). IDs that do not match any known ontology prefix
    (e.g. bare UMLS CUIs like "C0006142") are returned unchanged.

    This is the DisGeNET-side mirror of
    ``opentargets_loader._normalise_ontology_id``. Both loaders must
    apply the SAME normalization so cross-source Disease-node MERGE
    works correctly.

    Args:
        disease_id: any input (str, None, NaN, etc.). Non-string inputs
            are stringified first; None/NaN/empty return "".

    Returns:
        The canonical colon-form disease ID string. Empty string if the
        input was None/NaN/empty.
    """
    if disease_id is None:
        return ""
    # Handle pandas NaN (float nan) without raising.
    if isinstance(disease_id, float):
        try:
            if pd.isna(disease_id):
                return ""
        except (TypeError, ValueError):
            pass
    s = str(disease_id).strip()
    if not s:
        return ""
    m = re.match(r"^([A-Za-z]+)[_:](\w+)$", s)
    if m:
        prefix_lower = m.group(1).lower()
        if prefix_lower in _CANONICAL_ONTOLOGY_CASE:
            canonical_prefix = _CANONICAL_ONTOLOGY_CASE[prefix_lower]
            return f"{canonical_prefix}:{m.group(2)}"
    return s

# Phase 1 emits this CSV; resolve relative to the unified package layout.
_DEFAULT_PHASE1_PROCESSED_DIR: Path = (
    Path(__file__).resolve().parents[2] / "phase1" / "processed_data"
)
# v21: use the CORRECT prefixed filename that Phase 1 actually emits.
DEFAULT_DISGENET_CSV: Path = _DEFAULT_PHASE1_PROCESSED_DIR / "disgenet_gene_disease_associations.csv"
# Backward-compat alias for callers that still pass the old name.
_LEGACY_DISGENET_CSV: Path = _DEFAULT_PHASE1_PROCESSED_DIR / "gene_disease_associations.csv"


def _resolve_disgenet_csv(target_path: Optional[Path] = None) -> Path:
    """Resolve the DisGeNET CSV path, checking both v21 and legacy names."""
    if target_path is not None:
        return target_path
    if DEFAULT_DISGENET_CSV.exists():
        return DEFAULT_DISGENET_CSV
    if _LEGACY_DISGENET_CSV.exists():
        logger.warning(
            "disgenet_loader: using legacy filename %s. Phase 1's "
            "canonical output is %s - rename the file to silence "
            "this warning.",
            _LEGACY_DISGENET_CSV, DEFAULT_DISGENET_CSV,
        )
        return _LEGACY_DISGENET_CSV
    # Default: return the canonical name even if it doesn't exist yet
    # (download_disgenet will produce it).
    return DEFAULT_DISGENET_CSV


def download_disgenet(target_path: Optional[Path] = None) -> Path:
    """Run Phase 1's DisGeNET pipeline if needed, return CSV path.

    If Phase 1's cleaned CSV already exists AND is fresh, this is a no-op.
    Otherwise it invokes ``phase1.pipelines.disgenet_pipeline.DisgenetPipeline().run()``
    to download + clean + load.

    v22 ROOT FIX (audit section 7 finding 11 -- "Silent stale-CSV fallback"):
    the previous code returned ANY non-empty CSV with only an INFO log,
    regardless of age. A years-stale CSV would be silently used in
    production. Add a freshness check: if the CSV is older than
    ``DRUGOS_DISGENET_MAX_AGE_DAYS`` (default 30), warn loudly and
    re-run the pipeline (unless DRUGOS_ALLOW_STALE_CSV=1 is set).
    """
    import time as _time
    import os as _os
    out_path = _resolve_disgenet_csv(target_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        # v22: freshness check.
        max_age_days = float(_os.environ.get("DRUGOS_DISGENET_MAX_AGE_DAYS", "30"))
        allow_stale = _os.environ.get("DRUGOS_ALLOW_STALE_CSV", "") == "1"
        try:
            age_days = (_time.time() - out_path.stat().st_mtime) / 86400.0
        except OSError:
            age_days = 0.0
        if age_days > max_age_days and not allow_stale:
            logger.warning(
                "disgenet_loader: Phase 1 CSV %s is %.1f days old "
                "(max=%g). Re-running DisgenetPipeline to refresh. "
                "Set DRUGOS_ALLOW_STALE_CSV=1 to suppress.",
                out_path, age_days, max_age_days,
            )
            # Fall through to the pipeline invocation below.
        else:
            if age_days > max_age_days:
                logger.warning(
                    "disgenet_loader: using STALE Phase 1 CSV %s "
                    "(%.1f days old, max=%g) -- DRUGOS_ALLOW_STALE_CSV=1.",
                    out_path, age_days, max_age_days,
                )
            else:
                logger.info(
                    "disgenet_loader: using existing Phase 1 CSV %s "
                    "(age=%.1f days, max=%g)",
                    out_path, age_days, max_age_days,
                )
            return out_path
    try:
        # Import lazily so the Phase 2 package doesn't hard-depend on Phase 1
        # v77 ROOT FIX (scientific bug): the import was `from phase1.pipelines.disgenet_pipeline import DisgenetPipeline`
        # (lowercase 'g') but the actual class name is `DisGeNETPipeline` (capital G, NET).
        # This ImportError was ALWAYS raised, so the DisgenetPipeline NEVER actually
        # ran from the loader -- it always fell back to whatever stale CSV was present.
        # This is a silent scientific data-loss bug: the DisGeNET freshness policy
        # was a no-op because the pipeline that refreshes the CSV could never be
        # invoked. Fix: use the correct class name.
        from phase1.pipelines.disgenet_pipeline import DisGeNETPipeline  # type: ignore
        logger.info("disgenet_loader: running Phase 1 DisGeNETPipeline to produce %s", out_path)
        DisGeNETPipeline().run()
    except ImportError as exc:
        # v71 ROOT FIX (P2L-004): narrow the except to ONLY the expected
        # failure modes (ImportError, OSError, FileNotFoundError) and log
        # at ERROR (not WARNING). The previous bare ``except Exception``
        # caught programmer errors (AttributeError, TypeError, NameError)
        # in DisgenetPipeline and silently fell back to a potentially
        # stale/partial CSV -- masking Phase 1 regressions. Unexpected
        # exceptions are now RE-RAISED so operators see real bugs.
        logger.error(
            "disgenet_loader: Phase 1 DisgenetPipeline import failed (%s: %s). "
            "Falling back to whatever CSV is present at %s. This is an "
            "expected failure mode (Phase 1 not installed).",
            type(exc).__name__, exc, out_path,
        )
    except (OSError, FileNotFoundError) as exc:
        # v71 P2L-004: file-system errors (disk full, permission denied,
        # file not found) are expected failure modes -- fall back to CSV.
        logger.error(
            "disgenet_loader: Phase 1 DisgenetPipeline failed with file "
            "system error (%s: %s). Falling back to whatever CSV is "
            "present at %s.",
            type(exc).__name__, exc, out_path,
        )
    # v71 P2L-004: any OTHER exception (AttributeError, TypeError,
    # NameError, ValueError, RuntimeError, etc.) is a PROGRAMMER ERROR
    # in DisgenetPipeline -- re-raise so the bug is visible instead of
    # masked by a silent stale-CSV fallback.
    # v68 ROOT FIX (P2L-003): when the user passes a custom ``target_path``
    # AND the cached file was stale, the code above fell through to
    # ``DisgenetPipeline().run()``. But the Phase 1 pipeline writes to ITS
    # OWN default path (``DEFAULT_DISGENET_CSV``), NOT to ``target_path``
    # (or to the legacy ``_LEGACY_DISGENET_CSV`` path). So ``out_path``
    # was NEVER refreshed. The subsequent check ``if not out_path.exists()``
    # would pass (stale file exists) and return the STALE ``out_path``,
    # contradicting the freshness policy.
    # ROOT FIX: after the pipeline runs, if the canonical default CSV
    # exists and differs from ``out_path``, copy it to ``out_path`` so
    # the caller's pinned path (custom target_path OR legacy filename)
    # is actually refreshed. This ensures the freshness policy is
    # honored regardless of which path the caller pinned.
    if (
        out_path != DEFAULT_DISGENET_CSV
        and DEFAULT_DISGENET_CSV.exists()
        and DEFAULT_DISGENET_CSV.stat().st_size > 0
    ):
        import shutil
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(DEFAULT_DISGENET_CSV), str(out_path))
            logger.info(
                "disgenet_loader: copied refreshed canonical CSV %s to "
                "pinned path %s (v68 ROOT FIX P2L-003).",
                DEFAULT_DISGENET_CSV, out_path,
            )
        except OSError as copy_exc:
            logger.warning(
                "disgenet_loader: could not copy refreshed canonical CSV "
                "%s to pinned path %s (%s). The canonical CSV is fresh; "
                "the pinned path may be stale.",
                DEFAULT_DISGENET_CSV, out_path, copy_exc,
            )
    if not out_path.exists():
        raise FileNotFoundError(
            f"DisGeNET CSV not found at {out_path}. Run Phase 1 first."
        )
    return out_path


def parse_disgenet(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Read Phase 1's cleaned DisGeNET CSV into a DataFrame."""
    # v28 ROOT FIX (P2-L-9): the type signature says Optional[Path] but
    # downstream callers (e.g. run_pipeline.py:3047) pass plain ``str``
    # paths. Without coercion, ``path.exists()`` raises
    # ``AttributeError: 'str' object has no attribute 'exists'``. Coerce
    # to ``Path`` at the entry point so any path-like input works.
    if filepath is not None and not isinstance(filepath, Path):
        filepath = Path(filepath)
    path = _resolve_disgenet_csv(filepath)
    if not path.exists():
        download_disgenet(path)
    df = pd.read_csv(path)
    # Normalize column names to the contract the rest of Phase 2 expects.
    # Phase 1 emits: gene_symbol, disease_id, disease_name, source, score, ...
    required = {"gene_symbol", "disease_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DisGeNET CSV {path} missing required columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )
    return df


# v28 ROOT FIX (P2-L-15): streaming parser for production-scale DisGeNET
# CSVs. ``parse_disgenet`` loads the entire file into memory; production
# DisGeNET extracts can exceed 1M rows. ``iter_disgenet_chunked`` yields
# successive 10K-row DataFrames so callers with bounded memory can
# process the file incrementally.
def iter_disgenet_chunked(
    filepath: Optional[Path] = None,
    chunksize: int = 10_000,
) -> "pd.io.parsers.TextFileReader":
    """Stream Phase 1's DisGeNET CSV in fixed-size chunks.

    Yields
    ------
    pd.DataFrame
        Successive chunks of ``chunksize`` rows from the CSV. The final
        chunk may be smaller.

    Notes
    -----
    The first chunk's columns define the schema; subsequent chunks share
    the same dtype mapping. Callers that need to consume the entire file
    should iterate the returned reader:

        for chunk in iter_disgenet_chunked():
            nodes = disgenet_to_node_records(chunk)
            edges = disgenet_to_edge_records(chunk)
            ...

    Phase 1's DisGeNET CSV is typically small (<50 MB) but this API
    exists for symmetry with pubchem/omim loaders and to support
    memory-constrained deployments.
    """
    if filepath is not None and not isinstance(filepath, Path):
        filepath = Path(filepath)
    path = _resolve_disgenet_csv(filepath)
    if not path.exists():
        download_disgenet(path)
    return pd.read_csv(path, chunksize=chunksize, low_memory=False)


def disgenet_to_node_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Emit Disease and Gene node records from DisGeNET GDA rows."""
    nodes: List[Dict[str, Any]] = []
    seen_disease: set[str] = set()
    seen_gene: set[str] = set()
    for _, row in df.iterrows():
        disease_id = str(row.get("disease_id") or "").strip()
        # v70 ROOT FIX (P2L-047): normalize to canonical colon form so
        # DisGeNET-emitted Disease nodes MERGE with OpenTargets-emitted
        # Disease nodes for the same disease. Phase 1 should already
        # produce colon-form IDs, but this defensive normalization
        # catches any regression or stale-CSV edge case.
        if disease_id:
            disease_id = _normalise_disease_id_to_colon(disease_id)
        if disease_id and disease_id not in seen_disease:
            seen_disease.add(disease_id)
            nodes.append({
                "id": disease_id,
                "label": "Disease",
                "name": str(row.get("disease_name") or disease_id),
                "_source": "disgenet",
            })
        gene_symbol = str(row.get("gene_symbol") or "").strip()
        # Task 85 ROOT FIX: use ``gene_symbol`` as the PRIMARY KEY for
        # Gene nodes, NOT the bare NCBI gene ID. The previous code
        # preferred ``ncbi_gene_id`` (e.g. ``"2645"``) and only fell
        # back to ``SYM:<symbol>`` when NCBI was absent. But
        # ``id_crosswalk.py`` resolves gene->UniProt by indexing the
        # crosswalk on ``entry.gene_symbol.upper()`` (see
        # ``id_crosswalk.py`` line 549 / 1036-1037). A Gene node keyed
        # by ``"2645"`` cannot be found by the crosswalk -- the
        # crosswalk has no ``"2645"`` entry -- so the gene never gets
        # linked to its UniProt protein, breaking the Phase 1->Phase 2
        # graph connectivity. The fix: key Gene nodes on the gene
        # symbol (upper-cased to match the crosswalk's lookup key) and
        # preserve NCBI gene ID as a property for back-validation.
        # Empty/placeholder symbols are skipped so we never emit a
        # Gene node with an empty primary key.
        if (
            not gene_symbol
            or gene_symbol.upper() in {"NAN", "NONE", "NULL", ""}
        ):
            continue
        ncbi_gene_id_raw = (
            row.get("ncbi_gene_id")
            if row.get("ncbi_gene_id") is not None
            else row.get("gene_id")
        )
        ncbi_gene_id_norm = ""
        if ncbi_gene_id_raw is not None and str(ncbi_gene_id_raw).strip() not in ("", "nan"):
            raw = str(ncbi_gene_id_raw).strip()
            if raw.startswith("NCBIGene:"):
                raw = raw[len("NCBIGene:"):]
            try:
                ncbi_gene_id_norm = str(int(float(raw)))
            except (TypeError, ValueError):
                ncbi_gene_id_norm = ""
        # Canonical primary key: upper-cased gene symbol so it matches
        # the crosswalk's ``entry.gene_symbol.upper()`` lookup key.
        gene_id = gene_symbol.upper()
        if gene_id not in seen_gene:
            seen_gene.add(gene_id)
            node: Dict[str, Any] = {
                "id": gene_id,
                "label": "Gene",
                "name": gene_symbol,
                "gene_symbol": gene_symbol,  # preserved for downstream join
                "_source": "disgenet",
            }
            if ncbi_gene_id_norm:
                # Preserve NCBI gene ID as a property so back-validation
                # against the NCBI Gene database is still possible.
                node["ncbi_gene_id"] = ncbi_gene_id_norm
            nodes.append(node)
    return nodes


def disgenet_to_edge_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Emit (Gene, associated_with, Disease) edge records.

    v27 ROOT FIXES (P2-L-3 + P2-L-13):
      - **P2-L-3 (score scale)**: DisGeNET ``score`` / ``gda_score`` are
        already on a 0-1 scale (per DisGeNET docs at
        https://www.disgenet.org/documentation). Emit BOTH the raw
        source-specific score (``disgenet_score`` -- preserved for
        traceability) AND a canonical ``normalized_score`` in [0,1] for
        downstream model training / cross-source fusion.
      - **P2-L-13 (association_type collapse)**: the previous code
        collapsed ALL ``association_type`` values to
        ``rel_type="associated_with"``. Per the audit, distinct
        biological associations should remain distinct. New mapping:
          causal       -> ``associated_with``
          susceptibility -> ``susceptible_to``
          therapeutic  -> ``treats``
          biomarker    -> ``biomarker_for``
        The raw ``association_type`` is always preserved in ``props``.

    v28 ROOT FIX (P2-L-16): the previous code applied NO score threshold.
    DisGeNET GDA scores span 0-1; a 0.01-score association (text-mined
    noise from a single PubMed abstract) carried the SAME edge weight as
    a 0.95-score association (validated causal variant). Now drop edges
    with ``score < config.DISGENET_MIN_SCORE`` (default 0.3). The
    dropped-row count is logged at WARNING so operators can audit the
    loss. Rows with missing/unparseable scores are KEPT (they may carry
    high-quality curated evidence whose score was lost during ETL).
    """
    # v27 ROOT FIX (P2-L-13): distinct rel_type per association_type.
    # v42 ROOT FIX (P2 #9): moved to module-level constant (was rebuilt
    # inside the function on every call -- trivially cheap but inconsistent
    # with omim_loader._OMIM_ASSOC_TYPE_TO_REL which is module-level).
    # Now uses the module-level _DISGENET_ASSOC_TYPE_TO_REL defined above.

    # v28 ROOT FIX (P2-L-16): import DisGeNET min score threshold.
    try:
        from .config import DISGENET_MIN_SCORE as _DISGENET_MIN_SCORE
    except ImportError:
        _DISGENET_MIN_SCORE = 0.3
    _dropped_below_threshold = 0
    _total_seen = 0

    edges: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        _total_seen += 1
        disease_id = str(row.get("disease_id") or "").strip()
        # v70 ROOT FIX (P2L-047): normalize to canonical colon form so
        # DisGeNET-emitted Disease edges MERGE with OpenTargets-emitted
        # Disease edges for the same disease. Phase 1 should already
        # produce colon-form IDs, but this defensive normalization
        # catches any regression or stale-CSV edge case. Applied to
        # BOTH the edge emitter and the node emitter (above) so the
        # disease_dst_id on the edge matches the id on the Disease node.
        if disease_id:
            disease_id = _normalise_disease_id_to_colon(disease_id)
        gene_symbol = str(row.get("gene_symbol") or "").strip()
        # Task 85 ROOT FIX: mirror the node-emitter logic -- the Gene
        # endpoint of every disgenet_associated_with edge MUST be keyed
        # by the upper-cased gene symbol so the edge endpoint matches
        # the Gene node's primary key (see ``disgenet_to_node_records``)
        # AND so ``id_crosswalk.py`` can resolve the gene to UniProt.
        # The previous code emitted ``"2645"`` (bare NCBI gene ID) as
        # the source endpoint, which (a) did not match any Gene node
        # id (which used ``SYM:<symbol>`` as fallback) and (b) was
        # unresolvable by the crosswalk. Result: every DisGeNET edge
        # dangled off a non-existent Gene node, fragmenting the KG.
        if (
            not gene_symbol
            or gene_symbol.upper() in {"NAN", "NONE", "NULL", ""}
        ):
            continue
        gene_id = gene_symbol.upper()
        if not disease_id or not gene_id:
            continue
        # DisGeNET score: prefer ``gda_score`` (Phase 1's normalized name),
        # fall back to ``score``.
        score = row.get("gda_score")
        if score is None or str(score) == "nan":
            score = row.get("score")
        try:
            score_f = float(score) if score is not None and str(score) != "nan" else None
        except (TypeError, ValueError):
            score_f = None
        # v28 ROOT FIX (P2-L-16): apply min-score threshold. Rows with
        # missing scores are KEPT (curated evidence whose score was lost
        # during ETL may still be high-quality).
        if score_f is not None and score_f < _DISGENET_MIN_SCORE:
            _dropped_below_threshold += 1
            continue
        # v27 ROOT FIX (P2-L-3): DisGeNET scores already on 0-1 scale;
        # passthrough to ``normalized_score`` for cross-source fusion.
        if score_f is not None:
            normalized_score = min(max(score_f, 0.0), 1.0)
        else:
            normalized_score = None
        # v27 ROOT FIX (P2-L-13): map association_type to distinct rel_type.
        raw_assoc_type = str(row.get("association_type") or "").strip().lower()
        if raw_assoc_type == "nan":
            raw_assoc_type = ""
        rel_type = _DISGENET_ASSOC_TYPE_TO_REL.get(raw_assoc_type, "associated_with")
        edges.append({
            "src_id": gene_id,
            "dst_id": disease_id,
            "src_type": "Gene",
            "dst_type": "Disease",
            "rel_type": rel_type,
            "props": {
                "score": score_f,
                # v27 ROOT FIX (P2-L-3): raw source-specific score, preserved
                # under a descriptive name for traceability / debugging.
                "disgenet_score": score_f,
                # Canonical normalized score in [0,1] for cross-source fusion.
                "normalized_score": normalized_score,
                # v69 ROOT FIX (P2L-046): document the aggregation method and
                # score semantic so downstream consumers can fuse scores
                # correctly across sources. DisGeNET's score is an
                # ASSOCIATION PROBABILITY (0-1, integrated across evidence
                # sources server-side using a sum-then-normalize method).
                # DisGeNET pre-aggregates across sources server-side, so
                # each row is a SINGLE source-specific score (no further
                # dedup aggregation is needed). This is INCOMPATIBLE with
                # OpenTargets' MAX-pooling -- downstream fusion that
                # averages ``normalized_score`` across OpenTargets and
                # DisGeNET edges mixes MAX-pooled scores with sum-normalized
                # scores, producing biased rankings.
                # ``score_aggregation="sum_normalized"`` documents the method;
                # ``score_semantic="association_probability"`` documents what
                # the score MEANS. Downstream consumers MUST weight by
                # ``score_aggregation`` when fusing across sources.
                "score_aggregation": "sum_normalized",
                "score_semantic": "association_probability",
                "source": str(row.get("source") or "disgenet"),
                "evidence": "gene_disease_association",
                # v27 ROOT FIX (P2-L-13): ALWAYS preserve raw association_type.
                "association_type": raw_assoc_type or None,
            },
            "_source": "disgenet",
        })
    # v28 ROOT FIX (P2-L-16): log dropped rows so operators can audit.
    if _dropped_below_threshold > 0:
        logger.warning(
            "disgenet_to_edge_records: dropped %d of %d rows with score < %.3f "
            "(config.DISGENET_MIN_SCORE). Set DRUGOS_DISGENET_MIN_SCORE=0 to "
            "disable the threshold (not recommended in production).",
            _dropped_below_threshold, _total_seen, _DISGENET_MIN_SCORE,
        )
    return edges


__all__ = [
    "download_disgenet",
    "parse_disgenet",
    "iter_disgenet_chunked",
    "disgenet_to_node_records",
    "disgenet_to_edge_records",
    "DEFAULT_DISGENET_CSV",
    # v70 P2L-047: export the normalizer so other loaders / tests can
    # reuse the same canonical-form logic (mirrors the public surface
    # of opentargets_loader._normalise_ontology_id).
    "_normalise_disease_id_to_colon",
]
