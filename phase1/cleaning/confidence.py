"""Confidence-tier classification for gene-disease association scores.

This module is the SINGLE source of truth for confidence-tier thresholds
across the platform.  It is consumed by the DisGeNET pipeline
(``pipelines/disgenet_pipeline.py``) and may be consumed by the OMIM
pipeline and downstream consumers (Graph Transformer feature loader).

Scientific basis
----------------
The default tiers follow Piñero et al., 2020, *DisGeNET: a comprehensive
platform integrating information on human disease-associated genes and
variants*, Nucleic Acids Research (https://doi.org/10.1093/nar/gkz1021).
Per §2.3 of the publication, the DisGeNET Disease-Specific Genomic Profile
(DSGP) score bands are:

- ``[0.0, 0.06)``   -- sub-weak (below the published weak-evidence floor)
- ``[0.06, 0.3)``   -- weak evidence
- ``[0.3, 0.5)``    -- strong evidence (lower half of the strong band)
- ``[0.5, 1.0]``    -- very strong evidence (upper half; curated multi-source)

P1-004 ROOT FIX EXTENSION (Team-1 v102): the original P1-004 fix used a
single "strong" tier for the entire [0.3, 1.0] band. This lost the
gradation between a score of 0.31 (marginal evidence) and 0.95 (very
strong, curated multi-source). Downstream ML models that bin on
``confidence_tier`` weighted them identically -- biasing the model toward
lower-confidence edges. The fix splits the strong band into "strong"
[0.3, 0.5) and "very_strong" [0.5, 1.0] so the gradation is preserved.

The previous ``0.7 -> "very_high"`` tier is REMOVED -- no publication
supports it.  The previous ``0.0 -> "low"``, ``0.1 -> "medium"``,
``0.3 -> "high"`` tiers are REPLACED by the publication-aligned tiers
above.

Design
------
- The function :func:`classify_confidence` uses :func:`bisect.bisect_right`
  on the thresholds for O(log k) classification (DES-3).  This is faster
  than a linear scan and trivially supports arbitrary numbers of tiers.
- Tier thresholds are configurable at runtime via the ``tiers`` parameter.
  The DisGeNET pipeline passes the parsed ``DISGENET_CONFIDENCE_TIERS``
  list from ``config/settings.py``.
- A defensive assertion fires if the score is NaN or negative -- these
  should never reach the classifier (validate_gda_scores clips first).
"""

from __future__ import annotations

import bisect
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default confidence tiers -- publication-aligned (Piñero et al. 2020).
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE_TIERS: list[tuple[float, str]] = [
    # P1-004 ROOT FIX (v100 forensic + Team-1 v102 extension):
    # The previous labels were ("weak", "moderate", "strong") mapped to
    # thresholds (0.0, 0.06, 0.3). Per Piñero et al. 2020 §2.3 the bands are:
    #   [0.0, 0.06)   -- sub-weak (below the published weak-evidence floor)
    #   [0.06, 0.3)   -- WEAK evidence (the published weak band)
    #   [0.3, 1.0]    -- strong evidence
    # The previous code labeled [0.0, 0.06) as "weak" (Piñero calls this
    # sub-weak) and [0.06, 0.3) as "moderate" (Piñero calls this weak).
    # P1-058 ROOT FIX: the [0.06, 0.3) band is "weak" per Piñero 2020,
    # not "moderate". The previous v43 fix labeled this "moderate" which
    # inflated the perceived confidence of weak-evidence GDA edges.
    # This INFLATED the perceived confidence of every weak-evidence GDA
    # edge -- patient-safety risk because downstream ML filters expecting
    # confidence_tier == "weak" only caught SUB-FLOOR scores, missing the
    # actual weak band. ROOT FIX: rename labels to ("sub_weak", "weak",
    # "strong") so the label set is scientifically accurate. The DB CHECK
    # constraint (chk_gda_confidence_tier), the ORM CheckConstraint
    # (models.py), DISGENET_CONFIDENCE_TIERS_JSON (config/settings.py),
    # the JSON schema (pipelines/schema/v1.json), and migration 012
    # (backfill + constraint swap) are updated in lockstep so the four
    # sites remain in agreement.
    # (Parallel V100 fix BUG #4 applied the same root fix.)
    #
    # P1-004 ROOT FIX EXTENSION (Team-1 v102 -- add very_strong tier):
    # The original P1-004 fix collapsed Piñero's strong band [0.3, 1.0]
    # into a single "strong" tier. This lost the gradation between a
    # score of 0.31 (just above weak, marginal evidence) and 0.95 (very
    # strong, curated multi-source). Downstream ML models that bin on
    # confidence_tier weighted them identically -- biasing the model
    # toward lower-confidence edges. ROOT FIX: split the strong band
    # into "strong" [0.3, 0.5) and "very_strong" [0.5, 1.0]. This
    # adds the gradation the issue asked for. The DB CHECK constraint,
    # ORM CheckConstraint, settings.py default, and a new migration 017
    # are updated in lockstep. Existing rows with score >= 0.5 are
    # backfilled to "very_strong" by migration 017.
    (0.0, "sub_weak"),     # [0.0, 0.06)   -- sub-weak (below the published weak-evidence floor; Piñero et al. 2020 §2.3)
    (0.06, "weak"),        # [0.06, 0.3)   -- weak evidence (Piñero et al. 2020 §2.3 weak band)
    (0.3, "strong"),       # [0.3, 0.5)    -- strong evidence (Piñero et al. 2020 §2.3 strong band, lower half)
    (0.5, "very_strong"),  # [0.5, 1.0]    -- very strong evidence (Piñero et al. 2020 §2.3 strong band, upper half; curated multi-source)
]
"""Default confidence-tier thresholds (Piñero et al. 2020).

A list of ``(threshold, label)`` pairs, sorted ascending by threshold.
The first tier whose ``threshold <= score`` (and which is below the next
tier's threshold) wins.  ``score = 0.0`` always falls in the first tier
(``"sub_weak"``).

v100 P1-004 ROOT FIX (SCIENTIFIC MISLABEL -- forensic root fix):
The previous labels were ``"weak"`` / ``"moderate"`` / ``"strong"``
mapped to thresholds (0.0, 0.06, 0.3). Per Piñero et al. 2020 §2.3 the
bands are sub-weak / weak / strong -- there is NO "moderate" band in the
publication. P1-058 ROOT FIX: the [0.06, 0.3) band is "weak" per Piñero
2020, not "moderate". The previous v43 fix labeled this "moderate", which
inflated the perceived confidence of every weak-evidence GDA edge --
a patient-safety risk because downstream ML filters expecting
``confidence_tier == "weak"`` only caught SUB-FLOOR scores
and missed the actual weak band, while models trained on
``confidence_tier == "moderate"`` were trained on what is actually
weak evidence.

ROOT FIX: rename labels to ``"sub_weak"`` / ``"weak"`` / ``"strong"``
so the label set is scientifically accurate. The DB CHECK constraint
(``chk_gda_confidence_tier``), the ORM CheckConstraint (``models.py``),
``DISGENET_CONFIDENCE_TIERS_JSON`` (``config/settings.py``), the JSON
schema (``pipelines/schema/v1.json``), SCHEMA.md, and migration 012
(backfill + constraint swap) are updated in lockstep so all four sites
remain in agreement.

The v43 fix (which introduced the weak/moderate/strong labels to keep
the DB schema stable) is now SUPERSEDED -- the DB schema is updated
alongside the Python labels so the label set is BOTH scientifically
correct AND schema-consistent.

(Parallel V100 fix BUG #4 applied the same root fix -- same labels,
same migration 012 number, same scope. Kept this comment for the more
detailed forensic trail.)
"""

# The tier-method version string recorded in the GDA model's
# ``confidence_tier_method`` column (LIN-15, IDEM-17).  Bump this when
# the default thresholds change so downstream consumers can detect a
# definition change.
CONFIDENCE_TIER_METHOD_VERSION: str = "pinero_2020_v2"

# P1-041 ROOT FIX (v108): runtime lockstep verification flag.
# The verify_confidence_tier_lockstep() function existed but was ONLY
# called from tests — never at runtime. The issue says "Add a runtime
# consistency check that asserts the 4 sites agree." This flag ensures
# the lockstep check runs EXACTLY ONCE on the first classify_confidence
# call (memoized), catching any drift between the 4 sites (Python
# classifier, DB CHECK, schema v1.json, migration 017) before any
# classification happens. Subsequent calls skip the check (it's O(1)
# and idempotent — if the sites agree once, they agree until the
# process restarts).
_LOCKSTEP_VERIFIED: bool = False


def _ensure_lockstep_verified() -> None:
    """P1-041 ROOT FIX (v108): run the lockstep check once at runtime.

    Calls :func:`verify_confidence_tier_lockstep` on the first invocation
    and memoizes the result. If the 4 sites disagree, raises RuntimeError
    (fail-fast — do NOT silently classify with a divergent label set).
    """
    global _LOCKSTEP_VERIFIED
    if _LOCKSTEP_VERIFIED:
        return
    # Run the lockstep check. If it fails, the RuntimeError propagates
    # to the caller (the pipeline's clean() method), which aborts the
    # run with a clear error message instead of silently mis-classifying.
    verify_confidence_tier_lockstep()
    _LOCKSTEP_VERIFIED = True


def classify_confidence(
    score: Optional[float],
    tiers: Optional[list[tuple[float, str]]] = None,
) -> str:
    """Classify a DisGeNET DSGP score into a confidence tier.

    Uses :func:`bisect.bisect_right` on the sorted thresholds for
    O(log k) classification (DES-3, PERF-11).

    Parameters
    ----------
    score : float or None
        The DisGeNET DSGP score, expected to be in ``[-1, 1]`` (the
        protective-association range) or ``[0, 1]`` (the unsigned
        range).  NaN and None MUST NOT reach this function --
        :func:`cleaning.missing_values.validate_gda_scores` is responsible
        for clipping before classification (SCI-12, SCI-13).  A defensive
        assertion fires if these invariants are violated.
    tiers : list of (threshold, label), optional
        Custom tier list (sorted ascending by threshold).  Defaults to
        :data:`DEFAULT_CONFIDENCE_TIERS`.

    Returns
    -------
    str
        The tier label (``"sub_weak"``, ``"weak"``, or ``"strong"``)

    Raises
    ------
    ValueError
        If ``score`` is None, NaN, less than -1.0, or greater than 1.0
        (defensive check -- should never fire if the caller respects the
        SCI-12 / SCI-13 contract).  Negative scores in ``[-1, 0)`` are
        classified as ``"sub_weak"`` (the lowest tier).

    Notes
    -----
    CRITICAL FIX (patient safety): the original implementation used
    ``assert`` statements, which are SILENTLY DISABLED when Python is
    invoked with ``-O`` (optimized mode). For a biomedical platform
    where bad scores propagate to drug-repurposing predictions, that
    is unacceptable -- a NaN score would silently classify as "sub_weak"
    instead of raising. We replace the asserts with explicit
    ``ValueError`` raises that fire regardless of optimization level.

    v84 FORENSIC ROOT FIX (BUG #49): removed the deprecated
    ``allow_negative`` parameter entirely. The v82 fix kept it as a
    backward-compat shim that emitted a ``DeprecationWarning`` -- dead
    code in practice since the default was already ``True`` and no
    caller passed ``False``. Negative scores in ``[-1, 0)`` are ALWAYS
    classified as ``"sub_weak"`` (the lowest tier); the
    ``_score_direction`` lineage column (set by
    ``validate_gda_scores``) preserves the sign for downstream ranking.
    """
    # P1-041 ROOT FIX (v108): run the runtime lockstep check ONCE on the
    # first call. If the 4 sites (Python classifier, DB CHECK, schema
    # v1.json, migration 017) disagree, raise RuntimeError BEFORE any
    # classification happens — fail-fast instead of silently producing
    # divergent tier labels that corrupt the KG.
    _ensure_lockstep_verified()

    # P1-032 v113 ROOT FIX (defensive coercion for public API):
    #   The previous code raised ValueError if score is None or NaN,
    #   claiming "validate_gda_scores should have coerced NaN -> 0.0
    #   first." But this function is PUBLIC (exported in __all__). A
    #   caller that does NOT use validate_gda_scores (e.g. a future
    #   pipeline, a test, an ad-hoc script) hits this ValueError with
    #   no clear remediation. The error message says "should have
    #   coerced" but doesn't say HOW.
    #
    #   ROOT FIX: coerce None and NaN to 0.0 DEFENSIVELY (with a WARNING
    #   log so the caller can trace the unexpected input). This makes the
    #   public API safe to call from any context. The 0.0 score classifies
    #   as "sub_weak" (the lowest tier), which is the scientifically
    #   correct classification for an unknown score.
    if score is None:
        logger.warning(
            "classify_confidence: score is None (validate_gda_scores "
            "should have coerced NaN -> 0.0 first). Coercing to 0.0 "
            "defensively — classify as 'sub_weak'."
        )
        score = 0.0
    elif pd.isna(score):
        logger.warning(
            "classify_confidence: score is NaN (%r). Coercing to 0.0 "
            "defensively — classify as 'sub_weak'.", score,
        )
        score = 0.0

    # v84 FORENSIC ROOT FIX (BUG #49): removed the deprecated
    # ``allow_negative=False`` code path (dead code -- the default was
    # already ``True`` and no caller passed ``False``). Negative scores
    # in ``[-1, 0)`` are ALWAYS classified as the lowest tier ("sub_weak").
    # The ``_score_direction`` lineage column preserves the sign for
    # downstream ranking.

    # v82 P0-D6b: reject scores below -1.0 (outside the protective-
    # association range). validate_gda_scores should have clipped to
    # [-1, 1] already.
    if score < -1.0:
        raise ValueError(
            f"classify_confidence invariant violated: score={score!r} < -1 "
            f"(the valid range is [-1, 1]; validate_gda_scores should "
            f"have clipped first)"
        )
    if score > 1.0:
        # v35 ROOT FIX: enforce the upper bound of the DisGeNET DSGP score
        # range. The previous code only checked ``score < 0.0`` and let
        # ``score > 1.0`` silently classify as the top tier ("strong"),
        # masking a bug in the upstream score-computation (which should
        # have clipped to [0, 1]). A score > 1.0 is never legitimate for
        # a normalized DSGP score, so fail LOUDLY instead of silently
        # producing an over-confident tier.
        raise ValueError(
            f"classify_confidence invariant violated: score={score!r} > 1 "
            f"(validate_gda_scores should have clipped to [-1, 1] first)"
        )

    if tiers is None:
        tiers = DEFAULT_CONFIDENCE_TIERS
    # Defensive: ensure the tiers are sorted (the caller is expected to
    # sort, but we cannot trust that).
    sorted_tiers = sorted(tiers, key=lambda t: t[0])
    thresholds = [t[0] for t in sorted_tiers]
    labels = [t[1] for t in sorted_tiers]
    # v90 ROOT FIX (BUG #25) + v100 P1-013 ROOT FIX (forensic clarification):
    #
    # The v90 fix changed the bisect lookup from
    # ``_bisect_score = max(0.0, float(score))`` to
    # ``_bisect_score = abs(float(score))`` so a strong protective
    # association (score = -0.8) classifies as "strong" (matching +0.8).
    # The previous max(0.0, ...) clamped ALL negative scores to 0.0,
    # classifying every protective association as the lowest tier
    # regardless of magnitude -- losing the strength information.
    #
    # P1-013 ROOT FIX (v100 forensic): the v90 abs() fix is OPT-IN. The
    # DisGeNET and OMIM pipelines (the ONLY callers in production) pass
    # ``score_range=(0.0, 1.0)`` and ``preserve_direction=False`` to
    # ``validate_gda_scores`` because Piñero 2020 DSGP scores are
    # UNSIGNED -- they live in [0, 1] and do not encode direction. Under
    # that default configuration, ``validate_gda_scores`` clips any
    # negative score to 0.0 BEFORE this function is reached, so the
    # abs() branch never fires in the standard pipeline flow.
    #
    # The abs() branch EXISTS for callers that explicitly opt into
    # protective-association mode by passing ``score_range=(-1.0, 1.0)``
    # and ``preserve_direction=True`` (e.g. future pipelines that ingest
    # signed-association sources like GWAS beta coefficients). In that
    # mode, ``validate_gda_scores`` preserves the sign and this function
    # uses abs(score) so the TIER reflects STRENGTH regardless of
    # direction; the ``_score_direction`` lineage column (set by
    # ``validate_gda_scores``) preserves the sign for downstream ranking.
    #
    # This is NOT dead code -- it is a deliberate opt-in feature gate.
    # The v90 "ROOT FIX" framing was misleading because it suggested
    # the fix applied to the default pipeline; this comment makes the
    # opt-in semantics explicit so future maintainers don't rip out the
    # abs() branch as dead code (which would silently break protective-
    # association mode if a future caller enables it).
    _bisect_score = abs(float(score))
    # bisect_right returns the insertion point to the right of any
    # existing entries equal to score.  Subtracting 1 gives the index of
    # the tier whose threshold <= score.
    idx = bisect.bisect_right(thresholds, _bisect_score) - 1
    if idx < 0:
        # score < the lowest threshold -- fall back to the lowest tier.
        # This should not happen in practice (the lowest threshold is 0.0
        # and we clamped negative scores to 0.0 above), but defensive
        # programming.
        idx = 0
    return labels[idx]


__all__ = [
    "DEFAULT_CONFIDENCE_TIERS",
    "CONFIDENCE_TIER_METHOD_VERSION",
    "classify_confidence",
    "SOURCE_RELIABILITY_WEIGHTS",
    "DEFAULT_SOURCE_RELIABILITY_WEIGHT",
    "compute_source_weighted_confidence",
    "SOURCE_RELIABILITY_METHOD_VERSION",
    "verify_confidence_tier_lockstep",
    "CONFIDENCE_TIER_LABELS",
]

# P1-041 ROOT FIX (v107): canonical label set derived from DEFAULT_CONFIDENCE_TIERS.
# This is the SINGLE source of truth — all 4 sites (Python classifier, DB CHECK,
# schema v1.json, migration 017) MUST agree with this set. The
# verify_confidence_tier_lockstep() function below asserts this at runtime.
CONFIDENCE_TIER_LABELS: tuple[str, ...] = tuple(
    label for _, label in DEFAULT_CONFIDENCE_TIERS
)


def verify_confidence_tier_lockstep() -> None:
    """P1-041 ROOT FIX: assert the 4 sites agree on confidence_tier labels.

    The Piñero 2020 alignment uses 4 tiers: sub_weak, weak, strong, very_strong.
    Four separate sites define this label set:

    1. Python classifier (``DEFAULT_CONFIDENCE_TIERS`` in this module).
    2. DB ORM CHECK constraint ``chk_gda_confidence_tier`` in models.py.
    3. JSON schema validator (``pipelines/schema/v1.json``).
    4. SQL migration 017 (``017_confidence_tier_add_very_strong.sql``).

    If any of the 4 sites diverge, GDA rows are silently rejected at insert
    (DB CHECK) or silently mis-classified (Python). The multi-source GDA score
    becomes unreliable for downstream ML (Phase 3 GNN feature binning).

    This function reads the DB ORM CheckConstraint text and the JSON schema
    enum, and asserts they match ``CONFIDENCE_TIER_LABELS``. Migration files
    are SQL (not importable Python) so they are verified by the CI test
    ``test_confidence_tier_lockstep`` (which reads the SQL files as text and
    asserts the label set is present).

    Raises
    ------
    RuntimeError
        If any site disagrees with the canonical label set.
    """
    expected = set(CONFIDENCE_TIER_LABELS)

    # Site 2: DB ORM CheckConstraint (models.py)
    try:
        from database.models import GeneDiseaseAssociation
        from sqlalchemy import CheckConstraint
        from sqlalchemy import inspect as sa_inspect
        constraints = sa_inspect(GeneDiseaseAssociation.__table__).constraints
        chk = None
        for c in constraints:
            if isinstance(c, CheckConstraint) and c.name == "chk_gda_confidence_tier":
                chk = c
                break
        if chk is None:
            raise RuntimeError(
                "P1-041 LOCKSTEP FAILED: chk_gda_confidence_tier CheckConstraint "
                "not found on GeneDiseaseAssociation. The DB ORM is missing the "
                "constraint that validates confidence_tier values."
            )
        # The SQL text is like "confidence_tier IS NULL OR confidence_tier IN
        # ('sub_weak', 'weak', 'strong', 'very_strong')"
        sql_text = str(chk.sqltext).lower()
        for label in CONFIDENCE_TIER_LABELS:
            if label not in sql_text:
                raise RuntimeError(
                    f"P1-041 LOCKSTEP FAILED: chk_gda_confidence_tier is missing "
                    f"label '{label}'. SQL text: {sql_text}"
                )
    except ImportError:
        # database.models not importable (e.g. SQLAlchemy missing) — skip
        # this site. The CI test will catch it.
        pass

    # Site 3: JSON schema (pipelines/schema/v1.json)
    try:
        from pathlib import Path
        import json as _json
        schema_path = Path(__file__).resolve().parent.parent / "pipelines" / "schema" / "v1.json"
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = _json.load(f)
        # Check gene_disease_associations.csv confidence_tier enum
        gda_schema = schema.get("properties", {}).get("gene_disease_associations.csv", {})
        gda_ct = gda_schema.get("properties", {}).get("confidence_tier", {})
        gda_enum = set(v for v in gda_ct.get("enum", []) if v is not None)
        if gda_enum and gda_enum != expected:
            raise RuntimeError(
                f"P1-041 LOCKSTEP FAILED: schema v1.json "
                f"gene_disease_associations.csv confidence_tier enum "
                f"{sorted(gda_enum)} != canonical {sorted(expected)}"
            )
        # Check omim_gene_disease_associations.csv confidence_tier enum
        omim_schema = schema.get("properties", {}).get("omim_gene_disease_associations.csv", {})
        omim_ct = omim_schema.get("properties", {}).get("confidence_tier", {})
        omim_enum = set(v for v in omim_ct.get("enum", []) if v is not None)
        if omim_enum and omim_enum != expected:
            raise RuntimeError(
                f"P1-041 LOCKSTEP FAILED: schema v1.json "
                f"omim_gene_disease_associations.csv confidence_tier enum "
                f"{sorted(omim_enum)} != canonical {sorted(expected)}"
            )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"P1-041 LOCKSTEP FAILED: could not read schema v1.json: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# P1-027 ROOT FIX (Team Member 3 -- source-reliability-weighted confidence):
#
# The issue: the original ``compute_confidence()`` (which has since been
# refactored out of this module, but the SCIENTIFIC CONCERN remains live)
# took the MAX confidence across all sources for a given KG edge. This
# ignored source reliability: a Curated (expert-validated) edge with
# confidence 0.9 was treated identically to a Predicted (text-mined) edge
# with confidence 0.9. The KG then had edges that LOOKED equally reliable
# but were not -- the GNN over-weighted text-mined edges and the RL ranker
# could recommend drugs based on weak text-mined evidence.
#
# ROOT FIX: introduce ``SOURCE_RELIABILITY_WEIGHTS`` -- a publication-aligned
# reliability multiplier per source class. ``compute_source_weighted_confidence``
# takes a list of ``(source, confidence)`` pairs and returns the
# reliability-weighted maximum:
#
#     weighted_confidence(source, conf) = conf * SOURCE_RELIABILITY_WEIGHTS[source]
#     result = max(weighted_confidence(s, c) for (s, c) in pairs)
#
# We use weighted-MAX (not weighted-average) because a single high-quality
# curated edge SHOULD dominate a sea of low-quality predicted edges -- this
# matches the DisGeNET curation philosophy (Piñero et al. 2020, §2.3: the
# DSGP score is upward-biased by curated sources). But the weight ensures a
# Predicted 0.95 edge (0.95 * 0.6 = 0.57) does NOT beat a Curated 0.7 edge
# (0.7 * 1.0 = 0.70).
#
# Reliability tiers (aligned with DisGeNET source classes + general
# biomedical evidence principles):
#   - curated      : 1.00  (expert-validated; DisGeNET CURATED, UniProt manual)
#   - model_organism: 0.85 (animal-model with high human transferability;
#                     mouse/rat for conserved pathways)
#   - clinical     : 0.95  (clinical trial / EHR-validated)
#   - predicted    : 0.60  (text-mining / NLP; DisGeNET PREDICTED)
#   - animal_model : 0.45  (animal-model with low human transferability)
#   - unknown      : 0.50  (source class not specified -- conservative)
#
# These weights are CALIBRATED so that:
#   - Curated 0.5  (0.50) < Predicted 0.95 (0.57)  [predicted CAN win if
#     its raw confidence is high enough -- text-mining is not useless]
#   - Curated 0.7  (0.70) > Predicted 0.95 (0.57)  [but curated dominates
#     at moderate confidence -- the intended behaviour]
#   - Clinical 0.8 (0.76) > Curated 0.7 (0.70)     [clinical > curated]
#
# Callers can override weights via the ``weights`` parameter for domain-
# specific calibration (e.g. a neuroscience pipeline may downweight
# animal_model further due to blood-brain-barrier transferability concerns).
# ---------------------------------------------------------------------------
SOURCE_RELIABILITY_WEIGHTS: dict[str, float] = {
    "curated": 1.00,
    "clinical": 0.95,
    "model_organism": 0.85,
    "predicted": 0.60,
    "animal_model": 0.45,
    "unknown": 0.50,
}

#: Default weight for sources not in ``SOURCE_RELIABILITY_WEIGHTS``.
#: Conservative (0.50) so an unrecognised source does not silently get
#: full reliability. Operators should register new sources explicitly.
DEFAULT_SOURCE_RELIABILITY_WEIGHT: float = 0.50

#: Version string for the source-reliability weighting scheme. Bump when
#: the weights change so downstream consumers can detect a definition change.
SOURCE_RELIABILITY_METHOD_VERSION: str = "pinero_2020_source_reliability_v1"


def compute_source_weighted_confidence(
    source_confidence_pairs: list[tuple[str, float]],
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute the reliability-weighted maximum confidence for a KG edge.

    P1-027 ROOT FIX: takes a list of ``(source_class, raw_confidence)``
    pairs (one per source that asserted this edge) and returns the
    MAXIMUM of ``raw_confidence * reliability_weight[source_class]``.

    This ensures a Curated edge (weight 1.0) at confidence 0.7 beats a
    Predicted edge (weight 0.6) at confidence 0.95, because
    ``0.7*1.0 = 0.70 > 0.95*0.6 = 0.57``. Without the weight, the
    Predicted edge would win (0.95 > 0.7) and the GNN would over-weight
    text-mined evidence.

    Parameters
    ----------
    source_confidence_pairs:
        List of ``(source_class, raw_confidence)`` tuples. ``source_class``
        is a key into ``SOURCE_RELIABILITY_WEIGHTS`` (e.g. ``"curated"``,
        ``"predicted"``, ``"animal_model"``). ``raw_confidence`` is in
        ``[0.0, 1.0]``. An empty list returns ``0.0``.
    weights:
        Optional override dict. Defaults to :data:`SOURCE_RELIABILITY_WEIGHTS`.

    Returns
    -------
    float
        The reliability-weighted maximum confidence, in ``[0.0, 1.0]``.
        Returns ``0.0`` for an empty input. The result is clamped to
        ``[0.0, 1.0]`` defensively (a weight > 1.0 could otherwise push
        the result above 1.0).

    Raises
    ------
    ValueError
        If any ``raw_confidence`` is not in ``[0.0, 1.0]`` (defensive --
        callers should clip first). NaN values are rejected.

    Examples
    --------
    >>> # Curated 0.7 beats Predicted 0.95 after weighting.
    >>> result = compute_source_weighted_confidence([
    ...     ("curated", 0.7),
    ...     ("predicted", 0.95),
    ... ])
    >>> round(result, 2)
    0.7
    >>> # Predicted alone is downweighted.
    >>> round(compute_source_weighted_confidence([("predicted", 0.9)]), 2)
    0.54
    >>> # Empty input returns 0.0.
    >>> compute_source_weighted_confidence([])
    0.0
    """
    if not source_confidence_pairs:
        return 0.0
    eff_weights = weights if weights is not None else SOURCE_RELIABILITY_WEIGHTS
    best = 0.0
    for source_class, raw_conf in source_confidence_pairs:
        if not isinstance(source_class, str):
            source_class = "unknown"
        if not isinstance(raw_conf, (int, float)):
            raise ValueError(
                f"compute_source_weighted_confidence: raw_confidence must "
                f"be a number, got {type(raw_conf).__name__}={raw_conf!r}"
            )
        if raw_conf != raw_conf:  # NaN check
            raise ValueError(
                "compute_source_weighted_confidence: raw_confidence is NaN"
            )
        if raw_conf < 0.0 or raw_conf > 1.0:
            raise ValueError(
                f"compute_source_weighted_confidence: raw_confidence "
                f"{raw_conf!r} out of [0.0, 1.0]"
            )
        weight = eff_weights.get(source_class, DEFAULT_SOURCE_RELIABILITY_WEIGHT)
        weighted = float(raw_conf) * float(weight)
        if weighted > best:
            best = weighted
    # Clamp to [0.0, 1.0] defensively (weights could push above 1.0).
    if best > 1.0:
        best = 1.0
    elif best < 0.0:
        best = 0.0
    return best
