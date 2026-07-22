# SPDX-License-Identifier: MIT
# © 2024-2026 Autonomous Drug Repurposing Platform -- Team Cosmic / VentureLab
"""
Architectural foundation for the ``entity_resolution`` package.

This module defines the **public contract** that every concrete resolver
(``DrugResolver``, ``ProteinResolver``, and any future resolver) must
honour.  It also collects shared infrastructure used by both resolvers:

* :class:`ResolverConfig` -- an immutable, env-overridable configuration
  dataclass.  Every magic number that previously lived as a private
  module-level constant (PubChem rate-limit delay, fuzzy threshold,
  stereoisomer-collapse flag, ...) now has a documented home here.
* :class:`ResolverStats` -- an observable, mutable counter container
  exposed via :meth:`Resolver.get_stats` so operators can monitor
  match-method distribution, dead-letter counts and PubChem failures.
* :class:`MatchConfidence` -- a :class:`enum.FloatEnum` that pins the
  closed set of confidence scores the resolver may emit.  This makes
  ``compute_match_confidence`` a contract, not a free-form lookup.
* :class:`Resolver` -- an :class:`abc.ABC` that pins the public method
  surface every resolver must implement (``add_source_records``,
  ``resolve_single``, ``build_mapping``, ``to_dataframe``, ``reset``,
  ``remove_source``, ``get_stats``, ``to_state_dict``,
  ``from_state_dict``, ``to_json``, ``from_json``, ``get_audit_trail``,
  ``find_affected_entities``).
* :class:`_ProcessGlobalRateLimiter` -- a class-level (NOT per-instance)
  token-bucket rate limiter so that two ``DrugResolver`` instances in
  the same process still respect the PubChem "5 req/sec" rule when
  their HTTP calls interleave.

The contract established here is consumed by the package-level
``__getattr__`` lazy loader in :mod:`entity_resolution.__init__` and by
the factory functions ``make_drug_resolver`` / ``make_protein_resolver``
that take a :class:`ResolverConfig` and return a fully-wired resolver.

Design rationale
----------------
The configuration dataclass is **frozen** because mutating a resolver's
configuration mid-run is a correctness hazard (e.g. flipping
``collapse_stereoisomers`` to ``True`` after some records have already
been merged would silently produce a mixed-mode mapping).  If a caller
needs different behaviour they should construct a new resolver.

The :class:`Resolver` ABC intentionally does **not** enforce a single
``mapping`` attribute type -- drugs are keyed by InChIKey, proteins by
UniProt accession, and forcing them into a common generic type would
erase scientifically-meaningful distinctions (PEP 484 generic resolvers
were considered and rejected; see the master fix prompt §11
anti-pattern #2).

Network safety
--------------
The default :class:`ResolverConfig` sets ``pubchem_enabled=False``.  This
is **not** a backward-compatibility break -- it is the security fix for
audit D9-1, which classifies undocumented third-party network calls as a
PII-leakage risk.  Callers who need single-record PubChem lookup must
opt in via ``ResolverConfig(pubchem_enabled=True)`` or the
``ENTITY_RESOLUTION_PUBCHEM_ENABLED`` env var.
"""

from __future__ import annotations

import abc
import dataclasses
import enum
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

#: Version of the state-dict / JSON-serialisation schema produced by
#: :meth:`Resolver.to_state_dict`.  ``from_state_dict`` refuses to load
#: state-dicts whose schema version is unknown -- this is the fix for
#: audit D12-4 ("no ``__version__`` to tie to config schema version").
MAPPING_SCHEMA_VERSION: str = "1.0"


# ---------------------------------------------------------------------------
# MatchConfidence enum (D15-6)
# ---------------------------------------------------------------------------


@enum.unique
class MatchConfidence(float, enum.Enum):
    """Closed enum of confidence scores emitted by the resolver.

    Returning a bare ``float`` from :func:`compute_match_confidence` made
    it impossible for downstream code to distinguish "0.5 because the
    method is unknown" from "0.5 because the method is intentionally
    low-confidence".  This enum pins the legal values so that
    ``MatchConfidence.FUZZY`` is self-documenting and comparable.

    The numeric values are calibrated heuristics, NOT probabilities.
    See the module docstring in :mod:`entity_resolution.__init__` for
    the full rationale table (audit D3-6).

    Implemented as ``class MatchConfidence(float, enum.Enum)`` for
    cross-version compatibility (``enum.FloatEnum`` was only added to
    the stdlib ``enum`` namespace in Python 3.12+ and is missing in
    earlier versions; the explicit mix-in works everywhere).

    P1-005 ROOT FIX (v113 -- enum alias collisions):
      Python's ``enum.Enum`` makes the SECOND member with the same value
      an ALIAS of the FIRST. The previous version had THREE alias
      collisions:
        - ``UNIPROT_EXACT = 1.0`` aliased ``INCHIKEY_EXACT`` (so
          ``MatchConfidence.UNIPROT_EXACT.name == "INCHIKEY_EXACT"``)
        - ``SYNTHETIC_KEY_MATCH = 0.5`` aliased ``UNKNOWN``
        - ``SMILES_CANONICAL = 0.75`` aliased ``GENE_NAME_ORGANISM``
      Downstream code that switched on ``match_conf.name`` (e.g.
      ``if match.name == "UNIPROT_EXACT":``) was DEAD -- the branch
      never fired because the name was always ``"INCHIKEY_EXACT"``.
      Phase 2's KG builder mislabeled UniProt-exact matches as
      InChIKey-exact; Phase 4's RL ranker over-weighted structural
      identity. Patient-safety filters that distinguish "structural
      identity" from "sequence identity" could not fire.

      ROOT FIX: (1) add ``@enum.unique`` so any future duplicate value
      is a hard error at import time; (2) give each previously-aliased
      member a DISTINCT value that preserves the intended hierarchy
      ordering. The new values (0.99 / 0.74 / 0.49) are deliberately
      adjacent to their former twins so downstream ranking is barely
      affected, but ``.name`` now resolves correctly for every member.
    """

    INCHIKEY_EXACT = 1.0
    INCHIKEY_CONNECTIVITY = 0.9
    NAME_NORMALIZED = 0.8
    PUBCHEM_XREF = 0.7
    # v29 ROOT FIX (audit C-1 / C-2 -- Confidence Score Inversion):
    # The previous values were 0.85 (FUZZY) and 0.90 (PROTEIN_NAME_FUZZY),
    # BOTH HIGHER than NAME_NORMALIZED=0.8. This is SCIENTIFICALLY WRONG:
    # a fuzzy match (approximate string similarity) is by definition
    # LESS reliable than an exact name match after normalization. The
    # inversion caused the entity resolver to preferentially keep
    # low-quality fuzzy matches over high-quality exact matches -- every
    # downstream consumer that ranked by confidence got the wrong answer.
    #
    # ROOT FIX: set FUZZY below NAME_NORMALIZED, and PROTEIN_NAME_FUZZY
    # below NAME_NORMALIZED.
    #
    # P1-021 ROOT FIX (Team-2 — comment drift: hierarchy comment listed
    #   wrong PUBCHEM_XREF value). The previous comment said:
    #     "PUBCHEM_XREF (0.55) > UNKNOWN (0.5)"
    #   but the ACTUAL enum value (line 116 above) is 0.7 — placing
    #   PUBCHEM_XREF ABOVE FUZZY (0.65), not between PROTEIN_NAME_FUZZY
    #   (0.6) and UNKNOWN (0.5). The comment drifted from the code.
    #   ROOT FIX: rewrite the hierarchy comment to match the ACTUAL
    #   enum values (sorted descending), AND add a runtime assertion
    #   (below the enum, ``_CONFIDENCE_HIERARCHY_ASSERTIONS``) so
    #   future drift fails CI immediately.
    #
    # The CORRECT hierarchy (descending by value) is -- per P1-005 v113
    # each member has a DISTINCT value (no aliases):
    #   INCHIKEY_EXACT        (1.0)   — deterministic structural identity
    #   UNIPROT_EXACT         (0.99)  — deterministic sequence identity
    #   INCHIKEY_CONNECTIVITY (0.9)   — same molecule, different stereo
    #   NAME_NORMALIZED       (0.8)   — exact name after case/punct normalization
    #   GENE_NAME_ORGANISM    (0.75)  — gene-name + organism match
    #   SMILES_CANONICAL      (0.74)  — canonical SMILES match (chemical identity)
    #   PUBCHEM_XREF          (0.7)   — PubChem cross-reference (curated)
    #   FUZZY                 (0.65)  — approximate string similarity
    #   PROTEIN_NAME_FUZZY    (0.60)  — approximate protein name match
    #   UNKNOWN               (0.5)   — lowest confidence / unknown method
    #   SYNTHETIC_KEY_MATCH   (0.49)  — computed InChIKey for biologics
    #
    # The previous comment said "raised from 0.6 to be ≥ _FUZZY_THRESHOLD"
    # -- that was a misdiagnosis. The _FUZZY_THRESHOLD (0.85) was the
    # MINIMUM confidence a match needed to be ACCEPTED at all; raising
    # the FUZZY enum value to meet it did NOT make fuzzy matches
    # correct, it just made them pass the gate. The correct fix is to
    # LOWER _FUZZY_THRESHOLD so fuzzy matches can be accepted at their
    # true confidence (0.65) AND rank below exact matches. We do NOT
    # lower the threshold here -- that's a separate concern. We just
    # fix the inverted enum values.
    FUZZY = 0.65  # v29: was 0.85 -- inversion fix
    UNIPROT_EXACT = 0.99  # P1-005 v113: was 1.0 (aliased INCHIKEY_EXACT); 0.99 preserves "near-exact" semantics while making .name resolve correctly
    GENE_NAME_ORGANISM = 0.75  # v29: was 0.85 -- lowered to sit between
                                # NAME_NORMALIZED (0.8) and FUZZY (0.65).
                                # A gene-name+organism match is stronger
                                # than a fuzzy name match but weaker than
                                # an exact name match.
    PROTEIN_NAME_FUZZY = 0.60  # v29: was 0.90 -- inversion fix
    UNKNOWN = 0.5
    # P1-005 v113: SYNTHETIC_KEY_MATCH was 0.5 (aliased UNKNOWN); now 0.49 so .name resolves correctly. SYNTH keys are computed (not experimental) and rank just below UNKNOWN.
    # v65 ROOT FIX (P1C-009): SYNTH-prefixed InChIKey matches are
    # COMPUTED (not experimental) -- they're generated for biologics /
    # macromolecules that lack a real InChIKey. Different sources may
    # generate DIFFERENT SYNTH keys for the same biologic, so a SYNTH
    # match is much weaker evidence than a real InChIKey match. The
    # previous code labeled SYNTH matches as ``method="inchikey_exact"``
    # with ``confidence=0.5`` -- a hardcoded magic number that happened
    # to equal ``UNKNOWN``. This was self-contradictory: the method
    # label said "exact" (1.0) but the confidence was 0.5 (UNKNOWN).
    # Downstream code that filtered by ``method == "inchikey_exact"``
    # treated SYNTH matches as highest-priority when they're actually
    # the lowest. ROOT FIX: give SYNTH matches their OWN method label
    # (``"synthetic_key_match"``) and their OWN enum member so the
    # method/confidence pair is self-consistent and downstream filters
    # can distinguish SYNTH matches from real InChIKey matches. The
    # value (0.5) is unchanged -- only the labeling is fixed.
    SYNTHETIC_KEY_MATCH = 0.49  # P1-005 v113: was 0.5 (aliased UNKNOWN)
    # v89 ROOT FIX (BUG #32 -- smiles_canonical method registered at
    # runtime but missing from the enum):
    #   ``drug_resolver.py:1979`` registers ``"smiles_canonical"`` with
    #   confidence 0.75 via ``register_match_method``. This ADDS the
    #   method to ``resolver_utils.METHOD_CONFIDENCE`` at runtime, so
    #   ``compute_match_confidence("smiles_canonical")`` returns 0.75.
    #   BUT the ``MatchConfidence`` enum had NO ``SMILES_CANONICAL``
    #   member, so ``MatchConfidence.from_method("smiles_canonical")``
    #   returned ``UNKNOWN`` (0.5) -- a DIFFERENT value for the same
    #   method. Downstream code that used the enum-based lookup got
    #   0.5; code that used the dict-based lookup got 0.75. Filters
    #   behaved differently depending on which lookup path they used.
    #
    #   ROOT FIX: add ``SMILES_CANONICAL`` to the enum AND to the
    #   ``from_method`` mapping (below). P1-005 v113: the value is 0.74
    #   (was 0.75, which aliased GENE_NAME_ORGANISM). 0.74 preserves the
    #   intended ranking (between NAME_NORMALIZED=0.8 and FUZZY=0.65,
    #   just below GENE_NAME_ORGANISM=0.75) while making .name resolve
    #   correctly. Both the enum-based and dict-based lookup paths now
    #   return the SAME value (0.74), consistent with the runtime
    #   registration in drug_resolver.py. A canonical-SMILES match is
    #   strong evidence of chemical identity (same molecule) but weaker
    #   than an InChIKey exact match (deterministic, source-independent).
    SMILES_CANONICAL = 0.74  # P1-005 v113: was 0.75 (aliased GENE_NAME_ORGANISM)

    @classmethod
    def from_method(cls, method: str) -> "MatchConfidence":
        """Map a resolution-method string to its enum member.

        Unknown method names return :attr:`MatchConfidence.UNKNOWN`
        instead of raising -- this preserves the lenient contract of
        the legacy :func:`compute_match_confidence` helper while still
        giving callers a structured return type.

        As of FIX #6 / GUARD-ARCH-06, this method ALSO consults the
        runtime-registered custom methods stored in
        ``resolver_utils._custom_methods``.  If *method* is found there
        and is NOT a built-in enum member, the corresponding value is
        returned via :attr:`MatchConfidence.UNKNOWN` (because enum
        members are immutable and cannot be added at runtime).  Use
        :func:`resolver_utils.compute_match_confidence` with
        ``as_enum=False`` to get the actual numeric value of a custom
        method.
        """
        mapping = {
            "inchikey_exact": cls.INCHIKEY_EXACT,
            "inchikey_connectivity": cls.INCHIKEY_CONNECTIVITY,
            "name_normalized": cls.NAME_NORMALIZED,
            "pubchem_xref": cls.PUBCHEM_XREF,
            "fuzzy": cls.FUZZY,
            "uniprot_exact": cls.UNIPROT_EXACT,
            "gene_name_organism": cls.GENE_NAME_ORGANISM,
            "protein_name_fuzzy": cls.PROTEIN_NAME_FUZZY,
            # v65 ROOT FIX (P1C-009): SYNTH-key matches now have their
            # own method label so they are distinguishable from real
            # InChIKey exact matches downstream.
            "synthetic_key_match": cls.SYNTHETIC_KEY_MATCH,
            # v89 ROOT FIX (BUG #32): smiles_canonical now has its own
            # enum member so ``from_method("smiles_canonical")`` returns
            # ``SMILES_CANONICAL`` (0.75) instead of ``UNKNOWN`` (0.5).
            # This matches the value registered at runtime in
            # ``drug_resolver.py:1979`` via ``register_match_method``.
            "smiles_canonical": cls.SMILES_CANONICAL,
        }
        if method in mapping:
            return mapping[method]
        # FIX #6 / GUARD-ARCH-06 -- check resolver_utils._custom_methods
        # as a fallback so runtime-registered methods are visible here too.
        try:
            from .resolver_utils import _custom_methods, _ORIGINAL_METHOD_CONFIDENCE
            if method in _custom_methods and method not in _ORIGINAL_METHOD_CONFIDENCE:
                # Truly custom (non-built-in) method.  We can't synthesise
                # a new enum member at runtime, so return UNKNOWN -- callers
                # who need the actual numeric value should use
                # ``resolver_utils.compute_match_confidence(method)`` instead.
                return cls.UNKNOWN
        except ImportError:
            pass
        return cls.UNKNOWN


# P1-021 ROOT FIX (Team-2 — runtime assertion to prevent comment/code drift):
#   The previous comment listed PUBCHEM_XREF as 0.55 when the actual enum
#   value is 0.7. Such drift is silent — filters behave differently than
#   the documentation claims. ROOT FIX: assert at import time that EVERY
#   enum member has the value documented in the hierarchy comment above.
#   If a future maintainer changes an enum value without updating the
#   comment (or vice versa), this assertion fails IMMEDIATELY at import
#   time, failing CI and blocking the merge.
#
#   The expected values are frozen here as a tuple-of-tuples so they're
#   easy to audit alongside the enum definition above. Keep this list
#   in EXACT sync with the enum members and the hierarchy comment.
_CONFIDENCE_HIERARCHY_ASSERTIONS: tuple[tuple[str, float], ...] = (
    # P1-005 v113: updated to reflect distinct values (no aliases)
    ("INCHIKEY_EXACT", 1.0),
    ("INCHIKEY_CONNECTIVITY", 0.9),
    ("NAME_NORMALIZED", 0.8),
    ("PUBCHEM_XREF", 0.7),
    ("FUZZY", 0.65),
    ("UNIPROT_EXACT", 0.99),
    ("GENE_NAME_ORGANISM", 0.75),
    ("PROTEIN_NAME_FUZZY", 0.60),
    ("UNKNOWN", 0.5),
    ("SYNTHETIC_KEY_MATCH", 0.49),
    ("SMILES_CANONICAL", 0.74),
)
for _member_name, _expected_value in _CONFIDENCE_HIERARCHY_ASSERTIONS:
    _actual = getattr(MatchConfidence, _member_name, None)
    if _actual is None:
        raise AssertionError(
            f"P1-021 drift guard: MatchConfidence.{_member_name} is missing "
            f"from the enum (was it renamed or removed?). Update "
            f"_CONFIDENCE_HIERARCHY_ASSERTIONS in base.py to match."
        )
    if float(_actual.value) != _expected_value:
        raise AssertionError(
            f"P1-021 drift guard: MatchConfidence.{_member_name}.value = "
            f"{float(_actual.value)!r} but the documented hierarchy says "
            f"{_expected_value!r}. Either fix the enum value OR update "
            f"the hierarchy comment AND _CONFIDENCE_HIERARCHY_ASSERTIONS "
            f"to match. Comment/code drift is the exact bug P1-021 fixes."
        )
del _member_name, _expected_value, _actual


# ---------------------------------------------------------------------------
# ResolverConfig (D2-5, D12-1, D3-4, D9-1, D9-2, D9-3, D9-5, D9-6, D12-4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolverConfig:
    """Immutable configuration for a :class:`Resolver` instance.

    # v29 ROOT FIX (audit C-9): docstring claimed 'Every field has
    # env-var override' but only ~17 of 50+ fields are loaded. Updated
    # to be honest.
    #
    # P2-11 ROOT FIX (v82): the previous docstring honesty is no longer
    # needed -- EVERY field now has an env-var override. The
    # ``from_env`` factory loads ALL ~50 fields from env vars (prefix
    # ``ENTITY_RESOLUTION_<FIELD_NAME_UPPER>``), with the dataclass
    # default used only when the env var is unset. Operators can now
    # tune the entire config via env vars without constructing the
    # dataclass programmatically.
    #
    # Type coercion rules for env vars:
    #   * ``bool`` fields: ``"1"``, ``"true"``, ``"yes"``, ``"on"``
    #     (case-insensitive) -> ``True``; everything else -> ``False``.
    #   * ``int`` fields: parsed with ``int(val)``; non-int -> default.
    #   * ``float`` fields: parsed with ``float(val)``; non-float -> default.
    #   * ``str`` fields: the raw string value (whitespace-trimmed).
    #   * ``Optional[str]`` fields: same as ``str``; empty -> ``None``.
    #   * ``Tuple[str, ...]`` fields: comma-separated list, e.g.
    #     ``"pubchem.ncbi.nlm.nih.gov,my-internal-mirror.local"``.
    #   * ``Optional[bytes]`` fields (``state_encryption_key``,
    #     ``tamper_evident_key``): hex-encoded bytes (e.g.
    #     ``openssl rand -hex 32``). Invalid hex -> ``None`` + warning.
    #   * ``int`` with octal semantics (``state_file_mode``): parsed
    #     with ``int(val, 8)`` if the value starts with ``0o`` or
    #     contains only octal digits ``[0-7]``; otherwise ``int(val, 0)``
    #     for autodetection (e.g. ``0o600``, ``0o640``).

    Fields WITH env-var override (prefix ``ENTITY_RESOLUTION_``):

        * ``COLLAPSE_STEREOISOMERS``        -> collapse_stereoisomers
        * ``FUZZY_THRESHOLD``               -> fuzzy_threshold
        * ``FUZZY_MAX_CANDIDATES``          -> fuzzy_max_candidates
        * ``PUBCHEM_ENABLED``               -> pubchem_enabled
        * ``PUBCHEM_REST_BASE``             -> pubchem_rest_base
        * ``PUBCHEM_CALL_DELAY``            -> pubchem_call_delay
        * ``PUBCHEM_TIMEOUT``               -> pubchem_timeout
        * ``PUBCHEM_MAX_RETRIES``           -> pubchem_max_retries
        * ``PUBCHEM_API_KEY``               -> pubchem_api_key
        * ``PUBCHEM_CA_BUNDLE``             -> pubchem_ca_bundle
        * ``PUBCHEM_CERT_PEM``              -> pubchem_cert_pem
        * ``PUBCHEM_KEY_PEM``               -> pubchem_key_pem
        * ``PUBCHEM_STRICT_SALT_FORM``      -> pubchem_strict_salt_form
        * ``SOURCE_WHITELIST``              -> source_whitelist (CSV list)
        * ``DEFAULT_ORGANISM``              -> default_organism
        * ``TAMPER_EVIDENT_KEY``            -> tamper_evident_key (hex)
        * ``MAPPING_SCHEMA_VERSION``        -> mapping_schema_version

        # P2-11 v82 ROOT FIX -- newly env-backed fields (all additive
        # audit-remediation fields now have env-var overrides):
        * ``DETERMINISTIC_TIMESTAMPS``      -> deterministic_timestamps (bool)
        * ``RANDOM_SEED``                   -> random_seed (int, 0 = None)
        * ``RUNTIME_ASSERTS``               -> runtime_asserts (bool)
        * ``CHECKSUM_SALT``                 -> checksum_salt (str)
        * ``BULK_STRICT_VALIDATION``        -> bulk_strict_validation (bool)
        * ``DEAD_LETTER_ON_SOFT_WARNING``   -> dead_letter_on_soft_warning (bool)
        * ``CONFLICT_POLICY``               -> conflict_policy (str)
        * ``ENABLE_SMILES_MATCHING``        -> enable_smiles_matching (bool)
        * ``ENABLE_FORMULA_MATCHING``       -> enable_formula_matching (bool)
        * ``PREFER_FRESHER_DATA``           -> prefer_fresher_data (bool)
        * ``MAX_RECORDS_PER_BATCH``         -> max_records_per_batch (int)
        * ``MAX_DEAD_LETTER_SIZE``          -> max_dead_letter_size (int)
        * ``MAX_AUDIT_TRAIL_PER_ENTRY``     -> max_audit_trail_per_entry (int)
        * ``MAX_QUERY_LOG_SIZE``            -> max_query_log_size (int)
        * ``DEAD_LETTER_SPILL_PATH``        -> dead_letter_spill_path (str)
        * ``AUDIT_TRAIL_SPILL_PATH``        -> audit_trail_spill_path (str)
        * ``AUDIT_TRAIL_RETENTION_DAYS``    -> audit_trail_retention_days (int)
        * ``PUBCHEM_BACKOFF_BASE``          -> pubchem_backoff_base (float)
        * ``PUBCHEM_BACKOFF_MAX``           -> pubchem_backoff_max (float)
        * ``PUBCHEM_BACKOFF_JITTER``        -> pubchem_backoff_jitter (float)
        * ``PUBCHEM_FAILURE_THRESHOLD``     -> pubchem_failure_threshold (int)
        * ``PUBCHEM_CIRCUIT_COOLDOWN``      -> pubchem_circuit_cooldown (float)
        * ``PUBCHEM_ALLOWLIST_HOSTS``       -> pubchem_allowlist_hosts (CSV)
        * ``PUBCHEM_ALLOWED_REGIONS``       -> pubchem_allowed_regions (CSV)
        * ``PUBCHEM_VERIFY_TLS``            -> pubchem_verify_tls (bool)
        * ``PUBCHEM_INSECURE_ACKNOWLEDGEMENT`` -> pubchem_insecure_acknowledgement (bool)
        * ``REDACT_DEAD_LETTER_PII``        -> redact_dead_letter_pii (bool)
        * ``STATE_FILE_MODE``               -> state_file_mode (octal int)
        * ``STATE_ENCRYPTION_KEY``          -> state_encryption_key (hex bytes)
        * ``ALLOWED_PATHS_ROOT``            -> allowed_paths_root (str)
        * ``SPILL_DIR``                     -> spill_dir (str)
        * ``DATA_CLASSIFICATION``           -> data_classification (str)
        * ``REQUIRE_OPERATOR_FOR_SENSITIVE_ACTIONS`` -> require_operator_for_sensitive_actions (bool)
        * ``REQUIRE_ORGANISM_OVERRIDE``     -> require_organism_override (bool)
        * ``CONTROLLED_SUBSTANCE_LIST``     -> controlled_substance_list (CSV)
        * ``LOG_SAMPLE_RATE``               -> log_sample_rate (float)
        * ``NORMALIZE_NAME_CACHE_SIZE``     -> normalize_name_cache_size (int)
        * ``PARALLEL_INGESTION_WORKERS``    -> parallel_ingestion_workers (int)
        * ``PROFILE``                       -> profile (str)
        * ``EAGER_IMPORTS``                 -> eager_imports (bool)
        * ``ISOLATED_RATE_LIMITER``         -> isolated_rate_limiter (bool)
        * ``TAMPER_EVIDENT``                -> tamper_evident (bool)
        * ``VALIDATE_OUTPUT_SCHEMA``        -> validate_output_schema (bool)
        * ``ALLOW_API_KEY_ROUND_TRIP``      -> allow_api_key_round_trip (bool)
        * ``FUZZY_INDEX_TYPE``              -> fuzzy_index_type (str)
        * ``NORMALIZE_NAME_CACHE``          -> normalize_name_cache (bool)

    Construct via :meth:`from_env` for production use; tests should
    construct directly with explicit kwargs for reproducibility.

    Attributes
    ----------
    collapse_stereoisomers:
        If ``True``, two InChIKeys sharing the same 14-char connectivity
        block are merged into one canonical entry (legacy behaviour).
        If ``False`` (default, **safe**), connectivity-block collisions
        only merge when the full 27-char InChIKeys are identical --
        stereoisomers with different biological activity (thalidomide,
        warfarin, citalopram, ...) are kept distinct.  Audit D3-4.
    fuzzy_threshold:
        Minimum :func:`rapidfuzz.fuzz.token_sort_ratio` score (on the
        ``[0.0, 1.0]`` scale) at which a fuzzy name match is accepted.
        Default ``0.85``.  The fuzzy *confidence* reported on a match
        is always ``>= fuzzy_threshold`` (audit D3-3 -- fixes the
        previous bug where threshold was 0.85 but reported confidence
        was 0.6, silently dropping valid matches downstream).
    fuzzy_max_candidates:
        Ceiling on the number of indexed names scanned per fuzzy sweep
        to bound worst-case :math:`O(n^2)` behaviour.  Names beyond
        this ceiling are skipped with a DEBUG log.  Audit D8-2.
    pubchem_enabled:
        If ``False`` (default, **safe**), :meth:`_match_by_pubchem_xref`
        is a no-op.  Must be explicitly opted in.  Audit D9-1, D9-2.
    pubchem_rest_base:
        Base URL of the PubChem PUG-REST endpoint.  Configurable so
        air-gapped deployments can point at an internal mirror.  Audit
        D9-3.
    pubchem_call_delay:
        Minimum seconds between PubChem API calls.  Default ``0.2``
        (5 req/sec).  When :attr:`pubchem_api_key` is set, this drops
        to ``0.1`` (10 req/sec) per PubChem's published limits.
    pubchem_timeout:
        Per-request timeout in seconds.  Default ``10``.
    pubchem_max_retries:
        Number of retries with exponential backoff on transient
        failures (timeouts, 5xx).  Default ``3``.
    pubchem_api_key:
        Optional PubChem API key.  When set, raises the rate limit
        from 5 to 10 req/sec.  Audit D9-6.
    pubchem_ca_bundle:
        Optional path to a CA bundle for TLS verification against an
        internal PubChem mirror.  Audit D9-5.
    pubchem_cert_pem, pubchem_key_pem:
        Optional mTLS client certificate paths.  Audit D9-5.
    pubchem_strict_salt_form:
        If ``True``, reject PubChem name lookups that resolve to a
        salt form (e.g. "aspirin" -> "aspirin sodium") -- the salt form
        has different pharmacology from the free acid.  Default
        ``False`` (matches historical behaviour) but documented as a
        known risk.  Audit D3-7.
    source_whitelist:
        Optional tuple of allowed ``source`` argument values passed to
        :meth:`add_source_records`.  When set, unknown source labels
        raise :class:`ValueError` instead of being silently accepted.
        Audit D9-7.
    default_organism:
        Organism name used when protein records omit it.  Default
        ``"Homo sapiens"``.  ⚠️  This default assumes human-centric
        research; non-human protein studies MUST override it.
    mapping_schema_version:
        Version of the state-dict schema.  ``from_state_dict`` refuses
        to load mismatched versions.  Audit D12-4.
    """

    collapse_stereoisomers: bool = False
    # v42 FORENSIC ROOT FIX (P0-1): reconcile with drug_resolver._FUZZY_THRESHOLD
    # The v29 "Confidence Score Inversion" fix lowered MatchConfidence.FUZZY to
    # 0.65 and lowered the runtime gate _FUZZY_THRESHOLD to 0.60, but never
    # updated this default. The import-time sync check in
    # drug_resolver._check_module_constants_in_sync then raised RuntimeError
    # at module load, making the entire drug-resolution subsystem unusable.
    # ROOT FIX: lower the canonical default to 0.60 so it matches the runtime
    # gate. The env-var override (FUZZY_THRESHOLD) still works for operators
    # who want stricter matching.
    fuzzy_threshold: float = 0.60
    fuzzy_max_candidates: int = 10_000
    pubchem_enabled: bool = False
    pubchem_rest_base: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    pubchem_call_delay: float = 0.2
    pubchem_timeout: float = 10.0
    pubchem_max_retries: int = 3
    pubchem_api_key: Optional[str] = None
    pubchem_ca_bundle: Optional[str] = None
    pubchem_cert_pem: Optional[str] = None
    pubchem_key_pem: Optional[str] = None
    pubchem_strict_salt_form: bool = False
    source_whitelist: Optional[Tuple[str, ...]] = None
    default_organism: str = "Homo sapiens"
    mapping_schema_version: str = MAPPING_SCHEMA_VERSION
    # ----- Additive fields (audit C.7 / C.13 / C.14 / C.15 / C.16 / C.19 / C.20 / C.21 / C.24 / 7.11 / 14.4 / 14.5 / 14.7 / 14.8 / 1.4 / 6.2 / 6.3 / 6.5 / 6.11 / 8.14 / 8.15 / 8.18 / 9.16 / 11.5 / 12.18 / 14.22) -----
    # These are NEW fields added by the drug_resolver audit remediation.
    # They all have safe defaults so existing ResolverConfig constructions
    # keep working.  Each field is documented with its audit finding ID.
    deterministic_timestamps: bool = False  # audit C.7 / 7.3 / 7.4 / 7.5
    random_seed: Optional[int] = None  # audit 7.11
    runtime_asserts: bool = False  # audit C.25 / 10.2
    checksum_salt: str = ""  # audit C.8
    bulk_strict_validation: bool = False  # audit C.15 / 3.8
    dead_letter_on_soft_warning: bool = False  # audit C.15 / 6.8
    conflict_policy: str = "keep_existing"  # audit C.16
    enable_smiles_matching: bool = False  # audit 3.13
    enable_formula_matching: bool = False  # audit 3.16
    prefer_fresher_data: bool = False  # audit 5.8
    max_records_per_batch: int = 1_000_000  # audit 1.4
    max_dead_letter_size: int = 100_000  # audit 6.2 / 8.15
    max_audit_trail_per_entry: int = 1_000  # audit 8.14
    max_query_log_size: int = 10_000  # audit 11.22
    dead_letter_spill_path: Optional[str] = None  # audit 6.2
    audit_trail_spill_path: Optional[str] = None  # audit 8.14
    audit_trail_retention_days: int = 365  # audit 14.4
    pubchem_backoff_base: float = 0.2  # audit C.13 / 4.23
    pubchem_backoff_max: float = 30.0  # audit C.13
    pubchem_backoff_jitter: float = 0.25  # audit C.13 / 3.18
    pubchem_failure_threshold: int = 10  # audit C.14 / 6.3
    pubchem_circuit_cooldown: float = 60.0  # audit C.14 / 6.3
    pubchem_allowlist_hosts: Tuple[str, ...] = ("pubchem.ncbi.nlm.nih.gov",)  # audit 9.8
    pubchem_allowed_regions: Tuple[str, ...] = ("US",)  # audit 14.5
    pubchem_verify_tls: bool = True  # audit 9.9
    pubchem_insecure_acknowledgement: bool = False  # audit 9.9
    redact_dead_letter_pii: bool = False  # audit 9.13
    state_file_mode: int = 0o600  # audit 9.14
    state_encryption_key: Optional[bytes] = None  # audit 9.16 / 14.9
    allowed_paths_root: Optional[str] = None  # audit 9.15 / C.24
    spill_dir: Optional[str] = None  # audit 6.17
    data_classification: str = "internal"  # audit 14.7
    require_operator_for_sensitive_actions: bool = False  # audit 14.8
    require_organism_override: bool = False  # audit 12.20
    controlled_substance_list: Tuple[str, ...] = ()  # audit 9.25
    log_sample_rate: float = 0.01  # audit 4.17 / 11.5
    normalize_name_cache_size: int = 8192  # audit 8.18
    parallel_ingestion_workers: int = 0  # audit 8.8
    profile: Optional[str] = None  # audit 12.19
    eager_imports: bool = False  # audit 6.12
    isolated_rate_limiter: bool = False  # audit 7.12
    tamper_evident: bool = True  # audit 14.2
    # FIX P1-ER-18 (LOW): the previous implementation hard-coded the
    # HMAC key as ``b"protein-resolver-tamper-evident-key"`` directly
    # in protein_resolver.py -- anyone with source-code access could
    # forge valid signatures. This field lets operators supply a
    # deployment-specific key (via ``ENTITY_RESOLUTION_TAMPER_EVIDENT_KEY``
    # env var, hex-encoded). If ``tamper_evident=True`` but this field
    # is ``None``, the resolver logs a CRITICAL warning and skips
    # signing/verification -- tamper-evidence is effectively disabled
    # until the operator configures a key. This is safer than silently
    # using a known-to-attacker key.
    tamper_evident_key: Optional[bytes] = None
    validate_output_schema: bool = False  # audit C.17
    allow_api_key_round_trip: bool = False  # audit 7.15
    fuzzy_index_type: str = "exact"  # audit 8.1
    normalize_name_cache: bool = True  # audit 8.18

    # ----- env-var-backed factory -----

    @classmethod
    def from_env(cls, **overrides: Any) -> "ResolverConfig":
        """Build a :class:`ResolverConfig` from environment variables.

        P2-11 ROOT FIX (v82): EVERY field on the dataclass now has an
        env-var override (prefix ``ENTITY_RESOLUTION_<FIELD_NAME_UPPER>``).
        Previously, only ~17 of 50+ fields were loaded from env -- the
        rest fell back to dataclass defaults, forcing operators to
        construct the dataclass programmatically to tune anything not
        in the env-backed subset. The root fix loads ALL fields from
        env (using dataclass defaults only when the env var is unset),
        so operators can tune the entire config via env vars in
        production deployments (Docker, Kubernetes, Airflow, etc.).

        Type coercion rules (see the class docstring for the full table):
        bool / int / float / str / Optional[str] / Tuple[str, ...] /
        Optional[bytes] (hex) / octal int (state_file_mode).

        For boolean fields, ``"1"``, ``"true"``, ``"yes"``, ``"on"``
        (case-insensitive) are truthy; everything else is falsy.

        Parameters
        ----------
        **overrides:
            Explicit field values that take precedence over env vars.
            Useful in tests for reproducibility.

        Returns
        -------
        ResolverConfig
            A frozen, validated configuration.
        """
        prefix = "ENTITY_RESOLUTION_"

        def _get_bool(name: str, default: bool) -> bool:
            val = os.environ.get(prefix + name)
            if val is None:
                return default
            return val.strip().lower() in {"1", "true", "yes", "on"}

        def _get_float(name: str, default: float) -> float:
            val = os.environ.get(prefix + name)
            if val is None or val.strip() == "":
                return default
            try:
                return float(val)
            except ValueError:
                logger.warning(
                    "ResolverConfig.from_env: %s=%r is not a float, "
                    "using default %r",
                    prefix + name, val, default,
                )
                return default

        def _get_int(name: str, default: int) -> int:
            val = os.environ.get(prefix + name)
            if val is None or val.strip() == "":
                return default
            try:
                return int(val)
            except ValueError:
                logger.warning(
                    "ResolverConfig.from_env: %s=%r is not an int, "
                    "using default %r",
                    prefix + name, val, default,
                )
                return default

        def _get_str(name: str, default: Optional[str]) -> Optional[str]:
            val = os.environ.get(prefix + name)
            if val is None or val.strip() == "":
                return default
            return val

        # P2-11 v82: Optional[int] with sentinel semantics -- ``0`` means
        # ``None`` (no random seed). This lets operators disable a seed
        # via ``ENTITY_RESOLUTION_RANDOM_SEED=0`` instead of having to
        # unset the env var.
        def _get_optional_int(name: str, default: Optional[int]) -> Optional[int]:
            val = os.environ.get(prefix + name)
            if val is None or val.strip() == "":
                return default
            try:
                parsed = int(val)
            except ValueError:
                logger.warning(
                    "ResolverConfig.from_env: %s=%r is not an int, "
                    "using default %r",
                    prefix + name, val, default,
                )
                return default
            return None if parsed == 0 else parsed

        # P2-11 v82: comma-separated list -> tuple of stripped non-empty strings.
        def _get_csv_tuple(
            name: str, default: Tuple[str, ...]
        ) -> Tuple[str, ...]:
            val = os.environ.get(prefix + name)
            if val is None or val.strip() == "":
                return default
            return tuple(s.strip() for s in val.split(",") if s.strip())

        # P2-11 v82: octal int (for state_file_mode). Accepts ``0o600``,
        # ``0o640``, or plain octal digits ``600`` (interpreted as octal).
        def _get_octal_int(name: str, default: int) -> int:
            val = os.environ.get(prefix + name)
            if val is None or val.strip() == "":
                return default
            v = val.strip()
            try:
                if v.startswith("0o") or v.startswith("0O"):
                    return int(v, 8)
                # Plain digits -- interpret as octal (matches Unix chmod
                # convention: ``chmod 600 file`` interprets 600 as octal).
                if v.isdigit() and all(c in "01234567" for c in v):
                    return int(v, 8)
                # Fallback: int(val, 0) for autodetection (handles 0x...,
                # 0b..., etc. -- though these are unusual for file modes).
                return int(v, 0)
            except ValueError:
                logger.warning(
                    "ResolverConfig.from_env: %s=%r is not a valid octal "
                    "int, using default %o",
                    prefix + name, val, default,
                )
                return default

        # P2-11 v82: hex-encoded bytes (for state_encryption_key and
        # tamper_evident_key). Mirrors the existing tamper_evident_key
        # parsing logic below.
        def _get_hex_bytes(name: str) -> Optional[bytes]:
            val = os.environ.get(prefix + name)
            if val is None or val.strip() == "":
                return None
            try:
                return bytes.fromhex(val.strip())
            except ValueError:
                logger.warning(
                    "ResolverConfig.from_env: %s=%r is not valid hex -- "
                    "field will be None. Supply a hex-encoded value "
                    "(e.g. openssl rand -hex 32).",
                    prefix + name, val[:8] + "...",
                )
                return None

        # Source whitelist comes in as a comma-separated list.
        sw_raw = os.environ.get(prefix + "SOURCE_WHITELIST", "")
        source_whitelist: Optional[Tuple[str, ...]] = None
        if sw_raw.strip():
            source_whitelist = tuple(
                s.strip() for s in sw_raw.split(",") if s.strip()
            )

        # FIX P1-ER-18 (LOW): load the tamper-evident HMAC key from env.
        # (Preserved verbatim -- the new _get_hex_bytes helper centralises
        # the pattern, but the existing call site is kept for clarity.)
        tamper_key_raw = os.environ.get(prefix + "TAMPER_EVIDENT_KEY", "")
        tamper_evident_key: Optional[bytes] = None
        if tamper_key_raw.strip():
            try:
                tamper_evident_key = bytes.fromhex(tamper_key_raw.strip())
            except ValueError:
                logger.warning(
                    "ResolverConfig.from_env: %s=%r is not valid hex -- "
                    "tamper-evidence will be DISABLED. Supply a hex-encoded "
                    "key (e.g. openssl rand -hex 32).",
                    prefix + "TAMPER_EVIDENT_KEY", tamper_key_raw[:8] + "...",
                )

        # P2-11 v82 ROOT FIX: load ALL fields from env. Previously only
        # ~17 fields were loaded; the other ~40 fell back to dataclass
        # defaults. Now every field has an env-var override so operators
        # can tune the entire config in production.
        env_kwargs: Dict[str, Any] = {
            "collapse_stereoisomers": _get_bool(
                "COLLAPSE_STEREOISOMERS", False),
            "fuzzy_threshold": _get_float("FUZZY_THRESHOLD", 0.60),  # v42 P0-1
            "fuzzy_max_candidates": _get_int(
                "FUZZY_MAX_CANDIDATES", 10_000),
            "pubchem_enabled": _get_bool("PUBCHEM_ENABLED", False),
            "pubchem_rest_base": _get_str(
                "PUBCHEM_REST_BASE",
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug"),
            "pubchem_call_delay": _get_float("PUBCHEM_CALL_DELAY", 0.2),
            "pubchem_timeout": _get_float("PUBCHEM_TIMEOUT", 10.0),
            "pubchem_max_retries": _get_int("PUBCHEM_MAX_RETRIES", 3),
            "pubchem_api_key": _get_str("PUBCHEM_API_KEY", None),
            "pubchem_ca_bundle": _get_str("PUBCHEM_CA_BUNDLE", None),
            "pubchem_cert_pem": _get_str("PUBCHEM_CERT_PEM", None),
            "pubchem_key_pem": _get_str("PUBCHEM_KEY_PEM", None),
            "pubchem_strict_salt_form": _get_bool(
                "PUBCHEM_STRICT_SALT_FORM", False),
            "source_whitelist": source_whitelist,
            "default_organism": _get_str(
                "DEFAULT_ORGANISM", "Homo sapiens"),
            "mapping_schema_version": MAPPING_SCHEMA_VERSION,
            "tamper_evident_key": tamper_evident_key,

            # ----- P2-11 v82 ROOT FIX: newly env-backed fields -----
            "deterministic_timestamps": _get_bool(
                "DETERMINISTIC_TIMESTAMPS", False),
            "random_seed": _get_optional_int("RANDOM_SEED", None),
            "runtime_asserts": _get_bool("RUNTIME_ASSERTS", False),
            "checksum_salt": _get_str("CHECKSUM_SALT", ""),
            "bulk_strict_validation": _get_bool(
                "BULK_STRICT_VALIDATION", False),
            "dead_letter_on_soft_warning": _get_bool(
                "DEAD_LETTER_ON_SOFT_WARNING", False),
            "conflict_policy": _get_str("CONFLICT_POLICY", "keep_existing"),
            "enable_smiles_matching": _get_bool(
                "ENABLE_SMILES_MATCHING", False),
            "enable_formula_matching": _get_bool(
                "ENABLE_FORMULA_MATCHING", False),
            "prefer_fresher_data": _get_bool("PREFER_FRESHER_DATA", False),
            "max_records_per_batch": _get_int(
                "MAX_RECORDS_PER_BATCH", 1_000_000),
            "max_dead_letter_size": _get_int("MAX_DEAD_LETTER_SIZE", 100_000),
            "max_audit_trail_per_entry": _get_int(
                "MAX_AUDIT_TRAIL_PER_ENTRY", 1_000),
            "max_query_log_size": _get_int("MAX_QUERY_LOG_SIZE", 10_000),
            "dead_letter_spill_path": _get_str("DEAD_LETTER_SPILL_PATH", None),
            "audit_trail_spill_path": _get_str(
                "AUDIT_TRAIL_SPILL_PATH", None),
            "audit_trail_retention_days": _get_int(
                "AUDIT_TRAIL_RETENTION_DAYS", 365),
            "pubchem_backoff_base": _get_float("PUBCHEM_BACKOFF_BASE", 0.2),
            "pubchem_backoff_max": _get_float("PUBCHEM_BACKOFF_MAX", 30.0),
            "pubchem_backoff_jitter": _get_float(
                "PUBCHEM_BACKOFF_JITTER", 0.25),
            "pubchem_failure_threshold": _get_int(
                "PUBCHEM_FAILURE_THRESHOLD", 10),
            "pubchem_circuit_cooldown": _get_float(
                "PUBCHEM_CIRCUIT_COOLDOWN", 60.0),
            "pubchem_allowlist_hosts": _get_csv_tuple(
                "PUBCHEM_ALLOWLIST_HOSTS",
                ("pubchem.ncbi.nlm.nih.gov",)),
            "pubchem_allowed_regions": _get_csv_tuple(
                "PUBCHEM_ALLOWED_REGIONS", ("US",)),
            "pubchem_verify_tls": _get_bool("PUBCHEM_VERIFY_TLS", True),
            "pubchem_insecure_acknowledgement": _get_bool(
                "PUBCHEM_INSECURE_ACKNOWLEDGEMENT", False),
            "redact_dead_letter_pii": _get_bool(
                "REDACT_DEAD_LETTER_PII", False),
            "state_file_mode": _get_octal_int("STATE_FILE_MODE", 0o600),
            "state_encryption_key": _get_hex_bytes("STATE_ENCRYPTION_KEY"),
            "allowed_paths_root": _get_str("ALLOWED_PATHS_ROOT", None),
            "spill_dir": _get_str("SPILL_DIR", None),
            "data_classification": _get_str(
                "DATA_CLASSIFICATION", "internal"),
            "require_operator_for_sensitive_actions": _get_bool(
                "REQUIRE_OPERATOR_FOR_SENSITIVE_ACTIONS", False),
            "require_organism_override": _get_bool(
                "REQUIRE_ORGANISM_OVERRIDE", False),
            "controlled_substance_list": _get_csv_tuple(
                "CONTROLLED_SUBSTANCE_LIST", ()),
            "log_sample_rate": _get_float("LOG_SAMPLE_RATE", 0.01),
            "normalize_name_cache_size": _get_int(
                "NORMALIZE_NAME_CACHE_SIZE", 8192),
            "parallel_ingestion_workers": _get_int(
                "PARALLEL_INGESTION_WORKERS", 0),
            "profile": _get_str("PROFILE", None),
            "eager_imports": _get_bool("EAGER_IMPORTS", False),
            "isolated_rate_limiter": _get_bool(
                "ISOLATED_RATE_LIMITER", False),
            "tamper_evident": _get_bool("TAMPER_EVIDENT", True),
            "validate_output_schema": _get_bool(
                "VALIDATE_OUTPUT_SCHEMA", False),
            "allow_api_key_round_trip": _get_bool(
                "ALLOW_API_KEY_ROUND_TRIP", False),
            "fuzzy_index_type": _get_str("FUZZY_INDEX_TYPE", "exact"),
            "normalize_name_cache": _get_bool("NORMALIZE_NAME_CACHE", True),
        }
        # When an API key is present, double the allowed rate.
        if env_kwargs["pubchem_api_key"] and "pubchem_call_delay" not in overrides:
            env_kwargs["pubchem_call_delay"] = max(
                0.1, env_kwargs["pubchem_call_delay"] / 2.0
            )

        env_kwargs.update(overrides)
        cfg = cls(**env_kwargs)
        cfg.validate()
        return cfg

    # ----- validation -----

    def validate(self) -> None:
        """Validate the configuration; raise on impossible combinations.

        Raises
        ------
        ValueError
            If any field has an out-of-range value or an impossible
            combination (e.g. ``pubchem_enabled=True`` but the
            ``requests`` library is not installed).
        """
        if not 0.0 <= self.fuzzy_threshold <= 1.0:
            raise ValueError(
                f"fuzzy_threshold must be in [0, 1], got {self.fuzzy_threshold}"
            )
        if self.fuzzy_max_candidates < 1:
            raise ValueError(
                f"fuzzy_max_candidates must be >= 1, "
                f"got {self.fuzzy_max_candidates}"
            )
        if self.pubchem_call_delay < 0:
            raise ValueError(
                f"pubchem_call_delay must be >= 0, "
                f"got {self.pubchem_call_delay}"
            )
        if self.pubchem_timeout <= 0:
            raise ValueError(
                f"pubchem_timeout must be > 0, got {self.pubchem_timeout}"
            )
        if self.pubchem_max_retries < 0:
            raise ValueError(
                f"pubchem_max_retries must be >= 0, "
                f"got {self.pubchem_max_retries}"
            )
        if not self.default_organism or not self.default_organism.strip():
            raise ValueError("default_organism must be a non-empty string")
        if not self.mapping_schema_version:
            raise ValueError("mapping_schema_version must be non-empty")

        # If PubChem is enabled, ``requests`` must be importable.  We
        # check here rather than at construction so that air-gapped
        # deployments that never enable PubChem don't need ``requests``.
        if self.pubchem_enabled:
            try:
                import requests  # noqa: F401
            except ImportError as exc:
                raise ValueError(
                    "pubchem_enabled=True requires the 'requests' library "
                    "(install with: pip install requests)"
                ) from exc

        # Source whitelist entries must be non-empty strings.
        if self.source_whitelist is not None:
            for s in self.source_whitelist:
                if not isinstance(s, str) or not s.strip():
                    raise ValueError(
                        f"source_whitelist entries must be non-empty strings, "
                        f"got {s!r}"
                    )

    # ----- introspection -----

    def to_masked_dict(self) -> Dict[str, Any]:
        """Return a credential-masked dict suitable for logging.

        ``pubchem_api_key`` is replaced with ``"<redacted>"`` if set,
        ``None`` otherwise.  Every other field is logged verbatim.
        """
        d = dataclasses.asdict(self)
        if d.get("pubchem_api_key"):
            d["pubchem_api_key"] = "<redacted>"
        return d


# ---------------------------------------------------------------------------
# ResolverStats (D11-2)
# ---------------------------------------------------------------------------


@dataclass
class ResolverStats:
    """Mutable counter container exposed via :meth:`Resolver.get_stats`.

    Every counter is initialised to zero and incremented atomically via
    a :class:`threading.Lock` so that concurrent ``add_source_records``
    calls don't lose updates.  A snapshot is returned as a plain dict
    by :meth:`to_dict` for serialisation.

    The internal ``_lock`` is set in ``__post_init__`` and is **not**
    a dataclass field, so :func:`dataclasses.asdict` and
    :func:`json.dumps` can serialise the dataclass safely (locks can't
    be pickled).
    """

    records_ingested: int = 0
    records_matched: int = 0
    records_created: int = 0
    records_rejected: int = 0
    fuzzy_matches: int = 0
    connectivity_matches: int = 0
    name_matches: int = 0
    inchikey_exact_matches: int = 0
    # v82 FORENSIC ROOT FIX (P1-2): dedicated counter for UniProt exact
    # matches in ProteinResolver. The previous code incremented
    # ``inchikey_exact_matches`` (a DRUG metric) for protein UniProt
    # matches, conflating drug and protein metrics on operator dashboards.
    uniprot_exact_matches: int = 0
    pubchem_calls: int = 0
    pubchem_successes: int = 0
    pubchem_failures: int = 0
    pubchem_dead_lettered: int = 0
    stereoisomer_collapses: int = 0
    synthetic_keys_generated: int = 0
    duplicate_ids_detected: int = 0
    dead_lettered: int = 0
    # FIX P1-ER-20 (LOW): track synthetic-UID collisions so operators
    # can monitor data-quality issues (two different source records
    # providing the same raw_id and silently colliding on the same
    # synthetic UID). Incremented by
    # ``ProteinResolver._make_synthetic_uid_checked``.
    synthetic_uid_collisions: int = 0
    # FIX-P1-C-13: previously ``inc()`` was called with these names but
    # they were NOT declared as dataclass fields, so ``to_dict()`` (which
    # iterates ``dataclasses.fields``) silently dropped them from every
    # stats snapshot -- 9+ drug_resolver metrics and 4 protein_resolver
    # metrics were invisible to operators. Declaring them as fields makes
    # ``to_dict()`` include them.
    # --- drug_resolver metrics ---
    smiles_matches: int = 0
    pubchem_xref_matches: int = 0
    no_match_results: int = 0
    salt_forms_rejected: int = 0
    merge_conflicts_detected: int = 0
    mutations_rolled_back: int = 0
    index_mapping_desync: int = 0
    synthetic_key_collisions_resolved: int = 0
    soft_validation_warnings: int = 0
    # --- protein_resolver metrics ---
    cross_species_fuzzy_matches: int = 0
    ingestion_exceptions: int = 0
    organism_mismatches: int = 0
    records_promoted: int = 0
    # P1-D2: separate stat for UniProt ID exact matches in ProteinResolver.
    # Previously, resolve_single incremented inchikey_exact_matches for
    # UniProt exact matches, conflating drug (InChIKey) and protein
    # (UniProt) metrics.
    uniprot_exact_matches: int = 0

    def __post_init__(self) -> None:
        # Set the lock via object.__setattr__ to bypass any property
        # setter; the lock is NOT a dataclass field, so asdict/JSON
        # serialisation won't try to pickle it.
        object.__setattr__(self, "_lock", threading.Lock())

    def inc(self, field_name: str, amount: int = 1) -> None:
        """Atomically increment a named counter by ``amount``.

        Parameters
        ----------
        field_name:
            Name of the counter attribute to increment (must exist).
        amount:
            Integer delta (default 1; pass -1 to decrement).
        """
        lock = object.__getattribute__(self, "_lock")
        with lock:
            current = getattr(self, field_name, 0)
            setattr(self, field_name, current + amount)

    def to_dict(self) -> Dict[str, int]:
        """Return a JSON-serialisable snapshot of all counters.

        The internal ``_lock`` is excluded from the output.
        """
        lock = object.__getattribute__(self, "_lock")
        with lock:
            return {
                f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)
                if not f.name.startswith("_")
            }

    def reset(self) -> None:
        """Reset every counter to zero (called by :meth:`Resolver.reset`)."""
        lock = object.__getattribute__(self, "_lock")
        with lock:
            for f in dataclasses.fields(self):
                if f.name.startswith("_"):
                    continue
                setattr(self, f.name, 0)


# ---------------------------------------------------------------------------
# Resolver ABC (D1-3)
# ---------------------------------------------------------------------------


class Resolver(abc.ABC):
    """Abstract base class for every concrete entity resolver.

    The ABC pins the **public method surface** that callers can rely
    on regardless of whether they hold a :class:`DrugResolver` or a
    :class:`ProteinResolver`.  Concrete resolvers must implement every
    abstract method; the non-abstract helpers (``to_json``,
    ``from_json``, ``get_audit_trail``) are provided as default
    implementations that build on the abstract ones.

    The ABC intentionally does NOT enforce a common ``mapping``
    attribute type -- drugs are keyed by InChIKey, proteins by UniProt
    accession, and forcing them into a single generic signature would
    erase scientifically-meaningful distinctions (see master fix
    prompt §11 anti-pattern #2).

    Concrete subclasses are expected to expose three **instance
    attributes** (not abstract methods, because they are plain data
    attributes set in ``__init__``):

    * ``config`` -- a :class:`ResolverConfig` instance.
    * ``stats`` -- a :class:`ResolverStats` instance.
    * ``mapping`` -- a ``Dict[str, dict]`` mapping canonical key to
      canonical entry.
    """

    #: Schema version of the state-dict format produced by subclasses.
    MAPPING_SCHEMA_VERSION: ClassVar[str] = MAPPING_SCHEMA_VERSION

    # ----- ingestion -----

    @abc.abstractmethod
    def add_source_records(self, records: List[dict], source: str) -> None:
        """Ingest ``records`` labelled with ``source`` into the resolver."""

    # ----- single-record resolution -----

    @abc.abstractmethod
    def resolve_single(self, **kwargs: Any) -> Optional[dict]:
        """Resolve a single entity, returning the canonical record or ``None``."""

    # ----- bulk export -----

    @abc.abstractmethod
    def build_mapping(self, **kwargs: Any) -> Any:
        """Build the cross-source mapping and return a DataFrame."""

    @abc.abstractmethod
    def to_dataframe(self, chunksize: Optional[int] = None) -> Any:
        """Export the mapping as a DataFrame (optionally chunked)."""

    @abc.abstractmethod
    def to_records(self) -> List[dict]:
        """Export the mapping as a list of plain dicts (no pandas dependency)."""

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, dict]:
        """Export the mapping as a dict-of-dicts (JSON-serialisable)."""

    # ----- state serialisation (D7-4, D16-3) -----

    @abc.abstractmethod
    def to_state_dict(self) -> dict:
        """Serialise the resolver's full state to a JSON-compatible dict."""

    @classmethod
    @abc.abstractmethod
    def from_state_dict(cls, state: dict) -> "Resolver":
        """Reconstruct a resolver from a :meth:`to_state_dict` output."""

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialise the resolver's state to JSON.

        Parameters
        ----------
        path:
            If given, write JSON to that file path.  Otherwise return
            the JSON string.

        Returns
        -------
        str
            The JSON serialisation (also written to ``path`` if given).
        """
        state = self.to_state_dict()
        text = json.dumps(state, indent=2, sort_keys=True, default=str)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_json(cls, path_or_text: str) -> "Resolver":
        """Reconstruct a resolver from JSON file path or text.

        Parameters
        ----------
        path_or_text:
            Either a path to a JSON file or a JSON string.
        """
        text: str
        if path_or_text.lstrip().startswith("{"):
            text = path_or_text
        else:
            with open(path_or_text, "r", encoding="utf-8") as fh:
                text = fh.read()
        state = json.loads(text)
        return cls.from_state_dict(state)

    # ----- lifecycle / maintenance -----

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear all internal state -- equivalent to a fresh instance."""

    @abc.abstractmethod
    def remove_source(self, source: str) -> int:
        """Remove every entry whose only source is ``source``.

        Returns the number of entries removed.  Entries contributed by
        multiple sources are kept but have ``source`` removed from
        their ``sources`` list.
        """

    @abc.abstractmethod
    def get_stats(self) -> Dict[str, int]:
        """Return a JSON-serialisable snapshot of resolver counters."""

    @abc.abstractmethod
    def get_audit_trail(self, canonical_id: str) -> List[dict]:
        """Return the ordered list of merge events for ``canonical_id``."""

    @abc.abstractmethod
    def find_affected_entities(self, source: str) -> List[str]:
        """Return canonical IDs whose ``sources`` list contains ``source``."""


# ---------------------------------------------------------------------------
# Process-global rate limiter (D6-6)
# ---------------------------------------------------------------------------


class _ProcessGlobalRateLimiter:
    """Token-bucket rate limiter shared across all instances in a process.

    The previous per-instance rate limiter meant that two
    ``DrugResolver`` instances in the same Airflow worker would each
    independently sleep ``pubchem_call_delay`` seconds -- but their
    HTTP requests would interleave at twice the configured rate,
    violating PubChem's "5 req/sec" limit and risking IP bans.

    This limiter is keyed by ``(base_url, delay)`` so that resolvers
    pointing at different mirrors (audit D9-3) maintain independent
    buckets.
    """

    _buckets: ClassVar[Dict[Tuple[str, float], "_Bucket"]] = {}
    _class_lock: ClassVar[threading.Lock] = threading.Lock()

    class _Bucket:
        def __init__(self, delay: float) -> None:
            self.delay = delay
            self.last_call: float = 0.0
            self.lock = threading.Lock()

        def wait(self) -> None:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_call
                if elapsed < self.delay:
                    time.sleep(self.delay - elapsed)
                self.last_call = time.monotonic()

    @classmethod
    def acquire(cls, base_url: str, delay: float) -> None:
        """Block until a PubChem call to ``base_url`` is permitted.

        Parameters
        ----------
        base_url:
            PubChem (or mirror) base URL.
        delay:
            Minimum seconds since the last call to the same base URL.
        """
        key = (base_url, delay)
        with cls._class_lock:
            bucket = cls._buckets.get(key)
            if bucket is None or bucket.delay != delay:
                bucket = cls._Bucket(delay)
                cls._buckets[key] = bucket
        bucket.wait()

    @classmethod
    def _reset_for_tests(cls) -> None:
        """Clear all buckets -- used by the test suite for isolation."""
        with cls._class_lock:
            cls._buckets.clear()


# ---------------------------------------------------------------------------
# Constants exported for reuse by sibling modules
# ---------------------------------------------------------------------------

#: InChIKey format: 14 uppercase letters, hyphen, 10 uppercase letters,
#: hyphen, 1 uppercase letter, optionally followed by a hyphen + protonation
#: suffix (PubChem / ChEMBL legitimately emit ``-X`` suffixes for tautomeric
#: / salt-form records per the IUPAC InChIKey extension). Compiled once at
#: import for O(1) checks.
#:
#: P1-ER-3 ROOT FIX: pattern synchronized with normalizer.py / base.py /
#: models.py -- DO NOT diverge (audit P1-ER-3).
#:
#: v35 ROOT FIX (issue 38): import the canonical InChIKey regex from
#: ``cleaning._constants`` (single source of truth) instead of defining
#: it locally. The local definition is kept ONLY as a fallback for test
#: isolation / partial installs.
try:
    from cleaning._constants import (
        CANONICAL_INCHIKEY_REGEX as _CANONICAL_INCHIKEY_RE,
    )
except ImportError:
    # Fallback (test isolation): replicate the canonical pattern EXACTLY.
    _CANONICAL_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")

# Backward-compat aliases (the permissive ``INCHIKEY_PATTERN`` historically
# accepted an optional ``-X`` protonation suffix; the strict pattern does
# not). The permissive form is retained for any caller that still imports
# ``INCHIKEY_PATTERN`` directly -- but its use is deprecated; new code
# should use ``cleaning._constants.is_canonical_inchikey`` instead.
INCHIKEY_PATTERN: re.Pattern[str] = re.compile(
    r"^[A-Z]{14}-[A-Z]{10}-[A-Z](?:-[A-Za-z0-9]+)?$"
)

#: STRICT InChIKey format: exactly 27 chars, NO protonation suffix.
#: Used by :func:`is_strict_inchikey` (which is the DB write-boundary
#: validator that REJECTS suffixed keys).
#:
#: P1-ER-3 ROOT FIX: kept separate from :data:`INCHIKEY_PATTERN` so that
#: ``is_strict_inchikey`` can continue to reject suffixed keys at the DB
#: boundary while the permissive pattern is used everywhere else.
#: v35 ROOT FIX (issue 38): now delegates to the canonical regex imported
#: from ``cleaning._constants`` above.
_STRICT_INCHIKEY_PATTERN: re.Pattern[str] = _CANONICAL_INCHIKEY_RE

#: Prefix used to mark InChIKeys that were synthesised by the resolver
#: because the source record had no real InChIKey.  Audit D3-5, D13-7.
SYNTHETIC_INCHIKEY_PREFIX: str = "SYNTH"


def is_valid_inchikey(inchikey: Any) -> bool:
    """Return ``True`` iff *inchikey* matches the canonical InChIKey format.

    An InChIKey is ``[A-Z]{14}-[A-Z]{10}-[A-Z]`` (27 chars total).  This
    helper is exposed publicly via :mod:`entity_resolution.__init__` so
    that downstream code can validate user-supplied InChIKeys at API
    boundaries (audit D3-8).

    v16 ROOT FIX (CD-6): there were THREE definitions of
    ``is_valid_inchikey`` in the codebase with OPPOSITE behaviors:

      1. ``cleaning.normalizer.is_valid_inchikey`` -- PERMISSIVE
         (accepts SYNTH-prefixed and mixture keys).
      2. ``entity_resolution.base.is_valid_inchikey`` (this function) --
         STRICT (only standard 27-char pattern).
      3. ``entity_resolution.resolver_utils.is_valid_inchikey`` --
         DELEGATES to (1).

    So ``base.is_valid_inchikey("SYNTH-001")`` returned False while
    ``normalizer.is_valid_inchikey("SYNTH-001")`` returned True. Calls
    that used (2) rejected synthetic keys; calls that used (1) accepted
    them. Same name, opposite semantics -- silent corruption.

    The fix: ``base.is_valid_inchikey`` now delegates to
    ``cleaning.normalizer.is_valid_inchikey`` so the same name has ONE
    meaning everywhere. Callers that need STRICT validation (no SYNTH,
    no mixtures) should call ``is_strict_inchikey`` (new function below).
    """
    # v16 CD-6: delegate to the permissive validator for behavioral
    # parity with normalizer + resolver_utils. Use a lazy import to
    # avoid a circular dependency (cleaning.normalizer imports from
    # entity_resolution.base for INCHIKEY_PATTERN).
    try:
        from cleaning.normalizer import is_valid_inchikey as _normalizer_is_valid
        return _normalizer_is_valid(inchikey)
    except ImportError:
        # Fallback: strict pattern (no SYNTH, no mixtures).
        # v65 ROOT FIX (P1C-004): the previous fallback used
        # ``INCHIKEY_PATTERN`` (the PERMISSIVE pattern that accepts an
        # optional ``-X`` protonation suffix via ``(?:-[A-Za-z0-9]+)?``).
        # This meant the SAME InChIKey (e.g. ``BSYNRYMUTXBXSQ-UHFFFAOYSA-N-a``)
        # was REJECTED when ``cleaning.normalizer`` was importable but
        # ACCEPTED when it was not (test isolation / partial install).
        # The comment above said "strict pattern (no SYNTH, no mixtures)"
        # but ``INCHIKEY_PATTERN`` is NOT strict -- it accepts suffixed keys.
        # ROOT FIX: use ``_STRICT_INCHIKEY_PATTERN`` (which delegates to
        # the canonical 27-char regex from ``cleaning._constants``) so the
        # fallback matches the canonical validator's behavior exactly.
        if not isinstance(inchikey, str):
            return False
        return bool(_STRICT_INCHIKEY_PATTERN.match(inchikey))


def is_strict_inchikey(inchikey: Any) -> bool:
    """Return ``True`` iff *inchikey* matches the STRICT canonical InChIKey format.

    v16 CD-6: this is the STRICT version of ``is_valid_inchikey``.
    Use this when you want to REJECT synthetic and mixture keys (e.g.
    at the API boundary, or when filtering records that MUST have a
    real InChIKey for downstream graph construction).

    Accepts ONLY the standard 27-char pattern ``[A-Z]{14}-[A-Z]{10}-[A-Z]``.
    Rejects SYNTH-prefixed keys, mixture keys, AND keys with an
    optional protonation suffix (e.g. ``...-N-a``) -- that suffix is
    legitimate at the resolver level but MUST NOT be written to the DB
    canonical key column (audit P1-ER-3 / P1-ER-7).

    P1-ER-3 ROOT FIX: uses :data:`_STRICT_INCHIKEY_PATTERN` (no suffix)
    rather than :data:`INCHIKEY_PATTERN` (which now accepts the suffix).
    """
    if not isinstance(inchikey, str):
        return False
    return bool(_STRICT_INCHIKEY_PATTERN.match(inchikey))


def is_synthetic_inchikey(inchikey: Any) -> bool:
    """Return ``True`` iff *inchikey* was synthesised by the resolver.

    Synthetic keys carry the :data:`SYNTHETIC_INCHIKEY_PREFIX` (``"SYNTH"``)
    in their first 5 characters, signalling that the underlying molecule
    has no publicly-known InChIKey and the resolver invented a stable
    surrogate so the record could still be tracked.

    v35 ROOT FIX (issue 37): the check is now CASE-INSENSITIVE (calls
    ``.upper()`` on the input before comparing). Previously, a lowercase
    ``synthabc...`` or mixed-case ``SynthABC...`` key would NOT be
    detected as synthetic, even though ``make_synthetic_inchikey``
    always emits uppercase SYNTH (because the SHA-256 digest is upper-
    cased). Real-world data sources occasionally emit lowercase or
    mixed-case keys (e.g. CSV imports from case-insensitive filesystems),
    and those would silently fail the synthetic-key detection --
    defeating the resolver's purpose. The case-insensitive check aligns
    with ``cleaning._constants.CANONICAL_SYNTHETIC_INCHIKEY_REGEX``
    (which uses ``re.IGNORECASE``).

    Parameters
    ----------
    inchikey:
        Anything -- non-strings return ``False``.

    Returns
    -------
    bool
    """
    if not isinstance(inchikey, str):
        return False
    return inchikey.upper().startswith(SYNTHETIC_INCHIKEY_PREFIX)


def make_synthetic_inchikey(
    normalized_name: str,
    salt: str = "",
) -> str:
    """Generate a source-INDEPENDENT synthetic InChIKey.

    Audit D3-5 fixed the bug where synthetic keys were
    ``sha256(name:source)`` -- that meant the same InChIKey-less drug
    from ChEMBL vs DrugBank got two different synthetic keys and was
    never merged, defeating the resolver's purpose.  The new scheme
    hashes **only** the normalized name (plus an optional ``salt``
    used to disambiguate true collisions), so the same molecule from
    any source gets the same synthetic key.

    The synthetic key preserves the InChIKey *shape*
    (``[A-Z]{14}-[A-Z]{10}-[A-Z]``) so downstream code that pattern-
    matches InChIKeys keeps working.  The first 5 characters are
    ``"SYNTH"`` so :func:`is_synthetic_inchikey` can detect it.

    Parameters
    ----------
    normalized_name:
        Already-normalized name (use :func:`normalize_name` first).
    salt:
        Optional disambiguator appended to the name before hashing.
        Used when two distinct molecules share a normalized name
        (e.g. "cyclophosphamide" the brand vs "cyclophosphamide" the
        generic -- extremely rare but possible).

    Returns
    -------
    str
        A 27-char synthetic key starting with ``"SYNTH"``.
    """
    # v66 ROOT FIX (P1C-019 -- whitespace stripping causes synthetic-key collisions):
    #   The previous regex ``[\x00-\x20\x7f]`` stripped ALL control chars
    #   AND the space character (\x20). Two distinct biologics whose
    #   ``normalized_name`` was empty (``""``) or whitespace-only
    #   (``" "``) BOTH collapsed to ``clean_name = ""`` -> fell back to
    #   the literal string ``"unknown"`` -> produced the IDENTICAL
    #   synthetic InChIKey -> were silently merged into one canonical
    #   entry. The collision tracker only fires for non-synthetic keys,
    #   so this was undetectable downstream.
    #
    #   ROOT FIX (two layers):
    #     1. Narrow the regex to ``[\x00-\x1f\x7f]`` (control chars only,
    #        EXCLUDING \x20/space). A name like ``" "`` now survives as
    #        ``" "`` (distinct from ``""``), so its SHA-256 digest is
    #        unique rather than colliding with every other empty name.
    #     2. For TRULY empty names (``""`` after the narrowed regex),
    #        raise ``ValueError`` instead of silently producing a
    #        colliding ``"unknown"`` key. Callers MUST handle empty
    #        names explicitly (quarantine the record, or use a source+
    #        record-index disambiguator) rather than letting distinct
    #        biologics share one synthetic key. The previous ``or
    #        "unknown"`` fallback was the ROOT CAUSE of the collision.
    clean_name = re.sub(r"[\x00-\x1f\x7f]", "", str(normalized_name))
    if not clean_name or not clean_name.strip():
        # Empty or whitespace-only name -- refuse to mint a colliding
        # synthetic key. The caller must quarantine this record or
        # supply a disambiguating salt (e.g. source + record_index).
        raise ValueError(
            "make_synthetic_inchikey: normalized_name is empty or "
            "whitespace-only -- refusing to generate a colliding "
            "synthetic InChIKey. Quarantine the record or supply a "
            "disambiguating salt. (P1C-019 root fix)"
        )
    clean_salt = re.sub(r"[\x00-\x1f\x7f]", "", str(salt))
    payload = f"{clean_name}|{clean_salt}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # FIX-P1-C-8: SHA-256 hex chars include 0-9 which violate the canonical
    # InChIKey regex ``^[A-Z]{14}-[A-Z]{10}-[A-Z]$``. We convert the digest
    # to an integer and base-26 encode it (A-Z) so every char is a letter.
    # We need 20 base-26 chars (9 for block1 after the SYNTH prefix +
    # 10 for block2 + 1 for block3); 20 chars encode ~2e+28 values, well
    # within the 2**256 digest range.
    digest_int = int.from_bytes(digest, "big")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base26_chars: list[str] = []
    for _ in range(20):
        digest_int, rem = divmod(digest_int, 26)
        base26_chars.append(alphabet[rem])
    # ``base26_chars`` is LSB-first; reverse so the most-significant char
    # is first (cosmetic -- both directions are equally valid keys, but
    # MSB-first matches the hex-slice convention it replaces).
    base26_str = "".join(reversed(base26_chars))
    block1 = SYNTHETIC_INCHIKEY_PREFIX + base26_str[0:9]    # 14 chars (5 prefix + 9)
    block2 = base26_str[9:19]                               # 10 chars
    block3 = base26_str[19]                                 # 1 char
    return f"{block1}-{block2}-{block3}"
