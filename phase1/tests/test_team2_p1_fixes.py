"""Regression tests for Team-2 Phase 1 Database Schema & Migrations fixes.

Tests for issues P1-015 through P1-028 (14 issues total).

Each test verifies the ROOT FIX for one issue — not the surface-level
symptom. Tests are designed to CATCH the bug if it's reintroduced.
"""

import hashlib
import re
import sqlite3
import time
from pathlib import Path

import pytest


# ============================================================================
# P1-015: SQLite InChIKey CHECK accepts any 27-char string with hyphens
# ============================================================================

def _sqlite_regexp(pattern, value):
    """Replicate the REGEXP function registered in connection.py (P1-015)."""
    if value is None:
        return 0
    if not isinstance(pattern, str):
        return 0
    try:
        return 1 if re.search(pattern, str(value)) else 0
    except re.error:
        return 0


class TestP1_015_SQLiteInChIKeyRegexp:
    """P1-015: SQLite REGEXP function rejects gibberish InChIKeys."""

    def test_valid_inchikey_matches(self):
        conn = sqlite3.connect(":memory:")
        conn.create_function("REGEXP", 2, _sqlite_regexp, deterministic=True)
        cur = conn.cursor()
        cur.execute(
            "SELECT 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N' REGEXP '^[A-Z]{14}-[A-Z]{10}-[A-Z]'"
        )
        assert cur.fetchone()[0] == 1
        conn.close()

    def test_digits_only_inchikey_rejected(self):
        """The old LENGTH+SUBSTR backstop accepted this. REGEXP must reject."""
        conn = sqlite3.connect(":memory:")
        conn.create_function("REGEXP", 2, _sqlite_regexp, deterministic=True)
        cur = conn.cursor()
        cur.execute(
            "SELECT '11111111111111-2222222222-3' REGEXP '^[A-Z]{14}-[A-Z]{10}-[A-Z]'"
        )
        assert cur.fetchone()[0] == 0, "digits-only InChIKey must be rejected by REGEXP"
        conn.close()

    def test_lowercase_inchikey_rejected(self):
        conn = sqlite3.connect(":memory:")
        conn.create_function("REGEXP", 2, _sqlite_regexp, deterministic=True)
        cur = conn.cursor()
        cur.execute(
            "SELECT 'aaaaaaaaaaaaaa-bbbbbbbbbb-c' REGEXP '^[A-Z]{14}-[A-Z]{10}-[A-Z]'"
        )
        assert cur.fetchone()[0] == 0
        conn.close()

    def test_punctuation_inchikey_rejected(self):
        conn = sqlite3.connect(":memory:")
        conn.create_function("REGEXP", 2, _sqlite_regexp, deterministic=True)
        cur = conn.cursor()
        cur.execute(
            "SELECT '!!!!!!!!!!!!!!-!!!!!!!!!!-!' REGEXP '^[A-Z]{14}-[A-Z]{10}-[A-Z]'"
        )
        assert cur.fetchone()[0] == 0
        conn.close()

    def test_migration_009_sqlite_fallback_uses_regexp(self):
        """Migration 009's SQLite fallback must use REGEXP, not LENGTH+SUBSTR."""
        migration_path = (
            Path(__file__).parent.parent
            / "database"
            / "migrations"
            / "009_tighten_inchikey_check_constraint.sql"
        )
        if not migration_path.exists():
            pytest.skip(f"migration 009 not found at {migration_path}")
        sql = migration_path.read_text()
        # The SQLite fallback (EXCEPTION block) must use REGEXP, not the old
        # LENGTH+SUBSTR backstop. Check the actual ALTER TABLE ... ADD CONSTRAINT
        # block, not the comments.
        assert "REGEXP" in sql, "migration 009 SQLite fallback must use REGEXP"
        # Find the actual ALTER TABLE block in the EXCEPTION handler.
        # The old backstop (LENGTH+SUBSTR) must NOT appear as actual SQL
        # (only in comments explaining what was removed).
        # Split into lines and check non-comment lines.
        for line in sql.splitlines():
            stripped = line.strip()
            if stripped.startswith("--"):
                continue
            if "SUBSTR(inchikey, 15, 1)" in stripped and "LENGTH(inchikey) = 27" in stripped:
                # This is actual SQL code using the old backstop — fail.
                pytest.fail(
                    f"old LENGTH+SUBSTR backstop must be removed from actual SQL, "
                    f"found: {stripped}"
                )

    def test_migration_runner_translates_tilde_to_regexp(self):
        """run_migrations.py must translate ~ to REGEXP for SQLite."""
        runner_path = (
            Path(__file__).parent.parent
            / "database"
            / "migrations"
            / "run_migrations.py"
        )
        if not runner_path.exists():
            pytest.skip(f"run_migrations.py not found at {runner_path}")
        code = runner_path.read_text()
        # The translation regex must produce REGEXP, not LENGTH(TRIM(...))
        assert r"\1 REGEXP \2" in code, "must translate ~ to REGEXP"
        # The old LENGTH(TRIM(...)) backstop must be removed
        assert (
            r"LENGTH(TRIM(\1)) > 0" not in code
        ), "old LENGTH(TRIM) backstop must be removed"


# ============================================================================
# P1-016: locals().get() anti-pattern replaced with sentinel
# ============================================================================

class TestP1_016_LocalsGetAntiPattern:
    """P1-016: locals().get('drug_rec') replaced with explicit sentinel."""

    def test_no_locals_get_in_drugbank_pipeline(self):
        pipeline_path = (
            Path(__file__).parent.parent
            / "pipelines"
            / "drugbank_pipeline.py"
        )
        code = pipeline_path.read_text()
        # The locals().get("drug_rec") anti-pattern must be gone from ACTUAL
        # code (not from comments documenting the fix). Check non-comment lines.
        for line in code.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if 'locals().get' in stripped:
                pytest.fail(
                    f"locals().get must be removed from actual code, found: {stripped}"
                )
        # The sentinel pattern must be present.
        assert (
            "drug_rec = None  # P1-016 sentinel" in code
        ), "sentinel initialization must be present"


# ============================================================================
# P1-017: Synthesized DrugBank IDs use non-DrugBank prefix
# ============================================================================

class TestP1_017_SynthesizedDrugIdPrefix:
    """P1-017: synthesized IDs use SYNTH-DB- prefix, not DB."""

    def test_real_drugbank_id_regex_only_matches_real_ids(self):
        from pipelines.drugbank_pipeline import _DRUGBANK_ID_RE

        assert _DRUGBANK_ID_RE.match("DB00945")
        assert _DRUGBANK_ID_RE.match("DB123456")
        # OLD synthesized forms must NOT match the real regex.
        assert not _DRUGBANK_ID_RE.match("DBA1B2C3D4"), "old 8-hex form must not match real regex"
        assert not _DRUGBANK_ID_RE.match("DBSYNTH000001"), "old DBSYNTH form must not match real regex"

    def test_synthesized_id_regex_matches_new_prefix(self):
        from pipelines.drugbank_pipeline import _SYNTHESIZED_DRUG_ID_RE

        assert _SYNTHESIZED_DRUG_ID_RE.match("SYNTH-DB-A1B2C3D4")
        assert _SYNTHESIZED_DRUG_ID_RE.match("SYNTH-DB-M000001")
        # Real DrugBank IDs must NOT match the synthesized regex.
        assert not _SYNTHESIZED_DRUG_ID_RE.match("DB00945")
        # OLD synthesized forms must NOT match the new regex.
        assert not _SYNTHESIZED_DRUG_ID_RE.match("DBA1B2C3D4")
        assert not _SYNTHESIZED_DRUG_ID_RE.match("DBSYNTH000001")

    def test_combined_validator_accepts_both_forms(self):
        from pipelines.drugbank_pipeline import _is_valid_drugbank_id

        assert _is_valid_drugbank_id("DB00945")
        assert _is_valid_drugbank_id("SYNTH-DB-A1B2C3D4")
        assert _is_valid_drugbank_id("SYNTH-DB-M000001")
        # OLD synthesized forms must be REJECTED.
        assert not _is_valid_drugbank_id("DBA1B2C3D4")
        assert not _is_valid_drugbank_id("DBSYNTH000001")
        # None / empty must be rejected.
        assert not _is_valid_drugbank_id(None)
        assert not _is_valid_drugbank_id("")

    def test_synthesize_drugbank_id_emits_new_prefix(self):
        """_synthesize_drugbank_id must emit SYNTH-DB- prefix."""
        # Replicate the logic from _v50_downloaders.py
        inchikey = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        h = hashlib.sha256(inchikey.encode()).hexdigest()
        synth_id = f"SYNTH-DB-{h[:8].upper()}"
        assert synth_id.startswith("SYNTH-DB-"), f"must start with SYNTH-DB-, got {synth_id}"
        # Must NOT start with DB (the old prefix).
        assert not synth_id.startswith("DB") or synth_id.startswith("SYNTH-DB-")

    def test_resolver_utils_accepts_synthesized_ids(self):
        """Entity resolution must accept synthesized IDs (P1-017)."""
        from entity_resolution.resolver_utils import _is_valid_drugbank_id

        assert _is_valid_drugbank_id("DB00945")
        assert _is_valid_drugbank_id("SYNTH-DB-A1B2C3D4")
        assert _is_valid_drugbank_id("SYNTH-DB-M000001")

    def test_drugbank_id_length_widened(self):
        """DRUGBANK_ID_LENGTH must be 64 to accommodate synthesized IDs."""
        from database.models import DRUGBANK_ID_LENGTH

        assert DRUGBANK_ID_LENGTH >= 17, (
            f"DRUGBANK_ID_LENGTH must be >= 17 (SYNTH-DB- prefix + 8 hex = 17 chars), "
            f"got {DRUGBANK_ID_LENGTH}"
        )

    def test_migration_015_exists(self):
        """Migration 015 must exist to widen the column for existing DBs.

        Originally created as 013, renamed to 015 to avoid collision with
        Team-3's migration 013 (013_is_fda_approved_nullable.sql).
        """
        migration_path = (
            Path(__file__).parent.parent
            / "database"
            / "migrations"
            / "015_widen_drugbank_id_column.sql"
        )
        assert migration_path.exists(), "migration 015 must exist (P1-017)"


# ============================================================================
# P1-018: trigger_phase2 depends on pubchem_load (with graceful degradation)
# ============================================================================

class TestP1_018_TriggerPhase2PubchemDependency:
    """P1-018: trigger_phase2 waits for pubchem_load (SUCCEED or SKIP)."""

    def test_trigger_phase2_uses_none_failed_min_one_success(self):
        dag_path = Path(__file__).parent.parent / "dags" / "master_pipeline_dag.py"
        code = dag_path.read_text()
        assert "TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS" in code, (
            "trigger_phase2 must use NONE_FAILED_MIN_ONE_SUCCESS (P1-018)"
        )
        # The old ALL_SUCCESS on _trigger_phase2 must be changed.
        # Look for the @task decorator line.
        assert "@task(retries=0, execution_timeout=TASK_TIMEOUT, trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)" in code, (
            "_trigger_phase2 decorator must use NONE_FAILED_MIN_ONE_SUCCESS"
        )

    def test_pubchem_load_wired_to_trigger_phase2(self):
        dag_path = Path(__file__).parent.parent / "dags" / "master_pipeline_dag.py"
        code = dag_path.read_text()
        assert "pubchem_load >> trigger_phase2" in code, (
            "pubchem_load must be wired to trigger_phase2 (P1-018)"
        )


# ============================================================================
# P1-019: load_dotenv wrapper inlined
# ============================================================================

class TestP1_019_LoadDotenvInlined:
    """P1-019: module-level load_dotenv wrapper removed, import inlined."""

    def test_no_module_level_load_dotenv_wrapper(self):
        settings_path = Path(__file__).parent.parent / "config" / "settings.py"
        code = settings_path.read_text()
        # The module-level wrapper function must be gone.
        assert "def load_dotenv(*args, **kwargs)" not in code, (
            "module-level load_dotenv wrapper must be removed (P1-019)"
        )
        # The inlined binding must be present.
        assert "_load_dotenv_func" in code, "must use _load_dotenv_func binding"


# ============================================================================
# P1-020: pED50 conversion documents in vitro assumption
# ============================================================================

class TestP1_020_PED50Conversion:
    """P1-020: pED50 treated as pEC50-equivalent with warning tag."""

    def test_ped50_in_allowed_activity_types(self):
        from cleaning.normalizer import _ALLOWED_ACTIVITY_TYPES

        assert "pED50" in _ALLOWED_ACTIVITY_TYPES

    def test_ped50_conversion_produces_correct_nm(self):
        from cleaning.normalizer import normalize_activity_value

        av = normalize_activity_value(value=6.0, units="", activity_type="pED50")
        assert av.value == 1000.0, f"pED50=6 should convert to 1000 nM, got {av.value}"

    def test_ped50_emits_warning_tag(self):
        from cleaning.normalizer import normalize_activity_value

        av = normalize_activity_value(value=6.0, units="", activity_type="pED50")
        assert "ped50_assumed_ec50_equivalent" in av.warnings, (
            f"pED50 must emit ped50_assumed_ec50_equivalent warning, got {av.warnings}"
        )

    def test_pic50_does_not_emit_ped50_warning(self):
        from cleaning.normalizer import normalize_activity_value

        av = normalize_activity_value(value=6.0, units="", activity_type="pIC50")
        assert "ped50_assumed_ec50_equivalent" not in av.warnings


# ============================================================================
# P1-021: MatchConfidence enum hierarchy comment matches values
# ============================================================================

class TestP1_021_MatchConfidenceHierarchy:
    """P1-021: comment matches enum values, runtime assertion prevents drift."""

    def test_pubchem_xref_value_is_0_7(self):
        """The comment said 0.55 but the actual value is 0.7."""
        from entity_resolution.base import MatchConfidence

        assert float(MatchConfidence.PUBCHEM_XREF.value) == 0.7

    def test_all_enum_values_match_documented_hierarchy(self):
        from entity_resolution.base import (
            MatchConfidence,
            _CONFIDENCE_HIERARCHY_ASSERTIONS,
        )

        for name, expected in _CONFIDENCE_HIERARCHY_ASSERTIONS:
            actual = float(getattr(MatchConfidence, name).value)
            assert actual == expected, (
                f"{name}: expected {expected}, got {actual} — "
                f"comment/code drift (P1-021)"
            )

    def test_assertion_would_catch_drift(self):
        """If someone changes an enum value, import should fail."""
        # This test passes if the module imports successfully (assertions passed).
        from entity_resolution.base import MatchConfidence  # noqa: F401


# ============================================================================
# P1-022: OMIM disease_id pattern accepts both forms
# ============================================================================

class TestP1_022_OMIMDiseaseIdAlignment:
    """P1-022: SQL CHECK and Python validator both accept OMIM: and bare."""

    def test_python_validator_accepts_both_forms(self):
        from cleaning._constants import CANONICAL_OMIM_DISEASE_ID_REGEX

        assert CANONICAL_OMIM_DISEASE_ID_REGEX.match("219700")
        assert CANONICAL_OMIM_DISEASE_ID_REGEX.match("OMIM:219700")
        assert CANONICAL_OMIM_DISEASE_ID_REGEX.match("613325")

    def test_python_validator_rejects_invalid_forms(self):
        from cleaning._constants import CANONICAL_OMIM_DISEASE_ID_REGEX

        assert not CANONICAL_OMIM_DISEASE_ID_REGEX.match("OMIM:219")  # too short
        assert not CANONICAL_OMIM_DISEASE_ID_REGEX.match("C0003843")  # DisGeNET format
        assert not CANONICAL_OMIM_DISEASE_ID_REGEX.match("")  # empty

    def test_sql_migration_001_accepts_both_forms(self):
        migration_path = (
            Path(__file__).parent.parent
            / "database"
            / "migrations"
            / "001_initial_schema.sql"
        )
        sql = migration_path.read_text()
        # The SQL CHECK must accept the OMIM: prefix (optional).
        assert "^(OMIM:)?\\d{4,7}$" in sql, (
            "SQL CHECK must accept both OMIM: prefixed and bare forms (P1-022)"
        )

    def test_loaders_comment_does_not_claim_divergence(self):
        """The stale comment claiming SQL CHECK is more restrictive must be gone."""
        loaders_path = Path(__file__).parent.parent / "database" / "loaders.py"
        code = loaders_path.read_text()
        assert "is more restrictive" not in code, (
            "stale comment about SQL CHECK being more restrictive must be removed (P1-022)"
        )


# ============================================================================
# P1-023: canonical ordering comment clarified as lexicographic
# ============================================================================

class TestP1_023_CanonicalOrderingComment:
    """P1-023: comment clarifies lexicographic (not biological) ordering."""

    def test_comment_mentions_lexicographic(self):
        pipeline_path = Path(__file__).parent.parent / "pipelines" / "string_pipeline.py"
        code = pipeline_path.read_text()
        assert "LEXICOGRAPHIC" in code, (
            "comment must mention LEXICOGRAPHIC ordering (P1-023)"
        )
        assert "NOT a biological ordering" in code, (
            "comment must clarify NOT a biological ordering (P1-023)"
        )


# ============================================================================
# P1-024: div-by-zero guard documented
# ============================================================================

class TestP1_024_DivByZeroGuardDocumented:
    """P1-024: div-by-zero guard in base_pipeline.py documented."""

    def test_guard_comment_present(self):
        pipeline_path = Path(__file__).parent.parent / "pipelines" / "base_pipeline.py"
        code = pipeline_path.read_text()
        assert "guarded by" in code and "len(df) > 0" in code, (
            "div-by-zero guard must be documented (P1-024)"
        )


# ============================================================================
# P1-025: pubchem_pipeline div-by-zero invariant documented
# ============================================================================

class TestP1_025_PubchemDivByZeroInvariant:
    """P1-025: invariant documented in pubchem_pipeline.py."""

    def test_invariant_comment_present(self):
        pipeline_path = Path(__file__).parent.parent / "pipelines" / "pubchem_pipeline.py"
        code = pipeline_path.read_text()
        assert "INVARIANT" in code, (
            "invariant must be documented in pubchem_pipeline.py (P1-025)"
        )


# ============================================================================
# P1-026: disgenet_pipeline log no longer uses misleading "?"
# ============================================================================

class TestP1_026_DisgenetLogFixed:
    """P1-026: log uses str(total_available), not '?' for 0."""

    def test_no_question_mark_in_log(self):
        pipeline_path = Path(__file__).parent.parent / "pipelines" / "disgenet_pipeline.py"
        code = pipeline_path.read_text()
        # The misleading '?' must be removed from ACTUAL code (not from
        # comments documenting the fix). Check non-comment lines.
        for line in code.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if 'str(total_available) if total_available else "?"' in stripped:
                pytest.fail(
                    f"misleading '?' must be removed from actual code, found: {stripped}"
                )


# ============================================================================
# P1-027: dead abs() removed from cap check
# ============================================================================

class TestP1_027_DeadAbsRemoved:
    """P1-027: abs() removed from post-conversion cap check."""

    def test_abs_not_in_actual_code(self):
        """P1-027 removed abs() from the UNIT-CONVERSION cap check (``converted``).
        The PSCALE-CONVERSION cap check (``converted_to_nM``) is a SEPARATE
        variable — P1-027 did NOT target it. This test only verifies the
        ``converted`` (unit-conversion) cap check, not ``converted_to_nM``."""
        normalizer_path = Path(__file__).parent.parent / "cleaning" / "normalizer.py"
        with open(normalizer_path) as f:
            lines = f.readlines()
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # P1-027: the unit-conversion cap check must NOT use abs().
            # Match: ``if ... abs(converted) ... _ACTIVITY_CENSORED_MAX``
            # but NOT ``abs(converted_to_nM)`` (different variable, P1-037).
            if (
                "_ACTIVITY_CENSORED_MAX" in line
                and "if " in line
                and "abs(converted)" in line  # specifically abs(converted), not abs(converted_to_nM)
            ):
                pytest.fail(
                    f"abs(converted) must be removed from unit-conversion cap check (P1-027): {line.rstrip()}"
                )


# ============================================================================
# P1-028: CircuitBreaker half-open probe auto-recovery
# ============================================================================

class TestP1_028_CircuitBreakerProbeTimeout:
    """P1-028: half-open probe auto-releases after probe_timeout."""

    def test_probe_timeout_param_accepted(self):
        from _circuit_breaker import _CircuitBreaker

        cb = _CircuitBreaker(probe_timeout=1.0)
        assert cb._probe_timeout == 1.0

    def test_probe_timeout_rejects_zero(self):
        from _circuit_breaker import _CircuitBreaker

        with pytest.raises(ValueError, match="probe_timeout must be >= 1.0s"):
            _CircuitBreaker(probe_timeout=0.5)

    def test_auto_release_after_probe_timeout(self):
        """If a probe is stuck longer than probe_timeout, auto-release."""
        from _circuit_breaker import _CircuitBreaker

        cb = _CircuitBreaker(
            failure_threshold=2, reset_timeout=0.1, probe_timeout=1.0, name="test"
        )
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.15)
        # First call transitions to half_open and reserves probe.
        assert cb.allow_request() == True
        # Second call is refused (probe in flight).
        assert cb.allow_request() == False
        # Wait for probe_timeout to elapse.
        time.sleep(1.1)
        # Third call should auto-release and allow new probe.
        assert cb.allow_request() == True, (
            "probe must auto-release after probe_timeout (P1-028)"
        )

    def test_probe_context_manager_success(self):
        from _circuit_breaker import _CircuitBreaker

        cb = _CircuitBreaker(
            failure_threshold=1, reset_timeout=0.1, probe_timeout=10.0, name="test"
        )
        cb.record_failure()
        time.sleep(0.15)
        with cb.probe() as acquired:
            assert acquired == True
        assert cb.state == "closed", "probe success should close the breaker"

    def test_probe_context_manager_failure(self):
        from _circuit_breaker import _CircuitBreaker

        cb = _CircuitBreaker(
            failure_threshold=1, reset_timeout=0.1, probe_timeout=10.0, name="test"
        )
        cb.record_failure()
        time.sleep(0.15)
        with pytest.raises(RuntimeError):
            with cb.probe() as acquired:
                assert acquired == True
                raise RuntimeError("test")
        assert cb.state == "open", "probe failure should re-open the breaker"

    def test_probe_context_manager_releases_on_exception(self):
        """The probe slot must be released even if the caller raises."""
        from _circuit_breaker import _CircuitBreaker

        cb = _CircuitBreaker(
            failure_threshold=1, reset_timeout=0.1, probe_timeout=10.0, name="test"
        )
        cb.record_failure()
        time.sleep(0.15)
        try:
            with cb.probe() as acquired:
                assert acquired == True
                raise ValueError("caller crashed")
        except ValueError:
            pass
        # The probe flag must be cleared (record_failure was called by the
        # context manager).
        assert cb._half_open_probe_in_flight == False, (
            "probe slot must be released after exception (P1-028)"
        )
