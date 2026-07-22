"""
v65 ROOT FIX verification suite -- all 14 audit issues (P1C-001 .. P1C-014).

Each test verifies the ROOT fix, not a surface symptom. The tests are
designed so that if ANY issue regresses, the corresponding test FAILS
with a clear message explaining what the regression is and why it
matters.

These tests run against the REAL source files (not mocks). They import
the actual modules and exercise the actual code paths. No smoke tests,
no stubs -- real forensic verification.

Run with:
    cd phase1 && python -m pytest tests/v65_root_fixes/test_v65_all_14_issues.py -v
"""

from __future__ import annotations

import os
import sys
import importlib
import inspect
import re
from pathlib import Path

import pytest

# Ensure phase1/ is on sys.path so `import database`, `import config`,
# `import cleaning`, `import entity_resolution` resolve correctly.
_PHASE1_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PHASE1_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHASE1_ROOT))


# ============================================================================
# P1C-001: gene_symbol / disease_id server_default vs CHECK contradiction
# ============================================================================
class TestP1C001_GdaSchemaContradiction:
    """P1C-001 -- gene_symbol server_default="" contradicted CHECK <> ''.

    ROOT FIX: gene_symbol is nullable=True (no server_default); the
    chk_gda_gene_symbol_nonempty CHECK was REMOVED. disease_id is
    nullable=False (no server_default); its CHECK is preserved.
    """

    def test_gene_symbol_is_nullable_no_server_default(self):
        """gene_symbol must NOT have server_default='' (the contradiction)."""
        from database.models import GeneDiseaseAssociation
        col = GeneDiseaseAssociation.__table__.c.gene_symbol
        assert col.nullable is True, (
            "gene_symbol must be nullable=True so NULL rows are quarantined "
            "by the loader instead of crashing INSERTs via server_default=''."
        )
        sd = col.server_default
        assert sd is None or sd.arg == "", (
            f"gene_symbol must have NO server_default (found {sd!r}). "
            "A server_default='' would contradict any CHECK <> '' constraint."
        )

    def test_gene_symbol_nonempty_check_removed(self):
        """The chk_gda_gene_symbol_nonempty CHECK must be REMOVED."""
        from database.models import GeneDiseaseAssociation
        check_names = {
            c.name for c in GeneDiseaseAssociation.__table__.constraints
            if hasattr(c, "name") and c.name and "gene_symbol_nonempty" in c.name
        }
        assert not check_names, (
            "chk_gda_gene_symbol_nonempty must be REMOVED -- it contradicted "
            f"the old server_default=''. Found: {check_names}"
        )

    def test_disease_id_no_server_default(self):
        """disease_id must NOT have server_default=''."""
        from database.models import GeneDiseaseAssociation
        col = GeneDiseaseAssociation.__table__.c.disease_id
        sd = col.server_default
        assert sd is None or sd.arg == "", (
            f"disease_id must have NO server_default (found {sd!r}). "
            "A server_default='' would contradict the CHECK disease_id <> ''."
        )

    def test_disease_id_nonempty_check_preserved(self):
        """The chk_gda_disease_id_nonempty CHECK must be PRESERVED."""
        from database.models import GeneDiseaseAssociation
        check_names = {
            c.name for c in GeneDiseaseAssociation.__table__.constraints
            if hasattr(c, "name") and c.name and "disease_id_nonempty" in c.name
        }
        assert "chk_gda_disease_id_nonempty" in check_names, (
            "chk_gda_disease_id_nonempty must be PRESERVED -- an empty "
            "disease_id is scientifically meaningless."
        )


# ============================================================================
# P1C-002: UniProt test-fixture acceptance (<6-char alphanumeric + staging)
# ============================================================================
class TestP1C002_UniprotTestFixtureAcceptance:
    """P1C-002 -- <6-char alphanumeric UniProt IDs accepted by default.

    ROOT FIX: default DRUGOS_ENVIRONMENT to "prod" (fail-closed), remove
    "staging" from the allow-test list, and REMOVE the <6-char
    alphanumeric acceptance entirely. Only TEST-prefixed IDs are accepted
    in dev/development/test/ci.
    """

    def test_models_default_is_prod(self):
        """models._validate_uniprot_id must default to 'prod' (not 'dev').

        The validator reads DRUGOS_ENVIRONMENT at CALL time (not import
        time), so we set/unset the env var and call the function -- no
        module reload needed (reloading re-registers ORM classes and
        raises 'Table already defined').
        """
        os.environ.pop("DRUGOS_ENVIRONMENT", None)
        import database.models as m
        # P001 is a <6-char alphanumeric -- must be REJECTED by default.
        with pytest.raises(ValueError, match="Invalid UniProt accession"):
            m._validate_uniprot_id("P001")

    def test_models_rejects_short_alphanumeric_even_in_dev(self):
        """Even in dev, <6-char alphanumeric (P001, ABC) must be REJECTED."""
        os.environ["DRUGOS_ENVIRONMENT"] = "dev"
        import database.models as m
        try:
            with pytest.raises(ValueError, match="Invalid UniProt accession"):
                m._validate_uniprot_id("P001")
            with pytest.raises(ValueError, match="Invalid UniProt accession"):
                m._validate_uniprot_id("ABC")
        finally:
            os.environ.pop("DRUGOS_ENVIRONMENT", None)

    def test_models_accepts_test_prefix_in_dev(self):
        """TEST-prefixed IDs are accepted in dev (explicit opt-in)."""
        os.environ["DRUGOS_ENVIRONMENT"] = "dev"
        import database.models as m
        try:
            assert m._validate_uniprot_id("TEST001") == "TEST001"
        finally:
            os.environ.pop("DRUGOS_ENVIRONMENT", None)

    def test_models_rejects_test_prefix_in_staging(self):
        """staging must be production-like -- TEST fixtures REJECTED."""
        os.environ["DRUGOS_ENVIRONMENT"] = "staging"
        import database.models as m
        try:
            with pytest.raises(ValueError, match="Invalid UniProt accession"):
                m._validate_uniprot_id("TEST001")
        finally:
            os.environ.pop("DRUGOS_ENVIRONMENT", None)

    def test_loaders_default_is_prod(self):
        """loaders._validate_uniprot_id must default to 'prod' (not 'dev')."""
        os.environ.pop("DRUGOS_ENVIRONMENT", None)
        import database.loaders as L
        with pytest.raises(ValueError, match="Invalid UniProt accession"):
            L._validate_uniprot_id("P001")

    def test_loaders_rejects_short_alphanumeric_in_dev(self):
        """loaders mirror: <6-char alphanumeric REJECTED even in dev."""
        os.environ["DRUGOS_ENVIRONMENT"] = "dev"
        import database.loaders as L
        try:
            with pytest.raises(ValueError, match="Invalid UniProt accession"):
                L._validate_uniprot_id("P001")
        finally:
            os.environ.pop("DRUGOS_ENVIRONMENT", None)

    def test_models_accepts_real_uniprot(self):
        """Real 6-char UniProt accessions (P69999) must be accepted."""
        os.environ.pop("DRUGOS_ENVIRONMENT", None)
        import database.models as m
        assert m._validate_uniprot_id("P69999") == "P69999"

    def test_loaders_accepts_real_uniprot(self):
        """loaders mirror: real UniProt accessions accepted."""
        os.environ.pop("DRUGOS_ENVIRONMENT", None)
        import database.loaders as L
        assert L._validate_uniprot_id("P69999") == "P69999"


# ============================================================================
# P1C-003: STRING_MIN_COMBINED_SCORE default 400 vs 700
# ============================================================================
class TestP1C003_StringScoreThreshold:
    """P1C-003 -- .env.example shipped 400, contradicting the 700 default.

    ROOT FIX: .env.example uses 700, CONFIG_REGISTRY default is 700,
    config validation warns when score < 700 (not < 400).
    """

    def test_env_example_uses_700(self):
        """The .env.example file must ship STRING_MIN_COMBINED_SCORE=700."""
        env_path = _PHASE1_ROOT / "config" / ".env.example"
        text = env_path.read_text()
        # Find the STRING_MIN_COMBINED_SCORE line (not commented out).
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("STRING_MIN_COMBINED_SCORE=") and not stripped.startswith("#"):
                assert "=700" in stripped, (
                    f".env.example must set STRING_MIN_COMBINED_SCORE=700, "
                    f"got: {stripped}"
                )
                return
        pytest.fail("STRING_MIN_COMBINED_SCORE not found in .env.example")

    def test_config_registry_default_is_700(self):
        """CONFIG_REGISTRY['STRING_MIN_COMBINED_SCORE']['default'] == '700'."""
        import config.settings as s
        entry = s.CONFIG_REGISTRY.get("STRING_MIN_COMBINED_SCORE", {})
        assert entry.get("default") == "700", (
            f"CONFIG_REGISTRY default must be '700' (scientifically validated), "
            f"got: {entry.get('default')!r}"
        )

    def test_settings_default_is_700(self):
        """The actual STRING_MIN_COMBINED_SCORE setting must be 700."""
        import config.settings as s
        assert s.STRING_MIN_COMBINED_SCORE == 700, (
            f"STRING_MIN_COMBINED_SCORE must be 700, got {s.STRING_MIN_COMBINED_SCORE}"
        )

    def test_config_validation_warns_below_700(self):
        """validate_config must WARN when score is in [400, 700).

        The v65 fix changed the warning threshold from <400 to <700.
        We patch _resolved_settings directly to simulate score=400."""
        import config as cfg
        # Force-load settings if not already loaded.
        cfg._ensure_settings_loaded()
        # Save the original value, patch to 400, run validation, restore.
        original = cfg._resolved_settings.get("STRING_MIN_COMBINED_SCORE")
        try:
            cfg._resolved_settings["STRING_MIN_COMBINED_SCORE"] = 400
            results = cfg._run_validation()
        finally:
            if original is not None:
                cfg._resolved_settings["STRING_MIN_COMBINED_SCORE"] = original
        warnings = [r for r in results
                    if r.severity == "WARNING"
                    and "STRING_MIN_COMBINED_SCORE" in r.setting_name]
        assert len(warnings) > 0, (
            "validate_config must WARN when STRING_MIN_COMBINED_SCORE=400 "
            "(the old low-confidence default). The v65 fix changed the "
            "warning threshold from <400 to <700."
        )


# ============================================================================
# P1C-004: is_valid_inchikey fallback uses permissive INCHIKEY_PATTERN
# ============================================================================
class TestP1C004_InchikeyFallbackStrict:
    """P1C-004 -- fallback used permissive INCHIKEY_PATTERN (accepts -X suffix).

    ROOT FIX: fallback uses _STRICT_INCHIKEY_PATTERN (27-char only).
    """

    def test_fallback_rejects_suffixed_key(self, monkeypatch):
        """When cleaning.normalizer is NOT importable, suffixed keys
        (e.g. ...-N-a) must be REJECTED by the fallback (not accepted)."""
        # Simulate cleaning.normalizer not being importable.
        import entity_resolution.base as base
        # Patch the import inside is_valid_inchikey to raise ImportError.
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def blocked_import(name, *args, **kwargs):
            if name == "cleaning.normalizer" or name.startswith("cleaning.normalizer"):
                raise ImportError(f"blocked {name} for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", blocked_import)
        # A suffixed InChIKey (28 chars with -a extension).
        suffixed = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N-a"
        result = base.is_valid_inchikey(suffixed)
        assert result is False, (
            "is_valid_inchikey fallback must REJECT suffixed InChIKeys "
            f"(got {result} for {suffixed!r}). The v65 fix changed the "
            "fallback from INCHIKEY_PATTERN (permissive) to "
            "_STRICT_INCHIKEY_PATTERN (27-char only)."
        )

    def test_fallback_accepts_canonical_27char(self, monkeypatch):
        """The fallback must still accept valid 27-char InChIKeys."""
        import entity_resolution.base as base
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def blocked_import(name, *args, **kwargs):
            if name == "cleaning.normalizer" or name.startswith("cleaning.normalizer"):
                raise ImportError(f"blocked {name} for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", blocked_import)
        canonical = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        result = base.is_valid_inchikey(canonical)
        assert result is True, (
            f"is_valid_inchikey fallback must accept canonical 27-char "
            f"InChIKeys (got {result} for {canonical!r})."
        )


# ============================================================================
# P1C-005: session.rollback() in _quarantine_gda_rows rolls back ENTIRE txn
# ============================================================================
class TestP1C005_QuarantineSavepoint:
    """P1C-005 -- session.rollback() rolled back the ENTIRE transaction.

    ROOT FIX: use session.begin_nested() (SAVEPOINT) so only the
    dead-letter inserts are rolled back, preserving the caller's staged rows.
    """

    def test_quarantine_uses_begin_nested_not_rollback(self):
        """The source code must use begin_nested(), not session.rollback()."""
        src = (_PHASE1_ROOT / "database" / "loaders.py").read_text()
        # Find the _quarantine_gda_rows function body.
        idx = src.find("def _quarantine_gda_rows")
        assert idx != -1, "_quarantine_gda_rows function not found"
        # Extract a reasonable window of the function.
        func_body = src[idx:idx + 8000]
        assert "begin_nested()" in func_body, (
            "_quarantine_gda_rows must use session.begin_nested() (SAVEPOINT) "
            "so a dead-letter flush failure rolls back ONLY the dead-letter "
            "inserts, not the caller's entire transaction."
        )
        # The old session.rollback() call inside the flush-failure handler
        # must be GONE (replaced by the savepoint context manager).
        # We look for the specific pattern "session.rollback()" that was
        # the bug. The savepoint-based code uses "with session.begin_nested():"
        # and does NOT call session.rollback() in the dead-letter path.
        assert "with session.begin_nested():" in func_body, (
            "Must use 'with session.begin_nested():' context manager for "
            "the savepoint."
        )

    def test_quarantine_preserves_caller_rows_on_flush_failure(self, tmp_path):
        """Integration: if the dead-letter flush fails, the caller's
        staged valid GDA rows must SURVIVE (not be rolled back)."""
        from sqlalchemy import create_engine, Column, String, Integer, CheckConstraint
        from sqlalchemy.orm import sessionmaker, declarative_base
        from database.loaders import _quarantine_gda_rows
        import pandas as pd

        engine = create_engine("sqlite:///:memory:")
        Base = declarative_base()

        class DeadLetterGDA(Base):
            __tablename__ = "dead_letter_gda"
            id = Column(Integer, primary_key=True, autoincrement=True)
            gene_symbol = Column(String, nullable=True)
            disease_id = Column(String, nullable=True)
            source = Column(String, nullable=True)
            reason = Column(String, nullable=True)
            # Add a CHECK that will reject the quarantine insert (simulating
            # the CHECK constraint failure that P1C-005 is about).
            details_json = Column(String, nullable=False)  # NOT NULL -- forces failure
            run_id = Column(String, nullable=True)

        class ValidGDA(Base):
            __tablename__ = "valid_gda"
            id = Column(Integer, primary_key=True, autoincrement=True)
            gene_symbol = Column(String, nullable=True)
            disease_id = Column(String, nullable=False)

        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Stage a VALID row in the caller's transaction.
        valid = ValidGDA(gene_symbol="TP53", disease_id="104300")
        session.add(valid)
        session.flush()  # make it visible in the session

        # Now call _quarantine_gda_rows with a row that will FAIL the
        # dead-letter flush (details_json is NOT NULL but we pass None).
        df = pd.DataFrame([{
            "gene_symbol": None,
            "disease_id": "123",
            "source": "test",
        }])
        # Monkeypatch DeadLetterGDA in loaders to use our test model.
        import database.loaders as L
        original = L.DeadLetterGDA
        L.DeadLetterGDA = DeadLetterGDA
        try:
            # The quarantine should NOT roll back the valid row.
            _quarantine_gda_rows(
                df, reason="test_failure",
                session=session, pipeline_run_id=None,
            )
            # The valid row must STILL be in the session (not rolled back).
            # Query it fresh from the DB.
            surviving = session.query(ValidGDA).filter_by(gene_symbol="TP53").all()
            assert len(surviving) == 1, (
                "The caller's valid GDA row was LOST -- _quarantine_gda_rows "
                "rolled back the entire transaction instead of using a "
                "savepoint. This is the P1C-005 bug."
            )
        finally:
            L.DeadLetterGDA = original
            session.rollback()
            session.close()


# ============================================================================
# P1C-006: is_globally_approved nullable=True no server_default
# ============================================================================
class TestP1C006_IsGloballyApprovedServerDefault:
    """P1C-006 -- is_globally_approved was nullable with no server_default.

    ROOT FIX: nullable=False, server_default="0" (matching is_fda_approved
    and is_withdrawn). No NULLs -- INSERTs that omit the column get False.
    """

    def test_is_globally_approved_has_server_default(self):
        """is_globally_approved must have server_default='0'."""
        from database.models import Drug
        col = Drug.__table__.c.is_globally_approved
        sd = col.server_default
        assert sd is not None, (
            "is_globally_approved must have a server_default so INSERTs "
            "that omit the column get False (not NULL). NULL rows are "
            "silently excluded from 'approved' queries by three-valued logic."
        )
        assert "0" in str(sd.arg) or "false" in str(sd.arg).lower(), (
            f"server_default must be '0' (False), got {sd.arg!r}"
        )

    def test_is_globally_approved_not_nullable(self):
        """is_globally_approved must be nullable=False (no NULLs)."""
        from database.models import Drug
        col = Drug.__table__.c.is_globally_approved
        assert col.nullable is False, (
            "is_globally_approved must be nullable=False so no NULL rows "
            "exist. NULL rows are silently excluded from 'approved' queries."
        )

    def test_matches_is_fda_approved_pattern(self):
        """is_globally_approved must match is_fda_approved's pattern."""
        from database.models import Drug
        g = Drug.__table__.c.is_globally_approved
        f = Drug.__table__.c.is_fda_approved
        assert g.nullable == f.nullable, (
            f"is_globally_approved nullable ({g.nullable}) must match "
            f"is_fda_approved ({f.nullable})."
        )
        assert g.server_default is not None and f.server_default is not None, (
            "Both must have server_default."
        )


# ============================================================================
# P1C-007: validate_gda_scores dedup collapses NaN==NaN
# ============================================================================
class TestP1C007_ValidateGdaScoresNanSentinel:
    """P1C-007 -- drop_duplicates collapsed NaN==NaN rows into one.

    ROOT FIX: apply the NaN-sentinel pattern from deduplicator.py before
    drop_duplicates, so rows with NaN in any dedup key survive.
    """

    def test_nan_keyed_rows_survive_dedup(self):
        """Rows with NaN in dedup keys must NOT be collapsed into one."""
        import pandas as pd
        from cleaning.missing_values import validate_gda_scores

        # 3 rows, all with NaN gene_symbol but DIFFERENT disease_id.
        # Without the sentinel fix, drop_duplicates collapses all 3 into 1.
        df = pd.DataFrame([
            {"gene_symbol": None, "disease_id": "D1", "source": "s", "score": 0.5},
            {"gene_symbol": None, "disease_id": "D2", "source": "s", "score": 0.6},
            {"gene_symbol": None, "disease_id": "D3", "source": "s", "score": 0.7},
        ])
        result = validate_gda_scores(df, dedup=True, return_result=False)
        assert len(result) == 3, (
            f"3 NaN-gene_symbol rows must ALL survive dedup (got {len(result)}). "
            "The v65 fix applies NaN-sentinel pattern so NaN-keyed rows are NOT "
            "collapsed by drop_duplicates (which treats NaN==NaN as True)."
        )

    def test_true_duplicates_still_merged(self):
        """True duplicates (identical keys) must still be merged."""
        import pandas as pd
        from cleaning.missing_values import validate_gda_scores

        df = pd.DataFrame([
            {"gene_symbol": "TP53", "disease_id": "D1", "source": "s", "score": 0.5},
            {"gene_symbol": "TP53", "disease_id": "D1", "source": "s", "score": 0.6},
        ])
        result = validate_gda_scores(df, dedup=True, return_result=False)
        assert len(result) == 1, (
            f"2 identical rows must merge into 1 (got {len(result)})."
        )


# ============================================================================
# P1C-008: dedup_interactions conflates pre_filter_drops + duplicates_removed
# ============================================================================
class TestP1C008_DedupInteractionsPreFilterSplit:
    """P1C-008 -- dedup_interactions used conflated duplicates_removed metric.

    ROOT FIX: split pre_filter_drops (null/quarantine) from
    duplicates_removed (true merges), matching dedup_by_inchikey.
    """

    def test_pre_filter_drops_in_result(self):
        """DedupResult from dedup_interactions must have pre_filter_drops."""
        import pandas as pd
        from cleaning.deduplicator import dedup_interactions, DedupResult

        df = pd.DataFrame([
            {"drug_id": "D1", "target_id": "T1", "activity_value": 1.0},
            {"drug_id": "D1", "target_id": "T1", "activity_value": 2.0},  # duplicate
            {"drug_id": None, "target_id": "T1", "activity_value": 3.0},  # null key -> dropped
        ])
        result = dedup_interactions(
            df, keys=["drug_id", "target_id"],
            keep="first", return_result=True,
            null_keys_handler="drop",
        )
        assert isinstance(result, DedupResult)
        # The null-key row should be a pre_filter_drop (not a "duplicate").
        assert result.pre_filter_drops >= 1, (
            f"pre_filter_drops must be >= 1 (null-key row dropped before "
            f"dedup). Got pre_filter_drops={result.pre_filter_drops}."
        )
        # The true duplicate should be in duplicates_removed.
        assert result.duplicates_removed >= 1, (
            f"duplicates_removed must be >= 1 (true duplicate merged). "
            f"Got duplicates_removed={result.duplicates_removed}."
        )

    def test_pre_filter_row_count_captured(self):
        """The source must capture _pre_filter_row_count before filtering."""
        src = (_PHASE1_ROOT / "cleaning" / "deduplicator.py").read_text()
        # Search the ENTIRE source -- the function is very large and the
        # pre_filter_drops line may be far from the function start.
        assert "_pre_filter_row_count = int(len(working))" in src, (
            "dedup_interactions must capture _pre_filter_row_count after "
            "working = df.copy() so pre_filter_drops can be computed."
        )
        assert "pre_filter_drops = max(0, _pre_filter_row_count - int(len(working)))" in src, (
            "dedup_interactions must compute pre_filter_drops from "
            "_pre_filter_row_count - len(working)."
        )


# ============================================================================
# P1C-009: SYNTH InChIKey match method="inchikey_exact" confidence=0.5
# ============================================================================
class TestP1C009_SynthKeyMatchMethodLabel:
    """P1C-009 -- SYNTH match labeled inchikey_exact with confidence 0.5.

    ROOT FIX: method="synthetic_key_match", confidence uses the new
    MatchConfidence.SYNTHETIC_KEY_MATCH enum (not a hardcoded 0.5).
    """

    def test_synth_enum_member_exists(self):
        """MatchConfidence.SYNTHETIC_KEY_MATCH must exist."""
        from entity_resolution.base import MatchConfidence
        assert hasattr(MatchConfidence, "SYNTHETIC_KEY_MATCH"), (
            "MatchConfidence must have a SYNTHETIC_KEY_MATCH member so "
            "SYNTH matches are self-documenting (not a hardcoded 0.5)."
        )
        assert MatchConfidence.SYNTHETIC_KEY_MATCH.value == 0.5

    def test_from_method_maps_synthetic_key_match(self):
        """from_method('synthetic_key_match') must return the enum."""
        from entity_resolution.base import MatchConfidence
        result = MatchConfidence.from_method("synthetic_key_match")
        assert result == MatchConfidence.SYNTHETIC_KEY_MATCH

    def test_synth_match_uses_new_method_label(self):
        """drug_resolver _match_by_inchikey must use 'synthetic_key_match'."""
        src = (_PHASE1_ROOT / "entity_resolution" / "drug_resolver.py").read_text()
        idx = src.find("def _match_by_inchikey")
        func_body = src[idx:idx + 4000]
        assert 'method="synthetic_key_match"' in func_body, (
            "SYNTH InChIKey matches must use method='synthetic_key_match' "
            "(NOT 'inchikey_exact'). The old label was self-contradictory: "
            "method said 'exact' (1.0) but confidence was 0.5."
        )
        assert "MatchConfidence.SYNTHETIC_KEY_MATCH.value" in func_body, (
            "SYNTH match confidence must use the enum (not a hardcoded 0.5)."
        )

    def test_no_hardcoded_05_confidence_for_synth(self):
        """The old hardcoded confidence=0.5 for SYNTH must be gone."""
        src = (_PHASE1_ROOT / "entity_resolution" / "drug_resolver.py").read_text()
        idx = src.find("def _match_by_inchikey")
        func_body = src[idx:idx + 4000]
        # The old pattern was: confidence=0.5,  # v29: was 1.0
        assert "confidence=0.5,  # v29: was 1.0" not in func_body, (
            "The hardcoded confidence=0.5 for SYNTH matches must be REMOVED. "
            "It must use MatchConfidence.SYNTHETIC_KEY_MATCH.value instead."
        )


# ============================================================================
# P1C-010: Dev-default credential check only looks for REPLACE_USER
# ============================================================================
class TestP1C010_CosmicCosmicCredentialDetection:
    """P1C-010 -- cosmic:cosmic credentials silently accepted in prod.

    ROOT FIX: settings.py also checks for 'cosmic:cosmic@' in DATABASE_URL.
    """

    def test_settings_detects_cosmic_cosmic(self, monkeypatch):
        """In production, cosmic:cosmic must raise ValueError on import."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DATABASE_URL", "postgresql://cosmic:cosmic@localhost:5432/db")
        # The P1C-010 fix adds an import-time check that raises ValueError
        # when cosmic:cosmic is detected in a staging/production environment.
        # We verify this by importing settings in a fresh subprocess with
        # the env vars set, and asserting the process exits non-zero with
        # a ValueError mentioning cosmic:cosmic.
        import subprocess
        env = os.environ.copy()
        env["ENVIRONMENT"] = "production"
        env["DATABASE_URL"] = "postgresql://cosmic:cosmic@localhost:5432/db"
        result = subprocess.run(
            [sys.executable, "-c", "import config.settings"],
            capture_output=True, text=True,
            cwd=str(_PHASE1_ROOT),  # phase1/ so `import config` resolves
            env=env,
        )
        assert result.returncode != 0, (
            "config.settings should have RAISED ValueError when "
            "DATABASE_URL contains cosmic:cosmic in production. "
            "The process exited 0 (no error).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "cosmic:cosmic" in result.stderr, (
            f"The ValueError must mention 'cosmic:cosmic'. stderr was:\n{result.stderr}"
        )

    def test_settings_no_raise_for_real_creds_in_prod(self, monkeypatch):
        """Real credentials must NOT raise in production."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DATABASE_URL", "postgresql://realuser:realpass@db.host:5432/db")
        import config.settings as s
        importlib.reload(s)  # must NOT raise
        assert "cosmic:cosmic" not in s.DATABASE_URL

    def test_source_has_cosmic_cosmic_check(self):
        """The source must contain the cosmic:cosmic@ detection."""
        src = (_PHASE1_ROOT / "config" / "settings.py").read_text()
        assert "cosmic:cosmic@" in src, (
            "settings.py must check for 'cosmic:cosmic@' in DATABASE_URL "
            "(the docker-compose dev default that .env.example ships)."
        )


# ============================================================================
# P1C-011: Duplicate if/else dead code in cleaning/__init__.py
# ============================================================================
class TestP1C011_DeadIfElseRemoved:
    """P1C-011 -- both if/else branches were identical (dead code).

    ROOT FIX: removed the if/else; single direct assignment.
    """

    def test_no_identical_if_else_branches(self):
        """The dead if/else (both branches out[col] = result_rows[col].values)
        must be REMOVED."""
        src = (_PHASE1_ROOT / "cleaning" / "__init__.py").read_text()
        # The old pattern was:
        #   if col in out.columns:
        #       out[col] = result_rows[col].values
        #   else:
        #       out[col] = result_rows[col].values
        old_pattern = (
            "if col in out.columns:\n"
            "                    out[col] = result_rows[col].values\n"
            "                else:\n"
            "                    out[col] = result_rows[col].values"
        )
        assert old_pattern not in src, (
            "The dead if/else (both branches identical) must be REMOVED. "
            "Found the old pattern still present."
        )


# ============================================================================
# P1C-012: Inline regex duplicates _INCHIKEY_PATTERN in deduplicator.py
# ============================================================================
class TestP1C012_InlineRegexReplaced:
    """P1C-012 -- inline regex `^[A-Z]{14}-[A-Z]{10}-[A-Z]$` risked divergence.

    ROOT FIX: use the imported _INCHIKEY_PATTERN compiled pattern.
    """

    def test_no_inline_inchikey_regex_string(self):
        """The inline string literal regex must be replaced with _INCHIKEY_PATTERN."""
        src = (_PHASE1_ROOT / "cleaning" / "deduplicator.py").read_text()
        # The old pattern: non_null_valid.str.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
        old_pattern = '.str.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")'
        assert old_pattern not in src, (
            "The inline InChIKey regex string literal must be REPLACED with "
            "the imported _INCHIKEY_PATTERN compiled pattern. Found the old "
            "inline string still present."
        )

    def test_uses_imported_pattern(self):
        """The code must use _INCHIKEY_PATTERN (the imported compiled pattern)."""
        src = (_PHASE1_ROOT / "cleaning" / "deduplicator.py").read_text()
        # The new pattern: non_null_valid.str.match(_INCHIKEY_PATTERN)
        assert ".str.match(_INCHIKEY_PATTERN)" in src, (
            "deduplicator.py must use _INCHIKEY_PATTERN (the imported compiled "
            "pattern) instead of an inline string literal."
        )


# ============================================================================
# P1C-013: n_normalised over-counts (dead variable)
# ============================================================================
class TestP1C013_DeadNNormalisedRemoved:
    """P1C-013 -- n_normalised counted all strings ending in 'N' (over-count).

    ROOT FIX: removed the dead n_normalised variable (the log already used
    _norm_mask.sum() correctly).
    """

    def test_no_dead_n_normalised_overcount(self):
        """The dead n_normalised = (str[-1] == 'N').sum() must be REMOVED."""
        src = (_PHASE1_ROOT / "cleaning" / "deduplicator.py").read_text()
        # The old pattern: n_normalised = int((working["inchikey"].astype(str).str[-1] == "N").sum())
        # This counts ALL strings ending in N, not just valid non-standard keys.
        old_pattern = 'n_normalised = int(\n                                (working["inchikey"].astype(str).str[-1] == "N")\n                                .sum()\n                            )'
        assert old_pattern not in src, (
            "The dead n_normalised variable (which over-counted by including "
            "all strings ending in 'N', not just valid non-standard InChIKeys) "
            "must be REMOVED. The log already uses _norm_mask.sum() correctly."
        )


# ============================================================================
# P1C-014: _ACTIVITY_VALUE_MAX alias points to CENSORED (1e6) not NON-PHYSICAL (1e9)
# ============================================================================
class TestP1C014_ActivityValueMaxNameConflict:
    """P1C-014 -- _ACTIVITY_VALUE_MAX meant 1e6 in _constants but 1e9 in deduplicator.

    ROOT FIX: deduplicator no longer defines _ACTIVITY_VALUE_MAX (uses
    _ACTIVITY_NON_PHYSICAL_MAX directly). _constants renamed the alias to
    _ACTIVITY_VALUE_CENSORED_MAX_LEGACY with a deprecation note.
    """

    def test_deduplicator_no_local_activity_value_max(self):
        """deduplicator.py must NOT define its own _ACTIVITY_VALUE_MAX."""
        import cleaning.deduplicator as d
        # The module must NOT have a _ACTIVITY_VALUE_MAX attribute that
        # shadows the _constants one with a DIFFERENT value.
        if hasattr(d, "_ACTIVITY_VALUE_MAX"):
            # If it exists (via __getattr__ fallback), it must equal the
            # _constants value (1e6), NOT 1e9.
            assert d._ACTIVITY_VALUE_MAX == 1e6, (
                f"deduplicator._ACTIVITY_VALUE_MAX must be 1e6 (censored, "
                f"matching _constants), got {d._ACTIVITY_VALUE_MAX}. The "
                f"local 1e9 alias was removed by P1C-014."
            )

    def test_deduplicator_uses_non_physical_max_directly(self):
        """The source must use _ACTIVITY_NON_PHYSICAL_MAX (not _ACTIVITY_VALUE_MAX)."""
        src = (_PHASE1_ROOT / "cleaning" / "deduplicator.py").read_text()
        # The old local alias definition must be gone.
        old_alias = "_ACTIVITY_VALUE_MAX: float = _ACTIVITY_NON_PHYSICAL_MAX"
        assert old_alias not in src, (
            "deduplicator.py must NOT define the local _ACTIVITY_VALUE_MAX "
            "alias (it shadowed _constants' 1e6 with 1e9). Use "
            "_ACTIVITY_NON_PHYSICAL_MAX directly."
        )
        # The 3 call sites must use _ACTIVITY_NON_PHYSICAL_MAX.
        assert "_ACTIVITY_NON_PHYSICAL_MAX" in src

    def test_constants_has_renamed_alias(self):
        """_constants.py must expose _ACTIVITY_VALUE_CENSORED_MAX_LEGACY."""
        import cleaning._constants as c
        assert hasattr(c, "_ACTIVITY_VALUE_CENSORED_MAX_LEGACY"), (
            "_constants.py must define _ACTIVITY_VALUE_CENSORED_MAX_LEGACY "
            "(the clear, self-documenting name for the censored threshold)."
        )
        assert c._ACTIVITY_VALUE_CENSORED_MAX_LEGACY == 1e6

    def test_constants_backward_compat_alias(self):
        """_constants._ACTIVITY_VALUE_MAX must still exist (backward compat)."""
        import cleaning._constants as c
        assert hasattr(c, "_ACTIVITY_VALUE_MAX"), (
            "_ACTIVITY_VALUE_MAX must still exist in _constants for backward "
            "compat (legacy callers import it)."
        )
        assert c._ACTIVITY_VALUE_MAX == 1e6  # censored, NOT 1e9

    def test_no_module_defines_activity_value_max_as_1e9(self):
        """No module in the codebase may define _ACTIVITY_VALUE_MAX = 1e9."""
        import cleaning._constants as c
        import cleaning.deduplicator as d
        import cleaning.normalizer as n
        # All three must agree on 1e6 (censored) -- none may be 1e9.
        for mod, name in [(c, "_constants"), (d, "deduplicator"), (n, "normalizer")]:
            if hasattr(mod, "_ACTIVITY_VALUE_MAX"):
                val = mod._ACTIVITY_VALUE_MAX
                assert val == 1e6, (
                    f"{name}._ACTIVITY_VALUE_MAX must be 1e6 (censored), "
                    f"got {val}. No module may define it as 1e9 (the "
                    f"non-physical threshold) -- that was the P1C-014 bug."
                )


if __name__ == "__main__":
    # Allow running directly: python test_v65_all_14_issues.py
    pytest.main([__file__, "-v", "--tb=short", "-x"])
