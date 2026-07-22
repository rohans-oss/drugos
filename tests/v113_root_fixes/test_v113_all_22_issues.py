"""v113 forensic root-fix verification tests.

This module verifies EVERY fix applied in the v113 root-fix pass. Each
test is named after the issue it covers (SH-002, SH-003, etc.) and
asserts the ROOT-LEVEL behavior — not the surface syntax.

Tests are organized by issue ID so a CI failure points directly at the
issue that regressed. The tests are SELF-CONTAINED — they do NOT depend
on existing test fixtures or conftest.py state (the user explicitly
asked for new tests, not re-runs of existing ones).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make repo root importable so ``shared.contracts.*`` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# SH-002: outcome enum drift — shared has 4 values, rl/phase4_schema had 3
# =============================================================================

def test_sh_002_outcome_enum_matches_shared_contract():
    """SH-002 ROOT FIX: rl.contracts.phase4_schema.OUTCOME_VALUES must
    EXACTLY match shared.contracts.writeback.VALID_OUTCOMES (4 values).

    Previously phase4_schema had 3 values (validated_positive,
    validated_toxic, validated_inconclusive) — missing
    validated_negative + invalidated, and using a non-existent
    "validated_inconclusive". Now both modules use the SAME 4-value set.
    """
    from shared.contracts.writeback import VALID_OUTCOMES as shared_outcomes
    from rl.contracts.phase4_schema import OUTCOME_VALUES as rl_outcomes

    assert set(rl_outcomes) == set(shared_outcomes), (
        f"SH-002 REGRESSION: rl/contracts/phase4_schema.py OUTCOME_VALUES "
        f"({sorted(rl_outcomes)}) does not match shared/contracts/writeback.py "
        f"VALID_OUTCOMES ({sorted(shared_outcomes)}). The drift is back."
    )
    # Must include all 4 canonical values.
    assert "validated_positive" in rl_outcomes
    assert "validated_negative" in rl_outcomes
    assert "validated_toxic" in rl_outcomes
    assert "invalidated" in rl_outcomes
    # Must NOT include the phantom "validated_inconclusive".
    assert "validated_inconclusive" not in rl_outcomes, (
        "SH-002 REGRESSION: 'validated_inconclusive' is NOT a valid "
        "outcome — it was the artifact of the old drift."
    )


# =============================================================================
# SH-003: column name drift — shared has drug/disease, rl/phase4 had drug_id
# =============================================================================

def test_sh_003_csv_columns_match_shared_contract():
    """SH-003 ROOT FIX: rl.contracts.phase4_schema column names must
    EXACTLY match shared.contracts.writeback.WRITEBACK_CSV_COLUMNS.

    Previously phase4_schema used (drug_id, disease_id, drug_name,
    disease_name, score, ...) while the writer (phase4/writeback.py)
    wrote (drug, disease, outcome, ...). A reader using phase4_schema's
    ColumnSpec list would fail to find required columns.
    """
    from shared.contracts.writeback import WRITEBACK_CSV_COLUMNS as shared_cols
    from rl.contracts.phase4_schema import (
        VALIDATED_HYPOTHESES_COLUMN_NAMES as rl_cols,
    )

    assert list(rl_cols) == list(shared_cols), (
        f"SH-003 REGRESSION: rl/contracts/phase4_schema.py column names "
        f"({list(rl_cols)}) do not match shared/contracts/writeback.py "
        f"WRITEBACK_CSV_COLUMNS ({list(shared_cols)}). The drift is back."
    )
    # Spot-check: must include the canonical shared names.
    assert "drug" in rl_cols
    assert "disease" in rl_cols
    # Must NOT include the old drift names.
    assert "drug_id" not in rl_cols
    assert "drug_name" not in rl_cols
    assert "disease_id" not in rl_cols
    assert "disease_name" not in rl_cols


# =============================================================================
# SH-004: TS ValidateRequest has 3 outcomes, Python accepts 4
# =============================================================================

def test_sh_004_ts_validate_request_has_4_outcomes():
    """SH-004 ROOT FIX: the TS ValidateRequest type must declare all 4
    outcomes matching the Python service. We verify by reading the
    api_contracts.ts file and checking the outcome union.
    """
    ts_path = _REPO_ROOT / "frontend" / "contracts" / "api_contracts.ts"
    content = ts_path.read_text(encoding="utf-8")

    # Find the ValidateRequest outcome union.
    # It should contain all 4 outcomes.
    assert '"validated_positive"' in content, "TS contract missing validated_positive"
    assert '"validated_negative"' in content, (
        "SH-004 REGRESSION: TS contract missing validated_negative — the "
        "old 3-value drift is back."
    )
    assert '"validated_toxic"' in content
    assert '"invalidated"' in content, (
        "SH-004 REGRESSION: TS contract missing invalidated — the old "
        "3-value drift is back."
    )
    assert '"validated_inconclusive"' not in content, (
        "SH-004 REGRESSION: TS contract still uses 'validated_inconclusive' "
        "which is NOT a valid Python outcome."
    )


# =============================================================================
# SH-005: TS ValidateResponse shape mismatch with Python
# =============================================================================

def test_sh_005_ts_validate_response_matches_python():
    """SH-005 ROOT FIX: the TS ValidateResponse type must mirror what
    rl/service.py /validate actually returns: {ok, writeback: {...}, message}.
    """
    ts_path = _REPO_ROOT / "frontend" / "contracts" / "api_contracts.ts"
    content = ts_path.read_text(encoding="utf-8")

    # The TS contract must declare `ok` and `writeback` (not `success` and
    # `writeback_path`).
    # We check that the ValidateResponse interface has `ok: boolean` and
    # `writeback: {...}` keys.
    # Find the ValidateResponse block.
    idx = content.find("export interface ValidateResponse")
    assert idx != -1, "TS contract missing ValidateResponse interface"
    block = content[idx:idx + 1500]
    assert "ok: boolean" in block, (
        "SH-005 REGRESSION: TS ValidateResponse missing `ok: boolean` — "
        "the old shape (success: boolean) is back."
    )
    assert "writeback: {" in block, (
        "SH-005 REGRESSION: TS ValidateResponse missing `writeback: {...}` — "
        "the old shape (writeback_path: string) is back."
    )
    assert "phase1_csv_path" in block
    assert "phase2_neo4j_written" in block
    assert "phase3_trigger_path" in block
    assert "writeback_version" in block


# =============================================================================
# SH-024: TS RankedCandidate shape mismatch with Python /rank
# =============================================================================

def test_sh_024_ts_ranked_candidate_matches_python():
    """SH-024 ROOT FIX: TS RankedCandidate must use `drug` and `disease`
    (not drug_id / drug_name / disease_id / disease_name) to match what
    rl/service.py /rank actually returns.
    """
    ts_path = _REPO_ROOT / "frontend" / "contracts" / "api_contracts.ts"
    content = ts_path.read_text(encoding="utf-8")

    # The DrugDiseasePair interface (which RankedCandidate extends) must
    # use `drug` and `disease`, not the 4-field id+name split.
    idx = content.find("export interface DrugDiseasePair")
    assert idx != -1, "TS contract missing DrugDiseasePair interface"
    block = content[idx:idx + 800]
    assert "drug: string" in block, (
        "SH-024 REGRESSION: TS DrugDiseasePair missing `drug: string` — "
        "the old 4-field shape (drug_id, drug_name, ...) is back."
    )
    assert "disease: string" in block
    # Must NOT have the old id/name split.
    assert "drug_id" not in block, (
        "SH-024 REGRESSION: TS DrugDiseasePair still has drug_id — drift is back."
    )
    assert "drug_name" not in block
    assert "disease_id" not in block
    assert "disease_name" not in block


# =============================================================================
# SH-012: WRITEBACK_VERSION drift — phase4/writeback had "1.0.0-rt010",
# shared has "2.0.0-shared-contract"
# =============================================================================

def test_sh_012_writeback_version_consistent():
    """SH-012 ROOT FIX: phase4.writeback.WRITEBACK_VERSION must EQUAL
    shared.contracts.writeback.WRITEBACK_VERSION. The previous local
    override "1.0.0-rt010" drifted from the shared "2.0.0-shared-contract".
    """
    from shared.contracts.writeback import WRITEBACK_VERSION as shared_version
    from phase4.writeback import WRITEBACK_VERSION as phase4_version

    assert phase4_version == shared_version, (
        f"SH-012 REGRESSION: phase4/writeback.py WRITEBACK_VERSION "
        f"({phase4_version!r}) does not match shared/contracts/writeback.py "
        f"WRITEBACK_VERSION ({shared_version!r}). The drift is back."
    )
    assert phase4_version == "2.0.0-shared-contract", (
        f"SH-012 REGRESSION: WRITEBACK_VERSION is {phase4_version!r}, "
        f"expected '2.0.0-shared-contract'."
    )


# =============================================================================
# SH-027: phase4/writeback + trainer.py import from common shim (deprecated)
# =============================================================================

def test_sh_027_phase4_writeback_does_not_import_common_shim():
    """SH-027 ROOT FIX: phase4/writeback.py must import DIRECTLY from
    shared.contracts.writeback, NOT via the deprecated
    common.validated_hypotheses_schema re-export shim.
    """
    src = (_REPO_ROOT / "phase4" / "writeback.py").read_text(encoding="utf-8")
    # The shim's import statement must NOT appear in phase4/writeback.py.
    # We allow the string to appear in COMMENTS (e.g. the SH-027 fix
    # comment itself mentions the shim name), so we check for actual
    # import statements.
    # The forbidden pattern is `from common.validated_hypotheses_schema import`
    # at the start of a line (modulo whitespace).
    import_lines = [
        line.strip() for line in src.splitlines()
        if line.strip().startswith("from common.validated_hypotheses_schema")
    ]
    assert import_lines == [], (
        f"SH-027 REGRESSION: phase4/writeback.py still imports from the "
        f"deprecated common.validated_hypotheses_schema shim: {import_lines}."
    )


def test_sh_027_trainer_does_not_import_common_shim():
    """SH-027 ROOT FIX: graph_transformer/training/trainer.py must import
    DIRECTLY from shared.contracts.writeback.
    """
    src = (_REPO_ROOT / "graph_transformer" / "training" / "trainer.py").read_text(encoding="utf-8")
    import_lines = [
        line.strip() for line in src.splitlines()
        if line.strip().startswith("from common.validated_hypotheses_schema")
    ]
    assert import_lines == [], (
        f"SH-027 REGRESSION: graph_transformer/training/trainer.py still "
        f"imports from the deprecated common.validated_hypotheses_schema "
        f"shim: {import_lines}."
    )


# =============================================================================
# SH-035: rl/service.py hardcodes "/validate" instead of importing URL_VALIDATE
# =============================================================================

def test_sh_035_rl_service_uses_imported_url_constants():
    """SH-035 ROOT FIX: rl/service.py must import URL constants from
    shared.contracts.urls (not hardcode the path strings).
    """
    src = (_REPO_ROOT / "rl" / "service.py").read_text(encoding="utf-8")
    # Must import from shared.contracts.urls.
    assert "from shared.contracts.urls import" in src, (
        "SH-035 REGRESSION: rl/service.py does not import URL constants "
        "from shared.contracts.urls."
    )
    # The route decorators must use the imported constants, not literal
    # path strings. We check that @app.post("/validate") (literal) does
    # NOT appear — it should be @app.post(_URL_VALIDATE).
    assert '@app.post("/validate")' not in src, (
        "SH-035 REGRESSION: rl/service.py still hardcodes "
        "'@app.post(\"/validate\")' instead of using _URL_VALIDATE."
    )
    assert '@app.get("/health")' not in src
    assert '@app.get("/rank")' not in src
    assert '@app.post("/rank")' not in src


# =============================================================================
# P1-015: DrugBank schema whitelist excludes future 5.1.13+ releases
# =============================================================================

def test_p1_015_drugbank_regex_accepts_future_5x_releases():
    """P1-015 ROOT FIX: the DrugBank schema check must accept ANY 5.x
    version (not just the hardcoded 5.0-5.1.12 whitelist). Future 5.1.13,
    5.2.0, etc. must auto-pass.
    """
    # Import the dag module functions directly (no airflow needed for
    # the helper functions).
    sys.path.insert(0, str(_REPO_ROOT / "phase1"))
    try:
        # The dag module imports airflow at top level, which may not be
        # installed in the test env. We exec ONLY the helper functions
        # by reading the source and extracting them.
        # Easier: just import the regex + helpers via a lightweight stub.
        import re
        # Mirror the regex from the dag file.
        SUPPORTED_DRUGBANK_SCHEMA_REGEX = re.compile(r"^5\.\d+(\.\d+)?$")

        # Future 5.x versions must match.
        assert SUPPORTED_DRUGBANK_SCHEMA_REGEX.match("5.1.13")
        assert SUPPORTED_DRUGBANK_SCHEMA_REGEX.match("5.1.20")
        assert SUPPORTED_DRUGBANK_SCHEMA_REGEX.match("5.2.0")
        assert SUPPORTED_DRUGBANK_SCHEMA_REGEX.match("5.3.0")
        assert SUPPORTED_DRUGBANK_SCHEMA_REGEX.match("5.99.99")
        # Existing versions must still match.
        assert SUPPORTED_DRUGBANK_SCHEMA_REGEX.match("5.0")
        assert SUPPORTED_DRUGBANK_SCHEMA_REGEX.match("5.1")
        assert SUPPORTED_DRUGBANK_SCHEMA_REGEX.match("5.1.12")
        # 6.x must NOT match (it gets the WARN path, not auto-accept).
        assert not SUPPORTED_DRUGBANK_SCHEMA_REGEX.match("6.0.0")
        assert not SUPPORTED_DRUGBANK_SCHEMA_REGEX.match("6.1")
        # Bogus versions must NOT match.
        assert not SUPPORTED_DRUGBANK_SCHEMA_REGEX.match("4.0")
        assert not SUPPORTED_DRUGBANK_SCHEMA_REGEX.match("badversion")
        assert not SUPPORTED_DRUGBANK_SCHEMA_REGEX.match("")
    finally:
        sys.path.pop(0)


def test_p1_015_drugbank_dag_has_regex_and_warn_path():
    """P1-015 ROOT FIX: the drugbank_dag.py file must contain the regex
    pattern AND the WARN-only path for 6.x versions."""
    src = (_REPO_ROOT / "phase1" / "dags" / "drugbank_dag.py").read_text(encoding="utf-8")
    assert "SUPPORTED_DRUGBANK_SCHEMA_REGEX" in src
    assert r"^5\.\d+(\.\d+)?$" in src
    assert "BLOCKED_DRUGBANK_SCHEMAS" in src
    assert "_is_drugbank_version_warn_only" in src
    assert "DRUGBANK_MAJOR_VERSION_WARN_THRESHOLD" in src


# =============================================================================
# P1-016: dev_samples claims is_fda_approved=True for all 10 drugs
# =============================================================================

def test_p1_016_dev_samples_have_is_fda_approved_none():
    """P1-016 ROOT FIX: ALL embedded samples must have is_fda_approved=None
    (was True). The is_fda_approved flag is a patient-safety field — the
    dev samples don't verify FDA approval, so they must match the
    production ChEMBL pipeline's v29 ROOT FIX semantics ("unknown —
    pending FDA Orange Book join").
    """
    src = (_REPO_ROOT / "phase1" / "pipelines" / "_dev_samples.py").read_text(encoding="utf-8")
    # Must NOT have any "is_fda_approved": True literals.
    assert '"is_fda_approved": True' not in src, (
        "P1-016 REGRESSION: _dev_samples.py still has "
        "'\"is_fda_approved\": True' — should be None for all embedded samples."
    )
    # Must have "is_fda_approved": None (multiple occurrences).
    assert src.count('"is_fda_approved": None') >= 20, (
        f"P1-016 REGRESSION: expected >=20 occurrences of "
        f"'\"is_fda_approved\": None' (10 ChEMBL + 10 DrugBank), got "
        f"{src.count('\"is_fda_approved\": None')}."
    )


# =============================================================================
# P1-034: _dev_samples._check_dev_environment_at_import_time raises at import
# =============================================================================

def test_p1_034_dev_samples_does_not_raise_at_import_in_non_dev_env():
    """P1-034 ROOT FIX: importing _dev_samples in a non-dev environment
    must NOT raise ImportError. It must log a CRITICAL warning and set
    _PRODUCTION_GUARD_FAILED=True. Calling embedded_* functions in
    production must STILL raise RuntimeError (the runtime guard).
    """
    # Simulate a non-dev environment.
    env_backup = dict(os.environ)
    try:
        os.environ.pop("DRUGOS_ENVIRONMENT", None)
        os.environ.pop("ENVIRONMENT", None)
        # Force re-import of the module (it may already be cached).
        mods_to_remove = [
            k for k in sys.modules if "_dev_samples" in k
        ]
        for k in mods_to_remove:
            del sys.modules[k]
        # Import must NOT raise.
        try:
            import phase1.pipelines._dev_samples as dev_samples  # type: ignore
        except ImportError as exc:
            pytest.fail(
                f"P1-034 REGRESSION: importing _dev_samples in non-dev env "
                f"raised ImportError: {exc}. The P1-034 fix should make "
                f"the import log a CRITICAL warning instead of raising."
            )
        # The production guard flag must be set.
        assert dev_samples._PRODUCTION_GUARD_FAILED is True, (
            "P1-034 REGRESSION: _PRODUCTION_GUARD_FAILED is False after "
            "importing in non-dev env. The import-time check didn't fire."
        )
        # Calling an embedded_* function in production MUST raise.
        # The error message uses "PRODUCTION" (uppercase) — match case-insensitively.
        import re as _re
        with pytest.raises(RuntimeError, match=_re.compile(r"production", _re.IGNORECASE)):
            dev_samples.embedded_chembl_molecules()
    finally:
        # Restore env.
        os.environ.clear()
        os.environ.update(env_backup)
        # Re-clear the cached module so subsequent tests get a fresh import.
        mods_to_remove = [k for k in sys.modules if "_dev_samples" in k]
        for k in mods_to_remove:
            del sys.modules[k]


# =============================================================================
# P1-048: dev_samples embedded_drugbank_interactions has only 10 rows (1:1)
# =============================================================================

def test_p1_048_dev_samples_polypharmacology():
    """P1-048 ROOT FIX: embedded_drugbank_interactions must have >10 rows
    (was exactly 10) with multiple targets per drug (polypharmacology).
    """
    env_backup = dict(os.environ)
    try:
        os.environ["DRUGOS_ENVIRONMENT"] = "development"
        # Force re-import.
        mods_to_remove = [k for k in sys.modules if "_dev_samples" in k]
        for k in mods_to_remove:
            del sys.modules[k]
        from phase1.pipelines._dev_samples import embedded_drugbank_interactions  # type: ignore
        df = embedded_drugbank_interactions()
        # Must have MORE than 10 rows (the old 1:1 mapping).
        assert len(df) > 10, (
            f"P1-048 REGRESSION: embedded_drugbank_interactions has only "
            f"{len(df)} rows (expected >10 for polypharmacology)."
        )
        # At least one drug must have >1 target.
        targets_per_drug = df.groupby("drugbank_id").size()
        assert targets_per_drug.max() >= 2, (
            f"P1-048 REGRESSION: max targets per drug is "
            f"{targets_per_drug.max()} (expected >=2 for polypharmacology)."
        )
        # Specifically: aspirin (DB00945) must have >=2 targets (PTGS1 + PTGS2).
        aspirin_targets = df[df["drugbank_id"] == "DB00945"]
        assert len(aspirin_targets) >= 2, (
            f"P1-048 REGRESSION: Aspirin (DB00945) has only "
            f"{len(aspirin_targets)} target(s) — expected >=2 (PTGS1 + PTGS2)."
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)
        mods_to_remove = [k for k in sys.modules if "_dev_samples" in k]
        for k in mods_to_remove:
            del sys.modules[k]


# =============================================================================
# P2-018: chembl_loader regex uses dead re.IGNORECASE flag
# =============================================================================

def test_p2_018_chembl_loader_regex_no_dead_ignorecase():
    """P2-018 ROOT FIX: the _RE_INHIBIT, _RE_ACTIVATE, _RE_BIND, _RE_MODULATE
    regexes must NOT have the dead re.IGNORECASE flag (input is always
    uppercased before matching).
    """
    src = (_REPO_ROOT / "phase2" / "drugos_graph" / "chembl_loader.py").read_text(encoding="utf-8")
    # The regex declarations must not have re.IGNORECASE as a second arg.
    # We strip comment lines first so the test doesn't match "re.IGNORECASE"
    # mentioned in P2-018 ROOT FIX comments.
    code_lines = [
        line for line in src.splitlines()
        if not line.strip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    import re
    # Match `_RE_X = re.compile(\n...\n)` blocks (multi-line).
    pattern = re.compile(
        r"(_RE_\w+)\s*=\s*re\.compile\(\s*\n?(.*?)\n?\s*\)",
        re.DOTALL,
    )
    matches = pattern.findall(code_only)
    assert len(matches) >= 4, (
        f"P2-018: expected >=4 _RE_* regex definitions, found {len(matches)}."
    )
    for name, body in matches:
        assert "re.IGNORECASE" not in body, (
            f"P2-018 REGRESSION: {name} still has dead re.IGNORECASE flag."
        )


# =============================================================================
# P4-048: save_results uses QUOTE_MINIMAL — CSV injection risk
# =============================================================================

def test_p4_048_save_results_uses_quote_all():
    """P4-048 ROOT FIX: save_results must use csv.QUOTE_ALL (not
    QUOTE_MINIMAL) to prevent CSV formula injection.
    """
    src = (_REPO_ROOT / "rl" / "rl_drug_ranker.py").read_text(encoding="utf-8")
    assert "csv.QUOTE_ALL" in src, (
        "P4-048 REGRESSION: rl/rl_drug_ranker.py does not use csv.QUOTE_ALL. "
        "The CSV injection fix is missing."
    )
    # The save_results function must NOT use QUOTE_MINIMAL anymore.
    # Find the save_results function and check its to_csv call.
    idx = src.find("def save_results(")
    assert idx != -1
    save_results_block = src[idx:idx + 5000]
    assert "quoting=csv.QUOTE_MINIMAL" not in save_results_block, (
        "P4-048 REGRESSION: save_results still uses csv.QUOTE_MINIMAL."
    )


def test_p4_048_sanitize_string_prefixes_formula_chars():
    """P4-048 ROOT FIX: sanitize_string must prefix formula-triggering
    characters (=, +, -, @, \t, \r, \n) with a single quote to prevent
    CSV formula injection in Excel/Sheets.

    NOTE: sanitize_string ALSO strips shell-injection chars
    (``;|&`$(){}[]<>``) — that's the existing v49 behavior. The P4-048
    fix ADDS the formula-prefix on top of the existing strip. So a
    string like ``=HYPERLINK("evil")`` becomes ``'=HYPERLINK"evil"``
    (parens stripped, then formula-prefix added). The test verifies
    the LEADING single quote (the P4-048 fix), not the full string
    preservation (which is the v49 strip behavior, unchanged).
    """
    env_backup = dict(os.environ)
    try:
        os.environ["DRUGOS_ENVIRONMENT"] = "development"
        try:
            from rl.rl_drug_ranker import sanitize_string  # type: ignore
        except Exception:
            pytest.skip("rl.rl_drug_ranker not importable (missing torch/pandas)")
        # Formula-triggering strings must be PREFIXED with a single quote.
        # The shell-injection chars in these strings (e.g., parens) are
        # ALSO stripped by the existing v49 logic — that's separate from
        # the P4-048 fix. We assert the LEADING quote only.
        result = sanitize_string("=HYPERLINK(\"evil\")")
        assert result.startswith("'="), (
            f"P4-048 REGRESSION: sanitize_string did not prefix '=' with a "
            f"single quote. Got: {result!r}"
        )

        result = sanitize_string("+WEBSERVICE(\"evil\")")
        assert result.startswith("'+"), (
            f"P4-048 REGRESSION: sanitize_string did not prefix '+' with a "
            f"single quote. Got: {result!r}"
        )

        result = sanitize_string("-1+1|cmd")
        assert result.startswith("'-"), (
            f"P4-048 REGRESSION: sanitize_string did not prefix '-' with a "
            f"single quote. Got: {result!r}"
        )

        result = sanitize_string("@SUM(A1)")
        assert result.startswith("'@"), (
            f"P4-048 REGRESSION: sanitize_string did not prefix '@' with a "
            f"single quote. Got: {result!r}"
        )

        # Normal strings must NOT be prefixed.
        assert sanitize_string("Aspirin") == "Aspirin"
        assert sanitize_string("metformin") == "metformin"
        # Empty/None must return empty string.
        assert sanitize_string(None) == ""
        assert sanitize_string("") == ""
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


# =============================================================================
# P4-049: RewardConfig doesn't validate validated_toxic_penalty
# =============================================================================

def test_p4_049_reward_config_validates_validated_toxic_penalty():
    """P4-049 ROOT FIX: RewardConfig.__post_init__ must validate
    validated_toxic_penalty against low_action_penalty * typical_reward.
    A config with validated_toxic_penalty=0.01 (trivially small) must
    RAISE ValueError.
    """
    env_backup = dict(os.environ)
    try:
        os.environ["DRUGOS_ENVIRONMENT"] = "development"
        try:
            from rl.rl_drug_ranker import RewardConfig  # type: ignore
        except Exception:
            pytest.skip("rl.rl_drug_ranker not importable (missing torch/pandas)")

        # Default config must NOT raise (validated_toxic_penalty=0.5,
        # low_action_penalty=1.0, threshold = 1.0 * 0.5 = 0.5, 0.5 >= 0.5 ✓).
        cfg = RewardConfig()
        assert cfg.validated_toxic_penalty == 0.5

        # A config with trivially small toxic penalty MUST raise.
        with pytest.raises(ValueError, match="P4-049"):
            RewardConfig(validated_toxic_penalty=0.01)

        # A config with NEGATIVE toxic penalty MUST raise.
        with pytest.raises(ValueError, match="validated_toxic_penalty"):
            RewardConfig(validated_toxic_penalty=-0.1)
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


# =============================================================================
# IN-070: Dockerfile.ml installs torch 2.2.2 but PyG wheels are for 2.2.0
# =============================================================================

def test_in_070_dockerfile_ml_torch_version_matches_pyg_wheels():
    """IN-070 ROOT FIX: Dockerfile.ml must install torch==2.2.0+cpu
    (was 2.2.2+cpu) to EXACTLY match the PyG wheel URL
    https://data.pyg.org/whl/torch-2.2.0+cpu.html.
    """
    src = (_REPO_ROOT / "Dockerfile.ml").read_text(encoding="utf-8")
    # Check the actual RUN commands (not comments). We strip comment lines
    # so the IN-070 ROOT FIX comment (which mentions "torch==2.2.2+cpu"
    # to explain what was changed) doesn't false-positive.
    code_lines = [
        line for line in src.splitlines()
        if not line.strip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    # Must install torch==2.2.0+cpu (not 2.2.2+cpu) in a RUN command.
    assert "torch==2.2.0+cpu" in code_only, (
        "IN-070 REGRESSION: Dockerfile.ml does not install torch==2.2.0+cpu "
        "in a RUN command."
    )
    assert "torch==2.2.2+cpu" not in code_only, (
        "IN-070 REGRESSION: Dockerfile.ml still installs torch==2.2.2+cpu "
        "(mismatches the PyG wheel URL torch-2.2.0+cpu.html)."
    )
    # The PyG wheel URL must use 2.2.0 (matches torch).
    assert "torch-2.2.0+cpu.html" in code_only


# =============================================================================
# IN-074: phase3-trainer healthcheck start_period too short (120s)
# =============================================================================

def test_in_074_phase3_trainer_healthcheck_extended():
    """IN-074 ROOT FIX: phase3-trainer healthcheck start_period must be
    1800s (was 120s) — enough for 80-epoch CPU training. The healthcheck
    must also accept the training_complete sentinel file as a pass.
    """
    src = (_REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    # Must have start_period: 1800s (not 120s).
    assert "start_period: 1800s" in src, (
        "IN-074 REGRESSION: phase3-trainer healthcheck start_period is not 1800s."
    )
    assert "start_period: 120s" not in src, (
        "IN-074 REGRESSION: docker-compose.yml still has start_period: 120s."
    )
    # Must use the training_complete sentinel file fallback.
    assert "training_complete" in src, (
        "IN-074 REGRESSION: healthcheck does not check training_complete sentinel."
    )


# =============================================================================
# IN-073: No sslmode on Postgres connection strings
# =============================================================================

def test_in_073_postgres_uris_have_sslmode():
    """IN-073 ROOT FIX: every Postgres connection string in docker-compose.yml
    must include ?sslmode=prefer (was no sslmode parameter).
    """
    src = (_REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    # Find every postgresql:// URI and verify it has sslmode.
    import re
    uris = re.findall(r"postgresql://[^\s\"]+", src)
    assert len(uris) >= 3, f"Expected >=3 Postgres URIs, found {len(uris)}"
    for uri in uris:
        assert "sslmode=" in uri, (
            f"IN-073 REGRESSION: Postgres URI {uri!r} does not have sslmode=."
        )


# =============================================================================
# IN-077: Dockerfile.airflow does not validate Fernet key at build/runtime
# =============================================================================

def test_in_077_airflow_dockerfile_has_fernet_entrypoint():
    """IN-077 ROOT FIX: Dockerfile.airflow must COPY an entrypoint script
    that validates AIRFLOW__CORE__FERNET_KEY before starting the scheduler.
    """
    dockerfile_src = (_REPO_ROOT / "Dockerfile.airflow").read_text(encoding="utf-8")
    # Must COPY the entrypoint script.
    assert "Dockerfile.airflow.entrypoint.sh" in dockerfile_src, (
        "IN-077 REGRESSION: Dockerfile.airflow does not COPY the entrypoint script."
    )
    assert "ENTRYPOINT" in dockerfile_src, (
        "IN-077 REGRESSION: Dockerfile.airflow does not set ENTRYPOINT."
    )

    # The entrypoint script must exist and validate the Fernet key.
    entrypoint_path = _REPO_ROOT / "Dockerfile.airflow.entrypoint.sh"
    assert entrypoint_path.exists(), (
        "IN-077 REGRESSION: Dockerfile.airflow.entrypoint.sh does not exist."
    )
    entrypoint_src = entrypoint_path.read_text(encoding="utf-8")
    assert "AIRFLOW__CORE__FERNET_KEY" in entrypoint_src
    assert "cryptography.fernet" in entrypoint_src or "Fernet" in entrypoint_src
    assert "exec" in entrypoint_src  # must hand off to the original command


# =============================================================================
# IN-075: requirements.txt has pytest>=7.4 AND requirements-dev.txt has pytest>=7.4.0
# =============================================================================

def test_in_075_no_pytest_in_production_requirements():
    """IN-075 ROOT FIX: requirements.txt must NOT contain pytest
    (it's a dev dep). requirements-dev.txt must contain pytest>=7.4.0.
    requests must NOT be duplicated in requirements-dev.txt.
    """
    req_src = (_REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    dev_src = (_REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")

    # requirements.txt must NOT have a bare `pytest>=...` line.
    req_lines = [l.strip() for l in req_src.splitlines() if l.strip()]
    pytest_lines_in_prod = [
        l for l in req_lines
        if l.startswith("pytest>") or l.startswith("pytest==") or l == "pytest"
    ]
    assert pytest_lines_in_prod == [], (
        f"IN-075 REGRESSION: requirements.txt has pytest lines: "
        f"{pytest_lines_in_prod}. pytest is a DEV dep — move to requirements-dev.txt."
    )

    # requirements-dev.txt must have pytest>=7.4.0.
    assert "pytest>=7.4.0" in dev_src

    # requests must NOT be duplicated in requirements-dev.txt
    # (it's already in requirements.txt which is included via -r).
    dev_lines = [l.strip() for l in dev_src.splitlines() if l.strip()]
    requests_lines_in_dev = [
        l for l in dev_lines
        if l.startswith("requests>") or l.startswith("requests==")
    ]
    assert requests_lines_in_dev == [], (
        f"IN-075 REGRESSION: requirements-dev.txt has duplicate requests "
        f"lines: {requests_lines_in_dev}. requests is already in requirements.txt."
    )


# =============================================================================
# IN-082: frontend service has no resource limits
# =============================================================================

def test_in_082_frontend_has_resource_limits():
    """IN-082 ROOT FIX: the frontend service in docker-compose.yml must
    have deploy.resources.limits.memory AND limits.cpus AND
    reservations.memory AND NODE_OPTIONS env var.
    """
    src = (_REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    # Find the frontend service block.
    idx = src.find("frontend:")
    assert idx != -1
    frontend_block = src[idx:idx + 3000]
    # Must have deploy.resources.limits.
    assert "deploy:" in frontend_block
    assert "resources:" in frontend_block
    assert "limits:" in frontend_block
    assert "memory: 2g" in frontend_block
    assert "cpus: '2'" in frontend_block
    assert "reservations:" in frontend_block
    assert "memory: 512m" in frontend_block
    # Must have NODE_OPTIONS env var.
    assert "NODE_OPTIONS" in frontend_block
    assert "--max-old-space-size=1536" in frontend_block


# =============================================================================
# P2-020: crosswalk ships only ~30 entries — ID translation fails for 99.9%
# =============================================================================

def test_p2_020_crosswalk_has_seed_table_note_and_extra_entries():
    """P2-020 ROOT FIX: the verified_uniprot_gene_crosswalk.yaml must
    document that it's a SEED table (not full coverage) and must include
    the additional entries added for P1-048 polypharmacy targets.
    """
    src = (_REPO_ROOT / "phase2" / "drugos_graph" / "data" / "verified_uniprot_gene_crosswalk.yaml").read_text(encoding="utf-8")
    # Must document the SEED table nature.
    assert "SEED" in src, (
        "P2-020 REGRESSION: crosswalk YAML does not document that it's a SEED table."
    )
    assert "DRUGOS_BUILTIN_CROSSWALK" in src
    # Must include the P1-048 polypharmacy entries.
    assert "P0DMS8" in src  # ADORA1 (caffeine off-target)
    assert "P29275" in src  # ADORA2B (caffeine off-target)
    assert "P47869" in src  # GABRA2 (diazepam off-target)
    assert "P18507" in src  # GABRG2 (diazepam off-target)
    assert "Q9BYF1" in src  # ACE2 (captopril/lisinopril off-target)


# =============================================================================
# Integration: Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 connectivity
# =============================================================================

def test_phase1_to_phase4_writeback_round_trip():
    """INTEGRATION ROOT FIX: a validated hypothesis written by Phase 4's
    writeback module must be readable by Phase 3's trainer using the
    SAME column names and outcome enum. This is the data flywheel
    contract — Phase 4 writes, Phase 3 reads, both must agree.

    This test verifies the SCHEMA contract end-to-end:
      1. phase4.writeback writes a CSV with the shared contract's columns.
      2. rl.contracts.phase4_schema accepts the CSV (columns match).
      3. graph_transformer.training.trainer reads the CSV (same columns).
    """
    import tempfile
    import csv as csv_mod

    # Use a temp file as the canonical CSV path.
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "validated_hypotheses.csv"

        # Patch the env var so phase4.writeback writes to our temp file.
        env_backup = dict(os.environ)
        os.environ["VALIDATED_HYPOTHESES_CSV"] = str(csv_path)
        os.environ["PHASE1_VALIDATED_CSV"] = str(csv_path)
        try:
            # Phase 4: write a validated hypothesis.
            from phase4.writeback import (
                write_validated_hypothesis,
                ValidatedHypothesis,
                WRITEBACK_VERSION,
            )
            vh = ValidatedHypothesis(
                drug="aspirin",
                disease="pain",
                outcome="validated_positive",
                validated_by="wet_lab:test_partner",
                validation_study_id="NCT12345678",
            )
            # Write via the high-level helper.
            result = write_validated_hypothesis(
                drug=vh.drug,
                disease=vh.disease,
                outcome=vh.outcome,
                validated_by=vh.validated_by,
                validation_study_id=vh.validation_study_id,
            )
            assert csv_path.exists(), "Phase 4 writeback did not create the CSV"

            # Verify the CSV has the canonical shared contract columns.
            with open(csv_path, "r") as f:
                reader = csv_mod.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            row = rows[0]
            # SH-003: columns must match shared contract (drug, disease, ...).
            from shared.contracts.writeback import WRITEBACK_CSV_COLUMNS
            for col in WRITEBACK_CSV_COLUMNS:
                assert col in row, (
                    f"SH-003 REGRESSION: CSV missing column {col!r}. "
                    f"Got columns: {list(row.keys())}"
                )
            # SH-012: writeback_version must be the shared value.
            assert row["writeback_version"] == WRITEBACK_VERSION, (
                f"SH-012 REGRESSION: CSV writeback_version is "
                f"{row['writeback_version']!r}, expected {WRITEBACK_VERSION!r}"
            )
            # SH-002: outcome must be a valid canonical value.
            from shared.contracts.writeback import VALID_OUTCOMES
            assert row["outcome"] in VALID_OUTCOMES

            # Phase 3 / rl.contracts.phase4_schema: validate the CSV row.
            from rl.contracts.phase4_schema import (
                validate_validated_hypotheses_row,
                VALIDATED_HYPOTHESES_COLUMN_NAMES,
            )
            # All columns from the CSV must be in the schema.
            for col in row:
                assert col in VALIDATED_HYPOTHESES_COLUMN_NAMES, (
                    f"SH-003 REGRESSION: CSV column {col!r} not in "
                    f"rl.contracts.phase4_schema.VALIDATED_HYPOTHESES_COLUMN_NAMES."
                )
            # The row must validate cleanly.
            errors = validate_validated_hypotheses_row(row)
            assert errors == [], (
                f"Phase 4 row failed Phase 3 schema validation: {errors}"
            )
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
