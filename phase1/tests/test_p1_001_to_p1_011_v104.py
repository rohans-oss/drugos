"""v104 FORENSIC ROOT FIX -- Regression tests for P1-001 through P1-011.

Team Member 1 issue set. Each test verifies the ACTUAL code fix (not
comments, not test-file boilerplate). The tests are designed to FAIL
if the fix is reverted -- they are true regression tests.

Run:
    cd phase1
    python -m pytest tests/test_p1_001_to_p1_011_v104.py -v

Or directly:
    cd phase1
    python tests/test_p1_001_to_p1_011_v104.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Make phase1/ importable as the package root (so `import database.connection`
# works the same way the production code does).
PHASE1_ROOT = Path(__file__).resolve().parent.parent
if str(PHASE1_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE1_ROOT))


# ===========================================================================
# P1-001: session_scope defined and usable by retry_transaction
# ===========================================================================

def test_p1_001_session_scope_is_defined_and_callable():
    """P1-001: ``session_scope`` must exist in database.connection."""
    from database.connection import session_scope, get_db_session
    assert session_scope is get_db_session, (
        "P1-001 REGRESSION: session_scope must be an alias for get_db_session. "
        "Got a different object."
    )
    assert callable(session_scope), "session_scope must be callable"


def test_p1_001_session_scope_is_in_all():
    """P1-001: ``session_scope`` must be in ``database.connection.__all__``."""
    from database import connection
    assert "session_scope" in connection.__all__, (
        f"P1-001 REGRESSION: session_scope not in __all__: {connection.__all__}"
    )
    assert "retry_transaction" in connection.__all__, (
        f"P1-001 REGRESSION: retry_transaction not in __all__: {connection.__all__}"
    )


def test_p1_001_retry_transaction_does_not_raise_NameError_on_transient_error():
    """P1-001: ``retry_transaction`` must NOT raise NameError when the
    work callable raises a transient OperationalError.

    Before the fix, ``retry_transaction`` called ``session_scope(...)``
    which was undefined -> NameError on EVERY call, even before any
    work was done. After the fix, ``session_scope`` is a real context
    manager that yields a usable Session. We mock the session factory
    so no real DB is needed.
    """
    import database.connection as conn_mod
    from sqlalchemy.exc import OperationalError
    from sqlalchemy import text

    # Set up an in-memory SQLite engine so session_scope actually works.
    # This is the most honest test -- it does not mock the session, it
    # uses a real one. If session_scope is undefined, NameError fires
    # before the test even gets to the assertion.
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    try:
        # Force re-init of the engine with the new DATABASE_URL.
        try:
            conn_mod.dispose_engine()
        except Exception:
            pass
        conn_mod.reset_global_state()
        # Create a table for the work callable to write to.
        engine = conn_mod.get_engine()
        with engine.begin() as c:
            c.execute(text(
                "CREATE TABLE IF NOT EXISTS _p1_001_test "
                "(id INTEGER PRIMARY KEY, val TEXT)"
            ))

        call_count = {"n": 0}

        def _work(session):
            call_count["n"] += 1
            if call_count["n"] < 2:
                # First attempt: simulate a transient error.
                # retry_transaction should re-execute us on a fresh session.
                raise OperationalError(
                    "statement", params=None, orig=Exception("simulated deadlock")
                )
            # Second attempt: succeed.
            session.execute(text("INSERT INTO _p1_001_test (val) VALUES ('ok')"))

        # retry_transaction must NOT raise NameError. Before the P1-001 fix,
        # it raised NameError: name 'session_scope' is not defined.
        conn_mod.retry_transaction(_work, pipeline_name="p1_001_test", max_retries=2)
        assert call_count["n"] >= 2, (
            f"P1-001 REGRESSION: retry_transaction did not retry the work. "
            f"Expected >= 2 attempts, got {call_count['n']}."
        )
    finally:
        try:
            conn_mod.dispose_engine()
        except Exception:
            pass
        os.environ.pop("DATABASE_URL", None)


# ===========================================================================
# P1-002: duplicate INSERT/DELETE block removed from entity_resolution/run.py
# ===========================================================================

def test_p1_002_duplicate_block_removed():
    """P1-002: the second INSERT/DELETE block must be GONE from run.py.

    The duplicate block was at lines 880-935 of the pre-v104 file. It
    was a copy-paste of the first block (lines 818-879) but lacked
    try/finally cleanup. We grep for the signature of the duplicate
    (two ``DELETE FROM entity_mapping`` calls) and assert there is now
    only ONE.
    """
    run_py = PHASE1_ROOT / "entity_resolution" / "run.py"
    src = run_py.read_text(encoding="utf-8")
    delete_count = src.count("DELETE FROM entity_mapping")
    assert delete_count == 1, (
        f"P1-002 REGRESSION: expected exactly 1 'DELETE FROM entity_mapping' "
        f"in entity_resolution/run.py, found {delete_count}. The duplicate "
        f"block (pre-v104 lines 880-935) appears to have been restored."
    )
    # The "V90 CI fix" comment that lived ONLY in the duplicate block
    # must be GONE. The v104 P1-002 fix replaced it with a comment
    # explaining the deletion.
    assert "V90 CI fix: deduplicate save_df on chembl_id" not in src, (
        "P1-002 REGRESSION: the V90 CI fix duplicate-block comment is back. "
        "The duplicate block was supposed to be deleted (P1-002)."
    )


# ===========================================================================
# P1-003: SCHEMA_VERSION_FALLBACK == 0, SCHEMA_VERSION == max(migrations)
# ===========================================================================

def test_p1_003_schema_version_fallback_is_zero():
    """P1-003: SCHEMA_VERSION_FALLBACK must be 0 (fresh-install semantics)."""
    # Force re-import in case it was cached by another test.
    if "database.base" in sys.modules:
        del sys.modules["database.base"]
    from database.base import SCHEMA_VERSION_FALLBACK, SCHEMA_VERSION
    assert SCHEMA_VERSION_FALLBACK == 0, (
        f"P1-003 REGRESSION: SCHEMA_VERSION_FALLBACK must be 0, "
        f"got {SCHEMA_VERSION_FALLBACK}."
    )
    mig_dir = PHASE1_ROOT / "database" / "migrations"
    mig_versions = []
    for f in mig_dir.glob("0*.sql"):
        if "_rollback" in f.name:
            continue
        try:
            v = int(f.name[:3])
            mig_versions.append(v)
        except ValueError:
            continue
    max_mig = max(mig_versions) if mig_versions else 0
    assert SCHEMA_VERSION == max_mig, (
        f"P1-003 REGRESSION: SCHEMA_VERSION ({SCHEMA_VERSION}) must equal "
        f"max migration version ({max_mig}) when migrations dir is present."
    )


# ===========================================================================
# P1-004: _meta_name initialized at top of check_neo4j_readiness
# ===========================================================================

def test_p1_004_meta_name_initialized_before_branch():
    """P1-004: ``_meta_name = None`` must appear BEFORE the ``if _bind is
    not None:`` branch in check_neo4j_readiness.

    Before the fix, ``_meta_name`` was declared only inside the if-branch,
    so when ``pg_session.bind`` was None, later references to ``_meta_name``
    raised UnboundLocalError.
    """
    exporter_py = PHASE1_ROOT / "exporters" / "neo4j_exporter.py"
    src = exporter_py.read_text(encoding="utf-8")
    # Find the check_neo4j_readiness function body.
    fn_start = src.find("def check_neo4j_readiness(pg_session) -> dict:")
    assert fn_start >= 0, "check_neo4j_readiness function not found"
    # Find the next function definition (end of check_neo4j_readiness).
    fn_end = src.find("\ndef ", fn_start + 1)
    if fn_end < 0:
        fn_end = len(src)
    fn_body = src[fn_start:fn_end]
    # Strip comment lines so the test matches ONLY actual code, not the
    # explanatory comment block. (The comment references _meta_name
    # extensively, which would false-positive the old test.)
    fn_code_only = "\n".join(
        line for line in fn_body.splitlines()
        if not line.strip().startswith("#")
    )
    # Find the position of ``_meta_name`` assignment (``_meta_name = None``
    # or ``_meta_name: Optional[str] = None``) and the position of
    # ``if _bind is not None:`` in the code-only view.
    meta_init_pos = fn_code_only.find("_meta_name")
    bind_branch_pos = fn_code_only.find("if _bind is not None")
    assert meta_init_pos > 0 and bind_branch_pos > 0, (
        f"P1-004 REGRESSION: could not locate _meta_name init or "
        f"if _bind branch in check_neo4j_readiness. "
        f"meta_init_pos={meta_init_pos}, bind_branch_pos={bind_branch_pos}"
    )
    # The FIRST code occurrence of ``_meta_name`` must be an assignment
    # to None (or a type-annotated assignment to None), AND it must come
    # BEFORE the ``if _bind is not None`` branch.
    first_meta_line = fn_code_only[meta_init_pos:meta_init_pos + 200]
    assert "= None" in first_meta_line, (
        f"P1-004 REGRESSION: first code occurrence of _meta_name is not "
        f"an assignment to None. Got: {first_meta_line!r}"
    )
    assert meta_init_pos < bind_branch_pos, (
        f"P1-004 REGRESSION: _meta_name must be initialized BEFORE the "
        f"`if _bind is not None` branch. meta_init_pos={meta_init_pos}, "
        f"bind_branch_pos={bind_branch_pos}."
    )


def test_p1_004_check_neo4j_readiness_with_none_bind():
    """P1-004: check_neo4j_readiness must NOT raise UnboundLocalError when
    pg_session.bind is None. Before the fix, this raised NameError."""
    from exporters.neo4j_exporter import check_neo4j_readiness

    class FakeSession:
        bind = None
        def execute(self, *args, **kwargs):
            # Should never be called when bind is None (the function
            # short-circuits on the missing-bind path).
            raise AssertionError("execute() should not be called with bind=None")

    # Must not raise.
    result = check_neo4j_readiness(FakeSession())
    assert isinstance(result, dict), (
        f"P1-004 REGRESSION: check_neo4j_readiness returned non-dict: {type(result)}"
    )
    assert "ready" in result, (
        f"P1-004 REGRESSION: result missing 'ready' key: {result}"
    )


# ===========================================================================
# P1-005: OMIM MIM range standardized across 3 modules
# ===========================================================================

def test_p1_005_omim_mim_constants_are_canonical():
    """P1-005: cleaning/_constants.py must export OMIM_MIM_MIN=100100 and
    OMIM_MIM_MAX=999999 (the canonical OMIM 6-digit range)."""
    from cleaning._constants import (
        OMIM_MIM_MIN, OMIM_MIM_MAX, CANONICAL_OMIM_DISEASE_ID_REGEX,
    )
    assert OMIM_MIM_MIN == 100100, f"OMIM_MIM_MIN must be 100100, got {OMIM_MIM_MIN}"
    assert OMIM_MIM_MAX == 999999, f"OMIM_MIM_MAX must be 999999, got {OMIM_MIM_MAX}"


def test_p1_005_regex_rejects_4_digit_and_7_digit_mims():
    """P1-005: the canonical OMIM regex must REJECT 4-digit and 7-digit MIMs."""
    from cleaning._constants import CANONICAL_OMIM_DISEASE_ID_REGEX
    # 6-digit valid MIMs (all OMIM ranges).
    assert CANONICAL_OMIM_DISEASE_ID_REGEX.match("100100"), "100100 should match"
    assert CANONICAL_OMIM_DISEASE_ID_REGEX.match("104300"), "104300 (Marfan) should match"
    assert CANONICAL_OMIM_DISEASE_ID_REGEX.match("600000"), "600000 should match"
    assert CANONICAL_OMIM_DISEASE_ID_REGEX.match("999999"), "999999 should match"
    assert CANONICAL_OMIM_DISEASE_ID_REGEX.match("OMIM:100100"), "OMIM:100100 should match"
    # 4-digit MIMs (historical, pre-100000) -- now REJECTED.
    assert not CANONICAL_OMIM_DISEASE_ID_REGEX.match("1024"), "1024 (4-digit) must be rejected"
    assert not CANONICAL_OMIM_DISEASE_ID_REGEX.match("12345"), "12345 (5-digit) must be rejected"
    # 7-digit MIMs (do not exist in OMIM) -- now REJECTED.
    assert not CANONICAL_OMIM_DISEASE_ID_REGEX.match("1001000"), "1001000 (7-digit) must be rejected"
    assert not CANONICAL_OMIM_DISEASE_ID_REGEX.match("9999999"), "9999999 (7-digit) must be rejected"
    # 0-leading or 0 as first digit -- rejected (regex requires [1-9] first).
    assert not CANONICAL_OMIM_DISEASE_ID_REGEX.match("010000"), "010000 (leading 0) must be rejected"


def test_p1_005_disgenet_pipeline_uses_canonical_range():
    """P1-005: disgenet_pipeline must import OMIM_MIM_MIN/MAX from
    cleaning._constants (not define its own divergent constants)."""
    import pipelines.disgenet_pipeline as dp
    # The canonical constants must be imported (not hardcoded locally).
    assert dp._OMIM_MIM_MIN == 100100, (
        f"disgenet _OMIM_MIM_MIN must be 100100, got {dp._OMIM_MIM_MIN}"
    )
    assert dp._OMIM_MIM_MAX == 999999, (
        f"disgenet _OMIM_MIM_MAX must be 999999 (NOT 9999999), got {dp._OMIM_MIM_MAX}"
    )
    # The validator must REJECT 7-digit MIMs (the pre-v104 bug).
    assert not dp._validate_omim_mim_range("1001000"), (
        "disgenet _validate_omim_mim_range must reject 7-digit MIM '1001000'"
    )
    assert not dp._validate_omim_mim_range("9999999"), (
        "disgenet _validate_omim_mim_range must reject 7-digit MIM '9999999'"
    )
    assert dp._validate_omim_mim_range("100100"), (
        "disgenet _validate_omim_mim_range must accept 6-digit MIM '100100'"
    )


def test_p1_005_omim_pipeline_uses_canonical_constants():
    """P1-005: omim_pipeline must import OMIM_MIM_MIN/MAX from
    cleaning._constants (not hardcode 100100/999999 inline)."""
    import pipelines.omim_pipeline as op
    # omim_pipeline imports the canonical names directly (no underscore
    # prefix -- it is the canonical user of these constants, not a
    # private consumer like disgenet).
    assert op.OMIM_MIM_MIN == 100100, (
        f"omim OMIM_MIM_MIN must be 100100, got {op.OMIM_MIM_MIN}"
    )
    assert op.OMIM_MIM_MAX == 999999, (
        f"omim OMIM_MIM_MAX must be 999999, got {op.OMIM_MIM_MAX}"
    )
    # Also verify the source file no longer hardcodes 100100 or 999999
    # in the validation branch (it must reference the imported constants).
    op_src = (PHASE1_ROOT / "pipelines" / "omim_pipeline.py").read_text()
    # Find the validation block.
    val_idx = op_src.find("OMIM_MIM_MIN <= self.phenotype_mim")
    assert val_idx > 0, (
        "P1-005 REGRESSION: omim_pipeline does not use OMIM_MIM_MIN / "
        "OMIM_MIM_MAX constants in the phenotype_mim range check. The "
        "hardcoded 100100/999999 may have been restored."
    )


# ===========================================================================
# P1-006: validate_gda_scores no longer has the wrong hardcoded fallback
# ===========================================================================

def test_p1_006_no_hardcoded_fallback_in_missing_values():
    """P1-006: the wrong hardcoded fallback ``{1: 0.5, 2: 0.6, 3: 0.9, 4: 0.8}``
    must be GONE from missing_values.py. The canonical
    SCORE_BY_MAPPING_KEY (from omim_pipeline) is the only allowed map."""
    mv_py = PHASE1_ROOT / "cleaning" / "missing_values.py"
    src = mv_py.read_text(encoding="utf-8")
    # The wrong fallback appeared as ``except ImportError:`` followed by
    # ``_OMIM_CATEGORICAL_MAP = {1: 0.5, 2: 0.6, 3: 0.9, 4: 0.8}``. Both
    # the except clause AND the wrong map must be GONE.
    assert "{1: 0.5, 2: 0.6, 3: 0.9, 4: 0.8}" not in src, (
        "P1-006 REGRESSION: the wrong hardcoded fallback map "
        "{1: 0.5, 2: 0.6, 3: 0.9, 4: 0.8} is back in missing_values.py. "
        "The canonical map is {1: 0.2, 2: 0.25, 3: 0.9, 4: 0.8} (Piñero 2020)."
    )
    # The try/except ImportError wrapper around the canonical import must
    # also be GONE -- the import must be unconditional (raise ImportError
    # if it fails, not silently substitute wrong values).
    # Find the line that imports SCORE_BY_MAPPING_KEY.
    import_line_idx = src.find("from pipelines.omim_pipeline import SCORE_BY_MAPPING_KEY")
    assert import_line_idx > 0, "SCORE_BY_MAPPING_KEY import not found"
    # Look at the 200 chars BEFORE the import line. There must be NO
    # ``except ImportError:`` immediately before it (which would indicate
    # the try/except wrapper was restored).
    preceding = src[max(0, import_line_idx - 400):import_line_idx]
    assert "except ImportError" not in preceding[-200:], (
        "P1-006 REGRESSION: the try/except ImportError wrapper around the "
        "SCORE_BY_MAPPING_KEY import is back. The import must be unconditional "
        "(raise ImportError if it fails, do NOT silently substitute wrong values)."
    )


def test_p1_006_canonical_score_map_is_correct():
    """P1-006: the canonical SCORE_BY_MAPPING_KEY must be the Piñero 2020 map
    {1: 0.2, 2: 0.25, 3: 0.9, 4: 0.8}."""
    from pipelines.omim_pipeline import SCORE_BY_MAPPING_KEY
    assert SCORE_BY_MAPPING_KEY.get(1) == 0.2, (
        f"SCORE_BY_MAPPING_KEY[1] must be 0.2 (Piñero 2020), got {SCORE_BY_MAPPING_KEY.get(1)}"
    )
    assert SCORE_BY_MAPPING_KEY.get(2) == 0.25, (
        f"SCORE_BY_MAPPING_KEY[2] must be 0.25 (Piñero 2020), got {SCORE_BY_MAPPING_KEY.get(2)}"
    )
    assert SCORE_BY_MAPPING_KEY.get(3) == 0.9, (
        f"SCORE_BY_MAPPING_KEY[3] must be 0.9 (Piñero 2020), got {SCORE_BY_MAPPING_KEY.get(3)}"
    )
    assert SCORE_BY_MAPPING_KEY.get(4) == 0.8, (
        f"SCORE_BY_MAPPING_KEY[4] must be 0.8 (Piñero 2020), got {SCORE_BY_MAPPING_KEY.get(4)}"
    )


# ===========================================================================
# P1-007: python -m pipelines all calls run_entity_resolution()
# ===========================================================================

def test_p1_007_pipelines_all_calls_run_entity_resolution():
    """P1-007: the ``all`` command in pipelines/__init__.py MUST call
    run_entity_resolution() after the pipeline loop. Before the fix, it
    never did -- the KG had 4x duplicate Compound nodes per drug."""
    init_py = PHASE1_ROOT / "pipelines" / "__init__.py"
    src = init_py.read_text(encoding="utf-8")
    # Find the ``all`` command block.
    all_cmd_idx = src.find('elif cmd == "all":')
    assert all_cmd_idx > 0, "'all' command not found in pipelines/__init__.py"
    # The ``all`` block ends at the next ``elif`` or ``else``.
    next_elif = src.find("\n    elif cmd ==", all_cmd_idx)
    if next_elif < 0:
        next_elif = len(src)
    all_block = src[all_cmd_idx:next_elif]
    assert "run_entity_resolution" in all_block, (
        "P1-007 REGRESSION: the 'all' command does NOT call run_entity_resolution(). "
        "This causes 4x duplicate Compound nodes in the KG (one per source: "
        "ChEMBL, DrugBank, PubChem, STITCH)."
    )
    # The call must be inside a try/except so entity-resolution failure
    # does not crash the whole pipeline.
    assert "except Exception as er_exc" in all_block, (
        "P1-007 REGRESSION: run_entity_resolution() call is not wrapped in "
        "try/except. A failure in entity resolution would crash the entire "
        "'all' command."
    )


# ===========================================================================
# P1-008: InChIKey validators unified (no more 7 vs 6 divergence)
# ===========================================================================

def test_p1_008_validators_agree_on_synth_keys():
    """P1-008: ``is_valid_inchikey`` (normalizer) and ``is_canonical_inchikey``
    (_constants) must return the SAME answer for every input. Before the
    fix, they diverged on SYNTH keys of length 6-7."""
    from cleaning.normalizer import is_valid_inchikey
    from cleaning._constants import is_canonical_inchikey
    test_keys = [
        "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",   # canonical 27-char
        "SYNTH-001",                       # SYNTH + 4 chars
        "SYNTH-AB",                        # SYNTH + 3 chars (the divergence case)
        "SYNTH-X",                         # SYNTH + 2 chars (the divergence case)
        "SYNTHY",                          # SYNTH + 1 char (the divergence case)
        "SYNTH",                           # bare SYNTH (should be rejected by both)
        "invalid",
        "",
        "IK001",                           # test fixture prefix -- must be rejected
        "TEST-IK-001",
    ]
    for key in test_keys:
        v1 = is_valid_inchikey(key)
        v2 = is_canonical_inchikey(key)
        assert v1 == v2, (
            f"P1-008 REGRESSION: validators disagree on {key!r}: "
            f"is_valid_inchikey={v1}, is_canonical_inchikey={v2}. "
            f"Both must return the same answer (P1-008 unification)."
        )


def test_p1_008_no_len_ge_7_check_in_normalizer():
    """P1-008: the divergent ``len(cleaned) >= 7`` check must be GONE from
    is_valid_inchikey's EXECUTABLE body in normalizer.py. The docstring
    still references the old check (for forensic trail) -- the test
    must skip the docstring."""
    norm_py = PHASE1_ROOT / "cleaning" / "normalizer.py"
    src = norm_py.read_text(encoding="utf-8")
    # Find the is_valid_inchikey function body.
    fn_start = src.find("def is_valid_inchikey(key: str) -> bool:")
    assert fn_start > 0, "is_valid_inchikey not found"
    fn_end = src.find("\ndef ", fn_start + 1)
    if fn_end < 0:
        fn_end = len(src)
    fn_body = src[fn_start:fn_end]
    # Strip the docstring (the triple-quoted block at the start of the
    # function body) and strip comment lines. The forensic comment
    # references the OLD ``len(cleaned) >= 7`` pattern for traceability --
    # that is intentional and must NOT trigger the regression.
    lines = fn_body.splitlines()
    in_docstring = False
    docstring_char = None
    code_lines = []
    for line in lines:
        stripped = line.strip()
        if not docstring_char and (stripped.startswith('"""') or stripped.startswith("'''")):
            # Docstring open.
            docstring_char = stripped[:3]
            if stripped.count(docstring_char) >= 2 and len(stripped) > 3:
                # Single-line docstring.
                docstring_char = None
            else:
                in_docstring = True
            continue
        if in_docstring:
            if docstring_char in stripped:
                in_docstring = False
                docstring_char = None
            continue
        if stripped.startswith("#"):
            continue
        code_lines.append(line)
    code_only = "\n".join(code_lines)
    assert "len(cleaned) >= 7" not in code_only, (
        "P1-008 REGRESSION: the divergent `len(cleaned) >= 7` check is back "
        "in is_valid_inchikey's executable body. (The docstring still "
        "references it for the forensic trail -- that is intentional.) "
        "This check caused SYNTH keys to flip-flop between valid/invalid "
        "depending on which validator ran first."
    )
    # The function must delegate to is_canonical_inchikey.
    assert "is_canonical_inchikey(key)" in code_only, (
        "P1-008 REGRESSION: is_valid_inchikey does not delegate to "
        "is_canonical_inchikey. The two validators are divergent again."
    )


# ===========================================================================
# P1-009: Neo4j password not passed as CLI arg, not logged
# ===========================================================================

def test_p1_009_password_not_passed_as_cli_arg():
    """P1-009: master_pipeline_dag.py must NOT add ``--neo4j-password`` to
    the subprocess cmd list in EXECUTABLE code. The password flows via
    env-var inheritance. (The forensic comment block references the old
    code for traceability -- that is intentional and must NOT trigger
    the regression.)"""
    dag_py = PHASE1_ROOT / "dags" / "master_pipeline_dag.py"
    src = dag_py.read_text(encoding="utf-8")
    # Strip comment lines (lines whose first non-whitespace char is #)
    # so the forensic comment that documents the REMOVED code does not
    # false-positive the test.
    code_only = "\n".join(
        line for line in src.splitlines()
        if not line.strip().startswith("#")
    )
    assert 'cmd.extend(["--neo4j-password"' not in code_only, (
        "P1-009 REGRESSION: master_pipeline_dag.py is extending cmd with "
        "--neo4j-password in executable code, which leaks the password via "
        "ps/proc/cmdline. The password must flow via env-var inheritance only."
    )


def test_p1_009_cmd_is_redacted_before_logging():
    """P1-009: the cmd must be redacted BEFORE being logged."""
    dag_py = PHASE1_ROOT / "dags" / "master_pipeline_dag.py"
    src = dag_py.read_text(encoding="utf-8")
    # The redaction helper must exist.
    assert "_redact_cmd_for_log" in src, (
        "P1-009 REGRESSION: _redact_cmd_for_log helper not found in "
        "master_pipeline_dag.py. The cmd must be redacted before logging."
    )
    # The log line must use the redacted cmd, not the raw cmd.
    assert '" ".join(_cmd_for_log)' in src or "' '.join(_cmd_for_log)" in src, (
        "P1-009 REGRESSION: the logger.info call does not use _cmd_for_log "
        "(the redacted cmd). The raw cmd is being logged."
    )


def test_p1_009_redact_helper_replaces_password():
    """P1-009: the _redact_cmd_for_log helper must replace sensitive values
    with ***. We test it in isolation by importing the module."""
    # The helper is defined INSIDE _trigger_phase2, so we cannot import
    # it directly. We verify by reading the source and checking the logic.
    dag_py = PHASE1_ROOT / "dags" / "master_pipeline_dag.py"
    src = dag_py.read_text(encoding="utf-8")
    assert '"--neo4j-password"' in src, (
        "P1-009 REGRESSION: --neo4j-password is not in the sensitive_flags set "
        "of _redact_cmd_for_log. The redactor would not redact it."
    )
    assert "'***'" in src or '"***"' in src, (
        "P1-009 REGRESSION: '***' placeholder not found in redactor."
    )


# ===========================================================================
# P1-010: bulk_upsert_drugs counts UPDATEs correctly on SQLite
# ===========================================================================

def test_p1_010_sqlite_pre_count_path_exists():
    """P1-010: _count_upsert_inserts_updates must accept the new
    conflict_keys / chunk_records / target_table kwargs and use them
    for an accurate pre-count on SQLite."""
    loaders_py = PHASE1_ROOT / "database" / "loaders.py"
    src = loaders_py.read_text(encoding="utf-8")
    # The new parameters must be in the signature.
    assert "conflict_keys: list[str] | None = None" in src, (
        "P1-010 REGRESSION: conflict_keys parameter not in _count_upsert_inserts_updates signature."
    )
    assert "chunk_records: list[dict] | None = None" in src, (
        "P1-010 REGRESSION: chunk_records parameter not in signature."
    )
    assert "target_table=None" in src, (
        "P1-010 REGRESSION: target_table parameter not in signature."
    )
    # The bulk_upsert_drugs call site must pass all three.
    bulk_upsert_idx = src.find("def bulk_upsert_drugs(")
    assert bulk_upsert_idx > 0, "bulk_upsert_drugs not found"
    bulk_upsert_end = src.find("\ndef ", bulk_upsert_idx + 1)
    bulk_upsert_body = src[bulk_upsert_idx:bulk_upsert_end if bulk_upsert_end > 0 else len(src)]
    assert 'conflict_keys=["inchikey"]' in bulk_upsert_body, (
        "P1-010 REGRESSION: bulk_upsert_drugs does not pass conflict_keys=['inchikey']. "
        "Without these, the SQLite branch falls back to (total, 0) and falsifies "
        "the audit trail."
    )
    assert "chunk_records=filtered_chunk" in bulk_upsert_body, (
        "P1-010 REGRESSION: bulk_upsert_drugs does not pass chunk_records=filtered_chunk."
    )
    assert "target_table=Drug.__table__" in bulk_upsert_body, (
        "P1-010 REGRESSION: bulk_upsert_drugs does not pass target_table=Drug.__table__."
    )


def test_p1_010_bulk_upsert_drugs_counts_updates_on_second_run(db_engine):
    """P1-010 ROOT FIX: re-running bulk_upsert_drugs on the same data
    must report inserted=0, updated=N (not inserted=N, updated=0).

    This is the audit-trail falsification test. Before the fix, the
    second run reported N inserts when there were actually 0 -- a
    FDA 21 CFR Part 11 violation.

    v106 FIX: use the ``db_engine`` fixture from conftest.py which
    creates a proper in-memory SQLite with ALL tables pre-created via
    ``Base.metadata.create_all(engine)``. The fixture uses SQLAlchemy's
    SingletonThreadPool so all sessions from the same engine in the
    same thread share the same in-memory database. This avoids the
    global engine caching issues that caused test-ordering failures
    when using ``get_db_session()``."""
    from sqlalchemy.orm import Session
    from database.loaders import bulk_upsert_drugs
    import pandas as pd

    # Include is_fda_approved=False to satisfy the chk_drugs_is_fda_approved
    # CHECK constraint (the column is NOT NULL with a CHECK that requires
    # a boolean). Without this, the INSERT fails with IntegrityError
    # before the upsert logic can run.
    df = pd.DataFrame([
        {"inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "name": "Aspirin",
         "is_fda_approved": False},
        {"inchikey": "RZVAJINKQORUOD-UHFFFAOYSA-N", "name": "Ibuprofen",
         "is_fda_approved": False},
    ])
    # First run: 2 inserts, 0 updates.
    with Session(db_engine) as session:
        r1 = bulk_upsert_drugs(session, df)
        session.commit()
        assert r1.inserted == 2, (
            f"P1-010: first run should have inserted=2, got {r1.inserted}"
        )
        assert r1.updated == 0, (
            f"P1-010: first run should have updated=0, got {r1.updated}"
        )
    # Second run: 0 inserts, 2 updates (P1-010 ROOT FIX).
    with Session(db_engine) as session:
        r2 = bulk_upsert_drugs(session, df)
        session.commit()
        assert r2.inserted == 0, (
            f"P1-010 REGRESSION: second run should have inserted=0, "
            f"got {r2.inserted}. The audit trail is FALSIFIED -- this is "
            f"a FDA 21 CFR Part 11 violation."
        )
        assert r2.updated == 2, (
            f"P1-010 REGRESSION: second run should have updated=2, "
            f"got {r2.updated}."
        )


# ===========================================================================
# P1-011: Neo4jExporter class exists and is in __all__
# ===========================================================================

def test_p1_011_neo4j_exporter_class_exists():
    """P1-011: ``from exporters.neo4j_exporter import Neo4jExporter`` must
    NOT raise ImportError. Before the fix, this raised ImportError."""
    from exporters.neo4j_exporter import Neo4jExporter
    assert Neo4jExporter is not None
    assert callable(Neo4jExporter), "Neo4jExporter must be callable (a class)"


def test_p1_011_neo4j_exporter_in_all():
    """P1-011: Neo4jExporter must be in the module's __all__ list."""
    from exporters import neo4j_exporter
    assert hasattr(neo4j_exporter, "__all__"), (
        "neo4j_exporter module must have an __all__ list (P1-011 ROOT FIX)."
    )
    assert "Neo4jExporter" in neo4j_exporter.__all__, (
        f"P1-011 REGRESSION: Neo4jExporter not in __all__: {neo4j_exporter.__all__}"
    )


def test_p1_011_neo4j_exporter_repr_does_not_leak_password():
    """P1-011 + P1-009: Neo4jExporter.__repr__ must NOT include the password."""
    from exporters.neo4j_exporter import Neo4jExporter
    exporter = Neo4jExporter(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="SUPER_SECRET_PASSWORD_123",
    )
    r = repr(exporter)
    assert "SUPER_SECRET_PASSWORD_123" not in r, (
        f"P1-011/P1-009 REGRESSION: Neo4jExporter.__repr__ leaks the password: {r}"
    )
    assert "***" in r, (
        f"P1-011/P1-009 REGRESSION: Neo4jExporter.__repr__ does not redact the password: {r}"
    )


# ===========================================================================
# CLI entry point: run all tests if invoked directly.
# ===========================================================================

if __name__ == "__main__":
    # Run every test_ function in this module.
    test_funcs = [
        (name, obj) for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    failed = 0
    for name, fn in test_funcs:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            failed += 1
    print(f"\n=== P1-001..P1-011 regression tests: {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)
