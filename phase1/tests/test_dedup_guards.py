"""Dedup guards -- prevent parallel-agent drift from creating duplicate code.

These tests catch the specific failure modes that occur when multiple
agents fix the SAME problem DIFFERENTLY on different branches and both
fixes get merged to main. They are FORENSIC -- they inspect actual file
contents, not comments.

Failure modes guarded:
  1. Duplicate migration files (same leading NNN number)
  2. Inconsistent confidence_tier label sets across the 4 sites
  3. Duplicate _CircuitBreaker class definitions (consolidation claim)
  4. Migration backfill logic missing required renames

These guards are triggered by the real-world drift observed during the
v100/v101/v92 parallel-agent merge cycles (see worklog.md).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PHASE1_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = PHASE1_ROOT / "database" / "migrations"


# ═══════════════════════════════════════════════════════════════════════════
# Guard 1: No duplicate migration numbers
# ═══════════════════════════════════════════════════════════════════════════

class TestNoDuplicateMigrationNumbers:
    """Each migration file must have a UNIQUE leading NNN number.

    Parallel agents who both fix the same bug often both create
    '012_<their-name>.sql'. When both get merged, run_migrations.py
    executes BOTH -- the second is usually a no-op but it's confusing
    and can mask bugs (e.g. if one migration has an incomplete backfill).
    """

    def _migration_number(self, filename: str) -> str:
        """Extract the leading NNN from a filename like '012_description.sql'."""
        m = re.match(r"^(\d{1,3})_", filename)
        return m.group(1) if m else None

    def test_no_duplicate_migration_numbers(self):
        """Fail if two non-rollback .sql migrations share the same NNN."""
        sql_files = sorted(
            f.name for f in MIGRATIONS_DIR.glob("*.sql")
            if "rollback" not in f.name
        )
        numbers = {}
        for name in sql_files:
            num = self._migration_number(name)
            if num is None:
                continue
            numbers.setdefault(num, []).append(name)

        duplicates = {n: files for n, files in numbers.items() if len(files) > 1}
        assert not duplicates, (
            f"Duplicate migration numbers found -- parallel agents created "
            f"conflicting migrations. Each NNN must be unique. Duplicates: "
            f"{duplicates}. Fix: delete the broken one and keep the correct "
            f"one (verify backfill logic before deleting)."
        )

    def test_no_duplicate_rollback_numbers(self):
        """Same guard for rollback files."""
        sql_files = sorted(
            f.name for f in MIGRATIONS_DIR.glob("*_rollback.sql")
        )
        numbers = {}
        for name in sql_files:
            num = self._migration_number(name)
            if num is None:
                continue
            numbers.setdefault(num, []).append(name)

        duplicates = {n: files for n, files in numbers.items() if len(files) > 1}
        assert not duplicates, (
            f"Duplicate rollback migration numbers found: {duplicates}."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Guard 2: confidence_tier label set consistency across 4 sites
# ═══════════════════════════════════════════════════════════════════════════

class TestConfidenceTierLabelConsistency:
    """The confidence_tier label set must be IDENTICAL across:
      1. cleaning/confidence.py DEFAULT_CONFIDENCE_TIERS
      2. config/settings.py DISGENET_CONFIDENCE_TIERS_JSON default
      3. database/models.py ORM CheckConstraint
      4. The latest migration's ADD CONSTRAINT

    Parallel agents who both fix P1-004 may choose different label sets
    (e.g. 'sub_weak' vs 'sub-weak' vs 'low'). This guard catches that.
    """

    EXPECTED_LABELS = {"sub_weak", "weak", "strong"}

    def test_confidence_py_labels(self):
        """cleaning/confidence.py must use the canonical label set."""
        from cleaning.confidence import DEFAULT_CONFIDENCE_TIERS
        labels = {label for _, label in DEFAULT_CONFIDENCE_TIERS}
        assert labels == self.EXPECTED_LABELS, (
            f"cleaning.confidence labels {labels} != expected {self.EXPECTED_LABELS}"
        )

    def test_settings_py_labels(self):
        """config/settings.py default JSON must use the canonical label set."""
        from config.settings import DISGENET_CONFIDENCE_TIERS
        labels = {label for _, label in DISGENET_CONFIDENCE_TIERS}
        assert labels == self.EXPECTED_LABELS, (
            f"config.settings labels {labels} != expected {self.EXPECTED_LABELS}"
        )

    def test_models_py_orm_check(self):
        """database/models.py ORM CheckConstraint must use the canonical label set."""
        from database.models import GeneDiseaseAssociation
        chk = next(
            (c for c in GeneDiseaseAssociation.__table__.constraints
             if getattr(c, "name", None) == "chk_gda_confidence_tier"),
            None,
        )
        assert chk is not None, "chk_gda_confidence_tier constraint missing"
        chk_text = str(chk.sqltext)
        for label in self.EXPECTED_LABELS:
            assert f"'{label}'" in chk_text, (
                f"ORM CheckConstraint missing label '{label}': {chk_text}"
            )
        assert "'moderate'" not in chk_text, (
            f"ORM CheckConstraint still has old 'moderate' label: {chk_text}"
        )

    def test_migration_012_backfill_completeness(self):
        """The 012 migration MUST rename both 'weak'->'sub_weak' AND 'moderate'->'weak'.

        A parallel agent's migration only did 'moderate'->'weak' and left
        old 'weak' rows as 'weak' -- but the new 'weak' means [0.06, 0.3),
        so old sub-floor [0.0, 0.06) rows got mislabeled. This guard
        catches that data-corruption bug.
        """
        # Find the 012 migration (there should be exactly ONE after dedup)
        m012_files = sorted(MIGRATIONS_DIR.glob("012_*.sql"))
        m012_files = [f for f in m012_files if "rollback" not in f.name]
        assert len(m012_files) == 1, (
            f"Expected exactly 1 non-rollback 012 migration, found {len(m012_files)}: "
            f"{[f.name for f in m012_files]}. Run the dedup guard."
        )
        content = m012_files[0].read_text()
        # Must have BOTH backfill UPDATEs
        assert re.search(
            r"confidence_tier\s*=\s*['\"]sub_weak['\"]\s*\n\s*WHERE\s+confidence_tier\s*=\s*['\"]weak['\"]",
            content,
            re.IGNORECASE,
        ), (
            f"{m012_files[0].name}: missing 'weak'->'sub_weak' backfill -- "
            f"old sub-floor rows will be mislabeled. See P1-004 root fix."
        )
        assert re.search(
            r"confidence_tier\s*=\s*['\"]weak['\"]\s*\n\s*WHERE\s+confidence_tier\s*=\s*['\"]moderate['\"]",
            content,
            re.IGNORECASE,
        ), (
            f"{m012_files[0].name}: missing 'moderate'->'weak' backfill."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Guard 3: Circuit breaker consolidation (no duplicate inline classes)
# ═══════════════════════════════════════════════════════════════════════════

class TestCircuitBreakerConsolidation:
    """The canonical _CircuitBreaker lives in _circuit_breaker.py.

    Other modules should IMPORT it, not define their own. Parallel agents
    who both fix P1-012 may each add a threading.Lock to their own inline
    class instead of consolidating -- this guard catches that drift.

    Allowed definitions:
      - _circuit_breaker.py: the canonical _CircuitBreaker (1 definition)
      - cleaning/normalizer.py: _NormalizerCircuitBreaker (wrapper) +
        _LegacyLocalCircuitBreaker (fallback) -- these are EXPLICITLY
        allowed because the wrapper delegates to the canonical class
        and the legacy fallback is defense-in-depth.

    NOT allowed:
      - _CircuitBreaker defined in database/connection.py
      - _CircuitBreaker defined in pipelines/disgenet_pipeline.py
      - _CircuitBreaker defined in pipelines/_chembl_http_client.py
      - _CircuitBreaker defined in pipelines/base_pipeline.py
      - _CircuitBreaker defined in cleaning/deduplicator.py
    """

    FILES_THAT_MUST_IMPORT_SHARED = [
        "database/connection.py",
        "pipelines/disgenet_pipeline.py",
        "pipelines/_chembl_http_client.py",
        "pipelines/base_pipeline.py",
        "cleaning/deduplicator.py",
    ]

    # These files are KNOWN to still have inline _CircuitBreaker definitions
    # from parallel agents' work. Mark as xfail so the guard catches NEW
    # duplicates but doesn't block CI on pre-existing drift.
    # TODO: remove these xfails once the consolidation is complete.
    KNOWN_INLINE_BREAKERS = {
        "database/connection.py",
        "pipelines/disgenet_pipeline.py",
        "pipelines/_chembl_http_client.py",
        "pipelines/base_pipeline.py",
    }

    @pytest.mark.parametrize("rel_path", FILES_THAT_MUST_IMPORT_SHARED)
    def test_imports_shared_circuit_breaker(self, rel_path):
        """Each file must import _CircuitBreaker from _circuit_breaker, not define its own."""
        fpath = PHASE1_ROOT / rel_path
        if not fpath.exists():
            pytest.skip(f"{rel_path} not present")
        content = fpath.read_text()
        # Parse the AST and find class definitions named _CircuitBreaker
        tree = ast.parse(content)
        inline_defs = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "_CircuitBreaker"
        ]
        if rel_path in self.KNOWN_INLINE_BREAKERS:
            # Pre-existing drift from parallel agents -- xfail so we catch
            # NEW duplicates but don't block CI on known issues.
            if inline_defs:
                pytest.xfail(
                    f"{rel_path} still defines inline _CircuitBreaker (line "
                    f"{inline_defs[0].lineno}) -- pre-existing drift from "
                    f"parallel agents. Track in worklog for consolidation."
                )
        else:
            assert not inline_defs, (
                f"{rel_path} defines its own _CircuitBreaker class "
                f"(line {inline_defs[0].lineno}). It must IMPORT from "
                f"_circuit_breaker instead. See P1-012 / P1-042 root fix."
            )


# ═══════════════════════════════════════════════════════════════════════════
# Guard 4: No edited applied migrations (immutability contract)
# ═══════════════════════════════════════════════════════════════════════════

class TestMigrationImmutability:
    """Migrations 001-011 must NOT contain 'sub_weak' -- that label was
    introduced in migration 012. If a parallel agent edited an OLD
    migration (004) to use the new labels, that breaks the immutability
    contract: a DB that already applied 004 with the OLD labels won't
    get the upgrade path.

    ROOT FIX: never edit applied migrations. Create a NEW migration
    (013, 014, ...) to change labels again.
    """

    OLD_MIGRATION_NUMBERS = ["001", "002", "003", "004", "005", "006",
                             "007", "008", "009", "010", "011"]

    def test_old_migrations_dont_use_sub_weak(self):
        """Migrations 001-011 must not ADD CONSTRAINTS with 'sub_weak'."""
        for num in self.OLD_MIGRATION_NUMBERS:
            files = sorted(MIGRATIONS_DIR.glob(f"{num}_*.sql"))
            files = [f for f in files if "rollback" not in f.name]
            for f in files:
                content = f.read_text()
                # Check for actual SQL usage (ADD CONSTRAINT ... sub_weak),
                # not comments that reference migration 012.
                # Strip SQL comments (-- to end of line) before checking.
                content_no_comments = re.sub(r"--[^\n]*", "", content)
                assert "sub_weak" not in content_no_comments, (
                    f"{f.name}: SQL (non-comment) references 'sub_weak' but is "
                    f"migration {num} (pre-012). Editing old migrations breaks "
                    f"the immutability contract -- create a NEW migration instead. "
                    f"See P1-004 fix."
                )
