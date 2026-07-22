"""Phase 1 contracts package — the SINGLE source of truth for Phase 1 output schema.

TM1 TASK 7 ROOT FIX (Team Member 1, Phase 1 Real Data Pipeline):
  This package defines the canonical CSV column names, dtypes, and
  constraints for ALL 11 Phase 1 output files. Phase 2 (the bridge)
  imports this module instead of maintaining a divergent
  ``_PHASE1_EXPECTED_COLUMNS`` dict — eliminating the schema drift
  that previously caused silent data loss when Phase 1 changed a
  column name and Phase 2 didn't.

Public API
----------
- :data:`PHASE1_OUTPUT_SCHEMA` — dict mapping source key -> ColumnSpec list.
- :data:`PHASE1_CSV_FILENAMES` — dict mapping source key -> canonical CSV filename.
- :func:`validate_csv_file` — validate a single CSV against its schema.
- :func:`validate_output_dir` — validate a whole Phase 1 output directory.

Usage (Phase 2 bridge):
    from phase1.contracts.phase1_schema import PHASE1_OUTPUT_SCHEMA
    expected = PHASE1_OUTPUT_SCHEMA["drugs"].required_columns

Usage (Airflow DAG final task):
    from phase1.contracts.validate_output import validate_output_dir
    exit_code = validate_output_dir(Path("processed_data"))
"""
from __future__ import annotations

try:
    # When imported as ``phase1.contracts`` (e.g. from outside phase1/).
    from phase1.contracts.phase1_schema import (
        PHASE1_OUTPUT_SCHEMA,
        PHASE1_CSV_FILENAMES,
        ColumnSpec,
        ValidationIssue,
        get_required_columns,
        get_any_of_groups,
    )
except ImportError:
    # When imported as ``contracts`` (e.g. from inside phase1/, which
    # is how the existing pipelines/__init__.py and dags/ import their
    # submodules via ``sys.path.insert(0, phase1_root)``).
    from contracts.phase1_schema import (  # type: ignore[no-redef]
        PHASE1_OUTPUT_SCHEMA,
        PHASE1_CSV_FILENAMES,
        ColumnSpec,
        ValidationIssue,
        get_required_columns,
        get_any_of_groups,
    )

__all__ = [
    "PHASE1_OUTPUT_SCHEMA",
    "PHASE1_CSV_FILENAMES",
    "ColumnSpec",
    "ValidationIssue",
    "get_required_columns",
    "get_any_of_groups",
]
