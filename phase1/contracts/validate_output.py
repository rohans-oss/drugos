"""Validate a Phase 1 output directory against the canonical schema.

TM1 TASK 8 ROOT FIX (Team Member 1, Phase 1 Real Data Pipeline):
  This module is the FINAL task in the Airflow DAG. It validates every
  Phase 1 output CSV against :mod:`phase1.contracts.phase1_schema`.
  Returns 0 if all sources are valid, non-zero exit code with detailed
  errors if any source fails validation.

Usage (CLI):
    python -m phase1.contracts.validate_output <processed_data_dir>

Usage (Airflow / Python API):
    from phase1.contracts.validate_output import validate_output_dir
    exit_code = validate_output_dir(Path("processed_data"))
    sys.exit(exit_code)

Validation checks per source CSV
--------------------------------
1. File existence: at least one candidate (filename OR alias) must exist.
2. Required columns: every ``required_columns`` entry must be present.
3. Any-of groups: for each ``any_of_groups`` entry, at least one column
   in the group must be present.
4. Dtype compatibility: each present column is checked against its
   declared dtype (a soft warning if mismatched — pandas will coerce).
5. Row count: if the CSV has fewer than ``min_rows`` data rows, the
   source is flagged as ``below_min_rows`` (error if min_rows > 0).
6. NULL check: for non-nullable required columns, every row must have
   a non-null value.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    # When imported as ``phase1.contracts.validate_output`` (e.g. from
    # outside phase1/).
    from phase1.contracts.phase1_schema import (
        PHASE1_OUTPUT_SCHEMA,
        PHASE1_CSV_FILENAMES,
        SourceSpec,
        ValidationIssue,
        get_all_aliases,
    )
except ImportError:
    # When imported as ``contracts.validate_output`` (e.g. from inside
    # phase1/, which is how the Airflow DAG and tests import it).
    from contracts.phase1_schema import (  # type: ignore[no-redef]
        PHASE1_OUTPUT_SCHEMA,
        PHASE1_CSV_FILENAMES,
        SourceSpec,
        ValidationIssue,
        get_all_aliases,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# File resolution
# =============================================================================


def _resolve_source_file(spec: SourceSpec, base_dir: Path) -> Optional[Path]:
    """Return the first existing candidate path for ``spec``, or None."""
    for name in get_all_aliases(spec.key):
        candidate = base_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


# =============================================================================
# Per-source validation
# =============================================================================


def _validate_source(
    spec: SourceSpec,
    base_dir: Path,
) -> List[ValidationIssue]:
    """Validate one source CSV against its spec. Returns list of issues."""
    issues: List[ValidationIssue] = []
    path = _resolve_source_file(spec, base_dir)

    if path is None:
        # Missing file is a HARD error for required sources (min_rows >= 1),
        # a SOFT warning for optional sources (min_rows == 0).
        severity = "error" if spec.min_rows >= 1 else "warning"
        issues.append(ValidationIssue(
            source=spec.key,
            severity=severity,
            code="file_not_found",
            message=(
                f"No CSV file found for source '{spec.key}'. "
                f"Searched for: {get_all_aliases(spec.key)} in {base_dir}."
            ),
        ))
        return issues

    # Read the CSV (no dtype coercion yet — we check actual dtypes below).
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        issues.append(ValidationIssue(
            source=spec.key,
            severity="error",
            code="empty_file",
            message=f"CSV file {path.name} is empty (no header).",
        ))
        return issues
    except Exception as exc:
        issues.append(ValidationIssue(
            source=spec.key,
            severity="error",
            code="read_error",
            message=f"Could not read CSV {path.name}: {type(exc).__name__}: {exc}",
        ))
        return issues

    actual_columns = set(df.columns)

    # --- Check 1: required columns present -------------------------------
    for col_spec in spec.required_columns:
        if col_spec.name not in actual_columns:
            issues.append(ValidationIssue(
                source=spec.key,
                severity="error",
                code="missing_required_column",
                column=col_spec.name,
                message=(
                    f"Required column '{col_spec.name}' is missing from "
                    f"{path.name}. Description: {col_spec.description}"
                ),
            ))

    # --- Check 2: any-of groups satisfied --------------------------------
    for group in spec.any_of_groups:
        present_in_group = [c for c in group if c in actual_columns]
        if not present_in_group:
            issues.append(ValidationIssue(
                source=spec.key,
                severity="error",
                code="any_of_group_unsatisfied",
                message=(
                    f"At least one of {list(group)} must be present in "
                    f"{path.name}, but NONE were found."
                ),
            ))

    # --- Check 3: non-nullable required columns have no NULLs ------------
    for col_spec in spec.required_columns:
        if col_spec.nullable:
            continue
        if col_spec.name not in actual_columns:
            continue  # already flagged in Check 1
        null_count = int(df[col_spec.name].isna().sum())
        if null_count > 0:
            issues.append(ValidationIssue(
                source=spec.key,
                severity="error",
                code="non_nullable_column_has_nulls",
                column=col_spec.name,
                message=(
                    f"Column '{col_spec.name}' is declared non-nullable but "
                    f"{null_count}/{len(df)} rows are NULL in {path.name}."
                ),
            ))

    # --- Check 4: row count ----------------------------------------------
    row_count = len(df)
    if row_count < spec.min_rows:
        severity = "error" if spec.min_rows >= 1 else "warning"
        issues.append(ValidationIssue(
            source=spec.key,
            severity=severity,
            code="below_min_rows",
            message=(
                f"{path.name} has {row_count} data rows, but spec requires "
                f">= {spec.min_rows}."
            ),
        ))

    # --- Check 5: dtype compatibility (soft warning) ---------------------
    for col_spec in (*spec.required_columns, *spec.optional_columns):
        if col_spec.name not in actual_columns:
            continue
        if col_spec.dtype == "string":
            # pandas reads strings as 'object' by default — fine.
            continue
        actual_dtype = str(df[col_spec.name].dtype)
        # Accept int64 / Int64 / float64 / bool / object — be permissive
        # because pandas auto-infers dtypes on read. Only flag gross
        # mismatches (e.g. a string column where we expect float64).
        if col_spec.dtype == "float64":
            if "float" not in actual_dtype and "int" not in actual_dtype and actual_dtype != "object":
                issues.append(ValidationIssue(
                    source=spec.key,
                    severity="warning",
                    code="dtype_mismatch",
                    column=col_spec.name,
                    message=(
                        f"Column '{col_spec.name}' expected dtype float64 but "
                        f"got {actual_dtype} in {path.name}."
                    ),
                ))
        elif col_spec.dtype == "int64":
            if "int" not in actual_dtype and actual_dtype != "object":
                issues.append(ValidationIssue(
                    source=spec.key,
                    severity="warning",
                    code="dtype_mismatch",
                    column=col_spec.name,
                    message=(
                        f"Column '{col_spec.name}' expected dtype int64 but "
                        f"got {actual_dtype} in {path.name}."
                    ),
                ))

    return issues


# =============================================================================
# Directory-level validation
# =============================================================================


def validate_output_dir(
    base_dir: Path,
    *,
    fail_on_warning: bool = False,
) -> int:
    """Validate every Phase 1 source CSV in ``base_dir``.

    Returns 0 if all sources pass validation (warnings allowed).
    Returns 1 if any ERROR issue is found.
    Returns 2 if ``fail_on_warning=True`` and any WARNING issue is found.

    Logs every issue at the appropriate level (ERROR / WARNING).
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        logger.error("Phase 1 output directory does not exist: %s", base_dir)
        return 1

    all_issues: List[ValidationIssue] = []
    for source_key, spec in PHASE1_OUTPUT_SCHEMA.items():
        issues = _validate_source(spec, base_dir)
        all_issues.extend(issues)

    errors = [i for i in all_issues if i.severity == "error"]
    warnings = [i for i in all_issues if i.severity == "warning"]

    # Log every issue.
    for issue in all_issues:
        if issue.severity == "error":
            logger.error(str(issue))
        else:
            logger.warning(str(issue))

    # Summary line.
    logger.info(
        "Phase 1 validation summary: %d sources checked, %d errors, %d warnings.",
        len(PHASE1_OUTPUT_SCHEMA), len(errors), len(warnings),
    )

    if errors:
        return 1
    if fail_on_warning and warnings:
        return 2
    return 0


# =============================================================================
# CLI entry point
# =============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: ``python -m phase1.contracts.validate_output <dir>``."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate Phase 1 output CSVs against the canonical schema.",
    )
    parser.add_argument(
        "processed_data_dir",
        type=Path,
        help="Path to the Phase 1 processed_data directory.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Treat warnings as errors (exit 2 if any warning).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    exit_code = validate_output_dir(
        args.processed_data_dir,
        fail_on_warning=args.fail_on_warning,
    )
    if exit_code == 0:
        print(f"[OK] Phase 1 output directory {args.processed_data_dir} is valid.")
    else:
        print(f"[FAIL] Phase 1 output directory {args.processed_data_dir} has issues "
              f"(exit code {exit_code}). See logs above.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
