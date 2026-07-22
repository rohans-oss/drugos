"""
shared.monitoring — data flywheel step monitoring and alerting.

ISSUE #355: alert when a flywheel step fails.

The flywheel has 4 steps. Each step can fail silently:
    1. writeback fails → CSV not updated, validated hypothesis lost.
    2. trainer crashes → no fine-tune, model state stale.
    3. ranker doesn't load new bonuses → RL agent uses stale bonuses.
    4. checkpoint corrupt → next training run crashes or produces
       scientifically wrong predictions.

This module provides 4 health-check functions, one per step. Each
returns a FlywheelStepStatus. The orchestrator (Airflow / cron) calls
all 4 and emits an alert if any returns ok=False.

Usage:
    from shared.monitoring.flywheel_monitor import (
        check_writeback_health,
        check_retrain_trigger_health,
        check_checkpoint_health,
        check_rl_ranker_health,
        run_all_checks,
    )

    statuses = run_all_checks()
    for s in statuses:
        if not s.ok:
            alert(s.message)
"""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# How old can the latest validated hypothesis be before we alert?
# Default 7 days (weekly Airflow schedule). Override via env var.
FLYWHEEL_STALENESS_HOURS: int = int(os.environ.get("FLYWHEEL_STALENESS_HOURS", "168"))


@dataclass
class FlywheelStepStatus:
    """Status of a single flywheel step."""
    step: str
    ok: bool
    message: str
    last_run: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        icon = "OK" if self.ok else "FAIL"
        return f"[{icon}] {self.step}: {self.message}"


# ---------------------------------------------------------------------------
# Step 1: writeback CSV health
# ---------------------------------------------------------------------------

def check_writeback_health(
    csv_path: Optional[str] = None,
    staleness_hours: int = FLYWHEEL_STALENESS_HOURS,
) -> FlywheelStepStatus:
    """Verify the validated_hypotheses.csv exists, is non-empty, and is fresh.

    FAIL conditions:
        - CSV does not exist.
        - CSV has 0 rows.
        - CSV's latest entry is older than staleness_hours.
        - CSV's outcome column contains invalid values.
    """
    try:
        import sys
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from shared.contracts.writeback import (
            get_validated_csv_path,
            OUTCOME_COL,
            VALID_OUTCOMES,
            REQUIRED_COLUMNS,
        )
    except Exception as exc:
        return FlywheelStepStatus(
            step="writeback_csv",
            ok=False,
            message=f"failed to import shared.contracts.writeback: {exc}",
        )

    if csv_path is None:
        csv_path = get_validated_csv_path()

    if not os.path.exists(csv_path):
        return FlywheelStepStatus(
            step="writeback_csv",
            ok=False,
            message=f"validated_hypotheses.csv not found at {csv_path}",
            details={"csv_path": csv_path},
        )

    rows: List[Dict[str, str]] = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as exc:
        return FlywheelStepStatus(
            step="writeback_csv",
            ok=False,
            message=f"failed to read CSV: {exc}",
            details={"csv_path": csv_path},
        )

    if not rows:
        return FlywheelStepStatus(
            step="writeback_csv",
            ok=False,
            message=f"CSV has 0 rows (no validated hypotheses yet) at {csv_path}",
            details={"csv_path": csv_path, "row_count": 0},
        )

    # Check required columns.
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing_cols:
        return FlywheelStepStatus(
            step="writeback_csv",
            ok=False,
            message=f"CSV missing required columns: {missing_cols}",
            details={"csv_path": csv_path, "missing_columns": missing_cols},
        )

    # Check outcome values are valid.
    invalid_outcomes = [r[OUTCOME_COL] for r in rows if r.get(OUTCOME_COL) not in VALID_OUTCOMES]
    if invalid_outcomes:
        return FlywheelStepStatus(
            step="writeback_csv",
            ok=False,
            message=f"CSV has {len(invalid_outcomes)} rows with invalid outcome values: {set(invalid_outcomes)}",
            details={"csv_path": csv_path, "invalid_outcomes": list(set(invalid_outcomes))},
        )

    # Check freshness (latest validated_at).
    latest_ts_str = max(r.get("validated_at", "") for r in rows)
    try:
        latest_ts = datetime.fromisoformat(latest_ts_str.replace("Z", "+00:00"))
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - latest_ts).total_seconds() / 3600
        if age_hours > staleness_hours:
            return FlywheelStepStatus(
                step="writeback_csv",
                ok=False,
                message=f"CSV is STALE — latest entry is {age_hours:.1f}h old (threshold {staleness_hours}h)",
                details={
                    "csv_path": csv_path,
                    "row_count": len(rows),
                    "latest_validated_at": latest_ts_str,
                    "age_hours": age_hours,
                    "staleness_threshold_hours": staleness_hours,
                },
            )
    except Exception as exc:
        logger.warning("writeback health: failed to parse timestamp %r: %s", latest_ts_str, exc)

    return FlywheelStepStatus(
        step="writeback_csv",
        ok=True,
        message=f"CSV healthy — {len(rows)} validated hypotheses, latest at {latest_ts_str}",
        details={
            "csv_path": csv_path,
            "row_count": len(rows),
            "latest_validated_at": latest_ts_str,
        },
    )


# ---------------------------------------------------------------------------
# Step 2: retrain trigger JSON health
# ---------------------------------------------------------------------------

def check_retrain_trigger_health(
    trigger_path: Optional[str] = None,
) -> FlywheelStepStatus:
    """Verify the Phase 3 retrain trigger JSON exists and is parseable.

    FAIL conditions:
        - JSON does not exist.
        - JSON is not valid JSON.
        - JSON is not a list of dicts.
        - Any entry is missing required fields (drug, disease, outcome).
    """
    try:
        import sys
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from shared.contracts.writeback import VALID_OUTCOMES
    except Exception:
        VALID_OUTCOMES = [
            "validated_positive", "validated_toxic",
            "validated_negative", "invalidated",
        ]

    if trigger_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        trigger_path = str(repo_root / "graph_transformer" / "retrain_triggered.json")

    if not os.path.exists(trigger_path):
        return FlywheelStepStatus(
            step="retrain_trigger",
            ok=False,
            message=f"retrain trigger JSON not found at {trigger_path}",
            details={"trigger_path": trigger_path},
        )

    try:
        with open(trigger_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except json.JSONDecodeError as exc:
        return FlywheelStepStatus(
            step="retrain_trigger",
            ok=False,
            message=f"retrain trigger JSON is CORRUPT — failed to parse: {exc}",
            details={"trigger_path": trigger_path, "error": str(exc)},
        )

    if not isinstance(entries, list):
        return FlywheelStepStatus(
            step="retrain_trigger",
            ok=False,
            message=f"retrain trigger JSON is not a list (got {type(entries).__name__})",
            details={"trigger_path": trigger_path, "actual_type": type(entries).__name__},
        )

    invalid_entries = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            invalid_entries.append((i, "not a dict"))
            continue
        drug = entry.get("drug", "").strip()
        disease = entry.get("disease", "").strip()
        outcome = entry.get("outcome", "").strip().lower()
        if not drug or not disease:
            invalid_entries.append((i, f"missing drug/disease: {entry}"))
            continue
        if outcome not in VALID_OUTCOMES:
            invalid_entries.append((i, f"invalid outcome: {outcome!r}"))

    if invalid_entries:
        return FlywheelStepStatus(
            step="retrain_trigger",
            ok=False,
            message=f"retrain trigger JSON has {len(invalid_entries)} invalid entries: {invalid_entries[:3]}",
            details={"trigger_path": trigger_path, "invalid_entries": invalid_entries[:10]},
        )

    return FlywheelStepStatus(
        step="retrain_trigger",
        ok=True,
        message=f"retrain trigger JSON healthy — {len(entries)} entries",
        details={"trigger_path": trigger_path, "entry_count": len(entries)},
    )


# ---------------------------------------------------------------------------
# Step 3: checkpoint health
# ---------------------------------------------------------------------------

def check_checkpoint_health(
    checkpoint_path: Optional[str] = None,
) -> FlywheelStepStatus:
    """Verify the GT checkpoint exists, is loadable, and has required keys.

    FAIL conditions:
        - Checkpoint does not exist.
        - Checkpoint is not a valid torch save (corrupt).
        - Checkpoint is missing required keys (model_state_dict, known_pairs).
    """
    try:
        import torch
    except ImportError:
        return FlywheelStepStatus(
            step="checkpoint",
            ok=False,
            message="torch not installed — cannot verify checkpoint",
        )

    if checkpoint_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        # Try common locations.
        candidates = [
            repo_root / "graph_transformer" / "checkpoints" / "gt_checkpoint.pt",
            repo_root / "checkpoints" / "gt_checkpoint.pt",
            repo_root / "gt_checkpoint.pt",
        ]
        for c in candidates:
            if c.exists():
                checkpoint_path = str(c)
                break
        if checkpoint_path is None:
            return FlywheelStepStatus(
                step="checkpoint",
                ok=False,
                message=f"no checkpoint found in candidates: {[str(c) for c in candidates]}",
            )

    if not os.path.exists(checkpoint_path):
        return FlywheelStepStatus(
            step="checkpoint",
            ok=False,
            message=f"checkpoint not found at {checkpoint_path}",
            details={"checkpoint_path": checkpoint_path},
        )

    # Check file size (a corrupt checkpoint is often 0 bytes or partial).
    size = os.path.getsize(checkpoint_path)
    if size < 100:
        return FlywheelStepStatus(
            step="checkpoint",
            ok=False,
            message=f"checkpoint is suspiciously small ({size} bytes) — likely corrupt",
            details={"checkpoint_path": checkpoint_path, "size_bytes": size},
        )

    try:
        bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return FlywheelStepStatus(
            step="checkpoint",
            ok=False,
            message=f"checkpoint is CORRUPT — torch.load failed: {exc}",
            details={"checkpoint_path": checkpoint_path, "error": str(exc)},
        )

    if not isinstance(bundle, dict):
        return FlywheelStepStatus(
            step="checkpoint",
            ok=False,
            message=f"checkpoint is not a dict (got {type(bundle).__name__})",
            details={"checkpoint_path": checkpoint_path, "actual_type": type(bundle).__name__},
        )

    required_keys = ["model_state_dict", "known_pairs"]
    missing_keys = [k for k in required_keys if k not in bundle]
    if missing_keys:
        return FlywheelStepStatus(
            step="checkpoint",
            ok=False,
            message=f"checkpoint missing required keys: {missing_keys}",
            details={"checkpoint_path": checkpoint_path, "missing_keys": missing_keys},
        )

    return FlywheelStepStatus(
        step="checkpoint",
        ok=True,
        message=f"checkpoint healthy — {len(bundle.get('known_pairs', []))} known_pairs, size {size} bytes",
        details={
            "checkpoint_path": checkpoint_path,
            "size_bytes": size,
            "known_pairs_count": len(bundle.get("known_pairs", [])),
            "best_val_auc": bundle.get("best_val_auc"),
            "best_epoch": bundle.get("best_epoch"),
        },
    )


# ---------------------------------------------------------------------------
# Step 4: RL ranker bonus-loading health
# ---------------------------------------------------------------------------

def check_rl_ranker_health(
    expected_min_bonus_pairs: int = 0,
) -> FlywheelStepStatus:
    """Verify the RL ranker loaded the expected number of validated pairs.

    FAIL conditions:
        - rl.rl_drug_ranker cannot be imported.
        - _load_validated_hypotheses() raises.
        - Loaded pair count < expected_min_bonus_pairs.

    Args:
        expected_min_bonus_pairs: minimum expected count. Set to 0 to
            skip the count check (just verify the loader runs).
    """
    try:
        import sys
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from rl.rl_drug_ranker import (
            _load_validated_hypotheses,
            _load_validated_toxic_hypotheses,
        )
    except Exception as exc:
        return FlywheelStepStatus(
            step="rl_ranker",
            ok=False,
            message=f"failed to import rl.rl_drug_ranker: {exc}",
        )

    try:
        bonus_pairs = _load_validated_hypotheses()
        toxic_pairs = _load_validated_toxic_hypotheses()
    except Exception as exc:
        return FlywheelStepStatus(
            step="rl_ranker",
            ok=False,
            message=f"RL ranker loader raised: {exc}",
        )

    if len(bonus_pairs) < expected_min_bonus_pairs:
        return FlywheelStepStatus(
            step="rl_ranker",
            ok=False,
            message=(
                f"RL ranker loaded {len(bonus_pairs)} bonus pairs, "
                f"expected >= {expected_min_bonus_pairs}. The ranker "
                f"is NOT picking up new validated hypotheses."
            ),
            details={
                "bonus_pair_count": len(bonus_pairs),
                "toxic_pair_count": len(toxic_pairs),
                "expected_min_bonus_pairs": expected_min_bonus_pairs,
            },
        )

    return FlywheelStepStatus(
        step="rl_ranker",
        ok=True,
        message=f"RL ranker healthy — {len(bonus_pairs)} bonus pairs, {len(toxic_pairs)} toxic pairs loaded",
        details={
            "bonus_pair_count": len(bonus_pairs),
            "toxic_pair_count": len(toxic_pairs),
            "bonus_pairs_sample": list(bonus_pairs)[:5],
            "toxic_pairs_sample": list(toxic_pairs)[:5],
        },
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def run_all_checks(
    csv_path: Optional[str] = None,
    trigger_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    expected_min_bonus_pairs: int = 0,
    staleness_hours: int = FLYWHEEL_STALENESS_HOURS,
) -> List[FlywheelStepStatus]:
    """Run all 4 flywheel health checks and return the statuses.

    The orchestrator (Airflow / cron) calls this and alerts on any
    status with ok=False.
    """
    return [
        check_writeback_health(csv_path=csv_path, staleness_hours=staleness_hours),
        check_retrain_trigger_health(trigger_path=trigger_path),
        check_checkpoint_health(checkpoint_path=checkpoint_path),
        check_rl_ranker_health(expected_min_bonus_pairs=expected_min_bonus_pairs),
    ]


def alert_on_failures(statuses: List[FlywheelStepStatus]) -> int:
    """Log an alert for each failed status. Returns the count of failures.

    In production, replace the logger calls with your alerting system
    (PagerDuty, Slack webhook, Sentry, etc.).
    """
    failures = [s for s in statuses if not s.ok]
    for s in failures:
        logger.error(
            "FLYWHEEL ALERT: step=%s failed: %s | details=%s",
            s.step, s.message, s.details,
        )
    if not failures:
        logger.info("FLYWHEEL OK: all %d steps healthy", len(statuses))
    return len(failures)


__all__ = [
    "FlywheelStepStatus",
    "FLYWHEEL_STALENESS_HOURS",
    "check_writeback_health",
    "check_retrain_trigger_health",
    "check_checkpoint_health",
    "check_rl_ranker_health",
    "run_all_checks",
    "alert_on_failures",
]
