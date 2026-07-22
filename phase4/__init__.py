"""Phase 4 — RL-Driven Hypothesis Ranker & Data Flywheel Writeback.

Team Cosmic / Autonomous Drug Repurposing Platform.

This package implements:
  - The RL agent that ranks drug-disease hypotheses (in rl/ submodule)
  - The writeback module that closes the data flywheel loop (DOCX §10)
    by feeding pharma partner validations back into Phases 1, 2, and 3.

P4-022 ROOT FIX: previously empty (0 bytes). Now exports the public API
and version metadata for provenance tracking.
"""
from __future__ import annotations

__version__ = "4.1.0"
__schema_version__ = "4.1.0"

# Re-export public symbols from writeback.py so callers can do:
#   from phase4 import write_validated_hypothesis
# instead of:
#   from phase4.writeback import write_validated_hypothesis
from phase4.writeback import (
    WRITEBACK_VERSION,
    ValidationOutcome,
    ValidatedHypothesis,
    write_validated_hypothesis,
    writeback_to_phase1,
    writeback_to_phase2,
    writeback_to_phase3,
    list_validated_hypotheses,
)

__all__ = [
    "__version__",
    "__schema_version__",
    "WRITEBACK_VERSION",
    "ValidationOutcome",
    "ValidatedHypothesis",
    "write_validated_hypothesis",
    "writeback_to_phase1",
    "writeback_to_phase2",
    "writeback_to_phase3",
    "list_validated_hypotheses",
]
