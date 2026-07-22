"""
shared — cross-phase contracts, tests, docs, and monitoring.

This package is the SINGLE source of truth for schemas, contracts, and
cross-phase integration tests. Phase 1/2/3/4 modules import from here so
that a change to a contract propagates to all consumers atomically.

Subpackages:
    contracts/   — canonical schemas (paths, column names, outcome values,
                   edge labels, feature names). Re-exports common.* for
                   backward compatibility AND rl.contracts.phase4_schema
                   for the Task 321-335 contract-first architecture.
    tests/       — cross-phase integration tests (data flywheel E2E,
                   toxic penalty, checkpoint atomicity, contract
                   consistency).
    docs/        — architectural documentation (data_flywheel.md, etc.).
    monitoring/  — flywheel step monitoring and alerting.

Subpackages (from Task 321-335 contract-first architecture):
    contracts/  — cross-phase contract modules (URLs, feature names,
                  writeback). Each contract is imported by at least two
                  phases.
    tests/      — cross-phase contract consistency tests.
"""
from __future__ import annotations
