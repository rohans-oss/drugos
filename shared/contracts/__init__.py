"""
shared.contracts — canonical schemas shared across Phase 1/2/3/4.

Every consumer of the data flywheel (writeback, trainer, RL ranker, bridge)
imports from this package. NO module should hardcode its own path, column
name, outcome value, edge label, or feature name.

Modules:
    writeback      — validated_hypotheses schema (path, columns, outcomes,
                     edge labels, Neo4j node labels, atomic-write profile).
                     Re-exports from rl.contracts.phase4_schema for the
                     Task 321-335 contract-first architecture.
    feature_names  — canonical 17-column RL feature schema produced by
                     graph_transformer/gt_rl_bridge.py and consumed by
                     rl/rl_drug_ranker.py. Also exposes the 6 canonical
                     RL feature names (FEATURE_GNN_SCORE, etc.) from
                     Task 328.
    urls           — canonical URL contract (Task 327).
"""
from __future__ import annotations
