# ISSUE OWNERSHIP REGISTRY
# ====================================================================
#
# PURPOSE: Let parallel agents work on DIFFERENT ISSUES without breaking
# each other. NO CI required. NO sequential ordering. Just ONE rule:
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  BEFORE you start work on an issue:                              │
# │    1. git pull origin main                                       │
# │    2. Read this file — find your issue ID in the table below     │
# │    3. If STATUS=AVAILABLE → claim it (see HOW TO CLAIM)          │
# │    4. If STATUS=CLAIMED → DO NOT TOUCH — pick another issue      │
# │    5. If STATUS=DONE → fix merged but NOT verified — see VERIFY  │
# │    6. If STATUS=VERIFIED → fix merged AND tests pass — skip      │
# │    7. If STATUS=REOPENED → was DONE/VERIFIED but tests now fail  │
# │       — re-claim it (AVAILABLE) and do a deeper fix              │
# │    8. If STATUS=WONTFIX → intentionally not fixing — skip        │
# │    9. If your issue is NOT listed → add it to the table first    │
# └──────────────────────────────────────────────────────────────────┘
#
# This is the SINGLE SOURCE OF TRUTH for issue ownership. If two agents
# work on the same issue without checking this file, that's their bug.
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  HOW TO CLAIM AN ISSUE:                                          │
# │    1. git pull origin main                                       │
# │    2. Edit this file: find your issue row, change                │
# │       STATUS from AVAILABLE → CLAIMED                            │
# │    3. Add your AGENT_ID + TIMESTAMP + BRANCH                     │
# │    4. git add ISSUE_OWNERSHIP.md                                 │
# │    5. git commit -m "claim: <ISSUE_ID> by <AGENT_ID>"            │
# │    6. git push origin main                                       │
# │       (if push fails → someone else pushed first → pull + retry) │
# │    7. NOW create your fix branch and start work                  │
# └──────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  WHEN YOU FINISH:                                                │
# │    1. Merge your fix branch to main (as normal)                  │
# │    2. git pull origin main                                       │
# │    3. Change STATUS from CLAIMED → DONE                          │
# │    4. Add the merge commit hash + list of files you touched      │
# │    5. git add ISSUE_OWNERSHIP.md && git commit -m                │
# │       "done: <ISSUE_ID> merged <hash>" && git push               │
# └──────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  ISSUE LIFECYCLE (the partial-fix solution):                     │
# │                                                                  │
# │   AVAILABLE                                                      │
# │     │  (agent claims it)                                         │
# │     ↓                                                            │
# │   CLAIMED  ← agent is actively working                           │
# │     │  (agent merges fix + marks DONE)                           │
# │     ↓                                                            │
# │   DONE  ← fix is merged but NOT YET verified                     │
# │     │  (someone runs `verify`)                                   │
# │     ├──────────────────────┐                                     │
# │     ↓ tests pass            ↓ tests fail                         │
# │   VERIFIED                REOPENED                               │
# │     │                        │                                   │
# │     │                  (agent re-claims)                          │
# │     │                        ↓                                   │
# │     │                      AVAILABLE                              │
# │     │                                                            │
# │     │  (regression detected later)                               │
# │     ↓                                                            │
# │   REOPENED → AVAILABLE → CLAIMED → DONE → ...                    │
# │                                                                  │
# │  KEY INSIGHT: DONE ≠ VERIFIED. A second run MUST verify DONE     │
# │  issues before trusting them. Use:                               │
# │    python scripts/pre_commit_issue_guard.py verify               │
# │    python scripts/pre_commit_issue_guard.py verify P1-004        │
# │    python scripts/pre_commit_issue_guard.py verify --all         │
# └──────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  CONFLICT RESOLUTION (two agents claim same issue):              │
# │    First push wins. Second push fails with non-fast-forward →    │
# │    second agent pulls, sees the claim, picks another issue.      │
# │    Automatic. No human intervention.                             │
# └──────────────────────────────────────────────────────────────────┘
#
# ====================================================================
# AGENT REGISTRY
# ====================================================================
# Add your agent ID here when you start work.
#
# Active agents:
#   (none currently active)
#
# Example:
#   agent-manoj | Manoj | P1-012 consolidation | 2025-07-12T10:00Z
#
# ====================================================================

# ISSUE OWNERSHIP TABLE
# ----------------------
# Columns:
#   ISSUE_ID    — canonical bug ID (P1-001, P2-005, P4-018, FE-001, BUG-#42, etc.)
#   TITLE       — short description
#   PHASE       — 1, 2, 3, 4, or infra
#   STATUS      — AVAILABLE | CLAIMED | DONE | VERIFIED | REOPENED | WONTFIX
#                 AVAILABLE  — no one is working on it, claim it
#                 CLAIMED    — an agent is actively working on it, DO NOT TOUCH
#                 DONE       — fix merged to main, NOT YET verified by tests
#                 VERIFIED   — fix merged AND verification tests pass
#                 REOPENED   — was DONE/VERIFIED but tests now fail — needs re-fix
#                 WONTFIX    — intentionally not fixing (document why in NOTES)
#   AGENT_ID    — who claimed it (or "—" if AVAILABLE/DONE/VERIFIED)
#   BRANCH      — fix branch name
#   CLAIMED_AT  — UTC timestamp when claimed
#   FILES       — files this issue owns (the hook enforces exclusive access)
#   MERGED      — commit hash once merged (or "—")
#   VERIFIED_AT — UTC timestamp when verified (or "—" if not verified)
#   NOTES       — anything other agents should know
#
# Format (pipe-separated for easy parsing):
# ISSUE_ID | TITLE | PHASE | STATUS | AGENT_ID | BRANCH | CLAIMED_AT | FILES | MERGED | VERIFIED_AT | NOTES
# ====================================================================

## PHASE 1 — Data Ingestion & Pipeline Setup

### P1-001 .. P1-024 (v100 forensic root fixes — ALL DONE)

P1-001 | chembl dead second CSV read removed | 1 | VERIFIED | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/pipelines/chembl_pipeline.py | 5287742 | 2026-07-12T00:08Z | —
P1-002 | neo4j schema-qualified table lookup | 1 | VERIFIED | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/exporters/neo4j_exporter.py | 5287742 | 2026-07-12T00:08Z | —
P1-003 | docker-compose 3 new mounts | 1 | VERIFIED | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/docker-compose.yml | 5287742 | 2026-07-12T00:08Z | —
P1-004 | confidence tier Piñero mislabel (sub_weak/weak/strong) | 1 | VERIFIED | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/cleaning/confidence.py,phase1/config/settings.py,phase1/database/models.py,phase1/database/migrations/012_confidence_tier_pinero_alignment.sql,phase1/database/migrations/012_confidence_tier_pinero_alignment_rollback.sql,phase1/pipelines/schema/v1.json,phase1/pipelines/__init__.py,phase1/cleaning/SCHEMA.md,phase1/pipelines/disgenet_pipeline.py | 5287742 | 2026-07-12T00:07Z | 7-site lockstep update
P1-005 | is_homodimer server_default=FALSE | 1 | VERIFIED | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/database/models.py | 5287742 | 2026-07-12T00:08Z | —
P1-006 | _CircuitBreaker.state pure observation | 1 | VERIFIED | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/database/connection.py | 5287742 | 2026-07-12T00:08Z | —
P1-007 | OMIM mapping_key column drives re-map | 1 | VERIFIED | agent-v100+agent-v101 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/cleaning/missing_values.py,phase1/pipelines/omim_pipeline.py | 5287742 | 2026-07-12T00:09Z | merged with v101 lazy-import fix
P1-008 | _score_direction lineage captured | 1 | DONE | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/cleaning/missing_values.py | 5287742 | — | —
P1-009 | pubchem_load wire removed from trigger_phase2 | 1 | VERIFIED | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/dags/master_pipeline_dag.py | 5287742 | 2026-07-12T00:08Z | —
P1-010 | _original_score None>float TypeError fix | 1 | DONE | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/cleaning/missing_values.py | 5287742 | — | —
P1-011 | lambda truthiness simplified | 1 | DONE | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/cleaning/missing_values.py | 5287742 | — | —
P1-012 | normalizer circuit breaker thread safety | 1 | VERIFIED | agent-v100+agent-v101 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/cleaning/normalizer.py,phase1/_circuit_breaker.py | 5287742 | 2026-07-12T00:08Z | merged with v101 _NormalizerCircuitBreaker wrapper
P1-013 | abs() opt-in for protective-association mode | 1 | DONE | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/cleaning/confidence.py | 5287742 | — | —
P1-014 | disease_name per-row idempotency | 1 | DONE | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/cleaning/missing_values.py | 5287742 | — | —
P1-015 | association_type per-row idempotency | 1 | DONE | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/cleaning/missing_values.py | 5287742 | — | —
P1-016 | row-by-row circuit-open marking | 1 | DONE | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/cleaning/missing_values.py | 5287742 | — | —
P1-017 | deduplicator n_nan length mismatch | 1 | DONE | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/cleaning/deduplicator.py | 5287742 | — | —
P1-018 | DisGeNET centered ±50% jitter | 1 | VERIFIED | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/pipelines/disgenet_pipeline.py | 5287742 | 2026-07-12T00:08Z | —
P1-019 | NULL source_id app-level dedup | 1 | DONE | agent-v100+agent-v101 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/database/loaders.py | 5287742 | — | merged with v101 empty-string conversion fix
P1-020 | GDA early return when gene_symbol missing | 1 | DONE | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/database/loaders.py | 5287742 | — | —
P1-021 | stereo collapse WARNING + lineage | 1 | VERIFIED | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/cleaning/deduplicator.py | 5287742 | 2026-07-12T00:08Z | —
P1-022 | ENVIRONMENT eager-read documented | 1 | DONE | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/config/settings.py | 5287742 | — | —
P1-023 | production-default log WARNING level | 1 | DONE | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/config/settings.py | 5287742 | — | —
P1-024 | SYNTH% CHECK LENGTH<=27 cap | 1 | VERIFIED | agent-v100 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | phase1/database/models.py | 5287742 | 2026-07-12T00:08Z | —

### P1-025 .. P1-073 (v93/v101 forensic fixes by parallel agents)

P1-025 | (v93 fix — see v93_FORENSIC_AUDIT_FIX_SUMMARY) | 1 | DONE | agent-v93 | fix/v93-p1-025-to-p1-050-forensic-root-fixes | 2025-07-11T23:47Z | (see v93 docs) | 0d990a2 | — | parallel agent
P1-026 | (v93 fix) | 1 | DONE | agent-v93 | fix/v93-p1-025-to-p1-050-forensic-root-fixes | 2025-07-11T23:47Z | (see v93 docs) | 0d990a2 | — | parallel agent
P1-027 | (v93 fix — is_fda_approved conflation) | 1 | DONE | agent-v93+agent-v101 | fix/v93-p1-025-to-p1-050-forensic-root-fixes | 2025-07-11T23:47Z | phase1/pipelines/chembl_pipeline.py | 0d990a2 | — | merged with v101 fix
P1-028 | empty-string source_id conversion (cleaner) | 1 | DONE | agent-v101 | fix/v101-forensic-root-fixes-20-critical-bugs | 2025-07-11T23:47Z | phase1/database/loaders.py | 8a57783 | — | merged with P1-019
P1-029 | (v93 fix) | 1 | DONE | agent-v93 | fix/v93-p1-025-to-p1-050-forensic-root-fixes | 2025-07-11T23:47Z | (see v93 docs) | 0d990a2 | — | —
P1-030 | (v93 fix) | 1 | DONE | agent-v93 | fix/v93-p1-025-to-p1-050-forensic-root-fixes | 2025-07-11T23:47Z | (see v93 docs) | 0d990a2 | — | —
P1-031..P1-050 | (v93 forensic fixes — see v93 docs) | 1 | DONE | agent-v93 | fix/v93-p1-025-to-p1-050-forensic-root-fixes | 2025-07-11T23:47Z | (see v93 docs) | 0d990a2 | batch
P1-042 | circuit breaker consolidation (partial) | 1 | DONE | agent-v101 | fix/v101-forensic-root-fixes-20-critical-bugs | 2025-07-11T23:47Z | phase1/_circuit_breaker.py,phase1/cleaning/normalizer.py | 8a57783 | — | INCOMPLETE: 4 inline _CircuitBreaker classes still exist (see KNOWN GAPS)
P1-051..P1-073 | (v101/v92 fixes — see respective docs) | 1 | DONE | agent-v101 | fix/v101-forensic-root-fixes-20-critical-bugs | 2025-07-11T23:47Z | (see v101 docs) | 77309ac | batch

## PHASE 2 — Knowledge Graph Construction

P2-001..P2-058 | (Phase 2 forensic fixes — see FORENSIC_AUDIT_FIX_SUMMARY_V29.md) | 2 | DONE | agent-v82-v92 | various | 2025-07-11T23:47Z | (see P2 docs) | various | batch — verify if needed

## PHASE 3 — Graph Transformer

P3-001..P3-030 | (Phase 3 forensic fixes — see v90 ROOT_FIX_SUMMARY) | 3 | DONE | agent-v90 | fix/v90-phase3-forensic-root-fixes-bug1-30 | 2025-07-11T23:47Z | graph_transformer/ | (see v90 docs) | batch

## PHASE 4 — RL Hypothesis Ranker

P4-001..P4-077 | (Phase 4 forensic fixes — see v92_ROOT_FIX_SUMMARY) | 4 | DONE | agent-v92 | fix/v92-p4-049-to-077-forensic-root-fixes | 2025-07-11T23:47Z | rl/ | (see v92 docs) | batch

## FRONTEND

FE-001..FE-040 | (Frontend forensic fixes) | frontend | DONE | agent-v101 | fix/fe-001-to-fe-040-forensic-root-fix-v101 | 2025-07-11T23:47Z | frontend/ | 8a57783 | batch

## INFRA / DEDUP / COORDINATION

DEDUP-001 | remove duplicate migration 012 | infra | DONE | agent-dedup | fix/dedup-migration-012-and-circuit-breaker-audit | 2025-07-12T00:30Z | phase1/database/migrations/012_disgenet_confidence_tier_pinero_v100.sql,phase1/database/migrations/012_disgenet_confidence_tier_pinero_v100_rollback.sql,phase1/database/migrations/004_extend_gda_table_for_389_audit.sql | d2109dd | — | deleted broken duplicate, reverted 004 to original labels
DEDUP-002 | add dedup guard tests | infra | DONE | agent-dedup | fix/dedup-migration-012-and-circuit-breaker-audit | 2025-07-12T00:30Z | phase1/tests/test_dedup_guards.py | d2109dd | — | 4 guard classes
DEDUP-003 | issue-based ownership registry (this file) | infra | DONE | agent-dedup | fix/issue-based-ownership | 2025-07-12T01:30Z | ISSUE_OWNERSHIP.md,scripts/pre_commit_issue_guard.py | (this commit) | — | replaces AGENTS_FILE_OWNERSHIP.md
GAP-001 | consolidate inline _CircuitBreaker classes | infra | AVAILABLE | — | — | — | phase1/database/connection.py,phase1/pipelines/disgenet_pipeline.py,phase1/pipelines/_chembl_http_client.py,phase1/pipelines/base_pipeline.py,phase1/cleaning/__init__.py,phase1/entity_resolution/drug_resolver.py | — | — | 4 inline _CircuitBreaker classes remain after P1-042 partial fix. They should ALL import from phase1/_circuit_breaker.py. test_dedup_guards.py::TestCircuitBreakerConsolidation has 4 xfailed tests that will flip to PASS when this is done.
GAP-002 | fix test_p1_ci_dedup_regression.py failures | infra | AVAILABLE | — | — | — | phase1/tests/test_p1_ci_dedup_regression.py,.github/workflows/ci.yml | — | — | 7 pre-existing test failures (CI workflow drift + circuit breaker consolidation). Either fix the tests to match current state, or fix the code to match the tests.
GAP-003 | verify P2-001..P2-058 are actually fixed | 2 | AVAILABLE | — | — | — | phase2/ | — | — | the table above says DONE but I (agent-dedup) did not verify. Someone should spot-check 3-5 issues.
GAP-004 | verify P4-001..P4-077 are actually fixed | 4 | AVAILABLE | — | — | — | rl/ | — | — | same as GAP-003 for Phase 4.

# ====================================================================
# KNOWN GAPS — issues that need a NEW owner
# ====================================================================
#
# The GAP-xxx issues above are NOT done. Claim one if you have capacity.
# (They are in the main table above so the guard can parse them.)

# ====================================================================
# FILE → ISSUE MAP (auto-derived from the table above)
# ====================================================================
# This section is the inverse mapping — for each file, which issue owns it.
# The pre-commit hook uses THIS section to enforce exclusivity.
#
# If you're about to edit a file, find it here. The issue that owns it
# must be CLAIMED by YOU (or AVAILABLE/DONE with no active claim).
#
# Format: <file_path> → <ISSUE_ID> (STATUS)
# ====================================================================

phase1/pipelines/chembl_pipeline.py          → P1-001 (DONE), P1-027 (DONE)
phase1/exporters/neo4j_exporter.py           → P1-002 (DONE)
phase1/docker-compose.yml                    → P1-003 (DONE)
phase1/cleaning/confidence.py                → P1-004 (DONE), P1-013 (DONE)
phase1/config/settings.py                    → P1-004 (DONE), P1-022 (DONE), P1-023 (DONE)
phase1/database/models.py                    → P1-004 (DONE), P1-005 (DONE), P1-024 (DONE)
phase1/database/migrations/012_confidence_tier_pinero_alignment.sql → P1-004 (DONE)
phase1/database/migrations/012_confidence_tier_pinero_alignment_rollback.sql → P1-004 (DONE)
phase1/pipelines/schema/v1.json              → P1-004 (DONE)
phase1/pipelines/__init__.py                 → P1-004 (DONE)
phase1/cleaning/SCHEMA.md                    → P1-004 (DONE)
phase1/pipelines/disgenet_pipeline.py        → P1-004 (DONE), P1-018 (DONE)
phase1/database/connection.py                → P1-006 (DONE), GAP-001 (AVAILABLE)
phase1/cleaning/missing_values.py            → P1-007 (DONE), P1-008 (DONE), P1-010 (DONE), P1-011 (DONE), P1-014 (DONE), P1-015 (DONE), P1-016 (DONE)
phase1/pipelines/omim_pipeline.py            → P1-007 (DONE)
phase1/dags/master_pipeline_dag.py           → P1-009 (DONE)
phase1/cleaning/deduplicator.py              → P1-017 (DONE), P1-021 (DONE)
phase1/database/loaders.py                   → P1-019 (DONE), P1-020 (DONE), P1-028 (DONE)
phase1/cleaning/normalizer.py                → P1-012 (DONE)
phase1/_circuit_breaker.py                   → P1-012 (DONE), P1-042 (DONE), GAP-001 (AVAILABLE)
phase1/database/migrations/004_extend_gda_table_for_389_audit.sql → IMMUTABLE (do not edit)
phase1/database/migrations/001-011_*.sql     → IMMUTABLE (do not edit)
phase1/tests/test_dedup_guards.py            → DEDUP-002 (DONE)
phase1/tests/test_p1_ci_dedup_regression.py  → GAP-002 (AVAILABLE)
phase1/cleaning/__init__.py                  → GAP-001 (AVAILABLE)
phase1/entity_resolution/drug_resolver.py    → GAP-001 (AVAILABLE)
phase1/pipelines/base_pipeline.py            → GAP-001 (AVAILABLE)
phase1/pipelines/_chembl_http_client.py      → GAP-001 (AVAILABLE)
ISSUE_OWNERSHIP.md                           → (SHARED — anyone can edit this file)
AGENTS_FILE_OWNERSHIP.md                     → (DEPRECATED — superseded by ISSUE_OWNERSHIP.md)

# ====================================================================
# CHANGELOG (most recent first)
# ====================================================================
# 2025-07-12T01:30Z | agent-dedup | Created ISSUE_OWNERSHIP.md
#   Replaces file-based AGENTS_FILE_OWNERSHIP.md with issue-based registry.
#   All P1-001..P1-024 marked DONE. Known gaps listed for claiming.
#   Pre-commit hook updated to enforce issue-based exclusivity.
# 2025-07-12T00:30Z | agent-dedup | Created AGENTS_FILE_OWNERSHIP.md (file-based)
#   Superseded by this file.
