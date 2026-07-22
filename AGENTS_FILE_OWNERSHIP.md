# AGENTS FILE OWNERSHIP MAP
# ====================================================================
#
# ⚠️  DEPRECATED — use ISSUE_OWNERSHIP.md instead  ⚠️
# ====================================================================
# This file-based ownership map has been SUPERSEDED by issue-based
# ownership in ISSUE_OWNERSHIP.md. The new system:
#   - Claims ISSUES (bugs), not files — one issue can own many files
#   - Uses scripts/pre_commit_issue_guard.py (not pre_commit_ownership_guard.py)
#   - Is more accurate because agents fight over bugs, not files
#
# DO NOT use this file for coordination. Read ISSUE_OWNERSHIP.md instead.
# This file is kept only for historical reference and will be deleted
# after one release cycle.
# ====================================================================
#
# (Original content below — kept for reference, DO NOT USE)
# ====================================================================
#
# PURPOSE: Let parallel agents work WITHOUT breaking each other.
# NO CI required. NO sequential ordering required. Just ONE rule:
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  BEFORE you start work:                                          │
# │    1. git pull origin main                                       │
# │    2. Read this file                                             │
# │    3. Find the file/dir you want to touch in the OWNERSHIP table │
# │    4. If STATUS=AVAILABLE → claim it (see HOW TO CLAIM below)    │
# │    5. If STATUS=CLAIMED → DO NOT TOUCH — pick something else     │
# │    6. If STATUS=DONE → already fixed, verify then skip           │
# └──────────────────────────────────────────────────────────────────┘
#
# This file is the SINGLE SOURCE OF TRUTH. If two agents edit the same
# file without checking this file first, that's their bug — not a
# coordination problem. The file is small enough to read in 10 seconds.
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  HOW TO CLAIM A FILE:                                            │
# │    1. git pull origin main (get latest version of THIS file)     │
# │    2. Edit this file: change STATUS from AVAILABLE to CLAIMED    │
# │    3. Add your agent ID + timestamp + branch name                │
# │    4. git add AGENTS_FILE_OWNERSHIP.md && git commit -m          │
# │       "claim: <file> for <bug-id> by <agent-id>"                 │
# │    5. git push origin main (FAST — this file is tiny)            │
# │    6. If push fails (someone else pushed first), pull + retry    │
# │    7. NOW you can start working on the actual fix                │
# └──────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  WHEN YOU FINISH:                                                │
# │    1. Merge your fix branch to main (as normal)                  │
# │    2. git pull origin main                                       │
# │    3. Change STATUS from CLAIMED to DONE                         │
# │    4. Add the commit hash of your merged fix                     │
# │    5. git add + commit + push this file                          │
# │    6. Next agent can see it's done and skip                      │
# └──────────────────────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  CONFLICT RESOLUTION (if two agents claimed the same file):      │
# │    The agent whose push LANDED FIRST wins. The second agent's    │
# │    push will fail — they pull, see the claim, and pick another   │
# │    file. This is automatic — no human intervention needed.       │
# └──────────────────────────────────────────────────────────────────┘
#
# ====================================================================

# OWNERSHIP TABLE
# ----------------
# Format:
#   <file_or_dir> | <bug_ids> | <status> | <agent_id> | <branch> | <timestamp_utc> | <commit_or_note>
#
# STATUS values:
#   AVAILABLE  — no one is working on it, claim it
#   CLAIMED    — an agent is actively working on it, DO NOT TOUCH
#   DONE       — fix is merged to main, verify then skip
#   SHARED     — multiple agents can touch this file (see SHARED FILES rule below)
#
# ====================================================================

## PHASE 1 — Data Ingestion & Pipeline Setup

### phase1/cleaning/ (data cleaning modules)
phase1/cleaning/confidence.py            | P1-004        | DONE     | agent-v100-p1-004 | fix/p1-001-024-forensic-root-fix-v100  | 2025-07-11T23:47Z | merged 5287742 — sub_weak/weak/strong labels
phase1/cleaning/missing_values.py        | P1-007,008,010,011,014,015,016 | DONE | agent-v100-p1-007-016 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | merged 5287742 — OMIM mapping_key re-map, per-row idempotency, circuit-open marking
phase1/cleaning/deduplicator.py          | P1-017,021    | DONE     | agent-v100-p1-017-021 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | merged 5287742 — n_nan recompute, stereo collapse WARNING
phase1/cleaning/normalizer.py            | P1-012        | DONE     | agent-v100-p1-012 + agent-v101-p1-042 | fix/p1-001-024-forensic-root-fix-v100 + fix/v101-forensic-root-fixes-20-critical-bugs | 2025-07-11T23:47Z | merged 5287742 + 8a57783 — _NormalizerCircuitBreaker wrapper + locked legacy fallback
phase1/cleaning/SCHEMA.md                | P1-004        | DONE     | agent-v100-p1-004 | fix/p1-001-024-forensic-root-fix-v100  | 2025-07-11T23:47Z | merged 5287742
phase1/cleaning/__init__.py              | P1-012        | AVAILABLE | — | — | — | NEEDS CONSOLIDATION: still has inline _CircuitBreaker (see _circuit_breaker.py)

### phase1/config/ (settings)
phase1/config/settings.py                | P1-022,023    | DONE     | agent-v100-p1-022 | fix/p1-001-024-forensic-root-fix-v100  | 2025-07-11T23:47Z | merged 5287742 — ENVIRONMENT eager-read docs, WARNING log level
phase1/config/__init__.py                | —             | AVAILABLE | — | — | — | stable, no active fixes

### phase1/database/ (ORM + migrations + connection)
phase1/database/connection.py            | P1-006        | DONE     | agent-v100-p1-006 | fix/p1-001-024-forensic-root-fix-v100  | 2025-07-11T23:47Z | merged 5287742 — state property pure observation
phase1/database/models.py                | P1-004,005,024 | DONE    | agent-v100-p1-005-024 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | merged 5287742 — is_homodimer FALSE, SYNTH length cap, confidence_tier CHECK
phase1/database/loaders.py               | P1-019,020    | DONE     | agent-v100-p1-019-020 + agent-v101-p1-028 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | merged 5287742 — NULL source_id dedup, GDA early return
phase1/database/migrations/001-011_*.sql | (immutable)   | SHARED   | — | — | — | DO NOT EDIT — create 013+ instead. See test_dedup_guards.py
phase1/database/migrations/012_confidence_tier_pinero_alignment.sql | P1-004 | DONE | agent-v100-p1-004 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | merged 5287742 — canonical migration
phase1/database/migrations/012_*_rollback.sql | P1-004   | DONE     | agent-v100-p1-004 | fix/p1-001-024-forensic-root-fix-v100  | 2025-07-11T23:47Z | merged 5287742

### phase1/dags/ (Airflow DAGs)
phase1/dags/master_pipeline_dag.py       | P1-009        | DONE     | agent-v100-p1-009 | fix/p1-001-024-forensic-root-fix-v100  | 2025-07-11T23:47Z | merged 5287742 — pubchem_load wire removed
phase1/dags/*.py (other DAGs)            | —             | AVAILABLE | — | — | — | no active fixes

### phase1/exporters/ (Neo4j export)
phase1/exporters/neo4j_exporter.py       | P1-002        | DONE     | agent-v100-p1-002 | fix/p1-001-024-forensic-root-fix-v100  | 2025-07-11T23:47Z | merged 5287742 — schema-qualified table lookup

### phase1/pipelines/ (7 data source pipelines)
phase1/pipelines/chembl_pipeline.py      | P1-001        | DONE     | agent-v100-p1-001 + agent-v101-bug18 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | merged 5287742 — dead second CSV read removed
phase1/pipelines/disgenet_pipeline.py    | P1-018        | DONE     | agent-v100-p1-018 | fix/p1-001-024-forensic-root-fix-v100  | 2025-07-11T23:47Z | merged 5287742 — centered ±50% jitter
phase1/pipelines/omim_pipeline.py        | P1-007 (canonical map) | DONE | agent-v101-p1-007 | fix/v101-forensic-root-fixes-20-critical-bugs | 2025-07-11T23:47Z | merged — SCORE_BY_MAPPING_KEY is the single source of truth
phase1/pipelines/__init__.py             | P1-004 (docstring) | DONE | agent-v100-p1-004 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | merged 5287742
phase1/pipelines/schema/v1.json          | P1-004        | DONE     | agent-v100-p1-004 | fix/p1-001-024-forensic-root-fix-v100  | 2025-07-11T23:47Z | merged 5287742
phase1/pipelines/base_pipeline.py        | P1-012 (consolidation) | AVAILABLE | — | — | — | NEEDS CONSOLIDATION: still has inline _CircuitBreaker
phase1/pipelines/_chembl_http_client.py  | P1-012 (consolidation) | AVAILABLE | — | — | — | NEEDS CONSOLIDATION: still has inline _CircuitBreaker

### phase1/docker-compose.yml + docker/
phase1/docker-compose.yml                | P1-003        | DONE     | agent-v100-p1-003 + agent-v101-bug19 | fix/p1-001-024-forensic-root-fix-v100 | 2025-07-11T23:47Z | merged 5287742 — ./data ./exporters ./scripts mounts added

### phase1/_circuit_breaker.py (canonical circuit breaker)
phase1/_circuit_breaker.py               | P1-012 (canonical) | DONE | agent-v101-p1-042 | fix/v101-forensic-root-fixes-20-critical-bugs | 2025-07-11T23:47Z | merged — canonical _CircuitBreaker (other modules should import this)

### phase1/entity_resolution/
phase1/entity_resolution/drug_resolver.py | P1-012 (_PubChemCircuitBreaker) | AVAILABLE | — | — | — | NEEDS CONSOLIDATION: _PubChemCircuitBreaker should use canonical

### phase1/tests/ (test files — SHARED, anyone can add new ones)
phase1/tests/                            | —             | SHARED   | — | — | — | Anyone can ADD new test files. Do not EDIT existing tests without claiming.
phase1/tests/test_dedup_guards.py        | dedup guards  | DONE     | agent-dedup | fix/dedup-migration-012-and-circuit-breaker-audit | 2025-07-12T00:30Z | merged d2109dd — guards for parallel-agent drift
phase1/tests/test_p1_ci_dedup_regression.py | CI + dedup | AVAILABLE | — | — | — | has 7 pre-existing failures from drift (CI workflow + circuit breaker consolidation)

## PHASE 2 — Knowledge Graph Construction
phase2/                                  | —             | AVAILABLE | — | — | — | no active fixes from this session

## PHASE 3 — Graph Transformer
graph_transformer/                       | —             | AVAILABLE | — | — | — | no active fixes from this session

## PHASE 4 — RL Hypothesis Ranker
rl/                                      | —             | AVAILABLE | — | — | — | no active fixes from this session

## Frontend (Next.js)
frontend/                                | —             | SHARED   | — | — | — | parallel agents may touch; coordinate via PR comments

## Top-level scripts + docs
run_full_platform.py                     | —             | SHARED   | — | — | — | parallel agents may edit
run_pipeline.py                          | —             | SHARED   | — | — | — | parallel agents may edit
run_real_pipeline.py                     | —             | SHARED   | — | — | — | parallel agents may edit
run_unified.py                           | —             | SHARED   | — | — | — | parallel agents may edit
scripts/pre_commit_ownership_guard.py    | ownership     | DONE     | agent-dedup | fix/agents-ownership-map | 2025-07-12T01:00Z | local pre-commit hook (no CI needed)
AGENTS_FILE_OWNERSHIP.md                 | —             | SHARED   | — | — | — | anyone can edit (it's the coordination file itself)

# ====================================================================
# SHARED FILES RULE
# ====================================================================
# Files marked SHARED can be edited by multiple agents simultaneously.
# To avoid conflicts on SHARED files:
#   1. Pull main BEFORE you start editing
#   2. Make SMALL, FOCUSED changes (don't rewrite whole functions)
#   3. Pull main again RIGHT BEFORE you commit
#   4. If push fails, pull + rebase + resolve conflicts + retry
#   5. NEVER use --force push on SHARED files
#
# If you need to make a LARGE change to a SHARED file (e.g. refactor a
# whole module), CLAIM it first by changing its status from SHARED to
# CLAIMED. Other agents will see the claim and wait.

# ====================================================================
# IMMUTABLE FILES RULE
# ====================================================================
# Files marked with status (immutable) or in phase1/database/migrations/
# with numbers 001-011 must NEVER be edited. If you need to change what
# they do, create a NEW file (e.g. migration 013_*.sql) instead.
# The test_dedup_guards.py suite enforces this.

# ====================================================================
# AGENT REGISTRY
# ====================================================================
# Add your agent ID here when you start work so other agents know who's active.
#
# Active agents:
#   (none currently active — add yours below)
#
# Example:
#   agent-manoj-p1-cleaning | Manoj | 2025-07-12T10:00Z | working on P1-0xx in phase1/cleaning/

# ====================================================================
# CHANGELOG (most recent first)
# ====================================================================
# 2025-07-12T00:30Z | agent-dedup | Created AGENTS_FILE_OWNERSHIP.md
#   Initial ownership map. Reflects state of main as of commit d2109dd.
#   All P1-001..P1-024 fixes from agent-v100 are marked DONE.
#   Known pre-existing drift: 4 inline _CircuitBreaker classes need consolidation.
