# Unified Autonomous Drug Repurposing Platform — Makefile
# ========================================================
# Single entry point for all 4 phases:
#   Phase 1 (Data Ingestion) → Phase 2 (Knowledge Graph) →
#   Phase 3 (Graph Transformer) → Phase 4 (RL Hypothesis Ranker)

SHELL := /bin/bash
PYTHON ?= python3
PIP    ?= pip3

# R-030 root fix: added run-json run-neo4j run-4phase run-full-platform to .PHONY.
# R-019 root fix: run-pipeline target removed (file renamed to run_4phase.py).
# RT-012 root fix: added run-demo for explicit in-memory CI / demo runs.
# Task 363 root fix: run-json now emits manifest.json as JSON instead of
# calling the non-existent --json flag on run_unified.py.
.PHONY: help install test test-phase1 test-phase2 test-bridge test-all run run-full-platform run-unified run-4phase run-real run-demo dry-run run-json run-neo4j clean

help:
	@echo "Unified Autonomous Drug Repurposing Platform"
	@echo "============================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install         Install Python deps (top-level requirements.txt)"
	@echo ""
	@echo "Run (all 4 phases):"
	@echo "  make run             Full 4-phase run (Phase 1+2+3+4) — DEFAULT"
	@echo "                       (invokes run_4phase.py — the CANONICAL runner per ORCH-003)"
	@echo "  make run-4phase      Explicit alias for make run"
	@echo "  make run-full-platform  DEPRECATED (ORCH-003) — emits warning, same as make run"
	@echo "  make dry-run         Same as make run (alias)"
	@echo ""
	@echo "Run (partial):"
	@echo "  make run-unified     Phase 1+2 (+3+4 via --run-gt-rl flag per ORCH-002)"
	@echo "  make run-real        DEPRECATED (ORCH-003) — emits warning, use make run"
	@echo "  make run-json        Print pipeline manifest as JSON (Task 363)"
	@echo ""
	@echo "Test:"
	@echo "  make test-all        Run ALL tests across both phases + bridge"
	@echo "  make test-bridge     Run ONLY the Phase1<->Phase2 integration tests"
	@echo "  make test-phase1     Run Phase 1 tests only"
	@echo "  make test-phase2     Run Phase 2 tests only"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           Remove __pycache__ and .pytest_cache"

install:
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Dependencies installed. Run 'make run' for the full 4-phase pipeline."

# v100 ROOT FIX (R-016) + ORCH-003 ROOT FIX (v2): the DEFAULT run
# target now invokes run_4phase.py — the CANONICAL 4-phase runner.
dry-run: run

run: run-4phase

run-full-platform:
	@# v113 IN-072 ROOT FIX: deprecated target -- alias for ``make run``.
	@# The previous target invoked ``run_full_platform.py`` (a deprecated
	@# shim). The shim and its 5 sibling copies (3 at root + 3 in
	@# scripts/legacy/) have been deleted -- the canonical runner is
	@# ``run_4phase.py`` per ORCH-003. This alias preserves backward
	@# compat for any CI / external script that calls
	@# ``make run-full-platform``.
	@echo "NOTE: 'make run-full-platform' is DEPRECATED (ORCH-003, IN-072)."
	@echo "      Alias for 'make run' (invokes run_4phase.py)."
	@$(MAKE) --no-print-directory run

run-unified:
	@# v113 IN-072 ROOT FIX: deprecated target -- alias for ``make run``.
	@echo "NOTE: 'make run-unified' is DEPRECATED (ORCH-003, IN-072)."
	@echo "      Alias for 'make run' (invokes run_4phase.py)."
	@$(MAKE) --no-print-directory run

run-4phase:
	@if [ -n "$$DRUGOS_NEO4J_URI" ]; then \
		echo "RT-012: DRUGOS_NEO4J_URI is set — using DrugOSGraphBuilder (persists to Neo4j)"; \
		USE_NEO4J_BUILDER=1 $(PYTHON) run_4phase.py; \
	else \
		echo "RT-012: DRUGOS_NEO4J_URI not set — using RecordingGraphBuilder (in-memory, NOT persisted)"; \
		echo "  To persist the KG to Neo4j: export DRUGOS_NEO4J_URI=bolt://localhost:7687"; \
		echo "  and DRUGOS_NEO4J_USER / DRUGOS_NEO4J_PASSWORD, then re-run 'make run'."; \
		$(PYTHON) run_4phase.py; \
	fi

run-demo:
	@echo "RT-012: run-demo — using RecordingGraphBuilder (in-memory, NOT persisted to Neo4j)"
	$(PYTHON) run_4phase.py

run-real:
	@# v113 IN-072 ROOT FIX: deprecated target -- alias for ``make run``.
	@echo "NOTE: 'make run-real' is DEPRECATED (ORCH-003, IN-072)."
	@echo "      Alias for 'make run' (invokes run_4phase.py)."
	@$(MAKE) --no-print-directory run

run-json:
	@# Task 363 ROOT FIX: the previous target invoked `run_unified.py --json`
	@# but the --json flag does not exist on the canonical runner
	@# (run_4phase.py — see its argparse definitions). ROOT FIX: emit
	@# JSON by reading the manifest.json that run_4phase.py writes to
	@# the output directory. If no manifest exists, instruct the user
	@# to run `make run` first.
	@$(PYTHON) -c "import json,sys,glob,os; \
		out = glob.glob('output_v100/**/manifest.json', recursive=True) \
			or glob.glob('output_v100/manifest.json') \
			or glob.glob('output_v100/*/manifest.json'); \
		if not out: \
			sys.exit('No manifest.json found. Run `make run` first to generate pipeline output.'); \
		with open(out[0]) as f: \
			print(json.dumps(json.load(f), indent=2, default=str))"

run-neo4j:
	@if [ -z "$$DRUGOS_NEO4J_URI" ]; then echo "Set DRUGOS_NEO4J_URI env var first"; exit 1; fi
	USE_NEO4J_BUILDER=1 $(PYTHON) run_4phase.py

test-bridge:
	cd phase2 && $(PYTHON) -m pytest tests/test_phase1_phase2_bridge.py -v

test-phase1:
	cd phase1 && $(PYTHON) -m pytest tests/ -q --ignore=tests/test_disgenet_pipeline_institutional_v389.py

test-phase2:
	cd phase2 && $(PYTHON) -m pytest tests/ -q

test-all: test-bridge test-phase2 test-phase1
	@echo ""
	@echo "All test suites complete."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete."

# v113 IN-096 ROOT FIX: backup restore-test target.
# Run weekly (or before any disaster-recovery drill) to verify backups
# are restorable. Requires staging Postgres + Neo4j instances.
restore-test:
	@echo "Running backup restore-test (v113 IN-096)..."
	$(PYTHON) scripts/restore_test.py
