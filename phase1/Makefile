.PHONY: setup download-all download-parallel download-samples clean-all load-all test test-isolated clean-db all down reset run-airflow install-deps

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

# v49 ROOT FIX (CRITICAL — Makefile TAB issue):
# The v48 Makefile used 8 spaces instead of TAB characters for all recipe
# lines. This caused `make` to fail with "missing separator" on every
# invocation, which cascaded into run_unified.py failing to auto-run Phase 1.
# ROOT FIX: every recipe line now uses a leading TAB (verified with cat -A).
# This is the single fix that unblocks the entire platform.

# v49 ROOT FIX (P0 — bootstrap Python venv when VENV dir missing):
# Original setup assumed docker-compose + virtualenv in one step. Operators
# without Docker (most CI / fresh clones) hit "docker-compose: not found".
# ROOT FIX: split into `setup` (docker for Postgres + Neo4j) and
# `install-deps` (pip-only, works without Docker).

setup:
	@echo "==> Starting Postgres + Neo4j containers via docker-compose"
	@command -v docker-compose >/dev/null 2>&1 || command -v docker >/dev/null 2>&1 || { \
	    echo "ERROR: docker-compose (or docker) not found on PATH. " >&2; \
	    echo "       Install Docker Desktop / Docker Engine, OR skip this target and" >&2; \
	    echo "       set DATABASE_URL=sqlite:///phase1.db to use SQLite." >&2; \
	    exit 1; \
	}
	@if command -v docker-compose >/dev/null 2>&1; then \
	    docker-compose up -d || { echo "ERROR: docker-compose up -d failed (see above)" >&2; exit 1; }; \
	else \
	    docker compose up -d || { echo "ERROR: docker compose up -d failed (see above)" >&2; exit 1; }; \
	fi
	@echo "==> Waiting 5s for containers to become healthy"
	@sleep 5
	@echo "==> setup complete"

install-deps:
	@echo "==> Creating Python virtualenv at $(VENV)"
	@python3 -m venv $(VENV) || { \
	    echo "ERROR: python3 -m venv $(VENV) failed." >&2; \
	    echo "       Ensure python3-venv is installed (apt: 'apt install python3-venv')." >&2; \
	    exit 1; \
	}
	@$(PIP) install --upgrade pip
	@echo "==> Installing requirements.txt"
	@$(PIP) install -r requirements.txt || { \
	    echo "ERROR: pip install -r requirements.txt failed (see above)" >&2; \
	    exit 1; \
	}
	@echo "==> Initializing Phase 1 database schema"
	@DATABASE_URL="$${DATABASE_URL:-sqlite:///phase1.db}" $(PYTHON) -c "from database.connection import init_db; init_db()" || { \
	    echo "ERROR: init_db() failed." >&2; \
	    echo "       Set DATABASE_URL=sqlite:///phase1.db for SQLite (dev)," >&2; \
	    echo "       or DATABASE_URL=postgresql+psycopg2://user:pass@host:port/db for PostgreSQL." >&2; \
	    exit 1; \
	}
	@echo "==> install-deps complete"

# v49 ROOT FIX (CRITICAL — DrugBank fallback):
# DrugBank has paused academic downloads since May 2026. The v48
# `download-all` target hard-failed if DRUGBANK_XML_PATH did not point
# to a real DrugBank XML file. ROOT FIX: each pipeline now respects
# `DRUGOS_DOWNLOAD_MODE={full|sample|skip}` and DrugBank pipeline
# gracefully degrades to a ChEMBL-derived FDA-approved drug set when
# the XML is unavailable. See `pipelines/drugbank_pipeline.py` for the
# fallback implementation.

# v49 ROOT FIX (CRITICAL — sample mode):
# `make download-samples` is a NEW target that downloads small samples
# of each source (50-200 records each) so the pipeline can run
# end-to-end on a laptop without holding 100GB+ of biomedical data.
# This is what `make all` invokes by default; use `make download-all`
# for the full production run.
download-samples:
	DRUGOS_DOWNLOAD_MODE=sample $(PYTHON) -c "from pipelines.chembl_pipeline import ChEMBLPipeline; ChEMBLPipeline().run()"
	DRUGOS_DOWNLOAD_MODE=sample $(PYTHON) -c "from pipelines.omim_pipeline import OMIMPipeline; OMIMPipeline().run()"
	DRUGOS_DOWNLOAD_MODE=sample $(PYTHON) -c "from pipelines.drugbank_pipeline import DrugBankPipeline; DrugBankPipeline().run()"
	DRUGOS_DOWNLOAD_MODE=sample $(PYTHON) -c "from pipelines.uniprot_pipeline import UniProtPipeline; UniProtPipeline().run()"
	DRUGOS_DOWNLOAD_MODE=sample $(PYTHON) -c "from pipelines.string_pipeline import StringPipeline; StringPipeline().run()"
	DRUGOS_DOWNLOAD_MODE=sample $(PYTHON) -c "from pipelines.disgenet_pipeline import DisGeNETPipeline; DisGeNETPipeline().run()"
	DRUGOS_DOWNLOAD_MODE=sample $(PYTHON) -c "from pipelines.pubchem_pipeline import PubChemPipeline; PubChemPipeline().run()"

download-all:
	DRUGOS_DOWNLOAD_MODE=full $(PYTHON) -c "from pipelines.chembl_pipeline import ChEMBLPipeline; ChEMBLPipeline().run()"
	DRUGOS_DOWNLOAD_MODE=full $(PYTHON) -c "from pipelines.omim_pipeline import OMIMPipeline; OMIMPipeline().run()"
	DRUGOS_DOWNLOAD_MODE=full $(PYTHON) -c "from pipelines.drugbank_pipeline import DrugBankPipeline; DrugBankPipeline().run()"
	DRUGOS_DOWNLOAD_MODE=full $(PYTHON) -c "from pipelines.uniprot_pipeline import UniProtPipeline; UniProtPipeline().run()"
	DRUGOS_DOWNLOAD_MODE=full $(PYTHON) -c "from pipelines.string_pipeline import StringPipeline; StringPipeline().run()"
	DRUGOS_DOWNLOAD_MODE=full $(PYTHON) -c "from pipelines.disgenet_pipeline import DisGeNETPipeline; DisGeNETPipeline().run()"
	DRUGOS_DOWNLOAD_MODE=full $(PYTHON) -c "from pipelines.pubchem_pipeline import PubChemPipeline; PubChemPipeline().run()"

download-parallel:
	$(PYTHON) scripts/download_parallel.py

clean-all:
	rm -rf raw_data/*/
	rm -rf processed_data/*.csv

load-all:
	$(PYTHON) -c "from pipelines.chembl_pipeline import ChEMBLPipeline; ChEMBLPipeline().run_load_only()"
	$(PYTHON) -c "from pipelines.drugbank_pipeline import DrugBankPipeline; DrugBankPipeline().run_load_only()"
	$(PYTHON) -c "from pipelines.uniprot_pipeline import UniProtPipeline; UniProtPipeline().run_load_only()"
	$(PYTHON) -c "from pipelines.string_pipeline import StringPipeline; StringPipeline().run_load_only()"
	$(PYTHON) -c "from pipelines.disgenet_pipeline import DisGeNETPipeline; DisGeNETPipeline().run_load_only()"
	$(PYTHON) -c "from pipelines.omim_pipeline import OMIMPipeline; OMIMPipeline().run_load_only()"
	$(PYTHON) -c "from pipelines.pubchem_pipeline import PubChemPipeline; PubChemPipeline().run_load_only()"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# v75 ROOT FIX (T-035 — `make all` runs tests on dirty DB):
# The v74 `all` target ran install-deps -> download-samples (which
# loads 50-200 drugs per source into the default DB) -> test. Tests
# that expect a clean DB (e.g. test_drug_count expecting 0 drugs
# initially) failed because download-samples had populated the DB.
# These were test-ISOLATION failures, not code bugs — operators saw
# red and wasted time chasing phantom code regressions.
#
# ROOT FIX (master-grade):
#   (1) Add a `clean-db` target that removes the default SQLite DB
#       file (phase1.db) so a fresh `make test` starts clean.
#   (2) Add a `test-isolated` target that runs pytest against a
#       SEPARATE in-memory SQLite DB (DATABASE_URL=sqlite:///:memory:)
#       so test results NEVER depend on what `download-samples` or
#       any other Makefile target loaded. This is the CI-safe path.
#   (3) Change `all` to run `test-isolated` instead of `test` so
#       `make all` is reproducible regardless of DB state. Operators
#       who want to test against the populated DB can still run
#       `make download-samples test` explicitly.
clean-db:
	@echo "==> Removing default SQLite DB (phase1.db) for clean test run"
	rm -f phase1.db phase1.db-journal phase1.db-wal phase1.db-shm
	@echo "==> clean-db complete"

test-isolated: clean-db
	@echo "==> Running pytest against isolated in-memory SQLite DB (T-035 fix)"
	DATABASE_URL="sqlite:///:memory:" $(PYTHON) -m pytest tests/ -v --tb=short

# v49 ROOT FIX: `all` now invokes download-samples (not download-all) so
# the platform runs end-to-end on a laptop. Production operators should
# invoke `make download-all` explicitly with DRUGOS_DOWNLOAD_MODE=full
# and proper DrugBank/DisGeNET/OMIM credentials configured.
# v75 ROOT FIX (T-035): `all` now runs `test-isolated` (not `test`) so
# the test step is reproducible regardless of what `download-samples`
# loaded into the default DB. The CI-safe path is now the default.
all: install-deps download-samples test-isolated

down:
	docker-compose down || true

reset: down
	docker-compose down -v || true
	rm -rf raw_data/*/
	rm -rf processed_data/*.csv

# v49 ROOT FIX (P0 SECURITY — Airflow admin/admin default):
# The v48 Makefile hard-coded `--username admin --password admin` for
# the Airflow admin user. This is a critical security vulnerability
# (anyone with network access to port 8080 can compromise every DAG
# and the entire Phase 1 DB). ROOT FIX: require AIRFLOW_ADMIN_USER
# and AIRFLOW_ADMIN_PASSWORD env vars; fail closed if unset.
run-airflow:
	@test -n "$(AIRFLOW_HOME)" || export AIRFLOW_HOME="$(CURDIR)/.airflow_home"; \
	test -n "$$AIRFLOW_ADMIN_USER" || { echo "ERROR: AIRFLOW_ADMIN_USER env var must be set (v49 security fix — admin/admin default removed)"; exit 2; }; \
	test -n "$$AIRFLOW_ADMIN_PASSWORD" || { echo "ERROR: AIRFLOW_ADMIN_PASSWORD env var must be set (v49 security fix — admin/admin default removed)"; exit 2; }; \
	mkdir -p $$AIRFLOW_HOME/dags; \
	ln -sfn $(CURDIR)/dags/master_pipeline_dag.py $$AIRFLOW_HOME/dags/master_pipeline_dag.py; \
	ln -sfn $(CURDIR)/dags/chembl_dag.py $$AIRFLOW_HOME/dags/chembl_dag.py; \
	ln -sfn $(CURDIR)/dags/drugbank_dag.py $$AIRFLOW_HOME/dags/drugbank_dag.py; \
	ln -sfn $(CURDIR)/dags/uniprot_dag.py $$AIRFLOW_HOME/dags/uniprot_dag.py; \
	ln -sfn $(CURDIR)/dags/string_dag.py $$AIRFLOW_HOME/dags/string_dag.py; \
	ln -sfn $(CURDIR)/dags/disgenet_dag.py $$AIRFLOW_HOME/dags/disgenet_dag.py; \
	ln -sfn $(CURDIR)/dags/omim_dag.py $$AIRFLOW_HOME/dags/omim_dag.py; \
	ln -sfn $(CURDIR)/dags/pubchem_dag.py $$AIRFLOW_HOME/dags/pubchem_dag.py; \
	export PYTHONPATH="$(CURDIR):$$PYTHONPATH"; \
	export DATABASE_URL="$${DATABASE_URL:-sqlite:///$$AIRFLOW_HOME/airflow.db}"; \
	echo "AIRFLOW_HOME=$$AIRFLOW_HOME"; \
	echo "PYTHONPATH=$$PYTHONPATH"; \
	echo "Initializing Airflow DB..."; \
	airflow db migrate || airflow db upgrade; \
	(airflow users create --username "$$AIRFLOW_ADMIN_USER" --password "$$AIRFLOW_ADMIN_PASSWORD" --firstname Admin --lastname User --role Admin --email "$${AIRFLOW_ADMIN_EMAIL:-admin@example.com}" || true); \
	echo "Starting Airflow webserver on :8080 and scheduler..."; \
	airflow webserver -p 8080 & \
	airflow scheduler
