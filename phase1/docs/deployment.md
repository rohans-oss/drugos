# Phase 1 Deployment Guide

This document describes how to deploy, run, and monitor the Phase 1 data
ingestion pipeline for the Autonomous Drug Repurposing Platform.

## 1. Architecture Overview

Phase 1 ingests data from 7 free public biomedical databases into a
PostgreSQL staging database, orchestrated by Apache Airflow:

| Source | Schedule (UTC) | DAG ID |
|---|---|---|
| ChEMBL | Wed 04:00 | `chembl_pipeline` |
| DrugBank | Mon 03:00 | `drugbank_pipeline` |
| UniProt | Fri 04:00 | `uniprot_pipeline` |
| STRING | Sat 05:00 | `string_pipeline` |
| DisGeNET | Mon 02:00 | `disgenet_pipeline` |
| OMIM | Thu 07:00 | `omim_pipeline` |
| PubChem | Sat 08:00 | `pubchem_pipeline` |
| Master | Sun 02:00 | `drug_repurposing_master` |

The master DAG orchestrates all 7 sources in the correct dependency order
on Sunday at 02:00 UTC. Each source DAG can also be triggered
independently for ad-hoc refreshes.

## 2. Prerequisites

- Docker Engine 24.0+ with Docker Compose v2+
- 8 GB RAM minimum (16 GB recommended for full ChEMBL load)
- 50 GB free disk space (raw data + processed data + Postgres volume)
- API keys (optional but recommended):
  - `DISGENET_API_KEY` — for higher DisGeNET rate limits
  - `OMIM_API_KEY` — for the OMIM API (required for the OMIM pipeline)
- DrugBank XML file (manually positioned at `DRUGBANK_XML_PATH` due to
  licensing — the pipeline skips DrugBank gracefully if the file is
  missing).

## 3. Initial Deployment

### 3.1 Clone and configure

```bash
git clone https://github.com/MANOFHATERS/autonomous-drug-repurposing.git
cd autonomous-drug-repurposing/phase1

# Create a .env file with strong credentials (NEVER commit this).
cat > .env <<EOF
POSTGRES_USER=drug_admin
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_DB=drug_repurposing
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=$(openssl rand -base64 16)
NEO4J_PASSWORD=$(openssl rand -base64 16)
DRUGOS_ENVIRONMENT=production
DISGENET_API_KEY=your_key_here
OMIM_API_KEY=your_key_here
DRUGBANK_XML_PATH=/opt/airflow/raw_data/drugbank/full_database.xml
EOF
chmod 600 .env
```

### 3.2 Build and start the stack

```bash
docker compose build
docker compose up -d
```

The stack includes:
- `postgres` — PostgreSQL 15 (port 5432)
- `airflow-init` — one-shot init container (creates airflow DB + admin user)
- `airflow-webserver` — Airflow UI (port 8080)
- `airflow-scheduler` — Airflow scheduler
- `neo4j` — Neo4j 5.20 knowledge graph DB (ports 7474, 7687)
- `mlflow` — MLflow tracking server (port 5000)
- `setup` — one-shot container that creates raw_data/processed_data dirs

### 3.3 Verify deployment health

Wait for all services to be healthy:

```bash
docker compose ps
# All services should show Status = "healthy"
```

Open the Airflow UI at http://localhost:8080 and log in with
`AIRFLOW_ADMIN_USER` / `AIRFLOW_ADMIN_PASSWORD` from your `.env` file.

Verify all 8 DAGs are registered:
- In the Airflow UI, you should see 8 DAGs listed:
  `chembl_pipeline`, `drugbank_pipeline`, `uniprot_pipeline`,
  `string_pipeline`, `disgenet_pipeline`, `omim_pipeline`,
  `pubchem_pipeline`, `drug_repurposing_master`.
- If any DAG is missing, check the DAG import errors at
  `Admin → DAGs → Import Errors`.

### 3.4 Apply database migrations

The `airflow-init` entrypoint runs `python -c 'from database.connection
import init_db; init_db()'` which applies migrations via the ORM
(`Base.metadata.create_all()`). For production deployments that prefer
SQL-file migrations (e.g. to use a separate migration user with
DDL privileges), apply them manually:

```bash
docker compose exec airflow-scheduler python -m database.migrations.run_migrations
```

Verify migration health:

```bash
docker compose exec airflow-scheduler python -m database.migrations.run_migrations --check
# Expected output:
#   All applied:        True
#   Schema version OK:  True
#   Healthy:            True
#   ORM parity:         True
```

## 4. Triggering a Full Pipeline Run

### 4.1 Via the Airflow UI

1. Navigate to `http://localhost:8080`.
2. Click the "play" button next to `drug_repurposing_master`.
3. Click "Trigger Dag" (no configuration needed for the default run).
4. Click the DAG name to view the Graph view and monitor task progress.

The master DAG runs all 7 source pipelines in the correct dependency
order:
```
download_chembl  ──┐
download_drugbank ─┤→ entity_resolution → load_string
download_uniprot  ─┤                    → load_disgenet
download_string  ──┘                    → load_omim
download_disgenet                        → load_pubchem_enrichment
download_omim
download_pubchem
```

Expected runtime: 4-7 hours on first run (cold cache), 1-2 hours on
subsequent runs (incremental updates).

### 4.2 Via the Airflow CLI

```bash
docker compose exec airflow-scheduler airflow dags trigger drug_repurposing_master
```

### 4.3 Triggering a single source DAG

To refresh a single source (e.g. ChEMBL) without running the full
master pipeline:

```bash
docker compose exec airflow-scheduler airflow dags trigger chembl_pipeline
```

## 5. Monitoring Progress

### 5.1 Airflow UI

- **DAGs view** — overview of all DAGs, last run, next run, status.
- **Graph view** — task-level status for a specific DAG run
  (green = success, red = failed, yellow = running, gray = queued).
- **Task Instance view** — click a task to see logs, XCom values,
  retry history.
- **SLA Misses** — `Browse → SLA Misses` shows tasks that exceeded
  their SLA (5h for master, 3h for source DAGs).

### 5.2 Database queries

Check the pipeline_runs table for run history:

```sql
SELECT source, run_date, status, records_downloaded,
       records_cleaned, records_loaded, duration_seconds
FROM pipeline_runs
ORDER BY run_date DESC
LIMIT 20;
```

Check the schema_version table for applied migrations:

```sql
SELECT version, description, applied_at
FROM schema_version
ORDER BY version;
```

### 5.3 Log files

Airflow task logs are in the Airflow UI (click a task → "Log").
Pipeline-level logs are also written to `phase1/logs/` on the host
(via the volume mount).

### 5.4 Healthchecks

Each Docker service has a healthcheck. Check status with:

```bash
docker compose ps
# All services should show "healthy"
```

If a service is unhealthy:
- `postgres` — check `docker compose logs postgres` for startup errors.
- `airflow-webserver` — check `curl -f http://localhost:8080/health`.
- `airflow-scheduler` — check `docker compose exec airflow-scheduler
  airflow jobs check --job-type SchedulerJob`.
- `neo4j` — check `curl -f http://localhost:7474`.
- `mlflow` — check `curl -f http://localhost:5000/health`.

## 6. Rollback Procedures

### 6.1 Rollback a specific migration

If a migration introduced a regression, use the new `--rollback` CLI flag:

```bash
docker compose exec airflow-scheduler python -m database.migrations.run_migrations \
  --rollback 017_confidence_tier_add_very_strong.sql
```

This executes the `017_confidence_tier_add_very_strong_rollback.sql`
sidecar inside a single transaction. The `schema_version` row is
deleted so `check_migrations()` correctly reports the migration as
no longer applied.

### 6.2 Rollback the last N migrations

```bash
docker compose exec airflow-scheduler python -m database.migrations.run_migrations --down 2
```

This rolls back the 2 most recently applied migrations (in reverse
order). Stops on first failure.

### 6.3 Restore from backup

For catastrophic failures, restore the PostgreSQL volume from a
snapshot:

```bash
docker compose down
# Restore the postgres_data volume from your backup.
docker compose up -d
```

## 7. Troubleshooting

### 7.1 DAG not visible in Airflow UI

1. Check `Admin → DAGs → Import Errors` for Python import failures.
2. Verify the DAG file is in `phase1/dags/` and ends with `.py`.
3. Check the Airflow scheduler is healthy (`docker compose ps`).
4. Manually re-parse: `docker compose exec airflow-scheduler airflow
   dags reserialize`.

### 7.2 Migration drift (ORM vs SQL schema)

If `--check` reports `ORM parity: False`:

```bash
docker compose exec airflow-scheduler python -m database.migrations.run_migrations --check
```

The drift details will name the column and the mismatch. Fix the ORM
(`database/models.py`) or the migration SQL to align, then re-run
`--check`.

### 7.3 Rate limit (HTTP 429)

The retry policy uses exponential backoff (5min → 10min → 20min cap).
If you're still hitting rate limits:
- For DisGeNET: set `DISGENET_API_KEY` in `.env` for higher limits.
- For UniProt: the pipeline enforces 3 req/sec; do not increase.
- For PubChem: the pipeline batches InChIKey lookups (5 per request);
  do not increase.

### 7.4 Circuit breaker stuck open

If a source is permanently "stuck" in OPEN state:
1. Check the breaker log: `grep "circuit_breaker" phase1/logs/*.log`.
2. Verify the external API is healthy (e.g. `curl
   https://www.ebi.ac.uk/chembl/status`).
3. Restart the Airflow worker: `docker compose restart
   airflow-scheduler`.

The P1-028 ROOT FIX auto-releases stuck half-open probes after 5
minutes, so a manual restart should rarely be needed.

## 8. Production Hardening Checklist

Before promoting a deployment to production:

- [ ] All env vars in `.env` use strong, unique values (no defaults).
- [ ] `DRUGOS_ENVIRONMENT=production` is set.
- [ ] `DRUGOS_DEV_ALLOW_DEFAULT_DB` is NOT set (forces explicit DB URL).
- [ ] All 8 DAGs are visible in the Airflow UI.
- [ ] `python -m database.migrations.run_migrations --check` reports
      `Healthy: True` and `ORM parity: True`.
- [ ] Airflow admin user has a strong password (not `admin`).
- [ ] Neo4j password is set (not the default `neo4j`).
- [ ] PostgreSQL volume is backed up (daily snapshot recommended).
- [ ] Airflow metadata DB (`airflow` database) is backed up.
- [ ] Disk usage alerting is configured (50 GB minimum free space).
- [ ] Log retention is configured (Airflow logs rotate after 30 days).
- [ ] The DrugBank XML file is positioned at `DRUGBANK_XML_PATH` and
      is owned by UID 50000 (the airflow user inside the container).

## 9. Related Documentation

- `phase1/README.md` — Phase 1 overview.
- `phase1/CHANGELOG.md` — change history.
- `phase1/database/migrations/` — SQL migration files (001-017).
- `phase1/dags/` — Airflow DAG definitions.
- `phase1/pipelines/` — Source-specific ETL pipeline implementations.
- `phase1/cleaning/` — Data cleaning / normalization / deduplication.
- `phase1/entity_resolution/` — Cross-source entity resolution.
- `phase1/tests/` — Test suite (run with `pytest phase1/tests/`).
