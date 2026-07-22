# Autonomous Drug Repurposing Platform

> **Production-ready** | 4-phase ML pipeline | Graph Transformer × PPO RL

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start (Docker)](#quick-start-docker)
4. [Local Development](#local-development)
5. [Running the Pipeline](#running-the-pipeline)
6. [Service URLs](#service-urls)
7. [Environment Variables](#environment-variables)
8. [Architecture](#architecture)
9. [CI / Testing](#ci--testing)
10. [Contributing](#contributing)

---

## Overview

The platform discovers drug-repurposing hypotheses using a 4-phase pipeline:

| Phase | Module | Technology |
|-------|--------|-----------|
| **1** | Data Ingestion (ETL) | Airflow, ChEMBL, DrugBank, UniProt, STRING, DisGeNET |
| **2** | Knowledge Graph | Neo4j, networkx, phase1_bridge |
| **3** | Graph Transformer | PyTorch Geometric, link prediction |
| **4** | RL Ranker | Stable-Baselines3 PPO, gymnasium |

Results are surfaced in a **Next.js 14** dashboard with real-time candidate ranking, safety scoring, and a data-flywheel writeback loop.

---

## Prerequisites

| Requirement | Minimum Version |
|-------------|----------------|
| Python | 3.11+ |
| Node.js | 20+ |
| Docker | 24+ |
| Docker Compose | v2+ |

---

## Quick Start (Docker)

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd autonomous-drug-repurposing-main

# 2. Set required secrets
cp .env.example .env
# Edit .env — set POSTGRES_PASSWORD, NEO4J_PASSWORD, MLFLOW_ADMIN_PASSWORD,
# AIRFLOW__CORE__FERNET_KEY (generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# 3. Start the full stack
docker compose up -d

# 4. Check everything is healthy
docker compose ps
curl http://localhost:3000/api/health
```

The frontend dashboard will be available at **http://localhost:3000**.

> **First run note:** Phase 3 training (80 epochs) takes 60–120 min on CPU.
> The healthcheck `start_period` is set to 1800 s to account for this.
> Use `--gt-epochs 5 --rl-timesteps 100` for a quick smoke test.

---

## Local Development

### Python backend

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Optionally install dev extras
pip install -r requirements-dev.txt
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # configure backend service URLs
npm run dev                         # starts on http://localhost:3000
```

---

## Running the Pipeline

### Full 4-phase run (local)

```bash
# Requires: Phase 1 CSVs in phase1/processed_data/
python run_4phase.py \
  --phase1-dir phase1/processed_data \
  --output-dir rl/ \
  --gt-epochs 80 \
  --rl-timesteps 5000 \
  --seed 42
```

### Phase 1 ETL only (sample data for dev)

```bash
python -m phase1.pipelines samples   # writes demo CSVs to phase1/processed_data/
```

### Individual microservices

```bash
# Phase 1 Dataset Service  (port 8000)
cd phase1 && uvicorn service:app --port 8000

# Phase 2 KG Builder/API   (port 8001)
cd phase2 && uvicorn drugos_graph.kg_api:app --port 8001

# Phase 4 RL Ranker        (port 8003)
cd rl && uvicorn service:app --port 8003
```

---

## Service URLs

| Service | Local URL | Container |
|---------|-----------|-----------|
| Frontend dashboard | http://localhost:3000 | `drugos-frontend` |
| Phase 1 Dataset Service | http://localhost:8000 | `drugos-phase1-service` |
| Phase 2 KG API | http://localhost:8001 | `drugos-phase2-kg` |
| Phase 3 GT Trainer API | http://localhost:8002 | `drugos-phase3-gt` |
| Phase 4 RL Ranker API | http://localhost:8003 | `drugos-phase4-rl` |
| MLflow Tracking UI | http://localhost:5000 | `drugos-mlflow` |
| Airflow Webserver | http://localhost:8080 | `drugos-phase1-airflow` |
| PostgreSQL | localhost:5432 | `drugos-postgres` |
| Neo4j Browser | http://localhost:7474 | `drugos-neo4j` |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in all values. Key variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `POSTGRES_PASSWORD` | PostgreSQL master password | ✅ |
| `NEO4J_PASSWORD` | Neo4j password (min 8 chars) | ✅ |
| `MLFLOW_ADMIN_PASSWORD` | MLflow UI admin password | ✅ |
| `AIRFLOW__CORE__FERNET_KEY` | 32-byte base64 Fernet key | ✅ |
| `JWT_SECRET` | Frontend/backend auth secret | ✅ |
| `DRUGOS_REQUIRE_NEO4J` | Set to `1` to fail-fast if Neo4j unreachable | optional |
| `USE_NEO4J_BUILDER` | Set to `1` to use Neo4j for KG persistence | optional |
| `RL_SEED` | Random seed for RL inference (default: `42`) | optional |
| `RL_RATE_LIMIT` | Rate limit for RL service (default: `100/minute`) | optional |

**Never commit `.env` to version control.** It is listed in `.gitignore`.

---

## Architecture

```
phase1/ → ETL (Airflow DAGs) → processed_data CSVs
            ↓
phase2/ → phase1_bridge → KG builder → Neo4j graph
            ↓
graph_transformer/ → GT trainer → gt_predictions.csv
            ↓
rl/ → PPO ranker → top_candidates_*.csv
            ↓
frontend/ → Next.js dashboard + data flywheel writeback
```

Key design decisions:
- **Single source of truth**: all contracts live in `shared/contracts/`
- **Fail-fast guards**: Neo4j and normalization stats checked at service startup
- **VecNormalize sidecar**: `.vecnormalize.pkl` must exist alongside PPO `.zip` checkpoints
- **Data flywheel**: validated hypotheses are written back to Phase 1 CSV + Neo4j + retrain trigger

---

## CI / Testing

```bash
# Run cross-phase import integration test (no torch/neo4j needed)
pytest tests/test_cross_phase_imports.py -v

# Run phase1 unit tests
pytest phase1/tests/ -v --ignore=phase1/tests/test_dag_structure.py

# Run all RL unit tests (no GPU needed)
pytest rl/tests/ -v -k "not checkpoint"

# Frontend lint + build
cd frontend && npm run lint && npm run build
```

The CI pipeline (`.github/workflows/ci.yml`) runs on every PR:
1. Python lint (ruff + mypy)
2. Frontend lint + build (Next.js)
3. Python unit tests
4. Cross-phase import integration tests
5. Security scan (pip-audit + npm audit)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on:
- How to update shared contracts
- How to add a new data source to Phase 1
- How to update the RL reward function
- PR checklist and code review guidelines
