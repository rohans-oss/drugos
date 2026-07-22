# =============================================================================
# Dockerfile.ml — Multi-phase ML services base image (Phase 3 + Phase 4)
# =============================================================================
# Task 356 + 368 ROOT FIX: ``docker-compose.yml`` referenced ``Dockerfile.ml``
# (top-level) for the ``phase3-trainer`` and ``phase4-rl`` services, but the
# file did not exist. ``docker-compose build`` failed with:
#   failed to compute cache key: "/Dockerfile.ml" not found
#
# This Dockerfile builds a production-ready ML image with:
#   * Python 3.11 (matches CI matrix and Airflow base)
#   * PyTorch CPU build (Docker services are CPU-only; GPU services use a
#     separate Dockerfile.gpu — not needed for V1 launch per project docx §9).
#   * PyTorch Geometric (PyG) — required by graph_transformer/
#   * RDKit — required by Phase 1 fingerprinting + Phase 2 ChemBERTa encoder
#   * Stable-Baselines3 + Gymnasium — required by Phase 4 PPO trainer
#   * FastAPI + Uvicorn — required by scripts/gt_api.py and scripts/rl_api.py
#   * Biopython — required by the literature cross-check (V1 launch gate)
#
# This image is intentionally SHARED by Phase 3 and Phase 4 because:
#   1. Both need torch + PyG (Phase 3 trains the GT, Phase 4 ranks its output)
#   2. Sharing the base saves ~3GB of image storage in the registry
#   3. Sharing ensures both phases agree on torch / numpy / pandas versions
#      (a Phase 3 model trained on torch 2.1 must be loaded by Phase 4 with
#      torch 2.1 — version drift causes silent checkpoint corruption).
#
# Multi-stage build:
#   Stage 1 (builder): compile heavy deps (torch, PyG, RDKit wheels)
#   Stage 2 (runtime): copy only installed site-packages + repo source
# =============================================================================
FROM python:3.11-slim AS builder

# ─── System deps for building Python wheels ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# ─── Python build deps (torch CPU + PyG + scientific stack) ─────────────
# IN-070 ROOT FIX (forensic, root-level — torch/PyG version mismatch):
#   The previous Dockerfile installed ``torch==2.2.2+cpu`` but fetched
#   PyG wheels from ``https://data.pyg.org/whl/torch-2.2.0+cpu.html``
#   (the wheel page for torch 2.2.0, NOT 2.2.2). The ``+pt22cpu`` tag
#   on torch-scatter / torch-sparse means "PyTorch 2.2 CPU" — but the
#   wheel was built against torch 2.2.0's C++ ABI, not 2.2.2's. While
#   torch 2.2.x is ABI-stable within the minor version (so this works
#   in 99% of cases), the 1% edge case is an ``undefined symbol``
#   crash at runtime when a torch_scatter op uses a C++ symbol that
#   changed between 2.2.0 and 2.2.2. This is extremely hard to debug
#   because the import succeeds — the crash only happens when the
#   specific op is called.
#
#   ROOT FIX: align torch and PyG wheels to the EXACT same patch
#   version (2.2.0). The PyG project's official recommendation is to
#   use the EXACT torch version match. We use 2.2.0 (not 2.2.2)
#   because the PyG wheel page is published under the 2.2.0 URL —
#   using 2.2.2 would require checking if a 2.2.2 wheel page exists,
#   which adds a network dependency to the build.
#
#   Downgrading from 2.2.2 to 2.2.0 is safe — both are patch releases
#   within the 2.2 minor line, and the 2.2.0 → 2.2.2 changelog
#   contains only bug fixes (no API changes that affect PyG).
RUN pip install --no-cache-dir --prefix=/install \
    "torch==2.2.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --prefix=/install \
    "torch-geometric==2.5.3" \
    "torch-scatter==2.1.2+pt22cpu" \
    "torch-sparse==0.6.18+pt22cpu" \
    -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

RUN pip install --no-cache-dir --prefix=/install \
    "rdkit==2024.3.5" \
    "pandas==2.1.4" \
    "numpy==1.26.4" \
    "scikit-learn==1.4.2" \
    "networkx==3.2.1" \
    "transformers==4.40.2" \
    "stable-baselines3==2.3.0" \
    "gymnasium==0.29.1" \
    "biopython==1.83" \
    "mlflow==2.15.1" \
    "fastapi==0.110.3" \
    "uvicorn[standard]==0.29.0" \
    "pydantic==2.7.1" \
    "psycopg2-binary==2.9.9" \
    "sqlalchemy==2.0.30" \
    "neo4j==5.20.0" \
    "requests==2.31.0" \
    "rapidfuzz==3.9.0" \
    "python-dotenv==1.0.1" \
    "lxml==5.2.1" \
    "filelock==3.14.0" \
    "pyarrow==15.0.2" \
    "pyyaml==6.0.1" \
    "certifi==2024.2.2" \
    "prometheus-client==0.20.0" \
    "psutil==5.9.8"

# =============================================================================
# Stage 2 — Runtime image
# =============================================================================
FROM python:3.11-slim AS runtime

# ─── Runtime system deps (no compilers — smaller image) ──────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    libxml2 \
    libxslt1.1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ─── Copy installed Python packages from builder ────────────────────────
COPY --from=builder /install /usr/local

# ─── Working directory + non-root user ──────────────────────────────────
WORKDIR /opt/repo
RUN useradd -m -u 1000 drugos && chown -R drugos:drugos /opt/repo
USER drugos

# ─── Default command (overridden by docker-compose per service) ─────────
# The default is a no-op uvicorn invocation that exposes the GT service.
# docker-compose overrides this for phase4-rl (uses scripts.rl_api:app on
# port 8003) and for phase3-trainer (chains run_4phase.py then gt_api).
CMD ["uvicorn", "scripts.gt_api:app", "--host", "0.0.0.0", "--port", "8002"]
