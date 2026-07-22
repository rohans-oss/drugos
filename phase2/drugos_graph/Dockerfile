# =============================================================================
# phase2/drugos_graph/Dockerfile — Phase 2 KG builder service
# =============================================================================
# Task 356 ROOT FIX: ``docker-compose.yml`` referenced
# ``phase2/drugos_graph/Dockerfile`` for the ``phase2-kg-builder`` service
# but the file did not exist. The ``docker-compose build`` step failed
# with ``failed to compute cache key: "phase2/drugos_graph/Dockerfile"
# not found``, which means the entire platform stack (Phase 1 ETL → KG →
# GT → RL → frontend) could not be brought up with a single
# ``docker-compose up`` command — the V1 launch acceptance criterion.
#
# This image builds on top of ``Dockerfile.python-ml`` (the canonical ML
# base image with torch + PyG + RDKit + FastAPI) so the Phase 2 service
# has the same Python / torch versions as Phase 3 and Phase 4 (required
# for the GT checkpoint → RL ranker handoff to work — see the
# docker-compose.yml comment block on version drift).
#
# IN-015 ROOT FIX (Teammate 4, forensic, root-level): the previous
# Dockerfile used ``FROM drugos-python-ml:latest`` — a non-reproducible
# build anti-pattern. Two builds on different days could produce
# different images because ``drugos-python-ml:latest`` may have been
# rebuilt with different transitive deps. This breaks supply-chain
# attestation, makes SBOM diffing impossible, and means a compromised
# rebuild of ``drugos-python-ml:latest`` (e.g. via a CI token leak)
# silently ships to production.
#
# ROOT FIX:
#   1. Pin to a specific version tag (``drugos-python-ml:2.0.0``)
#      matching the pyproject.toml version.
#   2. Accept a ``BASE_IMAGE`` build-arg so CI can override (e.g. to
#      pin by digest ``drugos-python-ml@sha256:...`` in production).
#   3. CI should add a job that verifies the base image digest matches
#      the pinned value (future work — outside this Dockerfile's scope).
#
# Build context: repo root (``dockerfile: phase2/drugos_graph/Dockerfile``
# in docker-compose.yml sets context: .). The build copies the entire
# repo so ``uvicorn phase2.drugos_graph.kg_api:app`` can import sibling
# modules (phase1, graph_transformer) without a separate volume mount.
# =============================================================================
# IN-015 ROOT FIX: build-arg with version-pinned default. Override in
# production with: ``docker build --build-arg BASE_IMAGE=drugos-python-ml@sha256:<digest>``.
ARG BASE_IMAGE=drugos-python-ml:2.0.0
FROM ${BASE_IMAGE}

# Re-declare WORKDIR because the base image sets it but the build context
# resets it to /.
WORKDIR /opt/repo

# Copy only the source directories the KG service needs. Skipping tests /
# docs / .git keeps the image small (~250MB smaller than COPY . .).
COPY --chown=drugos:drugos phase2/ ./phase2/
COPY --chown=drugos:drugos phase1/ ./phase1/
COPY --chown=drugos:drugos graph_transformer/ ./graph_transformer/
COPY --chown=drugos:drugos common/ ./common/
COPY --chown=drugos:drugos scripts/ ./scripts/
COPY --chown=drugos:drugos run_4phase.py run_unified.py ./

# The KG service does NOT train any model — it loads the pre-built KG
# from Neo4j (or the in-memory CSV fallback) and serves /kg/stats,
# /kg/explore, /query, /cypher to the frontend. Default port 8001
# matches docker-compose.yml's port mapping.
ENV KG_SERVICE_PORT=8001 \
    KG_LOG_LEVEL=info \
    PYTHONPATH=/opt/repo:/opt/repo/phase1:/opt/repo/phase2

EXPOSE 8001

# Healthcheck runs against /healthz — the endpoint we added to
# phase2/drugos_graph/kg_api.py specifically for docker-compose.
HEALTHCHECK --interval=15s --timeout=10s --retries=5 --start-period=30s \
    CMD curl -fsS http://localhost:8001/healthz || exit 1

# Default command runs the KG API service. docker-compose.yml overrides
# this with the bridge-build + uvicorn chain (the bridge loads Phase 1
# CSVs into Neo4j before the service starts accepting requests).
CMD ["uvicorn", "phase2.drugos_graph.kg_api:app", "--host", "0.0.0.0", "--port", "8001"]
