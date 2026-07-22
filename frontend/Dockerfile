# =============================================================================
# frontend/Dockerfile — Next.js 16 standalone production image
# =============================================================================
# Task 367 ROOT FIX: ``docker-compose.yml`` referenced
# ``frontend/Dockerfile`` for the ``frontend`` service but the file did
# not exist. The frontend container could not be built, so the V1
# dashboard (the only user-facing surface per project docx §7) was
# unreachable in docker-compose deployments.
#
# Multi-stage build:
#   Stage 1 (deps):     install node_modules (cached layer — only rebuilt
#                        when package.json or bun.lock changes).
#   Stage 2 (builder):  run `next build` with standalone output enabled
#                        (next.config.ts output: 'standalone').
#   Stage 3 (runtime):  copy .next/standalone + .next/static + public
#                        into a minimal node:alpine image. Final image
#                        is ~120MB (vs ~1.2GB for a non-standalone build).
#
# Prerequisites:
#   * frontend/next.config.ts must set `output: 'standalone'` (verified
#     in next.config.ts at repo root — already set per the existing
#     package.json `build` script which copies .next/static into the
#     standalone directory).
# =============================================================================
FROM node:20-alpine AS deps

WORKDIR /app

# Copy lockfile + package.json first (cached layer — only invalidated
# when dependencies change). bun.lock is the canonical lockfile per the
# existing repo state; we install with npm for maximum CI compatibility
# (the package-lock.json is also present and tracked).
COPY frontend/package.json frontend/package-lock.json* frontend/bun.lock* ./

# Install deps. Use `npm ci` when package-lock.json exists (frozen,
# reproducible); fall back to `npm install` if only bun.lock is present.
RUN if [ -f package-lock.json ]; then \
        npm ci --no-audit --no-fund; \
    else \
        npm install --no-audit --no-fund; \
    fi

# ──────────────────────────────────────────────────────────────────────
# Stage 2 — Build the Next.js standalone bundle
# ──────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy installed node_modules from deps stage.
COPY --from=deps /app/node_modules ./node_modules

# Copy the rest of the frontend source.
COPY frontend/ ./

# Disable telemetry during build (avoids noise in CI logs).
ENV NEXT_TELEMETRY_DISABLED=1

# Build with standalone output. The package.json `build` script already
# handles copying .next/static and public/ into the standalone dir.
RUN npm run build

# ──────────────────────────────────────────────────────────────────────
# Stage 3 — Runtime image (minimal)
# ──────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS runtime

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Install curl for the docker-compose healthcheck (curl -fsS http://...).
RUN apk add --no-cache curl

# Copy the standalone bundle (includes node_modules + server.js).
COPY --from=builder /app/.next/standalone ./

# Copy static assets + public/ (not bundled in standalone by default).
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

# Non-root user (Node 20 alpine ships with `node` user uid 1000).
USER node

EXPOSE 3000

# Healthcheck hits the Next.js /api/health route (added by RT-011).
HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=60s \
    CMD curl -fsS http://localhost:3000/api/health || exit 1

# Start the standalone Next.js server (no need for `next start` — the
# standalone server.js is a self-contained Node HTTP server).
CMD ["node", "server.js"]
