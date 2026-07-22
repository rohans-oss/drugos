#!/usr/bin/env bash
# =============================================================================
# mlflow-entrypoint.sh — MLflow server entrypoint with auth (IN-009 v113)
# =============================================================================
# This script:
#   1. Creates the `mlflow` database in Postgres if it doesn't exist
#      (mlflow does NOT auto-create the database).
#   2. Generates the auth config at /tmp/mlflow_auth_config.yaml from the
#      MLFLOW_ADMIN_PASSWORD env var (sha256-hashed). The template at
#      /opt/mlflow/mlflow_auth_config.yaml has a placeholder hash that is
#      ALWAYS overwritten here -- the placeholder is never used at runtime.
#   3. Starts `mlflow server --app-name basic-auth` so the REST API + UI
#      require HTTP Basic Auth for every endpoint.
#
# Required env vars:
#   MLFLOW_ADMIN_PASSWORD   -- plaintext password (sha256-hashed at startup)
#   POSTGRES_USER           -- Postgres user (for DB creation)
#   POSTGRES_PASSWORD       -- Postgres password (for DB creation)
#
# Optional env vars:
#   MLFLOW_ADMIN_USERNAME   -- default: mlflow_admin (must match the config)
# =============================================================================
set -euo pipefail

: "${MLFLOW_ADMIN_PASSWORD:?ERROR: MLFLOW_ADMIN_PASSWORD env var must be set -- IN-009 v113 security fix removed the no-auth default}"
: "${POSTGRES_USER:=drugos}"
: "${POSTGRES_PASSWORD:=drugos_dev_password}"
: "${MLFLOW_ADMIN_USERNAME:=mlflow_admin}"

# --- Step 1: create the mlflow database if it doesn't exist ---------------
echo "[mlflow-entrypoint] Ensuring 'mlflow' database exists in Postgres..."
PGPASSWORD="${POSTGRES_PASSWORD}" psql -h postgres -U "${POSTGRES_USER}" -d postgres -c \
  "SELECT 'CREATE DATABASE mlflow' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\\gexec" \
  || { echo "[mlflow-entrypoint] WARNING: could not create mlflow DB (postgres not ready?). mlflow will retry."; }

# --- Step 2: generate the auth config with the real password hash --------
# The template at /opt/mlflow/mlflow_auth_config.yaml has a placeholder hash.
# We overwrite admin_password_hash with the sha256 of MLFLOW_ADMIN_PASSWORD.
AUTH_CONFIG="/tmp/mlflow_auth_config.yaml"
PASSWORD_HASH="$(python3 -c "import hashlib; print(hashlib.sha256('${MLFLOW_ADMIN_PASSWORD}'.encode()).hexdigest())")"

echo "[mlflow-entrypoint] Generating auth config at ${AUTH_CONFIG}..."
cat > "${AUTH_CONFIG}" <<EOF
authentication: basic_auth
admin_username: ${MLFLOW_ADMIN_USERNAME}
admin_password_hash: ${PASSWORD_HASH}
database_uri: sqlite:////mlflow/auth.db
authorization_function: mlflow.server.auth:authenticate_request_basic_auth
EOF
chmod 600 "${AUTH_CONFIG}"

# --- Step 3: start mlflow server with basic-auth app ---------------------
echo "[mlflow-entrypoint] Starting mlflow server with --app-name basic-auth..."
export MLFLOW_AUTH_CONFIG_PATH="${AUTH_CONFIG}"
exec mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 4 \
  --backend-store-uri "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres/mlflow" \
  --default-artifact-root /mlruns \
  --app-name basic-auth \
  --access-logfile -
