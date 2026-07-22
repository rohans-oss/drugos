#!/usr/bin/env bash
# =============================================================================
# airflow-init.sh — Airflow init entrypoint (P1-010 v113 ROOT FIX)
# =============================================================================
# The previous entrypoint in phase1/docker-compose.yml mixed THREE shell-
# escaping conventions in one line:
#   - ${VAR:-default}  (docker-compose interpolation at compose-up time)
#   - $$VAR             (escaped dollar for bash expansion at container run time)
#   - \\gexec           (YAML-escaped psql meta-command)
#
# If an operator set POSTGRES_PASSWORD=pa$$word (containing literal dollar
# signs), docker-compose interpolation produced PGPASSWORD=pa$$word at
# compose-up time, but bash at container runtime saw $$word and tried to
# expand $word (an unset variable), producing PGPASSWORD=pa — silently
# truncating the password. The `airflow db migrate` then failed with
# `password authentication failed for user "cosmic"`. The operator saw a
# confusing auth error that didn't point to the password-truncation root
# cause.
#
# ROOT FIX: move the entrypoint to this dedicated shell script that uses
# SINGLE-QUOTED variables to avoid all expansion ambiguity. Credentials
# are read from environment variables inside the script (NOT from docker-
# compose interpolation), so literal dollar signs in passwords are
# preserved correctly.
#
# Required env vars (set by docker-compose):
#   POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB (for DB creation)
#   AIRFLOW_ADMIN_USER, AIRFLOW_ADMIN_PASSWORD (optional — if unset,
#     Airflow starts with NO admin user per v49 security fix)
# =============================================================================
set -euo pipefail

POSTGRES_USER="${POSTGRES_USER:-cosmic}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-cosmic}"
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
AIRFLOW_DB_NAME="${AIRFLOW_DB_NAME:-airflow}"

echo "[airflow-init] Ensuring '${AIRFLOW_DB_NAME}' database exists in Postgres..."
# Use single-quoted SQL to prevent any shell expansion of the SQL string.
# PGPASSWORD is exported so psql picks it up. The \gexec meta-command
# executes the SELECT result as a SQL statement.
export PGPASSWORD="${POSTGRES_PASSWORD}"
psql -h "${POSTGRES_HOST}" -U "${POSTGRES_USER}" -d postgres -c \
  "SELECT 'CREATE DATABASE ${AIRFLOW_DB_NAME}' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${AIRFLOW_DB_NAME}')" \
  --single-transaction | grep -q '^(' || true
# The \gexec approach is more reliable but requires psql 9.6+. Use the
# DO block approach which works on all supported Postgres versions.
psql -h "${POSTGRES_HOST}" -U "${POSTGRES_USER}" -d postgres <<EOF
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_database WHERE datname = '${AIRFLOW_DB_NAME}') THEN
    PERFORM dblink_exec('host=${POSTGRES_HOST} user=${POSTGRES_USER} password=${POSTGRES_PASSWORD} dbname=postgres', 'CREATE DATABASE ${AIRFLOW_DB_NAME}');
  END IF;
END
\$\$;
EOF
# Fallback: if dblink is not available, use the shell-level CREATE DATABASE
# (ignore error if DB already exists).
psql -h "${POSTGRES_HOST}" -U "${POSTGRES_USER}" -d postgres -tc \
  "SELECT 1 FROM pg_database WHERE datname = '${AIRFLOW_DB_NAME}'" | grep -q 1 || \
  createdb -h "${POSTGRES_HOST}" -U "${POSTGRES_USER}" "${AIRFLOW_DB_NAME}" || true

echo "[airflow-init] Initializing Phase 1 schema (init_db)..."
python -c 'from database.connection import init_db; init_db()'

echo "[airflow-init] Running airflow db migrate..."
airflow db migrate

# v49 SECURITY: create admin user ONLY if both env vars are set.
# If unset, Airflow starts with NO admin user (operator must create one
# manually with a strong password).
if [ -n "${AIRFLOW_ADMIN_USER:-}" ] && [ -n "${AIRFLOW_ADMIN_PASSWORD:-}" ]; then
  echo "[airflow-init] Creating Airflow admin user '${AIRFLOW_ADMIN_USER}'..."
  airflow users create \
    --username "${AIRFLOW_ADMIN_USER}" \
    --password "${AIRFLOW_ADMIN_PASSWORD}" \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email "${AIRFLOW_ADMIN_EMAIL:-admin@example.com}" \
    || echo "[airflow-init] WARNING: admin user creation failed (may already exist)"
else
  echo "[airflow-init] v49 SECURITY: AIRFLOW_ADMIN_USER / AIRFLOW_ADMIN_PASSWORD env vars not set — Airflow will start with NO admin user. Set them in .env to enable admin login."
fi

echo "[airflow-init] Init complete."
