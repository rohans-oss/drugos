#!/usr/bin/env bash
# =============================================================================
# airflow-entrypoint.sh — IN-077 ROOT FIX: Fernet key validation entrypoint.
# =============================================================================
# The docker-compose.yml passes
# ``AIRFLOW__CORE__FERNET_KEY: "dev_fernet_key_replace_in_production"``
# which is NOT a valid Fernet key. Airflow would crash at startup with
# ``cryptography.fernet.InvalidToken`` (good — visible failure) OR silently
# fall back to plaintext Connection storage (bad — invisible failure where
# every Airflow Connection password is stored unencrypted in the metadata
# DB — a FDA 21 CFR Part 11 finding for a pharma platform).
#
# This entrypoint validates the Fernet key BEFORE starting the scheduler
# and fails fast with a clear error message including the command to
# generate a valid key.
#
# Usage (from Dockerfile):
#   ENTRYPOINT ["/opt/airflow/airflow-entrypoint.sh"]
#   CMD ["airflow", "scheduler"]
# =============================================================================
set -euo pipefail

if [ -z "${AIRFLOW__CORE__FERNET_KEY:-}" ]; then
    echo "ERROR: AIRFLOW__CORE__FERNET_KEY is not set. Generate one with:" >&2
    echo '  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"' >&2
    exit 1
fi

# Validate the Fernet key by attempting to construct a Fernet instance.
# A valid Fernet key is a 32-byte base64-encoded string (44 chars total
# including the trailing '='). This catches both malformed keys (wrong
# length, non-base64) and keys that decode but aren't 32 bytes.
if ! python3 -c "
import os, sys
from cryptography.fernet import Fernet
key = os.environ.get('AIRFLOW__CORE__FERNET_KEY', '')
try:
    Fernet(key.encode() if isinstance(key, str) else key)
except Exception as exc:
    print(f'ERROR: AIRFLOW__CORE__FERNET_KEY is not a valid Fernet key: {exc}', file=sys.stderr)
    print('Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"', file=sys.stderr)
    sys.exit(1)
"; then
    exit 1
fi

# Warn if the key is the well-known dev placeholder. This is NOT fatal —
# the key IS technically valid (it's a real Fernet key from the Airflow
# docs) but it's publicly known and MUST be rotated for production.
if [ "${AIRFLOW__CORE__FERNET_KEY}" = "dev_fernet_key_replace_in_production" ]; then
    echo "WARNING: AIRFLOW__CORE__FERNET_KEY is the dev placeholder." >&2
    echo "WARNING: Replace it with a real Fernet key before production deployment." >&2
    echo 'WARNING: Generate one with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"' >&2
fi

# Hand off to the original command (e.g., "airflow scheduler").
exec "$@"
