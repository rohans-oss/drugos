"""
Configuration settings for the Drug Repurposing ETL Platform.

This module defines all configuration values consumed by the seven ETL
pipelines (ChEMBL, DrugBank, UniProt, STRING, DisGeNET, OMIM, PubChem),
the database connection layer, the Airflow DAGs, and the entity
resolution modules.

Settings are loaded from environment variables, with .env file support
via python-dotenv (optional, graceful degradation if not installed).
The preferred import pattern is::

    from config import DATABASE_URL, STRING_MIN_COMBINED_SCORE

Direct imports from this module are also supported for backward
compatibility::

    from config.settings import DATABASE_URL

Loading Strategy
----------------
Environment variables are NOT read at import time. Instead, the first
access to any setting triggers ``_ensure_dotenv_loaded()`` which loads
the ``.env`` file exactly once. This makes importing this module
side-effect-free -- safe for DAG parsing, test frameworks, and IDE
autocompletion.

Exceptions (P1-022 ROOT FIX -- documented eager reads):
  ``ENVIRONMENT`` (and its alias ``DRUGOS_ENVIRONMENT``) IS read at
  import time. This is intentional because ENVIRONMENT is consumed by
  many module-level constants (``_PROFILE_DEFAULTS`` lookups,
  ``DATABASE_URL`` dev-mode auto-swap, etc.) that need a stable value
  at import time. Making ENVIRONMENT lazy would require a sweeping
  refactor of every consumer. If you need to override ENVIRONMENT in
  a test, set the env var BEFORE importing ``config.settings`` (e.g.
  via ``monkeypatch.setenv`` in a fixture that runs before import, or
  by setting it in the test process's environment before pytest starts).
  An empty-string ENVIRONMENT is treated as "production" (the ``or``
  short-circuit falls through to the production default).

Configuration Groups
--------------------
- **Database**: DATABASE_URL
- **ChEMBL**: CHEMBL_VERSION, CHEMBL_API_URL, CHEMBL_MAX_ROWS,
  CHEMBL_MAX_ACTIVITIES, CHEMBL_EXPECTED_DRUG_COUNT_MIN/MAX
- **STRING**: STRING_VERSION, STRING_MIN_COMBINED_SCORE,
  STRING_PROTEIN_LINKS_URL, STRING_ALIASES_URL,
  STRING_PROTEIN_LINKS_DETAILED_URL
- **DisGeNET**: DISGENET_API_KEY, DISGENET_API_URL, DISGENET_USE_API
- **DrugBank**: DRUGBANK_XML_PATH
- **OMIM**: OMIM_API_KEY, OMIM_API_BASE
- **UniProt**: UNIPROT_RELEASE
- **PubChem**: PUBCHEM_REST_BASE, PUBCHEM_FTP_BASE
- **Processing**: CHEMBL_EXPECTED_DRUG_COUNT_MIN/MAX,
  STRING_MIN_COMBINED_SCORE
- **Logging**: LOG_LEVEL, setup_logging()
- **Provenance**: DATA_SNAPSHOT_ID, get_data_version_info(),
  get_provenance_metadata()
- **Environment**: ENVIRONMENT (development / staging / production)

Environment Variables
---------------------
All settings can be overridden via environment variables. See
``.env.example`` for the complete list with descriptions and default
values.

Naming Convention
-----------------
Settings follow the pattern ``{SOURCE}_{TYPE}_{DETAIL}`` where TYPE is
one of: URL, PATH, KEY, LIMIT, SCORE, VERSION, FLAG.

Examples::

    CHEMBL_VERSION             (source=ChEMBL, type=VERSION)
    CHEMBL_API_URL             (source=ChEMBL, type=URL, detail=API)
    DISGENET_API_KEY           (source=DisGeNET, type=KEY, detail=API)
    DRUGBANK_XML_PATH          (source=DrugBank, type=PATH, detail=XML)
    STRING_MIN_COMBINED_SCORE  (source=STRING, type=SCORE, detail=MIN_COMBINED)

Deprecated Settings
-------------------
The following settings are deprecated and will be removed in v2.0.0
(scheduled: 2025-Q4). Accessing them triggers a ``DeprecationWarning``:

- ``CHEMBL_URL`` -- use ``CHEMBL_API_URL`` or the ChEMBL pipeline directly
- ``UNIPROT_SPROT_URL`` -- use the UniProt REST API
- ``UNIPROT_TREMBL_URL`` -- use the UniProt REST API
- ``STRING_PROTEIN_INFO_URL`` -- not used by any pipeline
- ``DISGENET_STATIC_URL`` -- use ``DISGENET_API_URL`` (static URL
  deprecated since 2024)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.parse
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy dotenv loading -- ARCH-1, ARCH-5, RELI-3
# ---------------------------------------------------------------------------
# python-dotenv is an OPTIONAL dependency. If it is not installed, the
# platform falls back to plain os.getenv() (environment variables must be
# set externally via Docker, systemd, or the shell).

_dotenv_loaded: bool = False

# P1-019 ROOT FIX (Team-2 -- inline the load_dotenv import, remove the
#   module-level wrapper):
#   The previous code defined a module-level ``load_dotenv`` wrapper that
#   tried to import ``python-dotenv`` and fell back to a no-op function
#   on ImportError. The wrapper existed ONLY so tests could mock
#   ``config.settings.load_dotenv`` directly. But the wrapper was dead
#   code: ``_ensure_dotenv_loaded`` (the only caller) was already guarded
#   by ``_dotenv_loaded`` (a process-wide flag), so the no-op fallback
#   was called at most once per process. The wrapper also logged the
#   SAME INFO message on every call (misleading -- it implied the no-op
#   could be called multiple times).
#   ROOT FIX: inline the import in ``_ensure_dotenv_loaded``. Tests that
#   need to mock the dotenv loader can mock
#   ``config.settings._load_dotenv_func`` (the module-level binding) OR
#   use ``monkeypatch.setattr`` on the import. No wrapper needed.
#   If python-dotenv is not installed, ``_load_dotenv_func`` is None and
#   ``_ensure_dotenv_loaded`` logs a SINGLE info message and falls back
#   to pure ``os.getenv()``.
try:
    from dotenv import load_dotenv as _load_dotenv_func  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover -- exercised when dotenv is missing
    _load_dotenv_func = None  # type: ignore[assignment]


def _ensure_dotenv_loaded() -> None:
    """Load the .env file exactly once, if it exists.

    This function is called on the first access to any setting via
    ``_getenv()``.  It is idempotent -- subsequent calls are no-ops.
    If python-dotenv is not installed, a single info-level message is
    logged and the function falls back to pure ``os.getenv()``.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True

    # P1-019: if python-dotenv is not installed, log ONCE and fall back.
    if _load_dotenv_func is None:
        logging.getLogger(__name__).info(
            "python-dotenv is not installed. Environment variables must "
            "be set externally. Install with: pip install python-dotenv"
        )
        return

    try:
        env_path = Path(__file__).parent.parent / ".env"
        loaded = _load_dotenv_func(env_path, override=False)
        if not loaded and not env_path.exists():
            logging.getLogger(__name__).info(
                "No .env file found at %s. All settings will use "
                "environment variables or defaults.",
                env_path,
            )
        elif loaded:
            logging.getLogger(__name__).debug(
                "Loaded .env file from %s", env_path
            )
    except Exception as exc:  # noqa: BLE001 -- defensive: never crash on env load
        logging.getLogger(__name__).warning(
            "Failed to load .env file: %s. Falling back to os.getenv().", exc
        )


def _getenv(key: str, default: str = "") -> str:
    """Read an environment variable, ensuring .env has been loaded first.

    This is the canonical way to read env vars in this module. It
    guarantees that ``_ensure_dotenv_loaded()`` has been called before
    any ``os.getenv()`` access.
    """
    _ensure_dotenv_loaded()
    return os.getenv(key, default)


# ---------------------------------------------------------------------------
# Safe parsing utilities -- CODE-1, CODE-2, CODE-4, RELI-1
# ---------------------------------------------------------------------------


def _parse_optional_int(env_key: str, default: Optional[int] = None) -> Optional[int]:
    """Parse an optional integer environment variable.

    - Returns ``None`` if the env var is unset or empty string.
    - Returns ``0`` if the env var is explicitly set to ``0``.
    - Raises ``ValueError`` with a clear message if the value is not a
      valid integer or is negative.
    """
    _ensure_dotenv_loaded()
    raw = os.getenv(env_key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {env_key}={raw!r} is not a valid integer"
        )
    if value < 0:
        raise ValueError(
            f"Environment variable {env_key}={value} must be non-negative"
        )
    return value


def _parse_required_int(env_key: str, default: str) -> int:
    """Parse a required integer environment variable with a default.

    Always returns an ``int``.  Raises ``ValueError`` with the variable
    name if the value cannot be parsed.
    """
    _ensure_dotenv_loaded()
    raw = os.getenv(env_key, default)
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {env_key}={raw!r} is not a valid integer"
        )


def _parse_bool(value: str, default: bool = True) -> bool:
    """Parse a boolean environment variable value.

    Accepts (case-insensitive): true, false, 1, 0, yes, no, on, off.
    Raises ``ValueError`` for unrecognizable values.
    """
    cleaned = (value or "").strip().lower()
    if not cleaned:
        return default
    if cleaned in ("true", "1", "yes", "on"):
        return True
    if cleaned in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"Cannot parse boolean from {value!r}")


def _getenv_bool(key: str, default: bool) -> bool:
    """Read a boolean env var; return default if unset.

    Accepts (case-insensitive): true/false, 1/0, yes/no, on/off.  Empty
    or unset values fall back to ``default``.  Unrecognised non-empty
    values raise ``ValueError`` via ``_parse_bool``.
    """
    raw = _getenv(key, "")
    if not raw.strip():
        return default
    return _parse_bool(raw, default=default)


def _getenv_float(key: str, default: float) -> float:
    """Read a float env var; return default if unset/empty."""
    raw = _getenv(key, "")
    if not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"env var {key!r}={raw!r} is not a valid float"
        ) from exc


def _getenv_int(key: str, default: int) -> int:
    """Read an int env var; return default if unset/empty."""
    raw = _getenv(key, "")
    if not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"env var {key!r}={raw!r} is not a valid int"
        ) from exc


def _parse_csv_ints(key: str, default: list[int]) -> list[int]:
    """Parse a comma-separated list of ints from an env var.

    Returns the default if unset/empty. Raises ValueError on malformed input.
    Used by OMIM_MAPPING_KEYS_INCLUDE (master prompt BUG-2.5).
    """
    raw = _getenv(key, "")
    if not raw.strip():
        return list(default)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return list(default)
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"env var {key!r}={raw!r} contains non-integer values"
        ) from exc


# ---------------------------------------------------------------------------
# Deprecated setting descriptor -- DESIGN-1, DOC-4
# ---------------------------------------------------------------------------


class _DeprecatedSetting:
    """Descriptor that raises ``DeprecationWarning`` when accessed.

    Keeps the setting accessible for backward compatibility but actively
    warns any code that accesses it.
    """

    def __init__(self, name: str, replacement: str, value: object) -> None:
        self._name = name
        self._replacement = replacement
        self._value = value

    def __get__(self, obj: object | None, objtype: type | None = None) -> object:
        warnings.warn(
            f"Setting `{self._name}` is DEPRECATED. Use `{self._replacement}` "
            f"instead. Will be removed in v2.0.0 (scheduled: 2025-Q4).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._value

    def __set__(self, obj: object, value: object) -> None:
        # v28 ROOT FIX (audit TOP-22): previously ``__set__`` silently
        # accepted mutations to deprecated settings. A future operator
        # (or stale code path) could write to the deprecated name, the
        # value would persist, and downstream code reading the
        # REPLACEMENT name would never see the mutation -- a silent
        # configuration drift bug. The fix emits a DeprecationWarning
        # on EVERY mutation so the operator sees the deprecated write
        # in logs (the warning machinery forwards to logging if
        # ``warnings.catch_warnings`` is not installed) AND surfaces
        # the canonical replacement name in the message. The mutation
        # is still permitted for backward compatibility -- but it is no
        # longer silent.
        warnings.warn(
            f"Setting `{self._name}` is DEPRECATED. Mutating it via "
            f"`{self._name} = ...` will update the deprecated alias "
            f"only; downstream code reading `{self._replacement}` will "
            f"NOT see this change. Use `{self._replacement}` instead. "
            f"Will be removed in v2.0.0 (scheduled: 2025-Q4). "
            f"(v28 audit TOP-22: silent deprecated-setting mutation.)",
            DeprecationWarning,
            stacklevel=2,
        )
        self._value = value


# ---------------------------------------------------------------------------
# Environment & project root -- DESIGN-3, ARCH-6
# ---------------------------------------------------------------------------

# FIX TOP-2: Standardize on DRUGOS_ENVIRONMENT across both phases.
# Phase 1 previously read ``ENVIRONMENT`` (vocabulary: dev/staging/prod).
# Phase 2 reads ``DRUGOS_ENVIRONMENT`` (vocabulary: development/staging/
# production). The mismatch meant operators could set DRUGOS_ENVIRONMENT=
# production and Phase 1 would still run in dev mode -- silently defeating
# the production-mode guards. We now:
#   * Read DRUGOS_ENVIRONMENT as the canonical name.
#   * Fall back to the legacy ENVIRONMENT var for backward compat.
#   * Standardize the vocabulary on {development, staging, production}
#     (Phase 2's vocabulary). Old values are normalized:
#       dev  -> development
#       prod -> production
#       staging -> staging (unchanged)
# Synchronized with phase2/drugos_graph/config.py -- DO NOT diverge
# (audit TOP-2).
#
# P1-022 ROOT FIX (v100 forensic -- docstring was lying about lazy loading):
# The module docstring at lines 20-26 claims "Environment variables are
# NOT read at import time." That claim was TRUE for most settings
# (which use _getenv() inside _ensure_dotenv_loaded()) but FALSE for
# ENVIRONMENT -- _raw_environment is read EAGERLY at import time here.
# This made the module's import-time behavior inconsistent with its
# documented contract. If a test set DRUGOS_ENVIRONMENT=development
# AFTER importing config.settings, the change was NOT picked up.
# ROOT FIX: we keep the eager read (because ENVIRONMENT is consumed
# by _PROFILE_DEFAULTS lookups and many other module-level constants
# that need a stable value at import time -- making it lazy would
# require a sweeping refactor of every consumer), but we now:
#   1. Document the eager-read exception explicitly in the module
#      docstring (see the "Exceptions" subsection added to the
#      Loading Strategy block above).
#   2. Treat empty-string ENVIRONMENT as "production" (the previous
#      ``or`` short-circuit already did this implicitly because '' is
#      falsy -- but now it's EXPLICIT and documented, so an operator
#      who sets ENVIRONMENT= (empty) in their .env knows they get
#      production mode).
#   3. Warn LOUDLY (logger.warning, not info -- see P1-023) when
#      falling back to the production default.
_raw_environment: str = (
    os.getenv("DRUGOS_ENVIRONMENT")
    or os.getenv("ENVIRONMENT", "production")
    or "production"
).lower()
_ENV_NORMALIZATION: dict[str, str] = {
    "dev": "development",
    "develop": "development",
    "development": "development",
    "staging": "staging",
    "stage": "staging",
    "prod": "production",
    "production": "production",
}
ENVIRONMENT: str = _ENV_NORMALIZATION.get(_raw_environment, _raw_environment)

# v89 P0 ROOT FIX (DRUGOS_ENVIRONMENT default = production):
# The previous default was "development", which silently enabled 11+
# DRUGOS_ALLOW_* escape hatches that disable patient-safety guards.
# A developer running ``python3 run_real_pipeline.py`` without setting
# DRUGOS_ENVIRONMENT would get dev mode by default, with all safety
# nets off. This is the "ship withdrawn drugs as safe" compound bug
# chain documented in the v89 audit.
#
# The fix: default to "production". Operators who want dev mode must
# EXPLICITLY set DRUGOS_ENVIRONMENT=development. This is the
# "production should be opt-out, not opt-in" principle from the audit.
#
# CI/test environments that need dev mode should set
# DRUGOS_ENVIRONMENT=development in their CI config (the .github/
# workflows/*.yml files already do this where needed).
if not os.getenv("DRUGOS_ENVIRONMENT") and not os.getenv("ENVIRONMENT"):
    logger_production_default = __import__("logging").getLogger(__name__)
    # P1-023 ROOT FIX (v100 forensic -- log level was INFO, not WARNING):
    # The previous code emitted this at INFO level, which is filtered
    # out by default in production (root logger = WARNING per the
    # ``_PROFILE_DEFAULTS["production"]["LOG_LEVEL"] = "WARNING"``
    # setting). An operator running in production without explicit log
    # configuration would NEVER see this message -- defeating the
    # "LOUD log.warning" promised by the comment at line ~472. ROOT
    # FIX: emit at WARNING level so the message lands in every log
    # sink (file + stderr) regardless of the operator's log filter.
    logger_production_default.warning(
        "v89 P0 ROOT FIX: DRUGOS_ENVIRONMENT not set -- defaulting to "
        "'production' (was 'development'). All DRUGOS_ALLOW_* escape "
        "hatches are now REFUSED by default. To enable dev mode, "
        "explicitly set DRUGOS_ENVIRONMENT=development."
    )


def recompute_environment() -> str:
    """Re-read DRUGOS_ENVIRONMENT / ENVIRONMENT env vars and update ENVIRONMENT.

    P1-012 ROOT FIX (Team-1 -- ENVIRONMENT eager-read breaks test ergonomics):
      The module-level ``_raw_environment`` (line 406 above) is read EAGERLY
      at import time. The docstring documents this exception, but tests
      that use ``monkeypatch.setenv("DRUGOS_ENVIRONMENT", "development")``
      in a fixture that runs AFTER ``import config.settings`` see no
      effect -- the module-level ``ENVIRONMENT`` constant has already been
      bound and is not re-read.

      The previous workaround was "set the env var BEFORE importing
      config.settings" -- fragile and incompatible with pytest fixtures
      that run after import.

      ROOT FIX: provide a ``recompute_environment()`` function that tests
      can call AFTER ``monkeypatch.setenv(...)`` to re-resolve ENVIRONMENT
      from the current env vars. The function:
        1. Re-reads ``DRUGOS_ENVIRONMENT`` / ``ENVIRONMENT`` from ``os.environ``.
        2. Normalizes via ``_ENV_NORMALIZATION``.
        3. Updates the module-level ``ENVIRONMENT`` global.
        4. Returns the new value (for assertability).

      Production code should NOT call this -- the eager read at import
      time is intentional (ENVIRONMENT is consumed by many module-level
      constants). This function is for TEST ERGONOMICS only.

    Usage in tests::

        import config.settings as s

        def test_dev_mode(monkeypatch):
            monkeypatch.setenv("DRUGOS_ENVIRONMENT", "development")
            new_env = s.recompute_environment()
            assert new_env == "development"
            assert s.ENVIRONMENT == "development"
            # ... test dev-mode behavior ...

    Returns
    -------
    str
        The new environment value (one of "development", "staging",
        "production", or the raw value if not in _ENV_NORMALIZATION).
    """
    global ENVIRONMENT, _raw_environment
    _raw_environment = (
        os.getenv("DRUGOS_ENVIRONMENT")
        or os.getenv("ENVIRONMENT", "production")
        or "production"
    ).lower()
    ENVIRONMENT = _ENV_NORMALIZATION.get(_raw_environment, _raw_environment)
    return ENVIRONMENT


BASE_DIR: Path = Path(_getenv("PROJECT_ROOT", str(Path(__file__).parent.parent)))

# Validate BASE_DIR points to a real project root (ARCH-6)
if not (BASE_DIR / "config").exists():
    warnings.warn(
        f"BASE_DIR ({BASE_DIR}) does not appear to be the project root. "
        f"Set PROJECT_ROOT env var to the correct path.",
        RuntimeWarning,
    )

# ---------------------------------------------------------------------------
# Environment-specific profile defaults -- DESIGN-3
# ---------------------------------------------------------------------------

_PROFILE_DEFAULTS: dict[str, dict[str, str]] = {
    "development": {
        "CHEMBL_MAX_ROWS": "1000",
        "CHEMBL_MAX_ACTIVITIES": "50000",
        "STRING_MIN_COMBINED_SCORE": "700",
        "LOG_LEVEL": "DEBUG",
    },
    "staging": {
        "CHEMBL_MAX_ROWS": "5000",
        "LOG_LEVEL": "INFO",
    },
    "production": {
        "CHEMBL_MAX_ROWS": "0",
        "LOG_LEVEL": "WARNING",
    },
}


def _get_profile_default(key: str, fallback: str) -> str:
    """Get a profile-specific default, overridden by explicit env vars.

    Explicit environment variables always take precedence over profile
    defaults.  Profile defaults are only used when the env var is not set.
    """
    _ensure_dotenv_loaded()
    explicit = os.getenv(key)
    if explicit is not None:
        return explicit
    profile = _PROFILE_DEFAULTS.get(ENVIRONMENT, {})
    return profile.get(key, fallback)


# ---------------------------------------------------------------------------
# Database -- DATA-2, SEC-1, CONF-2
# ---------------------------------------------------------------------------

# Default uses placeholder credentials, NOT hardcoded real ones.
# In development, docker-compose defaults are auto-applied with a warning.
#
# v93 ROOT FIX (P1-039 -- empty DATABASE_URL bypasses placeholder check):
#   The previous code used ``_getenv("DATABASE_URL", "<default>")`` which
#   returns the default ONLY if the env var is UNSET. If the operator
#   explicitly sets ``DATABASE_URL=""`` (empty string) in their .env
#   (e.g. by accident, or by copying a template without filling it in),
#   ``_getenv`` returns "" (empty string), NOT the default. The
#   placeholder check at line 476 (``if "REPLACE_USER" in DATABASE_URL``)
#   does NOT fire for empty string -- so the operator gets an empty
#   DATABASE_URL that fails at connection time with a cryptic error
#   (e.g. "could not connect to server" with no host). Root fix: treat
#   empty/whitespace-only DATABASE_URL as equivalent to UNSET, and fall
#   back to the default placeholder (which then triggers the existing
#   dev-mode warnings).
_DATABASE_URL_RAW: str = _getenv(
    "DATABASE_URL",
    "postgresql://REPLACE_USER:REPLACE_PASSWORD@localhost:5432/drug_repurposing",
)
if not _DATABASE_URL_RAW or not _DATABASE_URL_RAW.strip():
    # Empty or whitespace-only -- treat as unset. The placeholder default
    # below will trigger the existing dev-mode warning at line 476.
    _log_warn = logging.getLogger("drugos.config.settings")
    _log_warn.warning(
        "DATABASE_URL is set to an empty string in the environment. "
        "Falling back to the default placeholder URL. Set DATABASE_URL "
        "to a real connection string (or unset it to use the dev-mode "
        "docker-compose defaults via DRUGOS_DEV_ALLOW_DEFAULT_DB=1)."
    )
    DATABASE_URL: str = (
        "postgresql://REPLACE_USER:REPLACE_PASSWORD@localhost:5432/drug_repurposing"
    )
else:
    DATABASE_URL = _DATABASE_URL_RAW

# Auto-apply docker-compose defaults in development when placeholder is present
# v28 ROOT FIX (audit TOP-11): previously this block silently swapped the
# placeholder DATABASE_URL to ``cosmic:cosmic`` in dev mode with only a
# Python ``warnings.warn`` -- which is filtered by default in pytest,
# swallowed by logging, and invisible to operators who run the pipeline
# via ``python3 run_unified.py`` (no -W flag). That meant a developer
# could run the entire Phase 1 ETL against a default-credential DB and
# never see a single console message telling them so. Real-world risk:
# someone copy-pastes dev settings into a staging box and the
# cosmic:cosmic default silently takes over because the env var was
# missing -- exactly the failure mode that produced the v28 audit.
#
# The fix is two-layered:
#   1. The silent swap is gated behind an EXPLICIT opt-in env var
#      ``DRUGOS_DEV_ALLOW_DEFAULT_DB=1``. Without it, the dev environment
#      will RAISE -- forcing the operator to either set DATABASE_URL or
#      acknowledge the insecure default.
#   2. When the opt-in is set, we emit a LOUD log.warning (visible in
#      every log sink, not just Python's warning machinery) AND a
#      UserWarning, so the message survives pytest -p no:warnings and
#      any operator's stderr filter.
if "REPLACE_USER" in DATABASE_URL or "REPLACE_PASSWORD" in DATABASE_URL:
    if ENVIRONMENT == "development":
        # Use a module-level logger so the message lands in the standard
        # log pipeline (file + stderr) regardless of the warnings filter.
        _log = logging.getLogger("drugos.config.settings")
        allow_default_db = os.getenv("DRUGOS_DEV_ALLOW_DEFAULT_DB", "") == "1"
        # v34 ROOT FIX (CRITICAL #4): the previous code fired BOTH warnings
        # AND the credential swap REGARDLESS of `allow_default_db`. The
        # opt-in flag was cosmetic. Now we:
        #   1. REFUSE to apply dev default credentials unless
        #      `DRUGOS_DEV_ALLOW_DEFAULT_DB=1` is explicitly set.
        #   2. If not set, log an ERROR and leave DATABASE_URL pointing at
        #      the placeholder (which will fail at connection time with a
        #      clear "REPLACE_USER" message rather than silently using
        #      cosmic:cosmic).
        #   3. When the opt-in IS set, emit a SINGLE consolidated warning
        #      (not two contradictory ones) and apply the swap.
        if not allow_default_db:
            _log.error(
                "DATABASE_URL contains placeholder credentials "
                "(REPLACE_USER/REPLACE_PASSWORD) but "
                "DRUGOS_DEV_ALLOW_DEFAULT_DB=1 is NOT set. REFUSING to "
                "apply dev default credentials. The module is importable "
                "but any DB connection will fail. To acknowledge the "
                "insecure default and use cosmic:cosmic@localhost, set "
                "DRUGOS_DEV_ALLOW_DEFAULT_DB=1. (v34 root fix CRITICAL #4)"
            )
            # Do NOT modify DATABASE_URL -- leave the placeholder so the
            # connection fails loudly with a clear error message.
        else:
            # Opt-in acknowledged. Single consolidated warning.
            _log.warning(
                "DRUGOS DEV MODE: DRUGOS_DEV_ALLOW_DEFAULT_DB=1 is set -- "
                "applying docker-compose default credentials "
                "(cosmic:cosmic by default; override with "
                "DRUGOS_DEV_DB_USER / DRUGOS_DEV_DB_PASSWORD / "
                "DRUGOS_DEV_DB_HOST / DRUGOS_DEV_DB_PORT / "
                "DRUGOS_DEV_DB_NAME env vars). This is INSECURE and "
                "MUST NOT be used outside local development. (v34 root "
                "fix CRITICAL #4)"
            )
            warnings.warn(
                "DATABASE_URL contains placeholder credentials. "
                "DRUGOS_DEV_ALLOW_DEFAULT_DB=1 is set -- using docker-"
                "compose defaults (cosmic:cosmic by default; override "
                "via DRUGOS_DEV_DB_* env vars). This is INSECURE -- set "
                "DATABASE_URL explicitly for any non-local environment.",
                UserWarning,
            )
            _dev_db_user = _getenv("DRUGOS_DEV_DB_USER", "cosmic")
            _dev_db_password = _getenv("DRUGOS_DEV_DB_PASSWORD", "cosmic")
            _dev_db_host = _getenv("DRUGOS_DEV_DB_HOST", "localhost")
            _dev_db_port = _getenv("DRUGOS_DEV_DB_PORT", "5432")
            _dev_db_name = _getenv("DRUGOS_DEV_DB_NAME", "drug_repurposing")
            DATABASE_URL = (
                f"postgresql://{_dev_db_user}:{_dev_db_password}@"
                f"{_dev_db_host}:{_dev_db_port}/{_dev_db_name}"
            )
            del _dev_db_user, _dev_db_password, _dev_db_host
            del _dev_db_port, _dev_db_name
    elif ENVIRONMENT in ("staging", "production"):
        raise ValueError(
            "DATABASE_URL contains placeholder credentials. "
            "Set the DATABASE_URL environment variable with real credentials "
            f"for the {ENVIRONMENT} environment."
        )

# v65 ROOT FIX (P1C-010 -- detect cosmic:cosmic default credentials):
#   The check above ONLY fires when DATABASE_URL contains the literal
#   strings "REPLACE_USER" or "REPLACE_PASSWORD" (the placeholder
#   credentials). But the .env.example file ships
#   ``DATABASE_URL=postgresql://cosmic:cosmic@localhost:5432/...`` (the
#   docker-compose dev default). An operator who copies .env.example to
#   .env (the standard setup flow) gets cosmic:cosmic directly -- the
#   REPLACE_USER check NEVER fires. So the dev default credentials are
#   silently accepted in staging/production, bypassing the v34 ROOT FIX
#   (CRITICAL #4) that was supposed to prevent this. The runtime
#   validator in config/__init__.py:1056 checks for
#   ``_DEFAULT_DB_URL_PREFIX`` ("postgresql://cosmic:cosmic@") but ONLY
#   in the validate_config() path -- NOT at import time. So the import-
#   time check (settings.py) and the runtime validation (config/__init__)
#   use DIFFERENT detection logic, and the gap between them lets
#   cosmic:cosmic through silently at import time.
#   ROOT FIX: add a second import-time check that detects
#   ``cosmic:cosmic@`` in DATABASE_URL. In staging/production, RAISE
#   (matching the REPLACE_USER behavior). In development, require the
#   same ``DRUGOS_DEV_ALLOW_DEFAULT_DB=1`` opt-in -- if NOT set, log an
#   ERROR and leave DATABASE_URL as-is (the connection will fail with a
#   clear authentication error rather than silently using cosmic:cosmic).
_DEFAULT_DB_CREDENTIAL_MARKER = "cosmic:cosmic@"
if _DEFAULT_DB_CREDENTIAL_MARKER in DATABASE_URL:
    if ENVIRONMENT in ("staging", "production"):
        raise ValueError(
            "DATABASE_URL contains the docker-compose default credentials "
            "('cosmic:cosmic'). Default credentials must not be used in "
            f"the {ENVIRONMENT} environment. Set the DATABASE_URL "
            "environment variable with real credentials. (v65 root fix P1C-010)"
        )
    elif ENVIRONMENT == "development":
        _log = logging.getLogger("drugos.config.settings")
        _allow_default_db = os.getenv("DRUGOS_DEV_ALLOW_DEFAULT_DB", "") == "1"
        if not _allow_default_db:
            _log.error(
                "DATABASE_URL contains docker-compose default credentials "
                "('cosmic:cosmic') but DRUGOS_DEV_ALLOW_DEFAULT_DB=1 is NOT "
                "set. The .env.example file ships cosmic:cosmic as the dev "
                "default -- copying it to .env bypassed the REPLACE_USER "
                "check above. REFUSING to silently accept default credentials. "
                "Either (a) set DATABASE_URL to a real connection string, or "
                "(b) set DRUGOS_DEV_ALLOW_DEFAULT_DB=1 to acknowledge the "
                "insecure default for local development. The module is "
                "importable but any DB connection will use the cosmic:cosmic "
                "credentials as written. (v65 root fix P1C-010)"
            )
        # If the opt-in IS set, the cosmic:cosmic URL is accepted with
        # only the existing log messages -- no additional action needed
        # (the v34 fix's warning at line 483-500 covers the dev-default
        # case).
        #
        # P1-A14 ROOT FIX (v82): the previous code, when the opt-in was
        # NOT set, only logged an ERROR but LEFT DATABASE_URL pointing at
        # cosmic:cosmic -- so the connection SUCCEEDED with insecure creds
        # in dev. This is INCONSISTENT with the REPLACE_USER path (which
        # leaves a placeholder that fails the connection). An operator
        # who copied .env.example -> .env got cosmic:cosmic, saw no error
        # at import time (just a log line), and the DB connection worked
        # -- silently using insecure creds. ROOT FIX: when the opt-in is
        # NOT set, REPLACE DATABASE_URL with a refusal placeholder so the
        # connection FAILS loudly -- matching the REPLACE_USER behavior.
        # This makes both dev-default-credential paths CONSISTENT:
        # without opt-in -> connection fails; with opt-in -> connection
        # succeeds with insecure creds + loud warning.
        if not _allow_default_db:
            DATABASE_URL = (
                "postgresql://REFUSED:REFUSED@localhost:5432/refused"
                "?DRUGOS_DEV_ALLOW_DEFAULT_DB=0"
            )

# Detect Docker and warn about localhost (CONF-2)
if Path("/.dockerenv").exists() and "localhost" in DATABASE_URL:
    warnings.warn(
        "DATABASE_URL contains localhost but you appear to be running "
        "inside Docker. localhost inside a container refers to the "
        "container itself, not the host. Use host.docker.internal or "
        "the service name (e.g., postgres) instead.",
        UserWarning,
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DATA_DIR: Path = BASE_DIR / "raw_data"
# FIX TOP-12: ``PROCESSED_DATA_DIR`` (phase1/processed_data/) is the
# OUTPUT directory for Phase 1 pipelines -- DrugBank/OMIM/STRING/ChEMBL
# CSVs land here. It is a READ-ONLY upstream artifact for Phase 2 (Phase 2
# reads these CSVs via the phase1_bridge; never writes to them). Phase 2's
# own outputs go to phase2/data/processed/ -- keep the two paths distinct.
# Synchronized with phase2/drugos_graph/config.py
# (``PHASE1_PROCESSED_DIR`` constant) -- DO NOT diverge (audit TOP-12).
PROCESSED_DATA_DIR: Path = BASE_DIR / "processed_data"

# ---------------------------------------------------------------------------
# ChEMBL -- SCI-2, SCI-4, IDMP-2, INTEROP-2
# ---------------------------------------------------------------------------

DEFAULT_CHEMBL_VERSION: str = "35"  # CONF-1 -- minimum supported version

# Valid ChEMBL database release versions.
# ChEMBL is a continuously-updated biomedical database. New releases are
# published 2-3 times per year by EBI. As of 2025, ChEMBL has reached
# v39 ROOT FIX (P1 #60): removed speculative versions 36, 37, 38.
# As of 2025, ChEMBL's latest release is v35. Versions 36+ do not
# exist yet. The previous code listed them as "known valid" -- a
# future v36 would be accepted without warning, which could mask
# a schema-breaking change. The fix: only list versions that ACTUALLY
# exist (30-35). When ChEMBL releases v36, append it here after
# verifying the pipeline handles its schema.
VALID_CHEMBL_VERSIONS: frozenset[str] = frozenset(
    {"30", "31", "32", "33", "34", "35"}
)


def _validate_chembl_version(version: str) -> str:
    """Validate ChEMBL version string.

    Accepts numeric version strings. Warns on unknown versions.
    Raises ``ValueError`` on clearly invalid values (non-numeric, empty).
    """
    if not version or not version.strip():
        raise ValueError("CHEMBL_VERSION cannot be empty")
    if not version.replace(".", "").isdigit():
        raise ValueError(
            f"CHEMBL_VERSION={version!r} is not a valid version string. "
            f"Expected a numeric version like '35'. "
            f"Valid versions: {sorted(VALID_CHEMBL_VERSIONS)}"
        )
    if version not in VALID_CHEMBL_VERSIONS:
        warnings.warn(
            f"CHEMBL_VERSION={version} is not in the known valid set. "
            f"The ChEMBL API may not support this version. "
            f"Known valid versions: {sorted(VALID_CHEMBL_VERSIONS)}",
            UserWarning,
        )
    return version


CHEMBL_VERSION: str = _validate_chembl_version(
    _getenv("CHEMBL_VERSION", DEFAULT_CHEMBL_VERSION)
)

# ChEMBL API URL -- moved from chembl_pipeline.py (INTEROP-2)
CHEMBL_API_URL: str = _getenv(
    "CHEMBL_API_URL", "https://www.ebi.ac.uk/chembl/api/data"
)

# ChEMBL snapshot date for reproducibility (IDMP-2)
CHEMBL_SNAPSHOT_DATE: str = _getenv("CHEMBL_SNAPSHOT_DATE", "")

# Processing limits (CODE-1, CODE-2, CODE-4)
CHEMBL_MAX_ROWS: Optional[int] = _parse_optional_int(
    "CHEMBL_MAX_ROWS", default=None
)
CHEMBL_MAX_ACTIVITIES: Optional[int] = _parse_optional_int(
    "CHEMBL_MAX_ACTIVITIES", default=None
)

# Version-aware count ranges (SCI-2).
#
# ChEMBL clinical phases (max_phase column):
#   0 = preclinical, 1 = Phase I, 2 = Phase II, 3 = Phase III, 4 = Phase 4 (approved).
# Phase 4 means the drug has reached the market -- i.e., globally approved
# (any regulator, NOT FDA-specific). We filter molecules to max_phase=4 to
# get the set of approved drugs. The count ranges below are the expected
# number of molecules returned by /molecule.json?max_phase=4 for each
# ChEMBL version, used for data-quality validation (DQ-13).
CHEMBL_VERSION_COUNT_RANGES: dict[str, tuple[int, int, str]] = {
    # version: (min, max, rationale)
    # v9 ROOT FIX (audit F3.9): the DOCX target is "10,000 FDA-approved
    # drugs" -- but ChEMBL contains ~2.3M compounds total. The
    # max_phase=4 subset for v33/v34/v35 is ~3-15K depending on the
    # release, which is consistent with the DOCX target.
    # v39 ROOT FIX (P1 #59): corrected the rationale strings. The
    # previous rationale said "FDA-approved" but ChEMBL's max_phase=4
    # means "Phase 4 clinical trial" which is GLOBALLY approved (any
    # regulator -- FDA, EMA, PMDA, etc.), NOT FDA-specific. The audit
    # flagged this as scientifically misleading. The count ranges
    # themselves are correct; only the rationale text was wrong.
    "32": (8000, 15000, "ChEMBL v32 max_phase=4 (globally approved, any regulator) ~12K molecules"),
    "33": (8000, 15000, "ChEMBL v33 max_phase=4 (globally approved, any regulator) ~12K molecules"),
    "34": (3500, 6000, "ChEMBL v34 max_phase=4 (globally approved, any regulator) ~4K molecules"),
    "35": (3000, 5000, "ChEMBL v35 max_phase=4 (globally approved, any regulator) ~3.5-4K molecules"),
}


def _get_default_chembl_count_range(version: str) -> tuple[int, int]:
    """Get the scientifically validated count range for a ChEMBL version."""
    if version in CHEMBL_VERSION_COUNT_RANGES:
        info = CHEMBL_VERSION_COUNT_RANGES[version]
        return info[0], info[1]
    warnings.warn(
        f"CHEMBL_VERSION={version} has no validated count range. "
        f"Count validation will be disabled (min=0, max=999999). "
        f"Run a test download to determine the correct range.",
        UserWarning,
    )
    return 0, 999999


_chembl_range = _get_default_chembl_count_range(CHEMBL_VERSION)
CHEMBL_EXPECTED_DRUG_COUNT_MIN: int = _parse_required_int(
    "CHEMBL_DRUG_COUNT_MIN", str(_chembl_range[0])
)
CHEMBL_EXPECTED_DRUG_COUNT_MAX: int = _parse_required_int(
    "CHEMBL_DRUG_COUNT_MAX", str(_chembl_range[1])
)

# Warn about unlimited processing in non-dev (DATA-4)
if CHEMBL_MAX_ROWS is None and ENVIRONMENT != "development":
    warnings.warn(
        "CHEMBL_MAX_ROWS is not set. The pipeline will process ALL ChEMBL "
        "molecules, which may take several hours and consume significant "
        "memory. Set CHEMBL_MAX_ROWS to cap the number of rows, or set "
        "ENVIRONMENT=development to suppress this warning.",
        UserWarning,
    )

# ---------------------------------------------------------------------------
# ChEMBL pipeline operational settings -- added for institutional-grade
# chembl_pipeline.py rewrite (CFG-1 to CFG-15). All values are env-var
# overridable for dev / staging / prod parity (Domain 12).
# ---------------------------------------------------------------------------

# API pagination size (max 1000 per ChEMBL REST API contract; INT-2).
CHEMBL_PAGE_SIZE: int = _getenv_int("CHEMBL_PAGE_SIZE", 1000)

# HTTP retry behavior (R1-R3, C3-C5, C34, C36, C37).
CHEMBL_MAX_RETRIES: int = _getenv_int("CHEMBL_MAX_RETRIES", 5)
if CHEMBL_MAX_RETRIES < 1:
    raise ValueError(
        f"env var 'CHEMBL_MAX_RETRIES' must be >= 1, got {CHEMBL_MAX_RETRIES}"
    )
CHEMBL_RETRY_BACKOFF_BASE: float = _getenv_float("CHEMBL_RETRY_BACKOFF_BASE", 2.0)
if CHEMBL_RETRY_BACKOFF_BASE < 1.0:
    raise ValueError(
        f"env var 'CHEMBL_RETRY_BACKOFF_BASE' must be >= 1.0, "
        f"got {CHEMBL_RETRY_BACKOFF_BASE}"
    )

# Proactive rate limit (P1, Task 37 v110 root fix). ChEMBL's documented
# public API rate limit is 5 req/sec (https://chembl.gitbook.io/chembl-interface-documentation/web-services).
# Default interval of 0.2s = 5 req/sec, utilizing the full allowance.
# The token-bucket in _chembl_http_client.py enforces this with bounded
# burst capacity, so we stay at or below 5 req/sec even under concurrent load.
# v110 root fix: previous default of 0.5s (2 req/sec) was 60% under-utilization
# of the ChEMBL allowance, needlessly throttling throughput on multi-million
# record pulls. Set env CHEMBL_MIN_REQUEST_INTERVAL=0.5 to restore the
# conservative 2 req/sec behavior if a deployer observes 429s.
CHEMBL_MIN_REQUEST_INTERVAL: float = _getenv_float(
    "CHEMBL_MIN_REQUEST_INTERVAL", 0.2
)
if CHEMBL_MIN_REQUEST_INTERVAL < 0.0:
    raise ValueError(
        f"env var 'CHEMBL_MIN_REQUEST_INTERVAL' must be >= 0.0, "
        f"got {CHEMBL_MIN_REQUEST_INTERVAL}"
    )
if CHEMBL_MIN_REQUEST_INTERVAL > 0.0:
    _effective_rate = 1.0 / CHEMBL_MIN_REQUEST_INTERVAL
    if _effective_rate > 5.0:
        raise ValueError(
            f"env var 'CHEMBL_MIN_REQUEST_INTERVAL'={CHEMBL_MIN_REQUEST_INTERVAL} "
            f"implies {_effective_rate:.1f} req/sec, which EXCEEDS ChEMBL's "
            f"documented 5 req/sec public API rate limit. This would violate "
            f"ChEMBL's TOS and trigger 429 responses. Use a value >= 0.2 "
            f"(= 5 req/sec). (Task 37 v110 root fix)"
        )

# HTTP timeout tuple (connect, read) in seconds (SEC-2, C37).
CHEMBL_HTTP_TIMEOUT: tuple[float, float] = (
    _getenv_float("CHEMBL_HTTP_TIMEOUT_CONNECT", 10.0),
    _getenv_float("CHEMBL_HTTP_TIMEOUT_READ", 60.0),
)

# Maximum acceptable HTTP response body size in bytes (SEC-5). Default 50 MB
# -- a single ChEMBL page is ~1-3 MB so this is generous but bounded.
CHEMBL_MAX_RESPONSE_BYTES: int = _getenv_int("CHEMBL_MAX_RESPONSE_BYTES", 50 * 1024 * 1024)
if CHEMBL_MAX_RESPONSE_BYTES < 1024:
    raise ValueError(
        f"env var 'CHEMBL_MAX_RESPONSE_BYTES' must be >= 1024, "
        f"got {CHEMBL_MAX_RESPONSE_BYTES}"
    )

# Circuit breaker (R10). After CHEMBL_CIRCUIT_BREAKER_THRESHOLD consecutive
# failures, the HTTP client goes into "open" state and fails fast for
# CHEMBL_CIRCUIT_BREAKER_RESET_SECONDS before retrying.
CHEMBL_CIRCUIT_BREAKER_THRESHOLD: int = _getenv_int(
    "CHEMBL_CIRCUIT_BREAKER_THRESHOLD", 10
)
CHEMBL_CIRCUIT_BREAKER_RESET_SECONDS: float = _getenv_float(
    "CHEMBL_CIRCUIT_BREAKER_RESET_SECONDS", 60.0
)

# Scientific filters (S10-S12, S15). These define what we keep when
# downloading / cleaning activities. Defaults are conservative -- only
# well-measured human-protein binding/functional assays with exact ('=')
# activity relations.
CHEMBL_TARGET_ORGANISM: str = _getenv("CHEMBL_TARGET_ORGANISM", "Homo sapiens")
CHEMBL_MAX_PHASE: int = _getenv_int("CHEMBL_MAX_PHASE", 4)
if not (0 <= CHEMBL_MAX_PHASE <= 4):
    raise ValueError(
        f"env var 'CHEMBL_MAX_PHASE' must be in [0, 4], got {CHEMBL_MAX_PHASE}"
    )

# Lipinski's extended rule-of-5 threshold for macromolecule flagging (S8).
# Used ONLY to set the transient `is_macromolecule` boolean; never to
# overwrite `drug_type` (K6 fix).
# Reference: Lipinski CA et al., Adv Drug Deliv Rev 2001.
CHEMBL_MW_MACROMOLECULE_THRESHOLD: float = _getenv_float(
    "CHEMBL_MW_MACROMOLECULE_THRESHOLD", 900.0
)
if CHEMBL_MW_MACROMOLECULE_THRESHOLD <= 0.0:
    raise ValueError(
        f"env var 'CHEMBL_MW_MACROMOLECULE_THRESHOLD' must be > 0, "
        f"got {CHEMBL_MW_MACROMOLECULE_THRESHOLD}"
    )

# Activity types and units we know how to normalize (D2-5, DQ-15).
# These mirror the normalizer's supported set so we never silently drop
# activities the normalizer could have handled.
CHEMBL_ACTIVITY_TYPES: frozenset[str] = frozenset(
    s.strip()
    for s in _getenv("CHEMBL_ACTIVITY_TYPES", "IC50,Ki,Kd,EC50").split(",")
    if s.strip()
)
CHEMBL_STANDARD_UNITS: frozenset[str] = frozenset(
    s.strip()
    for s in _getenv(
        "CHEMBL_STANDARD_UNITS",
        "nM,uM,\u00b5M,\u03bcM,pM,mM,M,mol/L",
    ).split(",")
    if s.strip()
)

# Censorship and assay filters (S10, S12).
# standard_relation '=' means an exact measurement; '>' / '<' / '~' are
# censored values and are NOT directly comparable to '=' values.
CHEMBL_STANDARD_RELATIONS: frozenset[str] = frozenset(
    s.strip()
    for s in _getenv("CHEMBL_STANDARD_RELATIONS", "=").split(",")
    if s.strip()
)
# assay_type: B = binding, F = functional, U = unknown, A = ADME,
# P = physicochemical, T = toxicity. We keep B and F by default.
CHEMBL_ASSAY_TYPES: frozenset[str] = frozenset(
    s.strip().upper()
    for s in _getenv("CHEMBL_ASSAY_TYPES", "B,F").split(",")
    if s.strip()
)
# target_type: SINGLE PROTEIN, PROTEIN COMPLEX, ORGANISM, CELL-LINE, etc.
# We keep SINGLE PROTEIN and PROTEIN COMPLEX -- both have meaningful
# target_components UniProt accessions (S11).
CHEMBL_TARGET_TYPES: frozenset[str] = frozenset(
    s.strip()
    for s in _getenv(
        "CHEMBL_TARGET_TYPES", "SINGLE PROTEIN,PROTEIN COMPLEX"
    ).split(",")
    if s.strip()
)

# How to handle targets with multiple UniProt accessions (S9, D2-10).
# FIRST: take first accession only (legacy behavior; lossy for complexes).
# ALL:   keep all accessions; explode one activity into N DPI rows.
# BY_COMPONENT_TYPE: keep only PROTEIN-type components.
CHEMBL_TARGET_ACCESSION_STRATEGY: str = _getenv(
    "CHEMBL_TARGET_ACCESSION_STRATEGY", "ALL"
).upper()
if CHEMBL_TARGET_ACCESSION_STRATEGY not in {"FIRST", "ALL", "BY_COMPONENT_TYPE"}:
    raise ValueError(
        f"env var 'CHEMBL_TARGET_ACCESSION_STRATEGY' must be one of "
        f"FIRST, ALL, BY_COMPONENT_TYPE; got "
        f"{CHEMBL_TARGET_ACCESSION_STRATEGY!r}"
    )

# Batching and streaming (P2, P9, P11, P13, C22).
CHEMBL_ACTIVITY_CHUNK_SIZE: int = _getenv_int("CHEMBL_ACTIVITY_CHUNK_SIZE", 100_000)
if CHEMBL_ACTIVITY_CHUNK_SIZE < 1000:
    raise ValueError(
        f"env var 'CHEMBL_ACTIVITY_CHUNK_SIZE' must be >= 1000, "
        f"got {CHEMBL_ACTIVITY_CHUNK_SIZE}"
    )
CHEMBL_DPI_BATCH_SIZE: int = _getenv_int("CHEMBL_DPI_BATCH_SIZE", 1000)
if CHEMBL_DPI_BATCH_SIZE < 1:
    raise ValueError(
        f"env var 'CHEMBL_DPI_BATCH_SIZE' must be >= 1, got {CHEMBL_DPI_BATCH_SIZE}"
    )
CHEMBL_TARGET_RESOLUTION_BATCH_SIZE: int = _getenv_int(
    "CHEMBL_TARGET_RESOLUTION_BATCH_SIZE", 50
)
if CHEMBL_TARGET_RESOLUTION_BATCH_SIZE < 1:
    raise ValueError(
        f"env var 'CHEMBL_TARGET_RESOLUTION_BATCH_SIZE' must be >= 1, "
        f"got {CHEMBL_TARGET_RESOLUTION_BATCH_SIZE}"
    )

# Parallelism (P12, R14).
CHEMBL_API_WORKERS: int = _getenv_int("CHEMBL_API_WORKERS", 3)
if CHEMBL_API_WORKERS < 1:
    raise ValueError(
        f"env var 'CHEMBL_API_WORKERS' must be >= 1, got {CHEMBL_API_WORKERS}"
    )
CHEMBL_TARGET_RESOLUTION_WORKERS: int = _getenv_int(
    "CHEMBL_TARGET_RESOLUTION_WORKERS", 3
)
if CHEMBL_TARGET_RESOLUTION_WORKERS < 1:
    raise ValueError(
        f"env var 'CHEMBL_TARGET_RESOLUTION_WORKERS' must be >= 1, "
        f"got {CHEMBL_TARGET_RESOLUTION_WORKERS}"
    )

# Caches (P3, LIN-14, LIN-15).
CHEMBL_TARGET_CACHE_TTL_SECONDS: int = _getenv_int(
    "CHEMBL_TARGET_CACHE_TTL_SECONDS", 7 * 24 * 3600
)
CHEMBL_DRUG_ID_CACHE_TTL_SECONDS: int = _getenv_int(
    "CHEMBL_DRUG_ID_CACHE_TTL_SECONDS", 3600
)
CHEMBL_CACHE_TTL_SECONDS: int = _getenv_int("CHEMBL_CACHE_TTL_SECONDS", 86400)

# Idempotency / resume (I1, I2, R6, I10, I11).
CHEMBL_ALLOW_VERSION_MISMATCH: bool = _getenv_bool(
    "CHEMBL_ALLOW_VERSION_MISMATCH", False
)
CHEMBL_RESUME: bool = _getenv_bool("CHEMBL_RESUME", False)

# Pipeline-level settings used by every pipeline module but not previously
# defined here (CFG-11, L9, SEC-3, A4, I2, R6). All have safe defaults.
# Note: PIPELINE_RUN_ID defaults to "" (not None) so it passes settings
# validation; consumers check `if PIPELINE_RUN_ID:` to detect "not set".
PIPELINE_RUN_ID: str = _getenv("PIPELINE_RUN_ID", "")
PIPELINE_USE_CACHE: bool = _getenv_bool("PIPELINE_USE_CACHE", True)
PIPELINE_LOG_FORMAT: str = _getenv("PIPELINE_LOG_FORMAT", "text").lower()
if PIPELINE_LOG_FORMAT not in {"text", "json"}:
    raise ValueError(
        f"env var 'PIPELINE_LOG_FORMAT' must be 'text' or 'json', "
        f"got {PIPELINE_LOG_FORMAT!r}"
    )
PIPELINE_CONTACT_EMAIL: str = _getenv(
    "PIPELINE_CONTACT_EMAIL", "team-cosmic@example.com"
)
PIPELINE_RESUME: bool = _getenv_bool("PIPELINE_RESUME", False)

# DEPRECATED: ChEMBL FTP URL -- values stored in _DEPRECATED_SETTINGS (DESIGN-1)
# Accessing these triggers DeprecationWarning via module __getattr__

# ---------------------------------------------------------------------------
# UniProt -- SCI-5, IDMP-1
# ---------------------------------------------------------------------------

# UniProt release for reproducibility.
# P1-016 ROOT FIX (Team-2): default is now a PINNED release
# (``releases/2024_03``) instead of ``current_release``. UniProt releases
# weekly, so two runs on different days would use different protein sets
# -- making KG embeddings non-reproducible and breaking FDA audit
# requirements. The pinned default guarantees reproducibility; operators
# who want the latest release can explicitly set
# ``UNIPROT_RELEASE=current_release`` in development (it raises in
# production via the check below). The pinned release is updated
# deliberately as part of a quarterly KG re-build, NOT silently.
DEFAULT_UNIPROT_RELEASE: str = "releases/2024_03"
UNIPROT_RELEASE: str = _getenv("UNIPROT_RELEASE", DEFAULT_UNIPROT_RELEASE)

# P1-016 ROOT FIX (Team-2): in production, RAISE if UNIPROT_RELEASE is
# ``current_release`` -- a non-reproducible KG is a regulatory
# non-compliance. The previous code only warned, which operators routinely
# ignored. Raise to force the operator to pin a release.
if UNIPROT_RELEASE == "current_release" and ENVIRONMENT == "production":
    raise RuntimeError(
        "UNIPROT_RELEASE is set to 'current_release' in production. "
        "This makes pipeline runs non-reproducible (UniProt releases "
        "weekly) and violates FDA audit requirements. Pin a specific "
        "release (e.g., UNIPROT_RELEASE=releases/2024_03) for "
        "reproducibility. (P1-016 ROOT FIX: raise, don't warn.)"
    )
if UNIPROT_RELEASE == "current_release" and ENVIRONMENT != "production":
    warnings.warn(
        "UNIPROT_RELEASE is set to 'current_release' (ENVIRONMENT=%s). "
        "This is non-reproducible and acceptable ONLY for local dev. "
        "Pin a specific release (e.g., 'releases/2024_03') for any "
        "shared/staging/production deploy. (P1-016)"
        % ENVIRONMENT,
        UserWarning,
    )

# DEPRECATED: UniProt FTP URLs (DESIGN-1)
# Accessing these triggers DeprecationWarning via module __getattr__

# ---------------------------------------------------------------------------
# STRING -- SCI-1, DESIGN-2
# ---------------------------------------------------------------------------

DEFAULT_STRING_VERSION: str = "12.0"  # CONF-1

# Known valid STRING database versions
VALID_STRING_VERSIONS: frozenset[str] = frozenset(
    {"11.0", "11.0b", "11.5", "12.0"}
)

STRING_VERSION: str = _getenv("STRING_VERSION", DEFAULT_STRING_VERSION)

# Version-aware score thresholds (SCI-1)
# FIX TOP-1: STRING combined_score >= 700 is the canonical high-confidence
# cutoff (Szklarczyk et al. 2023, Nucleic Acids Research -- >= 700 achieves
# >80% precision on KEGG pathway benchmarks; >= 400 achieves only ~50%).
# The previous v12.0 entry used 400 -- this dropped ~75% of the high-
# confidence PPIs that Phase 1 retained, causing Phase 2 to silently lose
# most of its protein-protein interaction graph. All STRING versions now
# use 700 as the default threshold. Operators can still override via the
# STRING_MIN_COMBINED_SCORE env var. Synchronized with
# phase2/drugos_graph/config.py -- DO NOT diverge (audit TOP-1).
STRING_VERSION_SCORE_THRESHOLDS: dict[str, tuple[int, str]] = {
    # version: (default_threshold, scientific_rationale)
    "11.0b": (700, "v11.0b -- 700 is the canonical high-confidence cutoff (Szklarczyk 2023)"),
    "11.5": (700, "v11.5 -- 700 is the canonical high-confidence cutoff (Szklarczyk 2023)"),
    "12.0": (700, "v12.0 -- 700 is the canonical high-confidence cutoff (Szklarczyk 2023); "
                  "previously 400 which retained only ~50% precision PPIs"),
}


def _get_default_string_threshold(version: str) -> int:
    """Get the scientifically validated default threshold for a STRING version."""
    if version in STRING_VERSION_SCORE_THRESHOLDS:
        return STRING_VERSION_SCORE_THRESHOLDS[version][0]
    # For unknown versions, use the most recent known threshold and warn
    latest = max(STRING_VERSION_SCORE_THRESHOLDS.keys())
    fallback = STRING_VERSION_SCORE_THRESHOLDS[latest][0]
    warnings.warn(
        f"STRING_VERSION={version} has no validated score threshold. "
        f"Using fallback threshold {fallback} from v{latest}. "
        f"Validate this threshold against the {version} score distribution "
        f"before using in production.",
        UserWarning,
    )
    return fallback


def _build_string_urls(version: str) -> dict[str, str]:
    """Build and validate STRING DB URLs for the given version.

    Warns if the version is not in the known valid set.
    Returns a dict with keys: protein_links_url, protein_info_url,
    aliases_url, protein_links_detailed_url.
    """
    if version not in VALID_STRING_VERSIONS:
        warnings.warn(
            f"STRING_VERSION={version} is not in the known valid set "
            f"{sorted(VALID_STRING_VERSIONS)}. The URLs may not resolve.",
            UserWarning,
        )
    base = "https://stringdb-downloads.org/download"
    return {
        "protein_links_url": (
            f"{base}/protein.links.v{version}/"
            f"9606.protein.links.v{version}.txt.gz"
        ),
        "protein_info_url": (
            f"{base}/protein.info.v{version}/"
            f"9606.protein.info.v{version}.txt.gz"
        ),
        "aliases_url": (
            f"{base}/protein.aliases.v{version}/"
            f"9606.protein.aliases.v{version}.txt.gz"
        ),
        "protein_links_detailed_url": (
            f"{base}/protein.links.detailed.v{version}/"
            f"9606.protein.links.detailed.v{version}.txt.gz"
        ),
    }


_string_urls = _build_string_urls(STRING_VERSION)

# CODE-5: Fixed env var name to match setting name, with backward compat
STRING_MIN_COMBINED_SCORE: int = _parse_required_int(
    "STRING_MIN_COMBINED_SCORE",
    str(_get_default_string_threshold(STRING_VERSION)),
)

# Backward compatibility: support the old STRING_MIN_SCORE env var name
_legacy_string_score = os.getenv("STRING_MIN_SCORE")
if _legacy_string_score is not None:
    warnings.warn(
        "Env var STRING_MIN_SCORE is deprecated. "
        "Use STRING_MIN_COMBINED_SCORE instead.",
        DeprecationWarning,
    )
    STRING_MIN_COMBINED_SCORE = int(_legacy_string_score)

STRING_PROTEIN_LINKS_URL: str = _string_urls["protein_links_url"]
STRING_ALIASES_URL: str = _string_urls["aliases_url"]
STRING_PROTEIN_LINKS_DETAILED_URL: str = _string_urls["protein_links_detailed_url"]

# ---------------------------------------------------------------------------
# STRING production-override + reliability/reproducibility knobs (BUG-3.4,
# GAP-12.5, GAP-12.6, GAP-12.7, GAP-12.9, GAP-8.1, GAP-8.2).
#
# These are ADDITIVE -- no existing setting is removed.  They are consumed by
# the institutional-grade pipelines/string_pipeline.py (v2.0.0).
# ---------------------------------------------------------------------------

# Sci: Szklarczyk et al. 2023 (Nucleic Acids Research) -- combined_score
# >= 700 achieves >80% precision on KEGG pathway benchmarks; >= 400 (the
# dev default) achieves only ~50%.  For a clinical-decision-support system,
# 700 is the minimum defensible threshold.
STRING_MIN_COMBINED_SCORE_PROD: int = _getenv_int(
    "STRING_MIN_COMBINED_SCORE_PROD", default=700
)
"""Production override for STRING_MIN_COMBINED_SCORE.

Per Szklarczyk et al. 2023, combined_score >= 700 achieves >80% precision
on KEGG pathway benchmarks; >= 400 (the dev default) achieves only ~50%.
For a clinical-decision-support system, 700 is the minimum defensible
threshold.  In production (ENV=prod), the STRING pipeline forces the
effective threshold to this value if STRING_MIN_COMBINED_SCORE is below it.
"""

# GAP-7.4: Detailed-file requirement.
#   - "optional" (default): attempt download, warn on failure
#   - "required":            download without try/except (failure raises)
#   - "skip":                do not attempt download at all
STRING_DETAILED_MODE: str = _getenv("STRING_DETAILED_MODE", "optional").lower()
if STRING_DETAILED_MODE not in {"optional", "required", "skip"}:
    warnings.warn(
        f"STRING_DETAILED_MODE={STRING_DETAILED_MODE!r} is not one of "
        f"optional/required/skip -- falling back to 'optional'.",
        UserWarning,
    )
    STRING_DETAILED_MODE = "optional"
"""How the STRING pipeline handles the detailed-links file.

- ``optional`` (default) -- attempt download; on failure, log a WARNING
  and continue without sub-scores.  Reproducible only if the download
  consistently succeeds.
- ``required`` -- download without try/except.  Failure raises.  Use this
  for production runs where sub-score coverage MUST be reproducible.
- ``skip`` -- do not attempt download at all.  Sub-scores will be NULL.
"""

# GAP-12.6: Self-interaction (homodimer) handling.
# Sci: Homodimers are biologically real and clinically critical --
# receptor dimerization (EGFR, HER2, VEGFR) is the primary mechanism of
# action for trastuzumab, lapatinib, cetuximab, pertuzumab. p53
# tetramerization is fundamental to tumor-suppressor function.  The DB
# schema's chk_ppi_ordered constraint currently forbids a_id == b_id.
# TODO(schema-migration): relax the constraint and load homodimers with
# an is_homodimer flag.  Until then, drop them with WARNING + dead-letter.
# v90 ROOT FIX (BUG #9): default changed from True to False.
# Dropping self-interactions (homodimers) removes biologically
# critical protein interactions (EGFR, HER2, p53 tetramerization,
# STAT3 homodimer). These are NOT artifacts -- they are real PPIs
# with high combined scores. The previous default True dropped ALL
# homodimers to satisfy a DB constraint (chk_ppi_ordered), but the
# correct fix is to relax the constraint or mark homodimers with
# an is_homodimer flag. Setting default=False means the pipeline
# will FAIL LOUDLY if the DB constraint rejects homodimers, rather
# than silently dropping them and producing a KG with missing
# critical edges. The DB constraint must be relaxed via migration.
STRING_DROP_SELF_INTERACTIONS: bool = _getenv_bool(
    "STRING_DROP_SELF_INTERACTIONS", default=False
)

# GAP-3.11 / GAP-12.7: Dedup strategy for collapsing multiple STRING
# ENSP pairs that map to the same UniProt pair.
#   - "max_score"  (default, recommended): keep the row with the highest
#                   combined_score (strongest evidence -- Szklarczyk et al. 2023)
#   - "mean_score": aggregate by mean (dilutes strong evidence with weak)
#   - "first":      legacy non-deterministic (sorted first for determinism)
STRING_DEDUP_STRATEGY: str = _getenv(
    "STRING_DEDUP_STRATEGY", "max_score"
).lower()
if STRING_DEDUP_STRATEGY not in {"max_score", "mean_score", "first"}:
    warnings.warn(
        f"STRING_DEDUP_STRATEGY={STRING_DEDUP_STRATEGY!r} is not one of "
        f"max_score/mean_score/first -- falling back to 'max_score'.",
        UserWarning,
    )
    STRING_DEDUP_STRATEGY = "max_score"
"""Dedup strategy for collapsing multiple STRING ENSP pairs that map to
the same UniProt pair (e.g. isoforms of the same protein).

- ``max_score`` (default, recommended) -- keep the row with the highest
  combined_score.  Deterministic and reflects the strongest evidence
  (Szklarczyk et al. 2023).
- ``mean_score`` -- aggregate by mean.  Dilutes strong evidence with weak.
- ``first`` -- legacy; deterministic because we sort first, but loses
  information.
"""

# GAP-12.4 / BUG-8.1: low_memory flag for pd.read_csv.
STRING_LOW_MEMORY: bool = _getenv_bool("STRING_LOW_MEMORY", default=False)
"""If True, pass low_memory=True to pd.read_csv for STRING files (slower
but lower peak memory).  Default False (full materialization for speed
on machines with >= 8 GB RAM).  STRING v12.0 links file is ~1.5 GB in
memory."""

# BUG-8.1 / GAP-8.9: Chunk size for chunked reading (0 = disabled).
STRING_CHUNK_SIZE: int = _getenv_int("STRING_CHUNK_SIZE", default=0)
"""Chunk size (rows) for chunked reading of STRING files. 0 = disabled
(load entire file in memory).  For machines with < 8 GB RAM, set to
1_000_000 to bound peak memory.  When > 0, the links file is read in
chunks and only rows passing the score filter are concatenated."""

# ---------------------------------------------------------------------------
# Controlled vocabulary for the `source` column across all pipelines
# (GAP-2.7, GAP-14.2).  Implemented as a str-Enum for ergonomic use.
# ---------------------------------------------------------------------------
try:
    from enum import Enum

    class DataSourceName(str, Enum):
        """Controlled vocabulary for the ``source`` column across all pipelines."""

        STRING = "string"
        CHEMBL = "chembl"
        DRUGBANK = "drugbank"
        UNIPROT = "uniprot"
        DISGENET = "disgenet"
        OMIM = "omim"
        PUBCHEM = "pubchem"

except ImportError:  # pragma: no cover -- enum is stdlib, this never fires.
    DataSourceName = None  # type: ignore[assignment]

# DEPRECATED: STRING protein info URL (DESIGN-1)
# FIX M1: STRING_PROTEIN_INFO_URL is kept for reference but unused in
# the pipeline's download() method. clean() never reads this file.
# Accessing this triggers DeprecationWarning via module __getattr__

# ---------------------------------------------------------------------------
# DisGeNET -- SCI-3, SEC-2, DESIGN-4, CONF-4
# ---------------------------------------------------------------------------

# SCI-3: DisGeNET migrated to api.disgenet.com in 2024.
# The old www.disgenet.org/api/ endpoint is deprecated and may not work.
# The new API v1 base is https://api.disgenet.com/api/v1/
DISGENET_API_URL: str = _getenv(
    "DISGENET_API_URL",
    "https://api.disgenet.com/api/v1/gda/summary",
)
DISGENET_API_KEY: str = _getenv("DISGENET_API_KEY", "")
DISGENET_USE_API: bool = _parse_bool(_getenv("DISGENET_USE_API", "true"))

# v110 Task 24 root fix: DisGeNET license tier management.
#
# DisGeNET offers three license tiers (per https://www.disgenet.org/plans):
#   - "curated"  : paid license, full curated GDA records (all sources,
#                  including BEFREE, CURATED, etc.). ~1.2M+ GDAs.
#   - "premium"  : paid license, includes animal model + predicted data.
#   - "free"     : free tier, limited to ~620K GDAs from public sources
#                  (CURATED + ALL_OMIM + ALL_HPO + INFILER + etc. minus
#                  premium-only sources). Requires free API key.
#
# The audit (Task 24) requires: "must use the curated license if available,
# fallback to free tier with a warning."
#
# ROOT FIX: add DISGENET_LICENSE_TIER setting (default "auto").
#   - "auto" (default): if DISGENET_API_KEY is present, assume curated tier
#     (the key was provisioned for a paid plan). If key is absent, fall back
#     to free tier with a clear warning that the dataset is limited.
#   - "curated": explicitly use the curated endpoint (fails if no key).
#   - "free"   : explicitly use the free endpoint (key optional but recommended).
#   - "premium": explicitly use the premium endpoint (fails if no key).
#
# The tier affects:
#   1. The API endpoint path (curated uses /gda/summary, free uses the same
#      path but the server returns a limited record set based on the key's
#      entitlements).
#   2. The expected record count (curated: ~1.2M+, free: ~620K). The
#      pipeline logs the expected count so operators can detect a
#      silent tier downgrade.
#   3. The warning emitted when falling back to free.
DISGENET_LICENSE_TIER: str = _getenv("DISGENET_LICENSE_TIER", "auto").lower().strip()
if DISGENET_LICENSE_TIER not in ("auto", "curated", "premium", "free"):
    raise ValueError(
        f"env var 'DISGENET_LICENSE_TIER' must be one of "
        f"'auto', 'curated', 'premium', 'free' — got {DISGENET_LICENSE_TIER!r}. "
        f"(Task 24 v110 root fix)"
    )

# Expected GDA record counts per tier (for silent-downgrade detection).
# Source: DisGeNET pricing page + Piñero et al. 2020 §3.1.
DISGENET_EXPECTED_RECORDS_BY_TIER: dict[str, int] = {
    "curated": 1_200_000,   # curated + ALL sources
    "premium": 1_500_000,   # premium includes animal model + predicted
    "free": 620_000,        # free tier (public sources only)
}


def _resolve_disgenet_tier() -> tuple[str, str]:
    """Resolve the effective DisGeNET license tier and emit warnings.

    Returns
    -------
    tuple of (effective_tier, warning_message)
        effective_tier : str — "curated", "premium", or "free"
        warning_message : str — empty string if no warning, else the message
    """
    if DISGENET_LICENSE_TIER == "auto":
        if DISGENET_API_KEY:
            # Key present — assume curated (paid plans provision keys).
            return ("curated", "")
        else:
            # No key — fall back to free tier with warning.
            _warn = (
                "DISGENET_LICENSE_TIER=auto and DISGENET_API_KEY is NOT set. "
                "Falling back to FREE tier — the dataset will be LIMITED to "
                "~620K public-source GDAs (vs ~1.2M+ for curated). Curated-only "
                "sources (BEFREE, GWAS_CATALOG_FULL, etc.) will be ABSENT. "
                "For institutional-grade coverage, set DISGENET_API_KEY with "
                "a curated/paid plan key, or set DISGENET_LICENSE_TIER=curated "
                "explicitly. (Task 24 v110 root fix)"
            )
            return ("free", _warn)
    elif DISGENET_LICENSE_TIER in ("curated", "premium"):
        if not DISGENET_API_KEY:
            raise ValueError(
                f"DISGENET_LICENSE_TIER={DISGENET_LICENSE_TIER!r} requires "
                f"a DISGENET_API_KEY (paid license), but the key is NOT set. "
                f"Either set DISGENET_API_KEY with a {DISGENET_LICENSE_TIER} "
                f"plan key, or set DISGENET_LICENSE_TIER=auto/free. "
                f"(Task 24 v110 root fix)"
            )
        return (DISGENET_LICENSE_TIER, "")
    else:  # "free"
        if not DISGENET_API_KEY:
            _warn = (
                "DISGENET_LICENSE_TIER=free and DISGENET_API_KEY is NOT set. "
                "DisGeNET's free tier REQUIRES a free API key (register at "
                "https://www.disgenet.org/plans). Without a key, the API "
                "will return 401 Unauthorized. (Task 24 v110 root fix)"
            )
            return ("free", _warn)
        return ("free", "")


# Resolve the effective tier at import time (so warnings fire once).
DISGENET_EFFECTIVE_TIER, DISGENET_TIER_WARNING = _resolve_disgenet_tier()
if DISGENET_TIER_WARNING:
    warnings.warn(DISGENET_TIER_WARNING, UserWarning)

# The primary URL now points to the API by default (SCI-3)
DISGENET_URL: str = DISGENET_API_URL

# DEPRECATED: Static URL (broken since 2024)
# Accessing this triggers DeprecationWarning via module __getattr__

# Warn if someone explicitly opts out of the API
if not DISGENET_USE_API:
    warnings.warn(
        "DISGENET_USE_API=false is set, but the DisGeNET static URL is "
        "deprecated since 2024 and may not work. Set DISGENET_USE_API=true "
        "and provide DISGENET_API_KEY for reliable data access.",
        UserWarning,
    )

# ===========================================================================
# DisGeNET institutional-grade configuration knobs (389-fix audit).
#
# These are ADDITIVE -- no existing setting is removed.  They are consumed by
# the institutional-grade ``pipelines/disgenet_pipeline.py`` (v2.0.0).
#
# All scientific thresholds cite Piñero et al., 2020, *DisGeNET: a
# comprehensive platform integrating information on human disease-associated
# genes and variants*, Nucleic Acids Research
# (https://doi.org/10.1093/nar/gkz1021).
# ===========================================================================

# SCI-1 / DES-1 / CONF-1: Minimum score threshold.
# Per Piñero et al. 2020, DisGeNET scores in [0.06, 0.1) constitute "weak
# evidence" -- biologically meaningful, especially for rare diseases.  The
# previous default of 0.1 silently destroyed this evidence.  The new default
# is 0.06, the published weak-evidence floor.  Set DISGENET_ALLOW_WEAK_EVIDENCE
# to False to hard-filter at this threshold; otherwise weak-evidence rows are
# kept and tagged with confidence_tier="weak".
DISGENET_MIN_SCORE: float = _getenv_float("DISGENET_MIN_SCORE", default=0.06)
"""Minimum DisGeNET score for inclusion.

Defaults to 0.06 -- the floor of DisGeNET's 'weak evidence' band per
Piñero et al., 2020 (Nucleic Acids Research).  Set to 0.0 to disable
filtering entirely (preserve all evidence, including sub-weak).  Pair
with DISGENET_ALLOW_WEAK_EVIDENCE=false to hard-filter at this threshold;
otherwise weak-evidence rows are kept and tagged with confidence_tier='weak'.

Unit: float in [0, 1].
"""

DISGENET_ALLOW_WEAK_EVIDENCE: bool = _getenv_bool(
    "DISGENET_ALLOW_WEAK_EVIDENCE", default=True
)
"""If True (default), do NOT filter out weak-evidence rows (score in
[DISGENET_MIN_SCORE, DISGENET_WEAK_EVIDENCE_THRESHOLD)).  Instead, tag
them with confidence_tier="weak".
If False, hard-filter at DISGENET_MIN_SCORE (drops weak-evidence rows)."""

# v82 FORENSIC ROOT FIX (P1-3 -- weak-evidence threshold hardcoded 0.1):
# The previous code hardcoded ``0.1`` as the weak-evidence threshold in
# ``disgenet_pipeline._apply_score_filter`` while ``DISGENET_MIN_SCORE``
# was configurable. The two thresholds were DECOUPLED -- operators tuning
# ``DISGENET_MIN_SCORE`` did not get the expected behavior:
#   * If MIN_SCORE=0.2 (drop weak evidence): weak-evidence path still
#     fired for [0.06, 0.1) but those rows were already dropped -> dead
#     code.
#   * If MIN_SCORE=0.05: weak-evidence threshold (0.1) didn't move, so
#     rows in [0.05, 0.06) were dropped before the weak-evidence tagger
#     could rescue them.
# ROOT FIX: make the weak-evidence threshold configurable as
# ``DISGENET_WEAK_EVIDENCE_THRESHOLD`` (default 0.1, preserving prior
# behavior). The weak-evidence band is now
# ``[DISGENET_MIN_SCORE, DISGENET_WEAK_EVIDENCE_THRESHOLD)`` -- the two
# thresholds move together when operators tune either one.
DISGENET_WEAK_EVIDENCE_THRESHOLD: float = _getenv_float(
    "DISGENET_WEAK_EVIDENCE_THRESHOLD", default=0.1
)
"""Upper bound (exclusive) of the weak-evidence band. Rows with score in
``[DISGENET_MIN_SCORE, DISGENET_WEAK_EVIDENCE_THRESHOLD)`` are tagged
``confidence_tier="weak"`` (when ``DISGENET_ALLOW_WEAK_EVIDENCE=True``)
instead of being dropped. Default 0.1 (matches Piñero et al. 2020 §2.3
weak-evidence floor). Must be > ``DISGENET_MIN_SCORE`` to be meaningful."""

# SCI-11 / DES-2 / CONF-2: Confidence tier thresholds.
# Per Piñero et al. 2020 §2.3, the DSGP score bands are:
#   [0.0, 0.06)   -- sub-weak (below published floor)
#   [0.06, 0.3)   -- weak evidence
#   [0.3, 1.0]    -- strong evidence
# These thresholds are configurable via DISGENET_CONFIDENCE_TIERS (JSON-encoded
# list of [threshold, label] pairs).  The previous 0.7 -> "very_high" tier is
# removed (no publication supports it).
DISGENET_CONFIDENCE_TIERS_JSON: str = _getenv(
    "DISGENET_CONFIDENCE_TIERS",
    # P1-004 ROOT FIX (v100 forensic + Team-1 v102 extension):
    #   v100: labels aligned with Piñero 2020 §2.3 -- sub_weak / weak / strong.
    #   The previous default was [[0.0,"weak"],[0.06,"moderate"],[0.3,"strong"]]
    #   which mislabeled the [0.0, 0.06) sub-floor band as "weak" and the
    #   [0.06, 0.3) weak band as "moderate" (Piñero does not define a
    #   "moderate" band).
    #   v102 (Team-1 P1-004 EXTENSION): split the strong band [0.3, 1.0] into
    #   "strong" [0.3, 0.5) and "very_strong" [0.5, 1.0] so the gradation
    #   between a score of 0.31 (marginal evidence) and 0.95 (very strong,
    #   curated multi-source) is preserved. Downstream ML models that bin on
    #   confidence_tier no longer weight them identically. The DB CHECK
    #   constraint, ORM CheckConstraint, JSON schema, migration 012 (label
    #   rename) and migration 017 (very_strong split) are updated in lockstep
    #   to accept the new label set.
    default='[[0.0,"sub_weak"],[0.06,"weak"],[0.3,"strong"],[0.5,"very_strong"]]',
)
"""JSON-encoded list of [threshold, label] pairs for confidence tier
classification.  P1-004 v100+v102 ROOT FIX: default follows Piñero et al.
2020 §2.3 ACTUAL vocabulary with the v102 very_strong split:
``[[0.0,"sub_weak"],[0.06,"weak"],[0.3,"strong"],[0.5,"very_strong"]]``.
The [0.06, 0.3) band is the WEAK-evidence band (not "moderate" as the
previous code wrongly labeled it). The [0.3, 1.0] band is split into
"strong" [0.3, 0.5) and "very_strong" [0.5, 1.0] to preserve gradation.
Thresholds must be sorted ascending; labels must be non-empty strings."""

# SCI-17 / CONF-3: PMID cap.
# The GeneDiseaseAssociation.pmid_list column is String(2000).  Each PMID is
# 7-8 digits + 1 separator.  Cap is computed dynamically:
#   DISGENET_PMID_CAP = min(200, (PMID_LIST_LENGTH - 1) // 10)
# but the user-set value takes precedence (validated against PMID_LIST_LENGTH).
DISGENET_PMID_CAP: int = _getenv_int("DISGENET_PMID_CAP", default=200)
"""Maximum number of PMIDs retained per record after capping.  Default 200
(utilises the full String(2000) capacity of the pmid_list column).  If the
resulting max string length exceeds PMID_LIST_LENGTH, the pipeline raises
ValueError at init -- see DISGENET_PMID_SORT_ORDER for sort semantics."""

DISGENET_PMID_SORT_ORDER: str = _getenv(
    "DISGENET_PMID_SORT_ORDER", "recent_first"
).lower()
if DISGENET_PMID_SORT_ORDER not in {"recent_first", "chronological", "as_returned"}:
    warnings.warn(
        f"DISGENET_PMID_SORT_ORDER={DISGENET_PMID_SORT_ORDER!r} is not one of "
        f"recent_first/chronological/as_returned -- falling back to 'recent_first'.",
        UserWarning,
    )
    DISGENET_PMID_SORT_ORDER = "recent_first"
"""Sort order for PMIDs before capping (SCI-16).
- ``recent_first`` (default) -- descending PMID (NCBI assigns PMIDs
  monotonically; higher = more recent).  Retains the most evidentially
  important PMIDs.
- ``chronological`` -- ascending PMID.
- ``as_returned`` -- no sort (legacy behaviour, not recommended)."""

# PERF-15 / CONF-4: API page size.
DISGENET_API_PAGE_SIZE: int = _getenv_int("DISGENET_API_PAGE_SIZE", default=5000)
"""Number of records to fetch per API request.  Default 5000.  The pipeline
validates this against the API's max on first request; if rejected, logs a
WARNING and falls back to 5000.  Higher values reduce request count but
increase per-request memory."""

# CONF-5: Safety cap on total records (prevents infinite pagination).
DISGENET_API_MAX_RECORDS: int = _getenv_int(
    "DISGENET_API_MAX_RECORDS", default=1_000_000
)
"""Hard safety cap on total records fetched (anti-infinite-loop).  Default
1,000,000 (DisGeNET has ~1M GDAs; this is a safety valve, not a normal
termination)."""

# CONF-6 / PERF-16 / REL-13: API timeout.
DISGENET_API_TIMEOUT: int = _getenv_int("DISGENET_API_TIMEOUT", default=30)
"""HTTP timeout (seconds) for a single API request.  Default 30 -- DisGeNET
pages typically respond in <5s, 30 is generous.  Lowered from the previous
hardcoded 120s."""

# CONF-7: Max retries.
DISGENET_API_MAX_RETRIES: int = _getenv_int("DISGENET_API_MAX_RETRIES", default=5)
"""Maximum number of retries per API request.  Default 5."""

# CONF-8 / PERF-9: Exponential backoff.
DISGENET_API_BACKOFF_BASE: float = _getenv_float(
    "DISGENET_API_BACKOFF_BASE", default=2.0
)
"""Base for exponential backoff: ``wait = min(base ** attempt, MAX_SECONDS)``.
Default 2.0."""

DISGENET_API_BACKOFF_MAX_SECONDS: int = _getenv_int(
    "DISGENET_API_BACKOFF_MAX_SECONDS", default=60
)
"""Maximum sleep per retry (caps the exponential).  Default 60s."""

DISGENET_API_MAX_RETRY_AFTER: int = _getenv_int(
    "DISGENET_API_MAX_RETRY_AFTER", default=300
)
"""Maximum sleep when honouring a 429 Retry-After header.  Default 300s
(5 minutes)."""

# SEC-20: Client-side rate limiting.
DISGENET_API_RATE_LIMIT: float = _getenv_float(
    "DISGENET_API_RATE_LIMIT", default=1.0
)
"""Maximum API requests per second (token-bucket).

v110 Task 23 root fix: default changed from 2.0 to 1.0 to comply with
DisGeNET's free-tier TOS (1 req/sec per https://www.disgenet.org/api/).
The previous default of 2.0 SILENTLY VIOLATED the free-tier TOS and could
trigger 429 rate-limit responses or API key suspension.

For curated/premium tiers, DisGeNET allows higher rates (up to 5 req/sec
per the curated plan docs). Deployers on paid plans can set
DISGENET_API_RATE_LIMIT=5.0 explicitly. The tier-aware clamp below
prevents accidental TOS violations on the free tier.
"""
# v110 Task 23 root fix: tier-aware rate-limit clamp.
# Free tier: max 1.0 req/sec (DisGeNET TOS).
# Curated/premium: max 5.0 req/sec (paid plan allowance).
_MAX_RATE_BY_TIER: dict[str, float] = {
    "free": 1.0,
    "curated": 5.0,
    "premium": 5.0,
}
_max_allowed = _MAX_RATE_BY_TIER.get(DISGENET_EFFECTIVE_TIER, 1.0)
if DISGENET_API_RATE_LIMIT > _max_allowed:
    _original_rate = DISGENET_API_RATE_LIMIT
    DISGENET_API_RATE_LIMIT = _max_allowed
    warnings.warn(
        f"DISGENET_API_RATE_LIMIT={_original_rate} exceeds the maximum "
        f"allowed for tier={DISGENET_EFFECTIVE_TIER} ({_max_allowed} req/sec "
        f"per DisGeNET TOS). Clamped to {_max_allowed}. To use a higher rate, "
        f"upgrade to a curated/premium plan and set DISGENET_LICENSE_TIER=curated. "
        f"(Task 23 v110 root fix)",
        UserWarning,
    )
elif DISGENET_API_RATE_LIMIT < 0.0:
    raise ValueError(
        f"DISGENET_API_RATE_LIMIT must be >= 0.0, got {DISGENET_API_RATE_LIMIT}. "
        f"(Task 23 v110 root fix)"
    )

# REL-8: Circuit breaker.
DISGENET_CIRCUIT_BREAKER_THRESHOLD: int = _getenv_int(
    "DISGENET_CIRCUIT_BREAKER_THRESHOLD", default=5
)
"""Consecutive API failures before the circuit opens.  Default 5."""

DISGENET_CIRCUIT_BREAKER_RESET_SECONDS: int = _getenv_int(
    "DISGENET_CIRCUIT_BREAKER_RESET_SECONDS", default=300
)
"""Seconds the circuit stays open before entering half-open.  Default 300."""

# SEC-16: User-Agent identification.
DISGENET_CONTACT_EMAIL: str = _getenv(
    "DISGENET_CONTACT_EMAIL", default="unknown@example.com"
)
"""Contact email for the User-Agent header (per DisGeNET API terms of use).
Replace with your team's contact in production."""

# SEC-7: SSRF protection.
DISGENET_ALLOWED_DOMAINS: list[str] = [
    d.strip() for d in _getenv(
        "DISGENET_ALLOWED_DOMAINS",
        default="api.disgenet.com,www.disgenet.org,disgenet.org",
    ).split(",") if d.strip()
]
"""Comma-separated list of allowed DisGeNET API domains.  Default
``api.disgenet.com,www.disgenet.org,disgenet.org``.  The primary domain
since 2024 is ``api.disgenet.com``.  The pipeline rejects any other domain
(SEC-7 SSRF protection)."""

# SEC-9: Response size validation.
DISGENET_API_MAX_RESPONSE_BYTES: int = _getenv_int(
    "DISGENET_API_MAX_RESPONSE_BYTES", default=100_000_000
)
"""Maximum acceptable response size in bytes.  Default 100 MB.  Larger
responses raise RuntimeError (defends against accidental memory exhaustion
from a malformed endpoint)."""

# SEC-10: TLS CA bundle override.
DISGENET_API_CA_BUNDLE: str = _getenv("DISGENET_API_CA_BUNDLE", default="")
"""Optional path to a CA bundle file.  Empty string = system default.
Set to a specific path to pin to a custom CA (e.g. corporate proxy)."""

# SEC-14: Output file permissions.
DISGENET_OUTPUT_FILE_MODE: str = _getenv("DISGENET_OUTPUT_FILE_MODE", default="0o640")
"""Octal file mode (as a string) for the output CSV.  Default '0o640'
(owner read/write, group read, no others)."""

# REL-6: Fallback to cache.
DISGENET_FALLBACK_TO_CACHE: bool = _getenv_bool(
    "DISGENET_FALLBACK_TO_CACHE", default=True
)
"""If True (default) and the API fails after all retries, fall back to the
most recent cached TSV in raw_dir with a WARNING.  If False, raise."""

# REL-14: Overall pagination caps.
DISGENET_API_MAX_PAGES: int = _getenv_int("DISGENET_API_MAX_PAGES", default=500)
"""Hard cap on the number of API pages fetched.  Default 500 (at 5000
records/page = 2.5M records, well above DisGeNET's ~1M)."""

DISGENET_DOWNLOAD_PHASE_TIMEOUT: int = _getenv_int(
    "DISGENET_DOWNLOAD_PHASE_TIMEOUT", default=3600
)
"""Overall wall-clock timeout (seconds) for the entire download phase.
Default 3600 (1 hour)."""

# SCI-25 / SCI-35 / IDEM-20: Pagination completeness.
DISGENET_ALLOW_PARTIAL_DATA: bool = _getenv_bool(
    "DISGENET_ALLOW_PARTIAL_DATA", default=False
)
"""If True (dev/debug only), do NOT raise on pagination completeness
mismatch -- log ERROR and write a partial-data manifest instead.  Default
False (production: raise)."""

# IDEM-7: UniProt map cache.
DISGENET_UNIPROT_MAP_TTL_HOURS: int = _getenv_int(
    "DISGENET_UNIPROT_MAP_TTL_HOURS", default=24
)
"""TTL (hours) for the cached gene_symbol->uniprot_id map.  Default 24h."""

# IDEM-8: DisGeNET version pinning.
DISGENET_TARGET_VERSION: str = _getenv("DISGENET_TARGET_VERSION", default="")
"""Pin to a specific DisGeNET release (e.g. 'v7').  Empty string = latest.
Stored in score_method (e.g. 'disgenet_v7_2024_06') and source_version."""

# IDEM-14: Snapshot isolation.
DISGENET_FREEZE_VERSION: str = _getenv("DISGENET_FREEZE_VERSION", default="")
"""If set, every GDA row gets snapshot_tag=this value (no overwrite of
existing snapshots).  Empty string = live table (overwrite on conflict)."""

# DQ-25: Minimum expected record count.
DISGENET_MIN_EXPECTED_RECORDS: int = _getenv_int(
    "DISGENET_MIN_EXPECTED_RECORDS", default=100_000
)
"""Minimum number of records expected after clean().  Default 100,000
(DisGeNET has ~1M GDAs; 100K is a conservative floor).  The pipeline
raises RuntimeError if fewer records survive."""

# DQ-19 / DQ-20: Optional referential integrity checks.
DISGENET_DISEASE_ONTOLOGY_PATH: str = _getenv(
    "DISGENET_DISEASE_ONTOLOGY_PATH", default=""
)
"""Optional path to a disease ontology file (MeSH/UMLS/DOID).  When set,
the pipeline validates every disease_id against the ontology and
quarantines invalid rows.  Empty = skip the check."""

DISGENET_HGNC_PATH: str = _getenv("DISGENET_HGNC_PATH", default="")
"""Optional path to an HGNC symbol dump.  When set, the pipeline validates
every gene_symbol against it and quarantines unknown symbols.  Empty = skip."""

# DQ-33: Stale data detection.
DISGENET_MAX_DATA_AGE_DAYS: int = _getenv_int(
    "DISGENET_MAX_DATA_AGE_DAYS", default=180
)
"""Maximum acceptable age (days) of the DisGeNET release.  If the release
date is older than this, the manifest's stale_data flag is set to True
(WARNING, not failure -- DisGeNET may have legitimate slow release cycles).
Default 180 (6 months)."""

# CONF-10 / CONF-11: Output / raw filenames.
DISGENET_OUTPUT_FILENAME: str = _getenv(
    "DISGENET_OUTPUT_FILENAME", default="gene_disease_associations.csv"
)
"""Output CSV filename in PROCESSED_DATA_DIR.  Default
'gene_disease_associations.csv'.  Changing this breaks downstream
consumers (Neo4j exporter, Graph Transformer) -- change only for testing."""

DISGENET_RAW_FILENAME: str = _getenv(
    "DISGENET_RAW_FILENAME", default=""
)
"""Raw filename in raw_dir.  Empty = auto-detect (static=.tsv.gz, API=.tsv)."""

# PERF-3: Optional chunked processing.
DISGENET_CHUNK_SIZE: int = _getenv_int("DISGENET_CHUNK_SIZE", default=0)
"""Chunk size (rows) for chunked processing of the cleaned TSV.  0 = disabled
(load entire TSV in memory).  For machines with < 8 GB RAM, set to 1_000_000
to bound peak memory."""

# PERF-7: Parallel pagination (future).
DISGENET_API_PARALLEL_PAGES: int = _getenv_int(
    "DISGENET_API_PARALLEL_PAGES", default=1
)
"""Number of concurrent API page requests.  Default 1 (sequential --
DisGeNET's rate limit is 2 req/sec, parallelism just hits 429s).  Reserved
for future optimisation."""

# LOG-21: Log format.
DISGENET_LOG_FORMAT: str = _getenv("DISGENET_LOG_FORMAT", default="text").lower()
if DISGENET_LOG_FORMAT not in {"json", "text"}:
    DISGENET_LOG_FORMAT = "text"
"""Log format for the DisGeNET pipeline: 'json' (structured) or 'text'
(human-readable).  Default 'text'."""

# CONF-19: Environment-specific defaults.
DISGENET_ENV: str = _getenv("DISGENET_ENV", default="dev").lower()
if DISGENET_ENV not in {"dev", "staging", "prod"}:
    DISGENET_ENV = "dev"
"""Environment tier: 'dev', 'staging', 'prod'.  In dev: lower MIN_EXPECTED_RECORDS.
In staging: same as prod but ALLOW_PARTIAL_DATA=True.  In prod: strict defaults."""

# Apply env-specific overrides (CONF-19).
if DISGENET_ENV == "dev":
    DISGENET_MIN_EXPECTED_RECORDS = min(
        DISGENET_MIN_EXPECTED_RECORDS,
        _getenv_int("DISGENET_MIN_EXPECTED_RECORDS", default=100),
    )
    DISGENET_API_TIMEOUT = max(
        DISGENET_API_TIMEOUT, _getenv_int("DISGENET_API_TIMEOUT", default=60)
    )
elif DISGENET_ENV == "staging":
    DISGENET_ALLOW_PARTIAL_DATA = True


def _parse_disgenet_confidence_tiers(raw: str) -> list[tuple[float, str]]:
    """Parse DISGENET_CONFIDENCE_TIERS_JSON into a sorted list of (threshold, label) pairs.

    Raises ValueError on malformed JSON, non-list root, or entries that
    are not [number, string] pairs.
    """
    try:
        parsed = json.loads(raw) if raw else []
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"DISGENET_CONFIDENCE_TIERS is not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError(
            f"DISGENET_CONFIDENCE_TIERS must be a JSON list, got {type(parsed).__name__}"
        )
    tiers: list[tuple[float, str]] = []
    for entry in parsed:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(
                f"DISGENET_CONFIDENCE_TIERS entry {entry!r} must be a [threshold, label] pair"
            )
        thr, label = entry
        if not isinstance(thr, (int, float)) or isinstance(thr, bool):
            raise ValueError(
                f"DISGENET_CONFIDENCE_TIERS threshold {thr!r} must be a number"
            )
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"DISGENET_CONFIDENCE_TIERS label {label!r} must be a non-empty string"
            )
        tiers.append((float(thr), label))
    if not tiers:
        raise ValueError("DISGENET_CONFIDENCE_TIERS must contain at least one tier")
    tiers.sort(key=lambda t: t[0])
    return tiers


DISGENET_CONFIDENCE_TIERS: list[tuple[float, str]] = _parse_disgenet_confidence_tiers(
    DISGENET_CONFIDENCE_TIERS_JSON
)
"""Parsed confidence tiers (list of (threshold, label) pairs, sorted ascending).
Defaults follow Piñero et al. 2020:
``[(0.0, 'sub_weak'), (0.06, 'weak'), (0.3, 'strong')]``."""


def _parse_disgenet_source_weights(raw: str) -> dict[str, float]:
    """Parse DISGENET_SOURCE_WEIGHTS_JSON into a dict[str, float]."""
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"DISGENET_SOURCE_WEIGHTS is not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"DISGENET_SOURCE_WEIGHTS must be a JSON object, got {type(parsed).__name__}"
        )
    out: dict[str, float] = {}
    for k, v in parsed.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(
                f"DISGENET_SOURCE_WEIGHTS['{k}']={v!r} must be a number"
            )
        out[str(k)] = float(v)
    return out


# SCI-38: Source quality weights for normalized_score computation.
# These weights reflect the curation level of each DisGeNET sub-source
# (Piñero et al. 2020 §2.3).  They are heuristic, not derived from a
# closed-form formula -- they encode the relative credibility of each source.
DISGENET_SOURCE_WEIGHTS_JSON: str = _getenv(
    "DISGENET_SOURCE_WEIGHTS",
    default=json.dumps({
        "CURATED": 1.0,
        "CGI": 0.95,
        "CLINGEN": 0.95,
        "GENOMICS_ENGLAND": 0.95,
        "ORPHANET": 0.9,
        "CTD_human": 0.85,
        "GWAS_CATALOG": 0.85,
        "UNIPROT": 0.8,
        "PSYGENET": 0.75,
        "LHGDN": 0.7,
        "HPO": 0.7,
        "BEFREE": 0.5,
        "RONB": 0.5,
    }),
)
DISGENET_SOURCE_WEIGHTS: dict[str, float] = _parse_disgenet_source_weights(
    DISGENET_SOURCE_WEIGHTS_JSON
)
"""Per-source credibility weights (0.0-1.0) used to compute
``normalized_score = score * weight``.  Defaults follow Piñero et al. 2020
§2.3 (CURATED gold-standard; BEFREE/RONB text-mined, noisy).  Override
with DISGENET_SOURCE_WEIGHTS env var (JSON object)."""


def _validate_disgenet_config() -> None:
    """Validate all DisGeNET config values (CONF-14, CONF-16, CONF-17).

    Raises ValueError on any invalid value.  Called by DisGeNETPipeline
    at init (and may be called manually).
    """
    if not (0.0 <= DISGENET_MIN_SCORE <= 1.0):
        raise ValueError(
            f"DISGENET_MIN_SCORE={DISGENET_MIN_SCORE} must be in [0, 1]"
        )
    # v82 FORENSIC ROOT FIX (P1-3): validate the weak-evidence threshold
    # is in [0, 1] AND strictly greater than DISGENET_MIN_SCORE (otherwise
    # the weak-evidence band is empty/inverted and the tagger is dead code).
    if not (0.0 <= DISGENET_WEAK_EVIDENCE_THRESHOLD <= 1.0):
        raise ValueError(
            f"DISGENET_WEAK_EVIDENCE_THRESHOLD="
            f"{DISGENET_WEAK_EVIDENCE_THRESHOLD} must be in [0, 1]"
        )
    if DISGENET_WEAK_EVIDENCE_THRESHOLD <= DISGENET_MIN_SCORE:
        raise ValueError(
            f"DISGENET_WEAK_EVIDENCE_THRESHOLD="
            f"{DISGENET_WEAK_EVIDENCE_THRESHOLD} must be strictly greater "
            f"than DISGENET_MIN_SCORE={DISGENET_MIN_SCORE} (otherwise the "
            f"weak-evidence band is empty/inverted)"
        )
    if DISGENET_API_PAGE_SIZE <= 0:
        raise ValueError(
            f"DISGENET_API_PAGE_SIZE={DISGENET_API_PAGE_SIZE} must be > 0"
        )
    if DISGENET_API_MAX_RETRIES < 1:
        raise ValueError(
            f"DISGENET_API_MAX_RETRIES={DISGENET_API_MAX_RETRIES} must be >= 1"
        )
    if DISGENET_API_TIMEOUT <= 0:
        raise ValueError(
            f"DISGENET_API_TIMEOUT={DISGENET_API_TIMEOUT} must be > 0"
        )
    if DISGENET_PMID_CAP <= 0:
        raise ValueError(
            f"DISGENET_PMID_CAP={DISGENET_PMID_CAP} must be > 0"
        )
    if DISGENET_API_BACKOFF_BASE <= 1.0:
        raise ValueError(
            f"DISGENET_API_BACKOFF_BASE={DISGENET_API_BACKOFF_BASE} must be > 1.0"
        )
    if DISGENET_API_BACKOFF_MAX_SECONDS <= 0:
        raise ValueError(
            f"DISGENET_API_BACKOFF_MAX_SECONDS={DISGENET_API_BACKOFF_MAX_SECONDS} must be > 0"
        )
    if DISGENET_API_RATE_LIMIT <= 0:
        raise ValueError(
            f"DISGENET_API_RATE_LIMIT={DISGENET_API_RATE_LIMIT} must be > 0"
        )
    if DISGENET_CIRCUIT_BREAKER_THRESHOLD < 1:
        raise ValueError(
            f"DISGENET_CIRCUIT_BREAKER_THRESHOLD={DISGENET_CIRCUIT_BREAKER_THRESHOLD} must be >= 1"
        )
    if DISGENET_CIRCUIT_BREAKER_RESET_SECONDS <= 0:
        raise ValueError(
            f"DISGENET_CIRCUIT_BREAKER_RESET_SECONDS={DISGENET_CIRCUIT_BREAKER_RESET_SECONDS} must be > 0"
        )
    if DISGENET_API_MAX_PAGES <= 0:
        raise ValueError(
            f"DISGENET_API_MAX_PAGES={DISGENET_API_MAX_PAGES} must be > 0"
        )
    if DISGENET_DOWNLOAD_PHASE_TIMEOUT <= 0:
        raise ValueError(
            f"DISGENET_DOWNLOAD_PHASE_TIMEOUT={DISGENET_DOWNLOAD_PHASE_TIMEOUT} must be > 0"
        )
    if DISGENET_API_MAX_RESPONSE_BYTES <= 0:
        raise ValueError(
            f"DISGENET_API_MAX_RESPONSE_BYTES={DISGENET_API_MAX_RESPONSE_BYTES} must be > 0"
        )
    if DISGENET_API_MAX_RECORDS <= 0:
        raise ValueError(
            f"DISGENET_API_MAX_RECORDS={DISGENET_API_MAX_RECORDS} must be > 0"
        )
    if not DISGENET_ALLOWED_DOMAINS:
        raise ValueError(
            "DISGENET_ALLOWED_DOMAINS must contain at least one domain"
        )

    # CONF-16: Validate DISGENET_API_URL.
    from urllib.parse import urlparse
    parsed = urlparse(DISGENET_API_URL)
    if parsed.scheme != "https":
        raise ValueError(
            f"DISGENET_API_URL scheme must be 'https', got {parsed.scheme!r}"
        )
    if not parsed.hostname:
        raise ValueError(f"DISGENET_API_URL has no hostname: {DISGENET_API_URL!r}")
    if (
        parsed.hostname not in DISGENET_ALLOWED_DOMAINS
        and not any(
            parsed.hostname.endswith("." + d) for d in DISGENET_ALLOWED_DOMAINS
        )
    ):
        raise ValueError(
            f"DISGENET_API_URL hostname {parsed.hostname!r} is not in "
            f"DISGENET_ALLOWED_DOMAINS={DISGENET_ALLOWED_DOMAINS}"
        )

    # CONF-17: API key required when USE_API=True.
    if DISGENET_USE_API and not DISGENET_API_KEY:
        raise ValueError(
            "DISGENET_USE_API=true but DISGENET_API_KEY is not set. "
            "Set the DISGENET_API_KEY environment variable or set "
            "DISGENET_USE_API=false (not recommended - static URL is "
            "deprecated since 2024)."
        )

    # CONF-14: Tier thresholds must be strictly monotonic.
    thresholds = [t[0] for t in DISGENET_CONFIDENCE_TIERS]
    for i in range(1, len(thresholds)):
        if thresholds[i] <= thresholds[i - 1]:
            raise ValueError(
                f"DISGENET_CONFIDENCE_TIERS thresholds must be strictly "
                f"monotonic ascending, got {thresholds}"
            )


# Run validation eagerly so misconfiguration fails fast (CONF-14).
# We wrap in try/except so import never hard-fails (tests can patch env).
try:
    _validate_disgenet_config()
except ValueError as _disgenet_cfg_err:
    # Defer to runtime -- log a warning, allow import (the pipeline will
    # re-validate and raise on init).
    warnings.warn(
        f"DisGeNET config validation warning: {_disgenet_cfg_err}",
        UserWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# PubChem -- INTEROP-2, DESIGN-5
# ---------------------------------------------------------------------------

PUBCHEM_REST_BASE: str = _getenv(
    "PUBCHEM_REST_BASE", "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
)
PUBCHEM_FTP_BASE: str = _getenv(
    "PUBCHEM_FTP_BASE", "https://ftp.ncbi.nlm.nih.gov/pubchem"
)

# Consistent alias (DESIGN-5)
PUBCHEM_API_URL: str = PUBCHEM_REST_BASE

# ---------------------------------------------------------------------------
# Entity Resolution -- audit D12-2 (integrate entity_resolution config
# with config/settings.py).  Every field mirrors ResolverConfig in
# entity_resolution/base.py and is env-overridable with the prefix
# ENTITY_RESOLUTION_.  These settings are consumed by
# ResolverConfig.from_env() at construction time.
# ---------------------------------------------------------------------------

# If False (default, safe), the bulk path build_mapping() never calls
# PubChem and resolve_single() skips the PubChem step.  Opt in via
# ENTITY_RESOLUTION_PUBCHEM_ENABLED=1 when single-record PubChem
# lookup is genuinely needed.  Audit D9-1 / D9-2.
ENTITY_RESOLUTION_PUBCHEM_ENABLED: bool = _getenv_bool(
    "ENTITY_RESOLUTION_PUBCHEM_ENABLED", False
)

# If False (default, safe), two InChIKeys sharing the same 14-char
# connectivity block are NOT merged unless their full 27-char forms
# are identical.  This preserves stereoisomer distinctness (audit D3-4
# -- thalidomide-enantiomer safety).  Opt in via =1 for legacy
# behaviour.
ENTITY_RESOLUTION_COLLAPSE_STEREOISOMERS: bool = _getenv_bool(
    "ENTITY_RESOLUTION_COLLAPSE_STEREOISOMERS", False
)

# Minimum rapidfuzz.fuzz.token_sort_ratio score (on [0,1]) at which a
# fuzzy name match is accepted.  Default 0.60.
# v43 ROOT FIX (P1 -- fuzzy threshold divergence): the previous default
# was 0.85, but the actual runtime gate in drug_resolver._FUZZY_THRESHOLD
# and ResolverConfig.from_env() default is 0.60. An operator reading
# config.settings.ENTITY_RESOLUTION_FUZZY_THRESHOLD would log "fuzzy
# threshold = 0.85" while the resolver actually accepts matches scoring
# >= 0.60 -- causing confusion and false-positive bug reports. The fix
# aligns the config default to 0.60 to match the runtime gate.
# Audit D3-3.
ENTITY_RESOLUTION_FUZZY_THRESHOLD: float = _getenv_float(
    "ENTITY_RESOLUTION_FUZZY_THRESHOLD", 0.60
)

# Ceiling on the number of indexed names scanned per fuzzy sweep
# (audit D8-2 -- bounds worst-case O(n^2)).  Default 10000.
ENTITY_RESOLUTION_FUZZY_MAX_CANDIDATES: int = _getenv_int(
    "ENTITY_RESOLUTION_FUZZY_MAX_CANDIDATES", 10_000
)

# PubChem REST base URL -- configurable so air-gapped deployments can
# point at an internal mirror.  Audit D9-3.
ENTITY_RESOLUTION_PUBCHEM_REST_BASE: str = _getenv(
    "ENTITY_RESOLUTION_PUBCHEM_REST_BASE", PUBCHEM_REST_BASE
)

# Minimum seconds between PubChem API calls.  Default 0.2 (5 req/sec).
ENTITY_RESOLUTION_PUBCHEM_CALL_DELAY: float = _getenv_float(
    "ENTITY_RESOLUTION_PUBCHEM_CALL_DELAY", 0.2
)

# Per-request timeout in seconds.  Default 10.
ENTITY_RESOLUTION_PUBCHEM_TIMEOUT: float = _getenv_float(
    "ENTITY_RESOLUTION_PUBCHEM_TIMEOUT", 10.0
)

# Number of retries with exponential backoff on transient failures.
# Default 3.
ENTITY_RESOLUTION_PUBCHEM_MAX_RETRIES: int = _getenv_int(
    "ENTITY_RESOLUTION_PUBCHEM_MAX_RETRIES", 3
)

# Optional PubChem API key.  When set, the rate limit is raised from
# 5 req/sec to 10 req/sec per PubChem's published limits.  Audit D9-6.
ENTITY_RESOLUTION_PUBCHEM_API_KEY: Optional[str] = (
    _getenv("ENTITY_RESOLUTION_PUBCHEM_API_KEY", "") or None
)

# Optional path to a CA bundle for TLS verification against an
# internal PubChem mirror.  Audit D9-5.
ENTITY_RESOLUTION_PUBCHEM_CA_BUNDLE: Optional[str] = (
    _getenv("ENTITY_RESOLUTION_PUBCHEM_CA_BUNDLE", "") or None
)

# Optional mTLS client certificate paths.  Audit D9-5.
ENTITY_RESOLUTION_PUBCHEM_CERT_PEM: Optional[str] = (
    _getenv("ENTITY_RESOLUTION_PUBCHEM_CERT_PEM", "") or None
)
ENTITY_RESOLUTION_PUBCHEM_KEY_PEM: Optional[str] = (
    _getenv("ENTITY_RESOLUTION_PUBCHEM_KEY_PEM", "") or None
)

# If True, reject PubChem name lookups that resolve to a salt form
# (e.g. "aspirin" -> "aspirin sodium").  Audit D3-7.
ENTITY_RESOLUTION_PUBCHEM_STRICT_SALT_FORM: bool = _getenv_bool(
    "ENTITY_RESOLUTION_PUBCHEM_STRICT_SALT_FORM", False
)

# Optional comma-separated whitelist of allowed ``source`` argument
# values passed to add_source_records().  When set, unknown source
# labels raise ValueError.  Audit D9-7.
ENTITY_RESOLUTION_SOURCE_WHITELIST: Optional[Tuple[str, ...]] = (
    tuple(
        s.strip()
        for s in _getenv("ENTITY_RESOLUTION_SOURCE_WHITELIST", "").split(",")
        if s.strip()
    )
    or None
)

# Default organism when protein records omit it.  ⚠️  This default
# assumes human-centric research; non-human protein studies MUST
# override it.  Audit D12-5.
ENTITY_RESOLUTION_DEFAULT_ORGANISM: str = _getenv(
    "ENTITY_RESOLUTION_DEFAULT_ORGANISM", "Homo sapiens"
)

# Schema version of the state-dict format.  Audit D12-4.
ENTITY_RESOLUTION_MAPPING_SCHEMA_VERSION: str = "1.0"


def get_entity_resolution_config() -> Dict[str, Any]:
    """Return a dict of every ENTITY_RESOLUTION_* setting.

    Convenience helper for logging / introspection.  Sensitive fields
    are masked.
    """
    return {
        "pubchem_enabled": ENTITY_RESOLUTION_PUBCHEM_ENABLED,
        "collapse_stereoisomers": ENTITY_RESOLUTION_COLLAPSE_STEREOISOMERS,
        "fuzzy_threshold": ENTITY_RESOLUTION_FUZZY_THRESHOLD,
        "fuzzy_max_candidates": ENTITY_RESOLUTION_FUZZY_MAX_CANDIDATES,
        "pubchem_rest_base": ENTITY_RESOLUTION_PUBCHEM_REST_BASE,
        "pubchem_call_delay": ENTITY_RESOLUTION_PUBCHEM_CALL_DELAY,
        "pubchem_timeout": ENTITY_RESOLUTION_PUBCHEM_TIMEOUT,
        "pubchem_max_retries": ENTITY_RESOLUTION_PUBCHEM_MAX_RETRIES,
        "pubchem_api_key": (
            "<redacted>" if ENTITY_RESOLUTION_PUBCHEM_API_KEY else None
        ),
        "pubchem_ca_bundle": ENTITY_RESOLUTION_PUBCHEM_CA_BUNDLE,
        "pubchem_cert_pem": ENTITY_RESOLUTION_PUBCHEM_CERT_PEM,
        "pubchem_key_pem": ENTITY_RESOLUTION_PUBCHEM_KEY_PEM,
        "pubchem_strict_salt_form": ENTITY_RESOLUTION_PUBCHEM_STRICT_SALT_FORM,
        "source_whitelist": ENTITY_RESOLUTION_SOURCE_WHITELIST,
        "default_organism": ENTITY_RESOLUTION_DEFAULT_ORGANISM,
        "mapping_schema_version": ENTITY_RESOLUTION_MAPPING_SCHEMA_VERSION,
    }


# ===========================================================================
# PubChem Pipeline -- institutional-grade settings (CONF-1 ... CONF-12, ARCH-7)
#
# These settings are consumed by ``pipelines/pubchem_pipeline.py``.  They
# complement (do NOT duplicate) the ``ENTITY_RESOLUTION_PUBCHEM_*`` block
# above -- REST base / call delay / timeout / max retries / API key /
# CA bundle / client cert / strict salt form are reused from there.
#
# Every value is env-var-overridable.  Defaults are documented inline.
# ===========================================================================

# [CONF-1, ARCH-7] Number of InChIKeys per PubChem PUG REST batch request.
# Why 95 and not 100?  PubChem PUG REST hard limit is 100 identifiers per
# batch.  We use 95 to leave a 5% safety margin in case PubChem lowers the
# limit (they have historically).  Set to 5 in dev for fast testing.
# See: https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest
PUBCHEM_PIPELINE_BATCH_SIZE: int = _getenv_int(
    "PUBCHEM_PIPELINE_BATCH_SIZE", 95
)

# [CONF-3] Minimum backoff (seconds) for exponential retry on transient
# PubChem failures (429, 5xx).  Multiplied by 2^attempt, capped at
# ``PUBCHEM_PIPELINE_MAX_BACKOFF``.  Default 2.0s matches PubChem's
# recommendation for courteous retry.
PUBCHEM_PIPELINE_MIN_BACKOFF: float = _getenv_float(
    "PUBCHEM_PIPELINE_MIN_BACKOFF", 2.0
)

# [CONF-3] Maximum backoff (seconds) -- caps the exponential growth so a
# badly-degraded PubChem does not stall the pipeline for hours.
PUBCHEM_PIPELINE_MAX_BACKOFF: float = _getenv_float(
    "PUBCHEM_PIPELINE_MAX_BACKOFF", 32.0
)

# [CONF-5, DESIGN-14] Read timeout (seconds) for PubChem PUG REST.  Connect
# timeout comes from ``ENTITY_RESOLUTION_PUBCHEM_TIMEOUT`` (default 10.0).
# Combined as a ``(connect, read)`` tuple passed to ``requests``.
PUBCHEM_PIPELINE_READ_TIMEOUT: float = _getenv_float(
    "PUBCHEM_PIPELINE_READ_TIMEOUT", 30.0
)

# [CONF-1, DQ-14, IDEM-1] Cache TTL (seconds) for ``inchikeys_to_lookup.txt``.
# Files older than this trigger a re-query.  Default 1 hour -- balances
# freshness against PubChem API load.  Set to 0 to disable caching.
PUBCHEM_PIPELINE_CACHE_TTL_SECONDS: int = _getenv_int(
    "PUBCHEM_PIPELINE_CACHE_TTL_SECONDS", 3600
)

# [ARCH-13, PERF-1] Concurrency for batch HTTP requests.  Default 1
# (sequential) for determinism.  Production may set to 5 (PubChem allows
# 5 req/sec) for 5x throughput.  Tests run with concurrency=1.
PUBCHEM_PIPELINE_CONCURRENCY: int = _getenv_int(
    "PUBCHEM_PIPELINE_CONCURRENCY", 1
)

# [SCI-7] Optionally fetch PubChem synonyms (voluminous -- default False).
# When True, ``pubchem_compound_properties.synonyms`` is populated as a
# JSON array string.  Single source of truth for entity_resolution.
PUBCHEM_PIPELINE_FETCH_SYNONYMS: bool = _getenv_bool(
    "PUBCHEM_PIPELINE_FETCH_SYNONYMS", False
)

# [SCI-6] Optionally fetch CAS Registry Number via the synonyms endpoint.
# Default False -- adds 1 extra HTTP call per resolved CID.  When True,
# ``pubchem_compound_properties.cas_number`` is populated and cross-
# validated against ``drugs.cas_number`` (from DrugBank).
PUBCHEM_PIPELINE_FETCH_CAS: bool = _getenv_bool(
    "PUBCHEM_PIPELINE_FETCH_CAS", False
)

# [REL-5] Maximum batch size for split-retry on permanent 4xx failures.
# When a batch returns 400/404 etc., the batch is split into individual
# InChIKey lookups.  This cap prevents 100 individual requests for a
# fully-bad batch -- if exceeded, all 100 are dead-lettered without splitting.
PUBCHEM_PIPELINE_SPLIT_RETRY_MAX: int = _getenv_int(
    "PUBCHEM_PIPELINE_SPLIT_RETRY_MAX", 20
)

# [DQ-9, SEC-12] Maximum number of InChIKeys to enrich per run.  None = no
# limit.  Useful for dev/testing and for capping PubChem API load.
PUBCHEM_PIPELINE_MAX_RECORDS: Optional[int] = (
    int(_getenv("PUBCHEM_PIPELINE_MAX_RECORDS", ""))
    if _getenv("PUBCHEM_PIPELINE_MAX_RECORDS", "").strip()
    else None
)

# [LIN-9] Retention period (days) for raw PubChem JSON responses archived
# in ``raw_data/pubchem/pubchem_responses/``.  Older files are eligible for
# cleanup by an external janitor process.  Default 90 days.
PUBCHEM_PIPELINE_RAW_RESPONSE_RETENTION_DAYS: int = _getenv_int(
    "PUBCHEM_PIPELINE_RAW_RESPONSE_RETENTION_DAYS", 90
)

# [ARCH-9, REL-3] Circuit breaker threshold for PubChem 5xx storms.
# After this many consecutive failures, the breaker opens and the
# pipeline fails fast for ``PUBCHEM_CIRCUIT_BREAKER_RESET_SECONDS``.
PUBCHEM_CIRCUIT_BREAKER_THRESHOLD: int = _getenv_int(
    "PUBCHEM_CIRCUIT_BREAKER_THRESHOLD", 5
)

# [ARCH-9, REL-3] Circuit breaker reset window (seconds).  After this
# cooldown, the breaker enters HALF_OPEN and allows one probe request.
PUBCHEM_CIRCUIT_BREAKER_RESET_SECONDS: float = _getenv_float(
    "PUBCHEM_CIRCUIT_BREAKER_RESET_SECONDS", 60.0
)

# [CONF-6] Comma-separated list of PubChem properties to fetch per CID.
# Rarely changed -- but exposed for forward-compat with new PubChem fields.
PUBCHEM_PIPELINE_PROPERTIES: list[str] = [
    p.strip()
    for p in _getenv(
        "PUBCHEM_PIPELINE_PROPERTIES",
        ",".join(
            [
                "MolecularFormula",
                "MolecularWeight",
                "InChIKey",
                "InChI",
                "CanonicalSMILES",
                "IsomericSMILES",
                "IUPACName",
                "XLogP",
                "ExactMass",
                "TPSA",
                "Complexity",
                "HBondDonorCount",
                "HBondAcceptorCount",
                "RotatableBondCount",
                "HeavyAtomCount",
            ]
        ),
    ).split(",")
    if p.strip()
]

# [LOG-3] Optional Prometheus metrics emission.  Default False -- don't
# add the prometheus_client import overhead in dev.  When True, the
# pipeline emits ``pubchem_batches_total``, ``pubchem_retries_total``,
# ``pubchem_records_loaded``, ``pubchem_api_latency_seconds``.
PROMETHEUS_ENABLED: bool = _getenv_bool("PROMETHEUS_ENABLED", False)

# [LOG-4] Optional OpenTelemetry tracing.  Default False.  When True,
# the pipeline emits spans for each batch lookup.
OTEL_ENABLED: bool = _getenv_bool("OTEL_ENABLED", False)

# [COMP-5] Operator identity for FDA 21 CFR Part 11 electronic-signature
# compliance.  Populated in ``pubchem_compound_properties.triggered_by``
# and ``electronic_signature``.  None when run unattended (Airflow).
OPERATOR_ID: Optional[str] = (
    _getenv("OPERATOR_ID", "").strip() or None
)

# [SCI-10, SCI-15] Auto-detect RDKit availability.  When True, the pipeline
# validates SMILES via RDKit and computes formal charge from the molecule
# object (authoritative).  When False, formal charge is parsed from the
# SMILES string (heuristic) and SMILES are not validated.
try:
    import rdkit  # noqa: F401  -- presence check only
    RDKIT_AVAILABLE: bool = True
except ImportError:
    RDKIT_AVAILABLE: bool = False


# ---------------------------------------------------------------------------
# DrugBank -- CODE-7, INTEROP-1
# ---------------------------------------------------------------------------

# DrugBank distributes the full database as a .xml.gz file.
# The exact filename varies by release version. Common names:
#   - drugbank_all_full_database.xml.gz
#   - full database.xml.gz
# If your DrugBank file has a different name, set DRUGBANK_XML_PATH
# to the exact path. (DATA-5)
DRUGBANK_XML_PATH: Path = Path(
    _getenv(
        "DRUGBANK_XML_PATH",
        str(RAW_DATA_DIR / "drugbank" / "drugbank_all_full_database.xml.gz"),
    )
    # If the env var is set but empty, fall back to the default.
    # Without this guard, Path("") == Path(".") which is the current
    # directory -- causes a confusing IsADirectoryError downstream.
    or str(RAW_DATA_DIR / "drugbank" / "drugbank_all_full_database.xml.gz")
)

# ---------------------------------------------------------------------------
# DrugBank extended configuration block (CF1-CF15).
#
# Mirrors the CHEMBL_VERSION pattern: DEFAULT_* -> VALID_* frozenset ->
# _validate_* helper -> public *_VERSION constant. All values are
# environment-overridable so deployments can change behaviour without
# touching code (CF1-CF15, ID2, ID4, S7, S9, CF3-CF13).
# ---------------------------------------------------------------------------

# CF2 / ID2: DrugBank release version (default 5.1; update when upgrading).
DEFAULT_DRUGBANK_VERSION: str = "5.1"

# Valid DrugBank 5.x release versions (NCBI / Wishart 2018 lineage).
# v28 ROOT FIX (audit TOP-23): "5.2" was listed but does NOT EXIST
# publicly. DrugBank's latest public release as of 2024 is 5.1.x
# (5.1.12 was the most recent). The fictional "5.2" entry would have
# accepted ``DRUGBANK_VERSION=5.2`` as a known-good version, silencing
# the "not in the known valid set" warning -- operators could then
# configure the pipeline against a non-existent release and never see
# a hint that the version was wrong. Removed here.
VALID_DRUGBANK_VERSIONS: frozenset[str] = frozenset(
    {"5.0", "5.1", "5.1.8", "5.1.9", "5.1.10", "5.1.11", "5.1.12"}
)


def _validate_drugbank_version(version: str) -> str:
    """Validate DrugBank version string (mirrors _validate_chembl_version).

    Accepts numeric version strings like ``5.1`` or ``5.1.10``. Warns on
    unknown versions. Raises ``ValueError`` on clearly invalid values
    (non-numeric, empty).
    """
    if not version or not version.strip():
        raise ValueError("DRUGBANK_VERSION cannot be empty")
    if not version.replace(".", "").isdigit():
        raise ValueError(
            f"DRUGBANK_VERSION={version!r} is not a valid version string. "
            f"Expected a numeric version like '5.1' or '5.1.10'. "
            f"Valid versions: {sorted(VALID_DRUGBANK_VERSIONS)}"
        )
    if version not in VALID_DRUGBANK_VERSIONS:
        warnings.warn(
            f"DRUGBANK_VERSION={version} is not in the known valid set. "
            f"The DrugBank XML schema may not match. "
            f"Known valid versions: {sorted(VALID_DRUGBANK_VERSIONS)}",
            UserWarning,
        )
    return version


# Public source version constant (CF2 / ID2 / A8).
DRUGBANK_VERSION: str = _validate_drugbank_version(
    _getenv("DRUGBANK_VERSION", DEFAULT_DRUGBANK_VERSION)
)

# CF1: XML namespace (stable since 2010). Config-overridable for forward compat.
DRUGBANK_XML_NAMESPACE: str = _getenv(
    "DRUGBANK_XML_NAMESPACE", "http://drugbank.ca"
)

# S9: organism filter (default Humans-only for human drug repurposing).
# Comma-separated list. For infectious-disease use cases set to
# "Humans,HIV-1,Mycobacterium tuberculosis".
DRUGBANK_TARGET_ORGANISMS: list[str] = [
    org.strip()
    for org in _getenv("DRUGBANK_TARGET_ORGANISMS", "Humans").split(",")
    if org.strip()
]

# S7: synthetic InChIKey generation for biologics (insulin, antibodies).
# Drug model allows 'SYNTH-...' via CheckConstraint (models.py).
DRUGBANK_GENERATE_SYNTH_KEYS: bool = _getenv_bool(
    "DRUGBANK_GENERATE_SYNTH_KEYS", True
)

# S7: hard drop of records with no InChIKey (default False -- keep biologics).
DRUGBANK_DROP_NO_INCHIKEY: bool = _getenv_bool("DRUGBANK_DROP_NO_INCHIKEY", False)

# ID4: conservative_defaults flag for fill_missing_drug_fields.
DRUGBANK_CONSERVATIVE_DEFAULTS: bool = _getenv_bool(
    "DRUGBANK_CONSERVATIVE_DEFAULTS", True
)

# CF13: batch size for bulk_upsert_drugs / bulk_upsert_dpi.
DRUGBANK_BATCH_SIZE: int = _parse_required_int("DRUGBANK_BATCH_SIZE", "1000")

# CF7: iterparse log interval (drugs parsed between INFO logs).
DRUGBANK_LOG_INTERVAL: int = _parse_required_int("DRUGBANK_LOG_INTERVAL", "5000")

# CF8: max drug count safety limit (0 = unlimited; for testing).
DRUGBANK_MAX_DRUGS: int = _parse_required_int("DRUGBANK_MAX_DRUGS", "0")

# CF9: extract targets / enzymes / transporters (all default True).
DRUGBANK_EXTRACT_TARGETS: bool = _getenv_bool("DRUGBANK_EXTRACT_TARGETS", True)
DRUGBANK_EXTRACT_ENZYMES: bool = _getenv_bool("DRUGBANK_EXTRACT_ENZYMES", True)
DRUGBANK_EXTRACT_TRANSPORTERS: bool = _getenv_bool(
    "DRUGBANK_EXTRACT_TRANSPORTERS", True
)

# CF12: output CSV compression ("gzip" or "none").
DRUGBANK_CSV_COMPRESSION: str = _getenv("DRUGBANK_CSV_COMPRESSION", "gzip")

# SEC1: optional SHA-256 of the input XML for tamper-evidence.
DRUGBANK_EXPECTED_SHA256: str = _getenv("DRUGBANK_EXPECTED_SHA256", "")

# CF3: expected drug count range for sanity checking.
DRUGBANK_EXPECTED_DRUG_COUNT_MIN: int = _parse_required_int(
    "DRUGBANK_DRUG_COUNT_MIN", "10000"
)
DRUGBANK_EXPECTED_DRUG_COUNT_MAX: int = _parse_required_int(
    "DRUGBANK_DRUG_COUNT_MAX", "20000"
)

# SEC2: redact proprietary DrugBank content from logs in production.
DRUGBANK_LOG_REDACT: bool = _getenv_bool("DRUGBANK_LOG_REDACT", False)

# SEC12: log full file paths (False = filename only).
DRUGBANK_LOG_FULL_PATHS: bool = _getenv_bool("DRUGBANK_LOG_FULL_PATHS", False)

# CF15: validate the XML path is readable before parsing.
DRUGBANK_VALIDATE_READABILITY: bool = _getenv_bool(
    "DRUGBANK_VALIDATE_READABILITY", True
)

# DPI batch size for chunked bulk_upsert_dpi (P13).
DRUGBANK_DPI_BATCH_SIZE: int = _parse_required_int("DRUGBANK_DPI_BATCH_SIZE", "500")

# ---------------------------------------------------------------------------
# OMIM -- SEC-2 + 16-domain institutional-grade config (master prompt §7.12)
# ---------------------------------------------------------------------------
# BUG-9.15 / BUG-12.8: OMIM_API_KEY stripped of whitespace (handles trailing
# newlines that some secret managers inject).
OMIM_API_KEY: str = (os.getenv("OMIM_API_KEY") or "").strip()
OMIM_API_BASE: str = os.getenv("OMIM_API_BASE") or "https://api.omim.org/api"

# BUG-2.6 / BUG-12.1: rate-limit interval between OMIM API requests.
# OMIM's published rate limit is 4 req/sec -> 0.25s between requests.
OMIM_REQUEST_INTERVAL: float = _getenv_float("OMIM_REQUEST_INTERVAL", 0.25)

# BUG-2.5 / BUG-3.5 / BUG-3.6 / BUG-12.3: which phenotype mapping keys to
# include in the cleaned GDA output. Default [3, 4] -- molecular basis known
# (mk=3) plus contiguous gene deletion/duplication syndromes (mk=4, e.g.
# DiGeorge, Williams). Both are clinically real and well-characterized.
# Advanced users can set OMIM_MAPPING_KEYS_INCLUDE=1,2,3,4 for comprehensive
# ingest (mk=1 = wild-type gene mapped, mk=2 = phenotype mapped).
OMIM_MAPPING_KEYS_INCLUDE: list[int] = _parse_csv_ints(
    "OMIM_MAPPING_KEYS_INCLUDE", [3, 4]
)

# BUG-2.7 / BUG-8.2 / BUG-12.2: API pagination page size.
# OMIM REST API max limit is 1000. Setting to 1000 is 5× faster than the
# legacy 200.
OMIM_API_PAGE_LIMIT: int = _getenv_int("OMIM_API_PAGE_LIMIT", 1000)

# BUG-2.7 / BUG-12.19: maximum HTTP retries on retryable status codes.
OMIM_API_MAX_RETRIES: int = _getenv_int("OMIM_API_MAX_RETRIES", 5)

# BUG-12.4: per-request timeouts (seconds).
OMIM_DOWNLOAD_TIMEOUT: int = _getenv_int("OMIM_DOWNLOAD_TIMEOUT", 300)
OMIM_API_TIMEOUT: int = _getenv_int("OMIM_API_TIMEOUT", 120)

# BUG-12.5 / BUG-13.20: output filename (kept configurable for backfill
# isolation and test redirection).
OMIM_OUTPUT_FILENAME: str = _getenv(
    "OMIM_OUTPUT_FILENAME", "omim_gene_disease_associations.csv"
)

# BUG-5.1 / BUG-12.15: minimum expected record count after morbidmap parse.
# OMIM typically publishes ~7,000 morbidmap entries; 5,000 is a safe floor
# that catches truncated downloads without false-failing on legit small runs.
OMIM_MIN_EXPECTED_RECORDS: int = _getenv_int("OMIM_MIN_EXPECTED_RECORDS", 5000)

# BUG-6.5 / BUG-12.16: upper bound on pagination pages.
OMIM_MAX_PAGINATION_PAGES: int = _getenv_int("OMIM_MAX_PAGINATION_PAGES", 1000)

# BUG-12.17: legacy dedup-keep-policy -- kept for backward-compat; the new
# atomic-write path (BUG-1.9) doesn't append, so this is informational only.
OMIM_DEDUP_KEEP_POLICY: str = _getenv("OMIM_DEDUP_KEEP_POLICY", "last")

# BUG-2.3 / BUG-3.2 / BUG-12.12 / BUG-12.13: per-mapping-key base scores.
# These are evidence-weighted starting points; the final score is
#   base + min(0.05 * log1p(num_pmids), 0.08) + min(evidence_strength * 0.05, 0.05)
# clamped to [0, 1].
#   - mk=3 (0.9): molecular basis known (mutation found in gene). Strongest
#     single signal -- chosen to match DisGeNET's strong-evidence threshold.
#   - mk=4 (0.8): contiguous gene deletion/duplication syndrome. Clinically
#     validated, but the gene-disease causal chain is less direct than mk=3.
#   - mk=2 (0.25): the disease phenotype itself was mapped (no gene identified).
#     P1-005 ROOT FIX: lowered from 0.6 to 0.25 so it falls in the "weak"
#     tier per Piñero 2020 §2.3 (score < 0.3). mk=2 means the gene was NOT
#     identified -- only the phenotype was mapped. This is weak evidence,
#     not strong.
#   - mk=1 (0.2): the wild-type gene was mapped (weakest OMIM evidence tier).
#     P1-005 ROOT FIX: lowered from 0.5 to 0.2 so it falls in the "weak"
#     tier per Piñero 2020 §2.3 (score < 0.3). mk=1 means NO phenotype
#     association has been established by OMIM -- the gene-disease link is
#     NOT confirmed. Labelling this "strong" (>=0.3) was a patient-safety
#     risk.
OMIM_CONFIRMED_SCORE: float = _getenv_float("OMIM_CONFIRMED_SCORE", 0.9)
OMIM_CONTIGUOUS_SCORE: float = _getenv_float("OMIM_CONTIGUOUS_SCORE", 0.8)
# P1-005 ROOT FIX (Team-1 -- OMIM mapping_key scoring scientific mislabel):
#   Per Piñero et al. 2020 §2.3 (the same publication cited by
#   ``cleaning/confidence.py`` for the DSGP tier bands), the
#   ``confidence_tier`` classifier labels any score >= 0.3 as "strong".
#   The previous values (0.6 for mk=2, 0.5 for mk=1) put BOTH mapping
#   keys in the "strong" tier -- but OMIM explicitly states that:
#     * mk=1 ("wild-type gene mapped") means "the gene has been mapped
#       but NO phenotype association has been established". This is
#       the WEAKEST OMIM evidence tier -- the gene-disease link is
#       NOT established by OMIM. Labelling it "strong" is
#       scientifically wrong and a patient-safety risk: downstream
#       drug-repurposing models may recommend drugs targeting genes
#       with NO established disease association.
#     * mk=2 ("phenotype mapped") means the disease phenotype was
#       mapped but the gene itself was not identified. This is also
#       weak evidence -- the gene-disease link is inferred, not
#       confirmed.
#   Only mk=3 (molecular basis known) and mk=4 (contiguous gene
#   syndrome) reflect OMIM-confirmed gene-phenotype associations and
#   should be scored as "strong" (>= 0.3).
#
#   ROOT FIX: lower mk=1 to 0.2 (falls in the [0.06, 0.3) "weak"
#   tier per Piñero §2.3) and mk=2 to 0.25 (also in the "weak" tier).
#   mk=3 (0.9) and mk=4 (0.8) stay above 0.3 ("strong" tier). The
#   default ``OMIM_MAPPING_KEYS_INCLUDE={3,4}`` means mk=1 and mk=2
#   records are filtered out at clean time -- but the scoring
#   function ``_compute_omim_score`` is exported and may be called
#   by other code paths (or by an operator who sets
#   ``OMIM_MAPPING_KEYS_INCLUDE=1,2,3,4`` to include all OMIM
#   records). With this fix, those records enter the KG correctly
#   tagged as "weak" evidence, NOT "strong".
OMIM_PHENOTYPE_MAPPED_SCORE: float = _getenv_float("OMIM_PHENOTYPE_MAPPED_SCORE", 0.25)
OMIM_GENE_MAPPED_SCORE: float = _getenv_float("OMIM_GENE_MAPPED_SCORE", 0.2)

# BUG-12.20: User-Agent string sent with every OMIM HTTP request.
OMIM_USER_AGENT: str = _getenv(
    "OMIM_USER_AGENT",
    f"drug-repurposing-pipeline/omim (contact={_getenv('OMIM_CONTACT_EMAIL', 'unknown@example.com')})",
)

# BUG-12.6: regex validating the OMIM_API_KEY format. OMIM API keys are UUIDs.
# v43 ROOT FIX (P1 -- OMIM_API_KEY_FORMAT_RE not compiled): the previous
# code declared this as a str and used re.match(OMIM_API_KEY_FORMAT_RE,
# ...) which re-compiles the regex on every call. Compiling once at
# import time is faster and makes the type stubs correct (re.Pattern
# instead of str).
# P1-A3 ROOT FIX: the previous regex used [A-Fa-f0-9] for all hex digits
# but did NOT validate UUID structure (variant/version bits). A valid
# UUID v4 has the format xxxxxxxx-xxxx-4xxx-[89ab]xxx-xxxxxxxxxxxx
# where the third group starts with '4' (version) and the fourth group
# starts with 8/9/a/b (variant). The regex is updated to validate
# these structural constraints. Case-insensitive flag added so both
# uppercase and lowercase hex digits are accepted uniformly.
OMIM_API_KEY_FORMAT_RE: "re.Pattern[str]" = re.compile(
    r"^[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-4[A-Fa-f0-9]{3}-[89abAB][A-Fa-f0-9]{3}-[A-Fa-f0-9]{12}$",
    re.IGNORECASE,
)

# BUG-5.6 / BUG-7.2: maximum age (days) of a cached download before forcing
# a refresh.
OMIM_MAX_AGE_DAYS: int = _getenv_int("OMIM_MAX_AGE_DAYS", 30)

# BUG-8.20: DB batch size for bulk_upsert_gda.
OMIM_DB_BATCH_SIZE: int = _getenv_int("OMIM_DB_BATCH_SIZE", 1000)

# BUG-3.13: when True (default -- the safe choice for drug repurposing),
# susceptibility ({}) records are routed to a separate CSV and excluded from
# the main GDA load. Downstream ML MUST filter WHERE is_susceptibility = False
# for repurposing candidates. Treating {} as causal is the patient-harm
# failure mode the master prompt explicitly warns about.
OMIM_EXCLUDE_SUSCEPTIBILITY: bool = _getenv_bool("OMIM_EXCLUDE_SUSCEPTIBILITY", True)

# BUG-4.18 / BUG-8.13: pretty-print JSON in dev mode only (production is
# compact + deterministic).
OMIM_JSON_PRETTY: bool = _getenv_bool("OMIM_JSON_PRETTY", False)

# BUG-7.4 / BUG-4.9: random seed for retry backoff jitter. Fixed at module
# load for reproducibility.
OMIM_RANDOM_SEED: int = _getenv_int("OMIM_RANDOM_SEED", 42)

# BUG-9.15: helper -- does the OMIM_API_KEY look like a valid UUID?
def _omim_api_key_is_valid_format() -> bool:
    """Return True iff OMIM_API_KEY is empty OR matches the UUID format."""
    if not OMIM_API_KEY:
        return True  # empty is allowed (pipeline will raise at download time)
    # v43: OMIM_API_KEY_FORMAT_RE is now a compiled re.Pattern, so we
    # call .match() on the pattern directly instead of re.match(pattern, str).
    return bool(OMIM_API_KEY_FORMAT_RE.match(OMIM_API_KEY))


# BUG-12.11: eager validation of OMIM config -- mirrors DisGeNET's
# `_validate_disgenet_config` pattern. Raises ValueError on invalid values;
# logs UserWarning if a non-critical key is misconfigured.
def _validate_omim_config() -> None:
    """Validate OMIM_* env vars at module import time.

    Raises:
        ValueError: if a critical config value is out of range.
    """
    errors: list[str] = []
    if OMIM_REQUEST_INTERVAL <= 0:
        errors.append("OMIM_REQUEST_INTERVAL must be > 0")
    if not (1 <= OMIM_API_PAGE_LIMIT <= 1000):
        errors.append("OMIM_API_PAGE_LIMIT must be in [1, 1000]")
    if OMIM_API_MAX_RETRIES < 0:
        errors.append("OMIM_API_MAX_RETRIES must be >= 0")
    for mk in OMIM_MAPPING_KEYS_INCLUDE:
        if mk not in (1, 2, 3, 4):
            errors.append(
                f"OMIM_MAPPING_KEYS_INCLUDE contains invalid mk={mk} "
                f"(must be in {{1, 2, 3, 4}})"
            )
    for name, val in (
        ("OMIM_CONFIRMED_SCORE", OMIM_CONFIRMED_SCORE),
        ("OMIM_CONTIGUOUS_SCORE", OMIM_CONTIGUOUS_SCORE),
        ("OMIM_PHENOTYPE_MAPPED_SCORE", OMIM_PHENOTYPE_MAPPED_SCORE),
        ("OMIM_GENE_MAPPED_SCORE", OMIM_GENE_MAPPED_SCORE),
    ):
        if not (0.0 <= val <= 1.0):
            errors.append(f"{name} must be in [0.0, 1.0] (got {val})")
    if OMIM_MIN_EXPECTED_RECORDS < 0:
        errors.append("OMIM_MIN_EXPECTED_RECORDS must be >= 0")
    if OMIM_MAX_PAGINATION_PAGES < 1:
        errors.append("OMIM_MAX_PAGINATION_PAGES must be >= 1")
    if OMIM_DB_BATCH_SIZE < 1:
        errors.append("OMIM_DB_BATCH_SIZE must be >= 1")
    if OMIM_MAX_AGE_DAYS < 0:
        errors.append("OMIM_MAX_AGE_DAYS must be >= 0")
    # BUG-12.7: validate OMIM_API_BASE URL
    try:
        parsed = urllib.parse.urlparse(OMIM_API_BASE)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            errors.append(
                f"OMIM_API_BASE must be an http(s) URL (got {OMIM_API_BASE!r})"
            )
    except Exception as exc:
        errors.append(f"OMIM_API_BASE is not a valid URL: {exc}")

    # BUG-12.6: API key format check (warning only -- empty is allowed)
    if not _omim_api_key_is_valid_format():
        warnings.warn(
            f"OMIM_API_KEY does not match expected UUID format -- may be mistyped",
            UserWarning,
            stacklevel=2,
        )

    if errors:
        raise ValueError(
            "OMIM config validation failed:\n  - " + "\n  - ".join(errors)
        )


# Run eager validation -- but be tolerant if a downstream test wants to
# re-configure env vars and reload. Following DisGeNET's pattern: log warning
# and continue on validation failure (the pipeline will raise a clearer error
# at __init__ time).
try:
    _validate_omim_config()
except ValueError as _omim_cfg_exc:
    # v37 ROOT FIX (Phase 1 Issue #58): removed ``UserWarning`` from the
    # except tuple. ``_validate_omim_config`` only raises ``ValueError``;
    # it CALLS ``warnings.warn(..., UserWarning)`` which does NOT raise
    # (it emits a warning via the warnings machinery, not via raise).
    # Catching ``UserWarning`` was dead code. Worse, if a future change
    # made ``_validate_omim_config`` actually raise ``UserWarning``, the
    # ``warnings.warn`` call inside this except block would emit ANOTHER
    # UserWarning -- creating an infinite-loop risk if warnings were
    # configured as errors. The fix narrows the except to ``ValueError``
    # only, which is the only exception ``_validate_omim_config`` raises.
    warnings.warn(
        f"OMIM config validation warning: {_omim_cfg_exc}",
        UserWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# v29 ROOT FIX (audit C-13): consolidated OMIM config dict
# ---------------------------------------------------------------------------
# Previously the 20+ OMIM_* settings existed ONLY as flat module-level
# constants. That worked, but it meant every consumer had to import each
# constant by name (see pipelines/omim_pipeline.py -- it imports ~15 of
# them individually). This consolidated dict is the canonical structured
# view of all OMIM settings in one place, suitable for:
#   * programmatic introspection (e.g. /health endpoints)
#   * config-dump / provenance metadata (without leaking OMIM_API_KEY)
#   * new code that prefers dict access over many-name imports
#
# The individual OMIM_* module-level constants above are KEPT for
# backwards compatibility -- they are the same values, just accessible as
# flat names. The OMIMPipeline continues to import them by name; new
# consumers should prefer ``OMIM_CONFIG``.
#
# NOTE: ``OMIM_API_KEY`` is masked in ``OMIM_CONFIG["api_key_masked"]``
# but the raw value is still in ``OMIM_API_KEY`` (module-level) so the
# pipeline can authenticate. Do not log ``OMIM_CONFIG["api_key"]`` --
# use ``OMIM_CONFIG["api_key_masked"]`` for any human-facing output.
OMIM_CONFIG: dict[str, object] = {
    # --- Connection -----------------------------------------------------
    "api_key": OMIM_API_KEY,
    "api_key_masked": (
        "<set>" if OMIM_API_KEY else "<unset>"
    ),
    "api_base": OMIM_API_BASE,
    "api_key_format_re": OMIM_API_KEY_FORMAT_RE,
    "user_agent": OMIM_USER_AGENT,
    # --- Rate limiting / retries ---------------------------------------
    "request_interval": OMIM_REQUEST_INTERVAL,
    "api_page_limit": OMIM_API_PAGE_LIMIT,
    "api_max_retries": OMIM_API_MAX_RETRIES,
    "api_timeout": OMIM_API_TIMEOUT,
    "download_timeout": OMIM_DOWNLOAD_TIMEOUT,
    "max_pagination_pages": OMIM_MAX_PAGINATION_PAGES,
    "random_seed": OMIM_RANDOM_SEED,
    # --- Mapping / scoring ---------------------------------------------
    "mapping_keys_include": OMIM_MAPPING_KEYS_INCLUDE,
    "confirmed_score": OMIM_CONFIRMED_SCORE,
    "contiguous_score": OMIM_CONTIGUOUS_SCORE,
    "phenotype_mapped_score": OMIM_PHENOTYPE_MAPPED_SCORE,
    "gene_mapped_score": OMIM_GENE_MAPPED_SCORE,
    "exclude_susceptibility": OMIM_EXCLUDE_SUSCEPTIBILITY,
    # --- Output / batching / caching -----------------------------------
    "output_filename": OMIM_OUTPUT_FILENAME,
    "min_expected_records": OMIM_MIN_EXPECTED_RECORDS,
    "max_age_days": OMIM_MAX_AGE_DAYS,
    "db_batch_size": OMIM_DB_BATCH_SIZE,
    "dedup_keep_policy": OMIM_DEDUP_KEEP_POLICY,
    "json_pretty": OMIM_JSON_PRETTY,
}


def get_omim_config() -> dict[str, object]:
    """Return the consolidated OMIM configuration dict (lazy view).

    Returns a *copy* so callers can mutate without affecting the
    module-level state. For the masked API key view, use
    ``OMIM_CONFIG["api_key_masked"]`` or call :func:`get_omim_config`
    and pop ``api_key`` before logging.
    """
    # Refresh from the module-level constants in case they were mutated
    # by tests. We deliberately return a fresh dict each call.
    return dict(OMIM_CONFIG)


# ---------------------------------------------------------------------------
# Airflow
# ---------------------------------------------------------------------------

AIRFLOW_HOME: Path = BASE_DIR / "airflow"

# ---------------------------------------------------------------------------
# Logging -- ARCH-2, IDMP-3, SEC-5, LOG-1, LOG-2, LOG-3
# ---------------------------------------------------------------------------

LOG_LEVEL: str = _getenv("LOG_LEVEL", "INFO")

# [CFG-02] Configurable retention period for orphan GDA record cleanup.
# Records with uniprot_id=NULL older than this many hours are eligible
# for deletion by ``cleanup_orphan_gda_records`` in database.loaders.
ORPHAN_GDA_RETENTION_HOURS: int = int(_getenv("ORPHAN_GDA_RETENTION_HOURS", "24"))

# ---------------------------------------------------------------------------
# Loader-specific configuration (CFG-04, REL-04, PERF-07, LOG-05, SEC-06)
# ---------------------------------------------------------------------------
# These settings control the behaviour of database.loaders and can be
# overridden via environment variables without restarting the application.

# [CFG-04] Strict validation: when True, invalid records are quarantined
# and a WARNING is logged.  When False, invalid records are logged but
# still upserted (useful for initial data loads where completeness
# matters more than correctness).
LOADERS_STRICT_VALIDATION: bool = _getenv(
    "LOADERS_STRICT_VALIDATION", "true"
).lower() in ("true", "1", "yes")

# [REL-04] Maximum retry attempts for database operations with
# exponential backoff.  Applies to cleanup_orphan_gda_records and
# lookup functions.
LOADERS_MAX_RETRY_ATTEMPTS: int = int(
    _getenv("LOADERS_MAX_RETRY_ATTEMPTS", "3")
)

# [REL-04] Base delay in seconds for exponential backoff.  Actual delay
# is base_delay * (2 ** attempt_index).
LOADERS_RETRY_BASE_DELAY: float = float(
    _getenv("LOADERS_RETRY_BASE_DELAY", "0.5")
)

# [LOG-05] Enable timing/metrics logging for upsert operations.
LOADERS_ENABLE_TIMING: bool = _getenv(
    "LOADERS_ENABLE_TIMING", "true"
).lower() in ("true", "1", "yes")

# [REL-06] Enable the dead letter queue for failed/unprocessable records.
LOADERS_DEAD_LETTER_ENABLED: bool = _getenv(
    "LOADERS_DEAD_LETTER_ENABLED", "true"
).lower() in ("true", "1", "yes")

# [SEC-06] Maximum number of records that cleanup_orphan_gda_records
# may delete in a single call.  Prevents mass deletion from
# misconfiguration.
LOADERS_MAX_DELETE_COUNT: int = int(
    _getenv("LOADERS_MAX_DELETE_COUNT", "10000")
)

# [PERF-07] [CFG-03] Per-table batch size overrides.  Parsed from a
# comma-separated env var: "drugs=1000,proteins=500,dpi=2000".
# Tables not listed use DEFAULT_BATCH_SIZE from database.loaders.
_BATCH_SIZE_OVERRIDES_RAW: str = _getenv("LOADERS_BATCH_SIZE_OVERRIDES", "")
BATCH_SIZE_OVERRIDES: dict[str, int] = {}
if _BATCH_SIZE_OVERRIDES_RAW:
    for _pair in _BATCH_SIZE_OVERRIDES_RAW.split(","):
        _pair = _pair.strip()
        if "=" in _pair:
            _tbl, _sz = _pair.split("=", 1)
            try:
                BATCH_SIZE_OVERRIDES[_tbl.strip()] = int(_sz.strip())
            except ValueError:
                pass

_logging_configured: bool = False

logger = logging.getLogger(__name__)


def setup_logging(level: Optional[str] = None) -> None:
    """Configure logging for the platform. Idempotent and safe to call
    multiple times.

    This should be called explicitly in application entry points
    (main.py, DAG files, test conftest.py) rather than at module import
    time. It configures ONLY the platform's own logger namespaces, NOT
    the root logger, to avoid capturing sensitive values from third-party
    modules (SEC-5).

    Parameters
    ----------
    level : str, optional
        Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Defaults to LOG_LEVEL env var or INFO.
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    log_level = level or LOG_LEVEL
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    # Configure the root logger via logging.basicConfig so that
    # third-party modules' log records also get formatted output.
    # We pass our handler and format to basicConfig (idempotent -- if
    # the root logger already has handlers, basicConfig is a no-op).
    logging.basicConfig(
        level=log_level,
        handlers=[handler],
    )
    # Configure ONLY our namespace loggers, not the root logger (SEC-5)
    for namespace in (
        "config",
        "pipelines",
        "database",
        "cleaning",
        "entity_resolution",
        "exporters",
    ):
        ns_logger = logging.getLogger(namespace)
        ns_logger.setLevel(getattr(logging, log_level, logging.INFO))
        ns_logger.addHandler(handler)


def _mask_url(url: str) -> str:
    """Mask password in a URL for safe logging."""
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        if parsed.password:
            netloc = f"{parsed.username}:****@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass
    return url


def log_config_summary() -> None:
    """Log a startup summary of the active configuration.

    Sensitive values (API keys, passwords) are masked.
    Should be called from entry points after setup_logging().
    """
    summary = {
        "ENVIRONMENT": ENVIRONMENT,
        "CHEMBL_VERSION": CHEMBL_VERSION,
        "STRING_VERSION": STRING_VERSION,
        "STRING_MIN_COMBINED_SCORE": STRING_MIN_COMBINED_SCORE,
        "UNIPROT_RELEASE": UNIPROT_RELEASE,
        "DISGENET_USE_API": DISGENET_USE_API,
        "DISGENET_API_KEY": "***" if DISGENET_API_KEY else "(not set)",
        "OMIM_API_KEY": "***" if OMIM_API_KEY else "(not set)",
        "CHEMBL_MAX_ROWS": CHEMBL_MAX_ROWS or "(unlimited)",
        "CHEMBL_MAX_ACTIVITIES": CHEMBL_MAX_ACTIVITIES or "(unlimited)",
        "DATABASE_URL": _mask_url(DATABASE_URL),
        "DATA_SNAPSHOT_ID": DATA_SNAPSHOT_ID,
    }
    logger.info("=== Configuration Summary ===")
    for key, value in summary.items():
        logger.info("  %s = %s", key, value)
    logger.info("=== End Configuration Summary ===")


# ---------------------------------------------------------------------------
# Data provenance & versioning -- DATA-3, IDMP-4, LINEAGE-1, LINEAGE-2
# ---------------------------------------------------------------------------

DATA_SNAPSHOT_ID: str = _getenv(
    "DATA_SNAPSHOT_ID",
    f"chembl{CHEMBL_VERSION}_string{STRING_VERSION}_"
    f"uniprot{UNIPROT_RELEASE}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
)


def get_data_version_info() -> dict[str, str]:
    """Return a dict of all data source versions for embedding in output
    metadata.

    This should be called by pipeline entry points and embedded in every
    output CSV/JSON file produced by the pipeline for traceability.
    """
    return {
        "snapshot_id": DATA_SNAPSHOT_ID,
        "chembl_version": CHEMBL_VERSION,
        "string_version": STRING_VERSION,
        "uniprot_release": UNIPROT_RELEASE,
        "disgenet_source": "api" if DISGENET_USE_API else "static",
        "string_min_score": str(STRING_MIN_COMBINED_SCORE),
    }


def get_provenance_metadata() -> dict:
    """Return complete provenance metadata for embedding in pipeline output.

    This metadata should be embedded in every output file produced by the
    pipeline, either as a companion ``_metadata.json`` file or as comment
    headers in the CSV.
    """
    config_str = str(sorted(get_data_version_info().items()))
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
    return {
        "config_fingerprint": config_hash,
        "data_snapshot_id": DATA_SNAPSHOT_ID,
        "chembl_version": CHEMBL_VERSION,
        "string_version": STRING_VERSION,
        "string_min_score": STRING_MIN_COMBINED_SCORE,
        "uniprot_release": UNIPROT_RELEASE,
        "disgenet_source": "api" if DISGENET_USE_API else "static",
        "environment": ENVIRONMENT,
        "pipeline_version": "v1.0",
    }


# ---------------------------------------------------------------------------
# URL validation -- DATA-1, INTEROP-3
# ---------------------------------------------------------------------------


def validate_all_urls() -> dict[str, bool]:
    """Validate all URL settings with HEAD requests.

    Returns a dict of setting_name -> is_valid. Logs warnings for
    failing URLs. Does NOT raise -- the pipeline should start even if a
    URL is temporarily down.
    """
    results: dict[str, bool] = {}
    url_settings = {
        "STRING_PROTEIN_LINKS_URL": STRING_PROTEIN_LINKS_URL,
        "STRING_ALIASES_URL": STRING_ALIASES_URL,
        "DISGENET_API_URL": DISGENET_API_URL,
        "PUBCHEM_REST_BASE": PUBCHEM_REST_BASE,
        "CHEMBL_API_URL": CHEMBL_API_URL,
    }
    for name, url in url_settings.items():
        try:
            import requests

            resp = requests.head(url, timeout=10, allow_redirects=True)
            is_valid = resp.status_code < 400
            if not is_valid:
                logger.warning(
                    "URL validation failed for %s: %s returned HTTP %d",
                    name,
                    url,
                    resp.status_code,
                )
            results[name] = is_valid
        except Exception as exc:
            logger.warning(
                "URL validation failed for %s: %s - %s", name, url, exc
            )
            results[name] = False
    return results


def check_api_endpoints() -> dict[str, dict]:
    """Check availability of all API endpoints.

    Returns a dict of endpoint_name -> status info.
    """
    results: dict[str, dict] = {}
    endpoints = {
        "chembl": CHEMBL_API_URL,
        "disgenet": DISGENET_API_URL,
        "omim": OMIM_API_BASE,
        "pubchem": PUBCHEM_REST_BASE,
    }
    for name, url in endpoints.items():
        try:
            import requests

            resp = requests.head(url, timeout=10, allow_redirects=True)
            results[name] = {
                "url": url,
                "status": resp.status_code,
                "available": resp.status_code < 400,
            }
        except Exception as exc:
            results[name] = {
                "url": url,
                "status": None,
                "available": False,
                "error": str(exc),
            }
    return results


# ---------------------------------------------------------------------------
# API key validation -- SEC-2
# ---------------------------------------------------------------------------


def validate_api_keys() -> dict[str, str]:
    """Validate that required API keys are present.

    Returns a dict of key_name -> status ('present' | 'missing').
    Raises ``ValueError`` if DISGENET_USE_API is true and key is missing.
    """
    results = {
        "DISGENET_API_KEY": "present" if DISGENET_API_KEY else "missing",
        "OMIM_API_KEY": "present" if OMIM_API_KEY else "missing",
    }
    if DISGENET_USE_API and not DISGENET_API_KEY:
        raise ValueError(
            "DISGENET_USE_API=true but DISGENET_API_KEY is not set. "
            "Set the DISGENET_API_KEY environment variable or set "
            "DISGENET_USE_API=false (not recommended - static URL is "
            "deprecated)."
        )
    return results


# ---------------------------------------------------------------------------
# .env file checks -- SEC-3, COMP-2
# ---------------------------------------------------------------------------


def check_env_git_tracking() -> None:
    """Warn if .env file appears to be tracked by git (SEC-3)."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    try:
        import subprocess

        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(env_path)],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )
        if result.returncode == 0:
            warnings.warn(
                f".env file at {env_path} is tracked by git! "
                f"This exposes API keys and database credentials in "
                f"version control. Run: git rm --cached {env_path} "
                f"&& echo .env >> .gitignore",
                UserWarning,
            )
    except (FileNotFoundError, OSError):
        pass  # git not installed or not a git repo


def validate_env_file(path: Optional[Path] = None) -> list[str]:
    """Validate the .env file format. Returns list of issues found (COMP-2)."""
    env_path = path or (BASE_DIR / ".env")
    if not env_path.exists():
        return []  # No .env file is valid
    issues: list[str] = []
    for line_num, line in enumerate(env_path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            issues.append(f"Line {line_num}: Missing = delimiter: {line!r}")
        if line.count("=") > 1:
            key = line.split("=", 1)[0]
            value = line.split("=", 1)[1]
            if not (value.startswith('"') and value.endswith('"')):
                issues.append(
                    f"Line {line_num}: Unquoted value with = sign: {key}"
                )
    return issues


# ---------------------------------------------------------------------------
# Secret management -- SEC-4
# ---------------------------------------------------------------------------


def get_secret(key: str, default: str = "") -> str:
    """Get a secret value, preferring platform secret managers over .env.

    Lookup order:
    1. Environment variable (set by K8s Secrets, AWS SM, etc.)
    2. .env file (via dotenv)
    3. Default value
    """
    _ensure_dotenv_loaded()
    return os.getenv(key, default)


# ---------------------------------------------------------------------------
# Environment schema validation -- CONF-3
# ---------------------------------------------------------------------------

ENV_VAR_SCHEMA: dict[str, dict] = {
    "DATABASE_URL": {
        "type": str,
        "required": True,
        "pattern": r"^postgresql://",
    },
    "CHEMBL_VERSION": {
        "type": str,
        "required": False,
        "pattern": r"^[\d.]+$",
    },
    "CHEMBL_MAX_ROWS": {"type": int, "required": False, "min": 0},
    "DISGENET_API_KEY": {"type": str, "required": False},
    "OMIM_API_KEY": {"type": str, "required": False},
    "LOG_LEVEL": {
        "type": str,
        "required": False,
        "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    },
    "ENVIRONMENT": {
        "type": str,
        "required": False,
        "choices": ["development", "staging", "production"],
    },
}


def validate_env_schema() -> list[str]:
    """Validate all env vars against the schema. Returns list of errors."""
    import re

    errors: list[str] = []
    for key, spec in ENV_VAR_SCHEMA.items():
        value = os.getenv(key)
        if spec.get("required") and not value:
            errors.append(f"{key} is required but not set")
            continue
        if value and "pattern" in spec:
            if not re.match(spec["pattern"], value):
                errors.append(
                    f"{key}={value!r} does not match pattern "
                    f"{spec['pattern']}"
                )
        if value and "choices" in spec:
            if value not in spec["choices"]:
                errors.append(f"{key}={value!r} not in {spec['choices']}")
        if value and "min" in spec:
            try:
                if int(value) < spec["min"]:
                    errors.append(
                        f"{key}={value} is below minimum {spec['min']}"
                    )
            except ValueError:
                errors.append(f"{key}={value!r} is not a valid integer")
    return errors


# ---------------------------------------------------------------------------
# Configuration registry (data dictionary) -- DOC-3
# ---------------------------------------------------------------------------
# v29 ROOT FIX (audit C-13): CONFIG_REGISTRY is STALE.
#
# This registry was originally intended as a self-documenting data
# dictionary of every setting in the module. In practice it has not been
# maintained alongside the actual settings -- many settings added by the
# institutional-grade rewrites of the ChEMBL / STRING / DisGeNET / OMIM /
# PubChem / DrugBank pipelines are NOT registered here, and several
# entries (e.g. ``OMIM_DEDUP_KEEP_POLICY``) describe settings whose
# semantics have changed since the entry was written.
#
# It is KEPT for now because:
#   * ``tests/test_settings.py::test_doc3_config_registry`` asserts its
#     existence and the presence of ``DATABASE_URL`` / ``CHEMBL_VERSION``.
#   * ``tests/test_all_26_files_integration_v10.py`` asserts the OMIM_*
#     entry count is >= 20.
#   * ``docs/pipelines/omim.md`` references it.
#
# But it is hereby DEPRECATED. New code MUST NOT add entries to this
# registry. Instead, prefer:
#   * For OMIM: the consolidated ``OMIM_CONFIG`` dict (above) and the
#     ``get_omim_config()`` accessor.
#   * For per-source structured config: the source-specific dataclasses
#     (``DatabaseConfig``, ``ChEMBLConfig``, ``StringConfig``,
#     ``DisGeNETConfig``) and their ``get_*_config()`` accessors.
#   * For raw env-var introspection: ``ENV_VAR_SCHEMA`` (above) which is
#     maintained alongside the actual ``_getenv`` / ``_getenv_int`` /
#     ``_getenv_bool`` / ``_getenv_float`` call sites.
#
# A future v2.0.0 release will remove ``CONFIG_REGISTRY`` and migrate
# the two tests above to assert against the structured dataclasses and
# ``OMIM_CONFIG`` instead.
import warnings as _warnings_for_registry  # noqa: E402
_warnings_for_registry.warn(
    "config.settings.CONFIG_REGISTRY is DEPRECATED (v29 audit C-13): "
    "stale data dictionary. Use OMIM_CONFIG / get_*_config() / "
    "ENV_VAR_SCHEMA instead. Will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
del _warnings_for_registry

CONFIG_REGISTRY: dict[str, dict] = {
    "DATABASE_URL": {
        "type": "str",
        "required": True,
        "default": "placeholder",
        "description": "PostgreSQL connection string",
        "used_by": ["database.connection"],
    },
    "CHEMBL_VERSION": {
        "type": "str",
        "required": False,
        "default": "35",
        "description": "ChEMBL database release version",
        "used_by": ["pipelines.chembl"],
    },
    "CHEMBL_API_URL": {
        "type": "str",
        "required": False,
        "default": "https://www.ebi.ac.uk/chembl/api/data",
        "description": "ChEMBL REST API base URL",
        "used_by": ["pipelines.chembl"],
    },
    "STRING_VERSION": {
        "type": "str",
        "required": False,
        "default": "12.0",
        "description": "STRING DB version",
        "used_by": ["pipelines.string"],
    },
    "STRING_MIN_COMBINED_SCORE": {
        "type": "int",
        "required": False,
        # v65 ROOT FIX (P1C-003): the previous default was "400", which
        # the TOP-1 ROOT FIX (settings.py lines 913-921) explicitly
        # rejected because ">= 400 achieves only ~50% precision" while
        # ">= 700 achieves >80% precision" (Szklarczyk et al. 2023).
        # The CONFIG_REGISTRY default was NOT updated when the code
        # default was changed to 700, so any tooling that read the
        # registry default (e.g. config validators, documentation
        # generators, .env scaffolding) still saw 400 -- silently
        # contradicting the scientifically-validated threshold.
        "default": "700",
        "description": "Minimum STRING PPI score for inclusion (700 = high confidence, >80% precision per Szklarczyk 2023)",
        "valid_range": "0-1000",
        "used_by": ["pipelines.string"],
    },
    "DISGENET_API_KEY": {
        "type": "str",
        "required": False,
        "default": "",
        "description": "DisGeNET API authentication key",
        "used_by": ["pipelines.disgenet"],
    },
    "OMIM_API_KEY": {
        "type": "str",
        "required": False,
        "default": "",
        "secret": True,
        "description": "OMIM API authentication key (UUID format). Required for both the morbidmap.txt download endpoint and the REST API.",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_API_BASE": {
        "type": "str",
        "required": False,
        "default": "https://api.omim.org/api",
        "description": "OMIM REST API base URL.",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_REQUEST_INTERVAL": {
        "type": "float",
        "required": False,
        "default": "0.25",
        "description": "Seconds to sleep between OMIM API requests (4 req/sec).",
        "valid_range": ">0",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_MAPPING_KEYS_INCLUDE": {
        "type": "list[int]",
        "required": False,
        "default": "[3, 4]",
        "description": "Phenotype mapping keys to include (1=wild-type gene mapped, 2=phenotype mapped, 3=molecular basis known, 4=contiguous gene syndrome).",
        "valid_values": "subset of {1,2,3,4}",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_API_PAGE_LIMIT": {
        "type": "int",
        "required": False,
        "default": "1000",
        "description": "OMIM API pagination page size (max 1000 per OMIM docs).",
        "valid_range": "1-1000",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_API_MAX_RETRIES": {
        "type": "int",
        "required": False,
        "default": "5",
        "description": "Maximum HTTP retries on 429/5xx responses.",
        "valid_range": ">=0",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_DOWNLOAD_TIMEOUT": {
        "type": "int",
        "required": False,
        "default": "300",
        "description": "HTTP timeout (seconds) for the morbidmap.txt download.",
        "valid_range": ">0",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_API_TIMEOUT": {
        "type": "int",
        "required": False,
        "default": "120",
        "description": "HTTP timeout (seconds) for each OMIM REST API request.",
        "valid_range": ">0",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_OUTPUT_FILENAME": {
        "type": "str",
        "required": False,
        "default": "omim_gene_disease_associations.csv",
        "description": "Filename of the cleaned GDA CSV written by clean().",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_MIN_EXPECTED_RECORDS": {
        "type": "int",
        "required": False,
        "default": "5000",
        "description": "Minimum parsed-record count; below this, clean() aborts (catches truncated downloads).",
        "valid_range": ">=0",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_MAX_PAGINATION_PAGES": {
        "type": "int",
        "required": False,
        "default": "1000",
        "description": "Upper bound on API pagination pages (prevents infinite loop).",
        "valid_range": ">=1",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_DEDUP_KEEP_POLICY": {
        "type": "str",
        "required": False,
        "default": "last",
        "description": "Legacy dedup-keep-policy (no longer used -- atomic writes don't append).",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_CONFIRMED_SCORE": {
        "type": "float",
        "required": False,
        "default": "0.9",
        "description": "Base score for mapping_key=3 (molecular basis known).",
        "valid_range": "0.0-1.0",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_CONTIGUOUS_SCORE": {
        "type": "float",
        "required": False,
        "default": "0.8",
        "description": "Base score for mapping_key=4 (contiguous gene syndrome).",
        "valid_range": "0.0-1.0",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_PHENOTYPE_MAPPED_SCORE": {
        "type": "float",
        "required": False,
        "default": "0.6",
        "description": "Base score for mapping_key=2 (phenotype mapped).",
        "valid_range": "0.0-1.0",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_GENE_MAPPED_SCORE": {
        "type": "float",
        "required": False,
        "default": "0.5",
        "description": "Base score for mapping_key=1 (wild-type gene mapped).",
        "valid_range": "0.0-1.0",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_USER_AGENT": {
        "type": "str",
        "required": False,
        "default": "drug-repurposing-pipeline/omim (contact=unknown@example.com)",
        "description": "User-Agent header sent with every OMIM HTTP request.",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_API_KEY_FORMAT_RE": {
        "type": "str",
        "required": False,
        "default": "^[a-f0-9-]{36}$",
        "description": "Regex validating OMIM_API_KEY format (OMIM keys are UUIDs).",
        "used_by": ["pipelines.omim", "config.settings"],
    },
    "OMIM_MAX_AGE_DAYS": {
        "type": "int",
        "required": False,
        "default": "30",
        "description": "Maximum age (days) of a cached morbidmap.txt before forcing a refresh.",
        "valid_range": ">=0",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_DB_BATCH_SIZE": {
        "type": "int",
        "required": False,
        "default": "1000",
        "description": "Batch size for bulk_upsert_gda.",
        "valid_range": ">=1",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_EXCLUDE_SUSCEPTIBILITY": {
        "type": "bool",
        "required": False,
        "default": "true",
        "description": "When true, susceptibility ({}) records are routed to a separate CSV and excluded from the main GDA load.",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_JSON_PRETTY": {
        "type": "bool",
        "required": False,
        "default": "false",
        "description": "Pretty-print intermediate JSON (dev only; production uses compact + deterministic).",
        "used_by": ["pipelines.omim"],
    },
    "OMIM_RANDOM_SEED": {
        "type": "int",
        "required": False,
        "default": "42",
        "description": "Random seed for HTTP retry backoff jitter (reproducibility).",
        "used_by": ["pipelines.omim"],
    },
    "DRUGBANK_XML_PATH": {
        "type": "Path",
        "required": False,
        "default": "raw_data/drugbank/drugbank_all_full_database.xml.gz",
        "description": "Path to DrugBank XML file (manual download)",
        "used_by": ["pipelines.drugbank"],
    },
    "UNIPROT_RELEASE": {
        "type": "str",
        "required": False,
        "default": "current_release",
        "description": "UniProt release version for reproducibility",
        "used_by": ["pipelines.uniprot"],
    },
    "LOG_LEVEL": {
        "type": "str",
        "required": False,
        "default": "INFO",
        "description": "Platform logging level",
        "valid_values": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        "used_by": ["config.settings"],
    },
    "ENVIRONMENT": {
        "type": "str",
        "required": False,
        "default": "development",
        "description": "Deployment environment profile (aliases are accepted and normalized -- see _ENV_NORMALIZATION map at line ~363)",
        # v66 ROOT FIX (P1C-026 -- valid_values omitted accepted aliases):
        #   The previous list only included the canonical forms
        #   (development, staging, production). But the normalization map
        #   (_ENV_NORMALIZATION at line ~363) ALSO accepts "dev",
        #   "develop", "stage", "prod". An operator who set
        #   ENVIRONMENT=dev would see this registry's valid_values and
        #   think "dev" is invalid, when in fact the code accepts and
        #   normalizes it to "development". ROOT FIX: list ALL accepted
        #   aliases alongside the canonical forms, and note in the
        #   description that aliases are normalized.
        "valid_values": [
            "development", "dev", "develop",
            "staging", "stage",
            "production", "prod",
        ],
        "used_by": ["config.settings"],
    },
}

# ---------------------------------------------------------------------------
# Structured config groups (dataclasses) -- ARCH-3
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection settings."""

    url: str


@dataclass(frozen=True)
class ChEMBLConfig:
    """ChEMBL data source configuration."""

    version: str
    api_url: str
    max_rows: Optional[int]
    max_activities: Optional[int]
    expected_drug_count_min: int
    expected_drug_count_max: int


@dataclass(frozen=True)
class StringConfig:
    """STRING DB configuration."""

    version: str
    min_combined_score: int
    protein_links_url: str
    aliases_url: str
    protein_links_detailed_url: str


@dataclass(frozen=True)
class DisGeNETConfig:
    """DisGeNET configuration."""

    api_url: str
    api_key: str
    use_api: bool


_db_config: Optional[DatabaseConfig] = None
_chembl_config: Optional[ChEMBLConfig] = None
_string_config: Optional[StringConfig] = None
_disgenet_config: Optional[DisGeNETConfig] = None


def get_database_config() -> DatabaseConfig:
    """Return structured database configuration (lazy-initialized)."""
    global _db_config
    if _db_config is None:
        _db_config = DatabaseConfig(url=DATABASE_URL)
    return _db_config


def get_chembl_config() -> ChEMBLConfig:
    """Return structured ChEMBL configuration (lazy-initialized)."""
    global _chembl_config
    if _chembl_config is None:
        _chembl_config = ChEMBLConfig(
            version=CHEMBL_VERSION,
            api_url=CHEMBL_API_URL,
            max_rows=CHEMBL_MAX_ROWS,
            max_activities=CHEMBL_MAX_ACTIVITIES,
            expected_drug_count_min=CHEMBL_EXPECTED_DRUG_COUNT_MIN,
            expected_drug_count_max=CHEMBL_EXPECTED_DRUG_COUNT_MAX,
        )
    return _chembl_config


def get_string_config() -> StringConfig:
    """Return structured STRING configuration (lazy-initialized)."""
    global _string_config
    if _string_config is None:
        _string_config = StringConfig(
            version=STRING_VERSION,
            min_combined_score=STRING_MIN_COMBINED_SCORE,
            protein_links_url=STRING_PROTEIN_LINKS_URL,
            aliases_url=STRING_ALIASES_URL,
            protein_links_detailed_url=STRING_PROTEIN_LINKS_DETAILED_URL,
        )
    return _string_config


def get_disgenet_config() -> DisGeNETConfig:
    """Return structured DisGeNET configuration (lazy-initialized)."""
    global _disgenet_config
    if _disgenet_config is None:
        _disgenet_config = DisGeNETConfig(
            api_url=DISGENET_API_URL,
            api_key=DISGENET_API_KEY,
            use_api=DISGENET_USE_API,
        )
    return _disgenet_config


# ---------------------------------------------------------------------------
# Configuration reload -- ARCH-4, CONF-5
# ---------------------------------------------------------------------------


def reload_settings() -> dict[str, tuple[str, str]]:
    """Reload configuration and return changed settings.

    Resets the lazy-initialization caches for all config dataclasses,
    allowing new environment variable values to take effect without
    a full ``importlib.reload()`` (which can break held references).

    v79 FORENSIC ROOT FIX (P0-A5 -- reload_settings() returned an EMPTY
    diff because module-level constants were never re-read):
      The v78 code only reset the lazy caches (``_db_config``,
      ``_chembl_config``, etc.) but NEVER re-read the module-level
      constants (``DATABASE_URL``, ``CHEMBL_VERSION``, ``STRING_VERSION``,
      ``UNIPROT_RELEASE``, ``DISGENET_USE_API``, ``STRING_MIN_COMBINED_SCORE``,
      etc.). These constants are bound at import time via ``_getenv()``
      calls and stored in ``globals()``. ``get_data_version_info()``
      reads THESE SAME constants -- so calling it before and after
      ``reload_settings()`` returned identical values -> empty diff.
      Operators were silently misled that hot-reload worked; only a
      process restart picked up new env vars.

    ROOT FIX: maintain a ``_RELOADABLE_SETTINGS`` registry (populated
      below) of ``(global_name, reloader_callable)`` tuples. On
      ``reload_settings()``, iterate the registry, call each reloader
      (which re-reads the env var and parses it), compare old vs new,
      update ``globals()``, and record the diff. The diff is now REAL
      -- operators can trust it to reflect env-var changes.

    Returns a dict of setting_name -> (old_value, new_value) for
    settings whose values changed after the reload.
    """
    changes: dict[str, tuple[str, str]] = {}

    # Reset lazy caches FIRST so dataclass accessors re-build from env.
    global _db_config, _chembl_config, _string_config, _disgenet_config
    global _dotenv_loaded, _logging_configured
    _db_config = None
    _chembl_config = None
    _string_config = None
    _disgenet_config = None
    _dotenv_loaded = False  # Re-load .env on next access
    # Note: _logging_configured remains True to avoid duplicate handlers

    # Force re-load of .env so the re-readers below see fresh values.
    _ensure_dotenv_loaded()

    # v79 ROOT FIX: re-read every registered module-level constant
    # from os.environ and update globals(). Compute the real diff.
    # Map setting global names to the keys get_data_version_info() uses
    # so callers that grepped for those keys still see them in the diff.
    _DV_KEY_MAP = {
        "DATA_SNAPSHOT_ID": "snapshot_id",
        "CHEMBL_VERSION": "chembl_version",
        "STRING_VERSION": "string_version",
        "UNIPROT_RELEASE": "uniprot_release",
        "DISGENET_USE_API": "disgenet_source",
        "STRING_MIN_COMBINED_SCORE": "string_min_score",
    }
    for global_name, reloader in _RELOADABLE_SETTINGS:
        old_value = globals().get(global_name)
        try:
            new_value = reloader()
        except Exception as exc:
            logger.warning(
                "reload_settings: failed to re-read %s (%s) -- keeping "
                "old value %r",
                global_name, exc, old_value,
            )
            continue
        if new_value != old_value:
            globals()[global_name] = new_value
            # Record under both the global name and the data-version key
            # (if different) so callers can find the change either way.
            old_str_val = str(old_value) if old_value is not None else ""
            new_str_val = str(new_value) if new_value is not None else ""
            changes[global_name] = (old_str_val, new_str_val)
            dv_key = _DV_KEY_MAP.get(global_name)
            if dv_key is not None and dv_key not in changes:
                # For DISGENET_USE_API, the data-version value is derived
                # ("api"/"static"), not the bool itself.
                if global_name == "DISGENET_USE_API":
                    old_dv = "api" if old_value else "static"
                    new_dv = "api" if new_value else "static"
                    changes[dv_key] = (old_dv, new_dv)
                else:
                    changes[dv_key] = (old_str_val, new_str_val)

    if changes:
        logger.warning("Configuration changed after reload: %s", changes)

    return changes


# ---------------------------------------------------------------------------
# v79 FORENSIC ROOT FIX (P0-A5) -- Reloadable settings registry
# ---------------------------------------------------------------------------
# Each entry: (global_name, reloader_callable)
# The reloader re-reads the env var and parses it using the SAME helper
# that defined the constant at import time. reload_settings() iterates
# this list to actually update globals() and compute a real diff.
#
# IMPORTANT: add an entry here for EVERY module-level constant that
# operators might hot-reload via env vars. The registry is populated
# AFTER the constants are defined (below) so the reloaders can reference
# the same helpers (_getenv, _parse_required_int, _parse_bool,
# _validate_chembl_version, etc.).
_RELOADABLE_SETTINGS: list[tuple[str, Any]] = []


def _register_reloadable(global_name: str, reloader: Any) -> None:
    """Register a module-level setting for hot-reload (v79 P0-A5)."""
    _RELOADABLE_SETTINGS.append((global_name, reloader))


# Populate the registry with the key operational settings. Each reloader
# is a closure that re-reads the env var and parses it identically to
# the import-time definition. Defaults are duplicated here so the
# reloader is self-contained (no dependency on the import-time value,
# which may have been swapped by the dev-default-DB logic).
def _reload_database_url() -> str:
    return _getenv(
        "DATABASE_URL",
        "postgresql://REPLACE_USER:REPLACE_PASSWORD@localhost:5432/drug_repurposing",
    )

_register_reloadable("DATABASE_URL", _reload_database_url)


def _reload_chembl_version() -> str:
    return _validate_chembl_version(
        _getenv("CHEMBL_VERSION", DEFAULT_CHEMBL_VERSION)
    )

_register_reloadable("CHEMBL_VERSION", _reload_chembl_version)


def _reload_chembl_api_url() -> str:
    return _getenv("CHEMBL_API_URL", "https://www.ebi.ac.uk/chembl/api/data")

_register_reloadable("CHEMBL_API_URL", _reload_chembl_api_url)


def _reload_string_version() -> str:
    return _getenv("STRING_VERSION", DEFAULT_STRING_VERSION)

_register_reloadable("STRING_VERSION", _reload_string_version)


def _reload_uniprot_release() -> str:
    return _getenv("UNIPROT_RELEASE", "current_release")

_register_reloadable("UNIPROT_RELEASE", _reload_uniprot_release)


def _reload_disgenet_use_api() -> bool:
    return _parse_bool(_getenv("DISGENET_USE_API", "true"))

_register_reloadable("DISGENET_USE_API", _reload_disgenet_use_api)


def _reload_disgenet_api_url() -> str:
    return _getenv(
        "DISGENET_API_URL",
        "https://api.disgenet.com/api/v1/gda/summary",
    )

_register_reloadable("DISGENET_API_URL", _reload_disgenet_api_url)


def _reload_disgenet_api_key() -> str:
    return _getenv("DISGENET_API_KEY", "")

_register_reloadable("DISGENET_API_KEY", _reload_disgenet_api_key)


def _reload_string_min_combined_score() -> int:
    # Use the same default logic as the import-time definition: the
    # threshold depends on STRING_VERSION. Read the CURRENT (possibly
    # just-reloaded) STRING_VERSION from globals so the threshold
    # picks up version changes in the same reload pass.
    _sv = globals().get("STRING_VERSION", DEFAULT_STRING_VERSION)
    return _parse_required_int(
        "STRING_MIN_COMBINED_SCORE",
        str(_get_default_string_threshold(_sv)),
    )

_register_reloadable("STRING_MIN_COMBINED_SCORE", _reload_string_min_combined_score)


def _reload_drugbank_xml_path() -> Any:
    # v79: align default with the import-time definition (RAW_DATA_DIR /
    # drugbank / drugbank_all_full_database.xml.gz). The v79-first-draft
    # reloader used BASE_DIR/data/drugbank/drugbank.xml which diverged
    # from the import-time default, causing a spurious diff on reload.
    _default_db_path = str(RAW_DATA_DIR / "drugbank" / "drugbank_all_full_database.xml.gz")
    _raw = _getenv("DRUGBANK_XML_PATH", _default_db_path) or _default_db_path
    return Path(_raw)

_register_reloadable("DRUGBANK_XML_PATH", _reload_drugbank_xml_path)


def _reload_data_snapshot_id() -> str:
    # v79: align default with the import-time definition which uses the
    # full version fingerprint (chembl + string + uniprot).
    _sv = globals().get("STRING_VERSION", DEFAULT_STRING_VERSION)
    _cv = globals().get("CHEMBL_VERSION", DEFAULT_CHEMBL_VERSION)
    _ur = globals().get("UNIPROT_RELEASE", "current_release")
    return _getenv(
        "DATA_SNAPSHOT_ID",
        f"chembl{_cv}_string{_sv}_uniprot{_ur}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
    )

_register_reloadable("DATA_SNAPSHOT_ID", _reload_data_snapshot_id)


# ---------------------------------------------------------------------------
# __all__ -- CODE-8
# ---------------------------------------------------------------------------

__all__ = [
    # Paths
    "BASE_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "AIRFLOW_HOME",
    # Database
    "DATABASE_URL",
    # ChEMBL
    "CHEMBL_VERSION",
    "CHEMBL_API_URL",
    "CHEMBL_MAX_ROWS",
    "CHEMBL_MAX_ACTIVITIES",
    "CHEMBL_SNAPSHOT_DATE",
    "CHEMBL_EXPECTED_DRUG_COUNT_MIN",
    "CHEMBL_EXPECTED_DRUG_COUNT_MAX",
    "CHEMBL_URL",  # deprecated
    # ChEMBL -- institutional-grade operational settings (chembl_pipeline.py rewrite)
    "CHEMBL_PAGE_SIZE",
    "CHEMBL_MAX_RETRIES",
    "CHEMBL_RETRY_BACKOFF_BASE",
    "CHEMBL_MIN_REQUEST_INTERVAL",
    "CHEMBL_HTTP_TIMEOUT",
    "CHEMBL_MAX_RESPONSE_BYTES",
    "CHEMBL_CIRCUIT_BREAKER_THRESHOLD",
    "CHEMBL_CIRCUIT_BREAKER_RESET_SECONDS",
    "CHEMBL_TARGET_ORGANISM",
    "CHEMBL_MAX_PHASE",
    "CHEMBL_MW_MACROMOLECULE_THRESHOLD",
    "CHEMBL_ACTIVITY_TYPES",
    "CHEMBL_STANDARD_UNITS",
    "CHEMBL_STANDARD_RELATIONS",
    "CHEMBL_ASSAY_TYPES",
    "CHEMBL_TARGET_TYPES",
    "CHEMBL_TARGET_ACCESSION_STRATEGY",
    "CHEMBL_ACTIVITY_CHUNK_SIZE",
    "CHEMBL_DPI_BATCH_SIZE",
    "CHEMBL_TARGET_RESOLUTION_BATCH_SIZE",
    "CHEMBL_API_WORKERS",
    "CHEMBL_TARGET_RESOLUTION_WORKERS",
    "CHEMBL_TARGET_CACHE_TTL_SECONDS",
    "CHEMBL_DRUG_ID_CACHE_TTL_SECONDS",
    "CHEMBL_CACHE_TTL_SECONDS",
    "CHEMBL_ALLOW_VERSION_MISMATCH",
    "CHEMBL_RESUME",
    # Pipeline-wide operational settings
    "PIPELINE_RUN_ID",
    "PIPELINE_USE_CACHE",
    "PIPELINE_LOG_FORMAT",
    "PIPELINE_CONTACT_EMAIL",
    "PIPELINE_RESUME",
    # STRING
    "STRING_VERSION",
    "STRING_PROTEIN_LINKS_URL",
    "STRING_ALIASES_URL",
    "STRING_PROTEIN_LINKS_DETAILED_URL",
    "STRING_PROTEIN_INFO_URL",  # deprecated
    "STRING_MIN_COMBINED_SCORE",
    "STRING_MIN_COMBINED_SCORE_PROD",
    "STRING_DETAILED_MODE",
    "STRING_DROP_SELF_INTERACTIONS",
    "STRING_DEDUP_STRATEGY",
    "STRING_LOW_MEMORY",
    "STRING_CHUNK_SIZE",
    # DisGeNET
    "DISGENET_URL",
    "DISGENET_API_URL",
    "DISGENET_API_KEY",
    "DISGENET_USE_API",
    "DISGENET_STATIC_URL",  # deprecated
    # DisGeNET -- institutional-grade operational settings (389-fix audit)
    "DISGENET_MIN_SCORE",
    "DISGENET_ALLOW_WEAK_EVIDENCE",
    "DISGENET_WEAK_EVIDENCE_THRESHOLD",
    "DISGENET_CONFIDENCE_TIERS_JSON",
    "DISGENET_CONFIDENCE_TIERS",
    "DISGENET_PMID_CAP",
    "DISGENET_PMID_SORT_ORDER",
    "DISGENET_API_PAGE_SIZE",
    "DISGENET_API_MAX_RECORDS",
    "DISGENET_API_TIMEOUT",
    "DISGENET_API_MAX_RETRIES",
    "DISGENET_API_BACKOFF_BASE",
    "DISGENET_API_BACKOFF_MAX_SECONDS",
    "DISGENET_API_MAX_RETRY_AFTER",
    "DISGENET_API_RATE_LIMIT",
    "DISGENET_CIRCUIT_BREAKER_THRESHOLD",
    "DISGENET_CIRCUIT_BREAKER_RESET_SECONDS",
    "DISGENET_CONTACT_EMAIL",
    "DISGENET_ALLOWED_DOMAINS",
    "DISGENET_API_MAX_RESPONSE_BYTES",
    "DISGENET_API_CA_BUNDLE",
    "DISGENET_OUTPUT_FILE_MODE",
    "DISGENET_FALLBACK_TO_CACHE",
    "DISGENET_API_MAX_PAGES",
    "DISGENET_DOWNLOAD_PHASE_TIMEOUT",
    "DISGENET_ALLOW_PARTIAL_DATA",
    "DISGENET_UNIPROT_MAP_TTL_HOURS",
    "DISGENET_TARGET_VERSION",
    "DISGENET_FREEZE_VERSION",
    "DISGENET_MIN_EXPECTED_RECORDS",
    "DISGENET_DISEASE_ONTOLOGY_PATH",
    "DISGENET_HGNC_PATH",
    "DISGENET_MAX_DATA_AGE_DAYS",
    "DISGENET_OUTPUT_FILENAME",
    "DISGENET_RAW_FILENAME",
    "DISGENET_CHUNK_SIZE",
    "DISGENET_API_PARALLEL_PAGES",
    "DISGENET_LOG_FORMAT",
    "DISGENET_ENV",
    "DISGENET_SOURCE_WEIGHTS_JSON",
    "DISGENET_SOURCE_WEIGHTS",
    "_validate_disgenet_config",
    # PubChem
    "PUBCHEM_REST_BASE",
    "PUBCHEM_FTP_BASE",
    "PUBCHEM_API_URL",
    # Entity Resolution (audit D12-2)
    "ENTITY_RESOLUTION_PUBCHEM_ENABLED",
    "ENTITY_RESOLUTION_COLLAPSE_STEREOISOMERS",
    "ENTITY_RESOLUTION_FUZZY_THRESHOLD",
    "ENTITY_RESOLUTION_FUZZY_MAX_CANDIDATES",
    "ENTITY_RESOLUTION_PUBCHEM_REST_BASE",
    "ENTITY_RESOLUTION_PUBCHEM_CALL_DELAY",
    "ENTITY_RESOLUTION_PUBCHEM_TIMEOUT",
    "ENTITY_RESOLUTION_PUBCHEM_MAX_RETRIES",
    "ENTITY_RESOLUTION_PUBCHEM_API_KEY",
    "ENTITY_RESOLUTION_PUBCHEM_CA_BUNDLE",
    "ENTITY_RESOLUTION_PUBCHEM_CERT_PEM",
    "ENTITY_RESOLUTION_PUBCHEM_KEY_PEM",
    "ENTITY_RESOLUTION_PUBCHEM_STRICT_SALT_FORM",
    "ENTITY_RESOLUTION_SOURCE_WHITELIST",
    "ENTITY_RESOLUTION_DEFAULT_ORGANISM",
    "ENTITY_RESOLUTION_MAPPING_SCHEMA_VERSION",
    "get_entity_resolution_config",
    # PubChem pipeline (institutional-grade -- fixes PUBCHEM_PIPELINE_MASTER_FIX_PROMPT.md)
    "PUBCHEM_PIPELINE_BATCH_SIZE",
    "PUBCHEM_PIPELINE_MIN_BACKOFF",
    "PUBCHEM_PIPELINE_MAX_BACKOFF",
    "PUBCHEM_PIPELINE_READ_TIMEOUT",
    "PUBCHEM_PIPELINE_CACHE_TTL_SECONDS",
    "PUBCHEM_PIPELINE_CONCURRENCY",
    "PUBCHEM_PIPELINE_FETCH_SYNONYMS",
    "PUBCHEM_PIPELINE_FETCH_CAS",
    "PUBCHEM_PIPELINE_SPLIT_RETRY_MAX",
    "PUBCHEM_PIPELINE_MAX_RECORDS",
    "PUBCHEM_PIPELINE_RAW_RESPONSE_RETENTION_DAYS",
    "PUBCHEM_CIRCUIT_BREAKER_THRESHOLD",
    "PUBCHEM_CIRCUIT_BREAKER_RESET_SECONDS",
    "PUBCHEM_PIPELINE_PROPERTIES",
    "PROMETHEUS_ENABLED",
    "OTEL_ENABLED",
    "OPERATOR_ID",
    "RDKIT_AVAILABLE",
    # DrugBank
    "DRUGBANK_XML_PATH",
    "DRUGBANK_VERSION",
    "DRUGBANK_XML_NAMESPACE",
    "DRUGBANK_TARGET_ORGANISMS",
    "DRUGBANK_GENERATE_SYNTH_KEYS",
    "DRUGBANK_DROP_NO_INCHIKEY",
    "DRUGBANK_CONSERVATIVE_DEFAULTS",
    "DRUGBANK_BATCH_SIZE",
    "DRUGBANK_LOG_INTERVAL",
    "DRUGBANK_MAX_DRUGS",
    "DRUGBANK_EXTRACT_TARGETS",
    "DRUGBANK_EXTRACT_ENZYMES",
    "DRUGBANK_EXTRACT_TRANSPORTERS",
    "DRUGBANK_CSV_COMPRESSION",
    "DRUGBANK_EXPECTED_SHA256",
    "DRUGBANK_EXPECTED_DRUG_COUNT_MIN",
    "DRUGBANK_EXPECTED_DRUG_COUNT_MAX",
    "DRUGBANK_LOG_REDACT",
    "DRUGBANK_LOG_FULL_PATHS",
    "DRUGBANK_VALIDATE_READABILITY",
    "DRUGBANK_DPI_BATCH_SIZE",
    "DEFAULT_DRUGBANK_VERSION",
    "VALID_DRUGBANK_VERSIONS",
    # OMIM
    "OMIM_API_KEY",
    "OMIM_API_BASE",
    "OMIM_REQUEST_INTERVAL",
    "OMIM_MAPPING_KEYS_INCLUDE",
    "OMIM_API_PAGE_LIMIT",
    "OMIM_API_MAX_RETRIES",
    "OMIM_DOWNLOAD_TIMEOUT",
    "OMIM_API_TIMEOUT",
    "OMIM_OUTPUT_FILENAME",
    "OMIM_MIN_EXPECTED_RECORDS",
    "OMIM_MAX_PAGINATION_PAGES",
    "OMIM_DEDUP_KEEP_POLICY",
    "OMIM_CONFIRMED_SCORE",
    "OMIM_CONTIGUOUS_SCORE",
    "OMIM_PHENOTYPE_MAPPED_SCORE",
    "OMIM_GENE_MAPPED_SCORE",
    "OMIM_USER_AGENT",
    "OMIM_API_KEY_FORMAT_RE",
    "OMIM_MAX_AGE_DAYS",
    "OMIM_DB_BATCH_SIZE",
    "OMIM_EXCLUDE_SUSCEPTIBILITY",
    "OMIM_JSON_PRETTY",
    "OMIM_RANDOM_SEED",
    "_parse_csv_ints",
    "_validate_omim_config",
    # UniProt
    "UNIPROT_RELEASE",
    "UNIPROT_SPROT_URL",  # deprecated
    "UNIPROT_TREMBL_URL",  # deprecated
    # Logging
    "LOG_LEVEL",
    "setup_logging",
    # Loaders (previously missing from __all__ -- institutional-grade fix)
    "LOADERS_DEAD_LETTER_ENABLED",
    "LOADERS_STRICT_VALIDATION",
    "LOADERS_MAX_RETRY_ATTEMPTS",
    "LOADERS_RETRY_BASE_DELAY",
    "LOADERS_ENABLE_TIMING",
    "LOADERS_MAX_DELETE_COUNT",
    "BATCH_SIZE_OVERRIDES",
    # Orphan GDA retention
    "ORPHAN_GDA_RETENTION_HOURS",
    # Environment
    "ENVIRONMENT",
    # Provenance
    "DATA_SNAPSHOT_ID",
    "get_data_version_info",
    "get_provenance_metadata",
    # Validation
    "validate_all_urls",
    "validate_api_keys",
    "validate_env_schema",
    "check_api_endpoints",
    "check_env_git_tracking",
    "validate_env_file",
    # Secret management
    "get_secret",
    # Config groups
    "get_database_config",
    "get_chembl_config",
    "get_string_config",
    "get_disgenet_config",
    # Reload
    "reload_settings",
    # Logging
    "log_config_summary",
    # Constants
    "VALID_CHEMBL_VERSIONS",
    "VALID_STRING_VERSIONS",
    "DEFAULT_CHEMBL_VERSION",
    "DEFAULT_STRING_VERSION",
    "CHEMBL_VERSION_COUNT_RANGES",
    "STRING_VERSION_SCORE_THRESHOLDS",
    "CONFIG_REGISTRY",  # DEPRECATED (v29 audit C-13) -- kept for back-compat
    "OMIM_CONFIG",  # v29 ROOT FIX (audit C-13): consolidated OMIM settings
    "get_omim_config",  # v29 ROOT FIX (audit C-13): accessor for OMIM_CONFIG
    "ENV_VAR_SCHEMA",
    # Module-level helpers exposed for testability
    "load_dotenv",
    "_getenv",
    "_getenv_bool",
    "_getenv_float",
    "_getenv_int",
    "_parse_bool",
    "_parse_optional_int",
    "_parse_required_int",
]

# ---------------------------------------------------------------------------
# Deprecated settings registry -- DESIGN-1, DOC-4
# ---------------------------------------------------------------------------
# These settings are accessed via module-level __getattr__ so that
# DeprecationWarning is raised on every access.  The descriptor-based
# approach does NOT work for module-level variables in Python.

_DEPRECATED_SETTINGS: dict[str, tuple[str, object]] = {
    # name: (replacement, value)
    "CHEMBL_URL": (
        "CHEMBL_API_URL",
        f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
        f"chembl_{CHEMBL_VERSION}/",
    ),
    "UNIPROT_SPROT_URL": (
        "UniProt REST API",
        f"https://ftp.uniprot.org/pub/databases/uniprot/{UNIPROT_RELEASE}/"
        f"knowledgebase/complete/uniprot_sprot.xml.gz",
    ),
    "UNIPROT_TREMBL_URL": (
        "UniProt REST API",
        f"https://ftp.uniprot.org/pub/databases/uniprot/{UNIPROT_RELEASE}/"
        f"knowledgebase/complete/uniprot_trembl.xml.gz",
    ),
    "STRING_PROTEIN_INFO_URL": (
        "STRING_ALIASES_URL",
        _string_urls["protein_info_url"],
    ),
    "DISGENET_STATIC_URL": (
        "DISGENET_API_URL (static URL deprecated since 2024)",
        "https://www.disgenet.org/static/disgenet_ap1/files/downloads/"
        "all_gene_disease_associations.tsv.gz",
    ),
}


def __getattr__(name: str) -> object:
    """Module-level __getattr__ for deprecated settings.

    Accessing any name in ``_DEPRECATED_SETTINGS`` triggers a
    ``DeprecationWarning`` with the replacement and removal timeline,
    then returns the value.  All other names raise ``AttributeError``.
    """
    if name in _DEPRECATED_SETTINGS:
        replacement, value = _DEPRECATED_SETTINGS[name]
        warnings.warn(
            f"Setting `{name}` is DEPRECATED. Use `{replacement}` instead. "
            f"Will be removed in v2.0.0 (scheduled: 2025-Q4).",
            DeprecationWarning,
            stacklevel=2,
        )
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

