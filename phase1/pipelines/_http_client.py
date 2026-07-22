# MIT License -- Copyright (c) 2026 Team Cosmic / VentureLab -- see LICENSE
"""v65 ROOT FIX (P1-026): backwards-compatibility shim.

The canonical home for the ChEMBL HTTP client is now
:mod:`pipelines._chembl_http_client`. The previous file name
``_http_client.py`` was misleading because it implied a generic,
pipeline-agnostic HTTP utility -- but the implementation hard-codes
ChEMBL-specific behaviour (token-bucket parameters tuned for ChEMBL's
rate limits, ``CHEMBL_MAX_RESPONSE_BYTES`` size cap, the ChEMBL
User-Agent string, and the ChEMBL REST API URL contract). Only
``chembl_pipeline.py`` imports it.

The file was renamed to ``_chembl_http_client.py`` to reflect the
actual scope. This shim exists ONLY so existing tests and import sites
that reference ``pipelines._http_client`` continue to work without
modification. New code should import from
``pipelines._chembl_http_client`` directly.
"""
from pipelines._chembl_http_client import (  # noqa: F401
    ApiCallRecord,
    CircuitBreakerOpenError,
    HttpClientError,
    MaxResponseSizeExceeded,
    RateLimitedHttpClient,
)

__all__ = [
    "ApiCallRecord",
    "CircuitBreakerOpenError",
    "HttpClientError",
    "MaxResponseSizeExceeded",
    "RateLimitedHttpClient",
]
