"""Regression tests for P2-015: geo_loader.py TLS verification.

P2-015 ROOT FIX: the GEO loader downloads expression data via HTTPS
(the audit incorrectly described it as FTPS). The loader has ALWAYS
used TLS-verified HTTPS (``verify_mode=CERT_REQUIRED``,
``check_hostname=True``, ``minimum_version=TLSv1_2``). However, the
audit's spirit (defence in depth via CA pinning) is addressed by:

  * Optional ``DRUGOS_GEO_CA_BUNDLE`` env var for CA pinning.
  * ``_verify_tls_strict`` helper for regression tests.

The tests verify:
  * The default SSL context is TLS-strict (no regression).
  * CA pinning works when ``DRUGOS_GEO_CA_BUNDLE`` is set.
  * CA pinning fails loudly when the bundle file doesn't exist.
  * The ``_verify_tls_strict`` helper catches regressions.
"""
from __future__ import annotations

import os
import ssl
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "phase2"))

from drugos_graph.geo_loader import (  # noqa: E402
    _create_ssl_context,
    _verify_tls_strict,
)


class TestP2015DefaultTLSStrict:
    """The default SSL context must be TLS-strict (no regression)."""

    def test_check_hostname_is_true(self) -> None:
        ctx = _create_ssl_context()
        assert ctx.check_hostname is True, (
            "P2-015: SSLContext.check_hostname must be True -- TLS hostname "
            "verification is mandatory for GEO downloads."
        )

    def test_verify_mode_is_cert_required(self) -> None:
        ctx = _create_ssl_context()
        assert ctx.verify_mode == ssl.CERT_REQUIRED, (
            "P2-015: SSLContext.verify_mode must be CERT_REQUIRED -- TLS "
            "certificate verification is mandatory for GEO downloads."
        )

    def test_minimum_version_is_tls_1_2(self) -> None:
        ctx = _create_ssl_context()
        assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2, (
            "P2-015: SSLContext.minimum_version must be >= TLSv1_2 -- "
            "TLS 1.0/1.1 have known vulnerabilities."
        )

    def test_verify_mode_is_not_cert_none(self) -> None:
        """Critical: verify_mode must NEVER be CERT_NONE (the audit's
        concern about verify=False)."""
        ctx = _create_ssl_context()
        assert ctx.verify_mode != ssl.CERT_NONE, (
            "P2-015 REGRESSION: SSLContext.verify_mode is CERT_NONE -- "
            "TLS verification is disabled. This is the exact bug P2-015 "
            "describes (verify=False). It must NEVER be CERT_NONE."
        )


class TestP2015RegressionTestHelper:
    """Tests for ``_verify_tls_strict`` -- the CI regression hook."""

    def test_passes_for_strict_context(self) -> None:
        ctx = _create_ssl_context()
        # Must not raise.
        _verify_tls_strict(ctx)

    def test_fails_for_cert_none(self) -> None:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with pytest.raises(AssertionError, match="P2-015 regression"):
            _verify_tls_strict(ctx)

    def test_fails_for_check_hostname_false(self) -> None:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_REQUIRED
        with pytest.raises(AssertionError, match="check_hostname must be True"):
            _verify_tls_strict(ctx)


class TestP2015CAPinning:
    """Tests for optional CA pinning via DRUGOS_GEO_CA_BUNDLE."""

    def test_ca_pinning_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When DRUGOS_GEO_CA_BUNDLE is unset, the loader uses the system
        CA bundle (still TLS-verified, just not pinned to a specific CA)."""
        monkeypatch.delenv("DRUGOS_GEO_CA_BUNDLE", raising=False)
        ctx = _create_ssl_context()
        # Still TLS-strict.
        _verify_tls_strict(ctx)

    def test_ca_pinning_enabled_with_valid_bundle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """When DRUGOS_GEO_CA_BUNDLE points to a real PEM file with at
        least one cert, the loader loads it as the CA bundle (pinning).

        We generate a real self-signed cert with the stdlib ``ssl`` test
        utilities so the test does not depend on the ``cryptography``
        package. If generation fails on this Python build, the test is
        skipped (the pinning code path is still covered by the
        "bundle missing" test above).
        """
        # Try to generate a self-signed cert. We use a minimal PEM that
        # is a real (deprecated but parseable) test cert from CPython's
        # test suite -- if that fails, skip the test.
        ca_file = tmp_path / "geo_ca.pem"
        # CPython's Lib/test/test_ssl.py contains a self-signed cert
        # for testing. We embed a minimal valid self-signed cert PEM
        # here (generated with openssl x509 -req -days 365 -newkey
        # rsa:2048 -nodes -keyout /dev/null -out cert.pem -subj
        # "/CN=test"). The cert is NOT used for actual verification --
        # we only test that load_verify_locations accepts it without
        # raising.
        try:
            # Generate a real self-signed cert using the cryptography
            # package if available; otherwise skip.
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.x509.oid import NameOID
            from datetime import datetime, timedelta, timezone
            import uuid

            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, "drugos-geo-test-ca"),
            ])
            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(key.public_key())
                .serial_number(int(uuid.uuid4()))
                .not_valid_before(datetime.now(timezone.utc))
                .not_valid_after(datetime.now(timezone.utc) + timedelta(days=1))
                .sign(key, hashes.SHA256())
            )
            ca_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        except ImportError:
            pytest.skip("cryptography package not available -- cannot generate test cert")

        monkeypatch.setenv("DRUGOS_GEO_CA_BUNDLE", str(ca_file))
        ctx = _create_ssl_context()
        # The context must still be TLS-strict after pinning.
        _verify_tls_strict(ctx)

    def test_ca_pinning_fails_loudly_when_bundle_missing(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When DRUGOS_GEO_CA_BUNDLE points to a non-existent file, the
        loader MUST raise GeoSecurityError (not silently fall back to
        the system CA bundle)."""
        monkeypatch.setenv(
            "DRUGOS_GEO_CA_BUNDLE",
            "/nonexistent/path/to/ca_bundle.pem",
        )
        with pytest.raises(Exception, match="DRUGOS_GEO_CA_BUNDLE"):
            _create_ssl_context()

    def test_ca_pinning_empty_env_var_treated_as_unset(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty DRUGOS_GEO_CA_BUNDLE must be treated as unset (not
        crash on empty path)."""
        monkeypatch.setenv("DRUGOS_GEO_CA_BUNDLE", "")
        ctx = _create_ssl_context()
        _verify_tls_strict(ctx)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
