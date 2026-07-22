"""v60 ROOT FIX test suite -- verifies all 10 critical forensic issues are addressed.

This test suite is the GROUND TRUTH for the v60 root-fix release. Each test
verifies ONE of the 10 issues from the audit, plus an integration test that
verifies the Phase 1 ↔ Phase 2 connection is 100% wired.

Run with:
    cd phase2 && python -m pytest tests/v60_root_fixes/ -v
"""
