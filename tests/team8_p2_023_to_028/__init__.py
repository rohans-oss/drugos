"""Team 8 — P2-023 to P2-028 regression tests.

Each test verifies ONE root fix in isolation. Tests are designed to FAIL
if the fix is reverted (forensic root-cause verification, not surface
fixes). Run with:

    pytest tests/team8_p2_023_to_028/ -v

These tests do NOT depend on Neo4j, HuggingFace, or a GPU -- they
exercise the fixed code paths directly with synthetic inputs. They
import the REAL production modules (not test stubs).
"""
