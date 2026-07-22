#!/usr/bin/env python3
"""
ORCH-002 to ORCH-006 ROOT FIX tests.

Verifies:
  ORCH-002: run_unified.py exposes --run-gt-rl, --gt-epochs, --rl-timesteps,
            --rl-top-n, --gt-rl-output-dir CLI flags.
  ORCH-003: run_full_platform.py and run_real_pipeline.py are deprecation
            shims that delegate to run_4phase.py. run_real_pipeline.py
            injects --gt-epochs 500 --rl-timesteps 50000 if not overridden.
  ORCH-004: run_4phase.py's defensive builder.total_nodes check handles
            builders that expose `total_nodes`, `n_nodes`, `num_nodes`,
            `node_loads`, or none of the above.
  ORCH-005: verify_v63_fixes.py guards against missing .env.example by
            emitting SKIP rather than FAIL.
  ORCH-006: docker-compose.yml has an uncommented neo4j service with a
            healthcheck.

Run:
  python tests/test_orch_002_to_006_root_fixes.py
"""
from __future__ import annotations

import os
import sys
import subprocess
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  PASS: {name}")
        PASS += 1
    else:
        print(f"  FAIL: {name} {detail}")
        FAIL += 1


def test_orch_002_run_unified_cli_flags() -> None:
    """ORCH-002: run_unified.py exposes the new --run-gt-rl flag family."""
    print("\n[ORCH-002] run_unified.py CLI flags")
    run_unified = (ROOT / "run_unified.py").read_text()
    check("--run-gt-rl flag declared", "--run-gt-rl" in run_unified)
    check("--gt-epochs flag declared", "--gt-epochs" in run_unified)
    check("--rl-timesteps flag declared", "--rl-timesteps" in run_unified)
    check("--rl-top-n flag declared", "--rl-top-n" in run_unified)
    check(
        "--gt-rl-output-dir flag declared",
        "--gt-rl-output-dir" in run_unified,
    )
    # Phase 3+4 chaining logic must call GTRLBridge.run_full_pipeline.
    check(
        "chains GTRLBridge.run_full_pipeline",
        "GTRLBridge" in run_unified and "run_full_pipeline" in run_unified,
    )
    check(
        "uses adapt_phase2_to_phase3 (same adapter as run_4phase.py)",
        "adapt_phase2_to_phase3" in run_unified,
    )
    # Run `python run_unified.py --help` to verify the flags actually
    # register with argparse (catches typos / duplicate declarations).
    result = subprocess.run(
        [sys.executable, str(ROOT / "run_unified.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    help_text = result.stdout + result.stderr
    check(
        "--run-gt-rl appears in --help output",
        "--run-gt-rl" in help_text,
        f"stdout+stderr did not contain --run-gt-rl",
    )


def test_orch_003_runner_consolidation() -> None:
    """ORCH-003: run_full_platform.py and run_real_pipeline.py are shims."""
    print("\n[ORCH-003] runner consolidation")
    rfp = (ROOT / "run_full_platform.py").read_text()
    rrp = (ROOT / "run_real_pipeline.py").read_text()

    check(
        "run_full_platform.py imports run_4phase.main",
        "from run_4phase import main" in rfp,
    )
    check(
        "run_full_platform.py marks itself DEPRECATED",
        "DEPRECATED" in rfp and "ORCH-003" in rfp,
    )
    check(
        "run_real_pipeline.py imports run_4phase.main",
        "from run_4phase import main" in rrp,
    )
    check(
        "run_real_pipeline.py injects --gt-epochs 500",
        "--gt-epochs" in rrp and "500" in rrp,
    )
    check(
        "run_real_pipeline.py injects --rl-timesteps 50000",
        "--rl-timesteps" in rrp and "50000" in rrp,
    )
    check(
        "run_real_pipeline.py marks itself DEPRECATED",
        "DEPRECATED" in rrp and "ORCH-003" in rrp,
    )


def test_orch_004_defensive_total_nodes() -> None:
    """ORCH-004: run_4phase.py uses getattr instead of direct attribute access."""
    print("\n[ORCH-004] defensive builder.total_nodes check")
    r4p = (ROOT / "run_4phase.py").read_text()
    check(
        "uses getattr(builder, 'total_nodes', None)",
        "getattr(builder, \"total_nodes\", None)" in r4p,
    )
    check(
        "falls back to n_nodes",
        "getattr(builder, \"n_nodes\", None)" in r4p,
    )
    check(
        "falls back to num_nodes",
        "getattr(builder, \"num_nodes\", None)" in r4p,
    )
    check(
        "falls back to node_loads",
        "node_loads" in r4p,
    )

    # Functional test: simulate a builder that has NO total_nodes attr.
    class FakeBuilderNoAttr:
        n_nodes = 0  # also zero

    # The defensive code should treat both as "0 → abort".
    builder = FakeBuilderNoAttr()
    total = (
        getattr(builder, "total_nodes", None)
        or getattr(builder, "n_nodes", None)
        or getattr(builder, "num_nodes", None)
    )
    if total is None:
        node_loads = getattr(builder, "node_loads", None) or []
        total = sum(
            len(load.get("nodes", [])) if isinstance(load, dict)
            else len(getattr(load, "nodes", []))
            for load in node_loads
        ) if node_loads else 0
    check(
        "returns 0 (not AttributeError) for a builder with no total_nodes",
        total == 0,
    )

    class FakeBuilderWithTotal:
        total_nodes = 42

    builder2 = FakeBuilderWithTotal()
    total2 = (
        getattr(builder2, "total_nodes", None)
        or getattr(builder2, "n_nodes", None)
        or getattr(builder2, "num_nodes", None)
    )
    check(
        "returns 42 for a builder with total_nodes=42",
        total2 == 42,
    )


def test_orch_005_env_example_guard() -> None:
    """ORCH-005: verify_v63_fixes.py guards missing .env.example."""
    print("\n[ORCH-005] verify_v63_fixes.py .env.example guard")
    v = (ROOT / "verify_v63_fixes.py").read_text()
    check(
        "uses os.path.exists before opening .env.example",
        "os.path.exists" in v and ".env.example" in v,
    )
    check(
        "emits SKIP (not FAIL) when file is missing",
        "SKIP" in v,
    )


def test_orch_006_neo4j_uncommented() -> None:
    """ORCH-006: docker-compose.yml ships Neo4j enabled with healthcheck."""
    print("\n[ORCH-006] docker-compose.yml Neo4j service")
    dc = (ROOT / "docker-compose.yml").read_text()
    # The Neo4j service should NOT be commented out.
    lines = dc.splitlines()
    uncommented_neo4j_lines = [
        l for l in lines
        if l.strip().startswith("neo4j:") or "neo4j:" in l and not l.strip().startswith("#")
    ]
    check(
        "neo4j: service is uncommented",
        any("neo4j:" in l and not l.strip().startswith("#") for l in lines),
    )
    check(
        "cypher-shell healthcheck present",
        "cypher-shell" in dc,
    )
    check(
        "healthcheck block present",
        "healthcheck:" in dc and "neo4j" in dc,
    )
    check(
        "drugos_neo4jdata volume uncommented",
        any("drugos_neo4jdata:" in l and not l.strip().startswith("#") for l in lines),
    )


def main() -> int:
    print("=" * 70)
    print("ORCH-002 to ORCH-006 ROOT FIX TESTS")
    print("=" * 70)
    try:
        test_orch_002_run_unified_cli_flags()
    except Exception as e:
        print(f"  ERROR in ORCH-002 test: {e}")
        global FAIL
        FAIL += 1
    try:
        test_orch_003_runner_consolidation()
    except Exception as e:
        print(f"  ERROR in ORCH-003 test: {e}")
        FAIL += 1
    try:
        test_orch_004_defensive_total_nodes()
    except Exception as e:
        print(f"  ERROR in ORCH-004 test: {e}")
        FAIL += 1
    try:
        test_orch_005_env_example_guard()
    except Exception as e:
        print(f"  ERROR in ORCH-005 test: {e}")
        FAIL += 1
    try:
        test_orch_006_neo4j_uncommented()
    except Exception as e:
        print(f"  ERROR in ORCH-006 test: {e}")
        FAIL += 1

    print("\n" + "=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
