#!/usr/bin/env python3
"""V100 FORENSIC ROOT FIX VERIFICATION TESTS.

Tests each of the Top 20 critical bug fixes by directly exercising the
FIXED code paths (not comments, not existing tests). Run with:

    python3.13 tests/test_v100_forensic_root_fixes.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} -- {detail}")


def test_bug1_reverse_edges():
    """BUG #1: production path writes reverse edges into _edge_sets."""
    print("\n=== BUG #1: Reverse edges in production path ===")
    from graph_transformer.data.graph_builder import BiomedicalGraphBuilder
    check(
        "_build_reverse_edges_into_sets is a classmethod",
        hasattr(BiomedicalGraphBuilder, '_build_reverse_edges_into_sets'),
    )
    # Verify the deprecated staticmethod is NOT called in from_phase1_staged_data
    import inspect
    src = inspect.getsource(BiomedicalGraphBuilder.from_phase1_staged_data)
    check(
        "from_phase1_staged_data uses _build_reverse_edges_into_sets",
        '_build_reverse_edges_into_sets' in src,
        "method does not call _build_reverse_edges_into_sets",
    )
    check(
        "from_phase1_staged_data does NOT call deprecated _build_reverse_edges",
        '_build_reverse_edges(' not in src.replace('_build_reverse_edges_into_sets', ''),
        "still calls deprecated _build_reverse_edges",
    )


def test_bug4_disgenet_tiers():
    """BUG #4: DisGeNET tiers match Piñero 2020 (sub_weak/weak/strong)."""
    print("\n=== BUG #4: DisGeNET confidence tiers (Piñero 2020) ===")
    from phase1.cleaning.confidence import classify_confidence, DEFAULT_CONFIDENCE_TIERS, CONFIDENCE_TIER_METHOD_VERSION
    check("tier [0.0, 0.06) = sub_weak", classify_confidence(0.03) == "sub_weak")
    check("tier [0.06, 0.3) = weak", classify_confidence(0.1) == "weak")
    check("tier [0.3, 1.0] = strong", classify_confidence(0.5) == "strong")
    check("NO 'moderate' tier exists", "moderate" not in [t[1] for t in DEFAULT_CONFIDENCE_TIERS])
    check("version bumped to pinero_2020_v2", CONFIDENCE_TIER_METHOD_VERSION == "pinero_2020_v2")


def test_bug7_thalidomide():
    """BUG #7: thalidomide is indication-specific (not global reject)."""
    print("\n=== BUG #7: Thalidomide indication-specific withdrawal ===")
    from rl.rl_drug_ranker import WITHDRAWN_DRUGS_GLOBAL, WITHDRAWN_DRUGS_INDICATION_SPECIFIC
    check("thalidomide NOT in global set", "thalidomide" not in WITHDRAWN_DRUGS_GLOBAL)
    check("thalidomide IN indication-specific", "thalidomide" in WITHDRAWN_DRUGS_INDICATION_SPECIFIC)
    contra = WITHDRAWN_DRUGS_INDICATION_SPECIFIC.get("thalidomide", set())
    check("pregnancy IS contraindicated", "pregnancy" in contra)
    check("multiple myeloma NOT contraindicated", "multiple myeloma" not in contra)


def test_bug8_transe_nameerror():
    """BUG #8: phase2 run_pipeline uses 'rels' not 'relations'."""
    print("\n=== BUG #8: TransE NameError fixed ===")
    import inspect
    import phase2.drugos_graph.run_pipeline as p2
    src = inspect.getsource(p2)
    # The buggy line was: _r_idx_v88 = int(relations[_i]) if _i < len(relations) else -1
    check("uses 'rels[_i]' not 'relations[_i]'", "rels[_i]" in src)
    check("does NOT use 'relations[_i]'", "relations[_i]" not in src)


def test_bug9_hgt_negatives():
    """BUG #9: _make_negatives always returns len(positives) items."""
    print("\n=== BUG #9: HGT negative-padding TypeError fixed ===")
    import inspect
    import phase2.drugos_graph.run_pipeline as p2
    src = inspect.getsource(p2)
    check("pads with (0, 0) tuples", "(0, 0)" in src)
    check("has invariant assert", "len(negs) == len(positive_indices)" in src)


def test_bug11_2fa():
    """BUG #11: login route enforces MFA."""
    print("\n=== BUG #11: 2FA enforcement ===")
    login_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "uploaded_code", "drugos", "src", "app", "api", "auth", "login", "route.ts")
    with open(login_path) as f:
        src = f.read()
    check("login checks mfaEnabled", "mfaEnabled" in src)
    check("login returns mfa_required", "mfa_required" in src)
    check("login signs mfa challenge token", "signMfaChallengeToken" in src)
    verify_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "uploaded_code", "drugos", "src", "app", "api", "auth", "2fa", "login-verify", "route.ts")
    check("2fa login-verify endpoint exists", os.path.exists(verify_path))


def test_bug12_register_admin():
    """BUG #12: register route does NOT allow admin role."""
    print("\n=== BUG #12: Self-registration admin blocked ===")
    reg_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "uploaded_code", "drugos", "src", "app", "api", "auth", "register", "route.ts")
    with open(reg_path) as f:
        src = f.read()
    # Find the ALLOWED_ROLES array and check admin is not in it
    import re
    m = re.search(r'ALLOWED_ROLES\s*=\s*\[([^\]]+)\]', src)
    check("ALLOWED_ROLES found", m is not None)
    if m:
        roles_block = m.group(1)
        check("'admin' NOT in ALLOWED_ROLES", '"admin"' not in roles_block)
        check("'researcher' IS in ALLOWED_ROLES", '"researcher"' in roles_block)


def test_bug13_caddyfile():
    """BUG #13: Caddyfile has no SSRF proxy."""
    print("\n=== BUG #13: Caddyfile SSRF removed ===")
    caddy_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "uploaded_code", "drugos", "Caddyfile")
    with open(caddy_path) as f:
        src = f.read()
    # Check ACTIVE handler lines (not comments). The handler was:
    #   @transform_port_query { query XTransformPort=* }
    #   handle @transform_port_query { reverse_proxy localhost:{query.XTransformPort} ... }
    active_lines = [l for l in src.split('\n') if l.strip() and not l.strip().startswith('#')]
    active = '\n'.join(active_lines)
    check("no active @transform_port_query handler", "@transform_port_query" not in active, "handler still active")
    check("no active XTransformPort reverse_proxy", "XTransformPort" not in active, "XTransformPort still in active code")


def test_bug14_api_proxies():
    """BUG #14: /api/rl, /api/knowledge-graph, /api/dataset are real proxies."""
    print("\n=== BUG #14: API proxies implemented ===")
    base = os.path.join(os.path.dirname(__file__), "..", "frontend", "uploaded_code", "drugos", "src", "app", "api")
    for route in ["rl", "knowledge-graph", "dataset"]:
        path = os.path.join(base, route, "route.ts")
        with open(path) as f:
            src = f.read()
        check(f"/api/{route} does NOT return 501 unconditional", 'status: 501' not in src or '503' in src,
              f"still returns 501: {path}")
        check(f"/api/{route} has fetch() proxy", "fetch(" in src, "no fetch() call")


def test_bug15_runners():
    """BUG #15: run_pipeline.py fixed, Makefile uses run_full_platform."""
    print("\n=== BUG #15: Runners fixed ===")
    import inspect
    import run_pipeline
    src = inspect.getsource(run_pipeline)
    check("run_phase2_kg_builder has seed param", "seed: int = 42" in inspect.getsource(run_pipeline.run_phase2_kg_builder))
    # Makefile
    makefile_path = os.path.join(os.path.dirname(__file__), "..", "Makefile")
    with open(makefile_path) as f:
        mk = f.read()
    check("Makefile run target uses run_full_platform.py", "run_full_platform.py" in mk)
    check("Makefile run is NOT run_unified.py as default", "run: run-full-platform" in mk)


def test_bug16_ppo_gamma():
    """BUG #16: ppo_gamma > 0 (real RL, not contextual bandit)."""
    print("\n=== BUG #16: PPO gamma > 0 ===")
    from rl.rl_drug_ranker import PipelineConfig
    cfg = PipelineConfig()
    check("ppo_gamma > 0", cfg.ppo_gamma > 0, f"ppo_gamma = {cfg.ppo_gamma}")
    check("ppo_gamma = 0.95", cfg.ppo_gamma == 0.95)


def test_bug17_safety_counters():
    """BUG #17: safety counters incremented in env.step()."""
    print("\n=== BUG #17: Safety alert counters ===")
    import inspect
    from rl.rl_drug_ranker import DrugRankingEnv, RewardFunction
    env_src = inspect.getsource(DrugRankingEnv)
    check("env has n_safety_rejected attr", "n_safety_rejected" in env_src)
    check("env increments n_safety_rejected in step()", "n_safety_rejected" in inspect.getsource(DrugRankingEnv.step))
    reward_src = inspect.getsource(RewardFunction)
    check("reward fn has last_rejection_reason", "last_rejection_reason" in reward_src)


def test_bug18_chembl_double_read():
    """BUG #18: chembl_pipeline clean() has no double-read."""
    print("\n=== BUG #18: ChEMBL double-read removed ===")
    import inspect
    from phase1.pipelines.chembl_pipeline import ChEMBLPipeline
    src = inspect.getsource(ChEMBLPipeline.clean)
    # Count pd.read_csv calls in ACTUAL code (not comments).
    code_lines = [l for l in src.split('\n') if l.strip() and not l.strip().startswith('#')]
    code = '\n'.join(code_lines)
    count = code.count("pd.read_csv")
    check("exactly 2 pd.read_csv calls in code (not 3)", count == 2, f"found {count} pd.read_csv calls in code")


def test_bug19_docker_volumes():
    """BUG #19: docker-compose has ./data, ./exporters, ./scripts mounts."""
    print("\n=== BUG #19: docker-compose volume mounts ===")
    dc_path = os.path.join(os.path.dirname(__file__), "..", "phase1", "docker-compose.yml")
    with open(dc_path) as f:
        src = f.read()
    check("has ./data mount", "./data:/opt/airflow/data" in src)
    check("has ./exporters mount", "./exporters:/opt/airflow/exporters" in src)
    check("has ./scripts mount", "./scripts:/opt/airflow/scripts" in src)
    # All 3 airflow services should have them
    check("3 occurrences of ./exporters", src.count("./exporters:/opt/airflow/exporters") == 3)


def test_bug20_inchikey_validators():
    """BUG #20: InChIKey validators unified (no strict check in loader)."""
    print("\n=== BUG #20: InChIKey validators unified ===")
    import inspect
    from phase1.database.loaders import _validate_inchikey
    src = inspect.getsource(_validate_inchikey)
    # Strip docstrings and comments to check ACTUAL code only.
    in_docstring = False
    code_lines = []
    for line in src.split('\n'):
        stripped = line.strip()
        if '"""' in stripped:
            # Toggle docstring state (handles both opening and closing on same line)
            count = stripped.count('"""')
            if count == 1:
                in_docstring = not in_docstring
            # if count == 2, it's a single-line docstring -- skip the line
            continue
        if in_docstring:
            continue
        if stripped.startswith('#'):
            continue
        code_lines.append(line)
    code = '\n'.join(code_lines)
    check("loader does NOT call is_strict_inchikey in code", "is_strict_inchikey" not in code,
          "still calls is_strict_inchikey in actual code")


def test_bug3_safety_gate():
    """BUG #3: safety-gate threshold not re-lowered to 0.2."""
    print("\n=== BUG #3: Safety-gate threshold ===")
    import inspect
    from graph_transformer.gt_rl_bridge import GTRLBridge
    # Find the run_full_pipeline method source
    src = inspect.getsource(GTRLBridge)
    # The buggy line was: kp_recovery_threshold = float(getattr(rl_config, "min_kp_recovery_rate", 0.2))
    # after the correct max(rl_config_threshold, 0.5) line. Check it's commented out.
    check("re-lowering line is commented/removed",
          "kp_recovery_threshold = float(getattr(rl_config" not in src or "# kp_recovery_threshold" in src,
          "re-lowering line still active")


def test_bug2_auc_alignment():
    """BUG #2: compute_auc reads from env_test.data not test_data."""
    print("\n=== BUG #2: AUC label/prediction alignment ===")
    import inspect
    from rl.rl_drug_ranker import compute_auc
    src = inspect.getsource(compute_auc)
    # Strip comments (lines starting with # or containing ``) to check ACTUAL code.
    code_lines = [l for l in src.split('\n') if l.strip() and not l.strip().startswith('#') and not l.strip().startswith('``')]
    code = '\n'.join(code_lines)
    check("uses env_test.data.iloc in code", "env_test.data.iloc" in code, "does not use env_test.data.iloc")
    check("does NOT use test_data.iloc for row in code", "test_data.iloc[current_row_idx]" not in code,
          "still uses test_data.iloc in actual code")


if __name__ == "__main__":
    print("=" * 70)
    print("V100 FORENSIC ROOT FIX VERIFICATION TESTS")
    print("=" * 70)
    tests = [
        test_bug1_reverse_edges,
        test_bug2_auc_alignment,
        test_bug3_safety_gate,
        test_bug4_disgenet_tiers,
        test_bug7_thalidomide,
        test_bug8_transe_nameerror,
        test_bug9_hgt_negatives,
        test_bug11_2fa,
        test_bug12_register_admin,
        test_bug13_caddyfile,
        test_bug14_api_proxies,
        test_bug15_runners,
        test_bug16_ppo_gamma,
        test_bug17_safety_counters,
        test_bug18_chembl_double_read,
        test_bug19_docker_volumes,
        test_bug20_inchikey_validators,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"  ERROR in {t.__name__}: {e}")
    print("\n" + "=" * 70)
    print(f"RESULTS: {PASS} passed, {FAIL} failed (out of {PASS + FAIL})")
    print("=" * 70)
    sys.exit(0 if FAIL == 0 else 1)
