/**
 * ORCH-003 ROOT FIX (v2) regression test: the duplicate 4-phase runners
 * emit a deprecation notice and the Makefile default uses run_4phase.py.
 *
 * This test would have caught the original ORCH-003 bug: three runners
 * (run_4phase, run_full_platform, run_real_pipeline) did the same thing
 * with different code paths and different defaults, causing "works in CI,
 * breaks in prod" situations.
 *
 * Root fix:
 *   - run_full_platform.py and run_real_pipeline.py emit a stderr
 *     deprecation warning pointing to run_4phase.py.
 *   - The Makefile `make run` target now invokes run_4phase.py (not
 *     run_full_platform.py).
 *
 * This is a SOURCE-LEVEL regression test.
 */
import * as fs from "fs";
import * as path from "path";

const REPO_ROOT = path.resolve(__dirname, "../../../../../../../..");

function read(p: string): string {
  return fs.readFileSync(p, "utf-8");
}

describe("ORCH-003 (v2): duplicate runners deprecated, Makefile consolidated", () => {
  test("run_full_platform.py emits a deprecation notice on stderr", () => {
    const src = read(path.join(REPO_ROOT, "run_full_platform.py"));
    expect(src).toMatch(/ORCH-003 DEPRECATION NOTICE: run_full_platform\.py is deprecated/);
    expect(src).toMatch(/canonical 4-phase runner is now `run_4phase\.py`/);
  });

  test("run_real_pipeline.py emits a deprecation notice on stderr", () => {
    const src = read(path.join(REPO_ROOT, "run_real_pipeline.py"));
    expect(src).toMatch(/ORCH-003 DEPRECATION NOTICE: run_real_pipeline\.py is deprecated/);
    expect(src).toMatch(/canonical 4-phase runner is now `run_4phase\.py`/);
  });

  test("Makefile `make run` target invokes run_4phase.py", () => {
    const mk = read(path.join(REPO_ROOT, "Makefile"));
    // The `run:` target must depend on `run-4phase` (NOT run-full-platform).
    expect(mk).toMatch(/^run:\s*run-4phase\s*$/m);
    // The run-4phase target must invoke run_4phase.py.
    expect(mk).toMatch(/run-4phase:\n\t\$\(PYTHON\) run_4phase\.py/);
  });

  test("Makefile help text documents the deprecation", () => {
    const mk = read(path.join(REPO_ROOT, "Makefile"));
    expect(mk).toMatch(/DEPRECATED \(ORCH-003\)/);
    expect(mk).toMatch(/CANONICAL runner per ORCH-003/);
  });
});
