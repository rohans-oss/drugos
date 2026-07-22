#!/usr/bin/env python3
r"""BE-080 REAL ROOT FIX (v2): Consolidated ownership guard.

PRIOR STATE (the audit finding):
    Two pre-commit guard scripts existed with DIFFERENT schemas tracking
    DIFFERENT ownership concepts:
      - pre_commit_issue_guard.py parsed ISSUE_OWNERSHIP.md with regex
        `^(P[1-4]-\d{3}|FE-\d{3}|BUG-#\d+|...)\s*\|\s*...`
      - pre_commit_ownership_guard.py parsed AGENTS_FILE_OWNERSHIP.md with
        regex `^(\S+)\s*\|\s*([^|]+?)\s*\|\s*...`
    The prior "fix" added a deprecation header to this file and made
    main() delegate to pre_commit_ownership_guard.py — but left ~400 lines
    of dead code (parse_ownership_map, check_immutable_files,
    check_claimed_by_other, check_done_files_warning,
    check_unmapped_files, check_deprecated_files, run_pre_commit_hook,
    VERIFICATION_TESTS, run_verification_check, cmd_verify, cmd_list,
    cmd_status, _update_issue_statuses). None of these functions were
    called from main() — they were aspirational dead weight. An agent
    reading the file would waste time understanding functions that never
    run, and might "fix" a function that has no effect. This is exactly
    the "comments are fakes" pattern the audit warned about.

REAL ROOT FIX:
    Delete the dead code entirely. This file is now a THIN DELEGATION
    SHIM: every invocation (including subcommands like `verify`, `list`,
    `status`) is forwarded to pre_commit_ownership_guard.py, which is the
    single source of truth for ownership enforcement. The
    ISSUE_OWNERSHIP.md file is no longer parsed by ANY guard script.

    If you need to add a verification check, add it to
    pre_commit_ownership_guard.py's VERIFICATION_TESTS — do NOT
    re-introduce a parallel system here.

WHY NOT DELETE THE FILE ENTIRELY?
    Existing git hooks and CI workflows may reference this path. Deleting
    it would break those references silently. Keeping it as a thin shim
    (with a clear deprecation notice) lets us migrate callers gradually.
    Once all hooks/CI reference pre_commit_ownership_guard.py directly,
    this file can be removed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Delegate EVERY invocation to the unified ownership guard.

    All subcommands (verify, list, status) and all flags are forwarded
    as-is. The unified guard is the single source of truth.
    """
    ownership_guard = Path(__file__).resolve().parent / "pre_commit_ownership_guard.py"
    if not ownership_guard.exists():
        # v113 FORENSIC ROOT FIX (IN-079, MEDIUM — Security / CI-CD):
        # The previous code FAILED OPEN (returned 0) when the target
        # guard script was missing. For a security/ownership guard,
        # failing OPEN is the WRONG default -- it allows ANY commit
        # through with only a stderr warning. If
        # ``pre_commit_ownership_guard.py`` is accidentally deleted
        # (bad merge, ``git clean -xdf``, fresh clone missing the
        # file), the ownership guard silently stops enforcing. An
        # attacker (or careless developer) could commit to any file
        # without ownership checks.
        #
        # ROOT FIX: fail CLOSED (return 1) when the target script is
        # missing. This blocks ALL commits until the file is restored.
        # The operator MUST either restore the file or update their
        # git hook to point at the correct guard. A CI alert should
        # also be added (separate task) to detect this condition.
        print(
            "ERROR [BE-080 / IN-079]: pre_commit_ownership_guard.py NOT FOUND — "
            "ownership checks CANNOT be performed. FAILING CLOSED (exit 1) "
            "to prevent unauthorized commits. To fix: restore the file "
            "from git (``git checkout HEAD -- scripts/pre_commit_ownership_guard.py``) "
            "or update your git hook to point at the correct guard. "
            "v113 IN-072 ROOT FIX: this guard previously FAILED OPEN "
            "(exit 0) which silently disabled ownership enforcement.",
            file=sys.stderr,
        )
        return 1

    # Forward all arguments (including subcommands) to the unified guard.
    # sys.argv[0] is the script name; [1:] are the actual args.
    result = subprocess.run(
        [sys.executable, str(ownership_guard)] + sys.argv[1:],
        cwd=Path(__file__).resolve().parent.parent,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
