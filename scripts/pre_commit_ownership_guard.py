#!/usr/bin/env python3
"""Pre-commit guard: AGENTS_FILE_OWNERSHIP.md contract enforcement.

Runs in <0.1 seconds. No CI needed. Blocks commits that violate the
ownership contract:

  1. If you edit a file marked CLAIMED by another agent -> BLOCK
  2. If you edit a file marked DONE without bumping its status -> WARN
  3. If you edit an immutable migration (001-011) -> BLOCK
  4. If you forget to update AGENTS_FILE_OWNERSHIP.md when claiming a
     new file -> WARN (remind them to claim)

Install (run once, from repo root):
    cp scripts/pre_commit_ownership_guard.py .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

Or run manually before committing:
    python scripts/pre_commit_ownership_guard.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OWNERSHIP_FILE = REPO_ROOT / "AGENTS_FILE_OWNERSHIP.md"
MIGRATIONS_DIR = REPO_ROOT / "phase1" / "database" / "migrations"


def get_staged_files() -> set[str]:
    """Get the list of files staged for commit."""
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return {f for f in result.stdout.strip().split("\n") if f}


def parse_ownership_map() -> dict[str, dict]:
    """Parse the OWNERSHIP table into {file_path: {status, agent, ...}}."""
    if not OWNERSHIP_FILE.exists():
        return {}
    content = OWNERSHIP_FILE.read_text()
    ownership = {}
    # Match lines like:
    #   path | bug_ids | status | agent | branch | timestamp | note
    pattern = re.compile(
        r"^(\S+)\s*\|\s*([^|]+?)\s*\|\s*(\S+)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*(.*)$",
        re.MULTILINE,
    )
    for m in pattern.finditer(content):
        path, bug_ids, status, agent, branch, timestamp, note = m.groups()
        # Skip header rows and separators
        if path.startswith("#") or path.startswith("---") or path.startswith("<"):
            continue
        if path in ("file_or_dir", "--"):
            continue
        ownership[path] = {
            "bug_ids": bug_ids.strip(),
            "status": status.strip(),
            "agent": agent.strip(),
            "branch": branch.strip(),
            "timestamp": timestamp.strip(),
            "note": note.strip(),
        }
    return ownership


def check_immutable_migrations(staged: set[str]) -> list[str]:
    """Block edits to migrations 001-011."""
    errors = []
    for f in staged:
        # Match phase1/database/migrations/00X_*.sql where 00X is 001-011
        m = re.match(r".*migrations/(\d{3})_.*\.sql$", f)
        if m and 1 <= int(m.group(1)) <= 11:
            errors.append(
                f"BLOCKED: {f} is an immutable migration (001-011). "
                f"Editing applied migrations breaks the immutability contract. "
                f"Create a NEW migration (013+) instead. "
                f"See AGENTS_FILE_OWNERSHIP.md -> IMMUTABLE FILES RULE."
            )
    return errors


def check_claimed_files(staged: set[str], ownership: dict) -> list[str]:
    """Block edits to files CLAIMED by another agent."""
    errors = []
    # Get the current agent's identity from git config
    import subprocess
    user_result = subprocess.run(
        ["git", "config", "user.name"], capture_output=True, text=True,
    )
    current_agent = user_result.stdout.strip() or "unknown"

    for f in staged:
        if f == "AGENTS_FILE_OWNERSHIP.md":
            continue  # editing the ownership file itself is always allowed
        # Find the ownership entry for this file (exact match or parent dir)
        entry = ownership.get(f)
        if entry and entry["status"] == "CLAIMED":
            claimer = entry["agent"]
            if claimer and claimer != current_agent and claimer != "--":
                errors.append(
                    f"BLOCKED: {f} is CLAIMED by {claimer} (since {entry['timestamp']}). "
                    f"Branch: {entry['branch']}. Bug: {entry['bug_ids']}. "
                    f"DO NOT TOUCH -- pick another file or wait for {claimer} to finish. "
                    f"See AGENTS_FILE_OWNERSHIP.md."
                )
    return errors


def check_done_files_warning(staged: set[str], ownership: dict) -> list[str]:
    """Warn (don't block) when editing a file marked DONE."""
    warnings = []
    for f in staged:
        if f == "AGENTS_FILE_OWNERSHIP.md":
            continue
        entry = ownership.get(f)
        if entry and entry["status"] == "DONE":
            warnings.append(
                f"WARNING: {f} is marked DONE (merged in {entry.get('note', 'unknown')}). "
                f"If you're fixing a NEW bug, claim it first by updating "
                f"AGENTS_FILE_OWNERSHIP.md. If you're reverting, update the status."
            )
    return warnings


def check_new_file_claimed(staged: set[str], ownership: dict) -> list[str]:
    """Warn when adding a new code file without claiming it."""
    warnings = []
    code_extensions = {".py", ".sql", ".yml", ".yaml", ".json"}
    for f in staged:
        if f == "AGENTS_FILE_OWNERSHIP.md":
            continue
        if not any(f.endswith(ext) for ext in code_extensions):
            continue
        # Is this file in the ownership map?
        if f not in ownership:
            # Is it a test file? Tests are SHARED, no claim needed.
            if "/tests/" in f or f.startswith("tests/"):
                continue
            warnings.append(
                f"REMINDER: {f} is not in AGENTS_FILE_OWNERSHIP.md. "
                f"If this is a new code file, add an entry with STATUS=CLAIMED "
                f"so other agents know you're working on it."
            )
    return warnings


def main() -> int:
    if not OWNERSHIP_FILE.exists():
        # Ownership file doesn't exist yet -- skip checks (bootstrap mode)
        return 0

    staged = get_staged_files()
    if not staged:
        return 0  # nothing to check

    ownership = parse_ownership_map()

    errors = []
    warnings = []

    errors.extend(check_immutable_migrations(staged))
    errors.extend(check_claimed_files(staged, ownership))
    warnings.extend(check_done_files_warning(staged, ownership))
    warnings.extend(check_new_file_claimed(staged, ownership))

    if warnings:
        print("=" * 70)
        print("OWNERSHIP GUARD -- WARNINGS (commit will proceed):")
        print("=" * 70)
        for w in warnings:
            print(f"  ⚠️  {w}")
        print()

    if errors:
        print("=" * 70)
        print("OWNERSHIP GUARD -- ERRORS (commit BLOCKED):")
        print("=" * 70)
        for e in errors:
            print(f"  🚫 {e}")
        print()
        print("To fix: read AGENTS_FILE_OWNERSHIP.md and either:")
        print("  - claim the file first (edit the map, commit, push, then retry)")
        print("  - pick a different file that's AVAILABLE")
        print("  - if the claim is stale (>24h), ping the agent or claim it yourself")
        return 1

    if not warnings:
        print("✓ ownership guard: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
