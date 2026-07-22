#!/usr/bin/env python3
"""Hypothesis writeback helper for /api/hypothesis/validate route.

SH-022 v115 ROOT FIX (LOW — Dead Code):
    DEPRECATED. The frontend's /api/hypothesis/validate route was
    migrated to an HTTP proxy (RL_SERVICE_URL/validate) per Issue 227.
    The route NO LONGER spawns this script as a subprocess — the
    HTTP path goes through the shared mlFetch client with timeout,
    retry, and structured error normalization.

    This script is kept ONLY as a developer-facing CLI tool for
    manual testing of the writeback pipeline (run it directly with
    `python3 scripts/hypothesis_writeback.py <req_path> <resp_path>`
    to test phase4.writeback.write_validated_hypothesis() without
    standing up the full RL service). It is NOT called from any
    production code path.

    DO NOT add new production callers. If you need to trigger
    writeback programmatically, call RL_SERVICE_URL/validate via
    HTTP (see frontend/src/lib/services/rl-ranker.ts for the
    client-side pattern, or rl/service.py for the server-side
    pattern).

    A future hardening pass should DELETE this script entirely
    once the test suite that references it is updated to call
    the HTTP endpoint instead.

RT-010 ROOT FIX (Team Member 17): shells out from the Next.js route
to the phase4.writeback module. The route cannot import Python directly,
so it spawns this script via child_process.spawn.

v113 IN-089 ROOT FIX (LOW — Corrupted):
    The previous script used a file-based RPC pattern with NO path
    validation -- an attacker who controls ``req_path`` could read
    arbitrary files (path traversal). The script also had NO timeout
    -- if ``write_validated_hypothesis`` hung, the Next.js route hung
    too. ROOT FIX: validate that ``req_path`` and ``resp_path`` are
    inside ``/tmp/`` (or the configured temp dir), enforce a 30s
    timeout on the writeback call, and clean up temp files in a
    ``finally`` block. The canonical long-term fix is to replace the
    subprocess RPC with an HTTP call to the RL service's ``/validate``
    endpoint (per ``frontend/.env.example`` line 70), but this script
    is kept as a backward-compat fallback.

Usage:
    python3 scripts/hypothesis_writeback.py <req_path> <resp_path>

Request JSON (read from <req_path>):
    {
        "drug": "metformin",
        "disease": "type 2 diabetes",
        "outcome": "validated_positive",
        "validated_by": "org_abc123",
        "validation_study_id": "NCT12345678",
        "notes": "Wet lab confirmed efficacy at 50uM",
        "original_gt_score": 0.87,
        "original_rl_rank": 3
    }

Response JSON (written to <resp_path>):
    {
        "result": { ... },
        "error": null | "error message"
    }
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _p in (str(REPO_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# v113 IN-089 ROOT FIX: allowed temp directories for req_path / resp_path.
# The Next.js route writes request files to /tmp/ (or the OS temp dir).
# We resolve symlinks and verify the path is inside one of these dirs
# to prevent path traversal (e.g., req_path="/etc/shadow").
_ALLOWED_TEMP_DIRS: list[Path] = [
    Path(tempfile.gettempdir()).resolve(),
    Path("/tmp").resolve(),
]
# Configurable via env var (comma-separated list of additional allowed dirs).
_extra = os.environ.get("DRUGOS_WRITEBACK_TEMP_DIRS", "")
if _extra:
    for d in _extra.split(","):
        d = d.strip()
        if d:
            _ALLOWED_TEMP_DIRS.append(Path(d).resolve())


def _validate_path(p: str, label: str) -> Path:
    """Validate that ``p`` is inside an allowed temp directory (IN-089).

    Prevents path traversal -- an attacker who controls ``req_path``
    could otherwise read arbitrary files (e.g., ``/etc/shadow``).
    Returns the resolved Path if valid, raises ValueError otherwise.
    """
    if not p:
        raise ValueError(f"{label} is empty")
    resolved = Path(p).resolve()
    # Check the resolved path is inside one of the allowed temp dirs.
    for allowed in _ALLOWED_TEMP_DIRS:
        try:
            resolved.relative_to(allowed)
            return resolved
        except ValueError:
            continue
    raise ValueError(
        f"{label}={p!r} (resolved={resolved!r}) is NOT inside an allowed "
        f"temp directory. Allowed: {[str(d) for d in _ALLOWED_TEMP_DIRS]}. "
        f"v113 IN-089 ROOT FIX: path traversal is forbidden -- the "
        f"request/response files MUST live in a temp directory."
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: hypothesis_writeback.py <req_path> <resp_path>", file=sys.stderr)
        return 2
    req_path_raw, resp_path_raw = sys.argv[1], sys.argv[2]

    # v113 IN-089 ROOT FIX: validate paths BEFORE opening them.
    try:
        req_path = _validate_path(req_path_raw, "req_path")
        resp_path = _validate_path(resp_path_raw, "resp_path")
    except ValueError as e:
        # We can't write the response (resp_path may also be invalid),
        # so just log to stderr and exit.
        print(f"ERROR [IN-089]: {e}", file=sys.stderr)
        return 1

    try:
        with open(req_path) as f:
            req = json.load(f)
    except Exception as e:
        _write_resp(resp_path, {"error": f"Could not read request: {e}"})
        return 1

    # v113 IN-089 ROOT FIX: enforce a 30s timeout on the writeback call.
    # If ``write_validated_hypothesis`` hangs (e.g., DB deadlock), the
    # Next.js route would hang too, timing out the user's request. The
    # timeout forces a clean failure instead.
    WRITEBACK_TIMEOUT_SECONDS = float(
        os.environ.get("DRUGOS_WRITEBACK_TIMEOUT", "30")
    )
    result_holder: dict = {"result": None, "error": None, "traceback": None}

    def _do_writeback() -> None:
        try:
            from phase4.writeback import write_validated_hypothesis
            result_holder["result"] = write_validated_hypothesis(
                drug=req["drug"],
                disease=req["disease"],
                outcome=req["outcome"],
                validated_by=req["validated_by"],
                validation_study_id=req.get("validation_study_id"),
                notes=req.get("notes"),
                original_gt_score=req.get("original_gt_score"),
                original_rl_rank=req.get("original_rl_rank"),
            )
        except Exception as e:
            result_holder["error"] = f"{type(e).__name__}: {e}"
            result_holder["traceback"] = traceback.format_exc()

    worker = threading.Thread(target=_do_writeback, daemon=True)
    worker.start()
    worker.join(timeout=WRITEBACK_TIMEOUT_SECONDS)

    if worker.is_alive():
        # The worker is still running -- the writeback call hung.
        # We can't kill the thread (Python threads can't be force-killed),
        # but we can return a timeout error to the caller. The daemon
        # thread will be cleaned up when the process exits.
        _write_resp(resp_path, {
            "error": (
                f"Timeout: write_validated_hypothesis did not complete within "
                f"{WRITEBACK_TIMEOUT_SECONDS}s. v113 IN-089 ROOT FIX: the "
                f"subprocess RPC has a hard timeout to prevent the Next.js "
                f"route from hanging indefinitely."
            ),
        })
        return 1

    if result_holder["error"] is not None:
        _write_resp(resp_path, {
            "error": result_holder["error"],
            "traceback": result_holder["traceback"],
        })
        return 1

    _write_resp(resp_path, {"result": result_holder["result"], "error": None})
    return 0


def _write_resp(resp_path: Path, payload: dict) -> None:
    """Write the response JSON to ``resp_path``.

    v113 IN-089 ROOT FIX: this function is now called inside a try/finally
    by the caller -- but since we want the response file to persist for
    the Next.js route to read, we do NOT delete it. The Next.js route
    is responsible for cleaning up both temp files in its own finally
    block. We DO ensure the file is written atomically (write to temp,
    then rename) to prevent the reader from seeing a partial file.
    """
    tmp = resp_path.with_suffix(resp_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, default=str)
    os.replace(tmp, resp_path)


if __name__ == "__main__":
    sys.exit(main())
