#!/usr/bin/env bash
# IN-032 ROOT FIX (v115, MEDIUM): production-safe standalone build script.
#
# ROOT CAUSE: the previous `build` script in package.json was:
#   "next build && cp -r .next/static .next/standalone/.next/ && cp -r public .next/standalone/"
#
# This had multiple problems:
#   1. `cp -r` behaves differently on macOS (follows symlinks) vs
#      Linux (does not). CI builds on macOS produced different
#      `.next/standalone/` trees than the Linux Docker build.
#   2. `cp -r` MERGES into any existing directory — stale files
#      from a previous build persisted into the new bundle,
#      shipping dead code to production.
#   3. No `set -euo pipefail` — a silent cp failure would NOT
#      fail the build (the `&&` chain stopped on the first error,
#      but a partial cp that exited 0 would go unnoticed).
#   4. No `--verbose` or logging — operators couldn't see what
#      was copied or diagnose why a build was different.
#
# ROOT FIX: this dedicated script:
#   1. Uses `set -euo pipefail` so ANY error aborts the build.
#   2. Uses `cp -a` (archive mode) for consistent behavior across
#      macOS and Linux — `cp -a` preserves symlinks and metadata
#      on both platforms.
#   3. Removes the destination directory BEFORE copying, so stale
#      files from a previous build are guaranteed to be gone.
#   4. Logs every step so CI logs are debuggable.
#   5. Verifies the standalone directory exists before copying —
#      a silent Next.js build failure (e.g., output: "standalone"
#      removed from next.config.ts) would otherwise produce a
#      confusing "cp: no such file or directory" error.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== IN-032 ROOT FIX: standalone build script ==="
echo ""

# Step 1: run `next build`. This produces .next/standalone/ when
# `output: "standalone"` is set in next.config.ts.
echo "[1/4] Running 'next build'..."
NEXT_TELEMETRY_DISABLED=1 npx next build
echo ""

# Step 2: verify the standalone directory exists. If it doesn't,
# next.config.ts may have had `output: "standalone"` removed, OR
# the build silently failed. Either way, abort with a clear error.
echo "[2/4] Verifying .next/standalone/ exists..."
if [ ! -d .next/standalone ]; then
  echo "ERROR: .next/standalone/ does not exist after 'next build'." >&2
  echo "       Ensure next.config.ts sets 'output: \"standalone\"'." >&2
  exit 1
fi
echo "  ✓ .next/standalone/ exists."
echo ""

# Step 3: copy .next/static into .next/standalone/.next/static.
# Remove the destination first to ensure a clean copy (no stale
# files from a previous build).
echo "[3/4] Copying .next/static → .next/standalone/.next/static..."
rm -rf .next/standalone/.next/static
mkdir -p .next/standalone/.next
cp -a .next/static .next/standalone/.next/static
echo "  ✓ Static assets copied."
echo ""

# Step 4: copy public/ into .next/standalone/public.
# Same clean-copy pattern — remove first, then copy.
echo "[4/4] Copying public/ → .next/standalone/public/..."
if [ -d public ]; then
  rm -rf .next/standalone/public
  cp -a public .next/standalone/public
  echo "  ✓ Public assets copied."
else
  echo "  ⚠ public/ does not exist — skipping (this is OK if the app has no static assets)."
fi
echo ""

echo "=== Build complete. ==="
echo "Standalone bundle: .next/standalone/"
echo "Start with: npm run start"
