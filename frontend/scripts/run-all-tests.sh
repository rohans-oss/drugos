#!/usr/bin/env bash
# Run all backend tests + integration tests + E2E tests
# Usage: bash scripts/run-all-tests.sh

set -e

cd "$(dirname "$0")/.."
echo "==================================================="
echo "DrugOS Full Test Suite"
echo "==================================================="

# Ensure no leftover dev servers from previous runs
pkill -9 -f "next dev" 2>/dev/null || true
sleep 2
rm -f .next/dev/lock

echo ""
echo "=== 1. Backend unit tests (Jest) — 67 tests ==="
echo ""
rm -f db/test.db
# IN-031 ROOT FIX (v115, MEDIUM): use `npx jest` instead of `bun x jest`.
# The repo has both package-lock.json (npm) and bun.lock committed —
# having two lockfiles is an anti-pattern because they can drift. The
# Dockerfile uses `npm ci` (frozen, reproducible) so npm is the
# canonical package manager. Using `npx jest` aligns the test runner
# with the production install path. `npx` resolves to the locally-
# installed jest (from node_modules/.bin/jest) — no network fetch.
npx jest src/lib/services/__tests__/ --no-coverage --runInBand --forceExit 2>&1 | tail -20

echo ""
echo "=== 2. Integration tests (Node script) — 21 tests ==="
echo ""
pkill -9 -f "next dev" 2>/dev/null || true
sleep 2
rm -f .next/dev/lock
node scripts/run-integration-tests.js 2>&1 | grep -E "^  [✓✕]|=== Results" | tail -30

echo ""
echo "=== 3. E2E tests (Playwright) — 22 tests ==="
echo ""
pkill -9 -f "next dev" 2>/dev/null || true
sleep 2
rm -f .next/dev/lock
node scripts/run-e2e-tests.js 2>&1 | grep -E "^  [✓✕]|passed|failed" | tail -30

echo ""
echo "==================================================="
echo "All tests complete."
echo "==================================================="
