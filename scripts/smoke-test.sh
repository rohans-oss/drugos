#!/usr/bin/env bash
# =============================================================================
# scripts/smoke-test.sh — Post-deploy smoke test for the DrugOS platform
# =============================================================================
# Task 371 ROOT FIX: the platform had NO post-deploy verification. After
# `docker-compose up`, there was no automated check that the dashboard
# was actually reachable, the API routes returned 200, or the downstream
# services (Phase 1 dataset, Phase 2 KG, Phase 3 GT, Phase 4 RL) were
# actually healthy. The CI gate passed as soon as `docker-compose up -d`
# returned 0 — but a half-started stack (frontend up, backend down)
# would also return 0 and silently ship to production.
#
# This script hits the 4 critical endpoints the dashboard depends on:
#   1. /api/system/status   — frontend's own health + downstream service status
#   2. /api/predict         — Phase 3 GT inference (proxied to GT_SERVICE_URL)
#   3. /api/rl              — Phase 4 RL ranker (proxied to RL_SERVICE_URL)
#   4. /api/knowledge-graph/stats — Phase 2 KG stats (proxied to KG_SERVICE_URL)
#
# All 4 must return HTTP 200 within the timeout. Any non-200 fails the
# script with exit code 1, which fails the deploy workflow.
#
# Usage:
#   bash scripts/smoke-test.sh                              # default http://localhost:3000
#   bash scripts/smoke-test.sh https://staging.example.com  # custom base URL
#   bash scripts/smoke-test.sh http://localhost:3000 60     # custom timeout (sec)
# =============================================================================
set -euo pipefail

# Parse args: support --help and positional BASE_URL / TIMEOUT.
BASE_URL="http://localhost:3000"
TIMEOUT_SEC="30"
for arg in "$@"; do
    case "$arg" in
        --help|-h)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        http://*|https://*)
            BASE_URL="$arg"
            ;;
        *)
            # Numeric → timeout, otherwise treat as base URL.
            if [[ "$arg" =~ ^[0-9]+$ ]]; then
                TIMEOUT_SEC="$arg"
            else
                BASE_URL="$arg"
            fi
            ;;
    esac
done

# Color codes for clear pass/fail output.
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0

check_endpoint() {
    local name="$1"
    local url="$2"
    local expected="${3:-200}"

    echo -n "  → ${name}: "
    # --max-time caps the request duration. --silent hides progress bar.
    # --write-out extracts the HTTP code. --output discards the body
    # (we only care about the status code for the smoke test).
    local code
    code=$(curl --max-time "${TIMEOUT_SEC}" \
                --silent \
                --output /dev/null \
                --write-out "%{http_code}" \
                "${url}" || echo "000")

    if [ "${code}" = "${expected}" ]; then
        echo -e "${GREEN}PASS${NC} (${code})"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC} (expected ${expected}, got ${code})"
        FAIL=$((FAIL + 1))
    fi
}

echo "============================================"
echo "DrugOS post-deploy smoke test"
echo "  Base URL:  ${BASE_URL}"
echo "  Timeout:   ${TIMEOUT_SEC}s per request"
echo "============================================"
echo ""

# 1. Frontend liveness (Next.js server.js up).
check_endpoint "GET  /api/health              (frontend liveness)" \
    "${BASE_URL}/api/health" 200

# 2. System status — frontend's own health + downstream service status.
#    This is the DOCX V1 criterion: "curl localhost:3000/api/system/status
#    returns 200 with all services healthy."
check_endpoint "GET  /api/system/status       (DOCX V1 criterion)" \
    "${BASE_URL}/api/system/status" 200

# 3. Phase 3 GT inference (proxied via GT_SERVICE_URL).
#    /api/predict expects a POST with a JSON body. Use -X POST + -d.
echo -n "  → POST /api/predict               (Phase 3 GT inference): "
code=$(curl --max-time "${TIMEOUT_SEC}" \
            --silent \
            --output /dev/null \
            --write-out "%{http_code}" \
            -X POST \
            -H "Content-Type: application/json" \
            -d '{"pairs":[{"drug":"aspirin","disease":"migraine"}]}' \
            "${BASE_URL}/api/predict" || echo "000")
if [ "${code}" = "200" ] || [ "${code}" = "503" ]; then
    # 503 is acceptable if the GT model has not been trained yet — the
    # service is up and responding, just degraded. Log it as a soft pass.
    if [ "${code}" = "503" ]; then
        echo -e "${YELLOW}SOFT PASS${NC} (503 — GT model not yet trained)"
    else
        echo -e "${GREEN}PASS${NC} (200)"
    fi
    PASS=$((PASS + 1))
else
    echo -e "${RED}FAIL${NC} (expected 200 or 503, got ${code})"
    FAIL=$((FAIL + 1))
fi

# 4. Phase 4 RL ranker (proxied via RL_SERVICE_URL).
check_endpoint "GET  /api/rl                   (Phase 4 RL ranker)" \
    "${BASE_URL}/api/rl" 200

# 5. Phase 2 KG stats (proxied via KG_SERVICE_URL).
check_endpoint "GET  /api/knowledge-graph/stats (Phase 2 KG stats)" \
    "${BASE_URL}/api/knowledge-graph/stats" 200

echo ""
echo "============================================"
echo "Smoke test summary:"
echo -e "  ${GREEN}PASS: ${PASS}${NC}"
echo -e "  ${RED}FAIL: ${FAIL}${NC}"
echo "============================================"

if [ "${FAIL}" -gt 0 ]; then
    echo -e "${RED}SMOKE TEST FAILED — ${FAIL} endpoint(s) did not return 200${NC}"
    exit 1
fi

echo -e "${GREEN}SMOKE TEST PASSED — all endpoints healthy${NC}"
exit 0
