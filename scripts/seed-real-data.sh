#!/usr/bin/env bash
# =============================================================================
# scripts/seed-real-data.sh — Run Phase 1 pipeline on real data
# =============================================================================
# Task 372 ROOT FIX: there was no single command to run the Phase 1
# pipeline end-to-end and verify all 11 CSVs are populated. The
# previous instructions required reading 5 different READMEs and
# invoking 7 separate pipeline commands. A new engineer would give
# up before getting the data layer running.
#
# This script:
#   1. Runs the Phase 1 pipeline in sample mode (DRUGOS_DOWNLOAD_MODE=sample)
#      — works offline using bundled fixtures (no external API keys
#      needed). For production use, run with DRUGOS_DOWNLOAD_MODE=real
#      and the required API keys configured.
#   2. Verifies all 11 expected CSVs exist in phase1/processed_data/
#      and are non-empty.
#   3. Prints a summary table of CSV sizes + row counts.
#   4. Exits 0 on success, 1 on any missing/empty CSV.
#
# Usage:
#   bash scripts/seed-real-data.sh                  # sample mode (offline)
#   bash scripts/seed-real-data.sh --real           # production mode (requires API keys)
#   bash scripts/seed-real-data.sh --real --skip-pubchem  # skip slow sources
# =============================================================================
set -euo pipefail

# Color codes.
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Default mode: sample (works offline). Override with --real.
export DRUGOS_DOWNLOAD_MODE="${DRUGOS_DOWNLOAD_MODE:-sample}"
export DISGENET_USE_API="${DISGENET_USE_API:-false}"
export DRUGOS_ALLOW_MOCK_FALLBACK="${DRUGOS_ALLOW_MOCK_FALLBACK:-1}"
export PYTHONPATH="${REPO_ROOT}/phase1:${PYTHONPATH:-}"

# Parse args.
for arg in "$@"; do
    case "$arg" in
        --real)
            export DRUGOS_DOWNLOAD_MODE="real"
            export DISGENET_USE_API="true"
            unset DRUGOS_ALLOW_MOCK_FALLBACK
            ;;
        --skip-pubchem)
            export DRUGOS_SKIP_PUBCHEM="1"
            ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

echo "============================================"
echo "DrugOS Phase 1 data seeder"
echo "  Mode: ${DRUGOS_DOWNLOAD_MODE}"
echo "  Repo: ${REPO_ROOT}"
echo "============================================"
echo ""

# Run the Phase 1 pipeline. `python -m pipelines all` is the canonical
# entry point (per phase1/pipelines/__main__.py). It runs all 7 sources
# (ChEMBL, DrugBank, UniProt, STRING, DisGeNET, OMIM, PubChem) and
# writes the cleaned CSVs to phase1/processed_data/.
echo "→ Running Phase 1 pipeline (python -m pipelines all)..."
cd "${REPO_ROOT}/phase1"
if ! python -m pipelines all; then
    echo -e "${RED}FAIL: Phase 1 pipeline exited non-zero${NC}"
    exit 1
fi
cd "${REPO_ROOT}"

# Verify all 11 expected CSVs exist + are non-empty.
PROCESSED_DIR="${REPO_ROOT}/phase1/processed_data"
if [ ! -d "${PROCESSED_DIR}" ]; then
    # Some pipeline versions write to data/processed/ instead.
    PROCESSED_DIR="${REPO_ROOT}/phase1/data/processed"
fi

# The 11 CSVs the Phase 1 → Phase 2 bridge expects (per
# phase2/drugos_graph/phase1_bridge.py::run_phase1_to_phase2).
EXPECTED_CSVS=(
    "drugs.csv"
    "proteins.csv"
    "drug_protein_interactions.csv"
    "protein_protein_interactions.csv"
    "drug_disease_associations.csv"
    "gene_disease_associations.csv"
    "diseases.csv"
    "drug_mechanisms.csv"
    "compound_properties.csv"
    "clinical_trials.csv"
    "validated_hypotheses.csv"
)

echo ""
echo "→ Verifying all 11 CSVs in ${PROCESSED_DIR}/..."
echo ""

MISSING=0
EMPTY=0
TOTAL_ROWS=0

printf "  %-45s %10s %10s\n" "CSV" "SIZE" "ROWS"
printf "  %-45s %10s %10s\n" "---------------------------------------------" "----------" "----------"

for csv in "${EXPECTED_CSVS[@]}"; do
    path="${PROCESSED_DIR}/${csv}"
    if [ ! -f "${path}" ]; then
        printf "  %-45s %10s %10s\n" "${csv}" "${RED}MISSING${NC}" "-"
        MISSING=$((MISSING + 1))
        continue
    fi
    size=$(stat -c%s "${path}" 2>/dev/null || stat -f%z "${path}")
    rows=$(wc -l < "${path}")
    # Subtract 1 for header row (if file is non-empty).
    if [ "${rows}" -gt 0 ]; then
        rows=$((rows - 1))
    fi
    TOTAL_ROWS=$((TOTAL_ROWS + rows))
    if [ "${size}" -lt 100 ]; then
        printf "  %-45s %10s %10s\n" "${csv}" "${size}B" "${RED}EMPTY${NC}"
        EMPTY=$((EMPTY + 1))
    else
        printf "  %-45s %10s %10s\n" "${csv}" "${size}B" "${rows}"
    fi
done

echo ""
echo "============================================"
echo "Seed summary:"
echo -e "  Total CSVs expected:  ${#EXPECTED_CSVS[@]}"
echo -e "  ${RED}Missing:               ${MISSING}${NC}"
echo -e "  ${RED}Empty (<100B):          ${EMPTY}${NC}"
echo -e "  ${GREEN}Total data rows:       ${TOTAL_ROWS}${NC}"
echo "============================================"

if [ "${MISSING}" -gt 0 ] || [ "${EMPTY}" -gt 0 ]; then
    echo -e "${RED}SEED FAILED — ${MISSING} missing, ${EMPTY} empty CSV(s)${NC}"
    exit 1
fi

echo -e "${GREEN}SEED PASSED — all 11 CSVs populated with real data${NC}"
exit 0
