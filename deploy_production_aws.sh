#!/bin/bash
# ==============================================================================
# DrugOS - AWS Cloud Production Deployment & Training Script
# ==============================================================================
# Run this script on your AWS EC2 instance to execute full production-scale
# data ingestion (ChEMBL, OpenFDA, Open Targets, STRING, UniProt, DisGeNET, PubChem)
# and train the Graph Transformer & RL Ranker models on full bulk datasets.
# ==============================================================================

set -e

echo "=================================================================="
echo "  DrugOS AWS Production Deployment & Full-Scale Training"
echo "=================================================================="

# 1. Environment Setup
export DRUGOS_ENVIRONMENT="production"
export DRUGOS_DOWNLOAD_MODE="full"
export RL_SKIP_LITERATURE="0"

echo "[1/4] Installing system dependencies and Python packages..."
sudo apt-get update && sudo apt-get install -y build-essential git python3-pip python3-venv postgresql-client
python3 -m pip install --break-system-packages --upgrade pip || true
pip install --break-system-packages -r requirements.txt || python3 -m pip install --break-system-packages -r requirements.txt

# 2. Run Direct ETL for all 7 sources
echo "[2/4] Executing 7-Source Production ETL (OpenFDA, Open Targets, ChEMBL, UniProt, STRING, DisGeNET, PubChem)..."
python3 scripts/build_clean_7sources.py

# 3. Execute Full Production 4-Phase Model Training
echo "[3/4] Running Full Production 4-Phase Model Training..."
python3 run_4phase.py --gt-epochs 50 --rl-timesteps 50000

# 4. Start Production Web Dashboard
echo "[4/4] Starting DrugOS Platform Services..."
bash start-services.sh

echo "=================================================================="
echo "  ✅ AWS Production Deployment & Training Complete!"
echo "=================================================================="
