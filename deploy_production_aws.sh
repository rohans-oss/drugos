#!/bin/bash
# ==============================================================================
# DrugOS - AWS Full Enterprise Production Deployment & Training Script
# ==============================================================================
# Includes:
#   1. Neo4j Graph Database Server (Docker container)
#   2. Hugging Face ChemBERTa Transformer Deep Embedding Model
#   3. Embedded PostgreSQL User Database
#   4. All 7 Production Data Sources (OpenFDA, Open Targets, ChEMBL, UniProt, STRING, DisGeNET, PubChem)
#   5. 50-Epoch Graph Transformer & 50,000-Timestep RL Training
#   6. Live Next.js Web Application Deployment
# ==============================================================================

set -e

echo "=================================================================="
echo "  DrugOS AWS Full Enterprise Production Deployment & Training"
echo "=================================================================="

export DRUGOS_ENVIRONMENT="production"
export DRUGOS_DOWNLOAD_MODE="full"
export RL_SKIP_LITERATURE="0"

# 1. Install Docker & Neo4j Database Server if not running
echo "[1/6] Setting up Docker & Neo4j Database Server..."
if ! command -v docker &> /dev/null; then
  sudo apt-get update && sudo apt-get install -y docker.io
  sudo systemctl start docker || true
fi

if ! sudo docker ps | grep -q neo4j; then
  sudo docker stop neo4j 2>/dev/null || true
  sudo docker rm neo4j 2>/dev/null || true
  sudo docker run -d --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/drugos_password \
    neo4j:5.18.0 || true
  echo "  Waiting 10 seconds for Neo4j Database to start..."
  sleep 10
fi

export DRUGOS_NEO4J_URI="bolt://localhost:7687"
export USE_NEO4J_BUILDER="1"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="drugos_password"

# 2. Pre-download Hugging Face ChemBERTa Transformer Model Weights
echo "[2/6] Downloading & Warming up Hugging Face ChemBERTa Model Weights..."
pip cache purge 2>/dev/null || true
pip install --no-cache-dir --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu || true
pip install --no-cache-dir --break-system-packages -r requirements.txt || python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

python3 -c "
from transformers import AutoTokenizer, AutoModel
try:
    print('Downloading ChemBERTa weights from Hugging Face...')
    AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    print('ChemBERTa weights loaded successfully!')
except Exception as e:
    print('HuggingFace cache notice:', e)
"

# 3. Run Direct ETL for all 7 Production Sources
echo "[3/6] Executing 7-Source Production ETL (OpenFDA, Open Targets, ChEMBL, UniProt, STRING, DisGeNET, PubChem)..."
python3 scripts/build_clean_7sources.py

# 4. Execute Full Production 4-Phase Model Training
echo "[4/6] Running Full Production 4-Phase Model Training..."
python3 run_4phase.py --gt-epochs 50 --rl-timesteps 50000

# 5. Start Production Web Dashboard & Services
echo "[5/6] Starting DrugOS Platform Services..."
bash start-services.sh

echo "=================================================================="
echo "  ✅ AWS Enterprise Production Deployment & Training Complete!"
echo "  Web Application: http://YOUR_AWS_PUBLIC_IP:3000"
echo "  Neo4j Database Browser: http://YOUR_AWS_PUBLIC_IP:7474"
echo "=================================================================="
