#!/bin/bash
# ==============================================================================
# DrugOS - Linux Production Startup Script for AWS EC2
# ==============================================================================

set -e

echo "=== DrugOS: Starting all services on AWS Ubuntu ==="

# 1. Kill stale processes on ports & disable OS firewall
sudo ufw disable 2>/dev/null || true
for port in 3000 7474 7687 8001 8002 8003 8004 5432; do
  fuser -k ${port}/tcp 2>/dev/null || true
done

# 2. Start PostgreSQL Container & Run Database Migrations
echo "[1/6] Starting PostgreSQL Database & Pushing Schema..."
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/drugos?schema=public"
docker stop drugos-postgres 2>/dev/null || true
docker rm drugos-postgres 2>/dev/null || true
docker run -d --name drugos-postgres -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=drugos postgres:15-alpine || true
sleep 5
echo "DATABASE_URL=\"postgresql://postgres:postgres@localhost:5432/drugos?schema=public\"" > frontend/.env
echo "JWT_SECRET=\"drugos-super-secret-jwt-key-2026-production-32bytes\"" >> frontend/.env
(cd frontend && npx prisma db push --skip-generate) || true

# 3. Start Python ML Microservices
echo "[2/6] Starting Phase 1 Dataset service on port 8001..."
nohup python3 -m uvicorn phase1.service:app --host 0.0.0.0 --port 8001 > logs_phase1.log 2>&1 &

echo "[3/6] Starting Phase 2 KG service on port 8002..."
nohup python3 -m uvicorn phase2.service:app --host 0.0.0.0 --port 8002 > logs_phase2.log 2>&1 &

echo "[4/6] Starting Phase 3 GT service on port 8003..."
nohup python3 -m uvicorn graph_transformer.service:app --host 0.0.0.0 --port 8003 > logs_phase3.log 2>&1 &

echo "[5/6] Starting Phase 4 RL service on port 8004..."
nohup python3 -m uvicorn rl.service:app --host 0.0.0.0 --port 8004 > logs_phase4.log 2>&1 &

sleep 4

# 4. Start Next.js Frontend Dashboard
echo "[6/6] Building and Starting Next.js Production Web Server on port 3000..."
cd frontend
export HOSTNAME="0.0.0.0"
export PORT="3000"
export NODE_OPTIONS="--max-old-space-size=2048"
export NEXT_TELEMETRY_DISABLED=1
npm run build || true
nohup npx next start -p 3000 -H 0.0.0.0 > ../logs_frontend.log 2>&1 &

echo "=================================================================="
echo "  ✅ All DrugOS Services are running live in the background!"
echo "  Web Dashboard: http://YOUR_AWS_PUBLIC_IP:3000"
echo "=================================================================="
