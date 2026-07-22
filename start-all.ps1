#
#   DrugOS - Start All Services
#   Run this from the project root to spin up:
#     1. Embedded PostgreSQL (for auth / users DB)
#     2. Phase 1 Dataset service   (port 8001)
#     3. Phase 2 Knowledge Graph   (port 8002)
#     4. Phase 3 Graph Transformer (port 8003)
#     5. Phase 4 RL service        (port 8004)
#     6. Next.js dev server        (port 3000)
#

$PYTHON = "C:\Users\ADMIN\AppData\Local\Programs\Python\Python314\python.exe"
$ROOT    = Split-Path -Parent $MyInvocation.MyCommand.Path
$FRONTEND = Join-Path $ROOT "frontend"

Write-Host "=== DrugOS: Starting all services ===" -ForegroundColor Cyan

# --- Kill stale processes on relevant ports ---
foreach ($port in @(3000, 8001, 8002, 8003, 8004)) {
    $pids = (netstat -ano | Select-String ":$port\s" | ForEach-Object { ($_ -split '\s+')[-1] } | Sort-Object -Unique)
    foreach ($p in $pids) {
        if ($p -match '^\d+$' -and $p -ne '0') {
            try { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue } catch {}
        }
    }
}
Start-Sleep -Seconds 1
Write-Host "  Cleared stale processes." -ForegroundColor Gray

# --- Start embedded PostgreSQL (for auth DB) ---
Write-Host "  [1/6] Starting embedded PostgreSQL..." -ForegroundColor Yellow
Start-Process -FilePath "node" -ArgumentList "scripts/start-db.mjs" `
    -WorkingDirectory $FRONTEND -WindowStyle Hidden

Start-Sleep -Seconds 4

# --- Start Python ML services ---
Write-Host "  [2/6] Starting Phase 1 Dataset service on port 8001..." -ForegroundColor Yellow
Start-Process -FilePath $PYTHON -ArgumentList "-m", "uvicorn", "phase1.service:app", "--host", "127.0.0.1", "--port", "8001" `
    -WorkingDirectory $ROOT -WindowStyle Hidden

Write-Host "  [3/6] Starting Phase 2 KG service on port 8002..." -ForegroundColor Yellow
Start-Process -FilePath $PYTHON -ArgumentList "-m", "uvicorn", "phase2.service:app", "--host", "127.0.0.1", "--port", "8002" `
    -WorkingDirectory $ROOT -WindowStyle Hidden

Write-Host "  [4/6] Starting Phase 3 GT service on port 8003..." -ForegroundColor Yellow
Start-Process -FilePath $PYTHON -ArgumentList "-m", "uvicorn", "graph_transformer.service:app", "--host", "127.0.0.1", "--port", "8003" `
    -WorkingDirectory $ROOT -WindowStyle Hidden

Write-Host "  [5/6] Starting Phase 4 RL service on port 8004..." -ForegroundColor Yellow
Start-Process -FilePath $PYTHON -ArgumentList "-m", "uvicorn", "rl.service:app", "--host", "127.0.0.1", "--port", "8004" `
    -WorkingDirectory $ROOT -WindowStyle Hidden

# Wait for Python services to boot
Write-Host "  Waiting 6 seconds for ML services to come online..." -ForegroundColor Gray
Start-Sleep -Seconds 6

# Verify services
foreach ($port in @(8001, 8002, 8003, 8004)) {
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3 -ErrorAction Stop
        Write-Host "  Port $port : OK ($($r.service))" -ForegroundColor Green
    } catch {
        Write-Host "  Port $port : WARNING - service may still be starting" -ForegroundColor Red
    }
}

# --- Start Next.js (inherits current environment + .env.local) ---
Write-Host "  [6/6] Starting Next.js dev server on port 3000..." -ForegroundColor Yellow
Set-Location $FRONTEND
npm run dev
