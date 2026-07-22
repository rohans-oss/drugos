$python_exe = "C:\Users\ADMIN\AppData\Local\Programs\Python\Python314\python.exe"

Write-Host "Starting DrugOS ML Microservices..."

$ROOT = $PSScriptRoot

Start-Process -FilePath $python_exe -ArgumentList "-m uvicorn phase1.service:app --host 127.0.0.1 --port 8001" -WorkingDirectory $ROOT -WindowStyle Hidden
Start-Process -FilePath $python_exe -ArgumentList "-m uvicorn phase2.service:app --host 127.0.0.1 --port 8002" -WorkingDirectory $ROOT -WindowStyle Hidden
Start-Process -FilePath $python_exe -ArgumentList "-m uvicorn graph_transformer.service:app --host 127.0.0.1 --port 8003" -WorkingDirectory $ROOT -WindowStyle Hidden
Start-Process -FilePath $python_exe -ArgumentList "-m uvicorn rl.service:app --host 127.0.0.1 --port 8004" -WorkingDirectory $ROOT -WindowStyle Hidden

Write-Host "All 4 Phase services started on ports 8001, 8002, 8003, and 8004 in the background."
