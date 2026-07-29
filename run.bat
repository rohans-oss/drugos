@echo off
title DrugOS - Autonomous Drug Repurposing Platform
echo ===================================================================
echo   DrugOS Platform Startup
echo   Starting Embedded Database + All 4 ML Microservices + Next.js UI
echo ===================================================================
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0start-all.ps1"
pause
