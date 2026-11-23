# Starts the whole demo: FastAPI backend (new window) + Streamlit dashboard (this window).
# Usage, from the project folder:   .\start_demo.ps1
# If PowerShell blocks scripts, run:  powershell -ExecutionPolicy Bypass -File .\start_demo.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Virtual environment not found at $py (see README > Setup)." }
$api = "http://127.0.0.1:8000"

function Test-Backend {
    try { return (Invoke-WebRequest "$api/health" -UseBasicParsing -TimeoutSec 5).StatusCode -eq 200 } catch { return $false }
}

if (Test-Backend) {
    Write-Host "Backend already running at $api"
} else {
    Write-Host "Starting backend in a new window ($api) ..."
    Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "backend.main:app", "--port", "8000" -WorkingDirectory $PSScriptRoot
    $deadline = (Get-Date).AddSeconds(120)
    while (-not (Test-Backend)) {
        if ((Get-Date) -gt $deadline) { throw "Backend did not start within 120 s - check the backend window for errors." }
        Start-Sleep -Seconds 1
    }
    Write-Host "Backend is up (API docs: $api/docs)"
}

Write-Host "Starting dashboard (http://localhost:8501) - press Ctrl+C here to stop it; close the backend window to stop the API."
& $py -m streamlit run dashboard/app.py
