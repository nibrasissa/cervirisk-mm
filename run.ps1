# CerviRisk-MM PowerShell runner.
# Usage: .\run.ps1 <command>
# See: .\run.ps1 help (or no argument)

param([Parameter(Position=0)][string]$Command="help")

function Show-Help {
    Write-Host ""
    Write-Host "CerviRisk-MM runner" -ForegroundColor Cyan
    Write-Host "===================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\run.ps1 <command>"
    Write-Host ""
    Write-Host "Setup:" -ForegroundColor Yellow
    Write-Host "  install     pip install all dependencies"
    Write-Host "  verify      check all four data sources are reachable"
    Write-Host ""
    Write-Host "Pipeline:" -ForegroundColor Yellow
    Write-Host "  ingest      run all four data ingestion modules"
    Write-Host "  assemble    build the assembled patient table"
    Write-Host "  fit         quick training (~30s, for testing)"
    Write-Host "  train       full nested LOOCV + tuning (~60min, production)"
    Write-Host "  baseline    capture drift detection baseline"
    Write-Host ""
    Write-Host "Operate:" -ForegroundColor Yellow
    Write-Host "  test        run pytest suite"
    Write-Host "  serve       start FastAPI locally on port 8000"
    Write-Host "  ui          start Streamlit frontend on port 8501"
    Write-Host "  docker      build + run via docker-compose"
    Write-Host ""
    Write-Host "Demo:" -ForegroundColor Yellow
    Write-Host "  predict     end-to-end prediction demo on a sample patient"
    Write-Host "  drift       simulated post-vaccination drift demo"
    Write-Host ""
    Write-Host "Inspect:" -ForegroundColor Yellow
    Write-Host "  status      show what files exist + recommended next step"
    Write-Host ""
    Write-Host "Shortcuts:" -ForegroundColor Yellow
    Write-Host "  quickstart  install + assemble + fit + baseline + serve (~5 min)"
    Write-Host "  help        show this help"
    Write-Host ""
}

function Invoke-Quickstart {
    Write-Host ""
    Write-Host "CerviRisk-MM quickstart" -ForegroundColor Cyan
    Write-Host "=======================" -ForegroundColor Cyan
    Write-Host "End-to-end setup for reviewers. ~5 minutes."
    Write-Host ""

    Write-Host "[1/5] Installing dependencies..." -ForegroundColor Yellow
    pip install -q -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Write-Host "install failed"; return }

    Write-Host "[2/5] Checking data sources..." -ForegroundColor Yellow
    if (-not (Test-Path "data\raw\uci_cervical_cancer.parquet")) {
        Write-Host "      Ingesting UCI, 1000G, NCBI, PGS..."
        python -m src.ingestion
        if ($LASTEXITCODE -ne 0) { Write-Host "ingestion failed"; return }
    } else {
        Write-Host "      Data already on disk, skipping ingestion."
    }

    Write-Host "[3/5] Assembling patients..." -ForegroundColor Yellow
    python -m src.features.build_patient
    if ($LASTEXITCODE -ne 0) { Write-Host "assembly failed"; return }

    Write-Host "[4/5] Quick training (logreg, ~30s)..." -ForegroundColor Yellow
    python -m src.model.train --eval split --models logreg
    if ($LASTEXITCODE -ne 0) { Write-Host "training failed"; return }

    Write-Host "[5/5] Capturing drift baseline..." -ForegroundColor Yellow
    python -m src.drift.baseline
    if ($LASTEXITCODE -ne 0) { Write-Host "baseline failed"; return }

    Write-Host ""
    Write-Host "Quickstart complete." -ForegroundColor Green
    Write-Host "Now run:  .\run.ps1 serve     to boot the API on port 8000"
    Write-Host "      or  .\run.ps1 test      to run the full test suite"
    Write-Host "      or  .\run.ps1 drift     to see the drift detection demo"
    Write-Host ""
}

switch ($Command) {
    "install"     { pip install -r requirements.txt }
    "verify"      { python verify_data_sources.py }

    "ingest"      { python -m src.ingestion }
    "assemble"    { python -m src.features.build_patient }
    "fit"         { python -m src.model.train --eval split --models logreg }
    "train"       { python -m src.model.train --tune }
    "baseline"    { python -m src.drift.baseline }

    "test"        { python -m pytest tests/ -v }
    "serve"       { uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload }
    "docker"      { docker compose up --build }
    "ui"          { streamlit run frontend/app.py }

    "predict"     { python -m scripts.demo_predict }
    "drift"       { python -m scripts.demo_drift }

    "status"      { python -m scripts.status }

    "quickstart"  { Invoke-Quickstart }

    default       { Show-Help }
}
