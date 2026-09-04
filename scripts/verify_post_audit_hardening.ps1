param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
Push-Location $ProjectRoot
try {
    & $Python -m compileall -q src scripts tests
    if ($LASTEXITCODE -ne 0) { throw 'Python syntax verification failed' }

    & $Python -m ruff format --check alembic src scripts tests
    if ($LASTEXITCODE -ne 0) { throw 'Ruff format verification failed' }

    & $Python -m ruff check alembic src scripts tests
    if ($LASTEXITCODE -ne 0) { throw 'Ruff lint verification failed' }

    & $Python -c "import torch, torch_geometric; print('Phase 3/4 optional ML dependencies: PASS')"
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch / PyTorch Geometric verification failed' }

    & $Python -m mypy src/razortrust
    if ($LASTEXITCODE -ne 0) { throw 'mypy verification failed' }

    & $Python -m pytest
    if ($LASTEXITCODE -ne 0) { throw 'pytest verification failed' }

    docker run --rm -v "${ProjectRoot}/policies:/policies:ro" `
        openpolicyagent/opa:1.19.1-static test /policies -v
    if ($LASTEXITCODE -ne 0) { throw 'OPA policy verification failed' }

    docker compose -f docker-compose.staging.yml config --quiet
    if ($LASTEXITCODE -ne 0) { throw 'staging Compose validation failed' }

    & $Python scripts/run_docker_e2e.py
    if ($LASTEXITCODE -ne 0) { throw 'Docker E2E verification failed' }

    Write-Host 'POST-AUDIT HARDENING VERIFICATION PASSED' -ForegroundColor Green
}
finally {
    Pop-Location
}
