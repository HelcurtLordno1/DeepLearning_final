$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

# Windows Python creates Scripts/python.exe, not a Linux environment.
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Windows environment." }
}
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

# uv can reuse the Windows package cache; pip is the fallback.
if (Get-Command uv.exe -ErrorAction SilentlyContinue) {
    uv.exe pip install --python $python --link-mode copy -r environment\requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw "CUDA package installation failed." }
    uv.exe pip install --python $python --link-mode copy -r environment\requirements.txt
} else {
    & $python -m pip install -r environment\requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw "CUDA package installation failed." }
    & $python -m pip install -r environment\requirements.txt
}
if ($LASTEXITCODE -ne 0) { throw "Package installation failed." }
& $python scripts\check_environment.py
if ($LASTEXITCODE -ne 0) { throw "Environment verification failed." }
& $python -m ipykernel install --user --name final-project-windows --display-name "Final Project (Windows .venv)"
if ($LASTEXITCODE -ne 0) { throw "Notebook kernel installation failed." }
Write-Host "Ready. Activate with: .\.venv\Scripts\Activate.ps1"
