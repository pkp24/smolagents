$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot
Write-Host "Project root: $root"

if (-not (Test-Path "$root\.venv"))
{
    Write-Host "Creating virtual environment..."
    & python -m venv "$root\.venv"
}

$venvPy = Join-Path $root ".venv\Scripts\python.exe"

Write-Host "Upgrading pip/setuptools/wheel..."
& $venvPy -m pip install --upgrade pip setuptools wheel

Write-Host "Installing runtime dependencies..."
& $venvPy -m pip install helium selenium requests beautifulsoup4 pillow python-dotenv numpy

Write-Host "Installing smolagents (editable)..."
Push-Location $root
& $venvPy -m pip install -e .
Pop-Location

Write-Host "Setup complete. To open a shell with this venv activated, run:`n  pwsh -File .\smolagents_scripts\open_venv.ps1" -ForegroundColor Green

