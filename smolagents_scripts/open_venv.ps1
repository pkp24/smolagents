$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot
$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvActivate))
{
    Write-Host "Virtual environment not found. Running setup..." -ForegroundColor Yellow
    & pwsh -File (Join-Path $PSScriptRoot "setup_venv.ps1")
}

Write-Host "Opening a new PowerShell with venv activated..." -ForegroundColor Green
$cmd = "& '$venvActivate'; Write-Host 'VENV ACTIVE:' (Get-Command python).Source; Write-Host 'Tip: run python smolagents_scripts/webbrowser.py'"
Start-Process pwsh -ArgumentList @('-NoExit', '-Command', $cmd)

