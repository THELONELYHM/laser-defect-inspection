param(
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VirtualEnvironment = Join-Path $ProjectRoot '.venv'

if (-not (Test-Path -LiteralPath $VirtualEnvironment)) {
    python -m venv $VirtualEnvironment
}

$Python = Join-Path $VirtualEnvironment 'Scripts\python.exe'
if (-not $SkipInstall) {
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')
}

Write-Host 'Environment is ready.'
Write-Host "Python: $Python"
Write-Host "Activate: $VirtualEnvironment\Scripts\Activate.ps1"

