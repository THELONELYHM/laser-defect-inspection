param(
    [string]$OpenCVDir = $env:OpenCV_DIR,
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourceDir = Join-Path $ProjectRoot 'cpp'
$BuildDir = Join-Path $SourceDir 'build'

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw 'CMake was not found. Install CMake and reopen the terminal.'
}

$ConfigureArgs = @('-S', $SourceDir, '-B', $BuildDir)
if ($OpenCVDir) {
    $ConfigureArgs += "-DOpenCV_DIR=$OpenCVDir"
}

& cmake @ConfigureArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& cmake --build $BuildDir --config $Configuration
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Build finished: $BuildDir"

