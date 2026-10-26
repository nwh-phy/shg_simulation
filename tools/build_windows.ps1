$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repoRoot
try {
    python -X utf8 -B tools/build_windows.py
    if ($LASTEXITCODE -ne 0) { throw 'Windows build failed. See the printed build log.' }
} finally {
    Pop-Location
}
