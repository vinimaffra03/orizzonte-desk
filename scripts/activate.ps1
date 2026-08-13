$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:ORIZZONTE_HOME = $ProjectRoot
$env:UV_CACHE_DIR = Join-Path $ProjectRoot ".cache\uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot ".runtime\python"
$env:TEMP = Join-Path $ProjectRoot ".tmp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR, $env:UV_PYTHON_INSTALL_DIR, $env:TEMP | Out-Null
$ManagedPython = Get-ChildItem -Path $env:UV_PYTHON_INSTALL_DIR -Directory -Filter 'cpython-3.11*-windows-*' -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName 'python.exe') } |
    Sort-Object Name -Descending |
    Select-Object -First 1
if ($ManagedPython) {
    $env:UV_PYTHON = Join-Path $ManagedPython.FullName 'python.exe'
}
Write-Host "Orizzonte Desk: ambiente apontado para $ProjectRoot" -ForegroundColor Cyan
