$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
. (Join-Path $PSScriptRoot 'activate.ps1')

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv não foi encontrado. Instale em https://docs.astral.sh/uv/ e execute novamente.'
}

uv python install 3.11 --install-dir $env:UV_PYTHON_INSTALL_DIR
$ManagedPython = Get-ChildItem -Path $env:UV_PYTHON_INSTALL_DIR -Directory -Filter 'cpython-3.11*-windows-*' |
    Where-Object { Test-Path (Join-Path $_.FullName 'python.exe') } |
    Sort-Object Name -Descending |
    Select-Object -First 1
if (-not $ManagedPython) {
    throw 'Python 3.11 gerenciado não foi instalado no disco D:.'
}
$env:UV_PYTHON = Join-Path $ManagedPython.FullName 'python.exe'
uv sync --frozen --all-groups --python $env:UV_PYTHON
uv run orizzonte init
uv run orizzonte doctor
