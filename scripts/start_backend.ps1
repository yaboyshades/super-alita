Param(
  [Parameter(Mandatory=$false)][int]$Port = 8010
)

Write-Host "[backend] Starting FastAPI on :$Port"

# Move to repo root
Set-Location -Path (Join-Path $PSScriptRoot "..")

# Prefer uvicorn with explicit host/port to avoid port 8080 conflicts
python -m uvicorn app:app --host 127.0.0.1 --port $Port

