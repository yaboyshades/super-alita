Param(
  [Parameter(Mandatory=$false)][int]$Port = 8790,
  [Parameter(Mandatory=$false)][string]$BackendUrl = "http://127.0.0.1:8010",
  [Parameter(Mandatory=$false)][switch]$Dev
)

Write-Host "[agent] Starting with PORT=$Port, SUPER_ALITA_HTTP=$BackendUrl"

# Move to the extension folder
$extPath = Join-Path $PSScriptRoot "..\extensions\copilot-agent"
Set-Location -Path $extPath

# Environment for current PowerShell session
$env:PORT = "$Port"
$env:SUPER_ALITA_HTTP = $BackendUrl
$env:CORTEX_AGENT_DIRECT_RUFF = "0"

# Build if needed
if (-not (Test-Path ".\dist\server.js")) {
  Write-Host "[agent] dist/server.js not found, building..."
  npm run build
}

if ($Dev) {
  Write-Host "[agent] Running in dev mode (tsx)..."
  npm run dev
} else {
  Write-Host "[agent] Running server..."
  node .\dist\server.js
}
