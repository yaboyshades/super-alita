# Mangle Configuration Script for Super Alita
# ==========================================
#
# This script configures environment variables and settings for Mangle integration

# Set Mangle binary path (using our mock implementation)
$env:MANGLE_BIN_PATH = "$env:TEMP\mangle.bat" 

# Create mangle data directory if it doesn't exist
$mangleDataDir = ".\data\mangle"
if (-not (Test-Path $mangleDataDir)) {
    New-Item -Path $mangleDataDir -ItemType Directory -Force | Out-Null
    Write-Host "Created Mangle data directory at: $mangleDataDir"
}

# Set auto-discovery for abilities
$env:ALITA_AUTO_DISCOVER_ABILITIES = "on"

Write-Host "✅ Mangle environment configured successfully!"
Write-Host "- MANGLE_BIN_PATH: $env:MANGLE_BIN_PATH"
Write-Host "- Data directory: $mangleDataDir"
Write-Host "- Auto-discover abilities: $env:ALITA_AUTO_DISCOVER_ABILITIES"
