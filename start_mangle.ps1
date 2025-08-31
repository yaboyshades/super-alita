# Super Alita + Mangle Integration Startup Script for Windows
# This script initializes and starts the Super Alita server with Mangle integration

# Display banner
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Super Alita + Mangle Integration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Go is installed (needed for Mangle)
try {
    $goVersion = (go version)
    Write-Host "✅ Found Go installation" -ForegroundColor Green
    Write-Host "   $goVersion" -ForegroundColor Gray
}
catch {
    Write-Host "❌ Go not found. Please install Go first:" -ForegroundColor Red
    Write-Host "   https://golang.org/doc/install" -ForegroundColor Yellow
    exit 1
}

# For demonstration purposes, since Mangle is hard to install on Windows
# we'll create a mock mangle executable script
Write-Host "⚠️ Creating a mock Mangle executable for demonstration" -ForegroundColor Yellow
$mockMangleContent = @"
@echo off
echo [{"Name": "log4j", "Version": "2.14.0"}]
"@

$mockManglePath = Join-Path -Path $env:TEMP -ChildPath "mangle.bat"
Set-Content -Path $mockManglePath -Value $mockMangleContent

Write-Host "✅ Created mock Mangle executable for demonstration" -ForegroundColor Green
Write-Host "   Path: $mockManglePath" -ForegroundColor Gray
$manglePath = $mockManglePath

# Set environment variable
$env:MANGLE_BIN_PATH = $manglePath
Write-Host "✅ Set MANGLE_BIN_PATH environment variable" -ForegroundColor Green

# Check if .env file exists, create if not
if (-not (Test-Path ".env")) {
    Write-Host "⚠️ No .env file found, creating from .env.example" -ForegroundColor Yellow
    try {
        Copy-Item .env.example .env
        Write-Host "✅ Created .env file" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to create .env file" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "✅ Found existing .env file" -ForegroundColor Green
}

# Check Python environment and dependencies
try {
    $pythonVersion = (python --version)
    Write-Host "✅ Found Python installation" -ForegroundColor Green
    Write-Host "   $pythonVersion" -ForegroundColor Gray

    # Ensure virtual environment is activated
    if (-not $env:VIRTUAL_ENV) {
        if (Test-Path ".venv\Scripts\activate.ps1") {
            Write-Host "⚠️ Activating virtual environment..." -ForegroundColor Yellow
            & .\.venv\Scripts\activate.ps1
            Write-Host "✅ Virtual environment activated" -ForegroundColor Green
        }
        else {
            Write-Host "⚠️ No virtual environment found. Creating one..." -ForegroundColor Yellow
            python -m venv .venv
            & .\.venv\Scripts\activate.ps1
            Write-Host "✅ Virtual environment created and activated" -ForegroundColor Green
        }
    }
    else {
        Write-Host "✅ Virtual environment already activated" -ForegroundColor Green
    }

    # Check dependencies
    Write-Host "🔍 Checking dependencies..." -ForegroundColor Blue
    $fastApiInstalled = $false
    try {
        $null = python -c "import fastapi"
        $fastApiInstalled = $true
    }
    catch {
        $fastApiInstalled = $false
    }

    if (-not $fastApiInstalled) {
        Write-Host "⚠️ Installing dependencies..." -ForegroundColor Yellow
        python -m pip install -r requirements.txt -r requirements-test.txt
        Write-Host "✅ Dependencies installed" -ForegroundColor Green
    }
    else {
        Write-Host "✅ Dependencies already installed" -ForegroundColor Green
    }
}
catch {
    Write-Host "❌ Error with Python environment: $_" -ForegroundColor Red
    exit 1
}

# Start Super Alita server
Write-Host ""
Write-Host "🚀 Starting Super Alita server with Mangle integration..." -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
python -c "import uvicorn; from app import app; uvicorn.run(app, host='127.0.0.1', port=8080, reload=True)"
