#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup script for Super Alita Guardian VS Code Extension

.DESCRIPTION
    Automates the installation and setup of the Super Alita Guardian extension
    for VS Code Agent Mode integration.

.PARAMETER Mode
    Setup mode: 'install', 'dev', 'package', or 'clean'

.EXAMPLE
    .\setup-guardian-extension.ps1 -Mode install
    .\setup-guardian-extension.ps1 -Mode dev
#>

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('install', 'dev', 'package', 'clean')]
    [string]$Mode
)

$ErrorActionPreference = "Stop"
$ExtensionPath = ".vscode\extensions\super-alita-guardian"
$WorkspaceRoot = $PWD

Write-Host "🛡️ Super Alita Guardian Extension Setup" -ForegroundColor Cyan
Write-Host "Mode: $Mode" -ForegroundColor Yellow
Write-Host "Extension Path: $ExtensionPath" -ForegroundColor Gray
Write-Host ""

function Test-Prerequisites {
    Write-Host "🔍 Checking prerequisites..." -ForegroundColor Blue
    
    # Check Node.js
    try {
        $nodeVersion = node --version
        Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
    }
    catch {
        Write-Error "❌ Node.js not found. Please install Node.js 18+ first."
        exit 1
    }
    
    # Check npm
    try {
        $npmVersion = npm --version
        Write-Host "✅ npm: $npmVersion" -ForegroundColor Green
    }
    catch {
        Write-Error "❌ npm not found."
        exit 1
    }
    
    # Check VS Code
    try {
        $codeVersion = code --version | Select-Object -First 1
        Write-Host "✅ VS Code: $codeVersion" -ForegroundColor Green
    }
    catch {
        Write-Warning "⚠️ VS Code CLI not found. Extension can still be built."
    }
    
    # Check if vsce is available
    try {
        vsce --version | Out-Null
        Write-Host "✅ vsce (VS Code Extension Manager) found" -ForegroundColor Green
    }
    catch {
        Write-Host "📦 Installing vsce..." -ForegroundColor Yellow
        npm install -g vsce
    }
}

function Install-Dependencies {
    Write-Host "📦 Installing extension dependencies..." -ForegroundColor Blue
    
    Push-Location $ExtensionPath
    
    try {
        # Install dependencies
        npm install
        Write-Host "✅ Dependencies installed" -ForegroundColor Green
        
        # Install dev dependencies
        npm install --save-dev @types/vscode@^1.90.0 @types/node@^20.0.0 typescript@^5.0.0
        Write-Host "✅ Dev dependencies installed" -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
}

function Build-Extension {
    Write-Host "🔧 Building extension..." -ForegroundColor Blue
    
    Push-Location $ExtensionPath
    
    try {
        # Compile TypeScript
        npx tsc -p ./
        Write-Host "✅ TypeScript compiled successfully" -ForegroundColor Green
        
        # Verify output
        if (Test-Path "out\extension.js") {
            Write-Host "✅ Extension compiled to out\extension.js" -ForegroundColor Green
        } else {
            Write-Error "❌ Compilation failed - output not found"
        }
    }
    finally {
        Pop-Location
    }
}

function Package-Extension {
    Write-Host "📦 Packaging extension..." -ForegroundColor Blue
    
    Push-Location $ExtensionPath
    
    try {
        # Package extension
        vsce package
        
        $vsixFile = Get-ChildItem -Filter "*.vsix" | Select-Object -First 1
        if ($vsixFile) {
            Write-Host "✅ Extension packaged: $($vsixFile.Name)" -ForegroundColor Green
            return $vsixFile.FullName
        } else {
            Write-Error "❌ Packaging failed - VSIX file not found"
        }
    }
    finally {
        Pop-Location
    }
}

function Install-Extension {
    param([string]$VsixPath)
    
    Write-Host "🚀 Installing extension in VS Code..." -ForegroundColor Blue
    
    try {
        if ($VsixPath) {
            code --install-extension $VsixPath
        } else {
            Write-Host "📂 Installing from source..." -ForegroundColor Yellow
            code --install-extension $ExtensionPath
        }
        
        Write-Host "✅ Extension installed successfully" -ForegroundColor Green
        Write-Host "💡 Restart VS Code to activate the extension" -ForegroundColor Cyan
    }
    catch {
        Write-Error "❌ Failed to install extension: $_"
    }
}

function Start-DevMode {
    Write-Host "🔧 Starting development mode..." -ForegroundColor Blue
    
    Push-Location $ExtensionPath
    
    try {
        Write-Host "📝 Starting TypeScript watch mode..." -ForegroundColor Yellow
        Write-Host "💡 Press F5 in VS Code to launch Extension Development Host" -ForegroundColor Cyan
        Write-Host "💡 Press Ctrl+C to stop watch mode" -ForegroundColor Cyan
        
        npx tsc -watch -p ./
    }
    finally {
        Pop-Location
    }
}

function Clean-Extension {
    Write-Host "🧹 Cleaning extension build artifacts..." -ForegroundColor Blue
    
    Push-Location $ExtensionPath
    
    try {
        # Remove build outputs
        if (Test-Path "out") {
            Remove-Item -Recurse -Force "out"
            Write-Host "✅ Removed out directory" -ForegroundColor Green
        }
        
        # Remove VSIX files
        Get-ChildItem -Filter "*.vsix" | Remove-Item -Force
        Write-Host "✅ Removed VSIX files" -ForegroundColor Green
        
        # Remove node_modules if requested
        $response = Read-Host "Remove node_modules? (y/N)"
        if ($response -eq 'y' -or $response -eq 'Y') {
            if (Test-Path "node_modules") {
                Remove-Item -Recurse -Force "node_modules"
                Write-Host "✅ Removed node_modules" -ForegroundColor Green
            }
        }
    }
    finally {
        Pop-Location
    }
}

function Show-Usage {
    Write-Host ""
    Write-Host "🛡️ Super Alita Guardian Extension Ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage in VS Code:" -ForegroundColor Cyan
    Write-Host "  • Type '@alita' in chat for architectural guidance" -ForegroundColor White
    Write-Host "  • Use Command Palette: 'Super Alita: Show Telemetry Dashboard'" -ForegroundColor White
    Write-Host "  • Click '🛡️ Alita Guardian' in status bar for quick access" -ForegroundColor White
    Write-Host ""
    Write-Host "Available modes:" -ForegroundColor Cyan
    Write-Host "  • @alita review this code    (Guardian mode)" -ForegroundColor White
    Write-Host "  • @alita audit workspace     (Audit mode)" -ForegroundColor White
    Write-Host "  • @alita refactor this code  (Refactor mode)" -ForegroundColor White
    Write-Host "  • @alita generate plugin     (Generator mode)" -ForegroundColor White
    Write-Host ""
}

# Main execution
try {
    Test-Prerequisites
    
    switch ($Mode) {
        'install' {
            Install-Dependencies
            Build-Extension
            $vsixPath = Package-Extension
            Install-Extension -VsixPath $vsixPath
            Show-Usage
        }
        'dev' {
            Install-Dependencies
            Build-Extension
            Start-DevMode
        }
        'package' {
            Install-Dependencies
            Build-Extension
            Package-Extension
        }
        'clean' {
            Clean-Extension
        }
    }
    
    Write-Host ""
    Write-Host "🎉 Setup completed successfully!" -ForegroundColor Green
}
catch {
    Write-Host ""
    Write-Error "❌ Setup failed: $_"
    exit 1
}