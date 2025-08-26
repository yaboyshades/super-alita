#!/usr/bin/env pwsh
<#!
.SYNOPSIS
    Helper script to register new MCP tool servers

.DESCRIPTION
    Adds a new server entry to the VS Code `.vscode/mcp.json` file.

.PARAMETER AddTool
    Name of the tool server to register

.PARAMETER ConfigPath
    Optional path to the mcp.json file (defaults to `.vscode/mcp.json`)
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$AddTool,
    [string]$ConfigPath = ".vscode/mcp.json"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ConfigPath)) {
    Write-Error "mcp config not found at $ConfigPath"
    exit 1
}

$config = Get-Content $ConfigPath -Raw | ConvertFrom-Json

if (-not $config.servers) {
    $config | Add-Member -MemberType NoteProperty -Name servers -Value @{}
}

if ($config.servers.$AddTool) {
    Write-Host "MCP server '$AddTool' already exists" -ForegroundColor Yellow
    exit 0
}

$serverEntry = @{
    type    = "stdio"
    command = '${workspaceFolder}\.venv\Scripts\python.exe'
    args    = @(
        '${workspaceFolder}\mcp_server_wrapper.py',
        '--tool',
        $AddTool
    )
    env     = @{
        GEMINI_API_KEY     = '${env:GEMINI_API_KEY}'
        MCP_AGENT_API_KEY = '${env:GEMINI_API_KEY}'
        PYTHONPATH        = ''
    }
    cwd     = '${workspaceFolder}'
}

$config.servers | Add-Member -NotePropertyName $AddTool -NotePropertyValue $serverEntry
$config | ConvertTo-Json -Depth 5 | Set-Content -Path $ConfigPath

Write-Host "Added MCP server '$AddTool' to $ConfigPath" -ForegroundColor Green

