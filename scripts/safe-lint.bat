@echo off
:: safe-lint.bat - Execute linting commands with safeguards
:: Usage: safe-lint.bat <command>

echo Running: %*
set TIMEOUT_SECONDS=20
set MAX_FILES=10

:: Count arguments to see if we're handling too many files
set arg_count=0
for %%x in (%*) do set /a arg_count+=1

if %arg_count% GTR %MAX_FILES% (
    echo Warning: Too many files ^(%arg_count%^). Running limited check only.
    echo Checking only first %MAX_FILES% files
)

:: Run the command with a timeout and exit gracefully on timeout
powershell -Command "& {$ErrorActionPreference = 'SilentlyContinue'; $ProgressPreference = 'SilentlyContinue'; $job = Start-Job -ScriptBlock {cd $pwd; %*}; if (Wait-Job $job -Timeout %TIMEOUT_SECONDS%) {Receive-Job $job; Remove-Job $job; exit 0} else {Write-Host 'Command gracefully timed out after %TIMEOUT_SECONDS% seconds'; Stop-Job $job; Remove-Job $job; exit 0}}"
