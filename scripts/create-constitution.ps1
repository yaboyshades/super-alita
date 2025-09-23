param(
    [Parameter(Mandatory=$true)]
    [string]$PrinciplesPrompt,
    [string]$ProjectPath = (Get-Location)
)

function New-Constitution {
    param($Prompt, $Path)
    $MemoryPath = Join-Path $Path "memory"
    if (!(Test-Path $MemoryPath)) {
        New-Item -ItemType Directory -Path $MemoryPath -Force
    }
    $ConstitutionPath = Join-Path $MemoryPath "constitution.md"
    $Template = Get-Content "templates/constitution-template.md" -Raw
    # Substitute AI prompt here (placeholder)
    $Constitution = $Template -replace '\[.*?\]', $Prompt
    Set-Content -Path $ConstitutionPath -Value $Constitution
    $ChecklistPath = Join-Path $MemoryPath "constitution_update_checklist.md"
    Copy-Item "templates/constitution-checklist-template.md" $ChecklistPath -Force
    git add $MemoryPath
    git commit -m "feat: establish project constitution - $($Prompt.Substring(0,50))..."
    Write-Host "✅ Constitution created at $ConstitutionPath"
}

New-Constitution -Prompt $PrinciplesPrompt -Path $ProjectPath
