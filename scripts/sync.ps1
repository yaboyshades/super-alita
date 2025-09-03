Param(
  [string]$MainBranch = "master",
  [string]$Upstream = "upstream",
  [switch]$DryRun
)

Write-Host "== Super Alita Repo Sync (PowerShell) =="

function Exec($cmd) {
  Write-Host "--> $cmd" -ForegroundColor Cyan
  if ($DryRun) { return }
  Invoke-Expression $cmd
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $cmd" }
}

if ($DryRun) { Write-Host "(DryRun mode)" -ForegroundColor Yellow }

Exec "git fetch origin --prune"
try { Exec "git fetch $Upstream --prune" } catch { Write-Host "No upstream remote or fetch failed (ok)." -ForegroundColor Yellow }

Exec "git checkout $MainBranch"
Exec "git pull --ff-only origin $MainBranch"

try {
  git rev-parse "$Upstream/$MainBranch" *> $null
  if ($LASTEXITCODE -eq 0) {
    Write-Host "Merging upstream/$MainBranch..." -ForegroundColor Green
    if ($DryRun) { Write-Host "(DryRun) would attempt fast-forward/merge" }
    else {
      git merge --ff-only "$Upstream/$MainBranch" 2>$null
      if ($LASTEXITCODE -ne 0) {
        git merge --no-edit "$Upstream/$MainBranch" || Write-Host "Manual conflict resolution required" -ForegroundColor Red
      }
    }
  }
} catch {}

if (-not $DryRun) {
  if ((git diff --name-only).Length -gt 0) {
    git add -A
    git commit -m "chore(sync): manual sync $(Get-Date -Format o)" || Write-Host "Nothing to commit" -ForegroundColor Yellow
    git push origin $MainBranch || Write-Host "Push failed" -ForegroundColor Red
  } else { Write-Host "No changes after sync." }
}

Write-Host "Sync complete." -ForegroundColor Green
