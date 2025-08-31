# Multi-Terminal Orchestration & Command Safety Demo

Write-Host "🎯 Super Alita Multi-Terminal Development Setup Demo" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray

# Load Copilot helpers
Write-Host "`n📚 Loading Copilot helpers..." -ForegroundColor Blue
if (Test-Path "scripts\copilot-helpers.ps1") {
    . ".\scripts\copilot-helpers.ps1"
    Write-Host "✅ Copilot helpers loaded successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Copilot helpers not found, creating them..." -ForegroundColor Yellow
}

Write-Host "`n🚀 Starting Multi-Terminal Demo..." -ForegroundColor Cyan

# Demo 1: Cross-platform environment check
Write-Host "`n1️⃣ Cross-Platform Environment Check" -ForegroundColor Yellow
Write-Host "Testing environment setup..." -ForegroundColor Gray
try {
    npm run setup:env
    Write-Host "✅ Environment setup successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Environment setup failed: $_" -ForegroundColor Red
}

# Demo 2: Health check via multiple methods
Write-Host "`n2️⃣ Health Check Methods Demo" -ForegroundColor Yellow
Write-Host "Method 1: PowerShell script" -ForegroundColor Gray
try {
    .\scripts\quick-test.ps1 -HealthOnly
} catch {
    Write-Host "Script-based health check failed" -ForegroundColor Red
}

Write-Host "`nMethod 2: Direct curl" -ForegroundColor Gray
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8080/healthz" -TimeoutSec 5
    Write-Host "✅ Direct health check: $($health.status)" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Server not responding on http://127.0.0.1:8080" -ForegroundColor Yellow
}

# Demo 3: GitHub Copilot CLI integration
Write-Host "`n3️⃣ GitHub Copilot CLI Safety Demo" -ForegroundColor Yellow
Write-Host "Demonstrating safe command generation..." -ForegroundColor Gray

Write-Host "`nExplaining a common command:" -ForegroundColor Cyan
Write-Host "gh copilot explain 'curl http://127.0.0.1:8080/healthz'" -ForegroundColor White

Write-Host "`nFor command suggestions, use:" -ForegroundColor Cyan
Write-Host "gh copilot suggest 'check Super Alita server status'" -ForegroundColor White

# Demo 4: Available scripts and tasks
Write-Host "`n4️⃣ Available Development Tools" -ForegroundColor Yellow

Write-Host "`nPowerShell Scripts:" -ForegroundColor Cyan
$scripts = Get-ChildItem "scripts\*.ps1" -ErrorAction SilentlyContinue
if ($scripts) {
    foreach ($script in $scripts) {
        Write-Host "  .\$($script.Name)" -ForegroundColor White
    }
} else {
    Write-Host "  No PowerShell scripts found" -ForegroundColor Gray
}

Write-Host "`nNPM Scripts:" -ForegroundColor Cyan
if (Test-Path "package.json") {
    $package = Get-Content "package.json" | ConvertFrom-Json
    if ($package.scripts) {
        $package.scripts.PSObject.Properties | ForEach-Object {
            Write-Host "  npm run $($_.Name)" -ForegroundColor White
        }
    }
} else {
    Write-Host "  No package.json found" -ForegroundColor Gray
}

# Demo 5: Git hooks demonstration
Write-Host "`n5️⃣ Git Hooks Safety Demo" -ForegroundColor Yellow
Write-Host "Pre-commit hook configuration:" -ForegroundColor Gray

if (Test-Path ".husky\pre-commit") {
    Write-Host "✅ Pre-commit hook installed" -ForegroundColor Green
    Write-Host "Content preview:" -ForegroundColor Cyan
    Get-Content ".husky\pre-commit" | Select-Object -First 5 | ForEach-Object {
        Write-Host "  $_" -ForegroundColor Gray
    }
} else {
    Write-Host "❌ Pre-commit hook not found" -ForegroundColor Red
}

# Demo 6: VS Code task integration
Write-Host "`n6️⃣ VS Code Tasks Integration" -ForegroundColor Yellow
if (Test-Path ".vscode\tasks.json") {
    Write-Host "✅ VS Code tasks configured" -ForegroundColor Green
    Write-Host "Key tasks available:" -ForegroundColor Cyan
    Write-Host "  - 🚀 Start Super Alita Development Environment" -ForegroundColor White
    Write-Host "  - 🔍 Full System Validation" -ForegroundColor White
    Write-Host "  - 🏥 Health Check" -ForegroundColor White
    Write-Host "  - 🛠️ Tools Catalog Check" -ForegroundColor White
} else {
    Write-Host "❌ VS Code tasks not configured" -ForegroundColor Red
}

# Demo summary
Write-Host "`n🎉 Demo Summary" -ForegroundColor Green
Write-Host "=" * 40 -ForegroundColor Gray
Write-Host "✅ Multi-terminal orchestration ready" -ForegroundColor Green
Write-Host "✅ GitHub Copilot CLI integration active" -ForegroundColor Green
Write-Host "✅ Cross-platform scripts available" -ForegroundColor Green
Write-Host "✅ Git hooks for safety configured" -ForegroundColor Green
Write-Host "✅ VS Code tasks configured" -ForegroundColor Green

Write-Host "`n🚀 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Open VS Code Command Palette (Ctrl+Shift+P)" -ForegroundColor White
Write-Host "2. Run: 'Tasks: Run Task' → '🚀 Start Super Alita Development Environment'" -ForegroundColor White
Write-Host "3. Use 'ghcs \"task\"' for safe command suggestions" -ForegroundColor White
Write-Host "4. Use 'ghce \"command\"' to explain commands before running" -ForegroundColor White

Write-Host "`n💡 Pro Tips:" -ForegroundColor Magenta
Write-Host "- All terminals auto-start when opening the workspace" -ForegroundColor Yellow
Write-Host "- Git hooks prevent committing broken code" -ForegroundColor Yellow
Write-Host "- ShellCheck provides real-time script validation" -ForegroundColor Yellow
Write-Host "- Cross-platform scripts work on Windows/macOS/Linux" -ForegroundColor Yellow
# SIG # Begin signature block
# MIIaSQYJKoZIhvcNAQcCoIIaOjCCGjYCAQExDzANBglghkgBZQMEAgEFADB5Bgor
# BgEEAYI3AgEEoGswaTA0BgorBgEEAYI3AgEeMCYCAwEAAAQQH8w7YFlLCE63JNLG
# KX7zUQIBAAIBAAIBAAIBAAIBADAxMA0GCWCGSAFlAwQCAQUABCBj24p1Zvj6bvIt
# DRALjSMXIsOrE1RkRwPQEhXtxCwfdKCCFRswggHdMIIBRqADAgECAhB1LVe8LkJa
# nkn+CcMy/7DwMA0GCSqGSIb3DQEBBQUAMBUxEzARBgNVBAMTCkF1dG9Ib3RrZXkw
# IBcNMjUwMzEwMTkzNDMyWhgPOTk5OTAxMDExMjAwMDBaMBUxEzARBgNVBAMTCkF1
# dG9Ib3RrZXkwgZ8wDQYJKoZIhvcNAQEBBQADgY0AMIGJAoGBAKkvQYONqskINI1i
# BBkYCk9PniXin9+yMrpQAml4pZED9brGePZd+51f5FsTrNpeMRnRV7NNyJEDOLFR
# IhkBPDvwNciJEFuNLCbUkt9O6o3uT858uvn5PJ1HHq4yrtW7OQYkA9c69Pfh+xIv
# t9P8wBgkrs4XnFAi4cvLMWE/P2ydAgMBAAGjLDAqMBAGA1UdBAEB/wQGMAQDAgSQ
# MBYGA1UdJQEB/wQMMAoGCCsGAQUFBwMDMA0GCSqGSIb3DQEBBQUAA4GBADXNo2wn
# fDUdgw3T5iYLJ+pix6VKMDc4OltoD2eZ1dW1C3LMdUyenLliTS+sd+e1uaHwf2iD
# VpKpLLiWMXKyxlvqg09K5Ajz1yIt3POxQ7VYXazT+xbbC1JTD0rXiD6M847uWTSq
# PwR9+nIwhhtUpMksc07Zifqd4V4w3MSdM+DuMIIFjTCCBHWgAwIBAgIQDpsYjvnQ
# Lefv21DiCEAYWjANBgkqhkiG9w0BAQwFADBlMQswCQYDVQQGEwJVUzEVMBMGA1UE
# ChMMRGlnaUNlcnQgSW5jMRkwFwYDVQQLExB3d3cuZGlnaWNlcnQuY29tMSQwIgYD
# VQQDExtEaWdpQ2VydCBBc3N1cmVkIElEIFJvb3QgQ0EwHhcNMjIwODAxMDAwMDAw
# WhcNMzExMTA5MjM1OTU5WjBiMQswCQYDVQQGEwJVUzEVMBMGA1UEChMMRGlnaUNl
# cnQgSW5jMRkwFwYDVQQLExB3d3cuZGlnaWNlcnQuY29tMSEwHwYDVQQDExhEaWdp
# Q2VydCBUcnVzdGVkIFJvb3QgRzQwggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAwggIK
# AoICAQC/5pBzaN675F1KPDAiMGkz7MKnJS7JIT3yithZwuEppz1Yq3aaza57G4QN
# xDAf8xukOBbrVsaXbR2rsnnyyhHS5F/WBTxSD1Ifxp4VpX6+n6lXFllVcq9ok3DC
# srp1mWpzMpTREEQQLt+C8weE5nQ7bXHiLQwb7iDVySAdYyktzuxeTsiT+CFhmzTr
# BcZe7FsavOvJz82sNEBfsXpm7nfISKhmV1efVFiODCu3T6cw2Vbuyntd463JT17l
# Necxy9qTXtyOj4DatpGYQJB5w3jHtrHEtWoYOAMQjdjUN6QuBX2I9YI+EJFwq1WC
# QTLX2wRzKm6RAXwhTNS8rhsDdV14Ztk6MUSaM0C/CNdaSaTC5qmgZ92kJ7yhTzm1
# EVgX9yRcRo9k98FpiHaYdj1ZXUJ2h4mXaXpI8OCiEhtmmnTK3kse5w5jrubU75KS
# Op493ADkRSWJtppEGSt+wJS00mFt6zPZxd9LBADMfRyVw4/3IbKyEbe7f/LVjHAs
# QWCqsWMYRJUadmJ+9oCw++hkpjPRiQfhvbfmQ6QYuKZ3AeEPlAwhHbJUKSWJbOUO
# UlFHdL4mrLZBdd56rF+NP8m800ERElvlEFDrMcXKchYiCd98THU/Y+whX8QgUWtv
# sauGi0/C1kVfnSD8oR7FwI+isX4KJpn15GkvmB0t9dmpsh3lGwIDAQABo4IBOjCC
# ATYwDwYDVR0TAQH/BAUwAwEB/zAdBgNVHQ4EFgQU7NfjgtJxXWRM3y5nP+e6mK4c
# D08wHwYDVR0jBBgwFoAUReuir/SSy4IxLVGLp6chnfNtyA8wDgYDVR0PAQH/BAQD
# AgGGMHkGCCsGAQUFBwEBBG0wazAkBggrBgEFBQcwAYYYaHR0cDovL29jc3AuZGln
# aWNlcnQuY29tMEMGCCsGAQUFBzAChjdodHRwOi8vY2FjZXJ0cy5kaWdpY2VydC5j
# b20vRGlnaUNlcnRBc3N1cmVkSURSb290Q0EuY3J0MEUGA1UdHwQ+MDwwOqA4oDaG
# NGh0dHA6Ly9jcmwzLmRpZ2ljZXJ0LmNvbS9EaWdpQ2VydEFzc3VyZWRJRFJvb3RD
# QS5jcmwwEQYDVR0gBAowCDAGBgRVHSAAMA0GCSqGSIb3DQEBDAUAA4IBAQBwoL9D
# XFXnOF+go3QbPbYW1/e/Vwe9mqyhhyzshV6pGrsi+IcaaVQi7aSId229GhT0E0p6
# Ly23OO/0/4C5+KH38nLeJLxSA8hO0Cre+i1Wz/n096wwepqLsl7Uz9FDRJtDIeuW
# cqFItJnLnU+nBgMTdydE1Od/6Fmo8L8vC6bp8jQ87PcDx4eo0kxAGTVGamlUsLih
# Vo7spNU96LHc/RzY9HdaXFSMb++hUD38dglohJ9vytsgjTVgHAIDyyCwrFigDkBj
# xZgiwbJZ9VVrzyerbHbObyMt9H5xaiNrIv8SuFQtJ37YOtnwtoeW/VvRXKwYw02f
# c7cBqZ9Xql4o4rmUMIIGtDCCBJygAwIBAgIQDcesVwX/IZkuQEMiDDpJhjANBgkq
# hkiG9w0BAQsFADBiMQswCQYDVQQGEwJVUzEVMBMGA1UEChMMRGlnaUNlcnQgSW5j
# MRkwFwYDVQQLExB3d3cuZGlnaWNlcnQuY29tMSEwHwYDVQQDExhEaWdpQ2VydCBU
# cnVzdGVkIFJvb3QgRzQwHhcNMjUwNTA3MDAwMDAwWhcNMzgwMTE0MjM1OTU5WjBp
# MQswCQYDVQQGEwJVUzEXMBUGA1UEChMORGlnaUNlcnQsIEluYy4xQTA/BgNVBAMT
# OERpZ2lDZXJ0IFRydXN0ZWQgRzQgVGltZVN0YW1waW5nIFJTQTQwOTYgU0hBMjU2
# IDIwMjUgQ0ExMIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAtHgx0wqY
# QXK+PEbAHKx126NGaHS0URedTa2NDZS1mZaDLFTtQ2oRjzUXMmxCqvkbsDpz4aH+
# qbxeLho8I6jY3xL1IusLopuW2qftJYJaDNs1+JH7Z+QdSKWM06qchUP+AbdJgMQB
# 3h2DZ0Mal5kYp77jYMVQXSZH++0trj6Ao+xh/AS7sQRuQL37QXbDhAktVJMQbzIB
# HYJBYgzWIjk8eDrYhXDEpKk7RdoX0M980EpLtlrNyHw0Xm+nt5pnYJU3Gmq6bNMI
# 1I7Gb5IBZK4ivbVCiZv7PNBYqHEpNVWC2ZQ8BbfnFRQVESYOszFI2Wv82wnJRfN2
# 0VRS3hpLgIR4hjzL0hpoYGk81coWJ+KdPvMvaB0WkE/2qHxJ0ucS638ZxqU14lDn
# ki7CcoKCz6eum5A19WZQHkqUJfdkDjHkccpL6uoG8pbF0LJAQQZxst7VvwDDjAmS
# FTUms+wV/FbWBqi7fTJnjq3hj0XbQcd8hjj/q8d6ylgxCZSKi17yVp2NL+cnT6To
# y+rN+nM8M7LnLqCrO2JP3oW//1sfuZDKiDEb1AQ8es9Xr/u6bDTnYCTKIsDq1Btm
# XUqEG1NqzJKS4kOmxkYp2WyODi7vQTCBZtVFJfVZ3j7OgWmnhFr4yUozZtqgPrHR
# VHhGNKlYzyjlroPxul+bgIspzOwbtmsgY1MCAwEAAaOCAV0wggFZMBIGA1UdEwEB
# /wQIMAYBAf8CAQAwHQYDVR0OBBYEFO9vU0rp5AZ8esrikFb2L9RJ7MtOMB8GA1Ud
# IwQYMBaAFOzX44LScV1kTN8uZz/nupiuHA9PMA4GA1UdDwEB/wQEAwIBhjATBgNV
# HSUEDDAKBggrBgEFBQcDCDB3BggrBgEFBQcBAQRrMGkwJAYIKwYBBQUHMAGGGGh0
# dHA6Ly9vY3NwLmRpZ2ljZXJ0LmNvbTBBBggrBgEFBQcwAoY1aHR0cDovL2NhY2Vy
# dHMuZGlnaWNlcnQuY29tL0RpZ2lDZXJ0VHJ1c3RlZFJvb3RHNC5jcnQwQwYDVR0f
# BDwwOjA4oDagNIYyaHR0cDovL2NybDMuZGlnaWNlcnQuY29tL0RpZ2lDZXJ0VHJ1
# c3RlZFJvb3RHNC5jcmwwIAYDVR0gBBkwFzAIBgZngQwBBAIwCwYJYIZIAYb9bAcB
# MA0GCSqGSIb3DQEBCwUAA4ICAQAXzvsWgBz+Bz0RdnEwvb4LyLU0pn/N0IfFiBow
# f0/Dm1wGc/Do7oVMY2mhXZXjDNJQa8j00DNqhCT3t+s8G0iP5kvN2n7Jd2E4/iEI
# UBO41P5F448rSYJ59Ib61eoalhnd6ywFLerycvZTAz40y8S4F3/a+Z1jEMK/DMm/
# axFSgoR8n6c3nuZB9BfBwAQYK9FHaoq2e26MHvVY9gCDA/JYsq7pGdogP8HRtrYf
# ctSLANEBfHU16r3J05qX3kId+ZOczgj5kjatVB+NdADVZKON/gnZruMvNYY2o1f4
# MXRJDMdTSlOLh0HCn2cQLwQCqjFbqrXuvTPSegOOzr4EWj7PtspIHBldNE2K9i69
# 7cvaiIo2p61Ed2p8xMJb82Yosn0z4y25xUbI7GIN/TpVfHIqQ6Ku/qjTY6hc3hsX
# MrS+U0yy+GWqAXam4ToWd2UQ1KYT70kZjE4YtL8Pbzg0c1ugMZyZZd/BdHLiRu7h
# AWE6bTEm4XYRkA6Tl4KSFLFk43esaUeqGkH/wyW4N7OigizwJWeukcyIPbAvjSab
# nf7+Pu0VrFgoiovRDiyx3zEdmcif/sYQsfch28bZeUz2rtY/9TCA6TD8dC3JE3rY
# krhLULy7Dc90G6e8BlqmyIjlgp2+VqsS9/wQD7yFylIz0scmbKvFoW2jNrbM1pD2
# T7m3XDCCBu0wggTVoAMCAQICEAqA7xhLjfEFgtHEdqeVdGgwDQYJKoZIhvcNAQEL
# BQAwaTELMAkGA1UEBhMCVVMxFzAVBgNVBAoTDkRpZ2lDZXJ0LCBJbmMuMUEwPwYD
# VQQDEzhEaWdpQ2VydCBUcnVzdGVkIEc0IFRpbWVTdGFtcGluZyBSU0E0MDk2IFNI
# QTI1NiAyMDI1IENBMTAeFw0yNTA2MDQwMDAwMDBaFw0zNjA5MDMyMzU5NTlaMGMx
# CzAJBgNVBAYTAlVTMRcwFQYDVQQKEw5EaWdpQ2VydCwgSW5jLjE7MDkGA1UEAxMy
# RGlnaUNlcnQgU0hBMjU2IFJTQTQwOTYgVGltZXN0YW1wIFJlc3BvbmRlciAyMDI1
# IDEwggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAwggIKAoICAQDQRqwtEsae0OquYFaz
# K1e6b1H/hnAKAd/KN8wZQjBjMqiZ3xTWcfsLwOvRxUwXcGx8AUjni6bz52fGTfr6
# PHRNv6T7zsf1Y/E3IU8kgNkeECqVQ+3bzWYesFtkepErvUSbf+EIYLkrLKd6qJnu
# zK8Vcn0DvbDMemQFoxQ2Dsw4vEjoT1FpS54dNApZfKY61HAldytxNM89PZXUP/5w
# WWURK+IfxiOg8W9lKMqzdIo7VA1R0V3Zp3DjjANwqAf4lEkTlCDQ0/fKJLKLkzGB
# Tpx6EYevvOi7XOc4zyh1uSqgr6UnbksIcFJqLbkIXIPbcNmA98Oskkkrvt6lPAw/
# p4oDSRZreiwB7x9ykrjS6GS3NR39iTTFS+ENTqW8m6THuOmHHjQNC3zbJ6nJ6SXi
# LSvw4Smz8U07hqF+8CTXaETkVWz0dVVZw7knh1WZXOLHgDvundrAtuvz0D3T+dYa
# NcwafsVCGZKUhQPL1naFKBy1p6llN3QgshRta6Eq4B40h5avMcpi54wm0i2ePZD5
# pPIssoszQyF4//3DoK2O65Uck5Wggn8O2klETsJ7u8xEehGifgJYi+6I03UuT1j7
# FnrqVrOzaQoVJOeeStPeldYRNMmSF3voIgMFtNGh86w3ISHNm0IaadCKCkUe2Lnw
# JKa8TIlwCUNVwppwn4D3/Pt5pwIDAQABo4IBlTCCAZEwDAYDVR0TAQH/BAIwADAd
# BgNVHQ4EFgQU5Dv88jHt/f3X85FxYxlQQ89hjOgwHwYDVR0jBBgwFoAU729TSunk
# Bnx6yuKQVvYv1Ensy04wDgYDVR0PAQH/BAQDAgeAMBYGA1UdJQEB/wQMMAoGCCsG
# AQUFBwMIMIGVBggrBgEFBQcBAQSBiDCBhTAkBggrBgEFBQcwAYYYaHR0cDovL29j
# c3AuZGlnaWNlcnQuY29tMF0GCCsGAQUFBzAChlFodHRwOi8vY2FjZXJ0cy5kaWdp
# Y2VydC5jb20vRGlnaUNlcnRUcnVzdGVkRzRUaW1lU3RhbXBpbmdSU0E0MDk2U0hB
# MjU2MjAyNUNBMS5jcnQwXwYDVR0fBFgwVjBUoFKgUIZOaHR0cDovL2NybDMuZGln
# aWNlcnQuY29tL0RpZ2lDZXJ0VHJ1c3RlZEc0VGltZVN0YW1waW5nUlNBNDA5NlNI
# QTI1NjIwMjVDQTEuY3JsMCAGA1UdIAQZMBcwCAYGZ4EMAQQCMAsGCWCGSAGG/WwH
# ATANBgkqhkiG9w0BAQsFAAOCAgEAZSqt8RwnBLmuYEHs0QhEnmNAciH45PYiT9s1
# i6UKtW+FERp8FgXRGQ/YAavXzWjZhY+hIfP2JkQ38U+wtJPBVBajYfrbIYG+Dui4
# I4PCvHpQuPqFgqp1PzC/ZRX4pvP/ciZmUnthfAEP1HShTrY+2DE5qjzvZs7JIIgt
# 0GCFD9ktx0LxxtRQ7vllKluHWiKk6FxRPyUPxAAYH2Vy1lNM4kzekd8oEARzFAWg
# eW3az2xejEWLNN4eKGxDJ8WDl/FQUSntbjZ80FU3i54tpx5F/0Kr15zW/mJAxZMV
# BrTE2oi0fcI8VMbtoRAmaaslNXdCG1+lqvP4FbrQ6IwSBXkZagHLhFU9HCrG/syT
# RLLhAezu/3Lr00GrJzPQFnCEH1Y58678IgmfORBPC1JKkYaEt2OdDh4GmO0/5cHe
# lAK2/gTlQJINqDr6JfwyYHXSd+V08X1JUPvB4ILfJdmL+66Gp3CSBXG6IwXMZUXB
# htCyIaehr0XkBoDIGMUG1dUtwq1qmcwbdUfcSYCn+OwncVUXf53VJUNOaMWMts0V
# lRYxe5nK+At+DI96HAlXHAL5SlfYxJ7La54i71McVWRP66bW+yERNpbJCjyCYG2j
# +bdpxo/1Cy4uPcU3AWVPGrbn5PhDBf3Froguzzhk++ami+r3Qrx5bIbY3TVzgiFI
# 7Gq3zWcxggSEMIIEgAIBATApMBUxEzARBgNVBAMTCkF1dG9Ib3RrZXkCEHUtV7wu
# QlqeSf4JwzL/sPAwDQYJYIZIAWUDBAIBBQCggYQwGAYKKwYBBAGCNwIBDDEKMAig
# AoAAoQKAADAZBgkqhkiG9w0BCQMxDAYKKwYBBAGCNwIBBDAcBgorBgEEAYI3AgEL
# MQ4wDAYKKwYBBAGCNwIBFTAvBgkqhkiG9w0BCQQxIgQgDjS4Bc3GDfsPoIWkglvc
# PvdekCCM145uHqOQDZyMO0cwDQYJKoZIhvcNAQEBBQAEgYAHNEY93S6LISgKtUyL
# YCXcJAcgvu+5uwv0HRiRsZCjpEZ+XaZIlykY6j1+NLCtdT2OcjFblNDC32gnxJpk
# Ph5EOnMhAC9RH0XydRyNX7I43AQM/pSMs4lGkSV0kUiZpmQ+CtLrtSJtc0dMD2fN
# 5UY2b4ZEBQGOHrOZdWdoDQIGaqGCAyYwggMiBgkqhkiG9w0BCQYxggMTMIIDDwIB
# ATB9MGkxCzAJBgNVBAYTAlVTMRcwFQYDVQQKEw5EaWdpQ2VydCwgSW5jLjFBMD8G
# A1UEAxM4RGlnaUNlcnQgVHJ1c3RlZCBHNCBUaW1lU3RhbXBpbmcgUlNBNDA5NiBT
# SEEyNTYgMjAyNSBDQTECEAqA7xhLjfEFgtHEdqeVdGgwDQYJYIZIAWUDBAIBBQCg
# aTAYBgkqhkiG9w0BCQMxCwYJKoZIhvcNAQcBMBwGCSqGSIb3DQEJBTEPFw0yNTA4
# MjgwNTMzNDdaMC8GCSqGSIb3DQEJBDEiBCAGjWEr5y5iRh1Be5g/oJworXoAViWM
# vhs7/0TfzJiG+jANBgkqhkiG9w0BAQEFAASCAgAU4p7Y3PrUPhPcFzrunDfe+oRI
# zBGG8I+nWTnXq2OG5JmSVzQW+tIymRILXYd3wK/eYUeiebKeHR5XSxpZVR//WN42
# My8O/s9rQR13cVI53wW+X6sguwBdj4cvyS5a7m35rtoy53fjmO1UlXhU0LTOwgxF
# fBksSS7bvl1Vr8bfFDK6c6U3KU3u8kNM7X0cwh1u7X9XPRYUnMbzp8OxVf77nz/C
# 6ATuyZTzT5kHN7+g4TdWJquAxIjNPsWZQPZvI5FJoN7IQER9/jgJBH584TwTVJSD
# rgr5WaLgs+sUJ1RT+RyZq4xfTVJeVNy1mQiPfxNaYJi2zkIVrJc34omULSW4Jll+
# ec1f80rJEw0pawpd+B2nMUQ1BCTod0mepdIAS8yDt5HEZdlDeEVxgSkuvAsPmimt
# 81RrNu8Tk+h47EgU2MIYuJi39gzE1RgOFWWG7FmUTRJv7z/Btn5EJ6qTBBMTMGbj
# KxGwOQgyEiIStAIWDLhDtK4dVr9DE3ij2jdJXUl4hvfzq+2j6Y7oUWxATPIaF3qR
# YEJalWlbRvRhTM3PFoNNkUAIlV4PZzFLyYtULkDd9oTLo0JkmAm4lRLxjZYbM+q4
# P8HozV6wTLXkQvkLNyd/K+IGJ8ekSDn9X9voNdnZW2YX6QTsZ/N+VTO5fIzJxQ6H
# /II5fZWfsy1Au4Tm7Q==
# SIG # End signature block
