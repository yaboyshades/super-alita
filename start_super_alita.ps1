# Super Alita PowerShell Startup Script
# Launches all services and opens the chat interface

param(
    [int]$Port = 8080,
    [string]$Host = "127.0.0.1",
    [switch]$NoBrowser,
    [switch]$Verbose
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to write colored output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# Function to check if a service is responding
function Test-ServiceHealth {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 30
    )

    $endpoints = @("/health", "/healthz")
    $startTime = Get-Date

    while (((Get-Date) - $startTime).TotalSeconds -lt $TimeoutSeconds) {
        foreach ($endpoint in $endpoints) {
            try {
                $response = Invoke-WebRequest -Uri "$Url$endpoint" -TimeoutSec 5 -UseBasicParsing
                if ($response.StatusCode -eq 200) {
                    return $true
                }
            }
            catch {
                # Continue trying
            }
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

# Function to cleanup processes on exit
function Stop-Services {
    Write-ColorOutput "🛑 Stopping all services..." "Yellow"

    # Stop any uvicorn processes
    Get-Process -Name "python" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*uvicorn*" } |
        Stop-Process -Force -ErrorAction SilentlyContinue

    # Stop any MCP server processes
    Get-Process -Name "python" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*mcp_server*" } |
        Stop-Process -Force -ErrorAction SilentlyContinue

    Write-ColorOutput "✅ All services stopped. Goodbye! 👋" "Green"
}

# Register cleanup function for Ctrl+C
$null = Register-EngineEvent PowerShell.Exiting -Action { Stop-Services }

try {
    # Print banner
    Write-ColorOutput ""
    Write-ColorOutput "╔══════════════════════════════════════════════════════════════╗" "Cyan"
    Write-ColorOutput "║                    🤖 Super Alita Agent                     ║" "Cyan"
    Write-ColorOutput "║              AI-Powered Development Assistant                ║" "Cyan"
    Write-ColorOutput "╚══════════════════════════════════════════════════════════════╝" "Cyan"
    Write-ColorOutput ""
    Write-ColorOutput "🚀 Starting Super Alita Agent System..." "Yellow"
    Write-ColorOutput ""

    # Check if we're in the right directory
    if (-not (Test-Path "app.py")) {
        throw "app.py not found. Please run this script from the Super Alita root directory."
    }

    # Start main server
    Write-ColorOutput "Starting main server on ${Host}:${Port}..." "Yellow"

    $serverJob = Start-Job -ScriptBlock {
        param($Host, $Port, $WorkingDir)
        Set-Location $WorkingDir
        python -m uvicorn app:app --host $Host --port $Port --reload
    } -ArgumentList $Host, $Port, (Get-Location).Path

    # Start MCP server if available
    if (Test-Path "mcp_server_wrapper.py") {
        Write-ColorOutput "Starting MCP server..." "Yellow"

        $mcpJob = Start-Job -ScriptBlock {
            param($WorkingDir)
            Set-Location $WorkingDir
            python mcp_server_wrapper.py
        } -ArgumentList (Get-Location).Path
    }

    # Wait for services to be ready
    Write-ColorOutput "Waiting for services to be ready..." "Yellow"

    $mainUrl = "http://${Host}:${Port}"

    if (Test-ServiceHealth -Url $mainUrl -TimeoutSeconds 30) {
        Write-ColorOutput "✅ Main Server: $mainUrl (Ready)" "Green"
    }
    else {
        throw "❌ Main Server failed to start within timeout"
    }

    if (Test-Path "mcp_server_wrapper.py") {
        Write-ColorOutput "✅ MCP Server: Running with stdio transport" "Green"
    }

    # Open browser if requested
    if (-not $NoBrowser) {
        Start-Sleep -Seconds 3
        try {
            Start-Process $mainUrl
            Write-ColorOutput "🌐 Chat Interface: $mainUrl (Opened in browser)" "Green"
        }
        catch {
            Write-ColorOutput "🌐 Chat Interface: $mainUrl (Please open manually)" "Yellow"
        }
    }

    # Display status
    Write-ColorOutput ""
    Write-ColorOutput "╔══════════════════════════════════════════════════════════════╗" "Green"
    Write-ColorOutput "║                     🎉 Super Alita Ready!                   ║" "Green"
    Write-ColorOutput "╚══════════════════════════════════════════════════════════════╝" "Green"
    Write-ColorOutput ""
    Write-ColorOutput "🌐 Services Running:" "White"
    Write-ColorOutput "   • Main Server: $mainUrl" "Gray"
    Write-ColorOutput "   • Chat Interface: $mainUrl" "Gray"
    Write-ColorOutput "   • Health Check: $mainUrl/health" "Gray"
    Write-ColorOutput "   • API Documentation: $mainUrl/docs" "Gray"
    Write-ColorOutput ""
    Write-ColorOutput "🔧 Available Endpoints:" "White"
    Write-ColorOutput "   • Autogen: $mainUrl/autogen" "Gray"
    Write-ColorOutput "   • DeepCode: $mainUrl/deepcode" "Gray"
    Write-ColorOutput "   • Tools: $mainUrl/tools" "Gray"
    Write-ColorOutput "   • Perplexica Search: Available via chat interface" "Gray"
    Write-ColorOutput ""
    Write-ColorOutput "📝 Press Ctrl+C to stop all services." "Yellow"
    Write-ColorOutput ""

    # Keep running until Ctrl+C
    Write-ColorOutput "Super Alita is running. Press Ctrl+C to stop..." "Cyan"

    while ($true) {
        Start-Sleep -Seconds 1

        # Check if main server job is still running
        if ($serverJob.State -eq "Completed" -or $serverJob.State -eq "Failed") {
            $jobOutput = Receive-Job $serverJob
            if ($Verbose -and $jobOutput) {
                Write-ColorOutput "Server output: $jobOutput" "Gray"
            }
            throw "Main server process stopped unexpectedly"
        }
    }
}
catch {
    Write-ColorOutput "❌ Error: $($_.Exception.Message)" "Red"
    exit 1
}
finally {
    Stop-Services
}
# SIG # Begin signature block
# MIIaSQYJKoZIhvcNAQcCoIIaOjCCGjYCAQExDzANBglghkgBZQMEAgEFADB5Bgor
# BgEEAYI3AgEEoGswaTA0BgorBgEEAYI3AgEeMCYCAwEAAAQQH8w7YFlLCE63JNLG
# KX7zUQIBAAIBAAIBAAIBAAIBADAxMA0GCWCGSAFlAwQCAQUABCARZGd7rE1ooJz0
# eOwX+2VLNmAZWFt2oGoN2CTbhLJEaaCCFRswggHdMIIBRqADAgECAhB1LVe8LkJa
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
# MQ4wDAYKKwYBBAGCNwIBFTAvBgkqhkiG9w0BCQQxIgQg66csNlOJfZARnWH+kEjR
# w04QQxNHp7zQhioVD4ckmLUwDQYJKoZIhvcNAQEBBQAEgYBjqDGrieVkWlshWcdK
# dykwRVVG25b9Df6zOVLbKRbdh4PSZV4n0wuEwZ9vB4Dr9WS9MjiUD5ezkkWxWJtK
# ilJiQ0DEU6gFD+IEbmMvNzIN7l1LBh5CYLyffbjVtYFyT18x4J19O65hMdf9jzXg
# wIQV9Z/QTY+eBB96AiIKC4dtHaGCAyYwggMiBgkqhkiG9w0BCQYxggMTMIIDDwIB
# ATB9MGkxCzAJBgNVBAYTAlVTMRcwFQYDVQQKEw5EaWdpQ2VydCwgSW5jLjFBMD8G
# A1UEAxM4RGlnaUNlcnQgVHJ1c3RlZCBHNCBUaW1lU3RhbXBpbmcgUlNBNDA5NiBT
# SEEyNTYgMjAyNSBDQTECEAqA7xhLjfEFgtHEdqeVdGgwDQYJYIZIAWUDBAIBBQCg
# aTAYBgkqhkiG9w0BCQMxCwYJKoZIhvcNAQcBMBwGCSqGSIb3DQEJBTEPFw0yNTA4
# MjgxOTMxMTlaMC8GCSqGSIb3DQEJBDEiBCDdxCxUo41s/x3CHS+3Z8YhgAhMidbP
# L0aJp4Y6UWRObzANBgkqhkiG9w0BAQEFAASCAgCa6diprUXSThc8tllyNQmgNrZT
# lADNZ408l86R9PQ7QjBoReqKom7cUqRvOhhDdycgHuN5fDpFd8vpf5qCJuwX0L73
# nYTYfHAgfzh9l28eRPTkcm3UAPt1TpWBaC6esHOl38n5rQpXws2CWCZ4Afmesvn3
# 5ZSs3bM3+Nxi20HmL7AberNvK32dSm0PnbIzm9o3E0tmaIw+7u4jOaiLsCYleLpr
# Gqy3bJFiMIy8p8V83bMxZSMZqu3LurVHpfDaMlgxa0Qpowr3wAucUDMcCToN2sk8
# W+TtGoGkU6QutrKWZSOYyxp7ALjqPOhXmQLWol4Y9Vu44UgAIdewETLRxIOgef7e
# hIpVGRj/5AiX9xNpk8PXOV2OSm7PO8P6/LcrWxcd99oyvQRhxXWOVXUiLaI2y7As
# 50vMMK9h8SMfve5Co0hU5N8jqEks6F0qeljHqs/jhWVSfmc5BEEV6f8vleczr7gv
# TvYgWEKAd0Ppyn+9u83TVyDzP4JG4gwf16p/1G+e/kyPGqEHAZPBNrUzsahyt0pu
# VLzsbSyJWLvsddQEE452xxTn/lT6x89lelYEv8vHXz164fZ1NJT21MOfh9exu2Qd
# hLLWeWDPfQ4kyOTvnbjy6YghiFpy5W8ggg5AFT83LGEQpsicP+3P4rH76BEFKqGq
# RFseP46qf2CySoDBFg==
# SIG # End signature block
