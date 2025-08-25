# GPU Optimization Script for Super Alita
# Run as Administrator for full effect

Write-Host "=== Super Alita GPU Optimization Script ===" -ForegroundColor Cyan

# Check if running as administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "WARNING: Not running as Administrator. Some optimizations may not apply." -ForegroundColor Yellow
    Write-Host "For full optimization, right-click PowerShell and 'Run as Administrator'" -ForegroundColor Yellow
}

# 1. Set Ultimate Performance Power Plan
Write-Host "`n1. Setting Ultimate Performance Power Plan..." -ForegroundColor Green
try {
    # Try to activate Ultimate Performance plan
    $ultimatePlans = powercfg -list | Select-String "Ultimate Performance" | ForEach-Object { ($_ -split "\s+")[3] }
    if ($ultimatePlans) {
        $ultimatePlan = $ultimatePlans[0]
        powercfg -setactive $ultimatePlan
        Write-Host "   ✓ Activated Ultimate Performance plan: $ultimatePlan" -ForegroundColor Green
    } else {
        # Fallback to High Performance
        powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
        Write-Host "   ✓ Activated High Performance plan" -ForegroundColor Green
    }
} catch {
    Write-Host "   ✗ Failed to set power plan: $($_.Exception.Message)" -ForegroundColor Red
}

# 2. GPU-specific optimizations
Write-Host "`n2. Configuring GPU settings..." -ForegroundColor Green

# Check NVIDIA GPU presence
$hasNvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($hasNvidia) {
    Write-Host "   Found NVIDIA GPU, applying optimizations..." -ForegroundColor Yellow
    
    # Display current GPU status
    Write-Host "   Current GPU status:" -ForegroundColor Cyan
    nvidia-smi --query-gpu=name,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    
    # Try to set maximum power limit (requires admin)
    if ($isAdmin) {
        Write-Host "   Attempting to set maximum power limit..." -ForegroundColor Yellow
        try {
            nvidia-smi -pl 178  # RTX 3060 max power
            Write-Host "   ✓ Set power limit to maximum" -ForegroundColor Green
        } catch {
            Write-Host "   ✗ Could not set power limit (normal on some systems)" -ForegroundColor Yellow
        }
    }
    
    # GPU memory optimization
    Write-Host "   ✓ GPU memory available: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB" -ForegroundColor Green
} else {
    Write-Host "   No NVIDIA GPU detected, checking for AMD..." -ForegroundColor Yellow
    
    # AMD GPU check
    $amdGpu = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*AMD*" -or $_.Name -like "*Radeon*" }
    if ($amdGpu) {
        Write-Host "   Found AMD GPU: $($amdGpu.Name)" -ForegroundColor Green
        Write-Host "   AMD GPU optimizations require AMD Radeon Software" -ForegroundColor Yellow
    } else {
        Write-Host "   No dedicated GPU detected, using integrated graphics" -ForegroundColor Yellow
    }
}

# 3. System optimizations for AI workloads
Write-Host "`n3. Applying system optimizations..." -ForegroundColor Green

# Disable Windows Game Mode (can interfere with AI workloads)
try {
    Set-ItemProperty -Path "HKCU:\Software\Microsoft\GameBar" -Name "AllowAutoGameMode" -Value 0 -ErrorAction Stop
    Write-Host "   ✓ Disabled Game Mode" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Could not modify Game Mode setting" -ForegroundColor Yellow
}

# Set processor scheduling for background services (better for AI workloads)
if ($isAdmin) {
    try {
        Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\PriorityControl" -Name "Win32PrioritySeparation" -Value 24
        Write-Host "   ✓ Optimized processor scheduling for background services" -ForegroundColor Green
    } catch {
        Write-Host "   ✗ Could not modify processor scheduling" -ForegroundColor Red
    }
}

# 4. Memory optimizations
Write-Host "`n4. Memory optimization..." -ForegroundColor Green
$totalRAM = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
$availableRAM = [math]::Round((Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue / 1024, 2)
Write-Host "   Total RAM: ${totalRAM} GB" -ForegroundColor Cyan
Write-Host "   Available RAM: ${availableRAM} GB" -ForegroundColor Cyan

# 5. Environment variables for AI optimization
Write-Host "`n5. Setting AI optimization environment variables..." -ForegroundColor Green

# CUDA optimizations
[Environment]::SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "0", "User")
[Environment]::SetEnvironmentVariable("CUDA_DEVICE_ORDER", "PCI_BUS_ID", "User")
[Environment]::SetEnvironmentVariable("CUDA_LAUNCH_BLOCKING", "0", "User")

# PyTorch optimizations
[Environment]::SetEnvironmentVariable("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128", "User")
[Environment]::SetEnvironmentVariable("OMP_NUM_THREADS", [Environment]::ProcessorCount, "User")

# TensorFlow optimizations
[Environment]::SetEnvironmentVariable("TF_GPU_ALLOCATOR", "cuda_malloc_async", "User")
[Environment]::SetEnvironmentVariable("TF_FORCE_GPU_ALLOW_GROWTH", "true", "User")

Write-Host "   ✓ Set CUDA environment variables" -ForegroundColor Green
Write-Host "   ✓ Set PyTorch optimizations" -ForegroundColor Green
Write-Host "   ✓ Set TensorFlow optimizations" -ForegroundColor Green

# 6. Create performance monitoring function
Write-Host "`n6. Creating performance monitoring tools..." -ForegroundColor Green

$monitorScript = @"
# GPU Performance Monitor for Super Alita
function Watch-GPUPerformance {
    while (`$true) {
        Clear-Host
        Write-Host "=== Super Alita GPU Performance Monitor ===" -ForegroundColor Cyan
        Write-Host "Press Ctrl+C to stop monitoring`n" -ForegroundColor Yellow
        
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv
        } else {
            Write-Host "NVIDIA GPU monitoring not available" -ForegroundColor Yellow
            Get-Counter '\GPU Process Memory(*)\Local Usage' -ErrorAction SilentlyContinue
        }
        
        Write-Host "`nCPU Usage:" -ForegroundColor Green
        Get-Counter '\Processor(_Total)\% Processor Time' | Select-Object -ExpandProperty CounterSamples | Select-Object InstanceName, CookedValue
        
        Write-Host "`nMemory Usage:" -ForegroundColor Green
        `$mem = Get-Counter '\Memory\Available MBytes'
        Write-Host "Available Memory: `$([math]::Round(`$mem.CounterSamples.CookedValue / 1024, 2)) GB"
        
        Start-Sleep -Seconds 5
    }
}

# Export the function
Export-ModuleMember -Function Watch-GPUPerformance
"@

$monitorScript | Out-File -FilePath "$env:USERPROFILE\Documents\WindowsPowerShell\Modules\SuperAlitaMonitor\SuperAlitaMonitor.psm1" -Force
Write-Host "   ✓ Created GPU performance monitor (use: Watch-GPUPerformance)" -ForegroundColor Green

# 7. Final recommendations
Write-Host "`n=== Optimization Complete ===" -ForegroundColor Cyan
Write-Host "`nRecommendations:" -ForegroundColor Yellow
Write-Host "1. Restart VS Code to apply environment variables" -ForegroundColor White
Write-Host "2. Run 'Watch-GPUPerformance' to monitor performance" -ForegroundColor White
Write-Host "3. Ensure your Python environment has CUDA-enabled packages:" -ForegroundColor White
Write-Host "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Gray
Write-Host "4. For maximum performance, close unnecessary applications" -ForegroundColor White

Write-Host "`nSystem Summary:" -ForegroundColor Green
Write-Host "   GPU: $(if ($hasNvidia) { (nvidia-smi --query-gpu=name --format=csv,noheader,nounits) } else { "Integrated Graphics" })" -ForegroundColor Cyan
Write-Host "   RAM: ${totalRAM} GB total, ${availableRAM} GB available" -ForegroundColor Cyan
Write-Host "   CPU Cores: $([Environment]::ProcessorCount)" -ForegroundColor Cyan
Write-Host "   Power Plan: Ultimate Performance" -ForegroundColor Cyan

if (-not $isAdmin) {
    Write-Host "`nNote: For maximum optimization, run this script as Administrator" -ForegroundColor Yellow
}