#!/usr/bin/env pwsh
# Test Ollama GPU setup and model performance

Write-Host "🧪 Testing Ollama GPU Setup" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green

# Test 1: Check GPU availability
Write-Host "`n1️⃣ Checking GPU..." -ForegroundColor Yellow
try {
    $gpu = nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    Write-Host "✅ GPU Found: $gpu" -ForegroundColor Green
} catch {
    Write-Host "❌ NVIDIA GPU not found or nvidia-smi not available" -ForegroundColor Red
    exit 1
}

# Test 2: Check Ollama server
Write-Host "`n2️⃣ Checking Ollama server..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
    Write-Host "✅ Ollama server is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Ollama server not responding. Start it with: .\start_ollama_gpu.ps1" -ForegroundColor Red
    exit 1
}

# Test 3: Test models
Write-Host "`n3️⃣ Testing available models..." -ForegroundColor Yellow

$testModels = @("llama3.2:3b", "llama3.2:1b", "gpt-oss:20b")
$workingModels = @()

foreach ($model in $testModels) {
    Write-Host "   Testing $model..." -ForegroundColor Cyan
    try {
        # Quick test with very short prompt
        $testPrompt = "Hi"
        $startTime = Get-Date
        
        $result = & ollama run $model $testPrompt
        
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ $model works! (${duration}s)" -ForegroundColor Green
            $workingModels += $model
        } else {
            Write-Host "   ❌ $model failed" -ForegroundColor Red
        }
    } catch {
        Write-Host "   ❌ $model error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Test 4: Performance benchmark
if ($workingModels.Count -gt 0) {
    Write-Host "`n4️⃣ Performance benchmark..." -ForegroundColor Yellow
    $bestModel = $workingModels[0]
    Write-Host "   Using: $bestModel" -ForegroundColor Cyan
    
    $prompt = "Write a Python function to reverse a string"
    Write-Host "   Prompt: $prompt" -ForegroundColor Gray
    
    $startTime = Get-Date
    $result = & ollama run $bestModel $prompt
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    Write-Host "   ✅ Completed in ${duration} seconds" -ForegroundColor Green
    Write-Host "   📝 Response: $($result.Substring(0, [Math]::Min(100, $result.Length)))..." -ForegroundColor Gray
}

# Test 5: GPU memory usage
Write-Host "`n5️⃣ Final GPU status..." -ForegroundColor Yellow
try {
    $gpu = nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    $used, $total = $gpu -split ','
    $usage = [math]::Round(($used / $total) * 100, 1)
    Write-Host "   GPU Memory: $used MB / $total MB ($usage%)" -ForegroundColor Cyan
} catch {
    Write-Host "   Could not get GPU memory info" -ForegroundColor Yellow
}

Write-Host "`n🎉 GPU Setup Test Complete!" -ForegroundColor Green
Write-Host "Working models: $($workingModels -join ', ')" -ForegroundColor Cyan

if ($workingModels.Count -eq 0) {
    Write-Host "❌ No models are working. Try running: .\start_ollama_gpu.ps1" -ForegroundColor Red
    exit 1
} else {
    Write-Host "✅ GPU acceleration is working!" -ForegroundColor Green
}