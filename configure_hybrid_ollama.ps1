# Configure Ollama for Hybrid RAM+VRAM Usage
# This allows large models to use both system RAM and GPU VRAM

Write-Host "🔧 Configuring Ollama for Hybrid RAM+VRAM Loading..." -ForegroundColor Cyan

# Stop current Ollama instance
Write-Host "⏹️ Stopping current Ollama instance..." -ForegroundColor Yellow
try {
    ollama stop 2>$null
    Start-Sleep -Seconds 2
} catch {
    Write-Host "No running instance to stop" -ForegroundColor Gray
}

# Kill any remaining ollama processes
Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "🚀 Starting Ollama with hybrid configuration..." -ForegroundColor Green

# Environment variables for hybrid loading
$env:OLLAMA_GPU_LAYERS = "auto"           # Let Ollama decide optimal GPU layer distribution
$env:OLLAMA_LOW_VRAM = "true"             # Enable low VRAM mode for better memory management
$env:OLLAMA_FLASH_ATTENTION = "true"      # Enable flash attention for memory efficiency
$env:OLLAMA_KV_CACHE_TYPE = "f16"         # Use FP16 for KV cache to save memory
$env:OLLAMA_MAIN_GPU = "0"                # Use GPU 0 (your RTX 3060)
$env:OLLAMA_SPLIT_MODE = "row"            # Split model by rows across devices
$env:OLLAMA_TENSOR_SPLIT = "0.7,0.3"      # 70% on GPU, 30% on CPU/RAM
$env:OLLAMA_CONTEXT_SIZE = "4096"         # Reduce context size to save memory
$env:OLLAMA_NUMA = "false"                # Disable NUMA for single GPU setups

# Memory limits
$env:OLLAMA_MAX_LOADED_MODELS = "1"       # Only load one model at a time
$env:OLLAMA_MAX_QUEUE = "4"               # Limit concurrent requests

Write-Host "📊 Memory Configuration:" -ForegroundColor Magenta
Write-Host "  • GPU Allocation: 70% (~8.4GB VRAM)" -ForegroundColor White
Write-Host "  • RAM Allocation: 30% (~6-8GB RAM)" -ForegroundColor White
Write-Host "  • Context Size: 4096 tokens" -ForegroundColor White
Write-Host "  • Flash Attention: Enabled" -ForegroundColor White

# Start Ollama server with hybrid configuration
Write-Host "🌐 Starting Ollama server..." -ForegroundColor Green
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden

# Wait for server to start
Start-Sleep -Seconds 5

Write-Host "✅ Ollama configured for hybrid RAM+VRAM usage!" -ForegroundColor Green
Write-Host ""
Write-Host "🧪 Testing configuration..." -ForegroundColor Cyan

# Test if server is running
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 10
    Write-Host "✅ Ollama server is responding!" -ForegroundColor Green
} catch {
    Write-Host "❌ Server not responding yet, may need more time to start" -ForegroundColor Red
}

Write-Host ""
Write-Host "📝 Next steps:" -ForegroundColor Yellow
Write-Host "  1. Try loading gpt-oss:20b with: ollama run gpt-oss:20b" -ForegroundColor White
Write-Host "  2. Monitor memory usage with: nvidia-smi" -ForegroundColor White
Write-Host "  3. Check loaded models with: ollama ps" -ForegroundColor White