"""
🚀 Super Alita + Ollama Integration Guide
=======================================

This document shows how to configure Super Alita's DeepConf system
to work with Ollama-served models like gpt-oss:20b.

## Configuration Options

### 1. vLLM Client with Ollama (OpenAI-compatible API)

```python
from src.abilities.deepconf_ability import DeepConfAbility

# Configure for Ollama's OpenAI-compatible endpoint
ollama_config = {
    'vllm_base_url': 'http://localhost:11434/v1',  # Ollama OpenAI endpoint
    'model_name': 'gpt-oss:20b',                   # Your Ollama model
    'timeout': 60.0,                               # Longer timeout for large models
    'max_retries': 2
}

ability = DeepConfAbility(ollama_config)
```

### 2. Integration Steps

1. **Start Ollama Server**
   ```bash
   ollama serve
   ```

2. **Load Your Model**
   ```bash
   # For gpt-oss:20b (requires significant RAM/VRAM)
   ollama run gpt-oss:20b

   # Or for smaller models
   ollama run llama3.2:3b
   ```

3. **Initialize Super Alita**
   ```python
   success = await ability.initialize(None)
   if success:
       print("Ready for consensus sampling!")
   ```

4. **Use Consensus Sampling**
   ```python
   response = await ability.sample_consensus(
       prompt="What is the capital of France?",
       num_samples=3,
       mode=ConsensusMode.ONLINE,
       temperature=0.7
   )

   print(f"Consensus: {response.consensus_text}")
   print(f"Confidence: {response.consensus_confidence}")
   ```

## Model Recommendations

### gpt-oss:20b
- **Size**: 20B parameters (~13GB)
- **Use case**: High-quality consensus sampling
- **Memory**: Requires 16GB+ RAM/VRAM
- **Config**: Use 60s timeout, 2 retries

### llama3.2:3b
- **Size**: 3B parameters (~2GB)
- **Use case**: Fast consensus sampling
- **Memory**: Requires 4GB+ RAM
- **Config**: Use 30s timeout, 3 retries

### llama3.2:1b
- **Size**: 1B parameters (~1.3GB)
- **Use case**: Ultra-fast testing
- **Memory**: Requires 2GB+ RAM
- **Config**: Use 20s timeout, 3 retries

## Consensus Modes with Ollama

### Offline Mode
- Uses caching for repeated queries
- Best for production with known prompts
- Faster response times

### Online Mode
- Real-time generation for each sample
- Best for novel prompts
- Higher quality consensus

### Hybrid Mode
- Automatic selection based on cache availability
- Best balance of speed and quality
- Recommended for most use cases

## Troubleshooting

### Model Loading Issues
```bash
# Check available models
ollama list

# Check if model is loaded
curl http://localhost:11434/api/tags

# Test model directly
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "gpt-oss:20b",
  "prompt": "Hello",
  "stream": false
}'
```

### Memory Issues
- gpt-oss:20b requires significant memory
- Use smaller models for testing
- Check system resources before loading

### Connection Issues
- Ensure Ollama is running on port 11434
- Check firewall settings
- Verify model is loaded and ready

## Example Usage

```python
import asyncio
from src.abilities.deepconf_ability import DeepConfAbility, ConsensusMode

async def main():
    # Configure for your Ollama model
    config = {
        'vllm_base_url': 'http://localhost:11434/v1',
        'model_name': 'gpt-oss:20b',  # or 'llama3.2:3b'
        'timeout': 60.0
    }

    ability = DeepConfAbility(config)

    # Initialize
    if await ability.initialize(None):
        print("🎯 Super Alita ready with Ollama!")

        # Generate consensus
        response = await ability.sample_consensus(
            prompt="Explain quantum computing in simple terms",
            num_samples=3,
            mode=ConsensusMode.HYBRID,
            consensus_method="weighted_vote"
        )

        print(f"📝 Consensus: {response.consensus_text}")
        print(f"🎯 Confidence: {response.consensus_confidence:.2f}")
        print(f"📊 Method: {response.aggregation_method}")

        await ability.cleanup()
    else:
        print("❌ Failed to connect to Ollama")

if __name__ == "__main__":
    asyncio.run(main())
```

## Current Status

✅ **Configuration**: DeepConf ability can be configured for Ollama
✅ **API Compatibility**: vLLM client supports OpenAI-compatible endpoints
✅ **Plugin Integration**: Full plugin interface compliance
⚠ **Model Loading**: gpt-oss:20b requires sufficient disk space and memory
⚠ **Network**: Timeout issues may occur with large models

## Next Steps

1. Ensure sufficient disk space for model downloads
2. Load gpt-oss:20b or alternative model in Ollama
3. Test the full integration with consensus sampling
4. Optimize timeout and retry settings for your hardware

The integration is ready - just waiting for the model to be available!
"""
