# Ollama Model Behavior Issue: gpt-oss-20b-split

## Problem Description

The `gpt-oss-20b-split` model is generating verbose, unrelated text instead of responding appropriately to simple prompts like "hi". This is a common issue with large language models that need proper configuration.

## What's Happening

When you ran `ollama run gpt-oss-20b-split` and typed "hi", the model generated:
```
, I need to understand how the data for each of these properties is gathered and compiled, especially since it encompasses a wide array of data types. How does the process work for something like the surface temperature of a planet?

For planetary surface temperature, I recall that the information comes from a combination of in situ measurements, remote sensing data from spacecraft, and atmospheric models...
```

This indicates:
1. **Context Leakage**: The model is continuing from some internal context/training data
2. **Poor Instruction Following**: The model wasn't properly fine-tuned for chat/instruction following
3. **No Stop Tokens**: The model doesn't know when to stop generating

## Solutions

### 1. Immediate Fix: Stop Current Generation
In your Ollama terminal where the model is running:
- Press `Ctrl+C` to stop the current generation
- Type `/bye` to exit the chat session

### 2. Better Model Configuration

Create a new Modelfile with better parameters:

```modelfile
# tools/ollama/Modelfile.gpt-oss-20b-fixed
FROM gpt-oss:20b

# System prompt to control behavior
SYSTEM """You are a helpful assistant. Give brief, direct answers to questions. Do not elaborate unless specifically asked."""

# Parameters to control generation
PARAMETER temperature 0.1
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER repeat_last_n 64
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"
PARAMETER stop "\n\n"
```

Create the fixed model:
```bash
ollama create gpt-oss-20b-fixed -f tools/ollama/Modelfile.gpt-oss-20b-fixed
```

### 3. API Usage with Constraints

When using the model via API, always use these parameters:

```json
{
  "model": "gpt-oss-20b-split",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant. Give brief, direct answers. Do not elaborate unless asked."
    },
    {
      "role": "user",
      "content": "Hi!"
    }
  ],
  "max_tokens": 20,
  "temperature": 0.1,
  "top_p": 0.9,
  "frequency_penalty": 0.5,
  "presence_penalty": 0.3,
  "stop": ["\n\n", ".", "!", "?"]
}
```

### 4. Super Alita Integration

For Super Alita integration, use the working `llama3.2:1b` model:

```python
config = {
    'vllm_base_url': 'http://localhost:11434/v1',
    'model_name': 'llama3.2:1b',  # Use the working model
    'timeout': 30.0,
    'max_retries': 2
}
```

## Recommended Next Steps

1. **Stop the current Ollama session** (Ctrl+C in the terminal)
2. **Use llama3.2:1b for testing** - it's properly trained and responds well
3. **If you need the 20B model**, create a fixed version with the Modelfile above
4. **Test with Super Alita** using the working configuration

## Working Example

Here's a working test you can run:

```bash
# Test the working model
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "Hi!"}],
    "max_tokens": 10,
    "temperature": 0.1
  }'
```

This should give you a brief, appropriate response like "Hello! How can I help you today?"

## Summary

The issue is that `gpt-oss-20b-split` needs better configuration and constraints. Use `llama3.2:1b` for reliable testing, and if you need the larger model, create a properly configured version with the Modelfile approach.
