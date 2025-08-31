# Developer Secrets & Configuration Guide

This guide documents all environment variables and secrets used across the Super Alita DX Kit (agent runtime, backend Semantic Kernel service, VS Code extension, local model serving, and telemetry). It explains purpose, scope, when required, security considerations, and safe local-first workflows.

## Quick Reference Matrix

| Category | Variable | Required? | Purpose | Typical Value / Format | Used By | Fallback / Notes |
|----------|----------|-----------|---------|------------------------|---------|------------------|
| Core LLM | `LLM_PROVIDER` | Optional | Selects primary provider (`gemini`, `openai`, `anthropic`, `azure-openai`, `ollama`, `auto`). | `auto` | Runtime selector | `auto` will probe ordered list & fall back to local.
| Core LLM | `LLM_MODEL` | Optional | Overrides default model name for chosen provider. | e.g. `gpt-4o-mini`, `claude-3-sonnet`, `gemini-1.5-pro` | Runtime | If blank provider default used.
| Local HF | `HF_HOME` | Optional | Root cache for Hugging Face assets. | Absolute path | Download script / transformers | Defaults to user cache.
| Local HF | `HF_TOKEN` | Optional (private models) | Auth for gated/private model downloads. | hf_xxx token | download script | Omit for public models.
| Local Adapter | `LOCAL_MODEL_PATH` | Optional | Path to a locally downloaded model folder. | `models/Meta-Llama-3-8B-Instruct` | `local_adapter_server.py` | Needed for direct transformers load.
| Local Adapter | `DEVICE` | Optional | Inference device | `cuda`, `cpu`, `mps` | Adapter | Auto-detect w/ CUDA pref.
| Ollama | `OLLAMA_HOST` | Optional | Host:port for Ollama HTTP API. | `http://127.0.0.1:11434` | Backend & extension | Default Ollama binding.
| Ollama | `OLLAMA_MODEL` | Optional | Model tag served by Ollama. | `llama3.1:8b`, `qwen2:7b` | Backend & extension | Extension falls back to setting or prompt.
| Azure OpenAI | `AZURE_OPENAI_ENDPOINT` | Required (if provider azure-openai) | Base endpoint for Azure OpenAI. | `https://<name>.openai.azure.com/` | Backend/runtime | - |
| Azure OpenAI | `AZURE_OPENAI_KEY` | Required | API key | 32+ char secret | Backend/runtime | Store in secret manager.
| Azure OpenAI | `AZURE_OPENAI_DEPLOYMENT` | Required | Deployed model name | e.g. `gpt-4o-mini` | Backend/runtime | Maps to model.
| OpenAI | `OPENAI_API_KEY` | Required (if provider openai) | API key | `sk-...` | Backend/runtime | - |
| Anthropic | `ANTHROPIC_API_KEY` | Required (if provider anthropic) | API key | `sk-ant-...` | Backend/runtime | - |
| Google | `GOOGLE_API_KEY` | Required (if provider gemini) | API key | AI Studio key | Backend/runtime | - |
| Redis | `REDIS_URL` | Optional | Event bus / telemetry channel. | `redis://localhost:6379/0` | Event bus | Local memory fallback.
| Telemetry | `TELEMETRY_ENABLE` | Optional | Master toggle for telemetry. | `true` / `false` | Runtime, extension | Ensure consent.
| Telemetry | `TELEMETRY_DEBUG` | Optional | Verbose telemetry logging. | `1` | Runtime | Avoid in prod.
| SK Backend | `PORT` | Optional | Backend HTTP port. | `8001` | `semantic_kernel_agent.py` | Defaults if unset.
| Misc | `.env` | - | Collected variable definitions. | key=value lines | All | Do not commit secrets.

## Provider Selection Logic (`LLM_PROVIDER`)

`auto` strategy attempts providers in priority order:
1. Explicit local Ollama (if running & model loaded)
2. Local HF direct model (if `LOCAL_MODEL_PATH` present)
3. Azure OpenAI
4. OpenAI
5. Anthropic
6. Gemini
7. Fallback internal Super Alita client

Telemetry event `llm_fallback` is emitted with fields: `attempt_order`, `chosen`, `latency_ms`, `reason` when a fallback occurs.

## Secure Local Development Workflow

1. Copy `.env.example` to `.env` (never commit secrets).
2. Populate only the providers you actively use (principle of least privilege).
3. Prefer local models (Ollama or transformers) for routine dev to minimize API exposure.
4. Use a per-developer API key with minimum scopes & rotation schedule.
5. Validate secrets load early: run `python -m src.main --dry-run` or backend health endpoint.
6. Enable telemetry locally only if anonymized & compliant.

## Secrets Handling Best Practices

- Never echo secrets in terminals with screen sharing.
- Use OS keychain / secret manager for long-lived keys (e.g., Windows Credential Manager, 1Password).
- Avoid storing API keys in shell history: prefix with space in PowerShell (` SPACED=...`).
- Rotate keys regularly (30/60/90 day cadence).
- Restrict network egress of dev environments if possible.
- Review dependency chain for unintended exfiltration (disable auto update telemetry unless required).

## Ollama Integration

- Install Ollama: https://ollama.com/download
- Pull a model: `ollama pull llama3.1:8b`
- (Optional) Create a custom Modelfile for quantization / system prompts.
- Ensure the daemon is running (default port 11434).
- Set `OLLAMA_HOST` if non-default.
- Set `OLLAMA_MODEL` or allow extension command prompt to choose.
- Backend & extension both call `POST /api/chat` with messages array; streaming uses newline-delimited JSON.

## Local HF Model Workflow

Use provided script:
`python scripts/download_model.py --model meta-llama/Meta-Llama-3-8B-Instruct --output models/Meta-Llama-3-8B-Instruct`
Then launch adapter:
`python src/local_adapter_server.py --model-path models/Meta-Llama-3-8B-Instruct --device cuda`
Set `LLM_PROVIDER=auto` & `LLM_MODEL=auto`.

## Telemetry Considerations

Telemetry is opt-in. Ensure:
- No raw prompt/user PII.
- Hash or bucket latencies.
- Fallback events include reason categories only (timeout, auth_error, rate_limit, provider_down).

## Testing Secrets Loading

Add a pytest that asserts required variables for a selected provider are present (skip if provider not chosen). Example skeleton:
```python
import os, pytest

@pytest.mark.parametrize('provider,required', [
  ('openai', ['OPENAI_API_KEY']),
  ('azure-openai', ['AZURE_OPENAI_ENDPOINT','AZURE_OPENAI_KEY','AZURE_OPENAI_DEPLOYMENT']),
])
def test_provider_env(provider, required):
    if os.getenv('LLM_PROVIDER') != provider:
        pytest.skip('Different provider active')
    missing = [k for k in required if not os.getenv(k)]
    assert not missing, f"Missing: {missing}"
```

## Rotation & Incident Response

- Maintain an internal runbook listing contact for each provider.
- On leak: revoke key, rotate, audit logs, invalidate sessions, add detection rule.
- Record timeline & remediation in `SECURITY_INCIDENTS.md` (private).

## Appendix: Minimal .env Example

```
# Core
LLM_PROVIDER=auto
LLM_MODEL=auto

# Local (optional)
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b

# Azure (optional)
# AZURE_OPENAI_ENDPOINT=...
# AZURE_OPENAI_KEY=...
# AZURE_OPENAI_DEPLOYMENT=...

# OpenAI (optional)
# OPENAI_API_KEY=sk-...

# Anthropic (optional)
# ANTHROPIC_API_KEY=...

# Gemini (optional)
# GOOGLE_API_KEY=...

# Redis (optional)
# REDIS_URL=redis://localhost:6379/0

TELEMETRY_ENABLE=true
TELEMETRY_DEBUG=0
```

---
Maintainers: Update this document when introducing new environment variables or changing provider selection logic.
