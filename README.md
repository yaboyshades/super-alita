# Super Alita

Advanced, event-driven AI agent system with modular plugins, MCP integration, knowledge graph, streaming orchestration, and adaptive LLM routing.

Production-ready architecture with:

- Streaming orchestration
- Rich telemetry + MCP broadcast
- Fallback LLM routing (Gemini → local Super Alita → mock / local HF model)
- Knowledge graph + cognitive fabric (Atoms / Bonds)
- Modular plugin system
- OpenAI-compatible local adapter option

## Key Features

- Event bus with Redis optional backend
- MCP server + VS Code integration
- Atoms/Bonds cognitive fabric
- Modular plugin architecture
- Streaming single-turn agent router
- Tool execution + echo sample tool
- Real-time telemetry broadcasting via MCP
- Automatic LLM fallback (Gemini -> local Super Alita -> mock) with telemetry events
- Direct local Hugging Face model loading (`LLM_MODEL=hf:<model_id>`)

## Quick Start

Two equivalent setup paths are provided: Makefile workflow (recommended) or raw Python commands.

### 1. Environment file

```bash
cp .env.example .env  # then set at least one provider key or local model config
```

### 2. Install dependencies

Using Make (includes lint targets):

```bash
make deps               # CPU defaults, includes torch CPU build
# For GPU acceleration:
# pip install --index-url https://download.pytorch.org/whl/cu121 torch
# pip install -r requirements-gpu.txt
make lint  # optional
```

Or manually:

```bash
python -m venv .venv
./.venv/Scripts/Activate.ps1  # Windows PowerShell
pip install -e .
# GPU extras (optional):
# pip install --index-url https://download.pytorch.org/whl/cu121 torch
# pip install -r requirements-gpu.txt
# The above replaces the CPU build installed by default
```

### 3. Run the development server

```bash
make run
# or manually
python -m uvicorn src.main:app --reload --port 8080
```

### 4. Run tests

```bash
make test
# or manually
pytest -q
```

Health check:

```bash
curl http://127.0.0.1:8080/healthz
```

Debug utilities (`debug_fixed.py`, `debug_matching.py`, `utility_debug.py`) are under `scripts/`.

## LLM Fallback Configuration

Set `LLM_MODEL=auto` to enable automatic provider selection.

Order of preference:

1. Gemini (if `GEMINI_API_KEY` or `GOOGLE_API_KEY` set)
2. Local Super Alita OpenAI-compatible adapter (`SUPER_ALITA_BASE_URL`)
3. Deterministic mock (development/test)

Environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLM_MODEL` | Target model name or `auto` | `mock` |
| `SUPER_ALITA_BASE_URL` | Base URL for local adapter | `http://127.0.0.1:8080` |
| `SUPER_ALITA_MODEL` | Model name passed to adapter | `gpt-oss-20b-4bit` |
| `SUPER_ALITA_API_KEY` | Optional bearer token | (unset) |

Telemetry events emitted:

- `llm_fallback` when Super Alita fallback client is selected
- `performance_metric` with `metric=llm_stream_duration_s` per streamed turn

Example `.env`:

```dotenv
LLM_MODEL=auto
GEMINI_API_KEY=your_key_here   # optional; if absent will fallback
SUPER_ALITA_BASE_URL=http://127.0.0.1:8080
SUPER_ALITA_MODEL=gpt-oss-20b-4bit
```

Force local adapter explicitly:

```dotenv
LLM_MODEL=super-alita
```

## Telemetry

Telemetry events stream to MCP for real-time inspection. New events introduced:

- `llm_fallback` (selection decision)
- `performance_metric` (duration metrics)

## Development

Run tests:

```bash
pytest -q
```

Code style:

```bash
ruff check .
black .
```

## License

Apache 2.0 (placeholder – update as appropriate).

