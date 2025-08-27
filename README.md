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
- Optional LangChain runnable support via `reug_runtime.adapter` with local fallbacks

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

## VS Code Insiders + GPT-OSS Quick Start

To use VS Code Insiders with GPT-OSS hosted by Ollama (streamlined setup):

### Prerequisites

1. Install [VS Code Insiders](https://code.visualstudio.com/insiders/)
2. Install [Ollama](https://ollama.com/download)
3. Pull GPT-OSS model: `ollama pull gpt-oss:20b`
4. Start Ollama: `ollama serve`

### Setup

1. **Extension**: Install/enable the `alita-language-tools` extension
2. **Runtime** (PowerShell):
   ```powershell
   $env:LLM_MODEL="ollama:gpt-oss:20b"
   $env:OLLAMA_HOST="http://127.0.0.1:11434"
   python -m src.main
   ```
   Or Linux/macOS:
   ```bash
   export LLM_MODEL="ollama:gpt-oss:20b"
   export OLLAMA_HOST="http://127.0.0.1:11434"
   python -m src.main
   ```

### Usage in VS Code Insiders

- **Direct Ollama**: Command palette → `Alita: Invoke Agent (Ollama)`
  - Uses your local Ollama directly with model `gpt-oss:20b` (configured as default)
  - Streams response to a new Markdown document

- **Via Runtime**: Command palette → `Alita: Chat via Runtime (Stream)`
  - Posts to your runtime at `alita.runtime.host` (default: `http://127.0.0.1:8080`)
  - Streams response to Output channel "Alita Runtime Chat"

## DeepCode Integration

DeepCode orchestration is wired into the runtime with an in‑memory pub/sub event bus. A stub client is bundled; set environment to enable a real HTTP client.

### Environment

```bash
export DEEPCODE_API_URL=https://deepcode.mycompany.com
export DEEPCODE_API_KEY=your_key
export DEEPCODE_TIMEOUT_S=60
# Where to persist the latest proposal for retrieval:
export DEEPCODE_LATEST_PATH=./logs/deepcode_latest.json
```

### Endpoints

- `POST /deepcode/request`
  - Body: `{ "task_kind": "analyze" | "text2backend" | ..., "requirements"?: string, "repo_path"?: string }`
  - Returns: `{ status: "accepted", request: {...} }`

- `GET /deepcode/latest`
  - Returns the last successful proposal (plan, references, diffs, tests, docs, validation)

- `POST /deepcode/apply`
  - Body: `{ "paths"?: string[] }`
  - Delegates to orchestrator apply (if enabled)

### VS Code Commands

- `Alita: DeepCode — Analyze Workspace`
- `Alita: DeepCode — Generate From Prompt`

Both post to `/deepcode/request` on `alita.runtime.host`.

### Configuration

Extension defaults (can be customized in VS Code settings):

- `alita.ollama.host`: `http://127.0.0.1:11434`
- `alita.ollama.model`: `gpt-oss:20b`
- `alita.runtime.host`: `http://127.0.0.1:8080`

### Troubleshooting

If you get no response:
- **Runtime**: Check `curl http://127.0.0.1:8080/health` and runtime logs
- **Ollama**: Test directly with `curl -X POST http://127.0.0.1:11434/api/chat -H "Content-Type: application/json" -d '{"model": "gpt-oss:20b", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'`
- **Model**: Verify with `ollama list` or `ollama pull gpt-oss:20b`

### VS Code Extensions

This repository bundles several VS Code extensions under `extensions/`.
To build and test the new **Alita Language Tools** extension:

```bash
cd extensions/alita-language-tools
npm install
npm run compile
npm test
```

The extension exposes `alita.search` and `alita.skillset` commands. The existing extensions can be built and tested in a similar manner.

## MCP Server Installation Links

VS Code can install MCP server definitions directly via special links:

- [Sample install](vscode:mcp/install?url=https://example.com/mcp.json)

## LLM Fallback & Local Model Configuration

Set `LLM_MODEL=auto` to enable automatic provider selection.

Order of dynamic `LLM_MODEL=auto` preference (runtime selection):

1. Local Ollama (if reachable & `OLLAMA_MODEL` set)
2. Local Hugging Face direct model (`hf:<model_id>` or `LOCAL_MODEL_PATH`)
3. Azure OpenAI (env vars present)
4. OpenAI / Anthropic / Gemini (if configured)
5. Internal Super Alita fallback client / mock

Core environment variables (see `docs/02_developer_secrets.md` for full matrix):

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLM_MODEL` | Target model name or `auto` | `mock` |
| `SUPER_ALITA_BASE_URL` | Base URL for local adapter | `http://127.0.0.1:8080` |
| `SUPER_ALITA_MODEL` | Model name passed to adapter | `gpt-oss-20b-4bit` |
| `SUPER_ALITA_API_KEY` | Optional bearer token | (unset) |
| `OLLAMA_HOST` | Ollama daemon host | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | Ollama model tag | (unset) |
| `LOCAL_MODEL_PATH` | Local HF model directory | (unset) |

Telemetry events emitted:

- `llm_fallback` when Super Alita fallback client is selected
- `performance_metric` with `metric=llm_stream_duration_s` per streamed turn

Example `.env` (mixed local + cloud):

```dotenv
LLM_MODEL=auto
GEMINI_API_KEY=your_key_here   # optional; if absent will fallback
SUPER_ALITA_BASE_URL=http://127.0.0.1:8080
SUPER_ALITA_MODEL=gpt-oss-20b-4bit
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b
LOCAL_MODEL_PATH=models/Meta-Llama-3-8B-Instruct
```

Force specific providers explicitly:

```dotenv
# Force internal adapter
LLM_MODEL=super-alita

# Force Ollama (bypass cloud)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b

# Force direct HF model load
LLM_MODEL=hf:meta-llama/Meta-Llama-3-8B-Instruct
```

### Ollama Quick Start

1. Install from https://ollama.com/download
2. Pull a model: `ollama pull llama3.1:8b`
3. (Optional) Confirm: `curl http://127.0.0.1:11434/api/tags`
4. Set env: `OLLAMA_MODEL=llama3.1:8b LLM_PROVIDER=auto`
5. Use VS Code command: `Alita: Invoke Agent (Ollama)`

The extension settings `alita.ollama.host` and `alita.ollama.model` can override environment variables for editor-centric workflows.

### Local Hugging Face Model Direct Load

Download:
```bash
python scripts/download_model.py --model meta-llama/Meta-Llama-3-8B-Instruct --output models/Meta-Llama-3-8B-Instruct
```
Run the OpenAI-compatible adapter (optional, alternative to direct load):
```bash
python src/local_adapter_server.py --model-path models/Meta-Llama-3-8B-Instruct --device cuda
```
Or let the runtime load directly with `LLM_MODEL=hf:meta-llama/Meta-Llama-3-8B-Instruct`.

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

## Alita Developer Experience (DX) Kit

An integrated blueprint for AI-native development combining:

- Semantic Kernel backend agent (`backend/semantic_kernel_agent.py`)
- VS Code language tools extension with telemetry + LSP & Ollama integration (`extensions/alita-language-tools`)
- Experimental WASM component scaffold (`wasm/calculator`) now simplified to `add(a,b)`
- Developer guides: architectural overview, refactoring, testing, secrets (`docs/01_architectural_overview.md`, `docs/02_refactoring_guide.md`, `docs/03_testing_guide.md`, `docs/02_developer_secrets.md`)
- S-Tier evolution roadmap (`docs/08_s_tier_evolution_roadmap.md`) outlining predictive execution, multi-agent swarm, PEFT adapters, and WASM code radar.
* Swarm orchestrator (`backend/agent_orchestrator.py`) – OpenAI-compatible multi-agent execution (no Azure dependency)
* Fine-tuning scaffold (`backend/fine_tuning/train_adapter.py`) – LoRA adapter training from feedback JSON
* Predictive manager v2 (multi-action speculative cache, size/eviction policy)
* Code radar metrics (line length, complexity heuristic, nesting, duplication) via WASM module
* Extended docker-compose services: context-server, swarm, finetune (profiles)
* `scripts/setup.sh` bootstrap script

### Quick Start (DX Kit Extras)

Backend agent (auto-select Ollama if available, else Azure/OpenAI):

```bash
uvicorn backend.semantic_kernel_agent:app --reload --port 5001
```

WASM calculator (build & extract component):

```bash
cd wasm/calculator
cargo build --target wasm32-unknown-unknown --release
# Optional: convert to component if using component model toolchain
# wasm-tools component new target/wasm32-unknown-unknown/release/calculator.wasm -o calculator.wasm

Place `calculator.wasm` under `extensions/alita-language-tools/out/src/` to enable the `Alita: Run WASM Calculator` command (or adjust build copy step).
```

Extension adds commands:

- `Alita: Invoke Agent (Ollama)` – single prompt invocation against local model
- `Alita: Run WASM Calculator` – calls simplified `add(a,b)` in WASM module

Worker scaffold still includes `src/worker.ts` for future advanced component bindings.

See docs for deeper patterns and roadmap enhancements.
