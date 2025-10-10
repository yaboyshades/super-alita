# Super Alita

## Constitution and Addendum

- [Project Constitution](.github/CONSTITUTION.md)
- [Codecraft Addendum (CMA v5.3)](docs/cma-v5.3/codecraft-addendum.md)

Advanced, event-driven AI agent system with modular plugins, MCP integration, knowledge graph, streaming orchestration, and adaptive LLM routing.

Production-ready architecture with:

- Streaming orchestration
- Rich telemetry + MCP broadcast
- Fallback LLM routing (Gemini → local Super Alita → mock / local HF model)
- Knowledge graph + cognitive fabric (Atoms / Bonds)
- Modular plugin system
- OpenAI-compatible local adapter option

## Constitution and Addendum

Super Alita follows a constitutional contract plus a Codecraft addendum that govern every change:

- [Super-Alita Constitutional Framework](.github/CONSTITUTION.md)
- [Codecraft Addendum (CMA v5.3)](docs/cma-v5.3/codecraft-addendum.md)

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

### Default Workflow: Specification-Driven Development (SDD)

SDD (Specification-Driven Development) is the default pipeline for feature work. Start every initiative with `specify → plan → tasks` using the constitutional gates that ship with `src/sdd/`.

- Use the enhanced CLI: `python -m src.sdd.sdd_cli specify "..."` / `plan` / `tasks`.
- Shell helpers in `scripts/lib/constitutional-gates.sh` and `scripts/lib/sdd-common.sh` keep bespoke automation aligned with the CMA rules. `sdd-common.sh` exposes `log_json`, which drops `message=` entries (with a stderr warning) so structured logs stay machine readable.
- Run `scripts/smoke_sdd_specify.py` against a running server to confirm the FastAPI router and gates are healthy before coding.

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

User/API guide and deployment:

- User Guide: see `docs/USER_GUIDE.md` for the unified API (`/api/v1/query`), auth, rate limits, and CLI usage.
- Production Deployment: see `PRODUCTION_DEPLOYMENT_GUIDE.md` for environment hardening and Redis-backed rate limiting.
- Chat UI: open `http://localhost:8080/` to use the built-in chat interface. See `docs/CHAT_UI.md` for SSE details and endpoints.
- REUG v12.2+ Implementation Guide: see `docs/reug_v12_2_implementation_guide.md` for the plan to extend the KG‑enhanced LADDER system with advanced REUG modules and deployment.

### 4. Run tests

Super Alita provides comprehensive Jest-equivalent testing using pytest and related packages:

```bash
# Basic test run (Jest equivalent: jest)
make test
# or manually
pytest -q

# Coverage reporting (Jest equivalent: jest --coverage)
pytest --cov=src --cov-report=term-missing

# Parallel execution (Jest equivalent: built-in parallelism)
pytest -n auto

# Watch mode (Jest equivalent: jest --watch)
ptw

# Snapshot testing (Jest equivalent: Jest snapshots)
pytest --snapshot-update

# Run Jest-like pattern examples
python demo_jest_for_python.py
```

For complete Jest-to-Python testing guide, see [docs/jest_for_python_guide.md](docs/jest_for_python_guide.md).

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

# Enable Claude Sonnet 3.5 (Anthropic)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=claude-3-sonnet
```

### Claude Sonnet 3.5 Quick Start

1. Get your Anthropic API key and set `ANTHROPIC_API_KEY` in your `.env`.
2. Set `LLM_MODEL=claude-3-sonnet` to use Claude Sonnet 3.5 for all clients.
3. Claude Sonnet 3.5 will now be available in all model selection UIs and API calls.

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

## SDD Quickstart (Scripts)

Use the synchronous CLI wrappers in `src.sdd.sdd_cli` to drive the Spec-Driven Development workflow from the terminal.

### 1. Specify

```bash
python -m src.sdd.sdd_cli specify "Implement streaming telemetry audit trail" \
  --context '{"priority": "high", "owner": "platform"}' --format json
```

Sample output:

```json
{
  "success": true,
  "specification": "## Feature: Streaming telemetry audit trail\n- capture events for every tool call\n- persist summarized metrics per session",
  "feature_id": "feat-streaming-telemetry",
  "feature_path": "specs/feat-streaming-telemetry/spec.md",
  "analysis_results": {
    "mangle_enhanced": true,
    "summary": "Identified telemetry bus hooks and fallback strategies."
  },
  "constitutional_compliance": {
    "article_1": {
      "article": "Article 1 — Safety",
      "compliant": true,
      "score": 0.92,
      "violations": [],
      "suggestions": []
    }
  },
  "overall_compliance_score": 0.89,
  "compliance_threshold_met": true,
  "next_steps": [
    "Review generated specification with product owner."
  ],
  "timestamp": "2024-XX-XXTXX:XX:XX.XXXXXX"
}
```

### 2. Plan

```bash
python -m src.sdd.sdd_cli plan feat-streaming-telemetry --format json
```

Sample output:

```json
{
  "success": true,
  "implementation_plan": "1. Instrument event bus hooks\n2. Persist audit summaries\n3. Expose monitoring endpoint",
  "plan": "1. Instrument event bus hooks\n2. Persist audit summaries\n3. Expose monitoring endpoint",
  "plan_path": "specs/feat-streaming-telemetry/plan.md",
  "supporting_documents": [
    "specs/feat-streaming-telemetry/architecture.md"
  ],
  "analysis_results": {
    "dependency_analysis": {
      "total_dependencies": 4,
      "critical_modules": [
        "src/telemetry/event_bus.py",
        "src/reug_runtime/router.py"
      ]
    }
  },
  "constitutional_compliance": {
    "article_3": {
      "article": "Article 3 — Reliability",
      "compliant": true,
      "score": 0.95,
      "violations": [],
      "suggestions": []
    }
  },
  "overall_compliance_score": 0.93,
  "compliance_threshold_met": true,
  "technology_recommendations": [
    "FastAPI background tasks",
    "Redis stream for durable telemetry"
  ],
  "architecture_decisions": [
    "Record telemetry snapshots before tool dispatch",
    "Archive summaries nightly via batch worker"
  ],
  "next_steps": [
    "Confirm retention policy with reliability team."
  ],
  "timestamp": "2024-05-10T14:23:04.118209"
}
```

### 3. Tasks

```bash
python -m src.sdd.sdd_cli tasks feat-streaming-telemetry --format json
```

Sample output:

```json
{
  "success": true,
  "tasks_breakdown": "### Task List\n1. Wire telemetry interceptors\n2. Persist audit artifacts\n3. Ship monitoring endpoint",
  "tasks_path": "specs/feat-streaming-telemetry/tasks.md",
  "tasks": [
    {
      "id": "task-1",
      "title": "Wire telemetry interceptors",
      "description": "Capture STATE_TRANSITION and Ability events before dispatch.",
      "priority": "critical",
      "estimated_hours": 6,
      "dependencies": [],
      "acceptance_criteria": [
        "All router events emit telemetry payloads",
        "Unit tests validate event capture"
      ],
      "constitutional_requirements": [
        "Article 3 — Reliability"
      ]
    },
    {
      "id": "task-2",
      "title": "Persist audit artifacts",
      "description": "Store summarized telemetry in Redis with 7-day retention.",
      "priority": "high",
      "estimated_hours": 5,
      "dependencies": [
        "task-1"
      ],
      "acceptance_criteria": [
        "Redis persistence verified via integration test"
      ],
      "constitutional_requirements": [
        "Article 4 — Accountability"
      ]
    }
  ],
  "analysis_results": {
    "prioritization": {
      "critical_path": [
        "task-1",
        "task-2"
      ]
    }
  },
  "constitutional_compliance": {
    "article_4": {
      "article": "Article 4 — Accountability",
      "compliant": true,
      "score": 0.9,
      "violations": [],
      "suggestions": []
    }
  },
  "overall_compliance_score": 0.9,
  "compliance_threshold_met": true,
  "estimated_total_hours": 18,
  "critical_path": [
    "task-1",
    "task-2"
  ],
  "next_steps": [
    "Schedule load test for sustained event throughput."
  ],
  "timestamp": "2024-05-10T14:23:52.002771"
}
```

## Unified Consciousness Bootstrap

To launch the unified consciousness runtime from a single command, ensure dependencies are installed and run:

```bash
python unified_consciousness.py --config configs/unified_consciousness.yaml
```

Use `--run-forever` to keep monitoring loops active or execute `./start_unified_consciousness.sh` for Redis auto-start and environment setup.
To launch the advanced consciousness stack with all exploratory capabilities enabled, use:

```bash
./start_advanced_consciousness.sh
```

This script boots using `configs/unified_consciousness_advanced.yaml`, enabling tribal knowledge extraction, living architecture, temporal forecasting, philosophical diagnostics, and the architectural immune system.
