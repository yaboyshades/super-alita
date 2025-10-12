# Super Alita Workspace Instructions

Welcome to the Super Alita project!
This workspace runs in a dev container on **Ubuntu 24.04.2 LTS**.

## Overview

Super Alita is an advanced, event-driven AI agent system with modular plugins, MCP integration, knowledge graph, streaming orchestration, and adaptive LLM routing. The system features production-ready architecture with streaming orchestration, rich telemetry, fallback LLM routing, and a modular plugin system.

## Getting Started

### 1. Clone and Setup
```bash
git clone <repo-url>
cd super-alita
```

### 2. Environment Configuration
```bash
cp .env.example .env  # then set at least one provider key or local model config
```

### 3. Install Dependencies
Using Make (recommended):
```bash
make deps               # CPU defaults, includes torch CPU build
make lint              # optional code style check
```

Or manually:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ./.venv/Scripts/Activate.ps1  # Windows PowerShell
pip install -e .
```

### 4. Run the Development Server
```bash
make run
# or manually
python -m uvicorn src.main:app --reload --port 8080
```

### 5. Health Check
```bash
curl http://127.0.0.1:8080/healthz
```

### 6. Run Tests
```bash
make test
# or manually
pytest -q
```

## Useful Tools

- **Package management**: `apt`, `pip`
- **Version control**: `git`, `gh`
- **Container management**: `docker`, `kubectl`
- **Networking**: `ssh`, `scp`, `netstat`
- **Code quality**: `ruff`, `black`, `mypy`
- **Testing**: `pytest`

## Opening Webpages

To open a URL in the host's default browser:
```bash
"$BROWSER" <url>
```

## Directory Structure

- `/src` — Main source code (event-driven architecture, plugins, MCP integration)
- `/tests` — Unit and integration tests
- `/docs` — Comprehensive documentation
- `/scripts` — Utility and automation scripts
- `/cortex` — Cortex automation and workflow components
- `/mcp_server` — MCP (Model Context Protocol) server implementation
- `/examples` — Example implementations and demos
- `/tools` — Standalone tools and utilities
- `/config` — Configuration files
- `/prompts` — System prompts and templates
- `/schema` — Data schemas and validation
- `/reug` — REUG runtime components
- `/extensions` — Extensions and plugins
- `/docker` — Docker configuration

Each directory contains its own `INSTRUCTIONS.md` file for specific guidance.

## Key Features & Architecture

- **Event Bus**: Redis optional backend with real-time event streaming
- **MCP Integration**: VS Code integration with MCP server
- **Cognitive Fabric**: Atoms/Bonds system for knowledge representation
- **Plugin System**: Modular architecture with PluginInterface
- **LLM Routing**: Automatic fallback (Gemini → local Super Alita → mock)
- **Streaming**: Single-turn agent router with streaming orchestration
- **Telemetry**: Real-time telemetry broadcasting via MCP

## LLM Configuration

Set `LLM_MODEL=auto` to enable automatic provider selection.

Environment variables:
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` - For Gemini provider
- `OPENAI_API_KEY` - For OpenAI provider
- `ANTHROPIC_API_KEY` - For Claude provider
- `SUPER_ALITA_BASE_URL` - Local adapter URL (default: http://127.0.0.1:8080)

## Development Workflow

1. **Start with OpenSpec**: Run `openspec list` and `openspec list --specs` to review active changes and baseline capabilities. Create or update proposals under `openspec/changes/<change-id>/` before touching code; validate with `openspec validate <change-id> --strict`.
2. **Trace to Specs**: Read relevant capability specs in `openspec/specs/` and link tasks to modules (e.g., orchestrator, REUG runtime, plugins). Keep `tasks.md` synchronized with implementation progress.
3. **Code Style**: Run `ruff check .` and `black .`
4. **Testing**: Use `pytest -q` for quick tests
5. **Debugging**: Utilities available in `scripts/` directory
6. **Validation**: Run deployment checks with validation scripts

## Contact

For questions, open an issue or contact the maintainer.

---

**System Status**: PRODUCTION-READY
**Architecture**: Event-driven AI agent system
**Integration**: MCP + VS Code ready
