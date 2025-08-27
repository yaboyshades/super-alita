# Super Alita Development Instructions

**ALWAYS follow these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

**Environment Setup (Required First Steps):**
- Create environment file: `cp .env.example .env`
- Install Python dependencies: `pip install -r requirements.txt -r requirements-test.txt` -- takes 5 minutes. **NEVER CANCEL**. Set timeout to 10+ minutes.
- Start the development server: `uvicorn app:app --reload --port 8080`
- Health check: `curl http://127.0.0.1:8080/healthz`

**Building and Testing:**
- **CRITICAL**: Build times are minimal (Python-based), but dependency installation takes 5+ minutes
- Test the system: `python validate_deployment.py` -- takes 10 seconds. **NEVER CANCEL**. Set timeout to 60+ seconds.
- Run runtime tests: `PYTHONPATH=./src pytest -q tests/runtime/test_router_smoke.py` -- takes 5 seconds but **may fail due to syntax errors in codebase**
- **WARNING**: Many tests have syntax errors and import issues. Do not expect full test suite to pass.

**Code Quality:**
- Format with Black: `black . --check` -- takes 15 seconds. **NEVER CANCEL**. Set timeout to 120+ seconds.
- Lint with Ruff: `ruff check .` -- takes 1 second. **NEVER CANCEL**. Set timeout to 60+ seconds.
- **WARNING**: Makefile has formatting issues (spaces instead of tabs) - use individual commands instead of `make` targets

**VS Code Extension Development:**
- Navigate to extensions: `cd extensions/alita-language-tools`
- Install Node.js dependencies: `npm ci` -- takes 40 seconds. **NEVER CANCEL**. Set timeout to 300+ seconds.
- Compile extension: `npm run compile` -- takes 3 seconds. **NEVER CANCEL**. Set timeout to 180+ seconds.

## Validation Scenarios

**ALWAYS test actual functionality after making changes:**

1. **Basic System Health:**
   ```bash
   # Start server and test health
   uvicorn app:app --reload --port 8080 &
   sleep 5
   curl http://127.0.0.1:8080/healthz
   # Expected: {"status":"healthy","components":{"event_bus":{"status":"ok"},"ability_registry":{"status":"ok"},"kg":{"status":"ok"},"llm":{"status":"ok"}}}
   ```

2. **Tool Catalog Validation:**
   ```bash
   curl http://127.0.0.1:8080/tools/catalog
   # Expected: JSON array with tools like reug_start_turn, fs_read, fs_write, etc.
   ```

3. **Streaming Chat Integration:**
   ```bash
   curl -X POST http://127.0.0.1:8080/v1/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello", "session_id": "test"}' \
     --no-buffer
   # Expected: Server-sent events with TaskStarted and LLMChunk events
   ```

4. **Deployment Validation:**
   ```bash
   python validate_deployment.py
   # Expected: "🎉 ALL TESTS PASSED - Super Alita is ready for deployment!"
   ```

## Key Components

**Critical Files:**
- `app.py` - Main FastAPI application entry point
- `src/main.py` - Core application factory with plugin loading
- `src/reug_runtime/router.py` - Streaming orchestration engine
- `.env.example` - Environment template (copy to `.env`)
- `validate_deployment.py` - System validation script

**Dependencies:**
- Python 3.11+ required
- Node.js 20+ for VS Code extensions
- Redis optional (uses in-memory fallback)
- FastAPI for web framework
- Transformers and PyTorch for AI models

**Architecture:**
- Event-driven AI agent system
- MCP (Model Context Protocol) integration
- Plugin-based modularity
- Streaming single-turn agent responses
- Tool execution framework

## Project Structure

```
/
├── src/                    # Core Python source code
│   ├── main.py            # Application factory
│   ├── reug_runtime/      # Streaming orchestration
│   ├── core/              # Event bus and neural architecture
│   └── plugins/           # Modular plugin system
├── extensions/            # VS Code language tools
│   └── alita-language-tools/
├── tests/                 # Test suites (many have syntax errors)
├── app.py                 # FastAPI entry point
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── validate_deployment.py # System validation
```

## Common Issues and Workarounds

**Test Suite Issues:**
- Many tests have syntax errors (`from __future__ import annotations` placement)
- Import errors for missing modules (`libcst` not installed by default)
- Use `python validate_deployment.py` instead of full pytest suite for validation

**Makefile Issues:**
- Formatting errors (spaces instead of tabs)
- Use direct commands instead: `pip install -r requirements.txt -r requirements-test.txt`

**Development Dependencies:**
- libcst missing by default (used for MCP tools) - install with `pip install libcst` if needed
- Some VS Code extension dependencies have security warnings - normal for development

**Environment Variables:**
- System works with minimal `.env` setup (just copy from `.env.example`)
- Default mock LLM provider is functional for testing
- Redis optional (falls back to in-memory event bus)

## Performance Expectations

- **Dependency installation**: 5 minutes (heavy ML dependencies)
- **Server startup**: 2-3 seconds
- **Health check**: < 1 second
- **Deployment validation**: 10 seconds
- **Code formatting**: 15 seconds
- **VS Code extension build**: 40 seconds for deps + 3 seconds compile

**NEVER CANCEL commands that take expected time**. Always set appropriate timeouts:
- Dependency installation: 600+ seconds
- Any build/test command: 300+ seconds minimum

## Getting Started Checklist

1. **Environment Setup:**
   - [ ] `cp .env.example .env`
   - [ ] `pip install -r requirements.txt -r requirements-test.txt` (5 min)

2. **Validation:**
   - [ ] `python validate_deployment.py` (should pass all 7 tests)
   - [ ] `uvicorn app:app --reload --port 8080` (server starts)
   - [ ] `curl http://127.0.0.1:8080/healthz` (returns healthy status)

3. **Development Workflow:**
   - [ ] Make code changes
   - [ ] `python validate_deployment.py` (verify no regressions)
   - [ ] Test specific endpoints manually with curl
   - [ ] `ruff check .` and `black . --check` (code quality)

**Always validate changes with real functionality tests, not just build success.**

---

## Advanced Development (Original Documentation)

For deeper architectural patterns and advanced development, reference the existing documentation:

- **Event-driven neural architecture** with Redis/Memurai event bus (`src/core/event_bus.py`)
- **MCP (Model Context Protocol)** for tool creation and VS Code integration  
- **Atoms/Bonds cognitive fabric** - all outputs are structured as atoms with deterministic UUIDs
- **Plugin-based modularity** - all components inherit from `PluginInterface`

### Code Standards
- **Black 88 chars**, Ruff with selected rules (`pyproject.toml`)
- **Type hints everywhere**; prefer Pydantic models over dataclasses for events
- **pathlib.Path** not `os.path`; assume Windows paths in MCP tools
- **AST/libcst** transforms for refactoring, never regex patching
- **pytest** with parametrized edge cases; no print statements in tests

### Event System Patterns
```python
# Event creation (use keyword args, never positional dicts)
from src.core.events import create_event
event = create_event("cognitive_turn", turn_data=data, confidence=0.95)

# Async event handling
@pytest.mark.asyncio  # Required for all async tests
async def test_event_flow():
    # Use timezone-aware timestamps
    timestamp = datetime.now(timezone.utc)
```

