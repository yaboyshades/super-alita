# Super Alita Development Instructions

**ALWAYS follow these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

**Environment Setup (Required First Steps):**

- Create environment file: `cp .env.example .env`
- Install Python dependencies: `pip install -r requirements.txt -r requirements-test.txt` -- takes 5 minutes. **NEVER CANCEL**. Set timeout to 10+ minutes.
- Start the development server: `uvicorn app:app --reload --port 8080`
- Health check: `curl http://127.0.0.1:8080/healthz`

**Enhanced Consensus Algorithms (NEW):**

- **Enhanced Consensus Provider**: `src/abilities/enhanced_consensus_ability.py` with 5 consensus methods
- **Consensus Methods**: simple_vote, weighted_vote (default), confidence_based, semantic_similarity, ensemble_ranking
- **Provider Constructor**: `EnhancedConsensusProvider(config: Dict[str, Any] | None)` per `src/abilities/enhanced_consensus_ability.py:60`
- **Default Method**: "weighted_vote" per `src/main.py:1660` (not ensemble_ranking)
- **Config Setup**: App startup passes dict with base_url/model/timeout per `src/main.py:1575`
- **Direct Ollama Integration**: Bypasses complex dependencies with `http://localhost:11434/v1` API
- **Consensus Testing**: Use `/tools/catalog` to verify enhanced_consensus tool registration

Quick HTTP usage (registered tool id: `deepconf_consensus`):

```
curl -X POST http://127.0.0.1:8080/ability/execute/deepconf_consensus \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize three benefits of unit testing",
    "method": "weighted_vote",
    "num_samples": 4,
    "temperature": 0.7,
    "max_tokens": 256,
    "confidence_threshold": 0.7,
    "temperature_range": 0.2
  }'
```

**REUG Streaming Architecture:**

- **Streaming Router**: `src/reug_runtime/router.py` - single-turn orchestrator with tool execution
- **Event-Driven**: TaskStarted → LLMChunk → AbilityCalled → AbilitySucceeded/Failed → TaskSucceeded
- **SSE Endpoint**: `POST /v1/chat/stream` (text/event-stream) per `src/reug_runtime/router.py:206`
- **Tool Streaming**: Starts `execute_turn` iterator per `src/reug_runtime/router_tools.py:236`
- **Connection Issues**: Known "peer closed connection" errors during tool execution - use debugging tools
- **Tool Execution**: All tools execute via ability registry with timeout controls

Canonical endpoints (repo-accurate):

- `POST /v1/chat/stream` (SSE text/event-stream only, no JSON endpoint)
- `POST /tools/reug_start_turn` with `{"message":"...", "session_id":"..."}` → `{"run_id": "...", "stream_begun": true}`
- `POST /tools/reug_stream_next` with `{"run_id":"..."}` → `{"chunks":[...], "finished": bool}`
- `POST /ability/execute/deepconf_consensus` (direct tool execution)
- Tool catalog: Static schema + dynamic merge per `src/reug_runtime/router_tools.py:151,199`

**Building and Testing:**

- **CRITICAL**: Build times are minimal (Python-based), but dependency installation takes 5+ minutes
- Test the system: `python validate_deployment.py` -- takes 10 seconds. **NEVER CANCEL**. Set timeout to 60+ seconds.
- Run runtime tests: `PYTHONPATH=./src pytest -q tests/runtime/test_router_smoke.py` -- takes 5 seconds but **may fail due to syntax errors in codebase**
- **WARNING**: Many tests have syntax errors and import issues. Do not expect full test suite to pass.

**Code Quality:**

- Format with Black: `black . --check` -- takes 15 seconds. **NEVER CANCEL**. Set timeout to 120+ seconds.
- Lint with Ruff: `ruff check .` -- takes 1 second. **NEVER CANCEL**. Set timeout to 60+ seconds.
- **WARNING**: Makefile has formatting issues (spaces instead of tabs) - use individual commands instead of `make` targets

**Python Development Tools:**

- **Python Extensions Required**: ms-python.python, ms-python.vscode-pylance, ms-python.black-formatter, charliermarsh.ruff
- **Virtual Environment**: Always use `.venv\Scripts\python.exe` (configured as defaultInterpreterPath)
- **Type Checking**: Pylance in strict mode, mypy with `--strict --ignore-missing-imports`
- **Testing**: pytest with asyncio support, coverage reporting enabled
- **Jupyter**: Full notebook support with kernel management and rich outputs

**Python-Specific VS Code Tasks:**

- **Ruff Check/Fix**: `python -m ruff check .` and `python -m ruff check --fix .`
- **Black Format**: `python -m black .` with 79-character line length
- **Python Compile Check**: `python -m py_compile` for syntax validation
- **MCP Python Tools**: `format_and_lint_selection` tool for automated code quality

**Python Configuration Patterns:**

- **pyproject.toml**: Black (88 chars), Ruff (select E,W,F,I,B,C4,UP,SIM,PT), mypy strict mode
- **Import Organization**: isort with Black profile, trailing commas, 79-char lines
- **Environment**: Python 3.11+, virtual environment required, PYTHONPATH properly configured

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
- `src/reug_runtime/router.py` - Streaming orchestration engine with tool execution pipeline
- `src/abilities/enhanced_consensus_ability.py` - Advanced consensus algorithms with Ollama integration
- `validate_deployment.py` - System validation script (always use this)
- `src/vscode_integration/agent_mcp_server.py` - MCP server for VS Code integration
- `src/sandbox/exec_sandbox.py` - Secure code execution (mandatory for dynamic code)
- `.env.example` - Environment template (copy to `.env`)

**Enhanced Consensus Architecture (NEW):**

- **Consensus Methods**: 5 sophisticated algorithms for multi-response aggregation
- **Provider Constructor**: `EnhancedConsensusProvider(config: Dict[str, Any] | None)` - config-based, not kwargs
- **Config Setup**: App startup passes dict with base_url/model/timeout per `src/main.py:1575`
- **Default Method**: "weighted_vote" per `src/main.py:1660` (contract and executor default)
- **Direct Ollama API**: Bypasses complex ML dependencies with simple HTTP integration
- **Tool Registration**: Contract at `src/main.py:1616`, executor at `src/main.py:1631`, registration at `src/main.py:1635`
- **Tool Catalog**: Static schema (`src/reug_runtime/router_tools.py:151`) + dynamic merge (`router_tools.py:199`)

**Streaming Infrastructure:**

- **REUG Router**: Single-turn orchestrator via `execute_turn` with async iterator storage
- **SSE Endpoint**: `POST /v1/chat/stream` uses `sse_transformer(event_generator)` per `src/reug_runtime/router.py:206`
- **Tool Streaming**: Stores generators in `_STREAMS[run_id]` per `src/reug_runtime/router_tools.py:236`
- **Event Flow**: TaskStarted → LLMChunk → AbilityCalled → AbilitySucceeded/Failed → TaskSucceeded
- **Connection Management**: Known timeout issues during tool execution - requires debugging

**Dependencies:**

- Python 3.11+ required
- Node.js 20+ for VS Code extensions
- Redis optional (uses in-memory fallback)
- FastAPI for web framework
- Transformers and PyTorch for AI models

**Architecture:**

- Event-driven AI agent system with enhanced consensus capabilities
- MCP (Model Context Protocol) integration for VS Code tooling
- Plugin-based modularity with tool execution registry
- Streaming single-turn agent responses with timeout management
- Tool execution framework with direct Ollama API integration
- Neural-symbolic bridge for memory-enhanced reasoning
- REUG state machine orchestration with formal verification patterns

## Project Structure

```
/
├── src/                          # Core Python source code
│   ├── main.py                  # Application factory
│   ├── reug_runtime/            # Streaming orchestration
│   ├── core/                    # Event bus and neural architecture
│   ├── plugins/                 # Modular plugin system
│   ├── vscode_integration/      # MCP servers and VS Code tools
│   │   ├── agent_mcp_server.py  # External MCP server
│   │   └── builtin_mcp_provider.ts # Built-in MCP extension
│   └── sandbox/                 # Secure code execution
├── extensions/                   # VS Code language tools
│   └── alita-language-tools/
├── mcp_server/                   # Standalone MCP implementation
├── tests/                        # Test suites (many have syntax errors)
├── app.py                        # FastAPI entry point
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
└── validate_deployment.py       # System validation
```

## Common Issues and Workarounds

**Test Suite Issues:**

- Many tests have syntax errors (`from __future__ import annotations` placement)
- Import errors for missing modules (`libcst` not installed by default)
- Use `python validate_deployment.py` instead of full pytest suite for validation

**Makefile Issues:**

- Formatting errors (spaces instead of tabs)
- Use direct commands instead: `pip install -r requirements.txt -r requirements-test.txt`

**REUG Streaming Connection Issues:**

- Connection errors ("peer closed connection") are known issues during tool execution
- Use debugging tools in `src/reug_runtime/` for connection troubleshooting
- Streaming infrastructure has timeout controls but may drop connections under load
- System maintains 80% functionality despite streaming issues

**Enhanced Consensus Validation:**

- Test consensus algorithms: `curl http://127.0.0.1:8080/tools/catalog` (verify enhanced_consensus tool)
- Ability endpoint: `POST /ability/execute/deepconf_consensus` (see example above)
- Direct provider tests: `test_enhanced_consensus.py`, `test_enhanced_consensus_comprehensive.py`, `test_consensus_direct_enhanced.py`
- REUG path test: `test_consensus_via_reug.py`
- Validate Ollama integration: Check `http://localhost:11434/v1` API accessibility
- Confidence scoring: Test multiple consensus methods (simple_vote, weighted_vote, etc.)
- Ensemble ranking: Verify multi-factor scoring combining confidence, length, specificity

**Environment Variables:**

- System works with minimal `.env` setup (just copy from `.env.example`)
- Default mock LLM provider is functional for testing
- Redis optional (falls back to in-memory event bus)
- **Security modes**: Set `SUPER_ALITA_MODE` to `shadow` (plan), `act` (sandboxed), or `batch` (replay)

**Sandbox Security:**

- All dynamic execution must use `src/sandbox/exec_sandbox.py` - never raw `eval/exec`
- Process management via `src/core/proc.py` (no `shell=True`)
- YAML operations via `src/core/yaml_utils.py` (never unsafe loading)
- Policy guards cannot be bypassed - security is paramount

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
   - [ ] Validate enhanced consensus: `curl http://127.0.0.1:8080/tools/catalog`
   - [ ] `ruff check .` and `black . --check` (code quality)

**Always validate changes with real functionality tests, not just build success.**

---

## Advanced Development (Original Documentation)

For deeper architectural patterns and advanced development, reference the existing documentation:

- **Event-driven neural architecture** with Redis/Memurai event bus (`src/core/event_bus.py`)
- **MCP (Model Context Protocol)** for tool creation and VS Code integration
- **Atoms/Bonds cognitive fabric** - all outputs are structured as atoms with deterministic UUIDs
- **Plugin-based modularity** - all components inherit from `PluginInterface`

### Enhanced Consensus Algorithms (Advanced)

Super Alita implements **5 sophisticated consensus methods** for multi-response aggregation:

```python
# Enhanced Consensus Provider (src/abilities/enhanced_consensus_ability.py)
from src.abilities.enhanced_consensus_ability import EnhancedConsensusProvider, ConsensusMethod

# Canonical constructor: config dict, not kwargs
provider = EnhancedConsensusProvider(config={
    "base_url": "http://localhost:11434/v1",
    "model_name": "gpt-oss:20b",
    "timeout": 60
})

result = await provider.consensus_sampling(
    prompt="Your question",
    num_samples=3,
    method="weighted_vote",  # Default method per src/main.py:1660
    confidence_threshold=0.7
)
```

**Consensus Method Details:**

- **simple_vote**: Basic majority voting for identical responses
- **weighted_vote**: Confidence-weighted voting with score aggregation (DEFAULT)
- **confidence_based**: Highest confidence above threshold with fallback
- **semantic_similarity**: Word overlap clustering for related responses
- **ensemble_ranking**: Multi-factor scoring (confidence + length + specificity + uniqueness)

**Direct Ollama Integration:**

- Uses `http://localhost:11434/v1/chat/completions` API directly
- Bypasses complex transformer dependencies with simple HTTP calls
- Temperature diversity for sample generation (±0.2 range)
- Confidence estimation based on response characteristics
- Built-in timeout and retry mechanisms for robust operation

### REUG Streaming Architecture (Advanced)

The **REUG runtime router** (`src/reug_runtime/router.py`) implements single-turn streaming:

```python
# Streaming orchestration with tool execution
# Pattern: LLM chunks → tool calls → tool results → final answer
# Tools execute via ability_registry
# SSE endpoint uses sse_transformer(event_generator) per src/reug_runtime/router.py:206
```

**Event Flow Architecture:**

```
TaskStarted → LLMChunk(s) → AbilityCalled → AbilitySucceeded/Failed → TaskSucceeded
```

**Key Components:**

- **execute_turn**: Async generator that orchestrates single-turn workflow
- **Tool Streaming**: `_STREAMS[run_id] = execute_turn().aiter()` per `src/reug_runtime/router_tools.py:236`
- **SSE Endpoint**: `POST /v1/chat/stream` (text/event-stream only, no JSON endpoint)
- **Tool Mode**: `/tools/reug_start_turn` and `/tools/reug_stream_next` for JSON chunk streaming
- **Error Handling**: Tool failures captured with span IDs for traceability

**Known Issues:**

- Connection drops during tool execution ("peer closed connection")
- Timeout management needed for long-running consensus operations
- Streaming infrastructure maintains 80% functionality despite connection issues

### Canonical Endpoints Reference

Based on exact codebase locations per `src/main.py` and `src/reug_runtime/`:

```python
# Health Endpoints
GET /healthz                              # src/main.py:1336
GET /health                               # src/main.py:1348

# Agent Conversation
POST /v1/chat/stream                      # src/reug_runtime/router.py:206 (SSE only)

# Tool Discovery & Streaming
GET /tools/catalog                        # src/reug_runtime/router_tools.py:157
POST /tools/reug_start_turn              # src/reug_runtime/router_tools.py:210
POST /tools/reug_stream_next             # src/reug_runtime/router_tools.py:239

# Direct Tool Execution
POST /ability/execute/{tool_id}           # src/main.py:1841
# tool_id for consensus: "deepconf_consensus"

# Ability Registration Facts
# Contract: src/main.py:1616 (includes method enum, confidence fields)
# Executor: src/main.py:1631 (args dict → provider.consensus_sampling)
# Registration: src/main.py:1635 (register_tool with contract & executor)
```

**Tool Catalog Schema Notes:**

- **Static Schema**: `src/reug_runtime/router_tools.py:151` (omits method/confidence fields)
- **Dynamic Merge**: `src/reug_runtime/router_tools.py:199` (augments from registry)
- **Full Contract**: `src/main.py:1616` (complete method/defaults appear here)

### MCP Integration Patterns

Super Alita includes **MCP servers for code-quality tools**, not consensus tools:

```python
# MCP Server (src/mcp_server/server.py)
# Code-quality focused tools:
# - refactor/apply-result-pattern
# - format-and-lint
# - find-missing-docstrings

# VS Code MCP Wrapper (src/vscode_integration/agent_mcp_server.py)
# MCP glue layer for VS Code integration - not a consensus API surface
```

**MCP Layer Scope:**

- **In-tree MCP server**: Exposes code-quality tools (not consensus)
- **VS Code wrapper**: Provides MCP protocol glue for editor integration
- **Tool focus**: Refactoring, formatting, linting, documentation generation

### Plugin System Architecture

All plugins inherit from `PluginInterface` (`src/plugins/plugin_interface.py`):

```python
class MyPlugin(PluginInterface):
    async def initialize(self, event_bus: EventBus, **kwargs) -> bool:
        # Setup plugin with event bus
        await event_bus.subscribe("event_type", self.handle_event)
        return True

    async def process_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        # Process events and return results
        return {"processed": True, "data": event}
```

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

# Cognitive events for DTA 2.0 pipeline
cognitive_event = create_event("cognitive_turn_initiated",
    user_message="task", session_id="abc", conversation_id="123")

# Async event handling
@pytest.mark.asyncio  # Required for all async tests
async def test_event_flow():
    # Use timezone-aware timestamps
    timestamp = datetime.now(timezone.utc)
```

### Streaming Router Architecture

The **REUG runtime router** (`src/reug_runtime/router.py`) handles single-turn streaming:

```python
# Streaming orchestration with tool execution
# Pattern: LLM chunks → tool calls → tool results → final answer
# Tools execute via pp.state.ability_registry
# Compatible with <tool_call>, <tool_result>, <final_answer> tags
```

### Python Development Ecosystem

**Essential Extensions (.vscode/extensions.json):**

```json
{
  "recommendations": [
    "ms-python.python", // Core Python support
    "ms-python.vscode-pylance", // Language server
    "ms-python.black-formatter", // Code formatting
    "charliermarsh.ruff", // Fast linting
    "ms-python.mypy-type-checker", // Type checking
    "ms-toolsai.jupyter", // Notebook support
    "donjayamanne.python-extension-pack" // Productivity bundle
  ]
}
```

**Python Settings (.vscode/settings.json patterns):**

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
  "python.analysis.typeCheckingMode": "strict",
  "python.analysis.autoImportCompletions": true,
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit",
      "source.fixAll.ruff": "explicit"
    },
    "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "black-formatter.args": ["--line-length=79", "--target-version=py39"],
  "ruff.lint.run": "onType",
  "jupyter.alwaysTrustNotebooks": true
}
```

**Python MCP Tools Integration:**

```python
# Available via MCP: format_and_lint_selection tool
# Automated workflow: Ruff fix → Black format → return results
# Example usage in VS Code: Command Palette → "Format and Lint Selection"
```

### VS Code Tasks Integration

Use predefined VS Code tasks for consistent development workflow:

- **🚀 Start Super Alita Development Environment** - Multi-terminal orchestration
- **🔍 Full System Validation** - Runs `python validate_deployment.py`
- **🏥 Health Check** - Quick `curl http://127.0.0.1:8080/healthz`
- **🛠️ Tools Catalog Check** - Validates tool registration

Access via Command Palette (`Ctrl+Shift+P`) → "Tasks: Run Task"
