# Super Alita Agent System - Copilot Instructions

## Architecture Overview
Super Alita is a **self-evolving AI agent system** built on:
- **Event-driven neural architecture** with Redis/Memurai event bus (`src/core/event_bus.py`)
- **MCP (Model Context Protocol)** for tool creation and VS Code integration
- **Atoms/Bonds cognitive fabric** - all outputs are structured as atoms with deterministic UUIDs
- **Plugin-based modularity** - all components inherit from `PluginInterface`

## Key Workflows

### Development Environment
```bash
# Essential setup commands
pwsh .\Setup-MCP.ps1 -Bootstrap  # Initialize MCP server + VS Code integration
pwsh .\Setup-MCP.ps1 -Doctor     # Health check environment
python -m pytest                 # Run test suite
```

### MCP Tool Development
- Tools live in `mcp_server/src/mcp_server/tools/` (separate MCP project)
- Use `pwsh .\Setup-MCP.ps1 -AddTool YourTool` to scaffold new tools
- Test via VS Code Agent Mode: `MCP: Show Installed Servers` → verify `myCustomPythonAgent`

### Plugin Development
- All plugins extend `src/core/plugin_interface.py::PluginInterface`
- Must implement `name` property and `shutdown()` method
- Event emission: `create_event(event_type, **kwargs)` from `src/core/events.py`
- Register with event bus in `__init__`: `self.event_bus = event_bus`

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

## Code Standards
- **Black 88 chars**, Ruff with selected rules (`pyproject.toml`)
- **Type hints everywhere**; prefer Pydantic models over dataclasses for events
- **pathlib.Path** not `os.path`; assume Windows paths in MCP tools
- **AST/libcst** transforms for refactoring, never regex patching
- **pytest** with parametrized edge cases; no print statements in tests

## MCP Tools Guidelines
- **Default `dry_run=true`** - return unified diffs for review first
- **Workspace boundary** - never modify files outside `${workspaceFolder}`
- **Path safety** - always `Path(file_path).resolve()` and validate against workspace root
- **Error handling** - return structured `{"success": bool, "result": str, "error": str}`

## Critical Files/Patterns
- `src/main.py` - Orchestrator with `.env` loading and plugin registry
- `src/core/event_bus.py` - Redis-backed event system with orjson optimization
- `src/core/neural_atom.py` - Deterministic UUID generation for cognitive artifacts
- `tests/conftest.py` - Comprehensive test fixtures for event-driven testing
- `MCP_WORKFLOW_GUIDE.md` - Complete MCP development workflows
