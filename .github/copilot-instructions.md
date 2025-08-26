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


## 📚 Comprehensive Documentation
For detailed development guidance, refer to these comprehensive instruction sets:

- **`DETAILED_AGENT_INSTRUCTIONS.md`** - Complete development guide covering architecture, patterns, security, testing
- **`ADVANCED_DEVELOPMENT_PATTERNS.md`** - Advanced patterns for cognitive architecture, event streaming, plugin communication
- **`AGENT_QUICK_REFERENCE.md`** - Quick reference checklist for common patterns and workflows
- **`AGENTS.md`** - Repository guidelines and coding standards
- **`src/reug_runtime/AGENTS.md`** - Runtime-specific agent instructions

These documents provide comprehensive coverage of:
- Event-driven neural architecture patterns
- REUG Framework v3.7 and DTA 2.0 cognitive processing
- MCP tool development and VS Code integration
- Plugin development and inter-plugin communication
- Testing strategies for event-driven systems
- Security patterns and sandboxed execution
- Performance optimization and production deployment
- Debugging workflows and troubleshooting guides

## Directory-Specific Guidelines
For detailed instructions specific to different parts of the codebase, consult the nested AGENTS.md files:
- `src/reug_runtime/AGENTS.md` - Runtime agent deployment and operations
- `src/core/AGENTS.md` - Core system components and event bus
- `src/plugins/AGENTS.md` - Plugin development and lifecycle
- `src/sandbox/AGENTS.md` - Secure execution and sandboxing
- `src/neural/AGENTS.md` - Neural atom/bond system
- `mcp_server/AGENTS.md` - MCP server development
- `docs/AGENTS.md` - Documentation standards
- `tests/AGENTS.md` - Testing patterns and fixtures
- `tools/AGENTS.md` - Utility tools and scripts

## Security & Safety Guidelines
- **Never bypass sandbox policies** - All dynamic code execution must go through `src/sandbox/exec_sandbox.py`
- **Environment isolation** - Use proper environment variables, never hardcode secrets
- **Plugin boundaries** - Plugins must only communicate via event bus, no direct imports
- **Path traversal protection** - Always validate file paths against workspace boundaries
- **Redis security** - Event bus connections must use proper authentication in production

## Common Development Patterns

### Plugin Creation Workflow
```bash
# 1. Generate plugin scaffold
python -m src.core.meta_learning_creator generate --capability "your capability"

# 2. Implement plugin interface
# - Extend PluginInterface
# - Implement required methods
# - Add event handlers

# 3. Register plugin
# Add to plugin manifest or registry

# 4. Test plugin
pytest tests/plugins/test_your_plugin.py
```

### Event-Driven Development
```python
# Emit events with proper structure
from src.core.events import create_event
event = create_event(
    "tool_call",
    tool_name="example_tool",
    parameters={"key": "value"},
    source_plugin="your_plugin"
)
await self.event_bus.publish(event)

# Handle events asynchronously
async def handle_event(self, event: Dict[str, Any]) -> None:
    if event["type"] == "tool_result":
        await self.process_result(event["data"])
```

### Neural Atom Creation
```python
from src.core.neural_atom import create_atom
from src.neural.atom import NeuralAtom

# Create deterministic atoms
atom = create_atom(
    content="your content",
    atom_type="tool_output",
    title="Descriptive Title"
)
# UUID is automatically generated from content hash
```

