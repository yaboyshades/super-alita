# Source Code Instructions

This directory contains the main source code for Super Alita's event-driven AI agent system.

## Architecture Overview

The src directory is organized into modular components following the event-driven architecture pattern:

- **Core Components** (`core/`): Event bus, plugin interface, neural atoms
- **Plugins** (`plugins/`): Modular plugin implementations
- **Abilities** (`abilities/`): Discrete capabilities and tools
- **Integration** (`integration/`): External service integrations
- **MCP** (`mcp_server/`, `mcp_local/`): Model Context Protocol implementations
- **Orchestration** (`orchestration/`): Streaming and workflow management
- **Planning** (`planning/`): Decision policies and strategic planning
- **Telemetry** (`telemetry/`): Monitoring and metrics collection

## Structure Guidelines

- **Organize by feature**: Group related functionality into logical modules
- **Follow plugin pattern**: All components should inherit from `PluginInterface`
- **Event-driven design**: Use the event bus for inter-component communication
- **Type annotations**: All code must include comprehensive type hints
- **Documentation**: Include docstrings for all public interfaces

## Building

### Dependencies
```bash
# Install core dependencies
pip install -e .

# For GPU acceleration (optional)
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install -r requirements-gpu.txt
```

### Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Set required environment variables
export PYTHONPATH=./src
```

## Code Standards

- **Formatting**: Use Black (88 character line limit)
- **Linting**: Use Ruff with project configuration
- **Type Checking**: Use mypy for static analysis
- **Path Handling**: Use `pathlib.Path`, not `os.path`
- **Async/Await**: Prefer async patterns for I/O operations

## Key Modules

### Core Event System
- `core/event_bus.py` - Redis-backed event system
- `core/plugin_interface.py` - Base interface for all plugins
- `core/neural_atom.py` - Deterministic UUID generation

### Main Orchestrator
- `main.py` - Primary application entry point
- `orchestration/` - Streaming and workflow management

### Plugin Development
```python
from src.core.plugin_interface import PluginInterface
from src.core.events import create_event

class MyPlugin(PluginInterface):
    @property
    def name(self) -> str:
        return "my_plugin"

    async def shutdown(self) -> None:
        # Cleanup logic
        pass
```

### Event Creation
```python
# Always use keyword arguments
event = create_event("cognitive_turn", turn_data=data, confidence=0.95)
```

## Running

See the root `INSTRUCTIONS.md` for general run commands. The main entry point is:

```bash
python -m uvicorn src.main:app --reload --port 8080
```

## Testing

Run tests from the repository root:
```bash
pytest src/ -v
```

For specific module testing:
```bash
pytest src/core/ -v
pytest src/plugins/ -v
```
