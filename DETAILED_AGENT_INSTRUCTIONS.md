# Super Alita - Detailed Custom Instructions for Coding Agents

## System Overview

Super Alita is a sophisticated **self-evolving AI agent system** built on an event-driven neural architecture. This document provides comprehensive instructions for coding agents working on this complex system.

### Core Architecture Principles

1. **Event-Driven Neural Architecture**: All system components communicate through a Redis/Memurai-backed event bus
2. **MCP (Model Context Protocol) Integration**: Tools and VS Code integration through standardized protocol
3. **Atoms/Bonds Cognitive Fabric**: All outputs structured as atoms with deterministic UUIDs
4. **Plugin-Based Modularity**: All components inherit from `PluginInterface` for hot-swappable functionality
5. **Sandboxed Execution**: All dynamic code execution must go through secure sandboxing
6. **Multi-Modal LLM Support**: Automatic fallback between Gemini → local Super Alita → mock providers

## Directory Structure & Organization

```
super-alita/
├── src/                           # Main source code
│   ├── core/                      # Core system components
│   │   ├── event_bus.py          # Redis-backed event system
│   │   ├── plugin_interface.py   # Base plugin interface
│   │   ├── neural_atom.py        # Deterministic UUID generation
│   │   └── events.py             # Event creation patterns
│   ├── plugins/                   # Plugin implementations
│   ├── abilities/                 # Agent abilities/capabilities
│   ├── sandbox/                   # Secure execution environment
│   ├── reug_runtime/             # Runtime agent deployment
│   └── main.py                   # System orchestrator
├── mcp_server/                   # MCP server implementation
├── cortex/                       # Cognitive processing components
├── tests/                        # Test suite (mirrors src/)
│   └── conftest.py              # Comprehensive test fixtures
├── docs/                         # Documentation
└── scripts/                      # Utility scripts
```

## Development Environment Setup

### Essential Commands
```bash
# Environment setup
cp .env.example .env              # Configure environment
make deps                         # Install dependencies
make run                          # Start FastAPI dev server
make test                         # Run test suite
make lint                         # Format and lint code

# MCP Development
pwsh .\Setup-MCP.ps1 -Bootstrap   # Initialize MCP server + VS Code
pwsh .\Setup-MCP.ps1 -Doctor      # Health check environment
pwsh .\Setup-MCP.ps1 -AddTool YourTool  # Scaffold new tools
```

### Environment Configuration
Key environment variables in `.env`:
```bash
# Execution Guardrails
REUG_MAX_TOOL_CALLS=5
REUG_EXEC_TIMEOUT_S=20.0
SUPER_ALITA_MODE=shadow           # shadow/act/batch

# LLM Provider Configuration
LLM_MODEL=auto                    # Enable automatic fallback
GEMINI_API_KEY=your-key          # Primary provider
SUPER_ALITA_BASE_URL=http://127.0.0.1:8080  # Local adapter

# Event Bus & Persistence
REDIS_URL=redis://localhost:6379
REUG_EVENT_LOG_DIR=./logs/events
```

## Code Standards & Quality Requirements

### Formatting & Linting
- **Black** with 88-character line length
- **Ruff** for comprehensive linting (see `ruff.toml`)
- **MyPy** with strict type checking for `src/core` and `src/sandbox`
- **Pre-commit hooks** enforced for all commits

### Code Style Guidelines
```python
# Type hints are mandatory
from typing import Any, Optional, Dict, List
from pathlib import Path

# Use double quotes consistently
message = "Hello, Super Alita!"

# Prefer pathlib over os.path
config_path = Path("config") / "settings.yaml"

# All functions need type hints
def process_event(event_data: Dict[str, Any]) -> Optional[str]:
    """Process an event and return result."""
    return None

# Use Pydantic models for data structures
from pydantic import BaseModel, Field

class EventSchema(BaseModel):
    event_id: str = Field(description="Unique event identifier")
    timestamp: datetime
    payload: Dict[str, Any]
```

### Security Requirements
```python
# NEVER use raw eval/exec - use sandbox
from src.sandbox.exec_sandbox import safe_execute

# Subprocess calls must use proc.py utilities
from src.core.proc import run_command
result = run_command(["python", "script.py"])  # Never shell=True

# YAML operations via utils
from src.core.yaml_utils import safe_load_yaml, safe_dump_yaml

# All credentials via environment variables
import os
api_key = os.getenv("API_KEY")  # Never hardcode secrets
```

## Event-Driven Development Patterns

### Event Creation
```python
from src.core.events import create_event
from datetime import datetime, timezone

# ALWAYS use keyword arguments, never positional dicts
event = create_event(
    "cognitive_turn",
    turn_data=data,
    confidence=0.95,
    source_plugin="my_plugin"
)

# Use timezone-aware timestamps
timestamp = datetime.now(timezone.utc)
```

### Plugin Development
```python
from src.core.plugin_interface import PluginInterface
from src.core.events import create_event
import asyncio

class MyPlugin(PluginInterface):
    def __init__(self, event_bus, config: Dict[str, Any]):
        super().__init__()
        self.event_bus = event_bus
        self.config = config
        
    @property
    def name(self) -> str:
        """Unique plugin identifier."""
        return "my_plugin"
    
    async def initialize(self) -> bool:
        """Initialize plugin resources."""
        # Setup logic here
        return True
    
    async def shutdown(self) -> None:
        """Clean up plugin resources."""
        # Cleanup logic here
        pass
    
    async def handle_event(self, event: BaseEvent) -> None:
        """Handle incoming events."""
        if event.event_type == "my_event_type":
            # Process event
            result_event = create_event(
                "processing_complete",
                result={"status": "success"},
                source_plugin=self.name
            )
            await self.event_bus.emit(result_event)
```

### Async Testing Patterns
```python
import pytest
from unittest.mock import AsyncMock, Mock

@pytest.mark.asyncio  # Required for all async tests
async def test_event_flow():
    """Test event processing flow."""
    # Setup
    mock_bus = AsyncMock()
    plugin = MyPlugin(mock_bus, {})
    
    # Test
    await plugin.handle_event(test_event)
    
    # Verify
    mock_bus.emit.assert_called_once()
    
# Use fixtures from conftest.py
def test_with_mock_metrics(mock_metrics_registry):
    """Test using shared fixtures."""
    assert mock_metrics_registry is not None
```

## MCP Tool Development

### Tool Structure
Tools live in `mcp_server/src/mcp_server/tools/`:
```python
from typing import Dict, Any
from pathlib import Path

def my_tool(
    file_path: str,
    operation: str = "analyze",
    dry_run: bool = True  # Default to dry_run for safety
) -> Dict[str, Any]:
    """
    My custom tool implementation.
    
    Args:
        file_path: Path to target file
        operation: Operation to perform
        dry_run: Return diff preview instead of executing
        
    Returns:
        Structured result with success, result, and error fields
    """
    try:
        # Validate workspace boundary
        target_path = Path(file_path).resolve()
        workspace_root = Path.cwd().resolve()
        
        if not str(target_path).startswith(str(workspace_root)):
            return {
                "success": False,
                "result": "",
                "error": "Path outside workspace boundary"
            }
        
        if dry_run:
            return {
                "success": True,
                "result": "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,4 @@\n...",
                "error": ""
            }
        
        # Actual implementation here
        return {
            "success": True,
            "result": "Operation completed successfully",
            "error": ""
        }
        
    except Exception as e:
        return {
            "success": False,
            "result": "",
            "error": str(e)
        }

# Tool specification for MCP
my_tool_spec = {
    "name": "my_tool",
    "description": "Tool description for MCP",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "File path"},
            "operation": {"type": "string", "description": "Operation type"},
            "dry_run": {"type": "boolean", "default": True}
        },
        "required": ["file_path"]
    }
}
```

### MCP Tool Guidelines
- **Default `dry_run=true`**: Always return unified diffs for review first
- **Workspace boundary**: Never modify files outside `${workspaceFolder}`
- **Path safety**: Always `Path(file_path).resolve()` and validate
- **Error handling**: Return structured `{"success": bool, "result": str, "error": str}`
- **Windows compatibility**: Assume Windows paths in MCP tools

## Testing Strategies

### Test Organization
```bash
tests/
├── conftest.py              # Shared fixtures
├── core/                    # Core component tests
├── plugins/                 # Plugin tests
├── integration/             # Integration tests
└── test_*.py               # Individual test files
```

### Testing Patterns
```python
import pytest
from unittest.mock import AsyncMock, patch
from src.core.events import create_event

class TestMyComponent:
    """Test class following naming conventions."""
    
    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async operations."""
        pass
    
    @pytest.mark.integration_redis
    async def test_redis_integration(self):
        """Test requiring Redis."""
        pass
    
    @pytest.mark.parametrize("input_val,expected", [
        ("test1", "result1"),
        ("test2", "result2"),
    ])
    def test_parametrized(self, input_val, expected):
        """Parametrized test for edge cases."""
        pass

# Use shared fixtures
def test_with_fixtures(mock_event_bus, sample_event):
    """Test using conftest.py fixtures."""
    pass
```

### Coverage Requirements
- Target ≥70% test coverage
- New code requires tests
- Use `pytest -q` for quick runs
- Use `pytest -k pattern` for filtered runs
- Use `pytest -m integration_redis` for marked tests

## Neural Atom & Cognitive Fabric

### Atom Creation Patterns
```python
from src.core.neural_atom import create_neural_atom
from src.core.ids import generate_deterministic_uuid

# Create atoms with deterministic UUIDs
atom_id = generate_deterministic_uuid(content="my_content")
atom = create_neural_atom(
    atom_id=atom_id,
    content="Cognitive content",
    atom_type="thought",
    metadata={"confidence": 0.95}
)

# Bond creation between atoms
bond = create_bond(
    source_atom_id=atom1.id,
    target_atom_id=atom2.id,
    bond_type="causal_relationship",
    strength=0.8
)
```

### Memory & Context Management
```python
from src.core.memory import MemoryManager
from src.core.context_builder import build_context

# Context building for LLM interactions
context = build_context(
    history=conversation_history,
    relevant_atoms=memory.retrieve_relevant_atoms(query),
    active_tools=tool_registry.get_active_tools()
)
```

## Error Handling & Reliability

### Robust Error Patterns
```python
import logging
from src.core.error_recovery import retry_with_backoff
from src.core.reliability import circuit_breaker

logger = logging.getLogger(__name__)

@retry_with_backoff(max_retries=3, base_delay=0.25)
@circuit_breaker(failure_threshold=5, recovery_timeout=30)
async def reliable_operation():
    """Operation with built-in reliability patterns."""
    try:
        # Operation logic
        result = await external_api_call()
        return result
    except Exception as e:
        logger.error(f"Operation failed: {e}", exc_info=True)
        raise

# Event-driven error handling
error_event = create_event(
    "error_occurred",
    error_type=type(e).__name__,
    error_message=str(e),
    component="my_component",
    recovery_action="retry_scheduled"
)
await event_bus.emit(error_event)
```

## Performance & Optimization

### Event Bus Optimization
```python
# Use orjson for fast serialization
import orjson

# Batch event processing
async def process_event_batch(events: List[BaseEvent]):
    """Process events in batches for efficiency."""
    async with event_bus.batch_context():
        for event in events:
            await event_bus.emit(event)

# Memory-efficient streaming
async def stream_large_dataset():
    """Stream data without loading everything into memory."""
    async for chunk in data_source.stream():
        yield process_chunk(chunk)
```

### Redis Patterns
```python
from src.core.redis_event_bus import RedisEventBus

# Efficient Redis usage
async def efficient_redis_ops():
    """Use Redis efficiently with pipelines."""
    async with redis_client.pipeline() as pipe:
        pipe.set("key1", "value1")
        pipe.set("key2", "value2")
        await pipe.execute()
```

## Security & Sandboxing

### Safe Execution Patterns
```python
from src.sandbox.exec_sandbox import ExecutionSandbox
from src.core.schemas import SafeExecutionRequest

# Always use sandbox for dynamic execution
sandbox = ExecutionSandbox(
    timeout=30,
    memory_limit="512MB",
    allowed_modules=["math", "json", "datetime"]
)

request = SafeExecutionRequest(
    code="result = 2 + 2",
    context_vars={"input_data": data},
    allowed_builtins=["len", "str", "int"]
)

result = await sandbox.execute(request)
```

### Credential Management
```python
import os
from typing import Optional

def get_api_key(service: str) -> Optional[str]:
    """Safely retrieve API keys from environment."""
    return os.getenv(f"{service.upper()}_API_KEY")

# Never log or expose credentials
logger.info(f"Using {service} API: {'✓' if get_api_key(service) else '✗'}")
```

## Git & Development Workflow

### Commit Conventions
```bash
# Format: [module] Short description
git commit -m "[core] Add event batching support"
git commit -m "[plugins] Fix memory leak in analyzer"
git commit -m "[tests] Add integration tests for Redis bus"
git commit -m "[docs] Update MCP tool development guide"
```

### PR Requirements
Before submitting a PR:
1. Run `pre-commit run --all-files`
2. Ensure `pytest` passes with ≥70% coverage
3. Type-check with `mypy src/core src/sandbox`
4. Include summary, rationale, and linked issues
5. Update relevant documentation

### Branch Naming
```bash
feature/add-new-capability
fix/event-bus-memory-leak
refactor/plugin-interface-cleanup
docs/update-testing-guide
```

## Debugging & Diagnostics

### Logging Patterns
```python
import logging
import json

# Structured logging for better debugging
logger = logging.getLogger(__name__)

def log_event_processing(event: BaseEvent, result: Any):
    """Log event processing with structured data."""
    logger.info(
        "Event processed",
        extra={
            "event_id": event.event_id,
            "event_type": event.event_type,
            "processing_time_ms": result.processing_time,
            "success": result.success
        }
    )

# Use correlation IDs for tracing
correlation_id = event.correlation_id
logger.info(f"Processing event {event.event_type}", extra={"correlation_id": correlation_id})
```

### Diagnostic Tools
```bash
# Health checks
curl http://127.0.0.1:8080/healthz

# Debug utilities
python scripts/debug_fixed.py
python scripts/debug_matching.py
python scripts/utility_debug.py

# Event bus diagnostics
python -m src.core.event_bus --diagnose

# MCP server validation
python -m mcp_server.main --validate
```

## Common Patterns & Anti-Patterns

### ✅ Good Patterns
```python
# Event-driven communication
await event_bus.emit(create_event("task_completed", result=data))

# Type-safe plugin interfaces
class MyPlugin(PluginInterface):
    @property
    def name(self) -> str:
        return "my_plugin"

# Async context managers for resources
async with DatabaseConnection() as conn:
    result = await conn.execute(query)

# Proper error propagation
try:
    result = await risky_operation()
except SpecificError as e:
    logger.error("Known error occurred", exc_info=True)
    await self.handle_specific_error(e)
    raise
```

### ❌ Anti-Patterns
```python
# DON'T use raw eval/exec
eval(user_code)  # NEVER DO THIS

# DON'T bypass sandbox
subprocess.run(cmd, shell=True)  # SECURITY RISK

# DON'T use positional dicts for events
create_event("test", {"data": "value"})  # Use kwargs instead

# DON'T ignore type hints
def process_data(data):  # Missing type hints
    return data + 1

# DON'T commit secrets
API_KEY = "secret-key-123"  # Use environment variables
```

## Integration Points

### VS Code Integration
- MCP server runs as separate process
- Tools registered via `MCP: Show Installed Servers`
- Agent Mode integration through VS Code commands

### External Services
- Gemini API for primary LLM
- Redis for event bus (optional)
- Puter Cloud for remote execution
- Various search APIs (SerpAPI, Bing, Google)

### Telemetry & Monitoring
- Real-time event streaming via MCP
- Performance metrics collection
- Error tracking and alerting
- Resource usage monitoring

## Final Guidelines

1. **Think in Events**: Everything is an event in Super Alita
2. **Safety First**: Always use sandboxing for dynamic execution
3. **Type Everything**: Comprehensive type hints are required
4. **Test Thoroughly**: Event-driven systems need comprehensive testing
5. **Document Decisions**: Complex architecture requires clear documentation
6. **Monitor Performance**: Event bus and Redis usage should be optimized
7. **Secure by Default**: Never bypass security mechanisms
8. **Plugin Architecture**: Extend through plugins, not core modifications
9. **Async First**: Most operations should be async-compatible
10. **Deterministic UUIDs**: Use atoms/bonds pattern for traceability

This document should be treated as the canonical reference for all development work on Super Alita. When in doubt, refer to these patterns and guidelines to maintain consistency with the system's architecture and philosophy.