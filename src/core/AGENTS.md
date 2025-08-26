# Core System Components - Agent Instructions

## Overview
The `src/core/` directory contains the foundational components of the Super Alita system:
- **Event Bus** - Redis-backed pub/sub system for inter-plugin communication
- **Neural Atoms/Bonds** - Deterministic cognitive artifact system
- **Plugin Infrastructure** - Base interfaces and lifecycle management
- **Memory & State** - Persistent storage and session management
- **Security & Sandboxing** - Safe execution environments

## Key Files & Responsibilities

### Event System
- `event_bus.py` - Main Redis event bus implementation
- `events.py` - Event creation utilities and schemas
- `event_types.py` - Event type definitions and contracts
- `redis_event_bus.py` - Redis-specific implementation details

### Neural System
- `neural_atom.py` - Deterministic UUID generation for cognitive artifacts
- `neural_symbolic_bridge.py` - Integration between neural and symbolic processing
- `memory.py` - Persistent memory management
- `knowledge_graph.py` - Graph-based knowledge representation

### Plugin Infrastructure
- `plugin_interface.py` - Base interface all plugins must implement
- `plugin_loader.py` - Dynamic plugin discovery and loading
- `unified_registry.py` - Central plugin registry management

### Execution & Safety
- `secure_executor.py` - Safe execution contexts
- `sandbox_runner.py` - Sandboxed code execution
- `proc.py` - Process management utilities
- `yaml_utils.py` - Safe YAML parsing

## Development Guidelines

### Event Bus Programming
```python
# Always use create_event for consistency
from src.core.events import create_event

# Emit events with all required fields
event = create_event(
    "cognitive_turn",
    session_id=session_id,
    confidence=0.95,
    source_plugin=self.name
)

# Subscribe to events with proper error handling
async def handle_event(self, event: Dict[str, Any]) -> None:
    try:
        await self.process_event(event)
    except Exception as e:
        self.logger.error(f"Event handling failed: {e}")
        # Emit error event for monitoring
        error_event = create_event("plugin_error", 
                                  error=str(e), 
                                  source_plugin=self.name)
        await self.event_bus.publish(error_event)
```

### Neural Atom Best Practices
```python
from src.core.neural_atom import create_atom

# Create atoms with proper metadata
atom = create_atom(
    content=result_data,
    atom_type="tool_output", 
    title="Descriptive Title",
    metadata={
        "source": self.name,
        "timestamp": datetime.now(timezone.utc),
        "confidence": confidence_score
    }
)

# UUIDs are deterministic based on content
# Same content = same UUID (enables deduplication)
```

### Plugin Development
```python
from src.core.plugin_interface import PluginInterface

class YourPlugin(PluginInterface):
    def __init__(self, event_bus, config=None):
        super().__init__(event_bus, config)
        self.name = "your_plugin"
        
    async def shutdown(self) -> None:
        """Required cleanup method"""
        await self.cleanup_resources()
        
    async def handle_event(self, event: Dict[str, Any]) -> None:
        """Event handler implementation"""
        pass
```

## Testing Guidelines

### Event Bus Testing
```python
@pytest.mark.asyncio
async def test_event_flow(mock_event_bus):
    # Use provided fixtures
    plugin = YourPlugin(mock_event_bus)
    
    # Test event emission
    await plugin.emit_test_event()
    
    # Verify event was published
    assert mock_event_bus.publish.called
    published_event = mock_event_bus.publish.call_args[0][0]
    assert published_event["type"] == "expected_type"
```

### Memory Testing
```python
def test_neural_atom_deterministic():
    """Test that atoms have deterministic UUIDs"""
    content = {"key": "value"}
    atom1 = create_atom(content, "test_type", "Test")
    atom2 = create_atom(content, "test_type", "Test")
    
    assert atom1.uuid == atom2.uuid  # Same content = same UUID
```

## Security Considerations

### Safe Execution
- **Never use raw `eval()` or `exec()`** - Use `secure_executor.py`
- **Process isolation** - All subprocesses via `proc.py` (no `shell=True`)
- **YAML safety** - Only use `yaml_utils.py` for parsing

### Event Bus Security
- **Authentication** - Redis connections must be authenticated in production
- **Event validation** - All events must conform to defined schemas
- **Rate limiting** - Implement event rate limits to prevent abuse

### Plugin Sandboxing
- **Isolated namespaces** - Plugins cannot directly import each other
- **Resource limits** - CPU/memory limits enforced by sandbox
- **File system access** - Restricted to designated directories

## Common Patterns

### Error Recovery
```python
from src.core.error_recovery import RecoveryContext

async def resilient_operation(self):
    async with RecoveryContext(self.logger) as ctx:
        try:
            result = await self.risky_operation()
            return result
        except SpecificError as e:
            # Log and emit recovery event
            ctx.record_error(e)
            return await self.fallback_operation()
```

### Metrics Collection
```python
from src.core.metrics import MetricsCollector

with MetricsCollector.timer("operation_duration"):
    result = await self.timed_operation()
    
MetricsCollector.increment("operation_count")
MetricsCollector.gauge("memory_usage", get_memory_usage())
```

## Performance Guidelines
- **Async everywhere** - All I/O operations must be async
- **Connection pooling** - Reuse Redis connections
- **Event batching** - Batch related events when possible
- **Memory management** - Clean up resources in shutdown methods

## Debugging Tips
- **Event tracing** - Enable event bus logging for flow debugging
- **Memory leaks** - Monitor plugin resource usage
- **Redis monitoring** - Check Redis memory and connection counts
- **Performance profiling** - Use built-in metrics collection