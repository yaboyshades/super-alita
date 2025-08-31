# Plugin System - Agent Instructions

## Overview
The `src/plugins/` directory contains all plugins that extend Super Alita's capabilities:
- **Core Plugins** - Essential system components (planner, memory, creator)
- **Tool Plugins** - Specific capability implementations (calculator, web search)
- **Adapter Plugins** - External service integrations (Cortex, Puter, Perplexica)
- **Meta Plugins** - System introspection and self-improvement

## Plugin Architecture

### Plugin Lifecycle
1. **Discovery** - Plugin loader finds all `*_plugin.py` files
2. **Registration** - Plugins register with event bus and registry
3. **Initialization** - Plugin `__init__` method called with config
4. **Operation** - Plugin handles events and emits responses
5. **Shutdown** - Cleanup via `shutdown()` method

### Required Plugin Interface
```python
from src.core.plugin_interface import PluginInterface

class YourPlugin(PluginInterface):
    def __init__(self, event_bus, config=None):
        super().__init__(event_bus, config)
        self.name = "unique_plugin_name"  # Required

    @property
    def name(self) -> str:
        """Unique plugin identifier"""
        return self._name

    async def shutdown(self) -> None:
        """Required cleanup method"""
        await self.cleanup_resources()
```

## Key Plugin Categories

### Core System Plugins

#### LLM Planner Plugin (`llm_planner_plugin.py`)
- **Role**: Main planning and decision-making
- **Events**: Handles `user_input`, emits `tool_call` events
- **Safety**: Must validate all tool calls before execution

#### Memory Manager Plugin (`memory_manager_plugin.py`)
- **Role**: Persistent memory and state management
- **Events**: Handles `memory_store`, `memory_retrieve`
- **Storage**: Neural atoms in graph database

#### Creator Plugin (`creator_plugin.py`)
- **Role**: Dynamic tool generation via CREATOR framework
- **Events**: Handles `atom_gap` events, creates new tools
- **Safety**: Generated code must pass validation

### Tool Execution Plugins

#### Tool Executor Plugin (`tool_executor_plugin.py`)
- **Role**: Safe execution of tool calls
- **Events**: Handles `tool_call`, emits `tool_result`
- **Sandboxing**: All execution via `src/sandbox/exec_sandbox.py`

#### Calculator Plugin (`calculator_plugin.py`)
- **Role**: Mathematical computations
- **Events**: Handles math-related tool calls
- **Example**: Simple, well-tested plugin for reference

### External Integration Plugins

#### Puter Plugin (`puter_plugin.py`)
- **Role**: File system and process management via Puter.com
- **API**: RESTful integration with Puter cloud platform
- **Security**: All file operations validated

#### Perplexica Search Plugin (`perplexica_search_plugin.py`)
- **Role**: Web search and research capabilities
- **Integration**: External Perplexica service
- **Rate Limiting**: Implements request throttling

## Plugin Development Guidelines

### Plugin Creation Workflow
```bash
# 1. Create plugin file
touch src/plugins/your_feature_plugin.py

# 2. Implement plugin interface
# See template below

# 3. Add to plugin registry (if needed)
# Most plugins auto-discovered

# 4. Test plugin
pytest tests/plugins/test_your_feature_plugin.py
```

### Plugin Template
```python
from typing import Dict, Any, Optional
import logging
from src.core.plugin_interface import PluginInterface
from src.core.events import create_event

class YourFeaturePlugin(PluginInterface):
    """Plugin for [describe capability]"""

    def __init__(self, event_bus, config: Optional[Dict] = None):
        super().__init__(event_bus, config)
        self.name = "your_feature"
        self.logger = logging.getLogger(self.name)

        # Subscribe to relevant events
        self.event_bus.subscribe("your_event_type", self.handle_event)

    async def handle_event(self, event: Dict[str, Any]) -> None:
        """Handle incoming events"""
        try:
            event_type = event.get("type")

            if event_type == "your_event_type":
                await self.process_your_event(event)

        except Exception as e:
            self.logger.error(f"Event handling failed: {e}")
            await self.emit_error_event(e, event)

    async def process_your_event(self, event: Dict[str, Any]) -> None:
        """Process specific event type"""
        # Your implementation here
        result = await self.do_work(event["data"])

        # Emit result event
        result_event = create_event(
            "your_result_type",
            result=result,
            source_plugin=self.name,
            correlation_id=event.get("correlation_id")
        )
        await self.event_bus.publish(result_event)

    async def shutdown(self) -> None:
        """Clean up resources"""
        # Close connections, save state, etc.
        await super().shutdown()
```

### Event Handling Best Practices
```python
async def handle_event(self, event: Dict[str, Any]) -> None:
    """Robust event handling pattern"""

    # 1. Validate event structure
    if not self.validate_event(event):
        self.logger.warning(f"Invalid event: {event}")
        return

    # 2. Extract correlation ID for tracing
    correlation_id = event.get("correlation_id")

    # 3. Process with error handling
    try:
        result = await self.process_event(event)

        # 4. Emit success event
        await self.emit_result(result, correlation_id)

    except Exception as e:
        # 5. Emit error event for monitoring
        await self.emit_error(e, correlation_id)
```

## Testing Guidelines

### Plugin Testing Pattern
```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.plugins.your_feature_plugin import YourFeaturePlugin

@pytest.mark.asyncio
async def test_plugin_handles_event():
    # Setup
    mock_event_bus = MagicMock()
    plugin = YourFeaturePlugin(mock_event_bus)

    # Test event
    test_event = {
        "type": "your_event_type",
        "data": {"test": "data"},
        "correlation_id": "test-123"
    }

    # Execute
    await plugin.handle_event(test_event)

    # Verify
    assert mock_event_bus.publish.called
    published_event = mock_event_bus.publish.call_args[0][0]
    assert published_event["type"] == "your_result_type"

@pytest.mark.asyncio
async def test_plugin_shutdown():
    mock_event_bus = MagicMock()
    plugin = YourFeaturePlugin(mock_event_bus)

    # Should not raise exception
    await plugin.shutdown()
```

### Integration Testing
```python
@pytest.mark.integration
async def test_plugin_integration(event_bus_fixture):
    """Test plugin with real event bus"""
    plugin = YourFeaturePlugin(event_bus_fixture)

    # Send real event through system
    test_event = create_event("your_event_type", data={"test": True})
    await event_bus_fixture.publish(test_event)

    # Wait for processing
    await asyncio.sleep(0.1)

    # Verify side effects
    assert plugin.state_changed
```

## Security Guidelines

### Input Validation
```python
def validate_event(self, event: Dict[str, Any]) -> bool:
    """Validate event structure and content"""
    required_fields = ["type", "data"]

    # Check required fields
    if not all(field in event for field in required_fields):
        return False

    # Validate data types
    if not isinstance(event["data"], dict):
        return False

    # Additional validation logic
    return True
```

### Safe External Calls
```python
import aiohttp
from src.core.reliability import with_retry

@with_retry(max_attempts=3, backoff_seconds=1.0)
async def call_external_api(self, data: Dict) -> Dict:
    """Safely call external API with retries"""
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(self.api_url, json=data) as response:
            response.raise_for_status()
            return await response.json()
```

## Common Patterns

### State Management
```python
class StatefulPlugin(PluginInterface):
    def __init__(self, event_bus, config=None):
        super().__init__(event_bus, config)
        self.state = {}
        self.lock = asyncio.Lock()

    async def update_state(self, key: str, value: Any) -> None:
        async with self.lock:
            self.state[key] = value
```

### Configuration Management
```python
class ConfigurablePlugin(PluginInterface):
    def __init__(self, event_bus, config=None):
        super().__init__(event_bus, config)

        # Load plugin-specific config
        self.api_key = config.get("api_key") if config else None
        self.timeout = config.get("timeout", 30)
        self.enabled = config.get("enabled", True)

        if not self.api_key:
            self.logger.warning("No API key configured")
```

### Rate Limiting
```python
from src.core.rate_limiter import RateLimiter

class RateLimitedPlugin(PluginInterface):
    def __init__(self, event_bus, config=None):
        super().__init__(event_bus, config)
        self.rate_limiter = RateLimiter(
            max_requests=100,
            time_window=60  # 100 requests per minute
        )

    async def handle_event(self, event: Dict[str, Any]) -> None:
        if not await self.rate_limiter.acquire():
            self.logger.warning("Rate limit exceeded")
            return

        await self.process_event(event)
```

## Performance Guidelines
- **Async operations** - All I/O must be async
- **Resource cleanup** - Always implement proper shutdown
- **Memory usage** - Monitor and limit memory consumption
- **Connection pooling** - Reuse HTTP/DB connections
- **Caching** - Cache expensive computations when appropriate

## Debugging Tips
- **Correlation IDs** - Use for tracing events across plugins
- **Structured logging** - Include plugin name and event context
- **Event metrics** - Track processing times and error rates
- **Health checks** - Implement plugin health monitoring
