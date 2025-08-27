# Core System Tests - Agent Instructions

## Overview
The `tests/core/` directory contains comprehensive tests for core system components:
- **Event Bus Tests** - Redis event bus functionality and performance
- **Neural System Tests** - Neural atom/bond system validation
- **Plugin Tests** - Plugin infrastructure and lifecycle testing
- **Memory Tests** - Persistent storage and session management
- **Security Tests** - Safe execution and sandboxing validation

## Test Structure

### Directory Organization
```
tests/core/
├── test_event_bus.py          # Event bus functionality tests
├── test_neural_atom.py        # Neural atom system tests
├── test_plugin_interface.py   # Plugin interface tests
├── test_memory.py             # Memory management tests
├── test_security.py           # Security and safety tests
├── cortex/                    # Cortex integration tests
│   ├── test_cortex_client.py
│   └── test_cortex_weaning.py
└── fixtures/                  # Test data and fixtures
    ├── sample_events.json
    └── mock_plugins/
```

## Testing Guidelines

### Event Bus Testing
```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.core.event_bus import EventBus
from src.core.events import create_event

@pytest.mark.asyncio
async def test_event_publishing():
    """Test event publishing functionality"""
    event_bus = EventBus()
    await event_bus.initialize()
    
    # Create test event
    event = create_event(
        "test_event",
        data={"key": "value"},
        source="test_publisher"
    )
    
    # Test event publishing
    await event_bus.publish(event)
    
    # Verify event was published
    # Add assertions based on implementation
    await event_bus.cleanup()

@pytest.mark.asyncio
async def test_event_subscription():
    """Test event subscription and handling"""
    event_bus = EventBus()
    await event_bus.initialize()
    
    received_events = []
    
    async def test_handler(event):
        received_events.append(event)
        
    # Subscribe to events
    await event_bus.subscribe("test_event", test_handler)
    
    # Publish test event
    event = create_event("test_event", data={"test": True})
    await event_bus.publish(event)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Verify event was received
    assert len(received_events) == 1
    assert received_events[0]["type"] == "test_event"
    
    await event_bus.cleanup()

@pytest.mark.integration_redis
async def test_redis_event_bus_integration():
    """Integration test with actual Redis instance"""
    # This test requires Redis to be running
    redis_config = {
        "host": "localhost",
        "port": 6379,
        "db": 15  # Use test database
    }
    
    event_bus = EventBus(config=redis_config)
    await event_bus.initialize()
    
    try:
        # Test Redis connectivity
        await event_bus.health_check()
        
        # Test event flow through Redis
        event = create_event("integration_test", data={"redis": True})
        await event_bus.publish(event)
        
    finally:
        await event_bus.cleanup()
```

### Neural Atom Testing
```python
import pytest
from src.core.neural_atom import create_atom, NeuralAtom
from src.neural.atom import AtomType

def test_deterministic_atom_creation():
    """Test deterministic UUID generation for atoms"""
    content = "test content"
    
    # Create same atom twice
    atom1 = create_atom(content, AtomType.TOOL_OUTPUT, "Test Title")
    atom2 = create_atom(content, AtomType.TOOL_OUTPUT, "Test Title")
    
    # Should have same UUID (deterministic)
    assert atom1.uuid == atom2.uuid
    assert atom1.content == atom2.content

def test_atom_different_content():
    """Test different content produces different UUIDs"""
    atom1 = create_atom("content1", AtomType.TOOL_OUTPUT, "Title1")
    atom2 = create_atom("content2", AtomType.TOOL_OUTPUT, "Title2")
    
    # Should have different UUIDs
    assert atom1.uuid != atom2.uuid

@pytest.mark.asyncio
async def test_atom_persistence():
    """Test neural atom persistence and retrieval"""
    # Mock storage backend
    storage_mock = AsyncMock()
    
    atom = create_atom("test content", AtomType.MEMORY, "Test Memory")
    
    # Test storage
    await storage_mock.store_atom(atom)
    storage_mock.store_atom.assert_called_once_with(atom)
    
    # Test retrieval
    storage_mock.get_atom.return_value = atom
    retrieved = await storage_mock.get_atom(atom.uuid)
    
    assert retrieved.uuid == atom.uuid
    assert retrieved.content == atom.content
```

### Plugin Interface Testing
```python
import pytest
from unittest.mock import AsyncMock
from src.core.plugin_interface import PluginInterface

class TestPlugin(PluginInterface):
    """Test plugin implementation"""
    
    def __init__(self, event_bus, config=None):
        super().__init__(event_bus, config)
        self.name = "test_plugin"
        self.initialized = False
        
    async def initialize(self):
        self.initialized = True
        
    async def shutdown(self):
        self.initialized = False

@pytest.mark.asyncio
async def test_plugin_lifecycle():
    """Test plugin lifecycle management"""
    event_bus = AsyncMock()
    plugin = TestPlugin(event_bus)
    
    # Test initialization
    await plugin.initialize()
    assert plugin.initialized is True
    
    # Test shutdown
    await plugin.shutdown()
    assert plugin.initialized is False

@pytest.mark.asyncio
async def test_plugin_event_handling():
    """Test plugin event handling"""
    event_bus = AsyncMock()
    plugin = TestPlugin(event_bus)
    
    # Test event emission
    await plugin.emit_event("test_event", {"data": "value"})
    
    # Verify event bus was called
    event_bus.publish.assert_called_once()

def test_plugin_configuration():
    """Test plugin configuration handling"""
    config = {"setting1": "value1", "setting2": 42}
    event_bus = AsyncMock()
    
    plugin = TestPlugin(event_bus, config)
    
    assert plugin.config == config
    assert plugin.get_config("setting1") == "value1"
    assert plugin.get_config("setting2") == 42
    assert plugin.get_config("missing", "default") == "default"
```

### Memory System Testing
```python
import pytest
from unittest.mock import AsyncMock, patch
from src.core.memory import MemoryManager

@pytest.mark.asyncio
async def test_memory_storage():
    """Test memory storage functionality"""
    memory_manager = MemoryManager()
    
    # Test storing memory
    memory_data = {
        "key": "test_memory",
        "content": "This is a test memory",
        "tags": ["test", "memory"]
    }
    
    memory_id = await memory_manager.store(memory_data)
    assert memory_id is not None
    
    # Test retrieving memory
    retrieved = await memory_manager.retrieve(memory_id)
    assert retrieved["content"] == memory_data["content"]
    assert retrieved["tags"] == memory_data["tags"]

@pytest.mark.asyncio
async def test_memory_search():
    """Test memory search functionality"""
    memory_manager = MemoryManager()
    
    # Store test memories
    memories = [
        {"content": "Python programming", "tags": ["coding", "python"]},
        {"content": "JavaScript development", "tags": ["coding", "javascript"]},
        {"content": "Database design", "tags": ["database", "sql"]}
    ]
    
    for memory in memories:
        await memory_manager.store(memory)
    
    # Test tag-based search
    coding_memories = await memory_manager.search(tags=["coding"])
    assert len(coding_memories) == 2
    
    # Test content search
    python_memories = await memory_manager.search(content_query="Python")
    assert len(python_memories) == 1
```

### Security Testing
```python
import pytest
from src.core.secure_executor import SecureExecutor
from src.sandbox.registry import SecurityRegistry

@pytest.mark.asyncio
async def test_secure_code_execution():
    """Test secure code execution"""
    executor = SecureExecutor()
    
    # Test safe code execution
    safe_code = """
result = 2 + 2
print(f"Result: {result}")
"""
    
    result = await executor.execute(safe_code)
    assert result["success"] is True
    assert "Result: 4" in result["output"]

@pytest.mark.asyncio
async def test_dangerous_code_blocking():
    """Test blocking of dangerous code"""
    executor = SecureExecutor()
    
    # Test dangerous code is blocked
    dangerous_code = """
import os
os.system("rm -rf /")
"""
    
    with pytest.raises(SecurityError):
        await executor.execute(dangerous_code)

def test_security_registry():
    """Test security registry functionality"""
    registry = SecurityRegistry()
    
    # Test allowlist
    registry.add_allowed_function("math.sqrt")
    assert registry.is_allowed("math.sqrt") is True
    assert registry.is_allowed("os.system") is False
    
    # Test blocklist
    registry.add_blocked_pattern(r"eval\s*\(")
    assert registry.is_blocked("eval('malicious')") is True
    assert registry.is_blocked("evaluate_expression()") is False
```

## Performance Testing

### Load Testing
```python
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

@pytest.mark.performance
async def test_event_bus_throughput():
    """Test event bus throughput under load"""
    event_bus = EventBus()
    await event_bus.initialize()
    
    event_count = 1000
    start_time = time.time()
    
    # Publish events concurrently
    tasks = []
    for i in range(event_count):
        event = create_event("load_test", data={"index": i})
        tasks.append(event_bus.publish(event))
    
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    duration = end_time - start_time
    throughput = event_count / duration
    
    # Should handle at least 100 events per second
    assert throughput > 100
    
    await event_bus.cleanup()

@pytest.mark.performance
async def test_memory_retrieval_performance():
    """Test memory retrieval performance"""
    memory_manager = MemoryManager()
    
    # Store many memories
    memory_ids = []
    for i in range(100):
        memory_data = {"content": f"Memory {i}", "index": i}
        memory_id = await memory_manager.store(memory_data)
        memory_ids.append(memory_id)
    
    # Test bulk retrieval performance
    start_time = time.time()
    
    tasks = [memory_manager.retrieve(mid) for mid in memory_ids]
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Should retrieve 100 memories in under 1 second
    assert duration < 1.0
    assert len(results) == 100
```

## Integration Testing

### End-to-End Core System Tests
```python
@pytest.mark.integration
async def test_full_core_system_integration():
    """Test integration of all core components"""
    # Initialize all core components
    event_bus = EventBus()
    memory_manager = MemoryManager(event_bus)
    plugin_loader = PluginLoader(event_bus)
    
    await event_bus.initialize()
    await memory_manager.initialize()
    
    try:
        # Load test plugin
        test_plugin = await plugin_loader.load_plugin("test_plugin")
        
        # Test event flow
        event = create_event("integration_test", data={"test": True})
        await event_bus.publish(event)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Verify system state
        assert test_plugin.initialized is True
        
        # Test memory integration
        memory_data = {"content": "Integration test memory"}
        memory_id = await memory_manager.store(memory_data)
        retrieved = await memory_manager.retrieve(memory_id)
        
        assert retrieved["content"] == memory_data["content"]
        
    finally:
        await plugin_loader.shutdown_all()
        await memory_manager.shutdown()
        await event_bus.cleanup()
```

## Test Configuration

### Test Environment Setup
```python
# conftest.py additions for core tests
import pytest
import asyncio
from unittest.mock import AsyncMock

@pytest.fixture
async def event_bus():
    """Event bus fixture for testing"""
    bus = EventBus(config={"mode": "test"})
    await bus.initialize()
    yield bus
    await bus.cleanup()

@pytest.fixture
def mock_plugin():
    """Mock plugin fixture"""
    plugin = AsyncMock()
    plugin.name = "mock_plugin"
    plugin.initialized = True
    return plugin

@pytest.fixture
async def memory_manager():
    """Memory manager fixture"""
    manager = MemoryManager(config={"storage": "memory"})
    await manager.initialize()
    yield manager
    await manager.cleanup()

# Pytest markers for core tests
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.core
]
```

### Test Data Fixtures
```python
# fixtures/test_data.py
SAMPLE_EVENTS = [
    {
        "type": "tool_call",
        "tool_name": "calculator",
        "parameters": {"operation": "add", "a": 2, "b": 3}
    },
    {
        "type": "user_input", 
        "message": "Hello, how are you?",
        "session_id": "test_session"
    },
    {
        "type": "agent_reply",
        "content": "I'm doing well, thank you!",
        "confidence": 0.95
    }
]

SAMPLE_NEURAL_ATOMS = [
    {
        "content": "Test memory content",
        "atom_type": "memory",
        "title": "Test Memory"
    },
    {
        "content": "Tool execution result",
        "atom_type": "tool_output", 
        "title": "Calculator Result"
    }
]
```

## Debugging Test Issues

### Common Test Patterns
```python
# Debug helper for async tests
async def debug_async_test(coro, timeout=5.0):
    """Debug wrapper for async test functions"""
    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        logger.error(f"Test timed out after {timeout} seconds")
        raise
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        raise

# Event bus debugging
async def debug_event_flow(event_bus, event):
    """Debug event publishing and subscription"""
    published = False
    received = False
    
    async def debug_handler(received_event):
        nonlocal received
        received = True
        logger.debug(f"Received event: {received_event}")
    
    await event_bus.subscribe(event["type"], debug_handler)
    await event_bus.publish(event)
    published = True
    
    await asyncio.sleep(0.1)  # Wait for processing
    
    logger.debug(f"Event flow: published={published}, received={received}")
    return published and received
```

## Best Practices

### Test Organization
- Mirror the source code structure in test directories
- Use descriptive test names that explain what is being tested
- Group related tests using pytest classes
- Use fixtures for common test setup and teardown
- Separate unit tests, integration tests, and performance tests

### Async Testing
- Always use `@pytest.mark.asyncio` for async tests
- Clean up resources in async test teardown
- Use proper timeout handling for async operations
- Mock external dependencies in unit tests

### Performance Testing
- Use `@pytest.mark.performance` for performance tests
- Set reasonable performance expectations
- Test under various load conditions
- Monitor resource usage during tests