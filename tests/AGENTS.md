# Testing Standards - Agent Instructions

## Overview
The `tests/` directory contains comprehensive test coverage for Super Alita:
- **Unit Tests** - Individual component testing
- **Integration Tests** - Cross-component interaction testing
- **End-to-End Tests** - Full system workflow testing
- **Performance Tests** - Load and stress testing
- **Security Tests** - Vulnerability and penetration testing

## Test Structure

### Directory Organization
```
tests/
├── unit/                    # Unit tests mirroring src/ structure
│   ├── core/               # Core component tests
│   ├── plugins/            # Plugin tests
│   ├── neural/             # Neural system tests
│   └── sandbox/            # Sandbox tests
├── integration/            # Integration tests
│   ├── event_bus/          # Event bus integration
│   ├── plugin_communication/ # Plugin interaction tests
│   └── external_services/  # External API integration
├── e2e/                    # End-to-end tests
│   ├── workflows/          # Complete workflow tests
│   └── scenarios/          # User scenario tests
├── performance/            # Performance and load tests
├── security/               # Security and penetration tests
├── fixtures/               # Test data and fixtures
└── conftest.py            # Pytest configuration and fixtures
```

## Testing Framework

### Pytest Configuration
```python
# conftest.py
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from src.core.event_bus import EventBus
from src.core.neural_atom import create_atom

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def mock_event_bus():
    """Mock event bus for testing"""
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()
    mock_bus.subscribe = MagicMock()
    return mock_bus

@pytest.fixture
async def test_neural_store():
    """In-memory neural store for testing"""
    from src.neural.store import InMemoryNeuralStore
    return InMemoryNeuralStore()

@pytest.fixture
def sample_neural_atom():
    """Sample neural atom for testing"""
    return create_atom(
        content={"test": "data"},
        atom_type="test_atom",
        title="Test Atom"
    )

@pytest.fixture
async def test_plugin(mock_event_bus):
    """Sample plugin for testing"""
    from src.plugins.plugin_interface import PluginInterface
    
    class TestPlugin(PluginInterface):
        def __init__(self, event_bus):
            super().__init__(event_bus)
            self.name = "test_plugin"
            
        async def shutdown(self):
            pass
    
    return TestPlugin(mock_event_bus)
```

## Unit Testing Guidelines

### Test Structure Pattern
```python
# tests/unit/core/test_neural_atom.py
import pytest
from unittest.mock import patch, MagicMock
from src.core.neural_atom import create_atom, generate_atom_uuid

class TestNeuralAtom:
    """Test cases for neural atom functionality"""
    
    def test_create_atom_basic(self):
        """Test basic atom creation"""
        content = {"key": "value"}
        atom = create_atom(content, "test_type", "Test Title")
        
        assert atom.content == content
        assert atom.atom_type == "test_type"
        assert atom.title == "Test Title"
        assert atom.uuid is not None
    
    def test_atom_uuid_deterministic(self):
        """Test that atoms with same content have same UUID"""
        content = {"key": "value"}
        
        atom1 = create_atom(content, "test_type", "Test")
        atom2 = create_atom(content, "test_type", "Test")
        
        assert atom1.uuid == atom2.uuid
    
    def test_atom_uuid_different_content(self):
        """Test that different content produces different UUIDs"""
        atom1 = create_atom({"key": "value1"}, "test_type", "Test")
        atom2 = create_atom({"key": "value2"}, "test_type", "Test")
        
        assert atom1.uuid != atom2.uuid
    
    @pytest.mark.parametrize("content,atom_type,title", [
        ({"test": 1}, "type1", "Title1"),
        (["item1", "item2"], "type2", "Title2"),
        ("string content", "type3", "Title3"),
    ])
    def test_atom_creation_various_content_types(self, content, atom_type, title):
        """Test atom creation with various content types"""
        atom = create_atom(content, atom_type, title)
        
        assert atom.content == content
        assert atom.atom_type == atom_type
        assert atom.title == title

class TestUUIDGeneration:
    """Test UUID generation for atoms"""
    
    def test_uuid_consistency(self):
        """Test UUID generation is consistent"""
        content = {"test": "data"}
        
        uuid1 = generate_atom_uuid(content, "test", "Title")
        uuid2 = generate_atom_uuid(content, "test", "Title")
        
        assert uuid1 == uuid2
    
    def test_uuid_uniqueness(self):
        """Test different inputs produce different UUIDs"""
        base_content = {"test": "data"}
        
        uuid1 = generate_atom_uuid(base_content, "type1", "Title")
        uuid2 = generate_atom_uuid(base_content, "type2", "Title")
        uuid3 = generate_atom_uuid(base_content, "type1", "Different Title")
        
        assert len({uuid1, uuid2, uuid3}) == 3  # All different
```

### Plugin Testing Pattern
```python
# tests/unit/plugins/test_calculator_plugin.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.plugins.calculator_plugin import CalculatorPlugin

class TestCalculatorPlugin:
    """Test calculator plugin functionality"""
    
    @pytest.fixture
    def calculator_plugin(self, mock_event_bus):
        """Calculator plugin instance for testing"""
        return CalculatorPlugin(mock_event_bus)
    
    @pytest.mark.asyncio
    async def test_basic_arithmetic(self, calculator_plugin):
        """Test basic arithmetic operations"""
        event = {
            "type": "tool_call",
            "data": {
                "tool_name": "calculator",
                "operation": "add",
                "operands": [2, 3]
            },
            "correlation_id": "test-123"
        }
        
        await calculator_plugin.handle_event(event)
        
        # Verify result event was published
        mock_event_bus = calculator_plugin.event_bus
        assert mock_event_bus.publish.called
        
        published_event = mock_event_bus.publish.call_args[0][0]
        assert published_event["type"] == "tool_result"
        assert published_event["data"]["result"] == 5
    
    @pytest.mark.asyncio
    async def test_division_by_zero(self, calculator_plugin):
        """Test division by zero handling"""
        event = {
            "type": "tool_call",
            "data": {
                "tool_name": "calculator",
                "operation": "divide",
                "operands": [10, 0]
            }
        }
        
        await calculator_plugin.handle_event(event)
        
        # Verify error event was published
        published_event = calculator_plugin.event_bus.publish.call_args[0][0]
        assert published_event["type"] == "tool_error"
        assert "division by zero" in published_event["data"]["error"].lower()
    
    @pytest.mark.asyncio
    async def test_plugin_shutdown(self, calculator_plugin):
        """Test plugin shutdown"""
        await calculator_plugin.shutdown()
        # Verify cleanup was performed
        assert calculator_plugin.is_shutdown
```

## Integration Testing Guidelines

### Event Bus Integration
```python
# tests/integration/test_event_bus_integration.py
import pytest
import asyncio
from src.core.event_bus import EventBus
from src.plugins.llm_planner_plugin import LLMPlannerPlugin
from src.plugins.tool_executor_plugin import ToolExecutorPlugin

@pytest.mark.integration
class TestEventBusIntegration:
    """Test event bus integration between plugins"""
    
    @pytest.fixture
    async def event_bus_system(self):
        """Set up event bus with real plugins"""
        event_bus = EventBus()
        
        planner = LLMPlannerPlugin(event_bus)
        executor = ToolExecutorPlugin(event_bus)
        
        yield event_bus, planner, executor
        
        # Cleanup
        await planner.shutdown()
        await executor.shutdown()
        await event_bus.shutdown()
    
    @pytest.mark.asyncio
    async def test_tool_execution_flow(self, event_bus_system):
        """Test complete tool execution flow"""
        event_bus, planner, executor = event_bus_system
        
        # Simulate user input
        user_event = {
            "type": "user_input",
            "data": {"query": "Calculate 2 + 3"},
            "correlation_id": "test-flow-123"
        }
        
        # Publish user input
        await event_bus.publish(user_event)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Verify tool call was generated and executed
        # This would check the actual event flow
        # Implementation depends on event bus internals
    
    @pytest.mark.asyncio
    async def test_error_propagation(self, event_bus_system):
        """Test error propagation through event system"""
        event_bus, planner, executor = event_bus_system
        
        # Inject invalid tool call
        invalid_event = {
            "type": "tool_call",
            "data": {"tool_name": "nonexistent_tool"},
            "correlation_id": "error-test-123"
        }
        
        await event_bus.publish(invalid_event)
        await asyncio.sleep(0.1)
        
        # Verify error event was generated
        # Check error handling and recovery
```

### Database Integration
```python
# tests/integration/test_neural_store_integration.py
import pytest
from src.neural.store import NeuralStore
from src.core.neural_atom import create_atom

@pytest.mark.integration
@pytest.mark.asyncio
class TestNeuralStoreIntegration:
    """Test neural store with real database"""
    
    @pytest.fixture
    async def neural_store(self):
        """Real neural store for integration testing"""
        store = NeuralStore(config={"database_url": "sqlite:///:memory:"})
        await store.initialize()
        yield store
        await store.cleanup()
    
    async def test_atom_storage_and_retrieval(self, neural_store):
        """Test storing and retrieving atoms"""
        atom = create_atom(
            content={"integration": "test"},
            atom_type="test_atom",
            title="Integration Test Atom"
        )
        
        # Store atom
        await neural_store.store_atom(atom)
        
        # Retrieve atom
        retrieved = await neural_store.get_atom(atom.uuid)
        
        assert retrieved.uuid == atom.uuid
        assert retrieved.content == atom.content
    
    async def test_similarity_search(self, neural_store):
        """Test semantic similarity search"""
        # Store multiple related atoms
        atoms = [
            create_atom({"topic": "cats", "content": "cats are pets"}, "text", "About Cats"),
            create_atom({"topic": "dogs", "content": "dogs are pets"}, "text", "About Dogs"),
            create_atom({"topic": "cars", "content": "cars are vehicles"}, "text", "About Cars"),
        ]
        
        for atom in atoms:
            await neural_store.store_atom(atom)
        
        # Search for similar atoms
        query_atom = create_atom(
            {"topic": "pets", "content": "animals that live with humans"},
            "text",
            "Pet Query"
        )
        
        similar = await neural_store.find_similar_atoms(
            reference_atom=query_atom,
            similarity_threshold=0.5,
            max_results=10
        )
        
        # Should find cat and dog atoms as more similar than car atom
        assert len(similar) >= 2
        topics = [atom.content.get("topic") for atom in similar[:2]]
        assert "cats" in topics or "dogs" in topics
```

## End-to-End Testing

### Workflow Testing
```python
# tests/e2e/test_complete_workflows.py
import pytest
import asyncio
from src.main import SuperAlitaSystem

@pytest.mark.e2e
class TestCompleteWorkflows:
    """Test complete system workflows"""
    
    @pytest.fixture
    async def system(self):
        """Complete Super Alita system for E2E testing"""
        system = SuperAlitaSystem(config={"mode": "test"})
        await system.initialize()
        yield system
        await system.shutdown()
    
    @pytest.mark.asyncio
    async def test_simple_calculation_workflow(self, system):
        """Test simple calculation from user input to response"""
        # Send user query
        response = await system.process_user_input("What is 15 * 8?")
        
        # Verify response
        assert response["success"] is True
        assert "120" in response["content"]
        assert response["confidence"] > 0.8
    
    @pytest.mark.asyncio
    async def test_complex_research_workflow(self, system):
        """Test complex research workflow"""
        query = "Research the latest developments in quantum computing"
        
        response = await system.process_user_input(query)
        
        # Verify comprehensive response
        assert response["success"] is True
        assert len(response["content"]) > 500  # Substantial content
        assert "quantum" in response["content"].lower()
        
        # Check that research atoms were created
        research_atoms = await system.neural_store.query_atoms(
            atom_type="research_result"
        )
        assert len(research_atoms) > 0
    
    @pytest.mark.asyncio
    async def test_tool_creation_workflow(self, system):
        """Test dynamic tool creation workflow"""
        request = "Create a tool that converts Celsius to Fahrenheit"
        
        response = await system.process_user_input(request)
        
        # Verify tool was created
        assert response["success"] is True
        assert "tool created" in response["content"].lower()
        
        # Test the created tool
        test_response = await system.process_user_input("Convert 20°C to Fahrenheit")
        assert "68" in test_response["content"]  # 20°C = 68°F
```

### User Scenario Testing
```python
# tests/e2e/test_user_scenarios.py
@pytest.mark.e2e
class TestUserScenarios:
    """Test realistic user scenarios"""
    
    @pytest.mark.asyncio
    async def test_developer_workflow(self, system):
        """Test typical developer workflow"""
        # Scenario: Developer analyzing code and getting suggestions
        
        # 1. Code analysis request
        response1 = await system.process_user_input(
            "Analyze this Python function for potential improvements: " +
            "def slow_function(data): return [x*2 for x in data if x > 0]"
        )
        
        assert "improvement" in response1["content"].lower()
        
        # 2. Follow-up optimization request
        response2 = await system.process_user_input(
            "Show me the optimized version of that function"
        )
        
        assert "def" in response2["content"]  # Should contain code
        
        # 3. Test the continuity of conversation
        assert system.get_conversation_context() is not None
    
    @pytest.mark.asyncio
    async def test_researcher_workflow(self, system):
        """Test researcher workflow with information synthesis"""
        
        # 1. Initial research query
        response1 = await system.process_user_input(
            "What are the main challenges in renewable energy storage?"
        )
        
        # 2. Follow-up for specific technology
        response2 = await system.process_user_input(
            "Tell me more about battery technology solutions"
        )
        
        # 3. Synthesis request
        response3 = await system.process_user_input(
            "Summarize the key points from our discussion"
        )
        
        # Verify knowledge was accumulated and synthesized
        assert "storage" in response3["content"].lower()
        assert "battery" in response3["content"].lower()
```

## Performance Testing

### Load Testing
```python
# tests/performance/test_load.py
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

@pytest.mark.performance
class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, system):
        """Test system under concurrent load"""
        
        async def single_request(query_id):
            query = f"Calculate {query_id} * 2"
            start_time = time.time()
            response = await system.process_user_input(query)
            duration = time.time() - start_time
            return {"success": response["success"], "duration": duration}
        
        # Run 50 concurrent requests
        tasks = [single_request(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        durations = [r["duration"] for r in successful]
        
        assert len(successful) >= 45  # At least 90% success rate
        assert max(durations) < 30.0  # No request takes more than 30s
        assert sum(durations) / len(durations) < 5.0  # Average under 5s
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, system):
        """Test memory usage under load"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process many requests to test memory leaks
        for i in range(100):
            await system.process_user_input(f"Test query {i}")
            
            if i % 10 == 0:  # Check every 10 requests
                gc.collect()  # Force garbage collection
                current_memory = process.memory_info().rss
                memory_growth = current_memory - initial_memory
                
                # Memory should not grow excessively
                assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth
    
    def test_plugin_initialization_time(self):
        """Test plugin initialization performance"""
        from src.core.plugin_loader import PluginLoader
        
        start_time = time.time()
        loader = PluginLoader()
        plugins = loader.load_all_plugins()
        initialization_time = time.time() - start_time
        
        assert initialization_time < 5.0  # Should initialize quickly
        assert len(plugins) > 0  # Should find plugins
```

## Security Testing

### Security Test Suite
```python
# tests/security/test_security.py
import pytest
from src.sandbox.exec_sandbox import execute_sandboxed_code

@pytest.mark.security
class TestSecurity:
    """Security and vulnerability testing"""
    
    @pytest.mark.asyncio
    async def test_code_injection_prevention(self):
        """Test that code injection attacks are prevented"""
        
        malicious_codes = [
            "import os; os.system('rm -rf /')",
            "__import__('subprocess').call(['rm', '-rf', '/'])",
            "eval('malicious_code')",
            "exec('dangerous_operation()')",
            "open('/etc/passwd', 'r').read()",
        ]
        
        for malicious_code in malicious_codes:
            result = await execute_sandboxed_code(malicious_code, timeout=5)
            assert not result.success, f"Malicious code was executed: {malicious_code}"
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_prevention(self):
        """Test that resource exhaustion attacks are prevented"""
        
        # Memory bomb
        memory_bomb = "data = 'x' * (1024 * 1024 * 1024)"  # 1GB
        result = await execute_sandboxed_code(memory_bomb, memory_limit_mb=100)
        assert not result.success
        
        # CPU bomb
        cpu_bomb = "while True: pass"
        result = await execute_sandboxed_code(cpu_bomb, timeout=2)
        assert not result.success
        
        # Fork bomb (if subprocess access somehow available)
        fork_bomb = "import os; [os.fork() for _ in range(1000)]"
        result = await execute_sandboxed_code(fork_bomb, timeout=5)
        assert not result.success
    
    def test_input_validation(self):
        """Test input validation across system"""
        from src.core.validation import validate_user_input
        
        # Test various injection attempts
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "\x00\x01\x02",  # Null bytes
            "A" * 10000,     # Extremely long input
        ]
        
        for malicious_input in malicious_inputs:
            with pytest.raises(ValueError):
                validate_user_input(malicious_input)
    
    @pytest.mark.asyncio
    async def test_plugin_isolation(self, system):
        """Test that plugins cannot access each other directly"""
        
        # This test would verify plugin isolation
        # Implementation depends on plugin architecture
        pass
```

## Test Automation

### CI/CD Integration
```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1

  integration-tests:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:latest
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v2
      - name: Run integration tests
        run: pytest tests/integration/ -v -m integration

  security-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run security tests
        run: pytest tests/security/ -v -m security
```

### Test Coverage Reporting
```python
# scripts/test_coverage.py
import subprocess
import sys

def run_coverage_analysis():
    """Run comprehensive test coverage analysis"""
    
    # Run tests with coverage
    result = subprocess.run([
        "pytest", 
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term",
        "--cov-fail-under=80"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Tests failed or coverage below 80%")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    
    print("All tests passed with adequate coverage")

if __name__ == "__main__":
    run_coverage_analysis()
```

## Test Data Management

### Fixture Management
```python
# tests/fixtures/data_fixtures.py
import json
from pathlib import Path

def load_test_data(filename: str):
    """Load test data from fixtures directory"""
    fixtures_dir = Path(__file__).parent
    with open(fixtures_dir / filename) as f:
        return json.load(f)

def create_test_atoms():
    """Create standard test atoms for testing"""
    test_data = load_test_data("sample_atoms.json")
    return [create_atom(**atom_data) for atom_data in test_data]

# Sample test data files
# tests/fixtures/sample_atoms.json
[
    {
        "content": {"type": "calculation", "result": 42},
        "atom_type": "tool_output",
        "title": "Test Calculation"
    },
    {
        "content": {"query": "test query", "response": "test response"},
        "atom_type": "conversation",
        "title": "Test Conversation"
    }
]
```

## Best Practices

### Test Organization
- **Mirror source structure** - Tests should mirror the `src/` directory structure
- **Clear naming** - Test names should clearly describe what they test
- **Independence** - Tests should not depend on each other
- **Cleanup** - Always clean up resources in test teardown

### Test Quality
- **Single responsibility** - Each test should test one thing
- **Comprehensive coverage** - Test normal, edge, and error cases
- **Realistic data** - Use realistic test data, not just simple cases
- **Performance awareness** - Keep tests fast to encourage frequent running