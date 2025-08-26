# Testing Instructions

This directory contains unit and integration tests for the Super Alita system.

## Test Structure

The test directory mirrors the source code structure with comprehensive coverage:

- **Core Tests** (`core/`): Event bus, plugin interface, neural atom tests
- **Plugin Tests** (`plugins/`): Individual plugin validation  
- **Integration Tests** (`integration/`): End-to-end workflow testing
- **MCP Tests** (`mcp/`): Model Context Protocol testing
- **Performance Tests**: Load testing and concurrency validation

## Running Tests

### Quick Test Run
```bash
pytest -q
# or from root
make test
```

### Verbose Testing
```bash
pytest -v
```

### Specific Test Categories
```bash
# Core functionality
pytest tests/core/ -v

# Plugin system
pytest tests/plugins/ -v

# Integration tests
pytest tests/integration/ -v

# MCP server tests
pytest tests/mcp/ -v
```

### Async Testing
All async tests must use the pytest-asyncio marker:
```python
@pytest.mark.asyncio
async def test_async_function():
    # Test async functionality
    pass
```

### Performance and Concurrency Testing
```bash
# Concurrency tests
python run_concurrency_tests.py

# Load testing
pytest tests/core/test_concurrency.py::TestIntegrationScenarios::test_high_load_integration -v

# Circuit breaker tests
pytest tests/core/test_concurrency.py::TestCircuitBreaker -v
```

## Test Configuration

### Environment Setup
Tests use the configuration from `tests/conftest.py` which provides:
- Comprehensive test fixtures for event-driven testing
- Mock event bus for testing
- Temporary directories for file operations
- Test database configurations

### Test Data
- Use timezone-aware timestamps: `datetime.now(timezone.utc)`
- Prefer Pydantic models over dataclasses for test events
- Never use print statements in tests - use proper logging

## Adding Tests

### Test File Naming
- Test files must start with `test_`
- Match the module structure: `tests/core/test_event_bus.py`
- Use descriptive test function names: `test_event_creation_with_valid_data`

### Test Patterns
```python
import pytest
from datetime import datetime, timezone
from src.core.events import create_event

@pytest.mark.asyncio
async def test_event_flow():
    # Use timezone-aware timestamps
    timestamp = datetime.now(timezone.utc)
    
    # Create events with keyword args
    event = create_event("test_event", data={"key": "value"})
    
    # Assert expectations
    assert event.event_type == "test_event"
    assert event.data["key"] == "value"
```

### Parametrized Testing
Use pytest parametrize for edge cases:
```python
@pytest.mark.parametrize("input_data,expected", [
    ({"valid": True}, "success"),
    ({"valid": False}, "failure"),
    ({}, "default"),
])
def test_multiple_scenarios(input_data, expected):
    result = process_data(input_data)
    assert result == expected
```

## Test Categories

### Unit Tests
- Test individual functions and classes in isolation
- Mock external dependencies
- Focus on business logic validation

### Integration Tests  
- Test component interactions
- Use real event bus when possible
- Validate end-to-end workflows

### Performance Tests
- Load testing under high concurrency
- Memory usage validation
- Response time benchmarks

## Debugging Tests

### Running Single Tests
```bash
pytest tests/core/test_event_bus.py::test_specific_function -v -s
```

### Test Coverage
```bash
pytest --cov=src --cov-report=html
```

### Debug Mode
```bash
pytest --pdb  # Drop into debugger on failure
pytest -x     # Stop on first failure
```

## Continuous Integration

Tests are automatically run on:
- Pull request creation
- Push to main branch
- Scheduled nightly runs

All tests must pass before code can be merged to main branch.

## Test Dependencies

Core testing requirements are in `requirements-test.txt`:
- pytest
- pytest-asyncio
- pytest-cov
- pytest-mock

Install with:
```bash
pip install -r requirements-test.txt
```