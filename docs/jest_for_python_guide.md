# Jest for Python: Complete Feature Mapping Guide

This guide demonstrates how to achieve Jest-like testing functionality in Python using pytest and related packages. The Super Alita repository showcases these patterns in practice.

## Quick Setup

```bash
# Install the Jest-equivalent package stack
pip install pytest pytest-asyncio pytest-cov pytest-mock syrupy pytest-xdist pytest-watch

# Basic test run
pytest

# With coverage (Jest's --coverage equivalent)
pytest --cov=src --cov-report=term-missing

# Parallel execution (Jest's built-in parallelism equivalent)
pytest -n auto

# Watch mode (Jest's --watch equivalent)
ptw

# Filter tests (Jest's pattern matching equivalent)
pytest -k "test_add"
```

## Feature Mapping Table

| Jest Feature | Python Equivalent | Command/Usage |
|-------------|------------------|---------------|
| **Test Runner** | pytest | `pytest` |
| **Assertions** | `assert` with rich diff | `assert result == expected` |
| **Mocking** | `unittest.mock` + `pytest-mock` | `from unittest.mock import Mock` |
| **Spying** | `patch` with `wraps` | `patch.object(obj, 'method', wraps=obj.method)` |
| **Snapshots** | `syrupy` | `assert data == snapshot` |
| **Watch Mode** | `pytest-watch` | `ptw` |
| **Coverage** | `pytest-cov` | `pytest --cov=package` |
| **Parallel** | `pytest-xdist` | `pytest -n auto` |
| **Parametrized** | `@pytest.mark.parametrize` | `@pytest.mark.parametrize("a,b", [(1,2), (3,4)])` |
| **Setup/Teardown** | pytest fixtures | `@pytest.fixture` |
| **Grouping** | test classes | `class TestFeature:` |
| **Async Testing** | `pytest-asyncio` | `@pytest.mark.asyncio` |

## Detailed Examples

### 1. Test Runner and Assertions

**Jest:**
```javascript
test('addition works', () => {
    expect(add(2, 3)).toBe(5);
    expect(add(0, 0)).toBe(0);
});
```

**Python:**
```python
def test_addition_works():
    assert add(2, 3) == 5
    assert add(0, 0) == 0
```

### 2. Parametrized Tests (test.each equivalent)

**Jest:**
```javascript
test.each([
    [1, 2, 3],
    [0, 0, 0],
    [-1, 5, 4],
])('add(%i, %i) should return %i', (a, b, expected) => {
    expect(add(a, b)).toBe(expected);
});
```

**Python:**
```python
@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 5, 4),
])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

### 3. Mocking and Spying

**Jest:**
```javascript
test('mocks function call', () => {
    const mockFn = jest.fn(() => 'mocked');
    const result = mockFn('arg1', 'arg2');
    
    expect(mockFn).toHaveBeenCalledWith('arg1', 'arg2');
    expect(mockFn).toHaveBeenCalledTimes(1);
    expect(result).toBe('mocked');
});
```

**Python:**
```python
def test_mocks_function_call():
    mock_fn = Mock(return_value='mocked')
    result = mock_fn('arg1', 'arg2')
    
    mock_fn.assert_called_with('arg1', 'arg2')
    assert mock_fn.call_count == 1
    assert result == 'mocked'
```

**Jest Spy:**
```javascript
test('spies on existing method', () => {
    const service = new ApiService();
    const spy = jest.spyOn(service, 'fetchUser').mockResolvedValue({id: 1});
    
    await service.fetchUser(1);
    expect(spy).toHaveBeenCalledWith(1);
});
```

**Python Spy:**
```python
def test_spies_on_existing_method():
    service = ApiService()
    
    with patch.object(service, 'fetch_user', wraps=service.fetch_user) as spy:
        await service.fetch_user(1)
        spy.assert_called_with(1)
```

### 4. Snapshot Testing

**Jest:**
```javascript
test('user payload matches snapshot', () => {
    const payload = buildUserPayload(1, 'Alice');
    expect(payload).toMatchSnapshot();
});
```

**Python:**
```python
def test_user_payload_matches_snapshot(snapshot):
    payload = build_user_payload(1, 'Alice')
    assert payload == snapshot
```

**Update snapshots:** `pytest --snapshot-update`

### 5. Async Testing

**Jest:**
```javascript
test('async function works', async () => {
    const result = await fetchData();
    expect(result).toBeDefined();
});
```

**Python:**
```python
@pytest.mark.asyncio
async def test_async_function_works():
    result = await fetch_data()
    assert result is not None
```

### 6. Setup and Teardown

**Jest:**
```javascript
describe('Calculator Tests', () => {
    let calculator;
    
    beforeEach(() => {
        calculator = new Calculator();
    });
    
    afterEach(() => {
        calculator.cleanup();
    });
    
    test('addition works', () => {
        expect(calculator.add(2, 3)).toBe(5);
    });
});
```

**Python:**
```python
class TestCalculator:
    @pytest.fixture(autouse=True)
    def setup_calculator(self):
        self.calculator = Calculator()
        yield
        self.calculator.cleanup()
    
    def test_addition_works(self):
        assert self.calculator.add(2, 3) == 5
```

**Or using fixtures:**
```python
@pytest.fixture
def calculator():
    calc = Calculator()
    yield calc
    calc.cleanup()

def test_addition_works(calculator):
    assert calculator.add(2, 3) == 5
```

### 7. Error Testing

**Jest:**
```javascript
test('throws error for invalid input', () => {
    expect(() => {
        divide(1, 0);
    }).toThrow('Division by zero');
});
```

**Python:**
```python
def test_throws_error_for_invalid_input():
    with pytest.raises(ValueError, match="Division by zero"):
        divide(1, 0)
```

### 8. Test Grouping

**Jest:**
```javascript
describe('User Management', () => {
    describe('Authentication', () => {
        test('login succeeds', () => {
            // test implementation
        });
    });
});
```

**Python:**
```python
class TestUserManagement:
    class TestAuthentication:
        def test_login_succeeds(self):
            # test implementation
            pass
```

## Advanced Usage

### Watch Mode with File Filtering

```bash
# Watch specific directories
ptw tests/ src/

# Watch with specific test patterns  
ptw --ignore=__pycache__ tests/
```

### Coverage with Different Formats

```bash
# Terminal coverage report
pytest --cov=src --cov-report=term-missing

# HTML coverage report
pytest --cov=src --cov-report=html

# JSON coverage report (for CI)
pytest --cov=src --cov-report=json
```

### Parallel Testing with Different Strategies

```bash
# Auto-detect number of CPUs
pytest -n auto

# Specific number of workers
pytest -n 4

# Distribute by test file
pytest --dist=loadfile

# Distribute by test function
pytest --dist=loadscope
```

### Custom Markers (Jest's test categories equivalent)

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests"
]
```

```python
@pytest.mark.slow
def test_slow_operation():
    pass

@pytest.mark.integration  
def test_database_integration():
    pass
```

```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Run integration tests only
pytest -m integration
```

## Performance Testing

**Jest:**
```javascript
test('operation should be fast', () => {
    const start = Date.now();
    performOperation();
    const duration = Date.now() - start;
    expect(duration).toBeLessThan(100);
});
```

**Python:**
```python
def test_operation_should_be_fast():
    import time
    start = time.time()
    perform_operation()
    duration = (time.time() - start) * 1000  # Convert to ms
    assert duration < 100
```

## Configuration Files

### pytest.ini
```ini
[pytest]
addopts = -ra -q --cov=src
testpaths = tests
asyncio_mode = auto
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
```

### pyproject.toml
```toml
[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --cov=src --cov-report=term-missing"
testpaths = ["tests"]
asyncio_mode = "auto"
```

## Common Patterns

### Mock External API Calls
```python
@patch('requests.get')
def test_api_call(mock_get):
    mock_get.return_value.json.return_value = {'status': 'success'}
    result = fetch_external_data()
    assert result['status'] == 'success'
```

### Test Async Code with Timeouts
```python
@pytest.mark.asyncio
@pytest.mark.timeout(5)  # 5 second timeout
async def test_async_with_timeout():
    result = await slow_async_operation()
    assert result is not None
```

### Fixture Scopes
```python
@pytest.fixture(scope="session")  # Once per test session
def database():
    pass

@pytest.fixture(scope="module")   # Once per test module
def api_client():
    pass

@pytest.fixture(scope="class")    # Once per test class
def user_service():
    pass

@pytest.fixture(scope="function") # Once per test function (default)
def temp_data():
    pass
```

## Running Tests (Jest Command Equivalents)

| Jest Command | pytest Equivalent | Description |
|-------------|------------------|-------------|
| `jest` | `pytest` | Run all tests |
| `jest --watch` | `ptw` | Watch mode |
| `jest --coverage` | `pytest --cov=src` | Coverage report |
| `jest --verbose` | `pytest -v` | Verbose output |
| `jest --silent` | `pytest -q` | Quiet output |
| `jest test.js` | `pytest test_file.py` | Run specific file |
| `jest --testNamePattern="add"` | `pytest -k "add"` | Filter by test name |
| `jest --maxWorkers=4` | `pytest -n 4` | Parallel execution |
| `jest --updateSnapshot` | `pytest --snapshot-update` | Update snapshots |

This comprehensive mapping shows how pytest and its ecosystem provide all the functionality of Jest, often with more flexibility and power. The examples in `tests/test_jest_like_patterns.py` demonstrate these patterns in practice.