"""
Jest-like Testing Patterns in Python using pytest

This module demonstrates how pytest + related packages provide Jest-equivalent functionality.
It serves as a practical guide showing the mapping between Jest features and Python testing tools.

Jest Feature Mapping:
- Test runner and assertions: Jest's expect(...) → plain assert in pytest
- Mocks/spies: jest.fn()/jest.mock → unittest.mock + pytest-mock
- Snapshots: Jest snapshots → syrupy
- Watch mode: jest --watch → pytest-watch (ptw)
- Coverage: jest --coverage → pytest --cov
- Parametrized tests: test.each → @pytest.mark.parametrize
- Setup/teardown: beforeEach/afterEach/beforeAll/afterAll → pytest fixtures
- Parallel runs: Jest's parallelism → pytest-xdist (pytest -n auto)
"""

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest


# ===== Sample Functions to Test (Jest equivalents) =====


def add(a: int, b: int) -> int:
    """Simple function for testing parametrized tests (like Jest's test.each)."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Another simple function for testing."""
    return a * b


class Calculator:
    """Sample class for testing mocking patterns."""

    def __init__(self):
        self.history: List[str] = []

    def add(self, a: int, b: int) -> int:
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def get_history(self) -> List[str]:
        return self.history.copy()


class ApiService:
    """Sample service for testing async mocking patterns."""

    async def fetch_user(self, user_id: int) -> Dict[str, Any]:
        """Simulates an API call."""
        # In real implementation, this would make an HTTP request
        await asyncio.sleep(0.1)  # Simulate network delay
        return {"id": user_id, "name": f"User {user_id}", "active": True}

    async def update_user(self, user_id: int, data: Dict[str, Any]) -> bool:
        """Simulates updating a user."""
        await asyncio.sleep(0.05)
        return True


def build_user_payload(user_id: int, name: str) -> Dict[str, Any]:
    """Sample function for snapshot testing."""
    return {
        "version": 1,
        "user": {
            "id": user_id,
            "name": name,
            "created_at": "2024-01-01T00:00:00Z",  # Fixed for consistent snapshots
            "preferences": {"theme": "dark", "notifications": True},
        },
        "metadata": {"generated_by": "super-alita", "format": "json"},
    }


# ===== Parametrized Tests (Jest's test.each equivalent) =====


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 2, 3),
        (0, 0, 0),
        (-1, 5, 4),
        (10, -3, 7),
        (100, 200, 300),
    ],
)
def test_add_parametrized(a: int, b: int, expected: int):
    """
    Jest equivalent:
    test.each([
        [1, 2, 3],
        [0, 0, 0],
        [-1, 5, 4],
    ])('add(%i, %i) should return %i', (a, b, expected) => {
        expect(add(a, b)).toBe(expected);
    });
    """
    assert add(a, b) == expected


@pytest.mark.parametrize(
    "operation,a,b,expected",
    [
        ("add", 5, 3, 8),
        ("multiply", 4, 7, 28),
        ("add", -2, 6, 4),
        ("multiply", 0, 5, 0),
    ],
)
def test_calculator_operations(operation: str, a: int, b: int, expected: int):
    """Multiple operations with parametrized tests."""
    calc = Calculator()

    if operation == "add":
        result = calc.add(a, b)
    elif operation == "multiply":
        result = multiply(a, b)  # Using standalone function
    else:
        pytest.fail(f"Unknown operation: {operation}")

    assert result == expected


# ===== Mocking and Spying (Jest's jest.fn/jest.mock equivalent) =====


def test_calculator_with_mock():
    """
    Jest equivalent:
    const mockAdd = jest.fn((a, b) => a + b);
    const calc = new Calculator();
    calc.add = mockAdd;

    const result = calc.add(5, 3);
    expect(mockAdd).toHaveBeenCalledWith(5, 3);
    expect(mockAdd).toHaveBeenCalledTimes(1);
    expect(result).toBe(8);
    """
    calc = Calculator()

    # Mock the add method
    calc.add = Mock(return_value=8)

    result = calc.add(5, 3)

    # Assertions similar to Jest's expect
    calc.add.assert_called_once_with(5, 3)
    assert calc.add.call_count == 1
    assert result == 8


def test_calculator_spy_on_existing_method():
    """
    Jest equivalent:
    const calc = new Calculator();
    const addSpy = jest.spyOn(calc, 'add');

    calc.add(2, 3);
    expect(addSpy).toHaveBeenCalledWith(2, 3);
    expect(addSpy).toHaveReturnedWith(5);
    """
    calc = Calculator()

    with patch.object(calc, "add", wraps=calc.add) as add_spy:
        result = calc.add(2, 3)

        add_spy.assert_called_once_with(2, 3)
        assert result == 5
        assert len(calc.history) == 1
        assert "2 + 3 = 5" in calc.history[0]


def test_mock_with_side_effect():
    """
    Jest equivalent:
    const mockFn = jest.fn()
        .mockReturnValueOnce(10)
        .mockReturnValueOnce(20)
        .mockImplementation(() => { throw new Error('No more values'); });
    """
    mock_fn = Mock(side_effect=[10, 20, ValueError("No more values")])

    assert mock_fn() == 10
    assert mock_fn() == 20

    with pytest.raises(ValueError, match="No more values"):
        mock_fn()


# ===== Async Testing with Mocks =====


@pytest.mark.asyncio
async def test_async_service_mock():
    """
    Jest equivalent:
    const mockFetchUser = jest.fn().mockResolvedValue({ id: 1, name: 'Ada' });
    const service = new ApiService();
    service.fetchUser = mockFetchUser;

    const result = await service.fetchUser(1);
    expect(mockFetchUser).toHaveBeenCalledWith(1);
    expect(result.name).toBe('Ada');
    """
    service = ApiService()

    # Mock the async method
    service.fetch_user = AsyncMock(
        return_value={"id": 1, "name": "Ada", "active": True}
    )

    result = await service.fetch_user(1)

    service.fetch_user.assert_called_once_with(1)
    assert result["name"] == "Ada"
    assert result["id"] == 1


@pytest.mark.asyncio
async def test_async_service_with_patch():
    """Using patch decorator for async mocking."""
    with patch.object(ApiService, "fetch_user", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {"id": 42, "name": "Test User", "active": False}

        service = ApiService()
        result = await service.fetch_user(42)

        mock_fetch.assert_called_once_with(42)
        assert result["name"] == "Test User"
        assert not result["active"]


# ===== Snapshot Testing (Jest snapshots equivalent) =====


def test_user_payload_snapshot(snapshot):
    """
    Jest equivalent:
    test('builds user payload', () => {
        const payload = buildUserPayload(1, 'Alice');
        expect(payload).toMatchSnapshot();
    });

    Uses syrupy for snapshot testing - run with: pytest --snapshot-update
    """
    payload = build_user_payload(1, "Alice")
    assert payload == snapshot


def test_multiple_user_payloads_snapshot(snapshot):
    """Test multiple snapshots in one test."""
    users = [
        build_user_payload(1, "Alice"),
        build_user_payload(2, "Bob"),
        build_user_payload(3, "Charlie"),
    ]

    for i, user_payload in enumerate(users):
        assert user_payload == snapshot(name=f"user_{i+1}")


# ===== Setup and Teardown (Jest's beforeEach/afterEach equivalent) =====


@pytest.fixture
def calculator():
    """
    Jest equivalent:
    beforeEach(() => {
        calculator = new Calculator();
    });

    This fixture runs before each test method that uses it.
    """
    calc = Calculator()
    yield calc
    # Teardown code would go here (equivalent to afterEach)


@pytest.fixture(scope="module")
def shared_api_service():
    """
    Jest equivalent:
    beforeAll(() => {
        apiService = new ApiService();
    });

    This fixture runs once per module (equivalent to beforeAll/afterAll).
    """
    service = ApiService()
    yield service
    # Module-level teardown would go here


def test_calculator_fixture_usage(calculator):
    """Test using the calculator fixture."""
    result = calculator.add(10, 15)
    assert result == 25
    assert len(calculator.get_history()) == 1


def test_calculator_fixture_isolation(calculator):
    """Each test gets a fresh calculator instance."""
    # This calculator should be empty, proving isolation
    assert len(calculator.get_history()) == 0

    calculator.add(1, 1)
    assert len(calculator.get_history()) == 1


# ===== Performance and Timing Tests =====


def test_operation_performance():
    """
    Jest equivalent:
    test('operation should be fast', () => {
        const start = Date.now();
        performOperation();
        const duration = Date.now() - start;
        expect(duration).toBeLessThan(100);
    });
    """
    start_time = time.time()

    # Simulate some operation
    result = add(1000, 2000)

    duration = (time.time() - start_time) * 1000  # Convert to milliseconds

    assert result == 3000
    assert duration < 100, f"Operation took {duration}ms, expected < 100ms"


# ===== Error Testing =====


def test_error_handling():
    """
    Jest equivalent:
    test('should throw error for invalid input', () => {
        expect(() => {
            processInvalidData();
        }).toThrow('Invalid data');
    });
    """

    def divide_by_zero():
        return 1 / 0

    with pytest.raises(ZeroDivisionError):
        divide_by_zero()


def test_specific_error_message():
    """Test specific error messages."""

    def validate_user(name: str):
        if not name.strip():
            raise ValueError("Name cannot be empty")
        return True

    with pytest.raises(ValueError, match="Name cannot be empty"):
        validate_user("  ")


# ===== Grouping Tests (Jest's describe equivalent) =====


class TestCalculatorFeatures:
    """
    Jest equivalent:
    describe('Calculator Features', () => {
        // tests go here
    });

    Python uses classes to group related tests.
    """

    def test_basic_addition(self):
        calc = Calculator()
        assert calc.add(2, 3) == 5

    def test_history_tracking(self):
        calc = Calculator()
        calc.add(1, 2)
        calc.add(3, 4)

        history = calc.get_history()
        assert len(history) == 2
        assert "1 + 2 = 3" in history[0]
        assert "3 + 4 = 7" in history[1]


class TestAsyncApiService:
    """Group async service tests."""

    @pytest.mark.asyncio
    async def test_fetch_user_success(self):
        service = ApiService()

        with patch.object(
            service, "fetch_user", return_value={"id": 1, "name": "Test"}
        ):
            result = await service.fetch_user(1)
            assert result["name"] == "Test"

    @pytest.mark.asyncio
    async def test_update_user_success(self):
        service = ApiService()

        with patch.object(service, "update_user", return_value=True):
            success = await service.update_user(1, {"name": "Updated"})
            assert success is True


# ===== Custom Matchers (Jest's expect.extend equivalent) =====


def assert_between(value: float, min_val: float, max_val: float, message: str = ""):
    """
    Jest equivalent:
    expect.extend({
        toBeBetween(received, min, max) {
            const pass = received >= min && received <= max;
            return { pass, message: () => `expected ${received} to be between ${min} and ${max}` };
        }
    });
    """
    if not (min_val <= value <= max_val):
        raise AssertionError(
            message or f"Expected {value} to be between {min_val} and {max_val}"
        )


def test_custom_assertion():
    """Using custom assertion helper."""
    result = add(5, 7)
    assert_between(result, 10, 15, "Addition result should be in expected range")


# ===== Test Configuration and Markers =====


@pytest.mark.slow
def test_slow_operation():
    """
    Jest equivalent:
    test('slow operation', () => {
        // test implementation
    }, 30000); // 30 second timeout

    Use: pytest -m "not slow" to skip slow tests
    """
    time.sleep(0.1)  # Simulate slow operation
    assert True


@pytest.mark.unit
def test_unit_example():
    """Unit test marker - run with: pytest -m unit"""
    assert add(1, 1) == 2


@pytest.mark.integration
def test_integration_example():
    """Integration test marker - run with: pytest -m integration"""
    calc = Calculator()
    result = calc.add(5, 5)
    assert result == 10
    assert len(calc.get_history()) == 1
