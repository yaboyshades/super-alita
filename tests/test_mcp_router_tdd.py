#!/usr/bin/env python3
"""
Test-Driven MCP Integration Pack

This test suite defines the exact behavior we want from the MCP router
BEFORE implementing any code. This ensures we have clear requirements
and prevents debugging rabbit holes.

Testing Strategy:
- Frozen monotonic clock (no wall-clock sleeps)
- Event count assertions for observability
- Cache/single-flight invariants
- Deterministic behavior for CI/CD
- Property-based testing where appropriate
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

# Mock imports that we'll implement after tests pass
# This forces us to define the exact interface we need


class MockClock:
    """Frozen monotonic clock for deterministic testing"""

    def __init__(self, start_time: float = 1000.0):
        self._time = start_time

    def now(self) -> float:
        return self._time

    def advance(self, delta: float) -> None:
        self._time += delta


class MockEventBus:
    """Mock event bus that captures emitted events"""

    def __init__(self):
        self.events: list[dict[str, Any]] = []

    async def emit(self, event_type: str, **kwargs):
        event = {
            "type": event_type,
            "timestamp": datetime.now(UTC).isoformat(),
            **kwargs,
        }
        self.events.append(event)

    def get_events_by_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == event_type]

    def clear(self):
        self.events.clear()


class MockTLM:
    """Mock Tool Lifecycle Manager"""

    def __init__(self):
        self.instances = {
            "calculator": ("calc-001", "http://calc.local:8080"),
            "weather": ("weather-002", "http://weather.local:8080"),
            "invalid_tool": None,  # For testing tool not found
        }

    async def select_instance(self, tool: str) -> tuple[str, str] | None:
        return self.instances.get(tool)


class MockCircuitBreaker:
    """Mock circuit breaker with configurable behavior"""

    def __init__(self, should_fail: bool = False, should_open: bool = False):
        self.should_fail = should_fail
        self.should_open = should_open
        self.call_count = 0

    async def __aenter__(self):
        self.call_count += 1
        if self.should_open:
            from src.core.error_recovery import CircuitBreakerOpenError

            raise CircuitBreakerOpenError("Circuit breaker is open")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.should_fail and exc_type is None:
            # Simulate call failure
            raise Exception("Simulated call failure")


class MockMCPClient:
    """Mock MCP client with configurable responses"""

    def __init__(self):
        self.call_responses = {}
        self.call_count = 0
        self.calls_made = []

    def set_response(self, tool: str, response: dict[str, Any]):
        self.call_responses[tool] = response

    async def call_tool(
        self, base_url: str, tool: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        self.call_count += 1
        self.calls_made.append({"base_url": base_url, "tool": tool, "args": args})

        if tool in self.call_responses:
            return self.call_responses[tool]

        # Default response
        return {
            "success": True,
            "result": f"Mock result for {tool} with args {args}",
            "execution_time": 0.1,
        }


class MockCache:
    """Mock cache with hit/miss tracking"""

    def __init__(self):
        self.data = {}
        self.hits = 0
        self.misses = 0
        self.gets = []
        self.sets = []

    async def get(self, key: str) -> Any | None:
        self.gets.append(key)
        if key in self.data:
            self.hits += 1
            return self.data[key]
        else:
            self.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: float | None = None):
        self.sets.append({"key": key, "value": value, "ttl": ttl})
        self.data[key] = value

    def clear(self):
        self.data.clear()
        self.hits = 0
        self.misses = 0
        self.gets.clear()
        self.sets.clear()


# Test fixtures
@pytest.fixture
def mock_clock():
    return MockClock()


@pytest.fixture
def mock_event_bus():
    return MockEventBus()


@pytest.fixture
def mock_tlm():
    return MockTLM()


@pytest.fixture
def mock_breaker_factory():
    def factory(tool_key: str) -> MockCircuitBreaker:
        return MockCircuitBreaker()

    return factory


@pytest.fixture
def mock_mcp_client():
    return MockMCPClient()


@pytest.fixture
def mock_cache():
    return MockCache()


@pytest.fixture
async def mcp_router(
    mock_event_bus,
    mock_tlm,
    mock_breaker_factory,
    mock_mcp_client,
    mock_cache,
    mock_clock,
):
    """
    This will be our actual MCPRouter once implemented.
    For now, this defines the interface we need.
    """
    # Import the real MCPRouter
    from src.mcp.clients import MCPClientPool
    from src.mcp.router import MCPRouter

    # Create client pool with our mock client
    client_pool = MCPClientPool(clock=mock_clock)
    client_pool._clients = {
        "http://calc.local:8080": mock_mcp_client,
        "http://weather.local:8080": mock_mcp_client,
    }

    # Override get_client to always return our mock
    async def mock_get_client(base_url: str):
        return mock_mcp_client

    client_pool.get_client = mock_get_client

    return MCPRouter(
        tlm=mock_tlm,
        bus=mock_event_bus,
        client=client_pool,
        breaker_factory=mock_breaker_factory,
        cache=mock_cache,
        clock=mock_clock,
    )


# TEST SUITE: Define exact behavior we want


class TestMCPRouterBasicInvocation:
    """Test basic tool invocation flow"""

    @pytest.mark.asyncio
    async def test_successful_tool_invocation(
        self, mcp_router, mock_event_bus, mock_mcp_client, mock_cache
    ):
        """Test successful tool invocation emits correct events"""
        # Setup
        mock_mcp_client.set_response("calculator", {"result": 42, "success": True})

        # Execute
        result = await mcp_router.invoke(
            "calculator", {"operation": "add", "a": 1, "b": 2}, "test-correlation-123"
        )

        # Verify result
        assert result["result"] == 42
        assert result["success"] is True

        # Verify events
        started_events = mock_event_bus.get_events_by_type("tool_invocation_started")
        completed_events = mock_event_bus.get_events_by_type(
            "tool_invocation_completed"
        )

        assert len(started_events) == 1
        assert len(completed_events) == 1

        started = started_events[0]
        assert started["tool"] == "calculator"
        assert started["correlation_id"] == "test-correlation-123"
        assert started["instance_id"] == "calc-001"
        assert started["args"] == {"operation": "add", "a": 1, "b": 2}

        completed = completed_events[0]
        assert completed["tool"] == "calculator"
        assert completed["correlation_id"] == "test-correlation-123"
        assert completed["result"]["result"] == 42
        assert "duration_ms" in completed

    @pytest.mark.asyncio
    async def test_tool_not_found_failure(self, mcp_router, mock_event_bus):
        """Test tool not found scenario"""
        with pytest.raises(
            ValueError, match="No instance available for tool: invalid_tool"
        ):
            await mcp_router.invoke("invalid_tool", {}, "test-correlation-456")

        # Verify failure event
        failed_events = mock_event_bus.get_events_by_type("tool_invocation_failed")
        assert len(failed_events) == 1

        failed = failed_events[0]
        assert failed["tool"] == "invalid_tool"
        assert failed["correlation_id"] == "test-correlation-456"
        assert "No instance available" in failed["error"]


class TestMCPRouterCaching:
    """Test caching behavior and cache key generation"""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_execution(
        self, mcp_router, mock_event_bus, mock_cache, mock_mcp_client
    ):
        """Test that cache hits skip tool execution"""
        # Pre-populate cache
        cache_key = mcp_router._generate_cache_key("calculator", {"a": 1, "b": 2})
        await mock_cache.set(cache_key, {"result": "cached_value", "success": True})

        # Execute
        result = await mcp_router.invoke(
            "calculator", {"a": 1, "b": 2}, "cache-test-123"
        )

        # Verify cached result returned
        assert result["result"] == "cached_value"

        # Verify no tool execution occurred
        assert mock_mcp_client.call_count == 0

        # Verify cache hit event
        cached_events = mock_event_bus.get_events_by_type("tool_invocation_cached")
        assert len(cached_events) == 1

        cached = cached_events[0]
        assert cached["tool"] == "calculator"
        assert cached["correlation_id"] == "cache-test-123"
        assert cached["cache_key"] == cache_key

    @pytest.mark.asyncio
    async def test_deterministic_cache_keys(self, mcp_router):
        """Test that cache keys are deterministic regardless of argument order"""
        # Different argument orders should produce same cache key
        key1 = mcp_router._generate_cache_key(
            "calculator", {"a": 1, "b": 2, "op": "add"}
        )
        key2 = mcp_router._generate_cache_key(
            "calculator", {"b": 2, "op": "add", "a": 1}
        )
        key3 = mcp_router._generate_cache_key(
            "calculator", {"op": "add", "a": 1, "b": 2}
        )

        assert key1 == key2 == key3

        # Different args should produce different keys
        key4 = mcp_router._generate_cache_key(
            "calculator", {"a": 1, "b": 3, "op": "add"}
        )
        assert key1 != key4

        # Different tools should produce different keys
        key5 = mcp_router._generate_cache_key("weather", {"a": 1, "b": 2, "op": "add"})
        assert key1 != key5

    @pytest.mark.asyncio
    async def test_cache_population_after_execution(
        self, mcp_router, mock_cache, mock_mcp_client
    ):
        """Test that successful executions populate the cache"""
        # Setup
        mock_mcp_client.set_response("calculator", {"result": 100, "success": True})

        # Execute
        await mcp_router.invoke(
            "calculator", {"operation": "multiply", "a": 10, "b": 10}
        )

        # Verify cache was populated
        assert len(mock_cache.sets) == 1
        cache_set = mock_cache.sets[0]

        assert cache_set["value"]["result"] == 100
        assert cache_set["ttl"] == 300.0  # 5 minute TTL


class TestMCPRouterSingleFlight:
    """Test single-flight coalescing to prevent thundering herd"""

    @pytest.mark.asyncio
    async def test_concurrent_identical_requests_coalesced(
        self, mcp_router, mock_mcp_client
    ):
        """Test that concurrent identical requests are coalesced into single execution"""
        # Setup slow response to ensure concurrency
        original_call_tool = mock_mcp_client.call_tool

        async def slow_call_tool(base_url, tool, args):
            # Call the original to maintain call count tracking
            mock_mcp_client.call_count += 1
            mock_mcp_client.calls_made.append(
                {"base_url": base_url, "tool": tool, "args": args}
            )
            await asyncio.sleep(0.01)  # Small delay to ensure concurrency
            return {"result": "single_execution", "success": True}

        mock_mcp_client.call_tool = slow_call_tool

        # Execute multiple concurrent identical requests
        tasks = [
            mcp_router.invoke("calculator", {"a": 5, "b": 5}, f"concurrent-{i}")
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # Verify all got same result
        for result in results:
            assert result["result"] == "single_execution"

        # Verify only one actual tool execution occurred
        assert mock_mcp_client.call_count == 1

    @pytest.mark.asyncio
    async def test_different_requests_not_coalesced(self, mcp_router, mock_mcp_client):
        """Test that different requests are not coalesced"""
        mock_mcp_client.set_response(
            "calculator", {"result": "unique", "success": True}
        )

        # Execute concurrent different requests
        tasks = [
            mcp_router.invoke("calculator", {"a": i, "b": i}, f"different-{i}")
            for i in range(3)
        ]

        await asyncio.gather(*tasks)

        # Verify multiple executions occurred
        assert mock_mcp_client.call_count == 3


class TestMCPRouterCircuitBreaker:
    """Test circuit breaker integration"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_failure(
        self, mock_event_bus, mock_tlm, mock_mcp_client, mock_cache, mock_clock
    ):
        """Test circuit breaker open state prevents execution"""
        # Import the real MCPRouter
        from src.mcp.clients import MCPClientPool
        from src.mcp.router import MCPRouter

        # Create breaker factory that returns open breaker
        def open_breaker_factory(tool_key: str):
            return MockCircuitBreaker(should_open=True)

        # Create client pool with our mock client
        client_pool = MCPClientPool(clock=mock_clock)
        client_pool._clients = {"http://calc.local:8080": mock_mcp_client}

        # Override get_client to always return our mock
        async def mock_get_client(base_url: str):
            return mock_mcp_client

        client_pool.get_client = mock_get_client

        # Create router with open breaker
        router = MCPRouter(
            tlm=mock_tlm,
            bus=mock_event_bus,
            client=client_pool,
            breaker_factory=open_breaker_factory,
            cache=mock_cache,
            clock=mock_clock,
        )

        # Execute should fail due to open circuit
        with pytest.raises(Exception):  # The specific error depends on implementation
            await router.invoke("calculator", {"a": 1, "b": 2}, "breaker-test")


class TestMCPRouterMonotonicTiming:
    """Test monotonic clock usage for deterministic timing"""

    @pytest.mark.asyncio
    async def test_duration_calculation_uses_monotonic_clock(
        self, mcp_router, mock_event_bus, mock_clock, mock_mcp_client
    ):
        """Test that duration calculation uses injected monotonic clock"""
        mock_mcp_client.set_response("calculator", {"result": 123, "success": True})

        # Set initial time
        mock_clock._time = 1000.0

        # Mock call_tool to advance clock
        original_call_tool = mock_mcp_client.call_tool

        async def timed_call_tool(base_url, tool, args):
            mock_clock.advance(0.25)  # Advance 250ms
            return await original_call_tool(base_url, tool, args)

        mock_mcp_client.call_tool = timed_call_tool

        # Execute
        await mcp_router.invoke("calculator", {"a": 1, "b": 2}, "timing-test")

        # Verify duration in completed event
        completed_events = mock_event_bus.get_events_by_type(
            "tool_invocation_completed"
        )
        assert len(completed_events) == 1

        completed = completed_events[0]
        assert completed["duration_ms"] == 250.0  # 0.25 seconds = 250ms


class TestMCPRouterEventObservability:
    """Test comprehensive event emission for observability"""

    @pytest.mark.asyncio
    async def test_all_events_have_correlation_ids(
        self, mcp_router, mock_event_bus, mock_mcp_client
    ):
        """Test that all events include correlation IDs for tracing"""
        mock_mcp_client.set_response("weather", {"temperature": 72, "success": True})

        correlation_id = "trace-abc-123-def"
        await mcp_router.invoke("weather", {"city": "San Francisco"}, correlation_id)

        # Check all events have correlation ID
        all_events = mock_event_bus.events
        tool_events = [e for e in all_events if "tool_invocation" in e["type"]]

        for event in tool_events:
            assert event["correlation_id"] == correlation_id

    @pytest.mark.asyncio
    async def test_event_metadata_completeness(
        self, mcp_router, mock_event_bus, mock_mcp_client
    ):
        """Test that events contain all required metadata"""
        mock_mcp_client.set_response("calculator", {"answer": 42, "success": True})

        await mcp_router.invoke(
            "calculator", {"question": "meaning of life"}, "metadata-test"
        )

        # Verify started event metadata
        started_events = mock_event_bus.get_events_by_type("tool_invocation_started")
        started = started_events[0]

        required_fields = ["tool", "instance_id", "correlation_id", "args", "timestamp"]
        for field in required_fields:
            assert field in started, f"Missing required field: {field}"

        # Verify completed event metadata
        completed_events = mock_event_bus.get_events_by_type(
            "tool_invocation_completed"
        )
        completed = completed_events[0]

        required_fields = [
            "tool",
            "instance_id",
            "correlation_id",
            "result",
            "duration_ms",
            "timestamp",
        ]
        for field in required_fields:
            assert field in completed, f"Missing required field: {field}"


# Property-based test for cache key consistency
@pytest.mark.asyncio
async def test_cache_key_properties(mcp_router):
    """Property-based test for cache key behavior"""

    # Property: Same inputs always produce same cache key
    args1 = {"a": 1, "b": 2, "operation": "add"}
    key1a = mcp_router._generate_cache_key("calc", args1)
    key1b = mcp_router._generate_cache_key("calc", args1)
    assert key1a == key1b

    # Property: Different tools with same args produce different keys
    key_calc = mcp_router._generate_cache_key("calculator", args1)
    key_weather = mcp_router._generate_cache_key("weather", args1)
    assert key_calc != key_weather

    # Property: Different args produce different keys
    args2 = {"a": 1, "b": 3, "operation": "add"}
    key2 = mcp_router._generate_cache_key("calc", args2)
    assert key1a != key2

    # Property: Argument order doesn't matter
    args_ordered1 = {"a": 1, "b": 2, "c": 3}
    args_ordered2 = {"c": 3, "a": 1, "b": 2}
    args_ordered3 = {"b": 2, "c": 3, "a": 1}

    key_o1 = mcp_router._generate_cache_key("test", args_ordered1)
    key_o2 = mcp_router._generate_cache_key("test", args_ordered2)
    key_o3 = mcp_router._generate_cache_key("test", args_ordered3)

    assert key_o1 == key_o2 == key_o3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
