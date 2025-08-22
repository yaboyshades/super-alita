#!/usr/bin/env python3
"""
Test Suite for Race-Safe Circuit Breaker
Tests async safety, state transitions, monotonic clock, and probe protection.
"""

import asyncio
import random
from unittest.mock import AsyncMock

import pytest

from src.core.clock import FakeClock, MonotonicClock
from src.core.race_safe_breaker import (
    BreakerConfig,
    BreakerState,
    CircuitBreakerOpenError,
    ProbeGate,
    RaceSafeCircuitBreaker,
    decorrelated_jitter_backoff,
)


class DummyMetrics:
    """Mock metrics collector for testing"""

    def __init__(self):
        self.counters = {}
        self.observations = {}

    def inc(self, name: str, labels: dict[str, str]):
        key = (name, tuple(sorted(labels.items())))
        self.counters[key] = self.counters.get(key, 0) + 1

    def observe(self, name: str, value: float, labels: dict[str, str]):
        key = (name, tuple(sorted(labels.items())))
        self.observations.setdefault(key, []).append(value)


class TestDecorrelatedJitter:
    """Test decorrelated jitter backoff algorithm"""

    def test_jitter_respects_bounds(self):
        """Test that jitter stays within [base, cap] bounds"""
        random.seed(1234)  # Deterministic tests
        base, cap = 1.0, 60.0
        previous = 2.0

        for _ in range(100):
            next_value = decorrelated_jitter_backoff(previous, base, cap)
            assert base <= next_value <= cap
            previous = next_value

    def test_jitter_increases_over_time(self):
        """Test that jitter generally increases backoff over time"""
        random.seed(1234)
        base, cap = 1.0, 60.0
        previous = base
        increases = 0

        for _ in range(50):
            next_value = decorrelated_jitter_backoff(previous, base, cap)
            if next_value > previous:
                increases += 1
            previous = next_value

        # Should increase more often than not (jitter adds randomness)
        assert increases > 20  # At least 40% increases


class TestProbeGate:
    """Test single-flight probe gate protection"""

    @pytest.mark.asyncio
    async def test_single_probe_execution(self):
        """Test that only one probe executes at a time"""
        gate = ProbeGate()
        execution_count = 0

        async def probe_func():
            nonlocal execution_count
            execution_count += 1
            await asyncio.sleep(0.1)  # Simulate work
            return execution_count

        # Launch multiple concurrent probes
        tasks = [asyncio.create_task(gate.try_probe(probe_func)) for _ in range(5)]

        # First should succeed, others should raise RuntimeError
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = [r for r in results if not isinstance(r, Exception)]
        errors = [r for r in results if isinstance(r, RuntimeError)]

        assert len(successes) == 1
        assert len(errors) == 4
        assert execution_count == 1

    @pytest.mark.asyncio
    async def test_sequential_probes_work(self):
        """Test that sequential probes work correctly"""
        gate = ProbeGate()
        results = []

        async def probe_func():
            results.append(len(results) + 1)
            return len(results)

        # Execute probes sequentially
        for i in range(3):
            result = await gate.try_probe(probe_func)
            assert result == i + 1

        assert results == [1, 2, 3]


class TestRaceSafeCircuitBreaker:
    """Test race-safe circuit breaker functionality"""

    @pytest.fixture
    def config(self):
        """Standard breaker configuration for tests"""
        return BreakerConfig(
            failure_threshold=3,
            window_seconds=10.0,
            open_min_seconds=2.0,
            open_cap_seconds=8.0,
            base_seconds=1.0,
            half_open_max_probes=1,
        )

    @pytest.fixture
    def clock(self):
        """Fake clock for deterministic testing"""
        return FakeClock(100.0)

    @pytest.fixture
    def event_bus(self):
        """Mock event bus"""
        return AsyncMock()

    @pytest.fixture
    def breaker(self, config, clock, event_bus):
        """Circuit breaker instance with fake clock"""
        return RaceSafeCircuitBreaker("test_breaker", config, clock, event_bus)

    @pytest.mark.asyncio
    async def test_initial_state_closed(self, breaker):
        """Test that breaker starts in CLOSED state"""
        assert breaker.state == BreakerState.CLOSED
        assert await breaker.allow_request() is True

    @pytest.mark.asyncio
    async def test_failure_threshold_triggers_open(self, breaker, clock):
        """Test that failure threshold triggers OPEN state"""
        # Record failures up to threshold
        for i in range(breaker.config.failure_threshold):
            await breaker.record_result(False)
            if i < breaker.config.failure_threshold - 1:
                assert await breaker.allow_request() is True
            else:
                # Last failure should trigger OPEN state
                assert await breaker.allow_request() is False
                assert breaker.state == BreakerState.OPEN

    @pytest.mark.asyncio
    async def test_cooldown_transitions_to_half_open(self, breaker, clock):
        """Test that cooldown period transitions to HALF_OPEN"""
        # Force OPEN state
        await breaker.force_state(BreakerState.OPEN)
        assert await breaker.allow_request() is False

        # Advance time past cooldown
        clock.tick(breaker.config.open_min_seconds + 0.1)

        # Next request should transition to HALF_OPEN but still deny this request
        assert await breaker.allow_request() is False
        assert breaker.state == BreakerState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_success_transitions_to_closed(self, breaker):
        """Test successful probe in HALF_OPEN transitions to CLOSED"""
        await breaker.force_state(BreakerState.HALF_OPEN)

        # Record successful result
        await breaker.record_result(True)
        assert breaker.state == BreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_transitions_to_open(self, breaker):
        """Test failed probe in HALF_OPEN transitions back to OPEN"""
        await breaker.force_state(BreakerState.HALF_OPEN)

        # Record failed result
        await breaker.record_result(False)
        assert breaker.state == BreakerState.OPEN

    @pytest.mark.asyncio
    async def test_execute_with_breaker_success(self, breaker):
        """Test successful execution with circuit breaker"""

        async def success_func():
            return "success"

        result = await breaker.execute_with_breaker(success_func)
        assert result == "success"
        assert breaker.state == BreakerState.CLOSED

        metrics = breaker.get_metrics()
        assert metrics["total_calls"] == 1
        assert metrics["successful_calls"] == 1

    @pytest.mark.asyncio
    async def test_execute_with_breaker_failure(self, breaker):
        """Test failed execution with circuit breaker"""

        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await breaker.execute_with_breaker(failing_func)

        metrics = breaker.get_metrics()
        assert metrics["total_calls"] == 1
        assert metrics["failed_calls"] == 1

    @pytest.mark.asyncio
    async def test_execute_with_breaker_open_rejects(self, breaker):
        """Test that OPEN breaker rejects requests immediately"""
        await breaker.force_state(BreakerState.OPEN)

        async def any_func():
            return "should not execute"

        with pytest.raises(CircuitBreakerOpenError):
            await breaker.execute_with_breaker(any_func)

    @pytest.mark.asyncio
    async def test_sync_function_execution(self, breaker):
        """Test execution of synchronous functions"""

        def sync_func(x, y):
            return x + y

        # Wrap in lambda to match expected signature
        result = await breaker.execute_with_breaker(lambda: sync_func(2, 3))
        assert result == 5

    @pytest.mark.asyncio
    async def test_execute_probe_single_flight(self, breaker):
        """Test that probe execution is single-flight protected"""
        await breaker.force_state(BreakerState.HALF_OPEN)
        execution_count = 0

        async def probe_func():
            nonlocal execution_count
            execution_count += 1
            await asyncio.sleep(0.1)
            return execution_count

        # Launch multiple concurrent probes
        tasks = [
            asyncio.create_task(breaker.execute_probe(probe_func)) for _ in range(3)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Only one should succeed, others should be shed
        successes = [r for r in results if not isinstance(r, Exception)]
        shed_errors = [
            r
            for r in results
            if isinstance(r, CircuitBreakerOpenError)
            and "probe already in progress" in str(r)
        ]

        assert len(successes) == 1
        assert len(shed_errors) == 2
        assert execution_count == 1

    @pytest.mark.asyncio
    async def test_execute_probe_wrong_state_raises(self, breaker):
        """Test that probe execution in wrong state raises ValueError"""
        assert breaker.state == BreakerState.CLOSED

        async def probe_func():
            return "test"

        with pytest.raises(
            ValueError, match="Probe can only be executed in HALF_OPEN state"
        ):
            await breaker.execute_probe(probe_func)

    @pytest.mark.asyncio
    async def test_failure_window_decay(self, breaker, clock):
        """Test that failures outside window are decayed"""
        # Record failures
        for _ in range(2):
            await breaker.record_result(False)

        # Advance time past window
        clock.tick(breaker.config.window_seconds + 1)

        # Should still allow requests (failures decayed)
        assert await breaker.allow_request() is True

        # Add one more failure - shouldn't trip (only 1 in window)
        await breaker.record_result(False)
        assert await breaker.allow_request() is True

    @pytest.mark.asyncio
    async def test_monitored_exceptions_only(self, config, clock, event_bus):
        """Test that only monitored exceptions count as failures"""
        config.monitored_exceptions = [ValueError]
        breaker = RaceSafeCircuitBreaker("test", config, clock, event_bus)

        # RuntimeError should not count
        await breaker.record_result(False, RuntimeError("Not monitored"))
        assert breaker.get_metrics()["failed_calls"] == 0

        # ValueError should count
        await breaker.record_result(False, ValueError("Monitored"))
        assert breaker.get_metrics()["failed_calls"] == 1

    @pytest.mark.asyncio
    async def test_metrics_accuracy(self, breaker):
        """Test that metrics are accurately tracked"""
        # Mix of successes and failures
        await breaker.record_result(True)
        await breaker.record_result(False)
        await breaker.record_result(True)

        metrics = breaker.get_metrics()
        assert metrics["total_calls"] == 3
        assert metrics["successful_calls"] == 2
        assert metrics["failed_calls"] == 1
        assert metrics["success_rate"] == pytest.approx(66.67, rel=0.01)

    @pytest.mark.asyncio
    async def test_state_transition_events(self, breaker, event_bus):
        """Test that state transitions emit events"""
        await breaker.force_state(BreakerState.OPEN)

        # Verify event was emitted
        assert event_bus.emit.called
        event = event_bus.emit.call_args[0][0]
        assert event.event_type == "circuit_breaker_state_changed"
        assert event.breaker_name == "test_breaker"
        assert event.old_state == "closed"
        assert event.new_state == "open"

    @pytest.mark.asyncio
    async def test_reset_functionality(self, breaker):
        """Test circuit breaker reset functionality"""
        # Generate some state
        await breaker.record_result(False)
        await breaker.force_state(BreakerState.OPEN)

        # Reset
        await breaker.reset()

        # Should be back to initial state
        assert breaker.state == BreakerState.CLOSED
        metrics = breaker.get_metrics()
        assert metrics["total_calls"] == 0
        assert metrics["successful_calls"] == 0
        assert metrics["failed_calls"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_state_transitions(self, breaker):
        """Test concurrent access doesn't cause race conditions"""

        async def record_failures():
            for _ in range(5):
                await breaker.record_result(False)
                await asyncio.sleep(0.001)  # Small delay

        async def check_requests():
            for _ in range(10):
                await breaker.allow_request()
                await asyncio.sleep(0.001)

        # Run concurrent operations
        await asyncio.gather(record_failures(), check_requests())

        # Should have consistent state (no race conditions)
        metrics = breaker.get_metrics()
        assert metrics["failed_calls"] == 5

    def test_monotonic_clock_integration(self, config, event_bus):
        """Test integration with real monotonic clock"""
        real_clock = MonotonicClock()
        breaker = RaceSafeCircuitBreaker("test", config, real_clock, event_bus)

        # Should work with real clock
        t1 = real_clock.now()
        t2 = real_clock.now()
        assert t2 >= t1

    @pytest.mark.asyncio
    async def test_backoff_evolution(self, breaker, clock):
        """Test that backoff evolves with decorrelated jitter"""
        random.seed(1234)  # Deterministic jitter

        initial_backoff = breaker._backoff_previous

        # Force multiple OPEN transitions
        for _ in range(3):
            await breaker.force_state(BreakerState.CLOSED)
            await breaker.force_state(BreakerState.OPEN)

        # Backoff should have evolved
        assert breaker._backoff_previous != initial_backoff
        assert breaker._backoff_previous >= breaker.config.base_seconds
        assert breaker._backoff_previous <= breaker.config.open_cap_seconds


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios"""

    @pytest.fixture
    def config(self):
        return BreakerConfig(
            failure_threshold=3,
            window_seconds=5.0,
            open_min_seconds=1.0,
            open_cap_seconds=10.0,
        )

    @pytest.fixture
    def clock(self):
        return FakeClock(0.0)

    @pytest.fixture
    def breaker(self, config, clock):
        return RaceSafeCircuitBreaker("integration_test", config, clock)

    @pytest.mark.asyncio
    async def test_full_circuit_lifecycle(self, breaker, clock):
        """Test complete circuit breaker lifecycle"""
        # Start in CLOSED state
        assert breaker.state == BreakerState.CLOSED

        # Generate failures to trip breaker
        for _ in range(3):
            await breaker.record_result(False)

        # Should be OPEN now
        assert await breaker.allow_request() is False
        assert breaker.state == BreakerState.OPEN

        # Wait for cooldown
        clock.tick(breaker.config.open_min_seconds + 0.1)

        # Should transition to HALF_OPEN
        assert await breaker.allow_request() is False
        assert breaker.state == BreakerState.HALF_OPEN

        # Successful probe should close circuit
        await breaker.record_result(True)
        assert breaker.state == BreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_burst_failure_handling(self, breaker):
        """Test handling of burst failures"""

        async def sometimes_failing_func():
            # Fail 70% of the time
            if random.random() < 0.7:
                raise ValueError("Simulated failure")
            return "success"

        random.seed(1234)
        successes = 0
        circuit_breaker_rejections = 0

        # Simulate burst of requests
        for _ in range(20):
            try:
                result = await breaker.execute_with_breaker(sometimes_failing_func)
                if result == "success":
                    successes += 1
            except CircuitBreakerOpenError:
                circuit_breaker_rejections += 1
            except ValueError:
                # Expected failures
                pass

        # Should have some successes and some circuit breaker rejections
        assert successes > 0
        metrics = breaker.get_metrics()
        print(f"Successes: {successes}, CB rejections: {circuit_breaker_rejections}")
        print(f"Metrics: {metrics}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
