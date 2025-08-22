#!/usr/bin/env python3
"""
Test Suite for Error Recovery & Resilience System
Validates circuit breakers, retry policies, error handling, and recovery mechanisms.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.core.error_recovery import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    ErrorContext,
    ErrorRecoveryOrchestrator,
    ErrorSeverity,
    RecoveryAction,
)


class TestErrorContext:
    """Test ErrorContext data class"""

    def test_error_context_creation(self):
        """Test ErrorContext creation with defaults"""
        context = ErrorContext(
            component_name="test_component",
            operation_name="test_operation",
            error_type="ValueError",
            error_message="Test error",
        )

        assert context.component_name == "test_component"
        assert context.operation_name == "test_operation"
        assert context.error_type == "ValueError"
        assert context.error_message == "Test error"
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.retry_count == 0
        assert context.max_retries == 3
        assert isinstance(context.timestamp, datetime)

    def test_error_context_to_dict(self):
        """Test ErrorContext serialization"""
        context = ErrorContext(
            component_name="test_component",
            error_type="ValueError",
            error_message="Test error",
        )

        data = context.to_dict()

        assert data["component_name"] == "test_component"
        assert data["error_type"] == "ValueError"
        assert data["error_message"] == "Test error"
        assert data["severity"] == "medium"
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)


class TestCircuitBreaker:
    """Test Circuit Breaker functionality"""

    @pytest.fixture
    def config(self):
        """Circuit breaker configuration"""
        return CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=5.0,
            half_open_max_calls=2,
        )

    @pytest.fixture
    def event_bus(self):
        """Mock event bus"""
        return AsyncMock()

    @pytest.fixture
    def circuit_breaker(self, config, event_bus):
        """Circuit breaker instance"""
        return CircuitBreaker("test_circuit", config, event_bus)

    @pytest.mark.asyncio
    async def test_successful_calls(self, circuit_breaker):
        """Test successful function calls"""

        async def success_func():
            return "success"

        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.successful_calls == 1
        assert circuit_breaker.failed_calls == 0

    @pytest.mark.asyncio
    async def test_sync_function_call(self, circuit_breaker):
        """Test circuit breaker with synchronous function"""

        def sync_func(x, y):
            return x + y

        result = await circuit_breaker.call(sync_func, 2, 3)
        assert result == 5
        assert circuit_breaker.successful_calls == 1

    @pytest.mark.asyncio
    async def test_failure_threshold(self, circuit_breaker):
        """Test circuit breaker opens after failure threshold"""

        async def failing_func():
            raise ValueError("Test error")

        # Fail 3 times (threshold)
        for _ in range(3):
            with pytest.raises(ValueError, match="Test error"):
                await circuit_breaker.call(failing_func)

        # Circuit should now be open
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.failed_calls == 3

        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_half_open_state(self, circuit_breaker):
        """Test circuit breaker half-open state functionality"""
        # Force circuit to open state
        circuit_breaker.state = CircuitBreakerState.OPEN
        circuit_breaker.last_failure_time = datetime.now(UTC)
        circuit_breaker.failure_count = 5

        # Simulate timeout passage
        circuit_breaker.config.timeout_seconds = 0.1
        await asyncio.sleep(0.2)

        async def success_func():
            return "success"

        # Should transition to half-open
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_to_closed_transition(self, circuit_breaker):
        """Test transition from half-open to closed state"""
        # Set to half-open state
        circuit_breaker.state = CircuitBreakerState.HALF_OPEN
        circuit_breaker.success_count = 0

        async def success_func():
            return "success"

        # Call success_threshold times
        for _ in range(circuit_breaker.config.success_threshold):
            await circuit_breaker.call(success_func)

        # Should transition to closed
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_to_open_transition(self, circuit_breaker):
        """Test transition from half-open back to open on failure"""
        # Set to half-open state
        circuit_breaker.state = CircuitBreakerState.HALF_OPEN

        async def failing_func():
            raise ValueError("Test error")

        # One failure should transition back to open
        with pytest.raises(ValueError, match="Test error"):
            await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_half_open_max_calls(self, circuit_breaker):
        """Test half-open max calls limit"""
        # Set to half-open state
        circuit_breaker.state = CircuitBreakerState.HALF_OPEN
        circuit_breaker.half_open_calls = circuit_breaker.config.half_open_max_calls

        async def success_func():
            return "success"

        # Should raise CircuitBreakerOpenError due to max calls limit
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(success_func)

    @pytest.mark.asyncio
    async def test_event_emission(self, circuit_breaker, event_bus):
        """Test that circuit breaker emits events"""

        async def failing_func():
            raise ValueError("Test error")

        # Cause a failure
        with pytest.raises(ValueError, match="Test error"):
            await circuit_breaker.call(failing_func)

        # Verify failure event was emitted
        assert event_bus.emit.called
        event_args = event_bus.emit.call_args[0][0]
        assert event_args.event_type == "circuit_breaker_failure"

    def test_get_metrics(self, circuit_breaker):
        """Test circuit breaker metrics"""
        circuit_breaker.total_calls = 10
        circuit_breaker.successful_calls = 8
        circuit_breaker.failed_calls = 2

        metrics = circuit_breaker.get_metrics()

        assert metrics["name"] == "test_circuit"
        assert metrics["total_calls"] == 10
        assert metrics["successful_calls"] == 8
        assert metrics["failed_calls"] == 2
        assert metrics["success_rate"] == 80.0

    @pytest.mark.asyncio
    async def test_monitored_exceptions_only(self, config, event_bus):
        """Test that only monitored exceptions trigger circuit breaker"""
        config.monitored_exceptions = [ValueError]
        circuit_breaker = CircuitBreaker("test_circuit", config, event_bus)

        async def runtime_error_func():
            raise RuntimeError("Not monitored")

        async def value_error_func():
            raise ValueError("Monitored")

        # RuntimeError should not count toward failure
        with pytest.raises(RuntimeError):
            await circuit_breaker.call(runtime_error_func)
        assert circuit_breaker.failure_count == 0

        # ValueError should count toward failure
        with pytest.raises(ValueError, match="Monitored"):
            await circuit_breaker.call(value_error_func)
        assert circuit_breaker.failure_count == 1


class TestErrorRecoveryOrchestrator:
    """Test Error Recovery Orchestrator"""

    @pytest.fixture
    def event_bus(self):
        """Mock event bus"""
        return AsyncMock()

    @pytest.fixture
    def orchestrator(self, event_bus):
        """Error recovery orchestrator instance"""
        return ErrorRecoveryOrchestrator(event_bus)

    @pytest.mark.asyncio
    async def test_start_stop(self, orchestrator):
        """Test orchestrator start and stop"""
        assert not orchestrator._running

        await orchestrator.start()
        assert orchestrator._running
        assert orchestrator._monitor_task is not None

        await orchestrator.stop()
        assert not orchestrator._running

    def test_register_circuit_breaker(self, orchestrator):
        """Test circuit breaker registration"""
        config = CircuitBreakerConfig()
        circuit_breaker = orchestrator.register_circuit_breaker("test_cb", config)

        assert "test_cb" in orchestrator.circuit_breakers
        assert circuit_breaker.name == "test_cb"
        assert circuit_breaker.config == config

    @pytest.mark.asyncio
    async def test_handle_error_basic(self, orchestrator):
        """Test basic error handling"""
        error_context = ErrorContext(
            component_name="test_component",
            error_type="ValueError",
            error_message="Test error",
            severity=ErrorSeverity.LOW,
        )

        await orchestrator.handle_error(error_context)

        # Should record error in history
        assert len(orchestrator.error_history) == 1
        assert orchestrator.error_history[0] == error_context

        # Should track error pattern
        pattern = "test_component:ValueError"
        assert orchestrator.error_patterns[pattern] == 1

    @pytest.mark.asyncio
    async def test_handle_error_with_custom_actions(self, orchestrator):
        """Test error handling with custom recovery actions"""
        error_context = ErrorContext(
            component_name="test_component",
            error_type="ValueError",
            error_message="Test error",
        )

        custom_actions = [RecoveryAction.FALLBACK, RecoveryAction.ALERT_OPERATOR]
        await orchestrator.handle_error(error_context, custom_actions)

        # Should still record error
        assert len(orchestrator.error_history) == 1

    def test_determine_recovery_actions_by_severity(self, orchestrator):
        """Test recovery action determination based on severity"""
        # Low severity
        context_low = ErrorContext(severity=ErrorSeverity.LOW)
        actions_low = orchestrator._determine_recovery_actions(context_low)
        assert actions_low == [RecoveryAction.RETRY]

        # Medium severity
        context_medium = ErrorContext(severity=ErrorSeverity.MEDIUM)
        actions_medium = orchestrator._determine_recovery_actions(context_medium)
        assert RecoveryAction.RETRY in actions_medium
        assert RecoveryAction.FALLBACK in actions_medium

        # High severity
        context_high = ErrorContext(severity=ErrorSeverity.HIGH)
        actions_high = orchestrator._determine_recovery_actions(context_high)
        assert RecoveryAction.CIRCUIT_BREAK in actions_high
        assert RecoveryAction.FALLBACK in actions_high

        # Critical severity
        context_critical = ErrorContext(severity=ErrorSeverity.CRITICAL)
        actions_critical = orchestrator._determine_recovery_actions(context_critical)
        assert RecoveryAction.CIRCUIT_BREAK in actions_critical
        assert RecoveryAction.FALLBACK in actions_critical
        assert RecoveryAction.ALERT_OPERATOR in actions_critical

    def test_determine_recovery_actions_by_pattern(self, orchestrator):
        """Test recovery action determination based on error patterns"""
        # Create frequent error pattern
        error_pattern = "test_component:ValueError"
        orchestrator.error_patterns[error_pattern] = 10  # Frequent errors

        context = ErrorContext(
            component_name="test_component",
            error_type="ValueError",
            severity=ErrorSeverity.LOW,  # Normally just retry
        )

        actions = orchestrator._determine_recovery_actions(context)
        assert RecoveryAction.CIRCUIT_BREAK in actions

    @pytest.mark.asyncio
    async def test_handle_retry_recovery(self, orchestrator):
        """Test retry recovery action"""
        error_context = ErrorContext(retry_count=1, max_retries=3)
        result = await orchestrator._handle_retry(error_context)
        assert result is True  # Can still retry

        error_context.retry_count = 5
        result = await orchestrator._handle_retry(error_context)
        assert result is False  # Exhausted retries

    @pytest.mark.asyncio
    async def test_handle_circuit_break_recovery(self, orchestrator):
        """Test circuit breaker recovery action"""
        error_context = ErrorContext(component_name="test_component")
        result = await orchestrator._handle_circuit_break(error_context)

        assert result is True
        assert "test_component_circuit_breaker" in orchestrator.circuit_breakers

    @pytest.mark.asyncio
    async def test_handle_fallback_recovery(self, orchestrator):
        """Test fallback recovery action"""
        error_context = ErrorContext(component_name="test_component")
        result = await orchestrator._handle_fallback(error_context)
        assert result is True

    @pytest.mark.asyncio
    async def test_handle_alert_recovery(self, orchestrator, event_bus):
        """Test operator alert recovery action"""
        error_context = ErrorContext(
            component_name="test_component",
            error_message="Critical error",
        )

        result = await orchestrator._handle_alert(error_context)
        assert result is True

        # Verify alert event was emitted
        assert event_bus.emit.called
        event_args = event_bus.emit.call_args[0][0]
        assert event_args.event_type == "operator_alert"

    @pytest.mark.asyncio
    async def test_monitoring_loop_error_analysis(self, orchestrator):
        """Test monitoring loop error pattern analysis"""
        # Add recent errors
        for i in range(15):
            error = ErrorContext(
                component_name=f"component_{i % 3}",
                error_type="ValueError",
                timestamp=datetime.now(UTC),
            )
            orchestrator.error_history.append(error)

        # Run one iteration of analysis
        await orchestrator._analyze_error_patterns()

        # Should detect high error rate (15 errors in last 5 minutes)
        # This would typically log a warning

    @pytest.mark.asyncio
    async def test_cleanup_error_history(self, orchestrator):
        """Test error history cleanup"""
        # Add old error pattern
        old_pattern = "old_component:OldError"
        orchestrator.error_patterns[old_pattern] = 5

        # Add recent error pattern
        recent_error = ErrorContext(
            component_name="recent_component",
            error_type="RecentError",
            timestamp=datetime.now(UTC),
        )
        orchestrator.error_history.append(recent_error)
        recent_pattern = "recent_component:RecentError"
        orchestrator.error_patterns[recent_pattern] = 3

        await orchestrator._cleanup_error_history()

        # Old pattern should be removed (no recent errors)
        assert old_pattern not in orchestrator.error_patterns
        # Recent pattern should remain
        assert recent_pattern in orchestrator.error_patterns

    def test_get_system_status(self, orchestrator):
        """Test system status retrieval"""
        # Add some test data
        config = CircuitBreakerConfig()
        orchestrator.register_circuit_breaker("test_cb", config)

        error_context = ErrorContext(
            component_name="test_component",
            error_type="ValueError",
            severity=ErrorSeverity.HIGH,
        )
        orchestrator.error_history.append(error_context)
        orchestrator.error_patterns["test_component:ValueError"] = 2

        status = orchestrator.get_system_status()

        assert "running" in status
        assert "circuit_breakers" in status
        assert "error_statistics" in status
        assert "test_cb" in status["circuit_breakers"]
        assert status["error_statistics"]["total_errors"] == 1
        assert (
            status["error_statistics"]["error_patterns"]["test_component:ValueError"]
            == 2
        )


class TestIntegrationScenarios:
    """Test integration scenarios for error recovery system"""

    @pytest.fixture
    def event_bus(self):
        """Mock event bus"""
        return AsyncMock()

    @pytest.fixture
    def orchestrator(self, event_bus):
        """Configured orchestrator with circuit breakers"""
        orch = ErrorRecoveryOrchestrator(event_bus)

        # Register some circuit breakers
        config = CircuitBreakerConfig(
            failure_threshold=5
        )  # Higher threshold for testing
        orch.register_circuit_breaker("critical_service", config)
        orch.register_circuit_breaker("background_service", config)

        return orch

    @pytest.mark.asyncio
    async def test_cascading_failure_scenario(self, orchestrator):
        """Test handling of cascading failures across components"""
        # Simulate cascading failures
        components = ["service_a", "service_b", "service_c"]

        for component in components:
            for _ in range(3):  # Multiple failures per component
                error_context = ErrorContext(
                    component_name=component,
                    error_type="ConnectionError",
                    error_message=f"Connection failed in {component}",
                    severity=ErrorSeverity.HIGH,
                )
                await orchestrator.handle_error(error_context)

        # Check that circuit breakers were created for each component
        for component in components:
            cb_name = f"{component}_circuit_breaker"
            assert cb_name in orchestrator.circuit_breakers

        # Verify error patterns are tracked
        for component in components:
            pattern = f"{component}:ConnectionError"
            assert orchestrator.error_patterns[pattern] == 3

    @pytest.mark.asyncio
    async def test_recovery_workflow(self, orchestrator):
        """Test complete error recovery workflow"""
        # Start orchestrator
        await orchestrator.start()

        try:
            # Simulate error with escalating severity
            base_error = ErrorContext(
                component_name="critical_service",
                operation_name="process_data",
                error_type="TimeoutError",
                error_message="Operation timed out",
            )

            # First occurrence - low severity
            base_error.severity = ErrorSeverity.LOW
            await orchestrator.handle_error(base_error)

            # Second occurrence - medium severity
            base_error.severity = ErrorSeverity.MEDIUM
            base_error.retry_count = 1
            await orchestrator.handle_error(base_error)

            # Third occurrence - high severity
            base_error.severity = ErrorSeverity.HIGH
            base_error.retry_count = 2
            await orchestrator.handle_error(base_error)

            # Verify error tracking
            assert len(orchestrator.error_history) == 3
            pattern = "critical_service:TimeoutError"
            assert orchestrator.error_patterns[pattern] == 3

            # Get system status
            status = orchestrator.get_system_status()
            assert status["running"] is True
            assert status["error_statistics"]["total_errors"] >= 3

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_orchestrator(self, orchestrator):
        """Test circuit breaker integration with orchestrator"""
        circuit_breaker = orchestrator.circuit_breakers["critical_service"]

        # Define a failing function
        async def failing_service():
            raise ConnectionError("Service unavailable")

        # Trigger circuit breaker failures
        for _ in range(5):  # Use 5 failures to match the threshold
            with pytest.raises(ConnectionError, match="Service unavailable"):
                await circuit_breaker.call(failing_service)

        # Circuit should be open now
        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(failing_service)

        # Verify metrics
        metrics = circuit_breaker.get_metrics()
        assert metrics["failed_calls"] == 5
        assert metrics["state"] == "open"

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, orchestrator):
        """Test concurrent error handling scenarios"""
        # Create multiple error contexts for concurrent processing
        error_contexts = [
            ErrorContext(
                component_name=f"service_{i}",
                error_type="ConcurrentError",
                error_message=f"Error in service {i}",
                severity=ErrorSeverity.MEDIUM,
            )
            for i in range(10)
        ]

        # Handle errors concurrently
        tasks = [orchestrator.handle_error(context) for context in error_contexts]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        assert all(not isinstance(result, Exception) for result in results)

        # Verify all errors were recorded
        assert len(orchestrator.error_history) == 10

        # Verify patterns are tracked correctly
        for i in range(10):
            pattern = f"service_{i}:ConcurrentError"
            assert orchestrator.error_patterns[pattern] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
