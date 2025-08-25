"""
Comprehensive concurrency and fallback tests for Super Alita FSM.

Tests:
1. Fallback behavior when no tools are available
2. Re-entrant input queuing and processing
3. Stale completion metric increments
4. Mailbox pressure scenarios
5. Circuit breaker functionality
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any

# Import the modules under test
import sys
sys.path.append('src')

from core.states import StateMachine, State, TransitionTrigger, Context, ToolSpec
from core.session import Session, get_session
from core.metrics_registry import get_metrics_registry


class TestFallbackBehavior:
    """Test fallback behavior when no tools are available."""
    
    @pytest.mark.asyncio
    async def test_no_tools_triggers_fallback(self):
        """Test that absence of tools triggers fallback generation."""
        
        # Setup
        session = get_session("test_session")
        metrics = get_metrics_registry()
        
        fsm = StateMachine(session, metrics)
        context = Context(
            user_input="Analyze this complex data",
            detected_intent="analysis",
            tools_selected=[],  # No tools available
        )
        
        # Mock execution flow to return no tools
        with patch('core.execution_flow.find_applicable_tools', return_value=[]):
            # Transition to GENERATE state
            await fsm.transition(TransitionTrigger.TOOLS_SELECTED)
            
            # Should trigger fallback
            result = await fsm._handle_generate_state(context)
            
            # Assertions
            assert result == TransitionTrigger.RESPONSE_READY
            assert context.response is not None
            assert "fallback" in context.response.lower() or "unable" in context.response.lower()
            
            # Check metrics
            fallback_count = metrics.get_counter("sa_fsm_fallback_responses_total")
            assert fallback_count > 0
    
    @pytest.mark.asyncio
    async def test_tool_failure_triggers_fallback(self):
        """Test that tool execution failure triggers fallback."""
        
        session = get_session("test_session_2")
        metrics = get_metrics_registry()
        
        fsm = StateMachine(session, metrics)
        context = Context(
            user_input="Process this request",
            detected_intent="processing",
            tools_selected=[ToolSpec(name="failing_tool", args={})],
        )
        
        # Mock tool execution to fail
        with patch('core.execution_flow._execute_tools_with_comp_env') as mock_execute:
            mock_execute.side_effect = Exception("Tool execution failed")
            
            result = await fsm._handle_generate_state(context)
            
            # Should fall back to response generation
            assert result == TransitionTrigger.RESPONSE_READY
            assert context.response is not None
            
            # Check error recovery metrics
            error_count = metrics.get_counter("sa_fsm_errors_total")
            assert error_count > 0


class TestReEntrantInput:
    """Test re-entrant input handling and queuing."""
    
    @pytest.mark.asyncio
    async def test_concurrent_input_queuing(self):
        """Test that concurrent inputs are properly queued."""
        
        session = get_session("test_concurrent")
        metrics = get_metrics_registry()
        
        fsm = StateMachine(session, metrics)
        
        # Simulate multiple rapid inputs
        inputs = [
            "First request",
            "Second request", 
            "Third request"
        ]
        
        # Submit inputs concurrently
        tasks = []
        for i, user_input in enumerate(inputs):
            task = asyncio.create_task(
                fsm.handle_user_input(user_input, f"session_{i}")
            )
            tasks.append(task)
            
            # Small delay to ensure ordering
            await asyncio.sleep(0.01)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all inputs were processed
        assert len(results) == 3
        for result in results:
            assert not isinstance(result, Exception)
        
        # Check mailbox metrics
        max_mailbox_size = metrics.get_gauge("sa_fsm_mailbox_size_max")
        assert max_mailbox_size >= 2  # At least some queuing occurred
    
    @pytest.mark.asyncio
    async def test_mailbox_pressure_under_load(self):
        """Test mailbox pressure metrics under high load."""
        
        session = get_session("test_pressure")
        metrics = get_metrics_registry()
        
        fsm = StateMachine(session, metrics)
        
        # Generate high load
        rapid_inputs = [f"Request {i}" for i in range(20)]
        
        # Submit very rapidly
        tasks = []
        for user_input in rapid_inputs:
            task = asyncio.create_task(
                fsm.handle_user_input(user_input, "pressure_test")
            )
            tasks.append(task)
        
        # Wait a bit then check pressure
        await asyncio.sleep(0.1)
        
        mailbox_pressure = metrics.get_gauge("sa_fsm_mailbox_pressure")
        assert mailbox_pressure > 0.0  # Some pressure should be recorded
        
        # Wait for completion
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Final pressure should be lower
        final_pressure = metrics.get_gauge("sa_fsm_mailbox_pressure")
        assert final_pressure <= mailbox_pressure


class TestStaleCompletions:
    """Test stale completion detection and metrics."""
    
    @pytest.mark.asyncio
    async def test_stale_completion_detection(self):
        """Test that stale completions are detected and counted."""
        
        session = get_session("test_stale")
        metrics = get_metrics_registry()
        
        fsm = StateMachine(session, metrics)
        
        # Create a context with an old operation ID
        context = Context(
            user_input="Test request",
            session_id="test_stale",
            turn_id="old_turn_id"
        )
        
        # Start a new operation to make the old one stale
        await fsm.handle_user_input("New request", "test_stale")
        
        # Now try to complete the old operation
        await fsm.handle_response_ready(context)
        
        # Check stale completion metrics
        stale_count = metrics.get_counter("sa_fsm_stale_completions_total")
        assert stale_count > 0
    
    @pytest.mark.asyncio
    async def test_stale_rate_calculation(self):
        """Test that stale rate is calculated correctly."""
        
        session = get_session("test_stale_rate")
        metrics = get_metrics_registry()
        
        fsm = StateMachine(session, metrics)
        
        # Perform several operations where some become stale
        for i in range(5):
            context = Context(
                user_input=f"Request {i}",
                session_id="test_stale_rate",
                turn_id=f"turn_{i}"
            )
            
            if i < 3:
                # These will complete normally
                await fsm.handle_user_input(f"Request {i}", "test_stale_rate")
                await fsm.handle_response_ready(context)
            else:
                # These will become stale
                await fsm.handle_user_input(f"Request {i}", "test_stale_rate")
                # Don't complete, let them become stale
        
        # Check metrics
        total_ops = metrics.get_counter("sa_fsm_operations_total")
        stale_ops = metrics.get_counter("sa_fsm_stale_completions_total")
        
        assert total_ops >= 5
        assert stale_ops >= 2


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_mailbox_overflow_circuit_breaker(self):
        """Test circuit breaker trips on mailbox overflow."""
        
        session = get_session("test_circuit_breaker")
        metrics = get_metrics_registry()
        
        fsm = StateMachine(session, metrics)
        
        # Try to overwhelm the mailbox
        overflow_inputs = [f"Overflow {i}" for i in range(150)]  # Exceed MAILBOX_MAX_SIZE
        
        tasks = []
        for user_input in overflow_inputs:
            task = asyncio.create_task(
                fsm.handle_user_input(user_input, "test_circuit_breaker")
            )
            tasks.append(task)
        
        # Wait for some processing
        await asyncio.sleep(0.5)
        
        # Check if circuit breaker tripped
        breaker_trips = metrics.get_counter("sa_fsm_circuit_breaker_trips_total")
        assert breaker_trips > 0
        
        # Check if circuit breaker is open
        breaker_open = metrics.get_gauge("sa_fsm_circuit_breaker_open")
        assert breaker_open == 1.0
        
        # Clean up
        await asyncio.gather(*tasks, return_exceptions=True)
    
    @pytest.mark.asyncio
    async def test_transition_rate_limiting(self):
        """Test transition rate limiting circuit breaker."""
        
        session = get_session("test_rate_limit")
        metrics = get_metrics_registry()
        
        fsm = StateMachine(session, metrics)
        
        # Rapid fire transitions to exceed rate limit
        for i in range(15):  # Exceed TRANSITION_RATE_LIMIT
            await fsm.transition(TransitionTrigger.USER_INPUT_RECEIVED)
            # No delay - trying to exceed rate limit
        
        # Check for blocked transitions
        blocked_count = metrics.get_counter("sa_fsm_blocked_transitions_total")
        assert blocked_count > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_reset(self):
        """Test circuit breaker automatic reset after timeout."""
        
        session = get_session("test_reset")
        metrics = get_metrics_registry()
        
        fsm = StateMachine(session, metrics)
        
        # Trip the circuit breaker
        fsm._trip_circuit_breaker("test_reason")
        
        # Verify it's open
        assert fsm.circuit_breaker.is_open
        
        # Wait for timeout (using short timeout for test)
        original_timeout = fsm.CIRCUIT_BREAKER_TIMEOUT if hasattr(fsm, 'CIRCUIT_BREAKER_TIMEOUT') else 30
        fsm.CIRCUIT_BREAKER_TIMEOUT = 0.1  # 100ms for test
        
        await asyncio.sleep(0.2)
        
        # Try a transition which should reset the breaker
        result = fsm._check_circuit_breaker()
        
        # Should be reset now
        assert not fsm.circuit_breaker.is_open
        assert result is True
        
        # Restore original timeout
        fsm.CIRCUIT_BREAKER_TIMEOUT = original_timeout


class TestIntegrationScenarios:
    """Integration tests combining multiple concurrency scenarios."""
    
    @pytest.mark.asyncio
    async def test_high_load_integration(self):
        """Test system behavior under high concurrent load."""
        
        session = get_session("test_integration")
        metrics = get_metrics_registry()
        
        fsm = StateMachine(session, metrics)
        
        # Mixed workload: normal requests, failures, rapid inputs
        tasks = []
        
        # Normal requests
        for i in range(10):
            task = asyncio.create_task(
                fsm.handle_user_input(f"Normal request {i}", f"session_{i}")
            )
            tasks.append(task)
        
        # Rapid fire requests (potential mailbox pressure)
        for i in range(20):
            task = asyncio.create_task(
                fsm.handle_user_input(f"Rapid {i}", "rapid_session")
            )
            tasks.append(task)
        
        # Wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # System should handle the load gracefully
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        assert success_count >= len(tasks) * 0.8  # At least 80% success rate
        
        # Check final metrics
        final_metrics = {
            "operations_total": metrics.get_counter("sa_fsm_operations_total"),
            "mailbox_pressure": metrics.get_gauge("sa_fsm_mailbox_pressure"),
            "stale_completions": metrics.get_counter("sa_fsm_stale_completions_total"),
            "circuit_breaker_trips": metrics.get_counter("sa_fsm_circuit_breaker_trips_total"),
        }
        
        print(f"Final metrics: {final_metrics}")
        
        # Mailbox should be mostly drained
        assert final_metrics["mailbox_pressure"] < 0.2


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics before each test."""
    metrics = get_metrics_registry()
    metrics.reset()
    yield
    metrics.reset()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
