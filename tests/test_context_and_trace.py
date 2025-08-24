import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.core.context_builder import ContextAssembler
from src.core.trace import TurnTracer, get_tracer, trace_component_call
from src.core.correlation import get_correlation_id, get_session_id, set_correlation_id, set_session_id


class TestContextAssembler:
    """Test suite for ContextAssembler functionality"""

    def test_basic_context_creation(self):
        """Test basic context assembler initialization and base context"""
        assembler = ContextAssembler(
            user_input="test query",
            recent_events=[{"type": "test", "data": "value"}],
            active_goals=["goal1", "goal2"]
        )
        
        base_ctx = assembler._base()
        
        assert base_ctx["context_version"] == "1.0"
        assert base_ctx["user_input"] == "test query"
        assert len(base_ctx["recent_events"]) == 1
        assert len(base_ctx["active_goals"]) == 2
        assert "timestamp" in base_ctx
        assert "session_id" in base_ctx
        assert "correlation_id" in base_ctx

    def test_build_for_decision(self):
        """Test decision context building"""
        assembler = ContextAssembler(user_input="make a decision")
        ctx = assembler.build_for_decision()
        
        assert ctx["for"] == "decision"
        assert ctx["user_input"] == "make a decision"

    def test_build_for_tool_execution(self):
        """Test tool execution context building"""
        assembler = ContextAssembler(user_input="run tool")
        ctx = assembler.build_for_tool_execution("test_tool")
        
        assert ctx["for"] == "tool_execution"
        assert ctx["tool_name"] == "test_tool"

    def test_build_for_memory(self):
        """Test memory context building"""
        assembler = ContextAssembler(user_input="retrieve memories")
        ctx = assembler.build_for_memory()
        
        assert ctx["for"] == "memory"

    def test_normalize_memory_hits(self):
        """Test memory hits normalization"""
        hits = [
            {"atom_id": "test-1", "score": "0.85", "extra": "ignored"},
            {"atom_id": "test-2", "score": 0.75}
        ]
        
        normalized = ContextAssembler._normalize_memory_hits(hits)
        
        assert len(normalized) == 2
        assert normalized[0]["atom_id"] == "test-1"
        assert normalized[0]["score"] == 0.85
        assert "extra" not in normalized[0]
        assert normalized[1]["score"] == 0.75


class TestTurnTracer:
    """Test suite for TurnTracer functionality"""

    def test_tracer_initialization(self):
        """Test tracer initialization"""
        tracer = TurnTracer()
        assert tracer.current_turn == 0
        assert len(tracer.traces) == 0

    def test_new_turn(self):
        """Test turn increment"""
        tracer = TurnTracer()
        tracer.new_turn()
        assert tracer.current_turn == 1
        tracer.new_turn()
        assert tracer.current_turn == 2

    def test_log_entry(self):
        """Test logging trace entries"""
        tracer = TurnTracer()
        tracer.new_turn()
        
        tracer.log("test_event", "test_component", {"key": "value"}, 150.5)
        
        assert len(tracer.traces) == 1
        entry = tracer.traces[0]
        assert entry.turn_index == 1
        assert entry.event_type == "test_event"
        assert entry.component == "test_component"
        assert entry.details["key"] == "value"
        assert entry.duration_ms == 150.5

    def test_get_turn_traces(self):
        """Test retrieving traces for specific turn"""
        tracer = TurnTracer()
        tracer.new_turn()
        tracer.log("event1", "comp1")
        tracer.new_turn()
        tracer.log("event2", "comp2")
        tracer.log("event3", "comp3")
        
        turn1_traces = tracer.get_turn_traces(1)
        turn2_traces = tracer.get_turn_traces(2)
        
        assert len(turn1_traces) == 1
        assert len(turn2_traces) == 2
        assert turn1_traces[0].event_type == "event1"

    def test_get_latest_traces(self):
        """Test retrieving latest traces with limit"""
        tracer = TurnTracer()
        tracer.new_turn()
        
        # Add multiple traces
        for i in range(5):
            tracer.log(f"event{i}", "component")
            
        latest_3 = tracer.get_latest_traces(3)
        all_traces = tracer.get_latest_traces(0)
        
        assert len(latest_3) == 3
        assert len(all_traces) == 5
        assert latest_3[-1].event_type == "event4"  # Most recent


class TestTraceDecorator:
    """Test suite for trace_component_call decorator"""

    @pytest.mark.asyncio
    async def test_async_function_tracing(self):
        """Test tracing async functions"""
        tracer = get_tracer()
        tracer.new_turn()
        initial_trace_count = len(tracer.traces)
        
        @trace_component_call("test_component")
        async def async_test_function(value: int) -> int:
            await asyncio.sleep(0.01)  # Small delay for duration measurement
            return value * 2
            
        result = await async_test_function(5)
        
        assert result == 10
        assert len(tracer.traces) == initial_trace_count + 1
        
        latest_trace = tracer.traces[-1]
        assert latest_trace.component == "test_component"
        assert latest_trace.event_type == "component_call"
        assert latest_trace.details["function"] == "async_test_function"
        assert latest_trace.details["success"] is True
        assert latest_trace.duration_ms is not None
        assert latest_trace.duration_ms > 0

    def test_sync_function_tracing(self):
        """Test tracing sync functions"""
        tracer = get_tracer()
        tracer.new_turn()
        initial_trace_count = len(tracer.traces)
        
        @trace_component_call("sync_component")
        def sync_test_function(value: int) -> int:
            return value * 3
            
        result = sync_test_function(4)
        
        assert result == 12
        assert len(tracer.traces) == initial_trace_count + 1
        
        latest_trace = tracer.traces[-1]
        assert latest_trace.component == "sync_component"
        assert latest_trace.details["function"] == "sync_test_function"
        assert latest_trace.details["success"] is True

    @pytest.mark.asyncio
    async def test_function_exception_tracing(self):
        """Test tracing functions that raise exceptions"""
        tracer = get_tracer()
        tracer.new_turn()
        initial_trace_count = len(tracer.traces)
        
        @trace_component_call("error_component")
        async def failing_function():
            raise ValueError("Test error")
            
        with pytest.raises(ValueError, match="Test error"):
            await failing_function()
            
        assert len(tracer.traces) == initial_trace_count + 1
        
        latest_trace = tracer.traces[-1]
        assert latest_trace.component == "error_component"
        assert latest_trace.details["success"] is False
        assert "Test error" in latest_trace.details["error"]


class TestCorrelationIntegration:
    """Test integration between correlation, context, and trace modules"""

    def test_correlation_in_context(self):
        """Test that context assembler includes correlation IDs"""
        test_correlation_id = "test-correlation-123"
        test_session_id = "test-session-456"
        
        with patch('src.core.context_builder.get_correlation_id', return_value=test_correlation_id):
            with patch('src.core.context_builder.get_session_id', return_value=test_session_id):
                assembler = ContextAssembler(user_input="test")
                ctx = assembler.build_for_decision()
                
                assert ctx["correlation_id"] == test_correlation_id
                assert ctx["session_id"] == test_session_id

    def test_correlation_in_trace(self):
        """Test that trace entries include correlation IDs"""
        test_correlation_id = "trace-correlation-789"
        
        with patch('src.core.trace.get_correlation_id', return_value=test_correlation_id):
            tracer = TurnTracer()
            tracer.new_turn()
            tracer.log("test_event", "test_component")
            
            assert tracer.traces[-1].correlation_id == test_correlation_id

    def test_correlation_id_propagation(self):
        """Test correlation ID propagation through set/get"""
        original_id = get_correlation_id()
        
        # Set new correlation ID
        new_id = "propagation-test-id"
        set_correlation_id(new_id)
        
        try:
            assert get_correlation_id() == new_id
            
            # Verify it's used in context and trace
            assembler = ContextAssembler(user_input="test")
            ctx = assembler._base()
            assert ctx["correlation_id"] == new_id
            
            tracer = TurnTracer()
            tracer.new_turn()
            tracer.log("propagation_test", "test_component")
            assert tracer.traces[-1].correlation_id == new_id
            
        finally:
            # Restore original
            set_correlation_id(original_id)


@pytest.fixture(autouse=True)
def clean_tracer():
    """Reset tracer state between tests"""
    tracer = get_tracer()
    tracer.traces.clear()
    tracer.current_turn = 0
    yield
    tracer.traces.clear()
    tracer.current_turn = 0