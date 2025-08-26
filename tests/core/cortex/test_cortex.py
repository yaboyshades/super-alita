"""
Tests for Cortex runtime and modules
"""

import pytest
import asyncio
from typing import Dict, Any

from src.core.cortex import (
    CortexRuntime, 
    CortexContext,
    PerformanceMarker,
    CortexPhase,
    MarkerType,
    create_cortex_runtime
)
from src.core.cortex.modules import (
    TextPerceptionModule,
    LogicalReasoningModule, 
    PlanningActionModule,
    CortexInput,
    ModuleResult
)


@pytest.fixture
def cortex_runtime():
    """Create a test Cortex runtime"""
    return create_cortex_runtime()


@pytest.fixture
def cortex_context():
    """Create a test Cortex context"""
    runtime = create_cortex_runtime()
    return runtime.create_context(
        session_id="test_session",
        user_id="test_user",
        workspace="test_workspace"
    )


@pytest.mark.asyncio
class TestCortexRuntime:
    """Test Cortex runtime functionality"""
    
    async def test_runtime_initialization(self, cortex_runtime):
        """Test runtime initializes correctly"""
        await cortex_runtime.setup()
        
        assert cortex_runtime.running is True
        assert len(cortex_runtime.perception_modules) == 1
        assert len(cortex_runtime.reasoning_modules) == 1
        assert len(cortex_runtime.action_modules) == 1
        
        await cortex_runtime.shutdown()
        assert cortex_runtime.running is False
    
    async def test_context_creation(self, cortex_runtime):
        """Test context creation and management"""
        context = cortex_runtime.create_context(
            session_id="test_session",
            user_id="test_user",
            workspace="/test/workspace",
            custom_meta="test_value"
        )
        
        assert context.cycle_id is not None
        assert context.session_id == "test_session"
        assert context.user_id == "test_user"
        assert context.workspace == "/test/workspace"
        assert context.metadata["custom_meta"] == "test_value"
        assert context.cycle_id in cortex_runtime.active_cycles
    
    async def test_complete_processing_cycle(self, cortex_runtime, cortex_context):
        """Test complete perception → reasoning → action cycle"""
        await cortex_runtime.setup()
        
        try:
            # Test with text input
            input_text = "Create a new Python function to calculate fibonacci numbers"
            
            result = await cortex_runtime.process_cycle(input_text, cortex_context)
            
            # Verify results
            assert result.success is True
            assert result.cycle_id == cortex_context.cycle_id
            assert result.perception_result is not None
            assert result.reasoning_result is not None
            assert result.action_result is not None
            assert result.total_duration_ms > 0
            assert len(result.performance_markers) > 0
            
            # Check perception results
            perception = result.perception_result
            assert perception.processed_data is not None
            assert perception.confidence > 0
            assert "length" in perception.features
            assert "word_count" in perception.features
            
            # Check reasoning results
            reasoning = result.reasoning_result
            assert len(reasoning.conclusions) > 0
            assert reasoning.confidence > 0
            assert "intent" in reasoning.analysis
            
            # Check action results
            action = result.action_result
            assert len(action.actions) > 0
            assert len(action.priority_scores) > 0
            assert action.execution_plan is not None
            
        finally:
            await cortex_runtime.shutdown()
    
    async def test_error_handling(self, cortex_runtime):
        """Test error handling in processing cycle"""
        await cortex_runtime.setup()
        
        try:
            # Create context with invalid input
            context = cortex_runtime.create_context("error_test")
            
            # Test with None input (should cause error)
            result = await cortex_runtime.process_cycle(None, context)
            
            assert result.success is False
            assert result.error is not None
            assert "perception" in result.error.lower()
            
        finally:
            await cortex_runtime.shutdown()


@pytest.mark.asyncio 
class TestCortexModules:
    """Test individual Cortex modules"""
    
    async def test_text_perception_module(self):
        """Test text perception module"""
        module = TextPerceptionModule("test_perception")
        
        # Test valid text input
        cortex_input = CortexInput(
            raw_data="Hello world, this is a test message with questions?",
            context={},
            metadata={},
            cycle_id="test_cycle"
        )
        
        result = await module.process(cortex_input, {})
        
        assert result.success is True
        assert result.data is not None
        assert result.data.confidence > 0
        assert result.data.features["has_questions"] is True
        assert result.data.features["word_count"] == 9
        
        # Test invalid input
        invalid_input = CortexInput(
            raw_data=123,  # Not a string
            context={},
            metadata={},
            cycle_id="test_cycle"
        )
        
        result = await module.process(invalid_input, {})
        assert result.success is False
        assert "must be string" in result.error
    
    async def test_logical_reasoning_module(self):
        """Test logical reasoning module"""
        module = LogicalReasoningModule("test_reasoning")
        
        # Create mock perception result
        from src.core.cortex.modules import PerceptionResult
        
        perception_result = PerceptionResult(
            processed_data={"text": "Create a function"},
            features={
                "has_commands": True,
                "has_questions": False,
                "complexity_score": 0.3
            },
            confidence=0.8
        )
        
        result = await module.process(perception_result, {})
        
        assert result.success is True
        assert result.data is not None
        assert "User is requesting action" in result.data.conclusions
        assert result.data.analysis["intent"] == "command"
        assert result.data.reasoning_chain is not None
        assert len(result.data.reasoning_chain) == 2
    
    async def test_planning_action_module(self):
        """Test planning action module"""
        module = PlanningActionModule("test_action")
        
        # Create mock reasoning result
        from src.core.cortex.modules import ReasoningResult
        
        reasoning_result = ReasoningResult(
            analysis={"intent": "command", "complexity": "high"},
            conclusions=["User is requesting action", "High complexity task detected"],
            confidence=0.9
        )
        
        result = await module.process(reasoning_result, {})
        
        assert result.success is True
        assert result.data is not None
        assert len(result.data.actions) >= 1
        assert "execute_task" in [action["type"] for action in result.data.actions]
        assert "monitor_progress" in [action["type"] for action in result.data.actions]
        assert result.data.execution_plan is not None


@pytest.mark.asyncio
class TestPerformanceTracking:
    """Test performance tracking functionality"""
    
    async def test_performance_tracker(self):
        """Test performance marker tracking"""
        from src.core.cortex.markers import PerformanceTracker
        
        tracker = PerformanceTracker()
        
        # Test cycle tracking
        cycle_id = "test_cycle"
        start_marker_id = tracker.start_cycle(cycle_id)
        assert start_marker_id is not None
        assert tracker.current_cycle_id == cycle_id
        
        # Test phase tracking
        phase_marker_id = tracker.start_phase(CortexPhase.PERCEPTION)
        assert phase_marker_id is not None
        
        # Simulate some processing time
        await asyncio.sleep(0.01)
        
        end_phase_marker_id = tracker.end_phase(CortexPhase.PERCEPTION)
        assert end_phase_marker_id is not None
        
        # Test module execution tracking
        module_marker_id = tracker.track_module_execution(
            module_name="test_module",
            phase=CortexPhase.PERCEPTION,
            duration_ms=10.5,
            metrics={"test_metric": 42}
        )
        assert module_marker_id is not None
        
        # End cycle
        end_marker_id = tracker.end_cycle()
        assert end_marker_id is not None
        assert tracker.current_cycle_id is None
        
        # Verify markers
        markers = tracker.get_markers()
        assert len(markers) >= 4
        
        marker_types = [m.marker_type for m in markers]
        assert MarkerType.CYCLE_START in marker_types
        assert MarkerType.CYCLE_END in marker_types
        assert MarkerType.PHASE_START in marker_types
        assert MarkerType.PHASE_END in marker_types
        assert MarkerType.MODULE_EXECUTION in marker_types
    
    async def test_cortex_event_creation(self):
        """Test Cortex event creation with markers"""
        from src.core.cortex.markers import create_cortex_event, PerformanceMarker
        
        markers = [
            PerformanceMarker(
                id="marker_1",
                marker_type=MarkerType.CYCLE_START,
                phase=None,
                timestamp=1000.0
            ),
            PerformanceMarker(
                id="marker_2", 
                marker_type=MarkerType.CYCLE_END,
                phase=None,
                timestamp=1001.0,
                duration_ms=1000.0
            )
        ]
        
        cortex_event = create_cortex_event(
            event_type="test_event",
            cycle_id="test_cycle",
            phase=CortexPhase.PERCEPTION,
            markers=markers,
            context={"test": "value"}
        )
        
        assert cortex_event.cycle_id == "test_cycle"
        assert cortex_event.phase == CortexPhase.PERCEPTION
        assert len(cortex_event.markers) == 2
        assert cortex_event.context["test"] == "value"
        assert cortex_event.base_event is not None


# Integration test
@pytest.mark.asyncio
async def test_cortex_integration():
    """Test full Cortex integration with event bus"""
    from src.core.event_bus import EventBus
    
    # Create runtime with event bus
    runtime = create_cortex_runtime()
    event_bus = EventBus()
    
    await runtime.setup(event_bus=event_bus)
    
    try:
        # Create context and process input
        context = runtime.create_context("integration_test")
        result = await runtime.process_cycle(
            "Build a web scraper to extract product data", 
            context
        )
        
        # Verify successful processing
        assert result.success is True
        assert result.total_duration_ms > 0
        
        # Check that events were emitted
        # Note: This requires the event bus to have a way to check emitted events
        # For now, just verify no errors occurred
        
    finally:
        await runtime.shutdown()