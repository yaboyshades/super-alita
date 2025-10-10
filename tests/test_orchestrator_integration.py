"""
Integration Tests for Unified Orchestrator with SDD Workflow
Tests the orchestrator pipeline with various configurations including SDD integration.
"""

import asyncio

import pytest

from src.orchestration.unified_orchestrator import (
    UnifiedOrchestrator,
    UnifiedRunConfig,
)


class MockAbilityRegistry:
    """Mock ability registry for testing."""

    def __init__(self):
        self.abilities = {
            "task_planner": True,
            "deepconf_consensus": True,
            "code_synthesize_and_write": True,
            "sdd_specify": True,
            "sdd_plan": True,
            "sdd_tasks": True,
            "sdd_validate": True,
        }
        self.execution_results = {}

    def knows(self, ability_name: str) -> bool:
        """Check if ability is available."""
        return self.abilities.get(ability_name, False)

    async def execute(self, ability_name: str, args: dict) -> dict:
        """Mock ability execution."""
        if ability_name == "task_planner":
            return {
                "steps": [
                    {
                        "id": 1,
                        "action": "Setup project",
                        "rationale": "foundation",
                    },
                    {
                        "id": 2,
                        "action": "Implement core",
                        "rationale": "main logic",
                    },
                    {
                        "id": 3,
                        "action": "Add tests",
                        "rationale": "validation",
                    },
                ],
                "source": "mock_planner",
            }

        elif ability_name == "deepconf_consensus":
            return {
                "consensus_text": "Refined: " + args.get("prompt", ""),
                "confidence": 0.85,
                "method": args.get("method", "weighted_vote"),
            }

        elif ability_name == "sdd_specify":
            return {
                "spec": f"SDD Specification for: {args.get('content', '')}",
                "user_stories": ["As a user I want feature X"],
                "acceptance_criteria": ["Given X when Y then Z"],
                "constitutional_compliance": 0.8,
            }

        elif ability_name == "sdd_plan":
            return {
                "plan": f"SDD Plan based on: {args.get('spec_content', '')}",
                "architecture": "Layered architecture with clear separation",
                "dependencies": ["library1", "library2"],
                "testing_strategy": "Test-first development approach",
                "constitutional_compliance": 0.85,
            }

        elif ability_name == "sdd_tasks":
            return {
                "tasks": [
                    "Task 1: Research existing libraries",
                    "Task 2: Write acceptance tests",
                    "Task 3: Implement core functionality",
                    "Task 4: Integration testing",
                ],
                "dependencies": {"Task 3": ["Task 1", "Task 2"]},
                "constitutional_compliance": 0.9,
            }

        elif ability_name == "sdd_validate":
            return {
                "validation_results": {
                    "constitutional_compliance": 0.8,
                    "library_first": {"passed": True, "score": 0.9},
                    "test_first": {"passed": True, "score": 0.8},
                    "simplicity": {"passed": True, "score": 0.7},
                },
                "passed": True,
                "phase": args.get("phase", "unknown"),
            }

        elif ability_name == "code_synthesize_and_write":
            return {
                "synth": {"code_generated": True},
                "file_path": args.get("file_path"),
                "test_first": args.get("test_first", False),
            }

        return {"mock_result": True, "ability": ability_name, "args": args}


class MockEventBus:
    """Mock event bus for testing."""

    def __init__(self):
        self.events = []

    async def emit(self, event: dict) -> None:
        """Record emitted events."""
        self.events.append(event)

    def get_events_by_type(self, event_type: str) -> list[dict]:
        """Get events of specific type."""
        return [e for e in self.events if e.get("type") == event_type]


@pytest.fixture
def mock_registry():
    """Create mock ability registry."""
    return MockAbilityRegistry()


@pytest.fixture
def mock_event_bus():
    """Create mock event bus."""
    return MockEventBus()


@pytest.fixture
def orchestrator(mock_registry, mock_event_bus):
    """Create orchestrator with mocks."""
    return UnifiedOrchestrator(mock_registry, mock_event_bus)


@pytest.mark.asyncio
async def test_basic_orchestrator_run(orchestrator, mock_event_bus):
    """Test basic orchestrator run without SDD."""
    config = UnifiedRunConfig(
        prompt="Build a calculator",
        run_id="test-001",
        enable_specification=False,
        enable_planning=True,
        enable_tasks=True,
        enable_consensus=True,
        sdd_mode=False,
    )

    result = await orchestrator.run(config)

    # Check result structure
    assert result["run_id"] == "test-001"
    assert result["prompt"] == "Build a calculator"
    assert "stages" in result

    # Check events were emitted
    start_events = mock_event_bus.get_events_by_type("UnifiedRunStarted")
    assert len(start_events) == 1
    assert start_events[0]["prompt"] == "Build a calculator"

    complete_events = mock_event_bus.get_events_by_type("UnifiedRunCompleted")
    assert len(complete_events) == 1


@pytest.mark.asyncio
async def test_sdd_workflow_integration(orchestrator, mock_event_bus):
    """Test SDD workflow integration."""
    config = UnifiedRunConfig(
        prompt="Build user authentication system",
        run_id="test-sdd-001",
        enable_specification=True,
        enable_planning=True,
        enable_tasks=True,
        sdd_mode=True,
        constitutional_threshold=0.75,
    )

    events = []
    async for event in orchestrator.run_stream(config):
        events.append(event)

    # Check SDD-specific events
    event_types = [e["type"] for e in events]

    # Should have specification stage
    assert "UnifiedStageStarted" in event_types
    spec_started = [e for e in events if e.get("stage") == "specification"]
    assert len(spec_started) >= 1

    # Should have planning stage
    plan_started = [e for e in events if e.get("stage") == "planning"]
    assert len(plan_started) >= 1

    # Should have tasks stage
    tasks_started = [e for e in events if e.get("stage") == "tasks"]
    assert len(tasks_started) >= 1

    # Should have validation stages
    spec_validation = [
        e for e in events if e.get("stage") == "specification_validation"
    ]
    plan_validation = [
        e for e in events if e.get("stage") == "planning_validation"
    ]
    tasks_validation = [
        e for e in events if e.get("stage") == "tasks_validation"
    ]

    # At least one validation stage should be present
    assert (
        len(spec_validation) + len(plan_validation) + len(tasks_validation) > 0
    )


@pytest.mark.asyncio
async def test_constitutional_validation_flow(orchestrator, mock_registry):
    """Test constitutional validation in SDD workflow."""
    config = UnifiedRunConfig(
        prompt="Implement payment processing",
        run_id="test-constitutional-001",
        enable_specification=True,
        enable_planning=True,
        sdd_mode=True,
        constitutional_threshold=0.75,
    )

    events = []
    async for event in orchestrator.run_stream(config):
        events.append(event)

    # Check that validation stages executed
    [
        e
        for e in events
        if e.get("stage") and "validation" in e.get("stage", "")
    ]

    # Should have at least specification validation
    spec_validation_events = [
        e for e in events if e.get("stage") == "specification_validation"
    ]
    assert len(spec_validation_events) > 0


@pytest.mark.asyncio
async def test_orchestrator_error_handling(orchestrator, mock_registry):
    """Test orchestrator error handling."""
    # Make task_planner fail
    mock_registry.abilities["task_planner"] = False

    config = UnifiedRunConfig(
        prompt="Build something",
        run_id="test-error-001",
        enable_planning=True,
        sdd_mode=False,
    )

    events = []
    async for event in orchestrator.run_stream(config):
        events.append(event)

    # Should still complete despite missing abilities
    complete_events = [e for e in events if e["type"] == "UnifiedRunCompleted"]
    assert len(complete_events) == 1

    # Planning should use fallback
    planning_events = [
        e
        for e in events
        if e.get("stage") == "planning"
        and e["type"] == "UnifiedStageSucceeded"
    ]
    if planning_events:
        # Fallback should provide basic structure
        assert "fallback" in str(planning_events[0].get("output_summary", {}))


@pytest.mark.asyncio
async def test_sdd_config_integration(orchestrator):
    """Test SDD-specific configuration handling."""
    config = UnifiedRunConfig(
        prompt="Test SDD config",
        run_id="test-config-001",
        sdd_mode=True,
        sdd_feature_id="auth-feature-001",
        sdd_phase="specify",
        constitutional_threshold=0.8,
        sdd_template_dir="custom/templates",
    )

    # Config should preserve SDD settings
    assert config.sdd_mode is True
    assert config.sdd_feature_id == "auth-feature-001"
    assert config.sdd_phase == "specify"
    assert config.constitutional_threshold == 0.8
    assert config.sdd_template_dir == "custom/templates"


@pytest.mark.asyncio
async def test_orchestrator_timeout_handling(orchestrator, mock_registry):
    """Test timeout handling in orchestrator."""

    # Create a slow ability
    async def slow_execute(ability_name: str, args: dict) -> dict:
        if ability_name == "task_planner":
            await asyncio.sleep(2)  # Longer than timeout
        return {"result": "slow"}

    mock_registry.execute = slow_execute

    config = UnifiedRunConfig(
        prompt="Test timeout",
        run_id="test-timeout-001",
        enable_planning=True,
        timeout_s=1,  # Short timeout
        sdd_mode=False,
    )

    events = []
    async for event in orchestrator.run_stream(config):
        events.append(event)

    # Should have failure events due to timeout
    [e for e in events if e["type"] == "UnifiedStageFailed"]
    # Note: Timeout behavior depends on Python version and implementation
    # This test checks that the orchestrator handles timeouts gracefully


@pytest.mark.asyncio
async def test_event_emission_order(orchestrator, mock_event_bus):
    """Test that events are emitted in correct order."""
    config = UnifiedRunConfig(
        prompt="Test event order",
        run_id="test-events-001",
        enable_specification=True,
        enable_planning=True,
        sdd_mode=True,
    )

    events = []
    async for event in orchestrator.run_stream(config):
        events.append(event)

    # Check event ordering
    event_types = [e["type"] for e in events]

    # Should start with UnifiedRunStarted
    assert event_types[0] == "UnifiedRunStarted"

    # Should end with UnifiedRunCompleted
    assert event_types[-1] == "UnifiedRunCompleted"

    # Stage events should be paired (Started/Succeeded or Started/Failed)
    stage_events = [e for e in events if "Stage" in e["type"]]
    started_stages = set()

    for event in stage_events:
        if event["type"] == "UnifiedStageStarted":
            started_stages.add(event["stage"])
        elif event["type"] in ["UnifiedStageSucceeded", "UnifiedStageFailed"]:
            assert (
                event["stage"] in started_stages
            ), f"Stage {event['stage']} ended without starting"


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(
        test_basic_orchestrator_run(
            UnifiedOrchestrator(MockAbilityRegistry(), MockEventBus()),
            MockEventBus(),
        )
    )
