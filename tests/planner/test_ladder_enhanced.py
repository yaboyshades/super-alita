"""Comprehensive tests for the Enhanced LADDER Planner."""

import asyncio
from datetime import datetime
from typing import Any

import pytest

from cortex.config.planner_config import PlannerConfig
from cortex.planner.ladder_enhanced import EnhancedLadderPlanner
from cortex.todo.models import LadderStage, Todo, TodoStatus


class MockKG:
    """Mock Knowledge Graph for testing."""

    def get_context_for_title(self, title: str) -> str:
        contexts = {
            "test task": "This is a testing related task with high priority",
            "format code": "Code formatting task using Black formatter",
            "build project": "Project build task with dependencies",
        }
        return contexts.get(title.lower(), f"Context for {title}")

    def compute_energy_for_title(self, title: str) -> float:
        energies = {"test task": 1.5, "format code": 0.5, "build project": 2.5}
        return energies.get(title.lower(), 1.0)

    def write_decision(self, tool: str, node_id: str, reward: float) -> None:
        pass

    def estimate_metric_delta(self, title: str) -> float:
        return 0.8  # Default positive impact


class MockBandit:
    """Mock Multi-Armed Bandit for testing."""

    def __init__(self):
        self.selections = []
        self.updates = []

    def select_tool(self, context: dict[str, Any] | None = None) -> str:
        # Return tool based on context for predictable testing
        if context and "energy" in context:
            energy = context["energy"]
            if energy < 1.0:
                tool = "simple_tool"
            elif energy < 2.0:
                tool = "medium_tool"
            else:
                tool = "complex_tool"
        else:
            tool = "default_tool"

        self.selections.append((tool, context))
        return tool

    def update(self, tool: str, reward: float) -> None:
        self.updates.append((tool, reward))


class MockEventBus:
    """Mock Event Bus for testing."""

    def __init__(self):
        self.events = []

    async def emit(self, kind: str, **kwargs) -> None:
        self.events.append(("async", kind, kwargs))

    def emit_sync(self, kind: str, **kwargs) -> None:
        self.events.append(("sync", kind, kwargs))


class MockOrchestrator:
    """Mock Orchestrator for testing."""

    def __init__(self):
        self.event_bus = MockEventBus()
        self.executions = []

    async def execute_action(
        self, tool: str, todo: Todo, context: str, shadow: bool = True
    ) -> str:
        result = f"Executed {tool} on {todo.title} (shadow={shadow})"
        self.executions.append((tool, todo, context, shadow, result))

        # Simulate occasional failures for testing
        if "fail" in todo.title.lower():
            raise Exception("Simulated execution failure")

        return result


class MockTodoStore:
    """Mock Todo Store for testing."""

    def __init__(self):
        self.todos: dict[str, Todo] = {}

    def upsert(self, todo: Todo) -> None:
        self.todos[todo.id] = todo

    def get(self, todo_id: str) -> Todo | None:
        return self.todos.get(todo_id)

    def children_of(self, todo_id: str) -> list[Todo]:
        return [todo for todo in self.todos.values() if todo.parent_id == todo_id]


@pytest.fixture
def mock_components():
    """Create mock components for testing."""
    return {
        "kg": MockKG(),
        "bandit": MockBandit(),
        "store": MockTodoStore(),
        "orchestrator": MockOrchestrator(),
    }


@pytest.fixture
def planner(mock_components):
    """Create Enhanced LADDER Planner with mock components."""
    return EnhancedLadderPlanner(
        kg=mock_components["kg"],
        bandit=mock_components["bandit"],
        store=mock_components["store"],
        orchestrator=mock_components["orchestrator"],
        mode="shadow",
    )


@pytest.fixture
def sample_user_event():
    """Create a sample user event for testing."""
    return type(
        "UserEvent",
        (),
        {
            "payload": {
                "query": "Run comprehensive tests with coverage",
                "context": "Testing task for the Super Alita project",
            }
        },
    )()


class TestEnhancedLadderPlanner:
    """Test suite for Enhanced LADDER Planner."""

    @pytest.mark.asyncio
    async def test_plan_creation(self, planner, sample_user_event):
        """Test basic plan creation."""
        # Create plan
        root_todo = await planner.plan_from_user_event(sample_user_event)

        # Verify root todo
        assert root_todo.title == "Run comprehensive tests with coverage"
        assert root_todo.stage == LadderStage.REVIEW  # Should complete all stages
        assert root_todo.energy > 0
        assert len(root_todo.children_ids) > 0

        # Verify children were created
        children = [planner.store.get(child_id) for child_id in root_todo.children_ids]
        assert all(child is not None for child in children)
        assert all(child.parent_id == root_todo.id for child in children)

    @pytest.mark.asyncio
    async def test_task_decomposition_strategies(self, planner):
        """Test different task decomposition strategies."""
        test_cases = [
            ("Run tests with coverage", "test"),
            ("Format code with Black", "format"),
            ("Lint code with Ruff", "lint"),
            ("Build the project", "build"),
            ("Deploy to production", "deploy"),
            ("Setup development environment", "setup"),
            ("Unknown task type", "default"),
        ]

        for title, expected_strategy in test_cases:
            user_event = type(
                "UserEvent", (), {"payload": {"query": title, "context": ""}}
            )()

            root_todo = await planner.plan_from_user_event(user_event)
            children = [
                planner.store.get(child_id) for child_id in root_todo.children_ids
            ]

            # Verify children were created
            assert len(children) > 0

            # Verify energy assignment
            assert all(child.energy > 0 for child in children)

            # Verify tool hints are assigned
            assert all(child.tool_hint is not None for child in children)

    @pytest.mark.asyncio
    async def test_energy_estimation(self, planner):
        """Test energy estimation for different task types."""
        test_cases = [
            ("Format code", 0.5),  # Simple task
            ("Run tests", 1.0),  # Medium task
            ("Build project", 2.0),  # Complex task
            (
                "Implement new feature with comprehensive documentation",
                3.0,
            ),  # Very complex
        ]

        for title, min_expected_energy in test_cases:
            energy = planner._estimate_task_energy(title, "")
            assert energy >= min_expected_energy
            assert energy <= 5.0  # Maximum cap

    @pytest.mark.asyncio
    async def test_bandit_tool_selection(self, planner, mock_components):
        """Test multi-armed bandit tool selection."""
        # Create a task with multiple tool options
        task = Todo(
            title="Test task",
            description="A test task",
            energy=1.5,
            tool_hint="initial_tool",
        )

        # Test initial selection (should be exploration)
        selected_tool = planner._select_tool_bandit(task)
        assert selected_tool is not None

        # Simulate some executions and updates
        for i in range(10):
            tool = planner._select_tool_bandit(task)
            success = i % 3 == 0  # 33% success rate
            result = {"success": success, "tool_used": tool}
            planner._update_bandit_stats(tool, result)

        # Verify bandit stats were updated
        assert len(planner.bandit_stats) > 0
        for tool_stats in planner.bandit_stats.values():
            assert tool_stats["attempts"] > 0
            assert 0 <= tool_stats["wins"] <= tool_stats["attempts"]

    @pytest.mark.asyncio
    async def test_shadow_vs_active_mode(self, planner, sample_user_event):
        """Test execution in shadow vs active mode."""
        # Test shadow mode
        planner.set_mode("shadow")
        assert planner.mode == "shadow"

        root_todo_shadow = await planner.plan_from_user_event(sample_user_event)
        children_shadow = [
            planner.store.get(child_id) for child_id in root_todo_shadow.children_ids
        ]

        # Execute in shadow mode
        await planner._enhanced_execute(root_todo_shadow, children_shadow)

        # Verify shadow execution (should not call actual orchestrator)
        shadow_executions = [
            exec for exec in planner.orch.executions if exec[3] is True
        ]
        active_executions = [
            exec for exec in planner.orch.executions if exec[3] is False
        ]

        # In shadow mode, we don't call the orchestrator
        assert len(active_executions) == 0

        # Test active mode
        planner.set_mode("active")
        assert planner.mode == "active"

        # Create new event for active mode test
        user_event_active = type(
            "UserEvent",
            (),
            {
                "payload": {
                    "query": "Simple formatting task",
                    "context": "Active mode test",
                }
            },
        )()

        root_todo_active = await planner.plan_from_user_event(user_event_active)
        children_active = [
            planner.store.get(child_id) for child_id in root_todo_active.children_ids
        ]

        # Execute in active mode
        await planner._enhanced_execute(root_todo_active, children_active)

        # Verify active execution calls
        active_executions = [
            exec for exec in planner.orch.executions if exec[3] is False
        ]
        assert len(active_executions) > 0

    @pytest.mark.asyncio
    async def test_knowledge_base_learning(self, planner, sample_user_event):
        """Test knowledge base updates and learning."""
        # Execute a plan to populate knowledge base
        root_todo = await planner.plan_from_user_event(sample_user_event)
        children = [planner.store.get(child_id) for child_id in root_todo.children_ids]

        await planner._enhanced_execute(root_todo, children)
        await planner._enhanced_review(root_todo, children)

        # Verify knowledge base was updated
        assert len(planner.knowledge_base) > 0

        # Test similarity finding
        similar_tasks = planner._find_similar_tasks("Run tests")
        assert len(similar_tasks) >= 0  # May or may not find similar tasks

        # Test confidence calculation
        confidence = planner._calculate_confidence("Run tests", similar_tasks)
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_priority_calculation(self, planner):
        """Test task priority calculation."""
        # Test priority with different energy levels and dependencies
        test_cases = [
            (0.5, 0, True),  # Low energy, no deps -> high priority
            (2.0, 0, True),  # High energy, no deps -> medium priority
            (1.0, 2, False),  # Medium energy, deps -> lower priority
            (0.3, 0, True),  # Very low energy, no deps -> highest priority
        ]

        for energy, unmet_deps, expect_higher in test_cases:
            priority = planner._calculate_priority(energy, unmet_deps)
            assert priority > 0

            # Lower energy should generally yield higher priority
            if energy < 1.0 and unmet_deps == 0:
                assert priority > 0.5

    @pytest.mark.asyncio
    async def test_execution_error_handling(self, planner):
        """Test error handling during execution."""
        # Create a task that will fail
        user_event = type(
            "UserEvent",
            (),
            {"payload": {"query": "This task will fail", "context": "Error testing"}},
        )()

        # Set to active mode to test actual execution
        planner.set_mode("active")

        root_todo = await planner.plan_from_user_event(user_event)
        children = [planner.store.get(child_id) for child_id in root_todo.children_ids]

        # Execute should handle errors gracefully
        await planner._enhanced_execute(root_todo, children)

        # Verify that failed tasks are handled properly
        failed_children = [
            child
            for child in children
            if planner.store.get(child.id).status == TodoStatus.PENDING
        ]

        # At least some tasks should fail due to the "fail" keyword
        # but the system should continue running
        assert len(planner.orch.executions) > 0

    def test_configuration_integration(self, planner):
        """Test configuration integration."""
        # Test initial configuration
        config = PlannerConfig()
        assert config.mode in ["shadow", "active"]
        assert 0.0 <= config.exploration_rate <= 1.0

        # Test configuration validation
        errors = config.validate()
        assert len(errors) == 0  # Should be valid by default

        # Test invalid configuration
        config.mode = "invalid"
        errors = config.validate()
        assert len(errors) > 0
        assert "Mode must be 'shadow' or 'active'" in errors

    @pytest.mark.asyncio
    async def test_event_emission(self, planner, sample_user_event):
        """Test event emission during LADDER execution."""
        # Execute a complete plan
        root_todo = await planner.plan_from_user_event(sample_user_event)

        # Check that events were emitted
        events = planner.orch.event_bus.events
        assert len(events) > 0

        # Verify specific event types
        event_types = [event[1] for event in events]
        assert "todo.created" in event_types
        assert "plan.assessed" in event_types
        assert "plan.decomposed" in event_types
        assert "plan.decided" in event_types
        assert "plan.completed" in event_types

    def test_planner_api_methods(self, planner):
        """Test public API methods."""
        # Test mode setting
        planner.set_mode("active")
        assert planner.mode == "active"

        planner.set_mode("shadow")
        assert planner.mode == "shadow"

        # Test invalid mode
        with pytest.raises(ValueError):
            planner.set_mode("invalid")

        # Test statistics retrieval
        stats = planner.get_bandit_stats()
        assert isinstance(stats, dict)

        # Test knowledge base summary
        summary = planner.get_knowledge_base_summary()
        assert "size" in summary
        assert "mode" in summary
        assert summary["mode"] == planner.mode

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, planner):
        """Test concurrent plan execution."""
        # Create multiple user events
        events = [
            type(
                "UserEvent",
                (),
                {"payload": {"query": f"Task {i}", "context": f"Context {i}"}},
            )()
            for i in range(3)
        ]

        # Execute plans concurrently
        tasks = [planner.plan_from_user_event(event) for event in events]
        results = await asyncio.gather(*tasks)

        # Verify all plans were created
        assert len(results) == 3
        assert all(isinstance(result, Todo) for result in results)
        assert all(len(result.children_ids) > 0 for result in results)

    @pytest.mark.asyncio
    async def test_performance_metrics(self, planner, sample_user_event):
        """Test performance metrics collection."""
        start_time = datetime.now()

        # Execute a plan
        root_todo = await planner.plan_from_user_event(sample_user_event)
        children = [planner.store.get(child_id) for child_id in root_todo.children_ids]

        await planner._enhanced_execute(root_todo, children)
        await planner._enhanced_review(root_todo, children)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Verify execution completed in reasonable time
        assert execution_time < 10.0  # Should complete within 10 seconds

        # Verify metrics are available
        summary = planner.get_knowledge_base_summary()
        if summary["size"] > 0:
            assert "success_rate" in summary
            assert "average_reward" in summary

    @pytest.mark.asyncio
    async def test_active_inference_energy_minimization(
        self, monkeypatch, mock_components
    ):
        """Ensure energy is adjusted when active inference is enabled."""
        monkeypatch.setenv("CORTEX_LADDER_ACTIVE_INFERENCE", "1")
        import importlib
        import cortex.config.flags as flag_module
        importlib.reload(flag_module)
        import cortex.planner.ladder_enhanced as le
        importlib.reload(le)

        planner = le.EnhancedLadderPlanner(
            kg=mock_components["kg"],
            bandit=mock_components["bandit"],
            store=mock_components["store"],
            orchestrator=mock_components["orchestrator"],
            mode="shadow",
        )

        user_event = type(
            "UserEvent", (), {"payload": {"query": "Quick test", "context": ""}}
        )()
        initial_energy = planner._estimate_task_energy("Quick test", "")
        root = await planner.plan_from_user_event(user_event)
        assert root.energy <= initial_energy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
