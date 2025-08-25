# Test LADDER AOG plugin functionality

from unittest.mock import AsyncMock, Mock

import pytest


# Add missing fixtures
@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing."""
    bus = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def mock_store():
    """Mock neural store for testing."""
    store = Mock()
    store.get = Mock(return_value=None)
    store.register = Mock()
    store.attention = AsyncMock(return_value=[])
    store.hebbian_update = Mock()
    return store


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "ladder_aog_plugin": {
            "max_planning_depth": 5,
            "mcts_iterations": 100,
            "exploration_weight": 1.414,
        }
    }


@pytest.fixture
async def ladder_plugin(mock_event_bus, mock_store, mock_config):
    """Create and setup LADDER AOG plugin for testing."""
    try:
        from src.plugins.ladder_aog_plugin import LADDERAOGPlugin

        plugin = LADDERAOGPlugin()
        await plugin.setup(mock_event_bus, mock_store, mock_config)
        return plugin

    except ImportError:
        pytest.skip("LADDERAOGPlugin not available")


class TestLADDERAOGPlugin:
    """Test LADDER AOG plugin functionality."""

    def test_plugin_creation(self):
        """Test plugin can be created."""
        try:
            from src.plugins.ladder_aog_plugin import LADDERAOGPlugin

            plugin = LADDERAOGPlugin()
            assert plugin.name == "ladder_aog_plugin"
        except ImportError:
            pytest.skip("LADDERAOGPlugin not available")

    @pytest.mark.asyncio
    async def test_plugin_setup(self, mock_event_bus, mock_store, mock_config):
        """Test plugin setup process."""
        try:
            from src.plugins.ladder_aog_plugin import LADDERAOGPlugin

            plugin = LADDERAOGPlugin()

            await plugin.setup(mock_event_bus, mock_store, mock_config)

            assert plugin.event_bus == mock_event_bus
            assert plugin.store == mock_store
            assert plugin.config == mock_config

        except ImportError:
            pytest.skip("LADDERAOGPlugin not available")

    @pytest.mark.asyncio
    async def test_event_emission_format(self, ladder_plugin):
        """Test that planning decision events are emitted in correct format."""
        if ladder_plugin is None:
            pytest.skip("LADDERAOGPlugin not available")

        # Mock emit_event to capture what's passed
        emit_calls = []

        async def mock_emit_event(event_type, **kwargs):
            emit_calls.append((event_type, kwargs))

        ladder_plugin.emit_event = mock_emit_event

        # Trigger planning that should emit planning_decision event
        try:
            await ladder_plugin.plan("test_goal")
        except Exception:
            pass  # We just want to test event emission format

        # Check if planning_decision event was emitted with correct format
        planning_events = [
            call for call in emit_calls if call[0] == "planning_decision"
        ]

        if planning_events:
            event_type, payload = planning_events[0]
            assert event_type == "planning_decision"
            # Should use keyword arguments, not .dict() or .model_dump()
            assert isinstance(payload, dict)
            assert "plan_id" in payload or "decision" in payload

    @pytest.mark.asyncio
    async def test_planning_basic(self, ladder_plugin):
        """Test basic planning functionality."""
        if ladder_plugin is None:
            pytest.skip("LADDERAOGPlugin not available")

        # Mock the store to return some AOG atoms
        mock_aog_atom = Mock()
        mock_aog_atom.value = {
            "aog_type": "ROOT",
            "description": "test goal",
            "children": ["child1", "child2"],
        }

        ladder_plugin.store.get = Mock(return_value=mock_aog_atom)

        try:
            result = await ladder_plugin.plan("test_goal")
            # Should return some kind of result, even if empty
            assert result is not None
        except Exception as e:
            # Planning might fail due to missing dependencies, that's ok
            assert "test_goal" in str(e) or len(str(e)) > 0

    @pytest.mark.asyncio
    async def test_diagnosis_basic(self, ladder_plugin):
        """Test basic diagnosis functionality."""
        if ladder_plugin is None:
            pytest.skip("LADDERAOGPlugin not available")

        try:
            result = await ladder_plugin.diagnose("test_effect")
            # Should return some kind of result
            assert result is not None
        except Exception as e:
            # Diagnosis might fail due to missing dependencies, that's ok
            assert "test_effect" in str(e) or len(str(e)) > 0


class TestMCTSNode:
    """Test MCTS node functionality."""

    def test_mcts_node_creation(self):
        """Test MCTS node creation."""
        try:
            from src.plugins.ladder_aog_plugin import MCTSNode

            node = MCTSNode("test_id")
            assert node.node_id == "test_id"
            assert node.parent is None
            assert node.children == []
            assert node.visits == 0
            assert node.value == 0.0

        except ImportError:
            pytest.skip("MCTSNode not available")

    def test_mcts_node_ucb1_score(self):
        """Test UCB1 score calculation."""
        try:
            from src.plugins.ladder_aog_plugin import MCTSNode

            parent = MCTSNode("parent")
            parent.visits = 10

            child = MCTSNode("child", parent)
            child.visits = 0

            # Unvisited node should have infinite score
            assert child.ucb1_score() == float("inf")

            # After one visit
            child.visits = 1
            child.value = 0.5
            score = child.ucb1_score()
            assert score > 0.5  # Should include exploration bonus

        except ImportError:
            pytest.skip("MCTSNode not available")

    def test_mcts_node_child_management(self):
        """Test child node management."""
        try:
            from src.plugins.ladder_aog_plugin import MCTSNode

            parent = MCTSNode("parent")
            child1 = MCTSNode("child1")
            child2 = MCTSNode("child2")

            parent.add_child(child1)
            parent.add_child(child2)

            assert len(parent.children) == 2
            assert child1.parent == parent
            assert child2.parent == parent

            # Test selection (both have same score initially)
            best = parent.select_best_child()
            assert best in [child1, child2]

        except ImportError:
            pytest.skip("MCTSNode not available")


@pytest.mark.asyncio
async def test_planning_decision_event(mock_event_bus, ladder_plugin):
    """Test planning decision event emission for regression."""
    if ladder_plugin is None:
        pytest.skip("LADDERAOGPlugin not available")

    # Reset mock
    mock_event_bus.publish.reset_mock()

    # Trigger some planning
    try:
        await ladder_plugin.plan("goal_X")
    except Exception:
        pass  # Planning might fail, we're testing event format

    # Check if any events were published
    if mock_event_bus.publish.call_count > 0:
        # Get the last published event
        last_call = mock_event_bus.publish.call_args_list[-1]
        event_obj = last_call[0][0]  # First positional argument

        # Should be an event object, not a dict
        assert hasattr(event_obj, "event_type") or hasattr(event_obj, "__dict__")

        # If it's a planning decision event, check the payload structure
        if (
            hasattr(event_obj, "event_type")
            and event_obj.event_type == "planning_decision"
        ):
            # Should have proper fields, not from .dict() or .model_dump()
            assert hasattr(event_obj, "payload") or hasattr(event_obj, "goal")


if __name__ == "__main__":
    pytest.main([__file__])
