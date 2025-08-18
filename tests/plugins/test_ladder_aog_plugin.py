"""
Tests for LADDER-AOG Reasoning Plugin.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
import json

from src.plugins.ladder_aog_plugin import LADDERAOGPlugin, MCTSNode
from src.core.events import PlanningEvent, PlanningDecisionEvent
from src.core.aog import AOGNode, AOGNodeType
from src.core.neural_atom import NeuralAtom


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing."""
    bus = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.emit_event = AsyncMock()
    return bus


@pytest.fixture
def mock_store():
    """Mock neural store for testing."""
    store = AsyncMock()
    store.register_atom = AsyncMock()
    store.get_atom = AsyncMock()
    return store


@pytest.fixture
def plugin_config():
    """Plugin configuration for testing."""
    return {
        'ladder_aog': {
            'mcts_iterations': 50,
            'exploration_weight': 1.4,
            'max_depth': 10
        }
    }


@pytest.fixture
async def ladder_aog_plugin(mock_event_bus, mock_store, plugin_config):
    """Create and setup LADDER-AOG plugin for testing."""
    plugin = LADDERAOGPlugin()
    await plugin.setup(mock_event_bus, mock_store, plugin_config)
    return plugin


class TestMCTSNode:
    """Test MCTS node functionality."""
    
    def test_mcts_node_creation(self):
        """Test MCTS node creation."""
        aog_node = AOGNode(
            id="test_node",
            name="Test Node", 
            node_type=AOGAOGNodeType.AND,
            description="Test description"
        )
        
        mcts_node = MCTSNode(
            state={"key": "value"},
            aog_node=aog_node
        )
        
        assert mcts_node.state == {"key": "value"}
        assert mcts_node.aog_node == aog_node
        assert mcts_node.visits == 0
        assert mcts_node.value == 0.0
        assert mcts_node.children == []
        assert mcts_node.parent is None
    
    def test_ucb1_score_calculation(self):
        """Test UCB1 score calculation."""
        aog_node = AOGNode(
            id="test_node",
            name="Test Node",
            node_type=AOGAOGNodeType.AND,
            description="Test description"
        )
        
        # Test unvisited node
        mcts_node = MCTSNode(state={}, aog_node=aog_node)
        assert mcts_node.ucb1_score() == float('inf')
        
        # Test visited node with parent
        parent = MCTSNode(state={}, aog_node=aog_node)
        parent.visits = 10
        child = MCTSNode(state={}, aog_node=aog_node)
        child.visits = 3
        child.value = 1.5
        child.parent = parent
        
        score = child.ucb1_score()
        assert isinstance(score, float)
        assert score > 0
    
    def test_add_child(self):
        """Test adding child nodes."""
        parent_aog = AOGNode(
            id="parent",
            name="Parent",
            node_type=AOGNodeType.AND,
            description="Parent node"
        )
        child_aog = AOGNode(
            id="child", 
            name="Child",
            node_type=AOGNodeType.OR,
            description="Child node"
        )
        
        parent = MCTSNode(state={}, aog_node=parent_aog)
        child = MCTSNode(state={}, aog_node=child_aog)
        
        parent.add_child(child)
        
        assert child in parent.children
        assert child.parent == parent
    
    def test_select_best_child(self):
        """Test selecting best child by UCB1 score."""
        parent_aog = AOGNode(
            id="parent",
            name="Parent", 
            node_type=AOGNodeType.AND,
            description="Parent node"
        )
        
        parent = MCTSNode(state={}, aog_node=parent_aog)
        parent.visits = 10
        
        # Create children with different values
        child1_aog = AOGNode(id="child1", name="Child 1", node_type=AOGNodeType.OR, description="Child 1")
        child1 = MCTSNode(state={}, aog_node=child1_aog)
        child1.visits = 3
        child1.value = 1.0
        
        child2_aog = AOGNode(id="child2", name="Child 2", node_type=AOGNodeType.OR, description="Child 2")
        child2 = MCTSNode(state={}, aog_node=child2_aog)
        child2.visits = 2
        child2.value = 2.0
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        best = parent.select_best_child()
        # child2 should have higher UCB1 score due to higher average value
        assert best == child2
    
    def test_update_statistics(self):
        """Test updating node statistics."""
        aog_node = AOGNode(
            id="test_node",
            name="Test Node",
            node_type=AOGNodeType.AND, 
            description="Test description"
        )
        
        mcts_node = MCTSNode(state={}, aog_node=aog_node)
        
        # Update with reward
        mcts_node.update(0.5)
        assert mcts_node.visits == 1
        assert mcts_node.value == 0.5
        
        # Update again
        mcts_node.update(0.3)
        assert mcts_node.visits == 2
        assert mcts_node.value == 0.8


class TestLADDERAOGPlugin:
    """Test LADDER-AOG plugin functionality."""
    
    @pytest.mark.asyncio
    async def test_plugin_setup(self, mock_event_bus, mock_store, plugin_config):
        """Test plugin setup."""
        plugin = LADDERAOGPlugin()
        await plugin.setup(mock_event_bus, mock_store, plugin_config)
        
        assert plugin.name == "ladder_aog"
        assert plugin.config == plugin_config['ladder_aog']
        assert "task_planning" in plugin.aog_graphs
        
        # Check that AOG nodes were registered
        assert mock_store.register_atom.called
    
    @pytest.mark.asyncio
    async def test_plugin_start(self, ladder_aog_plugin, mock_event_bus):
        """Test plugin start."""
        await ladder_aog_plugin.start()
        
        # Check event subscriptions
        calls = mock_event_bus.subscribe.call_args_list
        event_types = [call[0][0] for call in calls]
        
        assert "planning" in event_types
        assert "diagnosis" in event_types
        assert "aog_update" in event_types
    
    @pytest.mark.asyncio
    async def test_aog_node_registration(self, ladder_aog_plugin):
        """Test AOG node registration as neural atoms."""
        aog_node = AOGNode(
            id="test_registration",
            name="Test Registration",
            node_type=AOGNodeType.TERMINAL,
            description="Test node for registration"
        )
        
        await ladder_aog_plugin._register_aog_node(aog_node)
        
        # Check that store.register_atom was called
        assert ladder_aog_plugin.store.register_atom.called
        
        # Get the call arguments
        call_args = ladder_aog_plugin.store.register_atom.call_args[0][0]
        assert isinstance(call_args, NeuralAtom)
        assert call_args.key == "aog:test_registration"
        assert call_args.value == aog_node
    
    @pytest.mark.asyncio
    async def test_planning_request_handling(self, ladder_aog_plugin):
        """Test handling of planning requests."""
        # Create mock planning event
        planning_event = PlanningEvent(
            goal="Complete task A",
            current_state={"position": "start"},
            action_space=["move", "pick", "place"]
        )
        
        # Mock store.get_atom to return child nodes
        async def mock_get_atom(key):
            if key.startswith("aog:"):
                node_id = key.split(":")[1]
                aog_node = AOGNode(
                    id=node_id,
                    name=f"Node {node_id}",
                    node_type=AOGNodeType.TERMINAL,
                    description=f"Mock node {node_id}"
                )
                atom = NeuralAtom(
                    key=key,
                    default_value=aog_node,
                    vector=[0.0] * 1024
                )
                return atom
            return None
        
        ladder_aog_plugin.store.get_atom.side_effect = mock_get_atom
        
        # Handle planning request
        await ladder_aog_plugin._handle_planning_request(planning_event)
        
        # Check that planning decision event was emitted
        assert ladder_aog_plugin.event_bus.emit_event.called
        
        # Get emitted event
        emit_calls = ladder_aog_plugin.event_bus.emit_event.call_args_list
        planning_decision_calls = [
            call for call in emit_calls 
            if call[0][0] == "planning_decision"
        ]
        
        assert len(planning_decision_calls) > 0
    
    @pytest.mark.asyncio
    async def test_mcts_planning_execution(self, ladder_aog_plugin):
        """Test MCTS planning execution."""
        # Setup mock session
        session_id = "test_session"
        root_aog = ladder_aog_plugin.aog_graphs["task_planning"]
        
        mcts_root = MCTSNode(
            state={"test": "state"},
            aog_node=root_aog
        )
        ladder_aog_plugin.mcts_trees[session_id] = mcts_root
        ladder_aog_plugin.reasoning_sessions[session_id] = {
            "type": "planning",
            "goal": "test goal",
            "iterations": 0
        }
        
        # Mock store.get_atom for child nodes
        async def mock_get_atom(key):
            if key.startswith("aog:"):
                node_id = key.split(":")[1]
                aog_node = AOGNode(
                    id=node_id,
                    name=f"Node {node_id}",
                    node_type=AOGNodeType.TERMINAL,
                    description=f"Mock node {node_id}"
                )
                atom = NeuralAtom(
                    key=key,
                    default_value=aog_node,
                    vector=[0.0] * 1024
                )
                return atom
            return None
        
        ladder_aog_plugin.store.get_atom.side_effect = mock_get_atom
        
        # Run MCTS planning
        plan = await ladder_aog_plugin._run_mcts_planning(session_id, iterations=5)
        
        # Check that a plan was generated
        assert isinstance(plan, list)
        assert ladder_aog_plugin.reasoning_sessions[session_id]["iterations"] == 5
    
    @pytest.mark.asyncio
    async def test_node_expansion(self, ladder_aog_plugin):
        """Test MCTS node expansion."""
        # Create test nodes
        parent_aog = AOGNode(
            id="parent_test",
            name="Parent Test",
            node_type=AOGNodeType.AND,
            description="Parent node for testing",
            children=["child1", "child2"]
        )
        
        parent_mcts = MCTSNode(
            state={"test": "state"},
            aog_node=parent_aog
        )
        
        # Mock store.get_atom for child nodes
        async def mock_get_atom(key):
            if key == "aog:child1":
                child_aog = AOGNode(
                    id="child1",
                    name="Child 1",
                    node_type=AOGNodeType.OR,
                    description="Child node 1"
                )
                return NeuralAtom(key=key, default_value=child_aog, vector=[0.0] * 1024)
            elif key == "aog:child2":
                child_aog = AOGNode(
                    id="child2", 
                    name="Child 2",
                    node_type=AOGNodeType.TERMINAL,
                    description="Child node 2"
                )
                return NeuralAtom(key=key, default_value=child_aog, vector=[0.0] * 1024)
            return None
        
        ladder_aog_plugin.store.get_atom.side_effect = mock_get_atom
        
        # Expand node
        await ladder_aog_plugin._expand_node(parent_mcts, "test_session")
        
        # Check that children were added
        assert len(parent_mcts.children) == 2
        assert all(child.parent == parent_mcts for child in parent_mcts.children)
    
    @pytest.mark.asyncio
    async def test_simulation(self, ladder_aog_plugin):
        """Test simulation for reward estimation."""
        aog_node = AOGNode(
            id="sim_test",
            name="Simulation Test",
            node_type=AOGNodeType.TERMINAL,
            description="Node for simulation testing"
        )
        
        mcts_node = MCTSNode(
            state={"test": "state"},
            aog_node=aog_node
        )
        
        # Run simulation
        reward = await ladder_aog_plugin._simulate(mcts_node, "test_session")
        
        # Check reward is valid
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
    
    @pytest.mark.asyncio
    async def test_backpropagation(self, ladder_aog_plugin):
        """Test reward backpropagation."""
        # Create chain of nodes
        root_aog = AOGNode(
            id="root",
            name="Root", 
            node_type=AOGNodeType.AND,
            description="Root node"
        )
        child_aog = AOGNode(
            id="child",
            name="Child",
            node_type=AOGNodeType.OR,
            description="Child node"
        )
        
        root = MCTSNode(state={}, aog_node=root_aog)
        child = MCTSNode(state={}, aog_node=child_aog)
        root.add_child(child)
        
        # Backpropagate reward
        await ladder_aog_plugin._backpropagate(child, 0.7)
        
        # Check that both nodes were updated
        assert child.visits == 1
        assert child.value == 0.7
        assert root.visits == 1
        assert root.value == 0.7
    
    def test_plan_confidence_calculation(self, ladder_aog_plugin):
        """Test plan confidence calculation."""
        # Test empty plan
        assert ladder_aog_plugin._calculate_plan_confidence(None) == 0.0
        assert ladder_aog_plugin._calculate_plan_confidence([]) == 0.0
        
        # Test plan with confidences
        plan = [
            {"action": "step1", "confidence": 0.8},
            {"action": "step2", "confidence": 0.6},
            {"action": "step3", "confidence": 0.9}
        ]
        
        confidence = ladder_aog_plugin._calculate_plan_confidence(plan)
        expected = (0.8 + 0.6 + 0.9) / 3
        assert abs(confidence - expected) < 0.001
    
    def test_causal_factors_extraction(self, ladder_aog_plugin):
        """Test extraction of causal factors."""
        # Test empty plan
        assert ladder_aog_plugin._extract_causal_factors(None) == []
        assert ladder_aog_plugin._extract_causal_factors([]) == []
        
        # Test plan with node IDs
        plan = [
            {"action": "step1", "node_id": "node1"},
            {"action": "step2", "node_id": "node2"},
            {"action": "step3"}  # Missing node_id
        ]
        
        factors = ladder_aog_plugin._extract_causal_factors(plan)
        assert factors == ["node1", "node2", "unknown"]
    
    @pytest.mark.asyncio
    async def test_plugin_shutdown(self, ladder_aog_plugin):
        """Test plugin shutdown."""
        # Add some data to clear
        ladder_aog_plugin.reasoning_sessions["test"] = {"data": "test"}
        ladder_aog_plugin.mcts_trees["test"] = MCTSNode(
            state={}, 
            aog_node=AOGNode(
                id="test",
                name="Test",
                node_type=AOGNodeType.AND,
                description="Test node"
            )
        )
        
        await ladder_aog_plugin.shutdown()
        
        # Check that data was cleared
        assert len(ladder_aog_plugin.reasoning_sessions) == 0
        assert len(ladder_aog_plugin.mcts_trees) == 0
