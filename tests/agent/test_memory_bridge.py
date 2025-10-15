"""Tests for memory bridge functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.agent.memory_bridge import MemoryBridge

@pytest.fixture
def mock_knowledge_graph():
    """Mock knowledge graph for testing."""
    kg = MagicMock()
    kg.create_atom = AsyncMock(return_value={"id": "atom_123"})
    kg.semantic_search = AsyncMock(return_value=[
        {"id": "result_1", "content": "test content", "success": True}
    ])
    kg.retrieve_relevant_context = AsyncMock(return_value=[
        {"context": "test context"}
    ])
    return kg

@pytest.fixture
def memory_bridge(mock_knowledge_graph):
    """Memory bridge fixture."""
    return MemoryBridge(mock_knowledge_graph)

@pytest.mark.asyncio
async def test_write_interaction(memory_bridge, mock_knowledge_graph):
    """Test writing interaction to knowledge graph."""
    interaction_data = {"action": "test_action", "result": "success"}
    context = {"user_id": "test_user"}
    
    atom_id = await memory_bridge.write_interaction("test_interaction", interaction_data, context)
    
    assert atom_id == "atom_123"
    mock_knowledge_graph.create_atom.assert_called_once()

@pytest.mark.asyncio
async def test_fetch_context(memory_bridge, mock_knowledge_graph):
    """Test fetching context from knowledge graph."""
    results = await memory_bridge.fetch_context("test query")
    
    assert len(results) > 0
    assert results[0]["id"] == "result_1"

@pytest.mark.asyncio  
async def test_store_reflection(memory_bridge):
    """Test storing agent reflection."""
    reflection_id = await memory_bridge.store_reflection(
        "session_123", "I learned something new", True, {"insight": "test"}
    )
    
    assert reflection_id == "atom_123"

@pytest.mark.asyncio
async def test_get_similar_successes(memory_bridge):
    """Test getting similar successful interactions."""
    successes = await memory_bridge.get_similar_successes("code generation")
    
    assert len(successes) > 0
    assert successes[0]["success"] is True

@pytest.mark.asyncio
async def test_learn_from_failure(memory_bridge):
    """Test learning from failed interactions."""
    failure_id = await memory_bridge.learn_from_failure(
        "session_456", "generate code", "timeout", [{"action": "generate"}]
    )
    
    assert failure_id == "atom_123"