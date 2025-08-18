"""
Test the enhanced SemanticMemoryPlugin functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from src.plugins.semantic_memory_plugin import SemanticMemoryPlugin


@pytest.mark.asyncio
async def test_semantic_memory_plugin_lifecycle():
    """Test SemanticMemoryPlugin setup, start, and shutdown."""
    
    # Create mock dependencies
    mock_event_bus = Mock()
    mock_event_bus.emit = AsyncMock()
    mock_store = Mock()
    mock_store.register_with_lineage = Mock()
    mock_store.get = Mock(return_value=None)
    mock_store.attention = AsyncMock(return_value=[])
    
    config = {
        "db_path": "./test_data/chroma_db",
        "collection_name": "test_memory"
    }
    
    # Create plugin
    plugin = SemanticMemoryPlugin()
    
    # Test name property
    assert plugin.name == "semantic_memory"
    
    # Test setup
    await plugin.setup(mock_event_bus, mock_store, config)
    assert plugin._event_bus == mock_event_bus
    assert plugin._store == mock_store
    assert plugin._config == config
    
    # Mock ChromaDB components for start
    with patch('chromadb.PersistentClient') as mock_chroma_client:
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client
        
        # Test start
        await plugin.start()
        assert plugin._chroma_client == mock_client
        assert plugin._collection == mock_collection
    
    # Test shutdown
    await plugin.shutdown()


@pytest.mark.asyncio
async def test_embed_text():
    """Test text embedding functionality."""
    
    plugin = SemanticMemoryPlugin()
    
    # Test embedding generation
    texts = ["hello world", "test embedding"]
    embeddings = await plugin.embed_text(texts)
    
    assert len(embeddings) == 2
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)
    assert all(emb.shape == (1024,) for emb in embeddings)
    assert all(emb.dtype == np.float32 for emb in embeddings)
    
    # Test deterministic embedding (same text should give same embedding)
    embeddings2 = await plugin.embed_text(["hello world"])
    np.testing.assert_array_almost_equal(embeddings[0], embeddings2[0])


@pytest.mark.asyncio
async def test_upsert_memory():
    """Test memory upsertion functionality."""
    
    # Setup plugin with mocks
    plugin = SemanticMemoryPlugin()
    
    mock_event_bus = Mock()
    mock_event_bus.emit = AsyncMock()
    mock_store = Mock()
    mock_store.register_with_lineage = Mock()
    
    mock_collection = Mock()
    mock_collection.upsert = Mock()
    
    plugin._event_bus = mock_event_bus
    plugin._store = mock_store
    plugin._collection = mock_collection
    plugin._config = {}
    
    # Test upsert
    content = {"type": "fact", "data": "The sky is blue"}
    hierarchy_path = ["general", "facts", "nature"]
    owner = "test_user"
    
    memory_id = await plugin.upsert(content, hierarchy_path, owner)
    
    # Verify memory ID format
    assert memory_id.startswith("mem_")
    assert len(memory_id) == 12  # "mem_" + 8 hex chars
    
    # Verify store interactions
    mock_store.register_with_lineage.assert_called_once()
    mock_collection.upsert.assert_called_once()
    mock_event_bus.emit.assert_called_once()


@pytest.mark.asyncio
async def test_query_memory():
    """Test memory querying functionality."""
    
    # Setup plugin with mocks
    plugin = SemanticMemoryPlugin()
    
    mock_store = Mock()
    
    # Mock atom for returned results
    mock_atom = Mock()
    mock_atom.value = {"type": "fact", "data": "Test memory content"}
    mock_store.get.return_value = mock_atom
    mock_store.attention = AsyncMock(return_value=[("mem_12345678", 0.85)])
    
    plugin._store = mock_store
    plugin._config = {}
    
    # Test query
    results = await plugin.query("test query", top_k=3)
    
    # Verify results
    assert len(results) == 1
    assert results[0]["content"] == mock_atom.value
    assert results[0]["score"] == 0.85
    
    # Verify store interaction
    mock_store.attention.assert_called_once()
    mock_store.get.assert_called_once_with("mem_12345678")


@pytest.mark.asyncio
async def test_query_empty_store():
    """Test querying when no memories exist."""
    
    plugin = SemanticMemoryPlugin()
    
    mock_store = Mock()
    mock_store.attention = AsyncMock(return_value=[])
    plugin._store = mock_store
    plugin._config = {}
    
    results = await plugin.query("test query")
    
    assert results == []


@pytest.mark.asyncio
async def test_memory_hierarchy():
    """Test hierarchical memory organization."""
    
    plugin = SemanticMemoryPlugin()
    
    mock_event_bus = Mock()
    mock_event_bus.emit = AsyncMock()
    mock_store = Mock()
    mock_store.register_with_lineage = Mock()
    mock_collection = Mock()
    mock_collection.upsert = Mock()
    
    plugin._event_bus = mock_event_bus
    plugin._store = mock_store
    plugin._collection = mock_collection
    plugin._config = {}
    
    # Test different hierarchy levels
    hierarchies = [
        ["general", "facts"],
        ["personal", "preferences", "food"],
        ["skills", "programming", "python"]
    ]
    
    for i, hierarchy in enumerate(hierarchies):
        content = {"id": f"test_mem_{i}", "data": f"Test content {i}"}
        memory_id = await plugin.upsert(content, hierarchy, "test_user")
        
        # Verify the text representation includes hierarchy
        call_args = mock_collection.upsert.call_args_list[i]
        metadata = call_args[1]["metadatas"][0]
        assert metadata["hierarchy_path"] == "::".join(hierarchy)


@pytest.mark.asyncio 
async def test_configuration_handling():
    """Test plugin configuration handling."""
    
    plugin = SemanticMemoryPlugin()
    
    config = {
        "db_path": "/custom/path",
        "collection_name": "custom_collection",
        "gemini_api_key": "test_key"
    }
    
    await plugin.setup(Mock(), Mock(), config)
    
    assert plugin._config["db_path"] == "/custom/path"
    assert plugin._config["collection_name"] == "custom_collection"
    assert plugin._config["gemini_api_key"] == "test_key"
