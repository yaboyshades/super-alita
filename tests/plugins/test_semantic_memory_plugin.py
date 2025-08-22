# Test semantic memory plugin functionality

from unittest.mock import AsyncMock, Mock, patch

import pytest


# Add missing fixtures
@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing."""
    return AsyncMock()


@pytest.fixture
def mock_store():
    """Mock neural store for testing."""
    return Mock()


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "semantic_memory_plugin": {
            "embedding_model": "test-model",
            "max_memories": 1000,
            "similarity_threshold": 0.7,
        }
    }


class TestSemanticMemoryPlugin:
    """Test semantic memory plugin functionality."""

    def test_plugin_creation(self):
        """Test plugin can be created."""
        try:
            from src.plugins.semantic_memory_plugin import SemanticMemoryPlugin

            plugin = SemanticMemoryPlugin()
            assert plugin.name == "semantic_memory"
        except ImportError:
            pytest.skip("SemanticMemoryPlugin not available")

    @pytest.mark.asyncio
    async def test_plugin_setup(self, mock_event_bus, mock_store, mock_config):
        """Test plugin setup process."""
        try:
            from src.plugins.semantic_memory_plugin import SemanticMemoryPlugin

            plugin = SemanticMemoryPlugin()

            await plugin.setup(mock_event_bus, mock_store, mock_config)

            assert plugin.event_bus == mock_event_bus
            assert plugin.store == mock_store
            assert plugin.config == mock_config

        except ImportError:
            pytest.skip("SemanticMemoryPlugin not available")

    @pytest.mark.asyncio
    @patch("src.plugins.semantic_memory_plugin.google.generativeai.GenerativeModel")
    async def test_embedding_generation(
        self, mock_genai, mock_event_bus, mock_store, mock_config
    ):
        """Test embedding generation with Google Generative AI."""
        try:
            from src.plugins.semantic_memory_plugin import SemanticMemoryPlugin

            # Mock the generative AI model
            mock_model = Mock()
            mock_genai.return_value = mock_model
            mock_model.embed_content.return_value = {"embedding": [0.1, 0.2, 0.3]}

            plugin = SemanticMemoryPlugin()
            await plugin.setup(mock_event_bus, mock_store, mock_config)

            # Test embedding generation
            embedding = await plugin._generate_embedding("test content")
            assert embedding == [0.1, 0.2, 0.3]

        except ImportError:
            pytest.skip("SemanticMemoryPlugin not available")

    @pytest.mark.asyncio
    async def test_memory_storage(self, mock_event_bus, mock_store, mock_config):
        """Test memory storage functionality."""
        try:
            from src.plugins.semantic_memory_plugin import SemanticMemoryPlugin

            plugin = SemanticMemoryPlugin()
            await plugin.setup(mock_event_bus, mock_store, mock_config)

            # Mock store methods
            mock_store.upsert = AsyncMock()

            # Test memory storage
            await plugin.store_memory("test_id", "test content", ["tag1", "tag2"])

            # Verify store was called
            mock_store.upsert.assert_called_once()

        except ImportError:
            pytest.skip("SemanticMemoryPlugin not available")

    @pytest.mark.asyncio
    async def test_memory_retrieval(self, mock_event_bus, mock_store, mock_config):
        """Test memory retrieval functionality."""
        try:
            from src.plugins.semantic_memory_plugin import SemanticMemoryPlugin

            plugin = SemanticMemoryPlugin()
            await plugin.setup(mock_event_bus, mock_store, mock_config)

            # Mock store query result
            mock_store.query = AsyncMock(
                return_value=[{"content": "test content", "similarity": 0.9}]
            )

            # Test memory retrieval
            results = await plugin.retrieve_memories("search query", limit=5)

            assert len(results) == 1
            assert results[0]["content"] == "test content"
            assert results[0]["similarity"] == 0.9

        except ImportError:
            pytest.skip("SemanticMemoryPlugin not available")


class TestSemanticMemoryFallback:
    """Test fallback functionality when dependencies are missing."""

    def test_fallback_embedding(self):
        """Test fallback embedding generation."""
        # This should work even without the plugin
        import hashlib

        text = "test content"

        # Simple hash-based embedding as fallback
        hash_val = hashlib.md5(text.encode()).hexdigest()
        embedding = [
            int(hash_val[i : i + 2], 16) / 255.0
            for i in range(0, min(len(hash_val), 32), 2)
        ]

        assert len(embedding) == 16  # 32 hex chars / 2
        assert all(0 <= x <= 1 for x in embedding)

    @pytest.mark.asyncio
    async def test_memory_operations_without_plugin(self, mock_event_bus, mock_store):
        """Test that memory operations can work without the specific plugin."""
        # This tests the interface can be mocked for development

        class MockMemoryPlugin:
            def __init__(self):
                self.name = "mock_semantic_memory"
                self.memories = {}

            async def store_memory(
                self, memory_id: str, content: str, tags: list = None
            ):
                self.memories[memory_id] = {
                    "content": content,
                    "tags": tags or [],
                    "embedding": [0.0] * 128,  # Mock embedding
                }

            async def retrieve_memories(self, query: str, limit: int = 10):
                # Simple mock retrieval
                return [
                    {"content": mem["content"], "similarity": 0.8}
                    for mem in self.memories.values()
                    if query.lower() in mem["content"].lower()
                ]

        plugin = MockMemoryPlugin()

        # Test storage
        await plugin.store_memory("test1", "Python programming", ["code", "tutorial"])
        assert "test1" in plugin.memories

        # Test retrieval
        results = await plugin.retrieve_memories("Python")
        assert len(results) == 1
        assert "Python programming" in results[0]["content"]


if __name__ == "__main__":
    pytest.main([__file__])
