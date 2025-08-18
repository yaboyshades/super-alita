"""
Test the plugin interface functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock

from src.core.plugin_interface import PluginInterface, BasePlugin, register_plugin, get_plugin_class


class TestPlugin(PluginInterface):
    """Test plugin implementation."""
    
    @property
    def name(self) -> str:
        return "test_plugin"
    
    async def setup(self, event_bus, store, config):
        await super().setup(event_bus, store, config)
        self.setup_called = True
    
    async def start(self) -> None:
        await super().start()
        self.start_called = True
    
    async def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.mark.asyncio
async def test_plugin_lifecycle():
    """Test plugin lifecycle management."""
    
    plugin = TestPlugin()
    
    # Test initial state
    assert plugin.name == "test_plugin"
    assert not plugin.is_running
    
    # Mock dependencies
    event_bus = Mock()
    store = Mock()
    config = {"test": "value"}
    
    # Test setup
    await plugin.setup(event_bus, store, config)
    assert plugin.event_bus == event_bus
    assert plugin.store == store
    assert plugin.config == config
    assert plugin.setup_called
    
    # Test start
    await plugin.start()
    assert plugin.is_running
    assert plugin.start_called
    
    # Test stop
    await plugin.stop()
    assert not plugin.is_running
    assert plugin.shutdown_called


@pytest.mark.asyncio
async def test_base_plugin():
    """Test base plugin implementation."""
    
    plugin = BasePlugin("base_test", "Test base plugin")
    
    assert plugin.name == "base_test"
    assert plugin.description == "Test base plugin"
    
    # Test lifecycle
    event_bus = Mock()
    store = Mock()
    config = {}
    
    await plugin.setup(event_bus, store, config)
    await plugin.start()
    await plugin.stop()


def test_plugin_registration():
    """Test plugin registration system."""
    
    # Register plugin
    register_plugin(TestPlugin)
    
    # Retrieve plugin class
    plugin_class = get_plugin_class("test_plugin")
    assert plugin_class == TestPlugin
    
    # Test non-existent plugin
    assert get_plugin_class("non_existent") is None


@pytest.mark.asyncio
async def test_plugin_health_check():
    """Test plugin health check functionality."""
    
    plugin = TestPlugin()
    
    # Test health check before start
    health = await plugin.health_check()
    assert health["status"] == "stopped"
    assert health["plugin"] == "test_plugin"
    
    # Setup and start plugin
    await plugin.setup(Mock(), Mock(), {})
    await plugin.start()
    
    # Test health check after start
    health = await plugin.health_check()
    assert health["status"] == "healthy"


@pytest.mark.asyncio
async def test_plugin_task_management():
    """Test plugin background task management."""
    
    plugin = TestPlugin()
    await plugin.setup(Mock(), Mock(), {})
    
    # Add a test task
    async def test_task():
        await asyncio.sleep(0.1)
        return "completed"
    
    task = plugin.add_task(test_task())
    assert task in plugin._tasks
    
    # Start plugin
    await plugin.start()
    
    # Wait for task completion
    result = await task
    assert result == "completed"
    
    # Stop plugin (should clean up tasks)
    await plugin.stop()
    assert len(plugin._tasks) == 0


@pytest.mark.asyncio
async def test_plugin_configuration():
    """Test plugin configuration access."""
    
    plugin = TestPlugin()
    config = {
        "setting1": "value1",
        "setting2": 42,
        "nested": {"key": "nested_value"}
    }
    
    await plugin.setup(Mock(), Mock(), config)
    
    # Test configuration access
    assert plugin.get_config("setting1") == "value1"
    assert plugin.get_config("setting2") == 42
    assert plugin.get_config("nonexistent", "default") == "default"
    assert plugin.get_config("nested") == {"key": "nested_value"}


if __name__ == "__main__":
    pytest.main([__file__])
