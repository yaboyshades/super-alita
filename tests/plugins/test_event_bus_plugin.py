"""
Test the EventBus Plugin functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.plugins.event_bus_plugin import EventBusPlugin


@pytest.mark.asyncio
async def test_event_bus_plugin_lifecycle():
    """Test EventBus plugin setup, start, and shutdown."""
    
    # Create mock dependencies
    mock_event_bus = Mock()
    mock_event_bus.stop = AsyncMock()
    mock_store = Mock()
    config = {"test": "config"}
    
    # Create plugin
    plugin = EventBusPlugin()
    
    # Test name property
    assert plugin.name == "event_bus"
    
    # Test setup
    await plugin.setup(mock_event_bus, mock_store, config)
    assert plugin._bus == mock_event_bus
    
    # Test start
    await plugin.start()
    # Should not raise any errors
    
    # Test bus accessor
    assert plugin.bus == mock_event_bus
    
    # Test shutdown
    await plugin.shutdown()
    mock_event_bus.stop.assert_called_once()


@pytest.mark.asyncio 
async def test_event_bus_plugin_properties():
    """Test EventBus plugin properties and interface compliance."""
    
    plugin = EventBusPlugin()
    
    # Test required properties
    assert plugin.name == "event_bus"
    assert plugin.version == "1.0.0"  # Default from PluginInterface
    assert "EventBus plugin" in plugin.description or "Plugin: event_bus" in plugin.description
    assert plugin.dependencies == []  # Default from PluginInterface
    
    # Test bus accessor before setup (should raise AttributeError)
    with pytest.raises(AttributeError):
        _ = plugin.bus
