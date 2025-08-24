#!/usr/bin/env python3
"""
Test the full integration of Puter plugin with Super Alita's unified system.
"""

import asyncio
import logging
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

# Test integration with main unified system
def test_puter_plugin_in_unified_system():
    """Test that Puter plugin can be loaded in the unified system."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
    
    from main_unified import _load_unified_plugins, AVAILABLE_PLUGINS
    
    # Load plugins
    _load_unified_plugins()
    
    # Check Puter plugin is available
    assert 'puter' in AVAILABLE_PLUGINS
    
    plugin_class = AVAILABLE_PLUGINS['puter']
    assert plugin_class.__name__ == "PuterPlugin"
    
    # Test instantiation
    plugin = plugin_class()
    assert plugin.name == "puter"


@pytest.mark.asyncio
async def test_puter_plugin_initialization_with_env():
    """Test Puter plugin initialization with environment configuration."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
    
    # Set environment variables
    os.environ["PUTER_BASE_URL"] = "https://test.puter.com"
    os.environ["PUTER_API_KEY"] = "test_key_123"
    os.environ["PUTER_WORKSPACE_ID"] = "test_workspace"
    
    try:
        from plugins.puter_plugin import PuterPlugin
        from core.global_workspace import GlobalWorkspace
        from core.neural_atom import NeuralStore
        
        # Create mocks
        workspace = MagicMock()
        workspace.subscribe = AsyncMock()
        store = MagicMock()
        
        # Initialize plugin
        plugin = PuterPlugin()
        
        # Test configuration loading from environment
        config = {
            "enabled": True,
            "puter_base_url": "https://default.puter.com",  # Should be overridden
            "puter_api_key": "default_key",  # Should be overridden
            "puter_workspace_id": "default",  # Should be overridden
        }
        
        await plugin.setup(workspace, store, config)
        
        # Verify configuration was loaded from environment
        assert plugin.puter_config["api_url"] == "https://test.puter.com"
        assert plugin.puter_config["api_key"] == "test_key_123"
        assert plugin.puter_config["workspace_id"] == "test_workspace"
        
        # Test startup
        await plugin.start()
        
        # Verify subscriptions were made
        assert workspace.subscribe.call_count >= 4  # Should subscribe to multiple events
        
        # Test shutdown
        await plugin.shutdown()
        
    finally:
        # Clean up environment
        for key in ["PUTER_BASE_URL", "PUTER_API_KEY", "PUTER_WORKSPACE_ID"]:
            os.environ.pop(key, None)


@pytest.mark.asyncio 
async def test_end_to_end_puter_workflow():
    """Test end-to-end workflow with Puter plugin."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
    
    from plugins.puter_plugin import PuterPlugin
    from core.events import create_event
    
    # Mock event bus that captures events
    class MockEventBus:
        def __init__(self):
            self.events = []
        
        async def emit(self, event_type, **kwargs):
            from core.events import create_event
            event = create_event(event_type, **kwargs)
            self.events.append(event)
            return event
        
        async def publish(self, event):
            self.events.append(event)
        
        async def subscribe(self, event_type, handler):
            pass
    
    # Set up plugin
    plugin = PuterPlugin()
    mock_bus = MockEventBus()
    mock_store = MagicMock()
    
    config = {
        "puter_base_url": "https://test.puter.com",
        "puter_api_key": "test_key",
        "puter_workspace_id": "test",
    }
    
    await plugin.setup(mock_bus, mock_store, config)
    await plugin.start()
    
    # Test file operation workflow
    file_event = create_event(
        "puter_file_operation",
        source_plugin="test",
        conversation_id="test_conv",
    )
    file_event.metadata = {
        "operation": "write",
        "file_path": "/test/file.txt",
        "content": "Hello Puter World!",
    }
    
    await plugin._handle_file_operation(file_event)
    
    # Verify operation was recorded and event emitted
    assert len(plugin.operation_history) == 1
    assert len(mock_bus.events) == 1
    
    completion_event = mock_bus.events[0]
    assert completion_event.event_type == "puter_operation_completed"
    assert completion_event.operation_type == "file_operation"
    assert completion_event.file_path == "/test/file.txt"
    
    # Verify neural atom was created
    atom = plugin.operation_history[0]
    assert atom.operation_type == "file_operation"
    assert atom.operation_data["file_path"] == "/test/file.txt"
    assert atom.get_deterministic_uuid() is not None
    
    await plugin.shutdown()


if __name__ == "__main__":
    # Run simple integration tests
    test_puter_plugin_in_unified_system()
    print("✅ Plugin integration test passed")
    
    # Run async tests
    asyncio.run(test_puter_plugin_initialization_with_env())
    print("✅ Environment configuration test passed")
    
    asyncio.run(test_end_to_end_puter_workflow())
    print("✅ End-to-end workflow test passed")
    
    print("🎉 All integration tests passed!")