#!/usr/bin/env python3
"""
Tests for Puter plugin integration with Super Alita's event-driven neural architecture.
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.events import create_event
from src.core.neural_atom import NeuralStore
from src.plugins.puter_plugin import PuterPlugin, PuterOperationAtom


class MockEventBus:
    """Mock event bus for testing."""
    
    def __init__(self):
        self.events = []
        self.subscribers = {}
    
    async def emit(self, event_type: str, **kwargs):
        event = create_event(event_type, **kwargs)
        self.events.append(event)
        return event
    
    async def subscribe(self, event_type: str, handler):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event):
        self.events.append(event)
    
    # Add the emit method that plugin_interface expects
    async def emit_event(self, event_type: str, **kwargs):
        return await self.emit(event_type, **kwargs)


class MockNeuralStore:
    """Mock neural store for testing."""
    
    def __init__(self):
        self.atoms = {}
        self.registered_atoms = []
    
    async def register(self, atom):
        self.registered_atoms.append(atom)
        if hasattr(atom, 'key'):
            self.atoms[atom.key] = atom


@pytest.fixture
def mock_event_bus():
    """Provide mock event bus for testing."""
    return MockEventBus()


@pytest.fixture
def mock_neural_store():
    """Provide mock neural store for testing."""
    return MockNeuralStore()


@pytest.fixture
def puter_config():
    """Provide test configuration for Puter plugin."""
    return {
        "puter_api_url": "https://test.puter.com",
        "puter_api_key": "test_key_123",
        "puter_workspace_id": "test_workspace",
    }


@pytest.fixture
async def puter_plugin(mock_event_bus, mock_neural_store, puter_config):
    """Provide configured Puter plugin for testing."""
    plugin = PuterPlugin()
    await plugin.setup(mock_event_bus, mock_neural_store, puter_config)
    await plugin.start()
    return plugin


class TestPuterOperationAtom:
    """Test PuterOperationAtom neural atom implementation."""
    
    def test_deterministic_uuid_generation(self):
        """Test that operation atoms generate deterministic UUIDs."""
        operation_data = {
            "operation": "read",
            "file_path": "/test/file.txt",
            "description": "Test file read",
        }
        
        # Create two atoms with same data
        atom1 = PuterOperationAtom("file_operation", operation_data)
        atom2 = PuterOperationAtom("file_operation", operation_data)
        
        # UUIDs should be identical
        assert atom1.get_deterministic_uuid() == atom2.get_deterministic_uuid()
        
        # Different data should produce different UUIDs
        different_data = operation_data.copy()
        different_data["file_path"] = "/test/different.txt"
        atom3 = PuterOperationAtom("file_operation", different_data)
        
        assert atom1.get_deterministic_uuid() != atom3.get_deterministic_uuid()
    
    def test_neural_atom_metadata(self):
        """Test that operation atoms have proper neural metadata."""
        operation_data = {
            "operation": "write",
            "file_path": "/test/output.txt",
            "description": "Test file write",
        }
        
        atom = PuterOperationAtom("file_operation", operation_data)
        
        # Check metadata
        assert atom.metadata.name.startswith("puter_file_operation_")
        assert atom.metadata.description == "Puter file_operation operation"
        assert "puter" in atom.metadata.tags
        assert "file_operation" in atom.metadata.tags
        assert "cloud" in atom.metadata.tags
        
        # Check capabilities
        expected_capabilities = ["cloud_storage", "process_execution", "file_io"]
        assert all(cap in atom.metadata.capabilities for cap in expected_capabilities)
    
    @pytest.mark.asyncio
    async def test_atom_execution(self):
        """Test that operation atoms can be executed."""
        operation_data = {
            "operation": "read",
            "file_path": "/test/file.txt",
            "description": "Test file read operation",
        }
        
        atom = PuterOperationAtom("file_operation", operation_data)
        result = await atom.execute()
        
        # Check execution result
        assert "content" in result
        assert result["memory_type"] == "textual"
        assert "metadata" in result
        assert result["metadata"]["name"] == atom.metadata.name


class TestPuterPlugin:
    """Test PuterPlugin implementation."""
    
    def test_plugin_name(self):
        """Test plugin name property."""
        plugin = PuterPlugin()
        assert plugin.name == "puter"
    
    @pytest.mark.asyncio
    async def test_plugin_setup(self, mock_event_bus, mock_neural_store, puter_config):
        """Test plugin setup with configuration."""
        plugin = PuterPlugin()
        await plugin.setup(mock_event_bus, mock_neural_store, puter_config)
        
        # Check configuration is stored
        assert plugin.puter_config["api_url"] == "https://test.puter.com"
        assert plugin.puter_config["api_key"] == "test_key_123"
        assert plugin.puter_config["workspace_id"] == "test_workspace"
        
        # Check operation history is initialized
        assert isinstance(plugin.operation_history, list)
        assert len(plugin.operation_history) == 0
    
    @pytest.mark.asyncio
    async def test_plugin_start_subscribes_to_events(self, puter_plugin, mock_event_bus):
        """Test that plugin subscribes to correct events on start."""
        # Check event subscriptions
        expected_events = [
            "puter_file_operation",
            "puter_process_execution", 
            "puter_workspace_sync",
            "tool_call",
        ]
        
        for event_type in expected_events:
            assert event_type in mock_event_bus.subscribers
            assert len(mock_event_bus.subscribers[event_type]) > 0
    
    @pytest.mark.asyncio
    async def test_plugin_shutdown_stores_atoms(self, puter_plugin, mock_neural_store):
        """Test that plugin stores operation history on shutdown."""
        # Add some operations to history
        operation_data = {"operation": "test", "description": "Test operation"}
        atom = PuterOperationAtom("test_operation", operation_data)
        puter_plugin.operation_history.append(atom)
        
        # Shutdown plugin
        await puter_plugin.shutdown()
        
        # Check atoms were registered
        assert len(mock_neural_store.registered_atoms) == 1
        assert mock_neural_store.registered_atoms[0] == atom
    
    @pytest.mark.asyncio
    async def test_file_operation_handling(self, puter_plugin, mock_event_bus):
        """Test file operation event handling."""
        # Create file operation event
        event = create_event(
            "puter_file_operation",
            source_plugin="test",
            conversation_id="test_conv_123",
        )
        event.metadata = {
            "operation": "read",
            "file_path": "/test/file.txt",
            "content": "",
        }
        
        # Handle the event
        await puter_plugin._handle_file_operation(event)
        
        # Check that operation was recorded
        assert len(puter_plugin.operation_history) == 1
        atom = puter_plugin.operation_history[0]
        assert atom.operation_type == "file_operation"
        assert atom.operation_data["operation"] == "read"
        assert atom.operation_data["file_path"] == "/test/file.txt"
        
        # Check that completion event was emitted
        completion_events = [
            e for e in mock_event_bus.events 
            if getattr(e, 'event_type', None) == "puter_operation_completed"
        ]
        assert len(completion_events) == 1
        
        completion_event = completion_events[0]
        assert completion_event.operation_type == "file_operation"
        assert completion_event.file_path == "/test/file.txt"
        assert completion_event.neural_atom_id == atom.get_deterministic_uuid()
    
    @pytest.mark.asyncio
    async def test_process_execution_handling(self, puter_plugin, mock_event_bus):
        """Test process execution event handling."""
        # Create process execution event
        event = create_event(
            "puter_process_execution",
            source_plugin="test",
            conversation_id="test_conv_123",
        )
        event.metadata = {
            "command": "python",
            "args": ["--version"],
            "working_dir": "/workspace",
        }
        
        # Handle the event
        await puter_plugin._handle_process_execution(event)
        
        # Check that operation was recorded
        assert len(puter_plugin.operation_history) == 1
        atom = puter_plugin.operation_history[0]
        assert atom.operation_type == "process_execution"
        assert atom.operation_data["command"] == "python"
        assert atom.operation_data["args"] == ["--version"]
        
        # Check that completion event was emitted
        completion_events = [
            e for e in mock_event_bus.events 
            if getattr(e, 'event_type', None) == "puter_operation_completed"
        ]
        assert len(completion_events) == 1
        
        completion_event = completion_events[0]
        assert completion_event.operation_type == "process_execution"
        assert completion_event.command == "python"
        assert completion_event.neural_atom_id == atom.get_deterministic_uuid()
    
    @pytest.mark.asyncio
    async def test_workspace_sync_handling(self, puter_plugin, mock_event_bus):
        """Test workspace sync event handling."""
        # Create workspace sync event
        event = create_event(
            "puter_workspace_sync",
            source_plugin="test",
            conversation_id="test_conv_123",
        )
        event.metadata = {
            "sync_type": "bidirectional",
            "local_path": "/local/workspace",
            "remote_path": "/remote/workspace",
        }
        
        # Handle the event
        await puter_plugin._handle_workspace_sync(event)
        
        # Check that operation was recorded
        assert len(puter_plugin.operation_history) == 1
        atom = puter_plugin.operation_history[0]
        assert atom.operation_type == "workspace_sync"
        assert atom.operation_data["sync_type"] == "bidirectional"
        assert atom.operation_data["local_path"] == "/local/workspace"
        
        # Check that completion event was emitted
        completion_events = [
            e for e in mock_event_bus.events 
            if getattr(e, 'event_type', None) == "puter_operation_completed"
        ]
        assert len(completion_events) == 1
        
        completion_event = completion_events[0]
        assert completion_event.operation_type == "workspace_sync"
        assert completion_event.sync_type == "bidirectional"
        assert completion_event.neural_atom_id == atom.get_deterministic_uuid()
    
    @pytest.mark.asyncio
    async def test_tool_call_handling(self, puter_plugin, mock_event_bus):
        """Test tool call event handling for Puter tools."""
        # Create tool call event for puter file read
        event = create_event(
            "tool_call",
            source_plugin="test",
            conversation_id="test_conv_123",
            session_id="test_session_123",
            tool_name="puter_file_read",
            tool_call_id="call_123456",
            parameters={"file_path": "/test/file.txt"},
        )
        
        # Handle the event
        await puter_plugin._handle_tool_call(event)
        
        # Check that file operation was triggered
        assert len(puter_plugin.operation_history) == 1
        atom = puter_plugin.operation_history[0]
        assert atom.operation_type == "file_operation"
        assert atom.operation_data["operation"] == "read"
    
    @pytest.mark.asyncio
    async def test_error_handling_file_operation(self, puter_plugin, mock_event_bus):
        """Test error handling in file operations."""
        # Create malformed event
        event = create_event(
            "puter_file_operation",
            source_plugin="test",
            conversation_id="test_conv_123",
        )
        # Missing metadata to trigger error
        
        # Handle the event (should not raise exception)
        await puter_plugin._handle_file_operation(event)
        
        # Check that error event was emitted
        error_events = [
            e for e in mock_event_bus.events 
            if getattr(e, 'event_type', None) == "puter_operation_failed"
        ]
        assert len(error_events) == 1
        
        error_event = error_events[0]
        assert error_event.operation_type == "file_operation"
        assert "missing" in error_event.error.lower()
    
    @pytest.mark.asyncio
    async def test_api_simulation_methods(self, puter_plugin):
        """Test API simulation methods."""
        # Test file operation simulation
        result = await puter_plugin._simulate_puter_file_operation("read", "/test/file.txt")
        assert result["success"] is True
        assert "content" in result
        
        # Test process execution simulation
        result = await puter_plugin._simulate_puter_process_execution("python", ["--version"], "/workspace")
        assert result["success"] is True
        assert "stdout" in result
        assert result["exit_code"] == 0
        
        # Test workspace sync simulation
        result = await puter_plugin._simulate_puter_workspace_sync("upload", "/local", "/remote")
        assert result["success"] is True
        assert result["files_synced"] == 42
    
    def test_get_capabilities(self, puter_plugin):
        """Test plugin capabilities reporting."""
        capabilities = puter_plugin.get_capabilities()
        
        expected_capabilities = {
            "file_operations": ["read", "write", "delete", "list"],
            "process_execution": True,
            "workspace_sync": True,
            "cloud_storage": True,
            "neural_atom_tracking": True,
            "deterministic_uuids": True,
        }
        
        assert capabilities == expected_capabilities
    
    def test_get_operation_history(self, puter_plugin):
        """Test operation history retrieval."""
        # Add some operations
        operation_data1 = {"operation": "read", "file_path": "/file1.txt"}
        operation_data2 = {"operation": "write", "file_path": "/file2.txt"}
        
        atom1 = PuterOperationAtom("file_operation", operation_data1)
        atom2 = PuterOperationAtom("file_operation", operation_data2)
        
        puter_plugin.operation_history.extend([atom1, atom2])
        
        # Get history
        history = puter_plugin.get_operation_history()
        
        # Should return a copy
        assert len(history) == 2
        assert history[0] == atom1
        assert history[1] == atom2
        assert history is not puter_plugin.operation_history  # Should be a copy


class TestPuterEventIntegration:
    """Test event-driven integration patterns."""
    
    @pytest.mark.asyncio
    async def test_event_creation_with_keyword_args(self, mock_event_bus):
        """Test that events are created with keyword args as required."""
        # Create event using create_event factory
        event = create_event(
            "puter_operation_completed",
            operation_type="file_operation",
            file_path="/test/file.txt",
            success=True,
            neural_atom_id="test_uuid_123",
            timestamp=datetime.now(timezone.utc),
            source_plugin="puter",
            conversation_id="test_conv",
        )
        
        # Check event properties
        assert event.event_type == "puter_operation_completed"
        assert hasattr(event, 'source_plugin')
        assert hasattr(event, 'conversation_id')
        assert hasattr(event, 'timestamp')
    
    @pytest.mark.asyncio
    async def test_timezone_aware_timestamps(self, puter_plugin, mock_event_bus):
        """Test that all timestamps are timezone-aware."""
        # Trigger a file operation
        event = create_event(
            "puter_file_operation",
            source_plugin="test",
            conversation_id="test_conv_123",
        )
        event.metadata = {
            "operation": "read",
            "file_path": "/test/file.txt",
        }
        
        await puter_plugin._handle_file_operation(event)
        
        # Check that emitted event has timezone-aware timestamp
        completion_events = [
            e for e in mock_event_bus.events 
            if getattr(e, 'event_type', None) == "puter_operation_completed"
        ]
        assert len(completion_events) == 1
        
        timestamp = completion_events[0].timestamp
        assert timestamp.tzinfo is not None  # Should be timezone-aware
    
    @pytest.mark.asyncio
    async def test_neural_atom_deterministic_uuids(self, puter_plugin):
        """Test that neural atoms generate deterministic UUIDs."""
        # Create two identical operations
        operation_data = {
            "operation": "read",
            "file_path": "/test/file.txt",
            "description": "Test operation",
        }
        
        atom1 = PuterOperationAtom("file_operation", operation_data)
        atom2 = PuterOperationAtom("file_operation", operation_data)
        
        # UUIDs should be deterministic
        assert atom1.get_deterministic_uuid() == atom2.get_deterministic_uuid()
        
        # Different operations should have different UUIDs
        different_data = operation_data.copy()
        different_data["file_path"] = "/different/file.txt"
        atom3 = PuterOperationAtom("file_operation", different_data)
        
        assert atom1.get_deterministic_uuid() != atom3.get_deterministic_uuid()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])