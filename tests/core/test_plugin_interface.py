"""Plugin Interface tests (conflict markers cleaned)."""

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

import pytest
from src.core.plugin_interface import PluginInterface


class MockPlugin(PluginInterface):
    """Test implementation of PluginInterface."""

    def __init__(self):
        super().__init__()
        self.setup_called = False
        self.start_called = False
        self.shutdown_called = False
        self.test_events = []

    @property
    def name(self) -> str:  # type: ignore[override]
        return "test_plugin"

    async def setup(self, event_bus, store, config: Dict[str, Any]):
        await super().setup(event_bus, store, config)
        self.setup_called = True

    async def start(self):
        await super().start()
        self.start_called = True
        await self.subscribe("test_event", self._handle_test_event)

    async def shutdown(self):
        self.shutdown_called = True

    async def _handle_test_event(self, event):
        self.test_events.append(event)


class TestPluginInterface:
    def test_plugin_creation(self):
        plugin = MockPlugin()
        assert plugin.name == "test_plugin"
        assert not plugin.setup_called
        assert not plugin.start_called
        assert not plugin.shutdown_called
        assert plugin.test_events == []

    @pytest.mark.asyncio
    async def test_plugin_lifecycle(self):
        plugin = MockPlugin()
        mock_event_bus = AsyncMock()
        mock_store = Mock()
        mock_config = {"test_setting": "value"}
        await plugin.setup(mock_event_bus, mock_store, mock_config)
        assert plugin.setup_called
        await plugin.start()
        assert plugin.start_called
        await plugin.shutdown()
        assert plugin.shutdown_called

    @pytest.mark.asyncio
    async def test_event_subscription(self):
        plugin = MockPlugin()
        mock_event_bus = AsyncMock()
        mock_store = Mock()
        await plugin.setup(mock_event_bus, mock_store, {})
        await plugin.start()
        mock_event_bus.subscribe.assert_called_with(
            "test_event", plugin._handle_test_event
        )

    @pytest.mark.asyncio
    async def test_task_management(self):
        plugin = MockPlugin()
        mock_event_bus = AsyncMock()
        mock_store = Mock()
        await plugin.setup(mock_event_bus, mock_store, {})

        async def test_task():
            await asyncio.sleep(0.01)
            return "completed"

        plugin.add_task(test_task())
        assert len(plugin._tasks) == 1
        await plugin.shutdown()
        assert plugin.shutdown_called

    def test_plugin_running_state(self):
        plugin = MockPlugin()
        assert not plugin.is_running
        plugin._is_running = True
        assert plugin.is_running
        plugin._is_running = False
        assert not plugin.is_running

    @pytest.mark.asyncio
    async def test_error_handling_in_lifecycle(self):
        class ErrorPlugin(PluginInterface):
            @property
            def name(self) -> str:
                return "error_plugin"

            async def setup(self, event_bus, store, config):
                await super().setup(event_bus, store, config)
                raise ValueError("Setup error")

            async def start(self):
                pass

            async def shutdown(self):
                pass

        plugin = ErrorPlugin()
        with pytest.raises(ValueError):
            await plugin.setup(Mock(), Mock(), {})

    @pytest.mark.asyncio
    async def test_configuration_handling(self):
        plugin = MockPlugin()
        config = {
            "test_plugin": {
                "setting1": "value1",
                "setting2": 42,
                "nested": {"key": "nested_value"},
            }
        }
        await plugin.setup(Mock(), Mock(), config)
        assert plugin.config == config
        plugin_config = plugin.config.get(plugin.name, {})
        assert plugin_config["setting1"] == "value1"
        assert plugin_config["setting2"] == 42
        assert plugin_config["nested"]["key"] == "nested_value"


def test_plugin_interface_abc():
    with pytest.raises(TypeError):
        PluginInterface()  # type: ignore

    class IncompletePlugin(PluginInterface):
        pass

    with pytest.raises(TypeError):
        IncompletePlugin()


@pytest.mark.asyncio
async def test_plugin_integration():
    plugin1 = MockPlugin()
    plugin2 = MockPlugin()
    mock_event_bus = AsyncMock()
    mock_store = Mock()
    await plugin1.setup(mock_event_bus, mock_store, {"plugin1_config": True})
    await plugin2.setup(mock_event_bus, mock_store, {"plugin2_config": True})
    await plugin1.start()
    await plugin2.start()
    assert plugin1.event_bus == plugin2.event_bus
    assert plugin1.store == plugin2.store
    assert plugin1.config != plugin2.config
    await plugin1.shutdown()
    await plugin2.shutdown()
    assert plugin1.shutdown_called and plugin2.shutdown_called
