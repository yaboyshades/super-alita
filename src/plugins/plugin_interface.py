"""
Plugin Interface for Super Alita

Base interface that all plugins must implement for integration with the REUG v9.0
cognitive architecture.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol


class EventBus(Protocol):
    """Protocol for event bus interface"""

    async def emit(self, event_type: str, data: dict[str, Any]) -> None: ...
    async def subscribe(self, event_type: str, handler: Any) -> None: ...


class PluginInterface(ABC):
    """
    Abstract base class for all Super Alita plugins

    All plugins must implement these core methods to integrate
    with the REUG v9.0 cognitive architecture.
    """

    def __init__(self, name: str):
        self.name = name
        self.is_enabled = True
        self.logger = logging.getLogger(f"plugin.{name}")

    @abstractmethod
    async def initialize(self, event_bus: EventBus, **kwargs: Any) -> bool:
        """
        Initialize the plugin with required dependencies

        Args:
            event_bus: The main event bus for communication
            **kwargs: Additional initialization parameters

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def process_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """
        Process an event and return result if applicable

        Args:
            event: Event data dictionary

        Returns:
            dict | None: Processing result or None if no action taken
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up plugin resources during shutdown
        """
        pass

    def enable(self) -> None:
        """Enable the plugin"""
        self.is_enabled = True
        self.logger.info(f"Plugin {self.name} enabled")

    def disable(self) -> None:
        """Disable the plugin"""
        self.is_enabled = False
        self.logger.info(f"Plugin {self.name} disabled")

    def get_capabilities(self) -> list[str]:
        """
        Get list of capabilities this plugin provides

        Returns:
            list[str]: List of capability names
        """
        return []

    def get_status(self) -> dict[str, Any]:
        """
        Get current plugin status and metrics

        Returns:
            dict: Status information
        """
        return {
            "name": self.name,
            "enabled": self.is_enabled,
            "capabilities": self.get_capabilities(),
        }


class BasePlugin(PluginInterface):
    """
    Concrete base implementation with common functionality

    Plugins can inherit from this class for standard behavior
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.event_bus: EventBus | None = None
        self.processed_events = 0
        self.errors_count = 0

    async def initialize(self, event_bus: EventBus, **kwargs: Any) -> bool:
        """Default initialization"""
        self.event_bus = event_bus
        self.logger.info(f"Plugin {self.name} initialized")
        return True

    async def process_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Default event processing - override in subclasses"""
        if not self.is_enabled:
            return None

        try:
            self.processed_events += 1
            return await self._handle_event(event)
        except Exception as e:
            self.errors_count += 1
            self.logger.error(f"Error processing event in {self.name}: {e}")
            return None

    async def _handle_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Override this method to handle specific events"""
        return None

    async def cleanup(self) -> None:
        """Default cleanup"""
        self.logger.info(f"Plugin {self.name} cleaned up")

    def get_status(self) -> dict[str, Any]:
        """Extended status with metrics"""
        status = super().get_status()
        status.update(
            {
                "processed_events": self.processed_events,
                "errors_count": self.errors_count,
            }
        )
        return status


# Plugin registry for managing multiple plugins
class PluginRegistry:
    """Registry for managing plugin lifecycle"""

    def __init__(self):
        self.plugins: dict[str, PluginInterface] = {}
        self.logger = logging.getLogger("plugin_registry")

    def register(self, plugin: PluginInterface) -> bool:
        """Register a plugin"""
        if plugin.name in self.plugins:
            self.logger.warning(f"Plugin {plugin.name} already registered")
            return False

        self.plugins[plugin.name] = plugin
        self.logger.info(f"Registered plugin: {plugin.name}")
        return True

    def unregister(self, name: str) -> bool:
        """Unregister a plugin"""
        if name not in self.plugins:
            return False

        del self.plugins[name]
        self.logger.info(f"Unregistered plugin: {name}")
        return True

    def get_plugin(self, name: str) -> PluginInterface | None:
        """Get a plugin by name"""
        return self.plugins.get(name)

    def list_plugins(self) -> list[str]:
        """List all registered plugin names"""
        return list(self.plugins.keys())

    async def initialize_all(
        self, event_bus: EventBus, **kwargs: Any
    ) -> dict[str, bool]:
        """Initialize all registered plugins"""
        results: dict[str, bool] = {}
        for name, plugin in self.plugins.items():
            try:
                results[name] = await plugin.initialize(event_bus, **kwargs)
            except Exception as e:
                self.logger.error(f"Failed to initialize plugin {name}: {e}")
                results[name] = False
        return results

    async def cleanup_all(self) -> None:
        """Cleanup all registered plugins"""
        for plugin in self.plugins.values():
            try:
                await plugin.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up plugin {plugin.name}: {e}")

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all plugins"""
        return {name: plugin.get_status() for name, plugin in self.plugins.items()}


# Example plugin implementation
class EchoPlugin(BasePlugin):
    """Example plugin that echoes events"""

    def __init__(self):
        super().__init__("echo_plugin")

    async def _handle_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Echo the event back"""
        return {"type": "echo_response", "original_event": event, "plugin": self.name}

    def get_capabilities(self) -> list[str]:
        return ["echo", "debug"]


if __name__ == "__main__":
    # Example usage
    async def example():
        registry = PluginRegistry()
        echo_plugin = EchoPlugin()

        registry.register(echo_plugin)

        # Mock event bus
        class MockEventBus:
            async def emit(self, event_type: str, data: dict[str, Any]) -> None:
                pass

            async def subscribe(self, event_type: str, handler: Any) -> None:
                pass

        event_bus = MockEventBus()
        await registry.initialize_all(event_bus)

        # Test event processing
        test_event = {"type": "test", "data": "hello"}
        result = await echo_plugin.process_event(test_event)
        print(f"Plugin result: {result}")

        await registry.cleanup_all()

    asyncio.run(example())
