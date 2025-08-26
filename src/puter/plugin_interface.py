"""
Base interface for agent plugins.
"""

from abc import ABC, abstractmethod
from typing import Any


class PluginInterface(ABC):
    """Base interface that all plugins must implement."""

    def __init__(self, config: dict[str, Any]):
        """Initialize plugin with configuration."""
        self.config = config
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the plugin."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up plugin resources."""

    @abstractmethod
    def get_plugin_info(self) -> dict[str, Any]:
        """Get plugin information."""
