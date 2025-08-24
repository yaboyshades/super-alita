"""
Base interface for agent plugins.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class PluginInterface(ABC):
    """Base interface that all plugins must implement."""

    def __init__(self, config: Dict[str, Any]):
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
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information."""
