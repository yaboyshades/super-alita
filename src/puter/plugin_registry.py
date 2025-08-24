"""
Plugin registry for managing agent plugins.
"""

import logging
from typing import Any, Dict, Optional, Type

from .plugin_interface import PluginInterface
from .puter_plugin import PuterPlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing and initializing plugins."""

    def __init__(self) -> None:
        self.plugins: Dict[str, PluginInterface] = {}
        self.plugin_classes: Dict[str, Type[PluginInterface]] = {
            "puter": PuterPlugin,
        }

    async def initialize_plugin(self, plugin_name: str, config: Dict[str, Any]) -> None:
        if plugin_name in self.plugins:
            logger.warning("Plugin %s already initialized", plugin_name)
            return
        if plugin_name not in self.plugin_classes:
            raise ValueError(f"Unknown plugin: {plugin_name}")
        plugin_class = self.plugin_classes[plugin_name]
        plugin = plugin_class(config)
        await plugin.initialize()
        self.plugins[plugin_name] = plugin
        logger.info("Successfully initialized plugin: %s", plugin_name)

    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        return self.plugins.get(plugin_name)

    async def cleanup_all(self) -> None:
        for name, plugin in list(self.plugins.items()):
            try:
                await plugin.cleanup()
                logger.info("Cleaned up plugin: %s", name)
            except Exception as exc:  # pragma: no cover - cleanup error
                logger.error("Error cleaning up plugin %s: %s", name, exc)
        self.plugins.clear()

    def list_available_plugins(self) -> list[str]:
        return list(self.plugin_classes.keys())

    def list_initialized_plugins(self) -> list[str]:
        return list(self.plugins.keys())
