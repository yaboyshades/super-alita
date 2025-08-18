# src/plugins/event_bus_plugin.py
"""
Plugin wrapper for the central asynchronous EventBus.

This plugin exposes the singleton EventBus to the rest of the system,
allowing all plugins to publish and subscribe to strongly-typed events
without direct coupling to the bus implementation.
"""

import logging
from typing import Any, Dict
from src.core.plugin_interface import PluginInterface
from src.core.event_bus import EventBus  # we will create this next

logger = logging.getLogger(__name__)


class EventBusPlugin(PluginInterface):
    """
    Lightweight plugin that simply exposes the singleton EventBus.

    Lifecycle:
      setup   → store reference to the bus
      start   → no-op (bus is passive)
      shutdown → graceful shutdown of the bus
    """

    @property
    def name(self) -> str:
        return "event_bus"

    async def setup(self, event_bus: EventBus, store, config: Dict[str, Any]) -> None:
        # The bus is already a singleton; just expose it.
        self._bus = event_bus

    async def start(self) -> None:
        logger.info("EventBus plugin started (passive component).")

    async def shutdown(self) -> None:
        logger.info("Shutting down EventBus plugin...")
        await self._bus.stop()

    # Public accessor for other plugins
    @property
    def bus(self) -> EventBus:
        return self._bus
