from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from src.core.plugin_interface import PluginInterface


class OakCoordinator(PluginInterface):
    """Coordinates OaK components and (optionally) emits periodic deliberation ticks."""

    def __init__(self):
        super().__init__()
        self._task: Optional[asyncio.Task] = None

    @property
    def name(self) -> str:
        return "oak_coordinator"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
        self.interval_sec = self.get_config("interval_sec", 0.5)

    async def start(self) -> None:
        await super().start()
        # simple ticker; callers can disable by not calling start()
        async def _ticker():
            while self.is_running:
                await self.emit_event("deliberation_tick")
                await asyncio.sleep(self.interval_sec)
        self._task = self.add_task(_ticker())

    async def shutdown(self) -> None:
        # The base class stop() method handles task cancellation.
        # We just need to call it.
        await super().shutdown()
        self._task = None
