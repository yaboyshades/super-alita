from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.core.plugin_interface import PluginInterface
from src.pipelines.autogen_pipeline import autogen_any


class AutogenCreatorPlugin(PluginInterface):
    """
    Listens for capability gap signals and triggers generic autogen pipeline.
    Emits telemetry suitable for OaK planning and bandit reward tracking.
    """

    def __init__(self) -> None:
        super().__init__()
        self.topics = [
            "capability_gap_detected",  # publish when a task lacks an ability
            "atom_gap_request",  # existing auto tools signal
            "knowledge_gap_detected",  # from knowledge gap detector
        ]

    @property
    def name(self) -> str:
        return "AutogenCreatorPlugin"

    async def setup(
        self,
        event_bus=None,
        store=None,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.store = store
        self.config = cfg or {}
        
        # Subscribe to gap events
        if self.event_bus:
            for topic in self.topics:
                await self.event_bus.subscribe(topic, self._handle_gap)

    async def start(self) -> None:
        self._is_running = True

    async def stop(self) -> None:
        self._is_running = False

    async def _handle_gap(self, event: dict[str, Any]) -> None:
        """Handle gap events by invoking autogen pipeline."""
        if not self._is_running:
            return
            
        # Extract description from various event formats
        data = event.get("data", {}) if isinstance(event, dict) else {}
        
        # Try different attribute access patterns
        if hasattr(event, 'payload') and not isinstance(event, dict):
            data = getattr(event, 'payload', {})
        elif hasattr(event, 'data') and not isinstance(event, dict):
            data = getattr(event, 'data', {})
            
        desc = str(
            data.get("description")
            or data.get("task") 
            or data.get("message")
            or ""
        ).strip()
        
        if not desc:
            return

        # Run autogen in thread to avoid blocking event loop
        loop = asyncio.get_event_loop()

        def _run():
            return autogen_any(description=desc, event_bus=self.event_bus)

        # Execute in thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(executor, _run)