"""
Detects when the agent encounters knowledge gaps and needs cortex assistance.
"""
from __future__ import annotations
from typing import Dict, Any, List

from src.core.plugin_interface import PluginInterface
from src.core.event_bus import EventBus
from src.core.events import create_event


class KnowledgeGapDetector(PluginInterface):
    """Detects when agent needs external assistance."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.confidence_threshold = 0.3  # Below this, seek help
        # Normalize patterns to lowercase; we'll lowercase incoming content
        self.gap_patterns = [
            "i don't know",
            "i'm not sure",
            "insufficient information",
            "need more data",
            "unclear how to proceed",
        ]

    @property
    def name(self) -> str:  # type: ignore[override]
        return "knowledge_gap_detector"

    async def setup(self, event_bus: Any, store: Any, config: Dict[str, Any]) -> None:  # type: ignore[override]
        self.event_bus = event_bus
        self.store = store
        self.config = config

    async def start(self) -> None:  # type: ignore[override]
        self.is_running = True

    async def check_reasoning_confidence(self, event: Dict[str, Any]):
        """Check if reasoning confidence is too low."""
        data = event.get("data", {}) if isinstance(event, dict) else {}
        confidence = float(data.get("confidence", 1.0))

        if confidence < self.confidence_threshold:
            await self.event_bus.publish(
                create_event(
                    "knowledge_gap",
                    event_version=1,
                    source_plugin=self.name,
                    gap_description=f"Low confidence reasoning: {confidence}",
                    context=data,
                    gap_type="low_confidence",
                )
            )

    async def check_navigation_success(self, event: Dict[str, Any]):
        """Check if navigation found useful paths."""
        data = event.get("data", {}) if isinstance(event, dict) else {}
        path_length = int(data.get("path_length", 0))

        if path_length <= 1:  # Only found starting atom
            await self.event_bus.publish(
                create_event(
                    "knowledge_gap",
                    event_version=1,
                    source_plugin=self.name,
                    gap_description="Navigation found no relevant connections",
                    context=data,
                    gap_type="isolated_knowledge",
                )
            )

    async def detect_uncertainty_patterns(self, event: Dict[str, Any]):
        """Detect uncertainty patterns in conversation."""
        data = event.get("data", {}) if isinstance(event, dict) else {}
        content = str(data.get("content", "")).lower()

        for pattern in self.gap_patterns:
            if pattern in content:
                await self.event_bus.publish(
                    create_event(
                        "knowledge_gap",
                        event_version=1,
                        source_plugin=self.name,
                        gap_description=f"Uncertainty pattern detected: {pattern}",
                        context=data,
                        gap_type="uncertainty_expression",
                    )
                )
                break
