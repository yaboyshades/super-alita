"""
Detects when the agent encounters knowledge gaps and needs cortex assistance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol
import uuid

from src.core.plugin_interface import PluginInterface
from src.core.event_bus import EventBus
from src.core.events import create_event
from src.core.utils import CooldownLRU


class CortexPolicy(Protocol):
    async def should_use_cortex(self, confidence: float, context: Dict[str, Any]) -> bool:
        ...


class KnowledgeGapDetector(PluginInterface):
    """Detects when agent needs external assistance."""

    def __init__(self, event_bus: EventBus, policy: CortexPolicy | None = None, *, cooldown_seconds: float = 300.0):
        self.event_bus = event_bus
        self.policy = policy
        self.confidence_threshold = 0.3
        self.max_hops = 3
        self.cooldown = CooldownLRU(ttl_seconds=cooldown_seconds)
        self.gap_patterns = [
            "i don't know",
            "i'm not sure",
            "insufficient information",
            "need more data",
            "unclear how to proceed",
        ]
        self.event_bus.subscribe("reasoning_complete", self.check_reasoning_confidence)
        self.event_bus.subscribe("navigation_complete", self.check_navigation_success)
        self.event_bus.subscribe("conversation_turn", self.detect_uncertainty_patterns)

    @property
    def name(self) -> str:  # type: ignore[override]
        return "knowledge_gap_detector"

    async def setup(self, event_bus: Any, store: Any, config: Dict[str, Any]) -> None:  # type: ignore[override]
        self.event_bus = event_bus
        self.store = store
        self.config = config

    async def start(self) -> None:  # type: ignore[override]
        self.is_running = True

    async def check_reasoning_confidence(self, event: Dict[str, Any]) -> None:
        data = event.get("data", {}) if isinstance(event, dict) else {}
        confidence = float(data.get("confidence", 1.0))
        if confidence < self.confidence_threshold:
            await self._maybe_publish_gap(
                gap_description=f"Low confidence reasoning: {confidence:.2f}",
                context=data,
                gap_type="low_confidence",
            )

    async def check_navigation_success(self, event: Dict[str, Any]) -> None:
        data = event.get("data", {}) if isinstance(event, dict) else {}
        path_length = int(data.get("path_length", 0))
        if path_length <= 1:
            await self._maybe_publish_gap(
                gap_description="Navigation found no relevant connections",
                context=data,
                gap_type="isolated_knowledge",
            )

    async def detect_uncertainty_patterns(self, event: Dict[str, Any]) -> None:
        data = event.get("data", {}) if isinstance(event, dict) else {}
        content = str(data.get("content", "")).lower()
        for pattern in self.gap_patterns:
            if pattern in content:
                await self._maybe_publish_gap(
                    gap_description=f"Uncertainty pattern detected: {pattern}",
                    context=data,
                    gap_type="uncertainty_expression",
                )
                break

    async def _maybe_publish_gap(self, *, gap_description: str, context: Dict[str, Any], gap_type: str) -> None:
        topic = context.get("topic_id") or context.get("prompt_hash") or context.get("goal") or gap_type
        if self.cooldown.hit(str(topic)):
            return
        hop_count = int(context.get("hop_count", 0))
        if hop_count >= self.max_hops:
            return
        if self.policy is not None:
            ok = await self.policy.should_use_cortex(
                confidence=float(context.get("confidence", 0.0)),
                context=context,
            )
            if not ok:
                return
        correlation_id = context.get("correlation_id") or str(uuid.uuid4())
        trace_id = context.get("trace_id") or correlation_id
        await self.event_bus.publish(
            create_event(
                "knowledge_gap",
                event_version=1,
                source_plugin=self.name,
                gap_description=gap_description,
                context={**context, "hop_count": hop_count + 1, "correlation_id": correlation_id, "trace_id": trace_id},
                gap_type=gap_type,
            )
        )
