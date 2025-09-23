"""Enhanced assistant wiring for the unified intelligence orchestrator."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any

from src.unified_intelligence.orchestrator import (
    HardenedOrchestrator,
    Request,
    UnifiedAdvice,
)
from src.unified_intelligence.telemetry import TelemetryMiddleware
from src.unified_intelligence.validation_checklist import ValidationChecklist

logger = logging.getLogger(__name__)


class EnhancedAssistant:
    """Thin convenience layer over the hardened orchestrator."""

    def __init__(self) -> None:
        self.orchestrator = HardenedOrchestrator()
        self.validator = ValidationChecklist()
        self.telemetry = TelemetryMiddleware()

    async def analyze_request(self, user_query: str, context: dict[str, Any]) -> UnifiedAdvice:
        request = Request(
            request_id=context.get("request_id", f"req-{datetime.utcnow().timestamp():.0f}"),
            ts=datetime.utcnow().isoformat(),
            intent_text=user_query,
            code_refs=context.get("files", []),
            context=context,
        )
        advice = await self.orchestrator.orchestrate(request)
        advice.telemetry.setdefault("request_id", request.request_id)
        advice.telemetry_headers = self.telemetry.generate_headers(advice)  # type: ignore[attr-defined]
        return advice

    async def validate_changes(self, proposed_changes: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self.validator.validate_async(proposed_changes)
        except AttributeError:
            return self.validator.validate(proposed_changes)

    async def generate_response(self, advice: UnifiedAdvice) -> str:
        advice_dict = asdict(advice)
        parts = [f"## Decision: {advice.decision.upper()}"]
        fused = advice_dict.get("scores", {}).get("fused", 0.0)
        parts.append(f"**Fused Score:** {fused:.2f}")

        reasons = advice_dict.get("reasons", [])
        if reasons:
            parts.append("\n### Rationale")
            parts.extend(f"- {reason}" for reason in reasons)

        recommendations = advice_dict.get("recommendations", [])
        if recommendations:
            parts.append("\n### Recommended Actions")
            for item in recommendations:
                action = item.get("action") if isinstance(item, dict) else item
                if action:
                    parts.append(f"- {action}")

        telemetry = advice_dict.get("telemetry", {})
        request_id = telemetry.get("request_id", "unknown")
        parts.append("\n---")
        parts.append(f"*request-id: {request_id}*")
        return "\n".join(parts)

    async def process_user_request(self, user_query: str, context: dict[str, Any] | None = None) -> str:
        context = context or {}
        advice = await self.analyze_request(user_query, context)

        if context.get("proposed_changes"):
            validation = await self.validate_changes(context["proposed_changes"])
            advice.validation = validation  # type: ignore[attr-defined]

        return await self.generate_response(advice)


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def process_query_sync(user_query: str, context: dict[str, Any] | None = None) -> str:
    loop = _ensure_event_loop()
    if loop.is_running():
        return asyncio.run(process_user_request(user_query, context or {}))
    return loop.run_until_complete(process_user_request(user_query, context or {}))


async def process_user_request(user_query: str, context: dict[str, Any] | None = None) -> str:
    assistant = EnhancedAssistant()
    return await assistant.process_user_request(user_query, context)
