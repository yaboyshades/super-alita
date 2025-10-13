"""ACE evolution helpers for the runtime learning stack.

This module intentionally keeps the ACE evolution surface area small so
that the runtime loop can rely on stable, well-typed behaviours while the
full ACE subsystem matures. The helpers here prioritise constitutional
safety and deterministic outputs so downstream consumers can trust the
learning signals being emitted.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable
from uuid import uuid4

from src.governance import ConstitutionalReasoner

logger = logging.getLogger(__name__)


def _coerce_patterns(patterns: Iterable[Any]) -> list[dict[str, Any]]:
    """Normalise raw pattern payloads into dictionaries."""

    normalised: list[dict[str, Any]] = []
    for item in patterns:
        if isinstance(item, dict):
            normalised.append(dict(item))
        else:
            normalised.append({"value": item})
    return normalised


def _coerce_validation(feedback: Any) -> dict[str, Any]:
    if isinstance(feedback, dict):
        return dict(feedback)
    return {"raw": feedback}


@dataclass(slots=True)
class StoredContext:
    """Structured record describing a persisted learning context."""

    context: dict[str, Any]
    stored_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    record_id: str = field(default_factory=lambda: str(uuid4()))

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "id": self.record_id,
            "context": self.context,
            "stored_at": self.stored_at.isoformat(),
        }
        return payload


class LearningMemoryStack:
    """In-memory memory stack that emits telemetry when contexts persist."""

    def __init__(self, event_bus: Any | None = None) -> None:
        self._event_bus = event_bus
        self._records: list[StoredContext] = []
        self._lock = asyncio.Lock()

    async def store_context(self, context: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(context, dict):
            raise ValueError("context must be a dictionary")
        record = StoredContext(context=dict(context))
        async with self._lock:
            self._records.append(record)
        await self._emit_event(record)
        return record.to_payload()

    async def _emit_event(self, record: StoredContext) -> None:
        if not self._event_bus:
            return
        payload = {
            "type": "MemoryContextStored",
            "event_type": "MemoryContextStored",
            "record": record.to_payload(),
        }
        try:
            await self._event_bus.emit(payload)
        except Exception:
            logger.exception("failed to emit memory context event", extra=payload)

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a shallow copy of stored context metadata."""

        return [record.to_payload() for record in self._records]


class ACEvolver:
    """Lightweight ACE evolution stub with constitutional guardrails."""

    def __init__(
        self,
        *,
        constitutional_reasoner: ConstitutionalReasoner | None = None,
    ) -> None:
        self._reasoner = constitutional_reasoner or ConstitutionalReasoner()
        self._history: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def evolve_from_patterns(
        self,
        patterns: Iterable[Any],
        validation_feedback: Any,
    ) -> dict[str, Any]:
        """Produce an evolved ACE context guarded by constitutional review."""

        coerced_patterns = _coerce_patterns(patterns)
        feedback = _coerce_validation(validation_feedback)
        async with self._lock:
            revision = len(self._history) + 1
            proposed_action = {
                "ability": "ace_evolution",
                "args": {
                    "patterns": coerced_patterns,
                    "validation": feedback,
                    "revision": revision,
                },
            }
            context = {"goal": "Evolve ACE context", "revision": revision}
            approved, reasoning = await self._reasoner.evaluate_action(
                proposed_action=proposed_action,
                current_context=context,
            )
            result = {
                "revision": revision,
                "patterns": coerced_patterns,
                "validation": feedback,
                "constitutional_reasoning": reasoning,
                "status": "approved" if approved else "rejected",
            }
            self._history.append(result)
            return result

    def history(self) -> list[dict[str, Any]]:
        """Expose recorded evolution history for observability/tests."""

        return list(self._history)


__all__ = ["ACEvolver", "LearningMemoryStack", "StoredContext"]
