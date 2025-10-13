"""Memory consolidation utilities for post-interaction learning.

The consolidation logic is inspired by rehearsal-based continual learning
mechanisms described by McClelland et al. (1995) and more recent reinforcement
learning replay buffers popularised by DeepMind's DQN work (Mnih et al., 2015).
These routines stitch together existing ACE, validation, and event pipelines so
that each interaction becomes an update to the shared world model.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol

logger = logging.getLogger(__name__)


class EventBusLike(Protocol):
    """Protocol describing the subset of the event bus API used here."""

    async def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        """Publish an event into the runtime bus."""


class ACEvolverLike(Protocol):
    """Minimal protocol for ACE evolution interfaces."""

    async def evolve_from_patterns(
        self, patterns: list[dict[str, Any]], validation_feedback: dict[str, Any]
    ) -> dict[str, Any]:
        """Return an evolved ACE context based on extracted patterns."""


class EventProcessorLike(Protocol):
    """Protocol for complex event processors used during consolidation."""

    async def extract_patterns(
        self, session_id: str, outcome: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract salient patterns from a turn outcome."""


class ValidatorLike(Protocol):
    """Protocol for validation components with asynchronous interfaces."""

    async def validate_outcome(self, outcome: dict[str, Any]) -> dict[str, Any]:
        """Validate the provided outcome and return structured feedback."""


class MemoryStackLike(Protocol):
    """Protocol describing knowledge graph or memory sink integrations."""

    async def store_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Persist the evolved context and return storage metadata."""


@dataclass(slots=True)
class ConsolidationArtifacts:
    """Container describing the by-products of a consolidation cycle."""

    patterns: list[dict[str, Any]]
    evolved_context: dict[str, Any]
    validation_feedback: dict[str, Any]
    storage_record: Optional[dict[str, Any]]


def _ensure_awaitable(value: Any) -> Awaitable[Any]:
    """Wrap synchronous values in an awaitable for uniform processing."""

    if asyncio.iscoroutine(value) or isinstance(value, Awaitable):
        return value  # type: ignore[return-value]

    async def _wrapper() -> Any:
        return value

    return _wrapper()


class IntelligenceConsolidator:
    """Bridge ACE, validation, and memory components into a learning pipeline.

    The consolidator is intentionally defensive: all dependencies are optional,
    enabling the runtime loop to opt-in gradually without sacrificing
    reliability. Each stage is awaited if implemented, otherwise skipped.
    """

    def __init__(
        self,
        *,
        ace_evolver: ACEvolverLike | None = None,
        event_processor: EventProcessorLike | None = None,
        validator: ValidatorLike | None = None,
        memory_stack: MemoryStackLike | None = None,
        event_bus: EventBusLike | None = None,
    ) -> None:
        self._ace_evolver = ace_evolver
        self._event_processor = event_processor
        self._validator = validator
        self._memory_stack = memory_stack
        self._event_bus = event_bus

    async def consolidate_interaction(
        self, session_id: str, interaction_outcome: dict[str, Any]
    ) -> ConsolidationArtifacts:
        """Consolidate the artefacts of a finished interaction.

        Args:
            session_id: Session identifier associated with the interaction.
            interaction_outcome: Structured record describing the turn outcome.

        Returns:
            ConsolidationArtifacts describing patterns, evolved context, and
            persistence metadata.

        Raises:
            ValueError: If ``session_id`` is empty or the outcome payload is not
                a mapping.
        """

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(interaction_outcome, dict):
            raise ValueError("interaction_outcome must be a dictionary")

        validation_feedback = await self._perform_validation(interaction_outcome)
        patterns = await self._extract_patterns(session_id, interaction_outcome)
        evolved_context = await self._evolve_context(patterns, validation_feedback)
        storage_record = await self._update_collective_memory(evolved_context)
        await self._emit_learning_event(
            session_id,
            patterns,
            evolved_context,
            validation_feedback,
        )
        return ConsolidationArtifacts(
            patterns=patterns,
            evolved_context=evolved_context,
            validation_feedback=validation_feedback,
            storage_record=storage_record,
        )

    async def _perform_validation(
        self, interaction_outcome: dict[str, Any]
    ) -> dict[str, Any]:
        if not self._validator:
            return interaction_outcome.get("validation", {})
        try:
            result = await _ensure_awaitable(
                self._validator.validate_outcome(interaction_outcome)
            )
        except Exception:
            logger.exception("Validation step failed; returning optimistic feedback")
            return interaction_outcome.get("validation", {})
        if not isinstance(result, dict):
            logger.debug("Validator returned non-dict payload; coercing to dict")
            return {"raw": result}
        return result

    async def _extract_patterns(
        self, session_id: str, interaction_outcome: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if not self._event_processor:
            return list(interaction_outcome.get("patterns", []))
        try:
            result = await _ensure_awaitable(
                self._event_processor.extract_patterns(
                    session_id=session_id, outcome=interaction_outcome
                )
            )
        except Exception:
            logger.exception("Pattern extraction failed; falling back to raw data")
            return list(interaction_outcome.get("patterns", []))
        if not isinstance(result, list):
            logger.debug("Event processor returned non-list payload; coercing")
            return [
                result
            ]  # type: ignore[list-item] - best effort; consumer handles schema
        coerced: list[dict[str, Any]] = [
            candidate for candidate in result if isinstance(candidate, dict)
        ]
        return coerced

    async def _evolve_context(
        self,
        patterns: list[dict[str, Any]],
        validation_feedback: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._ace_evolver:
            return {
                "patterns": patterns,
                "validation": validation_feedback,
            }
        try:
            result = await _ensure_awaitable(
                self._ace_evolver.evolve_from_patterns(
                    patterns=patterns, validation_feedback=validation_feedback
                )
            )
        except Exception:
            logger.exception("ACE evolution failed; emitting conservative context")
            return {
                "patterns": patterns,
                "validation": validation_feedback,
            }
        if not isinstance(result, dict):
            logger.debug("ACE evolver returned non-dict payload; wrapping result")
            return {"ace_result": result, "patterns": patterns}
        return result

    async def _update_collective_memory(
        self, evolved_context: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        if not self._memory_stack:
            return None
        try:
            record = await _ensure_awaitable(
                self._memory_stack.store_context(evolved_context)
            )
        except Exception:
            logger.exception("Failed to persist evolved context; continuing anyway")
            return None
        if not isinstance(record, dict):
            return {"raw": record}
        return record

    async def _emit_learning_event(
        self,
        session_id: str,
        patterns: list[dict[str, Any]],
        evolved_context: dict[str, Any],
        validation_feedback: dict[str, Any],
    ) -> None:
        if not self._event_bus:
            return
        payload = {
            "type": "intelligence_evolved",
            "session_id": session_id,
            "new_patterns": patterns,
            "evolved_context": evolved_context,
            "validation": validation_feedback,
        }
        try:
            await self._event_bus.emit(payload)
        except Exception:
            logger.exception("Failed to emit consolidation telemetry")
