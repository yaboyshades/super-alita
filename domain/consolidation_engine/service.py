"""Domain service skeleton for the Intelligence Consolidation Engine."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from .models import (
    ACEUpdateReceipt,
    ConsolidationEnvelope,
    ConsolidationEvent,
    ConsolidationEventPayload,
    ConsolidationRequestContext,
    ConsolidationResult,
)


@runtime_checkable
class ConsolidationFeatureFlagProvider(Protocol):
    """Minimal contract for feature flag lookups."""

    def is_enabled(self, key: str, default: bool = False) -> bool:
        """Return the cached flag state; failures must raise to allow fail-closed."""


@runtime_checkable
class ConsolidationEventPublisher(Protocol):
    """Protocol describing the outbound event bridge."""

    async def publish(self, event: ConsolidationEvent) -> None:
        """Publish the provided event asynchronously."""


@runtime_checkable
class ACEStoreAdapter(Protocol):
    """Adapter facade for ACE store integration."""

    async def apply_patch(self, patch: Any, *, dedupe_key: str) -> ACEUpdateReceipt:
        """Apply an idempotent patch to ACE state."""


@runtime_checkable
class AbilityRegistryAdapter(Protocol):
    """Subset of the ability registry interface consumed by consolidation."""

    async def execute(
        self, name: str, payload: Mapping[str, Any], *, correlation_id: str
    ) -> Mapping[str, Any]:
        """Execute an ability and return structured output."""


@dataclass(slots=True)
class ConsolidationMetrics:
    """Simple façade for metric instrumentation hooks."""

    attempts_counter: Any | None = None
    skips_counter: Any | None = None
    latency_histogram: Any | None = None

    def record_attempt(self) -> None:
        if callable(self.attempts_counter):
            self.attempts_counter()

    def record_skip(self, reason: str) -> None:
        if callable(self.skips_counter):
            self.skips_counter(reason)

    def observe_latency(self, value_ms: float) -> None:
        if callable(self.latency_histogram):
            self.latency_histogram(value_ms)


class ConsolidationEngine:
    """Coordinates validation, consolidation, and emission steps."""

    FEATURE_FLAG_KEY = "fea.consolidation.post_turn"

    def __init__(
        self,
        *,
        feature_flags: ConsolidationFeatureFlagProvider | None = None,
        event_publisher: ConsolidationEventPublisher | None = None,
        ace_store: ACEStoreAdapter | None = None,
        ability_registry: AbilityRegistryAdapter | None = None,
        metrics: ConsolidationMetrics | None = None,
    ) -> None:
        self._feature_flags = feature_flags
        self._event_publisher = event_publisher
        self._ace_store = ace_store
        self._ability_registry = ability_registry
        self._metrics = metrics or ConsolidationMetrics()

    def flag_enabled(self) -> bool:
        """Return whether the post-turn consolidation flag is active."""

        if self._feature_flags is None:
            return False
        with contextlib.suppress(Exception):
            return bool(
                self._feature_flags.is_enabled(self.FEATURE_FLAG_KEY, default=False)
            )
        return False

    async def consolidate_post_turn(
        self,
        envelope: ConsolidationEnvelope,
        *,
        request_context: ConsolidationRequestContext,
    ) -> ConsolidationResult:
        """Primary entrypoint invoked after each REUG turn."""

        start = time.perf_counter()
        flag_state = self.flag_enabled()
        request_context.feature_flag_state = flag_state

        if not flag_state:
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._metrics.record_skip("feature_flag_disabled")
            self._metrics.observe_latency(latency_ms)
            return ConsolidationResult(
                status="skipped",
                latency_ms=latency_ms,
                skip_reason="feature_flag_disabled",
            )

        self._metrics.record_attempt()
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._metrics.observe_latency(latency_ms)
        raise NotImplementedError(
            "Consolidation execution is not implemented in the skeleton."
        )

    @staticmethod
    def build_event(
        *,
        envelope: ConsolidationEnvelope,
        result: ConsolidationResult,
        context: ConsolidationRequestContext,
    ) -> ConsolidationEvent:
        """Construct a telemetry event from consolidation results."""

        payload = ConsolidationEventPayload(
            session_id=envelope.session_id,
            turn_id=envelope.turn_id,
            status=result.status,
            latency_ms=result.latency_ms or 0.0,
            patterns=result.patterns,
            validation=result.validation,
            ace_patch=result.ace_receipt.metadata if result.ace_receipt else {},
            skip_reason=result.skip_reason,
            trace_id=context.trace_id,
        )
        return ConsolidationEvent(payload=payload)
