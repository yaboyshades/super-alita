"""Domain service for the Intelligence Consolidation Engine."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from .models import (
    ACEUpdateReceipt,
    ConsolidationEnvelope,
    ConsolidationEvent,
    ConsolidationEventPayload,
    ConsolidationPatch,
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


@runtime_checkable
class ConstitutionalChecker(Protocol):
    """Subset of the constitutional reasoner used by consolidation."""

    async def evaluate_action(
        self, proposed_action: Mapping[str, Any], current_context: Mapping[str, Any]
    ) -> tuple[bool, str]:
        """Return approval state plus reasoning text."""


class ConsolidationEngine:
    """Coordinates validation, consolidation, and emission steps."""

    FEATURE_FLAG_KEY = "fea.consolidation.post_turn"
    MAX_LATENCY_MS = 20.0

    def __init__(
        self,
        *,
        feature_flags: ConsolidationFeatureFlagProvider | None = None,
        event_publisher: ConsolidationEventPublisher | None = None,
        ace_store: ACEStoreAdapter | None = None,
        ability_registry: AbilityRegistryAdapter | None = None,
        metrics: ConsolidationMetrics | None = None,
        constitutional_checker: ConstitutionalChecker | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._feature_flags = feature_flags
        self._event_publisher = event_publisher
        self._ace_store = ace_store
        self._ability_registry = ability_registry
        self._metrics = metrics or ConsolidationMetrics()
        self._constitutional_checker = constitutional_checker
        self._clock = clock or time.perf_counter
        self._logger = logging.getLogger(__name__)

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

        start = self._clock()
        flag_state = self.flag_enabled()
        request_context.feature_flag_state = flag_state

        if not flag_state:
            latency_ms = self._elapsed_ms(start)
            self._metrics.record_skip("feature_flag_disabled")
            self._metrics.observe_latency(latency_ms)
            return ConsolidationResult(
                status="skipped",
                latency_ms=latency_ms,
                skip_reason="feature_flag_disabled",
                validation={"approved": False, "reasoning": "feature_flag_disabled"},
            )

        self._metrics.record_attempt()
        try:
            validation = await self._run_constitutional_checks(envelope)
            if not validation.get("approved", True):
                latency_ms = self._elapsed_ms(start)
                self._metrics.record_skip("constitutional_rejection")
                result = ConsolidationResult(
                    status="rejected",
                    latency_ms=latency_ms,
                    validation=validation,
                    skip_reason="constitutional_rejection",
                )
                await self._safe_emit_event(envelope, result, request_context)
                self._metrics.observe_latency(latency_ms)
                return result

            patterns = self._extract_patterns(envelope)
            patch = self._build_patch(envelope, patterns)
            ace_receipt = await self._apply_patch(envelope, patch)

            follow_up_info = await self._maybe_execute_follow_up(
                envelope, request_context
            )
            latency_ms = self._elapsed_ms(start)

            validation_payload = dict(validation)
            if follow_up_info is not None:
                validation_payload["follow_up"] = follow_up_info
            if latency_ms > self.MAX_LATENCY_MS:
                validation_payload["latency_breach"] = True

            status = "deduplicated" if ace_receipt and ace_receipt.dedupe_hit else "applied"
            result = ConsolidationResult(
                status=status,
                latency_ms=latency_ms,
                patterns=patterns,
                validation=validation_payload,
                ace_receipt=ace_receipt,
            )
            await self._safe_emit_event(envelope, result, request_context)
            self._metrics.observe_latency(latency_ms)
            return result
        except Exception as exc:  # pragma: no cover - defensive
            latency_ms = self._elapsed_ms(start)
            self._metrics.record_skip("exception")
            failure = ConsolidationResult(
                status="failed",
                latency_ms=latency_ms,
                validation={"error": str(exc)},
                skip_reason="exception",
            )
            await self._safe_emit_event(envelope, failure, request_context)
            self._metrics.observe_latency(latency_ms)
            self._logger.exception("Consolidation execution failed")
            return failure

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

    async def _run_constitutional_checks(
        self, envelope: ConsolidationEnvelope
    ) -> Mapping[str, Any]:
        if self._constitutional_checker is None:
            return {"approved": True, "reasoning": "no_checker"}
        action = {
            "ability": "reug.consolidation.post_turn",
            "args": {
                "session_id": envelope.session_id,
                "turn_id": envelope.turn_id,
                "deduplication_key": envelope.deduplication_key,
            },
        }
        context = {
            "session_id": envelope.session_id,
            "turn_id": envelope.turn_id,
            "tool_outputs": list(envelope.tool_outputs),
            "reasoning_steps": len(envelope.reasoning_trace),
        }
        approved, reasoning = await self._constitutional_checker.evaluate_action(
            action, context
        )
        return {"approved": approved, "reasoning": reasoning}

    async def _apply_patch(
        self, envelope: ConsolidationEnvelope, patch: ConsolidationPatch
    ) -> ACEUpdateReceipt | None:
        if self._ace_store is None:
            return None
        raw = await self._ace_store.apply_patch(
            patch.model_dump(), dedupe_key=envelope.deduplication_key
        )
        if isinstance(raw, ACEUpdateReceipt):
            return raw
        if isinstance(raw, Mapping):
            return ACEUpdateReceipt.model_validate(raw)
        return ACEUpdateReceipt(applied=False, metadata={})

    async def _maybe_execute_follow_up(
        self, envelope: ConsolidationEnvelope, context: ConsolidationRequestContext
    ) -> Mapping[str, Any] | None:
        if self._ability_registry is None:
            return None
        metadata = envelope.metadata or {}
        follow_up = metadata.get("follow_up_ability")
        if not isinstance(follow_up, Mapping):
            return None
        name = str(follow_up.get("name") or follow_up.get("ability") or "").strip()
        if not name:
            return None
        payload = follow_up.get("payload")
        if not isinstance(payload, Mapping):
            payload = {}
        correlation_id = f"{envelope.session_id}:{envelope.turn_id}:{context.trace_id}"
        result = await self._ability_registry.execute(
            name, payload, correlation_id=correlation_id
        )
        return {"ability": name, "result": result}

    def _extract_patterns(
        self, envelope: ConsolidationEnvelope
    ) -> list[Mapping[str, Any]]:
        patterns: list[Mapping[str, Any]] = []
        for item in envelope.tool_outputs:
            if not isinstance(item, Mapping):
                continue
            tool_name = str(
                item.get("tool")
                or item.get("name")
                or item.get("ability")
                or "tool"
            )
            summary_source = item.get("summary") or item.get("result") or item.get("output")
            summary: str
            if isinstance(summary_source, str):
                summary = summary_source
            elif isinstance(summary_source, Mapping):
                summary = json.dumps(summary_source, sort_keys=True)[:256]
            else:
                summary = ""
            patterns.append({"tool": tool_name, "summary": summary})
        if envelope.reasoning_trace:
            patterns.append({"type": "reasoning", "steps": len(envelope.reasoning_trace)})
        return patterns

    def _build_patch(
        self, envelope: ConsolidationEnvelope, patterns: list[Mapping[str, Any]]
    ) -> ConsolidationPatch:
        payload = {
            "session_id": envelope.session_id,
            "turn_id": envelope.turn_id,
            "patterns": patterns,
            "metadata": dict(envelope.metadata),
        }
        serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        checksum = hashlib.sha256(serialized).hexdigest()
        operations = [
            {
                "op": "record_patterns",
                "session_id": envelope.session_id,
                "turn_id": envelope.turn_id,
                "patterns": patterns,
            }
        ]
        if envelope.metadata:
            operations.append({"op": "record_metadata", "metadata": dict(envelope.metadata)})
        return ConsolidationPatch(operations=operations, checksum=checksum)

    async def _safe_emit_event(
        self,
        envelope: ConsolidationEnvelope,
        result: ConsolidationResult,
        context: ConsolidationRequestContext,
    ) -> None:
        if self._event_publisher is None:
            return
        event = self.build_event(envelope=envelope, result=result, context=context)
        with contextlib.suppress(Exception):
            await self._event_publisher.publish(event)

    def _elapsed_ms(self, start: float) -> float:
        return (self._clock() - start) * 1000.0
