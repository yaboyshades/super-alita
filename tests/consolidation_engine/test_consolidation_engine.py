from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping

import pytest

from adapters.consolidation_engine import EnvironmentFeatureFlagProvider
from app.consolidation_engine import (
    ConsolidationAppConfig,
    configure_post_turn_consolidation,
)
from domain.consolidation_engine import (
    ConsolidationEngine,
    ConsolidationEnvelope,
    ConsolidationEvent,
    ConsolidationRequestContext,
    ConsolidationResult,
    ConsolidationMetrics,
)


class _ToggleFlagProvider:
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled

    def is_enabled(self, key: str, default: bool = False) -> bool:
        return self._enabled


class _AsyncEventBus:
    def __init__(self) -> None:
        self.published: list[Mapping[str, Any]] = []

    async def publish(self, event: Mapping[str, Any]) -> Mapping[str, Any]:
        self.published.append(event)
        return event


class _AbilityRegistry:
    async def execute(
        self, name: str, payload: Mapping[str, Any], *, correlation_id: str
    ) -> Mapping[str, Any]:
        return {"name": name, "correlation_id": correlation_id, "payload": payload}


class _ACEStore:
    async def apply_patch(self, patch: Any, *, dedupe_key: str) -> Mapping[str, Any]:
        return {"applied": False, "dedupe_key": dedupe_key, "patch": patch}


@pytest.fixture()
def sample_envelope() -> ConsolidationEnvelope:
    return ConsolidationEnvelope(
        session_id="session-1",
        turn_id="turn-1",
        timestamp=datetime.now(UTC),
        agent_snapshot={},
        reasoning_trace=[],
        tool_outputs=[],
        deduplication_key="session-1:turn-1",
    )


@pytest.fixture()
def request_context() -> ConsolidationRequestContext:
    return ConsolidationRequestContext(
        trace_id="trace-123",
        orchestrator_version="vTest",
    )


@pytest.mark.asyncio()
async def test_consolidation_skips_when_flag_disabled(
    sample_envelope: ConsolidationEnvelope,
    request_context: ConsolidationRequestContext,
) -> None:
    skips: list[str] = []
    latencies: list[float] = []
    metrics = ConsolidationMetrics(
        skips_counter=skips.append,
        latency_histogram=latencies.append,
    )
    engine = ConsolidationEngine(
        feature_flags=_ToggleFlagProvider(False),
        metrics=metrics,
    )

    result = await engine.consolidate_post_turn(
        sample_envelope, request_context=request_context
    )

    assert result.status == "skipped"
    assert result.skip_reason == "feature_flag_disabled"
    assert request_context.feature_flag_state is False
    assert skips == ["feature_flag_disabled"]
    assert len(latencies) == 1


class _ConstitutionalChecker:
    def __init__(self, approved: bool = True, reasoning: str = "ok") -> None:
        self.approved = approved
        self.reasoning = reasoning
        self.calls: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []

    async def evaluate_action(
        self, proposed_action: Mapping[str, Any], current_context: Mapping[str, Any]
    ) -> tuple[bool, str]:
        self.calls.append((proposed_action, current_context))
        return self.approved, self.reasoning


class _Publisher:
    def __init__(self) -> None:
        self.events: list[ConsolidationEvent] = []

    async def publish(self, event: ConsolidationEvent) -> None:
        self.events.append(event)


@pytest.mark.asyncio()
async def test_consolidation_executes_when_flag_enabled(
    sample_envelope: ConsolidationEnvelope,
    request_context: ConsolidationRequestContext,
) -> None:
    publisher = _Publisher()
    checker = _ConstitutionalChecker()
    ability_calls: list[Mapping[str, Any]] = []

    class _AbilityRegistry:
        async def execute(
            self,
            name: str,
            payload: Mapping[str, Any],
            *,
            correlation_id: str,
        ) -> Mapping[str, Any]:
            record = {
                "name": name,
                "payload": dict(payload),
                "correlation_id": correlation_id,
            }
            ability_calls.append(record)
            return record

    class _ACEStore:
        async def apply_patch(
            self, patch: Mapping[str, Any], *, dedupe_key: str
        ) -> Mapping[str, Any]:
            return {"applied": True, "dedupe_hit": False, "metadata": {"key": dedupe_key}}

    enriched_envelope = sample_envelope.model_copy(
        update={
            "tool_outputs": [{"tool": "echo", "result": "pong"}],
            "metadata": {
                "follow_up_ability": {"name": "post_process", "payload": {"foo": "bar"}}
            },
        }
    )

    engine = ConsolidationEngine(
        feature_flags=_ToggleFlagProvider(True),
        event_publisher=publisher,
        ace_store=_ACEStore(),
        ability_registry=_AbilityRegistry(),
        constitutional_checker=checker,
    )

    result = await engine.consolidate_post_turn(
        enriched_envelope, request_context=request_context
    )

    assert result.status == "applied"
    assert result.ace_receipt is not None
    assert result.ace_receipt.applied is True
    assert result.validation["approved"] is True
    assert "follow_up" in result.validation
    assert ability_calls
    assert publisher.events
    event = publisher.events[0]
    assert event.payload.session_id == enriched_envelope.session_id
    assert checker.calls


@pytest.mark.asyncio()
async def test_consolidation_rejects_on_constitutional_failure(
    sample_envelope: ConsolidationEnvelope,
    request_context: ConsolidationRequestContext,
) -> None:
    checker = _ConstitutionalChecker(approved=False, reasoning="denied")
    skips: list[str] = []
    metrics = ConsolidationMetrics(skips_counter=skips.append)
    publisher = _Publisher()

    engine = ConsolidationEngine(
        feature_flags=_ToggleFlagProvider(True),
        constitutional_checker=checker,
        metrics=metrics,
        event_publisher=publisher,
    )

    result = await engine.consolidate_post_turn(
        sample_envelope, request_context=request_context
    )

    assert result.status == "rejected"
    assert result.skip_reason == "constitutional_rejection"
    assert result.validation["reasoning"] == "denied"
    assert skips == ["constitutional_rejection"]
    assert publisher.events


def test_build_event_serializes_result(
    sample_envelope: ConsolidationEnvelope,
    request_context: ConsolidationRequestContext,
) -> None:
    result = ConsolidationResult(status="skipped", latency_ms=0.5, skip_reason="flag")
    event = ConsolidationEngine.build_event(
        envelope=sample_envelope,
        result=result,
        context=request_context,
    )
    assert isinstance(event, ConsolidationEvent)
    assert event.payload.status == "skipped"
    assert event.payload.skip_reason == "flag"
    assert event.payload.schema_version == "v1"


def test_environment_feature_flag_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = EnvironmentFeatureFlagProvider()
    key = "fea.consolidation.post_turn"
    env_key = key.replace(".", "_").upper()
    monkeypatch.setenv(env_key, "true")
    assert provider.is_enabled(key) is True
    monkeypatch.setenv(env_key, "off")
    assert provider.is_enabled(key, default=True) is False


@pytest.mark.asyncio()
async def test_configure_post_turn_consolidation_registers_hook(
    sample_envelope: ConsolidationEnvelope,
    request_context: ConsolidationRequestContext,
) -> None:
    class _Loop:
        def __init__(self) -> None:
            self.hooks: list[Any] = []

        def register_post_turn_hook(self, hook: Any) -> None:
            self.hooks.append(hook)

    loop = _Loop()
    bus = _AsyncEventBus()
    config = ConsolidationAppConfig(
        event_bus=bus,
        ability_registry=_AbilityRegistry(),
        ace_store=_ACEStore(),
        flag_provider=_ToggleFlagProvider(False),
    )
    engine = configure_post_turn_consolidation(loop=loop, config=config)

    assert engine.flag_enabled() is False
    assert loop.hooks
    hook = loop.hooks[0]
    result = await hook(sample_envelope, request_context=request_context)
    assert isinstance(result, ConsolidationResult)
    assert result.status == "skipped"
