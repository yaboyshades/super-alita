from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from src.reug_runtime.loop import execute_turn
from src.testing import llm_validation as llm_validation_module
from src.testing.llm_validation import CheckOutcome, LLMOutputValidator


@pytest.mark.asyncio
async def test_validator_aggregates_results_with_asyncio_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The validator should run all checks concurrently and combine their output."""

    gather_called = False
    original_gather = llm_validation_module.asyncio.gather

    async def tracking_gather(*args: Any, **kwargs: Any):
        nonlocal gather_called
        gather_called = True
        return await original_gather(*args, **kwargs)

    monkeypatch.setattr(llm_validation_module.asyncio, "gather", tracking_gather)

    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def make_check(
        name: str,
        result: CheckOutcome | dict[str, Any] | bool | tuple[bool, dict[str, Any]],
    ) -> Any:
        async def _check(
            agent_output: str, context: dict[str, Any] | None = None
        ) -> Any:
            calls.append((name, agent_output, context))
            await asyncio.sleep(0)
            return result

        return _check

    bias_check = make_check("bias", CheckOutcome(name="", passed=True, score=0.9))
    factual_check = make_check("factual_accuracy", {"passed": True, "score": 0.95})
    reasoning_check = make_check("reasoning", True)
    hallucination_check = make_check("hallucination", (True, {"evidence": "clean"}))

    validator = LLMOutputValidator(
        bias_check=bias_check,
        factual_accuracy_check=factual_check,
        reasoning_check=reasoning_check,
        hallucination_check=hallucination_check,
    )

    context = {"session_id": "session-123"}
    summary = await validator.validate_agent_output("final answer", context)

    assert gather_called, "validate_agent_output must use asyncio.gather"
    assert summary.passed is True
    assert set(summary.checks) == {"bias", "factual_accuracy", "reasoning", "hallucination"}
    assert all(call[1] == "final answer" for call in calls)
    assert all(ctx is context for _, _, ctx in calls)


class _NoopRegistry:
    def get_available_tools_schema(self) -> list[dict[str, Any]]:
        return []


class _NoopKG:
    pass


class _StaticModel:
    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        yield {"content": "validated output"}


class _CollectingBus:
    def __init__(self) -> None:
        self.emitted: list[dict[str, Any]] = []

    async def emit(self, event: dict[str, Any]) -> None:
        self.emitted.append(event)


@pytest.mark.asyncio
async def test_execute_turn_blocks_on_failed_validation() -> None:
    """Failed validation should surface as a TaskFailed event and halt success streaming."""

    async def pass_check(_: str, __: dict[str, Any] | None = None) -> bool:
        return True

    async def fail_check(_: str, __: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"passed": False, "reason": "factual drift"}

    validator = LLMOutputValidator(
        bias_check=pass_check,
        factual_accuracy_check=fail_check,
        reasoning_check=pass_check,
        hallucination_check=pass_check,
    )

    bus = _CollectingBus()
    registry = _NoopRegistry()
    kg = _NoopKG()
    model = _StaticModel()

    events: list[dict[str, Any]] = []
    async for event in execute_turn(
        "hello",
        "session-id",
        bus,
        registry,
        kg,
        model,
        output_validator=validator,
    ):
        events.append(event)

    failure_events = [ev for ev in events if ev.get("type") == "TaskFailed"]
    assert failure_events, "Validation failure should trigger TaskFailed event"
    failure_event = failure_events[0]
    assert failure_event.get("reason") == "output_validation_failed"
    assert failure_event.get("validation", {}).get("passed") is False
    assert "TaskSucceeded" not in {ev.get("type") for ev in events}
    assert any(
        ev.get("type") == "STATE_TRANSITION" and ev.get("to") == "RESPONDING_FAILURE"
        for ev in events
    )
    assert bus.emitted[-1] == failure_event
