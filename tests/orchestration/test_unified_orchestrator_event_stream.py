from __future__ import annotations

from typing import Any

import pytest

from src.orchestration.unified_orchestrator import (
    UnifiedOrchestrator,
    UnifiedRunConfig,
)


class RecordingEventBus:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class MinimalRegistry:
    def __init__(self) -> None:
        self._tools = {"task_planner", "deepconf_consensus"}

    def knows(self, name: str) -> bool:
        return name in self._tools

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "task_planner":
            return {
                "steps": [
                    {"id": 1, "action": "Plan", "rationale": "contract-test"}
                ]
            }
        if name == "deepconf_consensus":
            prompt = args.get("prompt", "")
            return {"consensus_text": f"{prompt} (consensus)"}
        raise ValueError(name)


@pytest.mark.asyncio
async def test_unified_orchestrator_emits_canonical_sequence() -> None:
    bus = RecordingEventBus()
    orchestrator = UnifiedOrchestrator(MinimalRegistry(), bus)
    config = UnifiedRunConfig.from_args(
        "Generate plan",
        {
            "enable_planning": True,
            "enable_consensus": True,
            "enable_tasks": False,
        },
    )

    events = [event async for event in orchestrator.run_stream(config)]

    assert events  # at least one event
    kinds = [event.get("kind") for event in events]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] in {"RunTerminated", "RunFailed"}

    sequences = [event["sequence"] for event in events]
    assert sequences == sorted(sequences)
    assert sequences[0] == 0

    planning_started = next(
        (event for event in events if event["kind"] == "StageStarted"), None
    )
    assert planning_started is not None
    assert planning_started.get("stage") == "planning"

    terminated = events[-1]
    assert terminated["kind"] == "RunTerminated"
    assert terminated["data"]["success"] is True
    assert terminated.get("constitutional_score") is not None

    # Event bus should see the same canonical events
    assert [event.get("kind") for event in bus.events] == kinds
