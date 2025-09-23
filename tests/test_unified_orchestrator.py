import pytest

from src.orchestration.unified_orchestrator import (
    UnifiedOrchestrator,
    UnifiedRunConfig,
)


class DummyRegistry:
    def __init__(self):
        self._tools = {"task_planner", "deepconf_consensus"}

    def knows(self, name: str) -> bool:  # minimal interface
        return name in self._tools

    async def execute(self, name: str, args):  # pragma: no cover - simple
        if name == "task_planner":
            return {
                "steps": [
                    {"id": 1, "action": "Do X", "rationale": "demo"}
                ]
            }
        if name == "deepconf_consensus":
            return {"consensus_text": args.get("prompt") + " (consensus)"}
        raise ValueError("unknown tool")


class DummyBus:
    async def emit(self, event):  # pragma: no cover - simple
        return None


@pytest.mark.asyncio
async def test_unified_basic_event_loop():  # type: ignore
    reg = DummyRegistry()
    orch = UnifiedOrchestrator(reg, DummyBus())
    cfg = UnifiedRunConfig.from_args(
        "Test prompt",
        {"enable_planning": True, "enable_consensus": True},
    )
    events = [event async for event in orch.run_stream(cfg)]
    kinds = [e.get("kind") for e in events]
    assert kinds[0] == "RunStarted"
    assert kinds[-1] == "RunTerminated"
    terminated = events[-1]
    assert terminated["data"]["success"] is True
    assert terminated.get("constitutional_score") is not None
    assert any(e.get("kind") == "StageStarted" for e in events)
