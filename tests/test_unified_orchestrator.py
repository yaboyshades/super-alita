from src.orchestration.unified_orchestrator import (
    UnifiedOrchestrator,
    UnifiedRunConfig,
)


class DummyRegistry:
    def __init__(self):
        self._tools = set()

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
    async def emit(self, _):  # pragma: no cover - simple
        return None


async def _collect(gen):
    out = []
    async for ev in gen:
        out.append(ev)
    return out


def test_unified_basic_event_loop(event_loop):  # type: ignore
    reg = DummyRegistry()
    # Register minimal tools
    reg._tools.update({"task_planner", "deepconf_consensus"})
    orch = UnifiedOrchestrator(reg, DummyBus())
    cfg = UnifiedRunConfig.from_args(
        "Test prompt",
        {"enable_planning": True, "enable_consensus": True},
    )
    events = event_loop.run_until_complete(_collect(orch.run_stream(cfg)))
    # Ensure start and done events exist
    kinds = {e.get("type") for e in events}
    assert "UnifiedRunStarted" in kinds
    assert "UnifiedRunCompleted" in kinds
    # Validate aggregate consensus
    done = [e for e in events if e.get("type") == "UnifiedRunCompleted"][0]
    assert done["aggregate"].get("consensus_text")
