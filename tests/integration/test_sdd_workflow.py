import pytest

from src.integration.ladder_multiagent_bridge import LADDERTask
from src.orchestrator.coordinator import UnifiedOrchestrator


@pytest.mark.asyncio
async def test_sdd_workflow_executes_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = UnifiedOrchestrator()

    async def fake_execute(task: LADDERTask) -> dict:
        return {
            "task_id": task.task_id,
            "code": "def foo(): return 42",
            "compliance_score": 0.8,
            "telemetry": {"events": 3},
        }

    monkeypatch.setattr(orchestrator, "execute_ladder_task", fake_execute)

    outcome = await orchestrator.run_sdd_workflow(
        feature_id="feat-unified-system",
        specification="Unify orchestrator, cognitive modules, and constitutional gates.",
    )

    assert outcome["compliance_score"] >= 0.75
    assert outcome["task_id"].startswith("feat-unified-system")
