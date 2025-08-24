import asyncio
import pytest
from typing import Any, Dict

from src.plugins.deepcode_orchestrator_plugin import DeepCodeOrchestratorPlugin, DeepCodeClientInterface


class FakeBus:
    async def subscribe(self, event_type: str, handler):
        return None


class FakeDC(DeepCodeClientInterface):
    async def plan(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return {"steps": ["A", "B"], "confidence": 0.72}

    async def collect_references(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {"snippets": [{"path": "x.py", "code": "print(1)"}], "confidence": 0.79}

    async def generate_code(self, plan: Dict[str, Any], refs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "proposal_id": "prop123",
            "diffs": [
                {
                    "path": "x.py",
                    "unified_diff": "--- a\n+++ b\n",
                    "new_content": "print(2)",
                    "confidence": 0.85,
                }
            ],
            "tests": [],
            "docs": [],
            "confidence": 0.85,
        }

    async def validate(self, impl: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "pass", "lint_errors": 0, "tests_passed": True, "confidence": 0.9}


@pytest.mark.asyncio
async def test_emits_full_sequence(monkeypatch):
    events = []

    class P(DeepCodeOrchestratorPlugin):
        async def emit_event(self, event_type: str, **fields):
            events.append({"event_type": event_type, **fields})
            return await super().emit_event(event_type, **fields)

    plugin = P(client=FakeDC())
    await plugin.setup(event_bus=FakeBus(), store=None, config={})
    await plugin.start()

    await plugin._on_request(
        {"request_id": "req1", "task_kind": "paper2code", "requirements": "do X", "conversation_id": "c1"}
    )
    await asyncio.sleep(0.05)

    kinds = [e["event_type"] for e in events]
    assert "deepcode_request_received" in kinds
    assert "deepcode_plan_ready" in kinds
    assert "deepcode_references_compiled" in kinds
    assert "deepcode_implementation_proposed" in kinds
    assert "deepcode_validation_report" in kinds
    assert "deepcode_ready_for_apply" in kinds
