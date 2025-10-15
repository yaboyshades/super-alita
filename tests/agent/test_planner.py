from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agent.abilities.registry import AbilityRegistry
from src.agent.planner import Planner


@pytest.mark.asyncio
async def test_generate_plan_orders_by_risk_and_adds_fallback(monkeypatch):
    knowledge_graph = SimpleNamespace(semantic_search=AsyncMock(return_value=[]))
    orchestrator = SimpleNamespace(generate=AsyncMock())
    registry = AbilityRegistry(abilities_config_path="config/abilities.yaml")

    planner = Planner(knowledge_graph, orchestrator, registry)

    monkeypatch.setattr(planner, "_find_similar_plans", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        planner,
        "_decompose_goal",
        AsyncMock(
            return_value=[
                {
                    "action": "file_operation",
                    "parameters": {"operation": "write", "path": "example"},
                },
                {
                    "action": "code_analysis",
                    "parameters": {"target": "module"},
                },
            ]
        ),
    )

    plan = await planner.generate_plan("Update a file", [], {})
    assert plan[0]["action"] == "code_analysis"
    assert plan[1]["action"] == "file_operation"
    assert plan[1]["fallback_action"] == "file_analysis"
    assert plan[1]["max_retries"] == 3


@pytest.mark.asyncio
async def test_llm_decomposition_fallback_to_rule_based(monkeypatch):
    knowledge_graph = SimpleNamespace(semantic_search=AsyncMock(return_value=[]))
    orchestrator = SimpleNamespace(generate=AsyncMock(side_effect=RuntimeError("llm error")))
    registry = AbilityRegistry(abilities_config_path="config/abilities.yaml")

    planner = Planner(knowledge_graph, orchestrator, registry)
    steps = await planner._llm_decompose_goal("Analyze code", [], {"relevant_patterns": []})
    assert steps[0]["action"] == "code_analysis"
