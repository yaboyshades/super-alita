from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agent.abilities.registry import AbilityRegistry
from src.agent.core_loop import CoreAgentLoop


@pytest.mark.asyncio
async def test_execute_task_runs_agent_loop_successfully():
    ability_registry = AbilityRegistry(abilities_config_path="config/abilities.yaml")

    class GenerationAdapter:
        async def execute(self, parameters, context):
            return {"generated": True, "parameters": parameters, "context": context}

        async def validate(self, parameters):
            return {"valid": True, "errors": [], "warnings": []}

        async def dry_run(self, parameters):
            return {"would_execute": True}

    ability_registry.register_ability_adapter("code_generation", GenerationAdapter())

    planner = SimpleNamespace(
        generate_plan=AsyncMock(
            return_value=[
                {
                    "action": "code_generation",
                    "parameters": {"requirement": "Build feature"},
                }
            ]
        )
    )

    async def _create_atom(atom_type, *_args, **_kwargs):
        return {"id": f"{atom_type}-id"}

    knowledge_graph = SimpleNamespace(
        semantic_search=AsyncMock(return_value=[]),
        create_atom=AsyncMock(side_effect=_create_atom),
        create_bond=AsyncMock(return_value={"id": "bond-1"}),
    )

    constitutional_reasoner = SimpleNamespace(
        evaluate_action=AsyncMock(return_value=(True, "ok"))
    )
    event_bus = SimpleNamespace(emit=AsyncMock())

    loop = CoreAgentLoop(
        constitutional_reasoner,
        event_bus,
        knowledge_graph,
        planner,
        ability_registry,
    )

    result = await loop.execute_task("Create new component", {"constraints": []})
    assert result["success"] is True
    assert result["goal"] == "Create new component"
    assert result["iterations"] == 1
    assert any("goal progress" in item.lower() for item in result["reflections"])

    assert knowledge_graph.create_atom.await_args_list[0].args[0] == "agent_session"
    assert event_bus.emit.await_args_list[0].args[0] == "session.started"
    assert event_bus.emit.await_args_list[-1].args[0] == "session.completed"

    knowledge_graph.create_bond.assert_awaited()
