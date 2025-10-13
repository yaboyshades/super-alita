from typing import Any

import pytest

from src.agents.coordination_protocol import AgentCallbacks, DistributedCognitionOrchestrator


@pytest.mark.asyncio
async def test_coordination_protocol_synthesises_solution():
    orchestrator = DistributedCognitionOrchestrator()

    async def analyze(problem: str, shared: dict[str, Any]):
        return {"contributions": {problem: "analysis"}, "insights": [problem]}

    async def share(analyses, shared):
        return {"contributions": {"signal": "shared"}}

    async def synthesize(analyses, shared):
        return {"solution": sorted(shared.keys()), "agents": list(analyses)}

    callbacks = AgentCallbacks(analyze=analyze, share=share, synthesize=synthesize)
    orchestrator.register_agent("alpha", callbacks)
    result = await orchestrator.coordinate_problem_solving("task", ["alpha"])
    assert "signal" in result["solution"]
    assert result["agents"] == ["alpha"]
