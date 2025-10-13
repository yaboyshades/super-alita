import pytest

from src.evolution import CapabilityEvolutionEngine


class _Monitor:
    def __init__(self, failures):
        self._failures = failures
        self.tracked = []

    async def get_recent_failures(self):
        return self._failures

    async def track_capability(self, capability_id, deployment_context):
        self.tracked.append((capability_id, deployment_context))


class _Generator:
    async def propose(self, gap):
        proposal = dict(gap)
        proposal.update({"id": "cap-1", "success_metrics": ["latency"]})
        return proposal


@pytest.mark.asyncio
async def test_capability_evolver_deploys_safe_capability():
    monitor = _Monitor([
        {"suggested_capability": "improve latency", "details": "timeout"},
    ])
    evolver = CapabilityEvolutionEngine(
        performance_monitor=monitor,
        capability_generator=_Generator(),
    )
    deployed = await evolver.evolve_capabilities()
    assert deployed and deployed[0]["id"] == "cap-1"
    assert monitor.tracked[0][0] == "cap-1"
