import pytest

from src.collective import CollectiveIntelligenceNetwork


class _Validator:
    async def validate(self, contribution, contributor):
        return {"approved": True, "confidence": 0.9}


class _Privacy:
    async def anonymize(self, learning, privacy_level):
        sanitized = dict(learning)
        sanitized["privacy"] = privacy_level
        return sanitized


class _SharedKG:
    def __init__(self) -> None:
        self.records = []

    async def integrate_learning(self, **kwargs):
        self.records.append(kwargs)


@pytest.mark.asyncio
async def test_collective_network_accepts_contribution():
    kg = _SharedKG()
    network = CollectiveIntelligenceNetwork(
        shared_knowledge_graph=kg,
        privacy_controller=_Privacy(),
        contribution_validator=_Validator(),
    )
    result = await network.contribute_learning(
        user_id="user-1",
        learning={"insight": "value", "audience": ["user-2"]},
        privacy_level="team",
    )
    assert result["accepted"] is True
    assert kg.records[0]["learning"]["privacy"] == "team"
