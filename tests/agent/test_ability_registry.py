from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agent.abilities.registry import AbilityRegistry, create_ability_registry


@pytest.mark.asyncio
async def test_registry_loads_config_and_stub_execution(tmp_path):
    config_file = tmp_path / "abilities.yaml"
    config_file.write_text(
        """
abilities:
  sample_ability:
    description: "Sample ability"
    parameters:
      value: "integer"
    required_parameters: ["value"]
    cost_estimate: 0.2
    risk_level: "low"
    timeout_seconds: 10
    constitutional_risk_factors: ["sample"]
        """
    )
    registry = create_ability_registry(str(config_file))
    ability = registry.get_ability("sample_ability")
    assert ability is not None
    assert ability.metadata.risk_level == "low"
    with pytest.raises(NotImplementedError):
        await ability.execute({"value": 1}, {})


@pytest.mark.asyncio
async def test_register_adapter_updates_behaviour():
    registry = AbilityRegistry(abilities_config_path="config/abilities.yaml")
    adapter_calls = SimpleNamespace(executed=False, validated=False)

    class Adapter:
        async def execute(self, parameters, context):
            adapter_calls.executed = True
            return {"status": "ok", "parameters": parameters, "context": context}

        async def validate(self, parameters):
            adapter_calls.validated = True
            return {"valid": True, "errors": [], "warnings": []}

        async def dry_run(self, parameters):
            return {"would_execute": True}

    registry.register_ability_adapter("code_analysis", Adapter())
    ability = registry.get_ability("code_analysis")
    assert ability is not None

    validation = await ability.validate({"code": "print('hello')"})
    assert validation["valid"] is True
    assert adapter_calls.validated is True

    result = await ability.execute({"code": "print('hi')"}, {"context": "test"})
    assert adapter_calls.executed is True
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_validate_ability_execution_includes_security_feedback():
    registry = AbilityRegistry(abilities_config_path="config/abilities.yaml")
    registry.register_ability_adapter(
        "search_code",
        SimpleNamespace(
            execute=AsyncMock(),
            validate=AsyncMock(return_value={"valid": True, "errors": [], "warnings": []}),
            dry_run=AsyncMock(),
        ),
    )

    class SecurityContext:
        def __init__(self):
            self.validate_input = AsyncMock(
                return_value=SimpleNamespace(valid=False, errors=["policy"], warnings=["warn"])
            )

    security = SecurityContext()
    result = await registry.validate_ability_execution(
        "search_code", {"query": "core loop"}, security
    )
    assert result["valid"] is False
    assert "policy" in result["errors"]
    assert "warn" in result["warnings"]
