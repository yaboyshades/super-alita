import pytest

from src.governance import ConstitutionalViolationError
from src.main import SimpleAbilityRegistry


@pytest.mark.asyncio
async def test_registry_blocks_constitutional_violation():
    registry = SimpleAbilityRegistry()
    registry.register_tool(
        contract={
            "tool_id": "dangerous_tool",
            "description": "Dangerous action",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        },
        executor=lambda args: {"status": "ok"},
    )
    with pytest.raises(ConstitutionalViolationError):
        await registry.execute("dangerous_tool", {"unsafe": True})


@pytest.mark.asyncio
async def test_registry_allows_safe_execution():
    registry = SimpleAbilityRegistry()
    result = await registry.execute("echo", {"payload": "hello"})
    assert result["echo"] == "hello"
