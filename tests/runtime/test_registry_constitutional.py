from typing import Any

import pytest

from src.governance import ConstitutionalViolationError
from src.main import SimpleAbilityRegistry


class _StubChecker:
    def __init__(self, approved: bool = True) -> None:
        self.approved = approved
        self.seen: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

    async def evaluate_action(
        self,
        proposed_action: dict[str, Any],
        current_context: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        self.seen.append((proposed_action, current_context))
        if self.approved:
            return True, "ok"
        return False, "rejected"


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


@pytest.mark.asyncio
async def test_registry_uses_injected_constitutional_checker():
    checker = _StubChecker()
    registry = SimpleAbilityRegistry(constitutional_checker=checker)
    await registry.execute("echo", {"payload": "hi"})
    assert checker.seen, "expected checker to be invoked"
    action, context = checker.seen[0]
    assert action["ability"] == "echo"
    assert isinstance(context, dict)
    assert context.get("goal")


@pytest.mark.asyncio
async def test_registry_respects_injected_block():
    registry = SimpleAbilityRegistry(constitutional_checker=_StubChecker(approved=False))
    with pytest.raises(ConstitutionalViolationError):
        await registry.execute("echo", {"payload": "nope"})
