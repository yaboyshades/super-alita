import pytest

from src.governance import ConstitutionalReasoner


@pytest.mark.asyncio
async def test_reasoner_approves_safe_action():
    reasoner = ConstitutionalReasoner()
    approved, reasoning = await reasoner.evaluate_action(
        {"ability": "echo", "args": {"payload": "hi"}},
        {"goal": "echo user input"},
    )
    assert approved is True
    assert "Transparency" in reasoning


@pytest.mark.asyncio
async def test_reasoner_blocks_unsafe_action():
    reasoner = ConstitutionalReasoner()
    approved, reasoning = await reasoner.evaluate_action(
        {"ability": "delete", "args": {"target": "system"}, "unsafe": True},
        {"goal": "danger"},
    )
    assert approved is False
    assert "⚠️" in reasoning
