from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from reug_runtime.router import execute_turn

from tests.runtime.fakes import FakeAbilityRegistry, FakeEventBus, FakeKG


class _UnicodeFormattingLLM:
    """LLM stub that emits a final answer with problematic punctuation."""

    def __init__(self) -> None:
        self._payload = (
            "Here is code:\n"
            "```python\n"
            "result = 5 × 3 − 2\n"
            "quote = '“smart” quotes'\n"
            'name = "O’Connor"\n'
            "```\n"
            "Outside — should stay fancy."
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[dict[str, str], None]:
        del messages, tools, timeout
        yield {"content": self._payload}


@pytest.mark.asyncio
async def test_execute_turn_normalizes_code_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALITA_FORMAT_ENFORCE", "true")

    bus = FakeEventBus()
    reg = FakeAbilityRegistry()
    kg = FakeKG()
    model = _UnicodeFormattingLLM()

    events = [
        event async for event in execute_turn("hi", "session", bus, reg, kg, model)
    ]

    final_event = events[-1]
    assert final_event["type"] == "TaskSucceeded"

    expected = (
        "Here is code:\n"
        "```python\n"
        "result = 5  *  3 - 2\n"
        "quote = '\"smart\" quotes'\n"
        'name = "O\'Connor"\n'
        "```\n"
        "Outside — should stay fancy."
    )

    assert final_event["data"]["content"] == expected
    # Ensure punctuation outside fences is untouched
    assert "Outside — should stay fancy." in final_event["data"]["content"]
