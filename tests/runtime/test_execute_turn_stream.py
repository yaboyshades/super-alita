import pytest

from reug_runtime.router import execute_turn
from tests.runtime.fakes import (
    FakeAbilityRegistry,
    FakeEventBus,
    FakeKG,
    FakeLLM,
)


@pytest.mark.asyncio
async def test_execute_turn_stream_normal_flow() -> None:
    bus = FakeEventBus()
    reg = FakeAbilityRegistry()
    kg = FakeKG()
    model = FakeLLM()

    gen = execute_turn("hi", "s1", bus, reg, kg, model)
    chunks: list[str] = []
    async for chunk in gen:
        chunks.append(chunk)

    output = "".join(chunks)
    assert "<tool_call>" in output
    assert output.endswith("</final_answer>")

    event_types = [e["type"] for e in bus.events]
    assert event_types.index("AbilityCalled") < event_types.index("AbilitySucceeded") < event_types.index(
        "TaskSucceeded"
    )

    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


class _EarlyFinalLLM:
    """LLM that emits tool call and final answer in one stream."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, messages, timeout):  # type: ignore[override]
        self.calls += 1
        if self.calls == 1:
            text = (
                '<tool_call>{"tool":"echo","args":{"payload":"hi"}}</tool_call>'
                '<final_answer>{"content":"early done","citations":[]}</final_answer>'
            )
            yield {"content": text}
        else:
            if False:
                yield {"content": ""}  # pragma: no cover


@pytest.mark.asyncio
async def test_execute_turn_stream_early_final_answer() -> None:
    bus = FakeEventBus()
    reg = FakeAbilityRegistry()
    kg = FakeKG()
    model = _EarlyFinalLLM()

    gen = execute_turn("hi", "s1", bus, reg, kg, model)
    chunks = [chunk async for chunk in gen]

    output = "".join(chunks)
    assert output.endswith("</final_answer>")
    assert model.calls == 1

    event_types = [e["type"] for e in bus.events]
    assert event_types.index("AbilityCalled") < event_types.index("AbilitySucceeded") < event_types.index(
        "TaskSucceeded"
    )

    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()
