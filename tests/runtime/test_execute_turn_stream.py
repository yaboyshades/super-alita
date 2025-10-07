import json

import pytest
from reug_runtime.router import Orchestrator, execute_turn

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
    assert (
        event_types.index("AbilityCalled")
        < event_types.index("AbilitySucceeded")
        < event_types.index("TaskSucceeded")
    )

    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


class _EarlyFinalLLM:
    """LLM that emits tool call and final answer in one stream."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, messages, timeout):  # type: ignore[override]
        del messages, timeout
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
    assert (
        event_types.index("AbilityCalled")
        < event_types.index("AbilitySucceeded")
        < event_types.index("TaskSucceeded")
    )

    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


class _StructuredToolLLM:
    """LLM that emits structured tool calls followed by assistant text."""

    async def stream_chat(self, messages, tools, timeout):  # type: ignore[override]
        del messages, tools, timeout
        yield {
            "type": "tool_calls",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "echo",
                        "arguments": json.dumps({"payload": "hi"}),
                    },
                }
            ],
        }
        yield {"content": "Model text."}


@pytest.mark.asyncio
async def test_reasoning_result_tracks_text_and_tool_calls() -> None:
    bus = FakeEventBus()
    reg = FakeAbilityRegistry()
    model = _StructuredToolLLM()
    orchestrator = Orchestrator(bus, reg, model, correlation_id="corr-1")

    messages = [{"role": "user", "content": "hi"}]
    tool_schemas = reg.get_available_tools_schema()

    async for _ in orchestrator._reasoning_step(messages, tool_schemas):
        pass

    snapshot = orchestrator._last_reasoning_result
    assert snapshot.text == "Model text."
    assert snapshot.tool_calls == [
        {
            "id": "call_1",
            "function": {
                "name": "echo",
                "arguments": json.dumps({"payload": "hi"}),
            },
        }
    ]

    tool_calls_for_acting = list(snapshot.tool_calls)
    async for _ in orchestrator._acting_step(tool_calls_for_acting):
        pass

    assert orchestrator._last_reasoning_result.text == "Model text."
    assert orchestrator._last_reasoning_result.tool_calls == snapshot.tool_calls
