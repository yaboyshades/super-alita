import asyncio

from reug.events import EventEmitter
from reug.fsm import ExecutionFlow
from reug.services import create_services


def test_user_input_tool(monkeypatch):
    """Ensure the fallback executor can capture live user input."""
    events: list[dict] = []
    emitter = EventEmitter(sink=lambda e: events.append(e))
    services = create_services(emitter)
    flow = ExecutionFlow(services, emitter)

    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")

    ctx = asyncio.run(
        flow.run(
            {
                "plan": [
                    {
                        "id": "s1",
                        "kind": "USER",
                        "args": {"prompt": "say something:"},
                    }
                ]
            }
        )
    )

    assert ctx.results[0]["response"] == "hello"
    tool_ok = [e for e in events if e.get("event_type") == "TOOL_CALL_OK"]
    assert tool_ok and tool_ok[0]["payload"]["tool_id"] == "tool.user_input"

