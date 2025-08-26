import asyncio

from reug.events import EventEmitter
from reug.services import create_services
from reug.fsm import ExecutionFlow


def test_trace_integrity(tmp_path, monkeypatch):
    monkeypatch.setenv("REUG_EVENT_LOG_DIR", str(tmp_path / "logs"))

    events = []
    emitter = EventEmitter(events.append)
    services = create_services(emitter)
    flow = ExecutionFlow(services, emitter)

    asyncio.run(flow.run({"sot": ["compute:1+1", "generate:hi", "compute:2+2"]}))

    corr_ids = {e.get("correlation_id") for e in events}
    assert len(corr_ids) == 1

    transitions = [
        (e["payload"]["from"], e["payload"]["to"])
        for e in events
        if e.get("event_type") == "STATE_TRANSITION"
    ]
    assert transitions[0] == ("AWAITING_INPUT", "DECOMPOSE_TASK")
    assert transitions[-1][1] == "RESPONDING_SUCCESS"

    starts = {e["span_id"] for e in events if e.get("event_type") == "TOOL_CALL_START"}
    ends = {
        e["span_id"]
        for e in events
        if e.get("event_type") in {"TOOL_CALL_OK", "TOOL_CALL_ERR"}
    }
    assert starts == ends

