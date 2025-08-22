import asyncio

from reug.events import EventEmitter
from reug.services import create_services
from reug.fsm import ExecutionFlow
from reug.kg import store as kg_store
from reug.tools import registry as tool_registry


def _missing_tool(step, ctx):
    return None, step.args


def test_dynamic_tool_and_kg(tmp_path, monkeypatch):
    monkeypatch.setenv("REUG_EVENT_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("REUG_TOOL_REGISTRY_DIR", str(tmp_path / "registry"))
    monkeypatch.setattr(kg_store, "TRIPLE_PATH", str(tmp_path / "kg" / "triples.jsonl"))

    events = []
    emitter = EventEmitter(events.append)
    services = create_services(emitter, tool_resolver=_missing_tool)
    flow = ExecutionFlow(services, emitter)

    asyncio.run(flow.run({"sot": ["compute:2+2"]}))

    reg = tool_registry.load_registry()
    assert len(reg) == 1

    ok_event = next(e for e in events if e["event_type"] == "TOOL_CALL_OK")
    triples = kg_store.query()
    assert triples and triples[0]["source_event_id"] == ok_event["event_id"]

