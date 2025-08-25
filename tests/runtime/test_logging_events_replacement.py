import json
import logging

import pytest

from src.plugins.core_utils_plugin_dynamic import CoreUtilsPlugin
from src.planning import sync_once
import scripts.todo_sync as todo_sync


class DummyEventBus:
    def __init__(self) -> None:
        self.events: list = []

    async def publish(self, event) -> None:  # pragma: no cover - simple stub
        self.events.append(event)

    async def subscribe(self, event_type, handler) -> None:  # pragma: no cover
        pass


class DummyStore:
    async def register_capabilities(self, name: str, data: dict) -> None:  # pragma: no cover
        self.name = name
        self.data = data


@pytest.mark.asyncio
async def test_core_utils_plugin_discovers_and_logs(caplog):
    caplog.set_level(logging.INFO)
    plugin = CoreUtilsPlugin()
    bus = DummyEventBus()
    store = DummyStore()

    await plugin.setup(bus, store, config={})
    await plugin.start()

    assert any("plugin started" in r.getMessage() for r in caplog.records)
    assert any(e.event_type == "capabilities_discovered" for e in bus.events)


@pytest.mark.asyncio
async def test_sync_once_main_logs(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    async def fake_sync() -> dict:
        return {
            "success": True,
            "metrics_processed": 1,
            "actions_taken": 1,
            "active_todos": 1,
            "risk_score": 0.1,
            "system_priority": "low",
            "elapsed_s": 0.01,
        }

    monkeypatch.setattr(sync_once, "sync_metrics_to_todos", fake_sync)
    exit_code: dict[str, int] = {}
    monkeypatch.setattr(sync_once.sys, "exit", lambda code: exit_code.setdefault("code", code))

    await sync_once.main()

    assert exit_code["code"] == 0
    assert any("Sync completed successfully" in r.getMessage() for r in caplog.records)


def test_todo_sync_logging(tmp_path, caplog):
    todo_file = tmp_path / "todos.json"
    todo_file.write_text(json.dumps({"todoList": [], "lastModified": ""}))
    caplog.set_level(logging.INFO)

    todo_sync.update_persistent_todos(todo_file, [{"title": "x"}])

    assert any("Updated 1 todos" in r.getMessage() for r in caplog.records)
