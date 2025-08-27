"""Tests for plugin capability collection runtime status."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any


class _FakePlugin:
    def __init__(self, name: str, running: bool) -> None:
        self.name = name
        self.is_running = running


def _setup_plugin_loader(monkeypatch, manifest: list[dict[str, Any]]) -> None:
    """Patch plugin loader helpers to return a static manifest."""

    info = {p["name"]: p | {"category": "uncategorized"} for p in manifest}

    import src.core.plugin_loader as loader

    monkeypatch.setattr(loader, "load_plugin_manifest", lambda path=None: manifest)
    monkeypatch.setattr(loader, "get_plugin_info", lambda m: info)


def test_collect_plugin_capabilities_reports_runtime_state(monkeypatch):
    """Active and inactive plugins should be reported correctly."""

    manifest = [
        {
            "name": "active",
            "module": "mod:Active",
            "enabled": True,
            "priority": 1,
            "depends_on": [],
            "description": "Active plugin",
        },
        {
            "name": "configured_only",
            "module": "mod:Configured",
            "enabled": True,
            "priority": 2,
            "depends_on": [],
            "description": "Configured plugin",
        },
        {
            "name": "loaded_only",
            "module": "mod:Loaded",
            "enabled": True,
            "priority": 3,
            "depends_on": [],
            "description": "Loaded but not running",
        },
    ]

    _setup_plugin_loader(monkeypatch, manifest)

    runtime_module = ModuleType("runtime_orchestrator")
    runtime_module.plugins = {
        "active": _FakePlugin("active", True),
        "loaded_only": _FakePlugin("loaded_only", False),
    }
    sys.modules["runtime_orchestrator"] = runtime_module

    from src.super_alita_mcp.capabilities_tool import _collect_plugin_capabilities

    try:
        result = _collect_plugin_capabilities()
    finally:
        del sys.modules["runtime_orchestrator"]

    by_name = {p["name"]: p for p in result}

    assert by_name["active"]["loaded"] is True
    assert by_name["active"]["running"] is True
    assert by_name["active"]["status"] == "running"

    assert by_name["configured_only"]["loaded"] is False
    assert by_name["configured_only"]["running"] is False
    assert by_name["configured_only"]["status"] == "configured"

    assert by_name["loaded_only"]["loaded"] is True
    assert by_name["loaded_only"]["running"] is False
    assert by_name["loaded_only"]["status"] == "loaded"
