"""Tests for the ToolCatalogService."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from reug_runtime.tools.service import ToolCatalogService


class _ListRegistry:
    """Simple registry stub that advertises one dynamic tool."""

    def get_available_tools_schema(self) -> list[dict[str, object]]:
        return [
            {
                "tool_id": "dynamic_echo",
                "description": "Echo payloads back to the caller",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            }
        ]


class _HeuristicRegistry:
    """Registry stub that records registrations and executions."""

    def __init__(self) -> None:
        self._known: set[str] = set()
        self.registered: list[dict[str, object]] = []
        self.executors: dict[
            str, Callable[[dict[str, object]], Awaitable[dict[str, object]]]
        ] = {}
        self.executed: list[dict[str, object]] = []

    def knows(self, tool_name: str) -> bool:
        return tool_name in self._known

    def register_tool(
        self,
        contract: dict[str, object],
        executor: Callable[[dict[str, object]], Awaitable[dict[str, object]]],
    ) -> None:
        self.registered.append(contract)
        tool_id = contract.get("tool_id") or contract.get("name")
        if isinstance(tool_id, str):
            self._known.add(tool_id)
            self.executors[tool_id] = executor

    async def execute(
        self, tool_name: str, args: dict[str, object]
    ) -> dict[str, object]:
        self.executed.append({"tool": tool_name, "args": dict(args)})
        return {"tool": tool_name, "args": dict(args)}

    def get_available_tools_schema(self) -> list[dict[str, object]]:
        return []


def test_list_tools_includes_dynamic(tmp_path: Path) -> None:
    """Dynamic tools from the registry should be merged into the catalog."""
    registry = _ListRegistry()
    service = ToolCatalogService(mcp_box_dir=str(tmp_path))

    tools = service.list_tools(registry)

    names = {tool["name"] for tool in tools}
    assert "reug_start_turn" in names  # static catalog entry
    assert "dynamic_echo" in names  # dynamic entry from registry


def test_register_dynamic_tool_persists_spec(tmp_path: Path) -> None:
    """Dynamic registrations should persist specs to the MCP box directory."""
    service = ToolCatalogService(mcp_box_dir=str(tmp_path))
    spec = {
        "tool_id": "runtime_helper",
        "description": "Example tool",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
    }

    tool_id = service.register_dynamic_tool(spec)

    persisted = tmp_path / f"{tool_id}.json"
    assert persisted.exists()
    data = json.loads(persisted.read_text(encoding="utf-8"))
    assert data["tool_id"] == "runtime_helper"
    assert data["description"] == "Example tool"


def test_ensure_tool_registered_github(tmp_path: Path) -> None:
    """GitHub heuristics should auto-register fetch proxies and persist specs."""
    registry = _HeuristicRegistry()
    service = ToolCatalogService(mcp_box_dir=str(tmp_path))

    ensured = service.ensure_tool_registered(
        "fetch_repo_file",
        {"owner": "octocat", "repo": "hello", "path": "README.md"},
        registry,
    )

    assert ensured is True
    assert any(
        contract["tool_id"] == "fetch_repo_file" for contract in registry.registered
    )

    executor = registry.executors.get("fetch_repo_file")
    assert executor is not None

    # Executor should proxy to registry.execute and preserve arguments
    payload: dict[str, object] = {
        "owner": "octocat",
        "repo": "hello",
        "path": "README.md",
    }

    async def _invoke() -> dict[str, object]:
        return await executor(payload)

    result = asyncio.run(_invoke())
    assert result["tool"] == "fetch_github_raw"
    assert result["args"]["owner"] == "octocat"

    # Spec should be persisted alongside dynamic registration
    persisted = tmp_path / "fetch_repo_file.json"
    assert persisted.exists()


def test_ensure_tool_registered_fallback(tmp_path: Path) -> None:
    """Unknown tools should fall back to the planning helper."""
    registry = _HeuristicRegistry()
    service = ToolCatalogService(mcp_box_dir=str(tmp_path))

    ensured = service.ensure_tool_registered("mystery_tool", {"task": "demo"}, registry)

    assert ensured is True
    executor = registry.executors.get("mystery_tool")
    assert executor is not None

    async def _run_executor() -> dict[str, object]:
        return await executor({"task": "demo"})

    result = asyncio.run(_run_executor())
    assert result["steps"][0].startswith("Understand: demo")
