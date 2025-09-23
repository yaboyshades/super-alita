"""Minimal MCP registry shim for tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolDefinition:
    name: str
    description: str = ""
    spec: dict[str, Any] | None = None

class UnknownToolError(KeyError):
    pass

class ToolRegistry:
    """Simple in-memory registry used in test environments."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        self._tools[definition.name] = definition

    def knows(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise UnknownToolError(name)
        return self._tools[name]
