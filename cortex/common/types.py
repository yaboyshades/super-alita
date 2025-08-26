from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: dict[str, Any]  # JSON-schema-like


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]


@dataclass
class PromptBundle:
    system: str
    tools: list[ToolSpec]
    history: list[dict[str, Any]]
    context: str | None = None
    reminders: str | None = None
    metadata: dict[str, Any] | None = None
