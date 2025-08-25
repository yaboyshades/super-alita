"""Typed definitions for tool routing/execution.

Provides strong typing to eliminate "partially unknown" diagnostics
and make plugin <-> execution flow contracts explicit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TypedDict, runtime_checkable


class ToolFunctionSpec(TypedDict, total=False):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolSpec(TypedDict, total=False):
    name: str
    description: str
    type: str
    function: ToolFunctionSpec
    plugin_name: str
    sot_step_id: str


@runtime_checkable
class ToolProvidingPlugin(Protocol):  # pragma: no cover - structural typing helper
    """Protocol for plugins that expose tooling interfaces used by the REUG flow."""

    name: str

    def get_tools(self) -> list[ToolSpec]:  # noqa: D401
        ...
    def route_tools(
        self,
        all_tools: Sequence[ToolSpec],
        user_input: str,
        memory_context: dict[str, Any],
    ) -> list[ToolSpec]:  # pragma: no cover - simple selection wrapper
        ...
    async def process_request(  # Optional; some plugins may expose this
        self, user_input: str, memory_context: dict[str, Any]
    ) -> Any:  # noqa: D401
        ...  # type: ignore[override]
