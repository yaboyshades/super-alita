"""
Event builder helpers for Super Alita.
Ensures required fields are present and typed correctly for ToolCallEvent creation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from src.core.events import ToolCallEvent


def build_tool_call_event(
    *,
    source_plugin: str,
    tool_name: str,
    parameters: Mapping[str, Any] | None,
    conversation_id: str | None,
    session_id: str | None,
    message_id: str | None = None,
    tool_call_id: str | None = None,
) -> ToolCallEvent:
    """
    Construct a valid ToolCallEvent with safe defaults.
    - Ensures all required fields are present
    - Generates ids with uuid4 when missing
    """
    params = dict(parameters or {})

    # Id defaults
    call_id = tool_call_id or f"tool_{uuid4()}"

    return ToolCallEvent(
        source_plugin=source_plugin,
        conversation_id=conversation_id or "default",
        session_id=session_id or "default",
        tool_call_id=call_id,
        tool_name=tool_name,
        parameters=params,
    )
