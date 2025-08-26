"""Simple MCP tool providing basic string statistics."""

from __future__ import annotations

from typing import Any

from mcp_server.server import app


@app.tool(
    name="string_info",
    description="Return uppercase variant and length for a non-empty string.",
)
async def MyNewTool(text: str) -> dict[str, Any]:
    """Compute basic information about ``text``.

    Args:
        text: Non-empty string to analyze.

    Returns:
        A mapping with the original string, an uppercase version, and its length.
        If validation fails, the mapping contains an ``error`` message instead.
    """
    if not isinstance(text, str):
        return {"error": "text must be a string"}
    if not text.strip():
        return {"error": "text must be a non-empty string"}
    return {"original": text, "upper": text.upper(), "length": len(text)}

