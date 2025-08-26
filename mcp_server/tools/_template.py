"""Template for creating new MCP tools.

Copy this file and replace placeholders with your tool logic.
"""

from __future__ import annotations

from typing import Any

from mcp_server.server import app


@app.tool(name="tool_name", description="Short description of the tool.")
async def tool_name(exampleParam: str) -> dict[str, Any]:
    """Explain the tool.

    Args:
        exampleParam: Description of the parameter.

    Returns:
        {"resultKey": "description"}  # Result schema hint.
    """
    raise NotImplementedError("Implement your tool logic")
