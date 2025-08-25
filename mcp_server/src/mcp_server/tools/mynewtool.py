from __future__ import annotations

from typing import Any

from mcp_server.server import app


@app.tool(
    name="MyNewTool",
    description="Describe what MyNewTool does, required args, and return shape.",
)
async def MyNewTool(example_arg: str) -> dict[str, Any]:
    """Short doc for humans & LLM.
    Args:
        example_arg: what it is.

    Returns:
        dict with structured result for the client.
    """
    # TODO: implement. Keep scope narrow. Validate inputs.
    return {"ok": True, "arg": example_arg}
