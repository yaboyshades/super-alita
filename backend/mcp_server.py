"""Minimal MCP server exposing search and fetch tools.

This module provides :func:`create_mcp_server` which returns a
:class:`FastMCP` instance.  The server registers two tools:

``search(query: str) -> list[str]``
    Return IDs of items whose description contains the query text.

``fetch(item_id: str) -> dict``
    Fetch the metadata for a given item ID.

The implementation uses an in-memory data set so the tools remain pure
and easy to test.  The module can be executed directly to start a stdio
MCP server:

``python backend/mcp_server.py``
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

# In-memory catalogue used by the search/fetch tools.  Keeping the data
# local avoids network access during tests.
_DATA: dict[str, dict] = {
    "1": {"title": "Fix login bug"},
    "2": {"title": "Add dark mode"},
    "3": {"title": "Improve documentation"},
}

app = FastMCP("backend-mcp")


@app.tool()
async def search(query: str) -> list[str]:
    """Return IDs of entries whose title contains ``query``."""
    q = query.lower()
    return [key for key, meta in _DATA.items() if q in meta["title"].lower()]


@app.tool()
async def fetch(item_id: str) -> dict[str, str]:
    """Return metadata for ``item_id``.

    Raises ``KeyError`` if the ID is unknown.
    """
    if item_id not in _DATA:
        raise KeyError(item_id)
    return _DATA[item_id]


def create_mcp_server() -> FastMCP:
    """Return the configured MCP server."""
    return app


if __name__ == "__main__":  # pragma: no cover - manual run helper
    create_mcp_server().run(transport="stdio")
