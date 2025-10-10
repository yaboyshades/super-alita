"""Deprecated MCP server package shim."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.mcp_server is deprecated; use src.mcp.server instead. "
    "This shim will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.mcp.server import  # noqa: E402 *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
