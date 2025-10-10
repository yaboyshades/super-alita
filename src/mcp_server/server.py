"""Backward compatibility shim for src.mcp_server.server."""
from __future__ import annotations

import warnings

warnings.warn(
    "src.mcp_server.server is deprecated; use src.mcp.server.mcp_server instead. "
    "This shim will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.mcp.server.mcp_server import *  # noqa: F401,F403
