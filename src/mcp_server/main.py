"""Backward compatibility shim for src.mcp_server.main."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.mcp_server.main is deprecated; use src.mcp.server.main instead. "
    "This shim will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.mcp.server.main import *  # noqa: E402,F401,F403
