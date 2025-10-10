"""Backward compatibility shim for src.mcp_server.github_tools."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.mcp_server.github_tools is deprecated; use src.mcp.server.github_tools instead. "
    "This shim will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.mcp.server.github_tools import *  # noqa: E402,F401,F403
