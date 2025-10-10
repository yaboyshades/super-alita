"""Backward compatibility shim for src.mcp_local.registry."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.mcp_local.registry is deprecated; use src.mcp.client.tool_factory instead. "
    "This shim will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.mcp.client.tool_factory import *  # noqa: E402,F401,F403
