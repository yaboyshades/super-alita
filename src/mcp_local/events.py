"""Backward compatibility shim for src.mcp_local.events."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.mcp_local.events is deprecated; use src.mcp.protocol.events instead. "
    "This shim will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.mcp.protocol.events import *  # noqa: E402,F401,F403
