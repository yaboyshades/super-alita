"""Backward compatibility shim for src.mcp_server.tools."""
from __future__ import annotations

import warnings

warnings.warn(
    "src.mcp_server.tools is deprecated; use src.mcp.server.tools instead. "
    "This shim will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.mcp.server.tools import *  # noqa: F401,F403
