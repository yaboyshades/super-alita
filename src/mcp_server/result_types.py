"""Backward compatibility shim for src.mcp_server.result_types."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.mcp_server.result_types is deprecated; use src.mcp.protocol.result_types instead. "
    "This shim will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.mcp.protocol.result_types import *  # noqa: F401,F403
