"""Backward compatibility shim for src.mcp_local.router."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.mcp_local.router is deprecated; use src.mcp.client.router instead. "
    "This shim will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.mcp.client.router import *  # noqa: E402,F401,F403
