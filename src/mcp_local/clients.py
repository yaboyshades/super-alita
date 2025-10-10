"""Backward compatibility shim for src.mcp_local.clients."""
from __future__ import annotations

import warnings

warnings.warn(
    "src.mcp_local.clients is deprecated; use src.mcp.client.mcp_client instead. "
    "This shim will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.mcp.client.mcp_client import *  # noqa: F401,F403
