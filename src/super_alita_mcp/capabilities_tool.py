"""Backward compatibility shim for src.super_alita_mcp.capabilities_tool."""
from __future__ import annotations

import warnings

warnings.warn(
    "src.super_alita_mcp.capabilities_tool is deprecated; use src.mcp.integrations.super_alita.capabilities_tool instead. "
    "This shim will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.mcp.integrations.super_alita.capabilities_tool import *  # noqa: F401,F403
