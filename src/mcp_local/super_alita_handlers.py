"""Backward compatibility shim for src.mcp_local.super_alita_handlers."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.mcp_local.super_alita_handlers is deprecated; use src.mcp.integrations.super_alita.handlers instead. "
    "This shim will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.mcp.integrations.super_alita.handlers import *  # noqa: F401,F403
