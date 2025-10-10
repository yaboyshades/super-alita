"""Backward compatibility shim for src.super_alita_mcp.super_alita."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.super_alita_mcp.super_alita is deprecated; use src.mcp.integrations.super_alita instead. "
    "This shim will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.mcp.integrations.super_alita import *  # noqa: F401,F403
