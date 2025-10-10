"""Backward compatibility shim for src.super_alita_mcp.fastapi_server."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.super_alita_mcp.fastapi_server is deprecated; use src.mcp.integrations.super_alita.fastapi_server instead. "
    "This shim will be removed in v4.0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.mcp.integrations.super_alita.fastapi_server import *  # noqa: E402,F401,F403
