"""Lightweight package init to avoid heavy imports at import time.

This allows importing submodules like ``reug_runtime.llm_client`` without
triggering the full router import (which may fail during partial checkouts
or conflict resolution). Routers are available via lazy attribute access.
"""

__all__ = ["router", "tools", "TOOL_CATALOG"]

def __getattr__(name: str):  # pragma: no cover - small helper
    if name == "router":
        from .router import router as _router

        return _router
    if name in {"tools", "TOOL_CATALOG"}:
        from .router_tools import tools as _tools, TOOL_CATALOG as _CAT

        return _tools if name == "tools" else _CAT
    raise AttributeError(name)
