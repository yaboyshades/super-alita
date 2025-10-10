"""Super Alita integration helpers for the unified MCP package."""

from __future__ import annotations

from fastapi import FastAPI

from .capabilities_tool import collect_capabilities
from .handlers import router
from .registry import ToolRegistry

# Alias for backward compatibility
SuperAlitaCapabilityRegistry = ToolRegistry

__all__ = [
    "register_super_alita_handlers",
    "router",
    "SuperAlitaCapabilityRegistry",
    "ToolRegistry",
    "collect_capabilities",
]


def register_super_alita_handlers(app: FastAPI) -> None:
    """Attach the Super Alita router to the provided FastAPI app."""
    app.include_router(router)
