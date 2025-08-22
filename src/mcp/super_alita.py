"""
Register Super Alita handlers with the MCP server.

This module initializes the Super Alita handlers and includes them in the MCP server.
"""

from fastapi import FastAPI

from .super_alita_handlers import router as super_alita_router


def register_super_alita_handlers(app: FastAPI) -> None:
    """Register Super Alita handlers with the FastAPI app."""
    app.include_router(super_alita_router)
