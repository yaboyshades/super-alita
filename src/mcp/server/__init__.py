"""Unified MCP server package."""

from __future__ import annotations

from .fastmcp import FastMCP
from .main import SuperAlitaMCPServer
from .mcp_server import app, register_github_tools

__all__ = [
    "FastMCP",
    "app",
    "register_github_tools",
    "SuperAlitaMCPServer",
]
