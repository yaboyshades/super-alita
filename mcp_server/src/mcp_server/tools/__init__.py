"""MCP tools package with optional resource and prompt registration."""

from toolforge import PROMPTS, RESOURCES, register_with

from mcp_server.server import app

register_with(app, RESOURCES, PROMPTS)
