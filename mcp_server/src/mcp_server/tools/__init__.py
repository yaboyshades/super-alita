"""MCP tools package with optional resource and prompt registration."""

from mcp_server.server import app
from toolforge import PROMPTS, RESOURCES, register_with

register_with(app, RESOURCES, PROMPTS)
