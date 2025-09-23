"""Local MCP compatibility shims for tests."""
from .registry import ToolDefinition, ToolRegistry, UnknownToolError

__all__ = ['ToolRegistry', 'UnknownToolError', 'ToolDefinition']
