"""Unified MCP package for Super Alita."""
from __future__ import annotations

from .client import MCPClient, MCPClientPool, MCPRouter
from .client import ToolRegistry as ClientToolRegistry
from .integrations import (
    GitHubIntegrationStatus,
    GitHubMCPIntegration,
    GitHubOperationReport,
)
from .protocol import (
    MCPCacheEvent,
    MCPCircuitBreakerEvent,
    MCPProvenanceEvent,
    MCPResult,
    MCPToolFailure,
    MCPToolInvocation,
    MCPToolResult,
    MCPToolSuccess,
    Result,
)
from .registry import ToolDefinition, UnknownToolError
from .registry import ToolRegistry as LegacyToolRegistry
from .server import FastMCP, SuperAlitaMCPServer, app, register_github_tools

__all__ = [
    "FastMCP",
    "SuperAlitaMCPServer",
    "app",
    "register_github_tools",
    "MCPClient",
    "MCPClientPool",
    "MCPRouter",
    "ClientToolRegistry",
    "LegacyToolRegistry",
    "ToolDefinition",
    "UnknownToolError",
    "MCPCacheEvent",
    "MCPCircuitBreakerEvent",
    "MCPProvenanceEvent",
    "MCPResult",
    "MCPToolFailure",
    "MCPToolInvocation",
    "MCPToolResult",
    "MCPToolSuccess",
    "Result",
    "GitHubMCPIntegration",
    "GitHubIntegrationStatus",
    "GitHubOperationReport",
]