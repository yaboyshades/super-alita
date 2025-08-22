"""
Co-Architect MCP Integration Module
===================================

This module provides integration between Co-Architect mode and the Model Context Protocol (MCP).
It allows Co-Architect to register and use tools through the MCP server, enabling
contract-first tool design and declarative guardrails.

Usage:
    from src.core.mcp_integration import MCPIntegration

    # Initialize integration
    mcp = MCPIntegration()

    # Register tools
    await mcp.register_tool("tool_name", tool_schema, tool_handler)

    # Execute tools
    result = await mcp.execute_tool("tool_name", tool_params)
"""

import logging
import os
from collections.abc import Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default MCP server settings
DEFAULT_MCP_HOST = "localhost"
DEFAULT_MCP_PORT = 5678
DEFAULT_MCP_BASE_URL = f"http://{DEFAULT_MCP_HOST}:{DEFAULT_MCP_PORT}"

# HTTP status codes
HTTP_OK = 200

# Environment variable for MCP server URL
MCP_BASE_URL_ENV_VAR = "MCP_BASE_URL"


class MCPIntegration:
    """
    Integration with Model Context Protocol for Co-Architect mode.

    This class provides methods for registering and executing tools through
    the Model Context Protocol (MCP) server, enabling seamless integration
    between Co-Architect mode and custom tools.
    """

    def __init__(self, base_url: str | None = None):
        """
        Initialize MCP integration.

        Args:
            base_url: Base URL of the MCP server. If None, uses environment variable
                    or default.
        """
        self.base_url = base_url or os.environ.get(
            MCP_BASE_URL_ENV_VAR, DEFAULT_MCP_BASE_URL
        )
        self._tools: dict[str, dict[str, Any]] = {}
        self._tool_handlers: dict[str, Callable] = {}
        logger.info(f"Initialized MCP integration with base URL: {self.base_url}")

    async def ping(self) -> bool:
        """
        Ping the MCP server to check connectivity.

        Returns:
            True if the server is reachable, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == HTTP_OK
        except Exception:
            logger.exception("Failed to ping MCP server")
            return False

    async def register_tool(
        self, tool_name: str, tool_schema: dict[str, Any], handler: Callable
    ) -> bool:
        """
        Register a tool with the MCP server.

        Args:
            tool_name: Name of the tool.
            tool_schema: JSON Schema of the tool.
            handler: Function to handle tool execution.

        Returns:
            True if registration succeeded, False otherwise.
        """
        try:
            # Store tool handler locally
            self._tool_handlers[tool_name] = handler

            # Register tool schema with MCP server
            async with httpx.AsyncClient(timeout=10.0) as client:
                payload = {
                    "name": tool_name,
                    "schema": tool_schema,
                    "description": tool_schema.get("description", f"Tool: {tool_name}"),
                }
                response = await client.post(
                    f"{self.base_url}/tools/register", json=payload
                )

                if response.status_code == HTTP_OK:
                    logger.info(
                        f"Successfully registered tool '{tool_name}' with MCP server"
                    )
                    self._tools[tool_name] = tool_schema
                    return True

                logger.error(f"Failed to register tool '{tool_name}': {response.text}")
                return False

        except Exception:
            logger.exception(f"Error registering tool '{tool_name}'")
            return False

    async def execute_tool(
        self, tool_name: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute a tool through the MCP server.

        Args:
            tool_name: Name of the tool to execute.
            params: Parameters to pass to the tool.

        Returns:
            Tool execution result.
        """
        try:
            # Check if tool exists and has a handler
            if tool_name not in self._tool_handlers:
                return {"error": f"Tool '{tool_name}' not registered"}

            handler = self._tool_handlers[tool_name]

            # Execute tool locally
            result = await handler(**params)
        except Exception as e:
            logger.exception(f"Error executing tool '{tool_name}'")
            return {"tool_name": tool_name, "success": False, "error": str(e)}

        # Return formatted result for MCP
        return {"tool_name": tool_name, "success": True, "result": result}

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        List all registered tools from the MCP server.

        Returns:
            List of registered tools with their schemas.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/tools")

                if response.status_code == HTTP_OK:
                    return response.json()

                logger.error(f"Failed to list tools: {response.text}")
                return []
        except Exception:
            logger.exception("Error listing tools")
            return []

    async def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down MCP integration")
        self._tools.clear()
        self._tool_handlers.clear()


async def create_mcp_integration() -> MCPIntegration:
    """
    Create and initialize an MCP integration instance.

    Returns:
        Initialized MCPIntegration instance.
    """
    integration = MCPIntegration()

    # Try to ping the server
    is_connected = await integration.ping()
    if not is_connected:
        logger.warning("MCP server is not reachable. Make sure it's running.")
    else:
        logger.info("Successfully connected to MCP server")

    return integration
