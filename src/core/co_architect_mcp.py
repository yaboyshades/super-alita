"""
Co-Architect MCP Connector Module
=================================

This module connects the Co-Architect mode with MCP integration,
allowing the Co-Architect to use MCP tools and implement
contract-first tool design patterns.

Usage:
    from src.core.co_architect_mcp import CoArchitectMCPConnector

    # Initialize connector
    connector = CoArchitectMCPConnector()

    # Start the connector
    await connector.start()

    # Register a tool
    await connector.register_tool("my_tool", tool_schema, tool_handler)

    # Execute a tool
    result = await connector.execute_tool("my_tool", tool_params)
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from src.core.co_architect_mode import CoArchitectMode
from src.core.mcp_integration import MCPIntegration, create_mcp_integration
from src.core.neural_atom import NeuralAtomMetadata, TextualMemoryAtom
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class CoArchitectMCPConnector:
    """
    Connector between Co-Architect mode and MCP integration.

    This class provides a bridge between Co-Architect mode and MCP integration,
    allowing Co-Architect to register and use tools through the MCP server,
    implementing contract-first tool design patterns.
    """

    def __init__(
        self,
        mcp_integration: MCPIntegration | None = None,
        co_architect: CoArchitectMode | None = None,
    ):
        """
        Initialize Co-Architect MCP connector.

        Args:
            mcp_integration: MCPIntegration instance. If None, a new one will be created.
            co_architect: CoArchitectMode instance. If None, a new one will be created.
        """
        self._mcp = mcp_integration
        self._co_architect = co_architect
        self._registered_tools: set[str] = set()
        self._initialized = False
        memory_metadata = NeuralAtomMetadata(
            name="co_architect_mcp_memory",
            description="Memory for Co-Architect MCP connector",
            capabilities=["mcp_connector", "tool_registration"],
        )
        self._memory_atom = TextualMemoryAtom(memory_metadata, "")

    async def start(self) -> None:
        """
        Start the connector.

        Initializes MCP integration and Co-Architect mode if they weren't provided.
        """
        if self._initialized:
            logger.warning("Co-Architect MCP connector already initialized")
            return

        # Initialize MCP integration if not provided
        if self._mcp is None:
            logger.info("Initializing MCP integration")
            self._mcp = await create_mcp_integration()

        # Initialize Co-Architect mode if not provided
        if self._co_architect is None:
            logger.info("Initializing Co-Architect mode")
            self._co_architect = CoArchitectMode()

        # Record initialization in memory
        self._memory_atom.store(
            {
                "event_type": "initialization",
                "timestamp": asyncio.get_event_loop().time(),
                "mcp_base_url": self._mcp.base_url,
            }
        )

        self._initialized = True
        logger.info("Co-Architect MCP connector initialized")

    async def register_tool(
        self, tool_name: str, tool_schema: dict[str, Any], handler: Callable
    ) -> bool:
        """
        Register a tool with both MCP and Co-Architect mode.

        Args:
            tool_name: Name of the tool.
            tool_schema: JSON Schema of the tool.
            handler: Function to handle tool execution.

        Returns:
            True if registration succeeded, False otherwise.
        """
        if not self._initialized:
            await self.start()

        # Register with MCP
        success = await self._mcp.register_tool(tool_name, tool_schema, handler)

        if success:
            # Add to registered tools
            self._registered_tools.add(tool_name)

            # Record registration in memory
            self._memory_atom.store(
                {
                    "event_type": "tool_registration",
                    "timestamp": asyncio.get_event_loop().time(),
                    "tool_name": tool_name,
                    "success": True,
                }
            )

            logger.info(
                f"Successfully registered tool '{tool_name}' with MCP and Co-Architect"
            )
        else:
            # Record failed registration in memory
            self._memory_atom.store(
                {
                    "event_type": "tool_registration",
                    "timestamp": asyncio.get_event_loop().time(),
                    "tool_name": tool_name,
                    "success": False,
                }
            )

            logger.error(f"Failed to register tool '{tool_name}'")

        return success

    async def execute_tool(
        self, tool_name: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute a tool through MCP.

        Args:
            tool_name: Name of the tool to execute.
            params: Parameters to pass to the tool.

        Returns:
            Tool execution result.
        """
        if not self._initialized:
            await self.start()

        # Record execution attempt in memory
        self._memory_atom.store(
            {
                "event_type": "tool_execution_attempt",
                "timestamp": asyncio.get_event_loop().time(),
                "tool_name": tool_name,
                "params": params,
            }
        )

        # Execute tool through MCP
        result = await self._mcp.execute_tool(tool_name, params)

        # Record execution result in memory
        self._memory_atom.store(
            {
                "event_type": "tool_execution_result",
                "timestamp": asyncio.get_event_loop().time(),
                "tool_name": tool_name,
                "success": result.get("success", False),
            }
        )

        return result

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        List all registered tools.

        Returns:
            List of registered tools with their schemas.
        """
        if not self._initialized:
            await self.start()

        return await self._mcp.list_tools()

    async def get_registered_tool_names(self) -> list[str]:
        """
        Get names of tools registered through this connector.

        Returns:
            List of tool names.
        """
        return list(self._registered_tools)

    async def get_memory_records(
        self, event_type: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Get memory records from the connector.

        Args:
            event_type: Type of events to filter by. If None, all events are returned.
            limit: Maximum number of records to return.

        Returns:
            List of memory records.
        """
        records = self._memory_atom.retrieve(limit)

        if event_type is not None:
            records = [r for r in records if r.get("event_type") == event_type]

        return records[:limit]

    async def shutdown(self) -> None:
        """
        Shutdown the connector and its components.
        """
        if self._initialized:
            # Record shutdown in memory
            self._memory_atom.store(
                {"event_type": "shutdown", "timestamp": asyncio.get_event_loop().time()}
            )

            # Shutdown MCP integration
            if self._mcp is not None:
                await self._mcp.shutdown()

            logger.info("Co-Architect MCP connector shut down")
            self._initialized = False


class CoArchitectMCPPlugin(PluginInterface):
    """
    Plugin for Co-Architect mode that integrates with MCP.

    This plugin provides tools and commands for Co-Architect mode
    to interact with MCP and use MCP tools.
    """

    def __init__(self):
        """Initialize Co-Architect MCP plugin."""
        super().__init__()
        self.name = "co_architect_mcp"
        self.description = "MCP integration for Co-Architect mode"
        self.connector = None

    async def initialize(self) -> bool:
        """
        Initialize the plugin.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        try:
            self.connector = CoArchitectMCPConnector()
            await self.connector.start()
        except Exception:
            logger.exception("Failed to initialize Co-Architect MCP plugin")
            return False
        return True

    async def process_event(
        self, event_type: str, event_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Process an event from Co-Architect mode.

        Args:
            event_type: Type of event.
            event_data: Event data.

        Returns:
            Response data for the event.
        """
        if event_type == "tool_request":
            # Handle tool request
            tool_name = event_data.get("tool_name")
            params = event_data.get("params", {})

            if not tool_name:
                return {"success": False, "error": "Tool name not provided"}

            result = await self.connector.execute_tool(tool_name, params)
            return {"success": True, "result": result}

        if event_type == "tool_list_request":
            # Handle tool list request
            tools = await self.connector.list_tools()
            return {"success": True, "tools": tools}

        # Default response for unhandled events
        return {"success": False, "error": f"Unhandled event type: {event_type}"}

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        if self.connector is not None:
            await self.connector.shutdown()
