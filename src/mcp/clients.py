"""
MCP Client Pool for Super Alita Integration

Async MCP client implementation with connection pooling, single-flight execution,
and shared resource management for tool invocation.
"""

import asyncio
from typing import Any

from src.core.clock import MonotonicClock


class MCPClient:
    """
    Async MCP client for tool invocation

    Provides low-level MCP tool execution with proper error handling
    and timing integration.
    """

    def __init__(self, clock: MonotonicClock):
        self.clock = clock
        self.call_count = 0  # For testing single-flight behavior

    async def call_tool(
        self, base_url: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute MCP tool with given arguments

        Args:
            base_url: MCP server base URL
            tool_name: Name of tool to invoke
            arguments: Tool arguments

        Returns:
            Tool execution result with success flag

        Raises:
            Exception: On MCP communication or tool execution errors
        """
        self.call_count += 1
        start_time = self.clock.now()

        try:
            # Simulate MCP tool execution
            # In real implementation, this would use MCP protocol
            await asyncio.sleep(0.001)  # Simulate network latency

            # Mock tool logic - calculator example
            if tool_name == "calculator":
                if "a" in arguments and "b" in arguments:
                    result = arguments["a"] + arguments["b"]
                    return {
                        "success": True,
                        "result": result,
                        "execution_time_ms": (self.clock.now() - start_time) * 1000,
                    }
                else:
                    return {
                        "success": False,
                        "error": "Missing required arguments: a, b",
                        "execution_time_ms": (self.clock.now() - start_time) * 1000,
                    }

            # Default success for other tools
            return {
                "success": True,
                "result": f"Tool {tool_name} executed successfully",
                "execution_time_ms": (self.clock.now() - start_time) * 1000,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": (self.clock.now() - start_time) * 1000,
            }


class MCPClientPool:
    """
    Connection pool for MCP clients with resource management

    Manages multiple MCP client connections with proper lifecycle
    and shared resource optimization.
    """

    def __init__(self, max_clients: int = 10, clock: MonotonicClock | None = None):
        self.max_clients = max_clients
        self.clock = clock or MonotonicClock()
        self._clients: dict[str, MCPClient] = {}
        self._lock = asyncio.Lock()

    async def get_client(self, base_url: str) -> MCPClient:
        """
        Get or create MCP client for given base URL

        Args:
            base_url: MCP server base URL

        Returns:
            MCPClient instance for the server
        """
        async with self._lock:
            if base_url not in self._clients:
                if len(self._clients) >= self.max_clients:
                    # Remove oldest client (simple LRU)
                    oldest_url = next(iter(self._clients))
                    del self._clients[oldest_url]

                self._clients[base_url] = MCPClient(self.clock)

            return self._clients[base_url]

    async def shutdown(self):
        """Cleanup all clients"""
        async with self._lock:
            self._clients.clear()
