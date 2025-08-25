#!/usr/bin/env python3
"""Simple working MCP server test for Super Alita Agent"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test MCP imports first
print("Testing MCP imports...")
try:
    from mcp.server import Server
    from mcp.types import CallToolRequest, CallToolResult, TextContent, Tool

    print("✅ MCP imports successful")
    MCP_AVAILABLE = True
except ImportError as e:
    print(f"❌ MCP import failed: {e}")
    MCP_AVAILABLE = False
    sys.exit(1)

# Test agent imports
print("Testing agent imports...")
try:
    from vscode_integration.agent_integration import SuperAlitaAgent

    print("✅ Agent imports successful")
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Agent import failed: {e}")
    AGENT_AVAILABLE = False


class SimpleAgentMcpServer:
    """Simplified version of the agent MCP server for testing"""

    def __init__(self, workspace_folder: str | None = None):
        self.workspace_folder = workspace_folder or str(Path.cwd())
        self.agent: SuperAlitaAgent | None = None
        self.server: Server | None = None

    def initialize_agent(self) -> bool:
        """Initialize the Super Alita Agent"""
        if not AGENT_AVAILABLE:
            print("⚠️ Agent not available, running without agent functionality")
            return False

        try:
            print("🚀 Initializing Super Alita Agent...")
            self.agent = SuperAlitaAgent(self.workspace_folder)
            return True
        except Exception as e:
            print(f"❌ Agent initialization error: {e}")
            return False

    def create_mcp_server(self) -> Server:
        """Create and configure the MCP server."""
        if not MCP_AVAILABLE:
            raise RuntimeError("MCP package not available")

        server = Server("super-alita-agent-simple")

        @server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools provided by Super Alita Agent."""
            return [
                Tool(
                    name="ping",
                    description="Simple ping test to verify server is working",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Message to echo back",
                            }
                        },
                    },
                ),
            ]

        @server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict[str, any] | None
        ) -> CallToolResult:
            """Handle tool execution requests."""
            args = arguments or {}

            if name == "ping":
                message = args.get("message", "pong")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Echo: {message}")]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                    isError=True,
                )

        return server


async def main():
    """Main entry point for the simple MCP server"""
    print("🚀 Simple Agent MCP Server Starting...")

    # Get workspace folder from environment or use current directory
    workspace_folder = Path.cwd()
    print(f"Workspace: {workspace_folder}")

    # Initialize server
    mcp_server = SimpleAgentMcpServer(str(workspace_folder))

    # Initialize agent
    agent_ready = mcp_server.initialize_agent()
    print(f"Agent ready: {agent_ready}")

    # Create MCP server
    server = mcp_server.create_mcp_server()
    print("✅ MCP server created successfully")

    # Test the server
    print("\n📋 Available tools:")
    tools = await server._handle_list_tools()
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    print("\n🧪 Testing ping tool:")
    result = await server._handle_call_tool("ping", {"message": "Hello from MCP!"})
    print(f"Result: {result}")

    print("\n✅ Simple MCP server test complete!")


if __name__ == "__main__":
    asyncio.run(main())
