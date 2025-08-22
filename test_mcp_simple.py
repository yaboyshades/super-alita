#!/usr/bin/env python3
"""Simple MCP server test to verify API usage"""

import asyncio

from mcp.server import Server
from mcp.types import Tool


async def main():
    # Test basic server creation
    server = Server("test-server")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [Tool(name="test_tool", description="A test tool")]

    print("✅ Server created successfully")
    print(f"Server name: {server.name}")


if __name__ == "__main__":
    asyncio.run(main())
