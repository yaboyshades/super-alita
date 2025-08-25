#!/usr/bin/env python3
"""Test MCP server connectivity and basic functionality."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any


async def test_mcp_server() -> dict[str, Any]:
    """Test MCP server by sending a simple request."""
    try:
        # Start the MCP server as a subprocess
        mcp_cmd = [
            sys.executable,
            str(Path(__file__).parent / "mcp_server_wrapper.py"),
        ]

        process = await asyncio.create_subprocess_exec(
            *mcp_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Send a simple test request
        test_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        request_data = json.dumps(test_request) + "\n"

        # Send request and wait for response with timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(request_data.encode()), timeout=5.0
            )

            return {
                "success": True,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "returncode": process.returncode,
            }

        except TimeoutError:
            process.kill()
            await process.wait()
            return {
                "success": False,
                "error": "MCP server timeout after 5 seconds",
                "returncode": -1,
            }

    except Exception as e:
        return {"success": False, "error": str(e), "returncode": -1}


async def main():
    """Main test function."""
    print("Testing MCP server connectivity...")
    result = await test_mcp_server()

    if result["success"]:
        print("✅ MCP server responded successfully")
        if result.get("stdout"):
            print(f"Stdout: {result['stdout'][:200]}...")
    else:
        print(f"❌ MCP server test failed: {result.get('error', 'Unknown error')}")
        if result.get("stderr"):
            print(f"Stderr: {result['stderr'][:200]}...")

    return result


if __name__ == "__main__":
    asyncio.run(main())
