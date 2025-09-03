#!/usr/bin/env python3
"""MCP Server Wrapper for Super Alita Agent Integration

This script creates an MCP (Model Context Protocol) server that wraps the
Super Alita Agent, allowing it to be used directly by VS Code's built-in
MCP support instead of requiring external server configuration.

The server provides tools for:
- Task management via VS Code todos
- LADDER planning integration
- Development workflow automation
- Agent command execution

Usage:
    python agent_mcp_server.py

Environment Variables:
    WORKSPACE_FOLDER: VS Code workspace folder path
    VSCODE_EXTENSION_MODE: Set to 'mcp-provider' when called from built-in provider
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from openai.types.responses.tool import InitializationOptions


# --- Make stdout/stderr UTF-8 & resilient on Windows ---
def _force_utf8_stdio():
    """Force UTF-8 encoding and make console output resilient to Unicode errors."""
    # Encourage UTF-8 everywhere
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # Python 3.7+: reconfigure TextIO to utf-8, and never crash on encode
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    # Windows console code page to UTF-8 (best effort)
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass


def _safe_str(s: str) -> str:
    """Safe string conversion that handles Unicode encoding errors."""
    try:
        return str(s)
    except UnicodeEncodeError:
        return s.encode("utf-8", "replace").decode("utf-8")


# Apply UTF-8 configuration immediately
_force_utf8_stdio()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# MCP Python SDK imports (Anthropic Model Context Protocol)
print("🔍 Attempting MCP imports...", file=sys.stderr)
try:
    print("  - Importing mcp.server...", file=sys.stderr)
    from mcp.server import Server

    print("  - Importing mcp.types...", file=sys.stderr)
    from mcp.types import (
        CallToolRequest,
        CallToolResult,
        TextContent,
        Tool,
    )

    print("✅ MCP imports successful", file=sys.stderr)
    MCP_AVAILABLE = True
except ImportError as e:
    print(f"❌ MCP import failed at: {e}", file=sys.stderr)
    print(f"Install with: `pip install mcp` - Original error: {e!r}", file=sys.stderr)
    MCP_AVAILABLE = False

# Import agent integration
try:
    from vscode_integration.agent_integration import SuperAlitaAgent

    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Agent integration not available: {e}", file=sys.stderr)
    AGENT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SuperAlitaMcpServer:
    """MCP Server wrapper for Super Alita Agent."""

    def __init__(self):
        self.agent: SuperAlitaAgent | None = None
        self.server: Server | None = None
        self.workspace_folder = Path.cwd()

        # Get workspace folder from environment if available
        import os

        if workspace_env := os.getenv("WORKSPACE_FOLDER"):
            self.workspace_folder = Path(workspace_env)

        logger.info(f"Super Alita MCP Server initializing in: {self.workspace_folder}")

    async def initialize_agent(self) -> bool:
        """Initialize the Super Alita Agent."""
        if not AGENT_AVAILABLE:
            logger.error("Agent integration not available")
            return False

        try:
            self.agent = SuperAlitaAgent(self.workspace_folder)
            success = await self.agent.initialize()

            if success:
                logger.info("✅ Super Alita Agent initialized successfully")
            else:
                logger.error("❌ Failed to initialize Super Alita Agent")

            return success

        except Exception as e:
            logger.error(f"❌ Agent initialization error: {e}")
            return False

    def create_mcp_server(self) -> Server:
        """Create and configure the MCP server."""
        if not MCP_AVAILABLE:
            raise RuntimeError("MCP package not available")

        server = Server("super-alita-agent")

        @server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools provided by Super Alita Agent."""
            return [
                Tool(
                    name="get_development_status",
                    description="Get comprehensive development status including tasks, completion rate, and recommendations",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
                Tool(
                    name="create_development_task",
                    description="Create a new development task with title, description, and priority",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Task title"},
                            "description": {
                                "type": "string",
                                "description": "Task description",
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"],
                                "description": "Task priority level",
                                "default": "medium",
                            },
                        },
                        "required": ["title", "description"],
                    },
                ),
                Tool(
                    name="complete_development_task",
                    description="Mark a development task as complete with optional completion notes",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "ID of the task to complete",
                            },
                            "notes": {
                                "type": "string",
                                "description": "Optional completion notes",
                                "default": "",
                            },
                        },
                        "required": ["task_id"],
                    },
                ),
                Tool(
                    name="plan_with_ladder",
                    description="Create a LADDER plan for a development goal",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": "Development goal to plan for",
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["shadow", "active"],
                                "description": "Planning mode",
                                "default": "shadow",
                            },
                        },
                        "required": ["goal"],
                    },
                ),
                Tool(
                    name="get_agent_recommendations",
                    description="Get intelligent recommendations for the developer based on current state",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
                Tool(
                    name="execute_agent_command",
                    description="Execute arbitrary agent commands for development automation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Command to execute (status, create_task, complete_task, plan, recommendations, help)",
                            },
                            "kwargs": {
                                "type": "object",
                                "description": "Additional keyword arguments for the command",
                                "default": {},
                            },
                        },
                        "required": ["command"],
                    },
                ),
                Tool(
                    name="get_development_insights",
                    description="Use Cortex LeanRAG to get development insights and recommendations for a specific query",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Development question or context to get insights about",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="plan_development_task",
                    description="Use Cortex planning capabilities to create a structured plan for a development task",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "Description of the development task to plan",
                            },
                        },
                        "required": ["task_description"],
                    },
                ),
                Tool(
                    name="deepconf_consensus",
                    description="Run enhanced consensus sampling (weighted_vote, simple_vote, etc.)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "num_samples": {"type": "integer", "default": 3},
                            "temperature": {"type": "number", "default": 0.7},
                            "max_tokens": {"type": "integer", "default": 512},
                            "method": {
                                "type": "string",
                                "enum": [
                                    "simple_vote",
                                    "weighted_vote",
                                    "confidence_based",
                                    "semantic_similarity",
                                    "ensemble_ranking",
                                ],
                                "default": "weighted_vote",
                            },
                            "confidence_threshold": {"type": "number", "default": 0.7},
                            "temperature_range": {"type": "number", "default": 0.2},
                        },
                        "required": ["prompt"],
                    },
                ),
            ]

        @server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict | None
        ) -> list[TextContent]:
            """Handle tool calls from the language model."""
            if not self.agent:
                return [
                    TextContent(
                        type="text", text="❌ Super Alita Agent not initialized"
                    )
                ]

            try:
                arguments = arguments or {}

                if name == "get_development_status":
                    result = await self.agent.get_development_status()

                elif name == "create_development_task":
                    title = arguments.get("title", "")
                    description = arguments.get("description", "")
                    priority = arguments.get("priority", "medium")
                    result = await self.agent.create_development_task(
                        title, description, priority
                    )

                elif name == "complete_development_task":
                    task_id = arguments.get("task_id", "")
                    notes = arguments.get("notes", "")
                    result = await self.agent.complete_development_task(task_id, notes)

                elif name == "plan_with_ladder":
                    goal = arguments.get("goal", "")
                    mode = arguments.get("mode", "shadow")
                    result = await self.agent.plan_with_ladder(goal, mode)

                elif name == "get_agent_recommendations":
                    result = await self.agent.get_agent_recommendations()

                elif name == "execute_agent_command":
                    command = arguments.get("command", "")
                    kwargs = arguments.get("kwargs", {})
                    result = await self.agent.execute_agent_command(command, **kwargs)

                elif name == "get_development_insights":
                    query = arguments.get("query", "")
                    result = await self.agent.get_development_insights(query)

                elif name == "plan_development_task":
                    task_description = arguments.get("task_description", "")
                    result = await self.agent.plan_development_task(task_description)

                elif name == "deepconf_consensus":
                    # Delegate to runtime's ability execution endpoint
                    base = os.getenv(
                        "MCP_RUNTIME_BASE", "http://127.0.0.1:8080"
                    ).rstrip("/")
                    try:
                        import httpx  # type: ignore

                        async with httpx.AsyncClient(timeout=30.0) as client:
                            resp = await client.post(
                                f"{base}/ability/execute/deepconf_consensus",
                                json=arguments,
                                headers={"Accept": "application/json"},
                            )
                            data = (
                                resp.json()
                                if resp.headers.get("content-type", "").startswith(
                                    "application/json"
                                )
                                else {
                                    "status": resp.status_code,
                                    "text": await resp.aread(),
                                }
                            )
                            result = data
                    except Exception as e:
                        result = {
                            "error": f"consensus_call_failed: {type(e).__name__}",
                            "detail": str(e),
                        }

                else:
                    result = {"error": f"Unknown tool: {name}"}

                # Format result as JSON for the language model
                result_text = json.dumps(result, indent=2, default=str)
                return [TextContent(type="text", text=result_text)]

            except Exception as e:
                error_text = f"❌ Tool execution error: {str(e)}"
                logger.error(error_text)
                return [TextContent(type="text", text=error_text)]

        self.server = server
        return server


def create_mcp_server() -> Server:
    """Create a Super Alita MCP server instance for external use."""
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP package not available")

    mcp_server = SuperAlitaMcpServer()
    return mcp_server.create_mcp_server()


async def run_mcp_server():
    """Run the Super Alita MCP server."""
    if not MCP_AVAILABLE:
        print(
            "❌ MCP package not available. Install with: pip install mcp",
            file=sys.stderr,
        )
        return 1

    try:
        # Create and initialize the server
        mcp_server = SuperAlitaMcpServer()

        # Initialize the agent
        agent_initialized = await mcp_server.initialize_agent()
        if not agent_initialized:
            print(
                "⚠️ Agent initialization failed, but MCP server will still run",
                file=sys.stderr,
            )

        # Create the MCP server
        server = mcp_server.create_mcp_server()

        # Run the server
        logger.info("🚀 Starting Super Alita MCP Server...")

        from mcp.server.stdio import stdio_server

        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="super-alita-agent",
                        server_version="1.0.0",
                        capabilities=server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )
        except BaseExceptionGroup as eg:
            # Python 3.11+: handle ExceptionGroup explicitly
            logger.error(
                "MCP server failed with %s sub-exception(s):", len(eg.exceptions)
            )
            for i, e in enumerate(eg.exceptions, 1):
                logger.exception(
                    "TaskGroup sub-exception #%s: %s", i, type(e).__name__, exc_info=e
                )
            raise RuntimeError("MCP server failed with TaskGroup exceptions") from eg
        except Exception as e:
            # Handle single exceptions or fallback for older Python versions
            logger.exception("MCP server error: %s", type(e).__name__, exc_info=e)
            raise

    except Exception as e:
        logger.error(f"❌ MCP Server error: {e}")
        return 1

    return 0


async def demo_mode():
    """Run in demo mode without MCP when packages aren't available."""
    print("🤖 Super Alita Agent MCP Server - Demo Mode")
    print("=" * 60)

    if AGENT_AVAILABLE:
        try:
            agent = SuperAlitaAgent()
            success = await agent.initialize()

            if success:
                print("✅ Agent initialized successfully")

                # Show development status
                status = await agent.get_development_status()
                print("\n📊 Development Status:")
                print(f"  Workspace: {status['workspace']}")
                print(f"  Completion Rate: {status['completion_rate']:.1%}")
                print(f"  Tasks: {status['task_summary']}")

                # Show recommendations
                recommendations = await agent.get_agent_recommendations()
                print("\n💡 Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")

                await agent.shutdown()

            else:
                print("❌ Agent initialization failed")

        except Exception as e:
            print(f"❌ Demo error: {e}")
    else:
        print("❌ Agent integration not available")

    print("\n📋 To use as MCP server:")
    print("  1. Install MCP: pip install mcp")
    print("  2. Configure in VS Code via built-in MCP provider")
    print("  3. Or use in .vscode/mcp.json configuration")


async def main():
    """Main entry point."""
    import os

    # Check if running in VS Code extension mode
    extension_mode = os.getenv("VSCODE_EXTENSION_MODE") == "mcp-provider"

    if extension_mode:
        logger.info("🔗 Running in VS Code built-in MCP provider mode")

    if MCP_AVAILABLE:
        return await run_mcp_server()
    else:
        await demo_mode()
        return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Super Alita MCP Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
