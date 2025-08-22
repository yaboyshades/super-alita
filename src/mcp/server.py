"""
Model Context Protocol (MCP) Server
==================================

This module implements a simple HTTP server for the Model Context Protocol,
which allows Co-Architect mode to register and use tools through a standardized API.
"""

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from uvicorn import Config, Server

from src.mcp.registry import ToolRegistry
from src.mcp.super_alita import register_super_alita_handlers

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("mcp-server")

# Create FastAPI app
app = FastAPI(title="Model Context Protocol Server", version="1.0.0")

# Create tool registry
registry = ToolRegistry()

# Register Super Alita handlers
register_super_alita_handlers(app)


# Models
class ToolSchema(BaseModel):
    name: str = Field(..., description="Name of the tool")
    tool_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for the tool"
    )
    description: str = Field("", description="Description of the tool")


class ToolExecutionRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to execute")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the tool"
    )


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Health status")
    version: str = Field("1.0.0", description="Server version")


class ToolRegistrationResponse(BaseModel):
    success: bool
    name: str


class ToolExecutionResponse(BaseModel):
    tool_name: str
    success: bool
    result: Any | None = None
    error: str | None = None


class RootResponse(BaseModel):
    name: str
    version: str
    endpoints: list[str]


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check server health."""
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/tools")
async def list_tools() -> dict[str, Any]:
    """List all registered tools."""
    tools_list = registry.list_tools()
    return {"tools": tools_list}


@app.post("/tools/register", response_model=ToolRegistrationResponse)
async def register_tool(tool: ToolSchema) -> ToolRegistrationResponse:
    """Register a new tool."""
    try:
        # Generate a simple async function that returns the tool schema
        tool_code = f"""
async def {tool.name}(**params):
    \"\"\"
    {tool.description}
    \"\"\"
    # This is a placeholder implementation
    return {{
        "name": "{tool.name}",
        "params": params,
        "schema": {json.dumps(tool.tool_schema)},
        "message": "Tool executed successfully"
    }}
"""
        # Register the tool using register_from_code
        registry.register_from_code(tool.name, tool_code)
    except Exception as e:
        logger.exception("Error registering tool")  # exception already has traceback
        raise HTTPException(status_code=400, detail=str(e)) from e

    return ToolRegistrationResponse(success=True, name=tool.name)


@app.post("/tools/execute", response_model=ToolExecutionResponse)
async def execute_tool(request: ToolExecutionRequest) -> ToolExecutionResponse:
    """Execute a tool."""
    tool_name = request.tool_name
    params = request.params

    # Check if tool exists
    if tool_name not in registry._tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    try:
        # Execute tool using invoke
        result = await registry.invoke(tool_name, params)
    except Exception as e:
        logger.exception("Error executing tool")  # exception already includes traceback
        return ToolExecutionResponse(tool_name=tool_name, success=False, error=str(e))

    return ToolExecutionResponse(tool_name=tool_name, success=True, result=result)


@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    """Root endpoint."""
    return RootResponse(
        name="Model Context Protocol Server",
        version="1.0.0",
        endpoints=["/health", "/tools", "/tools/register", "/tools/execute"],
    )


async def start_server(host: str = "localhost", port: int = 5678) -> None:
    """Start the MCP server."""
    config = Config(app=app, host=host, port=port, log_level="info")
    server = Server(config)

    await server.serve()


def main() -> None:
    """Start the MCP server when module is run directly."""
    host = os.environ.get("MCP_HOST", "localhost")
    port = int(os.environ.get("MCP_PORT", "5678"))

    logger.info("Starting MCP server on %s:%s", host, port)  # Lazy % formatting

    # Start server
    asyncio.run(start_server(host, port))


if __name__ == "__main__":
    main()
