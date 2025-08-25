# Super Alita MCP Tools

This directory contains scripts and configuration for integrating Super Alita with VS Code's Model Context Protocol (MCP).

## Setup Instructions

1. **Start the MCP Server and Register Tools**:
   - Run the VS Code task `super-alita:mcp:start`
   - This will start the MCP server and register Super Alita tools

2. **Use Super Alita Tools in VS Code**:
   - The tools are registered with the prefix `mcp_super_alita_k_*`
   - Examples:
     - `mcp_super_alita_k_ping` - Health check
     - `mcp_super_alita_k_mem0_add_memory` - Add memory

## Available Tools

- `mcp_super_alita_k_echo` - Echo back input value
- `mcp_super_alita_k_ping` - Health check ping
- `mcp_super_alita_k_get_agent_status` - Get current agent status and health
- `mcp_super_alita_k_get_agent_telemetry` - Get real-time agent telemetry data and events
- `mcp_super_alita_k_mem0_add_memory` - Store a memory with category and metadata
- `mcp_super_alita_k_mem0_get_all_memories` - Get all memories, optionally filtered by category
- `mcp_super_alita_k_mem0_search_memories` - Search memories by query and optional category
- `mcp_super_alita_k_mem0_store_architectural_decision` - Store an architectural decision
- `mcp_super_alita_k_mem0_store_debugging_pattern` - Store a debugging pattern
- `mcp_super_alita_k_mem0_store_session_learning` - Store a Co-Architect session learning

## Configuration

The configuration file is located at `src/mcp/super_alita_config.json`. This file defines the tools, their parameters, and MCP server settings.

## Debugging

If you encounter issues:

1. Check that the MCP server is running
2. Verify the tools are registered by visiting <http://localhost:5678/tools>
3. Check VS Code settings are correctly pointing to the config file

## Scripts

- `scripts/start_super_alita_mcp.ps1` - PowerShell script to start MCP server and register tools
- `scripts/register_super_alita_tools.py` - Python script to register tools with the MCP server
