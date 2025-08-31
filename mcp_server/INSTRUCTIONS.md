# MCP Server Instructions

This directory contains the Model Context Protocol (MCP) server implementation for Super Alita.

## Overview

The MCP server provides standardized tool execution and context management capabilities, enabling seamless integration with VS Code and other MCP-compatible clients. This implementation includes:

- **Tool Registry**: Centralized tool discovery and management
- **VS Code Integration**: Native MCP client support in VS Code
- **Fallback Mechanisms**: Graceful degradation when tools are unavailable
- **Schema Validation**: Structured tool input/output validation
- **Real-time Telemetry**: Broadcasting tool execution metrics

## Architecture

### Core Components
- **MCP Server Wrapper** (`mcp_server_wrapper.py`): Main server implementation with fallback
- **Tool Registry**: Centralized tool discovery and registration
- **Schema Validation**: Input/output validation for tool calls
- **Client Integration**: VS Code and other MCP client support

### Integration Points
- **Super Alita Core**: Event-driven integration with main system
- **Tool Execution**: Direct tool invocation and result handling
- **Telemetry Streaming**: Real-time metrics broadcasting
- **Configuration Management**: Dynamic configuration and setup

## Development Setup

### MCP Server Installation
```bash
# Initialize MCP server with VS Code integration
pwsh .\Setup-MCP.ps1 -Bootstrap

# Health check the environment
pwsh .\Setup-MCP.ps1 -Doctor

# Add new tools (automated scaffolding)
pwsh .\Setup-MCP.ps1 -AddTool YourToolName
```

### Manual Setup
```bash
# Install MCP dependencies
cd mcp_server
pip install -e .

# Configure VS Code MCP integration
# Add to VS Code settings.json:
{
    "mcp.servers": {
        "myCustomPythonAgent": {
            "command": "python",
            "args": ["mcp_server_entrypoint.py"],
            "cwd": "/path/to/super-alita"
        }
    }
}
```

## Tool Development

### Tool Structure
Tools follow a standardized interface pattern:

```python
from typing import Dict, Any
from pydantic import BaseModel

class ToolInput(BaseModel):
    """Input schema for the tool."""
    parameter: str
    optional_param: str = "default"

class ToolOutput(BaseModel):
    """Output schema for the tool."""
    success: bool
    result: str
    metadata: Dict[str, Any] = {}

async def execute_tool(input_data: ToolInput) -> ToolOutput:
    """
    Execute the tool with validated input.

    Args:
        input_data: Validated input parameters

    Returns:
        Structured output with results
    """
    try:
        # Tool implementation
        result = process_input(input_data.parameter)

        return ToolOutput(
            success=True,
            result=result,
            metadata={"execution_time": "0.5s"}
        )
    except Exception as e:
        return ToolOutput(
            success=False,
            result=f"Error: {str(e)}",
            metadata={"error_type": type(e).__name__}
        )
```

### Tool Registration
```python
from mcp_server import ToolRegistry

# Register a new tool
registry = ToolRegistry()
registry.register_tool(
    name="my_tool",
    description="Tool description",
    input_schema=ToolInput.schema(),
    output_schema=ToolOutput.schema(),
    execute_function=execute_tool
)
```

### Tool Scaffolding
Use the automated scaffolding system:

```bash
# Generate new tool template
pwsh .\Setup-MCP.ps1 -AddTool MyNewTool

# This creates:
# - mcp_server/src/mcp_server/tools/my_new_tool.py
# - Input/output schemas
# - Basic implementation template
# - Test stubs
```

## Configuration

### MCP Server Configuration
```python
# mcp_server/config.py
MCP_CONFIG = {
    "server": {
        "name": "myCustomPythonAgent",
        "version": "1.0.0",
        "timeout": 30.0
    },
    "tools": {
        "auto_discovery": True,
        "validation": True,
        "fallback_enabled": True
    },
    "telemetry": {
        "enabled": True,
        "broadcast_events": True
    }
}
```

### Environment Variables
```bash
# MCP-specific configuration
MCP_SERVER_NAME=myCustomPythonAgent
MCP_TIMEOUT=30.0
MCP_AUTO_DISCOVERY=true
MCP_VALIDATION_ENABLED=true
MCP_TELEMETRY_ENABLED=true
```

## VS Code Integration

### MCP Client Setup
1. **Install MCP Extension**: Install the MCP extension in VS Code
2. **Configure Server**: Add server configuration to VS Code settings
3. **Verify Connection**: Check "MCP: Show Installed Servers" in command palette
4. **Test Tools**: Invoke tools through the Agent Mode interface

### Agent Mode Usage
```bash
# In VS Code Command Palette
MCP: Show Installed Servers  # Verify server is running
MCP: Agent Mode             # Enter interactive mode
```

### Debugging MCP Integration
```bash
# Check MCP server logs
tail -f logs/mcp_server.log

# Test MCP connection
python mcp_server_entrypoint.py --test

# Validate tool registry
python -c "from mcp_server import ToolRegistry; print(ToolRegistry().list_tools())"
```

## Tool Guidelines

### Best Practices
- **Default Dry Run**: Always default `dry_run=true` for destructive operations
- **Workspace Boundary**: Never modify files outside `${workspaceFolder}`
- **Path Safety**: Use `Path(file_path).resolve()` and validate against workspace
- **Unified Diffs**: Return structured diffs for review before applying changes

### Error Handling
All tools must return structured error responses:

```python
{
    "success": False,
    "error": "Detailed error description",
    "error_type": "validation|network|processing|system",
    "recovery_suggestions": ["Try again later", "Check network connection"]
}
```

### Security Considerations
- **Path Validation**: Ensure all file paths are within workspace boundaries
- **Input Sanitization**: Validate and sanitize all user inputs
- **Permission Checks**: Verify permissions before file operations
- **Safe Defaults**: Use safe defaults for all operations

## Testing MCP Tools

### Unit Testing
```python
import pytest
from mcp_server.tools.my_tool import execute_tool, ToolInput

@pytest.mark.asyncio
async def test_tool_execution():
    input_data = ToolInput(parameter="test_value")
    result = await execute_tool(input_data)

    assert result.success is True
    assert "test_value" in result.result
```

### Integration Testing
```bash
# Test MCP server functionality
pytest mcp_server/tests/ -v

# Test specific tools
pytest mcp_server/tests/test_tools.py::test_my_tool -v

# Test VS Code integration
python mcp_server/tests/test_vscode_integration.py
```

### Manual Testing
```bash
# Start MCP server in test mode
python mcp_server_entrypoint.py --debug

# Test tool execution via MCP protocol
echo '{"method": "tools/call", "params": {"name": "my_tool", "arguments": {"parameter": "test"}}}' | python mcp_server_entrypoint.py
```

## Monitoring and Telemetry

### Event Broadcasting
MCP tools automatically broadcast telemetry events:

```python
# Events are automatically emitted for:
- Tool execution start/completion
- Performance metrics
- Error conditions
- Resource usage
```

### Performance Monitoring
```python
# Monitor tool performance
from mcp_server.telemetry import PerformanceMonitor

monitor = PerformanceMonitor()
metrics = monitor.get_tool_metrics("my_tool")
print(f"Average execution time: {metrics['avg_execution_time']}")
```

## Troubleshooting

### Common Issues

#### MCP Server Not Starting
- Check Python environment and dependencies
- Verify configuration files are valid
- Check for port conflicts

#### Tool Not Discovered
- Verify tool is in correct directory structure
- Check tool registration code
- Validate schema definitions

#### VS Code Integration Issues
- Check VS Code MCP extension is installed
- Verify server configuration in settings.json
- Check MCP server logs for connection errors

### Debugging Commands
```bash
# Verbose MCP server startup
python mcp_server_entrypoint.py --verbose --debug

# Check tool registry
python -c "
from mcp_server import ToolRegistry
registry = ToolRegistry()
print('Available tools:', registry.list_tools())
"

# Test tool execution
python -c "
import asyncio
from mcp_server.tools.echo import execute_echo, EchoInput
async def test():
    result = await execute_echo(EchoInput(message='test'))
    print(result)
asyncio.run(test())
"
```

### Log Analysis
```bash
# MCP server logs
tail -f logs/mcp_server.log

# Tool execution logs
grep "tool_execution" logs/mcp_server.log

# Error analysis
grep "ERROR" logs/mcp_server.log | tail -20
```

For advanced MCP development and integration, refer to the official MCP specification and VS Code MCP documentation.
