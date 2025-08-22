"""
# MCP Server

## Run locally
```bash
./mcp_server/.venv/Scripts/python.exe -m mcp_server.server --transport stdio
```

## Add a new tool
Use the script:
```powershell
pwsh ./Setup-MCP.ps1 -AddTool MyToolName
```

Then in your new file, import and register your tool:
```python
from mcp_server.server import app

@app.tool(name="your_tool", description="...")
async def your_tool(...):
    ...
```

## Configuration
Example MCP server configuration:
```json
{
    "servers": {
        "myCustomPythonAgent": {
            "type": "stdio",
            "command": "${workspaceFolder}/mcp_server/.venv/Scripts/python.exe",
            "args": [
                "${workspaceFolder}/mcp_server/src/mcp_server/server.py",
                "--transport",
                "stdio"
            ],
            "env": {
                "MCP_AGENT_API_KEY": "${input:agent-api-key}"
            },
            "inputs": [
                {
                    "id": "agent-api-key",
                    "type": "secret",
                    "description": "API Key for My Custom Agent (optional)",
                    "prompt": "Enter the API key (optional)",
                    "required": false
                }
            ],
            "cwd": "${workspaceFolder}/mcp_server"
        }
    }
}
```
