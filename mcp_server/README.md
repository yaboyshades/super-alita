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

## Naming conventions

Per the [MCP guide](../docs/mcp.md), tool modules and decorator names should use
`snake_case`, while parameter names follow `camelCase`. Adhering to these
conventions ensures MCP clients can discover and invoke tools reliably.

To bootstrap a new tool from the provided template, run:
```bash
python src/mcp_server/server.py --add-tool my_tool
```
This copies `tools/_template.py` into the tools package.

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
