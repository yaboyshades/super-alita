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

## Resources and prompts

Optional static resources and slash-command prompts can be registered via
`toolforge.py`. Append descriptors to the exported lists before starting the
server:

```python
from toolforge import (
    ResourceDescriptor,
    PromptDefinition,
    RESOURCES,
    PROMPTS,
)

RESOURCES.append(
    ResourceDescriptor(
        name="readme", mime_type="text/plain", content="Hello world!"
    )
)

PROMPTS.append(
    PromptDefinition(
        name="say_hello", description="Respond with a greeting.", content="hi!"
    )
)
```

Prompts registered this way appear as slash commands (e.g. `/say_hello`) in
clients that support MCP prompts.

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
