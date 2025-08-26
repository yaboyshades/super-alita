# VS Code Built-in MCP Server Integration Guide

## Overview

Instead of using external MCP servers, VS Code has built-in MCP (Model Context Protocol) support that allows extensions to register as native MCP server providers. This approach is more integrated and efficient than external server configurations.

## Key Benefits of Built-in MCP

✅ **Native Integration**: No external server processes needed
✅ **Automatic Discovery**: VS Code automatically finds and registers servers
✅ **Better Performance**: Direct communication without stdio overhead
✅ **Extension Lifecycle**: Managed by VS Code's extension system
✅ **Security**: Runs within VS Code's security context

## Current Implementation Status

### 🎯 What's Working

1. **Super Alita Agent Integration** (`src/vscode_integration/agent_integration.py`)
   - ✅ VS Code todos integration via SimpleTodoManager
   - ✅ Task creation, completion, and status tracking
   - ✅ Agent recommendations and development status
   - ✅ Command execution interface
   - ⚠️ LADDER planner (limited - initialization issue with EventBus)

2. **MCP Server Wrapper** (`src/vscode_integration/agent_mcp_server.py`)
   - ✅ Proper MCP protocol implementation
   - ✅ Tool definitions for all agent functions
   - ✅ Demo mode for testing without full MCP
   - ✅ Environment-aware configuration

3. **Built-in Provider Extension** (`src/vscode_integration/builtin_mcp_provider.ts`)
   - ✅ TypeScript extension using VS Code's `lm.registerMcpServerDefinitionProvider`
   - ✅ Automatic server discovery and configuration
   - ✅ Extension commands for management
   - ⚠️ Needs TypeScript compilation setup

### 🔧 Current Configuration

#### .vscode/mcp.json
```json
{
  "servers": {
    "superAlitaBuiltinAgent": {
      "type": "stdio",
      "command": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
      "args": ["${workspaceFolder}\\src\\vscode_integration\\agent_mcp_server.py"],
      "env": {
        "WORKSPACE_FOLDER": "${workspaceFolder}",
        "VSCODE_EXTENSION_MODE": "mcp-provider",
        "PYTHONPATH": "${workspaceFolder}"
      },
      "cwd": "${workspaceFolder}"
    }
  }
}
```

## How VS Code Built-in MCP Works

### 1. MCP Server Definition Provider

Extensions can register as MCP server providers using:

```typescript
import * as vscode from 'vscode';

// Register the provider
const registration = vscode.lm.registerMcpServerDefinitionProvider(
    'super-alita-agent-provider',
    new SuperAlitaMcpProvider()
);
```

### 2. Extension Contribution Points

In `package.json`:
```json
{
  "contributes": {
    "mcpServerDefinitionProviders": [
      {
        "id": "super-alita-agent-provider",
        "label": "Super Alita Agent Provider"
      }
    ]
  }
}
```

### 3. Provider Interface

```typescript
interface McpServerDefinitionProvider {
    provideMcpServerDefinitions(token: CancellationToken): ProviderResult<T[]>
    resolveMcpServerDefinition?(server: T, token: CancellationToken): ProviderResult<T>
    onDidChangeMcpServerDefinitions?: Event<void>
}
```

## Available MCP Extensions

Based on the VS Code marketplace search, these MCP-related extensions are available:

```vscode-extensions
automatalabs.copilot-mcp,ms-azuretools.vscode-azure-mcp-server,semanticworkbenchteam.mcp-server-vscode
```

### Key Extensions:
- **Copilot MCP**: Search, manage, and install open-source MCP servers
- **Azure MCP Server**: Azure integration and tooling
- **VSCode MCP Server**: VSCode tools and resources as MCP server

## Next Steps

### 1. Enable Built-in MCP Provider

To use the built-in provider instead of external servers:

```bash
# Compile TypeScript extension
cd src/vscode_integration
npm install @types/vscode@^1.85.0 @types/node@^20.0.0 typescript@^5.0.0
tsc -p ./
```

### 2. Install and Activate Extension

```bash
# Package the extension
npm install -g vsce
vsce package

# Install in VS Code
code --install-extension super-alita-builtin-mcp-1.0.0.vsix
```

### 3. Verify Integration

1. Open Command Palette (`Ctrl+Shift+P`)
2. Run "Super Alita: Show MCP Status"
3. Check that the agent is available as a built-in server

### 4. Test Agent Commands

```python
# Run agent directly
python src/vscode_integration/agent_integration.py

# Test MCP server
python src/vscode_integration/agent_mcp_server.py
```

## Troubleshooting

### EventBus Initialization Issue

The LADDER planner has an initialization issue:
```
⚠️ LADDER planner initialization failed: 'EventBus' object has no attribute 'initialize'
```

**Solution**: The EventBus in `src/core/event_bus.py` needs an `initialize()` method.

### MCP Package Dependencies

Ensure MCP is installed:
```bash
pip install mcp
```

### TypeScript Compilation

For the built-in provider extension:
```bash
# Install dependencies
npm install

# Compile
tsc -p ./
```

## Comparison: External vs Built-in MCP

| Feature | External MCP Server | Built-in MCP Provider |
|---------|-------------------|---------------------|
| **Setup** | Manual configuration in mcp.json | Automatic via extension |
| **Performance** | stdio communication overhead | Direct API calls |
| **Lifecycle** | Manual start/stop | Managed by VS Code |
| **Discovery** | Static configuration | Dynamic discovery |
| **Security** | Separate process | VS Code security context |
| **Debugging** | Separate logs | Integrated with extension host |

## Recommended Approach

1. **For Development**: Use the current external MCP setup for testing
2. **For Production**: Migrate to built-in MCP provider for better integration
3. **Hybrid**: Support both approaches for maximum compatibility

The built-in MCP provider offers better integration with VS Code's ecosystem and is the recommended approach for production deployments.