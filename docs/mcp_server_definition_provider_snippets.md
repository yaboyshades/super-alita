# MCP Server Definition Provider Snippets

Examples for using `vscode.lm.registerMcpServerDefinitionProvider` in extensions.

## TypeScript

```typescript
import * as vscode from 'vscode';

class ExampleMcpProvider implements vscode.lm.McpServerDefinitionProvider {
    provideMcpServerDefinitions(_token: vscode.CancellationToken) {
        return [{
            id: 'example',
            displayName: 'Example MCP',
            command: 'node',
            args: ['server.js'],
            env: {}
        }];
    }
}

export function activate() {
    vscode.lm.registerMcpServerDefinitionProvider(
        'example-mcp-provider',
        new ExampleMcpProvider()
    );
}
```

## Node.js (CommonJS)

```javascript
const vscode = require('vscode');

class ExampleMcpProvider {
  provideMcpServerDefinitions() {
    return [{
      id: 'example',
      displayName: 'Example MCP',
      command: 'node',
      args: ['server.js'],
      env: {}
    }];
  }
}

function activate() {
  vscode.lm.registerMcpServerDefinitionProvider(
    'example-mcp-provider',
    new ExampleMcpProvider()
  );
}

module.exports = { activate };
```
