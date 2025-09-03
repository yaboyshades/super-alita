# VS Code Bridge Integration

## Overview

The VS Code Bridge is a simulated integration that demonstrates how the Super Alita ecosystem can be connected to a developer's IDE to provide proactive assistance. This bridge detects TODO comments in source files and automatically enriches them with context from the ecosystem orchestrator.

## Architecture

The VS Code Bridge consists of:

1. **TODO Detection Engine**: Scans source files for TODO comments using regex patterns
2. **HTTP Client**: Makes requests to the Super Alita FastAPI service 
3. **IDE Action Simulator**: Simulates what a real VS Code extension would do (show notifications, insert snippets)

## Usage

### Command Line Demo

```bash
# Run the VS Code bridge simulation
python scripts/demo_todo_workflow.py "Implement LRU cache" --simulate-vscode --file "src/cache.py"
```

### Programmatic Usage

```python
from src.ecosystem.vscode_bridge import VSCodeBridgeSimulator

bridge = VSCodeBridgeSimulator(orchestrator_url="http://localhost:8080")
await bridge.scan_and_process_file("src/example.py")
```

## Configuration

The VS Code bridge can be configured with:

- `orchestrator_url`: Base URL of the Super Alita FastAPI service (default: http://localhost:8080)
- `auth_token`: Authentication token for API requests (default: dev-token)
- `user_id`: User identifier for the simulation (default: vscode_sim_user)

## TODO Detection Patterns

The bridge detects TODOs using the following patterns:

- `# TODO: Standard format`
- `# todo: Case insensitive` 
- `#TODO:No space required`
- `    # TODO: Indented comments`

## API Integration

The bridge makes HTTP POST requests to `/api/v1/developer-action` with the following payload structure:

```json
{
  "user_id": "vscode_sim_user",
  "action": "todo_detected", 
  "context": {
    "todo_text": "Implement feature X",
    "file_path": "src/example.py",
    "line_number": 42,
    "context_lines": ["surrounding", "code", "lines"]
  }
}
```

## Simulated IDE Actions

When the orchestrator responds, the bridge simulates:

1. **Toast Notifications**: Shows a notification with the generated Copilot prompt
2. **Snippet Insertion**: Displays available dynamic code snippets
3. **Status Updates**: Provides feedback on the orchestration process

## Real-World Implementation

In a production VS Code extension, this simulation would be replaced with:

- File system watchers for real-time TODO detection
- VS Code API calls for actual notifications and snippet insertion
- TypeScript/JavaScript implementation using the VS Code Extension API
- Workspace-aware configuration and settings

## Error Handling

The bridge handles common error scenarios gracefully:

- **File Not Found**: Logs error and continues
- **Network Errors**: Falls back with informative messages
- **API Errors**: Reports status codes and error details
- **Malformed Responses**: Handles partial or missing data

## Testing

Run the VS Code bridge tests:

```bash
pytest tests/test_vscode_team_integration.py::test_vscode_bridge* -v
```

## Future Enhancements

- Support for more comment styles (C++, Java, etc.)
- Configuration file for custom TODO patterns
- Integration with VS Code workspace settings
- Real-time file watching and processing
- Multi-language TODO detection