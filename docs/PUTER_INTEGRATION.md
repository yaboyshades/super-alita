# Puter Integration with Super Alita

This document describes the integration of Puter cloud environment with Super Alita's event-driven neural architecture.

## Overview

The Puter integration provides seamless cloud file storage, process execution, and workspace synchronization capabilities through:

- **Core Plugin**: `src/plugins/puter_plugin.py` - Event-driven plugin following PluginInterface
- **MCP Tools**: `mcp_server/src/mcp_server/tools/puter_tool.py` - VS Code integration tools
- **Neural Atoms**: Deterministic UUID generation for cognitive fabric integration
- **Comprehensive Tests**: Full test suite with 18+ test cases

## Features

### Core Plugin Capabilities

- **File Operations**: Read, write, delete files in Puter cloud storage
- **Process Execution**: Run commands in Puter cloud environment with security controls
- **Workspace Sync**: Bidirectional sync between local and cloud workspaces
- **Neural Atom Tracking**: All operations create atoms with genealogy metadata
- **Event-Driven Architecture**: Full integration with Super Alita's event bus

### MCP Tools

The MCP tools provide VS Code integration with workspace-safe operations:

- `puter_file_read` - Read files with workspace boundary validation
- `puter_file_write` - Write files with diff preview in dry-run mode
- `puter_execute` - Execute commands with security restrictions
- `puter_workspace_sync` - Sync workspace with cloud storage
- `puter_list_files` - List files in cloud directories

All tools default to `dry_run=true` and provide unified diff previews for safety.

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Puter Cloud Environment
# Default remote instance
PUTER_BASE_URL=https://puter.com
# PUTER_API_KEY=your_puter_api_key_here
# PUTER_WORKSPACE_ID=default
# For local hosting, override base URL:
# PUTER_BASE_URL=http://localhost:4100
```

### Plugin Configuration

The plugin is automatically enabled in the unified system with configuration:

```yaml
plugins:
  puter:
    enabled: true
    puter_base_url: "https://puter.com"
    puter_api_key: ""
    puter_workspace_id: "default"
```

Environment variables take precedence over configuration file settings.

## Usage Examples

### Event-Driven Operations

The plugin responds to events on the event bus:

```python
# File operation event
await event_bus.emit("puter_file_operation", 
    operation="write",
    file_path="/workspace/test.txt",
    content="Hello Puter World!",
    conversation_id="session_123"
)

# Process execution event  
await event_bus.emit("puter_process_execution",
    command="python",
    args=["--version"],
    working_dir="/workspace",
    conversation_id="session_123"
)

# Workspace sync event
await event_bus.emit("puter_workspace_sync",
    sync_type="bidirectional", 
    local_path="/local/workspace",
    remote_path="/remote/workspace",
    conversation_id="session_123"
)
```

### MCP Tool Usage

From VS Code or MCP clients:

```python
# Read a file (dry run by default)
result = await puter_file_read("test.txt", dry_run=True)

# Write a file with preview
result = await puter_file_write("test.txt", "content", dry_run=True) 
print(result["diff_preview"])  # Shows unified diff

# Execute command safely
result = await puter_execute("echo", ["hello"], dry_run=False)
```

### Tool Call Integration

The plugin automatically handles tool calls for Puter operations:

```python
# These tool calls are automatically routed to Puter plugin
await event_bus.emit("tool_call",
    tool_name="puter_file_read",
    parameters={"file_path": "/test/file.txt"},
    conversation_id="session_123"
)
```

## Neural Atom Integration

All Puter operations create neural atoms with deterministic UUIDs:

```python
# Each operation creates a PuterOperationAtom
operation_data = {
    "operation": "file_write",
    "file_path": "/test/file.txt", 
    "timestamp": "2024-01-01T00:00:00Z",
    "description": "File write operation on /test/file.txt"
}

atom = PuterOperationAtom("file_operation", operation_data)
uuid = atom.get_deterministic_uuid()  # Deterministic based on operation data
```

The atoms include:
- **Genealogy metadata**: Creation time, lineage, depth tracking
- **Neural capabilities**: Semantic embeddings, activation patterns
- **Performance metrics**: Usage count, success rate, execution time

## Event Flow

1. **Event Received**: Plugin receives Puter-specific events
2. **Neural Atom Created**: Operation atom generated with deterministic UUID
3. **API Simulation**: Cloud operation executed (currently simulated)
4. **Result Event Emitted**: Completion event with neural atom ID
5. **History Tracked**: Operation stored in plugin history
6. **Shutdown Storage**: Atoms registered with neural store on shutdown

## DeepCode

DeepCode proposals are mirrored to Puter so the cloud workspace reflects
generated code. The `deepcode_puter_bridge` plugin listens for
`deepcode_ready_for_apply` events and emits `puter_file_write` events for
each proposed diff, test, and doc. The Puter plugin consumes these events,
performs the writes, and records a `PuterOperationAtom` for every file to
maintain lineage.

## Security Features

### Workspace Boundary Validation

All file operations validate paths against workspace root:

```python
def _is_subpath(base: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(base)
        return True
    except ValueError:
        return False
```

### Safe Command Execution

Process execution is restricted to safe commands:

```python
safe_commands = {
    "echo", "cat", "ls", "pwd", "whoami", "date", 
    "python", "node", "npm", "git"
}
```

### Timeout Protection

All operations have configurable timeouts:
- File operations: Built-in async timeouts
- Process execution: 30-second timeout with cleanup
- API calls: Simulated 100ms network delay

## Testing

Comprehensive test suite with 18 test cases:

```bash
# Run all Puter tests
python -m pytest tests/runtime/test_puter_integration.py -v

# Run specific test categories
python -m pytest tests/runtime/test_puter_integration.py::TestPuterOperationAtom -v
python -m pytest tests/runtime/test_puter_integration.py::TestPuterPlugin -v
```

Test coverage includes:
- Neural atom deterministic UUID generation
- Event-driven operation handling
- Error handling and recovery
- Tool call routing
- API simulation methods
- Plugin lifecycle management

## Architecture Integration

The Puter plugin integrates seamlessly with Super Alita's architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Event Bus     │    │  Puter Plugin    │    │   Neural Store  │
│  (Redis/Memurai)│◄──►│  (Event Handler) │◄──►│  (Atom Storage) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Tools     │    │ Operation Atoms  │    │ Global Workspace│
│ (VS Code/Client)│    │(Deterministic ID)│    │ (Consciousness) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Future Enhancements

1. **Real API Integration**: Replace simulation with actual Puter API calls
2. **Authentication**: OAuth/JWT token management for Puter services  
3. **Advanced Sync**: Conflict resolution, incremental sync, compression
4. **Performance Optimization**: Caching, batch operations, connection pooling
5. **Enhanced Security**: Role-based access, audit logging, encryption
6. **Monitoring**: Metrics collection, health checks, alerting

## Troubleshooting

### Common Issues

1. **Plugin Not Loading**: Check that `puter` is in `PLUGIN_ORDER` and enabled in config
2. **Import Errors**: Verify `PYTHONPATH` includes `src` directory
3. **Configuration Issues**: Ensure environment variables are set correctly
4. **MCP Tools Not Found**: Check `PYTHONPATH` includes `mcp_server/src`

### Debug Commands

```bash
# Test plugin loading
python -c "from src.plugins.puter_plugin import PuterPlugin; print('✅ Plugin loads')"

# Test MCP tools
PYTHONPATH=./mcp_server/src python -c "from mcp_server.tools.puter_tool import puter_file_read; print('✅ MCP tools load')"

# Verify unified system integration  
python -c "from src.main_unified import _load_unified_plugins, AVAILABLE_PLUGINS; _load_unified_plugins(); print('✅ Unified integration' if 'puter' in AVAILABLE_PLUGINS else '❌ Not found')"
```

### Logging

Enable debug logging to see detailed plugin operation:

```python
import logging
logging.getLogger("src.plugins.puter_plugin").setLevel(logging.DEBUG)
```

## Contributing

When contributing to the Puter integration:

1. Follow the existing code patterns and architecture
2. Maintain deterministic UUID generation for neural atoms
3. Use keyword args for all event creation
4. Include comprehensive tests for new functionality
5. Update documentation for any API changes
6. Ensure MCP tools maintain workspace boundary safety