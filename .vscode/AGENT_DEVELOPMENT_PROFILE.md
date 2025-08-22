# Super Alita Agent Development Profile

## Overview

This VS Code profile is optimized for Super Alita agent development with event-driven architecture, cognitive loops, and neural atom patterns.

## Profile Configuration

### Extensions

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.debugpy", 
    "ms-python.mypy-type-checker",
    "charliermarsh.ruff",
    "github.copilot",
    "github.copilot-chat",
    "ms-vscode.test-adapter-converter",
    "littlefoxteam.vscode-python-test-adapter",
    "ms-vscode.powershell",
    "redhat.vscode-yaml",
    "ms-vscode.json",
    "bradlc.vscode-tailwindcss",
    "ms-vscode.vscode-json"
  ]
}
```

### Key Bindings Highlights

- `Ctrl+Shift+Alt+A` - Start agent development mode
- `Ctrl+Shift+Alt+V` - Run comprehensive validation  
- `Ctrl+Shift+Alt+H` - Health check
- `Ctrl+Shift+Alt+Q` - Full quality pipeline
- `Ctrl+Shift+Alt+C` - Switch to Co-Architect chat mode
- `Ctrl+Alt+M` - Start MCP server
- `Ctrl+Alt+Shift+M` - Stop MCP server

### Task Highlights

- **Agent Development**: `agents:dev` - Development mode with enhanced logging
- **Validation**: `alita:validate` - Comprehensive system validation
- **Quality**: `quality:full` - Complete code quality pipeline
- **Testing**: `alita:test` - Fast pytest execution
- **Health**: `health:check` - System health verification

### Settings Highlights

- **Copilot Enhanced**: Co-Architect mode, agent-specific instructions
- **Python Optimized**: Type checking, linting, formatting
- **Terminal Integration**: PowerShell with agent scripts
- **Search Optimization**: Excludes for large directories
- **Auto-formatting**: On save for Python, JSON, YAML

## Agent Development Workflow

### 1. Environment Setup

```powershell
# Start Redis (required for event bus)
.\scripts\alita-mcp.ps1 -RunDashboard

# Or use task: Ctrl+Shift+Alt+R
```

### 2. Development Mode

```powershell
# Start agent in development mode  
# Task: Ctrl+Shift+Alt+A
```

### 3. Code Quality

```powershell
# Run full quality pipeline
# Task: Ctrl+Shift+Alt+Q
```

### 4. Testing

```powershell
# Run comprehensive validation
# Task: Ctrl+Shift+Alt+V

# Run fast tests
# Task: Ctrl+Shift+Alt+T
```

### 5. Health Monitoring

```powershell
# System health check
# Task: Ctrl+Shift+Alt+H
```

## Copilot Integration

### Co-Architect Mode

- Enhanced architectural guidance
- Event contract enforcement
- Neural atom patterns
- Cognitive loop optimization

### Agent Development Instructions

- Event-driven patterns
- Batch processing optimization
- Deterministic testing
- Structured logging

## Project Structure Navigation

### Core Directories

- `src/core/` - Event bus, neural atoms, global workspace
- `src/plugins/` - Agent plugins and cognitive modules
- `tests/` - Comprehensive test suite
- `scripts/` - Development and deployment scripts

### Quick Navigation

- `Ctrl+Shift+Alt+1` - Core files
- `Ctrl+Shift+Alt+2` - Plugin files  
- `Ctrl+Shift+Alt+3` - Test files

## Best Practices

### Event-Driven Development

1. Always use structured event schemas
2. Implement proper error handling
3. Follow batch processing patterns
4. Use deterministic UUIDs

### Testing Patterns

1. Use pytest with fixtures
2. Mock external dependencies
3. Test event contracts
4. Validate cognitive loops

### Code Quality

1. Type hints for all functions
2. Structured logging throughout
3. Proper error handling
4. Documentation strings

## Troubleshooting

### Common Issues

1. **Redis Connection**: Ensure Redis/Memurai is running
2. **Event Delivery**: Check pub/sub subscriptions
3. **Plugin Conflicts**: Verify plugin order
4. **Performance**: Monitor event throughput

### Debug Commands

```powershell
# Health check
python quick_status_check.py

# Validation suite  
python comprehensive_validation_suite.py

# MCP test
.\scripts\alita-mcp.ps1 -RunTests
```

## Profile Export/Import

### Export Current Profile

1. Command Palette → "Preferences: Export Profile"
2. Select "Super Alita Agent Development"
3. Share generated URL

### Import Profile

1. Command Palette → "Preferences: Import Profile"
2. Use provided URL or file
3. Select components to import

## Team Collaboration

### Shared Components

- Settings and keybindings
- Tasks and launch configurations
- Extensions and workspace settings
- Copilot instructions and chat modes

### Individual Components

- User-specific API keys
- Local paths and preferences
- Personal keyboard shortcuts
- Custom snippets
