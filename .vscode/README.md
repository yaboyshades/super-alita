# VS Code Configuration for Super Alita Agent Development

This directory contains a comprehensive VS Code configuration optimized for Super Alita agent development with event-driven architecture, cognitive loops, and neural atom patterns.

## 📁 Configuration Files

| File | Purpose | Key Features |
|------|---------|--------------|
| `settings.json` | Core VS Code settings | Copilot enhancement, Python optimization, terminal integration |
| `tasks.json` | Build/test/dev tasks | Agent workflows, quality pipeline, MCP operations |
| `keybindings.json` | Custom shortcuts | Quick access to agent dev workflows |
| `extensions.json` | Recommended extensions | Python, Copilot, testing, formatting tools |
| `launch.json` | Debug configurations | Agent debugging, MCP server debugging |

## 📚 Documentation

| File | Purpose |
|------|---------|
| `AGENT_DEVELOPMENT_PROFILE.md` | Complete profile overview and configuration guide |
| `QUICK_START.md` | 30-second setup and essential workflows |
| `README.md` | This file - configuration overview |

## 🚀 Quick Setup

### 1. Install Recommended Extensions

Open Command Palette (`Ctrl+Shift+P`) → "Extensions: Show Recommended Extensions" → Install All

### 2. Configure Profile (Optional)

For team sharing:

1. Command Palette → "Preferences: Export Profile"
2. Select "Super Alita Agent Development"
3. Share URL with team members

### 3. Start Development

- **`Ctrl+Shift+Alt+A`** - Start agent development mode
- **`Ctrl+Shift+Alt+V`** - Run validation suite
- **`Ctrl+Shift+Alt+H`** - Health check

## ⌨️ Essential Hotkeys

### Agent Operations

- `Ctrl+Shift+Alt+A` - Start agents in development mode
- `Ctrl+Shift+Alt+V` - Run comprehensive validation
- `Ctrl+Shift+Alt+H` - System health check
- `Ctrl+Shift+Alt+Q` - Full quality pipeline

### MCP Operations  

- `Ctrl+Alt+M` - Start MCP server
- `Ctrl+Alt+Shift+M` - Stop MCP server
- `Ctrl+Alt+T` - Run MCP tests

### Development

- `Ctrl+Shift+Alt+T` - Fast pytest
- `Ctrl+Shift+Alt+L` - Lint code
- `Ctrl+Shift+Alt+Y` - Type check
- `Ctrl+Shift+Alt+C` - Co-Architect chat mode

## 🧠 Copilot Integration

### Co-Architect Mode

Enhanced architectural guidance with:

- Event contract enforcement
- Neural atom patterns  
- Cognitive loop optimization
- Batch processing patterns
- Deterministic testing guidance

### Agent Development Instructions

Located in `.github/copilot/` with specialized prompts for:

- Event-driven development
- Neural atom creation
- Plugin architecture
- Testing patterns

## 🔧 Tasks Overview

### Quality Pipeline

- `quality:full` - Complete pipeline (lint, format, type, test)
- `lint` - Ruff linting with auto-fix
- `format` - Code formatting with Ruff
- `typecheck` - Pyright type checking
- `test` - Pytest execution
- `coverage` - Test coverage report

### Agent Development

- `agents:dev` - Development mode with enhanced logging
- `alita:validate` - Comprehensive system validation
- `health:check` - System health verification
- `redis:start` - Start Redis server

### MCP Operations

- `alita:mcp:start` - Start MCP with dashboard
- `alita:mcp:stop` - Stop MCP processes
- `alita:mcp:test` - MCP validation tests

## 🎯 Development Patterns

### Event-Driven Architecture

All components communicate via structured events:

```python
from src.core.events import ToolCallEvent
event = ToolCallEvent(
    source_plugin=self.name,
    conversation_id=session_id,
    tool_name="example_tool",
    parameters={"param": "value"}
)
await self.event_bus.publish(event)
```

### Neural Atom Pattern

Modular intelligence units:

```python
class ExampleAtom(NeuralAtom):
    async def execute(self, input_data: Any) -> Any:
        # Core functionality
        pass
    
    def get_embedding(self) -> List[float]:
        # Semantic representation
        pass
    
    def can_handle(self, task: str) -> float:
        # Confidence score 0-1
        pass
```

### Testing Pattern

Deterministic and comprehensive:

```python
@pytest.fixture
def mock_event_bus():
    return AsyncMock(spec=EventBus)

async def test_event_handling(mock_event_bus):
    test_id = generate_deterministic_id("test", "scenario")
    # Test implementation with mocked dependencies
```

## 📊 Settings Highlights

### Copilot Enhancement

- Co-Architect chat mode enabled
- Agent-specific instructions loaded
- Custom toolsets for event-driven development
- Inline suggestions optimized for Python patterns

### Python Optimization

- Type checking with Pyright
- Linting with Ruff (fast, comprehensive)
- Auto-formatting on save
- Import organization
- Virtual environment detection

### Performance Optimization

- Excluded directories: `node_modules`, `__pycache__`, `.venv`, `htmlcov`
- File watching optimization
- Search exclusions for large directories
- Terminal integration with PowerShell

### Editor Enhancement

- Minimap enabled for large files
- Bracket pair colorization
- Sticky scroll for context
- Auto-save enabled
- Line numbers and whitespace rendering

## 🛠️ Troubleshooting

### Common Issues

#### Redis Connection

```powershell
# Start Redis manually
redis-server --port 6379 --bind 127.0.0.1

# Or use hotkey
Ctrl+Shift+Alt+R
```

#### Extension Issues

```powershell
# Reload window
Ctrl+Shift+P → "Developer: Reload Window"

# Check extension status
Ctrl+Shift+X → Verify all recommended extensions installed
```

#### Task Failures

```powershell
# Check terminal output
Ctrl+Shift+` → Review task execution logs

# Manual validation
python quick_status_check.py
```

## 📋 Configuration Maintenance

### Regular Updates

1. **Extensions**: Keep Python, Copilot, and linting extensions updated
2. **Settings**: Review settings.json for new VS Code features
3. **Tasks**: Update task configurations as workflows evolve
4. **Keybindings**: Adjust shortcuts based on team preferences

### Team Synchronization

1. **Profile Export**: Regular profile exports for new team members
2. **Configuration Review**: Quarterly review of settings and tasks
3. **Extension Alignment**: Ensure all team members use same extensions
4. **Workflow Updates**: Update tasks as development practices evolve

---

## 🎯 Ready for Agent Development

This configuration provides:

- ✅ Optimized agent development environment
- ✅ Event-driven architecture support
- ✅ Neural atom development patterns
- ✅ Comprehensive testing framework
- ✅ Quality automation pipeline
- ✅ Copilot Co-Architect integration

**Get started**: Open any Python file and press `Ctrl+Shift+Alt+A` to begin agent development!
