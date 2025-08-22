# Super Alita Agent Development - Quick Start Guide

## 🚀 Environment Setup (30 seconds)

### Prerequisites

- Python 3.8+
- Redis/Memurai running on localhost:6379
- VS Code with recommended extensions

### Instant Setup

```powershell
# 1. Clone and setup
git clone <repo-url> super-alita
cd super-alita

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Redis (required for event bus)
redis-server --port 6379 --bind 127.0.0.1

# 5. Verify setup
python quick_status_check.py
```

## 🎯 Core Workflows (Essential Hotkeys)

### Agent Development

- **`Ctrl+Shift+Alt+A`** - Start agents in dev mode
- **`Ctrl+Shift+Alt+V`** - Run validation suite
- **`Ctrl+Shift+Alt+H`** - Health check
- **`Ctrl+Shift+Alt+Q`** - Full quality pipeline

### MCP Operations

- **`Ctrl+Alt+M`** - Start MCP server
- **`Ctrl+Alt+Shift+M`** - Stop MCP server
- **`Ctrl+Alt+T`** - Run MCP tests

### Testing & Quality

- **`Ctrl+Shift+Alt+T`** - Fast pytest
- **`Ctrl+Shift+Alt+L`** - Lint code
- **`Ctrl+Shift+Alt+Y`** - Type check

## 🧠 Copilot Co-Architect Mode

### Activation

```
Ctrl+Shift+Alt+C - Switch to Co-Architect mode
Ctrl+Shift+Alt+I - Open Copilot chat with agent context
```

### Key Features

- Event contract enforcement
- Neural atom patterns
- Cognitive loop optimization
- Batch processing guidance
- Deterministic testing patterns

## 📁 Project Navigation

### Quick File Access

- **`Ctrl+Shift+Alt+1`** - Core files (`src/core/`)
- **`Ctrl+Shift+Alt+2`** - Plugin files (`src/plugins/`)
- **`Ctrl+Shift+Alt+3`** - Test files (`tests/`)

### Key Directories

```
src/core/          # Event bus, neural atoms, global workspace
src/plugins/       # Agent plugins and cognitive modules
tests/             # Comprehensive test suite
scripts/           # Development and deployment scripts
.vscode/           # VS Code configuration and tasks
.github/copilot/   # Copilot instructions and modes
```

## 🔧 Development Tasks

### Available Tasks (Command Palette: "Tasks: Run Task")

- `agents:dev` - Start agent development mode
- `alita:validate` - Comprehensive validation suite
- `quality:full` - Complete quality pipeline (lint, format, type, test)
- `health:check` - System health verification
- `redis:start` - Start Redis server
- `mcp:server` - Start MCP server

### Chemistry Plans

- `chemistry:new-plan` - Create new chemistry-infused plan
- `chemistry:open-prompt` - Open chemistry planner prompt

## 🎯 Agent Development Patterns

### Event-Driven Architecture

```python
# Always use structured events
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

```python
class ExampleAtom(NeuralAtom):
    async def execute(self, input_data: Any) -> Any:
        # Implementation
        pass
    
    def get_embedding(self) -> List[float]:
        # Semantic embedding
        pass
    
    def can_handle(self, task: str) -> float:
        # Confidence score 0-1
        pass
```

### Testing Pattern

```python
@pytest.fixture
def mock_event_bus():
    return AsyncMock(spec=EventBus)

async def test_event_handling(mock_event_bus):
    # Deterministic testing with UUIDv5
    test_id = generate_deterministic_id("test", "scenario")
    # Test implementation
```

## 🚨 Troubleshooting

### Common Issues

1. **Redis not running**: Start with `redis-server` or use `Ctrl+Shift+Alt+R`
2. **Event delivery failure**: Check Redis connection and pub/sub
3. **Plugin conflicts**: Verify plugin order in main.py
4. **Performance issues**: Monitor event throughput with health check

### Debug Commands

```powershell
# System health
python quick_status_check.py

# Comprehensive validation
python comprehensive_validation_suite.py

# MCP testing
.\scripts\alita-mcp.ps1 -RunTests

# Agent monitoring
python monitor_agent_detailed.py
```

## 🔄 Development Cycle

### 1. Code → Test → Validate

```powershell
# Write code...
# Quick test
Ctrl+Shift+Alt+T

# Full validation
Ctrl+Shift+Alt+V
```

### 2. Quality Assurance

```powershell
# Full quality pipeline
Ctrl+Shift+Alt+Q
```

### 3. Agent Testing

```powershell
# Start agent in dev mode
Ctrl+Shift+Alt+A

# Health monitoring
Ctrl+Shift+Alt+H
```

## 📋 Profile Export/Import

### Share Configuration

1. Command Palette → "Preferences: Export Profile"
2. Select "Super Alita Agent Development"
3. Share generated URL with team

### Import Profile

1. Command Palette → "Preferences: Import Profile"
2. Use provided URL
3. Select components to import

---

## 🎯 Ready to Code

With this setup, you're ready for high-performance agent development with:

- ✅ Event-driven architecture
- ✅ Cognitive loop patterns
- ✅ Neural atom development
- ✅ Comprehensive testing
- ✅ Quality automation
- ✅ Copilot optimization

**Start developing**: `Ctrl+Shift+Alt+A` → Begin agent development mode!
