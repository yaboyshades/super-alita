# Super Alita Agent Instructions - Quick Reference

## Essential Files
- `DETAILED_AGENT_INSTRUCTIONS.md` - Comprehensive development guide
- `ADVANCED_DEVELOPMENT_PATTERNS.md` - Advanced patterns and production code
- `.github/copilot-instructions.md` - VS Code Copilot integration
- `AGENTS.md` - Repository guidelines and standards
- `src/reug_runtime/AGENTS.md` - Runtime-specific instructions

## Quick Start Checklist

### 🚀 Environment Setup
```bash
# 1. Environment configuration
cp .env.example .env              # Configure your environment
# Edit .env with your API keys and settings

# 2. Dependencies
make deps                         # Install all dependencies
# OR: pip install -r requirements.txt -r requirements-test.txt

# 3. Health check
make run                          # Start dev server
curl http://127.0.0.1:8080/healthz  # Verify health

# 4. Testing
make test                         # Run test suite
make lint                         # Check code quality
```

### 🔧 MCP Development
```bash
# MCP server setup (Windows PowerShell)
pwsh .\Setup-MCP.ps1 -Bootstrap  # Initialize MCP + VS Code
pwsh .\Setup-MCP.ps1 -Doctor     # Health check
pwsh .\Setup-MCP.ps1 -AddTool MyTool  # Scaffold new tools

# Verify MCP integration
# In VS Code: Ctrl+Shift+P → "MCP: Show Installed Servers"
```

## 🏗️ Architecture Overview

```
Super Alita System
├── Event-Driven Neural Architecture
│   ├── Redis/Memurai Event Bus (src/core/event_bus.py)
│   ├── Neural Atoms & Bonds (src/core/neural_atom.py)
│   └── Plugin Communication (src/core/plugin_interface.py)
├── MCP Integration
│   ├── Tool Development (mcp_server/src/mcp_server/tools/)
│   ├── VS Code Integration
│   └── Agent Mode Support
├── Cognitive Fabric
│   ├── REUG Framework v3.7
│   ├── DTA 2.0 Processing
│   └── Deterministic UUIDs
└── Security & Sandboxing
    ├── Execution Sandbox (src/sandbox/)
    ├── Safe Code Execution
    └── Credential Management
```

## 📋 Code Standards Checklist

### ✅ Required Standards
- [ ] **Type hints** on all functions and methods
- [ ] **Black formatting** (88 characters)
- [ ] **Ruff linting** passes (see ruff.toml)
- [ ] **MyPy strict** type checking for core modules
- [ ] **Double quotes** for strings
- [ ] **pathlib.Path** instead of os.path
- [ ] **Async/await** for I/O operations
- [ ] **pytest** tests with ≥70% coverage

### ✅ Security Requirements
- [ ] **Never use** `eval()` or `exec()` - use `src/sandbox/exec_sandbox.py`
- [ ] **Never use** `subprocess` with `shell=True` - use `src/core/proc.py`
- [ ] **All credentials** from environment variables, never hardcoded
- [ ] **YAML operations** via `src/core/yaml_utils.py`
- [ ] **Sandbox all** dynamic code execution
- [ ] **Validate file paths** against workspace boundaries

## 🎯 Common Patterns

### Event Creation
```python
from src.core.events import create_event

# ✅ Correct - use keyword arguments
event = create_event(
    "cognitive_turn",
    turn_data=data,
    confidence=0.95,
    source_plugin="my_plugin"
)

# ❌ Wrong - no positional dicts
event = create_event("test", {"data": "value"})
```

### Plugin Development
```python
from src.core.plugin_interface import PluginInterface

class MyPlugin(PluginInterface):
    @property
    def name(self) -> str:
        return "my_plugin"
    
    async def initialize(self) -> bool:
        self.event_bus = event_bus  # Register event bus
        return True
    
    async def shutdown(self) -> None:
        # Cleanup logic
        pass
```

### Safe Execution
```python
# ✅ Safe execution
from src.sandbox.exec_sandbox import safe_execute
result = await safe_execute(code, context)

# ✅ Safe subprocess
from src.core.proc import run_command
result = run_command(["python", "script.py"])

# ❌ Never do this
eval(user_code)  # SECURITY RISK
subprocess.run(cmd, shell=True)  # SECURITY RISK
```

### Testing
```python
import pytest

@pytest.mark.asyncio  # Required for async tests
async def test_event_flow():
    # Use fixtures from conftest.py
    pass

@pytest.mark.integration_redis
async def test_redis_integration():
    # Tests requiring Redis
    pass
```

## 🛠️ MCP Tool Template

```python
from typing import Dict, Any
from pathlib import Path

def my_mcp_tool(
    file_path: str,
    operation: str = "analyze",
    dry_run: bool = True  # Always default to dry_run
) -> Dict[str, Any]:
    """MCP tool following Super Alita patterns."""
    
    try:
        # Validate workspace boundary
        target_path = Path(file_path).resolve()
        workspace_root = Path.cwd().resolve()
        
        if not str(target_path).startswith(str(workspace_root)):
            return {
                "success": False,
                "result": "",
                "error": "Path outside workspace boundary"
            }
        
        if dry_run:
            # Return diff preview
            return {
                "success": True,
                "result": "--- a/file.py\n+++ b/file.py\n...",
                "error": ""
            }
        
        # Implementation here
        return {
            "success": True,
            "result": "Operation completed",
            "error": ""
        }
        
    except Exception as e:
        return {
            "success": False,
            "result": "",
            "error": str(e)
        }
```

## 🔍 Debugging Commands

```bash
# Health checks
curl http://127.0.0.1:8080/healthz

# Debug utilities
python scripts/debug_fixed.py
python scripts/debug_matching.py
python scripts/utility_debug.py

# Testing patterns
pytest -q                         # Quick test run
pytest -k "test_name"            # Filter tests
pytest -m integration_redis     # Run marked tests
pytest --cov=src                 # Coverage report

# Code quality
pre-commit run --all-files       # Run all hooks
black src/ tests/                # Format code
ruff check src/ tests/           # Lint code
mypy src/core src/sandbox        # Type check
```

## 🚨 Critical Anti-Patterns

### ❌ Never Do These
```python
# Raw execution
eval(user_code)
exec(dynamic_code)
subprocess.run(cmd, shell=True)

# Hardcoded secrets
API_KEY = "secret-123"

# Missing type hints
def process_data(data):
    return data + 1

# Positional event arguments
create_event("test", {"data": "value"})

# Bypassing sandbox
os.system(command)

# Ignoring async patterns
def blocking_operation():
    time.sleep(10)  # Use await asyncio.sleep(10)
```

## 📊 Key Metrics to Monitor

- **Event Bus Latency**: <100ms for event processing
- **Test Coverage**: ≥70% required
- **Memory Usage**: Stable growth, <50MB per hour
- **Plugin Load Time**: <5 seconds per plugin
- **Redis Queue Depth**: <1000 events during normal operation

## 🎯 Development Workflow

1. **Feature Branch**: `feature/add-capability`
2. **Code**: Follow patterns in this guide
3. **Test**: Write tests first, achieve ≥70% coverage
4. **Lint**: `make lint` before commit
5. **Commit**: `[module] Short description`
6. **PR**: Include summary, rationale, tests
7. **Review**: Address feedback, maintain standards
8. **Merge**: Squash commits for clean history

## 🏷️ Environment Variables

```bash
# Execution Control
SUPER_ALITA_MODE=shadow          # shadow/act/batch
REUG_MAX_TOOL_CALLS=5
REUG_EXEC_TIMEOUT_S=20.0

# LLM Configuration
LLM_MODEL=auto                   # Enable fallback
GEMINI_API_KEY=your-key
OPENAI_API_KEY=your-key

# Infrastructure
REDIS_URL=redis://localhost:6379
REUG_EVENT_LOG_DIR=./logs/events
PYTHONPATH=./src
```

## 📚 Documentation Structure

```
docs/
├── agents.md                    # Living agent documentation
├── architecture.md             # System architecture
├── mcp.md                      # MCP integration guide
├── testing.md                  # Testing strategies
├── security/                   # Security guidelines
└── tools/                      # Tool documentation
```

## 🎉 Success Indicators

Your development follows Super Alita patterns when:

✅ All code passes `make lint` without warnings  
✅ Tests achieve ≥70% coverage with `pytest --cov`  
✅ Events use the `create_event()` pattern correctly  
✅ Plugins inherit from `PluginInterface`  
✅ MCP tools default to `dry_run=True`  
✅ No raw `eval/exec` or unsafe subprocess calls  
✅ Type hints are comprehensive and correct  
✅ Async patterns are used for I/O operations  
✅ Error handling is robust and informative  
✅ Documentation is updated for new features  

---

**Remember**: Super Alita is an event-driven, self-evolving system. Think in events, maintain safety, and leverage the cognitive fabric for intelligence. When in doubt, refer to the detailed instructions and advanced patterns documents.