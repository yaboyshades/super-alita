# Super Alita Python Snippets

Comprehensive Python code snippets integrated with the Copilot Prompt Optimizer extension, specifically designed for Super Alita development patterns.

## Features

- **100+ Python snippets** covering common patterns and Super Alita specific code
- **Intelligent categorization** with quick browse and search functionality
- **Type hints and docstrings** following Black 88-character formatting
- **Super Alita integration** for plugins, consensus, MCP tools, and events
- **VS Code integration** with keyboard shortcuts and command palette

## Quick Start

### Keyboard Shortcuts (Python files)

- `Ctrl+Shift+S` - Insert Super Alita Snippet
- `Ctrl+Shift+B` - Browse Snippet Library by Category
- `Ctrl+Shift+F` - Search Snippets

### Command Palette Commands

- "Insert Super Alita Snippet"
- "Browse Snippet Library"
- "Search Snippets"
- "Insert Snippet by Prefix"

## Snippet Categories

### Super Alita Patterns

- **alitaplugin** - Complete plugin class with event handling
- **consensus** - Enhanced consensus provider usage
- **mcptool** - MCP tool registration with ability registry
- **createevent** - Event creation following Super Alita patterns

### Functions & Methods

- **func** - Function with type hints and Google-style docstring
- **asyncfunc** - Async function with proper typing
- **lambda** - Lambda function

### Classes & Objects

- **mainclass** - Class with `__init__` and type hints
- **subclass** - Subclass with `super()` call

### Imports & Modules

- **pathlib** - pathlib.Path import (Super Alita standard)
- **imdt** - datetime with timezone support
- **imnp** - numpy as np
- **impd** - pandas as pd

### Control Flow

- **ifelse** - if/else statement
- **tryexcept** - try/except with specific exception
- **fori** - for loop
- **while** - while loop

### Testing & Pytest

- **asynctest** - Async test with pytest.mark.asyncio
- **parametrize** - Parametrized test

### File Operations

- **openfile** - File context manager
- **listcomp** - List comprehension with condition

## Super Alita Integration

All snippets follow established Super Alita patterns:

- **Type hints everywhere** - Functions have return type annotations
- **Proper docstrings** - Google/NumPy style with Args/Returns sections
- **Black formatting** - 88-character line length, trailing commas
- **pathlib over os.path** - Modern path handling
- **Timezone-aware datetime** - Always use `datetime.now(timezone.utc)`
- **Absolute imports** - `from src.core.events import create_event`

## Usage Examples

### Plugin Development

```python
# Type: alitaplugin
class MyPlugin(PluginInterface):
    """Plugin description"""

    async def initialize(self, event_bus, **kwargs) -> bool:
        """Initialize the plugin"""
        self.event_bus = event_bus
        return True
```

### Consensus Usage

```python
# Type: consensus
provider = EnhancedConsensusProvider(config={
    "base_url": "http://localhost:11434/v1",
    "model_name": "gpt-oss:20b",
    "timeout": 60
})

result = await provider.consensus_sampling(
    prompt="Your question",
    method="weighted_vote",
    confidence_threshold=0.7
)
```

### MCP Tool Registration

```python
# Type: mcptool
contract = AbilityContract(
    id="tool_id",
    name="Tool Name",
    description="Tool description",
    parameters={
        "type": "object",
        "properties": {
            "param_name": {"type": "string"}
        },
        "required": ["param_name"]
    }
)

async def tool_executor(args: dict[str, Any]) -> dict[str, Any]:
    """Execute the tool"""
    return {"result": "success"}

ability_registry.register_tool(contract, tool_executor)
```

## Backend Integration

The snippets are also available via the Python backend:

```python
from src.vscode_integration.snippet_library import SnippetLibrary

library = SnippetLibrary()
await library.initialize(event_bus)

# Get snippets by category
snippets = await library.get_snippets_by_category("super_alita")

# Search snippets
results = await library.search_snippets("consensus")

# Validate snippet quality
is_valid = await library.validate_snippet_quality(snippet)
```

## Contributing

To add new snippets:

1. Edit `snippets/python.json` with new snippet definitions
2. Update `src/vscode_integration/snippet_library.py` for backend integration
3. Follow Super Alita code standards (type hints, docstrings, Black formatting)
4. Test with `npm run compile && npm run package`

## Architecture

- **Frontend**: TypeScript VS Code extension with snippet browser
- **Backend**: Python plugin system with quality validation
- **Integration**: MCP (Model Context Protocol) for tooling
- **Standards**: Black, Ruff, mypy strict mode, pytest async patterns
