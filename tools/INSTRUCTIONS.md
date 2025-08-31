# Tools Instructions

This directory contains standalone tools and utilities for Super Alita development and operations.

## Overview

The tools directory provides specialized utilities that support various aspects of Super Alita development:

- **Development Tools**: Code generation, analysis, and development utilities
- **Deployment Tools**: Automated deployment and configuration tools
- **Monitoring Tools**: System monitoring and health check utilities
- **Data Tools**: Data processing and manipulation utilities
- **Integration Tools**: External service integration utilities

## Tool Categories

### Development Tools
- **Code Generators**: Automated code scaffolding and template generation
- **Analysis Tools**: Code analysis, dependency checking, performance profiling
- **Debug Utilities**: Debugging aids and diagnostic tools
- **Testing Tools**: Test automation and validation utilities

### Operations Tools
- **Deployment**: Automated deployment scripts and configuration management
- **Monitoring**: Health checks, performance monitoring, alerting
- **Backup**: Data backup and recovery utilities
- **Maintenance**: System maintenance and cleanup tools

### Integration Tools
- **API Clients**: Client libraries for external services
- **Data Converters**: Format conversion and data transformation tools
- **Migration Tools**: Data migration and schema updates
- **Sync Utilities**: Data synchronization and replication tools

## Usage

### Tool Execution
Tools can be executed directly or through the main Super Alita system:

```bash
# Direct execution
python tools/tool_name.py [options]

# Through module path
python -m tools.tool_name [options]

# With specific configuration
python tools/tool_name.py --config config.yaml
```

### Common Tool Patterns
Most tools follow consistent command-line patterns:

```bash
# Help and documentation
python tools/tool_name.py --help

# Verbose output
python tools/tool_name.py --verbose

# Dry-run mode (no actual changes)
python tools/tool_name.py --dry-run

# Configuration file
python tools/tool_name.py --config path/to/config.yaml
```

## Tool Development

### Tool Template
```python
#!/usr/bin/env python3
"""
Tool: [Tool Name]

Description: [Brief description of tool purpose]

Usage:
    python tool_name.py [options]

Options:
    --help          Show this help message
    --verbose       Enable verbose output
    --dry-run       Show what would be done without making changes
    --config FILE   Configuration file path

Examples:
    python tool_name.py --verbose
    python tool_name.py --config config.yaml --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

class ToolError(Exception):
    """Custom exception for tool-specific errors."""
    pass

class ToolConfig:
    """Configuration management for tools."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if self.config_path and self.config_path.exists():
            # Load from YAML/JSON file
            import yaml
            return yaml.safe_load(self.config_path.read_text())
        return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration values."""
        return {
            "timeout": 30,
            "retry_attempts": 3,
            "output_format": "json"
        }

class ToolRunner:
    """Main tool execution class."""

    def __init__(self, config: ToolConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.results = []

    def execute(self) -> Dict[str, Any]:
        """Execute the main tool functionality."""
        logger.info("Starting tool execution")

        try:
            if self.dry_run:
                logger.info("Dry-run mode: showing what would be done")
                return self._simulate_execution()
            else:
                return self._perform_execution()

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise ToolError(f"Execution failed: {e}") from e

    def _simulate_execution(self) -> Dict[str, Any]:
        """Simulate tool execution for dry-run mode."""
        return {
            "status": "simulated",
            "actions": ["Action 1", "Action 2"],
            "dry_run": True
        }

    def _perform_execution(self) -> Dict[str, Any]:
        """Perform actual tool execution."""
        # Implement tool-specific logic here
        return {
            "status": "completed",
            "results": self.results,
            "dry_run": False
        }

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the tool."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tool description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Configuration file path"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
        # Load configuration
        config = ToolConfig(args.config)

        # Create and run tool
        runner = ToolRunner(config, args.dry_run)
        result = runner.execute()

        # Output results
        if args.verbose:
            logger.info(f"Tool completed successfully: {result}")
        else:
            print(f"Status: {result['status']}")

        return 0

    except ToolError as e:
        logger.error(f"Tool error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Tool execution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### Configuration Management
Tools should support flexible configuration:

```yaml
# tool_config.yaml
tool:
  name: "example_tool"
  version: "1.0.0"

settings:
  timeout: 30
  retry_attempts: 3
  parallel_execution: true

output:
  format: "json"  # json, yaml, text
  file: "output.json"
  verbose: false

logging:
  level: "INFO"
  file: "tool.log"
```

### Error Handling
Implement robust error handling:

```python
class ToolError(Exception):
    """Base exception for tool errors."""
    pass

class ConfigurationError(ToolError):
    """Configuration-related errors."""
    pass

class ExecutionError(ToolError):
    """Execution-related errors."""
    pass

class ValidationError(ToolError):
    """Validation-related errors."""
    pass

# Usage in tools
try:
    validate_input(data)
except ValidationError as e:
    logger.error(f"Input validation failed: {e}")
    return {"success": False, "error": str(e)}
```

## Integration with Super Alita

### Event System Integration
Tools can integrate with the Super Alita event system:

```python
from src.core.events import create_event

def emit_tool_event(tool_name: str, action: str, result: Any):
    """Emit a tool execution event."""
    event = create_event(
        "tool_execution",
        tool_name=tool_name,
        action=action,
        result=result,
        timestamp=datetime.now(timezone.utc)
    )
    # Event will be processed by the event bus
```

### Plugin Integration
Tools can be exposed as Super Alita plugins:

```python
from src.core.plugin_interface import PluginInterface

class ToolPlugin(PluginInterface):
    """Plugin wrapper for a tool."""

    def __init__(self, tool_class):
        self.tool_class = tool_class

    @property
    def name(self) -> str:
        return f"tool_{self.tool_class.__name__.lower()}"

    async def execute_tool(self, **kwargs):
        """Execute the wrapped tool."""
        tool = self.tool_class()
        return await tool.execute(**kwargs)

    async def shutdown(self) -> None:
        """Cleanup when shutting down."""
        pass
```

## Testing Tools

### Unit Testing
```python
import pytest
from tools.tool_name import ToolRunner, ToolConfig

def test_tool_execution():
    """Test basic tool execution."""
    config = ToolConfig()
    runner = ToolRunner(config, dry_run=True)

    result = runner.execute()

    assert result["status"] == "simulated"
    assert result["dry_run"] is True

@pytest.mark.asyncio
async def test_async_tool():
    """Test asynchronous tool functionality."""
    config = ToolConfig()
    runner = AsyncToolRunner(config)

    result = await runner.execute()

    assert result["status"] == "completed"
```

### Integration Testing
```bash
# Test tool in isolation
python -m pytest tools/tests/test_tool_name.py -v

# Test tool integration with Super Alita
python -m pytest tests/integration/test_tool_integration.py -v

# End-to-end testing
./scripts/test_tools_e2e.sh
```

### Manual Testing
```bash
# Test dry-run functionality
python tools/tool_name.py --dry-run --verbose

# Test with different configurations
python tools/tool_name.py --config test_config.yaml

# Test error scenarios
python tools/tool_name.py --invalid-option
```

## Tool Documentation

### Documentation Standards
Each tool should include:

```markdown
# Tool Name

## Purpose
Brief description of what the tool does.

## Usage
Command-line usage examples.

## Configuration
Configuration options and examples.

## Examples
Practical usage examples.

## Troubleshooting
Common issues and solutions.
```

### Code Documentation
- **Docstrings**: Comprehensive function and class documentation
- **Type Hints**: Full type annotation for clarity
- **Comments**: Explain complex logic and decisions
- **Examples**: Include usage examples in docstrings

## Best Practices

### Development Guidelines
- **Single Responsibility**: Each tool should have a clear, focused purpose
- **Idempotent Operations**: Tools should be safe to run multiple times
- **Configurable**: Support configuration files and command-line options
- **Logging**: Provide appropriate logging for debugging and monitoring

### User Experience
- **Helpful Error Messages**: Provide clear, actionable error messages
- **Progress Indicators**: Show progress for long-running operations
- **Dry-Run Support**: Allow users to preview changes before applying
- **Documentation**: Include comprehensive help and examples

### Security Considerations
- **Input Validation**: Validate all inputs thoroughly
- **Path Safety**: Prevent directory traversal attacks
- **Credential Handling**: Secure handling of sensitive information
- **Permission Checks**: Verify permissions before operations

## Troubleshooting

### Common Issues
- **Permission Errors**: Check file and directory permissions
- **Configuration Issues**: Validate configuration file syntax
- **Dependency Problems**: Verify all required dependencies are installed
- **Environment Issues**: Check environment variables and paths

### Debugging Tools
```bash
# Run with debug logging
python tools/tool_name.py --verbose

# Use Python debugger
python -m pdb tools/tool_name.py

# Check tool configuration
python -c "from tools.tool_name import ToolConfig; print(ToolConfig().config)"
```

For specific tool documentation and usage, refer to the individual tool files and their embedded documentation.
