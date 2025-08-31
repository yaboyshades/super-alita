# Examples Instructions

This directory contains example implementations and demonstrations for Super Alita components.

## Overview

The examples directory provides practical demonstrations of Super Alita functionality:

- **Basic Usage**: Simple examples for getting started
- **Integration Examples**: Service integration demonstrations
- **Plugin Examples**: Sample plugin implementations
- **Workflow Examples**: Complete workflow demonstrations
- **API Examples**: REST API and client examples
- **Advanced Features**: Complex feature demonstrations

## Example Categories

### Getting Started Examples
- **Hello World**: Basic agent interaction
- **Simple Tool**: Minimal tool implementation
- **Event System**: Basic event bus usage
- **Plugin Creation**: Simple plugin development

### Integration Examples
- **MCP Integration**: Model Context Protocol examples
- **External APIs**: Third-party service integration
- **Database Integration**: Persistent storage examples
- **File Operations**: File system interaction examples

### Advanced Examples
- **Streaming Workflows**: Real-time processing examples
- **Multi-Agent Systems**: Agent coordination examples
- **Custom LLM Integration**: Alternative LLM provider setup
- **Performance Optimization**: Optimization techniques

## Running Examples

### Basic Example Execution
```bash
# From repository root
cd examples

# Run a specific example
python basic/hello_world.py

# Run with specific configuration
python integration/mcp_example.py --config config.yaml
```

### Interactive Examples
```bash
# Interactive agent demonstration
python interactive/agent_demo.py

# Follow prompts for input
# Examples will guide you through features
```

### Web-based Examples
```bash
# Start example web server
python web/example_server.py

# Open browser to http://localhost:8080
# Interact with web-based examples
```

## Example Structure

### Standard Example Format
Each example follows a consistent structure:

```python
#!/usr/bin/env python3
"""
Example: [Example Name]

Description: [Brief description of what this example demonstrates]

Prerequisites:
- [Required dependencies]
- [Configuration requirements]

Usage:
    python example_name.py [options]
"""

import asyncio
import logging
from pathlib import Path

# Configure logging for example
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main example execution."""
    logger.info("Starting example: [Example Name]")

    try:
        # Example implementation
        result = await demonstrate_feature()
        logger.info(f"Example completed successfully: {result}")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

def demonstrate_feature():
    """Demonstrate the specific feature."""
    # Implementation details
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration Files
Examples may include configuration files:

```yaml
# example_config.yaml
example:
  name: "Sample Example"
  parameters:
    api_key: "${API_KEY}"
    timeout: 30
    debug: true

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Creating New Examples

### Example Development Process
1. **Identify Use Case**: Choose a specific feature or integration to demonstrate
2. **Create Directory**: Organize by category (basic, integration, advanced)
3. **Implement Example**: Follow the standard example format
4. **Add Documentation**: Include README and inline comments
5. **Test Example**: Verify example works in clean environment
6. **Update Index**: Add to main examples documentation

### Example Template
```python
#!/usr/bin/env python3
"""
Example: [Your Example Name]

Description: [What this example demonstrates]

Prerequisites:
- Python 3.8+
- Super Alita dependencies
- [Any additional requirements]

Environment Variables:
- [Required environment variables]

Usage:
    python your_example.py [--option value]
"""

import argparse
import asyncio
import logging
from typing import Any, Dict

from src.core.plugin_interface import PluginInterface
from src.core.events import create_event

logger = logging.getLogger(__name__)

class ExamplePlugin(PluginInterface):
    """Example plugin implementation."""

    @property
    def name(self) -> str:
        return "example_plugin"

    async def shutdown(self) -> None:
        logger.info("Example plugin shutting down")

async def demonstrate_feature():
    """Demonstrate the main feature."""
    # Your implementation here
    plugin = ExamplePlugin()

    # Create and emit an event
    event = create_event("example_event", data={"message": "Hello World"})

    return {"status": "success", "plugin": plugin.name}

async def main():
    """Main example execution."""
    parser = argparse.ArgumentParser(description="Example description")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info("Starting example")

    try:
        result = await demonstrate_feature()
        logger.info(f"Example completed: {result}")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
```

## Example Documentation

### README Files
Each example category should include a README.md:

```markdown
# [Category] Examples

## Overview
Brief description of the example category.

## Examples
- `example1.py` - Description of first example
- `example2.py` - Description of second example

## Prerequisites
- Common prerequisites for all examples in category

## Usage
Common usage patterns for the category.
```

### Inline Documentation
- **Comprehensive Comments**: Explain each step clearly
- **Code Documentation**: Use docstrings for functions and classes
- **Configuration Examples**: Show sample configuration files
- **Error Scenarios**: Demonstrate error handling

## Testing Examples

### Automated Testing
```bash
# Run all example tests
pytest examples/tests/ -v

# Test specific category
pytest examples/tests/test_basic_examples.py -v

# Integration test examples
pytest examples/tests/test_integration_examples.py -v
```

### Manual Verification
```bash
# Verify examples work correctly
./scripts/verify_examples.sh

# Test specific example
python examples/basic/hello_world.py --test
```

### Environment Testing
Test examples in different environments:

```bash
# Test in minimal environment
docker run -v $(pwd):/app python:3.8-slim bash -c "cd /app && python examples/basic/hello_world.py"

# Test with full dependencies
python examples/integration/full_system_example.py
```

## Best Practices

### Code Quality
- **Clear and Readable**: Write self-explanatory code
- **Error Handling**: Include proper exception handling
- **Resource Cleanup**: Ensure proper resource management
- **Type Hints**: Use type annotations for clarity

### Educational Value
- **Progressive Complexity**: Start simple, build complexity gradually
- **Real-world Scenarios**: Use practical, realistic examples
- **Multiple Approaches**: Show different ways to accomplish tasks
- **Common Pitfalls**: Demonstrate how to avoid common mistakes

### Maintenance
- **Version Compatibility**: Ensure examples work with current version
- **Dependency Updates**: Keep dependencies up to date
- **Regular Testing**: Test examples in CI/CD pipeline
- **Documentation Updates**: Keep documentation synchronized with code

## Troubleshooting Examples

### Common Issues
- **Missing Dependencies**: Install required packages
- **Environment Variables**: Set required environment variables
- **Configuration Errors**: Check configuration file syntax
- **Permission Issues**: Verify file and directory permissions

### Debugging Examples
```bash
# Run example with debug logging
python example.py --verbose

# Use Python debugger
python -m pdb example.py

# Check environment
python -c "import sys; print(sys.path)"
```

### Getting Help
- **Read Documentation**: Check inline comments and README files
- **Check Prerequisites**: Verify all requirements are met
- **Review Logs**: Look at debug output for clues
- **Ask Questions**: Open issues for specific problems

For more specific guidance on individual examples, refer to the README files in each example subdirectory.
