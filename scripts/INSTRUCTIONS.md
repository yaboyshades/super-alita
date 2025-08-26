# Scripts Instructions

This directory contains utility scripts for Super Alita development, deployment, and maintenance.

## Script Categories

### Development Scripts
- **Setup Scripts**: Environment initialization and dependency management
- **Code Generation**: Automated code scaffolding and template generation
- **Documentation**: Living document generation and updates
- **Validation**: System checks and health verification

### Automation Scripts
- **Build Automation**: Compilation and packaging workflows
- **Testing**: Test execution and validation scripts
- **Deployment**: Production deployment and configuration
- **Monitoring**: Health checks and performance monitoring

## Usage

### Making Scripts Executable
Scripts in this directory are executable in the dev container:

```bash
# Make script executable
chmod +x script-name.sh

# Run the script
./script-name.sh
```

### Python Scripts
```bash
# Run Python utility scripts
python script-name.py

# With module path (if needed)
python -m scripts.script_name
```

### PowerShell Scripts
```bash
# Run PowerShell scripts
pwsh script-name.ps1
```

## Key Scripts

### Setup and Initialization
- `setup_ui_codegen.sh` - UI codegen environment setup
- Environment setup scripts for various components

### Documentation Management
- `update_agents_md.py` - Updates living documentation (agents.md)
  - Scans source code for plugin and ability information
  - Updates ownership information from CODEOWNERS
  - Maintains session ledger and changelog
  - Usage: `python scripts/update_agents_md.py`

### Validation and Testing
- Various validation scripts for system health checks
- Deployment validation and integration testing
- Debug utilities for troubleshooting

### Automation Workflows
- Git workflow automation
- CI/CD pipeline helpers
- Performance monitoring scripts

## Adding New Scripts

### Script Naming
- Use descriptive, hyphenated names: `setup-environment.sh`
- Include file extensions: `.py`, `.sh`, `.ps1`
- Group related scripts with common prefixes: `validate-*`, `setup-*`

### Script Structure

#### Shell Scripts
```bash
#!/usr/bin/env bash
set -euo pipefail

# Script description and usage
# Usage: ./script-name.sh [options]

# Load environment if present
if [[ -f .env ]]; then
    set -a && . ./.env && set +a
fi

# Main script logic
main() {
    echo "Starting script execution..."
    # Implementation here
}

main "$@"
```

#### Python Scripts
```python
#!/usr/bin/env python3
"""
Script description and purpose.

Usage:
    python script-name.py [options]

Environment Variables:
    VAR_NAME - Description of variable
"""

import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--option", help="Option description")
    args = parser.parse_args()
    
    # Implementation here
    logger.info("Script completed successfully")

if __name__ == "__main__":
    main()
```

### Documentation Requirements
All scripts must include:
- **Purpose**: Clear description of what the script does
- **Usage**: Command-line usage examples
- **Prerequisites**: Required dependencies or setup
- **Environment Variables**: Any required environment configuration
- **Examples**: Practical usage examples

### Error Handling
- Use `set -euo pipefail` in bash scripts for robust error handling
- Include meaningful error messages
- Provide recovery suggestions when possible
- Log errors appropriately

### Testing Scripts
- Include basic validation in scripts when possible
- Test scripts in the development environment before committing
- Document any external dependencies or requirements

## Environment Dependencies

### Required Tools
Scripts may depend on:
- **Git**: Version control operations
- **Python 3.8+**: Python script execution
- **PowerShell**: Cross-platform PowerShell scripts
- **Docker**: Container operations
- **Node.js**: JavaScript/TypeScript tools

### Environment Variables
Scripts may use environment variables from `.env`:
- `PYTHONPATH` - Python module path
- API keys for external services
- Configuration parameters for various components

## Execution Context

### Working Directory
Scripts should be run from the repository root unless otherwise specified:

```bash
# From repository root
./scripts/script-name.sh

# Or with explicit path
cd /path/to/super-alita
./scripts/script-name.sh
```

### Permissions
- Scripts must have appropriate execute permissions
- Avoid requiring sudo unless absolutely necessary
- Document any elevated permission requirements

## Integration with Build System

### Makefile Integration
Scripts can be integrated with the main Makefile:

```makefile
script-target:
	./scripts/script-name.sh
```

### CI/CD Integration
Scripts used in CI/CD pipelines should:
- Have robust error handling
- Produce structured output for parsing
- Support non-interactive execution
- Include timeout handling

## Debugging and Troubleshooting

### Debug Mode
Enable debug output in scripts:

```bash
# Bash scripts
set -x  # Enable debug tracing

# Python scripts with verbose logging
python script-name.py --verbose
```

### Log Files
Scripts should log to appropriate locations:
- Development: Console output
- Production: Structured log files
- CI/CD: Pipeline-compatible output

For questions about specific scripts or to report issues, refer to the script documentation or open an issue in the project repository.