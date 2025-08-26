# Documentation Instructions

This directory contains comprehensive project documentation for Super Alita.

## Documentation Structure

The docs directory provides detailed information about various aspects of the system:

- **Architecture** (`architecture.md`): System design and component overview
- **Architectural Overview** (`01_architectural_overview.md`): High-level system design and principles
- **Refactoring Guide** (`02_refactoring_guide.md`): Safe update practices for the streaming runtime
- **Agentic Workflows** (`03_agentic_workflows.md`): Continuous operational cycle
- **Advanced Development Patterns** (`04_advanced_patterns.md`): Patterns for building reliable agents
- **MCP Integration** (`mcp.md`, `mcp_integration_summary.md`): Model Context Protocol documentation
- **Runtime** (`runtime.md`): Runtime environment and execution details
- **Memory & Knowledge** (`memory.md`, `atoms-bonds.md`): Knowledge graph and memory systems
- **Agent Operations** (`agents.md`): Agent capabilities and workflow documentation
- **Testing & Diagnostics** (`testing.md`, `diagnostics.md`): Testing procedures and debugging
- **Integration Guides**: Service-specific integration documentation
- **Security**: Security-related documentation and guidelines

## Editing Documentation

### Markdown Standards
- Use **Markdown** for all documentation
- Follow consistent heading hierarchy (H1 for main sections, H2 for subsections)
- Include code blocks with proper language identifiers
- Use tables for structured data
- Add links to related documentation sections

### Content Guidelines
- **Clear and Concise**: Write for both technical and non-technical audiences
- **Code Examples**: Include working code snippets with proper syntax highlighting
- **Visual Aids**: Use diagrams and flowcharts where helpful
- **Cross-References**: Link to related documentation and source code
- **Version Information**: Keep documentation synchronized with code changes

### Documentation Templates

#### For Architecture Documentation
```markdown
# Component Name

## Overview
Brief description of the component's purpose and role.

## Architecture
Detailed architectural information.

## Configuration
Configuration options and environment variables.

## Usage Examples
Working code examples.

## API Reference
Interface and method documentation.

## Troubleshooting
Common issues and solutions.
```

#### For Integration Guides
```markdown
# Service Integration Guide

## Prerequisites
Required dependencies and setup.

## Installation
Step-by-step installation instructions.

## Configuration
Configuration examples and options.

## Usage
Practical usage examples.

## Testing
How to test the integration.

## Troubleshooting
Common issues and solutions.
```

## Updating Documentation

### Table of Contents
- Update the table of contents when adding new documents
- Maintain alphabetical ordering within categories
- Include brief descriptions for each document

### Cross-References
- Link to relevant source code files
- Reference related documentation sections
- Include links to external resources and dependencies

### Version Control
- Document breaking changes in architecture or APIs
- Include migration guides for major version updates
- Maintain changelog for documentation updates

## Generating Documentation

### Static Site Generation
If using a static site generator (like MkDocs or Sphinx):

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation
mkdocs build

# Serve locally for preview
mkdocs serve
```

### API Documentation
For auto-generated API documentation:

```bash
# Generate from source code
sphinx-apidoc -o docs/api src/
sphinx-build -b html docs/ docs/_build/
```

## Documentation Categories

### User Documentation
- Getting started guides
- Installation instructions
- Configuration examples
- Usage tutorials

### Developer Documentation  
- Architecture overviews
- API references
- Development workflows
- Testing procedures

### Operations Documentation
- Deployment guides
- Monitoring setup
- Troubleshooting procedures
- Performance tuning

### Integration Documentation
- Service integrations
- Plugin development
- MCP server setup
- External tool connections

## Style Guidelines

### Code Blocks
Always specify the language for syntax highlighting:

```python
# Python code example
from src.core.plugin_interface import PluginInterface

class MyPlugin(PluginInterface):
    pass
```

```bash
# Shell commands
python -m uvicorn src.main:app --reload
```

### Links and References
- Use descriptive link text instead of "click here"
- Include external link destinations
- Reference specific line numbers for source code links

### Images and Diagrams
- Store images in `docs/images/` subdirectory
- Use descriptive filenames
- Include alt text for accessibility
- Optimize images for web viewing

## Quality Assurance

### Documentation Review
- Verify all code examples work correctly
- Check links for accuracy and availability
- Test installation and setup instructions
- Validate configuration examples

### Accessibility
- Use proper heading structure
- Include alt text for images
- Ensure good color contrast
- Test with screen readers when possible

## Contributing to Documentation

1. Create a new branch for documentation changes
2. Follow the style guidelines outlined above
3. Test any code examples included
4. Update related cross-references
5. Submit a pull request with clear description of changes

For questions about documentation, refer to the main project issues or contact the maintainers.