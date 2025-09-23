# Qoder IDE Configuration for Super Alita

This directory contains the Qoder IDE configuration for the Super Alita project, mirroring and extending the VS Code Copilot setup found in `.vscode/` and `.github/`.

## Configuration Files

### Core Configuration

- `settings.json` - Main IDE settings including Python environment, formatting, linting, and Super Alita specific configurations
- `keybindings.json` - Custom keyboard shortcuts for SDD workflows, quality gates, and AI assistant integration
- `tasks.json` - Automated tasks for development, testing, validation, and deployment
- `launch.json` - Debug configurations for various Super Alita components
- `extensions.json` - Recommended extensions for optimal development experience

### AI Assistant Configuration

- `ai-instructions.md` - Comprehensive instructions for Qoder's AI assistant, including project context, development guidelines, and constitutional compliance requirements
- `snippets/python.json` - Constitutional SDD-compliant code snippets for rapid development

### Workspace Configuration

- `workspace.code-workspace` - Multi-folder workspace configuration with virtual folders for organized development
- `README.md` - This documentation file

## Key Features

### Constitutional SDD Integration

- **Constitutional Compliance**: All development follows the 6-article Constitutional Framework
- **SDD Pipeline**: Integrated Spec → Plan → Tasks → Implementation → Validation workflow
- **Quality Gates**: Automated constitutional validation with ≥0.75 compliance threshold
- **Unified Orchestration**: Event-driven pipeline execution with observability

### AI Assistant Capabilities

- **Context-Aware**: Deep understanding of Super Alita architecture and patterns
- **Constitutional Guidance**: Ensures all suggestions follow constitutional principles
- **SDD Workflow**: Integrated with Spec-Driven Development processes
- **Quality Assurance**: Built-in validation and testing recommendations

### Development Tools

- **Python Toolchain**: Black, Ruff, MyPy, pytest integration
- **MCP Protocol**: Model Context Protocol server support
- **Ollama Integration**: Local AI model support (GPT-OSS:20B)
- **Mangle Rules**: Code quality validation with dependency analysis

## Quick Start

1. **Environment Setup**:

   ```powershell
   python -m venv .venv
   . .venv/Scripts/Activate.ps1
   pip install -r requirements.txt -c constraints.txt
   ```

2. **Validate Installation**:

   - Use task: `Ctrl+Shift+Alt+V` or run `python validate_deployment.py`

3. **Start Development Server**:

   - Use task: `Ctrl+Alt+Shift+S` or run `uvicorn app:app --reload --port 8080`

4. **Run Quality Pipeline**:
   - Use task: `Ctrl+Shift+Alt+Q` for full quality check

## Key Keyboard Shortcuts

### AI Assistant

- `Ctrl+Shift+Alt+I` - Open AI chat
- `Ctrl+Shift+A` - AI chat with selection
- `Ctrl+Alt+Space` - Trigger AI suggestions

### SDD Workflow

- `Ctrl+Alt+P` - New SDD plan
- `Ctrl+Alt+Shift+C` - Constitutional validation
- `Ctrl+Alt+Shift+S` - SDD validation
- `Ctrl+Alt+Shift+U` - Unified orchestrator

### Quality & Testing

- `Ctrl+Shift+Alt+Q` - Full quality pipeline
- `Ctrl+Shift+Alt+T` - Run tests
- `Ctrl+Shift+Alt+L` - Lint code
- `Ctrl+Shift+Alt+F` - Format code

### Navigation

- `Ctrl+Shift+Alt+1` - Core source
- `Ctrl+Shift+Alt+2` - Plugins
- `Ctrl+Shift+Alt+3` - Tests
- `Ctrl+Shift+Alt+4` - SDD modules
- `Ctrl+Shift+Alt+5` - Unified Intelligence

## Constitutional Compliance

All development in Qoder IDE follows the Constitutional Framework:

1. **Library-First**: Prefer existing libraries and patterns
2. **Test-First**: Comprehensive testing for all components
3. **Simplicity**: Single responsibility, minimal complexity
4. **Integration**: Compatible with existing architecture
5. **Clarity**: Clear purpose and documentation
6. **Counterfactual**: Robust error handling and edge cases

## SDD Workflow Integration

The IDE is configured for Spec-Driven Development:

1. **Specify** (`/sdd/specify`) - Constitutional specification generation
2. **Plan** (`/sdd/plan`) - Plan creation with compliance validation
3. **Tasks** (`/sdd/tasks`) - Task breakdown with quality gates
4. **Validate** (`/sdd/validate`) - End-to-end compliance verification

Each stage maintains constitutional compliance ≥0.75 threshold.

## AI Assistant Guidelines

The Qoder AI assistant is configured with:

- **Deep Project Knowledge**: Understanding of Super Alita architecture
- **Constitutional Awareness**: All suggestions follow constitutional principles
- **SDD Integration**: Workflow-aware assistance
- **Quality Focus**: Built-in validation and testing guidance
- **Error Prevention**: Common issue identification and resolution

## Troubleshooting

### Common Issues

- **Constitutional validation failing**: Check SDD files for required elements
- **Import errors**: Ensure absolute imports from `src.*`
- **Quality gate failures**: Run full toolchain before committing
- **MCP connection issues**: Verify server is running on port 3000

### Diagnostic Commands

- `python validate_deployment.py` - System health check
- `python validate_sdd_epic.py` - SDD validation
- `ruff check src/` - Code linting
- `mypy --strict src/` - Type checking
- `pytest -q` - Test suite

## Advanced Features

### Mangle Integration

- **Code Reasoning**: SQLite-based fact store for dependency analysis
- **Rule Engine**: Constitutional compliance rules and validation
- **Quality Gates**: Automated rule execution in CI/CD

### Unified Orchestrator

- **Event-Driven**: Pipeline execution with observability
- **Run Ledger**: Audit trails and replay capabilities
- **Consensus Integration**: Enhanced decision-making

### Spec Kit Integration

- **Template Generation**: Automated SDD artifact creation
- **PowerShell Automation**: Windows-native workflow scripts
- **Constitutional Gates**: Quality enforcement at each stage

## File Structure

```
.qoder/
├── README.md                    # This documentation
├── settings.json               # Main IDE configuration
├── keybindings.json           # Keyboard shortcuts
├── tasks.json                 # Automated tasks
├── launch.json                # Debug configurations
├── extensions.json            # Recommended extensions
├── ai-instructions.md         # AI assistant configuration
├── workspace.code-workspace   # Multi-folder workspace
└── snippets/
    └── python.json           # Constitutional SDD snippets
```

This configuration provides a comprehensive development environment that maintains constitutional compliance while enabling rapid, high-quality development of Super Alita components.
