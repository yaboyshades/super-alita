# Spec Kit Integration Quick Start

## Overview

This project integrates GitHub Spec Kit (v0.0.46+) for Spec-Driven Development (SDD) workflows. Spec Kit provides structured feature development with constitutional validation and automated artifact generation.

## Prerequisites

1. **Install Spec Kit**: `uvx spec-kit --help` (verifies v0.0.46+)
2. **Configure PowerShell Profile**: Add the `Invoke-SpecKitWorkflow` function to `$PROFILE`
3. **Verify Integration**: Run `spec "test feature" constitution` to test

## Quick Commands

### PowerShell Alias (Recommended)

```powershell
# Full workflow
spec "Add user authentication" all

# Individual phases
spec "Implement API endpoints" plan
spec "Database migration" tasks
```

### VS Code Tasks (Command Palette → Tasks: Run Task)

- **Spec Kit: Full Workflow** - Complete SDD pipeline
- **Spec Kit: Constitution** - Set up project constitution
- **Spec Kit: Specify** - Create feature specification
- **Spec Kit: Plan** - Generate implementation plan
- **Spec Kit: Tasks** - Break down into tasks
- **Spec Kit: Implement** - Execute implementation

### Copilot Agent Mode

When using GitHub Copilot Agent Mode:

1. Start with Spec Kit to scaffold features
2. Use structured prompts from `.github/prompts/`
3. Follow constitutional guidelines from `memory/sdd/`
4. Run quality gates: `pytest -q`, `ruff check .`, `mypy --strict src`

## Workflow Phases

1. **Constitution** - Define project principles and governance
2. **Specify** - Create detailed feature specifications
3. **Plan** - Design implementation architecture
4. **Tasks** - Break down into executable tasks
5. **Implement** - Code and validate the feature

## Key Files

- **Instructions**: `.github/copilot-instructions.md` (Agent Mode guidance)
- **Prompts**: `.github/prompts/` (Structured prompts for each phase)
- **Templates**: `templates/sdd/` (Artifact templates)
- **Constitution**: `memory/sdd/constitutional_sdd_framework.md`
- **Scripts**: `scripts/` (PowerShell automation)

## Quality Gates

After implementation:

```bash
pytest -q                      # Unit tests
ruff check src tests           # Linting
mypy --strict src core         # Type checking
python scripts/unified_sdd_mangle.py  # Constitution validation
```

## Troubleshooting

- **Spec Kit not found**: Run `uvx spec-kit --help` to install
- **PowerShell alias not working**: Check `$PROFILE` configuration
- **MCP server issues**: Verify `.vscode/settings.json` spec-kit configuration
- **Quality gate failures**: Review constitution compliance in `memory/sdd/`
