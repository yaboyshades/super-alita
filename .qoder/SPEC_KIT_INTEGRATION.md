# GitHub Spec Kit Integration Guide

This document explains how to use the GitHub Spec Kit integration in Qoder IDE for Super Alita development, ensuring you're always working through a spec-driven lens with constitutional compliance.

## 🚀 Quick Start

### 1. Install Spec Kit

```bash
# Install or update to latest version
uvx --upgrade spec-kit

# Verify installation
uvx spec-kit --version
```

### 2. Initialize Project

```bash
# Set up Spec Kit configuration
uvx spec-kit init --constitutional --threshold 0.75

# Verify project status
uvx spec-kit status --verbose
```

## 🎯 Core Workflow

### Keyboard Shortcuts (Spec-Driven Development)

| Shortcut              | Command              | Description                        |
| --------------------- | -------------------- | ---------------------------------- |
| `Ctrl+S Ctrl+C`       | Constitution         | Establish constitutional framework |
| `Ctrl+S Ctrl+S`       | Specify              | Generate feature specifications    |
| `Ctrl+S Ctrl+P`       | Plan                 | Create implementation plans        |
| `Ctrl+S Ctrl+T`       | Tasks                | Break down into actionable tasks   |
| `Ctrl+S Ctrl+I`       | Implement            | Execute implementation             |
| `Ctrl+S Ctrl+V`       | Validate             | Comprehensive validation           |
| `Ctrl+S Ctrl+W`       | Full Workflow        | Execute complete SDD pipeline      |
| `Ctrl+S Ctrl+Q`       | Quick Spec           | Rapid specification generation     |
| `Ctrl+S Ctrl+Enter`   | Interactive          | Interactive workflow mode          |
| `Ctrl+S Ctrl+Space`   | Status               | Check project status               |
| `Ctrl+S Ctrl+Shift+C` | Constitutional Check | Validate constitutional compliance |

## 📋 Spec Kit Workflow Phases

### Phase 1: Constitutional Framework (`Ctrl+S Ctrl+C`)

Establishes the constitutional foundation for your feature:

- Validates against 6-article constitutional framework

- Ensures ≥0.75 compliance threshold
- Sets constitutional constraints for development

**Example:**

```bash

uvx spec-kit constitution "user-authentication-system"
```

### Phase 2: Feature Specification (`Ctrl+S Ctrl+S`)

Creates detailed feature specifications:

- User stories and acceptance criteria
- Technical requirements
- Constitutional compliance mapping
- Integration points

**Example:**

```bash
uvx spec-kit specify "user-authentication-system"
```

### Phase 3: Implementation Planning (`Ctrl+S Ctrl+P`)

Develops comprehensive implementation strategy:

- Technical architecture decisions

- Constitutional compliance strategy

- Risk assessment and mitigation
- Resource allocation

**Example:**

```bash

uvx spec-kit plan "user-authentication-system"
```

### Phase 4: Task Breakdown (`Ctrl+S Ctrl+T`)

Creates actionable development tasks:

- Granular task definitions
- Constitutional compliance per task
- Dependencies and sequencing
- Effort estimation

**Example:**

```bash
uvx spec-kit tasks "user-authentication-system"
```

### Phase 5: Implementation (`Ctrl+S Ctrl+I`)

Guides code implementation:

- Constitutional code generation
- Quality gate enforcement

- Real-time compliance monitoring
- Integration validation

**Example:**

```bash
uvx spec-kit implement "user-authentication-system"
```

### Phase 6: Validation (`Ctrl+S Ctrl+V`)

Comprehensive feature validation:

- Constitutional compliance verification
- Test coverage validation
- Integration testing
- Quality gate confirmation

**Example:**

```bash
uvx spec-kit validate "user-authentication-system"
```

## 🏗️ Constitutional Framework Integration

### The 6 Constitutional Articles

1. **Library-First**: Leverage existing libraries and established patterns
2. **Test-First**: Comprehensive testing for all components
3. **Simplicity**: Single responsibility, minimal complexity
4. **Integration**: Compatible with existing architecture
5. **Clarity**: Clear purpose and documentation

6. **Counterfactual**: Robust error handling and edge cases

### Constitutional Validation

Every Spec Kit phase includes constitutional validation:

- **Threshold**: ≥0.75 compliance score required

- **Blocking**: Non-compliant features cannot proceed

- **Real-time**: Continuous monitoring during development
- **Reporting**: Detailed compliance scoring and violation reports

## 🔧 Advanced Features

### Interactive Mode (`Ctrl+S Ctrl+Enter`)

Provides guided workflow with:

- Step-by-step constitutional guidance
- Interactive compliance checking
- Real-time feedback and suggestions
- Contextual help and documentation

### Project Status (`Ctrl+S Ctrl+Space`)

Comprehensive project overview:

- Overall constitutional compliance score
- Feature development status
- Quality gate status
- Pending constitutional violations

### Constitutional Compliance Check (`Ctrl+S Ctrl+Shift+C`)

Dedicated constitutional validation:

- Article-by-article compliance scoring
- Violation identification and remediation suggestions
- Trend analysis and improvement recommendations
- Integration with Mangle rule engine

## 📁 Project Structure

Spec Kit creates and maintains organized project structure:

```
.spec-kit/

├── config.yaml              # Spec Kit configuration

└── templates/               # Custom templates

specs/
├── feature-specs/           # Generated specifications
├── implementation-plans/    # Implementation strategies
├── task-breakdowns/        # Detailed task definitions

└── validation-reports/     # Validation results


artifacts/
├── constitutional-reports/  # Constitutional compliance reports
├── quality-metrics/        # Quality and performance metrics
└── generated-code/         # Spec Kit generated code
```

## 🎨 Code Snippets

Qoder IDE includes Spec Kit-aware snippets:

- `spec-feature` - Feature specification template
- `spec-plan` - Implementation plan template
- `spec-task` - Task definition template
- `spec-constitutional` - Constitutional compliance validator

## 🔄 Integration with Super Alita

### SDD Pipeline Integration

Spec Kit integrates seamlessly with Super Alita's SDD pipeline:

- `/sdd/specify` endpoints use Spec Kit specifications
- Constitutional validation enforced at each SDD stage
- Unified orchestrator coordinates Spec Kit workflows
- Event emission for observability and monitoring

### Mangle Rules Integration

Constitutional compliance backed by Mangle rule engine:

- Real-time code quality validation

- Dependency analysis and circular dependency detection
- Hot path identification and optimization
- Configuration cascade validation

### Unified Orchestrator

Event-driven workflow orchestration:

- Spec Kit phases trigger orchestrator events
- Constitutional gates enforced through orchestrator
- Observability and metrics collection
- Run ledger for audit trails and replay

## 🚦 Quality Gates

### Automatic Quality Gates

- **Constitutional Compliance**: ≥0.75 threshold
- **Test Coverage**: ≥85% code coverage
- **Code Quality**: Ruff, MyPy, Black compliance
- **Documentation**: Complete specification and documentation
- **Integration**: Successful integration tests

### Manual Quality Gates

- **Architectural Review**: Design pattern compliance
- **Security Review**: Security best practices validation
- **Performance Review**: Performance requirements validation
- **User Experience Review**: UX and accessibility validation

## 🛠️ Troubleshooting

### Common Issues

**Spec Kit Not Found**

```bash
# Install or update Spec Kit
uvx --upgrade spec-kit
uvx spec-kit --version
```

**Constitutional Validation Failing**

```bash
# Check detailed compliance report
uvx spec-kit constitutional --check --verbose --threshold 0.75

# Review specific violations

uvx spec-kit validate --constitutional-report
```

**Template Errors**

```bash
# Verify template configuration
uvx spec-kit status --templates



# Reset to default templates
uvx spec-kit reset --templates
```

**Integration Issues**

```bash

# Check Super Alita integration
python validate_deployment.py

# Verify SDD pipeline connectivity
curl -fsS http://127.0.0.1:8080/health/simple
```

## 📚 Best Practices

### Spec-Driven Development Workflow

1. **Always start with Constitution** - Establish constitutional framework first
2. **Iterative Specification** - Refine specifications based on constitutional feedback
3. **Constitutional Gates** - Never skip constitutional validation
4. **Continuous Validation** - Monitor compliance throughout development

5. **Documentation First** - Maintain clear specifications and documentation

### Constitutional Compliance

1. **Threshold Discipline** - Maintain ≥0.75 compliance threshold
2. **Article Balance** - Ensure balanced compliance across all 6 articles
3. **Violation Remediation** - Address constitutional violations immediately
4. **Continuous Improvement** - Regularly review and improve compliance

### Integration Patterns

1. **SDD Pipeline First** - Use SDD pipeline for all feature development
2. **Event-Driven** - Leverage unified orchestrator for workflow coordination
3. **Quality Gates** - Enforce quality gates at every development stage
4. **Observability** - Monitor and measure development workflows

## 🔮 Advanced Usage

### Custom Templates

Create project-specific templates in `.spec-kit/templates/`:

- Feature specification templates
- Implementation plan templates
- Task breakdown templates
- Validation checklist templates

### Constitutional Customization

Customize constitutional framework in `.spec-kit/config.yaml`:

- Adjust article weights
- Set custom thresholds
- Define project-specific rules
- Configure validation criteria

### Integration Extensions

Extend Spec Kit integration:

- Custom quality gates
- Additional validation rules
- External tool integration
- Workflow automation

---

**Remember**: Always work through a spec-driven lens. Every feature, every change, every improvement should start with constitutional specification and follow the complete Spec Kit workflow for maximum quality and constitutional compliance.

## ✅ **INSTALLATION COMPLETE**

**Spec Kit is now fully integrated and working!**

### Test the Integration

1. Press `Ctrl+S Ctrl+S` in Qoder IDE
2. Enter a feature description (e.g., "user authentication system")
3. Spec Kit will create a new branch and spec file automatically

### Daily Workflow

- `Ctrl+S Ctrl+C` - Check Spec Kit status
- `Ctrl+S Ctrl+S` - Create new feature specs
- Always start with specification before coding
- Use constitutional compliance checking throughout development

### Core Commands

```bash
# Check Spec Kit status
uvx --from git+https://github.com/github/spec-kit.git specify check

# Create new feature (via PowerShell script)
powershell .specify/scripts/powershell/create-new-feature.ps1 -Json "feature name"
```

The integration uses PowerShell scripts on Windows for optimal compatibility and includes full constitutional compliance checking with the 6-article framework.
