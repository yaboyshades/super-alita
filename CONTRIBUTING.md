# Super Alita Contributing Guide

Welcome to the Super Alita project! This guide will help you get started with contributing to our constitutional AI framework.

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (optional, for monitoring stack)

### Setup Development Environment

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-org/super-alita-clean.git
   cd super-alita-clean
   ```

2. **Install Dependencies**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Verify Setup**
   ```bash
   # Run constitutional validation
   python scripts/rule_validator.py src/
   
   # Run tests
   pytest tests/
   
   # Check code quality
   black --check src/
   flake8 src/
   ```

## 📋 Contributor Workflow

### Branching Strategy

We use GitHub Flow with constitutional compliance validation:

- **Main Branch**: `main` - Production-ready code
- **Feature Branches**: `feature/<ticket-id>-short-description`
- **Bugfix Branches**: `bugfix/<ticket-id>-description`
- **Documentation**: `docs/<description>`

### Development Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/123-add-new-validator
   ```

2. **Make Changes**
   - Write code following our [Constitutional Framework](#constitutional-framework)
   - Add tests for new functionality
   - Update documentation as needed

3. **Pre-commit Validation**
   ```bash
   # Automatic validation on commit
   git add .
   git commit -m "feat: add new constitutional validator"
   
   # Manual validation
   python scripts/rule_validator.py --format human src/
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/123-add-new-validator
   ```
   
   Create a Pull Request with:
   - Clear description of changes
   - Link to related issue
   - Screenshots/demos if applicable

### Pull Request Requirements

✅ **Must Pass Before Merge:**
- All tests pass
- Constitutional compliance validation passes
- Code review approved by maintainer
- Documentation updated (if applicable)
- No merge conflicts

❌ **Blocking Violations:**
- Missing unit tests for public functions
- Functions exceeding complexity thresholds
- Breaking changes without version bump
- Security vulnerabilities

## 🏛️ Constitutional Framework

Our project follows a six-article constitutional framework that ensures code quality and maintainability:

### Article I: Library-First
- **Principle**: Favor established libraries over custom implementations
- **Example**: Use `requests` instead of custom HTTP clients
- **Validation**: Automatic detection of reinvented wheel patterns

### Article II: Test-First  
- **Principle**: All public functions must have unit tests
- **Example**: Every function in `src/` needs corresponding test in `tests/`
- **Validation**: Test coverage analysis and enforcement

### Article III: Simplicity
- **Principle**: Keep functions simple and focused
- **Limits**: 
  - Max 50 lines per function
  - Max 5 parameters per function
  - Cyclomatic complexity < 10
- **Validation**: Static analysis of function complexity

### Article IV: Integration-First
- **Principle**: End-to-end integration tests required
- **Example**: API endpoints need integration test coverage
- **Validation**: Integration test presence verification

### Article V: Clarity
- **Principle**: Clear documentation and naming
- **Requirements**:
  - Public functions have docstrings
  - Descriptive variable/function names
  - No TODO/FIXME in production code
- **Validation**: Documentation coverage analysis

### Article VI: Versioning
- **Principle**: Proper semantic versioning for breaking changes
- **Requirements**:
  - Version bump for breaking changes
  - CHANGELOG.md updates
  - Migration guides for major versions
- **Validation**: Breaking change detection

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── fixtures/       # Test data
└── conftest.py     # Pytest configuration
```

### Writing Tests

1. **Unit Tests**
   ```python
   # tests/unit/test_validator.py
   import pytest
   from src.validator import ConstitutionalValidator
   
   def test_validator_validates_function():
       validator = ConstitutionalValidator()
       result = validator.validate_function("def test(): pass")
       assert result.is_valid
   ```

2. **Integration Tests**
   ```python
   # tests/integration/test_workflow_integration.py
   @pytest.mark.integration
   def test_complete_validation_workflow():
       # Test entire validation pipeline
       pass
   ```

3. **Running Tests**
   ```bash
   # All tests
   pytest
   
   # Unit tests only
   pytest tests/unit/
   
   # Integration tests
   pytest tests/integration/ -m integration
   
   # With coverage
   pytest --cov=src/ --cov-report=html
   ```

## 📊 Performance Monitoring

### Telemetry Integration

Our system includes comprehensive performance monitoring:

```python
from src.performance_monitoring.middleware import track_extension_call

@track_extension_call("my_extension", "validation")
async def validate_code(code):
    # Your validation logic
    pass
```

### Monitoring Stack

Start the monitoring stack for development:

```bash
cd monitoring/
docker-compose up -d

# Access dashboards
open http://localhost:3000  # Grafana (admin/admin123)
open http://localhost:9090  # Prometheus
open http://localhost:9093  # AlertManager
```

### Service Level Objectives (SLOs)

Our performance targets:
- **Latency**: p95 < 1000ms
- **Error Rate**: < 2%
- **Availability**: > 99.9%
- **CPU Usage**: < 80% sustained
- **Memory Usage**: < 70% sustained

## 🔧 Development Tools

### Code Quality Tools

```bash
# Formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/
pylint src/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

### Constitutional Validation

```bash
# Validate specific directory
python scripts/rule_validator.py src/

# JSON output for CI
python scripts/rule_validator.py --format json src/

# Verbose output for debugging
python scripts/rule_validator.py --verbose src/
```

## 🐛 Debugging and Troubleshooting

### Common Issues

1. **Pre-commit Hook Failures**
   ```bash
   # Skip hooks temporarily (not recommended)
   git commit --no-verify
   
   # Fix issues and retry
   pre-commit run --all-files
   ```

2. **Constitutional Validation Failures**
   ```bash
   # See detailed violations
   python scripts/rule_validator.py --verbose src/
   
   # Check specific rule
   python scripts/rule_validator.py --rules rules/constitution/ src/specific_file.py
   ```

3. **Test Failures**
   ```bash
   # Run specific test
   pytest tests/test_specific.py::test_function -v
   
   # Debug with pdb
   pytest --pdb tests/test_specific.py
   ```

### Getting Help

- 📖 **Documentation**: Check the [docs/](docs/) directory
- 🐛 **Issues**: Create an issue on GitHub
- 💬 **Discussions**: Use GitHub Discussions
- 📧 **Maintainers**: Contact the core team

## 📝 Documentation Standards

### Code Documentation

```python
def validate_constitutional_compliance(code: str) -> ValidationResult:
    """
    Validate code against constitutional framework.
    
    Args:
        code: Source code to validate
        
    Returns:
        ValidationResult containing compliance status and violations
        
    Raises:
        ValidationError: If code cannot be parsed
        
    Example:
        >>> validator = ConstitutionalValidator()
        >>> result = validator.validate("def hello(): pass")
        >>> assert result.is_compliant
    """
```

### Documentation Updates

- Update docstrings for API changes
- Add examples for new features
- Update README.md for user-facing changes
- Create migration guides for breaking changes

## 🎯 Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(validator): add constitutional rule validation
fix(telemetry): resolve memory leak in collector
docs(api): update authentication guide
test(integration): add end-to-end workflow tests
```

## 🏅 Recognition

Contributors who make significant improvements to the constitutional framework will be recognized in:
- CONTRIBUTORS.md
- Release notes
- Project documentation

Thank you for contributing to Super Alita! 🚀