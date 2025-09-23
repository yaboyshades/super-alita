# Performance Monitoring and Rule Automation System Setup Guide

## Quick Start

This guide will help you set up and configure the Performance Monitoring and Rule Automation System in your development environment.

## Prerequisites

- Python 3.9 or higher
- Git
- Access to the super-alita-clean repository
- Basic understanding of constitutional development principles

## Installation

### 1. Clone and Navigate to Repository

```bash
git clone <repository-url>
cd super-alita-clean
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Install Additional Performance Monitoring Dependencies

```bash
pip install psutil safety asyncio
```

## Configuration

### 1. Basic Configuration

Create a configuration file `config/performance_monitoring.json`:

```json
{
  "metrics_retention_hours": 24,
  "collection_interval_seconds": 10,
  "telemetry_buffer_size": 10000,
  "telemetry_flush_interval": 30,
  "dashboard_update_interval": 5,
  "constitutional_threshold": 0.75,
  "performance_thresholds": {
    "response_time_ms": 1000,
    "success_rate": 0.95
  },
  "security_config": {
    "check_dependencies": true,
    "check_code": true
  }
}
```

### 2. Constitutional Framework Configuration

The system uses a six-article constitutional framework:

- **Article I: Library-First** - Prefer existing libraries over custom implementations
- **Article II: Test-First** - Comprehensive test coverage required
- **Article III: Simplicity** - Keep code simple and maintainable
- **Article IV: Integration-First** - End-to-end integration testing
- **Article V: Clarity** - Clear documentation and naming
- **Article VI: Versioning** - Proper version management and compatibility

## Quick Test

Run the demo to verify your installation:

```bash
python demo_performance_monitoring.py
```

Expected output should show:
- System initialization
- Extension interaction monitoring
- Telemetry collection
- Constitutional compliance validation
- Commit validation with quality gates
- Dashboard interface functionality
- System health checks

## IDE Integration

### VS Code Integration

1. Add to your `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "performance.monitoring.enabled": true,
  "constitutional.compliance.threshold": 0.75,
  "constitutional.compliance.autoCheck": true
}
```

2. Add to `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Constitutional Compliance Check",
      "type": "shell",
      "command": "python",
      "args": ["-c", "from src.performance_monitoring.integration import run_commit_validation; import asyncio; asyncio.run(run_commit_validation('${input:commitMessage}', ['${file}'], '${input:diffData}'))"],
      "group": "test",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Performance Monitor Status",
      "type": "shell",
      "command": "python",
      "args": ["-c", "from src.performance_monitoring.integration import PerformanceMonitoringSystem; import asyncio; async def show_status(): s = PerformanceMonitoringSystem(); await s.start_system(); print(s.get_system_status()); await s.stop_system(); asyncio.run(show_status())"],
      "group": "test"
    }
  ],
  "inputs": [
    {
      "id": "commitMessage",
      "description": "Commit message",
      "default": "feat: update functionality",
      "type": "promptString"
    },
    {
      "id": "diffData",
      "description": "Diff data (simplified)",
      "default": "diff content here",
      "type": "promptString"
    }
  ]
}
```

### Qoder IDE Integration

1. Add to `.qoder/settings.json`:

```json
{
  "performance_monitoring": {
    "enabled": true,
    "real_time_compliance": true,
    "constitutional_threshold": 0.75,
    "auto_validate_on_save": true
  },
  "quality_gates": {
    "enabled": true,
    "block_on_failure": false,
    "show_notifications": true
  }
}
```

2. Add to `.qoder/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "start-performance-monitoring",
      "type": "shell",
      "command": "python",
      "args": ["demo_performance_monitoring.py"],
      "group": "monitoring",
      "isBackground": true,
      "problemMatcher": []
    },
    {
      "label": "constitutional-compliance-check",
      "type": "shell", 
      "command": "python",
      "args": ["-m", "pytest", "tests/test_performance_monitoring_system.py", "-v"],
      "group": "test"
    },
    {
      "label": "validate-commit",
      "type": "shell",
      "command": "python",
      "args": ["-c", "import asyncio; from src.performance_monitoring.integration import run_commit_validation; asyncio.run(run_commit_validation('$(git log -1 --pretty=%B)', ['$(git diff --name-only HEAD~1)'], '$(git diff HEAD~1)'))"],
      "group": "validation"
    }
  ]
}
```

## Git Hooks Integration

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
echo "Running constitutional compliance validation..."

python -c "
import asyncio
import subprocess
import sys
from src.performance_monitoring.integration import run_commit_validation

async def validate():
    try:
        # Get staged files
        result = subprocess.run(['git', 'diff', '--cached', '--name-only'], capture_output=True, text=True)
        changed_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        # Get diff
        diff_result = subprocess.run(['git', 'diff', '--cached'], capture_output=True, text=True)
        diff_data = diff_result.stdout
        
        # Get commit message from file if available
        commit_msg = 'Pre-commit validation'
        
        # Run validation
        validation_result = await run_commit_validation(commit_msg, changed_files, diff_data)
        
        if not validation_result['overall_passed']:
            print('❌ Constitutional compliance validation FAILED')
            for gate_result in validation_result['gate_results']:
                if not gate_result['passed']:
                    print(f'   {gate_result[\"gate_name\"]}: FAILED')
                    for violation in gate_result['violations'][:3]:  # Show first 3
                        print(f'     - {violation}')
            sys.exit(1)
        else:
            print('✅ Constitutional compliance validation PASSED')
            
    except Exception as e:
        print(f'⚠️  Validation error: {e}')
        # Allow commit to proceed on validation errors
        pass

asyncio.run(validate())
"

echo "Pre-commit validation complete."
```

Make it executable:

```bash
chmod +x .git/hooks/pre-commit
```

### Commit Message Hook

Create `.git/hooks/commit-msg`:

```bash
#!/bin/bash
# Validate commit message format and constitutional compliance

COMMIT_MSG_FILE=$1
COMMIT_MSG=$(cat $COMMIT_MSG_FILE)

echo "Validating commit message..."

python -c "
import sys
import re

commit_msg = '''$COMMIT_MSG'''

# Basic format validation
pattern = r'^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .{1,50}'
if not re.match(pattern, commit_msg):
    print('❌ Commit message format invalid')
    print('Format: type(scope): description')
    print('Types: feat, fix, docs, style, refactor, test, chore')
    sys.exit(1)

# Constitutional principle check
constitutional_keywords = ['library', 'test', 'simple', 'integrate', 'clear', 'version']
if any(keyword in commit_msg.lower() for keyword in constitutional_keywords):
    print('✅ Commit message includes constitutional principles')
else:
    print('⚠️  Consider referencing constitutional principles in commit message')

print('✅ Commit message validation passed')
"
```

Make it executable:

```bash
chmod +x .git/hooks/commit-msg
```

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/constitutional-compliance.yml`:

```yaml
name: Constitutional Compliance Check

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  constitutional-compliance:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        pip install psutil safety
    
    - name: Run constitutional compliance tests
      run: |
        python -m pytest tests/test_performance_monitoring_system.py -v
    
    - name: Validate commit constitutional compliance
      if: github.event_name == 'push'
      run: |
        python -c "
        import asyncio
        import subprocess
        from src.performance_monitoring.integration import run_commit_validation
        
        async def validate_commit():
            # Get commit info
            result = subprocess.run(['git', 'show', '--name-only', '--pretty=format:%B', 'HEAD'], 
                                  capture_output=True, text=True)
            lines = result.stdout.split('\n')
            commit_msg = lines[0]
            changed_files = [line for line in lines[1:] if line.strip() and not line.startswith(' ')]
            
            diff_result = subprocess.run(['git', 'show', 'HEAD'], capture_output=True, text=True)
            diff_data = diff_result.stdout
            
            validation_result = await run_commit_validation(commit_msg, changed_files, diff_data)
            
            if not validation_result['overall_passed']:
                print('Constitutional compliance validation FAILED')
                exit(1)
            print('Constitutional compliance validation PASSED')
        
        asyncio.run(validate_commit())
        "
    
    - name: Performance monitoring system health check
      run: |
        python -c "
        import asyncio
        from src.performance_monitoring.integration import PerformanceMonitoringSystem
        
        async def health_check():
            system = PerformanceMonitoringSystem()
            await system.start_system()
            health = await system.run_health_check()
            print(f'System health: {health[\"overall_health\"]}')
            if health['overall_health'] != 'healthy':
                for issue in health['issues']:
                    print(f'Issue: {issue}')
            await system.stop_system()
            if health['overall_health'] != 'healthy':
                exit(1)
        
        asyncio.run(health_check())
        "
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
ModuleNotFoundError: No module named 'src.performance_monitoring'
```

**Solution**: Ensure you're running from the project root and Python path includes the project:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python demo_performance_monitoring.py
```

#### 2. Permission Errors

```bash
PermissionError: [Errno 13] Permission denied
```

**Solution**: Check file permissions and ensure Git hooks are executable:

```bash
chmod +x .git/hooks/pre-commit
chmod +x .git/hooks/commit-msg
```

#### 3. Constitutional Validation Failures

```bash
Constitutional validation failed (score: 0.65 < 0.75)
```

**Solution**: Review the violation details and address them:
- Add missing tests for new code
- Add documentation to public functions
- Simplify complex functions
- Use existing libraries instead of custom implementations

#### 4. Performance Threshold Violations

```bash
Performance gate FAILED: Response time exceeds threshold
```

**Solution**: Optimize performance or adjust thresholds in config:

```json
{
  "performance_thresholds": {
    "response_time_ms": 2000,
    "success_rate": 0.90
  }
}
```

### Getting Help

1. **Check system health**: Run `python demo_performance_monitoring.py` to verify system status
2. **Run tests**: Execute `python -m pytest tests/test_performance_monitoring_system.py -v`
3. **Check logs**: Review application logs for detailed error information
4. **Constitutional compliance**: Review the six-article framework and ensure code follows principles

## Advanced Configuration

### Custom Constitutional Rules

You can extend the constitutional validation by adding custom rules:

```python
from src.performance_monitoring.core.constitutional_engine import ConstitutionalEngine

# Add custom validation rule
async def custom_validation_rule(data, context):
    # Custom validation logic here
    if some_condition:
        return ComplianceViolation(
            article=ConstitutionalArticle.ARTICLE_III_SIMPLICITY,
            severity="medium",
            description="Custom rule violation",
            suggestion="Fix suggestion"
        )
    return None

# Add to engine
engine = ConstitutionalEngine()
engine.validation_rules[ConstitutionalArticle.ARTICLE_III_SIMPLICITY].append(custom_validation_rule)
```

### Performance Monitoring Customization

```python
from src.performance_monitoring.integration import PerformanceMonitoringSystem

# Custom configuration
config = {
    "metrics_retention_hours": 48,  # Longer retention
    "collection_interval_seconds": 5,  # More frequent collection
    "constitutional_threshold": 0.80,  # Higher compliance threshold
}

system = PerformanceMonitoringSystem(config=config)
```

## Next Steps

1. **Integrate with your development workflow**: Set up IDE integration and Git hooks
2. **Customize thresholds**: Adjust constitutional and performance thresholds for your project
3. **Add custom rules**: Extend the constitutional validation with project-specific rules
4. **Monitor continuously**: Set up continuous monitoring in your CI/CD pipeline
5. **Train your team**: Ensure all developers understand the constitutional principles

---

*This setup guide ensures constitutional compliance through automated monitoring and validation while maintaining high performance standards.*