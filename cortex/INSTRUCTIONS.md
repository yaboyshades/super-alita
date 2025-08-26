# Cortex Component Instructions

This directory contains the Cortex automation and workflow management components for Super Alita.

## Overview

Cortex provides advanced automation capabilities including:
- **Workflow Automation**: Complex multi-step process orchestration
- **Git Integration**: Automated version control operations
- **Task Management**: Intelligent task scheduling and execution
- **Resource Management**: System resource optimization and monitoring
- **Integration Pipelines**: External service integration workflows

## Architecture

The Cortex system is organized into specialized modules:

### Core Components
- **Automation Engine**: Central workflow orchestration
- **Task Scheduler**: Intelligent task queuing and execution
- **Resource Monitor**: System resource tracking and optimization
- **Integration Hub**: External service coordination

### Automation Modules
- **Git Workflows** (`automation/git_workflow.py`): Version control automation
- **File Operations**: Automated file and directory management
- **System Tasks**: System-level automation and maintenance
- **Deployment**: Automated deployment and configuration management

## Development Guidelines

### Module Structure
Follow the established patterns in Cortex components:

```python
from typing import Any, Dict
import subprocess

class AutomationComponent:
    """Base class for Cortex automation components."""
    
    def execute_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an automation task."""
        return {
            "success": True,
            "result": "Task completed",
            "output": "Detailed output"
        }
```

### Git Automation Example
```python
from cortex.automation.git_workflow import GitAutomation

git_auto = GitAutomation()

# Create feature branch
result = git_auto.create_feature_branch("new-feature")
if result["success"]:
    print(f"Created branch: {result['branch']}")

# Auto-commit changes
result = git_auto.auto_commit("Updated documentation", ["README.md"])
```

### Error Handling
All Cortex components should return structured results:

```python
{
    "success": bool,           # Operation success status
    "result": str,            # Primary result data
    "output": str,            # Detailed output/logs
    "error": str,             # Error message if failed
    "metadata": dict          # Additional context
}
```

## Configuration

### Environment Setup
Cortex components may require specific environment configuration:

```bash
# Set Cortex-specific environment variables
export CORTEX_WORKSPACE_PATH=/path/to/workspace
export CORTEX_LOG_LEVEL=INFO
export CORTEX_MAX_CONCURRENT_TASKS=5
```

### Integration Configuration
Configure external service integrations:

```python
# Example configuration for external services
CORTEX_CONFIG = {
    "git": {
        "auto_commit": True,
        "branch_prefix": "feature/",
        "commit_template": "feat: {message}"
    },
    "automation": {
        "timeout_seconds": 300,
        "retry_attempts": 3,
        "parallel_execution": True
    }
}
```

## Usage Examples

### Workflow Automation
```python
from cortex.workflow import WorkflowEngine

# Define a workflow
workflow = WorkflowEngine()
workflow.add_step("setup", setup_environment)
workflow.add_step("process", process_data)
workflow.add_step("cleanup", cleanup_resources)

# Execute workflow
result = await workflow.execute()
```

### Task Scheduling
```python
from cortex.scheduler import TaskScheduler

scheduler = TaskScheduler()

# Schedule a recurring task
scheduler.schedule_task(
    name="backup_data",
    function=backup_function,
    schedule="0 2 * * *",  # Daily at 2 AM
    args={"backup_path": "/data/backups"}
)
```

### Resource Monitoring
```python
from cortex.monitoring import ResourceMonitor

monitor = ResourceMonitor()

# Get system metrics
metrics = monitor.get_system_metrics()
print(f"CPU Usage: {metrics['cpu_percent']}%")
print(f"Memory Usage: {metrics['memory_percent']}%")

# Set up alerts
monitor.set_alert_threshold("cpu_percent", 80)
monitor.set_alert_threshold("memory_percent", 85)
```

## Integration with Super Alita

### Event System Integration
Cortex components integrate with the Super Alita event system:

```python
from src.core.events import create_event

# Emit automation events
automation_event = create_event(
    "cortex_automation",
    task_type="git_workflow",
    status="completed",
    result=automation_result
)
```

### Plugin Integration
Cortex components can be exposed as Super Alita plugins:

```python
from src.core.plugin_interface import PluginInterface
from cortex.automation import AutomationEngine

class CortexPlugin(PluginInterface):
    def __init__(self):
        self.automation = AutomationEngine()
    
    @property
    def name(self) -> str:
        return "cortex_automation"
    
    async def shutdown(self) -> None:
        await self.automation.shutdown()
```

## Testing

### Unit Testing
Test individual Cortex components:

```bash
# Run Cortex-specific tests
pytest cortex/tests/ -v

# Test specific modules
pytest cortex/tests/test_automation.py -v
```

### Integration Testing
Test Cortex integration with Super Alita:

```bash
# Run integration tests
pytest tests/integration/test_cortex_integration.py -v
```

### Automation Testing
Test automation workflows:

```python
@pytest.mark.asyncio
async def test_git_automation():
    git_auto = GitAutomation()
    
    # Test branch creation
    result = git_auto.create_feature_branch("test-feature")
    assert result["success"] is True
    assert "test-feature" in result["branch"]
```

## Development Workflow

### Adding New Automation
1. Create automation module in appropriate subdirectory
2. Implement standard result format
3. Add configuration options
4. Write comprehensive tests
5. Update documentation
6. Integrate with event system if needed

### Code Quality
- Follow the same code standards as the main project
- Use type hints for all public interfaces
- Include comprehensive docstrings
- Handle errors gracefully with structured responses

## Performance Considerations

### Resource Management
- Monitor system resource usage during automation
- Implement timeouts for long-running operations
- Use async/await for I/O-bound operations
- Consider parallel execution for independent tasks

### Scalability
- Design for horizontal scaling when possible
- Implement proper connection pooling for external services
- Use caching for frequently accessed data
- Monitor and optimize memory usage

## Troubleshooting

### Common Issues
- **Timeout Errors**: Increase timeout values in configuration
- **Permission Issues**: Verify file and system permissions
- **Resource Exhaustion**: Monitor system resources and adjust limits
- **Integration Failures**: Check external service connectivity and credentials

### Debugging
Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger('cortex').setLevel(logging.DEBUG)
```

### Log Analysis
Check Cortex logs for automation issues:

```bash
# View recent automation logs
tail -f logs/cortex_automation.log

# Search for specific errors
grep "ERROR" logs/cortex_automation.log
```

For advanced Cortex development and troubleshooting, refer to the component-specific documentation in each subdirectory.