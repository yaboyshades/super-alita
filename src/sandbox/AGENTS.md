# Sandboxing & Secure Execution - Agent Instructions

## Overview
The `src/sandbox/` directory provides secure execution environments for dynamic code:
- **Execution Sandbox** - Safe Python code execution with resource limits
- **Security Registry** - Allowlist/blocklist management for safe operations
- **Isolation Controls** - Process and namespace isolation

## Key Files

### Core Components
- `exec_sandbox.py` - Main sandboxed execution engine
- `registry.py` - Security policy and allowlist management

## Security Model

### Execution Principles
1. **Zero Trust** - All dynamic code is untrusted by default
2. **Principle of Least Privilege** - Minimal permissions granted
3. **Resource Isolation** - CPU, memory, and file system limits
4. **Network Isolation** - Controlled network access
5. **Time Limits** - Execution timeout enforcement

### Threat Model
- **Malicious Code Injection** - Prevent arbitrary code execution
- **Resource Exhaustion** - Limit CPU/memory/disk usage
- **File System Access** - Restrict to designated directories
- **Network Access** - Control external connections
- **Process Escape** - Prevent sandbox breakouts

## Sandbox Usage Guidelines

### Safe Code Execution
```python
from src.sandbox.exec_sandbox import execute_sandboxed_code

# Execute user-provided code safely
result = await execute_sandboxed_code(
    code=user_code,
    timeout=30,  # 30 second limit
    memory_limit_mb=256,  # 256MB RAM limit
    allowed_imports=['math', 'json'],  # Limited imports
    globals_whitelist=['input_data'],  # Available variables
    capture_output=True
)

if result.success:
    print(f"Output: {result.output}")
    print(f"Return value: {result.return_value}")
else:
    print(f"Error: {result.error}")
    print(f"Error type: {result.error_type}")
```

### Registry-Based Security
```python
from src.sandbox.registry import SecurityRegistry

# Check if operation is allowed
registry = SecurityRegistry()

if registry.is_allowed_function('os.listdir'):
    # Safe to proceed
    result = os.listdir(path)
else:
    # Block potentially dangerous operation
    raise SecurityError("Function not allowed")

# Add new allowed function
registry.allow_function('custom.safe_function')
```

## Execution Environment

### Resource Limits
```python
# Default sandbox limits
DEFAULT_LIMITS = {
    'cpu_time': 30,      # 30 seconds CPU time
    'wall_time': 60,     # 60 seconds wall clock time
    'memory': 256 * MB,  # 256MB RAM
    'file_size': 10 * MB, # 10MB max file size
    'file_count': 100,   # Maximum 100 files
    'process_count': 1   # Single process only
}
```

### Allowed Operations
```python
# Safe built-in functions (allowlist)
SAFE_BUILTINS = {
    'abs', 'all', 'any', 'bool', 'dict', 'enumerate',
    'filter', 'float', 'int', 'len', 'list', 'map',
    'max', 'min', 'range', 'reversed', 'set', 'sorted',
    'str', 'sum', 'tuple', 'type', 'zip'
}

# Safe standard library modules
SAFE_MODULES = {
    'math', 'json', 'datetime', 'uuid', 'hashlib',
    're', 'base64', 'urllib.parse'
}

# Blocked operations (examples)
BLOCKED_OPERATIONS = {
    'open',           # File system access
    'exec', 'eval',   # Dynamic execution
    'import',         # Import restrictions
    '__import__',     # Import restrictions
    'compile',        # Code compilation
    'globals',        # Global access
    'locals',         # Local access
    'vars',           # Variable access
}
```

## Development Guidelines

### Adding New Safe Operations
```python
# To add a new safe function:
def register_safe_function(func_name: str, module: str = None):
    """Register a function as safe for sandbox execution"""
    registry = SecurityRegistry()
    
    # Validate function safety
    if not validate_function_safety(func_name, module):
        raise SecurityError(f"Function {func_name} is not safe")
    
    # Add to allowlist
    registry.allow_function(func_name, module)
    
    # Log for audit
    logger.info(f"Registered safe function: {func_name}")
```

### Custom Sandbox Policies
```python
class CustomSandboxPolicy:
    """Custom execution policy for specific use cases"""
    
    def __init__(self):
        self.allowed_modules = set()
        self.blocked_functions = set()
        self.resource_limits = {}
    
    def allow_module(self, module_name: str) -> None:
        """Allow specific module for execution"""
        if self.validate_module_safety(module_name):
            self.allowed_modules.add(module_name)
    
    def block_function(self, func_name: str) -> None:
        """Block specific function"""
        self.blocked_functions.add(func_name)
    
    def set_limit(self, resource: str, limit: int) -> None:
        """Set resource limit"""
        self.resource_limits[resource] = limit
```

## Testing Guidelines

### Sandbox Security Testing
```python
import pytest
from src.sandbox.exec_sandbox import execute_sandboxed_code

@pytest.mark.asyncio
async def test_sandbox_blocks_dangerous_code():
    """Test that dangerous operations are blocked"""
    
    dangerous_codes = [
        "import os; os.system('rm -rf /')",  # System calls
        "open('/etc/passwd', 'r')",          # File access
        "eval('malicious_code')",            # Dynamic execution
        "__import__('subprocess')",          # Import bypass
        "while True: pass",                  # Infinite loop
    ]
    
    for code in dangerous_codes:
        result = await execute_sandboxed_code(code, timeout=5)
        assert not result.success, f"Dangerous code was allowed: {code}"
        assert "SecurityError" in result.error_type

@pytest.mark.asyncio
async def test_sandbox_allows_safe_code():
    """Test that safe operations work correctly"""
    
    safe_code = """
import math
result = math.sqrt(16)
print(f"Square root of 16 is {result}")
"""
    
    result = await execute_sandboxed_code(safe_code)
    assert result.success
    assert "4.0" in result.output

@pytest.mark.asyncio 
async def test_resource_limits():
    """Test that resource limits are enforced"""
    
    # Memory exhaustion test
    memory_bomb = "data = 'x' * (1024 * 1024 * 1024)"  # 1GB string
    result = await execute_sandboxed_code(
        memory_bomb, 
        memory_limit_mb=100
    )
    assert not result.success
    assert "MemoryError" in result.error_type
    
    # Timeout test
    infinite_loop = "while True: pass"
    result = await execute_sandboxed_code(infinite_loop, timeout=1)
    assert not result.success
    assert "TimeoutError" in result.error_type
```

### Registry Testing
```python
def test_security_registry():
    """Test security registry functionality"""
    registry = SecurityRegistry()
    
    # Test allowlist
    registry.allow_function('math.sqrt')
    assert registry.is_allowed_function('math.sqrt')
    
    # Test blocklist
    registry.block_function('os.system')
    assert not registry.is_allowed_function('os.system')
    
    # Test module restrictions
    registry.allow_module('json')
    assert registry.is_allowed_module('json')
    assert not registry.is_allowed_module('subprocess')
```

## Security Best Practices

### Code Review Guidelines
1. **Never bypass sandbox** - All dynamic code must go through sandbox
2. **Validate inputs** - Sanitize all user-provided code
3. **Audit new allowlist entries** - Require security review
4. **Test attack vectors** - Include security tests in CI
5. **Monitor execution** - Log all sandbox usage

### Common Security Anti-Patterns
```python
# ❌ NEVER do this - bypasses sandbox
exec(user_code)
eval(user_expression)

# ❌ NEVER do this - allows arbitrary imports
__import__(user_module)

# ❌ NEVER do this - unrestricted file access
with open(user_filename, 'w') as f:
    f.write(user_data)

# ✅ DO this instead - use sandbox
result = await execute_sandboxed_code(
    user_code,
    allowed_imports=['math'],
    timeout=30
)
```

### Production Security
```python
# Production sandbox configuration
PRODUCTION_CONFIG = {
    'timeout': 10,        # Shorter timeouts
    'memory_limit_mb': 64, # Lower memory limits
    'network_enabled': False,  # No network access
    'file_system_readonly': True,  # Read-only file system
    'audit_logging': True,  # Full audit trail
    'allow_imports': [],   # No imports by default
}
```

## Monitoring & Observability

### Execution Metrics
```python
# Track sandbox usage
from src.core.metrics import MetricsCollector

# Execution time
with MetricsCollector.timer("sandbox.execution_time"):
    result = await execute_sandboxed_code(code)

# Success/failure rates
if result.success:
    MetricsCollector.increment("sandbox.execution.success")
else:
    MetricsCollector.increment("sandbox.execution.failure")
    MetricsCollector.increment(f"sandbox.error.{result.error_type}")

# Resource usage
MetricsCollector.gauge("sandbox.memory_usage", result.memory_used)
MetricsCollector.gauge("sandbox.cpu_time", result.cpu_time)
```

### Security Alerts
```python
# Alert on security violations
def alert_security_violation(code: str, violation_type: str):
    """Alert on detected security violations"""
    alert_data = {
        'severity': 'HIGH',
        'type': 'sandbox_security_violation',
        'violation_type': violation_type,
        'code_hash': hashlib.sha256(code.encode()).hexdigest()[:16],
        'timestamp': datetime.now(timezone.utc)
    }
    
    # Send to security monitoring
    send_security_alert(alert_data)
    
    # Log for audit
    logger.warning(f"Security violation detected: {violation_type}")
```

## Common Patterns

### Safe Tool Execution
```python
async def execute_tool_safely(tool_code: str, inputs: Dict) -> Dict:
    """Safely execute tool code with inputs"""
    
    # Prepare execution environment
    execution_code = f"""
# Tool inputs available as variables
{format_inputs_as_variables(inputs)}

# Tool code
{tool_code}

# Return result
result
"""
    
    # Execute with appropriate limits
    result = await execute_sandboxed_code(
        execution_code,
        timeout=30,
        memory_limit_mb=128,
        allowed_imports=['math', 'json', 'datetime']
    )
    
    if result.success:
        return {
            'success': True,
            'result': result.return_value,
            'output': result.output
        }
    else:
        return {
            'success': False,
            'error': result.error,
            'error_type': result.error_type
        }
```

### Progressive Permissions
```python
class ProgressivePermissions:
    """Grant permissions based on trust level"""
    
    def __init__(self):
        self.trust_levels = {
            'untrusted': {
                'timeout': 5,
                'memory_mb': 32,
                'allowed_modules': []
            },
            'basic': {
                'timeout': 15,
                'memory_mb': 64,
                'allowed_modules': ['math', 'json']
            },
            'trusted': {
                'timeout': 30,
                'memory_mb': 128,
                'allowed_modules': ['math', 'json', 'datetime', 'uuid']
            }
        }
    
    def get_limits(self, trust_level: str) -> Dict:
        """Get execution limits for trust level"""
        return self.trust_levels.get(trust_level, self.trust_levels['untrusted'])
```

## Performance Considerations
- **Sandbox overhead** - Each execution has initialization cost
- **Resource cleanup** - Properly clean up after execution
- **Connection pooling** - Reuse execution environments when safe
- **Caching** - Cache safe code analysis results
- **Parallel execution** - Multiple sandboxes can run concurrently