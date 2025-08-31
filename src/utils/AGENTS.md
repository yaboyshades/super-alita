# Utility Functions - Agent Instructions

## Overview
The `src/utils/` directory contains reusable utility functions and components:
- **Event Builders** - Helper functions for creating standardized events
- **Guardrails** - Security and safety validation utilities
- **Schema Validation** - Data validation and schema enforcement
- **Telemetry** - System monitoring and metrics collection
- **Terminal Animation** - CLI/terminal interface utilities

## Key Files & Responsibilities

### Core Utilities
- `event_builders.py` - Standardized event creation helpers
- `guardrails.py` - Security validation and safety checks
- `schema_validator.py` - Data schema validation utilities
- `telemetry.py` - System telemetry and monitoring
- `terminal_animation.py` - Terminal UI and animation utilities
- `__init__.py` - Module initialization and exports

## Development Guidelines

### Event Builder Patterns
```python
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from src.core.events import create_event

def create_tool_execution_event(
    tool_name: str,
    parameters: Dict[str, Any],
    session_id: str,
    correlation_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized tool execution event"""
    return create_event(
        "tool_execution_requested",
        tool_name=tool_name,
        parameters=parameters,
        session_id=session_id,
        correlation_id=correlation_id or f"tool_{tool_name}_{int(time.time())}",
        timestamp=datetime.now(timezone.utc),
        source="event_builder"
    )

def create_error_event(
    error_type: str,
    error_message: str,
    source_component: str,
    additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create standardized error event"""
    return create_event(
        "system_error",
        error_type=error_type,
        error_message=error_message,
        source_component=source_component,
        context=additional_context or {},
        timestamp=datetime.now(timezone.utc),
        severity="error"
    )
```

### Guardrails Implementation
```python
import hashlib
import os
import re
from pathlib import Path
from typing import List, Set, Dict, Any

class SecurityGuardrails:
    """Security validation and safety checks"""

    def __init__(self):
        self.allowed_file_extensions = {'.py', '.md', '.txt', '.json', '.yaml', '.yml'}
        self.blocked_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'shell=True'
        ]

    def validate_file_access(self, file_path: str, workspace_root: str) -> bool:
        """Validate file access is within workspace boundaries"""
        try:
            abs_path = Path(file_path).resolve()
            workspace_path = Path(workspace_root).resolve()

            # Check if path is within workspace
            return abs_path.is_relative_to(workspace_path)
        except (ValueError, OSError):
            return False

    def scan_code_for_dangers(self, code: str) -> List[str]:
        """Scan code for dangerous patterns"""
        violations = []

        for pattern in self.blocked_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"Dangerous pattern detected: {pattern}")

        return violations

    def validate_input_size(self, data: Any, max_size_mb: int = 10) -> bool:
        """Validate input data size"""
        try:
            data_str = str(data) if not isinstance(data, str) else data
            size_bytes = len(data_str.encode('utf-8'))
            max_size_bytes = max_size_mb * 1024 * 1024
            return size_bytes <= max_size_bytes
        except Exception:
            return False

def hash_top_files(directory: str, max_files: int = 100) -> str:
    """Generate hash of top files for integrity checking"""
    file_hashes = []

    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return ""

        # Get top files by modification time
        files = sorted(
            [f for f in dir_path.rglob('*') if f.is_file()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:max_files]

        for file_path in files:
            try:
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    file_hashes.append(f"{file_path.name}:{file_hash}")
            except (IOError, OSError):
                continue

        # Generate combined hash
        combined = "|".join(sorted(file_hashes))
        return hashlib.sha256(combined.encode()).hexdigest()

    except Exception:
        return ""

def repo_download_integrity(repo_path: str, expected_files: List[str]) -> bool:
    """Verify repository download integrity"""
    try:
        repo_dir = Path(repo_path)
        if not repo_dir.exists():
            return False

        # Check required files exist
        for file_name in expected_files:
            file_path = repo_dir / file_name
            if not file_path.exists():
                return False

        return True
    except Exception:
        return False
```

### Schema Validation
```python
from pydantic import BaseModel, validator
from typing import Any, Dict, List, Optional, Union

class EventSchema(BaseModel):
    """Base schema for all events"""
    event_type: str
    timestamp: datetime
    source: str
    correlation_id: Optional[str] = None
    data: Dict[str, Any] = {}

    @validator('event_type')
    def validate_event_type(cls, v):
        allowed_types = [
            'tool_call', 'tool_result', 'user_input', 'agent_reply',
            'system_error', 'plugin_loaded', 'session_started'
        ]
        if v not in allowed_types:
            raise ValueError(f"Invalid event type: {v}")
        return v

class ConfigSchema(BaseModel):
    """Schema for configuration validation"""
    plugin_name: str
    enabled: bool = True
    config: Dict[str, Any] = {}
    dependencies: List[str] = []

    @validator('plugin_name')
    def validate_plugin_name(cls, v):
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', v):
            raise ValueError("Invalid plugin name format")
        return v

def validate_json_schema(data: Dict[str, Any], schema_class: BaseModel) -> bool:
    """Validate data against Pydantic schema"""
    try:
        schema_class(**data)
        return True
    except Exception:
        return False
```

### Telemetry Collection
```python
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class TelemetryEvent:
    """Telemetry event data structure"""
    event_name: str
    timestamp: float
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

class ReadLedger:
    """Telemetry collection and reporting"""

    def __init__(self):
        self.events: List[TelemetryEvent] = []
        self.active_timers: Dict[str, float] = {}

    def log_event(self, event_name: str, metadata: Dict[str, Any] = None, tags: Dict[str, str] = None):
        """Log a telemetry event"""
        event = TelemetryEvent(
            event_name=event_name,
            timestamp=time.time(),
            metadata=metadata or {},
            tags=tags or {}
        )
        self.events.append(event)

    @contextmanager
    def timer(self, operation_name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.log_event(
                f"{operation_name}_duration",
                metadata={"duration_seconds": duration}
            )

    def log_ability_execution(self, ability_name: str, result: Dict[str, Any]):
        """Log ability execution telemetry"""
        self.log_event(
            "ability_executed",
            metadata={
                "ability_name": ability_name,
                "success": result.get("success", False),
                "execution_time": result.get("execution_time", 0)
            },
            tags={"component": "abilities"}
        )

    async def commit(self):
        """Commit telemetry data (placeholder for actual implementation)"""
        # In real implementation, this would send data to telemetry service
        logger.info(f"Committed {len(self.events)} telemetry events")
        self.events.clear()

def start_timer(operation_name: str) -> str:
    """Start a named timer"""
    timer_id = f"{operation_name}_{int(time.time() * 1000)}"
    # Store timer start time in global registry
    return timer_id

def end_timer(timer_id: str) -> float:
    """End a named timer and return duration"""
    # Calculate duration from global registry
    # Return elapsed time
    return 0.0  # Placeholder

def telemetry_footer() -> str:
    """Generate telemetry footer for reports"""
    return f"Generated at {datetime.now(timezone.utc).isoformat()}"
```

## Testing Guidelines

### Utility Testing Patterns
```python
import pytest
from unittest.mock import Mock, patch
from src.utils.guardrails import SecurityGuardrails, hash_top_files

def test_security_guardrails():
    """Test security validation"""
    guardrails = SecurityGuardrails()

    # Test safe code
    safe_code = "def hello(): return 'world'"
    violations = guardrails.scan_code_for_dangers(safe_code)
    assert len(violations) == 0

    # Test dangerous code
    dangerous_code = "eval('malicious code')"
    violations = guardrails.scan_code_for_dangers(dangerous_code)
    assert len(violations) > 0

def test_file_access_validation():
    """Test file access validation"""
    guardrails = SecurityGuardrails()

    # Test valid access
    assert guardrails.validate_file_access("/workspace/file.txt", "/workspace")

    # Test invalid access (path traversal)
    assert not guardrails.validate_file_access("/workspace/../etc/passwd", "/workspace")

@pytest.mark.asyncio
async def test_telemetry_collection():
    """Test telemetry event collection"""
    ledger = ReadLedger()

    # Log test event
    ledger.log_event("test_event", {"key": "value"})
    assert len(ledger.events) == 1

    # Test timer context manager
    with ledger.timer("test_operation"):
        await asyncio.sleep(0.01)  # Small delay

    duration_events = [e for e in ledger.events if "duration" in e.event_name]
    assert len(duration_events) == 1
```

### Performance Testing
```python
import time
import pytest
from src.utils.guardrails import hash_top_files

def test_hash_performance():
    """Test file hashing performance"""
    start_time = time.time()
    hash_result = hash_top_files("./src", max_files=50)
    duration = time.time() - start_time

    assert hash_result != ""
    assert duration < 1.0  # Should complete within 1 second

@pytest.mark.benchmark
def test_schema_validation_performance(benchmark):
    """Benchmark schema validation performance"""
    from src.utils.schema_validator import EventSchema

    test_data = {
        "event_type": "tool_call",
        "timestamp": datetime.now(),
        "source": "test",
        "data": {"key": "value"}
    }

    result = benchmark(validate_json_schema, test_data, EventSchema)
    assert result is True
```

## Performance Guidelines

### Efficient Utility Functions
```python
import functools
from typing import Any, Callable

def memoize_with_ttl(ttl_seconds: int = 300):
    """Memoization decorator with TTL"""
    def decorator(func: Callable) -> Callable:
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()

            if key in cache:
                value, timestamp = cache[key]
                if current_time - timestamp < ttl_seconds:
                    return value

            result = func(*args, **kwargs)
            cache[key] = (result, current_time)

            # Clean up old entries
            expired_keys = [
                k for k, (_, ts) in cache.items()
                if current_time - ts >= ttl_seconds
            ]
            for k in expired_keys:
                del cache[k]

            return result
        return wrapper
    return decorator

@memoize_with_ttl(ttl_seconds=60)
def expensive_validation(data: str) -> bool:
    """Example of expensive validation with caching"""
    # Simulate expensive operation
    time.sleep(0.1)
    return len(data) > 0
```

### Batch Processing
```python
from typing import List, Iterator, TypeVar

T = TypeVar('T')

def batch_process(items: List[T], batch_size: int = 100) -> Iterator[List[T]]:
    """Process items in batches for memory efficiency"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

async def process_telemetry_batch(events: List[TelemetryEvent], batch_size: int = 50):
    """Process telemetry events in batches"""
    for batch in batch_process(events, batch_size):
        # Process batch
        await asyncio.gather(*[process_single_event(event) for event in batch])
```

## Common Patterns

### Retry with Exponential Backoff
```python
import random
from typing import Callable, Any

async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
) -> Any:
    """Retry function with exponential backoff"""
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e

            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                if jitter:
                    delay *= (0.5 + random.random())  # Add jitter

                await asyncio.sleep(delay)
            else:
                raise last_exception
```

### Configuration Management
```python
import os
from typing import Dict, Any, Optional

class ConfigManager:
    """Centralized configuration management"""

    def __init__(self):
        self._config_cache: Dict[str, Any] = {}

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with caching"""
        if key in self._config_cache:
            return self._config_cache[key]

        # Try environment variable first
        env_value = os.getenv(key.upper())
        if env_value is not None:
            self._config_cache[key] = env_value
            return env_value

        # Fallback to default
        self._config_cache[key] = default
        return default

    def set_config(self, key: str, value: Any):
        """Set configuration value"""
        self._config_cache[key] = value

    def clear_cache(self):
        """Clear configuration cache"""
        self._config_cache.clear()
```

## Debugging Tips
- **Trace utility calls** - Log all utility function invocations for debugging
- **Performance monitoring** - Monitor utility function performance
- **Error aggregation** - Aggregate and analyze utility function errors
- **Cache analysis** - Monitor cache hit rates and effectiveness
- **Security auditing** - Regularly audit guardrail effectiveness
