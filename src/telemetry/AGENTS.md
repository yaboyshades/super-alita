# Telemetry & Monitoring - Agent Instructions

## Overview
The `src/telemetry/` directory contains telemetry collection and monitoring utilities:
- **MCP Broadcasting** - Telemetry broadcasting via MCP protocol
- **Plugin Wrapping** - Plugin telemetry and instrumentation
- **Performance Monitoring** - System performance metrics collection
- **Event Tracking** - Event-based telemetry and analytics

## Key Files & Responsibilities

### Telemetry Components
- `mcp_broadcaster.py` - MCP-based telemetry broadcasting
- `plugin_wrapper.py` - Plugin instrumentation and telemetry
- `__init__.py` - Module initialization and exports

## Development Guidelines

### MCP Telemetry Broadcasting
```python
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

@dataclass
class TelemetryEvent:
    """Telemetry event structure"""
    event_id: str
    event_type: str
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    tags: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        if self.tags is None:
            result['tags'] = {}
        return result

class MCPTelemetryBroadcaster:
    """MCP-based telemetry broadcasting"""
    
    def __init__(self, mcp_client, config: Dict[str, Any] = None):
        self.mcp_client = mcp_client
        self.config = config or {}
        self.event_queue: List[TelemetryEvent] = []
        self.batch_size = self.config.get('batch_size', 100)
        self.flush_interval = self.config.get('flush_interval', 30)  # seconds
        self._flush_task = None
        
    async def start(self):
        """Start telemetry broadcasting"""
        self._flush_task = asyncio.create_task(self._periodic_flush())
        
    async def stop(self):
        """Stop telemetry broadcasting"""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush_events()
        
    async def emit_event(self, event: TelemetryEvent):
        """Emit telemetry event"""
        self.event_queue.append(event)
        
        if len(self.event_queue) >= self.batch_size:
            await self._flush_events()
            
    async def _flush_events(self):
        """Flush queued events to MCP"""
        if not self.event_queue:
            return
            
        events_to_send = self.event_queue.copy()
        self.event_queue.clear()
        
        try:
            # Send events via MCP
            payload = {
                'type': 'telemetry_batch',
                'events': [event.to_dict() for event in events_to_send],
                'batch_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            await self.mcp_client.send_message(payload)
            logger.debug(f"Sent {len(events_to_send)} telemetry events")
            
        except Exception as e:
            logger.error(f"Failed to send telemetry events: {e}")
            # Re-queue events for retry
            self.event_queue.extend(events_to_send)
            
    async def _periodic_flush(self):
        """Periodically flush events"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
```

### Plugin Telemetry Wrapper
```python
import time
import inspect
from functools import wraps
from typing import Callable, Any, Dict
from src.core.plugin_interface import PluginInterface

class TelemetryPluginWrapper:
    """Wrapper for adding telemetry to plugins"""
    
    def __init__(self, plugin: PluginInterface, telemetry_broadcaster):
        self.plugin = plugin
        self.broadcaster = telemetry_broadcaster
        self.method_metrics: Dict[str, Dict[str, Any]] = {}
        
    def wrap_plugin_methods(self):
        """Wrap plugin methods with telemetry"""
        for method_name in dir(self.plugin):
            if method_name.startswith('_'):
                continue
                
            method = getattr(self.plugin, method_name)
            if not callable(method):
                continue
                
            wrapped_method = self._wrap_method(method_name, method)
            setattr(self.plugin, method_name, wrapped_method)
            
    def _wrap_method(self, method_name: str, method: Callable) -> Callable:
        """Wrap individual method with telemetry"""
        @wraps(method)
        async def async_wrapper(*args, **kwargs):
            return await self._execute_with_telemetry(
                method_name, method, args, kwargs, is_async=True
            )
            
        @wraps(method)
        def sync_wrapper(*args, **kwargs):
            return self._execute_with_telemetry_sync(
                method_name, method, args, kwargs
            )
            
        # Check if method is async
        if inspect.iscoroutinefunction(method):
            return async_wrapper
        else:
            return sync_wrapper
            
    async def _execute_with_telemetry(
        self, method_name: str, method: Callable, args, kwargs, is_async: bool = True
    ) -> Any:
        """Execute method with telemetry collection"""
        start_time = time.time()
        success = False
        error = None
        result = None
        
        try:
            if is_async:
                result = await method(*args, **kwargs)
            else:
                result = method(*args, **kwargs)
            success = True
            return result
            
        except Exception as e:
            error = str(e)
            raise
            
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Record method metrics
            self._record_method_metrics(method_name, duration, success, error)
            
            # Emit telemetry event
            await self._emit_method_telemetry(
                method_name, duration, success, error, args, kwargs
            )
            
    def _execute_with_telemetry_sync(
        self, method_name: str, method: Callable, args, kwargs
    ) -> Any:
        """Synchronous version of telemetry execution"""
        # Use asyncio.create_task for async telemetry in sync context
        coro = self._execute_with_telemetry(method_name, method, args, kwargs, False)
        
        # If we're in an async context, await it
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            # No running loop, execute synchronously
            pass
            
        # Execute the actual method synchronously
        return method(*args, **kwargs)
        
    def _record_method_metrics(
        self, method_name: str, duration: float, success: bool, error: Optional[str]
    ):
        """Record method execution metrics"""
        if method_name not in self.method_metrics:
            self.method_metrics[method_name] = {
                'call_count': 0,
                'total_duration': 0.0,
                'success_count': 0,
                'error_count': 0,
                'avg_duration': 0.0,
                'last_error': None
            }
            
        metrics = self.method_metrics[method_name]
        metrics['call_count'] += 1
        metrics['total_duration'] += duration
        
        if success:
            metrics['success_count'] += 1
        else:
            metrics['error_count'] += 1
            metrics['last_error'] = error
            
        metrics['avg_duration'] = metrics['total_duration'] / metrics['call_count']
        
    async def _emit_method_telemetry(
        self, method_name: str, duration: float, success: bool, 
        error: Optional[str], args, kwargs
    ):
        """Emit telemetry event for method execution"""
        event = TelemetryEvent(
            event_id=f"{self.plugin.name}_{method_name}_{int(time.time() * 1000)}",
            event_type="plugin_method_execution",
            timestamp=datetime.now(timezone.utc),
            source=self.plugin.name,
            data={
                'method_name': method_name,
                'duration_seconds': duration,
                'success': success,
                'error': error,
                'arg_count': len(args),
                'kwarg_count': len(kwargs)
            },
            tags={
                'plugin': self.plugin.name,
                'method': method_name,
                'status': 'success' if success else 'error'
            }
        )
        
        await self.broadcaster.emit_event(event)
        
    def get_plugin_metrics(self) -> Dict[str, Any]:
        """Get aggregated plugin metrics"""
        return {
            'plugin_name': self.plugin.name,
            'method_metrics': self.method_metrics,
            'total_calls': sum(m['call_count'] for m in self.method_metrics.values()),
            'total_errors': sum(m['error_count'] for m in self.method_metrics.values()),
            'avg_duration': sum(m['avg_duration'] for m in self.method_metrics.values()) / len(self.method_metrics) if self.method_metrics else 0
        }
```

### System Performance Monitoring
```python
import psutil
import asyncio
from typing import Dict, List

class SystemPerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self, telemetry_broadcaster, interval: int = 60):
        self.broadcaster = telemetry_broadcaster
        self.interval = interval
        self._monitoring_task = None
        
    async def start_monitoring(self):
        """Start system performance monitoring"""
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        
    async def stop_monitoring(self):
        """Stop system performance monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
                
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(self.interval)
                
    async def _collect_system_metrics(self):
        """Collect and emit system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics (if available)
            network = psutil.net_io_counters()
            
            # Create telemetry event
            event = TelemetryEvent(
                event_id=f"system_metrics_{int(time.time())}",
                event_type="system_performance",
                timestamp=datetime.now(timezone.utc),
                source="system_monitor",
                data={
                    'cpu': {
                        'percent': cpu_percent,
                        'count': cpu_count
                    },
                    'memory': {
                        'total': memory.total,
                        'available': memory.available,
                        'percent': memory.percent,
                        'used': memory.used
                    },
                    'disk': {
                        'total': disk.total,
                        'used': disk.used,
                        'free': disk.free,
                        'percent': (disk.used / disk.total) * 100
                    },
                    'network': {
                        'bytes_sent': network.bytes_sent,
                        'bytes_recv': network.bytes_recv,
                        'packets_sent': network.packets_sent,
                        'packets_recv': network.packets_recv
                    } if network else {}
                },
                tags={'component': 'system', 'type': 'performance'}
            )
            
            await self.broadcaster.emit_event(event)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
```

## Testing Guidelines

### Telemetry Testing
```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.telemetry.mcp_broadcaster import MCPTelemetryBroadcaster, TelemetryEvent

@pytest.mark.asyncio
async def test_telemetry_event_emission():
    """Test telemetry event emission"""
    mcp_client = AsyncMock()
    broadcaster = MCPTelemetryBroadcaster(mcp_client)
    
    # Create test event
    event = TelemetryEvent(
        event_id="test_123",
        event_type="test_event",
        timestamp=datetime.now(timezone.utc),
        source="test",
        data={"key": "value"}
    )
    
    # Emit event
    await broadcaster.emit_event(event)
    
    # Verify event is queued
    assert len(broadcaster.event_queue) == 1
    assert broadcaster.event_queue[0] == event

@pytest.mark.asyncio
async def test_telemetry_batch_flushing():
    """Test telemetry batch flushing"""
    mcp_client = AsyncMock()
    config = {'batch_size': 2}
    broadcaster = MCPTelemetryBroadcaster(mcp_client, config)
    
    # Emit multiple events to trigger flush
    for i in range(3):
        event = TelemetryEvent(
            event_id=f"test_{i}",
            event_type="test_event",
            timestamp=datetime.now(timezone.utc),
            source="test",
            data={"index": i}
        )
        await broadcaster.emit_event(event)
    
    # Should have flushed batch and have 1 remaining
    assert len(broadcaster.event_queue) == 1
    mcp_client.send_message.assert_called_once()

@pytest.mark.asyncio
async def test_plugin_telemetry_wrapper():
    """Test plugin telemetry wrapper"""
    class TestPlugin(PluginInterface):
        def __init__(self, event_bus):
            super().__init__(event_bus)
            self.name = "test_plugin"
            
        async def test_method(self, param: str) -> str:
            return f"result_{param}"
    
    event_bus = AsyncMock()
    broadcaster = AsyncMock()
    
    plugin = TestPlugin(event_bus)
    wrapper = TelemetryPluginWrapper(plugin, broadcaster)
    wrapper.wrap_plugin_methods()
    
    # Call wrapped method
    result = await plugin.test_method("test")
    
    # Verify result and telemetry
    assert result == "result_test"
    broadcaster.emit_event.assert_called_once()
```

### Performance Testing
```python
@pytest.mark.performance
async def test_telemetry_performance():
    """Test telemetry performance under load"""
    mcp_client = AsyncMock()
    broadcaster = MCPTelemetryBroadcaster(mcp_client)
    
    event_count = 1000
    start_time = time.time()
    
    # Emit many events
    for i in range(event_count):
        event = TelemetryEvent(
            event_id=f"perf_test_{i}",
            event_type="performance_test",
            timestamp=datetime.now(timezone.utc),
            source="perf_test",
            data={"index": i}
        )
        await broadcaster.emit_event(event)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Should handle events efficiently
    events_per_second = event_count / duration
    assert events_per_second > 1000  # At least 1000 events/second

@pytest.mark.integration
async def test_system_monitoring_integration():
    """Test system monitoring integration"""
    broadcaster = AsyncMock()
    monitor = SystemPerformanceMonitor(broadcaster, interval=1)
    
    # Start monitoring briefly
    await monitor.start_monitoring()
    await asyncio.sleep(2)  # Let it collect metrics
    await monitor.stop_monitoring()
    
    # Verify metrics were emitted
    assert broadcaster.emit_event.call_count > 0
    
    # Check event structure
    call_args = broadcaster.emit_event.call_args_list[0][0][0]
    assert call_args.event_type == "system_performance"
    assert "cpu" in call_args.data
    assert "memory" in call_args.data
```

## Security Guidelines

### Sensitive Data Filtering
```python
import re
from typing import Any, Dict, Set

class TelemetryDataSanitizer:
    """Sanitize telemetry data to remove sensitive information"""
    
    def __init__(self):
        self.sensitive_patterns = [
            r'password.*?=.*',
            r'token.*?=.*',
            r'key.*?=.*',
            r'secret.*?=.*',
            r'credential.*?=.*'
        ]
        self.sensitive_keys = {
            'password', 'token', 'key', 'secret', 'credential',
            'auth', 'authorization', 'session_id'
        }
        
    def sanitize_event_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize event data removing sensitive information"""
        if not isinstance(data, dict):
            return data
            
        sanitized = {}
        
        for key, value in data.items():
            if self._is_sensitive_key(key):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_event_data(value)
            elif isinstance(value, str):
                sanitized[key] = self._sanitize_string_value(value)
            else:
                sanitized[key] = value
                
        return sanitized
        
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if key contains sensitive data"""
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in self.sensitive_keys)
        
    def _sanitize_string_value(self, value: str) -> str:
        """Sanitize string values"""
        for pattern in self.sensitive_patterns:
            value = re.sub(pattern, "***REDACTED***", value, flags=re.IGNORECASE)
        return value
```

### Rate Limiting
```python
import time
from collections import defaultdict

class TelemetryRateLimiter:
    """Rate limiting for telemetry events"""
    
    def __init__(self, max_events_per_minute: int = 1000):
        self.max_events_per_minute = max_events_per_minute
        self.event_counts: Dict[str, List[float]] = defaultdict(list)
        
    def should_allow_event(self, source: str) -> bool:
        """Check if event should be allowed based on rate limits"""
        current_time = time.time()
        one_minute_ago = current_time - 60
        
        # Clean old timestamps
        self.event_counts[source] = [
            ts for ts in self.event_counts[source] 
            if ts > one_minute_ago
        ]
        
        # Check if under limit
        if len(self.event_counts[source]) >= self.max_events_per_minute:
            return False
            
        # Record this event
        self.event_counts[source].append(current_time)
        return True
        
    def get_current_rate(self, source: str) -> int:
        """Get current event rate for source"""
        current_time = time.time()
        one_minute_ago = current_time - 60
        
        recent_events = [
            ts for ts in self.event_counts[source] 
            if ts > one_minute_ago
        ]
        return len(recent_events)
```

## Performance Guidelines

### Async Telemetry Processing
```python
import asyncio
from asyncio import Queue
from typing import List

class AsyncTelemetryProcessor:
    """Asynchronous telemetry event processing"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.event_queue: Queue = Queue()
        self.workers: List[asyncio.Task] = []
        self.running = False
        
    async def start(self):
        """Start telemetry processing workers"""
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.max_workers)
        ]
        
    async def stop(self):
        """Stop telemetry processing workers"""
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
            
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
    async def submit_event(self, event: TelemetryEvent):
        """Submit event for processing"""
        await self.event_queue.put(event)
        
    async def _worker(self, worker_id: int):
        """Worker coroutine for processing events"""
        while self.running:
            try:
                # Get event with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(), timeout=1.0
                )
                
                # Process event
                await self._process_event(event)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                
    async def _process_event(self, event: TelemetryEvent):
        """Process individual telemetry event"""
        # Implementation for event processing
        pass
```

## Common Patterns

### Event Correlation
```python
import uuid
from contextlib import contextmanager

class TelemetryCorrelation:
    """Correlation tracking for telemetry events"""
    
    def __init__(self):
        self._correlation_stack: List[str] = []
        
    @contextmanager
    def correlation_context(self, correlation_id: str = None):
        """Context manager for correlation tracking"""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
            
        self._correlation_stack.append(correlation_id)
        try:
            yield correlation_id
        finally:
            self._correlation_stack.pop()
            
    def get_current_correlation_id(self) -> Optional[str]:
        """Get current correlation ID"""
        return self._correlation_stack[-1] if self._correlation_stack else None
        
    def create_correlated_event(
        self, event_type: str, source: str, data: Dict[str, Any]
    ) -> TelemetryEvent:
        """Create event with correlation ID"""
        correlation_id = self.get_current_correlation_id()
        
        event_data = data.copy()
        if correlation_id:
            event_data['correlation_id'] = correlation_id
            
        return TelemetryEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            source=source,
            data=event_data
        )
```

### Telemetry Aggregation
```python
from collections import defaultdict
from typing import Dict, Any, List

class TelemetryAggregator:
    """Aggregate telemetry events for reporting"""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size  # seconds
        self.aggregated_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
    def aggregate_event(self, event: TelemetryEvent):
        """Aggregate event data"""
        window_key = self._get_window_key(event.timestamp)
        event_key = f"{event.source}_{event.event_type}"
        
        if event_key not in self.aggregated_data[window_key]:
            self.aggregated_data[window_key][event_key] = {
                'count': 0,
                'source': event.source,
                'event_type': event.event_type,
                'first_timestamp': event.timestamp,
                'last_timestamp': event.timestamp,
                'total_duration': 0.0,
                'error_count': 0
            }
            
        agg_data = self.aggregated_data[window_key][event_key]
        agg_data['count'] += 1
        agg_data['last_timestamp'] = event.timestamp
        
        # Aggregate specific metrics
        if 'duration_seconds' in event.data:
            agg_data['total_duration'] += event.data['duration_seconds']
            
        if not event.data.get('success', True):
            agg_data['error_count'] += 1
            
    def _get_window_key(self, timestamp: datetime) -> str:
        """Get aggregation window key for timestamp"""
        window_start = int(timestamp.timestamp()) // self.window_size * self.window_size
        return str(window_start)
        
    def get_aggregated_metrics(self, window_key: str) -> Dict[str, Any]:
        """Get aggregated metrics for window"""
        return self.aggregated_data.get(window_key, {})
```

## Debugging Tips
- **Event tracing** - Trace telemetry event flow through system
- **Performance monitoring** - Monitor telemetry system performance impact
- **Data validation** - Validate telemetry data structure and content
- **Correlation tracking** - Use correlation IDs to trace related events
- **Rate limit monitoring** - Monitor rate limiting effectiveness