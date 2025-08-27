# System Orchestration - Agent Instructions

## Overview
The `src/orchestration/` directory contains components for coordinating and managing system operations:
- **Dispatcher** - Central request routing and task distribution
- **Router** - Intelligent routing between components and services
- **Cortex Weaning** - Transition management for external dependencies
- **Workflow Coordination** - Multi-step process orchestration

## Key Files & Responsibilities

### Core Orchestration Components
- `dispatcher.py` - Central request dispatcher and task coordination
- `router.py` - Intelligent routing logic for requests and events
- `cortex_weaning.py` - Management of Cortex integration lifecycle
- `__init__.py` - Module initialization and exports

## Development Guidelines

### Dispatcher Implementation
```python
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from src.core.plugin_interface import PluginInterface
from src.core.events import create_event

@dataclass
class DispatchRequest:
    """Request structure for dispatcher"""
    request_id: str
    request_type: str
    payload: Dict[str, Any]
    priority: int = 1
    timeout: Optional[int] = None

class TaskDispatcher(PluginInterface):
    """Central task dispatcher for system orchestration"""
    
    def __init__(self, event_bus, config=None):
        super().__init__(event_bus, config)
        self.name = "task_dispatcher"
        self.active_tasks: Dict[str, DispatchRequest] = {}
        self.worker_pools: Dict[str, List[str]] = {}
        
    async def dispatch_task(self, request: DispatchRequest) -> str:
        """Dispatch task to appropriate worker"""
        # Find best worker for task
        worker_id = await self._select_worker(request.request_type)
        
        # Register active task
        self.active_tasks[request.request_id] = request
        
        # Emit dispatch event
        event = create_event(
            "task_dispatched",
            request_id=request.request_id,
            worker_id=worker_id,
            task_type=request.request_type,
            source_plugin=self.name
        )
        await self.event_bus.publish(event)
        
        return worker_id
```

### Router Configuration
```python
from enum import Enum
from typing import Callable, Dict, Any

class RoutingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PRIORITY_BASED = "priority_based"
    CAPABILITY_MATCH = "capability_match"

class IntelligentRouter:
    """Intelligent routing for requests and events"""
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_MATCH):
        self.strategy = strategy
        self.route_table: Dict[str, List[str]] = {}
        self.load_metrics: Dict[str, float] = {}
        
    async def route_request(self, request_type: str, payload: Dict[str, Any]) -> str:
        """Route request to best available handler"""
        available_handlers = self.route_table.get(request_type, [])
        
        if not available_handlers:
            raise ValueError(f"No handlers available for request type: {request_type}")
            
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_handlers)
        elif self.strategy == RoutingStrategy.LEAST_LOADED:
            return self._least_loaded_select(available_handlers)
        elif self.strategy == RoutingStrategy.CAPABILITY_MATCH:
            return await self._capability_match_select(request_type, payload, available_handlers)
        else:
            return available_handlers[0]  # Default fallback
```

### Cortex Integration Management
```python
from typing import Optional
import asyncio

class CortexWeaningManager:
    """Manages transition away from Cortex dependencies"""
    
    def __init__(self, transition_timeline: int = 30):  # days
        self.transition_timeline = transition_timeline
        self.cortex_endpoints: Dict[str, bool] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        
    async def register_cortex_endpoint(self, endpoint: str, fallback_handler: Callable):
        """Register Cortex endpoint with fallback handler"""
        self.cortex_endpoints[endpoint] = True
        self.fallback_handlers[endpoint] = fallback_handler
        
    async def execute_with_fallback(self, endpoint: str, *args, **kwargs) -> Any:
        """Execute request with Cortex fallback to local handler"""
        if self.cortex_endpoints.get(endpoint, False):
            try:
                # Try Cortex first
                return await self._call_cortex_endpoint(endpoint, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Cortex endpoint {endpoint} failed: {e}")
                # Fall back to local handler
                return await self._execute_fallback(endpoint, *args, **kwargs)
        else:
            # Use local handler directly
            return await self._execute_fallback(endpoint, *args, **kwargs)
```

## Security Guidelines

### Request Validation
```python
from pydantic import BaseModel, validator
from typing import Any

class SecureDispatchRequest(BaseModel):
    """Secure request model with validation"""
    request_id: str
    request_type: str
    payload: Dict[str, Any]
    source_id: str
    timestamp: datetime
    
    @validator('request_type')
    def validate_request_type(cls, v):
        allowed_types = [
            'tool_execution', 'data_processing', 'analysis_request',
            'plugin_communication', 'system_command'
        ]
        if v not in allowed_types:
            raise ValueError(f"Invalid request type: {v}")
        return v
        
    @validator('payload')
    def validate_payload_size(cls, v):
        # Limit payload size to prevent DoS
        payload_str = json.dumps(v)
        if len(payload_str) > 1024 * 1024:  # 1MB limit
            raise ValueError("Payload too large")
        return v
```

### Authorization Checks
```python
def require_authorization(allowed_roles: List[str]):
    """Decorator for orchestration authorization"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, request: DispatchRequest, *args, **kwargs):
            # Extract authorization from request
            auth_token = request.payload.get('auth_token')
            user_role = await self._validate_auth_token(auth_token)
            
            if user_role not in allowed_roles:
                raise PermissionError(f"Insufficient permissions for {func.__name__}")
                
            return await func(self, request, *args, **kwargs)
        return wrapper
    return decorator
```

## Testing Guidelines

### Orchestration Testing
```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.orchestration.dispatcher import TaskDispatcher

@pytest.mark.asyncio
async def test_task_dispatching():
    """Test task dispatching functionality"""
    event_bus = AsyncMock()
    dispatcher = TaskDispatcher(event_bus)
    
    # Create test request
    request = DispatchRequest(
        request_id="test_123",
        request_type="data_processing",
        payload={"data": "test"}
    )
    
    # Mock worker selection
    dispatcher._select_worker = AsyncMock(return_value="worker_1")
    
    # Test dispatch
    worker_id = await dispatcher.dispatch_task(request)
    assert worker_id == "worker_1"
    assert request.request_id in dispatcher.active_tasks
    
    # Verify event emission
    event_bus.publish.assert_called_once()

@pytest.mark.asyncio
async def test_routing_strategies():
    """Test different routing strategies"""
    router = IntelligentRouter(RoutingStrategy.ROUND_ROBIN)
    
    # Setup route table
    router.route_table["test_type"] = ["handler_1", "handler_2", "handler_3"]
    
    # Test round robin
    results = []
    for _ in range(6):
        handler = await router.route_request("test_type", {})
        results.append(handler)
    
    # Should cycle through handlers
    assert results == ["handler_1", "handler_2", "handler_3"] * 2
```

### Integration Testing
- Test end-to-end request routing
- Verify fallback mechanisms work correctly
- Test load balancing under high load
- Validate timeout and error handling

## Performance Guidelines

### Async Orchestration
```python
import asyncio
from typing import List, Coroutine

async def parallel_orchestration(tasks: List[Coroutine]) -> List[Any]:
    """Execute multiple tasks in parallel"""
    # Use semaphore to limit concurrency
    semaphore = asyncio.Semaphore(10)
    
    async def limited_task(coro):
        async with semaphore:
            return await coro
    
    # Execute all tasks with concurrency limit
    return await asyncio.gather(*[limited_task(task) for task in tasks])

class PerformanceTracker:
    """Track orchestration performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        
    async def track_execution_time(self, operation_name: str, coro: Coroutine):
        """Track execution time for operations"""
        start_time = time.time()
        try:
            result = await coro
            execution_time = time.time() - start_time
            
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            self.metrics[operation_name].append(execution_time)
            
            return result
        except Exception as e:
            logger.error(f"Operation {operation_name} failed after {time.time() - start_time:.2f}s: {e}")
            raise
```

## Common Patterns

### Circuit Breaker Pattern
```python
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker for external service calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
                
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
            
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

### Event Sourcing
```python
from dataclasses import dataclass
from typing import List, Any
from datetime import datetime, timezone

@dataclass
class OrchestrationEvent:
    """Event for orchestration event sourcing"""
    event_id: str
    event_type: str
    aggregate_id: str
    data: Dict[str, Any]
    timestamp: datetime
    version: int

class EventStore:
    """Store orchestration events for replay and audit"""
    
    def __init__(self):
        self.events: List[OrchestrationEvent] = []
        
    async def append_event(self, event: OrchestrationEvent):
        """Append event to store"""
        event.timestamp = datetime.now(timezone.utc)
        event.version = len([e for e in self.events if e.aggregate_id == event.aggregate_id]) + 1
        self.events.append(event)
        
    async def get_events(self, aggregate_id: str) -> List[OrchestrationEvent]:
        """Get events for specific aggregate"""
        return [e for e in self.events if e.aggregate_id == aggregate_id]
        
    async def replay_events(self, aggregate_id: str, handler: Callable):
        """Replay events for aggregate reconstruction"""
        events = await self.get_events(aggregate_id)
        for event in sorted(events, key=lambda e: e.version):
            await handler(event)
```

## Debugging Tips
- **Request tracing** - Trace requests through entire orchestration flow
- **Load monitoring** - Monitor worker load and queue sizes
- **Timeout analysis** - Analyze timeout patterns and adjust limits
- **Fallback verification** - Verify fallback mechanisms work as expected
- **Performance profiling** - Profile orchestration bottlenecks and optimize