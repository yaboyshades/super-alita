# External Service Adapters - Agent Instructions

## Overview
The `src/adapters/` directory contains adapter implementations for integrating external services and frameworks:
- **LangChain Integration** - Adapter for LangChain framework components
- **Service Abstractions** - Common patterns for external service integration
- **Protocol Translation** - Converting between different service protocols

## Key Files & Responsibilities

### Framework Adapters
- `langchain_adapter.py` - LangChain framework integration adapter
- `__init__.py` - Module initialization and exports

## Development Guidelines

### Creating Service Adapters
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from src.core.plugin_interface import PluginInterface

class ServiceAdapter(ABC):
    """Base class for external service adapters"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.client = None
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to external service"""
        pass
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up connection resources"""
        pass
        
    @abstractmethod
    async def execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute request against external service"""
        pass
```

### LangChain Integration Patterns
```python
from langchain.schema import BaseMessage
from langchain.callbacks.base import BaseCallbackHandler

class SuperAlitaCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for LangChain integration"""
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Handle LLM execution start"""
        event = create_event(
            "langchain_llm_start",
            prompts=prompts,
            metadata=kwargs
        )
        asyncio.create_task(self.event_bus.publish(event))
        
    def on_llm_end(self, response, **kwargs):
        """Handle LLM execution completion"""
        event = create_event(
            "langchain_llm_end",
            response=response.dict(),
            metadata=kwargs
        )
        asyncio.create_task(self.event_bus.publish(event))
```

### Adapter Configuration
```python
# Environment-based configuration
ADAPTER_CONFIG = {
    'langchain': {
        'model_name': os.getenv('LANGCHAIN_MODEL', 'gpt-3.5-turbo'),
        'temperature': float(os.getenv('LANGCHAIN_TEMP', '0.7')),
        'max_tokens': int(os.getenv('LANGCHAIN_MAX_TOKENS', '1000')),
        'timeout': int(os.getenv('LANGCHAIN_TIMEOUT', '30'))
    }
}

def load_adapter_config(adapter_name: str) -> Dict[str, Any]:
    """Load configuration for specific adapter"""
    return ADAPTER_CONFIG.get(adapter_name, {})
```

## Security Guidelines

### API Key Management
```python
import os
from typing import Optional

def get_secure_api_key(service_name: str) -> Optional[str]:
    """Securely retrieve API key for external service"""
    key_env = f"{service_name.upper()}_API_KEY"
    api_key = os.getenv(key_env)
    
    if not api_key:
        logger.warning(f"No API key found for {service_name}")
        return None
        
    # Validate key format if needed
    if len(api_key) < 10:
        logger.error(f"Invalid API key format for {service_name}")
        return None
        
    return api_key
```

### Input Sanitization
```python
def sanitize_adapter_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize input data for external service calls"""
    sanitized = {}
    
    for key, value in data.items():
        # Remove potentially dangerous fields
        if key.startswith('_') or key in ['__class__', 'eval', 'exec']:
            continue
            
        # Sanitize string values
        if isinstance(value, str):
            sanitized[key] = value.strip()[:1000]  # Limit length
        elif isinstance(value, (int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, dict):
            sanitized[key] = sanitize_adapter_input(value)
            
    return sanitized
```

## Testing Guidelines

### Adapter Testing Pattern
```python
import pytest
from unittest.mock import AsyncMock, patch
from src.adapters.your_adapter import YourAdapter

@pytest.mark.asyncio
async def test_adapter_connection():
    """Test adapter connection establishment"""
    adapter = YourAdapter({'api_key': 'test_key'})
    
    with patch('aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.get.return_value.status = 200
        
        result = await adapter.connect()
        assert result is True

@pytest.mark.asyncio
async def test_adapter_request_execution():
    """Test adapter request execution"""
    adapter = YourAdapter({'api_key': 'test_key'})
    
    with patch.object(adapter, '_make_api_call') as mock_call:
        mock_call.return_value = {'result': 'success'}
        
        response = await adapter.execute_request({'query': 'test'})
        assert response['result'] == 'success'

@pytest.mark.asyncio
async def test_adapter_error_handling():
    """Test adapter error handling"""
    adapter = YourAdapter({'api_key': 'test_key'})
    
    with patch.object(adapter, '_make_api_call') as mock_call:
        mock_call.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            await adapter.execute_request({'query': 'test'})
```

### Integration Testing
- Test with actual external services using test credentials
- Verify rate limiting and timeout handling
- Test authentication and authorization flows
- Validate response format consistency

## Performance Guidelines

### Connection Management
```python
import aiohttp
from contextlib import asynccontextmanager

class OptimizedAdapter:
    def __init__(self, config):
        self.config = config
        self._session = None
        
    async def get_session(self):
        """Get or create HTTP session"""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.get('timeout', 30))
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
        
    async def cleanup(self):
        """Clean up resources"""
        if self._session:
            await self._session.close()
            self._session = None
```

### Response Caching
```python
from functools import wraps
import hashlib
import json

def cache_adapter_response(ttl_seconds: int = 300):
    """Decorator to cache adapter responses"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, request_data, *args, **kwargs):
            # Generate cache key
            cache_key = hashlib.md5(
                json.dumps(request_data, sort_keys=True).encode()
            ).hexdigest()
            
            # Check cache first
            cached_result = await self._get_cached_response(cache_key)
            if cached_result:
                return cached_result
                
            # Execute request and cache result
            result = await func(self, request_data, *args, **kwargs)
            await self._cache_response(cache_key, result, ttl_seconds)
            return result
            
        return wrapper
    return decorator
```

## Common Patterns

### Retry Logic
```python
import asyncio
from typing import Callable, Any

async def retry_adapter_call(
    func: Callable,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    *args,
    **kwargs
) -> Any:
    """Retry adapter calls with exponential backoff"""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = backoff_factor * (2 ** attempt)
                await asyncio.sleep(delay)
            else:
                raise last_exception
```

### Health Checks
```python
async def check_adapter_health(self) -> Dict[str, Any]:
    """Check adapter health and connectivity"""
    health_status = {
        'status': 'unknown',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'checks': {}
    }
    
    try:
        # Test connection
        connection_ok = await self._test_connection()
        health_status['checks']['connection'] = connection_ok
        
        # Test authentication
        auth_ok = await self._test_authentication()
        health_status['checks']['authentication'] = auth_ok
        
        # Overall status
        if all(health_status['checks'].values()):
            health_status['status'] = 'healthy'
        else:
            health_status['status'] = 'unhealthy'
            
    except Exception as e:
        health_status['status'] = 'error'
        health_status['error'] = str(e)
        
    return health_status
```

## Debugging Tips
- **Connection logging** - Log all external service interactions
- **Request/response tracing** - Trace full request/response cycles
- **Performance monitoring** - Monitor adapter response times
- **Error aggregation** - Aggregate and analyze adapter errors