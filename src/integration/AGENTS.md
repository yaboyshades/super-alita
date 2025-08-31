# Integration Components - Agent Instructions

## Overview
The `src/integration/` directory contains system integration components for connecting with external services:
- **External Service Integration** - Connectors for third-party services
- **API Gateway Integration** - API gateway and proxy components
- **Protocol Bridges** - Protocol translation and bridging
- **Service Mesh Integration** - Service mesh connectivity

## Key Files & Responsibilities

### Integration Components
- `cortex_integration.py` - Cortex system integration
- Additional integration modules (to be implemented)

## Development Guidelines

### Integration Service Pattern
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import aiohttp
import asyncio

class IntegrationStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    CONNECTING = "connecting"

@dataclass
class IntegrationConfig:
    """Configuration for external integration"""
    service_name: str
    endpoint_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    headers: Dict[str, str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

@dataclass
class IntegrationResult:
    """Result of integration operation"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseIntegration(ABC):
    """Base class for external service integrations"""

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.status = IntegrationStatus.DISCONNECTED
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_error: Optional[str] = None

    async def connect(self) -> bool:
        """Establish connection to external service"""
        try:
            self.status = IntegrationStatus.CONNECTING

            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.config.headers
            )

            # Test connection
            connection_test = await self._test_connection()

            if connection_test:
                self.status = IntegrationStatus.CONNECTED
                self.last_error = None
                return True
            else:
                self.status = IntegrationStatus.ERROR
                self.last_error = "Connection test failed"
                return False

        except Exception as e:
            self.status = IntegrationStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Connection failed for {self.config.service_name}: {e}")
            return False

    async def disconnect(self):
        """Close connection to external service"""
        if self.session:
            await self.session.close()
            self.session = None
        self.status = IntegrationStatus.DISCONNECTED

    @abstractmethod
    async def _test_connection(self) -> bool:
        """Test connection to service"""
        pass

    @abstractmethod
    async def send_request(self, endpoint: str, data: Dict[str, Any]) -> IntegrationResult:
        """Send request to external service"""
        pass

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None
    ) -> IntegrationResult:
        """Make HTTP request with retry logic"""
        if not self.session:
            return IntegrationResult(
                success=False,
                error="No active session - call connect() first"
            )

        url = f"{self.config.endpoint_url.rstrip('/')}/{endpoint.lstrip('/')}"

        for attempt in range(self.config.retry_attempts):
            try:
                start_time = time.time()

                async with self.session.request(
                    method, url, json=data, params=params
                ) as response:
                    response_time = time.time() - start_time
                    response_data = await response.json() if response.content_type == 'application/json' else await response.text()

                    return IntegrationResult(
                        success=response.status < 400,
                        data=response_data,
                        status_code=response.status,
                        response_time=response_time,
                        error=None if response.status < 400 else f"HTTP {response.status}"
                    )

            except asyncio.TimeoutError:
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                return IntegrationResult(
                    success=False,
                    error="Request timeout"
                )

            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                return IntegrationResult(
                    success=False,
                    error=str(e)
                )

        return IntegrationResult(
            success=False,
            error="Max retry attempts exceeded"
        )
```

### Cortex Integration
```python
class CortexIntegration(BaseIntegration):
    """Integration with Cortex system"""

    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.cortex_session_id: Optional[str] = None

    async def _test_connection(self) -> bool:
        """Test Cortex connection"""
        try:
            result = await self._make_request("GET", "/health")
            return result.success and result.data.get("status") == "ok"
        except Exception:
            return False

    async def send_request(self, endpoint: str, data: Dict[str, Any]) -> IntegrationResult:
        """Send request to Cortex"""
        # Add Cortex-specific headers
        if self.cortex_session_id:
            headers = {"X-Cortex-Session": self.cortex_session_id}
            if self.session:
                self.session.headers.update(headers)

        return await self._make_request("POST", endpoint, data)

    async def authenticate(self, credentials: Dict[str, str]) -> IntegrationResult:
        """Authenticate with Cortex"""
        result = await self._make_request("POST", "/auth", credentials)

        if result.success and result.data:
            self.cortex_session_id = result.data.get("session_id")

        return result

    async def execute_cortex_task(self, task_type: str, task_data: Dict[str, Any]) -> IntegrationResult:
        """Execute task in Cortex"""
        request_data = {
            "type": task_type,
            "data": task_data,
            "session_id": self.cortex_session_id
        }

        return await self.send_request("/tasks", request_data)

    async def get_cortex_status(self) -> IntegrationResult:
        """Get Cortex system status"""
        return await self._make_request("GET", "/status")

    async def stream_cortex_events(self, event_filter: Dict[str, Any] = None):
        """Stream events from Cortex"""
        if not self.session:
            raise RuntimeError("No active session")

        url = f"{self.config.endpoint_url}/events/stream"
        params = event_filter or {}

        try:
            async with self.session.get(url, params=params) as response:
                async for line in response.content:
                    if line:
                        try:
                            event_data = json.loads(line.decode('utf-8'))
                            yield event_data
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in event stream: {line}")

        except Exception as e:
            logger.error(f"Error streaming Cortex events: {e}")
            raise
```

### API Gateway Integration
```python
class APIGatewayIntegration:
    """Integration with API Gateway services"""

    def __init__(self, gateway_config: Dict[str, Any]):
        self.gateway_url = gateway_config["url"]
        self.api_key = gateway_config.get("api_key")
        self.service_registry: Dict[str, Dict[str, Any]] = {}

    async def register_service(
        self,
        service_name: str,
        service_config: Dict[str, Any]
    ) -> bool:
        """Register service with API gateway"""
        registration_data = {
            "name": service_name,
            "url": service_config["url"],
            "health_check": service_config.get("health_check", "/health"),
            "load_balancing": service_config.get("load_balancing", "round_robin"),
            "timeout": service_config.get("timeout", 30)
        }

        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

            async with session.post(
                f"{self.gateway_url}/services",
                json=registration_data,
                headers=headers
            ) as response:
                if response.status == 201:
                    self.service_registry[service_name] = service_config
                    return True
                else:
                    logger.error(f"Service registration failed: {response.status}")
                    return False

    async def unregister_service(self, service_name: str) -> bool:
        """Unregister service from API gateway"""
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

            async with session.delete(
                f"{self.gateway_url}/services/{service_name}",
                headers=headers
            ) as response:
                if response.status == 204:
                    self.service_registry.pop(service_name, None)
                    return True
                else:
                    return False

    async def proxy_request(
        self,
        service_name: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """Proxy request through API gateway"""
        gateway_endpoint = f"/proxy/{service_name}{endpoint}"

        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

            try:
                async with session.request(
                    method,
                    f"{self.gateway_url}{gateway_endpoint}",
                    json=data,
                    headers=headers
                ) as response:
                    response_data = await response.json() if response.content_type == 'application/json' else await response.text()

                    return IntegrationResult(
                        success=response.status < 400,
                        data=response_data,
                        status_code=response.status
                    )

            except Exception as e:
                return IntegrationResult(
                    success=False,
                    error=str(e)
                )
```

### Protocol Bridge
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MessageProtocol(Protocol):
    """Protocol for message handling"""

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message using protocol"""
        ...

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message using protocol"""
        ...

class ProtocolBridge:
    """Bridge between different protocols"""

    def __init__(self):
        self.protocols: Dict[str, MessageProtocol] = {}
        self.message_transformers: Dict[Tuple[str, str], Callable] = {}

    def register_protocol(self, name: str, protocol: MessageProtocol):
        """Register protocol handler"""
        self.protocols[name] = protocol

    def register_transformer(
        self,
        from_protocol: str,
        to_protocol: str,
        transformer: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        """Register message transformer between protocols"""
        self.message_transformers[(from_protocol, to_protocol)] = transformer

    async def bridge_message(
        self,
        message: Dict[str, Any],
        from_protocol: str,
        to_protocol: str
    ) -> bool:
        """Bridge message between protocols"""
        if from_protocol not in self.protocols:
            raise ValueError(f"Unknown source protocol: {from_protocol}")

        if to_protocol not in self.protocols:
            raise ValueError(f"Unknown target protocol: {to_protocol}")

        # Transform message if transformer exists
        transformer_key = (from_protocol, to_protocol)
        if transformer_key in self.message_transformers:
            transformer = self.message_transformers[transformer_key]
            transformed_message = transformer(message)
        else:
            transformed_message = message

        # Send via target protocol
        target_protocol = self.protocols[to_protocol]
        return await target_protocol.send_message(transformed_message)

# Example protocol implementations
class HTTPProtocol:
    """HTTP protocol implementation"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message via HTTP"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.post(
                f"{self.base_url}/messages",
                json=message
            ) as response:
                return response.status < 400
        except Exception:
            return False

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message via HTTP polling"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.get(
                f"{self.base_url}/messages/next"
            ) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception:
            return None

class WebSocketProtocol:
    """WebSocket protocol implementation"""

    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.websocket: Optional[aiohttp.ClientWebSocketResponse] = None

    async def connect(self):
        """Connect to WebSocket"""
        session = aiohttp.ClientSession()
        self.websocket = await session.ws_connect(self.ws_url)

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message via WebSocket"""
        if not self.websocket:
            await self.connect()

        try:
            await self.websocket.send_json(message)
            return True
        except Exception:
            return False

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message via WebSocket"""
        if not self.websocket:
            return None

        try:
            msg = await self.websocket.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                return json.loads(msg.data)
            return None
        except Exception:
            return None
```

### Service Mesh Integration
```python
class ServiceMeshIntegration:
    """Integration with service mesh (e.g., Istio, Linkerd)"""

    def __init__(self, mesh_config: Dict[str, Any]):
        self.mesh_config = mesh_config
        self.service_map: Dict[str, str] = {}  # service_name -> mesh_endpoint

    async def register_with_mesh(self, service_name: str, service_port: int) -> bool:
        """Register service with service mesh"""
        mesh_endpoint = f"{service_name}.{self.mesh_config['namespace']}.svc.cluster.local:{service_port}"

        # Register with service mesh control plane
        registration_data = {
            "service_name": service_name,
            "namespace": self.mesh_config["namespace"],
            "port": service_port,
            "labels": self.mesh_config.get("labels", {}),
            "annotations": self.mesh_config.get("annotations", {})
        }

        try:
            # Call service mesh API to register service
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.mesh_config['control_plane_url']}/services",
                    json=registration_data
                ) as response:
                    if response.status == 201:
                        self.service_map[service_name] = mesh_endpoint
                        return True
                    return False

        except Exception as e:
            logger.error(f"Service mesh registration failed: {e}")
            return False

    async def discover_services(self) -> List[Dict[str, Any]]:
        """Discover services in mesh"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.mesh_config['control_plane_url']}/services"
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return []

        except Exception as e:
            logger.error(f"Service discovery failed: {e}")
            return []

    async def call_mesh_service(
        self,
        service_name: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """Call service through mesh"""
        if service_name not in self.service_map:
            return IntegrationResult(
                success=False,
                error=f"Service {service_name} not found in mesh"
            )

        mesh_endpoint = self.service_map[service_name]
        url = f"http://{mesh_endpoint}{endpoint}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    response_data = await response.json()

                    return IntegrationResult(
                        success=response.status < 400,
                        data=response_data,
                        status_code=response.status
                    )

        except Exception as e:
            return IntegrationResult(
                success=False,
                error=str(e)
            )
```

## Testing Guidelines

### Integration Testing
```python
import pytest
from unittest.mock import AsyncMock, patch
from src.integration.cortex_integration import CortexIntegration, IntegrationConfig

@pytest.mark.asyncio
async def test_cortex_integration():
    """Test Cortex integration"""
    config = IntegrationConfig(
        service_name="cortex",
        endpoint_url="http://localhost:8080",
        api_key="test_key"
    )

    with patch('aiohttp.ClientSession') as mock_session:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

        integration = CortexIntegration(config)

        # Test connection
        connected = await integration.connect()
        assert connected is True
        assert integration.status == IntegrationStatus.CONNECTED

@pytest.mark.asyncio
async def test_api_gateway_integration():
    """Test API gateway integration"""
    gateway_config = {
        "url": "http://localhost:8000",
        "api_key": "test_key"
    }

    with patch('aiohttp.ClientSession') as mock_session:
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

        gateway = APIGatewayIntegration(gateway_config)

        service_config = {
            "url": "http://service:8080",
            "health_check": "/health"
        }

        # Test service registration
        registered = await gateway.register_service("test_service", service_config)
        assert registered is True
        assert "test_service" in gateway.service_registry

def test_protocol_bridge():
    """Test protocol bridge functionality"""
    bridge = ProtocolBridge()

    # Register mock protocols
    http_protocol = HTTPProtocol("http://localhost:8080")
    ws_protocol = WebSocketProtocol("ws://localhost:8080/ws")

    bridge.register_protocol("http", http_protocol)
    bridge.register_protocol("websocket", ws_protocol)

    # Register transformer
    def http_to_ws_transformer(message):
        return {"type": "websocket", "payload": message}

    bridge.register_transformer("http", "websocket", http_to_ws_transformer)

    # Test transformer registration
    assert ("http", "websocket") in bridge.message_transformers

@pytest.mark.integration
async def test_end_to_end_integration():
    """Test end-to-end integration flow"""
    # This would test actual integration with running services
    pass
```

### Mock Integration Testing
```python
class MockIntegration(BaseIntegration):
    """Mock integration for testing"""

    def __init__(self, config: IntegrationConfig, should_fail: bool = False):
        super().__init__(config)
        self.should_fail = should_fail
        self.request_log: List[Dict[str, Any]] = []

    async def _test_connection(self) -> bool:
        return not self.should_fail

    async def send_request(self, endpoint: str, data: Dict[str, Any]) -> IntegrationResult:
        self.request_log.append({
            "endpoint": endpoint,
            "data": data,
            "timestamp": time.time()
        })

        if self.should_fail:
            return IntegrationResult(
                success=False,
                error="Mock integration failure"
            )
        else:
            return IntegrationResult(
                success=True,
                data={"mock": "response", "endpoint": endpoint},
                status_code=200
            )

@pytest.fixture
def mock_integration():
    """Mock integration fixture"""
    config = IntegrationConfig(
        service_name="mock",
        endpoint_url="http://mock.example.com"
    )
    return MockIntegration(config)

@pytest.mark.asyncio
async def test_with_mock_integration(mock_integration):
    """Test using mock integration"""
    connected = await mock_integration.connect()
    assert connected is True

    result = await mock_integration.send_request("/test", {"data": "test"})
    assert result.success is True
    assert len(mock_integration.request_log) == 1
```

## Security Guidelines

### Secure Integration Patterns
```python
import ssl
from cryptography.fernet import Fernet

class SecureIntegration(BaseIntegration):
    """Secure integration with encryption and authentication"""

    def __init__(self, config: IntegrationConfig, encryption_key: bytes = None):
        super().__init__(config)
        self.encryption_key = encryption_key
        self.cipher_suite = Fernet(encryption_key) if encryption_key else None

    async def connect(self) -> bool:
        """Secure connection with SSL verification"""
        try:
            self.status = IntegrationStatus.CONNECTING

            # Create SSL context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED

            # Create secure session
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)

            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self.config.headers
            )

            # Test secure connection
            connection_test = await self._test_connection()

            if connection_test:
                self.status = IntegrationStatus.CONNECTED
                return True
            else:
                self.status = IntegrationStatus.ERROR
                return False

        except Exception as e:
            self.status = IntegrationStatus.ERROR
            self.last_error = str(e)
            return False

    def encrypt_data(self, data: Dict[str, Any]) -> str:
        """Encrypt data for transmission"""
        if not self.cipher_suite:
            raise ValueError("No encryption key configured")

        data_json = json.dumps(data)
        encrypted_data = self.cipher_suite.encrypt(data_json.encode())
        return encrypted_data.decode()

    def decrypt_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt received data"""
        if not self.cipher_suite:
            raise ValueError("No encryption key configured")

        decrypted_data = self.cipher_suite.decrypt(encrypted_data.encode())
        return json.loads(decrypted_data.decode())

    async def secure_send_request(
        self,
        endpoint: str,
        data: Dict[str, Any]
    ) -> IntegrationResult:
        """Send encrypted request"""
        if self.cipher_suite:
            encrypted_payload = self.encrypt_data(data)
            request_data = {"encrypted_payload": encrypted_payload}
        else:
            request_data = data

        return await self._make_request("POST", endpoint, request_data)
```

### API Key Management
```python
class APIKeyManager:
    """Secure API key management"""

    def __init__(self, key_store_path: str = None):
        self.key_store_path = key_store_path
        self.api_keys: Dict[str, str] = {}

    def store_api_key(self, service_name: str, api_key: str):
        """Store API key securely"""
        # In production, use proper secret management
        encrypted_key = self._encrypt_key(api_key)
        self.api_keys[service_name] = encrypted_key

    def get_api_key(self, service_name: str) -> Optional[str]:
        """Get API key for service"""
        encrypted_key = self.api_keys.get(service_name)
        if encrypted_key:
            return self._decrypt_key(encrypted_key)
        return None

    def _encrypt_key(self, key: str) -> str:
        """Encrypt API key"""
        # Simple encryption - use proper encryption in production
        return key[::-1]  # Reverse string as simple obfuscation

    def _decrypt_key(self, encrypted_key: str) -> str:
        """Decrypt API key"""
        return encrypted_key[::-1]  # Reverse back

    def rotate_key(self, service_name: str, new_key: str):
        """Rotate API key for service"""
        self.store_api_key(service_name, new_key)
        logger.info(f"API key rotated for service: {service_name}")
```

## Performance Guidelines

### Connection Pooling
```python
class ConnectionPoolManager:
    """Manage connection pools for integrations"""

    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self.connection_pools: Dict[str, aiohttp.ClientSession] = {}

    async def get_session(self, service_name: str, config: IntegrationConfig) -> aiohttp.ClientSession:
        """Get or create connection pool for service"""
        if service_name not in self.connection_pools:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=20
            )
            timeout = aiohttp.ClientTimeout(total=config.timeout)

            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=config.headers
            )

            self.connection_pools[service_name] = session

        return self.connection_pools[service_name]

    async def close_all_sessions(self):
        """Close all connection pools"""
        for session in self.connection_pools.values():
            await session.close()
        self.connection_pools.clear()
```

### Circuit Breaker for Integrations
```python
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class IntegrationCircuitBreaker:
    """Circuit breaker for integration calls"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED

    async def call(self, integration: BaseIntegration, endpoint: str, data: Dict[str, Any]) -> IntegrationResult:
        """Make integration call with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                return IntegrationResult(
                    success=False,
                    error="Circuit breaker is OPEN"
                )

        try:
            result = await integration.send_request(endpoint, data)

            if result.success:
                self._on_success()
            else:
                self._on_failure()

            return result

        except Exception as e:
            self._on_failure()
            return IntegrationResult(
                success=False,
                error=str(e)
            )

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

## Common Patterns

### Integration Factory
```python
class IntegrationFactory:
    """Factory for creating integrations"""

    @staticmethod
    def create_integration(service_type: str, config: IntegrationConfig) -> BaseIntegration:
        """Create integration based on service type"""
        if service_type == "cortex":
            return CortexIntegration(config)
        elif service_type == "api_gateway":
            return APIGatewayIntegration(config.dict())
        elif service_type == "generic":
            return BaseIntegration(config)
        else:
            raise ValueError(f"Unknown service type: {service_type}")

    @staticmethod
    def create_from_config_file(config_file: str) -> List[BaseIntegration]:
        """Create integrations from configuration file"""
        with open(config_file) as f:
            configs = json.load(f)

        integrations = []
        for service_config in configs:
            integration_config = IntegrationConfig(**service_config)
            integration = IntegrationFactory.create_integration(
                service_config["type"],
                integration_config
            )
            integrations.append(integration)

        return integrations
```

### Health Monitoring
```python
class IntegrationHealthMonitor:
    """Monitor health of integrations"""

    def __init__(self, integrations: List[BaseIntegration]):
        self.integrations = {
            integration.config.service_name: integration
            for integration in integrations
        }
        self.health_status: Dict[str, Dict[str, Any]] = {}

    async def check_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all integrations"""
        health_checks = []

        for service_name, integration in self.integrations.items():
            health_checks.append(self._check_service_health(service_name, integration))

        results = await asyncio.gather(*health_checks, return_exceptions=True)

        for i, (service_name, _) in enumerate(self.integrations.items()):
            if isinstance(results[i], Exception):
                self.health_status[service_name] = {
                    "status": "error",
                    "error": str(results[i]),
                    "timestamp": time.time()
                }
            else:
                self.health_status[service_name] = results[i]

        return self.health_status

    async def _check_service_health(self, service_name: str, integration: BaseIntegration) -> Dict[str, Any]:
        """Check health of single integration"""
        try:
            if integration.status != IntegrationStatus.CONNECTED:
                await integration.connect()

            # Test connection
            connection_ok = await integration._test_connection()

            return {
                "status": "healthy" if connection_ok else "unhealthy",
                "connection_status": integration.status.value,
                "last_error": integration.last_error,
                "timestamp": time.time()
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
```

## Debugging Tips
- **Connection tracing** - Trace all integration connections and requests
- **Response logging** - Log all integration responses for debugging
- **Health monitoring** - Monitor integration health and status
- **Circuit breaker logs** - Log circuit breaker state changes
- **Performance metrics** - Track integration response times and failure rates
