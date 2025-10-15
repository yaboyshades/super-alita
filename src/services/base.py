"""Base service class for dependency injection."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..app.config import ApplicationConfig
    from .registry import ServiceRegistry

class BaseService(ABC):
    """Base class for all application services."""
    
    def __init__(self, config: ApplicationConfig, registry: ServiceRegistry):
        self.config = config
        self.registry = registry
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
        self._started = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        pass
    
    async def startup(self) -> None:
        """Start the service (called after all services are initialized)."""
        if not self._initialized:
            await self.initialize()
        self._started = True
        self.logger.info(f"Service {self.__class__.__name__} started")
    
    async def shutdown(self) -> None:
        """Shutdown the service gracefully."""
        self._started = False
        self.logger.info(f"Service {self.__class__.__name__} stopped")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return {
            "service": self.__class__.__name__,
            "initialized": self._initialized,
            "started": self._started,
            "status": "healthy" if (self._initialized and self._started) else "unhealthy"
        }
    
    def get_service(self, service_name: str) -> Any:
        """Get another service from registry."""
        return self.registry.get(service_name)
    
    def require_service(self, service_name: str) -> Any:
        """Get required service from registry."""
        return self.registry.require(service_name)