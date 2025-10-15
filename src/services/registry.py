"""Service registry for dependency injection."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type, TypeVar

from ..app.config import ApplicationConfig
from .base import BaseService
from .event_bus import EventBusService
from .llm_client import LLMService
from .knowledge_graph import KnowledgeGraphService
from .ability_registry import AbilityRegistryService
from .constitutional import ConstitutionalService

T = TypeVar('T', bound=BaseService)

class ServiceRegistry:
    """Registry for managing application services."""
    
    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._services: Dict[str, BaseService] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all services in dependency order."""
        if self._initialized:
            return
        
        self.logger.info("Initializing service registry...")
        
        # Initialize services in dependency order
        services_to_init = [
            ("event_bus", EventBusService),
            ("knowledge_graph", KnowledgeGraphService),
            ("constitutional", ConstitutionalService),
            ("llm_client", LLMService),
            ("ability_registry", AbilityRegistryService),
        ]
        
        for service_name, service_class in services_to_init:
            try:
                self.logger.info(f"Initializing {service_name}...")
                service = service_class(self.config, self)
                await service.initialize()
                self._services[service_name] = service
                self.logger.info(f"✅ {service_name} initialized")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {service_name}: {e}")
                # Decide whether to continue or fail fast
                if service_name in {"event_bus", "constitutional"}:
                    raise  # Critical services
                # Continue with degraded functionality for optional services
        
        self._initialized = True
        self.logger.info("✅ Service registry initialized")
    
    async def startup(self) -> None:
        """Start all services."""
        for name, service in self._services.items():
            try:
                await service.startup()
                self.logger.info(f"✅ {name} service started")
            except Exception as e:
                self.logger.error(f"❌ {name} service startup failed: {e}")
    
    async def shutdown(self) -> None:
        """Stop all services gracefully."""
        # Shutdown in reverse order
        for name, service in reversed(list(self._services.items())):
            try:
                await service.shutdown()
                self.logger.info(f"✅ {name} service stopped")
            except Exception as e:
                self.logger.error(f"❌ {name} service shutdown failed: {e}")
    
    def get(self, service_name: str) -> Optional[BaseService]:
        """Get service by name."""
        return self._services.get(service_name)
    
    def require(self, service_name: str) -> BaseService:
        """Get service by name, raise if not found."""
        service = self._services.get(service_name)
        if service is None:
            raise ValueError(f"Required service not found: {service_name}")
        return service
    
    def get_typed(self, service_class: Type[T]) -> Optional[T]:
        """Get service by type."""
        for service in self._services.values():
            if isinstance(service, service_class):
                return service
        return None
    
    def require_typed(self, service_class: Type[T]) -> T:
        """Get service by type, raise if not found."""
        service = self.get_typed(service_class)
        if service is None:
            raise ValueError(f"Required service type not found: {service_class.__name__}")
        return service
    
    def list_services(self) -> Dict[str, str]:
        """List all registered services."""
        return {
            name: service.__class__.__name__ 
            for name, service in self._services.items()
        }