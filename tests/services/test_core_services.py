"""Tests for core service functionality."""

import pytest
from unittest.mock import MagicMock

from src.app.config import ApplicationConfig
from src.services.registry import ServiceRegistry

class TestServiceRegistry:
    """Test service registry functionality."""
    
    @pytest.fixture
    def test_config(self):
        return ApplicationConfig(profile="test")
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, test_config):
        """Test that services initialize correctly."""
        registry = ServiceRegistry(test_config)
        await registry.initialize()
        
        assert registry._initialized
        assert registry.get("event_bus") is not None
        assert registry.get("constitutional") is not None
    
    def test_service_list(self, test_config):
        """Test service listing."""
        registry = ServiceRegistry(test_config)
        services = registry.list_services()
        
        assert isinstance(services, dict)