"""
Simplified Mangle Integration Test

Tests the core Mangle integration components without requiring 
the full gRPC protobuf code generation.
"""
import pytest
pytest.skip("legacy test", allow_module_level=True)

import asyncio
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from core.mangle.metrics import PrometheusMetricsCollector
from core.mangle.redis_event_bus import RedisEventBus
from core.events import create_event, BaseEvent
from core.telemetry.simple_event_bus import SimpleEventBus

@pytest.fixture
def workspace_root():
    """Provide workspace root path."""
    return Path(__file__).parent

@pytest.fixture
def custom_registry():
    """Create a custom Prometheus registry for each test."""
    try:
        from prometheus_client import CollectorRegistry
        return CollectorRegistry()
    except ImportError:
        return None

@pytest.fixture
def mock_cortex_runtime():
    """Mock Cortex runtime for testing."""
    cortex = Mock()
    cortex.process_cycle = AsyncMock(return_value="Cortex processing complete")
    cortex.create_context = Mock(return_value={"session_id": "test_session"})
    cortex.modules = {"perception": Mock(), "reasoning": Mock(), "action": Mock()}
    return cortex

@pytest.fixture
def event_bus():
    """Provide simple event bus for testing."""
    return SimpleEventBus()

class TestPrometheusMetrics:
    """Test Prometheus metrics collection."""
    
    def test_metrics_collector_initialization(self, custom_registry):
        """Test metrics collector initialization."""
        collector = PrometheusMetricsCollector(custom_registry)
        assert not collector._initialized
        
        # Test lazy initialization
        collector.set_system_uptime(123.45)
        assert collector._initialized
        assert len(collector._metrics) > 0
    
    def test_system_metrics(self, custom_registry):
        """Test system metrics collection."""
        collector = PrometheusMetricsCollector(custom_registry)
        
        collector.set_system_uptime(100.0)
        collector.set_component_health("cortex", True)
        collector.set_component_health("redis", False)
        
        # Test health check
        health = collector.health_check()
        assert health["status"] == "healthy"
        assert "metrics_count" in health
    
    def test_cortex_metrics(self, custom_registry):
        """Test Cortex-specific metrics."""
        collector = PrometheusMetricsCollector(custom_registry)
        
        collector.inc_cortex_cycles("session_1", "success")
        collector.observe_cortex_cycle_duration("session_1", "perception", 0.5)
        collector.set_cortex_active_sessions(3)
        
        # Verify metrics are recorded
        assert collector._initialized
    
    def test_event_metrics(self, custom_registry):
        """Test event-related metrics."""
        collector = PrometheusMetricsCollector(custom_registry)
        
        collector.inc_events_emitted("cortex_cycle", "cortex_runtime")
        collector.inc_events_processed("system_status", "mangle_integration")
        
        # Verify metrics are recorded
        assert collector._initialized
    
    def test_knowledge_graph_metrics(self, custom_registry):
        """Test knowledge graph metrics."""
        collector = PrometheusMetricsCollector(custom_registry)
        
        collector.set_knowledge_atoms("concept", 100)
        collector.set_knowledge_bonds("relation", 50)
        collector.inc_knowledge_operations("create_atom", "success")
        collector.observe_knowledge_operation_duration("create_bond", 0.1)
        
        # Verify metrics are recorded
        assert collector._initialized
    
    def test_optimization_metrics(self, custom_registry):
        """Test optimization-related metrics."""
        collector = PrometheusMetricsCollector(custom_registry)
        
        collector.set_optimization_policies(5)
        collector.inc_optimization_decisions("policy_1", "thompson_sampling", "arm_1")
        collector.inc_optimization_rewards("policy_1", "arm_1")
        collector.observe_optimization_reward_value("policy_1", "arm_1", 0.8)
        collector.set_optimization_arm_performance("policy_1", "arm_1", "success_rate", 0.75)
        
        # Verify metrics are recorded
        assert collector._initialized
    
    def test_metrics_output(self, custom_registry):
        """Test metrics output generation."""
        collector = PrometheusMetricsCollector(custom_registry)
        
        # Add some metrics
        collector.set_system_uptime(100.0)
        collector.inc_cortex_cycles("session_1")
        
        # Test text output
        metrics_text = collector.get_metrics_text()
        assert "super_alita_uptime_seconds" in metrics_text
        assert "super_alita_cortex_cycles_total" in metrics_text
        
        # Test content type
        content_type = collector.get_content_type()
        assert content_type == "text/plain; version=0.0.4; charset=utf-8"

class TestRedisEventBusUnit:
    """Unit tests for Redis event bus (without actual Redis)."""
    
    def test_redis_event_bus_initialization(self, custom_registry):
        """Test Redis event bus initialization."""
        bus = RedisEventBus(
            redis_url="redis://localhost:6379",
            channel_prefix="test_alita",
            event_ttl=3600
        )
        
        assert bus.redis_url == "redis://localhost:6379"
        assert bus.channel_prefix == "test_alita"
        assert bus.event_ttl == 3600
        assert not bus._running
        assert bus._redis is None
    
    def test_channel_naming(self, custom_registry):
        """Test channel name generation."""
        bus = RedisEventBus(channel_prefix="test_alita")
        
        channel = bus._get_channel_name("cortex_cycle")
        assert channel == "test_alita:events:cortex_cycle"
        
        stream = bus._get_stream_name("system_status")
        assert stream == "test_alita:stream:system_status"
    
    def test_statistics(self, custom_registry):
        """Test event bus statistics."""
        bus = RedisEventBus()
        
        stats = bus.get_statistics()
        assert stats["events_published"] == 0
        assert stats["events_received"] == 0
        assert stats["connection_errors"] == 0
        assert stats["subscribers"] == 0
        assert not stats["is_connected"]
        assert not stats["is_running"]

@pytest.mark.asyncio
class TestEventHandling:
    """Test event handling capabilities."""
    
    async def test_event_creation_and_serialization(self, custom_registry):
        """Test event creation and serialization."""
        event = create_event(
            "cortex_cycle", 
            source_plugin="cortex_runtime",
            metadata={"session_id": "test_session", "cycle_count": 5}
        )
        
        assert event.event_type == "cortex_cycle"
        assert event.source_plugin == "cortex_runtime"
        assert event.metadata["session_id"] == "test_session"
        assert hasattr(event, 'event_id')
        assert hasattr(event, 'timestamp')
    
    async def test_simple_event_bus_functionality(self, event_bus):
        """Test simple event bus functionality."""
        events_received = []
        
        def handler(event):
            events_received.append(event)
        
        # Subscribe to events
        await event_bus.subscribe("test_event", handler)
        
        # Emit an event
        test_event = create_event("test_event", source_plugin="test")
        await event_bus.emit(test_event)
        
        # Verify event was received
        assert len(events_received) == 1
        assert events_received[0].event_type == "test_event"
        
        # Unsubscribe
        await event_bus.unsubscribe("test_event", handler)
        
        # Emit another event (should not be received)
        await event_bus.emit(test_event)
        assert len(events_received) == 1  # No new events

class TestMangleComponentsIntegration:
    """Test integration between Mangle components."""
    
    def test_metrics_and_event_integration(self, custom_registry):
        """Test metrics collection triggered by events."""
        collector = PrometheusMetricsCollector(custom_registry)
        
        # Simulate processing events and updating metrics
        collector.inc_events_emitted("cortex_cycle", "cortex_runtime")
        collector.inc_events_processed("cortex_cycle", "mangle_integration")
        collector.inc_cortex_cycles("session_1", "success")
        
        # Verify metrics were recorded
        assert collector._initialized
        
        # Get metrics output
        metrics_text = collector.get_metrics_text()
        assert "super_alita_events_emitted_total" in metrics_text
        assert "super_alita_cortex_cycles_total" in metrics_text
    
    def test_metrics_collection_lifecycle(self, custom_registry):
        """Test metrics collection throughout system lifecycle."""
        collector = PrometheusMetricsCollector(custom_registry)
        
        start_time = time.time()
        
        # Initialize system
        collector.set_component_health("cortex", True)
        collector.set_component_health("redis", True)
        collector.set_component_health("grpc", True)
        
        # Simulate system activity
        collector.inc_cortex_cycles("session_1")
        collector.inc_events_processed("system_status", "test")
        collector.set_optimization_policies(3)
        
        # Update uptime
        uptime = time.time() - start_time
        collector.set_system_uptime(uptime)
        
        # Verify health
        health = collector.health_check()
        assert health["status"] == "healthy"
        assert "timestamp" in health
    
    def test_component_health_monitoring(self, custom_registry):
        """Test component health monitoring."""
        collector = PrometheusMetricsCollector(custom_registry)
        
        # Test healthy system
        collector.set_component_health("cortex", True)
        collector.set_component_health("telemetry", True)
        collector.set_component_health("knowledge", True)
        collector.set_component_health("optimization", True)
        
        health = collector.health_check()
        assert health["status"] == "healthy"
        
        # Test degraded system
        collector.set_component_health("redis", False)
        
        # Health should still be healthy for metrics collector itself
        health = collector.health_check()
        assert health["status"] == "healthy"

@pytest.mark.asyncio
class TestEndToEndMangleValidation:
    """End-to-end validation tests for Mangle integration."""
    
    async def test_complete_metrics_and_events_flow(self, event_bus, custom_registry):
        """Test complete flow of events and metrics."""
        collector = PrometheusMetricsCollector(custom_registry)
        
        # Setup event handlers that update metrics
        async def cortex_handler(event):
            collector.inc_events_processed("cortex_cycle", "mangle_integration")
            session_id = event.metadata.get("session_id", "unknown")
            collector.inc_cortex_cycles(session_id)
        
        async def system_handler(event):
            collector.inc_events_processed("system_status", "mangle_integration")
        
        # Subscribe to events
        await event_bus.subscribe("cortex_cycle", cortex_handler)
        await event_bus.subscribe("system_status", system_handler)
        
        # Emit events
        cortex_event = create_event(
            "cortex_cycle",
            source_plugin="cortex_runtime",
            metadata={"session_id": "test_session"}
        )
        
        system_event = create_event(
            "system_status",
            source_plugin="system_monitor"
        )
        
        await event_bus.emit(cortex_event)
        await event_bus.emit(system_event)
        
        # Verify metrics were updated
        metrics_text = collector.get_metrics_text()
        assert "super_alita_events_processed_total" in metrics_text
        assert "super_alita_cortex_cycles_total" in metrics_text
    
    async def test_concurrent_metrics_collection(self, custom_registry):
        """Test concurrent metrics collection."""
        collector = PrometheusMetricsCollector(custom_registry)
        
        async def collect_metrics(session_id, count):
            for i in range(count):
                collector.inc_cortex_cycles(session_id)
                collector.inc_events_processed("test_event", "test_handler")
                await asyncio.sleep(0.001)  # Small delay
        
        # Run concurrent metric collection
        tasks = [
            collect_metrics("session_1", 10),
            collect_metrics("session_2", 15),
            collect_metrics("session_3", 8)
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify metrics are consistent
        health = collector.health_check()
        assert health["status"] == "healthy"
        
        metrics_text = collector.get_metrics_text()
        assert "super_alita_cortex_cycles_total" in metrics_text
    
    async def test_error_resilience(self, custom_registry):
        """Test system resilience to errors."""
        collector = PrometheusMetricsCollector(custom_registry)
        
        # Test with various edge cases
        collector.set_system_uptime(0.0)
        collector.set_component_health("", True)  # Empty component name
        collector.observe_cortex_cycle_duration("", "", 0.0)  # Empty values
        
        # Should still work
        health = collector.health_check()
        assert health["status"] == "healthy"
    
    async def test_metrics_performance(self, custom_registry):
        """Test metrics collection performance."""
        collector = PrometheusMetricsCollector(custom_registry)
        
        start_time = time.time()
        
        # Perform many metric operations
        for i in range(1000):
            collector.inc_cortex_cycles(f"session_{i % 10}")
            collector.inc_events_processed("test_event", "test_handler")
            if i % 100 == 0:
                collector.set_system_uptime(time.time() - start_time)
        
        collection_time = time.time() - start_time
        
        # Should be fast (less than 1 second for 1000 operations)
        assert collection_time < 1.0
        
        # Verify metrics are still accessible
        health = collector.health_check()
        assert health["status"] == "healthy"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
