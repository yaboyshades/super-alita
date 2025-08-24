"""
Direct Mangle Component Tests

Tests individual Mangle components directly without importing
the full module that has gRPC dependencies.
"""

import asyncio
import pytest
import time
from pathlib import Path

# Import components directly to avoid gRPC issues
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from core.mangle.metrics import PrometheusMetricsCollector
from core.mangle.redis_event_bus import RedisEventBus
from core.events import create_event


class TestPrometheusMetrics:
    """Test Prometheus metrics collection."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = PrometheusMetricsCollector()
        assert not collector._initialized
        
        # Test lazy initialization
        collector.set_system_uptime(123.45)
        assert collector._initialized
        assert len(collector._metrics) > 0
    
    def test_system_metrics(self):
        """Test system metrics collection."""
        collector = PrometheusMetricsCollector()
        
        collector.set_system_uptime(100.0)
        collector.set_component_health("cortex", True)
        collector.set_component_health("redis", False)
        
        # Test health check
        health = collector.health_check()
        assert health["status"] == "healthy"
        assert "metrics_count" in health
    
    def test_cortex_metrics(self):
        """Test Cortex-specific metrics."""
        collector = PrometheusMetricsCollector()
        
        collector.inc_cortex_cycles("session_1", "success")
        collector.observe_cortex_cycle_duration("session_1", "perception", 0.5)
        collector.set_cortex_active_sessions(3)
        
        # Verify metrics are recorded
        assert collector._initialized
    
    def test_event_metrics(self):
        """Test event-related metrics."""
        collector = PrometheusMetricsCollector()
        
        collector.inc_events_emitted("cortex_cycle", "cortex_runtime")
        collector.inc_events_processed("system_status", "mangle_integration")
        
        # Verify metrics are recorded
        assert collector._initialized
    
    def test_knowledge_graph_metrics(self):
        """Test knowledge graph metrics."""
        collector = PrometheusMetricsCollector()
        
        collector.set_knowledge_atoms("concept", 100)
        collector.set_knowledge_bonds("relation", 50)
        collector.inc_knowledge_operations("create_atom", "success")
        collector.observe_knowledge_operation_duration("create_bond", 0.1)
        
        # Verify metrics are recorded
        assert collector._initialized
    
    def test_optimization_metrics(self):
        """Test optimization-related metrics."""
        collector = PrometheusMetricsCollector()
        
        collector.set_optimization_policies(5)
        collector.inc_optimization_decisions("policy_1", "thompson_sampling", "arm_1")
        collector.inc_optimization_rewards("policy_1", "arm_1")
        collector.observe_optimization_reward_value("policy_1", "arm_1", 0.8)
        collector.set_optimization_arm_performance("policy_1", "arm_1", "success_rate", 0.75)
        
        # Verify metrics are recorded
        assert collector._initialized
    
    def test_metrics_output(self):
        """Test metrics output generation."""
        collector = PrometheusMetricsCollector()
        
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
    
    def test_redis_event_bus_initialization(self):
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
    
    def test_channel_naming(self):
        """Test channel name generation."""
        bus = RedisEventBus(channel_prefix="test_alita")
        
        channel = bus._get_channel_name("cortex_cycle")
        assert channel == "test_alita:events:cortex_cycle"
        
        stream = bus._get_stream_name("system_status")
        assert stream == "test_alita:stream:system_status"
    
    def test_statistics(self):
        """Test event bus statistics."""
        bus = RedisEventBus()
        
        stats = bus.get_statistics()
        assert stats["events_published"] == 0
        assert stats["events_received"] == 0
        assert stats["connection_errors"] == 0
        assert stats["subscribers"] == 0
        assert not stats["is_connected"]
        assert not stats["is_running"]


class TestMangleComponentsIntegration:
    """Test integration between working Mangle components."""
    
    def test_metrics_collection_lifecycle(self):
        """Test metrics collection throughout system lifecycle."""
        collector = PrometheusMetricsCollector()
        
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
    
    def test_component_health_monitoring(self):
        """Test component health monitoring."""
        collector = PrometheusMetricsCollector()
        
        # Test healthy system
        collector.set_component_health("cortex", True)
        collector.set_component_health("telemetry", True)
        collector.set_component_health("knowledge", True)
        collector.set_component_health("optimization", True)
        
        health = collector.health_check()
        assert health["status"] == "healthy"
        
        # Test with mixed health
        collector.set_component_health("redis", False)
        
        # Health should still be healthy for metrics collector itself
        health = collector.health_check()
        assert health["status"] == "healthy"
    
    def test_metrics_performance(self):
        """Test metrics collection performance."""
        collector = PrometheusMetricsCollector()
        
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
    
    def test_comprehensive_metrics_scenario(self):
        """Test comprehensive metrics collection scenario."""
        collector = PrometheusMetricsCollector()
        
        # Simulate a complete system run
        start_time = time.time()
        
        # System startup
        collector.set_component_health("cortex", True)
        collector.set_component_health("telemetry", True)
        collector.set_component_health("knowledge", True)
        collector.set_component_health("optimization", True)
        
        # Simulate agent activity
        sessions = ["session_1", "session_2", "session_3"]
        
        for i in range(50):
            session = sessions[i % len(sessions)]
            
            # Cortex processing
            collector.inc_cortex_cycles(session, "success")
            collector.observe_cortex_cycle_duration(session, "perception", 0.1)
            collector.observe_cortex_cycle_duration(session, "reasoning", 0.3)
            collector.observe_cortex_cycle_duration(session, "action", 0.2)
            
            # Event processing
            collector.inc_events_emitted("cortex_cycle", "cortex_runtime")
            collector.inc_events_processed("cortex_cycle", "knowledge_handler")
            
            # Knowledge graph operations
            if i % 5 == 0:
                collector.inc_knowledge_operations("create_atom", "success")
                collector.observe_knowledge_operation_duration("create_atom", 0.05)
            
            if i % 7 == 0:
                collector.inc_knowledge_operations("create_bond", "success")
                collector.observe_knowledge_operation_duration("create_bond", 0.03)
            
            # Optimization decisions
            if i % 3 == 0:
                collector.inc_optimization_decisions("policy_1", "thompson_sampling", "arm_1")
                collector.inc_optimization_rewards("policy_1", "arm_1")
                collector.observe_optimization_reward_value("policy_1", "arm_1", 0.8)
        
        # Update system metrics
        collector.set_cortex_active_sessions(len(sessions))
        collector.set_knowledge_atoms("concept", 100)
        collector.set_knowledge_bonds("relation", 75)
        collector.set_optimization_policies(1)
        
        final_uptime = time.time() - start_time
        collector.set_system_uptime(final_uptime)
        
        # Verify final state
        health = collector.health_check()
        assert health["status"] == "healthy"
        
        # Verify metrics output contains expected data
        metrics_text = collector.get_metrics_text()
        assert "super_alita_cortex_cycles_total" in metrics_text
        assert "super_alita_events_emitted_total" in metrics_text
        assert "super_alita_knowledge_operations_total" in metrics_text
        assert "super_alita_optimization_decisions_total" in metrics_text
        assert "super_alita_uptime_seconds" in metrics_text
        
        print(f"✅ Comprehensive metrics scenario completed in {final_uptime:.3f}s")
        print(f"✅ Metrics output length: {len(metrics_text)} characters")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])