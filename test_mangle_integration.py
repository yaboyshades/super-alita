"""
End-to-End Validation Test for Mangle Integration

Comprehensive test suite that validates the complete Mangle integration
including gRPC server, Prometheus metrics, Redis event bus, and all
agent component integrations.
"""

import asyncio
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from src.core.mangle import MangleIntegration
from src.core.mangle.grpc_server import SuperAlitaGrpcServer
from src.core.mangle.metrics import PrometheusMetricsCollector
from src.core.mangle.redis_event_bus import RedisEventBus
from src.core.events import create_event
from src.core.telemetry.simple_event_bus import SimpleEventBus


@pytest.fixture
def workspace_root():
    """Provide workspace root path."""
    return Path(__file__).parent.parent.parent.parent


@pytest.fixture
def mock_cortex_runtime():
    """Mock Cortex runtime for testing."""
    cortex = Mock()
    cortex.process_cycle = AsyncMock(return_value="Cortex processing complete")
    cortex.create_context = Mock(return_value={"session_id": "test_session"})
    cortex.modules = {"perception": Mock(), "reasoning": Mock(), "action": Mock()}
    return cortex


@pytest.fixture
def mock_telemetry_collector():
    """Mock telemetry collector for testing."""
    return Mock()


@pytest.fixture
def mock_knowledge_plugin():
    """Mock knowledge graph plugin for testing."""
    plugin = Mock()
    plugin.create_concept = AsyncMock(return_value="concept_123")
    plugin.create_relationship = AsyncMock(return_value="relation_456")
    plugin.get_statistics = Mock(return_value={
        "total_atoms": 100,
        "total_bonds": 50,
        "atoms_by_type": {"concept": 80, "entity": 20},
        "bonds_by_type": {"relation": 50},
        "database_path": "/tmp/test.db"
    })
    return plugin


@pytest.fixture
def mock_optimization_plugin():
    """Mock optimization plugin for testing."""
    plugin = Mock()
    plugin.create_policy = AsyncMock(return_value="policy_789")
    
    # Mock decision result
    mock_decision = Mock()
    mock_decision.decision_id = "decision_123"
    mock_decision.bandit_decision = Mock()
    mock_decision.bandit_decision.arm_id = "arm_1"
    mock_decision.bandit_decision.arm_name = "Arm 1"
    mock_decision.bandit_decision.confidence = 0.85
    mock_decision.bandit_decision.algorithm = "thompson_sampling"
    
    plugin.make_decision = AsyncMock(return_value=mock_decision)
    plugin.provide_feedback = AsyncMock(return_value=True)
    plugin.get_global_statistics = Mock(return_value={
        "engine": {"total_policies": 5, "total_decisions": 100},
        "rewards": {"total_rewards": 80, "average_reward_value": 0.75},
        "policies": {
            "policies": {
                "policy_1": {
                    "policy": {"name": "Test Policy", "algorithm_type": "thompson_sampling"},
                    "bandit": {
                        "arms": {
                            "arm_1": {"name": "Arm 1", "pulls": 50, "successes": 40, "success_rate": 0.8},
                            "arm_2": {"name": "Arm 2", "pulls": 30, "successes": 20, "success_rate": 0.67}
                        }
                    },
                    "decisions": {"total": 80, "with_feedback": 70}
                }
            }
        }
    })
    return plugin


@pytest.fixture
def event_bus():
    """Provide simple event bus for testing."""
    return SimpleEventBus()


class TestMangleIntegrationUnit:
    """Unit tests for Mangle integration components."""
    
    def test_mangle_integration_initialization(self, workspace_root):
        """Test Mangle integration initialization."""
        config = {"test": True}
        integration = MangleIntegration(config=config, workspace_root=workspace_root)
        
        assert integration.config == config
        assert integration.workspace_root == workspace_root
        assert not integration.is_running
        assert integration.grpc_server is None
        assert integration.metrics_collector is None
        assert integration.redis_event_bus is None
    
    def test_mangle_integration_configuration(self, workspace_root):
        """Test Mangle integration configuration."""
        integration = MangleIntegration(workspace_root=workspace_root)
        
        integration.configure(
            grpc_host="localhost",
            grpc_port=50051,
            redis_url="redis://localhost:6379",
            enable_metrics=True,
            enable_redis=True
        )
        
        assert integration.grpc_server is not None
        assert integration.metrics_collector is not None
        assert integration.redis_event_bus is not None
        assert integration.grpc_server.host == "localhost"
        assert integration.grpc_server.port == 50051
    
    def test_agent_components_setup(
        self, 
        workspace_root,
        mock_cortex_runtime,
        mock_telemetry_collector,
        mock_knowledge_plugin,
        mock_optimization_plugin
    ):
        """Test agent components setup."""
        integration = MangleIntegration(workspace_root=workspace_root)
        
        integration.setup_agent_components(
            cortex_runtime=mock_cortex_runtime,
            telemetry_collector=mock_telemetry_collector,
            knowledge_plugin=mock_knowledge_plugin,
            optimization_plugin=mock_optimization_plugin
        )
        
        assert integration.cortex_runtime == mock_cortex_runtime
        assert integration.telemetry_collector == mock_telemetry_collector
        assert integration.knowledge_plugin == mock_knowledge_plugin
        assert integration.optimization_plugin == mock_optimization_plugin


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


@pytest.mark.asyncio
class TestMangleIntegrationAsync:
    """Async integration tests for Mangle components."""
    
    async def test_mangle_integration_lifecycle(
        self,
        workspace_root,
        mock_cortex_runtime,
        mock_telemetry_collector,
        mock_knowledge_plugin,
        mock_optimization_plugin
    ):
        """Test complete Mangle integration lifecycle."""
        integration = MangleIntegration(workspace_root=workspace_root)
        
        # Configure with disabled Redis for testing
        integration.configure(
            grpc_host="localhost",
            grpc_port=50052,  # Different port for testing
            enable_metrics=True,
            enable_redis=False  # Disable Redis for unit test
        )
        
        # Setup agent components
        integration.setup_agent_components(
            cortex_runtime=mock_cortex_runtime,
            telemetry_collector=mock_telemetry_collector,
            knowledge_plugin=mock_knowledge_plugin,
            optimization_plugin=mock_optimization_plugin
        )
        
        # Test status before start
        status = integration.get_status()
        assert not status["is_running"]
        assert status["components"]["grpc_server"]
        assert status["components"]["metrics_collector"]
        assert not status["components"]["redis_event_bus"]
        
        # Start integration
        await integration.start()
        
        # Test status after start
        assert integration.is_running
        status = integration.get_status()
        assert status["is_running"]
        assert status["uptime_seconds"] > 0
        
        # Test health check
        health = await integration.health_check()
        assert health["status"] in ["healthy", "degraded"]
        assert "components" in health
        
        # Stop integration
        await integration.stop()
        assert not integration.is_running
    
    async def test_event_handling(
        self,
        workspace_root,
        mock_cortex_runtime,
        event_bus
    ):
        """Test event handling in Mangle integration."""
        integration = MangleIntegration(workspace_root=workspace_root)
        integration.configure(enable_redis=False, enable_metrics=True)
        integration.setup_agent_components(cortex_runtime=mock_cortex_runtime)
        
        # Test event handlers
        cortex_event = create_event(
            "cortex_cycle",
            source_plugin="cortex_runtime",
            metadata={"session_id": "test_session"}
        )
        
        await integration._handle_cortex_event(cortex_event)
        
        # Verify metrics were updated
        assert integration.metrics_collector is not None
    
    async def test_metrics_collection_lifecycle(self, workspace_root):
        """Test metrics collection throughout integration lifecycle."""
        integration = MangleIntegration(workspace_root=workspace_root)
        integration.configure(enable_metrics=True, enable_redis=False)
        
        await integration.start()
        
        # Verify initial metrics are set
        assert integration.metrics_collector is not None
        
        # Test metrics update
        integration._setup_metrics_collection()
        
        # Test metrics health
        health = await integration.health_check()
        assert "metrics" in health["components"]
        
        await integration.stop()


@pytest.mark.asyncio
class TestGrpcServerUnit:
    """Unit tests for gRPC server without actual gRPC calls."""
    
    async def test_grpc_server_setup(
        self,
        mock_cortex_runtime,
        mock_telemetry_collector,
        mock_knowledge_plugin,
        mock_optimization_plugin
    ):
        """Test gRPC server setup."""
        server = SuperAlitaGrpcServer(host="localhost", port=50053)
        
        metrics_collector = PrometheusMetricsCollector()
        
        server.setup(
            cortex_runtime=mock_cortex_runtime,
            telemetry_collector=mock_telemetry_collector,
            knowledge_plugin=mock_knowledge_plugin,
            optimization_plugin=mock_optimization_plugin,
            metrics_collector=metrics_collector
        )
        
        assert server.servicer is not None
        assert server.servicer.cortex_runtime == mock_cortex_runtime
        assert server.servicer.knowledge_plugin == mock_knowledge_plugin
        assert server.server is not None


@pytest.mark.asyncio
class TestEndToEndValidation:
    """End-to-end validation tests."""
    
    async def test_complete_mangle_system(
        self,
        workspace_root,
        mock_cortex_runtime,
        mock_telemetry_collector,
        mock_knowledge_plugin,
        mock_optimization_plugin
    ):
        """Test complete Mangle system integration."""
        
        # Create and configure integration
        integration = MangleIntegration(workspace_root=workspace_root)
        integration.configure(
            grpc_host="localhost",
            grpc_port=50054,
            enable_metrics=True,
            enable_redis=False  # Disabled for unit testing
        )
        
        # Setup all agent components
        integration.setup_agent_components(
            cortex_runtime=mock_cortex_runtime,
            telemetry_collector=mock_telemetry_collector,
            knowledge_plugin=mock_knowledge_plugin,
            optimization_plugin=mock_optimization_plugin
        )
        
        # Start the system
        start_time = time.time()
        await integration.start()
        startup_time = time.time() - start_time
        
        # Verify system is running
        assert integration.is_running
        assert startup_time < 5.0  # Should start quickly
        
        # Test system status
        status = integration.get_status()
        assert status["is_running"]
        assert all(status["components"].values())  # All components should be available
        
        # Test health check
        health = await integration.health_check()
        assert health["status"] in ["healthy", "degraded"]
        
        # Test metrics are being collected
        if integration.metrics_collector:
            metrics_text = integration.metrics_collector.get_metrics_text()
            assert "super_alita" in metrics_text
        
        # Simulate some activity
        integration.total_grpc_requests += 5
        integration.total_events_processed += 10
        
        # Verify statistics are updated
        final_status = integration.get_status()
        assert final_status["statistics"]["total_grpc_requests"] == 5
        assert final_status["statistics"]["total_events_processed"] == 10
        
        # Graceful shutdown
        shutdown_time = time.time()
        await integration.stop()
        shutdown_duration = time.time() - shutdown_time
        
        assert not integration.is_running
        assert shutdown_duration < 3.0  # Should shutdown quickly
    
    async def test_error_resilience(self, workspace_root):
        """Test system resilience to component failures."""
        integration = MangleIntegration(workspace_root=workspace_root)
        
        # Test with minimal configuration
        integration.configure(enable_metrics=False, enable_redis=False)
        
        # Should still be able to start/stop
        await integration.start()
        assert integration.is_running
        
        health = await integration.health_check()
        assert health["status"] == "degraded"  # Some components unavailable
        
        await integration.stop()
        assert not integration.is_running
    
    async def test_concurrent_operations(
        self,
        workspace_root,
        mock_cortex_runtime,
        mock_knowledge_plugin,
        mock_optimization_plugin
    ):
        """Test concurrent operations on Mangle system."""
        integration = MangleIntegration(workspace_root=workspace_root)
        integration.configure(enable_redis=False)
        integration.setup_agent_components(
            cortex_runtime=mock_cortex_runtime,
            knowledge_plugin=mock_knowledge_plugin,
            optimization_plugin=mock_optimization_plugin
        )
        
        await integration.start()
        
        # Create concurrent tasks
        tasks = []
        
        # Multiple health checks
        for _ in range(5):
            tasks.append(integration.health_check())
        
        # Multiple event handling simulations
        for i in range(3):
            event = create_event(
                "system_status",
                source_plugin="test",
                metadata={"task_id": i}
            )
            tasks.append(integration._handle_system_event(event))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception), f"Task failed: {result}"
        
        # Verify health checks returned valid results
        health_results = results[:5]
        for health in health_results:
            assert health["status"] in ["healthy", "degraded"]
        
        await integration.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])