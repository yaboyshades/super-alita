"""
Comprehensive test suite for Performance Monitoring and Rule Automation System.

Tests all components including performance monitoring, constitutional compliance,
telemetry collection, dashboard interface, and CI quality gates.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from src.performance_monitoring.ci.quality_gates import (
    ConstitutionalGate,
    QualityGatePipeline,
)
from src.performance_monitoring.core.constitutional_engine import (
    ConstitutionalArticle,
    ConstitutionalEngine,
)

# Import system components
from src.performance_monitoring.core.performance_monitor import PerformanceMonitor
from src.performance_monitoring.core.telemetry_bridge import TelemetryBridge
from src.performance_monitoring.dashboard.dashboard_interface import DashboardInterface
from src.performance_monitoring.integration import PerformanceMonitoringSystem


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor component."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a performance monitor instance for testing."""
        return PerformanceMonitor(
            metrics_retention_hours=1,
            collection_interval_seconds=1,
            enable_real_time_alerts=True
        )
    
    @pytest.mark.asyncio
    async def test_monitor_lifecycle(self, performance_monitor):
        """Test performance monitor start/stop lifecycle."""
        assert not performance_monitor._monitoring_active
        
        await performance_monitor.start_monitoring()
        assert performance_monitor._monitoring_active
        
        await performance_monitor.stop_monitoring()
        assert not performance_monitor._monitoring_active
    
    def test_track_extension_interaction(self, performance_monitor):
        """Test extension interaction tracking."""
        interaction = performance_monitor.track_extension_interaction(
            "test_extension", "test_function"
        )
        
        assert interaction.extension_id == "test_extension"
        assert interaction.function_name == "test_function"
        assert interaction.start_time is not None
        assert interaction.end_time is None
        
        # Complete the interaction
        performance_monitor.complete_interaction(interaction, success=True)
        
        assert interaction.end_time is not None
        assert interaction.duration_ms is not None
        assert interaction.success is True
    
    def test_add_metric(self, performance_monitor):
        """Test metric addition and aggregation."""
        performance_monitor.add_metric("test_metric", 100.0, "test_category")
        
        summary = performance_monitor.get_performance_summary()
        assert summary["metrics_count"] >= 1
        
        # Check metric aggregates
        aggregates = summary["metric_aggregates"]
        assert "test_category_test_metric" in aggregates
    
    def test_performance_summary(self, performance_monitor):
        """Test performance summary generation."""
        # Add some test data
        interaction = performance_monitor.track_extension_interaction("ext1", "func1")
        performance_monitor.complete_interaction(interaction, success=True)
        performance_monitor.add_metric("response_time", 500.0, "system")
        
        summary = performance_monitor.get_performance_summary()
        
        assert "timestamp" in summary
        assert "interactions_count" in summary
        assert "metrics_count" in summary
        assert "average_response_time_ms" in summary
        assert "success_rate" in summary
    
    def test_extension_statistics(self, performance_monitor):
        """Test extension-specific statistics."""
        # Add test interactions
        interaction1 = performance_monitor.track_extension_interaction("ext1", "func1")
        performance_monitor.complete_interaction(interaction1, success=True)
        
        interaction2 = performance_monitor.track_extension_interaction("ext1", "func2")
        performance_monitor.complete_interaction(interaction2, success=False, error_message="Test error")
        
        stats = performance_monitor.get_extension_statistics("ext1")
        
        assert stats["extension_id"] == "ext1"
        assert stats["total_interactions"] == 2
        assert stats["success_rate"] == 0.5  # 1 success out of 2


class TestTelemetryBridge:
    """Test suite for TelemetryBridge component."""
    
    @pytest.fixture
    def telemetry_bridge(self):
        """Create a telemetry bridge instance for testing."""
        return TelemetryBridge(
            buffer_size=100,
            flush_interval_seconds=1
        )
    
    @pytest.mark.asyncio
    async def test_telemetry_lifecycle(self, telemetry_bridge):
        """Test telemetry bridge start/stop lifecycle."""
        assert not telemetry_bridge._active
        
        await telemetry_bridge.start()
        assert telemetry_bridge._active
        
        await telemetry_bridge.stop()
        assert not telemetry_bridge._active
    
    def test_track_host_api_call(self, telemetry_bridge):
        """Test host API call tracking."""
        call = telemetry_bridge.track_host_api_call(
            "test_function",
            {"param1": "value1"},
            constitutional_impact="test_impact"
        )
        
        assert call.function_name == "test_function"
        assert call.constitutional_impact == "test_impact"
        assert call.start_time is not None
        
        # Complete the call
        telemetry_bridge.complete_host_api_call(call, result="success")
        
        assert call.end_time is not None
        assert call.result == "success"
        assert call.error is None
    
    def test_track_wasm_operation(self, telemetry_bridge):
        """Test WASM operation tracking."""
        operation = telemetry_bridge.track_wasm_operation(
            "test_component", "predict", 1024
        )
        
        assert operation.component_name == "test_component"
        assert operation.operation_type == "predict"
        assert operation.input_size_bytes == 1024
        
        # Complete the operation
        telemetry_bridge.complete_wasm_operation(
            operation, result="prediction", memory_usage_bytes=2048
        )
        
        assert operation.execution_result == "prediction"
        assert operation.memory_usage_bytes == 2048
    
    def test_telemetry_summary(self, telemetry_bridge):
        """Test telemetry summary generation."""
        # Add test data
        call = telemetry_bridge.track_host_api_call("func1", {})
        telemetry_bridge.complete_host_api_call(call)
        
        operation = telemetry_bridge.track_wasm_operation("comp1", "analyze", 512)
        telemetry_bridge.complete_wasm_operation(operation)
        
        summary = telemetry_bridge.get_telemetry_summary()
        
        assert "total_events" in summary
        assert "host_api_statistics" in summary
        assert "wasm_statistics" in summary
        assert "buffer_utilization" in summary


class TestConstitutionalEngine:
    """Test suite for ConstitutionalEngine component."""
    
    @pytest.fixture
    def constitutional_engine(self):
        """Create a constitutional engine instance for testing."""
        return ConstitutionalEngine(compliance_threshold=0.75)
    
    @pytest.mark.asyncio
    async def test_validate_compliance(self, constitutional_engine):
        """Test constitutional compliance validation."""
        test_data = {
            "type": "code_change",
            "file_path": "test.py",
            "changes": ["def test_function():", "    return True"]
        }
        
        result = await constitutional_engine.validate_compliance(test_data)
        
        assert result.overall_score >= 0.0
        assert result.overall_score <= 1.0
        assert result.threshold == 0.75
        assert len(result.article_scores) == len(ConstitutionalArticle)
    
    @pytest.mark.asyncio
    async def test_validate_code_change(self, constitutional_engine):
        """Test code change validation."""
        result = await constitutional_engine.validate_code_change(
            "src/test.py",
            ["def new_function():", "    pass"],
            {"author": "test_author"}
        )
        
        assert result.overall_score >= 0.0
        assert isinstance(result.violations, list)
        assert result.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_validate_commit(self, constitutional_engine):
        """Test commit validation."""
        result = await constitutional_engine.validate_commit(
            "feat: add new feature",
            ["src/feature.py"],
            "diff content"
        )
        
        assert result.overall_score >= 0.0
        assert result.threshold == 0.75
    
    def test_compliance_trend(self, constitutional_engine):
        """Test compliance trend analysis."""
        # Initially no data
        trend = constitutional_engine.get_compliance_trend()
        assert trend["status"] == "no_data"
        
        # Add some compliance history (would normally be added by validate_compliance)
        # This is a simplified test
        constitutional_engine.compliance_history = []  # Reset for clean test
        trend = constitutional_engine.get_compliance_trend()
        assert trend["status"] == "no_data"


class TestDashboardInterface:
    """Test suite for DashboardInterface component."""
    
    @pytest.fixture
    def dashboard(self):
        """Create a dashboard interface instance for testing."""
        return DashboardInterface(update_interval_seconds=1)
    
    @pytest.mark.asyncio
    async def test_dashboard_lifecycle(self, dashboard):
        """Test dashboard start/stop lifecycle."""
        assert not dashboard._update_active
        
        await dashboard.start_dashboard()
        assert dashboard._update_active
        
        await dashboard.stop_dashboard()
        assert not dashboard._update_active
    
    def test_update_performance_data(self, dashboard):
        """Test performance data updates."""
        # Mock performance monitor
        mock_monitor = Mock()
        mock_monitor.get_performance_summary.return_value = {
            "average_response_time_ms": 500.0,
            "success_rate": 0.95,
            "interactions_count": 10,
            "constitutional_compliance": {"status": "compliant", "score": 0.8}
        }
        
        dashboard.update_performance_data(mock_monitor)
        
        # Check that metrics were updated
        assert "response_time" in dashboard.metrics
        assert "success_rate" in dashboard.metrics
        assert dashboard.metrics["response_time"].value == 500.0
    
    def test_dashboard_state(self, dashboard):
        """Test dashboard state retrieval."""
        state = dashboard.get_dashboard_state()
        
        assert "timestamp" in state
        assert "metrics" in state
        assert "alerts" in state
        assert "status" in state
    
    def test_constitutional_dashboard(self, dashboard):
        """Test constitutional compliance dashboard."""
        dashboard._update_metric("constitutional_score", 0.8, "score", "success")
        
        const_dashboard = dashboard.get_constitutional_dashboard()
        
        assert "constitutional_metrics" in const_dashboard
        assert "constitutional_alerts" in const_dashboard
        assert "compliance_indicators" in const_dashboard


class TestQualityGates:
    """Test suite for Quality Gates components."""
    
    @pytest.fixture
    def quality_pipeline(self):
        """Create a quality gate pipeline for testing."""
        return QualityGatePipeline()
    
    @pytest.fixture
    def constitutional_gate(self):
        """Create a constitutional gate for testing."""
        mock_engine = Mock()
        mock_engine.validate_compliance = AsyncMock()
        return ConstitutionalGate(mock_engine, threshold=0.75)
    
    @pytest.mark.asyncio
    async def test_constitutional_gate_validation(self, constitutional_gate):
        """Test constitutional gate validation."""
        # Mock compliance score
        from unittest.mock import MagicMock
        mock_score = MagicMock()
        mock_score.is_compliant = True
        mock_score.overall_score = 0.8
        mock_score.violations = []
        mock_score.to_dict.return_value = {"overall_score": 0.8}
        
        constitutional_gate.constitutional_engine.validate_compliance.return_value = mock_score
        
        context = {"type": "test", "data": "test_data"}
        result = await constitutional_gate.validate(context)
        
        assert result.gate_name == "constitutional_compliance"
        assert result.passed is True
        assert result.score == 0.8
    
    @pytest.mark.asyncio
    async def test_quality_pipeline_execution(self, quality_pipeline, constitutional_gate):
        """Test quality pipeline execution."""
        quality_pipeline.add_gate(constitutional_gate)
        
        # Mock the gate validation
        constitutional_gate.constitutional_engine.validate_compliance.return_value = Mock(
            is_compliant=True,
            overall_score=0.8,
            violations=[],
            to_dict=lambda: {"overall_score": 0.8}
        )
        
        context = {"type": "test"}
        result = await quality_pipeline.execute_pipeline(context)
        
        assert "overall_passed" in result
        assert "gate_results" in result
        assert "summary" in result
        assert len(result["gate_results"]) == 1


class TestSystemIntegration:
    """Test suite for system integration."""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create a monitoring system for testing."""
        config = {
            "metrics_retention_hours": 1,
            "collection_interval_seconds": 1,
            "telemetry_buffer_size": 100,
            "dashboard_update_interval": 1
        }
        return PerformanceMonitoringSystem(config=config)
    
    @pytest.mark.asyncio
    async def test_system_lifecycle(self, monitoring_system):
        """Test complete system lifecycle."""
        assert not monitoring_system._running
        
        await monitoring_system.start_system()
        assert monitoring_system._running
        
        await monitoring_system.stop_system()
        assert not monitoring_system._running
    
    @pytest.mark.asyncio
    async def test_validate_code_change_integration(self, monitoring_system):
        """Test code change validation integration."""
        await monitoring_system.start_system()
        
        try:
            result = await monitoring_system.validate_code_change(
                "test.py",
                ["def test():", "    pass"]
            )
            
            assert "compliance_score" in result
            assert "validation_timestamp" in result
            
        finally:
            await monitoring_system.stop_system()
    
    def test_system_status(self, monitoring_system):
        """Test system status reporting."""
        status = monitoring_system.get_system_status()
        
        assert "system_running" in status
        assert "component_status" in status
        assert "dashboard_state" in status
    
    @pytest.mark.asyncio
    async def test_health_check(self, monitoring_system):
        """Test system health check."""
        health = await monitoring_system.run_health_check()
        
        assert "overall_health" in health
        assert "component_health" in health
        assert "timestamp" in health


# Integration tests
class TestEndToEndScenarios:
    """End-to-end test scenarios."""
    
    @pytest.mark.asyncio
    async def test_commit_validation_workflow(self):
        """Test complete commit validation workflow."""
        system = PerformanceMonitoringSystem()
        await system.start_system()
        
        try:
            # Simulate a commit validation
            result = await system.validate_commit(
                "feat: add constitutional compliance",
                ["src/compliance.py", "tests/test_compliance.py"],
                "diff content here"
            )
            
            assert "overall_passed" in result
            assert "gate_results" in result
            
            # Check that telemetry was recorded
            telemetry_summary = system.telemetry_bridge.get_telemetry_summary()
            assert telemetry_summary["total_events"] > 0
            
        finally:
            await system.stop_system()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_workflow(self):
        """Test performance monitoring workflow."""
        system = PerformanceMonitoringSystem()
        await system.start_system()
        
        try:
            # Simulate some extension interactions
            interaction = system.performance_monitor.track_extension_interaction(
                "test_extension", "test_function"
            )
            
            # Simulate some processing time
            await asyncio.sleep(0.1)
            
            system.performance_monitor.complete_interaction(interaction, success=True)
            
            # Check performance summary
            summary = system.performance_monitor.get_performance_summary()
            assert summary["interactions_count"] > 0
            
            # Check dashboard was updated
            dashboard_state = system.dashboard.get_dashboard_state()
            assert len(dashboard_state["metrics"]) > 0
            
        finally:
            await system.stop_system()


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])