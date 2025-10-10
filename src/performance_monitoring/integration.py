"""
Performance Monitoring and Rule Automation Integration

Main integration module that orchestrates all components of the performance
monitoring and constitutional compliance system.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ci.quality_gates import (
    ConstitutionalGate,
    PerformanceGate,
    QualityGatePipeline,
    SecurityGate,
)
from .core.constitutional_engine import ConstitutionalEngine
from .core.performance_monitor import PerformanceMonitor
from .core.telemetry_bridge import TelemetryBridge
from .dashboard.dashboard_interface import DashboardInterface

logger = logging.getLogger(__name__)


class PerformanceMonitoringSystem:
    """
    Main system integration for performance monitoring and constitutional compliance.
    
    Implements Article IV: Integration-First through comprehensive system integration.
    Implements Article I: Library-First through standard monitoring patterns.
    """
    
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        constitutional_threshold: float = 0.75
    ):
        self.config = config or self._get_default_config()
        self.constitutional_threshold = constitutional_threshold
        
        # Initialize core components
        self.performance_monitor = PerformanceMonitor(
            metrics_retention_hours=self.config.get("metrics_retention_hours", 24),
            collection_interval_seconds=self.config.get("collection_interval_seconds", 10)
        )
        
        self.telemetry_bridge = TelemetryBridge(
            buffer_size=self.config.get("telemetry_buffer_size", 10000),
            flush_interval_seconds=self.config.get("telemetry_flush_interval", 30)
        )
        
        self.constitutional_engine = ConstitutionalEngine(
            compliance_threshold=constitutional_threshold
        )
        
        self.dashboard = DashboardInterface(
            update_interval_seconds=self.config.get("dashboard_update_interval", 5)
        )
        
        # Initialize CI/CD pipeline
        self.quality_pipeline = QualityGatePipeline()
        self._setup_quality_gates()
        
        # System state
        self._running = False
        self._background_tasks = []
        
        logger.info("Performance Monitoring System initialized")

    async def start_system(self) -> None:
        """Start the complete performance monitoring system."""
        if self._running:
            logger.warning("System already running")
            return
        
        logger.info("Starting Performance Monitoring System...")
        
        try:
            # Start core components
            await self.performance_monitor.start_monitoring()
            await self.telemetry_bridge.start()
            await self.dashboard.start_dashboard()
            
            # Set up component integration
            self._setup_component_integration()
            
            # Start background integration tasks
            self._background_tasks = [
                asyncio.create_task(self._integration_loop()),
                asyncio.create_task(self._health_check_loop())
            ]
            
            self._running = True
            logger.info("Performance Monitoring System started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop_system()
            raise

    async def stop_system(self) -> None:
        """Stop the performance monitoring system."""
        if not self._running:
            return
        
        logger.info("Stopping Performance Monitoring System...")
        
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Stop core components
        try:
            await self.dashboard.stop_dashboard()
            await self.telemetry_bridge.stop()
            await self.performance_monitor.stop_monitoring()
        except Exception as e:
            logger.error(f"Error stopping components: {e}")
        
        logger.info("Performance Monitoring System stopped")

    async def validate_commit(
        self,
        commit_message: str,
        changed_files: list[str],
        diff_data: str
    ) -> dict[str, Any]:
        """Validate a commit against all quality gates."""
        context = {
            "type": "commit",
            "commit_message": commit_message,
            "changed_files": changed_files,
            "diff_data": diff_data,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        logger.info(f"Validating commit with {len(changed_files)} changed files")
        
        # Execute quality gate pipeline
        result = await self.quality_pipeline.execute_pipeline(context)
        
        # Track validation in telemetry
        self.telemetry_bridge._add_event(
            source="ci_pipeline",
            event_type="commit_validation",
            data={
                "passed": result["overall_passed"],
                "execution_time_ms": result["execution_time_ms"],
                "total_violations": result["summary"]["total_violations"]
            }
        )
        
        return result

    async def validate_code_change(
        self,
        file_path: str,
        changes: list[str],
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Validate code changes for constitutional compliance."""
        # Track the validation interaction
        interaction = self.performance_monitor.track_extension_interaction(
            "constitutional_validator", "validate_code_change"
        )
        
        try:
            compliance_score = await self.constitutional_engine.validate_code_change(
                file_path, changes, metadata
            )
            
            self.performance_monitor.complete_interaction(interaction, success=True)
            
            return {
                "compliance_score": compliance_score.to_dict(),
                "validation_timestamp": datetime.now(UTC).isoformat()
            }
            
        except Exception as e:
            self.performance_monitor.complete_interaction(
                interaction, success=False, error_message=str(e)
            )
            raise

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_running": self._running,
            "timestamp": datetime.now(UTC).isoformat(),
            "component_status": {
                "performance_monitor": self._running,
                "telemetry_bridge": self.telemetry_bridge._active,
                "constitutional_engine": True,
                "dashboard": self.dashboard._update_active,
                "quality_pipeline": len(self.quality_pipeline.gates) > 0
            },
            "dashboard_state": self.dashboard.get_dashboard_state(),
            "performance_summary": self.performance_monitor.get_performance_summary(),
            "telemetry_summary": self.telemetry_bridge.get_telemetry_summary(),
            "constitutional_compliance": self.constitutional_engine.get_compliance_trend(),
            "quality_gate_statistics": self.quality_pipeline.get_gate_statistics()
        }

    def get_constitutional_dashboard(self) -> dict[str, Any]:
        """Get constitutional compliance focused dashboard."""
        return {
            "constitutional_dashboard": self.dashboard.get_constitutional_dashboard(),
            "compliance_trend": self.constitutional_engine.get_compliance_trend(),
            "constitutional_telemetry": self.telemetry_bridge.get_constitutional_compliance_data(),
            "recent_validations": self.quality_pipeline.get_execution_history(5)
        }

    async def run_health_check(self) -> dict[str, Any]:
        """Run comprehensive system health check."""
        health_status = {
            "overall_health": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "component_health": {},
            "issues": []
        }
        
        # Check performance monitor health
        try:
            perf_summary = self.performance_monitor.get_performance_summary()
            health_status["component_health"]["performance_monitor"] = {
                "status": "healthy",
                "metrics_count": perf_summary.get("metrics_count", 0),
                "interactions_count": perf_summary.get("interactions_count", 0)
            }
        except Exception as e:
            health_status["component_health"]["performance_monitor"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["issues"].append(f"Performance monitor: {str(e)}")
        
        # Check telemetry bridge health
        try:
            telem_summary = self.telemetry_bridge.get_telemetry_summary()
            health_status["component_health"]["telemetry_bridge"] = {
                "status": "healthy",
                "total_events": telem_summary.get("total_events", 0),
                "event_rate": telem_summary.get("event_rate_per_minute", 0)
            }
        except Exception as e:
            health_status["component_health"]["telemetry_bridge"] = {
                "status": "unhealthy", 
                "error": str(e)
            }
            health_status["issues"].append(f"Telemetry bridge: {str(e)}")
        
        # Check constitutional engine health
        try:
            const_trend = self.constitutional_engine.get_compliance_trend()
            health_status["component_health"]["constitutional_engine"] = {
                "status": "healthy",
                "compliance_trend": const_trend
            }
        except Exception as e:
            health_status["component_health"]["constitutional_engine"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["issues"].append(f"Constitutional engine: {str(e)}")
        
        # Determine overall health
        if health_status["issues"]:
            health_status["overall_health"] = "degraded" if len(health_status["issues"]) < 2 else "unhealthy"
        
        return health_status

    def _setup_quality_gates(self) -> None:
        """Set up quality gates for the CI pipeline."""
        # Constitutional compliance gate
        constitutional_gate = ConstitutionalGate(
            self.constitutional_engine,
            threshold=self.constitutional_threshold
        )
        self.quality_pipeline.add_gate(constitutional_gate)
        
        # Performance gate
        performance_thresholds = self.config.get("performance_thresholds", {
            "response_time_ms": 1000,
            "success_rate": 0.95
        })
        performance_gate = PerformanceGate(
            self.performance_monitor,
            performance_thresholds
        )
        self.quality_pipeline.add_gate(performance_gate)
        
        # Security gate
        security_config = self.config.get("security_config", {
            "check_dependencies": True,
            "check_code": True
        })
        security_gate = SecurityGate(security_config)
        self.quality_pipeline.add_gate(security_gate)

    def _setup_component_integration(self) -> None:
        """Set up integration between components."""
        # Connect telemetry bridge to performance monitor
        self.telemetry_bridge.add_event_handler(self._handle_telemetry_event)
        
        # Connect performance monitor alerts to dashboard
        self.performance_monitor.add_alert_handler(self._handle_performance_alert)

    async def _integration_loop(self) -> None:
        """Background integration loop."""
        while self._running:
            try:
                # Update dashboard with current data
                self.dashboard.update_performance_data(self.performance_monitor)
                self.dashboard.update_constitutional_data(self.constitutional_engine)
                self.dashboard.update_telemetry_data(self.telemetry_bridge)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Integration loop error: {e}")
                await asyncio.sleep(30)

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                health_status = await self.run_health_check()
                
                # Log health issues
                if health_status["overall_health"] != "healthy":
                    logger.warning(f"System health: {health_status['overall_health']}")
                    for issue in health_status["issues"]:
                        logger.warning(f"  Issue: {issue}")
                
                await asyncio.sleep(300)  # Health check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(300)

    def _handle_telemetry_event(self, event) -> None:
        """Handle telemetry events for integration."""
        # Add telemetry events as performance metrics
        if event.duration_ms:
            self.performance_monitor.add_metric(
                f"telemetry_{event.source}_{event.event_type}_duration",
                event.duration_ms,
                "telemetry",
                {"source": event.source, "type": event.event_type}
            )

    def _handle_performance_alert(self, message: str, details: dict[str, Any]) -> None:
        """Handle performance alerts for dashboard integration."""
        # Forward performance alerts to dashboard
        alert_level = "warning" if "slow" in message.lower() else "error"
        self.dashboard._add_alert(alert_level, message, "performance")

    def _get_default_config(self) -> dict[str, Any]:
        """Get default system configuration."""
        return {
            "metrics_retention_hours": 24,
            "collection_interval_seconds": 10,
            "telemetry_buffer_size": 10000,
            "telemetry_flush_interval": 30,
            "dashboard_update_interval": 5,
            "performance_thresholds": {
                "response_time_ms": 1000,
                "success_rate": 0.95
            },
            "security_config": {
                "check_dependencies": True,
                "check_code": True
            }
        }


# Convenience functions for easy deployment

async def create_monitoring_system(
    config_path: Path | None = None,
    constitutional_threshold: float = 0.75
) -> PerformanceMonitoringSystem:
    """Create and initialize performance monitoring system."""
    config = None
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    
    system = PerformanceMonitoringSystem(
        config=config,
        constitutional_threshold=constitutional_threshold
    )
    
    await system.start_system()
    return system


async def run_commit_validation(
    commit_message: str,
    changed_files: list[str],
    diff_data: str,
    config_path: Path | None = None
) -> dict[str, Any]:
    """Run commit validation with temporary monitoring system."""
    system = await create_monitoring_system(config_path)
    
    try:
        result = await system.validate_commit(commit_message, changed_files, diff_data)
        return result
    finally:
        await system.stop_system()


if __name__ == "__main__":
    # Example usage
    async def main():
        system = await create_monitoring_system()
        
        # Example commit validation
        result = await system.validate_commit(
            "feat: add new feature",
            ["src/new_feature.py", "tests/test_new_feature.py"],
            "diff content here"
        )
        
        print("Validation result:", result["overall_passed"])
        print("System status:", system.get_system_status()["component_status"])
        
        await system.stop_system()
    
    asyncio.run(main())