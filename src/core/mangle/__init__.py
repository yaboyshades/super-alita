"""
Mangle Integration Orchestrator for Super Alita Agent System

Coordinates all Mangle components: gRPC server, Prometheus metrics, 
Redis event bus, and provides unified management interface.
"""

import asyncio
import logging
import signal
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Import core components that always work
from .metrics import PrometheusMetricsCollector
from .redis_event_bus import RedisEventBus

# Try to import gRPC server (optional dependency)
try:
    from .grpc_server import SuperAlitaGrpcServer
    grpc_available = True
except ImportError as e:
    logging.warning(f"gRPC server not available: {e}")
    SuperAlitaGrpcServer = None
    grpc_available = False

from ..cortex.runtime import CortexRuntime
from ..telemetry.collector import TelemetryCollector
from ..knowledge.plugin import KnowledgeGraphPlugin
from ..optimization.plugin import OptimizationPlugin


logger = logging.getLogger(__name__)


class MangleIntegration:
    """
    Complete Mangle integration orchestrator.
    
    Manages the full distributed agent system including:
    - gRPC server for external communication
    - Prometheus metrics for monitoring
    - Redis event bus for distributed messaging
    - Integration with all agent components
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        workspace_root: Optional[Path] = None
    ):
        """
        Initialize Mangle integration.
        
        Args:
            config: Configuration dictionary
            workspace_root: Root directory for the workspace
        """
        self.config = config or {}
        self.workspace_root = workspace_root or Path.cwd()
        
        # Component instances
        self.grpc_server = None
        self.metrics_collector: Optional[PrometheusMetricsCollector] = None
        self.redis_event_bus: Optional[RedisEventBus] = None
        
        # Agent components
        self.cortex_runtime: Optional[CortexRuntime] = None
        self.telemetry_collector: Optional[TelemetryCollector] = None
        self.knowledge_plugin: Optional[KnowledgeGraphPlugin] = None
        self.optimization_plugin: Optional[OptimizationPlugin] = None
        
        # State tracking
        self.is_running = False
        self.start_time: Optional[float] = None
        self.shutdown_handlers = []
        
        # Statistics
        self.total_grpc_requests = 0
        self.total_events_processed = 0
    
    def configure(
        self,
        grpc_host: str = "localhost",
        grpc_port: int = 50051,
        redis_url: str = "redis://localhost:6379",
        enable_metrics: bool = True,
        enable_redis: bool = True
    ) -> None:
        """
        Configure Mangle components.
        
        Args:
            grpc_host: gRPC server host
            grpc_port: gRPC server port
            redis_url: Redis connection URL
            enable_metrics: Whether to enable Prometheus metrics
            enable_redis: Whether to enable Redis event bus
        """
        
        # Initialize metrics collector
        if enable_metrics:
            self.metrics_collector = PrometheusMetricsCollector()
            logger.info("🚀 Prometheus metrics collector initialized")
        
        # Initialize Redis event bus
        if enable_redis:
            self.redis_event_bus = RedisEventBus(redis_url=redis_url)
            logger.info(f"🚀 Redis event bus configured for {redis_url}")
        
        # Initialize gRPC server
        if grpc_available and SuperAlitaGrpcServer:
            self.grpc_server = SuperAlitaGrpcServer(
                host=grpc_host,
                port=grpc_port
            )
            logger.info(f"🚀 gRPC server configured on {grpc_host}:{grpc_port}")
        else:
            logger.warning("🚀 gRPC server not available - skipping gRPC configuration")
    
    def setup_agent_components(
        self,
        cortex_runtime: Optional[CortexRuntime] = None,
        telemetry_collector: Optional[TelemetryCollector] = None,
        knowledge_plugin: Optional[KnowledgeGraphPlugin] = None,
        optimization_plugin: Optional[OptimizationPlugin] = None
    ) -> None:
        """
        Setup agent components for integration.
        
        Args:
            cortex_runtime: Cortex runtime instance
            telemetry_collector: Telemetry collector instance
            knowledge_plugin: Knowledge graph plugin instance
            optimization_plugin: Optimization plugin instance
        """
        self.cortex_runtime = cortex_runtime
        self.telemetry_collector = telemetry_collector
        self.knowledge_plugin = knowledge_plugin
        self.optimization_plugin = optimization_plugin
        
        logger.info("🚀 Agent components configured for Mangle integration")
    
    async def start(self) -> None:
        """Start all Mangle components."""
        if self.is_running:
            logger.warning("Mangle integration already running")
            return
        
        self.start_time = time.time()
        
        try:
            # Start Redis event bus
            if self.redis_event_bus:
                await self.redis_event_bus.connect()
                logger.info("🚀 Redis event bus connected")
            
            # Setup gRPC server with components
            if self.grpc_server:
                self.grpc_server.setup(
                    cortex_runtime=self.cortex_runtime,
                    telemetry_collector=self.telemetry_collector,
                    knowledge_plugin=self.knowledge_plugin,
                    optimization_plugin=self.optimization_plugin,
                    metrics_collector=self.metrics_collector
                )
                await self.grpc_server.start()
                logger.info("🚀 gRPC server started")
            
            # Initialize metrics collection
            if self.metrics_collector:
                self._setup_metrics_collection()
                logger.info("🚀 Metrics collection initialized")
            
            # Setup event bus integration
            if self.redis_event_bus:
                await self._setup_event_bus_integration()
                logger.info("🚀 Event bus integration configured")
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self.is_running = True
            logger.info("🚀 Mangle integration fully started")
            
        except Exception as e:
            logger.error(f"Failed to start Mangle integration: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop all Mangle components gracefully."""
        if not self.is_running:
            logger.warning("Mangle integration not running")
            return
        
        logger.info("🚀 Stopping Mangle integration...")
        
        try:
            # Execute shutdown handlers
            for handler in self.shutdown_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        handler()
                except Exception as e:
                    logger.error(f"Shutdown handler error: {e}")
            
            # Stop gRPC server
            if self.grpc_server:
                await self.grpc_server.stop()
                logger.info("🚀 gRPC server stopped")
            
            # Disconnect Redis event bus
            if self.redis_event_bus:
                await self.redis_event_bus.disconnect()
                logger.info("🚀 Redis event bus disconnected")
            
            self.is_running = False
            logger.info("🚀 Mangle integration stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Mangle integration: {e}")
    
    def _setup_metrics_collection(self) -> None:
        """Setup initial metrics collection."""
        if not self.metrics_collector:
            return
        
        # Set system info
        if self.start_time:
            uptime = time.time() - self.start_time
            self.metrics_collector.set_system_uptime(uptime)
        
        # Set component health
        self.metrics_collector.set_component_health("cortex", self.cortex_runtime is not None)
        self.metrics_collector.set_component_health("telemetry", self.telemetry_collector is not None)
        self.metrics_collector.set_component_health("knowledge", self.knowledge_plugin is not None)
        self.metrics_collector.set_component_health("optimization", self.optimization_plugin is not None)
        self.metrics_collector.set_component_health("grpc", self.grpc_server is not None)
        self.metrics_collector.set_component_health("redis", self.redis_event_bus is not None)
    
    async def _setup_event_bus_integration(self) -> None:
        """Setup event bus integration with agent components."""
        if not self.redis_event_bus:
            return
        
        # Subscribe to key event types for metrics collection
        await self.redis_event_bus.subscribe("cortex_cycle", self._handle_cortex_event)
        await self.redis_event_bus.subscribe("knowledge_operation", self._handle_knowledge_event)
        await self.redis_event_bus.subscribe("optimization_decision", self._handle_optimization_event)
        await self.redis_event_bus.subscribe("system_status", self._handle_system_event)
        
        logger.info("🚀 Event bus subscriptions configured")
    
    async def _handle_cortex_event(self, event) -> None:
        """Handle Cortex events for metrics."""
        if self.metrics_collector:
            self.metrics_collector.inc_events_processed("cortex_cycle", "mangle_integration")
            
            # Extract session info if available
            session_id = getattr(event, 'metadata', {}).get('session_id', 'unknown')
            self.metrics_collector.inc_cortex_cycles(session_id)
    
    async def _handle_knowledge_event(self, event) -> None:
        """Handle knowledge graph events for metrics."""
        if self.metrics_collector:
            self.metrics_collector.inc_events_processed("knowledge_operation", "mangle_integration")
            
            # Extract operation info if available
            operation = getattr(event, 'metadata', {}).get('operation', 'unknown')
            self.metrics_collector.inc_knowledge_operations(operation)
    
    async def _handle_optimization_event(self, event) -> None:
        """Handle optimization events for metrics."""
        if self.metrics_collector:
            self.metrics_collector.inc_events_processed("optimization_decision", "mangle_integration")
            
            # Extract decision info if available
            metadata = getattr(event, 'metadata', {})
            policy_id = metadata.get('policy_id', 'unknown')
            algorithm = metadata.get('algorithm', 'unknown')
            arm_id = metadata.get('arm_id', 'unknown')
            
            self.metrics_collector.inc_optimization_decisions(policy_id, algorithm, arm_id)
    
    async def _handle_system_event(self, event) -> None:
        """Handle system events for metrics."""
        if self.metrics_collector:
            self.metrics_collector.inc_events_processed("system_status", "mangle_integration")
        
        self.total_events_processed += 1
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def add_shutdown_handler(self, handler) -> None:
        """Add a shutdown handler."""
        self.shutdown_handlers.append(handler)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information."""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        status = {
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "start_time": self.start_time,
            "components": {
                "grpc_server": self.grpc_server is not None,
                "metrics_collector": self.metrics_collector is not None,
                "redis_event_bus": self.redis_event_bus is not None,
                "cortex_runtime": self.cortex_runtime is not None,
                "telemetry_collector": self.telemetry_collector is not None,
                "knowledge_plugin": self.knowledge_plugin is not None,
                "optimization_plugin": self.optimization_plugin is not None
            },
            "statistics": {
                "total_grpc_requests": self.total_grpc_requests,
                "total_events_processed": self.total_events_processed
            }
        }
        
        # Add component-specific status
        if self.redis_event_bus:
            status["redis_stats"] = self.redis_event_bus.get_statistics()
        
        if self.metrics_collector:
            try:
                health = self.metrics_collector.health_check()
                status["metrics_health"] = health
            except Exception as e:
                status["metrics_health"] = {"status": "error", "error": str(e)}
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {}
        }
        
        overall_healthy = True
        
        # Check gRPC server
        if self.grpc_server:
            health["components"]["grpc"] = {"status": "healthy"}
        else:
            health["components"]["grpc"] = {"status": "unavailable"}
            overall_healthy = False
        
        # Check Redis event bus
        if self.redis_event_bus:
            redis_health = await self.redis_event_bus.health_check()
            health["components"]["redis"] = redis_health
            if not redis_health.get("healthy", False):
                overall_healthy = False
        else:
            health["components"]["redis"] = {"status": "unavailable"}
        
        # Check metrics collector
        if self.metrics_collector:
            metrics_health = self.metrics_collector.health_check()
            health["components"]["metrics"] = metrics_health
            if metrics_health.get("status") != "healthy":
                overall_healthy = False
        else:
            health["components"]["metrics"] = {"status": "unavailable"}
        
        # Check agent components
        health["components"]["agent"] = {
            "cortex": self.cortex_runtime is not None,
            "telemetry": self.telemetry_collector is not None,
            "knowledge": self.knowledge_plugin is not None,
            "optimization": self.optimization_plugin is not None
        }
        
        if not all(health["components"]["agent"].values()):
            overall_healthy = False
        
        health["status"] = "healthy" if overall_healthy else "degraded"
        return health
    
    async def serve_forever(self) -> None:
        """Start the integration and serve forever."""
        await self.start()
        
        try:
            logger.info("🚀 Mangle integration serving forever...")
            while self.is_running:
                await asyncio.sleep(1)
                
                # Update metrics periodically
                if self.metrics_collector and self.start_time:
                    uptime = time.time() - self.start_time
                    self.metrics_collector.set_system_uptime(uptime)
        
        except KeyboardInterrupt:
            logger.info("🚀 Received keyboard interrupt")
        finally:
            await self.stop()


# Convenience functions for easy deployment

async def create_and_run_mangle_integration(
    config: Optional[Dict[str, Any]] = None,
    workspace_root: Optional[Path] = None,
    **kwargs
) -> MangleIntegration:
    """
    Create, configure, and run a complete Mangle integration.
    
    Args:
        config: Configuration dictionary
        workspace_root: Workspace root directory
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured and running MangleIntegration instance
    """
    integration = MangleIntegration(config=config, workspace_root=workspace_root)
    
    # Apply configuration
    integration.configure(**kwargs)
    
    # Start integration
    await integration.start()
    
    return integration


def main():
    """Main entry point for standalone Mangle integration."""
    import asyncio
    
    async def run():
        integration = MangleIntegration()
        integration.configure()
        await integration.serve_forever()
    
    asyncio.run(run())


if __name__ == "__main__":
    main()